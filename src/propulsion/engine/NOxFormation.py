import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import itertools
from dataclasses import dataclass, field
from typing import List
from ramjet_fixedgeometry import Geometry, Assumptions, RamHelp
from ramjet_fixedgeometry import Atmosphere


# =============================================================================
#  Assumptions
# =============================================================================

class NOX_Assumptions:
    # -- Flight conditions
    M0: float = 5.0          # Inlet Mach number                 [-]
    h0: float = 30_000       # Inlet Altitude                    [m]
    # -- Equivalence ratio
    phi: float = 0.50        # Equivalence ratio                 [-]
    # -- Fuel conditions
    T_fuel: float = 20       # Fuel Temperature                  [K]


# =============================================================================
#  Engine builder -- fully parametric on (Ma0, h0, phi)
# =============================================================================

class BuildEngine(NOX_Assumptions):

    def __init__(self, Ma0: float = None, h0: float = None, phi: float = None):
        """
        Parameters
        ----------
        Ma0  : flight Mach number   (overrides NOX_Assumptions.M0)
        h0   : altitude [m]         (overrides NOX_Assumptions.h0)
        phi  : equivalence ratio    (overrides NOX_Assumptions.phi)
        """
        Ma0 = Ma0 if Ma0 is not None else NOX_Assumptions.M0
        h0  = h0  if h0  is not None else NOX_Assumptions.h0
        phi = phi if phi is not None else NOX_Assumptions.phi

        geom = Geometry(
            A0  = 4.5,
            L01 = 0.60,
            L12 = 0.25,
            L23 = 0.42,
            L34 = 0.28,
            L45 = 0.35,
            L56 = 1.20,
            A2  = 4.05,
            A3  = 4.95,
            A4  = 4.95,
            A6  = 7.2,
        )

        assump = Assumptions(
            h0           = h0,
            Ma0          = Ma0,
            phi          = phi,
            theta        = 90.0,
            mixing_coeff = 0.176,
            Ma_COMB      = 0.3,
            Cf           = 0.003,
            HHV          = 141.8e6,
        )

        self.eng = RamHelp(geom=geom, assump=assump)
        eng = self.eng

        inp  = eng.station_0()
        iso  = eng.station_1(inp)
        sec2 = eng.section_12(iso)
        sec3 = eng.section_23(sec2)
        sec4 = eng.section_34(sec3)
        sec5 = eng.section_45(sec4)
        sec6 = eng.section_56(sec5)

        self.sections = {
            "inp": inp, "iso": iso,
            "sec2": sec2, "sec3": sec3, "sec4": sec4,
            "sec5": sec5, "sec6": sec6,
        }

        self.x, self.V = eng.velocity_distribution(
            iso, sec2, sec3, sec4, sec5, sec6
        )

        g = eng.geom
        self.L01 = g.L01
        self.L12 = g.L12
        self.L23 = g.L23
        self.L34 = g.L34
        self.L45 = g.L45

        self.iso  = iso
        self.sec3 = sec3
        self.phi  = phi


# =============================================================================
#  Residence time
# =============================================================================

class ResidenceTime:

    def __init__(self, Ma0=None, h0=None, phi=None, verbose=True):
        eng = BuildEngine(Ma0=Ma0, h0=h0, phi=phi)

        # x measured from engine inlet; start = L01+L12; end excludes nozzle
        x_start_comb = eng.L01 + eng.L12
        x_end_comb   = eng.L01 + eng.L12 + eng.L23 + eng.L34 + eng.L45

        mask = (eng.x >= x_start_comb) & (eng.x <= x_end_comb)
        self.x_comb = eng.x[mask]
        self.V_comb = eng.V[mask]

        if np.any(self.V_comb <= 0):
            raise ValueError("Non-positive velocity in combustion zone.")

        self.residence_time = float(np.trapz(1.0 / self.V_comb, self.x_comb))

        if verbose:
            print(f"\n-- Residence Time --")
            print(f"  Combustion zone x = [{x_start_comb:.3f}, {x_end_comb:.3f}] m")
            print(f"  tau_residence = {self.residence_time*1e3:.4f} ms")


# =============================================================================
#  Single-point NOx reactor
# =============================================================================

class NOxReactor(BuildEngine):

    def __init__(self, Ma0=None, h0=None, phi=None):
        super().__init__(Ma0=Ma0, h0=h0, phi=phi)
        self.gas      = ct.Solution("Z22_H2_ZNOx20.yaml")
        self.gas_air  = ct.Solution("Z22_H2_ZNOx20.yaml")
        self.gas_fuel = ct.Solution("Z22_H2_ZNOx20.yaml")

    def state_at_inlet(self):
        self.T_air     = self.iso["T"]
        self.P_in      = self.iso["P"]
        self.mdot_air  = self.iso["mdot"]
        self.V_in      = self.iso["V"]
        self.mdot_fuel = self.sec3["mfuel"]

    def _mixed_inlet_state(self, verbose=True):
        """
        Adiabatic mixing of cryogenic H2 + hot air.
        Enthalpy conservation: mdot_air*h_air + mdot_fuel*h_fuel = mdot_mix*h_mix
        Cantera resolves T_mix via gas.HP = (h_mix, P).
        """
        P = self.P_in

        self.gas_air.TPX  = self.T_air, P, 'O2:1.0, N2:3.76'
        h_air  = self.gas_air.enthalpy_mass

        self.gas_fuel.TPX = NOX_Assumptions.T_fuel, P, 'H2:1.0'
        h_fuel = self.gas_fuel.enthalpy_mass

        mdot_mix = self.mdot_air + self.mdot_fuel
        h_mix    = (self.mdot_air * h_air + self.mdot_fuel * h_fuel) / mdot_mix

        self.gas.set_equivalence_ratio(self.phi, 'H2:1.0', 'O2:1.0, N2:3.76')
        self.gas.HP = h_mix, P

        T_mix = self.gas.T

        if verbose:
            print(f"\n-- Fuel-Air Mixing --")
            print(f"  T_air        : {self.T_air:.1f} K")
            print(f"  T_fuel       : {NOX_Assumptions.T_fuel:.1f} K")
            print(f"  mdot_air     : {self.mdot_air:.4f} kg/s")
            print(f"  mdot_fuel    : {self.mdot_fuel:.4f} kg/s")
            print(f"  FAR          : {self.mdot_fuel / self.mdot_air:.5f}")
            print(f"  T_mix        : {T_mix:.1f} K")
            print(f"  DT from air  : {T_mix - self.T_air:.1f} K")

        return T_mix

    def area_at(self, x):
        """Piecewise area: diverging combustor then constant-area. No nozzle."""
        g = self.eng.geom
        if x <= self.L23:
            return g.A2 + (g.A3 - g.A2) * (x / self.L23)
        else:
            return g.A3

    def run_pfr(self, dt=1e-7, verbose=True):
        """PFR through combustion chamber only (L23 + L34). No nozzle."""
        self.state_at_inlet()
        self._mixed_inlet_state(verbose=verbose)

        mdot     = self.mdot_air + self.mdot_fuel
        r        = ct.IdealGasConstPressureReactor(self.gas)
        net      = ct.ReactorNet([r])
        L_comb   = self.L23 + self.L34
        z, t     = 0.0, 0.0
        sp_names = r.thermo.species_names

        def get_X(name):
            idx = sp_names.index(name) if name in sp_names else None
            return float(r.thermo.X[idx]) if idx is not None else 0.0

        results = {k: [] for k in ['z', 't', 'T', 'u',
                                    'NO', 'NO2', 'NOx', 'OH', 'H2', 'H2O']}

        while z < L_comb:
            rho = r.thermo.density
            A   = self.area_at(z)
            u   = mdot / (rho * A)
            dz  = u * dt
            z  += dz
            t  += dt
            net.advance(t)

            results['z'  ].append(z)
            results['t'  ].append(t)
            results['T'  ].append(r.T)
            results['u'  ].append(u)
            results['NO' ].append(get_X('NO'))
            results['NO2'].append(get_X('NO2'))
            results['NOx'].append(get_X('NO') + get_X('NO2'))
            results['OH' ].append(get_X('OH'))
            results['H2' ].append(get_X('H2'))
            results['H2O'].append(get_X('H2O'))

        self.results = {k: np.array(v) for k, v in results.items()}

        if verbose:
            self._print_summary()

        return self.results

    def _print_summary(self):
        r = self.results
        print(f"\n-- NOx PFR Results --")
        print(f"  Final T      : {r['T'][-1]:.1f} K")
        print(f"  Final NO     : {r['NO'][-1]*1e6:.2f} ppm")
        print(f"  Final NO2    : {r['NO2'][-1]*1e6:.2f} ppm")
        print(f"  Final NOx    : {r['NOx'][-1]*1e6:.2f} ppm")
        print(f"  Residence tau: {r['t'][-1]*1e3:.4f} ms")
        print(f"  Final H2     : {r['H2'][-1]*1e6:.2f} ppm")
        print(f"  Final H2O    : {r['H2O'][-1]*100:.3f} %")

    def plot(self, title=None):
        r = self.results
        z = r['z'] * 100   # to cm

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        fig.suptitle(title or 'NOx PFR -- Combustion Chamber', fontsize=13)

        axes[0].plot(z, r['T'], color='firebrick', lw=1.8)
        axes[0].set_ylabel('Temperature [K]')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(z, r['NO' ]*1e6, label='NO',  color='steelblue',  lw=1.8)
        axes[1].plot(z, r['NO2']*1e6, label='NO2', color='darkorange', lw=1.8)
        axes[1].plot(z, r['NOx']*1e6, label='NOx', color='purple',     lw=1.8, ls='--')
        axes[1].set_ylabel('Mole Fraction [ppm]')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(z, r['H2' ]*100, label='H2',       color='royalblue', lw=1.8)
        axes[2].plot(z, r['H2O']*100, label='H2O',      color='seagreen',  lw=1.8)
        axes[2].plot(z, r['OH' ]*1e6, label='OH (ppm)', color='goldenrod', lw=1.4, ls=':')
        axes[2].set_ylabel('Mole Fraction [%] / OH [ppm]')
        axes[2].set_xlabel('Axial position [cm]')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = (title or 'nox_pfr').replace(' ', '_').replace(',','').replace('=','') + '.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {fname}")


# =============================================================================
#  Sweep configuration and result container
# =============================================================================

@dataclass
class SweepConfig:
    """Define the sweep grid. Any single-element list fixes that parameter."""
    phi_range : List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    h0_range  : List[float] = field(default_factory=lambda: [25_000, 30_000, 35_000])
    Ma0_range : List[float] = field(default_factory=lambda: [4.5, 5.0, 5.5])
    dt        : float       = 1e-7
    verbose   : bool        = False   # suppress per-run prints in sweep mode


@dataclass
class SweepPoint:
    """Scalar exit quantities for one (phi, h0, Ma0) combination."""
    phi       : float
    h0        : float
    Ma0       : float
    T_mix     : float        # [K]   inlet mixing temperature
    T_final   : float        # [K]   combustor-exit temperature
    NO_ppm    : float        # [ppm]
    NO2_ppm   : float        # [ppm]
    NOx_ppm   : float        # [ppm]
    tau_ms    : float        # [ms]  PFR residence time
    converged : bool  = True
    error     : str   = ""


# =============================================================================
#  Sweep runner
# =============================================================================

class NOxSweep:
    """
    Full (phi x h0 x Ma0) parametric sweep.

    Usage
    -----
        cfg   = SweepConfig(phi_range=[0.3, 0.5, 0.7],
                            h0_range =[25_000, 30_000],
                            Ma0_range=[4.5, 5.0, 5.5])
        sweep = NOxSweep(cfg)
        sweep.save_csv()
        sweep.plot_nox_surface_phi07()
        sweep.plot_mach_vs_tau()
        sweep.plot_phi_vs_nox_M5_h30()
    """

    def __init__(self, config: SweepConfig = None):
        self.cfg     = config or SweepConfig()
        self.results : List[SweepPoint] = []
        self._run()

    # -------------------------------------------------------------------------
    #  Runner
    # -------------------------------------------------------------------------

    def _run(self):
        combos = list(itertools.product(
            self.cfg.phi_range,
            self.cfg.h0_range,
            self.cfg.Ma0_range,
        ))
        n = len(combos)
        print(f"\n{'='*62}")
        print(f"  NOx Sweep  --  {n} cases")
        print(f"  phi = {self.cfg.phi_range}")
        print(f"  h   = {self.cfg.h0_range} m")
        print(f"  Ma0 = {self.cfg.Ma0_range}")
        print(f"{'='*62}")

        for i, (phi, h0, Ma0) in enumerate(combos, 1):
            tag = f"  [{i:>3}/{n}]  phi={phi:.2f}  h={h0/1e3:.0f} km  M={Ma0:.1f}"
            print(tag, end="  ->  ", flush=True)
            try:
                reactor = NOxReactor(Ma0=Ma0, h0=h0, phi=phi)
                res     = reactor.run_pfr(dt=self.cfg.dt, verbose=self.cfg.verbose)

                pt = SweepPoint(
                    phi      = phi,
                    h0       = h0,
                    Ma0      = Ma0,
                    T_mix    = float(res['T'][0]),
                    T_final  = float(res['T'][-1]),
                    NO_ppm   = float(res['NO'][-1])  * 1e6,
                    NO2_ppm  = float(res['NO2'][-1]) * 1e6,
                    NOx_ppm  = float(res['NOx'][-1]) * 1e6,
                    tau_ms   = float(res['t'][-1])   * 1e3,
                    converged= True,
                )
                print(f"NOx={pt.NOx_ppm:7.1f} ppm   "
                      f"T_exit={pt.T_final:.0f} K   "
                      f"tau={pt.tau_ms:.3f} ms")

            except Exception as exc:
                pt = SweepPoint(
                    phi=phi, h0=h0, Ma0=Ma0,
                    T_mix=np.nan, T_final=np.nan,
                    NO_ppm=np.nan, NO2_ppm=np.nan, NOx_ppm=np.nan,
                    tau_ms=np.nan, converged=False, error=str(exc),
                )
                print(f"FAILED -- {exc}")

            self.results.append(pt)

        n_ok   = sum(p.converged for p in self.results)
        n_fail = n - n_ok
        print(f"\n  Sweep complete: {n_ok} converged, {n_fail} failed.\n")

    # -------------------------------------------------------------------------

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([vars(p) for p in self.results])

    def save_csv(self, path="nox_sweep_results.csv"):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"  Saved -> {path}")
        return df

    def _ok(self):
        return [p for p in self.results if p.converged]

    # =========================================================================
    #  Plot A — 3-D surface: Mach (x) vs Altitude (y) vs NOx ppm (z)
    #           fixed phi = 0.7
    # =========================================================================

    def plot_nox_surface_phi07(self, phi_target=0.7,
                               save="nox_surface_phi07.png"):
        """
        3-D surface of exit NOx [ppm] as a function of Mach number (x-axis)
        and altitude (y-axis), for a fixed equivalence ratio phi_target.
        """
        pts = [p for p in self._ok() if abs(p.phi - phi_target) < 1e-9]

        if not pts:
            print(f"  No converged results for phi={phi_target}. Skipping.")
            return

        Ma0_vals = sorted(set(p.Ma0 for p in pts))
        h0_vals  = sorted(set(p.h0  for p in pts))

        Ma_grid = np.array(Ma0_vals)
        h_grid  = np.array(h0_vals) / 1e3     # convert to km for axis label

        # Build NOx matrix  shape = (n_h, n_M)
        Z = np.full((len(h0_vals), len(Ma0_vals)), np.nan)
        for p in pts:
            i = h0_vals.index(p.h0)
            j = Ma0_vals.index(p.Ma0)
            Z[i, j] = p.NOx_ppm

        Ma_mesh, h_mesh = np.meshgrid(Ma_grid, h_grid)

        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(Ma_mesh, h_mesh, Z,
                               cmap='plasma', edgecolor='none', alpha=0.9)
        ax.scatter(
            [p.Ma0 for p in pts],
            [p.h0/1e3 for p in pts],
            [p.NOx_ppm for p in pts],
            color='white', s=20, zorder=5, depthshade=False
        )

        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label='NOx [ppm]')
        ax.set_xlabel('Mach number [-]', labelpad=10)
        ax.set_ylabel('Altitude [km]',   labelpad=10)
        ax.set_zlabel('NOx [ppm]',       labelpad=10)
        ax.set_title(f'Exit NOx — Mach vs Altitude  (phi = {phi_target})', pad=15)

        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")

    # =========================================================================
    #  Plot B — Mach number vs residence time
    #           one line per altitude, coloured by altitude
    # =========================================================================

    def plot_mach_vs_tau(self, phi_target=None,
                         save="nox_mach_vs_tau.png"):
        """
        Residence time tau [ms] vs Mach number.
        One line per altitude level; coloured by altitude.
        If phi_target is given, only that phi slice is used;
        otherwise all converged points are included (averaged over phi).
        """
        pts = self._ok()
        if phi_target is not None:
            pts = [p for p in pts if abs(p.phi - phi_target) < 1e-9]
            subtitle = f"phi = {phi_target}"
        else:
            subtitle = "all phi (mean)"

        h0_vals  = sorted(set(p.h0  for p in pts))
        Ma0_vals = sorted(set(p.Ma0 for p in pts))
        n_h      = len(h0_vals)
        cmap     = plt.cm.get_cmap('viridis', n_h)

        fig, ax = plt.subplots(figsize=(9, 5))

        for i, h0 in enumerate(h0_vals):
            sub = [p for p in pts if p.h0 == h0]
            # average over phi if multiple phi values present
            tau_by_mach = {}
            for p in sub:
                tau_by_mach.setdefault(p.Ma0, []).append(p.tau_ms)
            machs = sorted(tau_by_mach)
            taus  = [np.mean(tau_by_mach[m]) for m in machs]

            ax.plot(machs, taus, marker='o', color=cmap(i),
                    lw=1.8, label=f'h = {h0/1e3:.0f} km')

        ax.set_xlabel('Mach number [-]')
        ax.set_ylabel('Residence time tau [ms]')
        ax.set_title(f'Combustor Residence Time vs Mach  ({subtitle})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")

    # =========================================================================
    #  Plot C — phi vs NOx ppm
    #           fixed M = 5.0, h = 30 000 m
    # =========================================================================

    def plot_phi_vs_nox(self, Ma0_target=5.0, h0_target=30_000,
                        save="nox_phi_vs_nox_M5_h30.png"):
        """
        Exit NOx [ppm] vs equivalence ratio phi.
        Fixed at Ma0_target and h0_target.
        Also overlays NO and NO2 as separate lines.
        """
        pts = [p for p in self._ok()
               if abs(p.Ma0 - Ma0_target) < 1e-9
               and abs(p.h0  - h0_target)  < 1.0]
        pts = sorted(pts, key=lambda p: p.phi)

        if not pts:
            print(f"  No converged results for M={Ma0_target}, h={h0_target}. Skipping.")
            return

        phis    = [p.phi     for p in pts]
        NOx_ppm = [p.NOx_ppm for p in pts]
        NO_ppm  = [p.NO_ppm  for p in pts]
        NO2_ppm = [p.NO2_ppm for p in pts]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(phis, NOx_ppm, marker='o', color='purple',     lw=2.0,
                label='NOx (NO + NO2)', zorder=3)
        ax.plot(phis, NO_ppm,  marker='s', color='steelblue',  lw=1.6,
                linestyle='--', label='NO')
        ax.plot(phis, NO2_ppm, marker='^', color='darkorange', lw=1.6,
                linestyle='--', label='NO2')

        ax.set_xlabel('Equivalence ratio phi [-]')
        ax.set_ylabel('Exit NOx [ppm]')
        ax.set_title(f'NOx vs phi   (M = {Ma0_target},  h = {h0_target/1e3:.0f} km)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")

    # =========================================================================
    #  Legacy sweep plots (kept for completeness)
    # =========================================================================

    def plot_nox_summary(self, save="nox_sweep_summary.png"):
        cfg  = self.cfg
        pts  = self._ok()
        n_h  = len(cfg.h0_range)
        n_M  = len(cfg.Ma0_range)
        cmap = plt.cm.get_cmap('plasma', n_M)

        fig, axes = plt.subplots(1, n_h, figsize=(5*n_h, 5), sharey=True)
        if n_h == 1:
            axes = [axes]

        for ax, h0 in zip(axes, cfg.h0_range):
            for j, Ma0 in enumerate(cfg.Ma0_range):
                sub = sorted([p for p in pts if p.h0==h0 and p.Ma0==Ma0],
                             key=lambda p: p.phi)
                if not sub:
                    continue
                ax.plot([p.phi for p in sub], [p.NOx_ppm for p in sub],
                        marker='o', color=cmap(j), lw=1.8, label=f'M={Ma0}')
            ax.set_title(f'h = {h0/1e3:.0f} km')
            ax.set_xlabel('Equivalence ratio phi [-]')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        axes[0].set_ylabel('Exit NOx [ppm]')
        fig.suptitle('NOx Sweep -- Exit NOx vs phi', fontsize=13)
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")

    def plot_heatmaps(self, save="nox_sweep_heatmaps.png"):
        cfg = self.cfg
        pts = self._ok()
        n_h = len(cfg.h0_range)

        fig, axes = plt.subplots(1, n_h, figsize=(5*n_h, 4))
        if n_h == 1:
            axes = [axes]

        for ax, h0 in zip(axes, cfg.h0_range):
            grid = np.full((len(cfg.Ma0_range), len(cfg.phi_range)), np.nan)
            for p in pts:
                if p.h0 != h0:
                    continue
                i = cfg.Ma0_range.index(p.Ma0)
                j = cfg.phi_range.index(p.phi)
                grid[i, j] = p.NOx_ppm

            im = ax.imshow(grid, aspect='auto', origin='lower',
                           cmap='YlOrRd', interpolation='nearest')
            plt.colorbar(im, ax=ax, label='NOx [ppm]')
            ax.set_xticks(range(len(cfg.phi_range)))
            ax.set_xticklabels([f'{v:.2f}' for v in cfg.phi_range])
            ax.set_yticks(range(len(cfg.Ma0_range)))
            ax.set_yticklabels([f'{v:.1f}' for v in cfg.Ma0_range])
            ax.set_xlabel('phi [-]')
            ax.set_ylabel('Mach [-]')
            ax.set_title(f'h = {h0/1e3:.0f} km')
            for i in range(len(cfg.Ma0_range)):
                for j in range(len(cfg.phi_range)):
                    val = grid[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.0f}',
                                ha='center', va='center', fontsize=8)

        fig.suptitle('NOx Heatmap -- exit NOx [ppm]  (rows=Mach, cols=phi)', fontsize=13)
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")

    def plot_temperature_summary(self, save="nox_sweep_temperature.png"):
        cfg  = self.cfg
        pts  = self._ok()
        n_h  = len(cfg.h0_range)
        n_M  = len(cfg.Ma0_range)
        cmap = plt.cm.get_cmap('viridis', n_M)

        fig, axes = plt.subplots(2, n_h, figsize=(5*n_h, 8), sharey='row')
        if n_h == 1:
            axes = axes.reshape(2, 1)

        for col, h0 in enumerate(cfg.h0_range):
            for j, Ma0 in enumerate(cfg.Ma0_range):
                sub = sorted([p for p in pts if p.h0==h0 and p.Ma0==Ma0],
                             key=lambda p: p.phi)
                if not sub:
                    continue
                phis = [p.phi for p in sub]
                axes[0, col].plot(phis, [p.T_mix   for p in sub],
                                  marker='o', color=cmap(j), lw=1.8, label=f'M={Ma0}')
                axes[1, col].plot(phis, [p.T_final for p in sub],
                                  marker='s', color=cmap(j), lw=1.8, label=f'M={Ma0}')
            for row, ylabel in enumerate(['T_mix [K]', 'T_exit [K]']):
                axes[row, col].set_title(f'h = {h0/1e3:.0f} km')
                axes[row, col].set_xlabel('phi [-]')
                axes[row, col].set_ylabel(ylabel)
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].legend(fontsize=8)

        fig.suptitle('NOx Sweep -- Temperatures', fontsize=13)
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")

    def plot_residence_time(self, save="nox_sweep_tau.png"):
        cfg  = self.cfg
        pts  = self._ok()
        n_h  = len(cfg.h0_range)
        n_M  = len(cfg.Ma0_range)
        cmap = plt.cm.get_cmap('cool', n_M)

        fig, axes = plt.subplots(1, n_h, figsize=(5*n_h, 5), sharey=True)
        if n_h == 1:
            axes = [axes]

        for ax, h0 in zip(axes, cfg.h0_range):
            for j, Ma0 in enumerate(cfg.Ma0_range):
                sub = sorted([p for p in pts if p.h0==h0 and p.Ma0==Ma0],
                             key=lambda p: p.phi)
                if not sub:
                    continue
                ax.plot([p.phi for p in sub], [p.tau_ms for p in sub],
                        marker='o', color=cmap(j), lw=1.8, label=f'M={Ma0}')
            ax.set_title(f'h = {h0/1e3:.0f} km')
            ax.set_xlabel('phi [-]')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        axes[0].set_ylabel('Residence time tau [ms]')
        fig.suptitle('NOx Sweep -- Combustor Residence Time', fontsize=13)
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":

    # -- Baseline single-point run
    print("=" * 60)
    print("  Baseline single-point  phi=0.5  h=30 km  M=5.0")
    print("=" * 60)
    reactor = NOxReactor()
    reactor.run_pfr(dt=1e-7)
    reactor.plot(title="Baseline phi=0.5 h=30km M=5.0")

    # -- Parametric sweep
    cfg = SweepConfig(
        phi_range = [0.3, 0.4, 0.5, 0.6, 0.7],
        h0_range  = [20_000, 25_000, 30_000, 32_000],
        Ma0_range = [2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        dt        = 1e-7,
        verbose   = False,
    )

    sweep = NOxSweep(cfg)
    sweep.save_csv("nox_sweep_results.csv")

    # -- Three targeted plots
    sweep.plot_nox_surface_phi07(phi_target=0.7)   # 3-D: Mach x Alt x NOx at phi=0.7
    sweep.plot_mach_vs_tau()                        # Mach vs tau, coloured by altitude
    sweep.plot_phi_vs_nox(Ma0_target=5.0,           # phi vs NOx at M=5, h=30 km
                          h0_target=30_000)