import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import itertools
from dataclasses import dataclass, field
from typing import List
from ramjet_fixedgeometry import Geometry, Assumptions, RamHelp


# =============================================================================
#  Default assumptions
# =============================================================================

class NOX_Assumptions:
    M0     : float = 5.0       # flight Mach number        [-]
    h0     : float = 30_000    # altitude                  [m]
    phi    : float = 0.50      # equivalence ratio         [-]
    T_fuel : float = 20        # fuel temperature (cryo H2)[K]


# =============================================================================
#  Engine builder  —  fully parametric on (Ma0, h0, phi)
#
#  Two-pass build:
#    Pass 1  run engine with placeholder Ma_COMB=0.3 to get isolator exit state
#    Pass 2  derive Ma_COMB from mass continuity at combustor inlet (A2),
#            then rebuild so the combustor velocity field is physically correct
# =============================================================================

class BuildEngine(NOX_Assumptions):

    def __init__(self, Ma0: float = None,
                       h0:  float = None,
                       phi: float = None):

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

        # ── Pass 1: placeholder run to obtain isolator exit ───────────────────
        a0 = Assumptions(h0=h0, Ma0=Ma0, phi=phi, theta=90.0,
                         mixing_coeff=0.176, Ma_COMB=0.3,
                         Cf=0.003, HHV=141.8e6)
        e0       = RamHelp(geom=geom, assump=a0)
        iso_init = e0.station_1(e0.station_0())

        # ── Pass 2: derive Ma_COMB from isolator exit (mass continuity) ───────
        gamma   = 1.4
        R_air   = 287.05
        T_iso   = float(iso_init["T"])
        P_iso   = float(iso_init["P"])
        mdot    = float(iso_init["mdot"])
        rho_iso = P_iso / (R_air * T_iso)
        a_iso   = float(np.sqrt(gamma * R_air * T_iso))
        u_comb  = mdot / (rho_iso * geom.A2)
        Ma_COMB = float(np.clip(u_comb / a_iso, 0.05, 0.95))

        assump = Assumptions(h0=h0, Ma0=Ma0, phi=phi, theta=90.0,
                             mixing_coeff=0.176, Ma_COMB=Ma_COMB,
                             Cf=0.003, HHV=141.8e6)

        self.eng     = RamHelp(geom=geom, assump=assump)
        self.Ma_COMB = Ma_COMB
        eng          = self.eng

        inp  = eng.station_0()
        iso  = eng.station_1(inp)
        sec2 = eng.section_12(iso)
        sec3 = eng.section_23(sec2)
        sec4 = eng.section_34(sec3)
        sec5 = eng.section_45(sec4)
        sec6 = eng.section_56(sec5)

        self.sections = dict(inp=inp, iso=iso,
                             sec2=sec2, sec3=sec3, sec4=sec4,
                             sec5=sec5, sec6=sec6)

        self.x, self.V = eng.velocity_distribution(iso, sec2, sec3, sec4, sec5, sec6)

        g          = eng.geom
        self.L01   = g.L01
        self.L12   = g.L12
        self.L23   = g.L23
        self.L34   = g.L34
        self.L45   = g.L45
        self.iso   = iso
        self.sec3  = sec3
        self.phi   = phi
        self.L_comb = g.L23 + g.L34

        # ── Li et al. (2023) mixing efficiency η(x) ──────────────────────────
        _c, _th, _L = assump.mixing_coeff, assump.theta, self.L_comb

        def mixing_eta(x: float) -> float:
            s   = float(np.clip(x / _L, 1e-6, 1.0))
            e0  = s
            e90 = float(np.clip(1.01 + _c * np.log(s), 0.0, 1.0))
            if _th == 0.0:  return e0
            if _th == 90.0: return e90
            return _th / 90.0 * (e90 - e0) + e0

        self.mixing_eta = mixing_eta


# =============================================================================
#  Residence time  —  geometric integral ∫(1/V) dx over combustion zone
#  (independent of Cantera; uses engine velocity distribution directly)
# =============================================================================

class ResidenceTime:

    def __init__(self, Ma0=None, h0=None, phi=None, verbose=True):
        eng          = BuildEngine(Ma0=Ma0, h0=h0, phi=phi)
        x0           = eng.L01 + eng.L12
        x1           = x0 + eng.L23 + eng.L34
        mask         = (eng.x >= x0) & (eng.x <= x1)
        x_c, V_c     = eng.x[mask], eng.V[mask]

        if np.any(V_c <= 0):
            raise ValueError("Non-positive velocity in combustion zone.")

        self.residence_time = float(np.trapezoid(1.0 / V_c, x_c))

        if verbose:
            print(f"\n-- Residence Time --")
            print(f"  Combustion zone : [{x0:.3f}, {x1:.3f}] m")
            print(f"  tau_residence   : {self.residence_time*1e3:.4f} ms")


# =============================================================================
#  PSR Reactor
#
#  Methodology (following reviewer guidance):
#   1. Inlet state  — adiabatic mixing of cryogenic H2 + hot ram air
#   2. tau_res      — V_comb / Q_vol  (combustor volume / volumetric flow)
#   3. T_ad         — equilibrate(HP) on inlet mixture
#   4. PSR          — IdealGasConstPressureReactor, multicomponent transport,
#                     log-spaced time grid from 1e-4 * tau_res to 10 * tau_res
#   5. t1           — first time |qdot| >= qdot_threshold * max|qdot|
#                     (auto-ignition onset; skips numerically stiff t=0 region)
#   6. NOx result   — interpolated at t = t1 + tau_res
#   7. Validity     — T(t1+tau_res) >= 0.99 * T_ad
#   8. EI           — g NOx (NO2-equiv) / kg_fuel  via ICAO MW convention
# =============================================================================

class PSRReactor(BuildEngine):

    MECH = "Z22_H2_ZNOx20.yaml"

    def __init__(self, Ma0=None, h0=None, phi=None,
                 qdot_threshold: float = 0.05,
                 t_end_factor:   float = 10.0,
                 n_points:       int   = 2000):

        super().__init__(Ma0=Ma0, h0=h0, phi=phi)
        self.qdot_threshold = qdot_threshold
        self.t_end_factor   = t_end_factor
        self.n_points       = n_points

        self.gas      = ct.Solution(self.MECH)
        self.gas_air  = ct.Solution(self.MECH)
        self.gas_fuel = ct.Solution(self.MECH)

        # Multicomponent transport — important for H2 (Le ~ 0.3)
        for transport in ('multicomponent', 'mixture-averaged'):
            try:
                self.gas.transport_model = transport
                break
            except Exception:
                continue

    # ── Geometry ─────────────────────────────────────────────────────────────

    def _area_at(self, x: float) -> float:
        g = self.eng.geom
        if x <= self.L23:
            return g.A2 + (g.A3 - g.A2) * (x / self.L23)
        return g.A3

    def _combustor_volume(self) -> float:
        x = np.linspace(0.0, self.L_comb, 500)
        A = np.array([self._area_at(xi) for xi in x])
        return float(np.trapezoid(A, x))

    # ── Inlet state ───────────────────────────────────────────────────────────

    def _set_inlet_state(self, verbose: bool = True):
        T_air     = self.iso["T"]
        P         = self.iso["P"]
        mdot_air  = self.iso["mdot"]
        mdot_fuel = self.sec3["mfuel"]

        self.T_air     = T_air
        self.P_in      = P
        self.mdot_air  = mdot_air
        self.mdot_fuel = mdot_fuel

        self.gas_air.TPX  = T_air, P, 'O2:1.0, N2:3.76'
        self.gas_fuel.TPX = NOX_Assumptions.T_fuel, P, 'H2:1.0'
        h_mix = ((mdot_air  * self.gas_air.enthalpy_mass +
                  mdot_fuel * self.gas_fuel.enthalpy_mass) /
                 (mdot_air + mdot_fuel))

        self.gas.set_equivalence_ratio(self.phi, 'H2:1.0', 'O2:1.0, N2:3.76')
        self.gas.HP = h_mix, P
        self.T_mix  = self.gas.T

        if verbose:
            phi_g = self.gas.equivalence_ratio()
            print(f"\n-- PSR Inlet --")
            print(f"  T_air          : {T_air:.1f} K")
            print(f"  T_fuel         : {NOX_Assumptions.T_fuel:.1f} K")
            print(f"  T_mix          : {self.T_mix:.1f} K")
            print(f"  P_in           : {P/1e5:.3f} bar")
            print(f"  mdot_air       : {mdot_air:.4f} kg/s")
            print(f"  mdot_fuel      : {mdot_fuel*1e3:.4f} g/s")
            print(f"  phi (set)      : {self.phi:.4f}")
            print(f"  phi (global)   : {phi_g:.4f}")
            print(f"  FAR            : {mdot_fuel/mdot_air:.5f}")
            print(f"  Ma_COMB        : {self.Ma_COMB:.4f}")
            print(f"  transport      : {self.gas.transport_model}")

    # ── Adiabatic flame temperature ───────────────────────────────────────────

    def _T_adiabatic(self) -> float:
        g = ct.Solution(self.MECH)
        g.TPX = self.gas.T, self.gas.P, self.gas.X
        g.equilibrate('HP')
        return g.T

    # ── PSR run ───────────────────────────────────────────────────────────────

    def run_psr(self, verbose: bool = True) -> dict:
        self._set_inlet_state(verbose=verbose)

        T_ad   = self._T_adiabatic()
        V_comb = self._combustor_volume()

        # ── Residence time ────────────────────────────────────────────────────
        # Two independent estimates — both printed so you can cross-check.
        #
        # (A) Velocity integral  tau_V = ∫(1/u) dz  from engine 1D solution.
        #     This is the same calculation as ResidenceTime and is geometry-
        #     independent of area units.
        x0   = self.L01 + self.L12
        x1   = x0 + self.L23 + self.L34
        mask = (self.x >= x0) & (self.x <= x1)
        tau_V = float(np.trapezoid(1.0 / self.V[mask], self.x[mask]))

        # (B) Volume / volumetric-flow  tau_Q = V_comb / Q_vol
        #     Q_vol uses inlet density from Cantera (post-mixing state).
        #     Sensitive to A2/A3 area units — if V_comb looks wrong, check
        #     whether geometry areas are in m² or cm².
        Q_vol  = (self.mdot_air + self.mdot_fuel) / self.gas.density
        tau_Q  = V_comb / Q_vol

        # Use the velocity-integral estimate as the authoritative tau_res.
        # It comes directly from the engine's momentum solution and is
        # independent of area unit ambiguity.
        tau_res = tau_V

        if verbose:
            print(f"\n-- PSR Setup --")
            print(f"  V_comb         : {V_comb*1e3:.4f} L  "
                  f"({'check area units — >100 L is suspicious' if V_comb > 0.1 else 'OK'})")
            print(f"  Q_vol          : {Q_vol:.4f} m3/s")
            print(f"  tau_res (∫1/u) : {tau_V*1e3:.4f} ms   <-- used")
            print(f"  tau_res (V/Q)  : {tau_Q*1e3:.4f} ms   (cross-check)")
            print(f"  T_adiabatic    : {T_ad:.1f} K")

        # ── Reactor ───────────────────────────────────────────────────────────
        r   = ct.IdealGasConstPressureReactor(self.gas, clone=False)
        net = ct.ReactorNet([r])
        net.atol = 1e-18
        net.rtol = 1e-10

        sp   = r.phase.species_names
        gx   = lambda n: float(r.phase.X[sp.index(n)]) if n in sp else 0.0
        qdot = lambda: float(-np.dot(r.phase.partial_molar_enthalpies,
                                     r.phase.net_production_rates))

        t_span = np.logspace(
            np.log10(max(tau_res * 1e-4, 1e-10)),
            np.log10(tau_res * self.t_end_factor),
            self.n_points
        )

        store = {k: [] for k in ['t', 'T', 'qdot',
                                  'NO', 'NO2', 'NOx', 'OH', 'H2', 'H2O']}
        for t_tgt in t_span:
            net.advance(t_tgt)
            store['t'   ].append(t_tgt)
            store['T'   ].append(r.phase.T)
            store['qdot'].append(qdot())
            store['NO'  ].append(gx('NO'))
            store['NO2' ].append(gx('NO2'))
            store['NOx' ].append(gx('NO') + gx('NO2'))
            store['OH'  ].append(gx('OH'))
            store['H2'  ].append(gx('H2'))
            store['H2O' ].append(gx('H2O'))

        store = {k: np.array(v) for k, v in store.items()}

        # ── t1: auto-ignition onset ───────────────────────────────────────────
        q_abs     = np.abs(store['qdot'])
        threshold = self.qdot_threshold * float(q_abs.max())
        idx_t1    = int(np.argmax(q_abs >= threshold))
        if q_abs[idx_t1] < threshold:
            idx_t1 = 0
            if verbose:
                print(f"  WARNING: ignition threshold never reached.")
        t1 = float(store['t'][idx_t1])

        # ── Damköhler diagnostic ─────────────────────────────────────────────
        # The relevant question is not whether mixing is complete at the
        # combustor EXIT, but whether it is complete at the point of IGNITION.
        # The Li et al. model always returns eta=1 at s=1 by construction, so
        # evaluating at the exit always gives Da=0 — a mathematical artefact.
        #
        # Correct formulation:
        #   x_ign  = u_inlet * t1  — how far the parcel travels before igniting
        #   eta_ign = mixing_eta(x_ign) — mixing efficiency at that point
        #   Da = (1 - eta_ign)     — unmixed fraction at ignition onset
        #
        # Interpretation:
        #   Da << 1 : mostly mixed before ignition — premixed PSR appropriate
        #   Da ~  1 : half-mixed at ignition — premixed is marginal
        #   Da >> 1 : poorly mixed at ignition — premixed assumption is POOR
        #             (Da > 1 is unphysical here since Da is bounded [0,1];
        #              values close to 1 indicate very poor premixedness)

        tau_chem = t1 if t1 > 0 else float(store['t'][1])  # ignition delay [s]

        # Axial position reached by inlet parcel at ignition
        # Combustor-inlet velocity from engine velocity distribution
        x_comb_start = self.L01 + self.L12
        mask_inlet   = (self.x >= x_comb_start)
        u_inlet      = float(self.V[mask_inlet][0]) if mask_inlet.any() \
                       else float(self.iso["V"])
        x_ign    = float(np.clip(u_inlet * tau_chem, 0.0, self.L_comb))
        eta_ign  = float(self.mixing_eta(x_ign))
        eta_exit = float(self.mixing_eta(self.L_comb))
        Da       = 1.0 - eta_ign     # unmixed fraction at ignition: 0=good, 1=bad

        if Da < 0.1:
            Da_flag = "well-mixed at ignition — premixed PSR appropriate"
        elif Da < 0.3:
            Da_flag = "partially mixed at ignition — premixed acceptable with caution"
        else:
            Da_flag = "poorly mixed at ignition — premixed assumption is POOR"

        # Mixing timescale for reference
        tau_mix = tau_res * (1.0 - eta_exit)   # residual mixing time at exit

        if verbose:
            print(f"\n-- Damköhler Diagnostic --")
            print(f"  tau_chem (t1)  : {tau_chem*1e6:.3f} us  (ignition delay)")
            print(f"  u_inlet        : {u_inlet:.2f} m/s")
            print(f"  x_ignition     : {x_ign*100:.2f} cm  (into combustor)")
            print(f"  eta at x_ign   : {eta_ign:.4f}")
            print(f"  eta at exit    : {eta_exit:.4f}")
            print(f"  Da = 1-eta_ign : {Da:.4f}")
            print(f"  Assessment     : {Da_flag}")

        # ── Read quantities at t1 + tau_res ───────────────────────────────────
        t_read     = t1 + tau_res
        t_arr      = store['t']
        NOx_at_tau = float(np.interp(t_read, t_arr, store['NOx'])) * 1e6
        NO_at_tau  = float(np.interp(t_read, t_arr, store['NO']))  * 1e6
        NO2_at_tau = float(np.interp(t_read, t_arr, store['NO2'])) * 1e6
        T_at_tau   = float(np.interp(t_read, t_arr, store['T']))
        H2O_at_tau = float(np.interp(t_read, t_arr, store['H2O']))
        T_check_ok = T_at_tau >= 0.99 * T_ad

        # ── Emission indices ──────────────────────────────────────────────────
        # MW_mix is obtained by advancing a fresh reactor to t_read,
        # avoiding the "cannot integrate backwards" error that occurs when
        # t_read < t_end (net is already past t_read after the main loop).
        r2   = ct.IdealGasConstPressureReactor(self.gas, clone=False)
        net2 = ct.ReactorNet([r2])
        net2.atol = 1e-18
        net2.rtol = 1e-10
        net2.advance(t_read)
        MW_mix   = r2.phase.mean_molecular_weight
        FAR_inv  = (self.mdot_air + self.mdot_fuel) / self.mdot_fuel
        EI_NOx   = (NOx_at_tau * 1e-6) * (46.005 / MW_mix) * FAR_inv * 1e3
        EI_H2O   = H2O_at_tau           * (18.015 / MW_mix) * FAR_inv * 1e3

        if verbose:
            print(f"\n-- PSR Results --")
            print(f"  t1 (ignition)  : {t1*1e6:.3f} us")
            print(f"  Da             : {Da:.4f}  ({Da_flag})")
            print(f"  t_read (t1+tau): {t_read*1e3:.4f} ms")
            print(f"  T at t_read    : {T_at_tau:.1f} K")
            print(f"  T_adiabatic    : {T_ad:.1f} K")
            print(f"  T check        : {'PASS' if T_check_ok else 'FAIL — tau too short'}")
            print(f"  NOx            : {NOx_at_tau:.2f} ppm")
            print(f"  NO             : {NO_at_tau:.2f} ppm")
            print(f"  NO2            : {NO2_at_tau:.2f} ppm")
            print(f"  MW_mix         : {MW_mix:.2f} g/mol")
            print(f"  EI_NOx         : {EI_NOx:.4f} g/kg_fuel  (NO2-equiv, ICAO)")
            print(f"  EI_H2O         : {EI_H2O:.2f} g/kg_fuel")

        self.psr_results = {
            **store,
            'tau_res'    : tau_res,
            't1'         : t1,
            't_read'     : t_read,
            'T_ad'       : T_ad,
            'T_at_tau'   : T_at_tau,
            'T_check_ok' : T_check_ok,
            'NOx_at_tau' : NOx_at_tau,
            'NO_at_tau'  : NO_at_tau,
            'NO2_at_tau' : NO2_at_tau,
            'EI_NOx'     : EI_NOx,
            'EI_H2O'     : EI_H2O,
            'MW_mix'     : MW_mix,
            'Da'         : Da,
            'tau_mix'    : tau_mix,
            'tau_chem'   : tau_chem,
            'eta_exit'   : eta_exit,
            'Da_flag'    : Da_flag,
        }
        return self.psr_results

    # ── Plot ──────────────────────────────────────────────────────────────────

    def plot(self, title: str = None, save: str = None):
        r      = self.psr_results
        t_us   = r['t'] * 1e6
        t1_us  = r['t1']     * 1e6
        tr_us  = r['t_read'] * 1e6

        fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
        fig.suptitle(title or
                     f"PSR  phi={self.phi:.2f}  Ma_COMB={self.Ma_COMB:.2f}",
                     fontsize=13)

        def vl(ax):
            ax.axvline(t1_us, color='grey',  lw=1.2, ls=':', label=f't1={t1_us:.1f} us')
            ax.axvline(tr_us, color='black', lw=1.4, ls='--',
                       label=f't_read={tr_us:.1f} us')

        # Temperature
        axes[0].semilogx(t_us, r['T'], color='firebrick', lw=1.8)
        axes[0].axhline(r['T_ad'],        color='firebrick', lw=1.0, ls=':',
                        label=f"T_ad={r['T_ad']:.0f} K")
        axes[0].axhline(0.99 * r['T_ad'], color='orange',   lw=1.0, ls='--',
                        label='99% T_ad')
        vl(axes[0]);  axes[0].set_ylabel('Temperature [K]')
        axes[0].legend(fontsize=7);  axes[0].grid(True, alpha=0.3, which='both')

        # Heat release rate
        axes[1].semilogx(t_us, r['qdot'] / 1e6, color='darkorange', lw=1.8)
        axes[1].axhline(self.qdot_threshold * r['qdot'].max() / 1e6,
                        color='grey', lw=1.0, ls=':',
                        label=f'{self.qdot_threshold*100:.0f}% qdot_max (t1 threshold)')
        vl(axes[1]);  axes[1].set_ylabel('Heat release [MW/m³]')
        axes[1].legend(fontsize=7);  axes[1].grid(True, alpha=0.3, which='both')
        # Annotate Da in upper-right corner
        da_color = ('green' if r['Da'] < 0.1
                    else 'orange' if r['Da'] < 1.0
                    else 'red')
        axes[1].text(0.98, 0.95,
                     f"Da = {r['Da']:.3f}\n{r['Da_flag']}",
                     transform=axes[1].transAxes, fontsize=7,
                     ha='right', va='top',
                     color=da_color,
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.8))

        # NOx
        axes[2].semilogx(t_us, r['NO' ]*1e6, label='NO',  color='steelblue',  lw=1.8)
        axes[2].semilogx(t_us, r['NO2']*1e6, label='NO2', color='darkorange', lw=1.8)
        axes[2].semilogx(t_us, r['NOx']*1e6, label='NOx', color='purple',     lw=1.8, ls='--')
        axes[2].axhline(r['NOx_at_tau'], color='purple', lw=1.0, ls=':',
                        label=f"NOx @ tau = {r['NOx_at_tau']:.2f} ppm")
        vl(axes[2]);  axes[2].set_ylabel('Mole fraction [ppm]')
        axes[2].legend(fontsize=7);  axes[2].grid(True, alpha=0.3, which='both')

        # H2 / H2O / OH
        axes[3].semilogx(t_us, r['H2' ]*100, label='H2',      color='royalblue', lw=1.8)
        axes[3].semilogx(t_us, r['H2O']*100, label='H2O',     color='seagreen',  lw=1.8)
        axes[3].semilogx(t_us, r['OH' ]*1e6, label='OH (ppm)', color='goldenrod', lw=1.4, ls=':')
        vl(axes[3]);  axes[3].set_ylabel('Mole fraction [%] / OH [ppm]')
        axes[3].set_xlabel('Time [us] (log scale)')
        axes[3].legend(fontsize=7);  axes[3].grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        fname = save or (title or 'psr').replace(' ', '_').replace(',', '') + '.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {fname}")


# =============================================================================
#  Sweep
# =============================================================================

@dataclass
class SweepConfig:
    phi_range : List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    h0_range  : List[float] = field(default_factory=lambda: [25_000, 30_000, 35_000])
    Ma0_range : List[float] = field(default_factory=lambda: [4.5, 5.0, 5.5])
    verbose   : bool        = False


@dataclass
class SweepPoint:
    phi          : float
    h0           : float
    Ma0          : float
    T_mix        : float        # [K]
    T_at_tau     : float        # [K]   temperature at t1+tau_res
    T_ad         : float        # [K]   adiabatic flame temperature
    T_check_ok   : bool         # T_at_tau >= 0.99 * T_ad
    NOx_ppm      : float        # [ppm] at t1+tau_res
    NO_ppm       : float        # [ppm]
    NO2_ppm      : float        # [ppm]
    tau_res_ms   : float        # [ms]  geometric residence time
    t1_us        : float        # [us]  auto-ignition onset
    EI_NOx       : float        # [g/kg_fuel]
    EI_H2O       : float        # [g/kg_fuel]
    mdot_fuel_gs : float        # [g/s]
    MW_mix       : float        # [g/mol]
    Ma_COMB      : float        # [-]   derived combustor-inlet Mach
    Da           : float = np.nan  # [-]  Damköhler number
    eta_exit     : float = np.nan  # [-]  mixing efficiency at combustor exit
    converged    : bool  = True
    error        : str   = ""


class NOxSweep:
    """
    Parametric sweep over (phi, h0, Ma0) using PSRReactor.

    Usage
    -----
        cfg   = SweepConfig(phi_range=[0.3, 0.5, 0.7],
                            h0_range =[25_000, 30_000],
                            Ma0_range=[4.5, 5.0, 5.5])
        sweep = NOxSweep(cfg)
        sweep.save_csv()
        sweep.plot_nox_surface(phi_target=0.7)
        sweep.plot_mach_vs_tau()
        sweep.plot_phi_vs_nox(Ma0_target=5.0, h0_target=30_000)
    """

    def __init__(self, config: SweepConfig = None):
        self.cfg     = config or SweepConfig()
        self.results : List[SweepPoint] = []
        self._run()

    def _run(self):
        combos = list(itertools.product(
            self.cfg.phi_range,
            self.cfg.h0_range,
            self.cfg.Ma0_range,
        ))
        n = len(combos)
        print(f"\n{'='*62}")
        print(f"  NOx PSR Sweep  --  {n} cases")
        print(f"  phi = {self.cfg.phi_range}")
        print(f"  h   = {[h/1e3 for h in self.cfg.h0_range]} km")
        print(f"  Ma0 = {self.cfg.Ma0_range}")
        print(f"{'='*62}")

        for i, (phi, h0, Ma0) in enumerate(combos, 1):
            tag = f"  [{i:>3}/{n}]  phi={phi:.2f}  h={h0/1e3:.0f}km  M={Ma0:.1f}"
            print(tag, end="  ->  ", flush=True)
            try:
                psr = PSRReactor(Ma0=Ma0, h0=h0, phi=phi)
                r   = psr.run_psr(verbose=False)

                pt = SweepPoint(
                    phi          = phi,
                    h0           = h0,
                    Ma0          = Ma0,
                    T_mix        = psr.T_mix,
                    T_at_tau     = r['T_at_tau'],
                    T_ad         = r['T_ad'],
                    T_check_ok   = r['T_check_ok'],
                    NOx_ppm      = r['NOx_at_tau'],
                    NO_ppm       = r['NO_at_tau'],
                    NO2_ppm      = r['NO2_at_tau'],
                    tau_res_ms   = r['tau_res'] * 1e3,
                    t1_us        = r['t1']      * 1e6,
                    EI_NOx       = r['EI_NOx'],
                    EI_H2O       = r['EI_H2O'],
                    mdot_fuel_gs = psr.mdot_fuel * 1e3,
                    MW_mix       = r['MW_mix'],
                    Ma_COMB      = psr.Ma_COMB,
                    Da           = r['Da'],
                    eta_exit     = r['eta_exit'],
                    converged    = True,
                )
                flag = '' if pt.T_check_ok else '  [T FAIL]'
                print(f"NOx={pt.NOx_ppm:7.2f} ppm  "
                      f"EI={pt.EI_NOx:.4f} g/kg  "
                      f"T={pt.T_at_tau:.0f}/{pt.T_ad:.0f} K  "
                      f"tau={pt.tau_res_ms:.2f} ms{flag}")

            except Exception as exc:
                pt = SweepPoint(
                    phi=phi, h0=h0, Ma0=Ma0,
                    T_mix=np.nan, T_at_tau=np.nan, T_ad=np.nan,
                    T_check_ok=False,
                    NOx_ppm=np.nan, NO_ppm=np.nan, NO2_ppm=np.nan,
                    tau_res_ms=np.nan, t1_us=np.nan,
                    EI_NOx=np.nan, EI_H2O=np.nan,
                    mdot_fuel_gs=np.nan, MW_mix=np.nan, Ma_COMB=np.nan,
                    converged=False, error=str(exc),
                )
                print(f"FAILED -- {exc}")

            self.results.append(pt)

        n_ok = sum(p.converged for p in self.results)
        print(f"\n  Sweep complete: {n_ok}/{n} converged.\n")

    # ── Export ────────────────────────────────────────────────────────────────

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

    # ── Plot 1 — 3D surface: Mach x Altitude x NOx (fixed phi) ──────────────

    def plot_nox_surface(self, phi_target=0.7, save="nox_surface.png"):
        pts = [p for p in self._ok() if abs(p.phi - phi_target) < 1e-9]
        if not pts:
            print(f"  No data for phi={phi_target}")
            return

        Ma_vals = sorted(set(p.Ma0 for p in pts))
        h_vals  = sorted(set(p.h0  for p in pts))
        Z       = np.full((len(h_vals), len(Ma_vals)), np.nan)
        for p in pts:
            Z[h_vals.index(p.h0), Ma_vals.index(p.Ma0)] = p.NOx_ppm

        Ma_m, h_m = np.meshgrid(Ma_vals, [h/1e3 for h in h_vals])
        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Ma_m, h_m, Z, cmap='plasma',
                               edgecolor='none', alpha=0.9)
        ax.scatter([p.Ma0 for p in pts], [p.h0/1e3 for p in pts],
                   [p.NOx_ppm for p in pts], color='white', s=20, zorder=5)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label='NOx [ppm]')
        ax.set_xlabel('Mach [-]', labelpad=10)
        ax.set_ylabel('Altitude [km]', labelpad=10)
        ax.set_zlabel('NOx [ppm]', labelpad=10)
        ax.set_title(f'Exit NOx — Mach vs Altitude  (phi={phi_target})', pad=15)
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")

    # ── Plot 2 — Mach vs tau (from ResidenceTime, not PSR clock) ─────────────

    def plot_mach_vs_tau(self, phi_target=None, save="nox_mach_vs_tau.png"):
        cfg     = self.cfg
        phi_val = phi_target if phi_target is not None else cfg.phi_range[0]
        n_h     = len(cfg.h0_range)
        cmap    = plt.cm.get_cmap('viridis', n_h)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i, h0 in enumerate(sorted(cfg.h0_range)):
            taus = []
            for Ma0 in sorted(cfg.Ma0_range):
                try:
                    rt = ResidenceTime(Ma0=Ma0, h0=h0, phi=phi_val, verbose=False)
                    taus.append(rt.residence_time * 1e3)
                except Exception:
                    taus.append(np.nan)
            ax.plot(sorted(cfg.Ma0_range), taus, marker='o',
                    color=cmap(i), lw=1.8, label=f'h={h0/1e3:.0f} km')

        ax.set_xlabel('Mach number [-]')
        ax.set_ylabel('Residence time [ms]')
        ax.set_title(f'Combustor Residence Time vs Mach  (phi={phi_val})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")

    # ── Plot 3 — phi vs NOx at fixed M and h ─────────────────────────────────

    def plot_phi_vs_nox(self, Ma0_target=5.0, h0_target=30_000,
                        save="nox_phi_vs_nox.png"):
        pts = sorted(
            [p for p in self._ok()
             if abs(p.Ma0 - Ma0_target) < 1e-9
             and abs(p.h0  - h0_target)  < 1.0],
            key=lambda p: p.phi
        )
        if not pts:
            print(f"  No data for M={Ma0_target}, h={h0_target}")
            return

        phis = [p.phi     for p in pts]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(phis, [p.NOx_ppm for p in pts], 'o-', color='purple',
                lw=2.0, label='NOx')
        ax.plot(phis, [p.NO_ppm  for p in pts], 's--', color='steelblue',
                lw=1.6, label='NO')
        ax.plot(phis, [p.NO2_ppm for p in pts], '^--', color='darkorange',
                lw=1.6, label='NO2')
        ax.set_xlabel('Equivalence ratio phi [-]')
        ax.set_ylabel('NOx [ppm]')
        ax.set_title(f'NOx vs phi   (M={Ma0_target},  h={h0_target/1e3:.0f} km)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")

    # ── Plot 4 — NOx heatmap (rows=Mach, cols=phi) per altitude ──────────────

    def plot_heatmaps(self, save="nox_heatmaps.png"):
        cfg = self.cfg
        pts = self._ok()
        n_h = len(cfg.h0_range)

        fig, axes = plt.subplots(1, n_h, figsize=(5*n_h, 4))
        if n_h == 1:
            axes = [axes]

        for ax, h0 in zip(axes, sorted(cfg.h0_range)):
            grid = np.full((len(cfg.Ma0_range), len(cfg.phi_range)), np.nan)
            for p in pts:
                if abs(p.h0 - h0) < 1.0:
                    i = sorted(cfg.Ma0_range).index(p.Ma0)
                    j = sorted(cfg.phi_range).index(p.phi)
                    grid[i, j] = p.NOx_ppm
            im = ax.imshow(grid, aspect='auto', origin='lower',
                           cmap='YlOrRd', interpolation='nearest')
            plt.colorbar(im, ax=ax, label='NOx [ppm]')
            ax.set_xticks(range(len(cfg.phi_range)))
            ax.set_xticklabels([f'{v:.2f}' for v in sorted(cfg.phi_range)])
            ax.set_yticks(range(len(cfg.Ma0_range)))
            ax.set_yticklabels([f'{v:.1f}' for v in sorted(cfg.Ma0_range)])
            ax.set_xlabel('phi [-]')
            ax.set_ylabel('Mach [-]')
            ax.set_title(f'h = {h0/1e3:.0f} km')
            for i in range(len(cfg.Ma0_range)):
                for j in range(len(cfg.phi_range)):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f'{grid[i,j]:.1f}',
                                ha='center', va='center', fontsize=7)

        fig.suptitle('NOx [ppm] — PSR at combustor exit', fontsize=13)
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")

    # ── Plot 5 — EI_NOx surface (Mach x Altitude, fixed phi) ─────────────────

    def plot_ei_surface(self, phi_target=0.7, save="nox_ei_surface.png"):
        pts = [p for p in self._ok() if abs(p.phi - phi_target) < 1e-9]
        if not pts:
            print(f"  No data for phi={phi_target}")
            return

        Ma_vals = sorted(set(p.Ma0 for p in pts))
        h_vals  = sorted(set(p.h0  for p in pts))
        Z       = np.full((len(h_vals), len(Ma_vals)), np.nan)
        for p in pts:
            Z[h_vals.index(p.h0), Ma_vals.index(p.Ma0)] = p.EI_NOx

        Ma_m, h_m = np.meshgrid(Ma_vals, [h/1e3 for h in h_vals])
        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Ma_m, h_m, Z, cmap='YlOrRd',
                               edgecolor='none', alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1,
                     label='EI_NOx [g/kg_fuel]')
        ax.set_xlabel('Mach [-]', labelpad=10)
        ax.set_ylabel('Altitude [km]', labelpad=10)
        ax.set_zlabel('EI_NOx [g/kg_fuel]', labelpad=10)
        ax.set_title(f'EI NOx — Mach vs Altitude  (phi={phi_target})', pad=15)
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {save}")


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":

    # -- Single-point baseline
    print("=" * 60)
    print("  Baseline PSR  phi=0.5  h=30 km  M=5.0")
    print("=" * 60)
    psr = PSRReactor()
    psr.run_psr(verbose=True)
    psr.plot(title="PSR_baseline_phi0.5_h30km_M5.0")

    # -- Parametric sweep
    cfg = SweepConfig(
        phi_range = [0.3, 0.4, 0.5, 0.6, 0.7],
        h0_range  = [20_000, 25_000, 30_000, 32_000],
        Ma0_range = [2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        verbose   = False,
    )

    sweep = NOxSweep(cfg)
    sweep.save_csv("nox_sweep_results.csv")
    sweep.plot_nox_surface(phi_target=0.7)
    sweep.plot_mach_vs_tau()
    sweep.plot_phi_vs_nox(Ma0_target=5.0, h0_target=30_000)
    sweep.plot_heatmaps()
    sweep.plot_ei_surface(phi_target=0.7)