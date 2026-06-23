"""
staged_psr.py
=============
Staged PSR model for H2/air ramjet combustion with physically-motivated
incremental air addition driven by the Li et al. (2023) mixing efficiency
profile η(x).

Replaces the single-zone premixed PSR in nox_psr.py with a multi-stage
sequence that better represents diffusion-flame-like mixing delay:

  Stage 0:  fuel + air for φ_start (default 3.0)  →  react for τ_stage
  Stage k:  add Δmdot_air(k) proportional to Δη(k)  →  react for τ_stage
  ...
  Final:    total mdot_air = ramjet model value  →  target φ achieved

Design choices (per user spec):
  - Air increments  : proportional to η(x) increments (Li et al. profile)
  - τ per stage     : global τ_res (velocity integral) / N_stages
  - Readout         : per-stage + cumulative (NOx, H2O, T, Da)
  - Convergence     : each stage runs for its τ_stage; T check vs T_ad_stage

Dependencies:  nox_psr.py (PSRReactor, BuildEngine, NOX_Assumptions)
               cantera ≥ 3.0, numpy, matplotlib
"""

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass, field
from typing import List

from noss import PSRReactor, NOX_Assumptions


# =============================================================================
#  Per-stage result record
# =============================================================================

@dataclass
class StageResult:
    stage        : int
    phi_in       : float    # equiv. ratio at stage inlet (after air addition)
    T_in         : float    # [K]   mixed inlet temperature
    T_out        : float    # [K]   reactor outlet temperature
    T_ad         : float    # [K]   adiabatic flame temp at stage inlet state
    T_check_ok   : bool     # T_out >= 0.99 * T_ad
    NOx_ppm      : float    # [ppm] NOx mole fraction at stage outlet
    NO_ppm       : float    # [ppm]
    NO2_ppm      : float    # [ppm]
    H2O_ppm      : float    # mole fraction × 1e6
    EI_NOx       : float    # [g/kg_fuel]  cumulative at stage outlet
    tau_stage_ms : float    # [ms]  residence time for this stage
    t_start_ms   : float    # [ms]  cumulative time at stage inlet
    t_end_ms     : float    # [ms]  cumulative time at stage outlet
    mdot_air_in  : float    # [kg/s] air added at this stage boundary
    eta_x        : float    # [-]  mixing efficiency at stage boundary x
    Da_stage     : float    # [-]  1 - eta at stage inlet (unmixed fraction)
    x_boundary   : float    # [m]  x-position of stage boundary in combustor


# =============================================================================
#  StagedPSR
# =============================================================================

class StagedPSR(PSRReactor):
    """
    Multi-stage PSR with η(x)-proportional air addition.

    Parameters
    ----------
    Ma0, h0, phi    : flight condition + target equivalence ratio
    N_stages        : number of PSR stages (default 10)
    phi_start       : initial equivalence ratio for stage 0 (default 3.0)
    n_points        : time points per stage reactor integration
    qdot_threshold  : fraction of max qdot used to detect ignition (stage 0)
    """

    def __init__(self,
                 Ma0:            float = None,
                 h0:             float = None,
                 phi:            float = None,
                 N_stages:       int   = 10,
                 phi_start:      float = 3.0,
                 n_points:       int   = 1000,
                 qdot_threshold: float = 0.05):

        super().__init__(Ma0=Ma0, h0=h0, phi=phi,
                         n_points=n_points,
                         qdot_threshold=qdot_threshold)

        self.N_stages  = N_stages
        self.phi_start = phi_start

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _T_ad_of_gas(self, gas: ct.Solution) -> float:
        """Adiabatic flame temperature of current gas state."""
        g = ct.Solution(self.MECH)
        g.TPX = gas.T, gas.P, gas.X
        g.equilibrate('HP')
        return g.T

    def _MW_mix(self, gas: ct.Solution) -> float:
        return gas.mean_molecular_weight

    def _run_stage_reactor(self,
                           gas:    ct.Solution,
                           tau_s:  float,
                           n_pts:  int) -> dict:
        """
        Advance a ConstPressure reactor for exactly tau_s seconds.
        Returns dict of time-series arrays: t, T, qdot, NO, NO2, NOx, OH, H2, H2O.
        """
        r   = ct.IdealGasConstPressureReactor(gas, clone=False)
        net = ct.ReactorNet([r])
        net.atol = 1e-18
        net.rtol = 1e-10

        sp  = r.phase.species_names
        gx  = lambda n: float(r.phase.X[sp.index(n)]) if n in sp else 0.0
        qdot_fn = lambda: float(-np.dot(r.phase.partial_molar_enthalpies,
                                        r.phase.net_production_rates))

        t_span = np.logspace(
            np.log10(max(tau_s * 1e-4, 1e-12)),
            np.log10(max(tau_s, 1e-9)),
            n_pts
        )

        store = {k: [] for k in ['t', 'T', 'qdot',
                                  'NO', 'NO2', 'NOx', 'OH', 'H2', 'H2O']}
        for t_tgt in t_span:
            net.advance(t_tgt)
            store['t'   ].append(t_tgt)
            store['T'   ].append(r.phase.T)
            store['qdot'].append(qdot_fn())
            store['NO'  ].append(gx('NO'))
            store['NO2' ].append(gx('NO2'))
            store['NOx' ].append(gx('NO') + gx('NO2'))
            store['OH'  ].append(gx('OH'))
            store['H2'  ].append(gx('H2'))
            store['H2O' ].append(gx('H2O'))

        # Update gas to final state for caller to carry forward
        # (reactor already advanced; extract final phase state)
        return {k: np.array(v) for k, v in store.items()}

    def _mix_air_into_gas(self,
                          gas:       ct.Solution,
                          mdot_cur:  float,
                          mdot_air_add: float) -> ct.Solution:
        """
        Adiabatically mix an air increment into the current gas state.
        Conserves total enthalpy (HP-mix), returns updated gas.

        gas        : current Cantera Solution (will be modified in-place)
        mdot_cur   : total mass flow already in gas [kg/s]
        mdot_air_add: mass flow of air being added [kg/s]
        """
        # Air state: same pressure, engine inlet temperature (T_air)
        gas_air = ct.Solution(self.MECH)
        gas_air.TPX = self.T_air, gas.P, 'O2:1.0, N2:3.76'

        h_mix = ((mdot_cur       * gas.enthalpy_mass +
                  mdot_air_add   * gas_air.enthalpy_mass) /
                 (mdot_cur + mdot_air_add))

        # New composition: mass-weighted species
        Y_cur = gas.Y                   # shape (n_species,)
        Y_air = gas_air.Y
        f     = mdot_cur / (mdot_cur + mdot_air_add)
        Y_mix = f * Y_cur + (1.0 - f) * Y_air

        gas.HPY = h_mix, gas.P, Y_mix
        return gas

    # ── Stage boundary x-positions from η(x) ─────────────────────────────────

    def _compute_stage_boundaries(self) -> tuple:
        """
        Divide [0, L_comb] into N_stages sub-intervals such that each
        interval carries equal Δη from the Li et al. mixing profile.

        Returns
        -------
        x_bounds : (N_stages+1,) array  — x positions [0, x1, x2, ..., L_comb]
        eta_bounds: (N_stages+1,) array — η at each boundary
        """
        # Dense x-grid over combustor
        x_fine   = np.linspace(0.0, self.L_comb, 5000)
        eta_fine = np.array([self.mixing_eta(xi) for xi in x_fine])

        eta_0    = eta_fine[0]
        eta_end  = eta_fine[-1]
        delta_eta = (eta_end - eta_0) / self.N_stages

        x_bounds   = [0.0]
        eta_bounds  = [eta_0]

        for k in range(1, self.N_stages):
            target = eta_0 + k * delta_eta
            # Find first x where η ≥ target
            idx = int(np.argmax(eta_fine >= target))
            x_bounds.append(float(x_fine[idx]))
            eta_bounds.append(float(eta_fine[idx]))

        x_bounds.append(float(self.L_comb))
        eta_bounds.append(float(eta_end))

        return np.array(x_bounds), np.array(eta_bounds)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run_staged(self, verbose: bool = True) -> dict:
        """
        Run the staged PSR.

        Returns dict with:
          'stages'        : List[StageResult]
          'cumulative'    : dict of arrays (t_ms, T, NOx_ppm, NO_ppm,
                            NO2_ppm, H2O_ppm, EI_NOx, stage_idx)
          'tau_res'       : global residence time [s]
          'tau_stage'     : per-stage residence time [s]
          'T_ad_final'    : adiabatic flame temp at final stage inlet
          'NOx_final_ppm' : NOx at end of final stage
          'EI_NOx_final'  : EI_NOx at end of final stage [g/kg_fuel]
          'x_bounds'      : stage boundary x positions [m]
          'eta_bounds'    : η at each stage boundary
        """
        # ── 0. Engine inlet state ─────────────────────────────────────────────
        self._set_inlet_state(verbose=verbose)

        mdot_air_total  = self.mdot_air
        mdot_fuel_total = self.mdot_fuel
        P               = float(self.P_in)

        # ── 1. Stage boundaries from η(x) ────────────────────────────────────
        x_bounds, eta_bounds = self._compute_stage_boundaries()
        delta_eta = np.diff(eta_bounds)          # air fraction added per stage

        if verbose:
            print(f"\n-- Stage Boundaries (η-proportional) --")
            for k, (xb, eb) in enumerate(zip(x_bounds, eta_bounds)):
                print(f"  Stage {k:>2}  x={xb*100:6.2f} cm  η={eb:.4f}")

        # ── 2. Global τ_res (velocity integral) ──────────────────────────────
        x0   = self.L01 + self.L12
        x1   = x0 + self.L_comb
        mask = (self.x >= x0) & (self.x <= x1)
        tau_res   = float(np.trapezoid(1.0 / self.V[mask], self.x[mask]))
        tau_stage = tau_res / self.N_stages

        if verbose:
            print(f"\n-- Timing --")
            print(f"  tau_res   : {tau_res*1e3:.4f} ms")
            print(f"  tau_stage : {tau_stage*1e6:.2f} us  (x {self.N_stages} stages)")

        # ── 3. Stoichiometric F/A ratio for H2/air ───────────────────────────
        # φ = (F/A) / (F/A)_stoich  →  mdot_air_stoich = mdot_fuel / phi / FAR_stoich
        # For H2: stoichiometric O2 : H2 = 0.5, so FAR_stoich = MW_H2 / (0.5*MW_O2 + …)
        # Easier: use Cantera
        _g_stoich = ct.Solution(self.MECH)
        _g_stoich.set_equivalence_ratio(1.0, 'H2:1.0', 'O2:1.0, N2:3.76')
        FAR_stoich = _g_stoich.Y[_g_stoich.species_index('H2')]  / \
                     (1.0 - _g_stoich.Y[_g_stoich.species_index('H2')])
        # mdot_air to achieve a given phi:
        #   phi = (mdot_fuel / mdot_air) / FAR_stoich
        #   mdot_air = mdot_fuel / (phi * FAR_stoich)
        mdot_air_stage0 = mdot_fuel_total / (self.phi_start * FAR_stoich)

        # Guard: clamp stage-0 air using the velocity-field residence time.
        # tau_stage0 = integral_{x_comb_start}^{x_comb_start + x_bounds[1]} (1/u) dx
        # The fraction of tau_res in stage 0 caps how much air can be mixed there.
        x_comb_abs = self.L01 + self.L12               # absolute combustor inlet [m]
        x_s0_end   = x_comb_abs + float(x_bounds[1])  # stage-0 exit (absolute)
        mask_s0    = (self.x >= x_comb_abs) & (self.x <= x_s0_end)
        if mask_s0.sum() >= 2:
            tau_s0 = float(np.trapezoid(1.0 / self.V[mask_s0], self.x[mask_s0]))
        else:
            tau_s0 = tau_res / self.N_stages
        tau_frac_s0     = float(np.clip(tau_s0 / tau_res, 1e-3, 1.0 - 1e-3))
        mdot_air_max_s0 = mdot_air_total * tau_frac_s0
        clamped         = mdot_air_stage0 > mdot_air_max_s0
        if clamped:
            mdot_air_stage0 = mdot_air_max_s0

        if verbose:
            print(f'\n-- Stage 0 Init --')
            print(f'  FAR_stoich      : {FAR_stoich:.5f}')
            print(f'  phi_start       : {self.phi_start:.2f}')
            print(f'  tau_s0          : {tau_s0*1e6:.2f} us  '
                  f'(tau fraction = {tau_frac_s0:.4f})')
            print(f'  mdot_air_max_s0 : {mdot_air_max_s0*1e3:.4f} g/s  (tau-limit)')
            clamped_tag = '  [clamped]' if clamped else ''
            print(f'  mdot_air_stage0 : {mdot_air_stage0*1e3:.4f} g/s{clamped_tag}')
            print(f'  mdot_air_total  : {mdot_air_total*1e3:.4f} g/s')
            print(f'  mdot_fuel       : {mdot_fuel_total*1e6:.4f} mg/s')

        # Equal air chunks per stage.
        # Each stage after stage-0 receives an equal share of the remaining air,
        # but the chunk is clamped so that post-mix φ never drops below self.phi
        # (the target equivalence ratio).  Any residual air that would push below
        # the target is withheld — the φ at stage entry is therefore always >= phi.
        #
        # mdot_air_chunk = (mdot_air_total - mdot_air_stage0) / (N_stages - 1)
        # clamp per injection: don't add air that would take current φ < phi_target
        n_remaining = max(self.N_stages - 1, 1)
        mdot_air_chunk = (mdot_air_total - mdot_air_stage0) / n_remaining

        # φ_min_mdot_air: for a given current gas mass flow, the maximum air
        # addition that keeps post-mix φ >= self.phi.
        #   phi_post = mdot_fuel / ((mdot_air_cur + delta) * FAR_stoich)
        #   delta_max = mdot_fuel / (self.phi * FAR_stoich) - mdot_air_cur
        def _max_air_for_phi(mdot_fuel, mdot_air_cur):
            limit = mdot_fuel / (self.phi * FAR_stoich) - mdot_air_cur
            return max(limit, 0.0)

        if verbose:
            print(f"\n  Air distribution: equal chunks")
            print(f"  mdot_air_chunk  : {mdot_air_chunk*1e3:.4f} g/s  "
                  f"x {n_remaining} stages")
            print(f"  phi floor       : {self.phi:.3f}  (clamp limit per stage)")

        # ── 4. Initialise gas for stage 0 ────────────────────────────────────
        gas = ct.Solution(self.MECH)
        for transport in ('multicomponent', 'mixture-averaged'):
            try:
                gas.transport_model = transport
                break
            except Exception:
                continue

        # Stage 0: fuel + stage-0 air, adiabatic mix
        gas_air0  = ct.Solution(self.MECH)
        gas_air0.TPX = self.T_air, P, 'O2:1.0, N2:3.76'
        gas_fuel0 = ct.Solution(self.MECH)
        gas_fuel0.TPX = NOX_Assumptions.T_fuel, P, 'H2:1.0'

        mdot_cur  = mdot_fuel_total + mdot_air_stage0
        h_mix0    = ((mdot_fuel_total  * gas_fuel0.enthalpy_mass +
                      mdot_air_stage0  * gas_air0.enthalpy_mass) / mdot_cur)
        f_fuel    = mdot_fuel_total / mdot_cur
        f_air     = mdot_air_stage0 / mdot_cur
        Y_mix0    = f_fuel * gas_fuel0.Y + f_air * gas_air0.Y
        gas.HPY   = h_mix0, P, Y_mix0

        mdot_air_added_this_stage = mdot_air_stage0   # track per-stage air injection

        # ── 5. Stage loop ─────────────────────────────────────────────────────
        stage_results : List[StageResult] = []

        # Cumulative arrays (for plotting)
        cum_t_ms    = []
        cum_T       = []
        cum_NOx_ppm = []
        cum_NO_ppm  = []
        cum_NO2_ppm = []
        cum_H2O     = []
        cum_EI      = []
        cum_stage   = []

        t_cumulative = 0.0   # running clock [s]

        for k in range(self.N_stages):

            # ── φ at stage inlet ──────────────────────────────────────────────
            phi_in = float(gas.equivalence_ratio('H2:1.0', 'O2:1.0, N2:3.76',
                                                  basis='mole'))

            T_in = gas.T

            # ── Adiabatic flame temp for this stage state ─────────────────────
            T_ad_stage = self._T_ad_of_gas(gas)

            # ── Da for this stage: unmixed fraction at stage boundary ─────────
            # η at stage-k inlet boundary
            eta_in = float(eta_bounds[k])
            Da_stage = 1.0 - eta_in

            # ── Run reactor for τ_stage ───────────────────────────────────────
            ts = self._run_stage_reactor(gas, tau_stage, self.n_points)

            T_out = float(ts['T'][-1])
            T_check_ok = T_out >= 0.99 * T_ad_stage

            # Final species
            sp = gas.species_names
            NOx_out = (float(ts['NOx'][-1])) * 1e6
            NO_out  = (float(ts['NO' ][-1])) * 1e6
            NO2_out = (float(ts['NO2'][-1])) * 1e6
            H2O_out = float(ts['H2O'][-1])    * 1e6

            # ── EI_NOx cumulative at stage outlet ─────────────────────────────
            MW_out  = gas.mean_molecular_weight
            FAR_inv = (mdot_air_total + mdot_fuel_total) / mdot_fuel_total
            EI_NOx_out = (NOx_out * 1e-6) * (46.005 / MW_out) * FAR_inv * 1e3

            tau_stage_ms = tau_stage * 1e3
            t_start_ms   = t_cumulative * 1e3
            t_end_ms     = (t_cumulative + tau_stage) * 1e3

            # ── Accumulate time-series for cumulative plot ────────────────────
            for j, t_j in enumerate(ts['t']):
                cum_t_ms.append((t_cumulative + t_j) * 1e3)
                cum_T.append(ts['T'][j])
                cum_NOx_ppm.append(ts['NOx'][j] * 1e6)
                cum_NO_ppm.append(ts['NO'][j] * 1e6)
                cum_NO2_ppm.append(ts['NO2'][j] * 1e6)
                cum_H2O.append(ts['H2O'][j] * 1e6)
                # EI at each point (approximate, using exit MW)
                cum_EI.append((ts['NOx'][j]) * (46.005 / MW_out) * FAR_inv * 1e3)
                cum_stage.append(k)

            stage_results.append(StageResult(
                stage        = k,
                phi_in       = phi_in,
                T_in         = T_in,
                T_out        = T_out,
                T_ad         = T_ad_stage,
                T_check_ok   = T_check_ok,
                NOx_ppm      = NOx_out,
                NO_ppm       = NO_out,
                NO2_ppm      = NO2_out,
                H2O_ppm      = H2O_out,
                EI_NOx       = EI_NOx_out,
                tau_stage_ms = tau_stage_ms,
                t_start_ms   = t_start_ms,
                t_end_ms     = t_end_ms,
                mdot_air_in  = mdot_air_added_this_stage,
                eta_x        = eta_in,
                Da_stage     = Da_stage,
                x_boundary   = float(x_bounds[k]),
            ))

            if verbose:
                flag = '' if T_check_ok else '  [T FAIL]'
                print(f"  Stage {k:>2}  φ_in={phi_in:.3f}  "
                      f"T_in={T_in:.0f}K  T_out={T_out:.0f}K  "
                      f"T_ad={T_ad_stage:.0f}K  "
                      f"NOx={NOx_out:.2f}ppm  "
                      f"EI={EI_NOx_out:.4f} g/kg"
                      f"{flag}")

            # ── Advance cumulative clock ──────────────────────────────────────
            t_cumulative += tau_stage

            # ── Mix in air for NEXT stage (if not last) ───────────────────────
            if k < self.N_stages - 1:
                # Equal chunk, clamped so post-mix φ >= self.phi (target)
                mdot_add = min(mdot_air_chunk,
                               _max_air_for_phi(mdot_fuel_total, mdot_cur - mdot_fuel_total))
                mdot_add = max(mdot_add, 0.0)
                if mdot_add > 0.0:
                    gas = self._mix_air_into_gas(gas, mdot_cur, mdot_add)
                    mdot_cur += mdot_add
                mdot_air_added_this_stage = mdot_add   # recorded for next StageResult

        # ── 6. Summary ────────────────────────────────────────────────────────
        final = stage_results[-1]
        if verbose:
            print(f"\n-- Staged PSR Summary --")
            print(f"  N_stages        : {self.N_stages}")
            print(f"  tau_res         : {tau_res*1e3:.4f} ms")
            print(f"  tau_stage       : {tau_stage*1e6:.2f} us")
            print(f"  phi_start       : {self.phi_start:.2f}  (stage 0 inlet)")
            print(f"  phi_final       : {final.phi_in:.4f}  (stage N-1 inlet)")
            print(f"  T_out (final)   : {final.T_out:.1f} K")
            print(f"  T_ad  (final)   : {final.T_ad:.1f} K")
            print(f"  NOx (final)     : {final.NOx_ppm:.2f} ppm")
            print(f"  EI_NOx (final)  : {final.EI_NOx:.4f} g/kg_fuel")

        cum = dict(
            t_ms    = np.array(cum_t_ms),
            T       = np.array(cum_T),
            NOx_ppm = np.array(cum_NOx_ppm),
            NO_ppm  = np.array(cum_NO_ppm),
            NO2_ppm = np.array(cum_NO2_ppm),
            H2O_ppm = np.array(cum_H2O),
            EI_NOx  = np.array(cum_EI),
            stage   = np.array(cum_stage),
        )

        self.staged_results = dict(
            stages       = stage_results,
            cumulative   = cum,
            tau_res      = tau_res,
            tau_stage    = tau_stage,
            T_ad_final   = final.T_ad,
            NOx_final_ppm= final.NOx_ppm,
            EI_NOx_final = final.EI_NOx,
            x_bounds     = x_bounds,
            eta_bounds   = eta_bounds,
        )
        return self.staged_results

    # ── Plot ──────────────────────────────────────────────────────────────────

    def plot_staged(self, title: str = None, save: str = None):
        """
        Four-panel plot — x-axis is cumulative time [ms] throughout.
          Panel 1 — Temperature vs time  (T_ad steps, 99% T_ad steps)
          Panel 2 — Da = 1−η(x) per stage bar vs time
          Panel 3 — NOx / NO / NO2 [ppm] vs time
          Panel 4 — H2O [%] vs time

        On every panel:
          • grey dashed verticals at stage boundaries
          • black dashed vertical at τ_res (total residence time)
          • φ_in annotations below each stage midpoint (panel 1)
        """
        r      = self.staged_results
        stages = r['stages']
        cum    = r['cumulative']

        t_ms      = cum['t_ms']
        T         = cum['T']
        NOx       = cum['NOx_ppm']
        NO        = cum['NO_ppm']
        NO2       = cum['NO2_ppm']
        H2O       = cum['H2O_ppm'] / 1e4          # ppm → percent

        tau_res_ms   = r['tau_res'] * 1e3          # total residence time [ms]
        t_bounds_ms  = ([s.t_start_ms for s in stages]
                        + [stages[-1].t_end_ms])   # N+1 boundaries

        fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
        suptitle  = (title or
                     f"Staged PSR  N={self.N_stages}  "
                     f"φ_start={self.phi_start:.1f}→{self.phi:.2f}  "
                     f"Ma={self.Ma_COMB:.2f}")
        fig.suptitle(suptitle, fontsize=13, y=1.002)

        cmap   = plt.cm.get_cmap('tab10', self.N_stages)
        colors = [cmap(k) for k in range(self.N_stages)]

        # ── Shared helpers ────────────────────────────────────────────────────

        def _shade_stages(ax):
            for k, s in enumerate(stages):
                ax.axvspan(s.t_start_ms, s.t_end_ms,
                           alpha=0.06, color=colors[k])

        def _stage_vlines(ax):
            """Grey dashed verticals at interior stage boundaries."""
            for tb in t_bounds_ms[1:-1]:
                ax.axvline(tb, color='grey', lw=0.8, ls='--', alpha=0.55)

        def _tau_res_vline(ax):
            """Black dashed vertical at τ_res."""
            ax.axvline(tau_res_ms, color='black', lw=1.4, ls='--',
                       label=f'τ_res = {tau_res_ms:.3f} ms')

        # ── Panel 1: Temperature ──────────────────────────────────────────────
        ax = axes[0]
        _shade_stages(ax)
        ax.plot(t_ms, T, color='firebrick', lw=1.8, label='T(t)')
        for s in stages:
            ax.hlines(s.T_ad, s.t_start_ms, s.t_end_ms,
                      color='firebrick', lw=0.9, ls=':', alpha=0.55)
            ax.hlines(0.99 * s.T_ad, s.t_start_ms, s.t_end_ms,
                      color='orange', lw=0.8, ls='--', alpha=0.45)
        _stage_vlines(ax)
        _tau_res_vline(ax)
        ax.set_ylabel('Temperature [K]')
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which='both')

        # φ_in annotations — placed after ylim is known; deferred to after
        # tight_layout so we annotate in a second pass below.

        # ── Panel 2: Da per stage (bar, centred on stage midpoint time) ───────
        ax = axes[1]
        _shade_stages(ax)
        for k, s in enumerate(stages):
            mid   = (s.t_start_ms + s.t_end_ms) / 2.0
            width = (s.t_end_ms   - s.t_start_ms) * 0.72
            bar_color = ('green'  if s.Da_stage < 0.1 else
                         'orange' if s.Da_stage < 0.3 else 'red')
            ax.bar(mid, s.Da_stage, width=width,
                   color=bar_color, alpha=0.78, edgecolor='k', linewidth=0.5)
        _stage_vlines(ax)
        _tau_res_vline(ax)
        ax.axhline(0.1, color='green',  lw=0.9, ls=':',
                   label='Da = 0.1  (well-mixed)')
        ax.axhline(0.3, color='orange', lw=0.9, ls=':',
                   label='Da = 0.3  (marginal)')
        ax.set_ylabel('Da = 1 − η(x)  [−]')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

        # ── Panel 3: NOx / NO / NO2 ───────────────────────────────────────────
        ax = axes[2]
        _shade_stages(ax)
        ax.plot(t_ms, NOx, color='purple',     lw=1.8, ls='--', label='NOx')
        ax.plot(t_ms, NO,  color='steelblue',  lw=1.6,          label='NO')
        ax.plot(t_ms, NO2, color='darkorange', lw=1.6,          label='NO2')
        ax.axhline(stages[-1].NOx_ppm, color='purple', lw=0.9, ls=':',
                   label=f"NOx @ τ_res = {stages[-1].NOx_ppm:.2f} ppm")
        _stage_vlines(ax)
        _tau_res_vline(ax)
        ax.set_ylabel('Mole fraction [ppm]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ── Panel 4: H2O ─────────────────────────────────────────────────────
        ax = axes[3]
        _shade_stages(ax)
        ax.plot(t_ms, H2O, color='seagreen', lw=1.8, label='H₂O')
        _stage_vlines(ax)
        _tau_res_vline(ax)
        ax.set_ylabel('H₂O mole fraction [%]')
        ax.set_xlabel('Cumulative time [ms]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ── φ_in annotations on panel 1 (after layout so ylim is final) ───────
        plt.tight_layout()
        y_bot, y_top = axes[0].get_ylim()
        y_phi = y_bot + (y_top - y_bot) * 0.04   # just above bottom axis
        for s in stages:
            mid = (s.t_start_ms + s.t_end_ms) / 2.0
            axes[0].text(mid, y_phi,
                         f'φ={s.phi_in:.2f}',
                         ha='center', va='bottom',
                         fontsize=5.5, color='dimgrey',
                         rotation=90)

        fname = save or (suptitle.replace(' ', '_')
                                 .replace('→', '-')
                                 .replace('=', '')
                                 .replace('.', 'p') + '.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Plot saved -> {fname}")

    # ── Per-stage summary table (text) ────────────────────────────────────────

    def print_stage_table(self):
        r      = self.staged_results
        stages = r['stages']
        print(f"\n{'─'*100}")
        print(f"  {'S':>2}  {'φ_in':>6}  {'T_in':>7}  {'T_out':>7}  "
              f"{'T_ad':>7}  {'T_ok':>5}  {'NOx':>8}  "
              f"{'EI_NOx':>10}  {'Da':>6}  {'η_x':>6}  {'x_b[cm]':>8}")
        print(f"  {'':>2}  {'[-]':>6}  {'[K]':>7}  {'[K]':>7}  "
              f"{'[K]':>7}  {'':>5}  {'[ppm]':>8}  "
              f"{'[g/kg]':>10}  {'[-]':>6}  {'[-]':>6}  {'':>8}")
        print(f"{'─'*100}")
        for s in stages:
            print(f"  {s.stage:>2}  {s.phi_in:>6.3f}  {s.T_in:>7.1f}  "
                  f"{s.T_out:>7.1f}  {s.T_ad:>7.1f}  "
                  f"{'PASS' if s.T_check_ok else 'FAIL':>5}  "
                  f"{s.NOx_ppm:>8.2f}  {s.EI_NOx:>10.4f}  "
                  f"{s.Da_stage:>6.3f}  {s.eta_x:>6.4f}  "
                  f"{s.x_boundary*100:>8.2f}")
        print(f"{'─'*100}")


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Staged PSR  phi_start=3.0 → phi=0.5  h=30km  M=5.0")
    print("=" * 60)

    spsr = StagedPSR(
        Ma0       = 5.0,
        h0        = 30_000,
        phi       = 0.5,
        N_stages  = 30,
        phi_start = 30.0,
    )
    spsr.run_staged(verbose=True)
    spsr.print_stage_table()
    spsr.plot_staged(title="StagedPSR_phi0.5_h30km_M5")