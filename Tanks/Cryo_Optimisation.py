"""
tank_optimizer.py
═══════════════════════════════════════════════════════════════════════
Optimise LH₂ tank geometry (inner diameter d, cylindrical barrel
length L, number of parallel tanks N) to minimise hydrogen boil-off
over an aircraft mission while keeping pressure ≥ 1.2 bar.

Physics summary
───────────────
  Heat leak  Q ∝ k_ins · A_surface / t_ins
  A_surface for a cylinder+caps = π·d·L + π·d²
  Sphere (L→0) minimises A/V²/³ → optimizer pushes toward low L/d.
  More tanks (N↑) increases total surface area as N^(1/3) for fixed
  total volume → fewer, larger tanks reduce heat leak (and boil-off),
  but increase wall thickness and structural mass.

Pressure management
───────────────────
  P < P_MIN → autogenous pressurisation (boil liquid into ullage)
  P > P_MAX → vent ullage overboard (true H₂ loss, drives objective)

Algorithm
─────────
  For each integer N ∈ {1 … N_MAX}:
    Each tank holds M_H₂_total / N kg.
    scipy.differential_evolution minimises total vented H₂ [kg]
    over (d, L) subject to d ∈ [d_min, d_max], L ∈ [L_min, L_max].
    Polished with L-BFGS-B; best N chosen by lowest boil-off %.

Run
───
    python tank_optimizer.py                  # default 3 000 kg / 8 h
    python tank_optimizer.py --m 5000 --t 12  # custom mass & hours
"""

from __future__ import annotations
import argparse
import sys
import time
import warnings
import numpy as np
import CoolProp.CoolProp as cp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  SATURATION PROPERTY CACHE
#  Precomputed once at import time; replaces PropsSI("Q"=0/1,…)
#  calls in the hot simulation loop with fast np.interp lookups.
#  Cuts ~6 PropsSI calls per time step → 3-4× faster simulations.
# ═══════════════════════════════════════════════════════════════

class H2SatLookup:
    """
    Singleton. Builds a 400-point saturation table for H₂ at startup
    (~0.5 s one-off cost) and exposes fast interpolation methods.

    Coverage: 0.4 bar … 6.0 bar  (well outside the P_min–P_max band).
    """

    _instance: "H2SatLookup | None" = None

    @classmethod
    def get(cls) -> "H2SatLookup":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, n: int = 400) -> None:
        print("  Building H₂ saturation table … ", end="", flush=True)
        t0    = time.perf_counter()
        fluid = "Hydrogen"
        P     = np.linspace(0.40e5, 6.0e5, n)

        def _sat(prop, q):
            return np.array([cp.PropsSI(prop, "P", p, "Q", q, fluid) for p in P])

        self._P      = P
        self._T_sat  = _sat("T",      0)
        self._h_l    = _sat("H",      0)
        self._h_g    = _sat("H",      1)
        self._u_l    = _sat("U",      0)
        self._u_g    = _sat("U",      1)
        self._rho_l  = _sat("D",      0)
        self._rho_g  = _sat("D",      1)
        self._cv_g   = _sat("Cvmass", 1)
        self._cp_l   = _sat("Cpmass", 0)

        print(f"done ({time.perf_counter()-t0:.2f} s)")

    # ── Convenience interpolators ──────────────────────────────
    def _interp(self, arr, P):
        return float(np.interp(P, self._P, arr))

    def T_sat   (self, P): return self._interp(self._T_sat, P)
    def h_sat_l (self, P): return self._interp(self._h_l,   P)
    def h_sat_g (self, P): return self._interp(self._h_g,   P)
    def u_sat_l (self, P): return self._interp(self._u_l,   P)
    def u_sat_g (self, P): return self._interp(self._u_g,   P)
    def rho_sat_l(self, P): return self._interp(self._rho_l, P)
    def rho_sat_g(self, P): return self._interp(self._rho_g, P)
    def h_vap   (self, P): return self._interp(self._h_g - self._h_l, P)
    def cv_sat_g (self, P): return self._interp(self._cv_g, P)
    def cp_sat_l (self, P): return self._interp(self._cp_l, P)


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION  (override via CLI or edit defaults below)
# ═══════════════════════════════════════════════════════════════

@dataclass
class MissionConfig:
    """Aircraft mission parameters."""
    m_h2_total   : float =  40718.43      # total LH₂ on board        [kg]
    mission_time : float = 3.0 * 3600   # flight duration            [s]
    T_ambient    : float = 120.0        # effective skin temperature [K]
    T_liq_init   : float = 20.0         # initial liquid temperature [K]
    T_gas_init   : float = 25.0         # initial ullage temperature [K]
    fill_level   : float = 0.93         # V_liq / V_total at t=0    [-]
    P_init       : float = 2.0e5        # initial pressure           [Pa]
    # Pressure band
    P_min        : float = 1.2e5        # pressurisation trigger     [Pa]
    P_refill     : float = 1.5e5        # pressurisation target      [Pa]
    P_max        : float = 3.5e5        # venting trigger            [Pa]
    P_vent       : float = 3.0e5        # venting target             [Pa]


@dataclass
class OptimConfig:
    """Search space and solver parameters."""
    n_tanks_range   : range               = field(default_factory=lambda: range(1, 7))
    d_bounds        : Tuple[float, float] = (0.1,  3.0)   # diameter  [m]
    L_bounds        : Tuple[float, float] = (0.1, 25.0)  # length    [m]
    # Time steps
    dt_optim        : float = 500.0  # coarse step during optimisation [s]
    dt_verify       : float = 15.0   # fine step for final analysis    [s]
    # differential_evolution
    maxiter         : int   = 50
    popsize         : int   = 8
    tol             : float = 1e-3
    seed            : int   = 42
    # Insulation (held constant; vary here to explore sensitivity)
    ins_thickness   : float = 0.03   # [m]
    rho_ins         : float = 33.0   # [kg/m³]
    k_ins           : float = 0.026  # [W/m/K]
    # Structural
    delta_P_max          : float = 10.0e5
    delta_P_operational  : float =  3.0e5
    # Penalty for each Pa of pressure violation below P_min
    penalty_per_pa  : float = 1e3    # [kg/Pa]


# ═══════════════════════════════════════════════════════════════
#  TANK STRUCTURE
# ═══════════════════════════════════════════════════════════════

class TankStructure:
    """
    Structural + insulation mass model for a cylindrical cryogenic
    tank with two hemispherical end caps.

    Wall sizing — thin-wall hoop stress (cylinder governs; caps use
    membrane-sphere formula):
        t_cyl  = ΔP_max · r / (σ_y · η)
        t_caps = ΔP_max · r / (2 · σ_y · η)

    Material: CFRP laminate.
    Overhead (valves, flanges, mounting brackets): 6 % of structural mass.
    """

    # CFRP material properties
    SIGMA_YIELD = 450e6    # design allowable   [Pa]
    RHO_WALL    = 1_560.0  # density            [kg/m³]
    WELD_EFF    = 0.85     # joint efficiency   [-]
    T_WALL_MIN  = 0.002    # manufacturing min  [m]

    def __init__(
        self,
        diameter,
        cylindrical_length,
        delta_P_max         = 10.0e5,
        delta_P_operational = 3.0e5,
        ins_thickness       = 0.03,
        rho_ins             = 33.0,
        k_ins               = 0.026,
    ):
        self.d     = diameter
        self.L_cyl = cylindrical_length
        self.k_ins = k_ins
        self.t_ins = ins_thickness

        r = diameter / 2.0

        # Wall thicknesses
        t_cyl  = max(
            delta_P_max * r / (self.SIGMA_YIELD * self.WELD_EFF),
            self.T_WALL_MIN,
        )
        t_caps = max(
            delta_P_max * r / (2.0 * self.SIGMA_YIELD * self.WELD_EFF),
            self.T_WALL_MIN,
        )

        # Surface areas
        A_cyl  = np.pi * diameter * cylindrical_length
        A_caps = 4.0 * np.pi * r ** 2    # two hemispheres = one sphere

        # Masses
        m_wall = (A_cyl * t_cyl + A_caps * t_caps) * self.RHO_WALL
        m_ins  = (A_cyl + A_caps) * ins_thickness * rho_ins
        m_misc = 0.06 * (m_wall + m_ins)

        self.m_tank    = m_wall + m_ins + m_misc
        self.t_cyl     = t_cyl
        self.t_caps    = t_caps
        self.A_surface = A_cyl + A_caps


# ═══════════════════════════════════════════════════════════════
#  TANK THERMODYNAMICS
# ═══════════════════════════════════════════════════════════════

class TankThermodynamics:
    """
    Two-phase (ullage + bulk liquid) thermodynamic model of a
    cryogenic LH₂ tank following Parello et al. (Eq. 18).

    Extended with handle_pressurization() to close the pressure
    band from below.
    """

    def __init__(
        self,
        initial_mass,
        initial_pressure,
        initial_temperature_gas,
        initial_temperature_liq,
        fill_level          = 0.93,
        diameter            = 3.5,
        cylindrical_length  = 20.0,
        delta_P_max         = 10.0e5,
        delta_P_operational = 3.0e5,
        ins_thickness       = 0.03,
        rho_ins             = 33.0,
        k_ins               = 0.026,
    ):
        self.fluid      = "Hydrogen"
        self.fill_level = fill_level
        self.P          = initial_pressure
        self.T_l        = initial_temperature_liq
        self.T_g        = initial_temperature_gas

        rho_l_init = cp.PropsSI("D", "P", self.P, "T", self.T_l, self.fluid)

        V_liq_init  = initial_mass / rho_l_init
        V_total     = V_liq_init / fill_level
        self.volume = V_total
        self.V_l    = V_liq_init
        self.V_g    = V_total - V_liq_init

        self.d     = diameter
        self.L_cyl = cylindrical_length

        self.structure = TankStructure(
            diameter            = self.d,
            cylindrical_length  = self.L_cyl,
            delta_P_max         = delta_P_max,
            delta_P_operational = delta_P_operational,
            ins_thickness       = ins_thickness,
            rho_ins             = rho_ins,
            k_ins               = k_ins,
        )
        self.m_tank = self.structure.m_tank

        rho_g_init = cp.PropsSI("D", "P", self.P, "T", self.T_g, self.fluid)
        self.m_l   = rho_l_init * self.V_l
        self.m_g   = rho_g_init * self.V_g
        self.m_H2  = self.m_l + self.m_g

        self.T_sat = H2SatLookup.get().T_sat(self.P)
        self.T_l   = min(self.T_l, self.T_sat - 0.01)
        self.T_g   = max(self.T_g, self.T_sat + 0.2)

        r       = self.d / 2.0
        A_cyl   = np.pi * self.d * self.L_cyl
        a_cap   = r
        b_cap   = r / 2.0
        e2      = 1.0 - (b_cap / a_cap) ** 2
        e_      = np.sqrt(max(e2, 1e-12))
        A_caps  = 2.0 * 2.0 * np.pi * a_cap ** 2 * (
                  1.0 + (1.0 - e2) / e_ * np.arctanh(e_))
        self.A_surface = A_cyl + A_caps
        self.UA_ins    = k_ins * self.A_surface / ins_thickness

    # ── Gravimetric efficiency ──────────────────────────────────

    def gravimetric_efficiency(self):
        return self.m_H2 / (self.m_H2 + self.m_tank)

    def gravimetric_efficiency_adjusted(self, m_boiloff):
        return (self.m_H2 - m_boiloff) / (self.m_H2 + self.m_tank)

    # ── Pressure update ─────────────────────────────────────────

    def update_pressure(self):
        _sat      = H2SatLookup.get()
        rho_g     = max(self.m_g / self.V_g, 1e-8)
        self.P    = cp.PropsSI("P", "D", rho_g, "T", self.T_g, self.fluid)
        self.T_sat = _sat.T_sat(self.P)          # interp, not PropsSI
        self.T_l  = min(self.T_l, self.T_sat - 0.01)
        self.T_g  = max(self.T_g, self.T_sat + 0.2)

    # ── Liquid property lookup ──────────────────────────────────

    def liquid_properties(self):
        _sat = H2SatLookup.get()
        if abs(self.T_l - self.T_sat) < 0.05:
            # At saturation — use fast interp instead of PropsSI
            h_l   = _sat.h_sat_l(self.P)
            u_l   = _sat.u_sat_l(self.P)
            c_pl  = _sat.cp_sat_l(self.P)
            rho_l = _sat.rho_sat_l(self.P)
        else:
            h_l   = cp.PropsSI("H",      "P", self.P, "T", self.T_l, self.fluid)
            u_l   = cp.PropsSI("U",      "P", self.P, "T", self.T_l, self.fluid)
            c_pl  = cp.PropsSI("Cpmass", "P", self.P, "T", self.T_l, self.fluid)
            rho_l = cp.PropsSI("D",      "P", self.P, "T", self.T_l, self.fluid)
        return h_l, u_l, c_pl, rho_l

    # ── External heat flows ─────────────────────────────────────

    def external_heat_flows(self, T_ambient=280.0):
        f_liq  = self.V_l / self.volume
        A_g    = self.A_surface * (1.0 - f_liq)
        A_l    = self.A_surface * f_liq
        k_ins  = self.structure.k_ins
        t_ins  = self.structure.t_ins
        Q_eg   = k_ins / t_ins * A_g * max(T_ambient - self.T_g, 0.0)
        Q_el   = k_ins / t_ins * A_l * max(T_ambient - self.T_l, 0.0)
        return Q_eg, Q_el

    # ── Thermodynamic time step (Parello Eq. 18) ────────────────

    def thermodynamic_sys(self, Q_eg, Q_el, m_dot_f, dt=0.1):
        eps = 1e-8
        self.T_l = min(self.T_l, self.T_sat - 0.01)
        self.T_g = max(self.T_g, self.T_sat + 0.2)

        h_g  = cp.PropsSI("H",      "P", self.P, "T", self.T_g, self.fluid)
        u_g  = cp.PropsSI("U",      "P", self.P, "T", self.T_g, self.fluid)
        c_vg = cp.PropsSI("Cvmass", "P", self.P, "T", self.T_g, self.fluid)

        h_l, u_l, c_pl, rho_l = self.liquid_properties()

        # Saturation props from cache — no extra PropsSI calls
        _sat    = H2SatLookup.get()
        h_sat_g = _sat.h_sat_g(self.P)
        h_sat_l = _sat.h_sat_l(self.P)
        h_vap   = h_sat_g - h_sat_l

        A_interface = self.volume ** (2.0 / 3.0)
        h_interface = 5.0
        Q_gs = h_interface * A_interface * max(self.T_g - self.T_sat, 0.0)
        Q_sl = min(
            h_interface * A_interface * max(self.T_sat - self.T_l, 0.0),
            Q_gs,
        )

        denom      = max(h_vap + c_pl * (self.T_sat - self.T_l) + (h_g - h_sat_g), eps)
        m_dot_evap = max(0.0, (Q_gs - Q_sl) / denom)

        if self.T_l >= self.T_sat - 0.01:
            m_dot_boil = max(0.0, Q_el / h_vap)
            Q_el_sens  = 0.0
        else:
            m_dot_boil = 0.0
            Q_el_sens  = Q_el

        m_dot_g = m_dot_evap + m_dot_boil
        m_dot_l = -m_dot_f - m_dot_g

        V_l_next = max(0.0, (self.m_l + m_dot_l * dt) / rho_l)
        V_g_next = max(self.volume - V_l_next, 1e-6)
        V_dot_g  = (V_g_next - self.V_g) / dt
        V_dot_l  = -V_dot_g

        m_g_safe = max(self.m_g, eps)
        m_l_safe = max(self.m_l, eps)

        dT_g_dt = (
            Q_eg - Q_gs - self.P * V_dot_g
            + m_dot_g * (h_g - u_g)
        ) / (m_g_safe * c_vg)

        dT_l_dt = (
            Q_el_sens + Q_sl + self.P * V_dot_l
            - m_dot_evap * h_vap
        ) / (m_l_safe * c_pl)

        self.m_g  = max(self.m_g + m_dot_g * dt, eps)
        self.m_l  = max(self.m_l + m_dot_l * dt, eps)
        self.T_g += dT_g_dt * dt
        self.T_l += dT_l_dt * dt
        self.T_l  = min(self.T_l, self.T_sat - 0.01)
        self.T_g  = max(self.T_g, self.T_sat + 0.2)
        self.V_g  = V_g_next
        self.V_l  = V_l_next
        self.m_H2 = self.m_g + self.m_l
        self.update_pressure()

    # ── Venting (Parello Eq. 23) ────────────────────────────────

    def handle_venting(self, P_max_limit, P_target):
        if self.P < P_max_limit:
            return 0.0
        try:
            T2      = cp.PropsSI("T", "P", P_target, "Q", 1, self.fluid)
            m_g_2   = (P_target * self.m_g * self.T_g) / (self.P * T2)
            m_vented = max(0.0, self.m_g - m_g_2)
            self.m_g  = max(m_g_2, 1e-8)
            self.T_g  = T2
            self.update_pressure()
            return m_vented
        except ValueError:
            return 0.0

    # ── Pressurisation ──────────────────────────────────────────

    def handle_pressurization(
        self,
        P_min,
        P_target,
        T_press    = 80.0,
        autogenous = True,
    ):
        """
        Raise ullage pressure to P_target when P < P_min.

        autogenous=True  – vaporise liquid into ullage (no external mass)
        autogenous=False – inject external GH₂ at T_press

        Energy balance on fixed ullage volume V_g:
            m_g2 · u_g2  =  m_g1 · u_g1  +  dm · h_press
        Combined with EOS:
            P_target = P(ρ_g2, T_g2),  ρ_g2 = m_g2 / V_g
        Solved by bisection on T_g2 (30 iterations ≈ 0.001 K accuracy).

        Returns dm [kg] of pressurant consumed.
        """
        if self.P >= P_min:
            return 0.0

        try:
            _sat      = H2SatLookup.get()
            T_sat_tgt = _sat.T_sat(P_target)

            if autogenous:
                h_press     = _sat.h_sat_g(self.P)   # interp, not PropsSI
                T_press_eff = T_sat_tgt + 0.2
            else:
                h_press     = cp.PropsSI("H", "P", P_target, "T", T_press,
                                         self.fluid)
                T_press_eff = T_press

            u_g1 = cp.PropsSI("U", "P", self.P, "T", self.T_g, self.fluid)
            m_g1 = self.m_g

            def residual(T_g2):
                rho_g2 = cp.PropsSI("D", "P", P_target, "T", T_g2, self.fluid)
                m_g2   = rho_g2 * self.V_g
                dm     = m_g2 - m_g1
                if dm < 0:
                    return -1e6
                u_g2 = cp.PropsSI("U", "P", P_target, "T", T_g2, self.fluid)
                return m_g2 * u_g2 - (m_g1 * u_g1 + dm * h_press)

            T_lo = T_sat_tgt + 0.5
            T_hi = max(T_press_eff * 2.0, T_lo + 20.0)

            r_lo = residual(T_lo)
            r_hi = residual(T_hi)

            if r_lo * r_hi > 0:
                T_g2 = T_sat_tgt + 0.5
            else:
                for _ in range(15):       # 15 iters → ~0.03 K accuracy
                    T_mid = (T_lo + T_hi) / 2.0
                    r_mid = residual(T_mid)
                    if r_mid * r_lo > 0:
                        T_lo, r_lo = T_mid, r_mid
                    else:
                        T_hi = T_mid
                    if T_hi - T_lo < 0.05:
                        break
                T_g2 = (T_lo + T_hi) / 2.0

            rho_g2 = cp.PropsSI("D", "P", P_target, "T", T_g2, self.fluid)
            m_g2   = rho_g2 * self.V_g
            dm     = max(0.0, m_g2 - m_g1)

            if dm == 0.0:
                return 0.0

            self.m_g = m_g2
            self.T_g = T_g2

            if autogenous:
                rho_l    = _sat.rho_sat_l(P_target)   # interp
                self.m_l = max(self.m_l - dm, 1e-8)
                self.V_l = self.m_l / rho_l
                self.V_g = self.volume - self.V_l
            else:
                pass  # external source: V_g unchanged, m_H2 increases

            self.m_H2 = self.m_g + self.m_l
            self.update_pressure()
            return dm

        except ValueError:
            return 0.0


# ═══════════════════════════════════════════════════════════════
#  MISSION SIMULATOR
# ═══════════════════════════════════════════════════════════════

def simulate_mission(
    diameter          : float,
    cylindrical_length: float,
    n_tanks           : int,
    mission_cfg       : MissionConfig,
    optim_cfg         : OptimConfig,
    dt                : float,
    return_history    : bool = False,
) -> Dict:
    """
    Simulate a single representative tank for the full mission.
    Fuel draw is evenly split across n_tanks.

    Returns a dict with:
      vented_total   [kg]  total H₂ vented from ALL tanks
      P_min_bar      [bar] lowest pressure seen in any tank
      m_tank_total   [kg]  combined structural + insulation mass
      grav_eff       [-]   gravimetric efficiency (final)
      boiloff_pct    [%]   vented / m_h2_total × 100
      history        dict  (only when return_history=True)
    """
    mc   = mission_cfg
    oc   = optim_cfg
    eps  = 1e-6

    m_per_tank = mc.m_h2_total / n_tanks
    m_dot_f    = m_per_tank / mc.mission_time   # constant draw [kg/s]

    try:
        tank = TankThermodynamics(
            initial_mass             = m_per_tank,
            initial_pressure         = mc.P_init,
            initial_temperature_gas  = mc.T_gas_init,
            initial_temperature_liq  = mc.T_liq_init,
            fill_level               = mc.fill_level,
            diameter                 = diameter,
            cylindrical_length       = cylindrical_length,
            delta_P_max              = oc.delta_P_max,
            delta_P_operational      = oc.delta_P_operational,
            ins_thickness            = oc.ins_thickness,
            rho_ins                  = oc.rho_ins,
            k_ins                    = oc.k_ins,
        )
    except Exception:
        return {"vented_total": 1e9, "P_min_bar": 0.0,
                "m_tank_total": 1e9, "grav_eff": 0.0,
                "boiloff_pct": 100.0}

    total_vented = 0.0
    P_min_obs    = tank.P
    n_steps      = int(mc.mission_time / dt)

    history: Optional[Dict] = (
        {"t_h": [], "P_bar": [], "T_g_K": [], "T_l_K": [],
         "m_l_kg": [], "m_g_kg": [], "vented_cum_kg": []}
        if return_history else None
    )

    for _ in range(n_steps):
        if tank.m_l < 1.0:
            break

        Q_eg, Q_el = tank.external_heat_flows(T_ambient=mc.T_ambient)
        tank.thermodynamic_sys(Q_eg, Q_el, m_dot_f, dt=dt)

        # Venting — actual H₂ loss
        m_vent        = tank.handle_venting(mc.P_max, mc.P_vent)
        total_vented += m_vent

        # Pressurisation — prevent sub-minimum pressure
        tank.handle_pressurization(mc.P_min, mc.P_refill, autogenous=True)

        P_min_obs = min(P_min_obs, tank.P)

        if return_history:
            history["t_h"].append(_ * dt / 3600.0)
            history["P_bar"].append(tank.P / 1e5)
            history["T_g_K"].append(tank.T_g)
            history["T_l_K"].append(tank.T_l)
            history["m_l_kg"].append(tank.m_l)
            history["m_g_kg"].append(tank.m_g)
            history["vented_cum_kg"].append(total_vented * n_tanks)

    vented_all    = total_vented * n_tanks
    m_tank_total  = tank.m_tank * n_tanks
    m_h2_remain   = mc.m_h2_total - vented_all
    grav_eff      = m_h2_remain / max(m_h2_remain + m_tank_total, eps)
    boiloff_pct   = 100.0 * vented_all / max(mc.m_h2_total, eps)

    result = {
        "vented_total" : vented_all,
        "P_min_bar"    : P_min_obs / 1e5,
        "m_tank_total" : m_tank_total,
        "grav_eff"     : grav_eff,
        "boiloff_pct"  : boiloff_pct,
    }
    if return_history:
        result["history"] = history
    return result


# ═══════════════════════════════════════════════════════════════
#  OPTIMISER
# ═══════════════════════════════════════════════════════════════

def _make_objective(
    n_tanks    : int,
    mission_cfg: MissionConfig,
    optim_cfg  : OptimConfig,
):
    """
    Return a scalar objective function f(x) = f([d, L]).
    Primary term: total vented H₂ [kg] (minimise).
    Penalty term: pressure violation below P_min [Pa].
    """
    mc = mission_cfg
    oc = optim_cfg

    def objective(x):
        d, L = float(x[0]), float(x[1])

        # Geometric sanity: tank must hold the required liquid volume
        r       = d / 2.0
        V_caps  = (4.0 / 3.0) * np.pi * r ** 3
        V_cyl   = np.pi * r ** 2 * L
        V_total = V_caps + V_cyl
        rho_l   = cp.PropsSI("D", "P", mc.P_init, "T", mc.T_liq_init, "Hydrogen")
        V_liq_needed = (mc.m_h2_total / n_tanks) / rho_l
        if V_total * mc.fill_level < V_liq_needed:
            return 1e9   # geometry too small for required H₂ mass

        try:
            res = simulate_mission(
                diameter           = d,
                cylindrical_length = L,
                n_tanks            = n_tanks,
                mission_cfg        = mc,
                optim_cfg          = oc,
                dt                 = oc.dt_optim,
            )
            # Penalise pressure violations (residual after pressurisation)
            P_penalty = (
                max(0.0, mc.P_min - res["P_min_bar"] * 1e5)
                * oc.penalty_per_pa
            )
            return res["vented_total"] + P_penalty
        except Exception:
            return 1e9

    return objective


def run_optimization(
    mission_cfg: MissionConfig,
    optim_cfg  : OptimConfig,
    verbose    : bool = True,
) -> List[Dict]:
    """
    Main optimisation loop.  Sweeps over N tanks, returns a list of
    result dicts sorted by ascending boil-off percentage.
    """
    # Warm the saturation cache once before any simulation starts
    _sat_cache = H2SatLookup.get()

    results = []

    for n in optim_cfg.n_tanks_range:
        t0 = time.perf_counter()

        if verbose:
            n_evals_est = optim_cfg.popsize * (optim_cfg.popsize + 2) * optim_cfg.maxiter
            steps_est   = int(mission_cfg.mission_time / optim_cfg.dt_optim)
            print(f"\n{'─'*60}")
            print(f"  N = {n} tank(s) │ "
                  f"{mission_cfg.m_h2_total/n:.1f} kg each │ "
                  f"~{n_evals_est} evals × {steps_est} steps")
            print(f"{'─'*60}")

        obj = _make_objective(n, mission_cfg, optim_cfg)

        de_result = differential_evolution(
            obj,
            bounds     = [optim_cfg.d_bounds, optim_cfg.L_bounds],
            maxiter    = optim_cfg.maxiter,
            popsize    = optim_cfg.popsize,
            tol        = optim_cfg.tol,
            seed       = optim_cfg.seed,
            polish     = True,
            workers    = 1,
            callback   = (
                lambda xk, conv: print(
                    f"    d={xk[0]:5.2f} m  L={xk[1]:5.2f} m  "
                    f"obj={obj(xk):7.2f} kg"
                ) if verbose else None
            ),
        )

        d_opt, L_opt = float(de_result.x[0]), float(de_result.x[1])

        # Re-run at finer time step for verification
        res = simulate_mission(
            diameter           = d_opt,
            cylindrical_length = L_opt,
            n_tanks            = n,
            mission_cfg        = mission_cfg,
            optim_cfg          = optim_cfg,
            dt                 = optim_cfg.dt_verify,
        )

        elapsed = time.perf_counter() - t0

        row = {
            "N"            : n,
            "d_opt_m"      : d_opt,
            "L_opt_m"      : L_opt,
            "L_d_ratio"    : L_opt / d_opt,
            "boiloff_pct"  : res["boiloff_pct"],
            "vented_kg"    : res["vented_total"],
            "P_min_bar"    : res["P_min_bar"],
            "m_tank_kg"    : res["m_tank_total"],
            "grav_eff_pct" : res["grav_eff"] * 100.0,
            "converged"    : de_result.success,
            "elapsed_s"    : elapsed,
        }
        results.append(row)

        if verbose:
            print(
                f"  → d={d_opt:.3f} m  L={L_opt:.3f} m  "
                f"L/d={L_opt/d_opt:.2f}  "
                f"boil-off={res['boiloff_pct']:.3f}%  "
                f"P_min={res['P_min_bar']:.3f} bar  "
                f"[{elapsed:.0f}s]"
            )

    results.sort(key=lambda r: r["boiloff_pct"])
    return results


# ═══════════════════════════════════════════════════════════════
#  DETAILED VERIFICATION RUN  (best candidate, fine time step)
# ═══════════════════════════════════════════════════════════════

def verify_best(
    best: Dict,
    mission_cfg: MissionConfig,
    optim_cfg  : OptimConfig,
) -> Dict:
    """
    Run the best configuration at fine time step and return history.
    Optionally saves a CSV if pandas is available.
    """
    return simulate_mission(
        diameter           = best["d_opt_m"],
        cylindrical_length = best["L_opt_m"],
        n_tanks            = best["N"],
        mission_cfg        = mission_cfg,
        optim_cfg          = optim_cfg,
        dt                 = optim_cfg.dt_verify,
        return_history     = True,
    )


# ═══════════════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════

def print_results(results: List[Dict], mission_cfg: MissionConfig) -> None:
    """Print a formatted summary table and highlight the winner."""
    W = 80
    print("\n" + "═" * W)
    print("  OPTIMISATION RESULTS".center(W))
    print("  Sorted by boil-off percentage (ascending)".center(W))
    print("═" * W)

    header = (
        f"{'N':>3}  {'d [m]':>6}  {'L [m]':>6}  {'L/d':>5}  "
        f"{'Boil-off':>9}  {'Vented [kg]':>11}  "
        f"{'P_min [bar]':>11}  {'Tank mass [kg]':>14}  {'η_grav [%]':>10}"
    )
    print(header)
    print("-" * W)

    for i, r in enumerate(results):
        tag = " ← BEST" if i == 0 else ""
        p_ok = "✓" if r["P_min_bar"] >= mission_cfg.P_min / 1e5 else "✗"
        print(
            f"{r['N']:>3}  {r['d_opt_m']:>6.3f}  {r['L_opt_m']:>6.3f}  "
            f"{r['L_d_ratio']:>5.2f}  "
            f"{r['boiloff_pct']:>8.3f}%  {r['vented_kg']:>11.2f}  "
            f"{r['P_min_bar']:>10.3f}{p_ok}  {r['m_tank_kg']:>14.1f}  "
            f"{r['grav_eff_pct']:>10.2f}{tag}"
        )

    print("═" * W)
    best = results[0]
    print(f"\n  Best configuration: N={best['N']}  "
          f"d={best['d_opt_m']:.3f} m  L={best['L_opt_m']:.3f} m")
    print(f"  Total boil-off : {best['vented_kg']:.2f} kg  "
          f"({best['boiloff_pct']:.3f} % of {mission_cfg.m_h2_total:.0f} kg)")
    print(f"  Minimum pressure: {best['P_min_bar']:.3f} bar  "
          f"(limit: {mission_cfg.P_min/1e5:.2f} bar)")
    print(f"  Gravimetric efficiency: {best['grav_eff_pct']:.2f} %\n")


def save_history_csv(history: Dict, path: str = "best_tank_history.csv") -> None:
    """Write mission history to CSV (requires no extra dependencies)."""
    keys = [k for k in history if isinstance(history[k], list)]
    rows = zip(*[history[k] for k in keys])
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(f"{v:.6g}" for v in row) + "\n")
    print(f"  History written to {path}")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LH₂ tank geometry optimiser")
    p.add_argument("--m",    type=float, default=3000.0,
                   help="Total LH₂ mass [kg]  (default: 3000)")
    p.add_argument("--t",    type=float, default=8.0,
                   help="Mission duration [h]  (default: 8)")
    p.add_argument("--nmax", type=int,   default=6,
                   help="Max number of tanks to try (default: 6)")
    p.add_argument("--dmax", type=float, default=5.0,
                   help="Max tank diameter [m]  (default: 5.0)")
    p.add_argument("--lmax", type=float, default=25.0,
                   help="Max barrel length [m]  (default: 25.0)")
    p.add_argument("--ins",  type=float, default=0.03,
                   help="Insulation thickness [m] (default: 0.03)")
    p.add_argument("--csv",  action="store_true",
                   help="Save mission history of best config to CSV")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-iteration output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mc = MissionConfig(
        m_h2_total   = args.m,
        mission_time = args.t * 3600.0,
    )
    oc = OptimConfig(
        n_tanks_range = range(1, args.nmax + 1),
        d_bounds      = (0.8, args.dmax),
        L_bounds      = (0.05, args.lmax),
        ins_thickness = args.ins,
    )

    print("\n" + "═" * 60)
    print("  LH₂ TANK GEOMETRY OPTIMISER".center(60))
    print("═" * 60)
    print(f"  Total H₂     : {mc.m_h2_total:.0f} kg")
    print(f"  Mission time : {mc.mission_time/3600:.1f} h")
    print(f"  P band       : {mc.P_min/1e5:.2f} – {mc.P_max/1e5:.2f} bar")
    print(f"  Insulation   : {oc.ins_thickness*100:.0f} mm foam  "
          f"(k={oc.k_ins} W/m/K)")
    print(f"  Searching N  : {list(oc.n_tanks_range)}")
    print("═" * 60)

    results = run_optimization(mc, oc, verbose=not args.quiet)
    print_results(results, mc)

    if args.csv and results:
        best = results[0]
        print("  Running fine verification for CSV export …")
        det  = verify_best(best, mc, oc)
        if "history" in det:
            save_history_csv(det["history"])


if __name__ == "__main__":
    main()