"""
lh2_tank.py  —  Liquid hydrogen tank thermodynamic model
=========================================================
Two-node, 0-D model (liquid phase + vapour phase) in a horizontal cylindrical
tank, following the formulation of Mastropierro et al. (2026), Eng. Proc. 133, 45.

Structure
---------
  TankStructure          – geometry and material constants (dataclass)
  VentingParameters      – pressure-relief valve settings (dataclass)
  HorizontalGeometry     – circular-segment wetted-area solver
  ThermalResistance      – composite wall R-value [m²·K/W]
  InitialTankCondition   – derives and stores the full initial state
  Thermodynamics         – heat-flow methods (wall conduction, interface convection)
  TankODE                – ODE RHS: dormancy + venting + fuel outflow + active pressure control
  TankSolver             – wraps solve_ivp; runs and stores the solution

References
----------
  Mastropierro et al. (2026) Eng. Proc. 133, 45
    https://doi.org/10.3390/engproc2026133045
  Circular segment area: standard geometric result
  Saturation properties: CoolProp PropsSI
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
import CoolProp.CoolProp as CP


# ── 1. Tank geometry and material constants ──────────────────────────────────

@dataclass
class TankStructure:
    # --- geometry ---
    tank_length:     float = 36        # cylinder length               [m]
    tank_diameter:   float = 3.8        # inner diameter                [m]

    # --- aluminium wall ---
    alu_thickness:   float = 20e-3    # wall thickness                [m]
    alu_thermal_con: float = 230.0      # thermal conductivity          [W/m/K]

    # --- insulation layer (outside the alu wall) ---
    ins_thickness:   float = 10*0.06e-3      # insulation thickness          [m]
    ins_thermal_con: float = 1.5e-4      # thermal conductivity (foam)   [W/m/K]

    # --- operating conditions ---
    T_outside:       float = 273.15+60    # ambient temperature           [K]  (60 °C)
    P_fuel:          float = 3.0e5      # initial tank pressure         [Pa]
    fill_level:      float = 0.93       # volumetric fill level (V_l/V) [-]

    # --- propellant ---
    fluid:           str   = 'Hydrogen'


# ── 1b. Venting parameters ────────────────────────────────────────────────────

@dataclass
class VentingParameters:
    """
    Pressure-relief valve settings.

    The vent opens with a smooth sigmoid response centred at P_vent.
    m_vent = m_vent_max / (1 + exp(−(P − P_vent) / dP_smooth))

    This avoids discontinuities in the ODE while correctly representing a
    threshold valve.  m_vent_max should be set close to the expected boil-off
    rate so the valve modestly bleeds pressure without depleting the ullage.
    """
    P_vent:     float = 4.0e5    # vent-open pressure threshold          [Pa]
    dP_smooth:  float = 0.05e5   # sigmoid transition half-width         [Pa]
    m_vent_max: float = 0.04     # max vent mass-flow rate               [kg/s]


# ── 1c. Active pressure control parameters ───────────────────────────────────

@dataclass
class ActivePressureControl:
    """
    Active pressure-management valve that vents between two set points.

    When P >= P_high the valve opens and vents gas until P <= P_low,
    then closes.  Modelled as a sigmoid centred on P_high for the
    open direction; the hysteresis (P_low) is tracked via the ODE
    state variable y[7] (valve_open flag, 0–1).

    In practice the sigmoid width dP_smooth is kept narrow so the
    valve acts almost as a bang-bang controller.

    Parameters
    ----------
    P_high      pressure at which valve opens      [Pa]
    P_low       pressure at which valve closes     [Pa]
    m_vent_max  peak vent mass-flow rate            [kg/s]
    dP_smooth   sigmoid transition half-width      [Pa]
    """
    P_high:     float = 3.0e5    # vent-open set point   [Pa]
    P_low:      float = 1.0e5    # vent-close set point  [Pa]
    m_vent_max: float = 0.10     # peak vent rate        [kg/s]
    dP_smooth:  float = 0.03e5   # sigmoid width         [Pa]


# ── 2. Horizontal cylinder geometry ─────────────────────────────────────────

class HorizontalGeometry:
    """
    Resolves volumes and wetted surface areas for a horizontal cylinder
    at a given volumetric fill level.

    Liquid pools at the bottom; gas occupies the top segment.

    Parameters
    ----------
    L           cylinder length          [m]
    D           inner diameter           [m]
    fill_level  volumetric fill V_l/V   [-]  (0 < fill_level < 1)

    Attributes
    ----------
    V_total, V_l, V_g       volumes              [m³]
    A_l, A_g                total wetted areas   [m²]  (wall + endcaps)
    A_interface             gas–liquid interface [m²]  (chord × L)
    h_liquid                liquid height        [m]
    alpha                   half-angle           [rad]
    """

    def __init__(self, L: float, D: float, fill_level: float):
        if not 0.0 < fill_level < 1.0:
            raise ValueError(f"fill_level must be in (0, 1), got {fill_level}")

        R = D / 2.0

        # Fill fraction is a transcendental function of the half-angle alpha:
        #   f(alpha) = (alpha − sin α cos α) / π = fill_level
        # Invert numerically.
        def _area_fraction(a: float) -> float:
            return (a - np.sin(a) * np.cos(a)) / np.pi

        alpha = brentq(
            lambda a: _area_fraction(a) - fill_level,
            1e-9, np.pi - 1e-9,
            xtol=1e-12
        )
        self.alpha = alpha

        # Cross-sectional areas
        A_circle  = np.pi * R**2
        A_l_cross = R**2 * (alpha - np.sin(alpha) * np.cos(alpha))
        A_g_cross = A_circle - A_l_cross

        # Volumes
        self.V_total = A_circle  * L
        self.V_l     = A_l_cross * L
        self.V_g     = A_g_cross * L

        # Wetted wall arc lengths
        arc_l = 2.0 * alpha * R
        arc_g = 2.0 * (np.pi - alpha) * R

        # Surface areas: cylindrical wall + two flat endcaps
        self.A_l = arc_l * L + 2.0 * A_l_cross
        self.A_g = arc_g * L + 2.0 * A_g_cross

        # Gas–liquid interface: horizontal chord × cylinder length
        chord            = 2.0 * R * np.sin(alpha)
        self.A_interface = chord * L

        # Liquid height from bottom
        self.h_liquid = R * (1.0 - np.cos(alpha))

    def summary(self) -> str:
        lines = [
            "HorizontalGeometry",
            f"  alpha         = {np.degrees(self.alpha):.2f}  deg",
            f"  h_liquid      = {self.h_liquid:.4f} m",
            f"  V_total       = {self.V_total:.4f} m³",
            f"  V_l           = {self.V_l:.4f} m³  ({self.V_l/self.V_total*100:.2f} %)",
            f"  V_g           = {self.V_g:.4f} m³  ({self.V_g/self.V_total*100:.2f} %)",
            f"  A_l (wetted)  = {self.A_l:.4f} m²",
            f"  A_g (wetted)  = {self.A_g:.4f} m²",
            f"  A_interface   = {self.A_interface:.4f} m²",
        ]
        return "\n".join(lines)


# ── 3. Composite-wall thermal resistance ────────────────────────────────────

@dataclass
class ThermalResistance:
    """
    Area-specific thermal resistance R'' [m²·K/W] for a composite wall.
    Heat flow: Q_dot = A * ΔT / R''
    """
    alu_thickness:  float   # [m]
    alu_k:          float   # [W/m/K]
    ins_thickness:  float   # [m]
    ins_k:          float   # [W/m/K]

    @property
    def R_pp(self) -> float:
        """Area-specific resistance [m²·K/W]."""
        return self.alu_thickness / self.alu_k + self.ins_thickness / self.ins_k


# ── 4. Initial tank condition ─────────────────────────────────────────────────

class InitialTankCondition:
    """
    Derives and stores the complete initial thermodynamic state.

    Both phases are initialised on the saturation curve at P_fuel
    (quality Q=0 for liquid, Q=1 for vapour).

    Attributes
    ----------
    geom        HorizontalGeometry
    T_sat       saturation temperature at P             [K]
    T_l, T_g    phase temperatures (both = T_sat at t=0)[K]
    P           tank pressure                           [Pa]
    T_amb       ambient temperature                     [K]
    rho_l/g     phase densities                         [kg/m³]
    m_l/g       phase masses                            [kg]
    u_l/g       specific internal energies              [J/kg]
    U_l/g       total internal energies                 [J]
    h_l/g       specific enthalpies                     [J/kg]
    h_fg        latent heat of vaporisation             [J/kg]
    cv_l/g      isochoric specific heats                [J/kg/K]
    cp_l        isobaric specific heat, liquid          [J/kg/K]
    dPdT_sat    saturation slope dP/dT                  [Pa/K]
    R_wall      ThermalResistance instance
    """

    def __init__(self, cfg: TankStructure | None = None):
        if cfg is None:
            cfg = TankStructure()

        self.cfg   = cfg
        self.fluid = cfg.fluid
        self.P     = cfg.P_fuel
        self.T_amb = cfg.T_outside

        # Geometry
        self.geom = HorizontalGeometry(
            L          = cfg.tank_length,
            D          = cfg.tank_diameter,
            fill_level = cfg.fill_level,
        )

        # Saturation state
        self.T_sat = CP.PropsSI('T',      'P', self.P, 'Q', 0, self.fluid)
        self.T_l   = self.T_sat
        self.T_g   = self.T_sat

        self.rho_l = CP.PropsSI('D',      'P', self.P, 'Q', 0, self.fluid)
        self.rho_g = CP.PropsSI('D',      'P', self.P, 'Q', 1, self.fluid)

        self.u_l   = CP.PropsSI('U',      'P', self.P, 'Q', 0, self.fluid)
        self.u_g   = CP.PropsSI('U',      'P', self.P, 'Q', 1, self.fluid)

        self.h_l   = CP.PropsSI('H',      'P', self.P, 'Q', 0, self.fluid)
        self.h_g   = CP.PropsSI('H',      'P', self.P, 'Q', 1, self.fluid)
        self.h_fg  = self.h_g - self.h_l

        self.cv_l  = CP.PropsSI('CVMASS', 'P', self.P, 'Q', 0, self.fluid)
        self.cv_g  = CP.PropsSI('CVMASS', 'P', self.P, 'Q', 1, self.fluid)
        self.cp_l  = CP.PropsSI('CPMASS', 'P', self.P, 'Q', 0, self.fluid)

        # Saturation slope dP/dT (Clausius–Clapeyron, numerical from CoolProp)
        self.dPdT_sat = CP.PropsSI(
            'd(P)/d(T)|Dmass', 'P', self.P, 'Q', 0, self.fluid
        )

        # Masses and total internal energies
        self.m_l = self.rho_l * self.geom.V_l
        self.m_g = self.rho_g * self.geom.V_g
        self.U_l = self.m_l * self.u_l
        self.U_g = self.m_g * self.u_g

        # Wall thermal resistance
        self.R_wall = ThermalResistance(
            alu_thickness = cfg.alu_thickness,
            alu_k         = cfg.alu_thermal_con,
            ins_thickness = cfg.ins_thickness,
            ins_k         = cfg.ins_thermal_con,
        )

    def summary(self) -> str:
        lines = [
            "═" * 52,
            "  InitialTankCondition  —  LH₂ tank",
            "═" * 52,
            "",
            self.geom.summary(),
            "",
            "Thermodynamic state",
            f"  P             = {self.P/1e5:.4f}  bar",
            f"  T_sat         = {self.T_sat:.4f}  K",
            f"  rho_l         = {self.rho_l:.4f}  kg/m³",
            f"  rho_g         = {self.rho_g:.6f}  kg/m³",
            f"  m_l           = {self.m_l:.2f}   kg",
            f"  m_g           = {self.m_g:.4f}   kg",
            f"  u_l           = {self.u_l:.2f}   J/kg",
            f"  u_g           = {self.u_g:.2f}   J/kg",
            f"  h_fg          = {self.h_fg:.2f}   J/kg",
            f"  cv_l          = {self.cv_l:.2f}   J/kg/K",
            f"  cv_g          = {self.cv_g:.2f}   J/kg/K",
            f"  dP/dT_sat     = {self.dPdT_sat:.2f}   Pa/K",
            f"  U_l           = {self.U_l/1e6:.4f}   MJ",
            f"  U_g           = {self.U_g/1e6:.4f}   MJ",
            "",
            "Wall thermal resistance",
            f"  R''_alu       = {self.cfg.alu_thickness/self.cfg.alu_thermal_con:.6f}  m²·K/W",
            f"  R''_ins       = {self.cfg.ins_thickness/self.cfg.ins_thermal_con:.6f}  m²·K/W",
            f"  R''_total     = {self.R_wall.R_pp:.6f}  m²·K/W",
            "═" * 52,
        ]
        return "\n".join(lines)


# ── 5. Heat-flow methods ──────────────────────────────────────────────────────

class Thermodynamics(InitialTankCondition):
    """
    Extends InitialTankCondition with heat-flow calculations.

    Sign convention: heat flows positive INTO the fluid.

    External wall: Q_el, Q_eg  — conduction through composite wall
    Interface:     Q_li, Q_gi  — natural convection to saturation layer
                                  (Nusselt–Rayleigh, Table 1 of paper)
    """

    # Natural convection coefficients (Table 1, Mastropierro et al. 2026)
    _Cn_l: float = 0.0135
    _n_l:  float = 0.25
    _Cn_g: float = 0.27
    _n_g:  float = 0.25
    _g:    float = 9.81

    def __init__(self, cfg: TankStructure | None = None):
        super().__init__(cfg)

    def Q_ext_l(self) -> float:
        """Conductive heat into liquid through tank wall  [W]."""
        return self.geom.A_l * (self.T_amb - self.T_l) / self.R_wall.R_pp

    def Q_ext_g(self) -> float:
        """Conductive heat into vapour through tank wall  [W]."""
        return self.geom.A_g * (self.T_amb - self.T_g) / self.R_wall.R_pp

    def Q_ext_total(self) -> float:
        """Total external heat ingress  [W]."""
        return self.Q_ext_l() + self.Q_ext_g()

    def heat_flow_summary(self) -> str:
        lines = [
            "Heat flow summary (initial state)",
            f"  Q_ext_l        = {self.Q_ext_l():+.2f}  W  (wall → liquid)",
            f"  Q_ext_g        = {self.Q_ext_g():+.2f}  W  (wall → vapour)",
            f"  Q_ext_total    = {self.Q_ext_total():+.2f}  W",
        ]
        return "\n".join(lines)


# ── 6. CoolProp helpers (saturation-safe) ─────────────────────────────────────

_SAT_EPS = 5e-3  # K — tolerance band around the saturation curve

def _safe_liquid(prop: str, T: float, P: float, fluid: str) -> float:
    """
    Liquid-phase property, safe at the saturation boundary.
    Falls back to Q=0 call when T is within _SAT_EPS of T_sat.
    """
    T_sat = CP.PropsSI('T', 'P', P, 'Q', 0, fluid)
    if T <= T_sat + _SAT_EPS:
        return CP.PropsSI(prop, 'P', P, 'Q', 0, fluid)
    return CP.PropsSI(prop, 'T', T, 'P', P, fluid)


def _safe_gas(prop: str, T: float, P: float, fluid: str) -> float:
    """
    Gas-phase property, safe at the saturation boundary.
    Falls back to Q=1 call when T is within _SAT_EPS of T_sat.
    """
    T_sat = CP.PropsSI('T', 'P', P, 'Q', 0, fluid)
    if T <= T_sat + _SAT_EPS:
        return CP.PropsSI(prop, 'P', P, 'Q', 1, fluid)
    return CP.PropsSI(prop, 'T', T, 'P', P, fluid)


# ── 7. ODE right-hand side ────────────────────────────────────────────────────

class TankODE:
    """
    ODE right-hand side for the two-node LH₂ tank model.

    State vector  y = [m_l, m_g, T_l, T_g, P, m_vented, m_fuel_out]
                       0     1    2    3    4  5          6

    Mass equations
    --------------
      dm_l/dt = −m_evap − m_fuel_out                                (1)
      dm_g/dt =  m_evap − m_vent                                    (2)

    Energy equations  (Mastropierro et al. 2026, Eqs 3–4)
    -------------------------------------------------------
      dT_l/dt = (Q_el − Q_li − P·dVl/dt
                 − m_evap·h_l_sat − m_fuel_out·h_l − dm_l·u_l)
                / (m_l · cp_l)                                      (3)

      dT_g/dt = (Q_eg − Q_gi + P·dVg/dt
                 + m_evap·(h_g_sat − u_g) − m_vent·(h_g − u_g))
                / (m_g · cv_g)                                      (4)

    Boil-off  (Eq 5)
    ----------------
      m_evap = (Q_li + Q_gi) / h_fg                                 (5)

    Venting model
    -------------
    Smooth sigmoid centred at P_vent:
      m_vent = m_vent_max / (1 + exp(−(P − P_vent) / dP_smooth))
    Clamped to available gas to prevent m_g → negative.
    Vented gas leaves at its current enthalpy h_g(T_g, P).

    Fuel outflow
    ------------
    Constant liquid draw rate m_fuel_out [kg/s] (engine consumption).
    Liquid exits at its current enthalpy h_l(T_l, P).
    Clamped to available liquid mass.

    Volume coupling (incompressible liquid, fixed V_total)
    ------------------------------------------------------
      dVl/dt = dm_l/dt / rho_l   →   dVg/dt = −dVl/dt

    Pressure closure
    ----------------
      P = P_sat(T_l)   →   dP/dt = (dP/dT)_sat · dT_l/dt

    Integral trackers in the state vector (no ODE feedback)
    -------------------------------------------------------
      y[5] = m_vented       cumulative vented gas mass      [kg]
      y[6] = m_fuel_out     cumulative liquid to engine     [kg]
      y[7] = valve_state    active-control valve open frac  [0–1]
                            (smooth integrator for hysteresis)

    Parameters
    ----------
    ic            InitialTankCondition
    vent          VentingParameters  (safety relief, opens at P_vent)
    apc           ActivePressureControl  (None = disabled)
    m_fuel_out    liquid fuel draw rate to engine       [kg/s]  (default 0)
    L_char        characteristic length for Ra number  [m]
    """

    _Cn_l, _n_l = 0.0135, 0.25
    _Cn_g, _n_g = 0.27,   0.25
    _g = 9.81

    def __init__(
        self,
        ic:          InitialTankCondition,
        vent:        VentingParameters | None = None,
        apc:         ActivePressureControl | None = None,
        m_fuel_out:  float = 0.0,
        L_char:      float | None = None,
    ):
        self.fluid      = ic.fluid
        self.V_tot      = ic.geom.V_total
        self.A_l        = ic.geom.A_l
        self.A_g        = ic.geom.A_g
        self.A_int      = ic.geom.A_interface
        self.R_pp       = ic.R_wall.R_pp
        self.T_amb      = ic.T_amb
        self.L_char     = L_char if L_char is not None else ic.cfg.tank_diameter
        self.vent       = vent if vent is not None else VentingParameters()
        self.apc        = apc
        self.m_fuel_out = m_fuel_out

    # ── Natural convection helper ────────────────────────────────────────────

    def _h_nc(
        self, beta: float, nu: float, alpha_th: float,
        k: float, dT: float, Cn: float, n: float
    ) -> float:
        """Convection coefficient from Nu = Cn · Ra^n  [W/m²/K]."""
        if abs(dT) < 1e-12 or nu < 1e-20 or alpha_th < 1e-20:
            return 0.0
        Ra = self._g * beta * abs(dT) * self.L_char**3 / (nu * alpha_th)
        return Cn * Ra**n * k / self.L_char

    # ── ODE RHS ──────────────────────────────────────────────────────────────

    def __call__(self, t: float, y: list[float]) -> list[float]:
        """Returns dy/dt = [dm_l, dm_g, dT_l, dT_g, dP, dm_vented, dm_fuel_out, dvalve]."""

        m_l, m_g, T_l, T_g, P, _mv, _mf, valve_state = y

        # Guards
        m_l         = max(m_l, 1.0)
        m_g         = max(m_g, 1e-3)
        P           = max(P,   1.01e5)
        valve_state = np.clip(valve_state, 0.0, 1.0)

        fluid = self.fluid

        # ── Saturation temperature ──
        T_sat = CP.PropsSI('T', 'P', P, 'Q', 0, fluid)

        # ── Liquid thermodynamic properties ──
        rho_l  = _safe_liquid('D',      T_l, P, fluid)
        u_l    = _safe_liquid('U',      T_l, P, fluid)
        h_l    = _safe_liquid('H',      T_l, P, fluid)
        cp_l   = _safe_liquid('CPMASS', T_l, P, fluid)
        k_l    = _safe_liquid('CONDUCTIVITY', T_l, P, fluid)
        beta_l = _safe_liquid('ISOBARIC_EXPANSION_COEFFICIENT', T_l, P, fluid)
        nu_l   = _safe_liquid('VISCOSITY', T_l, P, fluid) / rho_l
        al_l   = k_l / (rho_l * cp_l)

        # ── Gas thermodynamic properties ──
        u_g    = _safe_gas('U',      T_g, P, fluid)
        h_g    = _safe_gas('H',      T_g, P, fluid)   # vent exit enthalpy
        cv_g   = _safe_gas('CVMASS', T_g, P, fluid)
        k_g    = _safe_gas('CONDUCTIVITY', T_g, P, fluid)
        beta_g = _safe_gas('ISOBARIC_EXPANSION_COEFFICIENT', T_g, P, fluid)
        rho_g  = _safe_gas('D',      T_g, P, fluid)
        nu_g   = _safe_gas('VISCOSITY', T_g, P, fluid) / rho_g
        al_g   = k_g / (rho_g * cv_g)

        # ── Saturation enthalpies at interface ──
        h_l_sat = CP.PropsSI('H', 'P', P, 'Q', 0, fluid)
        h_g_sat = CP.PropsSI('H', 'P', P, 'Q', 1, fluid)
        h_fg    = h_g_sat - h_l_sat

        # ── External wall heat (conduction) ──
        Q_el = self.A_l * (self.T_amb - T_l) / self.R_pp
        Q_eg = self.A_g * (self.T_amb - T_g) / self.R_pp

        # ── Interface natural convection  (Eq 5 prerequisites) ──
        # Q_li, Q_gi positive = heat flows from phase toward the interface
        h_li = self._h_nc(beta_l, nu_l, al_l, k_l, T_l - T_sat, self._Cn_l, self._n_l)
        h_gi = self._h_nc(beta_g, nu_g, al_g, k_g, T_g - T_sat, self._Cn_g, self._n_g)
        Q_li = h_li * self.A_int * (T_l - T_sat)
        Q_gi = h_gi * self.A_int * (T_g - T_sat)

        # ── Boil-off rate  (Eq 5) ──
        m_evap = (Q_li + Q_gi) / h_fg

        # ── Safety relief vent: sigmoid valve, capped to available gas ──
        vp = self.vent
        exponent_v = np.clip(-(P - vp.P_vent) / vp.dP_smooth, -500, 500)
        m_vent_sig = vp.m_vent_max / (1.0 + np.exp(exponent_v))
        m_vent     = min(m_vent_sig, max(m_g - 1e-3, 0.0) * 5.0)

        # ── Active pressure control (APC) ──
        # Hysteretic bang-bang controller modelled as a smooth ODE:
        #   valve_state ∈ [0, 1] — smoothly tracks open/closed
        #   Opens  (→ 1) when P >= P_high  via sigmoid
        #   Closes (→ 0) when P <= P_low   via sigmoid
        # The valve_state integrator has a fast time constant (tau = 10 s)
        # so it approximates bang-bang while staying ODE-smooth.
        m_apc = 0.0
        dvalve = 0.0
        if self.apc is not None:
            ap = self.apc
            tau_valve = 10.0  # s — valve response time constant
            # target_open  → 1 when P ≥ P_high (open)
            # target_close → 1 when P ≤ P_low  (close)
            exponent_h = np.clip(-(P - ap.P_high) / ap.dP_smooth, -500, 500)
            exponent_l = np.clip( (P - ap.P_low)  / ap.dP_smooth, -500, 500)
            target_open  = 1.0 / (1.0 + np.exp(exponent_h))
            target_close = 1.0 / (1.0 + np.exp(exponent_l))
            # Open dominates when P > P_high; close dominates when P < P_low
            valve_target = np.clip(target_open - target_close, 0.0, 1.0)
            dvalve = (valve_target - valve_state) / tau_valve
            m_apc_raw = valve_state * ap.m_vent_max
            m_apc = min(m_apc_raw, max(m_g - 1e-3, 0.0) * 5.0)

        # Total vent flow (safety + APC, both remove gas)
        m_vent_total = m_vent + m_apc

        # ── Fuel outflow: liquid to engine, capped to available liquid ──
        m_out = min(self.m_fuel_out, max(m_l - 1.0, 0.0) * 5.0)

        # ── Net mass rates ──
        dm_l_net = -m_evap - m_out
        dm_g_net =  m_evap - m_vent_total

        # ── PV work: volume coupling via incompressible-liquid assumption ──
        Vl_dot  = dm_l_net / rho_l
        PVl_dot = P * Vl_dot
        PVg_dot = -PVl_dot

        # ── Liquid energy equation  (Eq 3) ──
        dT_l = (
            Q_el
            - Q_li
            - PVl_dot
            - m_evap * h_l_sat
            - m_out  * h_l
            - dm_l_net * u_l
        ) / (m_l * cp_l)

        # ── Gas energy equation  (Eq 4) ──
        # Both safety vent and APC vent gas at current h_g
        dT_g = (
            Q_eg
            - Q_gi
            + PVg_dot
            + m_evap       * (h_g_sat - u_g)
            - m_vent_total * (h_g     - u_g)
        ) / (m_g * cv_g)

        # ── Pressure closure: P = P_sat(T_l) ──
        dPdT = CP.PropsSI('d(P)/d(T)|Dmass', 'P', P, 'Q', 0, fluid)
        dP   = dPdT * dT_l

        # ── Integral trackers ──
        d_m_vented   = m_vent_total
        d_m_fuel_out = m_out

        return [dm_l_net, dm_g_net, dT_l, dT_g, dP, d_m_vented, d_m_fuel_out, dvalve]


# ── 8. Solver ─────────────────────────────────────────────────────────────────

class TankSolver:
    """
    Integrates TankODE over a given time span.

    Uses LSODA, which automatically switches between stiff and non-stiff
    methods.  This is appropriate here because CoolProp calls introduce
    implicit stiffness that RK45 handles poorly near the vent threshold.

    Parameters
    ----------
    cfg           TankStructure
    vent          VentingParameters  (None → defaults: P_vent=4 bar)
    m_fuel_out    liquid draw rate to engine                [kg/s]  (default 0)
    t_end         integration end time                      [s]
    max_step      maximum ODE step                          [s]
    L_char        characteristic length for Ra number       [m]

    Attributes after solve()
    ------------------------
    t             time array                        [s]     shape (N,)
    m_l, m_g      phase masses                      [kg]    shape (N,)
    T_l, T_g      phase temperatures                [K]     shape (N,)
    P             tank pressure                     [Pa]    shape (N,)
    m_vented      cumulative vented gas mass         [kg]    shape (N,)
    m_fuel_drawn  cumulative liquid to engine        [kg]    shape (N,)
    sol           raw scipy OdeResult
    """

    def __init__(
        self,
        cfg:         TankStructure         | None = None,
        vent:        VentingParameters      | None = None,
        apc:         ActivePressureControl  | None = None,
        m_fuel_out:  float = 0.0,
        t_end:       float = 3600.0,
        max_step:    float = 120.0,
        L_char:      float | None = None,
    ):
        self.ic       = InitialTankCondition(cfg)
        self.ode      = TankODE(self.ic, vent=vent, apc=apc,
                                m_fuel_out=m_fuel_out, L_char=L_char)
        self.t_end    = t_end
        self.max_step = max_step

    def solve(self) -> TankSolver:
        """Run the ODE integration. Returns self for chaining."""
        ic  = self.ic
        apc = self.ode.apc
        if apc is not None:
            exp0 = np.clip(-(ic.P - apc.P_high) / apc.dP_smooth, -500, 500)
            valve_init = 1.0 / (1.0 + np.exp(exp0))
        else:
            valve_init = 0.0
        y0 = [ic.m_l, ic.m_g, ic.T_l, ic.T_g, ic.P, 0.0, 0.0, valve_init]

        self.sol = solve_ivp(
            self.ode,
            [0.0, self.t_end],
            y0,
            method   = 'LSODA',
            max_step = self.max_step,
            rtol     = 1e-4,
            atol     = 1e-5,
        )

        if not self.sol.success:
            raise RuntimeError(f"ODE solver failed: {self.sol.message}")

        self.t            = self.sol.t
        self.m_l          = self.sol.y[0]
        self.m_g          = self.sol.y[1]
        self.T_l          = self.sol.y[2]
        self.T_g          = self.sol.y[3]
        self.P            = self.sol.y[4]
        self.m_vented     = self.sol.y[5]
        self.m_fuel_drawn = self.sol.y[6]
        self.valve_state  = self.sol.y[7]
        return self

    def summary(self) -> str:
        t_h = self.t[-1] / 3600.0
        ic  = self.ic
        lines = [
            "TankSolver result",
            f"  duration        = {t_h:.2f} h  ({len(self.t)} steps)",
            f"  m_l:   {ic.m_l:.2f} → {self.m_l[-1]:.2f} kg   (Δ = {self.m_l[-1]-ic.m_l:+.2f} kg)",
            f"  m_g:   {ic.m_g:.4f} → {self.m_g[-1]:.4f} kg   (Δ = {self.m_g[-1]-ic.m_g:+.4f} kg)",
            f"  T_l:   {ic.T_l:.4f} → {self.T_l[-1]:.4f} K    (Δ = {self.T_l[-1]-ic.T_l:+.4f} K)",
            f"  T_g:   {ic.T_g:.4f} → {self.T_g[-1]:.4f} K    (Δ = {self.T_g[-1]-ic.T_g:+.4f} K)",
            f"  P:     {ic.P/1e5:.4f} → {self.P[-1]/1e5:.4f} bar  (Δ = {(self.P[-1]-ic.P)/1e5:+.4f} bar)",
            f"  m_vented:      {self.m_vented[-1]:.4f} kg",
            f"  m_fuel_drawn:  {self.m_fuel_drawn[-1]:.4f} kg",
        ]
        return "\n".join(lines)


# ── 9. Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg  = TankStructure()
    vent = VentingParameters(P_vent=4.0e5, dP_smooth=0.05e5, m_vent_max=0.04)
    apc  = ActivePressureControl(P_high=3.0e5, P_low=1.0e5, m_vent_max=0.10)

    thermo = Thermodynamics(cfg)
    print(thermo.summary())
    print()
    print(thermo.heat_flow_summary())
    print()

    print("Case 1: dormancy (1 h, no outflow, no venting)")
    s1 = TankSolver(cfg, vent=VentingParameters(P_vent=99e5),
                    m_fuel_out=0.0, t_end=3600).solve()
    print(s1.summary()); print()

    print("Case 2: dormancy + safety vent (1 h)")
    s2 = TankSolver(cfg, vent=vent, m_fuel_out=0.0, t_end=3600).solve()
    print(s2.summary()); print()

    print("Case 3: fuel draw 0.05 kg/s + safety vent (1 h)")
    s3 = TankSolver(cfg, vent=vent, m_fuel_out=0.05, t_end=3600).solve()
    print(s3.summary()); print()

    print("Case 4: active pressure control P_high=3 bar → P_low=1 bar (1 h)")
    s4 = TankSolver(cfg, vent=vent, apc=apc, m_fuel_out=0.0, t_end=3600).solve()
    print(s4.summary())