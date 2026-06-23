
# --- restructured-project import bootstrap ---
from pathlib import Path as _DSE_Path
import sys as _DSE_sys
_DSE_ROOT = next((p for p in _DSE_Path(__file__).resolve().parents if (p / "src").exists() and (p / "data").exists()), None)
if _DSE_ROOT is not None:
    for _DSE_p in [
        _DSE_ROOT / "src",
        _DSE_ROOT / "src" / "common",
        _DSE_ROOT / "src" / "aerodynamics" / "drag",
        _DSE_ROOT / "src" / "propulsion",
        _DSE_ROOT / "src" / "propulsion" / "engine",
        _DSE_ROOT / "src" / "thermal",
        _DSE_ROOT / "src" / "sizing",
        _DSE_ROOT / "src" / "tanks",
        _DSE_ROOT / "src" / "environment",
        _DSE_ROOT / "src" / "trade_offs",
        _DSE_ROOT / "external",
        _DSE_ROOT / "external" / "pycycle_examples",
    ]:
        if _DSE_p.exists() and str(_DSE_p) not in _DSE_sys.path:
            _DSE_sys.path.insert(0, str(_DSE_p))
# --- end bootstrap ---
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp

# ============================================================
# LH2 TANK THERMODYNAMIC + STRUCTURAL MODEL
# Based on: Parello et al., "Design and Integration of a
# Liquid Hydrogen Tank on an Aircraft", AIAA SciTech 2024
# ============================================================


class TankStructure:
    """
    Structural sizing of a cylindrical LH2 tank with
    ellipsoidal bulkheads, per Parello et al. [17] and
    CS-25 regulation requirements.

    The tank is assumed to be a cylinder of diameter `d`
    with two 2:1 ellipsoidal end caps (semi-major = d/2,
    semi-minor = d/4).  The barrel section carries skin +
    stringers + frames; the bulkheads carry skin only.
    """

    def __init__(
        self,
        # ── geometry ──────────────────────────────────────
        diameter,               # inner diameter  [m]
        cylindrical_length,     # barrel length   [m]
        # ── pressure loads ────────────────────────────────
        delta_P_max,            # max differential pressure [Pa]
        delta_P_operational,    # operational diff pressure [Pa]
        # ── aluminum properties (Table 1 defaults) ────────
        rho_al      = 2825,     # density            [kg/m³]
        nu_al       = 0.33,     # Poisson ratio      [-]
        E_al        = 73.8e9,   # Young modulus      [Pa]
        sigma_yield = 240e6,    # elastic limit      [Pa]
        sigma_ult   = 420e6,    # ultimate strength  [Pa]
        k_al        = 230,      # thermal cond.      [W/m/K]
        weld_eff    = 0.85,     # weld efficiency    [-]
        # ── insulation (Table 1 defaults: PU closed foam) ─
        ins_thickness  = 0.00,  # insulation thickness [m]
        rho_ins        = 33,    # foam density         [kg/m³]
        k_ins          = 0.1, # thermal cond.        [W/m/K]
    ):
        self.d        = diameter
        self.L_cyl    = cylindrical_length
        self.dP_max   = delta_P_max
        self.dP_op    = delta_P_operational
        self.rho_al   = rho_al
        self.E        = E_al
        self.sigma_y  = sigma_yield
        self.sigma_u  = sigma_ult
        self.nu       = nu_al
        self.k_al     = k_al
        self.ew       = weld_eff
        self.t_ins    = ins_thickness
        self.rho_ins  = rho_ins
        self.k_ins    = k_ins

        # Allowable stress: lower of yield/1.5 and ult/2.0
        self.sigma_a = min(sigma_yield / 1.5,
                           sigma_ult   / 2.0)

        # ── run sizing ────────────────────────────────────
        self._size()

    # --------------------------------------------------------
    # INTERNAL VOLUME (inner geometry)
    # --------------------------------------------------------

    @property
    def r(self):
        return self.d / 2.0

    @property
    def V_total(self):
        """
        Total inner volume = cylinder + two 2:1 ellipsoidal caps.
        Volume of one 2:1 oblate cap  = (2π/3) r² (r/2)
                                       = π r³ / 3
        """
        V_cyl  = np.pi * self.r**2 * self.L_cyl
        V_caps = 2.0 * (np.pi * self.r**3 / 3.0)
        self._V_total = V_cyl + V_caps
        return self._V_total


    # --------------------------------------------------------
    # STRUCTURAL SIZING  (CS-25 approach, Parello et al. [17])
    # --------------------------------------------------------

    def _size(self):
        r  = self.r
        dP = self.dP_max          # design to maximum pressure
        sa = self.sigma_a

        # ── Barrel skin ──────────────────────────────────
        # Hoop stress governs a thin-walled cylinder:
        #   σ_hoop = ΔP·r / t   →  t = ΔP·r / (σ_a · e_w)
        self.t_skin = dP * r / (sa * self.ew)
        A_skin_cyl  = np.pi * self.d * self.L_cyl   # outer ≈ inner
        self.m_skin = rho_factor = self.rho_al * A_skin_cyl * self.t_skin

        # ── Ellipsoidal bulkheads ─────────────────────────
        # For a 2:1 ellipsoid (a = r, b = r/2):
        # Max stress occurs at the equator (parallel stress):
        #   σ_β = ΔP·a² / (2·b·t) · (2·a²(b²-a²)+a⁴) / sqrt(...)
        # For 2:1 (b=a/2):  σ_β = ΔP·r / (2·t)  (equator governs)
        # => t_bulkhead = ΔP·r / (2·σ_a·e_w)
        self.t_bulkhead = dP * r / (2.0 * sa * self.ew)

        # Surface area of one 2:1 oblate spheroid cap (approx)
        # Using numerical approximation for oblate spheroid:
        # A ≈ 2π·a² · (1 + (1-e²)/e · arctanh(e))  where e² = 1-(b/a)²
        a_cap = r
        b_cap = r / 2.0
        e2    = 1.0 - (b_cap / a_cap)**2
        e     = np.sqrt(e2)
        A_cap = 2.0 * np.pi * a_cap**2 * (1.0 + (1.0 - e2) / e * np.arctanh(e))
        self.m_bulkheads = self.rho_al * 2.0 * A_cap * self.t_bulkhead

        # ── Stringers (longitudinal stiffeners) ───────────
        # Sized to prevent column buckling at ultimate load.
        # Number from skin-buckling criterion (empirical for
        # aircraft fuselage): n_str ≈ π·d / (20·t_skin)
        # Minimum 4 stringers.
        n_str_float = np.pi * self.d / (20.0 * self.t_skin)
        self.n_stringers = max(4, int(np.ceil(n_str_float)))

        # Stringer cross-section: sized to carry compressive
        # load at ultimate = 2·dP_op.  Simple L-section:
        # A_str = dP_op · r / (n_str · σ_a)
        A_str = (2.0 * self.dP_op * np.pi * r**2) / (
                 self.n_stringers * self.sigma_u)
        self.A_stringer = A_str
        self.m_stringers = (self.rho_al * A_str
                            * self.n_stringers * self.L_cyl)

        # ── Frames (circumferential stiffeners) ───────────
        # Pitch b_frame set so skin does not buckle at dP_op.
        # Using Euler buckling for curved panel:
        # b_frame ≈ π · t_skin · sqrt(E / (3·dP_op/r))
        b_frame = np.pi * self.t_skin * np.sqrt(
                  self.E / (3.0 * self.dP_op / r))
        b_frame = max(b_frame, 0.3)         # minimum 0.3 m pitch
        self.n_frames = max(2, int(np.ceil(self.L_cyl / b_frame)))

        # Frame cross-section (ring hoop load):
        # N_hoop = dP·r  →  A_frame = N_hoop·b_frame / σ_a
        A_frame = (self.dP_max * r * b_frame) / self.sigma_u
        self.A_frame = A_frame
        self.m_frames = (self.rho_al * A_frame
                         * self.n_frames * np.pi * self.d)

        # ── Insulation ────────────────────────────────────
        # Covers entire outer surface (cylinder + caps)
        A_ins = (A_skin_cyl + 2.0 * A_cap)
        self.m_insulation = self.rho_ins * A_ins * self.t_ins

        # ── Total tank structural mass ────────────────────
        self.m_tank = (
            self.m_skin
            + self.m_bulkheads
            + self.m_stringers
            + self.m_frames
            + self.m_insulation
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def summary(self):
        print("\n========== STRUCTURAL SIZING ==========")
        print(f"  Tank diameter      : {self.d:.3f} m")
        print(f"  Cylinder length    : {self.L_cyl:.3f} m")
        print(f"  Inner volume       : {self.V_total:.3f} m³")
        print(f"  Skin thickness     : {self.t_skin*1e3:.2f} mm")
        print(f"  Bulkhead thickness : {self.t_bulkhead*1e3:.2f} mm")
        print(f"  Stringers          : {self.n_stringers}  "
              f"(A={self.A_stringer*1e6:.1f} mm²)")
        print(f"  Frames             : {self.n_frames}  "
              f"(A={self.A_frame*1e6:.1f} mm²)")
        print(f"  Insulation mass    : {self.m_insulation:.1f} kg")
        print(f"  Total tank mass    : {self.m_tank:.1f} kg")
        print(f"  ─ skin             : {self.m_skin:.1f} kg")
        print(f"  ─ bulkheads        : {self.m_bulkheads:.1f} kg")
        print(f"  ─ stringers        : {self.m_stringers:.1f} kg")
        print(f"  ─ frames           : {self.m_frames:.1f} kg")
        print("=======================================")


# ============================================================
# THERMODYNAMIC MODEL
# ============================================================

class TankThermodynamics:
    """
    Two-phase (ullage + bulk liquid) thermodynamic model of
    a cryogenic LH2 tank.

    Geometry is derived from the initial hydrogen mass:
      1. rho_l at (P, T_liq)  →  V_liquid = m_H2 / rho_l
         (assuming the tank is entirely liquid at t=0, then
          the ullage fraction sets the actual fill level)
      2. V_total = V_liquid / fill_level
      3. A TankStructure is built whose inner volume equals
         V_total, and its mass is returned as m_tank.

    The model follows the 3-control-volume ODE system of
    Parello et al. (Eq. 18), simplified to 2 CV (ullage +
    liquid) with interface heat/mass exchange.
    """

    def __init__(
        self,
        initial_mass,            # total LH2 mass  [kg]
        initial_pressure,        # tank pressure   [Pa]
        initial_temperature_gas, # ullage T        [K]
        initial_temperature_liq, # liquid T        [K]
        fill_level = 0.93,       # V_liq / V_total [-]
        # ── tank geometry (auto-sized if None) ────────────
        diameter          = 3.5,  # inner diameter [m]; auto if None
        cylindrical_length= 20,  # barrel length  [m]; auto if None
        # ── structural / insulation parameters ───────────
        delta_P_max        = 10.0e5,  # max diff pressure  [Pa]
        delta_P_operational= 3.0e5,  # op  diff pressure  [Pa]
        ins_thickness      = 0.005,   # insulation thick.  [m]
        rho_ins            = 33,     # foam density       [kg/m³]
        k_ins              = 0.026,  # foam cond.         [W/m/K]
    ):
        self.fluid     = "Hydrogen"
        self.fill_level = fill_level

        # ── Initial thermodynamic state ───────────────────
        self.P   = initial_pressure
        self.T_l = initial_temperature_liq
        self.T_g = initial_temperature_gas

        # ── Liquid density at initial conditions ──────────
        rho_l_init = cp.PropsSI("D", "P", self.P,
                                "T", self.T_l, self.fluid)

        # ── Derive volumes from mass + fill level ─────────
        # V_liq = m_H2 / rho_l  (assume all LH2 is liquid
        # at t=0; ullage mass is negligible for sizing)
        V_liq_init  = initial_mass / rho_l_init
        V_total     = V_liq_init / fill_level
        self.volume = V_total                   # kept for reference

        self.V_l = V_liq_init
        self.V_g = V_total - V_liq_init         # ullage volume

        # ── Auto-size tank geometry if not provided ───────
        if diameter is None:
            # Choose a 2:1 length-to-diameter ratio for the
            # cylinder (reasonable for aircraft tanks).
            # V_total = π(d/2)²·L_cyl + 2·π(d/2)³/3
            # With L_cyl = 2d  →  V = π·d³/4 + π·d³/12
            #                     V = π·d³·(3+1)/12 = π·d³/3
            diameter = (3.0 * V_total / np.pi) ** (1.0 / 3.0)
        self.d = diameter

        if cylindrical_length is None:
            # Back-calculate from V_total and d
            r = diameter / 2.0
            V_caps = 2.0 * np.pi * r**3 / 3.0
            cylindrical_length = max(
                0.05,
                (V_total - V_caps) / (np.pi * r**2)
            )
        self.L_cyl = cylindrical_length

        # ── Structural model ──────────────────────────────
        self.structure = TankStructure(
            diameter           = self.d,
            cylindrical_length = self.L_cyl,
            delta_P_max        = delta_P_max,
            delta_P_operational= delta_P_operational,
            ins_thickness      = ins_thickness,
            rho_ins            = rho_ins,
            k_ins              = k_ins,
        )
        self.m_tank = self.structure.m_tank

        # ── Initial mass distribution ─────────────────────
        # Ullage gas density at (P, T_g)
        rho_g_init = cp.PropsSI("D", "P", self.P,
                                "T", self.T_g, self.fluid)

        self.m_l = rho_l_init * self.V_l
        self.m_g = rho_g_init * self.V_g
        self.m_H2 = self.m_l + self.m_g

        # ── Saturation temperature at initial pressure ────
        self.T_sat = cp.PropsSI("T", "P", self.P,
                                "Q", 0, self.fluid)

        # Enforce physical temperature bounds at init
        self.T_l = min(self.T_l, self.T_sat - 0.01)
        self.T_g = max(self.T_g, self.T_sat + 0.2)

        # ── Insulation thermal conductance (UA) ───────────
        # Simple 1-D conduction through flat insulation;
        # area = outer surface of cylinder + caps.
        r = self.d / 2.0
        A_cyl  = np.pi * self.d * self.L_cyl
        a_cap  = r
        b_cap  = r / 2.0
        e2     = 1.0 - (b_cap / a_cap)**2
        e_     = np.sqrt(e2)
        A_caps = 2.0 * 2.0 * np.pi * a_cap**2 * (
                 1.0 + (1.0 - e2) / e_ * np.arctanh(e_))
        self.A_surface = A_cyl + A_caps
        self.UA_ins = k_ins * self.A_surface / ins_thickness

    # ========================================================
    # GRAVIMETRIC EFFICIENCY
    # ========================================================

    def gravimetric_efficiency(self):
        return self.m_H2 / (self.m_H2 + self.m_tank)

    def gravimetric_efficiency_adjusted(self, m_boiloff):
        return (self.m_H2 - m_boiloff) / (self.m_H2 + self.m_tank)

    # ========================================================
    # PRESSURE UPDATE
    # ========================================================

    def update_pressure(self):
        rho_g = max(self.m_g / self.V_g, 1e-8)

        self.P = cp.PropsSI(
            "P", "D", rho_g, "T", self.T_g, self.fluid
        )
        P_min = 1.5e5  # Pa  (~0.15 bar, safely above triple point)
        if self.P < P_min:
            # either vent in reverse (pressurisation) or raise a warning
            self.P = P_min

        self.T_sat = cp.PropsSI(
            "T", "P", self.P, "Q", 0, self.fluid
        )

        self.T_l = min(self.T_l, self.T_sat - 0.01)
        self.T_g = max(self.T_g, self.T_sat + 0.2)

    # ========================================================
    # SAFE LIQUID PROPERTY EVALUATION
    # ========================================================

    def liquid_properties(self):
        if abs(self.T_l - self.T_sat) < 0.05:
            h_l   = cp.PropsSI("H",      "P", self.P, "Q", 0, self.fluid)
            u_l   = cp.PropsSI("U",      "P", self.P, "Q", 0, self.fluid)
            c_pl  = cp.PropsSI("Cpmass", "P", self.P, "Q", 0, self.fluid)
            rho_l = cp.PropsSI("D",      "P", self.P, "Q", 0, self.fluid)
        else:
            h_l   = cp.PropsSI("H",      "P", self.P, "T", self.T_l, self.fluid)
            u_l   = cp.PropsSI("U",      "P", self.P, "T", self.T_l, self.fluid)
            c_pl  = cp.PropsSI("Cpmass", "P", self.P, "T", self.T_l, self.fluid)
            rho_l = cp.PropsSI("D",      "P", self.P, "T", self.T_l, self.fluid)
        return h_l, u_l, c_pl, rho_l

    # ========================================================
    # HEAT FLOWS  (thermal circuit, Parello Eq. 2–16)
    # ========================================================

    def external_heat_flows(self, T_ambient=280.0, mach=0.0,
                            altitude=0.0, exposed_fraction=0.5):
        """
        Compute Q_eg and Q_el: heat entering ullage and liquid
        sections from the exterior through the insulation.

        Uses a simple 1-D resistance model:
          Q_total = UA_ins · (T_ambient – T_H2)

        The surface is split proportionally to fill level:
          fraction exposed to ullage  = 1 – fill_level
          fraction exposed to liquid  = fill_level
        """
        f_liq = self.V_l / self.volume   # current liquid fraction

        # Effective hydrogen temperatures for each section
        T_H2_g = self.T_g
        T_H2_l = self.T_l

        # Area split
        A_g = self.A_surface * (1.0 - f_liq)
        A_l = self.A_surface * f_liq

        k_ins = self.structure.k_ins
        t_ins = self.structure.t_ins

        Q_eg = k_ins / t_ins * A_g * (T_ambient - T_H2_g)
        Q_el = k_ins / t_ins * A_l * (T_ambient - T_H2_l)

        return max(Q_eg, 0.0), max(Q_el, 0.0)

    # ========================================================
    # THERMODYNAMIC UPDATE  (Parello Eq. 18 simplified)
    # ========================================================

    def thermodynamic_sys(
        self,
        Q_eg,         # heat into ullage   [W]
        Q_el,         # heat into liquid   [W]
        m_dot_f,      # fuel draw (from liquid) [kg/s]
        dt = 0.1      # time step          [s]
    ):
        eps = 1e-8

        # ── enforce temperature limits ────────────────────
        self.T_l = min(self.T_l, self.T_sat - 0.01)
        self.T_g = max(self.T_g, self.T_sat + 0.2)

        # ── gas properties ────────────────────────────────
        h_g  = cp.PropsSI("H",      "P", self.P, "T", self.T_g, self.fluid)
        u_g  = cp.PropsSI("U",      "P", self.P, "T", self.T_g, self.fluid)
        c_vg = cp.PropsSI("Cvmass", "P", self.P, "T", self.T_g, self.fluid)

        # ── liquid properties ─────────────────────────────
        h_l, u_l, c_pl, rho_l = self.liquid_properties()

        # ── saturation properties ─────────────────────────
        h_sat_g = cp.PropsSI("H", "P", self.P, "Q", 1, self.fluid)
        h_sat_l = cp.PropsSI("H", "P", self.P, "Q", 0, self.fluid)
        h_vap   = h_sat_g - h_sat_l

        # ── interface heat transfer (Parello §III.A.2) ────
        # Q_gs: ullage → interface  (drives evaporation)
        # modelled as convection over the liquid surface area
        A_interface = (self.volume) ** (2.0 / 3.0)  # proxy for surface
        h_interface = 5.0                             # [W/m²/K]

        Q_gs = h_interface * A_interface * max(self.T_g - self.T_sat, 0.0)

        # Q_sl: interface → bulk liquid  (constrained ≤ Q_gs)
        Q_sl = min(
            h_interface * A_interface * max(self.T_sat - self.T_l, 0.0),
            Q_gs
        )

        # ── boil-off mass flow (Eq. 18 numerator) ─────────
        denom = (h_vap
                 + c_pl * (self.T_sat - self.T_l)
                 + (h_g - h_sat_g))
        denom = max(denom, eps)

        m_dot_evap = max(0.0, (Q_gs - Q_sl) / denom)

        # Wall boil-off: if T_l ≥ T_sat all Q_el goes to boil
        if self.T_l >= self.T_sat - 0.01:
            m_dot_boil = max(0.0, Q_el / h_vap)
            Q_el_sens  = 0.0          # no sensible heating of liquid
        else:
            m_dot_boil = 0.0
            Q_el_sens  = Q_el

        m_dot_g = m_dot_evap + m_dot_boil     # mass into ullage
        m_dot_l = -m_dot_f - m_dot_g          # mass lost from liquid

        # ── volume update ─────────────────────────────────
        V_l_next = max(0.0,
                       (self.m_l + m_dot_l * dt) / rho_l)
        V_g_next = max(self.volume - V_l_next, 1e-6)

        V_dot_g = (V_g_next - self.V_g) / dt
        V_dot_l = -V_dot_g

        # ── energy equations (Eq. 18) ─────────────────────
        m_g_safe = max(self.m_g, eps)
        m_l_safe = max(self.m_l, eps)

        # Ullage temperature rate (Eq. 18, T_g dot):
        #   m_g·c_vg·dT_g/dt = Q_eg - Q_gs - P·V_dot_g
        #                     + m_dot_g·h_g - m_dot_g·u_g
        dT_g_dt = (
            Q_eg
            - Q_gs
            - self.P * V_dot_g
            + m_dot_g * h_g
            - m_dot_g * u_g
        ) / (m_g_safe * c_vg)

        # Liquid temperature rate (Eq. 18, T_l dot):
        #   m_l·c_pl·dT_l/dt = Q_el + Q_sl + P·V_dot_l
        #                     - m_dot_g·h_l + m_dot_f·h_l
        #                     - m_dot_l·u_l
        dT_l_dt = (
            Q_el_sens
            + Q_sl
            + self.P * V_dot_l
            - m_dot_evap * h_vap
        ) / (m_l_safe * c_pl)

        # ── update masses ─────────────────────────────────
        self.m_g = max(self.m_g + m_dot_g * dt, eps)
        self.m_l = max(self.m_l + m_dot_l * dt, eps)

        # ── update temperatures ───────────────────────────
        self.T_g += dT_g_dt * dt
        self.T_l += dT_l_dt * dt

        self.T_l = min(self.T_l, self.T_sat - 0.01)
        self.T_g = max(self.T_g, self.T_sat + 0.2)

        # ── update volumes ────────────────────────────────
        self.V_g = V_g_next
        self.V_l = V_l_next
        self.m_H2 = self.m_g + self.m_l

        # ── update pressure ───────────────────────────────
        self.update_pressure()

    # ========================================================
    # VENTING  (Parello Eq. 23)
    # ========================================================

    def handle_venting(self, P_max_limit, P_target):
        """
        Instantaneous venting: reduce ullage pressure from
        P_max_limit to P_target.

        m_vent = m_g,1 - m_g,2 = P2·m_g,1·T_g,1 / (P1·T2)
        where T2 = T_sat(P_target).
        """
       

        try:
            T2 = cp.PropsSI("T", "P", P_target, "Q", 1, self.fluid)

            m_g_2 = (P_target * self.m_g * self.T_g) / (self.P * T2)
            m_vented = max(0.0, self.m_g - m_g_2)

            self.m_g = max(m_g_2, 1e-8)
            self.T_g = T2      # ullage cools to saturation after vent
            self.update_pressure()
            return m_vented

        except ValueError:
            return 0.0


# ============================================================
# MAIN SIMULATION
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # INITIAL CONDITIONS  (matching Table 1 of Parello et al.)
    # ========================================================

    m_H2_initial         = 11420  # hydrogen mass per tank [kg]
                                     # (11420 kg total / 2 tanks)
    p_initial            = 5.0e5    # Pa  (2 bar absolute)
    t_liq_initial        = 20.0     # K
    t_gas_initial        = 25.0     # K
    fill_level_initial   = 0.97     # 97%  (Table 1)

    # ── Pressure limits (Table 1) ───────────────────────────
    P_min_diff = 0.5e5   # min pressure differential [Pa]
    P_max_diff = 1.0e5   # max pressure differential [Pa]

    # At sea level ambient ≈ 1.01325e5 Pa
    P_ambient_sl = 1.01325e5
    P_vent_trigger = P_ambient_sl + P_max_diff   # ~2.01 bar abs
    P_vent_target  = P_ambient_sl + P_min_diff   # ~1.51 bar abs

    # ── Build tank ──────────────────────────────────────────
    tank = TankThermodynamics(
        initial_mass            = m_H2_initial,
        initial_pressure        = p_initial,
        initial_temperature_gas = t_gas_initial,
        initial_temperature_liq = t_liq_initial,
        fill_level              = fill_level_initial,
        delta_P_max             = P_max_diff,
        delta_P_operational     = P_min_diff,
        ins_thickness           = 0.03,
        rho_ins                 = 33,
        k_ins                   = 0.026,
    )

    tank.structure.summary()

    print("\n============= INITIAL STATE =============")
    print(f"  Tank diameter      : {tank.d:.3f} m")
    print(f"  Cylinder length    : {tank.L_cyl:.3f} m")
    print(f"  Inner volume       : {tank.volume:.3f} m³")
    print(f"  Liquid volume      : {tank.V_l:.3f} m³")
    print(f"  Ullage volume      : {tank.V_g:.3f} m³")
    print(f"  Liquid mass        : {tank.m_l:.2f} kg")
    print(f"  Gas mass           : {tank.m_g:.4f} kg")
    print(f"  Total H2           : {tank.m_H2:.2f} kg")
    print(f"  Tank struct. mass  : {tank.m_tank:.1f} kg")
    print(f"  Pressure           : {tank.P/1e5:.3f} bar")
    print(f"  T_liquid           : {tank.T_l:.3f} K")
    print(f"  T_gas              : {tank.T_g:.3f} K")
    print(f"  T_sat              : {tank.T_sat:.3f} K")
    print(f"  Gravimetric eff.   : {100*tank.gravimetric_efficiency():.2f}%")

    # ========================================================
    # SIMULATION SETTINGS
    # ========================================================

    dt           = 10                   # time step [s]
    sim_duration = 10800              # total time [s]  (3 hours)
    time_steps   = np.arange(0, sim_duration, dt)

    T_ambient       = 125.65             # ambient temperature [K] (Table 1)
    m_dot_engine    = 0      # engine fuel draw  [kg/s] based on the 143kg/s air flow for 2 engines assuming 0.5 equivalence ratio
                                        

    # ========================================================
    # HISTORY STORAGE
    # ========================================================

    history = {
        "time":     [],
        "pressure": [],
        "T_g":      [],
        "T_l":      [],
        "T_sat":    [],
        "m_g":      [],
        "m_l":      [],
        "m_total":  [],
        "vented":   [],
        "Q_eg":     [],
        "Q_el":     [],
    }

    cumulative_vented = 0.0

    # ========================================================
    # MAIN TIME LOOP
    # ========================================================

    for t in time_steps:

        # ── venting check ────────────────────────────────
        vented_mass = tank.handle_venting(P_vent_trigger, P_vent_target)
        cumulative_vented += vented_mass

        # ── heat flows from insulation model ─────────────
        Q_eg, Q_el = tank.external_heat_flows(T_ambient=T_ambient)

        # ── thermodynamic step ───────────────────────────
        try:
            tank.thermodynamic_sys(
                Q_eg    = Q_eg,
                Q_el    = Q_el,
                m_dot_f = m_dot_engine,
                dt      = dt,
            )
        except ValueError as e:
            print(f"\nSimulation failed at t = {t:.1f} s")
            print(f"  P = {tank.P/1e5:.3f} bar")
            print(f"  T_g = {tank.T_g:.3f} K,  T_l = {tank.T_l:.3f} K")
            raise e

        # ── store ─────────────────────────────────────────
        history["time"].append(t)
        history["pressure"].append(tank.P / 1e5)
        history["T_g"].append(tank.T_g)
        history["T_l"].append(tank.T_l)
        history["T_sat"].append(tank.T_sat)
        history["m_g"].append(tank.m_g)
        history["m_l"].append(tank.m_l)
        history["m_total"].append(tank.m_H2)
        history["vented"].append(cumulative_vented)
        history["Q_eg"].append(Q_eg)
        history["Q_el"].append(Q_el)

    # ========================================================
    # FINAL STATE
    # ========================================================

    m_boiloff = m_H2_initial - tank.m_H2
    print("\n============== FINAL STATE ==============")
    print(f"  Final pressure     : {tank.P/1e5:.3f} bar")
    print(f"  Final liquid mass  : {tank.m_l:.2f} kg")
    print(f"  Final gas mass     : {tank.m_g:.4f} kg")
    print(f"  Total H2 remaining : {tank.m_H2:.2f} kg")
    print(f"  Total vented       : {cumulative_vented:.3f} kg")
    print(f"  Boil-off loss      : {m_boiloff:.3f} kg  "
          f"({100*m_boiloff/m_H2_initial:.2f}%)")
    print(f"  Adj. grav. eff.    : "
          f"{100*tank.gravimetric_efficiency_adjusted(cumulative_vented):.2f}%")

    # ========================================================
    # PLOTS
    # ========================================================

    t_arr = np.array(history["time"]) / 3600.0  # → hours

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # ── Pressure ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_arr, history["pressure"], label="Tank Pressure")
    ax.axhline(P_vent_trigger / 1e5, linestyle="--",
               color="red",  label="Vent Trigger")
    ax.axhline(P_vent_target  / 1e5, linestyle="--",
               color="green", label="Vent Target")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title("Tank Pressure")
    ax.grid(True)
    ax.legend()

    # ── Temperatures ──────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t_arr, history["T_g"],   label="Gas (Ullage)")
    ax.plot(t_arr, history["T_l"],   label="Liquid")
    ax.plot(t_arr, history["T_sat"], label="Saturation", linestyle=":")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Phase Temperatures")
    ax.grid(True)
    ax.legend()

    # ── Hydrogen masses ────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(t_arr, history["m_l"],     label="Liquid Mass")
    ax.plot(t_arr, history["m_g"],     label="Gas Mass")
    ax.plot(t_arr, history["m_total"], label="Total H2", linestyle="--")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Mass [kg]")
    ax.set_title("Hydrogen Inventory")
    ax.grid(True)
    ax.legend()

    # ── Cumulative venting ────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t_arr, history["vented"], color="orange", label="Cumulative Vented")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Mass [kg]")
    ax.set_title("Vented Hydrogen")
    ax.grid(True)
    ax.legend()

    # ── Heat flows ────────────────────────────────────────
    ax = axes[2, 0]
    ax.plot(t_arr, history["Q_eg"], label="Q_eg (to ullage)")
    ax.plot(t_arr, history["Q_el"], label="Q_el (to liquid)")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Heat Flow [W]")
    ax.set_title("External Heat Flows through Insulation")
    ax.grid(True)
    ax.legend()

    # ── Tank weight breakdown (pie) ───────────────────────
    ax = axes[2, 1]
    s  = tank.structure
    labels = ["Skin", "Bulkheads", "Stringers", "Frames", "Insulation"]
    sizes  = [s.m_skin, s.m_bulkheads, s.m_stringers,
              s.m_frames, s.m_insulation]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Tank Structure Mass Breakdown\n"
                 f"(Total = {s.m_tank:.0f} kg)")

    plt.suptitle(
        "LH2 Cryogenic Tank Simulation\n"
        f"m_H2={m_H2_initial:.0f} kg | "
        f"V={tank.volume:.1f} m³ | "
        f"d={tank.d:.2f} m | "
        f"L_cyl={tank.L_cyl:.2f} m",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()