

MACH              = 5.0     # Free-stream Mach number  [-]
ALTITUDE_KM       = 31.0    # Cruise altitude           [km]
WALL_TEMP_K       = 373.15   # Assumed wall temperature  [K]
IS_TURBULENT      = False    # True = turbulent BL (conservative for TPS sizing)
AOA_DEG           = 5.0     # Angle of attack           [degrees]

# Geometry
NOSE_RADIUS_M     = 0.010   # Nose tip radius of curvature   [m]
NOSE_WEDGE_DEG    = 10.0    # Nose cone half‑angle            [degrees]
COCKPIT_LENGTH_M  = 2.5     # Axial length of nose cone (tip to cockpit) [m]
FUSELAGE_LENGTH_M = 20.0    # Total fuselage length           [m]
WING_CHORD_M      = 5.0     # Wing chord (root)               [m]
WING_LE_RADIUS_M  = 0.005   # Wing leading-edge radius        [m]
WING_WEDGE_DEG    = 6.0     # Wing lower-surface wedge angle at AoA=0  [degrees]

# Analysis resolution
NUM_POINTS        = 200     # Points along each surface

# Output
SAVE_FIGURES      = False    # False → interactive display; True → save PNGs
FIGURE_PREFIX     = "heat_flux"



import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


# ── Atmospheric model ────────────────────────────────────────────────────────

def us_standard_atmosphere(altitude_km: float) -> Tuple[float, float, float]:
    """
    1976 U.S. Standard Atmosphere up to 86 km.

    Returns
    -------
    T  : static temperature  [K]
    P  : static pressure     [Pa]
    rho: density             [kg/m³]
    """
    R_air  = 287.05
    g0     = 9.80665
    gamma  = 1.4

    layers = [
        (0.0,   288.150, -6.5,   101325.0),
        (11.0,  216.650,  0.0,    22632.1),
        (20.0,  216.650,  1.0,     5474.89),
        (32.0,  228.650,  2.8,      868.019),
        (47.0,  270.650,  0.0,      110.906),
        (51.0,  270.650, -2.8,       66.9389),
        (71.0,  214.650, -2.0,        3.95642),
        (86.0,  186.870,  0.0,        0.3734),
    ]

    h = altitude_km
    if h < 0 or h > 86:
        raise ValueError(f"Altitude {h} km is outside the 0–86 km model range.")

    for i in range(len(layers) - 1):
        h_base, T_base, lapse, P_base = layers[i]
        h_top = layers[i + 1][0]
        if h <= h_top:
            dh = (h - h_base) * 1000.0
            if abs(lapse) < 1e-9:
                T = T_base
                P = P_base * math.exp(-g0 * dh / (R_air * T_base))
            else:
                lapse_m = lapse / 1000.0
                T = T_base + lapse_m * dh
                P = P_base * (T / T_base) ** (-g0 / (lapse_m * R_air))
            rho = P / (R_air * T)
            return T, P, rho

    h_base, T_base, lapse, P_base = layers[-1]
    dh = (h - h_base) * 1000.0
    T = T_base
    P = P_base * math.exp(-g0 * dh / (R_air * T_base))
    rho = P / (R_air * T)
    return T, P, rho


# ── Gas property helpers ──────────────────────────────────────────────────────

GAMMA   = 1.4
R_AIR   = 287.05
PR      = 0.71
CP      = GAMMA * R_AIR / (GAMMA - 1)

MU_REF  = 1.716e-5
T_REF   = 273.15
S_SUTH  = 110.4


def viscosity(T: float) -> float:
    """Dynamic viscosity via Sutherland's law [Pa·s]."""
    return MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)


def thermal_conductivity(T: float) -> float:
    """Thermal conductivity from Prandtl number [W/(m·K)]."""
    return viscosity(T) * CP / PR


# ── Shock and expansion relations ────────────────────────────────────────────

def normal_shock(M1: float) -> Tuple[float, float, float]:
    """Normal shock: returns (M2, T2/T1, rho2/rho1)."""
    g = GAMMA
    M1sq = M1 ** 2
    M2sq = (1.0 + (g - 1) / 2.0 * M1sq) / (g * M1sq - (g - 1) / 2.0)
    T2_T1 = (1.0 + 2.0 * g / (g + 1) * (M1sq - 1.0)) * \
            (2.0 + (g - 1) * M1sq) / ((g + 1) * M1sq)
    rho2_rho1 = (g + 1) * M1sq / (2.0 + (g - 1) * M1sq)
    return math.sqrt(M2sq), T2_T1, rho2_rho1


def prandtl_meyer_nu(M: float) -> float:
    """Prandtl–Meyer function ν(M) [radians]."""
    g = GAMMA
    gp = (g + 1) / (g - 1)
    return (math.sqrt(gp) * math.atan(math.sqrt((M ** 2 - 1) / gp))
            - math.atan(math.sqrt(M ** 2 - 1)))


def prandtl_meyer_mach(nu_target: float,
                        M_lo: float = 1.001,
                        M_hi: float = 50.0) -> float:
    """
    Invert the Prandtl–Meyer function: given ν [rad], find M.
    Uses bisection on ν(M) – ν_target = 0.
    """
    for _ in range(100):
        M_mid = 0.5 * (M_lo + M_hi)
        f = prandtl_meyer_nu(M_mid) - nu_target
        if abs(f) < 1e-9:
            break
        if f < 0:
            M_lo = M_mid
        else:
            M_hi = M_mid
    return M_mid


def isentropic_temperature(M1: float, M2: float, T1: float) -> float:
    """T2 for isentropic flow between M1 and M2."""
    g = GAMMA
    T2 = T1 * (1.0 + (g - 1) / 2.0 * M1 ** 2) / (1.0 + (g - 1) / 2.0 * M2 ** 2)
    return T2


def isentropic_density(M1: float, M2: float, rho1: float) -> float:
    """rho2 for isentropic flow between M1 and M2."""
    g = GAMMA
    rho2 = rho1 * ((1.0 + (g - 1) / 2.0 * M1 ** 2) /
                   (1.0 + (g - 1) / 2.0 * M2 ** 2)) ** (1.0 / (g - 1))
    return rho2


def oblique_shock_angle(M1: float, wedge_deg: float) -> float:
    """Weak-shock wave angle β [degrees] for flow deflection θ."""
    theta = math.radians(wedge_deg)
    g     = GAMMA
    mu    = math.asin(1.0 / M1)

    def theta_of_beta(beta):
        sb2 = math.sin(beta) ** 2
        cot = math.cos(beta) / math.sin(beta)
        num = 2.0 * cot * (M1 ** 2 * sb2 - 1.0)
        den = M1 ** 2 * (g + math.cos(2.0 * beta)) + 2.0
        val = max(-1.0, min(1.0, num / den))
        return math.atan(val)

    blo, bhi = mu + 1e-4, math.pi / 2.0 - 1e-4
    betas  = [blo + (bhi - blo) * i / 60 for i in range(61)]
    thetas = [theta_of_beta(b) for b in betas]
    idx    = thetas.index(max(thetas))
    theta_max   = thetas[idx]
    beta_at_max = betas[idx]

    if theta > theta_max:
        raise ValueError(
            f"Deflection {wedge_deg:.1f}° exceeds detachment limit "
            f"({math.degrees(theta_max):.1f}°) for M={M1:.2f}."
        )

    b_lo = mu + 1e-4
    b_hi = beta_at_max
    f_lo = theta_of_beta(b_lo) - theta
    f_hi = theta_of_beta(b_hi) - theta

    if f_lo * f_hi > 0:
        return math.degrees(beta_at_max)

    for _ in range(200):
        b_mid = 0.5 * (b_lo + b_hi)
        f_mid = theta_of_beta(b_mid) - theta
        if abs(f_mid) < 1e-9 or (b_hi - b_lo) < 1e-10:
            break
        if f_lo * f_mid < 0:
            b_hi = b_mid
            f_hi = f_mid
        else:
            b_lo = b_mid
            f_lo = f_mid

    return math.degrees(b_mid)


def oblique_shock_conditions(M1: float, rho1: float, T1: float, P1: float,
                              wedge_deg: float) -> Tuple[float, float, float, float]:
    """
    Post-oblique-shock edge conditions.
    Returns (M2, T2 [K], rho2 [kg/m³], u2 [m/s]).
    """
    beta_deg = oblique_shock_angle(M1, wedge_deg)
    beta     = math.radians(beta_deg)
    Mn1      = M1 * math.sin(beta)

    Mn2, T2_T1, rho2_rho1 = normal_shock(Mn1)

    T2   = T1 * T2_T1
    rho2 = rho1 * rho2_rho1

    theta = math.radians(wedge_deg)
    M2    = Mn2 / math.sin(beta - theta)
    a2    = math.sqrt(GAMMA * R_AIR * T2)
    u2    = M2 * a2
    return M2, T2, rho2, u2


def expansion_conditions(M1: float, rho1: float, T1: float, P1: float,
                          turn_deg: float) -> Tuple[float, float, float, float]:
    """
    Prandtl–Meyer expansion through a convex corner of angle turn_deg.

    The flow turns away from the surface (expansion), so pressure and
    temperature drop.

    Parameters
    ----------
    M1       : upstream Mach number
    rho1, T1, P1 : upstream conditions
    turn_deg : expansion angle (positive = turning away from surface) [degrees]

    Returns
    -------
    M2, T2, rho2, u2
    """
    if turn_deg <= 0:
        # No expansion; return unchanged
        a1 = math.sqrt(GAMMA * R_AIR * T1)
        return M1, T1, rho1, M1 * a1

    nu1 = prandtl_meyer_nu(M1)
    nu2 = nu1 + math.radians(turn_deg)

    # Cap at ν_max = ν(∞) ≈ π(√((γ+1)/(γ-1)) - 1)/2
    g = GAMMA
    nu_max = (math.pi / 2.0) * (math.sqrt((g + 1) / (g - 1)) - 1.0)
    if nu2 > nu_max:
        nu2 = nu_max

    M2   = prandtl_meyer_mach(nu2)
    T2   = isentropic_temperature(M1, M2, T1)
    rho2 = isentropic_density(M1, M2, rho1)
    a2   = math.sqrt(GAMMA * R_AIR * T2)
    u2   = M2 * a2
    return M2, T2, rho2, u2


# ── Reference enthalpy method ─────────────────────────────────────────────────

def reference_conditions(T_e: float, u_e: float, P_e: float,
                          T_w: float, turbulent: bool
                          ) -> Tuple[float, float, float]:
    """
    Eckert's reference enthalpy method.
    Returns (T_star, rho_star, mu_star).
    """
    r    = PR ** (1.0 / 3.0) if turbulent else math.sqrt(PR)
    T_aw = T_e * (1.0 + r * (GAMMA - 1.0) / 2.0 * (u_e / math.sqrt(GAMMA * R_AIR * T_e)) ** 2)

    h_w    = CP * T_w
    h_e    = CP * T_e
    h_aw   = CP * T_aw
    h_star = h_w + 0.50 * (h_e - h_w) + 0.22 * (h_aw - h_w)
    T_star = h_star / CP

    rho_star = P_e / (R_AIR * T_star)
    mu_star  = viscosity(T_star)
    return T_star, rho_star, mu_star


# ── Stagnation-point heat flux (Fay–Riddell) ─────────────────────────────────

def stagnation_heat_flux(M_inf: float, rho_inf: float, T_inf: float,
                          P_inf: float, V_inf: float,
                          nose_radius: float, T_w: float) -> Tuple[float, Dict]:
    """Fay–Riddell stagnation heat flux at the nose hemisphere."""
    M2, T2_T1, rho2_rho1 = normal_shock(M_inf)
    T_e   = T_inf * T2_T1
    rho_e = rho_inf * rho2_rho1
    P2_P1 = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (M_inf ** 2 - 1.0)
    P_e   = P_inf * P2_P1

    q_inf_dyn = 0.5 * rho_inf * V_inf ** 2
    du_dx     = (1.0 / nose_radius) * math.sqrt(2.0 * q_inf_dyn / rho_e)

    mu_e = viscosity(T_e)
    h0   = CP * T_inf + 0.5 * V_inf ** 2
    h_w  = CP * T_w

    q_stag = (0.57 / PR ** 0.6) * math.sqrt(rho_e * mu_e * du_dx) * max(h0 - h_w, 0.0)

    diag = dict(T_e=T_e, rho_e=rho_e, P_e=P_e,
                du_dx=du_dx, h0=h0, h_w=h_w, mu_e=mu_e)
    return q_stag, diag


# ── Leading-edge stagnation heat flux ─────────────────────────────────────────

def leading_edge_heat_flux(M_inf: float, rho_inf: float, T_inf: float,
                            P_inf: float, V_inf: float,
                            le_radius: float, T_w: float,
                            sweep_deg: float = 0.0) -> float:
    """Swept leading-edge stagnation heat flux (Fay–Riddell, cylinder)."""
    if le_radius <= 0:
        warnings.warn("LE radius ≤ 0; using 1 mm lower bound.", stacklevel=2)
        le_radius = 1e-3

    cos_sw = math.cos(math.radians(sweep_deg))
    V_n    = V_inf * cos_sw
    M_n    = M_inf * cos_sw
    if M_n < 1.001:
        M_n = 1.001

    M2n, T2_T1, rho2_rho1 = normal_shock(M_n)
    T_e   = T_inf * T2_T1
    rho_e = rho_inf * rho2_rho1
    P2_P1 = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (M_n ** 2 - 1.0)
    P_e   = P_inf * P2_P1

    q_inf_dyn_n = 0.5 * rho_inf * V_n ** 2
    du_dx       = (1.0 / le_radius) * math.sqrt(2.0 * q_inf_dyn_n / rho_e)

    mu_e = viscosity(T_e)
    h0   = CP * T_inf + 0.5 * V_inf ** 2
    h_w  = CP * T_w

    q_le = (0.57 / PR ** 0.6) * math.sqrt(rho_e * mu_e * du_dx) * max(h0 - h_w, 0.0)
    return q_le


# ── Flat-plate boundary-layer heat flux ───────────────────────────────────────

def flat_plate_heat_flux(x: float,
                          T_e: float, u_e: float, P_e: float,
                          T_w: float, turbulent: bool) -> float:
    """
    Local convective heat flux on a flat plate at streamwise position x.

    Laminar  : Nu_x = 0.332 · Re_x*^{1/2} · Pr*^{1/3}
    Turbulent: Nu_x = 0.0296 · Re_x*^{4/5} · Pr*^{1/3}
    """
    if x <= 0:
        return 0.0

    T_star, rho_star, mu_star = reference_conditions(T_e, u_e, P_e, T_w, turbulent)
    k_star = thermal_conductivity(T_star)

    Re_x = rho_star * u_e * x / mu_star
    if Re_x < 1.0:
        return 0.0

    if turbulent:
        Nu_x = 0.0296 * Re_x ** 0.8 * PR ** (1.0 / 3.0)
    else:
        Nu_x = 0.332 * math.sqrt(Re_x) * PR ** (1.0 / 3.0)

    h_conv = Nu_x * k_star / x

    r    = PR ** (1.0 / 3.0) if turbulent else math.sqrt(PR)
    M_e  = u_e / math.sqrt(GAMMA * R_AIR * T_e)
    T_aw = T_e * (1.0 + r * (GAMMA - 1.0) / 2.0 * M_e ** 2)

    return max(h_conv * max(T_aw - T_w, 0.0), 0.0)


# ── Edge condition factory ─────────────────────────────────────────────────────

def edge_conditions_from_deflection(M_inf: float, rho_inf: float,
                                     T_inf: float, P_inf: float,
                                     V_inf: float,
                                     deflection_deg: float) -> Dict:
    """
    Return boundary-layer edge conditions for a surface inclined at
    deflection_deg relative to the free-stream direction.

    deflection_deg > 0  →  compression (oblique shock)
    deflection_deg < 0  →  expansion   (Prandtl–Meyer)
    deflection_deg = 0  →  free stream (no wave)
    """
    if abs(deflection_deg) < 0.01:
        # Effectively zero: free-stream
        return dict(T_e=T_inf, u_e=V_inf, P_e=P_inf)

    if deflection_deg > 0:
        # Compression: oblique shock
        try:
            M2, T2, rho2, u2 = oblique_shock_conditions(
                M_inf, rho_inf, T_inf, P_inf, deflection_deg)
            P2 = rho2 * R_AIR * T2
            return dict(T_e=T2, u_e=u2, P_e=P2)
        except ValueError as exc:
            warnings.warn(f"Oblique shock failed ({exc}). "
                           "Falling back to free-stream.", stacklevel=2)
            return dict(T_e=T_inf, u_e=V_inf, P_e=P_inf)
    else:
        # Expansion: Prandtl–Meyer
        M2, T2, rho2, u2 = expansion_conditions(
            M_inf, rho_inf, T_inf, P_inf, abs(deflection_deg))
        P2 = rho2 * R_AIR * T2
        return dict(T_e=T2, u_e=u2, P_e=P2)

def _apply_deflection(self, inflow: Dict, deflection_deg: float) -> Dict:
    """Apply a compression (deflection>0) or expansion (deflection<0) to given inflow."""
    M_inf_in = inflow['u_e'] / math.sqrt(GAMMA * R_AIR * inflow['T_e'])
    rho_in = inflow['P_e'] / (R_AIR * inflow['T_e'])
    return edge_conditions_from_deflection(
        M_inf_in, rho_in, inflow['T_e'], inflow['P_e'], inflow['u_e'],
        deflection_deg)
# ── Full vehicle analysis ─────────────────────────────────────────────────────

class HypersonicVehicleAnalysis:
    """
    Top-level orchestrator for vehicle thermal analysis.

    Geometry (UPDATED)
    ------------------
    Nose cone (blunted) from tip to COCKPIT_LENGTH_M:
        - Edge conditions from oblique shock (nose_half_angle ± AoA)
        - Heat flux along cone surface using flat-plate theory.
    Aft fuselage (COCKPIT_LENGTH_M to FUSELAGE_LENGTH_M):
        - Edge conditions = post-shock properties from nose cone base.
        - Conservative: no shoulder expansion (overpredicts → safe).
        - Flat-plate theory with those edge conditions.
    """

    def _apply_deflection(self, inflow: Dict, deflection_deg: float) -> Dict:
        """Apply a compression (deflection>0) or expansion (deflection<0) to given inflow."""
        M_inf_in = inflow['u_e'] / math.sqrt(GAMMA * R_AIR * inflow['T_e'])
        rho_in = inflow['P_e'] / (R_AIR * inflow['T_e'])
        return edge_conditions_from_deflection(
            M_inf_in, rho_in, inflow['T_e'], inflow['P_e'], inflow['u_e'],
            deflection_deg)

    def _shoulder_expansion(self, post_shock_conditions: Dict, wedge_deg: float) -> Dict:
        """
        Expand the post‑shock flow around the shoulder to become axial.
        The flow turns by -wedge_deg (from cone‑parallel to body‑parallel).
        Returns new edge conditions (T_e, u_e, P_e) after expansion.
        """
        # Upstream (post‑shock) conditions
        T1 = post_shock_conditions['T_e']
        P1 = post_shock_conditions['P_e']
        u1 = post_shock_conditions['u_e']
        M1 = u1 / math.sqrt(GAMMA * R_AIR * T1)
        rho1 = P1 / (R_AIR * T1)

        # Expansion turn = -wedge_deg (positive because expansion direction)
        turn_deg = wedge_deg  # because we are turning away from the cone surface

        M2, T2, rho2, u2 = expansion_conditions(M1, rho1, T1, P1, turn_deg)
        P2 = rho2 * R_AIR * T2
        return dict(T_e=T2, u_e=u2, P_e=P2)

    def __init__(self,
                 mach: float,
                 altitude_km: float,
                 wall_temp_k: float,
                 is_turbulent: bool,
                 aoa_deg: float,
                 nose_radius_m: float,
                 nose_wedge_deg: float,
                 cockpit_length_m: float,
                 wing_le_radius_m: float,
                 wing_wedge_deg: float):

        self.M_inf          = mach
        self.altitude_km    = altitude_km
        self.T_w            = wall_temp_k
        self.is_turbulent   = is_turbulent
        self.aoa            = aoa_deg
        self.nose_radius    = nose_radius_m
        self.nose_wedge     = nose_wedge_deg
        self.cockpit_len    = cockpit_length_m
        self.wing_le_radius = wing_le_radius_m
        self.wing_wedge     = wing_wedge_deg

        # Atmospheric conditions
        self.T_inf, self.P_inf, self.rho_inf = us_standard_atmosphere(altitude_km)
        a_inf      = math.sqrt(GAMMA * R_AIR * self.T_inf)
        self.V_inf = mach * a_inf

        # ── Nose cone edge conditions (oblique shock / expansion) ──────────
        # Effective deflection = nose half‑angle ± AoA
        lower_deflection = self.nose_wedge + self.aoa   # windward
        upper_deflection = self.nose_wedge - self.aoa   # leeward (could be negative)

        self._nose_lower = edge_conditions_from_deflection(
            self.M_inf, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
            deflection_deg=lower_deflection)
        self._nose_upper = edge_conditions_from_deflection(
            self.M_inf, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
            deflection_deg=upper_deflection)

        # Post‑nose flow (what the wings actually see as freestream)
        self._post_nose_lower = self._nose_lower.copy()
        self._post_nose_upper = self._nose_upper.copy()

        # Wing edge conditions: apply wing wedge ± AoA ON TOP of post‑nose flow
        lower_wing_deflection = self.wing_wedge + self.aoa
        upper_wing_deflection = self.wing_wedge - self.aoa

        self._wing_lower = self._apply_deflection(self._post_nose_lower, lower_wing_deflection)
        self._wing_upper = self._apply_deflection(self._post_nose_upper, upper_wing_deflection)

        # Wing leading edge stagnation uses post‑nose flow as the “freestream”
        self._wing_le_M = self._post_nose_lower['u_e'] / math.sqrt(GAMMA * R_AIR * self._post_nose_lower['T_e'])
        self._wing_le_rho = self._post_nose_lower['P_e'] / (R_AIR * self._post_nose_lower['T_e'])
        self._wing_le_T = self._post_nose_lower['T_e']
        self._wing_le_P = self._post_nose_lower['P_e']
        self._wing_le_V = self._post_nose_lower['u_e']
        # ── Aft fuselage edge conditions (post‑shock from nose, no expansion) ─
        # Conservative: use the same post‑shock properties as at the nose cone base.
        self._body_lower = self._shoulder_expansion(self._nose_lower, self.nose_wedge)
        self._body_upper = self._shoulder_expansion(self._nose_upper, self.nose_wedge)

        self._print_conditions()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _print_conditions(self):
        print("\n" + "=" * 66)
        print("  HYPERSONIC HEAT FLUX ANALYSER  –  Initialised")
        print("=" * 66)
        print(f"  Mach number         : {self.M_inf:.2f}")
        print(f"  Altitude            : {self.altitude_km:.1f} km")
        print(f"  Angle of attack     : {self.aoa:.1f}°")
        print(f"  Free-stream T       : {self.T_inf:.1f} K")
        print(f"  Free-stream P       : {self.P_inf:.1f} Pa")
        print(f"  Free-stream density : {self.rho_inf:.5f} kg/m³")
        print(f"  Free-stream velocity: {self.V_inf:.1f} m/s")
        print(f"  Wall temperature    : {self.T_w:.1f} K")
        print(f"  Tw/Taw (approx)     : "
              f"{self.T_w / (self.T_inf * (1 + (GAMMA-1)/2 * self.M_inf**2)):.3f}")
        print(f"  Boundary layer      : "
              f"{'TURBULENT (conservative)' if self.is_turbulent else 'Laminar'}")
        print(f"  Nose radius         : {self.nose_radius*1000:.1f} mm")
        print(f"  Nose wedge angle    : {self.nose_wedge:.1f}°")
        print(f"  Cockpit length      : {self.cockpit_len:.1f} m")
        print(f"  Wing LE radius      : {self.wing_le_radius*1000:.1f} mm")
        print(f"  Wing wedge angle    : {self.wing_wedge:.1f}°")
        print()
        print(f"  {'Surface / Region':<32} {'T_e (K)':>9} {'u_e (m/s)':>11} {'P_e (Pa)':>11}")
        print(f"  {'-'*64}")
        for name, cond in [
            ("Nose cone – windward", self._nose_lower),
            ("Nose cone – leeward ", self._nose_upper),
            ("Aft body – windward  ", self._body_lower),
            ("Aft body – leeward   ", self._body_upper),
            ("Wing lower surface   ", self._wing_lower),
            ("Wing upper surface   ", self._wing_upper),
        ]:
            print(f"  {name:<32} {cond['T_e']:>9.1f} {cond['u_e']:>11.1f} {cond['P_e']:>11.1f}")
        print("=" * 66)

    # ── Public analysis methods ───────────────────────────────────────────────

    def run(self, fuselage_length: float, wing_chord: float,
            num_points: int = 100) -> Dict:
        """Run the complete vehicle analysis.  Returns results dict."""
        results = {}

        # ── 1. Nose stagnation ─────────────────────────────────────────────
        q_nose, diag_nose = stagnation_heat_flux(
            self.M_inf, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
            self.nose_radius, self.T_w)

        print("\n── Nose stagnation ──────────────────────────────────────────────")
        print(f"  Post-shock T_e     : {diag_nose['T_e']:.1f} K")
        print(f"  Velocity gradient  : {diag_nose['du_dx']:.2f} s⁻¹")
        print(f"  Δh (h0 − hw)       : {(diag_nose['h0'] - diag_nose['h_w'])/1e3:.1f} kJ/kg")
        print(f"  Heat flux          : {q_nose/1e3:.2f} kW/m²")

        results['nose'] = dict(q=q_nose, diag=diag_nose)

        # ── 2. Wing leading edge ───────────────────────────────────────────
        q_le = leading_edge_heat_flux(
            self.M_inf, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
            self.wing_le_radius, self.T_w, sweep_deg=0.0)

        print("\n── Wing leading edge ────────────────────────────────────────────")
        print(f"  LE radius          : {self.wing_le_radius*1000:.1f} mm")
        print(f"  Heat flux          : {q_le/1e3:.2f} kW/m²")

        results['wing_le'] = dict(q=q_le)

        # ── 3. Fuselage: nose cone (conical surface) + aft body ────────────
        # Axial coordinates from 0 to fuselage_length
        x_axial = np.linspace(0.05, fuselage_length, num_points)

        # Convert axial distance to distance along the cone surface for the nose region
        # Compute total streamwise distance from nose tip along the surface
        cos_wedge = math.cos(math.radians(self.nose_wedge))
        # Length along cone from tip to shoulder
        cone_surface_len = self.cockpit_len / cos_wedge

        q_lower_fuse = np.zeros_like(x_axial)
        q_upper_fuse = np.zeros_like(x_axial)

        for i, x in enumerate(x_axial):
            if x <= self.cockpit_len:
                # Nose cone region: distance from tip along cone
                s = x / cos_wedge
                q_lower_fuse[i] = flat_plate_heat_flux(s, **self._nose_lower,
                                                       T_w=self.T_w, turbulent=self.is_turbulent)
                q_upper_fuse[i] = flat_plate_heat_flux(s, **self._nose_upper,
                                                       T_w=self.T_w, turbulent=self.is_turbulent)
            else:
                # Aft body: total distance = cone length + axial distance aft of shoulder
                x_aft = x - self.cockpit_len
                s_total = cone_surface_len + x_aft
                q_lower_fuse[i] = flat_plate_heat_flux(s_total, **self._body_lower,
                                                       T_w=self.T_w, turbulent=self.is_turbulent)
                q_upper_fuse[i] = flat_plate_heat_flux(s_total, **self._body_upper,
                                                       T_w=self.T_w, turbulent=self.is_turbulent)
        q_avg_fuse = 0.5 * (q_lower_fuse + q_upper_fuse)

        print("\n── Fuselage distribution (nose cone + aft body) ────────────────")
        print(f"  Nose wedge angle   : {self.nose_wedge:.1f}°, cockpit length = {self.cockpit_len:.1f} m")
        print(f"  Windward side (δ = {self.nose_wedge+self.aoa:.1f}°): peak = {q_lower_fuse.max()/1e3:.2f} kW/m²")
        print(f"  Leeward side  (δ = {self.nose_wedge-self.aoa:+.1f}°): peak = {q_upper_fuse.max()/1e3:.2f} kW/m²")
        if self.nose_wedge - self.aoa < 0:
            print("    → Leeward nose cone is in expansion (Prandtl‑Meyer).")

        results['fuselage'] = dict(x=x_axial,
                                    q_upper=q_upper_fuse,
                                    q_lower=q_lower_fuse,
                                    q_avg=q_avg_fuse,
                                    cockpit_len=self.cockpit_len)

        # ── 4. Wing flat-plate distribution (unchanged) ─────────────────────
        x_wing = np.linspace(0.01, wing_chord, num_points)
        q_lower_wing = np.array([
            flat_plate_heat_flux(x, **self._wing_lower,
                                  T_w=self.T_w, turbulent=self.is_turbulent)
            for x in x_wing])
        q_upper_wing = np.array([
            flat_plate_heat_flux(x, **self._wing_upper,
                                  T_w=self.T_w, turbulent=self.is_turbulent)
            for x in x_wing])
        q_avg_wing = 0.5 * (q_lower_wing + q_upper_wing)

        lower_defl = self.wing_wedge + self.aoa
        upper_defl = self.wing_wedge - self.aoa
        print("\n── Wing chord distribution ──────────────────────────────────────")
        print(f"  Lower surface (δ = +{lower_defl:.1f}°): {q_lower_wing.max()/1e3:.2f} kW/m²")
        upper_mode = "shock" if upper_defl > 0 else "expansion"
        print(f"  Upper surface ({upper_mode}, δ = {upper_defl:+.1f}°): {q_upper_wing.max()/1e3:.2f} kW/m²")

        results['wing'] = dict(x=x_wing,
                                q_upper=q_upper_wing,
                                q_lower=q_lower_wing,
                                q_avg=q_avg_wing)

        # ── 5. Summary ─────────────────────────────────────────────────────
        peak_nose  = q_nose
        peak_le    = q_le
        peak_fuse  = max(q_lower_fuse.max(), q_upper_fuse.max())
        peak_wing  = max(q_lower_wing.max(), q_upper_wing.max())
        peak_all   = max(peak_nose, peak_le, peak_fuse, peak_wing)

        print("\n" + "=" * 66)
        print("  PEAK HEAT FLUX SUMMARY  (for TPS sizing)")
        print("=" * 66)
        print(f"  Nose stagnation    : {peak_nose/1e3:>8.2f} kW/m²")
        print(f"  Wing leading edge  : {peak_le/1e3:>8.2f} kW/m²")
        print(f"  Fuselage (max)     : {peak_fuse/1e3:>8.2f} kW/m²")
        print(f"  Wing chord (max)   : {peak_wing/1e3:>8.2f} kW/m²")
        print(f"  ──────────────────────────────────────────────────────────")
        print(f"  OVERALL PEAK       : {peak_all/1e3:>8.2f} kW/m²")
        print("=" * 66)
        print("  ⚠  Apply ≥ 1.5× safety factor for TPS material sizing.")
        print("  ⚠  Shock–shock / shock–BL interactions not modelled.")
        print("  ⚠  Dissociation ignored (valid below ~Mach 7 at altitude).")
        print("  ⚠  Aft body uses post‑shock properties w/o shoulder expansion.")
        print("      This overpredicts heating (conservative).")
        print("=" * 66)

        results['summary'] = dict(nose=peak_nose, wing_le=peak_le,
                                   fuselage=peak_fuse, wing=peak_wing,
                                   overall=peak_all)
        return results

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(self, results: Dict, save_prefix: Optional[str] = None):
        """Generate surface-distribution plots and summary bar chart."""

        aoa_str = f", AoA={self.aoa:.1f}°"
        suptitle = (f"Convective Heat Flux – M={self.M_inf:.1f}, "
                    f"h={self.altitude_km:.0f} km{aoa_str}, "
                    f"Tw={self.T_w:.0f} K, "
                    f"{'Turbulent' if self.is_turbulent else 'Laminar'} BL")

        # ── Figure 1: surface distributions (fuselage + wing) ───────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(suptitle, fontsize=11)

        # Fuselage plot (now with nose‑cone / aft body transition)
        d_fuse = results['fuselage']
        x_fuse = d_fuse['x']
        cockpit = d_fuse['cockpit_len']
        lower_label = f"Windward (δ_nose = {self.nose_wedge+self.aoa:.1f}°)"
        upper_label = f"Leeward  (δ_nose = {self.nose_wedge-self.aoa:+.1f}°)"

        axes[0].plot(x_fuse, d_fuse['q_lower'] / 1e3, 'r-', lw=1.8, label=lower_label)
        axes[0].plot(x_fuse, d_fuse['q_upper'] / 1e3, 'b--', lw=1.8, label=upper_label)
        axes[0].plot(x_fuse, d_fuse['q_avg'] / 1e3, 'k:', lw=1.4, label='Average', alpha=0.6)
        axes[0].axvline(cockpit, color='gray', linestyle='--', linewidth=1,
                         label=f'Nose cone ends (x={cockpit:.1f} m)')
        axes[0].set_xlabel('Axial distance from nose tip (m)', fontsize=10)
        axes[0].set_ylabel('Convective heat flux (kW/m²)', fontsize=10)
        axes[0].set_title('Fuselage heat flux – nose cone + aft body', fontsize=10)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Wing plot (unchanged)
        d_wing = results['wing']
        lower_w = self.wing_wedge + self.aoa
        upper_w = self.wing_wedge - self.aoa
        wing_lower_lbl = f"Lower (δ=+{lower_w:.1f}°)"
        wing_upper_lbl = f"Upper (δ={upper_w:+.1f}°, {'shock' if upper_w>0 else 'expansion'})"
        axes[1].plot(d_wing['x'], d_wing['q_lower'] / 1e3, 'r-', lw=1.8, label=wing_lower_lbl)
        axes[1].plot(d_wing['x'], d_wing['q_upper'] / 1e3, 'b--', lw=1.8, label=wing_upper_lbl)
        axes[1].plot(d_wing['x'], d_wing['q_avg'] / 1e3, 'k:', lw=1.4, label='Average', alpha=0.6)
        axes[1].set_xlabel('Distance from leading edge (m)', fontsize=10)
        axes[1].set_ylabel('Convective heat flux (kW/m²)', fontsize=10)
        axes[1].set_title('Wing chord heat flux distribution', fontsize=10)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_prefix:
            path = f"{save_prefix}_distributions.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")
        else:
            plt.show()

        # ── Figure 2: peak bar chart ───────────────────────────────────────
        s      = results['summary']
        labels = ['Nose\n(stagnation)', 'Wing\n(leading edge)',
                  'Fuselage\n(max)', 'Wing chord\n(max)']
        values = [s['nose'] / 1e3, s['wing_le'] / 1e3,
                  s['fuselage'] / 1e3, s['wing'] / 1e3]
        colors = ['#c0392b', '#e67e22', '#f1c40f', '#2980b9']

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        bars = ax2.bar(labels, values, color=colors,
                        edgecolor='black', linewidth=1.2)
        ax2.axhline(s['overall'] / 1e3, color='k', lw=1.5,
                     linestyle=':', label=f"Overall peak: {s['overall']/1e3:.1f} kW/m²")
        ax2.set_ylabel('Peak convective heat flux (kW/m²)', fontsize=11)
        ax2.set_title('Maximum heat flux per component  –  TPS design reference',
                       fontsize=11)
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.legend(fontsize=10)

        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + max(values) * 0.01,
                      f'{val:.1f}', ha='center', va='bottom',
                      fontweight='bold', fontsize=10)

        plt.tight_layout()
        if save_prefix:
            path = f"{save_prefix}_summary.png"
            fig2.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")
        else:
            plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    vehicle = HypersonicVehicleAnalysis(
        mach             = MACH,
        altitude_km      = ALTITUDE_KM,
        wall_temp_k      = WALL_TEMP_K,
        is_turbulent     = IS_TURBULENT,
        aoa_deg          = AOA_DEG,
        nose_radius_m    = NOSE_RADIUS_M,
        nose_wedge_deg   = NOSE_WEDGE_DEG,
        cockpit_length_m = COCKPIT_LENGTH_M,
        wing_le_radius_m = WING_LE_RADIUS_M,
        wing_wedge_deg   = WING_WEDGE_DEG,
    )

    results = vehicle.run(
        fuselage_length = FUSELAGE_LENGTH_M,
        wing_chord      = WING_CHORD_M,
        num_points      = NUM_POINTS,
    )

    vehicle.plot(
        results,
        save_prefix = FIGURE_PREFIX if SAVE_FIGURES else None,
    )
