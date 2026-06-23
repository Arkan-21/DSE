

MACH             = 5.0    # Free-stream Mach number  [-]
ALTITUDE_KM      = 30.0   # Cruise altitude           [km]
WALL_TEMP_K      = 373.15 # Wall temperature          [K]
IS_TURBULENT     = True   # True = turbulent BL
AOA_DEG          = 1.0    # Angle of attack           [degrees]

# Body geometry
BODY_RADIUS_M    = 2.0    # Cylinder (body) radius              [m]
OGIVE_LENGTH_M   = 10.493 # Axial length of ogive nose section  [m]
BODY_LENGTH_M    = 21.0   # Total body length (ogive + cylinder)[m]
NOSE_RADIUS_M    = 0.040  # Blunted tip radius of curvature     [m]

# ── Multi-segment wing (Table 12.5) ─────────────────────────────────────────
# Each entry is a WingSegment dataclass (defined below).
# Fields: name, chord_root, chord_tip, span, sweep_le_deg, dihedral_deg,
#         le_radius, wedge_half_angle_deg
#
# sweep_le_deg  : leading-edge sweep angle [degrees] – the PRIMARY new input
# dihedral_deg  : dihedral of this segment [degrees]  (positive = anhedral)
# le_radius     : LE bluntness radius [m]  (Fay-Riddell)
# wedge_half    : local section half-angle for compression/expansion [degrees]

WING_SEGMENTS = [
    # name          chord_root  chord_tip  span   sweep_le  dihedral  le_r    wedge
    ("Connection",  0.661,      0.661,     3.50,  64.00,     0.0,    0.005,   0.0),
    ("Inner",      33.14,      13.26,     7.02,  70.55,    +1.0,    0.012,   0.0),
    ("Mid",        13.26,      10.77,     4.33,  29.84,    +1.0,    0.010,   0.0),
    ("Outer",      10.77,       4.00,     3.29,  67.50,   -20.0,    0.008,   0.0),
]

# Analysis resolution
NUM_POINTS       = 500    # Points along the body (also used per wing segment)
NUM_SPAN_POINTS  = 60     # Spanwise stations per wing segment

# Output
SAVE_FIGURES     = False   # False → interactive; True → save PNGs
FIGURE_PREFIX    = "heat_flux_swept"

# ============================================================
#  END OF USER INPUTS
# ============================================================

import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Wing segment dataclass ────────────────────────────────────────────────────

@dataclass
class WingSegment:
    name:             str
    chord_root:       float   # m
    chord_tip:        float   # m
    span:             float   # m
    sweep_le_deg:     float   # degrees — leading-edge sweep angle
    dihedral_deg:     float   # degrees
    le_radius:        float   # m  — LE bluntness (Fay-Riddell)
    wedge_half_deg:   float   # degrees — section half-angle for BL edge cond.

    @property
    def sweep_le_rad(self)  -> float: return math.radians(self.sweep_le_deg)
    @property
    def dihedral_rad(self)  -> float: return math.radians(self.dihedral_deg)
    @property
    def cos_sweep(self)     -> float: return math.cos(self.sweep_le_rad)
    @property
    def sin_sweep(self)     -> float: return math.sin(self.sweep_le_rad)
    @property
    def taper_ratio(self)   -> float:
        return self.chord_tip / self.chord_root if self.chord_root > 0 else 1.0

    def local_chord(self, eta: float) -> float:
        """Chord at spanwise fraction eta ∈ [0, 1]."""
        return self.chord_root + eta * (self.chord_tip - self.chord_root)

    def normal_chord(self, eta: float) -> float:
        """Chord measured perpendicular to the LE: c_n = c(η) · cos(Λ)."""
        return self.local_chord(eta) * self.cos_sweep

    def effective_aoa(self, freestream_aoa_deg: float) -> float:
        """
        Effective incidence seen by this segment:
            δ_eff = AoA · cos(Λ)  +  dihedral
        The cos(Λ) factor projects the freestream flow angle onto the
        normal-to-LE plane; dihedral adds a geometric incidence offset.
        """
        return freestream_aoa_deg * self.cos_sweep + self.dihedral_deg


# ── Atmospheric model ─────────────────────────────────────────────────────────

def us_standard_atmosphere(altitude_km: float) -> Tuple[float, float, float]:
    """1976 U.S. Standard Atmosphere → (T [K], P [Pa], rho [kg/m³])."""
    R_air, g0 = 287.05, 9.80665
    layers = [
        (0.0,  288.150, -6.5,  101325.0),
        (11.0, 216.650,  0.0,   22632.1),
        (20.0, 216.650,  1.0,    5474.89),
        (32.0, 228.650,  2.8,     868.019),
        (47.0, 270.650,  0.0,     110.906),
        (51.0, 270.650, -2.8,      66.9389),
        (71.0, 214.650, -2.0,       3.95642),
        (86.0, 186.870,  0.0,       0.3734),
    ]
    h = altitude_km
    if not (0 <= h <= 86):
        raise ValueError(f"Altitude {h} km outside 0–86 km range.")
    for i in range(len(layers) - 1):
        h_base, T_base, lapse, P_base = layers[i]
        if h <= layers[i + 1][0]:
            dh = (h - h_base) * 1000.0
            if abs(lapse) < 1e-9:
                T = T_base
                P = P_base * math.exp(-g0 * dh / (R_air * T_base))
            else:
                lapse_m = lapse / 1000.0
                T = T_base + lapse_m * dh
                P = P_base * (T / T_base) ** (-g0 / (lapse_m * R_air))
            return T, P, P / (R_air * T)
    h_base, T_base, _, P_base = layers[-1]
    dh = (h - h_base) * 1000.0
    T  = T_base
    P  = P_base * math.exp(-g0 * dh / (R_air * T_base))
    return T, P, P / (R_air * T)


# ── Gas constants & property helpers ─────────────────────────────────────────

GAMMA  = 1.4
R_AIR  = 287.05
PR     = 0.71
CP     = GAMMA * R_AIR / (GAMMA - 1)
MU_REF = 1.716e-5
T_REF  = 273.15
S_SUTH = 110.4


def viscosity(T: float) -> float:
    return MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)


def thermal_conductivity(T: float) -> float:
    return viscosity(T) * CP / PR


# ── Shock / expansion relations ───────────────────────────────────────────────

def normal_shock(M1: float) -> Tuple[float, float, float]:
    g, M1sq = GAMMA, M1 ** 2
    M2sq  = (1 + (g - 1) / 2 * M1sq) / (g * M1sq - (g - 1) / 2)
    T2_T1 = (1 + 2 * g / (g + 1) * (M1sq - 1)) * (2 + (g - 1) * M1sq) / ((g + 1) * M1sq)
    rho2  = (g + 1) * M1sq / (2 + (g - 1) * M1sq)
    return math.sqrt(M2sq), T2_T1, rho2


def oblique_shock_angle(M1: float, wedge_deg: float) -> Optional[float]:
    theta = math.radians(wedge_deg)
    g     = GAMMA
    mu    = math.asin(1.0 / M1)

    def tob(b):
        sb2 = math.sin(b) ** 2
        cot = math.cos(b) / math.sin(b)
        val = 2 * cot * (M1 ** 2 * sb2 - 1) / (M1 ** 2 * (g + math.cos(2 * b)) + 2)
        return math.atan(max(-1.0, min(1.0, val)))

    betas  = [mu + 1e-4 + (math.pi / 2 - 1e-4 - mu - 1e-4) * i / 60 for i in range(61)]
    thetas = [tob(b) for b in betas]
    idx    = thetas.index(max(thetas))
    if theta > thetas[idx]:
        return None

    b_lo, b_hi = mu + 1e-4, betas[idx]
    for _ in range(200):
        bm = 0.5 * (b_lo + b_hi)
        fm = tob(bm) - theta
        if abs(fm) < 1e-9 or (b_hi - b_lo) < 1e-11:
            break
        if (tob(b_lo) - theta) * fm < 0:
            b_hi = bm
        else:
            b_lo = bm
    return math.degrees(bm)


def compression_edge(M1, rho1, T1, P1, wedge_deg):
    beta_deg = oblique_shock_angle(M1, wedge_deg)
    if beta_deg is None:
        warnings.warn(f"Oblique shock detached at M={M1:.2f}, δ={wedge_deg:.1f}°. "
                      "Using free-stream.", stacklevel=3)
        return dict(T_e=T1, u_e=M1 * math.sqrt(GAMMA * R_AIR * T1), P_e=P1)
    beta = math.radians(beta_deg)
    Mn1  = M1 * math.sin(beta)
    Mn2, T2_T1, rho2_r = normal_shock(Mn1)
    T2   = T1 * T2_T1
    rho2 = rho1 * rho2_r
    theta = math.radians(wedge_deg)
    M2   = Mn2 / math.sin(beta - theta)
    u2   = M2 * math.sqrt(GAMMA * R_AIR * T2)
    P2   = rho2 * R_AIR * T2
    return dict(T_e=T2, u_e=u2, P_e=P2)


def prandtl_meyer_nu(M):
    g  = GAMMA
    gp = (g + 1) / (g - 1)
    return math.sqrt(gp) * math.atan(math.sqrt((M ** 2 - 1) / gp)) - math.atan(math.sqrt(M ** 2 - 1))


def prandtl_meyer_M(nu_target):
    lo, hi = 1.001, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if prandtl_meyer_nu(mid) < nu_target:
            lo = mid
        else:
            hi = mid
    return mid


def expansion_edge(M1, rho1, T1, P1, turn_deg):
    if turn_deg <= 0:
        return dict(T_e=T1, u_e=M1 * math.sqrt(GAMMA * R_AIR * T1), P_e=P1)
    nu1   = prandtl_meyer_nu(M1)
    g     = GAMMA
    nu_max = math.pi / 2 * (math.sqrt((g + 1) / (g - 1)) - 1)
    nu2   = min(nu1 + math.radians(turn_deg), nu_max)
    M2    = prandtl_meyer_M(nu2)
    fac   = (1 + (g - 1) / 2 * M1 ** 2) / (1 + (g - 1) / 2 * M2 ** 2)
    T2    = T1 * fac
    rho2  = rho1 * fac ** (1 / (g - 1))
    u2    = M2 * math.sqrt(GAMMA * R_AIR * T2)
    P2    = rho2 * R_AIR * T2
    return dict(T_e=T2, u_e=u2, P_e=P2)


def tangent_wedge_edge(M_inf, rho_inf, T_inf, P_inf, V_inf, local_slope_deg):
    delta = local_slope_deg
    if abs(delta) < 0.01:
        return dict(T_e=T_inf, u_e=V_inf, P_e=P_inf)
    if delta > 0:
        return compression_edge(M_inf, rho_inf, T_inf, P_inf, delta)
    else:
        return expansion_edge(M_inf, rho_inf, T_inf, P_inf, abs(delta))


# ── Reference enthalpy & flat-plate heat flux ─────────────────────────────────

def reference_state(T_e, u_e, P_e, T_w, turbulent):
    r    = PR ** (1 / 3) if turbulent else math.sqrt(PR)
    M_e  = u_e / math.sqrt(GAMMA * R_AIR * T_e)
    T_aw = T_e * (1 + r * (GAMMA - 1) / 2 * M_e ** 2)
    h_star = CP * T_w + 0.5 * CP * (T_e - T_w) + 0.22 * CP * (T_aw - T_w)
    T_star = h_star / CP
    rho_star = P_e / (R_AIR * T_star)
    return T_star, rho_star, viscosity(T_star)


def flat_plate_qw(s, edge, T_w, turbulent):
    if s <= 0:
        return 0.0
    T_e, u_e, P_e = edge['T_e'], edge['u_e'], edge['P_e']
    T_star, rho_star, mu_star = reference_state(T_e, u_e, P_e, T_w, turbulent)
    k_star = thermal_conductivity(T_star)
    Re_s   = rho_star * u_e * s / mu_star
    if Re_s < 1.0:
        return 0.0
    Nu_s = (0.0296 * Re_s ** 0.8 if turbulent else 0.332 * Re_s ** 0.5) * PR ** (1 / 3)
    h_c  = Nu_s * k_star / s
    r    = PR ** (1 / 3) if turbulent else math.sqrt(PR)
    M_e  = u_e / math.sqrt(GAMMA * R_AIR * T_e)
    T_aw = T_e * (1 + r * (GAMMA - 1) / 2 * M_e ** 2)
    return max(h_c * (T_aw - T_w), 0.0)


# ── Stagnation-point heat flux (Fay-Riddell, body nose) ──────────────────────

def stagnation_qw(M_inf, rho_inf, T_inf, P_inf, V_inf, nose_r, T_w):
    _, T2_T1, rho2_r = normal_shock(M_inf)
    T_e   = T_inf * T2_T1
    rho_e = rho_inf * rho2_r
    P2_P1 = 1 + 2 * GAMMA / (GAMMA + 1) * (M_inf ** 2 - 1)
    P_e   = P_inf * P2_P1
    du_dx = (1 / nose_r) * math.sqrt(2 * 0.5 * rho_inf * V_inf ** 2 / rho_e)
    mu_e  = viscosity(T_e)
    h0    = CP * T_inf + 0.5 * V_inf ** 2
    h_w   = CP * T_w
    q_s   = (0.57 / PR ** 0.6) * math.sqrt(rho_e * mu_e * du_dx) * max(h0 - h_w, 0.0)
    return q_s, dict(T_e=T_e, rho_e=rho_e, P_e=P_e, du_dx=du_dx, h0=h0, h_w=h_w)


# ── Leading-edge stagnation heat flux per segment (sweep-corrected) ───────────

def le_stag_qw_swept(M_inf: float, rho_inf: float, T_inf: float,
                     P_inf: float, V_inf: float,
                     le_radius: float, T_w: float,
                     sweep_deg: float) -> Dict:
    """
    Fay-Riddell LE stagnation heat flux with FULL sweep-angle accounting.

    The independence principle (Küchemann / swept-wing theory) states that
    only the velocity component NORMAL to the leading edge drives the
    shock and boundary layer.  For a swept LE with angle Λ:

        V_n  = V_inf · cos(Λ)          normal velocity component
        M_n  = M_inf · cos(Λ)          normal Mach number
        q_LE = q_FR(M_n, V_n, le_r)   Fay-Riddell with normal-flow inputs

    This correctly gives:
        • q_LE → q_FR(M_inf)       as Λ → 0  (unswept, same as body nose)
        • q_LE → 0                 as Λ → 90°  (fully swept / razor LE)

    Parameters
    ----------
    sweep_deg : Leading-edge sweep angle [degrees] for this specific segment.

    Returns
    -------
    dict with keys: q_le, M_n, V_n, T_e_n, rho_e_n, du_dx_n, h0_n
    """
    if le_radius <= 0:
        le_radius = 1e-3

    sweep_rad = math.radians(sweep_deg)
    cos_sw    = math.cos(sweep_rad)

    # Normal-to-LE velocity component
    V_n = V_inf * cos_sw
    M_n = max(M_inf * cos_sw, 1.001)   # must be supersonic for normal shock to exist

    # Post-(normal-)shock conditions using the normal Mach number
    _, T2_T1, rho2_r = normal_shock(M_n)
    T_e_n   = T_inf * T2_T1
    rho_e_n = rho_inf * rho2_r
    P2_P1   = 1 + 2 * GAMMA / (GAMMA + 1) * (M_n ** 2 - 1)
    P_e_n   = P_inf * P2_P1

    # Velocity gradient at the stagnation line of a cylinder of radius le_radius
    # du/dx = (1/r) * sqrt(2 * q_inf / rho_e_n)
    # where q_inf = ½ ρ_inf V_n² is the normal dynamic pressure
    du_dx_n = (1.0 / le_radius) * math.sqrt(2.0 * 0.5 * rho_inf * V_n ** 2 / rho_e_n)

    mu_e_n = viscosity(T_e_n)

    # Total enthalpy based on full V_inf (the parallel component is preserved
    # through the shock; both components contribute to total enthalpy)
    h0_n = CP * T_inf + 0.5 * V_inf ** 2
    h_w  = CP * T_w

    q_le = (0.57 / PR ** 0.6) * math.sqrt(rho_e_n * mu_e_n * du_dx_n) * max(h0_n - h_w, 0.0)

    return dict(
        q_le    = q_le,
        M_n     = M_n,
        V_n     = V_n,
        sweep   = sweep_deg,
        cos_sw  = cos_sw,
        T_e_n   = T_e_n,
        rho_e_n = rho_e_n,
        du_dx_n = du_dx_n,
        h0_n    = h0_n,
    )


# ── Tangent-ogive geometry ────────────────────────────────────────────────────

class TangentOgive:
    def __init__(self, R, L):
        if L <= 0 or R <= 0:
            raise ValueError("Ogive R and L must be positive.")
        self.R    = R
        self.L    = L
        self.rho_c = (R ** 2 + L ** 2) / (2 * R)

    def radius(self, x):
        x = max(0.0, min(x, self.L))
        return math.sqrt(self.rho_c ** 2 - (self.L - x) ** 2) + self.R - self.rho_c

    def slope_deg(self, x):
        x    = max(1e-6, min(x, self.L - 1e-6))
        drdx = (self.L - x) / math.sqrt(self.rho_c ** 2 - (self.L - x) ** 2)
        return math.degrees(math.atan(drdx))

    def arc_length(self, x_stations):
        s = np.zeros_like(x_stations)
        for i in range(1, len(x_stations)):
            x0, x1 = x_stations[i - 1], x_stations[i]
            dx = x1 - x0
            xm = max(1e-6, min(0.5 * (x0 + x1), self.L - 1e-6))
            drdx = (self.L - xm) / math.sqrt(self.rho_c ** 2 - (self.L - xm) ** 2)
            s[i] = s[i - 1] + dx * math.sqrt(1 + drdx ** 2)
        return s


# ── Wing segment heat-flux computation ───────────────────────────────────────

def compute_segment_heat_flux(seg: WingSegment,
                              M_inf: float, rho_inf: float,
                              T_inf: float, P_inf: float, V_inf: float,
                              T_w: float, aoa_deg: float, turbulent: bool,
                              n_chord: int = 60, n_span: int = 40) -> Dict:
    """
    Compute chordwise heat-flux distributions at multiple spanwise stations
    for a single trapezoidal wing segment with given sweep.

    Sweep physics applied
    ---------------------
    1.  **LE stagnation**: full swept Fay-Riddell (see le_stag_qw_swept).
    2.  **Flat-plate BL on surface**: the BL runs in the chord-normal
        direction.  Edge conditions are computed with the *normal* Mach
        number M_n = M_inf·cos(Λ) and the *normal* velocity V_n = V_inf·cos(Λ)
        as the driving velocity.  The running length is the normal-to-LE
        chordwise arc length  s_n = x_c · cos(Λ)  where x_c is measured
        along the chord from the LE.
    3.  **Effective AoA** per segment: δ_eff = AoA·cos(Λ) + dihedral.

    Returns
    -------
    dict with:
        eta        : spanwise fraction array [0,1]
        x_chord    : normalised chordwise position [0,1]
        q_lower    : (n_span, n_chord) array  W/m²  lower surface
        q_upper    : (n_span, n_chord) array  W/m²  upper surface
        q_le       : (n_span,) array  W/m²  LE stag. flux per spanwise station
        le_diag    : list of per-station LE diagnostic dicts
        chord_arr  : (n_span,) local chord length [m]
        seg        : WingSegment reference
    """
    eta_arr    = np.linspace(0.0, 1.0, n_span)
    xc_arr     = np.linspace(0.01, 1.0, n_chord)  # normalised chord (0=LE, 1=TE)

    q_lower    = np.zeros((n_span, n_chord))
    q_upper    = np.zeros((n_span, n_chord))
    q_le_arr   = np.zeros(n_span)
    le_diag    = []
    chord_arr  = np.zeros(n_span)

    # Normal-to-LE Mach and velocity (constant across taper for a given segment)
    cos_sw  = seg.cos_sweep
    M_n     = max(M_inf * cos_sw, 1.001)
    V_n     = V_inf * cos_sw

    # Effective AoA for this segment (includes dihedral)
    delta_eff = seg.effective_aoa(aoa_deg)

    # Edge conditions for lower and upper surfaces (normal-flow frame)
    # The wedge deflection is the effective AoA in the normal plane
    edge_lower = tangent_wedge_edge(M_n, rho_inf, T_inf, P_inf, V_n,
                                     delta_eff + seg.wedge_half_deg)
    edge_upper = tangent_wedge_edge(M_n, rho_inf, T_inf, P_inf, V_n,
                                     -(delta_eff + seg.wedge_half_deg))

    for i, eta in enumerate(eta_arr):
        c_local  = seg.local_chord(eta)
        c_normal = seg.normal_chord(eta)   # c · cos(Λ)
        chord_arr[i] = c_local

        # LE stagnation flux (sweep-corrected)
        ld = le_stag_qw_swept(M_inf, rho_inf, T_inf, P_inf, V_inf,
                               seg.le_radius, T_w, seg.sweep_le_deg)
        q_le_arr[i] = ld['q_le']
        le_diag.append(ld)

        # Chordwise heat flux: running length = x_c · c_normal
        for j, xc in enumerate(xc_arr):
            s_n = xc * c_normal   # arc-length in normal-to-LE direction
            ql  = flat_plate_qw(s_n, edge_lower, T_w, turbulent)
            qu  = flat_plate_qw(s_n, edge_upper, T_w, turbulent)
            q_lower[i, j] = ql
            q_upper[i, j] = qu

    return dict(
        eta       = eta_arr,
        x_chord   = xc_arr,
        q_lower   = q_lower,
        q_upper   = q_upper,
        q_le      = q_le_arr,
        le_diag   = le_diag,
        chord_arr = chord_arr,
        seg       = seg,
    )


# ── Main analysis class ───────────────────────────────────────────────────────

class OgiveCylinderAnalysis:

    def __init__(self, mach, altitude_km, wall_temp_k, is_turbulent, aoa_deg,
                 body_radius_m, ogive_length_m, body_length_m, nose_radius_m,
                 wing_segments: List[WingSegment]):

        self.M        = mach
        self.alt      = altitude_km
        self.T_w      = wall_temp_k
        self.turb     = is_turbulent
        self.aoa      = aoa_deg
        self.R        = body_radius_m
        self.L_ogive  = ogive_length_m
        self.L_body   = body_length_m
        self.r_nose   = nose_radius_m
        self.wing_segs = wing_segments

        self.T_inf, self.P_inf, self.rho_inf = us_standard_atmosphere(altitude_km)
        self.a_inf = math.sqrt(GAMMA * R_AIR * self.T_inf)
        self.V_inf = mach * self.a_inf

        self.ogive = TangentOgive(body_radius_m, ogive_length_m)
        self._print_header()

    def _print_header(self):
        T_ratio = self.T_w / (self.T_inf * (1 + (GAMMA - 1) / 2 * self.M ** 2))
        print("\n" + "=" * 68)
        print("  OGIVE-CYLINDER AEROTHERMAL ANALYSIS  –  MULTI-SEGMENT SWEPT WING")
        print("=" * 68)
        print(f"  Mach                : {self.M:.2f}")
        print(f"  Altitude            : {self.alt:.1f} km")
        print(f"  AoA                 : {self.aoa:.1f}°")
        print(f"  T_inf / P_inf / rho : {self.T_inf:.1f} K  {self.P_inf:.1f} Pa  {self.rho_inf:.5f} kg/m³")
        print(f"  V_inf               : {self.V_inf:.1f} m/s")
        print(f"  Wall temperature    : {self.T_w:.1f} K   (Tw/Taw ≈ {T_ratio:.3f})")
        print(f"  Boundary layer      : {'TURBULENT' if self.turb else 'Laminar'}")
        print(f"  Body radius / length: {self.R:.3f} m / {self.L_body:.2f} m")
        print(f"  Ogive length        : {self.L_ogive:.2f} m  "
              f"(fineness = {self.L_ogive/self.R:.1f})")
        print(f"  Nose tip radius     : {self.r_nose*1000:.1f} mm")
        print()
        print(f"  {'Segment':<14} {'chord_root':>10} {'chord_tip':>10} {'span':>7} "
              f"{'sweep_LE':>9} {'dihedral':>9} {'r_LE[mm]':>9}")
        print(f"  {'-'*68}")
        for seg in self.wing_segs:
            print(f"  {seg.name:<14} {seg.chord_root:>10.3f} {seg.chord_tip:>10.3f} "
                  f"{seg.span:>7.2f} {seg.sweep_le_deg:>8.2f}° {seg.dihedral_deg:>8.1f}° "
                  f"{seg.le_radius*1e3:>8.1f}")
        print("=" * 68)

    # ── Body distribution ────────────────────────────────────────────────

    def _body_distribution(self, num_points):
        x_ogive = np.linspace(0.001, self.L_ogive, num_points // 2)
        x_cyl   = np.linspace(self.L_ogive, self.L_body, num_points // 2 + 1)[1:]
        x_all   = np.concatenate([x_ogive, x_cyl])

        x_fine_ogive = np.linspace(0.0, self.L_ogive, 2000)
        s_ogive_fine = self.ogive.arc_length(x_fine_ogive)
        s_at_junction = s_ogive_fine[-1]

        def arc_at_x(x_val):
            if x_val <= self.L_ogive:
                return float(np.interp(x_val, x_fine_ogive, s_ogive_fine))
            return s_at_junction + (x_val - self.L_ogive)

        q_wind  = np.zeros(len(x_all))
        q_lee   = np.zeros(len(x_all))
        slopes  = np.zeros(len(x_all))

        q_stag, diag_stag = stagnation_qw(
            self.M, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
            self.r_nose, self.T_w)

        for i, x in enumerate(x_all):
            s   = arc_at_x(x)
            phi = self.ogive.slope_deg(x) if x <= self.L_ogive else 0.0
            slopes[i] = phi

            edge_wind = tangent_wedge_edge(
                self.M, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
                phi + self.aoa)
            edge_lee  = tangent_wedge_edge(
                self.M, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
                phi - self.aoa)

            q_wind[i] = min(flat_plate_qw(s, edge_wind, self.T_w, self.turb), q_stag)
            q_lee[i]  = min(flat_plate_qw(s, edge_lee,  self.T_w, self.turb), q_stag)

        return dict(x=x_all, s=np.array([arc_at_x(x) for x in x_all]),
                    slope=slopes, q_wind=q_wind, q_lee=q_lee,
                    q_stag=q_stag, diag_stag=diag_stag,
                    s_junction=s_at_junction, x_junction=self.L_ogive)

    # ── All wing segments ────────────────────────────────────────────────

    def _all_wing_distributions(self, n_chord=60, n_span=40):
        results = []
        for seg in self.wing_segs:
            r = compute_segment_heat_flux(
                seg, self.M, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
                self.T_w, self.aoa, self.turb, n_chord, n_span)
            results.append(r)
        return results

    # ── Public run ───────────────────────────────────────────────────────

    def run(self, num_points=300, n_chord=60, n_span=40):
        print("\n── Computing body distribution …")
        body = self._body_distribution(num_points)

        print("── Computing wing segment distributions …")
        wing_results = self._all_wing_distributions(n_chord, n_span)

        # ── Console body report ────────────────────────────────────────
        q_s       = body['q_stag']
        q_w_max   = body['q_wind'].max()
        q_l_max   = body['q_lee'].max()
        x_wm      = body['x'][body['q_wind'].argmax()]
        x_lm      = body['x'][body['q_lee'].argmax()]

        d = body['diag_stag']
        print(f"\n── Nose stagnation ──────────────────────────────────────────────────")
        print(f"  Post-shock T_e     : {d['T_e']:.1f} K")
        print(f"  Velocity gradient  : {d['du_dx']:.2f} s⁻¹")
        print(f"  Δh (h0 − hw)       : {(d['h0']-d['h_w'])/1e3:.1f} kJ/kg")
        print(f"  Heat flux          : {q_s/1e3:.2f} kW/m²")

        print(f"\n── Body surface ─────────────────────────────────────────────────────")
        print(f"  Windward peak      : {q_w_max/1e3:.2f} kW/m²  at x = {x_wm:.3f} m")
        print(f"  Leeward  peak      : {q_l_max/1e3:.2f} kW/m²  at x = {x_lm:.3f} m")

        # ── Console wing report ────────────────────────────────────────
        print(f"\n── Wing segments (sweep-corrected LE + surface flux) ────────────────")
        print(f"  {'Segment':<14} {'Sweep':>7} {'cos(Λ)':>7} {'M_n':>6} "
              f"{'q_LE [kW/m²]':>14} {'q_surf_lo [kW/m²]':>18} {'q_surf_hi [kW/m²]':>18}")
        print(f"  {'-'*87}")

        all_peaks = [q_s, q_w_max, q_l_max]
        for wr in wing_results:
            seg      = wr['seg']
            ld_ref   = wr['le_diag'][len(wr['le_diag']) // 2]   # mid-span station
            q_le_mid = wr['q_le'].mean()
            q_lo_max = wr['q_lower'].max()
            q_hi_max = wr['q_upper'].max()
            all_peaks += [wr['q_le'].max(), q_lo_max, q_hi_max]
            print(f"  {seg.name:<14} {seg.sweep_le_deg:>6.1f}° {ld_ref['cos_sw']:>7.4f} "
                  f"{ld_ref['M_n']:>6.3f} {q_le_mid/1e3:>14.2f} "
                  f"{q_lo_max/1e3:>18.2f} {q_hi_max/1e3:>18.2f}")

        peak_all = max(all_peaks)

        print("\n" + "=" * 68)
        print("  PEAK HEAT FLUX SUMMARY")
        print("=" * 68)
        print(f"  Nose stagnation    : {q_s/1e3:>8.2f} kW/m²")
        print(f"  Body surface (max) : {max(q_w_max,q_l_max)/1e3:>8.2f} kW/m²")
        for wr in wing_results:
            seg = wr['seg']
            pk  = max(wr['q_le'].max(), wr['q_lower'].max(), wr['q_upper'].max())
            print(f"  Wing {seg.name:<10} : {pk/1e3:>8.2f} kW/m²  "
                  f"(Λ={seg.sweep_le_deg:.1f}°, M_n={M_inf_normal(self.M,seg.sweep_le_deg):.3f})")
        print(f"  {'─'*48}")
        print(f"  OVERALL PEAK       : {peak_all/1e3:>8.2f} kW/m²")
        print("=" * 68)
        print("  ⚠  Tangent-wedge valid for φ < ~20°; overestimates near blunt tip.")
        print("  ⚠  Independence principle accurate for Λ > ~30°.")
        print("  ⚠  Dissociation not modelled (γ=1.4); correction needed above M7.")
        print("  ⚠  Apply ≥ 1.5× safety margin for TPS sizing.")
        print("=" * 68)

        summary = dict(
            q_stag=q_s, q_body=max(q_w_max, q_l_max), peak_all=peak_all,
            wing_results=wing_results)
        return dict(body=body, wing_results=wing_results, summary=summary)

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot(self, results, save_prefix=None):
        body = results['body']
        wing_results = results['wing_results']
        x, q_w, q_l = body['x'], body['q_wind'], body['q_lee']
        x_j = self.L_ogive
        q_s = body['q_stag']
        title_base = self._title_base()

        # ────────────────────────────────────────────────────────────────
        # Figure 1: body geometry + body heat flux + sweep table
        # ────────────────────────────────────────────────────────────────
        fig1, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig1.suptitle(f"Body Aerothermal Heating  –  {title_base}", fontsize=11)

        # Body geometry
        ax_g = axes[0]
        x_og = np.linspace(0, self.L_ogive, 300)
        r_og = np.array([self.ogive.radius(xi) for xi in x_og])
        x_cy = np.array([self.L_ogive, self.L_body])
        r_cy = np.array([self.R, self.R])
        ax_g.fill_between(np.concatenate([x_og, x_cy]),
                           np.concatenate([r_og, r_cy]), alpha=0.15, color='steelblue')
        ax_g.fill_between(np.concatenate([x_og, x_cy]),
                           -np.concatenate([r_og, r_cy]), alpha=0.15, color='steelblue')
        ax_g.plot(np.concatenate([x_og, x_cy]),
                   np.concatenate([r_og, r_cy]), 'b-', lw=2)
        ax_g.plot(np.concatenate([x_og, x_cy]),
                   -np.concatenate([r_og, r_cy]), 'b-', lw=2)
        ax_g.axvline(x_j, color='gray', ls='--', lw=1, label='Ogive–cyl. junction')
        ax_g.set_xlabel('x [m]'); ax_g.set_ylabel('r [m]')
        ax_g.set_title('Body profile'); ax_g.set_aspect('equal', adjustable='datalim')
        ax_g.legend(fontsize=8); ax_g.grid(alpha=0.25)

        # Body heat flux
        ax_hf = axes[1]
        ax_hf.axvline(x_j, color='gray', ls='--', lw=1, label=f'Ogive–cyl. x={x_j:.1f} m')
        ax_hf.axhline(q_s / 1e3, color='purple', ls=':', lw=1.5,
                       label=f'Stag.: {q_s/1e3:.1f} kW/m²')
        if abs(self.aoa) < 0.01:
            ax_hf.plot(x, q_w / 1e3, 'b-', lw=2, label='Surface (AoA=0)')
        else:
            ax_hf.plot(x, q_w / 1e3, 'r-',  lw=2, label=f'Windward (+{self.aoa:.1f}°)')
            ax_hf.plot(x, q_l / 1e3, 'b--', lw=2, label=f'Leeward  (−{self.aoa:.1f}°)')
            ax_hf.plot(x, 0.5*(q_w+q_l)/1e3, 'k:', lw=1.2, alpha=0.5, label='Average')
        ax_hf.set_xlabel('Axial position x [m]'); ax_hf.set_ylabel('q [kW/m²]')
        ax_hf.set_title('Body surface heat flux'); ax_hf.legend(fontsize=8)
        ax_hf.grid(alpha=0.25)
        plt.tight_layout()
        _savefig(fig1, save_prefix, "body_heatflux")

        # ────────────────────────────────────────────────────────────────
        # Figure 2: LE stagnation heat flux vs sweep angle  (key plot)
        # Shows clearly how each segment's sweep reduces LE heating
        # ────────────────────────────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        sweeps_cont = np.linspace(0, 85, 200)
        # Reference unswept LE flux
        q_ref_unsw = le_stag_qw_swept(self.M, self.rho_inf, self.T_inf,
                                       self.P_inf, self.V_inf,
                                       self.wing_segs[0].le_radius,
                                       self.T_w, 0.0)['q_le']
        q_cont = np.array([
            le_stag_qw_swept(self.M, self.rho_inf, self.T_inf, self.P_inf,
                              self.V_inf, self.wing_segs[0].le_radius,
                              self.T_w, sw)['q_le']
            for sw in sweeps_cont])

        ax2.plot(sweeps_cont, q_cont / 1e3, 'k-', lw=2, label='Continuous curve (ref LE r)')
        # Mark each segment
        colors_seg = plt.cm.tab10(np.linspace(0, 0.7, len(self.wing_segs)))
        for seg, col, wr in zip(self.wing_segs, colors_seg, wing_results):
            q_le_mean = wr['q_le'].mean()
            ax2.scatter(seg.sweep_le_deg, q_le_mean / 1e3, s=140, zorder=5,
                        color=col, edgecolors='black', linewidths=1.2,
                        label=f"{seg.name}: Λ={seg.sweep_le_deg:.1f}°, "
                              f"q={q_le_mean/1e3:.1f} kW/m²")
            ax2.annotate(seg.name,
                         xy=(seg.sweep_le_deg, q_le_mean / 1e3),
                         xytext=(seg.sweep_le_deg + 1.5, q_le_mean / 1e3 + q_ref_unsw / 1e3 * 0.03),
                         fontsize=8)

        ax2.set_xlabel('Leading-edge sweep angle Λ [degrees]', fontsize=11)
        ax2.set_ylabel('LE stagnation heat flux  [kW/m²]', fontsize=11)
        ax2.set_title(f'Sweep Angle vs. LE Stagnation Heat Flux  –  {title_base}', fontsize=10)
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        _savefig(fig2, save_prefix, "sweep_vs_qLE")

        # ────────────────────────────────────────────────────────────────
        # Figure 3: chordwise heat flux for each segment (lower surface)
        # ────────────────────────────────────────────────────────────────
        ncols = min(len(wing_results), 4)
        nrows = math.ceil(len(wing_results) / ncols)
        fig3, axes3 = plt.subplots(nrows, ncols,
                                    figsize=(5 * ncols, 4 * nrows), squeeze=False)
        fig3.suptitle(f'Wing Segment Chordwise Heat Flux  –  {title_base}', fontsize=11)

        for k, wr in enumerate(wing_results):
            seg  = wr['seg']
            ax   = axes3[k // ncols][k % ncols]
            xc   = wr['x_chord']
            # Plot mid-span slice
            mid  = len(wr['eta']) // 2
            ql_mid = wr['q_lower'][mid] / 1e3
            qu_mid = wr['q_upper'][mid] / 1e3
            q_le_m = wr['q_le'][mid] / 1e3

            ax.plot(xc, ql_mid, 'r-',  lw=2, label=f'Lower  (δ_eff={seg.effective_aoa(self.aoa):+.1f}°)')
            ax.plot(xc, qu_mid, 'b--', lw=2, label=f'Upper')
            ax.axhline(q_le_m, color='purple', ls=':', lw=1.5,
                        label=f'LE stag. {q_le_m:.1f} kW/m²')
            ax.set_xlabel('x/c  (normal to LE)', fontsize=9)
            ax.set_ylabel('q [kW/m²]', fontsize=9)
            ax.set_title(f'{seg.name}  Λ={seg.sweep_le_deg:.1f}°  dih={seg.dihedral_deg:+.1f}°\n'
                         f'M_n={M_inf_normal(self.M,seg.sweep_le_deg):.3f}  '
                         f'cos(Λ)={seg.cos_sweep:.4f}', fontsize=9)
            ax.legend(fontsize=7.5)
            ax.grid(alpha=0.25)

        # Hide unused subplots
        for k in range(len(wing_results), nrows * ncols):
            axes3[k // ncols][k % ncols].set_visible(False)

        plt.tight_layout()
        _savefig(fig3, save_prefix, "wing_chordwise_flux")

        # ────────────────────────────────────────────────────────────────
        # Figure 4: spanwise heat flux (LE + surface peak) per segment
        # ────────────────────────────────────────────────────────────────
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        fig4.suptitle(f'Spanwise LE Stagnation Heat Flux by Segment  –  {title_base}', fontsize=11)

        cumulative_span = 0.0
        cmap4 = plt.cm.Set1(np.linspace(0, 0.7, len(wing_results)))
        for wr, col in zip(wing_results, cmap4):
            seg     = wr['seg']
            eta     = wr['eta']
            span_abs = cumulative_span + eta * seg.span
            ax4.plot(span_abs, wr['q_le'] / 1e3, '-', color=col, lw=2.2,
                     label=f"{seg.name}  Λ={seg.sweep_le_deg:.1f}°")
            ax4.fill_between(span_abs, wr['q_le'] / 1e3, alpha=0.12, color=col)
            # Annotate sweep angle
            mid_span = cumulative_span + 0.5 * seg.span
            ax4.annotate(f"Λ={seg.sweep_le_deg:.1f}°\ndih={seg.dihedral_deg:+.1f}°",
                         xy=(mid_span, wr['q_le'].mean() / 1e3),
                         fontsize=7.5, ha='center',
                         xytext=(0, 10), textcoords='offset points')
            cumulative_span += seg.span

        ax4.set_xlabel('Spanwise position (from fuselage) [m]', fontsize=11)
        ax4.set_ylabel('LE stagnation heat flux  [kW/m²]', fontsize=11)
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        plt.tight_layout()
        _savefig(fig4, save_prefix, "spanwise_LE_flux")

        # ────────────────────────────────────────────────────────────────
        # Figure 5: normal Mach number and cos(Λ) per segment bar chart
        # ────────────────────────────────────────────────────────────────
        fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(11, 5))
        fig5.suptitle(f'Sweep-Angle Decomposition  –  {title_base}', fontsize=11)

        seg_names = [seg.name for seg in self.wing_segs]
        sweeps    = [seg.sweep_le_deg for seg in self.wing_segs]
        cos_sws   = [seg.cos_sweep for seg in self.wing_segs]
        M_ns      = [M_inf_normal(self.M, sw) for sw in sweeps]
        colors_b  = plt.cm.tab10(np.linspace(0, 0.7, len(self.wing_segs)))

        bars_cos = ax5a.bar(seg_names, cos_sws, color=colors_b, edgecolor='k', lw=1.1)
        ax5a.set_ylabel('cos(Λ)  — normal-velocity fraction', fontsize=10)
        ax5a.set_title('cos(sweep) per segment', fontsize=10)
        ax5a.set_ylim(0, 1.05)
        ax5a.grid(axis='y', alpha=0.3)
        for bar, v, sw in zip(bars_cos, cos_sws, sweeps):
            ax5a.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.01,
                       f'cos({sw:.0f}°)={v:.3f}', ha='center', va='bottom', fontsize=8)

        bars_mn = ax5b.bar(seg_names, M_ns, color=colors_b, edgecolor='k', lw=1.1)
        ax5b.axhline(1.0, color='red', ls='--', lw=1.2, label='M_n = 1 (sonic)')
        ax5b.set_ylabel('Normal Mach number  M_n = M·cos(Λ)', fontsize=10)
        ax5b.set_title('Normal Mach number per segment', fontsize=10)
        ax5b.legend(fontsize=9)
        ax5b.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars_mn, M_ns):
            ax5b.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.01,
                       f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        _savefig(fig5, save_prefix, "sweep_decomposition")

        print("\n  All figures generated.")

    def plot_thermal_landscape(self, results, save_prefix=None):
        body  = results['body']
        x     = body['x']
        q_w   = body['q_wind'] / 1e3
        x_fine = np.linspace(0, self.L_body, 500)
        r_fine = np.array([
            self.ogive.radius(xi) if xi <= self.L_ogive else self.R
            for xi in x_fine])
        q_fine = np.interp(x_fine, x, q_w)

        fig, ax = plt.subplots(figsize=(10, 5))
        sc = ax.scatter(x_fine, r_fine, c=q_fine, cmap='plasma', s=20,
                        edgecolors='none', alpha=0.9,
                        vmin=q_fine.min(), vmax=q_fine.max())
        ax.fill_between(x_fine, -r_fine, r_fine, color='lightgray', alpha=0.3)
        ax.plot(x_fine,  r_fine, 'k-', lw=1.5)
        ax.plot(x_fine, -r_fine, 'k-', lw=1.5)
        ax.axvline(self.L_ogive, color='gray', ls='--', lw=1,
                    label=f'Ogive–cylinder junction')
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label('Convective heat flux [kW/m²]', fontsize=10)
        ax.set_xlabel('Axial position x [m]'); ax.set_ylabel('r [m]')
        ax.set_title(f'Body Thermal Landscape  –  {self._title_base()}', fontsize=11)
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        plt.tight_layout()
        _savefig(fig, save_prefix, "thermal_landscape")

    def export_aerothermal_table(self, results, save_prefix=None):
        body  = results['body']
        x_all = body['x']
        q_w   = body['q_wind']
        q_l   = body['q_lee']
        s     = body['s']

        stations = [
            ('Nose tip (near)',               0.001),
            ('Mid ogive',                     self.L_ogive / 2),
            ('Ogive–cylinder junction',       self.L_ogive),
            ('Cylinder quarter',              self.L_ogive + 0.25*(self.L_body-self.L_ogive)),
            ('Cylinder mid',                  self.L_ogive + 0.5 *(self.L_body-self.L_ogive)),
            ('Cylinder aft (tail)',           self.L_body),
        ]

        print("\nLaTeX aerothermal station table:\n")
        print("\\begin{tabular}{lcccc}")
        print("\\hline")
        print("Station & $x$ [m] & $s$ [m] & "
              "$\\dot{q}_\\text{wind}$ [kW/m²] & $\\dot{q}_\\text{lee}$ [kW/m²] \\\\")
        print("\\hline")
        for name, x_target in stations:
            idx = min(np.searchsorted(x_all, x_target), len(x_all)-1)
            print(f"{name} & {x_all[idx]:.2f} & {s[idx]:.2f} & "
                  f"{q_w[idx]/1e3:.1f} & {q_l[idx]/1e3:.1f} \\\\")
        print("\\hline")
        print("\\end{tabular}")

        # Wing LE table
        print("\nLaTeX wing LE heating table:\n")
        print("\\begin{tabular}{lcccc}")
        print("\\hline")
        print("Segment & Λ [°] & cos(Λ) & $M_n$ & "
              "$\\dot{q}_{\\text{LE,mean}}$ [kW/m²] \\\\")
        print("\\hline")
        for wr in results['wing_results']:
            seg = wr['seg']
            print(f"{seg.name} & {seg.sweep_le_deg:.2f} & {seg.cos_sweep:.4f} & "
                  f"{M_inf_normal(self.M,seg.sweep_le_deg):.3f} & "
                  f"{wr['q_le'].mean()/1e3:.2f} \\\\")
        print("\\hline")
        print("\\end{tabular}")

        if save_prefix:
            fname = f"{save_prefix}_aerothermal_table.csv"
            with open(fname, 'w') as f:
                f.write("Station,x [m],s [m],q_wind [kW/m2],q_lee [kW/m2]\n")
                for name, x_target in stations:
                    idx = min(np.searchsorted(x_all, x_target), len(x_all)-1)
                    f.write(f"{name},{x_all[idx]:.3f},{s[idx]:.3f},"
                            f"{q_w[idx]/1e3:.2f},{q_l[idx]/1e3:.2f}\n")
            with open(f"{save_prefix}_wing_LE_table.csv", 'w') as f:
                f.write("Segment,sweep_deg,cos_sweep,M_n,q_LE_mean_kW_m2\n")
                for wr in results['wing_results']:
                    seg = wr['seg']
                    f.write(f"{seg.name},{seg.sweep_le_deg:.2f},{seg.cos_sweep:.4f},"
                            f"{M_inf_normal(self.M,seg.sweep_le_deg):.3f},"
                            f"{wr['q_le'].mean()/1e3:.2f}\n")
            print(f"\n  Saved CSV tables: {save_prefix}_aerothermal_table.csv, "
                  f"{save_prefix}_wing_LE_table.csv")

    def _title_base(self):
        return (f"M={self.M:.1f}, h={self.alt:.0f} km, AoA={self.aoa:.1f}°, "
                f"{'Turb.' if self.turb else 'Lam.'} BL, Tw={self.T_w:.0f} K")


# ── Utility helpers ───────────────────────────────────────────────────────────

def M_inf_normal(M_inf: float, sweep_deg: float) -> float:
    """Normal-to-LE Mach number M_n = M · cos(Λ)."""
    return M_inf * math.cos(math.radians(sweep_deg))


def _savefig(fig, prefix, suffix):
    if prefix:
        fname = f"{prefix}_{suffix}.png"
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build WingSegment objects from table data
    segments = [
        WingSegment(*row) for row in WING_SEGMENTS
    ]

    vehicle = OgiveCylinderAnalysis(
        mach           = MACH,
        altitude_km    = ALTITUDE_KM,
        wall_temp_k    = WALL_TEMP_K,
        is_turbulent   = IS_TURBULENT,
        aoa_deg        = AOA_DEG,
        body_radius_m  = BODY_RADIUS_M,
        ogive_length_m = OGIVE_LENGTH_M,
        body_length_m  = BODY_LENGTH_M,
        nose_radius_m  = NOSE_RADIUS_M,
        wing_segments  = segments,
    )

    results = vehicle.run(num_points=NUM_POINTS,
                          n_chord=60, n_span=NUM_SPAN_POINTS)

    prefix = FIGURE_PREFIX if SAVE_FIGURES else None
    vehicle.plot(results, save_prefix=prefix)
    vehicle.plot_thermal_landscape(results, save_prefix=prefix)
    vehicle.export_aerothermal_table(results, save_prefix=prefix)
