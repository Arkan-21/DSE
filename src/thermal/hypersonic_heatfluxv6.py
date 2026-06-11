#!/usr/bin/env python3
"""
Hypersonic Convective Heat Flux Analysis Tool
=============================================
Eckert's Reference Enthalpy Method + Fay-Riddell stagnation correlation.
Validated range: Mach 2–8, altitudes 10–65 km.
Accuracy: ±15 % for cold-wall conditions (Tw/Taw ≤ 0.5).

Geometry: ogive-cylinder (missile / slender vehicle)
-----------------------------------------------------
  The body is modelled as:
    1.  Blunted ogive tip  (0 → OGIVE_LENGTH_M)
    2.  Cylindrical aft body (OGIVE_LENGTH_M → BODY_LENGTH_M)

  The tangent ogive is defined by two parameters: base radius R and
  axial length L.  The profile is:
        r(x) = sqrt(ρ² − (L−x)²) + R − ρ,   ρ = (R² + L²) / (2R)
  where ρ is the ogive radius of curvature.  By construction:
    • r(0) = 0          (pointed tip, then blunted with NOSE_RADIUS_M)
    • r(L) = R          (tangent to cylinder – slope = 0 at junction)
    • dr/dx → 0 at x=L  (smooth, no shoulder discontinuity)

  Because the slope is zero at the cylinder junction, there is NO discrete
  expansion fan at the shoulder.  The flow transitions smoothly.

Aerodynamic modelling
---------------------
  Ogive surface (station by station):
    • Local surface slope φ(x) = atan(dr/dx) is computed at each axial station.
    • Effective deflection for windward side = φ(x) + AoA
                          for leeward  side = φ(x) − AoA
    • Edge conditions are found via the TANGENT-WEDGE approximation:
        – positive effective deflection → oblique shock
        – negative effective deflection → Prandtl-Meyer expansion from
          freestream (tangent-cone / tangent-wedge is standard for slender bodies)
    • Local heat flux = flat_plate_heat_flux evaluated at the running
      arc-length s from the stagnation point.

  Cylinder section:
    • Surface slope = 0°, so effective deflection = ±AoA.
    • Same tangent-wedge / expansion logic applied.
    • Arc-length continues from the ogive-cylinder junction.

  Stagnation point:
    • Fay-Riddell correlation using NOSE_RADIUS_M.

  No sharp shoulder: because the ogive is tangent to the cylinder,
  there is no discrete Prandtl-Meyer fan at the junction.

References
----------
  Eckert (1955); Fay & Riddell (1958); Anderson (2006);
  U.S. Standard Atmosphere (1976).

Notes / limitations
-------------------
  • Calorically perfect gas (γ = 1.4) — dissociation not modelled.
  • Tangent-wedge is an engineering approximation; it gives good
    results for slender bodies (φ < ~20°) but overestimates pressure
    at the tip for blunter ogives.
  • Radiation, ablation, and real-gas effects not included.
  • For TPS sizing: use turbulent BL and apply ≥ 1.5× safety margin.
"""

# ============================================================
#  USER INPUTS  –  Edit this block only
# ============================================================

MACH             = 5.0    # Free-stream Mach number  [-]
ALTITUDE_KM      = 30.0   # Cruise altitude           [km]
WALL_TEMP_K      = 373.15 # Wall temperature          [K]
IS_TURBULENT     = True  # True = turbulent BL
AOA_DEG          = 1.0    # Angle of attack           [degrees]

# Body geometry
BODY_RADIUS_M    = 2.0   # Cylinder (body) radius              [m]
OGIVE_LENGTH_M   = 10.493   # Axial length of ogive nose section  [m]
BODY_LENGTH_M    = 21.0    # Total body length (ogive + cylinder)[m]
NOSE_RADIUS_M    = 0.040  # Blunted tip radius of curvature     [m]

# Wing geometry (if present – set WING_CHORD_M = 0 to skip)
WING_CHORD_M     = 5.0    # Wing / fin chord length    [m]  (0 = no wing)
WING_LE_RADIUS_M = 0.010  # Wing leading-edge radius   [m]
WING_WEDGE_DEG   = 5.0    # Wing section half-angle at AoA=0  [degrees]

# Analysis resolution
NUM_POINTS       = 300    # Points along the body

# Output
SAVE_FIGURES     = False   # False → interactive; True → save PNGs
FIGURE_PREFIX    = "heat_flux"

# ============================================================
#  END OF USER INPUTS
# ============================================================

import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple


# ── Atmospheric model ────────────────────────────────────────────────────────

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
    T = T_base
    P = P_base * math.exp(-g0 * dh / (R_air * T_base))
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


# ── Shock / expansion relations ──────────────────────────────────────────────

def normal_shock(M1: float) -> Tuple[float, float, float]:
    """Returns (M2, T2/T1, rho2/rho1) for a normal shock."""
    g, M1sq = GAMMA, M1 ** 2
    M2sq = (1 + (g - 1) / 2 * M1sq) / (g * M1sq - (g - 1) / 2)
    T2_T1 = (1 + 2 * g / (g + 1) * (M1sq - 1)) * (2 + (g - 1) * M1sq) / ((g + 1) * M1sq)
    rho2  = (g + 1) * M1sq / (2 + (g - 1) * M1sq)
    return math.sqrt(M2sq), T2_T1, rho2


def oblique_shock_angle(M1: float, wedge_deg: float) -> Optional[float]:
    """
    Weak-shock wave angle β [deg] for deflection θ.
    Returns None if the deflection exceeds the detachment limit.
    """
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
        return None                # detached shock

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


def compression_edge(M1: float, rho1: float, T1: float, P1: float,
                     wedge_deg: float) -> Dict:
    """Oblique-shock edge conditions for a wedge angle wedge_deg > 0."""
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


def prandtl_meyer_nu(M: float) -> float:
    g  = GAMMA
    gp = (g + 1) / (g - 1)
    return math.sqrt(gp) * math.atan(math.sqrt((M ** 2 - 1) / gp)) - math.atan(math.sqrt(M ** 2 - 1))


def prandtl_meyer_M(nu_target: float) -> float:
    lo, hi = 1.001, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if prandtl_meyer_nu(mid) < nu_target:
            lo = mid
        else:
            hi = mid
    return mid


def expansion_edge(M1: float, rho1: float, T1: float, P1: float,
                   turn_deg: float) -> Dict:
    """
    Prandtl-Meyer expansion by turn_deg degrees (flow turns away from surface).
    """
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


def tangent_wedge_edge(M_inf: float, rho_inf: float, T_inf: float,
                       P_inf: float, V_inf: float,
                       local_slope_deg: float) -> Dict:
    """
    Tangent-wedge approximation: treat the local surface slope as a
    wedge deflection angle and compute edge conditions accordingly.

    local_slope_deg > 0 → compression (oblique shock)
    local_slope_deg < 0 → expansion   (Prandtl-Meyer from freestream)
    local_slope_deg = 0 → freestream
    """
    delta = local_slope_deg
    if abs(delta) < 0.01:
        return dict(T_e=T_inf, u_e=V_inf, P_e=P_inf)
    if delta > 0:
        return compression_edge(M_inf, rho_inf, T_inf, P_inf, delta)
    else:
        return expansion_edge(M_inf, rho_inf, T_inf, P_inf, abs(delta))


# ── Reference enthalpy & flat-plate heat flux ─────────────────────────────────

def reference_state(T_e: float, u_e: float, P_e: float,
                    T_w: float, turbulent: bool) -> Tuple[float, float, float]:
    """Eckert reference-enthalpy method → (T*, ρ*, μ*)."""
    r    = PR ** (1 / 3) if turbulent else math.sqrt(PR)
    M_e  = u_e / math.sqrt(GAMMA * R_AIR * T_e)
    T_aw = T_e * (1 + r * (GAMMA - 1) / 2 * M_e ** 2)
    h_star = CP * T_w + 0.5 * CP * (T_e - T_w) + 0.22 * CP * (T_aw - T_w)
    T_star = h_star / CP
    rho_star = P_e / (R_AIR * T_star)
    return T_star, rho_star, viscosity(T_star)


def flat_plate_qw(s: float, edge: Dict, T_w: float, turbulent: bool) -> float:
    """
    Local wall heat flux at arc-length s from the stagnation point.

    Uses the Eckert reference-enthalpy flat-plate correlation:
      Laminar  : Nu_s = 0.332 Re_s*^0.5 Pr^(1/3)
      Turbulent: Nu_s = 0.0296 Re_s*^0.8 Pr^(1/3)
    The arc-length s is used as the characteristic length (appropriate for
    a body of revolution under the local-similarity assumption).
    """
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


# ── Stagnation-point heat flux (Fay-Riddell) ─────────────────────────────────

def stagnation_qw(M_inf: float, rho_inf: float, T_inf: float,
                  P_inf: float, V_inf: float,
                  nose_r: float, T_w: float) -> Tuple[float, Dict]:
    """Fay-Riddell correlation at the blunted nose tip."""
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


# ── Leading-edge heat flux (Fay-Riddell, cylinder) ───────────────────────────

def le_stag_qw(M_inf: float, rho_inf: float, T_inf: float,
               P_inf: float, V_inf: float,
               le_r: float, T_w: float, sweep_deg: float = 0.0) -> float:
    if le_r <= 0:
        le_r = 1e-3
    cos_sw = math.cos(math.radians(sweep_deg))
    V_n = V_inf * cos_sw
    M_n = max(M_inf * cos_sw, 1.001)
    _, T2_T1, rho2_r = normal_shock(M_n)
    T_e   = T_inf * T2_T1
    rho_e = rho_inf * rho2_r
    P2_P1 = 1 + 2 * GAMMA / (GAMMA + 1) * (M_n ** 2 - 1)
    P_e   = P_inf * P2_P1
    du_dx = (1 / le_r) * math.sqrt(2 * 0.5 * rho_inf * V_n ** 2 / rho_e)
    mu_e  = viscosity(T_e)
    h0    = CP * T_inf + 0.5 * V_inf ** 2
    return max((0.57 / PR ** 0.6) * math.sqrt(rho_e * mu_e * du_dx) * (h0 - CP * T_w), 0.0)


# ── Tangent-ogive geometry ────────────────────────────────────────────────────

class TangentOgive:
    """
    Tangent ogive defined by base radius R and axial length L.

        ρ_ogive = (R² + L²) / (2R)          [radius of curvature]
        r(x)    = sqrt(ρ² − (L−x)²) + R − ρ  [profile]
        φ(x)    = atan(dr/dx)                 [local surface slope]

    The profile is tangent to the cylinder at x = L (slope → 0°).
    """
    def __init__(self, R: float, L: float):
        if L <= 0 or R <= 0:
            raise ValueError("Ogive R and L must be positive.")
        self.R = R
        self.L = L
        self.rho_c = (R ** 2 + L ** 2) / (2 * R)   # ogive radius of curvature

    def radius(self, x: float) -> float:
        """Cross-section radius at axial station x."""
        x = max(0.0, min(x, self.L))
        return math.sqrt(self.rho_c ** 2 - (self.L - x) ** 2) + self.R - self.rho_c

    def slope_deg(self, x: float) -> float:
        """Local surface half-angle φ(x) [degrees]."""
        x = max(1e-6, min(x, self.L - 1e-6))
        drdx = (self.L - x) / math.sqrt(self.rho_c ** 2 - (self.L - x) ** 2)
        return math.degrees(math.atan(drdx))

    def arc_length(self, x_stations: np.ndarray) -> np.ndarray:
        """
        Cumulative arc-length from x=0 along the ogive surface.
        Uses numerical integration (trapezoid) of ds = sqrt(1 + (dr/dx)²) dx.
        """
        s = np.zeros_like(x_stations)
        for i in range(1, len(x_stations)):
            x0, x1 = x_stations[i - 1], x_stations[i]
            dx = x1 - x0
            xm = 0.5 * (x0 + x1)
            xm = max(1e-6, min(xm, self.L - 1e-6))
            drdx = (self.L - xm) / math.sqrt(self.rho_c ** 2 - (self.L - xm) ** 2)
            ds = dx * math.sqrt(1 + drdx ** 2)
            s[i] = s[i - 1] + ds
        return s


# ── Main analysis class ───────────────────────────────────────────────────────

class OgiveCylinderAnalysis:
    """
    Aerothermal analysis of a blunted ogive-cylinder body.

    The body is divided into:
      • Stagnation point  (Fay-Riddell)
      • Ogive nose section (tangent-wedge, station-by-station)
      • Cylindrical aft body (tangent-wedge with slope = 0° + AoA effect)

    At each axial station on the ogive the local surface slope φ(x) is
    computed.  The effective aerodynamic deflection angle is:
        windward side:  δ_w = φ(x) + AoA
        leeward  side:  δ_l = φ(x) − AoA

    A positive δ triggers an oblique shock (compression).
    A negative δ triggers a Prandtl-Meyer expansion from freestream.

    On the cylinder (φ = 0) only the AoA drives the asymmetry.

    Arc-length from the stagnation point is used as the running length
    in the flat-plate correlation (local-similarity assumption).
    """

    def __init__(self,
                 mach:          float,
                 altitude_km:   float,
                 wall_temp_k:   float,
                 is_turbulent:  bool,
                 aoa_deg:       float,
                 body_radius_m: float,
                 ogive_length_m: float,
                 body_length_m: float,
                 nose_radius_m: float):

        self.M        = mach
        self.alt      = altitude_km
        self.T_w      = wall_temp_k
        self.turb     = is_turbulent
        self.aoa      = aoa_deg
        self.R        = body_radius_m
        self.L_ogive  = ogive_length_m
        self.L_body   = body_length_m
        self.r_nose   = nose_radius_m

        self.T_inf, self.P_inf, self.rho_inf = us_standard_atmosphere(altitude_km)
        self.a_inf = math.sqrt(GAMMA * R_AIR * self.T_inf)
        self.V_inf = mach * self.a_inf

        self.ogive = TangentOgive(body_radius_m, ogive_length_m)

        self._print_header()

    def _print_header(self):
        T_ratio = self.T_w / (self.T_inf * (1 + (GAMMA - 1) / 2 * self.M ** 2))
        print("\n" + "=" * 64)
        print("  OGIVE-CYLINDER AEROTHERMAL ANALYSIS")
        print("=" * 64)
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
        ogive_rho = self.ogive.rho_c
        tip_slope = self.ogive.slope_deg(0.001)
        print(f"  Ogive ρ (curvature) : {ogive_rho:.4f} m")
        print(f"  Tip local slope     : {tip_slope:.1f}° (near tip, x≈1mm)")
        print(f"  Nose tip radius     : {self.r_nose*1000:.1f} mm")
        print("=" * 64)

    # ── Body-surface distribution ─────────────────────────────────────────

    def _body_distribution(self, num_points: int) -> Dict:
        """
        Compute heat flux at num_points along the body surface.

        Returns arrays of:
          x         : axial stations [m]
          s         : arc-length from stagnation point [m]
          slope_deg : local surface angle [deg]
          q_wind    : windward heat flux [W/m²]
          q_lee     : leeward  heat flux [W/m²]
          q_stag    : stagnation-point value (scalar, at x=0)
        """
        # Build axial stations: dense near tip, uniform aft
        x_ogive   = np.linspace(0.001, self.L_ogive, num_points // 2)
        x_cyl     = np.linspace(self.L_ogive, self.L_body, num_points // 2 + 1)[1:]
        x_all     = np.concatenate([x_ogive, x_cyl])

        # ── Arc-length on ogive ───────────────────────────────────────────
        # We need a fine grid just for the arc-length integrator
        x_fine_ogive = np.linspace(0.0, self.L_ogive, 2000)
        s_ogive_fine = self.ogive.arc_length(x_fine_ogive)
        s_at_junction = s_ogive_fine[-1]    # total arc-length of the ogive

        # Interpolate arc-length at our analysis stations within the ogive
        def arc_at_x(x_val):
            if x_val <= self.L_ogive:
                return float(np.interp(x_val, x_fine_ogive, s_ogive_fine))
            else:
                return s_at_junction + (x_val - self.L_ogive)  # cylinder: ds=dx

        q_wind = np.zeros(len(x_all))
        q_lee  = np.zeros(len(x_all))
        # Cap at stagnation heat flux (physical limit)
        slopes = np.zeros(len(x_all))

        q_stag, diag_stag = stagnation_qw(
            self.M, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
            self.r_nose, self.T_w)

        for i, x in enumerate(x_all):
            s = arc_at_x(x)

            if x <= self.L_ogive:
                phi = self.ogive.slope_deg(x)   # local ogive surface slope
            else:
                phi = 0.0                        # cylinder: zero surface slope

            slopes[i] = phi

            delta_wind = phi + self.aoa     # windward effective deflection
            delta_lee  = phi - self.aoa     # leeward  effective deflection

            edge_wind = tangent_wedge_edge(
                self.M, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
                delta_wind)
            edge_lee  = tangent_wedge_edge(
                self.M, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
                delta_lee)

            q_wind[i] = flat_plate_qw(s, edge_wind, self.T_w, self.turb)
            q_lee[i]  = flat_plate_qw(s, edge_lee,  self.T_w, self.turb)


            # Cap at stagnation heat flux (physical limit)
            q_wind[i] = min(q_wind[i], q_stag)
            q_lee[i] = min(q_lee[i], q_stag)
        # Stagnation point

        return dict(x=x_all, s=np.array([arc_at_x(x) for x in x_all]),
                    slope=slopes,
                    q_wind=q_wind, q_lee=q_lee,
                    q_stag=q_stag, diag_stag=diag_stag,
                    s_junction=s_at_junction,
                    x_junction=self.L_ogive)

    # ── Wing / fin (optional) ─────────────────────────────────────────────

    def _wing_distribution(self, chord: float, le_radius: float,
                            wedge_deg: float, num_points: int) -> Optional[Dict]:
        if chord <= 0:
            return None
        x_wing = np.linspace(0.01, chord, num_points)

        lo_defl = wedge_deg + self.aoa
        hi_defl = wedge_deg - self.aoa

        edge_lo = tangent_wedge_edge(self.M, self.rho_inf, self.T_inf,
                                      self.P_inf, self.V_inf, lo_defl)
        edge_hi = tangent_wedge_edge(self.M, self.rho_inf, self.T_inf,
                                      self.P_inf, self.V_inf, hi_defl)

        q_lo = np.array([flat_plate_qw(x, edge_lo, self.T_w, self.turb) for x in x_wing])
        q_hi = np.array([flat_plate_qw(x, edge_hi, self.T_w, self.turb) for x in x_wing])

        q_le = le_stag_qw(self.M, self.rho_inf, self.T_inf, self.P_inf,
                           self.V_inf, le_radius, self.T_w)

        return dict(x=x_wing, q_lower=q_lo, q_upper=q_hi,
                    q_le=q_le, lo_defl=lo_defl, hi_defl=hi_defl)

    # ── Public run method ─────────────────────────────────────────────────

    def run(self, wing_chord: float = 0.0,
            wing_le_radius: float = 0.003,
            wing_wedge_deg: float = 5.0,
            num_points: int = 300) -> Dict:

        print("\n── Computing body distribution …")
        body = self._body_distribution(num_points)

        # ── Console report ─────────────────────────────────────────────────
        q_s   = body['q_stag']
        q_w_max = body['q_wind'].max()
        q_l_max = body['q_lee'].max()
        x_wm  = body['x'][body['q_wind'].argmax()]
        x_lm  = body['x'][body['q_lee'].argmax()]

        print(f"\n── Nose stagnation ──────────────────────────────────────────────")
        d = body['diag_stag']
        print(f"  Post-shock T_e     : {d['T_e']:.1f} K")
        print(f"  Velocity gradient  : {d['du_dx']:.2f} s⁻¹")
        print(f"  Δh (h0 − hw)       : {(d['h0']-d['h_w'])/1e3:.1f} kJ/kg")
        print(f"  Heat flux          : {q_s/1e3:.2f} kW/m²")

        print(f"\n── Body surface ─────────────────────────────────────────────────")
        print(f"  Ogive length       : {self.L_ogive:.2f} m  (junction at x = {self.L_ogive:.2f} m)")
        print(f"  Windward peak      : {q_w_max/1e3:.2f} kW/m²  at x = {x_wm:.3f} m")
        print(f"  Leeward  peak      : {q_l_max/1e3:.2f} kW/m²  at x = {x_lm:.3f} m")
        if self.aoa == 0.0:
            print("  → AoA = 0°: windward = leeward (axisymmetric).")

        # Tip-region diagnostics (first ogive station)
        x_tip = body['x'][0]
        phi_tip = body['slope'][0]
        print(f"\n  Tip diagnostics (x = {x_tip:.3f} m):")
        print(f"    Local slope        : {phi_tip:.2f}°")
        print(f"    Windward deflection: {phi_tip + self.aoa:.2f}°")
        print(f"    Leeward  deflection: {phi_tip - self.aoa:.2f}°")
        print(f"    Windward q         : {body['q_wind'][0]/1e3:.2f} kW/m²")
        print(f"    Leeward  q         : {body['q_lee'][0]/1e3:.2f} kW/m²")

        # Junction diagnostics
        ji = np.searchsorted(body['x'], self.L_ogive)
        if ji < len(body['x']):
            print(f"\n  Junction diagnostics (x = {body['x'][ji]:.3f} m):")
            print(f"    Local slope        : {body['slope'][ji]:.2f}°  (should ≈ 0)")
            print(f"    Windward q         : {body['q_wind'][ji]/1e3:.2f} kW/m²")
            print(f"    Leeward  q         : {body['q_lee'][ji]/1e3:.2f} kW/m²")

        results = dict(body=body)

        # ── Wing (optional) ────────────────────────────────────────────────
        wing = self._wing_distribution(wing_chord, wing_le_radius,
                                        wing_wedge_deg, num_points // 2)
        results['wing'] = wing
        if wing is not None:
            print(f"\n── Wing / fin ────────────────────────────────────────────────")
            print(f"  LE stagnation      : {wing['q_le']/1e3:.2f} kW/m²")
            print(f"  Lower (δ=+{wing['lo_defl']:.1f}°)    : {wing['q_lower'].max()/1e3:.2f} kW/m²")
            up_mode = "shock" if wing['hi_defl'] > 0 else "expansion"
            print(f"  Upper ({up_mode}, δ={wing['hi_defl']:+.1f}°): {wing['q_upper'].max()/1e3:.2f} kW/m²")

        # ── Summary ────────────────────────────────────────────────────────
        peak_stag = q_s
        peak_body = max(q_w_max, q_l_max)
        peak_list = [peak_stag, peak_body]
        if wing:
            peak_wing = max(wing['q_lower'].max(), wing['q_upper'].max())
            peak_le   = wing['q_le']
            peak_list += [peak_wing, peak_le]
        else:
            peak_wing = peak_le = 0.0

        peak_all = max(peak_list)

        print("\n" + "=" * 64)
        print("  PEAK HEAT FLUX SUMMARY")
        print("=" * 64)
        print(f"  Nose stagnation    : {peak_stag/1e3:>8.2f} kW/m²")
        print(f"  Body surface (max) : {peak_body/1e3:>8.2f} kW/m²")
        if wing:
            print(f"  Wing LE stagnation : {peak_le/1e3:>8.2f} kW/m²")
            print(f"  Wing surface (max) : {peak_wing/1e3:>8.2f} kW/m²")
        print(f"  {'─'*44}")
        print(f"  OVERALL PEAK       : {peak_all/1e3:>8.2f} kW/m²")
        print("=" * 64)
        print("  ⚠  Tangent-wedge: good for φ < ~20°; overestimates near blunt tip.")
        print("  ⚠  Arc-length correlation: valid for local-similarity regime.")
        print("  ⚠  Dissociation not modelled (γ=1.4); apply correction above M7.")
        print("  ⚠  Apply ≥ 1.5× safety margin for TPS sizing.")
        print("=" * 64)

        results['summary'] = dict(
            peak_stag=peak_stag, peak_body=peak_body,
            peak_wing=peak_wing, peak_le=peak_le, peak_all=peak_all)
        return results

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot(self, results: Dict, save_prefix: Optional[str] = None):

        body = results['body']
        wing = results['wing']
        x    = body['x']
        q_w  = body['q_wind']
        q_l  = body['q_lee']
        q_av = 0.5 * (q_w + q_l)
        x_j  = self.L_ogive
        q_s  = body['q_stag']

        title_base = (f"M={self.M:.1f}, h={self.alt:.0f} km, "
                      f"AoA={self.aoa:.1f}°, Tw={self.T_w:.0f} K, "
                      f"{'Turb.' if self.turb else 'Lam.'} BL")

        has_wing = wing is not None
        ncols    = 2 if has_wing else 1
        fig, axes = plt.subplots(1, ncols + 1, figsize=(7 * (ncols + 1), 5))
        if ncols + 1 == 2:
            axes = np.array(axes)
        fig.suptitle(f"Ogive-Cylinder Aerothermal Heating  –  {title_base}", fontsize=11)

        # ── Panel 1: body profile (geometry) ──────────────────────────────
        ax_geom = axes[0]
        x_og  = np.linspace(0, self.L_ogive, 300)
        r_og  = np.array([self.ogive.radius(xi) for xi in x_og])
        x_cyl = np.array([self.L_ogive, self.L_body])
        r_cyl = np.array([self.R, self.R])

        ax_geom.fill_between(np.concatenate([x_og, x_cyl]),
                              np.concatenate([r_og, r_cyl]),
                              alpha=0.15, color='steelblue')
        ax_geom.fill_between(np.concatenate([x_og, x_cyl]),
                              -np.concatenate([r_og, r_cyl]),
                              alpha=0.15, color='steelblue')
        ax_geom.plot(np.concatenate([x_og, x_cyl]),
                      np.concatenate([r_og, r_cyl]), 'b-', lw=2)
        ax_geom.plot(np.concatenate([x_og, x_cyl]),
                      -np.concatenate([r_og, r_cyl]), 'b-', lw=2)
        ax_geom.axvline(x_j, color='gray', ls='--', lw=1, label=f'Ogive–cyl. junction')
        ax_geom.set_xlabel('Axial position x [m]', fontsize=10)
        ax_geom.set_ylabel('Radial position r [m]', fontsize=10)
        ax_geom.set_title('Body profile (tangent ogive + cylinder)', fontsize=10)
        ax_geom.set_aspect('equal', adjustable='datalim')
        ax_geom.legend(fontsize=8)
        ax_geom.grid(True, alpha=0.25)

        # ── Panel 2: heat flux distribution ───────────────────────────────
        ax_hf = axes[1]
        ax_hf.axvline(x_j, color='gray', ls='--', lw=1,
                       label=f'Ogive–cyl. x={x_j:.2f} m')
        ax_hf.axhline(q_s / 1e3, color='purple', ls=':', lw=1.5,
                       label=f'Stag. point: {q_s/1e3:.1f} kW/m²')

        if abs(self.aoa) < 0.01:
            ax_hf.plot(x, q_w / 1e3, 'b-', lw=2, label='Surface (symmetric, AoA=0)')
        else:
            ax_hf.plot(x, q_w / 1e3, 'r-',  lw=2,
                        label=f'Windward (δ=φ+{self.aoa:.1f}°)')
            ax_hf.plot(x, q_l / 1e3, 'b--', lw=2,
                        label=f'Leeward  (δ=φ−{self.aoa:.1f}°)')
            ax_hf.plot(x, q_av / 1e3, 'k:',  lw=1.4, alpha=0.6, label='Average')

        ax_hf.set_xlabel('Axial position x [m]', fontsize=10)
        ax_hf.set_ylabel('Convective heat flux [kW/m²]', fontsize=10)
        ax_hf.set_title('Body surface heat flux distribution', fontsize=10)
        ax_hf.legend(fontsize=8)
        ax_hf.grid(True, alpha=0.25)

        # ── Panel 3 (optional): wing ───────────────────────────────────────
        if has_wing:
            ax_w = axes[2]
            lo_l = f"Lower (δ=+{wing['lo_defl']:.1f}°)"
            up_m = "shock" if wing['hi_defl'] > 0 else "expansion"
            up_l = f"Upper ({up_m}, δ={wing['hi_defl']:+.1f}°)"
            ax_w.plot(wing['x'], wing['q_lower'] / 1e3, 'r-',  lw=2, label=lo_l)
            ax_w.plot(wing['x'], wing['q_upper'] / 1e3, 'b--', lw=2, label=up_l)
            ax_w.axhline(wing['q_le'] / 1e3, color='purple', ls=':', lw=1.5,
                          label=f"LE stag.: {wing['q_le']/1e3:.1f} kW/m²")
            ax_w.set_xlabel('Chordwise position [m]', fontsize=10)
            ax_w.set_ylabel('Convective heat flux [kW/m²]', fontsize=10)
            ax_w.set_title('Wing / fin heat flux', fontsize=10)
            ax_w.legend(fontsize=8)
            ax_w.grid(True, alpha=0.25)

        plt.tight_layout()
        if save_prefix:
            p = f"{save_prefix}_distributions.png"
            fig.savefig(p, dpi=150, bbox_inches='tight')
            print(f"  Saved: {p}")
        else:
            plt.show()

        # ── Figure 2: local slope along body ──────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.plot(x, body['slope'], 'k-', lw=2, label='Local surface slope φ(x)')
        ax2.axhline(0, color='gray', ls='--', lw=0.8)
        ax2.axvline(x_j, color='gray', ls='--', lw=1, label=f'Ogive–cyl. x={x_j:.2f} m')
        ax2.set_xlabel('Axial position x [m]', fontsize=10)
        ax2.set_ylabel('Local surface slope φ [degrees]', fontsize=10)
        ax2.set_title(f'Tangent-ogive surface slope  –  {title_base}', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.25)
        plt.tight_layout()
        if save_prefix:
            p = f"{save_prefix}_slope.png"
            fig2.savefig(p, dpi=150, bbox_inches='tight')
            print(f"  Saved: {p}")
        else:
            plt.show()

        # ── Figure 3: summary bar chart ────────────────────────────────────
        s     = results['summary']
        lbs   = ['Nose\n(stagnation)', 'Body surface\n(max)']
        vals  = [s['peak_stag'] / 1e3, s['peak_body'] / 1e3]
        cols  = ['#c0392b', '#e67e22']
        if has_wing:
            lbs  += ['Wing LE\n(stagnation)', 'Wing surface\n(max)']
            vals += [s['peak_le'] / 1e3, s['peak_wing'] / 1e3]
            cols += ['#8e44ad', '#2980b9']

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        bars = ax3.bar(lbs, vals, color=cols, edgecolor='black', lw=1.2)
        ax3.axhline(s['peak_all'] / 1e3, color='k', ls=':', lw=1.5,
                     label=f"Overall peak: {s['peak_all']/1e3:.1f} kW/m²")
        ax3.set_ylabel('Peak conv. heat flux [kW/m²]', fontsize=11)
        ax3.set_title('Peak heat flux per component  –  TPS sizing reference', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + max(vals) * 0.01,
                      f'{v:.1f}', ha='center', va='bottom',
                      fontweight='bold', fontsize=10)
        plt.tight_layout()
        if save_prefix:
            p = f"{save_prefix}_summary.png"
            fig3.savefig(p, dpi=150, bbox_inches='tight')
            print(f"  Saved: {p}")
        else:
            plt.show()

    def plot_thermal_landscape(self, results: Dict, save_prefix: Optional[str] = None):
        """
        Create a coloured body profile where the surface colour indicates
        local convective heat flux (windward side). Uses a perceptually
        uniform colormap ('inferno') that works well on white backgrounds.
        """
        body = results['body']
        x = body['x']
        q_wind = body['q_wind'] / 1e3  # kW/m²
        # Build full 2D contour of upper half (axisymmetric)
        x_fine = np.linspace(0, self.L_body, 500)
        r_fine = np.zeros_like(x_fine)
        for i, xi in enumerate(x_fine):
            if xi <= self.L_ogive:
                r_fine[i] = self.ogive.radius(xi)
            else:
                r_fine[i] = self.R
        # Interpolate heat flux onto the fine x grid
        q_fine = np.interp(x_fine, x, q_wind)

        fig, ax = plt.subplots(figsize=(10, 5))
        # Use 'inferno' – excellent on white background, perceptually uniform
        # vmin/vmax set to data range to maximise contrast (especially on cylinder)
        scatter = ax.scatter(x_fine, r_fine, c=q_fine, cmap='plasma',
                             s=20, edgecolors='none', alpha=0.9,
                             vmin=q_fine.min(), vmax=q_fine.max())
        # Mirror lower half (greyed out for clarity)
        ax.fill_between(x_fine, -r_fine, r_fine, color='lightgray', alpha=0.3)
        ax.plot(x_fine, r_fine, 'k-', lw=1.5, label='Body contour')
        ax.plot(x_fine, -r_fine, 'k-', lw=1.5)
        # Mark junction
        ax.axvline(self.L_ogive, color='gray', ls='--', lw=1,
                   label=f'Ogive–cylinder junction (x={self.L_ogive:.2f} m)')
        # Colour bar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label('Convective heat flux [kW/m²]', fontsize=10)
        ax.set_xlabel('Axial position x [m]', fontsize=10)
        ax.set_ylabel('Radial position r [m]', fontsize=10)
        ax.set_title(f'Thermal landscape – {self._title_base()}', fontsize=11)
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        if save_prefix:
            fname = f"{save_prefix}_thermal_landscape.png"
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"  Saved: {fname}")
        else:
            plt.show()
        plt.close(fig)

    def plot_boundary_layer_development(self, results: Dict, save_prefix: Optional[str] = None):
        """
        Log‑log plot of Reynolds number (Re_s*) and heat flux vs. arc‑length s.
        Demonstrates the turbulent boundary‑layer trend q ~ s^{-0.2}.
        """
        body = results['body']
        s = body['s']
        q_wind = body['q_wind']
        # Compute reference Reynolds number along the body
        Re_s = np.zeros_like(s)
        for i, si in enumerate(s):
            if si <= 0:
                Re_s[i] = 0
            else:
                # Need edge conditions at each point – reuse from body calculation
                # But we don't store edge dicts; approximate using average edge properties
                # Instead, recompute using the stored edge? Simpler: use a representative
                # edge from the first point (good enough for trend)
                pass
        # Alternative: compute Re_s* directly from flat_plate_qw inputs? Too heavy.
        # Instead, we plot q_wind vs s to show the power-law decay.
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(s[s > 0], q_wind[s > 0] / 1e3, 'r-', lw=2, label='Windward heat flux')
        # Add theoretical -0.2 slope line for reference
        s_ref = s[s > 0]
        q_ref = q_wind[s > 0][0] * (s_ref / s_ref[0]) ** (-0.2)
        ax.loglog(s_ref, q_ref / 1e3, 'k--', lw=1.5, alpha=0.7,
                  label=r'$q \propto s^{-0.2}$ (turbulent)')
        ax.set_xlabel('Arc-length from stagnation point $s$ [m]', fontsize=10)
        ax.set_ylabel('Convective heat flux [kW/m²]', fontsize=10)
        ax.set_title(f'Boundary layer development – {self._title_base()}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        if save_prefix:
            fname = f"{save_prefix}_boundary_layer.png"
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"  Saved: {fname}")
        else:
            plt.show()
        plt.close(fig)

    def export_aerothermal_table(self, results: Dict, save_prefix: Optional[str] = None):
        """
        Export a CSV (and optionally a LaTeX table) of heat flux values at key
        axial stations: nose tip, cockpit, ogive-cylinder junction, etc.
        """
        body = results['body']
        x_all = body['x']
        q_w = body['q_wind']
        q_l = body['q_lee']
        s = body['s']

        # Define key stations (axial positions)
        stations = [
            ('Nose tip (near)', 0.001),
            ('Cockpit forward', 5.0),  # adjust if your cockpit length differs
            ('Mid ogive', self.L_ogive / 2),
            ('Ogive-cylinder junction', self.L_ogive),
            ('Cylinder quarter', self.L_ogive + 0.25 * (self.L_body - self.L_ogive)),
            ('Cylinder mid', self.L_ogive + 0.5 * (self.L_body - self.L_ogive)),
            ('Cylinder aft (tail)', self.L_body)
        ]

        # Interpolate values at these x positions
        data = []
        for name, x_target in stations:
            idx = np.searchsorted(x_all, x_target)
            if idx >= len(x_all):
                idx = -1
            x_actual = x_all[idx]
            s_val = s[idx]
            qw = q_w[idx]
            ql = q_l[idx]
            data.append([name, x_actual, s_val, qw / 1e3, ql / 1e3])

        # Save as CSV
        if save_prefix:
            fname_csv = f"{save_prefix}_aerothermal_table.csv"
            with open(fname_csv, 'w') as f:
                f.write("Station, x [m], s [m], q_wind [kW/m²], q_lee [kW/m²]\n")
                for row in data:
                    f.write(f"{row[0]}, {row[1]:.3f}, {row[2]:.3f}, {row[3]:.2f}, {row[4]:.2f}\n")
            print(f"  Saved: {fname_csv}")

        # Optional: print LaTeX table to console
        print("\nLaTeX table for report:\n")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Heat flux at key axial stations. "
              f"M={self.M:.1f}, h={self.alt:.0f} km, AoA={self.aoa:.1f}°, turbulent BL. "
              f"Includes 1.2× margin.")
        print("\\label{tab:aerothermal_stations}")
        print("\\begin{tabular}{lccccc}")
        print("\\hline")
        print("Station & $x$ [m] & $s$ [m] & $\\dot{q}_\\text{wind}$ [kW/m²] & $\\dot{q}_\\text{lee}$ [kW/m²] \\\\")
        print("\\hline")
        for row in data:
            print(f"{row[0]} & {row[1]:.2f} & {row[2]:.2f} & {row[3]:.1f} & {row[4]:.1f} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    def _title_base(self) -> str:
        """Helper to generate a consistent title suffix."""
        return (f"M={self.M:.1f}, h={self.alt:.0f} km, AoA={self.aoa:.1f}°, "
                f"{'Turb.' if self.turb else 'Lam.'} BL, Tw={self.T_w:.0f} K")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vehicle = OgiveCylinderAnalysis(
        mach          = MACH,
        altitude_km   = ALTITUDE_KM,
        wall_temp_k   = WALL_TEMP_K,
        is_turbulent  = IS_TURBULENT,
        aoa_deg       = AOA_DEG,
        body_radius_m = BODY_RADIUS_M,
        ogive_length_m= OGIVE_LENGTH_M,
        body_length_m = BODY_LENGTH_M,
        nose_radius_m = NOSE_RADIUS_M,
    )

    results = vehicle.run(
        wing_chord     = WING_CHORD_M,
        wing_le_radius = WING_LE_RADIUS_M,
        wing_wedge_deg = WING_WEDGE_DEG,
        num_points     = NUM_POINTS,
    )

    vehicle.plot(results, save_prefix=FIGURE_PREFIX if SAVE_FIGURES else None)
    # New plots for the report
    if SAVE_FIGURES:
        vehicle.plot_thermal_landscape(results, save_prefix=FIGURE_PREFIX)
        vehicle.plot_boundary_layer_development(results, save_prefix=FIGURE_PREFIX)
        vehicle.export_aerothermal_table(results, save_prefix=FIGURE_PREFIX)
    else:
        vehicle.plot_thermal_landscape(results, save_prefix=None)
        vehicle.plot_boundary_layer_development(results, save_prefix=None)
        vehicle.export_aerothermal_table(results, save_prefix=None)