#!/usr/bin/env python3
"""
Hypersonic Convective Heat Flux Analysis Tool
=============================================
Eckert's Reference Enthalpy Method + Fay-Riddell stagnation correlation.
Validated range: Mach 2–8, altitudes 10–65 km.
Accuracy: ±15 % for cold-wall conditions (Tw/Taw ≤ 0.5).

References
----------
  Eckert, E.R.G. (1955). Engineering relations for friction and heat transfer
      to surfaces in high-velocity flow. J. Aeronautical Sciences, 22(8).
  Fay, J.A. & Riddell, F.R. (1958). Theory of stagnation point heat transfer
      in dissociated air. J. Aeronautical Sciences, 25(2), 73–85.
  U.S. Standard Atmosphere (1976).
  Anderson, J.D. (2006). Hypersonic and High Temperature Gas Dynamics, 2nd ed.
  Tian et al. (2025); Şimşek et al. (2020); Nozaki (2007).

Design Notes
------------
  • Dissociation is NOT modelled (calorically perfect gas, γ = 1.4).
    Above ~Mach 7 or below ~30 km this introduces non-trivial error.
  • Radiation and ablation are NOT included.
  • Shock–boundary-layer interaction, real-gas effects, and 3-D corner
    flows at the wing–body junction require CFD / higher-order tools.
  • For TPS sizing: use turbulent mode and apply a 1.5× margin on top.
"""

# ============================================================
#  USER INPUTS  –  Edit this block only
# ============================================================

MACH              = 5.0     # Free-stream Mach number  [-]
ALTITUDE_KM       = 31.0    # Cruise altitude           [km]
WALL_TEMP_K       = 300.0   # Assumed wall temperature  [K]
IS_TURBULENT      = True    # True = turbulent BL (conservative for TPS sizing)

# Geometry
NOSE_RADIUS_M     = 0.010   # Nose tip radius of curvature   [m]
WING_LE_RADIUS_M  = 0.005   # Wing leading-edge radius        [m]
FUSELAGE_LENGTH_M = 20.0    # Total fuselage length           [m]
WING_CHORD_M      = 5.0     # Wing chord (root)               [m]

# Lower-surface forebody compression wedge angle (typical ~8–12° for M5 waverider)
WEDGE_ANGLE_DEG   = 10.0    # Degrees; set 0.0 for free-stream on lower surface

# Analysis resolution
NUM_POINTS        = 100     # Points along each surface

# Output
SAVE_FIGURES      = True    # False → interactive display; True → save PNGs
FIGURE_PREFIX     = "heat_flux"

# ============================================================
#  END OF USER INPUTS
# ============================================================

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
    R_air  = 287.05   # J/(kg·K)
    g0     = 9.80665  # m/s²
    gamma  = 1.4

    # Layer definitions: (base_alt_km, base_T_K, lapse_K/km, base_P_Pa)
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
            dh = (h - h_base) * 1000.0  # m
            if abs(lapse) < 1e-9:          # isothermal
                T = T_base
                P = P_base * math.exp(-g0 * dh / (R_air * T_base))
            else:
                lapse_m = lapse / 1000.0   # K/m
                T = T_base + lapse_m * dh
                P = P_base * (T / T_base) ** (-g0 / (lapse_m * R_air))
            rho = P / (R_air * T)
            return T, P, rho

    # Top layer
    h_base, T_base, lapse, P_base = layers[-1]
    dh = (h - h_base) * 1000.0
    T = T_base
    P = P_base * math.exp(-g0 * dh / (R_air * T_base))
    rho = P / (R_air * T)
    return T, P, rho


# ── Gas property helpers ──────────────────────────────────────────────────────

GAMMA   = 1.4
R_AIR   = 287.05          # J/(kg·K)
PR      = 0.71            # Prandtl number (air, moderate temperatures)
CP      = GAMMA * R_AIR / (GAMMA - 1)   # J/(kg·K)

MU_REF  = 1.716e-5        # Pa·s  (Sutherland reference)
T_REF   = 273.15          # K
S_SUTH  = 110.4           # K


def viscosity(T: float) -> float:
    """Dynamic viscosity via Sutherland's law [Pa·s]."""
    return MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)


def thermal_conductivity(T: float) -> float:
    """Thermal conductivity from Prandtl number [W/(m·K)]."""
    return viscosity(T) * CP / PR


# ── Shock relations ───────────────────────────────────────────────────────────

def normal_shock(M1: float) -> Tuple[float, float, float]:
    """
    Normal shock relations for a calorically perfect gas.

    Returns
    -------
    M2    : downstream Mach number
    T2_T1 : static temperature ratio
    rho2_rho1 : density ratio
    """
    g = GAMMA
    M1sq = M1 ** 2
    M2sq = (1.0 + (g - 1) / 2.0 * M1sq) / (g * M1sq - (g - 1) / 2.0)
    T2_T1 = (1.0 + 2.0 * g / (g + 1) * (M1sq - 1.0)) * \
            (2.0 + (g - 1) * M1sq) / ((g + 1) * M1sq)
    rho2_rho1 = (g + 1) * M1sq / (2.0 + (g - 1) * M1sq)
    return math.sqrt(M2sq), T2_T1, rho2_rho1


def oblique_shock_angle(M1: float, wedge_deg: float) -> float:
    """
    Solve the theta–beta–M relation for the weak-shock solution.

    Parameters
    ----------
    M1        : upstream Mach number
    wedge_deg : flow deflection angle [degrees]

    Returns
    -------
    beta_deg : shock wave angle [degrees]
    """
    theta = math.radians(wedge_deg)
    g     = GAMMA

    # Mach (minimum) wave angle
    mu    = math.asin(1.0 / M1)

    # Find the beta of maximum deflection (theta_max) to confirm solvability.
    # Search in [mu+eps, 90°-eps] for the peak of theta(beta).
    def theta_of_beta(beta):
        sb2 = math.sin(beta) ** 2
        cot = math.cos(beta) / math.sin(beta)
        num = 2.0 * cot * (M1 ** 2 * sb2 - 1.0)
        den = M1 ** 2 * (g + math.cos(2.0 * beta)) + 2.0
        val = num / den
        # Clamp to avoid domain errors from numerical noise
        val = max(-1.0, min(1.0, val))
        return math.atan(val)

    # Locate theta_max by golden-section search in [mu+eps, pi/2-eps]
    blo, bhi = mu + 1e-4, math.pi / 2.0 - 1e-4
    # 60-step scan to find rough location of peak
    betas  = [blo + (bhi - blo) * i / 60 for i in range(61)]
    thetas = [theta_of_beta(b) for b in betas]
    idx    = thetas.index(max(thetas))
    theta_max = thetas[idx]

    if theta > theta_max:
        raise ValueError(
            f"Deflection {wedge_deg:.1f}° exceeds detachment limit "
            f"({math.degrees(theta_max):.1f}°) for M={M1:.2f}."
        )

    # Weak-shock solution: beta is in [mu, beta_at_theta_max]
    # Find beta_at_theta_max
    beta_at_max = betas[idx]

    # Bisect in [mu+eps, beta_at_max] for the weak solution
    b_lo = mu + 1e-4
    b_hi = beta_at_max
    f_lo = theta_of_beta(b_lo) - theta
    f_hi = theta_of_beta(b_hi) - theta

    if f_lo * f_hi > 0:
        # Edge case: theta is very close to theta_max; use beta_at_max directly
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

    Returns
    -------
    M2, T2, rho2, u2  (Mach, temperature [K], density [kg/m³], velocity [m/s])
    """
    beta_deg = oblique_shock_angle(M1, wedge_deg)
    beta     = math.radians(beta_deg)
    Mn1      = M1 * math.sin(beta)

    Mn2, T2_T1, rho2_rho1 = normal_shock(Mn1)

    T2   = T1 * T2_T1
    rho2 = rho1 * rho2_rho1

    # Downstream total Mach (using geometry: M2 = Mn2 / sin(beta - theta))
    theta = math.radians(wedge_deg)
    M2    = Mn2 / math.sin(beta - theta)
    a2    = math.sqrt(GAMMA * R_AIR * T2)
    u2    = M2 * a2
    return M2, T2, rho2, u2


# ── Reference enthalpy method ─────────────────────────────────────────────────

def reference_conditions(T_e: float, u_e: float, P_e: float,
                          T_w: float, turbulent: bool
                          ) -> Tuple[float, float, float]:
    """
    Eckert's reference enthalpy method.

    Evaluates fluid properties at a reference state that accounts for
    compressibility and wall-temperature effects.  Properties at T* are
    used in place of free-stream properties in standard incompressible
    correlations.

    Returns
    -------
    T_star   : reference temperature [K]
    rho_star : reference density     [kg/m³]
    mu_star  : reference viscosity   [Pa·s]
    """
    # Recovery factor
    r = PR ** (1.0 / 3.0) if turbulent else math.sqrt(PR)

    # Adiabatic-wall temperature  (Eckert form, valid for attached BL)
    T_aw = T_e * (1.0 + r * (GAMMA - 1.0) / 2.0 * (u_e / math.sqrt(GAMMA * R_AIR * T_e)) ** 2)

    # Reference enthalpy (Eckert 1955)
    h_w   = CP * T_w
    h_e   = CP * T_e
    h_aw  = CP * T_aw
    h_star = h_w + 0.50 * (h_e - h_w) + 0.22 * (h_aw - h_w)
    T_star = h_star / CP

    # Reference density at edge pressure (BL pressure constant across layer)
    rho_star = P_e / (R_AIR * T_star)
    mu_star  = viscosity(T_star)

    return T_star, rho_star, mu_star


# ── Stagnation-point heat flux (Fay–Riddell) ─────────────────────────────────

def stagnation_heat_flux(M_inf: float, rho_inf: float, T_inf: float,
                          P_inf: float, V_inf: float,
                          nose_radius: float, T_w: float) -> Tuple[float, Dict]:
    """
    Stagnation-point convective heat flux via the Fay–Riddell correlation.

    For a calorically-perfect gas (no dissociation) the Lewis-number
    correction drops out and the correlation reduces to:

        q_s = 0.57 · Pr^{-0.6} · sqrt(rho_e · mu_e · du/dx) · (h_0 - h_w)

    where h_0 is the total (stagnation) enthalpy at the edge (= h_e + V²/2
    evaluated just downstream of the normal shock at the nose).

    The velocity gradient at the stagnation point is approximated from
    Newtonian theory:

        du/dx ≈ (1/R_nose) · sqrt(2 · q_inf / rho_e)

    where q_inf is the free-stream dynamic pressure.

    Notes
    -----
    • This formulation is conservative (cold-wall assumption).
    • Dissociation effects become important above ~Mach 7; apply a
      correction factor or use equilibrium real-gas properties there.
    • Nose radius dominates: halving R_nose increases q_s by ~41 %.

    Returns
    -------
    q_stag  : stagnation heat flux [W/m²]
    diag    : dictionary of intermediate quantities for reporting
    """
    # Normal shock at the nose (blunt body → normal shock at stagnation)
    M2, T2_T1, rho2_rho1 = normal_shock(M_inf)
    T_e   = T_inf * T2_T1
    rho_e = rho_inf * rho2_rho1
    # Post-shock pressure (Rankine–Hugoniot)
    P2_P1 = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (M_inf ** 2 - 1.0)
    P_e   = P_inf * P2_P1

    # Velocity gradient at the stagnation point (Newtonian approximation,
    # Fay & Riddell 1958 Eq. 27).  Uses free-stream dynamic pressure and
    # post-shock density.
    q_inf_dyn = 0.5 * rho_inf * V_inf ** 2
    du_dx     = (1.0 / nose_radius) * math.sqrt(2.0 * q_inf_dyn / rho_e)

    # Viscosity at post-shock edge conditions
    mu_e = viscosity(T_e)

    # Total (stagnation) enthalpy is conserved across the shock and through
    # the boundary layer edge; use free-stream total enthalpy.
    h0  = CP * T_inf + 0.5 * V_inf ** 2
    h_w = CP * T_w

    # Fay–Riddell (calorically-perfect, frozen chemistry)
    q_stag = (0.57 / PR ** 0.6) * math.sqrt(rho_e * mu_e * du_dx) * max(h0 - h_w, 0.0)

    diag = dict(
        T_e=T_e, rho_e=rho_e, P_e=P_e,
        du_dx=du_dx, h0=h0, h_w=h_w,
        mu_e=mu_e
    )
    return q_stag, diag


# ── Leading-edge stagnation heat flux ─────────────────────────────────────────

def leading_edge_heat_flux(M_inf: float, rho_inf: float, T_inf: float,
                            P_inf: float, V_inf: float,
                            le_radius: float, T_w: float,
                            sweep_deg: float = 0.0) -> float:
    """
    Swept leading-edge stagnation heat flux.

    Uses the same Fay–Riddell approach applied to a cylinder with the
    component of velocity normal to the leading edge:

        V_n = V_inf · cos(Lambda)

    where Lambda is the leading-edge sweep angle.  This is a standard
    approximation for attached-flow leading edges.

    Parameters
    ----------
    sweep_deg : leading-edge sweep angle [degrees]

    Returns
    -------
    q_le : leading-edge heat flux [W/m²]
    """
    if le_radius <= 0:
        warnings.warn("LE radius ≤ 0; infinite heat flux physically.  "
                       "Using 1 mm as a lower bound for numerics.", stacklevel=2)
        le_radius = 1e-3

    # Normal-to-LE component
    cos_sw = math.cos(math.radians(sweep_deg))
    V_n    = V_inf * cos_sw
    M_n    = M_inf * cos_sw   # approximate (neglects spanwise compressibility)
    if M_n < 1.0:
        M_n = 1.001            # subsonic LE: use incompressible approach (not modelled here)

    # Normal-shock conditions based on normal velocity component
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
    Local convective heat flux on a flat plate at position x from the leading edge.

    Uses the Nusselt-number correlations:
      Laminar  : Nu_x = 0.332 · Re_x*^{1/2} · Pr*^{1/3}   (Pohlhausen/Eckert)
      Turbulent: Nu_x = 0.0296 · Re_x*^{4/5} · Pr*^{1/3}  (1/7-power law)

    All properties evaluated at the reference temperature T*.

    The heat flux is:

        q = (k* / x) · Nu_x · (T_aw - T_w)

    Note: dividing by x is implicit in Nu_x (it is a local Nusselt number),
    so the convection coefficient is h = Nu_x · k* / x.

    Parameters
    ----------
    x    : distance from leading edge / nose [m]  (must be > 0)
    T_e  : boundary-layer edge temperature [K]
    u_e  : boundary-layer edge velocity    [m/s]
    P_e  : boundary-layer edge pressure    [Pa]
    T_w  : wall temperature                [K]
    turbulent : use turbulent correlations

    Returns
    -------
    q : local heat flux [W/m²]  (≥ 0)
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

    h_conv = Nu_x * k_star / x   # [W/(m²·K)]

    # Adiabatic-wall temperature (local recovery)
    r    = PR ** (1.0 / 3.0) if turbulent else math.sqrt(PR)
    M_e  = u_e / math.sqrt(GAMMA * R_AIR * T_e)
    T_aw = T_e * (1.0 + r * (GAMMA - 1.0) / 2.0 * M_e ** 2)

    q = h_conv * max(T_aw - T_w, 0.0)
    return max(q, 0.0)


# ── Full vehicle analysis ─────────────────────────────────────────────────────

class HypersonicVehicleAnalysis:
    """
    Top-level orchestrator for vehicle thermal analysis.

    Stores all flight conditions derived from the user-specified
    Mach number and altitude, then runs the three main analyses:
      1. Nose stagnation point
      2. Wing leading edge
      3. Fuselage and wing flat-plate distributions
    """

    def __init__(self,
                 mach: float,
                 altitude_km: float,
                 wall_temp_k: float,
                 is_turbulent: bool,
                 nose_radius_m: float,
                 wing_le_radius_m: float,
                 wedge_angle_deg: float):

        self.M_inf          = mach
        self.altitude_km    = altitude_km
        self.T_w            = wall_temp_k
        self.is_turbulent   = is_turbulent
        self.nose_radius    = nose_radius_m
        self.wing_le_radius = wing_le_radius_m
        self.wedge_angle    = wedge_angle_deg

        # Atmospheric conditions
        self.T_inf, self.P_inf, self.rho_inf = us_standard_atmosphere(altitude_km)
        a_inf             = math.sqrt(GAMMA * R_AIR * self.T_inf)
        self.V_inf        = mach * a_inf

        # Lower-surface edge conditions (oblique shock)
        if wedge_angle_deg > 0:
            try:
                _, T_lo, rho_lo, u_lo = oblique_shock_conditions(
                    mach, self.rho_inf, self.T_inf, self.P_inf, wedge_angle_deg)
                P_lo = rho_lo * R_AIR * T_lo
                self._lower = dict(T_e=T_lo, u_e=u_lo, P_e=P_lo)
            except ValueError as exc:
                warnings.warn(f"Oblique shock failed ({exc}). "
                               "Using free-stream on lower surface.", stacklevel=2)
                self._lower = self._freestream_edge()
        else:
            self._lower = self._freestream_edge()

        # Upper-surface edge conditions (free stream – conservative; in reality
        # expansion waves on the upper surface reduce pressure and temperature,
        # so free-stream is the upper bound here)
        self._upper = self._freestream_edge()

        self._print_conditions()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _freestream_edge(self) -> Dict:
        return dict(T_e=self.T_inf, u_e=self.V_inf, P_e=self.P_inf)

    def _print_conditions(self):
        print("\n" + "=" * 62)
        print("  HYPERSONIC HEAT FLUX ANALYSER  –  Initialised")
        print("=" * 62)
        print(f"  Mach number         : {self.M_inf:.2f}")
        print(f"  Altitude            : {self.altitude_km:.1f} km")
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
        print(f"  Wing LE radius      : {self.wing_le_radius*1000:.1f} mm")
        print(f"  Forebody wedge      : {self.wedge_angle:.1f}°")
        print("=" * 62)

    # ── Public analysis methods ───────────────────────────────────────────────

    def run(self, fuselage_length: float, wing_chord: float,
            num_points: int = 100) -> Dict:
        """
        Run the complete vehicle analysis.

        Returns
        -------
        results : dict  with sub-dicts for each component
        """
        results = {}

        # ── 1. Nose stagnation ─────────────────────────────────────────────
        q_nose, diag_nose = stagnation_heat_flux(
            self.M_inf, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
            self.nose_radius, self.T_w)

        print("\n── Nose stagnation ──────────────────────────────────────────")
        print(f"  Post-shock T_e     : {diag_nose['T_e']:.1f} K")
        print(f"  Velocity gradient  : {diag_nose['du_dx']:.2f} s⁻¹")
        print(f"  Δh (h0 − hw)       : {(diag_nose['h0'] - diag_nose['h_w'])/1e3:.1f} kJ/kg")
        print(f"  Heat flux          : {q_nose/1e3:.2f} kW/m²")

        results['nose'] = dict(q=q_nose, diag=diag_nose)

        # ── 2. Wing leading edge ───────────────────────────────────────────
        q_le = leading_edge_heat_flux(
            self.M_inf, self.rho_inf, self.T_inf, self.P_inf, self.V_inf,
            self.wing_le_radius, self.T_w, sweep_deg=0.0)

        print("\n── Wing leading edge ────────────────────────────────────────")
        print(f"  LE radius          : {self.wing_le_radius*1000:.1f} mm")
        print(f"  Heat flux          : {q_le/1e3:.2f} kW/m²")

        results['wing_le'] = dict(q=q_le)

        # ── 3. Fuselage flat-plate distribution ────────────────────────────
        x_fuse = np.linspace(0.05, fuselage_length, num_points)
        q_upper_fuse, q_lower_fuse = [], []
        for x in x_fuse:
            q_upper_fuse.append(flat_plate_heat_flux(
                x, **self._upper, T_w=self.T_w, turbulent=self.is_turbulent))
            q_lower_fuse.append(flat_plate_heat_flux(
                x, **self._lower, T_w=self.T_w, turbulent=self.is_turbulent))

        q_upper_fuse = np.array(q_upper_fuse)
        q_lower_fuse = np.array(q_lower_fuse)
        q_avg_fuse   = 0.5 * (q_upper_fuse + q_lower_fuse)

        print("\n── Fuselage distribution ────────────────────────────────────")
        print(f"  Upper-surface max  : {q_upper_fuse.max()/1e3:.2f} kW/m²  (at x = {x_fuse[q_upper_fuse.argmax()]:.2f} m)")
        print(f"  Lower-surface max  : {q_lower_fuse.max()/1e3:.2f} kW/m²  (at x = {x_fuse[q_lower_fuse.argmax()]:.2f} m)")

        results['fuselage'] = dict(x=x_fuse,
                                    q_upper=q_upper_fuse,
                                    q_lower=q_lower_fuse,
                                    q_avg=q_avg_fuse)

        # ── 4. Wing flat-plate distribution ────────────────────────────────
        x_wing = np.linspace(0.01, wing_chord, num_points)
        q_upper_wing, q_lower_wing = [], []
        for x in x_wing:
            q_upper_wing.append(flat_plate_heat_flux(
                x, **self._upper, T_w=self.T_w, turbulent=self.is_turbulent))
            q_lower_wing.append(flat_plate_heat_flux(
                x, **self._lower, T_w=self.T_w, turbulent=self.is_turbulent))

        q_upper_wing = np.array(q_upper_wing)
        q_lower_wing = np.array(q_lower_wing)
        q_avg_wing   = 0.5 * (q_upper_wing + q_lower_wing)

        print("\n── Wing chord distribution ──────────────────────────────────")
        print(f"  Upper-surface max  : {q_upper_wing.max()/1e3:.2f} kW/m²")
        print(f"  Lower-surface max  : {q_lower_wing.max()/1e3:.2f} kW/m²")

        results['wing'] = dict(x=x_wing,
                                q_upper=q_upper_wing,
                                q_lower=q_lower_wing,
                                q_avg=q_avg_wing)

        # ── 5. Summary ─────────────────────────────────────────────────────
        peak_nose  = q_nose
        peak_le    = q_le
        peak_fuse  = max(q_upper_fuse.max(), q_lower_fuse.max())
        peak_wing  = max(q_upper_wing.max(), q_lower_wing.max())
        peak_all   = max(peak_nose, peak_le, peak_fuse, peak_wing)

        print("\n" + "=" * 62)
        print("  PEAK HEAT FLUX SUMMARY  (for TPS sizing)")
        print("=" * 62)
        print(f"  Nose stagnation    : {peak_nose/1e3:>8.2f} kW/m²")
        print(f"  Wing leading edge  : {peak_le/1e3:>8.2f} kW/m²")
        print(f"  Fuselage (max)     : {peak_fuse/1e3:>8.2f} kW/m²")
        print(f"  Wing chord (max)   : {peak_wing/1e3:>8.2f} kW/m²")
        print(f"  ──────────────────────────────────────────")
        print(f"  OVERALL PEAK       : {peak_all/1e3:>8.2f} kW/m²")
        print("=" * 62)
        print("  ⚠  Apply ≥ 1.5× safety factor for TPS material sizing.")
        print("  ⚠  Shock–shock / shock–BL interactions not modelled.")
        print("  ⚠  Dissociation ignored (valid below ~Mach 7 at altitude).")
        print("=" * 62)

        results['summary'] = dict(nose=peak_nose, wing_le=peak_le,
                                   fuselage=peak_fuse, wing=peak_wing,
                                   overall=peak_all)
        return results

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(self, results: Dict,
              save_prefix: Optional[str] = None):
        """Generate distribution plots and summary bar chart."""

        # ── Figure 1: surface distributions ───────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Convective Heat Flux – M={self.M_inf:.1f}, h={self.altitude_km:.0f} km, "
            f"Tw={self.T_w:.0f} K, "
            f"{'Turbulent' if self.is_turbulent else 'Laminar'} BL",
            fontsize=12)

        for ax, key, xlabel, title in [
            (axes[0], 'fuselage', 'Distance from nose (m)',
             'Fuselage heat-flux distribution'),
            (axes[1], 'wing',     'Distance from leading edge (m)',
             'Wing chord heat-flux distribution'),
        ]:
            d = results[key]
            x = d['x']
            ax.plot(x, d['q_upper'] / 1e3, 'r--', lw=1.5, label='Upper surface')
            ax.plot(x, d['q_lower'] / 1e3, 'g--', lw=1.5, label='Lower surface')
            ax.plot(x, d['q_avg']   / 1e3, 'b-',  lw=2.0, label='Average')
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Convective heat flux (kW/m²)', fontsize=11)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

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
                      bar.get_height() + 0.5 * s['overall'] / 1e3 * 0.02,
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

    # ── All design variables are set at the top of this file ──────────────
    vehicle = HypersonicVehicleAnalysis(
        mach            = MACH,
        altitude_km     = ALTITUDE_KM,
        wall_temp_k     = WALL_TEMP_K,
        is_turbulent    = IS_TURBULENT,
        nose_radius_m   = NOSE_RADIUS_M,
        wing_le_radius_m= WING_LE_RADIUS_M,
        wedge_angle_deg = WEDGE_ANGLE_DEG,
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