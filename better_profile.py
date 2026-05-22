"""
flight_profile_optimizer_multimaps.py

Multi-design thrust-map flight-profile optimizer.

What changed compared with the single-baseline-map version
----------------------------------------------------------
1) No single-map thrust linearization is used for final sizing.
   Instead, the code builds complete available-thrust maps for several
   turbojet design-thrust values and several ramjet design mass-flow values.

2) The optimizer constructs an optimal flight path for each design map pair:
      turbojet design thrust x ramjet design mdot
   It then selects the smallest feasible pair.

3) Optional interpolation between complete maps is supported.
   This is not the old simple linear scale from one baseline map. The engine
   model is evaluated at several actual design points first, then thrust is
   interpolated between those full maps.

4) The ramjet surrogate is removed by default. Exact ramjet calls are used.
   This is slower, but avoids contaminating the result with a surrogate shape.

Expected local files
--------------------
Put this script in the same folder as:
    ramjet_revised.py
    turbojet_pycycle_wrapper.py

Run:
    python flight_profile_optimizer_multimaps.py

Generated cache files
---------------------
    turbojet_multidesign_thrust_maps.npz
    ramjet_multidesign_thrust_maps.npz

The first run can be slow. Later runs load the cached map files unless
FORCE_REBUILD_MAPS = True.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import io
import traceback
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.optimize import differential_evolution, minimize


# =============================================================================
# 0. USER SETTINGS
# =============================================================================

RAMJET_FILE = "ramjet_revised.py"
TURBOJET_FILE = "turbojet_pycycle_wrapper.py"

TURBO_MAP_FILE = "turbojet_multidesign_thrust_maps.npz"
RAMJET_MAP_FILE = "ramjet_multidesign_thrust_maps.npz"

FORCE_REBUILD_MAPS = False

# NEXT-STEP RUN NOTE:
# This version uses 5 turbojet design cases x 5 ramjet design cases x
# 3 transition Mach values = 75 optimized design/transition cases.
# Ramjet optimization maps still use the calibrated surrogate to avoid full-map
# exact ramjet stalling; exact ramjet checks are disabled by default to keep runtime short.
# After the maps are successfully built once, set FORCE_REBUILD_TURBO_MAP and
# FORCE_REBUILD_RAMJET_MAP to False to load cached maps on later runs.

# Rebuild controls per engine map. Useful when only one map grid changed.
FORCE_REBUILD_TURBO_MAP = True
FORCE_REBUILD_RAMJET_MAP = True

# Optional map-building wall-clock limit. Set to None to disable.
MAX_MAP_BUILD_SECONDS = None
ALLOW_PARTIAL_MAPS_WHEN_TIMEOUT = True

# -------------------------------------------------------------------------
# TURBOJET SURROGATE SWITCH
# -------------------------------------------------------------------------
# Set True to build turbojet thrust maps with a fast analytical estimate instead
# of calling the full PyCycle wrapper at every Mach-altitude-design point.
# Use this for repeated S_PLAN iterations. Set False for final PyCycle maps.
USE_FAST_TURBOJET_SURROGATE = True

# Soft surrogate controls. The original conservative setting used
# rho^0.70 * (1 - 0.08*M^2), which can severely underpredict thrust
# during the high-altitude, high-Mach turbojet acceleration segment.
# These values are intended for sizing-loop exploration. For final validation,
# set USE_FAST_TURBOJET_SURROGATE = False and rebuild the turbojet maps.
TURBOJET_SURROGATE_ALT_EXPONENT = 0.35
TURBOJET_SURROGATE_MACH_LOSS_COEFF = 0.015
TURBOJET_SURROGATE_MIN_FACTOR = 0.30

# -------------------------------------------------------------------------
# RAMJET SURROGATE SWITCH
# -------------------------------------------------------------------------
# Set True to build the ramjet thrust maps with the fast surrogate instead of
# running the full ramjet model at every Mach-altitude-design point.
USE_FAST_RAMJET_SURROGATE = True

# Surrogate calibration point. These are the same style of settings as the
# earlier fast ramjet-map version.
RAMJET_SURROGATE_DESIGN_MACH = 4.0
RAMJET_SURROGATE_DESIGN_ALT_M = 25_000.0
RAMJET_SURROGATE_DESIGN_THRUST_KN_PER_ENGINE = 268.478

# Shape controls for the fast ramjet estimate.
RAMJET_SURROGATE_MACH_SIGMA = 1.10
RAMJET_SURROGATE_ALT_SIGMA_M = 13_000.0
RAMJET_SURROGATE_MIN_FACTOR = 0.15

# -------------------------------------------------------------------------
# CALIBRATED SURROGATE SETTINGS
# -------------------------------------------------------------------------
# First build sparse exact engine samples, then use exact/raw correction
# factors to generate cheap calibrated surrogate maps in later iterations.
USE_CALIBRATED_TURBOJET_SURROGATE = True
USE_CALIBRATED_RAMJET_SURROGATE = False  # tuned raw surrogate avoids sparse calibration underestimating Mach 5

TURBO_CALIBRATION_FILE = "turbojet_surrogate_calibration.npz"
RAMJET_CALIBRATION_FILE = "ramjet_surrogate_calibration_limited_mach_alt.npz"
FORCE_REBUILD_SURROGATE_CALIBRATION = False
FORCE_REBUILD_TURBO_CALIBRATION = False
FORCE_REBUILD_RAMJET_CALIBRATION = False

# Sparse exact samples. Keep these much smaller than the full map.
# The grids emphasize the transition region, because that is where feasibility
# is most sensitive.
TURBO_CAL_DESIGN_THRUST_GRID_LBF = np.array([150_000.0, 250_000.0])
TURBO_CAL_MACH_GRID = np.array([0.30, 0.80, 1.20, 1.60, 2.00, 2.30, 2.60, 2.80, 3.00])
TURBO_CAL_ALT_GRID_M = np.array([0.0, 5_000.0, 10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0])

# Ramjet exact samples for calibration. Keep this intentionally tiny because
# the exact ramjet model is the slow part. The full ramjet maps are still built
# on the normal dense grid afterwards, but using calibrated surrogate values.
#
# This grid uses only 2 x 3 x 4 = 24 exact ramjet calls instead of 140.
# Do not run exact ramjet calibration at 30 km: the ramjet solver can stall
# there. The 30 km map corrections are extrapolated from 20/25/27.5 km.
# The selected Mach points avoid the half-step values, which are not necessary
# for a first correction surface and made calibration much slower.
RAMJET_CAL_DESIGN_MDOT_GRID_KG_S = np.array([120.0, 300.0])
RAMJET_CAL_MACH_GRID = np.array([2.00, 3.00, 4.00, 5.00])
RAMJET_CAL_ALT_GRID_M = np.array([20_000.0, 25_000.0, 27_500.0])

# Correction-factor sanity limits. These prevent one failed/noisy exact point
# from destroying the full surrogate map.
SURROGATE_CORRECTION_MIN = 0.20
SURROGATE_CORRECTION_MAX = 5.00

# -------------------------------------------------------------------------
# RAMJET HIGH-MACH / HIGH-ALTITUDE FAST EXACT MODE
# -------------------------------------------------------------------------
# Do NOT skip high-Mach/high-altitude ramjet points. Instead, still run the
# ramjet model, but make the expensive ODE parts cheaper in that region by
# temporarily loosening solve_ivp tolerances inside the ramjet module.
#
# This keeps the point physically connected to the ramjet model instead of
# replacing it with the old surrogate or skipping it completely.
USE_FAST_RAMJET_SOLVER_IN_TROUBLE_REGION = True

# Altitude-dependent Mach threshold for fast ramjet computation.
# Meaning:
#   at 22.5 km -> fast from Mach 4.25
#   at 25.0 km -> fast from Mach 4.00
#   at 27.5 km -> fast from Mach 3.80
#   at 30.0 km -> fast from Mach 3.50
# Between these altitudes, the threshold is linearly interpolated.
# Below 22.5 km, fast mode is disabled unless USE_FAST_RAMJET_SOLVER_EVERYWHERE=True.
RAMJET_FAST_SOLVER_ALT_BREAKPOINTS_M = np.array([22_500.0, 25_000.0, 27_500.0, 30_000.0])
RAMJET_FAST_SOLVER_MACH_THRESHOLDS = np.array([4.25, 4.00, 3.80, 3.50])

# These tolerances are deliberately looser than typical high-accuracy ODE
# settings. If the results become too noisy, tighten to 1e-4 / 1e-6. If it is
# still too slow, loosen to 1e-2 / 1e-4.
RAMJET_FAST_SOLVER_RTOL = 1e-3
RAMJET_FAST_SOLVER_ATOL = 1e-5
RAMJET_FAST_SOLVER_METHOD = "RK23"  # usually cheaper than high-order stiff solvers for rough map points

# If True, fast solver settings are applied to ALL ramjet map points, not only
# the high-Mach/high-altitude region. Useful for quick exploratory runs.
USE_FAST_RAMJET_SOLVER_EVERYWHERE = False

# Basic numerical sanity filter.
MIN_VALID_THRUST_KN = 1e-3
MAX_VALID_THRUST_KN = 10_000.0

# Failure limits for map generation. These must exist before map building starts.
MAX_FAILED_RAMJET_POINTS = 50
MAX_FAILED_TURBOJET_POINTS = 30

# Keep these grids intentionally modest. They are complete engine-map design
# points, so increasing them multiplies run time.
#
# Suggested workflow:
#   1) Start with 3 x 3 design points.
#   2) Inspect the optimum.
#   3) Add local points around the optimum if needed.
TURBO_DESIGN_THRUST_GRID_LBF = np.array([
    200_000.0,
])

RAMJET_DESIGN_MDOT_GRID_KG_S = np.array([
    200.0,
])

# Optional final refinement. This optimizes continuous design variables between
# the complete maps using interpolation in the design dimension. Keep this True
# if you want a less blocky final answer without building a huge design grid.
ENABLE_INTERPOLATED_DESIGN_REFINEMENT = False

# Optional: optimize flight paths for extra interpolated design pairs, not only
# the original complete-map grid. These are NOT new engine-model calls; they use
# the 3D interpolators between the complete maps already built above.
ENABLE_INTERPOLATED_DESIGN_PAIR_SEARCH = False

# Interpolate only in the most promising region found by the coarse discrete search.
# The code first runs all complete-map design pairs, then builds a local dense
# interpolated grid around the best discrete result instead of sweeping the
# entire design space.
LOCAL_INTERP_AROUND_BEST_ONLY = True
LOCAL_INTERP_NEIGHBOR_STEPS = 1
INTERP_TURBO_DESIGN_POINTS = 7
INTERP_RAMJET_DESIGN_POINTS = 7

# Also refine the transition Mach only near the best coarse transition Mach.
# This does not build new engine maps; it only tests additional switch points
# inside the existing ramjet/turbojet map coverage.
LOCAL_INTERP_TRANSITION_POINTS = 5

# Optional: optimize the turbojet-to-ramjet transition Mach instead of fixing it.
# These values are checked as an outer discrete loop around each engine design pair.
# The ramjet Mach map must cover the lowest value here.
ENABLE_TRANSITION_MACH_SEARCH = True
TRANSITION_MACH_GRID = np.array([
    2.50,
    2.75,
    3.00,
])
DEFAULT_TRANSITION_MACH = 3.00


# If the best discrete result is on the top edge and still infeasible,
# interpolation cannot create a larger engine. Increase the actual complete-map
# design grid above, for example to 300_000 or 350_000 lbf, then rebuild maps.

# If True, only refine in a small box around the best discrete design.
LOCAL_REFINEMENT_AROUND_BEST_ONLY = False
LOCAL_REFINEMENT_NEIGHBOR_STEPS = 1

# After the discrete map-pair result, actively shrink the design variables using
# the interpolated thrust maps. This is what makes the final answer move away
# from exact grid values such as 150000 lbf or 150 kg/s.
ENABLE_FIXED_PROFILE_DESIGN_SHRINK = True
DESIGN_SHRINK_BISECTION_ITERS = 35
THRUST_FEASIBILITY_TOL_KN = 1e-3

# Exact ramjet map calls can be noisy/fragile. Keep debug on until the map is
# stable, then switch off if the terminal output becomes annoying.
RAMJET_DEBUG_ERRORS = True
MAX_RAMJET_ERROR_PRINTS = 8

# Map grids.
# Larger turbojet Mach grid for smoother interpolation and better path optimization.
# Keep the upper bound high enough to cover all candidate transition Mach numbers.
MACH_GRID_TURBO = np.array([
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.30,
    1.40,
    1.50,
    1.60,
    1.70,
    1.80,
    1.90,
    2.00,
    2.10,
    2.20,
    2.30,
    2.40,
    2.50,
    2.60,
    2.70,
    2.80,
    2.90,
    3.00,
])
MACH_GRID_RAMJET = np.array([
    2.50,
    2.75,
    3.00,
    3.25,
    3.50,
    3.75,
    4.00,
    4.25,
    4.50,
    4.75,
    5.00,
])
ALT_GRID_TURBO_M = np.linspace(0.0, 30_000.0, 10)
ALT_GRID_RAMJET_M = np.array([
    18_000.0,
    20_000.0,
    22_500.0,
    25_000.0,
    27_500.0,
    30_000.0,
])

# Mission.
M_TAKEOFF = 0.30
M_SWITCH = DEFAULT_TRANSITION_MACH  # kept for backward compatibility/reporting defaults
M_CRUISE = 5.00
H_CRUISE_M = 30_000.0
ACCEL_G_TARGET = 0.15

# Engine counts.
N_TURBOJETS = 2
N_RAMJETS = 2
RAMJET_PHI = 0.70

# Baseline values used only for reporting/normalizing the objective.
BASE_TURBO_DESIGN_THRUST_LBF = 150_000.0
BASE_RAMJET_DESIGN_MDOT = 200.0

# Flight-profile control Machs. Optimizer changes the free altitude knots.
MACH_KNOTS = np.array([0.30, 0.80, 1.20, 2.00, 3.00, 4.00, 5.00])
FIXED_ALT_KNOTS = {0: 0.0, len(MACH_KNOTS) - 1: H_CRUISE_M}
N_EVAL = 34

# Envelope constraints.
CL_MAX = 1.50
Q_MAX_PA = 120_000.0
THRUST_MARGIN = 0.00
BIG_PENALTY = 1e8

# Objective weights for choosing between feasible design map pairs.
W_TURBO_DESIGN = 1.0
W_RAMJET_DESIGN = 1.0

# Global optimizer settings. These are deliberately not huge, because the outer
# loop evaluates many design map pairs.
DE_MAXITER_PER_MAP = 70
DE_POPSIZE_PER_MAP = 8
LOCAL_MAXITER_PER_MAP = 250
RANDOM_SEED = 7

ALLOW_EXTRAPOLATION = False


# -------------------------------------------------------------------------
# FULL MISSION RANGE / DESCENT PLOT SETTINGS
# -------------------------------------------------------------------------
# Used only for plotting the complete takeoff-to-landing profile after the
# optimized climb/acceleration profile has been found.
TOTAL_MISSION_RANGE_M = 9_500_000.0

# The descent is unpowered and is sized to minimize descent range while keeping
# the combined horizontal and vertical acceleration at or below 0.15 g.
DESCENT_ACCEL_LIMIT_G = 0.15

# Plot/data resolution for the appended cruise and descent sections.
N_CRUISE_POINTS = 80
N_DESCENT_POINTS = 120

# Acceleration used to convert the optimized Mach-altitude climb into an
# approximate horizontal range. Keep this consistent with ACCEL_G_TARGET.
ACCEL_G_FOR_ASCENT_RANGE = ACCEL_G_TARGET

# Save the additional mission plots and CSV tables.
SAVE_FULL_MISSION_PLOTS = True
SAVE_FULL_MISSION_CSV = True

# -------------------------------------------------------------------------
# DIAGNOSTIC EXACT RAMJET CHECKS
# -------------------------------------------------------------------------
# Keep the optimization map on the fast/calibrated surrogate so the script does
# not stall while building hundreds of exact ramjet points. These diagnostic
# calls run only a few exact ramjet points after the optimization, so you can
# see whether the suspected weak regions are physical or map/surrogate artifacts.
RUN_EXACT_RAMJET_DIAGNOSTIC_CHECK = False
EXACT_RAMJET_DIAGNOSTIC_POINTS = [
    (2.50, 20_000.0),
    (2.75, 20_000.0),
    (3.00, 20_000.0),
    (3.00, 25_000.0),
    (4.00, 25_000.0),
    (5.00, 25_000.0),
    (5.00, 30_000.0),
]


# =============================================================================
# 1. DRAG MODEL
# =============================================================================

G = 9.81
R_GAS = 287.05
GAMMA = 1.4

# These still need to come from your outer aircraft sizing loop eventually.
W_TOG = 90_000
S_PLAN = 350
S_WET = 1050

MAC = 21.0
L_REF = 35.0
IF = 1.05
t_over_c = 0.05
sweep_deg = 35.0
sweep_rad = np.radians(sweep_deg)
AR = 7.0


def atmosphere_drag(alt_m: float) -> tuple[float, float]:
    g0 = 9.80665
    P0 = 101325.0
    T0 = 288.15
    L1 = -0.0065
    h11 = 11000.0
    T11 = T0 + L1 * h11
    P11 = P0 * (T11 / T0) ** (-g0 / (L1 * R_GAS))
    h20 = 20000.0
    T20 = T11
    P20 = P11 * np.exp(-g0 * (h20 - h11) / (R_GAS * T11))
    L3 = 0.0010

    if alt_m <= 11000.0:
        T = T0 + L1 * alt_m
        P = P0 * (T / T0) ** (-g0 / (L1 * R_GAS))
    elif alt_m <= 25000.0:
        T = T11
        P = P11 * np.exp(-g0 * (alt_m - h11) / (R_GAS * T))
    elif alt_m <= 40000.0:
        T = T20 + L3 * (alt_m - h20)
        P = P20 * (T / T20) ** (-g0 / (max(1e-4, L3) * R_GAS))
    else:
        T = 216.65
        P = 1000.0

    rho = P / (R_GAS * T)
    return rho, T


def reynolds_number(rho: float, v: float, temp: float, chord: float) -> float:
    mu_0 = 1.7894e-5
    T_0 = 273.15
    S_suth = 110.4
    mu = mu_0 * (temp / T_0) ** 1.5 * (T_0 + S_suth) / (temp + S_suth)
    return (rho * v * chord) / mu


def drag_and_required_thrust_kN(mach: float, altitude_m: float, accel_g: float = ACCEL_G_TARGET) -> dict:
    rho, T = atmosphere_drag(float(altitude_m))
    a = np.sqrt(GAMMA * R_GAS * T)
    M = float(mach)
    V = M * a
    q = 0.5 * rho * V**2
    q_safe = max(q, 1e-9)
    cl_needed = (W_TOG * G) / (q_safe * S_PLAN)

    if M < 1.2:
        regime = "transonic"
        Re_dyn = max(reynolds_number(rho, V, T, MAC), 10.0)
        cf_inc = 0.455 / (np.log10(Re_dyn) ** 2.58)
        cf_comp = cf_inc / (1 + 0.12 * M**2) ** 0.5
        cd_f = cf_comp * IF * (S_WET / S_PLAN)
        M_crit = 0.9 - 1.2 * t_over_c - 0.1 * (1 - np.cos(sweep_rad))
        M_peak = 1.05
        if M < M_crit:
            cd_wave = 0.0
        elif M <= M_peak:
            amplitude = 20 * (t_over_c**2.5) * np.cos(sweep_rad) ** 2
            cd_wave = amplitude * np.sin((M - M_crit) / (M_peak - M_crit) * (np.pi / 2)) ** 2
        else:
            amplitude_peak = 20 * (t_over_c**2.5) * np.cos(sweep_rad) ** 2
            cd_wave = amplitude_peak / np.sqrt(max(0.1, M**2 - 1.0))
        e_oswald = 0.85 - 0.02 * M
        cd_induced = (cl_needed**2) / (np.pi * AR * e_oswald)
        alpha_deg = np.degrees(cl_needed / (2 * np.pi * AR / (AR + 2)))
    else:
        regime = "supersonic" if M < 3.0 else "hypersonic"
        Re_dyn_super = max(reynolds_number(rho, V, T, MAC), 10.0)
        cf_super = 0.455 / (np.log10(Re_dyn_super) ** 2.58)
        cd_f_super = cf_super * IF * (S_WET / S_PLAN)
        alpha_rad_super = np.sqrt(np.abs(cl_needed**0.75) / 2)
        cd_wave_super = 2 * np.sin(alpha_rad_super) ** 3

        Re_dyn_hyper = max(reynolds_number(rho, V, T, L_REF), 10.0)
        cf_hyper = (0.074 / (Re_dyn_hyper ** 0.2)) * ((1 / (1 + 0.15 * M**2)) ** 0.58)
        cd_f_hyper = cf_hyper * 2.0
        cl_alpha_hyper = 4.0 / np.sqrt(max(0.01, M**2 - 1.0))
        alpha_rad_hyper = cl_needed / cl_alpha_hyper
        cd_wave_hyper = cl_needed * alpha_rad_hyper

        k = 7.0
        weight_hyper = 1.0 / (1.0 + np.exp(-k * (M - 3.0)))
        weight_super = 1.0 - weight_hyper
        cd_f = (cd_f_super * weight_super) + (cd_f_hyper * weight_hyper)
        cd_wave = (cd_wave_super * weight_super) + (cd_wave_hyper * weight_hyper)
        alpha_rad = (alpha_rad_super * weight_super) + (alpha_rad_hyper * weight_hyper)
        cd_induced = 0.0
        alpha_deg = np.degrees(alpha_rad)

    cd_total = cd_f + cd_wave + cd_induced
    drag_N = q * S_PLAN * cd_total
    accel_N = W_TOG * accel_g * G
    thrust_req_N = drag_N + accel_N

    return {
        "mach": M,
        "altitude_m": float(altitude_m),
        "rho": rho,
        "T_K": T,
        "V_m_s": V,
        "q_Pa": q,
        "CL": cl_needed,
        "CD": cd_total,
        "CD_f": cd_f,
        "CD_wave": cd_wave,
        "CD_induced": cd_induced,
        "alpha_deg": alpha_deg,
        "drag_kN": drag_N / 1000.0,
        "accel_kN": accel_N / 1000.0,
        "thrust_req_kN": thrust_req_N / 1000.0,
        "L_over_D": cl_needed / cd_total if cd_total > 0 else np.nan,
        "aero_regime": regime,
    }


# =============================================================================
# 2. ENGINE IMPORTS AND EXACT POINT EVALUATORS
# =============================================================================


def import_from_path(module_name: str, path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path!s}. Edit RAMJET_FILE / TURBOJET_FILE or move files.")
    loader = importlib.machinery.SourceFileLoader(module_name, str(path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def get_ramjet_module():
    return import_from_path("ramjet_model_dynamic", RAMJET_FILE)


@lru_cache(maxsize=1)
def get_turbojet_module():
    return import_from_path("turbojet_model_dynamic", TURBOJET_FILE)


def configure_turbojet_design(design_thrust_lbf_per_engine: float) -> None:
    tj = get_turbojet_module()
    if hasattr(tj, "DESIGN_THRUST_LBF"):
        tj.DESIGN_THRUST_LBF = float(design_thrust_lbf_per_engine)
    if hasattr(tj, "BASE_DESIGN_THRUST_LBF") and hasattr(tj, "DESIGN_THRUST_LBF"):
        tj.SCALE = tj.DESIGN_THRUST_LBF / tj.BASE_DESIGN_THRUST_LBF
    # Force model reset after changing design point, if wrapper supports it.
    if hasattr(tj, "reset_turbo_model"):
        tj.reset_turbo_model()


def turbojet_thrust_point_kN(mach: float, altitude_m: float, design_thrust_lbf_per_engine: float) -> float:
    tj = get_turbojet_module()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            T = tj.turbojet_thrust_kN(
                mach=float(mach),
                altitude_m=float(altitude_m),
                n_engines=N_TURBOJETS,
                clamp_negative=True,
            )
        T = float(T)
        if not np.isfinite(T):
            return np.nan
        return max(T, 0.0)
    except Exception:
        return np.nan


def turbojet_fast_surrogate_thrust_kN(
    mach: float,
    altitude_m: float,
    design_thrust_lbf_per_engine: float,
) -> float:
    """
    Fast turbojet thrust estimate for map generation.

    This is meant for rapid repeated S_PLAN / sizing-loop iterations, not for
    final engine validation. It avoids PyCycle calls while preserving the main
    trends: larger design thrust gives more thrust, high altitude reduces
    thrust, and high Mach reduces net thrust.

    Returns total thrust for all turbojet engines [kN].
    """
    M = float(mach)
    h = float(altitude_m)

    rho, _ = atmosphere_drag(h)
    rho0, _ = atmosphere_drag(0.0)

    altitude_factor = max((rho / max(rho0, 1e-12)) ** TURBOJET_SURROGATE_ALT_EXPONENT, 0.0)
    # Softer Mach lapse than the original M^2 law. The old law made
    # turbojet thrust collapse near the transition region and produced
    # artificial infeasibility compared with the exact PyCycle wrapper.
    mach_factor = max(
        TURBOJET_SURROGATE_MIN_FACTOR,
        1.0 - TURBOJET_SURROGATE_MACH_LOSS_COEFF * M**1.5,
    )

    static_total_kN = float(design_thrust_lbf_per_engine) * 0.0044482216152605 * N_TURBOJETS
    thrust_kN = static_total_kN * altitude_factor * mach_factor

    if not np.isfinite(thrust_kN):
        return np.nan
    return max(float(thrust_kN), 0.0)


_RAMJET_ERROR_COUNT = 0


def ramjet_thrust_point_kN(mach: float, altitude_m: float, design_mdot_kg_s: float, *, suppress_output: bool = True) -> float:
    global _RAMJET_ERROR_COUNT
    rj = get_ramjet_module()
    eng = rj.Ramjet()

    def _run_case():
        inp = eng.inlet_properties(h=float(altitude_m), Ma=float(mach), m_air=float(design_mdot_kg_s))
        iso = eng.isolator_properties(inp)
        sec2 = eng.combustor_properties2(iso)
        sec3 = eng.combustor_properties3(sec2, phi=RAMJET_PHI)
        sec4 = eng.combustor_properties4(sec3)
        sec5 = eng.nozzle_properties(sec4, inp)
        perf = eng.performance(inp, sec5, sec3)
        if "Fin" not in perf:
            raise KeyError(f"Ramjet performance output has no 'Fin'. Keys: {list(perf.keys())}")
        thrust_kN = float(perf["Fin"]) / 1000.0
        if not np.isfinite(thrust_kN):
            raise FloatingPointError(f"Ramjet Fin is not finite: {perf['Fin']}")
        return thrust_kN

    try:
        fast_solver = use_fast_ramjet_solver_settings(mach, altitude_m)
        with temporary_fast_ramjet_solve_ivp(rj, enabled=fast_solver):
            if suppress_output:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    thrust_kN = _run_case()
            else:
                thrust_kN = _run_case()
        return float(thrust_kN)
    except Exception as exc:
        _RAMJET_ERROR_COUNT += 1
        if RAMJET_DEBUG_ERRORS and _RAMJET_ERROR_COUNT <= MAX_RAMJET_ERROR_PRINTS:
            print("\nRAMJET EVALUATION FAILED")
            print("------------------------")
            print(f"Design mdot = {float(design_mdot_kg_s):.2f} kg/s")
            print(f"Mach        = {float(mach):.3f}")
            print(f"Altitude    = {float(altitude_m):.1f} m")
            print(f"Error       = {type(exc).__name__}: {exc}")
            traceback.print_exc()
            print()
        return np.nan


# =============================================================================
# 3. COMPLETE MULTI-DESIGN MAP BUILDING
# =============================================================================


def sanitize_thrust_value_kN(value: float, *, label: str, design_value: float, mach: float, altitude_m: float) -> float:
    """
    Reject non-physical/nonsensical thrust values before they enter the map.

    Returning NaN is intentional: NaNs are later excluded from interpolation
    instead of being smeared into nearby design points.
    """
    try:
        T = float(value)
    except Exception:
        return np.nan

    if not np.isfinite(T):
        return np.nan
    if T < MIN_VALID_THRUST_KN:
        return np.nan
    if T > MAX_VALID_THRUST_KN:
        print(
            f"WARNING: rejected huge {label} thrust value: "
            f"design={design_value}, M={mach:.3f}, h={altitude_m:.1f} m, T={T:.2f} kN"
        )
        return np.nan
    return T


def ramjet_fast_solver_mach_threshold(altitude_m: float) -> float:
    """
    Mach threshold for fast ramjet computation as a function of altitude.

    Returns np.inf below the first breakpoint, meaning fast mode is disabled
    below 22.5 km unless USE_FAST_RAMJET_SOLVER_EVERYWHERE=True.
    """
    h = float(altitude_m)
    alt_bp = RAMJET_FAST_SOLVER_ALT_BREAKPOINTS_M
    mach_bp = RAMJET_FAST_SOLVER_MACH_THRESHOLDS

    if h < alt_bp[0]:
        return np.inf
    if h >= alt_bp[-1]:
        return float(mach_bp[-1])
    return float(np.interp(h, alt_bp, mach_bp))


def use_fast_ramjet_solver_settings(mach: float, altitude_m: float) -> bool:
    """Return True when the ramjet point should use cheaper ODE solver settings."""
    if USE_FAST_RAMJET_SOLVER_EVERYWHERE:
        return True
    if not USE_FAST_RAMJET_SOLVER_IN_TROUBLE_REGION:
        return False

    threshold = ramjet_fast_solver_mach_threshold(altitude_m)
    return float(mach) >= threshold


@contextlib.contextmanager
def temporary_fast_ramjet_solve_ivp(ramjet_module, enabled: bool):
    """
    Temporarily patch the ramjet module's solve_ivp function, if it has one.

    Many custom ramjet scripts use:
        from scipy.integrate import solve_ivp
    In that case, patching ramjet_module.solve_ivp is enough to make the
    combustor/nozzle ODE calls cheaper without changing the ramjet equations.
    """
    if not enabled or not hasattr(ramjet_module, "solve_ivp"):
        yield
        return

    original_solve_ivp = ramjet_module.solve_ivp

    def faster_solve_ivp(fun, t_span, y0, *args, **kwargs):
        kwargs["rtol"] = min(float(kwargs.get("rtol", RAMJET_FAST_SOLVER_RTOL)), RAMJET_FAST_SOLVER_RTOL)
        kwargs["atol"] = min(float(kwargs.get("atol", RAMJET_FAST_SOLVER_ATOL)), RAMJET_FAST_SOLVER_ATOL)
        kwargs["method"] = kwargs.get("method", RAMJET_FAST_SOLVER_METHOD)
        return original_solve_ivp(fun, t_span, y0, *args, **kwargs)

    ramjet_module.solve_ivp = faster_solve_ivp
    try:
        yield
    finally:
        ramjet_module.solve_ivp = original_solve_ivp



def filter_nonsensical_design_slices(table: np.ndarray, design_grid: np.ndarray, label: str) -> np.ndarray:
    """
    Compatibility helper.

    This version does not apply an extra design-monotonicity filter. It simply
    returns the table unchanged, so no additional behavior is introduced.
    """
    return np.array(table, dtype=float, copy=True)


def ramjet_fast_surrogate_thrust_kN(mach: float, altitude_m: float, design_mdot_kg_s: float) -> float:
    """
    Fast ramjet thrust estimate used only when USE_FAST_RAMJET_SURROGATE=True.

    It preserves the multi-design-map structure by evaluating the surrogate at
    each ramjet design mdot. It avoids the expensive CEA/ODE ramjet calls.
    """
    M = float(mach)
    h = float(altitude_m)
    mdot_design = float(design_mdot_kg_s)

    rho, T = atmosphere_drag(h)
    a = np.sqrt(GAMMA * R_GAS * T)
    V = M * a

    rho_d, T_d = atmosphere_drag(RAMJET_SURROGATE_DESIGN_ALT_M)
    a_d = np.sqrt(GAMMA * R_GAS * T_d)
    V_d = RAMJET_SURROGATE_DESIGN_MACH * a_d

    flight_mdot_ratio = (rho * V) / max(rho_d * V_d, 1e-12)
    design_mdot_ratio = mdot_design / max(BASE_RAMJET_DESIGN_MDOT, 1e-12)

    # Tuned exploratory ramjet shape.
    # The old Gaussian bell was centered at Mach 4 and therefore reduced thrust
    # again at Mach 5, which made the cruise point look artificially weak.
    # For preliminary sizing, keep ramjet performance from collapsing after Mach 4.
    mach_factor = np.interp(
        M,
        np.array([2.50, 3.00, 3.50, 4.00, 4.50, 5.00]),
        np.array([0.45, 0.70, 0.90, 1.00, 1.12, 1.25]),
        left=0.35,
        right=1.25,
    )

    # Do not heavily penalize 30 km. The density term already reduces captured
    # mass flow at altitude, so a second strong altitude bell double-counts the loss.
    alt_factor = np.interp(
        h,
        np.array([18_000.0, 20_000.0, 25_000.0, 30_000.0]),
        np.array([0.90, 0.98, 1.00, 0.95]),
        left=0.85,
        right=0.95,
    )

    shape = max(RAMJET_SURROGATE_MIN_FACTOR, mach_factor * alt_factor)

    thrust_kN = (
        RAMJET_SURROGATE_DESIGN_THRUST_KN_PER_ENGINE
        * N_RAMJETS
        * design_mdot_ratio
        * flight_mdot_ratio
        * shape
    )

    if not np.isfinite(thrust_kN):
        return np.nan
    return max(float(thrust_kN), 0.0)



# =============================================================================
# 3A. SPARSE EXACT CALIBRATION FOR SURROGATE MAPS
# =============================================================================


def _safe_correction_factor(exact_kN: float, raw_surrogate_kN: float) -> float:
    """Return clipped exact/raw correction factor for calibration tables."""
    try:
        exact = float(exact_kN)
        raw = float(raw_surrogate_kN)
    except Exception:
        return np.nan
    if not np.isfinite(exact) or not np.isfinite(raw) or raw <= MIN_VALID_THRUST_KN or exact <= MIN_VALID_THRUST_KN:
        return np.nan
    corr = exact / raw
    if not np.isfinite(corr):
        return np.nan
    return float(np.clip(corr, SURROGATE_CORRECTION_MIN, SURROGATE_CORRECTION_MAX))


def _fill_and_interpolate_correction_table(table: np.ndarray, label: str) -> np.ndarray:
    """Fill missing correction factors and smooth obvious holes by nearest neighbor."""
    filled = complete_missing_values_for_interpolator(table, f"{label} calibration correction")
    return np.clip(filled, SURROGATE_CORRECTION_MIN, SURROGATE_CORRECTION_MAX)


def build_turbojet_surrogate_calibration(save_file: str | Path = TURBO_CALIBRATION_FILE) -> dict:
    """
    Build a sparse exact PyCycle correction table for the turbojet surrogate.

    The stored correction is:
        correction(design, altitude, Mach) = T_exact / T_raw_surrogate
    Later full maps use:
        T_calibrated = T_raw_surrogate * interpolated_correction
    """
    design_grid = np.asarray(TURBO_CAL_DESIGN_THRUST_GRID_LBF, dtype=float)
    mach_grid = np.asarray(TURBO_CAL_MACH_GRID, dtype=float)
    alt_grid_m = np.asarray(TURBO_CAL_ALT_GRID_M, dtype=float)
    correction = np.full((len(design_grid), len(alt_grid_m), len(mach_grid)), np.nan)
    exact_table = np.full_like(correction, np.nan, dtype=float)
    raw_table = np.full_like(correction, np.nan, dtype=float)

    print("\nBuilding sparse turbojet surrogate calibration")
    print("------------------------------------------------")
    print(f"Exact PyCycle samples: {correction.size}")
    print(f"Design samples: {design_grid}")
    print(f"Altitude samples: {alt_grid_m}")
    print(f"Mach samples: {mach_grid}")

    for d, design_lbf in enumerate(design_grid):
        configure_turbojet_design(design_lbf)
        for i, h in enumerate(alt_grid_m):
            for j, M in enumerate(mach_grid):
                raw = turbojet_fast_surrogate_thrust_kN(M, h, design_lbf)
                exact = turbojet_thrust_point_kN(M, h, design_lbf)
                corr = _safe_correction_factor(exact, raw)
                raw_table[d, i, j] = raw
                exact_table[d, i, j] = exact
                correction[d, i, j] = corr
                print(
                    f"turbo-cal design={design_lbf/1000:6.0f} klbf  "
                    f"h={h/1000:5.1f} km  M={M:4.2f}  "
                    f"raw={raw:9.2f} kN  exact={exact:9.2f} kN  C={corr:6.3f}"
                )

    correction = _fill_and_interpolate_correction_table(correction, "turbojet")
    np.savez(
        save_file,
        design_grid_lbf=design_grid,
        mach_grid=mach_grid,
        alt_grid_m=alt_grid_m,
        correction=correction,
        exact_thrust_kN=exact_table,
        raw_surrogate_thrust_kN=raw_table,
    )
    print(f"Saved turbojet surrogate calibration: {save_file}")
    return {
        "design_grid_lbf": design_grid,
        "mach_grid": mach_grid,
        "alt_grid_m": alt_grid_m,
        "correction": correction,
    }


def load_turbojet_surrogate_calibration(file: str | Path = TURBO_CALIBRATION_FILE) -> dict:
    data = np.load(file)
    return {
        "design_grid_lbf": data["design_grid_lbf"],
        "mach_grid": data["mach_grid"],
        "alt_grid_m": data["alt_grid_m"],
        "correction": data["correction"],
    }


@lru_cache(maxsize=1)
def get_turbojet_correction_interpolator() -> RegularGridInterpolator:
    rebuild = FORCE_REBUILD_SURROGATE_CALIBRATION or FORCE_REBUILD_TURBO_CALIBRATION or not Path(TURBO_CALIBRATION_FILE).exists()
    cal = build_turbojet_surrogate_calibration() if rebuild else load_turbojet_surrogate_calibration()
    if not rebuild:
        print(f"Loaded turbojet surrogate calibration: {TURBO_CALIBRATION_FILE}")
    return RegularGridInterpolator(
        points=(cal["design_grid_lbf"], cal["alt_grid_m"], cal["mach_grid"]),
        values=cal["correction"],
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


def turbojet_calibrated_surrogate_thrust_kN(mach: float, altitude_m: float, design_thrust_lbf_per_engine: float) -> float:
    raw = turbojet_fast_surrogate_thrust_kN(mach, altitude_m, design_thrust_lbf_per_engine)
    if not np.isfinite(raw):
        return np.nan
    corr = float(get_turbojet_correction_interpolator()([[design_thrust_lbf_per_engine, altitude_m, mach]])[0])
    if not np.isfinite(corr):
        corr = 1.0
    corr = float(np.clip(corr, SURROGATE_CORRECTION_MIN, SURROGATE_CORRECTION_MAX))
    return max(float(raw) * corr, 0.0)




def ramjet_calibration_mach_allowed(mach: float, altitude_m: float) -> bool:
    """Avoid exact ramjet calibration calls in regions that are known to stall.

    The skipped correction factors are left as NaN and then filled from nearby
    calibrated points so the full surrogate map can still cover the region.
    """
    M = float(mach)
    h = float(altitude_m)
    if h >= 27_500.0 and M > 3.0:
        return False
    if h >= 25_000.0 and M > 4.0:
        return False
    return True


def count_allowed_ramjet_calibration_points(design_grid: np.ndarray, alt_grid_m: np.ndarray, mach_grid: np.ndarray) -> int:
    count = 0
    for _design_mdot in design_grid:
        for h in alt_grid_m:
            for M in mach_grid:
                if ramjet_calibration_mach_allowed(M, h):
                    count += 1
    return count

def build_ramjet_surrogate_calibration(save_file: str | Path = RAMJET_CALIBRATION_FILE) -> dict:
    """
    Build a sparse exact ramjet correction table for the ramjet surrogate.

    The exact ramjet model is only evaluated on a sparse grid. The full sizing
    maps later use the calibrated surrogate instead of repeating all exact calls.
    """
    design_grid = np.asarray(RAMJET_CAL_DESIGN_MDOT_GRID_KG_S, dtype=float)
    mach_grid = np.asarray(RAMJET_CAL_MACH_GRID, dtype=float)
    alt_grid_m = np.asarray(RAMJET_CAL_ALT_GRID_M, dtype=float)
    correction = np.full((len(design_grid), len(alt_grid_m), len(mach_grid)), np.nan)
    exact_table = np.full_like(correction, np.nan, dtype=float)
    raw_table = np.full_like(correction, np.nan, dtype=float)

    print("\nBuilding sparse ramjet surrogate calibration")
    print("---------------------------------------------")
    exact_call_count = count_allowed_ramjet_calibration_points(design_grid, alt_grid_m, mach_grid)
    print(f"Exact ramjet samples requested before limits: {correction.size}")
    print(f"Exact ramjet samples after Mach-altitude limits: {exact_call_count}")
    print("Ramjet exact-call limits: at 27.5 km use Mach <= 3.0; at 25 km use Mach <= 4.0")
    print(f"Design mdot samples: {design_grid}")
    print(f"Altitude samples: {alt_grid_m}")
    print(f"Mach samples: {mach_grid}")

    for d, design_mdot in enumerate(design_grid):
        for i, h in enumerate(alt_grid_m):
            for j, M in enumerate(mach_grid):
                raw = ramjet_fast_surrogate_thrust_kN(M, h, design_mdot)
                raw_table[d, i, j] = raw

                if not ramjet_calibration_mach_allowed(M, h):
                    exact = np.nan
                    corr = np.nan
                    exact_table[d, i, j] = exact
                    correction[d, i, j] = corr
                    print(
                        f"ramjet-cal SKIP mdot={design_mdot:7.2f} kg/s  "
                        f"h={h/1000:5.1f} km  M={M:4.2f}  "
                        f"raw={raw:9.2f} kN  exact=   skipped  C=   fill"
                    )
                    continue

                exact = ramjet_thrust_point_kN(M, h, design_mdot)
                corr = _safe_correction_factor(exact, raw)
                exact_table[d, i, j] = exact
                correction[d, i, j] = corr
                print(
                    f"ramjet-cal mdot={design_mdot:7.2f} kg/s  "
                    f"h={h/1000:5.1f} km  M={M:4.2f}  "
                    f"raw={raw:9.2f} kN  exact={exact:9.2f} kN  C={corr:6.3f}"
                )

    correction = _fill_and_interpolate_correction_table(correction, "ramjet")
    np.savez(
        save_file,
        design_grid_mdot=design_grid,
        mach_grid=mach_grid,
        alt_grid_m=alt_grid_m,
        correction=correction,
        exact_thrust_kN=exact_table,
        raw_surrogate_thrust_kN=raw_table,
    )
    print(f"Saved ramjet surrogate calibration: {save_file}")
    return {
        "design_grid_mdot": design_grid,
        "mach_grid": mach_grid,
        "alt_grid_m": alt_grid_m,
        "correction": correction,
    }


def load_ramjet_surrogate_calibration(file: str | Path = RAMJET_CALIBRATION_FILE) -> dict:
    data = np.load(file)
    return {
        "design_grid_mdot": data["design_grid_mdot"],
        "mach_grid": data["mach_grid"],
        "alt_grid_m": data["alt_grid_m"],
        "correction": data["correction"],
    }


@lru_cache(maxsize=1)
def get_ramjet_correction_interpolator() -> RegularGridInterpolator:
    rebuild = FORCE_REBUILD_SURROGATE_CALIBRATION or FORCE_REBUILD_RAMJET_CALIBRATION or not Path(RAMJET_CALIBRATION_FILE).exists()
    cal = build_ramjet_surrogate_calibration() if rebuild else load_ramjet_surrogate_calibration()
    if not rebuild:
        print(f"Loaded ramjet surrogate calibration: {RAMJET_CALIBRATION_FILE}")
    return RegularGridInterpolator(
        points=(cal["design_grid_mdot"], cal["alt_grid_m"], cal["mach_grid"]),
        values=cal["correction"],
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


def ramjet_calibrated_surrogate_thrust_kN(mach: float, altitude_m: float, design_mdot_kg_s: float) -> float:
    raw = ramjet_fast_surrogate_thrust_kN(mach, altitude_m, design_mdot_kg_s)
    if not np.isfinite(raw):
        return np.nan
    corr = float(get_ramjet_correction_interpolator()([[design_mdot_kg_s, altitude_m, mach]])[0])
    if not np.isfinite(corr):
        corr = 1.0
    corr = float(np.clip(corr, SURROGATE_CORRECTION_MIN, SURROGATE_CORRECTION_MAX))
    return max(float(raw) * corr, 0.0)

def complete_missing_values_for_interpolator(table: np.ndarray, label: str) -> np.ndarray:
    """
    RegularGridInterpolator cannot work with NaNs in the grid.

    Bad or skipped values are not accepted as physical engine results. They are
    filled only after rejection so that interpolation can run numerically.
    """
    arr = np.array(table, dtype=float, copy=True)

    if np.all(~np.isfinite(arr)):
        raise RuntimeError(f"All values failed in {label} thrust table.")

    valid = np.isfinite(arr)
    if np.all(valid):
        return arr

    n_bad = np.size(arr) - np.count_nonzero(valid)
    print(f"WARNING: {label} table has {n_bad} invalid, skipped, or rejected points.")
    print("Those points are filled from nearest valid neighbors only for interpolation.")

    valid_indices = np.argwhere(valid)
    invalid_indices = np.argwhere(~valid)

    for idx in invalid_indices:
        distances = np.sum((valid_indices - idx) ** 2, axis=1)
        nearest = valid_indices[np.argmin(distances)]
        arr[tuple(idx)] = arr[tuple(nearest)]

    return arr


def time_budget_exceeded(start_time: float) -> bool:
    """Optional map-build wall-clock cutoff. Disabled when MAX_MAP_BUILD_SECONDS is None."""
    max_seconds = globals().get("MAX_MAP_BUILD_SECONDS", None)
    if max_seconds is None:
        return False
    return (time.time() - start_time) > float(max_seconds)


def build_turbojet_multidesign_map(save_file: str | Path = TURBO_MAP_FILE) -> dict:
    design_grid = np.asarray(TURBO_DESIGN_THRUST_GRID_LBF, dtype=float)
    mach_grid = np.asarray(MACH_GRID_TURBO, dtype=float)
    alt_grid_m = np.asarray(ALT_GRID_TURBO_M, dtype=float)
    table = np.full((len(design_grid), len(alt_grid_m), len(mach_grid)), np.nan)
    start_time = time.time()
    failed_points = 0

    print("\nBuilding complete turbojet maps")
    print("--------------------------------")
    print(f"Design points: {design_grid}")
    print(f"Grid per design: {len(alt_grid_m)} altitudes x {len(mach_grid)} Machs")
    mode_label = "surrogate" if USE_FAST_TURBOJET_SURROGATE else "exact PyCycle"
    print(f"Total turbojet map points: {table.size}")
    print(f"Turbojet map mode: {mode_label}")

    stop_early = False
    for d, design_lbf in enumerate(design_grid):
        if stop_early:
            break
        print(f"\nTurbojet design point {d + 1}/{len(design_grid)}: {design_lbf:,.0f} lbf per engine")
        configure_turbojet_design(design_lbf)
        for i, h in enumerate(alt_grid_m):
            if stop_early:
                break
            for j, M in enumerate(mach_grid):
                if time_budget_exceeded(start_time):
                    print("WARNING: map-build time budget exceeded during turbojet map generation.")
                    stop_early = True
                    break

                if USE_FAST_TURBOJET_SURROGATE:
                    if USE_CALIBRATED_TURBOJET_SURROGATE:
                        T_raw = turbojet_calibrated_surrogate_thrust_kN(M, h, design_lbf)
                        tag = "cal-surro"
                    else:
                        T_raw = turbojet_fast_surrogate_thrust_kN(M, h, design_lbf)
                        tag = "surrogate"
                else:
                    T_raw = turbojet_thrust_point_kN(M, h, design_lbf)
                    tag = "exact"

                T = sanitize_thrust_value_kN(T_raw, label="turbojet", design_value=design_lbf, mach=M, altitude_m=h)
                if not np.isfinite(T):
                    failed_points += 1
                table[d, i, j] = T
                print(f"turbo-{tag:9s} design={design_lbf/1000:6.0f} klbf  h={h/1000:5.1f} km  M={M:4.2f}  T={T:10.2f} kN")

                if failed_points > MAX_FAILED_TURBOJET_POINTS:
                    print("WARNING: too many failed turbojet points. Stopping turbojet map generation.")
                    stop_early = True
                    break

    if stop_early and not ALLOW_PARTIAL_MAPS_WHEN_TIMEOUT:
        raise RuntimeError("Turbojet map generation stopped early and partial maps are disabled.")

    table = filter_nonsensical_design_slices(table, design_grid, "turbojet")
    table = complete_missing_values_for_interpolator(table, "turbojet multidesign")
    np.savez(
        save_file,
        design_grid_lbf=design_grid,
        mach_grid=mach_grid,
        alt_grid_m=alt_grid_m,
        thrust_kN=table,
        n_engines=N_TURBOJETS,
    )
    print(f"Saved turbojet multidesign map: {save_file}")
    return {"design_grid_lbf": design_grid, "mach_grid": mach_grid, "alt_grid_m": alt_grid_m, "thrust_kN": table}


def build_ramjet_multidesign_map(save_file: str | Path = RAMJET_MAP_FILE) -> dict:
    design_grid = np.asarray(RAMJET_DESIGN_MDOT_GRID_KG_S, dtype=float)
    mach_grid = np.asarray(MACH_GRID_RAMJET, dtype=float)
    alt_grid_m = np.asarray(ALT_GRID_RAMJET_M, dtype=float)
    table = np.full((len(design_grid), len(alt_grid_m), len(mach_grid)), np.nan)
    start_time = time.time()
    failed_points = 0

    print("\nBuilding complete exact ramjet maps")
    print("-----------------------------------")
    print(f"Design mdot points: {design_grid}")
    print(f"Grid per design: {len(alt_grid_m)} altitudes x {len(mach_grid)} Machs")
    if USE_FAST_RAMJET_SURROGATE:
        print(f"Total ramjet surrogate evaluations: {table.size}")
        print("Ramjet surrogate is used. Exact ramjet map calls are skipped.")
    else:
        print(f"Total exact ramjet calls: {table.size}")
        print("No ramjet surrogate is used.")

    stop_early = False
    for d, design_mdot in enumerate(design_grid):
        if stop_early:
            break
        print(f"\nRamjet design point {d + 1}/{len(design_grid)}: {design_mdot:.2f} kg/s")
        for i, h in enumerate(alt_grid_m):
            if stop_early:
                break
            for j, M in enumerate(mach_grid):
                if time_budget_exceeded(start_time):
                    print("WARNING: map-build time budget exceeded during ramjet map generation.")
                    stop_early = True
                    break

                if USE_FAST_RAMJET_SURROGATE:
                    if USE_CALIBRATED_RAMJET_SURROGATE:
                        tag = "cal-s"
                        T_raw = ramjet_calibrated_surrogate_thrust_kN(M, h, design_mdot)
                    else:
                        tag = "surro"
                        T_raw = ramjet_fast_surrogate_thrust_kN(M, h, design_mdot)
                else:
                    tag = "fast" if use_fast_ramjet_solver_settings(M, h) else "exact"
                    T_raw = ramjet_thrust_point_kN(M, h, design_mdot)

                T = sanitize_thrust_value_kN(T_raw, label="ramjet", design_value=design_mdot, mach=M, altitude_m=h)
                if not np.isfinite(T):
                    failed_points += 1
                table[d, i, j] = T
                print(f"ramjet {tag:5s}  mdot={design_mdot:7.2f} kg/s  h={h/1000:5.1f} km  M={M:4.2f}  T={T:10.2f} kN")

                if failed_points > MAX_FAILED_RAMJET_POINTS:
                    print("WARNING: too many failed ramjet points. Stopping ramjet map generation.")
                    stop_early = True
                    break

    if stop_early and not ALLOW_PARTIAL_MAPS_WHEN_TIMEOUT:
        raise RuntimeError("Ramjet map generation stopped early and partial maps are disabled.")

    table = complete_missing_values_for_interpolator(table, "ramjet multidesign")
    np.savez(
        save_file,
        design_grid_mdot=design_grid,
        mach_grid=mach_grid,
        alt_grid_m=alt_grid_m,
        thrust_kN=table,
        phi=RAMJET_PHI,
        n_engines=N_RAMJETS,
    )
    print(f"Saved ramjet multidesign map: {save_file}")
    return {"design_grid_mdot": design_grid, "mach_grid": mach_grid, "alt_grid_m": alt_grid_m, "thrust_kN": table}


def load_turbo_map(file: str | Path) -> dict:
    data = np.load(file)
    return {
        "design_grid_lbf": data["design_grid_lbf"],
        "mach_grid": data["mach_grid"],
        "alt_grid_m": data["alt_grid_m"],
        "thrust_kN": data["thrust_kN"],
    }


def load_ramjet_map(file: str | Path) -> dict:
    data = np.load(file)
    return {
        "design_grid_mdot": data["design_grid_mdot"],
        "mach_grid": data["mach_grid"],
        "alt_grid_m": data["alt_grid_m"],
        "thrust_kN": data["thrust_kN"],
    }


def get_or_build_maps(force_rebuild: bool = FORCE_REBUILD_MAPS) -> tuple[dict, dict]:
    rebuild_turbo = force_rebuild or FORCE_REBUILD_TURBO_MAP
    rebuild_ramjet = force_rebuild or FORCE_REBUILD_RAMJET_MAP

    if rebuild_turbo or not Path(TURBO_MAP_FILE).exists():
        turbo_map = build_turbojet_multidesign_map()
    else:
        turbo_map = load_turbo_map(TURBO_MAP_FILE)
        print(f"Loaded existing turbojet multidesign map: {TURBO_MAP_FILE}")

    if rebuild_ramjet or not Path(RAMJET_MAP_FILE).exists():
        ramjet_map = build_ramjet_multidesign_map()
    else:
        ramjet_map = load_ramjet_map(RAMJET_MAP_FILE)
        print(f"Loaded existing ramjet multidesign map: {RAMJET_MAP_FILE}")

    return turbo_map, ramjet_map


# =============================================================================
# 4. INTERPOLATORS
# =============================================================================

@dataclass
class MultiDesignMaps:
    turbo_map: dict
    ramjet_map: dict
    turbo_interp_3d: RegularGridInterpolator
    ramjet_interp_3d: RegularGridInterpolator


def make_3d_interpolator(design_grid: np.ndarray, alt_grid: np.ndarray, mach_grid: np.ndarray, table: np.ndarray, label: str) -> RegularGridInterpolator:
    expected = (len(design_grid), len(alt_grid), len(mach_grid))
    if table.shape != expected:
        raise ValueError(f"{label} table shape mismatch. Expected {expected}, got {table.shape}.")
    fill_value = None if ALLOW_EXTRAPOLATION else np.nan
    return RegularGridInterpolator(
        points=(design_grid, alt_grid, mach_grid),
        values=table,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )


def prepare_multidesign_maps() -> MultiDesignMaps:
    turbo_map, ramjet_map = get_or_build_maps()
    turbo_interp = make_3d_interpolator(
        turbo_map["design_grid_lbf"], turbo_map["alt_grid_m"], turbo_map["mach_grid"], turbo_map["thrust_kN"], "turbojet"
    )
    ramjet_interp = make_3d_interpolator(
        ramjet_map["design_grid_mdot"], ramjet_map["alt_grid_m"], ramjet_map["mach_grid"], ramjet_map["thrust_kN"], "ramjet"
    )
    return MultiDesignMaps(turbo_map=turbo_map, ramjet_map=ramjet_map, turbo_interp_3d=turbo_interp, ramjet_interp_3d=ramjet_interp)


def transition_mach_candidates() -> np.ndarray:
    """Return transition Mach candidates that are covered by both engine maps."""
    if ENABLE_TRANSITION_MACH_SEARCH:
        candidates = np.asarray(TRANSITION_MACH_GRID, dtype=float)
    else:
        candidates = np.array([float(DEFAULT_TRANSITION_MACH)])

    turbo_m_min = float(np.min(MACH_GRID_TURBO))
    turbo_m_max = float(np.max(MACH_GRID_TURBO))
    ramjet_m_min = float(np.min(MACH_GRID_RAMJET))
    ramjet_m_max = float(np.max(MACH_GRID_RAMJET))
    valid = candidates[
        (candidates >= turbo_m_min)
        & (candidates <= turbo_m_max)
        & (candidates >= ramjet_m_min)
        & (candidates <= ramjet_m_max)
    ]
    if len(valid) == 0:
        raise ValueError("No transition Mach candidates are covered by both thrust maps.")
    return valid


def thrust_available_kN(
    maps: MultiDesignMaps,
    mach: float,
    altitude_m: float,
    turbo_design_lbf: float,
    ramjet_design_mdot: float,
    transition_mach: float = DEFAULT_TRANSITION_MACH,
) -> tuple[str, float]:
    if mach < transition_mach:
        mode = "turbojet"
        T = maps.turbo_interp_3d([[turbo_design_lbf, altitude_m, mach]])[0]
    else:
        mode = "ramjet"
        T = maps.ramjet_interp_3d([[ramjet_design_mdot, altitude_m, mach]])[0]
    T = float(T)
    if not np.isfinite(T):
        return mode, np.nan
    return mode, max(T, 0.0)


# =============================================================================
# 5. FLIGHT PROFILE PARAMETERIZATION
# =============================================================================


def unpack_altitude_knots(x: np.ndarray) -> np.ndarray:
    h = np.zeros_like(MACH_KNOTS, dtype=float)
    free_i = 0
    for i in range(len(MACH_KNOTS)):
        if i in FIXED_ALT_KNOTS:
            h[i] = FIXED_ALT_KNOTS[i]
        else:
            h[i] = x[free_i]
            free_i += 1
    return h


def make_profile(x: np.ndarray, n_eval: int = N_EVAL) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_knots = unpack_altitude_knots(x)
    interp = PchipInterpolator(MACH_KNOTS, h_knots, extrapolate=False)
    mach_grid = np.linspace(M_TAKEOFF, M_CRUISE, n_eval)
    altitude_grid = np.asarray(interp(mach_grid), dtype=float)
    return mach_grid, altitude_grid, MACH_KNOTS, h_knots


def altitude_bounds_for_knots(transition_mach: float = DEFAULT_TRANSITION_MACH) -> list[tuple[float, float]]:
    bounds = []
    free_indices = [i for i in range(len(MACH_KNOTS)) if i not in FIXED_ALT_KNOTS]
    for i in free_indices:
        M = MACH_KNOTS[i]
        if M < 1.0:
            bounds.append((0.0, 10_000.0))
        elif M < 2.0:
            bounds.append((1_000.0, 18_000.0))
        elif M < transition_mach:
            bounds.append((5_000.0, 26_000.0))
        elif M < M_CRUISE:
            bounds.append((18_000.0, H_CRUISE_M))
        else:
            bounds.append((H_CRUISE_M, H_CRUISE_M))
    return bounds


# =============================================================================
# 6. PROFILE EVALUATION FOR A FIXED DESIGN PAIR
# =============================================================================

@dataclass
class ProfileResult:
    feasible: bool
    objective: float
    design_score: float
    turbo_design_lbf_per_engine: float
    ramjet_design_mdot_kg_s: float
    transition_mach: float
    penalty: float
    max_deficit_kN: float
    min_margin_kN: float
    table: pd.DataFrame
    x_profile: np.ndarray


def design_score(turbo_design_lbf: float, ramjet_design_mdot: float) -> float:
    return W_TURBO_DESIGN * (turbo_design_lbf / BASE_TURBO_DESIGN_THRUST_LBF) + W_RAMJET_DESIGN * (ramjet_design_mdot / BASE_RAMJET_DESIGN_MDOT)


def evaluate_profile_for_design(x_profile: np.ndarray, maps: MultiDesignMaps, turbo_design_lbf: float, ramjet_design_mdot: float, transition_mach: float = DEFAULT_TRANSITION_MACH) -> ProfileResult:
    mach_grid, h_grid, _, h_knots = make_profile(x_profile)
    rows = []
    penalty = 0.0

    # Smooth monotonic-climb and altitude-range penalties.
    dh_knots = np.diff(h_knots)
    penalty += 1e5 * np.sum(np.minimum(dh_knots, 0.0) ** 2) / 1e8
    penalty += 1e5 * np.sum(np.maximum(-h_grid, 0.0) ** 2) / 1e8
    penalty += 1e5 * np.sum(np.maximum(h_grid - H_CRUISE_M, 0.0) ** 2) / 1e8

    margins = []
    deficits = []

    for M, h in zip(mach_grid, h_grid):
        aero = drag_and_required_thrust_kN(M, h, accel_g=ACCEL_G_TARGET)
        required_kN = aero["thrust_req_kN"] * (1.0 + THRUST_MARGIN)

        if CL_MAX is not None:
            penalty += 1e4 * max(0.0, aero["CL"] - CL_MAX) ** 2
        if Q_MAX_PA is not None:
            penalty += 1e-4 * max(0.0, aero["q_Pa"] - Q_MAX_PA) ** 2

        mode, available_kN = thrust_available_kN(maps, M, h, turbo_design_lbf, ramjet_design_mdot, transition_mach)
        if not np.isfinite(available_kN):
            available_kN = 0.0
            penalty += BIG_PENALTY

        margin = available_kN - required_kN
        deficit = max(0.0, -margin)
        margins.append(margin)
        deficits.append(deficit)

        rows.append({
            **aero,
            "mode": mode,
            "transition_mach": float(transition_mach),
            "T_available_kN": available_kN,
            "T_required_with_margin_kN": required_kN,
            "thrust_margin_kN": margin,
            "thrust_deficit_kN": deficit,
        })

    max_deficit = float(np.max(deficits)) if deficits else np.inf
    min_margin = float(np.min(margins)) if margins else -np.inf

    # For fixed design map-pair optimization, the goal is feasibility plus a
    # reasonable path. Do NOT include design score here; all profiles in this
    # call have the same design pair.
    objective = (
        penalty
        + 1e3 * max_deficit**2
        + 1.0 * np.sum(np.asarray(deficits) ** 2)
        - 1e-3 * min_margin
    )

    feasible = (max_deficit <= 1e-6) and (penalty < 1.0)
    df = pd.DataFrame(rows)
    return ProfileResult(
        feasible=feasible,
        objective=float(objective),
        design_score=design_score(turbo_design_lbf, ramjet_design_mdot),
        turbo_design_lbf_per_engine=float(turbo_design_lbf),
        ramjet_design_mdot_kg_s=float(ramjet_design_mdot),
        transition_mach=float(transition_mach),
        penalty=float(penalty),
        max_deficit_kN=max_deficit,
        min_margin_kN=min_margin,
        table=df,
        x_profile=np.asarray(x_profile, dtype=float),
    )


def objective_profile_only(x_profile: np.ndarray, maps: MultiDesignMaps, turbo_design_lbf: float, ramjet_design_mdot: float, transition_mach: float = DEFAULT_TRANSITION_MACH) -> float:
    return evaluate_profile_for_design(x_profile, maps, turbo_design_lbf, ramjet_design_mdot, transition_mach).objective


def optimize_profile_for_fixed_design(maps: MultiDesignMaps, turbo_design_lbf: float, ramjet_design_mdot: float, transition_mach: float = DEFAULT_TRANSITION_MACH, seed_offset: int = 0) -> ProfileResult:
    bounds = altitude_bounds_for_knots(transition_mach)
    result_de = differential_evolution(
        func=lambda z: objective_profile_only(z, maps, turbo_design_lbf, ramjet_design_mdot, transition_mach),
        bounds=bounds,
        strategy="best1bin",
        maxiter=DE_MAXITER_PER_MAP,
        popsize=DE_POPSIZE_PER_MAP,
        tol=0.008,
        polish=False,
        seed=RANDOM_SEED + seed_offset,
        workers=1,
        updating="immediate",
    )

    result_local = minimize(
        fun=lambda z: objective_profile_only(z, maps, turbo_design_lbf, ramjet_design_mdot, transition_mach),
        x0=result_de.x,
        bounds=bounds,
        method="Nelder-Mead",
        options={"maxiter": LOCAL_MAXITER_PER_MAP, "xatol": 2.0, "fatol": 1e-5, "disp": False},
    )

    x_best = result_local.x if result_local.fun < result_de.fun else result_de.x
    return evaluate_profile_for_design(x_best, maps, turbo_design_lbf, ramjet_design_mdot, transition_mach)


# =============================================================================
# 7. DISCRETE MAP-PAIR SEARCH: THIS IMPLEMENTS "OPTIMAL PATHS PER MAP"
# =============================================================================


def optimize_all_discrete_map_pairs(maps: MultiDesignMaps) -> tuple[ProfileResult, pd.DataFrame, list[ProfileResult]]:
    turbo_grid = maps.turbo_map["design_grid_lbf"]
    ramjet_grid = maps.ramjet_map["design_grid_mdot"]
    switch_grid = transition_mach_candidates()
    results: list[ProfileResult] = []
    summary_rows = []

    print("\nOptimizing one flight path per complete design-map pair and transition Mach")
    print("-----------------------------------------------------------------------")
    print(f"Turbojet design points : {len(turbo_grid)}")
    print(f"Ramjet design points   : {len(ramjet_grid)}")
    print(f"Transition Mach points : {len(switch_grid)} -> {np.array2string(switch_grid, precision=2)}")
    print(f"Total cases            : {len(turbo_grid) * len(ramjet_grid) * len(switch_grid)}")

    pair_index = 0
    total_cases = len(turbo_grid) * len(ramjet_grid) * len(switch_grid)
    for turbo_design in turbo_grid:
        for ramjet_design in ramjet_grid:
            for transition_mach in switch_grid:
                pair_index += 1
                print(
                    f"\nCase {pair_index}/{total_cases}: "
                    f"turbo={turbo_design:,.0f} lbf/engine, "
                    f"ramjet mdot={ramjet_design:.1f} kg/s, "
                    f"transition M={transition_mach:.2f}"
                )
                res = optimize_profile_for_fixed_design(
                    maps, turbo_design, ramjet_design, transition_mach, seed_offset=pair_index
                )
                results.append(res)
                summary_rows.append({
                    "turbo_design_lbf_per_engine": res.turbo_design_lbf_per_engine,
                    "ramjet_design_mdot_kg_s": res.ramjet_design_mdot_kg_s,
                    "transition_mach": res.transition_mach,
                    "feasible": res.feasible,
                    "design_score": res.design_score,
                    "profile_objective": res.objective,
                    "penalty": res.penalty,
                    "max_deficit_kN": res.max_deficit_kN,
                    "min_margin_kN": res.min_margin_kN,
                })
                print(
                    f"Result: feasible={res.feasible}, design_score={res.design_score:.3f}, "
                    f"transition M={res.transition_mach:.2f}, "
                    f"max_deficit={res.max_deficit_kN:.2f} kN, min_margin={res.min_margin_kN:.2f} kN"
                )

    summary = pd.DataFrame(summary_rows)
    feasible = [r for r in results if r.feasible]
    if feasible:
        best = min(feasible, key=lambda r: (r.design_score, r.transition_mach, -r.min_margin_kN))
    else:
        print("\nWARNING: No feasible discrete case found. Returning least-infeasible result.")
        best = min(results, key=lambda r: (r.max_deficit_kN, r.penalty, r.design_score))
    return best, summary, results


def _local_bounds_around_value(grid: np.ndarray, value: float, neighbor_steps: int = 1) -> tuple[float, float]:
    """Return a local interpolation interval around the nearest complete-map point."""
    grid = np.asarray(grid, dtype=float)
    i = int(np.argmin(np.abs(grid - float(value))))
    lo_i = max(0, i - int(neighbor_steps))
    hi_i = min(len(grid) - 1, i + int(neighbor_steps))
    return float(grid[lo_i]), float(grid[hi_i])


def make_interpolated_design_grids(
    maps: MultiDesignMaps,
    center_result: ProfileResult | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a denser design grid between complete engine maps.

    If LOCAL_INTERP_AROUND_BEST_ONLY is enabled, the dense grid is limited to
    the local box around the best coarse discrete result. This focuses the
    expensive trajectory optimizations on the region most likely to contain the
    ideal configuration, while still using only interpolated thrust maps.
    """
    turbo_grid = np.asarray(maps.turbo_map["design_grid_lbf"], dtype=float)
    ramjet_grid = np.asarray(maps.ramjet_map["design_grid_mdot"], dtype=float)

    if LOCAL_INTERP_AROUND_BEST_ONLY and center_result is not None:
        turbo_lo, turbo_hi = _local_bounds_around_value(
            turbo_grid, center_result.turbo_design_lbf_per_engine, LOCAL_INTERP_NEIGHBOR_STEPS
        )
        ramjet_lo, ramjet_hi = _local_bounds_around_value(
            ramjet_grid, center_result.ramjet_design_mdot_kg_s, LOCAL_INTERP_NEIGHBOR_STEPS
        )
    else:
        turbo_lo, turbo_hi = float(turbo_grid[0]), float(turbo_grid[-1])
        ramjet_lo, ramjet_hi = float(ramjet_grid[0]), float(ramjet_grid[-1])

    turbo_dense = np.linspace(turbo_lo, turbo_hi, int(INTERP_TURBO_DESIGN_POINTS))
    ramjet_dense = np.linspace(ramjet_lo, ramjet_hi, int(INTERP_RAMJET_DESIGN_POINTS))

    # Avoid tiny floating-point duplicates around the original map points.
    turbo_dense = np.unique(np.round(turbo_dense, 8))
    ramjet_dense = np.unique(np.round(ramjet_dense, 8))
    return turbo_dense, ramjet_dense


def make_local_transition_grid(center_result: ProfileResult | None = None) -> np.ndarray:
    """Return transition Mach candidates, optionally refined near the best coarse one."""
    coarse = transition_mach_candidates()
    if not (LOCAL_INTERP_AROUND_BEST_ONLY and center_result is not None):
        return coarse

    lo, hi = _local_bounds_around_value(coarse, center_result.transition_mach, 1)
    refined = np.linspace(lo, hi, int(LOCAL_INTERP_TRANSITION_POINTS))
    combined = np.unique(np.round(np.concatenate([refined, [center_result.transition_mach]]), 8))
    return combined


def optimize_all_interpolated_design_pairs(
    maps: MultiDesignMaps,
    center_result: ProfileResult | None = None,
) -> tuple[ProfileResult, pd.DataFrame, list[ProfileResult]]:
    """
    Optimize one flight path for each interpolated design pair and transition Mach.

    Available thrust at intermediate design values comes from the complete-map
    interpolators. The transition Mach is not interpolated; it is treated as a
    discrete operational/design choice.
    """
    turbo_grid, ramjet_grid = make_interpolated_design_grids(maps, center_result=center_result)
    switch_grid = make_local_transition_grid(center_result=center_result)
    results: list[ProfileResult] = []
    summary_rows = []

    print("\nOptimizing one flight path per INTERPOLATED design pair and transition Mach")
    print("--------------------------------------------------------------------------")
    print(f"Interpolated turbojet design points: {len(turbo_grid)}")
    print(f"Interpolated ramjet design points  : {len(ramjet_grid)}")
    print(f"Transition Mach points             : {len(switch_grid)} -> {np.array2string(switch_grid, precision=2)}")
    print(f"Total interpolated cases           : {len(turbo_grid) * len(ramjet_grid) * len(switch_grid)}")
    if LOCAL_INTERP_AROUND_BEST_ONLY and center_result is not None:
        print("Local interpolation is centered on the best coarse discrete case:")
        print(
            f"  turbo={center_result.turbo_design_lbf_per_engine:,.0f} lbf/engine, "
            f"ramjet mdot={center_result.ramjet_design_mdot_kg_s:.1f} kg/s, "
            f"transition M={center_result.transition_mach:.2f}"
        )
    print("These use thrust interpolation between complete maps; no new engine maps are built.")

    pair_index = 0
    total_cases = len(turbo_grid) * len(ramjet_grid) * len(switch_grid)
    for turbo_design in turbo_grid:
        for ramjet_design in ramjet_grid:
            for transition_mach in switch_grid:
                pair_index += 1
                print(
                    f"\nInterpolated case {pair_index}/{total_cases}: "
                    f"turbo={turbo_design:,.0f} lbf/engine, "
                    f"ramjet mdot={ramjet_design:.1f} kg/s, "
                    f"transition M={transition_mach:.2f}"
                )
                res = optimize_profile_for_fixed_design(
                    maps, turbo_design, ramjet_design, transition_mach, seed_offset=10_000 + pair_index
                )
                results.append(res)
                summary_rows.append({
                    "source": "interpolated_design_pair",
                    "turbo_design_lbf_per_engine": res.turbo_design_lbf_per_engine,
                    "ramjet_design_mdot_kg_s": res.ramjet_design_mdot_kg_s,
                    "transition_mach": res.transition_mach,
                    "feasible": res.feasible,
                    "design_score": res.design_score,
                    "profile_objective": res.objective,
                    "penalty": res.penalty,
                    "max_deficit_kN": res.max_deficit_kN,
                    "min_margin_kN": res.min_margin_kN,
                })
                print(
                    f"Result: feasible={res.feasible}, design_score={res.design_score:.3f}, "
                    f"transition M={res.transition_mach:.2f}, "
                    f"max_deficit={res.max_deficit_kN:.2f} kN, min_margin={res.min_margin_kN:.2f} kN"
                )

    summary = pd.DataFrame(summary_rows)
    feasible = [r for r in results if r.feasible]
    if feasible:
        best = min(feasible, key=lambda r: (r.design_score, r.transition_mach, -r.min_margin_kN))
    else:
        print("\nWARNING: No feasible interpolated case found. Returning least-infeasible interpolated result.")
        best = min(results, key=lambda r: (r.max_deficit_kN, r.penalty, r.design_score))
    return best, summary, results


# =============================================================================
# 8. OPTIONAL INTERPOLATED DESIGN REFINEMENT BETWEEN COMPLETE MAPS
# =============================================================================


def interpolated_design_bounds_around_best(maps: MultiDesignMaps, best: ProfileResult) -> list[tuple[float, float]]:
    turbo_grid = maps.turbo_map["design_grid_lbf"]
    ramjet_grid = maps.ramjet_map["design_grid_mdot"]

    if not LOCAL_REFINEMENT_AROUND_BEST_ONLY:
        return [(float(turbo_grid[0]), float(turbo_grid[-1])), (float(ramjet_grid[0]), float(ramjet_grid[-1]))]

    i_t = int(np.argmin(np.abs(turbo_grid - best.turbo_design_lbf_per_engine)))
    i_r = int(np.argmin(np.abs(ramjet_grid - best.ramjet_design_mdot_kg_s)))
    lo_t = max(0, i_t - LOCAL_REFINEMENT_NEIGHBOR_STEPS)
    hi_t = min(len(turbo_grid) - 1, i_t + LOCAL_REFINEMENT_NEIGHBOR_STEPS)
    lo_r = max(0, i_r - LOCAL_REFINEMENT_NEIGHBOR_STEPS)
    hi_r = min(len(ramjet_grid) - 1, i_r + LOCAL_REFINEMENT_NEIGHBOR_STEPS)
    return [(float(turbo_grid[lo_t]), float(turbo_grid[hi_t])), (float(ramjet_grid[lo_r]), float(ramjet_grid[hi_r]))]


def pack_design_and_profile(turbo_lbf: float, ramjet_mdot: float, x_profile: np.ndarray) -> np.ndarray:
    return np.concatenate([[turbo_lbf, ramjet_mdot], np.asarray(x_profile, dtype=float)])


def unpack_design_and_profile(z: np.ndarray) -> tuple[float, float, np.ndarray]:
    return float(z[0]), float(z[1]), np.asarray(z[2:], dtype=float)


def evaluate_design_and_profile(z: np.ndarray, maps: MultiDesignMaps, transition_mach: float = DEFAULT_TRANSITION_MACH) -> ProfileResult:
    turbo_lbf, ramjet_mdot, x_profile = unpack_design_and_profile(z)
    res = evaluate_profile_for_design(x_profile, maps, turbo_lbf, ramjet_mdot, transition_mach)
    # Continuous refinement objective: design score plus heavy feasibility penalties.
    res.objective = (
        res.design_score
        + 1e3 * res.max_deficit_kN**2
        + res.penalty
        - 1e-4 * res.min_margin_kN
    )
    return res


def objective_design_and_profile(z: np.ndarray, maps: MultiDesignMaps, transition_mach: float = DEFAULT_TRANSITION_MACH) -> float:
    return evaluate_design_and_profile(z, maps, transition_mach).objective


def normalize_to_unit(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    if hi <= lo:
        return 0.0
    return float((value - lo) / (hi - lo))


def denormalize_from_unit(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    u = float(np.clip(value, 0.0, 1.0))
    return float(lo + u * (hi - lo))


def pack_normalized_design_and_profile(
    turbo_lbf: float,
    ramjet_mdot: float,
    x_profile: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    physical = np.concatenate([[turbo_lbf, ramjet_mdot], np.asarray(x_profile, dtype=float)])
    return np.array([normalize_to_unit(v, b) for v, b in zip(physical, bounds)], dtype=float)


def unpack_normalized_design_and_profile(
    u: np.ndarray,
    bounds: list[tuple[float, float]],
) -> tuple[float, float, np.ndarray]:
    physical = np.array([denormalize_from_unit(v, b) for v, b in zip(u, bounds)], dtype=float)
    return float(physical[0]), float(physical[1]), physical[2:]


def objective_normalized_design_and_profile(
    u: np.ndarray,
    maps: MultiDesignMaps,
    bounds: list[tuple[float, float]],
    transition_mach: float = DEFAULT_TRANSITION_MACH,
) -> float:
    turbo_lbf, ramjet_mdot, x_profile = unpack_normalized_design_and_profile(u, bounds)
    z = pack_design_and_profile(turbo_lbf, ramjet_mdot, x_profile)
    return objective_design_and_profile(z, maps, transition_mach)


def profile_is_feasible_for_designs(
    maps: MultiDesignMaps,
    x_profile: np.ndarray,
    turbo_lbf: float,
    ramjet_mdot: float,
    transition_mach: float = DEFAULT_TRANSITION_MACH,
) -> bool:
    res = evaluate_profile_for_design(x_profile, maps, turbo_lbf, ramjet_mdot, transition_mach)
    return (res.max_deficit_kN <= THRUST_FEASIBILITY_TOL_KN) and (res.penalty < 1.0)


def shrink_designs_for_fixed_profile(maps: MultiDesignMaps, result: ProfileResult) -> ProfileResult:
    """
    Reduce turbojet thrust and ramjet mdot continuously for the final profile.

    The discrete search finds a feasible map pair. This function then asks:
    keeping that optimized flight path fixed, how low can the interpolated
    design variables go while still satisfying thrust constraints?
    """
    if not ENABLE_FIXED_PROFILE_DESIGN_SHRINK:
        return result

    x_profile = result.x_profile
    transition_mach = result.transition_mach
    turbo_grid = maps.turbo_map["design_grid_lbf"]
    ramjet_grid = maps.ramjet_map["design_grid_mdot"]

    turbo_lbf = float(result.turbo_design_lbf_per_engine)
    ramjet_mdot = float(result.ramjet_design_mdot_kg_s)

    def shrink_one(current: float, low: float, setter: str) -> float:
        nonlocal turbo_lbf, ramjet_mdot

        # If even the lower bound is feasible, use it.
        if setter == "turbo":
            feasible_low = profile_is_feasible_for_designs(maps, x_profile, low, ramjet_mdot, transition_mach)
        else:
            feasible_low = profile_is_feasible_for_designs(maps, x_profile, turbo_lbf, low, transition_mach)

        if feasible_low:
            return float(low)

        lo = float(low)
        hi = float(current)

        for _ in range(DESIGN_SHRINK_BISECTION_ITERS):
            mid = 0.5 * (lo + hi)
            if setter == "turbo":
                feasible_mid = profile_is_feasible_for_designs(maps, x_profile, mid, ramjet_mdot, transition_mach)
            else:
                feasible_mid = profile_is_feasible_for_designs(maps, x_profile, turbo_lbf, mid, transition_mach)

            if feasible_mid:
                hi = mid
            else:
                lo = mid

        return float(hi)

    # Coordinate-shrink a few times because lowering one engine size can expose
    # the other as the active constraint.
    for _ in range(3):
        turbo_lbf = shrink_one(turbo_lbf, float(turbo_grid[0]), "turbo")
        ramjet_mdot = shrink_one(ramjet_mdot, float(ramjet_grid[0]), "ramjet")

    shrunk = evaluate_profile_for_design(x_profile, maps, turbo_lbf, ramjet_mdot, transition_mach)
    shrunk.feasible = (shrunk.max_deficit_kN <= THRUST_FEASIBILITY_TOL_KN) and (shrunk.penalty < 1.0)
    return shrunk


def refine_between_complete_maps(maps: MultiDesignMaps, best_discrete: ProfileResult) -> ProfileResult:
    design_bounds = interpolated_design_bounds_around_best(maps, best_discrete)
    profile_bounds = altitude_bounds_for_knots(best_discrete.transition_mach)
    physical_bounds = design_bounds + profile_bounds
    unit_bounds = [(0.0, 1.0)] * len(physical_bounds)

    u0 = pack_normalized_design_and_profile(
        best_discrete.turbo_design_lbf_per_engine,
        best_discrete.ramjet_design_mdot_kg_s,
        best_discrete.x_profile,
        physical_bounds,
    )

    print("\nRefining design between complete maps by interpolation")
    print("------------------------------------------------------")
    print(f"Turbojet design bounds: {design_bounds[0][0]:,.0f} to {design_bounds[0][1]:,.0f} lbf/engine")
    print(f"Ramjet mdot bounds    : {design_bounds[1][0]:.1f} to {design_bounds[1][1]:.1f} kg/s")
    print("Using normalized design/profile variables, then shrinking final design values.")

    result = minimize(
        fun=lambda u: objective_normalized_design_and_profile(u, maps, physical_bounds, best_discrete.transition_mach),
        x0=u0,
        bounds=unit_bounds,
        method="Powell",
        options={"maxiter": 700, "xtol": 1e-4, "ftol": 1e-5, "disp": False},
    )

    u_best = result.x if result.fun < objective_normalized_design_and_profile(u0, maps, physical_bounds, best_discrete.transition_mach) else u0
    turbo_lbf, ramjet_mdot, x_profile = unpack_normalized_design_and_profile(u_best, physical_bounds)
    refined = evaluate_design_and_profile(pack_design_and_profile(turbo_lbf, ramjet_mdot, x_profile), maps, transition_mach=best_discrete.transition_mach)
    refined.feasible = (refined.max_deficit_kN <= THRUST_FEASIBILITY_TOL_KN) and (refined.penalty < 1.0)

    # Important final step: even if the optimizer stays near a grid value, this
    # bisection uses the interpolated maps to find the minimum continuous design
    # values for the final trajectory.
    refined = shrink_designs_for_fixed_profile(maps, refined)

    print(
        f"Interpolated refinement result: turbo={refined.turbo_design_lbf_per_engine:,.1f} lbf/engine, "
        f"ramjet mdot={refined.ramjet_design_mdot_kg_s:.3f} kg/s, "
        f"max_deficit={refined.max_deficit_kN:.6f} kN"
    )

    return refined


# =============================================================================
# 9. REPORTING
# =============================================================================


def print_map_quality(maps: MultiDesignMaps) -> None:
    print("\nMap sanity check")
    print("----------------")
    print(
        f"Turbo maps: min={np.nanmin(maps.turbo_map['thrust_kN']):.2f} kN, "
        f"max={np.nanmax(maps.turbo_map['thrust_kN']):.2f} kN"
    )
    print(
        f"Ramjet maps: min={np.nanmin(maps.ramjet_map['thrust_kN']):.2f} kN, "
        f"max={np.nanmax(maps.ramjet_map['thrust_kN']):.2f} kN"
    )


def plot_thrust_maps(
    maps: MultiDesignMaps,
    best: ProfileResult,
    *,
    n_mach: int = 180,
    n_alt: int = 160,
    save_png: bool = True,
    prefix: str = "optimized_flight_profile_multimaps",
) -> None:
    """
    Plot Mach-altitude heatmaps for the selected final design.

    Map layout:
        x-axis  = Mach number
        y-axis  = altitude [km]
        colour  = thrust [kN]

    Available thrust is taken from the selected turbojet/ramjet maps:
        - turbojet below best.transition_mach
        - ramjet at and above best.transition_mach

    Required thrust is computed from the drag model at each Mach-altitude point.
    """
    import matplotlib.pyplot as plt

    mach_values = np.linspace(M_TAKEOFF, M_CRUISE, n_mach)
    alt_values_m = np.linspace(0.0, H_CRUISE_M, n_alt)
    M_grid, H_grid = np.meshgrid(mach_values, alt_values_m)

    available = np.full_like(M_grid, np.nan, dtype=float)
    required = np.full_like(M_grid, np.nan, dtype=float)

    for i in range(H_grid.shape[0]):
        for j in range(H_grid.shape[1]):
            M = float(M_grid[i, j])
            h = float(H_grid[i, j])

            _, T_avail = thrust_available_kN(
                maps,
                mach=M,
                altitude_m=h,
                turbo_design_lbf=best.turbo_design_lbf_per_engine,
                ramjet_design_mdot=best.ramjet_design_mdot_kg_s,
                transition_mach=best.transition_mach,
            )
            available[i, j] = T_avail if np.isfinite(T_avail) else np.nan

            aero = drag_and_required_thrust_kN(M, h, accel_g=ACCEL_G_TARGET)
            required[i, j] = aero["thrust_req_kN"] * (1.0 + THRUST_MARGIN)

    # Mask invalid regions, for example the low-altitude ramjet region where
    # the ramjet map has no data.
    available_masked = np.ma.masked_invalid(available)
    required_masked = np.ma.masked_invalid(required)

    cmap_available = plt.get_cmap("viridis").copy()
    cmap_required = plt.get_cmap("viridis").copy()
    cmap_available.set_bad(color="white")
    cmap_required.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    mesh = ax.pcolormesh(
        mach_values,
        alt_values_m / 1000.0,
        available_masked,
        shading="auto",
        cmap=cmap_available,
    )
    fig.colorbar(mesh, ax=ax, label="Available thrust [kN]")
    ax.axvline(best.transition_mach, linestyle="--", label="Turbojet/ramjet switch")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Available thrust map")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_png:
        fig.savefig(f"{prefix}_available_thrust_map.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    mesh = ax.pcolormesh(
        mach_values,
        alt_values_m / 1000.0,
        required_masked,
        shading="auto",
        cmap=cmap_required,
    )
    fig.colorbar(mesh, ax=ax, label="Required thrust [kN]")
    ax.axvline(best.transition_mach, linestyle="--", label="Turbojet/ramjet switch")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Required thrust map")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_png:
        fig.savefig(f"{prefix}_required_thrust_map.png", dpi=300, bbox_inches="tight")

    if save_png:
        print(f"Saved available thrust map PNG: {prefix}_available_thrust_map.png")
        print(f"Saved required thrust map PNG: {prefix}_required_thrust_map.png")


def save_and_show_outputs(maps: MultiDesignMaps, best: ProfileResult, summary: pd.DataFrame, prefix: str = "optimized_flight_profile_multimaps") -> None:
    mach_grid, h_grid, mach_knots, h_knots = make_profile(best.x_profile)
    knots = pd.DataFrame({"Mach": mach_knots, "Altitude_m": h_knots, "Altitude_km": h_knots / 1000.0})

    print("\n" + "=" * 78)
    print("OPTIMIZED ENGINE SIZING FROM COMPLETE MULTI-DESIGN THRUST MAPS")
    print("=" * 78)
    print(f"Feasible result                  : {best.feasible}")
    print(f"Turbojet design thrust/engine    : {best.turbo_design_lbf_per_engine:,.0f} lbf")
    print(f"Turbojet design thrust/engine    : {best.turbo_design_lbf_per_engine * 0.0044482216152605:,.2f} kN")
    print(f"Ramjet design air mass flow      : {best.ramjet_design_mdot_kg_s:,.2f} kg/s")
    print(f"Turbojet/ramjet transition Mach  : {best.transition_mach:.2f}")
    if N_RAMJETS > 0:
        print(f"Ramjet design air mass flow/engine: {best.ramjet_design_mdot_kg_s / N_RAMJETS:,.2f} kg/s")
    print(f"Design score                     : {best.design_score:.6g}")
    print(f"Max thrust deficit               : {best.max_deficit_kN:.3f} kN")
    print(f"Minimum thrust margin            : {best.min_margin_kN:.3f} kN")
    print(f"Penalty                          : {best.penalty:.6g}")
    print(f"Turbojet surrogate used          : {USE_FAST_TURBOJET_SURROGATE}")
    print(f"Turbojet surrogate calibrated    : {USE_CALIBRATED_TURBOJET_SURROGATE}")
    print(f"Ramjet surrogate used            : {USE_FAST_RAMJET_SURROGATE}")
    print(f"Ramjet surrogate calibrated      : {USE_CALIBRATED_RAMJET_SURROGATE}")
    print("=" * 78)

    print("\nAltitude knots:")
    print(knots.to_string(index=False))

    print("\nWorst thrust margins:")
    worst = best.table.nsmallest(8, "thrust_margin_kN")
    print(
        worst[[
            "mach", "altitude_m", "mode", "thrust_req_kN", "T_available_kN",
            "thrust_margin_kN", "CL", "q_Pa",
        ]].to_string(index=False)
    )

    summary_file = f"{prefix}_design_pair_summary.csv"
    profile_file = f"{prefix}_best_profile.csv"
    summary.to_csv(summary_file, index=False)
    best.table.to_csv(profile_file, index=False)
    print(f"\nSaved summary CSV: {summary_file}")
    print(f"Saved best profile CSV: {profile_file}")

    try:
        import matplotlib.pyplot as plt

        df = best.table
        plt.figure(figsize=(9, 5))
        plt.plot(df["mach"], df["altitude_m"] / 1000.0, label="Optimized profile")
        plt.scatter(mach_knots, h_knots / 1000.0, label="Knots")
        plt.axvline(best.transition_mach, linestyle="--", label="Turbojet/ramjet switch")
        plt.xlabel("Mach")
        plt.ylabel("Altitude [km]")
        plt.title("Optimized Mach-altitude profile")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(9, 5))
        plt.plot(df["mach"], df["thrust_req_kN"], label="Required")
        plt.plot(df["mach"], df["T_available_kN"], label="Available from selected maps")
        plt.axvline(best.transition_mach, linestyle="--", label=f"M={best.transition_mach:.2f} switch")
        plt.xlabel("Mach")
        plt.ylabel("Thrust [kN]")
        plt.title("Thrust constraint")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(df["mach"], df["CL"], label="CL required")
        if CL_MAX is not None:
            ax1.axhline(CL_MAX, linestyle="--", label="CL limit")
        ax1.set_xlabel("Mach")
        ax1.set_ylabel("CL")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(df["mach"], df["q_Pa"] / 1000.0, linestyle=":", label="q")
        if Q_MAX_PA is not None:
            ax2.axhline(Q_MAX_PA / 1000.0, linestyle="--", label="q limit")
        ax2.set_ylabel("Dynamic pressure [kPa]")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
        plt.title("Envelope checks")
        fig.tight_layout()

        plot_thrust_maps(maps, best, prefix=prefix)

        mission = build_full_mission_profile(best)
        print_full_mission_summary(mission)
        if SAVE_FULL_MISSION_CSV:
            save_full_mission_csv(mission, prefix)
        plot_full_mission_profiles(mission, best, prefix=prefix)

        plt.show()
    except Exception as exc:
        print(f"\nPlotting failed: {exc}")





# =============================================================================
# 9B. FULL MISSION RANGE, DESCENT, AND MACH-ALTITUDE PLOTS
# =============================================================================

def velocity_from_mach_altitude(mach: float, altitude_m: float) -> float:
    """Return true airspeed [m/s] from Mach and altitude using this file's atmosphere."""
    _, T = atmosphere_drag(float(altitude_m))
    return float(mach) * float(np.sqrt(GAMMA * R_GAS * T))


def build_ascent_range_from_profile(mach: np.ndarray, altitude_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the optimized Mach-altitude climb/acceleration path to range.

    Assumption:
        The aircraft accelerates along the flight path with approximately
        ACCEL_G_FOR_ASCENT_RANGE * g. For each segment,

            ds = (V_2^2 - V_1^2) / (2 a)
            dx = sqrt(ds^2 - dh^2)

        This is an approximate geometric reconstruction for plotting only.
    """
    mach = np.asarray(mach, dtype=float)
    altitude_m = np.asarray(altitude_m, dtype=float)

    accel = float(ACCEL_G_FOR_ASCENT_RANGE) * G
    velocity = np.array([
        velocity_from_mach_altitude(M, h)
        for M, h in zip(mach, altitude_m)
    ])

    x = np.zeros_like(mach, dtype=float)

    for i in range(1, len(mach)):
        V0 = velocity[i - 1]
        V1 = velocity[i]
        dh = altitude_m[i] - altitude_m[i - 1]

        dV2 = max(0.0, V1**2 - V0**2)
        ds = dV2 / max(2.0 * accel, 1e-12)
        dx = np.sqrt(max(ds**2 - dh**2, 0.0))
        x[i] = x[i - 1] + dx

    return x, velocity


def solve_minimum_range_unpowered_descent(
    h_cruise_m: float,
    u_cruise_m_s: float,
    accel_limit_m_s2: float | None = None,
) -> dict:
    """
    Solve the unpowered descent/deceleration problem described in the report.

    Horizontal acceleration is constant and negative:
        a_x = -b_x

    The descent range is minimized by maximizing b_x while satisfying:
        b_x^2 + a_y^2 <= a_max^2

    The vertical acceleration magnitude follows from covering half the cruise
    altitude during half the descent time:
        a_y = h_cruise / (u_cruise / (2 b_x))^2
            = 4 h_cruise b_x^2 / u_cruise^2

    The vertical acceleration is downward during the first half of descent and
    upward during the second half, so both vertical and horizontal velocity are
    zero at touchdown.
    """
    h = float(h_cruise_m)
    u = float(u_cruise_m_s)
    a_max = float(accel_limit_m_s2 if accel_limit_m_s2 is not None else DESCENT_ACCEL_LIMIT_G * G)

    if h <= 0.0:
        raise ValueError("h_cruise_m must be positive for descent calculation.")
    if u <= 0.0:
        raise ValueError("u_cruise_m_s must be positive for descent calculation.")
    if a_max <= 0.0:
        raise ValueError("accel_limit_m_s2 must be positive.")

    def constraint_value(bx: float) -> float:
        ay = 4.0 * h * bx**2 / max(u**2, 1e-12)
        return bx**2 + ay**2 - a_max**2

    # constraint_value(0) < 0 and constraint_value(a_max) > 0 for h > 0.
    lo = 0.0
    hi = a_max
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if constraint_value(mid) <= 0.0:
            lo = mid
        else:
            hi = mid

    bx = lo
    ax = -bx
    ay_mag = 4.0 * h * bx**2 / max(u**2, 1e-12)
    t_descent = u / max(bx, 1e-12)
    x_descent = u**2 / max(2.0 * bx, 1e-12)
    a_total = float(np.sqrt(bx**2 + ay_mag**2))

    return {
        "a_x_descent_m_s2": ax,
        "a_x_magnitude_m_s2": bx,
        "a_y_magnitude_m_s2": ay_mag,
        "a_total_m_s2": a_total,
        "a_limit_m_s2": a_max,
        "t_descent_s": t_descent,
        "x_descent_m": x_descent,
    }


def build_full_mission_profile(best: ProfileResult) -> dict[str, np.ndarray | float | bool]:
    """
    Build a full takeoff-to-landing profile:
      1) optimized powered climb/acceleration,
      2) constant Mach-5 cruise at H_CRUISE_M,
      3) unpowered minimum-range descent/deceleration.

    The cruise distance is chosen so that the full mission reaches
    TOTAL_MISSION_RANGE_M when possible. If the climb plus minimum descent range
    already exceeds TOTAL_MISSION_RANGE_M, cruise distance is set to zero and the
    final range is allowed to exceed the requested range.
    """
    df = best.table.sort_values("mach").reset_index(drop=True)
    mach_ascent = df["mach"].to_numpy(dtype=float)
    h_ascent = df["altitude_m"].to_numpy(dtype=float)

    x_ascent, V_ascent = build_ascent_range_from_profile(mach_ascent, h_ascent)

    h_cruise = float(H_CRUISE_M)
    M_cruise = float(M_CRUISE)
    u_cruise = velocity_from_mach_altitude(M_cruise, h_cruise)

    descent = solve_minimum_range_unpowered_descent(
        h_cruise_m=h_cruise,
        u_cruise_m_s=u_cruise,
        accel_limit_m_s2=DESCENT_ACCEL_LIMIT_G * G,
    )

    x_climb_end = float(x_ascent[-1])
    x_descent_range = float(descent["x_descent_m"])
    requested_total_range = float(TOTAL_MISSION_RANGE_M)
    cruise_range = requested_total_range - x_climb_end - x_descent_range
    cruise_fits = cruise_range >= 0.0
    if not cruise_fits:
        cruise_range = 0.0

    # Cruise segment.
    x_cruise_start = x_climb_end
    x_cruise_end = x_cruise_start + cruise_range
    x_cruise = np.linspace(x_cruise_start, x_cruise_end, max(2, int(N_CRUISE_POINTS)))
    h_cruise_arr = np.full_like(x_cruise, h_cruise, dtype=float)
    mach_cruise_arr = np.full_like(x_cruise, M_cruise, dtype=float)

    # Descent segment. Use downward displacement y measured positive downward.
    t_descent = float(descent["t_descent_s"])
    bx = float(descent["a_x_magnitude_m_s2"])
    ax = float(descent["a_x_descent_m_s2"])
    ay_mag = float(descent["a_y_magnitude_m_s2"])
    t = np.linspace(0.0, t_descent, max(2, int(N_DESCENT_POINTS)))
    half_t = 0.5 * t_descent

    x_descent = x_cruise_end + u_cruise * t + 0.5 * ax * t**2
    y_down = np.empty_like(t)
    v_y_down = np.empty_like(t)

    first_half = t <= half_t
    t1 = t[first_half]
    y_down[first_half] = 0.5 * ay_mag * t1**2
    v_y_down[first_half] = ay_mag * t1

    t2 = t[~first_half] - half_t
    y_mid = 0.5 * ay_mag * half_t**2
    v_mid = ay_mag * half_t
    y_down[~first_half] = y_mid + v_mid * t2 - 0.5 * ay_mag * t2**2
    v_y_down[~first_half] = v_mid - ay_mag * t2

    h_descent = np.maximum(h_cruise - y_down, 0.0)
    u_horizontal = np.maximum(u_cruise - bx * t, 0.0)
    V_descent = np.sqrt(u_horizontal**2 + v_y_down**2)

    mach_descent = np.array([
        V / max(np.sqrt(GAMMA * R_GAS * atmosphere_drag(h)[1]), 1e-12)
        for V, h in zip(V_descent, h_descent)
    ])

    # Concatenate without repeating segment boundary points.
    x_total = np.concatenate([x_ascent, x_cruise[1:], x_descent[1:]])
    h_total = np.concatenate([h_ascent, h_cruise_arr[1:], h_descent[1:]])
    mach_total = np.concatenate([mach_ascent, mach_cruise_arr[1:], mach_descent[1:]])

    segment = np.concatenate([
        np.full(len(x_ascent), "ascent"),
        np.full(len(x_cruise) - 1, "cruise"),
        np.full(len(x_descent) - 1, "descent"),
    ])

    final_range = float(x_total[-1])

    return {
        "x_total_m": x_total,
        "h_total_m": h_total,
        "mach_total": mach_total,
        "segment": segment,
        "x_ascent_m": x_ascent,
        "h_ascent_m": h_ascent,
        "mach_ascent": mach_ascent,
        "x_cruise_m": x_cruise,
        "h_cruise_m_array": h_cruise_arr,
        "mach_cruise_array": mach_cruise_arr,
        "x_descent_m": x_descent,
        "h_descent_m": h_descent,
        "mach_descent": mach_descent,
        "V_ascent_m_s": V_ascent,
        "V_descent_m_s": V_descent,
        "x_climb_end_m": x_climb_end,
        "x_cruise_start_m": x_cruise_start,
        "x_cruise_end_m": x_cruise_end,
        "cruise_range_m": float(cruise_range),
        "descent_range_m": x_descent_range,
        "final_range_m": final_range,
        "requested_total_range_m": requested_total_range,
        "cruise_fits_requested_range": bool(cruise_fits),
        "h_cruise_m": h_cruise,
        "M_cruise": M_cruise,
        "u_cruise_m_s": u_cruise,
        **descent,
    }


def print_full_mission_summary(mission: dict) -> None:
    """Print the key values of the full takeoff-to-landing reconstruction."""
    print("\nFull takeoff-to-landing mission geometry")
    print("----------------------------------------")
    print(f"Requested total range       : {mission['requested_total_range_m'] / 1000.0:,.2f} km")
    print(f"Climb/acceleration range    : {mission['x_climb_end_m'] / 1000.0:,.2f} km")
    print(f"Cruise range                : {mission['cruise_range_m'] / 1000.0:,.2f} km")
    print(f"Descent range               : {mission['descent_range_m'] / 1000.0:,.2f} km")
    print(f"Final total range           : {mission['final_range_m'] / 1000.0:,.2f} km")
    print(f"Cruise fits requested range : {mission['cruise_fits_requested_range']}")
    print(f"Cruise Mach                 : {mission['M_cruise']:.2f}")
    print(f"Cruise altitude             : {mission['h_cruise_m'] / 1000.0:.2f} km")
    print(f"Cruise speed                : {mission['u_cruise_m_s']:.2f} m/s")
    print(f"Descent time                : {mission['t_descent_s']:.2f} s")
    print(f"Descent a_x                 : {mission['a_x_descent_m_s2']:.4f} m/s²")
    print(f"Descent |a_y|               : {mission['a_y_magnitude_m_s2']:.4f} m/s²")
    print(f"Descent total acceleration  : {mission['a_total_m_s2']:.4f} m/s²")
    print(f"Acceleration limit          : {mission['a_limit_m_s2']:.4f} m/s²")


def save_full_mission_csv(mission: dict, prefix: str) -> None:
    """Save full mission profile and descent summary as CSV files."""
    profile = pd.DataFrame({
        "range_m": mission["x_total_m"],
        "range_km": np.asarray(mission["x_total_m"]) / 1000.0,
        "altitude_m": mission["h_total_m"],
        "altitude_km": np.asarray(mission["h_total_m"]) / 1000.0,
        "mach": mission["mach_total"],
        "segment": mission["segment"],
    })
    profile_file = f"{prefix}_takeoff_to_landing_profile.csv"
    profile.to_csv(profile_file, index=False)

    summary_keys = [
        "requested_total_range_m", "x_climb_end_m", "cruise_range_m",
        "descent_range_m", "final_range_m", "cruise_fits_requested_range",
        "h_cruise_m", "M_cruise", "u_cruise_m_s", "t_descent_s",
        "a_x_descent_m_s2", "a_x_magnitude_m_s2", "a_y_magnitude_m_s2",
        "a_total_m_s2", "a_limit_m_s2",
    ]
    summary = pd.DataFrame([{
        key: mission[key]
        for key in summary_keys
    }])
    summary_file = f"{prefix}_descent_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"Saved takeoff-to-landing profile CSV: {profile_file}")
    print(f"Saved descent summary CSV: {summary_file}")


def plot_full_mission_profiles(mission: dict, best: ProfileResult, prefix: str = "optimized_flight_profile_multimaps") -> None:
    """
    Plot altitude vs range and altitude vs Mach from takeoff to landing.

    This version plots from the concatenated full-mission arrays and then masks
    by segment. That avoids dimension mismatches when a cruise/descent segment
    contains a different number of points than expected.
    """
    import matplotlib.pyplot as plt

    x_total_km = np.asarray(mission["x_total_m"], dtype=float) / 1000.0
    h_total_km = np.asarray(mission["h_total_m"], dtype=float) / 1000.0
    mach_total = np.asarray(mission["mach_total"], dtype=float)
    segment = np.asarray(mission["segment"])

    if not (len(x_total_km) == len(h_total_km) == len(mach_total) == len(segment)):
        raise ValueError(
            "Full-mission arrays have inconsistent lengths: "
            f"x={len(x_total_km)}, h={len(h_total_km)}, "
            f"Mach={len(mach_total)}, segment={len(segment)}"
        )

    segment_labels = [
        ("ascent", "Powered climb/acceleration"),
        ("cruise", "Cruise"),
        ("descent", "Unpowered descent/deceleration"),
    ]

    # ------------------------------------------------------------------
    # Altitude vs range.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for key, label in segment_labels:
        mask = segment == key
        if np.count_nonzero(mask) > 0:
            ax.plot(
                x_total_km[mask],
                h_total_km[mask],
                linewidth=2.0,
                label=label,
            )

    ax.axvline(float(mission["x_cruise_start_m"]) / 1000.0, linestyle="--", linewidth=1.1, label="Start cruise")
    ax.axvline(float(mission["x_cruise_end_m"]) / 1000.0, linestyle="--", linewidth=1.1, label="End cruise")
    ax.set_xlabel("Range [km]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Altitude vs range from takeoff to landing")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if SAVE_FULL_MISSION_PLOTS:
        fig.savefig(f"{prefix}_altitude_vs_range_takeoff_to_landing.png", dpi=300, bbox_inches="tight")
        print(f"Saved altitude-vs-range PNG: {prefix}_altitude_vs_range_takeoff_to_landing.png")

    # ------------------------------------------------------------------
    # Altitude vs Mach.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, label in segment_labels:
        mask = segment == key
        if np.count_nonzero(mask) > 0:
            ax.plot(
                mach_total[mask],
                h_total_km[mask],
                linewidth=2.0,
                marker="o" if key == "ascent" else None,
                markersize=3 if key == "ascent" else None,
                label=label,
            )

    ax.axvline(float(best.transition_mach), linestyle="--", linewidth=1.1, label=f"M={best.transition_mach:.2f} switch")
    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Altitude vs Mach number from takeoff to landing")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if SAVE_FULL_MISSION_PLOTS:
        fig.savefig(f"{prefix}_altitude_vs_mach_takeoff_to_landing.png", dpi=300, bbox_inches="tight")
        print(f"Saved altitude-vs-Mach PNG: {prefix}_altitude_vs_mach_takeoff_to_landing.png")


def run_exact_ramjet_diagnostic_check(best: ProfileResult, prefix: str = "optimized_flight_profile_multimaps") -> pd.DataFrame:
    """
    Run a tiny set of exact ramjet calls at critical points only.

    This is intentionally separate from map generation. The optimization still
    uses the fast/calibrated ramjet map, while these few exact checks identify
    whether the transition and Mach-5 regions are genuinely weak or just being
    underpredicted by the surrogate/interpolation.
    """
    if not RUN_EXACT_RAMJET_DIAGNOSTIC_CHECK:
        return pd.DataFrame()

    rows = []
    mdot = float(best.ramjet_design_mdot_kg_s)

    print("\nExact ramjet diagnostic check")
    print("-----------------------------")
    print(f"Using selected ramjet design mdot: {mdot:.2f} kg/s")
    print("Only a few exact points are evaluated; this should not rebuild the full exact map.")

    for M, h in EXACT_RAMJET_DIAGNOSTIC_POINTS:
        try:
            T_exact = ramjet_thrust_point_kN(M, h, mdot, suppress_output=True)
        except Exception as exc:
            T_exact = np.nan
            print(f"exact ramjet M={M:.2f}, h={h/1000:.1f} km FAILED: {type(exc).__name__}: {exc}")

        aero = drag_and_required_thrust_kN(M, h, ACCEL_G_TARGET)
        required = float(aero["thrust_req_kN"]) * (1.0 + THRUST_MARGIN)
        margin = T_exact - required if np.isfinite(T_exact) else np.nan

        rows.append({
            "mach": float(M),
            "altitude_m": float(h),
            "ramjet_design_mdot_kg_s": mdot,
            "exact_ramjet_T_available_kN": T_exact,
            "T_required_kN": required,
            "exact_thrust_margin_kN": margin,
            "CL": aero["CL"],
            "q_Pa": aero["q_Pa"],
        })

        print(
            f"M={M:4.2f}  h={h/1000:5.1f} km  "
            f"T_exact={T_exact:10.2f} kN  T_req={required:10.2f} kN  margin={margin:10.2f} kN"
        )

    df = pd.DataFrame(rows)
    out = f"{prefix}_exact_ramjet_diagnostic_points.csv"
    df.to_csv(out, index=False)
    print(f"Saved exact ramjet diagnostic points: {out}")
    return df

# =============================================================================
# 10. MAIN
# =============================================================================


def main():
    maps = prepare_multidesign_maps()
    print_map_quality(maps)

    best_discrete, summary_discrete, _ = optimize_all_discrete_map_pairs(maps)
    best = best_discrete
    summary = summary_discrete.copy()
    summary.insert(0, "source", "complete_map_pair")

    if ENABLE_INTERPOLATED_DESIGN_PAIR_SEARCH:
        best_interp_grid, summary_interp_grid, _ = optimize_all_interpolated_design_pairs(maps, center_result=best_discrete)
        summary = pd.concat([summary, summary_interp_grid], ignore_index=True)
        if best_interp_grid.feasible and (not best.feasible or best_interp_grid.design_score < best.design_score):
            print("\nUsing best interpolated design-pair search result.")
            best = best_interp_grid
        elif (not best.feasible) and (best_interp_grid.max_deficit_kN < best.max_deficit_kN):
            print("\nUsing least-infeasible interpolated design-pair search result.")
            best = best_interp_grid
        else:
            print("\nKeeping best complete-map-pair result after interpolated design-pair search.")

    if ENABLE_INTERPOLATED_DESIGN_REFINEMENT:
        refined = refine_between_complete_maps(maps, best)
        if refined.feasible and (not best.feasible or refined.design_score <= best.design_score * 1.02):
            print("\nUsing interpolated-design refined result.")
            best = refined
        elif (not best.feasible) and (refined.max_deficit_kN < best.max_deficit_kN):
            print("\nUsing least-infeasible interpolated-design refined result.")
            best = refined
        else:
            print("\nKeeping previous best result.")

    save_and_show_outputs(maps, best, summary)
    # Exact ramjet diagnostic check disabled by default because it is slow.
    # Uncomment the next line only if you deliberately want exact ramjet spot checks.
    # run_exact_ramjet_diagnostic_check(best)


if __name__ == "__main__":
    main()
