"""
flight_profile_optimizer_maps.py

Fast flight-profile optimizer using precomputed thrust lookup maps.

Main idea
---------
Do NOT call pyCycle / ramjet ODEs inside the optimizer loop.

Instead:
  1) Build turbojet thrust map once over Mach-altitude grid.
  2) Build ramjet thrust map once over Mach-altitude grid.
  3) Save maps as .npz files.
  4) During optimization, interpolate thrust from maps.

Mission:
  takeoff -> climb -> switch turbojet to ramjet at Mach 3 -> cruise at Mach 5, 30 km

Sizing variables solved after profile evaluation:
  - turbojet design thrust scale relative to uploaded turbojet baseline
  - ramjet design mass-flow scale relative to uploaded ramjet baseline

Outputs:
  turbojet_thrust_map.npz
  ramjet_thrust_map.npz

The optimizer prints final sizing values and shows plots interactively.
It does not save CSV, TXT, or PNG outputs.

Expected use:
  Put this file in the same folder as your uploaded files:
      Pasted text (2).txt        <-- ramjet code
      Geplakte code (3).py       <-- turbojet code

  Then run:
      python flight_profile_optimizer_maps.py

First run can still take time because it builds maps.
Second run is fast because it loads:
      turbojet_thrust_map.npz
      ramjet_thrust_map.npz

Dependencies:
  numpy
  pandas
  scipy
  matplotlib
  openmdao / pyCycle dependencies for your turbojet file
  your ramjet code dependencies
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import io
import os
import traceback
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

# Uploaded filenames. Edit if you rename the files.
RAMJET_FILE = "ramjet_revised.py"
TURBOJET_FILE = "turbojet_pycycle_wrapper.py"

# Lookup-map filenames.
TURBO_MAP_FILE = "turbojet_thrust_map.npz"
RAMJET_MAP_FILE = "ramjet_thrust_map.npz"

# If False, existing .npz maps are loaded.
# If True, maps are regenerated even when files already exist.
FORCE_REBUILD_MAPS = False

# Debug settings.
# Keep RAMJET_DEBUG_ERRORS=True until the ramjet map works at least once.
RAMJET_DEBUG_ERRORS = True
MAX_RAMJET_ERROR_PRINTS = 8

# -------------------------------------------------------------------------
# SPEED OPTION FOR RAMJET MAP
# -------------------------------------------------------------------------
# Exact ramjet calls are very slow because each point runs CEA + multiple ODE
# integrations. Keep this True for optimization. Then use the optional exact
# engine check at the end to verify a few points on the final profile.
USE_FAST_RAMJET_SURROGATE = True

# Surrogate is normalized to the ramjet's own design point.
RAMJET_SURROGATE_DESIGN_MACH = 4.0
RAMJET_SURROGATE_DESIGN_ALT_M = 25_000.0
RAMJET_SURROGATE_DESIGN_THRUST_KN_PER_ENGINE = 268.478

# Shape controls for the fast ramjet estimate.
# T ~ mdot_capture * mach_shape * altitude_shape
# This is intentionally conservative away from the design point.
RAMJET_SURROGATE_MACH_SIGMA = 1.10
RAMJET_SURROGATE_ALT_SIGMA_M = 13_000.0
RAMJET_SURROGATE_MIN_FACTOR = 0.15

# Coarse initial grids.
# Increase later once everything works.
#
# Turbojet map covers takeoff to transition.
# Ramjet map is intentionally NOT built at sea level: a ramjet at M=3-5 near sea
# level is usually numerically/physically ugly and is not part of this mission.
MACH_GRID_TURBO = np.linspace(0.30, 3.00, 8)
MACH_GRID_RAMJET = np.linspace(3.00, 5.00, 8)

ALT_GRID_TURBO_M = np.linspace(0.0, 30_000.0, 10)

# Includes the ramjet design point: 25 km, Mach 4.
ALT_GRID_RAMJET_M = np.array([
    18_000.0,
    20_000.0,
    22_500.0,
    25_000.0,
    27_500.0,
    30_000.0,
])

# Mission definition.
M_TAKEOFF = 0.30
M_SWITCH = 3.00
M_CRUISE = 5.00
H_CRUISE_M = 30_000.0

# Required acceleration.
ACCEL_G_TARGET = 0.15

# Engine counts.
N_TURBOJETS = 2
N_RAMJETS = 2

# Ramjet fuel setting for your ramjet model.
RAMJET_PHI = 0.70

# Baseline sizes from your uploaded models.
BASE_TURBO_DESIGN_THRUST_LBF = 150_000.0  # per engine, from your turbojet script
BASE_RAMJET_DESIGN_MDOT = 200.0           # kg/s, from your ramjet script

# Flight-profile control Machs.
# Optimizer changes the free altitude knots.
MACH_KNOTS = np.array([0.30, 0.80, 1.20, 2.00, 3.00, 4.00, 5.00])

# Fixed altitude knots:
# index 0: takeoff
# last index: cruise
FIXED_ALT_KNOTS = {
    0: 0.0,
    len(MACH_KNOTS) - 1: H_CRUISE_M,
}

# Number of trajectory points checked during optimization.
# Keep this small while developing. Increase to 50-80 for final.
N_EVAL = 30

# Envelope constraints.
# Set to None to disable.
CL_MAX = 1.50
Q_MAX_PA = 120_000.0

# Constraint smoothing / penalties.
THRUST_MARGIN = 0.00  # 0.05 gives 5% thrust margin.
BIG_PENALTY = 1e8

# Objective weights.
# Objective roughly equals turbo scale + ramjet scale + penalties.
W_TURBO = 1.0
W_RAMJET = 1.0

# Interpolator behavior outside map bounds.
# I recommend keeping this False. The optimizer should remain inside grid bounds.
ALLOW_EXTRAPOLATION = False


# =============================================================================
# 1. DRAG MODEL WRAPPED AS FUNCTIONS
# =============================================================================
# Updated from the standalone drag model, but keeping the old optimizer values
# for W_TOG, S_PLAN, and S_WET as requested.

G = 9.81
R_GAS = 287.05
GAMMA = 1.4

# Keep these from the old optimizer code.
W_TOG = 90_000            # Aircraft gross weight / TOGW [kg]
S_PLAN = 350              # Planform wing area [m^2]
S_WET = 1050              # Wetted area [m^2]

# Updated drag-model geometry/constants.
MAC = 21.0                # Mean Aerodynamic Chord [m]
L_REF = 35.0              # Characteristic length for high-speed Reynolds calculations [m]
IF = 1.05                 # Interference factor [+5%]
t_over_c = 0.05           # Relative thickness
sweep_deg = 35.0          # Wing sweep [deg]
sweep_rad = np.radians(sweep_deg)
AR = 7.0                  # Aspect ratio reference


def atmosphere_drag(alt_m: float) -> tuple[float, float]:
    """
    Updated ISA-like atmosphere from the standalone drag script.

    Returns:
        rho [kg/m^3], T [K]

    Note:
        The uploaded drag model uses a simple fallback above 40 km instead of
        raising an error. The mission target is 30 km, so this branch should not
        normally affect the optimized profile.
    """
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
    """Updated Reynolds-number calculation using Sutherland's law."""
    mu_0 = 1.7894e-5
    T_0 = 273.15
    S_suth = 110.4
    mu = mu_0 * (temp / T_0) ** 1.5 * (T_0 + S_suth) / (temp + S_suth)
    return (rho * v * chord) / mu


def drag_and_required_thrust_kN(
    mach: float,
    altitude_m: float,
    accel_g: float = ACCEL_G_TARGET,
) -> dict:
    """
    Updated drag model wrapped for the optimizer.

    Required thrust:
        T_req = Drag + W_TOG * accel_g * g

    Returns dictionary with:
        thrust_req_kN, drag_kN, q_Pa, CL, CD, alpha_deg, etc.
    """
    rho, T = atmosphere_drag(float(altitude_m))
    a = np.sqrt(GAMMA * R_GAS * T)
    M = float(mach)
    V = M * a
    q = 0.5 * rho * V**2
    q_safe = max(q, 1e-9)

    cl_needed = (W_TOG * G) / (q_safe * S_PLAN)

    # ---------------------------------------------------------------------
    # Aerodynamic regime switching logic from the updated drag model
    # ---------------------------------------------------------------------
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
            cd_wave = amplitude * np.sin(
                (M - M_crit) / (M_peak - M_crit) * (np.pi / 2)
            ) ** 2
        else:
            amplitude_peak = 20 * (t_over_c**2.5) * np.cos(sweep_rad) ** 2
            cd_wave = amplitude_peak / np.sqrt(max(0.1, M**2 - 1.0))

        e_oswald = 0.85 - 0.02 * M
        cd_induced = (cl_needed**2) / (np.pi * AR * e_oswald)
        alpha_deg = np.degrees(cl_needed / (2 * np.pi * AR / (AR + 2)))

    else:
        regime = "supersonic" if M < 3.0 else "hypersonic"

        # Pure supersonic properties.
        Re_dyn_super = max(reynolds_number(rho, V, T, MAC), 10.0)
        cf_super = 0.455 / (np.log10(Re_dyn_super) ** 2.58)
        cd_f_super = cf_super * IF * (S_WET / S_PLAN)

        alpha_rad_super = np.sqrt(np.abs(cl_needed**0.75) / 2)
        cd_wave_super = 2 * np.sin(alpha_rad_super) ** 3

        # Pure hypersonic properties.
        Re_dyn_hyper = max(reynolds_number(rho, V, T, L_REF), 10.0)
        cf_hyper = (0.074 / (Re_dyn_hyper ** 0.2)) * ((1 / (1 + 0.15 * M**2)) ** 0.58)
        cd_f_hyper = cf_hyper * 2.0

        cl_alpha_hyper = 4.0 / np.sqrt(max(0.01, M**2 - 1.0))
        alpha_rad_hyper = cl_needed / cl_alpha_hyper
        cd_wave_hyper = cl_needed * alpha_rad_hyper

        # Logistic blend centered at Mach 3.
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
# 2. DYNAMIC IMPORT HELPERS
# =============================================================================

def import_from_path(module_name: str, path: str | Path):
    """
    Import a Python source file from path.
    Works even if the ramjet source has .txt extension.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path!s}. Put this optimizer in the same folder "
            f"as your uploaded files or edit RAMJET_FILE / TURBOJET_FILE."
        )

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


# =============================================================================
# 3. SINGLE-POINT ENGINE EVALUATORS FOR MAP BUILDING
# =============================================================================

def turbojet_thrust_point_kN(mach: float, altitude_m: float) -> float:
    """
    Calls your uploaded turbojet wrapper once.

    Returns:
        total turbojet thrust [kN] for N_TURBOJETS engines
        at BASE_TURBO_DESIGN_THRUST_LBF per engine.
    """
    tj = get_turbojet_module()

    # Keep module at baseline design thrust.
    if hasattr(tj, "DESIGN_THRUST_LBF"):
        tj.DESIGN_THRUST_LBF = BASE_TURBO_DESIGN_THRUST_LBF

    if hasattr(tj, "BASE_DESIGN_THRUST_LBF") and hasattr(tj, "DESIGN_THRUST_LBF"):
        tj.SCALE = tj.DESIGN_THRUST_LBF / tj.BASE_DESIGN_THRUST_LBF

    # Reset model once after changing scale.
    if hasattr(tj, "reset_turbo_model") and not getattr(tj, "_MAP_BASELINE_READY", False):
        tj.reset_turbo_model()
        tj._MAP_BASELINE_READY = True

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


_RAMJET_ERROR_COUNT = 0


def ramjet_design_self_test() -> None:
    """
    Runs or reports the ramjet design-point calibration.

    In fast-surrogate mode, we do not call the exact ramjet model here because
    that defeats the point of speeding up map generation.
    """
    print("\nRamjet design-point self-test")
    print("-----------------------------")

    if USE_FAST_RAMJET_SURROGATE:
        T_total = RAMJET_SURROGATE_DESIGN_THRUST_KN_PER_ENGINE * N_RAMJETS
        print("USE_FAST_RAMJET_SURROGATE = True")
        print(
            f"Using calibrated design thrust: "
            f"{RAMJET_SURROGATE_DESIGN_THRUST_KN_PER_ENGINE:.3f} kN per engine"
        )
        print(f"Number of ramjets: {N_RAMJETS}")
        print(f"Total design thrust used by map: {T_total:.3f} kN\n")
        return

    print("Testing h=25.0 km, M=4.00, design mdot=200 kg/s")

    T = ramjet_thrust_point_kN(
        mach=4.0,
        altitude_m=25_000.0,
        suppress_output=False,
        raise_on_error=True,
    )

    print(f"Ramjet design-point thrust = {T:.3f} kN\n")



def ramjet_fast_surrogate_thrust_kN(mach: float, altitude_m: float) -> float:
    """
    Fast ramjet thrust estimate for map generation.

    Why this exists:
        The exact ramjet model is far too slow to call at every grid point
        because it runs CEA and several ODE integrations. This surrogate is
        used only to build a quick optimization map.

    Model:
        T(M,h) = T_design * N_RAMJETS
                 * [rho(h) V(M,h)] / [rho_design V_design]
                 * Mach bell factor
                 * altitude bell factor

    Since the ramjet file uses a fixed capture area, captured mass flow scales
    roughly with rho * V. The bell factors keep the estimate conservative away
    from the design condition.

    Final results should be checked with exact_engine_check_at_profile().
    """
    M = float(mach)
    h = float(altitude_m)

    rho, T = atmosphere_drag(h)
    a = np.sqrt(GAMMA * R_GAS * T)
    V = M * a

    rho_d, T_d = atmosphere_drag(RAMJET_SURROGATE_DESIGN_ALT_M)
    a_d = np.sqrt(GAMMA * R_GAS * T_d)
    V_d = RAMJET_SURROGATE_DESIGN_MACH * a_d

    mdot_ratio = (rho * V) / max(rho_d * V_d, 1e-12)

    mach_factor = np.exp(-0.5 * ((M - RAMJET_SURROGATE_DESIGN_MACH) / RAMJET_SURROGATE_MACH_SIGMA) ** 2)
    alt_factor = np.exp(-0.5 * ((h - RAMJET_SURROGATE_DESIGN_ALT_M) / RAMJET_SURROGATE_ALT_SIGMA_M) ** 2)

    shape = max(RAMJET_SURROGATE_MIN_FACTOR, mach_factor * alt_factor)

    thrust_total_kN = (
        RAMJET_SURROGATE_DESIGN_THRUST_KN_PER_ENGINE
        * N_RAMJETS
        * mdot_ratio
        * shape
    )

    if not np.isfinite(thrust_total_kN):
        return np.nan

    return max(float(thrust_total_kN), 0.0)


def ramjet_thrust_point_kN(
    mach: float,
    altitude_m: float,
    *,
    suppress_output: bool = True,
    raise_on_error: bool = False,
) -> float:
    """
    Calls your uploaded ramjet model once.

    Returns:
        ramjet net thrust [kN] at BASE_RAMJET_DESIGN_MDOT.

    Important:
        Earlier versions silently caught exceptions and returned NaN. That made
        "all values failed" impossible to diagnose. This version can print or
        raise the real exception.
    """
    global _RAMJET_ERROR_COUNT

    rj = get_ramjet_module()
    eng = rj.Ramjet()

    def _run_case():
        inp = eng.inlet_properties(
            h=float(altitude_m),
            Ma=float(mach),
            m_air=BASE_RAMJET_DESIGN_MDOT,
        )
        iso = eng.isolator_properties(inp)
        sec2 = eng.combustor_properties2(iso)
        sec3 = eng.combustor_properties3(sec2, phi=RAMJET_PHI)
        sec4 = eng.combustor_properties4(sec3)

        # In your ramjet code, combustor thermal choke is NOT an abort condition:
        # the nozzle routine treats the combustor exit as the natural throat.
        sec5 = eng.nozzle_properties(sec4, inp)
        perf = eng.performance(inp, sec5, sec3)

        if perf.get("thermal_choke", False):
            # This should only happen if nozzle_properties itself marks failure.
            raise RuntimeError("Ramjet performance returned thermal_choke=True.")

        if "Fin" not in perf:
            raise KeyError(f"Ramjet performance output has no 'Fin'. Keys: {list(perf.keys())}")

        thrust_kN = float(perf["Fin"]) / 1000.0

        if not np.isfinite(thrust_kN):
            raise FloatingPointError(f"Ramjet Fin is not finite: {perf['Fin']}")

        return thrust_kN

    try:
        if suppress_output:
            # Use StringIO instead of os.devnull.
            # On Windows, os.devnull opened with the default cp1252 encoding can
            # crash when the ramjet file prints Unicode characters like ─ or ṁ.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                thrust_kN = _run_case()
        else:
            thrust_kN = _run_case()

        # Do not convert negative thrust to NaN. A negative value is useful
        # information: it tells the optimizer this point cannot produce useful
        # ramjet thrust at the baseline size.
        return float(thrust_kN)

    except Exception as exc:
        _RAMJET_ERROR_COUNT += 1

        if raise_on_error:
            raise

        if RAMJET_DEBUG_ERRORS and _RAMJET_ERROR_COUNT <= MAX_RAMJET_ERROR_PRINTS:
            print()
            print("RAMJET EVALUATION FAILED")
            print("------------------------")
            print(f"Mach      = {float(mach):.3f}")
            print(f"Altitude  = {float(altitude_m):.1f} m")
            print(f"Error     = {type(exc).__name__}: {exc}")
            print("Traceback:")
            traceback.print_exc()
            print()

        return np.nan


# =============================================================================
# 4. MAP BUILDING / LOADING
# =============================================================================

def fill_missing_table_values(table: np.ndarray, label: str) -> np.ndarray:
    """
    Fills NaNs in a 2D thrust table using nearest valid value.

    Better approach for final work:
      inspect failed points and improve engine convergence there.

    This is included so a few failed pyCycle/ramjet points do not kill the
    whole optimizer.
    """
    arr = np.array(table, dtype=float, copy=True)

    if np.all(~np.isfinite(arr)):
        raise RuntimeError(
            f"All values failed in {label} thrust table. "
            f"For the ramjet, check the design-point self-test printed above. "
            f"The previous version hid the actual exception by returning NaN."
        )

    valid = np.isfinite(arr)

    if np.all(valid):
        return arr

    print(f"\nWARNING: {label} table has {np.size(arr) - np.count_nonzero(valid)} failed points.")
    print("Filling failed points with nearest valid neighbor. Verify these later.\n")

    valid_indices = np.argwhere(valid)
    invalid_indices = np.argwhere(~valid)

    for idx in invalid_indices:
        distances = np.sum((valid_indices - idx) ** 2, axis=1)
        nearest = valid_indices[np.argmin(distances)]
        arr[tuple(idx)] = arr[tuple(nearest)]

    return arr


def build_turbojet_map(
    mach_grid: np.ndarray = MACH_GRID_TURBO,
    alt_grid_m: np.ndarray = ALT_GRID_TURBO_M,
    save_file: str | Path = TURBO_MAP_FILE,
) -> dict:
    """
    Builds turbojet map:
        table shape = (n_altitudes, n_machs)
    """
    mach_grid = np.asarray(mach_grid, dtype=float)
    alt_grid_m = np.asarray(alt_grid_m, dtype=float)

    table = np.full((len(alt_grid_m), len(mach_grid)), np.nan)

    print("\nBuilding turbojet thrust map")
    print("----------------------------")
    print(f"Grid size: {len(alt_grid_m)} altitudes x {len(mach_grid)} Machs = {table.size} points")
    print(f"Baseline design thrust: {BASE_TURBO_DESIGN_THRUST_LBF:,.0f} lbf per engine")
    print(f"Number of engines: {N_TURBOJETS}")

    for i, h in enumerate(alt_grid_m):
        for j, M in enumerate(mach_grid):
            T = turbojet_thrust_point_kN(M, h)
            table[i, j] = T
            print(f"turbo  h={h/1000:5.1f} km  M={M:4.2f}  T={T:10.2f} kN")

    table = fill_missing_table_values(table, "turbojet")

    np.savez(
        save_file,
        mach_grid=mach_grid,
        alt_grid_m=alt_grid_m,
        thrust_kN=table,
        baseline_design_thrust_lbf=BASE_TURBO_DESIGN_THRUST_LBF,
        n_engines=N_TURBOJETS,
    )

    print(f"Saved turbojet map: {save_file}")
    return {
        "mach_grid": mach_grid,
        "alt_grid_m": alt_grid_m,
        "thrust_kN": table,
    }


def build_ramjet_map(
    mach_grid: np.ndarray = MACH_GRID_RAMJET,
    alt_grid_m: np.ndarray = ALT_GRID_RAMJET_M,
    save_file: str | Path = RAMJET_MAP_FILE,
) -> dict:
    """
    Builds ramjet map:
        table shape = (n_altitudes, n_machs)
    """
    mach_grid = np.asarray(mach_grid, dtype=float)
    alt_grid_m = np.asarray(alt_grid_m, dtype=float)

    table = np.full((len(alt_grid_m), len(mach_grid)), np.nan)

    print("\nBuilding ramjet thrust map")
    print("--------------------------")
    print(f"Grid size: {len(alt_grid_m)} altitudes x {len(mach_grid)} Machs = {table.size} points")
    print(f"Baseline design mdot: {BASE_RAMJET_DESIGN_MDOT:.2f} kg/s")
    print(f"Phi: {RAMJET_PHI:.3f}")
    print(f"Fast surrogate mode: {USE_FAST_RAMJET_SURROGATE}")

    for i, h in enumerate(alt_grid_m):
        for j, M in enumerate(mach_grid):
            if USE_FAST_RAMJET_SURROGATE:
                T = ramjet_fast_surrogate_thrust_kN(M, h)
            else:
                T = ramjet_thrust_point_kN(M, h)

            table[i, j] = T
            tag = "surrogate" if USE_FAST_RAMJET_SURROGATE else "exact"
            print(f"ramjet-{tag:9s} h={h/1000:5.1f} km  M={M:4.2f}  T={T:10.2f} kN")

    table = fill_missing_table_values(table, "ramjet")

    np.savez(
        save_file,
        mach_grid=mach_grid,
        alt_grid_m=alt_grid_m,
        thrust_kN=table,
        baseline_design_mdot=BASE_RAMJET_DESIGN_MDOT,
        phi=RAMJET_PHI,
        n_engines=N_RAMJETS,
        use_fast_surrogate=USE_FAST_RAMJET_SURROGATE,
        surrogate_design_thrust_kN_per_engine=RAMJET_SURROGATE_DESIGN_THRUST_KN_PER_ENGINE,
        surrogate_design_mach=RAMJET_SURROGATE_DESIGN_MACH,
        surrogate_design_alt_m=RAMJET_SURROGATE_DESIGN_ALT_M,
    )

    print(f"Saved ramjet map: {save_file}")
    return {
        "mach_grid": mach_grid,
        "alt_grid_m": alt_grid_m,
        "thrust_kN": table,
    }


def load_map(file: str | Path) -> dict:
    data = np.load(file)
    return {
        "mach_grid": data["mach_grid"],
        "alt_grid_m": data["alt_grid_m"],
        "thrust_kN": data["thrust_kN"],
    }


def get_or_build_maps(force_rebuild: bool = FORCE_REBUILD_MAPS) -> tuple[dict, dict]:
    turbo_exists = Path(TURBO_MAP_FILE).exists()
    ramjet_exists = Path(RAMJET_MAP_FILE).exists()

    if force_rebuild or not turbo_exists:
        turbo_map = build_turbojet_map()
    else:
        turbo_map = load_map(TURBO_MAP_FILE)
        print(f"Loaded existing turbojet map: {TURBO_MAP_FILE}")

    if force_rebuild or not ramjet_exists:
        # Fail early with a useful error if the ramjet cannot run at its own
        # nominal design point. This is much clearer than building an all-NaN map.
        ramjet_design_self_test()
        ramjet_map = build_ramjet_map()
    else:
        ramjet_map = load_map(RAMJET_MAP_FILE)
        print(f"Loaded existing ramjet map: {RAMJET_MAP_FILE}")

    return turbo_map, ramjet_map


# =============================================================================
# 5. MAP INTERPOLATORS
# =============================================================================

@dataclass
class ThrustMaps:
    turbo_map: dict
    ramjet_map: dict
    turbo_interp: RegularGridInterpolator
    ramjet_interp: RegularGridInterpolator


def make_interpolator(map_data: dict, label: str) -> RegularGridInterpolator:
    """
    Creates interpolator with input order:
        [altitude_m, mach]
    """
    alt_grid = map_data["alt_grid_m"]
    mach_grid = map_data["mach_grid"]
    table = map_data["thrust_kN"]

    if table.shape != (len(alt_grid), len(mach_grid)):
        raise ValueError(
            f"{label} map table shape mismatch. "
            f"Expected {(len(alt_grid), len(mach_grid))}, got {table.shape}."
        )

    fill_value = None if ALLOW_EXTRAPOLATION else np.nan

    return RegularGridInterpolator(
        points=(alt_grid, mach_grid),
        values=table,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )


def prepare_thrust_maps() -> ThrustMaps:
    turbo_map, ramjet_map = get_or_build_maps()
    turbo_interp = make_interpolator(turbo_map, "turbojet")
    ramjet_interp = make_interpolator(ramjet_map, "ramjet")

    return ThrustMaps(
        turbo_map=turbo_map,
        ramjet_map=ramjet_map,
        turbo_interp=turbo_interp,
        ramjet_interp=ramjet_interp,
    )


def thrust_from_maps_kN(maps: ThrustMaps, mach: float, altitude_m: float) -> tuple[str, float]:
    """
    Returns:
        mode, baseline available thrust [kN]
    """
    if mach < M_SWITCH:
        mode = "turbojet"
        T = maps.turbo_interp([[altitude_m, mach]])[0]
    else:
        mode = "ramjet"
        T = maps.ramjet_interp([[altitude_m, mach]])[0]

    T = float(T)
    if not np.isfinite(T):
        return mode, np.nan

    return mode, max(T, 0.0)


# =============================================================================
# 6. FLIGHT PROFILE PARAMETERIZATION
# =============================================================================

def unpack_altitude_knots(x: np.ndarray) -> np.ndarray:
    """
    x contains altitudes for non-fixed knots.
    Fixed knots are takeoff and cruise altitude.
    """
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
    """
    Returns:
        mach_grid, altitude_grid, mach_knots, altitude_knots
    """
    h_knots = unpack_altitude_knots(x)

    # PCHIP avoids large cubic overshoots and preserves monotonicity better.
    interp = PchipInterpolator(MACH_KNOTS, h_knots, extrapolate=False)

    mach_grid = np.linspace(M_TAKEOFF, M_CRUISE, n_eval)
    altitude_grid = np.asarray(interp(mach_grid), dtype=float)

    return mach_grid, altitude_grid, MACH_KNOTS, h_knots


def altitude_bounds_for_knots() -> list[tuple[float, float]]:
    """
    Bounds for free altitude knots.
    """
    free_indices = [i for i in range(len(MACH_KNOTS)) if i not in FIXED_ALT_KNOTS]
    bounds = []

    for i in free_indices:
        M = MACH_KNOTS[i]

        if M < 1.0:
            bounds.append((0.0, 10_000.0))
        elif M < 2.0:
            bounds.append((1_000.0, 18_000.0))
        elif M < M_SWITCH:
            bounds.append((5_000.0, 26_000.0))
        elif M < M_CRUISE:
            bounds.append((12_000.0, H_CRUISE_M))
        else:
            bounds.append((H_CRUISE_M, H_CRUISE_M))

    return bounds


# =============================================================================
# 7. EVALUATION AND OBJECTIVE
# =============================================================================

@dataclass
class SizingResult:
    objective: float
    turbo_design_lbf_per_engine: float
    ramjet_design_mdot_kg_s: float
    max_turbo_scale: float
    max_ramjet_scale: float
    penalty: float
    table: pd.DataFrame


def evaluate_candidate(x: np.ndarray, maps: ThrustMaps) -> SizingResult:
    mach_grid, h_grid, _, h_knots = make_profile(x)

    rows = []
    penalty = 0.0

    # Smooth monotonic-climb penalty.
    dh_knots = np.diff(h_knots)
    penalty += 1e5 * np.sum(np.minimum(dh_knots, 0.0) ** 2) / 1e8

    # Keep evaluated trajectory inside altitude bounds.
    penalty += 1e5 * np.sum(np.maximum(-h_grid, 0.0) ** 2) / 1e8
    penalty += 1e5 * np.sum(np.maximum(h_grid - H_CRUISE_M, 0.0) ** 2) / 1e8

    # Optional extra preference:
    # Encourage reaching high altitude before Mach 5, but do not force too aggressively.
    # penalty += 1e-4 * np.sum(np.maximum(0.0, 0.5 * H_CRUISE_M * (mach_grid / M_CRUISE) - h_grid) ** 2) / 1e6

    turbo_scales = []
    ramjet_scales = []

    for M, h in zip(mach_grid, h_grid):
        aero = drag_and_required_thrust_kN(M, h, accel_g=ACCEL_G_TARGET)
        required_kN = aero["thrust_req_kN"] * (1.0 + THRUST_MARGIN)

        # Envelope penalties.
        if CL_MAX is not None:
            penalty += 1e4 * max(0.0, aero["CL"] - CL_MAX) ** 2

        if Q_MAX_PA is not None:
            penalty += 1e-4 * max(0.0, aero["q_Pa"] - Q_MAX_PA) ** 2

        mode, baseline_T_kN = thrust_from_maps_kN(maps, M, h)

        if not np.isfinite(baseline_T_kN) or baseline_T_kN <= 1e-6:
            # Usually means outside thrust-map range or failed map point.
            scale = 1e6
            penalty += BIG_PENALTY
        else:
            scale = required_kN / baseline_T_kN

        if mode == "turbojet":
            turbo_scales.append(scale)
            turbo_scale = scale
            ramjet_scale = np.nan
        else:
            ramjet_scales.append(scale)
            turbo_scale = np.nan
            ramjet_scale = scale

        rows.append({
            **aero,
            "mode": mode,
            "baseline_T_available_kN": baseline_T_kN,
            "required_scale": scale,
            "turbo_scale": turbo_scale,
            "ramjet_scale": ramjet_scale,
        })

    max_turbo_scale = max(turbo_scales) if turbo_scales else 0.0
    max_ramjet_scale = max(ramjet_scales) if ramjet_scales else 0.0

    turbo_design_lbf = BASE_TURBO_DESIGN_THRUST_LBF * max_turbo_scale
    ramjet_design_mdot = BASE_RAMJET_DESIGN_MDOT * max_ramjet_scale

    objective_value = (
        W_TURBO * max_turbo_scale
        + W_RAMJET * max_ramjet_scale
        + penalty
    )

    df = pd.DataFrame(rows)
    df["sized_T_available_kN"] = np.where(
        df["mode"].eq("turbojet"),
        df["baseline_T_available_kN"] * max_turbo_scale,
        df["baseline_T_available_kN"] * max_ramjet_scale,
    )
    df["thrust_margin_kN"] = df["sized_T_available_kN"] - df["thrust_req_kN"]

    return SizingResult(
        objective=objective_value,
        turbo_design_lbf_per_engine=turbo_design_lbf,
        ramjet_design_mdot_kg_s=ramjet_design_mdot,
        max_turbo_scale=max_turbo_scale,
        max_ramjet_scale=max_ramjet_scale,
        penalty=penalty,
        table=df,
    )


def objective(x: np.ndarray, maps: ThrustMaps) -> float:
    return evaluate_candidate(x, maps).objective


# =============================================================================
# 8. OPTIMIZER
# =============================================================================

def optimize_profile(maps: ThrustMaps) -> tuple[np.ndarray, SizingResult]:
    """
    Global search followed by local polish.

    Since thrust calls are interpolated, this should be much faster than the
    first version.
    """
    bounds = altitude_bounds_for_knots()

    print("\nOptimizing profile using thrust maps")
    print("------------------------------------")
    print(f"Free altitude knots: {len(bounds)}")
    print(f"Evaluation points per candidate: {N_EVAL}")

    result_de = differential_evolution(
        func=lambda z: objective(z, maps),
        bounds=bounds,
        strategy="best1bin",
        maxiter=120,
        popsize=12,
        tol=0.005,
        polish=False,
        seed=7,
        workers=1,
        updating="immediate",
    )

    result_local = minimize(
        fun=lambda z: objective(z, maps),
        x0=result_de.x,
        bounds=bounds,
        method="Nelder-Mead",
        options={
            "maxiter": 500,
            "xatol": 2.0,
            "fatol": 1e-5,
            "disp": False,
        },
    )

    if result_local.fun < result_de.fun:
        x_best = result_local.x
    else:
        x_best = result_de.x

    sizing = evaluate_candidate(x_best, maps)
    return x_best, sizing


# =============================================================================
# 9. FINAL VERIFICATION AT MAP POINTS ONLY
# =============================================================================

def print_map_quality(maps: ThrustMaps) -> None:
    """
    Prints quick sanity checks on map values.
    """
    print("\nMap sanity check")
    print("----------------")
    print(
        f"Turbo map: min={np.nanmin(maps.turbo_map['thrust_kN']):.2f} kN, "
        f"max={np.nanmax(maps.turbo_map['thrust_kN']):.2f} kN"
    )
    print(
        f"Ramjet map: min={np.nanmin(maps.ramjet_map['thrust_kN']):.2f} kN, "
        f"max={np.nanmax(maps.ramjet_map['thrust_kN']):.2f} kN"
    )





def plot_thrust_maps(maps: ThrustMaps, mach_points: int = 120, alt_points: int = 120) -> None:
    """
    Plot two Mach-altitude heatmaps:
      1) baseline available thrust [kN]
      2) required thrust [kN]

    Plot convention:
      x-axis = Mach number
      y-axis = altitude [km]
      colour = thrust [kN]

    The available-thrust map uses the turbojet map below M_SWITCH and the
    ramjet map at/above M_SWITCH. Points outside an engine map's altitude/Mach
    range are left blank if ALLOW_EXTRAPOLATION = False.
    """
    import matplotlib.pyplot as plt

    # Use the full mission Mach/altitude plotting domain.
    mach_grid = np.linspace(M_TAKEOFF, M_CRUISE, mach_points)
    alt_grid_m = np.linspace(0.0, H_CRUISE_M, alt_points)

    available_kN = np.full((len(alt_grid_m), len(mach_grid)), np.nan)
    required_kN = np.full((len(alt_grid_m), len(mach_grid)), np.nan)

    for i, h in enumerate(alt_grid_m):
        for j, M in enumerate(mach_grid):
            _, T_avail = thrust_from_maps_kN(maps, M, h)
            available_kN[i, j] = T_avail
            required_kN[i, j] = drag_and_required_thrust_kN(
                M,
                h,
                accel_g=ACCEL_G_TARGET,
            )["thrust_req_kN"]

    # ------------------------------------------------------------------
    # Available thrust heatmap
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    mesh = plt.pcolormesh(
        mach_grid,
        alt_grid_m / 1000.0,
        available_kN,
        shading="auto",
    )
    plt.colorbar(mesh, label="Available thrust [kN]")
    plt.axvline(M_SWITCH, linestyle="--", label="Turbojet/ramjet switch")
    plt.xlabel("Mach number")
    plt.ylabel("Altitude [km]")
    plt.title("Available thrust map")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Required thrust heatmap
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    mesh = plt.pcolormesh(
        mach_grid,
        alt_grid_m / 1000.0,
        required_kN,
        shading="auto",
    )
    plt.colorbar(mesh, label="Required thrust [kN]")
    plt.axvline(M_SWITCH, linestyle="--", label="Turbojet/ramjet switch")
    plt.xlabel("Mach number")
    plt.ylabel("Altitude [km]")
    plt.title("Required thrust map")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()


# =============================================================================
# 10. OUTPUT
# =============================================================================

def save_outputs(
    x_best: np.ndarray,
    sizing: SizingResult,
    maps: ThrustMaps | None = None,
    prefix: str = "optimized_flight_profile_maps",
) -> None:
    """
    Print optimized sizing values and show plots.

    This function intentionally does NOT save:
      - CSV files
      - text summary files
      - PNG plots

    The only files saved by this script are the thrust maps:
      - turbojet_thrust_map.npz
      - ramjet_thrust_map.npz
    """
    mach_grid, h_grid, mach_knots, h_knots = make_profile(x_best)

    knots = pd.DataFrame({
        "Mach": mach_knots,
        "Altitude_m": h_knots,
        "Altitude_km": h_knots / 1000.0,
    })

    print("\n" + "=" * 78)
    print("OPTIMIZED ENGINE SIZING FROM THRUST MAPS")
    print("=" * 78)

    print(f"Turbojet design thrust per engine : {sizing.turbo_design_lbf_per_engine:,.0f} lbf")
    print(f"Turbojet design thrust per engine : {sizing.turbo_design_lbf_per_engine * 0.0044482216152605:,.2f} kN")
    print(f"Ramjet design air mass flow       : {sizing.ramjet_design_mdot_kg_s:,.2f} kg/s")

    if N_RAMJETS > 0:
        print(f"Ramjet design air mass flow/engine: {sizing.ramjet_design_mdot_kg_s / N_RAMJETS:,.2f} kg/s")

    print("-" * 78)
    print(f"Turbo scale vs baseline           : {sizing.max_turbo_scale:.4f}")
    print(f"Ramjet scale vs baseline          : {sizing.max_ramjet_scale:.4f}")
    print(f"Objective                         : {sizing.objective:.6g}")
    print(f"Penalty                           : {sizing.penalty:.6g}")
    print(f"Fast ramjet surrogate used        : {USE_FAST_RAMJET_SURROGATE}")
    print("=" * 78)

    print("\nAltitude knots:")
    print(knots.to_string(index=False))

    print("\nWorst thrust margins:")
    worst = sizing.table.nsmallest(8, "thrust_margin_kN")
    print(
        worst[[
            "mach",
            "altitude_m",
            "mode",
            "thrust_req_kN",
            "sized_T_available_kN",
            "thrust_margin_kN",
            "CL",
            "q_Pa",
        ]].to_string(index=False)
    )

    print("\nNo CSV, TXT, or PNG files were saved. Only thrust-map .npz files are saved/loaded.")

    # Interactive plots only.
    try:
        import matplotlib.pyplot as plt

        df = sizing.table

        # 2D Mach-altitude maps requested by the user.
        if maps is not None:
            plot_thrust_maps(maps)

        plt.figure(figsize=(9, 5))
        plt.plot(df["mach"], df["altitude_m"] / 1000.0, label="Optimized profile")
        plt.scatter(mach_knots, h_knots / 1000.0, label="Knots")
        plt.axvline(M_SWITCH, linestyle="--", label="Turbojet/ramjet switch")
        plt.xlabel("Mach")
        plt.ylabel("Altitude [km]")
        plt.title("Optimized Mach-altitude profile")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(9, 5))
        plt.plot(df["mach"], df["thrust_req_kN"], label="Required")
        plt.plot(df["mach"], df["sized_T_available_kN"], label="Available after sizing")
        plt.axvline(M_SWITCH, linestyle="--", label="M=3 switch")
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

        plt.show()

    except Exception as exc:
        print(f"\nPlotting failed: {exc}")


# =============================================================================
# 11. OPTIONAL FINAL EXACT CHECK ALONG OPTIMIZED PATH
# =============================================================================

def exact_engine_check_at_profile(
    x_best: np.ndarray,
    every_nth_point: int = 3,
) -> pd.DataFrame:
    """
    Optional: rerun the real engine models at some optimized trajectory points.

    This does NOT optimize and does NOT save a CSV.
    It prints and returns a DataFrame comparing:
        map-interpolated baseline thrust
    vs
        exact engine-model baseline thrust

    Use this after you get a good map-based result.

    It only checks every_nth_point by default to save time.
    """
    mach_grid, h_grid, _, _ = make_profile(x_best)
    rows = []

    print("\nRunning optional exact engine check")
    print("-----------------------------------")

    for idx, (M, h) in enumerate(zip(mach_grid, h_grid)):
        if idx % every_nth_point != 0 and idx != len(mach_grid) - 1:
            continue

        if M < M_SWITCH:
            mode = "turbojet"
            T_exact = turbojet_thrust_point_kN(M, h)
        else:
            mode = "ramjet"
            # This remains exact even when the optimization map used the fast
            # surrogate. It is intentionally slow, so only use it for a few
            # final verification points.
            T_exact = ramjet_thrust_point_kN(M, h)

        aero = drag_and_required_thrust_kN(M, h, ACCEL_G_TARGET)

        rows.append({
            "mach": M,
            "altitude_m": h,
            "mode": mode,
            "exact_baseline_T_kN": T_exact,
            "thrust_req_kN": aero["thrust_req_kN"],
            "CL": aero["CL"],
            "q_Pa": aero["q_Pa"],
        })

        print(f"{mode:8s} M={M:4.2f} h={h/1000:5.1f} km exact baseline T={T_exact:10.2f} kN")

    df = pd.DataFrame(rows)

    print("\nExact engine check table:")
    print(df.to_string(index=False))

    return df


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    maps = prepare_thrust_maps()
    print_map_quality(maps)

    x_best, sizing = optimize_profile(maps)
    save_outputs(x_best, sizing, maps)

    # Uncomment if you want to spend extra time verifying real engine calls.
    # exact_engine_check_at_profile(x_best, every_nth_point=3)


if __name__ == "__main__":
    main()
