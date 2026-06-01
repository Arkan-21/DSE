"""
plot_altitude_vs_range.py

Standalone altitude-vs-range demonstrator for the optimized transition profile.

What it does
------------
1) Imports your transition-profile optimizer.
2) Loads/uses the existing turbojet and ramjet thrust maps.
3) Runs the optimizer to get the optimized Mach-altitude profile.
4) Converts the Mach-altitude profile into an approximate range profile.
5) Adds Mach 5 cruise at 30 km.
6) Adds the unpowered descent using the combined sizing analyse_descent() model.
7) Plots altitude vs range interactively.

It does NOT save PNG/TXT/CSV files.

Expected files in the same folder
---------------------------------
Preferred:
    optimize_transition_profile.py

Also accepted:
    optimize_transition_profile_updated_drag.py
    optimize_transition_profile_plot_only.py

Plus the existing maps:
    turbojet_thrust_map.npz
    ramjet_thrust_map.npz
"""

from __future__ import annotations


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
from common.project_paths import data_file, thrust_map_file, source_file
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# User settings
# =============================================================================

# Leave as None to auto-detect your optimizer file in the current folder.
OPTIMIZER_FILE: str | None = None

# Leave as None to auto-detect your combined sizing file in the current folder.
# This is used ONLY for the descent model via analyse_descent().
COMBINED_SIZING_FILE: str | None = None

# Mission/range settings
TOTAL_RANGE_M = 9_500_000.0
CRUISE_TIME_S = 90.0 * 60.0

# Acceleration used to convert the optimized Mach-altitude path into range.
# This should match the transition optimizer's acceleration requirement.
ACCEL_G_FOR_RANGE = 0.15

# Number of points used to smooth the cruise/descent curves.
N_CRUISE_POINTS = 80
N_DESCENT_POINTS = 80

# Optional: show Mach markers along ascent
SHOW_MACH_LABELS = True


# =============================================================================
# Import helper
# =============================================================================

def find_optimizer_file() -> Path:
    if OPTIMIZER_FILE is not None:
        path = Path(OPTIMIZER_FILE)
        if not path.exists():
            raise FileNotFoundError(f"OPTIMIZER_FILE does not exist: {path}")
        return path

    candidates = [
        "optimize_transition_profile.py",
        "optimize_transition_profile_updated_drag.py",
        "optimize_transition_profile_plot_only.py",
        "flight_profile_optimizer_maps.py",
    ]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find a transition optimizer file. Set OPTIMIZER_FILE at the "
        "top of this script."
    )


def find_combined_sizing_file() -> Path:
    if COMBINED_SIZING_FILE is not None:
        path = Path(COMBINED_SIZING_FILE)
        if not path.exists():
            raise FileNotFoundError(f"COMBINED_SIZING_FILE does not exist: {path}")
        return path

    candidates = [
        source_file("sizing", "combined_sizing.py"),
        "combined_sizing_imported_profile_no_double_accel.py",
        "combined_sizing_with_imported_profile_fixed_import.py",
        "combined_sizing_with_imported_profile.py",
    ]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find a combined sizing file. Set COMBINED_SIZING_FILE at the "
        "top of this script."
    )


def import_module_from_path(path: Path, module_name: str):
    """
    Dynamic import that is safe for modules containing @dataclass.

    Python 3.13 dataclasses expect the module to already exist in sys.modules.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def import_optimizer(path: Path):
    return import_module_from_path(path, "transition_optimizer_for_range_plot")


def import_combined_sizing(path: Path):
    return import_module_from_path(path, "combined_sizing_for_range_plot")


# =============================================================================
# Range construction
# =============================================================================

def velocity_from_optimizer(module, mach: float, altitude_m: float) -> float:
    """
    Use the optimizer's atmosphere if available so the range plot is consistent.
    """
    if hasattr(module, "atmosphere_drag"):
        _, T = module.atmosphere_drag(float(altitude_m))
        gamma = getattr(module, "GAMMA", 1.4)
        R = getattr(module, "R_GAS", 287.05)
        return float(mach) * math.sqrt(gamma * R * T)

    # Fallback ISA
    if altitude_m <= 11_000:
        T = 288.15 - 0.0065 * altitude_m
    elif altitude_m <= 20_000:
        T = 216.65
    else:
        T = 216.65 + 0.001 * (altitude_m - 20_000.0)

    return float(mach) * math.sqrt(1.4 * 287.05 * T)


def build_ascent_range(module, mach: np.ndarray, altitude_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts the optimized Mach-altitude curve into approximate horizontal range.

    Assumption:
        The vehicle follows the optimized altitude/Mach path while accelerating
        with approximately ACCEL_G_FOR_RANGE * g along the flight path.

    For each segment:
        ds = (V2^2 - V1^2) / (2 a)
        dx = sqrt(ds^2 - dh^2)

    If a segment has too little ds for its altitude change, dx is clipped to 0.
    """
    g = getattr(module, "G", 9.81)
    accel = ACCEL_G_FOR_RANGE * g

    V = np.array([
        velocity_from_optimizer(module, M, h)
        for M, h in zip(mach, altitude_m)
    ])

    x = np.zeros_like(mach, dtype=float)

    for i in range(1, len(mach)):
        V0 = V[i - 1]
        V1 = V[i]
        h0 = altitude_m[i - 1]
        h1 = altitude_m[i]

        dV2 = max(0.0, V1**2 - V0**2)
        ds = dV2 / max(2.0 * accel, 1e-12)
        dh = h1 - h0

        # Horizontal range. If path length is smaller than altitude gain due
        # to the simple acceleration approximation, use zero horizontal distance.
        dx = math.sqrt(max(ds**2 - dh**2, 0.0))
        x[i] = x[i - 1] + dx

    return x, altitude_m, V


def append_cruise_and_descent(
    module,
    combined_module,
    x_ascent: np.ndarray,
    h_ascent: np.ndarray,
    mach_ascent: np.ndarray,
) -> dict[str, np.ndarray | float | bool]:
    """
    Append Mach 5 cruise and a descent computed using the combined sizing logic.

    This calls combined_module.analyse_descent(), so the descent range/time and
    accelerations match the combined sizing model. The descent curve is then
    reconstructed kinematically from:

        x(t) = x0 + V_cruise*t + 0.5*a_x*t^2
        h(t) = h_cruise + 0.5*a_y*t^2

    where a_x_descent < 0 and a_y_descent < 0.
    """
    if not hasattr(combined_module, "analyse_descent"):
        raise AttributeError(
            "The combined sizing file must define analyse_descent()."
        )

    h_cruise = float(getattr(module, "H_CRUISE_M", h_ascent[-1]))
    M_cruise = float(getattr(module, "M_CRUISE", mach_ascent[-1]))

    V_cruise = velocity_from_optimizer(module, M_cruise, h_cruise)
    cruise_range = CRUISE_TIME_S * V_cruise

    x_climb_end = float(x_ascent[-1])
    x_cruise_start = x_climb_end
    x_cruise_end = x_cruise_start + cruise_range

    descent = combined_module.analyse_descent(
        end_cruise_x=x_cruise_end,
        h_cruise=h_cruise,
        v_cruise=V_cruise,
        acc_tot=ACCEL_G_FOR_RANGE * getattr(module, "G", 9.81),
        total_range=TOTAL_RANGE_M,
    )

    x_descent_range = float(descent["x_descent"])
    t_descent = float(descent["t_descent"])
    a_x = float(descent["a_x_descent"])
    a_y = float(descent["a_y_descent"])

    # If combined sizing reports the planned descent did not fit, its
    # analyse_descent() returns the shortest feasible descent instead.
    x_final = float(descent.get("final_total_range", x_cruise_end + x_descent_range))
    descent_fits_range = bool(descent.get("descent_fits_total_range", True))

    # Cruise points, skip first point later to avoid duplication.
    x_cruise = np.linspace(x_cruise_start, x_cruise_end, N_CRUISE_POINTS)
    h_cruise_arr = np.full_like(x_cruise, h_cruise)

    # Reconstruct descent from combined sizing kinematics.
    t = np.linspace(0.0, t_descent, N_DESCENT_POINTS)
    x_descent = x_cruise_end + V_cruise * t + 0.5 * a_x * t**2
    h_descent = h_cruise + 0.5 * a_y * t**2

    # Numerical cleanup.
    h_descent = np.maximum(h_descent, 0.0)

    x_total = np.concatenate([
        x_ascent,
        x_cruise[1:],
        x_descent[1:],
    ])

    h_total = np.concatenate([
        h_ascent,
        h_cruise_arr[1:],
        h_descent[1:],
    ])

    return {
        "x_total": x_total,
        "h_total": h_total,
        "x_ascent": x_ascent,
        "h_ascent": h_ascent,
        "x_cruise_start": x_cruise_start,
        "x_cruise_end": x_cruise_end,
        "x_final": x_final,
        "cruise_range": cruise_range,
        "descent_range": x_descent_range,
        "V_cruise": V_cruise,
        "h_cruise": h_cruise,
        "M_cruise": M_cruise,
        "a_x_descent": a_x,
        "a_y_descent": a_y,
        "a_descent": float(descent["a_descent"]),
        "t_descent": t_descent,
        "descent_fits_range": descent_fits_range,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    optimizer_path = find_optimizer_file()
    combined_path = find_combined_sizing_file()

    print(f"Using optimizer file:       {optimizer_path}")
    print(f"Using combined sizing file: {combined_path}")

    module = import_optimizer(optimizer_path)
    combined_module = import_combined_sizing(combined_path)

    required_functions = ["prepare_thrust_maps", "optimize_profile", "make_profile"]
    missing = [name for name in required_functions if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"The optimizer file is missing required functions: {missing}"
        )

    maps = module.prepare_thrust_maps()
    x_best, sizing = module.optimize_profile(maps)

    mach, altitude_m, mach_knots, altitude_knots = module.make_profile(x_best)

    x_ascent, h_ascent, V_ascent = build_ascent_range(module, mach, altitude_m)
    mission = append_cruise_and_descent(module, combined_module, x_ascent, h_ascent, mach)

    print("\nAltitude-vs-range mission geometry")
    print("----------------------------------")
    print(f"Climb/accel range:     {mission['x_cruise_start'] / 1000.0:,.2f} km")
    print(f"Cruise range:          {mission['cruise_range'] / 1000.0:,.2f} km")
    print(f"Descent range:         {mission['descent_range'] / 1000.0:,.2f} km")
    print(f"Final total range:     {mission['x_final'] / 1000.0:,.2f} km")
    print(f"Descent fits range:    {mission['descent_fits_range']}")
    print(f"Descent time:          {mission['t_descent']:,.2f} s")
    print(f"Descent ax:            {mission['a_x_descent']:.4f} m/s²")
    print(f"Descent ay:            {mission['a_y_descent']:.4f} m/s²")
    print(f"Descent total accel:   {mission['a_descent']:.4f} m/s²")
    print(f"Cruise Mach:           {mission['M_cruise']:.2f}")
    print(f"Cruise altitude:       {mission['h_cruise'] / 1000.0:.2f} km")
    print(f"Cruise velocity:       {mission['V_cruise']:.2f} m/s")

    if hasattr(sizing, "turbo_design_lbf_per_engine"):
        print("\nImported optimized sizing")
        print("-------------------------")
        print(f"Turbojet design thrust: {sizing.turbo_design_lbf_per_engine:,.0f} lbf/engine")
        print(f"Ramjet design mdot:     {sizing.ramjet_design_mdot_kg_s:,.2f} kg/s total")

    # Plot
    plt.figure(figsize=(10, 5.6))
    plt.plot(
        np.asarray(mission["x_total"]) / 1000.0,
        np.asarray(mission["h_total"]) / 1000.0,
        linewidth=2.2,
        label="Altitude profile",
    )

    plt.axvline(
        mission["x_cruise_start"] / 1000.0,
        linestyle="--",
        linewidth=1.2,
        label="Start cruise",
    )
    plt.axvline(
        mission["x_cruise_end"] / 1000.0,
        linestyle="--",
        linewidth=1.2,
        label="End cruise",
    )

    plt.scatter(x_ascent / 1000.0, h_ascent / 1000.0, s=18, label="Optimized climb points")

    if SHOW_MACH_LABELS:
        label_indices = np.linspace(0, len(mach) - 1, 7, dtype=int)
        for idx in label_indices:
            plt.annotate(
                f"M{mach[idx]:.1f}",
                (x_ascent[idx] / 1000.0, h_ascent[idx] / 1000.0),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    plt.xlabel("Range [km]")
    plt.ylabel("Altitude [km]")
    plt.title("Altitude vs Range with Combined-Sizing Descent Model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
