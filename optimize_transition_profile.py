"""
optimize_transition_profile.py

Optimizes a Mach 1.2 -> 3.0 transition profile with altitude as a decision variable.

Goal:
    Use the least required propulsive force while still satisfying:
        thrust_available >= drag + 0.15g acceleration requirement

Concept:
    - Start at M = 1.2, h = 10 km.
    - Climb toward ramjet takeover around h = 18 km.
    - Turbojet is assumed active before ramjet takeover.
    - Ramjet may take over once altitude >= 18 km and Mach >= RAMJET_MIN_MACH.
    - The optimizer chooses altitude at each Mach station on a grid.
    - It minimizes peak required turbojet thrust before takeover first, then peak ramjet thrust after takeover.

Important:
    This script optimizes REQUIRED thrust using your drag model.
    It does not require thousands of pyCycle calls.

Optional:
    If CHECK_PYCYCLE_AVAILABLE_THRUST = True, it also calls your pyCycle wrapper
    along the optimized turbojet segment to check installed turbojet thrust.

Expected files in same folder:
    - turbojet_pycycle_wrapper.py   only needed if CHECK_PYCYCLE_AVAILABLE_THRUST = True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

MACHS = np.linspace(1.2, 3.0, 20)

START_ALTITUDE_M = 10_000.0
RAMJET_TAKEOVER_ALT_M = 18_000.0
RAMJET_MIN_MACH = 2.4
END_MIN_ALTITUDE_M = 18_000.0

# Candidate altitudes for optimizer.
# Smaller step = smoother/more accurate but more computation.
ALTITUDE_GRID_M = np.arange(10_000.0, 18_000.0 + 1.0, 250.0)

# Climb-rate proxy: maximum altitude gain per Mach number.
# Example: 6000 m/Mach means over delta-M=0.1, max climb is 600 m.
# Increase this if the optimizer cannot reach 18 km by M=3.
MAX_CLIMB_PER_MACH_M = 7_000.0

# Optional lower bound for climb. Keep 0 for monotonic climb only.
MIN_CLIMB_PER_MACH_M = 0.0

N_TURBOJET_ENGINES = 2
N_RAMJET_ENGINES = 2
DESIGN_MARGIN = 1.20

# If True, imports turbojet_pycycle_wrapper and checks pyCycle available thrust
# only along the optimized turbojet portion. Leave False while tuning.
CHECK_PYCYCLE_AVAILABLE_THRUST = False


# =============================================================================
# DRAG MODEL CONSTANTS FROM YOUR CURRENT CODE
# =============================================================================

W_TOG_DRAG = 90_000.0       # kg
S_PLAN_DRAG = 350.0         # m²
S_WET_DRAG = 1000.0         # m²
MAC_DRAG = 21.0             # m
IF_DRAG = 1.05
ACCEL_G_DRAG = 0.15
G_DRAG = 9.81
LBF_TO_KN = 0.0044482216152605


# =============================================================================
# DRAG MODEL
# =============================================================================

def drag_atmosphere_prompt(alt_m: float) -> tuple[float, float]:
    """Atmosphere from your pasted drag code. Returns rho [kg/m³], T [K]."""
    alt_m = float(alt_m)

    if alt_m <= 11_000:
        T = 288.15 - 0.0065 * alt_m
        rho = 1.225 * (T / 288.15) ** 4.256
    elif alt_m <= 25_000:
        T = 216.65
        rho = 0.3639 * np.exp(-0.000157 * (alt_m - 11_000))
    else:
        T = 216.65 + 0.003 * (alt_m - 25_000)
        rho = 0.0401 * (T / 216.65) ** -11.388

    return float(rho), float(T)


def reynolds_prompt(rho: float, v: float, temp: float, chord: float) -> float:
    """Reynolds number with Sutherland viscosity from your pasted drag code."""
    mu_0 = 1.7894e-5
    T_0 = 273.15
    S_suth = 110.4
    mu = mu_0 * (temp / T_0) ** 1.5 * (T_0 + S_suth) / (temp + S_suth)
    return float((rho * v * chord) / mu)


def drag_breakdown_prompt(M: float, altitude_m: float) -> dict[str, float]:
    """Drag and drag + 0.15g acceleration requirement."""
    M = float(M)
    altitude_m = float(altitude_m)

    rho, T = drag_atmosphere_prompt(altitude_m)
    a = np.sqrt(1.4 * 287.0 * T)
    V = M * a
    q = 0.5 * rho * V**2

    cl_needed = (W_TOG_DRAG * G_DRAG) / max(q * S_PLAN_DRAG, 1e-30)

    alpha_rad = np.sqrt((cl_needed**0.75) / 2.0)
    cd_wave = 2.0 * np.sin(alpha_rad) ** 3

    Re_dyn = reynolds_prompt(rho, V, T, MAC_DRAG)
    cf = 0.455 / (np.log10(Re_dyn) ** 2.58)
    cd_f = cf * IF_DRAG * (S_WET_DRAG / S_PLAN_DRAG)

    cd_total = cd_f + cd_wave
    drag_force_N = q * S_PLAN_DRAG * cd_total
    drag_friction_N = q * S_PLAN_DRAG * cd_f
    drag_wave_N = q * S_PLAN_DRAG * cd_wave
    thrust_required_N = drag_force_N + (W_TOG_DRAG * ACCEL_G_DRAG * G_DRAG)

    return {
        "M": M,
        "altitude_m": altitude_m,
        "altitude_km": altitude_m / 1000.0,
        "rho": float(rho),
        "T": float(T),
        "V": float(V),
        "q": float(q),
        "CL": float(cl_needed),
        "alpha_deg": float(np.degrees(alpha_rad)),
        "Re": float(Re_dyn),
        "cf": float(cf),
        "cd_f": float(cd_f),
        "cd_wave": float(cd_wave),
        "cd_total": float(cd_total),
        "drag_kN": float(drag_force_N / 1000.0),
        "drag_friction_kN": float(drag_friction_N / 1000.0),
        "drag_wave_kN": float(drag_wave_N / 1000.0),
        "required_thrust_0p15g_kN": float(thrust_required_N / 1000.0),
    }


# =============================================================================
# DYNAMIC-PROGRAMMING OPTIMIZER
# =============================================================================

@dataclass(frozen=True)
class Cost:
    peak_turbo_kN: float
    peak_ramjet_kN: float
    integrated_required_kN: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.peak_turbo_kN, self.peak_ramjet_kN, self.integrated_required_kN)


def better(a: Optional[Cost], b: Cost) -> bool:
    """Return True if cost b is better than existing cost a."""
    if a is None:
        return True
    return b.as_tuple() < a.as_tuple()


def optimize_profile(
    machs: np.ndarray = MACHS,
    altitude_grid_m: np.ndarray = ALTITUDE_GRID_M,
) -> pd.DataFrame:
    """
    Optimize altitude at each Mach station.

    State includes:
        altitude index
        takeover flag: 0 = still turbojet, 1 = ramjet has taken over

    Objective priority:
        1. Minimize peak turbojet required thrust before takeover.
        2. Minimize peak ramjet required thrust after takeover.
        3. Minimize integrated required thrust over the path.
    """
    machs = np.asarray(machs, dtype=float)
    altitude_grid_m = np.asarray(altitude_grid_m, dtype=float)

    nM = len(machs)
    nH = len(altitude_grid_m)

    # Precompute required thrust and drag data.
    req = np.zeros((nM, nH))
    drag = np.zeros((nM, nH))
    alpha = np.zeros((nM, nH))
    cl = np.zeros((nM, nH))
    cd_total = np.zeros((nM, nH))
    for i, M in enumerate(machs):
        for j, h in enumerate(altitude_grid_m):
            dr = drag_breakdown_prompt(M, h)
            req[i, j] = dr["required_thrust_0p15g_kN"]
            drag[i, j] = dr["drag_kN"]
            alpha[i, j] = dr["alpha_deg"]
            cl[i, j] = dr["CL"]
            cd_total[i, j] = dr["cd_total"]

    # Find start altitude grid point.
    start_j = int(np.argmin(np.abs(altitude_grid_m - START_ALTITUDE_M)))
    if abs(altitude_grid_m[start_j] - START_ALTITUDE_M) > 1e-6:
        print(f"Start altitude snapped to grid: {altitude_grid_m[start_j]/1000:.2f} km")

    # DP arrays: cost[i][j][mode], prev pointer.
    costs: list[list[list[Optional[Cost]]]] = [
        [[None, None] for _ in range(nH)] for _ in range(nM)
    ]
    prev: list[list[list[Optional[tuple[int, int]]]]] = [
        [[None, None] for _ in range(nH)] for _ in range(nM)
    ]

    # Start point is turbojet mode.
    costs[0][start_j][0] = Cost(
        peak_turbo_kN=req[0, start_j],
        peak_ramjet_kN=0.0,
        integrated_required_kN=req[0, start_j],
    )

    for i in range(1, nM):
        dM = machs[i] - machs[i - 1]
        max_dh = MAX_CLIMB_PER_MACH_M * dM
        min_dh = MIN_CLIMB_PER_MACH_M * dM

        for j_prev, h_prev in enumerate(altitude_grid_m):
            for mode_prev in (0, 1):
                old = costs[i - 1][j_prev][mode_prev]
                if old is None:
                    continue

                for j, h in enumerate(altitude_grid_m):
                    dh = h - h_prev

                    # Monotonic climb with slope bounds.
                    if dh < min_dh - 1e-9:
                        continue
                    if dh > max_dh + 1e-9:
                        continue

                    qualifies_for_ramjet = (h >= RAMJET_TAKEOVER_ALT_M and machs[i] >= RAMJET_MIN_MACH)
                    mode_new = 1 if (mode_prev == 1 or qualifies_for_ramjet) else 0

                    required = req[i, j]
                    integrated = old.integrated_required_kN + 0.5 * (req[i - 1, j_prev] + required) * dM

                    if mode_new == 0:
                        new_cost = Cost(
                            peak_turbo_kN=max(old.peak_turbo_kN, required),
                            peak_ramjet_kN=old.peak_ramjet_kN,
                            integrated_required_kN=integrated,
                        )
                    else:
                        new_cost = Cost(
                            peak_turbo_kN=old.peak_turbo_kN,
                            peak_ramjet_kN=max(old.peak_ramjet_kN, required),
                            integrated_required_kN=integrated,
                        )

                    if better(costs[i][j][mode_new], new_cost):
                        costs[i][j][mode_new] = new_cost
                        prev[i][j][mode_new] = (j_prev, mode_prev)

    # Choose final state: must end at or above END_MIN_ALTITUDE_M and ramjet taken over.
    best_state = None
    best_cost = None
    i = nM - 1
    for j, h in enumerate(altitude_grid_m):
        if h < END_MIN_ALTITUDE_M:
            continue
        c = costs[i][j][1]
        if c is not None and better(best_cost, c):
            best_cost = c
            best_state = (j, 1)

    if best_state is None:
        raise RuntimeError(
            "No feasible profile found. Try increasing MAX_CLIMB_PER_MACH_M or extending ALTITUDE_GRID_M."
        )

    # Backtrack.
    profile_j = np.zeros(nM, dtype=int)
    profile_mode = np.zeros(nM, dtype=int)
    j, mode = best_state
    for i in range(nM - 1, -1, -1):
        profile_j[i] = j
        profile_mode[i] = mode
        p = prev[i][j][mode]
        if p is None:
            break
        j, mode = p

    rows = []
    for i, M in enumerate(machs):
        j = profile_j[i]
        h = altitude_grid_m[j]
        mode = "ramjet" if profile_mode[i] == 1 else "turbojet"
        rows.append({
            "Mach": M,
            "altitude_m": h,
            "altitude_km": h / 1000.0,
            "engine_mode": mode,
            "required_thrust_0p15g_kN": req[i, j],
            "drag_kN": drag[i, j],
            "CL": cl[i, j],
            "alpha_deg": alpha[i, j],
            "cd_total": cd_total[i, j],
        })

    df = pd.DataFrame(rows)
    df["is_turbojet"] = df["engine_mode"].eq("turbojet")
    df["is_ramjet"] = df["engine_mode"].eq("ramjet")
    df["peak_turbo_required_kN"] = df.loc[df["is_turbojet"], "required_thrust_0p15g_kN"].max()
    df["peak_ramjet_required_kN"] = df.loc[df["is_ramjet"], "required_thrust_0p15g_kN"].max()

    return df


# =============================================================================
# OPTIONAL PYCYCLE CHECK
# =============================================================================

def add_pycycle_check(df: pd.DataFrame) -> pd.DataFrame:
    """Optionally add available pyCycle turbojet thrust along turbojet segment."""
    if not CHECK_PYCYCLE_AVAILABLE_THRUST:
        df["pycycle_turbojet_available_kN"] = np.nan
        df["turbojet_margin_kN"] = np.nan
        return df

    from turbojet_pycycle_wrapper import turbojet_thrust_kN, DESIGN_THRUST_LBF

    available = []
    margin = []
    for _, row in df.iterrows():
        if row["engine_mode"] == "turbojet":
            T = turbojet_thrust_kN(
                mach=float(row["Mach"]),
                altitude_m=float(row["altitude_m"]),
                n_engines=N_TURBOJET_ENGINES,
                clamp_negative=True,
            )
            available.append(T)
            margin.append(T - row["required_thrust_0p15g_kN"])
        else:
            available.append(np.nan)
            margin.append(np.nan)

    df = df.copy()
    df["pycycle_turbojet_available_kN"] = available
    df["turbojet_margin_kN"] = margin
    df["design_thrust_lbf_per_engine"] = DESIGN_THRUST_LBF
    return df


# =============================================================================
# PLOTTING + SIZING
# =============================================================================

def summarize_and_plot(df: pd.DataFrame) -> None:
    peak_turbo = float(df.loc[df["is_turbojet"], "required_thrust_0p15g_kN"].max())
    peak_ramjet = float(df.loc[df["is_ramjet"], "required_thrust_0p15g_kN"].max())

    suggested_turbo_lbf = DESIGN_MARGIN * peak_turbo / N_TURBOJET_ENGINES / LBF_TO_KN
    suggested_ramjet_kN_per_engine = DESIGN_MARGIN * peak_ramjet / max(N_RAMJET_ENGINES, 1)

    takeover_rows = df[df["engine_mode"] == "ramjet"]
    if len(takeover_rows) > 0:
        takeover = takeover_rows.iloc[0]
        takeover_text = f"M={takeover['Mach']:.2f}, h={takeover['altitude_km']:.2f} km"
    else:
        takeover_text = "no takeover"

    print("\nOptimized transition profile")
    print("----------------------------")
    print(df[["Mach", "altitude_km", "engine_mode", "required_thrust_0p15g_kN", "drag_kN", "alpha_deg"]].to_string(index=False))
    print("\nSizing summary")
    print("--------------")
    print(f"Ramjet takeover: {takeover_text}")
    print(f"Peak required turbojet total thrust: {peak_turbo:.2f} kN")
    print(f"Peak required ramjet total thrust:   {peak_ramjet:.2f} kN")
    print(f"Suggested turbojet design thrust per engine with {DESIGN_MARGIN:.0%} margin: {suggested_turbo_lbf:,.0f} lbf")
    print(f"Suggested ramjet thrust per engine with {DESIGN_MARGIN:.0%} margin: {suggested_ramjet_kN_per_engine:.2f} kN")

    if CHECK_PYCYCLE_AVAILABLE_THRUST and "turbojet_margin_kN" in df.columns:
        min_margin = df.loc[df["engine_mode"] == "turbojet", "turbojet_margin_kN"].min()
        print(f"Minimum pyCycle turbojet margin before handover: {min_margin:.2f} kN")

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax1.plot(df["Mach"], df["altitude_km"], marker="o", linewidth=2.5, label="Optimized altitude")
    ax1.axhline(RAMJET_TAKEOVER_ALT_M / 1000.0, linestyle="--", linewidth=1.5, label="Ramjet takeover altitude")
    ax1.set_xlabel("Mach number [-]")
    ax1.set_ylabel("Altitude [km]")
    ax1.grid(True, alpha=0.35)

    ax2 = ax1.twinx()
    turbo = df[df["engine_mode"] == "turbojet"]
    ram = df[df["engine_mode"] == "ramjet"]
    ax2.plot(df["Mach"], df["required_thrust_0p15g_kN"], linestyle=":", linewidth=2.0, label="Required thrust D + 0.15g")
    ax2.scatter(turbo["Mach"], turbo["required_thrust_0p15g_kN"], marker="s", s=55, label="Turbojet segment")
    ax2.scatter(ram["Mach"], ram["required_thrust_0p15g_kN"], marker="^", s=65, label="Ramjet segment")
    ax2.set_ylabel("Required thrust [kN]")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Optimized Mach 1.2 → 3.0 transition profile\nminimize turbojet required thrust before ramjet handover")
    fig.tight_layout()
    fig.savefig("optimized_transition_profile.png", dpi=300, bbox_inches="tight")
    plt.show()


def main() -> pd.DataFrame:
    df = optimize_profile(MACHS, ALTITUDE_GRID_M)
    df = add_pycycle_check(df)
    df.to_csv("optimized_transition_profile.csv", index=False)
    summarize_and_plot(df)
    print("\nSaved:")
    print("  optimized_transition_profile.csv")
    print("  optimized_transition_profile.png")
    return df


if __name__ == "__main__":
    main()
