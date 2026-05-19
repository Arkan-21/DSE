"""
thrust_drag_pycycle_main.py

Uses the drag model from your previous turbojet/drag plotting script, but replaces
its paper-style turbojet thrust calculation with the pyCycle afterburning turbojet
wrapper:

    from turbojet_pycycle_wrapper import turbojet_thrust_kN

Expected files in the same DSE folder:
    - thrust_drag_pycycle_main.py          <-- this file
    - turbojet_pycycle_wrapper.py          <-- pyCycle wrapper with DESIGN_THRUST_LBF = 90_000 lbf
    - pycycle/
    - example_cycles/

Notes:
    - pyCycle is slow. Do not start with hundreds of Mach/altitude points.
    - The wrapper already caches repeated Mach/altitude calls.
    - This script saves a CSV and a PNG plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from turbojet_pycycle_wrapper import turbojet_thrust_kN, DESIGN_THRUST_LBF


# =============================================================================
# SETTINGS
# =============================================================================

N_TURBOJET_ENGINES = 2

# Use fewer points first because pyCycle is slow.
# You can increase this later once the code behaves.
DEFAULT_MACHS = np.linspace(1.2, 3.5, 24)
DEFAULT_ALTITUDES_M = np.array([0, 5_000, 10_000, 15_000, 17_000, 20_000, 25_000], dtype=float)


# =============================================================================
# DRAG MODEL CONSTANTS FROM YOUR PASTED CODE
# =============================================================================

W_TOG_DRAG = 90_000.0       # kg
S_PLAN_DRAG = 350.0         # m²
S_WET_DRAG = 1000.0         # m²
MAC_DRAG = 21.0             # m
IF_DRAG = 1.05
ACCEL_G_DRAG = 0.15
G_DRAG = 9.81


# =============================================================================
# ATMOSPHERE + DRAG FUNCTIONS
# =============================================================================

def drag_atmosphere_prompt(alt_m: float) -> tuple[float, float]:
    """
    Atmosphere from your pasted drag code.

    Returns:
        rho [kg/m³], T [K]
    """
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
    """
    Drag and required thrust calculation from your pasted drag model.

    Returns:
        drag only and drag + 0.15g acceleration requirement.
    """
    M = float(M)
    altitude_m = float(altitude_m)

    rho, T = drag_atmosphere_prompt(altitude_m)
    a = np.sqrt(1.4 * 287.0 * T)
    V = M * a
    q = 0.5 * rho * V**2

    cl_needed = (W_TOG_DRAG * G_DRAG) / max(q * S_PLAN_DRAG, 1e-30)

    # Same as pasted code
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
        "drag_N": float(drag_force_N),
        "drag_kN": float(drag_force_N / 1000.0),
        "drag_friction_kN": float(drag_friction_N / 1000.0),
        "drag_wave_kN": float(drag_wave_N / 1000.0),
        "thrust_required_0p15g_N": float(thrust_required_N),
        "thrust_required_0p15g_kN": float(thrust_required_N / 1000.0),
    }


# =============================================================================
# PYCYCLE TURBOJET THRUST FUNCTION
# =============================================================================

def pycycle_turbojet_thrust_total_kN(M: float, altitude_m: float) -> float:
    """
    Total pyCycle afterburning turbojet thrust in kN.

    This replaces the old paper-style off_design(...) turbojet thrust.
    The engine size is controlled inside turbojet_pycycle_wrapper.py via:

        DESIGN_THRUST_LBF = 90_000.0
    """
    return float(
        turbojet_thrust_kN(
            mach=float(M),
            altitude_m=float(altitude_m),
            n_engines=N_TURBOJET_ENGINES,
            clamp_negative=True,
        )
    )


# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def make_thrust_drag_altitude_plot(
    machs: np.ndarray | None = None,
    altitudes_m: np.ndarray | None = None,
    output_png: str = "pycycle_thrust_drag_vs_mach_altitudes.png",
    output_csv: str = "pycycle_thrust_drag_vs_mach_altitudes.csv",
):
    """
    Plot pyCycle turbojet thrust and aircraft drag versus Mach number for
    different altitudes.
    """
    if machs is None:
        machs = DEFAULT_MACHS
    if altitudes_m is None:
        altitudes_m = DEFAULT_ALTITUDES_M

    rows = []

    print()
    print("Running pyCycle thrust + drag sweep")
    print("-----------------------------------")
    print(f"Design thrust per turbojet engine: {DESIGN_THRUST_LBF:,.0f} lbf")
    print(f"Number of turbojet engines: {N_TURBOJET_ENGINES}")
    print(f"Mach points: {len(machs)}")
    print(f"Altitude points: {len(altitudes_m)}")
    print(f"Total pyCycle calls: {len(machs) * len(altitudes_m)}")
    print()

    for h in altitudes_m:
        for M in machs:
            M = float(M)
            h = float(h)

            thrust_kN = pycycle_turbojet_thrust_total_kN(M, h)
            dr = drag_breakdown_prompt(M, h)

            drag_kN = dr["drag_kN"]
            required_kN = dr["thrust_required_0p15g_kN"]

            rows.append({
                "Mach": M,
                "altitude_m": h,
                "altitude_km": h / 1000.0,
                "pycycle_turbojet_thrust_kN": thrust_kN,
                "drag_kN": drag_kN,
                "required_thrust_0p15g_kN": required_kN,
                "net_vs_drag_kN": thrust_kN - drag_kN if np.isfinite(thrust_kN) else np.nan,
                "net_vs_0p15g_req_kN": thrust_kN - required_kN if np.isfinite(thrust_kN) else np.nan,
                "CL": dr["CL"],
                "alpha_deg": dr["alpha_deg"],
                "cd_total": dr["cd_total"],
                "cd_f": dr["cd_f"],
                "cd_wave": dr["cd_wave"],
                "drag_friction_kN": dr["drag_friction_kN"],
                "drag_wave_kN": dr["drag_wave_kN"],
                "rho": dr["rho"],
                "T": dr["T"],
                "V": dr["V"],
                "q": dr["q"],
                "Re": dr["Re"],
                "n_turbojet_engines": N_TURBOJET_ENGINES,
                "design_thrust_lbf_per_engine": DESIGN_THRUST_LBF,
            })

            print(
                f"h={h/1000:5.1f} km, M={M:4.2f}: "
                f"T={thrust_kN:10.2f} kN, "
                f"D={drag_kN:10.2f} kN, "
                f"D+0.15g={required_kN:10.2f} kN"
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7.5))

    for h in altitudes_m:
        sub = df[df["altitude_m"] == float(h)].sort_values("Mach")
        M_arr = sub["Mach"].to_numpy()
        T_arr = sub["pycycle_turbojet_thrust_kN"].to_numpy()
        D_arr = sub["drag_kN"].to_numpy()
        R_arr = sub["required_thrust_0p15g_kN"].to_numpy()

        label_h = f"{h/1000:.0f} km"
        ax.plot(M_arr, T_arr, linewidth=2.2, label=f"pyCycle turbojet thrust {label_h}")
        ax.plot(M_arr, D_arr, linestyle="--", linewidth=1.7, label=f"Drag {label_h}")
        ax.plot(M_arr, R_arr, linestyle=":", linewidth=1.5, label=f"Drag + 0.15g {label_h}")

    ax.axvline(3.0, color="black", linestyle="-.", linewidth=1.5, label="M=3 reference")
    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("Force [kN]")
    ax.set_title(
        "pyCycle afterburning turbojet thrust and drag vs Mach number\n"
        f"DESIGN_THRUST_LBF={DESIGN_THRUST_LBF:,.0f} per engine, "
        f"N_engines={N_TURBOJET_ENGINES}"
    )
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()

    print()
    print("Saved:")
    print(f"  {output_png}")
    print(f"  {output_csv}")

    return df


# =============================================================================
# QUICK REQUIRED DESIGN THRUST ESTIMATOR
# =============================================================================

def estimate_required_design_thrust_from_drag(
    df: pd.DataFrame,
    n_engines: int = N_TURBOJET_ENGINES,
    margin: float = 1.2,
) -> float:
    """
    Estimates required design thrust per engine in lbf from the maximum
    Drag + acceleration requirement in the dataframe.

    This does NOT resize pyCycle automatically. It only tells you the suggested
    design thrust level to put inside turbojet_pycycle_wrapper.py.
    """
    LBF_TO_KN = 0.0044482216152605

    max_required_kN = float(df["required_thrust_0p15g_kN"].max())
    per_engine_required_kN = margin * max_required_kN / n_engines
    per_engine_required_lbf = per_engine_required_kN / LBF_TO_KN

    return per_engine_required_lbf


if __name__ == "__main__":
    # Start with fewer points. Increase after testing.
    machs = np.linspace(1.2, 3.0, 8)
    altitudes_m = np.array([10_000, 12_000, 15_000, 18_000, 20_000, 22_000, 25_000, 28_000, 30_000], dtype=float)

    df = make_thrust_drag_altitude_plot(
        machs=machs,
        altitudes_m=altitudes_m,
        output_png="pycycle_thrust_drag_vs_mach_altitudes.png",
        output_csv="pycycle_thrust_drag_vs_mach_altitudes.csv",
    )

    suggested_lbf = estimate_required_design_thrust_from_drag(
        df,
        n_engines=N_TURBOJET_ENGINES,
        margin=1.2,
    )

    print()
    print("Sizing check")
    print("------------")
    print(f"Max drag only: {df['drag_kN'].max():.2f} kN")
    print(f"Max drag + 0.15g requirement: {df['required_thrust_0p15g_kN'].max():.2f} kN")
    print(f"Suggested design thrust per engine with 20% margin: {suggested_lbf:,.0f} lbf")
