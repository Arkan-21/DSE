import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from Engine.scramjet_01 import Scramjet as Scramjet_engine
from Engine.scramjet_01 import run_altitude_mach_map


# =============================================================================
# ISA atmosphere
# =============================================================================

def isa_temperature(altitude_m: float) -> float:
    if altitude_m <= 11_000.0:
        return 288.15 - 0.0065 * altitude_m
    elif altitude_m <= 20_000.0:
        return 216.65
    elif altitude_m <= 32_000.0:
        return 216.65 + 0.001 * (altitude_m - 20_000.0)
    else:
        return 228.65


def isa_pressure(altitude_m: float) -> float:
    g0 = 9.80665
    R = 287.05

    if altitude_m <= 11_000.0:
        T0 = 288.15
        L = -0.0065
        T = T0 + L * altitude_m
        return 101325.0 * (T / T0) ** (-g0 / (L * R))
    elif altitude_m <= 20_000.0:
        T = 216.65
        p11 = 22632.06
        return p11 * math.exp(-g0 * (altitude_m - 11_000.0) / (R * T))
    elif altitude_m <= 32_000.0:
        T20 = 216.65
        p20 = 5474.89
        L = 0.001
        T = T20 + L * (altitude_m - 20_000.0)
        return p20 * (T / T20) ** (-g0 / (L * R))
    else:
        T = 228.65
        p32 = 868.02
        return p32 * math.exp(-g0 * (altitude_m - 32_000.0) / (R * T))


def isa_density(altitude_m: float) -> float:
    R = 287.05
    return isa_pressure(altitude_m) / (R * isa_temperature(altitude_m))


def speed_of_sound(altitude_m: float) -> float:
    gamma_air = 1.4
    R = 287.05
    return math.sqrt(gamma_air * R * isa_temperature(altitude_m))


def dynamic_pressure_from_mach_altitude(M: float, altitude_m: float) -> float:
    rho = isa_density(altitude_m)
    V = M * speed_of_sound(altitude_m)
    return 0.5 * rho * V**2


# =============================================================================
# C_D polar interpolation
# =============================================================================
# C_D = a(M) C_L^2 + b(M) C_L + c(M)
# =============================================================================

MACH_POLAR_DATA = np.array([0.65, 0.9, 1.1, 1.3, 2.0, 5.37, 7.38, 10.61])

A_POLAR_DATA = np.array([0.3804, 0.3418, 0.3459, 0.4006, 0.6049, 1.0314, 1.2753, 1.1948])
B_POLAR_DATA = np.array([-0.0011, 0.0100, 0.0012, 0.0037, 0.0010, 0.0145, 0.0354, 0.0962])
C_POLAR_DATA = np.array([0.0070, 0.0174, 0.0382, 0.0337, 0.0268, 0.0121, 0.0101, 0.0081])

A_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, A_POLAR_DATA)
B_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, B_POLAR_DATA)
C_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, C_POLAR_DATA)


def cd_from_mach_cl(M: float, CL: float, clamp_mach: bool = True) -> tuple[float, dict[str, float]]:
    M_original = float(M)

    if clamp_mach:
        M_used = float(np.clip(M_original, MACH_POLAR_DATA.min(), MACH_POLAR_DATA.max()))
    else:
        if M_original < MACH_POLAR_DATA.min() or M_original > MACH_POLAR_DATA.max():
            raise ValueError("Mach outside available C_D polar range.")
        M_used = M_original

    a = float(A_POLAR_INTERP(M_used))
    b = float(B_POLAR_INTERP(M_used))
    c = float(C_POLAR_INTERP(M_used))

    CD = a * CL**2 + b * CL + c

    return CD, {
        "a_polar": a,
        "b_polar": b,
        "c_polar": c,
        "mach_used_for_cd": M_used,
    }


# =============================================================================
# C_L-alpha interpolation
# =============================================================================
# C_L = m(M) alpha_deg + k(M)
# =============================================================================

MACH_CL_ALPHA_DATA = np.array([0.65, 0.9, 1.1, 1.3, 2.0, 5.37, 7.38, 10.61])

CL_ALPHA_SLOPE_DATA = np.array([
    0.0430,
    0.0457,
    0.0428,
    0.0372,
    0.0271,
    0.0167,
    0.0128,
    0.0110,
])

CL_ALPHA_INTERCEPT_DATA = np.array([
    -0.0347,
    -0.0381,
    -0.0235,
    -0.0084,
    0.0011,
    -0.0032,
    -0.0030,
    -0.0048,
])

CL_ALPHA_SLOPE_INTERP = PchipInterpolator(MACH_CL_ALPHA_DATA, CL_ALPHA_SLOPE_DATA)
CL_ALPHA_INTERCEPT_INTERP = PchipInterpolator(MACH_CL_ALPHA_DATA, CL_ALPHA_INTERCEPT_DATA)


def cl_from_mach_alpha(M: float, alpha_deg: float, clamp_mach: bool = True) -> tuple[float, dict[str, float]]:
    M_original = float(M)

    if clamp_mach:
        M_used = float(np.clip(M_original, MACH_CL_ALPHA_DATA.min(), MACH_CL_ALPHA_DATA.max()))
    else:
        if M_original < MACH_CL_ALPHA_DATA.min() or M_original > MACH_CL_ALPHA_DATA.max():
            raise ValueError("Mach outside available C_L-alpha range.")
        M_used = M_original

    slope = float(CL_ALPHA_SLOPE_INTERP(M_used))
    intercept = float(CL_ALPHA_INTERCEPT_INTERP(M_used))
    CL = slope * alpha_deg + intercept

    return CL, {
        "cl_alpha_slope_per_deg": slope,
        "cl_alpha_intercept": intercept,
        "mach_used_for_cl_alpha": M_used,
    }


# =============================================================================
# Drag calculation
# =============================================================================

def mach_regime(M: float) -> str:
    if M < 0.8:
        return "subsonic"
    elif M < 1.2:
        return "transonic"
    elif M < 5.0:
        return "supersonic"
    return "hypersonic"


def drag_from_mach_alpha(
    M: float,
    alpha_deg: float,
    altitude_m: float,
    S_ref: float,
    clamp_mach: bool = True,
) -> dict[str, float | str]:
    CL, cl_info = cl_from_mach_alpha(M, alpha_deg, clamp_mach=clamp_mach)
    CD, cd_info = cd_from_mach_cl(M, CL, clamp_mach=clamp_mach)
    q = dynamic_pressure_from_mach_altitude(M, altitude_m)
    D = q * S_ref * CD

    return {
        "M": M,
        "alpha_deg": alpha_deg,
        "altitude_m": altitude_m,
        "S_ref": S_ref,
        "q": q,
        "CL": CL,
        "CD": CD,
        "D": D,
        "regime": mach_regime(M),
        **cl_info,
        **cd_info,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_drag_vs_mach_alpha_sweep(
    altitude_m: float = 35_000.0,
    S_ref: float = 425.682,
    alpha_values_deg: list[float] | tuple[float, ...] = (-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0),
    mach_min: float = 0.65,
    mach_max: float = 10.61,
    n_mach: int = 400,
    save: bool = False,
) -> None:
    M_values = np.linspace(mach_min, mach_max, n_mach)

    fig, ax = plt.subplots(figsize=(10, 6))

    for alpha in alpha_values_deg:
        drag_values = []
        for M in M_values:
            out = drag_from_mach_alpha(
                M=M,
                alpha_deg=alpha,
                altitude_m=altitude_m,
                S_ref=S_ref,
                clamp_mach=True,
            )
            drag_values.append(out["D"] / 1000.0)

        ax.plot(M_values, drag_values, label=fr"$\alpha$ = {alpha:g}°")

    # Mark regime boundaries.
    for x, label in [(0.8, "sub/trans"), (1.2, "trans/super"), (5.0, "super/hyper")]:
        ax.axvline(x, linestyle="--", linewidth=1.0)
        ax.text(x, ax.get_ylim()[1] * 0.95, label, rotation=90, va="top", ha="right", fontsize=8)

    ax.scatter(MACH_POLAR_DATA, np.zeros_like(MACH_POLAR_DATA), marker="x", label="data Mach points")

    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("Drag [kN]")
    ax.set_title(f"Drag vs Mach using C_L-alpha schedule, h = {altitude_m/1000:.1f} km, S = {S_ref:.1f} m²")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2)
    fig.tight_layout()

    if save:
        fig.savefig("drag_vs_mach_alpha_sweep.png", dpi=300, bbox_inches="tight")

    plt.show()


def print_sample_table(
    altitude_m: float = 35_000.0,
    S_ref: float = 425.682,
    alpha_deg: float = 3.5,
) -> None:
    print("\nSample drag table")
    print("-----------------")
    print(f"Altitude: {altitude_m:.1f} m")
    print(f"S_ref:    {S_ref:.3f} m²")
    print(f"alpha:    {alpha_deg:.2f} deg")
    print()
    print(f"{'M':>6s} {'regime':>12s} {'q [Pa]':>12s} {'CL':>10s} {'CD':>10s} {'D [kN]':>12s}")

    for M in MACH_POLAR_DATA:
        out = drag_from_mach_alpha(M, alpha_deg, altitude_m, S_ref)
        print(
            f"{out['M']:6.2f} "
            f"{out['regime']:>12s} "
            f"{out['q']:12.1f} "
            f"{out['CL']:10.4f} "
            f"{out['CD']:10.5f} "
            f"{out['D'] / 1000.0:12.2f}"
        )

def run_drag_altitude_mach_map(
    mach_range=None,
    alt_range=None,
    alpha_deg=3.5,
    S_ref=600.0,
):
    """
    Returns:
        Drag_map [alt, mach]
        Drag_table (pandas)
    """

    if mach_range is None:
        mach_range = np.arange(5.0, 10.5, 0.5)

    if alt_range is None:
        alt_range = np.arange(25.0, 32.0, 1.0)

    DRAG_map = np.full((len(alt_range), len(mach_range)), np.nan)

    rows = []

    for i, h_km in enumerate(alt_range):
        h_m = h_km * 1000.0

        for j, M in enumerate(mach_range):

            try:
                out = drag_from_mach_alpha(
                    M=M,
                    alpha_deg=alpha_deg,
                    altitude_m=h_m,
                    S_ref=S_ref,
                    clamp_mach=True,
                )

                DRAG_map[i, j] = out["D"]

                rows.append({
                    "Altitude_km": h_km,
                    "Mach": M,
                    "Drag_N": out["D"],
                    "Drag_kN": out["D"] / 1000.0,
                    "CL": out["CL"],
                    "CD": out["CD"],
                    "q_Pa": out["q"],
                    "regime": out["regime"],
                })

                print(
                    f"h={h_km:.1f} km | M={M:.2f} | "
                    f"D={out['D']:.1f} N"
                )

            except Exception as e:
                print(f"FAILED h={h_km}, M={M}")
                print(e)
                DRAG_map[i, j] = np.nan

    import pandas as pd
    table = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Plot contour
    # -----------------------------------------------------------------------
    M_grid, H_grid = np.meshgrid(mach_range, alt_range)

    plt.figure(figsize=(10, 6))
    cont = plt.contourf(M_grid, H_grid, DRAG_map / 1000.0, levels=40)
    plt.colorbar(cont).set_label("Drag [kN]")
    plt.xlabel("Mach")
    plt.ylabel("Altitude [km]")
    plt.title(f"Drag Map (α = {alpha_deg}°)")
    plt.tight_layout()

    plt.show()

    return DRAG_map, table

def run_net_force_map(Scramjet_engine, alpha_deg=3.5, S_ref=600.0):

    # -----------------------------------------------------------------------
    # Get thrust map
    # -----------------------------------------------------------------------
    ISP_map, THRUST_map, table = run_altitude_mach_map(Scramjet_engine)

    mach_range = np.arange(5.0, 10.5, 0.5)
    alt_range  = np.arange(25.0, 32.0, 1.0)

    # -----------------------------------------------------------------------
    # Compute drag map
    # -----------------------------------------------------------------------
    DRAG_map, drag_table = run_drag_altitude_mach_map(
        mach_range=mach_range,
        alt_range=alt_range,
        alpha_deg=alpha_deg,
        S_ref=S_ref
    )

    # -----------------------------------------------------------------------
    # Net force map
    # -----------------------------------------------------------------------
    NET_map = THRUST_map - DRAG_map

    # -----------------------------------------------------------------------
    # Plot: Net force contour
    # -----------------------------------------------------------------------
    M_grid, H_grid = np.meshgrid(mach_range, alt_range)

    plt.figure(figsize=(10, 6))

    levels = 50
    cont = plt.contourf(M_grid, H_grid, NET_map / 1000.0, levels=levels, cmap="coolwarm")

    plt.colorbar(cont).set_label("Net Force [kN] (Thrust - Drag)")

    plt.contour(
        M_grid,
        H_grid,
        NET_map,
        levels=[0],
        colors="black",
        linewidths=2
    )

    plt.xlabel("Mach")
    plt.ylabel("Altitude [km]")
    plt.title(f"Net Propulsive Map (α = {alpha_deg}°)")

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------
    # Flight envelope mask
    # -----------------------------------------------------------------------
    feasible = NET_map > 0

    print("\nFlight envelope summary:")
    print(f"Max net thrust: {np.nanmax(NET_map):.1f} N")
    print(f"Min net thrust: {np.nanmin(NET_map):.1f} N")
    print(f"Feasible points (T > D): {np.sum(feasible)} / {feasible.size}")

    return {
        "thrust_map": THRUST_map,
        "drag_map": DRAG_map,
        "net_map": NET_map,
        "thrust_table": table,
        "drag_table": drag_table,
        "feasible_mask": feasible,
    }

if __name__ == "__main__":
    # Tweak these for your vehicle / mission point.
    altitude_m = 35_000.0
    S_ref = 425.682

    alpha_values_deg = [-1.0, 0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0]

    print_sample_table(
        altitude_m=altitude_m,
        S_ref=S_ref,
        alpha_deg=3.5,
    )

    plot_drag_vs_mach_alpha_sweep(
        altitude_m=altitude_m,
        S_ref=S_ref,
        alpha_values_deg=alpha_values_deg,
        mach_min=0.65,
        mach_max=10.61,
        n_mach=500,
        save=False,
    )

    drag_map, drag_table =run_drag_altitude_mach_map(
    mach_range=None,
    alt_range=None,
    alpha_deg=3.5,
    S_ref=400.0)

    Scramjet_engine = Scramjet_engine()

    results =run_net_force_map(Scramjet_engine, alpha_deg=3.5, S_ref=600.0)
