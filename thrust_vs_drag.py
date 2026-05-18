import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# =============================================================================
# SETTINGS
# =============================================================================
# Your thrust polynomial coefficients are around 1000. In your previous plots you
# treated these as kN-scale values, so the default here is "kN".
# Change to "N" only if your EngineSim polynomial coefficients are actually in N.
THRUST_OUTPUT_UNITS = "kN"  # "kN" or "N"


# =============================================================================
# THRUST-MACH POLYNOMIAL DATA
# =============================================================================
# T = a M^2 + b M + c
# Altitude in m.
# Thrust unit follows THRUST_OUTPUT_UNITS.
# =============================================================================

THRUST_POLY_DATA = {
    "turbo": {
        0.0:     (1252.7,  -861.57, 1272.0),
        5000.0:  (1091.0, -1214.8,  1003.1),
        10000.0: (787.87, -1088.9,  705.08),
        15000.0: (439.92, -742.89,  465.0),
    },
    "ram": {
        17000.0: (-76.19,   1328.6, -1238.1),
        20000.0: (-29.524,  658.57, -403.33),
        25000.0: (-19.048,  347.14, -275.95),
    },
    "scram": {
        25000.0: (271.43, -3190.0, 12604.0),
        30000.0: (128.57, -1530.0, 6055.7),
    },
}

# Suggested plotting ranges. These are not hard physical limits; they simply avoid
# plotting obviously unsuitable engine types over huge Mach ranges.
DEFAULT_MACH_RANGES = {
    "turbo": (0.0, 3.0),
    "ram": (2.0, 6.0),
    "scram": (5.0, 10.5),
}


# =============================================================================
# BUILD THRUST COEFFICIENT INTERPOLATORS / EXTRAPOLATORS
# =============================================================================

_COEFF_INTERP: dict[str, tuple] = {}

for _ptype, _alt_data in THRUST_POLY_DATA.items():
    _alts = np.array(sorted(_alt_data.keys()), dtype=float)

    _COEFF_INTERP[_ptype] = (
        PchipInterpolator(_alts, [_alt_data[h][0] for h in _alts], extrapolate=True),
        PchipInterpolator(_alts, [_alt_data[h][1] for h in _alts], extrapolate=True),
        PchipInterpolator(_alts, [_alt_data[h][2] for h in _alts], extrapolate=True),
        float(_alts.min()),
        float(_alts.max()),
    )


def thrust_curve_vs_mach(
    propulsion_type: str,
    altitude_m: float,
    mach_values: np.ndarray,
    extrapolate_altitude: bool = True,
    clamp_negative_thrust: bool = True,
) -> np.ndarray:
    """Return thrust over Mach. Units are controlled by THRUST_OUTPUT_UNITS."""
    ptype = propulsion_type.lower()

    if ptype not in _COEFF_INTERP:
        raise ValueError("propulsion_type must be 'turbo', 'ram', or 'scram'.")

    a_interp, b_interp, c_interp, alt_min, alt_max = _COEFF_INTERP[ptype]

    altitude_m = float(altitude_m)
    is_extrapolated = altitude_m < alt_min or altitude_m > alt_max

    if is_extrapolated and not extrapolate_altitude:
        raise ValueError(
            f"Altitude {altitude_m:.1f} m is outside the available range for {ptype}: "
            f"{alt_min:.1f} to {alt_max:.1f} m."
        )

    M = np.asarray(mach_values, dtype=float)

    a = float(a_interp(altitude_m))
    b = float(b_interp(altitude_m))
    c = float(c_interp(altitude_m))

    thrust = a * M**2 + b * M + c

    if clamp_negative_thrust:
        thrust = np.maximum(thrust, 0.0)

    return thrust


def thrust_to_newtons(thrust: np.ndarray) -> np.ndarray:
    """Convert thrust array to Newtons."""
    units = THRUST_OUTPUT_UNITS.lower()

    if units == "n":
        return thrust
    if units == "kn":
        return thrust * 1000.0

    raise ValueError("THRUST_OUTPUT_UNITS must be 'N' or 'kN'.")


# =============================================================================
# ISA ATMOSPHERE
# =============================================================================

def isa_temperature(altitude_m: float) -> float:
    if altitude_m <= 11_000.0:
        return 288.15 - 0.0065 * altitude_m
    if altitude_m <= 20_000.0:
        return 216.65
    if altitude_m <= 32_000.0:
        return 216.65 + 0.001 * (altitude_m - 20_000.0)
    return 228.65


def isa_pressure(altitude_m: float) -> float:
    g0 = 9.80665
    R = 287.05

    if altitude_m <= 11_000.0:
        T0 = 288.15
        L = -0.0065
        T = T0 + L * altitude_m
        return 101325.0 * (T / T0) ** (-g0 / (L * R))

    if altitude_m <= 20_000.0:
        T = 216.65
        p11 = 22632.06
        return p11 * math.exp(-g0 * (altitude_m - 11_000.0) / (R * T))

    if altitude_m <= 32_000.0:
        T20 = 216.65
        p20 = 5474.89
        L = 0.001
        T = T20 + L * (altitude_m - 20_000.0)
        return p20 * (T / T20) ** (-g0 / (L * R))

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
# AERODYNAMIC POLARS
# =============================================================================
# C_D = a(M) C_L^2 + b(M) C_L + c(M)
# C_L = m(M) alpha_deg + k(M)
# =============================================================================

MACH_POLAR_DATA = np.array([0.65, 0.9, 1.1, 1.3, 2.0, 5.37, 7.38, 10.61])

A_POLAR_DATA = np.array([0.3804, 0.3418, 0.3459, 0.4006, 0.6049, 1.0314, 1.2753, 1.1948])
B_POLAR_DATA = np.array([-0.0011, 0.0100, 0.0012, 0.0037, 0.0010, 0.0145, 0.0354, 0.0962])
C_POLAR_DATA = np.array([0.0070, 0.0174, 0.0382, 0.0337, 0.0268, 0.0121, 0.0101, 0.0081])

CL_ALPHA_SLOPE_DATA = np.array([0.0430, 0.0457, 0.0428, 0.0372, 0.0271, 0.0167, 0.0128, 0.0110])
CL_ALPHA_INTERCEPT_DATA = np.array([-0.0347, -0.0381, -0.0235, -0.0084, 0.0011, -0.0032, -0.0030, -0.0048])

A_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, A_POLAR_DATA, extrapolate=False)
B_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, B_POLAR_DATA, extrapolate=False)
C_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, C_POLAR_DATA, extrapolate=False)
CL_ALPHA_SLOPE_INTERP = PchipInterpolator(MACH_POLAR_DATA, CL_ALPHA_SLOPE_DATA, extrapolate=False)
CL_ALPHA_INTERCEPT_INTERP = PchipInterpolator(MACH_POLAR_DATA, CL_ALPHA_INTERCEPT_DATA, extrapolate=False)


def cl_from_mach_alpha(M: float, alpha_deg: float, clamp_mach: bool = True) -> float:
    M_original = float(M)

    if clamp_mach:
        M_used = float(np.clip(M_original, MACH_POLAR_DATA.min(), MACH_POLAR_DATA.max()))
    else:
        if M_original < MACH_POLAR_DATA.min() or M_original > MACH_POLAR_DATA.max():
            raise ValueError("Mach outside available C_L-alpha range.")
        M_used = M_original

    slope = float(CL_ALPHA_SLOPE_INTERP(M_used))
    intercept = float(CL_ALPHA_INTERCEPT_INTERP(M_used))
    return slope * alpha_deg + intercept


def cd_from_mach_cl(M: float, CL: float, clamp_mach: bool = True) -> float:
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
    return a * CL**2 + b * CL + c


def drag_curve_vs_mach(
    altitude_m: float,
    mach_values: np.ndarray,
    alpha_deg: float,
    S_ref: float,
    clamp_mach: bool = True,
) -> np.ndarray:
    """Return drag in Newtons over Mach."""
    drag = []

    for M in mach_values:
        CL = cl_from_mach_alpha(M, alpha_deg, clamp_mach=clamp_mach)
        CD = cd_from_mach_cl(M, CL, clamp_mach=clamp_mach)
        q = dynamic_pressure_from_mach_altitude(M, altitude_m)
        drag.append(q * S_ref * CD)

    return np.array(drag)


# =============================================================================
# COMBINED PLOT
# =============================================================================

def plot_thrust_drag_same_altitude(
    propulsion_type: str = "scram",
    altitude_m: float = 30_000.0,
    alpha_deg: float = 3.5,
    S_ref: float = 600.0,
    mach_min: float | None = None,
    mach_max: float | None = None,
    n_mach: int = 400,
    extrapolate_altitude: bool = True,
    clamp_negative_thrust: bool = True,
    save: bool = False,
    filename: str = "thrust_drag_same_altitude.png",
) -> dict[str, np.ndarray]:
    """
    Plot thrust, drag, and net force against Mach at one altitude.

    All returned force arrays are in Newtons.
    The plotted force unit is kN.
    """
    ptype = propulsion_type.lower()

    if ptype not in DEFAULT_MACH_RANGES:
        raise ValueError("propulsion_type must be 'turbo', 'ram', or 'scram'.")

    if mach_min is None or mach_max is None:
        default_min, default_max = DEFAULT_MACH_RANGES[ptype]
        if mach_min is None:
            mach_min = default_min
        if mach_max is None:
            mach_max = default_max

    M_values = np.linspace(mach_min, mach_max, n_mach)

    thrust_model_units = thrust_curve_vs_mach(
        propulsion_type=ptype,
        altitude_m=altitude_m,
        mach_values=M_values,
        extrapolate_altitude=extrapolate_altitude,
        clamp_negative_thrust=clamp_negative_thrust,
    )
    thrust_N = thrust_to_newtons(thrust_model_units)

    drag_N = drag_curve_vs_mach(
        altitude_m=altitude_m,
        mach_values=M_values,
        alpha_deg=alpha_deg,
        S_ref=S_ref,
        clamp_mach=True,
    )

    net_N = thrust_N - drag_N

    plt.figure(figsize=(10, 6))
    plt.plot(M_values, thrust_N / 1000.0, label="Thrust")
    plt.plot(M_values, drag_N / 1000.0, label="Drag")
    plt.plot(M_values, net_N / 1000.0, linestyle="--", label="Net force = Thrust - Drag")
    plt.axhline(0.0, linewidth=1.0)

    plt.xlabel("Mach number [-]")
    plt.ylabel("Force [kN]")
    plt.title(
        f"{ptype.capitalize()} thrust, drag, and net force\n"
        f"h = {altitude_m / 1000.0:.1f} km, alpha = {alpha_deg:.1f} deg, S_ref = {S_ref:.1f} m²"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()

    print("\nSummary")
    print("-------")
    print(f"Propulsion type: {ptype}")
    print(f"Altitude:        {altitude_m / 1000.0:.2f} km")
    print(f"Mach range:      {mach_min:.2f} to {mach_max:.2f}")
    print(f"Thrust units in polynomial: {THRUST_OUTPUT_UNITS}")
    print(f"Max thrust:      {np.nanmax(thrust_N) / 1000.0:.2f} kN")
    print(f"Max drag:        {np.nanmax(drag_N) / 1000.0:.2f} kN")
    print(f"Max net force:   {np.nanmax(net_N) / 1000.0:.2f} kN")
    print(f"Min net force:   {np.nanmin(net_N) / 1000.0:.2f} kN")

    return {
        "Mach": M_values,
        "Thrust_N": thrust_N,
        "Drag_N": drag_N,
        "Net_N": net_N,
    }


# =============================================================================
# EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # Pick ONE case at a time.

    # Turbo example: same altitude, realistic-ish turbo Mach plotting range.
    results = plot_thrust_drag_same_altitude(
        propulsion_type="turbo",
        altitude_m=15_000.0,
        alpha_deg=3.5,
        S_ref=400.0,
        mach_min=0.65,
        mach_max=3.0,
        n_mach=400,
        save=False,
    )

    # Ramjet example:
    # results = plot_thrust_drag_same_altitude(
    #     propulsion_type="ram",
    #     altitude_m=25_000.0,
    #     alpha_deg=3.5,
    #     S_ref=400.0,
    #     mach_min=2.0,
    #     mach_max=6.0,
    #     n_mach=400,
    #     save=False,
    # )

    # Scramjet example:
    # results = plot_thrust_drag_same_altitude(
    #     propulsion_type="scram",
    #     altitude_m=30_000.0,
    #     alpha_deg=3.5,
    #     S_ref=400.0,
    #     mach_min=5.0,
    #     mach_max=10.5,
    #     n_mach=400,
    #     save=False,
    # )
