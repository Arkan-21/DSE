import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# =============================================================================
# Thrust-Mach polynomial data
# =============================================================================
# Form:
#     T = a M^2 + b M + c
#
# x = Mach number
# y = thrust from your EngineSim curves
#
# NOTE:
# If EngineSim gave thrust in kN, output is kN.
# If EngineSim gave thrust in N, output is N.
# =============================================================================

THRUST_POLY_DATA = {
    "turbo": {
        # altitude_m: (a, b, c)
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


# =============================================================================
# Helper functions
# =============================================================================

def thrust_from_poly(M: float | np.ndarray, coeffs: tuple[float, float, float]) -> float | np.ndarray:
    """
    Calculate thrust from a quadratic thrust-Mach polynomial.

    T = a M^2 + b M + c
    """
    a, b, c = coeffs
    M = np.asarray(M, dtype=float)
    return a * M**2 + b * M + c


# =============================================================================
# Build altitude extrapolators
# =============================================================================
# One PCHIP extrapolator is built for each coefficient: a(h), b(h), c(h).
# extrapolate=True allows altitudes outside the original data range.
# =============================================================================

_COEFF_INTERP: dict[str, tuple] = {}

for _ptype, _alt_data in THRUST_POLY_DATA.items():
    _alts = np.array(sorted(_alt_data.keys()), dtype=float)

    _COEFF_INTERP[_ptype] = (
        PchipInterpolator(
            _alts,
            [_alt_data[h][0] for h in _alts],
            extrapolate=True,
        ),  # a(altitude)

        PchipInterpolator(
            _alts,
            [_alt_data[h][1] for h in _alts],
            extrapolate=True,
        ),  # b(altitude)

        PchipInterpolator(
            _alts,
            [_alt_data[h][2] for h in _alts],
            extrapolate=True,
        ),  # c(altitude)

        float(_alts.min()),
        float(_alts.max()),
    )

del _ptype, _alt_data, _alts


# =============================================================================
# Main thrust function
# =============================================================================

def thrust_from_mach_altitude(
    propulsion_type: str,
    M: float,
    altitude_m: float,
    extrapolate_altitude: bool = True,
    clamp_negative_thrust: bool = True,
) -> tuple[float, dict[str, float | str | bool]]:
    """
    Calculate thrust for a given propulsion type, Mach number, and altitude.

    Parameters
    ----------
    propulsion_type : str
        One of: "turbo", "ram", or "scram".

    M : float
        Mach number.

    altitude_m : float
        Altitude in meters.

    extrapolate_altitude : bool
        If True, allow extrapolation outside the available altitude range.
        If False, raise an error outside the data range.

    clamp_negative_thrust : bool
        If True, negative thrust values are replaced by 0.

    Returns
    -------
    thrust : float
        Calculated thrust.

    info : dict
        Useful diagnostic information.
    """
    ptype = propulsion_type.lower()

    if ptype not in _COEFF_INTERP:
        raise ValueError("propulsion_type must be 'turbo', 'ram', or 'scram'.")

    a_interp, b_interp, c_interp, alt_min, alt_max = _COEFF_INTERP[ptype]

    M = float(M)
    altitude_m = float(altitude_m)

    is_extrapolated = altitude_m < alt_min or altitude_m > alt_max

    if is_extrapolated and not extrapolate_altitude:
        raise ValueError(
            f"Altitude {altitude_m:.1f} m is outside the available range for {ptype}: "
            f"{alt_min:.1f} to {alt_max:.1f} m."
        )

    a = float(a_interp(altitude_m))
    b = float(b_interp(altitude_m))
    c = float(c_interp(altitude_m))

    thrust_raw = a * M**2 + b * M + c

    if clamp_negative_thrust:
        thrust = max(thrust_raw, 0.0)
    else:
        thrust = thrust_raw

    info = {
        "propulsion_type": ptype,
        "M": M,
        "altitude_original_m": altitude_m,
        "altitude_used_m": altitude_m,
        "altitude_min_m": alt_min,
        "altitude_max_m": alt_max,
        "is_extrapolated": is_extrapolated,
        "a": a,
        "b": b,
        "c": c,
        "thrust_raw": thrust_raw,
        "thrust": thrust,
    }

    return thrust, info


# =============================================================================
# Thrust curve function
# =============================================================================

def thrust_curve_vs_mach(
    propulsion_type: str,
    altitude_m: float,
    mach_values: np.ndarray,
    extrapolate_altitude: bool = True,
    clamp_negative_thrust: bool = True,
) -> np.ndarray:
    """
    Calculate a thrust curve over a range of Mach numbers
    for one propulsion type and one altitude.
    """
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

    a = float(a_interp(altitude_m))
    b = float(b_interp(altitude_m))
    c = float(c_interp(altitude_m))

    M = np.asarray(mach_values, dtype=float)

    thrust_raw = a * M**2 + b * M + c

    if clamp_negative_thrust:
        thrust = np.maximum(thrust_raw, 0.0)
    else:
        thrust = thrust_raw

    return thrust


# =============================================================================
# Optional: get coefficients at any altitude
# =============================================================================

def coefficients_at_altitude(
    propulsion_type: str,
    altitude_m: float,
    extrapolate_altitude: bool = True,
) -> tuple[float, float, float]:
    """
    Return interpolated or extrapolated polynomial coefficients
    a, b, c at a given altitude.
    """
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

    a = float(a_interp(altitude_m))
    b = float(b_interp(altitude_m))
    c = float(c_interp(altitude_m))

    return a, b, c


# =============================================================================
# Example single-point thrust calculation
# =============================================================================

T, info = thrust_from_mach_altitude(
    propulsion_type="turbo",
    M=2.4,
    altitude_m=15000.0,
    extrapolate_altitude=True,
    clamp_negative_thrust=True,
)

print("Single-point thrust calculation")
print("--------------------------------")
print(f"Thrust = {T:.3f}")
print(info)


# =============================================================================
# Example plot: turbo thrust curves from 0 to 30 km
# =============================================================================

M_values = np.linspace(0.0, 6.0, 300)

plt.figure()

for h in [0, 5000, 10000, 15000, 20000, 25000, 30000]:
    T_values = thrust_curve_vs_mach(
        propulsion_type="turbo",
        altitude_m=h,
        mach_values=M_values,
        extrapolate_altitude=True,
        clamp_negative_thrust=True,
    )

    plt.plot(M_values, T_values, label=f"turbo, h={h / 1000:.0f} km")

plt.xlabel("Mach number")
plt.ylabel("Thrust")
plt.title("Turbo thrust curves with altitude extrapolation")
plt.grid(True)
plt.legend()
plt.show()


# =============================================================================
# Example plot: all propulsion types over useful altitude ranges
# =============================================================================

plt.figure()

plot_cases = [
    ("turbo", 0),
    ("turbo", 15000),
    ("turbo", 30000),

    ("ram", 17000),
    ("ram", 25000),
    ("ram", 30000),

    ("scram", 25000),
    ("scram", 30000),
]

for ptype, h in plot_cases:
    T_values = thrust_curve_vs_mach(
        propulsion_type=ptype,
        altitude_m=h,
        mach_values=M_values,
        extrapolate_altitude=True,
        clamp_negative_thrust=True,
    )

    plt.plot(M_values, T_values, label=f"{ptype}, h={h / 1000:.0f} km")

plt.xlabel("Mach number")
plt.ylabel("Thrust")
plt.title("Thrust curves with interpolation/extrapolation")
plt.grid(True)
plt.legend()
plt.show()