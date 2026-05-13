import numpy as np
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
# NOTE: Check your units. If EngineSim gave thrust in kN, then output is kN.
# If EngineSim gave N, output is N.
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


def thrust_from_poly(M: float, coeffs: tuple[float, float, float]) -> float:
    a, b, c = coeffs
    return a * M**2 + b * M + c


def thrust_from_mach_altitude(
    propulsion_type: str,
    M: float,
    altitude_m: float,
    clamp_altitude: bool = True,
) -> tuple[float, dict[str, float | str]]:
    """
    Interpolate thrust over altitude for a given propulsion type.

    Steps:
        1. Evaluate each altitude-specific thrust-Mach polynomial at Mach M.
        2. Interpolate those thrust values over altitude using PCHIP.

    propulsion_type:
        "turbo", "ram", or "scram"
    """

    propulsion_type = propulsion_type.lower()

    if propulsion_type not in THRUST_POLY_DATA:
        raise ValueError("propulsion_type must be 'turbo', 'ram', or 'scram'.")

    altitude_data = THRUST_POLY_DATA[propulsion_type]

    altitudes = np.array(sorted(altitude_data.keys()), dtype=float)

    if clamp_altitude:
        altitude_used = float(np.clip(altitude_m, altitudes.min(), altitudes.max()))
    else:
        if altitude_m < altitudes.min() or altitude_m > altitudes.max():
            raise ValueError(
                f"Altitude {altitude_m:.1f} m outside available range for {propulsion_type}: "
                f"{altitudes.min():.1f} to {altitudes.max():.1f} m."
            )
        altitude_used = float(altitude_m)

    thrust_values = np.array([
        thrust_from_poly(M, altitude_data[h])
        for h in altitudes
    ])

    thrust_alt_interp = PchipInterpolator(altitudes, thrust_values)
    thrust = float(thrust_alt_interp(altitude_used))

    return thrust, {
        "propulsion_type": propulsion_type,
        "M": M,
        "altitude_original_m": altitude_m,
        "altitude_used_m": altitude_used,
        "altitude_min_m": float(altitudes.min()),
        "altitude_max_m": float(altitudes.max()),
        "thrust": thrust,
    }


def thrust_curve_vs_mach(
    propulsion_type: str,
    altitude_m: float,
    mach_values: np.ndarray,
    clamp_altitude: bool = True,
) -> np.ndarray:
    thrust_values = []

    for M in mach_values:
        thrust, _ = thrust_from_mach_altitude(
            propulsion_type=propulsion_type,
            M=float(M),
            altitude_m=altitude_m,
            clamp_altitude=clamp_altitude,
        )
        thrust_values.append(thrust)

    return np.array(thrust_values)

T, info = thrust_from_mach_altitude(
    propulsion_type="ram",
    M=3.5,
    altitude_m=22000.0,
)

print(T)
print(info)

import matplotlib.pyplot as plt

M_values = np.linspace(0.0, 6.0, 300)

for h in [0, 5000, 10000, 15000]:
    T_values = thrust_curve_vs_mach("turbo", h, M_values)
    plt.plot(M_values, T_values, label=f"turbo, h={h/1000:.0f} km")

plt.xlabel("Mach number")
plt.ylabel("Thrust")
plt.grid(True)
plt.legend()
plt.show()