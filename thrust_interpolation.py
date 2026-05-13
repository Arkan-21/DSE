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


# Pre-build one PCHIP interpolant per coefficient (a, b, c) per engine type.
# This avoids rebuilding interpolators on every thrust query.
_COEFF_INTERP: dict[str, tuple] = {}
for _ptype, _alt_data in THRUST_POLY_DATA.items():
    _alts = np.array(sorted(_alt_data.keys()), dtype=float)
    _COEFF_INTERP[_ptype] = (
        PchipInterpolator(_alts, [_alt_data[h][0] for h in _alts]),  # a(alt)
        PchipInterpolator(_alts, [_alt_data[h][1] for h in _alts]),  # b(alt)
        PchipInterpolator(_alts, [_alt_data[h][2] for h in _alts]),  # c(alt)
        float(_alts.min()),
        float(_alts.max()),
    )
del _ptype, _alt_data, _alts


def thrust_from_mach_altitude(
    propulsion_type: str,
    M: float,
    altitude_m: float,
    clamp_altitude: bool = True,
) -> tuple[float, dict[str, float | str]]:
    ptype = propulsion_type.lower()
    if ptype not in _COEFF_INTERP:
        raise ValueError("propulsion_type must be 'turbo', 'ram', or 'scram'.")

    a_interp, b_interp, c_interp, alt_min, alt_max = _COEFF_INTERP[ptype]

    if clamp_altitude:
        altitude_used = float(np.clip(altitude_m, alt_min, alt_max))
    else:
        if altitude_m < alt_min or altitude_m > alt_max:
            raise ValueError(
                f"Altitude {altitude_m:.1f} m outside available range for {ptype}: "
                f"{alt_min:.1f} to {alt_max:.1f} m."
            )
        altitude_used = float(altitude_m)

    a = float(a_interp(altitude_used))
    b = float(b_interp(altitude_used))
    c = float(c_interp(altitude_used))
    thrust = a * M**2 + b * M + c

    return thrust, {
        "propulsion_type": ptype,
        "M": M,
        "altitude_original_m": altitude_m,
        "altitude_used_m": altitude_used,
        "altitude_min_m": alt_min,
        "altitude_max_m": alt_max,
        "thrust": thrust,
    }


def thrust_curve_vs_mach(
    propulsion_type: str,
    altitude_m: float,
    mach_values: np.ndarray,
    clamp_altitude: bool = True,
) -> np.ndarray:
    ptype = propulsion_type.lower()
    a_interp, b_interp, c_interp, alt_min, alt_max = _COEFF_INTERP[ptype]
    altitude_used = float(np.clip(altitude_m, alt_min, alt_max)) if clamp_altitude else float(altitude_m)
    a = float(a_interp(altitude_used))
    b = float(b_interp(altitude_used))
    c = float(c_interp(altitude_used))
    M = np.asarray(mach_values, dtype=float)
    return a * M**2 + b * M + c

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