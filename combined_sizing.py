import math
import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize


# =============================================================================
# Basic geometry and sizing equations
# =============================================================================

def s_wet(K_W: float, S_plan: float) -> float:
    return K_W * S_plan


def volume_from_tau(tau: float, S_plan: float) -> float:
    return tau * S_plan**1.5


def W_landinggear(W_to: float) -> float:
    return 0.01 * W_to**1.124


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


def mach_to_velocity(mach: float, altitude_m: float) -> float:
    return mach * speed_of_sound(altitude_m)


def dynamic_pressure_from_mach_altitude(mach: float, altitude_m: float) -> float:
    if mach <= 0:
        raise ValueError("mach must be positive for dynamic pressure calculation.")

    rho = isa_density(altitude_m)
    V = mach_to_velocity(mach, altitude_m)
    return 0.5 * rho * V**2


# =============================================================================
# C_D polar interpolation
# =============================================================================

MACH_POLAR_DATA = np.array([0.65, 0.9, 1.1, 1.3, 2.0, 5.37, 7.38, 10.61])

A_POLAR_DATA = np.array([0.3804, 0.3418, 0.3459, 0.4006, 0.6049, 1.0314, 1.2753, 1.1948])
B_POLAR_DATA = np.array([-0.0011, 0.0100, 0.0012, 0.0037, 0.0010, 0.0145, 0.0354, 0.0962])
C_POLAR_DATA = np.array([0.0070, 0.0174, 0.0382, 0.0337, 0.0268, 0.0121, 0.0101, 0.0081])

A_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, A_POLAR_DATA, extrapolate=True)
B_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, B_POLAR_DATA, extrapolate=True)
C_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, C_POLAR_DATA, extrapolate=True)


def interpolation_status(x: float, data: np.ndarray) -> str:
    x_min = float(np.min(data))
    x_max = float(np.max(data))

    if x < x_min:
        return "extrapolated_low"
    if x > x_max:
        return "extrapolated_high"
    return "interpolated"


def mach_regime(M: float) -> str:
    if M < 0.8:
        return "subsonic"
    elif M < 1.2:
        return "transonic"
    elif M < 5.0:
        return "supersonic"
    return "hypersonic"


# =============================================================================
# C_L-alpha interpolation
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

CL_ALPHA_SLOPE_INTERP = PchipInterpolator(
    MACH_CL_ALPHA_DATA,
    CL_ALPHA_SLOPE_DATA,
    extrapolate=True,
)

CL_ALPHA_INTERCEPT_INTERP = PchipInterpolator(
    MACH_CL_ALPHA_DATA,
    CL_ALPHA_INTERCEPT_DATA,
    extrapolate=True,
)


def cl_from_mach_alpha(
    M: float,
    alpha_deg: float,
    clamp_mach: bool = False,
) -> tuple[float, dict[str, float | str]]:
    M_original = float(M)

    if clamp_mach:
        M_used = float(np.clip(
            M_original,
            MACH_CL_ALPHA_DATA.min(),
            MACH_CL_ALPHA_DATA.max(),
        ))
    else:
        M_used = M_original

    slope = float(CL_ALPHA_SLOPE_INTERP(M_used))
    intercept = float(CL_ALPHA_INTERCEPT_INTERP(M_used))
    CL = slope * alpha_deg + intercept

    return CL, {
        "alpha_deg": alpha_deg,
        "cl_alpha_slope_per_deg": slope,
        "cl_alpha_intercept": intercept,
        "mach_original_cl_alpha": M_original,
        "mach_used_for_cl_alpha": M_used,
        "mach_cl_alpha_status": interpolation_status(M_original, MACH_CL_ALPHA_DATA),
        "mach_regime_cl_alpha": mach_regime(M_original),
    }


def alpha_required_from_mach_cl(
    M: float,
    CL_required: float,
    clamp_mach: bool = False,
) -> tuple[float, dict[str, float | str]]:
    M_original = float(M)

    if clamp_mach:
        M_used = float(np.clip(
            M_original,
            MACH_CL_ALPHA_DATA.min(),
            MACH_CL_ALPHA_DATA.max(),
        ))
    else:
        M_used = M_original

    slope = float(CL_ALPHA_SLOPE_INTERP(M_used))
    intercept = float(CL_ALPHA_INTERCEPT_INTERP(M_used))

    if abs(slope) < 1e-12:
        raise ValueError(f"C_L-alpha slope is too close to zero at Mach {M_used:.3f}.")

    alpha_required_deg = (CL_required - intercept) / slope

    return alpha_required_deg, {
        "alpha_required_deg": alpha_required_deg,
        "cl_alpha_slope_per_deg_required": slope,
        "cl_alpha_intercept_required": intercept,
        "mach_used_for_alpha_required": M_used,
        "mach_alpha_required_status": interpolation_status(M_original, MACH_CL_ALPHA_DATA),
    }


def cd_from_mach_cl(
    M: float,
    CL: float,
    clamp_mach: bool = False,
) -> tuple[float, dict[str, float | str]]:
    M_original = float(M)

    if clamp_mach:
        M_used = float(np.clip(
            M_original,
            MACH_POLAR_DATA.min(),
            MACH_POLAR_DATA.max(),
        ))
    else:
        M_used = M_original

    a = float(A_POLAR_INTERP(M_used))
    b = float(B_POLAR_INTERP(M_used))
    c = float(C_POLAR_INTERP(M_used))

    CD = a * CL**2 + b * CL + c

    return CD, {
        "a_polar": a,
        "b_polar": b,
        "c_polar": c,
        "mach_original": M_original,
        "mach_used_for_polar": M_used,
        "mach_polar_status": interpolation_status(M_original, MACH_POLAR_DATA),
        "mach_regime": mach_regime(M_original),
    }


# =============================================================================
# EngineSim thrust-Mach polynomial interpolation
# =============================================================================

THRUST_CURVE_TO_N = 1000.0

ENGINE_COUNT_BY_PROPULSION = {
    "turbo": 2,
    "ram": 2,
    "scram": 2,
}

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


def thrust_curve_value_from_poly(M: float, coeffs: tuple[float, float, float]) -> float:
    a, b, c = coeffs
    return a * M**2 + b * M + c


def thrust_from_mach_altitude(
    propulsion_type: str,
    M: float,
    altitude_m: float,
    clamp_altitude: bool = False,
    engine_count: int | None = None,
    minimum_thrust_N: float = 0.0,
) -> tuple[float, dict[str, float | str]]:
    propulsion_type = propulsion_type.lower()

    if propulsion_type not in THRUST_POLY_DATA:
        raise ValueError("propulsion_type must be 'turbo', 'ram', or 'scram'.")

    if engine_count is None:
        engine_count = ENGINE_COUNT_BY_PROPULSION[propulsion_type]

    altitude_data = THRUST_POLY_DATA[propulsion_type]
    altitudes = np.array(sorted(altitude_data.keys()), dtype=float)

    if clamp_altitude:
        altitude_used = float(np.clip(altitude_m, altitudes.min(), altitudes.max()))
    else:
        altitude_used = float(altitude_m)

    thrust_curve_values = np.array([
        thrust_curve_value_from_poly(M, altitude_data[h])
        for h in altitudes
    ])

    thrust_alt_interp = PchipInterpolator(
        altitudes,
        thrust_curve_values,
        extrapolate=True,
    )
    thrust_curve_value = float(thrust_alt_interp(altitude_used))

    thrust_per_engine_N = thrust_curve_value * THRUST_CURVE_TO_N
    thrust_total_N_raw = thrust_per_engine_N * engine_count
    thrust_total_N = max(minimum_thrust_N, thrust_total_N_raw)

    return thrust_total_N, {
        "propulsion_type": propulsion_type,
        "M_thrust": M,
        "altitude_original_m": altitude_m,
        "altitude_used_m": altitude_used,
        "altitude_min_m": float(altitudes.min()),
        "altitude_max_m": float(altitudes.max()),
        "thrust_curve_value_per_engine": thrust_curve_value,
        "thrust_per_engine_N": thrust_per_engine_N,
        "engine_count": engine_count,
        "thrust_total_N_raw": thrust_total_N_raw,
        "thrust_total_N": thrust_total_N,
        "altitude_interpolation_status": interpolation_status(altitude_m, altitudes),
    }


# =============================================================================
# K_W as function of tau and configuration
# =============================================================================

def k_w_from_tau(tau: float, configuration: str = "wing_body") -> float:
    if tau <= 0:
        raise ValueError("tau must be positive.")

    if configuration == "waverider":
        return 1053.3 * tau**4- 863.55* tau**3 + 243.06* tau**2 - 22.026 * tau + 3.3655
    elif configuration == "wing_body":
        return 1604.6 * tau**4 - 936.96 * tau**3 + 203.15 * tau**2 - 15.014* tau + 2.9932
    elif configuration == "blended_body":
        return -326 * tau**4 + 189.99* tau**3 -22.369* tau**2 + 3.7082* tau+ 2.3081
    else:
        raise ValueError("configuration must be 'waverider', 'wing_body', or 'blended_body'.")


# =============================================================================
# Mission segment model
# =============================================================================

@dataclass
class MissionSegment:
    name: str
    mode: str
    fuel_type: str = "none"
    propulsion_mode: str = "none"

    delta_h: float = 0.0
    V_initial: float = 0.0
    V_final: float = 0.0
    V_average: float = 0.0
    g: float = 9.81
    I_sp: float = 0.0

    mach_drag: float = 0.0
    altitude_drag: float = 0.0
    flight_path_angle_deg: float = 0.0
    n_normal_extra: float = 0.0

    T: float = 0.0
    delta_t: float = 0.0
    fixed_fraction: float = 1.0

    max_total_accel_g: float = 0.20

    # Optional overrides from the optimized transition-profile code.
    # When these are provided, the combined sizing loop uses the same
    # drag/thrust basis as the mission-profile optimizer instead of the
    # older local polar model.
    drag_override_N: float | None = None
    q_override_Pa: float | None = None
    CL_override: float | None = None
    CD_override: float | None = None
    LD_override: float | None = None
    thrust_req_override_N: float | None = None


def segment_lift_required(segment: MissionSegment, W_current: float) -> float:
    gamma_rad = math.radians(segment.flight_path_angle_deg)
    W_current_force = W_current * segment.g
    return W_current_force * (math.cos(gamma_rad) + segment.n_normal_extra)


def scheduled_alpha_for_segment(segment: MissionSegment) -> float:
    name = segment.name.lower()
    M = segment.mach_drag

    if "takeoff" in name:
        return 12.0
    if "landing" in name:
        return 10.5
    if "descent" in name:
        if M >= 5.0:
            return 2.0
        if M >= 2.0:
            return 2.5
        return 4.0
    if "cruise" in name:
        if M >= 5.0:
            return 3.5
        return 3.0
    if "horizontal_accel" in name or "accel" in name:
        if M >= 5.0:
            return 3.0
        if M >= 2.0:
            return 3.5
        return 4.0
    if "ascent" in name or "climb" in name:
        if M < 0.8:
            return 7.5
        if M < 1.2:
            return 6.0
        if M < 2.5:
            return 4.5
        if M < 4.0:
            return 4.0
        return 3.2

    if M < 0.8:
        return 7.0
    if M < 1.2:
        return 5.5
    if M < 2.5:
        return 4.5
    if M < 5.0:
        return 3.5
    return 3.0


def scheduled_cl_for_segment(segment: MissionSegment) -> tuple[float, dict[str, float | str]]:
    alpha_deg = scheduled_alpha_for_segment(segment)
    return cl_from_mach_alpha(segment.mach_drag, alpha_deg, clamp_mach=False)


def segment_aero_from_polar(
    segment: MissionSegment,
    W_current: float,
    S_plan: float,
) -> dict[str, Any]:
    if S_plan <= 0:
        raise ValueError("S_plan must be positive.")
    if segment.mach_drag <= 0.0:
        raise ValueError(f"Segment '{segment.name}' needs mach_drag > 0 for drag calculation.")

    # If this segment was imported from the optimized transition-profile code,
    # use that drag model directly. This keeps the combined sizing consistent
    # with the mission profile that already sized turbojet/ramjet thrust to
    # overcome drag + acceleration.
    if segment.drag_override_N is not None:
        q = (segment.q_override_Pa if segment.q_override_Pa is not None
             else dynamic_pressure_from_mach_altitude(segment.mach_drag, segment.altitude_drag))
        CL_used = segment.CL_override if segment.CL_override is not None else 0.0
        CD_used = segment.CD_override if segment.CD_override is not None else 0.0
        LD_used = segment.LD_override if segment.LD_override is not None else (CL_used / CD_used if CD_used > 0 else 0.0)

        return {
            "D": float(segment.drag_override_N),
            "L_required": segment_lift_required(segment, W_current),
            "q": float(q),
            "C_L_calc": float(CL_used),
            "C_L_used": float(CL_used),
            "C_L_force_balance": float(CL_used),
            "C_D_calc": float(CD_used),
            "L_over_D_calc": float(LD_used),
            "C_L_scheduled": float(CL_used),
            "lift_balance_ratio": 1.0,
            "force_balance_warning": False,
            "alpha_required_deg": 0.0,
            "alpha_deg": 0.0,
            "a_polar": 0.0,
            "b_polar": 0.0,
            "c_polar": 0.0,
            "mach_original": segment.mach_drag,
            "mach_used_for_polar": segment.mach_drag,
            "mach_polar_status": "imported_profile_drag",
            "mach_regime": mach_regime(segment.mach_drag),
            "thrust_req_imported_N": segment.thrust_req_override_N if segment.thrust_req_override_N is not None else 0.0,
        }

    q = dynamic_pressure_from_mach_altitude(segment.mach_drag, segment.altitude_drag)

    if q <= 0:
        raise ValueError(f"Segment '{segment.name}' dynamic pressure is non-positive.")

    L_required = segment_lift_required(segment, W_current)
    C_L_required = L_required / (q * S_plan)

    # Diagnostic only: what your scheduled alpha would have produced
    C_L_scheduled, cl_alpha_info = scheduled_cl_for_segment(segment)

    # Actual model fix: use the force-balance CL for drag
    C_L_used = C_L_required

    C_D_calc, polar_info = cd_from_mach_cl(
        segment.mach_drag,
        C_L_used,
        clamp_mach=False,
    )

    D_calc = q * S_plan * C_D_calc

    alpha_required_deg, alpha_required_info = alpha_required_from_mach_cl(
        segment.mach_drag,
        C_L_required,
        clamp_mach=False,
    )

    lift_balance_ratio = C_L_scheduled / C_L_required if C_L_required > 0 else 0.0

    return {
        "D": D_calc,
        "L_required": L_required,
        "q": q,

        # Actual coefficients used in the sizing loop
        "C_L_calc": C_L_used,
        "C_L_used": C_L_used,
        "C_L_force_balance": C_L_required,
        "C_D_calc": C_D_calc,
        "L_over_D_calc": C_L_used / C_D_calc if C_D_calc > 0 else 0.0,

        # Diagnostics
        "C_L_scheduled": C_L_scheduled,
        "lift_balance_ratio": lift_balance_ratio,
        "force_balance_warning": abs(C_L_used - C_L_required) / C_L_required > 0.05
        if C_L_required > 0 else False,

        **cl_alpha_info,
        **alpha_required_info,
        **polar_info,
    }


def segment_weight_fraction(segment: MissionSegment, W_current: float, S_plan: float) -> tuple[float, float, dict[str, Any]]:
    aero_info = segment_aero_from_polar(segment, W_current, S_plan)
    D_used = aero_info["D"]

    if segment.mode == "fixed":
        fraction = segment.fixed_fraction

        gamma_rad = math.radians(segment.flight_path_angle_deg)
        a_total_max = segment.max_total_accel_g * segment.g
        vertical_accel_m_s2 = segment.g * math.sin(gamma_rad)

        if abs(vertical_accel_m_s2) >= a_total_max:
            a_x_allow = 0.0
        else:
            a_x_allow = math.sqrt(a_total_max**2 - vertical_accel_m_s2**2)

        if segment.fuel_type == "none" or segment.propulsion_mode == "none":
            T_used = 0.0
        else:
            T_used = min(segment.T, D_used + W_current * a_x_allow)

        axial_accel_m_s2 = max(0.0, (T_used - D_used) / W_current)
        total_accel_m_s2 = math.sqrt(axial_accel_m_s2**2 + vertical_accel_m_s2**2)

        aero_info["T_available"] = segment.T
        aero_info["T_used"] = T_used
        aero_info["T_accel_limited"] = D_used + W_current * a_x_allow
        aero_info["axial_accel_m_s2"] = axial_accel_m_s2
        aero_info["vertical_accel_m_s2"] = vertical_accel_m_s2
        aero_info["total_accel_m_s2"] = total_accel_m_s2
        aero_info["axial_accel_g"] = axial_accel_m_s2 / segment.g
        aero_info["vertical_accel_g"] = vertical_accel_m_s2 / segment.g
        aero_info["total_accel_g"] = total_accel_m_s2 / segment.g
        aero_info["a_x_allow_m_s2"] = a_x_allow

    elif segment.mode == "T_gt_D":
        if segment.I_sp <= 0:
            raise ValueError(f"For segment '{segment.name}', I_sp must be positive.")
        if segment.V_average <= 0:
            raise ValueError(f"For segment '{segment.name}', V_average must be positive.")
        if W_current <= 0:
            raise ValueError(f"For segment '{segment.name}', W_current must be positive.")

        T_available = segment.T
        gamma_rad = math.radians(segment.flight_path_angle_deg)

        a_total_max = segment.max_total_accel_g * segment.g
        a_y = segment.g * math.sin(gamma_rad)

        if abs(a_y) >= a_total_max:
            a_x_allow = 0.0
        else:
            a_x_allow = math.sqrt(a_total_max**2 - a_y**2)

        # Imported transition-profile segments already include the acceleration
        # requirement in thrust_req_override_N:
        #     T_req = D + W_ref * 0.15 g
        # Use that imported requirement as the thrust cap for fuel burn, instead
        # of applying a new D + W_current * max_total_accel_g cap here.
        # This prevents accidentally using 0.30 g thrust in the combined sizing
        # after the mission optimizer already sized for 0.15 g.
        if (
            USE_IMPORTED_THRUST_REQUIREMENT_FOR_T_USED
            and segment.thrust_req_override_N is not None
            and segment.thrust_req_override_N > D_used
        ):
            T_accel_limited = float(segment.thrust_req_override_N)
            a_x_allow = max(0.0, (T_accel_limited - D_used) / W_current)
        else:
            T_accel_limited = D_used + W_current * a_x_allow

        T_used = min(T_available, T_accel_limited)

        if T_available <= D_used:
            raise ValueError(
                f"For segment '{segment.name}', available thrust is below drag. "
                f"T_available={T_available:.3f} N, D={D_used:.3f} N."
            )

        thrust_margin_numerical_tol = 1e-6 * max(1.0, D_used)
        if T_used <= D_used + thrust_margin_numerical_tol:
            raise ValueError(
                f"For segment '{segment.name}', acceleration-limited thrust is below drag. "
                f"T_used={T_used:.3f} N, D={D_used:.3f} N, "
                f"T_accel_limited={T_accel_limited:.3f} N, "
                f"T_available={T_available:.3f} N, "
                f"a_x_allow={a_x_allow:.6f} m/s², "
                f"max_total_accel_g={segment.max_total_accel_g:.3f}, "
                f"W_current={W_current:.3f} kg."
            )

        axial_accel_m_s2 = (T_used - D_used) / W_current
        vertical_accel_m_s2 = a_y
        total_accel_m_s2 = math.sqrt(axial_accel_m_s2**2 + vertical_accel_m_s2**2)

        axial_accel_g = axial_accel_m_s2 / segment.g
        vertical_accel_g = vertical_accel_m_s2 / segment.g
        total_accel_g = total_accel_m_s2 / segment.g

        delta_energy_height = segment.delta_h + (segment.V_final**2 - segment.V_initial**2) / (2.0 * segment.g)

        thrust_excess_ratio = 1.0 - D_used / T_used
        if thrust_excess_ratio <= 0.0:
            raise ValueError(
                f"For segment '{segment.name}', thrust excess is non-positive. "
                f"T_used={T_used:.3f} N, D={D_used:.3f} N."
            )

        exponent = -delta_energy_height / (
            segment.I_sp
            * segment.V_average
            * thrust_excess_ratio
        )

        if exponent < -700.0:
            raise ValueError(
                f"For segment '{segment.name}', fuel fraction underflow. "
                f"exponent={exponent:.3f}, T_used={T_used:.3f} N, D={D_used:.3f} N, "
                f"T/D={T_used / D_used:.4f}, 1-D/T={thrust_excess_ratio:.6f}."
            )

        fraction = math.exp(exponent)

        aero_info["T_available"] = T_available
        aero_info["T_accel_limited"] = T_accel_limited
        aero_info["T_used"] = T_used
        aero_info["thrust_excess_ratio"] = thrust_excess_ratio
        aero_info["axial_accel_m_s2"] = axial_accel_m_s2
        aero_info["vertical_accel_m_s2"] = vertical_accel_m_s2
        aero_info["total_accel_m_s2"] = total_accel_m_s2
        aero_info["axial_accel_g"] = axial_accel_g
        aero_info["vertical_accel_g"] = vertical_accel_g
        aero_info["total_accel_g"] = total_accel_g
        aero_info["a_x_allow_m_s2"] = a_x_allow

    elif segment.mode == "T_eq_D":
        if segment.I_sp <= 0:
            raise ValueError(f"For segment '{segment.name}', I_sp must be positive.")
        if W_current <= 0:
            raise ValueError(f"For segment '{segment.name}', W_current must be positive.")

        W_current_force = W_current * segment.g
        exponent = -(D_used * segment.delta_t) / (segment.I_sp * W_current_force)
        fraction = math.exp(exponent)

        aero_info["T_available"] = segment.T
        aero_info["T_used"] = D_used
        aero_info["T_accel_limited"] = D_used
        aero_info["axial_accel_m_s2"] = 0.0
        aero_info["vertical_accel_m_s2"] = 0.0
        aero_info["total_accel_m_s2"] = 0.0
        aero_info["axial_accel_g"] = 0.0
        aero_info["vertical_accel_g"] = 0.0
        aero_info["total_accel_g"] = 0.0
        aero_info["a_x_allow_m_s2"] = 0.0

    else:
        raise ValueError("mode must be 'fixed', 'T_gt_D', or 'T_eq_D'.")

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Invalid segment fraction in segment '{segment.name}': {fraction}")

    return fraction, D_used, aero_info


# =============================================================================
# Fuel, tank, weight and volume calculations
# =============================================================================

def fuel_masses_from_segments(
    W_to: float,
    S_plan: float,
    segments: list[MissionSegment],
    k_rf: float,
) -> tuple[
    float, float, float, float, float, list[float], float,
    dict[str, float], dict[str, str], dict[str, float], dict[str, dict[str, Any]]
]:
    if W_to <= 0:
        raise ValueError("W_to must be positive.")
    if S_plan <= 0:
        raise ValueError("S_plan must be positive.")
    if k_rf < 0:
        raise ValueError("k_rf must be non-negative.")

    W_current = W_to

    segment_fractions = []
    segment_burns = {}
    segment_propulsion_modes = {}
    segment_drags = {}
    segment_aero = {}

    mission_burn_JetA = 0.0
    mission_burn_LH2_ramjet = 0.0
    mission_burn_LH2_scramjet = 0.0

    for segment in segments:
        fraction, D_used, aero_info = segment_weight_fraction(segment, W_current, S_plan)
        burned_mass = W_current * (1.0 - fraction)

        if segment.fuel_type == "JetA":
            mission_burn_JetA += burned_mass
        elif segment.fuel_type == "LH2":
            if segment.propulsion_mode == "ramjet":
                mission_burn_LH2_ramjet += burned_mass
            elif segment.propulsion_mode == "scramjet":
                mission_burn_LH2_scramjet += burned_mass
            else:
                raise ValueError(
                    f"Segment '{segment.name}' uses LH2, so propulsion_mode must be 'ramjet' or 'scramjet'."
                )
        elif segment.fuel_type == "none":
            pass
        else:
            raise ValueError(f"fuel_type for segment '{segment.name}' must be 'LH2', 'JetA', or 'none'.")

        segment_fractions.append(fraction)
        segment_burns[segment.name] = burned_mass
        segment_propulsion_modes[segment.name] = segment.propulsion_mode
        segment_drags[segment.name] = D_used
        segment_aero[segment.name] = aero_info

        W_current *= fraction

    total_mission_fraction = float(np.prod(segment_fractions))

    W_fuel_JetA = (1.0 + k_rf) * mission_burn_JetA
    W_fuel_LH2_ramjet = (1.0 + k_rf) * mission_burn_LH2_ramjet
    W_fuel_LH2_scramjet = (1.0 + k_rf) * mission_burn_LH2_scramjet

    W_fuel_LH2 = W_fuel_LH2_ramjet + W_fuel_LH2_scramjet
    W_fuel_total = W_fuel_JetA + W_fuel_LH2

    return (
        W_fuel_total,
        W_fuel_LH2,
        W_fuel_JetA,
        W_fuel_LH2_ramjet,
        W_fuel_LH2_scramjet,
        segment_fractions,
        total_mission_fraction,
        segment_burns,
        segment_propulsion_modes,
        segment_drags,
        segment_aero,
    )


def tank_volume_two_fuels(
    W_fuel_LH2: float,
    W_fuel_JetA: float,
    rho_LH2: float,
    rho_JetA: float,
    k_pf: float,
) -> tuple[float, float, float]:
    if rho_LH2 <= 0 or rho_JetA <= 0:
        raise ValueError("Fuel densities must be positive.")
    if k_pf <= 0:
        raise ValueError("k_pf must be positive.")

    V_LH2 = W_fuel_LH2 / (rho_LH2 * k_pf)
    V_JetA = W_fuel_JetA / (rho_JetA * k_pf)
    return V_LH2 + V_JetA, V_LH2, V_JetA


def converge_TOGW_for_fixed_S_plan(
    I_str: float,
    K_W: float,
    S_plan: float,
    I_tps: float,
    KIT: float,
    I_tank: float,
    rho_LH2: float,
    rho_JetA: float,
    k_pf: float,
    I_sub: float,
    W_prop: float,
    W_payload: float,
    segments: list[MissionSegment],
    k_rf: float,
    W_to_guess: float,
    weight_tol: float = 1.0,
    weight_relaxation: float = 0.6,
    max_weight_iter: int = 500,
) -> tuple[float, dict[str, Any]]:
    S_wet = s_wet(K_W, S_plan)
    W_str = I_str * S_wet
    W_tps = I_tps * S_wet
    W_to = W_to_guess

    for weight_iteration in range(max_weight_iter):
        (
            W_fuel_total,
            W_fuel_LH2,
            W_fuel_JetA,
            W_fuel_LH2_ramjet,
            W_fuel_LH2_scramjet,
            segment_fractions,
            total_mission_fraction,
            segment_burns,
            segment_propulsion_modes,
            segment_drags,
            segment_aero,
        ) = fuel_masses_from_segments(W_to, S_plan, segments, k_rf)

        V_tank_capacity, V_LH2, V_JetA = tank_volume_two_fuels(
            W_fuel_LH2=W_fuel_LH2,
            W_fuel_JetA=W_fuel_JetA,
            rho_LH2=rho_LH2,
            rho_JetA=rho_JetA,
            k_pf=k_pf,
        )

        W_tank = (1.0 - KIT) * I_tank * V_tank_capacity
        W_lg = W_landinggear(W_to)
        W_sub = I_sub * W_to

        W_to_required = W_str + W_tps + W_lg + W_prop + W_tank + W_sub + W_payload + W_fuel_total
        weight_error = W_to_required - W_to

        if abs(weight_error) < weight_tol:
            return W_to, {
                "TOGW": W_to,
                "TOGW_required": W_to_required,
                "weight_error": weight_error,
                "weight_iterations": weight_iteration,
                "S_plan": S_plan,
                "K_W": K_W,
                "S_wet": S_wet,
                "W_str": W_str,
                "W_tps": W_tps,
                "W_landinggear": W_lg,
                "W_prop": W_prop,
                "W_tank": W_tank,
                "W_sub": W_sub,
                "W_payload": W_payload,
                "W_fuel": W_fuel_total,
                "W_fuel_LH2": W_fuel_LH2,
                "W_fuel_JetA": W_fuel_JetA,
                "W_fuel_LH2_ramjet": W_fuel_LH2_ramjet,
                "W_fuel_LH2_scramjet": W_fuel_LH2_scramjet,
                "V_tank_capacity": V_tank_capacity,
                "V_LH2": V_LH2,
                "V_JetA": V_JetA,
                "segment_fractions": segment_fractions,
                "segment_burns": segment_burns,
                "segment_propulsion_modes": segment_propulsion_modes,
                "segment_drags": segment_drags,
                "segment_aero": segment_aero,
                "total_mission_fraction": total_mission_fraction,
            }

        W_to_new = W_to + weight_relaxation * weight_error
        if W_to_new <= 0:
            raise RuntimeError("TOGW became non-physical.")
        W_to = W_to_new

    raise RuntimeError("TOGW did not converge.")


def required_volume(
    I_str: float,
    I_tps: float,
    K_W: float,
    S_plan: float,
    W_payload: float,
    W_fuel_LH2: float,
    W_fuel_JetA: float,
    rho_LH2: float,
    rho_JetA: float,
    k_pf: float,
    KIT: float,
    I_tank: float,
    rho_str: float,
    rho_tps: float,
    rho_tank_str: float,
    rho_payload: float,
    K_lg: float,
    K_sub: float,
    K_void: float,
    V_prop: float,
    V_tot_available: float,
) -> dict[str, float]:
    V_structure = (I_str * K_W * S_plan) / rho_str
    V_tps = (I_tps * K_W * S_plan) / rho_tps

    V_tank_capacity, V_LH2, V_JetA = tank_volume_two_fuels(
        W_fuel_LH2=W_fuel_LH2,
        W_fuel_JetA=W_fuel_JetA,
        rho_LH2=rho_LH2,
        rho_JetA=rho_JetA,
        k_pf=k_pf,
    )

    V_tank_structure = (1.0 - KIT) * I_tank * V_tank_capacity / rho_tank_str
    V_payload = W_payload / rho_payload

    V_landinggear = K_lg * V_tot_available
    V_subsystem = K_sub * V_tot_available
    V_void = K_void * V_tot_available

    V_required = (
        V_structure
        + V_tps
        + V_landinggear
        + V_prop
        + V_tank_structure
        + V_subsystem
        + V_void
        + V_payload
        + V_tank_capacity
    )

    return {
        "V_required": V_required,
        "V_structure": V_structure,
        "V_tps": V_tps,
        "V_landinggear": V_landinggear,
        "V_prop": V_prop,
        "V_tank_structure": V_tank_structure,
        "V_subsystem": V_subsystem,
        "V_void": V_void,
        "V_payload": V_payload,
        "V_tank_capacity": V_tank_capacity,
        "V_LH2": V_LH2,
        "V_JetA": V_JetA,
    }


def converge_S_plan_and_TOGW(
    tau: float,
    configuration: str,
    S_plan_guess: float,
    I_str: float,
    I_tps: float,
    KIT: float,
    I_tank: float,
    rho_LH2: float,
    rho_JetA: float,
    k_pf: float,
    I_sub: float,
    W_prop: float,
    V_prop: float,
    W_payload: float,
    rho_payload: float,
    segments: list[MissionSegment],
    k_rf: float = 0.06,
    W_to_guess: float = 100_000.0,
    rho_str: float = 2700.0,
    rho_tps: float = 500.0,
    rho_tank_str: float = 2700.0,
    K_lg: float = 0.01,
    K_sub: float = 0.02,
    K_void: float = 0.20,
    volume_tol: float = 1.0,
    weight_tol: float = 1.0,
    S_plan_relaxation: float = 0.5,
    weight_relaxation: float = 0.6,
    max_size_iter: int = 300,
    max_weight_iter: int = 500,
) -> tuple[float, float, dict[str, Any]]:
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if S_plan_guess <= 0:
        raise ValueError("S_plan_guess must be positive.")

    K_W = k_w_from_tau(tau, configuration)
    S_plan = S_plan_guess
    W_to_current_guess = W_to_guess

    for size_iteration in range(max_size_iter):
        W_to, weight_result = converge_TOGW_for_fixed_S_plan(
            I_str=I_str,
            K_W=K_W,
            S_plan=S_plan,
            I_tps=I_tps,
            KIT=KIT,
            I_tank=I_tank,
            rho_LH2=rho_LH2,
            rho_JetA=rho_JetA,
            k_pf=k_pf,
            I_sub=I_sub,
            W_prop=W_prop,
            W_payload=W_payload,
            segments=segments,
            k_rf=k_rf,
            W_to_guess=W_to_current_guess,
            weight_tol=weight_tol,
            weight_relaxation=weight_relaxation,
            max_weight_iter=max_weight_iter,
        )

        V_tot_available = volume_from_tau(tau, S_plan)

        volume_result = required_volume(
            I_str=I_str,
            I_tps=I_tps,
            K_W=K_W,
            S_plan=S_plan,
            W_payload=W_payload,
            W_fuel_LH2=weight_result["W_fuel_LH2"],
            W_fuel_JetA=weight_result["W_fuel_JetA"],
            rho_LH2=rho_LH2,
            rho_JetA=rho_JetA,
            k_pf=k_pf,
            KIT=KIT,
            I_tank=I_tank,
            rho_str=rho_str,
            rho_tps=rho_tps,
            rho_tank_str=rho_tank_str,
            rho_payload=rho_payload,
            K_lg=K_lg,
            K_sub=K_sub,
            K_void=K_void,
            V_prop=V_prop,
            V_tot_available=V_tot_available,
        )

        V_required = volume_result["V_required"]
        volume_error = V_required - V_tot_available

        if abs(volume_error) < volume_tol:
            return S_plan, W_to, {
                **weight_result,
                **volume_result,
                "tau": tau,
                "configuration": configuration,
                "K_W": K_W,
                "S_plan": S_plan,
                "TOGW": W_to,
                "V_tot_available": V_tot_available,
                "volume_error": volume_error,
                "size_iterations": size_iteration,
                "wing_loading": W_to / S_plan,
            }

        raw_correction = (V_required / V_tot_available) ** (2.0 / 3.0)
        correction = 1.0 + S_plan_relaxation * (raw_correction - 1.0)
        correction = max(0.5, min(1.5, correction))

        S_plan_new = S_plan * correction
        if S_plan_new <= 0:
            raise RuntimeError("S_plan became non-physical.")

        S_plan = S_plan_new
        W_to_current_guess = W_to

    raise RuntimeError("S_plan did not converge.")


# =============================================================================
# Mission profile helper
# =============================================================================

def acceleration_distance(V_initial: float, V_final: float, accel: float) -> float:
    if accel <= 0:
        raise ValueError("accel must be positive.")
    return abs(V_final**2 - V_initial**2) / (2.0 * accel)


def climb_distance_from_gamma(delta_h: float, gamma_deg: float) -> float:
    gamma_rad = math.radians(gamma_deg)

    if abs(math.tan(gamma_rad)) < 1e-12:
        raise ValueError("gamma_deg too close to zero for climb distance.")

    return abs(delta_h / math.tan(gamma_rad))


def constant_q_profile_check(
    M_start: float,
    h_start: float,
    M_end: float,
    h_end: float,
) -> dict[str, float]:
    q_start = dynamic_pressure_from_mach_altitude(M_start, h_start)
    q_end = dynamic_pressure_from_mach_altitude(M_end, h_end)

    return {
        "M_start": M_start,
        "h_start": h_start,
        "M_end": M_end,
        "h_end": h_end,
        "q_start": q_start,
        "q_end": q_end,
        "q_error": q_end - q_start,
        "q_ratio": q_end / q_start if q_start > 0 else float("nan"),
    }


def altitude_for_mach_at_dynamic_pressure(
    target_mach: float,
    q_target: float,
    h_low: float = 0.0,
    h_high: float = 50_000.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    if target_mach <= 0:
        raise ValueError("target_mach must be positive.")
    if q_target <= 0:
        raise ValueError("q_target must be positive.")

    q_low = dynamic_pressure_from_mach_altitude(target_mach, h_low)
    q_high = dynamic_pressure_from_mach_altitude(target_mach, h_high)

    if not (q_low >= q_target >= q_high):
        raise ValueError(
            f"q_target={q_target:.3f} Pa is not bracketed for M={target_mach:.3f}: "
            f"q({h_low:.1f} m)={q_low:.3f}, q({h_high:.1f} m)={q_high:.3f}."
        )

    for _ in range(max_iter):
        h_mid = 0.5 * (h_low + h_high)
        q_mid = dynamic_pressure_from_mach_altitude(target_mach, h_mid)

        if q_mid > q_target:
            h_low = h_mid
        else:
            h_high = h_mid

        if abs(h_high - h_low) < tol:
            break

    return 0.5 * (h_low + h_high)


def compute_8_segment_profile(
    gamma_climb_deg: float,
    h0: float = 0.0,
    h_step: float = 16_093.0,
    h_cruise: float = 30_000.0,
    M_takeoff_drag: float = 0.30,
    M_accel_1: float = 0.70,
    M_climb_1: float = 0.90,
    M_accel_2: float = 1.70,
    M_cruise: float = 5.0,
    cruise_time: float = 90.0 * 60.0,
    accel_for_distance: float = 0.15 * 9.81,
    descent_accel: float = 0.15 * 9.81,
    total_range: float = 9500e3,
) -> dict[str, Any]:
    """
    8-segment mission profile geometry:

    1. take-off
    2. acceleration to M = 0.7
    3. climb to h = 16.093 km and M = 0.9
    4. acceleration to M = 1.7
    5. climb along constant-q path to M = 5 at 30 km
       This file splits segment 5 by propulsion mode:
       5a. turbojet from M1.7 to M2.5
       5b. ramjet from M2.5 to M5
    6. cruise at M = 5 for 90 min at 30 km
    7. unpowered descent
    8. landing
    """
    V_takeoff_drag = mach_to_velocity(M_takeoff_drag, h0)
    V_accel_1 = mach_to_velocity(M_accel_1, h0)
    V_climb_1 = mach_to_velocity(M_climb_1, h_step)
    V_accel_2 = mach_to_velocity(M_accel_2, h_step)
    V_cruise = mach_to_velocity(M_cruise, h_cruise)

    q_check = constant_q_profile_check(
        M_start=M_accel_2,
        h_start=h_step,
        M_end=M_cruise,
        h_end=h_cruise,
    )

    dx_accel_1 = acceleration_distance(
        V_initial=V_takeoff_drag,
        V_final=V_accel_1,
        accel=accel_for_distance,
    )

    dx_climb_1 = climb_distance_from_gamma(
        delta_h=h_step - h0,
        gamma_deg=gamma_climb_deg,
    )

    dx_accel_2 = acceleration_distance(
        V_initial=V_climb_1,
        V_final=V_accel_2,
        accel=accel_for_distance,
    )

    dx_const_q_climb = climb_distance_from_gamma(
        delta_h=h_cruise - h_step,
        gamma_deg=gamma_climb_deg,
    )

    cruise_range = cruise_time * V_cruise

    cruise_start_x = (
        dx_accel_1
        + dx_climb_1
        + dx_accel_2
        + dx_const_q_climb
    )

    cruise_end_x = cruise_start_x + cruise_range

    descent = analyse_descent(
        end_cruise_x=cruise_end_x,
        h_cruise=h_cruise,
        v_cruise=V_cruise,
        acc_tot=descent_accel,
        total_range=total_range,
    )

    return {
        "h0": h0,
        "h_step": h_step,
        "h_cruise": h_cruise,
        "gamma_climb_deg": gamma_climb_deg,

        "M_takeoff_drag": M_takeoff_drag,
        "M_accel_1": M_accel_1,
        "M_climb_1": M_climb_1,
        "M_accel_2": M_accel_2,
        "M_cruise": M_cruise,

        "V_takeoff_drag": V_takeoff_drag,
        "V_accel_1": V_accel_1,
        "V_climb_1": V_climb_1,
        "V_accel_2": V_accel_2,
        "V_cruise": V_cruise,

        "q_const_start": q_check["q_start"],
        "q_const_end": q_check["q_end"],
        "q_const_error": q_check["q_error"],
        "q_const_ratio": q_check["q_ratio"],

        "dx_accel_1": dx_accel_1,
        "dx_climb_1": dx_climb_1,
        "dx_accel_2": dx_accel_2,
        "dx_const_q_climb": dx_const_q_climb,

        "cruise_start_x": cruise_start_x,
        "cruise_range": cruise_range,
        "cruise_end_x": cruise_end_x,

        **descent,
    }


def analyse_descent(
    end_cruise_x: float,
    h_cruise: float,
    v_cruise: float,
    acc_tot: float = 0.15 * 9.81,
    total_range: float = 9500e3,
) -> dict[str, float | bool]:
    remaining_range = total_range - end_cruise_x

    if remaining_range <= 0:
        feasible = find_feasible_descent_acceleration(h_cruise, v_cruise, acc_tot)
        return {
            **feasible,
            "descent_fits_total_range": False,
            "final_total_range": end_cruise_x + feasible["x_descent"],
        }

    a_x = -v_cruise**2 / (2.0 * remaining_range)
    t_descent = v_cruise / (-a_x)
    a_y_mag = 2.0 * h_cruise / t_descent**2
    a_descent = math.sqrt(a_x**2 + a_y_mag**2)

    if a_descent > acc_tot:
        feasible = find_feasible_descent_acceleration(h_cruise, v_cruise, acc_tot)
        return {
            **feasible,
            "descent_fits_total_range": False,
            "final_total_range": end_cruise_x + feasible["x_descent"],
        }

    return {
        "a_x_descent": a_x,
        "a_y_descent": -a_y_mag,
        "a_descent": a_descent,
        "t_descent": t_descent,
        "x_descent": remaining_range,
        "descent_fits_total_range": True,
        "final_total_range": total_range,
    }


def find_feasible_descent_acceleration(
    h_cruise: float,
    v_cruise: float,
    acc_tot: float = 0.15 * 9.81,
) -> dict[str, float]:
    def descent_range_from_ax(a_x: float) -> float:
        return v_cruise**2 / (2.0 * -a_x)

    def accel_margin(x: np.ndarray) -> float:
        a_x = x[0]
        t_descent = v_cruise / (-a_x)
        a_y_mag = 2.0 * h_cruise / t_descent**2
        return acc_tot**2 - (a_x**2 + a_y_mag**2)

    result = minimize(
        fun=lambda x: descent_range_from_ax(x[0]),
        x0=np.array([-0.5 * acc_tot]),
        method="SLSQP",
        bounds=[(-acc_tot, -1e-6)],
        constraints=[{"type": "ineq", "fun": accel_margin}],
    )

    if not result.success:
        raise RuntimeError(f"Descent optimisation failed: {result.message}")

    a_x = float(result.x[0])
    t_descent = v_cruise / (-a_x)
    a_y_mag = 2.0 * h_cruise / t_descent**2
    x_descent = descent_range_from_ax(a_x)

    return {
        "a_x_descent": a_x,
        "a_y_descent": -a_y_mag,
        "a_descent": math.sqrt(a_x**2 + a_y_mag**2),
        "t_descent": t_descent,
        "x_descent": x_descent,
    }


# =============================================================================
# Run
# =============================================================================


ENGINE_TRANSITION_MACH = {"T2R": 2.5, "R2S": 5.0}

ISP_BY_PROPULSION = {
    "turbojet": 2100.0,   # Jet-A
    "ramjet": 3500.0,     # LH2
    "scramjet": 1500.0,   # LH2
}

FUEL_BY_PROPULSION = {
    "turbojet": "JetA",
    "ramjet": "LH2",
    "scramjet": "LH2",
}

THRUST_KEY_BY_PROPULSION = {
    "turbojet": "turbo",
    "ramjet": "ram",
    "scramjet": "scram",
}


def propulsion_from_mach(M: float) -> str:
    if M < ENGINE_TRANSITION_MACH["T2R"]:
        return "turbojet"
    if M < ENGINE_TRANSITION_MACH["R2S"]:
        return "ramjet"
    return "scramjet"


def _mach_at_v_h(V: float, h: float) -> float:
    return V / speed_of_sound(h)


def _segment_from_state(name, x0, x1, h0, h1, V0, V1, gamma_deg, max_total_accel_g=0.30):
    """Create one sizing MissionSegment from endpoint states."""
    if x1 <= x0:
        return None

    h_mid = 0.5 * (h0 + h1)
    V_mid = 0.5 * (V0 + V1)
    M_mid = _mach_at_v_h(V_mid, h_mid)
    propulsion = propulsion_from_mach(M_mid)
    thrust_key = THRUST_KEY_BY_PROPULSION[propulsion]
    T_available, _ = thrust_from_mach_altitude(thrust_key, M_mid, h_mid)

    return MissionSegment(
        name=name,
        mode="T_gt_D",
        fuel_type=FUEL_BY_PROPULSION[propulsion],
        propulsion_mode=propulsion,
        delta_h=h1 - h0,
        V_initial=V0,
        V_final=V1,
        V_average=V_mid,
        I_sp=ISP_BY_PROPULSION[propulsion],
        mach_drag=M_mid,
        altitude_drag=h_mid,
        flight_path_angle_deg=gamma_deg,
        T=T_available,
        max_total_accel_g=max_total_accel_g,
    )


def build_segments_from_updated_mission_profile(
    gamma_mission: float = 7.0,
    h_cruise: float = 30_000.0,
    acc_tot: float = 0.15 * 9.81,
    M_cruise: float = 5.0,
    cruise_time: float = 90.0 * 60.0,
    total_range: float = 9500e3,
    fixed_takeoff_fraction: float = 0.990,
    fixed_landing_fraction: float = 0.997,
) -> tuple[list[MissionSegment], dict[str, float | bool]]:
    """
    Convert the continuous mission profile into the discrete sizing segments.

    Mission profile used:
      1. take-off, fixed fraction
      2. constant-acceleration climb from ground to h_cruise
      3. horizontal acceleration at h_cruise until M_cruise
      4. cruise at M_cruise for cruise_time
      5. unpowered descent sized to total_range, or shortest feasible descent
      6. landing, fixed fraction

    Engine split follows the mission profile convention:
      turbojet for M < 2.5, ramjet for 2.5 <= M < 5, scramjet for M >= 5.
    """
    if gamma_mission <= 0:
        raise ValueError("gamma_mission must be positive.")
    if h_cruise <= 0:
        raise ValueError("h_cruise must be positive.")

    gamma_rad = math.radians(gamma_mission)
    acc_x = acc_tot * math.cos(gamma_rad)
    acc_y = acc_tot * math.sin(gamma_rad)

    V_cruise = mach_to_velocity(M_cruise, h_cruise)
    a_cruise = speed_of_sound(h_cruise)

    # Ascent endpoint from your mission-profile script.
    V_y_to_cruise = math.sqrt(2.0 * acc_y * h_cruise)
    t_to_cruise = V_y_to_cruise / acc_y
    V_x_to_cruise = acc_x * t_to_cruise
    V_top = math.sqrt(V_x_to_cruise**2 + V_y_to_cruise**2)
    dx_to_cruise = V_x_to_cruise**2 / (2.0 * acc_x)

    # Horizontal acceleration endpoint.
    dx_hor_acc = max(0.0, (V_cruise**2 - V_x_to_cruise**2) / (2.0 * acc_tot))
    cruise_start_x = dx_to_cruise + dx_hor_acc
    cruise_range = cruise_time * V_cruise
    cruise_end_x = cruise_start_x + cruise_range

    descent = analyse_descent(
        end_cruise_x=cruise_end_x,
        h_cruise=h_cruise,
        v_cruise=V_cruise,
        acc_tot=acc_tot,
        total_range=total_range,
    )

    # Helper equations for states inside the ascent and horizontal-acceleration phases.
    def ascent_state_at_M(M_target: float):
        # Solve by bisection because speed of sound varies with altitude.
        lo, hi = 0.0, t_to_cruise
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            h = 0.5 * acc_y * mid**2
            V = acc_tot * mid
            M = _mach_at_v_h(V, h)
            if M < M_target:
                lo = mid
            else:
                hi = mid
        t = 0.5 * (lo + hi)
        return {
            "x": 0.5 * acc_x * t**2,
            "h": 0.5 * acc_y * t**2,
            "V": acc_tot * t,
            "M": M_target,
        }

    def horizontal_state_at_M(M_target: float):
        V = M_target * a_cruise
        dx = max(0.0, (V**2 - V_x_to_cruise**2) / (2.0 * acc_tot))
        return {"x": dx_to_cruise + dx, "h": h_cruise, "V": V, "M": M_target}

    # Build split points. Only include transition points that actually lie inside the phase.
    points = [{"label": "start", "x": 0.0, "h": 0.0, "V": 1.0, "M": 0.01}]

    M_top = _mach_at_v_h(V_top, h_cruise)
    for M_transition in (ENGINE_TRANSITION_MACH["T2R"], ENGINE_TRANSITION_MACH["R2S"]):
        if M_transition < M_top:
            points.append({"label": f"ascent_M{M_transition:g}", **ascent_state_at_M(M_transition)})

    points.append({"label": "top_of_climb", "x": dx_to_cruise, "h": h_cruise, "V": V_top, "M": M_top})

    for M_transition in (ENGINE_TRANSITION_MACH["T2R"], ENGINE_TRANSITION_MACH["R2S"]):
        if M_top < M_transition < M_cruise:
            points.append({"label": f"hacc_M{M_transition:g}", **horizontal_state_at_M(M_transition)})

    points.append({"label": "M_cruise", "x": cruise_start_x, "h": h_cruise, "V": V_cruise, "M": M_cruise})
    points = sorted(points, key=lambda p: p["x"])

    segments: list[MissionSegment] = []

    # Fixed takeoff; keep a tiny but nonzero Mach for aero calculations.
    T_takeoff, _ = thrust_from_mach_altitude("turbo", 0.20, 0.0)
    segments.append(MissionSegment(
        name="1_takeoff_turbojet",
        mode="fixed",
        fuel_type="JetA",
        propulsion_mode="turbojet",
        fixed_fraction=fixed_takeoff_fraction,
        mach_drag=0.30,
        altitude_drag=0.0,
        flight_path_angle_deg=5.0,
        T=T_takeoff,
    ))

    for i, (p0, p1) in enumerate(zip(points[:-1], points[1:]), start=2):
        gamma_seg = gamma_mission if p1["h"] > p0["h"] else 0.0
        seg = _segment_from_state(
            name=f"{i}_updated_profile_{p0['label']}_to_{p1['label']}",
            x0=p0["x"],
            x1=p1["x"],
            h0=p0["h"],
            h1=p1["h"],
            V0=p0["V"],
            V1=p1["V"],
            gamma_deg=gamma_seg,
            max_total_accel_g=0.30,
        )
        if seg is not None:
            segments.append(seg)

    T_scram_cruise, _ = thrust_from_mach_altitude("scram", M_cruise, h_cruise)
    segments.append(MissionSegment(
        name=f"{len(segments)+1}_cruise_M{M_cruise:g}_scramjet",
        mode="T_eq_D",
        fuel_type="LH2",
        propulsion_mode="scramjet",
        I_sp=ISP_BY_PROPULSION["scramjet"],
        mach_drag=M_cruise,
        altitude_drag=h_cruise,
        flight_path_angle_deg=0.0,
        T=T_scram_cruise,
        delta_t=cruise_time,
    ))

    segments.append(MissionSegment(
        name=f"{len(segments)+1}_unpowered_descent",
        mode="fixed",
        fuel_type="none",
        propulsion_mode="none",
        fixed_fraction=1.0,
        mach_drag=max(0.65, 0.5 * M_cruise),
        altitude_drag=0.5 * h_cruise,
        flight_path_angle_deg=-5.0,
    ))

    segments.append(MissionSegment(
        name=f"{len(segments)+1}_landing_turbojet",
        mode="fixed",
        fuel_type="JetA",
        propulsion_mode="turbojet",
        fixed_fraction=fixed_landing_fraction,
        mach_drag=0.25,
        altitude_drag=0.0,
        flight_path_angle_deg=-3.0,
        T=T_takeoff,
    ))

    profile = {
        "gamma_mission": gamma_mission,
        "h_cruise": h_cruise,
        "acc_tot": acc_tot,
        "M_cruise": M_cruise,
        "V_cruise": V_cruise,
        "V_top": V_top,
        "M_top": M_top,
        "dx_to_cruise": dx_to_cruise,
        "dx_hor_acc": dx_hor_acc,
        "cruise_start_x": cruise_start_x,
        "cruise_range": cruise_range,
        "cruise_end_x": cruise_end_x,
        **descent,
    }
    return segments, profile



# =============================================================================
# Import optimized mission profile from transition optimizer
# =============================================================================

OPTIMIZED_PROFILE_FILE = "optimize_transition_profile.py"
USE_IMPORTED_OPTIMIZED_PROFILE = True
MISSION_IMPORTED_THRUST_MARGIN = 1.00
# If True, imported profile segments use the optimizer's own thrust requirement
# T_req = D + 0.15g term, rather than recomputing a new acceleration cap in
# the combined sizing loop. This avoids double-counting acceleration.
USE_IMPORTED_THRUST_REQUIREMENT_FOR_T_USED = True
MISSION_OPTIMIZER_SUPPRESS_OUTPUT = False


def import_transition_optimizer(path: str | Path = OPTIMIZED_PROFILE_FILE):
    """Dynamically import the optimized transition-profile script."""
    path = Path(path)
    if not path.exists():
        # Convenience fallback: if this combined sizing script is in the same
        # folder as a differently named optimizer, try the latest plot-only name.
        fallback = Path("optimize_transition_profile_plot_only.py")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(
                f"Could not find transition optimizer file: {path}. "
                "Put optimize_transition_profile_updated_drag.py in the same folder "
                "as this combined sizing file, or edit OPTIMIZED_PROFILE_FILE."
            )

    spec = importlib.util.spec_from_file_location("transition_profile_optimizer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import transition optimizer from {path}")
    module = importlib.util.module_from_spec(spec)

    # Important for Python 3.12/3.13 + @dataclass:
    # dataclasses looks up sys.modules[cls.__module__] while the module is
    # being executed. Dynamic imports must register the module before exec.
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Avoid leaving a half-imported module behind after a failed import.
        sys.modules.pop(spec.name, None)
        raise

    return module


def run_transition_profile_optimizer(profile_file: str | Path = OPTIMIZED_PROFILE_FILE):
    """
    Run the optimized transition-profile code without calling its save/plot routine.

    Returns:
        module, x_best, sizing
    """
    module = import_transition_optimizer(profile_file)

    def _run():
        maps = module.prepare_thrust_maps()
        x_best, sizing = module.optimize_profile(maps)
        return x_best, sizing

    if MISSION_OPTIMIZER_SUPPRESS_OUTPUT:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            x_best, sizing = _run()
    else:
        x_best, sizing = _run()

    return module, x_best, sizing


def propulsion_from_imported_mode(mode: str) -> str:
    mode = str(mode).lower()
    if "turbo" in mode:
        return "turbojet"
    if "ram" in mode:
        return "ramjet"
    # The current transition optimizer uses turbojet below M=3 and ramjet from M=3.
    # This fallback keeps M=5 cruise on ramjet instead of scramjet.
    return "ramjet"


def fuel_from_imported_propulsion(propulsion: str) -> str:
    if propulsion == "turbojet":
        return "JetA"
    if propulsion == "ramjet":
        return "LH2"
    return "none"


def isp_from_imported_propulsion(propulsion: str) -> float:
    if propulsion == "turbojet":
        return ISP_BY_PROPULSION["turbojet"]
    if propulsion == "ramjet":
        return ISP_BY_PROPULSION["ramjet"]
    return 0.0


def build_segments_from_imported_optimized_profile(
    cruise_time: float = 90.0 * 60.0,
    total_range: float = 9500e3,
    fixed_takeoff_fraction: float = 0.990,
    fixed_landing_fraction: float = 0.997,
    profile_file: str | Path = OPTIMIZED_PROFILE_FILE,
) -> tuple[list[MissionSegment], dict[str, Any]]:
    """
    Import the optimized Mach-altitude profile and convert it into combined-sizing segments.

    The imported optimizer already sizes thrust so that:
        T_available >= drag + acceleration requirement

    Therefore these segments carry over:
        - Mach
        - altitude
        - drag from the optimizer
        - sized thrust from the optimizer
        - turbojet/ramjet mode

    Cruise at Mach 5 and 30 km is kept as ramjet, matching the transition optimizer.
    """
    module, x_best, sizing = run_transition_profile_optimizer(profile_file)
    df = sizing.table.copy().sort_values("mach").reset_index(drop=True)

    # Make sure required columns exist. This catches mismatch if the optimizer script changed.
    required_columns = [
        "mach", "altitude_m", "mode", "drag_kN", "thrust_req_kN",
        "sized_T_available_kN", "q_Pa", "CL", "CD", "L_over_D",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Transition optimizer output is missing columns: {missing}")

    segments: list[MissionSegment] = []

    # Fixed takeoff bookkeeping segment. Actual climb/acceleration starts from the first imported point.
    first = df.iloc[0]
    segments.append(MissionSegment(
        name="1_takeoff_turbojet_fixed",
        mode="fixed",
        fuel_type="JetA",
        propulsion_mode="turbojet",
        fixed_fraction=fixed_takeoff_fraction,
        mach_drag=max(0.05, float(first["mach"])),
        altitude_drag=max(0.0, float(first["altitude_m"])),
        flight_path_angle_deg=5.0,
        T=float(first["sized_T_available_kN"]) * 1000.0 * MISSION_IMPORTED_THRUST_MARGIN,
        drag_override_N=float(first["drag_kN"]) * 1000.0,
        thrust_req_override_N=float(first["thrust_req_kN"]) * 1000.0,
        q_override_Pa=float(first["q_Pa"]),
        CL_override=float(first["CL"]),
        CD_override=float(first["CD"]),
        LD_override=float(first["L_over_D"]),
    ))

    # Powered climb/acceleration segments from adjacent optimizer points.
    seg_index = 2
    for i in range(len(df) - 1):
        p0 = df.iloc[i]
        p1 = df.iloc[i + 1]

        M0, M1 = float(p0["mach"]), float(p1["mach"])
        h0, h1 = float(p0["altitude_m"]), float(p1["altitude_m"])
        if M1 <= M0:
            continue

        M_mid = 0.5 * (M0 + M1)
        h_mid = 0.5 * (h0 + h1)
        V0 = mach_to_velocity(M0, h0)
        V1 = mach_to_velocity(M1, h1)
        V_mid = max(1e-6, 0.5 * (V0 + V1))

        # Use the downstream point for mode because it enforces the M=3 switch cleanly.
        propulsion = propulsion_from_imported_mode(p1["mode"])
        fuel_type = fuel_from_imported_propulsion(propulsion)
        I_sp = isp_from_imported_propulsion(propulsion)

        # Average imported aero/thrust values over the interval.
        D_override_N = 0.5 * (float(p0["drag_kN"]) + float(p1["drag_kN"])) * 1000.0
        T_available_N = 0.5 * (float(p0["sized_T_available_kN"]) + float(p1["sized_T_available_kN"])) * 1000.0
        T_req_override_N = 0.5 * (float(p0["thrust_req_kN"]) + float(p1["thrust_req_kN"])) * 1000.0
        q_override = 0.5 * (float(p0["q_Pa"]) + float(p1["q_Pa"]))
        CL_override = 0.5 * (float(p0["CL"]) + float(p1["CL"]))
        CD_override = 0.5 * (float(p0["CD"]) + float(p1["CD"]))
        LD_override = 0.5 * (float(p0["L_over_D"]) + float(p1["L_over_D"]))

        # A small margin prevents numerical equality causing T<=D failures in the sizing loop.
        T_available_N *= MISSION_IMPORTED_THRUST_MARGIN

        # Flight-path angle is not used for range here; it only affects the acceleration limiter.
        # Keep it modest because the energy-height term already accounts for climb work.
        gamma_seg = 0.0

        segments.append(MissionSegment(
            name=f"{seg_index}_imported_profile_M{M0:.2f}_to_M{M1:.2f}_{propulsion}",
            mode="T_gt_D",
            fuel_type=fuel_type,
            propulsion_mode=propulsion,
            delta_h=h1 - h0,
            V_initial=V0,
            V_final=V1,
            V_average=V_mid,
            I_sp=I_sp,
            mach_drag=M_mid,
            altitude_drag=h_mid,
            flight_path_angle_deg=gamma_seg,
            T=T_available_N,
            max_total_accel_g=getattr(module, "ACCEL_G_TARGET", 0.15),
            drag_override_N=D_override_N,
            thrust_req_override_N=T_req_override_N,
            q_override_Pa=q_override,
            CL_override=CL_override,
            CD_override=CD_override,
            LD_override=LD_override,
        ))
        seg_index += 1

    # Ramjet cruise at the final optimizer point.
    last = df.iloc[-1]
    M_cruise = float(last["mach"])
    h_cruise = float(last["altitude_m"])
    V_cruise = mach_to_velocity(M_cruise, h_cruise)
    T_cruise_N = float(last["sized_T_available_kN"]) * 1000.0 * MISSION_IMPORTED_THRUST_MARGIN
    D_cruise_N = float(last["drag_kN"]) * 1000.0

    segments.append(MissionSegment(
        name=f"{seg_index}_cruise_M{M_cruise:.1f}_ramjet",
        mode="T_eq_D",
        fuel_type="LH2",
        propulsion_mode="ramjet",
        I_sp=ISP_BY_PROPULSION["ramjet"],
        mach_drag=M_cruise,
        altitude_drag=h_cruise,
        flight_path_angle_deg=0.0,
        T=T_cruise_N,
        delta_t=cruise_time,
        drag_override_N=D_cruise_N,
        thrust_req_override_N=float(last["thrust_req_kN"]) * 1000.0,
        q_override_Pa=float(last["q_Pa"]),
        CL_override=float(last["CL"]),
        CD_override=float(last["CD"]),
        LD_override=float(last["L_over_D"]),
    ))
    seg_index += 1

    descent = analyse_descent(
        end_cruise_x=cruise_time * V_cruise,
        h_cruise=h_cruise,
        v_cruise=V_cruise,
        acc_tot=getattr(module, "ACCEL_G_TARGET", 0.15) * 9.81,
        total_range=total_range,
    )

    segments.append(MissionSegment(
        name=f"{seg_index}_unpowered_descent",
        mode="fixed",
        fuel_type="none",
        propulsion_mode="none",
        fixed_fraction=1.0,
        mach_drag=max(0.65, 0.5 * M_cruise),
        altitude_drag=0.5 * h_cruise,
        flight_path_angle_deg=-5.0,
    ))
    seg_index += 1

    segments.append(MissionSegment(
        name=f"{seg_index}_landing_turbojet_fixed",
        mode="fixed",
        fuel_type="JetA",
        propulsion_mode="turbojet",
        fixed_fraction=fixed_landing_fraction,
        mach_drag=0.25,
        altitude_drag=0.0,
        flight_path_angle_deg=-3.0,
        T=float(first["sized_T_available_kN"]) * 1000.0 * MISSION_IMPORTED_THRUST_MARGIN,
        drag_override_N=float(first["drag_kN"]) * 1000.0,
        thrust_req_override_N=float(first["thrust_req_kN"]) * 1000.0,
        q_override_Pa=float(first["q_Pa"]),
        CL_override=float(first["CL"]),
        CD_override=float(first["CD"]),
        LD_override=float(first["L_over_D"]),
    ))

    profile = {
        "source_file": str(profile_file),
        "imported_rows": len(df),
        "M_start": float(df["mach"].iloc[0]),
        "M_switch": getattr(module, "M_SWITCH", 3.0),
        "M_cruise": M_cruise,
        "h_start": float(df["altitude_m"].iloc[0]),
        "h_cruise": h_cruise,
        "V_cruise": V_cruise,
        "cruise_time": cruise_time,
        "cruise_range": cruise_time * V_cruise,
        "total_range": total_range,
        "descent_fits_total_range": descent.get("descent_fits_total_range", True),
        "final_total_range": descent.get("final_total_range", cruise_time * V_cruise + descent.get("x_descent", 0.0)),
        "x_descent": descent.get("x_descent", 0.0),
        "turbo_design_lbf_per_engine": sizing.turbo_design_lbf_per_engine,
        "turbo_design_kN_per_engine": sizing.turbo_design_lbf_per_engine * 0.0044482216152605,
        "ramjet_design_mdot_kg_s_total": sizing.ramjet_design_mdot_kg_s,
        "ramjet_design_mdot_kg_s_per_engine": sizing.ramjet_design_mdot_kg_s / max(1, getattr(module, "N_RAMJETS", 1)),
        "use_imported_thrust_requirement_for_T_used": USE_IMPORTED_THRUST_REQUIREMENT_FOR_T_USED,
        "mission_imported_thrust_margin": MISSION_IMPORTED_THRUST_MARGIN,
        "turbo_scale": sizing.max_turbo_scale,
        "ramjet_scale": sizing.max_ramjet_scale,
        "mission_optimizer_objective": sizing.objective,
        "mission_optimizer_penalty": sizing.penalty,
    }

    return segments, profile

# =============================================================================
# Run with imported optimized mission profile
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Mission definition using imported optimized transition profile
    # -------------------------------------------------------------------------
    cruise_time = 90.0 * 60.0
    total_range = 9500e3

    if USE_IMPORTED_OPTIMIZED_PROFILE:
        segments, profile = build_segments_from_imported_optimized_profile(
            cruise_time=cruise_time,
            total_range=total_range,
            fixed_takeoff_fraction=0.990,
            fixed_landing_fraction=0.997,
            profile_file=OPTIMIZED_PROFILE_FILE,
        )
    else:
        # Fallback to the old internal mission-profile builder.
        segments, profile = build_segments_from_updated_mission_profile(
            gamma_mission=7.0,
            h_cruise=30_000.0,
            acc_tot=0.15 * 9.81,
            M_cruise=5.0,
            cruise_time=cruise_time,
            total_range=total_range,
            fixed_takeoff_fraction=0.990,
            fixed_landing_fraction=0.997,
        )

    print("\nImported optimized mission profile")
    print("----------------------------------")
    print(f"source_file:          {profile.get('source_file', 'internal_builder')}")
    print(f"rows imported:        {profile.get('imported_rows', 0)}")
    print(f"M_start:              {profile.get('M_start', 0.0):.3f}")
    print(f"M_switch:             {profile.get('M_switch', 3.0):.3f}")
    print(f"M_cruise:             {profile.get('M_cruise', 5.0):.3f}")
    print(f"h_cruise:             {profile.get('h_cruise', 30_000.0):.1f} m")
    print(f"V_cruise:             {profile.get('V_cruise', 0.0):.3f} m/s")
    print(f"cruise_range:         {profile.get('cruise_range', 0.0) / 1000:.3f} km")
    print(f"x_descent:            {profile.get('x_descent', 0.0) / 1000:.3f} km")
    print(f"final_total_range:    {profile.get('final_total_range', 0.0) / 1000:.3f} km")
    print(f"descent_fits_range:   {profile.get('descent_fits_total_range', True)}")
    print(f"turbo design thrust:  {profile.get('turbo_design_lbf_per_engine', 0.0):,.0f} lbf/engine")
    print(f"turbo design thrust:  {profile.get('turbo_design_kN_per_engine', 0.0):,.2f} kN/engine")
    print(f"ramjet mdot total:    {profile.get('ramjet_design_mdot_kg_s_total', 0.0):,.2f} kg/s")
    print(f"ramjet mdot/engine:   {profile.get('ramjet_design_mdot_kg_s_per_engine', 0.0):,.2f} kg/s")

    print("\nGenerated sizing segments")
    print("-------------------------")
    for s in segments:
        print(
            f"{s.name:<52s} "
            f"{s.propulsion_mode:<9s} "
            f"{s.fuel_type:<5s} "
            f"M_drag={s.mach_drag:>6.3f} "
            f"h_drag={s.altitude_drag:>9.1f} m "
            f"T={s.T:>11.1f} N "
            f"D_override={s.drag_override_N if s.drag_override_N is not None else 0.0:>11.1f} N "
            f"mode={s.mode:<6s}"
        )

    S_plan, W_to, result = converge_S_plan_and_TOGW(
        tau=0.12,
        configuration="blended_body",
        S_plan_guess=330.15,
        I_str=21.0,
        I_tps=6.0,
        KIT=1.0,
        I_tank=4.0,
        rho_LH2=70.0,
        rho_JetA=800.0,
        k_pf=1.0,
        I_sub=0.04,
        W_prop=17_053.50436,
        V_prop=166.0221082,
        W_payload=7_000.0,
        rho_payload=100.0,
        segments=segments,
        k_rf=0.06,
        W_to_guess=74_860.0,
        rho_str=2700.0,
        rho_tps=2500.0,
        rho_tank_str=2700.0,
        K_lg=0.01,
        K_sub=0.02,
        K_void=0.2,
        volume_tol=1.0,
        weight_tol=1.0,
        S_plan_relaxation=0.5,
        weight_relaxation=0.6,
        max_size_iter=300,
        max_weight_iter=500,
    )

    print("\nConverged values")
    print("----------------")
    print(f"Configuration:       {result['configuration']}")
    print(f"tau:                 {result['tau']:.3f}")
    print(f"K_W:                 {result['K_W']:.3f}")
    print(f"S_plan:              {S_plan:.3f} m²")
    print(f"TOGW:                {W_to:.3f} kg")
    print(f"V_tot_available:     {result['V_tot_available']:.3f} m³")
    print(f"V_required:          {result['V_required']:.3f} m³")
    print(f"Volume error:        {result['volume_error']:.6f} m³")
    print(f"Weight error:        {result['weight_error']:.6f} kg")
    print(f"Wing loading:        {result['wing_loading']:.3f} kg/m²")
    print(f"Size iterations:     {result['size_iterations']}")
    print(f"Weight iterations:   {result['weight_iterations']}")

    print("\nMission fuel data")
    print("-----------------")
    print(f"Segment fractions:          {result['segment_fractions']}")
    print(f"Total mission fraction:     {result['total_mission_fraction']:.6f}")
    print(f"W_fuel_total:               {result['W_fuel']:.3f} kg")
    print(f"W_fuel_JetA:                {result['W_fuel_JetA']:.3f} kg")
    print(f"W_fuel_LH2_total:           {result['W_fuel_LH2']:.3f} kg")
    print(f"W_fuel_LH2_ramjet:          {result['W_fuel_LH2_ramjet']:.3f} kg")
    print(f"W_fuel_LH2_scramjet:        {result['W_fuel_LH2_scramjet']:.3f} kg")
    print(f"V_LH2:                      {result['V_LH2']:.3f} m³")
    print(f"V_JetA:                     {result['V_JetA']:.3f} m³")

    print("\nSegment fuel burn before reserve")
    print("--------------------------------")
    for segment_name, burn in result["segment_burns"].items():
        mode = result["segment_propulsion_modes"][segment_name]
        print(f"{segment_name:<52s} [{mode:<8s}]: {burn:.3f} kg")

    print("\nSegment drag/thrust/aero values")
    print("-------------------------------")
    for segment in segments:
        D_used = result["segment_drags"].get(segment.name, 0.0)
        aero = result["segment_aero"].get(segment.name, {})
        T_used = aero.get("T_used", segment.T)
        T_available = aero.get("T_available", segment.T)

        print(
            f"{segment.name:<52s} "
            f"D={D_used:>12.3f} N   "
            f"T_used={T_used:>12.3f} N   "
            f"T_avail={T_available:>12.3f} N   "
            f"T/D={T_used / D_used if D_used > 0 else 0.0:>8.3f}   "
            f"CL_used={aero.get('C_L_used', 0.0):>8.4f}   "
            f"CD={aero.get('C_D_calc', 0.0):>8.5f}   "
            f"L/D={aero.get('L_over_D_calc', 0.0):>8.3f}   "
            f"q={aero.get('q', 0.0):>10.1f} Pa   "
            f"M={segment.mach_drag:>6.3f}   "
            f"drag_status={aero.get('mach_polar_status', 'unknown'):>21s}   "
            f"h={segment.altitude_drag:>9.1f} m   "
            f"a_x={aero.get('axial_accel_g', 0.0):>6.3f} g   "
            f"a_y={aero.get('vertical_accel_g', 0.0):>6.3f} g   "
            f"a_tot={aero.get('total_accel_g', 0.0):>6.3f} g   "
        )

    print("\nWeight breakdown")
    print("----------------")
    print(f"W_str:               {result['W_str']:.3f} kg")
    print(f"W_tps:               {result['W_tps']:.3f} kg")
    print(f"W_landinggear:       {result['W_landinggear']:.3f} kg")
    print(f"W_prop:              {result['W_prop']:.3f} kg")
    print(f"W_tank:              {result['W_tank']:.3f} kg")
    print(f"W_sub:               {result['W_sub']:.3f} kg")
    print(f"W_payload:           {result['W_payload']:.3f} kg")
    print(f"W_fuel:              {result['W_fuel']:.3f} kg")

    print("\nVolume breakdown")
    print("----------------")
    print(f"V_structure:         {result['V_structure']:.3f} m³")
    print(f"V_tps:               {result['V_tps']:.3f} m³")
    print(f"V_landinggear:       {result['V_landinggear']:.3f} m³")
    print(f"V_prop:              {result['V_prop']:.3f} m³")
    print(f"V_tank_structure:    {result['V_tank_structure']:.3f} m³")
    print(f"V_subsystem:         {result['V_subsystem']:.3f} m³")
    print(f"V_void:              {result['V_void']:.3f} m³")
    print(f"V_payload:           {result['V_payload']:.3f} m³")
    print(f"V_tank_capacity:     {result['V_tank_capacity']:.3f} m³")