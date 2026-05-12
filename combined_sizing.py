import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator


# =============================================================================
# Basic geometry and sizing equations
# =============================================================================

def s_wet(K_W: float, S_plan: float) -> float:
    return K_W * S_plan


def volume_from_tau(tau: float, S_plan: float) -> float:
    """
    Küchemann relation:
        V_tot = tau * S_plan^1.5
    """
    return tau * S_plan**1.5


def W_landinggear(W_to: float) -> float:
    """
    Landing gear weight correlation.

    W_to is in kg.
    """
    return 0.01 * W_to**1.124


# =============================================================================
# ISA atmosphere
# =============================================================================

def isa_temperature(altitude_m: float) -> float:
    """Approximate ISA static temperature [K]."""
    if altitude_m <= 11_000.0:
        return 288.15 - 0.0065 * altitude_m
    elif altitude_m <= 20_000.0:
        return 216.65
    elif altitude_m <= 32_000.0:
        return 216.65 + 0.001 * (altitude_m - 20_000.0)
    else:
        return 228.65


def isa_pressure(altitude_m: float) -> float:
    """Approximate ISA static pressure [Pa]."""
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
    """Approximate ISA static density [kg/m^3]."""
    R = 287.05
    return isa_pressure(altitude_m) / (R * isa_temperature(altitude_m))


def speed_of_sound(altitude_m: float) -> float:
    """Approximate ISA speed of sound [m/s]."""
    gamma_air = 1.4
    R = 287.05
    T = isa_temperature(altitude_m)
    return math.sqrt(gamma_air * R * T)


def mach_to_velocity(mach: float, altitude_m: float) -> float:
    return mach * speed_of_sound(altitude_m)


def dynamic_pressure_from_mach_altitude(mach: float, altitude_m: float) -> float:
    """Dynamic pressure q = 0.5 rho V^2 [Pa]."""
    if mach <= 0:
        raise ValueError("mach must be positive for dynamic pressure calculation.")

    rho = isa_density(altitude_m)
    V = mach_to_velocity(mach, altitude_m)
    return 0.5 * rho * V**2


# =============================================================================
# PCHIP drag polar interpolation
# =============================================================================
# Your drag polar has the form:
#     C_D = a(M) C_L^2 + b(M) C_L + c(M)
# The a, b, c coefficients are interpolated with Mach number.
# =============================================================================

MACH_POLAR_DATA = np.array([0.65, 0.9, 1.1, 1.3, 2.0, 5.37, 7.38, 10.61])

A_POLAR_DATA = np.array([0.3804, 0.3418, 0.3459, 0.4006, 0.6049, 1.0314, 1.2753, 1.1948])
B_POLAR_DATA = np.array([-0.0011, 0.0100, 0.0012, 0.0037, 0.0010, 0.0145, 0.0354, 0.0962])
C_POLAR_DATA = np.array([0.0070, 0.0174, 0.0382, 0.0337, 0.0268, 0.0121, 0.0101, 0.0081])

A_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, A_POLAR_DATA)
B_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, B_POLAR_DATA)
C_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, C_POLAR_DATA)


def mach_regime(M: float) -> str:
    if M < 0.8:
        return "subsonic"
    elif M < 1.2:
        return "transonic"
    elif M < 5.0:
        return "supersonic"
    else:
        return "hypersonic"


def cd_from_mach_cl(M: float, CL: float, clamp_mach: bool = True) -> tuple[float, dict[str, float | str]]:
    """
    Calculate C_D using the interpolated Mach-dependent drag polar:

        C_D = a(M) C_L^2 + b(M) C_L + c(M)

    If clamp_mach=True, Mach numbers outside the polar range are clamped to
    the nearest available polar. This is useful for takeoff/landing Mach values
    below 0.65.
    """
    M_original = float(M)

    if clamp_mach:
        M_used = float(np.clip(M_original, MACH_POLAR_DATA.min(), MACH_POLAR_DATA.max()))
    else:
        if M_original < MACH_POLAR_DATA.min() or M_original > MACH_POLAR_DATA.max():
            raise ValueError(
                f"Mach number {M_original:.3f} outside available polar range "
                f"{MACH_POLAR_DATA.min():.2f} <= M <= {MACH_POLAR_DATA.max():.2f}"
            )
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
        "mach_regime": mach_regime(M_original),
    }


# =============================================================================
# Ramjet equations
# =============================================================================

def ramjet_temperature_ratio_max(M0: float, gamma: float = 1.29) -> float:
    """
    Combustion temperature ratio T4/T0 that gives maximum thrust per frontal area.
    """
    if M0 <= 0:
        raise ValueError("M0 must be positive.")

    numerator = (1.0 + ((gamma - 1.0) / 2.0) * M0**2) ** 3
    denominator = (1.0 + ((gamma - 1.0) / 4.0) * M0**2) ** 2
    return numerator / denominator


def ramjet_thrust_and_isp(
    M0: float,
    altitude_m: float,
    A3: float,
    gamma: float = 1.29,
    T4_T0: float | None = None,
    h_f_mass: float = 120.0e6,
    eta_thrust: float = 0.85,
    eta_isp: float = 0.60,
) -> tuple[float, float, dict[str, float]]:
    """Ramjet thrust and Isp estimate."""
    if M0 <= 0:
        raise ValueError("M0 must be positive.")
    if A3 <= 0:
        raise ValueError("A3 must be positive.")
    if h_f_mass <= 0:
        raise ValueError("h_f_mass must be positive.")

    g0 = 9.80665
    p0 = isa_pressure(altitude_m)
    a0 = speed_of_sound(altitude_m)

    if T4_T0 is None:
        T4_T0 = ramjet_temperature_ratio_max(M0, gamma)

    stagnation_factor = 1.0 + ((gamma - 1.0) / 2.0) * M0**2
    root_term = math.sqrt(T4_T0 / stagnation_factor)

    thrust_ideal = p0 * A3 * gamma * M0**2 * (root_term - 1.0)
    thrust = eta_thrust * thrust_ideal

    isp_velocity_ideal = (
        (h_f_mass / a0)
        * (gamma - 1.0)
        * M0
        / (stagnation_factor * (root_term + 1.0))
    )
    I_sp = eta_isp * isp_velocity_ideal / g0

    info = {
        "M0": M0,
        "altitude_m": altitude_m,
        "p0": p0,
        "a0": a0,
        "A3": A3,
        "gamma": gamma,
        "T4_T0": T4_T0,
        "root_term": root_term,
        "thrust_ideal": thrust_ideal,
        "thrust": thrust,
        "isp_velocity_ideal": isp_velocity_ideal,
        "I_sp": I_sp,
        "eta_thrust": eta_thrust,
        "eta_isp": eta_isp,
    }

    return thrust, I_sp, info


# =============================================================================
# K_W as function of tau and configuration
# =============================================================================

def k_w_from_tau(tau: float, configuration: str = "wing_body") -> float:
    """Wetted-to-planform area ratio K_W as a function of Küchemann tau."""
    if tau <= 0:
        raise ValueError("tau must be positive.")

    if configuration == "waverider":
        return 5632.2 * tau**4 - 3106.0 * tau**3 + 621.37 * tau**2 - 46.623 * tau + 3.8167

    elif configuration == "wing_body":
        return 473.07 * tau**4 - 366.2 * tau**3 + 110.36 * tau**2 - 9.6647 * tau + 2.9019

    elif configuration == "blended_body":
        return 18.594 * tau**2 + 0.0084 * tau + 2.4274

    else:
        raise ValueError("configuration must be 'waverider', 'wing_body', or 'blended_body'.")


# =============================================================================
# Mission segment model
# =============================================================================

@dataclass
class MissionSegment:
    """
    One mission segment.

    mode:
        "fixed"  -> directly uses fixed_fraction
        "T_gt_D" -> uses exp[-Δ(h + V²/2g) / (Isp V (1 - D/T))]
        "T_eq_D" -> uses exp[-D Δt / (Isp W)]

    Drag calculation:
        L_required = W g cos(gamma) + W g n_normal_extra
        C_L = L_required / (q S_plan)
        C_D = interpolated_drag_polar(Mach, C_L)
        D = q S_plan C_D
    """

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


def segment_lift_required(segment: MissionSegment, W_current: float) -> float:
    """
    Required aerodynamic lift [N].

    For a straight climb/descent with no normal acceleration:
        L = W g cos(gamma)
    """
    gamma_rad = math.radians(segment.flight_path_angle_deg)
    W_current_force = W_current * segment.g
    return W_current_force * (math.cos(gamma_rad) + segment.n_normal_extra)


def segment_aero_from_polar(segment: MissionSegment, W_current: float, S_plan: float) -> dict[str, Any]:
    """
    Calculate drag using the interpolated C_D-C_L polar.
    """
    if S_plan <= 0:
        raise ValueError("S_plan must be positive.")
    if segment.mach_drag <= 0.0:
        raise ValueError(f"Segment '{segment.name}' needs mach_drag > 0 for drag calculation.")

    q = dynamic_pressure_from_mach_altitude(
        mach=segment.mach_drag,
        altitude_m=segment.altitude_drag,
    )

    if q <= 0:
        raise ValueError(f"Segment '{segment.name}' dynamic pressure is non-positive.")

    L_required = segment_lift_required(segment, W_current)
    C_L_calc = L_required / (q * S_plan)

    C_D_calc, polar_info = cd_from_mach_cl(
        M=segment.mach_drag,
        CL=C_L_calc,
        clamp_mach=True,
    )

    D_calc = q * S_plan * C_D_calc

    return {
        "D": D_calc,
        "L_required": L_required,
        "q": q,
        "C_L_calc": C_L_calc,
        "C_D_calc": C_D_calc,
        "L_over_D_calc": C_L_calc / C_D_calc if C_D_calc > 0 else 0.0,
        **polar_info,
    }


def segment_weight_fraction(
    segment: MissionSegment,
    W_current: float,
    S_plan: float,
) -> tuple[float, float, dict[str, Any]]:
    """
    Calculate W_i / W_{i-1} for one mission segment.
    """
    aero_info = segment_aero_from_polar(segment, W_current, S_plan)
    D_used = aero_info["D"]

    if segment.mode == "fixed":
        fraction = segment.fixed_fraction

    elif segment.mode == "T_gt_D":
        if segment.T <= D_used:
            raise ValueError(
                f"For segment '{segment.name}', mode='T_gt_D' requires T > D. "
                f"Current T={segment.T:.3f} N, D={D_used:.3f} N."
            )
        if segment.I_sp <= 0:
            raise ValueError(f"For segment '{segment.name}', I_sp must be positive.")
        if segment.V_average <= 0:
            raise ValueError(f"For segment '{segment.name}', V_average must be positive.")

        delta_energy_height = segment.delta_h + (segment.V_final**2 - segment.V_initial**2) / (2.0 * segment.g)

        exponent = -delta_energy_height / (
            segment.I_sp
            * segment.V_average
            * (1.0 - D_used / segment.T)
        )

        fraction = math.exp(exponent)

    elif segment.mode == "T_eq_D":
        if segment.I_sp <= 0:
            raise ValueError(f"For segment '{segment.name}', I_sp must be positive.")
        if W_current <= 0:
            raise ValueError(f"For segment '{segment.name}', W_current must be positive.")

        W_current_force = W_current * segment.g
        exponent = -(D_used * segment.delta_t) / (segment.I_sp * W_current_force)
        fraction = math.exp(exponent)

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
) -> tuple[float, float, float, float, float, list[float], float, dict[str, float], dict[str, str], dict[str, float], dict[str, dict[str, Any]]]:
    """
    Calculate total fuel mass and split into Jet-A and LH2.
    """
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
    """Tank capacity volume for separate LH2 and Jet-A masses."""
    if rho_LH2 <= 0 or rho_JetA <= 0:
        raise ValueError("Fuel densities must be positive.")
    if k_pf <= 0:
        raise ValueError("k_pf must be positive.")

    V_LH2 = W_fuel_LH2 / (rho_LH2 * k_pf)
    V_JetA = W_fuel_JetA / (rho_JetA * k_pf)
    V_total = V_LH2 + V_JetA

    return V_total, V_LH2, V_JetA


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

def find_ascent_state_at_mach(
    target_mach: float,
    h_cruise: float,
    gamma_deg: float,
    acc_tot: float,
) -> dict[str, float | bool]:
    """
    Find the state during constant-gamma ascent where Mach = target_mach.
    """
    gamma_rad = math.radians(gamma_deg)

    acc_x = acc_tot * math.cos(gamma_rad)
    acc_y = acc_tot * math.sin(gamma_rad)

    if acc_y <= 0:
        raise ValueError("acc_y must be positive.")

    t_to_cruise = math.sqrt(2.0 * h_cruise / acc_y)

    def mach_at_time(t: float) -> float:
        h = 0.5 * acc_y * t**2
        V = acc_tot * t
        return V / speed_of_sound(h)

    mach_at_cruise_height = mach_at_time(t_to_cruise)

    if mach_at_cruise_height < target_mach:
        return {
            "target_reached": False,
            "t": t_to_cruise,
            "h": h_cruise,
            "V": acc_tot * t_to_cruise,
            "Vx": acc_x * t_to_cruise,
            "Vy": acc_y * t_to_cruise,
            "x": 0.5 * acc_x * t_to_cruise**2,
            "mach": mach_at_cruise_height,
        }

    t_low = 0.0
    t_high = t_to_cruise

    for _ in range(100):
        t_mid = 0.5 * (t_low + t_high)
        if mach_at_time(t_mid) < target_mach:
            t_low = t_mid
        else:
            t_high = t_mid

    t = 0.5 * (t_low + t_high)
    h = 0.5 * acc_y * t**2
    V = acc_tot * t
    Vx = acc_x * t
    Vy = acc_y * t
    x = 0.5 * acc_x * t**2

    return {
        "target_reached": True,
        "t": t,
        "h": h,
        "V": V,
        "Vx": Vx,
        "Vy": Vy,
        "x": x,
        "mach": V / speed_of_sound(h),
    }


# =============================================================================
# Standalone drag calculation using the same PCHIP polar
# =============================================================================

def drag_from_speed_altitude(
    W_current: float,
    S_plan: float,
    altitude_m: float,
    velocity_m_s: float,
    flight_path_angle_deg: float = 0.0,
    n_normal_extra: float = 0.0,
    g: float = 9.81,
) -> dict[str, Any]:
    """
    Standalone drag calculator using the interpolated Mach-dependent drag polar.
    """
    if W_current <= 0:
        raise ValueError("W_current must be positive.")
    if S_plan <= 0:
        raise ValueError("S_plan must be positive.")
    if altitude_m < 0:
        raise ValueError("altitude_m must be non-negative.")
    if velocity_m_s <= 0:
        raise ValueError("velocity_m_s must be positive.")

    rho = isa_density(altitude_m)
    a_sound = speed_of_sound(altitude_m)
    mach = velocity_m_s / a_sound
    q = 0.5 * rho * velocity_m_s**2

    gamma_rad = math.radians(flight_path_angle_deg)
    W_force = W_current * g
    L_required = W_force * (math.cos(gamma_rad) + n_normal_extra)

    C_L_calc = L_required / (q * S_plan)
    C_D_calc, polar_info = cd_from_mach_cl(mach, C_L_calc, clamp_mach=True)
    D_calc = q * S_plan * C_D_calc

    return {
        "altitude_m": altitude_m,
        "velocity_m_s": velocity_m_s,
        "mach": mach,
        "rho": rho,
        "q": q,
        "L_required": L_required,
        "C_L_calc": C_L_calc,
        "C_D_calc": C_D_calc,
        "L_over_D_calc": C_L_calc / C_D_calc if C_D_calc > 0 else 0.0,
        "D": D_calc,
        **polar_info,
    }


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":

    # Mission profile
    gamma_mission = 10.0
    h0 = 0.0
    h_cruise = 35_000.0
    acc_tot = 0.15 * 9.81

    M_turbo_to_ram = 2.5
    M_cruise = 5.0
    cruise_time = 90.0 * 60.0

    gamma_rad = math.radians(gamma_mission)
    acc_x = acc_tot * math.cos(gamma_rad)
    acc_y = acc_tot * math.sin(gamma_rad)

    a_cruise = speed_of_sound(h_cruise)
    V_cruise = M_cruise * a_cruise

    t_to_cruise = math.sqrt(2.0 * h_cruise / acc_y)

    dv_x_to_cruise = acc_x * t_to_cruise
    dv_y_to_cruise = acc_y * t_to_cruise

    V_at_cruise_height = math.sqrt(dv_x_to_cruise**2 + dv_y_to_cruise**2)
    M_at_cruise_height = V_at_cruise_height / a_cruise
    M_horizontal_start = dv_x_to_cruise / a_cruise

    dx_to_cruise = 0.5 * acc_x * t_to_cruise**2
    dx_hor_acc = (V_cruise**2 - dv_x_to_cruise**2) / (2.0 * acc_tot)

    cruise_cond_start_x = dx_to_cruise + dx_hor_acc
    cruise_range = cruise_time * V_cruise
    cruise_cond_end_x = cruise_cond_start_x + cruise_range

    switch_state = find_ascent_state_at_mach(
        target_mach=M_turbo_to_ram,
        h_cruise=h_cruise,
        gamma_deg=gamma_mission,
        acc_tot=acc_tot,
    )

    print("\nMission profile estimate")
    print("------------------------")
    print(f"gamma:                      {gamma_mission:.2f} deg")
    print(f"h_cruise:                   {h_cruise:.1f} m")
    print(f"a_cruise:                   {a_cruise:.3f} m/s")
    print(f"V_cruise:                   {V_cruise:.3f} m/s")
    print(f"M_at_cruise_height:         {M_at_cruise_height:.3f}")
    print(f"M_horizontal_start:         {M_horizontal_start:.3f}")
    print(f"M2.5 reached during ascent: {switch_state['target_reached']}")
    print(f"h_at_M2.5:                  {switch_state['h']:.3f} m")
    print(f"V_at_M2.5:                  {switch_state['V']:.3f} m/s")
    print(f"dx_to_cruise:               {dx_to_cruise / 1000:.3f} km")
    print(f"dx_horizontal_acceleration: {dx_hor_acc / 1000:.3f} km")
    print(f"cruise_range:               {cruise_range / 1000:.3f} km")

    # Ramjet model
    if switch_state["target_reached"]:
        M_ramjet_avg = 0.5 * (M_turbo_to_ram + M_cruise)
        h_ramjet_avg = 0.5 * (switch_state["h"] + h_cruise)
    else:
        M_ramjet_avg = 0.5 * (M_horizontal_start + M_cruise)
        h_ramjet_avg = h_cruise

    A3_ramjet = 2 * 0.7739

    T_ramjet_calc, Isp_ramjet_calc, ramjet_info = ramjet_thrust_and_isp(
        M0=M_ramjet_avg,
        altitude_m=h_ramjet_avg,
        A3=A3_ramjet,
        gamma=1.29,
        T4_T0=None,
        h_f_mass=120.0e6,
        eta_thrust=0.9,
        eta_isp=0.9,
    )

    USE_RAMJET_THRUST_OVERRIDE = True
    RAMJET_THRUST_OVERRIDE = 100_000_000.0

    T_ramjet_physics = T_ramjet_calc
    if USE_RAMJET_THRUST_OVERRIDE:
        T_ramjet_calc = RAMJET_THRUST_OVERRIDE

    print("\nRamjet model estimate")
    print("---------------------")
    print(f"M_ramjet_avg:     {M_ramjet_avg:.3f}")
    print(f"h_ramjet_avg:     {h_ramjet_avg:.3f} m")
    print(f"A3_ramjet:        {A3_ramjet:.4f} m²")
    print(f"T_ramjet_physics: {T_ramjet_physics:.3f} N")
    print(f"T_ramjet_used:    {T_ramjet_calc:.3f} N")
    print(f"Isp_ramjet_calc:  {Isp_ramjet_calc:.3f} s")

    # Propulsion thrust values
    T_turbojet_operating = 100_035_000.0
    T_takeoff = T_turbojet_operating
    T_ramjet_acceleration = T_ramjet_calc
    T_scramjet_cruise = 0.0

    # Mission segments
    segments = [
        MissionSegment(
            name="1_takeoff",
            mode="fixed",
            fuel_type="JetA",
            propulsion_mode="turbojet",
            fixed_fraction=0.990,
            mach_drag=0.2,
            altitude_drag=h0,
            flight_path_angle_deg=0.0,
            T=T_takeoff,
        ),
    ]

    if switch_state["target_reached"]:
        segments.append(
            MissionSegment(
                name="2a_ascent_to_M2.5_turbojet",
                mode="T_gt_D",
                fuel_type="JetA",
                propulsion_mode="turbojet",
                delta_h=switch_state["h"],
                V_initial=0.0,
                V_final=switch_state["V"],
                V_average=0.5 * switch_state["V"],
                I_sp=2100.0,
                mach_drag=1.25,
                altitude_drag=0.5 * switch_state["h"],
                flight_path_angle_deg=gamma_mission,
                T=T_turbojet_operating,
            )
        )

        segments.append(
            MissionSegment(
                name="2b_ascent_M2.5_to_35km_ramjet",
                mode="T_gt_D",
                fuel_type="LH2",
                propulsion_mode="ramjet",
                delta_h=h_cruise - switch_state["h"],
                V_initial=switch_state["V"],
                V_final=V_at_cruise_height,
                V_average=0.5 * (switch_state["V"] + V_at_cruise_height),
                I_sp=3500.0,
                mach_drag=0.5 * (M_turbo_to_ram + M_at_cruise_height),
                altitude_drag=0.5 * (switch_state["h"] + h_cruise),
                flight_path_angle_deg=gamma_mission,
                T=T_ramjet_acceleration,
            )
        )

    else:
        segments.append(
            MissionSegment(
                name="2_ascent_to_35km_turbojet",
                mode="T_gt_D",
                fuel_type="JetA",
                propulsion_mode="turbojet",
                delta_h=h_cruise,
                V_initial=0.0,
                V_final=V_at_cruise_height,
                V_average=0.5 * V_at_cruise_height,
                I_sp=2100.0,
                mach_drag=0.5 * M_at_cruise_height,
                altitude_drag=0.5 * h_cruise,
                flight_path_angle_deg=gamma_mission,
                T=T_turbojet_operating,
            )
        )

    V_horizontal_accel_start = dv_x_to_cruise

    if V_horizontal_accel_start >= V_cruise:
        raise ValueError(
            "Horizontal speed at cruise altitude is already >= cruise speed. "
            "No horizontal acceleration segment to Mach 5 is needed."
        )

    segments.append(
        MissionSegment(
            name="3_horizontal_accel_to_M5_ramjet",
            mode="T_gt_D",
            fuel_type="LH2",
            propulsion_mode="ramjet",
            delta_h=0.0,
            V_initial=V_horizontal_accel_start,
            V_final=V_cruise,
            V_average=0.5 * (V_horizontal_accel_start + V_cruise),
            I_sp=3500.0,
            mach_drag=0.5 * (M_horizontal_start + M_cruise),
            altitude_drag=h_cruise,
            flight_path_angle_deg=0.0,
            T=T_ramjet_acceleration,
        )
    )

    segments.append(
        MissionSegment(
            name="4_cruise_M5_35km_scramjet",
            mode="T_eq_D",
            fuel_type="LH2",
            propulsion_mode="scramjet",
            I_sp=1500.0,
            mach_drag=M_cruise,
            altitude_drag=h_cruise,
            flight_path_angle_deg=0.0,
            T=T_scramjet_cruise,
            delta_t=cruise_time,
        )
    )

    segments.extend(
        [
            MissionSegment(
                name="5_unpowered_descent",
                mode="fixed",
                fuel_type="none",
                propulsion_mode="none",
                fixed_fraction=1.0,
                mach_drag=2.0,
                altitude_drag=0.5 * h_cruise,
                flight_path_angle_deg=-5.0,
            ),
            MissionSegment(
                name="6_landing",
                mode="fixed",
                fuel_type="JetA",
                propulsion_mode="turbojet",
                fixed_fraction=0.997,
                mach_drag=0.25,
                altitude_drag=h0,
                flight_path_angle_deg=0.0,
                T=T_takeoff,
            ),
        ]
    )

    # Run sizing
    S_plan, W_to, result = converge_S_plan_and_TOGW(
        tau=0.14,
        configuration="blended_body",
        S_plan_guess=450.0,
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
        W_to_guess=100_000.0,
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

    # Print results
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
        print(f"{segment_name:<42s} [{mode:<8s}]: {burn:.3f} kg")

    print("\nSegment drag/thrust/aero values")
    print("-------------------------------")
    for segment in segments:
        D_used = result["segment_drags"].get(segment.name, 0.0)
        aero = result["segment_aero"].get(segment.name, {})

        T_display = D_used if segment.mode == "T_eq_D" else segment.T

        print(
            f"{segment.name:<42s} "
            f"D={D_used:>12.3f} N   "
            f"T={T_display:>12.3f} N   "
            f"CL={aero.get('C_L_calc', 0.0):>8.4f}   "
            f"CD={aero.get('C_D_calc', 0.0):>8.5f}   "
            f"L/D={aero.get('L_over_D_calc', 0.0):>8.3f}   "
            f"q={aero.get('q', 0.0):>10.1f} Pa   "
            f"M={segment.mach_drag:>5.2f}   "
            f"polar_M={aero.get('mach_used_for_polar', segment.mach_drag):>5.2f}   "
            f"regime={aero.get('mach_regime', 'unknown'):>10s}   "
            f"h={segment.altitude_drag:>8.1f} m"
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

    # Standalone custom drag calculation
    custom_altitude_m = 35_000.0
    custom_velocity_m_s = 1_500.0
    custom_mass_kg = W_to
    custom_gamma_deg = 0.0

    custom_drag = drag_from_speed_altitude(
        W_current=custom_mass_kg,
        S_plan=S_plan,
        altitude_m=custom_altitude_m,
        velocity_m_s=custom_velocity_m_s,
        flight_path_angle_deg=custom_gamma_deg,
    )

    print("\nCustom drag calculation")
    print("-----------------------")
    print(f"Altitude:       {custom_drag['altitude_m']:.1f} m")
    print(f"Velocity:       {custom_drag['velocity_m_s']:.1f} m/s")
    print(f"Mach:           {custom_drag['mach']:.3f}")
    print(f"Mach polar used:{custom_drag['mach_used_for_polar']:.3f}")
    print(f"Regime:         {custom_drag['mach_regime']}")
    print(f"rho:            {custom_drag['rho']:.6f} kg/m³")
    print(f"q:              {custom_drag['q']:.1f} Pa")
    print(f"L_required:     {custom_drag['L_required']:.3f} N")
    print(f"CL:             {custom_drag['C_L_calc']:.4f}")
    print(f"CD:             {custom_drag['C_D_calc']:.5f}")
    print(f"L/D:            {custom_drag['L_over_D_calc']:.3f}")
    print(f"Drag:           {custom_drag['D']:.3f} N")
