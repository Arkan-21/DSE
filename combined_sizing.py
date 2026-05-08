import numpy as np
import math
from dataclasses import dataclass


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


def speed_of_sound(altitude_m: float) -> float:
    """
    Approximate ISA speed of sound.
    """
    gamma = 1.4
    R = 287.05

    if altitude_m <= 11_000.0:
        T = 288.15 - 0.0065 * altitude_m
    elif altitude_m <= 20_000.0:
        T = 216.65
    elif altitude_m <= 32_000.0:
        T = 216.65 + 0.001 * (altitude_m - 20_000.0)
    else:
        T = 228.65

    return math.sqrt(gamma * R * T)


def mach_to_velocity(mach: float, altitude_m: float) -> float:
    return mach * speed_of_sound(altitude_m)


def isa_pressure(altitude_m: float) -> float:
    """
    Approximate ISA static pressure [Pa].
    """
    g0 = 9.80665
    R = 287.05

    if altitude_m <= 11_000.0:
        T0 = 288.15
        L = -0.0065
        T = T0 + L * altitude_m
        p = 101325.0 * (T / T0) ** (-g0 / (L * R))

    elif altitude_m <= 20_000.0:
        T = 216.65
        p11 = 22632.06
        p = p11 * math.exp(-g0 * (altitude_m - 11_000.0) / (R * T))

    elif altitude_m <= 32_000.0:
        T20 = 216.65
        p20 = 5474.89
        L = 0.001
        T = T20 + L * (altitude_m - 20_000.0)
        p = p20 * (T / T20) ** (-g0 / (L * R))

    else:
        T = 228.65
        p32 = 868.02
        p = p32 * math.exp(-g0 * (altitude_m - 32_000.0) / (R * T))

    return p


# =============================================================================
# Ramjet equations
# =============================================================================

def ramjet_temperature_ratio_max(M0: float, gamma: float = 1.29) -> float:
    """
    Combustion temperature ratio T4/T0 that gives maximum thrust per frontal area.

    (T4/T0)_max =
        [1 + ((gamma - 1)/2) M0²]^3
        /
        [1 + ((gamma - 1)/4) M0²]^2
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
    """
    Ramjet thrust and Isp estimate.

    Thrust:
        T = p0 A3 gamma M0² [
                sqrt((T4/T0)/(1 + ((gamma-1)/2)M0²)) - 1
            ]

    Isp relation:
        Isp_velocity * a0 / H_f =
            ((gamma - 1) M0 / [1 + ((gamma - 1)/2)M0²])
            *
            [
                sqrt((T4/T0)/(1 + ((gamma-1)/2)M0²)) + 1
            ]

    Returns:
        thrust [N]
        I_sp [s]
        info dictionary
    """

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

    thrust_ideal = (
        p0
        * A3
        * gamma
        * M0**2
        * (root_term - 1.0)
    )

    thrust = eta_thrust * thrust_ideal

    # Your corrected version of the Isp expression
    isp_velocity_ideal = (
        (h_f_mass / a0)
        * (gamma - 1.0) * M0
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
    """
    Wetted-to-planform area ratio K_W as a function of Küchemann tau.
    """
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")

    if configuration == "waverider":
        return (
            5632.2 * tau**4
            - 3106.0 * tau**3
            + 621.37 * tau**2
            - 46.623 * tau
            + 3.8167
        )

    elif configuration == "wing_body":
        return (
            473.07 * tau**4
            - 366.2 * tau**3
            + 110.36 * tau**2
            - 9.6647 * tau
            + 2.9019
        )

    elif configuration == "blended_body":
        return (
            18.594 * tau**2
            + 0.0084 * tau
            + 2.4274
        )

    else:
        raise ValueError(
            "configuration must be one of: "
            "'waverider', 'wing_body', or 'blended_body'."
        )
    """
    return 2.407    


# =============================================================================
# Mission segment fuel calculation
# =============================================================================

@dataclass
class MissionSegment:
    """
    One mission segment.

    mode:
        "fixed"  -> directly uses fixed_fraction
        "T_gt_D" -> uses exp[-Δ(h + V²/2g) / (Isp V (1 - D/T))]
        "T_eq_D" -> uses exp[-D Δt / (Isp W)]

    fuel_type:
        "JetA"
        "LH2"
        "none"

    propulsion_mode:
        "turbojet"
        "ramjet"
        "scramjet"
        "none"
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
    D: float = 0.0
    T: float = 0.0

    delta_t: float = 0.0

    fixed_fraction: float = 1.0


def segment_weight_fraction(segment: MissionSegment, W_current: float) -> float:
    """
    Calculate W_i / W_{i-1} for one mission segment.
    """

    if segment.mode == "fixed":
        fraction = segment.fixed_fraction

    elif segment.mode == "T_gt_D":
        if segment.T <= segment.D:
            raise ValueError(
                f"For segment '{segment.name}', mode='T_gt_D' requires T > D. "
                f"Current T={segment.T:.3f} N, D={segment.D:.3f} N."
            )
        if segment.I_sp <= 0:
            raise ValueError(f"For segment '{segment.name}', I_sp must be positive.")
        if segment.V_average <= 0:
            raise ValueError(f"For segment '{segment.name}', V_average must be positive.")

        delta_energy_height = (
            segment.delta_h
            + (segment.V_final**2 - segment.V_initial**2) / (2.0 * segment.g)
        )

        exponent = -delta_energy_height / (
            segment.I_sp
            * segment.V_average
            * (1.0 - segment.D / segment.T)
        )

        fraction = math.exp(exponent)

    elif segment.mode == "T_eq_D":
        if segment.I_sp <= 0:
            raise ValueError(f"For segment '{segment.name}', I_sp must be positive.")
        if W_current <= 0:
            raise ValueError(f"For segment '{segment.name}', W_current must be positive.")

        W_current_force = W_current * segment.g

        exponent = -(segment.D * segment.delta_t) / (
            segment.I_sp * W_current_force
        )

        fraction = math.exp(exponent)

    else:
        raise ValueError("mode must be 'fixed', 'T_gt_D', or 'T_eq_D'.")

    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"Invalid segment fraction in segment '{segment.name}': {fraction}"
        )

    return fraction


def fuel_masses_from_segments(
    W_to: float,
    segments: list[MissionSegment],
    k_rf: float,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    list[float],
    float,
    dict[str, float],
    dict[str, float],
]:
    """
    Calculate total fuel mass and split into:
        Jet-A
        LH2 total
        LH2 ramjet
        LH2 scramjet
    """

    if W_to <= 0:
        raise ValueError("W_to must be positive.")
    if k_rf < 0:
        raise ValueError("k_rf must be non-negative.")

    W_current = W_to

    segment_fractions = []
    segment_burns = {}
    segment_propulsion_modes = {}

    mission_burn_JetA = 0.0
    mission_burn_LH2_ramjet = 0.0
    mission_burn_LH2_scramjet = 0.0

    for segment in segments:
        fraction = segment_weight_fraction(segment, W_current)
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
                    f"Segment '{segment.name}' uses LH2, so propulsion_mode "
                    "must be 'ramjet' or 'scramjet'."
                )

        elif segment.fuel_type == "none":
            pass

        else:
            raise ValueError(
                f"fuel_type for segment '{segment.name}' must be "
                "'LH2', 'JetA', or 'none'."
            )

        segment_fractions.append(fraction)
        segment_burns[segment.name] = burned_mass
        segment_propulsion_modes[segment.name] = segment.propulsion_mode

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
    )


def tank_volume_two_fuels(
    W_fuel_LH2: float,
    W_fuel_JetA: float,
    rho_LH2: float,
    rho_JetA: float,
    k_pf: float,
) -> tuple[float, float, float]:
    """
    Tank capacity volume for separate LH2 and Jet-A masses.
    """

    if rho_LH2 <= 0 or rho_JetA <= 0:
        raise ValueError("Fuel densities must be positive.")
    if k_pf <= 0:
        raise ValueError("k_pf must be positive.")

    V_LH2 = W_fuel_LH2 / (rho_LH2 * k_pf)
    V_JetA = W_fuel_JetA / (rho_JetA * k_pf)

    V_total = V_LH2 + V_JetA

    return V_total, V_LH2, V_JetA


# =============================================================================
# Inner loop: converge TOGW for fixed S_plan
# =============================================================================

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
    max_weight_iter: int = 200,
) -> tuple[float, dict]:

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
        ) = fuel_masses_from_segments(
            W_to=W_to,
            segments=segments,
            k_rf=k_rf,
        )

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

        W_to_required = (
            W_str
            + W_tps
            + W_lg
            + W_prop
            + W_tank
            + W_sub
            + W_payload
            + W_fuel_total
        )

        weight_error = W_to_required - W_to

        if abs(weight_error) < weight_tol:
            result = {
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
                "total_mission_fraction": total_mission_fraction,
            }

            return W_to, result

        W_to_new = W_to + weight_relaxation * weight_error

        if W_to_new <= 0:
            raise RuntimeError("TOGW became non-physical.")

        W_to = W_to_new

    raise RuntimeError("TOGW did not converge.")


# =============================================================================
# Volume calculation
# =============================================================================

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

    V_tank_structure = (
        (1.0 - KIT)
        * I_tank
        * V_tank_capacity
        / rho_tank_str
    )

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


# =============================================================================
# Outer loop: converge S_plan and TOGW
# =============================================================================

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
    max_size_iter: int = 100,
    max_weight_iter: int = 200,
) -> tuple[float, float, dict]:

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
            result = {
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

            return S_plan, W_to, result

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
# Run
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Mission profile: Mach 8, 30 km, 2000 km range
    #
    # Propulsion mode rule:
    #   turbojet : M < 3
    #   ramjet   : 3 <= M < 6
    #   scramjet : M >= 6
    # -------------------------------------------------------------------------

    h0 = 0.0
    h10 = 10_000.0
    h30 = 30_000.0

    V_M07_h0 = mach_to_velocity(0.7, h0)
    V_M09_h10 = mach_to_velocity(0.9, h10)
    V_M17_h10 = mach_to_velocity(1.7, h10)
    V_M8_h30 = mach_to_velocity(8.0, h30)

    # Internal split points for segment 5
    M_start_seg5 = 1.7
    M_end_seg5 = 8.0

    def interpolate_altitude_from_mach(M: float) -> float:
        return h10 + (M - M_start_seg5) / (M_end_seg5 - M_start_seg5) * (h30 - h10)

    h_M3 = interpolate_altitude_from_mach(3.0)
    h_M6 = interpolate_altitude_from_mach(6.0)

    V_M3_hM3 = mach_to_velocity(3.0, h_M3)
    V_M6_hM6 = mach_to_velocity(6.0, h_M6)

    cruise_range = 2_000_000.0
    cruise_time = cruise_range / V_M8_h30

    # -------------------------------------------------------------------------
    # Ramjet model for M3 to M6 segment
    # -------------------------------------------------------------------------

    M_ramjet_avg = 4.5
    h_ramjet_avg = 0.5 * (h_M3 + h_M6)

    A3_ramjet = 0.7739  # m²

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

    print("\nRamjet model estimate")
    print("---------------------")
    print(f"M_ramjet_avg:   {M_ramjet_avg:.2f}")
    print(f"h_ramjet_avg:   {h_ramjet_avg:.1f} m")
    print(f"A3_ramjet:      {A3_ramjet:.4f} m²")
    print(f"T4/T0:          {ramjet_info['T4_T0']:.3f}")
    print(f"p0:             {ramjet_info['p0']:.3f} Pa")
    print(f"T_ramjet:       {T_ramjet_calc:.3f} N")
    print(f"Isp_ramjet:     {Isp_ramjet_calc:.3f} s")

    # -------------------------------------------------------------------------
    # Propulsion thrust values [N]
    # -------------------------------------------------------------------------

    T_turbojet_operating = 133_446.6

    T_scramjet_acceleration = 72_930.0  # M6 -> M8 acceleration/climb
    T_scramjet_cruise = 68_640.0        # Mach 8 cruise

    # -------------------------------------------------------------------------
    # Segment-specific drag values [N]
    #
    # Each T_gt_D segment must satisfy T > D.
    # Cruise uses T_eq_D, so D_cruise_M8_30km = T_scramjet_cruise.
    # -------------------------------------------------------------------------

    D_takeoff = 0.0                         # not used, fixed fraction
    D_accel_M07 = 100_000.0
    D_accel_M09_climb_10km = 100_000.0
    D_accel_M17 = 100_000.0
    D_climb_M17_to_M3 = 100_000.0
    D_climb_M3_to_M6 = 60_000.0
    D_climb_M6_to_M8 = 60_000.0
    D_cruise_M8_30km = T_scramjet_cruise
    D_landing = 0.0                         # not used, fixed fraction

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------

    if T_turbojet_operating <= max(
        D_accel_M07,
        D_accel_M09_climb_10km,
        D_accel_M17,
        D_climb_M17_to_M3,
    ):
        raise ValueError(
            "Turbojet operating thrust must be larger than all turbojet "
            "T_gt_D segment drag values."
        )

    if T_ramjet_calc <= D_climb_M3_to_M6:
        raise ValueError(
            f"Ramjet thrust must be larger than D_climb_M3_to_M6. "
            f"T_ramjet_calc={T_ramjet_calc:.3f} N, "
            f"D_climb_M3_to_M6={D_climb_M3_to_M6:.3f} N."
        )

    if T_scramjet_acceleration <= D_climb_M6_to_M8:
        raise ValueError(
            f"Scramjet acceleration thrust must be larger than D_climb_M6_to_M8. "
            f"T_scramjet_acceleration={T_scramjet_acceleration:.3f} N, "
            f"D_climb_M6_to_M8={D_climb_M6_to_M8:.3f} N."
        )

    if abs(T_scramjet_cruise - D_cruise_M8_30km) > 1e-9:
        raise ValueError(
            "For T_eq_D cruise, set D_cruise_M8_30km equal to T_scramjet_cruise."
        )

    # -------------------------------------------------------------------------
    # Mission segments
    # -------------------------------------------------------------------------

    segments = [
        MissionSegment(
            name="1_takeoff",
            mode="fixed",
            fuel_type="JetA",
            propulsion_mode="turbojet",
            fixed_fraction=0.990,
        ),

        MissionSegment(
            name="2_accel_to_M0.7",
            mode="T_gt_D",
            fuel_type="JetA",
            propulsion_mode="turbojet",
            delta_h=0.0,
            V_initial=0.0,
            V_final=V_M07_h0,
            V_average=0.5 * V_M07_h0,
            I_sp=2200.0,
            D=D_accel_M07,
            T=T_turbojet_operating,
        ),

        MissionSegment(
            name="3_accel_to_M0.9_climb_10km",
            mode="T_gt_D",
            fuel_type="JetA",
            propulsion_mode="turbojet",
            delta_h=10_000.0,
            V_initial=V_M07_h0,
            V_final=V_M09_h10,
            V_average=0.5 * (V_M07_h0 + V_M09_h10),
            I_sp=2000.0,
            D=D_accel_M09_climb_10km,
            T=T_turbojet_operating,
        ),

        MissionSegment(
            name="4_accel_to_M1.7",
            mode="T_gt_D",
            fuel_type="JetA",
            propulsion_mode="turbojet",
            delta_h=0.0,
            V_initial=V_M09_h10,
            V_final=V_M17_h10,
            V_average=0.5 * (V_M09_h10 + V_M17_h10),
            I_sp=1600.0,
            D=D_accel_M17,
            T=T_turbojet_operating,
        ),

        MissionSegment(
            name="5a_climb_M1.7_to_M3_turbojet",
            mode="T_gt_D",
            fuel_type="JetA",
            propulsion_mode="turbojet",
            delta_h=h_M3 - h10,
            V_initial=V_M17_h10,
            V_final=V_M3_hM3,
            V_average=0.5 * (V_M17_h10 + V_M3_hM3),
            I_sp=1400.0,
            D=D_climb_M17_to_M3,
            T=T_turbojet_operating,
        ),

        MissionSegment(
            name="5b_climb_M3_to_M6_ramjet",
            mode="T_gt_D",
            fuel_type="LH2",
            propulsion_mode="ramjet",
            delta_h=h_M6 - h_M3,
            V_initial=V_M3_hM3,
            V_final=V_M6_hM6,
            V_average=0.5 * (V_M3_hM3 + V_M6_hM6),
            I_sp=Isp_ramjet_calc,
            D=D_climb_M3_to_M6,
            T=T_ramjet_calc,
        ),

        MissionSegment(
            name="5c_climb_M6_to_M8_scramjet",
            mode="T_gt_D",
            fuel_type="LH2",
            propulsion_mode="scramjet",
            delta_h=h30 - h_M6,
            V_initial=V_M6_hM6,
            V_final=V_M8_h30,
            V_average=0.5 * (V_M6_hM6 + V_M8_h30),
            I_sp=3600.0,
            D=D_climb_M6_to_M8,
            T=T_scramjet_acceleration,
        ),

        MissionSegment(
            name="6_cruise_M8_30km_scramjet",
            mode="T_eq_D",
            fuel_type="LH2",
            propulsion_mode="scramjet",
            I_sp=3300.0,
            D=D_cruise_M8_30km,
            T=T_scramjet_cruise,
            delta_t=cruise_time,
        ),

        MissionSegment(
            name="7_unpowered_descent",
            mode="fixed",
            fuel_type="none",
            propulsion_mode="none",
            fixed_fraction=1.0,
        ),

        MissionSegment(
            name="8_landing",
            mode="fixed",
            fuel_type="JetA",
            propulsion_mode="turbojet",
            fixed_fraction=0.997,
        ),
    ]

    S_plan, W_to, result = converge_S_plan_and_TOGW(
        tau=0.0446,
        configuration="blended_body",
        S_plan_guess=800.0,

        I_str=20.0,
        I_tps=6.0,

        KIT=1.0,
        I_tank=4.0,

        rho_LH2=70.0,
        rho_JetA=800.0,
        k_pf=1.0,

        I_sub=0.04,

        W_prop=54_264.5,
        V_prop=10.0,

        W_payload=10_000.0,
        rho_payload=100.0,

        segments=segments,
        k_rf=0.06,

        W_to_guess=100_000.0,

        rho_str=2700.0,
        rho_tps=500.0,
        rho_tank_str=2700.0,

        K_lg=0.01,
        K_sub=0.02,
        K_void=0.20,

        volume_tol=1.0,
        weight_tol=1.0,
        S_plan_relaxation=0.5,
        weight_relaxation=0.6,
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
        print(f"{segment_name:<38s} [{mode:<8s}]: {burn:.3f} kg")

    print("\nSegment drag/thrust values")
    print("--------------------------")
    for segment in segments:
        print(
            f"{segment.name:<38s} "
            f"D={segment.D:>10.3f} N   "
            f"T={segment.T:>10.3f} N   "
            f"Isp={segment.I_sp:>10.3f} s"
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