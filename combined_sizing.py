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


# =============================================================================
# K_W as function of tau and configuration
# =============================================================================

def k_w_from_tau(tau: float, configuration: str = "wing_body") -> float:
    """
    Wetted-to-planform area ratio K_W as a function of Küchemann tau.

    Polynomial fits:

    waverider:
        K_W = 5632.2*tau^4 - 3106*tau^3 + 621.37*tau^2 - 46.623*tau + 3.8167

    wing_body:
        K_W = 473.07*tau^4 - 366.2*tau^3 + 110.36*tau^2 - 9.6647*tau + 2.9019

    blended_body:
        K_W = 18.594*tau^2 + 0.0084*tau + 2.4274
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
        "JetA" for turbojet / conventional aircraft fuel
        "LH2" for ramjet / dual-mode ramjet
        "none" for unpowered segments

    Units:
        delta_h     [m]
        V_initial   [m/s]
        V_final     [m/s]
        V_average   [m/s]
        I_sp        [s]
        D           [N]
        T           [N]
        delta_t     [s]
        W_current   [kg], internally converted to N for T_eq_D
    """

    name: str
    mode: str
    fuel_type: str = "none"

    # For T > D segments
    delta_h: float = 0.0
    V_initial: float = 0.0
    V_final: float = 0.0
    V_average: float = 0.0
    g: float = 9.81
    I_sp: float = 0.0
    D: float = 0.0
    T: float = 0.0

    # For T = D segments
    delta_t: float = 0.0

    # For fixed historical/assumed fractions
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
                f"For segment '{segment.name}', mode='T_gt_D' requires T > D."
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

        # W_current is mass in kg.
        # The formula uses W as force, so convert kg -> N.
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
) -> tuple[float, float, float, list[float], float, dict[str, float]]:
    """
    Calculate total fuel mass and split into LH2 and Jet-A.

    Mission burn is calculated segment-by-segment:
        burned_mass_i = W_current * (1 - segment_fraction)

    Reserve/trapped fuel factor is then applied to each fuel type:
        W_fuel_type = (1 + k_rf) * mission_burn_type
    """

    if W_to <= 0:
        raise ValueError("W_to must be positive.")
    if k_rf < 0:
        raise ValueError("k_rf must be non-negative.")

    W_current = W_to

    segment_fractions = []
    segment_burns = {}

    mission_burn_LH2 = 0.0
    mission_burn_JetA = 0.0

    for segment in segments:
        fraction = segment_weight_fraction(segment, W_current)

        burned_mass = W_current * (1.0 - fraction)

        if segment.fuel_type == "LH2":
            mission_burn_LH2 += burned_mass
        elif segment.fuel_type == "JetA":
            mission_burn_JetA += burned_mass
        elif segment.fuel_type == "none":
            pass
        else:
            raise ValueError(
                f"fuel_type for segment '{segment.name}' must be "
                "'LH2', 'JetA', or 'none'."
            )

        segment_fractions.append(fraction)
        segment_burns[segment.name] = burned_mass

        W_current *= fraction

    total_mission_fraction = float(np.prod(segment_fractions))

    W_fuel_LH2 = (1.0 + k_rf) * mission_burn_LH2
    W_fuel_JetA = (1.0 + k_rf) * mission_burn_JetA

    W_fuel_total = W_fuel_LH2 + W_fuel_JetA

    return (
        W_fuel_total,
        W_fuel_LH2,
        W_fuel_JetA,
        segment_fractions,
        total_mission_fraction,
        segment_burns,
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
            segment_fractions,
            total_mission_fraction,
            segment_burns,
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
                "V_tank_capacity": V_tank_capacity,
                "V_LH2": V_LH2,
                "V_JetA": V_JetA,
                "segment_fractions": segment_fractions,
                "segment_burns": segment_burns,
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
    # Mission profile: Mach 5, 28 km, 9500 km range
    # -------------------------------------------------------------------------

    h0 = 0.0
    h10 = 10_000.0
    h28 = 28_000.0

    V_M07_h0 = mach_to_velocity(0.7, h0)
    V_M09_h10 = mach_to_velocity(0.9, h10)
    V_M17_h10 = mach_to_velocity(1.7, h10)
    V_M5_h28 = mach_to_velocity(5.0, h28)

    cruise_range = 9_500_000.0
    cruise_time = cruise_range / V_M5_h28

    segments = [
        MissionSegment(
            name="1_takeoff",
            mode="fixed",
            fuel_type="JetA",
            fixed_fraction=0.990,
        ),

        MissionSegment(
            name="2_accel_to_M0.7",
            mode="T_gt_D",
            fuel_type="JetA",
            delta_h=0.0,
            V_initial=0.0,
            V_final=V_M07_h0,
            V_average=0.5 * V_M07_h0,
            I_sp=2200.0,
            D=95_000.0,
            T=180_000.0,
        ),

        MissionSegment(
            name="3_accel_to_M0.9_climb_10km",
            mode="T_gt_D",
            fuel_type="JetA",
            delta_h=10_000.0,
            V_initial=V_M07_h0,
            V_final=V_M09_h10,
            V_average=0.5 * (V_M07_h0 + V_M09_h10),
            I_sp=2000.0,
            D=85_000.0,
            T=170_000.0,
        ),

        MissionSegment(
            name="4_accel_to_M2.5",
            mode="T_gt_D",
            fuel_type="JetA",
            delta_h=0.0,
            V_initial=V_M09_h10,
            V_final=V_M17_h10,
            V_average=0.5 * (V_M09_h10 + V_M17_h10),
            I_sp=1600.0,
            D=110_000.0,
            T=220_000.0,
        ),

        MissionSegment(
            name="5_climb_to_28km",
            mode="T_gt_D",
            fuel_type="LH2",
            delta_h=18_000.0,
            V_initial=V_M17_h10,
            V_final=V_M5_h28,
            V_average=0.5 * (V_M17_h10 + V_M5_h28),
            I_sp=2800.0,
            D=120_000.0,
            T=280_000.0,
        ),

        MissionSegment(
            name="6_cruise_M5_28km",
            mode="T_eq_D",
            fuel_type="LH2",
            I_sp=3000.0,
            D=100_000.0,
            delta_t=cruise_time,
        ),

        MissionSegment(
            name="7_unpowered_descent",
            mode="fixed",
            fuel_type="none",
            fixed_fraction=1.0,
        ),

        MissionSegment(
            name="8_landing",
            mode="fixed",
            fuel_type="JetA",
            fixed_fraction=0.997,
        ),
    ]

    S_plan, W_to, result = converge_S_plan_and_TOGW(
        tau=0.14,
        configuration="blended_body",  # choose: "waverider", "wing_body", "blended_body"
        S_plan_guess=433.871,

        I_str=24.0,
        I_tps=6.0,

        KIT=1.0,
        I_tank=4.0,

        rho_LH2=70.0,
        rho_JetA=800.0,
        k_pf=1.0,

        I_sub=0.04,

        W_prop=2_811.858,
        V_prop=10.0,

        W_payload=7_000.0,
        rho_payload=100.0,

        segments=segments,
        k_rf=0.06,

        W_to_guess=114_203.396,

        rho_str=2700.0,
        rho_tps=500.0,
        rho_tank_str=2700.0,

        K_lg=0.01,
        K_sub=0.02,
        K_void=0.30,

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
    print(f"W_fuel_LH2:                 {result['W_fuel_LH2']:.3f} kg")
    print(f"W_fuel_JetA:                {result['W_fuel_JetA']:.3f} kg")
    print(f"V_LH2:                      {result['V_LH2']:.3f} m³")
    print(f"V_JetA:                     {result['V_JetA']:.3f} m³")

    print("\nSegment fuel burn before reserve")
    print("--------------------------------")
    for segment_name, burn in result["segment_burns"].items():
        print(f"{segment_name:<32s}: {burn:.3f} kg")

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