import numpy as np
import math
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Basic equations
# -----------------------------------------------------------------------------

def s_wet(K_W: float, S_plan: float) -> float:
    return K_W * S_plan


def volume_from_tau(tau: float, S_plan: float) -> float:
    """
    Küchemann relation:
        V_tot = tau * S_plan^1.5
    """
    return tau * S_plan**1.5


def W_landinggear(W_to: float) -> float:
    return 0.01 * W_to**1.124


def tank_volume(
    r1: float,
    r2: float,
    rho1: float,
    rho2: float,
    W_fuel: float,
    k_pf: float,
) -> float:
    return (r1 / rho1 + r2 / rho2) * (W_fuel / k_pf)


# -----------------------------------------------------------------------------
# Mission fuel fraction
# -----------------------------------------------------------------------------

@dataclass
class MissionSegment:
    mode: str

    # For T > D
    delta_h: float = 0.0
    V_initial: float = 0.0
    V_final: float = 0.0
    V_average: float = 0.0
    g: float = 9.81
    I_sp: float = 0.0
    D: float = 0.0
    T: float = 0.0

    # For T = D
    delta_t: float = 0.0

    # For fixed fractions
    fixed_fraction: float = 1.0


def segment_weight_fraction(segment: MissionSegment, W_current: float) -> float:
    if segment.mode == "fixed":
        fraction = segment.fixed_fraction

    elif segment.mode == "T_gt_D":
        if segment.T <= segment.D:
            raise ValueError("For T_gt_D, T must be larger than D.")
        if segment.I_sp <= 0:
            raise ValueError("I_sp must be positive.")
        if segment.V_average <= 0:
            raise ValueError("V_average must be positive.")

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
            raise ValueError("I_sp must be positive.")
        if W_current <= 0:
            raise ValueError("W_current must be positive.")

        exponent = -(segment.D * segment.delta_t) / (
            segment.I_sp * W_current
        )

        fraction = math.exp(exponent)

    else:
        raise ValueError("mode must be 'fixed', 'T_gt_D', or 'T_eq_D'.")

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Invalid segment fraction: {fraction}")

    return fraction


def fuel_weight_from_segments(
    W_to: float,
    segments: list[MissionSegment],
    k_rf: float,
) -> tuple[float, list[float], float]:

    W_current = W_to
    segment_fractions = []

    for segment in segments:
        fraction = segment_weight_fraction(segment, W_current)
        segment_fractions.append(fraction)
        W_current *= fraction

    total_mission_fraction = float(np.prod(segment_fractions))

    W_fuel = (1.0 + k_rf) * W_to * (1.0 - total_mission_fraction)

    return W_fuel, segment_fractions, total_mission_fraction


# -----------------------------------------------------------------------------
# Inner loop: converge TOGW for fixed S_plan
# -----------------------------------------------------------------------------

def converge_TOGW_for_fixed_S_plan(
    I_str: float,
    K_W: float,
    S_plan: float,
    I_tps: float,
    KIT: float,
    I_tank: float,
    r1: float,
    r2: float,
    rho1: float,
    rho2: float,
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

        W_fuel, segment_fractions, total_mission_fraction = fuel_weight_from_segments(
            W_to=W_to,
            segments=segments,
            k_rf=k_rf,
        )

        V_tank_capacity = tank_volume(
            r1=r1,
            r2=r2,
            rho1=rho1,
            rho2=rho2,
            W_fuel=W_fuel,
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
            + W_fuel
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
                "W_fuel": W_fuel,
                "V_tank_capacity": V_tank_capacity,
                "segment_fractions": segment_fractions,
                "total_mission_fraction": total_mission_fraction,
            }

            return W_to, result

        W_to_new = W_to + weight_relaxation * weight_error

        if W_to_new <= 0:
            raise RuntimeError("TOGW became non-physical.")

        W_to = W_to_new

    raise RuntimeError("TOGW did not converge.")


# -----------------------------------------------------------------------------
# Volume calculation
# -----------------------------------------------------------------------------

def required_volume(
    I_str: float,
    I_tps: float,
    K_W: float,
    S_plan: float,
    W_fuel: float,
    W_payload: float,
    r1: float,
    r2: float,
    rho1: float,
    rho2: float,
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
    """
    Volume equation.

    Terms K_lg*Vtot, K_sub*Vtot, and K_void*Vtot use the current available
    volume estimate from Küchemann's equation.
    """

    V_structure = (I_str * K_W * S_plan) / rho_str

    V_tps = (I_tps * K_W * S_plan) / rho_tps

    V_tank_capacity = tank_volume(
        r1=r1,
        r2=r2,
        rho1=rho1,
        rho2=rho2,
        W_fuel=W_fuel,
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
    }


# -----------------------------------------------------------------------------
# Outer loop: converge S_plan and TOGW
# -----------------------------------------------------------------------------

def converge_S_plan_and_TOGW(
    tau: float,
    I_str: float,
    K_W: float,
    S_plan_guess: float,
    I_tps: float,
    KIT: float,
    I_tank: float,
    r1: float,
    r2: float,
    rho1: float,
    rho2: float,
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
            r1=r1,
            r2=r2,
            rho1=rho1,
            rho2=rho2,
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
            W_fuel=weight_result["W_fuel"],
            W_payload=W_payload,
            r1=r1,
            r2=r2,
            rho1=rho1,
            rho2=rho2,
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
                "S_plan": S_plan,
                "TOGW": W_to,
                "V_tot_available": V_tot_available,
                "volume_error": volume_error,
                "size_iterations": size_iteration,
                "wing_loading": W_to / S_plan,
            }

            return S_plan, W_to, result

        # From V = tau*S_plan^1.5:
        # S_new = S_old * (V_required / V_available)^(2/3)
        raw_correction = (V_required / V_tot_available) ** (2.0 / 3.0)

        correction = 1.0 + S_plan_relaxation * (raw_correction - 1.0)

        # Prevent unstable jumps
        correction = max(0.5, min(1.5, correction))

        S_plan_new = S_plan * correction

        if S_plan_new <= 0:
            raise RuntimeError("S_plan became non-physical.")

        S_plan = S_plan_new

        # Use last converged TOGW as next starting guess
        W_to_current_guess = W_to

    raise RuntimeError("S_plan did not converge.")


# -----------------------------------------------------------------------------
# Example run
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    segments = [
        MissionSegment(mode="fixed", fixed_fraction=0.995),

        MissionSegment(
            mode="T_gt_D",
            delta_h=10_000.0,
            V_initial=200.0,
            V_final=600.0,
            V_average=400.0,
            I_sp=1800.0,
            D=80_000.0,
            T=200_000.0,
        ),

        MissionSegment(
            mode="T_eq_D",
            I_sp=1800.0,
            D=84000.0,
            delta_t=5400,
        ),

        MissionSegment(mode="fixed", fixed_fraction=1.0),
        MissionSegment(mode="fixed", fixed_fraction=0.998),
    ]

    S_plan, W_to, result = converge_S_plan_and_TOGW(
        tau=0.14,
        I_str=20.0,
        K_W=2.793,
        S_plan_guess=433.871,
        I_tps=6.0,
        KIT=1.0,
        I_tank=4.0,
        r1=1.0,
        r2=0.0,
        rho1=70.0,
        rho2=1.0,
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
        K_void=0.20,
    )

    print("\nConverged values")
    print("----------------")
    print(f"S_plan:          {S_plan:.3f} m²")
    print(f"TOGW:            {W_to:.3f} kg")
    print(f"V_tot_available: {result['V_tot_available']:.3f} m³")
    print(f"V_required:      {result['V_required']:.3f} m³")
    print(f"Volume error:    {result['volume_error']:.6f} m³")
    print(f"Weight error:    {result['weight_error']:.6f} kg")
    print(f"Wing loading:    {result['wing_loading']:.3f} kg/m²")
    print(f"Size iterations: {result['size_iterations']}")
    print(f"Weight iters:    {result['weight_iterations']}")

    print("\nWeight breakdown")
    print("----------------")
    print(f"W_str:           {result['W_str']:.3f} kg")
    print(f"W_tps:           {result['W_tps']:.3f} kg")
    print(f"W_landinggear:   {result['W_landinggear']:.3f} kg")
    print(f"W_prop:          {result['W_prop']:.3f} kg")
    print(f"W_tank:          {result['W_tank']:.3f} kg")
    print(f"W_sub:           {result['W_sub']:.3f} kg")
    print(f"W_payload:       {result['W_payload']:.3f} kg")
    print(f"W_fuel:          {result['W_fuel']:.3f} kg")

    print("\nVolume breakdown")
    print("----------------")
    print(f"V_structure:     {result['V_structure']:.3f} m³")
    print(f"V_tps:           {result['V_tps']:.3f} m³")
    print(f"V_landinggear:   {result['V_landinggear']:.3f} m³")
    print(f"V_prop:          {result['V_prop']:.3f} m³")
    print(f"V_tank_structure:{result['V_tank_structure']:.3f} m³")
    print(f"V_subsystem:     {result['V_subsystem']:.3f} m³")
    print(f"V_void:          {result['V_void']:.3f} m³")
    print(f"V_payload:       {result['V_payload']:.3f} m³")
    print(f"V_tank_capacity: {result['V_tank_capacity']:.3f} m³")