import numpy as np
import math
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Basic weight functions
# -----------------------------------------------------------------------------

def s_wet(K_W: float, S_plan: float) -> float:
    return K_W * S_plan


def W_structural(S_wet: float, I_structural: float) -> float:
    return S_wet * I_structural


def W_tps(I_tps: float, S_wet: float) -> float:
    return I_tps * S_wet


def W_landinggear(W_to: float) -> float:
    return 0.01 * W_to**1.124


def fuel_tank_weight(KIT: float, I_tank: float, V_tank: float) -> float:
    return (1.0 - KIT) * I_tank * V_tank


def tank_volume(
    r1: float,
    r2: float,
    rho1: float,
    rho2: float,
    W_fuel: float,
    k_pf: float,
) -> float:
    return (r1 / rho1 + r2 / rho2) * (W_fuel / k_pf)


def subsystem_weight(I_sub: float, W_to: float) -> float:
    return I_sub * W_to


# -----------------------------------------------------------------------------
# Mission segment fuel fraction
# -----------------------------------------------------------------------------

@dataclass
class MissionSegment:
    """
    One mission segment for fuel-fraction calculation.

    mode:
        "T_gt_D"  -> use exp[-Δ(h + V²/2g) / (Isp V (1 - D/T))]
        "T_eq_D"  -> use exp[-D Δt / (Isp W)]
        "fixed"   -> directly use a known segment fraction, e.g. 0.995 or 1.0
    """

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
            raise ValueError("For mode='T_gt_D', thrust T must be greater than drag D.")
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

        exponent = -(segment.D * segment.delta_t) / (segment.I_sp * W_current)

        fraction = math.exp(exponent)

    else:
        raise ValueError("segment.mode must be 'T_gt_D', 'T_eq_D', or 'fixed'.")

    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"Segment fraction must satisfy 0 < fraction <= 1. "
            f"Got {fraction:.6g}."
        )

    return fraction


def fuel_weight_from_segments(
    W_to: float,
    segments: list[MissionSegment],
    k_rf: float,
) -> tuple[float, list[float], float]:
    """
    Calculate fuel weight from mission segment weight fractions.

    W_f = (1 + k_rf) * W_to * [1 - product(W_i / W_{i-1})]
    """

    if W_to <= 0:
        raise ValueError("W_to must be positive.")
    if k_rf < 0:
        raise ValueError("k_rf must be non-negative.")

    W_current = W_to
    segment_fractions = []

    for segment in segments:
        fraction = segment_weight_fraction(segment, W_current)
        segment_fractions.append(fraction)

        # Update current aircraft weight after this segment
        W_current *= fraction

    total_mission_fraction = float(np.prod(segment_fractions))

    W_fuel = (1.0 + k_rf) * W_to * (1.0 - total_mission_fraction)

    return W_fuel, segment_fractions, total_mission_fraction


# -----------------------------------------------------------------------------
# Iterative TOGW estimation
# -----------------------------------------------------------------------------

def TOGW_iterative(
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
    k_rf: float = 0.06,
    W_to_guess: float = 100_000.0,
    tol: float = 1.0,
    relaxation: float = 0.6,
    max_iter: int = 200,
) -> tuple[float, dict[str, float | list[float]]]:
    """
    Iteratively estimate takeoff gross weight.

    Weight buildup:
        W_to = W_str
             + W_tps
             + W_landinggear
             + W_prop
             + W_tank
             + W_sub
             + W_payload
             + W_fuel

    Fuel is calculated inside the iteration from:
        W_f = (1 + k_rf) * W_to * [1 - product(segment fractions)]
    """

    if S_plan <= 0:
        raise ValueError("S_plan must be positive.")
    if K_W <= 0:
        raise ValueError("K_W must be positive.")
    if W_to_guess <= 0:
        raise ValueError("W_to_guess must be positive.")
    if k_pf <= 0:
        raise ValueError("k_pf must be positive.")
    if rho1 <= 0 or rho2 <= 0:
        raise ValueError("Fuel densities must be positive.")
    if not segments:
        raise ValueError("At least one mission segment is required.")

    S_wet = s_wet(K_W, S_plan)

    W_str = W_structural(S_wet, I_str)
    W_TPS = W_tps(I_tps, S_wet)

    W_to = W_to_guess

    for iteration in range(max_iter):

        W_fuel, segment_fractions, total_mission_fraction = fuel_weight_from_segments(
            W_to=W_to,
            segments=segments,
            k_rf=k_rf,
        )

        V_tank = tank_volume(
            r1=r1,
            r2=r2,
            rho1=rho1,
            rho2=rho2,
            W_fuel=W_fuel,
            k_pf=k_pf,
        )

        W_tank = fuel_tank_weight(
            KIT=KIT,
            I_tank=I_tank,
            V_tank=V_tank,
        )

        W_lg = W_landinggear(W_to)

        W_sub = subsystem_weight(I_sub, W_to)

        W_to_required = (
            W_str
            + W_TPS
            + W_lg
            + W_prop
            + W_tank
            + W_sub
            + W_payload
            + W_fuel
        )

        error = W_to_required - W_to

        if abs(error) < tol:
            result = {
                "TOGW": W_to,
                "TOGW_required": W_to_required,
                "error": error,
                "iterations": iteration,
                "S_plan": S_plan,
                "K_W": K_W,
                "S_wet": S_wet,
                "W_str": W_str,
                "W_tps": W_TPS,
                "W_landinggear": W_lg,
                "W_prop": W_prop,
                "W_tank": W_tank,
                "W_sub": W_sub,
                "W_payload": W_payload,
                "W_fuel": W_fuel,
                "V_tank": V_tank,
                "segment_fractions": segment_fractions,
                "total_mission_fraction": total_mission_fraction,
            }

            return W_to, result

        W_to_new = W_to + relaxation * error

        if W_to_new <= 0:
            raise RuntimeError("TOGW iteration became non-physical.")

        W_to = W_to_new

    raise RuntimeError("TOGW iteration did not converge within max_iter.")


# -----------------------------------------------------------------------------
# Example run
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    segments = [
        # Warmup / takeoff / initial operation
        MissionSegment(
            mode="fixed",
            fixed_fraction=0.995,
        ),

        # Example climb or acceleration segment, T > D
        MissionSegment(
            mode="T_gt_D",
            delta_h=10_000.0,
            V_initial=200.0,
            V_final=600.0,
            V_average=400.0,
            g=9.81,
            I_sp=1800.0,
            D=80_000.0,
            T=200_000.0,
        ),

        # Example cruise segment, T = D
        MissionSegment(
            mode="T_eq_D",
            I_sp=1800.0,
            D=100_000.0,
            delta_t=600.0,
        ),

        # Unpowered descent
        MissionSegment(
            mode="fixed",
            fixed_fraction=1.0,
        ),

        # Landing
        MissionSegment(
            mode="fixed",
            fixed_fraction=0.998,
        ),
    ]

    W_to, result = TOGW_iterative(
        I_str=20.0,
        K_W=2.793,
        S_plan=433.871,
        I_tps=6.0,
        KIT=1.0,
        I_tank=4.0,
        r1=1.0,
        r2=0.0,
        rho1=70.0,
        rho2=1.0,
        k_pf=1.0,
        I_sub=0.04,
        W_prop=20_000.0,
        W_payload=7_000.0,
        segments=segments,
        k_rf=0.06,
        W_to_guess=100_000.0,
        tol=1.0,
        relaxation=0.6,
        max_iter=200,
    )

    print(f"\nConverged TOGW: {W_to:.2f} kg")
    print(f"Iterations: {result['iterations']}")

    print("\nWeight breakdown")
    print("----------------")
    print(f"W_str:         {result['W_str']:.2f} kg")
    print(f"W_tps:         {result['W_tps']:.2f} kg")
    print(f"W_landinggear: {result['W_landinggear']:.2f} kg")
    print(f"W_prop:        {result['W_prop']:.2f} kg")
    print(f"W_tank:        {result['W_tank']:.2f} kg")
    print(f"W_sub:         {result['W_sub']:.2f} kg")
    print(f"W_payload:     {result['W_payload']:.2f} kg")
    print(f"W_fuel:        {result['W_fuel']:.2f} kg")

    print("\nMission fuel data")
    print("-----------------")
    print(f"Segment fractions:      {result['segment_fractions']}")
    print(f"Total mission fraction: {result['total_mission_fraction']:.5f}")
    print(f"Tank volume:            {result['V_tank']:.2f} m³")
    print(f"Final residual error:   {result['error']:.4f} kg")