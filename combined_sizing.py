import numpy as np
import math


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
    W_p: float,
    W_f: float,
    W_to_guess: float = 100_000.0,
    tol: float = 1.0,
    relaxation: float = 0.6,
    max_iter: int = 200,
) -> tuple[float, dict[str, float]]:
    """
    Iteratively estimate takeoff gross weight using weight buildup.

    W_to = W_str + W_tps + W_lg + W_prop + W_tank + W_sub + W_payload + W_fuel

    Iterated terms:
        W_lg  = 0.01 * W_to^1.124
        W_sub = I_sub * W_to

    Fixed terms for now:
        W_str, W_tps, W_prop, W_tank, W_payload, W_fuel
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
        raise ValueError("fuel densities must be positive.")

    S_wet = s_wet(K_W, S_plan)

    W_str = W_structural(S_wet, I_str)
    W_TPS = W_tps(I_tps, S_wet)

    V_tank = tank_volume(
        r1=r1,
        r2=r2,
        rho1=rho1,
        rho2=rho2,
        W_fuel=W_f,
        k_pf=k_pf,
    )

    W_tank = fuel_tank_weight(
        KIT=KIT,
        I_tank=I_tank,
        V_tank=V_tank,
    )

    W_to = W_to_guess

    for iteration in range(max_iter):
        W_lg = W_landinggear(W_to)
        W_sub = subsystem_weight(I_sub, W_to)

        W_to_required = (
            W_str
            + W_TPS
            + W_lg
            + W_prop
            + W_tank
            + W_sub
            + W_p
            + W_f
        )

        error = W_to_required - W_to

        if abs(error) < tol:
            result = {
                "TOGW": W_to,
                "TOGW_required": W_to_required,
                "error": error,
                "iterations": iteration,
                "S_wet": S_wet,
                "W_str": W_str,
                "W_tps": W_TPS,
                "W_landinggear": W_lg,
                "W_prop": W_prop,
                "W_tank": W_tank,
                "W_sub": W_sub,
                "W_payload": W_p,
                "W_fuel": W_f,
                "V_tank": V_tank,
            }

            return W_to, result

        W_to_new = W_to + relaxation * error

        if W_to_new <= 0:
            raise RuntimeError("TOGW iteration became non-physical.")

        W_to = W_to_new

    raise RuntimeError("TOGW iteration did not converge within max_iter.")

W_to, result = TOGW_iterative(
    I_str=20.0,
    K_W=2.793,
    S_plan=765.2,
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
    W_p=10_000.0,
    W_f=40_000.0,
    W_to_guess=100_000.0,
)

print(f"Converged TOGW: {W_to:.2f} kg")
print(f"Iterations: {result['iterations']}")

for key, value in result.items():
    print(f"{key}: {value:.3f}")