import numpy as np
import matplotlib.pyplot as plt
import math

def TOGWcalc(W_empty, W_payload, W_fuel):
    return W_empty + W_payload + W_fuel

def s_wet(k_w, s_plan):
    return k_w * s_plan

def W_structural(s_wet, I_structural):
    return s_wet * I_structural

def W_tps(I_tps, s_wet):
    return I_tps * s_wet

def W_landinggear(W_to):
    0.01 * W_to**(1.124)

def fuel_tank_weight(KIT: float, I_tank: float, V_tank: float) -> float:
    
    return (1 - KIT) * I_tank * V_tank

def tank_volume(r1: float, r2: float, rho1: float, rho2: float,
                W_fuel: float, k_pf: float) -> float:
   
    return (r1 / rho1 + r2 / rho2) * (W_fuel / k_pf)

def subsystem_weight(I_sub: float, W_to: float) -> float:
    return I_sub * W_to

def segment_weight_fraction(T: float, D: float,
                             h: float, V: float, g: float,
                             I_sp: float, delta_t: float, W: float) -> float:

    if T != D:
        A = 1.0  # adjust if paper defines A differently
        numerator = A * (h + V ** 2 / (2 * g))
        denominator = I_sp * V * (1 - D / T)
        return math.exp(-numerator / denominator)
    else:
        # T = D: exp(-D / (I_sp * W * Δt)) — here expressed per unit weight
        return math.exp(-D / (I_sp * W )* delta_t)
    
def total_weight_fraction(segment_fractions, k_rf):
    return (1+k_rf)*(1+np.prod(segment_fractions))

def TOGW(I_str: float, K_W: float, S_plan: float,
                          I_tps: float,
                          KIT: float, I_tank: float,
                          r1: float, r2: float,
                          rho1: float, rho2: float,
                          k_pf: float,
                          I_sub: float,
                          W_prop: float,
                          W_p: float,
                          W_f: float,
                          W_to_guess: float = 100_000.0,
                          tol: float = 1.0,
                          max_iter: int = 200) -> float:
  
    tank_factor = (1 - KIT) * I_tank * (r1 / rho1 + r2 / rho2) * W_f / k_pf
    fixed = (I_str * K_W * S_plan
             + I_tps * K_W * S_plan
             + W_prop
             + tank_factor
             + W_p
             + W_f)
 
    W_to = W_to_guess
    for _ in range(max_iter):
        W_to_new = fixed + 0.01 * W_to ** 1.124 + I_sub * W_to
        if abs(W_to_new - W_to) < tol:
            return W_to_new
        W_to = W_to_new
 
    raise RuntimeError(
        f"takeoff_gross_weight did not converge after {max_iter} iterations. "
        f"Last W_to = {W_to:.2f}"
    )