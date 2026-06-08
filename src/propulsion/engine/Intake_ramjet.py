import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAMMA = 1.4          # ratio of specific heats for air


# ---------------------------------------------------------------------------
# Core aerodynamic relations
# ---------------------------------------------------------------------------

def prandtl_meyer(M: float) -> float:
    """Prandtl-Meyer function nu(M) in degrees (M >= 1)."""
    g = GAMMA
    return np.degrees(
        np.sqrt((g + 1) / (g - 1)) *
        np.arctan(np.sqrt((g - 1) / (g + 1) * (M**2 - 1))) -
        np.arctan(np.sqrt(M**2 - 1))
    )


def optimum_total_turn_angle(M_inf: float, n: int) -> float:
    """
    Prandtl-Meyer optimum total turning angle nu (degrees) - Eq. (1).
    Returns the maximum isentropic deflection for M_inf.
    """
    if n == 2:
        total_turn_local = 0.6687 * M_inf **3 - 8.4697 * M_inf**2 + 36.463 * M_inf - 28.726 
    elif n == 3:
        total_turn_local = 0.6768*M_inf**3 - 9.8723*M_inf**2 + 49.076*M_inf - 43.845
    else:
        raise ValueError("Optimum total turn angle is only defined for n=2 or n=3 ramps.")
    return total_turn_local


def theta_beta_M(theta_deg: float, M: float) -> float:
    """
    Oblique shock wave angle beta (degrees) from ramp deflection angle theta
    and upstream Mach number M  -  Eq. (2):

        tan(theta) = 2 cot(beta) * (M^2 sin^2(beta) - 1)
                     / (M^2 (gamma + cos(2 beta)) + 2)

    Returns the weak-shock (lower beta) solution.
    """
    g = GAMMA
    theta = np.radians(theta_deg)
    mu = np.arcsin(1.0 / M)            # Mach angle

    def equation(beta):
        return (np.tan(theta) -
                2.0 / np.tan(beta) *
                (M**2 * np.sin(beta)**2 - 1.0) /
                (M**2 * (g + np.cos(2.0 * beta)) + 2.0))

    # Scan from the Mach angle upward to find the first sign change.
    beta_lo = mu + 1e-6
    f_lo = equation(beta_lo)
    beta_hi = None
    for b_deg in np.arange(np.degrees(mu) + 1.0, 89.0, 0.5):
        b = np.radians(b_deg)
        if equation(b) * f_lo < 0:
            beta_hi = b
            break
    if beta_hi is None:
        raise ValueError(
            f"No oblique shock solution for theta={theta_deg:.2f} deg at M={M:.4f}. "
            "Check that the deflection angle is below the detachment limit.")
    return np.degrees(brentq(equation, beta_lo, beta_hi))


def normal_mach(M: float, beta_deg: float) -> float:
    """Normal Mach component upstream of the oblique shock: Mn = M sin(beta)."""
    return M * np.sin(np.radians(beta_deg))


def post_oblique_mach(M: float, beta_deg: float, theta_deg: float) -> float:
    """
    Mach number downstream of an oblique shock.
    Apply normal-shock relations to Mn then correct for the flow deflection.
    """
    g = GAMMA
    Mn1 = normal_mach(M, beta_deg)
    Mn2_sq = (1.0 + (g - 1.0) / 2.0 * Mn1**2) / (g * Mn1**2 - (g - 1.0) / 2.0)
    return np.sqrt(Mn2_sq) / np.sin(np.radians(beta_deg - theta_deg))


def stagnation_pressure_ratio(Mn: float) -> float:
    """
    Stagnation pressure ratio P01_out / P01_in across a normal shock at
    normal Mach number Mn  -  Eq. (3).
    """
    g = GAMMA
    term1 = ((g + 1.0) * Mn**2 / ((g - 1.0) * Mn**2 + 2.0)) ** (g / (g - 1.0))
    term2 = ((g + 1.0) / (2.0 * g * Mn**2 - (g - 1.0))) ** (1.0 / (g - 1.0))
    return term1 * term2


def theta_from_beta_M(beta_deg: float, M: float) -> float:
    """Compute flow deflection angle theta [deg] from shock angle beta and upstream M."""
    g = GAMMA
    beta = np.radians(beta_deg)
    tan_theta = (
        2.0 / np.tan(beta)
        * (M**2 * np.sin(beta)**2 - 1.0)
        * 1.0 / (M**2 * (g + np.cos(2.0 * beta)) + 2.0)
    )
    return np.degrees(np.arctan(tan_theta))


def check_detachment_limit(M: float) -> float:
    """Returns the maximum physically possible deflection angle (theta_max) for a given Mach number."""
    if M <= 1.0:
        return 0.0
    g = GAMMA
    mu = np.arcsin(1.0 / M)
    betas_scan = np.radians(np.arange(np.degrees(mu) + 0.1, 90.0, 0.05))
    thetas_scan = np.degrees(np.arctan(
        2.0 / np.tan(betas_scan) *
        (M**2 * np.sin(betas_scan)**2 - 1.0) /
        (M**2 * (g + np.cos(2.0 * betas_scan)) + 2.0)
    ))
    return float(np.nanmax(thetas_scan))


# ---------------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------------

def analyse_intake4(
    M_inf:          float,
    L_1:            float,
    theta_1_deg:    float,
    y_cowl:         float,
    delta_cowl_deg: float = 0.0,
    verbose:        bool  = True,
) -> dict:
    """
    Full mixed-compression intake analysis — 2-ramp + infinite internal reflections.
    Continues calculating oblique shock bounces down parallel walls until 
    detachment limit is breached, ending the channel flow with a Normal Shock.
    """
    assert M_inf >= 1.0, "Freestream Mach must be >= 1"

    theta_1 = theta_1_deg

    # ------------------------------------------------------------------
    # 1.  Oblique shock 1  (ramp-1 leading edge, M_inf)
    # ------------------------------------------------------------------
    beta_1   = theta_beta_M(theta_1, M_inf)
    Mn_1     = normal_mach(M_inf, beta_1)
    P0_rat_1 = stagnation_pressure_ratio(Mn_1)
    M_2      = post_oblique_mach(M_inf, beta_1, theta_1)

    stage_1 = {
        "stage":     "Ramp 1",
        "M_in":      M_inf,
        "theta_deg": theta_1,
        "beta_deg":  beta_1,
        "Mn":        Mn_1,
        "M_out":     M_2,
        "P0_ratio":  P0_rat_1,
    }

    # ------------------------------------------------------------------
    # 2.  Cowl-lip position
    # ------------------------------------------------------------------
    tan_b1 = np.tan(np.radians(beta_1))
    tan_t1 = np.tan(np.radians(theta_1))

    x_c  = y_cowl / tan_b1
    cowl = np.array([x_c, y_cowl])

    y_P1 = L_1 * tan_t1
    P1   = np.array([L_1, y_P1])

    assert x_c > L_1, (
        f"Cowl lip (x_c={x_c:.4f}) is upstream of ramp-1 tip (L_1={L_1:.4f}). "
        "Increase y_cowl or decrease L_1.")

    # ------------------------------------------------------------------
    # 3.  Back-calculate beta_2 and theta_2 from geometry
    # ------------------------------------------------------------------
    angle_global_2 = np.degrees(np.arctan2(y_cowl - y_P1, x_c - L_1))
    beta_2         = angle_global_2 - theta_1          
    theta_2        = theta_from_beta_M(beta_2, M_2)    

    Mn_2     = normal_mach(M_2, beta_2)
    P0_rat_2 = stagnation_pressure_ratio(Mn_2)
    M_3      = post_oblique_mach(M_2, beta_2, theta_2)

    stage_2 = {
        "stage":     "Ramp 2",
        "M_in":      M_2,
        "theta_deg": theta_2,
        "beta_deg":  beta_2,
        "Mn":        Mn_2,
        "M_out":     M_3,
        "P0_ratio":  P0_rat_2,
    }

    # ------------------------------------------------------------------
    # 4.  First Reflected shock angle (Cowl Lip Reflection)
    # ------------------------------------------------------------------
    deflection_ref = (theta_1 + theta_2) - delta_cowl_deg

    theta_max = check_detachment_limit(M_3)
    if deflection_ref >= theta_max:
        raise ValueError(
            f"Reflected shock deflection ({deflection_ref:.2f} deg) exceeds the "
            f"detachment limit ({theta_max:.2f} deg) at M_3={M_3:.4f}.")

    beta_ref   = theta_beta_M(deflection_ref, M_3)
    Mn_ref     = normal_mach(M_3, beta_ref)
    P0_rat_ref = stagnation_pressure_ratio(Mn_ref)
    theta_ref  = theta_from_beta_M(beta_ref, M_3)
    M_4        = post_oblique_mach(M_3, beta_ref, theta_ref)

    stage_ref = {
        "stage":     "Reflected shock 1",
        "M_in":      M_3,
        "theta_deg": theta_ref,
        "beta_deg":  beta_ref,
        "Mn":        Mn_ref,
        "M_out":     M_4,
        "P0_ratio":  P0_rat_ref,
    }

    # Calculate L_2 landing metrics based on First Reflection geometry
    phi_deg   = (theta_1 + theta_2) - beta_ref
    phi_rad   = np.radians(phi_deg)
    cos_phi   = np.cos(phi_rad)
    sin_phi   = np.sin(phi_rad)
    tan_ramp2 = np.tan(np.radians(theta_1 + theta_2))

    denom  = sin_phi - cos_phi * tan_ramp2
    numer  = y_P1 + (x_c - L_1) * tan_ramp2 - y_cowl

    assert abs(denom) > 1e-12, "Reflected shock ray is parallel to ramp-2 surface."
    t_land = numer / denom
    assert t_land > 0, "Reflected shock lands upstream."

    x_land     = x_c + t_land * cos_phi
    y_land     = y_cowl + t_land * sin_phi
    landing_pt = np.array([x_land, y_land])
    L_2        = x_land - L_1

    # Base list initialized with first 3 stages
    stages = [stage_1, stage_2, stage_ref]

    # ------------------------------------------------------------------
    # 5. Continuous Internal Bouncing Shocks (Parallel Channels)
    # ------------------------------------------------------------------
    # Once inside the parallel channel, the forced turn angle magnitude is 
    # consistently theta_bounce = theta_1 + theta_2 to align back to horizontal.
    theta_bounce = theta_1 + theta_2
    current_M = M_4
    reflection_counter = 2

    # Preserve the shock angle from Reflected Shock 1
    previous_beta = beta_ref

    current_M = M_4

    for reflection_counter in [2, 3]:

        b_ref = beta_ref

        theta_reflection = theta_from_beta_M(
            b_ref,
            current_M
            )

        Mn_r = normal_mach(current_M, b_ref)
        P0_rat_r = stagnation_pressure_ratio(Mn_r)

        M_next = post_oblique_mach(
            current_M,
            b_ref,
            theta_reflection
        )

        stage_loop = {
            "stage":     f"Reflected shock {reflection_counter}",
            "M_in":      current_M,
            "theta_deg": theta_reflection,
            "beta_deg":  b_ref,
            "Mn":        Mn_r,
            "M_out":     M_next,
            "P0_ratio":  P0_rat_r,
        }

        stages.append(stage_loop)

        current_M = M_next
        
    # ------------------------------------------------------------------
    # 6. Terminal normal shock
    # ------------------------------------------------------------------

    g = GAMMA

    Mn_ns = current_M

    P0_rat_ns = stagnation_pressure_ratio(Mn_ns)

    M_out_ns = np.sqrt(
        (1.0 + (g - 1.0) / 2.0 * current_M**2)
        / (g * current_M**2 - (g - 1.0) / 2.0)
    )

    stage_ns = {
        "stage":     "Terminal Normal Shock",
        "M_in":      current_M,
        "theta_deg": 0.0,
        "beta_deg": 90.0,
        "Mn":        Mn_ns,
        "M_out":     M_out_ns,
        "P0_ratio":  P0_rat_ns,
    }

    stages.append(stage_ns)

    current_M = M_out_ns

    # ------------------------------------------------------------------
    # 7.  Totals
    # ------------------------------------------------------------------
    total_P0 = np.prod([s["P0_ratio"] for s in stages])

    # ------------------------------------------------------------------
    # 8.  Verbose output
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 75)
        print(f"   Ramjet Mixed-Compression Intake Analysis (Multi-Reflection)")
        print(f"   M_inf = {M_inf:.2f}   |   gamma = {GAMMA}")
        print("=" * 75)
        print(f"   Inputs")
        print(f"    theta_1         = {theta_1:.2f} deg")
        print(f"    L_1             = {L_1:.4f}")
        print(f"    y_cowl          = {y_cowl:.4f}")
        print(f"    delta_cowl      = {delta_cowl_deg:.2f} deg  (below x-axis)")
        print()
        print(f"   Derived geometry")
        print(f"    beta_1          = {beta_1:.2f} deg")
        print(f"    Cowl lip C      : ({x_c:.4f},  {y_cowl:.4f})")
        print(f"    Ramp-1 tip P1   : ({L_1:.4f},  {y_P1:.4f})")
        print(f"    beta_2 (local)  = {beta_2:.2f} deg  →  theta_2 = {theta_2:.2f} deg")
        print(f"    Preserved reflection beta = {beta_ref:.2f} deg")
        print(f"    Landing point 1 : ({x_land:.4f},  {y_land:.4f})")
        print(f"    L_2 (computed)  = {L_2:.4f}")
        print()
        hdr = (f"   {'Stage':<25} {'M_in':>6} {'theta':>7} {'beta':>7}"
               f" {'Mn':>6} {'M_out':>7} {'P0 ratio':>10}")
        print(hdr)
        print("   " + "-" * 71)
        for s in stages:
            print(f"   {s['stage']:<25} {s['M_in']:>6.4f} "
                  f"{s['theta_deg']:>7.2f} {s['beta_deg']:>7.2f} "
                  f"{s['Mn']:>6.4f} {s['M_out']:>7.4f} "
                  f"{s['P0_ratio']:>10.6f}")
        print("   " + "-" * 71)
        print(f"   {'TOTAL P0 recovery':<52} {total_P0:>10.6f}")
        print(f"   {'Final Flow Mach':<52} {current_M:>10.4f}")
        print("=" * 75)

    return {
        "M_inf":                 M_inf,
        "theta_1_deg":          theta_1,
        "theta_2_deg":          theta_2,        
        "delta_cowl_deg":       delta_cowl_deg,
        "L_1":                  L_1,
        "L_2":                  L_2,            
        "y_cowl":               y_cowl,
        "cowl_lip":             cowl.tolist(),
        "ramp1_tip":            P1.tolist(),
        "landing_pt":           landing_pt.tolist(),
        "beta_1_deg":           beta_1,
        "beta_2_deg":           beta_2,
        "phi_ray_deg":          phi_deg,
        "deflection_ref_deg":   deflection_ref,
        "beta_reflected_deg":   beta_ref,
        "stages":               stages,
        "total_P0_recovery":    total_P0,
        "M_exit":               current_M,
        "phi_ref1_deg": phi_deg,
    }


def optimise_intake(
    M_inf: float,
    y_cowl: float,
    delta_cowl_deg: float,
    max_oblique_shocks: float, # Absolute limit on the number of oblique shocks
    L1_range: tuple = (0.01, 1.0),
    theta1_range: tuple = (1.0, 20.0),
    L1_step: float = 0.005,
    theta1_step: float = 0.5,
    M_exit_min: float = 0.1,  # Minimum allowable subsonic Mach after normal shock
    M_exit_max: float = 0.8,  # Maximum allowable subsonic Mach after normal shock
    print_all: bool = False,
):
    """
    Sweeps L1 and theta1 to find the geometric configuration that maximizes 
    the total pressure recovery, factoring in a user-defined limit on the 
    maximum allowed oblique shocks before the terminal normal shock.

    Constraints:
        - Physically valid shock system (no premature oblique detachment)
        - theta_2 > 0 and L_2 > 0 (valid structural layout)
        - Terminal exit Mach must safely fall within typical subsonic diffuser bounds
        - Total oblique shocks <= max_oblique_shocks
    """

    if print_all:
        print(
            f"{'L1':>8} {'theta1':>8} {'theta2':>8} "
            f"{'L2':>8} {'P0_rec':>12} {'M_final':>10} {'Tot_Shocks':>10} {'Obliques':>9}"
        )
        print("-" * 85)
        
    best = None

    L1_values = np.arange(L1_range[0], L1_range[1] + L1_step, L1_step)
    theta1_values = np.arange(theta1_range[0], theta1_range[1] + theta1_step, theta1_step)

    for L1 in L1_values:
        for theta1 in theta1_values:
            try:
                # Run multi-reflection analysis
                r = analyse_intake4(
                    M_inf=M_inf,
                    L_1=L1,
                    theta_1_deg=theta1,
                    y_cowl=y_cowl,
                    delta_cowl_deg=delta_cowl_deg,
                    verbose=False,
                )

                # --------------------------------------------------------------
                # Geometric & Kinematic Constraints
                # --------------------------------------------------------------
                if r["theta_2_deg"] <= 0 or r["L_2"] <= 0:
                    continue

                if not (M_exit_min <= r["M_exit"] <= M_exit_max):
                    continue

                # --------------------------------------------------------------
                # Shock Count Constraint Logic
                # --------------------------------------------------------------
                total_stages = len(r["stages"])
                
                # Count how many stages are oblique shocks vs normal shocks
                num_oblique = sum(
                    1 for s in r["stages"] if "Normal Shock" not in s["stage"]
                )

                if num_oblique > max_oblique_shocks:
                    continue

                P0 = r["total_P0_recovery"]

                if print_all:
                    print(
                        f"{L1:8.4f}"
                        f"{theta1:8.2f}"
                        f"{r['theta_2_deg']:8.2f}"
                        f"{r['L_2']:8.4f}"
                        f"{P0:12.6f}"
                        f"{r['M_exit']:10.4f}"
                        f"{total_stages:10d}"
                        f"{num_oblique:9d}"
                    )

                # Track the highest total pressure recovery within limits
                if best is None or P0 > best["total_P0_recovery"]:
                    best = r

            except Exception:
                # Catches math errors from unphysical shock configurations/detachment limits
                continue

    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # print("\n[1] Extended Analysis with Channel Reflections & Terminal Normal Shock")
    # analyse_intake4(M_inf=3, theta_1_deg=9, L_1=0.016, y_cowl=0.01366, delta_cowl_deg=4)

    # best = optimise_intake(M_inf=3.0, y_cowl=0.01366, delta_cowl_deg=4,max_oblique_shocks=5, L1_range=(0.01, 1), theta1_range=(1.0, 20.0), print_all=True, M_exit_max=0.8, L1_step=0.001)
    # print("\nBest configuration")
    # print("------------------")
    # print(f"L1      = {best['L_1']:.4f}")
    # print(f"L2      = {best['L_2']:.4f}")
    # print(f"theta1  = {best['theta_1_deg']:.2f} deg")
    # print(f"theta2  = {best['theta_2_deg']:.2f} deg")   
    # print(f"P0 rec  = {best['total_P0_recovery']:.6f}")
    # print(f"M_exit  = {best['M_exit']:.4f}")

    """
    ============================================================================================
    """

    print("\n[1] Extended Analysis with Channel Reflections & Terminal Normal Shock")
    analyse_intake4(M_inf=4.35, theta_1_deg = 10, L_1=1.925, y_cowl=1.2, delta_cowl_deg=4)

    best = optimise_intake(M_inf=4.35, y_cowl=1.2, delta_cowl_deg=4, L1_range=(0.01, 3), 
                           theta1_range=(1.0, 20.0), max_oblique_shocks= 10, M_exit_max=0.95, L1_step=0.005, print_all=True)
    print("\nBest configuration")
    print("------------------")
    print(f"L1      = {best['L_1']:.4f}")
    print(f"L2      = {best['L_2']:.4f}")
    print(f"theta1  = {best['theta_1_deg']:.2f} deg")
    print(f"theta2  = {best['theta_2_deg']:.2f} deg")   
    print(f"P0 rec  = {best['total_P0_recovery']:.6f}")
    print(f"M_exit  = {best['M_exit']:.4f}")