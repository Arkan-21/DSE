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
    total_turn_local = 0.6687 * M_inf **3 - 8.4697 * M_inf**2 + 36.463 * M_inf - 28.726 

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
    # This robustly locates the weak-shock root for any Mach/theta combination
    # without assuming a fixed upper bound (the function is non-monotone).
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


# ---------------------------------------------------------------------------
# Oswatitsch equal-shock-strength distribution
# ---------------------------------------------------------------------------

def oswatitsch_deflections(M_inf: float, n: int) -> list:
    """
    Equal-shock-strength (Oswatitsch) criterion - Eq. (4).

    Find the target normal Mach component Mn such that, when Mn is held
    equal across all n oblique shocks, the flow remains supersonic after
    the last ramp (M_out > 1.02) and compression is maximised.

    Returns a list of n ramp deflection angles [degrees].
    """
    def _simulate(Mn_target):
        """Walk n equal-Mn oblique shocks; return (thetas, M_after_last)."""
        M_cur = M_inf
        thetas_out = []
        for _ in range(n):
            if Mn_target >= M_cur:          # shock would be detached
                return None, None
            beta_rad = np.arcsin(Mn_target / M_cur)
            beta_deg = np.degrees(beta_rad)
            mu_deg   = np.degrees(np.arcsin(1.0 / M_cur))
            if beta_deg <= mu_deg + 0.05:   # too close to Mach angle
                return None, None
            g = GAMMA; b = beta_rad
            def eq(th, _b=b, _M=M_cur):
                return (np.tan(np.radians(th)) -
                        2.0 / np.tan(_b) *
                        (_M**2 * np.sin(_b)**2 - 1.0) /
                        (_M**2 * (g + np.cos(2.0 * _b)) + 2.0))
            try:
                theta = brentq(eq, 0.01, 45.0)
            except ValueError:
                return None, None
            thetas_out.append(theta)
            M_cur = post_oblique_mach(M_cur, beta_deg, theta)
            if M_cur <= 1.0:
                return None, None
        return thetas_out, M_cur

    # Bisect to find the highest Mn that keeps M_out > 1.02 after all n ramps.
    # Upper bound: conservative 95% of M_inf avoids arcsin domain issues.
    Mn_lo = 1.01
    Mn_hi = min(M_inf * 0.95, M_inf - 0.1)

    for _ in range(60):
        Mn_mid = (Mn_lo + Mn_hi) / 2.0
        _, M_out = _simulate(Mn_mid)
        if M_out is None or M_out <= 1.02:
            Mn_hi = Mn_mid      # too much compression -> reduce Mn
        else:
            Mn_lo = Mn_mid      # still supersonic -> can try more compression

    thetas, _ = _simulate(Mn_lo)
    if thetas is None:
        # Fallback: equal angular split of the Prandtl-Meyer angle
        nu_total = optimum_total_turn_angle(M_inf, n)
        thetas = [nu_total / n] * n
    return thetas

print(oswatitsch_deflections(3.0, 2))

# ---------------------------------------------------------------------------
# Reflected shock estimate (Eq. 6)
# ---------------------------------------------------------------------------

def reflected_shock_angle(betas: list, thetas: list,
                           delta_cowl: float = 0.0) -> float:
    """
    First estimate of the reflected shock angle beta_s in the isolator
    section - Eq. (6):

        beta_s = sum(beta_n) - sum(theta_n) + delta_cowl
    """
    return sum(betas) - sum(thetas) + delta_cowl


# ---------------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------------

def analyse_intake(M_inf: float, n: int,
                   thetas_override=None,
                   delta_cowl: float = 0.0,
                   verbose: bool = True) -> dict:
    """
    Full mixed-compression intake analysis.

    Parameters
    ----------
    M_inf           : freestream Mach number (>= 1)
    n               : number of compression ramps
    thetas_override : optional list of n ramp deflection angles [deg];
                      if None, the Oswatitsch equal-strength criterion is used
    delta_cowl      : cowl lip deflection angle [deg] for Eq. (6)
    verbose         : print formatted results table

    Returns
    -------
    dict with all computed quantities per stage
    """
    assert M_inf >= 1.0, "Freestream Mach number must be >= 1"
    assert n >= 1,       "Number of ramps must be >= 1"

    nu_opt = optimum_total_turn_angle(M_inf, n)

    if thetas_override is not None:
        assert len(thetas_override) == n, (
            f"Expected {n} deflection angles, got {len(thetas_override)}")
        thetas = list(thetas_override)
    else:
        thetas = oswatitsch_deflections(M_inf, n)

    # --- Walk through each oblique shock ---
    M_current      = M_inf
    total_P0_ratio = 1.0
    stages         = []

    for i, theta in enumerate(thetas):
        beta     = theta_beta_M(theta, M_current)
        Mn       = normal_mach(M_current, beta)
        P0_ratio = stagnation_pressure_ratio(Mn)
        M_next   = post_oblique_mach(M_current, beta, theta)

        stages.append({
            "stage":     f"Ramp {i + 1}",
            "M_in":      M_current,
            "theta_deg": theta,
            "beta_deg":  beta,
            "Mn":        Mn,
            "M_out":     M_next,
            "P0_ratio":  P0_ratio,
        })
        total_P0_ratio *= P0_ratio
        M_current = M_next

    # --- Final normal shock at cowl lip ---
    M_before_ns = M_current
    P0_ns       = stagnation_pressure_ratio(M_before_ns)
    g           = GAMMA
    M_after_ns  = np.sqrt(
        ((g - 1.0) * M_before_ns**2 + 2.0) /
        (2.0 * g * M_before_ns**2 - (g - 1.0))
    )
    total_P0_ratio *= P0_ns

    stages.append({
        "stage":     "Normal shock",
        "M_in":      M_before_ns,
        "theta_deg": 90.0,
        "beta_deg":  90.0,
        "Mn":        M_before_ns,
        "M_out":     M_after_ns,
        "P0_ratio":  P0_ns,
    })

    # --- Reflected shock angle ---
    betas_oblique  = [s["beta_deg"]  for s in stages[:-1]]
    thetas_oblique = [s["theta_deg"] for s in stages[:-1]]
    beta_ref       = reflected_shock_angle(betas_oblique, thetas_oblique,
                                           delta_cowl)

    results = {
        "M_inf":              M_inf,
        "n_ramps":            n,
        "nu_optimum_deg":     nu_opt,
        "total_deflection":   sum(thetas),
        "stages":             stages,
        "total_P0_recovery":  total_P0_ratio,
        "M_exit":             M_after_ns,
        "beta_reflected_deg": beta_ref,
    }

    if verbose:
        print("=" * 65)
        print(f"  Ramjet Mixed-Compression Intake Analysis")
        print(f"  M_inf = {M_inf:.2f}   |   {n} ramp(s)   |   gamma = {GAMMA}")
        print("=" * 65)
        print(f"  Prandtl-Meyer optimum total turn angle  nu = {nu_opt:.2f} deg")
        print(f"  Applied total deflection                   = {sum(thetas):.2f} deg")
        print()
        hdr = (f"  {'Stage':<16} {'M_in':>6} {'theta':>7} {'beta':>7}"
               f" {'Mn':>6} {'M_out':>7} {'P0 ratio':>10}")
        print(hdr)
        print("  " + "-" * 61)
        for s in stages:
            print(f"  {s['stage']:<16} {s['M_in']:>6.4f} "
                  f"{s['theta_deg']:>7.2f} {s['beta_deg']:>7.2f} "
                  f"{s['Mn']:>6.4f} {s['M_out']:>7.4f} "
                  f"{s['P0_ratio']:>10.6f}")
        print("  " + "-" * 61)
        print(f"  {'TOTAL P0 recovery':<40} {total_P0_ratio:>10.6f}")
        print(f"  {'Exit Mach (into subsonic diffuser)':<40} {M_after_ns:>10.4f}")
        print(f"  {'Reflected shock angle beta_s (Eq.6)':<40} {beta_ref:>9.2f} deg")
        print("=" * 65)

    return results



# ---------------------------------------------------------------------------
# Parametric sweep over ramp angles (replicates Table 1 of the paper)
# ---------------------------------------------------------------------------

def parametric_sweep(M_inf: float, n: int,
                     theta1_range: tuple, theta2_range: tuple,
                     step: float = 1.0) -> None:
    """
    Sweep combinations of ramp deflection angles and print a summary table.
    Currently supports n = 2 ramps (as studied in the paper).
    """
    assert n == 2, "Parametric sweep currently supports n=2 ramps only."
    t1_vals = np.arange(theta1_range[0], theta1_range[1] + step, step)
    t2_vals = np.arange(theta2_range[0], theta2_range[1] + step, step)

    print(f"\n  Parametric sweep  M_inf={M_inf},  n={n} ramps")
    print(f"  {'theta1':>7} {'theta2':>7} {'theta_tot':>10}"
          f" {'P0 recovery':>13} {'M_exit':>8}")
    print("  " + "-" * 52)

    best = {"P0": 0.0}
    for t1 in t1_vals:
        for t2 in t2_vals:
            try:
                r = analyse_intake(M_inf, n,
                                   thetas_override=[t1, t2],
                                   verbose=False)
                P0 = r["total_P0_recovery"]
                Me = r["M_exit"]
                print(f"  {t1:>7.1f} {t2:>7.1f} {t1+t2:>10.1f}"
                      f" {P0:>13.6f} {Me:>8.4f}")
                if P0 > best["P0"]:
                    best = {"P0": P0, "t1": t1, "t2": t2, "M_exit": Me}
            except Exception:
                pass

    print("  " + "-" * 52)
    print(f"  Best config: theta1={best['t1']:.1f} deg, theta2={best['t2']:.1f} deg,"
          f"  P0={best['P0']:.6f},  M_exit={best['M_exit']:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    #print("\n[1]  Design point from paper: M=3, 2 ramps, Oswatitsch criterion")
    #analyse_intake(M_inf=3.0, n=2)

    print("\n[2]  Paper optimal ramp angles: theta1=9 deg, theta2=14 deg  (M=3)")
    analyse_intake(M_inf=3.0, n=2, thetas_override=[9.0, 14.0])

    # print("\n[3]  Equal-angle comparison case: theta1=11 deg, theta2=11 deg  (M=3)")
    # print("     Note: the paper's 22+23 case exceeds the shock detachment limit")
    # print("     at the second ramp (max theta ~20.9 deg at M~1.89).")
    # print("     Using 11+11 as a valid equal-angle baseline instead.")
    # analyse_intake(M_inf=3.0, n=2, thetas_override=[11.0, 11.0])

    # print("\n[4]  M=4, 3 ramps, Oswatitsch criterion")
    # analyse_intake(M_inf=4.0, n=3)

    # print("\n[5]  Parametric sweep: theta1 in [5,15 deg], theta2 in [10,20 deg]")
    # parametric_sweep(M_inf=3.0, n=2,
    #                  theta1_range=(5, 15),
    #                  theta2_range=(10, 20),
    #                  step=1.0)
    
