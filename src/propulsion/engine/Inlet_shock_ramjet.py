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

def theta_from_beta_M(beta_deg: float, M: float) -> float:
    """
    Compute flow deflection angle theta [deg]
    from shock angle beta [deg] and upstream Mach number M.

    Uses the theta-beta-M relation directly.
    """
    g = GAMMA
    beta = np.radians(beta_deg)

    tan_theta = (
        2.0 / np.tan(beta)
        * (M**2 * np.sin(beta)**2 - 1.0)
        / (M**2 * (g + np.cos(2.0 * beta)) + 2.0)
    )

    return np.degrees(np.arctan(tan_theta))

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
                   delta_cowl: float = 0.0, y_cowl: float = 0.0,
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

    # # --- Final normal shock at cowl lip ---
    # M_before_ns = M_current
    # P0_ns       = stagnation_pressure_ratio(M_before_ns)
    # g           = GAMMA
    # M_after_ns  = np.sqrt(
    #     ((g - 1.0) * M_before_ns**2 + 2.0) /
    #     (2.0 * g * M_before_ns**2 - (g - 1.0))
    # )
    # total_P0_ratio *= P0_ns

    # stages.append({
    #     "stage":     "Normal shock",
    #     "M_in":      M_before_ns,
    #     "theta_deg": 90.0,
    #     "beta_deg":  90.0,
    #     "Mn":        M_before_ns,
    #     "M_out":     M_after_ns,
    #     "P0_ratio":  P0_ns,
    # })

    beta1 = stages[0]["beta_deg"]
    beta2 = stages[1]["beta_deg"]


    # --- Reflected shock angle ---
    betas_oblique  = [s["beta_deg"]  for s in stages]
    thetas_oblique = [s["theta_deg"] for s in stages]
    #beta_ref       = reflected_shock_angle(betas_oblique, thetas_oblique, delta_cowl)

    L_1 = 1.0
    L_2 = 1.0

    x_c = L_1 * (np.tan(np.radians(thetas[0]))-np.tan(np.radians(thetas[0]+beta2)))/ (np.tan(np.radians(beta1)) - np.tan(np.radians(thetas[0]+beta2)))
    #y_c = x_c * np.tan(np.radians(beta1))
    
    y_2 = L_1 * np.tan(np.radians(thetas[0])) +L_2 * np.tan(np.radians(thetas[0]+thetas[1]))
    x_2 = L_1 + L_2


    psi = np.degrees(
    np.arctan2(y_2 - y_cowl , x_2 - x_c))
    
    cowl_deflection = thetas[0] + thetas[1] - delta_cowl

    def residual(delta):
        beta_rs = theta_beta_M(theta_deg=cowl_deflection, M=M_current)
        beta_rs_frame = thetas[0] + thetas[1] - beta_rs
        return psi - beta_rs_frame

    x_c = brentq(residual , 0.01 , x_2 - 0.01)


    M_before_rs = M_current
    M_before_rs_normal = normal_mach(M_before_rs, beta_ref)
    P0_rs       = stagnation_pressure_ratio(M_before_rs_normal)
    theta_rs     = theta_from_beta_M(beta_ref, M_before_rs)
    M_after_rs  = post_oblique_mach(M_before_rs, beta_ref, theta_rs)  # theta=0 for reflected shock


    total_P0_ratio *= P0_rs

    stages.append({
        "stage":     "Reflected shock",
        "M_in":      M_before_rs,
        "theta_deg": theta_rs,
        "beta_deg":  beta_ref,
        "Mn":        M_before_rs_normal,
        "M_out":     M_after_rs,
        "P0_ratio":  P0_rs,
    })


    results = {
        "M_inf":              M_inf,
        "n_ramps":            n,
        "nu_optimum_deg":     nu_opt,
        "total_deflection":   sum(thetas),
        "stages":             stages,
        "total_P0_recovery":  total_P0_ratio,
        "M_exit":             M_after_rs,
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
        print(f"  {'Exit Mach (into subsonic diffuser)':<40} {M_after_rs:>10.4f}")
        print(f"  {'Reflected shock angle beta_s (Eq.6)':<40} {beta_ref:>9.2f} deg")
        print("=" * 65)

    return results

def analyse_intake2(
    M_inf: float,
    n: int,
    L_1: float,
    L_2: float,
    y_cowl: float,
    thetas_override=None,
    verbose: bool = True,
) -> dict:
    """
    Full mixed-compression intake analysis.

    Geometry (all coordinates origin at ramp 1 leading edge, x forward, y up):

        Ramp 1 tip  : (L_1,  L_1 * tan(theta_1))
        Ramp 2 tip  : (L_1 + L_2,  L_1*tan(theta_1) + L_2*tan(theta_1+theta_2))
        Cowl lip    : intersection of shock 1 and shock 2, at height y_cowl

    Shock sequence:
        1. Oblique shock from ramp 1 origin  → travels at angle beta_1
        2. Oblique shock from ramp 1 tip     → travels at angle (theta_1 + beta_2)
           These two shocks intersect at the cowl lip (x_c, y_cowl).
        3. Reflected shock from cowl lip, iterated so that it hits ramp 2 tip
           (L_1 + L_2, y_2).

    Parameters
    ----------
    M_inf           : freestream Mach number (>= 1)
    n               : number of compression ramps (currently must be 2)
    L_1             : length of ramp 1 (streamwise, same units as y_cowl)
    L_2             : length of ramp 2
    y_cowl          : height of the cowl lip above the ramp-1 base line [m / normalised]
    thetas_override : optional list of n ramp deflection angles [deg];
                      if None, the Oswatitsch equal-strength criterion is used
    verbose         : print formatted results table

    Returns
    -------
    dict with all computed quantities per stage plus geometric quantities
    """
    assert M_inf >= 1.0, "Freestream Mach number must be >= 1"
    assert n == 2,       "This implementation supports exactly 2 ramps"

    # ------------------------------------------------------------------
    # 1.  Deflection angles
    # ------------------------------------------------------------------
    nu_opt = optimum_total_turn_angle(M_inf, n)

    if thetas_override is not None:
        assert len(thetas_override) == n, (
            f"Expected {n} deflection angles, got {len(thetas_override)}")
        thetas = list(thetas_override)
    else:
        thetas = oswatitsch_deflections(M_inf, n)

    theta_1, theta_2 = thetas[0], thetas[1]

    # ------------------------------------------------------------------
    # 2.  Oblique shock 1  (from ramp-1 leading edge at origin)
    # ------------------------------------------------------------------
    beta_1   = theta_beta_M(theta_1, M_inf)
    Mn_1     = normal_mach(M_inf, beta_1)
    P0_rat_1 = stagnation_pressure_ratio(Mn_1)
    M_2      = post_oblique_mach(M_inf, beta_1, theta_1)   # flow after shock 1

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
    # 3.  Oblique shock 2  (from ramp-1 tip, in the M_2 stream)
    # ------------------------------------------------------------------
    beta_2   = theta_beta_M(theta_2, M_2)
    Mn_2     = normal_mach(M_2, beta_2)
    P0_rat_2 = stagnation_pressure_ratio(Mn_2)
    M_3      = post_oblique_mach(M_2, beta_2, theta_2)     # flow after shock 2

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
    # 4.  Cowl-lip location
    #     Shock 1 leaves the origin at angle beta_1 (measured from x-axis).
    #     Shock 2 leaves ramp-1 tip (L_1, L_1*tan(theta_1)) at angle
    #     (theta_1 + beta_2) from the x-axis.
    #
    #     Intersection (x_c, y_c):
    #       y_c = x_c * tan(beta_1)                              [shock 1]
    #       y_c = L_1*tan(theta_1) + (x_c - L_1)*tan(theta_1+beta_2)  [shock 2]
    #
    #     We solve for x_c, then use y_cowl as the *imposed* cowl height to
    #     verify consistency (or the user may set y_cowl = y_c for a perfect
    #     intake).  The reflected shock geometry uses the actual cowl point
    #     (x_c, y_cowl) as its origin.
    # ------------------------------------------------------------------
    tan_b1  = np.tan(np.radians(beta_1))
    tan_s2  = np.tan(np.radians(theta_1 + beta_2))   # shock-2 slope in global frame
    tan_t1  = np.tan(np.radians(theta_1))

    # Shock-1 line : y = x * tan_b1
    # Shock-2 line : y = L_1*tan_t1 + (x - L_1)*tan_s2
    # Setting equal:  x*tan_b1 = L_1*tan_t1 + (x-L_1)*tan_s2
    #                 x*(tan_b1 - tan_s2) = L_1*(tan_t1 - tan_s2)
    x_c = L_1 * (tan_t1 - tan_s2) / (tan_b1 - tan_s2)
    y_c = x_c * tan_b1   # geometric intersection height

    # Cowl lip is at the intersection x-position but at the user-supplied height
    # (allows the cowl to be raised/lowered relative to the ideal intersection).
    cowl = np.array([x_c, y_cowl])

    # ------------------------------------------------------------------
    # 5.  Ramp-2 tip location  (target for the reflected shock)
    # ------------------------------------------------------------------
    y_2      = L_1 * tan_t1 + L_2 * np.tan(np.radians(theta_1 + theta_2))
    tip_ramp2 = np.array([L_1 + L_2, y_2])

    # ------------------------------------------------------------------
    # 6.  Required reflected-shock angle
    #     The reflected shock travels from the cowl lip (x_c, y_cowl) to
    #     ramp-2 tip (L_1+L_2, y_2).  Its angle measured FROM THE x-AXIS is:
    #
    #       alpha_line = arctan2(y_cowl - y_2, x_c - (L_1+L_2))
    #                  (negative slope, pointing downward toward the ramp)
    #
    #     The incoming flow after shock 2 is inclined at -(theta_1+theta_2)
    #     from the x-axis (running along the ramp-2 surface).  The shock
    #     angle beta_ref is measured from that flow direction, so:
    #
    #       beta_ref = alpha_line - (-(theta_1+theta_2))
    #                = alpha_line + (theta_1+theta_2)
    #
    #     where alpha_line is the angle of the shock ray measured upward from
    #     the (negative-x) direction, i.e. the supplement of the geometric
    #     slope.
    # ------------------------------------------------------------------
    dx = tip_ramp2[0] - cowl[0]   # positive (ramp tip is downstream)
    dy = cowl[1]      - tip_ramp2[1]   # positive when cowl is above ramp tip

    # Angle of the reflected-shock ray from the local (post-shock-2) flow
    # direction.  The flow runs at angle +(theta_1+theta_2) above x-axis;
    # the shock ray points downward at angle alpha_geom below x-axis.
    alpha_geom = np.degrees(np.arctan2(dy, dx))   # deg above x-axis (should be >0)

    total_deflection_upstream = theta_1 + theta_2
    # beta_ref = angle between shock ray and the incoming streamline
    beta_ref = alpha_geom + total_deflection_upstream  # both measured from x-axis

    # ------------------------------------------------------------------
    # 7.  Flow properties through the reflected shock
    # ------------------------------------------------------------------
    mu3 = np.degrees(np.arcsin(1.0/M_3))

    if beta_ref <= mu3:
        raise ValueError("Reflected shock angle below Mach angle")
    Mn_ref     = normal_mach(M_3, beta_ref)
    P0_rat_ref = stagnation_pressure_ratio(Mn_ref)
    theta_ref  = theta_from_beta_M(beta_ref, M_3)   # flow turning at reflected shock
    M_4        = post_oblique_mach(M_3, beta_ref, theta_ref)

    stage_ref = {
        "stage":     "Reflected shock",
        "M_in":      M_3,
        "theta_deg": theta_ref,
        "beta_deg":  beta_ref,
        "Mn":        Mn_ref,
        "M_out":     M_4,
        "P0_ratio":  P0_rat_ref,
    }

    # ------------------------------------------------------------------
    # 8.  Totals
    # ------------------------------------------------------------------
    total_P0 = P0_rat_1 * P0_rat_2 * P0_rat_ref
    stages   = [stage_1, stage_2, stage_ref]

    # ------------------------------------------------------------------
    # 9.  Verbose output
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 65)
        print(f"  Ramjet Mixed-Compression Intake Analysis")
        print(f"  M_inf = {M_inf:.2f}   |   {n} ramp(s)   |   gamma = {GAMMA}")
        print("=" * 65)
        print(f"  Prandtl-Meyer optimum total turn  nu  = {nu_opt:.2f} deg")
        print(f"  Applied total deflection               = {sum(thetas):.2f} deg")
        print()
        print(f"  Geometry")
        print(f"    Ramp 1 tip     : ({L_1:.4f}, {L_1*tan_t1:.4f})")
        print(f"    Ramp 2 tip     : ({tip_ramp2[0]:.4f}, {tip_ramp2[1]:.4f})")
        print(f"    Shock intersect: ({x_c:.4f}, {y_c:.4f})")
        print(f"    Cowl lip       : ({cowl[0]:.4f}, {cowl[1]:.4f})")
        print(f"    Reflected shock angle (from flow) beta_ref = {beta_ref:.2f} deg")
        print()
        hdr = (f"  {'Stage':<18} {'M_in':>6} {'theta':>7} {'beta':>7}"
               f" {'Mn':>6} {'M_out':>7} {'P0 ratio':>10}")
        print(hdr)
        print("  " + "-" * 63)
        for s in stages:
            print(f"  {s['stage']:<18} {s['M_in']:>6.4f} "
                  f"{s['theta_deg']:>7.2f} {s['beta_deg']:>7.2f} "
                  f"{s['Mn']:>6.4f} {s['M_out']:>7.4f} "
                  f"{s['P0_ratio']:>10.6f}")
        print("  " + "-" * 63)
        print(f"  {'TOTAL P0 recovery':<44} {total_P0:>10.6f}")
        print(f"  {'Exit Mach (after reflected shock)':<44} {M_4:>10.4f}")
        print("=" * 65)

    return {
        "M_inf":                M_inf,
        "n_ramps":              n,
        "nu_optimum_deg":       nu_opt,
        "total_deflection_deg": sum(thetas),
        "thetas_deg":           thetas,
        # geometry
        "x_c":                  x_c,
        "y_c_geometric":        y_c,
        "cowl_lip":             cowl.tolist(),
        "ramp2_tip":            tip_ramp2.tolist(),
        "beta_reflected_deg":   beta_ref,
        # shock stages
        "stages":               stages,
        # totals
        "total_P0_recovery":    total_P0,
        "M_exit":               M_4,
    }

def analyse_intake3(
    M_inf:    float,
    L_1:      float,
    y_cowl:   float,
    thetas_override = None,
    delta_cowl_deg: float = 0.0,
    verbose:  bool = True,
) -> dict:
    """
    Full mixed-compression intake analysis — 2-ramp external + reflected shock.

    Fixed inputs
    ------------
    M_inf          : freestream Mach number (>= 1)
    L_1            : length of ramp 1  [m or normalised]
    theta_1_deg    : ramp 1 deflection angle  [deg]
    theta_2_deg    : ramp 2 deflection angle  [deg]
    y_cowl         : cowl-lip height above the ramp-1 baseline  [same units as L_1]
    delta_cowl_deg : deflection angle of the cowl inner surface w.r.t. the
                     incoming (post-shock-2) flow  [deg].
                     delta_cowl = 0  → cowl surface is parallel to the
                     post-shock-2 streamlines (no reflected shock).
                     delta_cowl > 0  → cowl turns the flow, generating a
                     reflected oblique shock.

    Computed outputs  (all in the return dict)
    ------------------------------------------
    Shocks 1 & 2 : standard oblique-shock properties
    x_c          : cowl-lip x-position (shock-1 / shock-2 intersection)
    beta_ref     : reflected shock angle from the local flow direction  [deg]
    landing_pt   : (x, y) where the reflected shock hits ramp 2
    L_2          : ramp-2 length  = distance along ramp-2 surface from
                   ramp-1 tip to the reflected-shock landing point
    total_P0     : overall stagnation-pressure recovery

    Geometry (origin at ramp-1 leading edge, x downstream, y up)
    -------------------------------------------------------------
    Ramp-1 tip  : P1 = (L_1,  L_1·tan θ₁)
    Ramp-2 surf : starts at P1, runs at angle (θ₁+θ₂) from x-axis
    Cowl lip    : C  = (x_c, y_cowl)
                  x_c is the intersection of shock-1 and shock-2 lines
    Reflected   : leaves C at (θ₁+θ₂ − β_ref) from x-axis (downward),
    shock ray     hits ramp-2 surface at landing_pt → defines L_2
    """
    assert M_inf >= 1.0, "Freestream Mach must be >= 1"

    theta_1 = thetas_override[0]
    theta_2 = thetas_override[1]

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
    # 2.  Oblique shock 2  (ramp-1 tip, M_2 stream)
    # ------------------------------------------------------------------
    beta_2   = theta_beta_M(theta_2, M_2)
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
    # 3.  Cowl-lip position
    #
    #   Shock-1 line (global frame, from origin):
    #       y = x · tan(β₁)
    #
    #   Shock-2 line (from ramp-1 tip P1 = (L_1, L_1·tan θ₁)):
    #       y = L_1·tan θ₁  +  (x − L_1)·tan(θ₁ + β₂)
    #
    #   The cowl x-position x_c is their intersection; y_cowl is fixed.
    # ------------------------------------------------------------------
    tan_b1 = np.tan(np.radians(beta_1))
    tan_s2 = np.tan(np.radians(theta_1 + beta_2))   # shock-2 global slope
    tan_t1 = np.tan(np.radians(theta_1))

    # x·tan_b1 = L_1·tan_t1 + (x − L_1)·tan_s2
    # x·(tan_b1 − tan_s2) = L_1·(tan_t1 − tan_s2)
    x_c   = L_1 * (tan_t1 - tan_s2) / (tan_b1 - tan_s2)
    y_c   = x_c * tan_b1             # geometric intersection height
    cowl  = np.array([x_c, y_cowl])  # user-supplied height (may differ from y_c)

    # ------------------------------------------------------------------
    # 4.  Reflected shock angle
    #
    #   The post-shock-2 flow runs at angle (θ₁+θ₂) above the x-axis.
    #   The cowl inner surface is inclined at delta_cowl w.r.t. that flow,
    #   so the reflected shock is an oblique shock in M_3 with deflection
    #   angle delta_cowl:
    #
    #       β_ref = θ_β_M(delta_cowl, M_3)    [measured from local flow]
    #
    #   In the global frame the shock ray leaves the cowl lip heading
    #   downward at:
    #       angle_ray_global = (θ₁+θ₂) − β_ref
    #   (positive = above x-axis; will be negative / shallow if β_ref is large)
    # ------------------------------------------------------------------
    deflection_ref = (theta_1 + theta_2) - delta_cowl_deg

    # Guard: check deflection_ref is below the detachment limit at M_3
    # Maximum deflection = theta at which d(theta)/d(beta) = 0
    g = GAMMA
    betas = np.radians(np.arange(np.degrees(np.arcsin(1.0/M_3)) + 0.1, 90.0, 0.05))
    thetas_scan = np.degrees(np.arctan(
        2.0 / np.tan(betas) *
        (M_3**2 * np.sin(betas)**2 - 1.0) /
        (M_3**2 * (g + np.cos(2.0 * betas)) + 2.0)
    ))
    theta_max = float(np.nanmax(thetas_scan))
    if deflection_ref >= theta_max:
        raise ValueError(
            f"Reflected shock deflection ({deflection_ref:.2f} deg) exceeds the "
            f"detachment limit ({theta_max:.2f} deg) at M_3={M_3:.4f}.\n"
            f"Increase delta_cowl_deg above {(theta_1+theta_2) - theta_max:.2f} deg "
            f"to bring the deflection below the limit.")
    

    beta_ref   = theta_beta_M(deflection_ref, M_3)   # shock angle from local flow
    Mn_ref     = normal_mach(M_3, beta_ref)
    P0_rat_ref = stagnation_pressure_ratio(Mn_ref)
    theta_ref  = theta_from_beta_M(beta_ref, M_3)    # actual flow turning
    M_4        = post_oblique_mach(M_3, beta_ref, theta_ref)

    stage_ref = {
        "stage":     "Reflected shock",
        "M_in":      M_3,
        "theta_deg": theta_ref,
        "beta_deg":  beta_ref,
        "Mn":        Mn_ref,
        "M_out":     M_4,
        "P0_ratio":  P0_rat_ref,
    }

    # ------------------------------------------------------------------
    # 5.  Reflected-shock landing point on ramp 2  →  L_2
    #
    #   Shock ray from C = (x_c, y_cowl):
    #       direction angle from x-axis:  φ = (θ₁+θ₂) − β_ref   [deg]
    #       parametric:  x = x_c + t·cos φ
    #                    y = y_cowl + t·sin φ      (t > 0 downstream)
    #
    #   Ramp-2 surface from P1 = (L_1, y_P1):
    #       y = y_P1 + (x − L_1)·tan(θ₁+θ₂)
    #
    #   Intersect → solve for t, then landing point and L_2.
    # ------------------------------------------------------------------
    phi_deg = (theta_1 + theta_2) - beta_ref          # global ray angle [deg]
    phi_rad = np.radians(phi_deg)
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)

    y_P1       = L_1 * tan_t1
    tan_ramp2  = np.tan(np.radians(theta_1 + theta_2))

    # y_cowl + t·sin_phi  =  y_P1 + (x_c + t·cos_phi − L_1)·tan_ramp2
    # t·(sin_phi − cos_phi·tan_ramp2)  =  y_P1 + (x_c − L_1)·tan_ramp2 − y_cowl
    denom = sin_phi - cos_phi * tan_ramp2
    numer = y_P1 + (x_c - L_1) * tan_ramp2 - y_cowl

    assert abs(denom) > 1e-12, (
        "Reflected shock ray is parallel to ramp-2 surface — no intersection.")
    t_land = numer / denom
    assert t_land > 0, (
        f"Reflected shock lands upstream (t={t_land:.4f}); "
        "check delta_cowl / geometry inputs.")

    x_land = x_c + t_land * cos_phi
    y_land = y_cowl + t_land * sin_phi
    landing_pt = np.array([x_land, y_land])

    # L_2 = distance along ramp-2 surface from P1 to landing point
    # (the ramp surface is inclined, so use the straight-line distance)
    L_2 = np.hypot(x_land - L_1, y_land - y_P1)

    # ------------------------------------------------------------------
    # 6.  Totals
    # ------------------------------------------------------------------
    total_P0 = P0_rat_1 * P0_rat_2 * P0_rat_ref
    stages   = [stage_1, stage_2, stage_ref]

    # ------------------------------------------------------------------
    # 7.  Verbose output
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 65)
        print(f"  Ramjet Mixed-Compression Intake Analysis")
        print(f"  M_inf = {M_inf:.2f}   |   gamma = {GAMMA}")
        print("=" * 65)
        print(f"  Inputs")
        print(f"    theta_1         = {theta_1:.2f} deg")
        print(f"    theta_2         = {theta_2:.2f} deg")
        print(f"    L_1             = {L_1:.4f}")
        print(f"    y_cowl          = {y_cowl:.4f}")
        print(f"    delta_cowl      = {delta_cowl_deg:.2f} deg")
        print()
        print(f"  Geometry (computed)")
        print(f"    Ramp-1 tip P1   : ({L_1:.4f},  {y_P1:.4f})")
        print(f"    Shock intersection (geometric y_c = {y_c:.4f})")
        print(f"    Cowl lip C      : ({x_c:.4f},  {y_cowl:.4f})")
        print(f"    Reflected-shock ray angle (global) phi = {phi_deg:.2f} deg")
        print(f"    Landing point   : ({x_land:.4f},  {y_land:.4f})")
        print(f"    L_2 (computed)  = {L_2:.4f}")
        print()
        hdr = (f"  {'Stage':<18} {'M_in':>6} {'theta':>7} {'beta':>7}"
               f" {'Mn':>6} {'M_out':>7} {'P0 ratio':>10}")
        print(hdr)
        print("  " + "-" * 63)
        for s in stages:
            print(f"  {s['stage']:<18} {s['M_in']:>6.4f} "
                  f"{s['theta_deg']:>7.2f} {s['beta_deg']:>7.2f} "
                  f"{s['Mn']:>6.4f} {s['M_out']:>7.4f} "
                  f"{s['P0_ratio']:>10.6f}")
        print("  " + "-" * 63)
        print(f"  {'TOTAL P0 recovery':<44} {total_P0:>10.6f}")
        print(f"  {'Exit Mach (after reflected shock)':<44} {M_4:>10.4f}")
        print("=" * 65)

    return {
        "M_inf":                M_inf,
        "thetas_deg":           [theta_1, theta_2],
        "delta_cowl_deg":       delta_cowl_deg,
        # geometry
        "L_1":                  L_1,
        "L_2":                  L_2,
        "y_cowl":               y_cowl,
        "y_c_geometric":        y_c,
        "cowl_lip":             cowl.tolist(),
        "ramp1_tip":            [L_1, y_P1],
        "landing_pt":           landing_pt.tolist(),
        "phi_ray_deg":          phi_deg,
        "beta_reflected_deg":   beta_ref,
        # shock stages
        "stages":               stages,
        # totals
        "total_P0_recovery":    total_P0,
        "M_exit":               M_4,
    }

def analyse_intake4(
    M_inf:          float,
    L_1:            float,
    theta_1_deg:    float,
    y_cowl:         float,
    delta_cowl_deg: float = 0.0,
    verbose:        bool  = True,
) -> dict:
    """
    Full mixed-compression intake analysis — 2-ramp + reflected shock.

    Design logic
    ------------
    1. theta_1, L_1, y_cowl are fixed → shock 1 gives M_2 and places the
       cowl lip at C = (x_c, y_cowl) on the shock-1 line.
    2. Shock 2 is *required* to pass through C from ramp-1 tip P1.
       The geometry of the P1→C line gives beta_2 (global), from which
       theta_2 is back-calculated via the oblique-shock relation at M_2.
    3. delta_cowl_deg (cowl plate angle below x-axis, positive downward)
       sets the reflected-shock deflection:
           deflection_ref = (theta_1 + theta_2) - delta_cowl_deg
    4. The reflected shock leaves C at the computed beta_ref and lands on
       ramp 2 at the landing point → L_2 is a computed output.

    Parameters
    ----------
    M_inf          : freestream Mach number (>= 1)
    L_1            : ramp-1 length  [m or normalised]
    theta_1_deg    : ramp-1 deflection angle  [deg]
    y_cowl         : cowl-lip height above the ramp-1 baseline  [same units]
    delta_cowl_deg : cowl inner-surface angle below the x-axis  [deg]
                     positive = tilted downward toward the flow
                     delta_cowl = theta_1+theta_2 → cowl parallel to flow (no shock)
                     delta_cowl = 0              → horizontal cowl (max deflection)
    verbose        : print formatted results table

    Computed outputs (return dict)
    ------------------------------
    theta_2_deg    : ramp-2 deflection angle back-calculated from geometry
    x_c, y_cowl   : cowl-lip coordinates
    L_2            : ramp-2 length (ramp-1 tip → reflected-shock landing point)
    landing_pt     : (x, y) where reflected shock hits ramp 2
    total_P0_recovery, M_exit, stages : aerodynamic results
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
    #
    #   Shock-1 passes through the origin at angle beta_1.
    #   The cowl lip C sits on this line at height y_cowl:
    #       x_c = y_cowl / tan(beta_1)
    # ------------------------------------------------------------------
    tan_b1 = np.tan(np.radians(beta_1))
    tan_t1 = np.tan(np.radians(theta_1))

    x_c  = y_cowl / tan_b1
    cowl = np.array([x_c, y_cowl])

    # Ramp-1 tip
    y_P1 = L_1 * tan_t1
    P1   = np.array([L_1, y_P1])

    assert x_c > L_1, (
        f"Cowl lip (x_c={x_c:.4f}) is upstream of ramp-1 tip (L_1={L_1:.4f}). "
        "Increase y_cowl or decrease L_1.")

    # ------------------------------------------------------------------
    # 3.  Back-calculate beta_2 and theta_2 from geometry
    #
    #   Shock 2 must pass through both P1 and C.
    #   Its slope in the global frame:
    #       tan(angle_global) = (y_cowl - y_P1) / (x_c - L_1)
    #       angle_global = arctan2(y_cowl - y_P1, x_c - L_1)   [deg]
    #
    #   The shock angle beta_2 is measured from the *local flow direction*
    #   which is inclined at theta_1 above the x-axis after shock 1:
    #       beta_2 = angle_global - theta_1
    #
    #   Then theta_2 follows from the oblique-shock theta-beta-M relation
    #   applied to M_2:
    #       theta_2 = theta_from_beta_M(beta_2, M_2)
    # ------------------------------------------------------------------
    angle_global_2 = np.degrees(np.arctan2(y_cowl - y_P1, x_c - L_1))
    beta_2         = angle_global_2 - theta_1          # shock angle from local flow
    theta_2        = theta_from_beta_M(beta_2, M_2)    # ramp-2 deflection (output)

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
    # 4.  Reflected shock angle
    #
    #   Post-shock-2 flow runs at (theta_1+theta_2) above the x-axis.
    #   Cowl inner surface is tilted delta_cowl below the x-axis.
    #   Total flow deflection seen by the reflected shock:
    #       deflection_ref = (theta_1 + theta_2) - delta_cowl_deg
    #
    #   Shock ray leaves C downward at global angle:
    #       phi = (theta_1 + theta_2) - beta_ref
    # ------------------------------------------------------------------
    deflection_ref = (theta_1 + theta_2) - delta_cowl_deg

    # Guard: check against detachment limit at M_3
    g = GAMMA
    betas_scan  = np.radians(np.arange(np.degrees(np.arcsin(1.0/M_3)) + 0.1, 90.0, 0.05))
    thetas_scan = np.degrees(np.arctan(
        2.0 / np.tan(betas_scan) *
        (M_3**2 * np.sin(betas_scan)**2 - 1.0) /
        (M_3**2 * (g + np.cos(2.0 * betas_scan)) + 2.0)
    ))
    theta_max = float(np.nanmax(thetas_scan))
    if deflection_ref >= theta_max:
        raise ValueError(
            f"Reflected shock deflection ({deflection_ref:.2f} deg) exceeds the "
            f"detachment limit ({theta_max:.2f} deg) at M_3={M_3:.4f}.\n"
            f"Increase delta_cowl_deg above "
            f"{(theta_1 + theta_2) - theta_max:.2f} deg.")

    beta_ref   = theta_beta_M(deflection_ref, M_3)
    Mn_ref     = normal_mach(M_3, beta_ref)
    P0_rat_ref = stagnation_pressure_ratio(Mn_ref)
    theta_ref  = theta_from_beta_M(beta_ref, M_3)
    M_4        = post_oblique_mach(M_3, beta_ref, theta_ref)

    stage_ref = {
        "stage":     "Reflected shock",
        "M_in":      M_3,
        "theta_deg": theta_ref,
        "beta_deg":  beta_ref,
        "Mn":        Mn_ref,
        "M_out":     M_4,
        "P0_ratio":  P0_rat_ref,
    }

    # ------------------------------------------------------------------
    # 5.  Reflected-shock landing point on ramp 2  →  L_2
    #
    #   Ray from C at global angle phi (downward):
    #       x = x_c + t·cos(phi)
    #       y = y_cowl + t·sin(phi)
    #
    #   Ramp-2 surface from P1 at angle (theta_1+theta_2):
    #       y = y_P1 + (x - L_1)·tan(theta_1+theta_2)
    #
    #   Solve for t → landing point → L_2 along ramp surface.
    # ------------------------------------------------------------------
    phi_deg   = (theta_1 + theta_2) - beta_ref
    phi_rad   = np.radians(phi_deg)
    cos_phi   = np.cos(phi_rad)
    sin_phi   = np.sin(phi_rad)
    tan_ramp2 = np.tan(np.radians(theta_1 + theta_2))

    denom  = sin_phi - cos_phi * tan_ramp2
    numer  = y_P1 + (x_c - L_1) * tan_ramp2 - y_cowl

    assert abs(denom) > 1e-12, (
        "Reflected shock ray is parallel to ramp-2 surface — no intersection.")
    t_land = numer / denom
    assert t_land > 0, (
        f"Reflected shock lands upstream (t={t_land:.4f}); "
        "check delta_cowl / geometry.")

    x_land    = x_c + t_land * cos_phi
    y_land    = y_cowl + t_land * sin_phi
    landing_pt = np.array([x_land, y_land])
    L_2        = x_land - L_1

    # ------------------------------------------------------------------
    # 6.  Totals
    # ------------------------------------------------------------------
    total_P0 = P0_rat_1 * P0_rat_2 * P0_rat_ref
    stages   = [stage_1, stage_2, stage_ref]

    # ------------------------------------------------------------------
    # 7.  Verbose output
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 65)
        print(f"  Ramjet Mixed-Compression Intake Analysis")
        print(f"  M_inf = {M_inf:.2f}   |   gamma = {GAMMA}")
        print("=" * 65)
        print(f"  Inputs")
        print(f"    theta_1         = {theta_1:.2f} deg")
        print(f"    L_1             = {L_1:.4f}")
        print(f"    y_cowl          = {y_cowl:.4f}")
        print(f"    delta_cowl      = {delta_cowl_deg:.2f} deg  (below x-axis)")
        print()
        print(f"  Derived geometry")
        print(f"    beta_1          = {beta_1:.2f} deg")
        print(f"    Cowl lip C      : ({x_c:.4f},  {y_cowl:.4f})")
        print(f"    Ramp-1 tip P1   : ({L_1:.4f},  {y_P1:.4f})")
        print(f"    beta_2 (local)  = {beta_2:.2f} deg  →  theta_2 = {theta_2:.2f} deg")
        print(f"    Deflection at cowl = (theta1+theta2) - delta_cowl"
              f" = {theta_1+theta_2:.2f} - {delta_cowl_deg:.2f}"
              f" = {deflection_ref:.2f} deg")
        print(f"    beta_ref        = {beta_ref:.2f} deg  (phi = {phi_deg:.2f} deg global)")
        print(f"    Landing point   : ({x_land:.4f},  {y_land:.4f})")
        print(f"    L_2 (computed)  = {L_2:.4f}")
        print()
        hdr = (f"  {'Stage':<18} {'M_in':>6} {'theta':>7} {'beta':>7}"
               f" {'Mn':>6} {'M_out':>7} {'P0 ratio':>10}")
        print(hdr)
        print("  " + "-" * 63)
        for s in stages:
            print(f"  {s['stage']:<18} {s['M_in']:>6.4f} "
                  f"{s['theta_deg']:>7.2f} {s['beta_deg']:>7.2f} "
                  f"{s['Mn']:>6.4f} {s['M_out']:>7.4f} "
                  f"{s['P0_ratio']:>10.6f}")
        print("  " + "-" * 63)
        print(f"  {'TOTAL P0 recovery':<44} {total_P0:>10.6f}")
        print(f"  {'Exit Mach (after reflected shock)':<44} {M_4:>10.4f}")
        print("=" * 65)

    return {
        "M_inf":                M_inf,
        "theta_1_deg":          theta_1,
        "theta_2_deg":          theta_2,        # computed output
        "delta_cowl_deg":       delta_cowl_deg,
        # geometry
        "L_1":                  L_1,
        "L_2":                  L_2,            # computed output
        "y_cowl":               y_cowl,
        "cowl_lip":             cowl.tolist(),
        "ramp1_tip":            P1.tolist(),
        "landing_pt":           landing_pt.tolist(),
        "beta_1_deg":           beta_1,
        "beta_2_deg":           beta_2,
        "phi_ray_deg":          phi_deg,
        "deflection_ref_deg":   deflection_ref,
        "beta_reflected_deg":   beta_ref,
        # shock stages
        "stages":               stages,
        # totals
        "total_P0_recovery":    total_P0,
        "M_exit":               M_4,
    }


def optimise_intake(
    M_inf,
    y_cowl,
    delta_cowl_deg,
    L1_range=(0.2, 2.0),
    theta1_range=(1.0, 20.0),
    L1_step=0.02,
    theta1_step=1,
    M_exit_max=1.5,
    print_all=False,
):
    """
    Sweep L1 and theta1 and find the maximum total pressure recovery.

    Constraints:
        - physically valid shock system
        - M_exit <= M_exit_max
        - theta_2 > 0
        - L_2 > 0
    """

    if print_all:
        print(
            f"{'L1':>8} {'theta1':>8} {'theta2':>8} "
            f"{'L2':>8} {'P0_rec':>12} {'M_exit':>10}"
        )
        print("-" * 65)
    best = None

    L1_values = np.arange(
        L1_range[0],
        L1_range[1] + L1_step,
        L1_step,
    )

    theta1_values = np.arange(
        theta1_range[0],
        theta1_range[1] + theta1_step,
        theta1_step,
    )

    for L1 in L1_values:
        for theta1 in theta1_values:

            try:
                r = analyse_intake4(
                    M_inf=M_inf,
                    L_1=L1,
                    theta_1_deg=theta1,
                    y_cowl=y_cowl,
                    delta_cowl_deg=delta_cowl_deg,
                    verbose=False,
                )

                # ----------------------------
                # Additional constraints
                # ----------------------------
                if r["M_exit"] > M_exit_max:
                    continue

                if r["theta_2_deg"] <= 0:
                    continue

                if r["L_2"] <= 0:
                    continue

                P0 = r["total_P0_recovery"]

                if print_all:
                    print(
                        f"{L1:8.3f}"
                        f"{theta1:8.2f}"
                        f"{r['theta_2_deg']:8.2f}"
                        f"{r['L_2']:8.3f}"
                        f"{P0:12.6f}"
                        f"{r['M_exit']:10.4f}"
                    )

                if best is None or P0 > best["total_P0_recovery"]:
                    best = r

            except Exception:
                # Non-physical geometry/shock system
                continue

    return best

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

    """
    Results of the paper with M_inf = 3, L_1 = 0.016, theta_1 = 9 deg, y_cowl = 0.01366, delta_cowl = 4 deg.
    x_cowl = 0.0274 just like in the paper 
   
    Stage                M_in   theta    beta     Mn   M_out   P0 ratio
    ---------------------------------------------------------------
    Ramp 1             3.0000    9.00   26.49 1.3379  2.5541   0.972249
    Ramp 2             2.5541   14.02   35.26 1.4746  1.9599   0.937637
    Reflected shock    1.9599   19.02   52.88 1.5628  1.2204   0.908726
    ---------------------------------------------------------------
    TOTAL P0 recovery                              0.828410
    Exit Mach (after reflected shock)                1.2204

    """
    print("\n[1]  Results of the paper for optimal ramp angles  (M=3)")
    analyse_intake4(M_inf=3, theta_1_deg= 9, L_1= 0.016, y_cowl=0.01366, delta_cowl_deg=4    )

   
    # best = optimise_intake(M_inf=3.0, y_cowl=0.01366, delta_cowl_deg=4, L1_range=(0.01, 1), theta1_range=(1.0, 20.0), print_all=True, M_exit_max=1.4, L1_step=0.005)
    # print("\nBest configuration")
    # print("------------------")
    # print(f"L1      = {best['L_1']:.4f}")
    # print(f"L2      = {best['L_2']:.4f}")
    # print(f"theta1  = {best['theta_1_deg']:.2f} deg")
    # print(f"theta2  = {best['theta_2_deg']:.2f} deg")   
    # print(f"P0 rec  = {best['total_P0_recovery']:.6f}")
    # print(f"M_exit  = {best['M_exit']:.4f}")

    """
    ==========================================================================================
    """
    
    
    
    # print("\n[1]  Design point from paper: M=3, 2 ramps, Oswatitsch criterion")
    # analyse_intake(M_inf=3.0, n=2)
    
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
    
