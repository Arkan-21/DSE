import numpy as np
from scipy.integrate import quad
from scipy.special import comb
from stl import mesh  # numpy-stl

# ── Sizing targets from your preliminary design ──────────────────────────────
S_PLAN_TARGET = 431.693   # m²
V_TARGET      = 1077.096  # m³
TAU_TARGET    = 0.120     # thickness ratio
L_REF         = 40.0      # initial guess for vehicle length, m  (you tune this)

# ── CST basis ─────────────────────────────────────────────────────────────────
def bernstein(i, n, psi):
    """i-th Bernstein basis polynomial of order n evaluated at psi in [0,1]."""
    return comb(n, i, exact=True) * psi**i * (1 - psi)**(n - i)

def cst_curve(psi, A, N1=0.75, N2=0.75):
    """
    CST shape: y(psi) = C(psi) * S(psi)
    A  : array of shape function coefficients (length = polynomial order + 1)
    N1, N2: class function exponents (0.75/0.75 gives blunt nose, blunt base)
    """
    C = psi**N1 * (1 - psi)**N2          # class function
    n = len(A) - 1
    S = sum(A[i] * bernstein(i, n, psi) for i in range(n + 1))
    return C * S

# ── Cross-section definition at a given body station x/L ─────────────────────
def cross_section(xi, A_upper, A_lower, half_width_fn, half_height_fn):
    """
    Returns (y_upper, y_lower) arrays as functions of lateral coordinate.
    xi         : normalised body station (0 = nose, 1 = base)
    half_width : local half-span (from planform)
    half_height: local max half-thickness (from τ and chord)
    """
    psi = np.linspace(0, 1, 200)
    w   = half_width_fn(xi)
    h   = half_height_fn(xi)
    y_u =  h * cst_curve(psi, A_upper)
    y_l = -h * cst_curve(psi, A_lower)
    x_c = w * (2 * psi - 1)   # lateral coordinate, -w to +w
    return x_c, y_u, y_l

# ── Planform: double-delta as initial guess ───────────────────────────────────
def half_span(xi, b_root_frac=0.30, LE_sweep1=70.0, LE_sweep2=55.0, kink=0.40):
    """
    Double-delta planform.
    Returns local half-span at body station xi (= x/L).
    LE_sweep1/2 in degrees; kink = xi at which sweep changes.
    """
    tan1 = np.tan(np.radians(LE_sweep1))
    tan2 = np.tan(np.radians(LE_sweep2))
    if xi <= kink:
        return xi * L_REF / (2 * tan1) if tan1 != 0 else 0.0
    else:
        s_kink = kink * L_REF / (2 * tan1)
        return s_kink + (xi - kink) * L_REF / (2 * tan2)

def half_height(xi, tau=TAU_TARGET):
    """Local half-thickness, assuming τ applied to local chord."""
    local_chord = L_REF * (1 - xi) * 0.5 + L_REF * 0.1  # simple taper; refine later
    return 0.5 * tau * local_chord

# ── Volume and planform area integration ─────────────────────────────────────
def compute_volume_and_splan(n_stations=100):
    xi_arr = np.linspace(0.01, 0.99, n_stations)
    dx     = L_REF / n_stations

    A_upper = np.array([0.20, 0.28, 0.22, 0.18])  # initial guess — 4th-order CST
    A_lower = np.array([0.18, 0.25, 0.20, 0.16])

    V_sum     = 0.0
    S_plan_sum = 0.0

    for xi in xi_arr:
        x_c, y_u, y_l = cross_section(xi, A_upper, A_lower, half_span, half_height)
        # cross-sectional area by trapezoidal integration
        thickness = y_u - y_l
        A_cs = np.trapezoid(thickness, x_c)
        V_sum += A_cs * dx

        # planform area contribution (projection)
        S_plan_sum += 2 * half_span(xi) * dx

    return V_sum, S_plan_sum

V_calc, S_calc = compute_volume_and_splan()
print(f"Calculated volume:        {V_calc:.1f} m³  (target {V_TARGET:.1f})")
print(f"Calculated planform area: {S_calc:.1f} m²  (target {S_PLAN_TARGET:.1f})")
print(f"Volume error:             {abs(V_calc - V_TARGET):.2f} m³")
print(f"S_plan error:             {abs(S_calc - S_PLAN_TARGET):.2f} m²")

# ── Next step: feed errors back into L_REF and CST coefficients via optimizer ─
# scipy.optimize.minimize with constraints V == V_TARGET, S_plan == S_PLAN_TARGET