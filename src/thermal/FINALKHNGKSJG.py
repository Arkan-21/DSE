
MACH             = 5.0
ALTITUDE_KM      = 30.0
WALL_TEMP_K      = 373        # used only when HOT_WALL = False
IS_TURBULENT     = True
AOA_DEG          = 1.0

# ── Hot-wall / radiation-equilibrium ──────────────────────────
HOT_WALL         = False       # True  → solve q_conv = ε·σ·T_w⁴ at every point
                               # False → use prescribed WALL_TEMP_K (cold-wall)
EMISSIVITY       = 0.85       # surface emissivity ε  (0.80–0.90 typical TPS)

# ── Body geometry ──────────────────────────────────────────────
BODY_RADIUS_M    = 2.0
OGIVE_LENGTH_M   = 10.493
BODY_LENGTH_M    = 21.0
NOSE_RADIUS_M    = 0.040

# ── Wing geometry ──────────────────────────────────────────────
WING_LE_RADIUS_M = 0.010        # leading-edge bluntness radius [m]

# Three span segments (root → tip): chord_root, chord_tip, span, sweep°, dihedral°, label
SEGMENTS = [
    (33.14, 13.26,  7.02, 70.55,  1.0, "Segment 1 (inboard,  Λ=70.6°)"),
    (13.26, 10.77,  4.33, 29.84,  1.0, "Segment 2 (mid,      Λ=29.8°)"),
    (10.77,  4.00,  3.29, 67.50, -20.0, "Segment 3 (outboard, Λ=67.5°)"),
]

NUM_POINTS    = 300
SAVE_FIGURES  = True
FIGURE_PREFIX = "heat_flux"

SEG_COLORS  = ["#E63946", "#2A9D8F", "#F4A261"]
SEG_MARKERS = ["o",       "s",       "^"      ]
# ============================================================

import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

SIGMA = 5.67e-8   # Stefan–Boltzmann constant [W m⁻² K⁻⁴]

# ── Atmosphere ────────────────────────────────────────────────────────────────
def us_standard_atmosphere(h_km):
    R, g = 287.05, 9.80665
    layers = [
        (0,  288.15, -6.5,  101325.0),
        (11, 216.65,  0.0,   22632.1),
        (20, 216.65,  1.0,    5474.89),
        (32, 228.65,  2.8,     868.019),
        (47, 270.65,  0.0,     110.906),
        (51, 270.65, -2.8,      66.9389),
        (71, 214.65, -2.0,       3.95642),
        (86, 186.87,  0.0,       0.3734),
    ]
    for i in range(len(layers) - 1):
        hb, Tb, lr, Pb = layers[i]
        if h_km <= layers[i + 1][0]:
            dh = (h_km - hb) * 1000
            if abs(lr) < 1e-9:
                T = Tb; P = Pb * math.exp(-g * dh / (R * Tb))
            else:
                lm = lr / 1000; T = Tb + lm * dh
                P = Pb * (T / Tb) ** (-g / (lm * R))
            return T, P, P / (R * T)
    hb, Tb, _, Pb = layers[-1]
    T = Tb; P = Pb * math.exp(-g * (h_km - hb) * 1000 / (R * Tb))
    return T, P, P / (R * T)

# ── Gas properties ────────────────────────────────────────────────────────────
GAMMA   = 1.4
R_AIR   = 287.05
PR      = 0.71
CP      = GAMMA * R_AIR / (GAMMA - 1)
MU_REF  = 1.716e-5
T_REF   = 273.15
S_SUTH  = 110.4

def viscosity(T):
    return MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)

def kappa(T):
    return viscosity(T) * CP / PR

# ── Shock / expansion ────────────────────────────────────────────────────────
def normal_shock(M1):
    g = GAMMA; m = M1 ** 2
    M2sq = (1 + (g - 1) / 2 * m) / (g * m - (g - 1) / 2)
    T2T1 = (1 + 2 * g / (g + 1) * (m - 1)) * (2 + (g - 1) * m) / ((g + 1) * m)
    r2   = (g + 1) * m / (2 + (g - 1) * m)
    return math.sqrt(M2sq), T2T1, r2

def oblique_shock_angle(M1, wedge_deg):
    theta = math.radians(wedge_deg); g = GAMMA
    mu = math.asin(1 / M1)
    def tob(b):
        sb2 = math.sin(b) ** 2; cot = math.cos(b) / math.sin(b)
        v = 2 * cot * (M1 ** 2 * sb2 - 1) / (M1 ** 2 * (g + math.cos(2 * b)) + 2)
        return math.atan(max(-1.0, min(1.0, v)))
    betas  = [mu + 1e-4 + (math.pi / 2 - 1e-4 - mu - 1e-4) * i / 60 for i in range(61)]
    thetas = [tob(b) for b in betas]
    idx = thetas.index(max(thetas))
    if theta > thetas[idx]: return None
    b_lo, b_hi = mu + 1e-4, betas[idx]
    for _ in range(200):
        bm = 0.5 * (b_lo + b_hi); fm = tob(bm) - theta
        if abs(fm) < 1e-9 or b_hi - b_lo < 1e-11: break
        if (tob(b_lo) - theta) * fm < 0: b_hi = bm
        else: b_lo = bm
    return math.degrees(bm)

def compression_edge(M1, r1, T1, P1, wedge):
    bg = oblique_shock_angle(M1, wedge)
    if bg is None:
        warnings.warn(f"Detached shock M={M1:.2f} δ={wedge:.1f}°")
        return dict(T_e=T1, u_e=M1 * math.sqrt(GAMMA * R_AIR * T1), P_e=P1)
    b = math.radians(bg); Mn1 = M1 * math.sin(b)
    Mn2, T2T1, r2r = normal_shock(Mn1)
    T2 = T1 * T2T1; r2 = r1 * r2r; th = math.radians(wedge)
    M2 = Mn2 / math.sin(b - th); u2 = M2 * math.sqrt(GAMMA * R_AIR * T2)
    return dict(T_e=T2, u_e=u2, P_e=r2 * R_AIR * T2)

def pm_nu(M):
    g = GAMMA; gp = (g + 1) / (g - 1)
    return math.sqrt(gp) * math.atan(math.sqrt((M ** 2 - 1) / gp)) - math.atan(math.sqrt(M ** 2 - 1))

def pm_M(nu):
    lo, hi = 1.001, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if pm_nu(mid) < nu: lo = mid
        else: hi = mid
    return mid

def expansion_edge(M1, r1, T1, P1, turn):
    if turn <= 0: return dict(T_e=T1, u_e=M1 * math.sqrt(GAMMA * R_AIR * T1), P_e=P1)
    nu1 = pm_nu(M1); g = GAMMA
    nu_max = math.pi / 2 * (math.sqrt((g + 1) / (g - 1)) - 1)
    nu2 = min(nu1 + math.radians(turn), nu_max)
    M2 = pm_M(nu2); fac = (1 + (g - 1) / 2 * M1 ** 2) / (1 + (g - 1) / 2 * M2 ** 2)
    T2 = T1 * fac; r2 = r1 * fac ** (1 / (g - 1))
    return dict(T_e=T2, u_e=M2 * math.sqrt(GAMMA * R_AIR * T2), P_e=r2 * R_AIR * T2)

def tangent_wedge_edge(M, r, T, P, V, delta):
    if abs(delta) < 0.01: return dict(T_e=T, u_e=V, P_e=P)
    return compression_edge(M, r, T, P, delta) if delta > 0 else expansion_edge(M, r, T, P, abs(delta))

# ── Reference enthalpy & flat-plate heat flux ─────────────────────────────────
def ref_state(T_e, u_e, P_e, T_w, turb):
    r    = PR ** (1 / 3) if turb else math.sqrt(PR)
    M_e  = u_e / math.sqrt(GAMMA * R_AIR * T_e)
    T_aw = T_e * (1 + r * (GAMMA - 1) / 2 * M_e ** 2)
    # Eckert reference enthalpy
    h_star = CP * T_w + 0.5 * CP * (T_e - T_w) + 0.22 * CP * (T_aw - T_w)
    T_s = h_star / CP; rs = P_e / (R_AIR * T_s)
    return T_s, rs, viscosity(T_s)

def adiabatic_Taw(T_e, u_e, turb):
    """Flat-plate adiabatic wall temperature."""
    r   = PR ** (1 / 3) if turb else math.sqrt(PR)
    M_e = u_e / math.sqrt(GAMMA * R_AIR * T_e)
    return T_e * (1 + r * (GAMMA - 1) / 2 * M_e ** 2)

def flat_plate_qw(s, edge, T_w, turb):
    if s <= 0: return 0.0
    T_e, u_e, P_e = edge['T_e'], edge['u_e'], edge['P_e']
    Ts, rs, mus = ref_state(T_e, u_e, P_e, T_w, turb)
    ks = kappa(Ts); Re = rs * u_e * s / mus
    if Re < 1: return 0.0
    Nu = (0.0296 * Re ** 0.8 if turb else 0.332 * Re ** 0.5) * PR ** (1 / 3)
    hc = Nu * ks / s
    Taw = adiabatic_Taw(T_e, u_e, turb)
    return max(hc * (Taw - T_w), 0.0)

# ── Stagnation correlations ───────────────────────────────────────────────────
def stagnation_qw(M, r_inf, T_inf, P_inf, V_inf, nose_r, T_w):
    _, T2T1, r2r = normal_shock(M)
    Te = T_inf * T2T1; re = r_inf * r2r
    P2P1 = 1 + 2 * GAMMA / (GAMMA + 1) * (M ** 2 - 1); Pe = P_inf * P2P1
    du_dx = (1 / nose_r) * math.sqrt(2 * 0.5 * r_inf * V_inf ** 2 / re)
    mue = viscosity(Te); h0 = CP * T_inf + 0.5 * V_inf ** 2; hw = CP * T_w
    qs = (0.57 / PR ** 0.6) * math.sqrt(re * mue * du_dx) * max(h0 - hw, 0)
    return qs, dict(T_e=Te, P_e=Pe, du_dx=du_dx, h0=h0, h_w=hw)

def le_stag_qw(M, r_inf, T_inf, P_inf, V_inf, le_r, T_w, sweep_deg):
    if le_r <= 0: le_r = 1e-3
    cL = math.cos(math.radians(sweep_deg))
    Vn = V_inf * cL; Mn = max(M * cL, 1.001)
    _, T2T1, r2r = normal_shock(Mn)
    Te = T_inf * T2T1; re = r_inf * r2r
    du_dx = (1 / le_r) * math.sqrt(2 * 0.5 * r_inf * Vn ** 2 / re)
    mue = viscosity(Te); h0 = CP * T_inf + 0.5 * V_inf ** 2
    return max((0.57 / PR ** 0.6) * math.sqrt(re * mue * du_dx) * (h0 - CP * T_w), 0.0)

# ── Radiation-equilibrium solver ─────────────────────────────────────────────
def find_equil_temp(q_func, T_aw, emissivity, T_lo=200.0):
    """
    Bisection solve for the radiation-equilibrium wall temperature T_w where:
        q_conv(T_w) = emissivity * SIGMA * T_w^4

    Parameters
    ----------
    q_func     : callable(T_w) → convective heat flux [W/m²] at wall temp T_w
    T_aw       : adiabatic wall temperature [K]  (upper physical bound for T_w)
    emissivity : surface emissivity  ε
    T_lo       : lower bound for bisection [K]

    Returns
    -------
    T_w_eq : radiation-equilibrium wall temperature [K]
    """
    T_hi = min(T_aw * 0.9999, 6000.0)

    def residual(Tw):
        return q_func(Tw) - emissivity * SIGMA * Tw ** 4

    r_lo = residual(T_lo)
    r_hi = residual(T_hi)

    # If residual is negative even at T_lo the surface barely heats at all
    if r_lo <= 0:
        return T_lo
    # If residual is positive even at T_hi the surface can't radiate fast enough
    if r_hi >= 0:
        return T_hi

    # Standard bisection
    for _ in range(120):
        if T_hi - T_lo < 0.05:
            break
        Tm = 0.5 * (T_lo + T_hi)
        if residual(Tm) > 0:
            T_lo = Tm
        else:
            T_hi = Tm

    return 0.5 * (T_lo + T_hi)

# ── Tangent ogive ─────────────────────────────────────────────────────────────
class TangentOgive:
    def __init__(self, R, L):
        self.R = R; self.L = L
        self.rho_c = (R ** 2 + L ** 2) / (2 * R)

    def radius(self, x):
        x = max(0.0, min(x, self.L))
        return math.sqrt(self.rho_c ** 2 - (self.L - x) ** 2) + self.R - self.rho_c

    def slope_deg(self, x):
        x = max(1e-6, min(x, self.L - 1e-6))
        drdx = (self.L - x) / math.sqrt(self.rho_c ** 2 - (self.L - x) ** 2)
        return math.degrees(math.atan(drdx))

    def arc_length(self, xs):
        s = np.zeros_like(xs)
        for i in range(1, len(xs)):
            x0, x1 = xs[i - 1], xs[i]
            xm = max(1e-6, min(0.5 * (x0 + x1), self.L - 1e-6))
            drdx = (self.L - xm) / math.sqrt(self.rho_c ** 2 - (self.L - xm) ** 2)
            s[i] = s[i - 1] + (x1 - x0) * math.sqrt(1 + drdx ** 2)
        return s

# ── Multi-segment swept wing ──────────────────────────────────────────────────
class SweptWingSegment:
    def __init__(self, chord_root, chord_tip, span, sweep_deg,
                 dihedral_deg, label, le_radius, aoa_deg,
                 M, r_inf, T_inf, P_inf, V_inf, T_w, turb,
                 hot_wall=False, emissivity=0.85):
        self.cr = chord_root; self.ct = chord_tip; self.span = span
        self.sweep = sweep_deg; self.dih = dihedral_deg; self.label = label
        self.le_r = le_radius; self.aoa = aoa_deg
        self.M = M; self.r = r_inf; self.T = T_inf; self.P = P_inf; self.V = V_inf
        self.T_w = T_w; self.turb = turb
        self.hot_wall = hot_wall; self.emissivity = emissivity
        self.c_mid = 0.5 * (chord_root + chord_tip)

    def compute(self, n=150):
        c = self.c_mid
        x = np.linspace(0.005 * c, c, n)
        thickness_ratio = 0.02
        wedge_half = math.degrees(math.atan(thickness_ratio / 2))
        delta_lo = wedge_half + self.aoa
        delta_up = wedge_half - self.aoa

        V_eff = self.M * math.sqrt(GAMMA * R_AIR * self.T)
        edge_lo = tangent_wedge_edge(self.M, self.r, self.T, self.P, V_eff, delta_lo)
        edge_up = tangent_wedge_edge(self.M, self.r, self.T, self.P, V_eff, delta_up)

        q_lo      = np.zeros(n); q_up      = np.zeros(n)
        Tw_lo_arr = np.zeros(n); Tw_up_arr = np.zeros(n)

        if self.hot_wall:
            # ─ Adiabatic wall temperatures for each surface ─────────────────
            Taw_lo = adiabatic_Taw(edge_lo['T_e'], edge_lo['u_e'], self.turb)
            Taw_up = adiabatic_Taw(edge_up['T_e'], edge_up['u_e'], self.turb)

            for i, xi in enumerate(x):
                # Lower surface (windward)
                def _qlo(Tw, s=xi, e=edge_lo): return flat_plate_qw(s, e, Tw, self.turb)
                Tw_lo = find_equil_temp(_qlo, Taw_lo, self.emissivity)
                Tw_lo_arr[i] = Tw_lo
                q_lo[i] = flat_plate_qw(xi, edge_lo, Tw_lo, self.turb)

                # Upper surface (leeward)
                def _qup(Tw, s=xi, e=edge_up): return flat_plate_qw(s, e, Tw, self.turb)
                Tw_up = find_equil_temp(_qup, Taw_up, self.emissivity)
                Tw_up_arr[i] = Tw_up
                q_up[i] = flat_plate_qw(xi, edge_up, Tw_up, self.turb)

            # ─ Leading-edge equilibrium ──────────────────────────────────────
            h0 = CP * self.T + 0.5 * self.V ** 2
            T0 = h0 / CP  # total temperature (LE stag upper bound)
            def _qle(Tw): return le_stag_qw(self.M, self.r, self.T, self.P, self.V,
                                             self.le_r, Tw, self.sweep)
            T_w_le = find_equil_temp(_qle, T0, self.emissivity)
            q_le   = le_stag_qw(self.M, self.r, self.T, self.P, self.V,
                                 self.le_r, T_w_le, self.sweep)
        else:
            T_w = self.T_w
            for i, xi in enumerate(x):
                q_lo[i] = flat_plate_qw(xi, edge_lo, T_w, self.turb)
                q_up[i] = flat_plate_qw(xi, edge_up, T_w, self.turb)
                Tw_lo_arr[i] = T_w
                Tw_up_arr[i] = T_w
            q_le   = le_stag_qw(self.M, self.r, self.T, self.P, self.V,
                                 self.le_r, T_w, self.sweep)
            T_w_le = T_w

        return dict(x=x, q_lower=q_lo, q_upper=q_up, q_le=q_le,
                    Tw_lower=Tw_lo_arr, Tw_upper=Tw_up_arr, T_w_le=T_w_le,
                    sweep=self.sweep, label=self.label,
                    delta_lo=delta_lo, delta_up=delta_up, M_eff=self.M,
                    chord_mid=c)

# ── Body analysis ─────────────────────────────────────────────────────────────
class OgiveCylinderAnalysis:
    def __init__(self, M, alt, T_w, turb, aoa, R, L_og, L_body, r_nose,
                 hot_wall=False, emissivity=0.85):
        self.M = M; self.alt = alt; self.T_w = T_w; self.turb = turb
        self.aoa = aoa; self.R = R; self.L_og = L_og
        self.L_body = L_body; self.r_nose = r_nose
        self.hot_wall = hot_wall; self.emissivity = emissivity
        self.T_inf, self.P_inf, self.rho_inf = us_standard_atmosphere(alt)
        self.a_inf = math.sqrt(GAMMA * R_AIR * self.T_inf)
        self.V_inf = M * self.a_inf
        self.ogive = TangentOgive(R, L_og)

    def run_body(self, n=300):
        x_og  = np.linspace(0.001, self.L_og,            n // 2)
        x_cy  = np.linspace(self.L_og, self.L_body,      n // 2 + 1)[1:]
        x_all = np.concatenate([x_og, x_cy])

        # Arc-length interpolation along ogive
        x_fine = np.linspace(0, self.L_og, 2000)
        s_fine = self.ogive.arc_length(x_fine)
        s_junc = s_fine[-1]

        def arc(xv):
            if xv <= self.L_og: return float(np.interp(xv, x_fine, s_fine))
            return s_junc + (xv - self.L_og)

        # ── Stagnation point ─────────────────────────────────────────────────
        if self.hot_wall:
            T0 = self.T_inf + 0.5 * self.V_inf ** 2 / CP  # total temp
            def _qstag(Tw): return stagnation_qw(self.M, self.rho_inf, self.T_inf,
                                                  self.P_inf, self.V_inf,
                                                  self.r_nose, Tw)[0]
            T_w_stag = find_equil_temp(_qstag, T0, self.emissivity)
        else:
            T_w_stag = self.T_w

        q_s, diag = stagnation_qw(self.M, self.rho_inf, self.T_inf,
                                   self.P_inf, self.V_inf, self.r_nose, T_w_stag)

        # ── Surface distribution ─────────────────────────────────────────────
        qw       = np.zeros(len(x_all))
        ql       = np.zeros(len(x_all))
        sl       = np.zeros(len(x_all))
        Tw_wind  = np.zeros(len(x_all))
        Tw_lee   = np.zeros(len(x_all))

        for i, x in enumerate(x_all):
            s   = arc(x)
            phi = self.ogive.slope_deg(x) if x <= self.L_og else 0.0
            sl[i] = phi

            ew = tangent_wedge_edge(self.M, self.rho_inf, self.T_inf,
                                    self.P_inf, self.V_inf, phi + self.aoa)
            el = tangent_wedge_edge(self.M, self.rho_inf, self.T_inf,
                                    self.P_inf, self.V_inf, phi - self.aoa)

            if self.hot_wall:
                Taw_w = adiabatic_Taw(ew['T_e'], ew['u_e'], self.turb)
                Taw_l = adiabatic_Taw(el['T_e'], el['u_e'], self.turb)

                def _qw(Tw, s=s, e=ew): return flat_plate_qw(s, e, Tw, self.turb)
                def _ql(Tw, s=s, e=el): return flat_plate_qw(s, e, Tw, self.turb)

                Tw_w = find_equil_temp(_qw, Taw_w, self.emissivity)
                Tw_l = find_equil_temp(_ql, Taw_l, self.emissivity)

                Tw_wind[i] = Tw_w
                Tw_lee[i]  = Tw_l
                qw[i] = flat_plate_qw(s, ew, Tw_w, self.turb)
                ql[i] = flat_plate_qw(s, el, Tw_l, self.turb)
            else:
                T_w = self.T_w
                Tw_wind[i] = T_w
                Tw_lee[i]  = T_w
                qw[i] = min(flat_plate_qw(s, ew, T_w, self.turb), q_s)
                ql[i] = min(flat_plate_qw(s, el, T_w, self.turb), q_s)

        s_all = np.array([arc(x) for x in x_all])
        return dict(x=x_all, s=s_all, slope=sl,
                    q_wind=qw, q_lee=ql,
                    q_stag=q_s, T_w_stag=T_w_stag,
                    Tw_wind=Tw_wind, Tw_lee=Tw_lee,
                    diag=diag, x_junc=self.L_og, s_junc=s_junc)

    def _title(self):
        mode = (f"Hot-wall ε={self.emissivity:.2f}" if self.hot_wall
                else f"Cold-wall Tw={self.T_w:.0f} K")
        return (f"M={self.M:.1f}, h={self.alt:.0f} km, AoA={self.aoa:.1f}°, "
                f"{'Turb.' if self.turb else 'Lam.'} BL, {mode}")

# ── Save helper ───────────────────────────────────────────────────────────────
def _save_or_show(fig, basepath):
    path = basepath + ".pdf"
    fig.savefig(path, format='pdf', bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)

# ── Figure 1: Body heat flux ──────────────────────────────────────────────────
def plot_body(body_data, vehicle, prefix):
    x  = body_data['x']; qw = body_data['q_wind'] / 1e3; ql = body_data['q_lee'] / 1e3
    qs = body_data['q_stag'] / 1e3; xj = body_data['x_junc']
    L  = vehicle.L_og; R  = vehicle.R; ogive = vehicle.ogive

    fig, (ax_g, ax_hf) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Ogive-Cylinder Aerothermal Heating  –  {vehicle._title()}", fontsize=12)

    xog = np.linspace(0, L, 400); rog = np.array([ogive.radius(xi) for xi in xog])
    xcy = np.array([L, vehicle.L_body]); rcy = np.array([R, R])
    ax_g.fill_between(np.r_[xog, xcy],  np.r_[rog, rcy], alpha=0.2, color='steelblue')
    ax_g.fill_between(np.r_[xog, xcy], -np.r_[rog, rcy], alpha=0.2, color='steelblue')
    ax_g.plot(np.r_[xog, xcy],  np.r_[rog, rcy], 'b-', lw=2)
    ax_g.plot(np.r_[xog, xcy], -np.r_[rog, rcy], 'b-', lw=2)
    ax_g.axvline(xj, color='gray', ls='--', lw=1, label=f'Junction x={xj:.2f} m')
    ax_g.set_xlabel('Axial x [m]'); ax_g.set_ylabel('Radial r [m]')
    ax_g.set_title('Body profile (tangent ogive + cylinder)'); ax_g.set_aspect('equal')
    ax_g.legend()

    ax_hf.axvline(xj, color='gray', ls='--', lw=1, label=f'Junction x={xj:.2f} m')
    ax_hf.axhline(qs, color='purple', ls=':', lw=1.5,
                  label=f'Stag. {qs:.1f} kW/m² (Tw={body_data["T_w_stag"]:.0f} K)')
    ax_hf.plot(x, qw, 'r-',  lw=2, label=f'Windward (φ+{vehicle.aoa:.1f}°)')
    ax_hf.plot(x, ql, 'b--', lw=2, label=f'Leeward  (φ−{vehicle.aoa:.1f}°)')
    ax_hf.set_xlabel('Axial x [m]'); ax_hf.set_ylabel('Heat flux [kW/m²]')
    ax_hf.set_title('Body surface heat flux distribution'); ax_hf.legend()

    plt.tight_layout()
    _save_or_show(fig, f"{prefix}_body")

# ── Figure 2: Body wall temperature distribution (new) ───────────────────────
def plot_body_Tw(body_data, vehicle, prefix):
    x       = body_data['x']
    Tw_w    = body_data['Tw_wind']
    Tw_l    = body_data['Tw_lee']
    T_w_stag = body_data['T_w_stag']
    xj      = body_data['x_junc']

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axvline(xj, color='gray', ls='--', lw=1, label=f'Junction x={xj:.2f} m')
    ax.axhline(T_w_stag, color='purple', ls=':', lw=1.5,
               label=f'Nose stag. T_w = {T_w_stag:.0f} K')
    ax.plot(x, Tw_w, 'r-',  lw=2, label='Windward equilibrium T_w')
    ax.plot(x, Tw_l, 'b--', lw=2, label='Leeward equilibrium T_w')
    ax.set_xlabel('Axial x [m]')
    ax.set_ylabel('Equilibrium wall temperature  T_w  [K]')
    ax.set_title(f'Radiation-Equilibrium Wall Temperature — Body\n{vehicle._title()}')
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, f"{prefix}_body_Tw")

# ── Figure 3: Wing all segments ───────────────────────────────────────────────
def plot_wing_all_segments(seg_results, qs_body, prefix, title_str):
    fig, (ax_lo, ax_up) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(f"Multi-Segment Swept Wing — Convective Heat Flux\n{title_str}", fontsize=12)

    for i, (sr, col, mk) in enumerate(zip(seg_results, SEG_COLORS, SEG_MARKERS)):
        xi  = sr['x'] / sr['chord_mid']
        qlo = sr['q_lower'] / 1e3; qup = sr['q_upper'] / 1e3
        qle = sr['q_le'] / 1e3;    lbl = sr['label']

        ax_lo.plot(xi, qlo, color=col, lw=2, label=lbl)
        ax_lo.plot(xi[::15], qlo[::15], color=col, marker=mk, ls='none', ms=6)
        ax_lo.axhline(qle, color=col, ls=':', lw=1.5)
        ax_lo.annotate(f'LE={qle:.0f}', xy=(0.01, qle), fontsize=7.5, color=col,
                       xytext=(0.06 + i * 0.12, qle + 0.4),
                       arrowprops=dict(arrowstyle='->', color=col, lw=0.8))

        ax_up.plot(xi, qup, color=col, lw=2, label=lbl)
        ax_up.plot(xi[::15], qup[::15], color=col, marker=mk, ls='none', ms=6)
        ax_up.axhline(qle, color=col, ls=':', lw=1.5)

    for ax, side, defl_info in [
            (ax_lo, 'Lower (windward)', f"δ = wedge + AoA={AOA_DEG:.1f}°"),
            (ax_up, 'Upper (leeward)',  f"δ = wedge − AoA={AOA_DEG:.1f}°")]:
        ax.set_xlabel('Normalised chordwise position  x/c  [–]', fontsize=11)
        ax.set_ylabel('Convective heat flux  [kW/m²]', fontsize=11)
        ax.set_title(f'{side} surface\n({defl_info})', fontsize=10)
        ax.legend(loc='upper right'); ax.set_xlim(0, 1)

    plt.tight_layout()
    _save_or_show(fig, f"{prefix}_wing_all_segments")

# ── Figure 4: Wing wall temperature distribution (new) ───────────────────────
def plot_wing_Tw(seg_results, prefix, title_str):
    fig, (ax_lo, ax_up) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(f"Wing Radiation-Equilibrium Wall Temperature\n{title_str}", fontsize=12)

    for sr, col, mk in zip(seg_results, SEG_COLORS, SEG_MARKERS):
        xi     = sr['x'] / sr['chord_mid']
        Tw_lo  = sr['Tw_lower']
        Tw_up  = sr['Tw_upper']
        T_w_le = sr['T_w_le']
        lbl    = sr['label']

        ax_lo.plot(xi, Tw_lo, color=col, lw=2, label=lbl)
        ax_lo.axhline(T_w_le, color=col, ls=':', lw=1.5, label=f'{lbl} LE={T_w_le:.0f} K')
        ax_up.plot(xi, Tw_up, color=col, lw=2, label=lbl)
        ax_up.axhline(T_w_le, color=col, ls=':', lw=1.5)

    for ax, side in [(ax_lo, "Lower (windward)"), (ax_up, "Upper (leeward)")]:
        ax.set_xlabel('Normalised chordwise position  x/c  [–]', fontsize=11)
        ax.set_ylabel('Equilibrium T_w  [K]', fontsize=11)
        ax.set_title(f'{side} surface equilibrium wall temperature', fontsize=10)
        ax.legend(fontsize=8, loc='best'); ax.set_xlim(0, 1)

    plt.tight_layout()
    _save_or_show(fig, f"{prefix}_wing_Tw")

# ── Figure 5: Spanwise LE stagnation ─────────────────────────────────────────
def plot_le_spanwise(seg_results, prefix, title_str):
    labels = [sr['label'] for sr in seg_results]
    qle    = [sr['q_le'] / 1e3 for sr in seg_results]
    sweeps = [sr['sweep'] for sr in seg_results]
    T_w_le = [sr['T_w_le'] for sr in seg_results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Leading-Edge Stagnation — Spanwise Variation\n{title_str}", fontsize=12)

    ax = axes[0]
    bars = ax.bar(range(len(labels)), qle, color=SEG_COLORS, edgecolor='k', lw=1.2, width=0.5)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('LE stagnation heat flux  [kW/m²]', fontsize=11)
    ax.set_title('Leading-edge stagnation per segment', fontsize=11)
    for b, v, tw in zip(bars, qle, T_w_le):
        ax.text(b.get_x() + b.get_width() / 2, v + max(qle) * 0.01,
                f'{v:.1f}\n({tw:.0f} K)', ha='center', va='bottom',
                fontweight='bold', fontsize=9)

    ax2 = axes[1]
    for i, (sw, q, col, mk) in enumerate(zip(sweeps, qle, SEG_COLORS, SEG_MARKERS)):
        ax2.scatter(sw, q, color=col, marker=mk, s=120, zorder=5, label=labels[i])
    ax2.set_xlabel('Leading-edge sweep angle  Λ  [°]', fontsize=11)
    ax2.set_ylabel('LE stagnation heat flux  [kW/m²]', fontsize=11)
    ax2.set_title('LE heat flux vs. sweep angle\n(higher sweep → lower q via cos(Λ))', fontsize=10)
    sw_arr  = np.linspace(0, 85, 200)
    q_ref   = qle[0] / math.cos(math.radians(sweeps[0])) ** 0.5
    q_trend = [q_ref * math.cos(math.radians(s)) ** 0.5 for s in sw_arr]
    ax2.plot(sw_arr, q_trend, 'k--', lw=1.2, alpha=0.5,
             label=r'$q \propto \cos(\Lambda)^{0.5}$ trend')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, f"{prefix}_le_spanwise")

# ── Figure 6: Normalised comparison ──────────────────────────────────────────
def plot_normalised_comparison(seg_results, body_data, prefix, title_str):
    q_stag = body_data['q_stag'] / 1e3
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Normalised Convective Heat Flux  (q / q_stag)\n{title_str}", fontsize=12)

    xb  = body_data['x']; qbw = body_data['q_wind'] / 1e3 / q_stag
    ax.plot(xb / xb[-1], qbw, 'k-', lw=2.5, label='Body windward (normalised x/L)')
    ax.axhline(1.0, color='purple', ls=':', lw=1.5, label=f'Nose stag. ({q_stag:.1f} kW/m²)')

    for i, (sr, col, mk) in enumerate(zip(seg_results, SEG_COLORS, SEG_MARKERS)):
        xi = sr['x'] / sr['chord_mid']
        ax.plot(xi, sr['q_lower'] / 1e3 / q_stag, color=col, lw=1.8,
                label=f"{sr['label']} lower")
        ax.plot(xi, sr['q_upper'] / 1e3 / q_stag, color=col, lw=1.2, ls='--')

    ax.set_xlabel('Normalised axial/chordwise position  [–]', fontsize=11)
    ax.set_ylabel('Normalised heat flux  q / q_stag  [–]', fontsize=11)
    ax.set_yscale('log'); ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    _save_or_show(fig, f"{prefix}_normalised")

# ── Figure 7: Peak heat flux bar chart ───────────────────────────────────────
def plot_summary_bar(body_data, seg_results, prefix, title_str):
    labels = ['Nose\n(stag.)', 'Body\n(max)']
    vals   = [body_data['q_stag'] / 1e3,
              max(body_data['q_wind'].max(), body_data['q_lee'].max()) / 1e3]
    cols   = ['#c0392b', '#e67e22']
    tw_labels = [f"{body_data['T_w_stag']:.0f} K", ""]

    for i, (sr, col) in enumerate(zip(seg_results, SEG_COLORS)):
        labels.append(f"Seg {i+1}\nLE stag.")
        vals.append(sr['q_le'] / 1e3)
        cols.append(col)
        tw_labels.append(f"{sr['T_w_le']:.0f} K")

        labels.append(f"Seg {i+1}\nsurface max")
        vals.append(max(sr['q_lower'].max(), sr['q_upper'].max()) / 1e3)
        cols.append(col)
        tw_labels.append("")

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(labels, vals, color=cols, edgecolor='k', lw=1.1, width=0.6)
    peak = max(vals)
    ax.axhline(peak, color='k', ls=':', lw=1.5, label=f'Overall peak: {peak:.1f} kW/m²')
    ax.set_ylabel('Peak conv. heat flux  [kW/m²]', fontsize=11)
    ax.set_title(f'Peak Heat Flux per Component — TPS Sizing Reference\n{title_str}', fontsize=11)
    for b, v, tw in zip(bars, vals, tw_labels):
        label_str = f'{v:.1f}' + (f'\n({tw})' if tw else '')
        ax.text(b.get_x() + b.get_width() / 2, v + peak * 0.01, label_str,
                ha='center', va='bottom', fontweight='bold', fontsize=8)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save_or_show(fig, f"{prefix}_summary_bar")

    print("\n" + "=" * 64)
    print("  PEAK HEAT FLUX SUMMARY")
    print("=" * 64)
    for l, v in zip(labels, vals):
        print(f"  {l.replace(chr(10),' '):30s}: {v:8.2f} kW/m²")
    print("=" * 64)
    print("  ⚠  Apply ≥ 1.5× safety factor for TPS sizing.")
    print("=" * 64)

# ── Cockpit region constants ──────────────────────────────────────────────────
COCKPIT_X_RANGE = (8.0, 10.5)
WING_ROOT_X     = 12.0

# ── Figure 8: Cockpit region detail ──────────────────────────────────────────
def plot_cockpit_region(body_data, prefix, title_str):
    x  = body_data['x']; qw = body_data['q_wind'] / 1e3; ql = body_data['q_lee'] / 1e3
    cp_lo, cp_hi = COCKPIT_X_RANGE
    mask = (x >= cp_lo - 1.0) & (x <= cp_hi + 1.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x[mask], qw[mask], color='#c0392b', lw=2.2, label='Windward')
    ax.plot(x[mask], ql[mask], color='#2980b9', lw=2.2, ls='--', label='Leeward')
    ax.axvspan(cp_lo, cp_hi, color='cyan', alpha=0.15,
               label=f'Cockpit ({cp_lo:.1f}–{cp_hi:.1f} m)')
    for xv in [cp_lo, cp_hi]:
        qwv = float(np.interp(xv, x, qw)); qlv = float(np.interp(xv, x, ql))
        ax.annotate(f'{qwv:.1f}', xy=(xv, qwv), xytext=(xv, qwv + 0.5),
                    fontsize=8, color='#c0392b', ha='center', fontweight='bold')
        ax.annotate(f'{qlv:.1f}', xy=(xv, qlv), xytext=(xv, qlv - 0.8),
                    fontsize=8, color='#2980b9', ha='center', fontweight='bold')
    ax.set_xlabel('Axial position  x  [m]', fontsize=11)
    ax.set_ylabel('Convective heat flux  [kW/m²]', fontsize=11)
    ax.set_title(f'Cockpit Region Heat Flux Detail\n{title_str}', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_or_show(fig, f"{prefix}_cockpit_detail")

# ── Figure 9: All heat flux overview ─────────────────────────────────────────
def plot_all_heat_flux_overview(body_data, seg_results, vehicle, prefix, title_str):
    x      = body_data['x']; qw = body_data['q_wind'] / 1e3; ql = body_data['q_lee'] / 1e3
    q_stag = body_data['q_stag'] / 1e3; xj = body_data['x_junc']

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x, qw, color='#c0392b', lw=2.2, label='Body windward')
    ax.plot(x, ql, color='#2980b9', lw=2.2, ls='--', label='Body leeward')
    ax.axhline(q_stag, color='purple', ls=':', lw=1.5,
               label=f'Nose stagnation: {q_stag:.1f} kW/m²')
    ax.scatter([0], [q_stag], color='purple', s=80, zorder=5, marker='*')
    ax.axvline(xj, color='gray', ls='--', lw=1, label=f'Ogive–cyl. junction (x={xj:.2f} m)')
    cp_lo, cp_hi = COCKPIT_X_RANGE
    ax.axvspan(cp_lo, cp_hi, color='cyan', alpha=0.12,
               label=f'Cockpit ({cp_lo:.1f}–{cp_hi:.1f} m)')

    for i, (sr, col, mk) in enumerate(zip(seg_results, SEG_COLORS, SEG_MARKERS)):
        qle     = sr['q_le'] / 1e3
        qlo_max = sr['q_lower'].max() / 1e3
        qup_max = sr['q_upper'].max() / 1e3
        c_mid   = sr['chord_mid']
        x0 = WING_ROOT_X; x1 = WING_ROOT_X + c_mid
        ax.scatter([x0], [qle], color=col, marker=mk, s=110, zorder=5,
                   edgecolors='k', linewidths=0.8,
                   label=f"{sr['label']}: LE={qle:.1f}, lower max={qlo_max:.1f}, upper max={qup_max:.1f}")
        ax.plot([x0, x1], [qlo_max, qlo_max], color=col, lw=1.6, ls='-')
        ax.plot([x0, x1], [qup_max, qup_max], color=col, lw=1.6, ls=':')

    ax.set_xlabel('Axial position  x  [m]  (wing curves shown relative to wing root)', fontsize=11)
    ax.set_ylabel('Convective heat flux  [kW/m²]', fontsize=11)
    ax.set_title(f'Overview: All Stagnation & Surface Heat Fluxes\n{title_str}', fontsize=12)
    ax.legend(fontsize=8, loc='upper right', ncol=1)
    ax.set_xlim(left=-0.5)
    plt.tight_layout()
    _save_or_show(fig, f"{prefix}_overview_all_qflux")

# ── Figure 10: Wing chordwise curves ─────────────────────────────────────────
def plot_wing_chordwise_curves(seg_results, prefix, title_str):
    fig, (ax_lo, ax_up) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(f"Wing Chordwise Heat Flux (Surface Only)\n{title_str}", fontsize=12)

    for i, (sr, col, mk) in enumerate(zip(seg_results, SEG_COLORS, SEG_MARKERS)):
        xi  = sr['x'] / sr['chord_mid']
        qlo = sr['q_lower'] / 1e3; qup = sr['q_upper'] / 1e3
        ax_lo.plot(xi, qlo, color=col, lw=2, label=sr['label'])
        ax_lo.plot(xi[::15], qlo[::15], color=col, marker=mk, ls='none', ms=6)
        ax_up.plot(xi, qup, color=col, lw=2, label=sr['label'])
        ax_up.plot(xi[::15], qup[::15], color=col, marker=mk, ls='none', ms=6)

    for ax, side in [(ax_lo, "Lower Surface (windward)"),
                     (ax_up, "Upper Surface (leeward)")]:
        ax.set_xlabel("Normalised chord position  x/c  [–]", fontsize=11)
        ax.set_ylabel("Convective heat flux  [kW/m²]", fontsize=11)
        ax.set_title(side, fontsize=11)
        ax.legend(fontsize=9, loc='best'); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)

    plt.tight_layout()
    _save_or_show(fig, f"{prefix}_wing_chordwise_curves")

# ── Figure 11: Thermal landscape ─────────────────────────────────────────────
def plot_thermal_landscape(body_data, seg_results, vehicle, prefix):
    x      = body_data['x']; qw = body_data['q_wind'] / 1e3; ql = body_data['q_lee'] / 1e3
    q_stag = body_data['q_stag'] / 1e3

    xf = np.linspace(0, vehicle.L_body, 600)
    rf = np.array([vehicle.ogive.radius(xi) if xi <= vehicle.L_og else vehicle.R for xi in xf])
    qwf = np.interp(xf, x, qw); qlf = np.interp(xf, x, ql)
    vmin = min(qwf.min(), qlf.min()); vmax = max(qwf.max(), qlf.max(), q_stag)

    fig, ax = plt.subplots(figsize=(13, 5))
    sc_up = ax.scatter(xf,  rf, c=qlf, cmap='plasma', s=18, zorder=3, vmin=vmin, vmax=vmax)
    sc_lo = ax.scatter(xf, -rf, c=qwf, cmap='plasma', s=18, zorder=3, vmin=vmin, vmax=vmax)
    ax.fill_between(xf, -rf, rf, color='lightgray', alpha=0.2, zorder=1)
    ax.plot(xf,  rf, 'k-', lw=1.5); ax.plot(xf, -rf, 'k-', lw=1.5)
    ax.axvline(vehicle.L_og, color='gray', ls='--', lw=1,
               label=f'Ogive–cyl. x={vehicle.L_og:.2f} m')
    cp_lo, cp_hi = COCKPIT_X_RANGE
    ax.axvspan(cp_lo, cp_hi, color='cyan', alpha=0.15, zorder=0,
               label=f'Cockpit ({cp_lo:.1f}–{cp_hi:.1f} m)')
    ax.axvline(cp_lo, color='teal', ls=':', lw=1); ax.axvline(cp_hi, color='teal', ls=':', lw=1)
    ax.text(0.01, 0.95, 'Leeward',  transform=ax.transAxes, fontsize=9,
            va='top',    ha='left', style='italic', color='dimgray')
    ax.text(0.01, 0.05, 'Windward', transform=ax.transAxes, fontsize=9,
            va='bottom', ha='left', style='italic', color='dimgray')
    ax.set_xlabel('Axial x  [m]'); ax.set_ylabel('Radial r  [m]')
    ax.set_title(f'Thermal Landscape (windward + leeward) — {vehicle._title()}', fontsize=11)
    ax.set_aspect('equal'); ax.legend(loc='upper right', fontsize=8)
    cbar = fig.colorbar(sc_up, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Convective heat flux  [kW/m²]', fontsize=10)
    plt.tight_layout()
    _save_or_show(fig, f"{prefix}_thermal_landscape")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  HYPERSONIC AEROTHERMAL ANALYSIS  (multi-segment swept wing)")
    print("  Mode: " + ("HOT-WALL — radiation equilibrium" if HOT_WALL
                        else f"COLD-WALL — T_w = {WALL_TEMP_K} K (prescribed)"))
    print("=" * 64)

    T_inf, P_inf, rho_inf = us_standard_atmosphere(ALTITUDE_KM)
    a_inf = math.sqrt(GAMMA * R_AIR * T_inf)
    V_inf = MACH * a_inf
    print(f"  M={MACH}, h={ALTITUDE_KM} km, T∞={T_inf:.1f} K, "
          f"P∞={P_inf:.1f} Pa, V∞={V_inf:.0f} m/s")
    print(f"  Total temperature T0 = {T_inf + 0.5*V_inf**2/CP:.1f} K")

    vehicle = OgiveCylinderAnalysis(
        MACH, ALTITUDE_KM, WALL_TEMP_K, IS_TURBULENT, AOA_DEG,
        BODY_RADIUS_M, OGIVE_LENGTH_M, BODY_LENGTH_M, NOSE_RADIUS_M,
        hot_wall=HOT_WALL, emissivity=EMISSIVITY)

    print("\n  Running body analysis …")
    body_data = vehicle.run_body(NUM_POINTS)

    qs    = body_data['q_stag'] / 1e3
    Tws   = body_data['T_w_stag']
    print(f"  Nose stagnation: q = {qs:.2f} kW/m²   T_w = {Tws:.1f} K")

    if HOT_WALL:
        Tw_w_arr = body_data['Tw_wind']
        Tw_l_arr = body_data['Tw_lee']
        print(f"  Body windward T_w : min={Tw_w_arr.min():.0f} K  "
              f"max={Tw_w_arr.max():.0f} K  mean={Tw_w_arr.mean():.0f} K")
        print(f"  Body leeward  T_w : min={Tw_l_arr.min():.0f} K  "
              f"max={Tw_l_arr.max():.0f} K  mean={Tw_l_arr.mean():.0f} K")

    seg_results = []
    print("\n  Running wing segment analyses …")
    for (cr, ct, sp, sw, dih, lbl) in SEGMENTS:
        seg = SweptWingSegment(
            chord_root=cr, chord_tip=ct, span=sp,
            sweep_deg=sw, dihedral_deg=dih, label=lbl,
            le_radius=WING_LE_RADIUS_M, aoa_deg=AOA_DEG,
            M=MACH, r_inf=rho_inf, T_inf=T_inf,
            P_inf=P_inf, V_inf=V_inf,
            T_w=WALL_TEMP_K, turb=IS_TURBULENT,
            hot_wall=HOT_WALL, emissivity=EMISSIVITY)
        sr = seg.compute(200)
        seg_results.append(sr)
        print(f"  {lbl}:")
        print(f"    LE: q={sr['q_le']/1e3:.2f} kW/m²  T_w={sr['T_w_le']:.0f} K")
        print(f"    Lower max: q={sr['q_lower'].max()/1e3:.2f} kW/m²  "
              f"T_w range [{sr['Tw_lower'].min():.0f}–{sr['Tw_lower'].max():.0f}] K")
        print(f"    Upper max: q={sr['q_upper'].max()/1e3:.2f} kW/m²  "
              f"T_w range [{sr['Tw_upper'].min():.0f}–{sr['Tw_upper'].max():.0f}] K")

    title = vehicle._title()
    px    = FIGURE_PREFIX

    print("\nGenerating vector PDF figures …")
    plot_body(body_data, vehicle, px)
    plot_body_Tw(body_data, vehicle, px)                         # NEW
    plot_wing_all_segments(seg_results, body_data['q_stag']/1e3, px, title)
    plot_wing_Tw(seg_results, px, title)                         # NEW
    plot_le_spanwise(seg_results, px, title)
    plot_normalised_comparison(seg_results, body_data, px, title)
    plot_summary_bar(body_data, seg_results, px, title)
    plot_thermal_landscape(body_data, seg_results, vehicle, px)
    plot_all_heat_flux_overview(body_data, seg_results, vehicle, px, title)
    plot_cockpit_region(body_data, px, title)
    plot_wing_chordwise_curves(seg_results, px, title)

    print("\nAll figures saved as PDF.")
