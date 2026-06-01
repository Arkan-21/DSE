"""
plot_e2r_validation.py
======================
E2R scramjet wall-pressure validation against Li et al. (2023), Fig. 9.

Freestream reference: p0 = static pressure at Ma0=6.7, h=28 km (ISA).
Experiment data: provided directly (not digitised manually).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

from ramjet_fixedgeometry import AirProperties, MixtureNASA, ShapiroODE


# ---------------------------------------------------------------------------
# Engine parameters  (Li et al. 2023, Tables 1 & 2)
# ---------------------------------------------------------------------------
L12, L23, L34, L45        = 0.4, 0.01, 1.0, 0.4
ALPHA12, ALPHA13, ALPHA14 = 1.0,  1.1,  2.5
MA1, T1, P1, MDOT1        = 3.6, 760.0, 26_600.0, 1.51
PHI, FAR_STOICH_H2        = 0.37, 1.0 / 34.35
Q_H2_HHV                  = 141.8e6
CF                         = 0.002
X1                         = 0.60

# ---------------------------------------------------------------------------
# Freestream reference pressure  (ISA, h=28 km)
# ---------------------------------------------------------------------------
_h0, _L, _T0, _P0 = 20_000, 0.001, 216.65, 5474.89
T_FS = _T0 + _L * (28_000 - _h0)
P_FS = _P0 * (T_FS / _T0) ** (-9.80665 / (_L * 287.05))

print(f"Freestream: T0={T_FS:.2f} K,  p0={P_FS:.2f} Pa")

# ---------------------------------------------------------------------------
# Experiment data
# ---------------------------------------------------------------------------
EXP_F = np.array([
    [0.8264462809917356,  17.28813559322034],
    [0.9256198347107438,  15.762711864406775],
    [1.0,                 34.57627118644067],
    [1.0495867768595042,  34.57627118644067],
    [1.115702479338843,   37.1186440677966],
    [1.140495867768595,   40.16949152542372],
    [1.1735537190082646,  43.72881355932203],
    [1.2644628099173554,  32.03389830508474],
    [1.3140495867768596,  27.457627118644062],
    [1.396694214876033,   20.338983050847453],
    [1.4545454545454546,  11.694915254237287],
    [1.537190082644628,   11.186440677966104],
    [1.6198347107438016,   9.661016949152547],
    [2.0661157024793386,   2.5423728813559308],
    [2.1322314049586777,   2.0338983050847474],
])

EXP_U = np.array([
    [0.8264462809917356,  17.28813559322034],
    [0.9256198347107438,  16.779661016949156],
    [1.0,                 16.27118644067796],
    [1.0495867768595042,   7.118644067796609],
    [1.115702479338843,    7.118644067796609],
    [1.256198347107438,   17.28813559322034],
    [1.322314049586777,   13.220338983050844],
    [1.3884297520661157,  14.745762711864401],
    [1.4545454545454546,   7.118644067796609],
    [1.5454545454545454,   7.627118644067792],
    [1.6528925619834711,   6.101694915254235],
    [1.7851239669421488,   4.067796610169488],
    [1.909090909090909,    2.0338983050847474],
    [2.0661157024793386,   2.0338983050847474],
    [2.12396694214876,     1.0169491525423666],
])

# ---------------------------------------------------------------------------
# Thermodynamic setup
# ---------------------------------------------------------------------------
ap  = AirProperties()
mix = MixtureNASA(ap)

_m = ap.AIR_BASE_COMPOSITION; _t = sum(_m.values())
_W = sum(_m[s]/_t * ap.MOLECULAR_WEIGHTS[s] for s in _m)
Y_AIR = {s: _m[s]/_t * ap.MOLECULAR_WEIGHTS[s] / _W for s in _m}

_W1 = mix.W_mix(Y_AIR); _R1 = mix.R_UNIVERSAL / _W1
_g1 = mix.gamma_mix(Y_AIR, T1)
A1  = MDOT1 / (P1 / (_R1*T1) * MA1 * math.sqrt(_g1*_R1*T1))
A2, A3, A4 = ALPHA12*A1, ALPHA13*A1, ALPHA14*A1

MFUEL = PHI * FAR_STOICH_H2 * MDOT1
YF    = MFUEL / (MDOT1 + MFUEL)
Y_COMB = {sp: (1-YF)*Y_AIR.get(sp, 0.) for sp in Y_AIR}
Y_COMB["H2"] = YF

x1 = X1
x2, x3, x4, x5 = x1+L12, x1+L12+L23, x1+L12+L23+L34, x1+L12+L23+L34+L45
SECS = [x1, x2, x3, x4, x5]

# ---------------------------------------------------------------------------
# Mixing efficiency  (Eq. 28, Li et al.)
# ---------------------------------------------------------------------------
def deta_dx(x, theta, dx=1e-5):
    def eta(xx):
        s   = max(min((xx - x3) / L34, 1.0), 1e-6)
        e0  = s
        e90 = max(min(1.01 + 0.176 * math.log(s), 1.0), 0.0)
        if theta == 0:  return e0
        if theta == 90: return e90
        return theta/90 * (e90 - e0) + e0
    return (eta(min(x+dx, x4)) - eta(max(x-dx, x3))) / (2*dx)

# ---------------------------------------------------------------------------
# Section integrator
# ---------------------------------------------------------------------------
def run_section(x0, x1_, A_in, A_out,
                Ma_in, T_in, p_in, mdot_in,
                Y, heat_fn=None, mass_rate=0.0):
    L_ = x1_ - x0
    def geo(x):
        f = (x - x0) / L_ if L_ > 0 else 0.
        A = A_in + (A_out - A_in) * f
        return A, (A_out - A_in) / L_ if L_ > 0 else 0., math.sqrt(4*A/math.pi)
    def comp(x, T, p): return Y
    def src(x, T, p, m, _): return (heat_fn(x, T, p) if heat_fn else 0.), mass_rate
    return ShapiroODE.integrate(
        x_start=x0, x_end=x1_,
        Ma2_in=Ma_in**2, p_in=p_in, T_in=T_in, mdot_in=mdot_in,
        geometry_fn=geo, composition_fn=comp, source_fn=src, mix=mix,
        switches={"area":True,"friction":True,"mass":True,
                  "heat":True,"MW":False,"gamma":False},
        Cf=CF, n_steps=600,
    )

# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
print("Running simulations …")

def cat(secs, field):
    return np.concatenate([s[field] for s in secs])

# shared inlet + injection (identical for all runs)
r12 = run_section(x1,x2,A1,A2, MA1,T1,P1,MDOT1, Y_AIR)
r23 = run_section(x2,x3,A2,A3,
                  r12["Ma"][-1], r12["T"][-1], r12["p"][-1], MDOT1, Y_AIR,
                  mass_rate=MFUEL/L23)
mdot3 = r23["mdot"][-1]
Ma3, T3, p3 = r23["Ma"][-1], r23["T"][-1], r23["p"][-1]

# unfueled
ru34 = run_section(x3,x4,A3,A4, Ma3,T3,p3, MDOT1, Y_AIR)
ru45 = run_section(x4,x5,A4,A4*2/ALPHA14,
                   ru34["Ma"][-1], ru34["T"][-1], ru34["p"][-1], MDOT1, Y_AIR)
x_u = cat([r12, r23, ru34, ru45], "x")
p_u = cat([r12, r23, ru34, ru45], "p") / P_FS

# fueled, theta=0
def h0(x,T,p): return YF * Q_H2_HHV / L34
rf34_0 = run_section(x3,x4,A3,A4, Ma3,T3,p3, mdot3, Y_COMB, heat_fn=h0)
rf45_0 = run_section(x4,x5,A4,A4*2/ALPHA14,
                     rf34_0["Ma"][-1], rf34_0["T"][-1], rf34_0["p"][-1], mdot3, Y_COMB)
x_f0 = cat([r12, r23, rf34_0, rf45_0], "x")
p_f0 = cat([r12, r23, rf34_0, rf45_0], "p") / P_FS

# fueled, theta=90
def h90(x,T,p): return YF * Q_H2_HHV * deta_dx(x, 90)
rf34_90 = run_section(x3,x4,A3,A4, Ma3,T3,p3, mdot3, Y_COMB, heat_fn=h90)
rf45_90 = run_section(x4,x5,A4,A4*2/ALPHA14,
                      rf34_90["Ma"][-1], rf34_90["T"][-1], rf34_90["p"][-1], mdot3, Y_COMB)
x_f90 = cat([r12, r23, rf34_90, rf45_90], "x")
p_f90 = cat([r12, r23, rf34_90, rf45_90], "p") / P_FS

print(f"  theta=0  peak: p/p0={p_f0.max():.1f}  at x={x_f0[p_f0.argmax()]:.3f} m")
print(f"  theta=90 peak: p/p0={p_f90.max():.1f}  at x={x_f90[p_f90.argmax()]:.3f} m")
print(f"  Exp fueled peak: p/p0={EXP_F[:,1].max():.1f}  at x={EXP_F[EXP_F[:,1].argmax(),0]:.3f} m")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
SEC_LABELS = ["Isolator", "Injection", "Combustor", "Nozzle"]
BAND_COLORS = ["0.94", "0.90", "0.94", "0.90"]

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.12)

# section bands
for i in range(4):
    ax.axvspan(SECS[i], SECS[i+1], color=BAND_COLORS[i], zorder=0)
for xs in SECS[1:-1]:
    ax.axvline(xs, color="0.65", linewidth=0.8, linestyle="--", zorder=1)
for i in range(4):
    ax.text((SECS[i]+SECS[i+1])/2, 50.5, SEC_LABELS[i],
            ha="center", va="top", fontsize=8, color="0.45")

# experiment
l_ef, = ax.plot(EXP_F[:,0], EXP_F[:,1],
                marker="^", linestyle="none", color="tab:red",
                markersize=8, markerfacecolor="none", markeredgewidth=1.8,
                zorder=5, label="Experiment — fueled")
l_eu, = ax.plot(EXP_U[:,0], EXP_U[:,1],
                marker="s", linestyle="none", color="tab:blue",
                markersize=8, markerfacecolor="none", markeredgewidth=1.8,
                zorder=5, label="Experiment — unfueled")

# model
l_mu, = ax.plot(x_u, p_u,
                color="tab:blue", linewidth=2.0, linestyle="--", zorder=4,
                label="Model — unfueled")
l_f0, = ax.plot(x_f0, p_f0,
                color="tab:orange", linewidth=1.8, linestyle=(0,(5,2)), zorder=4,
                label=r"Model — fueled, $\theta=0°$  (uniform heat)")
l_f90, = ax.plot(x_f90, p_f90,
                 color="tab:red", linewidth=2.0, zorder=4,
                 label=r"Model — fueled, $\theta=90°$  (front-loaded heat)")

ax.set_xlim(0.77, 2.65)
ax.set_ylim(-1, 54)
ax.set_xlabel("$x$  (m)", fontsize=12)
ax.set_ylabel("$p \\ / \\ p_0$", fontsize=12)
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(2))
ax.grid(True, which="major", linewidth=0.5, color="0.80")
ax.grid(True, which="minor", linewidth=0.3, color="0.90")

ax.legend(
    handles=[
        mpatches.Patch(color="none",
                       label="Experiment  (Li et al. 2023, Fig. 9):"),
        l_ef, l_eu,
        mpatches.Patch(color="none", label=" "),
        mpatches.Patch(color="none",
                       label="Shapiro Q1D model  (this work):"),
        l_mu, l_f0, l_f90,
    ],
    loc="upper right", fontsize=8.0, framealpha=0.95,
    handlelength=2.2, borderpad=0.8,
)

ax.set_title(
    "E2R Scramjet — Wall pressure  $p/p_0$\n"
    r"$p_0$ = freestream static at $Ma_0=6.7$, $h=28$ km"
    f"  ($p_0={P_FS:.0f}$ Pa)"
    "\n"
    r"Li et al. (2023), Energy 267, 126400  —  "
    r"$Ma_1=3.6$,  $T_1=760$ K,  H$_2$,  $\varphi=0.37$",
    fontsize=9.5, pad=8, linespacing=1.6,
)

OUTPUT = "e2r_wall_pressure.png"
plt.savefig(OUTPUT, dpi=150)
print(f"\nSaved to {OUTPUT}")
plt.show()