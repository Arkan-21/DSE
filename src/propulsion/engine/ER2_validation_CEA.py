"""
plot_e2r_validation.py  (v3)
============================
E2R scramjet wall-pressure validation against Li et al. (2023), Fig. 9.

Combustion model: NASA CEA equilibrium  —  dH/dx = (h_react − h_eq) · dη/dx
  Same approach as section_34 in ramjet_fixedgeometry.py.
  Mixing coefficient 0.11 (re-fitted to E2R geometry; main model uses 0.176).

Mass and heat addition are simultaneous, starting at x = 1.0 m.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

from ramjet_fixedgeometry import AirProperties, MixtureNASA, ShapiroODE, CEAComp


# ---------------------------------------------------------------------------
# E2R geometry  (Hiraiwa 2006, Fig. 1, all in metres)
# ---------------------------------------------------------------------------
W1 = 70e-3;  W2 = 78e-3;  W3 = 78e-3;  W4 = 120e-3; W5 = 200e-3
H1 = 90e-3;  H2 = 94e-3;  H3 = 94e-3
H4 = 94e-3 + 640e-3 * math.tan(math.radians(9.0))
H5 = 250e-3

L12 = 160e-3    # isolator
L23 = 200e-3    # const-area combustor
L34 = 640e-3    # diverging combustor
L45 = 330e-3    # nozzle

A1 = W1*H1;  A2 = W2*H2;  A3 = W3*H3;  A4 = W4*H4;  A5 = W5*H5

def D_h(W_, H_): return 2*W_*H_ / (W_ + H_)

print("E2R geometry (Hiraiwa 2006, Fig. 1):")
print(f"  A1={A1*1e4:.2f} cm²  A2={A2*1e4:.2f}  A3={A3*1e4:.2f}  "
      f"A4={A4*1e4:.2f}  A5={A5*1e4:.2f}")
print(f"  A4/A1={A4/A1:.3f}  A3/A1={A3/A1:.3f}  (paper: 2.5 / 1.1)")


# ---------------------------------------------------------------------------
# Boundary conditions  (Li et al. 2023, Table 2)
# ---------------------------------------------------------------------------
MA1, T1_in, P1_in, MDOT1 = 3.6, 760.0, 26_600.0, 1.51

ap  = AirProperties()
mix = MixtureNASA(ap)
_m  = ap.AIR_BASE_COMPOSITION
_t  = sum(_m.values())
_W  = sum(_m[s]/_t * ap.MOLECULAR_WEIGHTS[s] for s in _m)
Y_AIR = {s: _m[s]/_t * ap.MOLECULAR_WEIGHTS[s]/_W for s in _m}


# ---------------------------------------------------------------------------
# Combustion parameters
# ---------------------------------------------------------------------------
PHI          = 0.37
FAR_STOICH   = 1.0 / 34.35
CF           = 0.003

MFUEL = PHI * FAR_STOICH * MDOT1
YF    = MFUEL / (MDOT1 + MFUEL)   # H2 mass fraction in fully-mixed stream

# Frozen reactant composition (air + unburnt H2) — used as the "before" state
# for the CEA enthalpy difference.  All CEA product species initialised to 0.
Y_REACT = {sp: (1 - YF) * Y_AIR.get(sp, 0.0) for sp in Y_AIR}
Y_REACT["H2"] = YF
for sp in CEAComp.PROD_NAMES:
    Y_REACT.setdefault(sp, 0.0)

# O/F ratio for CEA lookups
OF_RATIO = (1.0 - YF) / YF


# ---------------------------------------------------------------------------
# Freestream reference pressure  (ISA, h=28 km)
# ---------------------------------------------------------------------------
_h0, _L, _T0, _P0 = 20_000, 0.001, 216.65, 5474.89
T_FS = _T0 + _L * (28_000 - _h0)
P_FS = _P0 * (T_FS / _T0) ** (-9.80665 / (_L * 287.05))
print(f"Freestream p0 = {P_FS:.2f} Pa  (Ma=6.7, h=28 km)")


# ---------------------------------------------------------------------------
# Station x-coordinates
# ---------------------------------------------------------------------------
x3 = 1.16
x2 = x3 - L23      # 1.0  — fuel injection / combustion start
x1 = x2 - L12      # 0.84
x4 = x3 + L34      # 1.84
x5 = x4 + L45      # 2.17
SECS   = [x1, x2, x3, x4, x5]
X_FUEL = x2         # combustion zone starts here
L_COMB = x4 - X_FUEL

print(f"Stations (m): x1={x1:.3f}  x2={x2:.3f}  x3={x3:.3f}  "
      f"x4={x4:.3f}  x5={x5:.3f}")
print(f"Combustion zone: x={X_FUEL:.3f} → {x4:.3f} m  (L={L_COMB:.3f} m)")


# ---------------------------------------------------------------------------
# Experiment data  (Li et al. 2023, Fig. 9 — digitised)
# ---------------------------------------------------------------------------
EXP_F = np.array([
    [0.8264462809917356,  17.28813559322034  ],
    [0.9256198347107438,  15.762711864406775 ],
    [1.0,                 34.57627118644067  ],
    [1.0495867768595042,  34.57627118644067  ],
    [1.115702479338843,   37.1186440677966   ],
    [1.140495867768595,   40.16949152542372  ],
    [1.1735537190082646,  43.72881355932203  ],
    [1.2644628099173554,  32.03389830508474  ],
    [1.3140495867768596,  27.457627118644062 ],
    [1.396694214876033,   20.338983050847453 ],
    [1.4545454545454546,  11.694915254237287 ],
    [1.537190082644628,   11.186440677966104 ],
    [1.6198347107438016,   9.661016949152547 ],
    [2.0661157024793386,   2.5423728813559308],
    [2.1322314049586777,   2.0338983050847474],
])
EXP_U = np.array([
    [0.8264462809917356,  17.28813559322034  ],
    [0.9256198347107438,  16.779661016949156 ],
    [1.0,                 16.27118644067796  ],
    [1.0495867768595042,   7.118644067796609 ],
    [1.115702479338843,    7.118644067796609 ],
    [1.256198347107438,   17.28813559322034  ],
    [1.322314049586777,   13.220338983050844 ],
    [1.3884297520661157,  14.745762711864401 ],
    [1.4545454545454546,   7.118644067796609 ],
    [1.5454545454545454,   7.627118644067792 ],
    [1.6528925619834711,   6.101694915254235 ],
    [1.7851239669421488,   4.067796610169488 ],
    [1.909090909090909,    2.0338983050847474],
    [2.0661157024793386,   2.0338983050847474],
    [2.12396694214876,     1.0169491525423666],
])


# ---------------------------------------------------------------------------
# CEA wrapper instance  (shared across all combustion sections)
# ---------------------------------------------------------------------------
cea = CEAComp()


# ---------------------------------------------------------------------------
# Mixing efficiency  η(x) and dη/dx
#
# Coordinate s = (x - X_FUEL) / L_COMB  ∈ [0, 1] over the full combustion zone.
# Coefficient 0.11 is re-fitted to the E2R dataset (Li et al. 2023).
# The main ramjet model uses 0.176 (original Li et al. formulation).
# ---------------------------------------------------------------------------
MIXING_COEFF = 0.11   # E2R-fitted value

def mixing_eta(x, theta=90):
    s    = max(min((x - X_FUEL) / L_COMB, 1.0), 1e-6)
    e0   = s
    e90  = max(min(1.01 + MIXING_COEFF * math.log(s), 1.0), 0.0)
    if theta == 0:   return e0
    if theta == 90:  return e90
    return theta / 90 * (e90 - e0) + e0

def deta_dx(x, theta=90, dx=1e-5):
    x_lo = max(x - dx, X_FUEL)
    x_hi = min(x + dx, x4)
    return (mixing_eta(x_hi, theta) - mixing_eta(x_lo, theta)) / (2 * dx)


# ---------------------------------------------------------------------------
# CEA composition and enthalpy helpers
# ---------------------------------------------------------------------------
def Y_eq_at(T, p_pa):
    """Equilibrium product composition at (T, p); falls back to reactants."""
    Yeq = cea.equilibrium_Y(T, p_pa, OF_RATIO)
    return Yeq if Yeq is not None else Y_REACT

def Y_blended(eta, T, p_pa):
    """Linear blend between frozen reactants and CEA equilibrium products."""
    Yeq  = Y_eq_at(T, p_pa)
    keys = set(Y_REACT) | set(Yeq)
    return {k: (1 - eta)*Y_REACT.get(k, 0.0) + eta*Yeq.get(k, 0.0)
            for k in keys}


# ---------------------------------------------------------------------------
# Section integrator
# ---------------------------------------------------------------------------
def run_section(x0, x1_, W_in, W_out, H_in, H_out,
                Ma_in, T_in, p_in, mdot_in,
                composition_fn, heat_fn=None, mass_rate=0.0,
                switches=None):
    """
    Integrate the Shapiro ODE over one duct section.

    composition_fn(x, T, p) → Y dict   used for γ, cp, h, MW
    heat_fn(x, T, p)        → dH/dx    heat source term  [J/(kg·m)]
    mass_rate               → dṁ/dx    [kg/(s·m)]
    """
    L_ = x1_ - x0
    if switches is None:
        switches = {"area": True, "friction": True, "mass": True,
                    "heat": True, "MW": False, "gamma": False}

    def geometry_fn(x):
        f   = (x - x0) / L_ if L_ > 0 else 0.0
        W_  = W_in + (W_out - W_in) * f
        H_  = H_in + (H_out - H_in) * f
        A   = W_ * H_
        dA  = ((W_out - W_in)*H_ + W_*(H_out - H_in)) / L_ if L_ > 0 else 0.0
        D   = D_h(W_, H_)
        return A, dA, D

    def source_fn(x, T, p, m, _):
        return (heat_fn(x, T, p) if heat_fn else 0.0), mass_rate

    return ShapiroODE.integrate(
        x_start=x0, x_end=x1_,
        Ma2_in=Ma_in**2, p_in=p_in, T_in=T_in, mdot_in=mdot_in,
        geometry_fn=geometry_fn,
        composition_fn=composition_fn,
        source_fn=source_fn,
        mix=mix,
        switches=switches,
        Cf=CF, n_steps=600,
    )


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
print("\nRunning simulations …")

def cat(secs, field):
    return np.concatenate([s[field] for s in secs])

# ── Shared: isolator  ────────────────────────────────────────────────────────
ISO_SW = {"area": True, "friction": False, "mass": False,
          "heat": False, "MW": False, "gamma": False}

r12 = run_section(
    x1, x2, W1, W1, H1, H1,
    MA1, T1_in, P1_in, MDOT1,
    composition_fn=lambda x, T, p: Y_AIR,
    switches=ISO_SW,
)

Ma_iso, T_iso, p_iso = r12["Ma"][-1], r12["T"][-1], r12["p"][-1]

# ── Unfueled: constant-area → diverging → nozzle ─────────────────────────────
r23u = run_section(x2, x3, W2, W3, H2, H3,
                   Ma_iso, T_iso, p_iso, MDOT1,
                   composition_fn=lambda x, T, p: Y_AIR)
ru34 = run_section(x3, x4, W3, W4, H3, H4,
                   r23u["Ma"][-1], r23u["T"][-1], r23u["p"][-1], MDOT1,
                   composition_fn=lambda x, T, p: Y_AIR)
ru45 = run_section(x4, x5, W4, W5, H4, H5,
                   ru34["Ma"][-1], ru34["T"][-1], ru34["p"][-1], MDOT1,
                   composition_fn=lambda x, T, p: Y_AIR)

x_u = cat([r12, r23u, ru34, ru45], "x")
p_u = cat([r12, r23u, ru34, ru45], "p") / P_FS

# ── Fueled θ=90: CEA heat release + mass addition from X_FUEL ───────────────
#
#   Heat release:  dH/dx = (h_reactants − h_equilibrium) · dη/dx   [CEA]
#   Composition:   blended Y(η, T, p) for accurate γ and cp
#   Mass addition: MFUEL spread uniformly over L_COMB
#
MASS_RATE_PER_M = MFUEL / L_COMB   # kg/(s·m)

def composition_fueled(x, T, p):
    """Blended composition — drives γ, cp, MW, h inside the ODE."""
    if x < X_FUEL or x > x4:
        return Y_AIR
    eta = mixing_eta(x, theta=90)
    return Y_blended(eta, T, p)

def heat_cea(x, T, p):
    """
    CEA-based heat release rate per unit length [J/(kg·m)].
    dH/dx = (h_reactants − h_equilibrium) · dη/dx
    """
    h_react = mix.h_mix(Y_REACT, T)
    h_eq    = mix.h_mix(Y_eq_at(T, p), T)
    return (h_react - h_eq) * deta_dx(x, theta=90)

# Section A: X_FUEL → x3  (constant-area, W2×H2)
rfA = run_section(
    X_FUEL, x3, W2, W2, H2, H2,
    Ma_iso, T_iso, p_iso, MDOT1,
    composition_fn=composition_fueled,
    heat_fn=heat_cea,
    mass_rate=MASS_RATE_PER_M,
)

# Section B: x3 → x4  (diverging, W3→W4, H3→H4)
rfB = run_section(
    x3, x4, W3, W4, H3, H4,
    rfA["Ma"][-1], rfA["T"][-1], rfA["p"][-1], rfA["mdot"][-1],
    composition_fn=composition_fueled,
    heat_fn=heat_cea,
    mass_rate=MASS_RATE_PER_M,
)

# Nozzle: frozen equilibrium products, no combustion
rf45 = run_section(
    x4, x5, W4, W5, H4, H5,
    rfB["Ma"][-1], rfB["T"][-1], rfB["p"][-1], rfB["mdot"][-1],
    composition_fn=lambda x, T, p: Y_eq_at(T, p),
)

x_f90 = cat([r12, rfA, rfB, rf45], "x")
p_f90 = cat([r12, rfA, rfB, rf45], "p") / P_FS

print(f"  θ=90 CEA peak: p/p0={p_f90.max():.1f}  "
      f"at x={x_f90[p_f90.argmax()]:.3f} m")
print(f"  Exp peak:      p/p0={EXP_F[:,1].max():.1f}  "
      f"at x={EXP_F[EXP_F[:,1].argmax(),0]:.3f} m")
mdot_exit = rfB["mdot"][-1]
print(f"  ṁ_fuel={MFUEL:.4f} kg/s  "
      f"ṁ_exit={mdot_exit:.4f} kg/s  (expected {MDOT1+MFUEL:.4f})")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
SEC_LABELS  = ["Isolator", "Const. area", "Div. combustor", "Nozzle"]
BAND_COLORS = ["0.94", "0.90", "0.94", "0.90"]

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.12)

for i in range(4):
    ax.axvspan(SECS[i], SECS[i+1], color=BAND_COLORS[i], zorder=0)
for xs in SECS[1:-1]:
    ax.axvline(xs, color="0.65", linewidth=0.8, linestyle="--", zorder=1)
for i in range(4):
    ax.text((SECS[i]+SECS[i+1])/2, 50.5, SEC_LABELS[i],
            ha="center", va="top", fontsize=8, color="0.45")



l_ef, = ax.plot(EXP_F[:,0], EXP_F[:,1],
                marker="^", linestyle="none", color="tab:red",
                markersize=8, markerfacecolor="none", markeredgewidth=1.8,
                zorder=5, label="Experiment — fueled")
l_eu, = ax.plot(EXP_U[:,0], EXP_U[:,1],
                marker="s", linestyle="none", color="tab:blue",
                markersize=8, markerfacecolor="none", markeredgewidth=1.8,
                zorder=5, label="unfueled")
l_mu, = ax.plot(x_u,   p_u,
                color="tab:blue", linewidth=2.0, linestyle="--", zorder=4,
                label="unfueled")
l_f90, = ax.plot(x_f90, p_f90,
                 color="tab:red", linewidth=2.0, zorder=4,
                 label=r"fueled, injection angle $\theta=90°$  ")

ax.set_xlim(0.77, 2.25)
ax.set_ylim(-1, 54)
ax.set_xlabel("$x$  (m)", fontsize=12)
ax.set_ylabel("$p\\ /\\ p_0$", fontsize=12)
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(2))
ax.grid(True, which="major", linewidth=0.5, color="0.80")
ax.grid(True, which="minor", linewidth=0.3, color="0.90")

ax.legend(handles=[
    mpatches.Patch(color="none", label="Experimental Data:"),
    l_ef, l_eu,
    mpatches.Patch(color="none", label=" "),
    mpatches.Patch(color="none", label="Model Data:"),
    l_mu, l_f90,
], loc="upper right", fontsize=8.0, framealpha=0.95,
   handlelength=2.2, borderpad=0.8)

'''ax.set_title(
    "E2R Scramjet — Wall pressure  $p/p_0$\n"
    r"Geometry: $W_1=70,\;W_{2-3}=78,\;W_4=120,\;W_5=200$ mm; "
    r"$H_1=90,\;H_{2-3}=94,\;H_4=" + f"{H4*1e3:.0f}" + r",\;H_5=250$ mm  (Hiraiwa 2006)"
    "\n"
    r"$p_0=$freestream static at $Ma_0=6.7$, $h=28$ km  "
    r"($p_0=$" + f"{P_FS:.0f} Pa)" + r"  —  $\varphi=0.37$  —  "
    r"CEA combustion model,  mass+heat from $x=1.0$ m",
    fontsize=9.5, pad=8, linespacing=1.6)'''

#OUTPUT = "/mnt/user-data/outputs/e2r_wall_pressure_v3.png"
#plt.savefig(OUTPUT, dpi=150)
#print(f"\nSaved to {OUTPUT}")
plt.show()