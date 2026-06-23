"""
constraint_visualizations.py
============================
Generates presentation-quality figures for every constraint used in the
aerodynamic shape optimisation (geometry_generator.py).

No OpenVSP or hypersonic-flow solver required — pure numpy / matplotlib.
Run:  python constraint_visualizations.py
Output: constraint_figures/  (one PNG per constraint + one summary sheet)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── shared style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

OUT_DIR = "constraint_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── constants (mirrored from geometry_generator.py) ───────────────────────────
TOTAL_LENGTH   = 60.5      # m  — fuselage
SHOCK_START_X  = 2.7       # m  — effective apex of attached bow shock (ahead of nose)
SHOCK_ANGLE    = 16.26     # °  — shock half-angle
CL_MIN         = 0.02      # —  — minimum lift coefficient
MACH_DESIGN    = 4.349
ALPHA_DESIGN   = 3.0       # °

# Baseline geometry (deparametrized values)
CON_CHORD  = 47.85          # m
CHORD0     = 39.14
CHORD1     = 18.70
CHORD2     = 11.27
CHORD3     =  3.50
CON_SPAN   =  3.50          # m (half-span each section)
SPAN1      =  6.00
SPAN2      =  3.33
SPAN3      =  3.29
CON_SWEEP  = 68.0           # °
SWEEP1     = 73.55
SWEEP2     = 64.84
SWEEP3     = 70.50
DIHEDRAL3  = -20.0          # ° (wing tip canted down)
WING_X_LOC =  9.0           # m  — LE at fuselage root

# ── fuselage geometry (from fairing_calculator.py) ────────────────────────────
FUSE_R      = 2.0          # fuselage max radius [m]
_R_TO_L     = 0.1906       # nose: R / L_nose ratio
_L_TO_RHO   = 0.3679       # nose: L_nose / rho (osculating circle radius)
NOSE_L      = FUSE_R / _R_TO_L          # nose section length ≈ 10.49 m
NOSE_RHO    = NOSE_L / _L_TO_RHO        # osculating circle radius ≈ 28.52 m
_TAIL_SH_L  = 50.0         # Sears-Haack reference body length [m]
_TAIL_START = 0.8          # tail occupies the last (1 − _TAIL_START) fraction of SH body
TAIL_L      = (1 - _TAIL_START) * _TAIL_SH_L   # physical tail length ≈ 10 m

# Pre-compute derived baseline quantities
_spans     = [CON_SPAN, SPAN1, SPAN2, SPAN3]
_sweeps    = [CON_SWEEP, SWEEP1, SWEEP2, SWEEP3]
_chords    = [CON_CHORD, CHORD0, CHORD1, CHORD2, CHORD3]

def _wing_stations():
    """Return leading-edge x positions and cumulative span positions."""
    x_le = [WING_X_LOC]
    for sp, sw in zip(_spans, _sweeps):
        x_le.append(x_le[-1] + sp * np.tan(np.radians(sw)))
    x_te   = [xl + c for xl, c in zip(x_le, _chords)]
    y_span = np.concatenate([[0], np.cumsum(_spans)])
    return np.array(x_le), np.array(x_te), y_span

X_LE, X_TE, Y_SPAN = _wing_stations()
HALF_SPAN   = Y_SPAN[-1]
X_LE_TIP    = X_LE[-1]                        # LE of outermost section, from nose
X_TIP_SHOCK = X_LE_TIP + SHOCK_START_X        # same, measured from shock apex
SHOCK_Y_AT_TIP = X_TIP_SHOCK * np.tan(np.radians(SHOCK_ANGLE))
SHOCK_MARGIN   = SHOCK_Y_AT_TIP - HALF_SPAN   # > 0 → feasible

MAX_TE         = X_TE.max()
STRUCT_MARGIN  = TOTAL_LENGTH - MAX_TE         # > 0 → feasible


# ─────────────────────────────────────────────────────────────────────────────
# helper: fuselage profile (nose arc + cylinder + Sears-Haack tail)
# ─────────────────────────────────────────────────────────────────────────────
def _sh_volume(R, L, tail_start):
    denom = (tail_start * (1 - tail_start)) ** (3 / 4)
    return (R * np.pi / 8 / denom) ** 2 * 3 * L / 2

def _sh_radius(L, x_norm, volume):
    return 8 / np.pi * np.sqrt(2 * volume / 3 / L) * (x_norm * (1 - x_norm)) ** (3 / 4)

def _fuselage_half_profile(n=300):
    """Return (x, r) half-profile: nose tip (x=0, r=0) → shoulder (r=FUSE_R) → tail end (r→0)."""
    n3 = max(n // 3, 30)
    # Nose: circular-arc fairing. In fairing_calculator coords, x_calc=0 is the shoulder,
    # x_calc=NOSE_L is the tip, so in fuselage coords x_fuse = NOSE_L - x_calc.
    x_nose = np.linspace(0, NOSE_L, n3)
    r_nose = np.sqrt(NOSE_RHO ** 2 - (NOSE_L - x_nose) ** 2) - (NOSE_RHO - FUSE_R)
    # Cylindrical mid-section
    x_cyl  = np.linspace(NOSE_L, TOTAL_LENGTH - TAIL_L, n3)
    r_cyl  = np.full_like(x_cyl, FUSE_R)
    # Tail: Sears-Haack body (same formulas as fairing_calculator.py)
    x_tail  = np.linspace(TOTAL_LENGTH - TAIL_L, TOTAL_LENGTH, n3)
    vol     = _sh_volume(FUSE_R, _TAIL_SH_L, _TAIL_START)
    x_norm  = _TAIL_START + (x_tail - (TOTAL_LENGTH - TAIL_L)) / TAIL_L * (1 - _TAIL_START)
    r_tail  = np.clip(_sh_radius(_TAIL_SH_L, x_norm, vol), 0, None)
    x = np.concatenate([x_nose, x_cyl[1:], x_tail[1:]])
    r = np.concatenate([r_nose, r_cyl[1:], r_tail[1:]])
    return x, r

def _draw_fuselage(ax, y_center=0.0, color="lightgrey", ec="k", lw=1.5,
                   alpha=0.85, label="Fuselage body", zorder=1, top_only=False):
    """Draw the fuselage longitudinal silhouette (nose arc + cylinder + Sears-Haack tail)."""
    x, r = _fuselage_half_profile()
    if top_only:
        poly_x = np.concatenate([x, x[::-1]])
        poly_y = np.concatenate([r + y_center, np.full(len(x), y_center)])
    else:
        poly_x = np.concatenate([x, x[::-1]])
        poly_y = np.concatenate([r + y_center, -r[::-1] + y_center])
    ax.fill(poly_x, poly_y, color=color, ec=ec, lw=lw, alpha=alpha,
            zorder=zorder, label=label)
    ax.plot(x,  r + y_center, color=ec, lw=lw, zorder=zorder + 1)
    if not top_only:
        ax.plot(x, -r + y_center, color=ec, lw=lw, zorder=zorder + 1)
    else:
        ax.plot([x[0], x[-1]], [y_center, y_center],
                color=ec, lw=lw * 0.7, ls="--", zorder=zorder + 1)


# ─────────────────────────────────────────────────────────────────────────────
# helper: draw a half-wing polygon on ax
# ─────────────────────────────────────────────────────────────────────────────
def _draw_wing(ax, color="steelblue", alpha=0.40, label=None, scale=1.0,
               x_offset=0.0, y_offset=0.0):
    xl = X_LE * scale + x_offset
    xt = X_TE * scale + x_offset
    ys = Y_SPAN * scale + y_offset
    poly_x = np.concatenate([xl, xt[::-1]])
    poly_y = np.concatenate([ys, ys[::-1]])
    ax.fill(poly_x, poly_y, color=color, alpha=alpha, label=label)
    ax.plot(xl, ys, "-",  color=color, lw=1.5)
    ax.plot(xt, ys, "--", color=color, lw=1.5)
    ax.plot([xl[0], xt[0]], [ys[0], ys[0]], "-",  color=color, lw=1.5)
    ax.plot([xl[-1], xt[-1]], [ys[-1], ys[-1]], "-", color=color, lw=1.5)
    return xl, xt, ys


# ═════════════════════════════════════════════════════════════════════════════
# 1 & 2.  SHOCK CONTAINMENT + STRUCTURAL FEASIBILITY (combined)
# ═════════════════════════════════════════════════════════════════════════════
def plot_shock_and_structures_constraint():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_aspect("equal")

    shock_tan = np.tan(np.radians(SHOCK_ANGLE))

    # --- Constraint 1: forbidden (outside-shock) region ---
    x_fill = [0, TOTAL_LENGTH, TOTAL_LENGTH, 0]
    y_shock_L = (TOTAL_LENGTH + SHOCK_START_X) * shock_tan
    y_fill = [(0 + SHOCK_START_X) * shock_tan, y_shock_L, 25, 25]
    ax.fill(x_fill, y_fill, color="tomato", alpha=0.12,
            label="Outside shock (forbidden)")

    # shock boundary line
    x_shock_line = np.array([-2.7, TOTAL_LENGTH])
    y_shock_line = (x_shock_line + SHOCK_START_X) * shock_tan
    ax.plot(x_shock_line, y_shock_line, "r--", lw=2.0,
            label=f"Bow shock boundary ({SHOCK_ANGLE}°)")

    # shock angle annotation
    ax.annotate("", xy=(15, (15 + SHOCK_START_X) * shock_tan),
                xytext=(15, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ax.text(15.4, (15 + SHOCK_START_X) * shock_tan / 2,
            f"{SHOCK_ANGLE}°", color="red", fontsize=9)

    # --- Shared geometry ---
    _draw_fuselage(ax, top_only=True, zorder=2, label="Fuselage body")
    _draw_wing(ax, color="steelblue", alpha=0.45, label="Wing planform (baseline)")

    
    # --- Constraint 1: wing tip annotation ---
    ax.plot(X_LE_TIP, HALF_SPAN, "bo", ms=9, zorder=6)
    '''
    ax.annotate(f"Wing tip LE\nx = {X_LE_TIP:.1f} m,  y = {HALF_SPAN:.2f} m\n"
                rf"Shock margin = {SHOCK_MARGIN:.2f} m  ✓",
                xy=(X_LE_TIP, HALF_SPAN),
                xytext=(X_LE_TIP - 18, HALF_SPAN + 1.2),
                arrowprops=dict(arrowstyle="->", color="navy", lw=1.2),
                fontsize=9, color="navy")
    '''
    # --- Constraint 2: fuselage end and max TE lines ---
    ax.axvline(TOTAL_LENGTH, color="darkred", lw=2.5, ls="-",
               label=f"Fuselage end  ({TOTAL_LENGTH} m)")
    ax.axvline(MAX_TE, color="navy", lw=2.0, ls=":",
               label=f"Max wing TE = {MAX_TE:.1f} m")
    
    # structural margin arrow (horizontal, above wing tip)
    y_arr = HALF_SPAN + 0.8
    ax.annotate("", xy=(TOTAL_LENGTH, y_arr), xytext=(MAX_TE, y_arr),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=2.0))
    # ax.text((TOTAL_LENGTH + MAX_TE) / 2, y_arr + 0.35,
    #         f"Struct. margin = {STRUCT_MARGIN:.1f} m  ✓",
    #         ha="center", color="purple", fontsize=10, fontweight="bold")

    ax.set_xlim(-3, TOTAL_LENGTH + 3)
    ax.set_ylim(-FUSE_R - 0.5, HALF_SPAN + 3.5)
    ax.set_xlabel("x (m)  —  fuselage axis")
    ax.set_ylabel("y (m)  —  half-span")
    # ax.set_title("Constraints 1 & 2 — Shock Containment and Structural Feasibility\n"
    #              "Wing tip LE inside bow shock cone  |  All wing TEs within fuselage length",
    #              fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "constraint_1_2_shock_structures.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MINIMUM REFERENCE AREA CONSTRAINT
# ═════════════════════════════════════════════════════════════════════════════
def plot_area_constraint():
    """Compare a shrunken wing (infeasible) vs the baseline (feasible)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Reference area from the baseline (trapezoidal rule, both halves)
    def _ref_area(scale):
        chds = np.array(_chords) * scale
        spns = np.array([0] + list(np.cumsum(np.array(_spans) * scale)))
        area = 0.0
        for i in range(len(chds) - 1):
            area += 0.5 * (chds[i] + chds[i+1]) * (spns[i+1] - spns[i])
        return 2 * area  # both halves

    target_area = _ref_area(1.0)  # baseline is the minimum

    for ax, scale, feasible in zip(axes, [0.72, 1.0], [False, True]):
        xl = X_LE * scale
        xt = X_TE * scale
        ys = Y_SPAN * scale
        color = "steelblue" if feasible else "tomato"
        poly_x = np.concatenate([xl, xt[::-1]])
        poly_y = np.concatenate([ys, ys[::-1]])
        ax.fill(poly_x, poly_y, color=color, alpha=0.45)
        ax.plot(xl, ys, "-",  color=color, lw=1.5)
        ax.plot(xt, ys, "--", color=color, lw=1.5)
        ax.plot([xl[0], xt[0]], [ys[0], ys[0]], "-",  color=color, lw=2)
        ax.plot([xl[-1], xt[-1]], [ys[-1], ys[-1]], "-", color=color, lw=2)

        area = _ref_area(scale)
        status = "✓  FEASIBLE" if feasible else "✗  INFEASIBLE"
        color_txt = "darkgreen" if feasible else "darkred"
        ax.text(0.5, 0.06, f"S_ref ≈ {area:.0f} m²\n{status}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=color_txt,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=color_txt, lw=2))

        # Fuselage silhouette
        fuse_r = 2.0 * scale
        ax.add_patch(plt.Rectangle((0, 0), TOTAL_LENGTH * scale, fuse_r,
                                   color="lightgrey", ec="grey", lw=1, alpha=0.5, zorder=0))

        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y / span (m)")
        ax.set_title(f"{'Baseline wing' if feasible else 'Scaled-down wing'} "
                     f"(scale = {scale:.2f})\n"
                     f"S_ref = {area:.0f} m²  "
                     f"{'≥' if feasible else '<'}  target {target_area:.0f} m²")
        ax.grid(True, alpha=0.3)

    # formula
    axes[0].text(0.02, 0.99,
                 r"$g_{\rm area} = (S_{\rm ref} - S_{\rm baseline}) / S_{\rm baseline} \geq 0$",
                 transform=axes[0].transAxes, va="top", fontsize=9.5,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="steelblue", lw=1.5))

    fig.suptitle("Constraint 3 — Minimum Reference Area\n"
                 "Projected planform area must be ≥ baseline S_ref to preserve lift capability",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "constraint_3_area.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MINIMUM LIFT COEFFICIENT CONSTRAINT
# ═════════════════════════════════════════════════════════════════════════════
def plot_cl_constraint():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: CL vs alpha (Newtonian-like flat-plate approximation for illustration)
    ax = axes[0]
    alpha = np.linspace(-2, 18, 300)
    CL = 0.05 * alpha * (1 - (alpha / 40) ** 2)  # smooth empirical-ish curve
    ax.plot(alpha, CL, "b-", lw=2.0, label="CL(α)  [illustrative]")
    ax.axhline(CL_MIN, color="red", lw=2.0, ls="--",
               label=f"CL_min = {CL_MIN}")
    ax.axvline(ALPHA_DESIGN, color="green", lw=1.8, ls=":",
               label=f"Design α = {ALPHA_DESIGN}°")
    ax.fill_between(alpha, CL_MIN, CL,
                    where=CL >= CL_MIN, color="green", alpha=0.15, label="Feasible (CL ≥ CL_min)")
    ax.fill_between(alpha, CL_MIN, CL,
                    where=CL < CL_MIN, color="red",   alpha=0.10, label="Infeasible")

    CL_design = 0.05 * ALPHA_DESIGN * (1 - (ALPHA_DESIGN / 40) ** 2)
    ax.plot(ALPHA_DESIGN, CL_design, "go", ms=10, zorder=6,
            label=f"Design CL ≈ {CL_design:.3f}")
    ax.annotate(f"CL = {CL_design:.3f}\n> CL_min = {CL_MIN}  ✓",
                xy=(ALPHA_DESIGN, CL_design),
                xytext=(ALPHA_DESIGN + 3, CL_design + 0.04),
                arrowprops=dict(arrowstyle="->", color="green"), fontsize=9)
    ax.set_xlabel("Angle of Attack α (°)")
    ax.set_ylabel("Lift Coefficient  CL")
    ax.set_title("CL vs Angle of Attack  (M = 4.35)")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 18)
    ax.set_ylim(-0.05, 0.7)

    # Right: CL vs Mach at design alpha
    ax = axes[1]
    mach = np.linspace(1.5, 12, 300)
    CL_M = 0.22 * (1 + 0.8 / mach ** 1.1) * np.sin(np.radians(ALPHA_DESIGN)) * np.cos(np.radians(ALPHA_DESIGN))
    ax.plot(mach, CL_M, "b-", lw=2.0, label=f"CL(M) at α = {ALPHA_DESIGN}°  [illustrative]")
    ax.axhline(CL_MIN, color="red", lw=2.0, ls="--", label=f"CL_min = {CL_MIN}")
    ax.axvline(MACH_DESIGN, color="green", lw=1.8, ls=":",
               label=f"Design M = {MACH_DESIGN}")
    ax.fill_between(mach, CL_MIN, CL_M,
                    where=CL_M >= CL_MIN, color="green", alpha=0.15, label="Feasible")
    ax.fill_between(mach, CL_MIN, CL_M,
                    where=CL_M < CL_MIN, color="red",   alpha=0.10, label="Infeasible")
    CL_design_M = 0.22 * (1 + 0.8 / MACH_DESIGN**1.1) * np.sin(np.radians(ALPHA_DESIGN)) * np.cos(np.radians(ALPHA_DESIGN))
    ax.plot(MACH_DESIGN, CL_design_M, "go", ms=10, zorder=6,
            label=f"Design CL ≈ {CL_design_M:.3f}")
    ax.set_xlabel("Mach Number")
    ax.set_ylabel("Lift Coefficient  CL")
    ax.set_title(f"CL vs Mach  (α = {ALPHA_DESIGN}°)")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # formula
    axes[0].text(0.02, 0.99,
                 r"$g_{\rm CL} = C_L - C_{L,\min} \geq 0$"
                 f"\n$C_{{L,min}} = {CL_MIN}$",
                 transform=axes[0].transAxes, va="top", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="steelblue", lw=1.5))

    fig.suptitle(f"Constraint 4 — Minimum Lift Coefficient\n"
                 f"CL ≥ {CL_MIN} at M = {MACH_DESIGN}, α = {ALPHA_DESIGN}°",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "constraint_4_cl.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  PARAMETER BOUNDS  (implicit variable constraints)
# ═════════════════════════════════════════════════════════════════════════════
def plot_parameter_bounds():
    # (label, lower, upper, baseline value) — all in human-readable physical units
    bounds_data = [
        ("con_chord  (m)",         0.0,   0.8 * TOTAL_LENGTH,   CON_CHORD),
        ("chord0  (% con_chord)",  0.1,   100.0,                CHORD0 / CON_CHORD * 100),
        ("chord1  (% chord0)",     0.1,   100.0,                CHORD1 / CHORD0   * 100),
        ("chord2  (% chord1)",     0.1,   100.0,                CHORD2 / CHORD1   * 100),
        ("chord3  (% chord2)",     0.1,   100.0,                CHORD3 / CHORD2   * 100),
        ("sweep 1  (°)",           0.0,    85.0,                SWEEP1),
        ("sweep 2  (°)",           0.0,    85.0,                SWEEP2),
        ("sweep 3  (°)",           0.0,    85.0,                SWEEP3),
        ("con_sweep  (°)",         0.0,    85.0,                CON_SWEEP),
        ("dihedral 1  (°)",      -90.0,    90.0,                1.0),
        ("dihedral 2  (°)",      -90.0,    90.0,                1.0),
        ("dihedral 3  (°)",      -90.0,    90.0,                DIHEDRAL3),
        ("wing_x_loc  (m)",        0.0,    0.5 * TOTAL_LENGTH,  WING_X_LOC),
        ("camber  (%)",            0.0,    40.0,                0.1),
        ("max_camber_loc  (0–1)",  0.0,     1.0,                0.4),
        ("con_span  (m)",          0.0,    15.0,                CON_SPAN),
    ]

    labels     = [d[0] for d in bounds_data]
    lows       = np.array([d[1] for d in bounds_data])
    highs      = np.array([d[2] for d in bounds_data])
    baselines  = np.array([d[3] for d in bounds_data])

    # Normalize to [0, 1] for uniform bar display
    ranges = highs - lows
    norm_base = (baselines - lows) / np.where(ranges > 0, ranges, 1.0)

    n = len(labels)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(y, 1.0, left=0.0, height=0.5, color="lightsteelblue",
            edgecolor="steelblue", lw=1.2, label="Feasible range")
    ax.scatter(norm_base, y, color="navy", s=70, zorder=5,
               label="Baseline design value")

    for i in range(n):
        ax.text(-0.01, i, f"{lows[i]:.1f}",  va="center", ha="right", fontsize=8, color="grey")
        ax.text( 1.01, i, f"{highs[i]:.1f}", va="center", ha="left",  fontsize=8, color="grey")
        ax.text(norm_base[i], i + 0.35,
                f"{baselines[i]:.2f}", va="bottom", ha="center",
                fontsize=7.5, color="navy")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["Lower\nbound", "25 %", "50 %", "75 %", "Upper\nbound"], fontsize=9)
    ax.set_xlim(-0.18, 1.18)
    ax.axvline(0.0, color="red", lw=1.2, ls="--", alpha=0.5)
    ax.axvline(1.0, color="red", lw=1.2, ls="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("Constraint 5 — Parameter Bounds (Implicit Design-Variable Constraints)\n"
                 "Each design variable is restricted to a physically meaningful range",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "constraint_5_bounds.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  ONE-PAGE SUMMARY  (2×2 panel)
# ═════════════════════════════════════════════════════════════════════════════
def plot_summary():
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    shock_tan = np.tan(np.radians(SHOCK_ANGLE))

    # ── Panel 1: shock ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_aspect("equal")
    x_fill = [0, TOTAL_LENGTH, TOTAL_LENGTH, 0]
    y_fill = [(0 + SHOCK_START_X)*shock_tan,
              (TOTAL_LENGTH + SHOCK_START_X)*shock_tan, 22, 22]
    ax.fill(x_fill, y_fill, color="tomato", alpha=0.12)
    xsl = np.array([0.0, TOTAL_LENGTH])
    ax.plot(xsl, (xsl + SHOCK_START_X)*shock_tan, "r--", lw=2,
            label=f"Shock ({SHOCK_ANGLE}°)")
    _draw_fuselage(ax, top_only=True, zorder=2, label="Fuselage body")
    _draw_wing(ax, color="steelblue", alpha=0.45, label="Wing")

    x_shock_at_span = HALF_SPAN / shock_tan - SHOCK_START_X
    y_arr = HALF_SPAN + 0.6
    ax.annotate("", xy=(x_shock_at_span, y_arr), xytext=(X_LE_TIP, y_arr),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.8))
    ax.text((X_LE_TIP + x_shock_at_span)/2, y_arr + 0.5,
            f"margin = {x_shock_at_span - X_LE_TIP:.2f} m",
            ha="center", color="darkgreen", fontsize=9, fontweight="bold")
    ax.set_xlim(-2, 65); ax.set_ylim(-FUSE_R - 0.5, 16)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("① Shock Containment\n"
                 r"$x_{\rm tip,shock}\cdot\tan\mu - b/2 \geq 0$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 2: structures ───────────────────────────────────────────────────
    ax = axes[0, 1]
    _draw_fuselage(ax, zorder=1)
    y0 = FUSE_R
    for i in range(len(X_LE) - 1):
        ax.fill([X_LE[i], X_LE[i+1], X_TE[i+1], X_TE[i]],
                [y0 + Y_SPAN[i]*0.4, y0 + Y_SPAN[i+1]*0.4,
                 y0 + Y_SPAN[i+1]*0.4, y0 + Y_SPAN[i]*0.4],
                color="steelblue", alpha=0.4 + 0.15*i)
    ax.axvline(TOTAL_LENGTH, color="red",  lw=2.5, ls="--", label=f"Fuselage end")
    ax.axvline(MAX_TE,       color="navy", lw=2.0, ls=":",  label=f"Max TE = {MAX_TE:.1f} m")
    y_arr2 = y0 + HALF_SPAN * 0.4 + 0.5
    ax.annotate("", xy=(TOTAL_LENGTH, y_arr2), xytext=(MAX_TE, y_arr2),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=1.8))
    ax.text((TOTAL_LENGTH + MAX_TE)/2, y_arr2 + 0.3,
            f"margin = {STRUCT_MARGIN:.1f} m",
            ha="center", color="purple", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 72); ax.set_ylim(-FUSE_R - 0.5, y0 + HALF_SPAN*0.4 + 2)
    ax.set_xlabel("x (m)"); ax.set_yticks([])
    ax.set_title("② Structural Feasibility\n"
                 r"$L_{\rm fuse} - x_{\rm TE,max} \geq 0$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="x")

    # ── Panel 3: area ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    def _area(sc):
        chds = np.array(_chords) * sc
        spns = np.concatenate([[0], np.cumsum(np.array(_spans) * sc)])
        a = sum(0.5*(chds[i]+chds[i+1])*(spns[i+1]-spns[i]) for i in range(len(chds)-1))
        return 2*a
    target = _area(1.0)
    scales = [0.72, 0.86, 1.0, 1.12]
    areas  = [_area(s) for s in scales]
    colors = ["tomato" if a < target else "steelblue" for a in areas]
    bars = ax.bar([f"{s:.2f}×" for s in scales], areas, color=colors, ec="k", width=0.55)
    ax.axhline(target, color="red", lw=2, ls="--", label=f"Min S_ref = {target:.0f} m²")
    for bar, a in zip(bars, areas):
        sym = "✓" if a >= target else "✗"
        ax.text(bar.get_x() + bar.get_width()/2, a + 20, sym,
                ha="center", fontsize=14,
                color="darkgreen" if a >= target else "darkred")
    ax.set_ylabel("Planform Area (m²)"); ax.set_xlabel("Wing scale factor")
    ax.set_title("③ Minimum Reference Area\n"
                 r"$(S_{\rm ref} - S_{\rm baseline})/S_{\rm baseline} \geq 0$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(areas) * 1.15)

    # ── Panel 4: CL ───────────────────────────────────────────────────────────
    ax = axes[1, 1]
    alpha_arr = np.linspace(-2, 18, 300)
    CL_arr = 0.05 * alpha_arr * (1 - (alpha_arr/40)**2)
    ax.plot(alpha_arr, CL_arr, "b-", lw=2, label="CL(α)  [illustrative]")
    ax.axhline(CL_MIN, color="red", lw=2, ls="--", label=f"CL_min = {CL_MIN}")
    ax.axvline(ALPHA_DESIGN, color="green", lw=1.8, ls=":", label=f"Design α = {ALPHA_DESIGN}°")
    ax.fill_between(alpha_arr, CL_MIN, CL_arr,
                    where=CL_arr >= CL_MIN, color="green", alpha=0.15, label="Feasible")
    ax.fill_between(alpha_arr, CL_MIN, CL_arr,
                    where=CL_arr < CL_MIN, color="red",   alpha=0.10, label="Infeasible")
    CL_d = 0.05 * ALPHA_DESIGN * (1 - (ALPHA_DESIGN/40)**2)
    ax.plot(ALPHA_DESIGN, CL_d, "go", ms=9, zorder=6, label=f"CL ≈ {CL_d:.3f}")
    ax.set_xlabel("Angle of Attack α (°)"); ax.set_ylabel("CL")
    ax.set_title(f"④ Minimum Lift Coefficient\n"
                 fr"$C_L \geq C_{{L,\min}} = {CL_MIN}$  at M={MACH_DESIGN}, α={ALPHA_DESIGN}°")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 18); ax.set_ylim(-0.05, 0.65)

    fig.suptitle("Aerodynamic Shape Optimisation — Constraint Overview",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, "constraint_summary.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating constraint visualisations -> {OUT_DIR}/\n")
    print(f"  Baseline values used:")
    print(f"    Half-span        = {HALF_SPAN:.2f} m")
    print(f"    Max trailing edge= {MAX_TE:.2f} m  (fuselage = {TOTAL_LENGTH} m)")
    print(f"    Shock margin     = {SHOCK_MARGIN:.2f} m")
    print(f"    Struct margin    = {STRUCT_MARGIN:.2f} m")
    print()
    plot_shock_and_structures_constraint()
    plot_area_constraint()
    plot_cl_constraint()
    plot_parameter_bounds()
    plot_summary()
    print(f"\nDone. Open  {OUT_DIR}/  to view the figures.")
