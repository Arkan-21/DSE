# ═══════════════════════════════════════════════════════════════════════════════
# Peak combustor temperature T₄ — sensitivity to φ, Ma₀ and h₀
# Colour encodes φ, consistent with every other plot in ramsens.py.
# Drop into ramsens.py before the final print(), or run standalone.
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if "res_phi" not in dir():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ramsens import (
        BASE, OUTDIR,
        PHI_VALS, PHI_COLORS, MA0_VALS, H0_VALS,
        run_sweep,
    )
    print("── Re-running sweeps for standalone mode ──")
    res_phi = run_sweep("phi", PHI_VALS, "φ", phi_is_x=True)
    res_ma0 = run_sweep("Ma0", MA0_VALS, "Ma0")
    res_h0  = run_sweep("h0",  H0_VALS,  "h0")

SCATTER_KW = dict(s=72, edgecolors="k", linewidths=0.3, zorder=4)
GRID_KW    = dict(color="#cccccc", lw=0.5, alpha=0.8)

def scatter_by_phi(ax, results, xcol, ycol, xscale=1.0):
    """Scatter plot with one colour per φ value, matching PHI_COLORS."""
    for phi in PHI_VALS:
        pts = [r for r in results if round(r["phi"], 2) == round(phi, 2)]
        if not pts:
            continue
        xs = np.array([p[xcol] * xscale for p in pts])
        ys = np.array([p[ycol]           for p in pts])
        ax.scatter(xs, ys, color=PHI_COLORS[round(phi, 2)], **SCATTER_KW)

def phi_legend(ax, loc="upper left"):
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PHI_COLORS[round(phi, 2)],
               markeredgecolor="k", markeredgewidth=0.3,
               markersize=7, label=f"φ = {phi:.2f}")
        for phi in PHI_VALS
    ]
    ax.legend(handles=handles, loc=loc, fontsize=7,
              ncol=2, framealpha=0.85, handletextpad=0.3, columnspacing=0.6)

# ── Figure layout: 3 panels + narrow colourbar column ────────────────────────
fig = plt.figure(figsize=(14, 4.5))
fig.suptitle(
    r"Peak combustor temperature  $T_4$  [K]  —  sensitivity to"
    r"  $\varphi$,  $\mathrm{Ma}_0$  and  $h_0$",
    fontsize=11, fontweight="bold", y=1.02,
)
gs   = fig.add_gridspec(1, 3, wspace=0.36)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

# ── (a) φ sweep: T₄ vs φ (x-axis IS φ, so each dot is its own φ colour) ─────
# res_phi has phi == x when phi_is_x=True, so xcol="x" == phi
scatter_by_phi(ax_a, res_phi, xcol="x", ycol="T4")
ax_a.axvline(BASE.phi, color="0.4", lw=1.0, linestyle=":", zorder=2)
ax_a.set_xlabel(r"Equivalence ratio  $\varphi$  [—]", fontsize=10)
ax_a.set_ylabel(r"$T_4$  [K]", fontsize=10)
ax_a.set_title(r"(a)  $\varphi$ sweep", fontsize=10, fontweight="bold")
ax_a.grid(**GRID_KW)
phi_legend(ax_a, loc="upper left")

# ── (b) Ma₀ sweep: T₄ vs Ma₀, one line/colour per φ ─────────────────────────
scatter_by_phi(ax_b, res_ma0, xcol="x", ycol="T4")
# Connect dots of the same φ with a thin line so trends read clearly
for phi in PHI_VALS:
    pts = sorted([r for r in res_ma0 if round(r["phi"], 2) == round(phi, 2)],
                 key=lambda r: r["x"])
    if len(pts) < 2:
        continue
    xs = [p["x"]  for p in pts]
    ys = [p["T4"] for p in pts]
    ax_b.plot(xs, ys, color=PHI_COLORS[round(phi, 2)],
              lw=0.9, alpha=0.6, zorder=3)
ax_b.axvline(BASE.Ma0, color="0.4", lw=1.0, linestyle=":", zorder=2)
ax_b.set_xlabel(r"Freestream Mach  $\mathrm{Ma}_0$  [—]", fontsize=10)
ax_b.set_ylabel(r"$T_4$  [K]", fontsize=10)
ax_b.set_title(r"(b)  $\mathrm{Ma}_0$ sweep", fontsize=10, fontweight="bold")
ax_b.grid(**GRID_KW)
phi_legend(ax_b, loc="upper left")

# ── (c) h₀ sweep: T₄ vs altitude, one line/colour per φ ─────────────────────
scatter_by_phi(ax_c, res_h0, xcol="x", ycol="T4", xscale=1e-3)
for phi in PHI_VALS:
    pts = sorted([r for r in res_h0 if round(r["phi"], 2) == round(phi, 2)],
                 key=lambda r: r["x"])
    if len(pts) < 2:
        continue
    xs = [p["x"] * 1e-3 for p in pts]
    ys = [p["T4"]        for p in pts]
    ax_c.plot(xs, ys, color=PHI_COLORS[round(phi, 2)],
              lw=0.9, alpha=0.6, zorder=3)
ax_c.axvline(BASE.h0 * 1e-3, color="0.4", lw=1.0, linestyle=":", zorder=2)
ax_c.set_xlabel(r"Cruise altitude  $h_0$  [km]", fontsize=10)
ax_c.set_ylabel(r"$T_4$  [K]", fontsize=10)
ax_c.set_title(r"(c)  $h_0$ sweep", fontsize=10, fontweight="bold")
ax_c.grid(**GRID_KW)
phi_legend(ax_c, loc="upper right")

fig.tight_layout(rect=[0, 0, 1, 1])
path = os.path.join(OUTDIR, "T4_vs_phi_Ma0_h0.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")