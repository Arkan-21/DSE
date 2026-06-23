"""
sensitivity_oat.py
==================
One-at-a-time sensitivity analysis for ramjet_fixedgeometry.py.
Each graph saved as a separate PNG to the desktop.

Parameters varied: φ, Ma0, h0, theta
Outputs tracked:   Fin, Isp, T4, Ma4, Ma6
"""

import sys, os, io, contextlib, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataclasses import replace

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ramjet_fixedgeometry import Geometry, Assumptions, RamHelp

OUTDIR = r"C:\Users\arkan\OneDrive\Desktop"

# ─── Baseline ────────────────────────────────────────────────────────────────
GEOM = Geometry(
    A0=4.5, L01=0.60, L12=0.5, L23=0.42, L34=0.28,
    L45=0.35, L56=1.20, A2=4.05, A3=4.95, A4=4.95, A6=7.2,
)
BASE = Assumptions(
    h0=30_000.0, Ma0=5.0, phi=0.9, theta=90.0,
    mixing_coeff=0.176, Ma_COMB=0.3, Cf=0.003, HHV=141.8e6,
)

# ─── Sweep grid ──────────────────────────────────────────────────────────────
PHI_VALS   = np.array([0.4, 0.5, 0.6, 0.70, 0.80, 0.90])
MA0_VALS   = np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
H0_VALS    = np.array([20e3, 22e3, 24e3, 26e3, 28e3, 30e3, 32e3])
THETA_VALS = np.array([0, 15, 30, 45, 60, 75, 90])

PHI_COLORS = {
    0.40: "#3b0f70",
    0.5: "#8c2981",
    0.60: "#de4968",
    0.70: "#fe9f6d",
    0.80: "#f8c932",
    0.90: "#fcffa4",
}

# ─── Engine runner ────────────────────────────────────────────────────────────
def run(assump, geom=GEOM):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            eng  = RamHelp(geom=geom, assump=assump)
            inp  = eng.station_0()
            iso  = eng.station_1(inp)
            sec2 = eng.section_12(iso)
            sec3 = eng.section_23(sec2)
            sec4 = eng.section_34(sec3)
            sec5 = eng.section_45(sec4)
            sec6 = eng.section_56(sec5)
            perf = eng.performance(inp, sec6, sec3)
        if perf.get("thermal_choke"): return None
        Fin = float(perf["Fin"]); Isp = float(perf["Isp"])
        T4  = float(sec4["T"]);   Ma4 = float(sec4["Ma"]); Ma6 = float(sec6["Ma"])
        import math
        if not all(map(math.isfinite, [Fin, Isp, T4, Ma4, Ma6])): return None
        return {"Fin": Fin, "Isp": Isp, "T4": T4, "Ma4": Ma4, "Ma6": Ma6}
    except:
        return None

# ─── Baseline run ────────────────────────────────────────────────────────────
print("Running baseline...", flush=True)
base_res = run(BASE)
assert base_res, "Baseline failed"
Fin_b = base_res["Fin"]; Isp_b = base_res["Isp"]
print(f"  Baseline  Fin={Fin_b/1e3:.2f} kN   Isp={Isp_b:.1f} s\n")

# ─── Sweep runner ────────────────────────────────────────────────────────────
def run_sweep(param, sweep_x, x_label, phi_is_x=False):
    results = []
    for phi in PHI_VALS:
        for xv in sweep_x:
            if phi_is_x:
                a = replace(BASE, phi=float(xv))
                phi_use = float(xv)
            else:
                a = replace(BASE, phi=phi, **{param: float(xv)})
                phi_use = phi
            r = run(a)
            label = f"{x_label}={xv:.3g}  φ={phi_use:.2f}"
            status = f"Fin={r['Fin']/1e3:.2f} kN  Isp={r['Isp']:.1f} s" if r else "FAILED"
            print(f"  {label:35s}  →  {status}", flush=True)
            if r:
                results.append({
                    "x": float(xv), "phi": phi_use,
                    "Fin": r["Fin"], "Isp": r["Isp"],
                    "T4": r["T4"], "Ma4": r["Ma4"], "Ma6": r["Ma6"],
                })
    return results

print(f"── Sweep: φ ({len(PHI_VALS)} values) ──")
res_phi   = run_sweep("phi",   PHI_VALS,   "φ",   phi_is_x=True)
print(f"\n── Sweep: Ma₀ ({len(MA0_VALS)} values × {len(PHI_VALS)} φ) ──")
res_ma0   = run_sweep("Ma0",   MA0_VALS,   "Ma0")
print(f"\n── Sweep: h₀ ({len(H0_VALS)} values × {len(PHI_VALS)} φ) ──")
res_h0    = run_sweep("h0",    H0_VALS,    "h0")
print(f"\n── Sweep: θ ({len(THETA_VALS)} values × {len(PHI_VALS)} φ) ──")
res_theta = run_sweep("theta", THETA_VALS, "theta")

total = sum(len(r) for r in [res_phi, res_ma0, res_h0, res_theta])
print(f"\nValid cases: {total}\n")

# ─── Plotting helpers ─────────────────────────────────────────────────────────
SCATTER_KW = dict(s=60, edgecolors="none", linewidths=0, zorder=4)
GRID_KW    = dict(color="#cccccc", lw=0.5, alpha=0.8)

def phi_legend(ax, loc="upper left"):
    handles = [Line2D([0],[0], marker="o", color="w",
                      markerfacecolor=PHI_COLORS[round(phi,2)],
                      markersize=7, label=f"φ={phi:.2f}")
               for phi in PHI_VALS]
    ax.legend(handles=handles, loc=loc, fontsize=6.5,
              ncol=2, framealpha=0.85, handletextpad=0.3, columnspacing=0.6)

def scatter_phi(ax, results, xcol, ycol, xscale=1.0):
    for phi in PHI_VALS:
        pts = [r for r in results if round(r["phi"],2) == round(phi,2)]
        if not pts: continue
        xs = np.array([p[xcol]*xscale for p in pts])
        ys = np.array([p[ycol] for p in pts])
        ax.scatter(xs, ys, color=PHI_COLORS[round(phi,2)], **SCATTER_KW)

def save_single(fig, filename):
    path = os.path.join(OUTDIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def make_single(results, xcol, ycol, xlabel, xunit, ylabel, title, filename,
                xscale=1.0, legend_loc="upper left"):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        f"Ma₀={BASE.Ma0}  h₀={BASE.h0/1e3:.0f} km  φ_base={BASE.phi}  θ={BASE.theta:.0f}°",
        fontsize=9, color="#555555",
    )
    scatter_phi(ax, results, xcol, ycol, xscale=xscale)
    ax.set_xlabel(f"{xlabel}  [{xunit}]", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(**GRID_KW)
    phi_legend(ax, loc=legend_loc)
    fig.tight_layout()
    save_single(fig, filename)

# ═══════════════════════════════════════════════════════════════════════════════
# φ sweep — 3 graphs
# ═══════════════════════════════════════════════════════════════════════════════
make_single(res_phi, "T4",  "Fin", "T₄",  "K",   "Thrust  [N]",
            "Combustor Exit Temp vs Thrust  (φ sweep)",
            "phi_T4_vs_Fin.png", legend_loc="upper left")

make_single(res_phi, "Ma6", "Fin", "Ma₆", "—",   "Thrust  [N]",
            "Nozzle Exit Mach vs Thrust  (φ sweep)",
            "phi_Ma6_vs_Fin.png", legend_loc="upper left")

make_single(res_phi, "Ma4", "Isp", "Ma₄", "—",   "Isp  [s]",
            "Combustor Exit Mach vs Isp  (φ sweep)",
            "phi_Ma4_vs_Isp.png", legend_loc="upper left")

# ═══════════════════════════════════════════════════════════════════════════════
# Ma₀ sweep — 6 graphs
# ═══════════════════════════════════════════════════════════════════════════════
make_single(res_ma0, "x",   "Fin", "Ma₀", "—",   "Thrust  [N]",
            "Thrust vs Ma₀  (coloured by φ)",
            "Ma0_vs_Fin.png", legend_loc="upper left")

make_single(res_ma0, "x",   "Isp", "Ma₀", "—",   "Isp  [s]",
            "Isp vs Ma₀  (coloured by φ)",
            "Ma0_vs_Isp.png", legend_loc="upper right")

make_single(res_ma0, "T4",  "Fin", "T₄",  "K",   "Thrust  [N]",
            "Thrust vs T₄  (Ma₀ sweep, coloured by φ)",
            "Ma0_T4_vs_Fin.png", legend_loc="upper left")

make_single(res_ma0, "Ma6", "Fin", "Ma₆", "—",   "Thrust  [N]",
            "Thrust vs Nozzle Exit Mach  (Ma₀ sweep, coloured by φ)",
            "Ma0_Ma6_vs_Fin.png", legend_loc="upper left")

make_single(res_ma0, "Ma4", "Isp", "Ma₄", "—",   "Isp  [s]",
            "Isp vs Combustor Exit Mach  (Ma₀ sweep, coloured by φ)",
            "Ma0_Ma4_vs_Isp.png", legend_loc="upper left")

# Pareto — Ma0 sweep
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(f"Ma₀={BASE.Ma0}  h₀={BASE.h0/1e3:.0f} km  φ_base={BASE.phi}  θ={BASE.theta:.0f}°",
             fontsize=9, color="#555555")
scatter_phi(ax, res_ma0, "Fin", "Isp")
ax.set_xlabel("Thrust  [N]", fontsize=10); ax.set_ylabel("Isp  [s]", fontsize=10)
ax.set_title("Thrust–Isp Pareto  (Ma₀ sweep, coloured by φ)", fontsize=10, fontweight="bold")
ax.grid(**GRID_KW); phi_legend(ax, loc="upper left")
fig.tight_layout(); save_single(fig, "Ma0_Pareto.png")

# ═══════════════════════════════════════════════════════════════════════════════
# h₀ sweep — 6 graphs
# ═══════════════════════════════════════════════════════════════════════════════
make_single(res_h0, "x",   "Fin", "h₀",  "km",  "Thrust  [N]",
            "Thrust vs h₀  (coloured by φ)",
            "h0_vs_Fin.png", xscale=1e-3, legend_loc="upper right")

make_single(res_h0, "x",   "Isp", "h₀",  "km",  "Isp  [s]",
            "Isp vs h₀  (coloured by φ)",
            "h0_vs_Isp.png", xscale=1e-3, legend_loc="upper right")

make_single(res_h0, "T4",  "Fin", "T₄",  "K",   "Thrust  [N]",
            "Thrust vs T₄  (h₀ sweep, coloured by φ)",
            "h0_T4_vs_Fin.png", legend_loc="upper left")

make_single(res_h0, "Ma6", "Fin", "Ma₆", "—",   "Thrust  [N]",
            "Thrust vs Nozzle Exit Mach  (h₀ sweep, coloured by φ)",
            "h0_Ma6_vs_Fin.png", legend_loc="upper left")

make_single(res_h0, "Ma4", "Isp", "Ma₄", "—",   "Isp  [s]",
            "Isp vs Combustor Exit Mach  (h₀ sweep, coloured by φ)",
            "h0_Ma4_vs_Isp.png", legend_loc="upper left")

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(f"Ma₀={BASE.Ma0}  h₀={BASE.h0/1e3:.0f} km  φ_base={BASE.phi}  θ={BASE.theta:.0f}°",
             fontsize=9, color="#555555")
scatter_phi(ax, res_h0, "Fin", "Isp")
ax.set_xlabel("Thrust  [N]", fontsize=10); ax.set_ylabel("Isp  [s]", fontsize=10)
ax.set_title("Thrust–Isp Pareto  (h₀ sweep, coloured by φ)", fontsize=10, fontweight="bold")
ax.grid(**GRID_KW); phi_legend(ax, loc="upper left")
fig.tight_layout(); save_single(fig, "h0_Pareto.png")

# ═══════════════════════════════════════════════════════════════════════════════
# θ sweep — 6 graphs
# ═══════════════════════════════════════════════════════════════════════════════
make_single(res_theta, "x",   "Fin", "θ",   "deg", "Thrust  [N]",
            "Thrust vs θ  (coloured by φ)",
            "theta_vs_Fin.png", legend_loc="upper left")

make_single(res_theta, "x",   "Isp", "θ",   "deg", "Isp  [s]",
            "Isp vs θ  (coloured by φ)",
            "theta_vs_Isp.png", legend_loc="upper left")

make_single(res_theta, "T4",  "Fin", "T₄",  "K",   "Thrust  [N]",
            "Thrust vs T₄  (θ sweep, coloured by φ)",
            "theta_T4_vs_Fin.png", legend_loc="upper left")

make_single(res_theta, "Ma6", "Fin", "Ma₆", "—",   "Thrust  [N]",
            "Thrust vs Nozzle Exit Mach  (θ sweep, coloured by φ)",
            "theta_Ma6_vs_Fin.png", legend_loc="upper left")

make_single(res_theta, "Ma4", "Isp", "Ma₄", "—",   "Isp  [s]",
            "Isp vs Combustor Exit Mach  (θ sweep, coloured by φ)",
            "theta_Ma4_vs_Isp.png", legend_loc="upper left")

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(f"Ma₀={BASE.Ma0}  h₀={BASE.h0/1e3:.0f} km  φ_base={BASE.phi}  θ={BASE.theta:.0f}°",
             fontsize=9, color="#555555")
scatter_phi(ax, res_theta, "Fin", "Isp")
ax.set_xlabel("Thrust  [N]", fontsize=10); ax.set_ylabel("Isp  [s]", fontsize=10)
ax.set_title("Thrust–Isp Pareto  (θ sweep, coloured by φ)", fontsize=10, fontweight="bold")
ax.grid(**GRID_KW); phi_legend(ax, loc="upper left")
fig.tight_layout(); save_single(fig, "theta_Pareto.png")

print("\nAll done.")