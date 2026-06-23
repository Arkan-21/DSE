from geometry_generator import does_it_fly, WEIGHT
import matplotlib.pyplot as plt

from plot_projections import GEOMETRIES

BASE = r"C:\Users\Maria\Documents\DSE\DSE\Final_analysis_optimization"

GEOMETRIES_seed = [
    (f"{BASE}\\sensitivity_optimum.stl",    "Sensitivity Optimum",   "o"),
    (f"{BASE}\\initial_sensitivity.stl",     "Sensitivity seed",      "x"),
    (f"{BASE}\\low_cl_optimum.stl",         "Optimum",                "x"),
    (f"{BASE}\\initial_geometry.stl",       "Seed",               "+"),
    (f"{BASE}\\midterm_geometry.stl",       "Midterm",               "*"),
]
GEOMETRIES_presentation = [
    (f"{BASE}\\low_cl_optimum.stl",         "Optimum",                "x"),
    (f"{BASE}\\initial_geometry.stl",       "Initial guess",               "+"),
    (f"{BASE}\\midterm_geometry.stl",       "Midterm",               "*"),
]
GEOMETRIES_thickness = [
    (f"{BASE}\\3aoa_naca4_refcl_optimum.stl",     "NACA0004",      "o"),
    (f"{BASE}\\3aoa_naca6_optimum.stl",         "NACA0006",                "x"),
]

GEOMETRIES_aoa = [
    (f"{BASE}\\3aoa_naca4_refcl_optimum.stl",     "AoA = 3",      "o"),
    (f"{BASE}\\6aoa_naca4_optimum.stl",         "AoA = 6",                "x"),
]


GEOMETRIES = GEOMETRIES_presentation  # change this to switch which set of geometries to analyze
alphas = list(range(10))

results = {path: {"L": [], "D": [], "LoD": []} for path, _, _ in GEOMETRIES}

for alpha in alphas:
    for path, _, _ in GEOMETRIES:
        L, answer, diff, D, CL = does_it_fly(path, alpha)
        results[path]["L"].append(CL)
        results[path]["D"].append(D)
        results[path]["LoD"].append(CL / D if D != 0 else float("inf"))

fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=True)

plot_keys  = ["L",                "D",           "LoD"]
titles     = ["Lift Coefficient vs. Angle of Attack",
              "Drag Coefficient vs. Angle of Attack",
              "Lift-to-Drag Ratio vs. Angle of Attack"]
ylabels    = ["Lift Coefficient (C_L)", "Drag Coefficient (C_D)", "Lift-to-Drag Ratio (L/D)"]


for ax, key, title, ylabel in zip(axes, plot_keys, titles, ylabels):
    for path, label, marker in GEOMETRIES:
        ax.plot(alphas, results[path][key], label=label, marker=marker)
    ax.set_title(title)
    ax.set_xlabel("Angle of Attack (alpha) [deg]")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=16, loc="best")

plt.tight_layout()
plt.show()
'''

fig, ax = plt.subplots(figsize=(8, 6))
plot_keys  = [     "LoD"]
titles     = [
              "Lift-to-Drag Ratio vs. Angle of Attack"]
ylabels    = ["Lift-to-Drag Ratio (L/D)"]

for path, label, marker in GEOMETRIES:
    ax.plot(alphas, results[path][plot_keys[0]], label=label, marker=marker, linewidth=2)
ax.set_title(titles[0])
ax.set_xlabel("Angle of Attack (alpha) [deg]")
ax.set_ylabel(ylabels[0])
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend(fontsize=16, loc="lower right")

plt.tight_layout()
plt.show()
'''

last_path, last_label, _ = GEOMETRIES[0]
last_L, last_answer, last_diff, *_ = does_it_fly(last_path, alphas[-1])
print(f"Final alpha={alphas[-1]} for {last_label} -> Lift: {last_L:.2f}, "
      f"Does it fly: {last_answer}, excess lift {last_diff / WEIGHT * 100:.2f}% of total weight")
