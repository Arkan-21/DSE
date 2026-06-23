import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from stl import mesh

BASE = r"C:\Users\Maria\Documents\DSE\DSE\Final_analysis_optimization"

GEOMETRIES = [
    #("Sensitivity Optimum", BASE + r"\sensitivity_optimum.stl"),
    ("Optimum",    BASE + r"\low_cl_optimum.stl"),
    #("Sensitivity seed",    BASE + r"\initial_sensitivity.stl"),
    ("Initial guess",            BASE + r"\initial_geometry.stl"),
    #("Midterm",            BASE + r"\midterm_geometry.stl"),
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
LINESTYLES = ["-", ":", "-.", ":"]


def outer_contours_2d(vectors_2d, resolution=1000):
    """
    Rasterize all projected triangles to a bitmap, then trace the outer
    boundary with matplotlib contour.  Works regardless of mesh topology
    or degenerate triangles — nothing inside the hull shows up.
    """
    all_pts = vectors_2d.reshape(-1, 2)
    x0, y0 = all_pts.min(axis=0)
    x1, y1 = all_pts.max(axis=0)
    pad = max(x1 - x0, y1 - y0) * 0.03
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad

    aspect = (x1 - x0) / (y1 - y0)
    w_px = resolution
    h_px = max(1, int(resolution / aspect))

    # Render filled triangles to an offscreen bitmap
    fig_r = plt.figure(figsize=(w_px / 100.0, h_px / 100.0), dpi=100)
    ax_r = fig_r.add_axes([0, 0, 1, 1])
    ax_r.set_xlim(x0, x1)
    ax_r.set_ylim(y0, y1)
    ax_r.axis("off")
    ax_r.patch.set_facecolor("black")
    fig_r.patch.set_facecolor("black")
    ax_r.add_collection(
        PolyCollection(vectors_2d, facecolors="white", edgecolors="white", linewidths=0.1)
    )
    fig_r.canvas.draw()

    w_act, h_act = fig_r.canvas.get_width_height()
    buf = np.asarray(fig_r.canvas.buffer_rgba())[:, :, 0].astype(float)
    buf = buf[::-1]  # flip: row 0 → y0, last row → y1
    plt.close(fig_r)

    # Trace boundary between filled (255) and empty (0)
    x_arr = np.linspace(x0, x1, w_act)
    y_arr = np.linspace(y0, y1, h_act)

    fig_c, ax_c = plt.subplots()
    cs = ax_c.contour(x_arr, y_arr, buf, levels=[128])
    contours = [p.vertices for p in cs.get_paths()]
    plt.close(fig_c)

    return contours


fig, ax = plt.subplots(figsize=(10, 6))

for (label, path), color in zip(GEOMETRIES, COLORS):
    try:
        m = mesh.Mesh.from_file(path)
    except Exception as e:
        print(f"Could not load {label}: {e}")
        continue

    first = True
    for contour in outer_contours_2d(m.vectors[:, :, :2]):
        ax.plot(contour[:, 0], contour[:, 1],
                color=color, linewidth=2, linestyle=LINESTYLES[GEOMETRIES.index((label, path)) % len(LINESTYLES)],
                label=label if first else "_nolegend_")
        first = False

ax.autoscale()
ax.set_aspect("equal")
ax.set_xlabel("X [m]", fontsize=11)
ax.set_ylabel("Y [m]", fontsize=11)
ax.legend(fontsize=18, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.35)
ax.set_title("Top-view contours(X–Y)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("geometry_projections.png", dpi=150, bbox_inches="tight")
plt.show()
