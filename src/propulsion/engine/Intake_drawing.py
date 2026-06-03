import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch




# --- restructured-project import bootstrap ---
from pathlib import Path as _DSE_Path
import sys as _DSE_sys
_DSE_ROOT = next((p for p in _DSE_Path(__file__).resolve().parents if (p / "src").exists() and (p / "data").exists()), None)
if _DSE_ROOT is not None:
    for _DSE_p in [
        _DSE_ROOT / "src",
        _DSE_ROOT / "src" / "common",
        _DSE_ROOT / "src" / "aerodynamics" / "drag",
        _DSE_ROOT / "src" / "propulsion",
        _DSE_ROOT / "src" / "propulsion" / "engine",
        _DSE_ROOT / "src" / "thermal",
        _DSE_ROOT / "src" / "sizing",
        _DSE_ROOT / "src" / "tanks",
        _DSE_ROOT / "src" / "environment",
        _DSE_ROOT / "src" / "trade_offs",
        _DSE_ROOT / "external",
        _DSE_ROOT / "external" / "pycycle_examples",
    ]:
        if _DSE_p.exists() and str(_DSE_p) not in _DSE_sys.path:
            _DSE_sys.path.insert(0, str(_DSE_p))
# --- end bootstrap ---

from .inlet_shock_ramjet import analyse_intake4  # for quick demo at the end

def draw_intake(
    # ---------- ramp geometry ----------
    L_1: float,
    L_2: float,
    theta_1_deg: float,
    theta_2_deg: float,
    # ---------- cowl geometry ----------
    y_cowl: float,
    L_cowl_upstream: float  = None,   # how far the cowl extends upstream of lip; default = 0.3*L_1
    L_cowl_downstream: float = None,  # how far the cowl extends downstream of lip; default = 0.4*L_2
    t_cowl: float           = 0.02,   # cowl lip thickness (in same units as L_1)
    # ---------- shock angles (deg, measured from x-axis in global frame) ----------
    beta_1_deg: float       = None,   # shock 1: from origin, angle from x-axis
    beta_2_deg: float       = None,   # shock 2 in global frame = theta_1 + beta_2_local
    beta_ref_deg: float     = None,   # reflected shock angle from x-axis (downward)
    # ---------- optional: pass the full results dict instead of individual angles ----------
    results: dict           = None,
    # ---------- display options ----------
    show_shocks: bool       = True,
    show_labels: bool       = True,
    show_mach_labels: bool  = True,
    mach_values: dict       = None,   # e.g. {"M1": 2.5, "M2": 2.1, "M3": 1.8, "M4": 1.5}
    figsize: tuple          = (12, 6),
    title: str              = None,
    ax: plt.Axes            = None,   # pass an existing Axes to embed in a larger figure
) -> plt.Figure:
    """
    Draw the 2-ramp mixed-compression intake.

    Parameters
    ----------
    L_1, L_2          : ramp lengths (same units throughout)
    theta_1_deg       : ramp 1 deflection angle [deg]
    theta_2_deg       : ramp 2 deflection angle [deg]
    y_cowl            : cowl lip height above the ramp-1 baseline [same units]
    L_cowl_upstream   : how far the cowl plate extends upstream from the lip
    L_cowl_downstream : how far the cowl plate extends downstream from the lip
    t_cowl            : thickness of the cowl lip block
    beta_1_deg        : shock 1 angle from x-axis [deg]  (needed to draw shocks)
    beta_2_deg        : shock 2 angle in global frame = theta_1 + beta_2_local [deg]
    beta_ref_deg      : reflected shock angle from x-axis, measured downward [deg]
    results           : if provided, all beta/theta/Mach values are extracted from it
                        (overrides individual beta_* and mach_values arguments)
    show_shocks       : draw the three shock lines
    show_labels       : annotate key points and angles
    show_mach_labels  : show Mach number in each flow region
    mach_values       : dict with keys M_inf, M2, M3, M4
    figsize           : (width, height) in inches
    title             : figure title (auto-generated if None)
    ax                : existing Axes to draw into

    Returns
    -------
    matplotlib Figure object
    """

    # ------------------------------------------------------------------
    # 0.  Unpack results dict if provided
    # ------------------------------------------------------------------
    if results is not None:
        thetas      = results["thetas_deg"]
        theta_1_deg = thetas[0]
        theta_2_deg = thetas[1]
        beta_1_deg  = results["stages"][0]["beta_deg"]
        # shock 2 global frame angle = theta_1 + beta_2_local
        beta_2_deg  = theta_1_deg + results["stages"][1]["beta_deg"]
        beta_ref_deg = results["beta_reflected_deg"]
        if show_mach_labels:
            mach_values = {
                "M_inf": results["M_inf"],
                "M2":    results["stages"][0]["M_out"],
                "M3":    results["stages"][1]["M_out"],
                "M4":    results["M_exit"],
            }

    # ------------------------------------------------------------------
    # 1.  Core geometry — key points
    # ------------------------------------------------------------------
    t1  = np.radians(theta_1_deg)
    t12 = np.radians(theta_1_deg + theta_2_deg)

    # Ramp corners
    O          = np.array([0.0, 0.0])                        # ramp-1 leading edge
    tip1 = np.array([L_1, -L_1 * np.tan(t1)])
    tip2 = np.array([L_1 + L_2,-L_1 * np.tan(t1) - L_2 * np.tan(t12)])

    # Cowl lip location
    # x_c comes from the intersection of shock 1 and shock 2 lines
    if beta_1_deg is not None and beta_2_deg is not None:
        tan_b1 = np.tan(np.radians(beta_1_deg))
        tan_s2 = np.tan(np.radians(beta_2_deg))
        tan_t1 = np.tan(t1)
        x_c    = L_1 * (tan_t1 - tan_s2) / (tan_b1 - tan_s2)
    else:
        # fallback: place cowl lip directly above ramp-2 tip x
        x_c = L_1 + L_2

    cowl_lip = np.array([x_c, y_cowl])

    # Cowl plate extents
    if L_cowl_upstream is None:
        L_cowl_upstream = 0.30 * L_1
    if L_cowl_downstream is None:
        L_cowl_downstream = 0.40 * L_2

    cowl_front = np.array([x_c, y_cowl])          # lip
    cowl_rear  = np.array([x_c + L_cowl_downstream, y_cowl])        

    # ------------------------------------------------------------------
    # 2.  Set up figure
    # ------------------------------------------------------------------
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # ------------------------------------------------------------------
    # 3.  Ramp body  (solid filled polygon)
    #     Outline: O → tip1 → tip2 → below tip2 → below O → close
    # ------------------------------------------------------------------
    body_depth = max(L_1, L_2) * 0.15
    ramp_body_x = [O[0],    tip1[0], tip2[0], tip2[0],       O[0]]
    ramp_body_y = [O[1], tip1[1], tip2[1],  tip2[1] + body_depth, O[1] + body_depth]
    ax.fill(ramp_body_x, ramp_body_y, color="#b0b8c8", zorder=2, label="_nolegend_")
    ax.plot(ramp_body_x[:3], ramp_body_y[:3],
            color="#2c3e50", linewidth=2.0, zorder=3)   # ramp surface only

    # ------------------------------------------------------------------
    # 4.  Cowl plate  (filled rectangle with a pointed lip)
    # ------------------------------------------------------------------
    # Top face of cowl
    cowl_top_y = y_cowl - t_cowl
    cowl_xs = [cowl_front[0], cowl_rear[0], cowl_rear[0], cowl_front[0]]
    cowl_ys = [ y_cowl, y_cowl,cowl_top_y,cowl_top_y]
    ax.fill(cowl_xs, cowl_ys, color="#b0b8c8", zorder=2, label="_nolegend_")
    ax.plot([cowl_front[0], cowl_rear[0]], [y_cowl, y_cowl],
            color="#2c3e50", linewidth=2.0, zorder=3)   # inner face
    ax.plot([cowl_front[0], cowl_rear[0]], [cowl_top_y, cowl_top_y],
            color="#2c3e50", linewidth=1.2, zorder=3, linestyle="--")  # outer face
    # Lip vertical edge
    ax.plot([cowl_lip[0], cowl_lip[0]], [y_cowl, cowl_top_y],
            color="#2c3e50", linewidth=2.0, zorder=3)

    # ------------------------------------------------------------------
    # 5.  Shocks
    # ------------------------------------------------------------------
    if show_shocks and beta_1_deg is not None:

        shock_kw  = dict(zorder=4, linewidth=1.8, linestyle="-")

        # --- Shock 1 : origin → cowl lip ---
        ax.plot([O[0], cowl_lip[0]], [O[1], cowl_lip[1]],
                color="#e74c3c", **shock_kw, label="Shock 1")
        ax.set_facecolor("#d62728")
        # --- Shock 2 : ramp-1 tip → cowl lip ---
        ax.plot([tip1[0], cowl_lip[0]], [tip1[1], cowl_lip[1]],
                color="#e67e22", **shock_kw, label="Shock 2")
        poly_M2 = np.array([ O, tip1, cowl_lip])

        ax.fill(poly_M2[:,0],poly_M2[:,1],color="#ff9933",zorder=1)

        poly_M3 = np.array([tip1,tip2,cowl_lip])

        ax.fill(poly_M3[:,0],poly_M3[:,1],color="#3ddc3d",zorder=1)
        # --- Reflected shock : cowl lip → ramp-2 tip ---
        if beta_ref_deg is not None:
            ax.plot([cowl_lip[0], tip2[0]], [cowl_lip[1], tip2[1]],
                    color="#8e44ad", **shock_kw, label="Reflected shock")

    # ------------------------------------------------------------------
    # 6.  Freestream arrow
    # ------------------------------------------------------------------
    arrow_len = L_cowl_upstream * 0.7
    arrow_y   = (y_cowl + cowl_top_y) / 2.0
    ax.annotate(
        "", xy=(cowl_front[0], arrow_y),
        xytext=(cowl_front[0] - arrow_len, arrow_y),
        arrowprops=dict(arrowstyle="-|>", color="#2980b9", lw=1.8),
        zorder=5,
    )

    # ------------------------------------------------------------------
    # 7.  Angle annotations
    # ------------------------------------------------------------------
    if show_labels:
        lkw = dict(fontsize=8, color="#2c3e50", zorder=6)

        # beta_1 arc at origin
        _draw_angle_arc(ax, O, 0, beta_1_deg or 0,
                        radius=0.12 * L_1, color="#e74c3c",
                        label=f"β₁={beta_1_deg:.1f}°" if beta_1_deg else "")

        # theta_1 at ramp-1 tip
        _draw_angle_arc(ax, tip1, 0, theta_1_deg,
                        radius=0.10 * L_1, color="#27ae60",
                        label=f"θ₁={theta_1_deg:.1f}°")

        # theta_2 at ramp-2 start (same point as tip1, second deflection)
        _draw_angle_arc(ax, tip1, theta_1_deg, theta_1_deg + theta_2_deg,
                        radius=0.17 * L_1, color="#16a085",
                        label=f"θ₂={theta_2_deg:.1f}°")

        # Label key points
        offset = 0.02 * (L_1 + L_2)
        ax.annotate("O (ramp 1 LE)", O, xytext=(O[0] - offset, O[1] - 4 * offset),
                    fontsize=7.5, color="#2c3e50",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8))
        ax.annotate("Ramp 1 tip", tip1,
                    xytext=(tip1[0] - 3 * offset, tip1[1] + 3 * offset),
                    fontsize=7.5, color="#2c3e50",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8))
        ax.annotate("Ramp 2 tip\n(reflected shock target)", tip2,
                    xytext=(tip2[0] + offset, tip2[1] - 4 * offset),
                    fontsize=7.5, color="#8e44ad",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8))
        ax.annotate("Cowl lip", cowl_lip,
                    xytext=(cowl_lip[0] + offset, cowl_lip[1] + 4 * offset),
                    fontsize=7.5, color="#2c3e50",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8))

    # ------------------------------------------------------------------
    # 8.  Mach number labels in each flow region
    # ------------------------------------------------------------------
    if show_mach_labels and mach_values:
        mv = mach_values
        mkw = dict(fontsize=9, ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white",
                             ec="#95a5a6", alpha=0.85))

        # Region 1: freestream, above cowl, upstream of lip
        ax.text(cowl_front[0] - 0.4 * L_cowl_upstream,
                y_cowl + 1.5 * t_cowl,
                f"$M_∞ = {mv.get('M_inf', ''):.3f}$", **mkw)

        # Region 2: between shock 1 and shock 2 — midway between O and tip1, upper half
        mid12_x = 0.55 * x_c
        mid12_y = 0.55 * y_cowl
        ax.text(mid12_x, mid12_y,
                f"$M_2 = {mv.get('M2', ''):.3f}$", **mkw)

        # Region 3: between shock 2 and reflected shock
        mid23_x = (tip1[0] + cowl_lip[0] + tip2[0]) / 3.0
        mid23_y = (tip1[1] + cowl_lip[1] + tip2[1]) / 3.0
        ax.text(mid23_x, mid23_y,
                f"$M_3 = {mv.get('M3', ''):.3f}$", **mkw)

        # Region 4: after reflected shock, near ramp-2 tip
        ax.text(tip2[0] - 0.15 * L_2, tip2[1] + 0.35 * (y_cowl - tip2[1]),
                f"$M_4 = {mv.get('M4', ''):.3f}$", **mkw)

    # ------------------------------------------------------------------
    # 9.  Legend, axes, title
    # ------------------------------------------------------------------
    if show_shocks:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]", fontsize=10)
    ax.set_ylabel("y  [m]", fontsize=10)

    if title is None:
        t_sum = theta_1_deg + theta_2_deg
        title = (f"Mixed-Compression Intake  "
                 f"(θ₁={theta_1_deg:.1f}°, θ₂={theta_2_deg:.1f}°, "
                 f"Σθ={t_sum:.1f}°)")
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Tidy margins
    all_x = [O[0], tip1[0], tip2[0], cowl_front[0], cowl_rear[0]]
    all_y = [O[1] - body_depth, tip2[1], cowl_top_y + t_cowl]
    margin_x = 0.08 * (max(all_x) - min(all_x))
    margin_y = 0.12 * (max(all_y) - min(all_y))
    ax.set_xlim(min(all_x) - margin_x - arrow_len * 1.2,
                max(all_x) + margin_x)
    ax.set_ylim(min(all_y) - margin_y,
                max(all_y) + margin_y)

    if standalone:
        fig.tight_layout()

    return fig




def draw_intake_cfd_style(results, figsize=(14,6)):

    # --------------------------------------------------
    # Extract geometry
    # --------------------------------------------------
    theta1 = results["theta_1_deg"]
    theta2 = results["theta_2_deg"]

    L1 = results["L_1"]
    L2 = results["L_2"]

    x_c, y_c = results["cowl_lip"]

    beta1 = results["beta_1_deg"]
    beta2 = results["beta_2_deg"]
    beta_ref = results["beta_reflected_deg"]

    M_inf = results["M_inf"]
    M2 = results["stages"][0]["M_out"]
    M3 = results["stages"][1]["M_out"]
    M4 = results["M_exit"]

    # --------------------------------------------------
    # Geometry (UPSIDE DOWN)
    # --------------------------------------------------
    t1 = np.radians(theta1)
    t12 = np.radians(theta1 + theta2)

    O = np.array([0.0, 0.0])

    P1 = np.array([
        L1,
        -L1*np.tan(t1)
    ])

    P2 = np.array([
        L1+L2,
        -L1*np.tan(t1)-L2*np.tan(t12)
    ])

    C = np.array([
        x_c,
        -y_c
    ])

    # downstream cowl
    x_cowl_end = P2[0] + 0.6*L2

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # --------------------------------------------------
    # Freestream background
    # --------------------------------------------------
    ax.set_facecolor("#d7191c")

    # --------------------------------------------------
    # M2 region
    # --------------------------------------------------
    region_M2 = np.array([
        O,
        P1,
        C
    ])

    ax.fill(
        region_M2[:,0],
        region_M2[:,1],
        color="#fdae61",
        alpha=0.95,
        zorder=1
    )

    # --------------------------------------------------
    # M3 region
    # --------------------------------------------------
    region_M3 = np.array([
        P1,
        P2,
        C
    ])

    ax.fill(
        region_M3[:,0],
        region_M3[:,1],
        color="#4daf4a",
        alpha=0.95,
        zorder=1
    )

    # --------------------------------------------------
    # M4 region
    # --------------------------------------------------
    region_M4 = np.array([
        C,
        P2,
        [x_cowl_end, P2[1]],
        [x_cowl_end, C[1]]
    ])

    ax.fill(
        region_M4[:,0],
        region_M4[:,1],
        color="#2c7bb6",
        alpha=0.95,
        zorder=1
    )

    # --------------------------------------------------
    # Intake walls
    # --------------------------------------------------

    wall_lw = 5

    ax.plot(
        [O[0],P1[0],P2[0]],
        [O[1],P1[1],P2[1]],
        color="black",
        lw=wall_lw,
        solid_capstyle="round",
        zorder=5
    )

    ax.plot(
        [C[0],x_cowl_end],
        [C[1],C[1]],
        color="black",
        lw=wall_lw,
        solid_capstyle="round",
        zorder=5
    )

    # --------------------------------------------------
    # Shock lines
    # --------------------------------------------------

    shock_color = "#fff176"

    ax.plot(
        [O[0],C[0]],
        [O[1],C[1]],
        color=shock_color,
        lw=3,
        zorder=10
    )

    ax.plot(
        [P1[0],C[0]],
        [P1[1],C[1]],
        color=shock_color,
        lw=3,
        zorder=10
    )

    ax.plot(
        [C[0],P2[0]],
        [C[1],P2[1]],
        color=shock_color,
        lw=3,
        zorder=10
    )

    # --------------------------------------------------
    # Cowl lip marker
    # --------------------------------------------------

    ax.scatter(
        C[0],
        C[1],
        s=80,
        color="black",
        zorder=20
    )

    # --------------------------------------------------
    # Flow direction
    # --------------------------------------------------

    ax.arrow(
        -0.5,
        -0.4*y_c,
        0.35,
        0,
        width=0.01,
        color="black",
        zorder=20
    )

    ax.text(
        -0.45,
        -0.25*y_c,
        r"$M_\infty$",
        fontsize=20
    )

    # --------------------------------------------------
    # Mach labels
    # --------------------------------------------------

    box = dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )

    ax.text(
        0.4*x_c,
        -0.45*y_c,
        f"M∞ = {M_inf:.2f}",
        bbox=box
    )

    ax.text(
        (O[0]+P1[0]+C[0])/3,
        (O[1]+P1[1]+C[1])/3,
        f"M₂ = {M2:.2f}",
        bbox=box
    )

    ax.text(
        (P1[0]+P2[0]+C[0])/3,
        (P1[1]+P2[1]+C[1])/3,
        f"M₃ = {M3:.2f}",
        bbox=box
    )

    ax.text(
        x_c + 0.45*(x_cowl_end-x_c),
        C[1] + 0.5*(P2[1]-C[1]),
        f"M₄ = {M4:.2f}",
        bbox=box
    )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    ax.text(
        C[0]+0.05,
        C[1]+0.15,
        "Cowl Lip",
        fontsize=14,
        weight="bold"
    )

    ax.text(
        0.4*x_c,
        0.5*C[1],
        "Shock 1",
        color="white",
        fontsize=16
    )

    ax.text(
        (P1[0]+C[0])/2,
        (P1[1]+C[1])/2,
        "Shock 2",
        color="white",
        fontsize=16
    )

    ax.text(
        (P2[0]+C[0])/2,
        (P2[1]+C[1])/2,
        "Reflected Shock",
        color="white",
        fontsize=16,
        rotation=55
    )

    # --------------------------------------------------
    # Styling
    # --------------------------------------------------

    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-0.6, x_cowl_end+0.2)

    ax.set_ylim(
        P2[1]-0.3,
        0.5
    )

    plt.tight_layout()

    return fig

# ------------------------------------------------------------------
# Helper: draw a small arc to indicate an angle, with a text label
# ------------------------------------------------------------------
def _draw_angle_arc(ax, vertex, angle_start_deg, angle_end_deg,
                    radius=0.1, color="gray", label=""):
    """Draw an arc from angle_start to angle_end around vertex, plus label."""
    angles = np.linspace(np.radians(angle_start_deg),
                         np.radians(angle_end_deg), 60)
    xs = vertex[0] + radius * np.cos(angles)
    ys = vertex[1] + radius * np.sin(angles)
    ax.plot(xs, ys, color=color, linewidth=1.2, zorder=5)

    if label:
        mid_angle = np.radians((angle_start_deg + angle_end_deg) / 2.0)
        lx = vertex[0] + 1.35 * radius * np.cos(mid_angle)
        ly = vertex[1] + 1.35 * radius * np.sin(mid_angle)
        ax.text(lx, ly, label, fontsize=7.5, color=color,
                ha="center", va="center", zorder=6)


# ------------------------------------------------------------------
# Quick demo  (runs when the file is executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    results = analyse_intake4(
    M_inf=3,
    L_1=1.4,
    theta_1_deg=10.0,
    y_cowl=1.3,
    delta_cowl_deg=4.0,
    verbose=False
)

fig = draw_intake_cfd_style(results)
plt.show()
