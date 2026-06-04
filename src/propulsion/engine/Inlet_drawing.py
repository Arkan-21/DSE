import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from Inlet_shock_ramjet import analyse_intake4  # for quick demo at the end


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

def draw_intake_cfd_style(results, figsize=(15,6), mirror=False):

    # --------------------------------------------------
    # Extract geometry & Mach numbers
    # --------------------------------------------------
    theta1 = results["theta_1_deg"]
    theta2 = results["theta_2_deg"]

    L1 = results["L_1"]
    L2 = results["L_2"]

    x_c, y_c = results["cowl_lip"]

    M_inf = results["M_inf"]
    M2 = results["stages"][0]["M_out"]
    M3 = results["stages"][1]["M_out"]
    M4 = results["M_exit"]

    # --------------------------------------------------
    # Geometry Mapping (UPSIDE DOWN natively, inverted if mirror=True)
    # --------------------------------------------------
    t1 = np.radians(theta1)
    t12 = np.radians(theta1 + theta2)

    # Inversion multiplier for the Y-coordinates
    flip = -1 if mirror else 1

    O = np.array([0.0, 0.0])

    P1 = np.array([
        L1,
        -L1*np.tan(t1) * flip
    ])

    P2 = np.array([
        L1+L2,
        (-L1*np.tan(t1) - L2*np.tan(t12)) * flip
    ])

    C = np.array([
        x_c,
        -y_c * flip
    ])

    # --------------------------------------------------
    # CRITICAL: Dynamic Scaling Factor Calculation
    # --------------------------------------------------
    # Calculate a scaling factor based on the actual height of the intake
    # This prevents text overlaps and arrow explosions regardless of your units.
    y_geom_min = min(O[1], P1[1], P2[1], C[1])
    y_geom_max = max(O[1], P1[1], P2[1], C[1])
    s_factor = max(abs(y_geom_max - y_geom_min), 1e-5) # Characteristic geometric scale

    # Scale the frame sizing parameters relative to the size of the device
    x_cowl_end = P2[0] + 0.6 * L2
    x_start = -0.35 * (L1 + L2)  # Relative to overall ramp length

    # --------------------------------------------------
    # Figure, Gridspec & Color Gradient Setup
    # --------------------------------------------------
    fig, (ax, ax_cbar) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [25, 1]})
    
    # Custom continuous palette mapping: Blue (M4) -> Green (M3) -> Yellow (M2) -> Red (M_inf)
    colors = ["#1976d2", "#388e3c", "#fbc02d", "#d32f2f"]
    cmap_name = "mach_gradient"
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    
    mach_min, mach_max = min(M_inf, M2, M3, M4), max(M_inf, M2, M3, M4)
    norm = mcolors.Normalize(vmin=mach_min, vmax=mach_max)

    bg_gray     = "#e0e0e0"  
    shock_color = "#ffffff"  

    ax.set_facecolor(bg_gray)

    # --------------------------------------------------
    # Smart Layout Scaling Calculations
    # --------------------------------------------------
    if mirror:
        y_floor = y_geom_max + 1.2 * s_factor
        y_top_limit = y_geom_min - 0.4 * s_factor
        y_bot_limit = y_floor
    else:
        y_floor = y_geom_min - 1.2 * s_factor
        y_top_limit = y_geom_max + 0.4 * s_factor
        y_bot_limit = y_floor

    # --------------------------------------------------
    # Flow Domain Plots (Mapped directly to continuous palette)
    # --------------------------------------------------
    region_inf = np.array([[x_start, O[1]], O, C, [x_cowl_end, C[1]], [x_cowl_end, y_floor], [x_start, y_floor]])
    ax.fill(region_inf[:,0], region_inf[:,1], color=cmap(norm(M_inf)), zorder=1)

    region_M2 = np.array([O, P1, C])
    ax.fill(region_M2[:,0], region_M2[:,1], color=cmap(norm(M2)), zorder=1)

    region_M3 = np.array([P1, P2, C])
    ax.fill(region_M3[:,0], region_M3[:,1], color=cmap(norm(M3)), zorder=1)

    region_M4 = np.array([C, P2, [x_cowl_end, P2[1]], [x_cowl_end, C[1]]])
    ax.fill(region_M4[:,0], region_M4[:,1], color=cmap(norm(M4)), zorder=1)

    # --------------------------------------------------
    # Structural Intake Walls
    # --------------------------------------------------
    wall_lw = 5.5
    wall_color = "#1a1a1a"

    ax.plot([x_start, O[0]], [O[1], O[1]], color=wall_color, lw=wall_lw, solid_capstyle="round", zorder=5)
    ax.plot([O[0],P1[0],P2[0]], [O[1],P1[1],P2[1]], color=wall_color, lw=wall_lw, solid_capstyle="round", zorder=5)
    ax.plot([P2[0], x_cowl_end], [P2[1], P2[1]], color=wall_color, lw=wall_lw, solid_capstyle="round", zorder=5)
    ax.plot([C[0], x_cowl_end], [C[1], C[1]], color=wall_color, lw=wall_lw, solid_capstyle="round", zorder=5)

    # --------------------------------------------------
    # Shock Wave Discontinuities
    # --------------------------------------------------
    ax.plot([O[0],C[0]], [O[1],C[1]], color=shock_color, linestyle="--", lw=2.5, zorder=10)
    ax.plot([P1[0],C[0]], [P1[1],C[1]], color=shock_color, linestyle="--", lw=2.5, zorder=10)
    ax.plot([C[0],P2[0]], [C[1],P2[1]], color=shock_color, linestyle="--", lw=2.5, zorder=10)

    # Cowl lip point marker
    ax.scatter(C[0], C[1], s=100, color=wall_color, edgecolors="white", linewidths=1.5, zorder=20)

    # --------------------------------------------------
    # Flow Direction Arrow (Scales dynamically)
    # --------------------------------------------------
    arrow_x = x_start + 0.25 * abs(x_start)
    arrow_w = 0.08 * s_factor
    arrow_hw = 0.25 * s_factor
    
    ax.arrow(arrow_x, -0.6*y_c * flip, abs(x_start)*0.35, 0, width=arrow_w, head_width=arrow_hw, color="black", zorder=20)
    ax.text(arrow_x + abs(x_start)*0.05, -0.3*y_c * flip if not mirror else -0.8*y_c * flip, 
            "Flow", fontsize=14, weight="bold", color="black", zorder=25)

    # --------------------------------------------------
    # Dynamically Aligned Line Labels (Parallel & Auto-Scaled Offsets)
    # --------------------------------------------------
    def get_line_angle_deg(pt1, pt2):
        return np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))

    # Shock 1 Label
    rot_s1 = get_line_angle_deg(O, C)
    ax.text(
        0.45 * C[0],
        (0.45 * C[1]) - (0.18 * s_factor * flip),
        "Shock 1",
        color="black",
        fontsize=10,
        weight="bold",
        rotation=rot_s1,
        rotation_mode='anchor',
        ha='center',
        va='top' if not mirror else 'bottom',
        zorder=25
    )

    # Shock 2 Label
    rot_s2 = get_line_angle_deg(P1, C)
    ax.text(
        ((P1[0] + C[0]) / 2) + 0.02 * (L1+L2),
        ((P1[1] + C[1]) / 2) - (0.15 * s_factor * flip),
        "Shock 2",
        color="black",
        fontsize=10,
        weight="bold",
        rotation=rot_s2,
        rotation_mode='anchor',
        ha='center',
        va='top' if not mirror else 'bottom',
        zorder=25
    )

    # Reflected Shock Label
    rot_ref = get_line_angle_deg(C, P2)
    ax.text(
        ((P2[0] + C[0]) / 2) - 0.02 * (L1+L2),
        ((P2[1] + C[1]) / 2) + (0.12 * s_factor * flip),
        "Reflected Shock",
        color="black",
        fontsize=10,
        weight="bold",
        rotation=rot_ref,
        rotation_mode='anchor',
        ha='center',
        va='bottom' if not mirror else 'top',
        zorder=25
    )

    # --------------------------------------------------
    # Mechanical Dimensions (Auto-scaled spacing)
    # --------------------------------------------------
    dim_color = "#333333"
    line_y_offset = 0.15 * s_factor * flip
    ext_y_offset = 0.25 * s_factor * flip
    text_y_offset = 0.32 * s_factor * flip
    
    # L_1 Dimension
    ax.plot([O[0], O[0]], [O[1], O[1] + ext_y_offset], color=dim_color, lw=1, alpha=0.5)
    ax.plot([P1[0], P1[0]], [P1[1], O[1] + ext_y_offset], color=dim_color, lw=1, alpha=0.5)
    ax.annotate('', xy=(O[0], O[1] + line_y_offset), xytext=(P1[0], O[1] + line_y_offset),
                arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1))
    ax.text(L1/2, O[1] + text_y_offset, f"$L_1$ = {L1:.2f}", color=dim_color, 
            ha='center', va='bottom' if not mirror else 'top', fontsize=11, weight='bold')

    # L_2 Dimension
    ax.plot([P2[0], P2[0]], [P2[1], O[1] + ext_y_offset], color=dim_color, lw=1, alpha=0.5)
    ax.annotate('', xy=(P1[0], O[1] + line_y_offset), xytext=(P2[0], O[1] + line_y_offset),
                arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1))
    ax.text(L1 + L2/2, O[1] + text_y_offset, f"$L_2$ = {L2:.2f}", color=dim_color, 
            ha='center', va='bottom' if not mirror else 'top', fontsize=11, weight='bold')

    # Component Height Profile (y_c)
    dim_x_loc = x_start + 0.35 * abs(x_start)
    ax.plot([dim_x_loc, C[0]], [C[1], C[1]], color='black', lw=1, linestyle=":", alpha=0.4)
    ax.annotate('', xy=(dim_x_loc, O[1]), xytext=(dim_x_loc, C[1]),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(dim_x_loc - (0.05 * abs(x_start)), C[1]/2, f"$y_c$ = {y_c:.2f}", color='black', 
            ha='right', va='center', fontsize=11, rotation=90)

    # Axis properties execution
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(x_start, x_cowl_end + 0.1 * (L1+L2))
    if mirror:
        ax.set_ylim(y_top_limit, y_bot_limit)
    else:
        ax.set_ylim(y_bot_limit, y_top_limit)

    # --------------------------------------------------
    # Continuous Colorbar Legend Generation
    # --------------------------------------------------
    cbar = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('Mach Number ($M$)', color='black', fontsize=12, weight='bold', labelpad=10)
    
    mach_ticks = sorted(list(set([M_inf, M2, M3, M4])))
    cbar.set_ticks(mach_ticks)
    cbar.set_ticklabels([f"{val:.2f}" for val in mach_ticks])
    ax_cbar.tick_params(labelsize=10, colors='black')

    plt.tight_layout()

    return fig

# ------------------------------------------------------------------
# Quick demo  (runs when the file is executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    results = analyse_intake4(
    M_inf=3,
    L_1=1,
    theta_1_deg=9,
    y_cowl=1.1968,
    delta_cowl_deg=4.0,
    verbose=False
)

fig = draw_intake_cfd_style(results)
plt.show()
