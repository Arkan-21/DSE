import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from Intake_ramjet import analyse_intake4

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

def draw_intake_cfd_style(results, figsize=(15, 6), mirror=False, stretch_factor=2.0):
    """
    Renders the intake shock structure cleanly.
    Ensures color zones match Mach numbers exactly at the shock line boundaries.
    """
    # --------------------------------------------------
    # Extract geometry & multi-reflection parameters
    # --------------------------------------------------
    theta1 = results["theta_1_deg"]
    theta2 = results["theta_2_deg"]
    delta_cowl = results["delta_cowl_deg"]

    L1 = results["L_1"]
    L2 = results["L_2"]

    x_c, y_c = results["cowl_lip"]
    M_inf = results["M_inf"]
    stages = results["stages"]

    # Global inversion multiplier for all Y-coordinates
    flip = -1 if mirror else 1

    t1 = np.radians(theta1)
    t12 = np.radians(theta1 + theta2)
    tcowl = np.radians(delta_cowl)

    # --------------------------------------------------
    # Structural Vertices Definition (With Vertical Stretch)
    # --------------------------------------------------
    O = np.array([0.0, 0.0])
    P1 = np.array([L1, -L1 * np.tan(t1) * flip * stretch_factor])
    P2 = np.array([L1 + L2, (-L1 * np.tan(t1) - L2 * np.tan(t12)) * flip * stretch_factor])
    C = np.array([x_c, -y_c * flip * stretch_factor])

    L_incline = 0.10 * L2
    C_incline_end = np.array([
        C[0] + L_incline,
        C[1] - L_incline * np.tan(tcowl) * flip * stretch_factor
    ])

    # Dynamic Canvas Sizing Bounds
    y_geom_min = min(O[1], P1[1], P2[1], C[1], C_incline_end[1])
    y_geom_max = max(O[1], P1[1], P2[1], C[1], C_incline_end[1])
    s_factor = max(abs(y_geom_max - y_geom_min), 1e-5)

    x_start = -0.35 * (L1 + L2)
    x_cowl_end = P2[0] + max(1.5 * L2, 0.4 * len(stages) * L2)  

    # --------------------------------------------------
    # Exact Wall Segment Definitions
    # --------------------------------------------------
    def get_cowl_y(x):
        if x < C[0]:
            return C[1]
        elif x <= C_incline_end[0]:
            t = (x - C[0]) / (C_incline_end[0] - C[0])
            return C[1] + t * (C_incline_end[1] - C[1])
        else:
            return C_incline_end[1]

    def get_ramp_y(x):
        if x < O[0]:
            return O[1]
        elif x <= P1[0]:
            t = (x - O[0]) / (P1[0] - O[0])
            return O[1] + t * (P1[1] - O[1])
        elif x <= P2[0]:
            t = (x - P1[0]) / (P2[0] - P1[0])
            return P1[1] + t * (P2[1] - P1[1])
        else:
            return P2[1]

    def line_intersection(p1, p2, p3, p4):
        denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
        if abs(denom) < 1e-8:
            return None
        ua = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / denom
        return np.array([p1[0] + ua * (p2[0] - p1[0]), p1[1] + ua * (p2[1] - p1[1])])

    def ray_intersect_walls(x_start_ray, y_start_ray, angle_deg, target_wall="top"):
        rad = np.radians(angle_deg)
        dx_proj = 10.0 * (L1 + L2)
        dy_proj = dx_proj * np.tan(rad) * stretch_factor
        
        ray_end = np.array([x_start_ray + dx_proj, y_start_ray + dy_proj])
        ray_p = (np.array([x_start_ray, y_start_ray]), ray_end)

        if target_wall == "top":
            seg = (C, C_incline_end)
            pt = line_intersection(ray_p[0], ray_p[1], seg[0], seg[1])
            if pt is not None and C[0] <= pt[0] <= C_incline_end[0]:
                return pt
            seg = (C_incline_end, np.array([x_cowl_end * 2, C_incline_end[1]]))
            pt = line_intersection(ray_p[0], ray_p[1], seg[0], seg[1])
            if pt is not None and pt[0] >= C_incline_end[0]:
                return pt
        else:
            seg = (O, P1)
            pt = line_intersection(ray_p[0], ray_p[1], seg[0], seg[1])
            if pt is not None and O[0] <= pt[0] <= P1[0]:
                return pt
            seg = (P1, P2)
            pt = line_intersection(ray_p[0], ray_p[1], seg[0], seg[1])
            if pt is not None and P1[0] <= pt[0] <= P2[0]:
                return pt
            seg = (P2, np.array([x_cowl_end * 2, P2[1]]))
            pt = line_intersection(ray_p[0], ray_p[1], seg[0], seg[1])
            if pt is not None and pt[0] >= P2[0]:
                return pt

        return np.array([x_start_ray + 0.1, y_start_ray])

    # --------------------------------------------------
    # Set up Figure & Mach Color Palette
    # --------------------------------------------------
    fig, (ax, ax_cbar) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [25, 1]})
    colors = ["#1976d2", "#388e3c", "#fbc02d", "#d32f2f"]
    cmap = mcolors.LinearSegmentedColormap.from_list("mach_gradient", colors, N=256)
    
    all_machs = [M_inf] + [s["M_in"] for s in stages] + [stages[-1]["M_out"]]
    norm = mcolors.Normalize(vmin=min(all_machs), vmax=max(all_machs))
    ax.set_facecolor("#e0e0e0")

    y_ambient_far = (y_geom_max + 1.5 * s_factor) if mirror else (y_geom_min - 1.5 * s_factor)
    y_top_limit = (y_geom_min - 0.5 * s_factor) if mirror else (y_geom_max + 0.5 * s_factor)
    y_bot_limit = (y_geom_max + 0.5 * s_factor) if mirror else (y_geom_min - 0.5 * s_factor)

    # Shading upstream unshocked ambient zones
    region_inf = np.array([[x_start, O[1]], O, C, [x_cowl_end, C[1]], [x_cowl_end, y_ambient_far], [x_start, y_ambient_far]])
    ax.fill(region_inf[:,0], region_inf[:,1], color=cmap(norm(M_inf)), zorder=1)

    # First ramp compression zone (Orange)
    region_M2 = np.array([O, P1, C])
    ax.fill(region_M2[:,0], region_M2[:,1], color=cmap(norm(stages[0]["M_out"])), zorder=1)

    # --------------------------------------------------
    # CRITICAL FIX: Split the background region at the (C, P2) line
    # --------------------------------------------------
    # Find the precise point on the first shock wave (O, C) directly under the cowl lip coordinate x
    x_cowl = C[0]
    y_shock_at_x_cowl = (C[1] / C[0]) * x_cowl
    pt_shock_under_cowl = np.array([x_cowl, y_shock_at_x_cowl])

    # 1. Left side of the cowl shock line (Stays Yellow)
    region_M3_yellow = np.array([P1, pt_shock_under_cowl, C])
    ax.fill(region_M3_yellow[:,0], region_M3_yellow[:,1], color=cmap(norm(stages[1]["M_out"])), zorder=1)

    # 2. Right side of the cowl shock line (Turns Green)
    region_M3_green = np.array([pt_shock_under_cowl, P2, C])
    ax.fill(region_M3_green[:,0], region_M3_green[:,1], color=cmap(norm(stages[2]["M_in"])), zorder=1)

    # --------------------------------------------------
    # Core Ray-Tracing Reflection Loop (Unchanged)
    # --------------------------------------------------
    shock_lines = [(O, C), (P1, C), (C, P2)]
    
    last_shock_vertex = P2.copy()
    last_bottom_vertex = P2.copy()
    last_top_vertex = C.copy()
    
    bouncing_up = True 
    x_ns = None  

    for s in stages[2:]:
        m_in = s["M_in"]
        beta = s["beta_deg"]
        
        if "Normal Shock" in s["stage"]:
            x_ns = last_shock_vertex[0]
            v_top = np.array([x_ns, get_cowl_y(x_ns)])
            v_bot = np.array([x_ns, get_ramp_y(x_ns)])
            
            shock_lines.append((v_top, v_bot))
            
            if bouncing_up:
                zone_inter = np.array([last_shock_vertex, last_top_vertex, v_top, v_bot, last_bottom_vertex])
            else:
                zone_inter = np.array([last_shock_vertex, last_bottom_vertex, v_bot, v_top, last_top_vertex])
            
            ax.fill(zone_inter[:,0], zone_inter[:,1], color=cmap(norm(m_in)), zorder=2)
            
            # Post-normal shock subsonic engine channel shading (Blue Zone)
            x_steps = np.linspace(x_ns, x_cowl_end, 100)
            top_wall_pts = [[x, get_cowl_y(x)] for x in x_steps]
            bot_wall_pts = [[x, get_ramp_y(x)] for x in reversed(x_steps)]
            zone_subsonic = np.array(top_wall_pts + bot_wall_pts)
            ax.fill(zone_subsonic[:,0], zone_subsonic[:,1], color=cmap(norm(s["M_out"])), zorder=2)
            break
            
        else:
            if bouncing_up:
                flow_dir = (theta1 + theta2) if not mirror else -(theta1 + theta2)
                angle_ray = (flow_dir + beta) if mirror else (flow_dir - beta)
                
                next_vertex = ray_intersect_walls(last_bottom_vertex[0], last_bottom_vertex[1], angle_ray, target_wall="top")
                
                zone_pts = np.array([last_shock_vertex, last_bottom_vertex, next_vertex, last_top_vertex])
                ax.fill(zone_pts[:,0], zone_pts[:,1], color=cmap(norm(m_in)), zorder=2)
                
                shock_lines.append((last_bottom_vertex.copy(), next_vertex.copy()))
                last_shock_vertex = last_bottom_vertex.copy()
                last_top_vertex = next_vertex.copy()
                bouncing_up = False
            else:
                flow_dir = delta_cowl if not mirror else -delta_cowl
                angle_ray = (flow_dir - beta) if mirror else (flow_dir + beta)
                
                next_vertex = ray_intersect_walls(last_top_vertex[0], last_top_vertex[1], angle_ray, target_wall="bottom")
                
                zone_pts = np.array([last_shock_vertex, last_top_vertex, next_vertex, last_bottom_vertex])
                ax.fill(zone_pts[:,0], zone_pts[:,1], color=cmap(norm(m_in)), zorder=2)
                
                shock_lines.append((last_top_vertex.copy(), next_vertex.copy()))
                last_shock_vertex = last_top_vertex.copy()
                last_bottom_vertex = next_vertex.copy()
                bouncing_up = True

    # --------------------------------------------------
    # Structural Wall Lines Generation & Shock Truncation
    # --------------------------------------------------
    wall_lw = 5.5
    wall_color = "#1a1a1a"
    x_wall_profile = np.linspace(x_start, x_cowl_end, 1000)
    
    ramp_y_profile = [get_ramp_y(x) for x in x_wall_profile]
    ax.plot(x_wall_profile, ramp_y_profile, color=wall_color, lw=wall_lw, zorder=5)
    
    cowl_wall_x = x_wall_profile[x_wall_profile >= C[0]]
    cowl_y_profile = [get_cowl_y(x) for x in cowl_wall_x]
    ax.plot(cowl_wall_x, cowl_y_profile, color=wall_color, lw=wall_lw, zorder=5)

    # Render dashed shock lines with a clean strict cutoff constraint at x_ns
    for pt1, pt2 in shock_lines:
        if x_ns is not None:
            if pt1[0] >= x_ns and pt2[0] >= x_ns:
                continue
            if pt1[0] < x_ns and pt2[0] > x_ns:
                t = (x_ns - pt1[0]) / (pt2[0] - pt1[0])
                pt2 = pt1 + t * (pt2 - pt1)
                
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color="#ffffff", linestyle="--", lw=2.5, zorder=10)

    ax.scatter(C[0], C[1], s=100, color=wall_color, edgecolors="white", linewidths=1.5, zorder=20)

    # --------------------------------------------------
    # Annotations & Layout Dimensions
    # --------------------------------------------------
    arrow_y = -0.6 * (y_c * stretch_factor) if not mirror else 0.6 * (y_c * stretch_factor)
    ax.arrow(arrow_x := x_start + 0.25 * abs(x_start), arrow_y, abs(x_start) * 0.35, 0, 
             width=0.08 * s_factor, head_width=0.25 * s_factor, color="black", zorder=20)
    ax.text(arrow_x + abs(x_start) * 0.05, -0.3 * (y_c * stretch_factor) if not mirror else 0.3 * (y_c * stretch_factor), "Flow", fontsize=14, weight="bold", zorder=25)

    dim_color = "#333333"
    line_y_offset = 0.15 * s_factor * flip
    ext_y_offset = 0.25 * s_factor * flip
    text_y_offset = 0.32 * s_factor * flip
    
    ax.plot([O[0], O[0]], [O[1], O[1] + ext_y_offset], color=dim_color, lw=1, alpha=0.5)
    ax.plot([P1[0], P1[0]], [P1[1], O[1] + ext_y_offset], color=dim_color, lw=1, alpha=0.5)
    ax.annotate('', xy=(O[0], O[1] + line_y_offset), xytext=(P1[0], O[1] + line_y_offset), arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1))
    ax.text(L1/2, O[1] + text_y_offset, f"$L_1$ = {L1:.4f}", color=dim_color, ha='center', va='bottom' if not mirror else 'top', fontsize=11, weight='bold')

    ax.plot([P2[0], P2[0]], [P2[1], O[1] + ext_y_offset], color=dim_color, lw=1, alpha=0.5)
    ax.annotate('', xy=(P1[0], O[1] + line_y_offset), xytext=(P2[0], O[1] + line_y_offset), arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1))
    ax.text(L1 + L2/2, O[1] + text_y_offset, f"$L_2$ = {L2:.4f}", color=dim_color, ha='center', va='bottom' if not mirror else 'top', fontsize=11, weight='bold')

    dim_x_loc = x_start + 0.35 * abs(x_start)
    ax.plot([dim_x_loc, C[0]], [C[1], C[1]], color='black', lw=1, linestyle=":", alpha=0.4)
    ax.annotate('', xy=(dim_x_loc, O[1]), xytext=(dim_x_loc, C[1]), arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(dim_x_loc - (0.05 * abs(x_start)), C[1]/2, f"$y_c$ = {y_c:.4f}", color='black', ha='right', va='center', fontsize=11, rotation=90)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(x_start, x_cowl_end + 0.1 * (L1+L2))
    ax.set_ylim(min(y_top_limit, y_bot_limit), max(y_top_limit, y_bot_limit))

    ax_cbar.clear()
    cbar = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('Mach Number ($M$)', color='black', fontsize=12, weight='bold', labelpad=10)
    
    mach_ticks = sorted(list(set(all_machs)))
    cbar.set_ticks(mach_ticks)
    cbar.set_ticklabels([f"{val:.2f}" for val in mach_ticks])
    ax_cbar.tick_params(labelsize=10, colors='black')

    plt.tight_layout()
    return fig
if __name__ == "__main__":
    results = analyse_intake4(
    M_inf=5,
    L_1=2.235,
    theta_1_deg=9,
    y_cowl=1.1968,
    delta_cowl_deg=4.0,
    verbose=False
)

fig = draw_intake_cfd_style(results, mirror=True)
plt.show()
