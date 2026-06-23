import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from Intake_ramjet import analyse_intake4
from scipy.optimize import brentq



def draw_intake_cfd_style(results, figsize=(15, 6), mirror=False, stretch_factor=1, delta_iso = 0.0, L_diffuser =1.0):
    theta1 = results["theta_1_deg"]
    theta2 = results["theta_2_deg"]
    delta_cowl = results["delta_cowl_deg"]
    phi_ref = results["phi_ref1_deg"]

    L1 = results["L_1"]
    L2 = results["L_2"]

    x_c, y_c = results["cowl_lip"]
    M_inf = results["M_inf"]
    stages = results["stages"]

    flip = -1 if mirror else 1

    t1 = np.radians(theta1)
    t12 = np.radians(theta1 + theta2)
    tcowl = np.radians(delta_cowl)
    tiso = np.radians(delta_iso)
    x_ns = None

    O = np.array([0.0, 0.0])
    P1 = np.array([L1, -L1 * np.tan(t1) * flip * stretch_factor])
    P2 = np.array([L1 + L2, (-L1 * np.tan(t1) - L2 * np.tan(t12)) * flip * stretch_factor])
    C = np.array([x_c, -y_c * flip * stretch_factor])

    L_incline = 0.30 * L2
    C_incline_end = np.array([
        C[0] + L_incline,
        C[1] - L_incline * np.tan(tcowl) * flip * stretch_factor
    ])

    y_geom_min = min(O[1], P1[1], P2[1], C[1], C_incline_end[1])
    y_geom_max = max(O[1], P1[1], P2[1], C[1], C_incline_end[1])
    s_factor = max(abs(y_geom_max - y_geom_min), 1e-5)

    x_start = -0.35 * (L1 + L2)
    x_cowl_end = P2[0] + max(1.5 * L2, 0.4 * len(stages) * L2)

    def area_mach(M, gamma=1.4):
        return ( (1.0 / M)* ((2.0 / (gamma + 1.0))* (1.0 + (gamma - 1.0) / 2.0 * M**2)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))

    def subsonic_mach_from_area_ratio(AAstar, gamma=1.4):
        f = lambda M: area_mach(M, gamma) - AAstar
        return brentq(f, 1e-6, 0.999999)
    
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

            y_base = P2[1]

            if x_ns is not None and x > x_ns:

                dx = x - x_ns

                if not mirror:
                    return (
                    y_base
                    - dx * np.tan(tiso) * stretch_factor
                    )
                else:
                    return (
                    y_base
                    + dx * np.tan(tiso) * stretch_factor
                        )

            return y_base

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

    fig, (ax, ax_cbar) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [25, 1]})
    #colors = ["#1976d2", "#388e3c", "#fbc02d", "#d32f2f"]
    #cmap = mcolors.LinearSegmentedColormap.from_list("mach_gradient", colors, N=256)
    colors = ["#1976d2", "#388e3c", "#fbc02d", "#d32f2f"]
    cmap = mcolors.LinearSegmentedColormap.from_list("mach_gradient",colors,N=256)
    #cmap = plt.get_cmap("turbo")
    #cmap = plt.get_cmap("viridis")
    #cmap = plt.get_cmap("plasma")
    #cmap = plt.get_cmap("jet")

    all_machs = [M_inf] + [s["M_in"] for s in stages] + [stages[-1]["M_out"]]

    diffuser_machs = []

    norm = mcolors.Normalize(vmin=0.4, vmax=max(all_machs))
    ax.set_facecolor("#e0e0e0")

    y_ambient_far = (y_geom_max + 1.5 * s_factor) if mirror else (y_geom_min - 1.5 * s_factor)
    y_top_limit = (y_geom_min - 0.1 * s_factor) if mirror else (y_geom_max + 0.1 * s_factor)
    y_bot_limit = (y_geom_max + 0.4 * s_factor) if mirror else (y_geom_min - 0.4 * s_factor)

    region_inf = np.array([[x_start, O[1]], O, C, [x_cowl_end, C[1]], [x_cowl_end, y_ambient_far], [x_start, y_ambient_far]])
    ax.fill(region_inf[:, 0], region_inf[:, 1], color=cmap(norm(M_inf)), zorder=1)

    region_M2 = np.array([O, P1, C])
    ax.fill(region_M2[:, 0], region_M2[:, 1], color=cmap(norm(stages[0]["M_out"])), zorder=1)

    # Region bounded by the ramp-2 shock (P1->C), reflected shock 1 (C->P2)
    # and the ramp-2 surface (P1->P2): this is the post-ramp-2 / pre-reflection
    # zone, uniformly at M_3 (= M_in of the first reflected shock).
    region_M3 = np.array([P1, C, P2])
    ax.fill(region_M3[:, 0], region_M3[:, 1], color=cmap(norm(stages[2]["M_in"])), zorder=1)

    shock_lines = [(O, C), (P1, C), (C, P2)]

    last_shock_vertex = P2.copy()
    last_bottom_vertex = P2.copy()
    last_top_vertex = C.copy()

    bouncing_up = True
    

    # Reflected shock 1 (C -> P2) is already drawn above, so start the bounce
    # loop at the SECOND reflection (stages[3:]).
    current_angle = phi_ref
    for s in stages[3:]:

        if "Normal Shock" in s["stage"]:

            x_ns = max(last_top_vertex[0], last_bottom_vertex[0])
            if x_ns is not None:
                x_cowl_end = x_ns + L_diffuser

            v_top = np.array([x_ns, get_cowl_y(x_ns)])
            v_bot = np.array([x_ns, get_ramp_y(x_ns)])

            shock_lines.append((v_top, v_bot))

            if bouncing_up:
                zone_inter = np.array([
                    last_shock_vertex,
                    last_top_vertex,
                    v_top,
                    v_bot,
                    last_bottom_vertex
                ])
            else:
                zone_inter = np.array([
                    last_shock_vertex,
                    last_bottom_vertex,
                    v_bot,
                    v_top,
                    last_top_vertex
                ])

            ax.fill(
                zone_inter[:, 0],
                zone_inter[:, 1],
                color=cmap(norm(s["M_in"])),
                zorder=2
            )

            M_ns = s["M_out"]
            x_steps = np.linspace(x_ns, x_cowl_end, 100)

            A0 = get_cowl_y(x_ns) - get_ramp_y(x_ns)

            A0_Astar = area_mach(M_ns)

            for i in range(len(x_steps)-1):
                
                xL = x_steps[i]
                xR = x_steps[i+1]

                AL = get_cowl_y(xL) - get_ramp_y(xL)
                AR = get_cowl_y(xR) - get_ramp_y(xR)

                AL_Astar = (AL / A0) * A0_Astar
                AR_Astar = (AR / A0) * A0_Astar

                ML = subsonic_mach_from_area_ratio(AL_Astar)
                MR = subsonic_mach_from_area_ratio(AR_Astar)

                if i == 0:
                    print("M_start =", ML)

                if i == len(x_steps)-2:
                    print("M_end =", MR)
                    
                Mavg = 0.5 * (ML + MR)

                poly = np.array([
                    [xL, get_ramp_y(xL)],
                    [xR, get_ramp_y(xR)],
                    [xR, get_cowl_y(xR)],
                    [xL, get_cowl_y(xL)]
                ])

                ax.fill(
                    poly[:,0],
                    poly[:,1],
                    color=cmap(norm(Mavg)),
                    zorder=2
                )
                diffuser_machs.append(Mavg)

            break
            
    # ----------------------------
    # Reflected oblique shocks
    # ----------------------------

        if bouncing_up:

            next_vertex = ray_intersect_walls(
                last_bottom_vertex[0],
                last_bottom_vertex[1],
                current_angle,
                target_wall="top"
            )
            

            if next_vertex[0] > C_incline_end[0]:

                zone_pts = np.array([
                    last_shock_vertex,
                    last_bottom_vertex,
                    next_vertex,
                    C_incline_end,
                    last_top_vertex
                ])

            else:

                zone_pts = np.array([
                    last_shock_vertex,
                    last_bottom_vertex,
                    next_vertex,
                    last_top_vertex
                ])

            ax.fill(
                zone_pts[:, 0],
                zone_pts[:, 1],
                color=cmap(norm(s["M_in"])),
                zorder=2
            )

            shock_lines.append(
                (last_bottom_vertex.copy(), next_vertex.copy())
            )

            last_shock_vertex = last_bottom_vertex.copy()
            last_top_vertex = next_vertex.copy()

            bouncing_up = False

        else:

            next_vertex = ray_intersect_walls(
                last_top_vertex[0],
                last_top_vertex[1],
                -current_angle,
                target_wall="bottom"
            )

            zone_pts = np.array([
                last_shock_vertex,
                last_top_vertex,
                next_vertex,
                last_bottom_vertex
            ])

            ax.fill(
                zone_pts[:, 0],
                zone_pts[:, 1],
                color=cmap(norm(s["M_in"])),
                zorder=2
            )

            shock_lines.append(
                (last_top_vertex.copy(), next_vertex.copy())
            )

            last_shock_vertex = last_top_vertex.copy()
            last_bottom_vertex = next_vertex.copy()

            bouncing_up = True
    
    all_machs.extend(diffuser_machs)
    wall_lw = 3.0
    wall_color = "#2b2b2b"
    x_wall_profile = np.linspace(x_start, x_cowl_end, 1000)

    ramp_y_profile = [get_ramp_y(x) for x in x_wall_profile]
    ax.plot(x_wall_profile, ramp_y_profile, color=wall_color, lw=wall_lw, zorder=5)

    cowl_thickness = 0.005 * (L1 + L2)

    cowl_wall_x = x_wall_profile[x_wall_profile >= C[0]]
    cowl_y_profile = np.array([get_cowl_y(x) for x in cowl_wall_x])

    cowl_lower = cowl_y_profile - cowl_thickness * flip

    ax.fill_between(
    cowl_wall_x,
    cowl_y_profile,
    cowl_lower,
    color="#505050",
    zorder=5
    )

    ax.plot(
    cowl_wall_x,
    cowl_y_profile,
    color="black",
    lw=2.5,
    zorder=6
    )

    ax.plot(
    cowl_wall_x,
    cowl_lower,
    color="black",
    lw=2.5,
    zorder=6
    )
    

    for pt1, pt2 in shock_lines:
        if x_ns is not None:
            if pt1[0] >= x_ns and pt2[0] >= x_ns:
                continue
            if pt1[0] < x_ns and pt2[0] > x_ns:
                t = (x_ns - pt1[0]) / (pt2[0] - pt1[0])
                pt2 = pt1 + t * (pt2 - pt1)

        ax.plot(
        [pt1[0], pt2[0]],
        [pt1[1], pt2[1]],
        color="white",
        lw=4.5,
        alpha=0.8,
        zorder=9
        )

        ax.plot(
        [pt1[0], pt2[0]],
        [pt1[1], pt2[1]],
        color="black",
        linestyle="--",
        lw=1.5,
        zorder=10
        )

    # Terminal normal shock: solid vertical line spanning the channel (the
    # truncation logic above intentionally skips it).
    if x_ns is not None:
        ax.plot(
        [x_ns, x_ns],
        [get_ramp_y(x_ns), get_cowl_y(x_ns)],
        color="white",
        lw=5,
        alpha=0.8,
        zorder=9
        )

        ax.plot(
        [x_ns, x_ns],
        [get_ramp_y(x_ns), get_cowl_y(x_ns)],
        color="black",
        lw=2,
        zorder=10
        )

    ax.scatter(C[0],C[1],s=140,color="#404040",edgecolors="white",linewidths=2.0,zorder=20)

    flow_x0 = x_start + 0.8 * abs(x_start)
    flow_x1 = x_start + 1.2 * abs(x_start)

    flow_y = (
    C[1] + 0.1 * (O[1] - C[1])
    if not mirror
    else
    C[1] - 0.1 * (C[1] - O[1])
    )

    ax.annotate(
    "",
    xy=(flow_x1, flow_y),
    xytext=(flow_x0, flow_y),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=2.5,
        color="black",
        shrinkA=0,
        shrinkB=0,
        mutation_scale=18,
    ),
    zorder=20
    )
    ax.text(
    0.5 * (flow_x0 + flow_x1),
    flow_y + 0.08 * s_factor * flip,
    "Post Bow Shock Flow",
    fontsize=12,
    weight="bold",
    ha="center",
    va="bottom" if not mirror else "top",
    color="black",
    zorder=21,
    )

    dim_color = "#333333"
    line_y_offset = 0.08 * s_factor * flip
    ext_y_offset = 0.12 * s_factor * flip
    text_y_offset = 0.16 * s_factor * flip

    ax.plot([O[0], O[0]], [O[1], O[1] + ext_y_offset], color=dim_color, lw=1, alpha=0.5)
    ax.plot([P1[0], P1[0]], [P1[1], O[1] + ext_y_offset], color=dim_color, lw=1, alpha=0.5)
    ax.annotate('', xy=(O[0], O[1] + line_y_offset), xytext=(P1[0], O[1] + line_y_offset), arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1))
    ax.text(L1 / 2, O[1] + text_y_offset, f"$L_1$ = {L1:.4f} m", color=dim_color, ha='center', va='bottom' if not mirror else 'top', fontsize=11, weight='bold')

    ax.plot([P2[0], P2[0]], [P2[1], O[1] + ext_y_offset], color=dim_color, lw=1, alpha=0.5)
    ax.annotate('', xy=(P1[0], O[1] + line_y_offset), xytext=(P2[0], O[1] + line_y_offset), arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1))
    ax.text(L1 + L2 / 2, O[1] + text_y_offset, f"$L_2$ = {L2:.4f} m", color=dim_color, ha='center', va='bottom' if not mirror else 'top', fontsize=11, weight='bold')

    dim_x_loc = x_start + 0.35 * abs(x_start)
    ax.plot([dim_x_loc, C[0]], [C[1], C[1]], color='black', lw=1, linestyle=":", alpha=0.4)
    ax.annotate('', xy=(dim_x_loc, O[1]), xytext=(dim_x_loc, C[1]), arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(dim_x_loc - (0.05 * abs(x_start)), C[1] / 2, f"$y_c$ = {y_c:.4f} m", color='black', ha='right', va='center', fontsize=11, rotation=90)

    L_iso = x_ns - P2[0]
    if x_ns is not None:

        ax.plot(
        [x_ns, x_ns],
        [get_ramp_y(x_ns), O[1] + ext_y_offset],
        color=dim_color,
        lw=1,
        alpha=0.5
        )

        ax.annotate(
        '',
        xy=(P2[0], O[1] + line_y_offset),
        xytext=(x_ns, O[1] + line_y_offset),
        arrowprops=dict(
            arrowstyle='<->',
            color=dim_color,
            lw=1
            )
        )

        ax.text(
        0.5 * (P2[0] + x_ns),
        O[1] + text_y_offset,
        rf"$L_{{iso}}$ = {x_ns - P2[0]:.4f} m",
        color=dim_color,
        ha='center',
        va='bottom' if not mirror else 'top',
        fontsize=11,
        weight='bold'
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
   

   

    ax.set_xlim(x_start, x_cowl_end)
    ax.set_ylim(min(y_top_limit, y_bot_limit), max(y_top_limit, y_bot_limit))

    ax_cbar.clear()
    cbar = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, orientation='vertical')
    cbar.outline.set_linewidth(1.2)
    cbar.outline.set_edgecolor("black")
    cbar.set_label(
    "Mach Number",
    fontsize=12,
    weight="bold"
    )

    mach_ticks = sorted(list(set(
    [M_inf] + [s["M_in"] for s in stages[:-1]] + [stages[-1]["M_out"]]
    ))) 
    cbar.set_ticks(mach_ticks)
    cbar.set_ticklabels([f"{val:.2f}" for val in mach_ticks])
    ax_cbar.tick_params(labelsize=10, colors='black')

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    results = analyse_intake4(
        M_inf=4.35,
        L_1= 1.94,
        theta_1_deg=9.8,
        y_cowl=1.2,
        delta_cowl_deg=4,
        verbose=False
    )

    fig = draw_intake_cfd_style(results, mirror= False, delta_iso=-6, L_diffuser=1)
    plt.show()
