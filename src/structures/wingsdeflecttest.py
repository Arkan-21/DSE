from stl import mesh
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import PchipInterpolator

E            = 100e9
C            = 6.98
C_shear      = 5.34
sigma_yield  = 700e6
alpha        = 0.8
n_fac        = 0.6
v            = 0.31
n_sections_span = 100
thermal = 10e-6
temp = 200
G = E/(2+2*v)

def read_load_file(file, plot, n_sections, safety=2.0):
    df = pd.read_csv(file)
    df = df[df["y"] >= 0.0][["y", "Fn_elem_[N]", "x_ac"]].reset_index(drop=True)
    df["y"] = df["y"] - df["y"].min()
    df = df.sort_values("y").reset_index(drop=True)

    y   = df["y"].values
    f   = df["Fn_elem_[N]"].values * safety
    xac = df["x_ac"].values

    y_tip = y.max()
    eta   = y / y_tip

    print(np.sum(f)/1e6)

    # --------------------------------------------------
    # Load distribution fit (unchanged)
    # --------------------------------------------------
    X = np.vstack([np.ones_like(eta), (1 - eta), (1 - eta)**2]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, f, rcond=None)

    def f_model(eps):
        return np.vstack([np.ones_like(eps), (1 - eps), (1 - eps)**2]).T @ coeffs

    y_uniform = np.linspace(0, y_tip, n_sections)
    eta_u     = y_uniform / y_tip
    fn        = np.clip(f_model(eta_u), 0, None)

    # --------------------------------------------------
    # Tip taper smoothing (unchanged)
    # --------------------------------------------------
    t0   = 0.85
    t    = y_uniform / y_tip
    mask = t > t0

    if np.any(mask):
        t_local = (t[mask] - t0) / (1 - t0)
        fn[mask] *= 1 - (3 * t_local**2 - 2 * t_local**3)

    fn[-1] = 0.0

    # --------------------------------------------------
    # Force normalization (unchanged)
    # --------------------------------------------------
    F_target = np.trapz(f, y)
    F_model  = np.trapz(fn, y_uniform)

    if F_model > 0:
        fn *= F_target / F_model

    # --------------------------------------------------
    # FIXED: x_ac linear reconstruction
    # --------------------------------------------------

    # remove first two bad datapoints
    y_fit   = y[2:]
    xac_fit = xac[2:]

    # linear fit x_ac = a*y + b
    a, b = np.polyfit(y_fit, xac_fit, 1)

    # reconstruct for full span
    x_ac_uniform = a * y_uniform + b

    return y_uniform, fn, x_ac_uniform, df


def plot_loads_2d(y_uniform, fn_uniform, df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_uniform, fn_uniform, color="steelblue", linewidth=2, label="Interpolated")
    ax.scatter(df["y"], df["Fn_elem_[N]"], color="tomato", s=30, zorder=5,
               label="Original data points")
    ax.set_xlabel("Spanwise position y (m)")
    ax.set_ylabel("Normal force per element Fn (N)")
    ax.set_title("Spanwise Normal Force Distribution")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("2d_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_loads_on_wing_3d(stations, y_uniform, fn_uniform):
    cmap = plt.get_cmap("jet")
    norm = mcolors.Normalize(vmin=np.min(fn_uniform), vmax=np.max(fn_uniform))
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection="3d")

    y_sorted = sorted(stations.keys())[1:]
    arrow_skip = max(1, len(y_sorted) // 25)
    arrow_scale = 0.15 * (max(y_sorted) - min(y_sorted)) / np.max(np.abs(fn_uniform))

    for idx, y in enumerate(y_sorted):
        gpts = stations[y]["global_pts"]
        x_g, z_g = gpts[:, 0], gpts[:, 2]
        fn_val = np.interp(y, y_uniform, fn_uniform)
        ax.plot(np.append(x_g, x_g[0]), np.full(len(x_g) + 1, y), np.append(z_g, z_g[0]),
                color="k", linewidth=0.8, alpha=0.6)
        if idx % arrow_skip == 0:
            ax.quiver(np.mean(x_g), y, np.mean(z_g), 0, 0, fn_val * arrow_scale,
                      color=cmap(norm(fn_val)), linewidth=1.0, arrow_length_ratio=0.12, normalize=False)

    # Tight axis limits around the data
    all_x = np.concatenate([stations[y]["global_pts"][:, 0] for y in y_sorted])
    all_z = np.concatenate([stations[y]["global_pts"][:, 2] for y in y_sorted])
    x_pad = 0.05 * np.ptp(all_x)
    z_pad = 0.05 * np.ptp(all_z)
    y_pad = 0.05 * np.ptp(y_sorted)

    ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    ax.set_ylim(min(y_sorted) - y_pad, max(y_sorted) + y_pad)
    ax.set_zlim(all_z.min() - z_pad * 6, all_z.max() + z_pad * 6)

    ax.set_xlabel("X [m]", fontsize=11, labelpad=10)
    ax.set_ylabel("Y [m]", fontsize=11, labelpad=10)
    ax.set_zlabel("Z [m]", fontsize=11, rotation=90, labelpad=18)
    ax.zaxis.set_rotate_label(False)
    ax.set_title("Spanwise Load Distribution", fontsize=13, pad=15)

    ax.tick_params(axis='both', labelsize=9)
    ax.tick_params(axis='z', labelsize=9)
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.view_init(elev=20, azim=-55)

    try:
        ax.set_box_aspect([max(np.ptp(ax.get_xlim()), 1e-6),
                           max(np.ptp(ax.get_ylim()), 1e-6),
                           max(np.ptp(ax.get_zlim()) * 3.5, 1e-6)])
    except Exception:
        pass

    ax.grid(True, alpha=0.25)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=-0.05)
    cbar.set_label("Fn per element [N]", fontsize=11, labelpad=6)
    cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(bottom=0.05, top=0.97, left=0.05, right=0.98)
    plt.savefig("3dloads_plot.pdf", dpi=200, bbox_inches="tight")
    plt.close()

def extract_station(triangles, y_target, n_pts=400):
    rel = triangles - np.array([0., y_target, 0.])[None, None, :]
    d   = rel @ np.array([0., 1., 0.])

    raw_pts = []
    for tri_idx in range(len(triangles)):
        d_tri = d[tri_idx]
        tri   = triangles[tri_idx]
        for i in range(3):
            j = (i + 1) % 3
            d0, d1 = d_tri[i], d_tri[j]
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                raw_pts.append(tri[i] + t * (tri[j] - tri[i]))
    if len(raw_pts) < 4:
        return None

    raw_pts = np.unique(np.round(raw_pts, 5), axis=0)
    x, z    = raw_pts[:, 0], raw_pts[:, 2]

    cx, cz = np.mean(x), np.mean(z)
    order  = np.argsort(np.arctan2(z - cz, x - cx))
    loop_x = x[order]
    loop_z = z[order]

    signed_area = 0.5 * np.sum(loop_x * np.roll(loop_z, -1)
                                - np.roll(loop_x, -1) * loop_z)
    if signed_area < 0:
        loop_x = loop_x[::-1]
        loop_z = loop_z[::-1]

    loop_x = np.append(loop_x, loop_x[0])
    loop_z = np.append(loop_z, loop_z[0])
    ds     = np.sqrt(np.diff(loop_x)**2 + np.diff(loop_z)**2)
    s      = np.concatenate([[0], np.cumsum(ds)]); s /= s[-1]
    s_new  = np.linspace(0, 1, n_pts, endpoint=False)
    loop_x = interp1d(s, loop_x)(s_new)
    loop_z = interp1d(s, loop_z)(s_new)

    cx2, cz2 = np.mean(loop_x), np.mean(loop_z)
    return {
        'x':          loop_x - cx2,
        'z':          loop_z - cz2,
        'origin':     np.array([cx2, y_target, cz2]),
        'global_pts': np.column_stack([loop_x,
                                       np.full(n_pts, y_target),
                                       loop_z]),
        'u_axis':     np.array([1., 0., 0.]),
        'v_axis':     np.array([0., 0., 1.]),
        'normal':     np.array([0., 1., 0.]),
    }


def create_dense_stations(triangles, n_sections, n_pts_per_station=120, **kwargs):
    y_min  = triangles[:, :, 1].min()
    y_max  = triangles[:, :, 1].max()
    margin = 0.01 * (y_max - y_min)
    dense  = {}

    for y in np.linspace(y_min + margin, y_max - margin, n_sections):
        st = extract_station(triangles, y)
        if st is None:
            print(f"  Warning: no section at y={y:.3f}")
            continue
        x, z   = st['x'], st['z']
        ds     = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
        s      = np.concatenate([[0], np.cumsum(ds)]); s /= s[-1]
        s_new  = np.linspace(0, 1, n_pts_per_station)
        x_r    = interp1d(s, x)(s_new)
        z_r    = interp1d(s, z)(s_new)
        origin = st['origin']
        dense[y] = {
            'x':          x_r,
            'z':          z_r,
            'global_pts': np.column_stack([x_r + origin[0],
                                           np.full(n_pts_per_station, y),
                                           z_r + origin[2]]),
            'origin':     origin,
            'u_axis':     np.array([1., 0., 0.]),
            'v_axis':     np.array([0., 0., 1.]),
            'normal':     np.array([0., 1., 0.]),
        }

    print(f"Extracted {len(dense)} / {n_sections} stations")
    return dense


def create_mesh(filename, n_sections, plot=False, span_crop=2.0,
                n_pts_per_station=120, **kwargs):
    m         = mesh.Mesh.from_file(filename)
    triangles = m.vectors

    mask = ((triangles[:, 0, 1] >= 0) &
            (triangles[:, 1, 1] >= 0) &
            (triangles[:, 2, 1] >= 0))
    triangles = triangles[mask]
    triangles[:, :, 1] -= triangles[:, :, 1].min()

    if span_crop > 0:
        mask_crop = ((triangles[:, 0, 1] >= span_crop) &
                     (triangles[:, 1, 1] >= span_crop) &
                     (triangles[:, 2, 1] >= span_crop))
        triangles = triangles[mask_crop]
        triangles[:, :, 1] -= triangles[:, :, 1].min()

    print(f"Triangles : {len(triangles)}")
    print(f"Y range   : {triangles[:,:,1].min():.3f} to {triangles[:,:,1].max():.3f}")

    stations = create_dense_stations(triangles, n_sections,
                                     n_pts_per_station=n_pts_per_station)
    if plot:
        _plot_mesh_and_stations(triangles, stations)
    return stations


def _plot_mesh_and_stations(triangles, stations):
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')
    poly = Poly3DCollection(triangles, alpha=0.3)
    poly.set_facecolor('cyan'); poly.set_edgecolor('gray')
    ax.add_collection3d(poly)
    for st in stations.values():
        pts = st['global_pts']
        pts_closed = np.vstack([pts, pts[0]])
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2],
                'r-', linewidth=1.0)
    verts = triangles.reshape(-1, 3)
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Wing stations')
    plt.tight_layout()
    plt.savefig("output_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


def section_prop_Iy(x, z, t):
    x = np.append(x, x[0])
    z = np.append(z, z[0])

    cx, cz, total_length = 0.0, 0.0, 0.0
    for i in range(len(x) - 1):
        ds          = np.sqrt((x[i+1]-x[i])**2 + (z[i+1]-z[i])**2)
        cx         += (x[i] + x[i+1]) / 2 * ds
        cz         += (z[i] + z[i+1]) / 2 * ds
        total_length += ds
    cx /= total_length
    cz /= total_length

    Iy = 0.0
    for i in range(len(x) - 1):
        ds    = np.sqrt((x[i+1]-x[i])**2 + (z[i+1]-z[i])**2)
        angle = np.arctan2(z[i+1]-z[i], x[i+1]-x[i])
        z_mid = (z[i+1] + z[i]) / 2
        Value   = 1/12 * t * ds**3 * np.sin(angle)**2 + ds * t * (cz - z_mid)**2
        Iy += 1.1*Value
    return Iy


def wing_deflection(stations, plot, t, t_end):
    y_list = sorted(stations.keys())
    y0     = y_list[0]
    n      = len(y_list) - 1

    t_arr      = np.linspace(t, t / t_end, n)
    y_mids     = np.array([((y_list[i] + y_list[i+1]) / 2) - y0 for i in range(n)])
    seclengths = np.array([y_list[i+1] - y_list[i]             for i in range(n)])
    Iy_arr     = np.array([section_prop_Iy(stations[y_list[i]]['x'],
                                            stations[y_list[i]]['z'],
                                            t_arr[i]) for i in range(n)])
    z_max = np.array([np.max(np.abs(stations[y_list[i]]['z'] -
                                     np.mean(stations[y_list[i]]['z'])))
                      for i in range(n)])

    P = load_span
    M = np.zeros(n)
    for i in range(n):
        for j in range(i, n):
            M[i] += P[j] * seclengths[j] * (y_mids[j] - y_mids[i])

    curvature  = M / (E * Iy_arr)
    slope      = np.zeros(n)
    deflection = np.zeros(n)
    for i in range(1, n):
        slope[i]      = slope[i-1] + 0.5 * (curvature[i] + curvature[i-1]) * seclengths[i]
        deflection[i] = (deflection[i-1]
                         + slope[i-1] * seclengths[i]
                         + (1/6) * (2*curvature[i-1] + curvature[i]) * seclengths[i]**2)

    # Split stress contributions
    bending_stress_max = (M * z_max) / Iy_arr
    thermal_stress_val = thermal_loads(thermal, temp, E)   # scalar constant
    stress_max         = bending_stress_max + thermal_stress_val

    stress_cross = (-M[0] * stations[y_list[0]]['z']) / Iy_arr[0]

    if plot:
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle('Wing Structural Analysis Along Span', fontsize=14, fontweight='bold')

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        # --- ax1: Spanwise Iy ---
        ax1.plot(y_mids, Iy_arr, color='steelblue', linewidth=2)
        ax1.fill_between(y_mids, Iy_arr, alpha=0.15, color='steelblue')
        ax1.set_ylabel('$I_y$ [m$^4$]', fontsize=11)
        ax1.set_xlabel('Span Position $y$ [m]', fontsize=11)
        ax1.set_title('Spanwise $I_y$ Distribution', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # --- ax2: Spanwise Deflection ---
        ax2.plot(y_mids, deflection, color='firebrick', linewidth=2)
        ax2.fill_between(y_mids, deflection, alpha=0.15, color='firebrick')
        ax2.set_ylabel('Deflection [m]', fontsize=11)
        ax2.set_xlabel('Span Position $y$ [m]', fontsize=11)
        ax2.set_title('Spanwise Deflection Distribution', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # --- ax3: Stress breakdown (bending + thermal) ---
        thermal_arr  = np.full_like(bending_stress_max, thermal_stress_val)
        total_stress = bending_stress_max + thermal_arr

        # Orange fill + line: bending component (spanwise-varying)
        ax3.fill_between(y_mids, 0, bending_stress_max / 1e6,
                         alpha=0.25, color='darkorange', label='_nolegend_')
        ax3.plot(y_mids, bending_stress_max / 1e6,
                 color='darkorange', linewidth=2, label='Bending stress')

        # Blue fill: constant thermal offset stacked on top of bending
        ax3.fill_between(y_mids, bending_stress_max / 1e6, total_stress / 1e6,
                         alpha=0.35, color='steelblue', label='_nolegend_')
        ax3.axhline(thermal_stress_val / 1e6, color='steelblue', linewidth=1.5,
                    linestyle='--',
                    label=f'Thermal stress ({thermal_stress_val / 1e6:.2f} MPa)')

        # Red line: total combined stress
        ax3.plot(y_mids, total_stress / 1e6,
                 color='firebrick', linewidth=2, label='Total stress')

        ax3.set_ylabel('Stress [MPa]', fontsize=11)
        ax3.set_xlabel('Span Position $y$ [m]', fontsize=11)
        ax3.set_title('Spanwise Stress Distribution', fontsize=11)
        ax3.legend(fontsize=8, loc='upper right')
        ax3.grid(True, linestyle='--', alpha=0.5)

        # --- ax4: Root section cross-sectional bending stress ---
        x_cs = stations[y_list[0]]['x']
        z_cs = stations[y_list[0]]['z']
        max_stress = np.max(np.abs(stress_cross))
        if max_stress == 0:
            max_stress = 1.0
        sc_norm = mcolors.TwoSlopeNorm(vmin=-max_stress, vcenter=0, vmax=max_stress)
        sc_cmap = cm.RdBu_r

        # Draw airfoil outline
        ax4.plot(np.append(x_cs, x_cs[0]), np.append(z_cs, z_cs[0]),
                 'k-', linewidth=1.2, alpha=0.5, zorder=2)

        # Arrows normal to surface, scaled by stress magnitude
        chord       = x_cs.max() - x_cs.min()
        arrow_scale = 0.06 * chord / max_stress
        step        = max(1, len(x_cs) // 40)

        for i in range(0, len(x_cs), step):
            s   = stress_cross[i]
            col = sc_cmap(sc_norm(s))
            alp = 0.4 + 0.6 * abs(s) / max_stress
            # Approximate outward normal from centroid
            nx  = x_cs[i] - np.mean(x_cs)
            nz  = z_cs[i] - np.mean(z_cs)
            nlen = np.sqrt(nx**2 + nz**2) + 1e-12
            nx /= nlen
            nz /= nlen

        sc = ax4.scatter(x_cs, z_cs, c=stress_cross, cmap=sc_cmap,
                         norm=sc_norm, s=12, zorder=4)
        ax4.set_xlabel('x [m]', fontsize=9)
        ax4.set_ylabel('z [m]', fontsize=9)
        ax4.set_title('Root Section Bending Stress', fontsize=11)
        ax4.grid(True, linestyle='--', alpha=0.3)

        # Exaggerate Z axis so the thin airfoil is visible
        x_range = x_cs.max() - x_cs.min()
        z_mid   = (z_cs.max() + z_cs.min()) / 2
        ax4.set_xlim(x_cs.min() - 0.02 * x_range, x_cs.max() + 0.02 * x_range)
        ax4.set_ylim(z_mid - 0.15 * x_range, z_mid + 0.15 * x_range)

        cbar = fig.colorbar(sc, ax=ax4, orientation='vertical', shrink=0.8, pad=0.04)
        cbar.set_label('Bending stress [Pa]', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        plt.tight_layout(h_pad=3.5)
        plt.savefig("bending_plot.pdf", dpi=150, bbox_inches="tight")
        plt.close()

    return deflection[-1], stress_max

def sensitivity_analysis_bending(stations):
    thicknesses = np.linspace(0.001, 0.010, 100)

    deflect_list, stress_list = [], []

    for t in thicknesses:
        d, s = wing_deflection(stations, False, t, 1)
        deflect_list.append(d)
        stress_list.append(np.max(s))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    ax1.plot(
        thicknesses * 1e3,
        deflect_list,
        linewidth=2,
        color='tab:blue'
    )
    ax1.set_ylabel('Deflection (m)')
    ax1.set_xlabel('Thickness (mm)')
    ax1.set_title('Deflection vs Thickness')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(
        thicknesses * 1e3,
        np.array(stress_list) / 1e6,
        linewidth=2,
        color='tab:red',
        label='Maximum Bending Stress'
    )

    # Yield stress line
    yield_stress_mpa = sigma_yield / 1e6

    ax2.axhline(
        yield_stress_mpa,
        color='k',
        linestyle=':',
        linewidth=2,
        label=f'Yield Stress ({yield_stress_mpa:.1f} MPa)'
    )

    ax2.set_ylabel('Max Stress (MPa)')
    ax2.set_xlabel('Thickness (mm)')
    ax2.set_title('Max Stress vs Thickness')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("sensbending_plot.pdf", dpi=150, bbox_inches="tight")
    plt.close()


def cross_section(x, z):
    x = np.append(x, x[0])
    z = np.append(z, z[0])
    cross_prod = x[:-1]*z[1:] - x[1:]*z[:-1]
    Am         = 0.5 * np.sum(cross_prod)
    Am_signed  = Am  # keep for Bredt (needs positive area)
    Am_abs     = abs(Am)
    
    # Centroid in global frame
    cx = np.sum((x[:-1] + x[1:]) * cross_prod) / (6 * Am)  
    
    # Distance from local leading edge
    x_le  = np.min(x[:-1])
    cx_le = cx - x_le
    
    perimeter = np.sum(np.sqrt(np.diff(x)**2 + np.diff(z)**2))
   
    return Am_abs, cx_le, perimeter


def wing_torque(stations, plot, t_start, t_end):
    y_list = sorted(stations.keys())
    n      = len(y_list) - 1
    y0     = y_list[0]

    t_arr      = np.linspace(t_start, t_start / t_end, n)
    seclengths = np.array([y_list[i+1] - y_list[i]             for i in range(n)])
    span_arr   = np.array([((y_list[i] + y_list[i+1]) / 2) - y0 for i in range(n)])

    result = np.array([
        cross_section(stations[y_list[i]]['x'], stations[y_list[i]]['z'])
        for i in range(n)
    ])
    Am_arr        = result[:, 0]
    cx_arr        = result[:, 1]
    perimeter_arr = result[:, 2]

    load = np.array(load_span[:n])

    x_le_arr = np.array([np.min(stations[y_list[i]]['x']) for i in range(n)])


    x_ac_centroid = xac_uniform[:n] + x_le_arr  
    arm = cx_arr + x_ac_centroid
    dTdy = load * arm

    T      = np.zeros(n)
    T[n-1] = 0.0
    for i in range(n - 2, -1, -1):
        T[i] = T[i+1] + 0.5 * (dTdy[i] + dTdy[i+1]) * seclengths[i]

    twist_list       = []
    total_twist_list = []
    shear_list       = []
    total_twist      = 0.0

    for i in range(n):
        J           = 4 * Am_arr[i]**2 / (perimeter_arr[i] / (10*t_arr[i]))
        d_twist     = T[i] / (G * J) * seclengths[i] * 180 / np.pi
        shear       = T[i] / (2 * t_arr[i] * Am_arr[i])
        total_twist += d_twist
        twist_list.append(d_twist)
        total_twist_list.append(total_twist)
        shear_list.append(shear)


    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Wing Torsional Analysis Along Span', fontsize=14, fontweight='bold')
        ax1.plot(span_arr, total_twist_list, color='mediumpurple', linewidth=2)
        ax1.fill_between(span_arr, total_twist_list, alpha=0.15, color='mediumpurple')
        ax1.set_ylabel('Twist Angle [deg]', fontsize=11)
        ax1.set_xlabel('Span Position $y$ [m]', fontsize=11)
        ax1.set_title('Spanwise Twist Distribution', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax2.plot(span_arr, np.array(shear_list) / 1e6, color='teal', linewidth=2)
        ax2.fill_between(span_arr, np.array(shear_list) / 1e6, alpha=0.15, color='teal')
        ax2.set_ylabel('Shear Stress [MPa]', fontsize=11)
        ax2.set_xlabel('Span Position $y$ [m]', fontsize=11)
        ax2.set_title('Spanwise Shear Stress Distribution', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("torque_plot.pdf", dpi=150, bbox_inches="tight")
        plt.close()

    return total_twist_list[-1], shear_list


def sensitivity_analysis_torque(stations):
    thicknesses = np.linspace(0.001, 0.010, 100)
    twist_list, shear_list = [], []

    for t in thicknesses:
        tw, sh = wing_torque(stations, False, t, 1)
        twist_list.append(tw)
        shear_list.append(np.max(np.abs(sh)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    ax1.plot(thicknesses * 1e3, twist_list,
             linewidth=2, color='tab:blue')
    ax1.set_ylabel('Twist Angle (degrees)')
    ax1.set_xlabel('Thickness (mm)')
    ax1.set_title('Twist Angle vs Thickness')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(thicknesses * 1e3,
             np.array(shear_list) / 1e6,
             linewidth=2,
             color='tab:red',
             label='Maximum Shear Stress')

    # Tresca yield criterion
    tresca_shear = sigma_yield / 2 / 1e6  # MPa

    ax2.axhline(
        tresca_shear,
        color='k',
        linestyle=':',
        linewidth=2,
        label=f'Tresca Yield ({tresca_shear:.1f} MPa)'
    )

    ax2.set_ylabel('Max Shear Stress (MPa)')
    ax2.set_xlabel('Thickness (mm)')
    ax2.set_title('Max Shear Stress vs Thickness')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("sensetorque_plot.pdf", dpi=150, bbox_inches="tight")
    plt.close()

def thermal_loads(alpha, temp,E):
    stress = alpha * temp* E
    return stress


def von_mises(stations, plot):
    thicknesses = np.linspace(0.001, 0.010,100)
    vm_list = []
    for t in thicknesses:
        _, max_stress = wing_deflection(stations, False, t, 1) 
        _, max_shear  = wing_torque(stations, False, t, 1)
        vm_list.append(np.sqrt(max_stress**2 + 3 * np.array(max_shear)**2))
  

    if plot:
        max_vm = np.max(np.array(vm_list), axis=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thicknesses * 1e3, max_vm / 1e6, linewidth=2)
        ax.axhline(sigma_yield / 1e6, linestyle='--', linewidth=1.5,
                   color='red', label=f'Yield {sigma_yield/1e6:.0f} MPa')
        ax.set_xlabel('Thickness [mm]')
        ax.set_ylabel('Maximum von Mises stress [MPa]')
        ax.set_title('Von Mises Yield Criterion vs Thickness')
        ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig("von_mises.pdf", dpi=150, bbox_inches="tight")
        plt.close()


def skin_buckling(stations):
    thicknesses = np.linspace(0.001, 0.010, 400)
    b           = np.linspace(0.05, 0.5, 400)

    T, B = np.meshgrid(thicknesses, b, indexing='ij')

    # --------------------------------------------------
    # Buckling surfaces
    # --------------------------------------------------
    sigma_cr = (
        alpha * (
            C / sigma_yield
            * (np.pi**2 * E)
            / (12 * (1 - v**2))
            * (T / B) ** 2
        ) ** (1 - n_fac)
        * sigma_yield
    )

    tau_cr = (
        alpha * (
            C_shear / sigma_yield
            * (np.pi**2 * E)
            / (12 * (1 - v**2))
            * (T / B) ** 2
        ) ** (1 - n_fac)
        * sigma_yield
    )

    # --------------------------------------------------
    # Applied stresses from structure model
    # --------------------------------------------------
    sigma_applied = np.zeros_like(thicknesses)
    tau_applied   = np.zeros_like(thicknesses)

    for i, t in enumerate(thicknesses):
        _, stress = wing_deflection(stations, False, t, 1)
        _, shear  = wing_torque(stations, False, t, 1)

        print(i)

        sigma_applied[i] = np.max(np.abs(stress))
        tau_applied[i]   = np.max(np.abs(shear))

    # Expand to 2D for contour logic
    sigma_boundary = sigma_cr - sigma_applied[:, None]
    tau_boundary   = tau_cr   - tau_applied[:, None]

    vmax = np.percentile(
        np.concatenate([sigma_cr.ravel(), tau_cr.ravel()]) / 1e6,
        97
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    X, Y = np.meshgrid(b * 1e3, thicknesses * 1e3)

    # ==================================================
    # Compression panel
    # ==================================================
    im1 = axes[0].imshow(
        sigma_cr / 1e6,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[
            b[0] * 1e3,
            b[-1] * 1e3,
            thicknesses[0] * 1e3,
            thicknesses[-1] * 1e3
        ],
        vmin=0,
        vmax=vmax
    )

    # Unsafe region (buckling)
    axes[0].contourf(
        X, Y,
        sigma_boundary,
        levels=[-1e20, 0],
        colors='red',
        alpha=0.18
    )

    cs1 = axes[0].contour(
        X, Y,
        sigma_boundary,
        levels=[0],
        colors='white',
        linewidths=2,
        linestyles='--'
    )

    axes[0].clabel(
        cs1,
        fmt=r'$\sigma_{cr}=\sigma_{applied}$',
        fontsize=9,
        inline=True,
        colors='white'
    )

    axes[0].set_title(r'$\sigma_{cr}$ (Compression)')
    axes[0].set_xlabel('b (mm)')
    axes[0].set_ylabel('Thickness (mm)')
    axes[0].grid(True, alpha=0.2)

    # ==================================================
    # Shear panel
    # ==================================================
    im2 = axes[1].imshow(
        tau_cr / 1e6,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[
            b[0] * 1e3,
            b[-1] * 1e3,
            thicknesses[0] * 1e3,
            thicknesses[-1] * 1e3
        ],
        vmin=0,
        vmax=vmax
    )

    # Unsafe region (buckling)
    axes[1].contourf(
        X, Y,
        tau_boundary,
        levels=[-1e20, 0],
        colors='red',
        alpha=0.18
    )

    cs2 = axes[1].contour(
        X, Y,
        tau_boundary,
        levels=[0],
        colors='white',
        linewidths=2,
        linestyles='--'
    )

    axes[1].clabel(
        cs2,
        fmt=r'$\tau_{cr}=\tau_{applied}$',
        fontsize=9,
        inline=True,
        colors='white'
    )

    axes[1].set_title(r'$\tau_{cr}$ (Shear)')
    axes[1].set_xlabel('b (mm)')
    axes[1].set_ylabel('Thickness (mm)')
    axes[1].grid(True, alpha=0.2)

    # --------------------------------------------------
    # Shared colorbar
    # --------------------------------------------------
    fig.colorbar(im1, ax=axes, label='Critical Stress (MPa)')

    #plt.tight_layout()
    plt.savefig("wing_buckling.pdf", dpi=150, bbox_inches="tight")
    plt.close()


def optimization(stations):
    thicknesses  = np.linspace(0.001, 0.020, 10)
    endings      = np.linspace(1, 10, 10)
    yield_stress = 2.0
    accepted, data_final = [], []

    for t in thicknesses:
        t = round(t, 3)
        _, max_stress = wing_deflection(stations, False, t, 1)
        _, max_shear  = wing_torque(stations, False, t, 1)
        vm = np.sqrt(max_stress**2 + 3 * np.array(max_shear)**2) / 1e6
        if (np.all(max_stress / 1e6 < yield_stress) and
                np.all(np.abs(max_shear) / 1e6 < yield_stress / 2) and
                np.all(vm < yield_stress)):
            accepted.append(t)

    for t in accepted:
        for t_end in endings:
            _, max_stress = wing_deflection(stations, False, t, t_end)
            _, max_shear  = wing_torque(stations, False, t, t_end)
            vm = np.sqrt(max_stress**2 + 3 * np.array(max_shear)**2) / 1e6
            if (np.all(max_stress / 1e6 < yield_stress) and
                    np.all(np.abs(max_shear) / 1e6 < yield_stress / 2) and
                    np.all(vm < yield_stress)):
                data_final.append((t, round(t_end, 2)))

    print(data_final)
    return data_final


stations = create_mesh('wings_only_ascii.stl', n_sections_span, True, span_crop=4)
y_values, load_span, xac_uniform, df = read_load_file('transonic_spanwise_loads.csv',
                                                       True, n_sections_span)

#plot_loads_2d(y_values, load_span, df)
plot_loads_on_wing_3d(stations, y_values, load_span)

#wing_deflection(stations, True, 3.5e-3, 1)
#wing_torque(stations, True, 3e-3, 1)

#sensitivity_analysis_bending(stations)
#sensitivity_analysis_torque(stations)
#von_mises(stations, True)
#skin_buckling(stations)
#optimization(stations)

