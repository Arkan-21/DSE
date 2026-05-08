import warnings as warn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from isa_atmosphere import T, density

km = lambda x: np.asarray(x) / 1e3            # helper: metres → km

def compute_flight_profile(gamma,h_cruise,acc_tot=0.15*9.81,x_sample=-1):

    gamma_rad = np.radians(gamma)
    a_cruise = np.sqrt(1.4 * 287.05 * T(h_cruise))
    V_cruise = 5 * a_cruise  # M=5 at cruise height
    cruise_range = 90 * 60 * V_cruise  # 90 minutes at cruise speed

    # Ascent to the cruise heigh 
    acc_x   = acc_tot * np.cos(gamma_rad)
    acc_y   = acc_tot * np.sin(gamma_rad)
    dv_y_to_cruise = np.sqrt(2 * acc_y * h_cruise) # this is assumed to be quickly damped by drag after reaching cruise height
    t_to_cruise = dv_y_to_cruise / acc_y
    dv_x_to_cruise = acc_x * t_to_cruise
    dx_to_cruise = dv_x_to_cruise**2 / (2 * acc_x)
    # Horizontal acceleration to M=5 after reaching cruise height
    dx_hor_acc   = (V_cruise**2 - dv_x_to_cruise**2) / (2 * acc_tot)

    cruise_cond_start_x = dx_to_cruise + dx_hor_acc
    cruise_cond_end_x = cruise_cond_start_x + cruise_range

    if x_sample >= 0:
        if x_sample < dx_to_cruise:
            # Sample point is in the ascent phase
            t_sample = np.sqrt(2 * x_sample / acc_x)
            dv_y_sample = acc_y * t_sample
            dv_x_sample = acc_x * t_sample
            v_sample = np.sqrt(dv_x_sample**2 + dv_y_sample**2)
            h_sample = dv_y_sample**2 / (2 * acc_y)
            density_sample = density(h_sample)
        elif x_sample < cruise_cond_start_x:
            # Sample point is in the horizontal acceleration phase
            x_acc = x_sample - dx_to_cruise
            v_sample = np.sqrt(2*acc_tot*x_acc + dv_x_to_cruise**2)
            h_sample = h_cruise  # altitude remains constant during horizontal acceleration
            density_sample = density(h_sample)
        else:
            # Sample point is in the cruise phase
            v_sample = V_cruise
            h_sample = h_cruise  # altitude remains constant during cruise
            density_sample = density(h_sample)
    else:
        h_sample = None
        v_sample = None
        density_sample = None

    return dx_to_cruise, cruise_cond_start_x, cruise_cond_end_x, dv_y_to_cruise, dv_x_to_cruise, V_cruise, a_cruise, h_sample, v_sample, density_sample


def run_sensitivity_study(gammas, heights, acc_tot=0.15*9.81, total_range=9500e3, x_sample=-1):
    for gamma in gammas:
        for h_cruise in heights:
            dx_to_cruise, cruise_cond_start_x, cruise_cond_end_x, dv_y_to_cruise, dv_x_to_cruise, V_cruise, a_cruise, h_sample, v_sample, density_sample = compute_flight_profile(gamma,h_cruise,acc_tot,x_sample)
            v_at_h_cruise = np.sqrt(dv_x_to_cruise**2 + dv_y_to_cruise**2)

            with open('sensitivity_study_results.csv', 'a') as f:
                if f.tell() == 0:  # if file is empty, write header
                    f.write("gamma (deg),h_cruise (m),dx_to_cruise (m),cruise_cond_start_x (m),cruise_cond_end_x (m),dv_y_to_cruise (m/s),dv_x_to_cruise (m/s),v_at_h_cruise (m/s),V_cruise (m/s),a_cruise (m/s^2),density_sample (kg/m^3), velocity_sample(m/s)\n")
                f.write(f"{gamma},{h_cruise},{dx_to_cruise},{cruise_cond_start_x},{cruise_cond_end_x},{dv_y_to_cruise},{dv_x_to_cruise},{v_at_h_cruise},{V_cruise},{a_cruise},{density_sample},{v_sample}\n")

def plot_mission_profile(gammas, h_cruise, acc_tot=0.15*9.81, total_range=9500e3, x_sample =-1,save=False, show=True):

    # Check for sensitivity study with respect to gamma (gammas is an array)
    if isinstance(gammas, (list, np.ndarray)):
        if isinstance(h_cruise, (list, np.ndarray)):
            run_sensitivity_study(gammas, h_cruise, acc_tot, total_range, x_sample)

            warn.warn("No plot generated for sensitivity study with respect to both gamma and h_cruise. Results saved to 'sensitivity_study_results.csv'.")
            return
        
        # ── Colour ramps ──────────────────────────────────────────────────────────────
        n           = len(gammas)
        blue_shades = plt.cm.Blues(np.linspace(0.4, 0.9, n))
        red_shades  = plt.cm.Reds (np.linspace(0.4, 0.9, n))

        # Marker styles for the two key events
        MARKER_H   = dict(marker='^', s=70, zorder=5)   # cruise height reached
        MARKER_M5  = dict(marker='*', s=120, zorder=5)  # M=5 reached

        # ── Figure ────────────────────────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Hypersonic Mission Profile  —  M5 Cruise at {h_cruise/1e3:.1f} km',
                    fontsize=13, fontweight='bold')

        # Track whether we've already added the markup labels to the legend
        _legend_h_added  = False
        _legend_m5_added = False

        for i, gamma in enumerate(gammas):
            dx_to_cruise, cruise_cond_start_x, cruise_cond_end_x, dv_y_to_cruise, dv_x_to_cruise, V_cruise, a_cruise, h_sample, v_sample, density_sample = compute_flight_profile(gamma,h_cruise,acc_tot,x_sample)
            lbl = f'γ = {gamma:.1f}°'
            # ── Altitude plot ──────────────────────────────────────────────────
            ax1.plot(km([0, dx_to_cruise]),
                        km([0, h_cruise]),
                        color=blue_shades[i], lw=1.8, label=lbl)
            ax1.plot(km([dx_to_cruise,
                            cruise_cond_end_x,
                            total_range]),
                        km([h_cruise, h_cruise, 0]),
                        color=red_shades[i], lw=1.8)

            # ▲ Cruise-height marker
            mkw_h = dict(color=red_shades[i], **MARKER_H,
                            label='Cruise height reached' if not _legend_h_added else '_nolegend_')
            ax1.scatter(km(dx_to_cruise), km(h_cruise), **mkw_h)
            
            # ★ M=5 marker
            mkw_m5 = dict(color=red_shades[i], **MARKER_M5,
                            label='M = 5 reached' if not _legend_m5_added else '_nolegend_')
            ax1.scatter(km(cruise_cond_start_x), km(h_cruise), **mkw_m5)
            
            # ── Velocity plot ──────────────────────────────────────────────────
            _acc_x = acc_tot * np.cos(np.radians(gamma))
            x_asc = np.linspace(0, dx_to_cruise, 200)
            ax2.plot(km(x_asc), acc_tot * np.sqrt(2 * x_asc / _acc_x),
                        color=blue_shades[i], lw=1.8, label=lbl)
            v_top = np.sqrt(dv_x_to_cruise**2 + dv_y_to_cruise**2)
            ax2.plot(km([dx_to_cruise, dx_to_cruise]), [v_top, dv_x_to_cruise],
                        color=red_shades[i], lw=1.2, ls='--')
            x_hacc = np.linspace(dx_to_cruise, cruise_cond_start_x, 200)
            ax2.plot(km(x_hacc), np.sqrt(2 * acc_tot * (x_hacc - dx_to_cruise) + dv_x_to_cruise**2),
                        color=red_shades[i], lw=1.8)
            ax2.plot(km([cruise_cond_start_x, cruise_cond_end_x, total_range]),
                        [V_cruise, V_cruise, 0],
                        color=red_shades[i], lw=1.8)

            ax2.scatter(km(dx_to_cruise), np.sqrt(dv_x_to_cruise**2 + dv_y_to_cruise**2), **dict(color=red_shades[i], **MARKER_H,
                        label='Cruise height reached' if not _legend_h_added else '_nolegend_'))
            
            ax2.scatter(km(cruise_cond_start_x), V_cruise, **dict(color=red_shades[i], **MARKER_M5,
                        label='M = 5 reached' if not _legend_m5_added else '_nolegend_'))
            
            if x_sample >= 0:
                ax2.scatter(km(x_sample), v_sample, color='purple', marker='X', s=100, zorder=5, label='Sample point')
                ax1.scatter(km(x_sample), km(h_sample), color='purple', marker='X', s=100, zorder=5)  # same sample point on altitude plot
                print(f"Sample point at x={x_sample} m: altitude={h_sample} m, velocity={v_sample} m/s, density={density_sample} kg/m³")
            
            _legend_h_added  = True
            _legend_m5_added = True

        # ── Altitude plot: shared decorations ────────────────────────────────────────
        ax1.set_xlabel('Range (km)', fontsize=11)
        ax1.set_ylabel('Altitude (km)', fontsize=11)
        ax1.set_title('Altitude Profile', fontsize=11)
        ax1.grid(True, alpha=0.35)


        # ── Velocity plot: shared decorations ────────────────────────────────────────
        ax2.set_xlabel('Range (km)', fontsize=11)
        ax2.set_ylabel('Velocity (m/s)', fontsize=11)
        ax2.set_title('Velocity Profile', fontsize=11)
        ax2.grid(True, alpha=0.35)

        # Secondary Mach axis
        ax2b = ax2.twinx()
        ax2b.set_ylim(np.array(ax2.get_ylim()) / a_cruise)
        ax2b.set_yticks([0, 1, 2, 3, 4, 5])
        ax2b.set_yticklabels([f'M {m}' for m in range(6)], fontsize=8)
        ax2b.set_ylabel('Mach number', fontsize=10)

        # collect handles from both axes, deduplicate by label
        handles, labels = [], []
        seen = set()
        for ax in (ax1, ax2):
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)

        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15)

        if save:
            plt.savefig('mission_profile.png', dpi=300, bbox_inches='tight')
        
        if show: 
            plt.show()
    

    elif isinstance(h_cruise, (list, np.ndarray)): # h_cruise sensitivity study

        # ── Colour ramps ──────────────────────────────────────────────────────────────
        n           = len(h_cruise)
        blue_shades = plt.cm.Blues(np.linspace(0.4, 0.9, n))
        red_shades  = plt.cm.Reds (np.linspace(0.4, 0.9, n))

        # Marker styles for the two key events
        MARKER_H   = dict(marker='^', s=70, zorder=5)   # cruise height reached
        MARKER_M5  = dict(marker='*', s=120, zorder=5)  # M=5 reached

        # ── Figure ────────────────────────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Hypersonic Mission Profile  —  M5 Cruise gamma {gammas:.1f} degrees',
                    fontsize=13, fontweight='bold')

        # Track whether we've already added the markup labels to the legend
        _legend_h_added  = False
        _legend_m5_added = False

        for i, h in enumerate(h_cruise):
            lbl = f'h_cruise = {h/1e3:.0f} km'
            dx_to_cruise, cruise_cond_start_x, cruise_cond_end_x, dv_y_to_cruise, dv_x_to_cruise, V_cruise, a_cruise, h_sample, v_sample, density_sample = compute_flight_profile(gammas, h, acc_tot, x_sample)
            # ── Altitude plot ──────────────────────────────────────────────────
            ax1.plot(km([0, dx_to_cruise]),
                        km([0, h]),
                        color=blue_shades[i], lw=1.8, label=lbl)
            ax1.plot(km([dx_to_cruise,
                            cruise_cond_end_x,
                            total_range]),
                        km([h, h, 0]),
                        color=red_shades[i], lw=1.8)

            # ▲ Cruise-height marker
            mkw_h = dict(color=red_shades[i], **MARKER_H,
                            label='Cruise height reached' if not _legend_h_added else '_nolegend_')
            ax1.scatter(km(dx_to_cruise), km(h), **mkw_h)

            # ★ M=5 marker
            mkw_m5 = dict(color=red_shades[i], **MARKER_M5,
                            label='M = 5 reached' if not _legend_m5_added else '_nolegend_')
            ax1.scatter(km(cruise_cond_start_x), km(h), **mkw_m5)

            # ── Velocity plot ──────────────────────────────────────────────────
            _acc_x = acc_tot * np.cos(np.radians(gammas))
            x_asc = np.linspace(0, dx_to_cruise, 200)
            ax2.plot(km(x_asc), acc_tot * np.sqrt(2 * x_asc / _acc_x),
                        color=blue_shades[i], lw=1.8, label=lbl)
            v_top = np.sqrt(dv_x_to_cruise**2 + dv_y_to_cruise**2)
            ax2.plot(km([dx_to_cruise, dx_to_cruise]), [v_top, dv_x_to_cruise],
                        color=red_shades[i], lw=1.2, ls='--')
            x_hacc = np.linspace(dx_to_cruise, cruise_cond_start_x, 200)
            ax2.plot(km(x_hacc), np.sqrt(2 * acc_tot * (x_hacc - dx_to_cruise) + dv_x_to_cruise**2),
                        color=red_shades[i], lw=1.8)
            ax2.plot(km([cruise_cond_start_x, cruise_cond_end_x, total_range]),
                        [V_cruise, V_cruise, 0],
                        color=red_shades[i], lw=1.8)

            ax2.scatter(km(dx_to_cruise), np.sqrt(dv_x_to_cruise**2 + dv_y_to_cruise**2), **dict(color=red_shades[i], **MARKER_H,
                        label='Cruise height reached' if not _legend_h_added else '_nolegend_'))

            ax2.scatter(km(cruise_cond_start_x), V_cruise, **dict(color=red_shades[i], **MARKER_M5,
                        label='M = 5 reached' if not _legend_m5_added else '_nolegend_'))
            
            if x_sample >= 0:
                ax2.scatter(km(x_sample), v_sample, color='purple', marker='X', s=100, zorder=5, label='Sample point')
                ax1.scatter(km(x_sample), km(h_sample), color='purple', marker='X', s=100, zorder=5)  # same sample point on altitude plot
                print(f"Sample point at x={x_sample} m: altitude={h_sample} m, velocity={v_sample} m/s, density={density_sample} kg/m³")

            _legend_h_added  = True
            _legend_m5_added = True

        # ── Altitude plot: shared decorations ────────────────────────────────────────
        ax1.set_xlabel('Range (km)', fontsize=11)
        ax1.set_ylabel('Altitude (km)', fontsize=11)
        ax1.set_title('Altitude Profile', fontsize=11)
        ax1.grid(True, alpha=0.35)

        # ── Velocity plot: shared decorations ────────────────────────────────────────
        ax2.set_xlabel('Range (km)', fontsize=11)
        ax2.set_ylabel('Velocity (m/s)', fontsize=11)
        ax2.set_title('Velocity Profile', fontsize=11)
        ax2.grid(True, alpha=0.35)

        # Secondary Mach axis
        ax2b = ax2.twinx()
        ax2b.set_ylim(np.array(ax2.get_ylim()) / a_cruise)
        ax2b.set_yticks([0, 1, 2, 3, 4, 5])
        ax2b.set_yticklabels([f'M {m}' for m in range(6)], fontsize=8)
        ax2b.set_ylabel('Mach number', fontsize=10)

        # collect handles from both axes, deduplicate by label
        handles, labels = [], []
        seen = set()
        for ax in (ax1, ax2):
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)

        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15)

        if save:
            plt.savefig('mission_profile.png', dpi=300, bbox_inches='tight')

        if show:
            plt.show()
    

    else:
        dx_to_cruise, cruise_cond_start_x, cruise_cond_end_x, dv_y_to_cruise, dv_x_to_cruise, V_cruise, a_cruise, h_sample, v_sample, density_sample = compute_flight_profile(gammas, h_cruise, acc_tot, x_sample)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Hypersonic Mission Profile  —  M5 Cruise at {h_cruise/1e3:.0f} km',
                    fontsize=13, fontweight='bold')
        ax1.plot(km([0, dx_to_cruise]),
                    km([0, h_cruise]),
                    color='blue', lw=1.8, label='Ascent to cruise height')
        ax1.plot(km([dx_to_cruise,
                        cruise_cond_start_x,
                        cruise_cond_end_x,
                        total_range]),
                    km([h_cruise, h_cruise, h_cruise, 0]),
                    color='red', lw=1.8, label='Horizontal acceleration to M=5 and cruise')
        ax1.scatter(km(cruise_cond_start_x), km(h_cruise), color='red', marker='*', s=120, zorder=5, label='M = 5 reached')
        ax1.set_xlabel('Range (km)', fontsize=11)
        ax1.set_ylabel('Altitude (km)', fontsize=11)
        ax1.set_title('Altitude Profile', fontsize=11)
        ax1.grid(True, alpha=0.35)

        _acc_x = acc_tot * np.cos(np.radians(gammas))
        x_asc = np.linspace(0, dx_to_cruise, 200)
        ax2.plot(km(x_asc), acc_tot * np.sqrt(2 * x_asc / _acc_x), color='blue', lw=1.8, label='Ascent to cruise height')
        v_top = np.sqrt(dv_x_to_cruise**2 + dv_y_to_cruise**2)
        ax2.plot(km([dx_to_cruise, dx_to_cruise]), [v_top, dv_x_to_cruise], color='red', lw=1.2, ls='--')
        x_hacc = np.linspace(dx_to_cruise, cruise_cond_start_x, 200)
        ax2.plot(km(x_hacc), np.sqrt(2 * acc_tot * (x_hacc - dx_to_cruise) + dv_x_to_cruise**2),
                    color='red', lw=1.8, label='Horizontal acceleration to M=5 and cruise')
        ax2.plot(km([cruise_cond_start_x, cruise_cond_end_x, total_range]),
                    [V_cruise, V_cruise, 0], color='red', lw=1.8)
        ax2.scatter(km(cruise_cond_start_x), V_cruise, color='red', marker='*', s=120, zorder=5, label='M = 5 reached')
        ax2.set_xlabel('Range (km)', fontsize=11)
        ax2.set_ylabel('Velocity (m/s)', fontsize=11)
        ax2.set_title('Velocity Profile', fontsize=11)
        ax2.grid(True, alpha=0.35)

        if x_sample >= 0:
            ax2.scatter(km(x_sample), v_sample, color='purple', marker='X', s=100, zorder=5, label='Sample point')
            ax1.scatter(km(x_sample), km(h_sample), color='purple', marker='X', s=100, zorder=5)  # same sample point on altitude plot
            print(f"Sample point at x={x_sample} m: altitude={h_sample} m, velocity={v_sample} m/s, density={density_sample} kg/m³")

        if save:
            plt.savefig('mission_profile.png', dpi=300, bbox_inches='tight')

        if show:
            plt.show()


if __name__ == "__main__":
    gammas = [5, 10, 15]  # flight path angles in degrees
    heights = [25e3, 30e3, 35e3]  # cruise altitudes in metres
    
    x = 10e3  # 10 km
    plot_mission_profile(gammas, heights[1], x_sample=x,save=False, show=False)

    plot_mission_profile(gammas[0], heights,x_sample=x, save=False, show=False)

    plot_mission_profile(gammas, heights, x_sample=x, save=False, show=False)

    plot_mission_profile(gammas[1], heights[1], x_sample=x, save=False, show=False)
    

    