import warnings as warn
warn.formatwarning = lambda msg, *_, **__: f"Warning: {msg}\n"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from isa_atmosphere import T, density
#from drag import drag_and_ramjet_at_condition
from empirical_drag import drag_from_mach_alpha 
from Engine.scramjet_01 import Engine
from thrust_interpolation import thrust_curve_vs_mach

km = lambda x: np.asarray(x) / 1e3            # helper: metres → km

ENGINE_TRANSITION_MACH = {"T2R": 3.0, "R2S": 5.0}  # Mach numbers at which engine transitions occur (for markers on the plot)

def scramjet_thrust(altitude, mach=5.0):
    eng = Engine()
    h = altitude # m
    Ma0  = mach
    mdot = 500.0
    phi  = 0.6

    inp  = eng.inlet_properties(h=h, Ma=Ma0, m_air=mdot)
    iso  = eng.isolator_properties(inp)
    sec2 = eng.combustor_properties2(iso)
    sec3 = eng.combustor_properties3(sec2, phi=phi)
    sec4 = eng.combustor_properties4(sec3)

    if sec4["thermal_choke"]:
        print("\n⚠ THERMAL CHOKE DETECTED in combustor!")
        print(f"  → Last Ma = {sec4['Ma4']:.4f}")
    else:
        sec5 = eng.nozzle_properties(sec4, inp)
        perf = eng.performance(inp, sec5, sec3)

    return perf["Fin"]

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

    a_x_descent, a_y_descent, x_descent = analyse_descent(cruise_cond_end_x, h_cruise, V_cruise, acc_tot)



    return dx_to_cruise, cruise_cond_start_x, cruise_cond_end_x, dv_y_to_cruise, dv_x_to_cruise, V_cruise, a_cruise, h_sample, v_sample, density_sample, a_x_descent, a_y_descent, x_descent

def compute_mach_profile(samples,dx_to_cruise, cruise_cond_start_x, cruise_cond_end_x, x_descent, dv_x_to_cruise, V_cruise, gamma,a_x_descent, a_y_descent, h_cruise, acc_tot=0.15*9.81):
    
    total_range = cruise_cond_end_x + x_descent
    x_profile = np.linspace(0, total_range, samples)
    v_profile = np.zeros_like(x_profile)
    h_profile = np.zeros_like(x_profile)
    M_profile = np.zeros_like(x_profile)
    t_profile = np.zeros_like(x_profile)
    Thrust_profile = np.zeros_like(x_profile)
    Drag_profile = np.zeros_like(x_profile)

    v_desc_max = -1
    t_acc_end = 0
    t_desc_max = 0
    for i, x in enumerate(x_profile):
        if x < dx_to_cruise:
            t = np.sqrt(2 * x / (acc_tot * np.cos(np.radians(gamma))))
            t_profile[i] = t
            v_profile[i] = np.sqrt((acc_tot * np.cos(np.radians(gamma)) * t)**2 + (acc_tot * np.sin(np.radians(gamma)) * t)**2)
            h_profile[i] = 0.5 * acc_tot * np.sin(np.radians(gamma)) * t**2
        elif x < cruise_cond_start_x:
            x_acc = x - dx_to_cruise
            v_profile[i] = np.sqrt(2*acc_tot*x_acc + dv_x_to_cruise**2)
            h_profile[i] = h_cruise
            t_profile[i] = (v_profile[i] - dv_x_to_cruise) / acc_tot
            t_acc_end = t_profile[i]
        elif x < cruise_cond_end_x:
            v_profile[i] = V_cruise
            h_profile[i] = h_cruise
            t_profile[i] = t_acc_end + (x - cruise_cond_start_x) / V_cruise
        else:
            v_x = np.sqrt(max(0, V_cruise**2 + 2*a_x_descent*(x - cruise_cond_end_x)))
            t = (V_cruise - v_x) / -a_x_descent
            t_profile[i] = t_acc_end + (cruise_cond_end_x - cruise_cond_start_x) / V_cruise + t
            h_phase1 = 0.5 * a_y_descent * t**2
            if h_phase1 < h_cruise / 2:
                h_profile[i] = h_cruise - h_phase1
                v_y = a_y_descent * t
                if v_y > v_desc_max:
                    v_desc_max = v_y
                    t_desc_max = t
            else:
                dt = t - t_desc_max
                v_y = max(0.0, v_desc_max - a_y_descent * dt)
                h_profile[i] = max(0.0, h_cruise / 2 - (v_desc_max * dt - 0.5 * a_y_descent * dt**2))

            v_profile[i] = np.sqrt(v_x**2 + v_y**2)
        T_profile = T(h_profile[i])
        a_profile = np.sqrt(1.4 * 287.05 * T_profile)
        M_profile[i] = v_profile[i] / a_profile
        alpha = 3
        S_ref = 425.682
        result = drag_from_mach_alpha(M_profile[i], alpha, h_profile[i], S_ref)
      
        Drag_profile[i] = result["D"]

        if M_profile[i]>=ENGINE_TRANSITION_MACH["R2S"]:
            if Thrust_profile[i-1] == 0:
                Thrust_profile[i] = scramjet_thrust(h_profile[i], M_profile[i])
            else:
                Thrust_profile[i] = Thrust_profile[i-1]
        elif M_profile[i] < ENGINE_TRANSITION_MACH["T2R"]:
            Thrust_profile[i] = thrust_curve_vs_mach("turbo", h_profile[i], np.array([M_profile[i]]))[0]*1000  # convert kN to N
        else:
            Thrust_profile[i] = thrust_curve_vs_mach("ram", h_profile[i], np.array([M_profile[i]]))[0]*1000  # convert kN to N

        '''
        if M_profile[i] < ENGINE_TRANSITION_MACH["R2S"] and M_profile[i] >= ENGINE_TRANSITION_MACH["T2R"]:

            result = drag_and_ramjet_at_condition(
                mass_kg=100_000,
                S_plan=450,
                altitude_m=h_profile[i],
                velocity_m_s=v_profile[i],
                CD0=0.040,
                k=0.171,
                A3_ramjet=2 * 0.7739,
                flight_path_angle_deg=gamma if x < cruise_cond_start_x else 0.0,
            )
            print(result["ramjet_thrust_N"])
            Thrust_profile[i] =  result["ramjet_thrust_N"]
        '''

    print(f"Shape of the thrust profile: {Thrust_profile.shape}")
    return x_profile, v_profile, h_profile, M_profile, t_profile, Thrust_profile, Drag_profile





def run_sensitivity_study(gammas, heights, acc_tot=0.15*9.81, total_range=9500e3, x_sample=-1):
    #clear the results file if it exists
    with open('sensitivity_study_results.csv', 'w') as f:
        f.write("gamma (deg),h_cruise (m),dx_to_cruise (m),cruise_cond_start_x (m),cruise_cond_end_x (m),dv_y_to_cruise (m/s),dv_x_to_cruise (m/s),v_at_h_cruise (m/s),V_cruise (m/s),a_cruise (m/s^2),density_sample (kg/m^3),a_descent (m/s^2),final_total_range (m)\n")
    for gamma in gammas:
        for h_cruise in heights:
            dx_to_cruise, cruise_cond_start_x, cruise_cond_end_x, dv_y_to_cruise, dv_x_to_cruise, V_cruise, a_cruise, h_sample, v_sample, density_sample, a_x_descent, a_y_descent, x_descent = compute_flight_profile(gamma,h_cruise,acc_tot,x_sample)
            v_at_h_cruise = np.sqrt(dv_x_to_cruise**2 + dv_y_to_cruise**2)
            a_descent = np.sqrt(a_x_descent**2 + a_y_descent**2)
            final_total_range = cruise_cond_end_x + x_descent

            with open('sensitivity_study_results.csv', 'a') as f:
               f.write(f"{gamma},{h_cruise},{dx_to_cruise},{cruise_cond_start_x},{cruise_cond_end_x},{dv_y_to_cruise},{dv_x_to_cruise},{v_at_h_cruise},{V_cruise},{a_cruise},{density_sample},{a_descent},{final_total_range}\n")


# ── Plotting helpers ───────────────────────────────────────────────────────────
# Marker styles for the two key events (used by all sweep plots)
_MARKER_H  = dict(marker='^', s=70,  zorder=5)   # cruise height reached
_MARKER_M5 = dict(marker='*', s=120, zorder=5)   # M=5 reached


def _plot_one_profile(ax1, ax2, ax3, ax4, gamma, h_cruise, acc_tot, x_sample, *,
                      color_ascent, color_main, label,
                      main_label='_nolegend_',
                      show_h_marker=True,  h_marker_label=False,
                      show_m5_marker=True, m5_marker_label=False):
    """Draw a single mission profile (altitude + velocity + Mach + Thrust) onto ax1, ax2, ax3, ax4.

    Returns a_cruise so the caller can build the secondary Mach axis.
    """
    (dx_to_cruise, cruise_cond_start_x, cruise_cond_end_x,
     dv_y_to_cruise, dv_x_to_cruise, V_cruise, a_cruise,
     h_sample, v_sample, density_sample,
     a_x_descent, a_y_descent, x_descent) = compute_flight_profile(gamma, h_cruise, acc_tot, x_sample)

    # ── Altitude plot ──────────────────────────────────────────────────────────
    ax1.plot(km([0, dx_to_cruise]),
             km([0, h_cruise]),
             color=color_ascent, lw=1.8, label=label)
    ax1.plot(km([dx_to_cruise, cruise_cond_end_x, cruise_cond_end_x + x_descent]),
             km([h_cruise, h_cruise, 0]),
             color=color_main, lw=1.8, label=main_label)

    if show_h_marker:
        ax1.scatter(km(dx_to_cruise), km(h_cruise),
                    color=color_main, **_MARKER_H,
                    label='Cruise height reached' if h_marker_label else '_nolegend_')
    if show_m5_marker:
        ax1.scatter(km(cruise_cond_start_x), km(h_cruise),
                    color=color_main, **_MARKER_M5,
                    label='M = 5 reached' if m5_marker_label else '_nolegend_')

    # ── Velocity plot ──────────────────────────────────────────────────────────
    _acc_x = acc_tot * np.cos(np.radians(gamma))
    x_asc = np.linspace(0, dx_to_cruise, 200)
    ax2.plot(km(x_asc), acc_tot * np.sqrt(2 * x_asc / _acc_x),
             color=color_ascent, lw=1.8, label=label)
    v_top = np.sqrt(dv_x_to_cruise**2 + dv_y_to_cruise**2)
    ax2.plot(km([dx_to_cruise, dx_to_cruise]), [v_top, dv_x_to_cruise],
             color=color_main, lw=1.2, ls='--')
    x_hacc = np.linspace(dx_to_cruise, cruise_cond_start_x, 200)
    ax2.plot(km(x_hacc), np.sqrt(2 * acc_tot * (x_hacc - dx_to_cruise) + dv_x_to_cruise**2),
             color=color_main, lw=1.8, label=main_label)
    ax2.plot(km([cruise_cond_start_x, cruise_cond_end_x, cruise_cond_end_x + x_descent]),
             [V_cruise, V_cruise, 0],
             color=color_main, lw=1.8)

    if show_h_marker:
        ax2.scatter(km(dx_to_cruise), v_top,
                    color=color_main, **_MARKER_H,
                    label='Cruise height reached' if h_marker_label else '_nolegend_')
    if show_m5_marker:
        ax2.scatter(km(cruise_cond_start_x), V_cruise,
                    color=color_main, **_MARKER_M5,
                    label='M = 5 reached' if m5_marker_label else '_nolegend_')

    if x_sample >= 0:
        ax2.scatter(km(x_sample), v_sample, color='purple', marker='X', s=100, zorder=5, label='Sample point')
        ax1.scatter(km(x_sample), km(h_sample), color='purple', marker='X', s=100, zorder=5)
        print(f"Sample point at x={x_sample} m: altitude={h_sample} m, velocity={v_sample} m/s, density={density_sample} kg/m³")

    # ── Mach plot ──────────────────────────────────────────────────────────────
    x_prof, _, _, M_prof, _ , Thrust_profile, Drag_profile = compute_mach_profile(
        5000, dx_to_cruise, cruise_cond_start_x, cruise_cond_end_x, x_descent,
        dv_x_to_cruise, V_cruise, gamma, a_x_descent, a_y_descent, h_cruise, acc_tot
    )
    mask_asc = x_prof <= dx_to_cruise
    ax3.plot(km(x_prof[mask_asc]),  M_prof[mask_asc],  color=color_ascent, lw=1.8, label=label)
    ax3.plot(km(x_prof[~mask_asc]), M_prof[~mask_asc], color=color_main,   lw=1.8, label=main_label)
    
    if show_h_marker:
        h_idx = min(np.searchsorted(x_prof, dx_to_cruise), len(M_prof) - 1)
        ax3.scatter(km(dx_to_cruise), M_prof[h_idx],
                    color=color_main, **_MARKER_H,
                    label='Cruise height reached' if h_marker_label else '_nolegend_')
    if show_m5_marker:
        ax3.scatter(km(cruise_cond_start_x), 5.0,
                    color=color_main, **_MARKER_M5,
                    label='M = 5 reached' if m5_marker_label else '_nolegend_')
    if x_sample >= 0:
        M_sample = v_sample / np.sqrt(1.4 * 287.05 * T(h_sample))
        ax3.scatter(km(x_sample), M_sample, color='purple', marker='X', s=100, zorder=5)

    # ── Thrust and Drag plot ──────────────────────────────────────────────────
    ax4.plot(km(x_prof[mask_asc]),  Drag_profile[mask_asc],  color=color_main, lw=1.8, label="Drag")
    ax4.plot(km(x_prof[~mask_asc]), Drag_profile[~mask_asc], color=color_main,   lw=1.8, label="Drag")
    if np.any(Thrust_profile > 0):
        ax4.plot(km(x_prof[mask_asc]), Thrust_profile[mask_asc], color=color_ascent, lw=1.8, label='Thrust (N)')
        ax4.plot(km(x_prof[~mask_asc]), Thrust_profile[~mask_asc], color=color_ascent, lw=1.8, label='Thrust (N)')

    
    return a_cruise, Thrust_profile, Drag_profile, x_prof


def _decorate_axes(ax1, ax2, ax3, ax4, total_range):
    """Set the shared axis labels, titles, grid, and minimum-range line."""
    ax1.set_xlabel('Range (km)', fontsize=11)
    ax1.set_ylabel('Altitude (km)', fontsize=11)
    ax1.set_title('Altitude Profile', fontsize=11)
    ax1.grid(True, alpha=0.35)
    ax1.axvline(km(total_range), color='black', lw=1.2, ls='--', label='Minimum required range')

    ax2.set_xlabel('Range (km)', fontsize=11)
    ax2.set_ylabel('Velocity (m/s)', fontsize=11)
    ax2.set_title('Velocity Profile', fontsize=11)
    ax2.grid(True, alpha=0.35)
    ax2.axvline(km(total_range), color='black', lw=1.2, ls='--', label='Minimum required range')

    ax3.set_xlabel('Range (km)', fontsize=11)
    ax3.set_ylabel('Mach number', fontsize=11)
    ax3.set_title('Mach Profile', fontsize=11)
    ax3.grid(True, alpha=0.35)
    ax3.axvline(km(total_range), color='black', lw=1.2, ls='--', label='Minimum required range')
    ax3.axhline(ENGINE_TRANSITION_MACH["T2R"], color='gray', lw=1.0, ls=':', label='T2R transition')
    ax3.axhline(ENGINE_TRANSITION_MACH["R2S"], color='gray', lw=1.0, ls=':', label='R2S transition')
    
    ax4.set_xlabel('Range (km)', fontsize=11)
    ax4.set_ylabel('Thrust/Drag (N)', fontsize=11)
    ax4.set_title('Thrust and Drag Profile', fontsize=11)
    ax4.grid(True, alpha=0.35)

def _finalize_sweep_figure(fig, ax1, ax2, ax3, ax4, a_cruise, save):
    """Add the secondary Mach axis and a deduplicated figure-level legend."""
    # Secondary Mach axis on velocity plot
    ax2b = ax2.twinx()
    ax2b.set_ylim(np.array(ax2.get_ylim()) / a_cruise)
    ax2b.set_yticks([0, 1, 2, 3, 4, 5])
    ax2b.set_yticklabels([f'M {m}' for m in range(6)], fontsize=8)
    ax2b.set_ylabel('Mach number', fontsize=10)

    # Collect handles from all axes, dedupe by label
    handles, labels = [], []
    seen = set()
    for ax in (ax1, ax2, ax3, ax4):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l == '_nolegend_' or l in seen:
                continue
            seen.add(l)
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    if save:
        plt.savefig('mission_profile.png', dpi=300, bbox_inches='tight')


def plot_mission_profile(gammas, h_cruise, acc_tot=0.15*9.81, total_range=9500e3, x_sample=-1, save=False, show=True):

    gammas_is_seq   = isinstance(gammas,   (list, np.ndarray))
    h_cruise_is_seq = isinstance(h_cruise, (list, np.ndarray))

    # ── Both swept → sensitivity study, no plot ───────────────────────────────
    if gammas_is_seq and h_cruise_is_seq:
        run_sensitivity_study(gammas, h_cruise, acc_tot, total_range, x_sample)
        warn.warn("No plot generated for sensitivity study with respect to both gamma and h_cruise. "
                  "Results saved to 'sensitivity_study_results.csv'.")
        return

    # ── Sweep over gamma (fixed h_cruise) or over h_cruise (fixed gamma) ──────
    if gammas_is_seq or h_cruise_is_seq:
        sweep_values = gammas if gammas_is_seq else h_cruise
        n = len(sweep_values)
        blue_shades = plt.cm.Blues(np.linspace(0.4, 0.9, n))
        red_shades  = plt.cm.Reds (np.linspace(0.4, 0.9, n))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
        if gammas_is_seq:
            fig.suptitle(f'Hypersonic Mission Profile  —  M5 Cruise at {h_cruise/1e3:.1f} km',
                         fontsize=13, fontweight='bold')
        else:
            fig.suptitle(f'Hypersonic Mission Profile  —  M5 Cruise gamma {gammas:.1f} degrees',
                         fontsize=13, fontweight='bold')

        a_cruise = None
        # Make a separate plot for the difference between thrust and drag, since it has a very different scale and is more relevant for the single-profile case
        fig2, ax5 = plt.subplots(figsize=(20, 4))
        
        for i, val in enumerate(sweep_values):
            if gammas_is_seq:
                g, h, lbl = val, h_cruise, f'γ = {val:.1f}°'
            else:
                g, h, lbl = gammas, val, f'h_cruise = {val/1e3:.0f} km'
                
                

            a_cruise, Thrust_profile, Drag_profile, x_prof = _plot_one_profile(
                ax1, ax2, ax3, ax4, g, h, acc_tot, x_sample,
                color_ascent=blue_shades[i], color_main=red_shades[i], label=lbl,
                show_h_marker=True,  h_marker_label=(i == 0),
                show_m5_marker=True, m5_marker_label=(i == 0),
            )
       
            plot_excess_thrust(ax5,lbl ,Thrust_profile, Drag_profile, x_prof)

        _decorate_axes(ax1, ax2, ax3, ax4, total_range)
        _finalize_sweep_figure(fig, ax1, ax2, ax3, ax4, a_cruise, save)

        
        
        if show:
            plt.show()
        return

    # ── Single scalar case ────────────────────────────────────────────────────
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 6))
    fig.suptitle(f'Hypersonic Mission Profile  —  M5 Cruise at {h_cruise/1e3:.0f} km',
                 fontsize=13, fontweight='bold')

    _, Thrust_profile, Drag_profile, x_prof = _plot_one_profile(
        ax1, ax2, ax3, ax4, gammas, h_cruise, acc_tot, x_sample,
        color_ascent='blue', color_main='red',
        label='Ascent to cruise height',
        main_label='Horizontal acceleration to M=5 and cruise',
        show_h_marker=False,
        show_m5_marker=True, m5_marker_label=True,
    )

    _decorate_axes(ax1, ax2, ax3, ax4, total_range)

    plot_excess_thrust(ax5, 'Single Profile', Thrust_profile, Drag_profile, x_prof)

    if save:
        plt.savefig('mission_profile.png', dpi=300, bbox_inches='tight')

    if show:
        plt.show()

def plot_excess_thrust(ax5, lbl, Thrust_profile, Drag_profile, x_prof):
    # Make a separate plot for the difference between thrust and drag, since it has a very different scale and is more relevant for the single-profile case
    
    ax5.plot(km(x_prof), Thrust_profile - Drag_profile, lw=1.8, label=lbl)
    ax5.set_xlabel('Range (km)', fontsize=11)
    ax5.set_ylabel('Net Force (N)', fontsize=11)
    ax5.set_title('Net Force Profile', fontsize=11)
    ax5.grid(True, alpha=0.35)
    ax5.axhline(0, color='black', lw=1.2, ls='--')
    ax5.legend()
    plt.tight_layout()


def analyse_descent(end_cruise,h_cruise, v_cruise, acc_tot=0.15*9.81, total_range=9500e3):

    # print all the inputs for debugging
    
    a_x_descent = -v_cruise**2/2/(total_range - end_cruise)
    t_descent = v_cruise / -a_x_descent
    a_y_descent = h_cruise / 0.5/(t_descent / 2)**2

    a_descent = np.sqrt(a_x_descent**2 + a_y_descent**2)

    if a_descent > acc_tot:
        warn.warn(f"Descent acceleration {a_descent:.2f} m/s² exceeds comfortable acceleration {acc_tot} m/s². Descent may not be feasible within the given range.\n")

        a_x_descent, a_y_descent, x_descent, _ = find_feasible_descent_acceleration(h_cruise, v_cruise, acc_tot)
        

     
        return a_x_descent, a_y_descent, x_descent
    
    return a_x_descent, a_y_descent, total_range - end_cruise


def descent_range(v_cruise, a_x):
    return v_cruise**2 / (2 * -a_x)

def land_condition(h_cruise, v_cruise,a_x, acc_tot=0.15*9.81):
    return a_x**2 + h_cruise**2/(0.5*v_cruise/a_x)**4 - acc_tot**2

from scipy.optimize import minimize

def find_feasible_descent_acceleration(h_cruise, v_cruise, acc_tot=0.15*9.81):
    # Decision variable: a_x (scalar, packed as length-1 array)
    #plot_descent_feasibility(h_cruise, v_cruise, acc_tot)
    objective = lambda x: descent_range(v_cruise, x[0])

    constraints = [{
        'type': 'ineq',
        'fun': lambda x: -land_condition(h_cruise, v_cruise, x[0], acc_tot),
    }]

    # a_x is strictly negative; |a_x| can't exceed the total accel budget
    bounds = [(-acc_tot, -1e-6)]

    # Start somewhere feasible and well inside the bounds
    x0 = [-0.5 * acc_tot]

    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    a_x_opt = result.x[0]
    a_y_opt = h_cruise/(v_cruise/a_x_opt/2)**2
    x_descent_opt = descent_range(v_cruise, a_x_opt)
    return a_x_opt,a_y_opt, x_descent_opt, result

def plot_descent_feasibility(h_cruise, v_cruise, acc_tot=0.15*9.81):
    a_x_vals = np.linspace(-acc_tot, -1e-6, 100)
    land_vals = [land_condition(h_cruise, v_cruise, a_x, acc_tot) for a_x in a_x_vals]
    plt.figure(figsize=(8, 5))
    plt.plot(a_x_vals, land_vals, label='Land condition (should be ≤ 0 for feasibility)')
    plt.axhline(0, color='red', linestyle='--', label='Feasibility boundary')
    plt.xlabel('Descent horizontal acceleration a_x (m/s²)')
    plt.ylabel('Land condition value')
    plt.title(f'Descent Feasibility for h_cruise={h_cruise/1e3:.1f} km, v_cruise={v_cruise:.1f} m/s')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    

if __name__ == "__main__":
    gammas = [5, 10, 15]  # flight path angles in degrees
    heights = [25e3, 30e3, 35e3]  # cruise altitudes in metres
    
    x = 10e3  # 10 km
    plot_mission_profile(gammas, heights[0],save=False, show=True)

    plot_mission_profile(gammas[0], heights, save=False, show=True)

    plot_mission_profile(gammas, heights, save=False, show=True)    

    plot_mission_profile(gammas[1], heights[1], save=False, show=True)


    