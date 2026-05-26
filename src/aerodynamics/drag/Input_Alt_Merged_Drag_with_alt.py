# -*- coding: utf-8 -*-
"""
Hypersonic aircraft drag estimation: Mach 0.7 - 6.0 at a fixed altitude.

Version 3 changes vs. the two source files:
  1. Wave-drag theories are placed with the correct regime
     (Ackeret  -> supersonic,  Newtonian impact -> hypersonic).
  2. Sigmoid blend between supersonic and hypersonic models is now
     centred at M = 4.0 (was M = 3.0) and softened (k = 4, was 7),
     because Ackeret stays usable to ~M=3-4 and Newtonian only
     starts earning its keep around M >= 4-5.
  3. Compressible skin-friction correction replaces the misapplied
     Prandtl-Glauert 1/sqrt(|1-M^2|) factor, which is a pressure
     correction and is not valid for Cf. The new form stays finite
     through M = 1.
  4. Wetted-area scaling for hypersonic Cf is unified with the
     supersonic branch (cf * IF * S_WET/S_PLAN), removing the
     arbitrary factor of 2.0.
  5. Newtonian inversion alpha = sqrt(CL / 2) corrected
     (the previous CL**0.75 exponent had no physical basis).
"""


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
from common.project_paths import data_file, thrust_map_file, source_file
import numpy as np
import matplotlib.pyplot as plt
import csv

# =============================================================================
# 1. GLOBAL VEHICLE INPUT DATA & CONSTANTS
# =============================================================================
G = 9.81
R_GAS = 287.05
GAMMA = 1.4

W_TOG = 111389.645     # Aircraft gross weight (kg)
S_PLAN = 425.682       # Planform wing area (m^2)
S_WET = 1188.937       # Total wetted area (m^2)
MAC = 21.0             # Mean Aerodynamic Chord (m)
L_REF = 35.0           # Characteristic length for high-speed Reynolds (m)
IF = 1.05              # Interference factor (+5%)
ACCEL_G_TARGET = 0.15  # Target acceleration (g)

# --- USER INPUT: SET YOUR FIXED ALTITUDE HERE ---
fixed_altitude_m = 30000.0

t_over_c = 0.05
sweep_deg = 35.0
sweep_rad = np.radians(sweep_deg)
AR = 7.0

mach_range = np.linspace(0.7, 6.0, 300)

# --- REGIME BLEND CONFIGURATION ---
M_BLEND_CENTER = 4.0   # Centre of the supersonic <-> hypersonic blend
K_BLEND        = 4.0   # Sigmoid steepness (lower = smoother transition)

# =============================================================================
# 2. ATMOSPHERE ENGINE (ISA MODEL)
# =============================================================================
def get_atmosphere(alt_m):
    g0 = 9.80665
    P0 = 101325.0
    T0 = 288.15

    L1 = -0.0065
    h11 = 11000.0
    T11 = T0 + L1 * h11
    P11 = P0 * (T11 / T0) ** (-g0 / (L1 * R_GAS))

    h20 = 20000.0
    T20 = T11
    P20 = P11 * np.exp(-g0 * (h20 - h11) / (R_GAS * T11))
    L3 = 0.0010

    if alt_m <= 11000.0:
        T = T0 + L1 * alt_m
        P = P0 * (T / T0) ** (-g0 / (L1 * R_GAS))
    elif alt_m <= 25000.0:
        T = T11
        P = P11 * np.exp(-g0 * (alt_m - h11) / (R_GAS * T))
    elif alt_m <= 40000.0:
        T = T20 + L3 * (alt_m - h20)
        P = P20 * (T / T20) ** (-g0 / (max(1e-4, L3) * R_GAS))
    else:
        T = 216.65
        P = 1000.0

    rho = P / (R_GAS * T)
    return rho, T


def get_reynolds(rho, v, temp, chord):
    """Sutherland's law for dynamic viscosity, then Re = rho V L / mu."""
    mu_0 = 1.7894e-5
    T_0 = 273.15
    S_suth = 110.4
    mu = mu_0 * (temp / T_0) ** 1.5 * (T_0 + S_suth) / (temp + S_suth)
    return (rho * v * chord) / mu


def compressible_cf(cf_inc, mach):
    """
    Compressibility correction for turbulent flat-plate skin friction.
    Simplified van Driest II / Hopkins-Keener form:
        Cf / Cf_inc = (1 + 0.144 * M^2)^(-0.65)
    This stays finite through M = 1 (unlike Prandtl-Glauert, which is
    a *pressure* correction and is not applicable to Cf).
    """
    return cf_inc / (1.0 + 0.144 * mach ** 2) ** 0.65


# =============================================================================
# 3. PERFORMANCE LOOP
# =============================================================================
results = {
    'mach': [], 'q': [], 'cl': [], 'alpha': [], 'cd': [],
    'cd_f': [], 'cd_wave': [], 'cd_induced': [], 'ld': [],
    'thrust': [], 'regime': []
}

print("\n" + "=" * 115)
print(f"FIXED ALTITUDE PERFORMANCE RUN AT: {fixed_altitude_m/1000:.2f} km")
print(f"Supersonic <-> Hypersonic blend centred at M = {M_BLEND_CENTER}, k = {K_BLEND}")
print("=" * 115)
print(f"{'MACH':<6} | {'q (kPa)':<8} | {'REGIME':<12} | {'C_L':<7} | "
      f"{'AoA(deg)':<8} | {'C_D Total':<9} | {'C_D Friction':<12} | "
      f"{'C_D Wave':<8} | {'L/D':<6} | {'THRUST(kN)':<10}")
print("=" * 115)

last_regime = ""

rho, T = get_atmosphere(fixed_altitude_m)
a = np.sqrt(GAMMA * R_GAS * T)

for idx, M in enumerate(mach_range):
    V = M * a
    q = 0.5 * rho * V ** 2
    q_kpa = q / 1000.0

    cl_needed = (W_TOG * G) / (q * S_PLAN)

    # -------------------------------------------------------------------------
    # AERODYNAMIC REGIME LOGIC
    # -------------------------------------------------------------------------

    # --- TRANSONIC FRONT-HALF (subsonic through M = 1.0) ---
    if M <= 1.0:
        regime_str = "Transonic"
        Re_dyn = get_reynolds(rho, V, T, MAC)
        cf_inc = 0.455 / (np.log10(Re_dyn) ** 2.58)
        cf_comp = compressible_cf(cf_inc, M)
        cd_f = cf_comp * IF * (S_WET / S_PLAN)

        M_crit = 0.9 - 1.2 * t_over_c - 0.1 * (1 - np.cos(sweep_rad))
        M_peak = 1.05
        if M < M_crit:
            cd_wave = 0.0
        else:
            amplitude = 20 * (t_over_c ** 2.5) * np.cos(sweep_rad) ** 2
            cd_wave = amplitude * np.sin(
                (M - M_crit) / (M_peak - M_crit) * (np.pi / 2)
            ) ** 2

        e_oswald = 0.85 - 0.02 * M
        cd_induced = (cl_needed ** 2) / (np.pi * AR * e_oswald)
        alpha_deg = np.degrees(cl_needed / (2 * np.pi * AR / (AR + 2)))

    # --- TRANSONIC BACK-HALF (1.0 < M < 1.2) ---
    elif M < 1.2:
        regime_str = "Transonic"
        Re_dyn = get_reynolds(rho, V, T, MAC)
        cf_inc = 0.455 / (np.log10(Re_dyn) ** 2.58)
        cf_comp = compressible_cf(cf_inc, M)
        cd_f = cf_comp * IF * (S_WET / S_PLAN)

        M_crit = 0.9 - 1.2 * t_over_c - 0.1 * (1 - np.cos(sweep_rad))
        M_peak = 1.05
        if M <= M_peak:
            amplitude = 20 * (t_over_c ** 2.5) * np.cos(sweep_rad) ** 2
            cd_wave = amplitude * np.sin(
                (M - M_crit) / (M_peak - M_crit) * (np.pi / 2)
            ) ** 2
        else:
            amplitude_peak = 20 * (t_over_c ** 2.5) * np.cos(sweep_rad) ** 2
            cd_wave = amplitude_peak / np.sqrt(max(0.1, M ** 2 - 1.0))

        e_oswald = 0.85 - 0.02 * M
        cd_induced = (cl_needed ** 2) / (np.pi * AR * e_oswald)
        alpha_deg = np.degrees(cl_needed / (2 * np.pi * AR / (AR + 2)))

    # --- SUPERSONIC + HYPERSONIC, SIGMOID-BLENDED (M >= 1.2) ---
    else:
        # Regime label tracks the blend midpoint
        if M < M_BLEND_CENTER:
            regime_str = "Supersonic"
        else:
            regime_str = "Hypersonic"

        # ---- (A) Supersonic model: Ackeret linear theory ----
        Re_dyn_super = get_reynolds(rho, V, T, MAC)
        cf_inc_super = 0.455 / (np.log10(Re_dyn_super) ** 2.58)
        cf_super = compressible_cf(cf_inc_super, M)
        cd_f_super = cf_super * IF * (S_WET / S_PLAN)

        cl_alpha_super = 4.0 / np.sqrt(max(0.01, M ** 2 - 1.0))
        alpha_rad_super = cl_needed / cl_alpha_super
        # Ackeret: Cd_wave = CL * alpha = CL^2 * sqrt(M^2-1) / 4
        cd_wave_super = cl_needed * alpha_rad_super

        # ---- (B) Hypersonic model: Newtonian impact theory ----
        Re_dyn_hyper = get_reynolds(rho, V, T, L_REF)
        cf_inc_hyper = 0.074 / (Re_dyn_hyper ** 0.2)
        # Eckert reference-enthalpy compressibility roll-off
        cf_hyper = cf_inc_hyper * ((1.0 / (1.0 + 0.15 * M ** 2)) ** 0.58)
        # Unified wetted-area scaling (was an arbitrary 2.0 before)
        cd_f_hyper = cf_hyper * IF * (S_WET / S_PLAN)

        # Newtonian: CL ~= 2 sin^2(a) cos(a) ~= 2 a^2  -> a = sqrt(CL/2)
        alpha_rad_hyper = np.sqrt(np.abs(cl_needed) / 2.0)
        cd_wave_hyper = 2.0 * np.sin(alpha_rad_hyper) ** 3

        # ---- Sigmoid blend ----
        weight_hyper = 1.0 / (1.0 + np.exp(-K_BLEND * (M - M_BLEND_CENTER)))
        weight_super = 1.0 - weight_hyper

        cd_f = cd_f_super * weight_super + cd_f_hyper * weight_hyper
        cd_wave = cd_wave_super * weight_super + cd_wave_hyper * weight_hyper
        alpha_rad = alpha_rad_super * weight_super + alpha_rad_hyper * weight_hyper

        # Lift-dependent drag is already inside Ackeret's CL*alpha term
        # and inside the Newtonian sin^3 term, so no separate K*CL^2 here.
        cd_induced = 0.0
        alpha_deg = np.degrees(alpha_rad)

    # -------------------------------------------------------------------------
    # TOTALS & LOG OUTPUT
    # -------------------------------------------------------------------------
    cd_total = cd_f + cd_wave + cd_induced
    drag_force = q * S_PLAN * cd_total
    thrust_req = drag_force + (W_TOG * ACCEL_G_TARGET * G)
    ld_ratio = cl_needed / cd_total if cd_total > 0 else 0
    thrust_kn = thrust_req / 1000

    if last_regime != "" and last_regime != regime_str:
        print("-" * 115)
    last_regime = regime_str

    if idx % 4 == 0 or M == 6.0:
        print(f"{M:<6.2f} | {q_kpa:<8.2f} | {regime_str:<12} | "
              f"{cl_needed:<7.4f} | {alpha_deg:<8.2f} | {cd_total:<9.5f} | "
              f"{cd_f:<12.5f} | {cd_wave:<8.5f} | {ld_ratio:<6.2f} | "
              f"{thrust_kn:<10.2f}")

    results['mach'].append(M)
    results['q'].append(q_kpa)
    results['cl'].append(cl_needed)
    results['alpha'].append(alpha_deg)
    results['cd'].append(cd_total)
    results['cd_f'].append(cd_f)
    results['cd_wave'].append(cd_wave)
    results['cd_induced'].append(cd_induced)
    results['ld'].append(ld_ratio)
    results['thrust'].append(thrust_kn)
    results['regime'].append(regime_str)

print("=" * 115 + "\n")

for key in results:
    if key != 'regime':
        results[key] = np.array(results[key])

# =============================================================================
# 4. CSV EXPORT
# =============================================================================
csv_filename = data_file("results_tables", "mach_fixed_altitude_output.csv")
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Mach", "Dynamic_Pressure_kPa", "Regime", "CL", "AoA_deg",
        "CD_Total", "CD_Friction", "CD_Wave", "L_D_Ratio",
        "Thrust_Required_kN"
    ])
    for i in range(len(results['mach'])):
        writer.writerow([
            results['mach'][i], results['q'][i], results['regime'][i],
            results['cl'][i], results['alpha'][i], results['cd'][i],
            results['cd_f'][i], results['cd_wave'][i], results['ld'][i],
            results['thrust'][i]
        ])
print(f"--> SUCCESS: 300-point database exported to '{csv_filename}'\n")

# =============================================================================
# 5. FLIGHT ENVELOPE PLOTS
# =============================================================================
valid_mask = (results['cd'] < 50.0) & (~np.isnan(results['cd']))
thrust_mask = (results['thrust'] < 1e5) & (~np.isnan(results['thrust']))

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Dynamic Pressure Profile
axs[0, 0].plot(results['mach'], results['q'], color='teal', lw=2.5)
axs[0, 0].set_title(f'Dynamic Pressure Profile at Fixed {fixed_altitude_m/1000:.1f} km')
axs[0, 0].set_ylabel('Dynamic Pressure $q$ (kPa)')

# Plot 2: Drag Coefficient Breakdown
axs[0, 1].plot(results['mach'][valid_mask], results['cd'][valid_mask],
               color='black', lw=3, label='Total $C_D$')
axs[0, 1].plot(results['mach'][valid_mask], results['cd_f'][valid_mask],
               color='blue', ls='--', label='Skin Friction ($C_{D,f}$)')
axs[0, 1].plot(results['mach'][valid_mask], results['cd_wave'][valid_mask],
               color='red', ls='--', label='Wave Drag')
axs[0, 1].plot(results['mach'][valid_mask], results['cd_induced'][valid_mask],
               color='green', ls=':', label='Transonic Induced ($C_{D,i}$)')
axs[0, 1].set_title('Continuous Drag Coefficient Breakdown')
axs[0, 1].set_ylabel('$C_D$')
axs[0, 1].legend()

# Plot 3: L/D
axs[1, 0].plot(results['mach'][valid_mask], results['ld'][valid_mask],
               color='darkorange', lw=2.5)
axs[1, 0].set_title('Aerodynamic Efficiency ($L/D$) Envelope')
axs[1, 0].set_ylabel('$C_L / C_D$')

# Plot 4: Thrust
axs[1, 1].plot(results['mach'][thrust_mask], results['thrust'][thrust_mask],
               color='crimson', lw=2.5)
axs[1, 1].set_title('Total Required Thrust (0.15 g)')
axs[1, 1].set_ylabel('Thrust (kN)')

# Regime bands now reflect the M=4 split between Supersonic and Hypersonic
for ax in axs.flat:
    ax.set_xlabel('Mach Number')
    ax.grid(True, alpha=0.3)
    ax.axvspan(0.5, 1.2, alpha=0.05, color='blue',
               label='Transonic' if ax == axs[0, 0] else "")
    ax.axvspan(1.2, M_BLEND_CENTER, alpha=0.05, color='green',
               label='Supersonic' if ax == axs[0, 0] else "")
    ax.axvspan(M_BLEND_CENTER, 6.0, alpha=0.05, color='red',
               label='Hypersonic' if ax == axs[0, 0] else "")

axs[0, 0].legend()
plt.tight_layout()
plt.show()