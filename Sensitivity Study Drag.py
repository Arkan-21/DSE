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
S_PLAN = 425.682       # Planform wing area (m2)
S_WET = 1188.937       # Total wetted area (m2)
MAC = 21.0             # Mean Aerodynamic Chord (m)
L_REF = 35.0           # Characteristic length for high-speed Reynolds calculations (m)
IF = 1.05              # Interference factor (+5%)
ACCEL_G_TARGET = 0.15  # Target acceleration (g)

# --- USER INPUT: SET YOUR FIXED ALTITUDE HERE ---
fixed_altitude_m = 11000.0  

t_over_c = 0.05        
sweep_deg = 35.0       
sweep_rad = np.radians(sweep_deg)
AR = 7.0               

mach_range = np.linspace(0.7, 6.0, 300)

# =============================================================================
# 2. ATMOSPHERE (ISA MODEL)
# =============================================================================
def get_atmosphere(alt_m):
    g0 = 9.80665
    P0 = 101325.0
    T0 = 288.15
    
    L1 = -0.0065
    h11 = 11000.0
    T11 = T0 + L1 * h11
    P11 = P0 * (T11 / T0)**(-g0 / (L1 * R_GAS))
    
    h20 = 20000.0
    T20 = T11
    P20 = P11 * np.exp(-g0 * (h20 - h11) / (R_GAS * T11))
    L3 = 0.0010
    
    if alt_m <= 11000.0:
        T = T0 + L1 * alt_m
        P = P0 * (T / T0)**(-g0 / (L1 * R_GAS))
    elif alt_m <= 25000.0:
        T = T11
        P = P11 * np.exp(-g0 * (alt_m - h11) / (R_GAS * T))
    elif alt_m <= 40000.0:
        T = T20 + L3 * (alt_m - h20)
        P = P20 * (T / T20)**(-g0 / (max(1e-4, L3) * R_GAS))
    else:
        T = 216.65
        P = 1000.0
        
    rho = P / (R_GAS * T)
    return rho, T

def get_reynolds(rho, v, temp, chord):
    mu_0 = 1.7894e-5
    T_0 = 273.15
    S_suth = 110.4
    mu = mu_0 * (temp / T_0)**1.5 * (T_0 + S_suth) / (temp + S_suth)
    return (rho * v * chord) / mu

# =============================================================================
# 3. CONTINUOUS ISO-ATMOSPHERIC PERFORMANCE LOOP
# =============================================================================
results = {
    'mach': [], 'q': [], 'cl': [], 'alpha': [], 'cd': [], 
    'cd_f': [], 'cd_wave': [], 'cd_induced': [], 'ld': [], 'thrust': [], 'regime': []
}

print("\n" + "="*115)
print(f"FIXED ALTITUDE PERFORMANCE RUN AT: {fixed_altitude_m/1000:.2f} km")
print("="*115)
print(f"{'MACH':<6} | {'q (kPa)':<8} | {'REGIME':<12} | {'C_L':<7} | {'AoA(deg)':<8} | {'C_D Total':<9} | {'C_D Friction':<12} | {'C_D Wave':<8} | {'L/D':<6} | {'THRUST(kN)':<10}")
print("="*115)

last_regime = ""

rho, T = get_atmosphere(fixed_altitude_m)
a = np.sqrt(GAMMA * R_GAS * T)

for idx, M in enumerate(mach_range):
    V = M * a
    q = 0.5 * rho * V**2
    q_kpa = q / 1000.0
    
    cl_needed = (W_TOG * G) / (q * S_PLAN)
    
    # -------------------------------------------------------------------------
    # AERODYNAMIC REGIME SWITCHING LOGIC (DIRECT TO NEWTONIAN AT M >= 1.2)
    # -------------------------------------------------------------------------
    
    # Subsonic / Transonic Front-Half
    if M <= 1.0:
        regime_str = "Transonic"
        Re_dyn = get_reynolds(rho, V, T, MAC)
        cf_inc = 0.455 / (np.log10(Re_dyn)**2.58)
        
        cf_comp = cf_inc / np.sqrt(np.abs(1.0 - M**2))
        cd_f = cf_comp * IF * (S_WET / S_PLAN)
        
        M_crit = 0.9 - 1.2 * t_over_c - 0.1 * (1 - np.cos(sweep_rad))
        M_peak = 1.05
        if M < M_crit:
            cd_wave = 0.0
        else:
            amplitude = 20 * (t_over_c**2.5) * np.cos(sweep_rad)**2
            cd_wave = amplitude * np.sin((M - M_crit) / (M_peak - M_crit) * (np.pi / 2))**2
            
        e_oswald = 0.85 - 0.02 * M
        cd_induced = (cl_needed**2) / (np.pi * AR * e_oswald)
        alpha_deg = np.degrees(cl_needed / (2 * np.pi * AR / (AR + 2)))
        
    # Transonic Back-Half
    elif M > 1.0 and M < 1.2:
        regime_str = "Transonic"
        Re_dyn = get_reynolds(rho, V, T, MAC)
        cf_inc = 0.455 / (np.log10(Re_dyn)**2.58)
        
        cf_comp = cf_inc / np.sqrt(np.abs(M**2 - 1.0))
        cd_f = cf_comp * IF * (S_WET / S_PLAN)
        
        M_crit = 0.9 - 1.2 * t_over_c - 0.1 * (1 - np.cos(sweep_rad))
        M_peak = 1.05
        if M <= M_peak:
            amplitude = 20 * (t_over_c**2.5) * np.cos(sweep_rad)**2
            cd_wave = amplitude * np.sin((M - M_crit) / (M_peak - M_crit) * (np.pi / 2))**2
        else:
            amplitude_peak = 20 * (t_over_c**2.5) * np.cos(sweep_rad)**2
            cd_wave = amplitude_peak / np.sqrt(max(0.1, M**2 - 1.0))
            
        e_oswald = 0.85 - 0.02 * M
        cd_induced = (cl_needed**2) / (np.pi * AR * e_oswald)
        alpha_deg = np.degrees(cl_needed / (2 * np.pi * AR / (AR + 2)))

    # --- PURE NEWTONIAN IMPACT REGIME FOR ALL UPPER MACH NUMBERS (M >= 1.2) ---
    else:
        if M < 3.0:
            regime_str = "Supersonic"  
        else:
            regime_str = "Hypersonic"
            
        # Using Newtonian Impact Theory directly starting at Mach 1.2
        Re_dyn_hyper = get_reynolds(rho, V, T, L_REF)
        cf_hyper = (0.074 / (Re_dyn_hyper**(0.2))) * ((1 / (1 + 0.15 * M**2))**0.58)
        cd_f = cf_hyper * 2.0
        
        # Newtonian angle of attack and wave drag generation
        alpha_rad = np.sqrt(np.abs(cl_needed**0.75) / 2)
        cd_wave = 2 * np.sin(alpha_rad)**3
        
        cd_induced = 0.0  
        alpha_deg = np.degrees(alpha_rad)

    # -------------------------------------------------------------------------
    # TOTALS & LOG EXPORT BUFFERING
    # -------------------------------------------------------------------------
    cd_total = cd_f + cd_wave + cd_induced
    drag_force = q * S_PLAN * cd_total
    thrust_req = drag_force + (W_TOG * ACCEL_G_TARGET * G)
    ld_ratio = cl_needed / cd_total if cd_total > 0 else 0
    thrust_kn = thrust_req / 1000

    if last_regime != "" and last_regime != regime_str:
        print("-"*115)
    last_regime = regime_str

    if idx % 4 == 0 or M == 6.0:
        print(f"{M:<6.2f} | {q_kpa:<8.2f} | {regime_str:<12} | {cl_needed:<7.4f} | {alpha_deg:<8.2f} | {cd_total:<9.5f} | {cd_f:<12.5f} | {cd_wave:<8.5f} | {ld_ratio:<6.2f} | {thrust_kn:<10.2f}")

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

print("="*115 + "\n")

for key in results:
    if key != 'regime':
        results[key] = np.array(results[key])

# =============================================================================
# 4. CSV AUTOMATED EXPORT 
# =============================================================================
csv_filename = "mach_fixed_altitude_output.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Mach", "Dynamic_Pressure_kPa", "Regime", "CL", "AoA_deg", "CD_Total", "CD_Friction", "CD_Wave", "L_D_Ratio", "Thrust_Required_kN"])
    for i in range(len(results['mach'])):
        writer.writerow([
            results['mach'][i], results['q'][i], results['regime'][i],
            results['cl'][i], results['alpha'][i], results['cd'][i],
            results['cd_f'][i], results['cd_wave'][i], results['ld'][i],
            results['thrust'][i]
        ])
print(f"--> SUCCESS: Full 300-point database completely exported to: '{csv_filename}'\n")

# =============================================================================
# 5. CONTINUOUS FLIGHT ENVELOPE VISUALIZATION
# =============================================================================
valid_mask = (results['cd'] < 50.0) & (~np.isnan(results['cd']))
alpha_mask = (results['alpha'] < 90.0) & (~np.isnan(results['alpha']))
thrust_mask = (results['thrust'] < 1e5) & (~np.isnan(results['thrust']))

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Dynamic Pressure Profile
axs[0, 0].plot(results['mach'], results['q'], color='teal', lw=2.5)
axs[0, 0].set_title(f'Dynamic Pressure Profile at Fixed {fixed_altitude_m/1000:.1f} km')
axs[0, 0].set_ylabel('Dynamic Pressure $q$ (kPa)')

# Plot 2: Comprehensive Drag Coefficient Breakdown
axs[0, 1].plot(results['mach'][valid_mask], results['cd'][valid_mask], color='black', lw=3, label='Total $C_D$')
axs[0, 1].plot(results['mach'][valid_mask], results['cd_f'][valid_mask], color='blue', ls='--', label='Skin Friction ($C_{D,f}$)')
axs[0, 1].plot(results['mach'][valid_mask], results['cd_wave'][valid_mask], color='red', ls='--', label='Wave Drag')
axs[0, 1].plot(results['mach'][valid_mask], results['cd_induced'][valid_mask], color='green', ls=':', label='Transonic Induced ($C_{D,i}$)')
axs[0, 1].set_title('Continuous Drag Coefficient Breakdown')
axs[0, 1].set_ylabel('$C_D$ Scale')
axs[0, 1].legend()

# Plot 3: Aerodynamic Efficiency (L/D Ratio)
axs[1, 0].plot(results['mach'][valid_mask], results['ld'][valid_mask], color='darkorange', lw=2.5)
axs[1, 0].set_title('Aerodynamic Efficiency ($L/D$ Ratio) Envelope')
axs[1, 0].set_ylabel('$C_L / C_D$')

# Plot 4: Total Mission Thrust Requirements
axs[1, 1].plot(results['mach'][thrust_mask], results['thrust'][thrust_mask], color='crimson', lw=2.5)
axs[1, 1].set_title('Total Required Thrust Configuration (0.15g)')
axs[1, 1].set_ylabel('Thrust (kN)')

for ax in axs.flat:
    ax.set_xlabel('Mach Number')
    ax.grid(True, alpha=0.3)
    ax.axvspan(0.5, 1.2, alpha=0.05, color='blue', label='Transonic' if ax == axs[0,0] else "")
    ax.axvspan(1.2, 3.0, alpha=0.05, color='green', label='Supersonic' if ax == axs[0,0] else "")
    ax.axvspan(3.0, 6.0, alpha=0.05, color='red', label='Hypersonic' if ax == axs[0,0] else "")

axs[0, 0].legend()
plt.tight_layout()
plt.show()

# ================================================================
# 6. MULTI-VARIABLE AERODYNAMIC SENSITIVITY
# ================================================================

def compute_cd_parametric(M, alt_m, delta_T=0.0, weight_kg=W_TOG, s_wet_ratio=S_WET/S_PLAN):
    """
    Evaluates CD for a given Mach number and atmospheric state while allowing
    independent scaling variations for Weight, Wetted Area, Altitude, Density and Temperature.
    """
    # 1. Evaluate baseline atmosphere
    rho_base, T_base = get_atmosphere(alt_m)
    
    # 2. Apply Temperature deviation 
    T = T_base + delta_T
    P = rho_base * R_GAS * T_base 
    rho = P / (R_GAS * T)
    
    a = np.sqrt(GAMMA * R_GAS * T)
    V = M * a
    q = 0.5 * rho * V**2
    
    if q <= 0:
        return np.nan
        
    # Required CL is direct function of weight and dynamic pressure
    cl_needed = (weight_kg * G) / (q * S_PLAN)
    
    if M <= 1.0:
        Re_dyn = get_reynolds(rho, V, T, MAC)
        cf_inc = 0.455 / (np.log10(Re_dyn)**2.58)
        cf_comp = cf_inc / np.sqrt(np.abs(1.0 - M**2))
        cd_f = cf_comp * IF * s_wet_ratio # Applied parametric wetted area ratio
        
        M_crit = 0.9 - 1.2 * t_over_c - 0.1 * (1 - np.cos(sweep_rad))
        M_peak = 1.05
        if M < M_crit:
            cd_wave = 0.0
        else:
            amplitude = 20 * (t_over_c**2.5) * np.cos(sweep_rad)**2
            cd_wave = amplitude * np.sin((M - M_crit) / (M_peak - M_crit) * (np.pi / 2))**2
        cd_induced = (cl_needed**2) / (np.pi * AR * (0.85 - 0.02 * M))
        
    elif M > 1.0 and M < 1.2:
        Re_dyn = get_reynolds(rho, V, T, MAC)
        cf_inc = 0.455 / (np.log10(Re_dyn)**2.58)
        cf_comp = cf_inc / np.sqrt(np.abs(M**2 - 1.0))
        cd_f = cf_comp * IF * s_wet_ratio
        
        M_crit = 0.9 - 1.2 * t_over_c - 0.1 * (1 - np.cos(sweep_rad))
        M_peak = 1.05
        if M <= M_peak:
            amplitude = 20 * (t_over_c**2.5) * np.cos(sweep_rad)**2
            cd_wave = amplitude * np.sin((M - M_crit) / (M_peak - M_crit) * (np.pi / 2))**2
        else:
            amplitude_peak = 20 * (t_over_c**2.5) * np.cos(sweep_rad)**2
            cd_wave = amplitude_peak / np.sqrt(max(0.1, M**2 - 1.0))
        cd_induced = (cl_needed**2) / (np.pi * AR * (0.85 - 0.02 * M))
        
    else: # M >= 1.2 Newtonian Impact Regime
        Re_dyn_hyper = get_reynolds(rho, V, T, L_REF)
        cf_hyper = (0.074 / (Re_dyn_hyper**(0.2))) * ((1 / (1 + 0.15 * M**2))**0.58)
        cd_f = cf_hyper * 2.0 
        
        alpha_rad = np.sqrt(np.abs(cl_needed**0.75) / 2)
        cd_wave = 2 * np.sin(alpha_rad)**3
        cd_induced = 0.0
        
    return cd_f + cd_wave + cd_induced

# --- Parametric Sweep Configurations ---
mach_sweep = np.linspace(0.7, 6.0, 200)
altitudes_requested = [5000.0, 10000.0, 15000.0, 20000.0, 30000.0]
weight_variants = [W_TOG * 0.8, W_TOG, W_TOG * 1.2]
wetted_ratios = [(S_WET/S_PLAN) * 0.8, S_WET/S_PLAN, (S_WET/S_PLAN) * 1.2]
isa_deviations = [-20.0, 0.0, 20.0]

# Isolated density sweep array (kg/m3) from thin stratospheric conditions up to sea level thickness
density_sweep = np.logspace(np.log10(0.01), np.log10(1.2), 200)
mach_targets = [0.8, 1.1, 1.5, 3.0, 5.0]
density_colors = {0.8: 'teal', 1.1: 'darkorange', 1.5: 'crimson', 3.0: 'purple', 5.0: 'black'}

# Clean (1, 2) layout configures plots 1 and 2 directly side by side
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
styles = ['--', '-', ':']

# Plot 1: Altitude Sensitivity (5 to 30 km)
for alt in altitudes_requested:
    cd_profile = [compute_cd_parametric(m, alt) for m in mach_sweep]
    axs[0].plot(mach_sweep, cd_profile, lw=2, label=f'Alt = {alt/1000:.0f} km')
#axs[0].set_title('1. CD Sensitivity to Altitude (Density/Pressure Decay)')
axs[0].legend()

# Plot 2: Weight Sensitivity
for i, w_var in enumerate(weight_variants):
    cd_profile = [compute_cd_parametric(m, fixed_altitude_m, weight_kg=w_var) for m in mach_sweep]
    axs[1].plot(mach_sweep, cd_profile, lw=2, ls=styles[i], label=f'Weight: {w_var/1000:.1f} metric tons')
#axs[1].set_title(f'2. CD Sensitivity to Vehicle Weight at {fixed_altitude_m/1000:.1f} km')
axs[1].legend()

# # Plot 3: Wetted Surface Area Ratio Sensitivity
# for i, r_var in enumerate(wetted_ratios):
#     cd_profile = [compute_cd_parametric(m, fixed_altitude_m, s_wet_ratio=r_var) for m in mach_sweep]
#     axs[1, 0].plot(mach_sweep, cd_profile, lw=2, ls=styles[i], label=f'Swet/Splan Ratio = {r_var:.3f}')
# axs[1, 0].set_title(r'3. CD Sensitivity to Friction Area ($S_{wet}/S_{plan}$)')
# axs[1, 0].legend()

# # Plot 4: Atmospheric Sensitivity (Temperature & Combined Density Shifts)
# for i, dT in enumerate(isa_deviations):
#     cd_profile = [compute_cd_parametric(m, fixed_altitude_m, delta_T=dT) for m in mach_sweep]
#     label_str = f'ISA Standard' if dT == 0 else f'ISA {dT:+.0f}°C (T/$\\rho$ Shift)'
#     axs[1, 1].plot(mach_sweep, cd_profile, lw=2, ls=styles[i], label=label_str)
# axs[1, 1].set_title(f'4. CD Sensitivity to Atmospheric Dev at {fixed_altitude_m/1000:.1f} km')
# axs[1, 1].legend()

# # Plot 5: Pure Isolated Density Sweep
# _, T_ref = get_atmosphere(fixed_altitude_m)
# a_ref = np.sqrt(GAMMA * R_GAS * T_ref)

# for m_val in mach_targets:
#     cd_vs_rho = []
#     V_val = m_val * a_ref
#     for rho_val in density_sweep:
#         if m_val <= 1.0:
#             Re_dyn = get_reynolds(rho_val, V_val, T_ref, MAC)
#             cf_inc = 0.455 / (np.log10(Re_dyn)**2.58)
#             cf_comp = cf_inc / np.sqrt(np.abs(1.0 - m_val**2))
#             cd_f = cf_comp * IF * (S_WET / S_PLAN)
            
#             M_crit = 0.9 - 1.2 * t_over_c - 0.1 * (1 - np.cos(sweep_rad))
#             M_peak = 1.05
#             if m_val < M_crit:
#                 cd_wave = 0.0
#             else:
#                 amplitude = 20 * (t_over_c**2.5) * np.cos(sweep_rad)**2
#                 cd_wave = amplitude * np.sin((m_val - M_crit) / (M_peak - M_crit) * (np.pi / 2))**2
#             cd_induced = (((W_TOG * G) / (0.5 * rho_val * V_val**2 * S_PLAN))**2) / (np.pi * AR * (0.85 - 0.02 * m_val))
            
#         elif m_val > 1.0 and m_val < 1.2:
#             Re_dyn = get_reynolds(rho_val, V_val, T_ref, MAC)
#             cf_inc = 0.455 / (np.log10(Re_dyn)**2.58)
#             cf_comp = cf_inc / np.sqrt(np.abs(m_val**2 - 1.0))
#             cd_f = cf_comp * IF * (S_WET / S_PLAN)
            
#             M_crit = 0.9 - 1.2 * t_over_c - 0.1 * (1 - np.cos(sweep_rad))
#             M_peak = 1.05
#             if m_val <= M_peak:
#                 amplitude = 20 * (t_over_c**2.5) * np.cos(sweep_rad)**2
#                 cd_wave = amplitude * np.sin((m_val - M_crit) / (M_peak - M_crit) * (np.pi / 2))**2
#             else:
#                 amplitude_peak = 20 * (t_over_c**2.5) * np.cos(sweep_rad)**2
#                 cd_wave = amplitude_peak / np.sqrt(max(0.1, m_val**2 - 1.0))
#             cd_induced = (((W_TOG * G) / (0.5 * rho_val * V_val**2 * S_PLAN))**2) / (np.pi * AR * (0.85 - 0.02 * m_val))
            
#         else:
#             Re_dyn_hyper = get_reynolds(rho_val, V_val, T_ref, L_REF)
#             cf_hyper = (0.074 / (Re_dyn_hyper**(0.2))) * ((1 / (1 + 0.15 * m_val**2))**0.58)
#             cd_f = cf_hyper * 2.0
            
#             cl_val = (W_TOG * G) / (0.5 * rho_val * V_val**2 * S_PLAN)
#             alpha_rad = np.sqrt(np.abs(cl_val**0.75) / 2)
#             cd_wave = 2 * np.sin(alpha_rad)**3
#             cd_induced = 0.0
            
#         cd_vs_rho.append(cd_f + cd_wave + cd_induced)
        
#     axs[2, 0].plot(density_sweep, cd_vs_rho, lw=2.5, color=density_colors[m_val], label=f'Mach {m_val:.1f}')

# axs[2, 0].set_title(r'5. CD Sensitivity to Isolated Ambient Air Density ($\rho$)')
# axs[2, 0].set_xlabel(r'Air Density $\rho$ (kg/m³) [Thin Air $\rightarrow$ Thick Air]')
# axs[2, 0].set_xscale('log')
# axs[2, 0].legend()

# Format the 1D subplot axes layout perfectly
for ax in axs:
    ax.set_ylabel(r'Total Drag Coefficient ($C_D$)')
    ax.set_xlabel('Mach Number')
    ax.set_yscale('log')
    ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_sensitivity_2_panel.png', dpi=150, bbox_inches='tight')
plt.show()