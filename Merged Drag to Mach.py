# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:08:28 2026

@author: SID-DRW
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. GLOBAL VEHICLE INPUT DATA & CONSTANTS
# =============================================================================
G = 9.81
R_GAS = 287.05
GAMMA = 1.4

# Shared Geometric and Mass Properties
W_TOG = 111389.645     # Aircraft gross weight (kg)
S_PLAN = 425.682       # Planform wing area (m2)
S_WET = 1188.937       # Total wetted area (m2)
MAC = 21.0             # Mean Aerodynamic Chord (m)
L_REF = 35.0           # Characteristic length for high-speed Reynolds calculations (m)
IF = 1.05              # Interference factor (+5%)
ACCEL_G_TARGET = 0.15  # Target acceleration (g)

# Wing Geometry for Wave Drag Estimation
t_over_c = 0.05        # 5% relative thickness
sweep_deg = 35.0       # 35 degrees wing sweep
sweep_rad = np.radians(sweep_deg)
AR = 7.0               # Standard aspect ratio reference

# Continuous Mach Range from Sub/Transonic up to Hypersonic
mach_range = np.linspace(0.5, 6.0, 300)

# =============================================================================
# 2. FLIGHT PROFILE & ATMOSPHERE ENGINE
# =============================================================================
def get_flight_profile(M):
    """
    Defines a unified, continuous flight profile (Altitude as a function of Mach).
    - Mach 0.5 to 1.2: Cruising/accelerating at a standard 11 km.
    - Mach 1.2 to 5.0: Linear climb from 11 km to 40 km.
    - Mach 5.0 to 6.0: Stratospheric hypersonic cruise/climb cap at 40 km.
    """
    if M < 1.2:
        return 11000.0
    elif M <= 5.0:
        return 11000.0 + (M - 1.2) * ((40000.0 - 11000.0) / (5.0 - 1.2))
    else:
        return 40000.0

def get_atmosphere(alt_m):
    """Calculates air density (rho) and temperature (T) using the ISA model up to 40km."""
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
    """Calculates the Reynolds number using Sutherland's law for dynamic viscosity."""
    mu_0 = 1.7894e-5
    T_0 = 273.15
    S_suth = 110.4
    mu = mu_0 * (temp / T_0)**1.5 * (T_0 + S_suth) / (temp + S_suth)
    return (rho * v * chord) / mu

# =============================================================================
# 3. CONTINUOUS CONTINUUM LOOP
# =============================================================================
results = {
    'mach': [], 'alt': [], 'cl': [], 'alpha': [], 'cd': [], 
    'cd_f': [], 'cd_wave': [], 'cd_induced': [], 'ld': [], 'thrust': []
}

for M in mach_range:
    alt = get_flight_profile(M)
    rho, T = get_atmosphere(alt)
    a = np.sqrt(GAMMA * R_GAS * T)
    V = M * a
    q = 0.5 * rho * V**2
    
    # Core Lift requirement (Lift = Weight)
    cl_needed = (W_TOG * G) / (q * S_PLAN)
    
    # -------------------------------------------------------------------------
    # AERODYNAMIC REGIME SWITCHING LOGIC
    # -------------------------------------------------------------------------
    
    # --- REGIME 1: TRANSONIC / LOW SUPERSONIC (M < 1.2) ---
    if M < 1.2:
        # Skin Friction (Turbulent with basic compressibility correction)
        Re_dyn = get_reynolds(rho, V, T, MAC)
        cf_inc = 0.455 / (np.log10(Re_dyn)**2.58)
        cf_comp = cf_inc / (1 + 0.12 * M**2)**0.5
        cd_f = cf_comp * IF * (S_WET / S_PLAN)
        
        # Empirical Transonic Wave Drag (Lock/Korn Method)
        M_crit = 0.9 - 1.2 * t_over_c - 0.1 * (1 - np.cos(sweep_rad))
        M_peak = 1.05
        if M < M_crit:
            cd_wave = 0.0
        elif M <= M_peak:
            amplitude = 20 * (t_over_c**2.5) * np.cos(sweep_rad)**2
            cd_wave = amplitude * np.sin((M - M_crit) / (M_peak - M_crit) * (np.pi / 2))**2
        else:
            amplitude_peak = 20 * (t_over_c**2.5) * np.cos(sweep_rad)**2
            cd_wave = amplitude_peak / np.sqrt(max(0.1, M**2 - 1.0))
            
        # Induced Drag (Oswald drops inside shock domain)
        e_oswald = 0.85 - 0.02 * M
        cd_induced = (cl_needed**2) / (np.pi * AR * e_oswald)
        
        # Back-calculate effective Alpha (Degrees) using linear lift-slope approximation
        alpha_deg = np.degrees(cl_needed / (2 * np.pi * AR / (AR + 2)))

    # --- REGIME 2: PURE SUPERSONIC (1.2 <= M < 3.0) ---
    elif M < 3.0:
        # Skin Friction (Standard Schlichting boundary layer)
        Re_dyn = get_reynolds(rho, V, T, MAC)
        cf = 0.455 / (np.log10(Re_dyn)**2.58)
        cd_f = cf * IF * (S_WET / S_PLAN)
        
        # Supersonic Wave Drag based on non-linear numerical trend matching
        alpha_rad = np.sqrt((cl_needed**0.75) / 2)
        cd_wave = 2 * np.sin(alpha_rad)**3
        cd_induced = 0.0  # Combined directly within supersonic wave drag formulation
        alpha_deg = np.degrees(alpha_rad)

    # --- REGIME 3: HYPERSONIC EXTRACTION (M >= 3.0) ---
    else:
        # Viscous Skin Friction (High-Speed Flat Plate Reynolds Analogy with Inversion Factor)
        Re_dyn = get_reynolds(rho, V, T, L_REF)
        cf = (0.074 / (Re_dyn**(0.2))) * ((1 / (1 + 0.15 * M**2))**0.58)
        cd_f = cf * 2.0  # Upper + lower reference bounds alignment
        
        # Wave/Pressure Drag from High-Speed Linearized Supersonic Slope Expansion
        cl_alpha = 4.0 / np.sqrt(M**2 - 1.0)
        alpha_rad = cl_needed / cl_alpha
        cd_wave = cl_needed * alpha_rad
        cd_induced = 0.0  # Contained inside pressure domain coupling
        alpha_deg = np.degrees(alpha_rad)

    # -------------------------------------------------------------------------
    # TOTALS & FORCE CALCULATIONS
    # -------------------------------------------------------------------------
    cd_total = cd_f + cd_wave + cd_induced
    drag_force = q * S_PLAN * cd_total
    thrust_req = drag_force + (W_TOG * ACCEL_G_TARGET * G)
    
    # Store Arrays
    results['mach'].append(M)
    results['alt'].append(alt / 1000) # In km
    results['cl'].append(cl_needed)
    results['alpha'].append(alpha_deg)
    results['cd'].append(cd_total)
    results['cd_f'].append(cd_f)
    results['cd_wave'].append(cd_wave)
    results['cd_induced'].append(cd_induced)
    results['ld'].append(cl_needed / cd_total)
    results['thrust'].append(thrust_req / 1000) # In kN

# =============================================================================
# 4. CONTINUOUS FLIGHT ENVELOPE VISUALIZATION
# =============================================================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Combined Flight Profile (Altitude Tracking)
axs[0, 0].plot(results['mach'], results['alt'], color='indigo', lw=2.5)
axs[0, 0].set_title('Continuous Mission Profile (Altitude vs Mach)')
axs[0, 0].set_ylabel('Altitude (km)')
axs[0, 0].axvspan(0.5, 1.2, alpha=0.1, color='blue', label='Transonic')
axs[0, 0].axvspan(1.2, 3.0, alpha=0.1, color='green', label='Supersonic')
axs[0, 0].axvspan(3.0, 6.0, alpha=0.1, color='red', label='Hypersonic')
axs[0, 0].legend()

# Plot 2: Comprehensive Drag Coefficient Breakdown
axs[0, 1].plot(results['mach'], results['cd'], color='black', lw=3, label='Total $C_D$')
axs[0, 1].plot(results['mach'], results['cd_f'], color='blue', ls='--', label='Skin Friction ($C_{D,f}$)')
axs[0, 1].plot(results['mach'], results['cd_wave'], color='red', ls='--', label='Wave / Pressure Drag')
axs[0, 1].plot(results['mach'], results['cd_induced'], color='green', ls=':', label='Transonic Induced ($C_{D,i}$)')
axs[0, 1].set_title('Continuous Drag Coefficient Breakdown')
axs[0, 1].set_ylabel('$C_D$ Scale')
axs[0, 1].legend()

# Plot 3: Aerodynamic Efficiency (L/D Ratio)
axs[1, 0].plot(results['mach'], results['ld'], color='darkorange', lw=2.5)
axs[1, 0].set_title('Aerodynamic Efficiency ($L/D$ Ratio) Envelope')
axs[1, 0].set_ylabel('$C_L / C_D$')

# Plot 4: Total Mission Thrust Requirements
axs[1, 1].plot(results['mach'], results['thrust'], color='crimson', lw=2.5)
axs[1, 1].set_title('Total Required Thrust Configuration (0.15g)')
axs[1, 1].set_ylabel('Thrust (kN)')

# Global layout settings across all frames
for ax in axs.flat:
    ax.set_xlabel('Mach Number')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()