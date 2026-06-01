# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:18:46 2026

@author: SID-DRW
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
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Input Data & Constants ---
W_TOG = 111389.645     # Take-off gross weight (kg)
S_PLAN = 425.682       # Planform wing area (m2)
S_WET = 1188.937       # Wetted area (m2)
MAC = 21.0             # Mean Aerodynamic Chord (m)
IF = 1.05              # Interference factor (+5%)
G = 9.81

# Specific transonic inputs
t_over_c = 0.05        # 5% relative thickness (thin wing for transonic/supersonic flight)
sweep_deg = 35.0       # 35 degrees wing sweep
sweep_rad = np.radians(sweep_deg)

# Mach range specifically for Transonic/Low-Supersonic regimes
mach_range = np.linspace(0.5, 2.0, 150)

# --- 2. Auxiliary Functions ---

def get_atmosphere(alt_m):
    """Calculates air density (rho) and temperature (T) using the ISA model up to 40km."""
    if alt_m <= 11000:
        T = 288.15 - 0.0065 * alt_m
        rho = 1.225 * (T / 288.15)**4.256
    elif alt_m <= 25000:
        T = 216.65
        rho = 0.3639 * np.exp(-0.000157 * (alt_m - 11000))
    else:
        T = 216.65 + 0.003 * (alt_m - 25000)
        rho = 0.0401 * (T / 216.65)**-11.388
    return rho, T

def get_reynolds(rho, v, temp, chord):
    """Calculates the Reynolds number using Sutherland's law for viscosity."""
    mu_0 = 1.7894e-5
    T_0 = 273.15
    S_suth = 110.4
    mu = mu_0 * (temp / T_0)**1.5 * (T_0 + S_suth) / (temp + S_suth)
    return (rho * v * chord) / mu

def calc_transonic_wave_drag(M, t_c, sweep):
    """
    Empirical approximation of the transonic wave drag curve (Lock's empirical method/Korn).
    Calculates the drag rise around Mach 1 as a function of thickness and sweep.
    """
    # Estimate the critical Mach number based on wing thickness and sweep
    M_crit = 0.9 - 1.2 * t_c - 0.1 * (1 - np.cos(sweep))
    M_peak = 1.05  # The peak of the drag rise typically occurs just past Mach 1
    
    if M < M_crit:
        return 0.0
    elif M <= M_peak:
        # Drag rises steeply (sine-squared transition)
        amplitude = 20 * (t_c**2.5) * np.cos(sweep)**2 # Estimated peak height
        return amplitude * np.sin((M - M_crit) / (M_peak - M_crit) * (np.pi / 2))**2
    else:
        # Past Mach 1.05, the wave drag coefficient gradually decreases (~ 1/sqrt(M^2-1))
        amplitude_peak = 20 * (t_c**2.5) * np.cos(sweep)**2
        return amplitude_peak / np.sqrt(max(0.1, M**2 - 1.0))

# --- 3. Main Calculation Loop ---
results = {'cl': [], 'cd': [], 'cd_f': [], 'cd_wave': [], 'cd_induced': [], 'ld': []}

# Simulating a realistic transonic cruise altitude (e.g., stable at 11 km)
# Computing transonic drag across varying altitudes would distort the coefficient trends
cruise_alt = 11000 

for M in mach_range:
    rho, T = get_atmosphere(cruise_alt)
    a = np.sqrt(1.4 * 287 * T)
    V = M * a
    
    # Dynamic pressure
    q = 0.5 * rho * V**2
    
    # 1. Required CL for Lift = Weight
    cl_needed = (W_TOG * G) / (q * S_PLAN)
    
    # 2. Skin Friction Drag (Including compressibility correction)
    Re_dyn = get_reynolds(rho, V, T, MAC)
    cf_inc = 0.455 / (np.log10(Re_dyn)**2.58)  # Incompressible (Schlichting)
    cf_comp = cf_inc / (1 + 0.12 * M**2)**0.5  # Compressibility correction
    cd_f = cf_comp * IF * (S_WET / S_PLAN)
    
    # 3. Transonic Wave Drag
    cd_wave = calc_transonic_wave_drag(M, t_over_c, sweep_rad)
    
    # 4. Lift-Induced Drag
    # Aspect Ratio approximation (assuming AR = 7.0 for this class of aircraft)
    AR = 7.0
    e_oswald = 0.85 - 0.02 * M  # Oswald efficiency factor drops slightly in the transonic regime
    cd_induced = (cl_needed**2) / (np.pi * AR * e_oswald)
    
    # Total Drag Coefficient
    cd_total = cd_f + cd_wave + cd_induced
    
    # Save results
    results['cl'].append(cl_needed)
    results['cd'].append(cd_total)
    results['cd_f'].append(cd_f)
    results['cd_wave'].append(cd_wave)
    results['cd_induced'].append(cd_induced)
    results['ld'].append(cl_needed / cd_total)

# --- 4. Visualization ---
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Drag Breakdown
axs[0].plot(mach_range, results['cd'], color='black', linewidth=2.5, label='Total CD')
axs[0].plot(mach_range, results['cd_f'], color='blue', linestyle='--', label='Skin Friction Drag (CD_f)')
axs[0].plot(mach_range, results['cd_wave'], color='red', linestyle='--', label='Transonic Wave Drag')
axs[0].plot(mach_range, results['cd_induced'], color='green', linestyle='--', label='Induced Drag (CD_i)')
axs[0].set_title('Transonic Drag Coefficient Breakdown (Altitude = 11 km)')
axs[0].set_ylabel('Drag Coefficient (CD)')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# Plot 2: Aerodynamic Efficiency
axs[1].plot(mach_range, results['ld'], color='purple', linewidth=2)
axs[1].set_title('Aerodynamic Efficiency (L/D Ratio) Through the Sound Barrier')
axs[1].set_xlabel('Mach Number')
axs[1].set_ylabel('L/D Ratio')
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()