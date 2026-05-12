# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:33:45 2026

@author: SID-DRW
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Input Data & Constanten ---
W_TOG = 111389.645     # Massa (kg)
S_PLAN = 425.682       # Vleugeloppervlak (m2)
S_WET = 1188.937       # Nat oppervlak (m2)
MAC = 21.0             # Gemiddelde koorde (m)
IF = 1.05              # Interference factor (+5%)
ACCEL_G = 0.15         # Doelversnelling (g)
G = 9.81

# Mach range voor de simulatie
mach_range = np.linspace(1.2, 6.0, 100)

# --- 2. Hulpfuncties ---

def get_atmosphere(alt_m):
    """Berekent rho en T op basis van ISA model tot 40km."""
    if alt_m <= 11000:
        T = 288.15 - 0.0065 * alt_m
        rho = 1.225 * (T / 288.15)**4.256
    elif alt_m <= 25000:
        T = 216.65
        rho = 0.3639 * np.exp(-0.000157 * (alt_m - 11000))
    else:
        T = 216.65 + 0.003 * (alt_m - 25000)
        rho = 0.0401 * (T / 216.65)**-11.388
    rho = 1.225
    T = 293
    return rho, T

def get_reynolds(rho, v, temp, chord):
    """Berekent Reynoldsgetal met Sutherland's viscositeit."""
    mu_0 = 1.7894e-5
    T_0 = 273.15
    S_suth = 110.4
    mu = mu_0 * (temp / T_0)**1.5 * (T_0 + S_suth) / (temp + S_suth)
    return (rho * v * chord) / mu

# --- 3. De Hoofdberekening (Loop door Mach nummers) ---
results = {'cl': [], 'cd': [], 'alpha': [], 'thrust': [], 'ld': [], "cdf": [], 'wave': []}

for M in mach_range:
    # Simulatie van klimprofiel: we klimmen lineair naar 40km
    current_alt = (M - 1.2) * (40000 / (5.0 - 1.2))
    rho, T = get_atmosphere(current_alt)
    a = np.sqrt(1.4 * 287 * T)
    V = M * a
    
    # A. Benodigde CL voor Lift = Gewicht
    cl_needed = (W_TOG * G) / (0.5 * rho * V**2 * S_PLAN)
    
    # B. Bereken Alpha en Wave Drag (Schakelen tussen regimes)
    #if M <= 3.0:
        # Supersonisch (Ackeret)
        #beta = np.sqrt(M**2 - 1)
        #alpha_rad = (cl_needed * beta) / 4
        #cd_wave = (4 * alpha_rad**2) / beta
    #else:
        # Hypersonisch (Newtonian)
        # Cl = 2 * sin^2(alpha) -> benadering alpha = sqrt(Cl/2)
    alpha_rad = np.sqrt(cl_needed / 2)
    cd_wave = 2 * np.sin(alpha_rad)**3
        
    # C. Wrijving (Jouw Schlichting berekening maar dan dynamisch)
    Re_dyn = get_reynolds(rho, V, T, MAC)
    cf = 0.455 / (np.log10(Re_dyn)**2.58)
    cd_f = cf * IF * (S_WET / S_PLAN)
    
    # D. Totaal en Thrust
    cd_total = cd_f + cd_wave
    drag_force = 0.5 * rho * V**2 * S_PLAN * cd_total
    thrust_req = drag_force + (W_TOG * ACCEL_G * G)
    
    # Opslaan
    results['cl'].append(cl_needed)
    results['cd'].append(cd_total)
    results['alpha'].append(np.degrees(alpha_rad))
    results['thrust'].append(thrust_req / 1000) # kN
    results['ld'].append(cl_needed / cd_total)
    results['cdf'].append( 0.5 * rho * V**2 * S_PLAN *cd_f/1000)
    results['wave'].append(0.5 * rho * V**2 * S_PLAN *cd_wave/1000)

# --- 4. Visualisatie ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(mach_range, results['cl'], color='blue', label='CL Needed')
axs[0, 0].set_title('Lift Coëfficiënt voor Cruise')
axs[0, 0].set_ylabel('CL')

axs[0, 1].plot(mach_range, results['alpha'], color='red', label='Alpha')
axs[0, 1].set_title('Benodigde Invalshoek (Alpha)')
axs[0, 1].set_ylabel('Graden')

axs[1, 0].plot(mach_range, results['ld'], color='orange', label='L/D')
axs[1, 0].set_title('Efficiëntie (L/D Ratio)')
axs[1, 0].set_ylabel('L/D')

axs[1, 1].plot(mach_range, results['thrust'], color='green', label='Required Thrust')
axs[1, 1].plot(mach_range, results['cdf'], color='red', label='friction')
axs[1, 1].plot(mach_range, results['wave'], color='yellow', label='wave')
axs[1, 1].set_title('Benodigde Thrust voor 0.15g')
axs[1, 1].set_ylabel('Thrust (kN)')

for ax in axs.flat:
    ax.set_xlabel('Mach Number')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()