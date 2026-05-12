# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:06:19 2026

@author: SID-DRW
"""

import numpy as np

def reynolds_calculator(altitude_m, mach, chord_length):
    """
    Berekent Reynoldsgetal, Snelheid en Atmosferische data tot 40km hoogte.
    """
    # --- 1. Atmosfeer Model (ISA) ---
    g = 9.80665
    R = 287.05
    
    if altitude_m <= 11000:  # Troposfeer
        T = 288.15 - 0.0065 * altitude_m
        P = 101325 * (T / 288.15)**5.2559
    elif altitude_m <= 20000:  # Lagere Stratosfeer
        T = 216.65
        P = 22632 * np.exp(-g * (altitude_m - 11000) / (R * T))
    elif altitude_m <= 32000:  # Middel Stratosfeer
        T = 216.65 + 0.001 * (altitude_m - 20000)
        P = 5474.9 * (T / 216.65)**-34.163
    else:  # Boven Stratosfeer (tot 47km)
        T = 228.65 + 0.0028 * (altitude_m - 32000)
        P = 868.02 * (T / 228.65)**-12.201

    rho = P / (R * T)
    
    # --- 2. Snelheid ---
    gamma = 1.4
    a = np.sqrt(gamma * R * T)  # Geluidssnelheid
    velocity = mach * a
    
    # --- 3. Viscositeit (Sutherland's Law) ---
    mu_0 = 1.7894e-5
    T_0 = 273.15
    S = 110.4
    mu = mu_0 * (T / T_0)**1.5 * (T_0 + S) / (T + S)
    
    # --- 4. Reynoldsgetal ---
    re = (rho * velocity * chord_length) / mu
    
    return {
        "Altitude (m)": altitude_m,
        "Mach": mach,
        "Velocity (m/s)": round(velocity, 2),
        "Velocity (km/h)": round(velocity * 3.6, 2),
        "Density (kg/m3)": round(rho, 5),
        "Temperature (K)": round(T, 2),
        "Reynolds Number": f"{re:.2e}"
    }

# --- TEST RUN ---
# Gegevens van jouw toestel:
mac = 21.0  # Je Mean Aerodynamic Chord in meters
altitudes = [0, 10000, 20000, 30000, 40000]
target_mach = 5.0

print(f"{'Hoogte (m)':<12} | {'Mach':<6} | {'Snelheid (m/s)':<15} | {'Reynolds':<12}")
print("-" * 55)

for alt in altitudes:
    res = reynolds_calculator(alt, target_mach, mac)
    print(f"{res['Altitude (m)']:<12} | {res['Mach']:<6} | {res['Velocity (m/s)']:<15} | {res['Reynolds Number']:<12}")