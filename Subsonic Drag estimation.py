# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:14:21 2026

@author: SID-DRW
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Invoergegevens ---
s_plan = 425.682          # Referentieoppervlakte (m2) - pas dit aan naar jouw waarde
s_wet = 1188.937          # Totaal nat oppervlak (m2) - pas dit aan naar jouw waarde
cd_0 = 0.00567          # Jouw berekende cd_friction (gebaseerd op S_plan)
ar = 1.2                # Aspect Ratio schatting voor hypersonic aircraft
e = 0.65                # Oswald efficiency factor (typisch voor delta wings)
rho = 1.225             # Luchtdichtheid op zeeniveau (kg/m3)
v_takeoff = 100.0       # Snelheid in m/s (~360 km/h)
mass = 200000.0         # Massa van het toestel in kg (voorbeeldwaarde)
g = 9.81                # Zwaartekrachtversnelling (m/s2)

# --- 2. Berekeningen ---
# Range van Lift Coefficient (van 0 tot rotatie-lift)
cl_range = np.linspace(0, 1.2, 100)

# Induced Drag: C_Di = Cl^2 / (pi * AR * e)
cd_induced = (cl_range**2) / (np.pi * ar * e)

# Totale Drag Coefficient: C_D = C_D0 + C_Di
cd_total = cd_0 + cd_induced

# Drag Force in Newtons bij Take-off snelheid
drag_force = 0.5 * rho * (v_takeoff**2) * s_plan * cd_total

# --- 3. Visualisatie ---
plt.figure(figsize=(10, 6))

# Plot Drag Polar
plt.subplot(1, 2, 1)
plt.plot(cd_total, cl_range, label='Drag Polar', color='blue', linewidth=2)
plt.axvline(cd_0, color='red', linestyle='--', label='Zero-lift Drag (C_D,0)')
plt.title('Drag Polar (Subsonic)')
plt.xlabel('Drag Coefficient ($C_D$)')
plt.ylabel('Lift Coefficient ($C_L$)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot Drag Force vs Cl
plt.subplot(1, 2, 2)
plt.plot(cl_range, drag_force / 1000, color='green', linewidth=2)
plt.title('Drag Force vs $C_L$ at 360 km/h')
plt.xlabel('Lift Coefficient ($C_L$)')
plt.ylabel('Drag Force (kN)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 4. Output voor Versnelling ---
# Voorbeeld bij Cl = 0.7 (rotatie)
cl_rot = 0.7
cd_rot = cd_0 + (cl_rot**2) / (np.pi * ar * e)
d_rot = 0.5 * rho * (v_takeoff**2) * s_plan * cd_rot
accel_g = 0.15
thrust_needed = d_rot + (mass * accel_g * g)

print(f"--- Resultaten bij Cl = {cl_rot} ---")
print(f"Totale Cd: {cd_rot:.5f}")
print(f"Weerstandskracht (D): {d_rot/1000:.2f} kN")
print(f"Benodigde Stuwkracht voor 0.15g versnelling: {thrust_needed/1000:.2f} kN")