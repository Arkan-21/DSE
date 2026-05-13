# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:01:22 2026

@author: SID-DRW
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuratie & Input ---
w_tog = 111389         # Massa (kg)
s_plan = 425            # Oppervlakte (m2)
accel_g_target = 0.15   # Doelversnelling (g)
g = 9.81
#cd_0 = 0.00567          # Jouw basis-frictie
rho = 0.05              # Luchtdichtheid op kruishoogte (kg/m3)
v_sound = 300.0         # Geluidssnelheid op hoogte (m/s)

# Mach range: van Mach 3 t/m Mach 8 (Hypersonisch regime)
mach_range = np.linspace(3.0, 6.0, 100)

# --- 2. Berekenings-logica ---
cl_list = []
cd_list = []
alpha_list = []
thrust_list = []
ld_ratio_list = []

for M in mach_range:
    V = M * v_sound
    
    # A. Benodigde Lift (L = W)
    lift_needed = w_tog * g
    
    # B. Benodigde CL
    # Cl = L / (0.5 * rho * V^2 * S)
    cl_needed = lift_needed / (0.5 * rho * (V**2) * s_plan)
    
    # C. Invalshoek (Alpha) via Newtonian Flow: Cl = 2 * sin^2(alpha)
    # Benadering voor kleine hoeken: alpha = sqrt(Cl / 2)
    alpha_rad = np.sqrt(cl_needed / 2)
    
    # D. Drag Coefficient (Newtonian): Cd_wave = 2 * sin^3(alpha)
    cd_induced = 2 * (alpha_rad**3)
    cd_0 = 0.005 + 0.001*M
    cd_total = cd_0 + cd_induced
    
    # E. Krachten (Newton)
    drag_force = 0.5 * rho * (V**2) * s_plan * cd_total
    thrust_req = drag_force + (w_tog * accel_g_target * g)
    
    # Opslaan voor grafieken
    cl_list.append(cl_needed)
    cd_list.append(cd_total)
    alpha_list.append(np.degrees(alpha_rad))
    thrust_list.append(thrust_req / 1000) # In kN
    ld_ratio_list.append(cl_needed / cd_total)

# --- 3. Visualisatie ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Grafiek 1: Benodigde Invalshoek
axs[0, 0].plot(mach_range, alpha_list, color='blue', lw=2)
axs[0, 0].set_title('Benodigde Invalshoek (Alpha)')
axs[0, 0].set_xlabel('Mach Number')
axs[0, 0].set_ylabel('Degrees')
axs[0, 0].grid(True, alpha=0.3)

# Grafiek 2: L/D Ratio (Efficiëntie)
axs[0, 1].plot(mach_range, ld_ratio_list, color='green', lw=2)
axs[0, 1].set_title('Aerodynamische Efficiëntie (L/D)')
axs[0, 1].set_xlabel('Mach Number')
axs[0, 1].set_ylabel('CL / CD')
axs[0, 1].grid(True, alpha=0.3)

# Grafiek 3: CL vs CD (Drag Polar trend)
axs[1, 0].plot(cd_list, cl_list, color='purple', lw=2)
axs[1, 0].set_title('Newtonian Drag Polar (CL vs CD)')
axs[1, 0].set_xlabel('CD')
axs[1, 0].set_ylabel('CL')
axs[1, 0].grid(True, alpha=0.3)

# Grafiek 4: Benodigde Stuwkracht voor 0.15g
axs[1, 1].plot(mach_range, thrust_list, color='red', lw=2)
axs[1, 1].set_title('Benodigde Stuwkracht (Thrust) voor 0.15g')
axs[1, 1].set_xlabel('Mach Number')
axs[1, 1].set_ylabel('Thrust (kN)')
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()