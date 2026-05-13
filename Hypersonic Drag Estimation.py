# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:01:22 2026

@author: SID-DRW
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuratie & Input ---
w_tog = 111389          # Massa (kg)
s_plan = 450            # Oppervlakte (m2)
accel_g_target = 0.15   # Doelversnelling (g)
g = 9.81
rho = 0.000821              # Luchtdichtheid (kg/m3) - Constant voor dit model
v_sound = 300.0         # Geluidssnelheid (m/s)

# Mach range: van Mach 3 t/m Mach 8 (we breiden hem iets uit)
mach_range = np.linspace(3.0, 8.0, 100)

cl_list, cd_total_list, thrust_list, alpha_list = [], [], [], []
drag_wave_list, drag_friction_list = [], []

# --- 2. Berekenings-logica ---
for M in mach_range:
    V = M * v_sound
    q = 0.5 * rho * (V**2)
    
    # A. Benodigde CL voor level flight
    cl_needed = (w_tog * g) / (q * s_plan)
    
    # B. Invalshoek (Newtonian: Cl = 2 * sin^2(alpha))
    # We vangen op als cl_needed > 2 (fysiek onmogelijk voor dit model)
    alpha_rad = np.arcsin(np.sqrt(np.clip(cl_needed / 2, 0, 1)))
    
    # C. CD_wave (Geïnduceerd door lift)
    cd_wave = 2 * (np.sin(alpha_rad)**3)
    
    # D. CD_0 (Wrijving) - NU VARIABEL
    # We simuleren dat wrijving toeneemt door hitte en viskeuze interactie
    # Een basiswaarde + een factor die stijgt met Mach
    cd_0 = 0.005 + (0.001 * M) 
    
    cd_total = cd_wave + cd_0
    
    # E. Krachten in Newton
    force_wave = q * s_plan * cd_wave
    force_friction = q * s_plan * cd_0
    
    total_drag = force_wave + force_friction
    accel_force = w_tog * accel_g_target * g
    
    thrust_req = total_drag + accel_force
    
    # Opslaan
    cl_list.append(cl_needed)
    cd_total_list.append(cd_total)
    alpha_list.append(np.degrees(alpha_rad))
    thrust_list.append(thrust_req / 1000) # kN
    drag_wave_list.append(force_wave / 1000)
    drag_friction_list.append(force_friction / 1000)

# --- 3. Visualisatie ---
plt.figure(figsize=(10, 6))
plt.plot(mach_range, drag_wave_list, label='Wave Drag (geïnduceerd)', color='blue', linestyle='--')
plt.plot(mach_range, drag_friction_list, label='Friction Drag (CD0)', color='orange', linestyle='--')
plt.plot(mach_range, thrust_list, label='Totale Benodigde Thrust', color='red', lw=3)

plt.title('Thrust Breakdown: Waarom de lijn nu wel stijgt')
plt.xlabel('Mach Number')
plt.ylabel('Kracht (kN)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(cl_list)