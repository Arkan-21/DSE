# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:14:21 2026

@author: SID-DRW
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Input Data ---
s_plan = 425.682          # Reference area (m2)
s_wet = 1188.937          # Total wetted area (m2)
cd_0 = 0.00567            # Calculated cd_friction (based on S_plan)
ar = 1.2                  # Aspect Ratio estimate for hypersonic aircraft
e = 0.65                  # Oswald efficiency factor (typical for delta wings)
rho = 1.225               # Air density at sea level (kg/m3)
v_takeoff = 100.0         # Speed in m/s (~360 km/h)
mass = 200000.0           # Aircraft mass in kg
g = 9.81                  # Acceleration due to gravity (m/s2)

# --- 2. Calculations ---
# Theoretical 2D lift curve slope per radian (2 * pi) (HELMBOLD MODEL)
a_0 = 2 * np.pi 

# Helmbold equation for low-aspect ratio wings per radian
cl_alpha_3d_rad = a_0 / (1 + (np.sqrt(1 + (a_0 / (np.pi * ar))**2)))
# Convert from radians to degrees
cl_alpha_3d_deg = cl_alpha_3d_rad * (np.pi / 180)

# =============================================================================
# Generate a full angle of attack sweep from 0 to 45 degrees
alpha_sweep_deg = np.linspace(0, 45, 250)
alpha_sweep_rad = np.radians(alpha_sweep_deg)

# Physical Aerodynamic Parameters
K_p = cl_alpha_3d_rad     # Potential lift slope derived directly from Helmbold
K_v = 9.5                 # Amplified Vortex lift factor from active flow control
alpha_c_deg = 18.0        # Critical angle of attack where vortex breakdown initiates

# 1. Calculate Pure Polhamus Components (Potential + Vortex Lift)
cl_potential = K_p * np.sin(alpha_sweep_rad) * (np.cos(alpha_sweep_rad)**2)
cl_vortex = K_v * (np.sin(alpha_sweep_rad)**2) * np.cos(alpha_sweep_rad)

# 2. Physics-based Vortex Health Function
vortex_health = 1.0 / (1.0 + np.exp((alpha_sweep_deg - alpha_c_deg) / 1.0))

# 3. Combine components dynamically to establish full 3D curve
cl_sweep_3d = cl_potential + cl_vortex * vortex_health

# Locate Cl_max and Stall Angle dynamically from the resulting physical physics curve
idx_stall = np.argmax(cl_sweep_3d)
cl_max = cl_sweep_3d[idx_stall]
alpha_stall = alpha_sweep_deg[idx_stall]

# =============================================================================
# Dynamically calculate the exact Cl required to lift the aircraft mass at v_takeoff
# L = Weight -> Cl_rot = (2 * m * g) / (rho * V^2 * S)
cl_rot = (2.0 * mass * g) / (rho * (v_takeoff**2) * s_plan)

# Dynamic Drag Polar Calculation spanning up to the calculated cl_max
cl_range = np.linspace(0, cl_max, 100)
cd_induced = (cl_range**2) / (np.pi * ar * e)
cd_total = cd_0 + cd_induced

# Drag Force array in Newtons at take-off speed across the operational Cl range
drag_force = 0.5 * rho * (v_takeoff**2) * s_plan * cd_total

# Dynamic Drag Coefficient specifically at the point of rotation
cd_rot = cd_0 + (cl_rot**2) / (np.pi * ar * e)
d_rot = 0.5 * rho * (v_takeoff**2) * s_plan * cd_rot

# Required Thrust calculation for a specified 0.15g runway acceleration
accel_g = 0.15
thrust_needed = d_rot + (mass * accel_g * g)

# Interpolate from the true non-linear curve to find the exact physical rotation alpha
alpha_rot = np.interp(cl_rot, cl_sweep_3d[:idx_stall], alpha_sweep_deg[:idx_stall])
# =============================================================================


# --- 3. Visualization ---
plt.figure(figsize=(15, 5)) 

# Plot Drag Polar (Dynamic limits)
plt.subplot(1, 3, 1) 
plt.plot(cd_total, cl_range, label='Drag Polar', color='blue', linewidth=2)
plt.axvline(cd_0, color='red', linestyle='--', label='Zero-lift Drag ($C_{D,0}$)')
plt.scatter(cd_rot, cl_rot, color='green', s=50, zorder=5, label='Rotation Point')
plt.title('Dynamic Drag Polar (Subsonic)')
plt.xlabel('Drag Coefficient ($C_D$)')
plt.ylabel('Lift Coefficient ($C_L$)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot Drag Force vs Cl (Dynamic limits)
plt.subplot(1, 3, 2)
plt.plot(cl_range, drag_force / 1000, color='green', linewidth=2)
plt.axvline(cl_rot, color='purple', linestyle=':', label='Required Takeoff $C_L$')
plt.title('Drag Force vs $C_L$ at 360 km/h')
plt.xlabel('Lift Coefficient ($C_L$)')
plt.ylabel('Drag Force (kN)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot Cl-alpha curve (Cleaned up & Fully Physics-driven)
plt.subplot(1, 3, 3)
plt.plot(alpha_sweep_deg, cl_sweep_3d, color='purple', linewidth=2.5, label='Physics-Based Profile')

# Highlight the dynamically derived rotation point on the lift curve
plt.scatter(alpha_rot, cl_rot, color='green', s=50, zorder=5, label='Rotation Point')
plt.axhline(cl_rot, color='green', linestyle=':', alpha=0.6)

# Plot the Stall point marker
plt.scatter(alpha_stall, cl_max, color='red', s=60, zorder=5, label='Dynamic Stall Point')
plt.annotate(f'$C_{{L,\\max}}$ = {cl_max:.2f}\n$\\alpha_{{stall}}$ = {alpha_stall:.1f}°',
             xy=(alpha_stall, cl_max),
             xytext=(alpha_stall + 3, cl_max - 0.3),
             arrowprops=dict(arrowstyle="->", color='black', lw=1))

plt.title('$C_L$ vs Angle of Attack ($\\alpha$)')
plt.xlabel('Angle of Attack ($\\alpha$ in degrees)')
plt.ylabel('Lift Coefficient ($C_L$)')
plt.xlim(0, 45)
plt.ylim(0, cl_max * 1.2)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


# --- 4. Output for Performance Verification ---
print("--- Dynamically Calculated Takeoff Parameters ---")
print(f"Required Takeoff Lift Coefficient (C_L_rot): {cl_rot:.4f}")
print(f"Total Cd at Rotation: {cd_rot:.5f}")
print(f"Aerodynamic Drag Force at Rotation (D): {d_rot/1000:.2f} kN")
print(f"Required Thrust for 0.15g acceleration: {thrust_needed/1000:.2f} kN")
print(f"Base Helmbold Linear Slope (C_L_alpha): {cl_alpha_3d_deg:.4f} per degree")
print(f"Maximum Lift Coefficient (C_L_max): {cl_max:.2f}")
print(f"Stall Angle of Attack: {alpha_stall:.2f}°")
print(f"Required Angle of Attack (alpha) for rotation: {alpha_rot:.2f}°")

# Safety Check Assertions
safety_margin = alpha_stall - alpha_rot
print(f"Takeoff Safety Margin (alpha_stall - alpha_rot): {safety_margin:.2f}°")
if safety_margin > 3.0:
    print("Verification: Aerodynamic takeoff safety margin is SATISFACTORY.")
else:
    print("Verification WARNING: Takeoff pitch attitude is too close to stall bounds!")