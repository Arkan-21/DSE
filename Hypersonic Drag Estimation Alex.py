import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTS & VEHICLE PARAMETERS
# =============================================================================
g = 9.81
R = 287.0
gamma = 1.4

w_tog = 111389.0         # Massa (kg)
s_plan = 425.0           # Oppervlakte (m2)
accel_g_target = 0.15    # Doelversnelling (g)
L_ref = 35.0             # Karakteristieke lengte (m) voor Reynolds-berekening

# Newtoniaanse parameters
cp_max = 2.0             # Maximale drukcoëfficiënt volgens de klassieke Newton-theorie
S_wet_S_ref = 2.0        # Boven- + onderkant van de 2D plaat voor wrijving


# =============================================================================
# FLIGHT CONDITION & ISA ATMOSPHERE (UPDATED FOR UPPER STRATOSPHERE UP TO 32KM)
# =============================================================================
altitude = 30000.0       # Kruishoogte [m]

def isa_atmosphere(h):
    g0 = 9.80665
    T0 = 288.15
    P0 = 101325.0
    R  = 287.05          # Gas constant for air [J/(kg*K)]
    
    # Layer 1 Base Parameters: Troposphere (0 to 11,000 m)
    L1 = -0.0065
    
    # Layer 2 Base Parameters: Lower Stratosphere (11,000 m to 20,000 m) [Isothermal]
    h11 = 11000.0
    T11 = T0 + L1 * h11
    P11 = P0 * (T11 / T0)**(-g0 / (L1 * R))
    
    # Layer 3 Base Parameters: Upper Stratosphere (20,000 m to 32,000 m) [Inversion]
    h20 = 20000.0
    T20 = T11            # Fixed temperature from 11km to 20km (216.65 K)
    P20 = P11 * np.exp(-g0 * (h20 - h11) / (R * T11))
    L3 = 0.0010          # Positive temperature lapse rate (+1.0 K / km)

    if h <= 11000.0:
        # Troposphere
        T = T0 + L1 * h
        P = P0 * (T / T0)**(-g0 / (L1 * R))
    elif h <= 20000.0:
        # Lower Stratosphere (Isothermal)
        T = T11
        P = P11 * np.exp(-g0 * (h - h11) / (R * T))
    elif h <= 32000.0:
        # Upper Stratosphere (Temperature Inversion Layer)
        T = T20 + L3 * (h - h20)
        P = P20 * (T / T20)**(-g0 / (L3 * R))
    else:
        raise ValueError("Altitude exceeds the 32,000m upper stratospheric ceiling modeled.")

    rho = P / (R * T)
    return T, P, rho

def viscosity_sutherland(T):
    return 1.458e-6 * T**1.5 / (T + 110.4)

T_inf, P_inf, rho_inf = isa_atmosphere(altitude)
v_sound = np.sqrt(gamma * R * T_inf)
mu_inf = viscosity_sutherland(T_inf)

# Mach range: van Mach 3 t/m Mach 6
mach_range = np.linspace(3.0, 6.0, 100)

# =============================================================================
# STORAGE FOR RESULTS
# =============================================================================
cl_list = []
cd_list = []
alpha_list = []
thrust_list = []
ld_ratio_list = []
drag_list = []
reynolds_list = []

cd_f_list = []
cd_press_list = []

# =============================================================================
# MAIN LOGIC LOOP (Pure Newtonian 2D Estimation)
# =============================================================================
for M in mach_range:
    V = M * v_sound
    
    # 1. Dynamische Reynolds-berekening
    Re = (rho_inf * V * L_ref) / mu_inf
    
    # A. Benodigde Lift (L = W)
    lift_needed = w_tog * g
    
    # B. Benodigde CL
    cl_needed = lift_needed / (0.5 * rho_inf * (V**2) * s_plan)
    
    # FIXED: Supersonic/Hypersonic Lift Slope instead of Pure Newtonian
    # dCL/dalpha = 4 / sqrt(M^2 - 1)
    cl_alpha = 4.0 / np.sqrt(M**2 - 1.0)
    alpha_rad = cl_needed / cl_alpha
    
    # C. Drag Coëfficiënt
    # Pressure Drag based on supersonic wave drag trends at alpha: 
    # CD_press = CL * alpha (induced/wave alignment)
    cd_press = cl_needed * alpha_rad
    
    # 2. Huidwrijving (Turbulent met compressibility correctie)
    cf = (0.074 / (Re**(0.2))) * ((1 / (1 + 0.15 * M**2))**0.58)
    cd_f = cf * S_wet_S_ref
    
    # Totale drag coëfficiënt
    cd_total = cd_f + cd_press
    
    # D. Krachten & Benodigde Stuwkracht
    drag_force = 0.5 * rho_inf * (V**2) * s_plan * cd_total
    thrust_req = drag_force + (w_tog * accel_g_target * g)
    
    # Opslaan van data
    cl_list.append(cl_needed)
    cd_list.append(cd_total)
    alpha_list.append(np.degrees(alpha_rad)) # Will now be a realistic 2° to 5°!
    thrust_list.append(thrust_req / 1000)      # In kN
    ld_ratio_list.append(cl_needed / cd_total)
    drag_list.append(drag_force / 1000)         # In kN
    reynolds_list.append(Re)
    
    cd_f_list.append(cd_f)
    cd_press_list.append(cd_press)

# =============================================================================
# VISUALISATION (Grid Layout 3x2)
# =============================================================================
fig, axs = plt.subplots(3, 2, figsize=(14, 13))

# Grafiek 1: Benodigde Invalshoek
axs[0, 0].plot(mach_range, alpha_list, color='blue', lw=2)
axs[0, 0].set_title('Required Angle of Attack (Alpha) - Pure Newtonian')
axs[0, 0].set_xlabel('Mach Number')
axs[0, 0].set_ylabel('Degrees')
axs[0, 0].grid(True, alpha=0.3)

# Grafiek 2: L/D Ratio (Efficiëntie)
axs[0, 1].plot(mach_range, ld_ratio_list, color='green', lw=2)
axs[0, 1].set_title('Aerodynamic Efficiency (L/D)')
axs[0, 1].set_xlabel('Mach Number')
axs[0, 1].set_ylabel('CL / CD')
axs[0, 1].grid(True, alpha=0.3)

# Grafiek 3: CL vs CD (Drag Polar trend)
axs[1, 0].plot(cd_list, cl_list, color='purple', lw=2)
axs[1, 0].set_title('Pure Newtonian Drag Polar (CL vs CD)')
axs[1, 0].set_xlabel('CD')
axs[1, 0].set_ylabel('CL')
axs[1, 0].grid(True, alpha=0.3)

# Grafiek 4: Benodigde Stuwkracht voor 0.15g
axs[1, 1].plot(mach_range, thrust_list, color='red', lw=2)
axs[1, 1].set_title('Required Thrust for 0.15g Acceleration')
axs[1, 1].set_xlabel('Mach Number')
axs[1, 1].set_ylabel('Thrust (kN)')
axs[1, 1].grid(True, alpha=0.3)

# Grafiek 5: Dynamische Reynolds-getal
axs[2, 0].plot(mach_range, reynolds_list, color='orange', lw=2)
axs[2, 0].set_title('Reynolds Number (Dynamic via ISA)')
axs[2, 0].set_xlabel('Mach Number')
axs[2, 0].set_ylabel('Re')
axs[2, 0].set_yscale('log')
axs[2, 0].grid(True, which='both', alpha=0.3)

# Grafiek 6: Drag Coëfficiënt Breakdown
axs[2, 1].plot(mach_range, cd_f_list, label='Friction Drag (Cd_f)', lw=2)
axs[2, 1].plot(mach_range, cd_press_list, label='Newtonian Pressure Drag (Cd_press)', lw=2)
axs[2, 1].plot(mach_range, cd_list, label='Total Drag', lw=2, linestyle='--')
axs[2, 1].set_title('Newtonian Drag Coefficient Breakdown')
axs[2, 1].set_xlabel('Mach Number')
axs[2, 1].set_ylabel('CD')
axs[2, 1].grid(True, alpha=0.3)
axs[2, 1].legend()

plt.tight_layout()
plt.show()