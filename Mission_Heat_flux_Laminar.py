import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# USER INPUTS
# =============================================================================

csv_file = "density_velocity_database.csv"

altitude = 25000.0  # [m]

nose_radius = 0.025
emissivity = 0.85
plate_length = 2.0

sigma = 5.670374419e-8
R = 287.0

# =============================================================================
# ISA TEMPERATURE MODEL (used ONLY for T_inf)
# =============================================================================

def isa_temperature(h):
    """
    ISA temperature up to 25 km
    """

    T0 = 288.15
    L = -0.0065

    if h <= 11000.0:
        return T0 + L * h
    else:
        return 216.65  # isothermal layer


T_inf = isa_temperature(altitude)

print("\n================================================")
print("ISA TEMPERATURE")
print("================================================")

print(f"Altitude = {altitude/1000:.2f} km")
print(f"T_inf    = {T_inf:.2f} K")

# =============================================================================
# AIR PROPERTY MODELS
# =============================================================================

def cp_air(T):
    return 1000 + 0.1 * (T - 300)


def gamma_air(T):
    gamma = 1.4 - 0.00005 * (T - 300)
    return max(gamma, 1.28)


def pr_air(T):
    return 0.72


def viscosity_sutherland(T):
    return 1.458e-6 * T**1.5 / (T + 110.4)


def conductivity(mu, cp, pr):
    return mu * cp / pr


# =============================================================================
# RADIATIVE EQUILIBRIUM SOLVER
# =============================================================================

def solve_equilibrium_temperature(h, T_aw, emissivity):

    T = min(T_aw, 1200.0)

    for _ in range(100):

        f = emissivity * sigma * T**4 - h * (T_aw - T)
        df = 4 * emissivity * sigma * T**3 + h

        T_new = T - f / df

        if abs(T_new - T) < 1e-3:
            return T_new

        T = T_new

    return T

# =============================================================================
# NORMAL SHOCK RELATIONS
# =============================================================================

def normal_shock(M1, T1, P1, rho1):

    gamma1 = gamma_air(T1)

    P2_P1 = 1 + (2 * gamma1 / (gamma1 + 1)) * (M1**2 - 1)

    rho2_rho1 = ((gamma1 + 1) * M1**2) / ((gamma1 - 1) * M1**2 + 2)

    T2_T1 = P2_P1 / rho2_rho1

    M2 = np.sqrt(
        (1 + 0.5 * (gamma1 - 1) * M1**2)
        /
        (gamma1 * M1**2 - 0.5 * (gamma1 - 1))
    )

    P2 = P1 * P2_P1
    rho2 = rho1 * rho2_rho1
    T2 = T1 * T2_T1

    return M2, T2, P2, rho2


# =============================================================================
# OBLIQUE SHOCK (M-BETA-THETA)
# =============================================================================

def theta_from_beta(M, beta, gamma):
    """
    Returns flow deflection angle theta (rad)
    from M-beta-theta relation
    """

    term1 = 2 / np.tan(beta)
    term2 = (M**2 * np.sin(beta)**2 - 1)
    term3 = M**2 * (gamma + np.cos(2 * beta)) + 2

    return np.arctan(term1 * term2 / term3)


def oblique_shock(M1, beta, T1, P1, rho1):

    gamma1 = gamma_air(T1)

    # Normal component
    Mn1 = M1 * np.sin(beta)

    if Mn1 <= 1.0:
        return None  # no shock

    # Normal shock on Mn1
    Mn2, T2, P2, rho2 = normal_shock(Mn1, T1, P1, rho1)

    # Flow deflection
    theta = theta_from_beta(M1, beta, gamma1)

    # Downstream Mach
    M2 = Mn2 / np.sin(beta - theta)

    return M2, T2, P2, rho2, theta


# =============================================================================
# STAGNATION HEATING
# =============================================================================

def stagnation_heating(rho_inf, u_inf, radius):

    return 1.7415e-4 * np.sqrt(rho_inf / radius) * u_inf**3


# =============================================================================
# READ DATABASE
# =============================================================================

print("\n================================================")
print("READING DATABASE")
print("================================================")

df = pd.read_csv(csv_file)

print(f"Loaded {len(df)} trajectory points")

# =============================================================================
# WORST-CASE POINT
# =============================================================================

print("\n================================================")
print("FINDING WORST-CASE POINT")
print("================================================")

screening_q = (
    1.83e-4
    * np.sqrt(df["rho"] / nose_radius)
    * df["v"]**3
)

idx = np.argmax(screening_q)

rho_inf = df["rho"].iloc[idx]
u_inf = df["v"].iloc[idx]

print(f"rho_inf (DB) = {rho_inf:.6f} kg/m³")
print(f"u_inf   (DB) = {u_inf:.2f} m/s")

# =============================================================================
# CONSISTENT PRESSURE (OPTION C)
# =============================================================================

P_inf = rho_inf * R * T_inf

gamma_inf = gamma_air(T_inf)
a_inf = np.sqrt(gamma_inf * R * T_inf)
mach = u_inf / a_inf

print("\n================================================")
print("FREESTREAM CONDITIONS (CONSISTENT)")
print("================================================")

print(f"T_inf   = {T_inf:.2f} K (ISA)")
print(f"P_inf   = {P_inf:.2f} Pa (reconstructed)")
print(f"rho_inf = {rho_inf:.5f} kg/m³ (database)")
print(f"u_inf   = {u_inf:.2f} m/s (database)")
print(f"Mach    = {mach:.3f}")

# =============================================================================
# POST-SHOCK CONDITIONS
# =============================================================================

M_edge, T_edge, P_edge, rho_edge = normal_shock(
    mach, T_inf, P_inf, rho_inf
)

gamma_edge = gamma_air(T_edge)

u_edge = M_edge * np.sqrt(gamma_edge * R * T_edge)

print("\n================================================")
print("POST-SHOCK EDGE CONDITIONS")
print("================================================")

print(f"M_edge   = {M_edge:.3f}")
print(f"T_edge   = {T_edge:.2f} K")
print(f"P_edge   = {P_edge:.2f} Pa")
print(f"rho_edge = {rho_edge:.5f} kg/m³")
print(f"u_edge   = {u_edge:.2f} m/s")

# =============================================================================
# EDGE PROPERTIES
# =============================================================================

cp_edge = cp_air(T_edge)
pr_edge = pr_air(T_edge)
mu_edge = viscosity_sutherland(T_edge)
k_edge = conductivity(mu_edge, cp_edge, pr_edge)

# =============================================================================
# STAGNATION
# =============================================================================

q_stag = stagnation_heating(rho_inf, u_inf, nose_radius)

T_stag = (q_stag / (emissivity * sigma))**0.25

# =============================================================================
# ADIABATIC WALL TEMPERATURES
# =============================================================================

r_lam = np.sqrt(pr_edge)
r_turb = pr_edge**(1/3)

T_aw_lam = T_edge * (1 + r_lam * 0.5 * (gamma_edge - 1) * M_edge**2)
T_aw_turb = T_edge * (1 + r_turb * 0.5 * (gamma_edge - 1) * M_edge**2)

# =============================================================================
# TRANSITION MODEL
# =============================================================================

Re_transition_start = 1e6 * M_edge
Re_transition_end = 3e6 * M_edge

# =============================================================================
# SHOCK ANGLE RANGE
# =============================================================================

beta_deg_range = np.linspace(15, 75, 6)  # degrees
beta_rad_range = np.radians(beta_deg_range)

# Storage for plotting
all_temperature_profiles = []

# =============================================================================
# LOOP OVER SHOCK ANGLES
# =============================================================================
x_vals = np.linspace(1e-4, plate_length, 700)

for beta_deg, beta in zip(beta_deg_range, beta_rad_range):

    result = oblique_shock(mach, beta, T_inf, P_inf, rho_inf)

    if result is None:
        continue

    M_edge, T_edge, P_edge, rho_edge, theta = result

    gamma_edge = gamma_air(T_edge)
    u_edge = M_edge * np.sqrt(gamma_edge * R * T_edge)

    print("\n================================================")
    print(f"SHOCK ANGLE β = {beta_deg:.1f} deg")
    print("================================================")
    print(f"Deflection θ = {np.degrees(theta):.2f} deg")
    print(f"M_edge       = {M_edge:.3f}")
    print(f"T_edge       = {T_edge:.2f} K")

    # Edge properties
    cp_edge = cp_air(T_edge)
    pr_edge = pr_air(T_edge)
    mu_edge = viscosity_sutherland(T_edge)
    k_edge = conductivity(mu_edge, cp_edge, pr_edge)

    # Adiabatic wall temps
    r_lam = np.sqrt(pr_edge)
    r_turb = pr_edge**(1/3)

    T_aw_lam = T_edge * (1 + r_lam * 0.5 * (gamma_edge - 1) * M_edge**2)
    T_aw_turb = T_edge * (1 + r_turb * 0.5 * (gamma_edge - 1) * M_edge**2)

    # Transition
    Re_transition_start = 1e6 * M_edge
    Re_transition_end = 3e6 * M_edge

    # Surface walkdown
    T_profile = []

    for x in x_vals:

        Re_x = rho_edge * u_edge * x / mu_edge

        # Laminar
        Nu_lam = 0.332 * Re_x**0.5 * pr_edge**(1/3)
        h_lam = Nu_lam * k_edge / x
        T_lam = solve_equilibrium_temperature(h_lam, T_aw_lam, emissivity)

        # Turbulent
        Nu_turb = 0.0296 * Re_x**0.8 * pr_edge**(1/3)
        h_turb = Nu_turb * k_edge / x
        T_turb = solve_equilibrium_temperature(h_turb, T_aw_turb, emissivity)

        # Blend
        if Re_x <= Re_transition_start:
            blend = 0.0
        elif Re_x >= Re_transition_end:
            blend = 1.0
        else:
            blend = (Re_x - Re_transition_start) / (
                Re_transition_end - Re_transition_start
            )

        T_boundary = (1 - blend) * T_lam + blend * T_turb

        # Stagnation blending (still from freestream)
        stag_blend = np.exp(-x / 0.03)

        T_phys = stag_blend * T_stag + (1 - stag_blend) * T_boundary

        T_profile.append(T_phys)

    all_temperature_profiles.append((beta_deg, T_profile))


# =============================================================================
# PLOT MULTI-ANGLE RESULTS
# =============================================================================

plt.figure(figsize=(13, 6))

for beta_deg, T_profile in all_temperature_profiles:
    plt.plot(
        x_vals,
        T_profile,
        linewidth=2,
        label=f"β = {beta_deg:.0f}°"
    )

plt.scatter([0], [T_stag], s=120, label="Stagnation")

plt.xlabel("Distance [m]")
plt.ylabel("Equilibrium Temperature [K]")
plt.title("Surface Temperature vs Shock Angle (Oblique Shock Model)")

plt.grid(True)
plt.legend()
plt.show()