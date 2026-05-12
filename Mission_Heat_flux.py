import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# USER INPUTS
# =============================================================================

csv_file = "density_velocity_database.csv"

mach = 5.0

nose_radius = 0.025
emissivity = 0.9

plate_length = 5.0

sigma = 5.670374419e-8

R = 287.0

# =============================================================================
# TEMPERATURE-DEPENDENT AIR PROPERTIES
# =============================================================================

def cp_air(T):

    # Simple engineering approximation for air
    # Valid roughly 200 K - 2000 K

    return 1000 + 0.1 * (T - 300)


def gamma_air(T):

    # Approximate gamma decrease with temperature

    gamma = 1.4 - 0.00005 * (T - 300)

    return max(gamma, 1.28)


def pr_air(T):

    return 0.72


def viscosity_sutherland(T):

    return 1.458e-6 * T**1.5 / (T + 110.4)


def conductivity(T, mu, cp, pr):

    return mu * cp / pr


# =============================================================================
# EQUILIBRIUM TEMPERATURE
# =============================================================================

def equilibrium_temperature(q):

    q = max(q, 1.0)

    return (q / (emissivity * sigma))**0.25


# =============================================================================
# NORMAL SHOCK
# =============================================================================

def normal_shock(M1, T1, P1, rho1):

    gamma1 = gamma_air(T1)

    P2_P1 = 1 + (2 * gamma1 / (gamma1 + 1)) * (M1**2 - 1)

    rho2_rho1 = ((gamma1 + 1) * M1**2) / (
        (gamma1 - 1) * M1**2 + 2
    )

    T2_T1 = P2_P1 / rho2_rho1

    M2 = np.sqrt(
        (
            1 + 0.5 * (gamma1 - 1) * M1**2
        ) /
        (
            gamma1 * M1**2
            - 0.5 * (gamma1 - 1)
        )
    )

    P2 = P1 * P2_P1
    rho2 = rho1 * rho2_rho1
    T2 = T1 * T2_T1

    return M2, T2, P2, rho2


# =============================================================================
# FAY-RIDDELL APPROXIMATION
# =============================================================================

def fay_riddell(rho_e, mu_e, velocity, radius, pr):

    q = (
        0.763
        * pr**(-0.6)
        * np.sqrt(rho_e * mu_e)
        * velocity**3
        / np.sqrt(radius)
    )

    return q


# =============================================================================
# READ DATABASE
# =============================================================================

print("\n================================================")
print("READING DATABASE")
print("================================================")

df = pd.read_csv(csv_file)

print(f"Loaded {len(df)} trajectory points")

# =============================================================================
# FIND WORST-CASE POINT
# =============================================================================

print("\n================================================")
print("FINDING WORST-CASE POINT")
print("================================================")

screening_q = (
    1.7415e-4
    * np.sqrt(df["rho"] / nose_radius)
    * df["v"]**3
)

idx = np.argmax(screening_q)

rho_inf = df["rho"].iloc[idx]
u_inf = df["v"].iloc[idx]

print(f"rho_inf = {rho_inf:.6f} kg/m³")
print(f"u_inf   = {u_inf:.2f} m/s")

# =============================================================================
# FREESTREAM CONDITIONS
# =============================================================================

gamma_inf = gamma_air(220)

T_inf = (u_inf / mach)**2 / (gamma_inf * R)

P_inf = rho_inf * R * T_inf

print("\n================================================")
print("FREESTREAM CONDITIONS")
print("================================================")

print(f"T_inf = {T_inf:.2f} K")
print(f"P_inf = {P_inf:.2f} Pa")

# =============================================================================
# POST-SHOCK CONDITIONS
# =============================================================================

M_edge, T_edge, P_edge, rho_edge = normal_shock(
    mach,
    T_inf,
    P_inf,
    rho_inf
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

k_edge = conductivity(
    T_edge,
    mu_edge,
    cp_edge,
    pr_edge
)

print("\n================================================")
print("EDGE PROPERTIES")
print("================================================")

print(f"cp_edge = {cp_edge:.2f} J/kgK")
print(f"gamma   = {gamma_edge:.3f}")
print(f"mu_edge = {mu_edge:.6e}")
print(f"k_edge  = {k_edge:.5f}")

# =============================================================================
# FAY-RIDDELL STAGNATION HEATING
# =============================================================================

q_stag = fay_riddell(
    rho_edge,
    mu_edge,
    u_edge,
    nose_radius,
    pr_edge
)

T_stag = equilibrium_temperature(q_stag)

print("\n================================================")
print("STAGNATION HEATING")
print("================================================")

print(f"q_stag = {q_stag:.3e} W/m²")
print(f"T_stag = {T_stag:.2f} K")

# =============================================================================
# RECOVERY TEMPERATURES
# =============================================================================

r_lam = np.sqrt(pr_edge)

r_turb = pr_edge**(1/3)

T_aw_lam = T_edge * (
    1
    + r_lam
    * 0.5
    * (gamma_edge - 1)
    * M_edge**2
)

T_aw_turb = T_edge * (
    1
    + r_turb
    * 0.5
    * (gamma_edge - 1)
    * M_edge**2
)

print("\n================================================")
print("ADIABATIC WALL TEMPERATURES")
print("================================================")

print(f"Taw laminar   = {T_aw_lam:.2f} K")
print(f"Taw turbulent = {T_aw_turb:.2f} K")

# =============================================================================
# TRANSITION MODEL
# =============================================================================

Re_transition_start = 1e6 * M_edge
Re_transition_end = 3e6 * M_edge

print("\n================================================")
print("TRANSITION MODEL")
print("================================================")

print(f"Transition onset Re_x = {Re_transition_start:.3e}")
print(f"Transition end   Re_x = {Re_transition_end:.3e}")

# =============================================================================
# SURFACE WALKDOWN
# =============================================================================

x_vals = np.linspace(1e-4, plate_length, 700)

T_physical = []
T_conservative = []

transition_start_x = None
transition_end_x = None

for x in x_vals:

    # -------------------------------------------------------------------------
    # Reynolds Number
    # -------------------------------------------------------------------------

    Re_x = rho_edge * u_edge * x / mu_edge

    # -------------------------------------------------------------------------
    # Laminar Correlation
    # -------------------------------------------------------------------------

    Nu_lam = 0.332 * Re_x**0.5 * pr_edge**(1/3)

    h_lam = Nu_lam * k_edge / x

    q_lam = h_lam * (T_aw_lam - 1000)

    T_lam = equilibrium_temperature(q_lam)

    # -------------------------------------------------------------------------
    # Turbulent Correlation
    # -------------------------------------------------------------------------

    Nu_turb = 0.0296 * Re_x**0.8 * pr_edge**(1/3)

    h_turb = Nu_turb * k_edge / x

    q_turb = h_turb * (T_aw_turb - 1000)

    T_turb = equilibrium_temperature(q_turb)

    # -------------------------------------------------------------------------
    # Physical Transition Model
    # -------------------------------------------------------------------------

    if Re_x <= Re_transition_start:

        blend = 0.0

    elif Re_x >= Re_transition_end:

        blend = 1.0

    else:

        blend = (
            (Re_x - Re_transition_start)
            /
            (Re_transition_end - Re_transition_start)
        )

    if blend > 0 and transition_start_x is None:
        transition_start_x = x

    if blend >= 1 and transition_end_x is None:
        transition_end_x = x

    T_boundary = (1 - blend) * T_lam + blend * T_turb

    # -------------------------------------------------------------------------
    # Stagnation blending
    # -------------------------------------------------------------------------

    stag_blend = np.exp(-x / 0.03)

    T_phys = (
        stag_blend * T_stag
        +
        (1 - stag_blend) * T_boundary
    )

    T_cons = (
        stag_blend * T_stag
        +
        (1 - stag_blend) * T_turb
    )

    T_physical.append(T_phys)

    T_conservative.append(T_cons)

# =============================================================================
# REPORT
# =============================================================================

print("\n================================================")
print("TRANSITION LOCATIONS")
print("================================================")

print(f"Transition starts near x = {transition_start_x:.4f} m")

if transition_end_x is not None:

    print(f"Transition completes near x = {transition_end_x:.4f} m")

else:

    print("Transition not fully completed")

print("\n================================================")
print("FINAL TEMPERATURES")
print("================================================")

print(f"Physical model max T     = {max(T_physical):.2f} K")
print(f"Conservative model max T = {max(T_conservative):.2f} K")

# =============================================================================
# PLOTS
# =============================================================================

plt.figure(figsize=(13,6))

plt.plot(
    x_vals,
    T_physical,
    linewidth=3,
    label="Physical Transition Model"
)

plt.plot(
    x_vals,
    T_conservative,
    linewidth=3,
    linestyle="--",
    label="Conservative Fully-Turbulent Model"
)

plt.axhline(
    T_stag,
    linestyle=":",
    label="Fay-Riddell Stagnation"
)

if transition_start_x is not None:

    plt.axvline(
        transition_start_x,
        linestyle=":",
        label="Transition Start"
    )

if transition_end_x is not None:

    plt.axvline(
        transition_end_x,
        linestyle="--",
        label="Transition End"
    )

plt.xlabel("Distance Along Surface [m]")
plt.ylabel("Equilibrium Temperature [K]")

plt.title("Hypersonic Surface Temperature Distribution")

plt.grid(True)
plt.legend()

plt.show()