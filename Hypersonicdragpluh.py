import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTS
# =============================================================================

g = 9.81
R = 287.0
gamma = 1.4

# =============================================================================
# VEHICLE PARAMETERS
# =============================================================================

m_tog = 111389.0                 # [kg]
S_ref = 425.0                    # [m²]

accel_g_target = 0.15            # longitudinal acceleration target [g]

# Characteristic length
L_ref = 35.0                     # [m]

# Wetted area approximation
S_wet = 2.2 * S_ref

# Effective aspect ratio
AR = 2.2

# Oswald efficiency
e = 0.80

# =============================================================================
# FLIGHT CONDITION
# =============================================================================

altitude = 30000.0               # [m]

# =============================================================================
# ISA ATMOSPHERE
# =============================================================================

def isa_atmosphere(h):

    g0 = 9.80665
    T0 = 288.15
    P0 = 101325.0
    L = -0.0065

    # Troposphere
    if h <= 11000:

        T = T0 + L * h

        P = P0 * (T / T0)**(-g0 / (L * R))

    # Lower stratosphere
    else:

        T = 216.65

        P11 = P0 * (216.65 / T0)**(-g0 / (L * R))

        P = P11 * np.exp(
            -g0 * (h - 11000)
            / (R * T)
        )

    rho = P / (R * T)

    return T, P, rho

# =============================================================================
# SUTHERLAND VISCOSITY
# =============================================================================

def viscosity_sutherland(T):

    return (
        1.458e-6
        * T**1.5
        / (T + 110.4)
    )

# =============================================================================
# ATMOSPHERIC CONDITIONS
# =============================================================================

T_inf, P_inf, rho_inf = isa_atmosphere(altitude)

a_inf = np.sqrt(gamma * R * T_inf)

mu_inf = viscosity_sutherland(T_inf)

# =============================================================================
# MACH RANGE
# =============================================================================

mach_range = np.linspace(3.0, 6.0, 100)

# =============================================================================
# STORAGE
# =============================================================================

cl_list = []
cd_list = []
alpha_list = []
thrust_list = []
ld_ratio_list = []
reynolds_list = []

cd_friction_list = []
cd_wave_list = []
cd_induced_list = []

# =============================================================================
# MAIN LOOP
# =============================================================================

for M in mach_range:

    # -------------------------------------------------------------------------
    # FREESTREAM CONDITIONS
    # -------------------------------------------------------------------------

    V = M * a_inf

    q_inf = 0.5 * rho_inf * V**2

    W = m_tog * g

    # -------------------------------------------------------------------------
    # REQUIRED LIFT COEFFICIENT
    # -------------------------------------------------------------------------

    CL = W / (q_inf * S_ref)

    # -------------------------------------------------------------------------
    # HYPERSONIC LIFT SLOPE
    #
    # CL_alpha ≈ 4 / sqrt(M² - 1)
    # -------------------------------------------------------------------------

    CL_alpha = 4 / np.sqrt(M**2 - 1)

    alpha_rad = CL / CL_alpha

    alpha_deg = np.degrees(alpha_rad)

    # -------------------------------------------------------------------------
    # REYNOLDS NUMBER
    # -------------------------------------------------------------------------

    Re = (
        rho_inf
        * V
        * L_ref
        / mu_inf
    )

    # -------------------------------------------------------------------------
    # TURBULENT SKIN FRICTION
    # -------------------------------------------------------------------------

    Cf_incompressible = 0.455 / (
        np.log10(Re)**2.58
    )

    # Hypersonic viscous growth correction
    Cf = (
        Cf_incompressible
        * (1 + 0.08 * M**1.5)
    )

    # -------------------------------------------------------------------------
    # FRICTION DRAG
    # -------------------------------------------------------------------------

    CD_friction = Cf * (S_wet / S_ref)

    # -------------------------------------------------------------------------
    # HYPERSONIC WAVE DRAG
    # -------------------------------------------------------------------------

    k_wave = 1.2

    CD_wave = (
        k_wave
        * (1 + 0.10 * (M - 1)**1.7)
        * alpha_rad**2
    )

    # -------------------------------------------------------------------------
    # INDUCED DRAG
    # -------------------------------------------------------------------------

    CD_induced = (
        CL**2
        / (np.pi * AR * e)
    )

    # -------------------------------------------------------------------------
    # BASE / VOLUME DRAG
    #
    # Represents entropy generation,
    # volumetric compression losses,
    # shock interaction effects
    # -------------------------------------------------------------------------

    CD_base = (
        0.012
        + 0.0025 * (M - 3)
    )

    # -------------------------------------------------------------------------
    # TOTAL DRAG COEFFICIENT
    # -------------------------------------------------------------------------

    CD_total = (
        CD_friction
        + CD_wave
        + CD_induced
        + CD_base
    )

    # -------------------------------------------------------------------------
    # DRAG FORCE
    # -------------------------------------------------------------------------

    D = q_inf * S_ref * CD_total

    # -------------------------------------------------------------------------
    # REQUIRED THRUST
    # -------------------------------------------------------------------------

    T_required = (
        D
        + m_tog * accel_g_target * g
    )

    # -------------------------------------------------------------------------
    # STORE RESULTS
    # -------------------------------------------------------------------------

    cl_list.append(CL)

    cd_list.append(CD_total)

    alpha_list.append(alpha_deg)

    thrust_list.append(T_required / 1000)

    ld_ratio_list.append(CL / CD_total)

    reynolds_list.append(Re)

    cd_friction_list.append(CD_friction)

    cd_wave_list.append(CD_wave)

    cd_induced_list.append(CD_induced)

# =============================================================================
# PLOTTING
# =============================================================================

fig, axs = plt.subplots(
    3,
    2,
    figsize=(14, 13)
)

# =============================================================================
# ANGLE OF ATTACK
# =============================================================================

axs[0, 0].plot(
    mach_range,
    alpha_list,
    lw=2
)

axs[0, 0].set_title(
    'Required Angle of Attack'
)

axs[0, 0].set_xlabel(
    'Mach Number'
)

axs[0, 0].set_ylabel(
    'Angle of Attack [deg]'
)

axs[0, 0].grid(True, alpha=0.3)

# =============================================================================
# L/D
# =============================================================================

axs[0, 1].plot(
    mach_range,
    ld_ratio_list,
    lw=2
)

axs[0, 1].set_title(
    'Aerodynamic Efficiency (L/D)'
)

axs[0, 1].set_xlabel(
    'Mach Number'
)

axs[0, 1].set_ylabel(
    'L/D'
)

axs[0, 1].grid(True, alpha=0.3)

# =============================================================================
# DRAG POLAR
# =============================================================================

axs[1, 0].plot(
    cd_list,
    cl_list,
    lw=2
)

axs[1, 0].set_title(
    'Hypersonic Drag Polar'
)

axs[1, 0].set_xlabel(
    r'$C_D$'
)

axs[1, 0].set_ylabel(
    r'$C_L$'
)

axs[1, 0].grid(True, alpha=0.3)

# =============================================================================
# THRUST REQUIRED
# =============================================================================

axs[1, 1].plot(
    mach_range,
    thrust_list,
    lw=2
)

axs[1, 1].set_title(
    'Required Thrust for 0.15g Acceleration'
)

axs[1, 1].set_xlabel(
    'Mach Number'
)

axs[1, 1].set_ylabel(
    'Thrust [kN]'
)

axs[1, 1].grid(True, alpha=0.3)

# =============================================================================
# REYNOLDS NUMBER
# =============================================================================

axs[2, 0].plot(
    mach_range,
    reynolds_list,
    lw=2
)

axs[2, 0].set_title(
    'Reynolds Number'
)

axs[2, 0].set_xlabel(
    'Mach Number'
)

axs[2, 0].set_ylabel(
    r'$Re$'
)

axs[2, 0].set_yscale('log')

axs[2, 0].grid(True, which='both', alpha=0.3)

# =============================================================================
# DRAG BREAKDOWN
# =============================================================================

axs[2, 1].plot(
    mach_range,
    cd_friction_list,
    label='Friction Drag',
    lw=2
)

axs[2, 1].plot(
    mach_range,
    cd_wave_list,
    label='Wave Drag',
    lw=2
)

axs[2, 1].plot(
    mach_range,
    cd_induced_list,
    label='Induced Drag',
    lw=2
)

axs[2, 1].plot(
    mach_range,
    cd_list,
    label='Total Drag',
    lw=3
)

axs[2, 1].set_title(
    'Drag Coefficient Breakdown'
)

axs[2, 1].set_xlabel(
    'Mach Number'
)

axs[2, 1].set_ylabel(
    r'$C_D$'
)

axs[2, 1].grid(True, alpha=0.3)

axs[2, 1].legend()

# =============================================================================

plt.tight_layout()

plt.show()