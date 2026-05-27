# =============================================================================
# 1D MULTI-LAYER TPS HEAT SOAK SOLVER
# =============================================================================
#
# PURPOSE
# -----------------------------------------------------------------------------
# Computes transient through-thickness heat soak for a hypersonic TPS stack.
#
# FEATURES
# -----------------------------------------------------------------------------
# - Multiple material layers
# - User-defined material thicknesses
# - Non-uniform meshing
# - Applied heat flux on aerodynamic side
# - Radiation on hot outer wall
# - Cabin-side convection cooling
# - Cabin-side radiation
# - Explicit transient conduction solver
#
# USE CASE
# -----------------------------------------------------------------------------
# Mach 5 cruise vehicle at ~25 km altitude.
#
# Goal:
# Determine whether the TPS sufficiently protects the internal structure and
# cabin from aerodynamic heating during long-duration cruise.
#
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

from tps_materials import MATERIALS

# =============================================================================
# TPS STACK DEFINITION
# =============================================================================
#
# LEFT  = hot aerodynamic side
# RIGHT = cabin-facing side
#
# =============================================================================

layers = [

    {
        "material": "Gamma_TiAl",
        "thickness": 0.002
    },

    {
        "material": "Aerogel",
        "thickness": 0.030
    },

    {
        "material": "Ti6Al4V",
        "thickness": 0.003
    }

]

# =============================================================================
# AEROTHERMAL CONDITIONS
# =============================================================================

# Representative Mach 5 acreage heating

q_in = 50000      # [W/m²]

# =============================================================================
# ENVIRONMENT CONDITIONS
# =============================================================================

# Approximate atmosphere at 25 km altitude

T_ambient = 221.0      # [K]

space_temperature = 221.0

# =============================================================================
# CABIN CONDITIONS
# =============================================================================

# Cabin air temperature maintained by ECS

T_cabin_air = 295.0      # [K]

# Internal convection coefficient
#
# Represents effectiveness of:
# - cabin airflow
# - ECS cooling
# - internal heat rejection
#
# Typical strong forced convection:
# 30-100 W/m²K

h_cabin = 50.0           # [W/m²K]

# =============================================================================
# SURFACE EMISSIVITIES
# =============================================================================

emissivity_left = 0.85

emissivity_right = 0.80

sigma = 5.670374419e-8

# =============================================================================
# SIMULATION TIME
# =============================================================================

total_time_hours = 1.5

total_time = total_time_hours * 3600     # [s]

# =============================================================================
# NON-UNIFORM MESH SETTINGS
# =============================================================================
#
# More cells:
# - more accurate
# - slower
#
# Fewer cells:
# - faster
#
# =============================================================================

mesh_settings = {

    "Gamma_TiAl": 40,
    "Aerogel": 10,
    "Ti6Al4V": 5

}

# =============================================================================
# BUILD DOMAIN
# =============================================================================

x = []

dx_array = []

k_array = []
rho_array = []
cp_array = []

material_tracker = []

current_x = 0.0

for layer in layers:

    material_name = layer["material"]

    material = MATERIALS[material_name]

    thickness = layer["thickness"]

    n_cells = mesh_settings[material_name]

    dx_local = thickness / n_cells

    for i in range(n_cells):

        x.append(current_x)

        dx_array.append(dx_local)

        k_array.append(material["thermal_conductivity"])

        rho_array.append(material["density"])

        cp_array.append(material["specific_heat"])

        material_tracker.append(material_name)

        current_x += dx_local

# =============================================================================
# CONVERT TO NUMPY ARRAYS
# =============================================================================

x = np.array(x)

dx_array = np.array(dx_array)

k_array = np.array(k_array)

rho_array = np.array(rho_array)

cp_array = np.array(cp_array)

Nx = len(x)

# =============================================================================
# INITIAL CONDITIONS
# =============================================================================

T = np.full(Nx, T_cabin_air)

# =============================================================================
# STABILITY
# =============================================================================

alpha_array = k_array / (rho_array * cp_array)

alpha_max = np.max(alpha_array)

dx_min = np.min(dx_array)

# Conservative explicit stability criterion

dt = 0.4 * dx_min**2 / alpha_max

nt = int(total_time / dt)

print("\n========================================")
print("SIMULATION SUMMARY")
print("========================================")

print(f"Total TPS thickness : {x[-1]:.4f} m")
print(f"Total nodes         : {Nx}")
print(f"Minimum dx          : {dx_min:.6e} m")
print(f"Stable timestep     : {dt:.6e} s")
print(f"Total timesteps     : {nt}")
print(f"Simulation time     : {total_time_hours:.2f} hr")

print("========================================\n")

# =============================================================================
# TIME MARCHING
# =============================================================================

for n in range(nt):

    T_new = T.copy()

    # =========================================================================
    # INTERNAL NODES
    # =========================================================================

    for i in range(1, Nx - 1):

        alpha = alpha_array[i]

        dx_local = dx_array[i]

        T_new[i] = (
            T[i]
            + alpha * dt
            * (
                T[i + 1]
                - 2 * T[i]
                + T[i - 1]
            )
            / dx_local**2
        )

    # =========================================================================
    # LEFT BOUNDARY (HOT AERODYNAMIC SIDE)
    # =========================================================================
    #
    # Surface energy balance:
    #
    # q_aero = q_rad + q_cond
    #
    # Conduction into wall:
    #
    # q_cond = -k dT/dx
    #
    # Implemented using a ghost-node approach.
    #
    # =========================================================================

    dx_left = dx_array[0]

    k_left = k_array[0]

    alpha_left = alpha_array[0]

    # -------------------------------------------------------------------------
    # Radiation loss from hot surface
    # -------------------------------------------------------------------------

    q_rad_left = (
            emissivity_left
            * sigma
            * (
                    T[0] ** 4
                    - space_temperature ** 4
            )
    )

    # -------------------------------------------------------------------------
    # Net heat INTO structure
    # -------------------------------------------------------------------------

    q_net_left = q_in - q_rad_left

    # -------------------------------------------------------------------------
    # Ghost node temperature
    # -------------------------------------------------------------------------
    #
    # Derived from:
    #
    # -k (T1 - Tghost)/(2dx) = q_net
    #
    # -------------------------------------------------------------------------

    T_ghost = (
            T[1]
            + (2 * dx_left * q_net_left / k_left)
    )

    # -------------------------------------------------------------------------
    # Explicit finite difference update
    # -------------------------------------------------------------------------

    T_new[0] = (
            T[0]
            + alpha_left * dt
            * (
                    T[1]
                    - 2 * T[0]
                    + T_ghost
            )
            / dx_left ** 2
    )

    # =========================================================================
    # RIGHT BOUNDARY (CABIN SIDE)
    # =========================================================================

    dx_right = dx_array[-1]

    # Convective cooling into cabin air

    q_conv_right = (
        h_cabin
        * (
            T[-1]
            - T_cabin_air
        )
    )

    # Radiation into cabin

    q_rad_right = (
        emissivity_right
        * sigma
        * (
            T[-1]**4
            - T_cabin_air**4
        )
    )

    # Total heat leaving backside wall

    q_out_right = q_conv_right + q_rad_right

    # Boundary update

    T_new[-1] = (
        T[-1]
        + (
            2 * dt
            / (
                rho_array[-1]
                * cp_array[-1]
                * dx_right
            )
        )
        * (
            k_array[-1]
            * (T[-2] - T[-1])
            / dx_right
            - q_out_right
        )
    )

    # =========================================================================
    # UPDATE TEMPERATURE FIELD
    # =========================================================================

    T = T_new.copy()

# =============================================================================
# FINAL RESULTS
# =============================================================================

print("\n========================================")
print("FINAL TEMPERATURES AFTER 90 MINUTES")
print("========================================")

print(f"Hot-side temperature  : {T[0]:.2f} K")
print(f"Backside temperature  : {T[-1]:.2f} K")

print("========================================\n")

# =============================================================================
# FINAL TEMPERATURE PROFILE
# =============================================================================

plt.figure(figsize=(11, 6))

plt.plot(
    x,
    T,
    linewidth=3,
    label="Temperature Profile"
)

# -----------------------------------------------------------------------------
# MATERIAL INTERFACES
# -----------------------------------------------------------------------------

current_interface = 0.0

for layer in layers[:-1]:

    current_interface += layer["thickness"]

    plt.axvline(
        current_interface,
        linestyle="--",
        linewidth=1
    )

# -----------------------------------------------------------------------------
# LABELS
# -----------------------------------------------------------------------------

plt.xlabel("TPS Thickness [m]")

plt.ylabel("Temperature [K]")

plt.title("TPS Temperature Distribution After 90 Minutes")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()