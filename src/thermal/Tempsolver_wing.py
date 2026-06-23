import numpy as np
import matplotlib.pyplot as plt

from tps_materials import MATERIALS

# =============================================================================
# USER INPUTS
# =============================================================================

layer_stack = [

    ("CVI-C/SiC", 0.004),
    ('IMI_Effective',     0.010),
    ("Gamma_TiAl",  0.003)

]

wing_area = 100.0          # m² represented

initial_temperature = 295.0
ambient_temperature = 295.0

dt = 0.002
n_nodes = 100

sigma = 5.670374419e-8

# =============================================================================
# HEAT FLUX PROFILE
# =============================================================================

peak_heat_flux = 40000.0     # W/m²

ramp_up_time = 1
steady_time = 2500.0
ramp_down_time = 1.0

post_cooling_time = 1.0

simulation_time = (
    ramp_up_time
    + steady_time
    + ramp_down_time
    + post_cooling_time
)

# =============================================================================
# HEAT FLUX FUNCTION
# =============================================================================

def heat_flux_profile(t):

    if t <= ramp_up_time:

        return (
            peak_heat_flux
            * t
            / ramp_up_time
        )

    elif t <= ramp_up_time + steady_time:

        return peak_heat_flux

    elif t <= (
        ramp_up_time
        + steady_time
        + ramp_down_time
    ):

        local_time = (
            t
            - ramp_up_time
            - steady_time
        )

        return (
            peak_heat_flux
            * (
                1.0
                - local_time / ramp_down_time
            )
        )

    else:

        return 0.0


# =============================================================================
# BUILD MESH
# =============================================================================

total_thickness = sum(
    thickness
    for _, thickness in layer_stack
)

dx = total_thickness / n_nodes

x = np.linspace(
    0,
    total_thickness,
    n_nodes
)

T = np.ones(n_nodes) * initial_temperature

rho = np.zeros(n_nodes)
cp = np.zeros(n_nodes)
k = np.zeros(n_nodes)
eps = np.zeros(n_nodes)

layer_ranges = {}

current_x = 0.0

for material_name, thickness in layer_stack:

    material = MATERIALS[material_name]

    start = int(
        current_x
        / total_thickness
        * n_nodes
    )

    current_x += thickness

    end = int(
        current_x
        / total_thickness
        * n_nodes
    )

    layer_ranges[material_name] = (
        start,
        end
    )

    rho[start:end] = material["density"]
    cp[start:end] = material["specific_heat"]
    k[start:end] = material["thermal_conductivity"]

    emissivity = material["emissivity"]

    if emissivity is None:
        emissivity = 0.8

    eps[start:end] = emissivity

# =============================================================================
# INTERFACE CONDUCTIVITIES
# =============================================================================

k_interface = np.zeros(n_nodes - 1)

for i in range(n_nodes - 1):

    k_interface[i] = (
        2.0
        * k[i]
        * k[i + 1]
        /
        (k[i] + k[i + 1])
    )

# =============================================================================
# STABILITY CHECK
# =============================================================================

alpha_max = np.max(
    k / (rho * cp)
)

Fo = (
    alpha_max
    * dt
    / dx**2
)

print(
    f"Maximum Fourier Number = {Fo:.4f}"
)

if Fo > 0.5:

    print(
        "WARNING: Explicit scheme may be unstable."
    )

# =============================================================================
# MASS CALCULATION
# =============================================================================

print("\nMass Breakdown")

total_mass = 0.0

for material_name, thickness in layer_stack:

    density = MATERIALS[
        material_name
    ]["density"]

    mass = (
        density
        * thickness
        * wing_area
    )

    total_mass += mass

    print(
        f"{material_name:20s}"
        f"{mass:10.1f} kg"
    )

print(
    f"\nTotal TPS Mass = "
    f"{total_mass:.1f} kg"
)

# =============================================================================
# STORAGE
# =============================================================================

n_steps = int(
    simulation_time / dt
)

time_hist = np.zeros(n_steps)

outer_temp_hist = np.zeros(n_steps)

heat_flux_hist = np.zeros(n_steps)

layer_max_temp = {

    name: initial_temperature

    for name, _ in layer_stack

}

layer_avg_temp = {

    name: []

    for name, _ in layer_stack

}

# =============================================================================
# ENERGY ACCOUNTING
# =============================================================================

energy_in = 0.0
energy_radiated = 0.0

# =============================================================================
# TRANSIENT SOLVER
# =============================================================================

for step in range(n_steps):

    t = step * dt

    time_hist[step] = t

    T_old = T.copy()

    q_aero = heat_flux_profile(t)

    heat_flux_hist[step] = q_aero

    q_rad = (

        eps[0]
        * sigma
        * (
            T_old[0]**4
            - ambient_temperature**4
        )

    )

    q_net = q_aero - q_rad

    energy_in += q_aero * dt
    energy_radiated += q_rad * dt

    # ==========================================================
    # INTERIOR NODES
    # ==========================================================

    for i in range(1, n_nodes - 1):

        q_left = (

            -k_interface[i - 1]
            * (
                T_old[i]
                - T_old[i - 1]
            )
            / dx

        )

        q_right = (

            -k_interface[i]
            * (
                T_old[i + 1]
                - T_old[i]
            )
            / dx

        )

        net_flux = q_left - q_right

        T[i] = (

            T_old[i]

            + dt
            * net_flux

            / (
                rho[i]
                * cp[i]
                * dx
            )

        )

    # ==========================================================
    # OUTER SURFACE
    # ==========================================================

    q_cond = (

        -k_interface[0]
        * (
            T_old[1]
            - T_old[0]
        )
        / dx

    )

    net_flux = q_net - q_cond

    T[0] = (

        T_old[0]

        + dt
        * net_flux

        / (
            rho[0]
            * cp[0]
            * dx
        )

    )

    # ==========================================================
    # INNER SURFACE
    # ==========================================================

    q_cond_inner = (

        -k_interface[-1]
        * (
            T_old[-1]
            - T_old[-2]
        )
        / dx

    )

    T[-1] = (

        T_old[-1]

        + dt
        * q_cond_inner

        / (
            rho[-1]
            * cp[-1]
            * dx
        )

    )

    # ==========================================================
    # RECORDS
    # ==========================================================

    outer_temp_hist[step] = T[0]

    for layer_name, (s, e) in layer_ranges.items():

        current_max = np.max(
            T[s:e]
        )

        if (
            current_max
            > layer_max_temp[layer_name]
        ):

            layer_max_temp[layer_name] = (
                current_max
            )

        layer_avg_temp[layer_name].append(

            np.mean(T[s:e])

        )

# =============================================================================
# ENERGY BALANCE
# =============================================================================

stored_energy = np.sum(

    rho
    * cp
    * (
        T
        - initial_temperature
    )
    * dx

)

print("\nEnergy Balance")

print(
    f"Incident Energy      = "
    f"{energy_in/1e6:.2f} MJ/m²"
)

print(
    f"Radiated Energy      = "
    f"{energy_radiated/1e6:.2f} MJ/m²"
)

print(
    f"Stored Energy        = "
    f"{stored_energy/1e6:.2f} MJ/m²"
)

# =============================================================================
# RESULTS
# =============================================================================

print(
    "\nMaximum Layer Temperatures"
)

for name, temp in layer_max_temp.items():

    limit = MATERIALS[name].get(
        "max_service_temp"
    )

    print(
        f"{name:20s}"
        f"{temp:10.1f} K"
        f"   Limit: {limit}"
    )

print(
    f"\nPeak Surface Temperature:"
    f" {np.max(outer_temp_hist):.1f} K"
)

# =============================================================================
# PLOTS
# =============================================================================

plt.figure(figsize=(10,6))

plt.plot(
    time_hist,
    heat_flux_hist / 1000
)

plt.xlabel("Time [s]")
plt.ylabel("Heat Flux [kW/m²]")
plt.title("Applied Heat Flux")
plt.grid()

plt.show()

plt.figure(figsize=(10,6))

plt.plot(
    time_hist,
    outer_temp_hist
)

plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.title("Outer Surface Temperature")
plt.grid()

plt.show()

plt.figure(figsize=(10,6))

plt.plot(
    x,
    T,
    linewidth=3
)

plt.xlabel("Thickness [m]")
plt.ylabel("Temperature [K]")
plt.title("Final Through-Thickness Temperature")
plt.grid()

plt.show()