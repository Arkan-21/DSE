import numpy as np
import matplotlib.pyplot as plt

from tps_materials import MATERIALS

# =============================================================================
# USER INPUTS
# =============================================================================

layer_stack = [
    ("SiC_SiC_CMC", 0.004),
    ("Aerogel",     0.030),
    ("Ti6Al4V",     0.002)
]

wall_area = 50.0                 # m²


dt = 0.005                      # s

initial_temperature = 295.0      # K

ambient_radiation_temp = 220.0   # K

cabin_setpoint = 295.0           # K
cabin_h = 8.0                    # W/m²K

n_nodes_total = 100

sigma = 5.670374419e-8


# =============================================================================
# HEAT FLUX PROFILE INPUTS
# =============================================================================

peak_heat_flux = 40000.0      # [W/m²]

ramp_up_time = 500         # [s]
steady_time = 2500           # [s]
ramp_down_time = 500       # [s]
post_cooling_time = 1

# Total mission duration implied by profile
simulation_time = (
    ramp_up_time
    + steady_time
    + ramp_down_time
    + post_cooling_time
)

# =============================================================================
# HEAT FLUX HISTORY
# =============================================================================

# =============================================================================
# HEAT FLUX HISTORY
# =============================================================================

def heat_flux_profile(t):

    # --------------------------------
    # Ramp-up phase
    # --------------------------------

    if t <= ramp_up_time:

        return peak_heat_flux * t / ramp_up_time

    # --------------------------------
    # Constant heating phase
    # --------------------------------

    elif t <= ramp_up_time + steady_time:

        return peak_heat_flux

    # --------------------------------
    # Ramp-down phase
    # --------------------------------

    elif t <= ramp_up_time + steady_time + ramp_down_time:

        t_ramp_down = (
            t
            - ramp_up_time
            - steady_time
        )

        return (
            peak_heat_flux
            * (
                1
                - t_ramp_down / ramp_down_time
            )
        )

    # --------------------------------
    # Post-cooling phase
    # --------------------------------

    else:

        return 0.0


# =============================================================================
# BUILD MESH
# =============================================================================

total_thickness = sum(t for _, t in layer_stack)

dx = total_thickness / n_nodes_total

x = np.linspace(0, total_thickness, n_nodes_total)

T = np.ones(n_nodes_total) * initial_temperature

rho = np.zeros(n_nodes_total)
cp = np.zeros(n_nodes_total)
k = np.zeros(n_nodes_total)
eps = np.zeros(n_nodes_total)

layer_ranges = {}

current_x = 0.0

for material_name, thickness in layer_stack:

    material = MATERIALS[material_name]

    start = int(current_x / total_thickness * n_nodes_total)

    current_x += thickness

    end = int(current_x / total_thickness * n_nodes_total)

    layer_ranges[material_name] = (start, end)
    interface_locations = []

    for _, (s, e) in layer_ranges.items():
        interface_locations.append(e)

    rho[start:end] = material["density"]
    cp[start:end] = material["specific_heat"]
    k[start:end] = material["thermal_conductivity"]

    emissivity = material.get("emissivity")

    if emissivity is None:
        emissivity = 0.8

    eps[start:end] = emissivity



# =============================================================================
# INTERFACE CONDUCTIVITIES (HARMONIC MEAN)
# =============================================================================

k_interface = np.zeros(n_nodes_total - 1)

for i in range(n_nodes_total - 1):

    k_interface[i] = (
        2.0 * k[i] * k[i + 1]
        / (k[i] + k[i + 1])
    )

# =============================================================================
# STABILITY CHECK
# =============================================================================

alpha_max = np.max(k / (rho * cp))

Fo_max = alpha_max * dt / dx**2

print(
    f"Maximum Fourier Number = {Fo_max:.4f}"
)

if Fo_max > 0.5:
    print(
        "WARNING: Explicit scheme may be unstable."
    )


# =============================================================================
# TPS MASS
# =============================================================================

tps_mass = 0

for mat_name, thickness in layer_stack:

    density = MATERIALS[mat_name]["density"]

    if density is not None:

        tps_mass += density * thickness * wall_area

print("\nTPS Mass = {:.1f} kg".format(tps_mass))

# =============================================================================
# STORAGE
# =============================================================================

n_steps = int(simulation_time / dt)

time_hist = np.zeros(n_steps)

outer_temp_hist = np.zeros(n_steps)
inner_temp_hist = np.zeros(n_steps)

hvac_power_hist = np.zeros(n_steps)

layer_max_temp = {
    name: initial_temperature
    for name, _ in layer_stack
}

# =============================================================================
# TRANSIENT SOLVER
# =============================================================================
# =============================================================================
# TRANSIENT SOLVER
# =============================================================================

energy_in = 0.0
radiated_energy = 0.0
cabin_energy = 0.0

for step in range(n_steps):

    t = step * dt

    time_hist[step] = t

    T_old = T.copy()

    # ==========================================================
    # OUTER BOUNDARY
    # ==========================================================

    q_aero = heat_flux_profile(t)

    q_rad = (
        eps[0]
        * sigma
        * (
            T_old[0]**4
            - ambient_radiation_temp**4
        )
    )

    q_net_surface = q_aero - q_rad

    energy_in += q_net_surface * dt
    radiated_energy += q_rad * dt

    # ==========================================================
    # INTERIOR NODES
    # ==========================================================

    for i in range(1, n_nodes_total - 1):

        q_left = (
            -k_interface[i - 1]
            * (T_old[i] - T_old[i - 1])
            / dx
        )

        q_right = (
            -k_interface[i]
            * (T_old[i + 1] - T_old[i])
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
    # OUTER NODE UPDATE
    # ==========================================================

    q_cond_out = (
        -k_interface[0]
        * (
            T_old[1]
            - T_old[0]
        )
        / dx
    )

    net_flux = (
        q_net_surface
        - q_cond_out
    )

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
    # CABIN SIDE HVAC
    # ==========================================================

    q_cabin = (
        cabin_h
        * (
            T_old[-1]
            - cabin_setpoint
        )
    )

    if q_cabin < 0:
        q_cabin = 0.0

    hvac_power = q_cabin * wall_area

    hvac_power_hist[step] = hvac_power

    cabin_energy += q_cabin * dt

    # ==========================================================
    # INNER NODE UPDATE
    # ==========================================================

    q_cond_in = (
        -k_interface[-1]
        * (
            T_old[-1]
            - T_old[-2]
        )
        / dx
    )

    net_flux = (
        q_cond_in
        - q_cabin
    )

    T[-1] = (
        T_old[-1]
        + dt
        * net_flux
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
    inner_temp_hist[step] = T[-1]

    for layer_name, (s, e) in layer_ranges.items():

        current_max = np.max(T[s:e])

        if current_max > layer_max_temp[layer_name]:

            layer_max_temp[layer_name] = current_max


# =============================================================================
# STORED THERMAL ENERGY
# =============================================================================

stored_energy = np.sum(
    rho
    * cp
    * (T - initial_temperature)
    * dx
)

print(
    "Stored energy per m² =",
    stored_energy / 1e6,
    "MJ/m²"
)

print(
    "Numerical energy added =",
    energy_in / 1e6,
    "MJ/m²"
)

print(
    "Radiated energy =",
    radiated_energy / 1e6,
    "MJ/m²"
)

print(
    "Cabin energy removed =",
    cabin_energy / 1e6,
    "MJ/m²"
)

energy_balance_error = (
    energy_in
    - radiated_energy
    - cabin_energy
    - stored_energy
)

print(
    "Energy balance error =",
    energy_balance_error / 1e6,
    "MJ/m²"
)







# =============================================================================
# RESULTS
# =============================================================================
print("\nInterface Temperatures")

for idx in interface_locations[:-1]:

    print(
        f"Node {idx:3d} : {T[idx]:.1f} K"
    )

total_input_energy = np.trapz(
    [heat_flux_profile(t) for t in time_hist],
    time_hist
)

print(
    "Incident energy per m² =",
    total_input_energy/1e6,
    "MJ/m²"
)



heat_flux_hist = np.array(
    [heat_flux_profile(t) for t in time_hist]
)

plt.figure(figsize=(10,5))

plt.plot(
    time_hist,
    heat_flux_hist/1000
)

plt.xlabel("Time [s]")
plt.ylabel("Heat Flux [kW/m²]")
plt.title("Applied Aerodynamic Heat Flux")

plt.grid()


cumulative_energy = np.zeros_like(time_hist)

for i in range(1, len(time_hist)):
    cumulative_energy[i] = (
        cumulative_energy[i-1]
        + hvac_power_hist[i] * dt
    )

print("\n==============================")
print("MAXIMUM LAYER TEMPERATURES")
print("==============================")

for layer_name, temp in layer_max_temp.items():

    print(
        f"{layer_name:20s} : {temp:8.1f} K"
    )

print("\nPeak Outer Surface Temp:")
print("{:.1f} K".format(np.max(outer_temp_hist)))

print("\nPeak Inner Wall Temp:")
print("{:.1f} K".format(np.max(inner_temp_hist)))

print("\nPeak HVAC Power:")
print("{:.1f} kW".format(
    np.max(hvac_power_hist)/1000
))

total_hvac_energy = np.trapz(
    hvac_power_hist,
    time_hist
)

print(
    "\nTotal HVAC Energy: {:.2f} MJ".format(
        total_hvac_energy / 1e6
    )
)

# =============================================================================
# PLOTS
# =============================================================================

plt.figure(figsize=(10,5))
plt.plot(time_hist, outer_temp_hist,
         label="Outer Surface")

plt.plot(time_hist, inner_temp_hist,
         label="Inner Wall")

plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.legend()
plt.grid()

plt.figure(figsize=(10,5))
plt.plot(time_hist,
         hvac_power_hist/1000)

plt.xlabel("Time [s]")
plt.ylabel("HVAC Power [kW]")
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(x, T)

plt.xlabel("Wall Thickness [m]")
plt.ylabel("Temperature [K]")

plt.title("Final Temperature Distribution")
plt.grid()


plt.figure(figsize=(10,5))

plt.plot(
    time_hist,
    hvac_power_hist/1000
)

plt.xlabel("Time [s]")
plt.ylabel("HVAC Cooling Power [kW]")
plt.title("Required HVAC Cooling Power")
plt.grid()



plt.figure(figsize=(10,5))

plt.plot(
    time_hist,
    cumulative_energy/1e6
)

plt.xlabel("Time [s]")
plt.ylabel("HVAC Energy [MJ]")
plt.title("Cumulative HVAC Energy Required")
plt.grid()

plt.show()