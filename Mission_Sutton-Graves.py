import pandas as pd
import numpy as np

# ── USER INPUT ───────────────────────────────────────────────────────────────
csv_file = "density_velocity_database.csv"

# Leading-edge nose radius [m]
rad = 0.02

# Material properties
emissivity = 0.85

# Stefan-Boltzmann constant
sigma = 5.670374419e-8  # W/m²/K⁴

# ── Sutton-Graves heat flux function ────────────────────────────────────────
def sutgrav(rho, rad, v):

    q = 1.7415e-4 * np.sqrt(rho / rad) * v**3

    return q

# ── Equilibrium wall temperature ────────────────────────────────────────────
def equilibrium_temperature(q, emissivity):

    T_eq = (q / (emissivity * sigma))**0.25

    return T_eq

# ── Read CSV ─────────────────────────────────────────────────────────────────
# Assumes:
# column 1 = density [kg/m³]
# column 2 = velocity [m/s]

df = pd.read_csv(csv_file)

# ── Compute all heat fluxes ──────────────────────────────────────────────────
q_values = sutgrav(df["rho"], rad, df["v"])

# ── Find maximum heat flux ───────────────────────────────────────────────────
q_max_index = np.argmax(q_values)

q_max = q_values.iloc[q_max_index]

rho_max = df["rho"].iloc[q_max_index]
v_max   = df["v"].iloc[q_max_index]

# ── Compute equilibrium temperature ──────────────────────────────────────────
T_eq_max = equilibrium_temperature(q_max, emissivity)

# ── Print results ────────────────────────────────────────────────────────────
print("\n========== Hypersonic TPS Screening ==========\n")

print(f"Total datapoints analysed: {len(df)}")

print("\nWorst-case heating condition:")

print(f"rho = {rho_max:.6e} kg/m^3")
print(f"v   = {v_max:.3f} m/s")

print("\nMaximum Sutton-Graves heat flux:")

print(f"q_max = {q_max:.6e} W/m^2")

print("\nRadiative equilibrium temperature:")

print(f"T_eq = {T_eq_max:.2f} K")
print(f"T_eq = {T_eq_max - 273.15:.2f} °C")

print("\nMaterial properties used:")

print(f"Emissivity = {emissivity:.3f}")
print(f"Nose radius = {rad:.3f} m")

