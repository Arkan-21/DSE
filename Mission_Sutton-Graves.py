import pandas as pd
import numpy as np

# ── USER INPUT ───────────────────────────────────────────────────────────────
csv_file = "density_velocity_data.csv"

# Define radius here
rad = 1.5  # example value [m]

# ── Sutton-Graves heat flux function ────────────────────────────────────────
def sutgrav(rho, rad, v):
    q = 1.7415e-4 * np.sqrt(rho / rad) * v**3
    return q

# ── Read CSV ─────────────────────────────────────────────────────────────────
# Assumes:
# column 1 = density
# column 2 = velocity

df = pd.read_csv(csv_file)

# Rename columns for clarity
df.columns = ["rho", "v"]

# ── Apply function to all rows ───────────────────────────────────────────────
df["q"] = sutgrav(df["rho"], rad, df["v"])

# ── Compute statistics ───────────────────────────────────────────────────────
q_min = df["q"].min()
q_max = df["q"].max()

min_row = df.loc[df["q"].idxmin()]
max_row = df.loc[df["q"].idxmax()]

# ── Print results ────────────────────────────────────────────────────────────
print("\n========== Sutton-Graves Heating Analysis ==========\n")

print(f"Total datapoints analysed: {len(df)}")

print("\nMinimum q:")
print(f"q_min = {q_min:.6e} W/m^2")
print(f"rho   = {min_row['rho']:.6e} kg/m^3")
print(f"v     = {min_row['v']:.3f} m/s")

print("\nMaximum q:")
print(f"q_max = {q_max:.6e} W/m^2")
print(f"rho   = {max_row['rho']:.6e} kg/m^3")
print(f"v     = {max_row['v']:.3f} m/s")

print("\nq range:")
print(f"{q_min:.6e}  →  {q_max:.6e} W/m^2")

# ── Optional: save full dataset with q column ────────────────────────────────
output_file = "density_velocity_with_q.csv"

df.to_csv(output_file, index=False)

print(f"\nFull dataset saved to: {output_file}")

