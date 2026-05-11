import numpy as np
import pandas as pd

from isa_atmosphere import T, density


# ── Flight profile computation ──────────────────────────────────────────────
def compute_flight_profile(gamma,
                           h_cruise,
                           acc_tot=0.15 * 9.81,
                           x_sample=-1):

    gamma_rad = np.radians(gamma)

    a_cruise = np.sqrt(1.4 * 287.05 * T(h_cruise))

    V_cruise = 5 * a_cruise

    # Ascent
    acc_x = acc_tot * np.cos(gamma_rad)
    acc_y = acc_tot * np.sin(gamma_rad)

    dv_y_to_cruise = np.sqrt(2 * acc_y * h_cruise)

    t_to_cruise = dv_y_to_cruise / acc_y

    dv_x_to_cruise = acc_x * t_to_cruise

    dx_to_cruise = dv_x_to_cruise**2 / (2 * acc_x)

    # Horizontal acceleration to Mach 5
    dx_hor_acc = (
        V_cruise**2 - dv_x_to_cruise**2
    ) / (2 * acc_tot)

    cruise_cond_start_x = dx_to_cruise + dx_hor_acc

    # ── Determine local trajectory state ───────────────────────────────────
    if x_sample < dx_to_cruise:

        # Ascent
        t_sample = np.sqrt(2 * x_sample / acc_x)

        dv_y_sample = acc_y * t_sample
        dv_x_sample = acc_x * t_sample

        v_sample = np.sqrt(
            dv_x_sample**2 + dv_y_sample**2
        )

        h_sample = dv_y_sample**2 / (2 * acc_y)

    else:

        # Horizontal acceleration
        x_acc = x_sample - dx_to_cruise

        v_sample = np.sqrt(
            2 * acc_tot * x_acc
            + dv_x_to_cruise**2
        )

        h_sample = h_cruise

    density_sample = density(h_sample)

    return (
        density_sample,
        v_sample,
        cruise_cond_start_x
    )


# ── Generate database up to Mach 5 cruise onset ────────────────────────────
def generate_density_velocity_database(
        gammas,
        heights,
        step=100,
        filename="density_velocity_database.csv"):

    rows = []

    for gamma in gammas:

        for h_cruise in heights:

            print(
                f"Processing gamma={gamma}°, "
                f"h={h_cruise/1000:.0f} km"
            )

            # Get cruise start location
            _, _, cruise_start_x = compute_flight_profile(
                gamma,
                h_cruise,
                x_sample=0
            )

            # Only sample UNTIL cruise begins
            x_samples = np.arange(
                0,
                cruise_start_x + step,
                step
            )

            for x in x_samples:

                rho, velocity, _ = compute_flight_profile(
                    gamma,
                    h_cruise,
                    x_sample=x
                )

                rows.append({
                    "gamma_deg": gamma,
                    "h_cruise_m": h_cruise,
                    "x_m": x,
                    "rho": rho,
                    "v": velocity
                })

    df = pd.DataFrame(rows)

    df.to_csv(filename, index=False)

    print(f"\nCSV generated: {filename}")

    print(f"Total datapoints: {len(df)}")

    print("\nFirst few rows:\n")

    print(df.head())


# ── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    gammas = [5, 10, 15]

    heights = [25e3, 30e3, 35e3]

    generate_density_velocity_database(
        gammas,
        heights,
        step=100
    )