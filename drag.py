import math
import numpy as np
import matplotlib.pyplot as plt



def isa_temperature(altitude_m):
    if altitude_m <= 11_000:
        return 288.15 - 0.0065 * altitude_m
    elif altitude_m <= 20_000:
        return 216.65
    elif altitude_m <= 32_000:
        return 216.65 + 0.001 * (altitude_m - 20_000)
    else:
        return 228.65


def isa_pressure(altitude_m):
    g0 = 9.80665
    R = 287.05

    if altitude_m <= 11_000:
        T0 = 288.15
        L = -0.0065
        T = T0 + L * altitude_m
        return 101325 * (T / T0) ** (-g0 / (L * R))

    elif altitude_m <= 20_000:
        T = 216.65
        p11 = 22632.06
        return p11 * math.exp(-g0 * (altitude_m - 11_000) / (R * T))

    elif altitude_m <= 32_000:
        T20 = 216.65
        p20 = 5474.89
        L = 0.001
        T = T20 + L * (altitude_m - 20_000)
        return p20 * (T / T20) ** (-g0 / (L * R))

    else:
        T = 228.65
        p32 = 868.02
        return p32 * math.exp(-g0 * (altitude_m - 32_000) / (R * T))


def isa_density(altitude_m):
    R = 287.05
    return isa_pressure(altitude_m) / (R * isa_temperature(altitude_m))


def speed_of_sound(altitude_m):
    gamma_air = 1.4
    R = 287.05
    return math.sqrt(gamma_air * R * isa_temperature(altitude_m))


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def aerodynamic_coefficients(mach, altitude_m):
    """
    Continuous approximate CD0 and k model.

    CD = CD0 + k * CL^2

    This is a conceptual sizing model, not CFD.
    It increases CD0 near transonic/supersonic/hypersonic speeds
    and slightly modifies k with Mach and altitude.
    """

    h_km = altitude_m / 1000.0

    # Compressibility / wave drag buildup
    transonic = smoothstep((mach - 0.75) / 0.45)
    supersonic = smoothstep((mach - 1.2) / 1.8)
    hypersonic = smoothstep((mach - 3.0) / 2.0)

    # Altitude correction: lower Reynolds number at high altitude usually hurts drag
    altitude_factor = smoothstep((h_km - 15.0) / 25.0)

    CD0 = (
        0.018
        + 0.010 * transonic
        + 0.012 * supersonic
        + 0.018 * hypersonic
        + 0.006 * altitude_factor
    )

    k = (
        0.075
        + 0.025 * transonic
        + 0.045 * supersonic
        + 0.080 * hypersonic
        + 0.015 * altitude_factor
    )

    return CD0, k


def drag_from_speed_altitude(
    mass_kg,
    S_plan,
    altitude_m,
    velocity_m_s,
    flight_path_angle_deg=0.0,
    g=9.81,
):
    rho = isa_density(altitude_m)
    a = speed_of_sound(altitude_m)
    mach = velocity_m_s / a
    q = 0.5 * rho * velocity_m_s**2

    CD0, k = aerodynamic_coefficients(mach, altitude_m)

    gamma = math.radians(flight_path_angle_deg)
    L_required = mass_kg * g * math.cos(gamma)

    CL = L_required / (q * S_plan)
    CD = CD0 + k * CL**2
    drag_N = q * S_plan * CD

    return {
        "altitude_m": altitude_m,
        "velocity_m_s": velocity_m_s,
        "mach": mach,
        "rho": rho,
        "q": q,
        "CD0": CD0,
        "k": k,
        "CL": CL,
        "CD": CD,
        "L/D": CL / CD,
        "drag_N": drag_N,
    }


mass_kg = 100_000
S_plan = 450

fixed_altitude_m = 35_000
fixed_velocity_m_s = 1500

speed_range = np.linspace(200, 1800, 200)
altitude_range = np.linspace(0, 40_000, 200)


drag_vs_speed = []

for V in speed_range:
    result = drag_from_speed_altitude(
        mass_kg=mass_kg,
        S_plan=S_plan,
        altitude_m=fixed_altitude_m,
        velocity_m_s=V,
    )
    drag_vs_speed.append(result["drag_N"])

plt.figure()
plt.plot(speed_range, np.array(drag_vs_speed) / 1000)
plt.xlabel("Velocity [m/s]")
plt.ylabel("Drag [kN]")
plt.title(f"Drag vs Speed at {fixed_altitude_m / 1000:.0f} km")
plt.grid(True)
plt.show()


drag_vs_altitude = []

for h in altitude_range:
    result = drag_from_speed_altitude(
        mass_kg=mass_kg,
        S_plan=S_plan,
        altitude_m=h,
        velocity_m_s=fixed_velocity_m_s,
    )
    drag_vs_altitude.append(result["drag_N"])

plt.figure()
plt.plot(altitude_range / 1000, np.array(drag_vs_altitude) / 1000)
plt.xlabel("Altitude [km]")
plt.ylabel("Drag [kN]")
plt.title(f"Drag vs Altitude at {fixed_velocity_m_s:.0f} m/s")
plt.grid(True)
plt.show()


example = drag_from_speed_altitude(
    mass_kg=mass_kg,
    S_plan=S_plan,
    altitude_m=fixed_altitude_m,
    velocity_m_s=fixed_velocity_m_s,
)

print("\nExample condition")
print("-----------------")
for key, value in example.items():
    print(f"{key}: {value:.4f}")