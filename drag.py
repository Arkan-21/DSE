import math


def isa_temperature(altitude_m):
    if altitude_m <= 11_000.0:
        return 288.15 - 0.0065 * altitude_m
    elif altitude_m <= 20_000.0:
        return 216.65
    elif altitude_m <= 32_000.0:
        return 216.65 + 0.001 * (altitude_m - 20_000.0)
    else:
        return 228.65


def isa_pressure(altitude_m):
    g0 = 9.80665
    R = 287.05

    if altitude_m <= 11_000.0:
        T0 = 288.15
        L = -0.0065
        T = T0 + L * altitude_m
        return 101325.0 * (T / T0) ** (-g0 / (L * R))

    elif altitude_m <= 20_000.0:
        T = 216.65
        p11 = 22632.06
        return p11 * math.exp(-g0 * (altitude_m - 11_000.0) / (R * T))

    elif altitude_m <= 32_000.0:
        T20 = 216.65
        p20 = 5474.89
        L = 0.001
        T = T20 + L * (altitude_m - 20_000.0)
        return p20 * (T / T20) ** (-g0 / (L * R))

    else:
        T = 228.65
        p32 = 868.02
        return p32 * math.exp(-g0 * (altitude_m - 32_000.0) / (R * T))


def isa_density(altitude_m):
    R = 287.05
    return isa_pressure(altitude_m) / (R * isa_temperature(altitude_m))


def speed_of_sound(altitude_m):
    gamma_air = 1.4
    R = 287.05
    T = isa_temperature(altitude_m)
    return math.sqrt(gamma_air * R * T)


def drag_from_speed_altitude(
    mass_kg,
    S_plan,
    altitude_m,
    velocity_m_s,
    CD0,
    k,
    flight_path_angle_deg=0.0,
    g=9.81,
):
    rho = isa_density(altitude_m)
    a = speed_of_sound(altitude_m)
    mach = velocity_m_s / a
    q = 0.5 * rho * velocity_m_s**2

    gamma = math.radians(flight_path_angle_deg)

    L_required = mass_kg * g * math.cos(gamma)
    CL = L_required / (q * S_plan)

    CD = CD0 + k * CL**2
    drag_N = q * S_plan * CD
    L_over_D = CL / CD

    return {
        "altitude_m": altitude_m,
        "velocity_m_s": velocity_m_s,
        "mach": mach,
        "rho": rho,
        "q": q,
        "CL": CL,
        "CD": CD,
        "L/D": L_over_D,
        "drag_N": drag_N,
    }


def ramjet_temperature_ratio_max(M0, gamma=1.29):
    """
    Temperature ratio T4/T0 giving maximum thrust per frontal area.
    """

    if M0 <= 0:
        raise ValueError("M0 must be positive.")

    numerator = (1.0 + ((gamma - 1.0) / 2.0) * M0**2) ** 3
    denominator = (1.0 + ((gamma - 1.0) / 4.0) * M0**2) ** 2

    return numerator / denominator


def ramjet_thrust_and_isp(
    mach,
    altitude_m,
    A3,
    gamma=1.29,
    T4_T0=None,
    h_f_mass=120.0e6,
    eta_thrust=0.85,
    eta_isp=0.60,
):

    if mach <= 0:
        raise ValueError("mach must be positive.")
    if A3 <= 0:
        raise ValueError("A3 must be positive.")
    if h_f_mass <= 0:
        raise ValueError("h_f_mass must be positive.")

    g0 = 9.80665

    p0 = isa_pressure(altitude_m)
    a0 = speed_of_sound(altitude_m)

    if T4_T0 is None:
        T4_T0 = ramjet_temperature_ratio_max(mach, gamma)

    stagnation_factor = 1.0 + ((gamma - 1.0) / 2.0) * mach**2
    root_term = math.sqrt(T4_T0 / stagnation_factor)

    thrust_ideal = (
        p0
        * A3
        * gamma
        * mach**2
        * (root_term - 1.0)
    )

    thrust_N = eta_thrust * thrust_ideal

    isp_velocity_ideal = (
        (h_f_mass / a0)
        * (gamma - 1.0)
        * mach
        / (stagnation_factor * (root_term + 1.0))
    )

    Isp_s = eta_isp * isp_velocity_ideal / g0

    info = {
        "mach": mach,
        "altitude_m": altitude_m,
        "p0": p0,
        "a0": a0,
        "A3": A3,
        "gamma": gamma,
        "T4_T0": T4_T0,
        "root_term": root_term,
        "thrust_ideal_N": thrust_ideal,
        "thrust_N": thrust_N,
        "isp_velocity_ideal": isp_velocity_ideal,
        "Isp_s": Isp_s,
        "eta_thrust": eta_thrust,
        "eta_isp": eta_isp,
    }

    return thrust_N, Isp_s, info


def drag_and_ramjet_at_condition(
    mass_kg,
    S_plan,
    altitude_m,
    velocity_m_s,
    CD0,
    k,
    A3_ramjet,
    flight_path_angle_deg=0.0,
):
    drag = drag_from_speed_altitude(
        mass_kg=mass_kg,
        S_plan=S_plan,
        altitude_m=altitude_m,
        velocity_m_s=velocity_m_s,
        CD0=CD0,
        k=k,
        flight_path_angle_deg=flight_path_angle_deg,
    )

    thrust_N, Isp_s, ramjet_info = ramjet_thrust_and_isp(
        mach=drag["mach"],
        altitude_m=altitude_m,
        A3=A3_ramjet,
        gamma=1.29,
        T4_T0=None,
        h_f_mass=120.0e6,
        eta_thrust=0.90,
        eta_isp=0.90,
    )

    return {
        **drag,
        "ramjet_thrust_N": thrust_N,
        "ramjet_Isp_s": Isp_s,
        "thrust_minus_drag_N": thrust_N - drag["drag_N"],
        "T_over_D": thrust_N / drag["drag_N"],
    }


result = drag_and_ramjet_at_condition(
    mass_kg=100_000,
    S_plan=450,
    altitude_m=35_000,
    velocity_m_s=1500,
    CD0=0.040,
    k=0.171,
    A3_ramjet=2 * 0.7739,
    flight_path_angle_deg=0.0,
)

for key, value in result.items():
    print(f"{key}: {value:.4f}")