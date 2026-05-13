import math
from dataclasses import dataclass


# =============================================================================
# ISA atmosphere and speed of sound
# =============================================================================

def isa_temperature(altitude_m: float) -> float:
    """ISA temperature model up to and above 32 km, matching the sizing code."""
    if altitude_m <= 11_000.0:
        return 288.15 - 0.0065 * altitude_m
    elif altitude_m <= 20_000.0:
        return 216.65
    elif altitude_m <= 32_000.0:
        return 216.65 + 0.001 * (altitude_m - 20_000.0)
    else:
        return 228.65


def speed_of_sound(altitude_m: float) -> float:
    """Speed of sound from ISA temperature."""
    gamma_air = 1.4
    R = 287.05
    return math.sqrt(gamma_air * R * isa_temperature(altitude_m))


# =============================================================================
# Switch calculation
# =============================================================================

@dataclass
class SwitchResult:
    target_mach: float
    target_reached: bool
    t_s: float
    h_m: float
    V_m_s: float
    Vx_m_s: float
    Vy_m_s: float
    x_m: float
    mach: float
    speed_of_sound_m_s: float
    t_to_cruise_s: float
    mach_at_cruise_height: float


def find_ascent_state_at_mach(
    target_mach: float,
    h_cruise: float,
    gamma_deg: float,
    acc_tot: float,
    max_bisection_iter: int = 100,
) -> SwitchResult:
    """
    Find when a constant-acceleration ascent reaches a target Mach number.

    This matches the logic used in your main sizing code:

        acc_x = acc_tot * cos(gamma)
        acc_y = acc_tot * sin(gamma)
        h     = 0.5 * acc_y * t^2
        V     = acc_tot * t
        Mach  = V / a(h)

    If the target Mach is not reached before h_cruise, the returned state is
    the state at cruise altitude and target_reached=False.
    """
    gamma_rad = math.radians(gamma_deg)
    acc_x = acc_tot * math.cos(gamma_rad)
    acc_y = acc_tot * math.sin(gamma_rad)

    if acc_y <= 0:
        raise ValueError("gamma_deg must give positive vertical acceleration.")
    if target_mach <= 0:
        raise ValueError("target_mach must be positive.")
    if h_cruise <= 0:
        raise ValueError("h_cruise must be positive.")
    if acc_tot <= 0:
        raise ValueError("acc_tot must be positive.")

    t_to_cruise = math.sqrt(2.0 * h_cruise / acc_y)

    def mach_at_time(t: float) -> float:
        h = 0.5 * acc_y * t**2
        V = acc_tot * t
        return V / speed_of_sound(h)

    mach_at_cruise_height = mach_at_time(t_to_cruise)

    if mach_at_cruise_height < target_mach:
        # Target not reached before cruise altitude.
        t = t_to_cruise
        target_reached = False
    else:
        # Bisection solve for target Mach.
        t_low = 0.0
        t_high = t_to_cruise

        for _ in range(max_bisection_iter):
            t_mid = 0.5 * (t_low + t_high)

            if mach_at_time(t_mid) < target_mach:
                t_low = t_mid
            else:
                t_high = t_mid

        t = 0.5 * (t_low + t_high)
        target_reached = True

    h = 0.5 * acc_y * t**2
    V = acc_tot * t
    Vx = acc_x * t
    Vy = acc_y * t
    x = 0.5 * acc_x * t**2
    a = speed_of_sound(h)
    mach = V / a

    return SwitchResult(
        target_mach=target_mach,
        target_reached=target_reached,
        t_s=t,
        h_m=h,
        V_m_s=V,
        Vx_m_s=Vx,
        Vy_m_s=Vy,
        x_m=x,
        mach=mach,
        speed_of_sound_m_s=a,
        t_to_cruise_s=t_to_cruise,
        mach_at_cruise_height=mach_at_cruise_height,
    )


def print_switch_result(result: SwitchResult) -> None:
    print("\nMach switch calculation")
    print("-----------------------")
    print(f"Target Mach:                 {result.target_mach:.3f}")
    print(f"Target reached before cruise: {result.target_reached}")
    print(f"Time at switch:              {result.t_s:.3f} s")
    print(f"Altitude at switch:          {result.h_m:.3f} m")
    print(f"Altitude at switch:          {result.h_m / 1000.0:.3f} km")
    print(f"Downrange at switch:         {result.x_m / 1000.0:.3f} km")
    print(f"Velocity at switch:          {result.V_m_s:.3f} m/s")
    print(f"Vx at switch:                {result.Vx_m_s:.3f} m/s")
    print(f"Vy at switch:                {result.Vy_m_s:.3f} m/s")
    print(f"Speed of sound at switch:    {result.speed_of_sound_m_s:.3f} m/s")
    print(f"Mach at switch:              {result.mach:.6f}")
    print(f"Time to cruise altitude:     {result.t_to_cruise_s:.3f} s")
    print(f"Mach at cruise altitude:     {result.mach_at_cruise_height:.6f}")


if __name__ == "__main__":
    # Match your current main-code assumptions.
    gamma_mission = 7.0
    h_cruise = 35_000.0
    acc_tot = 0.15 * 9.81

    # Change this value to test other turbo-to-ramjet switch Mach numbers.
    M_turbo_to_ram = 2.5

    result = find_ascent_state_at_mach(
        target_mach=M_turbo_to_ram,
        h_cruise=h_cruise,
        gamma_deg=gamma_mission,
        acc_tot=acc_tot,
    )

    print_switch_result(result)
