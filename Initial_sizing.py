from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from math import exp, log
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# -----------------------------------------------------------------------------
# Geometry and wetted-area formulas
# -----------------------------------------------------------------------------

def tau(v_tot: float, s_plan: float) -> float:
    """Kuechemann tau: tau = V_tot / S_plan**1.5."""
    if s_plan <= 0:
        raise ValueError("s_plan must be positive.")
    return v_tot / (s_plan ** 1.5)


def total_volume_from_tau(tau_value: float, s_plan: float) -> float:
    """Rearranged tau formula: V_tot = tau * S_plan**1.5."""
    if s_plan <= 0:
        raise ValueError("s_plan must be positive.")
    return tau_value * (s_plan ** 1.5)


def k_w_from_tau(tau, configuration="wing_body"):
    """
    Estimate wetted-to-planform area ratio K_w as a function of Küchemann tau.

    Polynomial fits:
        waverider:
            K_w = 5632.2*tau^4 - 3106*tau^3 + 621.37*tau^2 - 46.623*tau + 3.8167

        wing_body:
            K_w = 473.07*tau^4 - 366.2*tau^3 + 110.36*tau^2 - 9.6647*tau + 2.9019

        blended_body:
            K_w = 18.594*tau^2 + 0.0084*tau + 2.4274
    """

    tau = np.asarray(tau)

    if np.any(tau <= 0):
        raise ValueError("tau must be positive.")

    if configuration == "waverider":
        return (
            5632.2 * tau**4
            - 3106.0 * tau**3
            + 621.37 * tau**2
            - 46.623 * tau
            + 3.8167
        )

    elif configuration == "wing_body":
        return (
            473.07 * tau**4
            - 366.2 * tau**3
            + 110.36 * tau**2
            - 9.6647 * tau
            + 2.9019
        )

    elif configuration == "blended_body":
        return (
            18.594 * tau**2
            + 0.0084 * tau
            + 2.4274
        )

    else:
        raise ValueError(
            "configuration must be one of: "
            "'blended_body', 'wing_body', or 'waverider'"
        )


def wetted_area(k_w: float, s_plan: float) -> float:
    """S_wet = K_w(tau) * S_plan."""
    return k_w * s_plan

# -----------------------------------------------------------------------------
# Aerodynamic performance formulas
# -----------------------------------------------------------------------------

def lift_to_drag(mach: float, tau_value: float, A: float = 6.0, B: float = 2.0) -> float:
    r"""
    L/D = [A(M+B)/M] * [(1.0128 - 0.2797 ln(tau/0.03)) / (1 - M**2/673)].
    """
    if mach <= 0:
        raise ValueError("mach must be positive.")
    if tau_value <= 0:
        raise ValueError("tau_value must be positive.")

    numerator = 1.0128 - 0.2797 * log(tau_value / 0.03)
    denominator = 1.0 - (mach ** 2) / 673.0
    return (A * (mach + B) / mach) * (numerator / denominator)

def speed_of_sound(altitude_m: float) -> float:
    """Approximate speed of sound at cruise altitude (m/s).

    Baseline report S02-M05-SY01 sets the cruise altitude band at
    25-34 km. We default to 28 km (mid-band) for nominal sizing.
    """
    # ISA: a = sqrt(gamma * R * T)
    if altitude_m <= 11_000.0:
        T = 288.15 - 0.0065 * altitude_m
    elif altitude_m <= 20_000.0:
        T = 216.65
    elif altitude_m <= 32_000.0:
        T = 216.65 + 0.001 * (altitude_m - 20_000.0)
    else:
        T = 228.65
    return float(np.sqrt(1.4 * 287.05 * T))

# -----------------------------------------------------------------------------
# Range and fuel-fraction formulas
# -----------------------------------------------------------------------------

def range_factor(theta: float, q_cc: float, l_over_d: float) -> float:
    """RF = theta * Q_cc * (L/D)."""
    return theta * q_cc * l_over_d


def range_factor_from_delta_v(delta_v: float, isp: float, l_over_d: float) -> float:
    """RF = delta_v * Isp * (L/D)."""
    return delta_v * isp * l_over_d


def range_factor_from_mach(speed_of_sound: float, mach: float, isp: float, l_over_d: float) -> float:
    """RF = a * M * Isp * (L/D), where a is speed of sound and M is Mach number."""
    return speed_of_sound * mach * isp * l_over_d


def mission_range(rf: float, fuel_fraction: float) -> float:
    """Range = -RF * ln(1 - ff)."""
    if rf <= 0:
        raise ValueError("rf must be positive.")
    if not 0 <= fuel_fraction < 1:
        raise ValueError("fuel_fraction must satisfy 0 <= ff < 1.")
    return -rf * log(1.0 - fuel_fraction)


def fuel_fraction_from_range(range_value: float, rf: float) -> float:
    """ff = 1 - exp(-Range / RF)."""
    if rf <= 0:
        raise ValueError("rf must be positive.")
    return 1.0 - exp(-range_value / rf)


def fuel_fraction_breguet(range_value: float, isp: float, delta_v: float, l_over_d: float) -> float:
    """W_fuel / TOGW = 1 - exp[-Range / (Isp * delta_v * L/D)]."""
    return fuel_fraction_from_range(range_value, range_factor_from_delta_v(delta_v, isp, l_over_d))


def fuel_fraction(w_fuel: float, togw: float) -> float:
    """ff = W_fuel / TOGW."""
    if togw <= 0:
        raise ValueError("togw must be positive.")
    return w_fuel / togw


def fuel_weight(ff: float, togw: float) -> float:
    """W_fuel = ff * TOGW."""
    return ff * togw


# -----------------------------------------------------------------------------
# Volume formulas
# -----------------------------------------------------------------------------

def payload_volume(w_pay: float, rho_pay: float) -> float:
    """V_pay = W_pay / rho_pay."""
    if rho_pay <= 0:
        raise ValueError("rho_pay must be positive.")
    return w_pay / rho_pay


def fuel_volume(w_fuel: float, rho_fuel: float) -> float:
    """V_fuel = W_fuel / rho_fuel."""
    if rho_fuel <= 0:
        raise ValueError("rho_fuel must be positive.")
    return w_fuel / rho_fuel


def void_volume(v_tot: float, eta_v: float) -> float:
    """V_void = V_tot * (1 - eta_v)."""
    if not 0 <= eta_v <= 1:
        raise ValueError("eta_v must satisfy 0 <= eta_v <= 1.")
    return v_tot * (1.0 - eta_v)


def total_required_volume(v_pay: float, v_fuel: float, v_void: float) -> float:
    """V_tot = V_pay + V_fuel + V_void."""
    return v_pay + v_fuel + v_void


def available_volume_residual(v_available: float, v_required: float) -> float:
    """Residual for volume convergence: positive means available volume exceeds required volume."""
    return v_available - v_required


# -----------------------------------------------------------------------------
# Weight formulas
# -----------------------------------------------------------------------------

def systems_weight(r_sys: float, togw: float) -> float:
    """W_sys = (W_sys / TOGW) * TOGW = r_sys * TOGW."""
    return r_sys * togw


def propulsion_weight(w_prop_over_thrust: float, l_over_d: float, togw: float) -> float:
    r"""
    W_prop = (W_prop / Thrust) * (1 / (L/D)) * TOGW.
    """
    if l_over_d <= 0:
        raise ValueError("l_over_d must be positive.")
    return w_prop_over_thrust * (1.0 / l_over_d) * togw


def propulsion_weight_from_etw(etw: float, l_over_d: float, togw: float) -> float:
    r"""
    W_prop = (1 / ETW) * (1 / (L/D)) * TOGW.
    Here ETW = Thrust / W_prop.
    """
    if etw <= 0:
        raise ValueError("etw must be positive.")
    if l_over_d <= 0:
        raise ValueError("l_over_d must be positive.")
    return (1.0 / etw) * (1.0 / l_over_d) * togw


def structural_weight(i_str: float, k_w: float, s_plan: float) -> float:
    r"""
    W_str = (W_str / S_wet) * (S_wet / S_plan) * S_plan
          = I_str * K_w * S_plan.
    """
    return i_str * k_w * s_plan


def takeoff_gross_weight(w_pay: float, w_fuel: float, w_sys: float, w_prop: float, w_str: float) -> float:
    """TOGW = W_pay + W_fuel + W_sys + W_prop + W_str."""
    return w_pay + w_fuel + w_sys + w_prop + w_str


def available_weight_residual(togw_available: float, togw_required: float) -> float:
    """Residual for weight convergence: positive means available weight exceeds required weight."""
    return togw_available - togw_required


# -----------------------------------------------------------------------------
# Optional design-evaluation helper
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SizingInputs:
    mach: float
    range_value: float
    altitude_m: float
    w_pay: float
    rho_pay: float
    rho_fuel: float
    eta_v: float
    r_sys: float
    tau_value: float
    s_plan: float
    i_str: float   
    isp: float
    etw: float
    TOGW: float
    configuration: str = "wing_body"


def evaluate_design(inputs: SizingInputs) -> dict[str, float]:
    """
    Evaluate one sizing point for a chosen tau, S_plan, and TOGW.

    This computes one non-iterated design point. The iteration functions below
    call this repeatedly until weight and volume residuals converge.
    """

    k_w = k_w_from_tau(inputs.tau_value, inputs.configuration)
    s_wet = wetted_area(k_w, inputs.s_plan)

    v_available = total_volume_from_tau(inputs.tau_value, inputs.s_plan)

    l_d = lift_to_drag(inputs.mach, inputs.tau_value)

    a = speed_of_sound(inputs.altitude_m)

    rf = range_factor_from_mach(
        a,
        inputs.mach,
        inputs.isp,
        l_d,
    )

    ff = fuel_fraction_from_range(inputs.range_value, rf)

    togw_available = inputs.TOGW

    w_fuel = fuel_weight(ff, togw_available)
    w_sys = systems_weight(inputs.r_sys, togw_available)
    w_prop = propulsion_weight_from_etw(inputs.etw, l_d, togw_available)
    w_str = structural_weight(inputs.i_str, k_w, inputs.s_plan)

    togw_required = takeoff_gross_weight(
        inputs.w_pay,
        w_fuel,
        w_sys,
        w_prop,
        w_str,
    )

    v_pay = payload_volume(inputs.w_pay, inputs.rho_pay)
    v_fuel = fuel_volume(w_fuel, inputs.rho_fuel)
    v_void = void_volume(v_available, inputs.eta_v)
    v_required = total_required_volume(v_pay, v_fuel, v_void)

    return {
        "tau": inputs.tau_value,
        "S_plan": inputs.s_plan,
        "K_w": k_w,
        "S_wet": s_wet,
        "L_over_D": l_d,
        "RF": rf,
        "fuel_fraction": ff,
        "TOGW": togw_available,
        "TOGW_required": togw_required,
        "W_pay": inputs.w_pay,
        "W_fuel": w_fuel,
        "W_sys": w_sys,
        "W_prop": w_prop,
        "W_str": w_str,
        "V_available": v_available,
        "V_pay": v_pay,
        "V_fuel": v_fuel,
        "V_void": v_void,
        "V_required": v_required,
        "volume_residual": available_volume_residual(v_available, v_required),
        "weight_residual": available_weight_residual(
            togw_available,
            togw_required,
        ),
    }


def converge_togw(
    inputs: SizingInputs,
    tolerance_kg: float = 1.0,
    relaxation: float = 0.6,
    max_iterations: int = 100,
) -> tuple[SizingInputs, dict[str, float]]:
    """
    Iterate TOGW until TOGW_available ≈ TOGW_required.

    Positive weight_residual:
        TOGW guess is too high.

    Negative weight_residual:
        TOGW guess is too low.
    """

    current_inputs = inputs

    for iteration in range(max_iterations):
        result = evaluate_design(current_inputs)

        togw_available = result["TOGW"]
        togw_required = result["TOGW_required"]

        error = togw_required - togw_available

        if abs(error) < tolerance_kg:
            result["togw_iterations"] = iteration
            return current_inputs, result

        new_togw = togw_available + relaxation * error

        if new_togw <= 0:
            raise RuntimeError("TOGW iteration became non-physical.")

        current_inputs = replace(
            current_inputs,
            TOGW=new_togw,
        )

    result = evaluate_design(current_inputs)
    result["togw_iterations"] = max_iterations
    return current_inputs, result


def converge_s_plan_and_togw(
    inputs: SizingInputs,
    volume_tolerance_m3: float = 1.0,
    weight_tolerance_kg: float = 1.0,
    s_plan_relaxation: float = 0.5,
    togw_relaxation: float = 0.6,
    max_outer_iterations: int = 100,
    max_inner_iterations: int = 100,
) -> tuple[SizingInputs, dict[str, float]]:
    """
    Converge both S_plan and TOGW for a fixed tau.

    Outer loop:
        Adjust S_plan until volume residual is small.

    Inner loop:
        Adjust TOGW until weight residual is small for the current S_plan.
    """

    current_inputs = inputs

    for outer_iteration in range(max_outer_iterations):

        # First converge weight for this geometry.
        current_inputs, result = converge_togw(
            current_inputs,
            tolerance_kg=weight_tolerance_kg,
            relaxation=togw_relaxation,
            max_iterations=max_inner_iterations,
        )

        v_available = result["V_available"]
        v_required = result["V_required"]
        volume_residual = result["volume_residual"]

        if abs(volume_residual) < volume_tolerance_m3:
            result["s_plan_iterations"] = outer_iteration
            return current_inputs, result

        raw_correction = (v_required / v_available) ** (2.0 / 3.0)

        # Relax the update to avoid over-correcting.
        correction = 1.0 + s_plan_relaxation * (raw_correction - 1.0)
        # Prevent unstable jumps.
        correction = max(0.5, min(1.5, correction))

        new_s_plan = current_inputs.s_plan * correction

        if new_s_plan <= 0:
            raise RuntimeError("S_plan iteration became non-physical.")

        current_inputs = replace(
            current_inputs,
            s_plan=new_s_plan,
        )

    result["s_plan_iterations"] = max_outer_iterations
    return current_inputs, result


def sweep_tau_values(
    base_inputs: SizingInputs,
    tau_values: list[float],
) -> list[dict[str, float]]:
    """
    Run the full sizing convergence for several tau values.
    """

    results = []

    for tau_value in tau_values:
        trial_inputs = replace(
            base_inputs,
            tau_value=tau_value,
        )

        converged_inputs, result = converge_s_plan_and_togw(trial_inputs)

        result["converged_tau"] = converged_inputs.tau_value
        result["converged_S_plan"] = converged_inputs.s_plan
        result["converged_TOGW"] = converged_inputs.TOGW

        results.append(result)

    return results


def sweep_tau_and_istr(
    base_inputs: SizingInputs,
    tau_values: list[float],
    i_str_values: list[float],
) -> list[dict[str, float]]:
    """
    Run convergence for several tau and I_str combinations.
    """

    results = []

    for i_str in i_str_values:
        for tau_value in tau_values:

            trial_inputs = replace(
                base_inputs,
                tau_value=tau_value,
                i_str=i_str,
            )

            converged_inputs, result = converge_s_plan_and_togw(trial_inputs)

            result["converged_tau"] = converged_inputs.tau_value
            result["converged_S_plan"] = converged_inputs.s_plan
            result["converged_TOGW"] = converged_inputs.TOGW
            result["input_I_str"] = i_str

            results.append(result)

    return results


import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting


def plot_solution_space(
    base_inputs: SizingInputs,
    s_plan_values: np.ndarray,
    tau_values: np.ndarray,
    configuration: str = "wing_body",
    save_path: str | None = None,
):
    """
    Plot V_tot_required and V_tot_available as 3D surfaces over S_plan and tau.

    For each S_plan-tau combination:
        1. Set tau and S_plan
        2. Converge TOGW for that geometry
        3. Store V_required and V_available

    This produces a figure similar to the solution-space plots in the paper.
    """

    S_grid, tau_grid = np.meshgrid(s_plan_values, tau_values)

    V_available_grid = np.zeros_like(S_grid, dtype=float)
    V_required_grid = np.zeros_like(S_grid, dtype=float)

    for i in range(tau_grid.shape[0]):
        for j in range(tau_grid.shape[1]):

            trial_inputs = replace(
                base_inputs,
                tau_value=tau_grid[i, j],
                s_plan=S_grid[i, j],
            )

            # Converge only TOGW here.
            # We do NOT converge S_plan, because S_plan is the x-axis variable.
            _, result = converge_togw(trial_inputs)

            V_available_grid[i, j] = result["V_available"]
            V_required_grid[i, j] = result["V_required"]

    scale = 1e4

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        S_grid,
        tau_grid,
        V_required_grid / scale,
        color="red",
        alpha=0.85,
        linewidth=0.2,
        edgecolor="k",
    )

    ax.plot_surface(
        S_grid,
        tau_grid,
        V_available_grid / scale,
        color="blue",
        alpha=0.75,
        linewidth=0.2,
        edgecolor="k",
    )

    ax.set_xlabel(r"$S_{plan}$ [m²]", labelpad=10)
    ax.set_ylabel(r"$\tau$", labelpad=10)
    ax.set_zlabel(r"$V_{tot}$ [m³]", labelpad=10)

    ax.text2D(
        0.08,
        0.90,
        r"$\times 10^4$",
        transform=ax.transAxes,
        fontsize=11,
    )

    legend_handles = [
        Patch(facecolor="red", edgecolor="red", label=r"$V_{tot,req}$"),
        Patch(facecolor="blue", edgecolor="blue", label=r"$V_{tot,av}$"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
    )

    ax.set_title(f"Solution space for {configuration.replace('_', '-')}", pad=20)

    # Adjust viewing angle to resemble your reference figure.
    ax.view_init(elev=25, azim=225)

    # Optional: make grid lines dotted.
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"].update(
            {
                "linewidth": 0.6,
                "linestyle": ":",
            }
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# def plot_converged_istr21_results(results: list[dict[str, float]]):
#     """
#     Plot converged sizing variables versus S_plan for only I_str = 21.

#     Each point is one converged tau case.
#     """

#     # Keep only I_str = 21 results
#     istr21_results = [
#         result for result in results
#         if abs(result["input_I_str"] - 21.0) < 1e-6
#     ]

#     # Sort by tau so the curve connects points in tau order
#     istr21_results = sorted(
#         istr21_results,
#         key=lambda result: result["tau"]
#     )

#     s_plan = np.array([result["S_plan"] for result in istr21_results])
#     tau = np.array([result["tau"] for result in istr21_results])

#     # At convergence, V_available ≈ V_required.
#     # Use V_available as V_tot.
#     v_tot = np.array([result["V_available"] for result in istr21_results])
#     s_wet = np.array([result["S_wet"] for result in istr21_results])
#     w_str = np.array([result["W_str"] for result in istr21_results])
#     w_fuel = np.array([result["W_fuel"] for result in istr21_results])
#     togw = np.array([result["TOGW"] for result in istr21_results])
#     v_fuel = np.array([result["V_fuel"] for result in istr21_results])

#     # OWE = systems + propulsion + structure
#     owe = np.array([
#         result["W_sys"] + result["W_prop"] + result["W_str"]
#         for result in istr21_results
#     ])

#     plots = {
#         "V_tot": {
#             "values": v_tot,
#             "ylabel": r"$V_{tot}$ [m³]",
#             "title": r"$V_{tot}$ vs $S_{plan}$, $I_{str}=21$",
#         },
#         "S_wet": {
#             "values": s_wet,
#             "ylabel": r"$S_{wet}$ [m²]",
#             "title": r"$S_{wet}$ vs $S_{plan}$, $I_{str}=21$",
#         },
#         "W_str": {
#             "values": w_str,
#             "ylabel": r"$W_{str}$ [kg]",
#             "title": r"$W_{str}$ vs $S_{plan}$, $I_{str}=21$",
#         },
#         "OWE": {
#             "values": owe,
#             "ylabel": r"OWE [kg]",
#             "title": r"OWE vs $S_{plan}$, $I_{str}=21$",
#         },
#         "W_fuel": {
#             "values": w_fuel,
#             "ylabel": r"$W_{fuel}$ [kg]",
#             "title": r"$W_{fuel}$ vs $S_{plan}$, $I_{str}=21$",
#         },
#         "TOGW": {
#             "values": togw,
#             "ylabel": r"TOGW [kg]",
#             "title": r"TOGW vs $S_{plan}$, $I_{str}=21$",
#         },
#         "V_fuel": {
#             "values": v_fuel,
#             "ylabel": r"$V_{fuel}$ [m³]",
#             "title": r"$V_{fuel}$ vs $S_{plan}$, $I_{str}=21$",
#         },
#     }

#     for variable_name, plot_data in plots.items():

#         plt.figure(figsize=(7, 5))

#         plt.plot(
#             s_plan,
#             plot_data["values"],
#             marker="x",
#             linewidth=1.5,
#         )

#         # Add tau labels next to each point
#         for x, y, tau_value in zip(s_plan, plot_data["values"], tau):
#             plt.annotate(
#                 rf"$\tau={tau_value:.2f}$",
#                 xy=(x, y),
#                 xytext=(5, 5),
#                 textcoords="offset points",
#                 fontsize=9,
#             )

#         plt.xlabel(r"$S_{plan}$ [m²]")
#         plt.ylabel(plot_data["ylabel"])
#         plt.title(plot_data["title"])
#         plt.grid(True, linestyle=":")
#         plt.tight_layout()
#         plt.show()

def plot_converged_istr21_results(results: list[dict[str, float]]):
    """
    Plot converged sizing variables versus S_plan for only I_str = 21,
    all in one figure window.
    """

    istr24_results = [
        result for result in results
        if abs(result["input_I_str"] - 24.0) < 1e-6
    ]

    istr24_results = sorted(istr24_results, key=lambda result: result["tau"])

    s_plan = np.array([result["S_plan"] for result in istr24_results])
    tau    = np.array([result["tau"]    for result in istr24_results])

    v_tot  = np.array([result["V_available"] for result in istr24_results])
    s_wet  = np.array([result["S_wet"]       for result in istr24_results])
    w_str  = np.array([result["W_str"]       for result in istr24_results])
    w_fuel = np.array([result["W_fuel"]      for result in istr24_results])
    togw   = np.array([result["TOGW"]        for result in istr24_results])
    v_fuel = np.array([result["V_fuel"]      for result in istr24_results])
    owe    = np.array([
        result["W_sys"] + result["W_prop"] + result["W_str"]
        for result in istr24_results
    ])

    plots = [
        ("V_tot",  v_tot,  r"$V_{tot}$ [m³]",   r"$V_{tot}$"),
        ("S_wet",  s_wet,  r"$S_{wet}$ [m²]",   r"$S_{wet}$"),
        ("W_str",  w_str,  r"$W_{str}$ [kg]",   r"$W_{str}$"),
        ("OWE",    owe,    r"OWE [kg]",          r"OWE"),
        ("W_fuel", w_fuel, r"$W_{fuel}$ [kg]",  r"$W_{fuel}$"),
        ("TOGW",   togw,   r"TOGW [kg]",         r"TOGW"),
        ("V_fuel", v_fuel, r"$V_{fuel}$ [m³]",  r"$V_{fuel}$"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(r"Converged sizing results, $I_{str} = 24$", fontsize=13)

    # Flatten so we can index linearly; hide the unused 9th panel
    axes_flat = axes.flatten()

    # Delete unused subplot axes
    for ax in axes_flat[len(plots):]:
        fig.delaxes(ax)

    axes_flat = axes_flat[:len(plots)]

    for ax, (_, values, ylabel, short_title) in zip(axes_flat, plots):

        ax.plot(s_plan, values, marker="x", linewidth=1.5, color="steelblue")

        for x, y, tau_value in zip(s_plan, values, tau):
            ax.annotate(
                rf"$\tau={tau_value:.2f}$",
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

        ax.set_xlabel(r"$S_{plan}$ [m²]", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(short_title, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    example = SizingInputs(
        mach=5.0,
        range_value=9_500_000.0,
        altitude_m=28_000.0,
        w_pay=7000,
        rho_pay=100.0,
        rho_fuel=70.0,
        eta_v=0.7,
        r_sys=0.16,
        tau_value=0.16,
        s_plan=900.0,
        i_str=21.0,
        isp=1800.0,
        etw=8,
        TOGW=250_000.0,
    )

    converged_inputs, result = converge_s_plan_and_togw(example)

    '''
    print("\nConverged input values")
    print("----------------------")
    print(f"tau:    {converged_inputs.tau_value:.6g}")
    print(f"S_plan: {converged_inputs.s_plan:.6g} m²")
    print(f"TOGW:   {converged_inputs.TOGW:.6g} kg")

    print("\nConverged design results")
    print("------------------------")
    for key, value in result.items():
        print(f"{key}: {value:.6g}")
    '''
    
    tau_values = np.round(np.arange(0.06, 0.22 + 0.01, 0.01), 2)
    i_str_values = [15.0, 18.0, 21.0, 24.0]

    results = sweep_tau_and_istr(
        example,
        tau_values,
        i_str_values,
    )


    plot_converged_istr21_results(results)

    '''
    print("\nTau and I_str sensitivity sweep")
    print("-------------------------------")
    print(
        "I_str    tau      S_plan [m²]    TOGW [kg]      "
        "W_str [kg]     S_wet [m²]     fuel frac    V_res [m3]  W_res [kg]"
    )

    for result in results:
        print(
            f"{result['input_I_str']:<9.1f}"
            f"{result['tau']:<9.3f}"
            f"{result['S_plan']:<15.3f}"
            f"{result['TOGW']:<15.3f}"
            f"{result['W_str']:<15.3f}"
            f"{result['S_wet']:<15.3f}"
            f"{result['fuel_fraction']:<10.4f}"
            f"{result['volume_residual']:<13.3f}"
            f"{result['weight_residual']:<13.3f}"   
        )
    '''    

    '''        
    s_plan_values = np.linspace(1.0, 2000.0, 40)
    tau_values = np.linspace(0.001, 0.40, 40)

    plot_solution_space(
        base_inputs=example,
        s_plan_values=s_plan_values,
        tau_values=tau_values,
        configuration="wing_body",
        save_path="solution_space_wing_body.png",
        )
    '''