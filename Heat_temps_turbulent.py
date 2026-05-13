import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# =============================================================================
# PLOT STYLE
# =============================================================================

# =============================================================================
# FIGURE OUTPUT DIRECTORY
# =============================================================================

fig_dir = Path("figures")

fig_dir.mkdir(exist_ok=True)

mpl.rcParams.update({

    # Font
    "font.family": "serif",
    "font.size": 12,

    # Axes
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.linewidth": 1.0,

    # Tick marks
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    # Legend
    "legend.fontsize": 10,
    "legend.frameon": True,

    # Lines
    "lines.linewidth": 2.0,

    # Grid
    "grid.alpha": 0.3,
    "grid.linestyle": "--",

    # Figure export
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# =============================================================================
# USER INPUTS
# =============================================================================

csv_file = "density_velocity_database.csv"

altitude = 25000.0  # [m]

nose_radius = 0.03
emissivity = 0.85
plate_length = 20

sigma = 5.670374419e-8
R = 287.0

# =============================================================================
# ISA TEMPERATURE
# =============================================================================

def isa_temperature(h):
    if h <= 11000.0:
        return 288.15 - 0.0065 * h
    else:
        return 216.65

T_inf = isa_temperature(altitude)

# =============================================================================
# AIR MODELS
# =============================================================================

def cp_air(T):
    return 1000 + 0.1 * (T - 300)

def gamma_air(T):
    return max(1.4 - 0.00005 * (T - 300), 1.28)

def pr_air(T):
    return 0.72

def viscosity_sutherland(T):
    return 1.458e-6 * T**1.5 / (T + 110.4)

# =============================================================================
# NORMAL SHOCK
# =============================================================================

def normal_shock(M1, T1, P1, rho1):

    gamma1 = gamma_air(T1)

    P2_P1 = 1 + (2 * gamma1 / (gamma1 + 1)) * (M1**2 - 1)

    rho2_rho1 = (
        ((gamma1 + 1) * M1**2)
        /
        ((gamma1 - 1) * M1**2 + 2)
    )

    T2_T1 = P2_P1 / rho2_rho1

    M2 = np.sqrt(
        (
            1 + 0.5 * (gamma1 - 1) * M1**2
        )
        /
        (
            gamma1 * M1**2
            - 0.5 * (gamma1 - 1)
        )
    )

    return M2, T1 * T2_T1, P1 * P2_P1, rho1 * rho2_rho1

# =============================================================================
# THETA-BETA-M RELATION
# =============================================================================

def theta_beta_m_eq(beta, M, gamma, theta):

    lhs = np.tan(theta)

    rhs = (
        2
        * (1 / np.tan(beta))
        * (
            M**2 * np.sin(beta)**2 - 1
        )
        /
        (
            M**2 * (gamma + np.cos(2 * beta)) + 2
        )
    )

    return lhs - rhs

# =============================================================================
# SOLVE BETA FROM THETA
# =============================================================================

def solve_beta(M, theta, gamma):

    mu = np.arcsin(1 / M)

    beta_candidates = np.linspace(mu + 1e-5, np.radians(89.0), 5000)

    residuals = theta_beta_m_eq(
        beta_candidates,
        M,
        gamma,
        theta
    )

    sign_change = np.where(
        np.diff(np.sign(residuals))
    )[0]

    if len(sign_change) == 0:
        return None

    idx = sign_change[0]

    beta1 = beta_candidates[idx]
    beta2 = beta_candidates[idx + 1]

    # Bisection refinement
    for _ in range(60):

        beta_mid = 0.5 * (beta1 + beta2)

        f1 = theta_beta_m_eq(beta1, M, gamma, theta)
        fm = theta_beta_m_eq(beta_mid, M, gamma, theta)

        if f1 * fm < 0:
            beta2 = beta_mid
        else:
            beta1 = beta_mid

    return 0.5 * (beta1 + beta2)

# =============================================================================
# OBLIQUE SHOCK
# =============================================================================

def oblique_shock(M1, theta, T1, P1, rho1):

    gamma1 = gamma_air(T1)

    beta = solve_beta(M1, theta, gamma1)

    if beta is None:
        return None

    Mn1 = M1 * np.sin(beta)

    if Mn1 <= 1:
        return None

    Mn2, T2, P2, rho2 = normal_shock(
        Mn1,
        T1,
        P1,
        rho1
    )

    M2 = Mn2 / np.sin(beta - theta)

    return M2, T2, P2, rho2, beta

# =============================================================================
# STAGNATION HEATING
# =============================================================================

def stagnation_heating(rho_inf, u_inf, radius):
    return (
        1.74e-4
        * np.sqrt(rho_inf / radius)
        * u_inf**3
    )

# =============================================================================
# REFERENCE TEMPERATURE
# =============================================================================

def reference_temperature(T_e, T_w, M_e, gamma):

    return T_e * (
        0.5 * (1 + T_w / T_e)
        + 0.16 * ((gamma - 1) / 2) * M_e**2
    )

# =============================================================================
# DATABASE
# =============================================================================

df = pd.read_csv(csv_file)

screening_q = (
    1.83e-4
    * np.sqrt(df["rho"] / nose_radius)
    * df["v"]**3
)

idx = np.argmax(screening_q)

rho_inf = df["rho"].iloc[idx]
u_inf = df["v"].iloc[idx]

# =============================================================================
# FREESTREAM
# =============================================================================

P_inf = rho_inf * R * T_inf

gamma_inf = gamma_air(T_inf)

a_inf = np.sqrt(gamma_inf * R * T_inf)

mach = u_inf / a_inf

print(f"Freestream Mach = {mach:.2f}")

# =============================================================================
# STAGNATION TEMPERATURE
# =============================================================================

q_stag = stagnation_heating(
    rho_inf,
    u_inf,
    nose_radius
)

T_stag = (
    q_stag
    /
    (emissivity * sigma)
)**0.25

# =============================================================================
# GRID
# =============================================================================

x_vals = np.linspace(
    0.05,
    plate_length,
    600
)

# =============================================================================
# WEDGE ANGLE SWEEP
# =============================================================================

theta_deg_range = np.linspace(2, 20, 10)

theta_rad_range = np.radians(theta_deg_range)

all_profiles = []

for theta_deg, theta in zip(
    theta_deg_range,
    theta_rad_range
):

    result = oblique_shock(
        mach,
        theta,
        T_inf,
        P_inf,
        rho_inf
    )

    if result is None:
        print(f"No attached shock solution for θ={theta_deg:.1f}°")
        continue

    M_e, T_e, P_e, rho_e, beta = result

    gamma_e = gamma_air(T_e)

    u_e = M_e * np.sqrt(
        gamma_e * R * T_e
    )

    print(
        f"θ={theta_deg:.1f}°, "
        f"β={np.degrees(beta):.2f}°, "
        f"M2={M_e:.2f}"
    )

    T_profile = []
    q_profile = []
    Taw_profile = []
    Re_profile = []

    for x in x_vals:

        T_w = 1000.0

        for _ in range(100):

            T_star = reference_temperature(
                T_e,
                T_w,
                M_e,
                gamma_e
            )

            cp = cp_air(T_star)

            pr = pr_air(T_star)

            mu = viscosity_sutherland(T_star)

            Re_x = (
                rho_e
                * u_e
                * x
                / mu
            )

            # Compressible turbulent estimate
            Cf = 0.0592 / (Re_x**0.2)

            St = (
                (Cf / 2)
                * pr**(-2/3)
            )

            r = pr**(1/3)

            T_aw = T_e * (
                1
                + r
                * 0.5
                * (gamma_e - 1)
                * M_e**2
            )

            q_conv = (
                St
                * rho_e
                * u_e
                * cp
                * (T_aw - T_w)
            )

            q_conv = max(q_conv, 1.0)

            T_new = (
                q_conv
                /
                (emissivity * sigma)
            )**0.25

            if abs(T_new - T_w) < 1e-3:
                break

            T_w = 0.7 * T_w + 0.3 * T_new

        stag_blend = np.exp(-x / 0.02)

        T_final = (
            stag_blend * T_stag
            + (1 - stag_blend) * T_w
        )

        T_profile.append(T_final)
        q_profile.append(q_conv)
        Taw_profile.append(T_aw)
        Re_profile.append(Re_x)

    all_profiles.append({
        "theta_deg": theta_deg,
        "beta_deg": np.degrees(beta),
        "M_edge": M_e,
        "T_profile": T_profile,
        "q_profile": q_profile,
        "Taw_profile": Taw_profile,
        "Re_profile": Re_profile,
        "peak_T": np.max(T_profile),
        "peak_q": np.max(q_profile),
        "P_ratio": P_e / P_inf,
        "rho_ratio": rho_e / rho_inf
    })

# =============================================================================
# PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(10.0, 4.5))

for data in all_profiles:

    ax.plot(
        x_vals,
        data["T_profile"],
        label=fr'$\theta={data["theta_deg"]:.0f}^\circ$'
    )

ax.set_xlabel("Distance Along Plate [m]")

ax.set_ylabel("Wall Temperature [K]")

ax.set_xlim(0, plate_length)

ax.grid(True)

ax.legend(
    ncol=2,
    loc="best"
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()

fig.savefig(
    fig_dir / "wall_temperature.pdf"
)

# =============================================================================
# HEAT FLUX
# =============================================================================

fig, ax = plt.subplots(figsize=(6.0, 4.5))

for data in all_profiles:

    ax.plot(
        x_vals,
        data["q_profile"],
        label=fr'$\theta={data["theta_deg"]:.0f}^\circ$'
    )

ax.set_xlabel("Distance Along Plate [m]")

ax.set_ylabel(
    r"Convective Heat Flux [W/m$^2$]"
)

ax.set_xlim(0, plate_length)

ax.set_yscale("log")

ax.grid(True, which='both')

ax.legend(
    ncol=2,
    loc="best"
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()

fig.savefig(
    fig_dir / "heat_flux.pdf"
)

# =============================================================================
# ADIABATIC WALL TEMPERATURE
# =============================================================================

fig, ax = plt.subplots(figsize=(6.0, 4.5))

for data in all_profiles:

    ax.plot(
        x_vals,
        data["Taw_profile"],
        label=fr'$\theta={data["theta_deg"]:.0f}^\circ$'
    )

ax.set_xlabel("Distance Along Plate [m]")

ax.set_ylabel(
    "Adiabatic Wall Temperature [K]"
)

ax.set_xlim(0, plate_length)

ax.grid(True)

ax.legend(
    ncol=2,
    loc="best"
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()

fig.savefig(
    fig_dir / "adiabatic_temperature.pdf"
)

# =============================================================================
# REYNOLDS NUMBER
# =============================================================================

fig, ax = plt.subplots(figsize=(6.0, 4.5))

for data in all_profiles:

    ax.plot(
        x_vals,
        data["Re_profile"],
        label=fr'$\theta={data["theta_deg"]:.0f}^\circ$'
    )

ax.set_xlabel("Distance Along Plate [m]")

ax.set_ylabel(
    r"Local Reynolds Number $Re_x$"
)

ax.set_xlim(0, plate_length)

ax.set_yscale("log")

ax.grid(True, which='both')

ax.legend(
    ncol=2,
    loc="best"
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()

fig.savefig(
    fig_dir / "reynolds.pdf"
)

# =============================================================================

plt.show()
