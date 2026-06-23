"""
TPS Through-Thickness Thermal Solver
=====================================
Crank–Nicolson implicit finite-difference scheme.
  - Unconditionally stable → large dt allowed, no Fourier-number restriction
  - 2nd-order accurate in both space and time
  - Tri-diagonal system solved via scipy.linalg.solve_banded  O(N)

Outer BC  (x = 0):  q_aero(t) – ε σ (T⁴ – T_amb⁴)  = ρ cp dx/dt · ΔT
                    Radiation linearised each step as h_rad·(T – T_amb)
Inner BC  (x = L):  adiabatic  –or–  convective to cabin air

Outputs
-------
  • Time histories of outer / inner surface temperatures
  • Final through-thickness temperature profile
  • Layer-by-layer peak temperature vs material limit
  • Heat soak into the inner structural / fuel layer
  • Through-thickness snapshots at user-defined times

Bugs fixed vs original
----------------------
  1. tps_materials.py was absent; now provided as a companion file.
  2. shade_layers() was a no-op (loop body was `pass`) and unused; removed.
  3. q_inner_series list-comprehension multiplied by 0.0 (leftover debug
     artifact) and was never used downstream; removed entirely.
  4. last_name / last_name_key were redundant aliases for the same string;
     unified to last_name throughout.
  5. Guard against empty snapshots list before the final-frame append.
  6. layer_T_hist lists are now always exactly n_steps long (append moved
     to be unconditional per step), removing the need for the defensive
     time_hist[:len(hist)] slice.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

from tps_materials import MATERIALS

# =============================================================================
# USER INPUTS
# =============================================================================

layer_stack = [
    ("CVI_C_SiC",    0.003),# hot-face CMC tile
    ("IMI_Pt_Saffil48",     0.018),  # aerogel insulation blanket  # aerogel insulation blanket
    ("Ti_6Al_4V",    0.003),   # structural pressure shell (inner wall)
]

panel_area          = 100.0    # m²  (for mass & total-energy output)
initial_temperature = 310.0    # K
ambient_temperature = 280   # K

n_nodes = 300                  # spatial nodes
dt      = 0.3                  # time step [s]  — CN is unconditionally stable

sigma = 5.670374419e-8         # W/(m²·K⁴)

inner_bc        = "convective"  # "adiabatic" or "convective"
h_inner         = 20.0          # W/(m²·K)  (only used if inner_bc=="convective")
T_inner_amb     = 295.0        # K

fuel_flash_limit = 400.0       # K

# =============================================================================
# HEAT FLUX PROFILE
# =============================================================================

peak_heat_flux  = 79500    # W/m²
ramp_up_time    = 1.0         # s
steady_time     = 3600.0     # s
ramp_down_time  = 1        # s
post_cool_time  = 1       # s

simulation_time = ramp_up_time + steady_time + ramp_down_time + post_cool_time

snapshot_times = [0.0, 250.0, 500.0, 1000.0, 2000.0,
                  ramp_up_time + steady_time,
                  ramp_up_time + steady_time + ramp_down_time,
                  simulation_time]


def heat_flux_profile(t):
    if t <= ramp_up_time:
        return peak_heat_flux * t / ramp_up_time
    elif t <= ramp_up_time + steady_time:
        return peak_heat_flux
    elif t <= ramp_up_time + steady_time + ramp_down_time:
        return peak_heat_flux * (1.0
               - (t - ramp_up_time - steady_time) / ramp_down_time)
    else:
        return 0.0


# =============================================================================
# MESH
# =============================================================================

total_thickness = sum(th for _, th in layer_stack)
dx = total_thickness / n_nodes
x  = np.linspace(0.0, total_thickness, n_nodes)

rho = np.zeros(n_nodes)
cp  = np.zeros(n_nodes)
k   = np.zeros(n_nodes)
eps = np.zeros(n_nodes)

layer_ranges = {}
cx = 0.0
for name, th in layer_stack:
    mat   = MATERIALS[name]
    start = round(cx / total_thickness * n_nodes)
    cx   += th
    end   = min(round(cx / total_thickness * n_nodes), n_nodes)
    layer_ranges[name] = (start, end)
    rho[start:end] = mat["density"]
    cp [start:end] = mat["specific_heat"]
    k  [start:end] = mat["thermal_conductivity"]
    eps[start:end] = mat["emissivity"] if mat["emissivity"] is not None else 0.8

# harmonic-mean interface conductivities
k_int = 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])

alpha_max = np.max(k / (rho * cp))
Fo = alpha_max * dt / dx**2

print("=" * 62)
print("  TPS SOLVER  –  Crank–Nicolson (θ=0.5) implicit scheme")
print("=" * 62)
print(f"  Nodes      : {n_nodes}")
print(f"  dx         : {dx*1e3:.3f} mm")
print(f"  dt         : {dt:.3f} s")
print(f"  Max Fo     : {Fo:.1f}   (explicit limit is 0.5 — no limit here)")
print(f"  Sim time   : {simulation_time:.0f} s  ({simulation_time/60:.1f} min)")

# =============================================================================
# MASS
# =============================================================================

print("\n  Mass breakdown:")
total_mass = 0.0
for name, th in layer_stack:
    m = MATERIALS[name]["density"] * th * panel_area
    total_mass += m
    print(f"    {name:20s}  {m:8.1f} kg")
print(f"    {'TOTAL':20s}  {total_mass:8.1f} kg")

# =============================================================================
# CRANK–NICOLSON SYSTEM BUILDER
# =============================================================================
# For each node i, the discretised equation is:
#
#   [ρ cp dx / dt] (T_new - T_old)
#       = θ·(k_L (T_new[i-1]-T_new[i]) - k_R (T_new[i]-T_new[i+1])) / dx
#       + (1-θ)·(k_L (T_old[i-1]-T_old[i]) - k_R (T_old[i]-T_old[i+1])) / dx
#
# Rearranged to  A T_new = RHS  where A is tridiagonal.
# Outer node: replaces the phantom node with the surface energy balance.
# Inner node: adiabatic → zero-gradient ghost cell.
# =============================================================================

THETA = 0.5   # 0.5 = Crank–Nicolson; 1.0 = backward-Euler (also stable)


def build_system(T_old, q_aero):
    N  = n_nodes
    ab = np.zeros((3, N))  # solve_banded (1,1) format: [super, main, sub]
    b  = np.zeros(N)

    C = rho * cp * dx / dt   # thermal mass / time-step [W/m² K]

    # ---- interior nodes 1 … N-2 -------------------------------------------
    kL = k_int[:-1]          # k at i-½  (length N-2, indexed from i=1)
    kR = k_int[1:]           # k at i+½

    diag_i  = C[1:-1] + THETA/dx * (kL + kR)
    sup_i   = -THETA/dx * kR
    sub_i   = -THETA/dx * kL

    rhs_i   = (C[1:-1] - (1-THETA)/dx * (kL+kR)) * T_old[1:-1] \
             + (1-THETA)/dx * kL * T_old[:-2] \
             + (1-THETA)/dx * kR * T_old[2:]

    ab[1, 1:-1] = diag_i
    ab[0, 2:  ] = sup_i           # super diagonal stored at column j, row j-1
    ab[2, :-2 ] = sub_i           # sub   diagonal stored at column j, row j+1
    b[1:-1]     = rhs_i

    # ---- outer node i=0 ---------------------------------------------------
    # Energy balance per unit area per time-step:
    #   C[0]·dT  =  q_net * dt
    # q_net = q_aero - q_rad - q_cond
    # Radiation: linearise q_rad ≈ h_r·(T[0] - T_amb) where
    #   h_r = ε σ (T[0]²+T_amb²)(T[0]+T_amb)  evaluated at T_old[0]
    h_r  = eps[0] * sigma * (T_old[0]**2 + ambient_temperature**2) \
                          * (T_old[0]   + ambient_temperature)
    kR0  = k_int[0] / dx

    # Treat q_aero as fully explicit (it's prescribed, not unknown).
    # Radiation and conduction split θ / (1-θ) between new and old.
    ab[1, 0] = C[0] + THETA*(h_r + kR0)
    ab[0, 1] = -THETA*kR0
    b[0]     = (C[0] - (1-THETA)*(h_r + kR0)) * T_old[0] \
               + (1-THETA)*kR0 * T_old[1] \
               + q_aero \
               + h_r * ambient_temperature   # from linearised rad: h_r·T_amb

    # ---- inner node i=N-1 --------------------------------------------------
    kLN = k_int[-1] / dx

    if inner_bc == "adiabatic":
        # Zero flux: ghost cell T[N] = T[N-1]  → no net conduction
        ab[1, -1] = C[-1] + THETA*kLN   # conduction from left only (ghost = self)
        ab[2, -2] = -THETA*kLN
        b[-1]     = (C[-1] - (1-THETA)*kLN) * T_old[-1] \
                    + (1-THETA)*kLN * T_old[-2]
    else:
        h_eff = h_inner
        ab[1, -1] = C[-1] + THETA*(kLN + h_eff)
        ab[2, -2] = -THETA*kLN
        b[-1]     = (C[-1] - (1-THETA)*(kLN + h_eff)) * T_old[-1] \
                    + (1-THETA)*kLN * T_old[-2] \
                    + h_eff * T_inner_amb

    return ab, b


# =============================================================================
# STORAGE
# =============================================================================

n_steps      = int(simulation_time / dt) + 1
time_hist    = np.zeros(n_steps)
outer_T_hist = np.zeros(n_steps)
inner_T_hist = np.zeros(n_steps)
flux_hist    = np.zeros(n_steps)
qrad_hist    = np.zeros(n_steps)

layer_max_T  = {n: initial_temperature  for n, _ in layer_stack}
layer_T_hist = {n: np.zeros(n_steps)    for n, _ in layer_stack}   # FIX: pre-allocate

snapshots: list[tuple[float, np.ndarray]] = []
snap_set  = set(snapshot_times)
snap_done = set()

energy_in       = 0.0
energy_rad      = 0.0
heat_soak_inner = 0.0

T = np.full(n_nodes, initial_temperature)

# =============================================================================
# MAIN LOOP
# =============================================================================

for step in range(n_steps):
    t = step * dt
    time_hist[step] = t

    q_aero = heat_flux_profile(t)
    flux_hist[step] = q_aero

    # radiation from old T for energy accounting
    q_rad_old = eps[0] * sigma * (T[0]**4 - ambient_temperature**4)

    ab, b = build_system(T, q_aero)
    T = solve_banded((1, 1), ab, b)

    q_rad_new = eps[0] * sigma * (T[0]**4 - ambient_temperature**4)
    qrad_hist[step] = q_rad_new

    energy_in  += q_aero * dt
    energy_rad += 0.5*(q_rad_old + q_rad_new) * dt

    # heat soak into inner face via conduction at last interface
    q_soak = k_int[-1] * (T[-2] - T[-1]) / dx   # +ve = heat into last node
    heat_soak_inner += max(float(q_soak), 0.0) * dt

    outer_T_hist[step] = T[0]
    inner_T_hist[step] = T[-1]

    # FIX: write directly into pre-allocated arrays (no list.append needed)
    for name, (s, e) in layer_ranges.items():
        Tmax = float(np.max(T[s:e]))
        if Tmax > layer_max_T[name]:
            layer_max_T[name] = Tmax
        layer_T_hist[name][step] = float(np.mean(T[s:e]))

    for st in list(snap_set):
        if st not in snap_done and t >= st:
            snapshots.append((t, T.copy()))
            snap_done.add(st)

# FIX: guard against empty snapshots list before checking last element
if not snapshots or snapshots[-1][0] < time_hist[-1]:
    snapshots.append((time_hist[-1], T.copy()))

# =============================================================================
# ENERGY BALANCE
# =============================================================================

stored_energy = float(np.sum(rho * cp * (T - initial_temperature) * dx))
balance       = energy_in - energy_rad - stored_energy

print("\n" + "=" * 62)
print("  ENERGY BALANCE  (per m² of panel)")
print("=" * 62)
print(f"  Incident energy   : {energy_in/1e6:8.3f} MJ/m²")
print(f"  Radiated energy   : {energy_rad/1e6:8.3f} MJ/m²")
print(f"  Stored in TPS     : {stored_energy/1e6:8.3f} MJ/m²")
print(f"  Balance residual  : {balance/1e6:+8.4f} MJ/m²  (<1% is excellent)")
print(f"  Heat soak (inner) : {heat_soak_inner/1e3:8.2f} kJ/m²"
      f"  ({heat_soak_inner*panel_area/1e6:.2f} MJ total panel)")

# =============================================================================
# PEAK TEMPERATURES
# =============================================================================

last_name = list(layer_ranges)[-1]   # FIX: single consistent alias

print("\n" + "=" * 62)
print("  MAXIMUM LAYER TEMPERATURES")
print("=" * 62)
for name, Tmax in layer_max_T.items():
    limit = MATERIALS[name].get("max_service_temp")
    margin = (limit - Tmax) if limit else None
    flag = ""
    if limit and Tmax > limit:
        flag = "  *** EXCEEDS LIMIT ***"
    elif limit and (limit - Tmax) < 50:
        flag = "  ⚠ close to limit"
    lim_str = f"{limit} K" if limit else "N/A"
    mar_str = f"margin {margin:+.0f} K" if margin is not None else ""
    print(f"  {name:20s}  {Tmax:7.1f} K   limit {lim_str:8s}  {mar_str}{flag}")

T_inner_max = layer_max_T[last_name]

print(f"\n  Inner surface peak  : {inner_T_hist.max():.1f} K")
if T_inner_max > fuel_flash_limit:
    print(f"  *** Fuel flash limit ({fuel_flash_limit:.0f} K) EXCEEDED ***")
else:
    print(f"  Fuel/structure OK   (flash limit {fuel_flash_limit:.0f} K)")

# =============================================================================
# PLOTS
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

ACCENT  = "#D94F2B"
COLD    = "#2478C8"
SHADES  = plt.cm.inferno(np.linspace(0.1, 0.9, len(snapshots)))

cmap_layers = plt.cm.tab10

# ── Figure 1: Mission overview ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
(ax1, ax2), (ax3, ax4) = axes

# (a) Heat flux
ax1.fill_between(time_hist, flux_hist/1e3, alpha=0.18, color=ACCENT)
ax1.plot(time_hist, flux_hist/1e3, color=ACCENT, lw=2)
ax1.set(xlabel="Time [s]", ylabel="Heat Flux [kW/m²]",
        title="Applied Aerodynamic Heat Flux")
ax1.grid(alpha=0.25)

# (b) Surface temperatures
ax2.plot(time_hist, outer_T_hist, color=ACCENT, lw=2, label="Outer surface")
ax2.plot(time_hist, inner_T_hist, color=COLD,   lw=2, label="Inner surface")
last_limit = MATERIALS[last_name].get("max_service_temp")
if last_limit:
    ax2.axhline(last_limit, color=COLD, ls="--", lw=1.2, alpha=0.7,
                label=f"{last_name} limit {last_limit} K")
ax2.axhline(fuel_flash_limit, color="seagreen", ls=":", lw=1.4,
            label=f"Flash limit {fuel_flash_limit:.0f} K")
ax2.set(xlabel="Time [s]", ylabel="Temperature [K]",
        title="Surface Temperatures")
ax2.legend(fontsize=8.5, framealpha=0.5)
ax2.grid(alpha=0.25)

# (c) Layer-average temperatures
for i, (name, _) in enumerate(layer_stack):
    ax3.plot(time_hist, layer_T_hist[name],
             lw=1.8, color=cmap_layers(i), label=name)
ax3.set(xlabel="Time [s]", ylabel="Mean Layer Temperature [K]",
        title="Layer-Average Temperatures")
ax3.legend(fontsize=8.5, framealpha=0.5)
ax3.grid(alpha=0.25)

# (d) Cumulative heat soak — derived from enthalpy gain of the last layer
s_last, e_last = layer_ranges[last_name]
mass_per_area   = MATERIALS[last_name]["density"] * (e_last - s_last) * dx
cp_last         = MATERIALS[last_name]["specific_heat"]
dT_dt           = np.gradient(layer_T_hist[last_name], dt)
q_soak_inst     = np.maximum(mass_per_area * cp_last * dT_dt, 0.0)   # W/m²
cum_soak        = np.cumsum(q_soak_inst) * dt / 1e3                  # kJ/m²

ax4.fill_between(time_hist, cum_soak, alpha=0.18, color="#7B5EA7")
ax4.plot(time_hist, cum_soak, color="#7B5EA7", lw=2)
ax4.set(xlabel="Time [s]", ylabel="Cumulative Heat Soak [kJ/m²]",
        title=f"Heat Soak into {last_name} Layer")
ax4.grid(alpha=0.25)

plt.suptitle(
    f"TPS Thermal Analysis  ·  Crank–Nicolson Implicit Solver\n"
    f"Peak heat flux {peak_heat_flux/1e3:.0f} kW/m²  "
    f"·  {simulation_time:.0f} s mission",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()

plt.show()

# ── Figure 2: Through-thickness snapshots ──────────────────────────────────
fig2, ax = plt.subplots(figsize=(11, 6))
x_mm = x * 1e3

# Layer background bands
cx_mm = 0.0
for i, (name, th) in enumerate(layer_stack):
    end_mm = cx_mm + th * 1e3
    ax.axvspan(cx_mm, end_mm, alpha=0.10,
               color=cmap_layers(i), label=name)
    ax.text((cx_mm + end_mm) / 2,
            initial_temperature + 8,
            name.replace("_", "\n"), ha="center", va="bottom",
            fontsize=7.5, color=cmap_layers(i), fontweight="bold",
            linespacing=1.2)
    cx_mm = end_mm

for (t_snap, T_snap), col in zip(snapshots, SHADES):
    ax.plot(x_mm, T_snap, color=col, lw=1.8, label=f"t = {t_snap:.0f} s")

sm = plt.cm.ScalarMappable(
    cmap="inferno",
    norm=plt.Normalize(vmin=snapshots[0][0], vmax=snapshots[-1][0]))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label("Time [s]", fontsize=10)

ax.set(xlabel="Distance from outer surface [mm]",
       ylabel="Temperature [K]",
       title="Through-Thickness Temperature Profiles at Selected Times")
ax.grid(alpha=0.25)
plt.tight_layout()

plt.show()
