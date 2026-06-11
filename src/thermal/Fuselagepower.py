"""
TPS Through-Thickness Thermal Solver  —  Cabin AC Power Edition
================================================================
Crank–Nicolson implicit finite-difference scheme (θ = 0.5).
  - Unconditionally stable; large dt allowed
  - 2nd-order accurate in both space and time
  - Tri-diagonal system solved via scipy.linalg.solve_banded  O(N)

Outer BC  (x = 0):  q_aero(t) – ε σ (T⁴ – T_amb⁴)
Inner BC  (x = L):  convective to cabin air  →  q = h_inner · (T_wall – T_cabin)
                    The cabin air is held at T_cabin_setpoint by the AC.
                    AC power  =  max(0,  q_wall→air)  per m²  [W/m²]

New outputs vs base solver
--------------------------
  • Inner wall (Ti shell) temperature history
  • Cabin-air convective heat flux history  q_cabin(t)  [W/m²]
  • Instantaneous and cumulative AC cooling power  [W / kJ per m²]
  • Total AC energy for the full fuselage panel area  [kWh]

Geometry accuracy note
----------------------
This is a 1-D Cartesian (flat-plate) solver.  For a cylindrical fuselage
the governing equation in the radial direction is:

    ρ cp ∂T/∂t  =  (1/r) ∂/∂r [r k ∂T/∂r]

which differs from the Cartesian form by the (1/r) metric factor.
Three consequences are quantified by the geometric correction below:

  1. Curvature divergence  —  for the same wall thickness the radial area
     grows with radius, so the outer surface sees more aerodynamic heat per
     unit volume than a flat plate implies.  Correction factor:
         f_curv  =  r_outer / r_inner  =  (R + L) / R
     where R = fuselage radius, L = TPS wall thickness.
     For R = 2 m, L = 0.016 m:  f_curv ≈ 1.008  (< 1 % error — negligible).

  2. Aerodynamic heating distribution  —  on a flat plate q is uniform.
     On a cylinder the stagnation-line heat flux is ~√2 × higher than the
     leeward side; the circumferential average is roughly 0.6–0.7 × peak.
     This code uses the peak (conservative); set `q_circumferential_factor`
     to adjust.

  3. Inner surface area per unit outer area  —  again f_curv corrects this;
     the AC power output of this solver should be multiplied by
         (r_inner / r_outer)  ≈  0.992
     for R = 2 m.  This is included in the corrected AC energy printout.

Bottom line: for fuselage radii ≥ 1 m and typical TPS thicknesses (< 30 mm),
the flat-plate error on heat flux and AC power is < 2 %.  The dominant
uncertainty is the aerodynamic heating distribution (factor ~1.4–1.7 between
stagnation and leeward), not the wall-curvature term.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

from tps_materials import MATERIALS

# =============================================================================
# USER INPUTS
# =============================================================================

layer_stack = [
    ("CVI_C_SiC",    0.003),
    ("AETB_20",     0.002), # hot-face CMC tile
    ("Pyrogel_XT_E", 0.010),   # aerogel insulation blanket
    ("Ti_6Al_4V",    0.003),   # structural pressure shell (inner wall)
]

panel_area          = 100.0    # m²  fuselage panel area (outer surface)
initial_temperature = 295.0    # K
ambient_temperature = 295.0    # K   free-stream / ambient

n_nodes = 300                  # spatial nodes
dt      = 0.3                  # time step [s]

sigma = 5.670374419e-8         # W/(m²·K⁴)

# ── Cabin air conditioning ─────────────────────────────────────────────────
T_cabin_setpoint  = 294.15     # K  (21 °C)
h_inner           = 20.0       # W/(m²·K)  forced-convection coeff, cabin side
#   h ≈ 20 W/m²K is representative of light cabin forced-air circulation
#   (ASHRAE handbook: natural conv ~3–8; ducted cabin air ~15–30 W/m²K)
inner_bc          = "convective"   # must be "convective" for AC model

# ── Fuselage geometry for curvature correction ─────────────────────────────
fuselage_radius   = 2.0        # m  (radius to outer TPS surface)
q_circumferential_factor = 0.65
#   Circumferential average / peak ratio for a cylinder in hypersonic flow.
#   0.65 is a reasonable mid-fidelity estimate; 1.0 = conservative peak.

# ── Heat flux profile ──────────────────────────────────────────────────────
peak_heat_flux  = 25_000.0     # W/m²  peak aero heating at outer CMC surface
ramp_up_time    =     1.0      # s
steady_time     =  3600.0      # s   (60 min cruise)
ramp_down_time  =     1.0      # s
post_cool_time  =     1.0      # s

simulation_time = ramp_up_time + steady_time + ramp_down_time + post_cool_time

snapshot_times = [0.0, 250.0, 500.0, 1000.0, 2000.0,
                  ramp_up_time + steady_time,
                  ramp_up_time + steady_time + ramp_down_time,
                  simulation_time]

fuel_flash_limit = 400.0       # K  (kept for compatibility)


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

print("=" * 66)
print("  TPS SOLVER  –  Crank–Nicolson (θ=0.5)  +  Cabin AC Power")
print("=" * 66)
print(f"  Nodes      : {n_nodes}")
print(f"  dx         : {dx*1e3:.3f} mm")
print(f"  dt         : {dt:.3f} s")
print(f"  Max Fo     : {Fo:.1f}   (explicit limit is 0.5 — no limit here)")
print(f"  Sim time   : {simulation_time:.0f} s  ({simulation_time/60:.1f} min)")
print(f"  Cabin T    : {T_cabin_setpoint - 273.15:.1f} °C  ({T_cabin_setpoint:.2f} K)")
print(f"  h_inner    : {h_inner:.1f} W/(m²·K)")

print("\n  Mass breakdown:")
total_mass = 0.0
for name, th in layer_stack:
    m = MATERIALS[name]["density"] * th * panel_area
    total_mass += m
    print(f"    {name:20s}  {m:8.1f} kg")
print(f"    {'TOTAL':20s}  {total_mass:8.1f} kg")

# =============================================================================
# CURVATURE CORRECTION FACTORS
# =============================================================================

r_outer = fuselage_radius
r_inner = fuselage_radius - total_thickness
f_curv  = r_outer / r_inner   # > 1; outer area > inner area

print(f"\n  Fuselage radius    : {fuselage_radius:.2f} m")
print(f"  Curvature factor   : {f_curv:.4f}  (outer/inner area ratio)")
print(f"  Circumf. avg factor: {q_circumferential_factor:.2f}")
print(f"  → AC power corrected for both factors in final energy output.")

# =============================================================================
# CRANK–NICOLSON SYSTEM BUILDER
# =============================================================================
# Inner BC: convective to cabin air at fixed T_cabin_setpoint.
#
#   q_conv = h_inner * (T[-1] - T_cabin_setpoint)
#
# This is incorporated into the last-node equation exactly as in the base
# solver — T_cabin_setpoint acts as the "ambient" on the inner side.
# The AC must remove this flux to maintain T_cabin.  We record it each step.
# =============================================================================

THETA = 0.5


def build_system(T_old, q_aero):
    N  = n_nodes
    ab = np.zeros((3, N))
    b  = np.zeros(N)

    C = rho * cp * dx / dt

    # ── interior nodes ──────────────────────────────────────────────────────
    kL = k_int[:-1]
    kR = k_int[1:]

    diag_i = C[1:-1] + THETA/dx * (kL + kR)
    sup_i  = -THETA/dx * kR
    sub_i  = -THETA/dx * kL
    rhs_i  = (C[1:-1] - (1-THETA)/dx*(kL+kR)) * T_old[1:-1] \
            + (1-THETA)/dx * kL * T_old[:-2] \
            + (1-THETA)/dx * kR * T_old[2:]

    ab[1, 1:-1] = diag_i
    ab[0, 2:  ] = sup_i
    ab[2, :-2 ] = sub_i
    b[1:-1]     = rhs_i

    # ── outer node i = 0  (aero + radiation) ───────────────────────────────
    h_r  = eps[0] * sigma * (T_old[0]**2 + ambient_temperature**2) \
                          * (T_old[0]   + ambient_temperature)
    kR0  = k_int[0] / dx

    ab[1, 0] = C[0] + THETA*(h_r + kR0)
    ab[0, 1] = -THETA*kR0
    b[0]     = (C[0] - (1-THETA)*(h_r + kR0)) * T_old[0] \
               + (1-THETA)*kR0 * T_old[1] \
               + q_aero \
               + h_r * ambient_temperature

    # ── inner node i = N-1  (convection to cabin air) ───────────────────────
    # q_conv (positive = heat leaving the wall into the cabin) is:
    #   q_conv = h_inner * (T[-1] - T_cabin_setpoint)
    #
    # In the FD equation this appears as a loss term on the last node,
    # exactly analogous to the radiation linearisation on the outer node.
    # No linearisation needed here because the BC is already linear in T.
    kLN   = k_int[-1] / dx
    h_cab = h_inner

    ab[1, -1] = C[-1] + THETA*(kLN + h_cab)
    ab[2, -2] = -THETA*kLN
    b[-1]     = (C[-1] - (1-THETA)*(kLN + h_cab)) * T_old[-1] \
                + (1-THETA)*kLN * T_old[-2] \
                + h_cab * T_cabin_setpoint   # source term: h·T_cabin

    return ab, b


# =============================================================================
# STORAGE
# =============================================================================

n_steps      = int(simulation_time / dt) + 1
time_hist    = np.zeros(n_steps)
outer_T_hist = np.zeros(n_steps)
inner_T_hist = np.zeros(n_steps)   # Ti inner wall temperature
flux_hist    = np.zeros(n_steps)
qrad_hist    = np.zeros(n_steps)
q_cabin_hist = np.zeros(n_steps)   # heat flux into cabin air  [W/m²]

layer_max_T  = {n: initial_temperature for n, _ in layer_stack}
layer_T_hist = {n: np.zeros(n_steps)   for n, _ in layer_stack}

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

    q_rad_old = eps[0] * sigma * (T[0]**4 - ambient_temperature**4)

    ab, b = build_system(T, q_aero)
    T     = solve_banded((1, 1), ab, b)

    q_rad_new = eps[0] * sigma * (T[0]**4 - ambient_temperature**4)
    qrad_hist[step] = q_rad_new

    energy_in  += q_aero * dt
    energy_rad += 0.5*(q_rad_old + q_rad_new) * dt

    q_soak = k_int[-1] * (T[-2] - T[-1]) / dx
    heat_soak_inner += max(float(q_soak), 0.0) * dt

    outer_T_hist[step] = T[0]
    inner_T_hist[step] = T[-1]   # Ti inner wall

    # ── Convective flux from inner wall to cabin air  [W/m²] ────────────────
    # Positive = heat flowing from wall into cabin (AC must remove this).
    q_cab = h_inner * (T[-1] - T_cabin_setpoint)
    q_cabin_hist[step] = max(float(q_cab), 0.0)

    for name, (s, e) in layer_ranges.items():
        Tmax = float(np.max(T[s:e]))
        if Tmax > layer_max_T[name]:
            layer_max_T[name] = Tmax
        layer_T_hist[name][step] = float(np.mean(T[s:e]))

    for st in list(snap_set):
        if st not in snap_done and t >= st:
            snapshots.append((t, T.copy()))
            snap_done.add(st)

if not snapshots or snapshots[-1][0] < time_hist[-1]:
    snapshots.append((time_hist[-1], T.copy()))

# =============================================================================
# ENERGY BALANCE
# =============================================================================

stored_energy = float(np.sum(rho * cp * (T - initial_temperature) * dx))
balance       = energy_in - energy_rad - stored_energy

print("\n" + "=" * 66)
print("  ENERGY BALANCE  (per m² outer surface, Cartesian)")
print("=" * 66)
print(f"  Incident energy   : {energy_in/1e6:8.3f} MJ/m²")
print(f"  Radiated energy   : {energy_rad/1e6:8.3f} MJ/m²")
print(f"  Stored in TPS     : {stored_energy/1e6:8.3f} MJ/m²")
print(f"  Balance residual  : {balance/1e6:+8.4f} MJ/m²")
print(f"  Heat soak (inner) : {heat_soak_inner/1e3:8.2f} kJ/m²")

# =============================================================================
# AC POWER
# =============================================================================

# Cumulative AC energy per m² of inner (cabin) wall surface [J/m²]
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # NumPy 2.0 / 1.x
ac_energy_per_m2 = _trapz(q_cabin_hist, time_hist)              # J/m²  (flat plate)
peak_ac_flux     = q_cabin_hist.max()                        # W/m²

# Corrected for cylindrical geometry:
#   - Inner surface area per unit outer surface = r_inner / r_outer = 1/f_curv
#   - Circumferential average heating = q_circ_factor × peak
ac_energy_corrected  = ac_energy_per_m2 / f_curv * q_circumferential_factor
ac_energy_total_kWh  = ac_energy_corrected * panel_area / 3.6e6   # kWh
peak_ac_total_kW     = peak_ac_flux / f_curv * q_circumferential_factor \
                       * panel_area / 1e3                          # kW

print("\n" + "=" * 66)
print("  AC COOLING POWER")
print("=" * 66)
print(f"  Peak wall→cabin flux (flat plate) : {peak_ac_flux:8.1f} W/m²")
print(f"  AC energy  (flat plate, per m²)   : {ac_energy_per_m2/1e3:8.2f} kJ/m²")
print(f"  Geometry corrections applied:")
print(f"    Curvature factor (outer/inner)  : {f_curv:.4f}")
print(f"    Circumf. average factor         : {q_circumferential_factor:.2f}")
print(f"  AC energy  (corrected, per m²)    :"
      f" {ac_energy_corrected/1e3:8.2f} kJ/m²")
print(f"  AC energy  (full {panel_area:.0f} m² panel)     :"
      f" {ac_energy_total_kWh:8.3f} kWh")
print(f"  Peak AC demand (full panel)       : {peak_ac_total_kW:8.2f} kW")
print(f"\n  Note: AC demand is the heat that must be removed from the cabin")
print(f"  to maintain {T_cabin_setpoint - 273.15:.1f} °C.  Actual refrigeration power is")
print(f"  higher by 1/COP (typical aircraft vapour-cycle COP ≈ 2–3,")
print(f"  so multiply AC energy by ~0.33–0.5 for compressor shaft power).")

# =============================================================================
# PEAK TEMPERATURES
# =============================================================================

last_name = list(layer_ranges)[-1]

print("\n" + "=" * 66)
print("  MAXIMUM LAYER TEMPERATURES")
print("=" * 66)
for name, Tmax in layer_max_T.items():
    limit  = MATERIALS[name].get("max_service_temp")
    margin = (limit - Tmax) if limit else None
    flag   = ""
    if limit and Tmax > limit:
        flag = "  *** EXCEEDS LIMIT ***"
    elif limit and (limit - Tmax) < 50:
        flag = "  ⚠ close to limit"
    lim_str = f"{limit} K" if limit else "N/A"
    mar_str = f"margin {margin:+.0f} K" if margin is not None else ""
    print(f"  {name:20s}  {Tmax:7.1f} K   limit {lim_str:8s}  {mar_str}{flag}")

T_inner_wall_peak = inner_T_hist.max()
T_inner_wall_ss   = inner_T_hist[-1]

print(f"\n  Inner wall (Ti) peak    : {T_inner_wall_peak:.1f} K"
      f"  ({T_inner_wall_peak - 273.15:.1f} °C)")
print(f"  Inner wall (Ti) at t_end: {T_inner_wall_ss:.1f} K"
      f"  ({T_inner_wall_ss - 273.15:.1f} °C)")
print(f"  Cabin air setpoint      : {T_cabin_setpoint:.1f} K"
      f"  ({T_cabin_setpoint - 273.15:.1f} °C)")
if T_inner_wall_peak > fuel_flash_limit:
    print(f"  *** Inner wall exceeds flash limit ({fuel_flash_limit:.0f} K) ***")
else:
    print(f"  Inner wall below flash limit ({fuel_flash_limit:.0f} K)  ✓")

# =============================================================================
# PLOTS
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        130,
})

ACCENT   = "#D94F2B"
COLD     = "#2478C8"
AC_COL   = "#2AA876"
TI_COL   = "#E07B00"
SHADES   = plt.cm.inferno(np.linspace(0.1, 0.9, len(snapshots)))
cmap_lay = plt.cm.tab10

# Cumulative AC energy curve  [kJ/m²]  (flat plate)
cum_ac_kJ = np.cumsum(q_cabin_hist) * dt / 1e3

# ── Figure 1: Mission overview (2 × 3) ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
(ax1, ax2, ax3), (ax4, ax5, ax6) = axes

# (a) Heat flux
ax1.fill_between(time_hist, flux_hist/1e3, alpha=0.18, color=ACCENT)
ax1.plot(time_hist, flux_hist/1e3, color=ACCENT, lw=2)
ax1.set(xlabel="Time [s]", ylabel="Heat Flux [kW/m²]",
        title="Applied Aerodynamic Heat Flux")
ax1.grid(alpha=0.25)

# (b) Outer surface temperature
ax2.plot(time_hist, outer_T_hist,  color=ACCENT, lw=2, label="Outer (CMC)")
ax2.plot(time_hist, inner_T_hist,  color=TI_COL, lw=2, label="Inner wall (Ti)")
ax2.axhline(T_cabin_setpoint, color=AC_COL, ls=":", lw=1.5,
            label=f"Cabin setpoint {T_cabin_setpoint - 273.15:.0f} °C")
limit_ti = MATERIALS["Ti_6Al_4V"].get("max_service_temp")
if limit_ti:
    ax2.axhline(limit_ti, color=TI_COL, ls="--", lw=1.2, alpha=0.7,
                label=f"Ti-6Al-4V limit {limit_ti} K")
ax2.set(xlabel="Time [s]", ylabel="Temperature [K]",
        title="Surface Temperatures")
ax2.legend(fontsize=8, framealpha=0.5)
ax2.grid(alpha=0.25)

# (c) Inner wall temperature in °C — zoomed on engineering range
inner_C = inner_T_hist - 273.15
cabin_C = T_cabin_setpoint - 273.15
ax3.plot(time_hist, inner_C, color=TI_COL, lw=2, label="Ti inner wall")
ax3.axhline(cabin_C, color=AC_COL, ls=":", lw=1.5,
            label=f"Cabin air {cabin_C:.1f} °C")
ax3.fill_between(time_hist, cabin_C, inner_C,
                 where=(inner_C > cabin_C),
                 alpha=0.15, color=TI_COL,
                 label="ΔT driving AC load")
ax3.set(xlabel="Time [s]", ylabel="Temperature [°C]",
        title="Inner Wall Temperature (Ti-6Al-4V)")
ax3.legend(fontsize=8, framealpha=0.5)
ax3.grid(alpha=0.25)

# (d) Layer-average temperatures
for i, (name, _) in enumerate(layer_stack):
    ax4.plot(time_hist, layer_T_hist[name],
             lw=1.8, color=cmap_lay(i), label=name)
ax4.set(xlabel="Time [s]", ylabel="Mean Layer Temperature [K]",
        title="Layer-Average Temperatures")
ax4.legend(fontsize=8, framealpha=0.5)
ax4.grid(alpha=0.25)

# (e) Instantaneous AC power demand  [W/m²]
ax5.fill_between(time_hist, q_cabin_hist, alpha=0.18, color=AC_COL)
ax5.plot(time_hist, q_cabin_hist, color=AC_COL, lw=2)
ax5.set(xlabel="Time [s]", ylabel="AC Cooling Load [W/m²]",
        title="Instantaneous AC Cooling Load (inner wall → cabin)")
ax5.grid(alpha=0.25)

# (f) Cumulative AC energy  [kJ/m²]
ax6.fill_between(time_hist, cum_ac_kJ, alpha=0.18, color="#7B5EA7")
ax6.plot(time_hist, cum_ac_kJ, color="#7B5EA7", lw=2)
ax6.set(xlabel="Time [s]", ylabel="Cumulative AC Energy [kJ/m²]",
        title="Cumulative AC Energy Removed (flat-plate, per m²)")
ax6.grid(alpha=0.25)

plt.suptitle(
    f"TPS + Cabin AC  ·  Crank–Nicolson Implicit Solver\n"
    f"Peak flux {peak_heat_flux/1e3:.0f} kW/m²  ·  {simulation_time/60:.0f} min  ·  "
    f"Cabin {T_cabin_setpoint - 273.15:.0f} °C  ·  h_inner = {h_inner:.0f} W/m²K",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()

plt.show()

# ── Figure 2: Through-thickness snapshots ─────────────────────────────────
fig2, ax = plt.subplots(figsize=(11, 6))
x_mm = x * 1e3

cx_mm = 0.0
for i, (name, th) in enumerate(layer_stack):
    end_mm = cx_mm + th * 1e3
    ax.axvspan(cx_mm, end_mm, alpha=0.10, color=cmap_lay(i), label=name)
    ax.text((cx_mm + end_mm) / 2,
            initial_temperature + 8,
            name.replace("_", "\n"), ha="center", va="bottom",
            fontsize=7.5, color=cmap_lay(i), fontweight="bold",
            linespacing=1.2)
    cx_mm = end_mm

for (t_snap, T_snap), col in zip(snapshots, SHADES):
    ax.plot(x_mm, T_snap, color=col, lw=1.8, label=f"t = {t_snap:.0f} s")

ax.axhline(T_cabin_setpoint, color=AC_COL, ls=":", lw=1.5,
           label=f"Cabin setpoint {T_cabin_setpoint - 273.15:.0f} °C")

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
