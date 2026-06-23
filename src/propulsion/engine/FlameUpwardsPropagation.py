import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
#  RAMJET FLASHBACK ANALYSIS
#  Z22_H2_ZNOx20 mechanism — Zettervall, FOI 2020
#  Accounts for: separate H2/air temperatures, air kinetic energy,
#  momentum-correct mixed velocity
# ═══════════════════════════════════════════════════════════════

MECH = 'Z22_H2_ZNOx20.yaml'

# ── Molecular weights ───────────────────────────────────────────
MW_H2  = 2.016e-3    # kg/mol
MW_O2  = 32.0e-3
MW_N2  = 28.014e-3
MW_air = 0.21 * MW_O2 + 0.79 * MW_N2   # 28.85e-3 kg/mol

# Stoichiometric air-fuel ratio (mass) for H2
# H2 + 0.5 O2: moles air per mole H2 = 0.5/0.21
AFR_STOICH = (0.5 / 0.21 * MW_air) / MW_H2   # ≈ 34.3


# ───────────────────────────────────────────────────────────────
#  CORE FUNCTION: ramjet inlet mixing
# ───────────────────────────────────────────────────────────────
def ramjet_mixed_inlet(phi, T_air_static, U_air, T_H2, P):
    """
    Adiabatic mixing of air (high velocity) + H2 (negligible velocity).

    Conserves:
      - Total enthalpy  (static + kinetic)
      - Momentum        (H2 contribution ~ 0)

    Parameters
    ----------
    phi          : equivalence ratio
    T_air_static : air static temperature at combustor inlet [K]
    U_air        : air velocity at combustor inlet [m/s]
    T_H2         : hydrogen injection temperature [K]
    P            : pressure [Pa]

    Returns
    -------
    T_mix  : mixed static temperature [K]
    T0_mix : mixed stagnation temperature [K]
    U_mix  : mixed velocity [m/s]  <-- use for flashback check
    X_mix  : mole fraction dict of the mixture
    """
    # Mass of H2 per kg of air at this phi
    m_air = 1.0
    m_H2  = phi / AFR_STOICH
    m_mix = m_air + m_H2

    # Momentum conservation (H2 velocity ≈ 0)
    U_mix = (m_air * U_air) / m_mix

    # Mole fractions of mixture
    mol_air = m_air / MW_air
    mol_H2  = m_H2  / MW_H2
    mol_O2  = mol_air * 0.21
    mol_N2  = mol_air * 0.79
    mol_tot = mol_H2 + mol_O2 + mol_N2

    X_mix = {
        'H2': mol_H2 / mol_tot,
        'O2': mol_O2 / mol_tot,
        'N2': mol_N2 / mol_tot,
    }

    # Static enthalpies from Cantera
    g_air = ct.Solution(MECH)
    g_air.TPX = T_air_static, P, {'O2': 0.21, 'N2': 0.79}
    h_air = g_air.enthalpy_mass   # J/kg

    g_H2 = ct.Solution(MECH)
    g_H2.TPX = T_H2, P, {'H2': 1.0}
    h_H2 = g_H2.enthalpy_mass    # J/kg

    # Total enthalpy balance
    # Air contributes static enthalpy + kinetic energy
    # H2 contributes only static enthalpy (no KE)
    H_total      = m_air * (h_air + 0.5 * U_air**2) + m_H2 * h_H2
    h_total_mix  = H_total / m_mix
    h_static_mix = h_total_mix - 0.5 * U_mix**2

    # Cantera solves for T given h, P, X
    g_mix = ct.Solution(MECH)
    g_mix.TPX = 0.5 * (T_air_static + T_H2), P, X_mix  # initial guess
    g_mix.HPX = h_static_mix, P, X_mix                  # solve for T

    T_mix  = g_mix.T
    T0_mix = T_mix + U_mix**2 / (2.0 * g_mix.cp_mass)

    return T_mix, T0_mix, U_mix, X_mix


# ───────────────────────────────────────────────────────────────
#  CORE FUNCTION: laminar flame speed
# ───────────────────────────────────────────────────────────────
def laminar_flame_speed(phi, T_mix, P, X_mix=None):
    """
    1D freely-propagating flame speed [m/s].
    Uses X_mix if provided (ramjet premixed state),
    otherwise builds from phi assuming standard air composition.
    Returns np.nan on solver failure (outside flammability limits).
    """
    g = ct.Solution(MECH)

    if X_mix is not None:
        g.TPX = T_mix, P, X_mix
    else:
        g.set_equivalence_ratio(phi, 'H2', {'O2': 1.0, 'N2': 3.76})
        g.TP = T_mix, P

    flame = ct.FreeFlame(g, width=0.03)
    flame.set_refine_criteria(ratio=3, slope=0.07, curve=0.14)
    flame.soret_enabled = False

    try:
        flame.solve(loglevel=0, auto=True)
        return flame.velocity[0]
    except Exception as e:
        return np.nan


# ═══════════════════════════════════════════════════════════════
#  OPERATING POINT — set your values here
# ═══════════════════════════════════════════════════════════════
phi       = 1.0
T_air     = 800.0    # K   — air static temp at combustor inlet (post-diffuser)
U_air     = 150.0    # m/s — air velocity at combustor inlet
T_H2      = 280.0    # K   — H2 injection temperature
P         = 10e5     # Pa  — combustor pressure

# ── Compute mixed inlet conditions ─────────────────────────────
T_mix, T0_mix, U_mix, X_mix = ramjet_mixed_inlet(phi, T_air, U_air, T_H2, P)

print(f"\n{'='*55}")
print(f"  RAMJET INLET MIXING")
print(f"{'='*55}")
print(f"  Air  static T   : {T_air:.1f} K")
print(f"  Air  velocity   : {U_air:.1f} m/s")
print(f"  Air  total T    : {T_air + U_air**2/(2*1005):.1f} K")
print(f"  H2   temperature: {T_H2:.1f} K")
print(f"  KE contribution : {0.5*U_air**2:.0f} J/kg  "
      f"({100*(0.5*U_air**2)/(0.5*U_air**2 + 1005*T_air)*100:.2f}‰ of total)")
print(f"{'─'*55}")
print(f"  Mixed static  T : {T_mix:.1f} K")
print(f"  Mixed total   T : {T0_mix:.1f} K")
print(f"  Mixed velocity  : {U_mix:.2f} m/s")
print(f"  X(H2)  = {X_mix['H2']:.4f}   X(O2) = {X_mix['O2']:.4f}   "
      f"X(N2) = {X_mix['N2']:.4f}")
print(f"{'='*55}")

# ── Single operating point flame speed ─────────────────────────
print(f"\nComputing S_L at operating point...")
S_L = laminar_flame_speed(phi, T_mix, P, X_mix)

print(f"\n{'='*55}")
print(f"  FLASHBACK CHECK — operating point")
print(f"{'='*55}")
print(f"  Laminar flame speed S_L : {S_L:.3f} m/s")
print(f"  Mixed flow velocity U   : {U_mix:.3f} m/s")
print(f"  Margin  U - S_L         : {U_mix - S_L:+.3f} m/s")
if U_mix > S_L:
    print(f"  Status : SAFE  (flow holds flame downstream)")
else:
    print(f"  Status : *** FLASHBACK RISK ***")
print(f"{'='*55}")


# ═══════════════════════════════════════════════════════════════
#  SWEEPS
# ═══════════════════════════════════════════════════════════════

# ── Sweep 1: equivalence ratio ──────────────────────────────────
print(f"\nSweeping equivalence ratio...")
phi_arr   = np.linspace(0.3, 2.5, 30)
SL_phi    = []
Umix_phi  = []
Tmix_phi  = []

for phi_i in phi_arr:
    T_m, _, U_m, X_m = ramjet_mixed_inlet(phi_i, T_air, U_air, T_H2, P)
    sl = laminar_flame_speed(phi_i, T_m, P, X_m)
    SL_phi.append(sl)
    Umix_phi.append(U_m)
    Tmix_phi.append(T_m)
    print(f"  phi={phi_i:.2f}  T_mix={T_m:.0f}K  U_mix={U_m:.1f}m/s  "
          f"S_L={sl:.3f}m/s", end='\r')

SL_phi   = np.array(SL_phi)
Umix_phi = np.array(Umix_phi)
Tmix_phi = np.array(Tmix_phi)
margin_phi = Umix_phi - SL_phi

# ── Sweep 2: air velocity ───────────────────────────────────────
print(f"\nSweeping air inlet velocity...")
U_arr     = np.linspace(30, 400, 30)
SL_U      = []
Umix_U    = []
Tmix_U    = []

for U_i in U_arr:
    T_m, _, U_m, X_m = ramjet_mixed_inlet(phi, T_air, U_i, T_H2, P)
    sl = laminar_flame_speed(phi, T_m, P, X_m)
    SL_U.append(sl)
    Umix_U.append(U_m)
    Tmix_U.append(T_m)
    print(f"  U={U_i:.0f}m/s  T_mix={T_m:.0f}K  U_mix={U_m:.1f}m/s  "
          f"S_L={sl:.3f}m/s", end='\r')

SL_U     = np.array(SL_U)
Umix_U   = np.array(Umix_U)
Tmix_U   = np.array(Tmix_U)
margin_U = Umix_U - SL_U

# ── Sweep 3: air temperature ────────────────────────────────────
print(f"\nSweeping air temperature...")
Tair_arr  = np.linspace(400, 1200, 25)
SL_T      = []
Umix_T    = []
Tmix_T    = []

for T_i in Tair_arr:
    T_m, _, U_m, X_m = ramjet_mixed_inlet(phi, T_i, U_air, T_H2, P)
    sl = laminar_flame_speed(phi, T_m, P, X_m)
    SL_T.append(sl)
    Umix_T.append(U_m)
    Tmix_T.append(T_m)
    print(f"  T_air={T_i:.0f}K  T_mix={T_m:.0f}K  S_L={sl:.3f}m/s", end='\r')

SL_T     = np.array(SL_T)
Umix_T   = np.array(Umix_T)
Tmix_T   = np.array(Tmix_T)
margin_T = Umix_T - SL_T

# ── Sweep 4: pressure ───────────────────────────────────────────
print(f"\nSweeping pressure...")
P_arr    = np.linspace(1e5, 30e5, 20)
SL_P     = []
Umix_P   = []

for P_i in P_arr:
    T_m, _, U_m, X_m = ramjet_mixed_inlet(phi, T_air, U_air, T_H2, P_i)
    sl = laminar_flame_speed(phi, T_m, P_i, X_m)
    SL_P.append(sl)
    Umix_P.append(U_m)
    print(f"  P={P_i/1e5:.1f}bar  T_mix={T_m:.0f}K  S_L={sl:.3f}m/s", end='\r')

SL_P     = np.array(SL_P)
Umix_P   = np.array(Umix_P)
margin_P = np.array(Umix_P) - SL_P


# ═══════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle(
    f'Ramjet Flashback Analysis — Z22_H2_ZNOx20\n'
    f'Baseline: φ={phi}  T_air={T_air:.0f}K  U_air={U_air:.0f}m/s  '
    f'T_H2={T_H2:.0f}K  P={P/1e5:.0f}bar',
    fontsize=11
)

def shade_flashback(ax, x, margin):
    """Shade red where flashback (margin < 0), green where safe."""
    ax.fill_between(x, margin, 0,
                    where=(margin < 0),  alpha=0.25, color='red',   label='Flashback')
    ax.fill_between(x, margin, 0,
                    where=(margin >= 0), alpha=0.15, color='green', label='Safe')
    ax.axhline(0, color='red', lw=1.5, ls='--')

# ── Plot 1: S_L and U_mix vs phi ────────────────────────────────
ax = axes[0, 0]
ax.plot(phi_arr, SL_phi,   'b',  lw=2, label='S_L (flame speed)')
ax.plot(phi_arr, Umix_phi, 'k',  lw=2, label='U_mix (flow speed)')
ax.axvline(phi, ls=':', color='gray', label=f'φ = {phi}')
ax.fill_between(phi_arr, SL_phi, Umix_phi,
                where=(SL_phi > Umix_phi), alpha=0.2, color='red',   label='Flashback')
ax.fill_between(phi_arr, SL_phi, Umix_phi,
                where=(SL_phi <= Umix_phi), alpha=0.1, color='green', label='Safe')
ax.set_xlabel('Equivalence ratio φ')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('Flame speed & flow speed vs φ')
ax.legend(fontsize=8)
ax.set_ylim(bottom=0)

# ── Plot 2: Flashback margin vs phi ────────────────────────────
ax = axes[0, 1]
ax.plot(phi_arr, margin_phi, 'k', lw=2)
ax.axvline(phi, ls=':', color='gray', label=f'φ = {phi}')
shade_flashback(ax, phi_arr, margin_phi)
ax.set_xlabel('Equivalence ratio φ')
ax.set_ylabel('Margin  U_mix − S_L  (m/s)')
ax.set_title('Flashback safety margin vs φ')
ax.legend(fontsize=8)

# ── Plot 3: S_L and U_mix vs air velocity ───────────────────────
ax = axes[1, 0]
ax.plot(U_arr, SL_U,   'b', lw=2, label='S_L')
ax.plot(U_arr, Umix_U, 'k', lw=2, label='U_mix')
ax.plot(U_arr, Tmix_T * 0, alpha=0)   # dummy
ax.axvline(U_air, ls=':', color='gray', label=f'U_air = {U_air} m/s')
ax.fill_between(U_arr, SL_U, Umix_U,
                where=(SL_U > Umix_U),  alpha=0.2,  color='red',   label='Flashback')
ax.fill_between(U_arr, SL_U, Umix_U,
                where=(SL_U <= Umix_U), alpha=0.1,  color='green', label='Safe')
ax.set_xlabel('Air inlet velocity U_air (m/s)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('Flame speed & flow speed vs U_air')
ax.legend(fontsize=8)
ax.set_ylim(bottom=0)

# ── Plot 4: Flashback margin vs air velocity ────────────────────
ax = axes[1, 1]
ax.plot(U_arr, margin_U, 'k', lw=2)
ax.axvline(U_air, ls=':', color='gray', label=f'U_air = {U_air} m/s')
shade_flashback(ax, U_arr, margin_U)
ax.set_xlabel('Air inlet velocity U_air (m/s)')
ax.set_ylabel('Margin  U_mix − S_L  (m/s)')
ax.set_title('Flashback safety margin vs U_air')
ax.legend(fontsize=8)

# ── Plot 5: S_L and T_mix vs air temperature ────────────────────
ax = axes[2, 0]
ax2 = ax.twinx()
ax.plot(Tair_arr, SL_T,   'b', lw=2, label='S_L')
ax.plot(Tair_arr, Umix_T, 'k', lw=2, label='U_mix')
ax2.plot(Tair_arr, Tmix_T, 'r--', lw=1.5, label='T_mix (right)')
ax.axvline(T_air, ls=':', color='gray', label=f'T_air = {T_air} K')
ax.fill_between(Tair_arr, SL_T, Umix_T,
                where=(np.array(SL_T) > np.array(Umix_T)),
                alpha=0.2, color='red')
ax.fill_between(Tair_arr, SL_T, Umix_T,
                where=(np.array(SL_T) <= np.array(Umix_T)),
                alpha=0.1, color='green')
ax.set_xlabel('Air inlet temperature T_air (K)')
ax.set_ylabel('Velocity (m/s)')
ax2.set_ylabel('T_mix (K)', color='r')
ax.set_title('Flame speed vs air temperature\n(S_L rises steeply with T)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax.set_ylim(bottom=0)

# ── Plot 6: Flashback margin vs pressure ────────────────────────
ax = axes[2, 1]
ax.plot(P_arr / 1e5, margin_P, 'k', lw=2)
ax.axvline(P / 1e5, ls=':', color='gray', label=f'P = {P/1e5:.0f} bar')
shade_flashback(ax, P_arr / 1e5, margin_P)
ax.set_xlabel('Pressure (bar)')
ax.set_ylabel('Margin  U_mix − S_L  (m/s)')
ax.set_title('Flashback safety margin vs pressure\n(S_L drops with P for H₂ — higher P helps)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('flashback_ramjet_Z22.png', dpi=150)
plt.show()
print("\nDone. Plot saved to flashback_ramjet_Z22.png")