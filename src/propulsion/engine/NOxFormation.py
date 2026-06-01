import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
#  RAMJET NOx FORMATION STUDY
#  Z22_H2_ZNOx20 mechanism — Zettervall, FOI 2020
#  Ramjet-correct inlet: separate H2/air temps + air kinetic energy
# ═══════════════════════════════════════════════════════════════

MECH = 'Z22_H2_ZNOx20.yaml'

MW_H2  = 2.016e-3
MW_O2  = 32.0e-3
MW_N2  = 28.014e-3
MW_air = 0.21 * MW_O2 + 0.79 * MW_N2
AFR_STOICH = (0.5 / 0.21 * MW_air) / MW_H2   # ≈ 34.3


# ───────────────────────────────────────────────────────────────
#  RAMJET INLET MIXING  (identical to flashback script)
# ───────────────────────────────────────────────────────────────
def ramjet_mixed_inlet(phi, T_air_static, U_air, T_H2, P):
    """
    Adiabatic mixing: air (high velocity) + H2 (negligible velocity).
    Conserves total enthalpy (static + kinetic) and momentum.

    Returns T_mix, T0_mix, U_mix, X_mix
    """
    m_air = 1.0
    m_H2  = phi / AFR_STOICH
    m_mix = m_air + m_H2

    U_mix = (m_air * U_air) / m_mix

    mol_air = m_air / MW_air
    mol_H2  = m_H2  / MW_H2
    mol_tot = mol_H2 + mol_air * 0.21 / 0.21   # = mol_H2 + mol_air (all species)
    mol_O2  = mol_air * 0.21
    mol_N2  = mol_air * 0.79
    mol_tot = mol_H2 + mol_O2 + mol_N2

    X_mix = {
        'H2': mol_H2 / mol_tot,
        'O2': mol_O2 / mol_tot,
        'N2': mol_N2 / mol_tot,
    }

    g_air = ct.Solution(MECH)
    g_air.TPX = T_air_static, P, {'O2': 0.21, 'N2': 0.79}
    h_air = g_air.enthalpy_mass

    g_H2 = ct.Solution(MECH)
    g_H2.TPX = T_H2, P, {'H2': 1.0}
    h_H2 = g_H2.enthalpy_mass

    H_total      = m_air * (h_air + 0.5 * U_air**2) + m_H2 * h_H2
    h_total_mix  = H_total / m_mix
    h_static_mix = h_total_mix - 0.5 * U_mix**2

    g_mix = ct.Solution(MECH)
    g_mix.TPX = 0.5 * (T_air_static + T_H2), P, X_mix
    g_mix.HPX = h_static_mix, P, X_mix

    T_mix  = g_mix.T
    T0_mix = T_mix + U_mix**2 / (2.0 * g_mix.cp_mass)

    return T_mix, T0_mix, U_mix, X_mix


# ───────────────────────────────────────────────────────────────
#  NOx INTEGRATION  — run reactor for t_res, return all species
# ───────────────────────────────────────────────────────────────
def run_nox(phi, T_air, U_air, T_H2, P, t_residence,
            return_history=False):
    """
    Runs a constant-pressure batch reactor from mixed inlet conditions
    for t_residence seconds.

    Returns dict of NOx species in ppm at t_residence.
    If return_history=True, also returns (times, T_hist, nox_hist_dict).
    """
    T_mix, _, _, X_mix = ramjet_mixed_inlet(phi, T_air, U_air, T_H2, P)

    g = ct.Solution(MECH)
    g.TPX = T_mix, P, X_mix

    r   = ct.IdealGasConstPressureReactor(g)
    sim = ct.ReactorNet([r])
    sim.rtol = 1e-9
    sim.atol = 1e-15

    NOX_SPECIES = ['NO', 'NO2', 'N2O', 'HNO', 'NH', 'NH2', 'NH3', 'N']

    def get_X(gas, sp):
        return gas.X[gas.species_index(sp)]

    if return_history:
        times, T_hist = [], []
        hist = {sp: [] for sp in NOX_SPECIES}
        t = 0.0
        t_end = max(t_residence * 5, 0.05)
        while t < t_end:
            t = sim.step()
            times.append(t)
            T_hist.append(r.thermo.T)
            for sp in NOX_SPECIES:
                hist[sp].append(get_X(r.thermo, sp) * 1e6)
        times  = np.array(times)
        T_hist = np.array(T_hist)
        for sp in NOX_SPECIES:
            hist[sp] = np.array(hist[sp])
        # Interpolate at exact t_residence
        result = {sp: float(np.interp(t_residence, times, hist[sp]))
                  for sp in NOX_SPECIES}
        result['T_final'] = float(np.interp(t_residence, times, T_hist))
        result['T_mix']   = T_mix
        return result, (times, T_hist, hist)

    else:
        try:
            sim.advance(t_residence)
        except Exception:
            pass
        result = {sp: get_X(r.thermo, sp) * 1e6 for sp in NOX_SPECIES}
        result['T_final'] = r.thermo.T
        result['T_mix']   = T_mix
        return result


# ═══════════════════════════════════════════════════════════════
#  OPERATING POINT
# ═══════════════════════════════════════════════════════════════
phi         = 1.0
T_air       = 800.0    # K
U_air       = 150.0    # m/s
T_H2        = 280.0    # K
P           = 10e5     # Pa
t_residence = 2e-3     # s  ← your chamber residence time

# ── Single operating point with full time history ───────────────
print("Computing baseline NOx history...")
result, (times, T_hist, nox_hist) = run_nox(
    phi, T_air, U_air, T_H2, P, t_residence, return_history=True
)

print(f"\n{'='*55}")
print(f"  RAMJET NOx SUMMARY  —  operating point")
print(f"{'='*55}")
print(f"  Inlet:  T_air={T_air:.0f}K  U_air={U_air:.0f}m/s  "
      f"T_H2={T_H2:.0f}K  φ={phi}")
print(f"  Mixed:  T_mix={result['T_mix']:.1f} K")
print(f"  t_res = {t_residence*1e3:.2f} ms")
print(f"{'─'*55}")
print(f"  T_final  : {result['T_final']:.1f} K")
print(f"  NO       : {result['NO']:.4f}  ppm")
print(f"  NO2      : {result['NO2']:.4f}  ppm")
print(f"  N2O      : {result['N2O']:.6f} ppm")
print(f"  HNO      : {result['HNO']:.6f} ppm")
print(f"  NH3      : {result['NH3']:.6f} ppm")
print(f"  NOx total: {result['NO'] + result['NO2']:.4f}  ppm")
print(f"{'='*55}")


# ═══════════════════════════════════════════════════════════════
#  SWEEPS
# ═══════════════════════════════════════════════════════════════

def sweep(param_arr, sweep_key, fixed):
    """
    Generic sweep runner. sweep_key is one of:
    'phi', 't_res', 'T_air', 'U_air', 'T_H2', 'P'
    fixed is a dict with all other parameters.
    Returns array of result dicts.
    """
    results = []
    for val in param_arr:
        kw = dict(fixed)
        kw[sweep_key] = val
        r = run_nox(kw['phi'], kw['T_air'], kw['U_air'],
                    kw['T_H2'], kw['P'], kw['t_res'])
        results.append(r)
        print(f"  {sweep_key}={val:.4g}  NOx={r['NO']+r['NO2']:.4f} ppm  "
              f"T_mix={r['T_mix']:.0f}K  T_final={r['T_final']:.0f}K", end='\r')
    print()
    return results

base = dict(phi=phi, T_air=T_air, U_air=U_air,
            T_H2=T_H2, P=P, t_res=t_residence)

# ── Sweep 1: residence time ─────────────────────────────────────
print("\nSweeping residence time...")
t_arr   = np.logspace(-5, -1, 60)   # 10 µs → 100 ms
res_t   = sweep(t_arr, 't_res', base)
NOx_t   = np.array([r['NO'] + r['NO2'] for r in res_t])
NO_t    = np.array([r['NO']  for r in res_t])
NO2_t   = np.array([r['NO2'] for r in res_t])
N2O_t   = np.array([r['N2O'] for r in res_t])
Tf_t    = np.array([r['T_final'] for r in res_t])

# ── Sweep 2: equivalence ratio ──────────────────────────────────
print("Sweeping equivalence ratio...")
phi_arr  = np.linspace(0.4, 2.0, 30)
res_phi  = sweep(phi_arr, 'phi', base)
NOx_phi  = np.array([r['NO'] + r['NO2'] for r in res_phi])
Tmix_phi = np.array([r['T_mix']   for r in res_phi])
Tf_phi   = np.array([r['T_final'] for r in res_phi])

# ── Sweep 3: air temperature ────────────────────────────────────
print("Sweeping air temperature...")
Tair_arr = np.linspace(400, 1200, 25)
res_T    = sweep(Tair_arr, 'T_air', base)
NOx_T    = np.array([r['NO'] + r['NO2'] for r in res_T])
Tmix_T   = np.array([r['T_mix']   for r in res_T])
Tf_T     = np.array([r['T_final'] for r in res_T])

# ── Sweep 4: air velocity ───────────────────────────────────────
print("Sweeping air velocity...")
U_arr   = np.linspace(30, 400, 25)
res_U   = sweep(U_arr, 'U_air', base)
NOx_U   = np.array([r['NO'] + r['NO2'] for r in res_U])
Tmix_U  = np.array([r['T_mix']   for r in res_U])
Tf_U    = np.array([r['T_final'] for r in res_U])

# ── Sweep 5: pressure ───────────────────────────────────────────
print("Sweeping pressure...")
P_arr   = np.linspace(1e5, 30e5, 25)
res_P   = sweep(P_arr, 'P', base)
NOx_P   = np.array([r['NO'] + r['NO2'] for r in res_P])
Tmix_P  = np.array([r['T_mix']   for r in res_P])
Tf_P    = np.array([r['T_final'] for r in res_P])

# ── Sweep 6: H2 injection temperature ──────────────────────────
print("Sweeping H2 temperature...")
TH2_arr = np.linspace(100, 700, 25)
res_TH2 = sweep(TH2_arr, 'T_H2', base)
NOx_TH2 = np.array([r['NO'] + r['NO2'] for r in res_TH2])
Tmix_TH2= np.array([r['T_mix']   for r in res_TH2])
Tf_TH2  = np.array([r['T_final'] for r in res_TH2])


# ═══════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle(
    f'Ramjet NOx Formation — Z22_H2_ZNOx20\n'
    f'Baseline: φ={phi}  T_air={T_air:.0f}K  U_air={U_air:.0f}m/s  '
    f'T_H2={T_H2:.0f}K  P={P/1e5:.0f}bar  t_res={t_residence*1e3:.1f}ms',
    fontsize=10
)

# ── Plot 1: NOx species over time at operating point ────────────
ax = axes[0, 0]
colors = {'NO': 'blue', 'NO2': 'orange', 'N2O': 'green',
          'HNO': 'purple', 'NH3': 'brown'}
for sp, col in colors.items():
    ax.plot(times * 1e3, nox_hist[sp], label=sp, color=col, lw=1.8)
ax.axvline(t_residence * 1e3, ls='--', color='black', lw=1.2,
           label=f't_res={t_residence*1e3:.1f}ms')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Concentration (ppm)')
ax.set_title('NOx species — time evolution at operating point')
ax.set_xscale('log')
ax.legend(fontsize=8)

# ── Plot 2: Temperature over time ──────────────────────────────
ax = axes[0, 1]
ax.plot(times * 1e3, T_hist, 'r', lw=2)
ax.axvline(t_residence * 1e3, ls='--', color='black', lw=1.2,
           label=f't_res={t_residence*1e3:.1f}ms')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Temperature (K)')
ax.set_title('Temperature evolution at operating point')
ax.set_xscale('log')
ax.legend(fontsize=8)

# ── Plot 3: NOx vs residence time (all species) ─────────────────
ax = axes[1, 0]
ax.semilogx(t_arr * 1e3, NOx_t,  'k',      lw=2,   label='NOx (NO+NO2)')
ax.semilogx(t_arr * 1e3, NO_t,   'b--',    lw=1.5, label='NO')
ax.semilogx(t_arr * 1e3, NO2_t,  color='orange', ls='--', lw=1.5, label='NO2')
ax.semilogx(t_arr * 1e3, N2O_t,  'g--',    lw=1.5, label='N2O')
ax.axvline(t_residence * 1e3, ls=':', color='black',
           label=f't_res={t_residence*1e3:.1f}ms')
ax.set_xlabel('Residence time (ms)')
ax.set_ylabel('Concentration (ppm)')
ax.set_title('NOx vs residence time\n(confirms short t_res suppresses NOx)')
ax.legend(fontsize=8)

# ── Plot 4: NOx and T_final vs equivalence ratio ────────────────
ax = axes[1, 1]
ax2 = ax.twinx()
ax.plot(phi_arr, NOx_phi, 'b',  lw=2, label='NOx (left)')
ax2.plot(phi_arr, Tf_phi,  'r--', lw=1.5, label='T_final (right)')
ax2.plot(phi_arr, Tmix_phi,'r:', lw=1.2, label='T_mix (right)')
ax.axvline(phi, ls=':', color='gray', label=f'φ={phi}')
ax.set_xlabel('Equivalence ratio φ')
ax.set_ylabel('NOx (ppm)', color='b')
ax2.set_ylabel('Temperature (K)', color='r')
ax.set_title('NOx and temperature vs φ')
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)

plt.tight_layout()
plt.savefig('nox_study_1_Z22.png', dpi=150)


# ── Figure 2: parametric sweeps ─────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(
    f'Ramjet NOx — Parametric Sensitivity\n'
    f'Baseline: φ={phi}  T_air={T_air:.0f}K  U_air={U_air:.0f}m/s  '
    f'T_H2={T_H2:.0f}K  P={P/1e5:.0f}bar  t_res={t_residence*1e3:.1f}ms',
    fontsize=10
)

def twin_plot(ax, x, nox, tmix, tf, xlabel, title, marker_val=None, marker_label=''):
    ax2 = ax.twinx()
    ax.plot(x, nox,  'b',  lw=2,   label='NOx (left)')
    ax2.plot(x, tf,  'r--', lw=1.5, label='T_final (right)')
    ax2.plot(x, tmix,'r:',  lw=1.2, label='T_mix (right)')
    if marker_val is not None:
        ax.axvline(marker_val, ls=':', color='gray', label=marker_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('NOx (ppm)', color='b')
    ax2.set_ylabel('Temperature (K)', color='r')
    ax.set_title(title)
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)

twin_plot(axes2[0, 0], Tair_arr, NOx_T,   Tmix_T,   Tf_T,
          'Air inlet temperature (K)',
          'NOx vs air temperature\n(T_mix rises → more thermal NOx)',
          T_air, f'T_air={T_air:.0f}K')

twin_plot(axes2[0, 1], U_arr, NOx_U, Tmix_U, Tf_U,
          'Air inlet velocity U_air (m/s)',
          'NOx vs air velocity\n(higher U → more KE → higher T_mix)',
          U_air, f'U={U_air:.0f}m/s')

twin_plot(axes2[1, 0], P_arr / 1e5, NOx_P, Tmix_P, Tf_P,
          'Pressure (bar)',
          'NOx vs pressure\n(three-body reactions, NHx pathway)',
          P / 1e5, f'P={P/1e5:.0f}bar')

twin_plot(axes2[1, 1], TH2_arr, NOx_TH2, Tmix_TH2, Tf_TH2,
          'H2 injection temperature (K)',
          'NOx vs H2 temperature\n(T_mix shifts with H2 enthalpy)',
          T_H2, f'T_H2={T_H2:.0f}K')

plt.tight_layout()
plt.savefig('nox_study_2_Z22.png', dpi=150)

plt.show()
print("\nDone. Plots saved to nox_study_1_Z22.png and nox_study_2_Z22.png")