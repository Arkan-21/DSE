"""
sweep_ramjet_v4.py
==================
Multi-condition parameter sweep for the Ramjet class with live progress tracking.
Phase 1: Full sweep across all flight conditions.
Phase 2: Re-sweep only geometries (A2, A3, A6, L_comb, phi, A6_frac) that
         passed (no choke + met thrust) in EVERY flight condition from Phase 1.
 
Swept parameters
----------------
  A0      : inlet capture area        [m²]   3 values
  L_comb  : total combustor length    [m]    4 values  (60% L23, 40% L34)
  phi     : equivalence ratio         [—]    6 values
  A6_FRAC : nozzle exit area ratio    [—]    7 values
 
Flight Conditions Swept
-----------------------
  1. Ma=2.75, h=17 km,  Min Thrust = 300 kN
  2. Ma=4.40, h=29.3 km, Min Thrust = 170 kN
  3. Ma=5.00, h=30 km,   Min Thrust = 140 kN
"""
 
import sys, io, contextlib, itertools, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
 
sys.path.insert(0, "/home/claude")
from ramjet_fixedgeometry import Ramjet, Geometry, Assumptions
 
# ──────────────────────────────────────────────────────────────────────────────
# FLIGHT CONDITIONS MATRIX
# ──────────────────────────────────────────────────────────────────────────────
FLIGHT_CONDITIONS = [
    {"Ma0": 4.40, "h0": 29_300.0, "thrust_min": 170_000.0, "label": "M44_h293_T170"},
    {"Ma0": 5.00, "h0": 30_000.0, "thrust_min": 140_000.0, "label": "M5_h300_T140"},
    {"Ma0": 2.75, "h0": 17_000.0, "thrust_min": 300_000.0, "label": "M275_h17_T300"},
]
 
# ──────────────────────────────────────────────────────────────────────────────
# SWEEP RANGES
# ──────────────────────────────────────────────────────────────────────────────
A0_values     = [4.0, 4.1, 4.2, 4.3, 4.4, 4.5]
L_comb_values = [0.30, 0.50, 0.70, 0.90]
phi_values    = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
A6_FRAC       = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
 
# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY RATIOS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
A2_FRAC  = 0.90
A3_FRAC  = 1.10
L23_FRAC = 0.60
L34_FRAC = 0.40
L01_VAL  = 0.50
L12_VAL  = 0.25
L45_VAL  = 0.35
L56_VAL  = 1.20
MA_COMB  = 0.30
CF       = 0.003
THETA    = 90.0
MIX_COEF = 0.176
 
# Columns that uniquely identify an engine geometry (A1 excluded — flight-dependent)
GEOMETRY_COLS = ["A2_m2", "A3_m2", "A6_m2", "L_comb_m", "phi", "A6_frac"]
 
# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR = r"C:\Users\arkan\OneDrive\Desktop\DSE-1\src\propulsion\engine"
 
 
# ──────────────────────────────────────────────────────────────────────────────
# RUN ONE CASE (silently)
# ──────────────────────────────────────────────────────────────────────────────
def run_case(A0, L_comb, phi, A6_frac, Ma0, h0):
    L23 = L23_FRAC * L_comb
    L34 = L34_FRAC * L_comb
 
    geom = Geometry(
        A0=A0, L01=L01_VAL, L12=L12_VAL, L23=L23, L34=L34,
        L45=L45_VAL, L56=L56_VAL,
        A2=A2_FRAC * A0, A3=A3_FRAC * A0, A4=A3_FRAC * A0, A6=A6_frac * A0,
    )
    assump = Assumptions(
        h0=h0, Ma0=Ma0, phi=phi, theta=THETA,
        mixing_coeff=MIX_COEF, Ma_COMB=MA_COMB, Cf=CF,
        HHV=141.8e6, FAR_stoich=1.0/34.35,
    )
 
    try:
        eng = Ramjet(geom=geom, assump=assump)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            inp  = eng.station_0()
            iso  = eng.station_1(inp)
            sec2 = eng.section_12(iso)
            sec3 = eng.section_23(sec2)
            sec4 = eng.section_34(sec3)
            sec5 = eng.section_45(sec4)
            sec6 = eng.section_56(sec5)
            perf = eng.performance(inp, sec6, sec3)
 
        choke = any(s.get("thermal_choke", False) for s in [sec2, sec3, sec4]) \
                or perf.get("thermal_choke", False)
        if choke:
            return None, "thermal_choke"
 
        Fin = perf["Fin"]
        if not np.isfinite(Fin):
            return None, "non_finite"
 
        return {
            "A0_m2"    : A0,
            "L_comb_m" : L_comb,
            "phi"      : phi,
            "A6_frac"  : A6_frac,
            "A1_m2"    : iso["A"],
            "A2_m2"    : geom.A2,
            "A3_m2"    : geom.A3,
            "A6_m2"    : geom.A6,
            "L23_m"    : L23,
            "L34_m"    : L34,
            "AR_nozzle": geom.A6 / sec5["A"],
            "sigma_c"  : iso["sigma_c"],
            "T1_K"     : iso["T"],
            "P1_kPa"   : iso["P"] / 1e3,
            "mdot_air" : inp["mdot"],
            "mdot_fuel": sec3["mfuel"],
            "FAR"      : sec3["mfuel"] / inp["mdot"],
            "Ma4"      : sec4["Ma"],
            "T4_K"     : sec4["T"],
            "Tt4_K"    : sec4["Tt"],
            "P4_kPa"   : sec4["P"] / 1e3,
            "Pt4_kPa"  : sec4["Pt"] / 1e3,
            "Ma6"      : sec6["Ma"],
            "V6_ms"    : sec6["V"],
            "P6_kPa"   : sec6["P"] / 1e3,
            "T6_K"     : sec6["T"],
            "Fin_N"    : Fin,
            "Isp_s"    : perf["Isp"],
            "Ia_Ns_kg" : perf["Ia"],
        }, None
 
    except Exception as e:
        return None, f"exception: {e}"
 
 
# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY KEY HELPER
# ──────────────────────────────────────────────────────────────────────────────
def geom_key(row, decimals=9):
    """Tuple key from the 6 geometry columns, rounded to avoid float noise."""
    return tuple(round(row[c], decimals) for c in GEOMETRY_COLS)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# SINGLE-CONDITION SWEEP
# ──────────────────────────────────────────────────────────────────────────────
def run_sweep(condition, param_combos, phase_label="PHASE 1"):
    """
    Run the sweep for one flight condition over *param_combos*
    (iterable of (A0, L_comb, phi, A6_frac) tuples).
 
    Returns a DataFrame of passing rows and failure counters.
    """
    MA0        = condition["Ma0"]
    H0         = condition["h0"]
    THRUST_MIN = condition["thrust_min"]
    LABEL      = condition["label"]
 
    combos = list(param_combos)
    total  = len(combos)
 
    print(f"\n{'═'*80}")
    print(f"  [{phase_label}]  SWEEP: {LABEL}  —  Ma={MA0}  h={H0/1e3:.1f} km")
    print(f"  Cases to run : {total}   |   Thrust threshold : {THRUST_MIN/1e3:.0f} kN")
    print(f"{'═'*80}\n")
 
    passed = []
    fail_choke = fail_thrust = fail_err = 0
 
    for n_done, (A0, L_comb, phi, A6_frac) in enumerate(combos, 1):
        print(f"  [{n_done:>4}/{total}]  A0={A0:.1f}  Lcomb={L_comb:.2f}  "
              f"φ={phi:.2f}  A6_frac={A6_frac:.1f} … ", end="", flush=True)
 
        r, reason = run_case(A0, L_comb, phi, A6_frac, MA0, H0)
 
        if r is None:
            if reason == "thermal_choke": fail_choke += 1
            else:                         fail_err   += 1
            print(f"✗  ({reason})")
        elif r["Fin_N"] < THRUST_MIN:
            fail_thrust += 1
            print(f"✗  Fin={r['Fin_N']/1e3:.1f} kN  (< {THRUST_MIN/1e3:.0f} kN threshold)")
        else:
            print(f"✓  Fin={r['Fin_N']/1e3:.1f} kN   Isp={r['Isp_s']:.0f} s")
            passed.append(r)
 
    print(f"\n{'─'*80}")
    print(f"  [SUMMARY  {LABEL}]  Passed: {len(passed)}/{total}  |  "
          f"Choked: {fail_choke}  |  Low Fin: {fail_thrust}  |  Errors: {fail_err}")
    print(f"{'─'*80}\n")
 
    df = pd.DataFrame(passed)
    if not df.empty:
        df = df.sort_values("Fin_N", ascending=False).reset_index(drop=True)
        df.index += 1
    return df
 
 
# ──────────────────────────────────────────────────────────────────────────────
# SAVE CSV + PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def save_outputs(df, condition, phase_suffix=""):
    MA0        = condition["Ma0"]
    H0         = condition["h0"]
    THRUST_MIN = condition["thrust_min"]
    LABEL      = condition["label"]
    total      = len(df)
 
    # Print table
    print("═"*88)
    print(f"  {'VALID SOLUTIONS — ' + LABEL + phase_suffix + ' (sorted by thrust)':^84}")
    print("═"*88)
    hdr = (f"  {'#':>3}  {'A0[m²]':>7}  {'Lcomb[m]':>9}  {'φ':>5}  {'A6_frac':>7}  "
           f"{'Fin[N]':>9}  {'Isp[s]':>7}  {'Ma4':>5}  {'T4[K]':>7}  "
           f"{'Ma6':>5}  {'mdot_air':>9}")
    print(hdr)
    print("  " + "─"*(len(hdr)-2))
    for i, row in df.head(15).iterrows():
        print(f"  {i:>3}  {row['A0_m2']:>7.1f}  {row['L_comb_m']:>9.2f}  {row['phi']:>5.2f}  "
              f"{row['A6_frac']:>7.1f}  {row['Fin_N']:>9.0f}  {row['Isp_s']:>7.1f}  "
              f"{row['Ma4']:>5.4f}  {row['T4_K']:>7.1f}  {row['Ma6']:>5.3f}  "
              f"{row['mdot_air']:>9.2f}")
    if total > 15:
        print(f"  ... and {total - 15} more rows inside the saved CSV.")
    print()
 
    # CSV
    tag      = f"{LABEL}{phase_suffix}"
    csv_path = f"{OUT_DIR}\\sweepresults_{tag}.csv"
    df.to_csv(csv_path, index_label="rank")
    print(f"  ✓ CSV saved → {csv_path}")
 
    # Plots
    phis_uniq = sorted(df["phi"].unique())
    Lc_uniq   = sorted(df["L_comb_m"].unique())
    cmap_phi  = cm.get_cmap("plasma",  len(phis_uniq))
    cmap_Lc   = cm.get_cmap("viridis", len(Lc_uniq))
    phi_color = {p: cmap_phi(i) for i, p in enumerate(phis_uniq)}
    Lc_color  = {L: cmap_Lc(i) for i, L in enumerate(Lc_uniq)}
 
    def scat(ax, xcol, ycol, color_by, xlabel, ylabel, title):
        colors = phi_color if color_by == "phi" else Lc_color
        keys   = phis_uniq if color_by == "phi" else Lc_uniq
        lbl    = "φ" if color_by == "phi" else "L_comb"
        for k in keys:
            sub = df[df[color_by] == k]
            ax.scatter(sub[xcol], sub[ycol], c=[colors[k]]*len(sub),
                       s=90, label=f"{lbl}={k:.2f}",
                       edgecolors="k", linewidths=0.5, zorder=3)
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)
 
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.suptitle(
        f"Ramjet Sweep  —  Ma={MA0}, h={H0/1e3:.1f} km  [{tag}]\n"
        f"H₂ fuel, θ=90°, Ma_comb=0.30 | A2={A2_FRAC}·A0, A3={A3_FRAC}·A0\n"
        f"Valid: {total} cases",
        fontsize=11, fontweight="bold")
 
    scat(axes[0,0], "A0_m2",    "Fin_N", "phi",     "A₀ [m²]",    "Thrust [N]", "Thrust vs Inlet Area\n(coloured by φ)")
    scat(axes[0,1], "phi",      "Fin_N", "L_comb_m","φ [—]",       "Thrust [N]", "Thrust vs φ\n(coloured by L_comb)")
    scat(axes[0,2], "L_comb_m", "Fin_N", "phi",     "L_comb [m]",  "Thrust [N]", "Thrust vs Combustor Length\n(coloured by φ)")
    scat(axes[1,0], "A0_m2",    "Isp_s", "phi",     "A₀ [m²]",    "Isp [s]",    "Isp vs Inlet Area\n(coloured by φ)")
    scat(axes[1,1], "phi",      "Isp_s", "L_comb_m","φ [—]",       "Isp [s]",    "Isp vs φ\n(coloured by L_comb)")
    scat(axes[1,2], "Fin_N",    "Isp_s", "phi",     "Thrust [N]",  "Isp [s]",    "Thrust–Isp Pareto View\n(coloured by φ)")
 
    for ax in axes[0]:
        ax.axhline(THRUST_MIN, color="red", lw=1.3, ls="--", alpha=0.7,
                   label=f"Min {THRUST_MIN/1e3:.0f} kN")
 
    best = df.iloc[0]
    for ax, xcol in [(axes[0,0],"A0_m2"), (axes[0,1],"phi"), (axes[0,2],"L_comb_m")]:
        ax.annotate(f"  #1\n  {best['Fin_N']/1e3:.0f} kN",
                    xy=(best[xcol], best["Fin_N"]), fontsize=7.5,
                    color="red", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
                    xytext=(best[xcol] + 0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),
                            best["Fin_N"]*0.97))
 
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p1 = f"{OUT_DIR}\\ramjet_sweep_overview_{tag}.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ Overview plot → {p1}")
 
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle(f"Combustor & Nozzle State — {tag}", fontsize=11, fontweight="bold")
    scat(axes2[0], "T4_K", "Fin_N", "phi", "T₄ [K]",  "Thrust [N]", "Combustor Exit Temp vs Thrust")
    scat(axes2[1], "Ma6",  "Fin_N", "phi", "Ma₆ [—]", "Thrust [N]", "Nozzle Exit Mach vs Thrust")
    scat(axes2[2], "Ma4",  "Isp_s", "phi", "Ma₄ [—]", "Isp [s]",    "Combustor Exit Mach vs Isp")
    for ax in axes2[:2]:
        ax.axhline(THRUST_MIN, color="red", lw=1.3, ls="--", alpha=0.7)
    plt.tight_layout()
    p2 = f"{OUT_DIR}\\ramjet_sweep_combustor_{tag}.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ Combustor plot → {p2}")
 
    best_isp_idx = df["Isp_s"].idxmax()
    print(f"\n  Best thrust : {df['Fin_N'].max()/1e3:.1f} kN  "
          f"(A0={df.loc[1,'A0_m2']:.1f}, Lc={df.loc[1,'L_comb_m']:.2f}, "
          f"φ={df.loc[1,'phi']:.2f}, A6_frac={df.loc[1,'A6_frac']:.1f})")
    print(f"  Best Isp    : {df['Isp_s'].max():.1f} s  "
          f"(A0={df.loc[best_isp_idx,'A0_m2']:.1f}, Lc={df.loc[best_isp_idx,'L_comb_m']:.2f}, "
          f"φ={df.loc[best_isp_idx,'phi']:.2f}, A6_frac={df.loc[best_isp_idx,'A6_frac']:.1f})")
    print(f"{'═'*80}\n")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# ██████████████████████  PHASE 1 — FULL SWEEP  ████████████████████████████████
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "█"*80)
print("  PHASE 1 — Full sweep across all flight conditions")
print("█"*80)
 
all_combos  = list(itertools.product(A0_values, L_comb_values, phi_values, A6_FRAC))
phase1_dfs  = {}   # label → DataFrame of passing rows
 
for condition in FLIGHT_CONDITIONS:
    df = run_sweep(condition, all_combos, phase_label="PHASE 1")
    phase1_dfs[condition["label"]] = df
 
    if df.empty:
        print(f"  ⚠  No passing solutions for {condition['label']} in Phase 1.")
    else:
        save_outputs(df, condition, phase_suffix="")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# INTERSECT: find geometry keys that passed in EVERY condition
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "█"*80)
print("  COMPUTING GEOMETRY INTERSECTION ACROSS ALL FLIGHT CONDITIONS")
print("█"*80)
 
# Build a set of passing geometry keys per condition
passing_key_sets = {}
for label, df in phase1_dfs.items():
    if df.empty:
        passing_key_sets[label] = set()
    else:
        passing_key_sets[label] = set(df.apply(geom_key, axis=1))
 
# Intersect across all conditions
common_keys = set.intersection(*passing_key_sets.values()) if passing_key_sets else set()
 
print(f"\n  Passing geometries per condition:")
for label, keys in passing_key_sets.items():
    print(f"    {label:25s}: {len(keys):>5} geometries")
print(f"\n  ✓ Common to ALL conditions      : {len(common_keys)} geometries")
 
if not common_keys:
    print("\n  ✗ No geometry passes all conditions. Phase 2 skipped.")
    print("\nALL SWEEPS COMPLETE.")
    sys.exit(0)
 
# Convert common keys back to (A0, L_comb, phi, A6_frac) combos for Phase 2.
# Each key is (A2, A3, A6, L_comb, phi, A6_frac); recover inputs via the ratios.
#   A2 = A2_FRAC * A0  →  A0 = A2 / A2_FRAC
#   A6 = A6_frac * A0  →  A6_frac = A6 / A0
phase2_combos = []
seen = set()
for key in common_keys:
    A2_val, A3_val, A6_val, L_comb_val, phi_val, A6_frac_val = key
    A0_val = round(A2_val / A2_FRAC, 9)
    combo  = (A0_val, L_comb_val, phi_val, A6_frac_val)
    if combo not in seen:
        seen.add(combo)
        phase2_combos.append(combo)
 
phase2_combos.sort()
print(f"  Phase 2 parameter combinations : {len(phase2_combos)}\n")
 
# Pretty-print the winning geometries
print(f"  {'A0 [m²]':>8}  {'L_comb [m]':>10}  {'phi':>5}  {'A6_frac':>8}")
print("  " + "─"*38)
for A0_val, L_comb_val, phi_val, A6_frac_val in phase2_combos:
    print(f"  {A0_val:>8.3f}  {L_comb_val:>10.2f}  {phi_val:>5.2f}  {A6_frac_val:>8.2f}")
print()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# ██████████████████████  PHASE 2 — FILTERED RE-SWEEP  █████████████████████████
# ──────────────────────────────────────────────────────────────────────────────
print("█"*80)
print("  PHASE 2 — Re-sweep using only geometries that passed ALL conditions")
print("█"*80)
 
for condition in FLIGHT_CONDITIONS:
    df = run_sweep(condition, phase2_combos, phase_label="PHASE 2")
 
    if df.empty:
        print(f"  ⚠  No passing solutions for {condition['label']} in Phase 2.")
        continue
 
    save_outputs(df, condition, phase_suffix="_phase2")
 
print("ALL SWEEPS COMPLETE.")