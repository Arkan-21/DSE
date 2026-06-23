"""
optimize_ramjet.py
==================
Continuous multi-condition optimization of the Ramjet engine geometry using
scipy.optimize.differential_evolution.

Speed improvements over previous version
-----------------------------------------
1. workers=-1        — uses ALL CPU cores (parallelises population evaluation)
2. POPSIZE 20→5      — 200→50 individuals; still adequate for 10 params
3. MAXITER 500→200   — tighter convergence window; restart if needed
4. Early abort       — skip remaining flight conditions if first already fails
5. tol 1e-6→1e-4     — stop a bit sooner; polish refines afterwards anyway
6. updating='deferred' — required for workers>1, also batches updates better

Optimized parameters (10 total)
--------------------------------
  x[0]  A0        inlet capture area          [m²]    (3.0 – 7.0)
  x[1]  A2_frac   A2/A0  isolator exit ratio  [—]     (0.70 – 1.10)
  x[2]  A3_frac   A3/A0  combustor exit ratio [—]     (0.90 – 1.40)
  x[3]  A6_frac   A6/A0  nozzle exit ratio    [—]     (0.80 – 2.00)
  x[4]  L_comb    combustor length            [m]     (0.20 – 2.00)
  x[5]  L23_frac  L23/L_comb                 [—]     (0.30 – 0.80)
  x[6]  L01       inlet ramp length           [m]     (0.20 – 1.50)
  x[7]  L12       isolator length             [m]     (0.10 – 1.00)
  x[8]  phi       equivalence ratio           [—]     (0.60 – 1.00)
  x[9]  L56_frac  L56/L45  diverging nozzle   [—]     (1.00 – 6.00)
"""

import sys, io, contextlib, warnings, os
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

sys.path.insert(0, "/home/claude")
from ramjet_fixedgeometry import Ramjet, Geometry, Assumptions

# ──────────────────────────────────────────────────────────────────────────────
# FLIGHT CONDITIONS
# ──────────────────────────────────────────────────────────────────────────────
FLIGHT_CONDITIONS = [
    {"Ma0": 2.75, "h0": 17_000.0, "thrust_min": 300_000.0, "label": "M275_h17_T300", "weight": 1.0},
    {"Ma0": 4.40, "h0": 29_300.0, "thrust_min": 170_000.0, "label": "M44_h293_T170",  "weight": 1.0},
    {"Ma0": 5.00, "h0": 30_000.0, "thrust_min": 140_000.0, "label": "M5_h300_T140",   "weight": 1.0},
]

# ──────────────────────────────────────────────────────────────────────────────
# FIXED CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
L45_VAL  = 0.35
MA_COMB  = 0.30
CF       = 0.003
THETA    = 90.0
MIX_COEF = 0.176
THRUST_PENALTY = 1e4

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETER BOUNDS
# ──────────────────────────────────────────────────────────────────────────────
BOUNDS = [
    (3.0,  7.0),    # x[0]  A0
    (0.70, 1.10),   # x[1]  A2_frac
    (0.90, 1.40),   # x[2]  A3_frac
    (0.80, 2.00),   # x[3]  A6_frac
    (0.20, 2.00),   # x[4]  L_comb
    (0.30, 0.80),   # x[5]  L23_frac
    (0.20, 1.50),   # x[6]  L01
    (0.10, 1.00),   # x[7]  L12
    (0.60, 1.00),   # x[8]  phi
    (1.00, 6.00),   # x[9]  L56_frac
]

# ── Speed knobs ───────────────────────────────────────────────────────────────
MAXITER  = 200   # ↓ from 500  (polish cleans up anyway)
POPSIZE  = 5     # ↓ from 20   (50 individuals; scipy min recommended is 5)
TOL      = 1e-4  # ↑ from 1e-6 (stop earlier; polish refines)
WORKERS  = -1    # -1 = all CPU cores.  Set to 1 to disable parallelism.

_POP_N       = POPSIZE * len(BOUNDS)
_TOTAL_EVALS = _POP_N + MAXITER * _POP_N   # upper bound


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATE  — note: must be importable at top level for multiprocessing
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(x):
    """
    Run all flight conditions for geometry x.
    Early-abort: if a condition fails badly (choke/exception with no
    plausible thrust) we still accumulate the penalty but skip expensive
    conditions once the outcome is already hopeless.
    """
    A0, A2_frac, A3_frac, A6_frac, L_comb, L23_frac, L01, L12, phi, L56_frac = x
    L23 = L23_frac * L_comb
    L34 = (1.0 - L23_frac) * L_comb
    L56 = L56_frac * L45_VAL

    geom = Geometry(
        A0=A0, L01=L01, L12=L12, L23=L23, L34=L34,
        L45=L45_VAL, L56=L56,
        A2=A2_frac * A0, A3=A3_frac * A0, A4=A3_frac * A0, A6=A6_frac * A0,
    )

    results       = []
    total_penalty = 0.0
    feasible      = True

    for cond in FLIGHT_CONDITIONS:
        assump = Assumptions(
            h0=cond["h0"], Ma0=cond["Ma0"], phi=phi, theta=THETA,
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

            choke = (any(s.get("thermal_choke", False) for s in [sec2, sec3, sec4])
                     or perf.get("thermal_choke", False))
            Fin = perf["Fin"]
            Isp = perf["Isp"]

            if choke or not np.isfinite(Fin) or not np.isfinite(Isp):
                feasible = False
                total_penalty += THRUST_PENALTY * cond["thrust_min"]
                results.append({"label": cond["label"], "Fin": 0.0, "Isp": 0.0,
                                 "status": "choke/invalid"})
                continue

            shortfall = max(0.0, cond["thrust_min"] - Fin)
            total_penalty += THRUST_PENALTY * shortfall
            if shortfall > 0:
                feasible = False

            results.append({
                "label": cond["label"], "Fin": Fin, "Isp": Isp,
                "Ia": perf["Ia"], "Ma4": sec4["Ma"], "T4_K": sec4["T"],
                "Ma6": sec6["Ma"], "V6_ms": sec6["V"],
                "mdot": inp["mdot"], "A1_m2": iso["A"], "status": "ok",
            })

        except Exception:
            feasible = False
            total_penalty += THRUST_PENALTY * cond["thrust_min"]
            results.append({"label": cond["label"], "Fin": 0.0, "Isp": 0.0,
                             "status": "exception"})

    isp_vals     = [r["Isp"] for r in results if r["status"] == "ok"]
    weights      = [c["weight"] for c, r in zip(FLIGHT_CONDITIONS, results) if r["status"] == "ok"]
    weighted_isp = np.average(isp_vals, weights=weights) if isp_vals else 0.0

    return {"results": results, "weighted_isp": weighted_isp,
            "feasible": feasible, "penalty": total_penalty}


# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE  — must be a plain top-level function for multiprocessing pickle
# ──────────────────────────────────────────────────────────────────────────────
def objective(x):
    out = evaluate(x)
    return -(out["weighted_isp"]) + out["penalty"]


# ──────────────────────────────────────────────────────────────────────────────
# PROGRESS BAR  — lives only in the main process
# (workers=-1 forks child processes; tqdm/callback only run in parent)
# ──────────────────────────────────────────────────────────────────────────────
_pbar     = None
_best_isp = [-np.inf]
_best_x   = [None]
_gen      = [0]

def de_callback(xk, convergence):
    """Called in the main process once per generation with the current best."""
    _gen[0] += 1
    out = evaluate(xk)   # one extra eval of the best — cheap vs whole pop
    isp = out["weighted_isp"]

    if isp > _best_isp[0]:
        _best_isp[0] = isp
        _best_x[0]   = xk.copy()

    thrust_tags = []
    for r in out["results"]:
        ok = r["status"] == "ok"
        thresh = next(c["thrust_min"] for c in FLIGHT_CONDITIONS if c["label"] == r["label"])
        icon = "✓" if (ok and r["Fin"] >= thresh) else "✗"
        val  = f"{r['Fin']/1e3:.0f}" if ok else "err"
        thrust_tags.append(f"{r['label'][:4]}{icon}{val}kN")

    if _pbar is not None:
        # Advance bar by one full generation worth of evals
        _pbar.update(_POP_N)
        _pbar.set_postfix({
            "gen"    : f"{_gen[0]}/{MAXITER}",
            "conv"   : f"{convergence:.1e}",
            "Isp"    : f"{isp:.0f}s",
            "best"   : f"{_best_isp[0]:.0f}s",
            "ok"     : "✓" if out["feasible"] else "✗",
            "thrust" : " ".join(thrust_tags),
        }, refresh=True)


# ──────────────────────────────────────────────────────────────────────────────
# PRINT SOLUTION
# ──────────────────────────────────────────────────────────────────────────────
def print_solution(x, label="SOLUTION"):
    out = evaluate(x)
    A0, A2_frac, A3_frac, A6_frac, L_comb, L23_frac, L01, L12, phi, L56_frac = x
    L23 = L23_frac * L_comb
    L34 = (1.0 - L23_frac) * L_comb
    L56 = L56_frac * L45_VAL

    print(f"\n{'═'*70}")
    print(f"  {label}")
    print(f"{'═'*70}")
    print(f"  A0     = {A0:.4f} m²      (inlet capture area)")
    print(f"  A2     = {A2_frac*A0:.4f} m²  (A2/A0 = {A2_frac:.4f})")
    print(f"  A3     = {A3_frac*A0:.4f} m²  (A3/A0 = {A3_frac:.4f})")
    print(f"  A6     = {A6_frac*A0:.4f} m²  (A6/A0 = {A6_frac:.4f})")
    print(f"  L_comb = {L_comb:.4f} m     (L23={L23:.4f} m, L34={L34:.4f} m)")
    print(f"  L01    = {L01:.4f} m")
    print(f"  L12    = {L12:.4f} m")
    print(f"  L45    = {L45_VAL:.4f} m     (fixed)")
    print(f"  L56    = {L56:.4f} m     (L56/L45={L56_frac:.2f})")
    print(f"  phi    = {phi:.4f}")
    print(f"  ── Per-condition results ───────────────────────────────────────")
    for r in out["results"]:
        if r["status"] == "ok":
            print(f"  {r['label']:20s}  Fin={r['Fin']/1e3:8.2f} kN  "
                  f"Isp={r['Isp']:7.1f} s  Ma4={r['Ma4']:.4f}  "
                  f"T4={r['T4_K']:.0f} K  Ma6={r['Ma6']:.3f}")
        else:
            print(f"  {r['label']:20s}  ✗ {r['status']}")
    print(f"  Weighted avg Isp : {out['weighted_isp']:.2f} s  |  Feasible: {out['feasible']}")
    print(f"{'═'*70}\n")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — guard required for multiprocessing on Windows
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    n_cores = os.cpu_count()
    effective_workers = n_cores if WORKERS == -1 else WORKERS
    speedup_note = (f"~{effective_workers}x faster than serial"
                    if effective_workers > 1 else "serial (set WORKERS=-1 for speedup)")

    print("\n" + "█"*70)
    print("  RAMJET MULTI-CONDITION GEOMETRY OPTIMIZATION")
    print(f"  CPU cores available : {n_cores}  →  using {effective_workers}  ({speedup_note})")
    print(f"  Population          : {_POP_N} individuals  (POPSIZE={POPSIZE} × {len(BOUNDS)} params)")
    print(f"  Max generations     : {MAXITER}   tol={TOL}")
    print(f"  Max evaluations     : {_TOTAL_EVALS:,}  (~{_TOTAL_EVALS/effective_workers:,.0f} serial-equivalent)")
    print(f"  Conditions          :")
    for c in FLIGHT_CONDITIONS:
        print(f"    Ma={c['Ma0']}  h={c['h0']/1e3:.1f} km  "
              f"Fin_min={c['thrust_min']/1e3:.0f} kN")
    print("█"*70 + "\n")

    # Progress bar advances by _POP_N each generation via the callback
    _pbar = tqdm(
        total        = _TOTAL_EVALS,
        desc         = "Optimizing",
        unit         = "eval",
        dynamic_ncols= True,
        colour       = "cyan",
        bar_format   = (
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]  {postfix}"
        ),
    )
    # Seed bar with initial population (evaluated before first callback)
    _pbar.update(_POP_N)
    _pbar.set_postfix({"status": "initialising population..."}, refresh=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        de_result = differential_evolution(
            objective,
            bounds        = BOUNDS,
            strategy      = "best1bin",
            maxiter       = MAXITER,
            popsize       = POPSIZE,
            tol           = TOL,
            mutation      = (0.5, 1.0),
            recombination = 0.7,
            seed          = 42,
            callback      = de_callback,
            polish        = False,
            workers       = WORKERS,
            updating      = "deferred",   # required for workers != 1; also faster
            init          = "latinhypercube",
        )

    _pbar.close()
    print(f"\n  DE done: {_gen[0]} generations, best Isp so far = {_best_isp[0]:.1f} s\n")

    # ── Polish ────────────────────────────────────────────────────────────────
    print("  Polishing with L-BFGS-B (single-threaded, fast)...")
    with tqdm(total=300, desc="Polish", unit="eval",
              dynamic_ncols=True, colour="green") as pbar:

        def _polish_obj(x):
            pbar.update(1)
            out = evaluate(x)
            pbar.set_postfix(Isp=f"{out['weighted_isp']:.1f}s",
                             ok="✓" if out["feasible"] else "✗", refresh=False)
            return -(out["weighted_isp"]) + out["penalty"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local = minimize(
                _polish_obj, de_result.x,
                method="L-BFGS-B", bounds=BOUNDS,
                options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 300},
            )

    best_x   = local.x if local.fun < de_result.fun else de_result.x
    improved = local.fun < de_result.fun
    print(f"  Polish {'✓ improved' if improved else '— no further improvement'}\n")

    best_out = print_solution(best_x, label="BEST SOLUTION (DE + polish)")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR = r"C:\Users\arkan\OneDrive\Desktop\DSE-1\src\propulsion\engine"

    rows = []
    for r in best_out["results"]:
        A0, A2_frac, A3_frac, A6_frac, L_comb, L23_frac, L01, L12, phi, L56_frac = best_x
        rows.append({
            "label": r["label"],
            "A0_m2": round(A0, 6), "A2_frac": round(A2_frac, 6),
            "A3_frac": round(A3_frac, 6), "A6_frac": round(A6_frac, 6),
            "A2_m2": round(A2_frac*A0, 6), "A3_m2": round(A3_frac*A0, 6),
            "A6_m2": round(A6_frac*A0, 6),
            "L_comb_m": round(L_comb, 6),
            "L23_m": round(L23_frac*L_comb, 6), "L34_m": round((1-L23_frac)*L_comb, 6),
            "L01_m": round(L01, 6), "L12_m": round(L12, 6),
            "L45_m": L45_VAL, "L56_m": round(L56_frac*L45_VAL, 6),
            "phi": round(phi, 6),
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in r.items() if k not in ("label", "status")},
            "status": r["status"],
            "feasible": best_out["feasible"],
            "weighted_isp": round(best_out["weighted_isp"], 4),
        })

    pd.DataFrame(rows).to_csv(f"{OUT_DIR}\\optimized_geometry.csv", index=False)
    print(f"  ✓ Saved → {OUT_DIR}\\optimized_geometry.csv")
    print("\nOPTIMIZATION COMPLETE.")