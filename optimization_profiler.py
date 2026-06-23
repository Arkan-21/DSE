"""
optimization_profiler.py

Wraps geometry_generator's optimization pipeline with per-step timing.
Patches the six expensive call sites, runs the optimizer, then prints and
saves a breakdown of where time was spent.

Usage:
    python optimization_profiler.py
"""

import time
import functools
import collections
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# 1. Accumulator
# ---------------------------------------------------------------------------

class StepTimer:
    """Accumulates wall-clock time and call counts for named steps."""

    def __init__(self):
        self._calls  = collections.defaultdict(int)
        self._totals = collections.defaultdict(float)
        self._per_call = collections.defaultdict(list)  # individual sample times

    def record(self, step: str, elapsed: float):
        self._calls[step]  += 1
        self._totals[step] += elapsed
        self._per_call[step].append(elapsed)

    # ---- accessors ---------------------------------------------------------
    def calls(self, step):  return self._calls[step]
    def total(self, step):  return self._totals[step]
    def mean(self,  step):
        samples = self._per_call[step]
        return sum(samples) / len(samples) if samples else 0.0

    def all_steps(self):
        return list(self._totals.keys())

    def grand_total(self):
        return sum(self._totals.values())

    # ---- reporting ---------------------------------------------------------
    def report(self):
        gt = self.grand_total()
        header = f"{'Step':<38}  {'Calls':>6}  {'Total (s)':>10}  {'Mean (s)':>10}  {'Share':>7}"
        sep    = "─" * len(header)
        lines  = [sep, header, sep]
        for step in self._totals:
            pct = 100 * self._totals[step] / gt if gt else 0
            lines.append(
                f"  {step:<36}  {self._calls[step]:>6}  "
                f"{self._totals[step]:>10.2f}  "
                f"{self.mean(step):>10.3f}  "
                f"{pct:>6.1f}%"
            )
        lines.append(sep)
        lines.append(f"  {'TOTAL (instrumented)':<36}  {'':>6}  {gt:>10.2f}")
        lines.append(sep)
        return "\n".join(lines)

    def save_csv(self, path="profiler_results.csv"):
        import csv
        gt = self.grand_total()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "calls", "total_s", "mean_s", "share_pct"])
            for step in self._totals:
                pct = 100 * self._totals[step] / gt if gt else 0
                w.writerow([step, self._calls[step],
                             round(self._totals[step], 4),
                             round(self.mean(step), 4),
                             round(pct, 2)])
        print(f"  CSV saved → {path}")

    def plot(self, path="profiler_results.png"):
        steps  = list(self._totals.keys())
        totals = [self._totals[s] for s in steps]
        gt     = sum(totals)
        labels = [f"{s}\n{100*t/gt:.1f}%" for s, t in zip(steps, totals)]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- pie chart ---
        wedge_props = {"linewidth": 0.5, "edgecolor": "white"}
        axes[0].pie(totals, labels=labels, wedgeprops=wedge_props,
                    autopct=None, startangle=140)
        axes[0].set_title("Time share per step")

        # --- horizontal bar chart ---
        y_pos = range(len(steps))
        bars  = axes[1].barh(list(y_pos), totals, color="steelblue", edgecolor="white")
        axes[1].set_yticks(list(y_pos))
        axes[1].set_yticklabels(steps, fontsize=9)
        axes[1].set_xlabel("Total time (s)")
        axes[1].set_title("Absolute time per step")
        for bar, val in zip(bars, totals):
            axes[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                         f"{val:.1f}s", va="center", fontsize=8)
        axes[1].invert_yaxis()

        plt.suptitle("Optimization step timing breakdown", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Plot saved  → {path}")


TIMER = StepTimer()


# ---------------------------------------------------------------------------
# 2. Decorator factory
# ---------------------------------------------------------------------------

def timed(step_name: str, timer: StepTimer = TIMER):
    """Return a decorator that records wall-clock time of each call."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            timer.record(step_name, time.perf_counter() - t0)
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# 3. Import geometry_generator and patch it
# ---------------------------------------------------------------------------
import geometry_generator as gg
import understand_stl as ustl

# -- patch geometry_generator module-level functions --
_orig_generate_geometry = gg.generate_geometry
_orig_mesh_volume        = ustl.mesh_volume
_orig_shock              = gg.shock_constraint
_orig_structures         = gg.structures_feasibility_constraint
_orig_exact_vol          = gg.exact_volume_constraint
_orig_area               = gg.area_constraint

# Retrieve the hypersonic aero module that geometry_generator loaded
_hf = gg._hf

_orig_load_geometry     = _hf.load_geometry
_orig_run_case          = _hf.run_case_total_coef


@timed("generate_geometry")
def _timed_generate_geometry(*args, **kwargs):
    return _orig_generate_geometry(*args, **kwargs)


@timed("load_geometry (STL → mesh)")
def _timed_load_geometry(*args, **kwargs):
    return _orig_load_geometry(*args, **kwargs)


@timed("run_case_total_coef (aero solve)")
def _timed_run_case(*args, **kwargs):
    return _orig_run_case(*args, **kwargs)


@timed("mesh_volume")
def _timed_mesh_volume(*args, **kwargs):
    return _orig_mesh_volume(*args, **kwargs)


@timed("shock_constraint")
def _timed_shock(*args, **kwargs):
    return _orig_shock(*args, **kwargs)


@timed("structures_constraint")
def _timed_structures(*args, **kwargs):
    return _orig_structures(*args, **kwargs)


@timed("exact_volume_constraint")
def _timed_exact_vol(*args, **kwargs):
    return _orig_exact_vol(*args, **kwargs)


@timed("area_constraint")
def _timed_area(*args, **kwargs):
    return _orig_area(*args, **kwargs)


# Inject patches
gg.generate_geometry         = _timed_generate_geometry
gg.mesh_volume               = _timed_mesh_volume        # referenced directly in gg
ustl.mesh_volume             = _timed_mesh_volume        # referenced as imported name

# Patch the aero module object that gg holds a reference to
_hf.load_geometry            = _timed_load_geometry
_hf.run_case_total_coef      = _timed_run_case

gg.shock_constraint          = _timed_shock
gg.structures_feasibility_constraint = _timed_structures
gg.exact_volume_constraint   = _timed_exact_vol
gg.area_constraint           = _timed_area


# ---------------------------------------------------------------------------
# 4. Instrumented run_optimization
#    Identical to gg.run_optimization but uses the patched functions and
#    records per-iteration wall times for a breakdown across SLSQP phases.
# ---------------------------------------------------------------------------

def run_instrumented_optimization():
    import scipy.optimize as opt
    from geometry_generator import (
        OPTIMIZE_KEYS, BASELINE_PARAMS, RELATIVE_STEPS, BOUNDS,
        MACH, ALPHA,
        denormailize_params,
        save_history, save_optimal_geometry, process_history, text_summary,
    )

    history = []
    iter_wall_times = []   # wall time of the whole objective call, per call
    t_start = time.perf_counter()

    def objective(x):
        t_obj = time.perf_counter()
        intermediate_res_folder = "intermediate_results_profiler"
        os.makedirs(intermediate_res_folder, exist_ok=True)

        stl_path      = f"temp_{len(history)+1}.stl"
        full_stl_path = os.path.join(intermediate_res_folder, stl_path)
        params = denormailize_params(x)

        _timed_generate_geometry(**params, output_stl=full_stl_path)

        geom = _timed_load_geometry(full_stl_path)
        n, area = geom[0], geom[1]
        sref = float(np.sum(area * np.abs(n[:, 2])) / 2.0)

        CL, CD = _timed_run_case(geom, MACH, ALPHA, sref)
        ld = CL / CD if CD != 0 else 0.0

        span_c   = _timed_shock(**params)
        struct_c = _timed_structures(**params)
        vol      = _timed_mesh_volume(full_stl_path)

        history.append({
            "iter": len(history) + 1,
            "params": params.copy(),
            "CL": CL, "CD": CD, "LD": ld,
            "volume": vol,
            "span_constraint": span_c,
            "structures_constraint": struct_c,
            "sref": sref,
        })
        elapsed_obj = time.perf_counter() - t_obj
        iter_wall_times.append(elapsed_obj)
        print(f"  [profiler] iter {len(history):>4}  L/D={ld:.4f}  wall={elapsed_obj:.2f}s")
        return -ld

    # ------------------------------------------------------------------
    # Initial geometry (mirrors geometry_generator.run_optimization)
    # ------------------------------------------------------------------
    x0 = np.ones(len(OPTIMIZE_KEYS))
    params0 = denormailize_params(x0)

    print("\n[profiler] Generating initial geometry ...")
    _timed_generate_geometry(**params0, output_stl="initial_geometry_profiler.stl")
    geom0 = _timed_load_geometry("initial_geometry_profiler.stl")
    target_volume = _timed_mesh_volume("initial_geometry_profiler.stl")
    n0, area0 = geom0[0], geom0[1]
    init_sref = float(np.sum(area0 * np.abs(n0[:, 2])) / 2)
    CL0, CD0  = _timed_run_case(geom0, MACH, ALPHA, init_sref)
    ld0 = CL0 / CD0 if CD0 != 0 else 0.0
    print(f"[profiler] Initial L/D = {ld0:.4f}, Volume = {target_volume:.2f} m³, Sref = {init_sref:.2f} m²")

    # warm-up objective call
    objective(x0)
    input("\n[profiler] Press Enter to start the instrumented optimisation ...")

    # ------------------------------------------------------------------
    # Constraint jacobians (finite-difference, expensive)
    # ------------------------------------------------------------------
    def _volume_jac(constraint_fn, x, eps=0.01):
        f0 = constraint_fn(x)
        grad = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            xp    = x.copy()
            xp[i] += eps
            grad[i] = (constraint_fn(xp) - f0) / eps
        return grad

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: _timed_shock(**denormailize_params(x)),
            "jac": lambda x: _volume_jac(
                lambda p: _timed_shock(**denormailize_params(p)), x, eps=0.005),
        },
        {
            "type": "eq",
            "fun": lambda x: _timed_exact_vol(denormailize_params(x), target_volume),
            "jac": lambda x: _volume_jac(
                lambda p: _timed_exact_vol(denormailize_params(p), target_volume), x),
        },
        {
            "type": "ineq",
            "fun": lambda x: _timed_structures(**denormailize_params(x)),
            "jac": lambda x: _volume_jac(
                lambda p: _timed_structures(**denormailize_params(p)), x, eps=0.005),
        },
        {
            "type": "ineq",
            "fun": lambda x: _timed_area(denormailize_params(x), init_sref),
            "jac": lambda x: _volume_jac(
                lambda p: _timed_area(denormailize_params(p), init_sref), x),
        },
    ]

    relative_steps = np.array([RELATIVE_STEPS.get(p, 0.01) for p in OPTIMIZE_KEYS])

    print("[profiler] Starting optimisation ...")
    result = opt.minimize(
        objective, x0,
        method="SLSQP",
        jac="2-point",
        constraints=constraints,
        bounds=BOUNDS,
        options={"maxiter": 1, "disp": True, "finite_diff_rel_step": relative_steps},
    )

    t_end = time.perf_counter()
    wall_total = t_end - t_start

    # ------------------------------------------------------------------
    # Save optimisation artefacts (mirrors geometry_generator)
    # ------------------------------------------------------------------
    save_history(history, filename="optimisation_history_profiler.xlsx")
    save_optimal_geometry(
        history,
        best_vsp="optimal_geometry_profiler.vsp3",
        best_stl="optimal_geometry_profiler.stl",
        target_volume=target_volume,
    )
    process_history(history, target_volume=target_volume,
                    init_sref=init_sref, filename="optimisation_history_profiler.png")
    text_summary(history)

    # ------------------------------------------------------------------
    # Print & save timing report
    # ------------------------------------------------------------------
    print(f"\n[profiler] Total wall time: {wall_total:.1f} s  "
          f"({len(history)} objective calls, "
          f"{len(history)/wall_total:.2f} obj/s)")

    print("\n" + "═" * 70)
    print("  STEP-BY-STEP TIMING BREAKDOWN")
    print("═" * 70)
    print(TIMER.report())

    # Per-iteration wall time summary
    if iter_wall_times:
        print(f"\nObjective call wall times (s):")
        print(f"  min   = {min(iter_wall_times):.2f}")
        print(f"  mean  = {sum(iter_wall_times)/len(iter_wall_times):.2f}")
        print(f"  max   = {max(iter_wall_times):.2f}")
        print(f"  total = {sum(iter_wall_times):.2f}  ({100*sum(iter_wall_times)/wall_total:.1f}% of wall time)")

    TIMER.save_csv("profiler_results.csv")
    TIMER.plot("profiler_results.png")

    # ------------------------------------------------------------------
    # Per-iteration timeline plot
    # ------------------------------------------------------------------
    if iter_wall_times:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(1, len(iter_wall_times) + 1), iter_wall_times,
               color="steelblue", edgecolor="white", linewidth=0.4)
        ax.axhline(sum(iter_wall_times) / len(iter_wall_times),
                   color="crimson", linestyle="--", linewidth=1.2, label="mean")
        ax.set_xlabel("Objective call #")
        ax.set_ylabel("Wall time (s)")
        ax.set_title("Per-iteration objective wall time")
        ax.legend()
        plt.tight_layout()
        plt.savefig("profiler_iterations.png", dpi=150)
        plt.close()
        print("  Per-iter plot → profiler_iterations.png")

    return result, history, TIMER


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_instrumented_optimization()
