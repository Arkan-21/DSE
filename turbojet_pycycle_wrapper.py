import os
import contextlib
import warnings
from functools import lru_cache

import numpy as np
import openmdao.api as om
from openmdao.utils.om_warnings import SolverWarning

from example_cycles.afterburning_turbojet import MPABTurbojet


# ============================================================
# Hide OpenMDAO / NumPy warning spam
# ============================================================

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=SolverWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


# ============================================================
# Unit conversions
# ============================================================

LBF_TO_KN = 0.0044482216152605
M_TO_FT = 3.280839895


# ============================================================
# Engine sizing parameters
# ============================================================

# Original pyCycle example design thrust was 11,800 lbf.
# We scale the engine to 90,000 lbf per engine.
BASE_DESIGN_THRUST_LBF = 11_800.0
DESIGN_THRUST_LBF = 600_000.0

SCALE = DESIGN_THRUST_LBF / BASE_DESIGN_THRUST_LBF

DESIGN_T4_DEGR = 2550.0
DESIGN_COMP_PR = 18.0
DESIGN_COMP_EFF = 0.88
DESIGN_TURB_EFF = 0.90


# ============================================================
# Build pyCycle model ONCE
# ============================================================

def setup_ab_turbojet():
    """
    Builds the pyCycle afterburning turbojet model once.

    Do NOT call this inside loops.
    """

    prob = om.Problem()
    prob.model = engine = MPABTurbojet()

    prob.setup()

    # ------------------------------------------------------------
    # DESIGN point setup
    # ------------------------------------------------------------

    prob.set_val("DESIGN.fc.alt", 0.0, units="ft")
    prob.set_val("DESIGN.fc.MN", 0.000001)

    prob.set_val("DESIGN.balance.rhs:W", DESIGN_THRUST_LBF, units="lbf")
    prob.set_val("DESIGN.balance.rhs:FAR", DESIGN_T4_DEGR, units="degR")

    prob.set_val("DESIGN.comp.PR", DESIGN_COMP_PR)
    prob.set_val("DESIGN.comp.eff", DESIGN_COMP_EFF)
    prob.set_val("DESIGN.turb.eff", DESIGN_TURB_EFF)

    # DESIGN initial guesses, scaled from original example
    prob["DESIGN.balance.FAR"] = 0.01755078
    prob["DESIGN.balance.W"] = 168.00454616 * SCALE
    prob["DESIGN.balance.turb_PR"] = 4.46131867
    prob["DESIGN.fc.balance.Pt"] = 14.6959
    prob["DESIGN.fc.balance.Tt"] = 518.67

    # ------------------------------------------------------------
    # OFF-DESIGN initial guesses from original pyCycle example
    # Mass-flow guesses are scaled with engine size.
    # ------------------------------------------------------------

    W_guess = [
        168.0,
        225.0,
        168.005,
        225.917,
        166.074,
        141.2,
        61.70780608,
        145.635,
        71.53855266,
        33.347,
    ]

    FAR_guess = [
        0.01755,
        0.01,
        0.01755,
        0.01629,
        0.0168,
        0.01689,
        0.01872827,
        0.016083,
        0.01619524,
        0.015170,
    ]

    Nmech_guess = [
        8070.0,
        8000.0,
        8070.0,
        8288.85,
        8197.39,
        8181.03,
        8902.24164717,
        8326.586,
        8306.00268554,
        8467.2404,
    ]

    PR_guess = [
        4.4613,
        5.0,
        4.4613,
        4.8185,
        4.669,
        4.6425,
        4.42779036,
        4.8803,
        4.84652723,
        5.11582,
    ]

    for i, pt in enumerate(engine.od_pts):
        prob[pt + ".balance.W"] = W_guess[i] * SCALE
        prob[pt + ".balance.FAR"] = FAR_guess[i]
        prob[pt + ".balance.Nmech"] = Nmech_guess[i]
        prob[pt + ".turb.PR"] = PR_guess[i]

    prob.set_solver_print(level=-1)

    return prob, engine


# ============================================================
# Global model storage
# ============================================================

_TURBO_PROB = None
_TURBO_ENGINE = None


def get_turbo_model():
    """
    Lazy setup.

    First time you call the turbojet, it builds the model.
    After that, it reuses the same model.
    """

    global _TURBO_PROB, _TURBO_ENGINE

    if _TURBO_PROB is None:
        print("Building pyCycle afterburning turbojet model...")
        print(f"Design thrust per engine = {DESIGN_THRUST_LBF:,.0f} lbf")
        print(f"Scale factor = {SCALE:.3f}x original pyCycle example")
        _TURBO_PROB, _TURBO_ENGINE = setup_ab_turbojet()
        print("Turbojet model ready.")

    return _TURBO_PROB, _TURBO_ENGINE


def reset_turbo_model():
    """
    Clears the stored model and cache.
    Useful if you change DESIGN_THRUST_LBF and rerun in the same Python session.
    """

    global _TURBO_PROB, _TURBO_ENGINE

    _TURBO_PROB = None
    _TURBO_ENGINE = None
    turbojet_thrust_kN_cached.cache_clear()


# ============================================================
# Cached thrust function
# ============================================================

@lru_cache(maxsize=300)
def turbojet_thrust_kN_cached(mach_rounded, altitude_m_rounded):
    """
    Cached pyCycle turbojet thrust.

    Inputs:
        mach_rounded:
            Rounded Mach number.

        altitude_m_rounded:
            Rounded altitude in meters.

    Returns:
        Thrust in kN for ONE engine.
    """

    mach = float(mach_rounded)
    altitude_m = float(altitude_m_rounded)
    altitude_ft = altitude_m * M_TO_FT

    prob, engine = get_turbo_model()

    # Use last off-design point as our variable flight point
    point = engine.od_pts[-1]

    try:
        prob.set_val(point + ".fc.MN", mach)
        prob.set_val(point + ".fc.alt", altitude_ft, units="ft")

        # Hide pyCycle/OpenMDAO solver spam
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            prob.run_model()

        thrust_lbf = float(prob.get_val(point + ".perf.Fn", units="lbf")[0])
        thrust_kN = thrust_lbf * LBF_TO_KN

        if not np.isfinite(thrust_kN):
            return np.nan

        return thrust_kN

    except Exception as e:
        print()
        print("Turbojet pyCycle solve failed.")
        print(f"Mach       = {mach:.3f}")
        print(f"Altitude   = {altitude_m:.1f} m")
        print(f"Altitude   = {altitude_ft:.1f} ft")
        print(f"Error      = {e}")
        print()

        return np.nan


def turbojet_thrust_kN(
    mach,
    altitude_m,
    n_engines=2,
    clamp_negative=True,
):
    """
    Main function you call from your aircraft code.

    Inputs:
        mach:
            Flight Mach number.

        altitude_m:
            Altitude in meters.

        n_engines:
            Number of turbojet engines.

        clamp_negative:
            If True, negative thrust is replaced with 0.

    Returns:
        Total turbojet thrust in kN.
    """

    mach_rounded = round(float(mach), 2)
    altitude_m_rounded = round(float(altitude_m), -1)

    one_engine_kN = turbojet_thrust_kN_cached(
        mach_rounded,
        altitude_m_rounded,
    )

    if not np.isfinite(one_engine_kN):
        return np.nan

    if clamp_negative:
        one_engine_kN = max(one_engine_kN, 0.0)

    return n_engines * one_engine_kN


# ============================================================
# Optional: generate lookup table
# ============================================================

def generate_turbojet_table(
    mach_values,
    altitude_values_m,
    n_engines=2,
    save=True,
):
    """
    Generates a thrust table over Mach and altitude.

    This is much faster for plotting later because you only run pyCycle
    a small number of times, then save the results.
    """

    thrust_table = np.zeros((len(altitude_values_m), len(mach_values)))

    print()
    print("Generating turbojet thrust table")
    print("--------------------------------")

    for i, h in enumerate(altitude_values_m):
        for j, M in enumerate(mach_values):
            thrust_table[i, j] = turbojet_thrust_kN(
                mach=M,
                altitude_m=h,
                n_engines=n_engines,
            )

            print(
                f"M={M:.2f}, h={h / 1000:.1f} km, "
                f"T={thrust_table[i, j]:.2f} kN"
            )

    if save:
        np.save("turbojet_thrust_table.npy", thrust_table)
        np.save("turbojet_altitudes_m.npy", altitude_values_m)
        np.save("turbojet_machs.npy", mach_values)

        print()
        print("Saved:")
        print("  turbojet_thrust_table.npy")
        print("  turbojet_altitudes_m.npy")
        print("  turbojet_machs.npy")

    return thrust_table


# ============================================================
# Optional quick test
# ============================================================

if __name__ == "__main__":

    print()
    print("Testing pyCycle afterburning turbojet wrapper")
    print("============================================")
    print(f"Design thrust per engine: {DESIGN_THRUST_LBF:,.0f} lbf")
    print(f"Scale factor: {SCALE:.3f}x")

    # Original-ish safe point first
    # 70,000 ft = 21,336 m
    T_safe = turbojet_thrust_kN(
        mach=1.8,
        altitude_m=21336.0,
        n_engines=2,
    )

    print()
    print(f"Safe test: M=1.8, h=21.336 km, thrust = {T_safe:.3f} kN")

    # Gradual Mach sweep at 70,000 ft
    print()
    print("Mach sweep at 70,000 ft")
    print("-----------------------")

    for M in [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]:
        T = turbojet_thrust_kN(
            mach=M,
            altitude_m=21336.0,
            n_engines=2,
        )

        print(f"M={M:.2f}, thrust={T:.3f} kN")

    # Try Mach 3 last, but do not trust it blindly
    print()
    print("Trying Mach 3")
    print("-------------")

    T_m3 = turbojet_thrust_kN(
        mach=3.0,
        altitude_m=21336.0,
        n_engines=2,
    )

    print(f"M=3.0, h=21.336 km, thrust = {T_m3:.3f} kN")