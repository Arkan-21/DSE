"""
turbojet_mach_altitude_plot.py

Paper-style turbojet off-design calculator with:
- Design point set to M0 = 2.0 at altitude = 20 km
- ISA atmosphere calculator for T0 and P0
- Thrust plotted versus Mach number from 0 to 3.5
- Curves for altitudes from 0 to 25 km

The cycle method follows the uploaded paper-matching interim-value code:
fixed turbine throat area, fixed nozzle area, constant component efficiencies,
and N/N_DP used as the throttle/spool-speed proxy.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Gas:
    k: float
    cp: float
    R: float


@dataclass(frozen=True)
class Params:
    # Gas data
    air: Gas = Gas(k=1.4, cp=1005.0, R=287.0)
    gas: Gas = Gas(k=1.33, cp=1170.0, R=290.0)
    cpB: float = 1200.0
    FHV: float = 43e6

    # Losses / efficiencies
    sigma_inlet: float = 0.97
    sigma_burner: float = 0.98
    sigma_ab_off: float = 0.975
    sigma_nozzle: float = 0.96
    eta_compressor: float = 0.83
    eta_turbine: float = 0.90
    eta_burner: float = 0.98
    eta_mech: float = 0.99

    # New design point for the high-speed turbojet case
    M0_DP: float = 2.0
    altitude_DP_m: float = 20_000.0

    # Keep the paper's reference engine choices unless you change them here
    m0_DP: float = 250.0        # kg/s
    CPR_DP: float = 8.0        # compressor pressure ratio at design point
    Tt4_DP: float = 1800.0     # K, turbine inlet temperature at design point


P = Params()


def isa_atmosphere(altitude_m: float) -> Tuple[float, float]:
    """
    International Standard Atmosphere, simplified to 0-25 km.

    Returns:
        T0 [K], P0 [Pa]
    """
    h = float(altitude_m)
    T_sl = 288.15
    P_sl = 101325.0
    g0 = 9.80665
    R = 287.05287
    L = -0.0065

    if h <= 11_000.0:
        T = T_sl + L * h
        P = P_sl * (T / T_sl) ** (-g0 / (L * R))
    else:
        T11 = T_sl + L * 11_000.0
        P11 = P_sl * (T11 / T_sl) ** (-g0 / (L * R))
        T = 216.65
        P = P11 * np.exp(-g0 * (h - 11_000.0) / (R * T))

    return float(T), float(P)


def total_from_static(T: float, p_static: float, M: float, gas: Gas) -> Tuple[float, float, float, float]:
    """Return Tt, Pt, a, V for a perfect gas."""
    Tt = T * (1.0 + (gas.k - 1.0) / 2.0 * M**2)
    Pt = p_static * (1.0 + (gas.k - 1.0) / 2.0 * M**2) ** (gas.k / (gas.k - 1.0))
    a = sqrt(gas.k * gas.R * T)
    V = M * a
    return Tt, Pt, a, V


def bcr(gas: Gas) -> float:
    """Critical total-to-static pressure ratio."""
    return ((gas.k + 1.0) / 2.0) ** (gas.k / (gas.k - 1.0))


def fuel_air_ratio_paper(Tt3: float, Tt4: float, p: Params = P) -> float:
    """
    Fuel-air ratio used in the paper-matching version.

    fB = cpB * (Tt4 - Tt3) / (eta_B * FHV)
    """
    return p.cpB * (Tt4 - Tt3) / (p.eta_burner * p.FHV)


def turbine_min_area_values(m4: float, Tt4: float, Pt4: float, p: Params = P) -> Dict[str, float]:
    """Section 4_min turbine throat values, following the paper-matching code."""
    gas = p.gas
    B = bcr(gas)
    T4_min = Tt4 * 2.0 / (gas.k + 1.0)
    P4_min = p.sigma_burner * Pt4 / B
    rho4_min = P4_min / (gas.R * T4_min)
    c4_min = sqrt(gas.k * gas.R * T4_min)
    A4_min = m4 / (rho4_min * c4_min)
    return {
        "T4_min": T4_min,
        "P4_min": P4_min,
        "rho4_min": rho4_min,
        "c4_min": c4_min,
        "A4_min": A4_min,
    }


def nozzle_choked_values(m9: float, Tt9: float, Pt9: float, P0: float, p: Params = P) -> Dict[str, float]:
    """
    Section 9 and 9e values for choked/incompletely expanded nozzle.

    If the nozzle is not choked, this returns NaN for the paper's choked-nozzle
    quantities because the paper explicitly treats that model as inappropriate.
    """
    gas = p.gas
    B = bcr(gas)
    pressure_ratio = Pt9 / P0

    if pressure_ratio < B:
        return {
            "Bcr": B,
            "choked": False,
            "P9_IE": np.nan,
            "T9_IE": np.nan,
            "M9_IE": np.nan,
            "a9_IE": np.nan,
            "V9_IE": np.nan,
            "rho9_IE": np.nan,
            "A9_min": np.nan,
            "V9e": np.nan,
            "T9e": np.nan,
            "P9e": P0,
        }

    P9_IE = Pt9 / B
    T9_IE = Tt9 * (P9_IE / Pt9) ** ((gas.k - 1.0) / gas.k)
    M9_IE = sqrt((Tt9 / T9_IE - 1.0) * 2.0 / (gas.k - 1.0))
    a9_IE = sqrt(gas.k * gas.R * T9_IE)
    V9_IE = M9_IE * a9_IE
    rho9_IE = P9_IE / (gas.R * T9_IE)
    A9_min = m9 / (rho9_IE * V9_IE)

    # Equivalent expanded jet velocity used in the paper:
    # pressure thrust is folded into the velocity term.
    V9e = V9_IE + (P9_IE - P0) / (rho9_IE * V9_IE)
    T9e = Tt9 - V9e**2 / (2.0 * gas.cp)

    return {
        "Bcr": B,
        "choked": True,
        "P9_IE": P9_IE,
        "T9_IE": T9_IE,
        "M9_IE": M9_IE,
        "a9_IE": a9_IE,
        "V9_IE": V9_IE,
        "rho9_IE": rho9_IE,
        "A9_min": A9_min,
        "V9e": V9e,
        "T9e": T9e,
        "P9e": P0,
    }


def design_point(p: Params = P) -> Dict[str, float]:
    """Calculate the M=2, H=20 km design point and fixed engine areas."""
    air, gas = p.air, p.gas
    T0_DP, P0_DP = isa_atmosphere(p.altitude_DP_m)

    out: Dict[str, float] = {}
    out["M0"] = p.M0_DP
    out["altitude_m"] = p.altitude_DP_m
    out["T0"] = T0_DP
    out["P0"] = P0_DP
    out["m0"] = p.m0_DP
    out["CPR"] = p.CPR_DP
    out["Tt4"] = p.Tt4_DP

    # Section 0
    Tt0, Pt0, a0, V0 = total_from_static(T0_DP, P0_DP, p.M0_DP, air)
    out.update(Tt0=Tt0, Pt0=Pt0, a0=a0, V0=V0)

    # Section 2
    Tt2 = Tt0
    Pt2 = p.sigma_inlet * Pt0
    out.update(Tt2=Tt2, Pt2=Pt2)

    # Section 3 compressor outlet
    Tt3 = Tt2 * (1.0 + (p.CPR_DP ** ((air.k - 1.0) / air.k) - 1.0) / p.eta_compressor)
    Pt3 = p.CPR_DP * Pt2
    WC = air.cp * (Tt3 - Tt2)
    PC = p.m0_DP * WC
    out.update(Tt3=Tt3, Pt3=Pt3, WC=WC, PC=PC)

    # Section 4 combustor outlet / turbine inlet
    fB = fuel_air_ratio_paper(Tt3, p.Tt4_DP, p)
    if fB <= 0:
        raise ValueError(
            "Invalid design point: fuel-air ratio is <= 0. "
            "Increase Tt4_DP or reduce CPR_DP/design Mach."
        )
    mfB = p.m0_DP * fB
    Pt4 = p.sigma_burner * Pt3
    m4 = p.m0_DP * (1.0 + fB)
    out.update(fB=fB, mfB=mfB, Pt4=Pt4, m4=m4)

    # Section 4_min turbine minimum area
    out.update(turbine_min_area_values(m4, p.Tt4_DP, Pt4, p))

    # Section 5 turbine outlet / nozzle inlet
    Tt5 = p.Tt4_DP - WC / ((1.0 + fB) * gas.cp * p.eta_mech)
    TPR = (1.0 - (1.0 - Tt5 / p.Tt4_DP) / p.eta_turbine) ** (-gas.k / (gas.k - 1.0))
    Pt5 = Pt4 / TPR
    out.update(Tt5=Tt5, Pt5=Pt5, TPR=TPR)

    # Section 7 / Section 9
    Tt7 = Tt5
    Pt7 = p.sigma_ab_off * Pt5
    Tt9 = Tt7
    Pt9 = p.sigma_nozzle * Pt7
    out.update(Tt7=Tt7, Pt7=Pt7, Tt9=Tt9, Pt9=Pt9)

    # Choked nozzle and equivalent expansion
    out.update(nozzle_choked_values(m4, Tt9, Pt9, T0_DP*0 + P0_DP, p))
    if not out["choked"]:
        raise ValueError("Invalid design point: design nozzle is not choked.")

    # Performance using paper's equivalent expanded velocity
    thrust = p.m0_DP * ((1.0 + fB) * out["V9e"] - V0)
    ST = thrust / p.m0_DP
    SFC = mfB / thrust if thrust > 0 else np.nan
    out.update(thrust_N=thrust, thrust_kN=thrust / 1000.0, ST=ST,
               SFC_kg_N_s=SFC, SFC_kg_N_h=SFC * 3600.0)
    return out


def off_design(M0: float, altitude_m: float, n_ratio: float, dp: Dict[str, float], p: Params = P) -> Dict[str, float]:
    """Off-design calculation at a given Mach and altitude."""
    air, gas = p.air, p.gas
    T0, P0 = isa_atmosphere(altitude_m)

    out: Dict[str, float] = {"M0": M0, "altitude_m": altitude_m, "T0": T0, "P0": P0, "n_ratio": n_ratio}

    # Section 0 and 2
    Tt0, Pt0, a0, V0 = total_from_static(T0, P0, M0, air)
    Tt2 = Tt0
    Pt2 = p.sigma_inlet * Pt0
    out.update(Tt0=Tt0, Pt0=Pt0, a0=a0, V0=V0, Tt2=Tt2, Pt2=Pt2)

    # Off-design TIT from paper relation:
    # n/n_DP = sqrt((Tt4/Tt0) / (Tt4_DP/Tt0_DP))
    Tt4 = Tt0 * n_ratio**2 * p.Tt4_DP / dp["Tt0"]

    # Constant TPR = TPR_DP
    TPR = dp["TPR"]
    Tt5 = Tt4 * (1.0 - p.eta_turbine * (1.0 - TPR ** (-(gas.k - 1.0) / gas.k)))
    WT = gas.cp * (Tt4 - Tt5) * p.eta_mech
    out.update(Tt4=Tt4, Tt5=Tt5, WT=WT)

    # Iterative turbocomponent calculation to match fB
    fB = 0.0
    converged = False
    Tt3 = np.nan
    for iteration in range(1, 101):
        fB_old = fB
        Tt3 = Tt2 + (1.0 + fB_old) * WT / air.cp
        fB = fuel_air_ratio_paper(Tt3, Tt4, p)
        if fB <= 0:
            break
        if abs(fB - fB_old) / max(abs(fB), 1e-12) < 1e-5:
            converged = True
            break

    out.update(Tt3=Tt3, fB=fB, fB_previous=fB_old, iterations=iteration, converged=converged)

    if fB <= 0 or not np.isfinite(Tt3):
        out.update(valid=False, reason="fB <= 0 or invalid compressor/turbine match")
        out.update(CPR=np.nan, Pt3=np.nan, Pt4=np.nan, Pt5=np.nan, m4=np.nan, m0=np.nan, mf=np.nan,
                   Tt7=np.nan, Pt7=np.nan, Tt9=np.nan, Pt9=np.nan, Bcr=bcr(gas), choked=False,
                   P9_IE=np.nan, T9_IE=np.nan, M9_IE=np.nan, a9_IE=np.nan, V9_IE=np.nan,
                   rho9_IE=np.nan, A9_min=np.nan, V9e=np.nan, T9e=np.nan, P9e=P0,
                   thrust_N=np.nan, thrust_kN=np.nan, ST=np.nan, SFC_kg_N_s=np.nan, SFC_kg_N_h=np.nan)
        return out

    # Compressor pressure ratio and pressure levels
    CPR = (1.0 + p.eta_compressor * (Tt3 / Tt2 - 1.0)) ** (air.k / (air.k - 1.0))
    Pt3 = CPR * Pt2
    Pt4 = p.sigma_burner * Pt3
    Pt5 = Pt4 / TPR
    out.update(CPR=CPR, Pt3=Pt3, Pt4=Pt4, Pt5=Pt5, TPR=TPR)

    # Mass flow from fixed turbine minimum area
    B = bcr(gas)
    T4_min = Tt4 * 2.0 / (gas.k + 1.0)
    P4_min = p.sigma_burner * Pt4 / B
    rho4_min = P4_min / (gas.R * T4_min)
    c4_min = sqrt(gas.k * gas.R * T4_min)
    A4_min = dp["A4_min"]
    m4 = A4_min * rho4_min * c4_min
    m0 = m4 / (1.0 + fB)
    mf = m0 * fB
    out.update(T4_min=T4_min, P4_min=P4_min, rho4_min=rho4_min, c4_min=c4_min,
               A4_min=A4_min, m4=m4, m0=m0, mf=mf)

    # Section 7 and Section 9
    Tt7 = Tt5
    Pt7 = p.sigma_ab_off * Pt5
    Tt9 = Tt7
    Pt9 = p.sigma_nozzle * Pt7
    out.update(Tt7=Tt7, Pt7=Pt7, Tt9=Tt9, Pt9=Pt9)

    nozzle = nozzle_choked_values(m4, Tt9, Pt9, P0, p)
    out.update(nozzle)

    if not nozzle["choked"]:
        out.update(valid=False, reason="nozzle not choked; paper choked-nozzle model not applicable")
        out.update(thrust_N=np.nan, thrust_kN=np.nan, ST=np.nan, SFC_kg_N_s=np.nan, SFC_kg_N_h=np.nan)
        return out

    thrust = m0 * ((1.0 + fB) * out["V9e"] - V0)
    ST = thrust / m0
    SFC = mf / thrust if thrust > 0 else np.nan
    out.update(valid=True, reason="ok", thrust_N=thrust, thrust_kN=thrust / 1000.0, ST=ST,
               SFC_kg_N_s=SFC, SFC_kg_N_h=SFC * 3600.0)
    return out


def run_sweep(
    machs: Iterable[float],
    altitudes_m: Iterable[float],
    n_ratio: float = 1.0,
    p: Params = P,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    dp = design_point(p)
    rows: List[Dict[str, float]] = []
    for h in altitudes_m:
        for M in machs:
            rows.append(off_design(float(M), float(h), n_ratio, dp, p))
    return dp, pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, output_png: str = "thrust_vs_mach_altitude.png") -> None:
    plt.figure(figsize=(9, 6))
    for h, group in df.groupby("altitude_m"):
        group = group.sort_values("M0")
        plt.plot(group["M0"], group["thrust_kN"], marker="o", markersize=3, label=f"{h/1000:.0f} km")

    plt.xlabel("Mach number [-]")
    plt.ylabel("Thrust [kN]")
    plt.title("Turbojet thrust vs Mach number, design point M=2 at 20 km")
    plt.grid(True)
    plt.legend(title="Altitude")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.show()

# =============================================================================
# THRUST + DRAG VS MACH FOR DIFFERENT ALTITUDES
# =============================================================================
# Turbojet thrust:
#   Uses the paper-style turbojet model above, with design point M=2, h=20 km.
#
# Drag:
#   Uses the drag model pasted in the prompt:
#       W_TOG = 90,000 kg
#       S_PLAN = 350 m²
#       S_WET = 1000 m²
#       MAC = 21 m
#       IF = 1.05
#       alpha_rad = sqrt(CL^0.75 / 2)
#       cd_wave = 2 sin(alpha)^3
#       cf = 0.455 / log10(Re)^2.58
#       cd_f = cf * IF * S_WET/S_PLAN
#
# Plot:
#   Mach range starts at M=1.2, as requested.
# =============================================================================

# --- Drag constants from the prompt ---
W_TOG_DRAG = 90_000.0       # kg
S_PLAN_DRAG = 350.0         # m²
S_WET_DRAG = 1000.0         # m²
MAC_DRAG = 21.0             # m
IF_DRAG = 1.05
ACCEL_G_DRAG = 0.15
G_DRAG = 9.81


def drag_atmosphere_prompt(alt_m: float) -> tuple[float, float]:
    """
    Atmosphere from the pasted drag code.
    Returns rho [kg/m³], T [K].
    """
    alt_m = float(alt_m)
    if alt_m <= 11000:
        T = 288.15 - 0.0065 * alt_m
        rho = 1.225 * (T / 288.15)**4.256
    elif alt_m <= 25000:
        T = 216.65
        rho = 0.3639 * np.exp(-0.000157 * (alt_m - 11000))
    else:
        T = 216.65 + 0.003 * (alt_m - 25000)
        rho = 0.0401 * (T / 216.65)**-11.388
    return float(rho), float(T)


def reynolds_prompt(rho: float, v: float, temp: float, chord: float) -> float:
    """Reynolds number with Sutherland viscosity from the pasted drag code."""
    mu_0 = 1.7894e-5
    T_0 = 273.15
    S_suth = 110.4
    mu = mu_0 * (temp / T_0)**1.5 * (T_0 + S_suth) / (temp + S_suth)
    return float((rho * v * chord) / mu)


def drag_breakdown_prompt(M: float, altitude_m: float) -> dict[str, float]:
    """
    Drag and required thrust calculation from the pasted drag model.

    Returns drag only and drag + 0.15g acceleration requirement.
    """
    M = float(M)
    rho, T = drag_atmosphere_prompt(altitude_m)
    a = np.sqrt(1.4 * 287.0 * T)
    V = M * a
    q = 0.5 * rho * V**2

    cl_needed = (W_TOG_DRAG * G_DRAG) / max(q * S_PLAN_DRAG, 1e-30)

    # Same as pasted code:
    alpha_rad = np.sqrt((cl_needed**0.75) / 2.0)
    cd_wave = 2.0 * np.sin(alpha_rad)**3

    Re_dyn = reynolds_prompt(rho, V, T, MAC_DRAG)
    cf = 0.455 / (np.log10(Re_dyn)**2.58)
    cd_f = cf * IF_DRAG * (S_WET_DRAG / S_PLAN_DRAG)

    cd_total = cd_f + cd_wave
    drag_force_N = q * S_PLAN_DRAG * cd_total
    drag_friction_N = q * S_PLAN_DRAG * cd_f
    drag_wave_N = q * S_PLAN_DRAG * cd_wave
    thrust_required_N = drag_force_N + (W_TOG_DRAG * ACCEL_G_DRAG * G_DRAG)

    return {
        "M": M,
        "altitude_m": float(altitude_m),
        "rho": rho,
        "T": T,
        "V": float(V),
        "q": float(q),
        "CL": float(cl_needed),
        "alpha_deg": float(np.degrees(alpha_rad)),
        "Re": float(Re_dyn),
        "cf": float(cf),
        "cd_f": float(cd_f),
        "cd_wave": float(cd_wave),
        "cd_total": float(cd_total),
        "drag_N": float(drag_force_N),
        "drag_kN": float(drag_force_N / 1000.0),
        "drag_friction_kN": float(drag_friction_N / 1000.0),
        "drag_wave_kN": float(drag_wave_N / 1000.0),
        "thrust_required_0p15g_N": float(thrust_required_N),
        "thrust_required_0p15g_kN": float(thrust_required_N / 1000.0),
    }


def make_thrust_drag_altitude_plot(
    machs: np.ndarray | None = None,
    altitudes_m: np.ndarray | None = None,
    n_ratio: float = 1.0,
    p: Params = P,
    output_png: str = "thrust_drag_vs_mach_altitudes.png",
    output_csv: str = "thrust_drag_vs_mach_altitudes.csv",
):
    """
    Plot turbojet thrust and aircraft drag versus Mach number for different altitudes.
    """
    if machs is None:
        machs = np.linspace(1.2, 3.5, 80)
    if altitudes_m is None:
        altitudes_m = np.array([0, 5_000, 10_000, 15_000, 17_000, 20_000, 25_000], dtype=float)

    dp = design_point(p)
    rows = []

    for h in altitudes_m:
        for M in machs:
            tj = off_design(float(M), float(h), n_ratio, dp, p)
            dr = drag_breakdown_prompt(float(M), float(h))

            N_TURBOJET_ENGINES = 2
            thrust_kN = N_TURBOJET_ENGINES * float(tj.get("thrust_kN", np.nan))
            drag_kN = dr["drag_kN"]
            required_kN = dr["thrust_required_0p15g_kN"]

            rows.append({
                "Mach": float(M),
                "altitude_m": float(h),
                "altitude_km": float(h) / 1000.0,
                "turbojet_thrust_kN": thrust_kN,
                "drag_kN": drag_kN,
                "required_thrust_0p15g_kN": required_kN,
                "net_vs_drag_kN": thrust_kN - drag_kN if np.isfinite(thrust_kN) else np.nan,
                "net_vs_0p15g_req_kN": thrust_kN - required_kN if np.isfinite(thrust_kN) else np.nan,
                "CL": dr["CL"],
                "alpha_deg": dr["alpha_deg"],
                "cd_total": dr["cd_total"],
                "cd_f": dr["cd_f"],
                "cd_wave": dr["cd_wave"],
                "drag_friction_kN": dr["drag_friction_kN"],
                "drag_wave_kN": dr["drag_wave_kN"],
                "turbojet_valid": bool(tj.get("valid", True)),
                "turbojet_reason": tj.get("reason", "ok"),
                "m0_turbojet_kg_s": float(tj.get("m0", np.nan)),
                "CPR": float(tj.get("CPR", np.nan)),
                "Tt4_K": float(tj.get("Tt4", np.nan)),
            })

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
    except Exception:
        df = rows

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7.5))

    for h in altitudes_m:
        if hasattr(df, "loc"):
            sub = df[df["altitude_m"] == float(h)].sort_values("Mach")
            M_arr = sub["Mach"].to_numpy()
            T_arr = sub["turbojet_thrust_kN"].to_numpy()
            D_arr = sub["drag_kN"].to_numpy()
            R_arr = sub["required_thrust_0p15g_kN"].to_numpy()
        else:
            sub = [r for r in rows if r["altitude_m"] == float(h)]
            M_arr = np.array([r["Mach"] for r in sub])
            T_arr = np.array([r["turbojet_thrust_kN"] for r in sub])
            D_arr = np.array([r["drag_kN"] for r in sub])
            R_arr = np.array([r["required_thrust_0p15g_kN"] for r in sub])

        label_h = f"{h/1000:.0f} km"
        ax.plot(M_arr, T_arr, linewidth=2.2, label=f"Turbojet thrust {label_h}")
        ax.plot(M_arr, D_arr, linestyle="--", linewidth=1.7, label=f"Drag {label_h}")
        ax.plot(M_arr, R_arr, linestyle=":", linewidth=1.5, label=f"Drag + 0.15g {label_h}")

    ax.axvline(3.0, color="black", linestyle="-.", linewidth=1.5, label="M=3 reference")
    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("Force [kN]")
    ax.set_title(
        "Turbojet thrust and drag vs Mach number for different altitudes\n"
        "Mach range starts at M=1.2; drag model from pasted supersonic estimate"
    )
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()

    print("Design point:")
    print(f"  M0 = {dp['M0']:.2f}, altitude = {dp['altitude_m']/1000:.1f} km")
    print(f"  Tt4_DP = {dp['Tt4']:.1f} K, m0_DP = {dp['m0']:.3f} kg/s")
    print(f"  Design thrust = {dp['thrust_kN']:.3f} kN")
    print()
    print("Saved:")
    print(f"  {output_png}")
    print(f"  {output_csv}")

    return df


if __name__ == "__main__":
    machs = np.linspace(1.2, 3.5, 80)
    altitudes_m = np.array([0, 5_000, 10_000, 15_000, 17_000, 20_000, 25_000], dtype=float)

    make_thrust_drag_altitude_plot(
        machs=machs,
        altitudes_m=altitudes_m,
        n_ratio=1.0,
        p=P,
        output_png="thrust_drag_vs_mach_altitudes.png",
        output_csv="thrust_drag_vs_mach_altitudes.csv",
    )
