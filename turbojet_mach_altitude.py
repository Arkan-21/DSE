
"""
turbojet_thrust_drag_plot.py

Turbojet thrust-vs-Mach plot with aerodynamic drag overlaid.

- Turbojet design point:
    M0_DP = 2.0
    altitude_DP = 20 km
    ISA atmosphere supplies T0_DP and P0_DP

- Plot range:
    Mach 0 to 3.5
    Altitudes 0 to 25 km

- Drag model:
    Uses the C_D polar and C_L-alpha interpolation structure from the
    supplied drag code:
        C_D = a(M) C_L^2 + b(M) C_L + c(M)
        C_L = m(M) alpha_deg + k(M)

Notes:
    This remains a simplified educational model. The turbojet method follows
    the fixed-area off-design approach from the paper-style code.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, exp
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# =============================================================================
# ISA atmosphere
# =============================================================================

def isa_atmosphere(altitude_m: float) -> tuple[float, float, float]:
    """
    ISA atmosphere from 0 to 32 km.
    Returns:
        T [K], P [Pa], rho [kg/m^3]
    """
    h = float(altitude_m)
    g0 = 9.80665
    R = 287.05

    if h <= 11_000.0:
        T_sl = 288.15
        P_sl = 101325.0
        L = -0.0065
        T = T_sl + L * h
        P = P_sl * (T / T_sl) ** (-g0 / (L * R))
    elif h <= 20_000.0:
        T = 216.65
        P11 = 22632.06
        P = P11 * exp(-g0 * (h - 11_000.0) / (R * T))
    elif h <= 32_000.0:
        T20 = 216.65
        P20 = 5474.89
        L = 0.001
        T = T20 + L * (h - 20_000.0)
        P = P20 * (T / T20) ** (-g0 / (L * R))
    else:
        T = 228.65
        P32 = 868.02
        P = P32 * exp(-g0 * (h - 32_000.0) / (R * T))

    rho = P / (R * T)
    return T, P, rho


def speed_of_sound_from_T(T: float) -> float:
    return sqrt(1.4 * 287.05 * T)


def dynamic_pressure_from_mach_altitude(M: float, altitude_m: float) -> float:
    T, _, rho = isa_atmosphere(altitude_m)
    V = M * speed_of_sound_from_T(T)
    return 0.5 * rho * V**2


# =============================================================================
# Drag model from supplied drag-code structure
# =============================================================================

MACH_POLAR_DATA = np.array([0.65, 0.9, 1.1, 1.3, 2.0, 5.37, 7.38, 10.61])
A_POLAR_DATA = np.array([0.3804, 0.3418, 0.3459, 0.4006, 0.6049, 1.0314, 1.2753, 1.1948])
B_POLAR_DATA = np.array([-0.0011, 0.0100, 0.0012, 0.0037, 0.0010, 0.0145, 0.0354, 0.0962])
C_POLAR_DATA = np.array([0.0070, 0.0174, 0.0382, 0.0337, 0.0268, 0.0121, 0.0101, 0.0081])

CL_ALPHA_SLOPE_DATA = np.array([0.0430, 0.0457, 0.0428, 0.0372, 0.0271, 0.0167, 0.0128, 0.0110])
CL_ALPHA_INTERCEPT_DATA = np.array([-0.0347, -0.0381, -0.0235, -0.0084, 0.0011, -0.0032, -0.0030, -0.0048])

A_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, A_POLAR_DATA)
B_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, B_POLAR_DATA)
C_POLAR_INTERP = PchipInterpolator(MACH_POLAR_DATA, C_POLAR_DATA)
CL_ALPHA_SLOPE_INTERP = PchipInterpolator(MACH_POLAR_DATA, CL_ALPHA_SLOPE_DATA)
CL_ALPHA_INTERCEPT_INTERP = PchipInterpolator(MACH_POLAR_DATA, CL_ALPHA_INTERCEPT_DATA)


def cl_from_mach_alpha(M: float, alpha_deg: float, clamp_mach: bool = True) -> float:
    M_used = float(np.clip(M, MACH_POLAR_DATA.min(), MACH_POLAR_DATA.max())) if clamp_mach else float(M)
    slope = float(CL_ALPHA_SLOPE_INTERP(M_used))
    intercept = float(CL_ALPHA_INTERCEPT_INTERP(M_used))
    return slope * alpha_deg + intercept


def cd_from_mach_cl(M: float, CL: float, clamp_mach: bool = True) -> float:
    M_used = float(np.clip(M, MACH_POLAR_DATA.min(), MACH_POLAR_DATA.max())) if clamp_mach else float(M)
    a = float(A_POLAR_INTERP(M_used))
    b = float(B_POLAR_INTERP(M_used))
    c = float(C_POLAR_INTERP(M_used))
    return a * CL**2 + b * CL + c


def drag_from_mach_alpha(M: float, alpha_deg: float, altitude_m: float, S_ref: float) -> dict[str, float]:
    CL = cl_from_mach_alpha(M, alpha_deg, clamp_mach=True)
    CD = cd_from_mach_cl(M, CL, clamp_mach=True)
    q = dynamic_pressure_from_mach_altitude(M, altitude_m)
    D = q * S_ref * CD
    return {"CL": CL, "CD": CD, "q": q, "D_N": D, "D_kN": D / 1000.0}


# =============================================================================
# Turbojet model
# =============================================================================

@dataclass(frozen=True)
class Gas:
    k: float
    cp: float
    R: float


@dataclass(frozen=True)
class Params:
    air: Gas = Gas(k=1.4, cp=1005.0, R=287.0)
    gas: Gas = Gas(k=1.33, cp=1170.0, R=290.0)
    cpB: float = 1200.0
    FHV: float = 43e6

    sigma_inlet: float = 0.97
    sigma_burner: float = 0.98
    sigma_ab_off: float = 0.975
    sigma_nozzle: float = 0.96
    eta_compressor: float = 0.83
    eta_turbine: float = 0.90
    eta_burner: float = 0.98
    eta_mech: float = 0.99

    # Updated turbojet design point for your Mach 0 to 3 handover study.
    M0_DP: float = 2.0
    altitude_DP_m: float = 20_000.0

    # Sizing/design choices. Change these to resize the engine.
    m0_DP: float = 2 * 200.0
    CPR_DP: float = 8.0
    Tt4_DP: float = 1300.0


P = Params()


def total_from_static(T: float, p: float, M: float, gas: Gas) -> tuple[float, float, float, float]:
    Tt = T * (1.0 + (gas.k - 1.0) / 2.0 * M**2)
    Pt = p * (1.0 + (gas.k - 1.0) / 2.0 * M**2) ** (gas.k / (gas.k - 1.0))
    a = sqrt(gas.k * gas.R * T)
    V = M * a
    return Tt, Pt, a, V


def bcr(gas: Gas) -> float:
    return ((gas.k + 1.0) / 2.0) ** (gas.k / (gas.k - 1.0))


def fuel_air_ratio_paper(Tt3: float, Tt4: float, p: Params = P) -> float:
    return p.cpB * (Tt4 - Tt3) / (p.eta_burner * p.FHV)


def turbine_min_area_values(m4: float, Tt4: float, Pt4: float, p: Params = P) -> Dict[str, float]:
    gas = p.gas
    B = bcr(gas)
    T4_min = Tt4 * 2.0 / (gas.k + 1.0)
    P4_min = p.sigma_burner * Pt4 / B
    rho4_min = P4_min / (gas.R * T4_min)
    c4_min = sqrt(gas.k * gas.R * T4_min)
    A4_min = m4 / (rho4_min * c4_min)
    return {"T4_min": T4_min, "P4_min": P4_min, "rho4_min": rho4_min, "c4_min": c4_min, "A4_min": A4_min}


def nozzle_choked_values(m9: float, Tt9: float, Pt9: float, P0: float, p: Params = P) -> Dict[str, float]:
    gas = p.gas
    B = bcr(gas)
    P9_IE = Pt9 / B
    T9_IE = Tt9 * (P9_IE / Pt9) ** ((gas.k - 1.0) / gas.k)
    M9_IE = sqrt(max(0.0, (Tt9 / T9_IE - 1.0) * 2.0 / (gas.k - 1.0)))
    a9_IE = sqrt(gas.k * gas.R * T9_IE)
    V9_IE = M9_IE * a9_IE
    rho9_IE = P9_IE / (gas.R * T9_IE)
    A9_min = m9 / (rho9_IE * V9_IE)
    V9e = V9_IE + (P9_IE - P0) / (rho9_IE * V9_IE)
    T9e = Tt9 - V9e**2 / (2.0 * gas.cp)
    return {
        "Bcr": B, "P9_IE": P9_IE, "T9_IE": T9_IE, "M9_IE": M9_IE,
        "a9_IE": a9_IE, "V9_IE": V9_IE, "rho9_IE": rho9_IE,
        "A9_min": A9_min, "V9e": V9e, "T9e": T9e, "P9e": P0,
    }


def design_point(p: Params = P) -> Dict[str, float]:
    T0_DP, P0_DP, _ = isa_atmosphere(p.altitude_DP_m)
    air, gas = p.air, p.gas

    out: Dict[str, float] = {
        "M0": p.M0_DP, "altitude_m": p.altitude_DP_m,
        "T0": T0_DP, "P0": P0_DP, "m0": p.m0_DP,
        "CPR": p.CPR_DP, "Tt4": p.Tt4_DP,
    }

    Tt0, Pt0, a0, V0 = total_from_static(T0_DP, P0_DP, p.M0_DP, air)
    out.update(Tt0=Tt0, Pt0=Pt0, a0=a0, V0=V0)

    Tt2 = Tt0
    Pt2 = p.sigma_inlet * Pt0
    out.update(Tt2=Tt2, Pt2=Pt2)

    Tt3 = Tt2 * (1.0 + (p.CPR_DP ** ((air.k - 1.0) / air.k) - 1.0) / p.eta_compressor)
    Pt3 = p.CPR_DP * Pt2
    WC = air.cp * (Tt3 - Tt2)
    out.update(Tt3=Tt3, Pt3=Pt3, WC=WC)

    fB = fuel_air_ratio_paper(Tt3, p.Tt4_DP, p)
    mfB = p.m0_DP * fB
    Pt4 = p.sigma_burner * Pt3
    m4 = p.m0_DP * (1.0 + fB)
    out.update(fB=fB, mfB=mfB, Pt4=Pt4, m4=m4)

    out.update(turbine_min_area_values(m4, p.Tt4_DP, Pt4, p))

    Tt5 = p.Tt4_DP - WC / ((1.0 + fB) * gas.cp * p.eta_mech)
    TPR = (1.0 - (1.0 - Tt5 / p.Tt4_DP) / p.eta_turbine) ** (-gas.k / (gas.k - 1.0))
    Pt5 = Pt4 / TPR
    out.update(Tt5=Tt5, Pt5=Pt5, TPR=TPR)

    Tt7 = Tt5
    Pt7 = p.sigma_ab_off * Pt5
    Tt9 = Tt7
    Pt9 = p.sigma_nozzle * Pt7
    out.update(Tt7=Tt7, Pt7=Pt7, Tt9=Tt9, Pt9=Pt9)

    out.update(nozzle_choked_values(m4, Tt9, Pt9, P0_DP, p))

    thrust = p.m0_DP * ((1.0 + fB) * out["V9e"] - V0)
    ST = thrust / p.m0_DP
    SFC = mfB / thrust if thrust > 0 else np.nan
    out.update(thrust_N=thrust, thrust_kN=thrust / 1000.0, ST=ST, SFC_kg_N_h=SFC * 3600.0)
    return out


def off_design(M0: float, altitude_m: float, n_ratio: float, dp: Dict[str, float], p: Params = P) -> Dict[str, float]:
    T0, P0, _ = isa_atmosphere(altitude_m)
    air, gas = p.air, p.gas
    out: Dict[str, float] = {"M0": M0, "altitude_m": altitude_m, "T0": T0, "P0": P0, "n_ratio": n_ratio}

    Tt0, Pt0, a0, V0 = total_from_static(T0, P0, M0, air)
    out.update(Tt0=Tt0, Pt0=Pt0, a0=a0, V0=V0)

    Tt2 = Tt0
    Pt2 = p.sigma_inlet * Pt0
    out.update(Tt2=Tt2, Pt2=Pt2)

    # Paper relation:
    # n/n_DP = sqrt((Tt4/Tt0) / (Tt4_DP/Tt0_DP))
    Tt4 = Tt0 * n_ratio**2 * p.Tt4_DP / dp["Tt0"]

    # Prevent impossible cases where the requested combustor exit temperature is below
    # compressor inlet temperature. These points are marked invalid in the plot.
    if Tt4 <= Tt2:
        out.update(valid=False, thrust_N=np.nan, thrust_kN=np.nan)
        return out

    TPR = dp["TPR"]
    Tt5 = Tt4 * (1.0 - p.eta_turbine * (1.0 - TPR ** (-(gas.k - 1.0) / gas.k)))
    WT = gas.cp * (Tt4 - Tt5) * p.eta_mech
    out.update(Tt4=Tt4, Tt5=Tt5, WT=WT)

    fB = 0.0
    for iteration in range(1, 101):
        fB_old = fB
        Tt3 = Tt2 + (1.0 + fB_old) * WT / air.cp
        fB = fuel_air_ratio_paper(Tt3, Tt4, p)
        if fB <= 0:
            out.update(valid=False, thrust_N=np.nan, thrust_kN=np.nan)
            return out
        if abs(fB - fB_old) / max(abs(fB), 1e-12) < 1e-5:
            break

    CPR = (1.0 + p.eta_compressor * (Tt3 / Tt2 - 1.0)) ** (air.k / (air.k - 1.0))
    Pt3 = CPR * Pt2
    Pt4 = p.sigma_burner * Pt3
    Pt5 = Pt4 / TPR
    out.update(Tt3=Tt3, fB=fB, CPR=CPR, Pt3=Pt3, Pt4=Pt4, Pt5=Pt5, TPR=TPR)

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

    Tt7 = Tt5
    Pt7 = p.sigma_ab_off * Pt5
    Tt9 = Tt7
    Pt9 = p.sigma_nozzle * Pt7
    out.update(Tt7=Tt7, Pt7=Pt7, Tt9=Tt9, Pt9=Pt9)

    # Only use the choked-nozzle model when the critical static pressure remains above ambient.
    P9_critical = Pt9 / bcr(gas)
    if P9_critical < P0:
        out.update(valid=False, thrust_N=np.nan, thrust_kN=np.nan, P9_IE=P9_critical)
        return out

    nozzle = nozzle_choked_values(m4, Tt9, Pt9, P0, p)
    out.update(nozzle)

    thrust = m0 * ((1.0 + fB) * out["V9e"] - V0)
    ST = thrust / m0 if m0 > 0 else np.nan
    SFC = mf / thrust if thrust > 0 else np.nan
    out.update(valid=True, thrust_N=thrust, thrust_kN=thrust / 1000.0, ST=ST, SFC_kg_N_h=SFC * 3600.0)
    return out


# =============================================================================
# Combined thrust-drag plot
# =============================================================================

def build_thrust_drag_table(
    mach_values: Iterable[float],
    altitude_values_m: Iterable[float],
    alpha_deg: float = 3.5,
    S_ref: float = 400.0,
    n_ratio: float = 1.0,
    p: Params = P,
) -> pd.DataFrame:
    dp = design_point(p)
    rows = []

    for h in altitude_values_m:
        for M in mach_values:
            tj = off_design(M0=float(M), altitude_m=float(h), n_ratio=n_ratio, dp=dp, p=p)
            drag = drag_from_mach_alpha(M=float(M), alpha_deg=alpha_deg, altitude_m=float(h), S_ref=S_ref)

            row = {
                "Mach": float(M),
                "Altitude_m": float(h),
                "Altitude_km": float(h) / 1000.0,
                "alpha_deg": alpha_deg,
                "S_ref_m2": S_ref,
                "Thrust_N": tj.get("thrust_N", np.nan),
                "Thrust_kN": tj.get("thrust_kN", np.nan),
                "Drag_N": drag["D_N"],
                "Drag_kN": drag["D_kN"],
                "Net_N": tj.get("thrust_N", np.nan) - drag["D_N"],
                "Net_kN": tj.get("thrust_kN", np.nan) - drag["D_kN"],
                "CL": drag["CL"],
                "CD": drag["CD"],
                "q_Pa": drag["q"],
                "valid_turbojet": tj.get("valid", True),
                "m0_kg_s": tj.get("m0", np.nan),
                "CPR": tj.get("CPR", np.nan),
                "Tt4_K": tj.get("Tt4", np.nan),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def plot_thrust_and_drag(
    mach_values: Iterable[float],
    altitude_values_m: Iterable[float],
    alpha_deg: float = 3.5,
    S_ref: float = 400.0,
    n_ratio: float = 1.0,
    save_path: str = "turbojet_thrust_drag_vs_mach.png",
) -> pd.DataFrame:
    table = build_thrust_drag_table(
        mach_values=mach_values,
        altitude_values_m=altitude_values_m,
        alpha_deg=alpha_deg,
        S_ref=S_ref,
        n_ratio=n_ratio,
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    for h in altitude_values_m:
        sub = table[table["Altitude_m"] == float(h)].sort_values("Mach")
        label_h = f"{h/1000:.0f} km"

        ax.plot(sub["Mach"], sub["Thrust_kN"], linewidth=2.0, label=f"Thrust {label_h}")
        ax.plot(sub["Mach"], sub["Drag_kN"], linestyle="--", linewidth=1.6, label=f"Drag {label_h}")

    ax.axvline(P.M0_DP, linestyle=":", linewidth=1.5, label="Turbojet design M=2")
    ax.axvline(3.0, linestyle="-.", linewidth=1.5, label="Ramjet takeover check M=3")

    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("Force [kN]")
    ax.set_title(
        f"Turbojet thrust and aircraft drag vs Mach\n"
        f"Design point: M={P.M0_DP:g}, h={P.altitude_DP_m/1000:.0f} km | "
        f"Drag: α={alpha_deg:g}°, S_ref={S_ref:g} m²"
    )
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return table


def plot_thrust_and_drag_alpha_sweep(
    mach_values: Iterable[float],
    altitude_values_m: Iterable[float],
    alpha_values_deg: Iterable[float],
    S_ref: float = 400.0,
    n_ratio: float = 1.0,
    save_path: str = "turbojet_thrust_drag_vs_mach_alpha_sweep.png",
) -> pd.DataFrame:
    """
    Plot turbojet thrust vs Mach for each altitude, with drag curves for multiple
    angles of attack overlaid.

    Thrust is independent of alpha, so it is plotted once per altitude.
    Drag depends on alpha, so it is plotted for every altitude-alpha pair.
    """
    all_tables = []

    fig, ax = plt.subplots(figsize=(12, 7.5))

    for h in altitude_values_m:
        h_label = f"{h/1000:.0f} km"

        # Compute and plot thrust once for this altitude.
        table_thrust = build_thrust_drag_table(
            mach_values=mach_values,
            altitude_values_m=[h],
            alpha_deg=list(alpha_values_deg)[0],
            S_ref=S_ref,
            n_ratio=n_ratio,
        ).sort_values("Mach")

        ax.plot(
            table_thrust["Mach"],
            table_thrust["Thrust_kN"],
            linewidth=2.2,
            label=f"Thrust {h_label}",
        )

        # Compute and plot drag for each alpha.
        for alpha in alpha_values_deg:
            table_alpha = build_thrust_drag_table(
                mach_values=mach_values,
                altitude_values_m=[h],
                alpha_deg=float(alpha),
                S_ref=S_ref,
                n_ratio=n_ratio,
            ).sort_values("Mach")

            table_alpha["alpha_deg"] = float(alpha)
            all_tables.append(table_alpha)

            ax.plot(
                table_alpha["Mach"],
                table_alpha["Drag_kN"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label=f"Drag {h_label}, α={alpha:g}°",
            )

    ax.axvline(P.M0_DP, linestyle=":", linewidth=1.5, label="Turbojet design M=2")
    ax.axvline(3.0, linestyle="-.", linewidth=1.5, label="Ramjet takeover check M=3")

    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("Force [kN]")
    ax.set_title(
        f"Turbojet thrust and aircraft drag vs Mach\n"
        f"Design point: M={P.M0_DP:g}, h={P.altitude_DP_m/1000:.0f} km | "
        f"S_ref={S_ref:g} m² | multiple α"
    )
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return pd.concat(all_tables, ignore_index=True)


if __name__ == "__main__":
    mach_values = np.linspace(0.05, 3.5, 140)   # avoid exactly M=0 for cleaner drag/thrust handling
    altitude_values_m = np.array([0, 5_000, 10_000, 15_000, 20_000, 25_000], dtype=float)

    # Change these numbers for your aircraft.
    alpha_values_deg = [0.0, 3.5, 7.5]
    S_ref = 400.0

    dp = design_point(P)
    print("\nTurbojet design point")
    print("---------------------")
    print(f"M0_DP       = {dp['M0']:.2f}")
    print(f"h_DP        = {dp['altitude_m']/1000:.1f} km")
    print(f"T0_DP       = {dp['T0']:.2f} K")
    print(f"P0_DP       = {dp['P0']:.2f} Pa")
    print(f"Tt0_DP      = {dp['Tt0']:.2f} K")
    print(f"m0_DP       = {dp['m0']:.3f} kg/s")
    print(f"A4_min      = {dp['A4_min']:.6f} m²")
    print(f"A9_min      = {dp['A9_min']:.6f} m²")
    print(f"Thrust_DP   = {dp['thrust_kN']:.3f} kN")

    df = plot_thrust_and_drag_alpha_sweep(
        mach_values=mach_values,
        altitude_values_m=altitude_values_m,
        alpha_values_deg=alpha_values_deg,
        S_ref=S_ref,
        n_ratio=1.0,
        save_path="turbojet_thrust_drag_vs_mach_alpha_3cases.png",
    )

    df.to_csv("turbojet_thrust_drag_alpha_3cases_results.csv", index=False)
    print("\nSaved:")
    print("  turbojet_thrust_drag_vs_mach_alpha_3cases.png")
    print("  turbojet_thrust_drag_alpha_3cases_results.csv")

    # Print a few points near design/handover.
    # The Mach grid is made with np.linspace(), so it usually does not contain
    # exactly M = 1.0, 2.0, 3.0, or 3.5. Therefore, this block selects the
    # nearest available Mach number instead of filtering for exact rounded values.
    target_machs = [1.0, 2.0, 3.0, 3.5]
    sample_parts = []

    for target in target_machs:
        nearest_mach = df.loc[(df["Mach"] - target).abs().idxmin(), "Mach"]
        sample_parts.append(df[np.isclose(df["Mach"], nearest_mach)])

    sample = pd.concat(sample_parts, ignore_index=True)

    print("\nSample output:")
    print(
        sample[
            [
                "Altitude_km",
                "Mach",
                "alpha_deg",
                "Thrust_kN",
                "Drag_kN",
                "Net_kN",
                "m0_kg_s",
                "CPR",
                "Tt4_K",
            ]
        ].to_string(index=False)
    )
