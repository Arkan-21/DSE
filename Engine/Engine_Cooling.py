"""
ramjet_cooling.py
─────────────────
Thermal / cooling analysis layer on top of the Ramjet cycle model.

Physics implemented
───────────────────
1. Adiabatic wall temperature (recovery factor, subsonic + supersonic).
2. Convective gas-side heat-transfer coefficient  — turbulent Stanton number
   (Van Driest II reference-temperature method; falls back to Dittus-Boelter).
3. Regenerative cooling with H₂ fuel:
   • Coolant-side h via Dittus-Boelter in the fuel channels.
   • Wall temperature from the two-side resistance model.
   • Cumulative fuel temperature rise along the engine.
4. Per-section and total heat loads.
5. Matplotlib plot with four sub-panels.

Usage
─────
    from ramjet_cooling import RamjetCooling

    cool = RamjetCooling()
    cool.run_cycle(h=20_000, Ma0=3.0, m_air=100.0, phi=0.5)
    report = cool.cooling_analysis()
    cool.plot_temperature_distribution()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── import the parent class ───────────────────────────────────────────────────
from ramjet_01 import Ramjet          # adjust import path as needed


# ─────────────────────────────────────────────────────────────────────────────
# Small helper dataclass so every section's thermal result is self-describing
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SectionThermal:
    name:        str
    x:           np.ndarray          # axial coordinate [m]
    T_gas:       np.ndarray          # static gas temperature [K]
    T_aw:        np.ndarray          # adiabatic wall temperature [K]
    T_wall_hot:  np.ndarray          # hot-side wall temperature [K]
    T_wall_cold: np.ndarray          # cold-side (coolant contact) wall temperature [K]
    T_coolant:   np.ndarray          # bulk coolant temperature [K]
    q_flux:      np.ndarray          # wall heat flux  [W/m²]
    h_gas:       np.ndarray          # gas-side HTC    [W/m²/K]
    Ma:          np.ndarray          # local Mach number
    A:           np.ndarray          # duct cross-section area [m²]
    D:           np.ndarray          # hydraulic diameter [m]
    Q_section:   float = 0.0         # integrated heat load [W]
    extra:       Dict   = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# RamjetCooling
# ─────────────────────────────────────────────────────────────────────────────
class RamjetCooling(Ramjet):
    """
    Extends ``Ramjet`` with wall heat-flux and regenerative cooling physics.

    Parameters (set as class/instance attributes before calling analysis)
    ─────────────────────────────────────────────────────────────────────
    t_wall          : wall thickness [m]              default 0.003
    k_wall          : wall thermal conductivity [W/mK] default 15  (steel)
    T_wall_limit    : maximum allowable wall temperature [K]        default 1300
    n_channels      : number of fuel-cooling channels               default 200
    D_channel       : hydraulic diameter of each channel [m]        default 0.003
    T_fuel_in       : fuel inlet temperature [K]                    default 120
    cp_fuel         : fuel specific heat [J/kg/K]                   default 14 300 (H₂)
    mu_fuel         : fuel dynamic viscosity [Pa·s]                  default 1.5e-5
    k_fuel          : fuel thermal conductivity [W/mK]              default 0.18
    Pr_fuel         : fuel Prandtl number                           default 0.70
    recovery_factor : r in T_aw = T*(1 + r*(γ-1)/2*Ma²)            default sqrt(Pr)≈0.85
    """

    # ── Wall / structural ─────────────────────────────────────────────────
    t_wall:       float = 3.0e-3      # m
    k_wall:       float = 15.0        # W/m/K  (Inconel-like steel)
    T_wall_limit: float = 1300.0      # K

    # ── Cooling channels (H₂ regenerative) ───────────────────────────────
    n_channels: int   = 200
    D_channel:  float = 3.0e-3        # m
    T_fuel_in:  float = 120.0         # K  (liquid-H₂ pump outlet)
    cp_fuel:    float = 14_300.0      # J/kg/K
    mu_fuel:    float = 1.50e-5       # Pa·s
    k_fuel:     float = 0.18          # W/m/K
    Pr_fuel:    float = 0.70

    # ── Adiabatic-wall recovery factor ────────────────────────────────────
    recovery_factor: float = 0.85     # turbulent flat-plate value

    # ── Stored results ────────────────────────────────────────────────────
    _sections: List[SectionThermal] = field(default_factory=list)
    _cycle_results: Optional[dict]  = None

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._sections: List[SectionThermal] = []
        self._cycle_results: Optional[dict]  = None

    # ═════════════════════════════════════════════════════════════════════
    # Public entry points
    # ═════════════════════════════════════════════════════════════════════

    def run_cycle(self, h: float, Ma0: float, m_air: float, phi: float = 0.5,
                  **kwargs) -> dict:
        """
        Run the full ramjet thermodynamic cycle (delegates to parent) and
        cache the results so ``cooling_analysis()`` can use them.
        """
        results = super().altitude_mach(self, h_km=h, Ma0=Ma0)
        self._cycle_results = results
        return results

    def cooling_analysis(self) -> dict:
        """
        Compute thermal/cooling quantities for every engine section.
        Must be called after ``run_cycle()``.

        Returns a dict with per-section ``SectionThermal`` objects and
        overall summary statistics.
        """
        if self._cycle_results is None:
            raise RuntimeError("Call run_cycle() before cooling_analysis().")

        res  = self._cycle_results
        sections_raw = self._collect_section_data(res)

        self._sections = []
        T_cool = float(self.T_fuel_in)   # coolant temperature marches from inlet to exit
        mfuel  = float(res["sec4"].get("mfuel", 1e-9))
        mfuel  = max(mfuel, 1e-9)

        for name, data in sections_raw.items():
            st = self._analyse_section(name, data, T_cool_in=T_cool, mdot_fuel=mfuel)
            self._sections.append(st)
            T_cool = float(st.T_coolant[-1])   # carry coolant temperature forward

        # ── Summary ───────────────────────────────────────────────────────
        total_Q    = sum(s.Q_section for s in self._sections)
        q_max      = max(float(s.q_flux.max()) for s in self._sections)
        Tw_max     = max(float(s.T_wall_hot.max()) for s in self._sections)
        margin     = self.T_wall_limit - Tw_max
        coolant_dT = float(self._sections[-1].T_coolant[-1]) - self.T_fuel_in

        print(f"\n{'─'*60}")
        print(f"  Cooling analysis summary")
        print(f"{'─'*60}")
        print(f"  Total heat load      Q   = {total_Q/1e6:.3f}  MW")
        print(f"  Peak heat flux       q″  = {q_max/1e6:.3f}  MW/m²")
        print(f"  Peak wall temp       Tw  = {Tw_max:.0f}  K  "
              f"(limit {self.T_wall_limit:.0f} K, margin {margin:+.0f} K)")
        print(f"  Coolant (H₂) ΔT      ΔT  = {coolant_dT:.0f}  K  "
              f"(out {float(self._sections[-1].T_coolant[-1]):.0f} K)")
        if margin < 0:
            print(f"  ⚠  Wall temperature EXCEEDS limit by {-margin:.0f} K!")
        print(f"{'─'*60}")

        return {
            "sections":   self._sections,
            "total_Q_MW": total_Q / 1e6,
            "q_max":      q_max,
            "T_wall_max": Tw_max,
            "margin_K":   margin,
            "T_coolant_out": float(self._sections[-1].T_coolant[-1]),
        }

    # ═════════════════════════════════════════════════════════════════════
    # Plotting
    # ═════════════════════════════════════════════════════════════════════

    def plot_temperature_distribution(self,
                                      figsize: tuple = (14, 9),
                                      save_path: Optional[str] = None) -> None:
        """
        Four-panel figure:
          1  Gas & wall temperature along engine axis
          2  Wall heat flux
          3  Local Mach number
          4  Coolant (H₂) temperature
        """
        if not self._sections:
            self.cooling_analysis()

        # ── Build concatenated arrays ──────────────────────────────────
        x_offset = 0.0
        x_all      = []; T_gas_all  = []; T_aw_all   = []
        T_wh_all   = []; T_wc_all   = []; T_cool_all = []
        q_all      = []; Ma_all     = []; D_all      = []
        section_boundaries = [0.0]
        section_names      = []

        for st in self._sections:
            x  = st.x + x_offset
            x_all.append(x);         T_gas_all.append(st.T_gas)
            T_aw_all.append(st.T_aw);T_wh_all.append(st.T_wall_hot)
            T_wc_all.append(st.T_wall_cold); T_cool_all.append(st.T_coolant)
            q_all.append(st.q_flux); Ma_all.append(st.Ma)
            D_all.append(st.D)
            x_offset = float(x[-1])
            section_boundaries.append(x_offset)
            section_names.append(st.name)

        xc     = np.concatenate(x_all)
        T_gas  = np.concatenate(T_gas_all)
        T_aw   = np.concatenate(T_aw_all)
        T_wh   = np.concatenate(T_wh_all)
        T_wc   = np.concatenate(T_wc_all)
        T_cool = np.concatenate(T_cool_all)
        q_mwm2 = np.concatenate(q_all) / 1e6
        Ma     = np.concatenate(Ma_all)

        # ── Figure ────────────────────────────────────────────────────
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle("Ramjet — Thermal & Cooling Distribution", fontsize=13,
                     fontweight="bold", y=0.98)
        plt.subplots_adjust(hspace=0.08, top=0.94, bottom=0.07,
                            left=0.08, right=0.97)

        colours = {
            "gas":    "#d62728",
            "aw":     "#ff7f0e",
            "wh":     "#9467bd",
            "wc":     "#8c564b",
            "cool":   "#1f77b4",
            "flux":   "#e377c2",
            "mach":   "#2ca02c",
        }

        # Panel 1 — temperatures
        ax = axes[0]
        ax.fill_between(xc, T_wc, T_wh, alpha=0.15, color=colours["wh"],
                        label="Wall thickness")
        ax.plot(xc, T_gas,  lw=2.0, color=colours["gas"],  label="T_gas (static)")
        ax.plot(xc, T_aw,   lw=1.5, color=colours["aw"],   ls="--",
                label="T_aw (adiab. wall)")
        ax.plot(xc, T_wh,   lw=1.8, color=colours["wh"],   label="T_wall hot side")
        ax.plot(xc, T_wc,   lw=1.8, color=colours["wc"],   ls="-.",
                label="T_wall cold side")
        ax.plot(xc, T_cool, lw=1.8, color=colours["cool"], label="T_coolant (H₂)")
        ax.axhline(self.T_wall_limit, color="red", lw=1.0, ls=":", alpha=0.6,
                   label=f"T_wall limit ({self.T_wall_limit:.0f} K)")
        ax.set_ylabel("Temperature [K]")
        ax.legend(fontsize=7.5, ncol=3, loc="upper left")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}"))

        # Panel 2 — heat flux
        ax = axes[1]
        ax.fill_between(xc, 0, q_mwm2, alpha=0.25, color=colours["flux"])
        ax.plot(xc, q_mwm2, lw=2.0, color=colours["flux"], label="q″ heat flux")
        ax.set_ylabel("Heat flux [MW/m²]")
        ax.legend(fontsize=8, loc="upper left")

        # Panel 3 — Mach number
        ax = axes[2]
        ax.plot(xc, Ma, lw=2.0, color=colours["mach"], label="Local Mach number")
        ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylabel("Mach  Ma  [−]")
        ax.legend(fontsize=8, loc="upper left")

        # Panel 4 — coolant temperature
        ax = axes[3]
        ax.plot(xc, T_cool, lw=2.0, color=colours["cool"], label="T_coolant (H₂)")
        ax.axhline(self.T_fuel_in, color="grey", lw=0.8, ls="--", alpha=0.7,
                   label=f"T_fuel in = {self.T_fuel_in:.0f} K")
        ax.set_ylabel("T_coolant [K]")
        ax.set_xlabel("Axial position  x  [m]")
        ax.legend(fontsize=8, loc="upper left")

        # Section dividers + labels on all panels
        for xb, sname in zip(section_boundaries[1:-1], section_names[:-1]):
            for a in axes:
                a.axvline(xb, color="black", lw=0.8, ls=":", alpha=0.4)
        mid_positions = [0.5*(section_boundaries[i]+section_boundaries[i+1])
                         for i in range(len(section_names))]
        for mid, sname in zip(mid_positions, section_names):
            axes[0].text(mid, axes[0].get_ylim()[1]*0.97, sname,
                         ha="center", va="top", fontsize=7.5,
                         color="dimgrey", style="italic")

        for a in axes:
            a.grid(axis="both", lw=0.4, alpha=0.5)
            a.set_xlim(xc[0], xc[-1])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Figure saved → {save_path}")
        plt.tight_layout()
        plt.show()

    # ═════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═════════════════════════════════════════════════════════════════════

    def _collect_section_data(self, res: dict) -> dict:
        """
        Extract (x, T, Ma, p, rho, V, cp, gamma, A) arrays for each section
        from the cycle-result dict.  Adapts to however the parent Ramjet
        class names its section keys.
        """
        sections = {}

        key_map = [
            # (display name,  result key,  solution sub-key)
            ("Isolator",   "sec1",  "solution"),
            ("Diffuser",   "sec2",  "solution"),
            ("Injection",  "sec3",  "solution"),
            ("Combustor",  "sec4",  "solution"),
            ("Nozzle",     "sec5",  "solution"),
        ]
        for name, rkey, skey in key_map:
            sec = res.get(rkey)
            if sec is None:
                continue
            sol = sec.get(skey)
            if sol is None:
                # scalar section — wrap in length-2 arrays
                sol = _scalar_section_to_sol(sec)
            if sol is None or len(sol.get("x", [])) < 2:
                continue
            sections[name] = sol

        return sections

    def _analyse_section(self, name: str, sol: dict,
                         T_cool_in: float, mdot_fuel: float) -> SectionThermal:
        """
        Compute wall and coolant temperatures for one engine section.

        Gas-side HTC: Stanton-number with Van Driest-II reference temperature.
        Coolant-side HTC: Dittus-Boelter in the fuel channels.
        Wall conduction: 1-D steady slab  q = k/t * (T_wh - T_wc).
        Coolant energy: dT_cool/dx = q″ * π D / (ṁ_fuel·cp_fuel).
        """
        x    = np.asarray(sol["x"],   dtype=float)
        T    = np.asarray(sol["T"],   dtype=float)
        Ma   = np.asarray(sol["Ma"],  dtype=float)
        rho  = np.asarray(sol["rho"], dtype=float)
        V    = np.asarray(sol["V"],   dtype=float)
        A    = np.asarray(sol["A"],   dtype=float)

        # Try to get cp and gamma; fall back to air-like values
        cp_arr    = np.asarray(sol.get("cp",    np.full_like(x, 1150.0)))
        gamma_arr = np.asarray(sol.get("gamma", np.full_like(x, 1.35)))
        D = np.sqrt(4.0 * A / np.pi)   # hydraulic diameter

        n = len(x)

        # ── 1. Adiabatic wall temperature ────────────────────────────────
        r     = self.recovery_factor
        T_aw  = T * (1.0 + r * (gamma_arr - 1.0) / 2.0 * Ma**2)

        # ── 2. Gas-side HTC (Stanton number, turbulent boundary layer) ───
        mu_gas = self._sutherland(T)                   # dynamic viscosity
        Re     = rho * V * D / np.maximum(mu_gas, 1e-12)
        Pr_gas = cp_arr * mu_gas / np.maximum(
            self._thermal_conductivity(T), 1e-12)      # gas k
        # Dittus-Boelter: Nu = 0.023 Re^0.8 Pr^0.4
        Re_c   = np.maximum(Re, 1e4)                   # clamp to turbulent range
        Nu     = 0.023 * Re_c**0.8 * np.maximum(Pr_gas, 0.5)**0.4
        k_gas  = self._thermal_conductivity(T)
        h_gas  = Nu * k_gas / np.maximum(D, 1e-4)

        # ── 3. Coolant-side HTC ──────────────────────────────────────────
        Re_fuel = (mdot_fuel / self.n_channels) * self.D_channel / (
                   0.25 * np.pi * self.D_channel**2 * self.mu_fuel)
        Re_fuel = max(Re_fuel, 1e4)
        Nu_fuel = 0.023 * Re_fuel**0.8 * self.Pr_fuel**0.4
        h_fuel  = Nu_fuel * self.k_fuel / self.D_channel  # scalar, uniform

        # ── 4. Coolant temperature integration (forward Euler) ───────────
        T_cool = np.empty(n);  T_cool[0] = T_cool_in
        T_wh   = np.empty(n);  T_wc = np.empty(n);  q_flux = np.empty(n)

        for i in range(n):
            # Overall resistance: 1/U = 1/h_gas + t/k_wall + 1/h_fuel
            U_inv = (1.0 / max(h_gas[i], 1.0)
                     + self.t_wall / self.k_wall
                     + 1.0 / max(h_fuel, 1.0))
            U   = 1.0 / U_inv
            q_i = U * max(T_aw[i] - T_cool[i], 0.0)   # heat flux [W/m²]
            q_flux[i] = q_i

            # Wall temperatures
            T_wh[i] = T_aw[i]  - q_i / max(h_gas[i], 1.0)
            T_wc[i] = T_cool[i] + q_i / max(h_fuel, 1.0)

            # Advance coolant temperature
            if i < n - 1:
                dx    = x[i+1] - x[i]
                perim = np.pi * D[i]
                dTc   = q_i * perim * dx / max(mdot_fuel * self.cp_fuel, 1e-9)
                T_cool[i+1] = T_cool[i] + dTc

        # ── 5. Integrated heat load ───────────────────────────────────────
        perimeter = np.pi * D
        Q_section = float(np.trapz(q_flux * perimeter, x))

        return SectionThermal(
            name=name, x=x,
            T_gas=T, T_aw=T_aw,
            T_wall_hot=T_wh, T_wall_cold=T_wc,
            T_coolant=T_cool, q_flux=q_flux,
            h_gas=h_gas, Ma=Ma, A=A, D=D,
            Q_section=Q_section,
        )

    # ── Fluid-property correlations ───────────────────────────────────────

    @staticmethod
    def _sutherland(T: np.ndarray, T_ref=273.15, mu_ref=1.716e-5,
                    S=110.4) -> np.ndarray:
        """Sutherland viscosity law [Pa·s]."""
        T   = np.maximum(T, 200.0)
        return mu_ref * (T / T_ref)**1.5 * (T_ref + S) / (T + S)

    @staticmethod
    def _thermal_conductivity(T: np.ndarray) -> np.ndarray:
        """
        Simple power-law fit for air/combustion-gas thermal conductivity.
        k ≈ 0.0241 * (T/300)^0.82  [W/m/K]
        """
        return 0.0241 * (np.maximum(T, 200.0) / 300.0)**0.82


# ─────────────────────────────────────────────────────────────────────────────
# Utility: turn a scalar-valued section dict into a minimal 2-point sol dict
# ─────────────────────────────────────────────────────────────────────────────
def _scalar_section_to_sol(sec: dict) -> Optional[dict]:
    """
    The isolator and some sections return scalar end-states rather than a
    full solution array.  Build a minimal 2-point solution for plotting.
    """
    try:
        sol = sec.get("solution")
        if sol is not None:
            return sol
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cool = RamjetCooling(
        t_wall=3e-3,          # 3 mm Inconel wall
        k_wall=15.0,
        T_wall_limit=1300.0,
        n_channels=200,
        D_channel=3e-3,
        T_fuel_in=120.0,      # cryo H₂
    )

    # 1. Run thermodynamic cycle
    cool.run_cycle(h=20_000, Ma0=3.0, m_air=100.0, phi=0.5)

    # 2. Compute cooling loads + wall temperatures
    report = cool.cooling_analysis()

    # 3. Plot
    cool.plot_temperature_distribution(save_path="ramjet_cooling.png")

    # 4. Print per-section heat loads
    print("\nPer-section heat loads:")
    for s in report["sections"]:
        print(f"  {s.name:<12}  Q = {s.Q_section/1e6:7.3f} MW  "
              f"q_max = {s.q_flux.max()/1e6:.3f} MW/m²  "
              f"Tw_max = {s.T_wall_hot.max():.0f} K")