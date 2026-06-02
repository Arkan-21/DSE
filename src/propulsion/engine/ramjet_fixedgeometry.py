"""
ramjet_fixedgeometry.py
=======================
1-D quasi-steady Scramjet / Ramjet cycle model using the Shapiro
generalised-1D ODE with NASA 7-coefficient thermochemistry.

All user-adjustable parameters are collected in two dataclasses at the
top of the file:
  - Geometry   : physical dimensions of the engine
  - Assumptions: flight conditions and aerothermodynamic parameters

The remaining code (thermochemistry, ODE integrator, section solvers,
performance, plotting) is read-only infrastructure.

Station map
-----------
  0   Freestream / inlet face
  1   Isolator inlet           (A1 computed from flight conditions)
  2   Isolator exit            (A2)
  3   Diverging combustor exit (A3)  — simultaneous mass + heat addition, A2→A3
  4   Constant-area comb. exit (A4 = A3)  — continuing combustion, no mass addition
  5   Nozzle throat            (A_th, computed)
  6   Nozzle exit              (A6)

Sections
--------
  0→1  Intake / inlet compression            station_0 / station_1
  1→2  Isolator duct                         section_12
  2→3  Diverging combustor  (mass + heat)    section_23
  3→4  Constant-area comb.  (heat only)      section_34
  4→5  Nozzle convergent                     section_45
  5→6  Nozzle divergent                      section_56
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


# Optional NASA CEA — required by combustion sections.
try:
    import cea as _CEA           # noqa: F401
    _HAS_CEA = True
except Exception:                # noqa: BLE001
    _HAS_CEA = False


# ===========================================================================
# PARAMETERS — edit only this section
# ===========================================================================

@dataclass
class Geometry:
    """
    Physical dimensions of the engine.  All lengths [m], areas [m²].

    Station areas
    -------------
    A0   Inlet capture face
    A1   Isolator inlet          — derived at run-time, not set here
    A2   Isolator exit           — start of combustion zone
    A3   Diverging combustor exit / constant-area combustor inlet
    A4   Constant-area combustor exit  — must equal A3 (constant section)
    A5   Nozzle throat           — derived at run-time from Tt4, Pt4
    A6   Nozzle exit
    """

    # ── Inlet ───────────────────────────────────────────────────────────────
    A0: float = 8.0          # Inlet capture area                  [m²]

    # ── Section lengths ──────────────────────────────────────────────────────
    L01: float = 0.60        # Inlet / compression ramp            [m]
    L12: float = 0.15        # Isolator duct                       [m]
    L23: float = 0.30        # Diverging combustor                 [m]
    L34: float = 0.15        # Constant-area combustor             [m]
    L45: float = 0.40        # Nozzle convergent                   [m]
    L56: float = 1.20        # Nozzle divergent                    [m]

    # ── Station areas ────────────────────────────────────────────────────────
    # A1 is computed; A5 (throat) is computed. All others are fixed inputs.
    A2: float = 0.51         # Isolator exit                       [m²]
    A3: float = 0.65         # Diverging combustor exit            [m²]
    A4: float = 0.65         # Constant-area comb. exit (= A3)     [m²]
    A6: float = 16.0         # Nozzle exit                         [m²]


@dataclass
class Assumptions:
    """
    Flight conditions and aerothermodynamic modelling parameters.
    """

    # ── Flight conditions ────────────────────────────────────────────────────
    h0:  float = 30_000.0    # Altitude                            [m]
    Ma0: float = 6.0         # Flight Mach number                  [—]

    # ── Combustion ───────────────────────────────────────────────────────────
    phi:             float = 0.5       # Equivalence ratio φ        [—]
    theta:           float = 90.0     # Injection angle             [deg]
    mixing_coeff:    float = 0.176    # η curve coefficient (Li et al. 2023)
    # Note: validation script uses 0.11 (re-fitted to E2R geometry).

    # ── Aerothermodynamic constants ──────────────────────────────────────────
    Ma_COMB: float = 0.30    # Combustor-inlet Mach (subsonic ramjet) [—]
    Cf:      float = 0.003   # Skin-friction coefficient              [—]

    # ── Fuel properties (H₂ default) ────────────────────────────────────────
    HHV:        float = 141.8e6       # Fuel higher heating value [J/kg_fuel] (H₂)
    FAR_stoich: float = 1.0 / 34.35   # Stoichiometric fuel-to-air ratio   (H₂/air)


# ===========================================================================
# INFRASTRUCTURE — do not edit below
# ===========================================================================

# ---------------------------------------------------------------------------
# Thermochemistry — NASA 7-coefficient polynomials
# ---------------------------------------------------------------------------
class AirProperties:
    R_UNIVERSAL = 8.314462618  # J/(mol·K)

    NASA_DATA = {
        "N2": {
            "Trange": [200, 1000, 6000],
            "low":  [3.53100528, -1.23660988e-04, -5.02999433e-07,
                     2.43530612e-09, -1.40881235e-12, -1046.97628, 2.96747038],
            "high": [2.95257637,  1.39690040e-03, -4.92631603e-07,
                     7.86010195e-11, -4.60755204e-15, -923.948688, 5.87188762],
        },
        "O2": {
            "Trange": [200, 1000, 6000],
            "low":  [3.78245636, -2.99673416e-03,  9.84730201e-06,
                    -9.68129509e-09,  3.24372837e-12, -1063.94356, 3.65767573],
            "high": [3.69757819,  6.13519689e-04, -1.25884199e-07,
                     1.77528148e-11, -1.13643531e-15, -1233.93018, 3.18916559],
        },
        "Ar": {
            "Trange": [200, 1000, 6000],
            "low":  [2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.37967491],
            "high": [2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.37967491],
        },
        "CO2": {
            "Trange": [200, 1000, 6000],
            "low":  [2.35677352,  8.98459677e-03, -7.12356269e-06,
                     2.45919022e-09, -1.43699548e-13, -48371.9697, 9.90105222],
            "high": [4.63659493,  2.74146460e-03, -9.95897590e-07,
                     1.60391600e-10, -9.16198400e-15, -49024.9341, -1.93534855],
        },
        "H2O": {
            "Trange": [200, 1000, 6000],
            "low":  [4.19864056, -2.03643410e-03,  6.52040211e-06,
                    -5.48797062e-09,  1.77197250e-12, -30293.7267, -0.849032208],
            "high": [2.67703890,  2.97318160e-03, -7.73768890e-07,
                     9.44334890e-11, -4.26900770e-15, -29885.8940,  6.88255571],
        },
        "N": {
            "Trange": [200, 1000, 6000],
            "low":  [2.5, 0.0, 0.0, 0.0, 0.0, 56104.6378, 4.19390932],
            "high": [2.41594290,  1.74890650e-04, -1.19023690e-07,
                     3.02262450e-11, -2.03609820e-15, 56133.7730, 4.64960941],
        },
        "O": {
            "Trange": [200, 1000, 6000],
            "low":  [3.16826710, -3.27931884e-03,  6.64306396e-06,
                    -6.12806624e-09,  2.11265971e-12, 29122.2592, 2.05193346],
            "high": [2.54363697, -2.73162486e-05, -4.19029520e-09,
                     4.95481845e-12, -4.79553694e-16, 29226.0120, 4.92229457],
        },
        "NO": {
            "Trange": [200, 1000, 6000],
            "low":  [4.21859896, -4.63988124e-03,  1.10443049e-05,
                    -9.34055507e-09,  2.80554874e-12, 9845.09964, 2.28061001],
            "high": [3.26071234,  1.19101135e-03, -4.29122646e-07,
                     6.94481463e-11, -4.03295681e-15, 9921.43132, 6.36900518],
        },
        "H2": {
            "Trange": [200, 1000, 6000],
            "low":  [2.34433112,  7.98052075e-03, -1.94781510e-05,
                     2.01572094e-08, -7.37611761e-12, -917.935173, 0.683010238],
            "high": [2.93286575,  8.26607967e-04, -1.46402364e-07,
                     1.54100414e-11, -6.88804800e-16, -813.065581, -1.02432865],
        },
        "H": {
            "Trange": [200, 1000, 6000],
            "low":  [2.5, 0, 0, 0, 0, 25471.6270, -0.448813240],
            "high": [2.5, 0, 0, 0, 0, 25471.6270, -0.448813240],
        },
        "OH": {
            "Trange": [200, 1000, 6000],
            "low":  [3.99198424, -2.40106655e-03,  4.61664033e-06,
                    -3.87916306e-09,  1.36319502e-12, 3368.89836, -0.103998477],
            "high": [2.83853033,  1.10741289e-03, -2.94000209e-07,
                     4.20698729e-11, -2.42289890e-15, 3697.80808,  5.84494652],
        },
    }

    AIR_BASE_COMPOSITION = {
        "N2": 0.78084, "O2": 0.20946, "Ar": 0.00934, "CO2": 0.000407,
    }

    MOLECULAR_WEIGHTS = {
        "N2": 28.014, "O2": 31.998, "Ar": 39.948, "CO2": 44.010,
        "H2O": 18.015, "N": 14.007, "O": 15.999, "NO": 30.006,
        "H2": 2.016, "H": 1.008, "OH": 17.008,
    }

    def _nasa_coeffs(self, species, T):
        data = self.NASA_DATA[species]
        return np.array(data["low"] if T <= data["Trange"][1] else data["high"])

    def cp_over_R(self, species, T):
        a = self._nasa_coeffs(species, T)
        return a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4

    def h_over_RT(self, species, T):
        a = self._nasa_coeffs(species, T)
        return a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T

    def s_over_R(self, species, T):
        a = self._nasa_coeffs(species, T)
        return a[0]*np.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6]

    def gibbs_over_RT(self, species, T, P_atm):
        return self.h_over_RT(species, T) - self.s_over_R(species, T) + np.log(P_atm)

    def equilibrium_constants(self, T):
        dg1 = 2*self.gibbs_over_RT("N",  T, 1) - self.gibbs_over_RT("N2", T, 1)
        dg2 = 2*self.gibbs_over_RT("O",  T, 1) - self.gibbs_over_RT("O2", T, 1)
        dg3 = (self.gibbs_over_RT("NO", T, 1)
               - self.gibbs_over_RT("N", T, 1) - self.gibbs_over_RT("O", T, 1))
        return np.exp(-dg1), np.exp(-dg2), np.exp(-dg3)

    def equilibrium_composition(self, T, P_atm):
        if T < 1500.0:
            total = sum(self.AIR_BASE_COMPOSITION.values())
            return {k: v/total for k, v in self.AIR_BASE_COMPOSITION.items()}
        Kp1, Kp2, Kp3 = self.equilibrium_constants(T)
        x_N2_0 = self.AIR_BASE_COMPOSITION["N2"]
        x_O2_0 = self.AIR_BASE_COMPOSITION["O2"]
        x_Ar   = self.AIR_BASE_COMPOSITION["Ar"]
        x_CO2  = self.AIR_BASE_COMPOSITION["CO2"]
        N_at   = 2*x_N2_0;  O_at = 2*x_O2_0

        def eqs(v):
            xN2, xO2, xN, xO, xNO = v
            return [Kp1*xN2 - xN**2*P_atm,
                    Kp2*xO2 - xO**2*P_atm,
                    Kp3*xN*xO*P_atm - xNO,
                    2*xN2 + xN + xNO - N_at,
                    2*xO2 + xO + xNO - O_at]

        xN2, xO2, xN, xO, xNO = np.abs(fsolve(eqs, [x_N2_0*.9, x_O2_0*.9, 1e-6, 1e-6, 1e-6]))
        xT = xN2 + xO2 + xN + xO + xNO + x_Ar + x_CO2
        return {"N2": xN2/xT, "O2": xO2/xT, "N": xN/xT, "O": xO/xT,
                "NO": xNO/xT, "Ar": x_Ar/xT, "CO2": x_CO2/xT}

    def mixture_cp_cv(self, T, P_atm):
        comp = self.equilibrium_composition(T, P_atm)
        MW   = sum(comp[s]*self.MOLECULAR_WEIGHTS[s] for s in comp)
        cp_m = sum(comp[s]*self.cp_over_R(s, T)*self.R_UNIVERSAL for s in comp)
        cp   = cp_m / (MW*1e-3)
        R_s  = self.R_UNIVERSAL / (MW*1e-3)
        return cp, cp - R_s, cp / (cp - R_s)

    def specific_heat_ratio(self, T, P): return self.mixture_cp_cv(T, P/101325)[2]
    def specific_cp(self, T, P):        return self.mixture_cp_cv(T, P/101325)[0]
    def specific_R(self, T, P):
        cp, cv, _ = self.mixture_cp_cv(T, P/101325); return cp - cv


# ---------------------------------------------------------------------------
# Frozen-mixture thermodynamics
# ---------------------------------------------------------------------------
class MixtureNASA:
    R_UNIVERSAL = 8.314462618
    P_REF       = 101325.0

    def __init__(self, air_props: AirProperties):
        self.air = air_props
        self.W   = {s: air_props.MOLECULAR_WEIGHTS[s]*1e-3 for s in air_props.MOLECULAR_WEIGHTS}

    def h_i(self, s, T):  return self.air.h_over_RT(s, T) * self.R_UNIVERSAL * T / self.W[s]
    def cp_i(self, s, T): return self.air.cp_over_R(s, T) * self.R_UNIVERSAL / self.W[s]
    def s0_i(self, s, T): return self.air.s_over_R(s, T) * self.R_UNIVERSAL / self.W[s]

    def W_mix(self, Y):
        inv = sum(y / self.W[sp] for sp, y in Y.items() if y > 0)
        return 1.0 / max(inv, 1e-30)

    def X_from_Y(self, Y):
        Wm = self.W_mix(Y)
        return {sp: (y / self.W[sp]) * Wm for sp, y in Y.items()}

    def cp_mix(self, Y, T):  return sum(y * self.cp_i(sp, T) for sp, y in Y.items() if y > 0)
    def h_mix(self, Y, T):   return sum(y * self.h_i(sp, T)  for sp, y in Y.items() if y > 0)

    def s_mix(self, Y, T, p_pa):
        X = self.X_from_Y(Y); s = 0.0
        for sp, y in Y.items():
            if y <= 0: continue
            R_i = self.R_UNIVERSAL / self.W[sp]
            s  += y * (self.s0_i(sp, T) - R_i * np.log(max(X[sp], 1e-30) * p_pa / self.P_REF))
        return s

    def gamma_mix(self, Y, T):
        cp = self.cp_mix(Y, T); Wm = self.W_mix(Y)
        return cp / max(cp - self.R_UNIVERSAL / Wm, 1e-30)

    def stagnation_Tt(self, Y, T_static, h_target, tol=1e-3, max_iter=60):
        Tt = max(float(T_static), 200.0)
        for _ in range(max_iter):
            h_now  = self.h_mix(Y, Tt); cp_now = self.cp_mix(Y, Tt)
            if cp_now <= 0: break
            delta  = max(-400.0, min(800.0, (h_target - h_now) / cp_now))
            Tt_new = max(200.0, min(6000.0, Tt + delta))
            if abs(Tt_new - Tt) < tol: Tt = Tt_new; break
            Tt = Tt_new
        return Tt

    def stagnation_Pt(self, Y, T_static, Tt, p_static):
        # Correct isentropic stagnation pressure: find Pt such that
        #   s_mix(Y, Tt, Pt) = s_mix(Y, T_static, p_static)
        # i.e. solve  s_mix(Tt, Pt) - s_ref = 0  by Newton on ln(Pt).
        # This uses the full entropy (temperature + pressure + mixing terms)
        # and is therefore correct for variable-composition real-gas mixtures.
        # The previous approximation Pt = p·exp(Δs°/R) dropped the pressure-
        # and mixing-entropy terms, causing Pt steps when composition changes.
        s_ref = self.s_mix(Y, T_static, p_static)
        Pt    = float(p_static)  # initial guess
        R_mix = self.R_UNIVERSAL / self.W_mix(Y)
        for _ in range(40):
            s_now  = self.s_mix(Y, Tt, Pt)
            resid  = s_now - s_ref
            if abs(resid) < 1e-4: break
            # d(s_mix)/d(ln Pt) = -R_mix  (ideal-gas pressure term)
            Pt = max(1.0, Pt * np.exp(resid / R_mix))
        return float(Pt)

    def stagnation_state(self, Y, T, p_pa, V):
        h = self.h_mix(Y, T); s = self.s_mix(Y, T, p_pa); ht = h + 0.5*V*V
        Tt = self.stagnation_Tt(Y, T, ht); Pt = self.stagnation_Pt(Y, T, Tt, p_pa)
        return {"h": h, "s": s, "ht": ht, "st": s, "Tt": Tt, "Pt": Pt}


# ---------------------------------------------------------------------------
# NASA CEA wrapper
# ---------------------------------------------------------------------------
class CEAComp:
    PROD_NAMES = ["Ar", "CO2", "H", "H2", "H2O", "N", "NO", "N2", "O", "O2", "OH"]

    def __init__(self):
        if not _HAS_CEA:
            raise ImportError("NASA CEA package not installed (`pip install cea`).")
        self.cea  = _CEA
        self.reac = _CEA.Mixture(["H2", "Air"])
        self.prod = _CEA.Mixture(self.PROD_NAMES)
        self.solver   = _CEA.EqSolver(self.prod, reactants=self.reac)
        self.solution = _CEA.EqSolution(self.solver)
        self._fuel_w  = self.reac.moles_to_weights(np.array([1.0, 0.0]))
        self._oxid_w  = self.reac.moles_to_weights(np.array([0.0, 1.0]))
        self._cache: dict = {}

    def equilibrium_Y(self, T, p_pa, of_ratio):
        key = (round(float(T)), round(float(p_pa)/10)*10, round(float(of_ratio)*100)/100)
        if key in self._cache: return self._cache[key]
        T_c = float(np.clip(T, 250.0, 5500.0)); p_c = max(float(p_pa), 50.0)
        of_c = max(float(of_ratio), 0.01); Y = None
        try:
            w = self.reac.of_ratio_to_weights(self._oxid_w, self._fuel_w, of_c)
            self.solver.solve(self.solution, self.cea.TP, T_c, p_c/1e5, w)
            if bool(self.solution.converged):
                mf = self.solution.mass_fractions
                Y  = {sp: float(mf.get(sp, 0.0)) for sp in self.PROD_NAMES}
                if abs(sum(Y.values()) - 1.0) > 0.05: Y = None
        except Exception: Y = None
        self._cache[key] = Y; return Y


# ---------------------------------------------------------------------------
# Standard atmosphere (ISA, ≤32 km)
# ---------------------------------------------------------------------------
class Atmosphere:
    R_AIR = 287.05; G0 = 9.80665

    @staticmethod
    def _layer(h):
        if   h <= 11000: return 0.0,   -0.0065, 288.15, 101325.0
        elif h <= 20000: return 11000,  0.0,    216.65,  22632.1
        elif h <= 32000: return 20000,  0.001,  216.65,   5474.89
        else: raise ValueError(f"Altitude {h:.0f} m > 32 km ceiling.")

    @staticmethod
    def T(h):
        h0, L, T0, _ = Atmosphere._layer(h); return T0 + L*(h - h0)

    @staticmethod
    def P(h):
        h0, L, T0, P0 = Atmosphere._layer(h); dh = h - h0; T = T0 + L*dh
        return P0*(T/T0)**(-Atmosphere.G0/(L*Atmosphere.R_AIR)) if L != 0 \
               else P0*np.exp(-Atmosphere.G0*dh/(Atmosphere.R_AIR*T0))

    @staticmethod
    def rho(h): return Atmosphere.P(h) / (Atmosphere.R_AIR * Atmosphere.T(h))


# ---------------------------------------------------------------------------
# Shapiro generalised-1D ODE
# ---------------------------------------------------------------------------
class ShapiroODE:
    @staticmethod
    def derivatives(Ma2, p, T, gamma, Cp, dA_dx, A, D, Cf,
                    dH_dx, mdot, dmdot_dx, W, dW_dx, dgamma_dx, switches=None):
        sw = switches or {}
        on = lambda k: 1.0 if sw.get(k, True) else 0.0
        g = gamma; M2 = Ma2
        D1 = 1.0 - M2
        if abs(D1) < 1e-8: D1 = 1e-8 if D1 >= 0 else -1e-8
        g1m2 = 1.0 + (g-1.0)/2.0*M2; gM2 = g*M2
        fric = 4.0*Cf/D; heat = dH_dx/(Cp*T)
        dMa2_dx = M2*(
            -(2.0*g1m2/D1)*(dA_dx/A)                      * on("area")
            + ((1.0+gM2)/D1)*heat                          * on("heat")
            + (gM2*g1m2/D1)*fric                           * on("friction")
            + (2.0*(1.0+gM2)*g1m2/D1)*(dmdot_dx/mdot)     * on("mass")
            - ((1.0+gM2)/D1)*(dW_dx/W)                    * on("MW")
            - (dgamma_dx/g)                                * on("gamma"))
        dp_dx = p*(
            (gM2/D1)*(dA_dx/A)                             * on("area")
            - (gM2/D1)*heat                                * on("heat")
            - (gM2*(1.0+(g-1.0)*M2)/(2.0*D1))*fric        * on("friction")
            - (2.0*gM2*g1m2/D1)*(dmdot_dx/mdot)           * on("mass")
            + (gM2/D1)*(dW_dx/W)                          * on("MW"))
        dT_dx = T*(
            ((g-1.0)*M2/D1)*(dA_dx/A)                     * on("area")
            + ((1.0+gM2)/D1)*heat                          * on("heat")
            - (g*(g-1.0)*M2**2/(2.0*D1))*fric             * on("friction")
            - ((g-1.0)*M2*(1.0+gM2)/D1)*(dmdot_dx/mdot)  * on("mass")
            + ((g-1.0)*M2/D1)*(dW_dx/W)                  * on("MW"))
        return dMa2_dx, dp_dx, dT_dx

    @staticmethod
    def integrate(x_start, x_end, Ma2_in, p_in, T_in, mdot_in,
                  geometry_fn, composition_fn, source_fn, mix,
                  state_fn=None, switches=None, Cf=0.003, n_steps=1000,
                  ht_in=None):
        R_UNIV = mix.R_UNIVERSAL
        Y_in   = composition_fn(x_start, T_in, p_in)
        cp_in  = mix.cp_mix(Y_in, T_in); W_in = mix.W_mix(Y_in)
        R_in   = R_UNIV/W_in; gamma_in = cp_in/max(cp_in-R_in, 1e-30)
        V2_in  = max(Ma2_in, 1e-10)*gamma_in*R_in*T_in
        # If the caller supplies ht_in (carried from the previous section's ODE
        # state), use it directly — never recompute from h_mix(T_in) + 0.5*V²,
        # because h_mix has a composition-dependent reference datum that shifts
        # between sections and would introduce a spurious step in ht, Tt, Pt.
        if ht_in is None:
            ht_in = mix.h_mix(Y_in, T_in) + 0.5*V2_in

        T_cache = {"T": float(T_in)}

        def solve_T(M2, p, ht, x, T_guess=None):
            T = max(200.0, min(6000.0, float(T_guess) if T_guess is not None else T_cache["T"]))
            Y = cp = W = R = gamma = V2 = None; last_T = T
            for _ in range(60):
                Y = composition_fn(x, T, p); cp = mix.cp_mix(Y, T)
                W = mix.W_mix(Y); R = R_UNIV/W; gamma = cp/max(cp-R, 1e-30)
                h = mix.h_mix(Y, T); V2 = M2*gamma*R*T
                resid = (h + 0.5*V2) - ht
                if abs(resid) < 1.0: break
                deriv = cp + 0.5*M2*gamma*R
                if deriv <= 0: break
                step  = max(-200.0, min(400.0, -resid/deriv))
                T_new = max(200.0, min(6000.0, T + step))
                if abs(T_new - last_T) < 0.5:
                    T = T_new; Y = composition_fn(x, T, p); cp = mix.cp_mix(Y, T)
                    W = mix.W_mix(Y); R = R_UNIV/W; gamma = cp/max(cp-R, 1e-30)
                    V2 = M2*gamma*R*T; break
                last_T = T; T = T_new
            T_cache["T"] = T
            return T, Y, cp, W, R, gamma, V2

        sw_heat = True if switches is None else switches.get("heat",  True)
        sw_mass = True if switches is None else switches.get("mass",  True)
        sw_MW   = True if switches is None else switches.get("MW",    True)
        sw_gam  = True if switches is None else switches.get("gamma", True)

        def rhs(x, y):
            M2, p, ht, mdot = y
            T, Y, cp, W, R, gamma, V2 = solve_T(M2, p, ht, x)
            A, dA_dx, D = geometry_fn(x)
            dH_dx, dmdot_dx = source_fn(x, T, p, mdot, Y)
            if not sw_heat: dH_dx    = 0.0
            if not sw_mass: dmdot_dx = 0.0
            dW_dx = dgamma_dx = 0.0
            if sw_MW or sw_gam:
                dx_s = 1e-4; x_p = min(x+dx_s, x_end); x_m = max(x-dx_s, x_start)
                span = x_p - x_m
                if span > 0:
                    Y_p = composition_fn(x_p, T, p); Y_m = composition_fn(x_m, T, p)
                    if sw_MW:  dW_dx     = (mix.W_mix(Y_p) - mix.W_mix(Y_m)) / span
                    if sw_gam: dgamma_dx = (mix.gamma_mix(Y_p, T) - mix.gamma_mix(Y_m, T)) / span
            dM2_dx, dp_dx, _ = ShapiroODE.derivatives(
                Ma2=M2, p=p, T=T, gamma=gamma, Cp=cp,
                dA_dx=dA_dx, A=A, D=D, Cf=Cf, dH_dx=dH_dx,
                mdot=mdot, dmdot_dx=dmdot_dx,
                W=W, dW_dx=dW_dx, dgamma_dx=dgamma_dx, switches=switches)
            return [dM2_dx, dp_dx, dH_dx, dmdot_dx]

        def choke_event(x, y):    return y[0] - 1.0
        choke_event.terminal = True; choke_event.direction = 1
        def pressure_event(x, y): return y[1] - 1.0
        pressure_event.terminal = True; pressure_event.direction = -1

        sol = solve_ivp(rhs, t_span=(x_start, x_end),
                        y0=[max(Ma2_in,1e-10), max(p_in,1.0), float(ht_in), max(mdot_in,1e-9)],
                        method="DOP853", rtol=1e-6, atol=1e-6,
                        max_step=(x_end-x_start)/50,
                        events=[choke_event, pressure_event], dense_output=False)

        xs = sol.t; M2s = np.maximum(sol.y[0], 1e-12); ps = np.maximum(sol.y[1], 1.0)
        hts_arr = sol.y[2]; mdots = np.maximum(sol.y[3], 1e-9); Mas = np.sqrt(M2s)
        thermal_choke = len(sol.t_events[0]) > 0
        if thermal_choke:
            print(f"\n  ℹ Thermal choking at x = {sol.t_events[0][0]:.4f} m  (Ma → 1)")

        T_cache["T"] = float(T_in)
        Ts = np.empty_like(xs); Vs = np.empty_like(xs)
        cps = np.empty_like(xs); gammas = np.empty_like(xs)
        Rs  = np.empty_like(xs); rhos   = np.empty_like(xs)
        for i in range(len(xs)):
            T_i, Y_i, cp_i, W_i, R_i, g_i, V2_i = solve_T(M2s[i], ps[i], hts_arr[i], xs[i])
            Ts[i] = T_i; Vs[i] = np.sqrt(max(V2_i, 0.0))
            cps[i] = cp_i; gammas[i] = g_i; Rs[i] = R_i
            rhos[i] = ps[i] / max(R_i*T_i, 1e-12)

        As = np.array([geometry_fn(x)[0] for x in xs])
        if state_fn is None:
            def state_fn(T, p, V, x):
                return mix.stagnation_state(composition_fn(x, T, p), T, p, V)

        hs = np.empty_like(xs); ss   = np.empty_like(xs)
        hts2 = np.empty_like(xs); sts2 = np.empty_like(xs)
        Tts = np.empty_like(xs); Pts  = np.empty_like(xs)
        for i in range(len(xs)):
            st = state_fn(Ts[i], ps[i], Vs[i], xs[i])
            hs[i] = st["h"]; ss[i] = st["s"]
            # ht is the conserved ODE state variable — read it directly from
            # the integrator output (hts_arr) rather than recomputing
            # h_mix(T) + 0.5*V² from the recovered T and V.  The round-trip
            # through solve_T has finite Newton tolerance that accumulates into
            # visible drift in ht, Tt, and Pt across adiabatic sections.
            hts2[i] = hts_arr[i]; sts2[i] = st["st"]
            Tts[i]  = st["Tt"]; Pts[i]  = st["Pt"]

        return {"x": xs, "Ma": Mas, "Ma2": M2s, "p": ps, "P": ps, "T": Ts,
                "rho": rhos, "V": Vs, "Tt": Tts, "Pt": Pts, "pt": Pts,
                "h": hs, "s": ss, "ht": hts2, "st": sts2,
                "A": As, "mdot": mdots,
                "thermal_choke": thermal_choke,
                "solver_success": sol.success, "solver_message": sol.message}


# ===========================================================================
# Ramjet Engine
# ===========================================================================
class Ramjet:
    """
    1-D scramjet/ramjet cycle model — 6 sections, fixed geometry.

    Usage
    -----
        eng  = Ramjet(geom=Geometry(...), assump=Assumptions(...))
        inp  = eng.station_0()
        iso  = eng.station_1(inp)
        sec2 = eng.section_12(iso)
        sec3 = eng.section_23(sec2)          # diverging comb — mass + heat
        sec4 = eng.section_34(sec3)          # const-area comb — heat only
        sec5 = eng.section_45(sec4)          # nozzle convergent
        sec6 = eng.section_56(sec5)          # nozzle divergent
        perf = eng.performance(inp, sec6, sec3)
    """

    def __init__(self, geom: Geometry | None = None, assump: Assumptions | None = None):
        self.geom   = geom   or Geometry()
        self.assump = assump or Assumptions()
        self.air        = AirProperties()
        self.mixture    = MixtureNASA(self.air)
        self.shapiroODE = ShapiroODE()
        self._cea_comp  = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_cea(self):
        if self._cea_comp is None:
            self._cea_comp = CEAComp()
        return self._cea_comp

    def _f(self, x): return float(np.asarray(x).squeeze())

    def _air_Y(self):
        m = self.air.AIR_BASE_COMPOSITION; t = sum(m.values())
        W = sum((m[s]/t) * self.air.MOLECULAR_WEIGHTS[s] for s in m)
        return {s: (m[s]/t) * self.air.MOLECULAR_WEIGHTS[s] / W for s in m}

    def _frozen_state_fn(self, Y_const):
        def state_fn(T, p, V, x): return self.mixture.stagnation_state(Y_const, T, p, V)
        return state_fn

    def _std_result(self, result, A_exit, extra=None):
        """Package the last point of an ODE result into a uniform dict."""
        d = {
            "Ma":  self._f(result["Ma"][-1]),
            "T":   self._f(result["T"][-1]),
            "P":   self._f(result["p"][-1]),
            "rho": self._f(result["rho"][-1]),
            "V":   self._f(result["V"][-1]),
            "Tt":  self._f(result["Tt"][-1]),
            "Pt":  self._f(result["Pt"][-1]),
            "h":   self._f(result["h"][-1]),
            "ht":  self._f(result["ht"][-1]),
            "s":   self._f(result["s"][-1]),
            "A":   A_exit,
            "mdot": self._f(result["mdot"][-1]),
            "thermal_choke": result["thermal_choke"],
            "solution": result,
        }
        if extra: d.update(extra)
        return d

    def _build_combustion_fns(self, sec_in, mdot_air, L_total_comb):
        """
        Build the source and composition functions shared by sections 2→3 and
        3→4, using the same approach as the validation script:

          - Composition  : frozen reactant mixture (air + unburnt H2) throughout.
                           No CEA blending inside the ODE — composition is fixed.
          - Heat release : dH/dx = YF * HHV * dη/dx   (mixing-efficiency profile,
                           same η formula as Li et al. 2023)
          - Mass addition: dṁ/dx = MFUEL / L_total_comb  — fuel spread uniformly
                           over the FULL combustion zone (L23 + L34), active in
                           BOTH sections.

        x_offset is added to the local ODE coordinate so η sees a single
        continuous coordinate anchored at the start of station 2.

        Returns
        -------
        Y_react      : frozen reactant composition dict
        mfuel_total  : total fuel mass flow [kg/s]
        make_fns(x_offset) → (composition_fn, source_fn, state_fn)
        """
        g   = self.geom
        a   = self.assump
        mix = self.mixture

        mfuel_total = a.phi * a.FAR_stoich * mdot_air
        YF          = mfuel_total / (mdot_air + mfuel_total)

        # Frozen reactant composition: air diluted by fuel mass fraction + H2
        Y_react = {sp: (1.0 - YF) * sec_in["Y"].get(sp, 0.0) for sp in sec_in["Y"]}
        Y_react["H2"] = YF

        theta = a.theta
        coeff = a.mixing_coeff

        # Fuel spread uniformly over the full combustion zone
        dmdot_dx_const = mfuel_total / L_total_comb

        def mixing_eta(x_global):
            s   = float(np.clip(x_global / L_total_comb, 1e-6, 1.0))
            e0  = s
            e90 = float(np.clip(1.01 + coeff * np.log(s), 0.0, 1.0))
            if theta == 0.0:  return e0
            if theta == 90.0: return e90
            return theta / 90.0 * (e90 - e0) + e0

        def deta_dx_global(x_global, dx=1e-5):
            lo = max(x_global - dx, 0.0)
            hi = min(x_global + dx, L_total_comb)
            return (mixing_eta(hi) - mixing_eta(lo)) / (2.0 * dx)

        def make_fns(x_offset):
            def composition_fn(x_local, T, p):
                return Y_react

            def source_fn(x_local, T, p, mdot_local, Y):
                x_g   = x_offset + x_local
                dH_dx = YF * a.HHV * deta_dx_global(x_g)
                return dH_dx, dmdot_dx_const

            def state_fn(T, p, V, x_local):
                return mix.stagnation_state(Y_react, T, p, V)

            return composition_fn, source_fn, state_fn

        return Y_react, mfuel_total, make_fns

    # =====================================================================
    # Station 0 — Freestream / inlet capture
    # =====================================================================
    def station_0(self):
        g = self.geom; a = self.assump
        T0 = Atmosphere.T(a.h0); P0 = Atmosphere.P(a.h0); rho0 = Atmosphere.rho(a.h0)
        Y_air = self._air_Y()
        cp0   = self.mixture.cp_mix(Y_air, T0); W_m = self.mixture.W_mix(Y_air)
        R0    = self.mixture.R_UNIVERSAL / W_m; gamma0 = self.mixture.gamma_mix(Y_air, T0)
        a0    = np.sqrt(gamma0*R0*T0); V0 = a.Ma0*a0; A0 = float(g.A0)
        m_air = rho0*V0*A0
        h0    = self.mixture.h_mix(Y_air, T0); s0 = self.mixture.s_mix(Y_air, T0, P0)
        ht0   = h0 + 0.5*V0**2
        Tt0   = self.mixture.stagnation_Tt(Y_air, T0, ht0)
        Pt0   = self.mixture.stagnation_Pt(Y_air, T0, Tt0, P0)
        print(f"\n── Station 0 — Inlet  h={a.h0/1e3:.0f} km  Ma={a.Ma0:.2f}  A0={A0:.4f} m² ──")
        print(f"  T0={T0:.1f} K   P0={P0:.0f} Pa   ṁ={m_air:.3f} kg/s   Tt0={Tt0:.1f} K")
        return {"Ma": a.Ma0, "T": T0, "P": P0, "rho": rho0,
                "gamma": gamma0, "cp": cp0, "R": R0, "a": a0,
                "V": V0, "A": A0, "Tt": Tt0, "Pt": Pt0,
                "h": h0, "ht": ht0, "s": s0, "st": s0, "Y": Y_air, "mdot": m_air}

    # =====================================================================
    # Station 1 — Isolator inlet  (MIL-E-5008B pressure recovery + energy)
    # =====================================================================
    def pressure_recovery(self, Ma):
        Ma = float(Ma)
        if Ma <= 1.0:  return 1.0
        elif Ma <= 5.0: return 1.0 - 0.075*(Ma-1.0)**1.35
        else:           return 800.0/(Ma**4 + 935.0)

    def station_1(self, inp):
        g = self.geom; a = self.assump; mix = self.mixture
        Y_air = inp["Y"]
        Ma0 = self._f(inp["Ma"]); T0  = self._f(inp["T"]); P0  = self._f(inp["P"])
        Pt0 = self._f(inp["Pt"]); ht0 = self._f(inp["ht"]); s0 = self._f(inp["s"])
        mdot = self._f(inp["mdot"]); A0 = self._f(inp["A"]); rho0 = self._f(inp["rho"])
        V0   = self._f(inp["V"])

        M1 = float(a.Ma_COMB)
        sigma_c = self.pressure_recovery(Ma0); Pt1_target = sigma_c * Pt0

        def residual(v):
            T1g, p1g = max(v[0], 250.0), max(v[1], 100.0)
            W1 = mix.W_mix(Y_air); R1 = mix.R_UNIVERSAL/W1
            g1 = mix.gamma_mix(Y_air, T1g)
            V1 = M1*np.sqrt(g1*R1*T1g)
            h1 = mix.h_mix(Y_air, T1g)
            Tt1g = mix.stagnation_Tt(Y_air, T1g, ht0)
            Pt1g = mix.stagnation_Pt(Y_air, T1g, Tt1g, p1g)
            return [ht0 - (h1 + 0.5*V1**2), Pt1g - Pt1_target]

        T1, P1 = fsolve(residual, [600.0, 0.3*P0])
        W1 = mix.W_mix(Y_air); R1 = mix.R_UNIVERSAL/W1
        g1 = mix.gamma_mix(Y_air, T1); cp1 = mix.cp_mix(Y_air, T1)
        V1 = M1*np.sqrt(g1*R1*T1); h1  = mix.h_mix(Y_air, T1)
        s1 = mix.s_mix(Y_air, T1, P1)
        # Isolator is adiabatic: ht is conserved from inlet. Carry ht0 directly
        # rather than recomputing h_mix(T1) + 0.5*V1², which would re-introduce
        # the datum-shift error at the next section boundary.
        ht1 = ht0
        Tt1 = mix.stagnation_Tt(Y_air, T1, ht1); Pt1 = mix.stagnation_Pt(Y_air, T1, Tt1, P1)
        rho1 = P1/(R1*T1); A1 = mdot/(rho1*V1)

        sol = {"x": np.array([0.0, g.L01]), "Ma": np.array([Ma0, M1]),
               "T": np.array([T0, T1]), "Tt": np.array([self._f(inp["Tt"]), Tt1]),
               "p": np.array([P0, P1]), "P":  np.array([P0, P1]),
               "pt": np.array([Pt0, Pt1]), "Pt": np.array([Pt0, Pt1]),
               "A": np.array([A0, A1]), "rho": np.array([rho0, rho1]),
               "V": np.array([V0, V1]), "mdot": np.array([mdot, mdot]),
               "h": np.array([self._f(inp["h"]), h1]), "s": np.array([s0, s1]),
               "ht": np.array([self._f(inp["ht"]), ht1]), "st": np.array([s0, s1])}
        print(f"\n── Station 1 — Isolator inlet ──")
        print(f"  σ_c={sigma_c:.4f}  Ma1={M1:.3f}  T1={T1:.1f} K  P1={P1:.0f} Pa  A1={A1:.4f} m²")
        return {"Ma": M1, "T": T1, "P": P1, "V": V1, "A": A1,
                "Tt": Tt1, "Pt": Pt1, "rho": rho1, "gamma": g1, "cp": cp1, "R": R1,
                "sigma_c": sigma_c, "mdot": mdot, "h": h1, "ht": ht1, "s": s1,
                "Y": Y_air, "solution": sol}

    # =====================================================================
    # Section 1→2 — Isolator duct  (area taper, friction, air only)
    # =====================================================================
    def section_12(self, iso, switches=None):
        g = self.geom; a = self.assump; mix = self.mixture
        A1 = self._f(iso["A"]); A2 = g.A2; L = g.L12
        Ma1 = self._f(iso["Ma"]); T1 = self._f(iso["T"]); p1 = self._f(iso["P"])
        mdot = self._f(iso["mdot"]); Y_air = iso["Y"]

        def geometry_fn(x):
            A = A1 + (A2-A1)*(x/L); return A, (A2-A1)/L, np.sqrt(4.0*A/np.pi)
        def composition_fn(x, T, p): return Y_air
        def source_fn(x, T, p, m, Y): return 0.0, 0.0

        result = self.shapiroODE.integrate(
            0.0, L, Ma1**2, p1, T1, mdot,
            geometry_fn, composition_fn, source_fn, mix,
            state_fn=self._frozen_state_fn(Y_air), switches=switches, Cf=a.Cf, n_steps=300,
            ht_in=self._f(iso["ht"]))

        T_end = result["T"][-1]
        print(f"\n── Section 1→2 — Isolator duct ──")
        print(f"  Ma2={result['Ma'][-1]:.3f}  T2={T_end:.1f} K  p2={result['p'][-1]:.0f} Pa")
        return self._std_result(result, A2, {
            "Y": Y_air,
            "gamma": mix.gamma_mix(Y_air, T_end),
            "cp":    mix.cp_mix(Y_air, T_end),
            "R":     mix.R_UNIVERSAL / mix.W_mix(Y_air),
        })

    # =====================================================================
    # Section 2→3 — Diverging combustor  (mass + heat, both uniform)
    # =====================================================================
    def section_23(self, sec2, switches=None):
        """
        Diverging area (A2 → A3).
        Mass and heat addition both active, spread uniformly over the full
        combustion zone L23 + L34.  Composition is frozen reactant mixture.
        Heat release uses the mixing-efficiency profile (Li et al. 2023).
        """
        g = self.geom; a = self.assump; mix = self.mixture
        A2 = self._f(sec2["A"]); A3 = g.A3; L = g.L23
        L_comb = g.L23 + g.L34

        Ma2      = self._f(sec2["Ma"]); T2 = self._f(sec2["T"]); p2 = self._f(sec2["P"])
        mdot_air = self._f(sec2["mdot"])

        Y_react, mfuel_total, make_fns = self._build_combustion_fns(
            sec2, mdot_air, L_comb)
        composition_fn, source_fn, state_fn = make_fns(x_offset=0.0)

        def geometry_fn(x):
            A = A2 + (A3-A2)*(x/L); return A, (A3-A2)/L, np.sqrt(4.0*A/np.pi)

        result = self.shapiroODE.integrate(
            0.0, L, Ma2**2, p2, T2, mdot_air,
            geometry_fn, composition_fn, source_fn, mix,
            state_fn=state_fn, switches=switches, Cf=a.Cf, n_steps=400,
            ht_in=self._f(sec2["ht"]))

        mfuel_actual = max(self._f(result["mdot"][-1]) - mdot_air, 0.0)
        T_exit = result["T"][-1]; p_exit = result["p"][-1]
        print(f"\n── Section 2→3 — Diverging combustor ──")
        print(f"  Ma3={result['Ma'][-1]:.3f}  T3={T_exit:.1f} K  p3={p_exit:.0f} Pa  "
              f"ṁ_fuel={mfuel_actual:.4f} kg/s")
        return self._std_result(result, A3, {
            "Y": Y_react, "Y_react": Y_react,
            "mfuel": mfuel_actual, "mfuel_total": mfuel_total,
            "mdot_air": mdot_air, "phi": a.phi,
            # Pass make_fns through so section_34 can call it with x_offset=L23
            # without re-running _build_combustion_fns on a corrupted Y.
            "_make_fns": make_fns,
        })

    # =====================================================================
    # Section 3→4 — Constant-area combustor  (mass + heat, both continuing)
    # =====================================================================
    def section_34(self, sec3, switches=None):
        """
        Constant area (A3 = A4).  Mass and heat addition both continue at the
        same uniform rates as section 2→3.  The η coordinate is offset by L23
        so it is continuous from station 2.  Composition stays frozen reactant.
        """
        g = self.geom; a = self.assump; mix = self.mixture
        A3 = self._f(sec3["A"]); A4 = g.A4; L = g.L34
        L_comb = g.L23 + g.L34

        Ma3      = self._f(sec3["Ma"]); T3 = self._f(sec3["T"]); p3 = self._f(sec3["P"])
        mdot     = self._f(sec3["mdot"]); mdot_air = self._f(sec3["mdot_air"])
        Y_react   = sec3["Y_react"]
        make_fns  = sec3["_make_fns"]   # reuse exactly — never rebuild from sec3["Y"]

        # x_offset=L23 makes the η coordinate continuous from section_23.
        composition_fn, source_fn, state_fn = make_fns(x_offset=g.L23)

        if abs(A4 - A3) > 1e-6:
            print(f"  ⚠ A4 ({A4:.4f}) ≠ A3 ({A3:.4f}) — section 3→4 is constant-area; "
                  f"check Geometry.A4 = Geometry.A3.")

        def geometry_fn(x):
            D = np.sqrt(4.0*A3/np.pi); return A3, 0.0, D

        result = self.shapiroODE.integrate(
            0.0, L, Ma3**2, p3, T3, mdot,
            geometry_fn, composition_fn, source_fn, mix,
            state_fn=state_fn, switches=switches, Cf=a.Cf, n_steps=400,
            ht_in=self._f(sec3["ht"]))

        T_exit = result["T"][-1]; p_exit = result["p"][-1]
        mfuel_added = max(self._f(result["mdot"][-1]) - mdot, 0.0)
        print(f"\n── Section 3→4 — Constant-area combustor ──")
        print(f"  Ma4={result['Ma'][-1]:.3f}  T4={T_exit:.1f} K  p4={p_exit:.0f} Pa  "
              f"Δṁ_fuel={mfuel_added:.4f} kg/s")
        return self._std_result(result, A4, {
            "Y": Y_react, "mdot_air": mdot_air, "phi": a.phi,
        })

    # =====================================================================
    # Section 4→5 — Nozzle convergent
    # =====================================================================
    def section_45(self, sec4, switches=None):
        """
        Subsonic deceleration from combustor-exit Mach to the throat (Ma=1).
        Area tapers linearly from A4 to the isentropic throat area A_th.
        """
        g = self.geom; a = self.assump; mix = self.mixture
        Y_nz = sec4["Y"]; W_nz = mix.W_mix(Y_nz); R_nz = mix.R_UNIVERSAL/W_nz

        Ma4 = self._f(sec4["Ma"]); T4 = self._f(sec4["T"]); p4 = self._f(sec4["P"])
        Tt4 = self._f(sec4["Tt"]); Pt4 = self._f(sec4["Pt"])
        mdot = self._f(sec4["mdot"]); A4 = self._f(sec4["A"]); L = g.L45

        cp4 = mix.cp_mix(Y_nz, T4); g4 = cp4/max(cp4 - R_nz, 1e-30)
        T_th   = Tt4 * 2.0/(g4+1.0)
        P_th   = Pt4 * (2.0/(g4+1.0))**(g4/(g4-1.0))
        rho_th = P_th/(R_nz*T_th); a_th = np.sqrt(g4*R_nz*T_th)
        A_th   = mdot/(rho_th*a_th)

        print(f"\n── Section 4→5 — Nozzle convergent ──")
        print(f"  A4={A4:.4f} m²  →  A_throat={A_th:.4f} m²   "
              f"T_th={T_th:.1f} K  P_th={P_th:.0f} Pa")

        def geometry_fn(x):
            A = A4 + (A_th-A4)*(x/L); return A, (A_th-A4)/L, np.sqrt(4.0*A/np.pi)
        def composition_fn(x, T, p): return Y_nz
        def source_fn(x, T, p, m, Y): return 0.0, 0.0

        result = self.shapiroODE.integrate(
            0.0, L, Ma4**2, p4, T4, mdot,
            geometry_fn, composition_fn, source_fn, mix,
            state_fn=self._frozen_state_fn(Y_nz), switches=switches, Cf=a.Cf, n_steps=200,
            ht_in=self._f(sec4["ht"]))

        print(f"  Ma5={result['Ma'][-1]:.4f}  T5={result['T'][-1]:.1f} K  "
              f"p5={result['p'][-1]:.0f} Pa  (throat)")
        return self._std_result(result, A_th, {"Y": Y_nz, "A_throat": A_th})

    # =====================================================================
    # Section 5→6 — Nozzle divergent
    # =====================================================================
    def section_56(self, sec5, switches=None):
        """
        Supersonic expansion from throat (Ma=1) to nozzle exit area A6.
        Starts at Ma = 1.001 to step past the sonic singularity.
        """
        g = self.geom; a = self.assump; mix = self.mixture
        Y_nz  = sec5["Y"]; A_th = self._f(sec5["A_throat"])
        A6    = g.A6; L = g.L56

        # Use throat conditions as the ODE inlet
        T_th  = self._f(sec5["T"]); P_th = self._f(sec5["P"])
        mdot  = self._f(sec5["mdot"])

        if A6 <= A_th:
            A6 = A_th * 4.0
            print(f"  ⚠ A6 ≤ A_throat — widened to {A6:.4f} m²")
        print(f"\n── Section 5→6 — Nozzle divergent ──")
        print(f"  A_throat={A_th:.4f} m²  →  A6={A6:.4f} m²   AR={A6/A_th:.2f}")

        def geometry_fn(x):
            A = A_th + (A6-A_th)*(x/L); return A, (A6-A_th)/L, np.sqrt(4.0*A/np.pi)
        def composition_fn(x, T, p): return Y_nz
        def source_fn(x, T, p, m, Y): return 0.0, 0.0

        result = self.shapiroODE.integrate(
            0.0, L, 1.001**2, P_th, T_th, mdot,
            geometry_fn, composition_fn, source_fn, mix,
            state_fn=self._frozen_state_fn(Y_nz), switches=switches, Cf=a.Cf, n_steps=300,
            ht_in=self._f(sec5["ht"]))

        Ma6 = self._f(result["Ma"][-1]); T6 = self._f(result["T"][-1])
        p6  = self._f(result["p"][-1]);  V6 = self._f(result["V"][-1])
        print(f"  Ma6={Ma6:.3f}  T6={T6:.1f} K  p6={p6:.0f} Pa  V6={V6:.1f} m/s")
        return self._std_result(result, A6, {"Y": Y_nz})

    # =====================================================================
    # Performance
    # =====================================================================
    def performance(self, inp, sec6, sec3):
        """
        Uninstalled internal thrust, specific impulse, specific thrust.
        sec6 is the nozzle-exit section (station 6).
        sec3 carries mfuel (fuel mass flow) from the combustion section.
        """
        if sec6.get("thermal_choke", False):
            return {"thermal_choke": True}

        V0       = self._f(inp["V"]);   p0 = self._f(inp["P"]); A0 = self._f(inp["A"])
        mdot_air = self._f(inp["mdot"])
        V6       = self._f(sec6["V"]);  p6 = self._f(sec6["P"]); A6 = self._f(sec6["A"])
        mdot6    = self._f(sec6["mdot"]); mfuel = self._f(sec3["mfuel"])

        Fin = mdot6*V6 + p6*A6 - mdot_air*V0 - p0*A0
        Isp = Fin / (mfuel + mdot_air) * 9.80665
        Ia  = Fin / mdot_air
        print(f"\n── Performance ──")
        print(f"  Fin={Fin:.1f} N   Isp={Isp:.1f} s   Ia={Ia:.1f} N·s/kg")
        return {"Fin": Fin, "Isp": Isp, "Ia": Ia,
                "mfuel": mfuel, "mdot_air": mdot_air, "A0": A0, "A6": A6,
                "thermal_choke": False}

    # =====================================================================
    # Plot flowpath
    # =====================================================================
    def plot_flowpath(self, inp, iso, sec2, sec3, sec4, sec5, sec6):
        sections_raw = [
            (iso,  "solution"),
            (sec2, "solution"),
            (sec3, "solution"),
            (sec4, "solution"),
            (sec5, "solution"),
            (sec6, "solution"),
        ]
        sec_labels = ["0→1 Intake", "1→2 Isolator",
                      "2→3 Div. comb.", "3→4 C-A comb.",
                      "4→5 Nozzle conv.", "5→6 Nozzle div."]

        processed = []; x_offset = 0.0
        for sec, key in sections_raw:
            sol = sec[key]
            p_arr  = sol.get("p",  sol.get("P"))
            pt_arr = sol.get("pt", sol.get("Pt", p_arr))
            xs = np.asarray(sol["x"]) + x_offset
            processed.append({
                "x": xs, "Ma": np.asarray(sol["Ma"]),
                "T": np.asarray(sol["T"]), "Tt": np.asarray(sol.get("Tt", sol["T"])),
                "p": np.asarray(p_arr),    "pt": np.asarray(pt_arr),
                "V": np.asarray(sol["V"]), "mdot": np.asarray(sol["mdot"]),
                "A": np.asarray(sol.get("A", np.full_like(sol["x"], np.nan))),
                "h": np.asarray(sol.get("h",  np.zeros_like(sol["x"]))),
                "s": np.asarray(sol.get("s",  np.zeros_like(sol["x"]))),
                "ht": np.asarray(sol.get("ht", np.zeros_like(sol["x"]))),
                "st": np.asarray(sol.get("st", np.zeros_like(sol["x"]))),
            })
            x_offset = xs[-1]

        def cat(f): return np.concatenate([s[f] for s in processed])
        x = cat("x"); Ma = cat("Ma"); T = cat("T"); Tt = cat("Tt")
        p = cat("p"); pt = cat("pt"); V = cat("V"); mdot = cat("mdot")
        h_s = cat("h"); ht_s = cat("ht"); s_s = cat("s"); st_s = cat("st")
        A_arr = cat("A")

        fig, axs = plt.subplots(7, 1, figsize=(12, 26), sharex=True)

        axs[0].plot(x, Ma, lw=2.5, color="black", label="Mach")
        axs[0].axhline(1.0, color="red", lw=1, ls=":", alpha=0.5, label="Sonic")
        axs[0].set_ylabel("Mach Number"); axs[0].legend(loc="best")

        axs[1].plot(x, T,  lw=2, color="tab:red",  label="Static T")
        axs[1].plot(x, Tt, lw=2, color="darkred",  ls="--", label="Total T")
        axs[1].set_ylabel("Temperature [K]"); axs[1].legend(loc="best")

        axs[2].plot(x, p/1e3,  lw=2, color="tab:green",  label="Static P")
        axs[2].plot(x, pt/1e3, lw=2, color="darkgreen",  ls="--", label="Total P")
        axs[2].set_ylabel("Pressure [kPa]"); axs[2].set_yscale("log"); axs[2].legend(loc="best")

        axs[3].plot(x, h_s/1e6,  lw=2, color="tab:orange",  label="Static h")
        axs[3].plot(x, ht_s/1e6, lw=2, color="saddlebrown", ls="--", label="Total h")
        axs[3].set_ylabel("Enthalpy [MJ/kg]"); axs[3].legend(loc="best")

        axs[4].plot(x, s_s,  lw=2, color="tab:cyan", label="Static s")
        axs[4].plot(x, st_s, lw=2, color="tab:blue", ls="--", label="Total s")
        axs[4].set_ylabel("Entropy [J/kg/K]"); axs[4].legend(loc="best")

        axs[5].plot(x, V, lw=2, color="tab:blue"); axs[5].set_ylabel("Velocity [m/s]")

        axs[6].plot(x, mdot, lw=2, color="tab:purple", label="ṁ")
        axs[6].set_ylabel("Mass Flow [kg/s]"); axs[6].set_xlabel("Engine axial position [m]")
        if np.all(np.isfinite(A_arr)) and np.nanmax(A_arr) > 0:
            r = np.sqrt(A_arr/np.pi); r_norm = r/np.nanmax(r)
            sc = 0.45*np.nanmax(mdot)
            axs[6].fill_between(x, -sc*r_norm, sc*r_norm, color="lightgray", alpha=0.35)
            axs[6].plot(x,  sc*r_norm, color="black", lw=1.2)
            axs[6].plot(x, -sc*r_norm, color="black", lw=1.2, label="Geometry")
        axs[6].legend(loc="best")

        boundaries = [s["x"][-1] for s in processed[:-1]]
        for ax in axs:
            ax.grid(True, which="both", alpha=0.3)
            for b in boundaries: ax.axvline(b, color="gray", ls="--", alpha=0.6)
        ylim = axs[0].get_ylim()
        for i, lbl in enumerate(sec_labels):
            xm = (processed[i]["x"][0] + processed[i]["x"][-1]) / 2
            axs[0].text(xm, ylim[1]*0.90, lbl, ha="center", weight="bold", fontsize=7.5)

        plt.tight_layout(); plt.show()


# ===========================================================================
# Pretty-printer
# ===========================================================================
def print_section(title, props, fields):
    w = 34
    print(f"\n{'─'*65}\n  {title}\n{'─'*65}")
    for label, key, unit, scale in fields:
        val = props.get(key, float("nan"))
        try:    print(f"  {label:<{w}} {val*scale:>12.4f}  {unit}")
        except: print(f"  {label:<{w}} {'nan':>12}  {unit}")
    print(f"{'─'*65}")


# ===========================================================================
# Standalone utility
# ===========================================================================
def temperature_distribution(*sections):
    """
    Return the static temperature distribution along the engine axis.

    Pass section result dicts in order: iso, sec2, sec3, sec4, sec5, sec6
    (any trailing sections may be omitted).

    Returns
    -------
    x : np.ndarray  — axial position [m] from isolator inlet
    T : np.ndarray  — static temperature [K]
    """
    x_out, T_out, x_offset = [], [], 0.0
    for sec in sections:
        sol = sec["solution"]
        x_local = np.asarray(sol["x"])
        x_out.append(x_local + x_offset)
        T_out.append(np.asarray(sol["T"]))
        x_offset += x_local[-1]
    return np.concatenate(x_out), np.concatenate(T_out)


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":

    # ── Configure here ───────────────────────────────────────────────────────
    geom = Geometry(
        A0  = 6.0,      # Inlet capture area                      [m²]
        # Section lengths
        L01 = 0.60,     # Inlet / compression ramp                [m]
        L12 = 0.15,     # Isolator duct                           [m]
        L23 = 0.30,     # Diverging combustor                     [m]
        L34 = 0.15,     # Constant-area combustor                 [m]
        L45 = 0.40,     # Nozzle convergent                       [m]
        L56 = 1.20,     # Nozzle divergent                        [m]
        # Station areas (A1 and A5/throat computed at run-time)
        A2  = 0.51,     # Isolator exit                           [m²]
        A3  = 0.65,     # Diverging combustor exit                [m²]
        A4  = 0.65,     # Constant-area combustor exit (= A3)     [m²]
        A6  = 16.0,     # Nozzle exit                             [m²]
    )

    assump = Assumptions(
        h0           = 30_000.0,  # Altitude                      [m]
        Ma0          = 6.0,       # Flight Mach number            [—]
        phi          = 0.5,       # Equivalence ratio             [—]
        theta        = 90.0,      # Injection angle               [deg]
        mixing_coeff = 0.176,     # η curve coefficient           [—]
        Ma_COMB      = 0.30,      # Combustor-inlet Mach          [—]
        Cf           = 0.003,     # Skin-friction coefficient     [—]
        HHV          = 141.8e6,   # H₂ higher heating value       [J/kg_fuel]
    )
    # ────────────────────────────────────────────────────────────────────────

    print(f"\n{'═'*65}")
    print(f"  SCRAMJET/RAMJET  —  H₂ fuel  φ={assump.phi}  "
          f"h={assump.h0/1e3:.0f} km  Ma₀={assump.Ma0}")
    print(f"{'═'*65}")

    eng  = Ramjet(geom=geom, assump=assump)
    inp  = eng.station_0()
    iso  = eng.station_1(inp)
    sec2 = eng.section_12(iso)
    sec3 = eng.section_23(sec2)          # diverging comb — mass + heat
    sec4 = eng.section_34(sec3)          # const-area comb — heat only
    sec5 = eng.section_45(sec4)          # nozzle convergent
    sec6 = eng.section_56(sec5)          # nozzle divergent
    perf = eng.performance(inp, sec6, sec3)

    print_section("Station 0 — Freestream", inp, [
        ("Mach number",        "Ma",   "—",      1.0),
        ("Static temperature", "T",    "K",      1.0),
        ("Static pressure",    "P",    "kPa",    1e-3),
        ("Velocity",           "V",    "m/s",    1.0),
        ("Inlet area",         "A",    "m²",     1.0),
        ("Mass flow",          "mdot", "kg/s",   1.0),
        ("Total temperature",  "Tt",   "K",      1.0),
        ("Total pressure",     "Pt",   "kPa",    1e-3),
    ])
    print_section("Station 1 — Isolator inlet", iso, [
        ("Mach number",        "Ma",      "—",   1.0),
        ("Static temperature", "T",       "K",   1.0),
        ("Static pressure",    "P",       "kPa", 1e-3),
        ("Total temperature",  "Tt",      "K",   1.0),
        ("Total pressure",     "Pt",      "kPa", 1e-3),
        ("Pressure recovery",  "sigma_c", "—",   1.0),
        ("Isolator inlet area","A",       "m²",  1.0),
    ])
    print_section("Station 3 — Diverging combustor exit", sec3, [
        ("Mach number",        "Ma",    "—",    1.0),
        ("Static temperature", "T",     "K",    1.0),
        ("Static pressure",    "P",     "kPa",  1e-3),
        ("Total temperature",  "Tt",    "K",    1.0),
        ("Total pressure",     "Pt",    "kPa",  1e-3),
        ("Mass flow (total)",  "mdot",  "kg/s", 1.0),
        ("Fuel mass flow",     "mfuel", "kg/s", 1.0),
    ])
    print_section("Station 4 — Constant-area combustor exit", sec4, [
        ("Mach number",        "Ma",  "—",     1.0),
        ("Static temperature", "T",   "K",     1.0),
        ("Static pressure",    "P",   "kPa",   1e-3),
        ("Total temperature",  "Tt",  "K",     1.0),
        ("Total pressure",     "Pt",  "kPa",   1e-3),
        ("Static enthalpy",    "h",   "MJ/kg", 1e-6),
        ("Total enthalpy",     "ht",  "MJ/kg", 1e-6),
    ])
    print_section("Station 5 — Nozzle throat", sec5, [
        ("Mach number",        "Ma",  "—",   1.0),
        ("Static temperature", "T",   "K",   1.0),
        ("Static pressure",    "P",   "kPa", 1e-3),
        ("Throat area",        "A",   "m²",  1.0),
    ])
    print_section("Station 6 — Nozzle exit", sec6, [
        ("Mach number",        "Ma",  "—",   1.0),
        ("Static temperature", "T",   "K",   1.0),
        ("Static pressure",    "P",   "kPa", 1e-3),
        ("Velocity",           "V",   "m/s", 1.0),
        ("Total temperature",  "Tt",  "K",   1.0),
        ("Total pressure",     "Pt",  "kPa", 1e-3),
        ("Exit area",          "A",   "m²",  1.0),
    ])
    print_section("PERFORMANCE", perf, [
        ("Internal thrust Fin",  "Fin",      "N",      1.0),
        ("Specific impulse Isp", "Isp",      "s",      1.0),
        ("Specific thrust Ia",   "Ia",       "N·s/kg", 1.0),
        ("Air mass flow",        "mdot_air", "kg/s",   1.0),
        ("Inlet area A0",        "A0",       "m²",     1.0),
        ("Exit area A6",         "A6",       "m²",     1.0),
    ])
    print(f"\n  φ={assump.phi}  "
          f"FAR={sec3['mfuel']/inp['mdot']:.5f}  "
          f"ṁ_fuel={sec3['mfuel']:.4f} kg/s  "
          f"ṁ_air={inp['mdot']:.4f} kg/s")

    eng.plot_flowpath(inp, iso, sec2, sec3, sec4, sec5, sec6)

    # Temperature distribution example
    x_T, T_dist = temperature_distribution(iso, sec2, sec3, sec4, sec5, sec6)