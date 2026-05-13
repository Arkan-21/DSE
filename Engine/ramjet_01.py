from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


# Optional NASA CEA — required by combustor_properties4 (per-step equilibrium).
try:
    import cea as _CEA           # noqa: F401
    _HAS_CEA = True
except Exception:                # noqa: BLE001
    _HAS_CEA = False


# ---------------------------------------------------------------------------
# Thermochemistry data + simple air dissociation model
# ---------------------------------------------------------------------------
class AirProperties:
    R_UNIVERSAL = 8.314462618  # J/(mol·K)

    NASA_DATA = {
        "N2": {
            "Trange": [200, 1000, 6000],
            "low": [3.53100528, -1.23660988e-04, -5.02999433e-07,
                    2.43530612e-09, -1.40881235e-12, -1046.97628, 2.96747038],
            "high": [2.95257637, 1.39690040e-03, -4.92631603e-07,
                     7.86010195e-11, -4.60755204e-15, -923.948688, 5.87188762],
        },
        "O2": {
            "Trange": [200, 1000, 6000],
            "low": [3.78245636, -2.99673416e-03, 9.84730201e-06,
                    -9.68129509e-09, 3.24372837e-12, -1063.94356, 3.65767573],
            "high": [3.69757819, 6.13519689e-04, -1.25884199e-07,
                     1.77528148e-11, -1.13643531e-15, -1233.93018, 3.18916559],
        },
        "Ar": {
            "Trange": [200, 1000, 6000],
            "low":  [2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.37967491],
            "high": [2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.37967491],
        },
        "CO2": {
            "Trange": [200, 1000, 6000],
            "low": [2.35677352, 8.98459677e-03, -7.12356269e-06,
                    2.45919022e-09, -1.43699548e-13, -48371.9697, 9.90105222],
            "high": [4.63659493, 2.74146460e-03, -9.95897590e-07,
                     1.60391600e-10, -9.16198400e-15, -49024.9341, -1.93534855],
        },
        "H2O": {
            "Trange": [200, 1000, 6000],
            "low": [4.19864056, -2.03643410e-03, 6.52040211e-06,
                    -5.48797062e-09, 1.77197250e-12, -30293.7267, -0.849032208],
            "high": [2.67703890, 2.97318160e-03, -7.73768890e-07,
                     9.44334890e-11, -4.26900770e-15, -29885.8940, 6.88255571],
        },
        "N": {
            "Trange": [200, 1000, 6000],
            "low":  [2.5, 0.0, 0.0, 0.0, 0.0, 56104.6378, 4.19390932],
            "high": [2.41594290, 1.74890650e-04, -1.19023690e-07,
                     3.02262450e-11, -2.03609820e-15, 56133.7730, 4.64960941],
        },
        "O": {
            "Trange": [200, 1000, 6000],
            "low": [3.16826710, -3.27931884e-03, 6.64306396e-06,
                    -6.12806624e-09, 2.11265971e-12, 29122.2592, 2.05193346],
            "high": [2.54363697, -2.73162486e-05, -4.19029520e-09,
                     4.95481845e-12, -4.79553694e-16, 29226.0120, 4.92229457],
        },
        "NO": {
            "Trange": [200, 1000, 6000],
            "low": [4.21859896, -4.63988124e-03, 1.10443049e-05,
                    -9.34055507e-09, 2.80554874e-12, 9845.09964, 2.28061001],
            "high": [3.26071234, 1.19101135e-03, -4.29122646e-07,
                     6.94481463e-11, -4.03295681e-15, 9921.43132, 6.36900518],
        },
        "H2": {
            "Trange": [200, 1000, 6000],
            "low": [2.34433112, 7.98052075e-03, -1.94781510e-05,
                    2.01572094e-08, -7.37611761e-12, -917.935173, 0.683010238],
            "high": [2.93286575, 8.26607967e-04, -1.46402364e-07,
                     1.54100414e-11, -6.88804800e-16, -813.065581, -1.02432865],
        },
        "H": {
            "Trange": [200, 1000, 6000],
            "low":  [2.5, 0, 0, 0, 0, 25471.6270, -0.448813240],
            "high": [2.5, 0, 0, 0, 0, 25471.6270, -0.448813240],
        },
        "OH": {
            "Trange": [200, 1000, 6000],
            "low": [3.99198424, -2.40106655e-03, 4.61664033e-06,
                    -3.87916306e-09, 1.36319502e-12, 3368.89836, -0.103998477],
            "high": [2.83853033, 1.10741289e-03, -2.94000209e-07,
                     4.20698729e-11, -2.42289890e-15, 3697.80808, 5.84494652],
        },
    }

    AIR_BASE_COMPOSITION = {
        "N2": 0.78084,
        "O2": 0.20946,
        "Ar": 0.00934,
        "CO2": 0.000407,
    }

    MOLECULAR_WEIGHTS = {
        "N2": 28.014, "O2": 31.998, "Ar": 39.948, "CO2": 44.010,
        "H2O": 18.015, "N": 14.007, "O": 15.999, "NO": 30.006,
        "H2": 2.016,   "H": 1.008,   "OH": 17.008,
    }

    def _nasa_coeffs(self, species, T):
        data = self.NASA_DATA[species]
        return np.array(data["low"] if T <= data["Trange"][1] else data["high"])

    def cp_over_R(self, species, T):
        a = self._nasa_coeffs(species, T)
        return a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4

    def h_over_RT(self, species, T):
        a = self._nasa_coeffs(species, T)
        return (a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T)

    def s_over_R(self, species, T):
        a = self._nasa_coeffs(species, T)
        return (a[0]*np.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6])

    def gibbs_over_RT(self, species, T, P_atm):
        return self.h_over_RT(species, T) - self.s_over_R(species, T) + np.log(P_atm)

    def equilibrium_constants(self, T):
        dg1 = 2*self.gibbs_over_RT("N", T, 1) - self.gibbs_over_RT("N2", T, 1)
        dg2 = 2*self.gibbs_over_RT("O", T, 1) - self.gibbs_over_RT("O2", T, 1)
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
        N_atoms = 2*x_N2_0
        O_atoms = 2*x_O2_0

        def eqs(v):
            xN2, xO2, xN, xO, xNO = v
            return [
                Kp1*xN2 - xN**2*P_atm,
                Kp2*xO2 - xO**2*P_atm,
                Kp3*xN*xO*P_atm - xNO,
                2*xN2 + xN + xNO - N_atoms,
                2*xO2 + xO + xNO - O_atoms,
            ]

        x0 = [x_N2_0*0.9, x_O2_0*0.9, 1e-6, 1e-6, 1e-6]
        xN2, xO2, xN, xO, xNO = np.abs(fsolve(eqs, x0))
        xT = xN2 + xO2 + xN + xO + xNO + x_Ar + x_CO2
        return {
            "N2": xN2/xT, "O2": xO2/xT, "N": xN/xT,
            "O": xO/xT, "NO": xNO/xT, "Ar": x_Ar/xT, "CO2": x_CO2/xT,
        }

    def mixture_cp_cv(self, T, P_atm):
        comp = self.equilibrium_composition(T, P_atm)
        MW   = sum(comp[s]*self.MOLECULAR_WEIGHTS[s] for s in comp)
        cp_m = sum(comp[s]*self.cp_over_R(s, T)*self.R_UNIVERSAL for s in comp)
        cp   = cp_m / (MW*1e-3)
        R_s  = self.R_UNIVERSAL / (MW*1e-3)
        cv   = cp - R_s
        return cp, cv, cp/cv

    def specific_heat_ratio(self, T, P):
        return self.mixture_cp_cv(T, P/101325)[2]

    def specific_cp(self, T, P):
        return self.mixture_cp_cv(T, P/101325)[0]

    def specific_R(self, T, P):
        cp, cv, _ = self.mixture_cp_cv(T, P/101325)
        return cp - cv


# ---------------------------------------------------------------------------
# Generic frozen-mixture thermodynamics from NASA polynomials.
# Works for any mass-fraction dict over species that appear in AirProperties.NASA_DATA.
# Used to compute h, s and to find stagnation Tt, Pt in *integral* form
# (so variable Cp / composition is handled correctly).
# ---------------------------------------------------------------------------
class MixtureNASA:
    R_UNIVERSAL = 8.314462618   # J/(mol·K)
    P_REF       = 101325.0       # Pa (1 atm, NASA standard)

    def __init__(self, air_props: AirProperties):
        self.air = air_props
        # Pre-cache molecular weights (kg/mol)
        self.W = {s: air_props.MOLECULAR_WEIGHTS[s]*1e-3
                  for s in air_props.MOLECULAR_WEIGHTS}

    # ---- per-species ------------------------------------------------------
    def h_i(self, s, T):
        """h_i(T) [J/kg] — includes formation enthalpy via NASA a5."""
        return self.air.h_over_RT(s, T) * self.R_UNIVERSAL * T / self.W[s]

    def cp_i(self, s, T):
        """cp_i(T) [J/(kg·K)]."""
        return self.air.cp_over_R(s, T) * self.R_UNIVERSAL / self.W[s]

    def s0_i(self, s, T):
        """s°_i(T) at P_REF [J/(kg·K)]."""
        return self.air.s_over_R(s, T) * self.R_UNIVERSAL / self.W[s]

    # ---- mixture ----------------------------------------------------------
    def W_mix(self, Y):
        inv = 0.0
        for sp, y in Y.items():
            if y > 0:
                inv += y / self.W[sp]
        return 1.0 / max(inv, 1e-30)

    def X_from_Y(self, Y):
        Wm = self.W_mix(Y)
        return {sp: (y / self.W[sp]) * Wm for sp, y in Y.items()}

    def cp_mix(self, Y, T):
        return sum(y * self.cp_i(sp, T) for sp, y in Y.items() if y > 0)

    def h_mix(self, Y, T):
        return sum(y * self.h_i(sp, T) for sp, y in Y.items() if y > 0)

    def s_mix(self, Y, T, p_pa):
        """Specific entropy [J/(kg·K)] for an ideal gas mixture with Dalton's law.

        s = Σ Y_i [ s°_i(T)  −  R_i · ln(X_i · p / p_ref) ]
        """
        X = self.X_from_Y(Y)
        s = 0.0
        for sp, y in Y.items():
            if y <= 0:
                continue
            R_i = self.R_UNIVERSAL / self.W[sp]
            X_i = max(X[sp], 1e-30)
            s += y * (self.s0_i(sp, T) - R_i * np.log(X_i * p_pa / self.P_REF))
        return s

    def gamma_mix(self, Y, T):
        cp = self.cp_mix(Y, T)
        Wm = self.W_mix(Y)
        Rm = self.R_UNIVERSAL / Wm
        return cp / max(cp - Rm, 1e-30)

    # ---- stagnation solvers (integral form) -------------------------------
    def stagnation_Tt(self, Y, T_static, h_target, tol=1e-3, max_iter=60):
        """Solve h_mix(Y, Tt) = h_target via Newton iteration on Tt."""
        Tt = max(float(T_static), 200.0)
        for _ in range(max_iter):
            h_now  = self.h_mix(Y, Tt)
            cp_now = self.cp_mix(Y, Tt)
            if cp_now <= 0:
                break
            delta = (h_target - h_now) / cp_now
            # damp very large steps that could drag T out of NASA's [200,6000] range
            if delta > 800.0:
                delta = 800.0
            elif delta < -400.0:
                delta = -400.0
            Tt_new = max(200.0, min(6000.0, Tt + delta))
            if abs(Tt_new - Tt) < tol:
                Tt = Tt_new
                break
            Tt = Tt_new
        return Tt

    def stagnation_Pt(self, Y, T_static, Tt, p_static):
        """Isentropic Pt from s(Tt, Pt) = s(T_static, p_static) at fixed composition.

        Mixing-entropy term ( −R_i · Σ Y_i ln X_i ) is independent of T and p,
        so it cancels — only the Σ Y_i [s°(Tt) − s°(T)] − R_mix · ln(Pt/p) bit remains.
        """
        ds_T = 0.0
        for sp, y in Y.items():
            if y <= 0:
                continue
            ds_T += y * (self.s0_i(sp, Tt) - self.s0_i(sp, T_static))
        Wm = self.W_mix(Y)
        R_mix = self.R_UNIVERSAL / Wm
        try:
            return float(p_static) * float(np.exp(ds_T / R_mix))
        except OverflowError:
            return float(p_static)

    def stagnation_state(self, Y, T, p_pa, V):
        """One-shot helper: returns dict {h, s, ht, st, Tt, Pt}."""
        h  = self.h_mix(Y, T)
        s  = self.s_mix(Y, T, p_pa)
        ht = h + 0.5 * V * V
        Tt = self.stagnation_Tt(Y, T, ht)
        Pt = self.stagnation_Pt(Y, T, Tt, p_pa)
        return {"h": h, "s": s, "ht": ht, "st": s, "Tt": Tt, "Pt": Pt}


# ---------------------------------------------------------------------------
# Thin NASA CEA wrapper — equilibrium TP solve with caching.
# Returns per-species mass fractions in the same name space MixtureNASA uses,
# so the two can be composed seamlessly (CEA gives composition, MixtureNASA
# gives h / s / cp / Tt / Pt).
# ---------------------------------------------------------------------------
class CEAComp:
    PROD_NAMES = ["Ar", "CO2", "H", "H2", "H2O",
                  "N",  "NO",  "N2", "O", "O2", "OH"]  # all in AirProperties.NASA_DATA

    def __init__(self):
        if not _HAS_CEA:
            raise ImportError(
                "NASA CEA package not installed. Run `pip install cea` first."
            )
        self.cea = _CEA
        self.reac = _CEA.Mixture(["H2", "Air"])
        self.prod = _CEA.Mixture(self.PROD_NAMES)
        self.solver   = _CEA.EqSolver(self.prod, reactants=self.reac)
        self.solution = _CEA.EqSolution(self.solver)
        self._fuel_w  = self.reac.moles_to_weights(np.array([1.0, 0.0]))
        self._oxid_w  = self.reac.moles_to_weights(np.array([0.0, 1.0]))
        self._cache: dict = {}

    def equilibrium_Y(self, T, p_pa, of_ratio):
        """Return equilibrium mass-fraction dict {species: Y} at (T, p, O/F)."""
        key = (round(float(T)),
               round(float(p_pa) / 10) * 10,
               round(float(of_ratio) * 100) / 100)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        T_c  = float(np.clip(T, 250.0, 5500.0))
        p_c  = max(float(p_pa), 50.0)
        of_c = max(float(of_ratio), 0.01)

        Y: dict | None = None
        try:
            w = self.reac.of_ratio_to_weights(self._oxid_w, self._fuel_w, of_c)
            self.solver.solve(self.solution, self.cea.TP, T_c, p_c/1e5, w)
            if bool(self.solution.converged):
                # cea returns a {species_name: mass_fraction} dict (not array!)
                mf = self.solution.mass_fractions
                Y = {sp: float(mf.get(sp, 0.0)) for sp in self.PROD_NAMES}
                # Sanity check: sum should be ~1; if grossly off, treat as failure
                if abs(sum(Y.values()) - 1.0) > 0.05:
                    Y = None
        except Exception:  # noqa: BLE001
            Y = None

        self._cache[key] = Y  # may be None — caller decides fallback
        return Y


# ---------------------------------------------------------------------------
# Atmosphere (unchanged)
# ---------------------------------------------------------------------------
class Atmosphere:
    R_AIR = 287.05
    G0    = 9.80665

    @staticmethod
    def _layer(h):
        if   h <= 11000: return 0.0,   -0.0065, 288.15, 101325.0
        elif h <= 20000: return 11000,  0.0,    216.65,  22632.1
        elif h <= 32000: return 20000,  0.001,  216.65,   5474.89
        else: raise ValueError(f"Altitude {h:.0f} m > 32 km ceiling.")

    @staticmethod
    def T(h):
        h0, L, T0, _ = Atmosphere._layer(h)
        return T0 + L*(h - h0)

    @staticmethod
    def P(h):
        h0, L, T0, P0 = Atmosphere._layer(h)
        dh = h - h0; T = T0 + L*dh
        if L != 0:
            return P0*(T/T0)**(-Atmosphere.G0/(L*Atmosphere.R_AIR))
        return P0*np.exp(-Atmosphere.G0*dh/(Atmosphere.R_AIR*T0))

    @staticmethod
    def rho(h):
        return Atmosphere.P(h)/(Atmosphere.R_AIR*Atmosphere.T(h))


# ---------------------------------------------------------------------------
# Shapiro generalised-1D ODE.
# - `derivatives` keeps the per-phenomenon `switches` toggle.
# - `integrate` now optionally takes a `state_fn(T, p, V, x)` that returns
#   {h, s, ht, st, Tt, Pt}; if provided, those override the simple
#   constant-Cp Tt/Pt formulas in the result.
# ---------------------------------------------------------------------------
class ShapiroODE:
    @staticmethod
    def derivatives(Ma2, p, T, gamma, Cp, dA_dx, A, D, Cf,
                    dH_dx, mdot, dmdot_dx, W, dW_dx, dgamma_dx,
                    switches=None):
        if switches is None:
            switches = {
                "area": True, "friction": True, "mass": True,
                "heat": True, "MW": True, "gamma": True,
            }
        on = lambda key: 1.0 if switches.get(key, True) else 0.0
 
        g   = gamma
        M2  = Ma2
        D1  = 1.0 - M2
        if abs(D1) < 1e-8:
            D1 = 1e-8 if D1 >= 0 else -1e-8
 
        g1m2 = 1.0 + (g - 1.0) / 2.0 * M2
        gM2  = g * M2
        fric = 4.0 * Cf / D
        heat = dH_dx / (Cp * T)
 
        dMa2_dx = M2 * (
            -(2.0 * g1m2 / D1) * (dA_dx / A)                    * on("area")
            + ((1.0 + gM2) / D1) * heat                          * on("heat")
            + (gM2 * g1m2 / D1) * fric                           * on("friction")
            + (2.0 * (1.0 + gM2) * g1m2 / D1) * (dmdot_dx/mdot) * on("mass")
            - ((1.0 + gM2) / D1) * (dW_dx / W)                  * on("MW")
            - (dgamma_dx / g)                                     * on("gamma")
        )
 
        dp_dx = p * (
            (gM2 / D1) * (dA_dx / A)                             * on("area")
            - (gM2 / D1) * heat                                   * on("heat")
            - (gM2 * (1.0 + (g-1.0)*M2) / (2.0*D1)) * fric      * on("friction")
            - (2.0 * gM2 * g1m2 / D1) * (dmdot_dx/mdot)         * on("mass")
            + (gM2 / D1) * (dW_dx / W)                           * on("MW")
        )
 
        dT_dx = T * (
            ((g-1.0) * M2 / D1) * (dA_dx / A)                   * on("area")
            + ((1.0 + gM2) / D1) * heat                          * on("heat")
            - (g*(g-1.0)*M2**2 / (2.0*D1)) * fric               * on("friction")
            - ((g-1.0)*M2*(1.0+gM2) / D1) * (dmdot_dx/mdot)    * on("mass")
            + ((g-1.0)*M2 / D1) * (dW_dx / W)                   * on("MW")
        )
 
        return dMa2_dx, dp_dx, dT_dx
 
    # ------------------------------------------------------------------
    @staticmethod
    def integrate(x_start, x_end,
                  Ma2_in, p_in, T_in, mdot_in,
                  geometry_fn, composition_fn, source_fn,
                  mix,
                  state_fn=None,
                  switches=None,
                  Cf=0.003,
                  n_steps=1000):
        """
        Generalised 1-D Shapiro integration (energy-consistent T).
        Works for both subsonic (ramjet combustor) and supersonic sections.
        """
        R_UNIV = mix.R_UNIVERSAL
 
        Y_in     = composition_fn(x_start, T_in, p_in)
        cp_in    = mix.cp_mix(Y_in, T_in)
        W_in     = mix.W_mix(Y_in)
        R_in     = R_UNIV / W_in
        gamma_in = cp_in / max(cp_in - R_in, 1e-30)
        V2_in    = max(Ma2_in, 1e-10) * gamma_in * R_in * T_in
        h_in     = mix.h_mix(Y_in, T_in)
        ht_in    = h_in + 0.5 * V2_in
 
        T_cache = {"T": float(T_in)}
 
        def solve_T(M2, p, ht, x, T_guess=None):
            T = float(T_guess) if T_guess is not None else T_cache["T"]
            T = max(200.0, min(6000.0, T))
            Y = cp = W = R = gamma = V2 = None
            last_T = T
            for _ in range(60):
                Y      = composition_fn(x, T, p)
                cp     = mix.cp_mix(Y, T)
                W      = mix.W_mix(Y)
                R      = R_UNIV / W
                gamma  = cp / max(cp - R, 1e-30)
                h      = mix.h_mix(Y, T)
                V2     = M2 * gamma * R * T
                resid  = (h + 0.5 * V2) - ht
                if abs(resid) < 1.0:
                    break
                deriv = cp + 0.5 * M2 * gamma * R
                if deriv <= 0:
                    break
                step = -resid / deriv
                step = max(-200.0, min(400.0, step))
                T_new = max(200.0, min(6000.0, T + step))
                if abs(T_new - last_T) < 0.5:
                    T = T_new
                    Y     = composition_fn(x, T, p)
                    cp    = mix.cp_mix(Y, T)
                    W     = mix.W_mix(Y)
                    R     = R_UNIV / W
                    gamma = cp / max(cp - R, 1e-30)
                    V2    = M2 * gamma * R * T
                    break
                last_T = T
                T = T_new
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
 
            if sw_MW or sw_gam:
                dx_step = 1e-4
                x_p = min(x + dx_step, x_end)
                x_m = max(x - dx_step, x_start)
                span = x_p - x_m
                if span > 0:
                    Y_p = composition_fn(x_p, T, p)
                    Y_m = composition_fn(x_m, T, p)
                    dW_dx     = (mix.W_mix(Y_p) - mix.W_mix(Y_m)) / span     if sw_MW  else 0.0
                    dgamma_dx = (mix.gamma_mix(Y_p,T) - mix.gamma_mix(Y_m,T)) / span if sw_gam else 0.0
                else:
                    dW_dx = 0.0; dgamma_dx = 0.0
            else:
                dW_dx = 0.0; dgamma_dx = 0.0
 
            dM2_dx, dp_dx, _ = ShapiroODE.derivatives(
                Ma2=M2, p=p, T=T, gamma=gamma, Cp=cp,
                dA_dx=dA_dx, A=A, D=D, Cf=Cf,
                dH_dx=dH_dx,
                mdot=mdot, dmdot_dx=dmdot_dx,
                W=W, dW_dx=dW_dx, dgamma_dx=dgamma_dx,
                switches=switches,
            )
            return [dM2_dx, dp_dx, dH_dx, dmdot_dx]
 
        # ---- events -------------------------------------------------------
        def choke_event(x, y):
            """Fire (and stop) when subsonic flow reaches Ma² = 1."""
            return y[0] - 1.0
        choke_event.terminal  = True
        choke_event.direction = 1   # subsonic → sonic crossing only
 
        def pressure_event(x, y):
            return y[1] - 1.0
        pressure_event.terminal  = True
        pressure_event.direction = -1
 
        # ---- integrate ----------------------------------------------------
        y0 = [
            max(Ma2_in, 1e-10),
            max(p_in,   1.0),
            float(ht_in),
            max(mdot_in, 1e-9),
        ]
 
        sol = solve_ivp(
            fun=rhs,
            t_span=(x_start, x_end),
            y0=y0,
            method="DOP853",
            rtol=1e-6, atol=1e-6,
            max_step=(x_end - x_start) / 50,
            events=[choke_event, pressure_event],
            dense_output=False,
        )
 
        xs   = sol.t
        # ── FIX: do NOT clamp subsonic M² up to 1 ─────────────────────────
        # The original code had np.maximum(…, 1.000001) which broke subsonic
        # sections by misreporting all Ma as supersonic.  A plain floor at
        # near-zero is sufficient; the choke event handles the Ma→1 limit.
        M2s   = np.maximum(sol.y[0], 1e-12)
        ps    = np.maximum(sol.y[1], 1.0)
        hts_arr = sol.y[2]
        mdots = np.maximum(sol.y[3], 1e-9)
        Mas   = np.sqrt(M2s)
 
        thermal_choke = len(sol.t_events[0]) > 0
        if thermal_choke:
            x_choke = sol.t_events[0][0]
            print(f"\n  ℹ Thermal choking at x = {x_choke:.4f} m  "
                  f"(Ma → 1) — using exit as nozzle throat.")
 
        # ---- post-process ------------------------------------------------
        T_cache["T"] = float(T_in)
        Ts = np.empty_like(xs); Vs = np.empty_like(xs)
        cps = np.empty_like(xs); gammas = np.empty_like(xs)
        Rs  = np.empty_like(xs); rhos   = np.empty_like(xs)
        for i in range(len(xs)):
            T_i, Y_i, cp_i, W_i, R_i, g_i, V2_i = solve_T(
                M2s[i], ps[i], hts_arr[i], xs[i])
            Ts[i]     = T_i
            Vs[i]     = np.sqrt(max(V2_i, 0.0))
            cps[i]    = cp_i
            gammas[i] = g_i
            Rs[i]     = R_i
            rhos[i]   = ps[i] / max(R_i * T_i, 1e-12)
 
        As = np.array([geometry_fn(x)[0] for x in xs])
 
        if state_fn is None:
            def state_fn(T, p, V, x):
                Y = composition_fn(x, T, p)
                return mix.stagnation_state(Y, T, p, V)
 
        hs = np.empty_like(xs); ss   = np.empty_like(xs)
        hts2 = np.empty_like(xs); sts2 = np.empty_like(xs)
        Tts = np.empty_like(xs); Pts  = np.empty_like(xs)
        for i in range(len(xs)):
            st = state_fn(Ts[i], ps[i], Vs[i], xs[i])
            hs[i]   = st["h"];  ss[i]   = st["s"]
            hts2[i] = st["ht"]; sts2[i] = st["st"]
            Tts[i]  = st["Tt"]; Pts[i]  = st["Pt"]
 
        return {
            "x": xs, "Ma": Mas, "Ma2": M2s,
            "p": ps,  "P": ps,  "T": Ts,
            "rho": rhos, "V": Vs,
            "Tt": Tts, "Pt": Pts, "pt": Pts,
            "h": hs,   "s": ss,
            "ht": hts2, "st": sts2,
            "A": As,   "mdot": mdots,
            "thermal_choke": thermal_choke,
            "solver_success": sol.success,
            "solver_message": sol.message,
        }
 
 
# ---------------------------------------------------------------------------
# Ramjet Engine
# ---------------------------------------------------------------------------
class Ramjet:
    """
    1-D Ramjet cycle model.
 
    Flow path
    ─────────
    freestream (0) ──► inlet / oblique-shock diffuser
                   ──► isolator   (0 → 1, algebraic)   subsonic M1 ≈ Ma_COMB
                   ──► sec 1→2    (friction/area)        Shapiro ODE
                   ──► sec 2→3    (fuel injection)        Shapiro ODE
                   ──► sec 3→4    (combustion, CEA)        Shapiro ODE
                   ──► nozzle     (4 → 5)
                          ├─ isentropic throat (from Tt4, Pt4, ṁ)
                          └─ supersonic diverging Shapiro from Ma = 1.001
    """
 
    # ── Axial lengths [m] ─────────────────────────────────────────────────
    L01 = 0.60   # inlet / oblique-shock diffuser
    L12 = 0.15   # constant-area isolator segment
    L23 = 0.1   # fuel injection zone
    L34 = 0.30   # combustor  (longer than scramjet: subsonic mixing needs more room)
    L45 = 1.20   # nozzle (convergent-divergent, modelled as diverging section only)
 
    # ── Area ratios (relative to A1) ──────────────────────────────────────
    alpha12 = 0.85   # slight expansion: sec 1 → 2
    alpha13 = 0.75   # continued diffusion to combustor inlet (sec 1 → 3)
    alpha14 = 1.0371   # slight taper of combustor (sec 1 → 4 reference)
    alpha05 = 3.50   # nozzle exit / A0  (large: supersonic exit)
 
    # ── Aerothermodynamic parameters ──────────────────────────────────────
    Ma_COMB    = 0.30   # target Mach at combustor inlet (subsonic, replaces EPSILON*Ma0)
    EPSILON    = 0.10   # kept for informational prints; Ma_COMB takes precedence
    ETA_C      = 0.90   # combustion efficiency
    CF_DEFAULT = 0.003  # skin-friction coefficient
 
    Q_H2_HHV   = 141.8e6  # J/kg  (informational only)
 
    def __init__(self):
        self.air        = AirProperties()
        self.mixture    = MixtureNASA(self.air)
        self.shapiroODE = ShapiroODE()
        self._cea_comp  = None
 
    def _get_cea(self):
        if self._cea_comp is None:
            self._cea_comp = CEAComp()
        return self._cea_comp
 
    def _f(self, x):
        return float(np.asarray(x).squeeze())
 
    def _air_Y(self):
        moles     = self.air.AIR_BASE_COMPOSITION
        total     = sum(moles.values())
        W_air     = sum((moles[s]/total) * self.air.MOLECULAR_WEIGHTS[s] for s in moles)
        return {s: (moles[s]/total) * self.air.MOLECULAR_WEIGHTS[s] / W_air for s in moles}
 
    def _frozen_state_fn(self, Y_const):
        def state_fn(T, p, V, x):
            return self.mixture.stagnation_state(Y_const, T, p, V)
        return state_fn
 
    # =====================================================================
    # Section 0 — Freestream / inlet capture
    # =====================================================================
    def inlet_properties(self, h, Ma, m_air):
        T0   = Atmosphere.T(h)
        P0   = Atmosphere.P(h)
        rho0 = Atmosphere.rho(h)
        Y_air = self._air_Y()
 
        cp0    = self.mixture.cp_mix(Y_air, T0)
        W_kgmol = self.mixture.W_mix(Y_air)
        R0     = self.mixture.R_UNIVERSAL / W_kgmol
        gamma0 = self.mixture.gamma_mix(Y_air, T0)
 
        a0  = np.sqrt(gamma0 * R0 * T0)
        V0  = Ma * a0
        A0  = m_air / (rho0 * V0)
 
        h0  = self.mixture.h_mix(Y_air, T0)
        s0  = self.mixture.s_mix(Y_air, T0, P0)
        ht0 = h0 + 0.5 * V0**2
        Tt0 = self.mixture.stagnation_Tt(Y_air, T0, ht0)
        Pt0 = self.mixture.stagnation_Pt(Y_air, T0, Tt0, P0)
 
        print(f"\n── Inlet  h={h:.0f} m  Ma={Ma:.2f}  ṁ={m_air:.2f} kg/s ──")
        print(f"  T0={T0:.1f} K   P0={P0:.0f} Pa   rho0={rho0:.4f} kg/m³")
        print(f"  V0={V0:.1f} m/s   A0={A0:.4f} m²   Tt0={Tt0:.1f} K   Pt0={Pt0:.0f} Pa")
 
        return {
            "Ma": Ma, "Ma0": Ma, "T": T0, "T0": T0,
            "P": P0,  "P0": P0, "rho": rho0, "rho0": rho0,
            "gamma": gamma0, "cp": cp0, "R": R0, "a": a0,
            "V": V0,  "V0": V0, "A": A0, "A0": A0,
            "Tt": Tt0, "Tt0": Tt0, "Pt": Pt0, "Pt0": Pt0,
            "h": h0,  "ht": ht0, "s": s0, "st": s0,
            "Y": Y_air, "mdot": m_air,
        }
 
    # =====================================================================
    # Pressure recovery — MIL-E-5008B (ramjet standard)
    # =====================================================================
    def pressure_recovery(self, Ma):
        """
        MIL-E-5008B total-pressure recovery for a ramjet inlet.
 
          σ = 1                             Ma ≤ 1
          σ = 1 − 0.075 (Ma−1)^1.35        1 < Ma ≤ 5
          σ = 800 / (Ma⁴ + 935)            Ma > 5
        """
        Ma = float(Ma)
        if Ma <= 1.0:
            return 1.0
        elif Ma <= 5.0:
            return 1.0 - 0.075 * (Ma - 1.0)**1.35
        else:
            return 800.0 / (Ma**4 + 935.0)
 
    # =====================================================================
    # Section 1 — Isolator (algebraic, targets Ma_COMB at exit)
    # =====================================================================
    def isolator_properties(self, inlet_props):
        """
        Decelerates the captured flow to the subsonic combustor-inlet Mach
        number ``Ma_COMB`` via a combination of oblique shocks and subsonic
        diffusion.  The total-pressure loss is prescribed by the MIL-E-5008B
        pressure recovery.
        """
        mix   = self.mixture
        Y_air = inlet_props["Y"]
 
        Ma0  = self._f(inlet_props["Ma"])
        T0   = self._f(inlet_props["T"])
        P0   = self._f(inlet_props["P"])
        V0   = self._f(inlet_props["V"])
        Pt0  = self._f(inlet_props["Pt"])
        mdot = self._f(inlet_props["mdot"])
        A0   = self._f(inlet_props["A"])
        rho0 = self._f(inlet_props["rho"])
 
        ht0 = self._f(inlet_props["ht"])
        s0  = self._f(inlet_props["s"])
 
        # ── Combustor-inlet Mach: fixed subsonic target ──────────────────
        M1 = float(self.Ma_COMB)
 
        sigma_c    = self.pressure_recovery(Ma0)
        Pt1_target = sigma_c * Pt0
 
        def residual(vars_):
            T1g, p1g = vars_
            T1g = max(T1g, 250.0); p1g = max(p1g, 100.0)
            cp1    = mix.cp_mix(Y_air, T1g)
            W1     = mix.W_mix(Y_air)
            R1     = mix.R_UNIVERSAL / W1
            gamma1 = mix.gamma_mix(Y_air, T1g)
            V1     = M1 * np.sqrt(gamma1 * R1 * T1g)
            h1     = mix.h_mix(Y_air, T1g)
            eq1    = ht0 - (h1 + 0.5 * V1**2)
            Tt1g   = mix.stagnation_Tt(Y_air, T1g, ht0)
            Pt1g   = mix.stagnation_Pt(Y_air, T1g, Tt1g, p1g)
            eq2    = Pt1g - Pt1_target
            return [eq1, eq2]
 
        T1, P1 = fsolve(residual, x0=[600.0, 0.3 * P0])
 
        W1     = mix.W_mix(Y_air)
        R1     = mix.R_UNIVERSAL / W1
        gamma1 = mix.gamma_mix(Y_air, T1)
        cp1    = mix.cp_mix(Y_air, T1)
        V1     = M1 * np.sqrt(gamma1 * R1 * T1)
        h1     = mix.h_mix(Y_air, T1)
        s1     = mix.s_mix(Y_air, T1, P1)
        ht1    = h1 + 0.5 * V1**2
        Tt1    = mix.stagnation_Tt(Y_air, T1, ht1)
        Pt1    = mix.stagnation_Pt(Y_air, T1, Tt1, P1)
        rho1   = P1 / (R1 * T1)
        A1     = mdot / (rho1 * V1)
 
        L_iso = getattr(self, "L01", 0.6)
        sol = {
            "x":    np.array([0.0, L_iso]),
            "Ma":   np.array([Ma0, M1]),
            "T":    np.array([T0, T1]),
            "Tt":   np.array([self._f(inlet_props["Tt"]), Tt1]),
            "p":    np.array([P0, P1]),  "P":  np.array([P0, P1]),
            "pt":   np.array([Pt0, Pt1]),"Pt": np.array([Pt0, Pt1]),
            "A":    np.array([A0, A1]),
            "rho":  np.array([rho0, rho1]),
            "V":    np.array([V0, V1]),
            "mdot": np.array([mdot, mdot]),
            "h":    np.array([self._f(inlet_props["h"]), h1]),
            "s":    np.array([s0, s1]),
            "ht":   np.array([self._f(inlet_props["ht"]), ht1]),
            "st":   np.array([s0, s1]),
        }
 
        print(f"\n── Isolator ──")
        print(f"  σ_c = {sigma_c:.4f}   Ma1 = {M1:.3f}   T1 = {T1:.1f} K   "
              f"P1 = {P1:.0f} Pa   A1 = {A1:.4f} m²")
 
        return {
            "Ma": M1,   "Ma1": M1,
            "T":  T1,   "T1":  T1,
            "P":  P1,   "p1":  P1,  "P1": P1,
            "V":  V1,   "V1":  V1,
            "A":  A1,   "A1":  A1,
            "Tt": Tt1,  "Tt1": Tt1,
            "Pt": Pt1,  "Pt1": Pt1,
            "rho": rho1, "gamma": gamma1, "cp": cp1, "R": R1,
            "sigma_c": sigma_c, "mdot": mdot,
            "h": h1, "ht": ht1, "s": s1,
            "Y": Y_air,
            "solution": sol,
        }
 
    # =====================================================================
    # Section 1→2 — Constant-area / friction-only
    # =====================================================================
    def combustor_properties2(self, isolator_props, switches=None):
        L_12 = self._f(self.L12)
        A1   = self._f(isolator_props["A"])
        A2   = self._f(self.alpha12) * A1
 
        Ma1  = self._f(isolator_props["Ma"])
        T1   = self._f(isolator_props["T"])
        p1   = self._f(isolator_props["P"])
        mdot = self._f(isolator_props["mdot"])
        Y_air = isolator_props["Y"]
        W_air = self.mixture.W_mix(Y_air)
 
        def geometry_fn(x):
            A = A1 + (A2 - A1) * (x / L_12)
            dA_dx = (A2 - A1) / L_12
            D = np.sqrt(4.0 * A / np.pi)
            return A, dA_dx, D
 
        def composition_fn(x, T, p): return Y_air
        def source_fn(x, T, p, mdot_local, Y): return 0.0, 0.0
 
        state_fn = self._frozen_state_fn(Y_air)
        result = self.shapiroODE.integrate(
            x_start=0.0, x_end=L_12,
            Ma2_in=Ma1**2, p_in=p1, T_in=T1, mdot_in=mdot,
            geometry_fn=geometry_fn,
            composition_fn=composition_fn,
            source_fn=source_fn,
            mix=self.mixture,
            state_fn=state_fn,
            switches=switches,
            Cf=self.CF_DEFAULT, n_steps=300,
        )
 
        T_end = result["T"][-1]; p_end = result["p"][-1]
        return {
            "Ma":  self._f(result["Ma"][-1]), "Ma2": self._f(result["Ma"][-1]),
            "T":   self._f(T_end),            "T2":  self._f(T_end),
            "Tt":  self._f(result["Tt"][-1]),
            "P":   self._f(p_end),            "p2":  self._f(p_end),
            "Pt":  self._f(result["Pt"][-1]),
            "rho": self._f(result["rho"][-1]),
            "V":   self._f(result["V"][-1]),  "V2":  self._f(result["V"][-1]),
            "h":   self._f(result["h"][-1]),
            "ht":  self._f(result["ht"][-1]),
            "s":   self._f(result["s"][-1]),
            "A":   A2,
            "gamma": self.mixture.gamma_mix(Y_air, T_end),
            "cp":    self.mixture.cp_mix(Y_air, T_end),
            "R":     self.mixture.R_UNIVERSAL / W_air,
            "mdot": mdot,
            "Y":    Y_air,
            "solution": result,
        }
 
    def optimal_fuel_air_ratio(self):
        return 1.0 / 34.35  # H2/air stoichiometric
 
    # =====================================================================
    # Section 2→3 — Fuel injection (mass addition only)
    # =====================================================================
    def combustor_properties3(self, sec2, phi=0.0, switches=None):
        mix    = self.mixture
        Y_air  = sec2["Y"]
        sw_mass = True if switches is None else switches.get("mass", True)
 
        L_23 = self._f(self.L23)
        A2   = self._f(sec2["A"])
        A3   = self._f(self.alpha13) * A2 / self._f(self.alpha12)
 
        Ma2      = self._f(sec2["Ma"])
        T2       = self._f(sec2["T"])
        p2       = self._f(sec2["P"])
        mdot_air = self._f(sec2["mdot"])
 
        FAR_stoich      = self.optimal_fuel_air_ratio()
        FAR_actual      = phi * FAR_stoich
        mfuel_total     = FAR_actual * mdot_air
        dmdot_dx_const  = mfuel_total / L_23
 
        def Yf_at_mdot(mdot_local):
            return max((mdot_local - mdot_air) / max(mdot_local, 1e-30), 0.0)
 
        def Y_at_mdot(mdot_local):
            Yf = Yf_at_mdot(mdot_local)
            Ya = 1.0 - Yf
            Y  = {sp: Ya * Y_air[sp] for sp in Y_air}
            Y["H2"] = Y.get("H2", 0.0) + Yf
            return Y
 
        def Yf_at_x(x):
            mdot_local = mdot_air + dmdot_dx_const * x
            return Yf_at_mdot(mdot_local)
 
        def geometry_fn(x):
            A = A2 + (A3 - A2) * (x / L_23)
            dA_dx = (A3 - A2) / L_23
            D = np.sqrt(4 * A / np.pi)
            return A, dA_dx, D
 
        def composition_fn(x, T, p):
            if not sw_mass:
                return Y_air
            mdot_local = mdot_air + dmdot_dx_const * x
            return Y_at_mdot(mdot_local)
 
        def source_fn(x, T, p, mdot_local, Y):
            return 0.0, dmdot_dx_const
 
        def state_fn(T, p, V, x):
            Y = composition_fn(x, T, p)
            return mix.stagnation_state(Y, T, p, V)
 
        result = self.shapiroODE.integrate(
            x_start=0.0, x_end=L_23,
            Ma2_in=Ma2**2, p_in=p2, T_in=T2, mdot_in=mdot_air,
            geometry_fn=geometry_fn,
            composition_fn=composition_fn,
            source_fn=source_fn,
            mix=mix,
            state_fn=state_fn,
            switches=switches,
            Cf=self.CF_DEFAULT, n_steps=200,
        )
 
        Y_exit       = composition_fn(L_23, 0.0, 0.0)
        mfuel_actual = max(self._f(result["mdot"][-1]) - mdot_air, 0.0)
        return {
            "Ma3": self._f(result["Ma"][-1]),
            "T3":  self._f(result["T"][-1]),
            "p3":  self._f(result["p"][-1]),  "P3": self._f(result["p"][-1]),
            "rho3": self._f(result["rho"][-1]),
            "V3":   self._f(result["V"][-1]),
            "Tt3":  self._f(result["Tt"][-1]),
            "Pt3":  self._f(result["Pt"][-1]),
            "h3":   self._f(result["h"][-1]),
            "ht3":  self._f(result["ht"][-1]),
            "s3":   self._f(result["s"][-1]),
            "A3":  A3,
            "mdot": self._f(result["mdot"][-1]),
            "mfuel":           mfuel_actual,
            "mfuel_scheduled": mfuel_total,
            "phi": phi,
            "Y":   Y_exit,
            "Yf_at_x_fn": Yf_at_x,
            "solution": result,
        }
 
    # =====================================================================
    # Section 3→4 — Combustion (CEA equilibrium, subsonic Shapiro)
    # =====================================================================
    def combustor_properties4(self, sec3, switches=None):
        if not _HAS_CEA:
            raise ImportError(
                "NASA CEA is required for combustor_properties4. "
                "Install with `pip install cea`."
            )
 
        mix      = self.mixture
        cea_comp = self._get_cea()
        sw_heat  = True if switches is None else switches.get("heat", True)
 
        L_34  = self._f(self.L34)
        A3    = self._f(sec3["A3"])
        A1_ref = A3 / self._f(self.alpha13)
        A4    = self._f(self.alpha14) * A1_ref
 
        Ma3   = self._f(sec3["Ma3"])
        T3    = self._f(sec3["T3"])
        p3    = self._f(sec3["p3"])
        mdot  = self._f(sec3["mdot"])
 
        Y_react = dict(sec3["Y"])
        for sp in CEAComp.PROD_NAMES:
            Y_react.setdefault(sp, 0.0)
 
        Yf_react = float(Y_react.get("H2", 0.0))
        of_ratio = (1.0 - Yf_react) / Yf_react if Yf_react > 1e-12 else 1e6
 
        theta = 0  # parallel injection → linear η ramp
 
        def mixing_efficiency(x):
            s = np.clip(x / L_34, 1e-4, 1.0)
            if theta == 0.0:
                return float(s)
            a = float(np.clip(1.01 + 0.176 * np.log(s), 0.0, 1.0))
            if theta == 90.0:
                return a
            return theta/90.0 * (a - s) + s
 
        def deta_dx(x):
            h = 1e-4
            return (mixing_efficiency(min(x+h, L_34)) -
                    mixing_efficiency(max(x-h, 0.0))) / (2*h)
 
        def Y_eq_at(T, p_pa):
            Yeq = cea_comp.equilibrium_Y(T, p_pa, of_ratio)
            return Yeq if Yeq is not None else Y_react
 
        def Y_blended(eta, T, p_pa):
            Yeq  = Y_eq_at(T, p_pa)
            keys = set(Y_react) | set(Yeq)
            return {k: (1-eta)*Y_react.get(k,0.0) + eta*Yeq.get(k,0.0) for k in keys}
 
        def geometry_fn(x):
            A = A3 + (A4 - A3) * (x / L_34)
            dA_dx = (A4 - A3) / L_34
            D = np.sqrt(4 * A / np.pi)
            return A, dA_dx, D
 
        def composition_fn(x, T, p):
            if not sw_heat:
                return Y_react
            return Y_blended(mixing_efficiency(x), T, p)
 
        def source_fn(x, T, p, mdot_local, Y):
            h_react = mix.h_mix(Y_react, T)
            Yeq     = Y_eq_at(T, p)
            h_eq    = mix.h_mix(Yeq, T)
            dH_dx   = (h_react - h_eq) * deta_dx(x)
            return dH_dx, 0.0
 
        def state_fn(T, p, V, x):
            return mix.stagnation_state(composition_fn(x, T, p), T, p, V)
 
        result = self.shapiroODE.integrate(
            x_start=0.0, x_end=L_34,
            Ma2_in=Ma3**2, p_in=p3, T_in=T3, mdot_in=mdot,
            geometry_fn=geometry_fn,
            composition_fn=composition_fn,
            source_fn=source_fn,
            mix=mix,
            state_fn=state_fn,
            switches=switches,
            Cf=self.CF_DEFAULT, n_steps=500,
        )
 
        x_exit = result["x"][-1]
        T_exit, p_exit = result["T"][-1], result["p"][-1]
        Y_exit = composition_fn(x_exit, T_exit, p_exit)
 
        return {
            "Ma4":  self._f(result["Ma"][-1]),
            "T4":   self._f(result["T"][-1]),
            "p4":   self._f(result["p"][-1]), "P4": self._f(result["p"][-1]),
            "rho4": self._f(result["rho"][-1]),
            "V4":   self._f(result["V"][-1]),
            "Tt4":  self._f(result["Tt"][-1]),
            "Pt4":  self._f(result["Pt"][-1]),
            "h4":   self._f(result["h"][-1]),
            "ht4":  self._f(result["ht"][-1]),
            "s4":   self._f(result["s"][-1]),
            "A4":   A4,
            "mdot": mdot,
            "Y":    Y_exit,
            "solution": result,
            "thermal_choke": result["thermal_choke"],
        }
 
    # =====================================================================
    # Section 4→5 — Convergent-divergent nozzle (ramjet)
    # =====================================================================
    def nozzle_properties(self, sec4, inlet_props, switches=None):
        """
        Ramjet C-D nozzle.
 
        Step 1 — Isentropic throat
        ──────────────────────────
        Regardless of whether the combustor thermally choked (Ma4 ≈ 1) or
        exited subsonically (Ma4 < 1), the throat state is found from the
        stagnation conditions (Tt4, Pt4) and the continuity constraint:
 
            A_th = ṁ / (ρ_th · a_th)
 
        This is exact for isentropic flow to Ma = 1 and avoids the Shapiro
        singularity at D₁ = 1 − Ma² = 0 entirely.
 
        Step 2 — Supersonic diverging Shapiro
        ──────────────────────────────────────
        Starting from Ma = 1.001 (just past sonic) at A_throat, the Shapiro
        ODE is integrated over the diverging section to A5.  Friction losses
        are included; composition is frozen at the sec4 exit.
 
        Note: ``thermal_choke`` from sec4 is no longer an abort condition —
        the sonic combustor exit naturally IS the nozzle throat.
        """
        mix  = self.mixture
        Y_nz = sec4["Y"]
        W_nz = mix.W_mix(Y_nz)
        R_nz = mix.R_UNIVERSAL / W_nz
 
        Ma4  = self._f(sec4["Ma4"])
        T4   = self._f(sec4["T4"])
        p4   = self._f(sec4["p4"])
        Tt4  = self._f(sec4["Tt4"])
        Pt4  = self._f(sec4["Pt4"])
        mdot = self._f(sec4["mdot"])
        A4   = self._f(sec4["A4"])
 
        # Evaluate γ near the throat temperature for the isentropic relations.
        cp4  = mix.cp_mix(Y_nz, T4)
        g4   = cp4 / max(cp4 - R_nz, 1e-30)
 
        # ── Isentropic throat (Ma = 1) ────────────────────────────────────
        T_th   = Tt4 * 2.0 / (g4 + 1.0)
        P_th   = Pt4 * (2.0 / (g4 + 1.0))**(g4 / (g4 - 1.0))
        rho_th = P_th / (R_nz * T_th)
        a_th   = np.sqrt(g4 * R_nz * T_th)
        A_th   = mdot / (rho_th * a_th)   # throat area from continuity
 
        A0   = self._f(inlet_props["A0"])
        A5   = self._f(self.alpha05) * A0
        L_45 = self._f(self.L45)
 
        if A5 <= A_th:
            # Ensure exit is larger than throat; expand by factor 4 as fallback.
            A5 = A_th * 4.0
            print(f"  ⚠ A5 ≤ A_throat — widened to {A5:.4f} m²")
 
        print(f"\n── Nozzle ──")
        print(f"  Ma4 = {Ma4:.3f}  "
              f"{'(thermal choke → natural throat)' if sec4.get('thermal_choke') else '(subsonic → isentropic throat)'}")
        print(f"  A_throat = {A_th:.4f} m²   T_throat = {T_th:.1f} K   "
              f"P_throat = {P_th:.0f} Pa")
        print(f"  A5 = {A5:.4f} m²   AR_nozzle = A5/A_throat = {A5/A_th:.2f}")
 
        # ── Supersonic diverging section: A_throat → A5 ──────────────────
        # Start slightly supersonic so D₁ = 1 − Ma² < 0 throughout.
        Ma_start = 1.001
 
        def geometry_fn(x):
            A = A_th + (A5 - A_th) * (x / L_45)
            dA_dx = (A5 - A_th) / L_45
            D = np.sqrt(4.0 * A / np.pi)
            return A, dA_dx, D
 
        def composition_fn(x, T, p): return Y_nz
        def source_fn(x, T, p, m, Y): return 0.0, 0.0
 
        state_fn = self._frozen_state_fn(Y_nz)
 
        result = self.shapiroODE.integrate(
            x_start=0.0, x_end=L_45,
            Ma2_in=Ma_start**2, p_in=P_th, T_in=T_th, mdot_in=mdot,
            geometry_fn=geometry_fn,
            composition_fn=composition_fn,
            source_fn=source_fn,
            mix=mix,
            state_fn=state_fn,
            switches=switches,
            Cf=self.CF_DEFAULT, n_steps=200,
        )
 
        Ma5 = self._f(result["Ma"][-1])
        T5  = self._f(result["T"][-1])
        p5  = self._f(result["p"][-1])
        V5  = self._f(result["V"][-1])
        print(f"  Ma5 = {Ma5:.3f}   T5 = {T5:.1f} K   p5 = {p5:.0f} Pa   "
              f"V5 = {V5:.1f} m/s")
 
        return {
            "Ma5":  Ma5,
            "T5":   T5,
            "p5":   p5,   "P5": p5,
            "rho5": self._f(result["rho"][-1]),
            "V5":   V5,
            "Tt5":  self._f(result["Tt"][-1]),
            "Pt5":  self._f(result["Pt"][-1]),
            "h5":   self._f(result["h"][-1]),
            "ht5":  self._f(result["ht"][-1]),
            "s5":   self._f(result["s"][-1]),
            "A5":   A5,
            "A_throat": A_th,
            "mdot": mdot,
            "Y":    Y_nz,
            "solution": result,
            "thermal_choke": False,  # nozzle always completes
        }
    # =====================================================================
    # Performance
    # =====================================================================
    def performance(self, inlet_props, nozzle_props, sec3):
        if nozzle_props.get("thermal_choke", False):
            return {"thermal_choke": True}

        V0 = self._f(inlet_props["V0"])
        p0 = self._f(inlet_props["P0"])
        A0 = self._f(inlet_props["A0"])
        mdot_air = self._f(inlet_props["mdot"])

        V5 = self._f(nozzle_props["V5"])
        p5 = self._f(nozzle_props["p5"])
        A5 = self._f(nozzle_props["A5"])
        mdot5 = self._f(nozzle_props["mdot"])
        mfuel = self._f(sec3["mfuel"])

        Fin = mdot5*V5 + p5*A5 - mdot_air*V0 - p0*A0
        Isp = Fin / ((mfuel+mdot_air)) * 9.80665
        Ia  = Fin / mdot_air
        return {"Fin": Fin, "Isp": Isp, "Ia": Ia, "mfuel": mfuel, "thermal_choke": False}

    # =====================================================================
    # Plot
    # =====================================================================
    def plot_flowpath(self, inp, iso, sec2, sec3, sec4, sec5=None):
        sections = []

        def add_section(sol, x_offset):
            p_arr  = sol.get("p",  sol.get("P"))
            pt_arr = sol.get("pt", sol.get("Pt", p_arr))
            return {
                "x":    np.asarray(sol["x"]) + x_offset,
                "Ma":   np.asarray(sol["Ma"]),
                "T":    np.asarray(sol["T"]),
                "Tt":   np.asarray(sol.get("Tt", sol["T"])),
                "p":    np.asarray(p_arr),
                "pt":   np.asarray(pt_arr),
                "V":    np.asarray(sol["V"]),
                "mdot": np.asarray(sol["mdot"]),
                "A":    np.asarray(sol.get("A", np.full_like(sol["x"], np.nan))),
                "h":    np.asarray(sol.get("h",  np.zeros_like(sol["x"]))),
                "s":    np.asarray(sol.get("s",  np.zeros_like(sol["x"]))),
                "ht":   np.asarray(sol.get("ht", np.zeros_like(sol["x"]))),
                "st":   np.asarray(sol.get("st", np.zeros_like(sol["x"]))),
            }

        x0 = 0.0
        s_iso = add_section(iso["solution"],  x0); x0 = s_iso["x"][-1]
        s2    = add_section(sec2["solution"], x0); x0 = s2["x"][-1]
        s3    = add_section(sec3["solution"], x0); x0 = s3["x"][-1]
        s4    = add_section(sec4["solution"], x0); x0 = s4["x"][-1]
        sections.extend([s_iso, s2, s3, s4])
        #if sec5 is not None and not sec4.get("thermal_choke", False):
        s5 = add_section(sec5["solution"], x0)
        sections.append(s5)

        def cat(field): return np.concatenate([s[field] for s in sections])
        x, Ma  = cat("x"), cat("Ma")
        T, Tt  = cat("T"), cat("Tt")
        p, pt  = cat("p"), cat("pt")
        V      = cat("V")
        mdot   = cat("mdot")
        h_s, ht_s = cat("h"), cat("ht")
        s_s, st_s = cat("s"), cat("st")
        A_arr  = cat("A")

        fig, axs = plt.subplots(7, 1, figsize=(12, 26), sharex=True)

        # --- 1. Mach ----------------------------------------------------
        axs[0].plot(x, Ma, lw=2.5, color="black", label="Mach")
        axs[0].axhline(1.0, color="red", linestyle=":", alpha=0.5, label="Sonic")
        axs[0].set_ylabel("Mach Number")
        axs[0].legend(loc="best")

        # --- 2. T -------------------------------------------------------
        axs[1].plot(x, T,  lw=2,   color="tab:red",   label="Static T")
        axs[1].plot(x, Tt, lw=2,   color="darkred",   linestyle="--", label="Total T")
        axs[1].set_ylabel("Temperature [K]")
        axs[1].legend(loc="best")

        # --- 3. P -------------------------------------------------------
        axs[2].plot(x, p / 1e3,  lw=2, color="tab:green",  label="Static P")
        axs[2].plot(x, pt / 1e3, lw=2, color="darkgreen",  linestyle="--", label="Total P")
        axs[2].set_ylabel("Pressure [kPa]")
        axs[2].set_yscale("log")
        axs[2].legend(loc="best")

        # --- 4. Enthalpy -----------------------------------------------
        axs[3].plot(x, h_s  / 1e6, lw=2, color="tab:orange", label="Static h")
        axs[3].plot(x, ht_s / 1e6, lw=2, color="saddlebrown", linestyle="--", label="Total h")
        axs[3].set_ylabel("Enthalpy [MJ/kg]")
        axs[3].legend(loc="best")

        # --- 5. Entropy -------------------------------------------------
        axs[4].plot(x, s_s,  lw=2, color="tab:cyan",  label="Static s")
        axs[4].plot(x, st_s, lw=2, color="tab:blue",  linestyle="--", label="Total s")
        axs[4].set_ylabel("Entropy [J/kg/K]")
        axs[4].legend(loc="best")

        # --- 6. V -------------------------------------------------------
        axs[5].plot(x, V, lw=2, color="tab:blue")
        axs[5].set_ylabel("Velocity [m/s]")

        # --- 7. mdot + geometry silhouette -----------------------------
        axs[6].plot(x, mdot, lw=2, color="tab:purple", label="ṁ")
        axs[6].set_ylabel("Mass Flow [kg/s]")
        axs[6].set_xlabel("Position in Engine [m]")
        # Engine shape (radius equivalent) faint silhouette
        if np.all(np.isfinite(A_arr)) and np.nanmax(A_arr) > 0:
            r = np.sqrt(A_arr / np.pi)
            r_norm = r / np.nanmax(r)
            geom_scale = 0.45 * np.nanmax(mdot)
            axs[6].fill_between(x, -geom_scale*r_norm, geom_scale*r_norm,
                                color="lightgray", alpha=0.35, label="Geometry")
            axs[6].plot(x,  geom_scale*r_norm, color="black", lw=1.2)
            axs[6].plot(x, -geom_scale*r_norm, color="black", lw=1.2)
        axs[6].legend(loc="best")

        # Section boundaries + labels
        boundaries = [s["x"][-1] for s in sections[:-1]] if len(sections) > 1 else []
        labels = ["Isolator", "Comb 2", "Comb 3", "Comb 4", "Nozzle"]
        for ax in axs:
            ax.grid(True, which="both", alpha=0.3)
            for b in boundaries:
                ax.axvline(b, color="gray", linestyle="--", alpha=0.7)
        y_lim = axs[0].get_ylim()
        for i, label in enumerate(labels[:len(sections)]):
            x_mid = (sections[i]["x"][0] + sections[i]["x"][-1]) / 2
            axs[0].text(x_mid, y_lim[1]*0.92, label, ha="center", weight="bold")

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Run One Case
# ---------------------------------------------------------------------------
def altitude_mach(self, h_km, Ma0):
    """Helper to run a single case and print results."""
    eng = Ramjet()
    inp  = eng.inlet_properties(h=h_km*1e3, Ma=Ma0, m_air=1000.0)
    iso  = eng.isolator_properties(inp)
    sec2 = eng.combustor_properties2(iso)
    sec3 = eng.combustor_properties3(sec2, phi=0.5)
    sec4 = eng.combustor_properties4(sec3)
    sec5 = eng.nozzle_properties(sec4, inp)
    perf = eng.performance(inp, sec5, sec3)
    return perf


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------
def print_section(title, props, fields):
    w = 34
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")
    for label, key, unit, scale in fields:
        val = props.get(key, float("nan"))
        try:
            print(f"  {label:<{w}} {val*scale:>12.4f}  {unit}")
        except TypeError:
            print(f"  {label:<{w}} {'nan':>12}  {unit}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    eng = Ramjet()

    h_km = 20.0
    Ma0  = 3.0
    mdot = 100.0
    phi  = 0.5

    print(f"\n{'═'*65}")
    print(f"  SCRAMJET PERFORMANCE ANALYSIS (H₂ fuel, φ={phi})")
    print(f"  Flight: h={h_km} km, Ma₀={Ma0}, ṁ_air={mdot} kg/s")
    print(f"  CEA-backed equilibrium combustion + integral-form Tt/Pt")
    print(f"{'═'*65}")

    inp  = eng.inlet_properties(h=h_km*1e3, Ma=Ma0, m_air=mdot)
    iso  = eng.isolator_properties(inp)
    sec2 = eng.combustor_properties2(iso)
    sec3 = eng.combustor_properties3(sec2, phi=phi)

    # def find_alpha14_for_choke_at_exit(engine, sec3, tol=0.1):
    #     """
    #     Binary-search alpha14 so the thermal choke occurs at x ≈ L34 (combustor exit).
    #     Returns the found alpha14 and the choke position.

    #     Logic
    #     -----
    #     - thermal_choke=True  AND  x_last < L34 - tol  →  duct too narrow  → increase alpha14
    #     - thermal_choke=True  AND  x_last ≥ L34 - tol  →  choke at exit    → done
    #     - thermal_choke=False                           →  duct too wide    → decrease alpha14
    #     """
    #     L34 = float(engine.L34)
    #     alpha13 = float(engine.alpha13)

    #     lo = alpha13 * 1.00   # lower bound: constant-area combustor
    #     hi = alpha13 * 8.00   # upper bound: very diverging (won't choke)

    #     best_alpha14 = None
    #     best_x_choke = None

    #     print(f"\n── Bisection for alpha14 (target: choke at x ≈ {L34:.3f} m) ──")

    #     for iteration in range(60):
    #         mid = 0.5 * (lo + hi)
    #         engine.alpha14 = mid

    #         result4 = engine.combustor_properties4(sec3)
    #         sol     = result4["solution"]
    #         x_last  = float(sol["x"][-1])
    #         choked  = result4["thermal_choke"]

    #         print(f"  iter {iteration:2d}:  alpha14={mid:.4f}  AR={mid/alpha13:.3f}"
    #             f"  x_last={x_last:.4f}  choked={choked}")

    #         if not choked:
    #             # Duct diverges too fast — Ma never reaches 1 — shrink upper bound
    #             hi = mid
    #             continue

    #         # Choke occurred; record best candidate
    #         if best_alpha14 is None or abs(x_last - L34) < abs(best_x_choke - L34):
    #             best_alpha14 = mid
    #             best_x_choke = x_last

    #         if x_last >= L34 - tol:
    #             # Choke is at or past the desired exit
    #             hi = mid
    #         else:
    #             # Choke too early — need more divergence
    #             lo = mid

    #         if (hi - lo) < 1e-4:
    #             break

    #     print(f"\n  ✓  alpha14 = {best_alpha14:.4f}  "
    #         f"(A4/A3 = {best_alpha14/alpha13:.3f})  "
    #         f"choke at x = {best_x_choke:.4f} m")
    #     engine.alpha14 = best_alpha14
    #     return best_alpha14, best_x_choke





    # alpha14_opt, x_choke = find_alpha14_for_choke_at_exit(eng, sec3)
    # print(f"  → use engine.alpha14 = {alpha14_opt:.4f}")

    sec4 = eng.combustor_properties4(sec3)

    if sec4["thermal_choke"]:
        print(f"  ✓ Thermal choke at combustor exit — Ma4 = {sec4['Ma4']:.4f}  "
          f"(combustor exit = nozzle throat, this is the design intent)")
    
    sec5 = eng.nozzle_properties(sec4, inp)
    perf = eng.performance(inp, sec5, sec3)

    print_section("Section 0 — Freestream", inp, [
        ("Mach number Ma₀",        "Ma0", "—",   1.0),
        ("Temperature T₀",         "T0",  "K",   1.0),
        ("Pressure P₀",            "P0",  "kPa", 1e-3),
        ("Velocity V₀",            "V0",  "m/s", 1.0),
        ("Tt0 (integral)",         "Tt0", "K",   1.0),
        ("Pt0 (integral)",         "Pt0", "kPa", 1e-3),
    ])

    print_section("Section 1 — Isolator entrance", iso, [
        ("Mach number Ma₁",        "Ma1", "—",   1.0),
        ("Temperature T₁",         "T1",  "K",   1.0),
        ("Pressure p₁",            "p1",  "kPa", 1e-3),
        ("Velocity V₁",            "V1",  "m/s", 1.0),
        ("Tt₁",                    "Tt1", "K",   1.0),
        ("Pt₁",                    "Pt1", "kPa", 1e-3),
        ("Pressure recovery σc",   "sigma_c", "—", 1.0),
    ])

    print_section("Section 3 — Combustor fuel injection exit", sec3, [
        ("Mach number Ma₃",        "Ma3", "—",   1.0),
        ("Temperature T₃",         "T3",  "K",   1.0),
        ("Pressure p₃",            "p3",  "kPa", 1e-3),
        ("Velocity V₃",            "V3",  "m/s", 1.0),
        ("Tt₃",                    "Tt3", "K",   1.0),
        ("Pt₃",                    "Pt3", "kPa", 1e-3),
    ])

    print_section("Section 4 — Combustor exit", sec4, [
        ("Mach number Ma₄",        "Ma4", "—",   1.0),
        ("Temperature T₄",         "T4",  "K",   1.0),
        ("Pressure p₄",            "p4",  "kPa", 1e-3),
        ("Velocity V₄",            "V4",  "m/s", 1.0),
        ("Tt₄",                    "Tt4", "K",   1.0),
        ("Pt₄",                    "Pt4", "kPa", 1e-3),
        ("h₄ (static)",            "h4",  "MJ/kg", 1e-6),
        ("ht₄ (total)",            "ht4", "MJ/kg", 1e-6),
    ])

    print_section("Section 5 — Nozzle exit", sec5, [
        ("Mach number Ma₅",        "Ma5", "—",   1.0),
        ("Temperature T₅",         "T5",  "K",   1.0),
        ("Pressure p₅",            "p5",  "kPa", 1e-3),
        ("Velocity V₅",            "V5",  "m/s", 1.0),
        ("Tt₅",                    "Tt5", "K",   1.0),
        ("Pt₅",                    "Pt5", "kPa", 1e-3),
    ])

    print_section("PERFORMANCE METRICS", perf, [
        ("Internal thrust Fin",    "Fin", "N",      1.0),
        ("Specific impulse Isp",  "Isp", "s",      1.0),
        ("Specific thrust Ia",     "Ia",  "N·s/kg", 1.0),
    ])

    print(f"\n  Fuel: H₂,  φ={phi},  FAR={sec3['mfuel']/mdot:.5f}")
    print(f"  ṁ_fuel = {sec3['mfuel']:.6f} kg/s")
    print(f"\n  Sec4 inlet of_ratio = {(mdot)/(sec3['mfuel']+1e-30):.2f}")
    print(f"  Sec4 exit composition (top 5 mass fractions):")
    Y4 = sec4["Y"]
    top5 = sorted(Y4.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for sp, y in top5:
        print(f"    {sp:>4}: {y:.4f}")

    eng.plot_flowpath(inp, iso, sec2, sec3, sec4, sec5)

    