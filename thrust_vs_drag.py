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

# =============================================================================
# Ramjet thrust sweep and plotting
# =============================================================================
#
# This section uses the Ramjet class above, but instead of running only one
# altitude/Mach point, it runs many points and creates thrust-vs-Mach plots.
#
# Output:
#   - Ramjet thrust curves for several altitudes
#   - Optional contour plot: thrust as function of Mach and altitude
#   - Dictionary containing the thrust map and the sampled points
#
# NOTE:
# The ramjet model uses combustor_properties4(), which requires NASA CEA.
# If the package `cea` is not installed, the full combustor calculation cannot run.
# =============================================================================

def run_ramjet_single_point(
    h_km: float,
    Ma0: float,
    mdot: float = 100.0,
    phi: float = 0.5,
    suppress_output: bool = True,
) -> dict[str, float | bool | str]:
    """
    Run the ramjet model for one altitude and one freestream Mach number.

    Parameters
    ----------
    h_km : float
        Altitude in km.

    Ma0 : float
        Freestream Mach number.

    mdot : float
        Captured air mass flow rate [kg/s].

    phi : float
        Equivalence ratio.

    suppress_output : bool
        If True, suppress the detailed print output from every internal section.

    Returns
    -------
    dict
        Contains thrust and performance outputs. If the case fails, returns NaNs
        and stores the error message.
    """
    import contextlib
    import io

    def _run():
        eng = Ramjet()

        inp = eng.inlet_properties(
            h=h_km * 1000.0,
            Ma=Ma0,
            m_air=mdot,
        )

        iso = eng.isolator_properties(inp)
        sec2 = eng.combustor_properties2(iso)
        sec3 = eng.combustor_properties3(sec2, phi=phi)
        sec4 = eng.combustor_properties4(sec3)
        sec5 = eng.nozzle_properties(sec4, inp)
        perf = eng.performance(inp, sec5, sec3)

        return eng, inp, iso, sec2, sec3, sec4, sec5, perf

    try:
        import warnings

        # Treat numerical RuntimeWarnings as failed cases.
        # This prevents invalid points from contaminating the plot.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            if suppress_output:
                with contextlib.redirect_stdout(io.StringIO()):
                    eng, inp, iso, sec2, sec3, sec4, sec5, perf = _run()
            else:
                eng, inp, iso, sec2, sec3, sec4, sec5, perf = _run()

        thrust_N = float(perf.get("Fin", np.nan))
        isp_s = float(perf.get("Isp", np.nan))
        ia = float(perf.get("Ia", np.nan))
        mfuel = float(perf.get("mfuel", np.nan))

        if not np.isfinite(thrust_N):
            raise ValueError("Non-finite thrust result.")

        return {
            "success": True,
            "error": "",
            "Altitude_km": float(h_km),
            "Mach": float(Ma0),
            "Thrust_N": thrust_N,
            "Thrust_kN": thrust_N / 1000.0,
            "Isp_s": isp_s,
            "Ia_Ns_per_kg": ia,
            "mdot_air_kg_s": float(mdot),
            "mfuel_kg_s": mfuel,
            "phi": float(phi),
            "Ma_exit": float(sec5.get("Ma5", np.nan)),
            "T_exit_K": float(sec5.get("T5", np.nan)),
            "p_exit_Pa": float(sec5.get("p5", np.nan)),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "Altitude_km": float(h_km),
            "Mach": float(Ma0),
            "Thrust_N": np.nan,
            "Thrust_kN": np.nan,
            "Isp_s": np.nan,
            "Ia_Ns_per_kg": np.nan,
            "mdot_air_kg_s": float(mdot),
            "mfuel_kg_s": np.nan,
            "phi": float(phi),
            "Ma_exit": np.nan,
            "T_exit_K": np.nan,
            "p_exit_Pa": np.nan,
        }


def run_ramjet_thrust_sweep(
    altitudes_km: np.ndarray | list[float] | None = None,
    mach_values: np.ndarray | list[float] | None = None,
    mdot: float = 100.0,
    phi: float = 0.5,
    suppress_output: bool = True,
) -> tuple[np.ndarray, list[dict[str, float | bool | str]]]:
    """
    Run the ramjet model over a grid of altitude and Mach values.

    Returns
    -------
    thrust_map_N : np.ndarray
        Shape: [altitude index, Mach index]

    rows : list[dict]
        One dictionary per sampled point.
    """
    if not _HAS_CEA:
        raise ImportError(
            "NASA CEA is not installed. This ramjet model needs `cea` for "
            "combustor_properties4(). Install it first with: pip install cea"
        )

    if altitudes_km is None:
        # Ramjet-relevant transition band.
        altitudes_km = np.array([12, 14, 16, 18, 20, 22], dtype=float)

    if mach_values is None:
        # Ramjet-relevant speed range. You can make this finer later.
        mach_values = np.linspace(2.0, 5.0, 13)

    altitudes_km = np.asarray(altitudes_km, dtype=float)
    mach_values = np.asarray(mach_values, dtype=float)

    thrust_map_N = np.full((len(altitudes_km), len(mach_values)), np.nan)
    rows = []

    total_cases = len(altitudes_km) * len(mach_values)
    case_counter = 0

    print()
    print("Running ramjet thrust sweep")
    print("---------------------------")
    print(f"Altitudes [km]: {altitudes_km}")
    print(f"Mach values:    {mach_values}")
    print(f"mdot = {mdot:.2f} kg/s, phi = {phi:.3f}")
    print()

    for i, h_km in enumerate(altitudes_km):
        for j, M in enumerate(mach_values):
            case_counter += 1

            result = run_ramjet_single_point(
                h_km=h_km,
                Ma0=M,
                mdot=mdot,
                phi=phi,
                suppress_output=suppress_output,
            )

            rows.append(result)
            thrust_map_N[i, j] = result["Thrust_N"]

            if result["success"]:
                print(
                    f"[{case_counter:03d}/{total_cases:03d}] "
                    f"h={h_km:5.1f} km | M={M:4.2f} | "
                    f"T={result['Thrust_kN']:10.3f} kN"
                )
            else:
                print(
                    f"[{case_counter:03d}/{total_cases:03d}] "
                    f"h={h_km:5.1f} km | M={M:4.2f} | FAILED: {result['error']}"
                )

    return thrust_map_N, rows


def plot_ramjet_thrust_vs_mach_sweep(
    altitudes_km: np.ndarray | list[float] | None = None,
    mach_values: np.ndarray | list[float] | None = None,
    mdot: float = 100.0,
    phi: float = 0.5,
    suppress_output: bool = True,
    make_contour: bool = True,
    save_csv: bool = True,
    csv_filename: str = "ramjet_thrust_sweep_results.csv",
) -> dict[str, np.ndarray | list[dict[str, float | bool | str]]]:
    """
    Calculate and plot ramjet thrust versus Mach for multiple altitudes.

    This is the main function you should run.
    """
    if altitudes_km is None:
        altitudes_km = np.array([12, 14, 16, 18, 20, 22], dtype=float)

    if mach_values is None:
        mach_values = np.linspace(2.0, 5.0, 13)

    altitudes_km = np.asarray(altitudes_km, dtype=float)
    mach_values = np.asarray(mach_values, dtype=float)

    thrust_map_N, rows = run_ramjet_thrust_sweep(
        altitudes_km=altitudes_km,
        mach_values=mach_values,
        mdot=mdot,
        phi=phi,
        suppress_output=suppress_output,
    )

    thrust_map_kN = thrust_map_N / 1000.0

    failed_rows = [r for r in rows if not r["success"]]
    if failed_rows:
        print()
        print("Failed / skipped cases")
        print("----------------------")
        for r in failed_rows:
            print(
                f"h={r['Altitude_km']:5.1f} km | "
                f"M={r['Mach']:4.2f} | "
                f"{r['error']}"
            )

    # -------------------------------------------------------------------------
    # Plot 1: thrust vs Mach, one curve per altitude
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for i, h_km in enumerate(altitudes_km):
        valid = np.isfinite(thrust_map_kN[i, :])

        if np.count_nonzero(valid) == 0:
            print(f"No valid points to plot for h = {h_km:.1f} km.")
            continue

        plt.plot(
            mach_values[valid],
            thrust_map_kN[i, valid],
            marker="o",
            linewidth=2.0,
            label=f"h = {h_km:.0f} km",
        )

        # Show failed points on the x-axis as crosses.
        invalid = ~valid
        if np.any(invalid):
            plt.scatter(
                mach_values[invalid],
                np.zeros(np.count_nonzero(invalid)),
                marker="x",
                s=60,
                label=f"failed, h={h_km:.0f} km",
            )

    plt.xlabel("Mach number [-]")
    plt.ylabel("Ramjet thrust [kN]")
    plt.title(f"Ramjet thrust vs Mach, mdot={mdot:.0f} kg/s, phi={phi:.2f}")
    plt.grid(True, alpha=0.35)
    plt.legend(title="Altitude")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot 2: optional contour map
    # -------------------------------------------------------------------------
    if make_contour:
        M_grid, H_grid = np.meshgrid(mach_values, altitudes_km)

        if np.count_nonzero(np.isfinite(thrust_map_kN)) >= 4:
            plt.figure(figsize=(10, 6))

            masked_thrust = np.ma.masked_invalid(thrust_map_kN)
            contour = plt.contourf(M_grid, H_grid, masked_thrust, levels=30)
            plt.colorbar(contour).set_label("Ramjet thrust [kN]")

            # Show the actual computed sample points.
            valid_points = np.isfinite(thrust_map_kN)
            failed_points = ~valid_points

            plt.scatter(M_grid[valid_points], H_grid[valid_points], s=22, color="black", alpha=0.65, label="valid")
            if np.any(failed_points):
                plt.scatter(M_grid[failed_points], H_grid[failed_points], s=45, marker="x", color="black", label="failed")

            plt.xlabel("Mach number [-]")
            plt.ylabel("Altitude [km]")
            plt.title(f"Ramjet thrust map, mdot={mdot:.0f} kg/s, phi={phi:.2f}")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("Not enough valid points for a contour plot.")

    # -------------------------------------------------------------------------
    # Save numerical results, useful for your report or later interpolation.
    # -------------------------------------------------------------------------
    if save_csv:
        try:
            import pandas as pd
            table = pd.DataFrame(rows)
            table.to_csv(csv_filename, index=False)
            print(f"\nSaved sweep table to: {csv_filename}")
        except Exception as e:
            print(f"\nCould not save CSV because pandas failed: {e}")

    return {
        "altitudes_km": altitudes_km,
        "mach_values": mach_values,
        "thrust_map_N": thrust_map_N,
        "thrust_map_kN": thrust_map_kN,
        "rows": rows,
    }


# =============================================================================
# Run example
# =============================================================================

# =============================================================================
# Turbojet polynomial model for comparison
# =============================================================================
#
# The ramjet thrust below comes from the full Ramjet cycle model above.
# The turbojet thrust is still taken from your polynomial EngineSim-fit model.
# =============================================================================

from scipy.interpolate import PchipInterpolator


TURBO_THRUST_POLY_DATA = {
    # altitude_m: (a, b, c)
    0.0:     (1252.7,  -861.57, 1272.0),
    5000.0:  (1091.0, -1214.8,  1003.1),
    10000.0: (787.87, -1088.9,  705.08),
    15000.0: (439.92, -742.89,  465.0),
}


_turbo_alts = np.array(sorted(TURBO_THRUST_POLY_DATA.keys()), dtype=float)

_TURBO_A_INTERP = PchipInterpolator(
    _turbo_alts,
    [TURBO_THRUST_POLY_DATA[h][0] for h in _turbo_alts],
    extrapolate=True,
)

_TURBO_B_INTERP = PchipInterpolator(
    _turbo_alts,
    [TURBO_THRUST_POLY_DATA[h][1] for h in _turbo_alts],
    extrapolate=True,
)

_TURBO_C_INTERP = PchipInterpolator(
    _turbo_alts,
    [TURBO_THRUST_POLY_DATA[h][2] for h in _turbo_alts],
    extrapolate=True,
)


def turbo_thrust_curve_vs_mach(
    altitude_m: float,
    mach_values: np.ndarray,
    clamp_negative_thrust: bool = True,
) -> np.ndarray:
    """
    Turbojet thrust curve from the polynomial EngineSim fit.

    NOTE:
    This returns the same units as the polynomial coefficients.
    Your coefficients look like kN-scale values, so the plot label uses kN.
    """
    altitude_m = float(altitude_m)

    a = float(_TURBO_A_INTERP(altitude_m))
    b = float(_TURBO_B_INTERP(altitude_m))
    c = float(_TURBO_C_INTERP(altitude_m))

    M = np.asarray(mach_values, dtype=float)

    thrust = a * M**2 + b * M + c

    if clamp_negative_thrust:
        thrust = np.maximum(thrust, 0.0)

    return thrust


def turbo_thrust_at_mach_altitudes(
    mach_fixed: float,
    altitude_values_m: np.ndarray,
    clamp_negative_thrust: bool = True,
) -> np.ndarray:
    """
    Turbojet thrust at one fixed Mach number for many altitudes.
    """
    return np.array([
        turbo_thrust_curve_vs_mach(
            altitude_m=h,
            mach_values=np.array([mach_fixed]),
            clamp_negative_thrust=clamp_negative_thrust,
        )[0]
        for h in altitude_values_m
    ])


def ramjet_cycle_thrust_at_mach_altitudes(
    mach_fixed: float,
    altitude_values_m: np.ndarray,
    mdot: float = 100.0,
    phi: float = 0.5,
    n_ramjet_engines: int = 2,
    suppress_output: bool = True,
) -> np.ndarray:
    """
    Full ramjet-cycle thrust at one fixed Mach number for many altitudes.

    Uses run_ramjet_single_point() from the Ramjet model above.

    The ramjet model returns thrust for ONE ramjet engine.
    This function multiplies it by n_ramjet_engines.

    Returned thrust is in kN.
    """
    thrust_kN = []

    print()
    print(f"Full ramjet-cycle thrust at Mach {mach_fixed:g}")
    print("-----------------------------------------")
    print(f"Ramjet engines included: {n_ramjet_engines}")
    print(f"{'Altitude [km]':>14s} {'One engine [kN]':>18s} {'Total [kN]':>16s} {'Status':>12s}")

    for h_m in altitude_values_m:
        h_km = h_m / 1000.0

        out = run_ramjet_single_point(
            h_km=h_km,
            Ma0=mach_fixed,
            mdot=mdot,
            phi=phi,
            suppress_output=suppress_output,
        )

        if out["success"]:
            T_one_engine_kN = float(out["Thrust_kN"])
            T_total_kN = n_ramjet_engines * T_one_engine_kN
            status = "OK"
        else:
            T_one_engine_kN = np.nan
            T_total_kN = np.nan
            status = "FAILED"

        thrust_kN.append(T_total_kN)

        print(f"{h_km:14.2f} {T_one_engine_kN:18.3f} {T_total_kN:16.3f} {status:>12s}")

        if not out["success"]:
            print(f"    Error: {out['error']}")

    return np.array(thrust_kN, dtype=float)


# =============================================================================
# Color-coded Mach-3 comparison plot
# =============================================================================

def plot_turbo_polynomial_and_ramjet_cycle_mach_line(
    altitude_values_m: np.ndarray | None = None,
    mach_fixed: float = 3.0,
    mach_min: float = 0.0,
    mach_max: float = 6.0,
    n_mach: int = 800,
    mdot: float = 100.0,
    phi: float = 0.5,
    n_ramjet_engines: int = 2,
    suppress_output: bool = True,
    clamp_negative_turbo_thrust: bool = True,
) -> dict[str, np.ndarray]:
    """
    Plot turbojet thrust-vs-Mach curves and add full ramjet-cycle thrust points
    at a fixed Mach number.

    Color coding:
        - Each altitude gets one color.
        - Turbojet point and full ramjet-cycle point at the same altitude use
          the same color.
        - Turbojet point = circle marker.
        - Ramjet-cycle point = square marker.
        - Same-altitude points are connected with a thick colored line.

    The ramjet thrust is NOT the old polynomial ramjet fit here.
    It is calculated with the full Ramjet cycle model.

    The ramjet cycle model returns thrust for one engine, so this function
    multiplies ramjet thrust by n_ramjet_engines.
    """
    if altitude_values_m is None:
        altitude_values_m = np.linspace(12_000.0, 22_000.0, 11)

    altitude_values_m = np.asarray(altitude_values_m, dtype=float)
    M_values = np.linspace(mach_min, mach_max, n_mach)

    turbo_at_mach = turbo_thrust_at_mach_altitudes(
        mach_fixed=mach_fixed,
        altitude_values_m=altitude_values_m,
        clamp_negative_thrust=clamp_negative_turbo_thrust,
    )

    ramjet_cycle_at_mach = ramjet_cycle_thrust_at_mach_altitudes(
        mach_fixed=mach_fixed,
        altitude_values_m=altitude_values_m,
        mdot=mdot,
        phi=phi,
        n_ramjet_engines=n_ramjet_engines,
        suppress_output=suppress_output,
    )

    print()
    print(f"Turbo polynomial vs full ramjet-cycle thrust at Mach {mach_fixed:g}")
    print("----------------------------------------------------------------")
    print(f"Ramjet thrust is total for {n_ramjet_engines} engines.")
    print(f"{'Altitude [km]':>14s} {'Turbo poly':>16s} {'Ramjet total':>16s} {'Turbo - Ramjet':>18s}")

    for h, T_turbo, T_ram in zip(altitude_values_m, turbo_at_mach, ramjet_cycle_at_mach):
        diff = T_turbo - T_ram if np.isfinite(T_ram) else np.nan
        print(f"{h / 1000.0:14.2f} {T_turbo:16.3f} {T_ram:16.3f} {diff:18.3f}")

    fig, ax = plt.subplots(figsize=(13, 8))

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(altitude_values_m) - 1, 1)) for i in range(len(altitude_values_m))]

    # -------------------------------------------------------------------------
    # Background turbojet polynomial curves.
    # These show turbo thrust vs Mach at each altitude.
    # -------------------------------------------------------------------------
    for i, h in enumerate(altitude_values_m):
        color = colors[i]

        T_turbo_curve = turbo_thrust_curve_vs_mach(
            altitude_m=h,
            mach_values=M_values,
            clamp_negative_thrust=clamp_negative_turbo_thrust,
        )

        ax.plot(
            M_values,
            T_turbo_curve,
            linestyle="-",
            linewidth=1.6,
            alpha=0.55,
            color=color,
        )

    # -------------------------------------------------------------------------
    # Mach = fixed vertical line.
    # -------------------------------------------------------------------------
    ax.axvline(
        mach_fixed,
        color="black",
        linestyle=":",
        linewidth=2.5,
        label=f"Mach {mach_fixed:g}",
        zorder=8,
    )

    # -------------------------------------------------------------------------
    # Color-coded altitude pairs at Mach fixed.
    # Small x-offset avoids overplotting all markers on exactly the same x.
    # -------------------------------------------------------------------------
    offsets = np.linspace(-0.055, 0.055, len(altitude_values_m))

    for i, (h, offset, T_turbo, T_ramjet) in enumerate(
        zip(altitude_values_m, offsets, turbo_at_mach, ramjet_cycle_at_mach)
    ):
        color = colors[i]
        x_plot = mach_fixed + offset
        h_km = h / 1000.0

        # Turbo point
        ax.scatter(
            x_plot,
            T_turbo,
            marker="o",
            s=90,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=12,
        )

        # Only draw ramjet point if valid
        if np.isfinite(T_ramjet):
            ax.scatter(
                x_plot,
                T_ramjet,
                marker="s",
                s=90,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=12,
            )

            # Same-altitude pair connector
            ax.plot(
                [x_plot, x_plot],
                [T_turbo, T_ramjet],
                color=color,
                linewidth=3.0,
                alpha=0.95,
                zorder=11,
            )

            T_mid = 0.5 * (T_turbo + T_ramjet)
            label_text = f"{h_km:.0f} km"
        else:
            T_mid = T_turbo
            label_text = f"{h_km:.0f} km\nram fail"

        ax.text(
            x_plot + 0.035,
            T_mid,
            label_text,
            va="center",
            ha="left",
            fontsize=9,
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                alpha=0.9,
                pad=1.6,
            ),
            zorder=13,
        )

    # -------------------------------------------------------------------------
    # Custom legend.
    # -------------------------------------------------------------------------
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.8,
               label="turbo polynomial curve"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="gray",
               markeredgecolor="black", markersize=9,
               label=f"turbo polynomial at M={mach_fixed:g}"),
        Line2D([0], [0], marker="s", color="white", markerfacecolor="gray",
               markeredgecolor="black", markersize=9,
               label=f"full ramjet cycle, {n_ramjet_engines} engines, at M={mach_fixed:g}"),
        Line2D([0], [0], color="gray", linewidth=3,
               label="same-altitude connector"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2.5,
               label=f"Mach {mach_fixed:g}"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
    )

    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("Thrust [kN]")
    ax.set_title(
        f"Turbojet polynomial curves with full ramjet-cycle thrust points at Mach {mach_fixed:g} ({n_ramjet_engines} ramjets)"
    )
    ax.grid(True, alpha=0.35)
    ax.set_xlim(mach_min, mach_max)

    fig.tight_layout()
    plt.show()

    return {
        "altitude_m": altitude_values_m,
        "altitude_km": altitude_values_m / 1000.0,
        "mach_fixed": np.full_like(altitude_values_m, mach_fixed, dtype=float),
        "turbo_polynomial_thrust_kN": turbo_at_mach,
        "ramjet_cycle_total_thrust_kN": ramjet_cycle_at_mach,
        "difference_turbo_minus_ramjet_total_kN": turbo_at_mach - ramjet_cycle_at_mach,
    }


# =============================================================================
# Run example
# =============================================================================

# =============================================================================
# Scramjet model from uploaded code, renamed internally to avoid class conflicts
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import pandas as pd



# Optional NASA CEA — required by combustor_properties4 (per-step equilibrium).
try:
    import cea as _CEA           # noqa: F401
    _HAS_CEA = True
except Exception:                # noqa: BLE001
    _HAS_CEA = False


# ---------------------------------------------------------------------------
# Thermochemistry data + simple air dissociation model
# ---------------------------------------------------------------------------
class ScramAirProperties:
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
# Works for any mass-fraction dict over species that appear in ScramAirProperties.NASA_DATA.
# Used to compute h, s and to find stagnation Tt, Pt in *integral* form
# (so variable Cp / composition is handled correctly).
# ---------------------------------------------------------------------------
class ScramMixtureNASA:
    R_UNIVERSAL = 8.314462618   # J/(mol·K)
    P_REF       = 101325.0       # Pa (1 atm, NASA standard)

    def __init__(self, air_props: ScramAirProperties):
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
# Returns per-species mass fractions in the same name space ScramMixtureNASA uses,
# so the two can be composed seamlessly (CEA gives composition, ScramMixtureNASA
# gives h / s / cp / Tt / Pt).
# ---------------------------------------------------------------------------
class ScramCEAComp:
    PROD_NAMES = ["Ar", "CO2", "H", "H2", "H2O",
                  "N",  "NO",  "N2", "O", "O2", "OH"]  # all in ScramAirProperties.NASA_DATA

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
# ScramAtmosphere (unchanged)
# ---------------------------------------------------------------------------
class ScramAtmosphere:
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
        h0, L, T0, _ = ScramAtmosphere._layer(h)
        return T0 + L*(h - h0)

    @staticmethod
    def P(h):
        h0, L, T0, P0 = ScramAtmosphere._layer(h)
        dh = h - h0; T = T0 + L*dh
        if L != 0:
            return P0*(T/T0)**(-ScramAtmosphere.G0/(L*ScramAtmosphere.R_AIR))
        return P0*np.exp(-ScramAtmosphere.G0*dh/(ScramAtmosphere.R_AIR*T0))

    @staticmethod
    def rho(h):
        return ScramAtmosphere.P(h)/(ScramAtmosphere.R_AIR*ScramAtmosphere.T(h))


# ---------------------------------------------------------------------------
# Shapiro generalised-1D ODE.
# - `derivatives` keeps the per-phenomenon `switches` toggle.
# - `integrate` now optionally takes a `state_fn(T, p, V, x)` that returns
#   {h, s, ht, st, Tt, Pt}; if provided, those override the simple
#   constant-Cp Tt/Pt formulas in the result.
# ---------------------------------------------------------------------------
class ScramShapiroODE:
    @staticmethod
    def derivatives(Ma2, p, T, gamma, Cp, dA_dx, A, D, Cf,
                    dH_dx, mdot, dmdot_dx, W, dW_dx, dgamma_dx,
                    switches=None):
        if switches is None:
            switches = {
                "area": True,
                "friction": True,
                "mass": True,
                "heat": True,
                "MW": True,
                "gamma": True,
            }
        on = lambda key: 1.0 if switches.get(key, True) else 0.0

        g = gamma
        M2 = Ma2
        D1 = 1.0 - M2
        if abs(D1) < 1e-8:
            D1 = 1e-8 if D1 >= 0 else -1e-8

        g1m2 = 1.0 + (g - 1.0)/2.0 * M2
        gM2  = g * M2
        fric = 4.0 * Cf / D
        heat = dH_dx / (Cp * T)

        # --- dMa²/dx -----------------------------------------------------
        dMa2_dx = M2 * (
            -(2.0 * g1m2 / D1) * (dA_dx / A)        * on("area")
            + ((1.0 + gM2) / D1) * heat              * on("heat")
            + (gM2 * g1m2 / D1) * fric               * on("friction")
            + (2.0 * (1.0 + gM2) * g1m2 / D1) * (dmdot_dx / mdot) * on("mass")
            - ((1.0 + gM2) / D1) * (dW_dx / W)       * on("MW")
            - (dgamma_dx / g)                         * on("gamma")
        )

        # --- dp/dx -------------------------------------------------------
        dp_dx = p * (
            (gM2 / D1) * (dA_dx / A)                 * on("area")
            - (gM2 / D1) * heat                       * on("heat")
            - (gM2 * (1.0 + (g - 1.0) * M2) / (2.0 * D1)) * fric * on("friction")
            - (2.0 * gM2 * g1m2 / D1) * (dmdot_dx / mdot) * on("mass")
            + (gM2 / D1) * (dW_dx / W)               * on("MW")
        )

        # --- dT/dx (paper Eq. 17 sign convention; see comment in earlier
        #            revisions for the (1−γM²) alternative) ---------------
        dT_dx = T * (
            ((g - 1.0) * M2 / D1) * (dA_dx / A)       * on("area")
            + ((1.0 + gM2) / D1) * heat                * on("heat")
            - (g * (g - 1.0) * M2**2 / (2.0 * D1)) * fric * on("friction")
            - ((g - 1.0) * M2 * (1.0 + gM2) / D1) * (dmdot_dx / mdot) * on("mass")
            + ((g - 1.0) * M2 / D1) * (dW_dx / W)     * on("MW")
        )

        return dMa2_dx, dp_dx, dT_dx

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
        Solves the generalised 1D Shapiro flow with **energy-consistent T**.

        State variables are ``(Ma², p, h_t, ṁ)`` — stagnation enthalpy `h_t`
        replaces the static temperature in the state vector. At every rhs
        evaluation, T is recovered from `h_t` by Newton iteration on

            h_mix(Y(x,T,p), T) + ½ · Ma² · γ(Y,T) · R(Y) · T  =  h_t.

        This bypasses the (potentially form-dependent) `dT/dx` Shapiro
        coefficient: the Mach and pressure ODEs come from Shapiro as before,
        but temperature is *always* the one that satisfies energy + Mach.

        Parameters
        ----------
        composition_fn : callable(x, T, p) -> dict
            Returns local mass fractions {species: Y}. For sections without
            chemistry change, it can ignore T and p.
        source_fn      : callable(x, T, p, mdot, Y) -> (dH_dx, dmdot_dx)
            External heat addition [J/(kg·m)] and mass injection [kg/(s·m)].
        mix            : ScramMixtureNASA  — used for h, s, cp, W, γ from NASA polys.
        state_fn       : optional override for the h/s/Tt/Pt post-processing.
        switches       : per-phenomenon toggles forwarded to `derivatives`.
        """
        R_UNIV = mix.R_UNIVERSAL

        # ---- Initial h_t from inlet (T_in, p_in, Ma2_in, composition) ----
        Y_in     = composition_fn(x_start, T_in, p_in)
        cp_in    = mix.cp_mix(Y_in, T_in)
        W_in     = mix.W_mix(Y_in)
        R_in     = R_UNIV / W_in
        gamma_in = cp_in / max(cp_in - R_in, 1e-30)
        V2_in    = max(Ma2_in, 1.000001) * gamma_in * R_in * T_in
        h_in     = mix.h_mix(Y_in, T_in)
        ht_in    = h_in + 0.5 * V2_in

        T_cache = {"T": float(T_in)}

        # ---- Newton: T such that h(T,Y(T,p)) + ½ M² γ(T,Y) R(Y) T = h_t --
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
                # ∂h_t/∂T ≈ cp + ½ M² γ R   (γ and R only weakly T-dependent)
                deriv = cp + 0.5 * M2 * gamma * R
                if deriv <= 0:
                    break
                step = -resid / deriv
                # damp huge steps for stability
                if   step >  400.0: step =  400.0
                elif step < -200.0: step = -200.0
                T_new = max(200.0, min(6000.0, T + step))
                if abs(T_new - last_T) < 0.5:
                    T = T_new
                    Y      = composition_fn(x, T, p)
                    cp     = mix.cp_mix(Y, T)
                    W      = mix.W_mix(Y)
                    R      = R_UNIV / W
                    gamma  = cp / max(cp - R, 1e-30)
                    V2     = M2 * gamma * R * T
                    break
                last_T = T
                T = T_new
            T_cache["T"] = T
            return T, Y, cp, W, R, gamma, V2

        # Resolve switch mask once. The same dict gates BOTH the Shapiro
        # influence coefficients (inside `derivatives`) AND the underlying
        # physical sources here, so that disabling a phenomenon really
        # removes it from the simulation — not just from Mach/pressure.
        sw_heat = True if switches is None else switches.get("heat",  True)
        sw_mass = True if switches is None else switches.get("mass",  True)
        sw_MW   = True if switches is None else switches.get("MW",    True)
        sw_gam  = True if switches is None else switches.get("gamma", True)
        # `area` and `friction` are pure Shapiro-coefficient effects with
        # no external source term — the mask inside `derivatives` is enough.

        # ---------------- rhs ---------------------------------------------
        def rhs(x, y):
            M2, p, ht, mdot = y
            T, Y, cp, W, R, gamma, V2 = solve_T(M2, p, ht, x)

            A, dA_dx, D = geometry_fn(x)
            dH_dx, dmdot_dx = source_fn(x, T, p, mdot, Y)

            # ---- gate SOURCE TERMS by heat / mass switches ---------------
            #   heat=False ⇒ no energy enters the energy equation
            #                (dh_t/dx contribution from chemistry / external Q → 0)
            #   mass=False ⇒ no mass is injected (ṁ stays constant)
            if not sw_heat: dH_dx    = 0.0
            if not sw_mass: dmdot_dx = 0.0

            # ---- composition spatial derivatives at fixed (T, p) ---------
            # Zero them out if either compositional switch is off so the
            # diagnostic isolation is complete (no MW or γ drift in Shapiro).
            if sw_MW or sw_gam:
                dx_step = 1e-4
                x_p = min(x + dx_step, x_end); x_m = max(x - dx_step, x_start)
                span = x_p - x_m
                if span > 0:
                    Y_p = composition_fn(x_p, T, p)
                    Y_m = composition_fn(x_m, T, p)
                    dW_dx     = (mix.W_mix(Y_p)        - mix.W_mix(Y_m))        / span if sw_MW  else 0.0
                    dgamma_dx = (mix.gamma_mix(Y_p, T) - mix.gamma_mix(Y_m, T)) / span if sw_gam else 0.0
                else:
                    dW_dx = 0.0; dgamma_dx = 0.0
            else:
                dW_dx = 0.0; dgamma_dx = 0.0

            # Shapiro Ma² and p derivatives — Shapiro's dT/dx is ignored
            # (T comes from energy conservation via Newton on h_t).
            dM2_dx, dp_dx, _ = ScramShapiroODE.derivatives(
                Ma2=M2, p=p, T=T, gamma=gamma, Cp=cp,
                dA_dx=dA_dx, A=A, D=D, Cf=Cf,
                dH_dx=dH_dx,
                mdot=mdot, dmdot_dx=dmdot_dx,
                W=W, dW_dx=dW_dx, dgamma_dx=dgamma_dx,
                switches=switches,
            )

            # Energy equation: dh_t/dx = (external heat per unit mass per length)
            # Mass injection at flow's local stagnation enthalpy contributes 0
            # (Shapiro convention). To inject cold fuel, add the
            # (h_inject − h_t)·dṁ/(ṁ·dx) correction inside source_fn.
            dht_dx = dH_dx

            return [dM2_dx, dp_dx, dht_dx, dmdot_dx]

        # ---- events -------------------------------------------------------
        def choke_event(x, y):    return y[0] - 1.0
        choke_event.terminal = True; choke_event.direction = -1

        def pressure_event(x, y): return y[1] - 1.0
        pressure_event.terminal = True; pressure_event.direction = -1

        # ---- integrate ----------------------------------------------------
        y0 = [
            max(Ma2_in, 1.000001),
            max(p_in,   1.0),
            float(ht_in),
            max(mdot_in, 1e-9),
        ]

        sol = solve_ivp(
            fun=rhs,
            t_span=(x_start, x_end),
            y0=y0,
            method="DOP853",
            rtol=1e-6,
            atol=1e-6,
            max_step=(x_end - x_start) / 50,
            events=[choke_event, pressure_event],
            dense_output=False,
        )

        xs    = sol.t
        M2s   = np.maximum(sol.y[0], 1.000001)
        ps    = np.maximum(sol.y[1], 1.0)
        hts_arr = sol.y[2]
        mdots = np.maximum(sol.y[3], 1e-9)
        Mas   = np.sqrt(M2s)

        thermal_choke = len(sol.t_events[0]) > 0
        if thermal_choke:
            x_choke = sol.t_events[0][0]
            print(f"\n⚠ Thermal choking detected at x = {x_choke:.5f} m   (Ma → 1)")

        # ---- post-process: derive T, V, ... at each integration point ----
        # Reset T_cache so post-proc Newton starts fresh from T_in
        T_cache["T"] = float(T_in)
        Ts     = np.empty_like(xs)
        Vs     = np.empty_like(xs)
        cps    = np.empty_like(xs)
        gammas = np.empty_like(xs)
        Rs     = np.empty_like(xs)
        rhos   = np.empty_like(xs)
        for i in range(len(xs)):
            T_i, Y_i, cp_i, W_i, R_i, g_i, V2_i = solve_T(M2s[i], ps[i], hts_arr[i], xs[i])
            Ts[i]    = T_i
            Vs[i]    = np.sqrt(max(V2_i, 0.0))
            cps[i]   = cp_i
            gammas[i]= g_i
            Rs[i]    = R_i
            rhos[i]  = ps[i] / max(R_i * T_i, 1e-12)

        As = np.array([geometry_fn(x)[0] for x in xs])

        # ---- h, s, Tt, Pt via integral form / state_fn -------------------
        if state_fn is None:
            def state_fn_default(T, p, V, x):
                Y = composition_fn(x, T, p)
                return mix.stagnation_state(Y, T, p, V)
            state_fn = state_fn_default

        hs   = np.empty_like(xs)
        ss   = np.empty_like(xs)
        hts2 = np.empty_like(xs)
        sts2 = np.empty_like(xs)
        Tts  = np.empty_like(xs)
        Pts  = np.empty_like(xs)
        for i in range(len(xs)):
            st = state_fn(Ts[i], ps[i], Vs[i], xs[i])
            hs[i]   = st["h"]
            ss[i]   = st["s"]
            hts2[i] = st["ht"]
            sts2[i] = st["st"]
            Tts[i]  = st["Tt"]
            Pts[i]  = st["Pt"]

        return {
            "x": xs,
            "Ma": Mas, "Ma2": M2s,
            "p": ps,   "P":  ps,
            "T": Ts,
            "rho": rhos,
            "V": Vs,
            "Tt": Tts, "Pt": Pts, "pt": Pts,
            "h":  hs,  "s":  ss,
            "ht": hts2, "st": sts2,
            "A": As,
            "mdot": mdots,
            "thermal_choke": thermal_choke,
            "solver_success": sol.success,
            "solver_message": sol.message,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class Scramjet:
    L01 = 0.50
    L12 = 0.40
    L23 = 0.01
    L34 = 1.00
    L45 = 1.0
    alpha12 = 1.0
    alpha13 = 1.1
    alpha14 = 2.4
    alpha05 = 2.0

    EPSILON     = 0.4
    ETA_C       = 0.9
    CF_DEFAULT  = 0.003

    # Heating value used only for *informational* prints. The actual heat
    # release in sec4 is now computed self-consistently from CEA equilibrium
    # (h_react − h_eq), so this value is no longer the source of energy.
    Q_H2_HHV    = 141.8e6  # J/kg

    def __init__(self):
        self.air      = ScramAirProperties()
        self.mixture  = ScramMixtureNASA(self.air)
        self.shapiroODE = ScramShapiroODE()
        self._cea_comp = None

    def _get_cea(self):
        if self._cea_comp is None:
            self._cea_comp = ScramCEAComp()
        return self._cea_comp

    def _f(self, x):
        return float(np.asarray(x).squeeze())

    # ----- pure-air mass fractions (frozen) -------------------------------
    def _air_Y(self):
        moles = self.air.AIR_BASE_COMPOSITION
        total_mole = sum(moles.values())
        W_air = sum((moles[s]/total_mole) * self.air.MOLECULAR_WEIGHTS[s]
                    for s in moles)  # g/mol
        return {s: (moles[s]/total_mole) * self.air.MOLECULAR_WEIGHTS[s] / W_air
                for s in moles}

    # ----- generic state_fn factory for a fixed-composition section -------
    def _frozen_state_fn(self, Y_const):
        def state_fn(T, p, V, x):
            return self.mixture.stagnation_state(Y_const, T, p, V)
        return state_fn

    # =====================================================================
    # Section 0 — Freestream / inlet capture
    # =====================================================================
    def inlet_properties(self, h, Ma, m_air):
        T0   = ScramAtmosphere.T(h)
        P0   = ScramAtmosphere.P(h)
        rho0 = ScramAtmosphere.rho(h)

        Y_air = self._air_Y()

        cp0    = self.mixture.cp_mix(Y_air, T0)
        W_air_kgmol = self.mixture.W_mix(Y_air)
        R0     = self.mixture.R_UNIVERSAL / W_air_kgmol
        gamma0 = self.mixture.gamma_mix(Y_air, T0)

        a0 = np.sqrt(gamma0 * R0 * T0)
        V0 = Ma * a0
        A0 = m_air / (rho0 * V0)

        h0  = self.mixture.h_mix(Y_air, T0)
        s0  = self.mixture.s_mix(Y_air, T0, P0)
        ht0 = h0 + 0.5*V0**2
        Tt0 = self.mixture.stagnation_Tt(Y_air, T0, ht0)
        Pt0 = self.mixture.stagnation_Pt(Y_air, T0, Tt0, P0)

        print(f"\nInlet conditions at h={h:.0f} m, Ma={Ma:.2f}, m_air={m_air:.2f} kg/s:")
        print(f"  T0   = {T0:.2f} K")
        print(f"  P0   = {P0:.2f} Pa")
        print(f"  rho0 = {rho0:.4f} kg/m^3")
        print(f"  cp0  = {cp0:.2f} J/kg/K")
        print(f"  R0   = {R0:.2f} J/kg/K")
        print(f"  γ0   = {gamma0:.4f}")
        print(f"  V0   = {V0:.2f} m/s")
        print(f"  A0   = {A0:.4f} m^2")
        print(f"  Tt0  = {Tt0:.2f} K   (integral form)")
        print(f"  Pt0  = {Pt0:.2f} Pa  (integral form)")
        print(f"  h0   = {h0/1e6:.4f} MJ/kg")
        print(f"  ht0  = {ht0/1e6:.4f} MJ/kg")
        print(f"  s0   = {s0:.2f} J/kg/K")

        return {
            "Ma": Ma,    "Ma0": Ma,
            "T":  T0,    "T0":  T0,
            "P":  P0,    "P0":  P0,
            "rho": rho0, "rho0": rho0,
            "gamma": gamma0,
            "cp": cp0,
            "R": R0,
            "a": a0,
            "V":  V0,   "V0":  V0,
            "A":  A0,   "A0":  A0,
            "Tt": Tt0,  "Tt0": Tt0,
            "Pt": Pt0,  "Pt0": Pt0,
            "h":  h0,   "ht":  ht0,
            "s":  s0,   "st":  s0,
            "Y":  Y_air,
            "mdot": m_air,
        }

    # =====================================================================
    # Pressure recovery (Kantrowitz-style fit, paper)
    # =====================================================================
    def pressure_recovery(self, Ma):
        MaList = np.array([8.127, 7.641, 7.246, 6.866, 6.608, 6.349, 6.137,
                           5.954, 5.757, 5.605, 5.453, 5.286, 5.165, 5.028])
        sList = np.array([0.3022, 0.3183, 0.3339, 0.3505, 0.3634, 0.3774, 0.3887,
                          0.4000, 0.4124, 0.4231, 0.4339, 0.4452, 0.4543, 0.4661])
        return float(np.poly1d(np.polyfit(MaList, sList, 1))(Ma))

    # =====================================================================
    # Section 1 — Isolator (algebraic, total-enthalpy-conserving)
    # =====================================================================
    def isolator_properties(self, inlet_props):
        mix   = self.mixture
        Y_air = inlet_props["Y"]

        Ma0 = self._f(inlet_props["Ma"])
        T0  = self._f(inlet_props["T"])
        P0  = self._f(inlet_props["P"])
        V0  = self._f(inlet_props["V"])
        Pt0 = self._f(inlet_props["Pt"])
        mdot = self._f(inlet_props["mdot"])
        A0  = self._f(inlet_props["A"])
        rho0 = self._f(inlet_props["rho"])

        ht0 = self._f(inlet_props["ht"])
        s0  = self._f(inlet_props["s"])
        M1 = self.EPSILON * Ma0

        sigma_c    = self.pressure_recovery(Ma0)
        Pt1_target = sigma_c * Pt0

        # Energy: h(T1) + V1²/2 = ht0   with V1 = M1·sqrt(γ R T1)
        # Entropy step: s(T1, P1) = s0 + Δs_irrev  (we don't model the entropy
        # rise here; the target-Pt formula encodes it via σc.)
        def residual(vars_):
            T1g, p1g = vars_
            T1g = max(T1g, 250.0); p1g = max(p1g, 100.0)
            cp1    = mix.cp_mix(Y_air, T1g)
            W1     = mix.W_mix(Y_air)
            R1     = mix.R_UNIVERSAL / W1
            gamma1 = mix.gamma_mix(Y_air, T1g)
            V1     = M1 * np.sqrt(gamma1 * R1 * T1g)
            h1     = mix.h_mix(Y_air, T1g)
            eq1    = ht0 - (h1 + 0.5*V1**2)
            # Predict Pt at this state using the integral stagnation, compare to target
            Tt1g = mix.stagnation_Tt(Y_air, T1g, ht0)
            Pt1g = mix.stagnation_Pt(Y_air, T1g, Tt1g, p1g)
            eq2  = Pt1g - Pt1_target
            return [eq1, eq2]

        T1, P1 = fsolve(residual, x0=[1200.0, 0.2*P0])

        W1 = mix.W_mix(Y_air)
        R1 = mix.R_UNIVERSAL / W1
        gamma1 = mix.gamma_mix(Y_air, T1)
        cp1    = mix.cp_mix(Y_air, T1)
        V1     = M1 * np.sqrt(gamma1 * R1 * T1)

        h1   = mix.h_mix(Y_air, T1)
        s1   = mix.s_mix(Y_air, T1, P1)
        ht1  = h1 + 0.5*V1**2
        Tt1  = mix.stagnation_Tt(Y_air, T1, ht1)
        Pt1  = mix.stagnation_Pt(Y_air, T1, Tt1, P1)

        rho1 = P1 / (R1*T1)
        A1   = mdot / (rho1*V1)

        L_iso = getattr(self, "L01", 0.1)
        sol = {
            "x":    np.array([0.0, L_iso]),
            "Ma":   np.array([Ma0, M1]),
            "T":    np.array([T0, T1]),
            "Tt":   np.array([self._f(inlet_props["Tt"]), Tt1]),
            "p":    np.array([P0, P1]),  "P": np.array([P0, P1]),
            "pt":   np.array([Pt0, Pt1]),"Pt":np.array([Pt0, Pt1]),
            "A":    np.array([A0, A1]),
            "rho":  np.array([rho0, rho1]),
            "V":    np.array([V0, V1]),
            "mdot": np.array([mdot, mdot]),
            "h":    np.array([self._f(inlet_props["h"]), h1]),
            "s":    np.array([s0, s1]),
            "ht":   np.array([self._f(inlet_props["ht"]), ht1]),
            "st":   np.array([s0, s1]),
        }

        return {
            "Ma": M1,  "Ma1": M1,
            "T":  T1,  "T1":  T1,
            "P":  P1,  "p1":  P1, "P1": P1,
            "V":  V1,  "V1":  V1,
            "A":  A1,  "A1":  A1,
            "Tt": Tt1, "Tt1": Tt1,
            "Pt": Pt1, "Pt1": Pt1,
            "rho": rho1, "gamma": gamma1, "cp": cp1, "R": R1,
            "sigma_c": sigma_c, "mdot": mdot,
            "h":  h1,  "ht":  ht1, "s": s1,
            "Y":  Y_air,
            "solution": sol,
        }

    # =====================================================================
    # Section 1→2 — Constant-area / friction-only (no composition change)
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

        def composition_fn(x, T, p):
            return Y_air

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
            "Ma": self._f(result["Ma"][-1]), "Ma2": self._f(result["Ma"][-1]),
            "T":  self._f(T_end),            "T2":  self._f(T_end),
            "Tt": self._f(result["Tt"][-1]),
            "P":  self._f(p_end),            "p2":  self._f(p_end),
            "Pt": self._f(result["Pt"][-1]),
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
    # Section 2→3 — Fuel injection (mass addition only, no combustion)
    # Frozen mixing: H2 streams blend with air; composition evolves with x.
    # =====================================================================
    def combustor_properties3(self, sec2, phi=0.0, switches=None):
        mix    = self.mixture
        Y_air  = sec2["Y"]
        sw_mass = True if switches is None else switches.get("mass", True)
        W_h2  = self.air.MOLECULAR_WEIGHTS["H2"] * 1e-3
        W_air = mix.W_mix(Y_air)

        L_23 = self._f(self.L23)
        A2   = self._f(sec2["A"])
        A3   = self._f(self.alpha13) * A2 / self._f(self.alpha12)

        Ma2  = self._f(sec2["Ma"])
        T2   = self._f(sec2["T"])
        p2   = self._f(sec2["P"])
        mdot_air = self._f(sec2["mdot"])

        FAR_stoich  = self.optimal_fuel_air_ratio()
        FAR_actual  = phi * FAR_stoich
        mfuel_total = FAR_actual * mdot_air
        dmdot_dx_const = mfuel_total / L_23

        def Yf_at_mdot(mdot_local):
            return max((mdot_local - mdot_air) / max(mdot_local, 1e-30), 0.0)

        def Y_at_mdot(mdot_local):
            """Mass-fraction dict at this local mass flow (frozen mixing)."""
            Yf = Yf_at_mdot(mdot_local)
            Ya = 1.0 - Yf
            Y = {sp: Ya * Y_air[sp] for sp in Y_air}
            Y["H2"] = Y.get("H2", 0.0) + Yf
            return Y

        def Yf_at_x(x):
            """Closed-form ṁ(x) ⇒ Yf(x), used by post-processing state_fn."""
            mdot_local = mdot_air + dmdot_dx_const * x
            return Yf_at_mdot(mdot_local)

        def geometry_fn(x):
            A = A2 + (A3 - A2) * (x / L_23)
            dA_dx = (A3 - A2) / L_23
            D = np.sqrt(4 * A / np.pi)
            return A, dA_dx, D

        def composition_fn(x, T, p):
            # If `mass` is disabled, freeze composition at inlet air —
            # no fuel was ever injected, so Yf stays 0 throughout.
            if not sw_mass:
                return Y_air
            mdot_local = mdot_air + dmdot_dx_const * x
            return Y_at_mdot(mdot_local)

        def source_fn(x, T, p, mdot_local, Y):
            # No external heat; mass injection at flow's local stagnation
            # enthalpy is the Shapiro standard convention. The integrator
            # masks `dmdot_dx` to zero when `mass=False`.
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

        Y_exit = composition_fn(L_23, 0.0, 0.0)  # honours sw_mass

        # Report ACTUAL injected fuel — what the integrator's mass-flow ODE
        # produced. With mass=False the integrator zeroed dṁ/dx, so
        # mdot_exit == mdot_air ⇒ mfuel_actual = 0. With mass=True it
        # equals the scheduled mfuel_total. Downstream consumers (sec4,
        # performance) should use this physically realised value.
        mfuel_actual = max(self._f(result["mdot"][-1]) - mdot_air, 0.0)
        return {
            "Ma3": self._f(result["Ma"][-1]),
            "T3":  self._f(result["T"][-1]),
            "p3":  self._f(result["p"][-1]), "P3": self._f(result["p"][-1]),
            "rho3": self._f(result["rho"][-1]),
            "V3":  self._f(result["V"][-1]),
            "Tt3": self._f(result["Tt"][-1]),
            "Pt3": self._f(result["Pt"][-1]),
            "h3":  self._f(result["h"][-1]),
            "ht3": self._f(result["ht"][-1]),
            "s3":  self._f(result["s"][-1]),
            "A3":  A3,
            "mdot": self._f(result["mdot"][-1]),
            "mfuel":           mfuel_actual,    # honours sw_mass (0 if off)
            "mfuel_scheduled": mfuel_total,     # what φ asked for, pre-mask
            "phi": phi,
            "Y": Y_exit,
            "Yf_at_x_fn": Yf_at_x,
            "solution": result,
        }

    # =====================================================================
    # Section 3→4 — Combustion with per-step CEA equilibrium
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

        L_34 = self._f(self.L34)

        A3      = self._f(sec3["A3"])
        A1_ref  = A3 / self._f(self.alpha13)
        A4      = self._f(self.alpha14) * A1_ref

        Ma3   = self._f(sec3["Ma3"])
        T3    = self._f(sec3["T3"])
        p3    = self._f(sec3["p3"])
        mdot  = self._f(sec3["mdot"])
        mfuel = self._f(sec3["mfuel"])         # actual injected (= 0 if mass=False)

        # Reactant composition at sec4 inlet (= sec3 exit) — air + any injected H2.
        Y_react = dict(sec3["Y"])
        for sp in ScramCEAComp.PROD_NAMES:
            Y_react.setdefault(sp, 0.0)

        # O/F mass ratio is derived from the ACTUAL H₂ mass fraction in the
        # reactant stream, not the scheduled φ.  If sec3 ran with mass=False,
        # Y_react is pure air ⇒ Yf_react = 0 ⇒ of_ratio → ∞ ⇒ CEA returns
        # essentially air at every (T, p), h_eq ≈ h_react, and Q_eff ≈ 0.
        # That is, **no fuel ⇒ no combustion**, exactly as physics demands —
        # even if the `heat` switch itself is left ON.
        Yf_react = float(Y_react.get("H2", 0.0))
        of_ratio = (1.0 - Yf_react) / Yf_react if Yf_react > 1e-12 else 1e6

        theta = 0  # injection angle (0 = parallel ⇒ η = x/L linear)

        def mixing_efficiency(x):
            s = np.clip(x / L_34, 1e-4, 1.0)
            if theta == 0.0:
                return float(s)
            a = float(np.clip(1.01 + 0.176 * np.log(s), 0.0, 1.0))
            if theta == 90.0:
                return a
            return theta/90.0 * (a - s) + s

        def deta_dx(x):
            h_step = 1e-4
            return (mixing_efficiency(min(x + h_step, L_34))
                    - mixing_efficiency(max(x - h_step, 0.0))) / (2 * h_step)

        def Y_eq_at(T, p_pa):
            """Return equilibrium mass fractions, or Y_react if CEA fails (no chemistry)."""
            Yeq = cea_comp.equilibrium_Y(T, p_pa, of_ratio)
            return Yeq if Yeq is not None else Y_react

        def Y_blended(eta, T, p_pa):
            Yeq = Y_eq_at(T, p_pa)
            keys = set(Y_react) | set(Yeq)
            return {k: (1-eta) * Y_react.get(k, 0.0) + eta * Yeq.get(k, 0.0)
                    for k in keys}

        # ------- Shapiro callbacks (geometry / composition / source) ------
        def geometry_fn(x):
            A = A3 + (A4 - A3) * (x / L_34)
            dA_dx = (A4 - A3) / L_34
            D = np.sqrt(4 * A / np.pi)
            return A, dA_dx, D

        def composition_fn(x, T, p):
            # If `heat` is disabled, no combustion takes place — composition
            # stays at the reactant mixture (frozen H2 + air).
            if not sw_heat:
                return Y_react
            eta = mixing_efficiency(x)
            return Y_blended(eta, T, p)

        def source_fn(x, T, p, mdot_local, Y):
            # Heat released at this point: (h_react − h_eq)|_T × dη/dx.
            # Evaluating both at the same T gives the "chemistry energy"
            # liberated per unit fuel-mixing progress — automatically
            # incorporates dissociation losses at high T (h_eq rises).
            # The integrator masks dH_dx to zero if `heat=False`, so we
            # don't need to short-circuit here.
            h_react = mix.h_mix(Y_react, T)
            Yeq     = Y_eq_at(T, p)
            h_eq    = mix.h_mix(Yeq, T)
            dH_dx   = (h_react - h_eq) * deta_dx(x)
            return dH_dx, 0.0

        def state_fn(T, p, V, x):
            Y = composition_fn(x, T, p)   # honours sw_heat
            return mix.stagnation_state(Y, T, p, V)

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

        # Exit composition — used to freeze the nozzle.  Honours sw_heat.
        x_exit = result["x"][-1]
        T_exit, p_exit = result["T"][-1], result["p"][-1]
        Y_exit = composition_fn(x_exit, T_exit, p_exit)

        return {
            "Ma4": self._f(result["Ma"][-1]),
            "T4":  self._f(result["T"][-1]),
            "p4":  self._f(result["p"][-1]), "P4": self._f(result["p"][-1]),
            "rho4": self._f(result["rho"][-1]),
            "V4":  self._f(result["V"][-1]),
            "Tt4": self._f(result["Tt"][-1]),
            "Pt4": self._f(result["Pt"][-1]),
            "h4":  self._f(result["h"][-1]),
            "ht4": self._f(result["ht"][-1]),
            "s4":  self._f(result["s"][-1]),
            "A4":  A4,
            "mdot": mdot,
            "Y":   Y_exit,
            "solution": result,
            "thermal_choke": result["thermal_choke"],
        }

    # =====================================================================
    # Section 4→5 — Nozzle (frozen at sec4 exit composition)
    # =====================================================================
    def nozzle_properties(self, sec4, inlet_props, switches=None):
        if sec4["thermal_choke"]:
            return {"thermal_choke": True}

        mix    = self.mixture
        Y_nz   = sec4["Y"]
        W_nz   = mix.W_mix(Y_nz)

        L_45 = self._f(self.L45)
        A4   = self._f(sec4["A4"])
        A0   = self._f(inlet_props["A0"])
        A5   = self._f(self.alpha05) * A0
        Ma4  = self._f(sec4["Ma4"])
        T4   = self._f(sec4["T4"])
        p4   = self._f(sec4["p4"])
        mdot = self._f(sec4["mdot"])

        def geometry_fn(x):
            A = A4 + (A5 - A4) * (x / L_45)
            dA_dx = (A5 - A4) / L_45
            D = np.sqrt(4 * A / np.pi)
            return A, dA_dx, D

        def composition_fn(x, T, p): return Y_nz
        def source_fn(x, T, p, mdot_local, Y): return 0.0, 0.0

        state_fn = self._frozen_state_fn(Y_nz)

        result = self.shapiroODE.integrate(
            x_start=0.0, x_end=L_45,
            Ma2_in=Ma4**2, p_in=p4, T_in=T4, mdot_in=mdot,
            geometry_fn=geometry_fn,
            composition_fn=composition_fn,
            source_fn=source_fn,
            mix=mix,
            state_fn=state_fn,
            switches=switches,
            Cf=self.CF_DEFAULT, n_steps=200,
        )

        return {
            "Ma5": self._f(result["Ma"][-1]),
            "T5":  self._f(result["T"][-1]),
            "p5":  self._f(result["p"][-1]), "P5": self._f(result["p"][-1]),
            "rho5": self._f(result["rho"][-1]),
            "V5":  self._f(result["V"][-1]),
            "Tt5": self._f(result["Tt"][-1]),
            "Pt5": self._f(result["Pt"][-1]),
            "h5":  self._f(result["h"][-1]),
            "ht5": self._f(result["ht"][-1]),
            "s5":  self._f(result["s"][-1]),
            "A5":  A5,
            "mdot": mdot,
            "Y":   Y_nz,
            "solution": result,
            "thermal_choke": False,
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
        if sec5 is not None and not sec4.get("thermal_choke", False):
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
    eng = Scramjet()
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



def evaluate_engine(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, e


# ---------------------------------------------------------------------------
# 1) Altitude - Mach map sweep
# ---------------------------------------------------------------------------
def run_altitude_mach_map(eng):

    mach_range = np.arange(5.0, 10.5, 0.5)
    alt_range  = np.arange(25.0, 32.0, 1.0)

    ISP_map    = np.full((len(alt_range), len(mach_range)), np.nan)
    THRUST_map = np.full((len(alt_range), len(mach_range)), np.nan)

    for i, h in enumerate(alt_range):
        for j, M in enumerate(mach_range):

            perf, err = evaluate_engine(altitude_mach, eng, h_km=h, Ma0=M)

            if err is not None:
                print(f"FAILED at h={h:.1f}, M={M:.2f}")
                print(err)
                continue

            if perf.get("thermal_choke", False):
                ISP_map[i, j] = np.nan
                THRUST_map[i, j] = np.nan
            else:
                ISP_map[i, j]    = perf["Isp"]
                THRUST_map[i, j] = perf["Fin"]

            print(
                f"h={h:.1f} km | M={M:.2f} | "
                f"Isp={ISP_map[i,j]:.2f} s | "
                f"Fin={THRUST_map[i,j]:.2f} N"
            )

    # -----------------------------------------------------------------------
    # Build meshgrid for plotting
    # -----------------------------------------------------------------------
    M_grid, H_grid = np.meshgrid(mach_range, alt_range)

    # -----------------------------------------------------------------------
    # Build table (long format)
    # -----------------------------------------------------------------------
    rows = []
    for i, h in enumerate(alt_range):
        for j, M in enumerate(mach_range):
            rows.append({
                "Altitude_km": h,
                "Mach": M,
                "Isp_s": ISP_map[i, j],
                "Thrust_N": THRUST_map[i, j]
            })

    results_table = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Plot: Isp
    # -----------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    cont1 = plt.contourf(M_grid, H_grid, ISP_map, levels=40)
    plt.colorbar(cont1).set_label("Specific Impulse Isp [s]")
    plt.xlabel("Mach Number")
    plt.ylabel("Altitude [km]")
    plt.title("Scramjet Specific Impulse Map")
    plt.tight_layout()

    # -----------------------------------------------------------------------
    # Plot: Thrust
    # -----------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    cont2 = plt.contourf(M_grid, H_grid, THRUST_map, levels=40)
    plt.colorbar(cont2).set_label("Internal Thrust Fin [N]")
    plt.xlabel("Mach Number")
    plt.ylabel("Altitude [km]")
    plt.title("Scramjet Internal Thrust Map")
    plt.tight_layout()

    plt.show()

    return ISP_map, THRUST_map, results_table


# ---------------------------------------------------------------------------
# 2) Mass flow sweep
# ---------------------------------------------------------------------------
def run_mdot_sweep(eng, h_km=25.0, Ma0=5.0, phi=0.5):

    mdot_range = np.arange(1.0, 500.0, 10.0)

    ISP_list    = []
    THRUST_list = []

    for mdot in mdot_range:

        try:
            inp  = eng.inlet_properties(h=h_km*1e3, Ma=Ma0, m_air=mdot)
            iso  = eng.isolator_properties(inp)
            sec2 = eng.combustor_properties2(iso)
            sec3 = eng.combustor_properties3(sec2, phi=phi)
            sec4 = eng.combustor_properties4(sec3)

            if sec4["thermal_choke"]:
                ISP_list.append(np.nan)
                THRUST_list.append(np.nan)
                print(f"ṁ={mdot:.1f} kg/s -> THERMAL CHOKE")
                continue

            sec5 = eng.nozzle_properties(sec4, inp)
            perf = eng.performance(inp, sec5, sec3)

            ISP_list.append(perf["Isp"])
            THRUST_list.append(perf["Fin"])

            print(
                f"ṁ={mdot:.1f} kg/s | "
                f"Isp={perf['Isp']:.2f} s | "
                f"Fin={perf['Fin']:.2f} N"
            )

        except Exception as e:
            ISP_list.append(np.nan)
            THRUST_list.append(np.nan)

            print(f"FAILED at mdot={mdot:.1f}")
            print(e)

    # -----------------------------------------------------------------------
    # Plot Isp
    # -----------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(mdot_range, ISP_list)
    plt.xlabel("Air Mass Flow ṁ_air [kg/s]")
    plt.ylabel("Specific Impulse Isp [s]")
    plt.title("Isp vs Air Mass Flow")
    plt.grid(True)
    plt.tight_layout()

    # -----------------------------------------------------------------------
    # Plot Thrust
    # -----------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(mdot_range, THRUST_list)
    plt.xlabel("Air Mass Flow ṁ_air [kg/s]")
    plt.ylabel("Internal Thrust Fin [N]")
    plt.title("Thrust vs Air Mass Flow")
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    return np.array(ISP_list), np.array(THRUST_list)

# =============================================================================
# Scramjet single-point runner and Mach-5 ramjet-scramjet plot
# =============================================================================
#
# This section extends the original Mach-3 turbojet/ramjet plot file with:
#
#   - Full scramjet-cycle thrust at Mach 5
#   - Full ramjet-cycle thrust at Mach 5
#   - Both multiplied by engine count
#   - Same color-coded altitude-pair style as the Mach-3 plot
#
# =============================================================================

def run_scramjet_single_point(
    h_km: float,
    Ma0: float,
    mdot: float = 100.0,
    phi: float = 0.5,
    suppress_output: bool = True,
) -> dict[str, float | bool | str]:
    """
    Run the full scramjet model for one altitude and Mach number.

    Returns thrust for ONE scramjet engine.
    """
    import contextlib
    import io
    import warnings

    def _run():
        eng = Scramjet()

        inp = eng.inlet_properties(
            h=h_km * 1000.0,
            Ma=Ma0,
            m_air=mdot,
        )

        iso = eng.isolator_properties(inp)
        sec2 = eng.combustor_properties2(iso)
        sec3 = eng.combustor_properties3(sec2, phi=phi)
        sec4 = eng.combustor_properties4(sec3)
        sec5 = eng.nozzle_properties(sec4, inp)
        perf = eng.performance(inp, sec5, sec3)

        return eng, inp, iso, sec2, sec3, sec4, sec5, perf

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            if suppress_output:
                with contextlib.redirect_stdout(io.StringIO()):
                    eng, inp, iso, sec2, sec3, sec4, sec5, perf = _run()
            else:
                eng, inp, iso, sec2, sec3, sec4, sec5, perf = _run()

        thrust_N = float(perf.get("Fin", np.nan))

        if not np.isfinite(thrust_N):
            raise ValueError("Non-finite scramjet thrust result.")

        return {
            "success": True,
            "error": "",
            "Altitude_km": float(h_km),
            "Mach": float(Ma0),
            "Thrust_N": thrust_N,
            "Thrust_kN": thrust_N / 1000.0,
            "Isp_s": float(perf.get("Isp", np.nan)),
            "Ia_Ns_per_kg": float(perf.get("Ia", np.nan)),
            "mdot_air_kg_s": float(mdot),
            "mfuel_kg_s": float(perf.get("mfuel", np.nan)),
            "phi": float(phi),
            "Ma_exit": float(sec5.get("Ma5", np.nan)) if isinstance(sec5, dict) else np.nan,
            "T_exit_K": float(sec5.get("T5", np.nan)) if isinstance(sec5, dict) else np.nan,
            "p_exit_Pa": float(sec5.get("p5", np.nan)) if isinstance(sec5, dict) else np.nan,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "Altitude_km": float(h_km),
            "Mach": float(Ma0),
            "Thrust_N": np.nan,
            "Thrust_kN": np.nan,
            "Isp_s": np.nan,
            "Ia_Ns_per_kg": np.nan,
            "mdot_air_kg_s": float(mdot),
            "mfuel_kg_s": np.nan,
            "phi": float(phi),
            "Ma_exit": np.nan,
            "T_exit_K": np.nan,
            "p_exit_Pa": np.nan,
        }


def ramjet_cycle_thrust_at_mach_altitudes_for_mach5(
    mach_fixed: float,
    altitude_values_m: np.ndarray,
    mdot: float = 100.0,
    phi: float = 0.5,
    n_ramjet_engines: int = 2,
    suppress_output: bool = True,
) -> np.ndarray:
    """
    Full ramjet-cycle thrust at one fixed Mach number for many altitudes.

    The Ramjet model returns thrust for ONE engine, so this multiplies by
    n_ramjet_engines.

    Returned thrust is in kN.
    """
    thrust_total_kN = []

    print()
    print(f"Full ramjet-cycle thrust at Mach {mach_fixed:g}")
    print("-----------------------------------------")
    print(f"Ramjet engines included: {n_ramjet_engines}")
    print(f"{'Altitude [km]':>14s} {'One engine [kN]':>18s} {'Total [kN]':>16s} {'Status':>12s}")

    for h_m in altitude_values_m:
        h_km = h_m / 1000.0

        out = run_ramjet_single_point(
            h_km=h_km,
            Ma0=mach_fixed,
            mdot=mdot,
            phi=phi,
            suppress_output=suppress_output,
        )

        if out["success"]:
            one_engine_kN = float(out["Thrust_kN"])
            total_kN = n_ramjet_engines * one_engine_kN
            status = "OK"
        else:
            one_engine_kN = np.nan
            total_kN = np.nan
            status = "FAILED"

        thrust_total_kN.append(total_kN)

        print(f"{h_km:14.2f} {one_engine_kN:18.3f} {total_kN:16.3f} {status:>12s}")
        if not out["success"]:
            print(f"    Error: {out['error']}")

    return np.array(thrust_total_kN, dtype=float)


def scramjet_cycle_thrust_at_mach_altitudes(
    mach_fixed: float,
    altitude_values_m: np.ndarray,
    mdot: float = 100.0,
    phi: float = 0.5,
    n_scramjet_engines: int = 2,
    suppress_output: bool = True,
) -> np.ndarray:
    """
    Full scramjet-cycle thrust at one fixed Mach number for many altitudes.

    The Scramjet model returns thrust for ONE engine, so this multiplies by
    n_scramjet_engines.

    Returned thrust is in kN.
    """
    thrust_total_kN = []

    print()
    print(f"Full scramjet-cycle thrust at Mach {mach_fixed:g}")
    print("------------------------------------------")
    print(f"Scramjet engines included: {n_scramjet_engines}")
    print(f"{'Altitude [km]':>14s} {'One engine [kN]':>18s} {'Total [kN]':>16s} {'Status':>12s}")

    for h_m in altitude_values_m:
        h_km = h_m / 1000.0

        out = run_scramjet_single_point(
            h_km=h_km,
            Ma0=mach_fixed,
            mdot=mdot,
            phi=phi,
            suppress_output=suppress_output,
        )

        if out["success"]:
            one_engine_kN = float(out["Thrust_kN"])
            total_kN = n_scramjet_engines * one_engine_kN
            status = "OK"
        else:
            one_engine_kN = np.nan
            total_kN = np.nan
            status = "FAILED"

        thrust_total_kN.append(total_kN)

        print(f"{h_km:14.2f} {one_engine_kN:18.3f} {total_kN:16.3f} {status:>12s}")
        if not out["success"]:
            print(f"    Error: {out['error']}")

    return np.array(thrust_total_kN, dtype=float)


def plot_ramjet_scramjet_cycle_mach_line(
    altitude_values_m: np.ndarray | None = None,
    mach_fixed: float = 5.0,
    mdot_ramjet: float = 100.0,
    mdot_scramjet: float = 100.0,
    phi_ramjet: float = 0.5,
    phi_scramjet: float = 0.5,
    n_ramjet_engines: int = 2,
    n_scramjet_engines: int = 2,
    suppress_output: bool = True,
) -> dict[str, np.ndarray]:
    """
    Same style as the Mach-3 plot, but for ramjet-to-scramjet transition
    at Mach 5.

    Color coding:
        - Each altitude gets one color.
        - Ramjet and scramjet at the same altitude use the same color.
        - Ramjet point = circle marker.
        - Scramjet point = square marker.
        - Same-altitude points are connected with a thick colored line.

    Both ramjet and scramjet models return thrust per engine, so both are
    multiplied by their engine counts.
    """
    if altitude_values_m is None:
        altitude_values_m = np.linspace(22_000.0, 32_000.0, 11)

    altitude_values_m = np.asarray(altitude_values_m, dtype=float)

    ramjet_total_kN = ramjet_cycle_thrust_at_mach_altitudes_for_mach5(
        mach_fixed=mach_fixed,
        altitude_values_m=altitude_values_m,
        mdot=mdot_ramjet,
        phi=phi_ramjet,
        n_ramjet_engines=n_ramjet_engines,
        suppress_output=suppress_output,
    )

    scramjet_total_kN = scramjet_cycle_thrust_at_mach_altitudes(
        mach_fixed=mach_fixed,
        altitude_values_m=altitude_values_m,
        mdot=mdot_scramjet,
        phi=phi_scramjet,
        n_scramjet_engines=n_scramjet_engines,
        suppress_output=suppress_output,
    )

    print()
    print(f"Full ramjet-cycle vs full scramjet-cycle thrust at Mach {mach_fixed:g}")
    print("---------------------------------------------------------------------")
    print(f"Ramjet engines: {n_ramjet_engines}, Scramjet engines: {n_scramjet_engines}")
    print(f"{'Altitude [km]':>14s} {'Ramjet total':>16s} {'Scram total':>16s} {'Ram - Scram':>16s}")

    for h, T_ram, T_scram in zip(altitude_values_m, ramjet_total_kN, scramjet_total_kN):
        diff = T_ram - T_scram if np.isfinite(T_ram) and np.isfinite(T_scram) else np.nan
        print(f"{h / 1000.0:14.2f} {T_ram:16.3f} {T_scram:16.3f} {diff:16.3f}")

    fig, ax = plt.subplots(figsize=(13, 8))

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(altitude_values_m) - 1, 1)) for i in range(len(altitude_values_m))]
    offsets = np.linspace(-0.055, 0.055, len(altitude_values_m))

    for i, (h, offset, T_ram, T_scram) in enumerate(
        zip(altitude_values_m, offsets, ramjet_total_kN, scramjet_total_kN)
    ):
        color = colors[i]
        x_plot = mach_fixed + offset
        h_km = h / 1000.0

        if np.isfinite(T_ram):
            ax.scatter(
                x_plot,
                T_ram,
                marker="o",
                s=95,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=12,
            )

        if np.isfinite(T_scram):
            ax.scatter(
                x_plot,
                T_scram,
                marker="s",
                s=95,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=12,
            )

        if np.isfinite(T_ram) and np.isfinite(T_scram):
            ax.plot(
                [x_plot, x_plot],
                [T_ram, T_scram],
                color=color,
                linewidth=3.0,
                alpha=0.95,
                zorder=11,
            )
            T_mid = 0.5 * (T_ram + T_scram)
            label_text = f"{h_km:.0f} km"
        elif np.isfinite(T_ram):
            T_mid = T_ram
            label_text = f"{h_km:.0f} km\nscram fail"
        elif np.isfinite(T_scram):
            T_mid = T_scram
            label_text = f"{h_km:.0f} km\nram fail"
        else:
            continue

        ax.text(
            x_plot + 0.035,
            T_mid,
            label_text,
            va="center",
            ha="left",
            fontsize=9,
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                alpha=0.9,
                pad=1.6,
            ),
            zorder=13,
        )

    ax.axvline(
        mach_fixed,
        color="black",
        linestyle=":",
        linewidth=2.5,
        zorder=8,
    )

    ax.set_xlim(mach_fixed - 0.35, mach_fixed + 0.65)

    finite_values = np.concatenate([
        ramjet_total_kN[np.isfinite(ramjet_total_kN)],
        scramjet_total_kN[np.isfinite(scramjet_total_kN)],
    ])

    if finite_values.size > 0:
        ymin = np.nanmin(finite_values)
        ymax = np.nanmax(finite_values)
        margin = 0.12 * max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - margin, ymax + margin)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor="gray",
               markeredgecolor="black", markersize=9,
               label=f"ramjet total, {n_ramjet_engines} engines, at M={mach_fixed:g}"),
        Line2D([0], [0], marker="s", color="white", markerfacecolor="gray",
               markeredgecolor="black", markersize=9,
               label=f"scramjet total, {n_scramjet_engines} engines, at M={mach_fixed:g}"),
        Line2D([0], [0], color="gray", linewidth=3,
               label="same-altitude connector"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2.5,
               label=f"Mach {mach_fixed:g}"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
    )

    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("Thrust [kN]")
    ax.set_title(
        f"Ramjet-to-scramjet fixed-Mach comparison at Mach {mach_fixed:g}"
    )
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    plt.show()

    return {
        "altitude_m": altitude_values_m,
        "altitude_km": altitude_values_m / 1000.0,
        "mach_fixed": np.full_like(altitude_values_m, mach_fixed, dtype=float),
        "ramjet_total_thrust_kN": ramjet_total_kN,
        "scramjet_total_thrust_kN": scramjet_total_kN,
        "difference_ramjet_minus_scramjet_kN": ramjet_total_kN - scramjet_total_kN,
    }


# =============================================================================
# Run both transition plots
# =============================================================================

# =============================================================================
# Drag model with variable alpha / AOA
# =============================================================================
#
# This is based on the drag code you uploaded:
#   - ISA atmosphere
#   - C_L(alpha, Mach)
#   - C_D(Mach, C_L)
#   - D = q S_ref C_D
#
# Drag is calculated in kN so it can be plotted directly with thrust.
# =============================================================================

def isa_temperature_drag(altitude_m: float) -> float:
    if altitude_m <= 11_000.0:
        return 288.15 - 0.0065 * altitude_m
    elif altitude_m <= 20_000.0:
        return 216.65
    elif altitude_m <= 32_000.0:
        return 216.65 + 0.001 * (altitude_m - 20_000.0)
    else:
        return 228.65


def isa_pressure_drag(altitude_m: float) -> float:
    g0 = 9.80665
    R = 287.05

    if altitude_m <= 11_000.0:
        T0 = 288.15
        L = -0.0065
        T = T0 + L * altitude_m
        return 101325.0 * (T / T0) ** (-g0 / (L * R))
    elif altitude_m <= 20_000.0:
        T = 216.65
        p11 = 22632.06
        return p11 * np.exp(-g0 * (altitude_m - 11_000.0) / (R * T))
    elif altitude_m <= 32_000.0:
        T20 = 216.65
        p20 = 5474.89
        L = 0.001
        T = T20 + L * (altitude_m - 20_000.0)
        return p20 * (T / T20) ** (-g0 / (L * R))
    else:
        T = 228.65
        p32 = 868.02
        return p32 * np.exp(-g0 * (altitude_m - 32_000.0) / (R * T))


def isa_density_drag(altitude_m: float) -> float:
    R = 287.05
    return isa_pressure_drag(altitude_m) / (R * isa_temperature_drag(altitude_m))


def speed_of_sound_drag(altitude_m: float) -> float:
    gamma_air = 1.4
    R = 287.05
    return np.sqrt(gamma_air * R * isa_temperature_drag(altitude_m))


def dynamic_pressure_from_mach_altitude_drag(M: float, altitude_m: float) -> float:
    rho = isa_density_drag(altitude_m)
    V = M * speed_of_sound_drag(altitude_m)
    return 0.5 * rho * V**2


# C_D = a(M) C_L^2 + b(M) C_L + c(M)
MACH_POLAR_DATA_DRAG = np.array([0.65, 0.9, 1.1, 1.3, 2.0, 5.37, 7.38, 10.61])

A_POLAR_DATA_DRAG = np.array([0.3804, 0.3418, 0.3459, 0.4006, 0.6049, 1.0314, 1.2753, 1.1948])
B_POLAR_DATA_DRAG = np.array([-0.0011, 0.0100, 0.0012, 0.0037, 0.0010, 0.0145, 0.0354, 0.0962])
C_POLAR_DATA_DRAG = np.array([0.0070, 0.0174, 0.0382, 0.0337, 0.0268, 0.0121, 0.0101, 0.0081])

A_POLAR_INTERP_DRAG = PchipInterpolator(MACH_POLAR_DATA_DRAG, A_POLAR_DATA_DRAG)
B_POLAR_INTERP_DRAG = PchipInterpolator(MACH_POLAR_DATA_DRAG, B_POLAR_DATA_DRAG)
C_POLAR_INTERP_DRAG = PchipInterpolator(MACH_POLAR_DATA_DRAG, C_POLAR_DATA_DRAG)


# C_L = m(M) alpha_deg + k(M)
CL_ALPHA_SLOPE_DATA_DRAG = np.array([
    0.0430,
    0.0457,
    0.0428,
    0.0372,
    0.0271,
    0.0167,
    0.0128,
    0.0110,
])

CL_ALPHA_INTERCEPT_DATA_DRAG = np.array([
    -0.0347,
    -0.0381,
    -0.0235,
    -0.0084,
    0.0011,
    -0.0032,
    -0.0030,
    -0.0048,
])

CL_ALPHA_SLOPE_INTERP_DRAG = PchipInterpolator(MACH_POLAR_DATA_DRAG, CL_ALPHA_SLOPE_DATA_DRAG)
CL_ALPHA_INTERCEPT_INTERP_DRAG = PchipInterpolator(MACH_POLAR_DATA_DRAG, CL_ALPHA_INTERCEPT_DATA_DRAG)


def cl_from_mach_alpha_drag(M: float, alpha_deg: float, clamp_mach: bool = True) -> float:
    M_original = float(M)

    if clamp_mach:
        M_used = float(np.clip(M_original, MACH_POLAR_DATA_DRAG.min(), MACH_POLAR_DATA_DRAG.max()))
    else:
        if M_original < MACH_POLAR_DATA_DRAG.min() or M_original > MACH_POLAR_DATA_DRAG.max():
            raise ValueError("Mach outside available C_L-alpha range.")
        M_used = M_original

    slope = float(CL_ALPHA_SLOPE_INTERP_DRAG(M_used))
    intercept = float(CL_ALPHA_INTERCEPT_INTERP_DRAG(M_used))
    return slope * alpha_deg + intercept


def cd_from_mach_cl_drag(M: float, CL: float, clamp_mach: bool = True) -> float:
    M_original = float(M)

    if clamp_mach:
        M_used = float(np.clip(M_original, MACH_POLAR_DATA_DRAG.min(), MACH_POLAR_DATA_DRAG.max()))
    else:
        if M_original < MACH_POLAR_DATA_DRAG.min() or M_original > MACH_POLAR_DATA_DRAG.max():
            raise ValueError("Mach outside available C_D polar range.")
        M_used = M_original

    a = float(A_POLAR_INTERP_DRAG(M_used))
    b = float(B_POLAR_INTERP_DRAG(M_used))
    c = float(C_POLAR_INTERP_DRAG(M_used))
    return a * CL**2 + b * CL + c


def drag_kN_from_mach_alpha_altitude(
    M: float,
    alpha_deg: float,
    altitude_m: float,
    S_ref: float,
    clamp_mach: bool = True,
) -> float:
    """
    Drag in kN at one Mach, altitude, alpha, and reference area.
    """
    CL = cl_from_mach_alpha_drag(M, alpha_deg, clamp_mach=clamp_mach)
    CD = cd_from_mach_cl_drag(M, CL, clamp_mach=clamp_mach)
    q = dynamic_pressure_from_mach_altitude_drag(M, altitude_m)
    D_N = q * S_ref * CD
    return D_N / 1000.0


def drag_kN_for_alpha_altitude_grid(
    mach_fixed: float,
    altitude_values_m: np.ndarray,
    alpha_values_deg: list[float] | tuple[float, ...] | np.ndarray,
    S_ref: float,
) -> dict[float, np.ndarray]:
    """
    Returns drag curves at fixed Mach for multiple alpha values.

    Output:
        {alpha_deg: drag_kN_array_over_altitude}
    """
    alpha_values_deg = list(alpha_values_deg)
    drag_by_alpha = {}

    for alpha in alpha_values_deg:
        drag_by_alpha[float(alpha)] = np.array([
            drag_kN_from_mach_alpha_altitude(
                M=mach_fixed,
                alpha_deg=float(alpha),
                altitude_m=float(h),
                S_ref=S_ref,
                clamp_mach=True,
            )
            for h in altitude_values_m
        ])

    return drag_by_alpha


def _add_drag_to_fixed_mach_plot(
    ax,
    mach_fixed: float,
    altitude_values_m: np.ndarray,
    alpha_values_deg: list[float] | tuple[float, ...] | np.ndarray,
    S_ref: float,
    x_start_offset: float = 0.18,
    x_spacing: float = 0.055,
) -> dict[float, np.ndarray]:
    """
    Add variable-alpha drag points/lines to an existing fixed-Mach plot.

    Drag is offset to the right of the Mach line so it does not overlap with
    the engine thrust pair markers.
    """
    drag_by_alpha = drag_kN_for_alpha_altitude_grid(
        mach_fixed=mach_fixed,
        altitude_values_m=altitude_values_m,
        alpha_values_deg=alpha_values_deg,
        S_ref=S_ref,
    )

    alpha_values_deg = list(alpha_values_deg)

    # Use a separate marker style for drag.
    # Each alpha becomes one line through altitude-varying drag points.
    for j, alpha in enumerate(alpha_values_deg):
        drag_values = drag_by_alpha[float(alpha)]
        x_drag = np.full_like(altitude_values_m, mach_fixed + x_start_offset + j * x_spacing, dtype=float)

        ax.plot(
            x_drag,
            drag_values,
            marker="^",
            linestyle="-.",
            linewidth=2.0,
            markersize=7,
            alpha=0.9,
            label=fr"drag, $\alpha$={alpha:g}°",
            zorder=9,
        )

        # Label only a few points so the plot does not become unreadable.
        for idx in [0, len(altitude_values_m)//2, len(altitude_values_m)-1]:
            h_km = altitude_values_m[idx] / 1000.0
            ax.text(
                x_drag[idx] + 0.015,
                drag_values[idx],
                f"{h_km:.0f} km",
                fontsize=7,
                va="center",
                ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.0),
                zorder=10,
            )

    return drag_by_alpha


# =============================================================================
# Mach-3 plot with drag included
# =============================================================================

def plot_turbo_ramjet_mach3_with_drag(
    altitude_values_m: np.ndarray | None = None,
    mach_fixed: float = 3.0,
    mach_min: float = 0.0,
    mach_max: float = 6.0,
    n_mach: int = 800,
    mdot: float = 100.0,
    phi: float = 0.5,
    n_ramjet_engines: int = 2,
    suppress_output: bool = True,
    clamp_negative_turbo_thrust: bool = True,
    alpha_values_deg: list[float] | tuple[float, ...] = (0.0, 3.5, 7.5),
    S_ref: float = 400.0,
) -> dict[str, np.ndarray | dict[float, np.ndarray]]:
    """
    Mach-3 turbojet/ramjet transition plot with drag for variable alpha.

    Thrust:
        - turbojet polynomial thrust
        - full ramjet-cycle thrust, multiplied by n_ramjet_engines

    Drag:
        - D = q S_ref C_D(M, C_L(alpha))
        - plotted for each alpha in alpha_values_deg
    """
    if altitude_values_m is None:
        altitude_values_m = np.linspace(12_000.0, 22_000.0, 11)

    altitude_values_m = np.asarray(altitude_values_m, dtype=float)
    M_values = np.linspace(mach_min, mach_max, n_mach)

    turbo_at_mach = turbo_thrust_at_mach_altitudes(
        mach_fixed=mach_fixed,
        altitude_values_m=altitude_values_m,
        clamp_negative_thrust=clamp_negative_turbo_thrust,
    )

    ramjet_cycle_at_mach = ramjet_cycle_thrust_at_mach_altitudes(
        mach_fixed=mach_fixed,
        altitude_values_m=altitude_values_m,
        mdot=mdot,
        phi=phi,
        n_ramjet_engines=n_ramjet_engines,
        suppress_output=suppress_output,
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(altitude_values_m) - 1, 1)) for i in range(len(altitude_values_m))]

    # Background turbojet thrust curves.
    for i, h in enumerate(altitude_values_m):
        T_turbo_curve = turbo_thrust_curve_vs_mach(
            altitude_m=h,
            mach_values=M_values,
            clamp_negative_thrust=clamp_negative_turbo_thrust,
        )

        ax.plot(
            M_values,
            T_turbo_curve,
            linestyle="-",
            linewidth=1.5,
            alpha=0.45,
            color=colors[i],
        )

    # Mach line
    ax.axvline(
        mach_fixed,
        color="black",
        linestyle=":",
        linewidth=2.5,
        zorder=8,
    )

    # Thrust pairs at Mach fixed.
    offsets = np.linspace(-0.055, 0.055, len(altitude_values_m))

    for i, (h, offset, T_turbo, T_ramjet) in enumerate(
        zip(altitude_values_m, offsets, turbo_at_mach, ramjet_cycle_at_mach)
    ):
        color = colors[i]
        x_plot = mach_fixed + offset
        h_km = h / 1000.0

        ax.scatter(
            x_plot,
            T_turbo,
            marker="o",
            s=90,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=12,
        )

        if np.isfinite(T_ramjet):
            ax.scatter(
                x_plot,
                T_ramjet,
                marker="s",
                s=90,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=12,
            )

            ax.plot(
                [x_plot, x_plot],
                [T_turbo, T_ramjet],
                color=color,
                linewidth=3.0,
                alpha=0.95,
                zorder=11,
            )

            T_mid = 0.5 * (T_turbo + T_ramjet)
            label_text = f"{h_km:.0f} km"
        else:
            T_mid = T_turbo
            label_text = f"{h_km:.0f} km\nram fail"

        ax.text(
            x_plot + 0.035,
            T_mid,
            label_text,
            va="center",
            ha="left",
            fontsize=8,
            color="black",
            bbox=dict(facecolor="white", edgecolor=color, linewidth=1.2, alpha=0.9, pad=1.4),
            zorder=13,
        )

    # Add drag for variable alpha.
    drag_by_alpha = _add_drag_to_fixed_mach_plot(
        ax=ax,
        mach_fixed=mach_fixed,
        altitude_values_m=altitude_values_m,
        alpha_values_deg=alpha_values_deg,
        S_ref=S_ref,
        x_start_offset=0.22,
        x_spacing=0.065,
    )

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.8,
               label="turbo polynomial curve"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="gray",
               markeredgecolor="black", markersize=9,
               label=f"turbo at M={mach_fixed:g}"),
        Line2D([0], [0], marker="s", color="white", markerfacecolor="gray",
               markeredgecolor="black", markersize=9,
               label=f"ramjet total, {n_ramjet_engines} engines"),
        Line2D([0], [0], marker="^", color="gray", markerfacecolor="gray",
               markeredgecolor="black", markersize=9,
               linestyle="-.", label=f"drag points, S={S_ref:.0f} m²"),
        Line2D([0], [0], color="gray", linewidth=3,
               label="same-altitude thrust connector"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2.5,
               label=f"Mach {mach_fixed:g}"),
    ]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=legend_elements + handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
    )

    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("Force [kN]")
    ax.set_title(
        f"Mach {mach_fixed:g}: Turbojet/Ramjet thrust with drag for variable AOA"
    )
    ax.grid(True, alpha=0.35)
    ax.set_xlim(mach_min, mach_fixed + 0.22 + 0.065 * len(alpha_values_deg) + 0.25)

    fig.tight_layout()
    plt.show()

    return {
        "altitude_m": altitude_values_m,
        "altitude_km": altitude_values_m / 1000.0,
        "mach_fixed": np.full_like(altitude_values_m, mach_fixed, dtype=float),
        "turbo_polynomial_thrust_kN": turbo_at_mach,
        "ramjet_cycle_total_thrust_kN": ramjet_cycle_at_mach,
        "drag_by_alpha_kN": drag_by_alpha,
        "alpha_values_deg": np.array(alpha_values_deg, dtype=float),
        "S_ref": np.array([S_ref]),
    }


# =============================================================================
# Mach-5 plot with drag included
# =============================================================================

def plot_ramjet_scramjet_mach5_with_drag(
    altitude_values_m: np.ndarray | None = None,
    mach_fixed: float = 5.0,
    mdot_ramjet: float = 100.0,
    mdot_scramjet: float = 100.0,
    phi_ramjet: float = 0.5,
    phi_scramjet: float = 0.5,
    n_ramjet_engines: int = 2,
    n_scramjet_engines: int = 2,
    suppress_output: bool = True,
    alpha_values_deg: list[float] | tuple[float, ...] = (0.0, 3.5, 7.5),
    S_ref: float = 400.0,
) -> dict[str, np.ndarray | dict[float, np.ndarray]]:
    """
    Mach-5 ramjet/scramjet transition plot with drag for variable alpha.

    Thrust:
        - full ramjet cycle thrust, multiplied by n_ramjet_engines
        - full scramjet cycle thrust, multiplied by n_scramjet_engines

    Drag:
        - D = q S_ref C_D(M, C_L(alpha))
        - plotted for each alpha in alpha_values_deg
    """
    if altitude_values_m is None:
        altitude_values_m = np.linspace(22_000.0, 32_000.0, 11)

    altitude_values_m = np.asarray(altitude_values_m, dtype=float)

    ramjet_total_kN = ramjet_cycle_thrust_at_mach_altitudes_for_mach5(
        mach_fixed=mach_fixed,
        altitude_values_m=altitude_values_m,
        mdot=mdot_ramjet,
        phi=phi_ramjet,
        n_ramjet_engines=n_ramjet_engines,
        suppress_output=suppress_output,
    )

    scramjet_total_kN = scramjet_cycle_thrust_at_mach_altitudes(
        mach_fixed=mach_fixed,
        altitude_values_m=altitude_values_m,
        mdot=mdot_scramjet,
        phi=phi_scramjet,
        n_scramjet_engines=n_scramjet_engines,
        suppress_output=suppress_output,
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(altitude_values_m) - 1, 1)) for i in range(len(altitude_values_m))]
    offsets = np.linspace(-0.055, 0.055, len(altitude_values_m))

    for i, (h, offset, T_ram, T_scram) in enumerate(
        zip(altitude_values_m, offsets, ramjet_total_kN, scramjet_total_kN)
    ):
        color = colors[i]
        x_plot = mach_fixed + offset
        h_km = h / 1000.0

        if np.isfinite(T_ram):
            ax.scatter(
                x_plot,
                T_ram,
                marker="o",
                s=95,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=12,
            )

        if np.isfinite(T_scram):
            ax.scatter(
                x_plot,
                T_scram,
                marker="s",
                s=95,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=12,
            )

        if np.isfinite(T_ram) and np.isfinite(T_scram):
            ax.plot(
                [x_plot, x_plot],
                [T_ram, T_scram],
                color=color,
                linewidth=3.0,
                alpha=0.95,
                zorder=11,
            )
            T_mid = 0.5 * (T_ram + T_scram)
            label_text = f"{h_km:.0f} km"
        elif np.isfinite(T_ram):
            T_mid = T_ram
            label_text = f"{h_km:.0f} km\nscram fail"
        elif np.isfinite(T_scram):
            T_mid = T_scram
            label_text = f"{h_km:.0f} km\nram fail"
        else:
            continue

        ax.text(
            x_plot + 0.035,
            T_mid,
            label_text,
            va="center",
            ha="left",
            fontsize=8,
            color="black",
            bbox=dict(facecolor="white", edgecolor=color, linewidth=1.2, alpha=0.9, pad=1.4),
            zorder=13,
        )

    ax.axvline(
        mach_fixed,
        color="black",
        linestyle=":",
        linewidth=2.5,
        zorder=8,
    )

    # Add drag for variable alpha.
    drag_by_alpha = _add_drag_to_fixed_mach_plot(
        ax=ax,
        mach_fixed=mach_fixed,
        altitude_values_m=altitude_values_m,
        alpha_values_deg=alpha_values_deg,
        S_ref=S_ref,
        x_start_offset=0.22,
        x_spacing=0.065,
    )

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor="gray",
               markeredgecolor="black", markersize=9,
               label=f"ramjet total, {n_ramjet_engines} engines"),
        Line2D([0], [0], marker="s", color="white", markerfacecolor="gray",
               markeredgecolor="black", markersize=9,
               label=f"scramjet total, {n_scramjet_engines} engines"),
        Line2D([0], [0], marker="^", color="gray", markerfacecolor="gray",
               markeredgecolor="black", markersize=9,
               linestyle="-.", label=f"drag points, S={S_ref:.0f} m²"),
        Line2D([0], [0], color="gray", linewidth=3,
               label="same-altitude thrust connector"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2.5,
               label=f"Mach {mach_fixed:g}"),
    ]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=legend_elements + handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
    )

    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("Force [kN]")
    ax.set_title(
        f"Mach {mach_fixed:g}: Ramjet/Scramjet thrust with drag for variable AOA"
    )
    ax.grid(True, alpha=0.35)
    ax.set_xlim(mach_fixed - 0.35, mach_fixed + 0.22 + 0.065 * len(alpha_values_deg) + 0.25)

    fig.tight_layout()
    plt.show()

    return {
        "altitude_m": altitude_values_m,
        "altitude_km": altitude_values_m / 1000.0,
        "mach_fixed": np.full_like(altitude_values_m, mach_fixed, dtype=float),
        "ramjet_total_thrust_kN": ramjet_total_kN,
        "scramjet_total_thrust_kN": scramjet_total_kN,
        "drag_by_alpha_kN": drag_by_alpha,
        "alpha_values_deg": np.array(alpha_values_deg, dtype=float),
        "S_ref": np.array([S_ref]),
    }


# =============================================================================
# Run both transition plots with drag
# =============================================================================

if __name__ == "__main__":

    # Change these to test different AOA schedules.
    alpha_values_deg = [0.0, 3.5, 7.5]
    S_ref = 400.0

    # -------------------------------------------------------------------------
    # Plot 1: Mach 3 turbojet/ramjet + drag
    # -------------------------------------------------------------------------
    altitude_values_m_mach3 = np.linspace(12_000.0, 22_000.0, 11)

    results_mach3 = plot_turbo_ramjet_mach3_with_drag(
        altitude_values_m=altitude_values_m_mach3,
        mach_fixed=3.0,
        mach_min=0.0,
        mach_max=6.0,
        n_mach=800,
        mdot=100.0,
        phi=0.5,
        n_ramjet_engines=2,
        suppress_output=True,
        clamp_negative_turbo_thrust=True,
        alpha_values_deg=alpha_values_deg,
        S_ref=S_ref,
    )

    # -------------------------------------------------------------------------
    # Plot 2: Mach 5 ramjet/scramjet + drag
    # -------------------------------------------------------------------------
    altitude_values_m_mach5 = np.linspace(22_000.0, 32_000.0, 11)

    results_mach5 = plot_ramjet_scramjet_mach5_with_drag(
        altitude_values_m=altitude_values_m_mach5,
        mach_fixed=5.0,
        mdot_ramjet=100.0,
        mdot_scramjet=100.0,
        phi_ramjet=0.5,
        phi_scramjet=0.5,
        n_ramjet_engines=2,
        n_scramjet_engines=2,
        suppress_output=True,
        alpha_values_deg=alpha_values_deg,
        S_ref=S_ref,
    )
