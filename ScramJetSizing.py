from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


class AirProperties:
    R_UNIVERSAL = 8.314462  # J/(mol·K)

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
            "low": [2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.37967491],
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
            "low": [2.5, 0.0, 0.0, 0.0, 0.0, 56104.6378, 4.19390932],
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
        dg1 = 2*self.gibbs_over_RT("N",T,1) - self.gibbs_over_RT("N2",T,1)
        dg2 = 2*self.gibbs_over_RT("O",T,1) - self.gibbs_over_RT("O2",T,1)
        dg3 = (self.gibbs_over_RT("NO",T,1)
               - self.gibbs_over_RT("N",T,1) - self.gibbs_over_RT("O",T,1))
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
        """Specific gas constant R [J/kg/K] for the equilibrium mixture."""
        cp, cv, _ = self.mixture_cp_cv(T, P/101325)
        return cp - cv


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


class Engine:
    L12 = 0.40; L23 = 0.01; L34 = 1.00; L45 = 0.40
    alpha12 = 1.0; alpha13 = 1.1; alpha14 = 2.5; alpha05 = 2.0
    EPSILON = 0.4
    ETA_C   = 0.9
    CF_DEFAULT = 0.003

    def __init__(self):
        self.air = AirProperties()

    # ------------------------------------------------------------------
    def inlet_properties(self, h: float, Ma: float, m_air: float) -> dict:
        T0   = Atmosphere.T(h)
        P0   = Atmosphere.P(h)
        rho0 = Atmosphere.rho(h)

        gamma0 = self.air.specific_heat_ratio(T0, P0)
        cp0    = self.air.specific_cp(T0, P0)
        R0     = self.air.specific_R(T0, P0)

        a0 = np.sqrt(gamma0 * R0 * T0)
        V0 = Ma * a0
        A0 = m_air / (rho0 * V0)

        # Stagnation: iterate because cp = cp(T)
        Tt0 = T0 + 0.5*V0**2 / cp0          # first-order estimate
        Pt0 = P0*(Tt0/T0)**(gamma0/(gamma0 - 1))

        return {
            "Ma":    Ma,    # FIX: expose Ma so isolator_properties can read it
            "T0":    T0,
            "P0":    P0,
            "rho0":  rho0,
            "gamma0": gamma0,
            "cp0":   cp0,
            "R0":    R0,
            "a0":    a0,
            "V0":    V0,    # FIX: expose V0
            "A0":    A0,
            "Tt0":   Tt0,
            "Pt0":   Pt0,
            "mdot":  m_air, # FIX: expose mdot so isolator can compute A1
        }

    # ------------------------------------------------------------------
    def pressure_recovery(self, Ma: float) -> float:   # FIX: add self
        MaList = np.array([
            8.126582278481013, 7.640506792672073, 7.245569156695016,
            6.8658223212519776, 6.6075949367088604, 6.349367552165743,
            6.136709092538568, 5.954429916188687, 5.75696225709553,
            5.605063522918315, 5.4531647887411, 5.2860764129252376,
            5.1645569620253164, 5.027847869486749
        ])
        sigma_list = np.array([
            0.3021505460144819, 0.31827957959571584, 0.333870966418766,
            0.3505376467581838, 0.36344086131769493, 0.3774193693935738,
            0.38870968469678685, 0.3999999897454365, 0.4123655985650173,
            0.42311828761917336, 0.4338709766733293, 0.4451613022311057,
            0.4543010920289637, 0.46612904383579723
        ])
        coeffs = np.polyfit(MaList, sigma_list, 1)
        return float(np.poly1d(coeffs)(Ma))

    # ------------------------------------------------------------------
    def isolator_properties(self, inlet_props: dict) -> dict:
        Ma0    = inlet_props["Ma"]
        T0     = inlet_props["T0"]
        P0     = inlet_props["P0"]
        V0     = inlet_props["V0"]
        Pt0    = inlet_props["Pt0"]
        gamma0 = inlet_props["gamma0"]
        cp0    = inlet_props["cp0"]
        mdot   = inlet_props["mdot"]

        # Step 1: Ma1 from fixed epsilon = 0.4 (paper §2.2, eq.10)
        M1 = self.EPSILON * Ma0

        # Step 2: total enthalpy conserved (adiabatic wall, eq.13)
        Ht0 = cp0*T0 + 0.5*V0**2

        # Step 3: iteratively solve T1 — implicit because R and γ depend on T
        def energy_residual(T1_arr):
            T1g = float(T1_arr[0])
            R1  = self.air.specific_R(T1g, P0)
            g1  = self.air.specific_heat_ratio(T1g, P0)
            cp1 = g1*R1/(g1 - 1)
            V1  = M1*np.sqrt(g1*R1*T1g)
            return [Ht0 - (cp1*T1g + 0.5*V1**2)]

        T1  = float(fsolve(energy_residual, x0=[1200.0])[0])

        # Step 4: consistent thermo at T1
        R1     = self.air.specific_R(T1, P0)
        gamma1 = self.air.specific_heat_ratio(T1, P0)
        cp1    = gamma1*R1/(gamma1 - 1)
        V1     = M1*np.sqrt(gamma1*R1*T1)

        # Step 5: Tt1 from recovered enthalpy (adiabatic ⟹ Ht conserved)
        # Ht0 = cp1*Tt1  →  Tt1 = Ht0/cp1          FIX: replaces wrong formula
        Tt1 = Ht0 / cp1

        # Step 6: pressure recovery → pt1 → p1  (isentropic relation at section 1)
        sigma_c = self.pressure_recovery(Ma0)
        pt1     = sigma_c * Pt0
        p1      = pt1 * (T1/Tt1)**(gamma1/(gamma1 - 1))

        # Step 7: density and capture area (eqs. 12, 14)
        rho1 = p1 / (R1*T1)
        A1   = mdot / (rho1*V1)

        return {
            "Ma1":    M1,
            "T1":     T1,
            "Tt1":    Tt1,
            "p1":     p1,
            "pt1":    pt1,
            "V1":     V1,
            "A1":     A1,
            "rho1":   rho1,
            "gamma1": gamma1,
            "cp1":    cp1,
            "R1":     R1,
            "sigma_c": sigma_c,
        }
    def optimal_fuel_air_ratio(self) -> float:
        return 0.0291 # FIX: placeholder, replace with actual calculation
    

    def combustor_properties(self, isolator_props: dict) -> dict:
        mfuel =self.optimal_fuel_air_ratio() * isolator_props["mdot"]
        mtotal = isolator_props["mdot"] + mfuel

        




# ── pretty-print helper ───────────────────────────────────────────────────────
def print_section(title, props, fields):
    w = 32
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    for label, key, unit, scale in fields:
        val = props.get(key, float("nan"))
        print(f"  {label:<{w}} {val*scale:>12.4f}  {unit}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    eng  = Engine()

    # ── flight condition ──────────────────────────────────────────────
    h_km = 30.0          # altitude [km]  — paper uses 30 km as example
    Ma0  = 5.0           # flight Mach number
    mdot = 143.0           # air mass-flow rate [kg/s]  (paper normalises to 1 kg/s)

    # ── Section 0 : freestream / inlet entrance ───────────────────────
    inp = eng.inlet_properties(h=h_km*1e3, Ma=Ma0, m_air=mdot)

    print_section("SECTION 0 — Freestream (inlet entrance)", inp, [
        ("Altitude",               "—",     "km",   1e-3),   # placeholder
        ("Flight Mach number",     "Ma",    "—",    1.0),
        ("Static temperature T₀",  "T0",    "K",    1.0),
        ("Static pressure P₀",     "P0",    "kPa",  1e-3),
        ("Air density ρ₀",         "rho0",  "kg/m³",1.0),
        ("Specific heat ratio γ₀", "gamma0","—",    1.0),
        ("Speed of sound a₀",      "a0",    "m/s",  1.0),
        ("Flight velocity V₀",     "V0",    "m/s",  1.0),
        ("Capture area A₀",        "A0",    "m²",   1.0),
        ("Total temperature Tt₀",  "Tt0",   "K",    1.0),
        ("Total pressure Pt₀",     "Pt0",   "kPa",  1e-3),
        ("Mass-flow rate ṁ",       "mdot",  "kg/s", 1.0),
    ])

    # fix the altitude row (not stored in dict, just display)
    # ── Section 1 : isolator entrance ────────────────────────────────
    iso = eng.isolator_properties(inp)

    print_section("SECTION 1 — Isolator entrance (combustor inlet)", iso, [
        ("Mach number Ma₁",        "Ma1",    "—",    1.0),
        ("Static temperature T₁",  "T1",     "K",    1.0),
        ("Total temperature Tt₁",  "Tt1",    "K",    1.0),
        ("Static pressure p₁",     "p1",     "kPa",  1e-3),
        ("Total pressure pt₁",     "pt1",    "kPa",  1e-3),
        ("Velocity V₁",            "V1",     "m/s",  1.0),
        ("Density ρ₁",             "rho1",   "kg/m³",1.0),
        ("Area A₁",                "A1",     "m²",   1.0),
        ("Specific heat ratio γ₁", "gamma1", "—",    1.0),
        ("Spec. heat cp₁",         "cp1",    "J/kg/K",1.0),
        ("Gas constant R₁",        "R1",     "J/kg/K",1.0),
        ("Pressure recovery σc",   "sigma_c","—",    1.0),
    ])

    print(f"\n  [Altitude = {h_km:.1f} km,  Ma₀ = {Ma0},  ṁ = {mdot} kg/s]\n")
