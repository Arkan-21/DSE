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

class ShapiroODE:
    """
    Quasi-1D Shapiro differential equations for a duct with area change,
    heat addition, wall friction, and mass addition .

    All differentials are per unit streamwise step dx [m].

    State vector  s = [Ma², p, T]
    Inputs per step:
        Ma2   – Mach number squared
        p     – static pressure [Pa]
        T     – static temperature [K]
        gamma – local specific heat ratio  γ
        Cp    – local specific heat at constant pressure [J/(kg·K)]
        dA_dx – area gradient  dA/dx  [m²/m]  (positive = diverging)
        A     – local cross-sectional area [m²]
        D     – local hydraulic diameter [m]
        Cf    – wall friction coefficient (dimensionless, typically 0.002–0.005)
        dH_dx – heat release rate  dH/dx  [J/(kg·m)]  (zero in isolator/nozzle)
        mdot  – local mass-flow rate [kg/s]
        dmdot_dx – streamwise gradient of mass-flow rate [kg/(s·m)] (fuel injection)
        W     – local mixture molar mass [kg/mol]
        dW_dx – streamwise gradient of molar mass [kg/(mol·m)]

    Returns d(Ma²)/dx, dp/dx, dT/dx  — the three Shapiro derivatives.
    """

    @staticmethod
    def derivatives(
        Ma2: float,
        p: float,
        T: float,
        gamma: float,
        Cp: float,
        dA_dx: float,
        A: float,
        D: float,
        Cf: float,
        dH_dx: float,
        mdot: float,
        dmdot_dx: float,
        W: float,
        dW_dx: float,
    ) -> tuple[float, float, float]:
        """Return (dMa2_dx, dp_dx, dT_dx) from Shapiro Eqs. (15)–(17).

        The singularity at Ma=1 is *not* handled here; the caller must ensure
        the flow stays away from sonic conditions (|1 - Ma²| > tolerance).
        """
        g   = gamma
        M2  = Ma2
        dA  = dA_dx          # dA/dx
        dH  = dH_dx          # dH/dx  [J/(kg·m)]
        dm  = dmdot_dx       # d(ṁ)/dx
        dW  = dW_dx          # dW/dx

        D1  = 1.0 - M2                              # denominator (1 – Ma²)

        # Frequently recurring groups
        g1m2 = 1.0 + (g - 1.0) / 2.0 * M2          # 1 + (γ-1)/2 · Ma²
        gM2  = g * M2
        fric = 4.0 * Cf / D                          # 4Cf/D  [1/m]

        # -- dMa²/dx  (Eq. 15) --
        dMa2_dx = (
            -2.0 * g1m2 / D1 * dA / A
            + (1.0 + gM2) / D1 * (dH / (Cp * T))
            + gM2 * g1m2 / D1 * fric                #dx is missing
            + 2.0 * (1.0 + gM2) * g1m2 / D1 * dm / mdot
            - (1.0 + gM2) / D1 * dW / W
            - (1.0 / g) * (g / (g - 1.0))           # -dγ/γ  ≈ 0 (frozen γ per step)
            * 0.0                                    # set to zero; caller updates γ
        )

        # -- dp/dx  (Eq. 16) --
        dp_dx = (
            gM2 / D1 * dA / A
            - gM2 / D1 * (dH / (Cp * T))
            - gM2 * (1.0 + (g - 1.0) * M2) / (2.0 * D1) * fric
            - 2.0 * gM2 * g1m2 / D1 * dm / mdot
            + gM2 / D1 * dW / W
        ) * p

        # -- dT/dx  (Eq. 17) --
        dT_dx = (
            (g - 1.0) * M2 / D1 * dA / A
            + (1.0 + gM2) / D1 * (dH / (Cp * T))
            - g * (g - 1.0) * M2**2 / (2.0 * D1) * fric
            - (g - 1.0) * M2 * (1.0 + gM2) / D1 * dm / mdot
            + (g - 1.0) * M2 / D1 * dW / W
        ) * T

        return dMa2_dx, dp_dx, dT_dx

    @staticmethod
    def integrate(
        x_start: float,
        x_end: float,
        Ma2_in: float,
        p_in: float,
        T_in: float,
        mdot_in: float,
        geometry_fn,          # callable(x) → (A, dA_dx, D)
        thermo_fn,            # callable(T, p) → (gamma, Cp, W, dW_dx)
        source_fn,            # callable(x, mdot) → (dH_dx, dmdot_dx)
        Cf: float = 0.003,
        n_steps: int = 200,
    ) -> dict:
        """
        Integrate the Shapiro equations from x_start to x_end using a
        simple 4th-order Runge-Kutta scheme.

        Parameters
        ----------
        x_start, x_end : float
            Streamwise integration limits [m].
        Ma2_in, p_in, T_in : float
            Inlet Mach²,  static pressure [Pa], static temperature [K].
        mdot_in : float
            Inlet mass-flow rate [kg/s].
        geometry_fn : callable(x) → (A [m²], dA_dx [m²/m], D [m])
            Returns local area, area gradient, and hydraulic diameter.
        thermo_fn : callable(T, p) → (gamma, Cp [J/kg/K], W [kg/mol], dW_dx [kg/mol/m])
            Returns local thermodynamic properties.
            dW_dx is usually 0 for a single-species stream.
        source_fn : callable(x, mdot) → (dH_dx [J/kg/m], dmdot_dx [kg/s/m])
            Returns local heat-release and mass-addition rates.
        Cf : float
            Wall friction coefficient (constant along duct).
        n_steps : int
            Number of integration steps.

        Returns
        -------
        dict with arrays:
            x, Ma, Ma2, p, T, rho, V, Tt, Pt, A, mdot
        """
        dx = (x_end - x_start) / n_steps
        xs    = np.zeros(n_steps + 1)
        Ma2s  = np.zeros(n_steps + 1)
        ps    = np.zeros(n_steps + 1)
        Ts    = np.zeros(n_steps + 1)
        mdots = np.zeros(n_steps + 1)

        xs[0]    = x_start
        Ma2s[0]  = Ma2_in
        ps[0]    = p_in
        Ts[0]    = T_in
        mdots[0] = mdot_in

        def _step(x, Ma2, p, T, mdot):
            A, dA_dx, D      = geometry_fn(x)
            g, Cp, W, dW_dx  = thermo_fn(T, p)
            dH_dx, dmdot_dx  = source_fn(x, mdot)
            dM, dpx, dTx = ShapiroODE.derivatives(
                Ma2, p, T, g, Cp, dA_dx, A, D, Cf, dH_dx, mdot, dmdot_dx, W, dW_dx
            )
            return dM, dpx, dTx, dmdot_dx

        for i in range(n_steps):
            x   = xs[i]
            M2  = Ma2s[i]
            p   = ps[i]
            T   = Ts[i]
            md  = mdots[i]

            k1M, k1p, k1T, k1m = _step(x,          M2,           p,           T,           md)
            k2M, k2p, k2T, k2m = _step(x + dx/2,   M2+dx/2*k1M, p+dx/2*k1p, T+dx/2*k1T, md+dx/2*k1m)
            k3M, k3p, k3T, k3m = _step(x + dx/2,   M2+dx/2*k2M, p+dx/2*k2p, T+dx/2*k2T, md+dx/2*k2m)
            k4M, k4p, k4T, k4m = _step(x + dx,     M2+dx*k3M,   p+dx*k3p,   T+dx*k3T,   md+dx*k3m)

            xs[i+1]    = x + dx
            Ma2s[i+1]  = M2 + dx/6 * (k1M + 2*k2M + 2*k3M + k4M)
            ps[i+1]    = p  + dx/6 * (k1p + 2*k2p + 2*k3p + k4p)
            Ts[i+1]    = T  + dx/6 * (k1T + 2*k2T + 2*k3T + k4T)
            mdots[i+1] = md + dx/6 * (k1m + 2*k2m + 2*k3m + k4m)

        # Derived quantities at each station
        Mas  = np.sqrt(np.maximum(Ma2s, 0.0))
        As   = np.array([geometry_fn(x)[0] for x in xs])
        gs   = np.array([thermo_fn(Ts[i], ps[i])[0] for i in range(len(xs))])
        Cps  = np.array([thermo_fn(Ts[i], ps[i])[1] for i in range(len(xs))])
        Rs   = Cps * (gs - 1.0) / gs                   # specific gas constant per station
        Vs   = Mas * np.sqrt(gs * Rs * Ts)
        rhos = ps / (Rs * Ts)
        Tts  = Ts + Vs**2 / (2.0 * Cps)
        Pts  = ps * (Tts / Ts) ** (gs / (gs - 1.0))

        return {
            "x":    xs,
            "Ma":   Mas,
            "Ma2":  Ma2s,
            "p":    ps,
            "T":    Ts,
            "rho":  rhos,
            "V":    Vs,
            "Tt":   Tts,
            "Pt":   Pts,
            "A":    As,
            "mdot": mdots,
        }
    
class Engine:

    #-------------------ENGINE GEOMETRIC RATIOS (from paper)-------------------
    L12=0.40,
    L23=0.01,
    L34=1.00,
    L45=0.40,
    alpha12=1.0,
    alpha13=1.1,
    alpha14=2.5,
    alpha05=2.0,

    def __init__(self):
        self.air = AirProperties()
        self.shapiroODE = ShapiroODE()


    #----mach ratio inlet-isolator based on the maximal tempreture at isolator inlet----
    EPSILON = 0.4
    #----compressive efficiency for the 4 shock waves in the isolator-------------------
    ETA_C   = 0.9

    #----wall friction coefficient------------------------------------------------------
    CF_DEFAULT = 0.003


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
            "Ma":    Ma,    
            "T0":    T0,
            "P0":    P0,
            "rho0":  rho0,
            "gamma0": gamma0,
            "cp0":   cp0,
            "R0":    R0,
            "a0":    a0,
            "V0":    V0,    
            "A0":    A0,
            "Tt0":   Tt0,
            "Pt0":   Pt0,
            "mdot":  m_air, 
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

        # Step 1: Ma1 from fixed epsilon = 0.4 
        M1 = self.EPSILON * Ma0

        # Step 2: total enthalpy conserved (
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
    
    def combustor_properties2(self, isolator_props: dict) -> dict:
        L_12 = self.L12
        A1  = isolator_props["A1"]
        A2  = self.alpha12 * A1
        cf = self.CF_DEFAULT
        n_steps = 300

        Ma1 = isolator_props["Ma1"]
        T1  = isolator_props["T1"]  
        p1  = isolator_props["p1"]
        mdot = isolator_props["mdot"]
        A1  = isolator_props["A1"]

        gamma1 = isolator_props["gamma1"]
        cp1    = isolator_props["cp1"]

        def geometry_fn(x):

            # linear area variation
            A = A1 + (A2 - A1) * (x / L_12)

            dA_dx = (A2 - A1) / L_12

            # circular hydraulic diameter
            D = np.sqrt(4 * A / np.pi)

            return A, dA_dx, D
        
        def thermo_fn(T, p):
            gamma = self.air.specific_heat_ratio(T, p)
            Cp    = self.air.specific_cp(T, p)
            comp = self.air.equilibrium_composition(T, p/101325)

            W = (sum(comp[s] * self.air.MOLECULAR_WEIGHTS[s]
                    for s in comp)* 1e-3)
            dW_dx = 0.0
            return gamma, Cp, W, dW_dx
        
        def source_fn(x, mdot_local):
            dH_dx = 0.0
            dmdot_dx = 0.0
            return dH_dx, dmdot_dx
        
        result = self.shapiroODE.integrate(
            x_start = 0.0,
            x_end   = L_12,
            Ma2_in  = Ma1**2,
            p_in    = p1,
            T_in    = T1,
            mdot_in = mdot,
            geometry_fn = geometry_fn,
            thermo_fn   = thermo_fn,
            source_fn   = source_fn,
            Cf          = cf,
            n_steps     = n_steps,
        )

        # ----------------------------------------------------------
        # Extract outlet state
        # ----------------------------------------------------------
        Ma2   = result["Ma"][-1]
        T2    = result["T"][-1]
        p2    = result["p"][-1]
        rho2  = result["rho"][-1]
        V2    = result["V"][-1]
        Tt2   = result["Tt"][-1]
        Pt2   = result["Pt"][-1]

        gamma2 = self.air.specific_heat_ratio(T2, p2)

        cp2 = self.air.specific_cp(T2, p2)

        R2 = self.air.specific_R(T2, p2)

        return {
            "Ma2": Ma2,
            "T2": T2,
            "Tt2": Tt2,
            "p2": p2,
            "pt2": Pt2,
            "rho2": rho2,
            "V2": V2,
            "A2": A2,
            "gamma2": gamma2,
            "cp2": cp2,
            "R2": R2,
            "mdot": mdot,
            "solution": result,
        }


    def optimal_fuel_air_ratio(self) -> float:
        return 0.0291 # FIX: use nasa cea to make it more accurate, this is just a placeholder value for hydrogen
    
    def combustor_properties3(
        self,
        combustor_properties2: dict
    ) -> dict:
        """
        Section 2 -> 3

        Fuel injection region.

        Assumptions:
        ------------
        - fuel added gradually
        - NO combustion yet
        - NO heat release
        - momentum affected by mass addition
        """

        # ----------------------------------------------------------
        # Geometry
        # ----------------------------------------------------------
        L_23 = self.L23

        A2 = combustor_properties2["A2"]

        A3 = self.alpha13 * A2

        cf = self.CF_DEFAULT

        n_steps = 300

        # ----------------------------------------------------------
        # Inlet state
        # ----------------------------------------------------------
        Ma2  = combustor_properties2["Ma2"]

        T2   = combustor_properties2["T2"]

        p2   = combustor_properties2["p2"]

        mdot_air = combustor_properties2["mdot"]

        # ----------------------------------------------------------
        # Fuel injection
        # ----------------------------------------------------------
        combustion_phi = 1.0

        FAR = self.optimal_fuel_air_ratio()

        mfuel_total = FAR * mdot_air * combustion_phi

        # distributed fuel injection
        dmdot_dx_const = mfuel_total / L_23

        # ----------------------------------------------------------
        # Geometry function
        # ----------------------------------------------------------
        def geometry_fn(x):

            A = A2 + (A3 - A2) * (x / L_23)

            dA_dx = (A3 - A2) / L_23

            D = np.sqrt(4 * A / np.pi)

            return A, dA_dx, D

        # ----------------------------------------------------------
        # Thermodynamics
        # ----------------------------------------------------------
        def thermo_fn(T, p):

            gamma = self.air.specific_heat_ratio(T, p)

            Cp = self.air.specific_cp(T, p)

            # air molecular weight
            W_air = 28.97e-3

            # hydrogen molecular weight
            W_H2 = 2.016e-3

            # crude mixture estimate
            xfuel = min(
                (dmdot_dx_const * L_23) /
                (mdot_air + dmdot_dx_const * L_23),
                0.99
            )

            W = (1 - xfuel) * W_air + xfuel * W_H2

            dW_dx = 0.0

            return gamma, Cp, W, dW_dx

        # ----------------------------------------------------------
        # Source terms
        # ----------------------------------------------------------
        def source_fn(x, mdot_local):

            # NO heat release yet
            dH_dx = 0.0

            # fuel addition only
            dmdot_dx = dmdot_dx_const

            return dH_dx, dmdot_dx

        # ----------------------------------------------------------
        # Integrate Shapiro equations
        # ----------------------------------------------------------
        result = self.shapiroODE.integrate(
            x_start = 0.0,
            x_end   = L_23,

            Ma2_in  = Ma2**2,

            p_in    = p2,

            T_in    = T2,

            mdot_in = mdot_air,

            geometry_fn = geometry_fn,

            thermo_fn = thermo_fn,

            source_fn = source_fn,

            Cf = cf,

            n_steps = n_steps,
        )

        # ----------------------------------------------------------
        # Extract outlet state
        # ----------------------------------------------------------
        Ma3  = result["Ma"][-1]

        T3   = result["T"][-1]

        p3   = result["p"][-1]

        rho3 = result["rho"][-1]

        V3   = result["V"][-1]

        Tt3  = result["Tt"][-1]

        Pt3  = result["Pt"][-1]

        mdot3 = result["mdot"][-1]

        gamma3 = self.air.specific_heat_ratio(T3, p3)

        cp3 = self.air.specific_cp(T3, p3)

        R3 = self.air.specific_R(T3, p3)

        return {
            "Ma3": Ma3,
            "T3": T3,
            "Tt3": Tt3,
            "p3": p3,
            "pt3": Pt3,
            "rho3": rho3,
            "V3": V3,
            "A3": A3,
            "gamma3": gamma3,
            "cp3": cp3,
            "R3": R3,
            "mdot": mdot3,
            "mfuel": mfuel_total,
            "solution": result,
        }

    def combustor_properties4(
        self,
        combustor_properties3: dict
    ) -> dict:
        """
        Section 3 -> 4

        Combustion region.

        Assumptions:
        ------------
        - no additional fuel injection
        - equilibrium combustion
        - finite mixing efficiency
        - distributed heat release
        """

        # ----------------------------------------------------------
        # Geometry
        # ----------------------------------------------------------
        L_34 = self.L34

        A3 = combustor_properties3["A3"]

        A4 = self.alpha14 * A3

        cf = self.CF_DEFAULT

        n_steps = 400

        # ----------------------------------------------------------
        # Inlet state
        # ----------------------------------------------------------
        Ma3 = combustor_properties3["Ma3"]

        T3 = combustor_properties3["T3"]

        p3 = combustor_properties3["p3"]

        mdot = combustor_properties3["mdot"]

        mfuel = combustor_properties3["mfuel"]

        # ----------------------------------------------------------
        # Combustion parameters
        # ----------------------------------------------------------
        combustion_phi = 1.0

        # hydrogen lower heating value
        LHV = 120e6     # J/kg

        # injector angle
        theta_deg = 90.0

        # ----------------------------------------------------------
        # Mixing efficiency
        # ----------------------------------------------------------
        def mixing_efficiency(x):

            xi = x

            # normalized combustor coordinate
            s = (xi - 0.0) / L_34

            s = np.clip(s, 1e-6, 1.0)

            if theta_deg == 0.0:

                eta_mix = s

            elif theta_deg == 90.0:

                eta_mix = 1.0 + 0.176 * np.log(s)

            else:
                raise ValueError(
                    "Only θ = 0° or 90° supported."
                )

            return np.clip(eta_mix, 0.0, 1.0)

        # ----------------------------------------------------------
        # Geometry function
        # ----------------------------------------------------------
        def geometry_fn(x):

            A = A3 + (A4 - A3) * (x / L_34)

            dA_dx = (A4 - A3) / L_34

            D = np.sqrt(4 * A / np.pi)

            return A, dA_dx, D

        # ----------------------------------------------------------
        # Thermodynamics
        # ----------------------------------------------------------
        def thermo_fn(T, p):

            gamma = self.air.specific_heat_ratio(T, p)

            Cp = self.air.specific_cp(T, p)

            comp = self.air.equilibrium_composition(
                T,
                p / 101325
            )

            W = (
                sum(
                    comp[s] * self.air.MOLECULAR_WEIGHTS[s]
                    for s in comp
                )
                * 1e-3
            )

            dW_dx = 0.0

            return gamma, Cp, W, dW_dx

        # ----------------------------------------------------------
        # Heat release source term
        # ----------------------------------------------------------
        def source_fn(x, mdot_local):

            # no more fuel injection
            dmdot_dx = 0.0

            # local mixing efficiency
            eta_mix = mixing_efficiency(x)

            # heat release distributed along combustor
            q_total = eta_mix * mfuel * LHV

            dH_dx = q_total / (L_34 * mdot_local)

            return dH_dx, dmdot_dx

        # ----------------------------------------------------------
        # Integrate Shapiro equations
        # ----------------------------------------------------------
        result = self.shapiroODE.integrate(
            x_start = 0.0,
            x_end   = L_34,

            Ma2_in  = Ma3**2,

            p_in    = p3,

            T_in    = T3,

            mdot_in = mdot,

            geometry_fn = geometry_fn,

            thermo_fn = thermo_fn,

            source_fn = source_fn,

            Cf = cf,

            n_steps = n_steps,
        )

        # ----------------------------------------------------------
        # Extract outlet state
        # ----------------------------------------------------------
        Ma4  = result["Ma"][-1]

        T4   = result["T"][-1]

        p4   = result["p"][-1]

        rho4 = result["rho"][-1]

        V4   = result["V"][-1]

        Tt4  = result["Tt"][-1]

        Pt4  = result["Pt"][-1]

        gamma4 = self.air.specific_heat_ratio(T4, p4)

        cp4 = self.air.specific_cp(T4, p4)

        R4 = self.air.specific_R(T4, p4)

        return {
            "Ma4": Ma4,
            "T4": T4,
            "Tt4": Tt4,
            "p4": p4,
            "pt4": Pt4,
            "rho4": rho4,
            "V4": V4,
            "A4": A4,
            "gamma4": gamma4,
            "cp4": cp4,
            "R4": R4,
            "mdot": mdot,
            "solution": result,
        }
  




        




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
