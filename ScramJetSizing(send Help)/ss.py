from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp



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
        "H2": {
            "Trange": [200, 1000, 6000],
            "low": [2.34433112, 7.98052075e-03, -1.94781510e-05,
                    2.01572094e-08, -7.37611761e-12, -917.935173, 0.683010238],
            "high": [2.93286575, 8.26607967e-04, -1.46402364e-07,
                     1.54100414e-11, -6.88804800e-16, -813.065581, -1.02432865],
        },
        "H": {
            "Trange": [200, 1000, 6000],
            "low": [2.5, 0, 0, 0, 0, 25471.6270, -0.448813240],
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
    @staticmethod
    def derivatives(Ma2, p, T, gamma, Cp, dA_dx, A, D, Cf, dH_dx, mdot, dmdot_dx, W, dW_dx, dgamma_dx):
        
        g = gamma; M2 = Ma2
        D1 = 1.0 - M2
        g1m2 = 1.0 + (g - 1.0) / 2.0 * M2
        gM2 = g * M2
        fric = 4.0 * Cf / D

        dMa2_dx = (
            -2.0 * g1m2 / D1 * dA_dx / A
            + (1.0 + gM2) / D1 * (dH_dx / (Cp * T))
            + gM2 * g1m2 / D1 * fric
            + 2.0 * (1.0 + gM2) * g1m2 / D1 * dmdot_dx / mdot
            - (1.0 + gM2) / D1 * dW_dx / W - dgamma_dx/gamma
            )*M2
        

        dp_dx = (
            gM2 / D1 * dA_dx / A
            - gM2 / D1 * (dH_dx / (Cp * T))
            - gM2 * (1.0 + (g - 1.0) * M2) / (2.0 * D1) * fric
            - 2.0 * gM2 * g1m2 / D1 * dmdot_dx / mdot
            + gM2 / D1 * dW_dx / W
        ) * p

        dT_dx = (
            (g - 1.0) * M2 / D1 * dA_dx / A
            + (1.0 + gM2) / D1 * (dH_dx / (Cp * T))
            - g * (g - 1.0) * M2**2 / (2.0 * D1) * fric
            - (g - 1.0) * M2 * (1.0 + gM2) / D1 * dmdot_dx / mdot
            + (g - 1.0) * M2 / D1 * dW_dx / W
        ) * T

        return dMa2_dx, dp_dx, dT_dx

    @staticmethod
    def integrate(x_start, x_end,
                Ma2_in, p_in, T_in, mdot_in,
                geometry_fn, thermo_fn, source_fn,
                Cf=0.003,
                n_steps=1000):

        # -------------------------------------------------------------
        # Governing ODE system
        # -------------------------------------------------------------

        def rhs(x, y):

            M2, p, T, mdot = y

            A, dA_dx, D = geometry_fn(x)

            gamma, Cp, W, dW_dx, dgamma_dx = thermo_fn(T, p)

            dH_dx, dmdot_dx = source_fn(x, mdot)

            dM2_dx, dp_dx, dT_dx = ShapiroODE.derivatives(
                Ma2=M2,
                p=p,
                T=T,
                gamma=gamma,
                Cp=Cp,
                dA_dx=dA_dx,
                A=A,
                D=D,
                Cf=Cf,
                dH_dx=dH_dx,
                mdot=mdot,
                dmdot_dx=dmdot_dx,
                W=W,
                dW_dx=dW_dx,
                dgamma_dx=dgamma_dx
            )

            return [
                dM2_dx,
                dp_dx,
                dT_dx,
                dmdot_dx
            ]

        # -------------------------------------------------------------
        # Choking event
        # -------------------------------------------------------------

        def choke_event(x, y):

            M2 = y[0]

            # Stop at sonic condition
            return M2 - 1.0

        choke_event.terminal = True
        choke_event.direction = -1

        # -------------------------------------------------------------
        # Nonphysical temperature event
        # -------------------------------------------------------------

        def temperature_event(x, y):

            T = y[2]
            return T - 1.0

        temperature_event.terminal = True
        temperature_event.direction = -1

        # -------------------------------------------------------------
        # Nonphysical pressure event
        # -------------------------------------------------------------

        def pressure_event(x, y):

            p = y[1]
            return p - 1.0

        pressure_event.terminal = True
        pressure_event.direction = -1

        # -------------------------------------------------------------
        # Initial state
        # -------------------------------------------------------------

        y0 = [
            max(Ma2_in, 3.000001),
            max(p_in, 1.0),
            max(T_in, 1.0),
            max(mdot_in, 1e-9)
        ]

        # -------------------------------------------------------------
        # Integrate
        # -------------------------------------------------------------

        sol = solve_ivp(
            fun=rhs,
            t_span=(x_start, x_end),
            y0=y0,

            # BEST OPTION FOR THIS PROBLEM
            method="BDF",

            # Tight tolerances
            rtol=1e-6,
            atol=1e-8,

            # Allow adaptive stepping
            max_step=(x_end - x_start)/50,

            events=[
                choke_event,
                temperature_event,
                pressure_event
            ],

            dense_output=False
        )

        # -------------------------------------------------------------
        # Extract solution
        # -------------------------------------------------------------

        xs = sol.t

        Ma2s = np.maximum(sol.y[0], 1.000001)
        ps = np.maximum(sol.y[1], 1.0)
        Ts = np.maximum(sol.y[2], 1.0)
        mdots = np.maximum(sol.y[3], 1e-9)

        Mas = np.sqrt(Ma2s)

        # -------------------------------------------------------------
        # Determine if choking occurred
        # -------------------------------------------------------------

        thermal_choke = len(sol.t_events[0]) > 0

        if thermal_choke:

            x_choke = sol.t_events[0][0]

            print(f"\n⚠ Thermal choking detected")
            print(f"   x = {x_choke:.5f} m")
            print(f"   M ≈ 1")

        # -------------------------------------------------------------
        # Derived flow properties
        # -------------------------------------------------------------

        As = np.array([
            geometry_fn(x)[0]
            for x in xs
        ])

        gs = np.array([
            thermo_fn(Ts[i], ps[i])[0]
            for i in range(len(xs))
        ])

        Cps = np.array([
            thermo_fn(Ts[i], ps[i])[1]
            for i in range(len(xs))
        ])

        Rs = Cps * (gs - 1.0) / gs

        Vs = Mas * np.sqrt(
            np.maximum(gs * Rs * Ts, 0.0)
        )

        rhos = ps / np.maximum(
            Rs * Ts,
            1e-12
        )

        Tts = Ts + Vs**2 / (
            2*np.maximum(Cps, 1e-12)
        )

        exponent = gs / np.maximum(gs - 1.0, 1e-12)

        Pts = ps * np.maximum(
            Tts / np.maximum(Ts, 1e-12),
            1.0
        ) ** exponent

        return {
            "x": xs,
            "Ma": Mas,
            "Ma2": Ma2s,
            "p": ps,
            "T": Ts,
            "rho": rhos,
            "V": Vs,
            "Tt": Tts,
            "Pt": Pts,
            "A": As,
            "mdot": mdots,
            "thermal_choke": thermal_choke,
            "solver_success": sol.success,
            "solver_message": sol.message,
        }


class Engine:
    L01 = 0.4
    L12 = 0.40
    L23 = 0.1
    L34 = 1.00
    L45 = 0.40
    alpha12 = 1.0
    alpha13 = 1.1
    alpha14 = 2.5
    alpha05 = 2.0

    EPSILON = 0.4
    ETA_C = 0.9
    CF_DEFAULT = 0.003

    def __init__(self):
        self.air = AirProperties()
        self.shapiroODE = ShapiroODE()

    def _f(self, x):
        """Force scalar float."""
        return float(np.asarray(x).squeeze())

    def inlet_properties(self, h, Ma, m_air):
        T0 = Atmosphere.T(h)
        P0 = Atmosphere.P(h)
        rho0 = Atmosphere.rho(h)

        gamma0 = self.air.specific_heat_ratio(T0, P0)
        cp0 = self.air.specific_cp(T0, P0)
        R0 = self.air.specific_R(T0, P0)

        a0 = np.sqrt(gamma0 * R0 * T0)
        V0 = Ma * a0
        A0 = m_air / (rho0 * V0)

        Tt0 = T0 + 0.5*V0**2 / cp0
        Pt0 = P0*(Tt0/T0)**(gamma0/(gamma0 - 1))
        print(f"\nInlet conditions at h={h:.0f} m, Ma={Ma:.2f}, m_air={m_air:.2f} kg/s:")
        print(f"  T0 = {T0:.2f} K")
        print(f"  P0 = {P0:.2f} Pa")
        print(f"  rho0 = {rho0:.4f} kg/m^3")
        print(f"  gamma0 = {gamma0:.4f}")
        print(f"  cp0 = {cp0:.2f} J/kg/K")
        print(f"  R0 = {R0:.2f} J/kg/K")
        print(f"  a0 = {a0:.2f} m/s")
        print(f"  V0 = {V0:.2f} m/s")
        print(f"  A0 = {A0:.4f} m^2")
        print(f"  Tt0 = {Tt0:.2f} K")
        print(f"  Pt0 = {Pt0:.2f} Pa")
        print(f"  mdot = {m_air:.2f} kg/s")
        return {
            "Ma": Ma, 
            "T": T0, 
            "P": P0, 
            "rho": rho0,
            "gamma": gamma0, 
            "cp": cp0, 
            "R": R0,
            "a": a0, 
            "V": V0, 
            "A": A0,
            "Tt": Tt0, 
            "Pt": Pt0, 
            "mdot": m_air,
        }
        

    def pressure_recovery(self, Ma):
        MaList = np.array([8.127, 7.641, 7.246, 6.866, 6.608, 6.349, 6.137,
                           5.954, 5.757, 5.605, 5.453, 5.286, 5.165, 5.028])
        sList = np.array([0.3022, 0.3183, 0.3339, 0.3505, 0.3634, 0.3774, 0.3887,
                          0.4000, 0.4124, 0.4231, 0.4339, 0.4452, 0.4543, 0.4661])
        return float(np.poly1d(np.polyfit(MaList, sList, 1))(Ma))

    def isolator_properties(self, inlet_props):
            
            #Initial conditions at inlet (section 0)
        Ma0 = self._f(inlet_props["Ma"])
        T0 = self._f(inlet_props["T"])
        Tt1 = self._f(inlet_props["Tt"])
        P0 = self._f(inlet_props["P"])
        V0 = self._f(inlet_props["V"])
        Pt0 = self._f(inlet_props["Pt"])
        gamma0 = self._f(inlet_props["gamma"])
        cp0 = self._f(inlet_props["cp"])
        mdot = self._f(inlet_props["mdot"])
        rho0 = self._f(inlet_props["rho"])
        R0 = self._f(inlet_props["R"])
        a0 = self._f(inlet_props["a"])   
        A0 = self._f(inlet_props["A"])


        M1 = self.EPSILON * Ma0
        Ht0 = cp0*T0 + 0.5*V0**2

        sigma_c = self.pressure_recovery(Ma0)
        Pt1_target = sigma_c * Pt0


        def residual(vars_):
            T1g, p1g = vars_

            # Thermodynamic properties
            R1 = self.air.specific_R(T1g, p1g)
            gamma1 = self.air.specific_heat_ratio(T1g, p1g)
            cp1 = gamma1 * R1 / (gamma1 - 1)

            # Velocity from Mach definition
            a1 = np.sqrt(gamma1 * R1 * T1g)
            V1 = M1 * a1

            # --- Equation 1: total enthalpy conservation ---
            eq1 = Ht0 - (cp1*T1g + 0.5*V1**2)

            # Total temperature from enthalpy
            Tt1g = Ht0 / cp1

            # Predicted total pressure
            Pt1g = p1g * (Tt1g/T1g)**(gamma1/(gamma1 - 1))

            # --- Equation 2: pressure recovery relation ---
            eq2 = Pt1g - Pt1_target

            return [eq1, eq2]
        
        T1, P1 = fsolve(residual,x0=[1200.0, 0.2*P0])

        R1 = self.air.specific_R(T1, P1)
        gamma1 = self.air.specific_heat_ratio(T1, P1)
        cp1 = gamma1*R1/(gamma1 - 1)
        V1 = M1*np.sqrt(gamma1*R1*T1)

        Tt1 = T1 + 0.5*V1**2 / cp1
        Pt1 = P1*(Tt1/T1)**(gamma1/(gamma1 - 1))


        rho1 = P1 / (R1*T1)
        A1 = mdot / (rho1*V1)

            # --- ADDED: Solution dictionary for plotting ---
            # We assume the isolator has a length self.L12. 
            # We create a simple linear transition from inlet (0) to isolator exit (1).
        L_iso = getattr(self, 'L01', 0.1)  # Default to 0.1 if L12 isn't defined
            
        sol = {
                "x": np.array([0.0, L_iso]),
                "Ma": np.array([Ma0, M1]),
                "T": np.array([T0, T1]),
                "Tt": np.array([Tt1, Tt1]),
                "P": np.array([P0, P1]),
                "Pt": np.array([Pt0, Pt1]),
                "A": np.array([A0, A1]),
                "rho": np.array([rho0, rho1]),
                "V": np.array([V0, V1]),
                "mdot": np.array([mdot, mdot])
            }

        return {
                "Ma": M1, 
                "T": T1, 
                "Tt": Tt1, 
                "P": P1, 
                "Pt": Pt1,
                "V": V1, 
                "A": A1, 
                "rho": rho1, 
                "gamma": gamma1,
                "cp": cp1, 
                "R": R1, 
                "sigma_c": sigma_c, 
                "mdot": mdot,
                "solution": sol,
         }

    def combustor_properties2(self, isolator_props):
        """Section 1→2: Isolator (friction only, constant area)"""
        L_12 = self._f(self.L12)
        A1 = self._f(isolator_props["A"])
        A2 = self._f(self.alpha12) * A1

        Ma1 = self._f(isolator_props["Ma"])
        T1 = self._f(isolator_props["T"])
        p1 = self._f(isolator_props["P"])
        mdot = self._f(isolator_props["mdot"])

        def geometry_fn(x):
            A = A1 + (A2 - A1) * (x / L_12)
            dA_dx = (A2 - A1) / L_12
            D = np.sqrt(4.0 * A / np.pi)
            return A, dA_dx, D

        def thermo_fn(T, p):
            gamma = self.air.specific_heat_ratio(T, p)
            Cp = self.air.specific_cp(T, p)
            comp = self.air.equilibrium_composition(T, p / 101325)
            W = sum(comp[s] * self.air.MOLECULAR_WEIGHTS[s] for s in comp) * 1e-3
            return gamma, Cp, W, 0.0, 0.0

        def source_fn(x, mdot_local):
            return 0.0, 0.0

        result = self.shapiroODE.integrate(
            x_start=0.0, x_end=L_12, Ma2_in=Ma1**2, p_in=p1, T_in=T1, mdot_in=mdot,
            geometry_fn=geometry_fn, thermo_fn=thermo_fn, source_fn=source_fn,
            Cf=self.CF_DEFAULT, n_steps=300,
        )

        return {
            "Ma": self._f(result["Ma"][-1]),
            "T": self._f(result["T"][-1]),
            "Tt": self._f(result["Tt"][-1]),
            "P": self._f(result["p"][-1]),
            "Pt": self._f(result["Pt"][-1]),
            "rho": self._f(result["rho"][-1]),
            "V": self._f(result["V"][-1]),
            "A": A2,
            "gamma": self.air.specific_heat_ratio(result["T"][-1], result["p"][-1]),
            "cp": self.air.specific_cp(result["T"][-1], result["p"][-1]),
            "R": self.air.specific_R(result["T"][-1], result["p"][-1]),
            "mdot": mdot,
            "solution": result,
        }

    def optimal_fuel_air_ratio(self):
        """Stoichiometric fuel-to-air ratio for H2 (paper Table 3)"""
        return 1.0 / 34.35  # H2 + 0.5 O2 → H2O

    def combustor_properties3(self, combustor_properties2, phi=0):
        """
        Section 2→3: Fuel injection
        Mass addition only (no combustion heat release yet),
        but with evolving H2-air mixture properties.
        """

        L_23 = self._f(self.L23)

        A2 = self._f(combustor_properties2["A"])
        A3 = self._f(self.alpha13) * A2 / self._f(self.alpha12)

        Ma2 = self._f(combustor_properties2["Ma"])
        T2 = self._f(combustor_properties2["T"])
        p2 = self._f(combustor_properties2["P"])
        mdot_air = self._f(combustor_properties2["mdot"])

        # Stoichiometric + actual fuel-air ratio
        FAR_stoich = self.optimal_fuel_air_ratio()
        FAR_actual = phi * FAR_stoich

        # Total injected fuel flow
        mfuel_total = FAR_actual * mdot_air

        # ---------------------------------------------------------
        # Geometry
        # ---------------------------------------------------------

        def geometry_fn(x):
            A = A2 + (A3 - A2) * (x / L_23)
            dA_dx = (A3 - A2) / L_23
            D = np.sqrt(4 * A / np.pi)
            return A, dA_dx, D

        # ---------------------------------------------------------
        # Thermodynamics
        # ---------------------------------------------------------

        def thermo_fn(T, p):

            # Local fuel fraction added up to position x
            # Estimated from local mdot increase
            # (updated internally by integrator)
            mdot_local = thermo_fn.mdot_current

            Yf = max((mdot_local - mdot_air) / mdot_local, 0.0)
            Ya = 1.0 - Yf

            # Molecular weights [kg/mol]
            W_air = 28.97e-3
            W_h2 = 2.016e-3

            # Correct mixture molecular weight
            W = 1.0 / (Ya/W_air + Yf/W_h2)

            # Approximate frozen mixture Cp
            cp_air = self.air.specific_cp(T, p)
            cp_h2 = 14.3e3  # J/kg/K approximate gaseous H2 Cp

            Cp = Ya * cp_air + Yf * cp_h2

            # Approximate mixture gamma
            R_univ = 8.314462618
            Rmix = R_univ / W

            gamma = Cp / (Cp - Rmix)

            # Composition gradients neglected
            dW_dx = 0.0
            dgamma_dx = 0.0

            return gamma, Cp, W, dW_dx, dgamma_dx

        # storage variable for local mdot
        thermo_fn.mdot_current = mdot_air

        # ---------------------------------------------------------
        # Source terms
        # ---------------------------------------------------------

        def source_fn(x, mdot_local):

            # Update thermo access to local mdot
            thermo_fn.mdot_current = mdot_local

            # Uniform distributed injection
            dmdot_dx = mfuel_total / L_23

            # No combustion heat release yet
            dH_dx = 0.0

            return dH_dx, dmdot_dx

        # ---------------------------------------------------------
        # Integrate
        # ---------------------------------------------------------

        result = self.shapiroODE.integrate(
            x_start=0.0,
            x_end=L_23,
            Ma2_in=Ma2**2,
            p_in=p2,
            T_in=T2,
            mdot_in=mdot_air,
            geometry_fn=geometry_fn,
            thermo_fn=thermo_fn,
            source_fn=source_fn,
            Cf=self.CF_DEFAULT,
            n_steps=50,
        )

        return {
            "Ma3": self._f(result["Ma"][-1]),
            "T3": self._f(result["T"][-1]),
            "P3": self._f(result["p"][-1]),
            "rho3": self._f(result["rho"][-1]),
            "V3": self._f(result["V"][-1]),
            "Tt3": self._f(result["Tt"][-1]),
            "Pt3": self._f(result["Pt"][-1]),
            "A3": A3,
            "mdot": self._f(result["mdot"][-1]),
            "mfuel": mfuel_total,
            "phi": phi,
            "solution": result,
        }

 

    def combustor_properties4(self, combustor_properties3):
        """Section 3→4: Combustion (heat release, no mass addition)"""
        L_34 = self._f(self.L34)

        A3 = self._f(combustor_properties3["A3"])
        A1_ref = self._f(combustor_properties3["A3"]) / self._f(self.alpha13)
        A4 = self._f(self.alpha14) * A1_ref

        Ma3 = self._f(combustor_properties3["Ma3"])
        T3 = self._f(combustor_properties3["T3"])
        p3 = self._f(combustor_properties3["P3"])

        mdot = self._f(combustor_properties3["mdot"])
        mfuel = self._f(combustor_properties3["mfuel"])

        Q_H2 = 141e6  # J/kg (higher heating value, paper Table 3)
        theta = 0  # injection angle

        # Total heat available [J/s] = mfuel [kg/s] * Q [J/kg]
        Q_total = mfuel * Q_H2

        def mixing_efficiency(x):
            s = np.clip(x / L_34, 1e-4, 1.0)

            if theta == 0.0:
                return s
            else:
                return np.clip(
                    1.01 + 0.176 * np.log(s),
                    0.0,
                    1.0
            )

        def geometry_fn(x):
            A = A3 + (A4 - A3) * (x / L_34)
            dA_dx = (A4 - A3) / L_34
            D = np.sqrt(4 * A / np.pi)
            return A, dA_dx, D

        def thermo_fn(T, p):
            gamma = self.air.specific_heat_ratio(T, p)
            Cp = self.air.specific_cp(T, p)
            comp = self.air.equilibrium_composition(T, p / 101325)
            W = sum(comp[s] * self.air.MOLECULAR_WEIGHTS[s] for s in comp) * 1e-3
            dW_dx = 0.0
            dgamma_dx = 0.0
            return gamma, Cp, W, dW_dx, dgamma_dx

        def source_fn(x, mdot_local):
            # dη/dx computed via finite difference
            h = 1e-4
            eta_p = mixing_efficiency(min(x + h, L_34))
            eta_m = mixing_efficiency(max(x - h, 0))
            deta_dx = (eta_p - eta_m) / (2 * h)

            # Heat release rate [J/(kg·m)]
            dH_dx = (Q_total / mdot_local) * deta_dx

            return dH_dx, 0.0

        result = self.shapiroODE.integrate(
            x_start=0.0, x_end=L_34, Ma2_in=Ma3**2, p_in=p3, T_in=T3, mdot_in=mdot,
            geometry_fn=geometry_fn, thermo_fn=thermo_fn, source_fn=source_fn,
            Cf=self.CF_DEFAULT, n_steps=500,
        )

        return {
            "Ma4": result["Ma"][-1],
            "T4": result["T"][-1],
            "p4": result["p"][-1],
            "rho4": result["rho"][-1],
            "V4": result["V"][-1],
            "Tt4": result["Tt"][-1],
            "Pt4": result["Pt"][-1],
            "A4": A4,
            "mdot": mdot,
            "solution": result,
            "thermal_choke": result["thermal_choke"],
        }

    def nozzle_properties(self, combustor_properties4, inlet_props):
        """Section 4→5: Nozzle expansion (frozen flow, isentropic)"""
        if combustor_properties4["thermal_choke"]:
            return {"thermal_choke": True}

        L_45 = self._f(self.L45)

        A4 = self._f(combustor_properties4["A"])
        A0 = self._f(inlet_props["A"])
        A5 = self._f(self.alpha05) * A0

        Ma4 = self._f(combustor_properties4["Ma4"])
        T4 = self._f(combustor_properties4["T4"])
        p4 = self._f(combustor_properties4["p4"])
        Tt4 = self._f(combustor_properties4["Tt4"])
        Pt4 = self._f(combustor_properties4["Pt4"])

        mdot = self._f(combustor_properties4["mdot"])
        P_amb = self._f(inlet_props["P0"])

        def geometry_fn(x):
            A = A4 + (A5 - A4) * (x / L_45)
            dA_dx = (A5 - A4) / L_45
            D = np.sqrt(4 * A / np.pi)
            return A, dA_dx, D

        def thermo_fn(T, p):
            gamma = self.air.specific_heat_ratio(T, p)
            Cp = self.air.specific_cp(T, p)
            comp = self.air.equilibrium_composition(T, p / 101325)
            W = sum(comp[s] * self.air.MOLECULAR_WEIGHTS[s] for s in comp) * 1e-3
            dW_dx = 0.0
            dgamma_dx = 0.0
            return gamma, Cp, W, dW_dx, dgamma_dx

        def source_fn(x, mdot_local):
            return 0.0, 0.0

        result = self.shapiroODE.integrate(
            x_start=0.0, x_end=L_45, Ma2_in=Ma4**2, p_in=p4, T_in=T4, mdot_in=mdot,
            geometry_fn=geometry_fn, thermo_fn=thermo_fn, source_fn=source_fn,
            Cf=self.CF_DEFAULT, n_steps=200,
        )

        return {
            "Ma5": result["Ma"][-1],
            "T5": result["T"][-1],
            "p5": result["p"][-1],
            "rho5": result["rho"][-1],
            "V5": result["V"][-1],
            "Tt5": result["Tt"][-1],
            "Pt5": result["Pt"][-1],
            "A5": A5,
            "mdot": mdot,
            "solution": result,
            "thermal_choke": False,
        }

    def performance(self, inlet_props, nozzle_props, combustor3_props):
        """Compute Ia, Im, Iv, Ih per paper eqs.(33)–(37)"""
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

        mfuel = self._f(combustor3_props["mfuel"])

        # Internal thrust eq.(36)
        Fin = mdot5*V5 + p5*A5 - mdot_air*V0 - p0*A0

        # Specific thrust eq.(33) [N·s/kg]
        Ia = Fin / mdot_air

        # Mass specific impulse eq.(34) [s]
        g0 = 9.80665
        #Im = Fin / (mfuel * g0)

        # Volume specific impulse eq.(35) [kg·s/m³]
        rho_H2 = 70.8  # kg/m³ liquid H2
        #Iv = Im * rho_H2

        # Energy specific impulse eq.(37) [N·s/MJ]
        Q_H2 = 142.8e6  # J/kg
        #Ih = Fin / (mfuel * Q_H2) * 1e6  # scale to N·s/MJ

        return {
            "Fin": Fin,
            "Ia": Ia,
            #"Im": Im,
            #"Iv": Iv,
            #"Ih": Ih,
            "thermal_choke": False,
        }
    def plot_flowpath(self, inp, iso, sec2, sec3, sec4, sec5=None):
            """
            Plot flow properties through the entire engine including the isolator.
            """

            sections = []

            # Helper for concatenation with x-offset
            def add_section(sol, x_offset):
                        return {
                            "x": sol["x"] + x_offset,
                            "Ma": sol["Ma"],
                            "T": sol["T"],
                            "Tt": sol.get("Tt", sol["T"]), # Fallback to static if Tt missing
                            "p": sol["p"],
                            "pt": sol.get("pt", sol["p"]), # Fallback to static if pt missing
                            "V": sol["V"],
                            "mdot": sol["mdot"],
                        }

            # Start plotting from the beginning of the isolator
            x0 = 0.0

            # 1. Isolator Section
            s_iso = add_section(iso["solution"], x0)
            x0 = s_iso["x"][-1]

            # 2. Combustor Section 2
            s2 = add_section(sec2["solution"], x0)
            x0 = s2["x"][-1]

            # 3. Combustor Section 3
            s3 = add_section(sec3["solution"], x0)
            x0 = s3["x"][-1]

            # 4. Combustor Section 4
            s4 = add_section(sec4["solution"], x0)
            x0 = s4["x"][-1]

            sections.extend([s_iso, s2, s3, s4])

            # 5. Nozzle Section (if exists)
            if sec5 is not None and not sec4.get("thermal_choke", False):
                s5 = add_section(sec5["solution"], x0)
                sections.append(s5)

            # Concatenate all arrays
            x = np.concatenate([s["x"] for s in sections])
            Ma = np.concatenate([s["Ma"] for s in sections])
            T = np.concatenate([s["T"] for s in sections])
            Tt = np.concatenate([s["Tt"] for s in sections])
            p = np.concatenate([s["p"] for s in sections])
            pt = np.concatenate([s["pt"] for s in sections])
            V = np.concatenate([s["V"] for s in sections])
            mdot = np.concatenate([s["mdot"] for s in sections])

            # Calculate logical boundaries for vertical lines
            # Assuming L_iso is the length of the isolator section
            x_iso_end = s_iso["x"][-1]
            x_sec2_end = s2["x"][-1]
            x_sec3_end = s3["x"][-1]
            x_sec4_end = s4["x"][-1]

            fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

            # 0. Mach Number
            axs[0].plot(x, Ma, lw=2.5, color='black', label="Mach")
            axs[0].axhline(1.0, color='red', linestyle=':', alpha=0.5, label="Sonic Limit")
            axs[0].set_ylabel("Mach Number")
            axs[0].legend()

            # 1. Temperature (Static vs Total)
            axs[1].plot(x, T, lw=2, color='tab:red', label="Static T")
            axs[1].plot(x, Tt, lw=2, color='darkred', linestyle='--', label="Total T")
            axs[1].set_ylabel("Temperature [K]")
            axs[1].legend()

            # 2. Pressure (Static vs Total)
            axs[2].plot(x, p / 1e3, lw=2, color='tab:green', label="Static P")
            axs[2].plot(x, pt / 1e3, lw=2, color='darkgreen', linestyle='--', label="Total P")
            axs[2].set_ylabel("Pressure [kPa]")
            axs[2].legend()

            # 3. Velocity
            axs[3].plot(x, np.concatenate([s["V"] for s in sections]), lw=2, color='tab:blue')
            axs[3].set_ylabel("Velocity [m/s]")

            # 4. Mass Flow
            axs[4].plot(x, np.concatenate([s["mdot"] for s in sections]), lw=2, color='tab:purple')
            axs[4].set_ylabel("Mass Flow [kg/s]")
            axs[4].set_xlabel("Position in Engine [m]")

            # --- Boundaries and Labels ---
            boundaries = [sections[0]["x"][-1], sections[1]["x"][-1], sections[2]["x"][-1], sections[3]["x"][-1]]
            labels = ["Isolator", "Comb 2", "Comb 3", "Comb 4", "Nozzle"]
            
            for ax in axs:
                ax.grid(True, which='both', alpha=0.3)
                for b in boundaries:
                    ax.axvline(b, color='gray', linestyle='--', alpha=0.7)
            
            # Section Label placement
            y_lim = axs[0].get_ylim()
            for i, label in enumerate(labels[:len(sections)]):
                x_mid = (sections[i]["x"][0] + sections[i]["x"][-1]) / 2
                axs[0].text(x_mid, y_lim[1]*0.9, label, ha='center', weight='bold')

            plt.tight_layout()
            plt.show()


def print_section(title, props, fields):
    w = 34
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")
    for label, key, unit, scale in fields:
        val = props.get(key, float("nan"))
        print(f"  {label:<{w}} {val*scale:>12.4f}  {unit}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    eng = Engine()

    h_km = 30.0
    Ma0 = 6.0
    mdot = 100.0
    phi = 0.0

    print(f"\n{'═'*65}")
    print(f"  SCRAMJET PERFORMANCE ANALYSIS (H₂ fuel, φ={phi})")
    print(f"  Flight: h={h_km} km, Ma₀={Ma0}, ṁ_air={mdot} kg/s")
    print(f"{'═'*65}")

    inp = eng.inlet_properties(h=h_km*1e3, Ma=Ma0, m_air=mdot)
    iso = eng.isolator_properties(inp)
    sec2 = eng.combustor_properties2(iso)
    sec3 = eng.combustor_properties3(sec2, phi=phi)
    sec4 = eng.combustor_properties4(sec3)

    if sec4["thermal_choke"]:
        print("\n⚠ THERMAL CHOKE DETECTED in combustor!")
        print("  → Flow became sonic (Ma → 1) before combustor exit")
        print("  → Increase area ratio or reduce heat release rate")
        print(f"  → Last Ma = {sec4['Ma4']:.4f}")
    else:
        sec5 = eng.nozzle_properties(sec4, inp)
        perf = eng.performance(inp, sec5, sec3)

        print_section("Section 0 — Freestream", inp, [
            ("Mach number Ma₀", "Ma", "—", 1.0),
            ("Temperature T₀", "T0", "K", 1.0),
            ("Pressure P₀", "P0", "kPa", 1e-3),
            ("Velocity V₀", "V0", "m/s", 1.0),
        ])

        print_section("Section 1 — Isolator entrance", iso, [
            ("Mach number Ma₁", "Ma1", "—", 1.0),
            ("Temperature T₁", "T1", "K", 1.0),
            ("Pressure p₁", "p1", "kPa", 1e-3),
            ("Velocity V₁", "V1", "m/s", 1.0),
            ("Pressure recovery σc", "sigma_c", "—", 1.0),
        ])

        print_section("Section 4 — Combustor exit", sec4, [
            ("Mach number Ma₄", "Ma4", "—", 1.0),
            ("Temperature T₄", "T4", "K", 1.0),
            ("Pressure p₄", "p4", "kPa", 1e-3),
            ("Velocity V₄", "V4", "m/s", 1.0),
        ])

        print_section("Section 5 — Nozzle exit", sec5, [
            ("Mach number Ma₅", "Ma5", "—", 1.0),
            ("Temperature T₅", "T5", "K", 1.0),
            ("Pressure p₅", "p5", "kPa", 1e-3),
            ("Velocity V₅", "V5", "m/s", 1.0),
        ])

        print_section("PERFORMANCE METRICS", perf, [
            ("Internal thrust Fin", "Fin", "N", 1.0),
            ("Specific thrust Ia", "Ia", "N·s/kg", 1.0),
            ("Mass specific impulse Im", "Im", "s", 1.0),
            ("Volume specific impulse Iv", "Iv", "kg·s/m³", 1.0),
            ("Energy specific impulse Ih", "Ih", "N·s/MJ", 1.0),
        ])

        print(f"\n  Fuel: H₂,  φ={phi},  FAR={sec3['mfuel']/mdot:.5f}")
        print(f"  ṁ_fuel = {sec3['mfuel']:.6f} kg/s\n")

        print("sec2 x end:", sec2["solution"]["x"][-1])
        print("sec3 x end:", sec3["solution"]["x"][-1])
        print("sec4 x end:", sec4["solution"]["x"][-1])

        eng.plot_flowpath(inp, iso,  sec2, sec3, sec4, sec5)