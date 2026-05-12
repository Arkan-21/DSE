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
        assert abs(cp - cv - self.R_UNIVERSAL / (MW*1e-3)) < 1e-6, "Inconsistent cp, cv, R values."


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

