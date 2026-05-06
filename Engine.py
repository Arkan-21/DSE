from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


class AirProperties:
    """Air thermodynamics using NASA CEA polynomial data and equilibrium dissociation."""

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
        "N2": 28.014,
        "O2": 31.999,
        "Ar": 39.948,
        "CO2": 44.010,
        "H2O": 18.015,
        "N": 14.007,
        "O": 15.999,
        "NO": 30.006,
    }

    def _nasa_coeffs(self, species: str, T: float) -> np.ndarray:
        data = self.NASA_DATA[species]
        return np.array(data["low"] if T <= data["Trange"][1] else data["high"])

    def cp_over_R(self, species: str, T: float) -> float:
        a = self._nasa_coeffs(species, T)
        return a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] * T**4

    def h_over_RT(self, species: str, T: float) -> float:
        a = self._nasa_coeffs(species, T)
        return (a[0]
                + a[1] * T / 2.0
                + a[2] * T**2 / 3.0
                + a[3] * T**3 / 4.0
                + a[4] * T**4 / 5.0
                + a[5] / T)

    def s_over_R(self, species: str, T: float) -> float:
        a = self._nasa_coeffs(species, T)
        return (a[0] * np.log(T)
                + a[1] * T
                + a[2] * T**2 / 2.0
                + a[3] * T**3 / 3.0
                + a[4] * T**4 / 4.0
                + a[6])

    def gibbs_over_RT(self, species: str, T: float, P_atm: float) -> float:
        return self.h_over_RT(species, T) - self.s_over_R(species, T) + np.log(P_atm)

    def equilibrium_constants(self, T: float) -> tuple[float, float, float]:
        dg1 = 2.0 * self.gibbs_over_RT("N", T, 1.0) - self.gibbs_over_RT("N2", T, 1.0)
        dg2 = 2.0 * self.gibbs_over_RT("O", T, 1.0) - self.gibbs_over_RT("O2", T, 1.0)
        dg3 = (self.gibbs_over_RT("NO", T, 1.0)
               - self.gibbs_over_RT("N", T, 1.0)
               - self.gibbs_over_RT("O", T, 1.0))
        return np.exp(-dg1), np.exp(-dg2), np.exp(-dg3)

    def equilibrium_composition(self, T: float, P_atm: float) -> dict[str, float]:
        if T < 1500.0:
            total = sum(self.AIR_BASE_COMPOSITION.values())
            return {k: v / total for k, v in self.AIR_BASE_COMPOSITION.items()}

        Kp1, Kp2, Kp3 = self.equilibrium_constants(T)

        x_N2_0 = self.AIR_BASE_COMPOSITION["N2"]
        x_O2_0 = self.AIR_BASE_COMPOSITION["O2"]
        x_Ar = self.AIR_BASE_COMPOSITION["Ar"]
        x_CO2 = self.AIR_BASE_COMPOSITION["CO2"]

        N_atoms = 2.0 * x_N2_0
        O_atoms = 2.0 * x_O2_0 + 2.0 * x_CO2

        def equilibrium_equations(variables: np.ndarray) -> list[float]:
            xN2, xO2, xN, xO, xNO = variables
            eq1 = Kp1 * xN2 - xN**2 * P_atm
            eq2 = Kp2 * xO2 - xO**2 * P_atm
            eq3 = Kp3 * xN * xO * P_atm - xNO
            eq4 = 2.0 * xN2 + xN + xNO - N_atoms
            eq5 = 2.0 * xO2 + xO + xNO - (O_atoms - 2.0 * x_CO2)
            return [eq1, eq2, eq3, eq4, eq5]

        x0 = [x_N2_0 * 0.9, x_O2_0 * 0.9, 1e-6, 1e-6, 1e-6]
        xN2, xO2, xN, xO, xNO = np.abs(fsolve(equilibrium_equations, x0))
        xT = xN2 + xO2 + xN + xO + xNO + x_Ar + x_CO2

        return {
            "N2": xN2 / xT,
            "O2": xO2 / xT,
            "N": xN / xT,
            "O": xO / xT,
            "NO": xNO / xT,
            "Ar": x_Ar / xT,
            "CO2": x_CO2 / xT,
        }

    def mixture_cp_cv(self, T: float, P_atm: float) -> tuple[float, float, float]:
        composition = self.equilibrium_composition(T, P_atm)
        molecular_weight = sum(
            composition[species] * self.MOLECULAR_WEIGHTS[species]
            for species in composition
        )

        cp_molar = sum(
            composition[species] * self.cp_over_R(species, T) * self.R_UNIVERSAL
            for species in composition
        )
        cp = cp_molar / (molecular_weight * 1e-3)
        r_spec = self.R_UNIVERSAL / (molecular_weight * 1e-3)
        cv = cp - r_spec
        gamma = cp / cv
        return cp, cv, gamma

    def specific_heat_ratio(self, T: float, P: float) -> float:
        """Return gamma for air at temperature T [K] and pressure P [Pa]."""
        if T <= 0.0:
            raise ValueError("Temperature must be positive.")
        if P <= 0.0:
            raise ValueError("Pressure must be positive.")

        P_atm = P / 101325.0
        return self.mixture_cp_cv(T, P_atm)[2]

    def plot_gamma_map(
        self,
        T_min: float = 200.0,
        T_max: float = 3000.0,
        P_min: float = 100.0,
        P_max: float = 1e7,
        n_T: int = 60,
        n_P: int = 60,
    ) -> None:
        """Plot air gamma as a function of temperature [K] and pressure [Pa]."""
        T = np.linspace(T_min, T_max, n_T)
        P = np.logspace(np.log10(P_min), np.log10(P_max), n_P)
        Tg, Pg = np.meshgrid(T, P)
        G = np.zeros_like(Tg)

        for i in range(Pg.shape[0]):
            for j in range(Pg.shape[1]):
                G[i, j] = self.specific_heat_ratio(Tg[i, j], Pg[i, j])

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(Tg, Pg, G, levels=50, cmap="viridis")
        plt.yscale("log")
        plt.colorbar(contour, label="γ")
        plt.xlabel("Temperature [K]")
        plt.ylabel("Pressure [Pa]")
        plt.title("Specific Heat Ratio γ of Air")
        plt.tight_layout()
        plt.show()

class Atmosphere:
    R_AIR = 287.05
    G0    = 9.80665
 
    @staticmethod
    def _layer(h):
        if   h <= 11000: return 0,     -0.0065, 288.15, 101325.0
        elif h <= 20000: return 11000,  0.0,    216.65,  22632.1
        elif h <= 32000: return 20000,  0.001,  216.65,   5474.89
        else: raise ValueError(f"Altitude {h:.0f} m > 32 km ceiling")
 
    @staticmethod
    def T(h):
        h0, L, T0, _ = Atmosphere._layer(h)
        return T0 + L * (h - h0)
 
    @staticmethod
    def P(h):
        h0, L, T0, P0 = Atmosphere._layer(h)
        dh = h - h0; T = T0 + L * dh
        if L != 0:
            return P0 * (T / T0) ** (-Atmosphere.G0 / (L * Atmosphere.R_AIR))
        return P0 * np.exp(-Atmosphere.G0 * dh / (Atmosphere.R_AIR * T0))
 
    @staticmethod
    def rho(h): return Atmosphere.P(h) / (Atmosphere.R_AIR * Atmosphere.T(h))
 
    @staticmethod
    def a(h): return np.sqrt(1.4 * Atmosphere.R_AIR * Atmosphere.T(h))
 
class Engine:
    def __init__(self):
        self.air = AirProperties()

    def specific_heat_ratio(self, T, P):
        return self.air.specific_heat_ratio(T, P)
# =========
if __name__ == "__main__":
    air = AirProperties()
    print("Gamma at 300 K, 1 atm:", air.specific_heat_ratio(300.0, 101325.0))
    print("Gamma at 2500 K, 1 atm:", air.specific_heat_ratio(2500.0, 101325.0))
    air.plot_gamma_map()