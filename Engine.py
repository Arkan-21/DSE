import numpy as np
import math
import cea

class Equations:

    def __init__(self, engine):
        self.engine = engine

    def mass_flow_rate(self, h, M):
        return Area*self.engine.rho(h)*M*self.engine.a(h)


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
        return P0 * math.exp(-Atmosphere.G0 * dh / (Atmosphere.R_AIR * T0))
 
    @staticmethod
    def rho(h): return Atmosphere.P(h) / (Atmosphere.R_AIR * Atmosphere.T(h))
 
    @staticmethod
    def a(h): return math.sqrt(1.4 * Atmosphere.R_AIR * Atmosphere.T(h))

    def specific_heat_ratio(self, h):
        specific_heat_ratio = cea.
        return 
    

from cea_wrap import CEA

def gamma_from_TP(T, P, fuel, oxidizer, phi=1.0):
    """
    Compute gamma (Cp/Cv) from NASA CEA at specified Temperature and Pressure.

    Parameters:
        T (float): Temperature [K]
        P (float): Pressure [bar]
        fuel (str): Fuel name (CEA format)
        oxidizer (str): Oxidizer name (CEA format)
        phi (float): Equivalence ratio (default = 1)

    Returns:
        gamma (float): Specific heat ratio Cp/Cv
    """

    cea = CEA()

    # Define a TP problem (this is the key part)
    problem = cea.tp(
        temperature=T,
        pressure=P,
        fuel=fuel,
        oxidizer=oxidizer,
        phi=phi
    )

    result = problem.run()

    return result["gamma"]   # "real" gamma
 