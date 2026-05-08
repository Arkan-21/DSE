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
        "O2": 31.998,   # corrected: 2 * 15.999
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
        x_Ar   = self.AIR_BASE_COMPOSITION["Ar"]
        x_CO2  = self.AIR_BASE_COMPOSITION["CO2"]

        # CO2 oxygen is locked in CO2 (not dissociated), so only track free O atoms
        N_atoms = 2.0 * x_N2_0
        O_atoms = 2.0 * x_O2_0  # FIX: do not double-count CO2 oxygen

        def equilibrium_equations(variables: np.ndarray) -> list[float]:
            xN2, xO2, xN, xO, xNO = variables
            eq1 = Kp1 * xN2 - xN**2 * P_atm
            eq2 = Kp2 * xO2 - xO**2 * P_atm
            eq3 = Kp3 * xN * xO * P_atm - xNO
            eq4 = 2.0 * xN2 + xN + xNO - N_atoms
            eq5 = 2.0 * xO2 + xO + xNO - O_atoms  # FIX: use corrected O_atoms
            return [eq1, eq2, eq3, eq4, eq5]

        x0 = [x_N2_0 * 0.9, x_O2_0 * 0.9, 1e-6, 1e-6, 1e-6]
        xN2, xO2, xN, xO, xNO = np.abs(fsolve(equilibrium_equations, x0))
        xT = xN2 + xO2 + xN + xO + xNO + x_Ar + x_CO2

        return {
            "N2":  xN2  / xT,
            "O2":  xO2  / xT,
            "N":   xN   / xT,
            "O":   xO   / xT,
            "NO":  xNO  / xT,
            "Ar":  x_Ar / xT,
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
        cp      = cp_molar / (molecular_weight * 1e-3)
        r_spec  = self.R_UNIVERSAL / (molecular_weight * 1e-3)
        cv      = cp - r_spec
        gamma   = cp / cv
        return cp, cv, gamma

    def specific_heat_ratio(self, T: float, P: float) -> float:
        """Return gamma for air at temperature T [K] and pressure P [Pa]."""
        if T <= 0.0:
            raise ValueError("Temperature must be positive.")
        if P <= 0.0:
            raise ValueError("Pressure must be positive.")
        P_atm = P / 101325.0
        return self.mixture_cp_cv(T, P_atm)[2]

    def specific_cp(self, T: float, P: float) -> float:
        """Return cp [J/(kg·K)] for air at temperature T [K] and pressure P [Pa]."""
        if T <= 0.0:
            raise ValueError("Temperature must be positive.")
        if P <= 0.0:
            raise ValueError("Pressure must be positive.")
        P_atm = P / 101325.0
        return self.mixture_cp_cv(T, P_atm)[0]

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
    R_AIR = 287.05   # J/(kg·K)
    G0    = 9.80665  # m/s²

    @staticmethod
    def _layer(h: float) -> tuple[float, float, float, float]:
        """Return (h0, lapse_rate, T0, P0) for the ISA layer containing h [m]."""
        if   h <= 11000: return 0.0,   -0.0065, 288.15, 101325.0
        elif h <= 20000: return 11000,  0.0,    216.65,  22632.1
        elif h <= 32000: return 20000,  0.001,  216.65,   5474.89
        else: raise ValueError(f"Altitude {h:.0f} m exceeds 32 km model ceiling.")

    @staticmethod
    def T(h: float) -> float:
        """Static temperature [K] at geometric altitude h [m]."""
        h0, L, T0, _ = Atmosphere._layer(h)
        return T0 + L * (h - h0)

    @staticmethod
    def P(h: float) -> float:
        """Static pressure [Pa] at geometric altitude h [m]."""
        h0, L, T0, P0 = Atmosphere._layer(h)
        dh = h - h0
        T  = T0 + L * dh
        if L != 0.0:
            return P0 * (T / T0) ** (-Atmosphere.G0 / (L * Atmosphere.R_AIR))
        return P0 * np.exp(-Atmosphere.G0 * dh / (Atmosphere.R_AIR * T0))

    @staticmethod
    def rho(h: float) -> float:
        """Air density [kg/m³] at altitude h [m]."""
        return Atmosphere.P(h) / (Atmosphere.R_AIR * Atmosphere.T(h))

    @staticmethod
    def a(h: float, gamma: float = 1.4) -> float:
        """Speed of sound [m/s] at altitude h [m].

        Parameters
        ----------
        h : float
            Geometric altitude [m].
        gamma : float, optional
        """
        gamma = AirProperties.specific_heat_ratio(Atmosphere.T(h), Atmosphere.P(h))
        return np.sqrt(gamma * Atmosphere.R_AIR * Atmosphere.T(h))


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
    # -------------------------------------------------------------------------
    # Scramjet geometry (Table 1 of Li et al. 2023)
    # -------------------------------------------------------------------------
    L12 = 0.40   # isolator length [m]
    L23 = 0.01   # fuel-injection section length [m]
    L34 = 1.00   # mixing/combustion section length [m]
    L45 = 0.40   # nozzle length [m]

    # Area ratios relative to section 1 (inlet exit = combustor entrance)
    alpha12 = 1.0   # A2/A1
    alpha13 = 1.1   # A3/A1
    alpha14 = 2.5   # A4/A1
    alpha05 = 2.0   # A5/A0  (nozzle exit / freestream capture)

    # Inlet compression parameters (paper §2.2)
    EPSILON   = 0.4    # Mach-number ratio Ma1/Ma0
    ETA_C     = 0.9    # adiabatic compression efficiency ηc

    # Default wall friction coefficient
    CF_DEFAULT = 0.003

    def __init__(self):
        self.air = AirProperties()

    def inlet_properties(
        self,
        h: float,
        M: float,
        m_air: float,
    ) -> dict:
        """Compute freestream and total inlet conditions.

        Parameters
        ----------
        h : float
            Flight altitude [m].
        M : float
            Flight Mach number.
        m_air : float
            Air mass-flow rate [kg/s].

        Returns
        -------
        dict with keys:
            T0, P0, rho0       – freestream static conditions
            gamma0, cp0        – real-gas thermo at freestream
            a0, V              – speed of sound and flight velocity [m/s]
            A0                 – freestream capture area [m²]
            Tt0, Pt0           – total (stagnation) temperature [K] and pressure [Pa]
        """
        # --- Freestream static conditions from ISA ---
        T0   = Atmosphere.T(h)
        P0   = Atmosphere.P(h)
        rho0 = Atmosphere.rho(h)

        # --- Real-gas thermo at freestream ---
        gamma0 = self.air.specific_heat_ratio(T0, P0)   # FIX: compute before a0
        cp0    = self.air.specific_cp(T0, P0)           # J/(kg·K)

        # --- Kinematics ---
        a0 = Atmosphere.a(h, gamma0)                    # FIX: pass gamma to staticmethod
        V  = M * a0

        # --- Freestream capture area ---
        A0 = m_air / (rho0 * V)

        # --- Stagnation conditions (calorically imperfect, but frozen chemistry) ---
        # Tt0 from energy: cp(T0)*T0 + 0.5*V² = cp(T0)*Tt0  (first-order approximation)
        Tt0 = T0 + 0.5 * V**2 / cp0
        # Isentropic total pressure using freestream gamma
        Pt0 = P0 * (Tt0 / T0) ** (gamma0 / (gamma0 - 1.0))

        return {
            "T0":    T0,
            "P0":    P0,
            "rho0":  rho0,
            "gamma0": gamma0,
            "cp0":   cp0,
            "a0":    a0,
            "V":     V,
            "A0":    A0,
            "Tt0":   Tt0,
            "Pt0":   Pt0,
        }
    
    def compression_efficiency(self, n: int) -> float:
        if n==4:
            return 0.92
        elif n==3:
            return 0.88
        elif n==2:
            return 0.82
        elif n==1:
            return 0.72


    def pressure_recovery(Ma: float) -> float:
        """
        Compute pressure recovery factor from inlet to isolator exit
        using a polynomial fit of experimental data.
        """

        MaList = np.array([
            8.126582278481013, 7.640506792672073, 7.245569156695016,
            6.8658223212519776, 6.6075949367088604, 6.349367552165743,
            6.136709092538568, 5.954429916188687, 5.75696225709553,
            5.605063522918315, 5.4531647887411, 5.2860764129252376,
            5.1645569620253164, 5.027847869486749
        ])

        pressure_recovery_coef = np.array([
            0.3021505460144819, 0.31827957959571584, 0.333870966418766,
            0.3505376467581838, 0.36344086131769493, 0.3774193693935738,
            0.38870968469678685, 0.3999999897454365, 0.4123655985650173,
            0.42311828761917336, 0.4338709766733293, 0.4451613022311057,
            0.4543010920289637, 0.46612904383579723
        ])

        # Polynomial fit to the data (linear fit for simplicity; can be improved with higher-order polynomials)
        coeffs = np.polyfit(MaList, pressure_recovery_coef, 1)
        poly = np.poly1d(coeffs)

        return float(poly(Ma))
    
    def isolator_properties(self, inlet_props: dict) -> dict:
        
        Ma0 = inlet_props["Ma"]
        T0  = inlet_props["T0"]
        P0  = inlet_props["P0"]
        V0  = inlet_props["V0"]
        Pt0 = inlet_props["Pt0"]
        gamma0 = inlet_props["gamma0"]
        R = 287.05  # J/kg·K for air

        # Step 1: Ma1 from fixed epsilon (eq. 10, justified by T1 < 1560K limit)
        epsilon = 0.4
        M1 = epsilon * Ma0

        # Step 2: Iteratively solve T1 from energy conservation (eq. 13)
        Cp0 = gamma0 * R / (gamma0 - 1)
        Ht0 = Cp0 * T0 + 0.5 * V0**2  # total enthalpy, conserved (adiabatic wall)

        def energy_residual(T1_guess):
            gamma1 = self.air.specific_heat_ratio(T1_guess, P0)
            Cp1 = gamma1 * R / (gamma1 - 1)
            V1 = M1 * np.sqrt(gamma1 * R * T1_guess)
            return Ht0 - (Cp1 * T1_guess + 0.5 * V1**2)

        T1 = fsolve(energy_residual, x0=1200.0)[0]

        # Step 3: Derived quantities at section 1
        gamma1 = self.air.specific_heat_ratio(T1, P0)
        Cp1    = gamma1 * R / (gamma1 - 1)
        V1     = M1 * np.sqrt(gamma1 * R * T1)
        Tt1    = T0 * (1 + (gamma0 - 1) / 2 * Ma0**2) * Cp0 / Cp1  # total temp

        # Step 4: Pressure recovery → p1, pt1
        sigma_c = self.pressure_recovery(Ma0)   # ηc = 0.9, 4 shocks
        pt1 = sigma_c * Pt0
        p1  = pt1 * (T1 / Tt1) ** (gamma1 / (gamma1 - 1))

        # Step 5: Density and area from mass conservation (eqs. 12, 14)
        rho1 = p1 / (R * T1)
        A1   = inlet_props["mdot"] / (rho1 * V1)

        return {
            "Ma": M1, "T1": T1, "Tt1": Tt1,
            "p1": p1, "pt1": pt1, "V1": V1,
            "A1": A1, "rho1": rho1, "gamma1": gamma1
        }

# =========
if __name__ == "__main__":
    air = AirProperties()
    print("Gamma at 300 K, 1 atm :", air.specific_heat_ratio(300.0,  101325.0))
    print("Gamma at 2500 K, 1 atm:", air.specific_heat_ratio(2500.0, 101325.0))

    eng = Engine()

    # Example: 10 km altitude, Mach 0.8, 100 kg/s air mass flow
    props = eng.inlet_properties(h=10000.0, M=0.8, m_air=100.0)
    print("\n--- Inlet conditions at h=10 km, M=0.8, ṁ=100 kg/s ---")
    for k, v in props.items():
        print(f"  {k:8s} = {v:.4g}")

    air.plot_gamma_map()