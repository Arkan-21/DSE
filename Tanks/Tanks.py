import numpy as np
import matplotlib.pyplot as plt
import coolprop as cp

class TankThermodynamics:
    def __init__(self, initial_volume, initial_pressure, initial_temperature, m_tank, fill_level=0.97):
        # Geometrical and Mass parameters
        self.volume = initial_volume      # Total tank volume (V_tot = V_g + V_l) [m^3]
        self.m_tank = m_tank              # Structural tank weight [kg]
        self.fill_level = fill_level      # Initial liquid volume fraction
        
        # State variables
        self.pressure = initial_pressure  # Ullage pressure Pg [Pa]
        self.T_g = initial_temperature    # Gas phase temperature [K]
        
        # Fluid configuration (Para-Hydrogen is typical for cryogenic storage)
        self.fluid = "Parahydrogen"
        
        # Calculate saturation state at initial pressure
        self.T_sat = cp.CoolProp.PropsSI("T", "P", self.pressure, "Q", 0, self.fluid)
        self.T_l = min(initial_temperature, self.T_sat) # Liquid temperature constrained to <= T_sat
        
        # Calculate Phase Densities
        rho_l = cp.CoolProp.PropsSI("D", "P", self.pressure, "T", self.T_l, self.fluid)
        rho_g = cp.CoolProp.PropsSI("D", "P", self.pressure, "T", self.T_g, self.fluid)
        
        # Volumes and Masses
        self.V_l = self.volume * self.fill_level
        self.V_g = self.volume * (1.0 - self.fill_level)
        self.m_l = self.V_l * rho_l
        self.m_H2 = self.m_l + (self.V_g * rho_g)
        self.m_g = self.V_g * rho_g

    def gravimetric_efficiency(self):
        """Computes standard gravimetric efficiency (Eq. 1)"""
        return self.m_H2 / (self.m_H2 + self.m_tank)

    def gravimetric_efficiency_adjusted(self, m_boiloff):
        """Computes boil-off adjusted gravimetric efficiency (Eq. 26)"""
        return (self.m_H2 - m_boiloff) / (self.m_H2 + self.m_tank)

    def T_skin(self, T_inf, T_dp):
        """Computes external sky temperature approximation profile (Eq. 10)"""
        return T_inf * (0.711 + 0.0056 * T_dp + 0.0073 * T_dp**2 + 0.0013)**0.25
        
    def T_recovery(self, T_inf, Pr, gamma, M):
        """Computes boundary layer aero-thermal recovery temperature (Eq. 3)"""
        return T_inf * (1.0 + 0.5 * Pr**0.5 * (gamma - 1.0) * M**2)
        
    def heat_conv(self, Nu, k, L):
        """Convective heat transfer coefficient (h = Nu * k / L) (Eq. 4, Eq. 13)"""
        return Nu * k / L
    
    def Nu_natural_air(self, Ra):
        """Nusselt equation for natural external air convection (Eq. 6)"""
        return 0.555 * Ra**0.25 + 0.447
    
    def Nu_LH2(self, Ra):
        """Nusselt equation for natural convective liquid hydrogen wall (Eq. 14)"""
        return 0.0605 * Ra**(1.0 / 3.0)
    
    def Nu_GH2(self):
        """Nusselt setting for natural gaseous hydrogen wall (Eq. 15)"""
        return 17.0
    
    def Ra(self, g, beta, dT, L, Pr, nu):
        """Rayleigh number formulation (Eq. 7)"""
        return (g * beta * abs(dT) * L**3 * Pr) / nu

    def Conv_heatflow(self, T_skin, T_recovery, A_ext, h_conv):
        """External convective heat exchange (Eq. 2)"""
        return (T_skin - T_recovery) * A_ext * h_conv
    
    def Radiative_heatflow(self, T_skin, T_sky, A_ext, sigma, epsilon):
        """External radiative heat transfer towards the sky (Eq. 8)"""
        return (T_skin**4 - T_sky**4) * A_ext * epsilon * sigma
    
    def Solar_heatflow(self, q_solar, A_ext_solar, alpha):
        """Solar irradiation tracking (Eq. 11)"""
        return q_solar * A_ext_solar * alpha

    def thermal_circuit_solver(self, T_inf, T_recovery, T_sky, q_solar, A_ext, A_ext_solar, alpha, epsilon, sigma, h_ext_conv, h_cond_insulation, k_H2, L_int, is_liquid=True):
        """
        Solves the steady state system (Eq. 16 / Eq. 22) for a given surface section.
        Enforces: 
           Q_ext,conv + Q_rad - Q_solar - Q_cond = 0
           Q_int,conv [+/- Q_boil] - Q_cond = 0
        Returns (Q_to_fluid, T_wall, T_skin)
        """
        # Determine internal fluid bounds reference
        T_fluid = self.T_l if is_liquid else self.T_g
        Nu_int = self.Nu_LH2(1e6) if is_liquid else self.Nu_GH2() # Reference Ra placeholder or constant
        h_int_conv = self.heat_conv(Nu_int, k_H2, L_int)

        # Numerical iterative solver to match steady-state skin & wall temperatures
        T_skin_guess = T_inf
        T_wall_guess = T_fluid + 2.0
        
        learning_rate = 0.01
        for _ in range(200):
            # 1. External thermal streams
            Q_ext_conv = self.Conv_heatflow(T_skin_guess, T_recovery, A_ext, h_ext_conv)
            Q_rad = self.Radiative_heatflow(T_skin_guess, T_sky, A_ext, sigma, epsilon)
            Q_solar = self.Solar_heatflow(q_solar, A_ext_solar, alpha)
            
            # Conduction down the foam core layer
            Q_cond = (T_wall_guess - T_skin_guess) * A_ext * h_cond_insulation
            
            # 2. Internal convective transport
            Q_int_conv = (T_fluid - T_wall_guess) * A_ext * h_int_conv
            
            # Residual checking 
            res_ext = Q_ext_conv + Q_rad - Q_solar - Q_cond
            res_int = Q_int_conv - Q_cond # base Eq. 16 formulation
            
            # Update guesses
            T_skin_guess -= learning_rate * res_ext
            T_wall_guess += learning_rate * res_int
            
            if abs(res_ext) < 1e-2 and abs(res_int) < 1e-2:
                break
                
        Q_to_fluid = -Q_int_conv # Enters into fluid control volume
        return Q_to_fluid, T_wall_guess, T_skin_guess

    def thermodynamic_sys(self, Q_eg, Q_el, m_dot_f, m_dot_press = 0.0, dt = 10):
        """
        Executes State updates derived over the 3-control volume approach (Eq. 18).
        Inputs:
            Q_eg       : Summed incoming external thermal streams to Ullage [W]
            Q_el       : Summed incoming external thermal streams to Bulk Liquid [W]
            m_dot_f    : Fuel pump mass flow demand extracted out by engines [kg/s]
            m_dot_press: Active forced vaporizer feed loop [kg/s]
            dt         : Sizing timestep resolution [s]
        """
        # Fetch thermodynamic properties from CoolProp fluid base
        h_g = cp.CoolProp.PropsSI("H", "P", self.pressure, "T", self.T_g, self.fluid)
        u_g = cp.CoolProp.PropsSI("U", "P", self.pressure, "T", self.T_g, self.fluid)
        c_vg = cp.CoolProp.PropsSI("Cvmass", "P", self.pressure, "T", self.T_g, self.fluid)
        
        h_l = cp.CoolProp.PropsSI("H", "P", self.pressure, "T", self.T_l, self.fluid)
        u_l = cp.CoolProp.PropsSI("U", "P", self.pressure, "T", self.T_l, self.fluid)
        c_pl = cp.CoolProp.PropsSI("Cpmass", "P", self.pressure, "T", self.T_l, self.fluid)
        
        h_sat_g = cp.CoolProp.PropsSI("H", "P", self.pressure, "Q", 1, self.fluid)
        h_sat_l = cp.CoolProp.PropsSI("H", "P", self.pressure, "Q", 0, self.fluid)
        h_vap = h_sat_g - h_sat_l
        
        # Interface dynamics parameters (Aviation boundary layout: Q_gs >= Q_sl constraint)
        Q_gs = 0.1 * Q_eg  # Empirical scaling of interface interaction from ullage heat
        Q_sl = min(0.05 * Q_el, Q_gs) 
        
        # Standard wall boil off tracking condition if bulk liquid tracks superheated saturation limit
        m_dot_boil = 0.0
        if self.T_l >= self.T_sat:
            self.T_l = self.T_sat
            m_dot_boil = max(0.0, Q_el / h_vap)
            
        # 1. Resolve Mass Change Derivatives (Eq. 18)
        m_dot_g = ((Q_gs - Q_sl) / (h_vap + c_pl*(self.T_sat - self.T_l) + (h_g - h_sat_g))) + m_dot_boil + m_dot_press
        m_dot_l = - m_dot_g - m_dot_f
        
        # Volumetric evaluation updates
        rho_l = cp.CoolProp.PropsSI("D", "P", self.pressure, "T", self.T_l, self.fluid)
        V_l_next = max(0.0, (self.m_l + m_dot_l * dt) / rho_l)
        V_g_next = max(0.0, self.volume - V_l_next)
        V_dot_g = (V_g_next - self.V_g) / dt
        V_dot_l = - V_dot_g
        
        # 2. Temperature Derivatives Evaluation (Eq. 18)
        dT_g_dt = (Q_eg - Q_gs - self.pressure*V_dot_g + m_dot_g*h_g + h_l*m_dot_boil - m_dot_g*u_g) / (self.m_g * c_vg)
        dT_l_dt = (Q_el + Q_sl + self.pressure*V_dot_l - m_dot_g*h_l + h_l*m_dot_f - m_dot_l*u_l) / (self.m_l * c_pl)
        
        # Integration step execution
        self.m_g += m_dot_g * dt
        self.m_l += m_dot_l * dt
        self.T_g += dT_g_dt * dt
        self.T_l += dT_l_dt * dt
        self.V_g = V_g_next
        self.V_l = V_l_next
        self.m_H2 = self.m_g + self.m_l
        
        # Re-evaluate internal ideal gas pressure step propagation
        # Pg * Vg = mg * R * Tg
        R_H2 = cp.CoolProp.PropsSI("GAS_CONSTANT", self.fluid) / cp.CoolProp.PropsSI("MOLAR_MASS", self.fluid)
        self.pressure = (self.m_g * R_H2 * self.T_g) / self.V_g
        self.T_sat = cp.CoolProp.PropsSI("T", "P", self.pressure, "Q", 0, self.fluid)

    def handle_instantaneous_venting(self, P_max_limit, P_min_target):
        """Executes instant venting tracking dynamics criteria (Eq. 23)"""
        if self.pressure >= P_max_limit:
            m_g_old = self.m_g
            # Reset constraints 
            self.pressure = P_min_target
            self.T_g = cp.CoolProp.PropsSI("T", "P", self.pressure, "Q", 1, self.fluid) # drops to saturation
            
            rho_g_new = cp.CoolProp.PropsSI("D", "P", self.pressure, "T", self.T_g, self.fluid)
            self.m_g = self.V_g * rho_g_new
            m_vented = max(0.0, m_g_old - self.m_g)
            return m_vented
        return 0.0