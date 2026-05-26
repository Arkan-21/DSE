import numpy as np

n_pax = 32

tau = 0.12

#Aircraft Weight:
TOGW = 96119.406                    #kg    
W_fuel_LH2_ramjet = 26208.436       #kg
W_fuel_LH2_turbojet = 11684.005     #kg
W_Payload = 3840                    #kg
W_str = 22959.661                   #kg         
W_tps = 6559.903                    #kg     
W_landinggear = 3990.587            #kg
W_prop = 17053.504                  #kg
W_sub = 3847.590                    #kg
W_payload = 3840.000                #kg

#Aircraft Volume:
V_tot = 982.167                     #m^3
V_fuel_LH2_ramjet = 374.334         #m^3
V_fuel_LH2_turbojet = 167.655       #m^3  
V_structure = 8.504                 #m^3
V_tps = 2.187                       #m^3    
V_landinggear = 9.824               #m^3
V_prop = 166.022                    #m^3
V_subsystem = 19.647                #m^3
V_void = 196.470                    #m^3
V_payload = 38.400                  #m^3
V_tank_capacity = 541.989           #m^3

#Aircraft Geometry:
S_plan = 405.944                    #m^2
S_wetted = 1092.801                 #m^2



Interior Dimensions:
Seat_pitch = 1                      #m
Seat_width = 0.5                    #m
Seat_height = 1.65                  #m
N_seats_per_row	= 8         
Aisle_width	= 0.5                   #m
Aisle_height = 2.3                  #m  
N_aisles = 2
Cockpit_length	= 2.5               #m
fuselage_width = 7                  #m
fuselage_height = 3.5               #m
volume_cargo_per_passenger = 0.13   #m^3
Additioanal_cargo_volume = 11.83    #m^3