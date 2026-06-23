import os
import time
from unittest import result

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openvsp as vsp
import scipy.optimize as opt
from bayes_opt import BayesianOptimization, acquisition
from fairing_calculator import discretize_nose, discretize_tail
from stl_newtonian_spanwise import load_geometry


import importlib.util as _ilu
_hf_spec = _ilu.spec_from_file_location(
    "hypersonic_aero",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hypersonic Flow Noise.py"),
)
_hf = _ilu.module_from_spec(_hf_spec)
_hf_spec.loader.exec_module(_hf)
from understand_stl import exclude_fuselage_faces, mesh_volume


MACH = 4.349
ALPHA = 3.0
WEIGHT = 96119.4 *9.81 *0.8
SPEED_OF_SOUND = 302.7
VOLUME = 982.17 # m^3


RHO = 0.01503 * 1.225  # kg/m^3, density at 50,000 ft (ISA) 
S_REF = 829.5  
#CL_MIN=WEIGHT / (0.5 * RHO * (MACH*SPEED_OF_SOUND)**2 * S_REF)  # target lift coefficient for level flight at 50,000 ft#
CL_MIN = 0.02
print(f"Target CL for level flight at 50,000 ft: {CL_MIN:.4f}")

TOTAL_LENGTH = 60.5  # m, total length of the fuselage

MAX_ITER = 20
DEBUG_MODE = False
FILE_NAME = "optimisation_history_3aoa_naca4_lowcl"
# Single source of truth: param name → initial value.
# Order defines the layout of the flat x array the optimizer passes around.

BASELINE_PARAMS = {
    "con_chord": 47.85/TOTAL_LENGTH,
    "chord0": 39.14/47.85, # parametrized by con_chord
    "chord1": 18.7/39.14,  # parametrized by chord0
    "chord2": 11.27/18.7,  # parametrized by chord1
    "chord3": 3.50/11.27,  # parametrized by chord2
    "span1":   6.00, "sweep1": 73.55/90, "dihedral1":  1.0/90, # angles are parametrized as a fraction of 90
    "span2":   3.33, "sweep2": 64.84/90, "dihedral2":  1.0/90,
    "span3":   3.29, "sweep3": 70.50/90, "dihedral3": -20.0/90,
    "wing_x_location": 9.0/TOTAL_LENGTH,  # as fraction of total length
    "camber": 0.001, "thickness": 0.04, "max_camber_loc": 0.4,
    "con_sweep": 68/90, "con_span": 3.5,
}

BASELINE_PARAMS_SENSITIVITY= {
    "con_chord": 6.0/TOTAL_LENGTH,
    "chord0": 5.0/6.0, 
    "chord1": 4.0/5.0, 
    "chord2": 3.0/4.0, 
    "chord3": 2.0/3.0,
    "span1":   4.125, "sweep1": 30.0/90, "dihedral1":  1.0/90,
    "span2":   4.125, "sweep2": 30.0/90, "dihedral2":  1.0/90,
    "span3":   4.125, "sweep3": 30.0/90, "dihedral3": 2.0/90,
    "wing_x_location": 30.0/TOTAL_LENGTH,  # as fraction of total length
    "camber": 0.001, "thickness": 0.04, "max_camber_loc": 0.4,
    "con_sweep": 30.0/90, "con_span": 4.125,
}
BOUNDS = (
    # chords (absolute 0→1, normalized by baseline fraction of total length)
    (0.001, 0.8/BASELINE_PARAMS["con_chord"]),      # Connection chord (con_chord) should not be heigher than 80% of the total length
    (0.001, 1.0/BASELINE_PARAMS["chord0"]),         # chord0 should be biger or equal to the connection chord
    (0.001, 1.0/BASELINE_PARAMS["chord1"]),         # chord1 should be biger or equal to chord0
    (0.001, 1.0/BASELINE_PARAMS["chord2"]),         # chord2 should be biger or equal to chord1
    (0.001, 1.0/BASELINE_PARAMS["chord3"]),         # chord3 should be biger or equal to chord2
    # span1/sweep1/dihedral1 (spans unconstrained above; sweeps 0→85°; dihedral ±90°)
    (0.0, None), (0.0, (85/90)/BASELINE_PARAMS["sweep1"]), (-1.0/BASELINE_PARAMS["dihedral1"], 1.0/BASELINE_PARAMS["dihedral1"]),
    # span2/sweep2/dihedral2
    (0.0, None), (0.0, (85/90)/BASELINE_PARAMS["sweep2"]), (-1.0/BASELINE_PARAMS["dihedral2"], 1.0/BASELINE_PARAMS["dihedral2"]),
    # span3/sweep3/dihedral3 (dihedral3 baseline is -20°, so range [-90°,+90°] → [-4.5, 4.5])
    (0.0, None), (0.0, (85/90)/BASELINE_PARAMS["sweep3"]), (1.0/BASELINE_PARAMS["dihedral3"], -1.0/BASELINE_PARAMS["dihedral3"]),
    # wing_x_location (0→50% of body length)
    (0.0, 0.5/BASELINE_PARAMS["wing_x_location"]),
    # thickness [2%, 20%]]
    #(0.02/0.02, 0.20/0.02), 
    #camber [0%, 40%], max_camber_loc [0, 100% chord]
    (0.0, 0.40/BASELINE_PARAMS["camber"]), (0.0, 1.0/BASELINE_PARAMS["max_camber_loc"]),
    # con_sweep (0→85°), con_span (unconstrained)
    (0.0, (85/90)/BASELINE_PARAMS["con_sweep"]), (0.0, None),
)


OPTIMIZE_KEYS = [
    "con_chord", "chord0", "chord1", "chord2", "chord3",
    "span1", "sweep1", "dihedral1",
    "span2", "sweep2", "dihedral2",
    "span3", "sweep3", "dihedral3",
    "wing_x_location", 
    "camber", "max_camber_loc",
    "con_sweep", "con_span",
]
RELATIVE_STEPS = {
    "chord0": 0.01, "chord1": 0.01, "chord2": 0.01, "chord3": 0.01,
    "span1": 0.01, "sweep1": 0.01, "dihedral1": 0.01,
    "span2": 0.01, "sweep2": 0.01, "dihedral2": 0.01,
    "span3": 0.01, "sweep3": 0.01, "dihedral3": 0.01,
    "wing_x_location": 0.1,
    "camber": 0.01, "thickness": 0.01, "max_camber_loc": 0.01,
    "con_chord": 0.01, "con_sweep": 0.01, "con_span": 0.01,
}


# ----------------------------------------------------------------------
# Geometry generation
# ----------------------------------------------------------------------
def apply_station_skinning(xsec_id, target_angle, target_curvature):
    # 1. List of feature lines we want to modify
    features = ["Top", "Bottom", "Left", "Right"]
    
    for feat in features:
        # --- ACTIVATE C2 CONTINUITY ---
        # Set continuity to 2.0 (C2)
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, f"Continuity{feat}"), 2.0)
        
        # --- SET SLOPES (ANGLES) ---
        # Enable manual control for Angle (Left/Before and Right/After columns)
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, f"{feat}LAngleSet"), 1.0)
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, f"{feat}RAngleSet"), 1.0)
        # Assign your slope values
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, f"{feat}LAngle"), target_angle)
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, f"{feat}RAngle"), target_angle)
        
        # --- SET CURVATURES ---
        # Enable manual control for Curvature (Left/Before and Right/After columns)
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, f"{feat}LCurveSet"), 1.0)
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, f"{feat}RCurveSet"), 1.0)
        # Assign your curvature values
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, f"{feat}LCurve"), target_curvature)
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, f"{feat}RCurve"), target_curvature)


def generate_geometry(
        chord0, chord1, chord2, chord3,
        span1, sweep1, dihedral1,
        span2, sweep2, dihedral2,
        span3, sweep3, dihedral3,
        camber, thickness, max_camber_loc,
        wing_x_location,  # as fraction of total length
        con_chord, con_sweep, con_span,
        functional_length=43,
        tail_fairing=7.0,
        cabin_width=4.0, cabin_height=4.0,
        sec_num=30, 
        output_vsp3="supercool_VFFP.vsp3",
        output_stl=None,
        verbose=False
):
    # This function generates the geometry of the wing and fuselage using OpenVSP.
    # It defines the parameters for the wing and fuselage, creates the geometries,
    # and saves the model to a VSP file.

    # check how many non-default arguments have been passed in
    import inspect
    sig = inspect.signature(generate_geometry)
    # map of params that have defaults
    defaults = {name: param.default for name, param in sig.parameters.items() if param.default is not inspect._empty}
    # count how many of those parameters differ from their default value
    nondefault_count = 0
    for name, default_val in defaults.items():
        if name in locals(  ):
            try:
                if locals()[name] != default_val:
                    nondefault_count += 1
            except Exception:
                # if comparison fails, assume it's non-default
                nondefault_count += 1
    max_deg_freedom = len(defaults) - 7 # subtract the numbers of fixed prameters
    if verbose:
        print(f"{nondefault_count} degrees of freedom used (out of {max_deg_freedom} possible).")

        #print spanwise sweep distribution for the wing
        print("\nWing spanwise sweep distribution:")
        print(f"Span 1: {span1} m, Sweep 1: {sweep1} deg")
        print(f"Span 2: {span1+span2} m, Sweep 2: {sweep2} deg")
        print(f"Span 3: {span1+span2+span3} m, Sweep 3: {sweep3} deg")
    
    # Insert nose stations
    x,_,y_sec, slopes, curvatures = discretize_nose(num_points=15,R=cabin_width/2)
    slopes = -slopes  # invert slopes to match the correct direction for skinning
    # revert the order in slopes and curvatures to start from the nose tip and move towards the fuselage, for easier station insertion in VSP
    slopes = slopes[::-1]
    curvatures = curvatures[::-1]
    combined_nose_length = x[-1]
    total_length = combined_nose_length + functional_length + tail_fairing

    # revers x to start at the nose tip and increase towards the fuselage, for easier station insertion in VSP
    y_sec = [max(2*y_sec[i], 0.0) for i in range(len(y_sec))]  # convert from radius to diameter for the cross-section widths
    y_sec = y_sec[::-1]
    
    # Insert tail fairing stations
    x_tail, radius_tail, slopes_tail, curvatures_tail = discretize_tail(R=cabin_width/2, L=total_length, tail_start=1-(tail_fairing/total_length), num_points=10)
    y_sec_tail = [max(2*r, 0.0) for r in radius_tail]  # convert from radius to diameter for the cross-section widths
    x_tail = x_tail*total_length  # convert from fraction back to actual length along the fuselage
    slopes_tail = slopes_tail / total_length          # dr/d(x_frac) → dr/dx_physical (dimensionless)
    curvatures_tail = curvatures_tail / total_length**2  # d²r/d(x_frac)² → d²r/dx_physical²


    total_length = functional_length + combined_nose_length + tail_fairing

    XS_FOUR_SERIES = getattr(vsp, "XS_FOUR_SERIES", 7)
    XS_FOUR_DIGIT_MOD = getattr(vsp, "XS_FOUR_DIGIT_MOD", 13)  
    thickness = np.round(thickness, 4)  # ensure it's a nice round number for the VSP parameter
    camber = np.round(camber, 4)

    # ----------------------------------------------------------------------
    # 2. BUILD THE WING  (3 spanwise sections)
    # ----------------------------------------------------------------------
    vsp.VSPRenew()
    vsp.ClearVSPModel()

    wing = vsp.AddGeom("WING", "")
    
    vsp.SetGeomName(wing, "WingGeom")



    # A new wing has ONE section. Add two more so XSec_2 and XSec_3 exist.
    vsp.InsertXSec(wing, 0, XS_FOUR_DIGIT_MOD)   # -> 2 sections
    vsp.InsertXSec(wing, 1, XS_FOUR_SERIES)   # -> 3 sections
    vsp.InsertXSec(wing, 2, XS_FOUR_SERIES)   # -> 4 sections 
    vsp.InsertXSec(wing, 3, XS_FOUR_SERIES)   
    


    # Section 1 is the connection between the wing and the fuselage. 
    vsp.SetParmVal(wing, "Root_Chord", "XSec_1", con_chord*total_length)  # set the root chord to the calculated value for a smooth connection
    vsp.SetParmVal(wing, "Tip_Chord",  "XSec_1", chord0*con_chord*total_length)   # set the tip chord of section 1 to the desired root chord of the wing
    vsp.SetParmVal(wing, "Span",       "XSec_1", con_span)    # set the span of section 1 to the desired value
    vsp.SetParmVal(wing, "Sweep",      "XSec_1", con_sweep*90)   # set the sweep of section 1 to the desired value
    vsp.SetParmVal(wing, "Dihedral",   "XSec_1", 0.0)   # keep the dihedral of section 1 flat for a smooth connection
    vsp.SetParmVal(wing, "SectTess_U",     "XSec_1",2*sec_num)  # Default is usually 4. Increase to 15 or 20.
    vsp.SetParmVal(wing, "Sweep_Location", "XSec_1", 0.0)
    
    vsp.Update()    # update after EACH section — they're connected

    # Section 2 is the ONLY section with an independent root chord.
    vsp.SetParmVal(wing, "Root_Chord", "XSec_2", chord0*con_chord*total_length)  # set the root chord of section 2 to the desired value for the wing root
    vsp.SetParmVal(wing, "Tip_Chord",  "XSec_2", chord1*chord0*con_chord*total_length)
    vsp.SetParmVal(wing, "Span",       "XSec_2", span1)
    vsp.SetParmVal(wing, "Sweep",      "XSec_2", sweep1*90)
    vsp.SetParmVal(wing, "Dihedral",   "XSec_2", dihedral1*90)
    vsp.SetParmVal(wing, "SectTess_U",     "XSec_2",sec_num)
    vsp.SetParmVal(wing, "Sweep_Location", "XSec_2", 0.0)
    vsp.Update()    # update after EACH section — they're connected

    # Section 3: its root = section 1's tip (CHORD1), automatically. Only set the tip.
    vsp.SetParmVal(wing, "Tip_Chord", "XSec_3", chord2*chord1*chord0*con_chord*total_length)
    vsp.SetParmVal(wing, "Span",      "XSec_3", span2)
    vsp.SetParmVal(wing, "Sweep",     "XSec_3", sweep2*90)
    vsp.SetParmVal(wing, "Dihedral",   "XSec_3", dihedral2*90)
    vsp.SetParmVal(wing, "SectTess_U",     "XSec_3",sec_num)
    vsp.SetParmVal(wing, "Sweep_Location", "XSec_3", 0.0)  # sweep applies fully to the tip of this section
    vsp.Update()

    # Section 4: root = section 2's tip (CHORD2). Set the final tip chord.
    vsp.SetParmVal(wing, "Tip_Chord", "XSec_4", chord3*chord2*chord1*chord0*con_chord*total_length)
    vsp.SetParmVal(wing, "Span",      "XSec_4", span3)
    vsp.SetParmVal(wing, "Sweep",     "XSec_4", sweep3*90)
    vsp.SetParmVal(wing, "Dihedral",   "XSec_4", dihedral3*90)
    vsp.SetParmVal(wing, "SectTess_U",     "XSec_4",sec_num)
    vsp.SetParmVal(wing, "Sweep_Location", "XSec_4", 0.0)
    vsp.Update()

    # ---- Set every cross-section airfoil to NACA 0004 ----
    xsec_surf = vsp.GetXSecSurf(wing, 0)
    num_xsec = vsp.GetNumXSec(xsec_surf)
    for i in range(num_xsec):
        if i == 0:
            vsp.ChangeXSecShape(xsec_surf, i, XS_FOUR_DIGIT_MOD)   # modified 4-digit at the root
            vsp.Update()
            # max thickness of the connection matches the fuselage height
            tc_ratio = np.round(cabin_height / (con_chord * total_length), 2)
            vsp.SetParmVal(wing, "Camber",     f"XSecCurve_{i}", camber)     # symmetric
            vsp.SetParmVal(wing, "ThickChord", f"XSecCurve_{i}", tc_ratio)   # thickness from root chord
            vsp.SetParmVal(wing, "LERadIndx",  f"XSecCurve_{i}", 3.0)        # LE radius index (I)
            vsp.SetParmVal(wing, "ThickLoc",   f"XSecCurve_{i}", 0.5)        # max-thickness location (M)
        else:
            vsp.ChangeXSecShape(xsec_surf, i, XS_FOUR_SERIES)      # standard 4-series elsewhere
            vsp.Update()
            vsp.SetParmVal(wing, "Camber",     f"XSecCurve_{i}", camber)     # symmetric
            vsp.SetParmVal(wing, "CamberLoc",    f"XSecCurve_{i}", max_camber_loc)      # max-camber location (P)
            vsp.SetParmVal(wing, "ThickChord", f"XSecCurve_{i}", thickness)  # 4% thick
    vsp.Update()

    vsp.SetParmVal(wing, "X_Rel_Location", "XForm", wing_x_location*total_length)   # move the wing 20 units aft
    vsp.Update()

    
    
    # ----------------------------------------------------------------------
    # 3. ADD A FUSELAGE  (set cross-sections by object ID, NOT group strings)
    # ----------------------------------------------------------------------
    XS_ELLIPSE = getattr(vsp, "XS_ELLIPSE", 2)

    fuse = vsp.AddGeom("FUSELAGE", "")
    vsp.SetGeomName(fuse, "Fuselage")


    vsp.SetParmVal(fuse, "Length", "Design", total_length)     # geom-level parm: this one IS addressable by group
    vsp.Update()

    fuse_surf = vsp.GetXSecSurf(fuse, 0)

    def set_xsec_parm(xsec, name, val):
        pid = vsp.GetXSecParm(xsec, name)              # find parm WITHIN this cross-section
        if pid:
            vsp.SetParmVal(pid, val)                   # 2-ARG form — set by parm ID
        else:
            print(f"   [warn] '{name}' not found")

    #add remaining stations
    stations = [] # (x-fraction along length, width, height) for the interior stations
    #Append the nose section to the statiotions list
    stations = [(x_i/total_length, y_i, y_i, 0.0, slope_i, curvature_i) for x_i, y_i, slope_i, curvature_i in zip(x, y_sec, slopes, curvatures)]
    stations.append( (combined_nose_length/total_length, cabin_width, cabin_height, 0.0, 0.0, 0.0) )
    stations.append( (combined_nose_length/total_length+0.25*(functional_length/total_length), cabin_width, cabin_height, 0.0, 0.0, 0.0) )
    stations.append( (combined_nose_length/total_length+0.5*(functional_length/total_length), cabin_width, cabin_height, 0.0, 0.0, 0.0) )
    stations.append( (combined_nose_length/total_length+0.75*(functional_length/total_length), cabin_width, cabin_height, 0.0, 0.0, 0.0) )
    stations.append( ((combined_nose_length+functional_length)/(total_length), cabin_width, cabin_height, 0.0, 0.0, 0.0) )
    stations += [(x_i/total_length, y_i, y_i, 0.0, slope_i, curvature_i) for x_i, y_i, slope_i, curvature_i in zip(x_tail, y_sec_tail, slopes_tail, curvatures_tail)]

    needed = len(stations)  # +1 for the final tip station that defines the end of the fuselage
    while vsp.GetNumXSec(fuse_surf) < needed:
        vsp.InsertXSec(fuse, vsp.GetNumXSec(fuse_surf) - 2, XS_ELLIPSE)

    '''
    vsp.ChangeXSecShape(fuse_surf, 0, getattr(vsp, "XS_CIRCLE", 1))
    nose = vsp.GetXSec(fuse_surf, 0)            # re-fetch after the shape change
    set_xsec_parm(nose, "Circle_Diameter", 2 * nose_radius)
    '''
    vsp.Update()

   
    for i, (xloc, w, h, z, slope, curvature) in enumerate(stations):
    
        vsp.ChangeXSecShape(fuse_surf, i, XS_ELLIPSE)
        xsec = vsp.GetXSec(fuse_surf, i)               # RE-FETCH after shape change
        set_xsec_parm(xsec, "XLocPercent",   xloc)
        set_xsec_parm(xsec, "ZLocPercent",   z)   
        set_xsec_parm(xsec, "Ellipse_Width",  w)
        set_xsec_parm(xsec, "Ellipse_Height", h)
        set_xsec_parm(xsec,  "SectTess_U", sec_num)
        apply_station_skinning(xsec, np.arctan(slope)*180/np.pi,  curvature)  # apply skinning to the nose stations based on the calculated slopes and curvatures
    vsp.Update()
    vsp.WriteVSPFile(output_vsp3)
    if output_stl:
        vsp.ExportFile(output_stl, vsp.SET_ALL, vsp.EXPORT_STL)
        exclude_fuselage_faces(output_stl, (combined_nose_length, total_length-tail_fairing, cabin_width/2), output_stl)

    if verbose:
        print(f"Fuselage built and saved to {output_vsp3}")

    return wing, fuse

# ----------------------------------------------------------------------
# CONSTRAINTS
# ----------------------------------------------------------------------
def shock_constraint( 
        span1, 
        span2, 
        span3, 
        sweep1, sweep2, sweep3,
        wing_x_location,  # as fraction of total length
        con_span, con_sweep, **kwargs):
    
    span = span1 + span2 + span3 + con_span
    x_tip = wing_x_location*60.5+span1*np.tan(np.radians(sweep1*90)) + span2*np.tan(np.radians(sweep2*90)) + span3*np.tan(np.radians(sweep3*90)) + con_span*np.tan(np.radians(con_sweep*90))
    
    start_of_shock = 2.7
    angle = np.radians(16.26)
    x_tip += start_of_shock

    
    return x_tip*np.tan(angle) - span # This should be higher than 0 to ebsure shock doen't hit the wing

def structures_feasibility_constraint(
        chord0, chord1, chord2, chord3,
        span1, sweep1,
        span2, sweep2,
        span3, sweep3,
        wing_x_location,  # as fraction of total length
        con_chord, con_span, con_sweep, **kwargs):  
    
    total_length = 60.5
    x_disp_per_seg = [0, con_span*np.tan(np.radians(con_sweep*90)), span1*np.tan(np.radians(sweep1*90)), span2*np.tan(np.radians(sweep2*90)), span3*np.tan(np.radians(sweep3*90))]
    root_chord_x_pos = np.cumsum(x_disp_per_seg) + wing_x_location*total_length
    tip_chord_x_pos = root_chord_x_pos + np.array([con_chord*total_length, chord0*con_chord*total_length, chord1*chord0*con_chord*total_length, chord2*chord1*chord0*con_chord*total_length, chord3*chord2*chord1*chord0*con_chord*total_length])

    tip_tip_pos = max(tip_chord_x_pos)
    if DEBUG_MODE:
        print(chord3*chord2*chord1*chord0*con_chord*total_length)
        print(wing_x_location * total_length)
        print(x_disp_per_seg)
        print(f"Tip of the wing is at {tip_tip_pos:.2f} m along the fuselage, total length is {total_length} m")
    return total_length - tip_tip_pos # ensure the tip of the wing doesn't extend beyond 110% of the total length, to allow for some structures at the rear of the plane


def simple_geometry(chord_root = 1, chord_tip =0.5, span = 5, sweep = 0.5, dihedral = 0.1, wing_x_location = 0):
    # This function generates a simple wing geometry for testing purposes.
    # It creates a wing with specified root and tip chords, span, sweep, dihedral, and location along the fuselage.
    # The generated geometry is saved to a VSP file.

    vsp.VSPRenew()
    vsp.ClearVSPModel()

    wing = vsp.AddGeom("WING", "")
    vsp.SetGeomName(wing, "SimpleWing")

    # Set parameters for the wing
    vsp.SetParmVal(wing, "Root_Chord", "XSec_1", chord_root)
    vsp.SetParmVal(wing, "Tip_Chord",  "XSec_1", chord_tip)
    vsp.SetParmVal(wing, "Span",       "XSec_1", span)
    vsp.SetParmVal(wing, "Sweep",      "XSec_1", sweep * 90)  # Convert to degrees
    vsp.SetParmVal(wing, "Dihedral",   "XSec_1", dihedral * 90)  # Convert to degrees
    vsp.SetParmVal(wing, "X_Rel_Location", "XForm", wing_x_location)

    vsp.Update()
    
    output_vsp3 = "simple_wing.vsp3"
    vsp.WriteVSPFile(output_vsp3)

    # Call structure constraint unsepcified argiemts are 0 
    const = structures_feasibility_constraint(
        chord0=chord_root, chord1=chord_tip, chord2=1.0, chord3=1.0,
        span1=span, sweep1=sweep,
        span2=0.0, sweep2=0.0,
        span3=0.0, sweep3=0.0,
        wing_x_location=wing_x_location,  # as fraction of total length
        functional_length=0,
        tail_fairing=0.0,
        cabin_width=0.0, con_chord=1.0/60.5, con_span=0.0, con_sweep=0.0 )
    
    return const

def area_constraint(params, target_area):
    
    generate_geometry(**params, output_stl="temp_area_check.stl")
    geom = _hf.load_geometry("temp_area_check.stl")
    n, area = geom[0], geom[1]
    sref = float(np.sum(area * np.abs(n[:, 2])) / 2.0)

    return (sref - target_area)/target_area # must be positive to ensure the wing area is above the target

def exact_volume_constraint(params, volume_target):
    
    generate_geometry(**params, output_stl="temp_volume_check.stl")
    volume = mesh_volume("temp_volume_check.stl")

    return (volume - volume_target)/volume_target
    
def denormalize_params(x, optimize_keys=OPTIMIZE_KEYS, baseline_params=BASELINE_PARAMS):
    params = baseline_params.copy()
    for i, key in enumerate(optimize_keys):
        params[key] = x[i] * baseline_params[key]
    return params

def normalize_params(params):
    """Convert a dict {name: actual_value} to a normalized array (each / its default)."""
    return np.array([params[name] / BASELINE_PARAMS[name] for name in BASELINE_PARAMS.keys()])



# ----------------------------------------------------------------------
# Post processing functions
# ----------------------------------------------------------------------
def process_history(history,target_volume=None,init_sref=None, filename="optimisation_history.png"):
    iters = [h["iter"] for h in history]
    lds   = [h["LD"]   for h in history]
    cls   = [h["CL"]   for h in history]
    cds   = [h["CD"]   for h in history]
    volumes = [h["volume"] for h in history]
    span_constraints = [h["span_constraint"] for h in history]
    structures_constraints = [h["structures_constraint"] for h in history]
    area_constraints = [h["sref"] for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(iters, lds,  'b-', lw=1.5)
    axes[0].set_ylabel("L/D")
    axes[1].plot(iters, cls, 'g-', lw=1.5)
    axes[1].set_ylabel("CL")
    axes[1].axhline(0.02, color='k', linestyle='--', label='CL target')
    axes[2].plot(iters, cds, 'r-', lw=1.5)
    axes[2].set_ylabel("CD")

    fig.suptitle("Optimisation convergence history")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes[0, 0].plot(iters, span_constraints, 'c-', lw=1.5)
    axes[0, 0].set_ylabel("Span Constraint")
    axes[0, 1].plot(iters, structures_constraints, 'm-', lw=1.5)
    axes[0, 1].set_ylabel("Structures Constraint")
    axes[1, 0].plot(iters, area_constraints, 'y-', lw=1.5)
    if init_sref:
        axes[1, 0].axhline(init_sref, color='k', linestyle='--', label='Initial Sref')
        axes[1, 0].legend()
    axes[1, 0].set_ylabel("Area Constraint")
    axes[1, 1].plot(iters, volumes, 'm-', lw=1.5)
    if target_volume:
        axes[1, 1].axhline(target_volume, color='k', linestyle='--', label='Target Volume')
        axes[1, 1].legend()
    axes[1, 1].set_ylabel("Volume")
    axes[1, 1].set_xlabel("Function evaluation")

    plt.suptitle("Constraint and volume history")
    plt.tight_layout()
    plt.savefig(filename.replace(".png", "_constraints.png"), dpi=300)
    plt.close()
    
    

def text_summary(history):
    best = max(history, key=lambda h: h["LD"])
    print("\n" + "="*50)
    print(f"Optimisation complete  —  {len(history)} evaluations")
    print(f"Best L/D = {best['LD']:.3f}  (iter {best['iter']})")
    print("="*50)
    for name, val in best["params"].items():
        diff = val - BASELINE_PARAMS[name]
        sign = "+" if diff >= 0 else ""
        print(f"  {name:<20}: {val:>9.4f}   ({sign}{diff:.4f} from baseline)")

def save_optimal_geometry(history, best_vsp="optimal_geometry.vsp3", best_stl="optimal_geometry.stl", target_volume=None):
    def is_feasible(h):
        if h.get("span_constraint", 0) < 0:
            return False
        if h.get("structures_constraint", 0) < 0:
            return False
        if target_volume is not None:
            if abs(h["volume"] - target_volume) / target_volume > 0.1:  # ±10% volume tolerance
                return False
        return True

    feasible = [h for h in history if is_feasible(h)]
    if feasible:
        best = max(feasible, key=lambda h: h["LD"])
        print(f"Best feasible entry: iter {best['iter']}, L/D = {best['LD']:.3f}  ({len(feasible)}/{len(history)} entries feasible)")
    else:
        # Find 5% most feasible for structures constraint
        sorted_by_struct = sorted(history, key=lambda h: h.get("structures_constraint", float('inf')), reverse=True)
        top_struct = sorted_by_struct[:max(1, len(sorted_by_struct) // 20)]
        best_struct = max(top_struct, key=lambda h: h["LD"])
        print(f"No fully feasible entry found — best among top 5% by structures constraint: iter {best_struct['iter']}, L/D = {best_struct['LD']:.3f}")
        best = best_struct

    generate_geometry(**best["params"], output_vsp3=best_vsp, output_stl=best_stl)
    try:
        vsp.CaptureScreen(best_vsp.replace(".vsp3", ".png"), False)
        print(f"Screenshot saved as {best_vsp.replace('.vsp3', '.png')}")
    except Exception:
        print(f"Optimal geometry saved as {best_vsp}  (open in OpenVSP to view)")

def save_history(history, filename="optimisation_history.xlsx"):
    rows = []
    for h in history:
        row = {"iter": h["iter"], "CL": h["CL"], "CD": h["CD"], "L/D": h["LD"], "volume": h["volume"], "span_constraint": h["span_constraint"], "structures_constraint": h["structures_constraint"], "area_constraint": h["sref"]}
        row.update(h["params"])
        rows.append(row)
    df = pd.DataFrame(rows)
    try:
        out_path = filename
        df.to_excel(out_path, index=False)
    except ModuleNotFoundError:
        out_path = filename.replace(".xlsx", ".csv")
        df.to_csv(out_path, index=False)
    print(f"Full history ({len(history)} evaluations) saved to {out_path}")


def retrieve_iteration(iteration, csv_path=None, output_vsp3=None, output_stl=None):
    """Regenerate and save the geometry from a specific iteration in a history CSV.

    Parameters
    ----------
    iteration : int
        The iteration number to retrieve (matches the 'iter' column).
    csv_path : str, optional
        Path to the history CSV. Defaults to FILE_NAME + '.csv'.
    output_vsp3 : str, optional
        Output .vsp3 path. Defaults to 'retrieved_iter_<N>.vsp3'.
    output_stl : str, optional
        Output .stl path. Pass None to skip STL export.
    """
    if csv_path is None:
        csv_path = f"{FILE_NAME}.csv"
    if output_vsp3 is None:
        output_vsp3 = f"retrieved_iter_{iteration}.vsp3"

    df = pd.read_csv(csv_path)
    matches = df[df["iter"] == iteration]
    if matches.empty:
        available = f"{int(df['iter'].min())}–{int(df['iter'].max())}"
        raise ValueError(f"Iteration {iteration} not found in '{csv_path}'. Available range: {available}")

    row = matches.iloc[0]
    print(f"Iteration {iteration}: L/D={row['L/D']:.4f}  CL={row['CL']:.4f}  CD={row['CD']:.4f}  volume={row['volume']:.2f} m³")

    param_keys = list(BASELINE_PARAMS.keys())
    params = {k: float(row[k]) for k in param_keys if k in df.columns}

    generate_geometry(**params, output_vsp3=output_vsp3, output_stl=output_stl)
    print(f"Geometry saved to {output_vsp3}")
    return params


def plot_history_clean(csv_path=None, output_prefix=None):
    """Plot CL, CD, L/D and constraint histories from a CSV, masking unphysical points.

    Unphysical thresholds
    ---------------------
    CL                : must be in (0, 1)
    CD                : must be in (0, 1)
    L/D               : physical only when both CL and CD are physical
    span_constraint   : must be in (-50, 50)
    structures_constraint : must be in (-50, 50)
    area_constraint   : must be in (0, 2000)

    Unphysical points are plotted as a red × at y=0; physical points form the line.
    """
    if csv_path is None:
        csv_path = f"{FILE_NAME}.csv"
    if output_prefix is None:
        output_prefix = csv_path.replace(".csv", "")

    df = pd.read_csv(csv_path)
    iters = df["iter"].to_numpy()

    cl_ok   = (df["CL"]  > 0.0) & (df["CL"]  < 1.0)
    cd_ok   = (df["CD"]  > 0.0) & (df["CD"]  < 0.10)
    ld_ok   = cl_ok & cd_ok
    span_ok = (df["span_constraint"]       > -50) & (df["span_constraint"]       < 50)
    str_ok  = (df["structures_constraint"] > -50) & (df["structures_constraint"] < 50)
    area_ok = (df["area_constraint"]       >   0) & (df["area_constraint"]       < 2000)

    def _masked(series, mask):
        arr = series.to_numpy(dtype=float).copy()
        arr[~mask.to_numpy()] = np.nan
        return arr

    def _plot_series(ax, x, series, mask, color, ylabel, hline=None, hline_label=None):
        ax.plot(x, _masked(series, mask), color=color, lw=1.5)
        bad = ~mask.to_numpy()
        if bad.any():
            ax.plot(x[bad], np.zeros(bad.sum()), "rx", ms=8, mew=2, label=f"unphysical ({bad.sum()})")
            ax.legend(fontsize=8)
        if hline is not None:
            ax.axhline(hline, color="k", ls="--", lw=1, label=hline_label)
            ax.legend(fontsize=8)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_ylabel(ylabel)

    # --- CL / CD / L/D ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    _plot_series(axes[0], iters, df["CL"],  cl_ok, "g", "CL",  hline=CL_MIN, hline_label=f"CL_MIN = {CL_MIN:.3f}")
    _plot_series(axes[1], iters, df["CD"],  cd_ok, "r", "CD")
    _plot_series(axes[2], iters, df["L/D"], ld_ok, "b", "L/D")
    axes[2].set_xlabel("Function evaluation")
    fig.suptitle("Convergence history  (red × = unphysical, set to 0)")
    plt.tight_layout()
    out1 = f"{output_prefix}_convergence_clean.png"
    fig.savefig(out1, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved {out1}")

    # --- constraints ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    _plot_series(axes[0], iters, df["span_constraint"],       span_ok, "c", "Span constraint (m)",       hline=0)
    _plot_series(axes[1], iters, df["structures_constraint"], str_ok,  "m", "Structures constraint (m)", hline=0)
    _plot_series(axes[2], iters, df["area_constraint"],       area_ok, "y", "Reference area (m²)")
    axes[2].set_xlabel("Function evaluation")
    fig.suptitle("Constraint history  (red × = unphysical, set to 0)")
    plt.tight_layout()
    out2 = f"{output_prefix}_constraints_clean.png"
    fig.savefig(out2, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved {out2}")


def run_optimization():
    """Run SLSQP optimisation. Returns (scipy result, history list)."""
    history = []   # each entry: {"iter", "params", "CL", "CD", "LD"}
    _eval_cache = {}   # tuple(x) -> CL; every objective() call populates this, _cl_jac reads it
    t_start = time.time()

    def objective(x):  
        intermediate_res_folder = "intermediate_results_6aoa_naca4_refcl"
        os.makedirs(intermediate_res_folder, exist_ok=True)

        stl_path = f"temp_{len(history)+1}.stl"
        
        full_stl_path = os.path.join(intermediate_res_folder, stl_path)
        params = denormalize_params(x)
        generate_geometry(**params, output_stl=full_stl_path)

        geom = _hf.load_geometry(full_stl_path)
        n, area = geom[0], geom[1]
        sref = float(np.sum(area * np.abs(n[:, 2])) / 2.0)
        
        CL, CD = _hf.run_case_total_coef(geom, MACH, ALPHA, sref)
        _eval_cache[tuple(x)] = CL   # cache for _cl_constraint / _cl_jac
        ld = CL / CD if CD != 0 else 0.0

        span_constraint = shock_constraint(**params)
        structures_constraint = structures_feasibility_constraint(**params)
        history.append({"iter": len(history) + 1,
                         "params": params.copy(),
                         "CL": CL, "CD": CD, "LD": ld, "volume": 0.0, "span_constraint": span_constraint, "structures_constraint": structures_constraint, "sref": sref})
        it = len(history) + 1
        col = 3
        param_items = list(params.items())
        rows = [param_items[i:i+col] for i in range(0, len(param_items), col)]
        print(f"\n{'─'*62}")
        print(f"  Iter {it:>3}   L/D = {ld:.4f}   CL = {CL:.4f}   CD = {CD:.4f}, Volume = {history[-1]['volume']:.2f} m^3, Shock constraint = {span_constraint:.2f} m, Structures constraint = {structures_constraint:.2f}, reference area = {sref:.2f} m^2")
        print(f"{'─'*62}")
        for row in rows:
            parts = []
            for k, v in row:
                delta = v - BASELINE_PARAMS[k]
                sign = "+" if delta >= 0 else ""
                parts.append(f"{k:<18} {v:>9.5f} ({sign}{delta:.5f})")
            print("  " + "   ".join(parts))
        print(f"{'─'*62}")
        

        return -ld

    x0 = np.ones(len(OPTIMIZE_KEYS))   # normalized start point (each param / its default = 1)
    params = denormalize_params(x0)
    first_stl_path = "initial_geometry.stl"
    generate_geometry(**params, output_stl=first_stl_path)

    geom = _hf.load_geometry(first_stl_path)
    target_volume = mesh_volume(first_stl_path)
    print(f"Target volume (from initial geometry): {target_volume:.2f} m^3")

    # Objective check on initial geometry
    n, area = geom[0], geom[1]
    init_sref = float(np.sum(area * np.abs(n[:, 2])) / 2)

    #CL_MIN = WEIGHT*9.81/(0.5*RHO*(MACH*SPEED_OF_SOUND)**2*init_sref)  # CL required to generate enough lift to support the weight
    CL_MIN = 0.02
    min_lift = 0.02
    print(f"Lift constraint (CL required to support weight): {CL_MIN:.4f}")
    print(f"Initial geometry reference area (Sref): {init_sref:.2f} m^2")
    CL, CD = _hf.run_case_total_coef(geom, MACH, ALPHA, init_sref)
    ld = CL / CD if CD != 0 else 0.0
    print(f"Initial geometry: L/D = {ld:.4f}   CL = {CL:.4f}   CD = {CD:.4f}   Volume = {target_volume:.2f} m^3")
    structures_const = structures_feasibility_constraint(**params)
    print(f"Initial geometry structures constraint: {structures_const:.2f} m (positive is feasible)")

    # call objective and compare to the above to ensure consistency before starting the optimization loop
    obj_val = objective(x0)
    print(f"Objective function value at initial geometry: {obj_val:.4f}  (should be {-ld:.4f})")
    input("Press Enter to start optimization...")


    def _volume_jac(constraint_fn, x, eps=0.01):
        f0 = constraint_fn(x)
        grad = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            grad[i] = (constraint_fn(xp) - f0) / eps
        return grad

    
    

    def _cl_constraint(x):
        key = tuple(x)
        if key in _eval_cache:
            return _eval_cache[key] - CL_MIN
        # Fallback for the rare case this is called before objective at the same x
        params = denormalize_params(x)
        generate_geometry(**params, output_stl="temp_cl_jac.stl")
        geom = _hf.load_geometry("temp_cl_jac.stl")
        n, area = geom[0], geom[1]
        sref_tmp = float(np.sum(area * np.abs(n[:, 2])) / 2.0)
        CL, _ = _hf.run_case_total_coef(geom, MACH, ALPHA, sref_tmp)
        _eval_cache[key] = CL
        return CL - CL_MIN

    def _cl_jac(x):
        # Mirror scipy's '2-point' step: h = rel_step * max(1, |x|).
        # SLSQP computes the objective gradient (calling objective at x+h_i) before
        # constraint Jacobians, so _eval_cache already holds CL at every x+h_i here.
        f0 = _cl_constraint(x)
        grad = np.zeros(len(x), dtype=float)
        for i, key in enumerate(OPTIMIZE_KEYS):
            h = RELATIVE_STEPS.get(key, 0.01) * max(1.0, abs(x[i]))
            xp = x.copy()
            xp[i] += h
            grad[i] = (_cl_constraint(xp) - f0) / h
        return grad

    constraints = [
        # shock_constraint returns (span_available - span_used): must stay >= 0
        {'type': 'ineq', 'fun': lambda x: shock_constraint(**denormalize_params(x)), 'jac': lambda x: _volume_jac(lambda p: shock_constraint(**denormalize_params(p)), x, eps=0.005)},
        #{'type': 'eq', 'fun': lambda x: exact_volume_constraint(denormalize_params(x), target_volume), 'jac': lambda x: _volume_jac(lambda p: exact_volume_constraint(denormalize_params(p), target_volume), x)},
        {'type': 'ineq', 'fun': lambda x: structures_feasibility_constraint(**denormalize_params(x)), 'jac': lambda x: _volume_jac(lambda p: structures_feasibility_constraint(**denormalize_params(p)), x, eps=0.005)},
        {'type': 'ineq', 'fun': lambda x: area_constraint(denormalize_params(x), init_sref), 'jac': lambda x: _volume_jac(lambda p: area_constraint(denormalize_params(p), init_sref), x)},
        {'type': 'ineq', 'fun': _cl_constraint, 'jac': _cl_jac},
    ]

    # In normalized space every parameter ≈ 1.0, so a fractional step is already
    # the correct absolute step — no further normalization needed.
    relative_steps = np.array([RELATIVE_STEPS.get(param, 0.01) for param in OPTIMIZE_KEYS])

    print("Starting optimisation...")
    result = opt.minimize(objective, x0, method='SLSQP',
                          jac='2-point',
                          constraints=constraints,
                          bounds=BOUNDS,
                          options={'maxiter': MAX_ITER, 'disp': True, 'finite_diff_rel_step': relative_steps})

    t_end = time.time()
    save_history(history, filename=f"{FILE_NAME}.xlsx")
    save_optimal_geometry(history, best_vsp=f"{FILE_NAME}.vsp3", best_stl=f"{FILE_NAME}.stl", target_volume=target_volume)
    
    process_history(history, target_volume=target_volume, init_sref=init_sref, filename=f"{FILE_NAME}.png")
    text_summary(history)
    
    print(f"Optimisation finished in {t_end - t_start:.1f} seconds")
    print(f"Volume of the optimized geometry: {mesh_volume(f'{FILE_NAME}.stl'):.2f} m^3")
    print(f"Constraint violations:{target_volume - mesh_volume(f'{FILE_NAME}.stl'):.2f} m^3 (negative means volume constraint is satisfied)")
    
    return result, history


def does_it_fly(stl_file, alpha):
    geom = _hf.load_geometry(stl_file)
    
    
    n, area = geom[0], geom[1]
    sref = float(np.sum(area * np.abs(n[:, 2])) / 2)

    CL, CD = _hf.run_case_total_coef(geom, MACH, alpha, sref)
    L = 0.5*RHO*(MACH*SPEED_OF_SOUND)**2*sref*CL
    D = 0.5*RHO*(MACH*SPEED_OF_SOUND)**2*sref*CD

    return L, L>WEIGHT, L-WEIGHT, CD, CL


def summarize_in_table(csv_file, iter):
    # This function reads given iteration in the csv file and prints table with the geometric parameters denormalized and deparametrized
    df = pd.read_csv(csv_file)
    row = df[df["iter"] == iter].iloc[0]
    params = {k: row[k] for k in BASELINE_PARAMS.keys()}
    params_vec = np.array(list(params.values()))
    CL, CD, LD, volume = row["CL"], row["CD"], row["L/D"], row["volume"]
    print(f"Iteration {iter}:")
    print(f"  CL = {CL:.4f}, CD = {CD:.4f}, L/D = {LD:.4f}, Volume = {volume:.2f} m^3")
    denorm_params = params.copy()
    
    # Deparametrize: convert from absolute values back to the original parameter space (e.g. chord ratios back to actual chords)
    deparam_params = {}   
    deparam_params["con_chord"] = denorm_params["con_chord"]*TOTAL_LENGTH  # con_chord is a ratio of total length, so multiply back to get actual chord0
    deparam_params["chord0"] = denorm_params["chord0"] * deparam_params["con_chord"]  # chord0 is a ratio of con_chord, so multiply back to get actual chord0
    deparam_params["chord1"] = denorm_params["chord1"] * deparam_params["chord0"]  # chord1 is a ratio of chord0, so multiply back to get actual chord1
    deparam_params["chord2"] = denorm_params["chord2"] * deparam_params["chord1"]  # chord2 is a ratio of chord1, so multiply back to get actual chord2    
    deparam_params["chord3"] = denorm_params["chord3"] * deparam_params["chord2"]  # chord3 is a ratio of chord2, so multiply back to get actual chord3
    deparam_params["span1"] = denorm_params["span1"]  # span1 is an absolute value, so no change needed
    deparam_params["sweep1"] = denorm_params["sweep1"] * 90  # sweep1 is a ratio of 90 degrees, so multiply back to get actual sweep1 in degrees
    deparam_params["dihedral1"] = denorm_params["dihedral1"] * 90  # dihedral1 is a ratio of 90 degrees, so multiply back to get actual dihedral1 in degrees
    deparam_params["span2"] = denorm_params["span2"]  # span2 is an absolute value, so no change needed
    deparam_params["sweep2"] = denorm_params["sweep2"] * 90  # sweep2 is a ratio of 90 degrees, so multiply back to get actual sweep2 in degrees
    deparam_params["dihedral2"] = denorm_params["dihedral2"] * 90  # dihedral2 is a ratio of 90 degrees, so multiply back to get actual dihedral2 in degrees
    deparam_params["span3"] = denorm_params["span3"]  # span3 is an absolute value, so no change needed
    deparam_params["sweep3"] = denorm_params["sweep3"] * 90  # sweep3 is a ratio of 90 degrees, so multiply back to get actual sweep3 in degrees
    deparam_params["dihedral3"] = denorm_params["dihedral3"] * 90  # dihedral3 is a ratio of 90 degrees, so multiply back to get actual dihedral3 in degrees
    deparam_params["wing_x_location"] = denorm_params["wing_x_location"] * TOTAL_LENGTH  # wing_x_location is a ratio of total length, so multiply
    deparam_params["con_sweep"] = denorm_params["con_sweep"] * 90  # con_sweep is an absolute value, so no change needed
    deparam_params["con_span"] = denorm_params["con_span"]  # con_span is an absolute value, so no change needed
 
    print("  Denormalized and deparametrized parameters:")
    for k, v in deparam_params.items():
        print(f"    {k:<20}: {v:.4f}")


if __name__ == "__main__":

    '''
    generate_geometry validation: vary the sweep angles and confirm they are set correctly in the generated VSP file.
    generate_geometry(sweep1=60, sweep2=30, sweep3=60, output_vsp3='conf1.vsp3', verbose=True)
    generate_geometry(sweep1=80, sweep2=30, sweep3=50, output_vsp3='conf2.vsp3', verbose=True)
    generate_geometry(sweep1=60, sweep2=40, sweep3=80, output_vsp3='conf3.vsp3', verbose=True)
    '''

    #generate_geometry(output_stl="initial_geometry_mesh_refined_10.stl")  # generate initial geometry and save to STL for volume measurement
    #generate_geometry(chord0=39.682615, chord1=13.134241, chord2=11.923226, chord3=7.2583349, sweep1=77.892133, sweep2=21.700884, sweep3=51.171643)  # generate default geometry and save to VSP file
    #generate_geometry(output_vsp3="baseline_geometry.vsp3", output_stl="baseline_geometry.stl", verbose=True)  # generate initial geometry and save to STL for volume measurement
    #run_optimization()
    
    #plot_history_clean(csv_path=f"Final_analysis_optimization\optimisation_history_sensitivity.csv", output_prefix="optimisation_history_sensitivity")  # plot the history of the optimization, masking unphysical points
    csv_path=r"C:\Users\Maria\Documents\DSE\DSE\Final_analysis_optimization\optimisation_history_3aoa_naca4_lowcl.csv"
    #summarize_in_table(csv_path, iter=430)  # print the parameters and results of a specific iteration in a readable table format
    retrieve_iteration(430, csv_path=csv_path, output_vsp3=f"idk.vsp3", output_stl=f"idk.stl")  # regenerate geometry from the first iteration and save to VSP and STL
    #const=simple_geometry()  # generate a simple wing geometry for testing purposes
    #print(f"Structures feasibility constraint value for simple geometry: {const:.2f} m (should be positive to be feasible)")
    #run_bayesian_optimization()
