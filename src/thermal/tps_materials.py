# =============================================================================
# TPS MATERIAL DATABASE
# =============================================================================
# File name suggestion:
# tps_materials.py
#
# This file stores material properties for hypersonic TPS analyses.
# Values are approximate engineering values intended for conceptual design.
#
# Units:
# ---------------------------------------------------------------------------
# density               [kg/m^3]
# thermal_conductivity  [W/m/K]
# specific_heat         [J/kg/K]
# emissivity            [-]
# max_service_temp      [K]
# youngs_modulus        [Pa]
# poisson_ratio         [-]
# thermal_expansion     [1/K]
# yield_strength        [Pa]
# creep_temp_limit      [K]
#
# IMPORTANT:
# - Many properties are temperature dependent in reality.
# - Replace placeholders with literature/manufacturer data when available.
# - Some values intentionally left as None when uncertain.
# =============================================================================

MATERIALS = {

    # =========================================================================
    # TITANIUM ALLOYS
    # =========================================================================

    "Ti6Al4V": {

        "type": "Titanium Alloy",

        "density": 4430,

        "thermal_conductivity": 7.187,

        "specific_heat": 522.6,

        "emissivity": 0,

        "max_service_temp": 873,

        "youngs_modulus": 113e9,

        "poisson_ratio": 0.34,

        "thermal_expansion": 8.6e-6,

        "yield_strength": 880e6,

        "creep_temp_limit": 773,

        "notes":
            "Conventional aerospace titanium alloy. "
            "Good strength-to-weight but poor above ~600C."
    },


    "Gamma_TiAl": {

        "type": "Gamma Titanium Aluminide",

        "density": 3900,

        "thermal_conductivity": 11.0,

        "specific_heat": 750,

        "emissivity": 0.70,

        "max_service_temp": 1173,

        "youngs_modulus": 170e9,

        "poisson_ratio": 0.27,

        "thermal_expansion": 11e-6,

        "yield_strength": 450e6,

        "creep_temp_limit": 1073,

        "notes":
            "High-temperature lightweight intermetallic. "
            "Strong candidate for hypersonic hot structures."
    },


    # =========================================================================
    # NICKEL SUPERALLOYS
    # =========================================================================

    "Inconel_718": {

        "type": "Nickel Superalloy",

        "density": 8190,

        "thermal_conductivity": 11.4,

        "specific_heat": 435,

        "emissivity": 0.80,

        "max_service_temp": 1250,

        "youngs_modulus": 200e9,

        "poisson_ratio": 0.29,

        "thermal_expansion": 13e-6,

        "yield_strength": 1030e6,

        "creep_temp_limit": 1100,

        "notes":
            "Excellent creep resistance and oxidation resistance. "
            "Heavy."
    },


    "Haynes_230": {

        "type": "Nickel Superalloy",

        "density": 8970,

        "thermal_conductivity": 8.4,

        "specific_heat": 460,

        "emissivity": 0.80,

        "max_service_temp": 1425,

        "youngs_modulus": 205e9,

        "poisson_ratio": 0.31,

        "thermal_expansion": 12.8e-6,

        "yield_strength": None,

        "creep_temp_limit": 1200,

        "notes":
            "Very strong oxidation resistance at extreme temperatures."
    },


    # =========================================================================
    # CMC MATERIALS
    # =========================================================================

    "SiC_SiC_CMC": {

        "type": "CMC",

        "density": 2700,

        "thermal_conductivity": 10.0,

        "specific_heat": 800,

        "emissivity": 0.70,

        "max_service_temp": 1600,

        "youngs_modulus": None,

        "poisson_ratio": None,

        "thermal_expansion": 4e-6,

        "yield_strength": None,

        "creep_temp_limit": 1700,

        "notes":
            "Silicon-carbide ceramic matrix composite. "
            "Excellent high-temperature capability."
    },

    "IMI_Effective": {

        "type": "Integrated Multilayer Insulation",

        "density": 80,

        "thermal_conductivity": 0.010,

        "specific_heat": 1000,

        "emissivity": 0.30,

        "max_service_temp": 1400,

        "youngs_modulus": None,

        "poisson_ratio": None,

        "thermal_expansion": None,

        "yield_strength": None,

        "creep_temp_limit": None,

        "notes":
            "Effective-property representation of a high-temperature "
            "Integrated Multilayer Insulation (IMI) system. "
            "Represents combined radiation-shield and fibrous-layer "
            "performance for conceptual hypersonic TPS sizing. "
            "Thermal conductivity is an equivalent through-thickness "
            "value intended for 1D thermal analysis."
    },


    "CVI-C/SiC": {

        "type": "Carbon-Carbon Composite",

        "density": 2100,

        "thermal_conductivity": 10,

        "specific_heat": 750,

        "emissivity": 0.76,

        "max_service_temp": 1473,

        "youngs_modulus": None,

        "poisson_ratio": None,

        "thermal_expansion": 1e-6,

        "yield_strength": None,

        "creep_temp_limit": None,

        "notes":
            "Requires oxidation protection."
    },


    # =========================================================================
    # INSULATION MATERIALS
    # =========================================================================

    "IMI_Insulation": {

        "type": "Insulation",

        "density": None,

        "thermal_conductivity": 0.05,

        "specific_heat": None,

        "emissivity": None,

        "max_service_temp": None,

        "youngs_modulus": None,

        "poisson_ratio": None,

        "thermal_expansion": None,

        "yield_strength": None,

        "creep_temp_limit": None,

        "notes":
            "Placeholder IMI insulation properties. "
            "Replace with actual supplier/manufacturer data."
    },


    "Aerogel": {

        "type": "Aerogel Insulation",

        "density": 110,

        "thermal_conductivity": 0.049,

        "specific_heat": 900,

        "emissivity": 0.90,

        "max_service_temp": 1000,

        "youngs_modulus": None,

        "poisson_ratio": None,

        "thermal_expansion": None,

        "yield_strength": None,

        "creep_temp_limit": None,

        "notes":
            "Excellent insulation, fragile mechanically."
    },


    # =========================================================================
    # STRUCTURAL MATERIALS
    # =========================================================================

    "Aluminum_7075_T6": {

        "type": "Aluminum Alloy",

        "density": 2810,

        "thermal_conductivity": 130,

        "specific_heat": 960,

        "emissivity": 0.10,

        "max_service_temp": 393,

        "youngs_modulus": 71e9,

        "poisson_ratio": 0.33,

        "thermal_expansion": 23e-6,

        "yield_strength": 500e6,

        "creep_temp_limit": 373,

        "notes":
            "Very temperature sensitive."
    },


    "CFRP": {

        "type": "Carbon Fiber Reinforced Polymer",

        "density": 1600,

        "thermal_conductivity": None,

        "specific_heat": 900,

        "emissivity": 0.85,

        "max_service_temp": 450,

        "youngs_modulus": None,

        "poisson_ratio": None,

        "thermal_expansion": None,

        "yield_strength": None,

        "creep_temp_limit": None,

        "notes":
            "Matrix temperature limits dominate."
    }

}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_material(name):
    """
    Returns material property dictionary.
    """

    if name not in MATERIALS:
        raise ValueError(f"Material '{name}' not found.")

    return MATERIALS[name]


def list_materials():
    """
    Prints all available materials.
    """

    print("\nAvailable materials:\n")

    for mat in MATERIALS:
        print(f" - {mat}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":

    list_materials()

    titanium = get_material("Gamma_TiAl")

    print("\nGamma Titanium Aluminide Properties:\n")

    for key, value in titanium.items():
        print(f"{key}: {value}")