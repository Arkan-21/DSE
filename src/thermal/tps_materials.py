

MATERIALS = {

    # =========================================================================
    # HOT-FACE CERAMICS & CERAMIC MATRIX COMPOSITES  (surface-capable)
    # =========================================================================

    "CVI_C_SiC": {
        "density": 2100.0,
        "specific_heat": 1400,
        "thermal_conductivity": 6.5,
        "emissivity": 0.85,
        "max_service_temp": 1900.0,
        "notes": "CVI-densified C/SiC CMC; good oxidation resistance with SiC coating.",
    },

    "CVI_SiC_SiC": {
        "density": 2700.0,
        "specific_heat": 620,
        "thermal_conductivity": 9.5,
        "emissivity": 0.85,
        "max_service_temp": 1650.0,
        "notes": "CVI-densified SiC/SiC CMC; superior oxidation resistance to C/SiC, lower max temp.",
    },

    "SiC_SiC_CMC": {
        "density": 2700.0,
        "specific_heat": 800.0,
        "thermal_conductivity": 4.5,
        "emissivity": 0.85,
        "max_service_temp": 1700.0,
        "notes": "SiC fibre / SiC matrix CMC; excellent oxidation resistance; turbine-heritage.",
    },
    "C_C_SiC": {
        "density": 1800.0,
        "specific_heat": 800.0,
        "thermal_conductivity": 12.0,
        "emissivity": 0.88,
        "max_service_temp": 2000.0,
        "notes": "C/C with SiC coating; ultra-high temp; used on X-38 nose cap.",
    },
    "Monolithic_SiC": {
        "density": 3210.0,
        "specific_heat": 750.0,
        "thermal_conductivity": 60.0,
        "emissivity": 0.83,
        "max_service_temp": 1900.0,
        "notes": "Dense sintered SiC; high k limits insulation use; good for leading edges.",
    },
    "UHTC_ZrB2_SiC": {
        "density": 5560.0,
        "specific_heat": 468.0,
        "thermal_conductivity": 60.0,
        "emissivity": 0.80,
        "max_service_temp": 2500.0,
        "notes": "ZrB2-20SiC UHTC; oxidation-stable to ~2300 K; leading-edge material.",
    },
    "Hafnium_Carbide": {
        "density": 12200.0,
        "specific_heat": 200.0,
        "thermal_conductivity": 10.0,
        "emissivity": 0.70,
        "max_service_temp": 3000.0,
        "notes": "HfC ceramic; highest-temp stagnation cap material; very dense.",
    },

    # =========================================================================
    # FIBROUS / TILE INSULATION  (intermediate layer, no structural role)
    # =========================================================================

    "IMI_Pt_Saffil48": {
        "density": 50.0,
        "specific_heat": 1130.0,
        "thermal_conductivity": 0.1,
        "emissivity": 0.45,
        "max_service_temp": 1700.0,
        "notes": "Internal Multiscreen Insulation, Pt-coated ceramic screens with "
                 "SAFFIL felt spacers (ρ_spacer=48 kg/m³, as in X-38 Chin Panel "
                 "20-screen module); λ_eff is strongly T- and P-dependent "
                 "(rises with ambient pressure and hot-side temp) — 0.06 W/mK is "
                 "an approximate mid-range value near 1 atm at moderate T, NOT "
                 "a constant; use parametric model of Weiland et al. for accurate "
                 "sizing. ε=0.45 is the AGED platinum-screen emissivity (worst case "
                 "after thermal cycling); fresh Au screens ≈0.14+6e-5·T. "
                 "1700°C hot-face limit per IMI material capability.",
    },

    "AETB_20": {
        "density": 320.0,
        "specific_heat": 880.0,
        "thermal_conductivity": 0.07,
        "emissivity": None,
        "max_service_temp": 1530.0,
        "notes": "AETB-20 tile; 320 kg/m³; successor to Shuttle HRSI tiles.",
    },
    "FRCI_12": {
        "density": 192.0,
        "specific_heat": 879.0,
        "thermal_conductivity": 0.065,
        "emissivity": None,
        "max_service_temp": 1530.0,
        "notes": "FRCI-12 tile; used on Shuttle fuselage side panels.",
    },
    "Alumina_Fibre_Blanket": {
        "density": 130.0,
        "specific_heat": 1050.0,
        "thermal_conductivity": 0.12,
        "emissivity": None,
        "max_service_temp": 1600.0,
        "notes": "Saffil-type alumina fibre blanket; recrystallises above 1600 K.",
    },
    "Mullite_Fibre_Board": {
        "density": 400.0,
        "specific_heat": 960.0,
        "thermal_conductivity": 0.25,
        "emissivity": None,
        "max_service_temp": 1700.0,
        "notes": "Mullite fibre board; dimensionally stable; good chemical resistance.",
    },
    "Min_K_Board": {
        "density": 320.0,
        "specific_heat": 1000.0,
        "thermal_conductivity": 0.022,
        "emissivity": None,
        "max_service_temp": 1273.0,
        "notes": "Microporous opacified silica board; lowest λ rigid insulator below 1000 K.",
    },
    "Pyrogel_XT_E": {
        "density": 160.0,
        "specific_heat": 1100.0,
        "thermal_conductivity": 0.016,
        "emissivity": None,
        "max_service_temp": 923.0,
        "notes": "Pyrogel XT-E aerogel blanket; λ≈0.016 W/mK at 500 K; 650 °C limit.",
    },
    "Cryogel_Z": {
        "density": 128.0,
        "specific_heat": 1000.0,
        "thermal_conductivity": 0.014,
        "emissivity": None,
        "max_service_temp": 394.0,
        "notes": "Cryogel Z aerogel blanket; cryogenic fuel-tank insulation.",
    },
    "BN_Fibrous_Board": {
        "density": 400.0,
        "specific_heat": 1000.0,
        "thermal_conductivity": 1.50,
        "emissivity": None,
        "max_service_temp": 2000.0,
        "notes": "h-BN fibrous board; electrically insulating; very high temp in inert atmosphere.",
    },

    # =========================================================================
    # STRUCTURAL METALS
    # =========================================================================

    "Gamma_TiAl": {
        "density": 3900.0,
        "specific_heat": 530.0,
        "thermal_conductivity": 22.0,
        "emissivity": None,
        "max_service_temp": 1173.0,
        "notes": "γ-TiAl intermetallic; low density; specific strength to 900 °C.",
    },
    "Al_2024": {
        "density": 2780.0,
        "specific_heat": 875.0,
        "thermal_conductivity": 121.0,
        "emissivity": None,
        "max_service_temp": 450.0,
        "notes": "Al 2024-T3; standard fuselage skin; poor above 450 K sustained.",
    },
    "Al_2099": {
        "density": 2630.0,
        "specific_heat": 900.0,
        "thermal_conductivity": 95.0,
        "emissivity": None,
        "max_service_temp": 480.0,
        "notes": "Al-Li 2099; 5% lighter than 2024; better elevated-temperature properties.",
    },
    "Ti_6Al_4V": {
        "density": 4430.0,
        "specific_heat": 526.0,
        "thermal_conductivity": 6.7,
        "emissivity": None,
        "max_service_temp": 600.0,
        "notes": "Ti-6Al-4V; excellent specific strength; cabin pressure shell to 600 K.",
    },
    "Ti_Beta_21S": {
        "density": 4940.0,
        "specific_heat": 500.0,
        "thermal_conductivity": 8.0,
        "emissivity": None,
        "max_service_temp": 700.0,
        "notes": "Ti Beta-21S; higher high-temp strength than Ti-6Al-4V; X-43 heritage.",
    },
    "Inconel_625": {
        "density": 8440.0,
        "specific_heat": 410.0,
        "thermal_conductivity": 9.8,
        "emissivity": 0.70,
        "max_service_temp": 1173.0,
        "notes": "Inconel 625 Ni superalloy; metallic TPS option; oxidation-stable to ~1150 K.",
    },
    "Haynes_230": {
        "density": 8970.0,
        "specific_heat": 397.0,
        "thermal_conductivity": 8.9,
        "emissivity": 0.72,
        "max_service_temp": 1323.0,
        "notes": "Haynes 230; excellent long-duration oxidation resistance; combustor-liner use.",
    },
    "Rene_N5": {
        "density": 8650.0,
        "specific_heat": 440.0,
        "thermal_conductivity": 11.0,
        "emissivity": 0.65,
        "max_service_temp": 1423.0,
        "notes": "René N5 single-crystal Ni superalloy; turbine blade material.",
    },
    "Refractory_W": {
        "density": 19300.0,
        "specific_heat": 134.0,
        "thermal_conductivity": 130.0,
        "emissivity": 0.45,
        "max_service_temp": 3300.0,
        "notes": "Pure tungsten; highest melting point metal; nozzle throat / plasma-facing.",
    },

    # =========================================================================
    # THERMAL BARRIER COATINGS & SURFACE COATINGS
    # =========================================================================

    "YSZ_TBC": {
        "density": 5600.0,
        "specific_heat": 505.0,
        "thermal_conductivity": 2.0,
        "emissivity": 0.70,
        "max_service_temp": 1473.0,
        "notes": "7YSZ TBC (APS/EB-PVD); turbine-blade coating.",
    },
    "CMAS_Resistant_TBC": {
        "density": 6200.0,
        "specific_heat": 430.0,
        "thermal_conductivity": 1.6,
        "emissivity": 0.72,
        "max_service_temp": 1573.0,
        "notes": "Gd2Zr2O7 next-gen TBC; CMAS-resistant; lower λ than YSZ.",
    },
    "RCC_Coating": {
        "density": 1850.0,
        "specific_heat": 800.0,
        "thermal_conductivity": 6.5,
        "emissivity": 0.89,
        "max_service_temp": 1923.0,
        "notes": "SiC oxidation coating over C/C (RCC); Shuttle nose-cap heritage.",
    },

    # =========================================================================
    # CABIN / INTERIOR MATERIALS
    # =========================================================================

    "PEEK_CF_30": {
        "density": 1450.0,
        "specific_heat": 1100.0,
        "thermal_conductivity": 0.35,
        "emissivity": None,
        "max_service_temp": 523.0,
        "notes": "CF-30/PEEK; aerospace interior structure; excellent FST; 250 °C continuous.",
    },
    "Kapton_Film": {
        "density": 1420.0,
        "specific_heat": 1090.0,
        "thermal_conductivity": 0.12,
        "emissivity": 0.85,
        "max_service_temp": 673.0,
        "notes": "Kapton HN; MLI substrate / vapour barrier.",
    },
    "Nextel_Fabric": {
        "density": 2700.0,
        "specific_heat": 880.0,
        "thermal_conductivity": 0.18,
        "emissivity": None,
        "max_service_temp": 1700.0,
        "notes": "3M Nextel 312 fabric; surface durability coat over fibrous tiles.",
    },

    # =========================================================================
    # ABLATIVES
    # =========================================================================

    "PICA": {
        "density": 240.0,
        "specific_heat": 1400.0,
        "thermal_conductivity": 0.27,
        "emissivity": 0.90,
        "max_service_temp": 3100.0,
        "notes": "PICA ablator; Dragon/Stardust heritage; effective to >3000 K via ablation.",
    },
    "AVCOAT": {
        "density": 512.0,
        "specific_heat": 1400.0,
        "thermal_conductivity": 0.24,
        "emissivity": 0.88,
        "max_service_temp": 3000.0,
        "notes": "Avcoat ablator; Orion/Apollo heritage; epoxy-novolac/silica microspheres.",
    },
    "SLA_561V": {
        "density": 288.0,
        "specific_heat": 1200.0,
        "thermal_conductivity": 0.12,
        "emissivity": 0.87,
        "max_service_temp": 2500.0,
        "notes": "SLA-561V; Mars lander backshell ablator; lower heat flux than lunar return.",
    },
    "Cork_Ablative": {
        "density": 400.0,
        "specific_heat": 1900.0,
        "thermal_conductivity": 0.10,
        "emissivity": 0.85,
        "max_service_temp": 1200.0,
        "notes": "Cork-phenolic ablative; Ariane 5 interstage; cheap; moderate flux only.",
    },
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
