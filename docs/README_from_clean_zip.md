# DSE Group 31 — Mach 5

This is a cleaned tutor-facing copy of the project.

## What was cleaned

The following non-essential files were removed from the submitted zip:

- `.git/`
- `.idea/`, `.vscode/`, `.ai/`
- `__pycache__/`
- `.pyc` compiled Python files
- generated OpenMDAO output folders named `*_out/`

No Python source files were intentionally edited in this cleaned copy.

## How to read this project

Start with these likely main files:

| Area | Suggested files |
|---|---|
| Mission / sizing | `combined_sizing.py`, `Initial_sizing.py` |
| Atmosphere utilities | `isa_atmosphere.py` |
| Aerodynamics / drag | `Merged Drag to Mach Numerical and Graph with Altitude Input.py`, `Sensitivity Study Drag.py` |
| Propulsion | `better_profile.py`, `ramjet_revised.py`, `turbojet_pycycle_wrapper.py`, `Engine/` |
| Thermal | `Heat_temps_turbulent.py`, `Mission_Heat_flux_Laminar.py`, `Radii_alt_graphing.py` |
| Tanks | `Tanks/` |
| Trade-offs | `Trade-offs/` |
| External/reference code | `pycycle/`, `example_cycles/`, `RegenCooling/` |
| Legacy/unverified files | `Old_rubbish/` |

## Important note about running scripts

Many scripts load data files using bare filenames such as:

- `density_velocity_database.csv`
- `mach_fixed_altitude_output.csv`
- `turbojet_multidesign_thrust_maps.npz`
- `ramjet_multidesign_thrust_maps.npz`

For now, run scripts from the project root folder unless the script itself says otherwise.

## Extra documentation

See the `docs/` folder for:

- `restructuring_notes.md`
- `file_inventory.csv`
- `python_dependency_inventory.csv`
- `suggested_categories.csv`
