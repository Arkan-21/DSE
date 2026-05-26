# Restructure summary

Implemented from `docs/restructuring_notes.md`.

## What changed

- Moved source scripts into `src/` by discipline.
- Moved `.csv` and `.npz` inputs into `data/`.
- Moved figures/reports into `outputs/`.
- Moved PyCycle examples and RegenCooling into `external/`.
- Moved old/unverified scripts into `archive/legacy_unverified/`.
- Preserved original filenames to minimize breakage.
- Added `src/common/project_paths.py` for robust data/source/output paths.
- Added import bootstraps to 63 project Python/likely-Python files.
- Rewrote bare data/source filename literals in 12 files.

## Files with patched path literals

- `archive/legacy_unverified/Missiondensitiesv.py`
- `archive/legacy_unverified/optimize_transition_profile.py`
- `archive/legacy_unverified/plot_altitude_vs_range.py`
- `src/propulsion/better_profile.py`
- `src/thermal/Heat_temps_turbulent.py`
- `src/thermal/Mission_Heat_flux_Laminar.py`
- `src/aerodynamics/drag/Improved Merged Drag with Altitude Input`
- `src/aerodynamics/drag/Improved Merged Drag with Altitude Input.py`
- `src/aerodynamics/drag/Input Altitude Merged Drag.py`
- `src/aerodynamics/drag/Input_Alt_Merged_Drag_with_alt.py`
- `src/aerodynamics/drag/Merged Drag to Mach Numerical and Graph with Altitude Input.py`
- `src/aerodynamics/drag/Sensitivity Study Drag.py`

## Caution

This is a structural cleanup, not a scientific refactor. Redundant drag scripts were grouped but not merged because choosing canonical equations requires domain review.
