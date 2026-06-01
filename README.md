# DSE Group 31 — Mach 5 aircraft project

This is a tutor-facing restructured version of the project zip. The original files were reorganized into a clearer source/data/output layout while preserving original filenames wherever possible.

## Folder layout

```text
src/common/              shared utilities, atmosphere, project path helpers
src/aerodynamics/drag/   subsonic/transonic/supersonic/hypersonic drag scripts
src/propulsion/          ramjet/turbojet models, thrust-map/profile scripts
src/propulsion/engine/   original Engine folder contents
src/thermal/             heat flux, Reynolds, wall/stagnation temperature scripts
src/sizing/              initial and combined sizing scripts
src/tanks/               tank sizing/optimization scripts
src/environment/         boom and coastal-zone map scripts
src/trade_offs/          trade-off scripts
data/                    input CSV/NPZ data used by scripts
outputs/                 generated figures/reports and reference output files
external/                third-party or large external code trees: PyCycle, RegenCooling
archive/legacy_unverified/ old/unverified scripts kept for traceability
docs/                    inventories, migration map, restructuring notes
```

## Main entry points

Recommended scripts to inspect first:

- `src/sizing/combined_sizing.py`
- `src/sizing/Initial_sizing.py`
- `src/propulsion/better_profile.py`
- `src/propulsion/ramjet_revised.py`
- `src/propulsion/turbojet_pycycle_wrapper.py`
- `src/common/isa_atmosphere.py`
- `src/thermal/Heat_temps_turbulent.py`
- `src/thermal/Mission_Heat_flux_Laminar.py`
- `src/environment/Boom_map.py`

## Running scripts

Run from the project root when possible, for example:

```bash
python src/sizing/combined_sizing.py
python src/propulsion/better_profile.py
python "src/aerodynamics/drag/Merged Drag to Mach.py"
```

A small import bootstrap was added to project scripts so common modules and data paths continue to resolve after moving files.

## Traceability

See `docs/migration_map.csv` for every original file path and its new location.
See `docs/duplicate_files.csv` for exact duplicate files that may be removable later.
