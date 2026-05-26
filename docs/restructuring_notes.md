# DSE zip static analysis and restructuring plan

## Scope analyzed

- Original zip: `DSE - Copy.zip`
- Files in zip including `.git`: 1915
- Files excluding `.git`: 542
- Python source files: 199
- Compiled/cache files (`.pyc`): 224
- Data files: 32
- Figures/reports: 54

## Main findings

1. The zip includes repository/internal clutter that tutors should not receive:
   - `.git/` history
   - `.idea/`, `.vscode/`, `.ai/`
   - `__pycache__/` and `.pyc` files
   - OpenMDAO output folders such as `*_out/`

2. There are many root-level scripts with overlapping names, especially drag scripts:
   - `Hypersonic Drag Estimation*.py`
   - `Supersonic Drag Estimation*.py`
   - `Merged Drag to Mach*.py`
   - `Input Altitude Merged Drag*.py`
   - `Sensitivity Study Drag.py`

3. Several scripts depend on files by bare filename. Examples:
   - `density_velocity_database.csv`
   - `mach_fixed_altitude_output.csv`
   - `turbojet_multidesign_thrust_maps.npz`
   - `ramjet_multidesign_thrust_maps.npz`

   This means data files should not be moved until code paths are patched or a project-root runner is introduced.

4. Exact duplicate content found:
   - `SectionImages/13.png`
   - `SectionImages/14.png`

5. `Variables.py` is empty and can probably be removed or replaced by a proper constants module.

## Recommended safe target structure

```text
DSE_Group31_Mach5/
├── README.md
├── docs/
│   ├── file_inventory.csv
│   ├── python_dependency_inventory.csv
│   └── restructuring_notes.md
├── src/
│   ├── common/              # ISA atmosphere, constants, shared utilities
│   ├── aerodynamics/        # drag and Mach-regime scripts
│   ├── propulsion/          # engine, ramjet, turbojet, scramjet, PyCycle wrappers
│   ├── thermal/             # heat flux, Reynolds, wall temperature
│   ├── sizing/              # initial/combined sizing, SHM
│   ├── tanks/               # cryogenic/kerosene tank models
│   ├── environment/         # boom and coastal exclusion maps
│   └── trade_offs/          # trade-off studies
├── data/
│   ├── atmosphere/
│   ├── thrust_maps/
│   └── results_tables/
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── generated_tables/
├── external/
│   ├── pycycle/
│   └── RegenCooling/
└── archive/
    └── legacy_unverified/
```

## Recommended phase order

### Phase 1 — safe cleanup

Do this before moving source files:
- remove `.git`, `.idea`, `.vscode`, `.ai`
- remove every `__pycache__`
- remove every `.pyc`
- keep all `.py`, `.csv`, `.npz`, `.pkl`, `.png`, `.pdf`
- add a README explaining how to run the main scripts

### Phase 2 — identify canonical scripts

Suggested canonical entry points:
- Mission/sizing: `combined_sizing.py`, `Initial_sizing.py`
- Atmosphere utilities: `isa_atmosphere.py`
- Propulsion/thrust maps: `better_profile.py`, `ramjet_revised.py`, `turbojet_pycycle_wrapper.py`
- Thermal: `Heat_temps_turbulent.py`, `Mission_Heat_flux_Laminar.py`, `Radii_alt_graphing.py`
- Aerodynamics: choose one canonical merged drag script from the overlapping `Merged Drag...` / `Input Altitude...` group

### Phase 3 — move code only after path strategy is chosen

Recommended path strategy:

```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
```

Then replace plain filenames with explicit paths, for example:

```python
pd.read_csv(DATA_DIR / "density_velocity_database.csv")
np.load(DATA_DIR / "thrust_maps" / "turbojet_multidesign_thrust_maps.npz")
plt.savefig(OUTPUT_DIR / "figures" / "wall_temperature.pdf")
```

## Files generated with this report

- `file_inventory.csv`: all files, sizes, extensions, line counts where readable
- `python_dependency_inventory.csv`: imports and referenced data/output filenames per Python file
- `suggested_categories.csv`: rough target category for each file
