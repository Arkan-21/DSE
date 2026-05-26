"""Shared path helpers for the restructured DSE project.

Use these instead of bare filenames so scripts continue to work no matter
which folder they are run from.
"""
from __future__ import annotations


# --- restructured-project import bootstrap ---
from pathlib import Path as _DSE_Path
import sys as _DSE_sys
_DSE_ROOT = next((p for p in _DSE_Path(__file__).resolve().parents if (p / "src").exists() and (p / "data").exists()), None)
if _DSE_ROOT is not None:
    for _DSE_p in [
        _DSE_ROOT / "src",
        _DSE_ROOT / "src" / "common",
        _DSE_ROOT / "src" / "aerodynamics" / "drag",
        _DSE_ROOT / "src" / "propulsion",
        _DSE_ROOT / "src" / "propulsion" / "engine",
        _DSE_ROOT / "src" / "thermal",
        _DSE_ROOT / "src" / "sizing",
        _DSE_ROOT / "src" / "tanks",
        _DSE_ROOT / "src" / "environment",
        _DSE_ROOT / "src" / "trade_offs",
        _DSE_ROOT / "external",
        _DSE_ROOT / "external" / "pycycle_examples",
    ]:
        if _DSE_p.exists() and str(_DSE_p) not in _DSE_sys.path:
            _DSE_sys.path.insert(0, str(_DSE_p))
# --- end bootstrap ---
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXTERNAL_DIR = PROJECT_ROOT / "external"

def data_file(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)

def thrust_map_file(filename: str) -> Path:
    return DATA_DIR / "thrust_maps" / filename

def output_file(*parts: str) -> Path:
    path = OUTPUT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def source_file(*parts: str) -> Path:
    return SRC_DIR.joinpath(*parts)

def project_file(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
