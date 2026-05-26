# Automatically make restructured source folders importable when running from the project root.

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
import sys
_ROOT = Path(__file__).resolve().parent
_CANDIDATES = [
    _ROOT / "src",
    _ROOT / "src" / "common",
    _ROOT / "src" / "aerodynamics" / "drag",
    _ROOT / "src" / "propulsion",
    _ROOT / "src" / "propulsion" / "engine",
    _ROOT / "src" / "thermal",
    _ROOT / "src" / "sizing",
    _ROOT / "src" / "tanks",
    _ROOT / "src" / "environment",
    _ROOT / "src" / "trade_offs",
    _ROOT / "external",
    _ROOT / "external" / "pycycle_examples",
]
for _p in _CANDIDATES:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
