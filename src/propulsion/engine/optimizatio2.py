"""
find_common_geometries.py
--------------------------
Reads all sweep-result CSV files in a folder (searched recursively), finds
geometry configurations that appear in every file (excluding A1_m2), then
ranks them by how closely their A1_m2 values match across files
(smallest spread = best match).

Geometry key columns: A2_m2, A3_m2, A6_m2, L_comb_m, phi, A6_frac
A1_m2 is excluded from the geometry key (it varies per flight condition).

Usage:
    python find_common_geometries.py                   # searches the folder the script lives in
    python find_common_geometries.py path/to/folder    # specify any folder
"""

import sys
import os
import glob
import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────

# Columns that define a unique geometry (A1 intentionally excluded)
GEOMETRY_COLS = ["A2_m2", "A3_m2", "A6_m2", "L_comb_m", "phi", "A6_frac"]

# Column whose value we want to compare across files to rank similarity
MATCH_COL = "A1_m2"

# ── Helpers ──────────────────────────────────────────────────────────────────

def round_key(df: pd.DataFrame, cols: list, decimals: int = 9) -> pd.Series:
    """Return a tuple key per row built from rounded geometry values."""
    rounded = df[cols].round(decimals)
    return rounded.apply(lambda row: tuple(row), axis=1)


def load_files(folder: str) -> list:
    """
    Load all CSV files in *folder* (non-recursive) whose columns include
    the expected geometry columns. Returns (filename, DataFrame) pairs.
    Also tries one level of subdirectories if nothing is found at the top level.
    """
    # First try the exact folder given
    pattern = os.path.join(folder, "*.csv")
    paths = sorted(glob.glob(pattern))

    # If nothing found, search recursively one level deeper
    if not paths:
        pattern_rec = os.path.join(folder, "**", "*.csv")
        paths = sorted(glob.glob(pattern_rec, recursive=True))

    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in '{folder}' (including subdirectories).\n"
            f"  Tip: pass the folder that contains your sweepresults_*.csv files as an argument:\n"
            f"       python find_common_geometries.py C:\\path\\to\\csv\\folder"
        )

    # Filter to only files that actually have the expected columns
    valid = []
    skipped = []
    for p in paths:
        try:
            header = pd.read_csv(p, nrows=0)
            has_geom = all(c in header.columns for c in GEOMETRY_COLS)
            has_a1   = MATCH_COL in header.columns
            if has_geom and has_a1:
                df = pd.read_csv(p)
                valid.append((os.path.basename(p), df))
                print(f"  Loaded  {os.path.basename(p):50s}  ({len(df):>5} rows)")
            else:
                skipped.append(os.path.basename(p))
        except Exception as e:
            skipped.append(f"{os.path.basename(p)} ({e})")

    if skipped:
        print(f"  Skipped (missing expected columns): {', '.join(skipped)}")

    if not valid:
        raise ValueError(
            "None of the CSV files found contain the expected geometry columns.\n"
            f"  Expected: {GEOMETRY_COLS + [MATCH_COL]}"
        )

    return valid


# ── Main logic ────────────────────────────────────────────────────────────────

def main(folder: str) -> pd.DataFrame:
    print(f"\nSearching for CSV files in: {os.path.abspath(folder)}\n")
    file_data = load_files(folder)

    if len(file_data) < 2:
        raise ValueError(
            f"Need at least 2 matching CSV files, but only found {len(file_data)}."
        )

    # For each file, build a dict: geometry_key → list of A1 values
    file_geom_maps = []
    for fname, df in file_data:
        key_series = round_key(df, GEOMETRY_COLS)
        geom_map: dict = {}
        for key, a1 in zip(key_series, df[MATCH_COL]):
            geom_map.setdefault(key, []).append(a1)
        file_geom_maps.append(geom_map)

    # Intersect geometry keys across all files
    common_keys = set(file_geom_maps[0].keys())
    for gm in file_geom_maps[1:]:
        common_keys &= set(gm.keys())

    print(f"\nGeometries appearing in all {len(file_data)} files: {len(common_keys)}\n")

    if not common_keys:
        print("No common geometries found.")
        return pd.DataFrame()

    # For each common geometry, compute A1 spread (std dev across files)
    records = []
    for key in common_keys:
        per_file_a1 = [np.mean(gm[key]) for gm in file_geom_maps]
        a1_mean  = float(np.mean(per_file_a1))
        a1_std   = float(np.std(per_file_a1, ddof=0))
        a1_range = float(np.max(per_file_a1) - np.min(per_file_a1))

        row = dict(zip(GEOMETRY_COLS, key))
        row["A1_mean_across_files"]  = round(a1_mean,  6)
        row["A1_std_across_files"]   = round(a1_std,   6)
        row["A1_range_across_files"] = round(a1_range, 6)

        for (fname, _), a1_val in zip(file_data, per_file_a1):
            col_name = f"A1_{fname.replace('.csv', '')}"
            row[col_name] = round(a1_val, 6)

        records.append(row)

    result = pd.DataFrame(records)

    # Sort: smallest A1 std = closest match across files → rank 1
    result = result.sort_values("A1_std_across_files").reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Default: the folder this script lives in (not the working directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = sys.argv[1] if len(sys.argv) > 1 else script_dir

    df_result = main(folder)

    if df_result.empty:
        sys.exit(0)

    print("=" * 80)
    print("Common geometries ordered by closest A1_m2 match (smallest std first)")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.6f}".format)
    print(df_result.to_string(index=False))

    # Save CSV next to the script
    out_path = os.path.join(script_dir, "common_geometries.csv")
    df_result.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print(f"Total common geometries: {len(df_result)}")