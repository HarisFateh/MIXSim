#!/usr/bin/env python3
"""Validate generated full-study summaries against manuscript reference CSVs.

Run the full experiments first. This script intentionally compares values only
at the precision reported in the manuscript and separately enforces clearance
and conservation checks where those diagnostics are available.
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STUDIES = {
    "isolated": {
        "actual": ROOT / "outputs/isolated_intersection_cav_penetration/summary.csv",
        "reference": ROOT / "reference_results/isolated_cav_penetration.csv",
        "keys": ["cav_share_pct"],
        "columns": ["tts_veh_h", "pa_veh", "clearance_time_min"],
    },
    "grid_penetration": {
        "actual": ROOT / "outputs/grid_cav_penetration/summary.csv",
        "reference": ROOT / "reference_results/grid_cav_penetration.csv",
        "keys": ["cav_share_pct"],
        "columns": ["tts_veh_h", "pa_veh", "clearance_time_min"],
    },
    "time_gap": {
        "actual": ROOT / "outputs/grid_cav_time_gap/summary.csv",
        "reference": ROOT / "reference_results/grid_cav_time_gap.csv",
        "keys": ["tau_cav_s"],
        "columns": [
            "all_cav_saturation_vph",
            "mean_movement_saturation_vph",
            "tts_veh_h",
            "pa_veh",
            "clearance_time_min",
        ],
    },
    "signal_control": {
        "actual": ROOT / "outputs/grid_signal_control_comparison/summary.csv",
        "reference": ROOT / "reference_results/grid_signal_control_comparison.csv",
        "keys": ["tau_cav_s", "cav_share_pct", "controller"],
        "columns": [
            "tts_veh_h",
            "tts_reduction_vs_ft_pct",
            "pa_veh",
            "clearance_time_min",
        ],
    },
}


def _norm_key_value(v):
    if isinstance(v, float):
        return round(v, 10)
    return v


def validate_one(name: str, spec: dict) -> list[str]:
    actual_path = spec["actual"]
    ref_path = spec["reference"]
    errors: list[str] = []
    if not actual_path.exists():
        return [f"{name}: missing generated file {actual_path.relative_to(ROOT)}"]
    actual = pd.read_csv(actual_path)
    ref = pd.read_csv(ref_path)

    keys = spec["keys"]
    cols = spec["columns"]
    for col in keys + cols:
        if col not in actual.columns:
            errors.append(f"{name}: actual summary missing column {col}")
        if col not in ref.columns:
            errors.append(f"{name}: reference summary missing column {col}")
    if errors:
        return errors

    def indexed(df: pd.DataFrame):
        out = {}
        for _, row in df.iterrows():
            key = tuple(_norm_key_value(row[k]) for k in keys)
            out[key] = row
        return out

    a = indexed(actual)
    r = indexed(ref)
    if set(a) != set(r):
        errors.append(f"{name}: case keys differ; actual={sorted(a)} reference={sorted(r)}")
        return errors

    for key in sorted(r, key=str):
        for col in cols:
            av = float(a[key][col])
            rv = float(r[key][col])
            if round(av, 2) != round(rv, 2):
                errors.append(
                    f"{name} {key} {col}: actual={av:.8f}, reference={rv:.8f}"
                )

    if "cleared" in actual.columns:
        uncleared = actual.loc[~actual["cleared"].astype(bool)]
        if not uncleared.empty:
            errors.append(f"{name}: {len(uncleared)} reported cases did not clear")
    if "max_abs_residual_veh" in actual.columns:
        worst = float(actual["max_abs_residual_veh"].abs().max())
        if worst >= 1e-8:
            errors.append(f"{name}: max conservation residual {worst:.3e} >= 1e-8 veh")

    return errors


def main() -> int:
    all_errors: list[str] = []
    for name, spec in STUDIES.items():
        errs = validate_one(name, spec)
        if errs:
            print(f"[FAIL] {name}")
            for e in errs:
                print(f"  - {e}")
            all_errors.extend(errs)
        else:
            print(f"[PASS] {name}")
    if all_errors:
        print(f"\nValidation failed with {len(all_errors)} issue(s).")
        return 1
    print("\nAll generated study summaries match the reference results at manuscript precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
