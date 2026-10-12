#!/usr/bin/env python3
"""Reproduce the 4x4-grid CAV-penetration study reported in the manuscript."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from studies.common import build_grid_simulation, run_and_measure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="outputs/grid_cav_penetration")
    parser.add_argument("--max-clearance-min", type=float, default=180.0)
    parser.add_argument("--cav-shares", default="0,0.25,0.50,0.75,1.0", help="Comma-separated CAV shares in [0,1].")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    demand_s = 10*60 if args.quick else 60*60
    max_clearance_s = 10*60 if args.quick else args.max_clearance_min*60

    shares = [float(x.strip()) for x in args.cav_shares.split(",") if x.strip()]
    if args.quick:
        shares = shares[:2] if len(shares) > 1 else shares

    rows = []
    for share in shares:
        sim, _, _ = build_grid_simulation(
            cav_share=share,
            cav_time_gap_s=0.9,
            routing_assignment="dynamic_shortest",
            controller="capacity_weighted_max_pressure",
            demand_s=demand_s,
        )
        m, _, _ = run_and_measure(sim, demand_s=demand_s, max_clearance_s=max_clearance_s)
        rows.append({
            "cav_share_pct": 100*share,
            "tst_veh_h": m.tst_veh_h,
            "pa_veh": m.pa_veh,
            "clearance_time_min": m.clearance_time_min,
            "cleared": m.cleared,
            "entered_veh": m.entered_veh,
            "exited_veh": m.exited_veh,
            "max_abs_residual_veh": m.max_abs_residual_veh,
            "final_system_veh": m.final_system_veh,
        })
        print(f"CAV={100*share:.0f}% TST={m.tst_veh_h:.2f}, PA={m.pa_veh:.2f}, CT={m.clearance_time_min:.2f}")

    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
