#!/usr/bin/env python3
"""Reproduce the 27-case mixed-autonomy/signal-control interaction table."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from studies.common import build_grid_simulation, run_and_measure

CONTROLLERS = {
    "FT": ("fixed_time", "FT"),
    "DPFT": ("demand_proportional", "DPFT"),
    "MP": ("max_pressure", "MP"),
}


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Short smoke-test matrix.")
    parser.add_argument("--output-dir", default="outputs/grid_signal_control_comparison")
    parser.add_argument("--cav-shares", default="0.25,0.50,0.75")
    parser.add_argument("--cav-time-gaps", default="3.6,1.8,0.9")
    parser.add_argument("--max-clearance-min", type=float, default=180.0)
    parser.add_argument("--controllers", default="FT,DPFT,MP", help="Comma-separated subset of FT,DPFT,MP.")
    args = parser.parse_args()

    shares = _parse_csv_floats(args.cav_shares)
    gaps = _parse_csv_floats(args.cav_time_gaps)
    if args.quick:
        shares = [shares[0]]
        gaps = [gaps[0], gaps[-1]] if len(gaps) > 1 else gaps

    requested_controllers = [x.strip().upper() for x in args.controllers.split(",") if x.strip()]
    unknown = [x for x in requested_controllers if x not in CONTROLLERS]
    if unknown:
        raise ValueError(f"unknown controller labels: {unknown}")
    selected_controllers = [CONTROLLERS[x] for x in requested_controllers]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    demand_s = 2*60 if args.quick else 60*60
    max_clearance_s = 2*60 if args.quick else args.max_clearance_min*60

    rows = []
    for tau in gaps:
        for share in shares:
            for controller_id, short_name in selected_controllers:
                sim, demand, _ = build_grid_simulation(
                    cav_share=share,
                    cav_time_gap_s=tau,
                    routing_assignment="dynamic_shortest",
                    controller=controller_id,
                    demand_s=demand_s,
                )
                m, _, _ = run_and_measure(sim, demand_s=demand_s, max_clearance_s=max_clearance_s)
                rows.append({
                    "tau_cav_s": tau,
                    "cav_share_pct": 100*share,
                    "controller": short_name,
                    "tst_veh_h": m.tst_veh_h,
                    "pa_veh": m.pa_veh,
                    "clearance_time_min": m.clearance_time_min,
                    "cleared": m.cleared,
                    "entered_veh": m.entered_veh,
                    "exited_veh": m.exited_veh,
                    "max_abs_residual_veh": m.max_abs_residual_veh,
            "final_system_veh": m.final_system_veh,
                })
                print(f"tau={tau:.1f}s CAV={100*share:.0f}% {short_name}: TST={m.tst_veh_h:.2f}, PA={m.pa_veh:.2f}, CT={m.clearance_time_min:.2f}")

    df = pd.DataFrame(rows)
    df["tst_reduction_vs_ft_pct"] = float("nan")
    for (_, _), block in df.groupby(["tau_cav_s", "cav_share_pct"], sort=False):
        ft_rows = block.loc[block["controller"] == "FT", "tst_veh_h"]
        if ft_rows.empty:
            continue
        ft = float(ft_rows.iloc[0])
        for idx in block.index:
            df.loc[idx, "tst_reduction_vs_ft_pct"] = 100.0 * (ft - float(df.loc[idx, "tst_veh_h"])) / ft
    df.to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
