#!/usr/bin/env python3
"""Reproduce the isolated-intersection CAV-penetration study."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from studies.common import build_isolated_intersection_simulation, run_and_measure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="outputs/isolated_intersection_cav_penetration")
    parser.add_argument("--max-clearance-min", type=float, default=60.0)
    parser.add_argument("--cav-shares", default="0,0.25,0.50,0.75,1.0", help="Comma-separated CAV shares in [0,1].")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shares = [float(x.strip()) for x in args.cav_shares.split(",") if x.strip()]
    if args.quick:
        shares = shares[:2] if len(shares) > 1 else shares
    demand_s = 10*60 if args.quick else 60*60
    max_clearance_s = 10*60 if args.quick else args.max_clearance_min*60

    rows, curves = [], []
    for share in shares:
        sim = build_isolated_intersection_simulation(cav_share=share)
        m, ts, _ = run_and_measure(sim, demand_s=demand_s, max_clearance_s=max_clearance_s)
        rows.append({
            "cav_share_pct": 100*share,
            "tts_veh_h": m.tts_veh_h,
            "pa_veh": m.pa_veh,
            "clearance_time_min": m.clearance_time_min,
            "cleared": m.cleared,
            "entered_veh": m.entered_veh,
            "exited_veh": m.exited_veh,
            "max_abs_residual_veh": m.max_abs_residual_veh,
            "final_system_veh": m.final_system_veh,
        })
        c = ts[["time_min", "physical_vehicles", "virtual_queue"]].copy()
        c["total_system_veh"] = c["physical_vehicles"] + c["virtual_queue"]
        c["cav_share_pct"] = 100*share
        curves.append(c)
        print(f"CAV={100*share:.0f}% TTS={m.tts_veh_h:.2f}, PA={m.pa_veh:.2f}, CT={m.clearance_time_min:.2f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    all_curves = pd.concat(curves, ignore_index=True)
    all_curves.to_csv(out / "timeseries.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for share in shares:
        c = all_curves[all_curves["cav_share_pct"] == 100*share]
        ax.plot(c["time_min"], c["total_system_veh"], label=f"{100*share:.0f}% CAV")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Total system vehicles [veh]")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_single_cav_system_accumulation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
