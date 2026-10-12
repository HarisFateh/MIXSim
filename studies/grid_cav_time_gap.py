#!/usr/bin/env python3
"""Reproduce the 4x4-grid CAV time-gap study."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from studies.common import (
    TIME_GAP_STOCHASTIC_THETA,
    V_FF_MPS,
    VEHICLE_LENGTH_M,
    build_grid_simulation,
    run_and_measure,
)


def mean_movement_saturation(sim, history, *, sample_every: int = 10) -> tuple[float, float]:
    vals = []
    for idx in range(0, len(history.commodity_state), max(1, int(sample_every))):
        state = history.commodity_state[idx]
        time_s = float(history.time_s[idx])
        for m_idx in range(sim.net.n_movements):
            vals.append(sim.saturation_controller.saturation_vps(sim.net, m_idx, time_s, state=state, simulator=sim))
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="outputs/grid_cav_time_gap")
    parser.add_argument("--stochastic-theta", type=float, default=TIME_GAP_STOCHASTIC_THETA)
    parser.add_argument("--max-clearance-min", type=float, default=180.0)
    parser.add_argument("--cav-time-gaps", default="3.0,2.4,1.8,1.2,0.6", help="Comma-separated CAV time gaps [s].")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gaps = [float(x.strip()) for x in args.cav_time_gaps.split(",") if x.strip()]
    if args.quick and len(gaps) > 3:
        gaps = [gaps[0], gaps[len(gaps)//2], gaps[-1]]
    demand_s = 2*60 if args.quick else 60*60
    max_clearance_s = 2*60 if args.quick else args.max_clearance_min*60

    rows = []
    for tau in gaps:
        sim, _, _ = build_grid_simulation(
            cav_share=0.45,
            cav_time_gap_s=tau,
            routing_assignment="stochastic",
            stochastic_theta=args.stochastic_theta,
            controller="capacity_weighted_max_pressure",
            demand_s=demand_s,
        )
        m, _, history = run_and_measure(sim, demand_s=demand_s, max_clearance_s=max_clearance_s)
        mean_sat, max_sat = mean_movement_saturation(sim, history)
        all_cav_sat_vph = 3600.0 * V_FF_MPS / (V_FF_MPS*tau + VEHICLE_LENGTH_M)
        rows.append({
            "tau_cav_s": tau,
            "all_cav_saturation_vph": all_cav_sat_vph,
            "mean_movement_saturation_vph": 3600.0*mean_sat,
            "max_movement_saturation_vph": 3600.0*max_sat,
            "tst_veh_h": m.tst_veh_h,
            "pa_veh": m.pa_veh,
            "clearance_time_min": m.clearance_time_min,
            "cleared": m.cleared,
            "entered_veh": m.entered_veh,
            "exited_veh": m.exited_veh,
            "final_system_veh": m.final_system_veh,
            "stochastic_theta": args.stochastic_theta,
            "max_abs_residual_veh": m.max_abs_residual_veh,
        })
        print(f"tau_CAV={tau:.2f}s TST={m.tst_veh_h:.2f}, PA={m.pa_veh:.2f}, CT={m.clearance_time_min:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    s = df.sort_values("tau_cav_s")
    ax.plot(s["tau_cav_s"], s["all_cav_saturation_vph"], marker="o", label="All-CAV theoretical")
    ax.plot(s["tau_cav_s"], s["mean_movement_saturation_vph"], marker="s", label="Mean movement")
    ax.set_xlabel(r"CAV time gap $\tau_{\mathrm{CAV}}$ [s]")
    ax.set_ylabel("Saturation flow [veh/h]")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_grid_cav_time_gap_saturation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
