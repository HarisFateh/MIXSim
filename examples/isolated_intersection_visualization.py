#!/usr/bin/env python3
"""Generate an interactive HTML replay for the manuscript isolated intersection.

The example uses the *same one-hour demand profile and signal plan* as the
isolated-intersection CAV-penetration study.  The default 50% CAV case is used
only to provide one representative mixed-traffic replay; it can be changed from
the command line.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mixsim.html_visual import HTMLVisualizer  # noqa: E402
from mixsim.study_tools import (  # noqa: E402
    downsample_history,
    history_through_clearance,
    network_timeseries,
)
from studies.common import (  # noqa: E402
    CLEARANCE_TOL_VEH,
    DEMAND_DURATION_S,
    build_isolated_intersection_simulation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an interactive HTML replay of the isolated-intersection study configuration."
    )
    parser.add_argument("--cav-share", type=float, default=0.50, help="CAV share in [0, 1].")
    parser.add_argument("--cav-time-gap", type=float, default=0.9, help="CAV time gap [s].")
    parser.add_argument("--html-stride", type=int, default=6, help="Keep one HTML frame every N CTM steps.")
    parser.add_argument("--clearance-min", type=float, default=30.0, help="Maximum post-demand simulation horizon [min].")
    parser.add_argument("--quick", action="store_true", help="Short smoke test; does not reproduce the full one-hour loading period.")
    parser.add_argument("--output-dir", default="outputs/isolated_intersection_visualization")
    args = parser.parse_args()

    if not 0.0 <= args.cav_share <= 1.0:
        raise ValueError("--cav-share must be between 0 and 1")

    sim = build_isolated_intersection_simulation(
        cav_share=args.cav_share,
        cav_time_gap_s=args.cav_time_gap,
        capture_routing_history=True,
    )
    demand_s = 5.0 * 60.0 if args.quick else DEMAND_DURATION_S
    clearance_s = 5.0 * 60.0 if args.quick else args.clearance_min * 60.0
    history = sim.run(int(round(demand_s / sim.dt)), flush_time_s=clearance_s)
    if not args.quick:
        history = history_through_clearance(
            sim,
            history,
            demand_end_s=DEMAND_DURATION_S,
            clearance_tol=CLEARANCE_TOL_VEH,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    network_timeseries(sim, history).to_csv(output_dir / "network_timeseries.csv", index=False)

    html_path = output_dir / "isolated_intersection.html"
    HTMLVisualizer(
        sim.raw_network,
        downsample_history(history, max(1, args.html_stride)),
        routing_controller=sim.routing,
        signal_controller=sim.signal_controller,
        vehicle_class_names=list(sim.class_names),
        dt=float(sim.dt),
    ).save(str(html_path))

    print(f"HTML replay: {html_path}")
    print("Demand source: studies.common.add_isolated_intersection_demand")
    print(f"CAV share: {100.0 * args.cav_share:.0f}%")


if __name__ == "__main__":
    main()
