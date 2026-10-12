#!/usr/bin/env python3
"""Two-intersection MIXSim tutorial.

This is the recommended starting point for GitHub users. It demonstrates:
OD-class commodities, route choice, two signalized nodes, cell transmission,
movement-specific HDV/CAV saturation, downstream supply, virtual-origin queues,
CSV diagnostics, and the optional HTML replay.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mixsim import (  # noqa: E402
    CTMSimulator,
    Demand,
    FixedSignalController,
    MovementCompositionSaturationController,
    VehicleClass,
    build_two_intersection_tutorial,
)
from mixsim.study_tools import run_publication_exports, network_timeseries  # noqa: E402


def add_tutorial_demand(demand: Demand, net, *, cav_share: float = 0.35) -> None:
    profile_main = Demand.smooth_trapezoidal_profile(0.0, 700.0, 0.0, 5*60, 15*60, 20*60)
    profile_side = Demand.smooth_trapezoidal_profile(0.0, 350.0, 0.0, 5*60, 15*60, 20*60)
    od_specs = [
        (1, "D1", profile_main),  # passes through both signals
        (1, "D2", profile_side),  # turns at N1
        (4, "D1", profile_side),  # enters at N1 and continues to N2
        (6, "D3", profile_side),  # side approach at N2
    ]
    for origin_link, destination, profile in od_specs:
        for class_name, share in (("HDV", 1.0-cav_share), ("CAV", cav_share)):
            demand.add_od_profile(
                origin_link=origin_link,
                destination_node=destination,
                class_name=class_name,
                profile=lambda t, p=profile, s=share: s*p(t),
                assignment="dynamic_shortest",
                network=net,
            )


def build_tutorial_sim(*, cav_share: float = 0.35):
    net = build_two_intersection_tutorial()
    classes = [
        VehicleClass("HDV", time_gap_s=3.0, vehicle_length_m=5.0, reference_share=1.0-cav_share),
        VehicleClass("CAV", time_gap_s=0.9, vehicle_length_m=5.0, reference_share=cav_share),
    ]
    demand = Demand(seed=7)
    add_tutorial_demand(demand, net, cav_share=cav_share)
    signal = FixedSignalController()
    sim = CTMSimulator(
        net, classes, demand, demand,
        signal_controller=signal,
        saturation_controller=MovementCompositionSaturationController(),
        dt=5.0,
    )
    return net, classes, demand, signal, sim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Short smoke-test run.")
    parser.add_argument("--skip-html", action="store_true", help="Skip the interactive HTML replay.")
    parser.add_argument("--output-dir", default="outputs/two_intersection_tutorial")
    parser.add_argument("--cav-share", type=float, default=0.35)
    args = parser.parse_args()

    net, _, demand, signal, sim = build_tutorial_sim(cav_share=args.cav_share)
    demand_s = 5*60 if args.quick else 20*60
    flush_s = 5*60 if args.quick else 20*60
    history = sim.run(int(round(demand_s/sim.dt)), flush_time_s=flush_s)
    ts = network_timeseries(sim, history)

    outputs = run_publication_exports(
        sim=sim,
        history=history,
        demand=demand,
        routing=demand,
        signal_controller=signal,
        output_dir=Path(args.output_dir),
        prefix="two_intersection",
        routing_sample_every=5 if args.quick else 2,
        html_stride=5 if args.quick else 2,
        max_links_per_page=8,
        include_sending_receiving=True,
        include_realized_diagnostic=True,
        local_fd_points=True,
        skip_html=args.skip_html,
    )

    print(f"nodes={len(net.nodes)}, links={len(net.links)}, movements={len(net.movements)}")
    print(f"commodities={sim.n_commodities}")
    print(f"entered={ts['cumulative_entered'].iloc[-1]:.3f} veh")
    print(f"exited={ts['cumulative_exited'].iloc[-1]:.3f} veh")
    print(f"max |conservation residual|={outputs.max_abs_residual:.3e} veh")
    print(f"outputs: {outputs.output_dir}")


if __name__ == "__main__":
    main()
