from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mixsim import (  # noqa: E402
    CTMSimulator,
    Demand,
    DemandProportionalFixedTimeController,
    FixedSignalController,
    HeterogeneousMaxPressureController,
    MovementCompositionSaturationController,
    VehicleClass,
    build_grid_4x4,
    build_single_intersection,
)
from mixsim.shortest_path import FloydWarshallRouter  # noqa: E402
from mixsim.study_tools import IntervalRoutingController, network_timeseries  # noqa: E402

DT_S = 5.0
V_FF_MPS = 8.33
VEHICLE_LENGTH_M = 5.0
HDV_TIME_GAP_S = 3.0
DEFAULT_CAV_TIME_GAP_S = 0.9
DEMAND_DURATION_S = 60.0 * 60.0
CLEARANCE_TOL_VEH = 1.0e-4
MAX_CLEARANCE_S = 180.0 * 60.0
ROUTING_UPDATE_S = 30.0

ISOLATED_PEAK_DEMAND_VPH = 850.0
GRID_DIRECT_PEAK_DEMAND_VPH = 580.0
GRID_ROUTING_PEAK_DEMAND_VPH = 250.0
DEMAND_RISE_END_S = 20.0 * 60.0
DEMAND_PEAK_END_S = 40.0 * 60.0
DEMAND_FALL_END_S = 60.0 * 60.0

# Stochastic-routing sensitivity used by the CAV time-gap study.
# This value is aligned with the current manuscript and archived reference run.
TIME_GAP_STOCHASTIC_THETA = 0.01


def build_vehicle_classes(*, cav_share: float, cav_time_gap_s: float = DEFAULT_CAV_TIME_GAP_S) -> list[VehicleClass]:
    return [
        VehicleClass(
            "HDV",
            time_gap_s=HDV_TIME_GAP_S,
            vehicle_length_m=VEHICLE_LENGTH_M,
            reference_share=1.0 - float(cav_share),
        ),
        VehicleClass(
            "CAV",
            time_gap_s=float(cav_time_gap_s),
            vehicle_length_m=VEHICLE_LENGTH_M,
            reference_share=float(cav_share),
        ),
    ]


def isolated_demand_profile(*, scale: float = 1.0):
    """Return the one-hour isolated-intersection demand-rate profile."""
    return Demand.smooth_trapezoidal_profile(
        base_vph=0.0,
        peak_vph=ISOLATED_PEAK_DEMAND_VPH * float(scale),
        rise_start_s=0.0,
        peak_start_s=DEMAND_RISE_END_S,
        peak_end_s=DEMAND_PEAK_END_S,
        fall_end_s=DEMAND_FALL_END_S,
    )


def grid_direct_demand_profile(*, scale: float = 1.0):
    """Return the one-hour demand-rate profile for each direct grid OD."""
    return Demand.smooth_trapezoidal_profile(
        0.0,
        GRID_DIRECT_PEAK_DEMAND_VPH * float(scale),
        0.0,
        DEMAND_RISE_END_S,
        DEMAND_PEAK_END_S,
        DEMAND_FALL_END_S,
    )


def grid_routing_demand_profile(*, scale: float = 1.0):
    """Return the one-hour demand-rate profile for each route-choice grid OD."""
    return Demand.smooth_trapezoidal_profile(
        0.0,
        GRID_ROUTING_PEAK_DEMAND_VPH * float(scale),
        0.0,
        DEMAND_RISE_END_S,
        DEMAND_PEAK_END_S,
        DEMAND_FALL_END_S,
    )


def add_isolated_intersection_demand(demand: Demand, net, *, cav_share: float, scale: float = 1.0) -> None:
    """Add the manuscript isolated-intersection demand to ``demand``."""
    p = isolated_demand_profile(scale=scale)
    for origin_link, destination in [(1, "D3"), (2, "D4")]:
        demand.add_od_profile(
            origin_link=origin_link,
            destination_node=destination,
            class_name="HDV",
            profile=lambda t, p=p, s=1.0 - cav_share: s * p(t),
            assignment="fixed_shortest",
            network=net,
        )
        demand.add_od_profile(
            origin_link=origin_link,
            destination_node=destination,
            class_name="CAV",
            profile=lambda t, p=p, s=cav_share: s * p(t),
            assignment="fixed_shortest",
            network=net,
        )


def add_grid_demand(
    demand: Demand,
    net,
    *,
    cav_share: float,
    scale: float = 1.0,
    routing_assignment: str = "dynamic_shortest",
    stochastic_theta: float = 0.05,
) -> None:
    """Shared 4x4-grid demand used by all manuscript grid studies.

    Eight direct ODs peak at 580 veh/h and use fixed shortest paths.  Sixteen
    routing ODs peak at 250 veh/h and use the requested assignment rule.
    """
    origin_link_ids = [1, 6, 11, 16, 21, 26, 31, 36]
    direct_destination_by_origin = {
        1: "D2", 6: "D3", 11: "D6", 16: "D7",
        21: "D10", 26: "D11", 31: "D14", 36: "D15",
    }
    routing_destinations_by_origin = {
        1: ["D10", "D11"],
        6: ["D11", "D15"],
        11: ["D10", "D14"],
        16: ["D14", "D15"],
        21: ["D2", "D3"],
        26: ["D3", "D6"],
        31: ["D2", "D6"],
        36: ["D6", "D7"],
    }

    direct_profile = grid_direct_demand_profile(scale=scale)
    routing_profile = grid_routing_demand_profile(scale=scale)

    reachability = FloydWarshallRouter()
    reachability.build(net, {lid: float(link.length_m) for lid, link in net.links.items()})

    for origin_link in origin_link_ids:
        od_specs = [(direct_destination_by_origin[origin_link], direct_profile, "fixed_shortest")]
        od_specs.extend((d, routing_profile, routing_assignment) for d in routing_destinations_by_origin[origin_link])
        start_node = net.links[origin_link].to_node
        for destination_name, profile, assignment in od_specs:
            destination_node = net.node_name_to_id[destination_name]
            if not reachability.can_reach(start_node, destination_node):
                raise RuntimeError(f"{destination_name} is unreachable from L{origin_link}")
            for class_name, share in (("HDV", 1.0 - cav_share), ("CAV", cav_share)):
                demand.add_od_profile(
                    origin_link=origin_link,
                    destination_node=destination_name,
                    class_name=class_name,
                    profile=lambda t, p=profile, s=share: float(s) * p(t),
                    assignment=assignment,
                    stochastic_theta=float(stochastic_theta),
                    network=net,
                )


def build_isolated_intersection_simulation(
    *,
    cav_share: float,
    cav_time_gap_s: float = DEFAULT_CAV_TIME_GAP_S,
    capture_routing_history: bool = False,
) -> CTMSimulator:
    net = build_single_intersection()
    classes = build_vehicle_classes(cav_share=cav_share, cav_time_gap_s=cav_time_gap_s)
    demand = Demand(seed=1)
    add_isolated_intersection_demand(demand, net, cav_share=cav_share)
    return CTMSimulator(
        net,
        classes,
        demand,
        demand,
        signal_controller=FixedSignalController(),
        saturation_controller=MovementCompositionSaturationController(),
        dt=DT_S,
        capture_routing_history=capture_routing_history,
    )


def _integrate_profile(profile, *, horizon_s: float, dt_s: float = 60.0) -> float:
    total = 0.0
    n = int(math.ceil(float(horizon_s) / float(dt_s)))
    for step in range(n):
        t = step * dt_s
        step_s = min(dt_s, max(0.0, horizon_s - t))
        if step_s <= 0.0:
            break
        total += max(0.0, float(profile(t))) * step_s / 3600.0
    return total


def _movement_to_phase_id(net) -> dict[int, int]:
    out: dict[int, int] = {}
    for node in net.nodes.values():
        for phase in node.phases:
            for movement_id in phase.movement_ids:
                out[int(movement_id)] = int(phase.phase_id)
    return out


def _movement_id_by_from_to(net) -> dict[tuple[int, int], int]:
    return {(int(m.from_link), int(m.to_link)): int(m.movement_id) for m in net.movements.values()}


def demand_proportional_phase_greens(
    net,
    demand: Demand,
    *,
    demand_s: float = DEMAND_DURATION_S,
    cycle_s: float = 60.0,
    min_green_s: float = 10.0,
) -> dict[int, float]:
    """Static DPFT split used in the manuscript signal-control experiment."""
    router = FloydWarshallRouter()
    router.build(net, {lid: float(link.length_m) for lid, link in net.links.items()})
    movement_to_phase = _movement_to_phase_id(net)
    movement_by_edge = _movement_id_by_from_to(net)
    phase_demand = {
        int(phase.phase_id): 0.0
        for node in net.nodes.values()
        for phase in node.phases
    }

    def propagate(from_link: int, destination_node: int, amount: float, depth: int = 0) -> None:
        if amount <= 1.0e-12 or depth > 30:
            return
        node_id = int(net.links[from_link].to_node)
        if node_id == int(destination_node):
            return
        for to_link, probability in router.outgoing_probabilities(node_id, int(destination_node)).items():
            p = float(probability)
            if p <= 0.0:
                continue
            movement_id = movement_by_edge.get((int(from_link), int(to_link)))
            if movement_id is not None:
                phase_id = movement_to_phase.get(movement_id)
                if phase_id is not None:
                    phase_demand[phase_id] += amount * p
            propagate(int(to_link), int(destination_node), amount * p, depth + 1)

    for profile in demand.od_profiles:
        propagate(
            int(profile.origin_link),
            int(profile.destination_node),
            _integrate_profile(profile.profile, horizon_s=demand_s),
        )

    greens: dict[int, float] = {}
    for node in net.nodes.values():
        phase_ids = [int(p.phase_id) for p in node.phases]
        if not phase_ids:
            continue
        minimum_total = len(phase_ids) * float(min_green_s)
        if minimum_total > cycle_s + 1.0e-12:
            raise ValueError("cycle too short for the requested minimum green")
        remaining = float(cycle_s) - minimum_total
        weights = np.array([max(0.0, phase_demand.get(pid, 0.0)) for pid in phase_ids], dtype=float)
        if weights.sum() <= 1.0e-12:
            weights[:] = 1.0
        weights /= weights.sum()
        for pid, weight in zip(phase_ids, weights):
            greens[pid] = float(min_green_s + remaining * weight)
    return greens


def build_grid_simulation(
    *,
    cav_share: float,
    cav_time_gap_s: float = DEFAULT_CAV_TIME_GAP_S,
    routing_assignment: str = "dynamic_shortest",
    stochastic_theta: float = 0.05,
    controller: str = "fixed_time",
    demand_s: float = DEMAND_DURATION_S,
    routing_update_s: float = ROUTING_UPDATE_S,
    capture_routing_history: bool = False,
) -> tuple[CTMSimulator, Demand, object]:
    net = build_grid_4x4(include_turn_movements=True)
    classes = build_vehicle_classes(cav_share=cav_share, cav_time_gap_s=cav_time_gap_s)
    demand = Demand(seed=2)
    add_grid_demand(
        demand,
        net,
        cav_share=cav_share,
        routing_assignment=routing_assignment,
        stochastic_theta=stochastic_theta,
    )

    if controller == "fixed_time":
        signal = FixedSignalController()
    elif controller == "demand_proportional":
        signal = DemandProportionalFixedTimeController(
            cycle_s=60.0,
            phase_green_by_id=demand_proportional_phase_greens(net, demand, demand_s=demand_s),
            yellow_s=0.0,
            all_red_s=0.0,
        )
    elif controller == "max_pressure":
        signal = HeterogeneousMaxPressureController(
            decision_interval_s=10.0,
            min_green_s=20.0,
            lost_time_s=0.0,
            use_class_weights=True,
            capacity_weighted=False,
        )
    elif controller == "capacity_weighted_max_pressure":
        # Kept fixed across the time-gap sweep, matching the archived run that
        # generated the manuscript table. This avoids changing controller weights
        # while the traffic-model time gap is varied.
        signal = HeterogeneousMaxPressureController(
            decision_interval_s=10.0,
            min_green_s=20.0,
            lost_time_s=0.0,
            use_class_weights=True,
            capacity_weighted=True,
            class_weights=[HDV_TIME_GAP_S / DEFAULT_CAV_TIME_GAP_S, 1.0],
        )
    else:
        raise ValueError(f"unknown controller: {controller}")

    routing = IntervalRoutingController(demand, update_interval_s=float(routing_update_s))
    sim = CTMSimulator(
        net,
        classes,
        demand,
        routing,
        signal_controller=signal,
        saturation_controller=MovementCompositionSaturationController(),
        dt=DT_S,
        capture_routing_history=capture_routing_history,
    )
    return sim, demand, signal


@dataclass(frozen=True)
class Metrics:
    tst_veh_h: float
    pa_veh: float
    clearance_time_min: float
    cleared: bool
    entered_veh: float
    exited_veh: float
    max_abs_residual_veh: float
    final_system_veh: float


def run_and_measure(
    sim: CTMSimulator,
    *,
    demand_s: float = DEMAND_DURATION_S,
    max_clearance_s: float = MAX_CLEARANCE_S,
    clearance_tol: float = CLEARANCE_TOL_VEH,
) -> tuple[Metrics, pd.DataFrame, object]:
    history = sim.run(int(round(float(demand_s) / sim.dt)), flush_time_s=float(max_clearance_s))
    ts_full = network_timeseries(sim, history).copy()
    total = ts_full["physical_vehicles"].astype(float) + ts_full["virtual_queue"].astype(float)
    candidates = ts_full.index[(ts_full["time_s"].astype(float) >= demand_s) & (total <= clearance_tol)].tolist()
    if candidates:
        end_idx = int(candidates[0])
        cleared = True
    else:
        end_idx = len(ts_full) - 1
        cleared = False
    ts = ts_full.iloc[: end_idx + 1].copy()
    total = ts["physical_vehicles"].astype(float) + ts["virtual_queue"].astype(float)
    metrics = Metrics(
        tst_veh_h=float(total.sum() * sim.dt / 3600.0),
        pa_veh=float(total.max()),
        clearance_time_min=max(0.0, (float(ts["time_s"].iloc[-1]) - demand_s) / 60.0),
        cleared=cleared,
        entered_veh=float(ts["cumulative_entered"].iloc[-1]),
        exited_veh=float(ts["cumulative_exited"].iloc[-1]),
        max_abs_residual_veh=float(ts["conservation_residual"].abs().max()),
        final_system_veh=float(total.iloc[-1]),
    )
    return metrics, ts, history


def assignment_counts(demand: Demand) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in demand.od_profiles:
        out[p.assignment] = out.get(p.assignment, 0) + 1
    return out
