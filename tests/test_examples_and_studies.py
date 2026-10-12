from __future__ import annotations


def test_two_intersection_tutorial_has_real_route_choices_and_runs():
    from two_intersection_tutorial import build_tutorial_sim

    net, _, _, _, sim = build_tutorial_sim(cav_share=0.35)
    assert len(net.links) == 7
    assert len(net.movements) == 8
    hist = sim.run(24)
    assert max(abs(x) for x in hist.residual) < 1e-8

    outgoing = {}
    for movement in net.movements.values():
        outgoing.setdefault(movement.from_link, set()).add(movement.to_link)
    assert any(len(destinations) == 2 for destinations in outgoing.values())


def test_grid_demand_counts_and_assignments():
    from common import assignment_counts, build_grid_simulation

    sim, demand, _ = build_grid_simulation(
        cav_share=0.45,
        routing_assignment="dynamic_shortest",
        controller="fixed_time",
        demand_s=600.0,
    )
    assert sim.n_commodities == 48  # 24 ODs x 2 vehicle classes
    assert assignment_counts(demand) == {"fixed_shortest": 16, "dynamic_shortest": 32}


def test_time_gap_study_uses_stochastic_routing_and_fixed_control_weights():
    from common import TIME_GAP_STOCHASTIC_THETA, assignment_counts, build_grid_simulation

    sim, demand, signal = build_grid_simulation(
        cav_share=0.45,
        cav_time_gap_s=1.8,
        routing_assignment="stochastic",
        stochastic_theta=TIME_GAP_STOCHASTIC_THETA,
        controller="capacity_weighted_max_pressure",
        demand_s=600.0,
    )
    assert assignment_counts(demand) == {"fixed_shortest": 16, "stochastic": 32}
    assert list(signal.class_weights) == [3.0 / 0.9, 1.0]
    assert sim.time_gap_s.tolist() == [3.0, 1.8]


def test_signal_control_comparison_settings_match_manuscript():
    from common import build_grid_simulation

    _, _, dpft = build_grid_simulation(
        cav_share=0.5,
        cav_time_gap_s=0.9,
        controller="demand_proportional",
        demand_s=600.0,
    )
    _, _, mp = build_grid_simulation(
        cav_share=0.5,
        cav_time_gap_s=0.9,
        controller="max_pressure",
        demand_s=600.0,
    )
    assert dpft.cycle_s == 60.0
    assert min(dpft.phase_green_by_id.values()) >= 10.0 - 1e-12
    assert mp.decision_interval_s == 10.0
    assert mp.min_green_s == 20.0
    assert mp.lost_time_s == 0.0
    assert mp.capacity_weighted is False


def test_manuscript_configuration_and_requested_demand_totals():
    from common import (
        TIME_GAP_STOCHASTIC_THETA,
        _integrate_profile,
        build_grid_simulation,
        build_isolated_intersection_simulation,
    )

    assert TIME_GAP_STOCHASTIC_THETA == 0.01

    isolated = build_isolated_intersection_simulation(cav_share=0.5)
    phase_indices = next(iter(isolated.net.node_phases.values()))
    assert len(phase_indices) == 2
    assert [isolated.net.phase_green_s[i] for i in phase_indices] == [50.0, 50.0]
    assert all(isolated.net.phase_yellow_s[i] == 0.0 for i in phase_indices)
    assert all(isolated.net.phase_all_red_s[i] == 0.0 for i in phase_indices)

    isolated_requested = sum(
        _integrate_profile(profile.profile, horizon_s=3600.0, dt_s=5.0)
        for profile in isolated.demand.od_profiles
    )
    assert abs(isolated_requested - 1133.3333333333333) < 1e-6

    _, grid_demand, baseline = build_grid_simulation(
        cav_share=0.45,
        cav_time_gap_s=1.8,
        routing_assignment="stochastic",
        stochastic_theta=TIME_GAP_STOCHASTIC_THETA,
        controller="capacity_weighted_max_pressure",
        demand_s=3600.0,
    )
    assert baseline.decision_interval_s == 10.0
    assert baseline.min_green_s == 20.0
    assert baseline.lost_time_s == 0.0
    assert baseline.capacity_weighted is True
    assert list(baseline.class_weights) == [3.0 / 0.9, 1.0]

    grid_requested = sum(
        _integrate_profile(profile.profile, horizon_s=3600.0, dt_s=5.0)
        for profile in grid_demand.od_profiles
    )
    assert abs(grid_requested - 5760.0) < 1e-6


def test_routing_history_capture_is_explicitly_configurable():
    from common import build_grid_simulation

    sim_no_history, _, _ = build_grid_simulation(
        cav_share=0.5,
        controller="capacity_weighted_max_pressure",
        demand_s=60.0,
        capture_routing_history=False,
    )
    hist_no_history = sim_no_history.run(2)
    assert all(frame == [] for frame in hist_no_history.routing_probabilities)

    sim_with_history, _, _ = build_grid_simulation(
        cav_share=0.5,
        controller="capacity_weighted_max_pressure",
        demand_s=60.0,
        capture_routing_history=True,
    )
    hist_with_history = sim_with_history.run(2)
    assert len(hist_with_history.routing_probabilities) == len(hist_with_history.time_s)
