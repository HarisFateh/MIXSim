import numpy as np


def test_state_dependent_routing_cost_is_instantaneous_travel_time():
    from mixsim.travel_time import TravelTimeEstimator
    from mixsim import VehicleClass, build_single_intersection

    net = build_single_intersection()
    link = net.links[1]
    classes = [
        VehicleClass("HDV", time_gap_s=3.0, vehicle_length_m=5.0),
        VehicleClass("CAV", time_gap_s=0.9, vehicle_length_m=5.0),
    ]
    est = TravelTimeEstimator(min_speed_mps=0.5)

    tff = sum(c.length_m / c.free_flow_speed_mps for c in link.cells)
    assert np.isclose(est.link_travel_time(link, None), tff)

    n_cells = max(c.cell_id for l in net.links.values() for c in l.cells) + 1
    x = np.zeros((n_cells, 2))
    class_index = np.array([0, 1], dtype=int)

    # At low density, the triangular FD is on its free-flow branch, so the
    # instantaneous travel-time cost remains the free-flow travel time.
    for cell in link.cells:
        x[cell.cell_id, 0] = 1.0
    assert np.isclose(
        est.link_travel_time(
            link,
            x,
            vehicle_classes=classes,
            commodity_class_index=class_index,
        ),
        tff,
    )

    # At 50% jam density with all HDVs, the state is congested.  The current
    # FD speed is u = w (rho_jam-rho) / rho = w, with w = 5/3 m/s.
    x[:] = 0.0
    for cell in link.cells:
        storage = cell.jam_density_vpm * cell.length_m
        x[cell.cell_id, 0] = 0.5 * storage
    expected_speed = 5.0 / 3.0
    expected_tt = sum(c.length_m / expected_speed for c in link.cells)
    assert np.isclose(
        est.link_travel_time(
            link,
            x,
            vehicle_classes=classes,
            commodity_class_index=class_index,
        ),
        expected_tt,
    )

    # The same density with all CAVs has a larger backward wave speed and
    # therefore a lower instantaneous travel-time cost.
    x[:] = 0.0
    for cell in link.cells:
        storage = cell.jam_density_vpm * cell.length_m
        x[cell.cell_id, 1] = 0.5 * storage
    cav_tt = est.link_travel_time(
        link,
        x,
        vehicle_classes=classes,
        commodity_class_index=class_index,
    )
    assert cav_tt < expected_tt

def test_fixed_shortest_dynamic_and_stochastic_assignment_available():
    from mixsim import Demand, VehicleClass, build_two_intersection_tutorial
    net = build_two_intersection_tutorial()
    classes = [VehicleClass("HDV", 3.0), VehicleClass("CAV", 0.9)]
    d = Demand(seed=1)
    d.add_od_profile(origin_link=1, destination_node="D1", class_name="HDV", profile=Demand.constant_profile(0), assignment="fixed_shortest", network=net)
    d.add_od_profile(origin_link=1, destination_node="D2", class_name="CAV", profile=Demand.constant_profile(0), assignment="dynamic_shortest", network=net)
    d.add_od_profile(origin_link=4, destination_node="D1", class_name="HDV", profile=Demand.constant_profile(0), assignment="stochastic", stochastic_theta=0.01, network=net)
    d.prepare_commodities(net, classes)
    d.update_assignment(net, classes, 0.0, None)
    for c in d.commodities:
        routes = d.current_routes_by_commodity[(c.origin_link, c.commodity_id)]
        if net.links[c.origin_link].to_node != c.destination_node:
            assert abs(sum(routes.values()) - 1.0) < 1e-12


def test_partial_green_overlap_is_exact():
    from mixsim import FixedSignalController, Network, NodeType, TurnType
    net = Network()
    net.add_node(NodeType.ORIGIN, 0, 0, "O")
    n = net.add_node(NodeType.SIGNALIZED, 100, 0, "N")
    net.add_node(NodeType.DESTINATION, 200, 0, "D")
    l1 = net.add_link("O", "N", length_m=100.0, n_cells=1)
    l2 = net.add_link("N", "D", length_m=100.0, n_cells=1)
    m = net.add_movement("m", n, l1, l2, TurnType.STRAIGHT, saturation_vps=1.0)
    net.add_phase(n, "p", [m], green=3.0, yellow=2.0, all_red=0.0)
    comp = net.compile(5.0, max_wave_speed_mps=0.0)
    ctrl = FixedSignalController()
    assert abs(ctrl.movement_green_duration(comp, 0, 0.0, 5.0) - 3.0) < 1e-12


def test_movement_saturation_uses_routing_weighted_composition():
    from mixsim import (
        CTMSimulator, Demand, FixedSignalController, MovementCompositionSaturationController,
        Network, NodeType, TurnType, VehicleClass,
    )
    net = Network()
    net.add_node(NodeType.ORIGIN, 0, 0, "O")
    n = net.add_node(NodeType.SIGNALIZED, 100, 0, "N")
    net.add_node(NodeType.DESTINATION, 200, 50, "D_hdv")
    net.add_node(NodeType.DESTINATION, 200, -50, "D_cav")
    l1 = net.add_link("O", "N", length_m=100.0, n_cells=1)
    l2 = net.add_link("N", "D_hdv", length_m=100.0, n_cells=1)
    l3 = net.add_link("N", "D_cav", length_m=100.0, n_cells=1)
    m1 = net.add_movement("m1", n, l1, l2, TurnType.STRAIGHT)
    m2 = net.add_movement("m2", n, l1, l3, TurnType.STRAIGHT)
    net.add_phase(n, "both", [m1, m2], green=5.0, yellow=0.0, all_red=0.0)
    d = Demand(seed=1)
    d.add_od_profile(origin_link=1, destination_node="D_hdv", class_name="HDV", profile=Demand.constant_profile(0), assignment="fixed_shortest", network=net)
    d.add_od_profile(origin_link=1, destination_node="D_cav", class_name="CAV", profile=Demand.constant_profile(0), assignment="fixed_shortest", network=net)
    sim = CTMSimulator(net, [VehicleClass("HDV",3.0), VehicleClass("CAV",0.9)], d, d, FixedSignalController(), MovementCompositionSaturationController(), dt=5.0)
    d.update_assignment(net, sim.vehicle_classes, 0.0, sim.x)
    cell = sim.net.movement_from_cell[0]
    sim.x[cell, 0] = 10.0
    sim.x[cell, 1] = 10.0
    sat_h = sim.saturation_controller.saturation_vps(sim.net, 0, 0.0, state=sim.x, simulator=sim)
    sat_c = sim.saturation_controller.saturation_vps(sim.net, 1, 0.0, state=sim.x, simulator=sim)
    vf = sim.net.cell_vf_mps[cell]
    assert np.isclose(sat_h, vf/(vf*3.0+5.0))
    assert np.isclose(sat_c, vf/(vf*0.9+5.0))
    assert sat_c > sat_h


def test_max_pressure_weights_are_normalized_time_gaps():
    from mixsim import HeterogeneousMaxPressureController
    class S:
        time_gap_s = np.array([3.0, 0.9])
    w = HeterogeneousMaxPressureController()._weights(S())
    assert np.allclose(w, [3.0/0.9, 1.0])
