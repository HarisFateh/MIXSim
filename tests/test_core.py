import numpy as np
import pytest


def test_public_api_is_intentional_and_small():
    import mixsim
    required = {
        "VehicleClass", "Network", "Demand", "CTMSimulator",
        "FixedSignalController", "DemandProportionalFixedTimeController",
        "HeterogeneousMaxPressureController", "MovementCompositionSaturationController",
        "build_single_intersection", "build_two_intersection_tutorial", "build_grid_4x4",
    }
    assert required.issubset(set(mixsim.__all__))
    for obsolete in [
        "GreenWaveSignalController", "WebsterSignalController",
        "CAVAwareGreedyController", "MPCMaxPressureController",
        "ClosedFormLocalMPCController", "QMaxSaturationController",
    ]:
        assert not hasattr(mixsim, obsolete)


def test_vehicle_class_time_gap_name():
    from mixsim import VehicleClass
    v = VehicleClass("CAV", time_gap_s=0.9, vehicle_length_m=5.0)
    assert v.time_gap_s == 0.9
    with pytest.raises(ValueError):
        VehicleClass("bad", time_gap_s=0.0)


def test_manuscript_grid_shape():
    from mixsim import build_grid_4x4
    net = build_grid_4x4(include_turn_movements=True)
    assert len(net.nodes) == 32
    assert len(net.links) == 40
    assert len(net.movements) == 64
    signalized = [n for n in net.nodes.values() if str(n.node_type.value).lower() == "signalized"]
    assert len(signalized) == 16


def test_default_jam_density_and_cfl_guard():
    from mixsim import CTMSimulator, Demand, VehicleClass, build_single_intersection
    net = build_single_intersection()
    demand = Demand()
    demand.add_od_profile(
        origin_link=1, destination_node="D3", class_name="HDV",
        profile=Demand.constant_profile(0.0), assignment="fixed_shortest", network=net,
    )
    vehicles = [VehicleClass("HDV", 3.0, 5.0), VehicleClass("CAV", 0.9, 5.0)]
    sim = CTMSimulator(net, vehicles, demand, demand, dt=5.0)
    assert abs(sim.net.cell_jam_density_vpm[0] - 0.2) < 1e-12

    demand2 = Demand()
    demand2.add_od_profile(
        origin_link=1, destination_node="D3", class_name="HDV",
        profile=Demand.constant_profile(0.0), assignment="fixed_shortest", network=net,
    )
    with pytest.raises(ValueError, match="CFL condition violated"):
        CTMSimulator(
            net,
            [VehicleClass("HDV", 3.0, 5.0), VehicleClass("CAV", 0.1, 5.0)],
            demand2, demand2, dt=5.0,
        )


def test_composition_dependent_fd_parameters_match_model_equations():
    from mixsim import CTMSimulator, Demand, VehicleClass, build_single_intersection
    net = build_single_intersection()
    demand = Demand()
    demand.add_od_profile(
        origin_link=1, destination_node="D3", class_name="HDV",
        profile=Demand.constant_profile(0.0), assignment="fixed_shortest", network=net,
    )
    vehicles = [
        VehicleClass("HDV", 3.0, 5.0, reference_share=0.5),
        VehicleClass("CAV", 0.9, 5.0, reference_share=0.5),
    ]
    sim = CTMSimulator(net, vehicles, demand, demand, dt=5.0)
    tau_bar = 0.5*3.0 + 0.5*0.9
    vf = 8.33
    expected_qmax = vf / (vf*tau_bar + 5.0)
    expected_w = 5.0 / tau_bar
    qmax, w = sim.effective_parameters()
    assert np.isclose(qmax[0], expected_qmax)
    assert np.isclose(w[0], expected_w)


def test_demand_units_and_conservative_small_run():
    from mixsim import CTMSimulator, Demand, FixedSignalController, MovementCompositionSaturationController, VehicleClass, build_single_intersection
    net = build_single_intersection()
    demand = Demand(seed=1)
    demand.add_od_profile(
        origin_link=1, destination_node="D3", class_name="HDV",
        profile=Demand.constant_profile(360.0), assignment="fixed_shortest", network=net,
    )
    demand.add_od_profile(
        origin_link=2, destination_node="D4", class_name="CAV",
        profile=Demand.constant_profile(360.0), assignment="fixed_shortest", network=net,
    )
    sim = CTMSimulator(
        net,
        [VehicleClass("HDV", 3.0, 5.0), VehicleClass("CAV", 0.9, 5.0)],
        demand, demand,
        FixedSignalController(), MovementCompositionSaturationController(), dt=5.0,
    )
    hist = sim.run(24)
    assert len(hist.time_s) >= 24
    assert max(abs(x) for x in hist.residual) < 1e-8
