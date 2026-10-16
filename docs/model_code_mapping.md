# Manuscript-to-code mapping

| Manuscript quantity / mechanism | Main code location |
|---|---|
| Vehicle classes and class time gap | `src/mixsim/vehicle_type.py::VehicleClass` |
| Cell composition | `src/mixsim/simulator.py::cell_class_shares` |
| Composition-dependent capacity and wave speed | `src/mixsim/simulator.py::effective_parameters` |
| Sending / receiving | `src/mixsim/simulator.py::sending`, `src/mixsim/simulator.py::receiving` |
| OD commodities and routing fractions | `src/mixsim/demand.py` |
| Instantaneous state-based travel-time routing cost | `src/mixsim/travel_time.py::TravelTimeEstimator` |
| Shortest paths | `src/mixsim/shortest_path.py::FloydWarshallRouter` |
| Movement composition and saturation | `src/mixsim/saturation_controller.py::MovementCompositionSaturationController` |
| Effective green duration and phase logic | `src/mixsim/signal_controller.py` |
| Fixed time | `FixedSignalController` |
| Demand-proportional fixed time | `DemandProportionalFixedTimeController` |
| Class-weighted Max Pressure | `HeterogeneousMaxPressureController` |
| Conservation update / virtual queues | `src/mixsim/simulator.py` |
| Isolated, grid, and tutorial networks | `src/mixsim/scenarios.py` |
| Manuscript demand and study configuration | `studies/common.py` |
| Interactive browser replay | `src/mixsim/html_visual.py` |
