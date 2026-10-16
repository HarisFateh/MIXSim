# Reproducing the manuscript results

Run all commands from the repository root after installing MIXSim. The study scripts write generated files to `outputs/`, which is intentionally excluded from version control.

## Isolated-intersection CAV penetration

```bash
python studies/isolated_intersection_cav_penetration.py
```

CAV shares are 0, 25, 50, 75, and 100%. The signal plan has two 50 s green phases with no yellow or all-red interval. System metrics use physical accumulation plus virtual-origin queues (`A+B`) and the common `1e-4 veh` clearance threshold.


## Routing cost used by dynamic and stochastic assignment

Dynamic and stochastic assignment use a current-state travel-time estimate. For each nonempty cell, MIXSim evaluates the local triangular fundamental diagram and sets the instantaneous cell speed to `u = q/rho`; empty cells use free-flow speed. The link cost is the sum of cell travel times `dx/max(u, epsilon_u)`, with `epsilon_u = 0.5 m/s` as a numerical lower bound. Consequently, the route cost is measured in seconds rather than being an abstract congestion penalty.

## 4x4-grid CAV penetration

```bash
python studies/grid_cav_penetration.py
```

The eight direct ODs use fixed shortest paths. The 16 route-choice ODs use dynamic shortest paths updated every 30 s. Signal control is the class-weighted, capacity-weighted Max Pressure baseline with a 10 s decision interval, 20 s minimum green, and zero switching lost time.

## 4x4-grid CAV time-gap study

```bash
python studies/grid_cav_time_gap.py
```

CAV share is 45%; CAV time gaps are 3.0, 2.4, 1.8, 1.2, and 0.6 s. Route-choice ODs use stochastic assignment with `theta = 0.01`. The Max Pressure class weights remain fixed at `[3.0/0.9, 1.0]` across the sweep.

## Mixed autonomy and signal-control comparison

```bash
python studies/grid_signal_control_comparison.py
```

The full matrix contains 3 CAV shares × 3 CAV time gaps × 3 signal controllers = 27 cases. FT and DPFT use 60 s cycles; MP uses a 10 s decision interval, 20 s minimum green, and no switching lost time.

For a faster complete table run, the safety clearance horizon may be reduced to 60 min because every reported case clears before 56 min:

```bash
python studies/grid_signal_control_comparison.py --max-clearance-min 60
```

This does not change the reported metrics because metrics are evaluated only through the first cleared state.

## Manuscript figures

```bash
python studies/generate_demand_profiles.py
python studies/figures/composition_dependent_fd.py
python studies/figures/fd_sensitivity.py
```

## Validate generated results

After the four full numerical studies have been run:

```bash
python tools/validate_reference_results.py
```

The validator compares the generated summaries against `reference_results/` at the precision reported in the manuscript. It also verifies that all reported cases clear and that conservation residuals remain below `1e-8 veh`.
