# Release validation — MIXSim v1.1.0

Validation date: 2026-10-14

This release updates the manuscript routing formulation and the public code so that dynamic and stochastic route choice use an instantaneous travel-time cost derived from the current MIXSim traffic state rather than the former occupied-storage penalty.

## Repository integrity

- `pytest -q`: **17 passed**.
- `python -m compileall -q src examples studies tools`: **passed**.
- Python wheel build (`mixsim-1.1.0-py3-none-any.whl`): **passed**.
- Public API version: **1.1.0**.
- Generated outputs, caches, build products, and virtual environments are excluded from version control.

## Routing-cost revision

For each nonempty cell, the current composition-dependent triangular fundamental diagram is used to obtain the instantaneous speed `u = q/rho`; empty cells use free-flow speed. The link routing cost is the sum of cell travel times `dx/max(u, epsilon_u)` and therefore has physical units of seconds. Fixed shortest paths continue to use free-flow travel times. Dynamic and stochastic routing use the state-dependent travel-time estimate and are refreshed every 30 s in the manuscript grid studies.

The implementation is in `src/mixsim/travel_time.py::TravelTimeEstimator` and is exercised by routing regression tests in `tests/test_routing_and_signals.py`.

## Manuscript study validation

The routing-sensitive grid studies were rerun after the routing-cost revision. The resulting reference values are stored in `reference_results/`.

### Isolated-intersection CAV penetration

The isolated-intersection study uses fixed routing and is therefore numerically unchanged. All five cases clear and satisfy conservation.

### 4x4-grid CAV penetration

All five dynamic-routing cases were rerun. Representative values are:

```text
0% CAV:   TTS = 2114.72 veh·h, PA = 2736.54 veh, CT = 45.58 min
50% CAV:  TTS = 1092.78 veh·h, PA = 1786.81 veh, CT = 21.92 min
100% CAV: TTS = 421.01 veh·h,  PA = 636.15 veh,  CT = 10.92 min
```

### 4x4-grid CAV time-gap study

All five stochastic-routing cases were rerun with `theta = 0.01 s^-1`. Representative values are:

```text
tau_CAV = 3.0 s: TTS = 2290.85 veh·h, PA = 2874.38 veh, CT = 78.17 min
tau_CAV = 1.8 s: TTS = 1752.82 veh·h, PA = 2477.85 veh, CT = 67.08 min
tau_CAV = 0.6 s: TTS = 1231.56 veh·h, PA = 1960.85 veh, CT = 56.42 min
```

Movement saturation is now recorded during the simulation step in which it is evaluated, so post-processing uses the actual historical saturation values rather than reconstructing them from a later routing state.

### Mixed autonomy and signal-control comparison

All 27 combinations (3 CAV shares × 3 CAV time gaps × 3 controllers) were rerun under the revised dynamic travel-time routing cost. All cases clear and satisfy conservation. For example:

```text
75% CAV, tau_CAV = 0.9 s:
FT:   TTS = 729.77 veh·h, PA = 1265.53 veh, CT = 8.42 min
DPFT: TTS = 664.97 veh·h, PA = 1145.54 veh, CT = 8.25 min
MP:   TTS = 637.53 veh·h, PA = 1093.92 veh, CT = 11.50 min
```

## Continuous integration

`.github/workflows/ci.yml` tests Python 3.10, 3.11, and 3.12. The workflow installs the package, runs unit/regression tests, compiles public Python sources, generates quick HTML replays, and smoke-tests the manuscript study scripts. Full manuscript simulations remain part of release validation rather than per-commit CI.
