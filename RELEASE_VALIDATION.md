# Release validation — MIXSim v1.0.0

Validation date: 2026-10-12

This release was validated after the repository was reorganized into a publication-oriented academic structure. Public scripts are named by scientific purpose rather than by temporary development prefixes.

## Repository integrity

- `pytest -q`: **17 passed**.
- `python -m compileall -q src examples studies tools`: **passed**.
- Python wheel build (`mixsim-1.0.0-py3-none-any.whl`): **passed**.
- Public API version: **1.0.0**.
- Generated outputs, caches, build products, and virtual environments are excluded from version control.

## Public examples

The following public workflows were smoke-tested successfully:

- `examples/two_intersection_tutorial.py`;
- `examples/isolated_intersection_visualization.py`;
- `examples/grid_network_visualization.py`.

Both dedicated visualization examples were also run with their full default one-hour loading configurations. They generated valid standalone HTML files:

- isolated intersection: approximately 0.27 MB;
- 4x4 grid: approximately 5.3 MB.

The two HTML examples import the same demand constructors used by the numerical studies:

- `studies.common.add_isolated_intersection_demand`;
- `studies.common.add_grid_demand`.

The test suite separately verifies the requested one-hour demand totals:

- isolated intersection: **1133.333333 veh**;
- 4x4 grid: **5760 veh**.

## Manuscript study validation

All reported numerical cases were rerun during release validation and compared with `reference_results/` at the precision reported in the manuscript.

### Isolated-intersection CAV penetration

- 5 CAV-penetration cases validated.
- Maximum conservation residual across the validated summary: approximately **3.27e-11 veh**.
- All cases cleared.

### 4x4-grid CAV penetration

- 5 CAV-penetration cases validated.
- Maximum conservation residual: approximately **1.55e-9 veh**.
- All cases cleared.

### 4x4-grid CAV time-gap study

- 5 CAV time-gap cases validated.
- Maximum conservation residual: approximately **9.91e-10 veh**.
- All cases cleared.

### Mixed autonomy and signal-control comparison

- 27 combinations validated: 3 CAV shares × 3 CAV time gaps × 3 controllers.
- Maximum conservation residual: approximately **1.69e-9 veh**.
- All 27 cases cleared.

The reference validator reported:

```text
[PASS] isolated
[PASS] grid_penetration
[PASS] time_gap
[PASS] signal_control

All generated study summaries match the reference results at manuscript precision.
```

After the final demand-definition consolidation, representative full cases were rerun again and reproduced the validated values, including:

```text
Grid penetration, 50% CAV:
TST = 1092.85 veh·h, PA = 1786.81 veh, CT = 21.83 min

Grid time-gap study, tau_CAV = 1.8 s:
TST = 1739.36 veh·h, PA = 2468.54 veh, CT = 66.50 min

Signal-control comparison, 75% CAV, tau_CAV = 0.9 s, MP:
TST = 637.25 veh·h, PA = 1093.92 veh, CT = 11.50 min
```

## Continuous integration

`.github/workflows/ci.yml` tests Python 3.10, 3.11, and 3.12. The workflow installs the package, runs unit/regression tests, compiles public Python sources, generates quick HTML replays, and smoke-tests all manuscript study scripts and figure generators.

Full manuscript simulations are intentionally kept out of per-commit CI because they are substantially more expensive. They are covered by the release-validation procedure and `tools/validate_reference_results.py`.
