# MIXSim

[![CI](https://github.com/HarisFateh/MIXSim/actions/workflows/ci.yml/badge.svg)](https://github.com/HarisFateh/MIXSim/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

MIXSim is a **multiclass first-order macroscopic traffic simulation framework** for mixed-autonomy signalized urban networks. Human-driven vehicles (HDVs) and Connected and Automated Vehicles (CAVs) are represented as separate classes within origin-destination commodities. Local class composition modifies both the triangular fundamental-diagram parameters and the saturation flow of signalized movements.

This repository contains the simulation core, reproducible manuscript studies, interactive HTML visualization examples, automated tests, documentation, and validated reference results.

![Conceptual Rendering](docs/assets/MIXSimNet.pdf)


## Installation

```bash
git clone https://github.com/HarisFateh/MIXSim.git
cd MIXSim
python -m pip install -e ".[dev]"
pytest -q
```

Python 3.10 or newer is required.

## Quick start

The two-intersection tutorial is the recommended first example:

```bash
python examples/two_intersection_tutorial.py --quick --skip-html
```

For the complete tutorial, including the standalone interactive HTML replay:

```bash
python examples/two_intersection_tutorial.py
```

The tutorial contains two signalized intersections with genuine OD-dependent route choices and demonstrates multiclass commodities, movement-specific saturation, virtual-origin queues, signal phases, routing fractions, conservation diagnostics, and browser-based playback.

See [docs/two_intersection_tutorial.md](docs/two_intersection_tutorial.md).

## Interactive HTML examples using the manuscript demand

Two dedicated visualization examples use the **same demand builders as the numerical studies**, avoiding duplicated or inconsistent demand definitions.

### Isolated intersection

```bash
python examples/isolated_intersection_visualization.py
```

This uses the manuscript one-hour isolated-intersection demand profile and signal plan. The generated replay is:

```text
outputs/isolated_intersection_visualization/isolated_intersection.html
```

### 4x4 grid

```bash
python examples/grid_network_visualization.py
```

This uses the manuscript one-hour grid OD demand, dynamic shortest-path routing, and the capacity-weighted Max Pressure baseline used in the CAV-penetration study. The generated replay is:

```text
outputs/grid_network_visualization/grid_network.html
```

See [docs/html_visualizations.md](docs/html_visualizations.md) for options and file-size guidance.

## Reproducible numerical studies

| Study | Script |
|---|---|
| Isolated intersection: CAV penetration | `studies/isolated_intersection_cav_penetration.py` |
| 4x4 grid: CAV penetration | `studies/grid_cav_penetration.py` |
| 4x4 grid: CAV time gap | `studies/grid_cav_time_gap.py` |
| 4x4 grid: mixed autonomy × signal control | `studies/grid_signal_control_comparison.py` |
| Demand-profile figures | `studies/generate_demand_profiles.py` |
| Fundamental-diagram figures | `studies/figures/composition_dependent_fd.py`, `studies/figures/fd_sensitivity.py` |

Quick smoke tests:

```bash
python studies/isolated_intersection_cav_penetration.py --quick
python studies/grid_cav_penetration.py --quick
python studies/grid_cav_time_gap.py --quick
python studies/grid_signal_control_comparison.py --quick
```

Full reproduction instructions are in [docs/reproduce_manuscript.md](docs/reproduce_manuscript.md).

## Metric convention

All reported system-level metrics use physical network accumulation `A(t)` plus virtual-origin queues `B(t)`:

```text
TST = dt/3600 * sum_t [A(t) + B(t)]
PA  = max_t [A(t) + B(t)]
CT  = first post-demand state with A(t) + B(t) <= 1e-4 veh
```

Validated numerical targets are stored in `reference_results/`.

## Validation

Continuous integration runs on Python 3.10, 3.11, and 3.12. Local fast validation is:

```bash
python -m pip install -e ".[dev]"
python -m pip check
pytest -q
python -m compileall -q src examples studies tools
python examples/isolated_intersection_visualization.py --quick --output-dir outputs/ci_isolated_visual
python examples/grid_network_visualization.py --quick --output-dir outputs/ci_grid_visual
```

After running the full numerical studies:

```bash
python tools/validate_reference_results.py
```

See [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md) for the release validation record and [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md) for reproducibility decisions.

## Repository structure

```text
src/mixsim/          simulation library
examples/            tutorials and interactive HTML visualization examples
studies/             reproducible manuscript studies and figure scripts
tests/               automated unit and regression tests
tools/               reference-result validation utilities
docs/                installation, tutorials, model mapping, and reproduction notes
reference_results/   validated manuscript table values
```

Generated outputs, caches, build products, and virtual environments are excluded by `.gitignore`.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). If MIXSim contributes to published work, please cite the associated manuscript and software release.

## License

MIXSim is released under the MIT License. See [LICENSE](LICENSE).
