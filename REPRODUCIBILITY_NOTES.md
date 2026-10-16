# Reproducibility notes

This repository was rebuilt from the latest supplied MIXSim code archive and re-audited against the current manuscript configuration.

## Resolved configuration items

The following items are now aligned between code and manuscript:

- dynamic and stochastic routing use an instantaneous travel-time cost derived from the current composition-dependent cell states and triangular fundamental diagram, replacing the former occupied-storage penalty;

- stochastic routing in the CAV time-gap study uses `theta = 0.01`;
- the grid CAV-penetration and CAV time-gap studies use the class-weighted, capacity-weighted Max Pressure baseline with a 10 s decision interval, 20 s minimum green, and zero switching lost time;
- Max Pressure class weights are held fixed at `[3.0/0.9, 1.0]` during the CAV time-gap sweep;
- the isolated intersection uses two 50 s green phases, no yellow, and no all-red interval;
- the public API and documentation use **time gap**, not reaction time or following-time.

## Routing-cost revision and refreshed reference results

Dynamic and stochastic assignment now use an instantaneous travel-time cost derived from the current composition-dependent triangular fundamental diagram. The previous occupied-storage penalty has been removed from route choice and retained only as a diagnostic quantity. Because this change affects route decisions, all routing-sensitive grid studies were rerun and `reference_results/` was regenerated.

The system metrics continue to use the manuscript definitions:

```text
TTS = dt/3600 * sum_t [A(t) + B(t)]
PA  = max_t [A(t) + B(t)]
CT  = first post-demand state with A(t) + B(t) <= 1e-4 veh
```

The isolated-intersection study uses fixed routing and is numerically unchanged. Grid CAV-penetration, CAV time-gap, and signal-control comparison values in the reference CSVs correspond to the revised travel-time routing formulation.

## What was removed from the development archive

The public release intentionally excludes material that is not used by the revised manuscript, testing, or the tutorial:

- MFD/NMFD study scripts and old MFD-facing documentation;
- the former Green Wave comparison;
- experimental greedy/MPC controller classes not used in the manuscript;
- obsolete compatibility aliases;
- duplicate time-gap/headway plotting scripts;
- generated `outputs/` directories and caches;
- nested ZIP archives and stray development files;
- the old isolated-intersection plotting-only script with a hard-coded local path.

## Tutorial correction

The previous two-intersection example had only one admissible outgoing movement from each incoming approach, so the claimed routing demonstration was effectively trivial. The current tutorial has two admissible outgoing movements at each signalized node and exercises genuine OD-dependent route choice.
