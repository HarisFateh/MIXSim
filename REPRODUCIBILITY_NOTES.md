# Reproducibility notes

This repository was rebuilt from the latest supplied MIXSim code archive and re-audited against the current manuscript configuration.

## Resolved configuration items

The following items are now aligned between code and manuscript:

- stochastic routing in the CAV time-gap study uses `theta = 0.01`;
- the grid CAV-penetration and CAV time-gap studies use the class-weighted, capacity-weighted Max Pressure baseline with a 10 s decision interval, 20 s minimum green, and zero switching lost time;
- Max Pressure class weights are held fixed at `[3.0/0.9, 1.0]` during the CAV time-gap sweep;
- the isolated intersection uses two 50 s green phases, no yellow, and no all-red interval;
- the public API and documentation use **time gap**, not reaction time or following-time.

## Important metric audit performed for the release

The manuscript defines the system metrics using physical accumulation `A(t)` plus virtual-origin queues `B(t)`, and it specifies a common clearance tolerance of `1e-4 veh`. During full reruns, two older archived result tables were found to have been generated with different conventions:

1. The old isolated-intersection table used a clearance tolerance of `1e-6 veh`. TST and PA were unchanged, but the reported CT values were longer.
2. The old grid CAV-penetration table used **physical accumulation only** for TST and PA, even though the manuscript defines TST and PA using `A(t)+B(t)`.

The release does **not** preserve those inconsistent legacy conventions. The study scripts and `reference_results/` now use the manuscript's stated common definitions:

```text
TST = dt/3600 * sum_t [A(t) + B(t)]
PA  = max_t [A(t) + B(t)]
CT  = first post-demand state with A(t) + B(t) <= 1e-4 veh
```

Accordingly, the manuscript must use the corrected isolated-intersection CT values and corrected grid CAV-penetration TST/PA values stored in `reference_results/`. The CAV time-gap and mixed-autonomy/signal-control tables already follow the common metric definitions and did not require numerical changes.

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
