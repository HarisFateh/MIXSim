# Changelog

## 1.1.0 — travel-time routing revision

- replaced the heuristic occupied-storage routing penalty with an instantaneous state-based link travel-time estimate derived from the current composition-dependent triangular fundamental diagram;
- dynamic and stochastic route choice now use costs with physical units of seconds;
- retained free-flow costs for fixed shortest-path assignment;
- updated routing tests, documentation, manuscript mapping, and reference-result workflow.

## 1.0.0 — manuscript reproducibility release

- organized the public repository into `src/`, `examples/`, `studies/`, `tests/`, `tools/`, and `docs/`;
- added dedicated isolated-intersection and 4x4-grid HTML visualization examples that reuse the exact manuscript demand builders;
- removed development-oriented filename prefixes and renamed public scripts by scientific purpose;
- aligned stochastic-routing sensitivity with the manuscript (`theta = 0.01`);
- documented the held-fixed capacity-weighted Max Pressure baseline and isolated-intersection signal timing;
- standardized system metrics to total accumulation `A+B` and a `1e-4 veh` clearance threshold;
- validated the corrected isolated-intersection CT and grid CAV-penetration TTS/PA reference values;
- added configurable routing-history capture so batch studies avoid unnecessary visualization overhead while HTML examples retain routing playback;
- added multi-version CI, smoke tests for public scripts, and reference-result validation;
- retained only functionality used by the manuscript, public examples, and repository tests;
- removed obsolete MFD/NMFD, Green Wave, intermediate MPC/greedy, duplicate, generated, and cache artifacts.
