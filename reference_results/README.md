# Reference results

These CSV files contain the validated manuscript reference values for MIXSim v1.1.0.

All TTS and PA values use total system accumulation `A(t)+B(t)`. Clearance time uses the common threshold `A(t)+B(t) <= 1e-4 veh`.

The routing-sensitive grid studies were regenerated after replacing the former occupied-storage routing penalty with the instantaneous state-based travel-time cost derived from the current composition-dependent triangular fundamental diagram. The isolated-intersection study uses fixed routing and is unchanged by this revision.

Use `tools/validate_reference_results.py` after running the full study scripts to compare generated summaries with these files at manuscript precision.
