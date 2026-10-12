# Interactive HTML visualizations

MIXSim includes standalone browser-based replays for both networks used in the manuscript. The example scripts deliberately call the **same demand builders used by the numerical studies**, so the visualization examples do not maintain a second, potentially inconsistent set of demand definitions.

## Isolated intersection

```bash
python examples/isolated_intersection_visualization.py
```

The default replay uses 50% CAV penetration, the manuscript one-hour loading profile (two approaches, each peaking at 850 veh/h), and the two-phase 50 s + 50 s fixed-time signal plan. The output is:

```text
outputs/isolated_intersection_visualization/isolated_intersection.html
```

## 4x4 grid

```bash
python examples/grid_network_visualization.py
```

The default replay uses 50% CAV penetration, the manuscript grid demand (eight direct ODs peaking at 580 veh/h and sixteen route-choice ODs peaking at 250 veh/h), dynamic shortest-path routing updated every 30 s, and the capacity-weighted Max Pressure baseline used in the CAV-penetration study. The output is:

```text
outputs/grid_network_visualization/grid_network.html
```

Both HTML files are self-contained and can be opened directly in a modern browser. Use `--html-stride` to trade animation resolution for file size. `--quick` is intended only for CI/smoke testing and does not reproduce the full one-hour loading period.
