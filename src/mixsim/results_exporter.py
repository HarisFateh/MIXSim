from __future__ import annotations

import csv
from pathlib import Path


class ResultsExporter:
    """CSV exporter for the corrected MIXSim simulator."""

    def __init__(self, simulator, history) -> None:
        self.sim = simulator
        self.history = history
        self.net = simulator.net
        self.raw_network = simulator.raw_network

    def export_all(self, prefix: str = "ctm") -> None:
        prefix_path = Path(prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        self.export_links(f"{prefix}_links.csv")
        self.export_cells(f"{prefix}_cells.csv")
        self.export_movements(f"{prefix}_movements.csv")
        self.export_entry_exit(f"{prefix}_entry_exit.csv")
        self.export_interior_boundary_flows(f"{prefix}_interior_boundary_flows.csv")
        self.export_residual(f"{prefix}_residual.csv")

    def export_links(self, filename: str) -> None:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "time_s", "link_id", "link_name", "from_node", "to_node",
                "vehicles", "density_ratio", "density_veh_km",
            ])
            for ti, time_s in enumerate(self.history.time_s):
                counts = self.history.link_vehicle_count[ti]
                ratios = self.history.link_density_ratio[ti]
                for link_id, link in self.raw_network.links.items():
                    vehicles = float(counts.get(link_id, 0.0))
                    length_km = link.length_m / 1000.0
                    w.writerow([
                        time_s,
                        link_id,
                        link.link_name or f"L{link_id}",
                        link.from_node,
                        link.to_node,
                        vehicles,
                        float(ratios.get(link_id, 0.0)),
                        vehicles / length_km if length_km > 0 else 0.0,
                    ])

    def export_cells(self, filename: str) -> None:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = [
                "time_s", "cell_id", "link_id", "index_on_link", "vehicles_total",
                "density_veh_km", "density_ratio",
            ]
            for name in self.sim.class_names:
                header.append(f"vehicles_{name}")
            w.writerow(header)
            for ti, time_s in enumerate(self.history.time_s):
                x_class = self.history.cell_state[ti]
                for cell_id in range(self.net.n_cells):
                    values = x_class[cell_id]
                    total = float(values.sum())
                    length_m = float(self.net.cell_length_m[cell_id])
                    storage = float(self.net.cell_storage_veh[cell_id])
                    w.writerow([
                        time_s,
                        cell_id,
                        int(self.net.cell_link_id[cell_id]),
                        int(self.net.cell_index_on_link[cell_id]),
                        total,
                        total / length_m * 1000.0 if length_m > 0 else 0.0,
                        total / storage if storage > 0 else 0.0,
                        *[float(v) for v in values],
                    ])

    def export_movements(self, filename: str) -> None:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "time_s", "movement_id", "movement_name", "from_link", "to_link",
                "flow_veh_step", "flow_vph",
            ])
            dt = float(self.sim.dt)
            for ti, time_s in enumerate(self.history.time_s):
                flows = self.history.movement_flow[ti]
                for mid, mv in self.raw_network.movements.items():
                    step_flow = float(flows.get(mid, 0.0))
                    w.writerow([
                        time_s,
                        mid,
                        mv.movement_name,
                        mv.from_link,
                        mv.to_link,
                        step_flow,
                        step_flow * 3600.0 / dt,
                    ])

    def export_entry_exit(self, filename: str) -> None:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "time_s", "link_id", "entry_flow_veh_step", "exit_flow_veh_step",
                "entry_flow_vph", "exit_flow_vph",
            ])
            dt = float(self.sim.dt)
            for ti, time_s in enumerate(self.history.time_s):
                entry = self.history.entry_flow[ti]
                exit_ = self.history.exit_flow[ti]
                link_ids = sorted(set(entry.keys()) | set(exit_.keys()) | set(self.raw_network.links.keys()))
                for lid in link_ids:
                    e = float(entry.get(lid, 0.0))
                    x = float(exit_.get(lid, 0.0))
                    w.writerow([time_s, lid, e, x, e * 3600.0 / dt, x * 3600.0 / dt])

    def export_interior_boundary_flows(self, filename: str) -> None:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "time_s", "time_min", "link_id", "boundary_index",
                "upstream_cell_id", "downstream_cell_id",
                "upstream_density_veh_km", "downstream_density_veh_km",
                "flow_veh_step", "flow_vph",
            ])
            records_by_time = getattr(self.history, "interior_flow_records", [])
            for ti, time_s in enumerate(self.history.time_s):
                records = records_by_time[ti] if ti < len(records_by_time) else []
                for r in records:
                    w.writerow([
                        time_s,
                        float(time_s) / 60.0,
                        r.get("link_id"),
                        r.get("boundary_index"),
                        r.get("upstream_cell_id"),
                        r.get("downstream_cell_id"),
                        r.get("upstream_density_veh_km"),
                        r.get("downstream_density_veh_km"),
                        r.get("flow_veh_step"),
                        r.get("flow_vph"),
                    ])

    def export_residual(self, filename: str) -> None:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "conservation_residual"])
            for time_s, res in zip(self.history.time_s, self.history.residual):
                w.writerow([time_s, float(res)])
