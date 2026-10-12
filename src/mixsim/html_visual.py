# html_visual.py

import json
from typing import Any

from .network import Network
from .simulator import SimulationHistory


class HTMLVisualizer:
    def __init__(
        self,
        network: Network,
        history: SimulationHistory,
        routing_controller: Any | None = None,
        signal_controller: Any | None = None,
        vehicle_class_names: list[str] | None = None,
        dt: float | None = None,
    ):
        self.network = network
        self.history = history
        self.routing_controller = routing_controller
        self.signal_controller = signal_controller
        self.vehicle_class_names = vehicle_class_names or self._infer_class_names()
        self.dt = dt if dt is not None else self._infer_dt()

    def _infer_dt(self) -> float:
        if len(self.history.time_s) > 1:
            return float(self.history.time_s[1] - self.history.time_s[0])
        return 1.0

    def _infer_class_names(self) -> list[str]:
        if self.routing_controller is not None:
            if hasattr(self.routing_controller, "class_names"):
                return list(self.routing_controller.class_names)

            if hasattr(self.routing_controller, "routes"):
                names = sorted(
                    {
                        class_name
                        for _, class_name in self.routing_controller.routes.keys()
                    }
                )
                if names:
                    return names

        return ["HDV", "CAV"]

    def _build_signal_history(self) -> list[dict]:
        """
        Build frame-by-frame signal status for the HTML sidebar.

        Priority order:
        1. use signal states recorded by CTMSimulator during the run;
        2. fall back to reconstructing from the supplied signal_controller;
        3. if no controller exists, the JavaScript uses static phase timing.

        Recorded simulator states are necessary for stateful adaptive
        controllers, because their final internal state is not enough to
        reconstruct the past phase sequence after the run has finished.
        """

        recorded_green = list(getattr(self.history, "signal_green", []))
        recorded_phase = list(getattr(self.history, "active_phase_by_node", []))

        def node_records_for_frame(frame_index: int) -> dict:
            node_records: dict[str, dict] = {}
            active_by_node = recorded_phase[frame_index] if frame_index < len(recorded_phase) else {}
            for node_id, node in self.network.nodes.items():
                phase_records = []
                active_phase_id = active_by_node.get(node_id, active_by_node.get(str(node_id), None)) if isinstance(active_by_node, dict) else None
                for phase in getattr(node, "phases", []):
                    phase_records.append(
                        {
                            "phase_id": phase.phase_id,
                            "phase_name": phase.phase_name,
                            "movement_ids": list(phase.movement_ids),
                            "green": float(phase.green),
                            "yellow": float(phase.yellow),
                            "all_red": float(phase.all_red),
                            "active": str(phase.phase_id) == str(active_phase_id),
                        }
                    )
                node_records[str(node_id)] = {
                    "active_phase_id": active_phase_id,
                    "phases": phase_records,
                }
            return node_records

        if len(recorded_green) == len(self.history.time_s) and recorded_green:
            out = []
            for i, time_s in enumerate(self.history.time_s):
                movement_green = {
                    str(k): bool(v)
                    for k, v in dict(recorded_green[i]).items()
                }
                out.append(
                    {
                        "time_s": float(time_s),
                        "nodes": node_records_for_frame(i),
                        "movement_green": movement_green,
                        "source": "recorded",
                    }
                )
            return out

        signal_history = []
        if self.signal_controller is None:
            return signal_history

        try:
            compiled = self.network.compile(self.dt)
        except Exception:
            return signal_history

        for time_s in self.history.time_s:
            time_s = float(time_s)
            node_records = {}
            for node_id, node in self.network.nodes.items():
                phase_records = []
                for phase in node.phases:
                    green_s = float(phase.green)
                    if hasattr(self.signal_controller, "get_phase_green"):
                        try:
                            p_idx = compiled.phase_id_to_index[phase.phase_id]
                            green_s = float(
                                self.signal_controller.get_phase_green(
                                    net=compiled,
                                    node_id=node_id,
                                    phase_index=p_idx,
                                    time_s=time_s,
                                )
                            )
                        except Exception:
                            green_s = float(phase.green)
                    phase_records.append(
                        {
                            "phase_id": phase.phase_id,
                            "phase_name": phase.phase_name,
                            "movement_ids": list(phase.movement_ids),
                            "green": green_s,
                            "yellow": float(phase.yellow),
                            "all_red": float(phase.all_red),
                        }
                    )
                node_records[str(node_id)] = {"phases": phase_records}

            movement_green = {}
            for movement_id, movement in self.network.movements.items():
                try:
                    m_idx = compiled.movement_id_to_index[movement_id]
                    is_green = bool(
                        self.signal_controller.movement_is_green(
                            net=compiled,
                            movement_index=m_idx,
                            time_s=time_s,
                        )
                    )
                except Exception:
                    is_green = False
                movement_green[str(movement_id)] = is_green

            signal_history.append(
                {
                    "time_s": time_s,
                    "nodes": node_records,
                    "movement_green": movement_green,
                    "source": "recomputed",
                }
            )
        return signal_history

    def _build_routing_history(self) -> list[dict]:
        """
        Build frame-by-frame routing probabilities for the HTML sidebar.

        Priority order:
        1. use routing probabilities recorded by CTMSimulator during the run;
        2. fall back to recomputing assignment from saved states.
        """

        recorded = list(getattr(self.history, "routing_probabilities", []))

        if len(recorded) == len(self.history.time_s) and recorded:
            routing_history: list[dict] = []
            previous_rows = None
            for frame_index, time_s in enumerate(self.history.time_s):
                rows_this_frame = list(recorded[frame_index] or [])

                # Routing is usually updated less often than the CTM time step.
                # Store only changed routing frames, then let JavaScript reuse
                # the latest available routing record for intermediate frames.
                if previous_rows is not None and rows_this_frame == previous_rows:
                    continue
                previous_rows = rows_this_frame

                movement_probs: dict[str, dict[str, float]] = {}
                grouped: dict[tuple[str, str, str], dict] = {}
                for row in rows_this_frame:
                    movement_id = str(row.get("movement_id"))
                    class_name = str(row.get("class_name", ""))
                    destination_name = str(row.get("destination_name", row.get("destination_node", "")))
                    probability = float(row.get("probability", 0.0) or 0.0)
                    if probability <= 0.0:
                        continue

                    movement_probs.setdefault(movement_id, {})
                    movement_probs[movement_id][class_name] = max(
                        float(movement_probs[movement_id].get(class_name, 0.0)),
                        probability,
                    )

                    key = (movement_id, class_name, destination_name)
                    if key not in grouped:
                        grouped[key] = {
                            "class_name": class_name,
                            "destination_node": row.get("destination_node"),
                            "destination_name": destination_name,
                            "probability": probability,
                            "commodity_count": int(row.get("commodity_count", 1) or 1),
                        }
                    else:
                        grouped[key]["probability"] = max(float(grouped[key]["probability"]), probability)
                        grouped[key]["commodity_count"] = int(grouped[key].get("commodity_count", 1)) + int(row.get("commodity_count", 1) or 1)

                movement_destination_probs: dict[str, list[dict]] = {}
                for (movement_id, _class_name, _destination_name), row in grouped.items():
                    movement_destination_probs.setdefault(movement_id, []).append(row)

                routing_history.append(
                    {
                        "time_s": float(time_s),
                        "frame_index": int(frame_index),
                        "movement_probabilities": movement_probs,
                        "movement_destination_probabilities": movement_destination_probs,
                        "source": "recorded",
                    }
                )
            return routing_history

        routing_history = []

        if self.routing_controller is None:
            return routing_history

        try:
            from .vehicle_type import VehicleClass
            vehicle_classes = [
                VehicleClass(name=class_name)
                for class_name in self.vehicle_class_names
            ]
        except Exception:
            vehicle_classes = []

        commodity_metadata = []
        if hasattr(self.routing_controller, "commodity_metadata"):
            try:
                commodity_metadata = self.routing_controller.commodity_metadata()
            except Exception:
                commodity_metadata = []

        for t_index, time_s in enumerate(self.history.time_s):
            time_s = float(time_s)

            if hasattr(self.routing_controller, "update_assignment"):
                try:
                    commodity_state = None
                    if hasattr(self.history, "commodity_state") and len(self.history.commodity_state) > t_index:
                        commodity_state = self.history.commodity_state[t_index]
                    elif len(self.history.cell_state) > t_index:
                        commodity_state = self.history.cell_state[t_index]
                    self.routing_controller.update_assignment(
                        network=self.network,
                        vehicle_classes=vehicle_classes,
                        time_s=time_s,
                        commodity_state=commodity_state,
                    )
                    if hasattr(self.routing_controller, "commodity_metadata"):
                        commodity_metadata = self.routing_controller.commodity_metadata()
                except Exception:
                    pass

            movement_probs = {}
            movement_destination_probs = {}

            for movement_id, movement in self.network.movements.items():
                class_probs = {}

                for class_name in self.vehicle_class_names:
                    try:
                        pval = float(
                            self.routing_controller.probability(
                                from_link=movement.from_link,
                                to_link=movement.to_link,
                                class_name=class_name,
                                time_s=time_s,
                            )
                        )
                    except TypeError:
                        try:
                            pval = float(
                                self.routing_controller.probability(
                                    from_link=movement.from_link,
                                    to_link=movement.to_link,
                                    class_name=class_name,
                                )
                            )
                        except Exception:
                            pval = 0.0
                    except Exception:
                        pval = 0.0

                    class_probs[class_name] = pval

                movement_probs[str(movement_id)] = class_probs

                destination_rows = []
                for commodity in commodity_metadata:
                    commodity_id = commodity.get("commodity_id")
                    class_name = commodity.get("class_name")
                    destination_node = commodity.get("destination_node")
                    destination_name = commodity.get("destination_name")

                    try:
                        pval = float(
                            self.routing_controller.probability(
                                from_link=movement.from_link,
                                to_link=movement.to_link,
                                class_name=class_name,
                                destination_node=destination_node,
                                commodity_id=commodity_id,
                                time_s=time_s,
                            )
                        )
                    except Exception:
                        pval = 0.0

                    destination_rows.append(
                        {
                            "commodity_id": commodity_id,
                            "class_name": class_name,
                            "destination_node": destination_node,
                            "destination_name": destination_name,
                            "probability": pval,
                        }
                    )

                movement_destination_probs[str(movement_id)] = destination_rows

            routing_history.append(
                {
                    "time_s": time_s,
                    "movement_probabilities": movement_probs,
                    "movement_destination_probabilities": movement_destination_probs,
                    "source": "recomputed",
                }
            )

        return routing_history

    def save(self, filename: str = "ctm_visualization.html") -> None:
        signal_history = self._build_signal_history()
        routing_history = self._build_routing_history()

        data = {
            "nodes": {
                str(node_id): {
                    "id": node_id,
                    "x": node.x,
                    "y": node.y,
                    "type": node.node_type.value,
                    "name": node.node_name
                    if getattr(node, "node_name", None) is not None
                    else f"N{node_id}",
                    "phases": [
                        {
                            "phase_id": phase.phase_id,
                            "phase_name": phase.phase_name,
                            "movement_ids": list(phase.movement_ids),
                            "green": float(phase.green),
                            "yellow": float(phase.yellow),
                            "all_red": float(phase.all_red),
                        }
                        for phase in node.phases
                    ],
                }
                for node_id, node in self.network.nodes.items()
            },
            "links": {
                str(link_id): {
                    "id": link_id,
                    "from": link.from_node,
                    "to": link.to_node,
                    "length_m": float(sum(cell.length_m for cell in link.cells)),
                    "num_cells": len(link.cells),
                    "cell_ids": [cell.cell_id for cell in link.cells],
                    "cell_lengths": [float(cell.length_m) for cell in link.cells],
                    "cell_jam_storage": [
                        float(cell.jam_density_vpm * cell.length_m)
                        for cell in link.cells
                    ],
                }
                for link_id, link in self.network.links.items()
            },
            "movements": {
                str(movement_id): {
                    "id": movement_id,
                    "name": movement.movement_name,
                    "node_id": movement.node_id,
                    "from_link": movement.from_link,
                    "to_link": movement.to_link,
                    "turn_type": str(
                        movement.turn_type.value
                        if hasattr(movement.turn_type, "value")
                        else movement.turn_type
                    ),
                }
                for movement_id, movement in self.network.movements.items()
            },
            "vehicle_class_names": self.vehicle_class_names,
            "commodity_metadata": (
                self.routing_controller.commodity_metadata()
                if self.routing_controller is not None
                and hasattr(self.routing_controller, "commodity_metadata")
                else []
            ),
            "time_s": list(self.history.time_s),
            "cell_state": [x.tolist() for x in self.history.cell_state],
            "link_density_ratio": list(self.history.link_density_ratio),
            "link_vehicle_count": list(self.history.link_vehicle_count),
            "movement_flow": list(self.history.movement_flow),
            "entry_flow": list(getattr(self.history, "entry_flow", [])),
            "exit_flow": list(getattr(self.history, "exit_flow", [])),
            "signal_history": signal_history,
            "routing_history": routing_history,
            "is_flush": list(getattr(self.history, "is_flush", [])),
            "dt": self.dt,
        }

        html = self.build_html(data)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Saved HTML visualization to {filename}")

    def build_html(self, data: dict) -> str:
        json_data = json.dumps(data)

        html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>CTM Network Visualizer</title>

<style>
:root {
    --bg: #050812;
    --panel: #0b1224;
    --panel2: #101a33;
    --panel3: #162445;
    --border: #22345f;
    --border2: #314978;
    --text: #e7eefc;
    --muted: #98a9cc;
    --blue: #38bdf8;
    --cyan: #22d3ee;
    --green: #22c55e;
    --red: #ef4444;
    --orange: #f59e0b;
    --purple: #a78bfa;
    --cellbase: #334155;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    background: radial-gradient(circle at top left, #13234b 0, var(--bg) 42%);
    color: var(--text);
}

#titlebar {
    height: 54px;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: rgba(6, 10, 22, 0.92);
    position: sticky;
    top: 0;
    z-index: 5;
}

#title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 0.2px;
}

#subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-left: auto;
}

#main {
    display: grid;
    grid-template-columns: minmax(700px, 1fr) 560px;
    gap: 14px;
    padding: 14px;
    align-items: start;
}

#canvasWrap {
    background: rgba(5, 8, 18, 0.88);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.25);
}

svg {
    width: 100%;
    max-width: 1120px;
    height: 690px;
    background:
        linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px),
        #040816;
    background-size: 34px 34px;
    border-radius: 12px;
    display: block;
}

#panel {
    background: rgba(11,18,36,0.95);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 10px;
    max-height: calc(100vh - 82px);
    overflow-y: auto;
    box-shadow: 0 16px 40px rgba(0,0,0,0.25);
}

.section {
    border: 1px solid var(--border);
    background: rgba(16,26,51,0.72);
    border-radius: 12px;
    margin-bottom: 10px;
    overflow: hidden;
}

.section > summary {
    cursor: pointer;
    list-style: none;
    padding: 11px 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 800;
    color: #f3f7ff;
    background: linear-gradient(90deg, rgba(56,189,248,0.11), transparent);
}

.section > summary::-webkit-details-marker {
    display: none;
}

.section > summary::before {
    content: "▸";
    color: var(--cyan);
    transition: transform 0.15s;
}

.section[open] > summary::before {
    transform: rotate(90deg);
}

.sectionBody {
    padding: 12px;
    border-top: 1px solid rgba(255,255,255,0.06);
}

.row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 8px;
}

.row3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin-bottom: 8px;
}

label {
    display: block;
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 4px;
}

select,
input[type="number"],
input[type="text"] {
    width: 100%;
    padding: 7px 8px;
    background: #091126;
    color: var(--text);
    border: 1px solid var(--border2);
    border-radius: 8px;
    outline: none;
}

input[type="range"] {
    width: 100%;
    accent-color: var(--cyan);
}

button {
    background: #12214a;
    color: var(--text);
    border: 1px solid #31528d;
    border-radius: 8px;
    padding: 8px 10px;
    cursor: pointer;
    font-weight: 700;
}

button:hover {
    background: #193064;
    border-color: var(--cyan);
}

button.primary {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    border-color: #60a5fa;
}

button.greenBtn {
    background: linear-gradient(135deg, #059669, #16a34a);
    border-color: #86efac;
}

.statusbar {
    margin-top: 10px;
    padding: 9px 11px;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    color: #dce8ff;
    font-size: 13px;
}

.pillLine {
    display: flex;
    gap: 7px;
    flex-wrap: wrap;
    margin-top: 8px;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border: 1px solid var(--border2);
    background: rgba(255,255,255,0.04);
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 12px;
    color: var(--muted);
}

.pill b {
    color: var(--text);
}

.toggleGrid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 7px;
}

.toggle {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 8px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    color: #cbd8f5;
    font-size: 12px;
}

.toggle input {
    accent-color: var(--cyan);
}

.small {
    font-size: 12px;
    color: var(--muted);
}

.mono {
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
}

.link-info {
    line-height: 1.55;
    font-size: 14px;
    color: #eaf1ff;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}

th, td {
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 6px;
    text-align: right;
    vertical-align: top;
}

th:first-child, td:first-child {
    text-align: left;
}

th {
    color: #c6d6ff;
    background: rgba(255,255,255,0.035);
    position: sticky;
    top: 0;
    z-index: 1;
}

.tableWrap {
    max-height: 360px;
    overflow: auto;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
}

.cell-color-box {
    width: 18px;
    height: 13px;
    border-radius: 4px;
    display: inline-block;
    border: 1px solid rgba(255,255,255,0.20);
}

.legend-bar {
    width: 100%;
    height: 16px;
    border-radius: 999px;
    border: 1px solid #33436d;
    background: linear-gradient(
        to right,
        rgb(0,0,131),
        rgb(0,60,170),
        rgb(5,255,255),
        rgb(255,255,0),
        rgb(250,0,0),
        rgb(128,0,0)
    );
}

.node-label {
    fill: #dbeafe;
    font-size: 11px;
    font-weight: 800;
    paint-order: stroke;
    stroke: rgba(3,7,18,0.85);
    stroke-width: 3px;
}

.node-id-label {
    fill: #fef3c7;
    font-size: 10px;
    font-weight: 700;
    paint-order: stroke;
    stroke: rgba(3,7,18,0.9);
    stroke-width: 3px;
}

.link-label {
    fill: #dbeafe;
    font-size: 10px;
    font-weight: 700;
    paint-order: stroke;
    stroke: rgba(3,7,18,0.9);
    stroke-width: 3px;
}

.length-label {
    fill: #9fb2df;
    font-size: 10px;
    paint-order: stroke;
    stroke: rgba(3,7,18,0.9);
    stroke-width: 3px;
}

.footer-mark {
    fill: #00d1ff;
    font-size: 12px;
    font-weight: bold;
}

.intersection-box {
    fill: rgba(90, 120, 180, 0.20);
    stroke: #c7d7ff;
    stroke-width: 1.4;
    rx: 4;
    ry: 4;
}

.origin-dest-node {
    fill: #96a6d8;
    stroke: #dce6ff;
    stroke-width: 1;
}

.stopbar {
    stroke-width: 3;
    stroke-linecap: round;
}

.phase-badge {
    display: inline-block;
    padding: 3px 7px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 800;
}

.green {
    background: #0e9f6e;
    color: white;
}

.red {
    background: #dc2626;
    color: white;
}

.yellow {
    background: #f59e0b;
    color: black;
}

.prob {
    color: #7dd3fc;
    font-weight: 800;
}

.green-time {
    color: #86efac;
    font-weight: 800;
}

.groupCard {
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.025);
    border-radius: 10px;
    margin-bottom: 8px;
    overflow: hidden;
}

.groupHead {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    background: rgba(56,189,248,0.06);
    padding: 8px 10px;
    font-weight: 800;
    cursor: pointer;
}

.groupBody {
    padding: 8px;
}

.hidden {
    display: none !important;
}

#flowCanvas {
    width: 100%;
    height: 220px;
    background: #081022;
    border: 1px solid #24385f;
    border-radius: 10px;
}

@media (max-width: 1200px) {
    #main {
        grid-template-columns: 1fr;
    }
    #panel {
        max-height: none;
    }
}
</style>
</head>

<body>
<div id="titlebar">
    <div id="title">CTM Network Visualizer</div>
    <div id="subtitle">interactive playback · display toggles · grouped signals/routing</div>
</div>

<div id="main">
    <div id="canvasWrap">
        <svg id="svg" width="1120" height="690"></svg>
        <div class="statusbar" id="networkStatus"></div>
        <div class="pillLine" id="quickStats"></div>
    </div>

    <div id="panel">

        <details class="section" open>
            <summary>Playback</summary>
            <div class="sectionBody">
                <div class="row3">
                    <button id="firstButton">⏮ First</button>
                    <button id="back10Button">⏪ -10</button>
                    <button id="stepBackButton">◀ Step</button>
                </div>
                <div class="row3">
                    <button id="reverseButton">Reverse</button>
                    <button class="greenBtn" id="playButton">Play</button>
                    <button id="pauseButton">Pause</button>
                </div>
                <div class="row3">
                    <button id="stepForwardButton">Step ▶</button>
                    <button id="forward10Button">+10 ⏩</button>
                    <button id="lastButton">Last ⏭</button>
                </div>

                <div class="row">
                    <div>
                        <label>Playback speed</label>
                        <select id="speedSelect">
                            <option value="1200">Very slow</option>
                            <option value="800">Slow</option>
                            <option value="450" selected>Normal</option>
                            <option value="180">Fast</option>
                            <option value="70">Very fast</option>
                        </select>
                    </div>
                    <div>
                        <label>Jump size</label>
                        <input id="jumpSize" type="number" min="1" value="10">
                    </div>
                </div>

                <label>Time step</label>
                <input id="slider" type="range" min="0" max="0" value="0" step="1">
                <div id="timeText" class="small mono"></div>
            </div>
        </details>

        <details class="section" open>
            <summary>Map Display</summary>
            <div class="sectionBody">
                <div class="row">
                    <div>
                        <label>Metric shown on map</label>
                        <select id="metricSelect">
                            <option value="density" selected>Density ratio</option>
                            <option value="vehicles">Vehicles per cell</option>
                        </select>
                    </div>
                    <div>
                        <label>Cell line width</label>
                        <input id="cellWidthInput" type="number" min="1" max="14" step="0.5" value="5.5">
                    </div>
                </div>

                <div class="toggleGrid">
                    <label class="toggle"><input type="checkbox" id="showLinkIds" checked> Link IDs</label>
                    <label class="toggle"><input type="checkbox" id="showLinkLengths" checked> Link lengths</label>
                    <label class="toggle"><input type="checkbox" id="showNodeNames" checked> Node names</label>
                    <label class="toggle"><input type="checkbox" id="showNodeIds"> Node IDs</label>
                    <label class="toggle"><input type="checkbox" id="showArrows" checked> Direction arrows</label>
                    <label class="toggle"><input type="checkbox" id="showStopbars" checked> Signal stop bars</label>
                    <label class="toggle"><input type="checkbox" id="showIntersectionBoxes" checked> Intersection boxes</label>
                    <label class="toggle"><input type="checkbox" id="highlightSelected" checked> Highlight selected link</label>
                </div>

                <div style="margin-top:10px">
                    <button id="cleanViewButton">Clean view</button>
                    <button id="fullViewButton">Full labels</button>
                </div>
            </div>
        </details>

        <details class="section" open>
            <summary>Link and Cell Inspector</summary>
            <div class="sectionBody">
                <label>Select link</label>
                <select id="linkSelect"></select>
                <div id="linkInfo" class="link-info">Click a link or pick one above.</div>

                <h4>Cells on Selected Link</h4>
                <div id="cellDetails"></div>
            </div>
        </details>

        <details class="section" open>
            <summary>Signals</summary>
            <div class="sectionBody">
                <div class="row">
                    <div>
                        <label>Node filter</label>
                        <select id="signalNodeFilter">
                            <option value="all">All signalized nodes</option>
                        </select>
                    </div>
                    <div>
                        <label>View</label>
                        <select id="signalViewMode">
                            <option value="grouped" selected>Grouped by node</option>
                            <option value="table">Flat table</option>
                        </select>
                    </div>
                </div>
                <label class="toggle"><input type="checkbox" id="showOnlyGreenSignals"> Show only green movements</label>
                <div id="signalStatus"></div>
            </div>
        </details>

        <details class="section">
            <summary>Routing Probabilities</summary>
            <div class="sectionBody">
                <div class="row">
                    <div>
                        <label>Node filter</label>
                        <select id="routingNodeFilter">
                            <option value="all">All nodes</option>
                        </select>
                    </div>
                    <div>
                        <label>Min probability shown</label>
                        <input id="routingMinProb" type="number" min="0" max="1" step="0.01" value="0.0001">
                    </div>
                </div>
                <div class="row">
                    <div>
                        <label>Class filter</label>
                        <select id="routingClassFilter">
                            <option value="all">All classes</option>
                        </select>
                    </div>
                    <div>
                        <label>Destination filter</label>
                        <select id="routingDestinationFilter">
                            <option value="all">All destinations</option>
                        </select>
                    </div>
                </div>
                <label class="toggle"><input type="checkbox" id="routingGrouped" checked> Group by node</label>
                <div id="routingStatus"></div>
            </div>
        </details>

        <details class="section" open>
            <summary>Entered / Exited Flow</summary>
            <div class="sectionBody">
                <label>Flow unit</label>
                <select id="flowUnit">
                    <option value="vph">veh/h</option>
                    <option value="vps">veh/s</option>
                    <option value="vstep">veh/step</option>
                </select>
                <canvas id="flowCanvas" width="520" height="220"></canvas>
                <div id="flowSummary" class="small" style="margin-top:6px;"></div>
            </div>
        </details>

        <details class="section">
            <summary>Legend</summary>
            <div class="sectionBody">
                <div class="legend-bar"></div>
                <div class="row" style="margin-top:6px">
                    <span class="small">Low value</span>
                    <span class="small" style="text-align:right">High value</span>
                </div>
                <div class="small">
                    Cell values use a jet-like colormap. For density, the color is normalized to jam ratio 0–1.
                    For vehicles, the color is normalized by the maximum vehicles/cell at the current time step.
                </div>
            </div>
        </details>
    </div>
</div>

<script>
const DATA = __DATA__;

const svg = document.getElementById("svg");
const slider = document.getElementById("slider");
const timeText = document.getElementById("timeText");
const networkStatus = document.getElementById("networkStatus");
const quickStats = document.getElementById("quickStats");

const playButton = document.getElementById("playButton");
const reverseButton = document.getElementById("reverseButton");
const pauseButton = document.getElementById("pauseButton");
const speedSelect = document.getElementById("speedSelect");

const firstButton = document.getElementById("firstButton");
const lastButton = document.getElementById("lastButton");
const back10Button = document.getElementById("back10Button");
const forward10Button = document.getElementById("forward10Button");
const stepBackButton = document.getElementById("stepBackButton");
const stepForwardButton = document.getElementById("stepForwardButton");
const jumpSize = document.getElementById("jumpSize");

const metricSelect = document.getElementById("metricSelect");
const linkSelect = document.getElementById("linkSelect");
const linkInfo = document.getElementById("linkInfo");
const cellDetails = document.getElementById("cellDetails");

const signalStatus = document.getElementById("signalStatus");
const signalNodeFilter = document.getElementById("signalNodeFilter");
const signalViewMode = document.getElementById("signalViewMode");
const showOnlyGreenSignals = document.getElementById("showOnlyGreenSignals");

const routingStatus = document.getElementById("routingStatus");
const routingNodeFilter = document.getElementById("routingNodeFilter");
const routingClassFilter = document.getElementById("routingClassFilter");
const routingDestinationFilter = document.getElementById("routingDestinationFilter");
const routingMinProb = document.getElementById("routingMinProb");
const routingGrouped = document.getElementById("routingGrouped");

const flowCanvas = document.getElementById("flowCanvas");
const flowCtx = flowCanvas.getContext("2d");
const flowUnit = document.getElementById("flowUnit");
const flowSummary = document.getElementById("flowSummary");

const showLinkIds = document.getElementById("showLinkIds");
const showLinkLengths = document.getElementById("showLinkLengths");
const showNodeNames = document.getElementById("showNodeNames");
const showNodeIds = document.getElementById("showNodeIds");
const showArrows = document.getElementById("showArrows");
const showStopbars = document.getElementById("showStopbars");
const showIntersectionBoxes = document.getElementById("showIntersectionBoxes");
const highlightSelected = document.getElementById("highlightSelected");
const cellWidthInput = document.getElementById("cellWidthInput");
const cleanViewButton = document.getElementById("cleanViewButton");
const fullViewButton = document.getElementById("fullViewButton");

slider.max = Math.max(0, DATA.time_s.length - 1);

let selectedLinkId = null;
let timer = null;
let playDirection = 1;

const SVG_W = 1120;
const SVG_H = 690;
const MARGIN = 76;

const nodeValues = Object.values(DATA.nodes);
const xs = nodeValues.map(n => Number(n.x));
const ys = nodeValues.map(n => Number(n.y));

const minX = Math.min(...xs);
const maxX = Math.max(...xs);
const minY = Math.min(...ys);
const maxY = Math.max(...ys);

const rangeX = Math.max(1e-9, maxX - minX);
const rangeY = Math.max(1e-9, maxY - minY);

const scaleUniform = Math.min(
    (SVG_W - 2 * MARGIN) / rangeX,
    (SVG_H - 2 * MARGIN) / rangeY
);

const drawingW = rangeX * scaleUniform;
const drawingH = rangeY * scaleUniform;
const offsetX = (SVG_W - drawingW) / 2;
const offsetY = (SVG_H - drawingH) / 2;

function scaleX(x) {
    return offsetX + (Number(x) - minX) * scaleUniform;
}

function scaleY(y) {
    return SVG_H - (offsetY + (Number(y) - minY) * scaleUniform);
}

function clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
}

function makeSvgElement(tag) {
    return document.createElementNS("http://www.w3.org/2000/svg", tag);
}

function jetColor(v) {
    const x = clamp(Number(v || 0), 0, 1);
    const r = clamp(1.5 - Math.abs(4 * x - 3), 0, 1);
    const g = clamp(1.5 - Math.abs(4 * x - 2), 0, 1);
    const b = clamp(1.5 - Math.abs(4 * x - 1), 0, 1);
    return `rgb(${Math.round(255*r)}, ${Math.round(255*g)}, ${Math.round(255*b)})`;
}

function totalVehiclesInCell(cellArray) {
    let s = 0.0;
    for (const v of (cellArray || [])) {
        s += Number(v || 0);
    }
    return s;
}

function nodeLabel(nodeId) {
    const n = DATA.nodes[String(nodeId)];
    return n ? n.name : String(nodeId);
}

function linkLabel(linkId) {
    return "L" + String(linkId);
}

function movementNodeName(movement) {
    const node = DATA.nodes[String(movement.node_id)];
    return node ? node.name : String(movement.node_id);
}

function findCellJamStorage(cellId) {
    for (const link of Object.values(DATA.links)) {
        const idx = link.cell_ids.indexOf(cellId);
        if (idx >= 0) {
            return Number(link.cell_jam_storage[idx] || 1.0);
        }
    }
    return 1.0;
}

function getCellMetricValue(cellId, tIndex) {
    const x = DATA.cell_state[tIndex] || [];
    const cellState = x[cellId] || [];
    const vehicles = totalVehiclesInCell(cellState);
    const jam = findCellJamStorage(cellId);
    const density = jam > 0 ? vehicles / jam : 0.0;
    return metricSelect.value === "vehicles" ? vehicles : density;
}

function getMetricNormalizationMax(tIndex) {
    if (metricSelect.value === "density") {
        return 1.0;
    }

    let maxVehicles = 1.0;
    for (const link of Object.values(DATA.links)) {
        for (const cellId of link.cell_ids) {
            const value = getCellMetricValue(cellId, tIndex);
            if (value > maxVehicles) maxVehicles = value;
        }
    }
    return maxVehicles;
}

function drawStaticBase() {
    svg.innerHTML = "";

    const defs = makeSvgElement("defs");

    const marker = makeSvgElement("marker");
    marker.setAttribute("id", "arrow");
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "9");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "6");
    marker.setAttribute("markerHeight", "6");
    marker.setAttribute("orient", "auto");

    const mpath = makeSvgElement("path");
    mpath.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    mpath.setAttribute("fill", "#9fb7ff");
    marker.appendChild(mpath);
    defs.appendChild(marker);
    svg.appendChild(defs);

    const cellLayer = makeSvgElement("g");
    cellLayer.setAttribute("id", "cellLayer");
    svg.appendChild(cellLayer);

    const stopbarLayer = makeSvgElement("g");
    stopbarLayer.setAttribute("id", "stopbarLayer");
    svg.appendChild(stopbarLayer);

    const nodeLayer = makeSvgElement("g");
    nodeLayer.setAttribute("id", "nodeLayer");
    svg.appendChild(nodeLayer);

    const overlayLayer = makeSvgElement("g");
    overlayLayer.setAttribute("id", "overlayLayer");
    svg.appendChild(overlayLayer);

    drawCellsGeometry();
    drawNodes();
    drawStopBars();
    drawFooter();
    applyDisplayToggles();
}

function drawCellsGeometry() {
    const cellLayer = document.getElementById("cellLayer");

    for (const [linkId, link] of Object.entries(DATA.links)) {
        const fromNode = DATA.nodes[String(link.from)];
        const toNode = DATA.nodes[String(link.to)];
        if (!fromNode || !toNode) continue;

        const x1 = scaleX(fromNode.x);
        const y1 = scaleY(fromNode.y);
        const x2 = scaleX(toNode.x);
        const y2 = scaleY(toNode.y);

        const dx = x2 - x1;
        const dy = y2 - y1;
        const L = Math.sqrt(dx * dx + dy * dy) || 1.0;

        const ux = dx / L;
        const uy = dy / L;

        const startPad = 17;
        const endPad = 20;
        const usable = Math.max(1, L - startPad - endPad);

        const n = Number(link.num_cells);
        const seg = usable / n;

        const linkGroup = makeSvgElement("g");
        linkGroup.setAttribute("id", "linkgroup_" + linkId);
        cellLayer.appendChild(linkGroup);

        const hit = makeSvgElement("line");
        hit.setAttribute("x1", x1 + ux * startPad);
        hit.setAttribute("y1", y1 + uy * startPad);
        hit.setAttribute("x2", x2 - ux * endPad);
        hit.setAttribute("y2", y2 - uy * endPad);
        hit.setAttribute("stroke", "transparent");
        hit.setAttribute("stroke-width", 20);
        hit.style.cursor = "pointer";
        hit.onclick = () => {
            selectedLinkId = linkId;
            linkSelect.value = linkId;
            updateFrame(Number(slider.value));
        };
        linkGroup.appendChild(hit);

        for (let i = 0; i < n; i++) {
            const sx = x1 + ux * (startPad + i * seg);
            const sy = y1 + uy * (startPad + i * seg);
            const ex = x1 + ux * (startPad + (i + 1) * seg);
            const ey = y1 + uy * (startPad + (i + 1) * seg);

            const line = makeSvgElement("line");
            const cellId = link.cell_ids[i];

            line.setAttribute("id", "cellseg_" + cellId);
            line.setAttribute("data-link-id", linkId);
            line.setAttribute("x1", sx);
            line.setAttribute("y1", sy);
            line.setAttribute("x2", ex);
            line.setAttribute("y2", ey);
            line.setAttribute("stroke", "var(--cellbase)");
            line.setAttribute("stroke-width", cellWidthInput.value);
            line.setAttribute("stroke-linecap", "round");
            line.style.cursor = "pointer";
            line.onclick = () => {
                selectedLinkId = linkId;
                linkSelect.value = linkId;
                updateFrame(Number(slider.value));
            };
            linkGroup.appendChild(line);
        }

        const arrowLine = makeSvgElement("line");
        arrowLine.setAttribute("id", "arrow_" + linkId);
        arrowLine.setAttribute("class", "map-arrow");
        arrowLine.setAttribute("x1", x2 - ux * (endPad + 18));
        arrowLine.setAttribute("y1", y2 - uy * (endPad + 18));
        arrowLine.setAttribute("x2", x2 - ux * endPad);
        arrowLine.setAttribute("y2", y2 - uy * endPad);
        arrowLine.setAttribute("stroke", "#9fb7ff");
        arrowLine.setAttribute("stroke-width", 2.0);
        arrowLine.setAttribute("marker-end", "url(#arrow)");
        linkGroup.appendChild(arrowLine);

        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;

        const linkIdLabel = makeSvgElement("text");
        linkIdLabel.setAttribute("id", "linklabel_" + linkId);
        linkIdLabel.setAttribute("x", mx + 4);
        linkIdLabel.setAttribute("y", my - 10);
        linkIdLabel.setAttribute("class", "link-label");
        linkIdLabel.textContent = "L" + linkId;
        linkGroup.appendChild(linkIdLabel);

        const lenLabel = makeSvgElement("text");
        lenLabel.setAttribute("id", "lengthlabel_" + linkId);
        lenLabel.setAttribute("x", mx + 4);
        lenLabel.setAttribute("y", my + 7);
        lenLabel.setAttribute("class", "length-label");
        lenLabel.textContent = Number(link.length_m).toFixed(0) + " m";
        linkGroup.appendChild(lenLabel);
    }
}

function drawNodes() {
    const nodeLayer = document.getElementById("nodeLayer");

    for (const [nodeId, node] of Object.entries(DATA.nodes)) {
        const x = scaleX(node.x);
        const y = scaleY(node.y);

        if (String(node.type).toLowerCase().includes("signal")) {
            const box = makeSvgElement("rect");
            box.setAttribute("id", "nodebox_" + nodeId);
            box.setAttribute("x", x - 9);
            box.setAttribute("y", y - 9);
            box.setAttribute("width", 18);
            box.setAttribute("height", 18);
            box.setAttribute("class", "intersection-box");
            nodeLayer.appendChild(box);
        } else {
            const circle = makeSvgElement("circle");
            circle.setAttribute("id", "nodecircle_" + nodeId);
            circle.setAttribute("cx", x);
            circle.setAttribute("cy", y);
            circle.setAttribute("r", 4);
            circle.setAttribute("class", "origin-dest-node");
            nodeLayer.appendChild(circle);
        }

        const label = makeSvgElement("text");
        label.setAttribute("id", "nodelabel_" + nodeId);
        label.setAttribute("x", x - 22);
        label.setAttribute("y", y - 13);
        label.setAttribute("class", "node-label");
        label.textContent = node.name;
        nodeLayer.appendChild(label);

        const idLabel = makeSvgElement("text");
        idLabel.setAttribute("id", "nodeidlabel_" + nodeId);
        idLabel.setAttribute("x", x + 11);
        idLabel.setAttribute("y", y + 18);
        idLabel.setAttribute("class", "node-id-label");
        idLabel.textContent = "ID " + nodeId;
        nodeLayer.appendChild(idLabel);
    }
}

function drawStopBars() {
    const layer = document.getElementById("stopbarLayer");

    for (const [movementId, movement] of Object.entries(DATA.movements)) {
        const link = DATA.links[String(movement.from_link)];
        if (!link) continue;

        const fromNode = DATA.nodes[String(link.from)];
        const toNode = DATA.nodes[String(link.to)];
        if (!fromNode || !toNode) continue;

        const x1 = scaleX(fromNode.x);
        const y1 = scaleY(fromNode.y);
        const x2 = scaleX(toNode.x);
        const y2 = scaleY(toNode.y);

        const dx = x2 - x1;
        const dy = y2 - y1;
        const L = Math.sqrt(dx * dx + dy * dy) || 1.0;
        const ux = dx / L;
        const uy = dy / L;

        const px = x2 - ux * 22;
        const py = y2 - uy * 22;

        const nx = -uy;
        const ny = ux;
        const half = 7;

        const bar = makeSvgElement("line");
        bar.setAttribute("id", "stopbar_" + movementId);
        bar.setAttribute("class", "stopbar");
        bar.setAttribute("x1", px - nx * half);
        bar.setAttribute("y1", py - ny * half);
        bar.setAttribute("x2", px + nx * half);
        bar.setAttribute("y2", py + ny * half);
        bar.setAttribute("stroke", "#ff4d6d");
        layer.appendChild(bar);
    }
}

function drawFooter() {
    const footer1 = makeSvgElement("text");
    footer1.setAttribute("x", SVG_W - 86);
    footer1.setAttribute("y", SVG_H - 22);
    footer1.setAttribute("class", "footer-mark");
    footer1.textContent = "CTM";
    svg.appendChild(footer1);
}

function initSelectors() {
    linkSelect.innerHTML = "";

    for (const [linkId, link] of Object.entries(DATA.links)) {
        const fromNode = DATA.nodes[String(link.from)];
        const toNode = DATA.nodes[String(link.to)];

        const opt = document.createElement("option");
        opt.value = linkId;
        opt.textContent = `Link ${linkId}: ${fromNode ? fromNode.name : link.from} → ${toNode ? toNode.name : link.to}`;
        linkSelect.appendChild(opt);
    }

    const first = Object.keys(DATA.links)[0];
    if (first !== undefined) {
        selectedLinkId = first;
        linkSelect.value = first;
    }

    const signalNodes = new Set();
    for (const movement of Object.values(DATA.movements)) {
        signalNodes.add(String(movement.node_id));
    }

    const sortedNodeIds = [...signalNodes].sort((a,b) => nodeLabel(a).localeCompare(nodeLabel(b)));
    for (const nodeId of sortedNodeIds) {
        const opt1 = document.createElement("option");
        opt1.value = nodeId;
        opt1.textContent = nodeLabel(nodeId);
        signalNodeFilter.appendChild(opt1);

        const opt2 = document.createElement("option");
        opt2.value = nodeId;
        opt2.textContent = nodeLabel(nodeId);
        routingNodeFilter.appendChild(opt2);
    }

    for (const cls of (DATA.vehicle_class_names || [])) {
        const opt = document.createElement("option");
        opt.value = cls;
        opt.textContent = cls;
        routingClassFilter.appendChild(opt);
    }

    const destinations = new Set();
    for (const c of (DATA.commodity_metadata || [])) {
        if (c.destination_name !== undefined && c.destination_name !== null) {
            destinations.add(String(c.destination_name));
        }
    }
    for (const d of [...destinations].sort()) {
        const opt = document.createElement("option");
        opt.value = d;
        opt.textContent = d;
        routingDestinationFilter.appendChild(opt);
    }
}

function getActiveSignalRecord(tIndex) {
    if (DATA.signal_history && DATA.signal_history.length > tIndex) {
        return DATA.signal_history[tIndex];
    }
    return null;
}

function getActiveRoutingRecord(tIndex) {
    const records = DATA.routing_history || [];
    if (!records.length) return null;

    // Full format: one routing record per animation frame.
    if (records.length === DATA.time_s.length) {
        return records[Math.max(0, Math.min(tIndex, records.length - 1))];
    }

    // Compact format: only routing-change frames are stored. Use the latest
    // routing record whose timestamp is not greater than the visible frame.
    const currentTime = Number((DATA.time_s || [])[tIndex] || 0.0);
    let chosen = records[0];
    for (const rec of records) {
        if (Number(rec.time_s || 0.0) <= currentTime + 1.0e-9) chosen = rec;
        else break;
    }
    return chosen;
}

function getStaticNodePhaseStatus(nodeId, timeS) {
    const node = DATA.nodes[String(nodeId)];
    const phases = node ? (node.phases || []) : [];

    if (phases.length === 0) {
        return {
            status: "none",
            phaseName: "unsignalized",
            activeMovementIds: [],
            activeGreen: 0.0
        };
    }

    let cycle = 0.0;
    for (const p of phases) {
        cycle += Number(p.green || 0) + Number(p.yellow || 0) + Number(p.all_red || 0);
    }

    if (cycle <= 0) {
        return {
            status: "none",
            phaseName: "no cycle",
            activeMovementIds: [],
            activeGreen: 0.0
        };
    }

    const tc = timeS % cycle;
    let elapsed = 0.0;

    for (const p of phases) {
        const g = Number(p.green || 0);
        const y = Number(p.yellow || 0);
        const ar = Number(p.all_red || 0);
        const total = g + y + ar;

        if (elapsed <= tc && tc < elapsed + g) {
            return {
                status: "green",
                phaseName: p.phase_name,
                activeMovementIds: p.movement_ids || [],
                activeGreen: g
            };
        }

        if (elapsed + g <= tc && tc < elapsed + g + y) {
            return {
                status: "yellow",
                phaseName: p.phase_name,
                activeMovementIds: [],
                activeGreen: g
            };
        }

        if (elapsed + g + y <= tc && tc < elapsed + total) {
            return {
                status: "red",
                phaseName: p.phase_name,
                activeMovementIds: [],
                activeGreen: g
            };
        }

        elapsed += total;
    }

    return {
        status: "red",
        phaseName: "all-red",
        activeMovementIds: [],
        activeGreen: 0.0
    };
}

function getActiveMovements(tIndex) {
    const timeS = Number(DATA.time_s[tIndex] || 0);
    const signalRecord = getActiveSignalRecord(tIndex);
    const activeMovements = new Set();

    if (signalRecord) {
        for (const [movementId, isGreen] of Object.entries(signalRecord.movement_green || {})) {
            if (isGreen) activeMovements.add(String(movementId));
        }
    } else {
        for (const [nodeId] of Object.entries(DATA.nodes)) {
            const s = getStaticNodePhaseStatus(nodeId, timeS);
            for (const m of s.activeMovementIds) {
                activeMovements.add(String(m));
            }
        }
    }

    return activeMovements;
}

function updateSignalStatus(tIndex) {
    const signalRecord = getActiveSignalRecord(tIndex);
    const activeMovements = getActiveMovements(tIndex);
    const nodeFilter = signalNodeFilter.value;
    const onlyGreen = showOnlyGreenSignals.checked;

    for (const [nodeId] of Object.entries(DATA.nodes)) {
        const box = document.getElementById("nodebox_" + nodeId);
        if (!box) continue;

        let anyGreen = false;
        for (const [movementId, movement] of Object.entries(DATA.movements)) {
            if (String(movement.node_id) === String(nodeId) && activeMovements.has(String(movementId))) {
                anyGreen = true;
                break;
            }
        }

        if (anyGreen) {
            box.setAttribute("stroke", "#22c55e");
            box.setAttribute("fill", "rgba(34,197,94,0.20)");
        } else {
            box.setAttribute("stroke", "#ef4444");
            box.setAttribute("fill", "rgba(239,68,68,0.12)");
        }
    }

    for (const [movementId] of Object.entries(DATA.movements)) {
        const bar = document.getElementById("stopbar_" + movementId);
        if (!bar) continue;
        bar.setAttribute("stroke", activeMovements.has(String(movementId)) ? "#22c55e" : "#f43f5e");
    }

    if (signalViewMode.value === "table") {
        renderSignalFlatTable(tIndex, signalRecord, activeMovements, nodeFilter, onlyGreen);
    } else {
        renderSignalGrouped(tIndex, signalRecord, activeMovements, nodeFilter, onlyGreen);
    }
}

function findPhaseForMovement(signalRecord, movement) {
    let phaseName = "";
    let greenTime = 0.0;

    if (signalRecord) {
        const nodeSignal = signalRecord.nodes[String(movement.node_id)] || {};
        const phases = nodeSignal.phases || [];

        for (const p of phases) {
            if ((p.movement_ids || []).map(String).includes(String(movement.id))) {
                phaseName = p.phase_name;
                greenTime = Number(p.green || 0.0);
                break;
            }
        }
    } else {
        const node = DATA.nodes[String(movement.node_id)];
        for (const p of (node ? node.phases : []) || []) {
            if ((p.movement_ids || []).map(String).includes(String(movement.id))) {
                phaseName = p.phase_name;
                greenTime = Number(p.green || 0.0);
                break;
            }
        }
    }

    return { phaseName, greenTime };
}

function renderSignalFlatTable(tIndex, signalRecord, activeMovements, nodeFilter, onlyGreen) {
    let html = `
        <div class="tableWrap">
        <table>
            <tr>
                <th>Node</th>
                <th>Phase</th>
                <th>Green s</th>
                <th>Movement</th>
                <th>Status</th>
            </tr>
    `;

    for (const [movementId, movement] of Object.entries(DATA.movements)) {
        if (nodeFilter !== "all" && String(movement.node_id) !== nodeFilter) continue;

        const isGreen = activeMovements.has(String(movementId));
        if (onlyGreen && !isGreen) continue;

        const node = DATA.nodes[String(movement.node_id)];
        if (!node || (node.phases || []).length === 0) continue;

        const pinfo = findPhaseForMovement(signalRecord, { ...movement, id: movementId });
        const badgeClass = isGreen ? "green" : "red";
        const label = isGreen ? "green" : "red";

        html += `
            <tr>
                <td>${node.name}</td>
                <td>${pinfo.phaseName}</td>
                <td><span class="green-time">${pinfo.greenTime.toFixed(1)}</span></td>
                <td>${movement.name}</td>
                <td><span class="phase-badge ${badgeClass}">${label}</span></td>
            </tr>
        `;
    }

    html += "</table></div>";
    signalStatus.innerHTML = html;
}

function renderSignalGrouped(tIndex, signalRecord, activeMovements, nodeFilter, onlyGreen) {
    const byNode = {};
    for (const [movementId, movement] of Object.entries(DATA.movements)) {
        if (nodeFilter !== "all" && String(movement.node_id) !== nodeFilter) continue;

        const isGreen = activeMovements.has(String(movementId));
        if (onlyGreen && !isGreen) continue;

        const nodeId = String(movement.node_id);
        if (!byNode[nodeId]) byNode[nodeId] = [];
        byNode[nodeId].push({ movementId, movement, isGreen });
    }

    let html = "";
    for (const nodeId of Object.keys(byNode).sort((a,b) => nodeLabel(a).localeCompare(nodeLabel(b)))) {
        const rows = byNode[nodeId];
        const greenCount = rows.filter(r => r.isGreen).length;
        html += `
            <div class="groupCard">
                <div class="groupHead" onclick="toggleGroupBody(this)">
                    <span>${nodeLabel(nodeId)}</span>
                    <span class="small">${greenCount}/${rows.length} green</span>
                </div>
                <div class="groupBody">
                    <table>
                        <tr><th>Phase</th><th>Green</th><th>Movement</th><th>Status</th></tr>
        `;

        for (const row of rows) {
            const pinfo = findPhaseForMovement(signalRecord, { ...row.movement, id: row.movementId });
            html += `
                <tr>
                    <td>${pinfo.phaseName}</td>
                    <td><span class="green-time">${pinfo.greenTime.toFixed(1)}</span></td>
                    <td>${row.movement.name}</td>
                    <td><span class="phase-badge ${row.isGreen ? "green" : "red"}">${row.isGreen ? "green" : "red"}</span></td>
                </tr>
            `;
        }

        html += "</table></div></div>";
    }

    signalStatus.innerHTML = html || `<div class="small">No signal records match the selected filter.</div>`;
}

function toggleGroupBody(head) {
    const body = head.nextElementSibling;
    if (body) body.classList.toggle("hidden");
}

function updateRoutingStatus(tIndex) {
    const routingRecord = getActiveRoutingRecord(tIndex);

    if (!routingRecord) {
        routingStatus.innerHTML = `
            <div class="small">
                No routing controller data was passed to HTMLVisualizer.<br>
                Use HTMLVisualizer(..., routing_controller=demand, vehicle_class_names=["HDV","CAV"]).
            </div>
        `;
        return;
    }

    const nodeFilter = routingNodeFilter.value;
    const classFilter = routingClassFilter.value;
    const destFilter = routingDestinationFilter.value;
    const minProb = Number(routingMinProb.value || 0.0);

    const destinationProbs = routingRecord.movement_destination_probabilities || {};
    const aggregateProbs = routingRecord.movement_probabilities || {};

    const rows = [];

    for (const [movementId, movement] of Object.entries(DATA.movements)) {
        if (nodeFilter !== "all" && String(movement.node_id) !== nodeFilter) continue;

        const fromLink = DATA.links[String(movement.from_link)];
        const toLink = DATA.links[String(movement.to_link)];
        if (!fromLink || !toLink) continue;

        const fromNode = DATA.nodes[String(fromLink.from)];
        const toNode = DATA.nodes[String(toLink.to)];
        const odText = `${fromNode ? fromNode.name : fromLink.from}→${toNode ? toNode.name : toLink.to}`;
        const rowsForMovement = destinationProbs[String(movementId)] || [];

        if (rowsForMovement.length > 0) {
            for (const row of rowsForMovement) {
                const p = Number(row.probability || 0.0);
                if (p < minProb) continue;
                if (classFilter !== "all" && row.class_name !== classFilter) continue;
                if (destFilter !== "all" && String(row.destination_name) !== destFilter) continue;

                rows.push({
                    nodeId: String(movement.node_id),
                    movementName: movement.name,
                    fromTo: `L${movement.from_link}→L${movement.to_link}<br><span class="small">${odText}</span>`,
                    className: row.class_name,
                    destination: row.destination_name,
                    probability: p
                });
            }
        } else {
            for (const cls of DATA.vehicle_class_names || []) {
                if (classFilter !== "all" && cls !== classFilter) continue;

                const p = Number(((aggregateProbs[String(movementId)] || {})[cls]) || 0.0);
                if (p < minProb) continue;

                rows.push({
                    nodeId: String(movement.node_id),
                    movementName: movement.name,
                    fromTo: `L${movement.from_link}→L${movement.to_link}<br><span class="small">${odText}</span>`,
                    className: cls,
                    destination: "aggregate",
                    probability: p
                });
            }
        }
    }

    if (routingGrouped.checked) {
        renderRoutingGrouped(rows);
    } else {
        renderRoutingTable(rows);
    }
}

function renderRoutingTable(rows) {
    let html = `
        <div class="tableWrap">
        <table>
            <tr>
                <th>Movement</th>
                <th>From→To</th>
                <th>Class</th>
                <th>Destination</th>
                <th>Prob.</th>
            </tr>
    `;

    for (const r of rows) {
        html += `
            <tr>
                <td>${r.movementName}</td>
                <td>${r.fromTo}</td>
                <td>${r.className}</td>
                <td>${r.destination}</td>
                <td><span class="prob">${r.probability.toFixed(3)}</span></td>
            </tr>
        `;
    }

    html += "</table></div>";
    routingStatus.innerHTML = html || `<div class="small">No routing rows match the filter.</div>`;
}

function renderRoutingGrouped(rows) {
    const byNode = {};
    for (const r of rows) {
        if (!byNode[r.nodeId]) byNode[r.nodeId] = [];
        byNode[r.nodeId].push(r);
    }

    let html = "";
    for (const nodeId of Object.keys(byNode).sort((a,b) => nodeLabel(a).localeCompare(nodeLabel(b)))) {
        const nodeRows = byNode[nodeId];
        html += `
            <div class="groupCard">
                <div class="groupHead" onclick="toggleGroupBody(this)">
                    <span>${nodeLabel(nodeId)}</span>
                    <span class="small">${nodeRows.length} probability rows</span>
                </div>
                <div class="groupBody">
                    <table>
                        <tr><th>Movement</th><th>Class</th><th>Destination</th><th>Prob.</th></tr>
        `;

        for (const r of nodeRows) {
            html += `
                <tr>
                    <td>${r.movementName}<br><span class="small">${r.fromTo}</span></td>
                    <td>${r.className}</td>
                    <td>${r.destination}</td>
                    <td><span class="prob">${r.probability.toFixed(3)}</span></td>
                </tr>
            `;
        }

        html += "</table></div></div>";
    }

    routingStatus.innerHTML = html || `<div class="small">No routing rows match the filter.</div>`;
}

function updateCellColors(tIndex) {
    const maxNorm = getMetricNormalizationMax(tIndex);
    const width = Number(cellWidthInput.value || 5.5);

    for (const [linkId, link] of Object.entries(DATA.links)) {
        for (const cellId of link.cell_ids) {
            const seg = document.getElementById("cellseg_" + cellId);
            if (!seg) continue;

            const raw = getCellMetricValue(cellId, tIndex);
            const norm = metricSelect.value === "density"
                ? clamp(raw, 0, 1)
                : clamp(raw / maxNorm, 0, 1);

            seg.setAttribute("stroke", jetColor(norm));

            const selected = highlightSelected.checked && selectedLinkId !== null && String(linkId) === String(selectedLinkId);
            seg.setAttribute("stroke-width", selected ? width + 2.5 : width);
        }
    }
}

function updateLinkInfo(tIndex) {
    if (selectedLinkId === null) return;

    const link = DATA.links[selectedLinkId];
    if (!link) return;

    const fromNode = DATA.nodes[String(link.from)];
    const toNode = DATA.nodes[String(link.to)];

    const linkVeh = Number((DATA.link_vehicle_count[tIndex] || {})[selectedLinkId] || 0.0);
    const linkDen = Number((DATA.link_density_ratio[tIndex] || {})[selectedLinkId] || 0.0);

    linkInfo.innerHTML = `
        <b>Link ${selectedLinkId}</b><br>
        From: ${fromNode ? fromNode.name : link.from} → ${toNode ? toNode.name : link.to}<br>
        Length: ${Number(link.length_m).toFixed(1)} m<br>
        Cells: ${link.num_cells}<br>
        Vehicles on link: ${linkVeh.toFixed(3)}<br>
        Density ratio: ${linkDen.toFixed(4)}
    `;
}

function updateCellDetails(tIndex) {
    if (selectedLinkId === null) {
        cellDetails.innerHTML = "";
        return;
    }

    const link = DATA.links[selectedLinkId];
    if (!link) return;

    const x = DATA.cell_state[tIndex] || [];
    const xPrev = DATA.cell_state[Math.max(0, tIndex - 1)] || [];

    let html = `
        <div class="small" style="margin:8px 0 6px 0;">
            Time = ${Number(DATA.time_s[tIndex] || 0).toFixed(1)} s,
            step = ${tIndex}
        </div>

        <div class="tableWrap">
        <table class="cell-table">
            <tr>
                <th>Cell</th>
                <th>Color</th>
                <th>Veh</th>
                <th>ΔVeh</th>
                <th>Density</th>
                <th>Class values</th>
            </tr>
    `;

    const maxNorm = getMetricNormalizationMax(tIndex);

    for (let i = 0; i < link.cell_ids.length; i++) {
        const cellId = link.cell_ids[i];

        const classVals = x[cellId] || [];
        const classValsPrev = xPrev[cellId] || [];

        const vehicles = totalVehiclesInCell(classVals);
        const vehiclesPrev = totalVehiclesInCell(classValsPrev);
        const deltaVehicles = vehicles - vehiclesPrev;

        const jam = Number(link.cell_jam_storage[i] || 1.0);
        const density = jam > 0 ? vehicles / jam : 0.0;

        const raw = metricSelect.value === "density" ? density : vehicles;
        const norm = metricSelect.value === "density"
            ? clamp(raw, 0, 1)
            : clamp(raw / maxNorm, 0, 1);

        const color = jetColor(norm);

        html += `
            <tr>
                <td>${i}</td>
                <td><span class="cell-color-box" style="background:${color};"></span></td>
                <td>${vehicles.toFixed(4)}</td>
                <td>${deltaVehicles.toFixed(4)}</td>
                <td>${density.toFixed(4)}</td>
                <td>${classVals.map(v => Number(v || 0).toFixed(4)).join(", ")}</td>
            </tr>
        `;
    }

    html += "</table></div>";
    cellDetails.innerHTML = html;
}

function totalFlow(dictObj) {
    let s = 0.0;
    for (const v of Object.values(dictObj || {})) {
        s += Number(v || 0);
    }
    return s;
}

function flowConvert(vPerStep) {
    if (flowUnit.value === "vps") return vPerStep / DATA.dt;
    if (flowUnit.value === "vstep") return vPerStep;
    return vPerStep * 3600.0 / DATA.dt;
}

function flowUnitLabel() {
    if (flowUnit.value === "vps") return "veh/s";
    if (flowUnit.value === "vstep") return "veh/step";
    return "veh/h";
}

function drawFlowPlot(tIndex) {
    const ctx = flowCtx;
    const W = flowCanvas.width;
    const H = flowCanvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#081022";
    ctx.fillRect(0, 0, W, H);

    const n = DATA.time_s.length;

    if (n <= 1 || !DATA.entry_flow || !DATA.exit_flow || DATA.entry_flow.length === 0) {
        ctx.fillStyle = "#8ca1cf";
        ctx.font = "12px Arial";
        ctx.fillText("No entry/exit history available.", 14, 24);
        flowSummary.innerHTML = "";
        return;
    }

    const entrySeries = [];
    const exitSeries = [];

    for (let i = 0; i < n; i++) {
        entrySeries.push(flowConvert(totalFlow(DATA.entry_flow[i])));
        exitSeries.push(flowConvert(totalFlow(DATA.exit_flow[i])));
    }

    const padL = 46;
    const padR = 14;
    const padT = 18;
    const padB = 30;

    const ymax = Math.max(1.0, ...entrySeries, ...exitSeries);

    function sx(i) {
        return padL + i / (n - 1) * (W - padL - padR);
    }

    function sy(v) {
        return H - padB - v / ymax * (H - padT - padB);
    }

    ctx.strokeStyle = "#21365e";
    ctx.lineWidth = 1;

    for (let g = 0; g <= 4; g++) {
        const y = padT + g / 4 * (H - padT - padB);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(W - padR, y);
        ctx.stroke();
    }

    ctx.strokeStyle = "#7f91bc";
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, H - padB);
    ctx.lineTo(W - padR, H - padB);
    ctx.stroke();

    function plotLine(vals, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.4;
        ctx.beginPath();

        for (let i = 0; i < vals.length; i++) {
            const x = sx(i);
            const y = sy(vals[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        ctx.stroke();
    }

    plotLine(entrySeries, "#38bdf8");
    plotLine(exitSeries, "#fbbf24");

    const xNow = sx(tIndex);
    ctx.strokeStyle = "#ffffff";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(xNow, padT);
    ctx.lineTo(xNow, H - padB);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#bcd0ff";
    ctx.font = "11px Arial";
    ctx.textAlign = "right";
    ctx.fillText(ymax.toFixed(1), padL - 4, padT + 4);
    ctx.fillText((ymax / 2).toFixed(1), padL - 4, sy(ymax / 2) + 4);
    ctx.fillText("0", padL - 4, H - padB + 4);

    ctx.textAlign = "left";
    ctx.fillStyle = "#38bdf8";
    ctx.fillText("Entered", padL + 6, 13);
    ctx.fillStyle = "#fbbf24";
    ctx.fillText("Exited", padL + 78, 13);

    flowSummary.innerHTML = `
        Entered: ${entrySeries[tIndex].toFixed(3)} ${flowUnitLabel()}<br>
        Exited: ${exitSeries[tIndex].toFixed(3)} ${flowUnitLabel()}
    `;
}

function updateNetworkStatus(tIndex) {
    const timeS = Number(DATA.time_s[tIndex] || 0);
    const isFlush = (DATA.is_flush || [])[tIndex] ? true : false;

    let totalVeh = 0.0;
    const x = DATA.cell_state[tIndex] || [];

    for (let c = 0; c < x.length; c++) {
        totalVeh += totalVehiclesInCell(x[c]);
    }

    let totalEntered = 0.0;
    let totalExited = 0.0;

    if (DATA.entry_flow && DATA.entry_flow.length > 0) {
        for (let i = 0; i <= tIndex; i++) totalEntered += totalFlow(DATA.entry_flow[i]);
    }

    if (DATA.exit_flow && DATA.exit_flow.length > 0) {
        for (let i = 0; i <= tIndex; i++) totalExited += totalFlow(DATA.exit_flow[i]);
    }

    networkStatus.textContent =
        `t=${timeS.toFixed(1)} s  |  step ${tIndex}/${slider.max}  |  ${totalVeh.toFixed(1)} vehicles in network  |  ` +
        `${totalEntered.toFixed(1)} entered  |  ${totalExited.toFixed(1)} exited` +
        (isFlush ? "  |  FLUSH PERIOD" : "");

    quickStats.innerHTML = `
        <span class="pill">Nodes <b>${Object.keys(DATA.nodes).length}</b></span>
        <span class="pill">Links <b>${Object.keys(DATA.links).length}</b></span>
        <span class="pill">Movements <b>${Object.keys(DATA.movements).length}</b></span>
        <span class="pill">Classes <b>${(DATA.vehicle_class_names || []).join(", ")}</b></span>
        <span class="pill">dt <b>${Number(DATA.dt || 0).toFixed(2)} s</b></span>
    `;
}

function applyDisplayToggles() {
    for (const [linkId] of Object.entries(DATA.links)) {
        const linkLabelEl = document.getElementById("linklabel_" + linkId);
        const lenLabelEl = document.getElementById("lengthlabel_" + linkId);
        const arrowEl = document.getElementById("arrow_" + linkId);

        if (linkLabelEl) linkLabelEl.style.display = showLinkIds.checked ? "" : "none";
        if (lenLabelEl) lenLabelEl.style.display = showLinkLengths.checked ? "" : "none";
        if (arrowEl) arrowEl.style.display = showArrows.checked ? "" : "none";
    }

    for (const [nodeId, node] of Object.entries(DATA.nodes)) {
        const nodeLabelEl = document.getElementById("nodelabel_" + nodeId);
        const nodeIdEl = document.getElementById("nodeidlabel_" + nodeId);
        const nodeBoxEl = document.getElementById("nodebox_" + nodeId);

        if (nodeLabelEl) nodeLabelEl.style.display = showNodeNames.checked ? "" : "none";
        if (nodeIdEl) nodeIdEl.style.display = showNodeIds.checked ? "" : "none";
        if (nodeBoxEl) nodeBoxEl.style.display = showIntersectionBoxes.checked ? "" : "none";
    }

    const stopbarLayer = document.getElementById("stopbarLayer");
    if (stopbarLayer) stopbarLayer.style.display = showStopbars.checked ? "" : "none";

    updateCellColors(Number(slider.value || 0));
}

function updateFrame(tIndex) {
    tIndex = clamp(Number(tIndex || 0), 0, Number(slider.max || 0));
    slider.value = tIndex;

    const timeS = Number(DATA.time_s[tIndex] || 0);
    timeText.textContent = `Time = ${timeS.toFixed(1)} s | step = ${tIndex} / ${slider.max}`;

    updateCellColors(tIndex);
    updateSignalStatus(tIndex);
    updateRoutingStatus(tIndex);
    updateLinkInfo(tIndex);
    updateCellDetails(tIndex);
    drawFlowPlot(tIndex);
    updateNetworkStatus(tIndex);
    applyDisplayToggles();
}

function step(delta) {
    const current = Number(slider.value || 0);
    const next = clamp(current + delta, 0, Number(slider.max || 0));
    updateFrame(next);
}

function play(direction = 1) {
    pause();
    playDirection = direction;

    timer = setInterval(() => {
        let t = Number(slider.value);
        t += playDirection;

        if (t > Number(slider.max)) t = 0;
        if (t < 0) t = Number(slider.max);

        updateFrame(t);
    }, Number(speedSelect.value));
}

function pause() {
    if (timer !== null) {
        clearInterval(timer);
        timer = null;
    }
}

function setCleanView() {
    showLinkIds.checked = false;
    showLinkLengths.checked = false;
    showNodeIds.checked = false;
    showNodeNames.checked = true;
    showArrows.checked = true;
    showStopbars.checked = false;
    showIntersectionBoxes.checked = true;
    applyDisplayToggles();
}

function setFullView() {
    showLinkIds.checked = true;
    showLinkLengths.checked = true;
    showNodeIds.checked = true;
    showNodeNames.checked = true;
    showArrows.checked = true;
    showStopbars.checked = true;
    showIntersectionBoxes.checked = true;
    applyDisplayToggles();
}

slider.addEventListener("input", () => updateFrame(Number(slider.value)));
metricSelect.addEventListener("change", () => updateFrame(Number(slider.value)));
linkSelect.addEventListener("change", () => {
    selectedLinkId = linkSelect.value;
    updateFrame(Number(slider.value));
});
flowUnit.addEventListener("change", () => drawFlowPlot(Number(slider.value)));

playButton.addEventListener("click", () => play(1));
reverseButton.addEventListener("click", () => play(-1));
pauseButton.addEventListener("click", () => pause());

firstButton.addEventListener("click", () => updateFrame(0));
lastButton.addEventListener("click", () => updateFrame(Number(slider.max)));
stepBackButton.addEventListener("click", () => step(-1));
stepForwardButton.addEventListener("click", () => step(1));
back10Button.addEventListener("click", () => step(-Math.max(1, Number(jumpSize.value || 10))));
forward10Button.addEventListener("click", () => step(Math.max(1, Number(jumpSize.value || 10))));

speedSelect.addEventListener("change", () => {
    if (timer !== null) play(playDirection);
});

for (const el of [
    showLinkIds, showLinkLengths, showNodeNames, showNodeIds,
    showArrows, showStopbars, showIntersectionBoxes, highlightSelected,
    cellWidthInput
]) {
    el.addEventListener("change", () => applyDisplayToggles());
}

cleanViewButton.addEventListener("click", setCleanView);
fullViewButton.addEventListener("click", setFullView);

for (const el of [
    signalNodeFilter, signalViewMode, showOnlyGreenSignals,
    routingNodeFilter, routingClassFilter, routingDestinationFilter,
    routingMinProb, routingGrouped
]) {
    el.addEventListener("change", () => updateFrame(Number(slider.value)));
}

document.addEventListener("keydown", (e) => {
    if (e.target && ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;

    if (e.code === "Space") {
        e.preventDefault();
        if (timer === null) play(1);
        else pause();
    }

    if (e.key === "ArrowRight") step(1);
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "Home") updateFrame(0);
    if (e.key === "End") updateFrame(Number(slider.max));
});

drawStaticBase();
initSelectors();
updateFrame(0);
</script>

</body>
</html>
"""
        return html.replace("__DATA__", json_data)
