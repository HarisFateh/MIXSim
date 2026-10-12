from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .node_type import NodeType, TurnType


# ============================================================
# Logical network objects
# ============================================================

@dataclass
class Cell:
    """
    One CTM cell.

    Units
    -----
    length_m:
        Cell length in meters.

    free_flow_speed_mps:
        Free-flow speed in meters/second.

    wave_speed_mps:
        Nominal backward congestion wave speed in meters/second. In the default
        mixed-autonomy simulator, the CTM receiving wave speed is computed
        dynamically from the local HDV/CAV composition; this field is retained
        as network metadata and for external diagnostics.

    jam_density_vpm:
        Jam density in vehicles/meter. For the default time-gap spacing
        closure with a common vehicle length of 5 m, the consistent value is
        1/5 = 0.2 veh/m.

    capacity_vps:
        Nominal cell capacity in vehicles/second. In the default simulator,
        cell sending/receiving capacity is computed dynamically from local
        HDV/CAV composition. This field is used as a fallback movement
        saturation when a Movement does not specify saturation_vps explicitly.
    """

    cell_id: int
    link_id: int
    index_on_link: int

    length_m: float
    free_flow_speed_mps: float
    wave_speed_mps: float
    jam_density_vpm: float
    capacity_vps: float


@dataclass
class Link:
    """
    Directional road link composed of CTM cells.
    """

    link_id: int
    from_node: int
    to_node: int
    cells: list[Cell] = field(default_factory=list)

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    @property
    def length_m(self) -> float:
        return float(sum(cell.length_m for cell in self.cells))

    @property
    def link_name(self) -> str:
        return f"L{self.link_id}"

    @property
    def first_cell_id(self) -> int:
        return self.cells[0].cell_id

    @property
    def last_cell_id(self) -> int:
        return self.cells[-1].cell_id


@dataclass
class Movement:
    """
    Turning movement straight a node.

    from_link and to_link are integer link IDs.
    """

    movement_id: int
    movement_name: str

    node_id: int
    from_link: int
    to_link: int

    turn_type: TurnType = TurnType.STRAIGHT
    saturation_vps: float | None = None


@dataclass
class Phase:
    """
    Signal phase.

    A phase is a set of movements that are green together.
    """

    phase_id: int
    phase_name: str
    movement_ids: list[int]

    green: float = 0.0
    yellow: float = 0.0
    all_red: float = 0.0

    @property
    def total_duration(self) -> float:
        return self.green + self.yellow + self.all_red


@dataclass
class Node:
    """
    Network node.

    A node can be an origin, destination, signalized intersection,
    or unsignalized intersection.
    """

    node_id: int
    node_type: NodeType

    node_name: str | None = None

    x: float = 0.0
    y: float = 0.0

    phases: list[Phase] = field(default_factory=list)


# ============================================================
# Compiled network
# ============================================================

@dataclass
class CompiledNetwork:
    """
    Flat NumPy version of the logical network.
    """

    n_cells: int
    n_links: int
    n_nodes: int
    n_movements: int
    n_phases: int

    cell_length_m: np.ndarray
    cell_link_id: np.ndarray
    cell_index_on_link: np.ndarray
    cell_vf_mps: np.ndarray
    cell_w_mps: np.ndarray
    cell_kj_vpm: np.ndarray
    cell_jam_density_vpm: np.ndarray
    cell_capacity_vps: np.ndarray
    cell_storage_veh: np.ndarray

    upstream_cell: np.ndarray
    downstream_cell: np.ndarray

    is_source_cell: np.ndarray
    is_sink_cell: np.ndarray

    movement_from_cell: np.ndarray
    movement_to_cell: np.ndarray
    movement_from_link: np.ndarray
    movement_to_link: np.ndarray
    movement_node_index: np.ndarray
    movement_node_id: np.ndarray
    movement_saturation_vps: np.ndarray
    movement_merge_priority: np.ndarray

    phase_node_index: np.ndarray
    phase_node_id: np.ndarray
    phase_green_s: np.ndarray
    phase_yellow_s: np.ndarray
    phase_all_red_s: np.ndarray
    phase_movement_mask: np.ndarray

    node_id_to_index: dict[int, int]
    node_index_to_id: dict[int, int]

    node_name_to_id: dict[str, int]
    node_id_to_name: dict[int, str]

    link_id_to_index: dict[int, int]
    link_index_to_id: dict[int, int]

    movement_id_to_index: dict[int, int]
    movement_index_to_id: dict[int, int]

    phase_id_to_index: dict[int, int]
    phase_index_to_id: dict[int, int]

    link_cells: dict[int, list[int]]
    link_first_cell: dict[int, int]
    link_last_cell: dict[int, int]

    node_incoming_links: dict[int, list[int]]
    node_outgoing_links: dict[int, list[int]]
    node_movements: dict[int, list[int]]
    node_phases: dict[int, list[int]]

    merge_groups: dict[int, np.ndarray]
    diverge_groups: dict[int, np.ndarray]

    node_coordinates: dict[int, tuple[float, float]]
    link_from_to: dict[int, tuple[int, int]]

    def summary(self) -> dict:
        return {
            "n_cells": self.n_cells,
            "n_links": self.n_links,
            "n_nodes": self.n_nodes,
            "n_movements": self.n_movements,
            "n_phases": self.n_phases,
            "total_length_m": float(np.sum(self.cell_length_m)),
            "total_length_km": float(np.sum(self.cell_length_m) / 1000.0),
        }


# ============================================================
# Network builder
# ============================================================

class Network:
    """
    Editable logical network.

    Internally:
        node_id: int
        link_id: int
        movement_id: int
        phase_id: int
    """

    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.links: dict[int, Link] = {}
        self.movements: dict[int, Movement] = {}

        self.node_name_to_id: dict[str, int] = {}
        self.node_id_to_name: dict[int, str] = {}

        self._next_node_id = 1
        self._next_link_id = 1
        self._next_cell_id = 0
        self._next_movement_id = 0
        self._next_phase_id = 0

    def __repr__(self) -> str:
        n_phases = sum(len(node.phases) for node in self.nodes.values())
        return (
            f"Network("
            f"{len(self.nodes)} nodes, "
            f"{len(self.links)} links, "
            f"{len(self.movements)} movements, "
            f"{n_phases} phases)"
        )

    # ------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------

    def _resolve_node_id(self, node: int | str | Node) -> int:
        if isinstance(node, Node):
            return node.node_id

        if isinstance(node, int):
            if node not in self.nodes:
                raise ValueError(f"Unknown node_id: {node}")
            return node

        if isinstance(node, str):
            if node not in self.node_name_to_id:
                raise ValueError(f"Unknown node_name: {node}")
            return self.node_name_to_id[node]

        raise TypeError(f"Invalid node reference: {node}")

    def _resolve_link_id(self, link: int | Link) -> int:
        if isinstance(link, Link):
            return link.link_id

        if isinstance(link, int):
            if link not in self.links:
                raise ValueError(f"Unknown link_id: {link}")
            return link

        raise TypeError(f"Invalid link reference: {link}")

    def _resolve_movement_id(self, movement: int | Movement) -> int:
        if isinstance(movement, Movement):
            return movement.movement_id

        if isinstance(movement, int):
            if movement not in self.movements:
                raise ValueError(f"Unknown movement_id: {movement}")
            return movement

        raise TypeError(f"Invalid movement reference: {movement}")

    # ------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------

    def add_node(
        self,
        node_type: NodeType,
        x: float = 0.0,
        y: float = 0.0,
        node_name: str | None = None,
    ) -> Node:
        node_id = self._next_node_id
        self._next_node_id += 1

        if node_name is None:
            node_name = f"N{node_id}"

        if node_name in self.node_name_to_id:
            raise ValueError(f"Node name already exists: {node_name}")

        node = Node(
            node_id=node_id,
            node_type=node_type,
            x=x,
            y=y,
            node_name=node_name,
        )

        self.nodes[node_id] = node
        self.node_name_to_id[node_name] = node_id
        self.node_id_to_name[node_id] = node_name

        return node

    def add_link(
        self,
        from_node: int | str,
        to_node: int | str,
        length_m: float,
        n_cells: int,
        free_flow_speed_mps: float = 8.33,
        wave_speed_mps: float = 4.17,
        jam_density_vpm: float = 1.0 / 5.0,
        capacity_vps: float = 0.5,
    ) -> Link:

        if isinstance(from_node, str):
            if from_node not in self.node_name_to_id:
                raise ValueError(f"Unknown from_node name: {from_node}")
            from_node = self.node_name_to_id[from_node]

        if isinstance(to_node, str):
            if to_node not in self.node_name_to_id:
                raise ValueError(f"Unknown to_node name: {to_node}")
            to_node = self.node_name_to_id[to_node]

        if from_node not in self.nodes:
            raise ValueError(f"Unknown from_node: {from_node}")

        if to_node not in self.nodes:
            raise ValueError(f"Unknown to_node: {to_node}")

        if length_m <= 0:
            raise ValueError("length_m must be positive")

        if n_cells <= 0:
            raise ValueError("n_cells must be positive")

        link_id = self._next_link_id
        self._next_link_id += 1

        cell_length = length_m / n_cells

        link = Link(
            link_id=link_id,
            from_node=from_node,
            to_node=to_node,
        )

        for i in range(n_cells):
            cell = Cell(
                cell_id=self._next_cell_id,
                link_id=link_id,
                index_on_link=i,
                length_m=cell_length,
                free_flow_speed_mps=free_flow_speed_mps,
                wave_speed_mps=wave_speed_mps,
                jam_density_vpm=jam_density_vpm,
                capacity_vps=capacity_vps,
            )

            link.cells.append(cell)
            self._next_cell_id += 1

        self.links[link_id] = link
        return link

    def add_movement(
        self,
        movement_name: str,
        node_id: int | Node,
        from_link: int | Link,
        to_link: int | Link,
        turn_type: TurnType = TurnType.STRAIGHT,
        saturation_vps: float | None = None,
    ) -> Movement:
        node_id = self._resolve_node_id(node_id)
        from_link_id = self._resolve_link_id(from_link)
        to_link_id = self._resolve_link_id(to_link)

        from_link_obj = self.links[from_link_id]
        to_link_obj = self.links[to_link_id]

        if from_link_obj.to_node != node_id:
            raise ValueError(
                f"Movement {movement_name} is invalid: "
                f"from_link {from_link_id} does not end at node {node_id}"
            )

        if to_link_obj.from_node != node_id:
            raise ValueError(
                f"Movement {movement_name} is invalid: "
                f"to_link {to_link_id} does not start at node {node_id}"
            )

        movement = Movement(
            movement_id=self._next_movement_id,
            movement_name=movement_name,
            node_id=node_id,
            from_link=from_link_id,
            to_link=to_link_id,
            turn_type=turn_type,
            saturation_vps=saturation_vps,
        )

        self.movements[movement.movement_id] = movement
        self._next_movement_id += 1

        return movement

    def add_phase(
        self,
        node_id: int | Node,
        phase_name: str,
        movement_ids: list[int | Movement],
        green: float,
        yellow: float = 0.0,
        all_red: float = 0.0,
    ) -> Phase:
        node_id = self._resolve_node_id(node_id)

        if green <= 0:
            raise ValueError("green must be positive")

        resolved_movement_ids = [
            self._resolve_movement_id(movement)
            for movement in movement_ids
        ]

        for movement_id in resolved_movement_ids:
            movement = self.movements[movement_id]

            if movement.node_id != node_id:
                raise ValueError(
                    f"Movement {movement_id} belongs to node "
                    f"{movement.node_id}, not node {node_id}"
                )

        phase = Phase(
            phase_id=self._next_phase_id,
            phase_name=phase_name,
            movement_ids=resolved_movement_ids,
            green=green,
            yellow=yellow,
            all_red=all_red,
        )

        self.nodes[node_id].phases.append(phase)
        self._next_phase_id += 1

        return phase

    # ------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------

    def incoming_links(self, node_id: int | Node) -> list[int]:
        node_id = self._resolve_node_id(node_id)
        return [
            link_id
            for link_id, link in self.links.items()
            if link.to_node == node_id
        ]

    def outgoing_links(self, node_id: int | Node) -> list[int]:
        node_id = self._resolve_node_id(node_id)
        return [
            link_id
            for link_id, link in self.links.items()
            if link.from_node == node_id
        ]

    def movements_from_link(self, link: int | Link) -> list[Movement]:
        link_id = self._resolve_link_id(link)
        return [
            movement
            for movement in self.movements.values()
            if movement.from_link == link_id
        ]

    def movements_to_link(self, link: int | Link) -> list[Movement]:
        link_id = self._resolve_link_id(link)
        return [
            movement
            for movement in self.movements.values()
            if movement.to_link == link_id
        ]

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []

        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.SIGNALIZED and len(node.phases) == 0:
                errors.append(f"Node {node_id} is signalized but has no phases")

        for link_id, link in self.links.items():
            if link.from_node not in self.nodes:
                errors.append(
                    f"Link {link_id} has unknown from_node {link.from_node}"
                )

            if link.to_node not in self.nodes:
                errors.append(
                    f"Link {link_id} has unknown to_node {link.to_node}"
                )

            if link.num_cells == 0:
                errors.append(f"Link {link_id} has no cells")

            for cell in link.cells:
                if cell.length_m <= 0:
                    errors.append(f"Cell {cell.cell_id} has non-positive length")

                if cell.free_flow_speed_mps <= 0:
                    errors.append(
                        f"Cell {cell.cell_id} has non-positive free-flow speed"
                    )

                if cell.wave_speed_mps <= 0:
                    errors.append(
                        f"Cell {cell.cell_id} has non-positive wave speed"
                    )

                if cell.jam_density_vpm <= 0:
                    errors.append(
                        f"Cell {cell.cell_id} has non-positive jam density"
                    )

                if cell.capacity_vps <= 0:
                    errors.append(
                        f"Cell {cell.cell_id} has non-positive capacity"
                    )

        for movement_id, movement in self.movements.items():
            if movement.from_link not in self.links:
                errors.append(f"Movement {movement_id} has unknown from_link")

            if movement.to_link not in self.links:
                errors.append(f"Movement {movement_id} has unknown to_link")

            if movement.node_id not in self.nodes:
                errors.append(f"Movement {movement_id} has unknown node")

        return errors

    # ------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------

    def compile(self, dt: float, *, max_wave_speed_mps: float | None = None) -> CompiledNetwork:
        errors = self.validate()

        if errors:
            msg = "\n".join(errors)
            raise ValueError(f"Network validation failed:\n{msg}")

        all_cells: list[Cell] = []

        for link in self.links.values():
            all_cells.extend(link.cells)

        all_cells = sorted(all_cells, key=lambda c: c.cell_id)

        n_cells = len(all_cells)
        n_links = len(self.links)
        n_nodes = len(self.nodes)
        n_movements = len(self.movements)

        node_id_to_index = {
            node_id: i
            for i, node_id in enumerate(self.nodes.keys())
        }

        node_index_to_id = {
            i: node_id
            for node_id, i in node_id_to_index.items()
        }

        link_id_to_index = {
            link_id: i
            for i, link_id in enumerate(self.links.keys())
        }

        link_index_to_id = {
            i: link_id
            for link_id, i in link_id_to_index.items()
        }

        movement_id_to_index = {
            movement_id: i
            for i, movement_id in enumerate(self.movements.keys())
        }

        movement_index_to_id = {
            i: movement_id
            for movement_id, i in movement_id_to_index.items()
        }

        cell_length_m = np.array(
            [cell.length_m for cell in all_cells],
            dtype=float,
        )

        cell_link_id = np.array(
            [cell.link_id for cell in all_cells],
            dtype=int,
        )

        cell_index_on_link = np.array(
            [cell.index_on_link for cell in all_cells],
            dtype=int,
        )

        cell_vf_mps = np.array(
            [cell.free_flow_speed_mps for cell in all_cells],
            dtype=float,
        )

        cell_w_mps = np.array(
            [cell.wave_speed_mps for cell in all_cells],
            dtype=float,
        )

        cell_kj_vpm = np.array(
            [cell.jam_density_vpm for cell in all_cells],
            dtype=float,
        )

        cell_capacity_vps = np.array(
            [cell.capacity_vps for cell in all_cells],
            dtype=float,
        )

        cell_storage_veh = cell_kj_vpm * cell_length_m

        cfl_speed_mps = cell_vf_mps.copy()
        if max_wave_speed_mps is not None:
            cfl_speed_mps = np.maximum(cfl_speed_mps, float(max_wave_speed_mps))
        if np.any(cfl_speed_mps * dt > cell_length_m + 1e-9):
            bad = np.where(cfl_speed_mps * dt > cell_length_m + 1e-9)[0]
            raise ValueError(
                "CFL condition violated. "
                f"Bad cells: {bad.tolist()}. "
                "Need cell_length >= max(vf, w_max) * dt when a maximum "
                "wave speed is supplied, otherwise cell_length >= vf * dt."
            )

        upstream_cell = np.full(n_cells, -1, dtype=int)
        downstream_cell = np.full(n_cells, -1, dtype=int)

        link_cells: dict[int, list[int]] = {}
        link_first_cell: dict[int, int] = {}
        link_last_cell: dict[int, int] = {}

        for link_id, link in self.links.items():
            cids = [cell.cell_id for cell in link.cells]

            link_cells[link_id] = cids
            link_first_cell[link_id] = cids[0]
            link_last_cell[link_id] = cids[-1]

            for i, cell_id in enumerate(cids):
                if i > 0:
                    upstream_cell[cell_id] = cids[i - 1]

                if i < len(cids) - 1:
                    downstream_cell[cell_id] = cids[i + 1]

        is_source_cell = np.zeros(n_cells, dtype=bool)
        is_sink_cell = np.zeros(n_cells, dtype=bool)

        for link_id, link in self.links.items():
            from_node = self.nodes[link.from_node]
            to_node = self.nodes[link.to_node]

            if from_node.node_type == NodeType.ORIGIN:
                is_source_cell[link_first_cell[link_id]] = True

            if to_node.node_type == NodeType.DESTINATION:
                is_sink_cell[link_last_cell[link_id]] = True

        movement_from_cell = np.zeros(n_movements, dtype=int)
        movement_to_cell = np.zeros(n_movements, dtype=int)
        movement_from_link = np.zeros(n_movements, dtype=int)
        movement_to_link = np.zeros(n_movements, dtype=int)
        movement_node_index = np.zeros(n_movements, dtype=int)
        movement_node_id = np.zeros(n_movements, dtype=int)
        movement_saturation_vps = np.zeros(n_movements, dtype=float)
        movement_merge_priority = np.ones(n_movements, dtype=float)

        node_movements: dict[int, list[int]] = {}

        for movement_id, movement in self.movements.items():
            m_idx = movement_id_to_index[movement_id]

            from_cell = link_last_cell[movement.from_link]
            to_cell = link_first_cell[movement.to_link]

            movement_from_cell[m_idx] = from_cell
            movement_to_cell[m_idx] = to_cell
            movement_from_link[m_idx] = movement.from_link
            movement_to_link[m_idx] = movement.to_link
            movement_node_index[m_idx] = node_id_to_index[movement.node_id]
            movement_node_id[m_idx] = movement.node_id

            if movement.saturation_vps is None:
                movement_saturation_vps[m_idx] = cell_capacity_vps[from_cell]
            else:
                movement_saturation_vps[m_idx] = movement.saturation_vps

            node_movements.setdefault(movement.node_id, []).append(m_idx)

        all_phases: list[tuple[int, Phase]] = []

        for node_id, node in self.nodes.items():
            for phase in node.phases:
                all_phases.append((node_id, phase))

        n_phases = len(all_phases)

        phase_id_to_index: dict[int, int] = {}
        phase_index_to_id: dict[int, int] = {}

        phase_node_index = np.zeros(n_phases, dtype=int)
        phase_node_id = np.zeros(n_phases, dtype=int)
        phase_green_s = np.zeros(n_phases, dtype=float)
        phase_yellow_s = np.zeros(n_phases, dtype=float)
        phase_all_red_s = np.zeros(n_phases, dtype=float)

        phase_movement_mask = np.zeros(
            (n_phases, n_movements),
            dtype=bool,
        )

        node_phases: dict[int, list[int]] = {}

        for p_idx, (node_id, phase) in enumerate(all_phases):
            phase_id_to_index[phase.phase_id] = p_idx
            phase_index_to_id[p_idx] = phase.phase_id

            phase_node_index[p_idx] = node_id_to_index[node_id]
            phase_node_id[p_idx] = node_id
            phase_green_s[p_idx] = phase.green
            phase_yellow_s[p_idx] = phase.yellow
            phase_all_red_s[p_idx] = phase.all_red

            node_phases.setdefault(node_id, []).append(p_idx)

            for movement_id in phase.movement_ids:
                m_idx = movement_id_to_index[movement_id]
                phase_movement_mask[p_idx, m_idx] = True

        node_incoming_links: dict[int, list[int]] = {}
        node_outgoing_links: dict[int, list[int]] = {}

        for link_id, link in self.links.items():
            node_incoming_links.setdefault(link.to_node, []).append(link_id)
            node_outgoing_links.setdefault(link.from_node, []).append(link_id)

        for node_id in node_incoming_links:
            node_incoming_links[node_id].sort()

        for node_id in node_outgoing_links:
            node_outgoing_links[node_id].sort()

        merge_groups: dict[int, list[int]] = {}
        diverge_groups: dict[int, list[int]] = {}

        for m_idx in range(n_movements):
            to_cell = int(movement_to_cell[m_idx])
            from_cell = int(movement_from_cell[m_idx])

            merge_groups.setdefault(to_cell, []).append(m_idx)
            diverge_groups.setdefault(from_cell, []).append(m_idx)

        merge_groups_np = {
            cell_id: np.array(movement_indices, dtype=int)
            for cell_id, movement_indices in merge_groups.items()
        }

        diverge_groups_np = {
            cell_id: np.array(movement_indices, dtype=int)
            for cell_id, movement_indices in diverge_groups.items()
        }

        node_coordinates = {
            node_id: (node.x, node.y)
            for node_id, node in self.nodes.items()
        }

        link_from_to = {
            link_id: (link.from_node, link.to_node)
            for link_id, link in self.links.items()
        }

        return CompiledNetwork(
            n_cells=n_cells,
            n_links=n_links,
            n_nodes=n_nodes,
            n_movements=n_movements,
            n_phases=n_phases,

            cell_length_m=cell_length_m,
            cell_link_id=cell_link_id,
            cell_index_on_link=cell_index_on_link,
            cell_vf_mps=cell_vf_mps,
            cell_w_mps=cell_w_mps,
            cell_kj_vpm=cell_kj_vpm,
            cell_jam_density_vpm=cell_kj_vpm,
            cell_capacity_vps=cell_capacity_vps,
            cell_storage_veh=cell_storage_veh,

            upstream_cell=upstream_cell,
            downstream_cell=downstream_cell,

            is_source_cell=is_source_cell,
            is_sink_cell=is_sink_cell,

            movement_from_cell=movement_from_cell,
            movement_to_cell=movement_to_cell,
            movement_from_link=movement_from_link,
            movement_to_link=movement_to_link,
            movement_node_index=movement_node_index,
            movement_node_id=movement_node_id,
            movement_saturation_vps=movement_saturation_vps,
            movement_merge_priority=movement_merge_priority,

            phase_node_index=phase_node_index,
            phase_node_id=phase_node_id,
            phase_green_s=phase_green_s,
            phase_yellow_s=phase_yellow_s,
            phase_all_red_s=phase_all_red_s,
            phase_movement_mask=phase_movement_mask,

            node_id_to_index=node_id_to_index,
            node_index_to_id=node_index_to_id,

            node_name_to_id=self.node_name_to_id.copy(),
            node_id_to_name=self.node_id_to_name.copy(),

            link_id_to_index=link_id_to_index,
            link_index_to_id=link_index_to_id,

            movement_id_to_index=movement_id_to_index,
            movement_index_to_id=movement_index_to_id,

            phase_id_to_index=phase_id_to_index,
            phase_index_to_id=phase_index_to_id,

            link_cells=link_cells,
            link_first_cell=link_first_cell,
            link_last_cell=link_last_cell,

            node_incoming_links=node_incoming_links,
            node_outgoing_links=node_outgoing_links,
            node_movements=node_movements,
            node_phases=node_phases,

            merge_groups=merge_groups_np,
            diverge_groups=diverge_groups_np,

            node_coordinates=node_coordinates,
            link_from_to=link_from_to,
        )