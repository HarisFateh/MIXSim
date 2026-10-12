from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .network import Network, CompiledNetwork
from .vehicle_type import VehicleClass
from .demand import Demand, Commodity
from .signal_controller import SignalController, FixedSignalController
from .saturation_controller import SaturationController, MovementCompositionSaturationController


@dataclass
class SimulationHistory:
    time_s: list[float] = field(default_factory=list)

    # class_state[t][cell, class]
    cell_state: list[np.ndarray] = field(default_factory=list)

    # commodity_state[t][cell, commodity]
    commodity_state: list[np.ndarray] = field(default_factory=list)

    link_vehicle_count: list[dict[int, float]] = field(default_factory=list)
    link_density_ratio: list[dict[int, float]] = field(default_factory=list)

    movement_flow: list[dict[int, float]] = field(default_factory=list)
    movement_flow_by_commodity: list[dict[int, list[float]]] = field(default_factory=list)

    entry_flow: list[dict[int, float]] = field(default_factory=list)
    entry_flow_by_commodity: list[dict[int, list[float]]] = field(default_factory=list)

    exit_flow: list[dict[int, float]] = field(default_factory=list)
    exit_flow_by_commodity: list[dict[int, list[float]]] = field(default_factory=list)

    residual: list[float] = field(default_factory=list)
    virtual_queue_total: list[float] = field(default_factory=list)

    # Interior CTM boundary flows, recorded per step after each update.
    # Keys use "link_id:boundary_index", where boundary_index=1 means
    # cell 1 -> cell 2 on that link. Values are vehicles per simulation step.
    interior_flow: list[dict[str, float]] = field(default_factory=list)

    # Same data with useful metadata for FD plotting/export.
    interior_flow_records: list[list[dict]] = field(default_factory=list)

    is_flush: list[bool] = field(default_factory=list)

    # Signal state recorded at each saved frame. Keys are logical movement ids.
    # Values are True when the movement was green during the interval leading
    # to this frame. For the initial frame, values are evaluated at t=0.
    signal_green: list[dict[int, bool]] = field(default_factory=list)

    # Active logical phase id by node, when available. For fixed-time plans,
    # this is inferred from the movement greens. For adaptive controllers, it
    # is read from the controller state at the time of the CTM step.
    active_phase_by_node: list[dict[int, int | None]] = field(default_factory=list)

    # Routing table recorded at each saved frame. Each row is a visible
    # movement/commodity probability, so HTML playback can show route changes
    # without recomputing assignment after the simulation has finished.
    routing_probabilities: list[list[dict]] = field(default_factory=list)


class CTMSimulator:
    """
    OD-specific multiclass CTM with node-based movement assignment.

    State:
        x[cell, commodity]

    No movement labels are attached to cells. At each node, last-cell sending is
    split into desired movement demand by routing probabilities. The simulator
    subtracts only realized movement flow, so blocked desired flow remains in
    the upstream last cell and is not redistributed in the same step.
    """

    def __init__(
        self,
        network: Network,
        vehicle_classes: list[VehicleClass],
        demand: Demand,
        routing: Demand,
        signal_controller: SignalController | None = None,
        saturation_controller: SaturationController | None = None,
        dt: float = 5.0,
        merge_allocation: str = "proportional",
        check_jam_density_consistency: bool = True,
        reference_class_share_by_cell: np.ndarray | None = None,
        capture_routing_history: bool = True,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.raw_network = network
        self.vehicle_classes = list(vehicle_classes)
        if not self.vehicle_classes:
            raise ValueError("At least one vehicle class is required")
        _pre_lengths = np.array([vc.vehicle_length_m for vc in self.vehicle_classes], dtype=float)
        _pre_tau = np.array([vc.time_gap_s for vc in self.vehicle_classes], dtype=float)
        if np.any(_pre_lengths <= 0) or np.any(_pre_tau <= 0):
            raise ValueError("Vehicle-class time gaps and effective standstill spacings must be positive")
        if not np.allclose(_pre_lengths, _pre_lengths[0], rtol=0, atol=1e-12):
            raise ValueError(
                "The default storage model assumes one common vehicle length. "
                "Use equal vehicle_length_m values or implement a space-based "
                "class-specific storage extension."
            )
        _pre_wmax = float(_pre_lengths[0] / np.min(_pre_tau))
        self.net: CompiledNetwork = network.compile(dt, max_wave_speed_mps=_pre_wmax)
        self.class_names = [vc.name for vc in self.vehicle_classes]
        self.n_classes = len(self.vehicle_classes)
        self.demand = demand
        self.routing = routing
        self.signal_controller = signal_controller or FixedSignalController()
        self.saturation_controller = saturation_controller or MovementCompositionSaturationController()
        self.dt = float(dt)
        if merge_allocation not in {"proportional", "priority"}:
            raise ValueError("merge_allocation must be 'proportional' or 'priority'")
        self.merge_allocation = merge_allocation
        self.check_jam_density_consistency = bool(check_jam_density_consistency)
        self.capture_routing_history = bool(capture_routing_history)

        if hasattr(self.demand, "prepare_commodities"):
            self.commodities: list[Commodity] = self.demand.prepare_commodities(
                network=self.raw_network,
                vehicle_classes=self.vehicle_classes,
            )
        else:
            raise TypeError("demand must provide prepare_commodities")

        if not self.commodities:
            raise ValueError("No commodities were created. Add OD demand profiles first.")

        self.n_commodities = len(self.commodities)
        self.commodity_class_index = np.array(
            [c.class_index for c in self.commodities], dtype=int
        )
        self.commodity_origin_link = np.array(
            [c.origin_link for c in self.commodities], dtype=int
        )
        self.commodity_destination_node = np.array(
            [c.destination_node for c in self.commodities], dtype=int
        )

        raw_ref = np.array([vc.reference_share for vc in self.vehicle_classes], dtype=float)
        if raw_ref.sum() <= 0:
            raw_ref = np.ones(self.n_classes, dtype=float)
        self.reference_class_share = raw_ref / raw_ref.sum()

        if reference_class_share_by_cell is None:
            self.reference_class_share_by_cell = np.tile(
                self.reference_class_share[None, :],
                (self.net.n_cells, 1),
            )
        else:
            ref_cell = np.asarray(reference_class_share_by_cell, dtype=float)
            if ref_cell.shape != (self.net.n_cells, self.n_classes):
                raise ValueError(
                    "reference_class_share_by_cell must have shape "
                    f"(n_cells, n_classes) = {(self.net.n_cells, self.n_classes)}"
                )
            if np.any(ref_cell < 0):
                raise ValueError("reference_class_share_by_cell cannot contain negative values")
            row_sum = ref_cell.sum(axis=1)
            if np.any(row_sum <= 0):
                raise ValueError("Each reference_class_share_by_cell row must have positive sum")
            self.reference_class_share_by_cell = ref_cell / row_sum[:, None]

        self.time_gap_s = np.array(
            [vc.time_gap_s for vc in self.vehicle_classes], dtype=float
        )
        self.vehicle_length_m = np.array(
            [vc.vehicle_length_m for vc in self.vehicle_classes], dtype=float
        )

        if np.any(self.time_gap_s <= 0):
            raise ValueError("All vehicle-class time gaps must be positive")
        if np.any(self.vehicle_length_m <= 0):
            raise ValueError("All vehicle lengths must be positive")

        self._check_common_storage_model()
        self._check_cfl_condition()

        self.x = np.zeros((self.net.n_cells, self.n_commodities), dtype=float)
        self.virtual_queue = np.zeros(self.n_commodities, dtype=float)

        self.cumulative_requested = 0.0
        self.cumulative_exited = 0.0
        self.initial_system_vehicles = 0.0

        self.history = SimulationHistory()


    def _check_common_storage_model(self) -> None:
        """Check jam-density consistency with the time-gap spacing model.

        The default MIXSim formulation uses a common physical vehicle
        length ell_v and therefore the jam density implied by
        L_m(u) = u tau_m + ell_v is rho_jam = 1 / ell_v.  This check prevents
        qmax, w, and storage from becoming three inconsistent knobs.
        """
        if not np.allclose(self.vehicle_length_m, self.vehicle_length_m[0], rtol=0, atol=1e-12):
            raise ValueError(
                "The default storage model assumes one common vehicle length. "
                "Use equal vehicle_length_m values or implement a space-based "
                "class-specific storage extension."
            )
        self.common_vehicle_length_m = float(self.vehicle_length_m[0])
        implied = 1.0 / self.common_vehicle_length_m
        self.implied_jam_density_vpm = implied
        if self.check_jam_density_consistency:
            err = np.max(np.abs(self.net.cell_jam_density_vpm - implied))
            if err > 1e-9:
                raise ValueError(
                    "Inconsistent jam density. The time-gap spacing model "
                    "requires rho_jam = 1 / vehicle_length_m = "
                    f"{implied:.12g} veh/m, but the network contains values "
                    f"from {self.net.cell_jam_density_vpm.min():.12g} to "
                    f"{self.net.cell_jam_density_vpm.max():.12g}."
                )

    def _check_cfl_condition(self) -> None:
        """Validate the conservative CTM CFL condition.

        The simulator uses the general sending and receiving functions

            S = min((vf dt / dx) n, qmax dt)
            R = min(qmax dt, (w dt / dx) free_space).

        For numerical consistency we require both free-flow and maximum
        backward-wave travel distances during one step to be no longer than
        the cell length.  The maximum wave speed is taken over vehicle classes
        because mixed composition can vary during a run.
        """
        # Worst case over all possible local compositions.  For HDV/CAV this
        # is the all-CAV cell when tau_CAV is the minimum time gap.
        max_wave = float(self.common_vehicle_length_m / np.min(self.time_gap_s))
        self.max_backward_wave_mps = max_wave
        max_speed = np.maximum(self.net.cell_vf_mps, max_wave)
        violation = max_speed * self.dt - self.net.cell_length_m
        if np.max(violation) > 1.0e-9:
            bad = np.where(violation > 1.0e-9)[0].tolist()
            raise ValueError(
                "CFL condition violated for cells "
                f"{bad}: require max(vf, w_max) * dt <= dx. "
                "Use a smaller dt or longer cells."
            )

    # ------------------------------------------------------------------
    # Aggregation and CTM functions
    # ------------------------------------------------------------------

    def class_state_from_commodity(self, x: np.ndarray | None = None) -> np.ndarray:
        x = self.x if x is None else x
        out = np.zeros((self.net.n_cells, self.n_classes), dtype=float)
        for q in range(self.n_commodities):
            out[:, self.commodity_class_index[q]] += x[:, q]
        return out

    def cell_class_shares(self, x_class: np.ndarray) -> np.ndarray:
        total = x_class.sum(axis=1)
        shares = np.zeros_like(x_class)
        nonzero = total > 1e-12
        shares[nonzero] = x_class[nonzero] / total[nonzero, None]
        shares[~nonzero] = self.reference_class_share_by_cell[~nonzero]
        return shares

    def effective_parameters(self, x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return dynamic capacity qmax[cell] in veh/s and wave speed w[cell] in m/s."""
        x = self.x if x is None else x
        x_class = self.class_state_from_commodity(x)
        shares = self.cell_class_shares(x_class)

        tau_bar = shares @ self.time_gap_s
        vf = self.net.cell_vf_mps
        ell_v = self.common_vehicle_length_m

        qmax = vf / (vf * tau_bar + ell_v)
        w = ell_v / tau_bar
        return qmax, w

    def sending(self, x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return aggregate and commodity sending arrays."""
        x = self.x if x is None else x
        total = x.sum(axis=1)
        qmax, _ = self.effective_parameters(x)
        move_fraction = np.minimum(1.0, self.net.cell_vf_mps * self.dt / self.net.cell_length_m)
        desired_total = total * move_fraction
        cap = qmax * self.dt
        S_total = np.minimum(desired_total, cap)

        S = np.zeros_like(x)
        nz = total > 1e-12
        S[nz] = x[nz] * (S_total[nz] / total[nz])[:, None]
        return S_total, S

    def receiving(self, x: np.ndarray | None = None) -> np.ndarray:
        """Return aggregate receiving capacity in vehicles per simulation step.

        Important unit correction:
            R_i = min(qmax_i * dt, w_i * dt * (rho_jam_i - rho_i))

        Since free_space = (rho_jam_i - rho_i) * dx_i, this is equivalent to

            R_i = min(qmax_i * dt, (w_i * dt / dx_i) * free_space).

        The older expression (w/vf) * free_space is only valid when
        dx_i = vf_i * dt. Here the code enforces the CFL condition
        max(vf_i, w_max) * dt <= dx_i, so using w/vf would overestimate
        receiving for longer cells and would distort FD diagnostics.
        """
        x = self.x if x is None else x
        total = x.sum(axis=1)
        qmax, w = self.effective_parameters(x)
        free_space = np.maximum(0.0, self.net.cell_storage_veh - total)
        storage_term = (w * self.dt / self.net.cell_length_m) * free_space
        return np.minimum(qmax * self.dt, storage_term)

    # ------------------------------------------------------------------
    # Diagnostics/history
    # ------------------------------------------------------------------

    def _link_vehicle_count(self, x: np.ndarray) -> dict[int, float]:
        out = {}
        for link_id, cells in self.net.link_cells.items():
            out[link_id] = float(np.sum(x[cells, :]))
        return out

    def _link_density_ratio(self, x: np.ndarray) -> dict[int, float]:
        out = {}
        total_per_cell = x.sum(axis=1)
        for link_id, cells in self.net.link_cells.items():
            vehicles = float(np.sum(total_per_cell[cells]))
            storage = float(np.sum(self.net.cell_storage_veh[cells]))
            out[link_id] = vehicles / storage if storage > 0 else 0.0
        return out

    def conservation_residual(self) -> float:
        A = float(np.sum(self.x))
        B = float(np.sum(self.virtual_queue))
        return A + B - self.initial_system_vehicles - self.cumulative_requested + self.cumulative_exited

    def _capture_signal_state(self, time_s: float) -> tuple[dict[int, bool], dict[int, int | None]]:
        """Capture signal greens and active phases for browser playback.

        The HTML visualizer should not reconstruct adaptive signal states from
        the controller after the run, because stateful controllers such as
        MaxPressure may have changed their internal active phase many times.
        This method records the signal state used during the CTM step.
        """
        movement_green: dict[int, bool] = {}
        for m_idx in range(self.net.n_movements):
            movement_id = int(self.net.movement_index_to_id[m_idx])
            try:
                movement_green[movement_id] = bool(
                    self.signal_controller.movement_is_green(self.net, m_idx, float(time_s))
                )
            except Exception:
                movement_green[movement_id] = False

        active_phase_by_node: dict[int, int | None] = {}
        controller_active = getattr(self.signal_controller, "active_phase_by_node", {})
        for node_id, phase_indices in self.net.node_phases.items():
            if not phase_indices:
                continue

            phase_index = controller_active.get(node_id, None) if isinstance(controller_active, dict) else None
            if phase_index is not None:
                try:
                    active_phase_by_node[int(node_id)] = int(self.net.phase_index_to_id[int(phase_index)])
                    continue
                except Exception:
                    pass

            active_phase_id: int | None = None
            for p_idx in phase_indices:
                mask = self.net.phase_movement_mask[int(p_idx)]
                served = [int(self.net.movement_index_to_id[m]) for m in np.where(mask)[0]]
                if served and any(movement_green.get(mid, False) for mid in served):
                    active_phase_id = int(self.net.phase_index_to_id[int(p_idx)])
                    break
            active_phase_by_node[int(node_id)] = active_phase_id

        return movement_green, active_phase_by_node

    def _capture_routing_state(self, time_s: float, *, min_probability: float = 1e-10) -> list[dict]:
        """Capture visible OD-routing probabilities for browser playback.

        The raw routing model is commodity-specific. To keep the HTML file
        compact, this method aggregates commodities by movement, vehicle class,
        and destination. For dynamic-shortest routing the probabilities are the
        same for commodities that share a destination at the same link, so the
        max/representative value is sufficient for visualization.
        """
        if not hasattr(self.routing, "probability"):
            return []

        grouped: dict[tuple[int, str, int], dict] = {}
        for movement_id, movement in self.raw_network.movements.items():
            for commodity in self.commodities:
                try:
                    p = float(
                        self.routing.probability(
                            from_link=int(movement.from_link),
                            to_link=int(movement.to_link),
                            class_name=commodity.class_name,
                            time_s=float(time_s),
                            commodity_id=int(commodity.commodity_id),
                            destination_node=int(commodity.destination_node),
                        )
                    )
                except TypeError:
                    try:
                        p = float(
                            self.routing.probability(
                                from_link=int(movement.from_link),
                                to_link=int(movement.to_link),
                                class_name=commodity.class_name,
                                time_s=float(time_s),
                            )
                        )
                    except Exception:
                        p = 0.0
                except Exception:
                    p = 0.0
                if p <= min_probability:
                    continue

                key = (int(movement_id), str(commodity.class_name), int(commodity.destination_node))
                if key not in grouped:
                    grouped[key] = {
                        "movement_id": int(movement_id),
                        "node_id": int(movement.node_id),
                        "from_link": int(movement.from_link),
                        "to_link": int(movement.to_link),
                        "class_name": str(commodity.class_name),
                        "destination_node": int(commodity.destination_node),
                        "destination_name": str(commodity.destination_name),
                        "probability": float(p),
                        "commodity_count": 1,
                    }
                else:
                    grouped[key]["probability"] = max(float(grouped[key]["probability"]), float(p))
                    grouped[key]["commodity_count"] = int(grouped[key].get("commodity_count", 1)) + 1

        return list(grouped.values())

    def save_state(
        self,
        time_s: float,
        *,
        movement_flow: dict[int, float] | None = None,
        movement_flow_by_commodity: dict[int, list[float]] | None = None,
        entry_flow: dict[int, float] | None = None,
        entry_flow_by_commodity: dict[int, list[float]] | None = None,
        exit_flow: dict[int, float] | None = None,
        exit_flow_by_commodity: dict[int, list[float]] | None = None,
        interior_flow: dict[str, float] | None = None,
        interior_flow_records: list[dict] | None = None,
        is_flush: bool = False,
        signal_green: dict[int, bool] | None = None,
        active_phase_by_node: dict[int, int | None] | None = None,
        routing_probabilities: list[dict] | None = None,
    ) -> None:
        self.history.time_s.append(float(time_s))
        self.history.commodity_state.append(self.x.copy())
        self.history.cell_state.append(self.class_state_from_commodity(self.x))
        self.history.link_vehicle_count.append(self._link_vehicle_count(self.x))
        self.history.link_density_ratio.append(self._link_density_ratio(self.x))
        self.history.movement_flow.append(movement_flow or {})
        self.history.movement_flow_by_commodity.append(movement_flow_by_commodity or {})
        self.history.entry_flow.append(entry_flow or {})
        self.history.entry_flow_by_commodity.append(entry_flow_by_commodity or {})
        self.history.exit_flow.append(exit_flow or {})
        self.history.exit_flow_by_commodity.append(exit_flow_by_commodity or {})
        self.history.interior_flow.append(interior_flow or {})
        self.history.interior_flow_records.append(interior_flow_records or [])
        self.history.residual.append(self.conservation_residual())
        self.history.virtual_queue_total.append(float(np.sum(self.virtual_queue)))
        self.history.is_flush.append(bool(is_flush))
        self.history.signal_green.append(dict(signal_green or {}))
        self.history.active_phase_by_node.append(dict(active_phase_by_node or {}))
        self.history.routing_probabilities.append(list(routing_probabilities or []))

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------


    def _allocate_link_supply(
        self,
        *,
        supply: float,
        source_vectors: list[np.ndarray],
        priorities: list[float],
    ) -> list[np.ndarray]:
        """Allocate one downstream first-cell supply among candidate sources.

        ``proportional`` is the original fair-share CTM rule: every candidate
        source receives the same scalar supply factor.  ``priority`` uses an
        iterative priority-weighted allocation over aggregate source demands
        and then preserves each source's commodity composition.
        """
        demands = np.array([float(np.sum(v)) for v in source_vectors], dtype=float)
        total = float(np.sum(demands))
        if total <= 1e-12:
            return [np.zeros_like(v) for v in source_vectors]
        if supply >= total - 1e-12:
            return [v.copy() for v in source_vectors]
        supply = max(0.0, float(supply))

        if self.merge_allocation == "proportional":
            chi = supply / total if total > 0 else 0.0
            return [chi * v for v in source_vectors]

        # Priority-weighted allocation with redistribution when a source's
        # demand is below its priority share.
        priorities_arr = np.array(priorities, dtype=float)
        if np.any(priorities_arr <= 0):
            raise ValueError("All merge priorities must be positive")
        allocated = np.zeros_like(demands)
        active = demands > 1e-12
        remaining_supply = supply
        while remaining_supply > 1e-12 and np.any(active):
            weight_sum = float(np.sum(priorities_arr[active]))
            if weight_sum <= 0:
                break
            progressed = False
            for idx in np.where(active)[0]:
                share = remaining_supply * priorities_arr[idx] / weight_sum
                add = min(demands[idx] - allocated[idx], share)
                if add > 1e-12:
                    allocated[idx] += add
                    progressed = True
                if allocated[idx] >= demands[idx] - 1e-12:
                    active[idx] = False
            new_remaining = supply - float(np.sum(allocated))
            if not progressed or abs(new_remaining - remaining_supply) < 1e-12:
                break
            remaining_supply = max(0.0, new_remaining)

        out: list[np.ndarray] = []
        for v, d, a in zip(source_vectors, demands, allocated):
            out.append(np.zeros_like(v) if d <= 1e-12 else v * (a / d))
        return out

    def step(self, time_s: float, *, flush: bool = False) -> None:
        # Update routing using the old state.
        if hasattr(self.routing, "update_assignment"):
            self.routing.update_assignment(
                network=self.raw_network,
                vehicle_classes=self.vehicle_classes,
                time_s=time_s,
                commodity_state=self.x,
            )

        # Give adaptive controllers one chance to update their internal
        # phase decisions using the old state.  Fixed controllers implement
        # update as a no-op.
        if hasattr(self.signal_controller, "update"):
            self.signal_controller.update(
                self.net,
                time_s,
                state=self.x,
                simulator=self,
            )

        signal_green, active_phase_by_node = self._capture_signal_state(time_s)
        routing_probabilities = (
            self._capture_routing_state(time_s)
            if self.capture_routing_history
            else []
        )

        S_total, S = self.sending(self.x)
        R = self.receiving(self.x)

        inflow = np.zeros_like(self.x)
        outflow = np.zeros_like(self.x)

        interior_flow: dict[str, float] = {}
        interior_flow_records: list[dict] = []

        # --------------------------------------------------------------
        # Interior CTM flows: universal boundary F_{ell,i}, i=1..N_l-1.
        # These are the correct empirical flows for FD checks because they
        # are not signalized node flows and are not exit flows.
        # --------------------------------------------------------------
        x_total_old = self.x.sum(axis=1)
        for link_id, cells in self.net.link_cells.items():
            for boundary_index, (a, b) in enumerate(zip(cells[:-1], cells[1:]), start=1):
                Y = min(float(S_total[a]), float(R[b]))
                key = f"{int(link_id)}:{int(boundary_index)}"
                interior_flow[key] = Y
                upstream_density_vpm = float(x_total_old[a]) / float(self.net.cell_length_m[a])
                downstream_density_vpm = float(x_total_old[b]) / float(self.net.cell_length_m[b])
                interior_flow_records.append({
                    "link_id": int(link_id),
                    "boundary_index": int(boundary_index),
                    "upstream_cell_id": int(a),
                    "downstream_cell_id": int(b),
                    "upstream_density_veh_km": upstream_density_vpm * 1000.0,
                    "downstream_density_veh_km": downstream_density_vpm * 1000.0,
                    "flow_veh_step": float(Y),
                    "flow_vph": float(Y) * 3600.0 / self.dt,
                })
                if S_total[a] > 1e-12 and Y > 0:
                    vec = S[a] * (Y / S_total[a])
                    outflow[a] += vec
                    inflow[b] += vec

        # --------------------------------------------------------------
        # Candidate external entry into first cells.
        # --------------------------------------------------------------
        requested = self.demand.requested_demand_by_commodity(
            time_s, self.dt, flush=flush
        )
        entry_candidate = np.zeros((self.net.n_links + 1, self.n_commodities), dtype=float)

        for q, amount in requested.items():
            self.cumulative_requested += float(amount)
            self.virtual_queue[q] += float(amount)

        for q, commodity in enumerate(self.commodities):
            origin_link = commodity.origin_link
            if self.virtual_queue[q] > 0:
                entry_candidate[origin_link, q] = self.virtual_queue[q]

        # --------------------------------------------------------------
        # Candidate node movement flows and exits from last cells.
        # --------------------------------------------------------------
        movement_candidate_total = np.zeros(self.net.n_movements, dtype=float)
        movement_candidate_by_comm = np.zeros((self.net.n_movements, self.n_commodities), dtype=float)
        # Routing-weighted last-cell vehicle mass p(l,k,h,t)n(l,I_l,h,t),
        # aggregated by vehicle class.  This is a vehicle count used by the
        # movement-composition saturation controller to estimate HDV/CAV shares;
        # it is not itself a saturation flow rate.
        movement_weighted_class_mass = np.zeros((self.net.n_movements, self.n_classes), dtype=float)
        self._movement_weighted_class_mass = movement_weighted_class_mass
        exit_by_link_comm: dict[int, np.ndarray] = {
            link_id: np.zeros(self.n_commodities, dtype=float)
            for link_id in self.raw_network.links
        }

        # Exit candidates: if d(link) is destination of commodity.
        for link_id, link in self.raw_network.links.items():
            last_cell = self.net.link_last_cell[link_id]
            node = link.to_node
            for q, commodity in enumerate(self.commodities):
                if node == commodity.destination_node:
                    exit_by_link_comm[link_id][q] = S[last_cell, q]

        # Movement candidates for non-destination commodities.
        for m_idx in range(self.net.n_movements):
            from_link = int(self.net.movement_from_link[m_idx])
            to_link = int(self.net.movement_to_link[m_idx])
            from_cell = int(self.net.movement_from_cell[m_idx])
            from_node = self.raw_network.links[from_link].to_node

            demand_vec = np.zeros(self.n_commodities, dtype=float)
            for q, commodity in enumerate(self.commodities):
                if from_node == commodity.destination_node:
                    continue
                p = 0.0
                try:
                    p = float(
                        self.routing.probability(
                            from_link=from_link,
                            to_link=to_link,
                            class_name=commodity.class_name,
                            commodity_id=q,
                            destination_node=commodity.destination_node,
                            time_s=time_s,
                        )
                    )
                except TypeError:
                    p = float(
                        self.routing.probability(
                            from_link=from_link,
                            to_link=to_link,
                            class_name=commodity.class_name,
                            time_s=time_s,
                        )
                    )
                if p > 0:
                    demand_vec[q] = p * S[from_cell, q]
                    movement_weighted_class_mass[m_idx, int(commodity.class_index)] += p * self.x[from_cell, q]

            D = float(np.sum(demand_vec))
            if D <= 1e-12:
                continue

            try:
                green_duration_s = float(
                    self.signal_controller.movement_green_duration(
                        self.net,
                        m_idx,
                        time_s,
                        self.dt,
                    )
                )
            except AttributeError:
                green_duration_s = self.dt if self.signal_controller.movement_is_green(self.net, m_idx, time_s) else 0.0
            green_duration_s = min(max(0.0, green_duration_s), self.dt)
            if green_duration_s <= 1e-12:
                continue

            sat_vps = self.saturation_controller.saturation_vps(
                self.net,
                m_idx,
                time_s,
                state=self.x,
                simulator=self,
            )
            cap = max(0.0, float(sat_vps) * green_duration_s)
            cand = min(D, cap)
            if cand <= 1e-12:
                continue

            movement_candidate_total[m_idx] = cand
            movement_candidate_by_comm[m_idx] = demand_vec * (cand / D)

        # --------------------------------------------------------------
        # Supply allocation into each link first cell.
        # This simultaneously limits external entry and upstream movements.
        # --------------------------------------------------------------
        movement_realized_by_comm = np.zeros_like(movement_candidate_by_comm)
        entry_realized = np.zeros_like(entry_candidate)

        for to_link_id, link in self.raw_network.links.items():
            first_cell = self.net.link_first_cell[to_link_id]

            source_keys: list[tuple[str, int]] = []
            source_vectors: list[np.ndarray] = []
            priorities: list[float] = []

            if float(np.sum(entry_candidate[to_link_id])) > 1e-12:
                source_keys.append(("entry", to_link_id))
                source_vectors.append(entry_candidate[to_link_id].copy())
                priorities.append(1.0)

            for m_idx in range(self.net.n_movements):
                if int(self.net.movement_to_link[m_idx]) == to_link_id and movement_candidate_total[m_idx] > 1e-12:
                    source_keys.append(("movement", m_idx))
                    source_vectors.append(movement_candidate_by_comm[m_idx].copy())
                    priorities.append(float(self.net.movement_merge_priority[m_idx]))

            realized_vectors = self._allocate_link_supply(
                supply=float(R[first_cell]),
                source_vectors=source_vectors,
                priorities=priorities,
            )

            for (kind, idx), vec in zip(source_keys, realized_vectors):
                if kind == "entry":
                    entry_realized[idx] = vec
                else:
                    movement_realized_by_comm[idx] = vec

        # --------------------------------------------------------------
        # Apply external entry, movement flows, and exits to boundary arrays.
        # --------------------------------------------------------------
        entry_flow_by_link: dict[int, float] = {}
        entry_flow_by_link_comm: dict[int, list[float]] = {}
        for link_id in self.raw_network.links:
            first_cell = self.net.link_first_cell[link_id]
            vec = entry_realized[link_id]
            if np.sum(vec) > 1e-12:
                inflow[first_cell] += vec
            for q in range(self.n_commodities):
                self.virtual_queue[q] -= vec[q]
            total = float(np.sum(vec))
            if total > 1e-12:
                entry_flow_by_link[link_id] = total
                entry_flow_by_link_comm[link_id] = vec.tolist()

        movement_flow: dict[int, float] = {}
        movement_flow_by_comm: dict[int, list[float]] = {}
        for m_idx in range(self.net.n_movements):
            vec = movement_realized_by_comm[m_idx]
            total = float(np.sum(vec))
            if total <= 1e-12:
                continue
            from_cell = int(self.net.movement_from_cell[m_idx])
            to_cell = int(self.net.movement_to_cell[m_idx])
            outflow[from_cell] += vec
            inflow[to_cell] += vec
            movement_id = self.net.movement_index_to_id[m_idx]
            movement_flow[movement_id] = total
            movement_flow_by_comm[movement_id] = vec.tolist()

        exit_flow_by_link: dict[int, float] = {}
        exit_flow_by_link_comm: dict[int, list[float]] = {}
        for link_id, vec in exit_by_link_comm.items():
            total = float(np.sum(vec))
            if total <= 1e-12:
                continue
            last_cell = self.net.link_last_cell[link_id]
            outflow[last_cell] += vec
            self.cumulative_exited += total
            exit_flow_by_link[link_id] = total
            exit_flow_by_link_comm[link_id] = vec.tolist()

        # --------------------------------------------------------------
        # Universal conservation update.
        # --------------------------------------------------------------
        x_new = self.x + inflow - outflow
        # Numerical hygiene. Do not hide real errors.
        if np.min(x_new) < -1e-7:
            raise RuntimeError(f"Negative vehicle count detected: {np.min(x_new)}")
        x_new[np.abs(x_new) < 1e-12] = 0.0
        x_new = np.maximum(x_new, 0.0)

        # Storage check.
        total = x_new.sum(axis=1)
        if np.max(total - self.net.cell_storage_veh) > 1e-6:
            bad = np.where(total - self.net.cell_storage_veh > 1e-6)[0]
            raise RuntimeError(f"Cell storage violation at cells {bad.tolist()}")

        self.x = x_new
        self.virtual_queue = np.maximum(self.virtual_queue, 0.0)

        self.save_state(
            time_s + self.dt,
            movement_flow=movement_flow,
            movement_flow_by_commodity=movement_flow_by_comm,
            entry_flow=entry_flow_by_link,
            entry_flow_by_commodity=entry_flow_by_link_comm,
            exit_flow=exit_flow_by_link,
            exit_flow_by_commodity=exit_flow_by_link_comm,
            interior_flow=interior_flow,
            interior_flow_records=interior_flow_records,
            is_flush=flush,
            signal_green=signal_green,
            active_phase_by_node=active_phase_by_node,
            routing_probabilities=routing_probabilities,
        )

    def run(self, n_steps: int, *, flush_time_s: float = 0.0) -> SimulationHistory:
        if n_steps < 0:
            raise ValueError("n_steps cannot be negative")
        if flush_time_s < 0:
            raise ValueError("flush_time_s cannot be negative")

        self.initial_system_vehicles = float(np.sum(self.x) + np.sum(self.virtual_queue))
        self.cumulative_requested = 0.0
        self.cumulative_exited = 0.0
        self.history = SimulationHistory()

        if hasattr(self.routing, "update_assignment"):
            self.routing.update_assignment(
                network=self.raw_network,
                vehicle_classes=self.vehicle_classes,
                time_s=0.0,
                commodity_state=self.x,
            )
        if hasattr(self.signal_controller, "update"):
            self.signal_controller.update(
                self.net,
                0.0,
                state=self.x,
                simulator=self,
            )
        signal_green, active_phase_by_node = self._capture_signal_state(0.0)
        routing_probabilities = (
            self._capture_routing_state(0.0)
            if self.capture_routing_history
            else []
        )
        self.save_state(
            0.0,
            is_flush=False,
            signal_green=signal_green,
            active_phase_by_node=active_phase_by_node,
            routing_probabilities=routing_probabilities,
        )

        t = 0.0
        for _ in range(int(n_steps)):
            self.step(t, flush=False)
            t += self.dt

        n_flush = int(math.ceil(flush_time_s / self.dt))
        for _ in range(n_flush):
            self.step(t, flush=True)
            t += self.dt

        return self.history
