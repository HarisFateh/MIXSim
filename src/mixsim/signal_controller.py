from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .network import CompiledNetwork


class SignalController:
    """Base class for signal controllers.

    Controllers may optionally implement ``update(net, time_s, state, simulator)``.
    The simulator calls this once per CTM step before candidate movement flows are
    computed.  ``movement_is_green`` is then queried for individual movements.
    """

    def update(
        self,
        net: CompiledNetwork,
        time_s: float,
        *,
        state: np.ndarray | None = None,
        simulator: Any | None = None,
    ) -> None:
        return None

    def movement_is_green(
        self,
        net: CompiledNetwork,
        movement_index: int,
        time_s: float,
    ) -> bool:
        raise NotImplementedError

    def movement_green_duration(
        self,
        net: CompiledNetwork,
        movement_index: int,
        time_s: float,
        dt: float,
    ) -> float:
        """Effective green duration for one CTM step, in seconds.

        The default implementation assumes the whole step is either green or red
        at ``time_s``. Fixed-time controllers override this to compute exact
        overlap with the green interval.
        """
        return float(dt) if self.movement_is_green(net, movement_index, time_s) else 0.0


def _overlap_duration_mod_cycle(
    start_s: float,
    end_s: float,
    cycle_s: float,
    intervals: list[tuple[float, float]],
) -> float:
    """Return overlap of [start_s, end_s) with periodic intervals."""
    if end_s <= start_s or cycle_s <= 0.0 or not intervals:
        return 0.0
    total = 0.0
    t = float(start_s)
    # Split at cycle boundaries so modulo intervals are easy to test.
    while t < end_s - 1e-12:
        pos = t % cycle_s
        rem_cycle = cycle_s - pos
        seg_end = min(end_s, t + rem_cycle)
        seg_a = pos
        seg_b = pos + (seg_end - t)
        for a, b in intervals:
            total += max(0.0, min(seg_b, b) - max(seg_a, a))
        t = seg_end
    return min(max(0.0, total), max(0.0, end_s - start_s))


class FixedSignalController(SignalController):
    """Fixed-time signal controller using the phases stored in ``Network``."""

    def movement_is_green(
        self,
        net: CompiledNetwork,
        movement_index: int,
        time_s: float,
    ) -> bool:
        node_id = int(net.movement_node_id[movement_index])
        phase_indices = net.node_phases.get(node_id, [])

        # Unsignalized/no-phase node: all movements are allowed.
        if not phase_indices:
            return True

        cycle_s = 0.0
        for p_idx in phase_indices:
            cycle_s += (
                float(net.phase_green_s[p_idx])
                + float(net.phase_yellow_s[p_idx])
                + float(net.phase_all_red_s[p_idx])
            )

        if cycle_s <= 0.0:
            return True

        t_cycle = time_s % cycle_s
        elapsed = 0.0

        for p_idx in phase_indices:
            green = float(net.phase_green_s[p_idx])
            yellow = float(net.phase_yellow_s[p_idx])
            all_red = float(net.phase_all_red_s[p_idx])
            total = green + yellow + all_red

            if elapsed <= t_cycle < elapsed + green:
                return bool(net.phase_movement_mask[p_idx, movement_index])

            elapsed += total

        return False

    def movement_green_duration(
        self,
        net: CompiledNetwork,
        movement_index: int,
        time_s: float,
        dt: float,
    ) -> float:
        node_id = int(net.movement_node_id[movement_index])
        phase_indices = net.node_phases.get(node_id, [])
        if not phase_indices:
            return float(dt)

        cycle_s = 0.0
        for p_idx in phase_indices:
            cycle_s += (
                float(net.phase_green_s[p_idx])
                + float(net.phase_yellow_s[p_idx])
                + float(net.phase_all_red_s[p_idx])
            )
        if cycle_s <= 0.0:
            return float(dt)

        intervals: list[tuple[float, float]] = []
        elapsed = 0.0
        for p_idx in phase_indices:
            green = float(net.phase_green_s[p_idx])
            yellow = float(net.phase_yellow_s[p_idx])
            all_red = float(net.phase_all_red_s[p_idx])
            if bool(net.phase_movement_mask[p_idx, movement_index]) and green > 0.0:
                intervals.append((elapsed, elapsed + green))
            elapsed += green + yellow + all_red

        return _overlap_duration_mod_cycle(
            float(time_s),
            float(time_s) + float(dt),
            cycle_s,
            intervals,
        )

    def get_phase_green(
        self,
        net: CompiledNetwork,
        node_id: int,
        phase_index: int,
        time_s: float,
    ) -> float:
        return float(net.phase_green_s[phase_index])




@dataclass
class PhasePlanSignalController(SignalController):
    """Fixed-cycle controller with an explicit phase plan.

    ``phase_green_by_id`` is keyed by logical phase id, not phase index.  This
    makes the plan stable after compilation.  The controller is used by the
    example scripts for fixed-time and demand-proportional fixed-time experiments.
    """

    cycle_s: float = 60.0
    phase_green_by_id: dict[int, float] = field(default_factory=dict)
    phase_offset_by_node: dict[int, float] = field(default_factory=dict)
    yellow_s: float = 0.0
    all_red_s: float = 0.0

    def _phase_green(self, net: CompiledNetwork, phase_index: int) -> float:
        phase_id = int(net.phase_index_to_id[int(phase_index)])
        if phase_id in self.phase_green_by_id:
            return max(0.0, float(self.phase_green_by_id[phase_id]))
        return max(0.0, float(net.phase_green_s[int(phase_index)]))

    def movement_is_green(self, net: CompiledNetwork, movement_index: int, time_s: float) -> bool:
        node_id = int(net.movement_node_id[movement_index])
        phase_indices = net.node_phases.get(node_id, [])
        if not phase_indices:
            return True
        cycle_s = max(1e-9, float(self.cycle_s))
        offset = float(self.phase_offset_by_node.get(node_id, 0.0))
        t_cycle = (float(time_s) - offset) % cycle_s
        elapsed = 0.0
        for p_idx in phase_indices:
            green = self._phase_green(net, int(p_idx))
            total = green + max(0.0, float(self.yellow_s)) + max(0.0, float(self.all_red_s))
            if elapsed <= t_cycle < elapsed + green:
                return bool(net.phase_movement_mask[int(p_idx), movement_index])
            elapsed += total
        return False

    def movement_green_duration(
        self,
        net: CompiledNetwork,
        movement_index: int,
        time_s: float,
        dt: float,
    ) -> float:
        node_id = int(net.movement_node_id[movement_index])
        phase_indices = net.node_phases.get(node_id, [])
        if not phase_indices:
            return float(dt)
        cycle_s = max(1e-9, float(self.cycle_s))
        offset = float(self.phase_offset_by_node.get(node_id, 0.0))
        intervals: list[tuple[float, float]] = []
        elapsed = 0.0
        for p_idx in phase_indices:
            green = self._phase_green(net, int(p_idx))
            if bool(net.phase_movement_mask[int(p_idx), movement_index]) and green > 0.0:
                intervals.append((elapsed, elapsed + green))
            elapsed += green + max(0.0, float(self.yellow_s)) + max(0.0, float(self.all_red_s))
        return _overlap_duration_mod_cycle(
            float(time_s) - offset,
            float(time_s) + float(dt) - offset,
            cycle_s,
            intervals,
        )

    def get_phase_green(self, net: CompiledNetwork, node_id: int, phase_index: int, time_s: float) -> float:
        return self._phase_green(net, int(phase_index))


@dataclass
class DemandProportionalFixedTimeController(PhasePlanSignalController):
    """Demand-proportional fixed-time controller with precomputed green splits.

    The experiment integrates static OD demand, propagates it over free-flow
    shortest paths, and allocates a fixed cycle among phases proportionally to
    those phase demand totals subject to a minimum green.  This is intentionally
    a transparent fixed-split heuristic; it is not the classical Webster
    optimum-cycle formula.
    """

    pass


@dataclass
class HeterogeneousMaxPressureController(SignalController):
    """Class-weighted max-pressure controller for HDV/CAV CTM states.

    The controller scores each available phase independently at every signalized
    node.  For movement ``a -> b`` the pressure is computed from the upstream
    last cell and downstream first cell using passenger-car-equivalent class
    weights.  By default these weights are derived from time gaps, so HDVs
    can contribute more pressure than CAVs because they consume larger time
    gaps.  This is intentionally a transparent control baseline, not a
    horizon-based MPC.
    """

    decision_interval_s: float = 10.0
    min_green_s: float = 20.0
    lost_time_s: float = 5.0
    use_class_weights: bool = True
    capacity_weighted: bool = False
    class_weights: list[float] | None = None

    active_phase_by_node: dict[int, int] = field(default_factory=dict)
    last_switch_time_by_node: dict[int, float] = field(default_factory=dict)
    lost_until_by_node: dict[int, float] = field(default_factory=dict)
    last_decision_bucket_by_node: dict[int, int] = field(default_factory=dict)

    def _weights(self, simulator: Any | None) -> np.ndarray | None:
        if not self.use_class_weights:
            return None
        if self.class_weights is not None:
            w = np.asarray(self.class_weights, dtype=float)
            if np.any(w <= 0):
                raise ValueError("class_weights must be positive")
            return w
        if simulator is None:
            return None
        tau = np.asarray(simulator.time_gap_s, dtype=float)
        return tau / float(np.min(tau))

    def _phase_score(
        self,
        net: CompiledNetwork,
        phase_index: int,
        *,
        state: np.ndarray,
        simulator: Any,
        weights: np.ndarray | None,
    ) -> float:
        if state is None or simulator is None:
            return 0.0
        x_class = getattr(simulator, "_controller_x_class_cache", None)
        if x_class is None:
            x_class = simulator.class_state_from_commodity(state)
        mask = net.phase_movement_mask[phase_index]
        score = 0.0
        for m_idx in np.where(mask)[0]:
            up_cell = int(net.movement_from_cell[m_idx])
            down_cell = int(net.movement_to_cell[m_idx])
            if weights is None:
                upstream = float(np.sum(x_class[up_cell, :]))
                downstream = float(np.sum(x_class[down_cell, :]))
            else:
                upstream = float(np.dot(weights, x_class[up_cell, :]))
                downstream = float(np.dot(weights, x_class[down_cell, :]))
            pressure = max(0.0, upstream - downstream)
            if self.capacity_weighted:
                pressure *= float(net.movement_saturation_vps[m_idx])
            score += pressure
        return float(score)

    def update(
        self,
        net: CompiledNetwork,
        time_s: float,
        *,
        state: np.ndarray | None = None,
        simulator: Any | None = None,
    ) -> None:
        if state is None or simulator is None:
            return None
        simulator._controller_x_class_cache = simulator.class_state_from_commodity(state)
        weights = self._weights(simulator)
        self._score_time_s = float(time_s)
        interval = max(float(self.decision_interval_s), 1e-9)
        bucket = int(np.floor(float(time_s) / interval))

        for node_id, phase_indices in net.node_phases.items():
            if not phase_indices:
                continue
            if node_id not in self.active_phase_by_node:
                scores = [
                    self._phase_score(net, p, state=state, simulator=simulator, weights=weights)
                    for p in phase_indices
                ]
                chosen = int(phase_indices[int(np.argmax(scores))])
                self.active_phase_by_node[node_id] = chosen
                self.last_switch_time_by_node[node_id] = float(time_s)
                self.lost_until_by_node[node_id] = -np.inf
                self.last_decision_bucket_by_node[node_id] = bucket
                continue

            if self.last_decision_bucket_by_node.get(node_id) == bucket:
                continue
            self.last_decision_bucket_by_node[node_id] = bucket

            active = self.active_phase_by_node[node_id]
            elapsed = float(time_s) - float(self.last_switch_time_by_node.get(node_id, -np.inf))
            if elapsed < float(self.min_green_s):
                continue

            scores = [
                self._phase_score(net, p, state=state, simulator=simulator, weights=weights)
                for p in phase_indices
            ]
            chosen = int(phase_indices[int(np.argmax(scores))])
            if chosen != active:
                self.active_phase_by_node[node_id] = chosen
                self.last_switch_time_by_node[node_id] = float(time_s)
                self.lost_until_by_node[node_id] = float(time_s) + max(0.0, float(self.lost_time_s))
        return None

    def movement_is_green(self, net: CompiledNetwork, movement_index: int, time_s: float) -> bool:
        node_id = int(net.movement_node_id[movement_index])
        phase_indices = net.node_phases.get(node_id, [])
        if not phase_indices:
            return True
        if float(time_s) < float(self.lost_until_by_node.get(node_id, -np.inf)):
            return False
        active = self.active_phase_by_node.get(node_id)
        if active is None:
            return False
        return bool(net.phase_movement_mask[int(active), movement_index])

