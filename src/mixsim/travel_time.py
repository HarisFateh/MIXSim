from __future__ import annotations

import math
import numpy as np

from .network import Network


class TravelTimeEstimator:
    """Instantaneous state-based link travel-time estimator for routing.

    Dynamic and stochastic route choice use the current MIXSim traffic state
    to construct a physically interpretable travel-time cost.  For each cell,
    the local triangular fundamental diagram gives the equilibrium flow

        q = min(v_f * rho, w * (rho_jam - rho)),

    with composition-dependent backward wave speed ``w = L_v / tau_bar``.
    For a nonempty cell the corresponding instantaneous speed is ``u = q/rho``;
    an empty cell uses its free-flow speed.  Link travel time is the sum of
    ``dx / max(u, min_speed_mps)`` over its cells.

    The estimate is a snapshot cost: it is the travel time that would result
    if the current cell states persisted while the link were traversed.  It is
    therefore suitable for online routing, while not claiming to be a
    predictive or experienced travel time.
    """

    def __init__(
        self,
        min_speed_mps: float = 0.5,
        max_cost: float = 1.0e9,
    ) -> None:
        if min_speed_mps <= 0:
            raise ValueError("min_speed_mps must be positive")
        if max_cost <= 0:
            raise ValueError("max_cost must be positive")
        self.min_speed_mps = float(min_speed_mps)
        self.max_cost = float(max_cost)

    def link_free_flow_time(self, link) -> float:
        if not link.cells:
            return self.max_cost
        total = 0.0
        for cell in link.cells:
            vf = float(cell.free_flow_speed_mps)
            if vf <= 0:
                return self.max_cost
            total += float(cell.length_m) / vf
        return min(self.max_cost, total)

    def link_density_ratio(self, link, commodity_state=None) -> float:
        """Return occupied-storage fraction for diagnostics only.

        This quantity is retained for diagnostics/backward compatibility.  It
        is no longer used as the routing cost.
        """
        if commodity_state is None:
            return 0.0
        total_vehicles = 0.0
        total_storage = 0.0
        for cell in link.cells:
            total_vehicles += float(np.sum(commodity_state[cell.cell_id]))
            total_storage += float(cell.jam_density_vpm) * float(cell.length_m)
        if total_storage <= 0:
            return 0.0
        return max(0.0, total_vehicles / total_storage)

    @staticmethod
    def _class_counts(
        row: np.ndarray,
        n_classes: int,
        commodity_class_index: np.ndarray | None,
    ) -> np.ndarray | None:
        if commodity_class_index is None:
            return None
        idx = np.asarray(commodity_class_index, dtype=int)
        if idx.ndim != 1 or idx.size != row.size:
            raise ValueError(
                "commodity_class_index must contain one class index per commodity"
            )
        if idx.size and (idx.min() < 0 or idx.max() >= n_classes):
            raise ValueError("commodity_class_index contains an invalid class index")
        counts = np.zeros(n_classes, dtype=float)
        np.add.at(counts, idx, np.asarray(row, dtype=float))
        return counts

    def cell_speed_mps(
        self,
        cell,
        commodity_state=None,
        *,
        vehicle_classes=None,
        commodity_class_index: np.ndarray | None = None,
    ) -> float:
        """Instantaneous equilibrium speed associated with the current cell state."""
        vf = float(cell.free_flow_speed_mps)
        if vf <= 0:
            return self.min_speed_mps

        if commodity_state is None:
            return vf

        row = np.asarray(commodity_state[cell.cell_id], dtype=float)
        n = float(np.sum(row))
        if n <= 1.0e-12:
            return vf

        dx = float(cell.length_m)
        if dx <= 0:
            return self.min_speed_mps
        rho = n / dx
        rho_jam = float(cell.jam_density_vpm)
        if rho_jam <= 0:
            return self.min_speed_mps

        # For the MIXSim mixed-autonomy model, reconstruct the local
        # composition-dependent FD from the vehicle-class time gaps.
        if vehicle_classes is not None and commodity_class_index is not None:
            classes = list(vehicle_classes)
            if not classes:
                return vf
            class_counts = self._class_counts(row, len(classes), commodity_class_index)
            assert class_counts is not None
            total = float(np.sum(class_counts))
            if total <= 1.0e-12:
                return vf
            shares = class_counts / total
            tau = np.array([float(vc.time_gap_s) for vc in classes], dtype=float)
            lengths = np.array([float(vc.vehicle_length_m) for vc in classes], dtype=float)
            if np.any(tau <= 0) or np.any(lengths <= 0):
                raise ValueError("vehicle-class time gaps and standstill spacings must be positive")
            tau_bar = float(np.dot(shares, tau))
            # The default simulator requires a common standstill spacing; the
            # weighted form keeps this helper well-defined for external users.
            ell_v = float(np.dot(shares, lengths))
            w = ell_v / tau_bar
        else:
            # Generic fallback for users of the routing helper outside the
            # mixed-autonomy simulator: use network FD metadata.
            w = float(cell.wave_speed_mps)
            if w <= 0:
                return vf

        congested_supply = max(0.0, w * (rho_jam - rho))
        q = min(vf * rho, congested_supply)
        u = q / rho if rho > 0 else vf
        if not math.isfinite(u):
            return self.min_speed_mps
        return max(0.0, min(vf, float(u)))

    def link_travel_time(
        self,
        link,
        commodity_state=None,
        *,
        vehicle_classes=None,
        commodity_class_index: np.ndarray | None = None,
    ) -> float:
        """Return instantaneous link travel time in seconds."""
        if not link.cells:
            return self.max_cost
        total_time = 0.0
        for cell in link.cells:
            u = self.cell_speed_mps(
                cell,
                commodity_state,
                vehicle_classes=vehicle_classes,
                commodity_class_index=commodity_class_index,
            )
            effective_speed = max(self.min_speed_mps, float(u))
            total_time += float(cell.length_m) / effective_speed
            if total_time >= self.max_cost:
                return self.max_cost
        return min(self.max_cost, total_time)

    def link_costs(
        self,
        network: Network,
        commodity_state=None,
        *,
        vehicle_classes=None,
        commodity_class_index: np.ndarray | None = None,
    ) -> dict[int, float]:
        return {
            link_id: self.link_travel_time(
                link,
                commodity_state,
                vehicle_classes=vehicle_classes,
                commodity_class_index=commodity_class_index,
            )
            for link_id, link in network.links.items()
        }
