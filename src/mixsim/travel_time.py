from __future__ import annotations

import numpy as np

from .network import Network


class TravelTimeEstimator:
    """Density-based link cost for dynamic routing."""

    def __init__(
        self,
        min_speed_mps: float = 0.5,
        density_penalty_power: float = 2.0,
        max_cost: float = 1.0e9,
    ) -> None:
        self.min_speed_mps = float(min_speed_mps)
        self.density_penalty_power = float(density_penalty_power)
        self.max_cost = float(max_cost)

    def link_free_flow_time(self, link) -> float:
        if not link.cells:
            return self.max_cost
        length = sum(c.length_m for c in link.cells)
        vf = float(np.mean([c.free_flow_speed_mps for c in link.cells]))
        if vf <= 0:
            return self.max_cost
        return length / vf

    def link_density_ratio(self, link, commodity_state=None) -> float:
        if commodity_state is None:
            return 0.0
        total_vehicles = 0.0
        total_storage = 0.0
        for c in link.cells:
            total_vehicles += float(np.sum(commodity_state[c.cell_id]))
            total_storage += c.jam_density_vpm * c.length_m
        if total_storage <= 0:
            return 0.0
        return max(0.0, total_vehicles / total_storage)

    def link_travel_time(self, link, commodity_state=None) -> float:
        t_ff = self.link_free_flow_time(link)
        ratio = min(0.999, self.link_density_ratio(link, commodity_state))
        speed_factor = max(1e-3, 1.0 - ratio)
        return min(self.max_cost, t_ff / (speed_factor ** self.density_penalty_power))

    def link_costs(self, network: Network, commodity_state=None) -> dict[int, float]:
        return {
            link_id: self.link_travel_time(link, commodity_state)
            for link_id, link in network.links.items()
        }
