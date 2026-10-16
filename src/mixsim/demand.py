from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import random

import numpy as np

from .network import Network
from .vehicle_type import VehicleClass
from .travel_time import TravelTimeEstimator
from .shortest_path import FloydWarshallRouter

DemandProfileFunction = Callable[[float], float]


@dataclass(frozen=True)
class Commodity:
    commodity_id: int
    origin_link: int
    destination_node: int
    class_name: str
    class_index: int
    destination_name: str


@dataclass
class ODDemandProfile:
    origin_link: int
    destination_node: int
    class_name: str
    profile: DemandProfileFunction
    assignment: str = "dynamic_shortest"  # fixed_shortest, dynamic_shortest, stochastic
    stochastic_theta: float = 0.05
    rate_unit: str = "veh/h"  # veh/h or veh/s


class Demand:
    """
    OD demand generator and OD-specific routing controller.

    Commodities are origin-link, destination-node, and vehicle-class specific.
    Routing probabilities are evaluated at nodes and do not classify cells by
    movement. Blocked movement demand is never redistributed within the same
    time step; spillback is handled in the simulator by realized flows.
    """

    def __init__(
        self,
        seed: int | None = 1,
        travel_time_estimator: TravelTimeEstimator | None = None,
    ) -> None:
        self.od_profiles: list[ODDemandProfile] = []
        self.travel_time_estimator = travel_time_estimator or TravelTimeEstimator()
        self.router = FloydWarshallRouter()
        self.random = random.Random(seed)
        self.commodities: list[Commodity] = []
        self.commodity_by_key: dict[tuple[int, int, str], int] = {}
        self._profile_by_commodity: dict[int, list[ODDemandProfile]] = {}
        self.current_routes_by_commodity: dict[tuple[int, int], dict[int, float]] = {}
        self._fixed_routes_by_commodity: dict[tuple[int, int], dict[int, float]] = {}
        self._prepared = False

    @staticmethod
    def constant_profile(vehicles_per_hour: float) -> DemandProfileFunction:
        """Return a constant demand profile in vehicles/hour."""
        def profile(time_s: float) -> float:
            return float(vehicles_per_hour)
        return profile

    @staticmethod
    def constant_profile_vps(vehicles_per_second: float) -> DemandProfileFunction:
        """Return a constant demand profile in vehicles/second."""
        def profile(time_s: float) -> float:
            return float(vehicles_per_second)
        return profile

    @staticmethod
    def trapezoidal_profile(
        base_vph: float,
        peak_vph: float,
        rise_start_s: float,
        peak_start_s: float,
        peak_end_s: float,
        fall_end_s: float,
    ) -> DemandProfileFunction:
        if peak_start_s <= rise_start_s:
            raise ValueError("peak_start_s must be greater than rise_start_s")
        if peak_end_s < peak_start_s:
            raise ValueError("peak_end_s must be >= peak_start_s")
        if fall_end_s <= peak_end_s:
            raise ValueError("fall_end_s must be greater than peak_end_s")

        def profile(time_s: float) -> float:
            if time_s < rise_start_s:
                return float(base_vph)
            if time_s < peak_start_s:
                r = (time_s - rise_start_s) / (peak_start_s - rise_start_s)
                return float(base_vph + r * (peak_vph - base_vph))
            if time_s <= peak_end_s:
                return float(peak_vph)
            if time_s < fall_end_s:
                r = (time_s - peak_end_s) / (fall_end_s - peak_end_s)
                return float(peak_vph + r * (base_vph - peak_vph))
            return float(base_vph)
        return profile


    @staticmethod
    def smooth_trapezoidal_profile(
        base_vph: float,
        peak_vph: float,
        rise_start_s: float,
        peak_start_s: float,
        peak_end_s: float,
        fall_end_s: float,
    ) -> DemandProfileFunction:
        """Return a smooth trapezoidal demand profile in vehicles/hour.

        The rising and falling segments use a cubic smoothstep curve rather
        than a sharp linear ramp.  This is useful for presentation figures and
        for simulation studies where the demand should increase and decrease
        gradually.
        """
        if peak_start_s <= rise_start_s:
            raise ValueError("peak_start_s must be greater than rise_start_s")
        if peak_end_s < peak_start_s:
            raise ValueError("peak_end_s must be >= peak_start_s")
        if fall_end_s <= peak_end_s:
            raise ValueError("fall_end_s must be greater than peak_end_s")

        def smoothstep(r: float) -> float:
            r = min(1.0, max(0.0, float(r)))
            return r * r * (3.0 - 2.0 * r)

        def profile(time_s: float) -> float:
            if time_s < rise_start_s:
                return float(base_vph)
            if time_s < peak_start_s:
                r = smoothstep((time_s - rise_start_s) / (peak_start_s - rise_start_s))
                return float(base_vph + r * (peak_vph - base_vph))
            if time_s <= peak_end_s:
                return float(peak_vph)
            if time_s < fall_end_s:
                r = smoothstep((time_s - peak_end_s) / (fall_end_s - peak_end_s))
                return float(peak_vph + r * (base_vph - peak_vph))
            return float(base_vph)
        return profile

    def add_od_profile(
        self,
        *,
        origin_link: int,
        destination_node: int | str,
        class_name: str,
        profile: DemandProfileFunction,
        assignment: str = "dynamic_shortest",
        stochastic_theta: float = 0.05,
        rate_unit: str = "veh/h",
        network: Network | None = None,
    ) -> None:
        if isinstance(destination_node, str):
            if network is None:
                raise ValueError("Pass network when destination_node is a name")
            destination_node = network.node_name_to_id[destination_node]
        if assignment not in {"fixed_shortest", "dynamic_shortest", "stochastic"}:
            raise ValueError("assignment must be 'fixed_shortest', 'dynamic_shortest', or 'stochastic'")
        if rate_unit not in {"veh/h", "veh/s"}:
            raise ValueError("rate_unit must be 'veh/h' or 'veh/s'")
        self.od_profiles.append(
            ODDemandProfile(
                origin_link=int(origin_link),
                destination_node=int(destination_node),
                class_name=str(class_name),
                profile=profile,
                assignment=assignment,
                stochastic_theta=float(stochastic_theta),
                rate_unit=rate_unit,
            )
        )
        self._prepared = False

    def prepare_commodities(
        self,
        network: Network,
        vehicle_classes: list[VehicleClass],
    ) -> list[Commodity]:
        class_index = {vc.name: i for i, vc in enumerate(vehicle_classes)}
        commodities: list[Commodity] = []
        self.commodity_by_key.clear()
        self._profile_by_commodity.clear()
        self.current_routes_by_commodity.clear()
        self._fixed_routes_by_commodity.clear()

        for prof in self.od_profiles:
            if prof.origin_link not in network.links:
                raise ValueError(f"Unknown origin_link {prof.origin_link}")
            if prof.destination_node not in network.nodes:
                raise ValueError(f"Unknown destination_node {prof.destination_node}")
            if prof.class_name not in class_index:
                raise ValueError(f"Unknown class_name {prof.class_name}")
            key = (prof.origin_link, prof.destination_node, prof.class_name)
            if key not in self.commodity_by_key:
                cid = len(commodities)
                self.commodity_by_key[key] = cid
                dest_name = network.node_id_to_name.get(prof.destination_node, f"N{prof.destination_node}")
                commodities.append(
                    Commodity(
                        commodity_id=cid,
                        origin_link=prof.origin_link,
                        destination_node=prof.destination_node,
                        class_name=prof.class_name,
                        class_index=class_index[prof.class_name],
                        destination_name=dest_name,
                    )
                )
            cid = self.commodity_by_key[key]
            self._profile_by_commodity.setdefault(cid, []).append(prof)

        self.commodities = commodities
        self._prepared = True
        return list(self.commodities)

    def requested_demand_by_commodity(
        self,
        time_s: float,
        dt: float,
        *,
        flush: bool = False,
    ) -> dict[int, float]:
        out = {c.commodity_id: 0.0 for c in self.commodities}
        if flush:
            return out
        for cid, profiles in self._profile_by_commodity.items():
            total = 0.0
            for p in profiles:
                value = float(p.profile(time_s))
                if p.rate_unit == "veh/h":
                    total += value * dt / 3600.0
                elif p.rate_unit == "veh/s":
                    total += value * dt
                else:  # defensive; add_od_profile validates this
                    raise ValueError(f"Unsupported rate_unit: {p.rate_unit}")
            out[cid] = max(0.0, total)
        return out

    def _link_costs(
        self,
        network: Network,
        commodity_state=None,
        vehicle_classes: list[VehicleClass] | None = None,
    ) -> dict[int, float]:
        commodity_class_index = None
        if self.commodities:
            commodity_class_index = np.array(
                [c.class_index for c in self.commodities], dtype=int
            )
        return self.travel_time_estimator.link_costs(
            network,
            commodity_state,
            vehicle_classes=vehicle_classes,
            commodity_class_index=commodity_class_index,
        )

    def _routes_for_commodity(
        self,
        network: Network,
        router: FloydWarshallRouter,
        link_costs: dict[int, float],
        commodity: Commodity,
        *,
        mode: str,
        theta: float,
    ) -> dict[tuple[int, int], dict[int, float]]:
        """Build route fractions for one commodity over all upstream links."""
        routes_by_key: dict[tuple[int, int], dict[int, float]] = {}
        dest = commodity.destination_node

        for from_link_id, from_link in network.links.items():
            node = from_link.to_node
            key = (from_link_id, commodity.commodity_id)
            if node == dest:
                routes_by_key[key] = {}
                continue

            candidates: list[int] = []
            costs: list[float] = []
            for to_link_id in [lid for lid, lnk in network.links.items() if lnk.from_node == node]:
                to_link = network.links[to_link_id]
                if not router.can_reach(to_link.to_node, dest):
                    continue
                rem = router.distance(to_link.to_node, dest)
                cost = float(link_costs.get(to_link_id, math.inf)) + rem
                if math.isfinite(cost):
                    candidates.append(to_link_id)
                    costs.append(cost)

            if not candidates:
                routes_by_key[key] = {}
                continue

            if mode == "stochastic":
                vals = np.array(costs, dtype=float)
                vals = vals - np.min(vals)
                weights = np.exp(-float(theta) * vals)
                if not np.isfinite(weights).all() or weights.sum() <= 0:
                    probs = np.ones(len(candidates)) / len(candidates)
                else:
                    probs = weights / weights.sum()
                routes = {lid: float(p) for lid, p in zip(candidates, probs)}
            else:
                min_cost = min(costs)
                best = [lid for lid, c in zip(candidates, costs) if abs(c - min_cost) <= 1e-9]
                p = 1.0 / len(best)
                routes = {lid: (p if lid in best else 0.0) for lid in candidates}

            routes_by_key[key] = routes

        return routes_by_key

    def update_assignment(
        self,
        network: Network,
        vehicle_classes: list[VehicleClass],
        time_s: float,
        commodity_state=None,
    ) -> None:
        if not self._prepared:
            self.prepare_commodities(network, vehicle_classes)

        # Dynamic and stochastic modes use current state-dependent link costs.
        dynamic_link_costs = self._link_costs(
            network,
            commodity_state,
            vehicle_classes=vehicle_classes,
        )
        dynamic_router = FloydWarshallRouter()
        dynamic_router.build(network, dynamic_link_costs)

        # Fixed-shortest mode uses a one-time static/free-flow routing table.
        fixed_link_costs = self._link_costs(
            network,
            None,
            vehicle_classes=vehicle_classes,
        )
        fixed_router = FloydWarshallRouter()
        fixed_router.build(network, fixed_link_costs)

        self.current_routes_by_commodity.clear()

        # Route fractions depend on destination, assignment mode, theta and the
        # common network link costs, but not on vehicle class or origin.  Many
        # MIXSim commodities therefore share the same routing template.  Build
        # each unique template once per assignment update and copy it to the
        # relevant commodity IDs.  This preserves the mathematical assignment
        # exactly while avoiding repeated all-link route calculations.
        dynamic_templates: dict[tuple[int, str, float], dict[int, dict[int, float]]] = {}
        fixed_templates: dict[tuple[int, str, float], dict[int, dict[int, float]]] = {}

        for commodity in self.commodities:
            assignment = self._profile_by_commodity.get(commodity.commodity_id, [None])[0]
            mode = assignment.assignment if assignment is not None else "dynamic_shortest"
            theta = assignment.stochastic_theta if assignment is not None else 0.05
            template_key = (int(commodity.destination_node), str(mode), float(theta))

            if mode == "fixed_shortest":
                needed = [
                    (from_link_id, commodity.commodity_id)
                    for from_link_id in network.links
                ]
                if not all(key in self._fixed_routes_by_commodity for key in needed):
                    if template_key not in fixed_templates:
                        raw = self._routes_for_commodity(
                            network,
                            fixed_router,
                            fixed_link_costs,
                            commodity,
                            mode="fixed_shortest",
                            theta=theta,
                        )
                        fixed_templates[template_key] = {
                            int(from_link): dict(probs)
                            for (from_link, _), probs in raw.items()
                        }
                    template = fixed_templates[template_key]
                    for from_link_id in network.links:
                        self._fixed_routes_by_commodity[(from_link_id, commodity.commodity_id)] = dict(
                            template.get(int(from_link_id), {})
                        )
                for key in needed:
                    self.current_routes_by_commodity[key] = dict(
                        self._fixed_routes_by_commodity.get(key, {})
                    )
            else:
                if template_key not in dynamic_templates:
                    raw = self._routes_for_commodity(
                        network,
                        dynamic_router,
                        dynamic_link_costs,
                        commodity,
                        mode=mode,
                        theta=theta,
                    )
                    dynamic_templates[template_key] = {
                        int(from_link): dict(probs)
                        for (from_link, _), probs in raw.items()
                    }
                template = dynamic_templates[template_key]
                for from_link_id in network.links:
                    self.current_routes_by_commodity[(from_link_id, commodity.commodity_id)] = dict(
                        template.get(int(from_link_id), {})
                    )

    def probability(
        self,
        from_link: int,
        to_link: int,
        class_name: str | None = None,
        time_s: float = 0.0,
        *,
        commodity_id: int | None = None,
        destination_node: int | None = None,
    ) -> float:
        if commodity_id is not None:
            return float(
                self.current_routes_by_commodity.get((from_link, commodity_id), {}).get(to_link, 0.0)
            )
        # Fallback aggregate probability if caller does not supply commodity_id.
        vals = []
        for (fl, cid), routes in self.current_routes_by_commodity.items():
            if fl != from_link:
                continue
            c = self.commodities[cid]
            if class_name is not None and c.class_name != class_name:
                continue
            if destination_node is not None and c.destination_node != destination_node:
                continue
            vals.append(float(routes.get(to_link, 0.0)))
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def commodity_metadata(self) -> list[dict]:
        return [
            {
                "commodity_id": c.commodity_id,
                "origin_link": c.origin_link,
                "class_name": c.class_name,
                "class_index": c.class_index,
                "destination_node": c.destination_node,
                "destination_name": c.destination_name,
            }
            for c in self.commodities
        ]
