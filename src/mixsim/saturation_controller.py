from __future__ import annotations

from typing import Any

import numpy as np

from .network import CompiledNetwork


class SaturationController:
    """Base interface for movement saturation controllers.

    Implementations return a movement saturation flow rate in vehicles/second.
    The simulator multiplies this rate by the effective green duration inside
    the CTM step to obtain the movement service amount.
    """

    def saturation_vps(
        self,
        net: CompiledNetwork,
        movement_index: int,
        time_s: float,
        *,
        state=None,
        simulator: Any | None = None,
    ) -> float:
        raise NotImplementedError


class FixedSaturationController(SaturationController):
    """Use the nominal movement saturation value stored in the network."""

    def saturation_vps(
        self,
        net: CompiledNetwork,
        movement_index: int,
        time_s: float,
        *,
        state=None,
        simulator: Any | None = None,
    ) -> float:
        return float(net.movement_saturation_vps[movement_index])


class MovementCompositionSaturationController(SaturationController):
    """Movement-specific saturation from routing-weighted last-cell composition.

    For movement ``(ell,k)``, this controller first computes the class-specific
    amount of traffic in the last cell of link ``ell`` that is assigned to
    movement ``k``:

        N_lk^m(t) = sum_{h:m(h)=m} p(l,k,h,t) n(l,I_l,h,t).

    These values are vehicle counts, not saturation rates.  They are used only
    to infer the movement-specific class composition.  The saturation flow rate
    is then computed from the time-gap spacing relation:

        mu(l,k,t) = vf(l) / (vf(l) * tau_bar_lk(t) + L_v).

    The output unit is vehicles/second, so the simulator can use
    ``mu(l,k,t) * green_duration`` as a vehicle amount during one CTM step.
    """

    def __init__(self, *, multiplier: float = 1.0, max_saturation_vps: float | None = None) -> None:
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        self.multiplier = float(multiplier)
        self.max_saturation_vps = None if max_saturation_vps is None else float(max_saturation_vps)

    def _routing_probability(
        self,
        *,
        simulator: Any,
        from_link: int,
        to_link: int,
        commodity_id: int,
        commodity: Any,
        time_s: float,
    ) -> float:
        try:
            return float(
                simulator.routing.probability(
                    from_link=from_link,
                    to_link=to_link,
                    class_name=commodity.class_name,
                    commodity_id=commodity_id,
                    destination_node=commodity.destination_node,
                    time_s=time_s,
                )
            )
        except TypeError:
            return float(
                simulator.routing.probability(
                    from_link=from_link,
                    to_link=to_link,
                    class_name=commodity.class_name,
                    time_s=time_s,
                )
            )

    def saturation_vps(
        self,
        net: CompiledNetwork,
        movement_index: int,
        time_s: float,
        *,
        state=None,
        simulator: Any | None = None,
    ) -> float:
        if state is None or simulator is None:
            sat = float(net.movement_saturation_vps[movement_index])
            if self.max_saturation_vps is not None:
                sat = min(sat, self.max_saturation_vps)
            return float(max(0.0, self.multiplier * sat))

        from_link = int(net.movement_from_link[movement_index])
        to_link = int(net.movement_to_link[movement_index])
        from_cell = int(net.movement_from_cell[movement_index])
        from_node = simulator.raw_network.links[from_link].to_node

        cached = getattr(simulator, "_movement_weighted_class_mass", None)
        if cached is not None and np.shape(cached) == (net.n_movements, simulator.n_classes):
            class_mass = np.asarray(cached[movement_index], dtype=float).copy()
        else:
            class_mass = np.zeros(simulator.n_classes, dtype=float)
            for q, commodity in enumerate(simulator.commodities):
                # If the commodity's destination is this node, it exits rather than
                # requesting any outgoing movement.
                if int(from_node) == int(commodity.destination_node):
                    continue
                p = self._routing_probability(
                    simulator=simulator,
                    from_link=from_link,
                    to_link=to_link,
                    commodity_id=q,
                    commodity=commodity,
                    time_s=time_s,
                )
                if p <= 0.0:
                    continue
                # Correct use of p*n: vehicle mass assigned to this movement.
                class_mass[int(commodity.class_index)] += p * float(state[from_cell, q])

        total = float(np.sum(class_mass))
        if total <= 1e-12:
            # Empty movement-specific queue: use the local empty-cell reference
            # composition when available.  This keeps the controller well defined
            # before vehicles reach the stop line.
            if hasattr(simulator, "reference_class_share_by_cell"):
                alpha = np.asarray(simulator.reference_class_share_by_cell[from_cell], dtype=float)
            else:
                alpha = np.asarray(simulator.reference_class_share, dtype=float)
        else:
            alpha = class_mass / total

        tau_bar = float(np.dot(alpha, simulator.time_gap_s))
        vf = float(net.cell_vf_mps[from_cell])
        ell = float(simulator.common_vehicle_length_m)
        sat = self.multiplier * vf / (vf * tau_bar + ell)
        if self.max_saturation_vps is not None:
            sat = min(sat, self.max_saturation_vps)
        return float(max(0.0, sat))

