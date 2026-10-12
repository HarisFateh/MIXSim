from __future__ import annotations

import math

from .network import Network


class FloydWarshallRouter:
    """All-pairs shortest paths and first-link probabilities."""

    def __init__(self) -> None:
        self.network: Network | None = None
        self.node_ids: list[int] = []
        self.node_to_index: dict[int, int] = {}
        self.dist: list[list[float]] = []
        self.next_nodes: list[list[set[int]]] = []
        self.link_by_edge: dict[tuple[int, int], int] = {}
        self.outgoing_links: dict[int, list[int]] = {}
        self.link_costs: dict[int, float] = {}

    def build(self, network: Network, link_costs: dict[int, float]) -> None:
        self.network = network
        self.link_costs = dict(link_costs)
        self.node_ids = list(network.nodes.keys())
        self.node_to_index = {node_id: i for i, node_id in enumerate(self.node_ids)}
        n = len(self.node_ids)

        self.dist = [[math.inf] * n for _ in range(n)]
        self.next_nodes = [[set() for _ in range(n)] for _ in range(n)]
        self.link_by_edge = {}
        self.outgoing_links = {node_id: [] for node_id in self.node_ids}

        for i, node_id in enumerate(self.node_ids):
            self.dist[i][i] = 0.0
            self.next_nodes[i][i].add(node_id)

        for link_id, link in network.links.items():
            u = link.from_node
            v = link.to_node
            if u not in self.node_to_index or v not in self.node_to_index:
                continue
            self.outgoing_links.setdefault(u, []).append(link_id)
            self.link_by_edge[(u, v)] = link_id
            i = self.node_to_index[u]
            j = self.node_to_index[v]
            cost = float(link_costs.get(link_id, math.inf))
            if cost < self.dist[i][j] - 1e-9:
                self.dist[i][j] = cost
                self.next_nodes[i][j] = {v}
            elif abs(cost - self.dist[i][j]) <= 1e-9:
                self.next_nodes[i][j].add(v)

        for k in range(n):
            for i in range(n):
                if math.isinf(self.dist[i][k]):
                    continue
                for j in range(n):
                    alt = self.dist[i][k] + self.dist[k][j]
                    if alt < self.dist[i][j] - 1e-9:
                        self.dist[i][j] = alt
                        self.next_nodes[i][j] = set(self.next_nodes[i][k])
                    elif abs(alt - self.dist[i][j]) <= 1e-9 and alt < math.inf:
                        self.next_nodes[i][j].update(self.next_nodes[i][k])

    def distance(self, from_node: int, to_node: int) -> float:
        if from_node not in self.node_to_index or to_node not in self.node_to_index:
            return math.inf
        return self.dist[self.node_to_index[from_node]][self.node_to_index[to_node]]

    def outgoing_probabilities(self, from_node: int, dest_node: int) -> dict[int, float]:
        """Equal split over first links on shortest paths from node to destination."""
        if self.network is None:
            return {}
        if from_node not in self.node_to_index or dest_node not in self.node_to_index:
            return {}

        i = self.node_to_index[from_node]
        j = self.node_to_index[dest_node]
        first_nodes = sorted(self.next_nodes[i][j])
        if not first_nodes or math.isinf(self.dist[i][j]):
            return {}

        links: list[int] = []
        for v in first_nodes:
            lid = self.link_by_edge.get((from_node, v))
            if lid is not None:
                links.append(lid)

        if not links:
            return {}
        p = 1.0 / len(links)
        return {lid: p for lid in links}

    def can_reach(self, from_node: int, dest_node: int) -> bool:
        return math.isfinite(self.distance(from_node, dest_node))
