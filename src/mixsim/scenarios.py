"""Reusable networks used by the manuscript studies and public tutorial."""
from __future__ import annotations

from .network import Network
from .node_type import NodeType, TurnType

DEFAULT_DX_M = 100.0
DEFAULT_VF_MPS = 8.33
DEFAULT_W_MPS = 4.17
DEFAULT_JAM_VPM = 1.0 / 5.0
DEFAULT_CAP_VPS = 0.5


def _add_link(net: Network, from_node: str, to_node: str, *, length_m: float = 400.0, n_cells: int = 4):
    return net.add_link(
        from_node,
        to_node,
        length_m=length_m,
        n_cells=n_cells,
        free_flow_speed_mps=DEFAULT_VF_MPS,
        wave_speed_mps=DEFAULT_W_MPS,
        jam_density_vpm=DEFAULT_JAM_VPM,
        capacity_vps=DEFAULT_CAP_VPS,
    )


def build_single_intersection() -> Network:
    """Four-link isolated signalized intersection used in the manuscript."""
    net = Network()
    net.add_node(NodeType.ORIGIN, -400, 0, "O1")
    net.add_node(NodeType.ORIGIN, 0, 400, "O2")
    n1 = net.add_node(NodeType.SIGNALIZED, 0, 0, "N1")
    net.add_node(NodeType.DESTINATION, 400, 0, "D3")
    net.add_node(NodeType.DESTINATION, 0, -400, "D4")
    l1 = _add_link(net, "O1", "N1")
    l2 = _add_link(net, "O2", "N1")
    l3 = _add_link(net, "N1", "D3")
    l4 = _add_link(net, "N1", "D4")
    m13 = net.add_movement("L1_to_L3", n1, l1, l3, TurnType.STRAIGHT)
    m24 = net.add_movement("L2_to_L4", n1, l2, l4, TurnType.STRAIGHT)
    # 50 + 50 s gives the equal-split fixed-time plan used in the isolated study.
    net.add_phase(n1, "eastbound", [m13], green=50.0, yellow=0.0, all_red=0.0)
    net.add_phase(n1, "southbound", [m24], green=50.0, yellow=0.0, all_red=0.0)
    return net


def build_two_intersection_tutorial() -> Network:
    """Small two-intersection network for the public tutorial.

    The tutorial intentionally includes route choices at both intersections.
    Each incoming approach has two admissible outgoing movements, so the
    example exercises OD commodities, routing fractions, movement-specific
    composition, signal phases, downstream supply, and virtual-origin queues
    without requiring the full 4x4 manuscript network.
    """
    net = Network()

    net.add_node(NodeType.ORIGIN, -400.0, 0.0, "O1")
    net.add_node(NodeType.ORIGIN, 0.0, 400.0, "O2")
    net.add_node(NodeType.ORIGIN, 400.0, 400.0, "O3")
    n1 = net.add_node(NodeType.SIGNALIZED, 0.0, 0.0, "N1")
    n2 = net.add_node(NodeType.SIGNALIZED, 400.0, 0.0, "N2")
    net.add_node(NodeType.DESTINATION, 800.0, 0.0, "D1")
    net.add_node(NodeType.DESTINATION, 0.0, -400.0, "D2")
    net.add_node(NodeType.DESTINATION, 400.0, -400.0, "D3")

    l1 = _add_link(net, "O1", "N1")
    l2 = _add_link(net, "N1", "N2")
    l3 = _add_link(net, "N2", "D1")
    l4 = _add_link(net, "O2", "N1")
    l5 = _add_link(net, "N1", "D2")
    l6 = _add_link(net, "O3", "N2")
    l7 = _add_link(net, "N2", "D3")

    # N1: both outgoing links are available from each incoming approach.
    m12 = net.add_movement("L1_to_L2", n1, l1, l2, TurnType.STRAIGHT)
    m15 = net.add_movement("L1_to_L5", n1, l1, l5, TurnType.RIGHT)
    m42 = net.add_movement("L4_to_L2", n1, l4, l2, TurnType.RIGHT)
    m45 = net.add_movement("L4_to_L5", n1, l4, l5, TurnType.STRAIGHT)
    net.add_phase(n1, "N1_west_approach", [m12, m15], green=35.0, yellow=0.0, all_red=0.0)
    net.add_phase(n1, "N1_north_approach", [m42, m45], green=25.0, yellow=0.0, all_red=0.0)

    # N2: same idea, so OD destination determines the movement choice.
    m23 = net.add_movement("L2_to_L3", n2, l2, l3, TurnType.STRAIGHT)
    m27 = net.add_movement("L2_to_L7", n2, l2, l7, TurnType.RIGHT)
    m63 = net.add_movement("L6_to_L3", n2, l6, l3, TurnType.RIGHT)
    m67 = net.add_movement("L6_to_L7", n2, l6, l7, TurnType.STRAIGHT)
    net.add_phase(n2, "N2_west_approach", [m23, m27], green=35.0, yellow=0.0, all_red=0.0)
    net.add_phase(n2, "N2_north_approach", [m63, m67], green=25.0, yellow=0.0, all_red=0.0)

    return net


def build_grid_4x4(*, include_turn_movements: bool = True) -> Network:
    """Build the 40-link directed 4x4 signalized grid used in the manuscript.

    The layout follows the reference grid shown in the manuscript: 16
    signalized internal nodes (N1--N16), 8 boundary origin nodes, 8 boundary
    destination nodes, and 40 directed links (L1--L40).  Horizontal rows and
    vertical columns alternate direction, so each signalized intersection has
    two incoming links and two outgoing links.  With turn movements enabled,
    every incoming approach can choose either outgoing link at the node,
    producing 64 signalized movements in total.

    Boundary origins:
        O1, O4, O5, O8, O9, O12, O13, O16

    Boundary destinations:
        D2, D3, D6, D7, D10, D11, D14, D15

    This is intentionally not the simplified east/south-only grid.  It is the
    alternating-direction 4x4 test network used for the CAV-penetration, CAV time-gap,
    and signal-control experiments.
    """
    net = Network()

    # Internal signalized nodes, coordinates match the manuscript figure.
    coords = {
        "N1": (0, 0), "N2": (400, 0), "N3": (800, 0), "N4": (1200, 0),
        "N5": (0, -400), "N6": (400, -400), "N7": (800, -400), "N8": (1200, -400),
        "N9": (0, -800), "N10": (400, -800), "N11": (800, -800), "N12": (1200, -800),
        "N13": (0, -1200), "N14": (400, -1200), "N15": (800, -1200), "N16": (1200, -1200),
        "O1": (-350, 0), "D2": (1550, 0),
        "D3": (-350, -400), "O4": (1550, -400),
        "O5": (-350, -800), "D6": (1550, -800),
        "D7": (-350, -1200), "O8": (1550, -1200),
        "O9": (0, 400), "D10": (0, -1550),
        "D11": (400, 400), "O12": (400, -1550),
        "O13": (800, 400), "D14": (800, -1550),
        "D15": (1200, 400), "O16": (1200, -1550),
    }
    origin_names = {"O1", "O4", "O5", "O8", "O9", "O12", "O13", "O16"}
    dest_names = {"D2", "D3", "D6", "D7", "D10", "D11", "D14", "D15"}
    for name, (x, y) in coords.items():
        if name in origin_names:
            typ = NodeType.ORIGIN
        elif name in dest_names:
            typ = NodeType.DESTINATION
        else:
            typ = NodeType.SIGNALIZED
        net.add_node(typ, x, y, name)

    # Add links in the exact L1--L40 order of the reference grid.
    directed_links = [
        ("O1", "N1"), ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "D2"),
        ("O4", "N8"), ("N8", "N7"), ("N7", "N6"), ("N6", "N5"), ("N5", "D3"),
        ("O5", "N9"), ("N9", "N10"), ("N10", "N11"), ("N11", "N12"), ("N12", "D6"),
        ("O8", "N16"), ("N16", "N15"), ("N15", "N14"), ("N14", "N13"), ("N13", "D7"),
        ("O9", "N1"), ("N1", "N5"), ("N5", "N9"), ("N9", "N13"), ("N13", "D10"),
        ("O12", "N14"), ("N14", "N10"), ("N10", "N6"), ("N6", "N2"), ("N2", "D11"),
        ("O13", "N3"), ("N3", "N7"), ("N7", "N11"), ("N11", "N15"), ("N15", "D14"),
        ("O16", "N16"), ("N16", "N12"), ("N12", "N8"), ("N8", "N4"), ("N4", "D15"),
    ]
    for u, v in directed_links:
        _add_link(net, u, v, length_m=400.0, n_cells=4)

    def orientation(link_id: int) -> str:
        link = net.links[link_id]
        u = net.nodes[link.from_node]
        v = net.nodes[link.to_node]
        return "horizontal" if abs(u.y - v.y) <= abs(u.x - v.x) else "vertical"

    # Each signalized node has two incoming and two outgoing links.  Phase 1
    # serves the horizontal incoming approach; Phase 2 serves the vertical
    # incoming approach.  With turning enabled, the phase serves both available
    # outgoing links from that approach.
    for node_name in [f"N{i}" for i in range(1, 17)]:
        node_id = net.node_name_to_id[node_name]
        incoming = [lid for lid, lnk in net.links.items() if lnk.to_node == node_id]
        outgoing = [lid for lid, lnk in net.links.items() if lnk.from_node == node_id]
        if len(incoming) != 2 or len(outgoing) != 2:
            raise RuntimeError(f"{node_name} must have exactly two incoming and two outgoing links")

        phase_movements = {"horizontal": [], "vertical": []}
        for from_l in incoming:
            in_orient = orientation(from_l)
            for to_l in outgoing:
                out_orient = orientation(to_l)
                if (not include_turn_movements) and out_orient != in_orient:
                    continue
                turn = TurnType.STRAIGHT if out_orient == in_orient else TurnType.RIGHT
                mv = net.add_movement(
                    f"L{from_l}_to_L{to_l}",
                    node_id,
                    from_l,
                    to_l,
                    turn,
                    saturation_vps=0.65,
                )
                phase_movements[in_orient].append(mv.movement_id)

        if phase_movements["horizontal"]:
            net.add_phase(node_id, f"{node_name}_horizontal", phase_movements["horizontal"], green=30.0, yellow=0.0, all_red=0.0)
        if phase_movements["vertical"]:
            net.add_phase(node_id, f"{node_name}_vertical", phase_movements["vertical"], green=30.0, yellow=0.0, all_red=0.0)
    return net
