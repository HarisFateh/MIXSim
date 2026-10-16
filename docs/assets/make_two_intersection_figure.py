from pathlib import Path
import sys
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from mixsim import build_two_intersection_tutorial
from mixsim.node_type import NodeType

net = build_two_intersection_tutorial()
fig, ax = plt.subplots(figsize=(8.0, 4.6))
for link in net.links.values():
    u = net.nodes[link.from_node]
    v = net.nodes[link.to_node]
    ax.annotate('', xy=(v.x, v.y), xytext=(u.x, u.y), arrowprops=dict(arrowstyle='->', lw=1.8))
    ax.text((u.x+v.x)/2, (u.y+v.y)/2+22, f'L{link.link_id}', ha='center', va='center', fontsize=9)
for node_id, node in net.nodes.items():
    if node.node_type == NodeType.SIGNALIZED:
        marker, size = 's', 260
    elif node.node_type == NodeType.ORIGIN:
        marker, size = 'o', 170
    else:
        marker, size = 'D', 170
    ax.scatter([node.x],[node.y],s=size,marker=marker,zorder=5,edgecolors='black',facecolors='white')
    ax.text(node.x, node.y-50, net.node_id_to_name[node_id], ha='center', va='top', fontsize=10)
ax.set_aspect('equal')
ax.axis('off')
fig.tight_layout()
out=Path(__file__).resolve().parent
fig.savefig(out/'two_intersection_tutorial_network.svg', bbox_inches='tight', transparent=True)
fig.savefig(out/'two_intersection_tutorial_network.pdf', bbox_inches='tight', transparent=True)
