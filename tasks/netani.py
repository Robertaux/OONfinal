import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import random

"""
graph = {
    "A": {"connected_nodes": ["C", "D", "E", "B"], "position": [-314174,-180052]},
    "B": {"connected_nodes": ["A", "D", "C", "E", "F"], "position": [589314,-248509]},
    "C": {"connected_nodes": ["B", "D", "A"], "position": [589341,126640]},
    "D": {"connected_nodes": ["A", "C", "F", "B"], "position": [99578,486901]},
    "E": {"connected_nodes": ["A", "B", "F"], "position": [-255232,115182]},
    "F": {"connected_nodes": ["B", "E", "D"], "position": [404609,-146862]},
}
"""
graph = {
    "A": {"connected_nodes": ["B", "C", "D"], "position": [-350e3, 150e3]},
    "B": {"connected_nodes": ["A", "D", "F"], "position": [-100e3, 400e3]},
    "C": {"connected_nodes": ["A", "D", "E"], "position": [-200e3, -300e3]},
    "D": {"connected_nodes": ["A", "B", "C", "E", "F"], "position": [0, 0]},
    "E": {"connected_nodes": ["C", "D", "F"], "position": [150e3, -350e3]},
    "F": {"connected_nodes": ["B", "D", "E"], "position": [300e3, 250e3]},
}


G = nx.DiGraph()
for node, data in graph.items():
    G.add_node(node, pos=data["position"])
    for neighbor in data["connected_nodes"]:
        G.add_edge(node, neighbor)

edge_strength = {edge: 0 for edge in G.edges()}

colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]
cm = LinearSegmentedColormap.from_list("custom_colormap", colors, N=10)

pos = nx.kamada_kawai_layout(G)

fig, ax = plt.subplots(figsize=(10, 6))

sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=10))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label('Edge Strength')

nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000,
        node_color='lightblue', font_color='black', font_size=14,
        arrowsize=12, edge_color=[cm(0)], width=5, ax=ax)

def update(frame):
    global edge_strength

    if all(value == 10 for value in edge_strength.values()):
        ani.event_source.stop()
        return

    num_edges_to_update = random.randint(1, 3)
    updated_edges = random.sample(list(G.edges()), num_edges_to_update)

    for edge in updated_edges:
        if edge_strength[edge] < 10:
            edge_strength[edge] += 1

    ax.clear()
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000, node_color='lightblue')
    edge_colors = [cm(edge_strength[edge] / 10) for edge in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=5, arrows=True, arrowsize=12)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=14, font_weight='bold')
    sm.set_array([])

ani = FuncAnimation(fig, update, frames=150, interval=300, repeat=False)
ani.save("network_animation_fixed_old.gif", writer=PillowWriter(fps=5))

plt.show()
