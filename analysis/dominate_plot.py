import networkx as nx
import matplotlib.pyplot as plt


G = nx.DiGraph()

# Normal situation
# ZIP be the best (0.05-0.25 batch interval)
# dominance_data = {
#     'RaF': ['ZIC'],
#     'GVWY': ['AA', 'RaF', 'ZIC', 'MIX'],
#     'AA': ['RaF', 'ZIC'],
#     'MIX': ['RaF', 'ZIC', 'AA'],
#     'ZIC': [],
#     'ZIP': ['ZIC', 'GVWY', 'AA', 'MIX', 'RaF']
# }

# GVWY the best (0.25-2 batch interval)

# dominance_data = {
#     'RaF': ['ZIC'],
#     'GVWY': ['ZIP', 'AA', 'RaF', 'ZIC', 'MIX'],
#     'AA': ['RaF', 'ZIC'],
#     'MIX': ['RaF', 'ZIP', 'ZIC', 'AA'],
#     'ZIC': [],
#     'ZIP': ['ZIC', 'AA', 'RaF']
# }

# Dynamic Market data low interval (0.05-0.25)

# dominance_data = {
#         'RaF': ['ZIC'],
#         'GVWY': ['AA', 'RaF', 'ZIC', 'MIX'],
#         'AA': ['RaF'],
#         'MIX': ['RaF', 'ZIC', 'AA'],
#         'ZIC': ['AA'],
#         'ZIP': ['ZIC', 'GVWY', 'AA', 'MIX', 'RaF']
#     }

# Dynamic Market data high interval (0.25-2)

dominance_data = {
        'RaF': ['ZIC'],
        'GVWY': ['ZIP', 'AA', 'RaF', 'ZIC', 'MIX'],
        'AA': ['RaF'],
        'MIX': ['RaF', 'ZIP', 'ZIC', 'AA'],
        'ZIC': ['AA'],
        'ZIP': ['ZIC', 'AA', 'RaF']
    }

# Sensitive Market data only bacth interval = 0.05

# dominance_data = {
#         'RaF': ['ZIC'],
#         'GVWY': ['MIX', 'RaF', 'ZIC'],
#         'AA': ['RaF', 'ZIC', 'GVWY'],
#         'MIX': ['RaF', 'ZIC', 'AA'],
#         'ZIC': [],
#         'ZIP': ['ZIC', 'GVWY', 'AA', 'MIX', 'RaF']
#     }

algorithms = list(dominance_data.keys())
G.add_nodes_from(algorithms)


for algo1, dominated_algos in dominance_data.items():
    for algo2 in dominated_algos:
        G.add_edge(algo1, algo2)


pos = nx.circular_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=800, node_color='white', edgecolors='black')
nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True, arrowsize=20, arrowstyle='->')


labels = {n: f"{n}\n{G.out_degree(n)}/{G.in_degree(n)}" for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=8)  # 只改变这里的字体大小


plt.axis('off')
plt.tight_layout()
plt.show()