import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

n = 100
m = 2

print("Generating scale-free graph with %d nodes" % n)
G = nx.barabasi_albert_graph(n, m)
print("Number of Nodes: ",nx.number_of_nodes(G))
print("Number of Edges: ",nx.number_of_edges(G))
print("Average Shortest Path Length: %f" % nx.average_shortest_path_length(G))

pos = nx.random_layout(G)
nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
plt.show()
