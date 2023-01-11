import generate_network
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx

def check_graph(data):
    '''Show Graph Information'''
    print("Structure of Graph:\n>>>", data)
    print("Key of Graph:\n>>>", data.keys)
    print("Count of Nodes:\n>>>", data.num_nodes)
    print("Count of Edges:\n>>>", data.num_edges)
    print("Count of Features in a Node:\n>>>", data.num_node_features)
    print("Is There Isorated Nodes?:\n>>>", data.has_isolated_nodes())
    print("Is There Self-loops?:\n>>>", data.has_self_loops())
    print("========== Features of Nodes: x ==========\n", data['feature'])
    print("========== Class of Nodes: label =============\n", data['label'])
    print("========== Type of Edge ==================\n", data['edge_index'])

def show_graph():

  n = 100
  a, _ = generate_network.generate_network()

  print("========== Graph A ==========")
  check_graph(from_networkx(a))

  # pos = nx.circular_layout(a)
  pos = nx.spring_layout(a)
  pattern = [ 'blue' if node < n else 'green' if node < 2*n else 'orange' if node < 3*n else 'red' for node in a.nodes() ]
  nx.draw(a, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  plt.show()

show_graph()
