import sample
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
  a,b = sample.generate_sample(n)

  '''
  pos = nx.circular_layout(y)
  pattern = [ 'gray' if y.nodes[node]['label'] == -1 else 'blue' if node < n else 'green' if node < 2*n else 'orange' if node < 3*n else 'red' for node in y.nodes() ]
  nx.draw(y, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  plt.show()
  '''

  '''
  pos = nx.circular_layout(a)
  pattern = [ 'blue' if node < n else 'green' if node < 2*n else 'orange' if node < 3*n else 'red' for node in a.nodes() ]
  nx.draw(a, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  plt.show()
  '''

  a_data = from_networkx(a)
  b_data = from_networkx(b)

  print("========== Graph A ==========")
  check_graph(a_data)
  print("========== Graph B ==========")
  check_graph(b_data)

show_graph()
