import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch

def generate_flexible_linked_sample(n=100, m=2, graph_num=4, link_level=10):
  G = []
  for i in range(graph_num):
    G.append(nx.barabasi_albert_graph(n, m))

  deg_order_list = []
  for i in range(n):
    tmp = []
    for j in range(graph_num):
      tmp.append(sorted(dict(G[j].degree()).items(), key=lambda x: x[1], reverse=True)[i][0] + j*n)
    deg_order_list.append(tmp)

  if deg_order_list[0][0] != max(dict(G[0].degree()).items(), key=lambda x: x[1])[0]:
    print("error")

  cent_order_list = []
  for i in range(n):
    tmp = []
    for j in range(graph_num):
      tmp.append(sorted(dict(G[j].degree()).items(), key=lambda x: x[1], reverse=True)[i][0] + j*n)
    cent_order_list.append(tmp)

  if cent_order_list[0][0] != max(dict(G[0].degree()).items(), key=lambda x: x[1])[0]:
    print("error")

  for i in range(graph_num):
    mapping = {}
    for j in range(n):
      mapping[j] = j + i*n
    G[i] = nx.relabel_nodes(G[i], mapping)

  H = G[0]
  for i in range(1, graph_num):
    H = nx.compose(H, G[i])

  for i in range(link_level):
    for j in range(graph_num):
      for k in range(j+1, graph_num):
        r = np.random.rand()
        if r < 0.5:
          H.add_edge(deg_order_list[i][j], deg_order_list[i][k])

  I = H.copy()
  for node in I.nodes():
    I.nodes[node]['label'] = -1

  for i in range(graph_num):
      I.nodes[cent_order_list[0][i]]['label'] = i

  for node in H.nodes():
    H.nodes[node]['label'] = node // n

  for node in H.nodes():
    feature = np.zeros(n, dtype=np.float32)
    feature[node % n] = 1
    feature = torch.from_numpy(feature)
    H.nodes[node]['feature'] = feature
    I.nodes[node]['feature'] = feature

  '''
  pos = nx.circular_layout(I)
  pattern = [ 'gray' if I.nodes[node]['label'] == -1 else 'blue' if node < n else 'green' if node < 2*n else 'orange' if node < 3*n else 'red' for node in I.nodes() ]
  nx.draw(I, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  plt.show()
  '''
  '''
  pos = nx.circular_layout(H)
  pattern = [ 'blue' if node < n else 'green' if node < 2*n else 'orange' if node < 3*n else 'red' for node in H.nodes() ]
  nx.draw(H, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  plt.show()
  '''
  # print(H.number_of_nodes())
  # print(H.nodes())
  # print(cent_order_list)
  return H, I, cent_order_list

generate_flexible_linked_sample(100, 2, 4, 50)
