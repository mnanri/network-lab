import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_sample(n=100, m=2, graph_num=4, link_level=5):
  G = []
  for i in range(graph_num):
    G.append(nx.barabasi_albert_graph(n, m))

  deg_order_list = []
  for i in range(n):
    tmp = []
    for j in range(graph_num):
      tmp.append(sorted(dict(G[j].degree()).items(), key=lambda x: x[1], reverse=True)[i][0] + j*n)
    deg_order_list.append(tmp)

  # Labeling Order with Degree Centrality / Closeness Centrality / PageRank
  cent_order_list = []
  for i in range(n):
    tmp = []
    for j in range(graph_num):
      # tmp.append(sorted(dict(G[j].degree()).items(), key=lambda x: x[1])[i][0] + j*n)
      tmp.append(sorted(dict(nx.closeness_centrality(G[j])).items(), key=lambda x: x[1])[i][0] + j*n)
      # tmp.append(sorted(dict(nx.pagerank(G[j])).items(), key=lambda x: x[1])[i][0] + j*n)
    cent_order_list.append(tmp)

  # if cent_order_list[0][0] != max(dict(nx.closeness_centrality(G[0])).items(), key=lambda x: x[1])[0]:
  #   print("error")

  for i in range(graph_num):
    mapping = {}
    for j in range(n):
      mapping[j] = j + n*(i)
    G[i] = nx.relabel_nodes(G[i], mapping)

  H = G[0]
  for i in range(graph_num-1):
    H = nx.compose(H, G[i+1])

  for i in range(graph_num):
    for j in range(i+1, graph_num):
      for k in range(link_level):
        for l in range(k, link_level):
          r = np.random.rand()
          p1 = 1 - 1/nx.degree(H)[deg_order_list[k][i]]
          p2 = 1 - 1/nx.degree(H)[deg_order_list[l][j]]
          p = p1 * p2
          if r < p:
            H.add_edge(deg_order_list[k][i], deg_order_list[l][j])

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

  return H, I, cent_order_list

generate_sample()
