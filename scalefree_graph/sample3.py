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

  # Labeling Order with Degree Centrality / Closeness Centrality / PageRank
  cent_order_list = []
  for i in range(n):
    tmp = []
    for j in range(graph_num):
      # tmp.append(sorted(dict(G[j].degree()).items(), key=lambda x: x[1])[i][0] + j*n)
      tmp.append(sorted(dict(nx.closeness_centrality(G[j])).items(), key=lambda x: x[1])[i][0] + j*n)
      # tmp.append(sorted(dict(nx.pagerank(G[j])).items(), key=lambda x: x[1], reverse=True)[i][0] + j*n)
    cent_order_list.append(tmp)

  # if cent_order_list[0][0] != max(dict(G[0].degree()).items(), key=lambda x: x[1])[0]:
  #   print("error")

  for i in range(graph_num):
    mapping = {}
    for j in range(n):
      mapping[j] = j + i*n
    G[i] = nx.relabel_nodes(G[i], mapping)

  H = G[0]
  for i in range(1, graph_num):
    H = nx.compose(H, G[i])

  cnt = 0
  for i in range(link_level):
    for j in range(graph_num):
      for k in range(j+1, graph_num):
        r = np.random.rand()
        p1 = 1 - 1/nx.degree(H)[deg_order_list[i][j]]
        p2 = 1 - 1/nx.degree(H)[deg_order_list[i][k]]
        p = p1 * p2
        if r < p:
          cnt += 1
          H.add_edge(deg_order_list[i][j], deg_order_list[i][k])

  for node in H.nodes():
    H.nodes[node]['label'] = node // n

  for node in H.nodes():
    feature = np.zeros(n, dtype=np.float32)
    feature[node % n] = 1
    feature = torch.from_numpy(feature)
    H.nodes[node]['feature'] = feature

  print("link num: ", len(H.edges()))
  print("link num between class: ", cnt)

  pos = nx.circular_layout(H)
  pattern = [ 'blue' if node < n else 'green' if node < 2*n else 'orange' if node < 3*n else 'red' for node in H.nodes() ]
  nx.draw(H, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  plt.show()

  return H, cent_order_list

def generate_flexible_linked_sample_(n=100, m=2, graph_num=4, link_level=10):
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

  # Labeling Order with Degree Centrality / Closeness Centrality / PageRank
  cent_order_list = []
  for i in range(n):
    tmp = []
    for j in range(graph_num):
      # tmp.append(sorted(dict(G[j].degree()).items(), key=lambda x: x[1])[i][0] + j*n)
      # tmp.append(sorted(dict(nx.closeness_centrality(G[j])).items(), key=lambda x: x[1])[i][0] + j*n)
      tmp.append(sorted(dict(nx.pagerank(G[j])).items(), key=lambda x: x[1], reverse=True)[i][0] + j*n)
    cent_order_list.append(tmp)

  # if cent_order_list[0][0] != max(dict(G[0].degree()).items(), key=lambda x: x[1])[0]:
  #   print("error")

  for i in range(graph_num):
    mapping = {}
    for j in range(n):
      mapping[j] = j + i*n
    G[i] = nx.relabel_nodes(G[i], mapping)

  H = G[0]
  for i in range(1, graph_num):
    H = nx.compose(H, G[i])

  cnt = 0
  for i in range(link_level):
    for j in range(graph_num):
      for k in range(j+1, graph_num):
        r = np.random.rand()
        p1 = 1 - 1/nx.degree(H)[deg_order_list[i][j]]
        p2 = 1 - 1/nx.degree(H)[deg_order_list[i][k]]
        p = p1 * p2
        if r < p:
          cnt += 1
          H.add_edge(deg_order_list[i][j], deg_order_list[i][k])

  for node in H.nodes():
    H.nodes[node]['label'] = node // n

  for node in H.nodes():
    feature = np.zeros(n, dtype=np.float32)
    feature[node % n] = 1
    feature = torch.from_numpy(feature)
    H.nodes[node]['feature'] = feature

  print("link num: ", len(H.edges()))
  print("link num between class: ", cnt)

  pos = nx.circular_layout(H)
  pattern = [ 'blue' if node < n else 'green' if node < 2*n else 'orange' if node < 3*n else 'red' for node in H.nodes() ]
  nx.draw(H, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  plt.show()

  return H, cent_order_list

generate_flexible_linked_sample(100,2,4,20)
