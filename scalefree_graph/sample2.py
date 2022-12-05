import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch

def generate_umblance_network(n1=100, n2=25, m=2, graph_num=4, large_graph_num=2):
  G = []
  small_graph_num = graph_num - large_graph_num
  for i in range(large_graph_num):
    G.append(nx.barabasi_albert_graph(n1, m))

  for i in range(small_graph_num):
    G.append(nx.barabasi_albert_graph(n2, m))

  max_deg_nodes = {}
  second_max_deg_nodes = {}
  third_max_deg_nodes = {}
  for i in range(large_graph_num):
    max_deg_nodes[i] = max(dict(G[i].degree()).items(), key=lambda x: x[1])[0] + i*n1
    second_max_deg_nodes[i] = sorted(dict(G[i].degree()).items(), key=lambda x: x[1], reverse=True)[1][0] + i*n1
    third_max_deg_nodes[i] = sorted(dict(G[i].degree()).items(), key=lambda x: x[1], reverse=True)[2][0] + i*n1

  for i in range(small_graph_num):
    max_deg_nodes[i+large_graph_num] = max(dict(G[i+large_graph_num].degree()).items(), key=lambda x: x[1])[0] + i*n2 + large_graph_num*n1
    second_max_deg_nodes[i+large_graph_num] = sorted(dict(G[i+large_graph_num].degree()).items(), key=lambda x: x[1], reverse=True)[1][0] + i*n2 + large_graph_num*n1
    third_max_deg_nodes[i+large_graph_num] = sorted(dict(G[i+large_graph_num].degree()).items(), key=lambda x: x[1], reverse=True)[2][0] + i*n2 + large_graph_num*n1

  min_deg_nodes = {}
  second_min_deg_nodes = {}
  third_min_deg_nodes = {}
  for i in range(large_graph_num):
    min_deg_nodes[i] = min(dict(G[i].degree()).items(), key=lambda x: x[1])[0] + i*n1
    second_min_deg_nodes[i] = sorted(dict(G[i].degree()).items(), key=lambda x: x[1])[1][0] + i*n1
    third_min_deg_nodes[i] = sorted(dict(G[i].degree()).items(), key=lambda x: x[1])[2][0] + i*n1

  for i in range(small_graph_num):
    min_deg_nodes[i+large_graph_num] = min(dict(G[i+large_graph_num].degree()).items(), key=lambda x: x[1])[0] + i*n2 + large_graph_num*n1
    second_min_deg_nodes[i+large_graph_num] = sorted(dict(G[i+large_graph_num].degree()).items(), key=lambda x: x[1])[1][0] + i*n2 + large_graph_num*n1
    third_min_deg_nodes[i+large_graph_num] = sorted(dict(G[i+large_graph_num].degree()).items(), key=lambda x: x[1])[2][0] + i*n2 + large_graph_num*n1

  max_cent_nodes = {}
  for i in range(large_graph_num):
    max_cent_nodes[i] = max(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0] + i*n1

  for i in range(small_graph_num):
    max_cent_nodes[i+large_graph_num] = max(nx.betweenness_centrality(G[i+large_graph_num]).items(), key=lambda x: x[1])[0] + i*n2 + large_graph_num*n1

  min_cent_nodes = {}
  for i in range(large_graph_num):
    min_cent_nodes[i] = min(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0]

  for i in range(small_graph_num):
    min_cent_nodes[i+large_graph_num] = min(nx.betweenness_centrality(G[i+large_graph_num]).items(), key=lambda x: x[1])[0] + i*n2 + large_graph_num*n1

  large_graph_cent_order_list = []
  for i in range(n1):
    tmp = []
    for j in range(large_graph_num):
      tmp.append(sorted(dict(G[j].degree()).items(), key=lambda x: x[1], reverse=True)[i][0])
    large_graph_cent_order_list.append(tmp)

  if large_graph_cent_order_list[0][0] != max(dict(G[0].degree()).items(), key=lambda x: x[1])[0]:
    print("error")

  for i in range(n1):
    for j in range(large_graph_num):
      large_graph_cent_order_list[i][j] += j*n1

  small_graph_cent_order_list = []
  for i in range(n2):
    tmp = []
    for j in range(small_graph_num):
      tmp.append(sorted(dict(G[j+large_graph_num].degree()).items(), key=lambda x: x[1], reverse=True)[i][0])
    small_graph_cent_order_list.append(tmp)

  if small_graph_cent_order_list[0][0] != max(dict(G[large_graph_num].degree()).items(), key=lambda x: x[1])[0]:
    print("error")

  '''
  max_bet_nodes = {}
  for i in range(graph_num):
    print("node that has the largest betweenness in graph %d: %d" % (i, max(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0]))
    max_bet_nodes[i] = max(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0]

  max_pg_nodes = {}
  for i in range(graph_num):
    print("node that has the largest pagerank in graph %d: %d" % (i, max(nx.pagerank(G[i]).items(), key=lambda x: x[1])[0]))
    max_pg_nodes[i] = max(nx.pagerank(G[i]).items(), key=lambda x: x[1])[0]
  '''

  for i in range(large_graph_num):
    mapping = {}
    for j in range(n1):
      mapping[j] = j + n1*i
    G[i] = nx.relabel_nodes(G[i], mapping)

  for i in range(small_graph_num):
    mapping = {}
    for j in range(n2):
      mapping[j] = j + n2*i + n1*large_graph_num
    G[i+large_graph_num] = nx.relabel_nodes(G[i+large_graph_num], mapping)

  H = G[0]
  for i in range(graph_num-1):
    H = nx.compose(H, G[i+1])

  # producing artificially binomial distribution
  for i in range(graph_num):
    for j in range(i+1,graph_num):
      rand1 = np.random.rand()
      rand2 = np.random.rand()
      rand3 = np.random.rand()
      rand4 = np.random.rand()
      rand5 = np.random.rand()
      rand6 = np.random.rand()
      if rand1 < 0.5:
        H.add_edge(max_deg_nodes[i], max_deg_nodes[j])
      if rand2 < 0.5:
        H.add_edge(max_deg_nodes[i], second_max_deg_nodes[j])
      if rand3 < 0.5:
        H.add_edge(max_deg_nodes[i], third_max_deg_nodes[j])
      if rand4 < 0.5:
        H.add_edge(second_max_deg_nodes[i], second_max_deg_nodes[j])
      if rand5 < 0.5:
        H.add_edge(second_max_deg_nodes[i], third_max_deg_nodes[j])
      if rand6 < 0.5:
        H.add_edge(third_max_deg_nodes[i], third_max_deg_nodes[j])

  '''
  for i in range(graph_num):
    for j in range(i+1, graph_num):
      rand1 = np.random.rand()
      rand2 = np.random.rand()
      rand3 = np.random.rand()
      rand4 = np.random.rand()
      rand5 = np.random.rand()
      rand6 = np.random.rand()
      if rand1 < 0.5:
        H.add_edge(min_deg_nodes[i], min_deg_nodes[j])
      if rand2 < 0.5:
        H.add_edge(min_deg_nodes[i], second_min_deg_nodes[j])
      if rand3 < 0.5:
        H.add_edge(min_deg_nodes[i], third_min_deg_nodes[j])
      if rand4 < 0.5:
        H.add_edge(second_min_deg_nodes[i], second_min_deg_nodes[j])
      if rand5 < 0.5:
        H.add_edge(second_min_deg_nodes[i], third_min_deg_nodes[j])
      if rand6 < 0.5:
        H.add_edge(third_min_deg_nodes[i], third_min_deg_nodes[j])
  '''

  I = H.copy()
  for node in I.nodes():
    I.nodes[node]['label'] = -1

  for i in range(graph_num):
    I.nodes[max_cent_nodes[i]]['label'] = i

  # for i in range(graph_num):
  #   I.nodes[min_cent_nodes[i]+i*n]['label'] = i

  for i in range(large_graph_num):
    for j in range(n1):
      H.nodes[j+i*n1]['label'] = i

  for i in range(small_graph_num):
    for j in range(n2):
      H.nodes[j+i*n2+n1*large_graph_num]['label'] = i+large_graph_num

  for node in H.nodes():
    feature = np.zeros(n1, dtype=np.float32)
    feature[node % n1] = 1
    feature = torch.from_numpy(feature)
    H.nodes[node]['feature'] = feature
    I.nodes[node]['feature'] = feature

  '''
  pos = nx.circular_layout(I)
  pattern = [ 'gray' if I.nodes[node]['label'] == -1 else 'blue' if node < n1 else 'green' if node < 2*n1 else 'orange' if node < n2+2*n1 else 'red' for node in I.nodes() ]
  nx.draw(I, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  plt.show()
  '''

  pos = nx.circular_layout(H)
  pattern = [ 'blue' if node < n1 else 'green' if node < 2*n1 else 'orange' if node < n2+2*n1 else 'red' for node in H.nodes() ]
  nx.draw(H, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  plt.show()

  return H, I, large_graph_cent_order_list, small_graph_cent_order_list

generate_umblance_network()
