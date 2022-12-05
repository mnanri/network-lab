import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_sample(n=100, m=2, graph_num=4):

  '''
  print("Generating scale-free graph with %d nodes" % n)
  G = nx.barabasi_albert_graph(n, m)
  print("Number of Nodes: ",nx.number_of_nodes(G))
  print("Number of Edges: ",nx.number_of_edges(G))
  print("Max Degree: ",np.max(list(dict(G.degree()).values())))
  print("Average Shortest Path Length: %f" % nx.average_shortest_path_length(G))

  pos = nx.random_layout(G)
  nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
  # plt.show()
  '''

  G = []
  for i in range(graph_num):
    G.append(nx.barabasi_albert_graph(n, m))

  max_deg_nodes = {}
  second_max_deg_nodes = {}
  third_max_deg_nodes = {}
  for i in range(graph_num):
    # print("node that has the largest degree in graph %d: %d" % (i, max(dict(G[i].degree()).items(), key=lambda x: x[1])[0]))
    max_deg_nodes[i] = max(dict(G[i].degree()).items(), key=lambda x: x[1])[0] + i*n
    second_max_deg_nodes[i] = sorted(dict(G[i].degree()).items(), key=lambda x: x[1], reverse=True)[1][0] + i*n
    third_max_deg_nodes[i] = sorted(dict(G[i].degree()).items(), key=lambda x: x[1], reverse=True)[2][0] + i*n

  min_deg_nodes = {}
  second_min_deg_nodes = {}
  third_min_deg_nodes = {}
  for i in range(graph_num):
    # print("node that has the smallest degree in graph %d: %d" % (i, min(dict(G[i].degree()).items(), key=lambda x: x[1])[0]))
    min_deg_nodes[i] = min(dict(G[i].degree()).items(), key=lambda x: x[1])[0] + i*n
    second_min_deg_nodes[i] = sorted(dict(G[i].degree()).items(), key=lambda x: x[1])[1][0] + i*n
    third_min_deg_nodes[i] = sorted(dict(G[i].degree()).items(), key=lambda x: x[1])[2][0] + i*n

  max_cent_nodes = {}
  for i in range(graph_num):
    # print("node that has the largest centrality in graph %d: %d" % (i, max(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0]))
    max_cent_nodes[i] = max(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0] + i*n

  cent_order_list = []
  for i in range(n):
    tmp = []
    for j in range(graph_num):
      tmp.append(sorted(dict(G[j].degree()).items(), key=lambda x: x[1], reverse=True)[i][0])
    cent_order_list.append(tmp)

  if cent_order_list[0][0] != max(dict(G[0].degree()).items(), key=lambda x: x[1])[0]:
    print("error")

  for i in range(n):
    for j in range(graph_num):
      cent_order_list[i][j] += j*n

  # min_cent_nodes = {}
  # for i in range(graph_num):
  #   # print("node that has the smallest centrality in graph %d: %d" % (i, min(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0]))
  #   min_cent_nodes[i] = min(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0]

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

  for i in range(graph_num):
    mapping = {}
    for j in range(n):
      mapping[j] = j + n*(i)
    G[i] = nx.relabel_nodes(G[i], mapping)

  H = G[0]
  for i in range(graph_num-1):
    H = nx.compose(H, G[i+1])

  # producing artificially binomial distribution
  for i in range(graph_num):
    for j in range(i+1, graph_num):
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

# generate_sample()
