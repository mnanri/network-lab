import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch

def generate_labeled_list(G, option='degree'):
  ordered_list = []
  for i in range(len(G)):
    if option == 'degree':
      ordered_list.append(sorted(dict(G[i].degree()).items(), key=lambda x: x[1], reverse=True))
    elif option == 'closeness':
      ordered_list.append(sorted(dict(nx.closeness_centrality(G[i])).items(), key=lambda x: x[1], reverse=True))
    elif option == 'betweenness':
      ordered_list.append(sorted(dict(nx.betweenness_centrality(G[i])).items(), key=lambda x: x[1], reverse=True))
    elif option == 'pagerank':
      ordered_list.append(sorted(dict(nx.pagerank(G[i])).items(), key=lambda x: x[1], reverse=True))
    else :
      print('label error')
      exit()

  labeling_order_list = []
  for i in G[0].nodes:
    tmp = []
    for j in range(len(G)):
      tmp.append(ordered_list[j][i][0] + j*len(G[0].nodes))
    labeling_order_list.append(tmp)

  return labeling_order_list

def generate_network(n=100, m=2, community=4, link=10, option='degree'):
  G = []
  for i in range(community):
    G.append(nx.barabasi_albert_graph(n, m))

  dg_order_list_key = []
  dg_order_list_val = []
  for i in range(community):
    tmp = sorted(dict(G[i].degree()).items(), key=lambda x: x[1], reverse=True)
    tmp_key = []
    tmp_val = []
    for j in range(n):
      tmp_key.append(tmp[j][0] + i*n)
      tmp_val.append(tmp[j][1])
    dg_order_list_key.append(tmp_key)
    dg_order_list_val.append(tmp_val)

  max_dg_list = []
  for i in range(community):
    max_dg_list.append(dg_order_list_val[i][0])

  for i in range(community):
    for j in range(n):
      dg_order_list_val[i][j] /= max_dg_list[i]

  degree_dict = {}
  for i in range(community):
    for j in range(n):
      degree_dict[dg_order_list_key[i][j]] = dg_order_list_val[i][j]

  for i in range(community):
    mappping = {}
    for j in range(n):
      mappping[j] = j + i*n
    G[i] = nx.relabel_nodes(G[i], mappping)

  H = G[0]
  for i in range(1, community):
    H = nx.compose(H, G[i])

  cnt = 0
  for i in range(community):
    for j in range(i+1, community):
      for k in range(link):
        for l in range(k, link):
          r = np.random.rand()
          p = degree_dict[dg_order_list_key[i][k]]
          q = degree_dict[dg_order_list_key[j][l]]
          if r < p * q:
            cnt += 1
            H.add_edge(dg_order_list_key[i][k], dg_order_list_key[j][l])

  for node in H.nodes():
    H.nodes[node]['label'] = node // n

  for node in H.nodes():
    feature = np.zeros(n, dtype=np.float32)
    feature[node % n] = 1
    feature = torch.from_numpy(feature)
    H.nodes[node]['feature'] = feature

  ordered_list = []
  for i in range(community):
    if option == 'degree':
      ordered_list.append(sorted(dict(G[i].degree()).items(), key=lambda x: x[1], reverse=True))
    elif option == 'closeness':
      ordered_list.append(sorted(dict(nx.closeness_centrality(G[i])).items(), key=lambda x: x[1], reverse=True))
    elif option == 'betweenness':
      ordered_list.append(sorted(dict(nx.betweenness_centrality(G[i])).items(), key=lambda x: x[1], reverse=True))
    elif option == 'pagerank':
      ordered_list.append(sorted(dict(nx.pagerank(G[i])).items(), key=lambda x: x[1], reverse=True))
    else :
      print('label error')
      exit()

  # print(ordered_list)

  labeling_order_list = []
  for i in range(n):
    tmp = []
    for j in range(community):
      tmp.append(ordered_list[j][i][0])
    labeling_order_list.append(tmp)

  # print(labeling_order_list)

  # Show Network
  # pos = nx.circular_layout(H)
  # pos = nx.spring_layout(H)
  # pattern = [ 'blue' if node < n else 'green' if node < 2*n else 'orange' if node < 3*n else 'red' for node in H.nodes() ]
  # nx.draw(H, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  # plt.show()

  return H, labeling_order_list

def generate_network2(n=100, m=2, community=4, link=10):
  G = []
  for i in range(community):
    G.append(nx.barabasi_albert_graph(n, m))

  dg_order_list_key = []
  dg_order_list_val = []
  for i in range(community):
    tmp = sorted(dict(G[i].degree()).items(), key=lambda x: x[1], reverse=True)
    tmp_key = []
    tmp_val = []
    for j in range(n):
      tmp_key.append(tmp[j][0] + i*n)
      tmp_val.append(tmp[j][1])
    dg_order_list_key.append(tmp_key)
    dg_order_list_val.append(tmp_val)

  max_dg_list = []
  for i in range(community):
    max_dg_list.append(dg_order_list_val[i][0])

  for i in range(community):
    for j in range(n):
      dg_order_list_val[i][j] /= max_dg_list[i]

  degree_dict = {}
  for i in range(community):
    for j in range(n):
      degree_dict[dg_order_list_key[i][j]] = dg_order_list_val[i][j]

  for i in range(community):
    mappping = {}
    for j in range(n):
      mappping[j] = j + i*n
    G[i] = nx.relabel_nodes(G[i], mappping)

  H = G[0]
  for i in range(1, community):
    H = nx.compose(H, G[i])

  cnt = 0
  for i in range(community):
    for j in range(i+1, community):
      for k in range(link):
        for l in range(k, link):
          r = np.random.rand()
          p = degree_dict[dg_order_list_key[i][k]]
          q = degree_dict[dg_order_list_key[j][l]]
          if r < p * q:
            cnt += 1
            H.add_edge(dg_order_list_key[i][k], dg_order_list_key[j][l])

  for node in H.nodes():
    H.nodes[node]['label'] = -1

  candidate = [ i for i in range(n) ]
  for i in range(community):
    random.shuffle(candidate)
    for j in range(link):
      H.nodes[dg_order_list_key[i][candidate[j]]]['label'] = i

  for node in H.nodes():
    feature = np.zeros(n, dtype=np.float32)
    feature[node % n] = 1
    feature = torch.from_numpy(feature)
    H.nodes[node]['feature'] = feature

  # pos = nx.circular_layout(H)
  pos = nx.spring_layout(H)
  pattern = [ 'gray' if H.nodes[node]['label'] == -1 else 'blue' if node < n else 'green' if node < 2*n else 'orange' if node < 3*n else 'red' for node in H.nodes() ]
  nx.draw(H, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  plt.show()

  return H

generate_network()
# generate_network2()
