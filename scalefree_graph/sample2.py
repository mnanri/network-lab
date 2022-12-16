import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch

def generate_umblance_sample(n1=100, n2=25, m=2, graph_num=4, large_graph_num=2, link_level=20):
  G = []
  small_graph_num = graph_num - large_graph_num
  for i in range(large_graph_num):
    G.append(nx.barabasi_albert_graph(n1, m))

  for i in range(small_graph_num):
    G.append(nx.barabasi_albert_graph(n2, m))

  deg_order_list = []
  for i in range(n1):
    tmp = []
    for j in range(large_graph_num):
      tmp.append(sorted(dict(G[j].degree()).items(), key=lambda x: x[1], reverse=True)[i][0] + j*n1)
    if i < n2:
      for j in range(small_graph_num):
        tmp.append(sorted(dict(G[j+large_graph_num].degree()).items(), key=lambda x: x[1], reverse=True)[i][0] + j*n2 + large_graph_num*n1)
    deg_order_list.append(tmp)

  deg_cent_dict = {}
  for i in range(n1):
    for j in range(large_graph_num):
      deg_cent_dict[deg_order_list[i][j]] = nx.degree_centrality(G[j])[deg_order_list[i][j]-j*n1]
  for i in range(n2):
    for j in range(small_graph_num):
      deg_cent_dict[deg_order_list[i][j+large_graph_num]] = nx.degree_centrality(G[j+large_graph_num])[deg_order_list[i][j+large_graph_num]-j*n2-large_graph_num*n1]
  max_deg_cent_list = []
  for i in range(graph_num):
    max_deg_cent_list.append(deg_cent_dict[deg_order_list[0][i]])
  for i in range(n1):
    for j in range(large_graph_num):
      deg_cent_dict[deg_order_list[i][j]] /= max_deg_cent_list[j]
  for i in range(n2):
    for j in range(large_graph_num, graph_num):
      deg_cent_dict[deg_order_list[i][j]] /= max_deg_cent_list[j]
  # print(deg_cent_dict)

  # Labeling Order with Degree Centrality / Closeness Centrality
  large_graph_cent_order_list = []
  for i in range(n1):
    tmp = []
    for j in range(large_graph_num):
      # tmp.append(sorted(dict(G[j].degree()).items(), key=lambda x: x[1])[i][0] + j*n1)
      tmp.append(sorted(dict(nx.closeness_centrality(G[j])).items(), key=lambda x: x[1], reverse=True)[i][0] + j*n1)
      # tmp.append(sorted(dict(nx.pagerank(G[j])).items(), key=lambda x: x[1], reverse=True)[i][0] + j*n1)
    large_graph_cent_order_list.append(tmp)

  # Labeling Order with Degree Centrality / Closeness Centrality
  small_graph_cent_order_list = []
  for i in range(n2):
    tmp = []
    for j in range(small_graph_num):
      # tmp.append(sorted(dict(G[j+large_graph_num].degree()).items(), key=lambda x: x[1])[i][0] + j*n2 + large_graph_num*n1)
      tmp.append(sorted(dict(nx.closeness_centrality(G[j+large_graph_num])).items(), key=lambda x: x[1], reverse=True)[i][0] + j*n2 + large_graph_num*n1)
      # tmp.append(sorted(dict(nx.pagerank(G[j+large_graph_num])).items(), key=lambda x: x[1], reverse=True)[i][0] + j*n2 + large_graph_num*n1)
    small_graph_cent_order_list.append(tmp)

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

  for i in range(large_graph_num):
    for j in range(i+1,large_graph_num):
      for k in range(link_level):
        for l in range(k, link_level):
          r = np.random.rand()
          p1 = deg_cent_dict[deg_order_list[k][i]]
          p2 = deg_cent_dict[deg_order_list[l][j]]
          p = p1 * p2
          if r < p:
            H.add_edge(deg_order_list[k][i], deg_order_list[l][j])

  for i in range(large_graph_num):
    for j in range(large_graph_num, graph_num):
      for k in range(link_level):
        for l in range(k, link_level):
          if l >= n2:
            break
          r = np.random.rand()
          p1 = deg_cent_dict[deg_order_list[k][i]]
          p2 = deg_cent_dict[deg_order_list[l//(n1//n2)][j]]
          p = p1 * p2 / (n1//n2)
          if r < p:
            H.add_edge(deg_order_list[k][i], deg_order_list[l//(n1//n2)][j])

  for i in range(large_graph_num, graph_num):
    for j in range(i+1,graph_num):
      for k in range(link_level//(n1//n2)):
        for l in range(k, link_level//(n1//n2)):
          if l >= n2:
            break
          r = np.random.rand()
          p1 = deg_cent_dict[deg_order_list[k][i]]
          p2 = deg_cent_dict[deg_order_list[l][j]]
          p = p1 * p2
          if r < p:
            H.add_edge(deg_order_list[k][i], deg_order_list[l][j])

  I = H.copy()
  for node in I.nodes():
    I.nodes[node]['label'] = -1

  for i in range(large_graph_num):
    I.nodes[large_graph_cent_order_list[0][i]]['label'] = i

  for i in range(small_graph_num):
    I.nodes[small_graph_cent_order_list[0][i]]['label'] = i+large_graph_num

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
  '''
  pos = nx.circular_layout(H)
  pattern = [ 'blue' if node < n1 else 'green' if node < 2*n1 else 'orange' if node < n2+2*n1 else 'red' for node in H.nodes() ]
  nx.draw(H, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
  plt.show()
  '''

  return H, I, large_graph_cent_order_list, small_graph_cent_order_list

# generate_umblance_sample()
