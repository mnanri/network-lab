import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

n = 100
m = 2
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

graph_num = 4
G = []
for i in range(graph_num):
  G.append(nx.barabasi_albert_graph(n, m))

max_deg_nodes = {}
second_max_deg_nodes = {}
third_max_deg_nodes = {}
for i in range(graph_num):
  # print("max degree of graph %d: %d" % (i, np.max(list(dict(G[i].degree()).values()))))
  # print("average shortest path length of graph %d: %f" % (i, nx.average_shortest_path_length(G[i])))
  print("node that has the largest degree in graph %d: %d" % (i, max(dict(G[i].degree()).items(), key=lambda x: x[1])[0]))
  max_deg_nodes[i] = max(dict(G[i].degree()).items(), key=lambda x: x[1])[0]
  second_max_deg_nodes[i] = sorted(dict(G[i].degree()).items(), key=lambda x: x[1], reverse=True)[1][0]
  third_max_deg_nodes[i] = sorted(dict(G[i].degree()).items(), key=lambda x: x[1], reverse=True)[2][0]

max_cent_nodes = {}
for i in range(graph_num):
  print("node that has the largest centrality in graph %d: %d" % (i, max(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0]))
  max_cent_nodes[i] = max(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0]

max_bet_nodes = {}
for i in range(graph_num):
  print("node that has the largest betweenness in graph %d: %d" % (i, max(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0]))
  max_bet_nodes[i] = max(nx.betweenness_centrality(G[i]).items(), key=lambda x: x[1])[0]

max_pg_nodes = {}
for i in range(graph_num):
  print("node that has the largest pagerank in graph %d: %d" % (i, max(nx.pagerank(G[i]).items(), key=lambda x: x[1])[0]))
  max_pg_nodes[i] = max(nx.pagerank(G[i]).items(), key=lambda x: x[1])[0]

for i in range(graph_num-1):
  mapping = {}
  for j in range(n):
    mapping[j] = j + n*(i+1)
  G[i+1] = nx.relabel_nodes(G[i+1], mapping)

H = G[0]
for i in range(graph_num-1):
  H = nx.compose(H, G[i+1])

for i in range(graph_num):
  for j in range(i+1, graph_num):
    rand1 = np.random.rand()
    rand2 = np.random.rand()
    rand3 = np.random.rand()
    if rand1 < 0.5:
      H.add_edge(max_deg_nodes[i], max_deg_nodes[j]+j*n)
    if rand2 < 0.5:
      H.add_edge(max_deg_nodes[i], second_max_deg_nodes[j]+j*n)
    if rand3 < 0.5:
      H.add_edge(max_deg_nodes[i], third_max_deg_nodes[j]+j*n)

for i in range(graph_num):
  for j in range(i+1, graph_num):
    rand1 = np.random.rand()
    rand2 = np.random.rand()
    if rand1 < 0.5:
      H.add_edge(second_max_deg_nodes[i], second_max_deg_nodes[j]+j*n)
    if rand2 < 0.5:
      H.add_edge(second_max_deg_nodes[i], third_max_deg_nodes[j]+j*n)

for i in range(graph_num):
  for j in range(i+1, graph_num):
    rand = np.random.rand()
    if rand < 0.5:
      H.add_edge(third_max_deg_nodes[i], third_max_deg_nodes[j]+j*n)

I = H.copy()
for node in I.nodes():
  I.nodes[node]['label'] = -1

for i in range(graph_num):
  I.nodes[max_pg_nodes[i]+i*n]['label'] = i

pos = nx.circular_layout(I)
pattern = [ 'gray' if I.nodes[node]['label'] == -1 else 'blue' if node < n else 'green' if node < 2*n else 'orange' if node < 3*n else 'red' for node in I.nodes() ]
nx.draw(I, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
plt.show()

for node in H.nodes():
  if node < n:
    H.nodes[node]['label'] = 0
  elif node < 2*n:
    H.nodes[node]['label'] = 1
  elif node < 3*n:
    H.nodes[node]['label'] = 2
  else:
    H.nodes[node]['label'] = 3

'''
pos = nx.circular_layout(H)
pattern = [ 'blue' if node < n else 'green' if node < 2*n else 'orange' if node < 3*n else 'red' for node in H.nodes() ]
nx.draw(H, pos, node_size=20, alpha=0.5, node_color=pattern, edge_color='gray', with_labels=False)
plt.show()
'''
