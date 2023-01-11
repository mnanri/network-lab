import csv
import random
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import torch
import generate_network

class Net(torch.nn.Module):
  def __init__(self, n):
    super(Net, self).__init__()
    hidden_layer1 = 32
    hidden_layer2 = 4
    output_layer = 2
    self.conv1 = GCNConv(n, hidden_layer1)
    self.conv2 = GCNConv(hidden_layer1, hidden_layer2)
    self.conv3 = GCNConv(hidden_layer2, output_layer)
    self.conv4 = GCNConv(output_layer, hidden_layer2)


  def forward(self, data):
    x, edge_index = data.feature, data.edge_index
    x = self.conv1(x, edge_index)
    x = torch.tanh(x)
    x = self.conv2(x, edge_index)
    x = torch.tanh(x)
    x = self.conv3(x, edge_index)
    x = torch.tanh(x)
    y = self.conv4(x, edge_index)
    y = F.log_softmax(y, dim=1)
    return x, y

def execute_nodes_labeled_ratio():
  max_nodes_per_class = 450
  min_nodes_per_class = 50
  roop = 10

  guarantee_ratio_list = {}
  guarantee_number_list = {}
  for i in range(min_nodes_per_class, max_nodes_per_class+1, 10):
    guarantee_ratio_list[i] = []
    guarantee_number_list[i] = []

  threshold = 0.8

  for r in range(roop):
    for n in range(min_nodes_per_class, max_nodes_per_class+1, 10):
      m = 2
      community = 4
      link = n // 10
      option = 'degree'
      net, list = generate_network.generate_network(n, m, community, link, option)
      data = from_networkx(net)

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model = Net(n).to(device)

      for cnt in range(n):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        epoch_num = 300
        t = data.label

        samples = []
        tmp = [i for i in range(n)]
        random.shuffle(tmp)
        for i in range(cnt):
          for j in list[tmp[i]]:
            samples.append(j)

        # for i in range(cnt):
        #   for j in list[i]:
        #     samples.append(j)

        # for i in range(cnt):
        #   for j in list[(i+link)%n]:
        #     samples.append(j)

        for _ in range(epoch_num):
          optimizer.zero_grad()
          _, out = model(data)
          loss = F.nll_loss(out[samples], data.label[samples])
          loss.backward()
          optimizer.step()

        model.eval()
        _, out = model(data)
        pred = out.max(dim=1)[1]
        err = 0
        for i, p in enumerate(pred):
          if p != t[i]:
            err += 1
        if 1 - err / len(pred) >= threshold:
          guarantee_ratio_list[n].append((cnt)/n)
          guarantee_number_list[n].append(cnt*community)
          # print(f"({cnt}x{graph_num} nodes of {n}x{graph_num} is labeled) Accuracy is guaranteed to be above {threshold*100}%")
          break
        elif cnt == n-1:
          guarantee_ratio_list[n].append(1.0)
          guarantee_number_list[n].append((cnt+1)*community)
          print(f"unexpected: cannot guarantee accuracy for {n}x{community} nodes")

    print(f"roop {r+1}/{roop} finished")

  with open(f'./scalefree_graph/task3_data/10lnk_{option}_mean_a.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(guarantee_ratio_list.keys())
    writer.writerow([sum(v)/len(v) for v in guarantee_ratio_list.values()])

  with open(f'./scalefree_graph/task3_data/10lnk_{option}_median_a.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(guarantee_ratio_list.keys())
    writer.writerow(np.median([v for v in guarantee_ratio_list.values()], axis=1))

  with open(f'./scalefree_graph/task3_data/10lnk_{option}_mean_b.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(guarantee_number_list.keys())
    writer.writerow([sum(v)/len(v) for v in guarantee_number_list.values()])

  with open(f'./scalefree_graph/task3_data/10lnk_{option}_median_b.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(guarantee_number_list.keys())
    writer.writerow(np.median([v for v in guarantee_number_list.values()], axis=1))

execute_nodes_labeled_ratio()
