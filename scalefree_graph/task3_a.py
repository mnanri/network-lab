import csv
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import sample
import torch
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
  def __init__(self, n):
    super(Net, self).__init__()
    hidden_layer1 = 16
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

def main():
  max_nodes_per_class = 450
  min_nodes_per_class = 50
  roop = 20

  guarantee_ratio_list = {}
  guarantee_number_list = {}
  for i in range(min_nodes_per_class, max_nodes_per_class+1, 10):
    guarantee_ratio_list[i] = []
    guarantee_number_list[i] = []

  threshold = 0.8

  for r in range(roop):
    for n in range(min_nodes_per_class, max_nodes_per_class+1, 10):
      m = 2
      graph_num = 4
      link_level = n // 10
      a,_,c = sample.generate_sample(n, m, graph_num, link_level)
      dataA = from_networkx(a)

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model = Net(n).to(device)

      for cnt in range(n):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        epoch_num = 300
        t = dataA.label

        samples = []
        for i in range(cnt):
          for j in c[i]:
            samples.append(j)

        for _ in range(epoch_num):
          optimizer.zero_grad()
          _, out = model(dataA)
          loss = F.nll_loss(out[samples], dataA.label[samples])
          loss.backward()
          optimizer.step()

        model.eval()
        _, out = model(dataA)
        pred = out.max(dim=1)[1]
        err = 0
        for i, p in enumerate(pred):
          if p != t[i]:
            err += 1
        if 1 - err / len(pred) >= threshold:
          guarantee_ratio_list[n].append((cnt)/n)
          guarantee_number_list[n].append(cnt*graph_num)
          # print(f"({cnt}x{graph_num} nodes of {n}x{graph_num} is labeled) Accuracy is guaranteed to be above {threshold*100}%")
          break
        elif cnt == n-1:
          guarantee_ratio_list[n].append(1.0)
          guarantee_number_list[n].append((cnt+1)*graph_num)
          print(f"unexpected: cannot guarantee accuracy for {n}x{graph_num} nodes")

    print(f"roop {r+1}/{roop} finished")

  fig = plt.figure()
  fig.suptitle(f'The Ratio of Labeled Nodes to Guarantee Accuracy {threshold*100}%\n(20% nodes is used in links, calculate mean of {roop} samples)')
  ax = fig.add_subplot(111)
  ax.plot(guarantee_ratio_list.keys(), [sum(v)/len(v) for v in guarantee_ratio_list.values()])
  ax.set_xlabel('Number of Nodes per Class')
  ax.set_ylabel('Ratio of Labeled Nodes')
  ax.grid(axis='y', color='gray', linestyle='--')
  fig.savefig('./scalefree_graph/task3_figures/task3_mean_10perLink_a.png')

  fig2 = plt.figure()
  fig2.suptitle(f'The Number of Labeled Nodes to Guarantee Accuracy {threshold*100}%\n(20% nodes is used in links, calculate mean of {roop} samples)')
  ax2 = fig2.add_subplot(111)
  ax2.plot(guarantee_number_list.keys(), [sum(v)/len(v) for v in guarantee_number_list.values()])
  ax2.set_xlabel('Number of nodes per class')
  ax2.set_ylabel('Number of labeled nodes')
  ax2.grid(axis='y', color='gray', linestyle='--')
  fig2.savefig('./scalefree_graph/task3_figures/task3_mean_10perLink_b.png')

  with open(f'./scalefree_graph/task3_data/task3_mean_10perLink_a.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(guarantee_ratio_list.keys())
    writer.writerow([sum(v)/len(v) for v in guarantee_ratio_list.values()])

  with open(f'./scalefree_graph/task3_data/task3_med_10perLink_a.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(guarantee_ratio_list.keys())
    writer.writerow(np.median([v for v in guarantee_ratio_list.values()], axis=1))

  with open(f'./scalefree_graph/task3_data/task3_mean_10perLink_b.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(guarantee_number_list.keys())
    writer.writerow([sum(v)/len(v) for v in guarantee_number_list.values()])

  with open(f'./scalefree_graph/task3_data/task3_med_10perLink_b.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(guarantee_number_list.keys())
    writer.writerow(np.median([v for v in guarantee_number_list.values()], axis=1))

main()
