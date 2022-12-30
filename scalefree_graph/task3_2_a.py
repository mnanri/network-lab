import csv
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import sample3
import sample
import torch
import matplotlib.pyplot as plt
import numpy as np

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
  n = 400
  m = 2
  graph_num = 4
  max_link_level = n
  min_link_level = 0
  roop = 10

  x_axis = {}
  guarantee_ratio_list = {}
  guarantee_number_list = {}
  for i in range(min_link_level, max_link_level+1, n // 40):
    x_axis[i] = -1
    guarantee_ratio_list[i] = []
    guarantee_number_list[i] = []

  threshold = 0.8

  for r in range(roop):
    for link_level in range(min_link_level, max_link_level+1, n // 40):
      # a,c = sample3.generate_flexible_linked_sample(n, m, graph_num, link_level)
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

        for epoch in range(epoch_num):
          optimizer.zero_grad()
          _, y = model(dataA)
          loss = F.nll_loss(y[samples], t[samples])
          loss.backward()
          optimizer.step()
          # print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

        model.eval()
        _, y = model(dataA)
        pred = y.max(dim=1)[1]
        err = 0
        for i,p in enumerate(pred):
          if p != t[i]:
            err += 1
        if 1 - err/len(pred) >= threshold:
          guarantee_ratio_list[link_level].append(cnt/n)
          guarantee_number_list[link_level].append(cnt*graph_num)
          print(f"({cnt}x{graph_num} nodes of {n}x{graph_num} is labeled) Accuracy is guaranteed to be above {threshold*100}%")
          break
        elif cnt == n-1:
          guarantee_ratio_list[link_level].append(1)
          guarantee_number_list[link_level].append(n*graph_num)
          print(f"unexpected: cannot guarantee accuracy for {n}x{graph_num} nodes")

      x_axis[link_level] = round(link_level/n, 2)

    with open(f'./scalefree_graph/task3_data/task3_link_mean_n{n}_ta.csv', 'w') as f:
      writer = csv.writer(f)
      writer.writerow(x_axis.values())
      writer.writerow([sum(v)/len(v) for v in guarantee_ratio_list.values()])

    with open(f'./scalefree_graph/task3_data/task3_link_med_n{n}_ta.csv', 'w') as f:
      writer = csv.writer(f)
      writer.writerow(x_axis.values())
      writer.writerow(np.median([v for v in guarantee_ratio_list.values()], axis=1))

    with open(f'./scalefree_graph/task3_data/task3_link_mean_n{n}_tb.csv', 'w') as f:
      writer = csv.writer(f)
      writer.writerow(x_axis.values())
      writer.writerow([sum(v)/len(v) for v in guarantee_number_list.values()])

    with open(f'./scalefree_graph/task3_data/task3_link_med_n{n}_tb.csv', 'w') as f:
      writer = csv.writer(f)
      writer.writerow(x_axis.values())
      writer.writerow(np.median([v for v in guarantee_number_list.values()], axis=1))

    print(f"=============== roop {r+1}/{roop} is finished ===============")

  fig = plt.figure()
  fig.suptitle(f'The Ratio of Labeled Nodes to Guarantee Accuracy {threshold*100}% in n={n}\n(calculate mean of {roop} samples)')
  ax = fig.add_subplot(111)
  ax.plot(x_axis.values(), [sum(v)/len(v) for v in guarantee_ratio_list.values()])
  ax.set_xlabel('Number of Node used in Probabilistic Link per 2 Classes')
  ax.set_ylabel('Ratio of Labeled Nodes')
  ax.grid(axis='x', color='gray', linestyle='--')
  ax.grid(axis='y', color='gray', linestyle='--')
  fig.savefig(f'./scalefree_graph/task3_figures/task3_link_mean_n{n}_a.png')

  '''
  fig1 = plt.figure()
  fig1.suptitle(f'The Ratio of Labeled Nodes to Guarantee Accuracy {threshold*100}% in n={n}\n(calculate median of {roop} samples)')
  ax1 = fig1.add_subplot(111)
  ax1.plot(x_axis.values(), np.median([v for v in guarantee_ratio_list.values()], axis=1))
  ax1.set_xlabel('Number of Node used in Probabilistic Link per 2 Classes')
  ax1.set_ylabel('Ratio of Labeled Nodes')
  ax1.grid(axis='x', color='gray', linestyle='--')
  ax1.grid(axis='y', color='gray', linestyle='--')
  fig1.savefig(f'./scalefree_graph/task3_figures/task3_link_med_n{n}_a.png')
  '''

  fig2 = plt.figure()
  fig2.suptitle(f'The Number of Labeled Nodes to Guarantee Accuracy {threshold*100}% in n={n}\n(calculate mean of {roop} samples)')
  ax2 = fig2.add_subplot(111)
  ax2.plot(x_axis.values(), [sum(v)/len(v) for v in guarantee_number_list.values()])
  ax2.set_xlabel('Number of Node used in Probabilistic Link per 2 Classes')
  ax2.set_ylabel('Number of labeled nodes')
  ax2.grid(axis='x', color='gray', linestyle='--')
  ax2.grid(axis='y', color='gray', linestyle='--')
  fig2.savefig(f'./scalefree_graph/task3_figures/task3_link_mean_n{n}_b.png')

  '''
  fig3 = plt.figure()
  fig3.suptitle(f'The Number of Labeled Nodes to Guarantee Accuracy {threshold*100}% in n={n}\n(calculate median of {roop} samples)')
  ax3 = fig3.add_subplot(111)
  ax3.plot(x_axis.values(), np.median([v for v in guarantee_number_list.values()], axis=1))
  ax3.set_xlabel('Number of Node used in Probabilistic Link per 2 Classes')
  ax3.set_ylabel('Number of labeled nodes')
  ax3.grid(axis='x', color='gray', linestyle='--')
  ax3.grid(axis='y', color='gray', linestyle='--')
  fig3.savefig(f'./scalefree_graph/task3_figures/task3_link_med_n{n}_b.png')
  '''

  with open(f'./scalefree_graph/task3_data/task3_link_mean_n{n}_fa.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(x_axis.values())
    writer.writerow([sum(v)/len(v) for v in guarantee_ratio_list.values()])

  with open(f'./scalefree_graph/task3_data/task3_link_med_n{n}_fa.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(x_axis.values())
    writer.writerow(np.median([v for v in guarantee_ratio_list.values()], axis=1))

  with open(f'./scalefree_graph/task3_data/task3_link_mean_n{n}_fb.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(x_axis.values())
    writer.writerow([sum(v)/len(v) for v in guarantee_number_list.values()])

  with open(f'./scalefree_graph/task3_data/task3_link_med_n{n}_fsb.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(x_axis.values())
    writer.writerow(np.median([v for v in guarantee_number_list.values()], axis=1))

main()
