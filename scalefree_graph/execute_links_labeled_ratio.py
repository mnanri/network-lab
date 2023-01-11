import csv
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import torch
import numpy as np
import generate_network

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

def execute_links_labeled_ratio():
  n = 400
  m = 2
  community = 4
  max_link = n
  min_link = 0
  option = 'degree'
  roop = 10

  x_axis = {}
  guarantee_ratio_list = {}
  guarantee_number_list = {}
  for i in range(min_link, max_link+1, n // 40):
    x_axis[i] = -1
    guarantee_ratio_list[i] = []
    guarantee_number_list[i] = []

  threshold = 0.8

  for r in range(roop):
    for link in range(min_link, max_link+1, n // 40):
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
        for i in range(cnt):
          for j in list[i]:
            samples.append(j)

        for epoch in range(epoch_num):
          optimizer.zero_grad()
          _, y = model(data)
          loss = F.nll_loss(y[samples], t[samples])
          loss.backward()
          optimizer.step()
          # print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

        model.eval()
        _, y = model(data)
        pred = y.max(dim=1)[1]
        err = 0
        for i,p in enumerate(pred):
          if p != t[i]:
            err += 1
        if 1 - err/len(pred) >= threshold:
          guarantee_ratio_list[link].append(cnt/n)
          guarantee_number_list[link].append(cnt*community)
          print(f"({cnt}x{community} nodes of {n}x{community} is labeled) Accuracy is guaranteed to be above {threshold*100}%")
          break
        elif cnt == n-1:
          guarantee_ratio_list[link].append(1)
          guarantee_number_list[link].append(n*community)
          print(f"unexpected: cannot guarantee accuracy for {n}x{community} nodes")

      x_axis[link] = round(link/n, 2)

    with open(f'./scalefree_graph/task3_data/n{n}_{option}_link_mean_a.csv', 'w') as f:
      writer = csv.writer(f)
      writer.writerow(x_axis.values())
      writer.writerow([sum(v)/len(v) for v in guarantee_ratio_list.values()])

    with open(f'./scalefree_graph/task3_data/n{n}_{option}_link_median_a.csv', 'w') as f:
      writer = csv.writer(f)
      writer.writerow(x_axis.values())
      writer.writerow(np.median([v for v in guarantee_ratio_list.values()], axis=1))

    with open(f'./scalefree_graph/task3_data/n{n}_{option}_link_mean_b.csv', 'w') as f:
      writer = csv.writer(f)
      writer.writerow(x_axis.values())
      writer.writerow([sum(v)/len(v) for v in guarantee_number_list.values()])

    with open(f'./scalefree_graph/task3_data/n{n}_{option}_link_median_b.csv', 'w') as f:
      writer = csv.writer(f)
      writer.writerow(x_axis.values())
      writer.writerow(np.median([v for v in guarantee_number_list.values()], axis=1))

    print(f"=============== roop {r+1}/{roop} is finished ===============")

execute_links_labeled_ratio()
