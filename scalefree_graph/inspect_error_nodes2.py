import csv
import random
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import torch
import generate_network
import networkx as nx
import matplotlib.pyplot as plt

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

def inspect_error_nodes():
  n = 400
  m = 2
  community = 4
  link = n // 10
  option = 'degree'
  roop = 10
  all_degree = [0]*(n+link)
  error_degree = [0]*(n+link)
  for r  in range(roop):
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

      for _ in range(epoch_num):
        optimizer.zero_grad()
        _, out = model(data)
        loss = F.nll_loss(out[samples], t[samples])
        loss.backward()
        optimizer.step()

      model.eval()
      _, out = model(data)
      pred = out.max(dim=1)[1]
      err = 0
      for i,p in enumerate(pred):
        if p != t[i]:
          err += 1
      if err/len(pred) > 0.2:
        print('Labeled Node:' + str(cnt) + ', Accuracy: ' + str(1 - err/len(pred)))
        continue

      if r == roop-1:
        network_degree = []
        error_nodes = []
        for i,p in enumerate(pred):
          network_degree.append(dict(net.degree())[i])
          all_degree[dict(net.degree())[i]] += (1/roop)
          if p != t[i]:
            error_nodes.append(dict(net.degree())[i])
            error_degree[dict(net.degree())[i]] += (1/roop)

        fig = plt.figure()
        fig.suptitle('Error nodes: Accuracy 80%')
        ax = fig.add_subplot(111)
        ax.set_yscale('log')
        ax.hist(network_degree, bins=range(1, max(network_degree) + 2), align='left', rwidth=0.8, label='all nodes')
        ax.hist(error_nodes, bins=range(1, max(network_degree) + 2), align='left', rwidth=0.8, label='error nodes')
        ax.legend()
        ax.set_xlabel('Degree')
        ax.set_ylabel('Number of nodes')
        ax.grid(axis='y', color='gray', linestyle='dashed')
        fig.savefig(f'./scalefree_graph/error_nodes_{option}_desc.png')

      break

    print(f'============ roop {r+1} finished ============')

  with open(f'./scalefree_graph/error_nodes_{option}_desc.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(all_degree)
    writer.writerow(error_degree)

inspect_error_nodes()
