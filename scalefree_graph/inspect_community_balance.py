from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import torch
import generate_old_network

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

def inspect_community_balance():
  small_network_nodes_list = [400, 320, 264, 200, 160, 132, 112, 100, 88, 80, 72]
  # small_network_nodes_list = [100, 80, 66, 50]
  roop = 40

  guarantee_ratio_list = {}
  guarantee_number_list = {}
  for i in range(len(small_network_nodes_list)):
    l = i
    if l == 0:
      l = 1
    elif l == 1:
      l = 1.5
    guarantee_ratio_list[l] = []
    guarantee_number_list[l] = []

  threshold = 0.8

  for r in range(roop):
    for k, n2 in enumerate(small_network_nodes_list):
      l = k
      if l == 0:
        l = 1
      elif l == 1:
        l = 1.5
      n1 = int(n2*l)
      m = 2
      community = 4
      large_graph_num = 2
      net, large_list, small_list = generate_old_network.generate_old_network(n1, n2, m, community, large_graph_num)
      data = from_networkx(net)

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model = Net(n1).to(device)

      for cnt in range(n1):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        epoch_num = 300
        t = data.label

        samples = []
        for i in range(cnt):
          for j in large_list[i]:
            samples.append(j)
        for i in range(cnt//(n1//n2)):
          if i == n2:
            break
          for j in small_list[i]:
            samples.append(j)

        for _ in range(epoch_num):
          optimizer.zero_grad()
          _, y = model(data)
          loss = F.nll_loss(y[samples], t[samples])
          loss.backward()
          optimizer.step()

        model.eval()
        _, y = model(data)
        pred = y.max(dim=1)[1]
        err = 0
        for i,p in enumerate(pred):
          if p != t[i]:
            err += 1

        if 1 - err/len(pred) >= threshold:
          cnt2 = cnt//(n1//n2)
          if cnt2 > n2:
            cnt2 = n2
          guarantee_ratio_list[l].append((cnt*large_graph_num + cnt2*(community-large_graph_num))/(n1*large_graph_num + n2*(community-large_graph_num)))
          guarantee_number_list[l].append(cnt*large_graph_num + cnt2*(community-large_graph_num))
          print(f'Guarantee {threshold*100}% accuracy with {cnt*large_graph_num+cnt2*(community-large_graph_num)} labeled nodes in {n1*large_graph_num+n2*(community-large_graph_num)} nodes')
          break
        elif cnt == n1-1:
          guarantee_ratio_list[l].append(1.0)
          guarantee_number_list[l].append(n1*large_graph_num + n2*(community-large_graph_num))
          print(f'Cannot guarantee {threshold*100}% accuracy with {n1*large_graph_num+n2*(community-large_graph_num)} nodes')

    print(f'========== roop {r+1}/{roop} finished ==========')

inspect_community_balance()
