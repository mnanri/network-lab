from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import torch
import generate_old_network

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

def execute_label_accuracy_umbalance():
  n1 = 180
  n2 = 18
  m = 2
  community = 4
  large_graph_num = 2

  roop = 40
  acc_mean = {}
  for i in range(n1):
    acc_mean[i] = []
  for r in range(roop):
    net, large_list, small_list = generate_old_network.generate_umbalance_network(n1, n2, m, community, large_graph_num)
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
      acc_mean[cnt].append(1 - err/len(pred))

    print(f"roop {r+1}/{roop} done")

execute_label_accuracy_umbalance()
