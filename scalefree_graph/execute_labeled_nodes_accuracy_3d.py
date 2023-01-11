from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import torch
import generate_network

class Net(torch.nn.Module):
  def __init__(self, n):
    super(Net, self).__init__()
    hidden_layer1 = 32
    hidden_layer2 = 8
    output_layer = 3
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

def execute_label_accuracy_3d():
  n = 200
  m = 2
  community = 8
  link = n // 10
  option = 'degree'
  roop = 20
  acc_list = {}
  for i in range(n):
    acc_list[i] = []
  for r in range(roop):
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
        output = model(data)[1]
        loss = F.nll_loss(output[samples], t[samples])
        loss.backward()
        optimizer.step()

      model.eval()
      _, output = model(data)
      pred = output.max(dim=1)[1]
      acc = pred.eq(t).sum().item() / t.size(0)
      acc_list[cnt].append(acc)
      # print(f"roop: {r+1}/{roop}, cnt: {cnt}, acc: {acc}")

    print(f"roop: {r+1}/{roop} finished")

execute_label_accuracy_3d()
