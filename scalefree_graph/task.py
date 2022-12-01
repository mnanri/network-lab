import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import sample
import torch
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    n = 100
    hidden_layer1 = 34
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

def generate_figure(tmp, t, epoch):
  tmp = tmp.T
  tmp = tmp.detach().numpy()
  # print(tmp)
  # print(tmp.shape)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(tmp[0], tmp[1], c=t, alpha=0.5, s=20)
  fig.savefig('./scalefree_graph/figures/epoch{}.png'.format(epoch+1))

def main():
  n = 100
  # a is full labeled graph, b is partial labeled graph
  a,b = sample.generate_sample(n)
  dataA = from_networkx(a)
  _ = from_networkx(b)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Net().to(device)
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

  epoch_num = 300
  t = dataA.label
  for epoch in range(epoch_num):
    optimizer.zero_grad()
    tmp, out = model(dataA)
    # print(out)
    # print(out.shape)
    if (epoch+1)%50 == 0:
      generate_figure(tmp, t, epoch)

    out = F.log_softmax(out, dim=1)
    loss = F.cross_entropy(out, t)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

main()
