import random
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import generate_network
import torch
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
  def __init__(self,n):
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

def generate_figure(tmp, t, epoch):
  tmp = tmp.T
  tmp = tmp.detach().numpy()
  fig = plt.figure()
  fig.suptitle(f'Epoch: {epoch+1:03d}(with n labeled nodes per class)')
  ax = fig.add_subplot(projection='3d')
  ax.scatter(tmp[0], tmp[1], tmp[2], c=t, alpha=0.5, s=20)
  fig.savefig('./scalefree_graph/task_figures/n100_c8_50perLabeled/epoch{}.png'.format(epoch+1))

def main():
  n = 100
  m = 2
  community = 8
  net, list = generate_network.generate_network(n, m, community, n//10)
  data = from_networkx(net)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Net(n).to(device)
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

  epoch_num = 1
  t = data.label

  samples = []
  tmp = [i for i in range(n)]
  random.shuffle(tmp)
  for i in range(n//2):
    for j in list[tmp[i]]:
      samples.append(j)

  for epoch in range(epoch_num):
    optimizer.zero_grad()
    tmp, out = model(data)

    if (epoch+1)%50 == 0 or epoch == 0:
      generate_figure(tmp, t, epoch)

    loss = F.nll_loss(out, t)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

  model.eval()
  _, out = model(data)
  pred = out.max(dim=1)[1]
  err = 0
  for i, p in enumerate(pred):
    if p != t[i]:
      err += 1
  print(f"Accuracy: {(1 - err / len(pred))*100:.2f}%")

main()
