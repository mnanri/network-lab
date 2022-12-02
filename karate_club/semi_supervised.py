from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import KarateClub

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    node_features = 34
    hidden_size = 4
    classes = 4
    out_dim = 2
    self.conv1 = GCNConv(node_features, hidden_size)
    self.conv2 = GCNConv(hidden_size, classes)
    self.conv3 = GCNConv(classes, out_dim)
    self.conv4 = GCNConv(out_dim, classes)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index
    x = self.conv1(x, edge_index)
    x = torch.tanh(x)
    x = self.conv2(x, edge_index)
    x = torch.tanh(x)
    x = self.conv3(x, edge_index)
    x = torch.tanh(x)
    y = self.conv4(x, edge_index)
    y = F.log_softmax(y, dim=1)
    return x,y

def generate_figure(tmp, t, epoch):
  tmp = tmp.T
  tmp = tmp.detach().numpy()
  fig = plt.figure()
  fig.suptitle(f'Epoch: {epoch+1:03d}(with 4 labeled nodes)')
  ax = fig.add_subplot(111)
  ax.scatter(tmp[0], tmp[1], c=t, alpha=0.5, s=20)
  fig.savefig('./karate_club/figures/epoch{}_1.png'.format(epoch+1))

def generate_model():
  dataset = KarateClub()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Net().to(device)
  model.train()
  data = dataset[0]
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  samples = []
  for i in range(dataset.num_classes):
    for j, k in enumerate(data.y):
      if k == i:
        samples.append(j)
        break

  for epoch in range(300):
    optimizer.zero_grad()
    tmp, out = model(data)
    if (epoch+1) % 50 == 0 or epoch == 0:
      generate_figure(tmp, data.y, epoch)
    loss = F.nll_loss(out[samples], data.y[samples])
    loss.backward()
    optimizer.step()
    # print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

  model.eval()
  _, out = model(data)
  pred = out.max(dim=1)[1]
  err = 0
  for i, p in enumerate(pred):
    if p != data.y[i]:
      err += 1
  print('Accuracy: {:.4f}%'.format((1 - err / len(pred)) * 100))
  acc = (1 - err / len(pred)) * 100
  # return acc

generate_model()
