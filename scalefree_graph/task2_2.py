from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import sample3
import torch
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

def main():
  n = 100
  m = 2
  graph_num = 4
  link_level = 50
  a,c = sample3.generate_flexible_linked_sample(n, m, graph_num, link_level)
  dataA = from_networkx(a)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Net(n).to(device)

  acc = {}
  for cnt in range(n):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    epoch_num = 300
    t = dataA.label

    sample = []
    for i in range(cnt):
      for j in c[i]:
        sample.append(j)

    for _ in range(epoch_num):
      optimizer.zero_grad()
      _, out = model(dataA)
      loss = F.nll_loss(out[sample], t[sample])
      loss.backward()
      optimizer.step()

    model.eval()
    _, out = model(dataA)
    pred = out.max(dim=1)[1]
    err = 0
    for i,p in enumerate(pred):
      if p != t[i]:
        err += 1
    print(f"({cnt}x{graph_num} nodes is labeled) Accuracy: {(1 - err / len(pred))*100:.2f}%")
    acc[cnt] = (1 - err / len(pred))

  # print(acc)
  fig = plt.figure()
  fig.suptitle(f'Accuracy and Number of Labeled Nodes(n={n} link={link_level}) per Class')
  ax = fig.add_subplot(111)
  ax.plot(acc.keys(), acc.values())
  ax.set_xlabel('Number of Labeled Nodes per Class')
  ax.set_ylabel('Accuracy')
  ax.grid(axis='y', color='gray', linestyle='--')
  fig.savefig(f'./scalefree_graph/task2_figures/task2_n{n}_link{link_level}.png')

main()
