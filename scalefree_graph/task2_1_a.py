from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import sample2
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
  n1 = 180
  n2 = 18
  m = 2
  graph_num = 4
  large_graph_num = 2

  roop = 40
  acc_mean = {}
  for i in range(n1):
    acc_mean[i] = []
  for r in range(roop):
    a,_,lc,sc = sample2.generate_umblance_sample(n1, n2, m, graph_num, large_graph_num)
    dataA = from_networkx(a)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(n1).to(device)

    for cnt in range(n1):
      model.train()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

      epoch_num = 300
      t = dataA.label

      samples = []
      for i in range(cnt):
        for j in lc[i]:
          samples.append(j)

      for i in range(cnt//(n1//n2)):
        if i == n2:
          break
        for j in sc[i]:
          samples.append(j)

      for _ in range(epoch_num):
        optimizer.zero_grad()
        _, y = model(dataA)
        loss = F.nll_loss(y[samples], t[samples])
        loss.backward()
        optimizer.step()

      model.eval()
      _, y = model(dataA)
      pred = y.max(dim=1)[1]
      err = 0
      for i,p in enumerate(pred):
        if p != t[i]:
          err += 1
      acc_mean[cnt].append(1 - err/len(pred))

    print(f"roop {r+1}/{roop} done")

  fig = plt.figure()
  fig.suptitle(f'Accuracy and Number of Labeled Nodes(n1={n1}, n2={n2}) per Class\n(calculated mean of {roop} samples)')
  ax = fig.add_subplot(111)
  ax.plot([i for i in range(n1)], [sum(acc_mean[i])/len(acc_mean[i]) for i in range(n1)])
  ax.set_xlabel(f'Number of Labeled Nodes per Large Class(small class is 1/{n1//n2})')
  ax.set_ylabel('Accuracy')
  ax.grid(axis='y', color='gray', linestyle='--')
  fig.savefig(f'./scalefree_graph/task2_figures/task2_mean_l{n1}_s{n2}.png')

  fig2 = plt.figure()
  fig2.suptitle(f'Accuracy and Number of Labeled Nodes(n1={n1}, n2={n2}) per Class\n(calculated mean of {roop} samples)')
  ax2 = fig2.add_subplot(111)
  ax2.plot([i for i in range((n1+n2)//2)], [sum(acc_mean[i])/len(acc_mean[i]) for i in range((n1+n2)//2)])
  ax2.set_xlabel(f'Number of Labeled Nodes per Large Class(small class is 1/{n1//n2})')
  ax2.set_ylabel('Accuracy')
  ax2.grid(axis='y', color='gray', linestyle='--')
  fig2.savefig(f'./scalefree_graph/task2_figures/task2_mean_l{n1}_s{n2}_another.png')

main()
