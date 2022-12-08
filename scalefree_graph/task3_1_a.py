from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import sample2
import torch
import matplotlib.pyplot as plt

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

def main():
  max_n2_nodes_per_class = 101
  min_n2_nodes_per_class = 20
  roop = 20

  guarantee_ratio_list = {}
  guarantee_number_list = {}
  for n2 in range(min_n2_nodes_per_class, max_n2_nodes_per_class+1, 3):
    guarantee_ratio_list[n2] = []
    guarantee_number_list[n2] = []

  threshold = 0.8

  for r in range(roop):
    for n2 in range(min_n2_nodes_per_class, max_n2_nodes_per_class+1, 3):
      n1 = n2 * 4
      m = 2
      graph_num = 4
      large_graph_num = 2
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
          for j in sc[i]:
            samples.append(j)

        for _ in range(epoch_num):
          optimizer.zero_grad()
          _, y = model(dataA)
          loss = F.nll_loss(y[samples], t[samples])
          loss.backward()
          optimizer.step()

        model.eval()
        _, out = model(dataA)
        pred = out.max(dim=1)[1]
        err = 0
        for i,p in enumerate(pred):
          if p != t[i]:
            err += 1

        if 1 - err / len(pred) >= threshold:
          guarantee_ratio_list[n2].append((cnt*large_graph_num+((cnt//(n1//n2))*(graph_num-large_graph_num)))/(n1*large_graph_num+n2*(graph_num-large_graph_num)))
          guarantee_number_list[n2].append(cnt*large_graph_num+((cnt//(n1//n2))*(graph_num-large_graph_num)))
          print(f'Guarantee {threshold} accuracy with {cnt*large_graph_num+((cnt//(n1//n2))*(graph_num-large_graph_num))} labeled nodes in {n1*large_graph_num+n2*(graph_num-large_graph_num)} nodes')
          break
        elif cnt == n1-1:
          guarantee_ratio_list[n2].append(1.0)
          guarantee_number_list[n2].append(n1*large_graph_num+n2*(graph_num-large_graph_num))
          print(f"unexpected: cannot guarantee accuracy {threshold*100}%")

    print(f'========== roop {r+1}/{roop} finished ==========')

  fig = plt.figure()
  fig.suptitle(f'The ratio of labeled nodes to guarantee {threshold*100}% accuracy\n(calculate mean of {roop} samples)')
  ax = fig.add_subplot(111)
  ax.plot(guarantee_ratio_list.keys(), [sum(v)/len(v) for v in guarantee_ratio_list.values()])
  ax.set_xlabel(f'Number of nodes per small class (large class has {n1//n2}times nodes)')
  ax.set_ylabel('Ratio of labeled nodes')
  ax.grid(axis='y', color='gray', linestyle='--')
  fig.savefig(f'./scalefree_graph/task3_figures/task3_mean_{n1//n2}times_a.png')

  fig2 = plt.figure()
  fig2.suptitle(f'The number of labeled nodes to guarantee {threshold*100}% accuracy\n(calculate mean of {roop} samples)')
  ax2 = fig2.add_subplot(111)
  ax2.plot(guarantee_number_list.keys(), [sum(v)/len(v) for v in guarantee_number_list.values()])
  ax2.set_xlabel(f'Number of nodes per small class (large class has {n1//n2}times nodes)')
  ax2.set_ylabel('Number of labeled nodes')
  ax2.grid(axis='y', color='gray', linestyle='--')
  fig2.savefig(f'./scalefree_graph/task3_figures/task3_mean_{n1//n2}times_b.png')

main()
