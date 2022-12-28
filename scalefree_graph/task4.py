from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import sample2
import torch
import matplotlib.pyplot as plt
import numpy as np

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
      graph_num = 4
      large_graph_num = 2
      a,_,lc,sc = sample2.generate_umblance_sample(n1, n2, m, graph_num, large_graph_num, n1//10)
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

        if 1 - err/len(pred) >= threshold:
          cnt2 = cnt//(n1//n2)
          if cnt2 > n2:
            cnt2 = n2
          guarantee_ratio_list[l].append((cnt*large_graph_num + cnt2*(graph_num-large_graph_num))/(n1*large_graph_num + n2*(graph_num-large_graph_num)))
          guarantee_number_list[l].append(cnt*large_graph_num + cnt2*(graph_num-large_graph_num))
          print(f'Guarantee {threshold*100}% accuracy with {cnt*large_graph_num+cnt2*(graph_num-large_graph_num)} labeled nodes in {n1*large_graph_num+n2*(graph_num-large_graph_num)} nodes')
          break
        elif cnt == n1-1:
          guarantee_ratio_list[l].append(1.0)
          guarantee_number_list[l].append(n1*large_graph_num + n2*(graph_num-large_graph_num))
          print(f'Cannot guarantee {threshold*100}% accuracy with {n1*large_graph_num+n2*(graph_num-large_graph_num)} nodes')

    print(f'========== roop {r+1}/{roop} finished ==========')

  fig = plt.figure()
  fig.suptitle(f'The ratio of labeled nodes to guarantee {threshold*100}% accuracy\n(sum of node is {small_network_nodes_list[0]*4}, calculate mean {roop} samples)')
  ax = fig.add_subplot(111)
  ax.plot(guarantee_ratio_list.keys(), [sum(v)/len(v) for v in guarantee_ratio_list.values()])
  ax.set_xlabel('Nodes of large network has x times of small network')
  ax.set_ylabel('Ratio of labeled nodes')
  ax.grid(axis='y', color='gray', linestyle='dashed')
  fig.savefig(f'./scalefree_graph/task4_figures/task4_mean_ratio_n{small_network_nodes_list[0]*4}_20perLink.png')

  fig2 = plt.figure()
  fig2.suptitle(f'The number of labeled nodes to guarantee {threshold*100}% accuracy\n(sum of node is {small_network_nodes_list[0]*4}, calculate mean {roop} samples)')
  ax2 = fig2.add_subplot(111)
  ax2.plot(guarantee_number_list.keys(), [sum(v)/len(v) for v in guarantee_number_list.values()])
  ax2.set_xlabel('Nodes of large network has x times of small network')
  ax2.set_ylabel('Number of labeled nodes')
  ax2.grid(axis='y', color='gray', linestyle='dashed')
  fig2.savefig(f'./scalefree_graph/task4_figures/task4_mean_number_n{small_network_nodes_list[0]*4}_20perLink.png')

  x = np.array([1,1.5,2,3,4,5,6,7,8,9,10])
  # x = np.array([1,1.5,2,3])
  y = np.array([sum(v)/len(v) for v in guarantee_ratio_list.values()])
  z = np.array([sum(v)/len(v) for v in guarantee_number_list.values()])

  '''
  res1 = np.polyfit(x, y, 1)
  res2 = np.polyfit(x, y, 2)
  res3 = np.polyfit(x, y, 3)

  y1 = np.poly1d(res1)(x)
  y2 = np.poly1d(res2)(x)
  y3 = np.poly1d(res3)(x)

  ret1 = np.polyfit(x, z, 1)
  ret2 = np.polyfit(x, z, 2)
  ret3 = np.polyfit(x, z, 3)

  z1 = np.poly1d(ret1)(x)
  z2 = np.poly1d(ret2)(x)
  z3 = np.poly1d(ret3)(x)

  fig3 = plt.figure()
  fig3.suptitle(f'The ratio of labeled nodes to guarantee {threshold*100}% accuracy\n(sum of node is {small_network_nodes_list[0]*4}, calculate mean {roop} samples)')
  ax3 = fig3.add_subplot(111)
  ax3.scatter(x, y, label='data')
  ax3.plot(x, y1, label='linear')
  ax3.plot(x, y2, label='quadratic')
  ax3.plot(x, y3, label='cubic')
  ax3.legend()
  ax3.set_xlabel('Nodes of large network has x times of small network')
  ax3.set_ylabel('Ratio of labeled nodes')
  ax3.grid(axis='y', color='gray', linestyle='dashed')
  fig3.savefig(f'./scalefree_graph/task4_figures/task4_mean_ratio_n{small_network_nodes_list[0]*4}_fit.png')

  fig4 = plt.figure()
  fig4.suptitle(f'The number of labeled nodes to guarantee {threshold*100}% accuracy\n(sum of node is {small_network_nodes_list[0]*4}, calculate mean {roop} samples)')
  ax4 = fig4.add_subplot(111)
  ax4.scatter(x, z, label='data')
  ax4.plot(x, z1, label='linear')
  ax4.plot(x, z2, label='quadratic')
  ax4.plot(x, z3, label='cubic')
  ax4.legend()
  ax4.set_xlabel('Nodes of large network has x times of small network')
  ax4.set_ylabel('Number of labeled nodes')
  ax4.grid(axis='y', color='gray', linestyle='dashed')
  fig4.savefig(f'./scalefree_graph/task4_figures/task4_mean_number_n{small_network_nodes_list[0]*4}_fit.png')
  '''

main()
