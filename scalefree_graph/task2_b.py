import csv
import random
import time
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import sample
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
  n = 400
  m = 2
  graph_num = 4
  link_level = n // 10
  roop = 40
  acc_mean = {}
  for i in range(n):
    acc_mean[i] = []
  duration = []
  for r in range(roop):
    start = time.time()
    a,_,c = sample.generate_sample(n, m, graph_num, link_level)
    dataA = from_networkx(a)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(n).to(device)

    for cnt in range(n):
      model.train()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

      epoch_num = 300
      t = dataA.label

      samples = []
      # for i in range(cnt):
      #   for j in c[i]:
      #     samples.append(j)

      # for i in range(cnt):
      #   for j in c[n-1-i]:
      #     samples.append(j)

      tmp = [i for i in range(n)]
      random.shuffle(tmp)
      for i in range(cnt):
        for j in c[tmp[i]]:
          samples.append(j)

      for _ in range(epoch_num):
        optimizer.zero_grad()
        _, out = model(dataA)
        loss = F.nll_loss(out[samples], t[samples])
        loss.backward()
        optimizer.step()

      model.eval()
      _, out = model(dataA)
      pred = out.max(dim=1)[1]
      err = 0
      for i,p in enumerate(pred):
        if p != t[i]:
          err += 1
      acc_mean[cnt].append(1 - err/len(pred))

    end = time.time()
    duration.append(end - start)
    print(f'Duration: {end - start} sec')
    print(f'roop {r+1}/{roop} done')

  fig = plt.figure()
  fig.suptitle(f'Accuracy and Number of Labeled Nodes(n={n}) per Class\n({link_level/n*100}% nodes is used in link, claculated mean of {roop} samples)')
  ax = fig.add_subplot(111)
  ax.plot([i for i in range(n)], [sum(acc_mean[i])/len(acc_mean[i]) for i in range(n)])
  ax.set_xlabel('Number of Labeled Nodes per Class')
  ax.set_ylabel('Accuracy')
  ax.grid(axis='x', color='gray', linestyle='--')
  ax.grid(axis='y', color='gray', linestyle='--')
  fig.savefig(f'./scalefree_graph/task2_figures/task2_mean_n{n}_10perLink_dg_rand.png')

  with open(f'./scalefree_graph/task2_data/task2_n{n}_10perLink_{roop}samples_dg_rand.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow([i for i in range(n)])
    writer.writerow([sum(acc_mean[i])/len(acc_mean[i]) for i in range(n)])

  print(f'Average Duration: {sum(duration)/len(duration)}')

main()
