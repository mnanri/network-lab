from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import sample
import torch
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
  def __init__(self, n):
    super(Net, self).__init__()
    hidden_layer1 = 16
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

def main():
  max_nodes_per_class = 250
  min_nodes_per_class = 20
  guarantee = {}
  assurance = {}
  threshold = 0.8
  for n in range(min_nodes_per_class, max_nodes_per_class+1, 5):
    m = 2
    graph_num = 8
    # a is full labeled graph, b is partial labeled graph
    a,b,c = sample.generate_sample(n, m, graph_num)
    dataA = from_networkx(a)
    _ = from_networkx(b)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(n).to(device)

    for cnt in range(n):
      model.train()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

      epoch_num = 300
      t = dataA.label

      samples = []
      for i in range(cnt):
        for j in c[i]:
          samples.append(j)
      # for i, d in enumerate(dataB.label):
      #   if d != -1:
      #     samples.append(i)

      for _ in range(epoch_num):
        optimizer.zero_grad()
        _, out = model(dataA)

        # print(out)
        # print(out.shape)

        loss = F.nll_loss(out[samples], dataA.label[samples])
        # loss = F.nll_loss(out, t)
        # print(loss)
        # print(loss.shape)
        loss.backward()
        optimizer.step()
        # print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

      model.eval()
      _, out = model(dataA)
      pred = out.max(dim=1)[1]
      err = 0
      for i, p in enumerate(pred):
        if p != t[i]:
          err += 1
      # print(f"({cnt+1}x{graph_num} nodes of {n}x{graph_num} is labeled) Accuracy: {(1 - err / len(pred))*100:.2f}%")
      if 1 - err / len(pred) >= threshold:
        guarantee[n] = (cnt)/n
        assurance[n] = cnt
        print(f"({cnt}x{graph_num} nodes of {n}x{graph_num} is labeled) Accuracy is guaranteed to be above {threshold*100}%")
        break
      elif cnt == n-1:
        guarantee[n] = 1.0
        assurance[n] = cnt

  fig = plt.figure()
  fig.suptitle('The ratio of labeled nodes to guarantee 0.8 accuracy')
  ax = fig.add_subplot(111)
  ax.plot(guarantee.keys(), guarantee.values())
  ax.set_xlabel('Number of nodes per class')
  ax.set_ylabel('Ratio of labeled nodes')
  ax.grid(axis='y', color='gray', linestyle='--')
  fig.savefig('./scalefree_graph/task3_3d_a.png')

  fig2 = plt.figure()
  fig2.suptitle('The number of labeled nodes to guarantee 0.8 accuracy')
  ax = fig2.add_subplot(111)
  ax.plot(assurance.keys(), assurance.values())
  ax.set_xlabel('Number of nodes per class')
  ax.set_ylabel('Number of labeled nodes')
  ax.grid(axis='y', color='gray', linestyle='--')
  fig2.savefig('./scalefree_graph/task3_3d_b.png')

main()
