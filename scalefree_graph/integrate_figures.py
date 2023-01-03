from turtle import pd

from matplotlib import pyplot as plt


def show_integarte_figures(tk=2, n=400, links=10, samples=20, tp=""):
  if tp != "":
    tp = f"_{tp}"

  fig = plt.figure()
  fig.suptitle(f"Degree Centrality :n={n}, {links}% nodes is used in link, {samples}samples")
  ax = fig.add_subplot(111)

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_n{n}_{links}perLink_{samples}samples_dg{tp}.csv', 'r') as f:
    dg = f.read().splitlines()
  ax.plot([i for i in range(n)], [float(dg[i]) for i in range(n)], label="Degree")

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_n{n}_{links}perLink_{samples}samples_cc{tp}.csv', 'r') as f:
    cc = f.read().splitlines()
  ax.plot([i for i in range(n)], [float(cc[i]) for i in range(n)], label="Closeness")

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_n{n}_{links}perLink_{samples}samples_bc{tp}.csv', 'r') as f:
    bc = f.read().splitlines()
  ax.plot([i for i in range(n)], [float(bc[i]) for i in range(n)], label="Betweenness")

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_n{n}_{links}perLink_{samples}samples_pr{tp}.csv', 'r') as f:
    pr = f.read().splitlines()
  ax.plot([i for i in range(n)], [float(pr[i]) for i in range(n)], label="PageRank")

  ax.legend()
  ax.set_xlabel('Number of Labeled Nodes per Class')
  ax.set_ylabel('Accuracy')
  ax.grid(axis='x', color='gray', linestyle='--')
  ax.grid(axis='y', color='gray', linestyle='--')
  fig.savefig(f'./scalefree_graph/task{tk}_data/task{tk}_n{n}_{links}perLink_{samples}samples_{tp}.png')

show_integarte_figures()
