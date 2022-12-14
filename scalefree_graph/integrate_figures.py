import csv
from matplotlib import pyplot as plt

def integarte_figures2(tk=2, n=400, links=10, samples=20, tp=""):
  if tp != "":
    tp = f"_{tp}"

  fig = plt.figure()
  fig.suptitle(f"DESC with Condition: n={n}")
  ax = fig.add_subplot(111)

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_n{n}_{links}perLink_{samples*2}samples_rand.csv', 'r') as f:
    reader = csv.reader(f)
    r = [row for row in reader]
  ax.plot([float(r[0][i]) for i in range(n)], [float(r[1][i]) for i in range(n)], label="Random")

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_n{n}_{links}perLink_{samples}samples_dg{tp}.csv', 'r') as f:
    reader = csv.reader(f)
    dg = [row for row in reader]
    # print(dg)
  ax.plot([float(dg[0][i]) for i in range(n)], [float(dg[1][i]) for i in range(n)], label="Degree")

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_n{n}_{links}perLink_{samples}samples_cc{tp}.csv', 'r') as f:
    reader = csv.reader(f)
    cc = [row for row in reader]
  ax.plot([float(cc[0][i]) for i in range(n)], [float(cc[1][i]) for i in range(n)], label="Closeness")

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_n{n}_{links}perLink_{samples}samples_bc{tp}.csv', 'r') as f:
    reader = csv.reader(f)
    bc = [row for row in reader]
  ax.plot([float(bc[0][i]) for i in range(n)], [float(bc[1][i]) for i in range(n)], label="Betweenness")

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_n{n}_{links}perLink_{samples}samples_pr{tp}.csv', 'r') as f:
    reader = csv.reader(f)
    pr = [row for row in reader]
  ax.plot([float(pr[0][i]) for i in range(n)], [float(pr[1][i]) for i in range(n)], label="PageRank")

  ax.legend()
  ax.set_xlabel('Number of Labeled Nodes per Class')
  ax.set_ylabel('Accuracy')
  ax.grid(axis='x', color='gray', linestyle='--')
  ax.grid(axis='y', color='gray', linestyle='--')
  fig.savefig(f'./scalefree_graph/task{tk}_data/task{tk}_n{n}_{links}perLink_{samples}samples{tp}.png')

def integrate_figures3(tk=3, st="mean", links=10, op="a"):
  title = ""
  if op == "a":
    title = "Ratio"
  else :
    title = "Number"

  fig = plt.figure()
  fig.suptitle(f"{title}: Labeled Nodes to Reach 80% Accuracy")
  ax = fig.add_subplot(111)

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_{st}_{links}perLink_{op}_rand.csv', 'r') as f:
    reader = csv.reader(f)
    r = [row for row in reader]
  ax.plot([float(r[0][i]) for i in range(41)], [float(r[1][i]) for i in range(41)], label="(a)")

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_{st}_{links}perLink_{op}_bc.csv', 'r') as f:
    reader = csv.reader(f)
    bc = [row for row in reader]
  ax.plot([float(bc[0][i]) for i in range(41)], [float(bc[1][i]) for i in range(41)], label="(d)")

  with open(f'./scalefree_graph/task{tk}_data/task{tk}_{st}_{links}perLink_{op}_dg.csv', 'r') as f:
    reader = csv.reader(f)
    dg = [row for row in reader]
  ax.plot([float(dg[0][i]) for i in range(41)], [float(dg[1][i]) for i in range(41)], label="(j)")

  ax.legend()
  ax.set_xlabel('Number of Nodes per Class')
  ax.set_ylabel('Ratio of Labeled Nodes')
  ax.grid(axis='x', color='gray', linestyle='--')
  ax.grid(axis='y', color='gray', linestyle='--')
  fig.savefig(f'./scalefree_graph/task{tk}_data/task{tk}_{st}_{links}perLink_{op}.png')

# integarte_figures2(2, 400, 10, 20, "com")
# integrate_figures3(3, "mean", 10, "a")
