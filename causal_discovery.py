from data_processing import load_data, get_datasets
from patsy.highlevel import dmatrices
import patsy
import cdt
import numpy as np
import networkx as nx
import pprint
import pandas as pd


ignore_na = patsy.missing.NAAction(NA_types=[])


WINDOW_SIZE = 1

df = load_data()
datasets = get_datasets(df)

reg_eq = "case_count ~ " + " + ".join(list(df.columns[2:].values))

data = datasets["alpha"]

train_size = 0.75  # percentage of data for training only
train_idx = int(len(data) * train_size)
train_data = data.iloc[:train_idx].reset_index()

y, X = dmatrices(reg_eq, data=train_data, return_type="dataframe",
                 NA_action=ignore_na)


normalized_df = (train_data - train_data.mean()) / train_data.std()
normalized_df.drop(columns=[c for c in normalized_df.columns if any(np.isnan(normalized_df[c]))], inplace=True)
normalized_df.drop(columns=["week", "death_count", "year"], inplace=True)
print(normalized_df.columns)

copies = []
col_names = []
upper_base = len(normalized_df) - WINDOW_SIZE
for i in range(WINDOW_SIZE + 1):
    copies.append(normalized_df.iloc[i:upper_base+i].copy().reset_index(drop=True))
    col_names.extend([f"{c}_{WINDOW_SIZE - i}" for c in copies[-1].columns])

normalized_df = pd.concat(copies, axis=1, ignore_index=True)
normalized_df.columns = col_names


glasso = cdt.independence.graph.Glasso()
skeleton = glasso.predict(normalized_df)
# print(nx.adjacency_matrix(skeleton).todense())
new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg='aracne')
# print(nx.adjacency_matrix(new_skeleton).todense())


model = cdt.causality.graph.GES()
output_graph = model.predict(normalized_df, new_skeleton)

tiers = [[("case_count_0", "root_0")]]
print(output_graph.edges)
used_edges = set()
while len(tiers[-1]):
    print(tiers[-1])
    prev = [x for x, _ in tiers[-1]]
    new_tier = {e for e in output_graph.edges if e[1] in prev} - used_edges
    tiers.append(new_tier)
    used_edges.update(new_tier)

edges = []
for t in tiers:
    edges.extend(t)
edge_set = set(edges)
queue = [(e0, e1) for e0, e1 in edges if int(e0[-1]) < int(e1[-1])]

while len(queue):
    to_del = queue.pop()
    sink = to_del[1]
    parents = [s0 for s0, s1 in edge_set if s1 == to_del[0]]
    edge_set.discard(to_del)
    for p in parents:
        edge_set.add((p, sink))
        if int(p[-1]) < int(sink[-1]):
            queue.append((p, sink))

edge_set.discard(("case_count_0", "root_0"))

pprint.pp(edge_set)
