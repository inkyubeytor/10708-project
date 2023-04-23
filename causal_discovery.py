from data_processing import load_data, get_datasets
from patsy.highlevel import dmatrices
import patsy
import cdt
import numpy as np
import networkx as nx
import pprint


ignore_na = patsy.missing.NAAction(NA_types=[])


df = load_data()
datasets = get_datasets(df)

reg_eq = "case_count ~ " + " + ".join(list(df.columns[2:].values))

data = datasets["alpha"]

train_size = 0.25  # percentage of data for training only
train_idx = int(len(data) * train_size)
train_data = data.iloc[:train_idx].reset_index()

y, X = dmatrices(reg_eq, data=train_data, return_type="dataframe",
                 NA_action=ignore_na)


normalized_df = (train_data - train_data.mean()) / train_data.std()
normalized_df.drop(columns=[c for c in normalized_df.columns if any(np.isnan(normalized_df[c]))], inplace=True)
normalized_df.drop(columns=["week"], inplace=True)
glasso = cdt.independence.graph.Glasso()
skeleton = glasso.predict(normalized_df)
# print(nx.adjacency_matrix(skeleton).todense())
new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg='aracne')
# print(nx.adjacency_matrix(new_skeleton).todense())


model = cdt.causality.graph.GES()
output_graph = model.predict(normalized_df, new_skeleton)
# print(nx.adjacency_matrix(output_graph).todense())
# print(output_graph.edges)
# print([e for e in output_graph.edges if e[1] == "case_count"])
tiers = [[("case_count", "")]]
while len(tiers[-1]):
    tiers.append([e for e in output_graph.edges if e[1] in [x for x, _ in tiers[-1]]])
pprint.pprint(tiers)
