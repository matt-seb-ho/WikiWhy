import numpy as np
import pandas as pd
import argparse

psr = argparse.ArgumentParser()
psr.add_argument("auto_eval", help="csv path")
psr.add_argument("human_eval", help="csv path")
args = psr.parse_args()

human = pd.read_csv(args.human_eval, index_col="id")
auto = pd.read_csv(args.auto_eval, index_col="id")

tf_cols = (
    [f"correct{i}" for i in range(1, 4)] 
    + [f"similar{i}" for i in range(1, 4)]
)

for col in tf_cols:
    human[col] = (human[col] == 'T').astype('int8')

human['correct'] = human['correct1'] + human['correct2'] + human['correct3']
human['similar'] = human['similar1'] + human['similar2'] + human['similar3']

auto_cols = ["precise", "predicted_positive", "covered", "relevant", "true_positive"]
for col in auto_cols:
    human[col] = auto[col]

df = human[["correct", "similar"] + auto_cols]

df["u_p"] = df["precise"] / df["predicted_positive"]
df["u_r"] = df["covered"] / df["relevant"]
df["o_p"] = df["true_positive"] / df["predicted_positive"]
df["o_r"] = df["true_positive"] / df["relevant"]
df["u_f1"] = 2.0 * (df["u_p"] * df["u_r"]) / (df["u_p"] + df["u_r"])
df["o_f1"] = 2.0 * (df["o_p"] * df["o_r"]) / (df["o_p"] + df["o_r"])

corr_mat = df.drop(columns=["predicted_positive", "relevant"]).corr()
print(corr_mat)

print('\n')

cols = corr_mat[["u_f1", "o_f1"]]
print(cols.loc[["correct", "similar"]])
