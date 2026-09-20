# experiments/029-solid-growth-ko/scripts/same_genotype_spread.py
# [[experiments.029-solid-growth-ko.scripts.closure_recompute]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/029-solid-growth-ko/scripts/same_genotype_spread

"""How much of the spread among the fitness entries of ONE genotype is replicate noise within a
screen, and how much is a systematic difference between screens? Reads the 029 closure entries
(closure_recompute.py scan) and writes results/same_genotype_spread.json. The question it answers:
whether the screen of origin is a hidden variable a model would need, or noise.
"""
import json, sys
import os, os.path as osp
import numpy as np, pandas as pd

from dotenv import load_dotenv
load_dotenv()
C = osp.join(os.environ["DATA_ROOT"], "data/torchcell/experiments/029-solid-growth-ko/closure")
OUT = osp.join(os.environ["EXPERIMENT_ROOT"], "029-solid-growth-ko/results/same_genotype_spread.json")
RESULTS: dict = {}
e = pd.read_parquet(osp.join(C, "entries.parquet"))
f = e[(e["exp_type"] == "fitness") & ~e["dataset"].str.contains("Essentiality|SynthLeth")].copy()
f["screen"] = f["dataset"].str.replace("Dataset", "") + "@" + f["temp"].astype(int).astype(str)

def pair_corr(df, key_a, key_b, label):
    a = df[df["screen"] == key_a].groupby("genes")["value"].mean()
    b = df[df["screen"] == key_b].groupby("genes")["value"].mean()
    j = a.index.intersection(b.index)
    if len(j) < 50:
        print(f"  {label}: n={len(j)} (too few)"); return
    x, y = a.loc[j].to_numpy(), b.loc[j].to_numpy()
    RESULTS[label] = {"n": int(len(j)), "r": float(np.corrcoef(x, y)[0, 1]), "mean_diff": float(np.mean(x - y)), "median_abs_diff": float(np.median(np.abs(x - y)))}
    print(f"  {label}: n={len(j):,}  r={np.corrcoef(x, y)[0,1]:.3f}  mean diff={np.mean(x-y):+.4f}  median|diff|={np.median(np.abs(x-y)):.4f}")

for order in (1, 2):
    d = f[f["order"] == order]
    print(f"\n===== order {order}: {d['genes'].nunique():,} gene sets, {len(d):,} fitness entries")
    # variance decomposition over gene sets with >1 entry
    g = d.groupby("genes")
    multi = d[g["value"].transform("size") > 1]
    tot = multi.groupby("genes")["value"].var(ddof=0)
    within = multi.groupby(["genes", "screen"])["value"].var(ddof=0)
    n_ws = multi.groupby(["genes", "screen"]).size()
    within_pooled = (within * n_ws).groupby(level=0).sum() / multi.groupby("genes").size()
    RESULTS[f"order{order}_variance"] = {"gene_sets_multi": int(len(tot)), "total_sd": float(np.sqrt(tot.mean())), "within_screen_sd": float(np.sqrt(within_pooled.mean())), "between_screen_share": float(1 - within_pooled.sum()/tot.sum()), "entries_per_screen": {k: int(v) for k, v in d["screen"].value_counts().items()}}
    print(f"  gene sets with >1 entry: {len(tot):,}; mean total sd {np.sqrt(tot.mean()):.4f}; "
          f"mean within-screen sd {np.sqrt(within_pooled.mean()):.4f}; "
          f"between-screen share of variance {(1 - within_pooled.sum()/tot.sum()):.2f}")
    print("  entries per screen:", d["screen"].value_counts().to_dict())
    if order == 1:
        pair_corr(d, "SmfCostanzo2016@26", "SmfCostanzo2016@30", "Costanzo 26 C vs 30 C, same gene")
        pair_corr(d, "SmfCostanzo2016@30", "SmfKuzmin2018@26", "Costanzo 30 C vs Kuzmin 2018 query single")
        pair_corr(d, "SmfCostanzo2016@30", "SmfKuzmin2020@26", "Costanzo 30 C vs Kuzmin 2020 query single")
        # marker orientation within Costanzo 30
        c30 = d[d["screen"] == "SmfCostanzo2016@30"]
        km = c30[c30["marker"] == "kanmx"].groupby("genes")["value"].mean()
        nm = c30[c30["marker"] == "natmx"].groupby("genes")["value"].mean()
        j = km.index.intersection(nm.index)
        print(f"  Costanzo 30 C kanmx vs natmx, same gene: n={len(j):,} r={np.corrcoef(km.loc[j], nm.loc[j])[0,1]:.3f} median|diff|={np.median(np.abs(km.loc[j]-nm.loc[j])):.4f}")
    else:
        pair_corr(d, "DmfCostanzo2016@26", "DmfCostanzo2016@30", "Costanzo 26 C vs 30 C, same pair")
        pair_corr(d, "DmfCostanzo2016@30", "DmfKuzmin2018@26", "Costanzo 30 C vs Kuzmin 2018, same pair")
        pair_corr(d, "DmfCostanzo2016@30", "DmfKuzmin2020@26", "Costanzo 30 C vs Kuzmin 2020, same pair")
        pair_corr(d, "DmfKuzmin2018@26", "DmfKuzmin2020@26", "Kuzmin 2018 vs 2020, same pair")
        # two orientations within Costanzo 30 (query/array swapped): marker string differs
        c30 = d[d["screen"] == "DmfCostanzo2016@30"]
        o1 = c30[c30["marker"] == "natmx|kanmx"].groupby("genes")["value"].mean()
        o2 = c30[c30["marker"] == "kanmx|natmx"].groupby("genes")["value"].mean()
        j = o1.index.intersection(o2.index)
        print(f"  Costanzo 30 C, the two query/array orientations of one pair: n={len(j):,} r={np.corrcoef(o1.loc[j], o2.loc[j])[0,1]:.3f} median|diff|={np.median(np.abs(o1.loc[j]-o2.loc[j])):.4f}")

# interaction: same pair, two screens
gi = e[(e["exp_type"] == "gene interaction") & (e["order"] == 2)].copy()
gi["screen"] = gi["dataset"].str.replace("Dataset", "") + "@" + gi["temp"].astype(int).astype(str)
print("\n===== digenic interaction, same pair across screens")
pair_corr(gi, "DmiCostanzo2016@26", "DmiCostanzo2016@30", "Costanzo 26 C vs 30 C")
pair_corr(gi, "DmiCostanzo2016@30", "DmiKuzmin2018@26", "Costanzo 30 C vs Kuzmin 2018")
pair_corr(gi, "DmiCostanzo2016@30", "DmiKuzmin2020@26", "Costanzo 30 C vs Kuzmin 2020")
c30 = gi[gi["screen"] == "DmiCostanzo2016@30"]
o1 = c30[c30["marker"] == "natmx|kanmx"].groupby("genes")["value"].mean()
o2 = c30[c30["marker"] == "kanmx|natmx"].groupby("genes")["value"].mean()
j = o1.index.intersection(o2.index)
print(f"  Costanzo 30 C, two orientations: n={len(j):,} r={np.corrcoef(o1.loc[j], o2.loc[j])[0,1]:.3f}")

json.dump(RESULTS, open(OUT, "w"), indent=2)
print("wrote", OUT)
