#!/usr/bin/env python
"""Compute per-(gene, modality) furfural enrichment from guide barcode counts,
and validate against the paper's known hits.

Enrichment (paper's definition): per replicate, normalized(furfural After) /
normalized(untreated Before); log2; averaged over 3 biological triplicates.
Rounds are iterative (5/10/15 mM) in accumulating backgrounds
(R1: bAID; R2: +SIZ1i; R3: +SIZ1i +NAT1a).
"""

import glob
import os

import numpy as np
import pandas as pd

WORK = os.environ["LIAN_WORK"]

man = pd.read_csv(os.path.join(WORK, "run_manifest.tsv"), sep="\t")
gm = pd.read_csv(
    os.path.join(WORK, "guide_map.tsv"), sep="\t"
)  # guide_id, mod, gene, spacer, refseq

# --- load count matrix (guide_id x run) ---
mat = {}
for f in glob.glob(os.path.join(WORK, "counts", "*.tsv")):
    run = os.path.basename(f)[:-4]
    if run == "qc":
        continue
    s = pd.read_csv(f, sep="\t", header=None, index_col=0).iloc[:, 0]
    mat[run] = s
counts = pd.DataFrame(mat).fillna(0.0)
counts.index.name = "guide_id"
print("count matrix:", counts.shape)

# --- CPM normalize (pseudocount 1 read) ---
cpm = (counts + 1.0) / (counts.sum(axis=0) + len(counts)) * 1e6

# --- per-round per-replicate log2(after/before), mean over triplicates ---
rounds = {}
for rnd in (1, 2, 3):
    sub = man[man["round"] == rnd]
    fcs = []
    for rep in (1, 2, 3):
        aft = sub[(sub.condition == "after") & (sub.replicate == rep)]["run"].iloc[0]
        bef = sub[(sub.condition == "before") & (sub.replicate == rep)]["run"].iloc[0]
        fcs.append(np.log2(cpm[aft] / cpm[bef]))
    fc = pd.concat(fcs, axis=1)
    rounds[rnd] = pd.DataFrame(
        {
            f"r{rnd}_log2fc_mean": fc.mean(axis=1),
            f"r{rnd}_log2fc_sd": fc.std(axis=1, ddof=1),
        }
    )

enr = gm.set_index("guide_id").join(pd.concat(rounds.values(), axis=1))
enr.to_csv(os.path.join(WORK, "guide_enrichment.tsv"), sep="\t")

# --- aggregate per (gene, modality): median over the gene's guides ---
agg = (
    enr.groupby(["gene", "mod"])
    .agg(
        n_guides=("spacer", "size"),
        r1=("r1_log2fc_mean", "median"),
        r2=("r2_log2fc_mean", "median"),
        r3=("r3_log2fc_mean", "median"),
    )
    .reset_index()
)
agg.to_csv(os.path.join(WORK, "gene_modality_enrichment.tsv"), sep="\t", index=False)


# --- VALIDATION vs known hits ---
def top(col, k=8):
    return (
        agg.dropna(subset=[col])
        .sort_values(col, ascending=False)
        .head(k)[["gene", "mod", col, "n_guides"]]
    )


modname = {"a": "activation", "i": "interference", "d": "deletion"}
for rnd in ("r1", "r2", "r3"):
    print(f"\n=== TOP {rnd} (gene, modality) enrichment ===")
    t = top(rnd)
    t["mod"] = t["mod"].map(modname)
    print(t.to_string(index=False))

print("\n=== paper's key hits — where do they rank? ===")
for gene, mod, rnd in [
    ("SIZ1", "i", "r1"),
    ("SLX5", "i", "r1"),
    ("NAT1", "a", "r2"),
    ("PDR1", "i", "r3"),
]:
    d = (
        agg.dropna(subset=[rnd])
        .sort_values(rnd, ascending=False)
        .reset_index(drop=True)
    )
    hit = d[(d.gene == gene) & (d["mod"] == mod)]
    if len(hit):
        rank = hit.index[0] + 1
        print(
            f"  {gene}{mod} ({rnd}): rank {rank}/{len(d)}  log2fc={hit[rnd].iloc[0]:.2f}"
        )
    else:
        print(f"  {gene}{mod} ({rnd}): NOT FOUND")
print("ENRICH_COMPLETE")
