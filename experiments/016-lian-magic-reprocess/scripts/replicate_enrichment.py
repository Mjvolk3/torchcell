#!/usr/bin/env python
# experiments/016-lian-magic-reprocess/scripts/replicate_enrichment.py
# [[experiments.016-lian-magic-reprocess.scripts.replicate_enrichment]]
"""Emit the PER-REPLICATE furfural log2 enrichment, the values the mean table hid.

``enrichment.py`` computes, per round and per biological replicate, ``log2(CPM[furfural
after] / CPM[untreated before])`` and then writes only the mean and the SD over the three
replicates. The three per-replicate numbers are the measurements; this script writes them
out instead, one column per (round, replicate):
``r{rnd}_rep{rep}_log2fc`` -> ``guide_enrichment_replicates.tsv``, which
``torchcell/datasets/scerevisiae/lian2019.py`` consumes as one record per (guide, round,
replicate).

The arithmetic is COPIED from ``enrichment.py`` (same count matrix, same CPM with a
pseudocount of one read, same replicate pairing from ``run_manifest.tsv``), and that is
asserted rather than assumed: the mean and the sample SD over the three emitted columns
must reproduce ``guide_enrichment_final.tsv``'s ``r{rnd}_log2fc_mean`` /
``r{rnd}_log2fc_sd`` for every guide (checked to 1e-9; measured 1.3e-15 on
2026-11-24). The ``is_control`` / ``corrupted_gene`` flags are carried over from that same
sha256-pinned table, so no flag is recomputed here.

Replicate labels are the authors' own: SRA ``PRJNA504483`` names each library
``Ff1a`` ... ``3Ff3b`` (``LibraryName`` = ``SampleName``), round 1 carrying no prefix and
rounds 2 and 3 the prefixes ``2`` and ``3``, with ``a`` the furfural-stressed and ``b`` the
untreated library of that replicate. ``run_manifest.tsv``'s ``library`` column is those
names verbatim (21/21 runs agree with the SRA run report).

Run inside ``reproduce.sh``, or standalone with ``LIAN_WORK`` set to the pipeline work dir
that holds ``counts/``, ``guide_map.tsv``, ``run_manifest.tsv`` and
``guide_enrichment_final.tsv``.
"""

import glob
import os

import numpy as np
import pandas as pd

WORK = os.environ["LIAN_WORK"]
ROUNDS = (1, 2, 3)
REPLICATES = (1, 2, 3)
#: the largest absolute disagreement allowed between the recomputed mean / SD over the
#: emitted replicate columns and the pinned mean table
TOLERANCE = 1e-9

guide_map = pd.read_csv(os.path.join(WORK, "guide_map.tsv"), sep="\t")
manifest = pd.read_csv(os.path.join(WORK, "run_manifest.tsv"), sep="\t")
final = pd.read_csv(os.path.join(WORK, "guide_enrichment_final.tsv"), sep="\t")

# --- count matrix (guide_id x run), exactly as enrichment.py builds it ---
matrix = {}
for path in glob.glob(os.path.join(WORK, "counts", "*.tsv")):
    run = os.path.basename(path)[:-4]
    if run == "qc":
        continue
    matrix[run] = pd.read_csv(path, sep="\t", header=None, index_col=0).iloc[:, 0]
counts = pd.DataFrame(matrix).fillna(0.0)
counts.index.name = "guide_id"
print(f"count matrix: {counts.shape}", flush=True)

# --- CPM normalize (pseudocount 1 read) ---
cpm = (counts + 1.0) / (counts.sum(axis=0) + len(counts)) * 1e6

# --- per round, per replicate log2(after / before) ---
columns: dict[str, pd.Series] = {}
labels: dict[str, str] = {}
for rnd in ROUNDS:
    sub = manifest[manifest["round"] == rnd]
    for rep in REPLICATES:
        after = sub[(sub.condition == "after") & (sub.replicate == rep)]
        before = sub[(sub.condition == "before") & (sub.replicate == rep)]
        name = f"r{rnd}_rep{rep}_log2fc"
        columns[name] = np.log2(cpm[after["run"].iloc[0]] / cpm[before["run"].iloc[0]])
        labels[name] = f"{after['library'].iloc[0]}/{before['library'].iloc[0]}"
replicates = pd.DataFrame(columns)

# --- the pinned mean table must be reproduced by the emitted columns ---
joined = final.set_index("guide_id").join(replicates)
for rnd in ROUNDS:
    block = joined[[f"r{rnd}_rep{rep}_log2fc" for rep in REPLICATES]]
    measured = joined[f"r{rnd}_log2fc_mean"]
    present = measured.notna()
    if not (block.notna().all(axis=1) == present).all():
        raise RuntimeError(
            f"round {rnd}: a guide with a released mean is missing a replicate value, "
            "or the other way round"
        )
    mean_diff = (measured[present] - block[present].mean(axis=1)).abs().max()
    sd_diff = (
        (joined[f"r{rnd}_log2fc_sd"][present] - block[present].std(axis=1, ddof=1))
        .abs()
        .max()
    )
    print(
        f"round {rnd}: {int(present.sum())} guides, max |mean - recomputed| "
        f"{mean_diff:.3e}, max |sd - recomputed| {sd_diff:.3e}",
        flush=True,
    )
    if mean_diff > TOLERANCE or sd_diff > TOLERANCE:
        raise RuntimeError(
            f"round {rnd} does not reproduce the pinned mean table "
            f"(mean {mean_diff:.3e}, sd {sd_diff:.3e} > {TOLERANCE:.0e})"
        )

out_columns = [
    "guide_id",
    "mod",
    "gene",
    "spacer",
    "is_control",
    "corrupted_gene",
    *columns,
]
out = final.set_index("guide_id").join(replicates).reset_index()[out_columns]
# the guide_map is the row order of the pinned table; a reorder would break the loader's
# positional join of the CRISPRd design library onto the enrichment rows
if list(out["guide_id"]) != list(guide_map["guide_id"]):
    raise RuntimeError("the emitted row order is not the guide_map's row order")
path = os.path.join(WORK, "guide_enrichment_replicates.tsv")
out.to_csv(path, sep="\t", index=False)
print(
    f"wrote guide_enrichment_replicates.tsv: {len(out)} guides x "
    f"{len(columns)} (round, replicate) columns; labels "
    + ", ".join(f"{name}={label}" for name, label in labels.items()),
    flush=True,
)
print("REPLICATE_ENRICHMENT_COMPLETE", flush=True)
