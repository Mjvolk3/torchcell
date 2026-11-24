# experiments/050-ecoli-benchmark/scripts/build_splits.py
# [[experiments.050-ecoli-benchmark.scripts.build_splits]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/050-ecoli-benchmark/scripts/build_splits
"""Build the pinned split files of the four E. coli benchmark tasks.

Reads the committed protein clusters (``cluster_proteins.py``), builds each task with
:func:`torchcell.tasks.ecoli.build_task` and writes
``torchcell/tasks/data/<task>.splits.csv.gz``: one row per sample, one column per split.
Train, validation and test hold about 80 %, 10 % and 10 % of the unit that is split.

Single-gene tasks (the unit is the gene):

- ``random_s0`` to ``random_s4``: genes at random.
- ``seqid30_s0`` to ``seqid30_s2``, ``seqid50_s0`` to ``seqid50_s2``: whole protein
  clusters at 30 % and 50 % sequence identity.
- ``position_s0`` to ``position_s2``: whole chromosome blocks (40 contiguous blocks of
  equal gene count), so neighbors in an operon stay together.

Gene-pair task:

- ``random_s0`` to ``random_s4``: pairs at random; the two orders of a pair and
  repeated measurements of it stay together.
- ``gene_holdout_s*``, ``seqid30_gene_holdout_s*``, ``seqid50_gene_holdout_s*``,
  ``position_gene_holdout_s*`` (three seeds each): the genes are split first (at random,
  by cluster, by block); a pair is test when either gene is a test gene, validation
  when neither is test and either is validation, train when both are train genes.

Also writes ``results/splits_summary.csv`` (sizes of every split) and prints the sha256
of each file, which is pinned in ``torchcell.tasks.ecoli.SPLIT_SHA256``.

    python experiments/050-ecoli-benchmark/scripts/build_splits.py
"""

from __future__ import annotations

import json
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.tasks.ecoli import ECOLI_TASKS, build_task, write_splits
from torchcell.tasks.splits import (
    assign_groups,
    block_groups,
    pair_parts,
    part_sizes,
    random_split,
)
from torchcell.tasks.task import BenchmarkTask

RESULTS = "experiments/050-ecoli-benchmark/results"
DATA = "torchcell/tasks/data"
RANDOM_SEEDS = range(5)
GROUP_SEEDS = range(3)
N_BLOCKS = 40


def gene_groups(task: BenchmarkTask, genes: list[str]) -> dict[str, dict[str, str]]:
    """Grouping name -> gene -> group, for the genes of a task."""
    out: dict[str, dict[str, str]] = {}
    for threshold in (30, 50):
        table = pd.read_csv(
            osp.join(
                RESULTS, "clusters", f"{task.spec.assembly_set}.seqid{threshold}.tsv"
            ),
            sep="\t",
            dtype=str,
        )
        cluster = dict(zip(table["locus_tag"], table["cluster"]))
        out[f"seqid{threshold}"] = {g: cluster[g] for g in genes}
    starts = [int(task.genome[g].start) for g in genes]
    out["position"] = dict(zip(genes, block_groups(starts, N_BLOCKS)))
    return out


def single_gene_splits(task: BenchmarkTask) -> pd.DataFrame:
    """Every split of a single-gene task, one column each."""
    genes = [str(g) for g in task.genes[:, 0]]
    columns: dict[str, np.ndarray] = {}
    for seed in RANDOM_SEEDS:
        columns[f"random_s{seed}"] = random_split(genes, seed)
    for name, group in gene_groups(task, sorted(set(genes))).items():
        for seed in GROUP_SEEDS:
            columns[f"{name}_s{seed}"] = assign_groups([group[g] for g in genes], seed)
    return pd.DataFrame(columns)


def gene_pair_splits(task: BenchmarkTask) -> pd.DataFrame:
    """Every split of the gene-pair task, one column each."""
    a = [str(g) for g in task.genes[:, 0]]
    b = [str(g) for g in task.genes[:, 1]]
    unordered = ["|".join(sorted(pair)) for pair in zip(a, b)]
    genes = sorted(set(a) | set(b))
    columns: dict[str, np.ndarray] = {}
    for seed in RANDOM_SEEDS:
        columns[f"random_s{seed}"] = random_split(unordered, seed)
    groupings = {"gene_holdout": {g: g for g in genes}}
    for name, group in gene_groups(task, genes).items():
        groupings[f"{name}_gene_holdout"] = group
    for name, group in groupings.items():
        for seed in GROUP_SEEDS:
            part = dict(zip(genes, assign_groups([group[g] for g in genes], seed)))
            columns[f"{name}_s{seed}"] = pair_parts(
                [part[g] for g in a], [part[g] for g in b]
            )
    return pd.DataFrame(columns)


def main() -> None:
    """Write the four split files, their sizes and their sha256."""
    load_dotenv(".env")
    rows = []
    digests = {}
    for name, spec in ECOLI_TASKS.items():
        task = build_task(name)
        splits = (
            single_gene_splits(task)
            if spec.kind == "single_gene"
            else gene_pair_splits(task)
        )
        digests[name] = write_splits(
            osp.join(DATA, f"{name}.splits.csv.gz"), task, splits
        )
        for column in splits.columns:
            rows.append(
                {
                    "task": name,
                    "split": column,
                    "samples": len(task),
                    **part_sizes(splits[column].to_numpy(dtype=str)),
                    "excluded_records": len(task.excluded),
                }
            )
        print(f'    "{name}": "{digests[name]}",')
    pd.DataFrame(rows).to_csv(osp.join(RESULTS, "splits_summary.csv"), index=False)
    with open(osp.join(RESULTS, "splits_sha256.json"), "w", encoding="utf-8") as handle:
        json.dump(digests, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
