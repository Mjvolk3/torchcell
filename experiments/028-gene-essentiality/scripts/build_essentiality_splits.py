# experiments/028-gene-essentiality/scripts/build_essentiality_splits.py
# [[experiments.028-gene-essentiality.scripts.build_essentiality_splits]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/build_essentiality_splits.py

r"""Per-gene essentiality labels and a PINNED train/val/test split, one file per design.

WHY A SPLIT FILE. The wave-9 protocol pinned the 223 released test genes and let the data
module draw train/val at 80/10/10 over the remaining 5,461 records, of which only 926 carry
a label. That gave a validation set of 75 labeled genes with about 12 essential, on which a
per-epoch AUROC has a Hanley-McNeil standard error near 0.07 (hypothesis for the spiky
curves, tested by the wave-12 cells that read this file), spent 10 percent of the labeled
genes on a random test block nothing reports, and forwarded 4,535 unlabeled records per
epoch under a False mask. A split file fixes all three: it names every record's split, so
the trainer drops what it does not name (``drop_unpinned``), and it carries the labels it
was stratified on, so a cell's labels and its split are one artifact.

TWO LABEL SOURCES.

- ``fcl``: Merzbacher 2025's released labels on 1,121 genes (inversion applied once, class
  counts asserted), the labels FCL itself trained on. Same supervision budget as the
  comparator.
- ``genome``: the released labels where they exist, and elsewhere the union build's own
  evidence: a genotype with an SGD ``is_essential`` record is essential, a genotype with a
  single-mutant fitness record (a viable deletion strain, Costanzo 2016 or Kuzmin 2018) and
  no SGD record is nonessential. About five times the labeled genes. Where the two sources
  disagree the released label wins, so the held-out 223 are scored on exactly the
  comparator's labels either way; the disagreements are counted and listed.

THE TEST SET NEVER MOVES: the released held-out genes, whichever label source. Validation is
a class-stratified fraction of the remaining labeled genes drawn at ``--seed``; ``--folds K``
writes K files whose validation blocks partition those genes, for a cross-validated read.

    python experiments/028-gene-essentiality/scripts/build_essentiality_splits.py \
        --labels fcl --val-fraction 0.2 --seed 0
    python experiments/028-gene-essentiality/scripts/build_essentiality_splits.py \
        --labels genome --folds 5 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import os.path as osp
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "scripts"))

import train_essentiality as t  # noqa: E402

SPLITS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results", "splits")
LABEL_SOURCES = ("fcl", "genome", "genome_heldout")
#: The genome-wide labels as a LARGER HELD-OUT SET: train and val on the released
#: labeled genes (the 195 released held-out genes dropped, so the read is not on them),
#: test on every gene the released file never touched (SGD-essential, or present in a
#: fitness screen and so nonessential). Label origin is the class on this test set:
#: every essential test gene is an SGD annotation and every nonessential one a
#: deletion-collection fitness record, which the report and every read of it must say.
GENOME_HELDOUT_NOTE = (
    "record indices are into the 028 essentiality build; train and val are the released "
    "labeled genes outside the released held-out set (val class-stratified), the released "
    "held-out genes are dropped, and test is every gene the released file never labeled: "
    "essential from SGD, nonessential from a fitness record, so label origin coincides "
    "with class on the test set; every record the file does not name is dropped by the "
    "trainer."
)


class SplitReport(BaseModel):
    """Counts and provenance written beside a split's record indices."""

    label_source: str
    seed: int
    fold: int | None
    folds: int | None
    val_fraction: float | None
    n_genes_labeled: int
    n_genes_essential: int
    n_genes_from_released: int
    n_genes_from_sgd: int
    n_genes_from_fitness: int
    n_disagreements_released_vs_build: int
    disagreements: list[dict[str, Any]]
    n_genes_absent_from_build: int
    test_genes: int
    test_essential: int
    val_genes: int
    val_essential: int
    train_genes: int
    train_essential: int
    n_records: dict[str, int]
    note: str = Field(
        default=(
            "record indices are into the 028 essentiality build; test = the released "
            "held-out genes; val is class-stratified over the remaining labeled genes; "
            "every record the file does not name is dropped by the trainer."
        )
    )


def build_labels(
    dataset: Any, source: str
) -> tuple[dict[str, int], dict[str, str], list[dict[str, Any]]]:
    """``gene -> label`` and ``gene -> provenance`` under one label source."""
    released, _ = t.read_fcl_essentiality_split(
        osp.join(DATA_ROOT, t.FCL_ESSENTIALITY_SPLIT)
    )
    labels: dict[str, int] = dict(released)
    provenance = {g: "released" for g in released}
    disagreements: list[dict[str, Any]] = []
    if source == "fcl":
        return labels, provenance, disagreements
    gene_index = dataset.is_any_deletion_gene_index
    label_index = dataset.phenotype_label_index
    essential_records = set(label_index[t.ESSENTIALITY_LABEL])
    fitness_records = set(label_index["fitness"])
    for gene, records in gene_index.items():
        records = {int(r) for r in records}
        if records & essential_records:
            build_label = 1
            origin = "sgd"
        elif records & fitness_records:
            build_label = 0
            origin = "fitness"
        else:
            continue
        if gene in released:
            if released[gene] != build_label:
                disagreements.append(
                    {
                        "gene": gene,
                        "released": released[gene],
                        "build": build_label,
                        "build_origin": origin,
                    }
                )
            continue
        labels[gene] = build_label
        provenance[gene] = origin
    return labels, provenance, disagreements


def stratified_partition(
    genes: list[str], labels: dict[str, int], k: int, rng: np.random.Generator
) -> list[list[str]]:
    """K blocks with the essential fraction held in each."""
    blocks: list[list[str]] = [[] for _ in range(k)]
    for cls in (0, 1):
        members = [g for g in genes if labels[g] == cls]
        rng.shuffle(members)
        for i, g in enumerate(members):
            blocks[i % k].append(g)
    return blocks


def write_split(
    dataset: Any,
    labels: dict[str, int],
    provenance: dict[str, str],
    disagreements: list[dict[str, Any]],
    test_genes: set[str],
    val_genes: set[str],
    args: argparse.Namespace,
    fold: int | None,
    note: str | None = None,
) -> str:
    gene_index = dataset.is_any_deletion_gene_index
    split_records: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_genes: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    absent = 0
    for gene in sorted(labels):
        records = gene_index.get(gene)
        if not records:
            absent += 1
            continue
        name = "test" if gene in test_genes else "val" if gene in val_genes else "train"
        split_records[name].extend(int(r) for r in records)
        split_genes[name].append(gene)
    for name, recs in split_records.items():
        assert len(recs) == len(set(recs)), name
    all_recs = [r for recs in split_records.values() for r in recs]
    assert len(all_recs) == len(set(all_recs)), "a record landed in two splits"
    report = SplitReport(
        label_source=args.labels,
        seed=args.seed,
        fold=fold,
        folds=args.folds,
        val_fraction=None if args.folds else args.val_fraction,
        n_genes_labeled=len(labels),
        n_genes_essential=sum(labels.values()),
        n_genes_from_released=sum(1 for p in provenance.values() if p == "released"),
        n_genes_from_sgd=sum(1 for p in provenance.values() if p == "sgd"),
        n_genes_from_fitness=sum(1 for p in provenance.values() if p == "fitness"),
        n_disagreements_released_vs_build=len(disagreements),
        disagreements=disagreements,
        n_genes_absent_from_build=absent,
        test_genes=len(split_genes["test"]),
        test_essential=sum(labels[g] for g in split_genes["test"]),
        val_genes=len(split_genes["val"]),
        val_essential=sum(labels[g] for g in split_genes["val"]),
        train_genes=len(split_genes["train"]),
        train_essential=sum(labels[g] for g in split_genes["train"]),
        n_records={k: len(v) for k, v in split_records.items()},
        **({"note": note} if note else {}),
    )
    stem = (
        f"ess_{args.labels}_"
        + (
            f"fold{fold}of{args.folds}"
            if fold is not None
            else f"val{args.val_fraction:g}"
        )
        + f"_s{args.seed}"
    )
    path = osp.join(SPLITS_DIR, f"{stem}.json.gz")
    os.makedirs(SPLITS_DIR, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(
            {
                "labels": labels,
                "provenance": provenance,
                "genes": split_genes,
                "train": sorted(split_records["train"]),
                "val": sorted(split_records["val"]),
                "test": sorted(split_records["test"]),
                "report": report.model_dump(),
            },
            fh,
        )
    print(
        f"[split] {stem}: genes train {report.train_genes} ({report.train_essential} ess) "
        f"val {report.val_genes} ({report.val_essential}) test {report.test_genes} "
        f"({report.test_essential}); records {report.n_records}; labels from released "
        f"{report.n_genes_from_released}, sgd {report.n_genes_from_sgd}, fitness "
        f"{report.n_genes_from_fitness}; {len(disagreements)} disagreements; "
        f"{absent} labeled genes absent from the build",
        flush=True,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", choices=LABEL_SOURCES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=None)
    args = parser.parse_args()

    dataset = t.build_dataset(t.ESSENTIALITY_HEAD)
    labels, provenance, disagreements = build_labels(dataset, args.labels)
    # The held-out set from the released file itself, so no label source can widen it.
    with open(osp.join(DATA_ROOT, t.FCL_ESSENTIALITY_SPLIT)) as handle:
        test_genes = {
            str(r["knockout"]).strip()
            for r in csv.DictReader(handle)
            if str(r["test"]).strip() == "1" and str(r["essential"]).strip()
        }
    assert len(test_genes) == 223
    gene_index = dataset.is_any_deletion_gene_index
    note: str | None = None
    if args.labels == "genome_heldout":
        # The genome-wide labels as the test set; the released held-out genes leave the
        # file, so train and val are the released labeled genes outside them.
        assert not args.folds, "the genome held-out split has no fold form"
        released_test = test_genes
        labels = {g: v for g, v in labels.items() if g not in released_test}
        provenance = {g: p for g, p in provenance.items() if g in labels}
        test_genes = {g for g, p in provenance.items() if p != "released"}
        assert all(
            (provenance[g] == "sgd") == bool(labels[g]) for g in test_genes
        ), "label origin no longer coincides with class on the genome test set"
        note = GENOME_HELDOUT_NOTE
    pool = sorted(g for g in labels if g not in test_genes and gene_index.get(g))
    rng = np.random.default_rng(args.seed)
    paths: list[str] = []
    if args.folds:
        blocks = stratified_partition(pool, labels, args.folds, rng)
        for k, block in enumerate(blocks):
            paths.append(
                write_split(
                    dataset,
                    labels,
                    provenance,
                    disagreements,
                    test_genes,
                    set(block),
                    args,
                    k,
                )
            )
    else:
        k = max(2, round(1.0 / args.val_fraction))
        blocks = stratified_partition(pool, labels, k, rng)
        paths.append(
            write_split(
                dataset,
                labels,
                provenance,
                disagreements,
                test_genes,
                set(blocks[0]),
                args,
                None,
                note=note,
            )
        )
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
