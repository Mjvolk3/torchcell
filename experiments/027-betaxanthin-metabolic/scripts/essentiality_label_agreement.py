# experiments/027-betaxanthin-metabolic/scripts/essentiality_label_agreement.py
# [[experiments.027-betaxanthin-metabolic.scripts.essentiality_label_agreement]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/essentiality_label_agreement
"""How far two curated sources agree on the essentiality label, which bounds the endpoint.

Essentiality is binary, so its reproducibility bound is not a correlation: it is the accuracy
a predictor that reproduces one source perfectly would score against the other. This script
compares the label the trainer uses, Merzbacher's released yeast split with its inversion
fixed, against an independently curated list, the SGD essential-gene set.

The SGD dataset is a LIST of essential genes with no negative class, so a gene's absence from
it counts as nonessential only for genes the split names. That asymmetry is why the two
disagreement directions are reported separately rather than as one confusion matrix.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/essentiality_label_agreement.py

Writes ``results/essentiality_label_agreement.json``.
"""

from __future__ import annotations

import csv
import json
import os
import os.path as osp

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
SPLIT = osp.join(
    DATA_ROOT, "data", "merzbacher2025_fcl", "deletionprediction-main", "data", "yeast_essentiality_test_split.csv"
)
SGD_ROOT = osp.join(DATA_ROOT, "data", "torchcell", "gene_essentiality_sgd")
OUT = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results", "essentiality_label_agreement.json")
#: The released column carries 1.0 for a NONESSENTIAL gene; see train_bx.FCL_ESSENTIAL_VALUE.
FCL_ESSENTIAL_VALUE = 0.0


class AgreementReport(BaseModel):
    split_path: str
    sgd_root: str
    n_split_genes: int
    n_split_essential: int
    n_sgd_essential_genes: int
    split_essential_in_sgd: int
    split_nonessential_in_sgd: int = Field(description="Genes the split calls nonessential that SGD lists as essential.")
    split_essential_absent_from_sgd: int
    n_disagreements: int
    agreement_rate: float = Field(description="Highest accuracy a perfect reproduction of SGD scores against the split.")
    majority_rate: float = Field(description="Accuracy of always predicting the majority class of the split.")
    headroom_over_majority: float
    disagreeing_genes_sgd_essential_split_not: list[str]
    disagreeing_genes_split_essential_sgd_not: list[str]


def main() -> None:
    labels: dict[str, int] = {}
    with open(SPLIT) as handle:
        for row in csv.DictReader(handle):
            raw = str(row["essential"]).strip()
            if raw:
                labels[str(row["knockout"]).strip()] = int(float(raw) == FCL_ESSENTIAL_VALUE)

    from torchcell.datasets.scerevisiae.sgd import GeneEssentialitySgdDataset

    dataset = GeneEssentialitySgdDataset(root=SGD_ROOT)
    sgd: set[str] = set()
    for i in range(len(dataset)):
        record = dataset[i]
        sgd.add(record["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"])

    essential = [g for g, v in labels.items() if v == 1]
    nonessential = [g for g, v in labels.items() if v == 0]
    in_sgd_ess = [g for g in essential if g in sgd]
    in_sgd_non = sorted(g for g in nonessential if g in sgd)
    absent_ess = sorted(g for g in essential if g not in sgd)
    n_dis = len(in_sgd_non) + len(absent_ess)
    majority = max(len(essential), len(nonessential)) / len(labels)

    report = AgreementReport(
        split_path=SPLIT,
        sgd_root=SGD_ROOT,
        n_split_genes=len(labels),
        n_split_essential=len(essential),
        n_sgd_essential_genes=len(sgd),
        split_essential_in_sgd=len(in_sgd_ess),
        split_nonessential_in_sgd=len(in_sgd_non),
        split_essential_absent_from_sgd=len(absent_ess),
        n_disagreements=n_dis,
        agreement_rate=1.0 - n_dis / len(labels),
        majority_rate=majority,
        headroom_over_majority=(1.0 - n_dis / len(labels)) - majority,
        disagreeing_genes_sgd_essential_split_not=in_sgd_non,
        disagreeing_genes_split_essential_sgd_not=absent_ess,
    )
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    print(json.dumps({k: v for k, v in report.model_dump().items() if not k.startswith("disagreeing")}, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
