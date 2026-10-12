# experiments/028-gene-essentiality/scripts/essentiality_confusion.py
# [[experiments.028-gene-essentiality.scripts.essentiality_confusion]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/essentiality_confusion

r"""The yeast-GEM benchmark's own statistic, computed for every scorer on record.

yeast-GEM reports gene essentiality as a confusion matrix at one rule: a deletion is
essential when its FBA growth ratio on the Kennedy synthetic complete medium is below
$10^{-6}$, scored against the Stanford deletion project's inviable list over the 1,107
verified ORFs in the model (``results/external/yeast-GEM-9.1.1/``). Accuracy is the
headline; the majority class alone gives 0.856 on that gene set. This script puts every
scorer of this document on the same footing: the confusion matrix, accuracy, precision
and recall for the essential class, specificity, balanced accuracy, and the majority-class
accuracy of the gene set it was scored on.

- yeast-GEM 9.1.1, read from its per-gene file (its labels, its genes).
- FBA on yeast-GEM 9.0.2 on four media at the same $10^{-6}$ rule, on the released labels
  (1,121 genes, and the 195 held-out genes).
- The released Flux Cone Learning predictions at $P(\text{viable}) < 0.5$ on the 223
  released held-out genes and on the 195.
- The round-12 fold ensemble on the 195 held-out genes at two thresholds: the logit's
  zero, which is the decision boundary the class-weighted loss was trained to, and the
  threshold that calls as many genes essential as the labels hold (31), so precision and
  recall are read at the label prevalence.

It also measures how far the two label sources agree: on the genes both name, the
fraction on which the Stanford inviable list and the released split give the same call.

    python experiments/028-gene-essentiality/scripts/essentiality_confusion.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from fcl_comparison import load_fba_scores, load_fcl_scores, load_labels  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
EXTERNAL = osp.join(RESULTS, "external", "yeast-GEM-9.1.1")
PRED_DIR = osp.join(DATA_ROOT, "test-predictions-028")
FBA_CSV = osp.join(RESULTS, "fba_essentiality_per_gene.csv")
MEDIA = ("shipped", "SM", "SC", "YPD")
KO_TOL = 1e-6
ENSEMBLE_PREFIX = "w23_chromprotcv"


class ConfusionRow(BaseModel):
    """One scorer at one rule on one gene set."""

    scorer: str
    rule: str
    gene_set: str
    n: int
    n_essential: int
    tp: int
    fn: int
    fp: int
    tn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    balanced_accuracy: float
    majority_accuracy: float


def confusion(
    scorer: str, rule: str, gene_set: str, call: np.ndarray, label: np.ndarray
) -> ConfusionRow:
    """Essential is the positive class; ``call`` and ``label`` are 0/1 arrays."""
    tp = int(((call == 1) & (label == 1)).sum())
    fn = int(((call == 0) & (label == 1)).sum())
    fp = int(((call == 1) & (label == 0)).sum())
    tn = int(((call == 0) & (label == 0)).sum())
    n = tp + fn + fp + tn
    pos = tp + fn
    return ConfusionRow(
        scorer=scorer,
        rule=rule,
        gene_set=gene_set,
        n=n,
        n_essential=pos,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        accuracy=(tp + tn) / n,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / pos if pos else 0.0,
        specificity=tn / (tn + fp) if tn + fp else 0.0,
        balanced_accuracy=0.5 * ((tp / pos if pos else 0.0) + (tn / (tn + fp) if tn + fp else 0.0)),
        majority_accuracy=max(pos, n - pos) / n,
    )


def yeast_gem_rows() -> tuple[ConfusionRow, dict[str, int]]:
    """yeast-GEM 9.1.1's own confusion matrix, and its experimental label per gene."""
    table = pd.read_csv(osp.join(EXTERNAL, "essentialGenes.tsv"), sep="\t")
    # Their TP is 'true non-essential' (both viable), TN 'true essential' (both
    # inviable), FP 'false non-essential' (model viable, experiment inviable), FN 'false
    # essential' (model inviable, experiment viable).
    exp_inviable = table["classification"].isin(["TN", "FP"]).to_numpy().astype(int)
    model_inviable = table["classification"].isin(["TN", "FN"]).to_numpy().astype(int)
    row = confusion(
        "yeast-GEM 9.1.1 (its own benchmark)",
        "growth ratio < 1e-6, Kennedy SC medium",
        "1,107 verified ORFs, Stanford inviable list",
        model_inviable,
        exp_inviable,
    )
    return row, dict(zip(table["gene"], exp_inviable.tolist(), strict=True))


def ensemble_scores() -> pd.Series:
    """The round-12 fold ensemble's mean logit per held-out gene."""
    frames = []
    for f in sorted(glob.glob(osp.join(PRED_DIR, f"{ENSEMBLE_PREFIX}*_s*.json"))):
        with open(f) as fh:
            preds = json.load(fh)["predictions"]
        frames.append(
            pd.Series({p["genes"][0]: p["pred"] for p in preds if p["pinned"]})
        )
    assert len(frames) == 5, f"expected five fold dumps, found {len(frames)}."
    return pd.concat(frames, axis=1).mean(axis=1)


def main() -> None:
    labels = load_labels()
    held = labels[labels["held_out"] == 1]
    rows: list[ConfusionRow] = []
    gem_row, gem_label = yeast_gem_rows()
    rows.append(gem_row)

    # FBA on 9.0.2 at the same rule, on the released labels.
    with open(FBA_CSV, newline="") as fh:
        fba = {r["gene"]: {m: float(r[m]) for m in MEDIA} for r in csv.DictReader(fh)}
    for medium in MEDIA:
        for name, frame in (("1,121 labeled", labels), ("195 held-out", held)):
            genes = [g for g in frame.index if g in fba]
            if name == "195 held-out":
                genes = [g for g in genes if g in set(ours_195(fba))]
            call = np.array([int(fba[g][medium] < KO_TOL) for g in genes])
            label = frame.loc[genes, "is_essential"].to_numpy()
            rows.append(
                confusion(
                    f"FBA, yeast-GEM 9.0.2, {medium}",
                    "growth ratio < 1e-6",
                    f"{name} (released labels)",
                    call,
                    label,
                )
            )

    # The released FCL predictions at P(viable) < 0.5.
    fcl = load_fcl_scores()  # negative P(viable)
    for name, genes in (
        ("223 released held-out", [g for g in held.index if g in fcl.index]),
        ("195 held-out", [g for g in ours_195(fba) if g in fcl.index]),
    ):
        call = np.array([int(-fcl[g] < 0.5) for g in genes])
        label = held.loc[genes, "is_essential"].to_numpy()
        rows.append(
            confusion(
                "Flux Cone Learning (released predictions)",
                "P(viable) < 0.5",
                f"{name} (released labels)",
                call,
                label,
            )
        )
    # The released FBA predictions the paper used, at growth < 1e-6.
    fba_rel = load_fba_scores()  # negative growth
    genes = [g for g in ours_195(fba) if g in fba_rel.index]
    call = np.array([int(-fba_rel[g] < KO_TOL) for g in genes])
    rows.append(
        confusion(
            "FBA (released predictions)",
            "growth < 1e-6",
            "195 held-out (released labels)",
            call,
            held.loc[genes, "is_essential"].to_numpy(),
        )
    )

    # The round-12 ensemble on the 195.
    ens = ensemble_scores()
    genes = list(ens.index)
    label = held.loc[genes, "is_essential"].to_numpy()
    logit = ens.to_numpy()
    rows.append(
        confusion(
            "transformer, round-12 fold ensemble",
            "logit > 0",
            "195 held-out (released labels)",
            (logit > 0).astype(int),
            label,
        )
    )
    k = int(label.sum())
    cut = np.sort(logit)[::-1][k - 1]
    rows.append(
        confusion(
            "transformer, round-12 fold ensemble",
            f"top {k} logits (label prevalence)",
            "195 held-out (released labels)",
            (logit >= cut).astype(int),
            label,
        )
    )

    # Label agreement between the two sources on the genes both name.
    both = [g for g in labels.index if g in gem_label]
    ours = labels.loc[both, "is_essential"].to_numpy()
    theirs = np.array([gem_label[g] for g in both])
    agreement = {
        "n_genes_in_both": len(both),
        "n_released_labeled": int(len(labels)),
        "n_yeast_gem_benchmark": int(len(gem_label)),
        "agree_frac": float((ours == theirs).mean()),
        "released_essential_stanford_viable": int(((ours == 1) & (theirs == 0)).sum()),
        "released_viable_stanford_inviable": int(((ours == 0) & (theirs == 1)).sum()),
    }
    out = {
        "note": (
            "essential is the positive class; TP/FN/FP/TN are in that convention (the "
            "yeast-GEM file's own convention is inverted and is mapped). Majority accuracy "
            "is the larger class's share of the gene set scored."
        ),
        "rows": [r.model_dump() for r in rows],
        "label_agreement": agreement,
        "yeast_gem_source": osp.join(EXTERNAL, "provenance.json"),
    }
    path = osp.join(RESULTS, "essentiality_confusion.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    for r in rows:
        print(
            f"{r.scorer:44s} {r.rule:36s} {r.gene_set:36s} n={r.n:5d} ess={r.n_essential:3d} "
            f"acc {r.accuracy:.3f} prec {r.precision:.3f} rec {r.recall:.3f} "
            f"spec {r.specificity:.3f} bal {r.balanced_accuracy:.3f} maj {r.majority_accuracy:.3f}"
        )
    print(agreement)
    print(path)


def ours_195(fba: dict[str, dict[str, float]]) -> list[str]:
    """The 195 held-out genes the model scores, from the FBA table's flag."""
    with open(FBA_CSV, newline="") as fh:
        return [r["gene"] for r in csv.DictReader(fh) if r["ours_195"] == "1"]


if __name__ == "__main__":
    main()
