# experiments/027-betaxanthin-metabolic/scripts/analyze_wave9_ib_external.py
# [[experiments.027-betaxanthin-metabolic.wave9-isobutanol]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/analyze_wave9_ib_external.py

r"""The isobutanol EXTERNAL DIRECTION CHECK, and the wave-9 three-way summary.

TWO QUESTIONS, BOTH PRE-REGISTERED BEFORE THE CELLS RAN.

1. **The three-way.** ``ibprec - ibpool`` and ``ibscalar - ibpool``, paired by seed, on the
   pinned Merzbacher test genes, against the noise floor the three ``ibpool`` seeds
   establish themselves. The scoring rule is the 027 one and is not re-derived here: the
   number in each result file is the pinned-test Spearman at the epoch selected by the
   centered 5-epoch moving average of validation Pearson.

2. **The external check.** Each cell holds the 224-strain validated re-screen out of its
   loss, its validation curve and its test score, and dumps its prediction for every one of
   those strains (``isobutanol_external_check`` in the result file). This script joins those
   predictions to the re-screen's own fold changes and asks what fraction of the predicted
   top 30 reproduced at FC >= 2 and of the predicted bottom 30 at FC <= 0.5.

   The ceiling those fractions are read against is the re-screen's own reproducibility over
   the 222 genes measured in both rounds: 59 of 60 screen-up strains stayed at FC >= 2 and
   150 of 150 screen-down strains stayed at FC <= 0.5. A random 30 of the panel would
   reproduce at the panel's own base rate, which is reported beside every fraction, because
   the panel is SELECTED on |log FC| and its base rates are nothing like the screen's.

WHY THE JOIN LIVES HERE RATHER THAN IN THE TRAINER. The re-screen's fold change is not
recoverable from the training build: ``MeanExperimentDeduplicator`` averaged it into the
screen's value on 208 of the 222 co-measured genotypes, so the pure n=3 number has to come
from ``IsobutanolValidatedLopez2024Dataset`` itself. Keeping that dependency out of the
trainer is what stops a 3-hour GPU cell dying at its last step because a dataset build
moved, which is live: both Lopez LMDBs were rebuilt on 2026-09-14.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/analyze_wave9_ib_external.py
"""

import json
import os
import os.path as osp
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")

#: The paper's own hit thresholds, which are also the re-screen's reporting blocks.
FC_UP = 2.0
FC_DOWN = 0.5
TOP_K = 30
SEEDS = (2700, 2701, 2702)
ARMS = ("ibpool", "ibscalar", "ibprec")


def validated_fold_change() -> dict[str, float]:
    """``{systematic ORF -> validated n=3 mean fold change}`` from the Lopez re-screen.

    Read from the dataset class rather than from the LMDB bytes, so the record schema is
    the one the loader asserts and a build that moved is an error rather than a silent
    mismatch. The LMDB is never rebuilt by this call: the dataset base class skips
    ``process()`` when ``processed/`` exists.
    """
    from torchcell.datasets.scerevisiae.lopez2024 import (
        IsobutanolValidatedLopez2024Dataset,
    )
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    root = osp.join(DATA_ROOT, "data/torchcell/isobutanol_validated_lopez2024")
    dataset = IsobutanolValidatedLopez2024Dataset(root=root, genome=genome)
    out: dict[str, float] = {}
    for i in range(len(dataset)):
        record = dataset[i]
        experiment = record["experiment"]
        genes = [
            p.systematic_gene_name
            for p in experiment.genotype.perturbations
            if p.perturbation_type.endswith("deletion")
        ]
        assert len(genes) == 1, (
            f"validated record {i} has {len(genes)} deletions; the panel is single-gene."
        )
        levels = experiment.phenotype.metabolite_level
        assert set(levels) == {"isobutanol"}, (
            f"validated record {i} carries {sorted(levels)}, expected only isobutanol."
        )
        out[str(genes[0])] = float(levels["isobutanol"])
    print(f"[validated] {len(out)} strains read from {root}", flush=True)
    return out


def load_result(arm: str, seed: int) -> dict[str, Any] | None:
    path = osp.join(RESULTS_DIR, f"wave9_{arm}_e200_s{seed}.json")
    if not osp.exists(path):
        return None
    with open(path) as fh:
        return dict(json.load(fh))


def external_block(
    result: dict[str, Any], validated: dict[str, float]
) -> dict[str, Any]:
    """Fractions of the predicted top and bottom 30 that reproduced in the re-screen."""
    check = result["isobutanol_external_check"]
    rows: list[tuple[str, float, float]] = []
    for row in check["predictions"]:
        for gene in row["genes"]:
            if gene in validated:
                rows.append(
                    (gene, float(row["pred_log10_fold_change"]), validated[gene])
                )
    assert rows, "no predicted strain joined the validated panel."
    rows.sort(key=lambda r: r[1], reverse=True)
    top = rows[:TOP_K]
    bottom = rows[-TOP_K:]
    measured = np.array([r[2] for r in rows])
    predicted = np.array([r[1] for r in rows])
    truth = np.log10(measured)
    return {
        "n_joined": len(rows),
        "top_k": TOP_K,
        "top_reproduced_up": sum(1 for _, _, fc in top if fc >= FC_UP),
        "bottom_reproduced_down": sum(1 for _, _, fc in bottom if fc <= FC_DOWN),
        "panel_base_rate_up": float((measured >= FC_UP).mean()),
        "panel_base_rate_down": float((measured <= FC_DOWN).mean()),
        "expected_top_up_at_base_rate": float((measured >= FC_UP).mean() * TOP_K),
        "expected_bottom_down_at_base_rate": float(
            (measured <= FC_DOWN).mean() * TOP_K
        ),
        "spearman_on_panel": float(
            np.corrcoef(
                np.argsort(np.argsort(predicted)), np.argsort(np.argsort(truth))
            )[0, 1]
        ),
        "note": (
            "the panel is SELECTED on |log FC|, so its base rates are far above the "
            "screen's 5.5 percent up / 6.8 percent down; the expected counts are what a "
            "random 30 of the panel would give, and the empirical ceilings are 59/60 up "
            "and 150/150 down."
        ),
    }


def main() -> None:
    scores: dict[str, dict[int, float]] = {arm: {} for arm in ARMS}
    externals: dict[str, dict[int, Any]] = {arm: {} for arm in ARMS}
    validated = validated_fold_change()
    for arm in ARMS:
        for seed in SEEDS:
            result = load_result(arm, seed)
            if result is None:
                print(f"[missing] wave9_{arm}_e200_s{seed}.json", flush=True)
                continue
            scores[arm][seed] = float(result["test"]["pinned"]["spearman"])
            externals[arm][seed] = external_block(result, validated)

    summary: dict[str, Any] = {
        "scoring_rule": (
            "pinned-test Spearman on the held-out Merzbacher genes at the epoch selected "
            "by the centered 5-epoch moving average of validation Pearson; never a "
            "maximum over epochs."
        ),
        "scores": {arm: scores[arm] for arm in ARMS},
        "external": externals,
        "contrasts": {},
    }
    for arm in ("ibscalar", "ibprec"):
        paired = [
            scores[arm][s] - scores["ibpool"][s]
            for s in SEEDS
            if s in scores[arm] and s in scores["ibpool"]
        ]
        summary["contrasts"][f"{arm}_minus_ibpool"] = {
            "per_seed": paired,
            "mean": float(np.mean(paired)) if paired else float("nan"),
            "sd": float(np.std(paired, ddof=1)) if len(paired) > 1 else float("nan"),
            "n": len(paired),
        }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = osp.join(RESULTS_DIR, "wave9_ib_summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary["contrasts"], indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
