# experiments/027-betaxanthin-metabolic/scripts/loss_contrast.py
# [[experiments.027-betaxanthin-metabolic.scripts.loss_contrast]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/loss_contrast
"""Squared error against Gaussian CRPS against CRPS with the spread held at the label's noise.

The three arms are byte-identical apart from the loss: the same context-precursor readout on
the pooled backbone, the same seeds, the same 200-epoch budget, the same selection rule. The
contrast is PAIRED BY SEED, because the seed-to-seed spread on this endpoint is larger than
any effect anyone has claimed on it.

The third arm exists because a learned spread is an escape hatch: widening sigma lowers the
CRPS without fitting anything. Classical test theory fixes the value that is not an escape,
``sqrt(1 - reliability)``, which is 0.40 for betaxanthin at the reliability of 0.84 measured
in ``target_reliability.json``.

Reports the per-seed pinned-test Spearman, the paired difference against CRPS, its mean and
standard deviation, and how many seeds agree in sign. At three seeds a sign that agrees on
all three is p = 0.25 under the null, so the sign column is a direction and not a result.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/loss_contrast.py

Writes ``results/loss_contrast.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
OUT = osp.join(RESULTS, "loss_contrast.json")
SEEDS = (2700, 2701, 2702)
#: arm label -> result file template. The CRPS arm is the reference the others are paired to.
ARMS = {
    "crps_learned_sigma": "wave9_bxctxprec_e200_s{seed}.json",
    "mse": "wave10_bxctxprec_mse_e200_s{seed}.json",
    "crps_sigma_fixed_0.40": "wave10_bxctxprec_sfix040_e200_s{seed}.json",
}
REFERENCE = "crps_learned_sigma"


class ArmResult(BaseModel):
    """One loss arm's per-seed scores on the pinned test split."""

    model_config = ConfigDict(extra="forbid")
    arm: str
    seeds_present: list[int]
    spearman_by_seed: dict[str, float]
    pearson_by_seed: dict[str, float]
    select_epoch_by_seed: dict[str, int]
    prediction_sd_ratio_by_seed: dict[str, float] = Field(
        description="Validation prediction spread over target spread at the selected epoch; a collapsed arm reads near zero."
    )
    mean_spearman: float
    sd_spearman: float


class PairedContrast(BaseModel):
    """One arm against the reference arm, differenced within each seed."""

    model_config = ConfigDict(extra="forbid")
    arm: str
    against: str
    n_pairs: int
    difference_by_seed: dict[str, float]
    mean_difference: float
    sd_difference: float
    seeds_agreeing_in_sign: int


class LossContrastReport(BaseModel):
    """Every loss arm and every paired contrast on one endpoint."""

    model_config = ConfigDict(extra="forbid")
    endpoint: str = "betaxanthin pinned-test Spearman, context-precursor readout"
    arms: list[ArmResult]
    contrasts: list[PairedContrast]
    note: str


def read_arm(label: str, template: str) -> ArmResult | None:
    spearman: dict[str, float] = {}
    pearson: dict[str, float] = {}
    epochs: dict[str, int] = {}
    sd_ratio: dict[str, float] = {}
    for seed in SEEDS:
        path = osp.join(RESULTS, template.format(seed=seed))
        if not osp.isfile(path):
            continue
        with open(path) as fh:
            d = json.load(fh)
        pinned = d["test"]["pinned"]
        spearman[str(seed)] = float(pinned["spearman"])
        pearson[str(seed)] = float(pinned["pearson"])
        epochs[str(seed)] = int(d["select_epoch"])
        history = d.get("history") or []
        row = next(
            (e for e in history if e.get("epoch") == d["select_epoch"]),
            history[-1] if history else {},
        )
        sd_ratio[str(seed)] = float(row.get("val_pred_sd_ratio", float("nan")))
    if not spearman:
        return None
    values = np.array(list(spearman.values()), dtype=float)
    return ArmResult(
        arm=label,
        seeds_present=[int(s) for s in spearman],
        spearman_by_seed=spearman,
        pearson_by_seed=pearson,
        select_epoch_by_seed=epochs,
        prediction_sd_ratio_by_seed=sd_ratio,
        mean_spearman=float(values.mean()),
        sd_spearman=float(values.std(ddof=1)) if values.size > 1 else float("nan"),
    )


def main() -> None:
    arms = [a for a in (read_arm(k, v) for k, v in ARMS.items()) if a is not None]
    reference = next((a for a in arms if a.arm == REFERENCE), None)
    assert reference is not None, f"the reference arm {REFERENCE} has no result files."
    contrasts: list[PairedContrast] = []
    for arm in arms:
        if arm.arm == REFERENCE:
            continue
        shared = sorted(set(arm.spearman_by_seed) & set(reference.spearman_by_seed))
        diffs = {
            s: arm.spearman_by_seed[s] - reference.spearman_by_seed[s] for s in shared
        }
        values = np.array(list(diffs.values()), dtype=float)
        sign = int(max((values > 0).sum(), (values < 0).sum())) if values.size else 0
        contrasts.append(
            PairedContrast(
                arm=arm.arm,
                against=REFERENCE,
                n_pairs=len(shared),
                difference_by_seed=diffs,
                mean_difference=float(values.mean()) if values.size else float("nan"),
                sd_difference=float(values.std(ddof=1))
                if values.size > 1
                else float("nan"),
                seeds_agreeing_in_sign=sign,
            )
        )
    report = LossContrastReport(
        arms=arms,
        contrasts=contrasts,
        note=(
            "Paired by seed. At three seeds the minimum detectable paired difference on this "
            "endpoint is about 0.03, so a mean difference below that is a direction rather "
            "than a result, and all three agreeing in sign is p = 0.25 under the null."
        ),
    )
    with open(OUT, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    for arm in report.arms:
        per = " ".join(f"{s}:{v:+.4f}" for s, v in arm.spearman_by_seed.items())
        print(
            f"{arm.arm:24s} mean {arm.mean_spearman:+.4f} sd {arm.sd_spearman:.4f} | {per}"
        )
    for c in report.contrasts:
        per = " ".join(f"{s}:{v:+.4f}" for s, v in c.difference_by_seed.items())
        print(
            f"  {c.arm} minus {c.against}: mean {c.mean_difference:+.4f} "
            f"sd {c.sd_difference:.4f} n {c.n_pairs} sign {c.seeds_agreeing_in_sign}/{c.n_pairs} | {per}"
        )
    print(OUT)


if __name__ == "__main__":
    main()
