# experiments/027-betaxanthin-metabolic/scripts/essentiality_contrast.py
# [[experiments.027-betaxanthin-metabolic.scripts.essentiality_contrast]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/essentiality_contrast
"""Gene essentiality: the pooled backbone against the biomass flux readout, paired by seed.

Released Merzbacher FCL split, 195 pinned test genes of which 31 are essential, 80 epochs,
seeds 2700 to 2702, selection on validation. The two arms are byte-identical apart from
the module: `pooled` has no flux layer and reads the perturbed-gene context; `flux` reads
the biomass column r_2111 of the anchored projection head's flux vector.

Two readings are kept apart, because the first seed pair conflated them:

* **The paired AUROC difference**, flux minus pooled, per seed and on average. At three
  seeds all agreeing in sign is p = 0.25 under the null, so the sign is a direction.
* **Which cells collapsed.** A cell whose validation prediction spread ratio is below
  ``COLLAPSE_SD_RATIO`` at its selected epoch predicts a constant; its AUROC is a
  tie-broken rank of noise and its accuracy equals the majority-class rate 0.841. The
  contrast is reported on all seeds and again on the seeds where NEITHER arm collapsed.

Published comparators on this split (Merzbacher 2023): FCL 0.73, FBA 0.69.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/essentiality_contrast.py

Writes ``results/essentiality_contrast.json``.
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
OUT = osp.join(RESULTS, "essentiality_contrast.json")
SEEDS = (2700, 2701, 2702)
ARMS = {
    "pooled": "wave9_ess_pool_e80_s{seed}.json",
    "flux": "wave9_ess_flux_e80_s{seed}.json",
}
REFERENCE = "pooled"
#: Below this validation prediction spread ratio at the selected epoch a cell is a
#: constant predictor. The collapsed cells read 0.0000 to 0.0005; the live ones 0.79 to 3.1.
COLLAPSE_SD_RATIO = 0.05
FCL_TEST_AUROC = 0.73
FBA_AUROC = 0.69


class CellResult(BaseModel):
    """One cell's pinned-test essentiality block at its selected epoch."""

    model_config = ConfigDict(extra="forbid")
    arm: str
    seed: int
    select_epoch: int
    auroc: float
    accuracy: float
    majority_class_rate: float
    recall_essential: float
    val_pred_sd_ratio: float
    collapsed: bool
    wandb_run_id: str | None
    wandb_url: str | None


class PairedContrast(BaseModel):
    """Flux minus pooled AUROC, differenced within each seed."""

    model_config = ConfigDict(extra="forbid")
    seeds: list[int]
    difference_by_seed: dict[str, float]
    mean_difference: float
    sd_difference: float
    seeds_agreeing_in_sign: int


class EssentialityContrastReport(BaseModel):
    """Every cell, the paired contrast on all seeds and on the uncollapsed seeds."""

    model_config = ConfigDict(extra="forbid")
    endpoint: str = (
        "pinned-test AUROC on the released FCL split, 195 genes, 31 essential"
    )
    cells: list[CellResult]
    all_seeds: PairedContrast
    uncollapsed_seeds: PairedContrast | None = Field(
        description="The same contrast on seeds where neither arm collapsed; None if no seed qualifies."
    )
    collapsed_cells: list[str]
    comparators: dict[str, float] = {"fcl_test": FCL_TEST_AUROC, "fba": FBA_AUROC}
    cells_above_fcl: list[str]
    note: str


def read_cell(arm: str, seed: int) -> CellResult:
    with open(osp.join(RESULTS, ARMS[arm].format(seed=seed))) as fh:
        d = json.load(fh)
    pinned = d["test"]["pinned"]
    row = next(e for e in d["history"] if e["epoch"] == d["select_epoch"])
    ratio = float(row["val_pred_sd_ratio"])
    return CellResult(
        arm=arm,
        seed=seed,
        select_epoch=int(d["select_epoch"]),
        auroc=float(pinned["auroc"]),
        accuracy=float(pinned["accuracy"]),
        majority_class_rate=float(pinned["majority_class_rate"]),
        recall_essential=float(pinned["recall_essential"]),
        val_pred_sd_ratio=ratio,
        collapsed=ratio < COLLAPSE_SD_RATIO,
        wandb_run_id=d.get("wandb_run_id"),
        wandb_url=d.get("wandb_url"),
    )


def contrast(cells: list[CellResult], seeds: list[int]) -> PairedContrast | None:
    if not seeds:
        return None
    by = {(c.arm, c.seed): c.auroc for c in cells}
    diffs = {str(s): by[("flux", s)] - by[(REFERENCE, s)] for s in seeds}
    values = np.array(list(diffs.values()), dtype=float)
    return PairedContrast(
        seeds=seeds,
        difference_by_seed=diffs,
        mean_difference=float(values.mean()),
        sd_difference=float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        seeds_agreeing_in_sign=int(max((values > 0).sum(), (values < 0).sum())),
    )


def main() -> None:
    cells = [read_cell(arm, seed) for arm in ARMS for seed in SEEDS]
    collapsed = [f"{c.arm}_s{c.seed}" for c in cells if c.collapsed]
    clean = [s for s in SEEDS if not any(c.collapsed for c in cells if c.seed == s)]
    report = EssentialityContrastReport(
        cells=cells,
        all_seeds=contrast(cells, list(SEEDS)),
        uncollapsed_seeds=contrast(cells, clean),
        collapsed_cells=collapsed,
        cells_above_fcl=[
            f"{c.arm}_s{c.seed}" for c in cells if c.auroc > FCL_TEST_AUROC
        ],
        note=(
            "A collapsed cell (spread ratio below 0.05) predicts a constant: its accuracy "
            "equals the majority rate and its AUROC ranks noise. Collapse is not confined "
            "to the flux arm on this target: pooled seed 2702 collapsed as well. Three "
            "seeds agreeing in sign is p = 0.25 under the null."
        ),
    )
    with open(OUT, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    for c in cells:
        flag = " COLLAPSED" if c.collapsed else ""
        print(
            f"{c.arm:7s} s{c.seed} ep {c.select_epoch:3d} auroc {c.auroc:.4f} "
            f"acc {c.accuracy:.4f} recall_ess {c.recall_essential:.3f} "
            f"sd_ratio {c.val_pred_sd_ratio:.4f}{flag}"
        )
    for label, block in (
        ("all seeds", report.all_seeds),
        ("uncollapsed seeds", report.uncollapsed_seeds),
    ):
        if block is None:
            print(f"{label}: none")
            continue
        per = " ".join(f"{s}:{v:+.4f}" for s, v in block.difference_by_seed.items())
        print(
            f"{label}: flux minus pooled mean {block.mean_difference:+.4f} "
            f"sd {block.sd_difference:.4f} sign {block.seeds_agreeing_in_sign}/{len(block.seeds)} | {per}"
        )
    print("above FCL 0.73:", report.cells_above_fcl)
    print(OUT)


if __name__ == "__main__":
    main()
