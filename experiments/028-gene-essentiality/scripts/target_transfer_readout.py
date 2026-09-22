# experiments/028-gene-essentiality/scripts/target_transfer_readout.py
# [[experiments.028-gene-essentiality.scripts.target_transfer_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/target_transfer_readout

r"""The layer-free module carried to betaxanthin and the amino acids: paired reads.

Rounds 117 to 119 train the 028 backbone (pathway multi-hot with ProtT5, width 12) on
the Cachera betaxanthin screen and on Mulleder's 19 amino acids under the 027 protocol
(Merzbacher's 639 genes pinned to test, CRPS, the epoch selected on validation Pearson),
with and without the store affine. Every arm is paired by seed against the control that
shares its backbone and has no affine. The betaxanthin statistic is the pinned-test
Spearman and the top-50 enrichment of high producers; the amino-acid statistic is the
mean per-column Spearman over the 19 amino acids on the pinned genotypes.

The comparators are the numbers 027 recorded: betaxanthin 0.131 (flux precursor
readout, three seeds), its module-off control 0.115, the kNN over ProtT5 0.145, FCL's
RandomForest 0.039; amino acids 0.068 (the best arm), its layer-free control 0.061, the
pooled arm 0.048. They are copied here from the 027 notes as constants and are not
recomputed.

Writes ``results/target_transfer_readout.json``. Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/target_transfer_readout.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
RESULTS = osp.join(os.environ["EXPERIMENT_ROOT"], "028-gene-essentiality", "results")
SEEDS = (2700, 2701, 2702, 2703, 2704, 2705)
#: From notes/experiments.027-betaxanthin-metabolic.*.md (2026-09-14): pinned Spearman.
COMPARATORS = {
    "betaxanthin": {
        "027 flux precursor readout, three seeds": 0.1313,
        "027 module-off context control, three seeds": 0.1152,
        "kNN over ProtT5 of the deleted gene (k 25)": 0.1446,
        "FCL RandomForest (Merzbacher 2025)": 0.0391,
    },
    "mulleder19": {
        "027 best arm (conc_nocstr), three seeds": 0.0677,
        "027 layer-free control (ctxconc), three seeds": 0.0605,
        "027 pooled arm, three seeds": 0.0479,
        "kNN over ProtT5 (job 1911)": 0.0450,
    },
}
#: (label, head, arm stem, control stem); stems take ``_h12l1p1_e200_s<seed>.json``.
PAIRS = [
    (
        "betaxanthin: whole-vector store affine against the control",
        "betaxanthin",
        "w134_bxstore",
        "w134_bxctx",
    ),
    (
        "betaxanthin: two summary columns against the control",
        "betaxanthin",
        "w136_bxratio",
        "w134_bxctx",
    ),
    (
        "betaxanthin: whole vector, affine decayed and slowed, against the control",
        "betaxanthin",
        "w136_bxstorewd",
        "w134_bxctx",
    ),
    (
        "amino acids: whole-vector store affine against the control",
        "mulleder19",
        "w135_aastore",
        "w135_aactx",
    ),
]


class Cell(BaseModel):
    """One cell's held-out read at its selected epoch."""

    seed: int
    select_epoch: int
    select_val_pearson: float
    spearman: float
    pearson: float
    top50_enrichment: float | None = None
    top50_hits: float | None = None


class Arm(BaseModel):
    """One arm over its seeds."""

    stem: str
    cells: list[Cell]
    mean_spearman: float
    sd_spearman: float
    mean_top50_enrichment: float | None = None


class Pair(BaseModel):
    """An arm against its control, paired by seed."""

    name: str
    head: str
    arm: Arm
    control: Arm
    n: int
    mean_difference: float
    sd_difference: float
    n_positive: int
    differences: list[float]


class Report(BaseModel):
    """The readout."""

    comparators: dict[str, dict[str, float]]
    pairs: list[Pair]
    note: str


def read_cell(stem: str, seed: int, head: str) -> Cell | None:
    path = osp.join(RESULTS, f"{stem}_h12l1p1_e200_s{seed}.json")
    if not osp.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    t = d["test"]["pinned"]
    if head == "mulleder19":
        return Cell(
            seed=seed,
            select_epoch=int(d["select_epoch"]),
            select_val_pearson=float(d["select_val_pearson"]),
            spearman=float(t["mean_spearman"]),
            pearson=float(t["mean_pearson"]),
        )
    return Cell(
        seed=seed,
        select_epoch=int(d["select_epoch"]),
        select_val_pearson=float(d["select_val_pearson"]),
        spearman=float(t["spearman"]),
        pearson=float(t["pearson"]),
        top50_enrichment=float(t["top50"]["enrichment"]),
        top50_hits=float(t["top50"]["hits"]),
    )


def read_arm(stem: str, head: str, seeds: tuple[int, ...]) -> Arm:
    cells = [c for s in seeds if (c := read_cell(stem, s, head)) is not None]
    sp = np.array([c.spearman for c in cells]) if cells else np.array([np.nan])
    enr = [c.top50_enrichment for c in cells if c.top50_enrichment is not None]
    return Arm(
        stem=stem,
        cells=cells,
        mean_spearman=float(sp.mean()),
        sd_spearman=float(sp.std(ddof=1)) if len(sp) > 1 else float("nan"),
        mean_top50_enrichment=float(np.mean(enr)) if enr else None,
    )


def main() -> None:
    pairs: list[Pair] = []
    for name, head, arm_stem, ctrl_stem in PAIRS:
        arm = read_arm(arm_stem, head, SEEDS)
        ctrl = read_arm(ctrl_stem, head, SEEDS)
        by_seed = {c.seed: c for c in ctrl.cells}
        diffs = [
            c.spearman - by_seed[c.seed].spearman
            for c in arm.cells
            if c.seed in by_seed
        ]
        if not diffs:
            print(f"{name}: no paired seeds yet")
            continue
        d = np.array(diffs)
        pair = Pair(
            name=name,
            head=head,
            arm=arm,
            control=ctrl,
            n=len(d),
            mean_difference=float(d.mean()),
            sd_difference=float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
            n_positive=int((d > 0).sum()),
            differences=[float(x) for x in d],
        )
        pairs.append(pair)
        print(
            f"{name}: n {pair.n} arm {arm.mean_spearman:.4f} (sd {arm.sd_spearman:.4f}) "
            f"control {ctrl.mean_spearman:.4f} (sd {ctrl.sd_spearman:.4f}) diff "
            f"{pair.mean_difference:+.4f} sd {pair.sd_difference:.4f} positive {pair.n_positive}/{pair.n}"
            + (
                f"; top-50 enrichment {arm.mean_top50_enrichment:.2f} vs {ctrl.mean_top50_enrichment:.2f}"
                if arm.mean_top50_enrichment is not None
                and ctrl.mean_top50_enrichment is not None
                else ""
            )
        )
        for c in arm.cells:
            k = by_seed.get(c.seed)
            print(
                f"   s{c.seed}: arm {c.spearman:+.4f} (ep {c.select_epoch}, val r {c.select_val_pearson:+.3f})"
                + (
                    f" control {k.spearman:+.4f} (ep {k.select_epoch}, val r {k.select_val_pearson:+.3f})"
                    if k
                    else ""
                )
            )
    report = Report(
        comparators=COMPARATORS,
        pairs=pairs,
        note=(
            "Pinned-test Spearman at the validation-selected epoch (centered five-epoch "
            "moving average of validation Pearson), paired by seed; the amino-acid "
            "statistic is the mean per-column Spearman over the 19 amino acids. Comparators "
            "are constants copied from the 027 notes."
        ),
    )
    out = osp.join(RESULTS, "target_transfer_readout.json")
    with open(out, "w") as fh:
        fh.write(report.model_dump_json(indent=2))
    print(out)


if __name__ == "__main__":
    main()
