# experiments/028-gene-essentiality/scripts/cls_readout.py
# [[experiments.028-gene-essentiality.scripts.cls_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/cls_readout

"""Round 79: the class token through the perturbation operator, paired by seed.

Until the port of 2026-09-20 the class token was sliced off before the perturbation
operator, so it encoded the unperturbed graph and was the same vector for every strain.
Every pooled readout in this document therefore carried the genotype through the
perturbed-gene pool alone. With ``--perturb-cls`` the token is query row 0 over the same
deleted-gene keys, no parameters are added, and the gene rows are unchanged.

The read was fixed in the pack files at submission (``conf/packs/w98_*.pack``): two arms,
six seeds each on the released split, each paired by seed against the identical cell
without the flag.

- ``pathlogcls`` against ``w49_pathfbapfbalogfree`` (the round-33 layer-free cell, 0.908);
- ``pathpoolcls`` against ``w39_pathpool`` (the pooled cell with no solve and no affine,
  0.860; three seeds there, so that pair is read on the seeds both arms ran).

Statistics per arm: the seed-mean held-out AUROC with and without, the paired difference
with its standard deviation and sign count, and the across-strain spread of the class
token from the epoch rows (``cls_strain_sd``), which is exactly 0 without the flag.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/cls_readout.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")

#: Arm label -> (stem with the flag, stem without), both formatted with the seed.
ARMS: dict[str, tuple[str, str]] = {
    "layer-free solve vector in the affine": (
        "w98_pathlogcls_h12l1p1_e200_s{s}",
        "w49_pathfbapfbalogfree_h12l1p1_e200_s{s}",
    ),
    "pooled cell, no solve and no affine": (
        "w98_pathpoolcls_h12l1p1_e200_s{s}",
        "w39_pathpool_h12l1p1_e200_s{s}",
    ),
}
SEEDS = (2700, 2701, 2702, 2703, 2704, 2705)


class ArmRead(BaseModel):
    """One arm: the flag on against the same cell with it off, paired by seed."""

    arm: str
    seeds: list[int]
    with_flag: list[float]
    without_flag: list[float]
    mean_with: float
    mean_without: float
    mean_difference: float
    sd_difference: float
    n_positive: int
    n: int
    #: Across-strain spread of the class token at the last epoch, mean over seeds.
    cls_strain_sd_with: float
    cls_strain_sd_without: float


class Report(BaseModel):
    """Every arm, and what the flag costs."""

    arms: list[ArmRead]
    note: str


def result(stem: str) -> dict | None:
    path = osp.join(RESULTS, f"{stem}.json")
    if not osp.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    reads: list[ArmRead] = []
    for arm, (on_stem, off_stem) in ARMS.items():
        seeds, on, off, sd_on, sd_off = [], [], [], [], []
        for s in SEEDS:
            a, b = result(on_stem.format(s=s)), result(off_stem.format(s=s))
            if a is None or b is None:
                continue
            seeds.append(s)
            on.append(float(a["test"]["pinned"]["auroc"]))
            off.append(float(b["test"]["pinned"]["auroc"]))
            sd_on.append(float(a["history"][-1].get("cls_strain_sd", float("nan"))))
            sd_off.append(float(b["history"][-1].get("cls_strain_sd", 0.0)))
        if not seeds:
            print(f"{arm}: no paired cell yet")
            continue
        d = np.array(on) - np.array(off)
        reads.append(
            ArmRead(
                arm=arm,
                seeds=seeds,
                with_flag=on,
                without_flag=off,
                mean_with=float(np.mean(on)),
                mean_without=float(np.mean(off)),
                mean_difference=float(d.mean()),
                sd_difference=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                n_positive=int((d > 0).sum()),
                n=len(d),
                cls_strain_sd_with=float(np.nanmean(sd_on)),
                cls_strain_sd_without=float(np.mean(sd_off)),
            )
        )
        r = reads[-1]
        print(
            f"{arm:42s} with {r.mean_with:.4f} without {r.mean_without:.4f} "
            f"diff {r.mean_difference:+.4f} sd {r.sd_difference:.4f} "
            f"positive {r.n_positive}/{r.n}  cls sd {r.cls_strain_sd_with:.3f} "
            f"against {r.cls_strain_sd_without:.3f}"
        )
    if not reads:
        return
    out = osp.join(RESULTS, "cls_readout.json")
    with open(out, "w") as fh:
        json.dump(
            Report(
                arms=reads,
                note=(
                    "Paired by seed, the same split and the same program; the flag adds "
                    "no parameters. cls_strain_sd is the last epoch row's across-strain "
                    "spread of the class token, exactly 0 without the flag because the "
                    "token is then the encoding of the unperturbed graph. At one deleted "
                    "gene the token's query attends over one key, so what it can add over "
                    "the perturbed-gene pool is the wild-type token beside it."
                ),
            ).model_dump(),
            fh,
            indent=2,
        )
    print(out)


if __name__ == "__main__":
    main()
