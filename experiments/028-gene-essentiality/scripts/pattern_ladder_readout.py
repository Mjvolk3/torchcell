# experiments/028-gene-essentiality/scripts/pattern_ladder_readout.py
# [[experiments.028-gene-essentiality.scripts.pattern_ladder_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/pattern_ladder_readout

r"""The layer-pattern ladder and the propagated-label stores on essentiality, paired by seed.

The 019 KL regularizer and the 020 hard mask both index the second transformer block,
and every cell of the study ran one block, so no essentiality cell had read the nine
graphs (``graph_reg_loss`` 0.0 at every epoch). The ladder (packs w134 to w138,
2026-09-27) reads them on the round-33 incumbent recipe at width 36 with nine heads:
one unmasked block (the width control), two unmasked blocks (the depth control), two
blocks with the regularizer on the second, two with the hard mask on the second, and
three with the hard mask on the second and third. Beside it, the graphs' one-hop label
propagation of essentiality (``graph_label_propagation.py --write-store``) as the
affine store, alone (w139) and appended to the pFBA vector (w140), on the width-12
incumbent recipe.

Every pair is arm against control on the same six seeds: held-out AUROC at the
validation-selected epoch, the paired difference, and the count of seeds on which the
arm is above. A pair whose cells have not all landed is reported as pending. Writes
``results/pattern_ladder_readout.json``. Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/pattern_ladder_readout.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import time

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SEEDS = list(range(2700, 2706))

STEMS = {
    "w49": "w49_pathfbapfbalogfree_h12l1p1_e200",
    "w134": "w134_pathlogw36_h36l1p1_e200",
    "w135": "w135_pathlogss_h36l2p1_e200",
    "w136": "w136_pathlogskl_h36l2p1_e200",
    "w137": "w137_pathlogsmask_h36l2p1_e200",
    "w138": "w138_pathlogsmm_h36l3p1_e200",
    "w139": "w139_pathgraphlabel_h12l1p1_e200",
    "w140": "w140_pathpfbagraphlabel_h12l1p1_e200",
}
PAIRS = [
    ("width 36, one block, against the width-12 incumbent", "w134", "w49"),
    ("two unmasked blocks against one", "w135", "w134"),
    ("KL regularizer on the second block against two unmasked blocks", "w136", "w135"),
    ("hard mask on the second block against two unmasked blocks", "w137", "w135"),
    ("hard mask on the second and third blocks against on the second", "w138", "w137"),
    ("hard mask on the second and third blocks against one unmasked block", "w138", "w134"),
    ("the graphs' label-propagation store against the pFBA vector", "w139", "w49"),
    ("pFBA with the label-propagation store appended against pFBA alone", "w140", "w49"),
    ("pFBA with the label-propagation store appended against the store alone", "w140", "w139"),
]


class Cell(BaseModel):
    seed: int
    auroc: float
    select_epoch: int
    n_parameters: int
    max_graph_reg_loss: float


class Arm(BaseModel):
    pack: str
    stem: str
    cells: list[Cell]
    mean_auroc: float
    sd_auroc: float
    min_auroc: float
    max_auroc: float
    min_select_epoch: int
    max_select_epoch: int
    n_parameters: int
    max_graph_reg_loss: float


class Pair(BaseModel):
    name: str
    arm: Arm
    control: Arm
    n: int
    mean_difference: float
    sd_difference: float
    n_positive: int


def load_arm(pack: str) -> Arm | None:
    stem = STEMS[pack]
    cells = []
    for s in SEEDS:
        path = osp.join(RESULTS, f"{stem}_s{s}.json")
        if not osp.isfile(path):
            return None
        r = json.load(open(path))
        cells.append(
            Cell(
                seed=s,
                auroc=float(r["test"]["pinned"]["auroc"]),
                select_epoch=int(r["select_epoch"]),
                n_parameters=int(r["n_parameters"]["total"]),
                max_graph_reg_loss=float(max(h["graph_reg_loss"] for h in r["history"])),
            )
        )
    a = np.array([c.auroc for c in cells])
    assert len({c.n_parameters for c in cells}) == 1
    return Arm(
        pack=pack,
        stem=stem,
        cells=cells,
        mean_auroc=float(a.mean()),
        sd_auroc=float(a.std(ddof=1)),
        min_auroc=float(a.min()),
        max_auroc=float(a.max()),
        min_select_epoch=min(c.select_epoch for c in cells),
        max_select_epoch=max(c.select_epoch for c in cells),
        n_parameters=cells[0].n_parameters,
        max_graph_reg_loss=max(c.max_graph_reg_loss for c in cells),
    )


def main() -> None:
    started = time.time()
    arms = {p: load_arm(p) for p in STEMS}
    pairs: list[Pair] = []
    pending: list[str] = []
    for name, a, c in PAIRS:
        if arms[a] is None or arms[c] is None:
            pending.append(name)
            continue
        x = np.array([k.auroc for k in arms[a].cells])
        y = np.array([k.auroc for k in arms[c].cells])
        d = x - y
        pairs.append(
            Pair(
                name=name,
                arm=arms[a],
                control=arms[c],
                n=len(d),
                mean_difference=float(d.mean()),
                sd_difference=float(d.std(ddof=1)),
                n_positive=int((d > 0).sum()),
            )
        )
        print(
            f"{name:78s} {arms[a].mean_auroc:.4f} (sd {arms[a].sd_auroc:.4f}) vs "
            f"{arms[c].mean_auroc:.4f}: {d.mean():+.4f} sd {d.std(ddof=1):.4f}, "
            f"{int((d > 0).sum())}/{len(d)} above; reg loss max {arms[a].max_graph_reg_loss:.4g}",
            flush=True,
        )
    for name in pending:
        print(f"{name:78s} pending", flush=True)
    out = {
        "generated_by": "experiments/028-gene-essentiality/scripts/pattern_ladder_readout.py",
        "statistic": (
            "held-out AUROC on the released split at the validation-selected epoch, six "
            "seeds, arm against control paired by seed"
        ),
        "arms": {p: (a.model_dump() if a else None) for p, a in arms.items()},
        "pairs": [p.model_dump() for p in pairs],
        "pending": pending,
        "seconds": time.time() - started,
    }
    path = osp.join(RESULTS, "pattern_ladder_readout.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(path)


if __name__ == "__main__":
    main()
