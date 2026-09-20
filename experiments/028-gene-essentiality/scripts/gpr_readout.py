# experiments/028-gene-essentiality/scripts/gpr_readout.py
# [[experiments.028-gene-essentiality.scripts.gpr_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/gpr_readout

"""Round 77: six gene-protein-reaction redundancy columns beside the solve vector.

The reads were fixed in the pack files at submission (``conf/packs/w96_*.pack``):

- on the released split, six seeds of the cell with the columns (``w96_pathloggpr``)
  against the same seeds of the round-33 cell without them (``w49_pathfbapfbalogfree``):
  the seed-mean held-out AUROC and the paired difference;
- on the same cells, the mean misrank of the 19 held-out essential genes whose FBA
  knockout growth ratio is 1.0, against 0.134 without the columns, and of the four of
  them an alternative gene set covers on every reaction against the fifteen holding a
  reaction of their own (``isozyme_affine_store.json``);
- on the ten repeated outer splits, three seeds each, the per-split seed-mean AUROC paired
  against the same cell without the columns (``w93``): mean, standard deviation and sign
  count. The test blocks overlap, so no p-value.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/gpr_readout.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
PRED_DIR = os.getenv("TC_TEST_PRED_DIR", osp.join(DATA_ROOT, "test-predictions-028"))
SEEDS = (2700, 2701, 2702, 2703, 2704, 2705)
N_SPLITS = 10


class Paired(BaseModel):
    """One paired comparison: with the columns minus without."""

    name: str
    with_columns: list[float]
    without_columns: list[float]
    mean_with: float
    mean_without: float
    mean_difference: float
    sd_difference: float
    n_positive: int
    n: int


class Report(BaseModel):
    """The whole read."""

    released_auroc: Paired
    released_misrank_fba_viable_essential: Paired
    released_misrank_covered_by_alternatives: Paired
    released_misrank_sole_reaction: Paired
    resplit_auroc: Paired
    note: str


def paired(name: str, a: list[float], b: list[float]) -> Paired:
    d = np.array(a) - np.array(b)
    return Paired(
        name=name,
        with_columns=a,
        without_columns=b,
        mean_with=float(np.mean(a)),
        mean_without=float(np.mean(b)),
        mean_difference=float(d.mean()),
        sd_difference=float(d.std(ddof=1)),
        n_positive=int((d > 0).sum()),
        n=len(d),
    )


def auroc(stem: str) -> float:
    with open(osp.join(RESULTS, f"{stem}.json")) as fh:
        return float(json.load(fh)["test"]["pinned"]["auroc"])


def essential_misrank(stem: str, genes: list[str]) -> float:
    """Mean over ``genes`` of the fraction of nonessential genes scoring above each."""
    with open(osp.join(RESULTS, f"{stem}.json")) as fh:
        dump_path = json.load(fh)["test"]["dump_path"]
    if not osp.isfile(dump_path):
        dump_path = osp.join(PRED_DIR, osp.basename(dump_path))
    with open(dump_path) as fh:
        preds = [p for p in json.load(fh)["predictions"] if p.get("pinned", True)]
    score = {p["genes"][0]: float(p["pred"]) for p in preds}
    negatives = np.array(
        [float(p["pred"]) for p in preds if int(round(float(p["target"]))) == 0]
    )
    return float(np.mean([np.mean(negatives > score[g]) for g in genes]))


def main() -> None:
    with open(osp.join(RESULTS, "isozyme_affine_store.json")) as fh:
        essential = json.load(fh)["held_out_essential"]
    viable = [r for r in essential if r["ratio_ypd"] >= 0.999]
    groups = {
        "viable": [r["gene"] for r in viable],
        "covered": [r["gene"] for r in viable if r["gpr_n_sole"] == 0],
        "sole": [r["gene"] for r in viable if r["gpr_n_sole"] > 0],
    }
    assert len(groups["viable"]) == 19, len(groups["viable"])
    with_stem = "w96_pathloggpr_h12l1p1_e200_s{s}"
    without_stem = "w49_pathfbapfbalogfree_h12l1p1_e200_s{s}"

    def misrank_pair(name: str, genes: list[str]) -> Paired:
        return paired(
            name,
            [essential_misrank(with_stem.format(s=s), genes) for s in SEEDS],
            [essential_misrank(without_stem.format(s=s), genes) for s in SEEDS],
        )

    def split_mean(pattern: str) -> float:
        paths = sorted(glob.glob(osp.join(RESULTS, pattern)))
        assert len(paths) == 3, (pattern, len(paths))
        return float(np.mean([auroc(osp.basename(p)[: -len(".json")]) for p in paths]))

    report = Report(
        released_auroc=paired(
            "released split, held-out AUROC per seed",
            [auroc(with_stem.format(s=s)) for s in SEEDS],
            [auroc(without_stem.format(s=s)) for s in SEEDS],
        ),
        released_misrank_fba_viable_essential=misrank_pair(
            "mean misrank, 19 essential genes FBA calls viable", groups["viable"]
        ),
        released_misrank_covered_by_alternatives=misrank_pair(
            "mean misrank, the 4 covered by an alternative gene set", groups["covered"]
        ),
        released_misrank_sole_reaction=misrank_pair(
            "mean misrank, the 15 holding a reaction of their own", groups["sole"]
        ),
        resplit_auroc=paired(
            "ten repeated outer splits, seed-mean AUROC per split",
            [
                split_mean(f"w96_resplit{r}pathloggpr_h12l1p1_e200_s*.json")
                for r in range(N_SPLITS)
            ],
            [
                split_mean(f"w93_resplit{r}pathlogfree_h12l1p1_e200_s*.json")
                for r in range(N_SPLITS)
            ],
        ),
        note=(
            "Differences are with the columns minus without. For a misrank a negative "
            "difference is an improvement. Repeated-split test blocks overlap, so the "
            "differences are not independent and no p-value is reported."
        ),
    )
    for p in report.model_dump().values():
        if isinstance(p, dict):
            print(
                f"{p['name']:62s} with {p['mean_with']:.4f} without {p['mean_without']:.4f} "
                f"diff {p['mean_difference']:+.4f} sd {p['sd_difference']:.4f} "
                f"positive {p['n_positive']}/{p['n']}"
            )
    out = osp.join(RESULTS, "gpr_readout.json")
    with open(out, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    print(out)


if __name__ == "__main__":
    main()
