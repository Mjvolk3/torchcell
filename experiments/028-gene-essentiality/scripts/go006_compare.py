# experiments/028-gene-essentiality/scripts/go006_compare.py
# [[experiments.028-gene-essentiality.scripts.go006_compare]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/go006_compare

r"""The two GO models on the experiment 006 GO against the same models on 028's GO.

Reads the result files of ``graph_baselines.py --arms go400`` and of
``dcell_style_baseline.py`` (arm ``dcell_style_shared``) under the two GO stores
(``go006_store.py`` writes the 006 one) and tabulates, per model:

- the released split: seed AUROCs, seed mean and sd (population sd over the seeds, as the
  result files carry it). The committed DCell-style released read has six seeds and the
  006 read three, so the current-GO numbers are given over all of its seeds and over the
  seeds the two reads share;
- the ten repeated outer splits: the per-split seed mean under each GO, the mean over the
  splits, and the PAIRED per-split difference (006 GO minus current GO) with its mean,
  its sd over the splits (ddof 1) and the number of splits higher and lower. No p-value,
  because the test blocks of the repeated splits overlap.

Writes ``results/go006_comparison.json``.

    PYTHONPATH=$PWD python experiments/028-gene-essentiality/scripts/go006_compare.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
N_SPLITS = 10
GCN_INPUTS = (
    "GO (400 epochs)",
    "GO+FBA ratios (400 epochs)",
    "ProtT5+GO+FBA ratios (400 epochs)",
)
DCELL_ARM = "dcell_style_shared"


class Released(BaseModel):
    """One model on the released split under the two GO stores."""

    model_config = ConfigDict(extra="forbid")
    file_006: str
    file_current: str
    seeds_006: list[int]
    seeds_current: list[int]
    auroc_006: list[float]
    auroc_current: list[float]
    mean_006: float
    sd_006: float
    mean_current: float
    sd_current: float
    shared_seeds: list[int]
    mean_current_shared_seeds: float
    sd_current_shared_seeds: float
    difference_shared_seeds: float


class SplitPair(BaseModel):
    """One repeated outer split: the seed mean under each GO."""

    model_config = ConfigDict(extra="forbid")
    split: int
    seeds_006: list[int]
    seeds_current: list[int]
    mean_006: float
    mean_current: float
    difference: float


class Resplits(BaseModel):
    """One model on the ten repeated outer splits under the two GO stores."""

    model_config = ConfigDict(extra="forbid")
    files_006: list[str]
    files_current: list[str]
    n_splits: int
    mean_006: float
    sd_006: float
    mean_current: float
    sd_current: float
    difference_mean: float
    difference_sd: float
    n_006_higher: int
    n_006_lower: int
    splits: list[SplitPair]


class ModelComparison(BaseModel):
    """One model row: released split and repeated splits."""

    model_config = ConfigDict(extra="forbid")
    model: str
    inputs: str
    released: Released
    resplits: Resplits


class Comparison(BaseModel):
    """``results/go006_comparison.json``."""

    model_config = ConfigDict(extra="forbid")
    statistic: str
    models: list[ModelComparison]


def load(name: str) -> dict:
    """One result file under ``results``."""
    with open(osp.join(RESULTS, name)) as fh:
        return dict(json.load(fh))


def gcn_row(name: str, inputs: str, seeds: list[int]) -> tuple[list[int], list[float]]:
    """The seeds and held-out AUROCs of one GCN input row; the go400 files carry no
    seed list, and both reads ran the script's default three seeds in order.
    """
    rows = [r for r in load(name)["rows"] if r["inputs"] == inputs]
    assert len(rows) == 1, (name, inputs)
    assert len(rows[0]["test_auroc"]) == len(seeds), name
    return seeds, [float(a) for a in rows[0]["test_auroc"]]


def dcell_row(name: str, split_file: str) -> tuple[list[int], list[float]]:
    """The seeds and held-out AUROCs of the shared DCell-style arm on one split."""
    rows = [
        r
        for r in load(name)["rows"]
        if r["model"] == DCELL_ARM and r["split_file"] == split_file
    ]
    assert len(rows) == 1, (name, split_file)
    return [int(s) for s in rows[0]["seeds"]], [float(a) for a in rows[0]["test_auroc"]]


def released(
    file_006: str,
    file_current: str,
    new: tuple[list[int], list[float]],
    old: tuple[list[int], list[float]],
) -> Released:
    """The released split under the two stores, and on the seeds both reads ran."""
    shared = [s for s in new[0] if s in old[0]]
    assert shared == new[0], "the 006 read has a seed the current read lacks"
    old_shared = [old[1][old[0].index(s)] for s in shared]
    return Released(
        file_006=file_006,
        file_current=file_current,
        seeds_006=new[0],
        seeds_current=old[0],
        auroc_006=new[1],
        auroc_current=old[1],
        mean_006=float(np.mean(new[1])),
        sd_006=float(np.std(new[1])),
        mean_current=float(np.mean(old[1])),
        sd_current=float(np.std(old[1])),
        shared_seeds=shared,
        mean_current_shared_seeds=float(np.mean(old_shared)),
        sd_current_shared_seeds=float(np.std(old_shared)),
        difference_shared_seeds=float(np.mean(new[1]) - np.mean(old_shared)),
    )


def resplits(
    files_006: list[str],
    files_current: list[str],
    new: list[tuple[list[int], list[float]]],
    old: list[tuple[list[int], list[float]]],
) -> Resplits:
    """The paired per-split difference, 006 GO minus current GO."""
    pairs = [
        SplitPair(
            split=r,
            seeds_006=new[r][0],
            seeds_current=old[r][0],
            mean_006=float(np.mean(new[r][1])),
            mean_current=float(np.mean(old[r][1])),
            difference=float(np.mean(new[r][1]) - np.mean(old[r][1])),
        )
        for r in range(N_SPLITS)
    ]
    assert all(p.seeds_006 == p.seeds_current for p in pairs), "seed lists differ"
    diff = np.array([p.difference for p in pairs])
    return Resplits(
        files_006=files_006,
        files_current=files_current,
        n_splits=len(pairs),
        mean_006=float(np.mean([p.mean_006 for p in pairs])),
        sd_006=float(np.std([p.mean_006 for p in pairs], ddof=1)),
        mean_current=float(np.mean([p.mean_current for p in pairs])),
        sd_current=float(np.std([p.mean_current for p in pairs], ddof=1)),
        difference_mean=float(diff.mean()),
        difference_sd=float(diff.std(ddof=1)),
        n_006_higher=int((diff > 0).sum()),
        n_006_lower=int((diff < 0).sum()),
        splits=pairs,
    )


def main() -> None:
    """Tabulate both models and write ``results/go006_comparison.json``."""
    seeds = [2700, 2701, 2702]
    models: list[ModelComparison] = []
    new_files = [f"graph_baselines_go400_fcl_resplit{r}_go006.json" for r in range(10)]
    old_files = [f"graph_baselines_go400_fcl_resplit{r}.json" for r in range(10)]
    for inputs in GCN_INPUTS:
        models.append(
            ModelComparison(
                model="GCN, union of nine",
                inputs=inputs,
                released=released(
                    "graph_baselines_go400_go006.json",
                    "graph_baselines_go400.json",
                    gcn_row("graph_baselines_go400_go006.json", inputs, seeds),
                    gcn_row("graph_baselines_go400.json", inputs, seeds),
                ),
                resplits=resplits(
                    new_files,
                    old_files,
                    [gcn_row(f, inputs, seeds) for f in new_files],
                    [gcn_row(f, inputs, seeds) for f in old_files],
                ),
            )
        )
    split_files = [f"ess_fcl_resplit{r}_val0.2_s0.json.gz" for r in range(N_SPLITS)]
    models.append(
        ModelComparison(
            model=DCELL_ARM,
            inputs="GO",
            released=released(
                "dcell_style_released_go006.json",
                "dcell_style_released.json",
                dcell_row(
                    "dcell_style_released_go006.json", "ess_fcl_val0.2_s0.json.gz"
                ),
                dcell_row("dcell_style_released.json", "ess_fcl_val0.2_s0.json.gz"),
            ),
            resplits=resplits(
                ["dcell_style_resplits_go006.json"],
                ["dcell_style_resplits.json"],
                [dcell_row("dcell_style_resplits_go006.json", f) for f in split_files],
                [dcell_row("dcell_style_resplits.json", f) for f in split_files],
            ),
        )
    )
    report = Comparison(
        statistic=(
            "held-out AUROC at the epoch selected on validation; released: mean and "
            "population sd over seeds; resplits: per-split seed mean, mean and sd "
            "(ddof 1) over the ten splits, paired difference 006 GO minus current GO"
        ),
        models=models,
    )
    out = osp.join(RESULTS, "go006_comparison.json")
    with open(out, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    for m in models:
        rel, res = m.released, m.resplits
        print(
            f"{m.model:20s} {m.inputs:34s} released 006 {rel.mean_006:.4f} sd "
            f"{rel.sd_006:.4f} | current {rel.mean_current_shared_seeds:.4f} sd "
            f"{rel.sd_current_shared_seeds:.4f} (seeds {rel.shared_seeds}) || ten splits "
            f"006 {res.mean_006:.4f} current {res.mean_current:.4f} diff "
            f"{res.difference_mean:+.4f} sd {res.difference_sd:.4f} higher "
            f"{res.n_006_higher}/{res.n_splits}"
        )
    print(out)


if __name__ == "__main__":
    main()
