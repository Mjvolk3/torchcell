# experiments/028-gene-essentiality/scripts/simplification_readout.py
# [[experiments.028-gene-essentiality.scripts.simplification_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/simplification_readout

"""Rounds 84 to 99: what each simplification of the metabolic module costs.

Three simplifications were measured after round 83, each of which makes the module
cheaper or more portable: dropping the three media that are not the labels' own (one
experiment per record), dropping the solved flux vector (no linear program at all), and
dropping rule-count columns. This reads them together so that no two are compared across
different baselines.

The reads were fixed in the pack files at submission (``conf/packs/w103`` to ``w117``):

- ``ladder``: the released split. Per recipe the seed-mean held-out AUROC, its spread, and
  the five-fold ensemble at one seed per fold and, where run, three.
- ``resplit``: the ten repeated outer splits. Per pair of recipes the per-split seed mean
  over the seeds BOTH arms ran, the mean paired difference, its standard deviation and
  sign count. The test blocks overlap, so no p-value.
- ``subsets`` (round 84): the rule counts with and without the sole-reaction column,
  paired by seed against all six and against none.
- ``budget`` (round 97): 400 epochs against 200, paired by seed, with the epoch the
  validation loss selected in each arm. Cells are deterministic given the seed, so a
  longer budget changes the score only if it moves the selected epoch.

An arm with no result file yet is skipped and listed under ``missing``.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/simplification_readout.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import re

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
N_SPLITS = 10

#: key -> (label, media, solver, cell stem, fold-ensemble prefix)
LADDER: dict[str, tuple[str, int, bool, str, str | None]] = {
    "four_solve_counts": (
        "four ratios, flux vector, rule counts",
        4,
        True,
        "w96_pathloggpr",
        "w99_pathloggprcv",
    ),
    "one_solve_counts": (
        "YPD ratio, flux vector, rule counts",
        1,
        True,
        "w102_ypdgpr",
        "w103_ypdgprcv",
    ),
    "four_solve": (
        "four ratios, flux vector",
        4,
        True,
        "w49_pathfbapfbalogfree",
        "w50_pathlogfreecv",
    ),
    "four_counts": (
        "four ratios, rule counts",
        4,
        False,
        "w101_pathgpronly",
        "w101_pathgpronlycv",
    ),
    "two_counts": (
        "YPD and SC ratios, rule counts",
        2,
        False,
        "w116_twomed",
        "w116_twomedcv",
    ),
    "one_counts": (
        "YPD ratio, rule counts",
        1,
        False,
        "w113_ypdgpronly",
        "w113_ypdgpronlycv",
    ),
    "four_only": ("four ratios", 4, False, "w38_protpathfba", None),
    "one_only": ("YPD ratio", 1, False, "w102_ypdonly", None),
    "none": ("no metabolism", 0, False, "w39_pathpool", None),
}

#: key -> (waves, arm) of a repeated-split recipe. Later seeds of one arm were sometimes
#: written under a later wave prefix (round 78's w97 extends w93 and w94), so an arm pools
#: every wave listed; the seeds never overlap between waves.
RESPLIT: dict[str, tuple[tuple[str, ...], str]] = {
    "four_solve_counts": (("w96",), "pathloggpr"),
    "one_solve_counts": (("w104",), "ypdgpr"),
    "four_solve": (("w93", "w97", "w120"), "pathlogfree"),
    "four_counts": (("w106",), "gpronly"),
    "two_counts": (("w117",), "twomedgpronly"),
    "one_counts": (("w114", "w118"), "ypdgpronly"),
    "four_only": (("w94", "w97", "w120"), "protpathfba"),
    "none": (("w94", "w121"), "pathpool"),
}

#: (name, a, b): the difference read is a minus b.
RESPLIT_PAIRS: list[tuple[str, str, str]] = [
    (
        "one medium for four, solve and counts kept",
        "one_solve_counts",
        "four_solve_counts",
    ),
    ("counts for the flux vector, four media", "four_counts", "four_solve"),
    ("counts over the four ratios alone", "four_counts", "four_only"),
    ("flux vector over the four ratios alone", "four_solve", "four_only"),
    ("counts added to the flux vector, four media", "four_solve_counts", "four_solve"),
    ("no solve, counts kept, four media", "four_counts", "four_solve_counts"),
    ("one medium for four, no solve", "one_counts", "four_counts"),
    ("two media for four, no solve", "two_counts", "four_counts"),
    ("two media for one, no solve", "two_counts", "one_counts"),
    (
        "no solve and one medium, against the full recipe",
        "one_counts",
        "four_solve_counts",
    ),
    # Goal 2: every form against the same cell with no metabolism on the logit.
    ("one medium, solve and counts, over no metabolism", "one_solve_counts", "none"),
    ("four media, solve and counts, over no metabolism", "four_solve_counts", "none"),
    ("four media and the solve, over no metabolism", "four_solve", "none"),
    ("four media and the counts, over no metabolism", "four_counts", "none"),
    ("two media and the counts, over no metabolism", "two_counts", "none"),
    ("one medium and the counts, over no metabolism", "one_counts", "none"),
    ("four ratios alone, over no metabolism", "four_only", "none"),
]


class LadderRow(BaseModel):
    """One recipe on the released split."""

    key: str
    label: str
    media: int
    solver: bool
    seeds: list[int]
    cell_values: list[float]
    cell_mean: float
    cell_sd: float
    fold_ensemble_one_seed: float | None
    fold_ensemble_three_seeds: float | None


class Paired(BaseModel):
    """A paired difference, a minus b."""

    name: str
    a: str
    b: str
    n: int
    seeds_per_unit: list[int]
    mean_a: float
    mean_b: float
    mean_difference: float
    sd_difference: float
    n_positive: int
    #: Repeated-split rows only: the difference per split, and over every (split, seed)
    #: pair both arms ran, how many the first recipe won.
    per_unit_difference: list[float] | None = None
    n_pairs: int | None = None
    n_pairs_positive: int | None = None
    #: Budget rows only: the validation-selected epoch per seed, in both arms.
    select_epochs_a: list[int] | None = None
    select_epochs_b: list[int] | None = None


class Report(BaseModel):
    """Everything the simplification rounds measured."""

    ladder: list[LadderRow]
    resplit: list[Paired]
    subsets: list[Paired]
    budget: list[Paired]
    missing: list[str]
    note: str


def by_seed(stem: str, epochs: int = 200) -> dict[int, float]:
    """Held-out AUROC per seed for one cell stem."""
    out = {}
    for path in glob.glob(osp.join(RESULTS, f"{stem}_h12l1p1_e{epochs}_s*.json")):
        seed = int(re.search(r"_s(\d+)\.json$", path).group(1))  # type: ignore[union-attr]
        with open(path) as fh:
            out[seed] = float(json.load(fh)["test"]["pinned"]["auroc"])
    return out


def select_epochs(stem: str, epochs: int, seeds: list[int]) -> list[int]:
    out = []
    for seed in seeds:
        with open(osp.join(RESULTS, f"{stem}_h12l1p1_e{epochs}_s{seed}.json")) as fh:
            out.append(int(json.load(fh)["select_epoch"]))
    return out


def ensemble(prefix: str | None, suffix: str) -> float | None:
    if prefix is None:
        return None
    path = osp.join(RESULTS, f"{prefix}{suffix}_ensemble.json")
    if not osp.isfile(path):
        return None
    with open(path) as fh:
        return float(json.load(fh)["ensemble_auroc"])


def paired_by_seed(
    name: str, a: str, b: str, ea: int = 200, eb: int = 200
) -> Paired | None:
    va, vb = by_seed(a, ea), by_seed(b, eb)
    seeds = sorted(set(va) & set(vb))
    if len(seeds) < 2:
        return None
    x, y = np.array([va[s] for s in seeds]), np.array([vb[s] for s in seeds])
    d = x - y
    return Paired(
        name=name,
        a=a,
        b=b,
        n=len(seeds),
        seeds_per_unit=[1] * len(seeds),
        mean_a=float(x.mean()),
        mean_b=float(y.mean()),
        mean_difference=float(d.mean()),
        sd_difference=float(d.std(ddof=1)),
        n_positive=int((d > 0).sum()),
    )


def paired_by_split(name: str, a: str, b: str) -> Paired | None:
    def pooled(key: str, r: int) -> dict[int, float]:
        waves, arm = RESPLIT[key]
        out: dict[int, float] = {}
        for wave in waves:
            cells = by_seed(f"{wave}_resplit{r}{arm}")
            assert not set(cells) & set(out), (key, r, wave)
            out.update(cells)
        return out

    xs, ys, depth = [], [], []
    pair_differences: list[float] = []
    for r in range(N_SPLITS):
        va, vb = pooled(a, r), pooled(b, r)
        seeds = sorted(set(va) & set(vb))
        if not seeds:
            continue
        xs.append(float(np.mean([va[s] for s in seeds])))
        ys.append(float(np.mean([vb[s] for s in seeds])))
        depth.append(len(seeds))
        pair_differences.extend(va[s] - vb[s] for s in seeds)
    if len(xs) < 2:
        return None
    d = np.array(xs) - np.array(ys)
    return Paired(
        name=name,
        a=a,
        b=b,
        n=len(xs),
        seeds_per_unit=depth,
        mean_a=float(np.mean(xs)),
        mean_b=float(np.mean(ys)),
        mean_difference=float(d.mean()),
        sd_difference=float(d.std(ddof=1)),
        n_positive=int((d > 0).sum()),
        per_unit_difference=[float(v) for v in d],
        n_pairs=len(pair_differences),
        n_pairs_positive=int(sum(v > 0 for v in pair_differences)),
    )


def main() -> None:
    missing: list[str] = []
    ladder = []
    for key, (label, media, solver, stem, cv) in LADDER.items():
        values = by_seed(stem)
        if not values:
            missing.append(stem)
            continue
        v = np.array([values[s] for s in sorted(values)])
        ladder.append(
            LadderRow(
                key=key,
                label=label,
                media=media,
                solver=solver,
                seeds=sorted(values),
                cell_values=[float(x) for x in v],
                cell_mean=float(v.mean()),
                cell_sd=float(v.std(ddof=1)),
                fold_ensemble_one_seed=ensemble(cv, ""),
                fold_ensemble_three_seeds=ensemble(cv, "_s3"),
            )
        )
        r = ladder[-1]
        print(
            f"{label:42s} media {media} solver {int(solver)} n {len(v)} "
            f"cell {r.cell_mean:.4f} sd {r.cell_sd:.4f} folds "
            f"{r.fold_ensemble_one_seed} / {r.fold_ensemble_three_seeds}"
        )

    def collect(items: list[Paired | None], names: list[str]) -> list[Paired]:
        kept = []
        for item, name in zip(items, names, strict=True):
            if item is None:
                missing.append(name)
                continue
            kept.append(item)
            print(
                f"{item.name:52s} n {item.n} a {item.mean_a:.4f} b {item.mean_b:.4f} "
                f"diff {item.mean_difference:+.4f} sd {item.sd_difference:.4f} "
                f"positive {item.n_positive}/{item.n} seeds "
                f"{min(item.seeds_per_unit)}-{max(item.seeds_per_unit)}"
            )
        return kept

    print("-- ten repeated splits")
    resplit = collect(
        [paired_by_split(*p) for p in RESPLIT_PAIRS], [p[0] for p in RESPLIT_PAIRS]
    )
    print("-- rule-count subsets, released split")
    subset_spec = [
        (
            "sole-reaction count alone, against all six",
            "w103_pathlogsole",
            "w96_pathloggpr",
        ),
        ("five without it, against all six", "w103_pathlognosole", "w96_pathloggpr"),
        (
            "sole-reaction count alone, against none",
            "w103_pathlogsole",
            "w49_pathfbapfbalogfree",
        ),
        (
            "five without it, against none",
            "w103_pathlognosole",
            "w49_pathfbapfbalogfree",
        ),
    ]
    subsets = collect(
        [paired_by_seed(*s) for s in subset_spec], [s[0] for s in subset_spec]
    )
    print("-- 400 epochs against 200, released split")
    budget_spec = [
        ("flux vector and counts", "w115_gpr400", "w96_pathloggpr"),
        ("counts alone", "w115_gpronly400", "w101_pathgpronly"),
    ]
    budget = collect(
        [paired_by_seed(n, a, b, 400, 200) for n, a, b in budget_spec],
        [s[0] for s in budget_spec],
    )
    for row in budget:
        seeds = sorted(set(by_seed(row.a, 400)) & set(by_seed(row.b, 200)))
        row.select_epochs_a = select_epochs(row.a, 400, seeds)
        row.select_epochs_b = select_epochs(row.b, 200, seeds)
        print(
            f"  selected epochs {row.name}: {row.select_epochs_a} / {row.select_epochs_b}"
        )

    out = osp.join(RESULTS, "simplification_readout.json")
    with open(out, "w") as fh:
        json.dump(
            Report(
                ladder=ladder,
                resplit=resplit,
                subsets=subsets,
                budget=budget,
                missing=missing,
                note=(
                    "Every difference is a minus b. Repeated-split units are splits, each "
                    "the mean over the seeds both arms ran (seeds_per_unit); their test "
                    "blocks overlap, so no p-value. Released-split units are seeds."
                ),
            ).model_dump(),
            fh,
            indent=2,
        )
    print("missing:", missing)
    print(out)


if __name__ == "__main__":
    main()
