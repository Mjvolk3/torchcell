# experiments/027-betaxanthin-metabolic/scripts/fcl_comparison.py
# [[experiments.027-betaxanthin-metabolic.scripts.fcl_comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/fcl_comparison
"""Our essentiality arms against Flux Cone Learning and FBA, on ONE gene set with error bars.

"Three cells clear the published 0.73" was not a comparison. Three separate things were
being mixed:

1. **FCL's 0.73 and FBA's 0.69 are computed on DIFFERENT gene sets.** Recomputed here from
   the released per-gene predictions: FCL scores 0.7308 on the 223 held-out genes, which
   reproduces the published 0.73; FBA scores 0.6931 on ALL 1,121 labeled genes, which
   reproduces the published 0.69, and only 0.6300 on the 223 held-out genes. So the
   published pair is not a head-to-head, and quoting "FCL 0.73 against FBA 0.69" as if the
   two ran on the same genes overstates FBA by about 0.06.
2. **We score 195 genes, not 223.** Twenty-eight of the released test genes are absent from
   our cell graph, so our AUROC is on a subset and is not comparable with either published
   number until the baselines are restricted to the same subset.
3. **No number here carried an interval.** With 31 essential genes among 195, the sampling
   error on an AUROC is large, and the seed-to-seed spread of our own arms (0.578 to 0.793)
   is larger still.

This script fixes all three. Every method is scored on the SAME gene set, three of them:
the 1,121 labeled genes, the 223 released test genes, and the 195 genes we actually score.
Every AUROC carries a percentile bootstrap interval over genes, and our arms carry the
across-seed spread as well, which is the larger of the two uncertainties.

The baselines are recomputed from the authors' own released per-gene outputs rather than
transcribed from the paper, so the comparison is reproducible and the reproduction of the
published values is itself the check that the files are being read correctly.

Sources, all under ``$DATA_ROOT/data/merzbacher2025_fcl/deletionprediction-main/``:

* ``data/yeast_essentiality_test_split.csv`` : the gene labels and the held-out flag. The
  column named ``essential`` is a VIABILITY flag (1.0 marks a NONESSENTIAL gene); the
  inversion is applied once, in ``train_bx.read_fcl_essentiality_split``.
* ``figures/fig3/fig3a_test.csv`` : FCL's per-gene held-out predictions, five folds of a
  RandomForest on the flux cone, 620 rows per gene. ``score`` is P(viable), so the
  essentiality score is its negation. Averaged over rows within a gene.
* ``figures/fig3/fig3a_fba.csv`` : the FBA baseline, cobrapy ``single_gene_deletion``
  growth per gene. Low growth means essential, so the essentiality score is the negated
  growth rate.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/fcl_comparison.py

Writes ``results/fcl_comparison.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import roc_auc_score

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
OUT = osp.join(RESULTS, "fcl_comparison.json")
FCL_ROOT = osp.join(DATA_ROOT, "data/merzbacher2025_fcl/deletionprediction-main")
SPLIT_CSV = osp.join(FCL_ROOT, "data/yeast_essentiality_test_split.csv")
FCL_PRED_CSV = osp.join(FCL_ROOT, "figures/fig3/fig3a_test.csv")
FBA_PRED_CSV = osp.join(FCL_ROOT, "figures/fig3/fig3a_fba.csv")
#: Where train_bx.py dumps its per-record test predictions.
PRED_DIR = osp.join(DATA_ROOT, "test-predictions-027-wave7")
OUR_ARMS = {
    "pooled_no_module": "pooled_essentiality_s{seed}.json",
    "flux_biomass": "flux_projection_anchored_essentiality_s{seed}.json",
}
SEEDS = (2700, 2701, 2702)
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
#: The published values this script must reproduce, or the files are being misread.
PUBLISHED = {"fcl_test_223": 0.73, "fba_all_1121": 0.69}
PUBLISHED_TOLERANCE = 0.01


class ScoredMethod(BaseModel):
    """One method's AUROC on one gene set, with a bootstrap interval over genes."""

    model_config = ConfigDict(extra="forbid")
    method: str
    gene_set: str
    n_genes: int
    n_essential: int
    auroc: float
    ci_low: float = Field(description="2.5th percentile of the gene bootstrap.")
    ci_high: float = Field(description="97.5th percentile of the gene bootstrap.")
    seed: int | None = Field(
        default=None, description="Our arms only; the baselines have no seed."
    )


class ArmAcrossSeeds(BaseModel):
    """One of our arms summarized over seeds on one gene set."""

    model_config = ConfigDict(extra="forbid")
    arm: str
    gene_set: str
    auroc_by_seed: dict[str, float]
    mean: float
    sd: float
    minimum: float
    maximum: float


class HeadToHead(BaseModel):
    """One arm against one baseline on the same genes, per seed."""

    model_config = ConfigDict(extra="forbid")
    arm: str
    baseline: str
    gene_set: str
    baseline_auroc: float
    difference_by_seed: dict[str, float]
    seeds_above_baseline: int
    n_seeds: int
    verdict: str


class FclComparisonReport(BaseModel):
    """Every method on every gene set, plus the head-to-head our claims rest on."""

    model_config = ConfigDict(extra="forbid")
    gene_sets: dict[str, int]
    reproduction_check: dict[str, float] = Field(
        description="Recomputed published values; each must land within 0.01 of the paper."
    )
    scored: list[ScoredMethod]
    arms_across_seeds: list[ArmAcrossSeeds]
    head_to_head: list[HeadToHead]
    note: str


def bootstrap_auroc(
    label: np.ndarray, score: np.ndarray, n: int = N_BOOTSTRAP
) -> tuple[float, float, float]:
    """Point AUROC and a percentile interval, resampling GENES with replacement.

    A resample that draws one class only has no AUROC and is discarded rather than counted
    as 0.5, which would pull the interval toward chance.
    """
    point = float(roc_auc_score(label, score))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: list[float] = []
    size = label.shape[0]
    for _ in range(n):
        idx = rng.integers(0, size, size)
        if len(np.unique(label[idx])) < 2:
            continue
        draws.append(float(roc_auc_score(label[idx], score[idx])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def load_labels() -> pd.DataFrame:
    """``gene, essential, held_out`` for the 1,121 labeled genes, inversion applied."""
    split = pd.read_csv(SPLIT_CSV)
    split = split[split["essential"].notna()].copy()
    # The column is a viability flag; 1.0 marks a NONESSENTIAL gene.
    split["is_essential"] = (split["essential"].astype(float) == 0.0).astype(int)
    split["held_out"] = (split["test"].astype(int) == 1).astype(int)
    out = split[["knockout", "is_essential", "held_out"]].rename(
        columns={"knockout": "gene"}
    )
    assert len(out) == 1121, f"expected 1,121 labeled genes, read {len(out)}."
    assert int(out["is_essential"].sum()) == 157, "expected 157 essential genes."
    assert int(out["held_out"].sum()) == 223, "expected 223 held-out genes."
    return out.set_index("gene")


def load_fcl_scores() -> pd.Series:
    """FCL's essentiality score per held-out gene, averaged over its folds and repeats."""
    pred = pd.read_csv(FCL_PRED_CSV)
    per = pred.groupby("knockout_name")["score"].mean()
    # `score` is P(viable): a high score means the gene is NOT essential.
    return -per


def load_fba_scores() -> pd.Series:
    """The FBA baseline's essentiality score per gene, the negated deletion growth rate."""
    fba = pd.read_csv(FBA_PRED_CSV, index_col=0)
    return -fba.set_index("orf")["growth"]


def load_our_scores(arm: str, seed: int) -> pd.Series | None:
    """Our arm's per-gene logit on its pinned test genes, or None when the dump is absent."""
    path = osp.join(PRED_DIR, OUR_ARMS[arm].format(seed=seed))
    if not osp.isfile(path):
        return None
    with open(path) as fh:
        dump = json.load(fh)
    rows = [p for p in dump["predictions"] if p["pinned"]]
    # One deleted gene per genotype on this head; a genotype deleting two is not in the split.
    return pd.Series(
        {str(p["genes"][0]): float(p["pred"]) for p in rows if len(p["genes"]) == 1}
    )


def score_on(
    method: str, gene_set: str, scores: pd.Series, labels: pd.DataFrame, genes: set[str]
) -> ScoredMethod | None:
    shared = sorted(genes & set(scores.index) & set(labels.index))
    if len(shared) < 10:
        return None
    y = labels.loc[shared, "is_essential"].to_numpy()
    s = scores.loc[shared].to_numpy()
    point, lo, hi = bootstrap_auroc(y, s)
    return ScoredMethod(
        method=method,
        gene_set=gene_set,
        n_genes=len(shared),
        n_essential=int(y.sum()),
        auroc=point,
        ci_low=lo,
        ci_high=hi,
    )


def main() -> None:
    labels = load_labels()
    fcl = load_fcl_scores()
    fba = load_fba_scores()
    ours = {
        (arm, seed): load_our_scores(arm, seed) for arm in OUR_ARMS for seed in SEEDS
    }
    present = {k: v for k, v in ours.items() if v is not None}
    assert present, f"no prediction dumps under {PRED_DIR}."

    all_genes = set(labels.index)
    held_out = set(labels[labels["held_out"] == 1].index)
    # The genes WE score: the intersection over the arms that have a dump, so every method
    # is compared on one fixed set rather than on whatever each arm happened to resolve.
    scored_genes = (
        set.intersection(*(set(v.index) for v in present.values())) & held_out
    )
    gene_sets = {
        "all_labeled_1121": all_genes,
        "released_held_out_223": held_out,
        "ours_scored": scored_genes,
    }

    scored: list[ScoredMethod] = []
    for name, genes in gene_sets.items():
        for method, series in (("FCL", fcl), ("FBA", fba)):
            block = score_on(method, name, series, labels, genes)
            if block is not None:
                scored.append(block)
        for (arm, seed), series in present.items():
            block = score_on(arm, name, series, labels, genes)
            if block is not None:
                block.seed = seed
                scored.append(block)

    reproduction = {
        "fcl_test_223": next(
            s.auroc
            for s in scored
            if s.method == "FCL" and s.gene_set == "released_held_out_223"
        ),
        "fba_all_1121": next(
            s.auroc
            for s in scored
            if s.method == "FBA" and s.gene_set == "all_labeled_1121"
        ),
    }
    for key, value in reproduction.items():
        assert abs(value - PUBLISHED[key]) <= PUBLISHED_TOLERANCE, (
            f"{key} recomputes to {value:.4f} against the published {PUBLISHED[key]}, "
            f"which is outside the {PUBLISHED_TOLERANCE} tolerance. The released files are "
            "not being read the way the paper computed them, so nothing downstream is a "
            "comparison."
        )

    across: list[ArmAcrossSeeds] = []
    head_to_head: list[HeadToHead] = []
    for arm in OUR_ARMS:
        by_seed = {
            str(s.seed): s.auroc
            for s in scored
            if s.method == arm and s.gene_set == "ours_scored"
        }
        if not by_seed:
            continue
        values = np.array(list(by_seed.values()), dtype=float)
        across.append(
            ArmAcrossSeeds(
                arm=arm,
                gene_set="ours_scored",
                auroc_by_seed=by_seed,
                mean=float(values.mean()),
                sd=float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                minimum=float(values.min()),
                maximum=float(values.max()),
            )
        )
        for baseline in ("FCL", "FBA"):
            ref = next(
                (
                    s.auroc
                    for s in scored
                    if s.method == baseline and s.gene_set == "ours_scored"
                ),
                None,
            )
            if ref is None:
                continue
            diffs = {k: v - ref for k, v in by_seed.items()}
            above = sum(1 for v in diffs.values() if v > 0)
            head_to_head.append(
                HeadToHead(
                    arm=arm,
                    baseline=baseline,
                    gene_set="ours_scored",
                    baseline_auroc=ref,
                    difference_by_seed=diffs,
                    seeds_above_baseline=above,
                    n_seeds=len(diffs),
                    verdict=(
                        f"{above} of {len(diffs)} seeds above {baseline}; the arm's "
                        f"across-seed range is {min(by_seed.values()):.3f} to "
                        f"{max(by_seed.values()):.3f}, so a single seed above the baseline "
                        "is not evidence the arm beats it."
                    ),
                )
            )

    report = FclComparisonReport(
        gene_sets={k: len(v) for k, v in gene_sets.items()},
        reproduction_check=reproduction,
        scored=scored,
        arms_across_seeds=across,
        head_to_head=head_to_head,
        note=(
            "Every AUROC on one row of a gene set is computed on the same genes. The "
            "published FCL 0.73 is the 223 held-out genes and the published FBA 0.69 is "
            "all 1,121 labeled genes, so the published pair is not a head-to-head; on the "
            "held-out genes FBA scores about 0.63. Bootstrap intervals resample genes and "
            "hold the method fixed, so they capture sampling error and not seed noise; for "
            "our arms the across-seed spread is the larger of the two."
        ),
    )
    with open(OUT, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)

    print(f"gene sets requested: {report.gene_sets}")
    print(f"reproduction of published values: {reproduction}")
    for name in gene_sets:
        rows = [s for s in scored if s.gene_set == name]
        if not rows:
            continue
        # EVERY ROW PRINTS ITS OWN n. A method is scored on the intersection of the
        # requested set with the genes it has predictions for, and those differ: FCL
        # released held-out predictions only, and our arms resolve 195 of the 223. A single
        # header count would be the first row's and would misdescribe the others.
        print(f"\nrequested gene set: {name}")
        for s in sorted(rows, key=lambda r: -r.auroc):
            tag = f" s{s.seed}" if s.seed is not None else ""
            print(
                f"  {s.method + tag:28s} n {s.n_genes:4d} ({s.n_essential:3d} ess)  "
                f"AUROC {s.auroc:.4f}  95% CI [{s.ci_low:.3f}, {s.ci_high:.3f}]"
            )
    for h in head_to_head:
        print(f"\n{h.arm} vs {h.baseline} on our genes: {h.verdict}")
    print(f"\n{OUT}")


if __name__ == "__main__":
    main()
