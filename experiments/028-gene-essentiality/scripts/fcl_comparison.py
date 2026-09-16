# experiments/028-gene-essentiality/scripts/fcl_comparison.py
# [[experiments.028-gene-essentiality.scripts.fcl_comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/fcl_comparison
"""Our essentiality arms against Flux Cone Learning and FBA, on ONE gene set with error bars.

"Three cells clear the published 0.73" was not a comparison. Three separate things were
being mixed:

1. **The paper publishes THREE yeast AUROCs, on THREE gene sets.** Recomputed here from
   the released files, each with the authors' own statistic (``figures/fig3/fig3a.ipynb``
   and Supplementary Table S5):

   * **FCL 0.78 +/- 0.065, the number printed in the Fig. 3a legend** ("5-fold
     crossvalidation"): the mean and SD over five folds of a SAMPLE-level AUROC on each
     fold's validation genes, which lie inside the 898 TRAINING genes. Recomputed 0.777
     +/- 0.065 from ``fig3a_rf.csv``. Each fold validates 179 genes (25 essential); 590
     distinct genes appear and a gene sits in one to four folds, so the folds are
     resamples of the training side rather than a partition of it.
   * **FCL 0.73, Table S5's "FCL (test)" row**: the 223 held-out genes, scored by running
     them through all five fold models and averaging the deletion-level score (Methods).
     Recomputed 0.7308 from ``fig3a_test.csv`` with exactly that averaging.
   * **FBA 0.69**: ALL genes in the GEM, one cobrapy growth rate each. Recomputed 0.6945
     on the 1,124 released rows and 0.6931 on the 1,121 of them the split labels. On the
     223 held-out genes FBA scores 0.6300.

   So neither published pair is a head-to-head. The figure's 0.78 against 0.69 compares
   training-side genes against all genes; Table S5's 0.73 against 0.69 compares the
   held-out genes against all genes and overstates FBA by about 0.06 on those genes. The
   comparator for a model scored on the held-out genes is the 0.73 line, never the 0.78.
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
transcribed from the paper, so the comparison is reproducible and the reproduction of all
three published values is itself the check that the files are being read correctly.

**Whose labels our model trains on.** The same ones. ``train_essentiality.py`` takes the label from
the released split below, joined to each single-deletion genotype by its deleted gene; the
knowledge-graph build supplies only the genotypes and the cell graph, and every phenotype
column it carries is ignored for this head. The 223 held-out genes are pinned as our test
split, and our validation genes are drawn from the 898 training genes. The independently
curated SGD essential-gene list agrees with the released label on 98.8 percent of the 1,121
genes (``essentiality_label_agreement.json``), so the queried label would not be a
different target, but it is not the one used.

Sources, all under ``$DATA_ROOT/data/merzbacher2025_fcl/deletionprediction-main/``:

* ``data/yeast_essentiality_test_split.csv`` : the gene labels and the held-out flag. The
  column named ``essential`` is a VIABILITY flag (1.0 marks a NONESSENTIAL gene); the
  inversion is applied once, in ``train_essentiality.read_fcl_essentiality_split``.
* ``figures/fig3/fig3a_rf.csv`` : FCL's five cross-validation folds, 124 flux samples per
  validation gene per fold, the file the notebook draws the Fig. 3a FCL curves from.
* ``figures/fig3/fig3a_test.csv`` : FCL's per-gene held-out predictions, five folds of a
  RandomForest on the flux cone, 620 rows per gene. ``score`` is P(viable), so the
  essentiality score is its negation. Averaged over rows within a gene.
* ``figures/fig3/fig3a_fba.csv`` : the FBA baseline, cobrapy ``single_gene_deletion``
  growth per gene. Low growth means essential, so the essentiality score is the negated
  growth rate.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/fcl_comparison.py

Writes ``results/fcl_comparison.json``.
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import roc_auc_score

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
OUT = osp.join(RESULTS, "fcl_comparison.json")
FCL_ROOT = osp.join(DATA_ROOT, "data/merzbacher2025_fcl/deletionprediction-main")
SPLIT_CSV = osp.join(FCL_ROOT, "data/yeast_essentiality_test_split.csv")
FCL_CV_CSV = osp.join(FCL_ROOT, "figures/fig3/fig3a_rf.csv")
FCL_PRED_CSV = osp.join(FCL_ROOT, "figures/fig3/fig3a_test.csv")
FBA_PRED_CSV = osp.join(FCL_ROOT, "figures/fig3/fig3a_fba.csv")
#: Where train_bx.py dumps its per-record test predictions, one file per RESULT FILE.
PRED_DIR = osp.join(DATA_ROOT, "test-predictions-028")
#: The result files of every essentiality cell that trained to its full budget. A cell is
#: found through its result, and its dump through the result's stem, never by a guessed
#: dump name: the old name `<arm>_essentiality_s<seed>.json` was shared by every width of
#: the wave-11 ladder and the last cell to finish overwrote the others (2026-09-16).
RESULT_GLOBS = ("wave9_ess_*_e80_s*.json", "wave11_ess_*_e80_s*.json")
#: What the arm is called in the report; the readout is what differs between the two.
ARM_LABEL = {"pooled": "pooled", "flux_projection_anchored": "flux_biomass"}
#: The fields on which a dump must agree with the result that claims it.
IDENTITY_FIELDS = (
    "head",
    "arm",
    "seed",
    "hidden",
    "layers",
    "attention_heads",
    "preprocessor_layers",
    "epochs",
)
#: `preprocessor_layers` was added to the cell in wave 11 with this default, which is what
#: every earlier cell ran; a result that predates the field is labeled with it.
PREPROCESSOR_LAYERS_BEFORE_WAVE_11 = 2
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
#: The published values this script must reproduce, or the files are being misread. Each
#: names the gene set and the statistic the paper used for it, since the three differ.
PUBLISHED: dict[str, tuple[float, str, str, str]] = {
    "fcl_cv_5fold_mean": (
        0.78,
        "validation genes of five folds inside the 898 training genes (179 per fold, 590 "
        "distinct)",
        "mean over folds of a sample-level AUROC (124 flux samples per gene)",
        "Fig. 3a legend and Table S5 'FCL (5-fold CV)'",
    ),
    "fcl_cv_5fold_sd": (
        0.065,
        "the same five folds",
        "population SD over the five fold AUROCs (numpy default ddof=0, as the notebook)",
        "Fig. 3a legend and Table S5 'FCL (5-fold CV)'",
    ),
    "fcl_test_223": (
        0.73,
        "the 223 held-out genes",
        "gene-level AUROC of the score averaged over the five fold models (Methods)",
        "Table S5 'FCL (test)'",
    ),
    "fba_all_1121": (
        0.69,
        "all labeled genes (1,121 of the 1,124 released rows carry a label)",
        "single AUROC of the negated cobrapy deletion growth rate",
        "Fig. 3a legend and Table S5 'FBA'",
    ),
}
PUBLISHED_TOLERANCE = 0.01


class PublishedValue(BaseModel):
    """One number the paper prints, what it was computed on, and our recomputation."""

    model_config = ConfigDict(extra="forbid")
    name: str
    published: float
    recomputed: float
    gene_set: str
    statistic: str
    where_printed: str


class OurCell(BaseModel):
    """One of our trained cells: where its result and its per-gene predictions live."""

    model_config = ConfigDict(extra="forbid")
    label: str = Field(description="Arm plus architecture, e.g. pooled_h24l1p1.")
    seed: int
    result_file: str
    dump_file: str
    hidden: int
    layers: int
    preprocessor_layers: int
    n_parameters: int
    test_auroc_in_result: float = Field(
        description="The trainer's own pinned-gene AUROC, which the dump must reproduce."
    )


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
    reproduction_check: list[PublishedValue] = Field(
        description="Every published yeast AUROC recomputed from the released files with "
        "the paper's own statistic; each must land within 0.01 of the paper."
    )
    cells: list[OurCell]
    cells_without_predictions: dict[str, str] = Field(
        description="Result stems with no per-gene dump, and why; they are not scored."
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


def load_fcl_cv_fold_aurocs(labels: pd.DataFrame) -> list[float]:
    """The five cross-validation AUROCs behind the Fig. 3a legend, the notebook's way.

    ``fig3a.ipynb`` computes ``roc_curve(fold.true_label, fold.score)`` per fold over the
    SAMPLE rows, so every gene weighs in with its 124 flux samples and no averaging within
    a gene happens first. ``true_label`` and ``score`` are both in the viability direction;
    an AUROC is invariant under flipping both, so it is computed on the essentiality
    direction here to match the rest of the file. The file's genes are asserted to lie
    inside the released TRAINING genes, which is what makes this number a different gene
    set from the held-out 0.73.
    """
    cv = pd.read_csv(FCL_CV_CSV)
    train_genes = set(labels[labels["held_out"] == 0].index)
    assert set(cv["knockout_name"]) <= train_genes, (
        "the cross-validation file names genes outside the released training split, so "
        "it is not the validation-fold file the Fig. 3a legend was computed from."
    )
    per_gene_fold = cv.groupby(["knockout_name", "fold"]).size()
    assert per_gene_fold.min() == per_gene_fold.max() == 124, (
        f"expected 124 flux samples per gene per fold, read "
        f"{per_gene_fold.min()} to {per_gene_fold.max()}."
    )
    aurocs: list[float] = []
    for _, fold in cv.groupby("fold"):
        y = (fold["true_label"].to_numpy(dtype=float) == 0.0).astype(int)
        aurocs.append(float(roc_auc_score(y, -fold["score"].to_numpy(dtype=float))))
    assert len(aurocs) == 5, f"expected five folds, read {len(aurocs)}."
    return aurocs


def load_fba_scores() -> pd.Series:
    """The FBA baseline's essentiality score per gene, the negated deletion growth rate."""
    fba = pd.read_csv(FBA_PRED_CSV, index_col=0)
    return -fba.set_index("orf")["growth"]


def _identity(cell: dict[str, Any]) -> dict[str, Any]:
    return {k: cell.get(k) for k in IDENTITY_FIELDS}


def discover_cells() -> tuple[list[tuple[OurCell, pd.Series]], dict[str, str]]:
    """Every essentiality cell with a per-gene dump that is verifiably its own.

    A result file names its cell; the dump is ``PRED_DIR/<result stem>.json`` and must
    agree with the result on every identity field, or it is some other cell's file and the
    script stops. A result with no such dump is reported, not scored, and not guessed at.
    """
    found: list[tuple[OurCell, pd.Series]] = []
    missing: dict[str, str] = {}
    files = sorted(f for g in RESULT_GLOBS for f in glob.glob(osp.join(RESULTS, g)))
    assert files, f"no essentiality result files under {RESULTS}."
    for result_file in files:
        with open(result_file) as fh:
            result = json.load(fh)
        cell = result["cell"]
        assert cell["head"] == "essentiality", (
            f"{result_file} is not an essentiality cell."
        )
        stem = osp.splitext(osp.basename(result_file))[0]
        dump_file = osp.join(PRED_DIR, f"{stem}.json")
        if not osp.isfile(dump_file):
            missing[stem] = (
                "no dump named after the result; the cell wrote "
                f"{result['test'].get('dump_path')} under the pre-2026-09-16 naming and a "
                "later cell overwrote it (archive_essentiality_dumps.py lists it as lost)."
            )
            continue
        with open(dump_file) as fh:
            dump = json.load(fh)
        assert _identity(dump["cell"]) == _identity(cell), (
            f"{dump_file} holds {_identity(dump['cell'])}, not {_identity(cell)}."
        )
        rows = [p for p in dump["predictions"] if p["pinned"]]
        # One deleted gene per genotype on this head; a genotype deleting two is not in the
        # split.
        scores = pd.Series(
            {str(p["genes"][0]): float(p["pred"]) for p in rows if len(p["genes"]) == 1}
        )
        preproc = cell.get("preprocessor_layers")
        if preproc is None:
            preproc = PREPROCESSOR_LAYERS_BEFORE_WAVE_11
        # Cells built from 2026-09-16 carry no perturbation readout head, which shifts the
        # initialization at a given seed; the parameter tally is where that shows, and the
        # label says it so an old cell and its rerun are never averaged as one arm.
        head_marker = "" if "perturbation_head" in result["n_parameters"] else "_nohead"
        found.append(
            (
                OurCell(
                    label=(
                        f"{ARM_LABEL[cell['arm']]}_h{cell['hidden']}l{cell['layers']}"
                        f"p{preproc}{head_marker}"
                    ),
                    seed=int(cell["seed"]),
                    result_file=result_file,
                    dump_file=dump_file,
                    hidden=int(cell["hidden"]),
                    layers=int(cell["layers"]),
                    preprocessor_layers=int(preproc),
                    n_parameters=int(result["n_parameters"]["total"]),
                    test_auroc_in_result=float(result["test"]["pinned"]["auroc"]),
                ),
                scores,
            )
        )
    keys = [(c.label, c.seed) for c, _ in found]
    assert len(keys) == len(set(keys)), (
        f"two result files resolve to one (label, seed): {sorted(keys)}; a seed within an "
        "arm must be one cell, or the across-seed summary averages two runs as one."
    )
    return found, missing


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
    cells, missing = discover_cells()
    assert cells, f"no essentiality cell has a verifiable dump under {PRED_DIR}."

    all_genes = set(labels.index)
    held_out = set(labels[labels["held_out"] == 1].index)
    # The genes WE score: the intersection over the cells that have a dump, so every method
    # is compared on one fixed set rather than on whatever each cell happened to resolve.
    scored_genes = set.intersection(*(set(s.index) for _, s in cells)) & held_out
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
        for cell, series in cells:
            block = score_on(cell.label, name, series, labels, genes)
            if block is not None:
                block.seed = cell.seed
                scored.append(block)
    # THE DUMP MUST REPRODUCE THE TRAINER'S OWN NUMBER. The trainer scored its pinned genes
    # at the selected epoch and wrote that AUROC into the result; the dump it wrote in the
    # same pass, scored here on the same genes, has to give the same value, or the two
    # files are not from one pass.
    for cell, _ in cells:
        here = next(
            s.auroc
            for s in scored
            if s.method == cell.label
            and s.seed == cell.seed
            and s.gene_set == "ours_scored"
        )
        assert abs(here - cell.test_auroc_in_result) < 1e-6, (
            f"{cell.dump_file} scores {here:.6f} on the pinned genes but its result "
            f"recorded {cell.test_auroc_in_result:.6f}."
        )

    cv_aurocs = np.array(load_fcl_cv_fold_aurocs(labels), dtype=float)
    recomputed = {
        "fcl_cv_5fold_mean": float(cv_aurocs.mean()),
        "fcl_cv_5fold_sd": float(cv_aurocs.std()),
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
    reproduction: list[PublishedValue] = []
    for key, (published, gene_set, statistic, where) in PUBLISHED.items():
        value = recomputed[key]
        assert abs(value - published) <= PUBLISHED_TOLERANCE, (
            f"{key} recomputes to {value:.4f} against the published {published}, "
            f"which is outside the {PUBLISHED_TOLERANCE} tolerance. The released files are "
            "not being read the way the paper computed them, so nothing downstream is a "
            "comparison."
        )
        reproduction.append(
            PublishedValue(
                name=key,
                published=published,
                recomputed=value,
                gene_set=gene_set,
                statistic=statistic,
                where_printed=where,
            )
        )

    across: list[ArmAcrossSeeds] = []
    head_to_head: list[HeadToHead] = []
    for arm in dict.fromkeys(c.label for c, _ in cells):
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
        cells=[c for c, _ in cells],
        cells_without_predictions=missing,
        scored=scored,
        arms_across_seeds=across,
        head_to_head=head_to_head,
        note=(
            "Every AUROC on one row of a gene set is computed on the same genes. The paper "
            "prints three yeast AUROCs on three gene sets: the Fig. 3a legend's FCL 0.78 "
            "+/- 0.07 is five cross-validation folds inside the 898 training genes, Table "
            "S5's FCL 0.73 is the 223 held-out genes, and FBA 0.69 is all labeled genes, "
            "so no published pair is a head-to-head; on the held-out genes FBA scores "
            "about 0.63. A model scored on the held-out genes is compared against the 0.73 "
            "line. Our labels are the released split's, joined per genotype by deleted "
            "gene; the graph build supplies genotypes only. Bootstrap intervals resample "
            "genes and hold the method fixed, so they capture sampling error and not seed "
            "noise; for our arms the across-seed spread is the larger of the two."
        ),
    )
    with open(OUT, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)

    print(f"gene sets requested: {report.gene_sets}")
    print("our cells with a verified per-gene dump:")
    for c, _ in cells:
        print(
            f"  {c.label:24s} s{c.seed}  {c.n_parameters:>9,d} params  "
            f"{osp.basename(c.result_file)}"
        )
    for stem, why in missing.items():
        print(f"  NOT SCORED {stem}: {why}")
    print("reproduction of published values:")
    for p in reproduction:
        print(
            f"  {p.name:18s} published {p.published:<6} recomputed {p.recomputed:.4f}  "
            f"[{p.gene_set}; {p.statistic}]"
        )
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
