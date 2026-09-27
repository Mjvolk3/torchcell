# experiments/032-amino-acid-module/scripts/verify_doc_numbers.py
# [[experiments.032-amino-acid-module.scripts.verify_doc_numbers]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/032-amino-acid-module/scripts/verify_doc_numbers

"""Every number typed into notes-tex/032-amino-acid-module's prose, checked against its source file.

The essentiality verifier's rule and helpers, applied to this document: each check names
the claim, the quantity as the document states it, and how to recompute it from the
study's results tree. Add one whenever a number is written into prose. Exits non-zero
on any mismatch. Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/032-amino-acid-module/scripts/verify_doc_numbers.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, osp.join(os.environ["EXPERIMENT_ROOT"], "028-gene-essentiality", "scripts")
)
from verify_doc_numbers import (  # noqa: E402
    Check,
    concordance,
    concordance_summary,
    degeneracy,
    ensemble,
    physics_store,
    store_baseline,
    transfer,
    transfer_comparator,
)

RESULTS = osp.join(os.environ["EXPERIMENT_ROOT"], "032-amino-acid-module", "results")


def polytope(*keys: str):
    """A value of the concentration-polytope store's summary, by key path."""

    def run() -> float:
        import json

        d = json.load(
            open(
                osp.join(
                    RESULTS,
                    "target_concentration_polytope_store_amino_acids_mdfcap100poly.json",
                )
            )
        )
        for k in keys:
            d = d[k]
        return float(d)

    return run


def isobutanol_cross(kind: str, key: str, field: str = "spearman"):
    """A value of the isobutanol cross-endpoint read (isobutanol_amino_acid_correlation.py):
    ``kind`` "feature" keys on the amino acid, "pool" on the input set, "top" on the report.
    """

    def run() -> float:
        import json

        import numpy as np

        d = json.load(open(osp.join(RESULTS, "isobutanol_amino_acid_correlation.json")))
        if kind == "top":
            return float(d[key])
        if kind == "feature":
            f = next(x for x in d["features"] if x["key"] == key)
            return float(f[field])
        if kind == "feature_count_excluding_zero":
            return float(
                sum(1 for x in d["features"] if x["ci_low"] * x["ci_high"] > 0)
            )
        if kind == "mean_ceiling":
            return float(np.mean([x["ceiling"] for x in d["features"]]))
        if kind == "pool":
            p = next(x for x in d["pool"] if x["inputs"] == key)
            if field == "permuted":
                return float(np.mean(p["permuted_spearman_oof"]))
            if field == "fold_min":
                return float(min(p["spearman_by_fold"]))
            if field == "fold_max":
                return float(max(p["spearman_by_fold"]))
            return float(p[field])
        raise KeyError(kind)

    return run


def score_mlp(hidden: int, arm: str, key: str):
    """A value of one arm of the CPU perceptron sweep over the betaxanthin score
    (target_go_models.py --target amino_acids --store SM_gem_bxscore --mlp-hidden <hidden>).
    """

    def run() -> float:
        import json

        d = json.load(
            open(osp.join(RESULTS, f"go_models_amino_acids_bxscore_h{hidden}.json"))
        )
        assert d["store"] == "SM_gem_bxscore" and d["mlp_hidden"] == hidden
        a = next(x for x in d["arms"] if x["arm"] == arm)
        if key == "select_epoch_min":
            return float(min(r["select_epoch"] for r in a["seeds"]))
        if key == "select_epoch_max":
            return float(max(r["select_epoch"] for r in a["seeds"]))
        return float(a[key])

    return run


def go_model(arm: str, key: str):
    """A value of one arm of the GO-models read (target_go_models.py)."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, "go_models_amino_acids.json")))
        a = next(x for x in d["arms"] if x["arm"] == arm)
        return float(a[key])

    return run


def cross_feature(screen: str, key: str, field: str):
    """One statistic of the betaxanthin-against-amino-acid read for one quantity."""

    def run() -> float:
        import json

        d = json.load(
            open(osp.join(RESULTS, "betaxanthin_amino_acid_correlation.json"))
        )
        s = next(x for x in d["features"] if x["screen"] == screen and x["key"] == key)
        return float(s[field])

    return run


def cooper_ladder(key: str):
    """A value of the Cooper-target ladder (``results/store_baselines_cooper.json``):
    ``<rep>:<model>`` for a selected cell's mean Spearman, ``min`` or ``max`` over every
    cell, ``n_test`` or ``n_pinned_test``.
    """

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, "store_baselines_cooper.json")))
        reps = d["representations"]
        if key == "n_test":
            return float(d["n"]["test"])
        if key == "n_pinned_test":
            return float(d["n_pinned"]["test"])
        cells = [
            reps[r][m]["selected_on_val"]["test_pinned"]["mean_spearman"]
            for r in reps
            for m in ("ridge", "knn", "forest")
        ]
        if key == "min":
            return float(min(cells))
        if key == "max":
            return float(max(cells))
        rep, _, model = key.partition(":")
        return float(
            reps[rep][model]["selected_on_val"]["test_pinned"]["mean_spearman"]
        )

    return run


def mulleder_retest(key: str):
    """A value of the 027 retest read: a top-level mean, ``<amino acid>:<field>``,
    ``min:<field>`` or ``max:<field>`` over the nineteen, or ``spearman_ceiling``, the
    square root of the mean Spearman retest.
    """

    def run() -> float:
        import json
        import math

        d = json.load(
            open(
                osp.join(
                    os.environ["EXPERIMENT_ROOT"],
                    "027-betaxanthin-metabolic",
                    "results",
                    "mulleder_retest_reliability.json",
                )
            )
        )
        if key == "spearman_ceiling":
            return math.sqrt(float(d["mean_spearman_retest"]))
        if ":" in key:
            head, _, field = key.partition(":")
            vals = [float(r[field]) for r in d["per_amino_acid"]]
            if head == "min":
                return min(vals)
            if head == "max":
                return max(vals)
            return float(
                next(r for r in d["per_amino_acid"] if r["amino_acid"] == head)[field]
            )
        return float(d[key])

    return run


def cross_summary(key: str):
    """A summary of the betaxanthin-against-amino-acid read."""

    def run() -> float:
        import json

        import numpy as np

        d = json.load(
            open(osp.join(RESULTS, "betaxanthin_amino_acid_correlation.json"))
        )
        screen, _, field = key.partition(":")
        if field in ("n_shared_mulleder", "n_shared_cooper", "betaxanthin_reliability"):
            return float(d[field])
        if field in ("ceiling_min", "ceiling_max", "abs_spearman_max"):
            rows = [x for x in d["features"] if x["screen"] == screen]
            if field == "abs_spearman_max":
                return float(max(abs(x["spearman"]) for x in rows))
            vals = [x["ceiling"] for x in rows if x["ceiling"] is not None]
            return float(min(vals) if field == "ceiling_min" else max(vals))
        screen, _, inputs = screen.partition("/")
        p = next(
            x
            for x in d["pool"]
            if x["screen"] == screen and x["inputs"] == (inputs or "pool")
        )
        if field == "spearman_oof":
            return float(p["spearman_oof"])
        if field == "fold_min":
            return float(min(p["spearman_by_fold"]))
        if field == "fold_max":
            return float(max(p["spearman_by_fold"]))
        if field == "permuted_mean":
            return float(np.mean(p["permuted_spearman_oof"]))
        if field == "permuted_sd":
            return float(np.std(p["permuted_spearman_oof"]))
        if field == "n_with_go":
            return float(p["n_with_go"])
        if field in ("folds_above_pool", "folds_above_go"):
            other = next(
                x
                for x in d["pool"]
                if x["screen"] == screen and x["inputs"] == field.split("_")[-1]
            )
            return float(
                sum(
                    a > b
                    for a, b in zip(p["spearman_by_fold"], other["spearman_by_fold"])
                )
            )
        raise KeyError(key)

    return run


def aa_pair_cells(pair: str, side: str, key: str, which: str):
    """The per-seed cells of one side of an amino-acid transfer-readout pair: ``min`` or
    ``max`` of ``select_epoch``, ``select_val_pearson`` or ``spearman``.
    """

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, "transfer_readout.json")))
        p = next(q for q in d["pairs"] if q["name"] == pair)
        v = [c[key] for c in p[side]["cells"]]
        if which == "min":
            return float(min(v))
        if which == "max":
            return float(max(v))
        raise KeyError(which)

    return run


def aa_seed_level(stem: str, which: str):
    """A single-seed statistic of one amino-acid arm over its seeds, from the ensemble
    readout: mean over all seeds, its sd, the lowest and highest seed.
    """

    def run() -> float:
        import json

        import numpy as np

        d = json.load(open(osp.join(RESULTS, "ensemble_readout.json")))
        a = next(q for q in d["arms"] if q["stem"] == stem)
        v = np.array([x["spearman"] for x in a["single"]], dtype=float)
        if which == "first6":
            return float(v[:6].mean())
        if which == "last6":
            return float(v[6:].mean())
        if which == "last6_sd":
            return float(v[6:].std(ddof=1))
        if which == "mean":
            return float(v.mean())
        if which == "sd":
            return float(v.std(ddof=1))
        if which == "min":
            return float(v.min())
        if which == "max":
            return float(v.max())
        if which == "range":
            return float(v.max() - v.min())
        if which == "n":
            return float(v.size)
        raise KeyError(which)

    return run


CHECKS: list[Check] = [
    # sec:cross-target -- the betaxanthin score against the amino-acid pool.
    Check(
        claim="cross-target: strains shared with Mulleder",
        stated=4477,
        recompute=cross_summary("mulleder2016:n_shared_mulleder"),
        tolerance=0.5,
    ),
    Check(
        claim="cross-target: strains shared with Cooper",
        stated=4225,
        recompute=cross_summary("cooper2010:n_shared_cooper"),
        tolerance=0.5,
    ),
    Check(
        claim="cross-target: betaxanthin reliability",
        stated=0.84,
        recompute=cross_summary(":betaxanthin_reliability"),
        tolerance=0.006,
    ),
    Check(
        claim="cross-target: methionine Spearman",
        stated=0.130,
        recompute=cross_feature("mulleder2016", "methionine", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: methionine interval low",
        stated=0.100,
        recompute=cross_feature("mulleder2016", "methionine", "ci_low"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: methionine interval high",
        stated=0.157,
        recompute=cross_feature("mulleder2016", "methionine", "ci_high"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: threonine Spearman",
        stated=0.093,
        recompute=cross_feature("mulleder2016", "threonine", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: aspartate Spearman",
        stated=0.091,
        recompute=cross_feature("mulleder2016", "aspartate", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: tyrosine Spearman",
        stated=-0.083,
        recompute=cross_feature("mulleder2016", "tyrosine", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: arginine Spearman",
        stated=-0.077,
        recompute=cross_feature("mulleder2016", "arginine", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: glutamate Spearman",
        stated=-0.065,
        recompute=cross_feature("mulleder2016", "glutamate", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: aspartate top-5 enrichment",
        stated=3.3,
        recompute=cross_feature("mulleder2016", "aspartate", "top5_enrichment"),
        tolerance=0.06,
    ),
    Check(
        claim="cross-target: aspartate top-5 overlap",
        stated=37,
        recompute=cross_feature("mulleder2016", "aspartate", "top5_overlap"),
        tolerance=0.5,
    ),
    Check(
        claim="cross-target: aspartate top-5 expected",
        stated=11.2,
        recompute=cross_feature("mulleder2016", "aspartate", "top5_expected"),
        tolerance=0.06,
    ),
    Check(
        claim="cross-target: threonine top-5 enrichment",
        stated=3.0,
        recompute=cross_feature("mulleder2016", "threonine", "top5_enrichment"),
        tolerance=0.06,
    ),
    Check(
        claim="cross-target: Mulleder ceiling min",
        stated=0.20,
        recompute=cross_summary("mulleder2016:ceiling_min"),
        tolerance=0.006,
    ),
    Check(
        claim="cross-target: Mulleder ceiling max",
        stated=0.54,
        recompute=cross_summary("mulleder2016:ceiling_max"),
        tolerance=0.006,
    ),
    # sec:endpoint, sec:concordance, sec:standing -- the retest bounds a second screen,
    # the square root of the reliability bounds a model (corrected 2026-09-25).
    Check(
        claim="retest: mean Spearman",
        stated=0.20,
        recompute=mulleder_retest("mean_spearman_retest"),
        tolerance=0.006,
    ),
    Check(
        claim="retest: mean Pearson ceiling",
        stated=0.62,
        recompute=mulleder_retest("mean_pearson_ceiling"),
        tolerance=0.006,
    ),
    Check(
        claim="retest: Spearman ceiling, sqrt of the mean retest",
        stated=0.45,
        recompute=mulleder_retest("spearman_ceiling"),
        tolerance=0.006,
    ),
    Check(
        claim="retest: asparagine ceiling",
        stated=0.00,
        recompute=mulleder_retest("asparagine:pearson_ceiling"),
        tolerance=0.006,
    ),
    Check(
        claim="retest: methionine ceiling",
        stated=0.13,
        recompute=mulleder_retest("methionine:pearson_ceiling"),
        tolerance=0.006,
    ),
    Check(
        claim="retest: glutamine ceiling",
        stated=0.85,
        recompute=mulleder_retest("glutamine:pearson_ceiling"),
        tolerance=0.006,
    ),
    Check(
        claim="retest: QC injections",
        stated=237,
        recompute=mulleder_retest("n_qc_injections"),
        tolerance=0.5,
    ),
    Check(
        claim="retest: QC sd min",
        stated=0.03,
        recompute=mulleder_retest("min:qc_sd_log10"),
        tolerance=0.006,
    ),
    Check(
        claim="retest: QC sd max",
        stated=0.13,
        recompute=mulleder_retest("max:qc_sd_log10"),
        tolerance=0.006,
    ),
    Check(
        claim="retest: single-measurement sd min",
        stated=0.16,
        recompute=mulleder_retest("min:sd_single_measurement_log10"),
        tolerance=0.006,
    ),
    Check(
        claim="retest: single-measurement sd max",
        stated=0.27,
        recompute=mulleder_retest("max:sd_single_measurement_log10"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 174, the flux-and-annotation arm on twelve seeds.
    Check(
        claim="round 174: GO HiGHS twelve seeds",
        stated=0.056,
        recompute=aa_seed_level("w177_aadelgopfba_h12l1p1_e200", "mean"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 174: GO HiGHS twelve-seed sd",
        stated=0.003,
        recompute=aa_seed_level("w177_aadelgopfba_h12l1p1_e200", "sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 174: GO HiGHS twelve-seed min",
        stated=0.051,
        recompute=aa_seed_level("w177_aadelgopfba_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 174: GO HiGHS twelve-seed max",
        stated=0.060,
        recompute=aa_seed_level("w177_aadelgopfba_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 174: GO affine twelve seeds (pair)",
        stated=0.057,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against GO alone, twelve seeds",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 174: GO affine twelve-seed sd (pair)",
        stated=0.002,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against GO alone, twelve seeds",
            "sd_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 174: flux beside GO on twelve seeds",
        stated=-0.002,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against GO alone, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 174: flux beside GO on twelve seeds sd",
        stated=0.003,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against GO alone, twelve seeds",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 174: flux beside GO seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against GO alone, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 174: GO HiGHS twelve-seed ensemble",
        stated=0.057,
        recompute=ensemble("w177_aadelgopfba_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 174: GO affine twelve-seed ensemble",
        stated=0.058,
        recompute=ensemble("w170_aadelgo_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 183, the betaxanthin score with GO in the affine.
    Check(
        claim="round 183: score with GO arm",
        stated=0.053,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: score with GO arm sd",
        stated=0.010,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: GO alone level (pair)",
        stated=0.058,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: vs GO alone",
        stated=-0.005,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: vs GO alone sd",
        stated=0.010,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: vs GO alone seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 183: control level (pair)",
        stated=0.049,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against the control",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: vs control",
        stated=0.004,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against the control",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: vs control sd",
        stated=0.013,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against the control",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: vs control seeds above",
        stated=5,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against the control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 183: score with GO six-seed ensemble",
        stated=0.056,
        recompute=ensemble("w202_aadelgobxscore_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: seed min",
        stated=0.040,
        recompute=aa_seed_level("w202_aadelgobxscore_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: seed max",
        stated=0.065,
        recompute=aa_seed_level("w202_aadelgobxscore_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: select epoch min",
        stated=2,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 183: select epoch max",
        stated=166,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 183: validation Pearson min",
        stated=0.056,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: validation Pearson max",
        stated=0.077,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: GO alone select epoch min (pair)",
        stated=2,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "control",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 183: GO alone select epoch max (pair)",
        stated=3,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "control",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 183: GO alone validation Pearson min (pair)",
        stated=0.106,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "control",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: GO alone validation Pearson max (pair)",
        stated=0.112,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score with GO against GO alone",
            "control",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 183: the store's columns",
        stated=2363,
        recompute=lambda: float(
            json.load(
                open(osp.join(RESULTS, "target_store_concat_SM_gem_gobxscore.json"))
            )["n_columns"]
        ),
        tolerance=0.5,
    ),
    # sec:rounds -- round 184, the betaxanthin score alone in the affine.
    Check(
        claim="round 184: score alone arm",
        stated=0.060,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against GO alone",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: score alone arm sd",
        stated=0.015,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against GO alone",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: score with GO level (pair)",
        stated=0.053,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against the score with GO",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: vs score with GO",
        stated=0.006,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against the score with GO",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: vs score with GO sd",
        stated=0.020,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against the score with GO",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: vs score with GO seeds above",
        stated=4,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against the score with GO",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 184: GO alone level (pair)",
        stated=0.058,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against GO alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: vs GO alone",
        stated=0.002,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against GO alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: vs GO alone sd",
        stated=0.016,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against GO alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: vs GO alone seeds above",
        stated=3,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against GO alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 184: control level (pair)",
        stated=0.049,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against the control",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: vs control",
        stated=0.010,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against the control",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: vs control sd",
        stated=0.018,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against the control",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: vs control seeds above",
        stated=4,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against the control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 184: score alone six-seed ensemble",
        stated=0.073,
        recompute=ensemble("w203_aadelbxscore_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: control twelve-seed ensemble",
        stated=0.067,
        recompute=ensemble("w141_aadel_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: seed min",
        stated=0.036,
        recompute=aa_seed_level("w203_aadelbxscore_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: seed max",
        stated=0.080,
        recompute=aa_seed_level("w203_aadelbxscore_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: select epoch min",
        stated=30,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score alone in the affine against GO alone",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 184: select epoch max",
        stated=151,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score alone in the affine against GO alone",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 184: validation Pearson min",
        stated=0.098,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score alone in the affine against GO alone",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: validation Pearson max",
        stated=0.123,
        recompute=aa_pair_cells(
            "amino acids: measured betaxanthin score alone in the affine against GO alone",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 184: GO affine sd (pair)",
        stated=0.001,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against GO alone",
            "sd_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 189, GO alone standardized, the encoding control.
    Check(
        claim="round 189: GO standardized arm",
        stated=0.053,
        recompute=transfer(
            "amino acids: GO alone standardized against GO alone under log1p",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: GO standardized arm sd",
        stated=0.012,
        recompute=transfer(
            "amino acids: GO alone standardized against GO alone under log1p",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: GO log1p level (pair)",
        stated=0.058,
        recompute=transfer(
            "amino acids: GO alone standardized against GO alone under log1p",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: vs GO log1p",
        stated=-0.005,
        recompute=transfer(
            "amino acids: GO alone standardized against GO alone under log1p",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: vs GO log1p sd",
        stated=0.013,
        recompute=transfer(
            "amino acids: GO alone standardized against GO alone under log1p",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: vs GO log1p seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: GO alone standardized against GO alone under log1p",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 189: six-seed ensemble",
        stated=0.061,
        recompute=ensemble("w208_aadelgostd_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: seed min",
        stated=0.039,
        recompute=aa_seed_level("w208_aadelgostd_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: seed max",
        stated=0.073,
        recompute=aa_seed_level("w208_aadelgostd_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: select epoch min",
        stated=2,
        recompute=aa_pair_cells(
            "amino acids: GO alone standardized against GO alone under log1p",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 189: select epoch max",
        stated=196,
        recompute=aa_pair_cells(
            "amino acids: GO alone standardized against GO alone under log1p",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 189: validation Pearson min",
        stated=0.036,
        recompute=aa_pair_cells(
            "amino acids: GO alone standardized against GO alone under log1p",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: validation Pearson max",
        stated=0.075,
        recompute=aa_pair_cells(
            "amino acids: GO alone standardized against GO alone under log1p",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: score with GO vs GO standardized",
        stated=0.001,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against GO alone under the same standardized encoding",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: score with GO vs GO standardized sd",
        stated=0.013,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against GO alone under the same standardized encoding",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: score with GO vs GO standardized seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: measured betaxanthin score with GO against GO alone under the same standardized encoding",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 189: forest GO with score (ladder)",
        stated=0.078,
        recompute=store_baseline(
            "amino_acids", "go_bxscore", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 189: forest GO alone (ladder)",
        stated=0.063,
        recompute=store_baseline("amino_acids", "go", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    # sec:isobutanol-cross -- the isobutanol biosensor against the pool (2026-09-27).
    Check(
        claim="isobutanol cross: shared strains",
        stated=4360,
        recompute=isobutanol_cross("top", "n_shared"),
        tolerance=0.5,
    ),
    Check(
        claim="isobutanol cross: screen strains",
        stated=4554,
        recompute=isobutanol_cross("top", "n_lopez"),
        tolerance=0.5,
    ),
    Check(
        claim="isobutanol cross: pool ridge",
        stated=0.102,
        recompute=isobutanol_cross("pool", "pool", "spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: GO ridge",
        stated=0.017,
        recompute=isobutanol_cross("pool", "go", "spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: GO with pool ridge",
        stated=0.098,
        recompute=isobutanol_cross("pool", "go+pool", "spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: pool fold min",
        stated=0.056,
        recompute=isobutanol_cross("pool", "pool", "fold_min"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: pool fold max",
        stated=0.132,
        recompute=isobutanol_cross("pool", "pool", "fold_max"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: pool permuted",
        stated=-0.003,
        recompute=isobutanol_cross("pool", "pool", "permuted"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: leucine",
        stated=-0.029,
        recompute=isobutanol_cross("feature", "leucine"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: isoleucine",
        stated=-0.026,
        recompute=isobutanol_cross("feature", "isoleucine"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: valine",
        stated=0.004,
        recompute=isobutanol_cross("feature", "valine"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: arginine",
        stated=0.093,
        recompute=isobutanol_cross("feature", "arginine"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: arginine interval low",
        stated=0.063,
        recompute=isobutanol_cross("feature", "arginine", "ci_low"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: arginine interval high",
        stated=0.124,
        recompute=isobutanol_cross("feature", "arginine", "ci_high"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: arginine top-5 enrichment",
        stated=2.4,
        recompute=isobutanol_cross("feature", "arginine", "top5_enrichment"),
        tolerance=0.06,
    ),
    Check(
        claim="isobutanol cross: glutamine",
        stated=-0.075,
        recompute=isobutanol_cross("feature", "glutamine"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: aspartate",
        stated=-0.051,
        recompute=isobutanol_cross("feature", "aspartate"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: threonine",
        stated=-0.044,
        recompute=isobutanol_cross("feature", "threonine"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: proline",
        stated=-0.042,
        recompute=isobutanol_cross("feature", "proline"),
        tolerance=0.0006,
    ),
    Check(
        claim="isobutanol cross: intervals excluding zero",
        stated=7,
        recompute=isobutanol_cross("feature_count_excluding_zero", ""),
        tolerance=0.5,
    ),
    Check(
        claim="isobutanol cross: mean ceiling",
        stated=0.24,
        recompute=isobutanol_cross("mean_ceiling", ""),
        tolerance=0.006,
    ),
    Check(
        claim="isobutanol cross: screen reliability",
        stated=0.31,
        recompute=isobutanol_cross("top", "isobutanol_reliability"),
        tolerance=0.006,
    ),
    # sec:physics -- the CPU perceptron sweep over the betaxanthin score (2026-09-26).
    Check(
        claim="score mlp: score alone 16",
        stated=0.050,
        recompute=score_mlp(16, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp: score alone 256",
        stated=0.054,
        recompute=score_mlp(256, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp: score alone 1024",
        stated=0.055,
        recompute=score_mlp(1024, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp: score alone 256 select epoch min",
        stated=198,
        recompute=score_mlp(256, "store_mlp", "select_epoch_min"),
        tolerance=0.5,
    ),
    Check(
        claim="score mlp: score alone 1024 select epoch max",
        stated=198,
        recompute=score_mlp(1024, "store_mlp", "select_epoch_max"),
        tolerance=0.5,
    ),
    Check(
        claim="score mlp: GO with score 16",
        stated=0.056,
        recompute=score_mlp(16, "go_store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp: GO with score 256",
        stated=0.064,
        recompute=score_mlp(256, "go_store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp: GO with score 1024",
        stated=0.066,
        recompute=score_mlp(1024, "go_store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp: GO with score 1024 sd",
        stated=0.002,
        recompute=score_mlp(1024, "go_store_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp: GO with score 1024 ensemble",
        stated=0.066,
        recompute=score_mlp(1024, "go_store_mlp", "ensemble_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp: GO with score 1024 select epoch min",
        stated=10,
        recompute=score_mlp(1024, "go_store_mlp", "select_epoch_min"),
        tolerance=0.5,
    ),
    Check(
        claim="score mlp: GO with score 1024 select epoch max",
        stated=11,
        recompute=score_mlp(1024, "go_store_mlp", "select_epoch_max"),
        tolerance=0.5,
    ),
    Check(
        claim="score mlp: GO perceptron alone (existing)",
        stated=0.054,
        recompute=go_model("go_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp: score adds beside GO at 1024",
        stated=0.012,
        recompute=lambda: (
            score_mlp(1024, "go_store_mlp", "single_mean_spearman")()
            - go_model("go_mlp", "single_mean_spearman")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp: forest gain (ladder)",
        stated=0.015,
        recompute=lambda: (
            store_baseline("amino_acids", "go_bxscore", "forest", "mean_spearman")()
            - store_baseline("amino_acids", "go", "forest", "mean_spearman")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp e400: score alone 256",
        stated=0.055,
        recompute=lambda: float(
            next(
                a
                for a in json.load(
                    open(
                        osp.join(
                            RESULTS, "go_models_amino_acids_bxscore_h256_e400.json"
                        )
                    )
                )["arms"]
                if a["arm"] == "store_mlp"
            )["single_mean_spearman"]
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp e400: score alone 1024",
        stated=0.053,
        recompute=lambda: float(
            next(
                a
                for a in json.load(
                    open(
                        osp.join(
                            RESULTS, "go_models_amino_acids_bxscore_h1024_e400.json"
                        )
                    )
                )["arms"]
                if a["arm"] == "store_mlp"
            )["single_mean_spearman"]
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="score mlp e400: score alone 256 select epoch",
        stated=398,
        recompute=lambda: float(
            max(
                r["select_epoch"]
                for r in next(
                    a
                    for a in json.load(
                        open(
                            osp.join(
                                RESULTS, "go_models_amino_acids_bxscore_h256_e400.json"
                            )
                        )
                    )["arms"]
                    if a["arm"] == "store_mlp"
                )["seeds"]
            )
        ),
        tolerance=0.5,
    ),
    Check(
        claim="score mlp e400: score alone 1024 select epoch min",
        stated=321,
        recompute=lambda: float(
            min(
                r["select_epoch"]
                for r in next(
                    a
                    for a in json.load(
                        open(
                            osp.join(
                                RESULTS, "go_models_amino_acids_bxscore_h1024_e400.json"
                            )
                        )
                    )["arms"]
                    if a["arm"] == "store_mlp"
                )["seeds"]
            )
        ),
        tolerance=0.5,
    ),
    Check(
        claim="score mlp e400: score alone 1024 select epoch max",
        stated=384,
        recompute=lambda: float(
            max(
                r["select_epoch"]
                for r in next(
                    a
                    for a in json.load(
                        open(
                            osp.join(
                                RESULTS, "go_models_amino_acids_bxscore_h1024_e400.json"
                            )
                        )
                    )["arms"]
                    if a["arm"] == "store_mlp"
                )["seeds"]
            )
        ),
        tolerance=0.5,
    ),
    # sec:rounds -- round 195, twelve seeds of the score alone.
    Check(
        claim="round 195: score alone second six",
        stated=0.053,
        recompute=aa_seed_level("w203_aadelbxscore_h12l1p1_e200", "last6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: score alone second six sd",
        stated=0.006,
        recompute=aa_seed_level("w203_aadelbxscore_h12l1p1_e200", "last6_sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: score alone first six",
        stated=0.060,
        recompute=aa_seed_level("w203_aadelbxscore_h12l1p1_e200", "first6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: score alone twelve seeds",
        stated=0.056,
        recompute=aa_seed_level("w203_aadelbxscore_h12l1p1_e200", "mean"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: score alone twelve-seed sd",
        stated=0.012,
        recompute=aa_seed_level("w203_aadelbxscore_h12l1p1_e200", "sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: twelve-seed min",
        stated=0.036,
        recompute=aa_seed_level("w203_aadelbxscore_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: twelve-seed max",
        stated=0.080,
        recompute=aa_seed_level("w203_aadelbxscore_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: vs GO alone twelve seeds",
        stated=-0.001,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against GO alone, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: vs GO alone twelve seeds sd",
        stated=0.011,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against GO alone, twelve seeds",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: vs GO alone twelve seeds above",
        stated=4,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against GO alone, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 195: GO alone twelve-seed level (pair)",
        stated=0.057,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against GO alone, twelve seeds",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: vs control twelve seeds",
        stated=0.008,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against the control, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: vs control twelve seeds sd",
        stated=0.017,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against the control, twelve seeds",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: vs control twelve seeds above",
        stated=7,
        recompute=transfer(
            "amino acids: measured betaxanthin score alone in the affine against the control, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 195: twelve-seed ensemble",
        stated=0.072,
        recompute=ensemble("w203_aadelbxscore_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: control twelve-seed ensemble",
        stated=0.067,
        recompute=ensemble("w141_aadel_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: GO twelve-seed ensemble",
        stated=0.058,
        recompute=ensemble("w170_aadelgo_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 195: spread ratio to GO alone (six times)",
        stated=6,
        recompute=lambda: (
            transfer(
                "amino acids: measured betaxanthin score alone in the affine against GO alone, twelve seeds",
                "sd_spearman",
                "arm",
            )()
            / transfer(
                "amino acids: measured betaxanthin score alone in the affine against GO alone, twelve seeds",
                "sd_spearman",
                "control",
            )()
        ),
        tolerance=0.6,
    ),
    # sec:concordance -- the strain's betaxanthin score as an input (2026-09-25).
    Check(
        claim="ladder: betaxanthin score ridge",
        stated=0.034,
        recompute=store_baseline("amino_acids", "bx_score", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: betaxanthin score kNN",
        stated=0.026,
        recompute=store_baseline("amino_acids", "bx_score", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: betaxanthin score forest",
        stated=0.071,
        recompute=store_baseline("amino_acids", "bx_score", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with betaxanthin score ridge",
        stated=0.044,
        recompute=store_baseline("amino_acids", "go_bxscore", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with betaxanthin score kNN",
        stated=0.024,
        recompute=store_baseline("amino_acids", "go_bxscore", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with betaxanthin score forest",
        stated=0.078,
        recompute=store_baseline(
            "amino_acids", "go_bxscore", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO forest (score comparison)",
        stated=0.063,
        recompute=store_baseline("amino_acids", "go", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: betaxanthin score ridge, aspartate",
        stated=0.185,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "store_baselines_amino_acids.json")))[
                "representations"
            ]["bx_score"]["ridge"]["selected_on_val"]["test_pinned"]["per_feature"][
                "aspartate"
            ]["spearman"]
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: betaxanthin score ridge, threonine",
        stated=0.125,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "store_baselines_amino_acids.json")))[
                "representations"
            ]["bx_score"]["ridge"]["selected_on_val"]["test_pinned"]["per_feature"][
                "threonine"
            ]["spearman"]
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: betaxanthin score ridge, tyrosine",
        stated=0.084,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "store_baselines_amino_acids.json")))[
                "representations"
            ]["bx_score"]["ridge"]["selected_on_val"]["test_pinned"]["per_feature"][
                "tyrosine"
            ]["spearman"]
        ),
        tolerance=0.0006,
    ),
    # sec:concordance -- Cooper's peaks as a ladder target (2026-09-25).
    Check(
        claim="cooper ladder: test genotypes with a row",
        stated=816,
        recompute=cooper_ladder("n_test"),
        tolerance=0.5,
    ),
    Check(
        claim="cooper ladder: pinned test genotypes",
        stated=561,
        recompute=cooper_ladder("n_pinned_test"),
        tolerance=0.5,
    ),
    Check(
        claim="cooper ladder: lowest cell",
        stated=-0.025,
        recompute=cooper_ladder("min"),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper ladder: highest cell",
        stated=0.041,
        recompute=cooper_ladder("max"),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper ladder: HiGHS store with ProtT5 forest",
        stated=0.041,
        recompute=cooper_ladder("pfba_highs+prot_T5_all:forest"),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper ladder: MOMA ridge",
        stated=0.039,
        recompute=cooper_ladder("moma:ridge"),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper ladder: GO forest",
        stated=0.019,
        recompute=cooper_ladder("go:forest"),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper ladder: ProtT5 ridge",
        stated=0.028,
        recompute=cooper_ladder("prot_T5_all:ridge"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: Mulleder pool ridge out of fold",
        stated=0.226,
        recompute=cross_summary("mulleder2016:spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: Mulleder pool fold min",
        stated=0.182,
        recompute=cross_summary("mulleder2016:fold_min"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: Mulleder pool fold max",
        stated=0.272,
        recompute=cross_summary("mulleder2016:fold_max"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: Mulleder pool permuted mean",
        stated=-0.002,
        recompute=cross_summary("mulleder2016:permuted_mean"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: Mulleder pool permuted sd",
        stated=0.016,
        recompute=cross_summary("mulleder2016:permuted_sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: Cooper largest single peak",
        stated=0.063,
        recompute=cross_summary("cooper2010:abs_spearman_max"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: Cooper pool ridge out of fold",
        stated=0.083,
        recompute=cross_summary("cooper2010:spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: Cooper pool permuted mean",
        stated=-0.003,
        recompute=cross_summary("cooper2010:permuted_mean"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: Mulleder strains outside the GO store",
        stated=557,
        recompute=lambda: 4477 - cross_summary("mulleder2016/go:n_with_go")(),
        tolerance=0.5,
    ),
    Check(
        claim="cross-target: GO ridge on the Mulleder strains",
        stated=0.165,
        recompute=cross_summary("mulleder2016/go:spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: GO with the pool on the Mulleder strains",
        stated=0.237,
        recompute=cross_summary("mulleder2016/go+pool:spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: GO with the pool fold min",
        stated=0.207,
        recompute=cross_summary("mulleder2016/go+pool:fold_min"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: GO with the pool fold max",
        stated=0.311,
        recompute=cross_summary("mulleder2016/go+pool:fold_max"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: GO with the pool above the pool, folds",
        stated=4,
        recompute=cross_summary("mulleder2016/go+pool:folds_above_pool"),
        tolerance=0.5,
    ),
    Check(
        claim="cross-target: GO with the pool above GO, folds",
        stated=5,
        recompute=cross_summary("mulleder2016/go+pool:folds_above_go"),
        tolerance=0.5,
    ),
    Check(
        claim="cross-target: the pool's addition beyond GO (Mulleder)",
        stated=0.072,
        recompute=lambda: (
            cross_summary("mulleder2016/go+pool:spearman_oof")()
            - cross_summary("mulleder2016/go:spearman_oof")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: GO ridge on the Cooper strains",
        stated=0.193,
        recompute=cross_summary("cooper2010/go:spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: GO with the Cooper pool",
        stated=0.196,
        recompute=cross_summary("cooper2010/go+pool:spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: the pool's addition beyond GO (Cooper)",
        stated=0.003,
        recompute=lambda: (
            cross_summary("cooper2010/go+pool:spearman_oof")()
            - cross_summary("cooper2010/go:spearman_oof")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="cross-target: Cooper ceiling min",
        stated=0.00,
        recompute=cross_summary("cooper2010:ceiling_min"),
        tolerance=0.006,
    ),
    Check(
        claim="cross-target: Cooper ceiling max",
        stated=0.49,
        recompute=cross_summary("cooper2010:ceiling_max"),
        tolerance=0.006,
    ),
    Check(
        claim="cross-target: the betaxanthin study's best arm (round 151)",
        stated=0.211,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against the HiGHS store alone",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    # sec:physics -- GO on the ladder.
    Check(
        claim="ladder: aa GO ridge",
        stated=0.044,
        recompute=store_baseline("amino_acids", "go", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: aa GO kNN",
        stated=0.025,
        recompute=store_baseline("amino_acids", "go", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: aa GO forest",
        stated=0.063,
        recompute=store_baseline("amino_acids", "go", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: aa GO with HiGHS ridge",
        stated=0.046,
        recompute=store_baseline("amino_acids", "gopfba", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: aa GO with HiGHS kNN",
        stated=0.029,
        recompute=store_baseline("amino_acids", "gopfba", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: aa GO with HiGHS forest",
        stated=0.062,
        recompute=store_baseline("amino_acids", "gopfba", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: bx GO with HiGHS forest (cited)",
        stated=0.204,
        recompute=store_baseline("betaxanthin", "gopfba", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    # sec:physics -- the annotation-only family on the amino acids.
    Check(
        claim="go-models: aa GO perceptron",
        stated=0.054,
        recompute=go_model("go_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: aa GO perceptron sd",
        stated=0.002,
        recompute=go_model("go_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: aa ProtT5 perceptron",
        stated=0.054,
        recompute=go_model("prot_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: aa ProtT5 perceptron sd",
        stated=0.008,
        recompute=go_model("prot_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: aa GO with ProtT5",
        stated=0.065,
        recompute=go_model("go_prot_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: aa GO with ProtT5 sd",
        stated=0.003,
        recompute=go_model("go_prot_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: aa GO with ProtT5 ensemble",
        stated=0.069,
        recompute=go_model("go_prot_mlp", "ensemble_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: aa DCell hierarchy",
        stated=0.035,
        recompute=go_model("dcell_style_shared", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: aa DCell hierarchy sd",
        stated=0.015,
        recompute=go_model("dcell_style_shared", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: aa store perceptron",
        stated=0.019,
        recompute=go_model("store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: aa GO with store",
        stated=0.042,
        recompute=go_model("go_store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: aa all three",
        stated=0.045,
        recompute=go_model("go_prot_store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 138, the linear MOMA store on the amino acids.
    Check(
        claim="target-transfer: aa MOMA store arm",
        stated=0.052,
        recompute=transfer(
            "amino acids: linear MOMA store on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa MOMA store arm sd",
        stated=0.006,
        recompute=transfer(
            "amino acids: linear MOMA store on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa MOMA store vs control",
        stated=0.003,
        recompute=transfer(
            "amino acids: linear MOMA store on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa MOMA store vs control sd",
        stated=0.011,
        recompute=transfer(
            "amino acids: linear MOMA store on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa MOMA store vs control seeds above",
        stated=4,
        recompute=transfer(
            "amino acids: linear MOMA store on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa MOMA store vs parsimonious store",
        stated=0.012,
        recompute=transfer(
            "amino acids: linear MOMA store on deletion-only graphs against the parsimonious store there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa MOMA store vs parsimonious store sd",
        stated=0.008,
        recompute=transfer(
            "amino acids: linear MOMA store on deletion-only graphs against the parsimonious store there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa MOMA store vs parsimonious store seeds above",
        stated=5,
        recompute=transfer(
            "amino acids: linear MOMA store on deletion-only graphs against the parsimonious store there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa MOMA store six-seed ensemble",
        stated=0.056,
        recompute=ensemble("w155_aadelmoma_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 141, the driving-force concentration store on the amino acids.
    Check(
        claim="target-transfer: aa driving-force store arm",
        stated=0.045,
        recompute=transfer(
            "amino acids: driving-force concentration store on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa driving-force store arm sd",
        stated=0.005,
        recompute=transfer(
            "amino acids: driving-force concentration store on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa driving-force store vs control",
        stated=-0.005,
        recompute=transfer(
            "amino acids: driving-force concentration store on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa driving-force store vs control sd",
        stated=0.007,
        recompute=transfer(
            "amino acids: driving-force concentration store on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa driving-force store vs control seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: driving-force concentration store on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa driving-force store vs parsimonious store",
        stated=0.004,
        recompute=transfer(
            "amino acids: driving-force concentration store on deletion-only graphs against the parsimonious store there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa driving-force store vs parsimonious store sd",
        stated=0.009,
        recompute=transfer(
            "amino acids: driving-force concentration store on deletion-only graphs against the parsimonious store there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa driving-force store vs parsimonious store seeds above",
        stated=5,
        recompute=transfer(
            "amino acids: driving-force concentration store on deletion-only graphs against the parsimonious store there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa driving-force store six-seed ensemble",
        stated=0.050,
        recompute=ensemble("w158_aadelmdf_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 146, the store arm at width 6 on the amino acids.
    Check(
        claim="target-transfer: aa width-6 store arm",
        stated=0.025,
        recompute=transfer(
            "amino acids: deletion-only store at width 6 against width 12",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa width-6 store arm sd",
        stated=0.018,
        recompute=transfer(
            "amino acids: deletion-only store at width 6 against width 12",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa width-6 store vs width-12 store",
        stated=-0.015,
        recompute=transfer(
            "amino acids: deletion-only store at width 6 against width 12",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa width-6 store vs width-12 store sd",
        stated=0.017,
        recompute=transfer(
            "amino acids: deletion-only store at width 6 against width 12",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa width-6 store vs width-12 store seeds above",
        stated=1,
        recompute=transfer(
            "amino acids: deletion-only store at width 6 against width 12", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa width-6 store six-seed ensemble",
        stated=0.046,
        recompute=ensemble("w163_aadelstoreh6_h6l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 148, the polytope mean on the amino acids.
    Check(
        claim="target-transfer: aa polytope mean arm",
        stated=0.032,
        recompute=transfer(
            "amino acids: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa polytope mean arm sd",
        stated=0.008,
        recompute=transfer(
            "amino acids: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa polytope mean vs control",
        stated=-0.017,
        recompute=transfer(
            "amino acids: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa polytope mean vs control sd",
        stated=0.011,
        recompute=transfer(
            "amino acids: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa polytope mean vs control seeds above",
        stated=0,
        recompute=transfer(
            "amino acids: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa polytope mean vs parsimonious store",
        stated=-0.008,
        recompute=transfer(
            "amino acids: driving-force polytope mean on deletion-only graphs against the parsimonious store there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa polytope mean vs parsimonious store sd",
        stated=0.012,
        recompute=transfer(
            "amino acids: driving-force polytope mean on deletion-only graphs against the parsimonious store there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa polytope mean vs parsimonious store seeds above",
        stated=1,
        recompute=transfer(
            "amino acids: driving-force polytope mean on deletion-only graphs against the parsimonious store there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa polytope mean vs driving-force point",
        stated=-0.012,
        recompute=transfer(
            "amino acids: driving-force polytope mean on deletion-only graphs against the driving-force point",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa polytope mean vs driving-force point sd",
        stated=0.009,
        recompute=transfer(
            "amino acids: driving-force polytope mean on deletion-only graphs against the driving-force point",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa polytope mean vs driving-force point seeds above",
        stated=0,
        recompute=transfer(
            "amino acids: driving-force polytope mean on deletion-only graphs against the driving-force point",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa polytope mean six-seed ensemble",
        stated=0.035,
        recompute=ensemble("w165_aadelmdfpoly_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 143, the sampled cone on the amino acids.
    Check(
        claim="target-transfer: aa sampled cone arm",
        stated=0.019,
        recompute=transfer(
            "amino acids: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa sampled cone arm sd",
        stated=0.007,
        recompute=transfer(
            "amino acids: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa sampled cone vs control",
        stated=-0.03,
        recompute=transfer(
            "amino acids: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa sampled cone vs control sd",
        stated=0.007,
        recompute=transfer(
            "amino acids: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa sampled cone vs control seeds above",
        stated=0,
        recompute=transfer(
            "amino acids: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa sampled cone vs parsimonious store",
        stated=-0.021,
        recompute=transfer(
            "amino acids: sampled-cone mean store on deletion-only graphs against the parsimonious store there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa sampled cone vs parsimonious store sd",
        stated=0.01,
        recompute=transfer(
            "amino acids: sampled-cone mean store on deletion-only graphs against the parsimonious store there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa sampled cone vs parsimonious store seeds above",
        stated=0,
        recompute=transfer(
            "amino acids: sampled-cone mean store on deletion-only graphs against the parsimonious store there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa sampled cone six-seed ensemble",
        stated=0.023,
        recompute=ensemble("w160_aadelsampled_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 153, GO in the affine slot on the amino acids.
    Check(
        claim="target-transfer: aa GO affine arm",
        stated=0.058,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO affine arm sd",
        stated=0.001,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO affine vs control",
        stated=0.009,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO affine vs control sd",
        stated=0.007,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO affine vs control seeds above",
        stated=5,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa GO affine vs parsimonious store",
        stated=0.018,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot on deletion-only graphs against the parsimonious store there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO affine vs parsimonious store sd",
        stated=0.005,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot on deletion-only graphs against the parsimonious store there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO affine vs parsimonious store seeds above",
        stated=6,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot on deletion-only graphs against the parsimonious store there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa GO affine six-seed ensemble",
        stated=0.059,
        recompute=ensemble("w170_aadelgo_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 157, dropout 0.3 on the store arm.
    Check(
        claim="target-transfer: aa dropout 0.3 arm",
        stated=0.045,
        recompute=transfer(
            "amino acids: deletion-only store at dropout 0.3 against 0.1",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa dropout 0.3 arm sd",
        stated=0.005,
        recompute=transfer(
            "amino acids: deletion-only store at dropout 0.3 against 0.1",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa dropout 0.3 vs 0.1",
        stated=0.005,
        recompute=transfer(
            "amino acids: deletion-only store at dropout 0.3 against 0.1",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa dropout 0.3 vs 0.1 sd",
        stated=0.009,
        recompute=transfer(
            "amino acids: deletion-only store at dropout 0.3 against 0.1",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa dropout 0.3 vs 0.1 seeds above",
        stated=5,
        recompute=transfer(
            "amino acids: deletion-only store at dropout 0.3 against 0.1", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa store 0.1 level (round 157 pair)",
        stated=0.040,
        recompute=transfer(
            "amino acids: deletion-only store at dropout 0.3 against 0.1",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa dropout 0.3 vs control",
        stated=-0.005,
        recompute=transfer(
            "amino acids: deletion-only store at dropout 0.3 against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa dropout 0.3 vs control sd",
        stated=0.009,
        recompute=transfer(
            "amino acids: deletion-only store at dropout 0.3 against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa dropout 0.3 vs control seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: deletion-only store at dropout 0.3 against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa control level (round 157 pair)",
        stated=0.049,
        recompute=transfer(
            "amino acids: deletion-only store at dropout 0.3 against deletion-only graphs",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa dropout 0.3 six-seed ensemble",
        stated=0.049,
        recompute=ensemble("w173_aadelstoredrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa store six-seed ensemble (round 157 pair)",
        stated=0.043,
        recompute=ensemble("w143_aadelstore_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa control six-seed ensemble (round 157 pair)",
        stated=0.062,
        recompute=ensemble("w141_aadel_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: bx dropout 0.3 gain (cited in round 157)",
        stated=0.024,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.3 against 0.1",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 160, the HiGHS store with GO appended on the prototroph.
    Check(
        claim="target-transfer: aa GO with HiGHS arm",
        stated=0.055,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against GO alone",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS arm sd",
        stated=0.002,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against GO alone",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS vs GO alone",
        stated=-0.003,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against GO alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS vs GO alone sd",
        stated=0.002,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against GO alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS vs GO alone seeds above",
        stated=0,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against GO alone", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa GO alone level (round 160 pair)",
        stated=0.058,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against GO alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS vs store",
        stated=0.015,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against the parsimonious store alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS vs store sd",
        stated=0.004,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against the parsimonious store alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS vs store seeds above",
        stated=6,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against the parsimonious store alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa store level (round 160 pair)",
        stated=0.040,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against the parsimonious store alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS six-seed ensemble",
        stated=0.057,
        recompute=ensemble("w177_aadelgopfba_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO six-seed ensemble (round 160 pair)",
        stated=0.059,
        recompute=ensemble("w170_aadelgo_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa store six-seed ensemble (round 160 pair)",
        stated=0.043,
        recompute=ensemble("w143_aadelstore_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: bx flux gain beside GO (cited in round 160)",
        stated=0.009,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against GO alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 162, GO with the HiGHS vector at dropout 0.3 on the prototroph.
    Check(
        claim="target-transfer: aa GO with HiGHS dropout arm",
        stated=0.055,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS dropout arm sd",
        stated=0.002,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS dropout vs 0.1",
        stated=0.000,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS dropout vs 0.1 sd",
        stated=0.001,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS dropout vs 0.1 seeds above",
        stated=4,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS 0.1 level (round 162 pair)",
        stated=0.055,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS dropout vs store dropout",
        stated=0.011,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended at dropout 0.3 against the store at 0.3",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS dropout vs store dropout sd",
        stated=0.007,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended at dropout 0.3 against the store at 0.3",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS dropout vs store dropout seeds above",
        stated=6,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended at dropout 0.3 against the store at 0.3",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa store dropout level (round 162 pair)",
        stated=0.045,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended at dropout 0.3 against the store at 0.3",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS dropout six-seed ensemble",
        stated=0.056,
        recompute=ensemble("w179_aadelgopfbadrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS six-seed ensemble (round 162 pair)",
        stated=0.057,
        recompute=ensemble("w177_aadelgopfba_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa store dropout six-seed ensemble (round 162 pair)",
        stated=0.049,
        recompute=ensemble("w173_aadelstoredrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO with HiGHS margin over the store at 0.1 (cited)",
        stated=0.015,
        recompute=transfer(
            "amino acids: HiGHS store with GO appended against the parsimonious store alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 165, the GO affine at dropout 0.3 on the prototroph.
    Check(
        claim="target-transfer: aa GO dropout arm",
        stated=0.058,
        recompute=transfer(
            "amino acids: GO affine at dropout 0.3 against 0.1", "mean_spearman", "arm"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO dropout arm sd",
        stated=0.001,
        recompute=transfer(
            "amino acids: GO affine at dropout 0.3 against 0.1", "sd_spearman", "arm"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO dropout vs 0.1",
        stated=0.000,
        recompute=transfer(
            "amino acids: GO affine at dropout 0.3 against 0.1", "mean_difference"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO dropout vs 0.1 sd",
        stated=0.001,
        recompute=transfer(
            "amino acids: GO affine at dropout 0.3 against 0.1", "sd_difference"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO dropout vs 0.1 seeds above",
        stated=4,
        recompute=transfer(
            "amino acids: GO affine at dropout 0.3 against 0.1", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa GO 0.1 level (round 165 pair)",
        stated=0.058,
        recompute=transfer(
            "amino acids: GO affine at dropout 0.3 against 0.1",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO dropout vs store dropout",
        stated=0.013,
        recompute=transfer(
            "amino acids: GO affine at dropout 0.3 against the store at 0.3",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO dropout vs store dropout sd",
        stated=0.005,
        recompute=transfer(
            "amino acids: GO affine at dropout 0.3 against the store at 0.3",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO dropout vs store dropout seeds above",
        stated=6,
        recompute=transfer(
            "amino acids: GO affine at dropout 0.3 against the store at 0.3",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa store dropout level (round 165 pair)",
        stated=0.045,
        recompute=transfer(
            "amino acids: GO affine at dropout 0.3 against the store at 0.3",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO dropout six-seed ensemble",
        stated=0.059,
        recompute=ensemble("w182_aadelgodrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa GO six-seed ensemble (round 165 pair)",
        stated=0.059,
        recompute=ensemble("w170_aadelgo_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa store dropout six-seed ensemble (round 165 pair)",
        stated=0.049,
        recompute=ensemble("w173_aadelstoredrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 168, twelve seeds of the GO affine and its control.
    Check(
        claim="round 168: arm level on twelve",
        stated=0.057,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot against the control, twelve seeds",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: arm sd on twelve",
        stated=0.002,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot against the control, twelve seeds",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: control level on twelve",
        stated=0.048,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot against the control, twelve seeds",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: control sd on twelve",
        stated=0.010,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot against the control, twelve seeds",
            "sd_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: paired difference on twelve",
        stated=0.009,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot against the control, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: paired difference sd on twelve",
        stated=0.010,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot against the control, twelve seeds",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: seeds above on twelve",
        stated=10,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot against the control, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 168: seeds paired",
        stated=12,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot against the control, twelve seeds",
            "n",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 168: first-six paired difference",
        stated=0.009,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: first-six seeds above",
        stated=5,
        recompute=transfer(
            "amino acids: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 168: arm lowest seed",
        stated=0.053,
        recompute=aa_seed_level("w170_aadelgo_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: arm highest seed",
        stated=0.059,
        recompute=aa_seed_level("w170_aadelgo_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: arm seed range",
        stated=0.007,
        recompute=aa_seed_level("w170_aadelgo_h12l1p1_e200", "range"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: control lowest seed",
        stated=0.023,
        recompute=aa_seed_level("w141_aadel_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: control highest seed",
        stated=0.059,
        recompute=aa_seed_level("w141_aadel_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: control seed range",
        stated=0.036,
        recompute=aa_seed_level("w141_aadel_h12l1p1_e200", "range"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: control twelve-seed ensemble",
        stated=0.067,
        recompute=ensemble("w141_aadel_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 168: arm twelve-seed ensemble",
        stated=0.058,
        recompute=ensemble("w170_aadelgo_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    # sec:physics -- the sampled cone on the prototroph.
    Check(
        claim="store-baselines: aa sampled cone ridge",
        stated=-0.004,
        recompute=store_baseline(
            "amino_acids", "sampled_mean", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: aa sampled cone kNN",
        stated=0.031,
        recompute=store_baseline("amino_acids", "sampled_mean", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: aa sampled cone forest",
        stated=0.023,
        recompute=store_baseline(
            "amino_acids", "sampled_mean", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="physics: aa sampled cone lethal deletions",
        stated=161,
        recompute=lambda: float(
            __import__("json").load(
                open(osp.join(RESULTS, "target_sampled_flux_store_amino_acids.json"))
            )["n_lethal"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="physics: aa sampled cone distance to point",
        stated=5.9,
        recompute=lambda: float(
            __import__("json").load(
                open(osp.join(RESULTS, "target_sampled_flux_store_amino_acids.json"))
            )["median_rel_l2_log1p_mean_vs_source"]
        ),
        tolerance=0.06,
    ),
    # sec:physics -- the driving-force polytope sampled on the prototroph.
    Check(
        claim="physics: polytope wild-type constrained reactions",
        stated=249,
        recompute=polytope("wild_type", "n_active"),
        tolerance=0.5,
    ),
    Check(
        claim="physics: polytope wild-type binding reactions",
        stated=59,
        recompute=polytope("wild_type", "n_binding"),
        tolerance=0.5,
    ),
    Check(
        claim="physics: polytope wild-type constrained metabolites",
        stated=319,
        recompute=polytope("wild_type", "n_constrained_metabolites"),
        tolerance=0.5,
    ),
    Check(
        claim="physics: polytope wild-type spread over constrained",
        stated=2.08,
        recompute=polytope("wild_type", "median_sd_ln_c_constrained"),
        tolerance=0.006,
    ),
    Check(
        claim="physics: polytope wild-type spread over free",
        stated=2.62,
        recompute=polytope("wild_type", "median_sd_ln_c_free"),
        tolerance=0.006,
    ),
    Check(
        claim="physics: polytope rows differing from wild type",
        stated=405,
        recompute=polytope("n_mean_differs_from_wild_type"),
        tolerance=0.5,
    ),
    Check(
        claim="store-baselines: aa polytope mean ridge",
        stated=0.016,
        recompute=store_baseline(
            "amino_acids", "mdfcap100_poly_mean", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: aa polytope mean kNN",
        stated=0.011,
        recompute=store_baseline(
            "amino_acids", "mdfcap100_poly_mean", "knn", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: aa polytope mean forest",
        stated=0.008,
        recompute=store_baseline(
            "amino_acids", "mdfcap100_poly_mean", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: aa polytope spread ridge",
        stated=0.018,
        recompute=store_baseline(
            "amino_acids", "mdfcap100_poly_spread", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: aa polytope spread kNN",
        stated=0.009,
        recompute=store_baseline(
            "amino_acids", "mdfcap100_poly_spread", "knn", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: aa polytope spread forest",
        stated=0.001,
        recompute=store_baseline(
            "amino_acids", "mdfcap100_poly_spread", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    # sec:target-transfer -- rounds 117 and 118.
    Check(
        claim="target-transfer: amino-acid store arm mean",
        stated=0.025,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid store arm sd",
        stated=0.015,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid control mean",
        stated=0.034,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "mean_spearman",
            "control",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid control sd",
        stated=0.025,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "sd_spearman",
            "control",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid paired difference",
        stated=-0.009,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid paired sd",
        stated=0.018,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: amino-acid control six-seed ensemble",
        stated=0.058,
        recompute=ensemble("w135_aactx_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: amino-acid store six-seed ensemble",
        stated=0.035,
        recompute=ensemble("w135_aastore_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: 027 amino-acid best comparator",
        stated=0.068,
        recompute=transfer_comparator("mulleder19", "027 concentration readout"),
    ),
    Check(
        claim="target-transfer: 027 amino-acid context comparator",
        stated=0.061,
        recompute=transfer_comparator("mulleder19", "027 context-only"),
    ),
    Check(
        claim="target-transfer: 027 amino-acid pooled comparator",
        stated=0.048,
        recompute=transfer_comparator("mulleder19", "027 pooled"),
    ),
    # sec:target-transfer -- round 124, the amino-acid deletion-only graph.
    Check(
        claim="target-transfer: amino-acid deletion-only arm mean",
        stated=0.049,
        recompute=transfer(
            "amino acids: deletion-only cell graphs against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid deletion-only arm sd",
        stated=0.007,
        recompute=transfer(
            "amino acids: deletion-only cell graphs against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid deletion-only paired difference",
        stated=0.015,
        recompute=transfer(
            "amino acids: deletion-only cell graphs against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid deletion-only paired sd",
        stated=0.026,
        recompute=transfer(
            "amino acids: deletion-only cell graphs against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid deletion-only seeds above",
        stated=5,
        recompute=transfer(
            "amino acids: deletion-only cell graphs against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: amino-acid deletion-only six-seed ensemble",
        stated=0.062,
        recompute=ensemble("w141_aadel_h12l1p1_e200", "spearman"),
    ),
    # sec:target-transfer -- round 121, the amino-acid production store.
    Check(
        claim="target-transfer: aa 20-column arm mean",
        stated=0.031,
        recompute=transfer(
            "amino acids: 19 synthesis ratios and the growth ratio against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa 20-column arm sd",
        stated=0.004,
        recompute=transfer(
            "amino acids: 19 synthesis ratios and the growth ratio against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa 20-column paired difference",
        stated=-0.003,
        recompute=transfer(
            "amino acids: 19 synthesis ratios and the growth ratio against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa 20-column paired sd",
        stated=0.025,
        recompute=transfer(
            "amino acids: 19 synthesis ratios and the growth ratio against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa 20-column seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: 19 synthesis ratios and the growth ratio against the control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa 42-summary arm mean",
        stated=0.023,
        recompute=transfer(
            "amino acids: growth vector with all 42 summaries against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa 42-summary arm sd",
        stated=0.009,
        recompute=transfer(
            "amino acids: growth vector with all 42 summaries against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa 42-summary paired difference",
        stated=-0.011,
        recompute=transfer(
            "amino acids: growth vector with all 42 summaries against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa 42-summary paired sd",
        stated=0.027,
        recompute=transfer(
            "amino acids: growth vector with all 42 summaries against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa 42-summary seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: growth vector with all 42 summaries against the control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa 20-column six-seed ensemble",
        stated=0.030,
        recompute=ensemble("w138_aaprodstat_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: aa 42-summary six-seed ensemble",
        stated=0.026,
        recompute=ensemble("w138_aaprodfull_h12l1p1_e200", "spearman"),
    ),
    # sec:target-transfer -- round 126, the amino-acid store on the corrected graph.
    Check(
        claim="target-transfer: aa corrected-graph store arm mean",
        stated=0.040,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store arm sd",
        stated=0.005,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs corrected control",
        stated=-0.009,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against deletion-only graphs",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs corrected control sd",
        stated=0.011,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against deletion-only graphs",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs corrected control seeds above",
        stated=1,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs old-graph store",
        stated=0.015,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against the store affine",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs old-graph store sd",
        stated=0.014,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against the store affine",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs old-graph store seeds above",
        stated=6,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against the store affine",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa corrected-graph store six-seed ensemble",
        stated=0.043,
        recompute=ensemble("w143_aadelstore_h12l1p1_e200", "spearman"),
    ),
    # sec:target-transfer -- round 136, rule counts on the amino acids.
    Check(
        claim="target-transfer: aa rule-count store arm",
        stated=0.033,
        recompute=transfer(
            "amino acids: vector with rule counts on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa rule-count store arm sd",
        stated=0.009,
        recompute=transfer(
            "amino acids: vector with rule counts on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa rule-count store vs control",
        stated=-0.016,
        recompute=transfer(
            "amino acids: vector with rule counts on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa rule-count store vs control sd",
        stated=0.012,
        recompute=transfer(
            "amino acids: vector with rule counts on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa rule-count store vs control seeds above",
        stated=1,
        recompute=transfer(
            "amino acids: vector with rule counts on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa rule-count store vs plain store",
        stated=-0.007,
        recompute=transfer(
            "amino acids: vector with rule counts on deletion-only graphs against the store affine there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa rule-count store vs plain store sd",
        stated=0.013,
        recompute=transfer(
            "amino acids: vector with rule counts on deletion-only graphs against the store affine there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa rule-count store vs plain store seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: vector with rule counts on deletion-only graphs against the store affine there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa plain store level",
        stated=0.04,
        recompute=transfer(
            "amino acids: vector with rule counts on deletion-only graphs against the store affine there",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa rule-count store six-seed ensemble",
        stated=0.037,
        recompute=ensemble("w150_aadelgpr_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa plain store six-seed ensemble",
        stated=0.043,
        recompute=ensemble("w143_aadelstore_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:target-transfer -- round 137, the solve head on the amino acids.
    Check(
        claim="target-transfer: aa solve head arm",
        stated=0.053,
        recompute=transfer(
            "amino acids: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa solve head arm sd",
        stated=0.008,
        recompute=transfer(
            "amino acids: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa solve head vs control",
        stated=0.004,
        recompute=transfer(
            "amino acids: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa solve head vs control sd",
        stated=0.011,
        recompute=transfer(
            "amino acids: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa solve head vs control seeds above",
        stated=4,
        recompute=transfer(
            "amino acids: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa solve head vs store affine",
        stated=0.013,
        recompute=transfer(
            "amino acids: solve head (hidden 256) on deletion-only graphs against the store affine there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa solve head vs store affine sd",
        stated=0.008,
        recompute=transfer(
            "amino acids: solve head (hidden 256) on deletion-only graphs against the store affine there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa solve head vs store affine seeds above",
        stated=6,
        recompute=transfer(
            "amino acids: solve head (hidden 256) on deletion-only graphs against the store affine there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa solve head six-seed ensemble",
        stated=0.063,
        recompute=ensemble("w151_aadelhead256_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:target-transfer -- round 129, the 300-epoch amino-acid control.
    Check(
        claim="target-transfer: aa 300-epoch control",
        stated=0.049,
        recompute=transfer(
            "amino acids: deletion-only graphs for 300 epochs against 200",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa 300-epoch control sd",
        stated=0.009,
        recompute=transfer(
            "amino acids: deletion-only graphs for 300 epochs against 200",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa 200-epoch control",
        stated=0.049,
        recompute=transfer(
            "amino acids: deletion-only graphs for 300 epochs against 200",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa 200-epoch control sd",
        stated=0.007,
        recompute=transfer(
            "amino acids: deletion-only graphs for 300 epochs against 200",
            "sd_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa 300 vs 200 epochs",
        stated=-0.001,
        recompute=transfer(
            "amino acids: deletion-only graphs for 300 epochs against 200",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa 300 vs 200 epochs sd",
        stated=0.003,
        recompute=transfer(
            "amino acids: deletion-only graphs for 300 epochs against 200",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa 300 vs 200 epochs seeds above",
        stated=1,
        recompute=transfer(
            "amino acids: deletion-only graphs for 300 epochs against 200", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa 300-epoch six-seed ensemble",
        stated=0.060,
        recompute=ensemble("w146_aadel_h12l1p1_e300", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa 200-epoch six-seed ensemble",
        stated=0.062,
        recompute=ensemble("w141_aadel_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:target-transfer -- round 122, the wide amino-acid control.
    Check(
        claim="target-transfer: aa wide control arm",
        stated=0.034,
        recompute=transfer(
            "amino acids: width 90, two blocks, against the width-12 control",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa wide control arm sd",
        stated=0.010,
        recompute=transfer(
            "amino acids: width 90, two blocks, against the width-12 control",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa wide control vs width-12 control",
        stated=0.000,
        recompute=transfer(
            "amino acids: width 90, two blocks, against the width-12 control",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa wide control vs width-12 control sd",
        stated=0.031,
        recompute=transfer(
            "amino acids: width 90, two blocks, against the width-12 control",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: aa wide control seeds above",
        stated=3,
        recompute=transfer(
            "amino acids: width 90, two blocks, against the width-12 control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa wide control six-seed ensemble",
        stated=0.050,
        recompute=ensemble("w139_aawide_h90l2p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:target-physics -- the direction rule, the MOMA and HiGHS stores, degeneracy.
    Check(
        claim="target-physics: aa sign10 conflicts",
        stated=84,
        recompute=physics_store(
            "amino_acids", "sign10", "thermo", "n_conflicts_with_plain_wild_type_flux"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: aa sign30 conflicts",
        stated=44,
        recompute=physics_store(
            "amino_acids", "sign30", "thermo", "n_conflicts_with_plain_wild_type_flux"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: aa MOMA lethal",
        stated=175,
        recompute=physics_store("amino_acids", "moma", "n_lethal"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: aa MOMA lethal in the source",
        stated=161,
        recompute=physics_store("amino_acids", "moma", "n_lethal_source"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: aa MOMA ratios moved",
        stated=63,
        recompute=physics_store("amino_acids", "moma", "n_genes_ratio_moved_gt_0.01"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: aa MOMA vectors moved",
        stated=567,
        recompute=physics_store("amino_acids", "moma", "n_genes_log1p_vector_moved"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: aa MOMA median distance",
        stated=0.0,
        recompute=physics_store(
            "amino_acids", "moma", "median_relative_l2_log1p_to_source"
        ),
        tolerance=0.0005,
    ),
    Check(
        claim="target-physics: SM store no-effect vectors differing",
        stated=324,
        recompute=degeneracy("SM_gem", "n_no_effect_vector_differs"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: SM store no-effect deletions",
        stated=918,
        recompute=degeneracy("SM_gem", "n_no_effect"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: aa MOMA store no-effect vectors differing",
        stated=11,
        recompute=degeneracy("SM_gem_moma", "n_no_effect_vector_differs"),
        tolerance=0.5,
    ),
    # sec:target-physics -- the store read without the backbone.
    Check(
        claim="store-baselines: aa HiGHS store kNN",
        stated=0.022,
        recompute=store_baseline("amino_acids", "pfba_highs", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: aa ProtT5 forest",
        stated=0.052,
        recompute=store_baseline(
            "amino_acids", "prot_T5_all", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: aa HiGHS store with ProtT5 forest",
        stated=0.058,
        recompute=store_baseline(
            "amino_acids", "pfba_highs+prot_T5_all", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    # sec:target-physics -- Mulleder 2016 against Cooper 2010.
    Check(
        claim="concordance: shared strains",
        stated=4051,
        recompute=concordance_summary("n_shared_strains"),
        tolerance=0.5,
    ),
    Check(
        claim="concordance: alanine Spearman",
        stated=-0.006,
        recompute=concordance("alanine", "alanine", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="concordance: glycine Spearman",
        stated=0.059,
        recompute=concordance("glycine", "glycine", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="concordance: glycine p",
        stated=0.0002,
        recompute=concordance("glycine", "glycine", "spearman_p"),
        tolerance=0.00005,
    ),
    Check(
        claim="concordance: glutamate Spearman",
        stated=0.043,
        recompute=concordance("glutamate", "glutamate", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="concordance: aspartate Spearman",
        stated=0.031,
        recompute=concordance("aspartate", "aspartate", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="concordance: lysine b top-5 overlap",
        stated=22,
        recompute=concordance("lysine_b", "lysine", "top5_overlap"),
        tolerance=0.5,
    ),
    Check(
        claim="concordance: lysine b top-5 expected",
        stated=10.1,
        recompute=concordance("lysine_b", "lysine", "top5_expected"),
        tolerance=0.06,
    ),
    Check(
        claim="concordance: lysine b top-5 enrichment",
        stated=2.2,
        recompute=concordance("lysine_b", "lysine", "top5_enrichment"),
        tolerance=0.06,
    ),
    Check(
        claim="concordance: matrix median |rho|",
        stated=0.018,
        recompute=concordance_summary("matrix_abs_median"),
        tolerance=0.0006,
    ),
    Check(
        claim="concordance: matrix max |rho|",
        stated=0.127,
        recompute=concordance_summary("matrix_abs_max"),
        tolerance=0.0006,
    ),
    Check(
        claim="concordance: Mulleder retest Spearman mean",
        stated=0.20,
        recompute=concordance_summary("mulleder_retest_spearman_mean"),
        tolerance=0.005,
    ),
    Check(
        claim="concordance: Mulleder retest Pearson mean",
        stated=0.43,
        recompute=concordance_summary("mulleder_retest_pearson_mean"),
        tolerance=0.005,
    ),
    Check(
        claim="concordance: Cooper duplicate spread peaks",
        stated=17,
        recompute=concordance_summary("cooper_spread_n_peaks"),
        tolerance=0.5,
    ),
    Check(
        claim="concordance: Cooper duplicate rows sd min",
        stated=0.1,
        recompute=concordance_summary("cooper_spread_sd_dup_min"),
        tolerance=0.006,
    ),
    Check(
        claim="concordance: Cooper duplicate rows sd max",
        stated=0.53,
        recompute=concordance_summary("cooper_spread_sd_dup_max"),
        tolerance=0.006,
    ),
    Check(
        claim="concordance: Cooper all rows sd min",
        stated=0.11,
        recompute=concordance_summary("cooper_spread_sd_all_min"),
        tolerance=0.006,
    ),
    Check(
        claim="concordance: Cooper all rows sd max",
        stated=0.51,
        recompute=concordance_summary("cooper_spread_sd_all_max"),
        tolerance=0.006,
    ),
    Check(
        claim="concordance: Cooper duplicate rows sd median",
        stated=0.29,
        recompute=concordance_summary("cooper_spread_sd_dup_median"),
        tolerance=0.006,
    ),
    Check(
        claim="concordance: Cooper all rows sd median",
        stated=0.28,
        recompute=concordance_summary("cooper_spread_sd_all_median"),
        tolerance=0.006,
    ),
    Check(
        claim="concordance: Cooper duplicate Spearman mean",
        stated=0.06,
        recompute=concordance_summary("cooper_retest_spearman_mean"),
        tolerance=0.005,
    ),
    Check(
        claim="concordance: Cooper duplicate log-Pearson mean",
        stated=0.10,
        recompute=concordance_summary("cooper_retest_pearson_log_mean"),
        tolerance=0.005,
    ),
]


def main() -> None:
    failed = 0
    for check in CHECKS:
        got = check.recompute()
        agrees = abs(got - check.stated) < check.tolerance
        failed += 0 if agrees else 1
        mark = "ok  " if agrees else "WRONG"
        print(f"{mark} {check.claim:58s} document {check.stated:,g}  files {got:,.4f}")
    print(f"\n{len(CHECKS)} checks, {failed} disagreeing with the document")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
