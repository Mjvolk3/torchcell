# experiments/028-gene-essentiality/scripts/verify_doc_numbers.py
# [[experiments.028-gene-essentiality.scripts.verify_doc_numbers]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/verify_doc_numbers

"""Every number typed into the document's prose, checked against the file it came from.

Tables and figures are written by scripts, so they cannot drift. Prose numbers are typed
by hand, and ``make check`` validates style, citations, provenance and float sizing but
never whether a number in a sentence still matches its source. A number that was right
when written and wrong after a rerun survives every gate the build has.

Each check names the claim, the quantity as the document states it, and how to recompute
it. Add one whenever a number is written into prose; a claim with no check here is a
claim nobody can rebuild. Exits non-zero on any mismatch.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/verify_doc_numbers.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import sys
from collections.abc import Callable

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")


class Check(BaseModel):
    """One number the prose states, and the recomputation that has to agree with it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Where the number appears, as a section label plus the claim in a few words.
    claim: str
    #: The value as the document prints it.
    stated: float
    #: Recomputes the value from ``results/``.
    recompute: Callable[[], float]
    #: Allowed difference. Prose rounds, so this is the rounding step, not a tolerance
    #: for disagreement: 0.0006 for a three-decimal number, 0.5 for an integer.
    tolerance: float = 0.0006


def seed_mean(*patterns: str) -> Callable[[], float]:
    """The mean held-out AUROC over every cell matching the patterns."""

    def run() -> float:
        values = [
            float(json.load(open(p))["test"]["pinned"]["auroc"])
            for pat in patterns
            for p in sorted(glob.glob(osp.join(RESULTS, pat)))
        ]
        assert values, f"no cells matched {patterns}"
        return float(np.mean(values))

    return run


def field(name: str, *path: str | int) -> Callable[[], float]:
    """A scalar reached by walking keys and indices from a result file's root."""

    def run() -> float:
        node = json.load(open(osp.join(RESULTS, name)))
        for key in path:
            node = node[key]
        return float(node)

    return run


def consensus(*keys: str | int) -> Callable[[], float]:
    """One value of ``consensus_missed.json``, addressed by a key path."""

    def run() -> float:
        node = json.load(open(osp.join(RESULTS, "consensus_missed.json")))
        for k in keys:
            node = node[k]
        return float(node)

    return run


def consensus_term(name: str, key: str) -> Callable[[], float]:
    """One statistic of one enriched term among the essential genes."""

    def run() -> float:
        report = json.load(open(osp.join(RESULTS, "consensus_missed.json")))
        top = report["classes"][0]["top"]
        return float(next(t for t in top if t["name"] == name)[key])

    return run


def simplification(block: str, name: str, key: str) -> Callable[[], float]:
    """One statistic of one named row of the simplification readout."""

    def run() -> float:
        rows = json.load(open(osp.join(RESULTS, "simplification_readout.json")))[block]
        tag = "label" if block == "ladder" else "name"
        return float(next(r for r in rows if r[tag] == name)[key])

    return run


def dcell(read_name: str, arm: str, key: str) -> Callable[[], float]:
    """One statistic of one DCell arm from its own result files."""

    def run() -> float:
        with open(osp.join(RESULTS, f"dcell_style_{read_name}.json")) as fh:
            payload = json.load(fh)
        block = {
            "released": lambda: next(r for r in payload["rows"] if r["model"] == arm),
            "folds": lambda: next(
                e for e in payload["fold_ensembles"] if e["arm"] == arm
            ),
            "resplits": lambda: next(p for p in payload["paired"] if p["arm"] == arm),
        }[read_name]()
        return float(block[key])

    return run


def calibration_growth(tier: str, scale: float) -> Callable[[], float]:
    """Median growth at one capacity scale under one turnover tier."""

    def run() -> float:
        rows = json.load(open(osp.join(RESULTS, f"capacity_calibration_{tier}.json")))[
            "rows"
        ]
        return float(
            next(r["growth_median"] for r in rows if r["capacity_scale"] == scale)
        )

    return run


def store_stat(store_contains: str, key: str) -> Callable[[], float]:
    """One statistic of one store from the discriminability read."""

    def run() -> float:
        stores = json.load(open(osp.join(RESULTS, "store_discriminability.json")))[
            "stores"
        ]
        return float(next(s for s in stores if store_contains in s["store"])[key])

    return run


def enrichment(
    held_out: str, model: str, gene_class: str, term_name: str, key: str
) -> Callable[[], float]:
    """One statistic of one term in one family of the misranked-gene enrichment."""

    def run() -> float:
        fams = json.load(open(osp.join(RESULTS, "misclass_enrichment.json")))[
            "families"
        ]
        fam = next(
            f
            for f in fams
            if held_out in f["held_out_set"]
            and f["model"] == model
            and f["gene_class"].startswith(gene_class)
        )
        rows = fam["named_features"] + fam["top_specific"] + fam["top"]
        return float(next(r for r in rows if r["name"] == term_name)[key])

    return run


def genome_arm(label_contains: str) -> Callable[[], float]:
    """Pooled AUROC of one arm on the genome-wide held-out set."""

    def run() -> float:
        arms = json.load(open(osp.join(RESULTS, "genome_heldout_readout.json")))["arms"]
        return float(
            next(a for a in arms if label_contains in a["label"])["pooled_auroc"]
        )

    return run


def resplit(*path: str | int) -> Callable[[], float]:
    """A scalar of the round-75 repeated-outer-split read."""
    return field("resplit_readout.json", *path)


def resplit_go_seed_mean() -> float:
    """Seed-mean AUROC of graph convolution on GO over the ten repeated outer splits."""
    means = []
    for r in range(10):
        rows = json.load(
            open(osp.join(RESULTS, f"graph_baselines_go400_fcl_resplit{r}.json"))
        )["rows"]
        means.append(next(x for x in rows if x["inputs"] == "GO (400 epochs)")["mean"])
    return float(np.mean(means))


def gpr(block: str, key: str) -> Callable[[], float]:
    """One statistic of the round-77 gene-rule-column read."""
    return field("gpr_readout.json", block, key)


def cls_arm(index: int, key: str) -> Callable[[], float]:
    """One statistic of one arm of the round-79 perturbed-class-token read."""
    return field("cls_readout.json", "arms", index, key)


CHECKS: list[Check] = [
    # sec:store-discriminability -- the stores and the two cuts.
    Check(
        claim="store-discriminability: vertex store",
        stated=0.840,
        recompute=seed_mean("w52_pathlogsimplex_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="store-discriminability: clamp columns cut",
        stated=0.888,
        recompute=seed_mean(
            "w84_pathlogsimplexdeclamp_h12l1p1_e200_s*.json",
            "w85_pathlogsimplexdeclamp_h12l1p1_e200_s*.json",
        ),
    ),
    Check(
        claim="store-discriminability: non-clamp cut (control)",
        stated=0.846,
        recompute=seed_mean("w86_pathlogsimplextopvar_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="store-discriminability: pFBA store",
        stated=0.908,
        # Round 68's own six seeds; round 81 added three more of this arm as controls.
        recompute=seed_mean("w49_pathfbapfbalogfree_h12l1p1_e200_s270[0-5].json"),
    ),
    Check(
        claim="store-discriminability: dose, 25 columns",
        stated=0.874,
        recompute=seed_mean("w87_pathlogsimplexdeclamp25_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="store-discriminability: dose, 100 columns",
        stated=0.896,
        recompute=seed_mean("w87_pathlogsimplexdeclamp100_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="store-discriminability: cut store fold ensemble",
        stated=0.901,
        recompute=field("w90_pathlogsimplexdeclampcv_ensemble.json", "ensemble_auroc"),
    ),
    Check(
        claim="store-discriminability: vertex clamp columns in the widest ten",
        stated=10,
        tolerance=0.5,
        recompute=store_stat("simplex", "n_clamp_columns_in_top10"),
    ),
    Check(
        claim="store-discriminability: vertex columns ranking above 0.7 alone",
        stated=298,
        tolerance=0.5,
        recompute=store_stat("simplex", "n_columns_auroc_above_0_7"),
    ),
    Check(
        claim="store-discriminability: columns the clamp rule cuts from the vertex store",
        stated=429,
        tolerance=0.5,
        recompute=field("declamp_store.json", "stores", 0, "n_columns_dropped"),
    ),
    Check(
        claim="store-discriminability: columns the same rule cuts from pFBA",
        stated=1,
        tolerance=0.5,
        recompute=field("declamp_store.json", "stores", 1, "n_columns_dropped"),
    ),
    # sec:kcat-tier -- the capacity box and the three forced constructions.
    Check(
        claim="kcat-tier: measured median nominal ceiling",
        stated=34.6,
        tolerance=0.05,
        recompute=field(
            "capacity_calibration_measured.json",
            "parameters",
            "nominal_ceiling_quantiles",
            "0.5",
        ),
    ),
    Check(
        claim="kcat-tier: predicted median nominal ceiling",
        stated=16.5,
        tolerance=0.05,
        recompute=field(
            "capacity_calibration_predicted.json",
            "parameters",
            "nominal_ceiling_quantiles",
            "0.5",
        ),
    ),
    Check(
        claim="kcat-tier: measured growth at scale 100",
        stated=0.156,
        recompute=calibration_growth("measured", 100.0),
    ),
    Check(
        claim="kcat-tier: predicted growth at scale 100",
        stated=0.103,
        recompute=calibration_growth("predicted", 100.0),
    ),
    Check(
        claim="kcat-tier: measured growth at scale 1,000",
        stated=0.532,
        recompute=calibration_growth("measured", 1000.0),
    ),
    Check(
        claim="kcat-tier: predicted growth at scale 1,000",
        stated=0.203,
        recompute=calibration_growth("predicted", 1000.0),
    ),
    Check(
        claim="kcat-tier: hard forced column, twelve seeds",
        stated=0.699,
        recompute=seed_mean(
            "w30_lpcapbio100_h12l1p1_e200_s*.json",
            "w88_lpcapbio100_h12l1p1_e200_s*.json",
            "w89_lpcapbio100_h12l1p1_e200_s*.json",
        ),
    ),
    Check(
        claim="kcat-tier: forced column on the looser tier",
        stated=0.707,
        recompute=seed_mean(
            "w83_lpcapbio100meas_h12l1p1_e200_s*.json",
            "w88_lpcapbio100meas_h12l1p1_e200_s*.json",
        ),
    ),
    Check(
        claim="kcat-tier: forced column on the elastic program",
        stated=0.715,
        recompute=seed_mean(
            "w32_lpelbio100_h12l1p1_e200_s*.json", "w89_lpelbio100_h12l1p1_e200_s*.json"
        ),
    ),
    # sec:store-scale -- what sizes the store over the solid-growth build.
    Check(
        claim="store-scale: distinct (subset, medium) keys",
        stated=469474,
        tolerance=0.5,
        recompute=field("store_key_count.json", "n_distinct_keys"),
    ),
    Check(
        claim="store-scale: records in the build",
        stated=13525071,
        tolerance=0.5,
        recompute=field("store_key_count.json", "n_records"),
    ),
    Check(
        claim="store-scale: build wall time, minutes",
        stated=71,
        tolerance=0.6,
        recompute=lambda: field("solid_growth_store_SC_ipm.json", "seconds")() / 60.0,
    ),
    # sec:summary -- the one-page conclusions.
    Check(
        claim="summary: GO graph + cached solve head, fold average",
        stated=0.954,
        recompute=lambda: next(
            c["auroc"]
            for c in json.load(open(osp.join(RESULTS, "family_ensembles_folds.json")))[
                "combinations"
            ]
            if len(c["families"]) == 2
            and any("GO convolution" in f for f in c["families"])
            and any("cached solve" in f for f in c["families"])
        ),
    ),
    Check(
        claim="summary: layer-free fold ensemble",
        stated=0.920,
        recompute=field("w50_pathlogfreecv_ensemble.json", "ensemble_auroc"),
    ),
    Check(
        claim="summary: GO convolution fold ensemble",
        stated=0.917,
        recompute=field(
            "graph_baselines_go_folds.json", "arms", "GCN400", "ensemble_auroc"
        ),
    ),
    Check(
        claim="summary: no metabolism",
        stated=0.858,
        recompute=seed_mean("w39_pathpool_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="summary: FBA ratios on the logit",
        stated=0.890,
        recompute=seed_mean("w38_protpathfba_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="summary: solve vector on the logit",
        stated=0.911,
        recompute=seed_mean("w49_pathfbapfbalogfree_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="summary: cached solve, flux head",
        stated=0.882,
        recompute=seed_mean("w43_pathlpoffgateflux_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="summary: learned ceilings, flux head",
        stated=0.845,
        recompute=seed_mean("w39_pathlpelgateflux_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="summary: learned ceilings, gated growth",
        stated=0.867,
        recompute=seed_mean("w39_pathlpelgate100_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="summary: genome-wide, layer-free transformer",
        stated=0.570,
        recompute=genome_arm("layer-free log1p"),
    ),
    Check(
        claim="summary: genome-wide, graph convolution on ProtT5",
        stated=0.755,
        recompute=genome_arm("union of nine, ProtT5"),
    ),
    Check(
        claim="summary: essentials FBA calls viable, transformer effect",
        stated=0.91,
        tolerance=0.006,
        recompute=enrichment(
            "released",
            "transformer, layer-free solve",
            "essential",
            "FBA knockout growth ratio 1.0",
            "effect",
        ),
    ),
    Check(
        claim="summary: the same, misrank of the annotated",
        stated=0.134,
        recompute=enrichment(
            "released",
            "transformer, layer-free solve",
            "essential",
            "FBA knockout growth ratio 1.0",
            "mean_misrank_annotated",
        ),
    ),
    Check(
        claim="summary: the same under the GO graph model",
        stated=0.51,
        tolerance=0.006,
        recompute=enrichment(
            "released",
            "graph convolution on GO",
            "essential",
            "FBA knockout growth ratio 1.0",
            "effect",
        ),
    ),
    Check(
        claim="summary: steroid biosynthesis, graph model",
        stated=0.96,
        tolerance=0.006,
        recompute=enrichment(
            "released",
            "graph convolution on GO",
            "nonessential",
            "steroid biosynthetic process",
            "effect",
        ),
    ),
    Check(
        claim="summary: genome-wide essentials of low degree, graph model",
        stated=0.70,
        tolerance=0.006,
        recompute=enrichment(
            "genome-wide",
            "graph convolution on ProtT5",
            "essential",
            "physical degree under the class median",
            "effect",
        ),
    ),
    Check(
        claim="summary: genome-wide rRNA processing, graph model",
        stated=0.92,
        tolerance=0.006,
        recompute=enrichment(
            "genome-wide",
            "graph convolution on ProtT5",
            "nonessential",
            "rRNA processing",
            "effect",
        ),
    ),
    # sec:round75 -- ten repeated outer splits.
    Check(
        claim="round75: average, mean of ten",
        stated=0.896,
        recompute=resplit("mean_auroc", "average"),
    ),
    Check(
        claim="round75: GO convolution, mean of ten",
        stated=0.869,
        recompute=resplit("mean_auroc", "go_convolution"),
    ),
    Check(
        claim="round75: layer-free, mean of ten",
        stated=0.848,
        recompute=resplit("mean_auroc", "layer_free"),
    ),
    Check(
        claim="round75: average minus layer-free",
        stated=0.048,
        recompute=resplit("differences", 0, "mean"),
    ),
    Check(
        claim="round75: average minus GO convolution",
        stated=0.026,
        recompute=resplit("differences", 1, "mean"),
    ),
    Check(
        claim="round75: layer-free minus GO convolution",
        stated=-0.022,
        recompute=resplit("differences", 2, "mean"),
    ),
    Check(
        claim="round75: average minus the better single",
        stated=0.023,
        recompute=resplit("differences", 3, "mean"),
    ),
    Check(
        claim="round75: better-single positive splits",
        stated=9,
        tolerance=0.5,
        recompute=resplit("differences", 3, "n_positive"),
    ),
    Check(
        claim="round75: layer-free minus FBA ratio affine",
        stated=0.011,
        recompute=resplit("differences", 4, "mean"),
    ),
    Check(
        claim="round75: its positive splits",
        stated=7,
        tolerance=0.5,
        recompute=resplit("differences", 4, "n_positive"),
    ),
    Check(
        claim="round75: layer-free minus no metabolism",
        stated=0.047,
        recompute=resplit("differences", 5, "mean"),
    ),
    Check(
        claim="round75: layer-free cells, seed mean over thirty",
        stated=0.843,
        recompute=seed_mean("w93_resplit*pathlogfree_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="round75: GO convolution, seed mean over ten splits",
        stated=0.868,
        recompute=resplit_go_seed_mean,
    ),
    Check(
        claim="hard-genes: FBA-viable essentials fully covered by alternatives",
        stated=4,
        tolerance=0.5,
        recompute=field(
            "isozyme_affine_store.json",
            "held_out_essential_fba_viable",
            "n_fully_covered_by_alternatives",
        ),
    ),
    Check(
        claim="hard-genes: FBA-viable essentials with a sole reaction",
        stated=15,
        tolerance=0.5,
        recompute=field(
            "isozyme_affine_store.json",
            "held_out_essential_fba_viable",
            "n_with_a_sole_reaction",
        ),
    ),
    # sec:round76 -- the physical degree as affine columns.
    Check(
        claim="round76: control, wild-type rows, seed mean",
        stated=0.566,
        recompute=seed_mean("w95_genomepathlogwt_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="round76: degree columns, genome-wide seed mean",
        stated=0.602,
        recompute=seed_mean("w95_genomepathlogwtdeg_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="round76: degree columns, genome-wide pooled",
        stated=0.593,
        recompute=genome_arm("physical degree and clustering in the affine"),
    ),
    Check(
        claim="round76: degree columns on the released split",
        stated=0.906,
        recompute=seed_mean("w95_pathlogwtdeg_h12l1p1_e200_s*.json"),
    ),
    # sec:round77 -- the gene-protein-reaction redundancy columns.
    Check(
        claim="round77: released split, with the columns",
        stated=0.918,
        recompute=gpr("released_auroc", "mean_with"),
    ),
    Check(
        claim="round77: released split, without",
        stated=0.908,
        recompute=gpr("released_auroc", "mean_without"),
    ),
    Check(
        claim="round77: released split, positive seeds",
        stated=6,
        tolerance=0.5,
        recompute=gpr("released_auroc", "n_positive"),
    ),
    Check(
        claim="round77: ten splits, with the columns",
        stated=0.862,
        recompute=gpr("resplit_auroc", "mean_with"),
    ),
    Check(
        claim="round77: ten splits, without",
        stated=0.843,
        recompute=gpr("resplit_auroc", "mean_without"),
    ),
    Check(
        claim="round77: ten splits, difference",
        stated=0.019,
        recompute=gpr("resplit_auroc", "mean_difference"),
    ),
    Check(
        claim="round77: ten splits, positive",
        stated=8,
        tolerance=0.5,
        recompute=gpr("resplit_auroc", "n_positive"),
    ),
    Check(
        claim="round77: misrank of the 19, with",
        stated=0.120,
        recompute=gpr("released_misrank_fba_viable_essential", "mean_with"),
    ),
    Check(
        claim="round77: misrank of the 15 sole-reaction genes, with",
        stated=0.089,
        recompute=gpr("released_misrank_sole_reaction", "mean_with"),
    ),
    Check(
        claim="round77: the same, without",
        stated=0.124,
        recompute=gpr("released_misrank_sole_reaction", "mean_without"),
    ),
    Check(
        claim="round77: misrank of the 4 covered genes, with",
        stated=0.234,
        recompute=gpr("released_misrank_covered_by_alternatives", "mean_with"),
    ),
    Check(
        claim="round77: the same, without",
        stated=0.171,
        recompute=gpr("released_misrank_covered_by_alternatives", "mean_without"),
    ),
    # sec:round75 -- round 78's six-seed repeat of the vector-against-ratios read.
    Check(
        claim="round78: vector minus ratios, six-seed means",
        stated=0.010,
        recompute=resplit("differences", 6, "mean"),
    ),
    Check(
        claim="round78: its spread",
        stated=0.025,
        tolerance=0.0006,
        recompute=resplit("differences", 6, "sd"),
    ),
    Check(
        claim="round78: its positive splits",
        stated=8,
        tolerance=0.5,
        recompute=resplit("differences", 6, "n_positive"),
    ),
    # sec:round79 and sec:round81 -- the perturbed class token, fifteen pairs.
    Check(
        claim="round79: layer-free with the flag",
        stated=0.905,
        recompute=cls_arm(0, "mean_with"),
    ),
    Check(
        claim="round79: layer-free without",
        stated=0.911,
        recompute=cls_arm(0, "mean_without"),
    ),
    Check(
        claim="round79: layer-free paired difference",
        stated=-0.0054,
        recompute=cls_arm(0, "mean_difference"),
    ),
    Check(
        claim="round79: its positive seeds",
        stated=2,
        tolerance=0.5,
        recompute=cls_arm(0, "n_positive"),
    ),
    Check(
        claim="round79: its pairs", stated=9, tolerance=0.5, recompute=cls_arm(0, "n")
    ),
    Check(
        claim="round79: pooled cell with the flag",
        stated=0.855,
        recompute=cls_arm(1, "mean_with"),
    ),
    Check(
        claim="round79: pooled cell without",
        stated=0.858,
        recompute=cls_arm(1, "mean_without"),
    ),
    Check(
        claim="round79: its paired difference",
        stated=-0.0028,
        recompute=cls_arm(1, "mean_difference"),
    ),
    Check(
        claim="round79: its positive seeds",
        stated=1,
        tolerance=0.5,
        recompute=cls_arm(1, "n_positive"),
    ),
    Check(
        claim="round79: its pairs", stated=6, tolerance=0.5, recompute=cls_arm(1, "n")
    ),
    Check(
        claim="round79: class-token spread with the flag",
        stated=0.52,
        tolerance=0.006,
        recompute=cls_arm(0, "cls_strain_sd_with"),
    ),
    Check(
        claim="round79: class-token spread without",
        stated=0.0,
        tolerance=0.0006,
        recompute=cls_arm(0, "cls_strain_sd_without"),
    ),
    Check(
        claim="round80: rule-count fold ensemble",
        stated=0.934,
        recompute=field("w99_pathloggprcv_ensemble.json", "ensemble_auroc"),
    ),
    Check(
        claim="round80: its fold mean",
        stated=0.913,
        recompute=field("w99_pathloggprcv_ensemble.json", "fold_mean_auroc"),
    ),
    Check(
        claim="round80: GO convolution averaged with it",
        stated=0.955,
        recompute=lambda: max(
            c["auroc"]
            for c in json.load(open(osp.join(RESULTS, "family_ensembles_folds.json")))[
                "combinations"
            ]
        ),
    ),
    # sec:round82 -- the factorial of the solve and the rules.
    Check(
        claim="round82: ratios only",
        stated=0.890,
        recompute=seed_mean("w38_protpathfba_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="round82: rule counts alone",
        stated=0.902,
        recompute=seed_mean("w101_pathgpronly_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="round82: rule counts alone, fold ensemble",
        stated=0.909,
        recompute=field("w101_pathgpronlycv_ensemble.json", "ensemble_auroc"),
    ),
    # sec:round83 -- the single-experiment ablation.
    Check(
        claim="round83: YPD only, ratio + vector + counts",
        stated=0.916,
        recompute=seed_mean("w102_ypdgpr_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="round83: YPD only, ratio + vector",
        stated=0.912,
        recompute=seed_mean("w102_ypdvec_h12l1p1_e200_s*.json"),
    ),
    Check(
        claim="round83: YPD only, ratio alone",
        stated=0.877,
        recompute=seed_mean("w102_ypdonly_h12l1p1_e200_s*.json"),
    ),
    # sec:round84-98 -- what each simplification costs.
    Check(
        claim="round84-98: one medium for four, solve and cou / mean_difference",
        stated=0.002,
        recompute=simplification(
            "resplit",
            "one medium for four, flux vector and counts kept",
            "mean_difference",
        ),
    ),
    Check(
        claim="round84-98: one medium for four, solve and cou / n_positive",
        stated=6,
        recompute=simplification(
            "resplit", "one medium for four, flux vector and counts kept", "n_positive"
        ),
    ),
    Check(
        claim="round84-98: counts over the four ratios alone / mean_difference",
        stated=0.013,
        recompute=simplification(
            "resplit", "counts over the four ratios alone", "mean_difference"
        ),
    ),
    Check(
        claim="round84-98: flux vector over the four ratios a / mean_difference",
        stated=0.012,
        recompute=simplification(
            "resplit", "flux vector over the four ratios alone", "mean_difference"
        ),
    ),
    Check(
        claim="round84-98: counts added to the flux vector, f / mean_difference",
        stated=0.019,
        recompute=simplification(
            "resplit", "counts added to the flux vector, four media", "mean_difference"
        ),
    ),
    Check(
        claim="round84-98: no flux vector, counts kept, four media / mean_difference",
        stated=-0.018,
        recompute=simplification(
            "resplit", "no flux vector, counts kept, four media", "mean_difference"
        ),
    ),
    Check(
        claim="round84-98: no flux vector, counts kept, four media / n_positive",
        stated=1,
        recompute=simplification(
            "resplit", "no flux vector, counts kept, four media", "n_positive"
        ),
    ),
    Check(
        claim="round84-98: one medium for four, no flux vector / mean_difference",
        stated=-0.011,
        recompute=simplification(
            "resplit", "one medium for four, no flux vector", "mean_difference"
        ),
    ),
    Check(
        claim="round84-98: no solve and one medium, against t / mean_difference",
        stated=-0.029,
        recompute=simplification(
            "resplit",
            "no flux vector and one medium, against the full recipe",
            "mean_difference",
        ),
    ),
    Check(
        claim="round84-98: no solve and one medium, against t / n_positive",
        stated=0,
        recompute=simplification(
            "resplit",
            "no flux vector and one medium, against the full recipe",
            "n_positive",
        ),
    ),
    Check(
        claim="round84-98: YPD and SC ratios, rule counts / cell_mean",
        stated=0.896,
        recompute=simplification(
            "ladder", "YPD and SC ratios, rule counts", "cell_mean"
        ),
    ),
    Check(
        claim="round84-98: YPD ratio, rule counts / cell_mean",
        stated=0.888,
        recompute=simplification("ladder", "YPD ratio, rule counts", "cell_mean"),
    ),
    Check(
        claim="round84-98: YPD ratio, rule counts / fold_ensemble_one_seed",
        stated=0.902,
        recompute=simplification(
            "ladder", "YPD ratio, rule counts", "fold_ensemble_one_seed"
        ),
    ),
    Check(
        claim="round84-98: four ratios, rule counts / cell_mean",
        stated=0.902,
        recompute=simplification("ladder", "four ratios, rule counts", "cell_mean"),
    ),
    Check(
        claim="round84-98: four ratios, flux vector, rule cou / fold_ensemble_three_seeds",
        stated=0.933,
        recompute=simplification(
            "ladder",
            "four ratios, flux vector, rule counts",
            "fold_ensemble_three_seeds",
        ),
    ),
    Check(
        claim="round84-98: YPD ratio, flux vector, rule count / fold_ensemble_three_seeds",
        stated=0.932,
        recompute=simplification(
            "ladder", "YPD ratio, flux vector, rule counts", "fold_ensemble_three_seeds"
        ),
    ),
    Check(
        claim="round84-98: four ratios, flux vector / fold_ensemble_three_seeds",
        stated=0.921,
        recompute=simplification(
            "ladder", "four ratios, flux vector", "fold_ensemble_three_seeds"
        ),
    ),
    Check(
        claim="round84-98: four ratios, rule counts / fold_ensemble_three_seeds",
        stated=0.909,
        recompute=simplification(
            "ladder", "four ratios, rule counts", "fold_ensemble_three_seeds"
        ),
    ),
    Check(
        claim="round84-98: sole-reaction count alone, against / mean_a",
        stated=0.914,
        recompute=simplification(
            "subsets", "sole-reaction count alone, against all six", "mean_a"
        ),
    ),
    Check(
        claim="round84-98: five without it, against all six / mean_a",
        stated=0.918,
        recompute=simplification(
            "subsets", "five without it, against all six", "mean_a"
        ),
    ),
    Check(
        claim="round84-98: five without it, against all six / mean_difference",
        stated=-0.001,
        recompute=simplification(
            "subsets", "five without it, against all six", "mean_difference"
        ),
    ),
    Check(
        claim="round84-98: sole-reaction count alone, against / mean_difference",
        stated=0.006,
        recompute=simplification(
            "subsets", "sole-reaction count alone, against none", "mean_difference"
        ),
    ),
    Check(
        claim="round84-98: flux vector and counts / mean_difference",
        stated=0.0,
        recompute=simplification("budget", "flux vector and counts", "mean_difference"),
    ),
    Check(
        claim="round84-98: counts alone / mean_difference",
        stated=0.0,
        recompute=simplification("budget", "counts alone", "mean_difference"),
    ),
    Check(
        claim="round98: YPD and SC with counts, fold ensemble",
        stated=0.905,
        recompute=simplification(
            "ladder", "YPD and SC ratios, rule counts", "fold_ensemble_one_seed"
        ),
    ),
    # sec:summary and sec:round84-98 -- the module against no metabolism, nine seeds.
    Check(
        claim="summary: single-medium module minus no metabolism",
        stated=0.061,
        recompute=simplification("resplit", "one medium, flux vector and counts, over no metabolism", "mean_difference"),
    ),
    Check(
        claim="summary: ... splits higher",
        stated=10,
        recompute=simplification("resplit", "one medium, flux vector and counts, over no metabolism", "n_positive"),
    ),
    Check(
        claim="summary: ... split-and-seed pairs higher",
        stated=89,
        recompute=simplification("resplit", "one medium, flux vector and counts, over no metabolism", "n_pairs_positive"),
    ),
    Check(
        claim="round84-98: four media, flux vector and counts, over no metabolism, mean_difference",
        stated=0.059,
        recompute=simplification("resplit", "four media, flux vector and counts, over no metabolism", "mean_difference"),
    ),
    Check(
        claim="round84-98: four media, flux vector and counts, over no metabolism, n_pairs_positive",
        stated=88,
        recompute=simplification("resplit", "four media, flux vector and counts, over no metabolism", "n_pairs_positive"),
    ),
    Check(
        claim="round84-98: four media and the flux vector, over no metabolism, mean_difference",
        stated=0.04,
        recompute=simplification("resplit", "four media and the flux vector, over no metabolism", "mean_difference"),
    ),
    Check(
        claim="round84-98: four media and the counts, over no metabolism, mean_difference",
        stated=0.041,
        recompute=simplification("resplit", "four media and the counts, over no metabolism", "mean_difference"),
    ),
    Check(
        claim="round84-98: four ratios alone, over no metabolism, mean_difference",
        stated=0.028,
        recompute=simplification("resplit", "four ratios alone, over no metabolism", "mean_difference"),
    ),
    Check(
        claim="round84-98: two media and the counts, over no metabolism, mean_difference",
        stated=0.036,
        recompute=simplification("resplit", "two media and the counts, over no metabolism", "mean_difference"),
    ),
    Check(
        claim="round84-98: one medium and the counts, over no metabolism, mean_difference",
        stated=0.032,
        recompute=simplification("resplit", "one medium and the counts, over no metabolism", "mean_difference"),
    ),
    Check(
        claim="round106: no metabolism, fold ensemble at three seeds per fold",
        stated=0.878,
        recompute=simplification("ladder", "no metabolism", "fold_ensemble_three_seeds"),
    ),
    Check(
        claim="round101: YPD and SM with counts, cells",
        stated=0.892,
        recompute=simplification("ladder", "YPD and SM ratios, rule counts", "cell_mean"),
    ),
    Check(
        claim="round101: YPD and SM with counts, fold ensemble",
        stated=0.907,
        recompute=simplification(
            "ladder", "YPD and SM ratios, rule counts", "fold_ensemble_one_seed"
        ),
    ),
    Check(
        claim="round101: YPD and SM against YPD and SC, by seed",
        stated=-0.004,
        recompute=simplification(
            "second_medium", "YPD and SM, against YPD and SC", "mean_difference"
        ),
    ),
    # sec:consensus-missed -- the intersection of the predictors on the repeated splits.
    Check(
        claim="consensus: genes held out at least once",
        stated=874,
        recompute=consensus("n_held_out_at_least_once"),
    ),
    Check(
        claim="consensus: labeled genes",
        stated=1121,
        recompute=consensus("n_labeled"),
    ),
    Check(
        claim="consensus: essential genes held out",
        stated=140,
        recompute=consensus("classes", 0, "n_genes"),
    ),
    Check(
        claim="consensus: essential mean misrank, module",
        stated=0.133,
        recompute=consensus("essential_mean_misrank", "transformer, single-experiment module"),
    ),
    Check(
        claim="consensus: essential mean misrank, no metabolism",
        stated=0.193,
        recompute=consensus("essential_mean_misrank", "transformer, no metabolism"),
    ),
    Check(
        claim="consensus: essential mean misrank, graph convolution",
        stated=0.132,
        recompute=consensus("essential_mean_misrank", "graph convolution on GO"),
    ),
    Check(
        claim="consensus: essential mean misrank, DCell",
        stated=0.19,
        recompute=consensus("essential_mean_misrank", "DCell hierarchy, shared input"),
    ),
    Check(
        claim="consensus: essential mean misrank, FBA",
        stated=0.29,
        recompute=consensus("essential_mean_misrank", "FBA growth ratio, YPD"),
    ),
    Check(
        claim="consensus: fewest essential genes a learned predictor misses",
        stated=7,
        recompute=consensus("essential_n_missed", "graph convolution on GO"),
    ),
    Check(
        claim="consensus: most essential genes a learned predictor misses",
        stated=15,
        recompute=consensus("essential_n_missed", "DCell hierarchy, shared input"),
    ),
    Check(
        claim="consensus: missed by none",
        stated=111,
        recompute=consensus("essential_missed_by_k_learned", 0),
    ),
    Check(
        claim="consensus: missed by one",
        stated=16,
        recompute=consensus("essential_missed_by_k_learned", 1),
    ),
    Check(
        claim="consensus: missed by two",
        stated=9,
        recompute=consensus("essential_missed_by_k_learned", 2),
    ),
    Check(
        claim="consensus: missed by three",
        stated=4,
        recompute=consensus("essential_missed_by_k_learned", 3),
    ),
    Check(
        claim="consensus: missed by all four",
        stated=0,
        recompute=consensus("essential_missed_by_k_learned", 4),
    ),
    Check(
        claim="consensus: mean consensus misrank",
        stated=0.055,
        recompute=consensus("classes", 0, "mean_consensus"),
    ),
    Check(
        claim="consensus: agreement, the two transformers",
        stated=0.78,
        recompute=consensus("agreement_essential", "transformer, single-experiment module", "transformer, no metabolism"),
        tolerance=0.006,
    ),
    Check(
        claim="consensus: agreement, module and graph convolution",
        stated=0.4,
        recompute=consensus("agreement_essential", "transformer, single-experiment module", "graph convolution on GO"),
        tolerance=0.006,
    ),
    Check(
        claim="consensus: agreement, graph convolution and DCell",
        stated=0.43,
        recompute=consensus("agreement_essential", "graph convolution on GO", "DCell hierarchy, shared input"),
        tolerance=0.006,
    ),
    Check(
        claim="consensus: agreement, graph convolution and FBA",
        stated=0.08,
        recompute=consensus("agreement_essential", "graph convolution on GO", "FBA growth ratio, YPD"),
        tolerance=0.006,
    ),
    Check(
        claim="consensus: rho with GO term count",
        stated=-0.002,
        recompute=consensus("classes", 0, "spearman_consensus_vs_go_terms"),
    ),
    Check(
        claim="consensus: rho with degree",
        stated=-0.022,
        recompute=consensus("classes", 0, "spearman_consensus_vs_degree"),
    ),
    Check(
        claim="consensus: terms tested",
        stated=179,
        recompute=consensus("classes", 0, "n_terms_tested"),
    ),
    Check(
        claim="consensus: terms under q 0.1",
        stated=11,
        recompute=consensus("classes", 0, "n_terms_q_below_0_1"),
    ),
    Check(
        claim="consensus: FBA knockout growth ratio 1.0, n_annotated",
        stated=70,
        recompute=consensus_term("FBA knockout growth ratio 1.0", "n_annotated"),
    ),
    Check(
        claim="consensus: FBA knockout growth ratio 1.0, mean_misrank_annotated",
        stated=0.082,
        recompute=consensus_term("FBA knockout growth ratio 1.0", "mean_misrank_annotated"),
    ),
    Check(
        claim="consensus: FBA knockout growth ratio 1.0, mean_misrank_rest",
        stated=0.028,
        recompute=consensus_term("FBA knockout growth ratio 1.0", "mean_misrank_rest"),
    ),
    Check(
        claim="consensus: FBA knockout growth ratio 1.0, effect",
        stated=0.78,
        recompute=consensus_term("FBA knockout growth ratio 1.0", "effect"),
        tolerance=0.006,
    ),
    Check(
        claim="consensus: organic substance transport, n_annotated",
        stated=10,
        recompute=consensus_term("organic substance transport", "n_annotated"),
    ),
    Check(
        claim="consensus: organic substance transport, mean_misrank_annotated",
        stated=0.19,
        recompute=consensus_term("organic substance transport", "mean_misrank_annotated"),
    ),
    Check(
        claim="consensus: organic substance transport, mean_misrank_rest",
        stated=0.045,
        recompute=consensus_term("organic substance transport", "mean_misrank_rest"),
    ),
    Check(
        claim="consensus: organic substance transport, effect",
        stated=0.83,
        recompute=consensus_term("organic substance transport", "effect"),
        tolerance=0.006,
    ),
    Check(
        claim="consensus: cellular component organization, n_annotated",
        stated=14,
        recompute=consensus_term("cellular component organization", "n_annotated"),
    ),
    Check(
        claim="consensus: cellular component organization, mean_misrank_annotated",
        stated=0.12,
        recompute=consensus_term("cellular component organization", "mean_misrank_annotated"),
    ),
    Check(
        claim="consensus: cellular component organization, mean_misrank_rest",
        stated=0.048,
        recompute=consensus_term("cellular component organization", "mean_misrank_rest"),
    ),
    Check(
        claim="consensus: monoatomic ion transport, n_annotated",
        stated=5,
        recompute=consensus_term("monoatomic ion transport", "n_annotated"),
    ),
    Check(
        claim="consensus: monoatomic ion transport, mean_misrank_annotated",
        stated=0.184,
        recompute=consensus_term("monoatomic ion transport", "mean_misrank_annotated"),
    ),
    Check(
        claim="consensus: monoatomic ion transport, mean_misrank_rest",
        stated=0.05,
        recompute=consensus_term("monoatomic ion transport", "mean_misrank_rest"),
    ),
    # sec:round108 -- DCell's hierarchy over these splits.
    Check(
        claim="round108: shared input, released split",
        stated=0.825,
        recompute=dcell("released", "dcell_style_shared", "mean"),
    ),
    Check(
        claim="round108: shared input, fold ensemble",
        stated=0.848,
        recompute=dcell("folds", "dcell_style_shared", "ensemble_auroc"),
    ),
    Check(
        claim="round108: published per-gene input, released split",
        stated=0.578,
        recompute=dcell("released", "dcell_published_input", "mean"),
    ),
    Check(
        claim="round108: published per-gene input, fold ensemble",
        stated=0.651,
        recompute=dcell("folds", "dcell_published_input", "ensemble_auroc"),
    ),
    Check(
        claim="round108: shared input minus the graph, ten splits",
        stated=-0.067,
        recompute=dcell("resplits", "dcell_style_shared", "difference_mean"),
    ),
    Check(
        claim="round108: ... splits the hierarchy wins",
        stated=0,
        recompute=dcell("resplits", "dcell_style_shared", "n_arm_above"),
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
