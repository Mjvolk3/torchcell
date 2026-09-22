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

import csv
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


def graph_module(name: str, key: str) -> Callable[[], float]:
    """Job 2684: one paired difference of the graph family with the solve columns."""

    def run() -> float:
        rows = json.load(open(osp.join(RESULTS, "resplit_readout.json")))["graph_module"]
        return float(next(r for r in rows if r["name"].startswith(name))[key])

    return run


def portable_average(block: str, name: str, key: str) -> Callable[[], float]:
    """The family average with the single-experiment form as the transformer family."""

    def run() -> float:
        report = json.load(open(osp.join(RESULTS, "resplit_readout_portable.json")))
        if block == "mean_auroc":
            return float(report["mean_auroc"][name])
        return float(next(d for d in report["differences"] if d["name"] == name)[key])

    return run


def background(medium: str, key: str) -> Callable[[], float]:
    """One value of the BY4741 background probe for one medium."""

    def run() -> float:
        report = json.load(open(osp.join(RESULTS, "fba_background_probe.json")))
        return float(next(m for m in report["media"] if m["medium"] == medium)[key])

    return run


def fold_average(families: tuple[str, ...], key: str) -> Callable[[], float]:
    """One row of the fold-protocol family averages, addressed by its family names."""

    def run() -> float:
        rows = json.load(open(osp.join(RESULTS, "family_ensembles_folds.json")))
        row = next(c for c in rows["combinations"] if tuple(c["families"]) == families)
        return float(row[key])

    return run


def go006(model: str, inputs: str, block: str, key: str) -> Callable[[], float]:
    """One value of the trigenic-ontology comparison for one model."""

    def run() -> float:
        models = json.load(open(osp.join(RESULTS, "go006_comparison.json")))["models"]
        row = next(m for m in models if m["model"] == model and m["inputs"] == inputs)
        return float(row[block][key])

    return run


def go006_store(key: str) -> Callable[[], float]:
    def run() -> float:
        return float(json.load(open(osp.join(RESULTS, "go006_store.json")))[key])

    return run


def dcell_affine(statistic: str) -> Callable[[], float]:
    """The hierarchy with the affine (job 2698) minus the plain hierarchy, ten splits."""

    def run() -> float:
        a = json.load(open(osp.join(RESULTS, "dcell_style_resplits_affine.json")))
        b = json.load(open(osp.join(RESULTS, "dcell_style_resplits.json")))
        ra = {r["split_file"]: r["mean"] for r in a["rows"] if r["model"] == "dcell_style_shared_affine"}
        rb = {r["split_file"]: r["mean"] for r in b["rows"] if r["model"] == "dcell_style_shared"}
        d = np.array([ra[k] - rb[k] for k in sorted(rb)])
        return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "n_positive": float((d > 0).sum())}[statistic]

    return run


def go006_tuned(statistic: str) -> Callable[[], float]:
    """Job 2701: the tuned hierarchy on the trigenic store minus the current store."""

    def run() -> float:
        t = json.load(open(osp.join(RESULTS, "dcell_style_resplits_go006tuned.json")))
        c = json.load(open(osp.join(RESULTS, "dcell_style_resplits.json")))
        g = lambda d: {r["split_file"]: r["mean"] for r in d["rows"] if r["model"] == "dcell_style_shared"}  # noqa: E731
        rt, rc = g(t), g(c)
        d = np.array([rt[k] - rc[k] for k in sorted(rc)])
        if statistic == "vs_gcn":
            return float(t["paired"][0]["difference_mean"])
        return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "n_positive": float((d > 0).sum())}[statistic]

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
        recompute=seed_mean("w102_ypdgpr_h12l1p1_e200_s270[0-5].json"),
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
    # sec:family-ensembles -- job 2691, the module as an affine on the graph family's logit.
    Check(
        claim="job 2691: GCN, GO | affine: YPD ratio (400, mean",
        stated=0.011,
        recompute=graph_module("GCN, GO | affine: YPD ratio (400", "mean"),
    ),
    Check(
        claim="job 2691: GCN, GO | affine: YPD ratio (400, splits higher",
        stated=10,
        recompute=graph_module("GCN, GO | affine: YPD ratio (400", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="job 2691: GCN, GO | affine: YPD ratio+vector+counts, mean",
        stated=0.018,
        recompute=graph_module("GCN, GO | affine: YPD ratio+vector+counts", "mean"),
    ),
    Check(
        claim="job 2691: GCN, GO | affine: YPD ratio+vector+counts, splits higher",
        stated=9,
        recompute=graph_module("GCN, GO | affine: YPD ratio+vector+counts", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="job 2691: GCN, GO | affine: four ratios+vector+counts, mean",
        stated=0.021,
        recompute=graph_module("GCN, GO | affine: four ratios+vector+counts", "mean"),
    ),
    Check(
        claim="job 2691: GCN, GO | affine: four ratios+vector+counts, splits higher",
        stated=9,
        recompute=graph_module("GCN, GO | affine: four ratios+vector+counts", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="job 2691: both families with the module, average",
        stated=0.897,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "resplit_readout_portable_affine.json")))[
                "mean_auroc"
            ]["average"]
        ),
    ),
    Check(
        claim="job 2691: graph family with the module, alone",
        stated=0.887,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "resplit_readout_portable_affine.json")))[
                "mean_auroc"
            ]["go_convolution"]
        ),
    ),
    Check(
        claim="job 2691: released split, GO | affine: YPD ratio (400 epochs)",
        stated=0.914,
        recompute=lambda: float(
            next(
                r
                for r in json.load(
                    open(osp.join(RESULTS, "graph_baselines_go_affine.json"))
                )["rows"]
                if r["inputs"] == "GO | affine: YPD ratio (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2691: released split, GO | affine: YPD ratio+vector+counts (40",
        stated=0.911,
        recompute=lambda: float(
            next(
                r
                for r in json.load(
                    open(osp.join(RESULTS, "graph_baselines_go_affine.json"))
                )["rows"]
                if r["inputs"] == "GO | affine: YPD ratio+vector+counts (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2691: released split, GO | affine: four ratios+vector+counts (",
        stated=0.91,
        recompute=lambda: float(
            next(
                r
                for r in json.load(
                    open(osp.join(RESULTS, "graph_baselines_go_affine.json"))
                )["rows"]
                if r["inputs"] == "GO | affine: four ratios+vector+counts (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2691: released split, GO (400 epochs)",
        stated=0.908,
        recompute=lambda: float(
            next(
                r
                for r in json.load(
                    open(osp.join(RESULTS, "graph_baselines_go_affine.json"))
                )["rows"]
                if r["inputs"] == "GO (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2693: GCN, GO | affine: YPD ratio (400 epochs), mean",
        stated=0.009,
        recompute=graph_module("GCN, GO | affine: YPD ratio (400 epochs), minus graph convolution on GO, six", "mean"),
    ),
    Check(
        claim="job 2693: GCN, GO | affine: YPD ratio (400 epochs), splits higher",
        stated=10,
        recompute=graph_module("GCN, GO | affine: YPD ratio (400 epochs), minus graph convolution on GO, six", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="job 2693: GCN, GO | affine: YPD ratio+vector+count, mean",
        stated=0.017,
        recompute=graph_module("GCN, GO | affine: YPD ratio+vector+counts (400 epochs), minus graph convolution on GO, six", "mean"),
    ),
    Check(
        claim="job 2693: GCN, GO | affine: YPD ratio+vector+count, splits higher",
        stated=7,
        recompute=graph_module("GCN, GO | affine: YPD ratio+vector+counts (400 epochs), minus graph convolution on GO, six", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="job 2693: GCN, GO | affine: four ratios+vector+cou, mean",
        stated=0.019,
        recompute=graph_module("GCN, GO | affine: four ratios+vector+counts (400 epochs), minus graph convolution on GO, six", "mean"),
    ),
    Check(
        claim="job 2693: GCN, GO | affine: four ratios+vector+cou, splits higher",
        stated=7,
        recompute=graph_module("GCN, GO | affine: four ratios+vector+counts (400 epochs), minus graph convolution on GO, six", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="job 2696: genome-wide, GO (400 epochs)",
        stated=0.702,
        recompute=lambda: float(
            next(
                r
                for r in json.load(
                    open(
                        osp.join(
                            RESULTS, "graph_baselines_go_affine_genome_heldout.json"
                        )
                    )
                )["rows"]
                if r["inputs"] == "GO (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2696: genome-wide, GO | affine: YPD ratio+vector+coun",
        stated=0.767,
        recompute=lambda: float(
            next(
                r
                for r in json.load(
                    open(
                        osp.join(
                            RESULTS, "graph_baselines_go_affine_genome_heldout.json"
                        )
                    )
                )["rows"]
                if r["inputs"] == "GO | affine: YPD ratio+vector+counts (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2696: genome-wide, GO | affine: YPD ratio (400 epochs",
        stated=0.704,
        recompute=lambda: float(
            next(
                r
                for r in json.load(
                    open(
                        osp.join(
                            RESULTS, "graph_baselines_go_affine_genome_heldout.json"
                        )
                    )
                )["rows"]
                if r["inputs"] == "GO | affine: YPD ratio (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2697: released, ProtT5",
        stated=0.859,
        recompute=lambda: float(
            next(
                r
                for r in json.load(open(osp.join(RESULTS, "graph_baselines_prot_affine_s6.json")))["rows"]
                if r["inputs"] == "ProtT5 (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2697: released, with the affine",
        stated=0.891,
        recompute=lambda: float(
            next(
                r
                for r in json.load(open(osp.join(RESULTS, "graph_baselines_prot_affine_s6.json")))["rows"]
                if r["inputs"] == "ProtT5 | affine: YPD ratio+vector+counts (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2697: genome-wide, ProtT5",
        stated=0.751,
        recompute=lambda: float(
            next(
                r
                for r in json.load(open(osp.join(RESULTS, "graph_baselines_prot_affine_s6_genome_heldout.json")))["rows"]
                if r["inputs"] == "ProtT5 (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2697: genome-wide, with the affine",
        stated=0.766,
        recompute=lambda: float(
            next(
                r
                for r in json.load(open(osp.join(RESULTS, "graph_baselines_prot_affine_s6_genome_heldout.json")))["rows"]
                if r["inputs"] == "ProtT5 | affine: YPD ratio+vector+counts (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2702: genome-wide six seeds, GO",
        stated=0.698,
        recompute=lambda: float(
            next(
                r
                for r in json.load(
                    open(
                        osp.join(
                            RESULTS, "graph_baselines_go_affine_s6_genome_heldout.json"
                        )
                    )
                )["rows"]
                if r["inputs"] == "GO (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2702: genome-wide six seeds, with the affine",
        stated=0.768,
        recompute=lambda: float(
            next(
                r
                for r in json.load(
                    open(
                        osp.join(
                            RESULTS, "graph_baselines_go_affine_s6_genome_heldout.json"
                        )
                    )
                )["rows"]
                if r["inputs"] == "GO | affine: YPD ratio+vector+counts (400 epochs)"
            )["mean"]
        ),
    ),
    Check(
        claim="job 2702: genome-wide six seeds, ratio alone",
        stated=0.696,
        recompute=lambda: float(
            next(
                r
                for r in json.load(
                    open(
                        osp.join(
                            RESULTS, "graph_baselines_go_affine_s6_genome_heldout.json"
                        )
                    )
                )["rows"]
                if r["inputs"] == "GO | affine: YPD ratio (400 epochs)"
            )["mean"]
        ),
    ),
    # sec:family-ensembles -- job 2684, the solve columns on the graph family's nodes.
    Check(
        claim="job 2684: GCN, GO+solve log1p, mean",
        stated=-0.009,
        recompute=graph_module("GCN, GO+solve log1p", "mean"),
    ),
    Check(
        claim="job 2684: GCN, GO+solve log1p, splits higher",
        stated=1,
        recompute=graph_module("GCN, GO+solve log1p", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="job 2684: GCN, GO+FBA ratios+solve, mean",
        stated=-0.01,
        recompute=graph_module("GCN, GO+FBA ratios+solve", "mean"),
    ),
    Check(
        claim="job 2684: GCN, GO+FBA ratios+solve, splits higher",
        stated=1,
        recompute=graph_module("GCN, GO+FBA ratios+solve", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="job 2684: MLP, GO+solve, mean",
        stated=-0.013,
        recompute=graph_module("MLP, GO+solve", "mean"),
    ),
    Check(
        claim="job 2684: MLP, GO+solve, splits higher",
        stated=4,
        recompute=graph_module("MLP, GO+solve", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="job 2684: GCN, solve log1p, mean",
        stated=-0.189,
        recompute=graph_module("GCN, solve log1p", "mean"),
    ),
    Check(
        claim="job 2684: GCN, solve log1p, splits higher",
        stated=0,
        recompute=graph_module("GCN, solve log1p", "n_positive"),
        tolerance=0.5,
    ),
    # sec:round84-98 -- round 107, the two full recipes at nine seeds on the released split.
    Check(
        claim="round107: one medium, nine seeds",
        stated=0.912,
        recompute=simplification("released_pairs", "one medium for four, flux vector and counts kept, released split", "mean_a"),
    ),
    Check(
        claim="round107: four media, nine seeds",
        stated=0.92,
        recompute=simplification("released_pairs", "one medium for four, flux vector and counts kept, released split", "mean_b"),
    ),
    Check(
        claim="round107: paired difference",
        stated=-0.009,
        recompute=simplification("released_pairs", "one medium for four, flux vector and counts kept, released split", "mean_difference"),
    ),
    Check(
        claim="round107: seeds the one-medium cell wins",
        stated=1,
        recompute=simplification("released_pairs", "one medium for four, flux vector and counts kept, released split", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round107: seeds paired",
        stated=9,
        recompute=simplification("released_pairs", "one medium for four, flux vector and counts kept, released split", "n"),
        tolerance=0.5,
    ),
    Check(
        claim="round107: added one-medium seeds under 0.90",
        stated=2,
        recompute=lambda: float(
            sum(
                json.load(
                    open(osp.join(RESULTS, f"w102_ypdgpr_h12l1p1_e200_s{s}.json"))
                )["test"]["pinned"]["auroc"]
                < 0.90
                for s in (2706, 2707, 2708)
            )
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round84-98: full recipe, cells at nine seeds",
        stated=0.920,
        recompute=simplification(
            "ladder", "four ratios, flux vector, rule counts", "cell_mean"
        ),
    ),
    # sec:round75 -- the average with the single-experiment form.
    Check(
        claim="portable average: average",
        stated=0.907,
        recompute=portable_average("mean_auroc", "average", ""),
    ),
    Check(
        claim="portable average: graph",
        stated=0.871,
        recompute=portable_average("mean_auroc", "go_convolution", ""),
    ),
    Check(
        claim="portable average: transformer",
        stated=0.869,
        recompute=portable_average("mean_auroc", "layer_free", ""),
    ),
    Check(
        claim="portable average: margin",
        stated=0.029,
        recompute=portable_average("differences", "average minus the better single model", "mean"),
    ),
    Check(
        claim="portable average: margin sd",
        stated=0.013,
        recompute=portable_average("differences", "average minus the better single model", "sd"),
    ),
    Check(
        claim="portable average: margin splits",
        stated=10,
        recompute=portable_average("differences", "average minus the better single model", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="portable average: single models",
        stated=-0.003,
        recompute=portable_average("differences", "layer-free minus GO convolution", "mean"),
    ),
    Check(
        claim="portable average: transformer higher",
        stated=4,
        recompute=portable_average("differences", "layer-free minus GO convolution", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="portable average: released folds, alone",
        stated=0.929,
        recompute=fold_average(("YPD ratio + vector + counts, pathway + ProtT5, five folds (rounds 84 to 98)",), "auroc"),
    ),
    Check(
        claim="portable average: released folds, with the graph family",
        stated=0.953,
        recompute=fold_average(("GO convolution, five folds (400 epochs)", "YPD ratio + vector + counts, pathway + ProtT5, five folds (rounds 84 to 98)"), "auroc"),
    ),
    Check(
        claim="portable average: released folds, interval low",
        stated=0.92,
        recompute=fold_average(("GO convolution, five folds (400 epochs)", "YPD ratio + vector + counts, pathway + ProtT5, five folds (rounds 84 to 98)"), "ci_low"),
        tolerance=0.006,
    ),
    Check(
        claim="portable average: released folds, interval high",
        stated=0.98,
        recompute=fold_average(("GO convolution, five folds (400 epochs)", "YPD ratio + vector + counts, pathway + ProtT5, five folds (rounds 84 to 98)"), "ci_high"),
        tolerance=0.006,
    ),
    Check(
        claim="job 2695: graph family with the affine, fold ensemble",
        stated=0.930,
        recompute=fold_average(("GO convolution with the module as an affine, five folds (job 2695)",), "auroc"),
    ),
    Check(
        claim="job 2695: graph family with the affine, averaged with the transformer",
        stated=0.941,
        recompute=fold_average(("GO convolution with the module as an affine, five folds (job 2695)", "YPD ratio + vector + counts, pathway + ProtT5, five folds (rounds 84 to 98)"), "auroc"),
    ),
    Check(
        claim="job 2695: graph family without the affine, fold ensemble (same job)",
        stated=0.917,
        recompute=lambda: float(
            json.load(
                open(osp.join(RESULTS, "graph_baselines_go_affine_folds_s3.json"))
            )["arms"]["GCN400"]["ensemble_auroc"]
        ),
    ),
    # sec:consensus-missed -- the BY4741 background probe.
    Check(
        claim="background probe: moved on YPD",
        stated=63,
        recompute=background("YPD", "n_moved"),
        tolerance=0.5,
    ),
    Check(
        claim="background probe: essential at ratio one, wild type",
        stated=77,
        recompute=background("YPD", "n_essential_ratio_one_wild_type"),
        tolerance=0.5,
    ),
    Check(
        claim="background probe: essential at ratio one, background",
        stated=83,
        recompute=background("YPD", "n_essential_ratio_one_background"),
        tolerance=0.5,
    ),
    Check(
        claim="background probe: YPD AUROC wild type",
        stated=0.717,
        recompute=background("YPD", "auroc_labeled_wild_type"),
    ),
    Check(
        claim="background probe: YPD AUROC background",
        stated=0.713,
        recompute=background("YPD", "auroc_labeled_background"),
    ),
    Check(
        claim="background probe: SC AUROC wild type",
        stated=0.718,
        recompute=background("SC", "auroc_labeled_wild_type"),
    ),
    Check(
        claim="background probe: SC AUROC background",
        stated=0.714,
        recompute=background("SC", "auroc_labeled_background"),
    ),
    Check(
        claim="background probe: YPD growth as defined",
        stated=0.535,
        recompute=background("YPD", "growth_as_defined"),
    ),
    Check(
        claim="background probe: YPD growth as defined, ura3 deleted",
        stated=0.0,
        recompute=background("YPD", "growth_as_defined_ura3"),
    ),
    # sec:round108 -- the trigenic DCell runs' ontology.
    Check(
        claim="go006 store: subsystems after containment",
        stated=2655,
        recompute=go006_store("n_nodes_after_containment"),
        tolerance=0.5,
    ),
    Check(
        claim="go006 store: columns",
        stated=2651,
        recompute=go006_store("n_columns"),
        tolerance=0.5,
    ),
    Check(
        claim="go006 store: shared terms",
        stated=1421,
        recompute=go006_store("n_terms_shared_with_current"),
        tolerance=0.5,
    ),
    Check(
        claim="go006 store: columns per labeled gene, trigenic",
        stated=34.5,
        recompute=go006_store("mean_columns_per_labeled_gene_006"),
        tolerance=0.06,
    ),
    Check(
        claim="go006 store: columns per labeled gene, current",
        stated=46.5,
        recompute=go006_store("mean_columns_per_labeled_gene_current"),
        tolerance=0.06,
    ),
    Check(
        claim="go006: graph, ten splits",
        stated=-0.002,
        recompute=go006("GCN, union of nine", "GO (400 epochs)", "resplits", "difference_mean"),
    ),
    Check(
        claim="go006: graph, splits higher",
        stated=4,
        recompute=go006("GCN, union of nine", "GO (400 epochs)", "resplits", "n_006_higher"),
        tolerance=0.5,
    ),
    Check(
        claim="go006: graph, released trigenic",
        stated=0.905,
        recompute=go006("GCN, union of nine", "GO (400 epochs)", "released", "mean_006"),
    ),
    Check(
        claim="go006: graph, released current",
        stated=0.909,
        recompute=go006("GCN, union of nine", "GO (400 epochs)", "released", "mean_current_shared_seeds"),
    ),
    Check(
        claim="go006: dcell, released trigenic",
        stated=0.714,
        recompute=go006("dcell_style_shared", "GO", "released", "mean_006"),
    ),
    Check(
        claim="go006: dcell, released current, shared seeds",
        stated=0.841,
        recompute=go006("dcell_style_shared", "GO", "released", "mean_current_shared_seeds"),
    ),
    Check(
        claim="go006: dcell, ten splits",
        stated=-0.02,
        recompute=go006("dcell_style_shared", "GO", "resplits", "difference_mean"),
    ),
    Check(
        claim="go006: dcell, splits higher",
        stated=2,
        recompute=go006("dcell_style_shared", "GO", "resplits", "n_006_higher"),
        tolerance=0.5,
    ),
    Check(
        claim="go006: dcell, sd over splits",
        stated=0.041,
        recompute=go006("dcell_style_shared", "GO", "resplits", "difference_sd"),
    ),
    # sec:round109 -- the module on ProtT5 alone, released split.
    Check(
        claim="round110: ProtT5 with the module",
        stated=0.883,
        recompute=simplification("released_pairs", "ProtT5 only: the module over no metabolism, released split", "mean_a"),
    ),
    Check(
        claim="round110: ProtT5 without",
        stated=0.771,
        recompute=simplification("released_pairs", "ProtT5 only: the module over no metabolism, released split", "mean_b"),
    ),
    Check(
        claim="round110: paired gain",
        stated=0.112,
        recompute=simplification("released_pairs", "ProtT5 only: the module over no metabolism, released split", "mean_difference"),
    ),
    Check(
        claim="round110: seeds higher",
        stated=6,
        recompute=simplification("released_pairs", "ProtT5 only: the module over no metabolism, released split", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round110: the same pair on the pathway backbone",
        stated=0.058,
        recompute=simplification(
            "released_pairs",
            "pathway backbone: the module over no metabolism, released split",
            "mean_difference",
        ),
    ),
    Check(
        claim="go006 tune: best validation on the trigenic store",
        stated=0.876,
        recompute=lambda: max(
            t["val_mean"]
            for t in json.load(
                open(osp.join(RESULTS, "dcell_style_tune_go006.json"))
            )["tune"]
        ),
    ),
    Check(
        claim="go006 tune: best validation on the current store",
        stated=0.873,
        recompute=lambda: max(
            t["val_mean"]
            for t in json.load(open(osp.join(RESULTS, "dcell_style_tune.json")))[
                "tune"
            ]
        ),
    ),
    Check(
        claim="job 2700: hierarchy on the trigenic store at its tuned configuration",
        stated=0.709,
        recompute=lambda: float(
            json.load(
                open(osp.join(RESULTS, "dcell_style_released_go006tuned.json"))
            )["rows"][0]["mean"]
        ),
    ),
    Check(
        claim="round111: ProtT5 with the module, fold ensemble",
        stated=0.892,
        recompute=simplification(
            "ladder", "ProtT5 only: YPD ratio, flux vector, rule counts", "fold_ensemble_one_seed"
        ),
    ),
    Check(
        claim="round111: ProtT5 without, fold ensemble",
        stated=0.795,
        recompute=simplification("ladder", "ProtT5 only: no metabolism", "fold_ensemble_one_seed"),
    ),
    # sec:round109 -- the ten splits.
    Check(
        claim="round109: module over none, ten splits",
        stated=0.102,
        recompute=simplification("resplit", "ProtT5 only: one medium, flux vector and counts, over no metabolism", "mean_difference"),
    ),
    Check(
        claim="round109: splits higher",
        stated=10,
        recompute=simplification("resplit", "ProtT5 only: one medium, flux vector and counts, over no metabolism", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round109: pairs higher",
        stated=30,
        recompute=simplification("resplit", "ProtT5 only: one medium, flux vector and counts, over no metabolism", "n_pairs_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round109: against the pathway backbone, both with the module",
        stated=-0.027,
        recompute=simplification("resplit", "ProtT5 only against the pathway backbone, both with the module", "mean_difference"),
    ),
    Check(
        claim="round109: against the pathway backbone, neither",
        stated=-0.063,
        recompute=simplification("resplit", "ProtT5 only against the pathway backbone, neither with the module", "mean_difference"),
    ),
    # sec:round112 -- the corrected medium.
    Check(
        claim="round112: released, corrected",
        stated=0.913,
        recompute=simplification("released_pairs", "corrected YPD store against the Suthers store, released split", "mean_a"),
    ),
    Check(
        claim="round112: released, old",
        stated=0.916,
        recompute=simplification("released_pairs", "corrected YPD store against the Suthers store, released split", "mean_b"),
    ),
    Check(
        claim="round112: released, paired",
        stated=-0.003,
        recompute=simplification("released_pairs", "corrected YPD store against the Suthers store, released split", "mean_difference"),
    ),
    Check(
        claim="round112: released, seeds higher",
        stated=3,
        recompute=simplification("released_pairs", "corrected YPD store against the Suthers store, released split", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round112: ten splits, paired",
        stated=-0.007,
        recompute=simplification("resplit", "corrected YPD store against the Suthers store, one medium, vector and counts", "mean_difference"),
    ),
    Check(
        claim="round112: ten splits, sd",
        stated=0.009,
        recompute=simplification("resplit", "corrected YPD store against the Suthers store, one medium, vector and counts", "sd_difference"),
    ),
    Check(
        claim="round112: ten splits, higher",
        stated=2,
        recompute=simplification("resplit", "corrected YPD store against the Suthers store, one medium, vector and counts", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round112: ten splits, over none",
        stated=0.059,
        recompute=simplification("resplit", "corrected YPD store: one medium, flux vector and counts, over no metabolism", "mean_difference"),
    ),
    Check(
        claim="round112: ten splits, over none, higher",
        stated=10,
        recompute=simplification("resplit", "corrected YPD store: one medium, flux vector and counts, over no metabolism", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round112: fewest genes changing rank",
        stated=182,
        recompute=lambda: float(
            min(
                json.load(open(osp.join(RESULTS, "simplification_readout.json")))[
                    "rank_shifts"
                ][0]["n_rank_changed"]
            )
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round112: most genes changing rank",
        stated=189,
        recompute=lambda: float(
            max(
                json.load(open(osp.join(RESULTS, "simplification_readout.json")))[
                    "rank_shifts"
                ][0]["n_rank_changed"]
            )
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round112: largest shift",
        stated=97,
        recompute=lambda: float(
            max(
                json.load(open(osp.join(RESULTS, "simplification_readout.json")))[
                    "rank_shifts"
                ][0]["max_abs_shift"]
            )
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round112: largest shift among the essential genes",
        stated=46,
        recompute=lambda: float(
            max(
                json.load(open(osp.join(RESULTS, "simplification_readout.json")))[
                    "rank_shifts"
                ][0]["max_abs_shift_essential"]
            )
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round112: labeled genes whose YPD ratio changed",
        stated=78,
        recompute=lambda: float(
            sum(
                abs(float(a["YPD"]) - float(b["YPD"])) > 1e-6
                for a, b in zip(
                    list(csv.DictReader(open(osp.join(RESULTS, "fba_essentiality_per_gene.csv")))),
                    list(csv.DictReader(open(osp.join(RESULTS, "fba_essentiality_per_gene_nuc.csv")))),
                    strict=True,
                )
            )
        ),
        tolerance=0.5,
    ),
    Check(
        claim="job 2698: hierarchy with the affine, ten splits",
        stated=0.036,
        recompute=dcell_affine("mean"),
    ),
    Check(
        claim="job 2698: ... sd",
        stated=0.013,
        recompute=dcell_affine("sd"),
    ),
    Check(
        claim="job 2698: ... splits higher",
        stated=10,
        recompute=dcell_affine("n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="job 2698: hierarchy with the affine, released",
        stated=0.846,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "dcell_style_released_affine.json")))[
                "rows"
            ][0]["mean"]
        ),
    ),
    Check(
        claim="job 2698: ... against the graph",
        stated=-0.032,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "dcell_style_resplits_affine.json")))[
                "paired"
            ][0]["difference_mean"]
        ),
    ),
    Check(
        claim="job 2701: tuned hierarchy on the trigenic store, ten splits",
        stated=-0.012,
        recompute=go006_tuned("mean"),
    ),
    Check(
        claim="job 2701: ... splits higher",
        stated=5,
        recompute=go006_tuned("n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="job 2701: ... sd",
        stated=0.038,
        recompute=go006_tuned("sd"),
    ),
    Check(
        claim="job 2701: ... against the graph",
        stated=-0.079,
        recompute=go006_tuned("vs_gcn"),
    ),
    # sec:round113 -- offline enzyme-constrained stores.
    Check(
        claim="round113: two-sigma, paired",
        stated=0.0,
        recompute=simplification("released_pairs", "kcat two-sigma ceilings against the plain store, released split", "mean_difference"),
    ),
    Check(
        claim="round113: two-sigma, seeds higher",
        stated=3,
        recompute=simplification("released_pairs", "kcat two-sigma ceilings against the plain store, released split", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round113: sampled, paired",
        stated=0.003,
        recompute=simplification("released_pairs", "kcat-sampled store against the plain store, released split", "mean_difference"),
    ),
    Check(
        claim="round113: sampled, seeds higher",
        stated=2,
        recompute=simplification("released_pairs", "kcat-sampled store against the plain store, released split", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round113: sampled, sd",
        stated=0.014,
        recompute=simplification("released_pairs", "kcat-sampled store against the plain store, released split", "sd_difference"),
    ),
    Check(
        claim="round113: moma, paired",
        stated=-0.002,
        recompute=simplification("released_pairs", "MOMA store against the plain store, released split", "mean_difference"),
    ),
    Check(
        claim="round113: moma, seeds higher",
        stated=2,
        recompute=simplification("released_pairs", "MOMA store against the plain store, released split", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round113: point, paired",
        stated=-0.004,
        recompute=simplification("released_pairs", "kcat point ceilings against the plain store, released split", "mean_difference"),
    ),
    Check(
        claim="round113 store point: capacity_scale",
        stated=3000,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_kcat_bounded_store_point.json")))["capacity_scale"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round113 store point: n_ceilings_binding_wild_type",
        stated=0,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_kcat_bounded_store_point.json")))["n_ceilings_binding_wild_type"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round113 store sigma1: capacity_scale",
        stated=1000,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_kcat_bounded_store_sigma1.json")))["capacity_scale"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round113 store sigma2: capacity_scale",
        stated=100,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_kcat_bounded_store_sigma2.json")))["capacity_scale"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round113 store sigma2: n_ceilings_binding_wild_type",
        stated=2,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_kcat_bounded_store_sigma2.json")))["n_ceilings_binding_wild_type"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round113 store sigma2: median_relative_l2_log1p_to_plain",
        stated=0.07,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_kcat_bounded_store_sigma2.json")))["median_relative_l2_log1p_to_plain"]
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round113 store moma: n_lethal",
        stated=105,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_kcat_bounded_store_moma.json")))["n_lethal"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round113 store moma: n_lethal_plain",
        stated=91,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_kcat_bounded_store_moma.json")))["n_lethal_plain"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round113 store point: n_units_with_abundance",
        stated=1400,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_kcat_bounded_store_point.json")))["n_units_with_abundance"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round113 store point: n_reactions_with_ceiling",
        stated=995,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_kcat_bounded_store_point.json")))["n_reactions_with_ceiling"]
        ),
        tolerance=0.5,
    ),
    # sec:round114 -- the objective ladder and the GECKO pool stores.
    Check(
        claim="round114: plain FBA, paired",
        stated=-0.01,
        recompute=simplification("released_pairs", "plain FBA, no parsimony against the plain store, released split", "mean_difference"),
    ),
    Check(
        claim="round114: plain FBA, seeds higher",
        stated=2,
        recompute=simplification("released_pairs", "plain FBA, no parsimony against the plain store, released split", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round114: plain FBA, sd",
        stated=0.021,
        recompute=simplification("released_pairs", "plain FBA, no parsimony against the plain store, released split", "sd_difference"),
    ),
    Check(
        claim="round114: lambda 1e-3",
        stated=-0.004,
        recompute=simplification("released_pairs", "parsimony in the objective, lambda 1e-3 against the plain store, released split", "mean_difference"),
    ),
    Check(
        claim="round114: lambda 1e-4",
        stated=-0.004,
        recompute=simplification("released_pairs", "parsimony in the objective, lambda 1e-4 against the plain store, released split", "mean_difference"),
    ),
    Check(
        claim="round114: pool 0.125",
        stated=-0.011,
        recompute=simplification("released_pairs", "GECKO pool 0.125 against the plain store, released split", "mean_difference"),
    ),
    Check(
        claim="round114: pool 0.125, seeds higher",
        stated=1,
        recompute=simplification("released_pairs", "GECKO pool 0.125 against the plain store, released split", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round114: pool 0.25",
        stated=-0.004,
        recompute=simplification("released_pairs", "GECKO pool 0.25 against the plain store, released split", "mean_difference"),
    ),
    Check(
        claim="round114: pool 0.125 sigma1",
        stated=-0.005,
        recompute=simplification("released_pairs", "GECKO pool 0.125, kcat one sigma up against the plain store, released split", "mean_difference"),
    ),
    Check(
        claim="round114: pool 0.125 min enzyme",
        stated=-0.01,
        recompute=simplification("released_pairs", "GECKO pool 0.125, parsimony on enzyme mass against the plain store, released split", "mean_difference"),
    ),
    Check(
        claim="round114 store pool125: n_variables",
        stated=11972,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool125.json")))["n_variables"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round114 store pool125: n_units_in_program",
        stated=3710,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool125.json")))["n_units_in_program"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round114 store pool125: n_reactions_enzyme_constrained",
        stated=2695,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool125.json")))["n_reactions_enzyme_constrained"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round114 store pool125: wild_type_growth_fraction",
        stated=0.12,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool125.json")))["wild_type_growth_fraction"]
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round114 store pool250: wild_type_growth_fraction",
        stated=0.18,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool250.json")))["wild_type_growth_fraction"]
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round114 store pool125s1: wild_type_growth_fraction",
        stated=0.28,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool125s1.json")))["wild_type_growth_fraction"]
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round114 store pool125: n_genes_pool_binding",
        stated=1030,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool125.json")))["n_genes_pool_binding"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round114 store pool125: n_genes_any_unit_usage_changed_viable",
        stated=302,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool125.json")))["n_genes_any_unit_usage_changed_viable"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round114 store pool125: n_genes_allocation_moved_gt_0.01_of_pool_viable",
        stated=28,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool125.json")))["n_genes_allocation_moved_gt_0.01_of_pool_viable"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round114 store pool250: n_genes_allocation_moved_gt_0.01_of_pool_viable",
        stated=82,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool250.json")))["n_genes_allocation_moved_gt_0.01_of_pool_viable"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round114 store pool125s1: n_genes_allocation_moved_gt_0.01_of_pool_viable",
        stated=89,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool125s1.json")))["n_genes_allocation_moved_gt_0.01_of_pool_viable"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round114: sign conflicts at 10 kJ/mol",
        stated=111,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool125sign.json")))[
                "thermo"
            ]["n_conflicts_with_plain_wild_type_flux"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round114: sign conflicts at 30 kJ/mol",
        stated=81,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_pool125sign30.json")))[
                "thermo"
            ]["n_conflicts_with_plain_wild_type_flux"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round114: plain FBA distance to the store",
        stated=6.1,
        recompute=lambda: float(
            json.load(open(osp.join(RESULTS, "ess_gecko_store_plainfba.json")))[
                "median_relative_l2_log1p_to_plain"
            ]
        ),
        tolerance=0.06,
    ),
    # sec:round115 -- the learned allocation with the pool-fill term.
    Check(
        claim="round115: forced column, paired",
        stated=0.062,
        recompute=simplification("released_pairs", "pool-fill term, forced growth column, against the pool by construction", "mean_difference"),
    ),
    Check(
        claim="round115: forced column, seeds higher",
        stated=3,
        recompute=simplification("released_pairs", "pool-fill term, forced growth column, against the pool by construction", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round115: gated, paired",
        stated=0.01,
        recompute=simplification("released_pairs", "pool-fill term, gated readout, against the gated elastic cell", "mean_difference"),
    ),
    Check(
        claim="round115: gated, seeds higher",
        stated=2,
        recompute=simplification("released_pairs", "pool-fill term, gated readout, against the gated elastic cell", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round115: gated, sd",
        stated=0.046,
        recompute=simplification("released_pairs", "pool-fill term, gated readout, against the gated elastic cell", "sd_difference"),
    ),
    Check(
        claim="round115: peak memory of one cell, GB (sacct MaxRSS of job 2730)",
        stated=40,
        recompute=lambda: 41419424 / 1024**2,
        tolerance=0.6,
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
