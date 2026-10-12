# experiments/028-gene-essentiality/scripts/essentiality_gate_numbers.py
# [[experiments.028-gene-essentiality.scripts.essentiality_gate_numbers]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/essentiality_gate_numbers
"""Every number of notes-tex/essentiality-figure-gate, from the 028 result files.

The gate document for the manuscript's gene-essentiality figure prints nothing typed by
hand: each value in its prose and captions is a macro written here and each table row is
written here, from the same committed result files ``summary_figure.py`` draws:

    results/fcl_comparison.json           Flux Cone Learning and FBA on the held-out genes
    results/family_ensembles_folds.json   every fold ensemble and post hoc average, 195 genes
    results/resplit_readout.json          the two families and their average on ten other splits
    results/simplification_readout.json   the module forms on one backbone, seeds and splits
    results/genome_heldout_readout.json   the genome-wide 4,674 genes with no GEM reaction
    results/dcell_style_folds.json        the DCell-style comparator under the fold protocol

Writes:
    notes-tex/essentiality-figure-gate/tables/gate_numbers.tex        \\newcommand macros
    notes-tex/essentiality-figure-gate/tables/gate_family_rows.tex    every fold ensemble
    notes-tex/essentiality-figure-gate/tables/gate_genome_rows.tex    every genome-wide arm
    notes-tex/essentiality-figure-gate/tables/gate_figure_names.tex   newest renders by stem

    PYTHONPATH=$PWD python experiments/028-gene-essentiality/scripts/essentiality_gate_numbers.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
from typing import Any

from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
REPO = osp.dirname(EXPERIMENT_ROOT)
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
TABLES = osp.join(REPO, "notes-tex", "essentiality-figure-gate", "tables")
IMAGES = osp.join(ASSET_IMAGES_DIR, "028-gene-essentiality")

DIGIT_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

#: family_ensembles_folds.json names of the single families the prose names.
FAMILY = {
    "goGcn": "GO convolution, five folds (400 epochs)",
    "layerFree": "layer-free log1p vector + affine, pathway + ProtT5 (round 34)",
    "ruleCounts": "+ GPR redundancy counts, pathway + ProtT5, five folds (round 80)",
    "oneMedium": "YPD ratio + vector + counts, pathway + ProtT5, five folds (rounds 84 to 98)",
    "indicator": "indicator + ProtT5 + affine, five folds (round 12)",
}
#: simplification_readout.json ladder keys the prose names.
LADDER = {"none": "none", "oneSolveCounts": "one_solve_counts", "fourSolveCounts": "four_solve_counts", "oneOnly": "one_only", "fourOnly": "four_only"}
#: genome_heldout_readout.json labels the prose names.
GENOME = {
    "layerFree": "layer-free log1p vector + affine, pathway + ProtT5 (round 33 form)",
    "gcnProt": "graph convolution, union of nine, ProtT5",
    "gcnDegree": "graph convolution, physical graph, degree and clustering",
    "fba": "FBA knockout ratio on YPD (0 of 4674 genes carry a reaction)",
}


def macro_name(key: str) -> str:
    return "".join(DIGIT_WORDS[int(ch)] if ch.isdigit() else ch for ch in key if ch.isalnum())


def fmt(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if abs(v) >= 1 and float(v).is_integer():
            return f"{int(v):,}"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def signed(v: float) -> str:
    return f"{v:+.3f}"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("+", "+")


def read(name: str) -> dict[str, Any]:
    with open(osp.join(RESULTS, name)) as f:
        return json.load(f)


def one(rows: list[dict[str, Any]], **where: Any) -> dict[str, Any]:
    hits = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    assert len(hits) == 1, (where, len(hits))
    return hits[0]


def family_row(combos: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return one([c for c in combos if len(c["families"]) == 1], families=[name])


def macros() -> dict[str, str]:
    m: dict[str, str] = {}
    fcl = read("fcl_comparison.json")
    m["nHeldOut"] = fmt(int(fcl["gene_sets"]["ours_scored"]))
    m["nReleasedHeldOut"] = fmt(int(fcl["gene_sets"]["released_held_out_223"]))
    m["nLabeled"] = fmt(int(fcl["gene_sets"]["all_labeled_1121"]))
    h2h = [r for r in fcl["head_to_head"] if r["baseline"] == "FCL" and r["gene_set"] == "ours_scored"]
    assert h2h, "no FCL head-to-head on the scored genes"
    m["fclAuroc"] = fmt(float(h2h[0]["baseline_auroc"]))
    fba = [r for r in fcl["scored"] if r["method"] == "FBA" and r["n_genes"] == fcl["gene_sets"]["ours_scored"]]
    assert fba, "no FBA row on the scored genes"
    m["fbaAuroc"] = fmt(float(fba[0]["auroc"]))
    m["fbaAurocLo"] = fmt(float(fba[0]["ci_low"]))
    m["fbaAurocHi"] = fmt(float(fba[0]["ci_high"]))
    fcl_all = one(fcl["scored"], method="FCL", gene_set="all_labeled_1121")
    m["fclReleasedAuroc"] = fmt(float(fcl_all["auroc"]))
    m["fclReleasedN"] = fmt(int(fcl_all["n_genes"]))
    rep = one(fcl["reproduction_check"], name="fcl_cv_5fold_mean")
    m["fclPublishedCv"] = fmt(float(rep["published"]))
    m["fclRecomputedCv"] = fmt(float(rep["recomputed"]))

    fam = read("family_ensembles_folds.json")
    m["nEssentialHeldOut"] = fmt(int(fam["n_essential"]))
    combos = fam["combinations"]
    singles = [c for c in combos if len(c["families"]) == 1]
    pairs = [c for c in combos if len(c["families"]) == 2]
    best_pair = max(pairs, key=lambda c: c["auroc"])
    best_single = max(singles, key=lambda c: c["auroc"])
    m["bestAverage"] = fmt(float(best_pair["auroc"]))
    m["bestAverageLo"] = fmt(float(best_pair["ci_low"]))
    m["bestAverageHi"] = fmt(float(best_pair["ci_high"]))
    m["bestAverageNames"] = tex_escape(" with ".join(best_pair["families"]))
    m["bestSingle"] = fmt(float(best_single["auroc"]))
    m["bestSingleLo"] = fmt(float(best_single["ci_low"]))
    m["bestSingleHi"] = fmt(float(best_single["ci_high"]))
    m["bestSingleName"] = tex_escape(best_single["families"][0])
    m["nCombinations"] = fmt(len(combos))
    m["nPairAverages"] = fmt(len(pairs))
    for key, name in FAMILY.items():
        hits = [c for c in singles if c["families"][0].startswith(name.split(" (")[0])]
        assert hits, f"family {name!r} not among the single-family ensembles"
        c = max(hits, key=lambda c: c["auroc"])
        m[f"fam{key}"] = fmt(float(c["auroc"]))
        m[f"fam{key}Lo"] = fmt(float(c["ci_low"]))
        m[f"fam{key}Hi"] = fmt(float(c["ci_high"]))

    rs = read("resplit_readout.json")
    m["nResplits"] = fmt(len(rs["splits"]))
    m["resplitTestN"] = fmt(int(rs["splits"][0]["n_test"]))
    for k, v in rs["mean_auroc"].items():
        m["resplitMean" + macro_name(k)] = fmt(float(v))
    for d in rs["differences"]:
        key = "resplitDiff" + macro_name(d["name"].replace("minus", "Minus").replace(" ", ""))
        m[key] = signed(float(d["mean"]))
        m[key + "Up"] = fmt(int(d["n_positive"]))
        m[key + "Of"] = fmt(int(d["n_splits"]))

    sim = read("simplification_readout.json")
    ladder = {r["key"]: r for r in sim["ladder"]}
    for key, lk in LADDER.items():
        r = ladder[lk]
        m[f"lad{key}"] = fmt(float(r["cell_mean"]))
        m[f"lad{key}Sd"] = fmt(float(r["cell_sd"]))
        m[f"lad{key}Seeds"] = fmt(len(r["seeds"]))
        if r["fold_ensemble_one_seed"] is not None:
            m[f"lad{key}Ens"] = fmt(float(r["fold_ensemble_one_seed"]))
    gain = one(sim["resplit"], a="one_solve_counts", b="none")
    m["gainOneMediumOverNone"] = signed(float(gain["mean_difference"]))
    m["gainOneMediumOverNoneUp"] = fmt(int(gain["n_positive"]))
    m["gainOneMediumOverNoneOf"] = fmt(int(gain["n"]))
    m["gainOneMediumOverNonePairsUp"] = fmt(int(gain["n_pairs_positive"]))
    m["gainOneMediumOverNonePairs"] = fmt(int(gain["n_pairs"]))
    four = one(sim["resplit"], a="one_solve_counts", b="four_solve_counts")
    m["oneMinusFour"] = signed(float(four["mean_difference"]))
    m["oneMinusFourUp"] = fmt(int(four["n_positive"]))
    m["oneMinusFourOf"] = fmt(int(four["n"]))

    gh = read("genome_heldout_readout.json")
    m["genomeN"] = fmt(int(gh["n_test_genes"]))
    m["genomeEssential"] = fmt(int(gh["n_test_essential"]))
    arms = {a["label"]: a for a in gh["arms"]}
    for key, label in GENOME.items():
        a = arms[label]
        m[f"gen{key}"] = fmt(float(a["pooled_auroc"]))
        m[f"gen{key}Lo"] = fmt(float(a["pooled_ci"][0]))
        m[f"gen{key}Hi"] = fmt(float(a["pooled_ci"][1]))
    best_genome = max(gh["arms"], key=lambda a: a["pooled_auroc"])
    m["genBest"] = fmt(float(best_genome["pooled_auroc"]))
    m["genBestName"] = tex_escape(best_genome["label"])

    dc = read("dcell_style_folds.json")
    shared = one(dc["fold_ensembles"], arm="dcell_style_shared")
    m["dcellEnsemble"] = fmt(float(shared["ensemble_auroc"]))
    m["dcellEnsembleLo"] = fmt(float(shared["ensemble_ci"][0]))
    m["dcellEnsembleHi"] = fmt(float(shared["ensemble_ci"][1]))
    m["dcellFoldMean"] = fmt(float(shared["fold_mean"]))
    m["dcellTerms"] = fmt(int(dc["hierarchy"]["n_kept_terms"]))
    return m


def write_macros(m: dict[str, str], path: str) -> None:
    lines = [
        "%% SOURCE: written by experiments/028-gene-essentiality/scripts/essentiality_gate_numbers.py",
        "%% from the result files named in its docstring; regenerate, do not edit.",
    ]
    for k, v in sorted(m.items()):
        lines.append(f"\\newcommand{{\\g{macro_name(k)}}}{{{v}}}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(m)} macros)")


def write_family_rows(path: str) -> None:
    fam = read("family_ensembles_folds.json")
    rows = sorted(fam["combinations"], key=lambda c: -c["auroc"])
    lines = [
        "%% SOURCE: experiments/028-gene-essentiality/results/family_ensembles_folds.json,",
        "%% rows written by essentiality_gate_numbers.py; regenerate, do not edit.",
    ]
    for c in rows:
        kind = "average" if len(c["families"]) > 1 else "single family"
        lines.append(
            f"{tex_escape(' with '.join(c['families']))} & {kind} & {int(c['n_cells'])} & "
            f"{float(c['auroc']):.3f} & [{float(c['ci_low']):.3f}, {float(c['ci_high']):.3f}] \\\\"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def write_genome_rows(path: str) -> None:
    gh = read("genome_heldout_readout.json")
    rows = sorted(gh["arms"], key=lambda a: -a["pooled_auroc"])
    lines = [
        "%% SOURCE: experiments/028-gene-essentiality/results/genome_heldout_readout.json,",
        "%% rows written by essentiality_gate_numbers.py; regenerate, do not edit.",
    ]
    for a in rows:
        lines.append(
            f"{tex_escape(a['label'])} & {a['kind']} & {int(a['n_cells'])} & "
            f"{float(a['pooled_auroc']):.3f} & [{float(a['pooled_ci'][0]):.3f}, {float(a['pooled_ci'][1]):.3f}] \\\\"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def write_figure_names(path: str) -> None:
    stems = {"figGoalOne": "summary_goal1_", "figGoalTwo": "summary_goal2_"}
    lines = [
        "%% SOURCE: written by essentiality_gate_numbers.py; the newest SVG of each stem under",
        "%% notes/assets/images/028-gene-essentiality/, converted by `make plots`.",
    ]
    for macro, stem in stems.items():
        files = sorted(glob.glob(osp.join(IMAGES, stem + "*.svg")))
        assert files, f"no render of {stem}* under {IMAGES}"
        name = osp.splitext(osp.basename(files[-1]))[0]
        lines.append(f"\\newcommand{{\\{macro}}}{{figures/{name}.pdf}}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main() -> None:
    os.makedirs(TABLES, exist_ok=True)
    write_macros(macros(), osp.join(TABLES, "gate_numbers.tex"))
    write_family_rows(osp.join(TABLES, "gate_family_rows.tex"))
    write_genome_rows(osp.join(TABLES, "gate_genome_rows.tex"))
    write_figure_names(osp.join(TABLES, "gate_figure_names.tex"))
    print("finished")


if __name__ == "__main__":
    main()
