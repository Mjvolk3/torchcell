# experiments/044-organism-similarity/scripts/08_notes_tex_numbers.py
# [[experiments.044-organism-similarity.scripts.08_notes_tex_numbers]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/044-organism-similarity/scripts/08_notes_tex_numbers.py

r"""Every number the 044 notes-tex review document prints, generated from result files.

Writes, under ``notes-tex/database/044-organism-similarity/``:

- ``numbers.tex``: one ``\newcommand`` per number the prose and captions cite, so no
  digit in the document is typed by hand.
- ``tables/headline.tex``: the headline-numbers table.
- ``tables/organisms.tex``: the organism table (34 rows).
- ``figures.tex``: ``\newcommand``s holding the current timestamped figure file names
  (read from ``results/figure_stats.json``), so a re-render only needs this script rerun.

Reads only ``results/`` (written by 01-07). Run from the repo root after 07_figures.py:
``PYTHONPATH=. ~/miniconda3/envs/torchcell/bin/python
experiments/044-organism-similarity/scripts/08_notes_tex_numbers.py``.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXP_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "044-organism-similarity")
RES = osp.join(EXP_DIR, "results")
REPO = osp.dirname(os.environ["EXPERIMENT_ROOT"].rstrip("/"))
DOC = osp.join(REPO, "notes-tex", "database", "044-organism-similarity")
SCRIPT = "experiments/044-organism-similarity/scripts/08_notes_tex_numbers.py"
HEADER = f"%% GENERATED FILE -- do not hand-edit.\n%% SOURCE: {SCRIPT}\n"

#: Pair labels in headline_numbers.json -> macro stem.
PAIR_STEM = {
    "human/chimp": "HumChimp",
    "human/macaque": "HumMacaque",
    "human/mouse": "HumMouse",
    "human/chicken": "HumChicken",
    "S288C/S. paradoxus": "ScerSpar",
    "S288C/S. eubayanus": "ScerSeub",
    "SD108/CBS573": "SdCbs",
}
#: Pair categories -> (macro stem, table label).
CAT_STEM = {
    "within_genus": ("Genus", r"Within \org{Saccharomyces}"),
    "within_mammals": ("Mammals", "Within mammals"),
    "mammal_bird": ("MammalBird", "Mammal vs chicken"),
    "within_saccharomycotina": ("Ycotina", "Within Saccharomycotina"),
    "within_bacteria": ("Bacteria", "Within bacteria"),
    "yeast_bacteria": ("YeastBacteria", "Saccharomycotina vs bacteria"),
}
#: Fig 5 pair classes -> macro stem.
CLASS_STEM = {
    "within_genus": "Genus",
    "within_mammals": "Mammals",
    "mammal_bird": "MammalBird",
    "within_saccharomycotina": "Ycotina",
    "within_bacteria": "Bacteria",
    "yeast_bacteria": "YeastBacteria",
    "yeast_animal": "YeastAnimal",
    "animal_bacteria": "AnimalBacteria",
    "basidio_yeast": "BasidioYeast",
    "basidio_animal": "BasidioAnimal",
    "basidio_bacteria": "BasidioBacteria",
}


def tex_escape(s: str) -> str:
    """Escape the LaTeX specials that occur in organism names and accessions."""
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
    ):
        s = s.replace(a, b)
    return s


def org_name(display: str) -> str:
    """Italic binomial (first two tokens), the rest upright."""
    toks = display.split(" ")
    return (
        r"\org{"
        + " ".join(toks[:2])
        + "}"
        + (" " + tex_escape(" ".join(toks[2:])) if len(toks) > 2 else "")
    )


def pct(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    head = json.load(open(osp.join(RES, "headline_numbers.json")))
    stats = json.load(open(osp.join(RES, "figure_stats.json")))
    lit = json.load(open(osp.join(RES, "literature_values.json")))
    org = json.load(open(osp.join(RES, "organisms.json")))["organisms"]
    summ = pd.read_csv(osp.join(RES, "clade_distance_summary.csv"))
    f4, f5, f6 = stats["fig4"], stats["fig5"], stats["fig6"]
    iso = head["isolates_vs_s288c"]
    macros: dict[str, str] = {}

    def m(name: str, value: str) -> None:
        assert name.isalpha(), name
        assert name not in macros, name
        macros[name] = value

    # organisms and pairs
    n_prot = sum(1 for o in org if o["proteome"] is not None)
    m("nOrganisms", f"{len(org)}")
    m("nProteomes", f"{n_prot}")
    m("nPairs", f"{f5['n_pairs']}")
    # isolates vs S288C
    m("nIsolates", f"{iso['n_isolates']:,}")
    m("nOrfs", f"{iso['n_orfs_per_isolate']:,}")
    m("isoMedMedId", pct(100 * iso["median_of_median_identity"], 2))
    m("isoMedMedDiv", pct(100 * (1 - iso["median_of_median_identity"]), 2))
    m("isoMedMeanId", pct(100 * iso["median_of_mean_identity"], 2))
    m("isoMinMeanId", pct(100 * iso["min_mean_identity"], 2))
    m("isoMaxMeanId", pct(100 * iso["max_mean_identity"], 2))
    m("isoEqLenId", pct(100 * iso["median_of_mean_identity_equal_length"], 2))
    # literature
    lo, hi = lit["protein_identity_vs_reference"]
    m("litIdLo", pct(100 * lo, 2))
    m("litIdHi", pct(100 * hi, 2))
    m("litDivMid", pct(100 * (1 - (lo + hi) / 2), 2))
    m("litSitesLo", f"{lit['peptide_altering_sites'][0]:,}")
    m("litSitesHi", f"{lit['peptide_altering_sites'][1]:,}")
    m("litResiduesM", f"{lit['human_per_gene_proteome_residues'] / 1e6:.1f}")
    # pairs
    for label, stem in PAIR_STEM.items():
        p = head["pairs"][label]
        m(f"aai{stem}", pct(p["aai_mean"], 2))
        m(f"aaiShort{stem}", pct(p["aai_mean"], 1))
        m(f"aaiMed{stem}", pct(p["aai_median"], 2))
        m(f"rbh{stem}", f"{p['n_rbh']:,}")
        m(f"of{stem}", pct(p["ortholog_fraction"], 3))
        m(f"sc{stem}", pct(p["aai_single_copy_og"], 2))
    pe = head["pairs"]["S288C/E. coli"]
    m("aaiScerEcoli", pct(pe["aai_mean"], 2))
    m("rbhScerEcoli", f"{pe['n_rbh']:,}")
    m("ofScerEcoli", pct(pe["ortholog_fraction"], 3))
    # categories
    for cat, (stem, _) in CAT_STEM.items():
        c = head["categories"][cat]
        m(f"n{stem}", f"{c['n_pairs']}")
        m(f"aaiMedian{stem}", pct(c["aai_median"], 1))
        m(f"aaiMin{stem}", pct(c["aai_min"], 1))
        m(f"aaiMax{stem}", pct(c["aai_max"], 1))
    sc = head["single_copy_vs_rbh_aai"]
    m("scN", f"{sc['n_pairs']}")
    m("scR", pct(sc["pearson_r"], 3))
    m("scDiff", pct(-sc["median_difference_sc_minus_rbh"], 2))
    # Fig 6 (document Figure 1)
    m("sdAai", pct(f6["sd108_aai_mean"], 1))
    m("sdOf", pct(f6["sd108_ortholog_fraction"], 3))
    m("sdOfPct", pct(100 * f6["sd108_ortholog_fraction"], 1))
    m("sdRbh", f"{f6['sd108_n_rbh']:,}")
    m("sdGenes", f"{f6['sd108_n_genes']:,}")
    m("cbsAai", pct(f6["cbs573_aai_mean"], 1))
    m("minVertAai", pct(f6["min_vertebrate_aai"], 1))
    m("nVertPairs", f"{f6['n_vertebrate_pairs']}")
    m("ycoAaiMin", pct(f6["saccharomycotina_aai_min"], 1))
    m("ycoAaiMax", pct(f6["saccharomycotina_aai_max"], 1))
    m("ycoOfMin", pct(f6["saccharomycotina_of_min"], 3))
    m("ycoOfMax", pct(f6["saccharomycotina_of_max"], 3))
    m("animAaiMin", pct(f6["animal_from_s288c_aai_min"], 1))
    m("animAaiMax", pct(f6["animal_from_s288c_aai_max"], 1))
    m("bacAaiMin", pct(f6["bacteria_from_s288c_aai_min"], 1))
    m("bacAaiMax", pct(f6["bacteria_from_s288c_aai_max"], 1))
    m("bacOfMax", pct(f6["bacteria_from_s288c_of_max"], 3))
    m("nSccPartners", f"{f6['n_s288c_partners']}")
    m("nYco", f"{f6['n_per_column']['saccharomycotina']}")
    m("nAnimal", f"{f6['n_per_column']['animal']}")
    m("nBac", f"{f6['n_per_column']['bacteria']}")
    m("nYard", f"{f6['n_human_yardstick']}")
    # Fig 4 (document Figure 5)
    m("nLeaves", f"{f4['n_leaves']}")
    m("nNamedClades", f"{f4['n_named_clades']}")
    m("nNoClade", f"{f4['n_isolates_no_clade']}")
    m("nAllPairs", f"{f4['n_all_pairs']:,}")
    m("medAllPairs", pct(f4["median_all_pairs"], 2))
    m("nBelow", f"{f4['n_named_below_overall']}")
    m("nNamedWithPairs", f"{f4['n_named_clades_with_pairs']}")
    m("cladeMedMin", pct(f4["min_clade_median"], 4))
    m("cladeMedMinName", tex_escape(f4["min_clade_median_clade"]))
    m("cladeMedMax", pct(f4["max_clade_median"], 2))
    m("cladeMedMaxName", tex_escape(f4["max_clade_median_clade"]))
    m("medCladeMed", pct(f4["median_of_clade_medians"], 2))
    m("pooledWithin", pct(f4["pooled_within_named_median"], 2))
    m("minBoxPairs", f"{f4['min_box_pairs']}")
    m("nBoxClades", f"{f4['n_box_clades']}")
    m("nStripClades", f"{f4['n_strip_clades']}")
    m("noCladeMed", pct(f4["no_clade_median_within"], 2))
    m("nearestStrain", tex_escape(f4["nearest_to_s288c"]))
    m("nearestDiv", pct(100 * f4["nearest_mean_divergence"], 4))
    m("nearestClade", tex_escape(f4["nearest_clade"]))
    m(
        "nearestCladeN",
        f"{int(summ.set_index('clade').loc[f4['nearest_clade'], 'n_isolates'])}",
    )
    m("nNegBranches", f"{f4['n_negative_branches']}")
    # Fig 5 (document Figure 6)
    for cls, stem in CLASS_STEM.items():
        m(f"nCls{stem}", f"{f5['n_per_class'][cls]}")
    m("nClasses", f"{len(CLASS_STEM)}")

    os.makedirs(osp.join(DOC, "tables"), exist_ok=True)
    with open(osp.join(DOC, "numbers.tex"), "w") as fh:
        fh.write(HEADER)
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    # figure file names (timestamped until approved)
    with open(osp.join(DOC, "figures.tex"), "w") as fh:
        fh.write(HEADER)
        for key, macro in (
            ("fig6", "figTransfer"),
            ("fig1", "figClaim"),
            ("fig2", "figHeatmaps"),
            ("fig3", "figTangle"),
            ("fig4", "figClades"),
            ("fig5", "figOrthologs"),
        ):
            name = osp.splitext(osp.basename(stats["paths"][key]))[0]
            fh.write(f"\\newcommand{{\\{macro}}}{{figures/{name}.pdf}}\n")
    # headline table
    rows = [
        (
            "Isolate vs S288C protein identity, median over isolates of per-ORF median",
            rf"{macros['isoMedMedId']}\% ($n$ = {macros['nIsolates']} isolates, "
            rf"{macros['nOrfs']} ORFs each)",
            "06",
        ),
        (
            "Same, median over isolates of mean over ORFs",
            rf"{macros['isoMedMeanId']}\% (range {macros['isoMinMeanId']} to "
            rf"{macros['isoMaxMeanId']}\%)",
            "06",
        ),
        (
            "Same, mean over equal-length ORFs only, median over isolates",
            rf"{macros['isoEqLenId']}\%",
            "06",
        ),
        (
            "One human genome vs GRCh38, peptide-altering sites (literature)",
            rf"{macros['litSitesLo']} to {macros['litSitesHi']}, i.e. "
            rf"{macros['litIdLo']} to {macros['litIdHi']}\% identity",
            "00",
        ),
    ]
    for label, stem in PAIR_STEM.items():
        name = label.replace("S. paradoxus", r"\org{S. paradoxus}").replace(
            "S. eubayanus", r"\org{S. eubayanus}"
        )
        rows.append(
            (
                f"{name} AAI, mean / median over RBHs",
                rf"{macros['aai' + stem]} / {macros['aaiMed' + stem]}\% "
                rf"($n_\mathrm{{RBH}}$ {macros['rbh' + stem]}; ortholog fraction "
                rf"{macros['of' + stem]})",
                "03",
            )
        )
    rows.append(
        (
            r"S288C/\org{E. coli} AAI",
            rf"{macros['aaiScerEcoli']}\% ($n_\mathrm{{RBH}}$ "
            rf"{macros['rbhScerEcoli']}; ortholog fraction {macros['ofScerEcoli']})",
            "03",
        )
    )
    rows.append(
        (
            r"S288C/\org{P. kudriavzevii} SD108 AAI",
            rf"{macros['sdAai']}\% ($n_\mathrm{{RBH}}$ {macros['sdRbh']} of "
            rf"{macros['sdGenes']} SD108 genes; ortholog fraction {macros['sdOf']})",
            "03",
        )
    )
    for cat, (stem, lab) in CAT_STEM.items():
        rows.append(
            (
                f"{lab} ({macros['n' + stem]} pairs)",
                rf"AAI median {macros['aaiMedian' + stem]}\%, range "
                rf"{macros['aaiMin' + stem]} to {macros['aaiMax' + stem]}\%",
                "07",
            )
        )
    rows.append(
        (
            f"Single-copy OG AAI vs RBH AAI, {macros['scN']} eukaryote pairs",
            rf"Pearson $r$ {macros['scR']}; proxy lower by {macros['scDiff']} "
            "points at the median",
            "06, 07",
        )
    )
    rows.append(
        (
            f"Within-clade SNP distance, {macros['nNamedWithPairs']} named clades "
            "with a pair",
            rf"clade medians {macros['cladeMedMin']} to {macros['cladeMedMax']}\%; "
            rf"all {macros['nAllPairs']} isolate pairs, median {macros['medAllPairs']}\%",
            "07",
        )
    )
    with open(osp.join(DOC, "tables", "headline.tex"), "w") as fh:
        fh.write(HEADER)
        fh.write(
            "\\begin{table}[H]\\centering\\footnotesize\n"
            "\\caption[]{Headline numbers. Script column: the numbered script under "
            "\\texttt{experiments/044-organism-similarity/scripts/} that computed the "
            "value; every value is read from \\texttt{results/headline\\_numbers.json} "
            "or \\texttt{results/figure\\_stats.json}.}\n\\label{tab:headline}\n"
            "\\begin{tabular}{@{}L{78mm} L{82mm} l@{}}\n\\toprule\n"
            "Quantity & Value & Script \\\\\n\\midrule\n"
        )
        for q, v, s in rows:
            fh.write(f"{q} & {v} & {s} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # organism table
    with open(osp.join(DOC, "tables", "organisms.tex"), "w") as fh:
        fh.write(HEADER)
        fh.write(
            "\\begin{table}[H]\\centering\\footnotesize\n"
            f"\\caption[]{{The {len(org)} organisms, {n_prot} with a proteome. Source: "
            "\\emph{tier} is a torchcell genomes tier set, \\emph{NCBI} an assembly "
            "fetched with \\texttt{datasets}; \\emph{Per gene} is the protein count after "
            "keeping the longest protein per gene (\\texttt{02\\_proteome\\_prep.py}; "
            "\\texttt{results/organisms.json}).}\n\\label{tab:organisms}\n"
            "\\begin{tabular}{@{}l l l l r@{}}\n\\toprule\n"
            "Organism & Source & Accession or tier set & Strain & Per gene \\\\\n"
            "\\midrule\n"
        )
        for o in org:
            p = o["proteome"]
            if p is None:
                src, acc, n = (
                    "absent",
                    tex_escape(o["reference_accession"]) + " (no annotation)",
                    "none",
                )
            else:
                src = p["source"] if p["source"] != "ncbi" else "NCBI"
                acc = tex_escape(p["accession"] or p["tier_set"])
                n = f"{o['prep']['n_proteins_prepped']:,}"
            strain = tex_escape(o["strain"] or "none")
            fh.write(
                f"{org_name(o['display'])} & {src} & {acc} & {strain} & {n} \\\\\n"
            )
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(
        f"wrote {len(macros)} macros, {len(rows)} headline rows, {len(org)} organisms"
    )


if __name__ == "__main__":
    main()
