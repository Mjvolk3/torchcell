# experiments/043-expression-proteome-morphology/scripts/table_data_size_043.py
# [[experiments.043-expression-proteome-morphology.scripts.table_data_size_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/table_data_size_043
r"""The size of the expression, proteome and morphology data behind the 043 cell model, as one table.

One row per dataset the 043 store serves: what was measured, the genotype class, the
strain records, and the genes (proteins, features) a record carries. The record counts
are the served store's own census (``build_043_store.py`` writes
``results/build_043_store_census_v2.json`` when it builds the store, 173,805 records on
KG release 2026.10.17-4295f7ca); the per-record label counts are the medians of the full
data pass (``data_shapes_043.py``, ``results/data_shapes_043.json``). Written for the
future-directions section of the dissertation (chapter 6, the file
``ch08-scaling-strain-engineering.tex``), where the point is how many strain records with a
molecular or cellular readout one host already yields.

    python experiments/043-expression-proteome-morphology/scripts/table_data_size_043.py \\
        [--thesis-tables-dir <thesis>/thesis/tables]
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import shutil

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(ROOT, "results")
CENSUS = osp.join(RESULTS, "build_043_store_census_v2.json")
SHAPES = osp.join(RESULTS, "data_shapes_043.json")
OUT = osp.join(RESULTS, "tab-ch08-omics-data-size.tex")
MD_OUT = osp.join(RESULTS, "tab-ch08-omics-data-size.md")

#: (dataset class in the store, display name, measurement, genotype, unit of the
#: per-record count, dissertation bibliography key or "" when references.bib has none)
ROWS = {
    "Expression": [
        (
            "MicroarrayKemmeren2014Dataset",
            "Kemmeren 2014",
            "microarray, log2 ratio to wild type",
            "single deletion",
            "genes",
            "kemmerenLargeScaleGeneticPerturbationsReveal2014",
        ),
        (
            "SmMicroarraySameith2015Dataset",
            "Sameith 2015, singles",
            "microarray, log2 ratio",
            "single deletion",
            "genes",
            "sameithHighResolutionGeneExpressionAtlas2015",
        ),
        (
            "DmMicroarraySameith2015Dataset",
            "Sameith 2015, doubles",
            "microarray, log2 ratio",
            "double deletion",
            "genes",
            "sameithHighResolutionGeneExpressionAtlas2015",
        ),
        (
            "MicroarrayScoreHu2007Dataset",
            "Hu 2007",
            "microarray, error-model score",
            "single deletion",
            "genes",
            "",
        ),
        (
            "TfDeletionPseudobulkJackson2020Dataset",
            "Jackson 2020",
            "single-cell RNA-seq, pseudobulk UMI",
            "single deletion",
            "genes",
            "jacksonGeneRegulatoryNetworkReconstructionUsing2020",
        ),
        (
            "NadalRibellesPerturbSeq2025Dataset",
            "Nadal-Ribelles 2025",
            "Perturb-seq, pseudobulk log2 fold change",
            "single deletion",
            "genes",
            "nadalribellesSingleCellResolvedGenotypePhenotype2025",
        ),
        (
            "CaudalPanTranscriptome2024Dataset",
            "Caudal 2024",
            "RNA-seq, TPM",
            "natural isolate",
            "genes",
            "caudalPanTranscriptomeRevealsLargeAccessory2024",
        ),
        (
            "Boocock2025Dataset",
            "Boocock 2025",
            "single-cell RNA-seq, UMI per cell",
            "segregant",
            "genes",
            "boocockSinglecellEQTLMapping2025",
        ),
        (
            "Nguessan2025Dataset",
            "N'Guessan 2025",
            "single-cell RNA-seq, UMI per cell",
            "segregant",
            "genes",
            "",
        ),
        (
            "PooledPromoterExpressionVaishnav2022Dataset",
            "Vaishnav 2022",
            "sort-seq promoter reporter",
            "80-mer promoter variant",
            "reporter",
            "vaishnavEvolutionEvolvabilityEngineeringGeneRegulatory2022",
        ),
    ],
    "Proteome": [
        (
            "ProteomeMessner2023Dataset",
            "Messner 2023",
            "MaxLFQ protein abundance",
            "single deletion",
            "proteins",
            "messnerProteomicLandscapeGenomeWideGenetic2023",
        ),
        (
            "ProteomeMuenzner2024Dataset",
            "Muenzner 2024",
            "MaxLFQ protein abundance",
            "natural isolate",
            "proteins",
            "muenznerNaturalProteomeDiversityLinksAneuploidy2024",
        ),
        (
            "ProteinTurnoverMuenzner2024Dataset",
            "Muenzner 2024, turnover",
            "protein degradation rate",
            "natural isolate",
            "proteins",
            "muenznerNaturalProteomeDiversityLinksAneuploidy2024",
        ),
        (
            "ProteomeZelezniak2018Dataset",
            "Zelezniak 2018",
            "protein abundance, log signal",
            "single deletion",
            "proteins",
            "zelezniakMachineLearningPredictsYeastMetabolome2018",
        ),
    ],
    "Morphology": [
        (
            "ScmdOhya2005Dataset",
            "Ohya 2005",
            "CalMorph, 281 cell features",
            "single deletion",
            "features",
            "ohyaHighDimensionalLargeScalePhenotyping2005",
        )
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--thesis-tables-dir", default="", help="copy the table there as well"
    )
    args = ap.parse_args()

    with open(CENSUS) as f:
        census = json.load(f)
    with open(SHAPES) as f:
        shapes = json.load(f)["datasets"]
    n_by_dataset = census["n_records_by_dataset_raw"]
    assert sum(n_by_dataset.values()) == census["n_served"], (
        "census rows do not sum to the served count"
    )
    listed = {cls for rows in ROWS.values() for cls, *_ in rows}
    assert listed == set(n_by_dataset), (
        f"table rows and census datasets differ: {listed ^ set(n_by_dataset)}"
    )

    caption = (
        "\\textbf{Strain records with an expression, proteome or morphology readout in the TorchCell "
        "store.} One row per dataset: the readout, the genotype class, the strain records the store "
        "serves and the median number of genes, proteins or features one record carries, on the "
        f"{census['served_release']} release. A record is one strain, or one cell for the two "
        "single-cell segregant studies and one promoter variant for the sort-seq reporter. The "
        "Sameith single deletions are the Kemmeren arrays and the Ohya morphology is held out of "
        "training, so the thirteen training datasets hold "
        f"{census['n_served'] - n_by_dataset['SmMicroarraySameith2015Dataset'] - n_by_dataset['ScmdOhya2005Dataset']:,} records."
    )
    header = [
        "%% SOURCE: torchcell experiments/043-expression-proteome-morphology/scripts/table_data_size_043.py "
        "(reads results/build_043_store_census_v2.json and results/data_shapes_043.json)",
        "%% AUTO-GENERATED -- do not hand-edit; edits are lost on the next run.",
        "%% Regenerate: ~/miniconda3/envs/torchcell/bin/python experiments/043-expression-proteome-morphology/scripts/table_data_size_043.py "
        "--thesis-tables-dir <thesis>/thesis/tables",
        f"%% Records: the served store census (KG release {census['served_release']}, store {census['store_version']}, "
        f"GilaHyper job {census['slurm_job_id']}). Per-record counts: medians over the full data pass of data_shapes_043.py.",
        "%% Hu 2007 and N'Guessan 2025 carry no \\cite because references.bib has no key for them (checked 2026-11-02).",
    ]
    lines = header + [
        "\\begin{table}[!ht]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{5pt}",
        f"\\caption{{{caption}}}",
        "\\label{tab:ch08-omics-data-size}",
        "\\begin{tabular}{@{}l l l r r@{}}",
        "\\toprule",
        "Source & Readout & Genotype & Strain records & Measured per record (median) \\\\",
        "\\midrule",
    ]
    md = [
        "| Source | Readout | Genotype | Strain records | Measured per record (median) |",
        "|---|---|---|---:|---:|",
    ]
    total = 0
    for i, (section, rows) in enumerate(ROWS.items()):
        if i:
            lines.append("\\addlinespace")
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{section}}}}} \\\\")
        md.append(f"| **{section}** | | | | |")
        for cls, name, readout, genotype, unit, key in rows:
            n = n_by_dataset[cls]
            total += n
            per = int(round(shapes[cls]["label_keys_per_record"]["median"]))
            src = f"{name} \\cite{{{key}}}" if key else name
            lines.append(
                f"{src} & {readout} & {genotype} & {n:,} & {per:,} {unit} \\\\"
            )
            md.append(f"| {name} | {readout} | {genotype} | {n:,} | {per:,} {unit} |")
    lines += [
        "\\midrule",
        f"All & & & {total:,} & \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    md.append(f"| **All** | | | **{total:,}** | |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(MD_OUT, "w") as f:
        f.write("\n".join(md) + "\n")
    print(OUT)
    print(MD_OUT)
    if args.thesis_tables_dir:
        dst = osp.join(args.thesis_tables_dir, "tab-ch08-omics-data-size.tex")
        shutil.copy(OUT, dst)
        print(dst)


if __name__ == "__main__":
    main()
