# experiments/008-xue-ffa/scripts/tf_single_mutant_fitness.py
# [[experiments.008-xue-ffa.scripts.tf_single_mutant_fitness]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/tf_single_mutant_fitness
#
# WHAT growth cost does each of the ten transcription-factor deletions of Xue et al. 2025
# carry on its own, in the yeast genetic-interaction screens?
#
# The 008 document reports trigenic interactions among ten TF deletions (FKH1, GCN5, MED4,
# OPI1, RFX1, RGR1, RPD3, SPT3, TFC7, YAP6) read out on fatty acid titer in a pox1 faa1
# faa4 chassis. A titer interaction is easier to read when the single-mutant GROWTH of each
# factor is on the table beside it, because a factor that is itself sick contributes a
# fitness term that the titer readout does not separate out.
#
# SOURCES, and the exact column each number is read from:
#
#   Costanzo et al. 2016 (SGA, single mutant fitness of the deletion / ts / DAmP arrays)
#     file   $DATA_ROOT/data/torchcell/smf_costanzo2016/raw/
#              strain_ids_and_single_mutant_fitness.xlsx
#     cols   "Single mutant fitness (26°)" + " stddev", "Single mutant fitness (30°)" + " stddev"
#     The stddev is a BOOTSTRAP SE over control screens, not a colony SD, and the
#     resampling unit is the screen (see SmfCostanzo2016Dataset: n = 17 screens).
#
#   Kuzmin et al. 2018 (trigenic tau-SGA)
#     file   $DATA_ROOT/data/torchcell/smf_kuzmin2018/raw/aao1729_data_s1.tsv
#     cols   "Array single mutant fitness", "Query single/double mutant fitness"
#     Neither column carries an uncertainty, so uncertainty is left empty here.
#
#   Kuzmin et al. 2020
#     files  $DATA_ROOT/data/torchcell/smf_kuzmin2020/raw/aaz5667-Table-S5.xlsx (the table
#            SmfKuzmin2020Dataset consumes; "Mutant type" == "Single mutant", cols
#            "Fitness" and "St.dev."), plus the array single-mutant column carried by
#            aaz5667-Table-S1.xlsx and aaz5667-Table-S3.xlsx.
#
# Fitness in every one of these tables is relative to wild type = 1.
#
# ESSENTIALITY is taken from the mirrored SGD gene records
# ($DATA_ROOT/data/sgd/genome/genes/<ORF>.json), using the same rule
# GeneEssentialitySgdDataset applies: a phenotype_details entry with mutant_type "null" and
# phenotype "inviable". The strain is reported alongside, because a null-inviable record in
# a non-S288C background is not the same evidence as one in S288C.
#
# Raw tables are read directly with pandas. Nothing here builds or downloads a dataset.

import json
import os
import os.path as osp
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results")

# The ten deleted transcription factors of the Xue et al. 2025 combinatorial panel.
TF_GENES = [
    "FKH1",
    "GCN5",
    "MED4",
    "OPI1",
    "RFX1",
    "RGR1",
    "RPD3",
    "SPT3",
    "TFC7",
    "YAP6",
]

COSTANZO_REL = (
    "data/torchcell/smf_costanzo2016/raw/strain_ids_and_single_mutant_fitness.xlsx"
)
KUZMIN2018_REL = "data/torchcell/smf_kuzmin2018/raw/aao1729_data_s1.tsv"
KUZMIN2020_S1_REL = "data/torchcell/smf_kuzmin2020/raw/aaz5667-Table-S1.xlsx"
KUZMIN2020_S3_REL = "data/torchcell/smf_kuzmin2020/raw/aaz5667-Table-S3.xlsx"
KUZMIN2020_S5_REL = "data/torchcell/smf_kuzmin2020/raw/aaz5667-Table-S5.xlsx"

# Sample sizes as the loaders document them. Neither is a column in the released file.
N_SAMPLES_COSTANZO_SMF_SCREENS = 17  # SmfCostanzo2016Dataset, bootstrap over screens
N_SAMPLES_KUZMIN_COMBINED = 4  # SmfKuzmin2020Dataset, colony sample SD

SOURCE_ORDER = [
    "costanzo2016_26C",
    "costanzo2016_30C",
    "kuzmin2018_array",
    "kuzmin2018_query",
    "kuzmin2020_S1_array",
    "kuzmin2020_S3_array",
    "kuzmin2020_S5_single",
]


@lru_cache(maxsize=None)
def read_excel_once(rel_file: str, skiprows: int) -> pd.DataFrame:
    """Read one raw Excel table from the mirror, once per process."""
    return pd.read_excel(osp.join(DATA_ROOT, rel_file), skiprows=skiprows)


@lru_cache(maxsize=None)
def read_kuzmin2018_once() -> pd.DataFrame:
    """Read the Kuzmin 2018 raw interaction table, once per process."""
    return pd.read_csv(osp.join(DATA_ROOT, KUZMIN2018_REL), sep="\t")


def allele_from_strain_id(strain_id: str) -> str:
    """Classify the SGA allele type from the strain-ID suffix.

    Mirrors SmfCostanzo2016Dataset.preprocess_raw, which is where the suffix
    convention is defined: damp -> DAmP, tsa/tsq -> temperature-sensitive allele,
    dma -> KanMX deletion array, sn -> NatMX deletion, S -> suppression allele.
    """
    suffix = strain_id.split("_", 1)[1] if "_" in strain_id else strain_id
    if "damp" in suffix:
        return "damp"
    if "tsa" in suffix or "tsq" in suffix:
        return "temperature_sensitive"
    if "dma" in suffix:
        return "KanMX_deletion"
    if "sn" in suffix:
        return "NatMX_deletion"
    if "S" in suffix:
        return "suppression_allele"
    return "unknown"


def sgd_essentiality(orf: str) -> str:
    """Report the SGD null-inviable evidence for an ORF, with its strain background."""
    path = osp.join(DATA_ROOT, "data/sgd/genome/genes", f"{orf}.json")
    if not osp.exists(path):
        return "no SGD record mirrored"
    with open(path) as f:
        record = json.load(f)
    inviable_null = [
        p
        for p in record.get("phenotype_details", [])
        if p["mutant_type"] == "null" and p["phenotype"]["display_name"] == "inviable"
    ]
    if not inviable_null:
        return "no (no SGD null-inviable record)"
    strains = sorted({p["strain"]["display_name"] for p in inviable_null})
    if "S288C" in strains:
        return "yes (SGD null inviable, S288C)"
    return f"yes (SGD null inviable, strain {'/'.join(strains)}; no S288C record)"


def blank_row(gene: str, orf: str, source: str, essential: str, rel_file: str) -> dict:
    """Return the row used when a gene has no strain at all in a source."""
    return {
        "gene": gene,
        "orf": orf,
        "source": source,
        "allele": "absent",
        "strain_id": "",
        "fitness": "",
        "uncertainty": "",
        "uncertainty_type": "",
        "n": "",
        "essential": essential,
        "file": rel_file,
        "column": "",
    }


def costanzo_rows(gene: str, orf: str, essential: str) -> list[dict]:
    """Read every Costanzo 2016 SMF strain for one ORF, at 26 C and at 30 C."""
    df = read_excel_once(COSTANZO_REL, 0)
    sub = df[df["Systematic gene name"] == orf]
    rows: list[dict] = []
    for temp, source in ((26, "costanzo2016_26C"), (30, "costanzo2016_30C")):
        fit_col = f"Single mutant fitness ({temp}°)"
        std_col = f"Single mutant fitness ({temp}°) stddev"
        if sub.empty:
            rows.append(blank_row(gene, orf, source, essential, COSTANZO_REL))
            continue
        for _, r in sub.iterrows():
            fitness = r[fit_col]
            std = r[std_col]
            reported = pd.notna(fitness)
            rows.append(
                {
                    "gene": gene,
                    "orf": orf,
                    "source": source,
                    "allele": allele_from_strain_id(r["Strain ID"]),
                    "strain_id": r["Strain ID"],
                    "fitness": fitness if reported else "",
                    "uncertainty": std if pd.notna(std) else "",
                    "uncertainty_type": "bootstrap_se" if pd.notna(std) else "",
                    "n": N_SAMPLES_COSTANZO_SMF_SCREENS if reported else "",
                    "essential": essential,
                    "file": COSTANZO_REL,
                    "column": fit_col,
                }
            )
    return rows


def kuzmin2018_rows(gene: str, orf: str, essential: str) -> list[dict]:
    """Read the Kuzmin 2018 array and query single-mutant fitness for one ORF."""
    df = read_kuzmin2018_once()
    rows: list[dict] = []

    array_orf = df["Array strain ID"].str.split("_", expand=True)[0]
    arr = df[array_orf == orf].drop_duplicates(
        subset=["Array strain ID", "Array allele name", "Array single mutant fitness"]
    )
    if arr.empty:
        rows.append(blank_row(gene, orf, "kuzmin2018_array", essential, KUZMIN2018_REL))
    else:
        for _, r in arr.iterrows():
            rows.append(
                {
                    "gene": gene,
                    "orf": orf,
                    "source": "kuzmin2018_array",
                    "allele": allele_from_strain_id(r["Array strain ID"]),
                    "strain_id": r["Array strain ID"],
                    "fitness": r["Array single mutant fitness"],
                    "uncertainty": "",
                    "uncertainty_type": "",
                    "n": "",
                    "essential": essential,
                    "file": KUZMIN2018_REL,
                    "column": "Array single mutant fitness",
                }
            )

    # The query column is a single-mutant fitness only on the digenic rows; on trigenic
    # rows it is the fitness of the double-mutant query, which is not an SMF.
    digenic = df[df["Combined mutant type"] == "digenic"]
    query_first_orf = digenic["Query strain ID"].str.split("+", expand=True)[0]
    qry = digenic[query_first_orf == orf].drop_duplicates(
        subset=["Query strain ID", "Query allele name"]
    )
    if qry.empty:
        rows.append(blank_row(gene, orf, "kuzmin2018_query", essential, KUZMIN2018_REL))
    else:
        for _, r in qry.iterrows():
            rows.append(
                {
                    "gene": gene,
                    "orf": orf,
                    "source": "kuzmin2018_query",
                    "allele": "KanMX_deletion" if "Δ" in r["Query allele name"] else "allele",
                    "strain_id": r["Query strain ID"],
                    "fitness": r["Query single/double mutant fitness"],
                    "uncertainty": "",
                    "uncertainty_type": "",
                    "n": "",
                    "essential": essential,
                    "file": KUZMIN2018_REL,
                    "column": "Query single/double mutant fitness",
                }
            )
    return rows


def kuzmin2020_array_rows(
    gene: str, orf: str, essential: str, rel_file: str, source: str
) -> list[dict]:
    """Read the array single-mutant fitness column of a Kuzmin 2020 interaction table."""
    df = read_excel_once(rel_file, 1)
    array_orf = df["Array strain ID"].astype(str).str.split("_", expand=True)[0]
    sub = df[array_orf == orf].drop_duplicates(
        subset=["Array strain ID", "Array allele name", "Array single mutant fitness"]
    )
    if sub.empty:
        return [blank_row(gene, orf, source, essential, rel_file)]
    return [
        {
            "gene": gene,
            "orf": orf,
            "source": source,
            "allele": allele_from_strain_id(str(r["Array strain ID"])),
            "strain_id": r["Array strain ID"],
            "fitness": r["Array single mutant fitness"],
            "uncertainty": "",
            "uncertainty_type": "",
            "n": "",
            "essential": essential,
            "file": rel_file,
            "column": "Array single mutant fitness",
        }
        for _, r in sub.iterrows()
    ]


def kuzmin2020_single_rows(gene: str, orf: str, essential: str) -> list[dict]:
    """Read the Kuzmin 2020 Table S5 single-mutant rows for one ORF."""
    df = read_excel_once(KUZMIN2020_S5_REL, 1)
    singles = df[df["Mutant type"] == "Single mutant"]
    sub = singles[singles["ORF1"] == orf]
    if sub.empty:
        return [
            blank_row(gene, orf, "kuzmin2020_S5_single", essential, KUZMIN2020_S5_REL)
        ]
    rows: list[dict] = []
    for _, r in sub.iterrows():
        std = r["St.dev."]
        rows.append(
            {
                "gene": gene,
                "orf": orf,
                "source": "kuzmin2020_S5_single",
                "allele": "KanMX_deletion" if "Δ" in str(r["Allele1"]) else "allele",
                "strain_id": r["Query Strain ID"],
                "fitness": r["Fitness"],
                "uncertainty": std if pd.notna(std) else "",
                "uncertainty_type": "sample_sd" if pd.notna(std) else "",
                "n": N_SAMPLES_KUZMIN_COMBINED if pd.notna(std) else "",
                "essential": essential,
                "file": KUZMIN2020_S5_REL,
                "column": "Fitness",
            }
        )
    return rows


def absence_audit(gene: str, orf: str) -> list[str]:
    """Cross-check an ORF-keyed absence against the allele-name columns of each file.

    An "absent" verdict reached by systematic name alone would be wrong if a file keyed a
    strain under a different ORF, so every allele-name column is searched for the gene name
    as well, and any hit whose ORF differs is reported.
    """
    name = gene.lower()
    hits: list[str] = []

    cost = read_excel_once(COSTANZO_REL, 0)
    m = cost[cost["Allele/Gene name"].str.lower().str.startswith(name)]
    for sys_name in sorted(set(m["Systematic gene name"])):
        if sys_name != orf:
            hits.append(f"{COSTANZO_REL} allele name {name}* under ORF {sys_name}")

    k18 = read_kuzmin2018_once()
    for col, id_col in (
        ("Array allele name", "Array strain ID"),
        ("Query allele name", "Query strain ID"),
    ):
        m = k18[k18[col].str.lower().str.contains(name, regex=False)]
        for sid in sorted(set(m[id_col])):
            if not sid.startswith(orf):
                hits.append(f"{KUZMIN2018_REL} {col} {name} under strain {sid}")

    for rel in (KUZMIN2020_S1_REL, KUZMIN2020_S3_REL):
        k = read_excel_once(rel, 1)
        m = k[k["Array allele name"].astype(str).str.lower().str.contains(name, regex=False)]
        for sid in sorted(set(m["Array strain ID"].astype(str))):
            if not sid.startswith(orf):
                hits.append(f"{rel} Array allele name {name} under strain {sid}")

    k5 = read_excel_once(KUZMIN2020_S5_REL, 1)
    m = k5[k5["Gene1"].astype(str).str.upper() == gene]
    for o1 in sorted(set(m["ORF1"].astype(str))):
        if o1 != orf:
            hits.append(f"{KUZMIN2020_S5_REL} Gene1 {gene} under ORF {o1}")

    return hits


def fitness_cell(row: pd.Series) -> str:
    """Format one fitness value with its uncertainty, or say why there is none."""
    if row["allele"] == "absent":
        return "absent"
    if row["fitness"] == "" or pd.isna(row["fitness"]):
        return "strain present, fitness not reported"
    text = f"{float(row['fitness']):.3f}"
    if row["uncertainty"] != "" and pd.notna(row["uncertainty"]):
        text += f" ({float(row['uncertainty']):.3f})"
    return text


def tex_escape(text: str) -> str:
    """Escape the LaTeX-special characters that appear in these strain and allele names."""
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("Δ", r"$\Delta$")
    )


# How each allele class prints in the Supplementary table. The deletion markers are two
# ways of building the same null and are named only when their numbers differ.
SHORT_ALLELE = {
    "KanMX_deletion": "KanMX",
    "NatMX_deletion": "NatMX",
    "damp": "DAmP",
    "temperature_sensitive": "ts",
}


def main() -> None:
    """Look up the ten factors in each screen, then write the CSV, the TeX body, and print."""
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )

    records: list[dict] = []
    for gene in TF_GENES:
        orf = genome.resolve_gene_name(gene).systematic_name
        essential = sgd_essentiality(orf)
        records += costanzo_rows(gene, orf, essential)
        records += kuzmin2018_rows(gene, orf, essential)
        records += kuzmin2020_array_rows(
            gene, orf, essential, KUZMIN2020_S1_REL, "kuzmin2020_S1_array"
        )
        records += kuzmin2020_array_rows(
            gene, orf, essential, KUZMIN2020_S3_REL, "kuzmin2020_S3_array"
        )
        records += kuzmin2020_single_rows(gene, orf, essential)

    df = pd.DataFrame.from_records(records)
    df["source"] = pd.Categorical(df["source"], categories=SOURCE_ORDER, ordered=True)
    df = df.sort_values(["gene", "source", "strain_id"]).reset_index(drop=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = osp.join(RESULTS_DIR, "tf_single_mutant_fitness.csv")
    df.to_csv(csv_path, index=False)

    df["fitness_cell"] = df.apply(fitness_cell, axis=1)

    tex_lines = []
    for _, r in df.iterrows():
        tex_lines.append(
            " & ".join(
                [
                    tex_escape(r["gene"]),
                    tex_escape(r["orf"]),
                    tex_escape(r["source"]),
                    tex_escape(r["allele"]),
                    tex_escape(r["strain_id"]) if r["strain_id"] else "--",
                    tex_escape(r["fitness_cell"]),
                    tex_escape(r["essential"]),
                ]
            )
            + r" \\"
        )
    tex_path = osp.join(RESULTS_DIR, "tf_single_mutant_fitness.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")

    # One row per gene for the Supplementary table. Eighty-four rows is the audit trail
    # and belongs in the CSV; what a reader needs on the page is which allele each screen
    # carries and what it scored, so a gene with two identical deletion strains collapses
    # to one number and a gene with two allele CLASSES keeps both, since a DAmP and a
    # temperature-sensitive allele are different perturbations rather than two estimates.
    # The fitness column carries "absent" and "strain present, fitness not reported"
    # beside the numbers, so the numeric view is taken here rather than assumed.
    df["fitness_value"] = pd.to_numeric(df["fitness"], errors="coerce")
    compact_lines = []
    for gene, g in df.groupby("gene", observed=True, sort=True):
        orf = g["orf"].iloc[0]
        cells = []
        for source in ("costanzo2016_26C", "costanzo2016_30C"):
            sub = g[(g["source"] == source) & g["fitness_value"].notna()]
            by_allele = sub.groupby("allele", observed=True)["fitness_value"].first()
            # Name the allele only where the alleles DISAGREE. Two marker swaps of one
            # deletion give the same number and naming both says nothing; a DAmP against
            # a temperature-sensitive allele gives two numbers and the reader needs to
            # know which is which.
            if by_allele.round(3).nunique() > 1:
                cells.append(", ".join(f"{v:.2f} ({SHORT_ALLELE[k]})"
                                       for k, v in by_allele.items()))
            elif len(by_allele):
                cells.append(f"{by_allele.iloc[0]:.2f}")
            else:
                cells.append("--")
        kz = g[g["source"].astype(str).str.startswith("kuzmin") & g["fitness_value"].notna()]
        cells.append(f"{kz['fitness_value'].mean():.2f}" if len(kz) else "--")
        allele = ", ".join(sorted({SHORT_ALLELE[a] for a in g["allele"]
                                   if a != "absent"})) or "no strain"
        note = "inviable" if g["essential"].iloc[0].startswith("yes") else ""
        compact_lines.append(" & ".join([
            tex_escape(gene), tex_escape(orf), tex_escape(allele),
            *(tex_escape(c) for c in cells), note]) + r" \\")
    # HYPHENS, not underscores, in this one file name. It is the only artifact here that a
    # .tex file pulls in with \input, and LaTeX's file-name parsing chokes on the
    # underscore, reporting it as a misplaced \noalign at the \bottomrule after the table.
    compact_path = osp.join(RESULTS_DIR, "tf-single-mutant-fitness-compact.tex")
    # The file carries the WHOLE tabular, not just its rows. A file of bare rows cannot be
    # \input from inside a tabular: LaTeX reports a misplaced \noalign at the \bottomrule
    # that follows, whatever the path or the line endings. Input at top level of a float is
    # unproblematic, so the environment travels with the rows.
    header = (r"Gene & ORF & Screen allele & Costanzo 26\,\textdegree C & "
              r"Costanzo 30\,\textdegree C & Kuzmin & \\")
    with open(compact_path, "w") as f:
        f.write("\n".join([
            "%% SOURCE: generated by experiments/008-xue-ffa/scripts/"
            "tf_single_mutant_fitness.py from the raw single-mutant-fitness tables of",
            "%% Costanzo 2016 and Kuzmin 2018/2020; do not edit by hand.",
            r"\begin{tabular}{llllll@{}l}", r"\toprule", header, r"\midrule",
            *compact_lines, r"\bottomrule", r"\end{tabular}", ""]))
    print(f"compact {compact_path}")

    cols = ["gene", "orf", "source", "allele", "strain_id", "fitness_cell", "n"]
    widths = {c: max(len(c), *(len(str(v)) for v in df[c])) for c in cols}
    header = "  ".join(c.upper().ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    print("\nAllele-name cross-check of the ORF-keyed lookup")
    for gene in TF_GENES:
        orf = genome.resolve_gene_name(gene).systematic_name
        hits = absence_audit(gene, orf)
        print(f"  {gene:5s} {'clean' if not hits else '; '.join(hits)}")

    print("\nSGD null-inviable evidence per gene")
    for gene, ess in df.groupby("gene", observed=True)["essential"].first().items():
        print(f"  {gene:5s} {ess}")

    print(f"\ncsv {csv_path}")
    print(f"tex {tex_path}")


if __name__ == "__main__":
    main()
