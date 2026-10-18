# experiments/database/scripts/yeast_datasets_readout_table.py
# [[experiments.database.scripts.yeast_datasets_readout_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/yeast_datasets_readout_table
r"""The yeast datasets a served KG release carries, as one table.

Rows and experiment counts come from the committed release snapshot
``database/releases/<release>.json`` (what the served store holds, dataset by dataset).
The snapshot before it marks which rows are new. The three descriptive columns are the
one hand-authored input, ``CURATED`` below: whether the readout is gene-level (a
property of a gene or a named locus: an expression level, a protein abundance, a copy
number), what the readout is, and what genotype each record carries. Every yeast
dataset in the snapshot must have a registry entry and every entry must be in the
snapshot, so a dataset added to the KG without a row here fails the run.

Writes:
  - experiments/database/results/yeast_datasets_readout_table.csv
  - notes/experiments.database.yeast-datasets-readout-table.md  (the note, frontmatter kept)
  - notes/assets/pdf-output/experiments.database.yeast-datasets-readout-table.pdf  (with --pdf)

Run from the repo root:
  python experiments/database/scripts/yeast_datasets_readout_table.py --pdf
  python experiments/database/scripts/yeast_datasets_readout_table.py --release 2026.10.17-4295f7ca
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from torchcell.paper.tables import Column, PaperTable, Row, read_frontmatter

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RELEASES = REPO / "database" / "releases"
CSV_OUT = REPO / "experiments" / "database" / "results" / "yeast_datasets_readout_table.csv"
NOTE_OUT = REPO / "notes" / "experiments.database.yeast-datasets-readout-table.md"
PDF_OUT = REPO / "notes" / "assets" / "pdf-output" / "experiments.database.yeast-datasets-readout-table.pdf"
PDF_SCRIPT = REPO / "notes" / "assets" / "publish" / "scripts" / "bib_tex_pdf.sh"
ORGANISM = "scerevisiae"


class Curated(BaseModel):
    """The hand-authored description of one dataset's readout and genotype."""

    gene_level: bool
    readout: str
    genotype: str
    section: str


def _c(gene_level: bool, readout: str, genotype: str, section: str) -> Curated:
    return Curated(gene_level=gene_level, readout=readout, genotype=genotype, section=section)


EXPRESSION = "Expression"
PROTEOME = "Proteome"
GENOME = "Genome architecture"
FITNESS = "Fitness and genetic interaction"
VIABILITY = "Viability"
CHEMGEN = "Environment and chemical genomics"
ISOLATES = "Natural isolates and segregants"
MORPHOLOGY = "Morphology"
METABOLITE = "Metabolite and product"

CURATED: dict[str, Curated] = {
    "MicroarrayKemmeren2014Dataset": _c(True, "expression, microarray", "kanMX deletions", EXPRESSION),
    "SmMicroarraySameith2015Dataset": _c(True, "expression, microarray", "GSTF single deletions", EXPRESSION),
    "DmMicroarraySameith2015Dataset": _c(True, "expression, microarray", "GSTF double deletions", EXPRESSION),
    "MicroarrayScoreHu2007Dataset": _c(True, "expression score, microarray", "TF deletions", EXPRESSION),
    "InductionTimecourseHackett2020Dataset": _c(True, "expression timecourse, microarray", "inducible regulator overexpression", EXPRESSION),
    "NadalRibellesPerturbSeq2025Dataset": _c(True, "pseudobulk expression, Perturb-seq", "genome-scale deletions", EXPRESSION),
    "TfDeletionPseudobulkJackson2020Dataset": _c(True, "pseudobulk single-cell expression", "TF deletions", EXPRESSION),
    "CaudalPanTranscriptome2024Dataset": _c(True, "expression, RNA-seq TPM", "natural isolates", EXPRESSION),
    "Boocock2025Dataset": _c(True, "single-cell expression", "segregants, four crosses", EXPRESSION),
    "Nguessan2025Dataset": _c(True, "single-cell expression", "segregants, BY x RM", EXPRESSION),
    "PooledPromoterExpressionVaishnav2022Dataset": _c(True, "reporter expression per promoter, sort-seq", "random promoter library", EXPRESSION),
    "ProteomeMessner2023Dataset": _c(True, "proteome", "kanMX deletions", PROTEOME),
    "ProteomeMessner2021Dataset": _c(True, "proteome", "BY4741 under 16 drugs", PROTEOME),
    "ProteomeMuenzner2024Dataset": _c(True, "proteome, SWATH-MS", "natural isolates", PROTEOME),
    "ProteinTurnoverMuenzner2024Dataset": _c(True, "protein turnover, dynamic SILAC", "natural isolates", PROTEOME),
    "ProteomeZelezniak2018Dataset": _c(True, "proteome", "kinase deletions", PROTEOME),
    "GenomeArchitecturePuddu2019Dataset": _c(True, "copy number of named loci", "YKO deletions, homozygous diploid", GENOME),
    "SmfBaryshnikova2010Dataset": _c(False, "fitness", "SGA single mutants", FITNESS),
    "SmfCostanzo2016Dataset": _c(False, "fitness", "SGA single mutants", FITNESS),
    "DmfCostanzo2016Dataset": _c(False, "fitness", "SGA double mutants", FITNESS),
    "DmiCostanzo2016Dataset": _c(False, "genetic interaction", "SGA double mutants", FITNESS),
    "SmfKuzmin2018Dataset": _c(False, "fitness", "SGA single mutants", FITNESS),
    "DmfKuzmin2018Dataset": _c(False, "fitness", "SGA double mutants", FITNESS),
    "DmiKuzmin2018Dataset": _c(False, "genetic interaction", "SGA double mutants", FITNESS),
    "TmfKuzmin2018Dataset": _c(False, "fitness", "SGA triple mutants", FITNESS),
    "TmiKuzmin2018Dataset": _c(False, "genetic interaction", "SGA triple mutants", FITNESS),
    "SmfKuzmin2020Dataset": _c(False, "fitness", "SGA single mutants", FITNESS),
    "DmfKuzmin2020Dataset": _c(False, "fitness", "SGA double mutants", FITNESS),
    "DmiKuzmin2020Dataset": _c(False, "genetic interaction", "SGA double mutants", FITNESS),
    "TmfKuzmin2020Dataset": _c(False, "fitness", "SGA triple mutants", FITNESS),
    "TmiKuzmin2020Dataset": _c(False, "genetic interaction", "SGA triple mutants", FITNESS),
    "SmfODuibhir2014Dataset": _c(False, "fitness, relative growth rate", "kanMX deletions", FITNESS),
    "GeneEssentialitySgdDataset": _c(False, "essentiality, viability", "deletions and alleles", VIABILITY),
    "SynthLethalityYeastSynthLethDbDataset": _c(False, "synthetic lethality", "gene pairs", VIABILITY),
    "SynthRescueYeastSynthLethDbDataset": _c(False, "synthetic rescue", "gene pairs", VIABILITY),
    "EnvChemgenAuesukaree2009Dataset": _c(False, "stress tolerance, categorical", "kanMX deletions", CHEMGEN),
    "EnvChemgenCostanzo2021Dataset": _c(False, "condition fitness", "SGA single mutants", CHEMGEN),
    "EnvChemgenHoepfner2014Dataset": _c(False, "HIP-HOP sensitivity score", "barcoded deletions, het and hom", CHEMGEN),
    "EnvChemgenMota2024Dataset": _c(False, "growth response", "kanMX deletions", CHEMGEN),
    "EnvChemgenParsons2006Dataset": _c(False, "growth response, barcode pool", "barcoded deletions plus compounds", CHEMGEN),
    "EnvChemgenPiotrowski2017Dataset": _c(False, "growth response z-score, barcode pool", "barcoded deletion pool plus compounds", CHEMGEN),
    "EnvChemgenVanacloig2022Dataset": _c(False, "growth response", "barcoded deletions plus compounds", CHEMGEN),
    "EnvChemgenWildenhain2015Dataset": _c(False, "growth response z-score", "barcoded deletions plus compounds", CHEMGEN),
    "EnvChemgenYoshikawa2009Dataset": _c(False, "growth response, barcode pool", "barcoded deletions plus compounds", CHEMGEN),
    "HetHillenmeyer2008Dataset": _c(False, "fitness", "heterozygous deletions plus compounds", CHEMGEN),
    "HomHillenmeyer2008Dataset": _c(False, "fitness", "homozygous deletions plus compounds", CHEMGEN),
    "FattyAcidSmith2006Dataset": _c(False, "clear-zone score, ordinal", "kanMX deletions", CHEMGEN),
    "CrispriChemgenSmith2016Dataset": _c(False, "guide fitness", "CRISPRi plus drugs", CHEMGEN),
    "CrisprMagicLian2019Dataset": _c(False, "guide enrichment", "CRISPRa, CRISPRi, CRISPRd plus furfural", CHEMGEN),
    "CrispriMormino2022Dataset": _c(False, "growth response", "CRISPRi plus acetic acid", CHEMGEN),
    "InhibitorBioscreenVolk2021Dataset": _c(False, "Bioscreen growth (private)", "in-house strain under inhibitors", CHEMGEN),
    "Bloom2019Dataset": _c(False, "growth", "segregants, 16 crosses", ISOLATES),
    "CrispriSegregantHale2024Dataset": _c(False, "growth response", "segregants plus CRISPRi", ISOLATES),
    "IsolateGrowthPeter2018Dataset": _c(False, "colony growth", "natural isolates", ISOLATES),
    "DoublingTimeDeChiara2022Dataset": _c(False, "doubling time", "natural isolates", ISOLATES),
    "CellYieldDeChiara2022Dataset": _c(False, "cell yield", "natural isolates", ISOLATES),
    "LifeCycleDeChiara2022Dataset": _c(False, "sporulation and life span", "natural isolates", ISOLATES),
    "ScmdOhya2005Dataset": _c(False, "morphology, CalMorph", "kanMX deletions", MORPHOLOGY),
    "ScmdOhnuki2018Dataset": _c(False, "morphology, CalMorph", "essential-gene heterozygous diploids", MORPHOLOGY),
    "ScmdOhnuki2022Dataset": _c(False, "morphology, CalMorph", "drug-hypersensitive deletions", MORPHOLOGY),
    "AminoAcidCooper2010Dataset": _c(False, "amino-acid levels", "kanMX deletions", METABOLITE),
    "AminoAcidMulleder2016Dataset": _c(False, "amino-acid levels", "kanMX deletions", METABOLITE),
    "MetaboliteZelezniak2018Dataset": _c(False, "metabolome", "kinase deletions", METABOLITE),
    "MetaboliteDaSilveira2014Dataset": _c(False, "lipidome", "kinase and phosphatase deletions", METABOLITE),
    "OrganicAcidYoshida2012Dataset": _c(False, "organic-acid titers", "kanMX deletions", METABOLITE),
    "BetaxanthinCachera2023Dataset": _c(False, "betaxanthin proxy", "CRI-SPA deletions and gene additions", METABOLITE),
    "CarotenoidOzaydin2013Dataset": _c(False, "colony color score", "deletions plus carotenoid pathway", METABOLITE),
    "IsobutanolScreenLopez2024Dataset": _c(False, "biosensor fluorescence", "kanMX deletions", METABOLITE),
    "IsobutanolValidatedLopez2024Dataset": _c(False, "isobutanol titer", "kanMX deletions", METABOLITE),
    "FattyAcidXue2025Dataset": _c(False, "free fatty acid titer (in-house)", "combinatorial deletions", METABOLITE),
}

SECTION_ORDER = [EXPRESSION, PROTEOME, GENOME, FITNESS, VIABILITY, CHEMGEN, ISOLATES, MORPHOLOGY, METABOLITE]

COLUMNS = [
    Column(header="Dataset", align="l"),
    Column(header="Experiments", align="r"),
    Column(header="Gene-level", align="l"),
    Column(header="Readout", align="l"),
    Column(header="Genotype", align="l"),
    Column(header="New", align="l"),
]


class YeastRow(BaseModel):
    dataset: str
    experiments: int
    gene_level: bool
    readout: str
    genotype: str
    section: str
    new: bool
    previous_experiments: int | None


def _snapshot(release: str) -> dict:
    path = RELEASES / f"{release}.json"
    if not path.exists():
        raise SystemExit(f"no committed snapshot {path.relative_to(REPO)}")
    return json.loads(path.read_text())


def _releases_on_disk() -> list[str]:
    names = [p.stem for p in RELEASES.glob("*.json") if not p.name.endswith(".closures.json")]
    return sorted(names)


def build_rows(current: dict, previous: dict) -> list[YeastRow]:
    cur = {k: v for k, v in current["datasets"].items() if v.get("organism") == ORGANISM}
    prev = previous["datasets"]
    missing = sorted(set(cur) - set(CURATED))
    extra = sorted(set(CURATED) - set(cur))
    if missing or extra:
        raise SystemExit(
            f"CURATED and the snapshot disagree: not in CURATED {missing}; not served {extra}"
        )
    rows = []
    for name, rec in cur.items():
        c = CURATED[name]
        rows.append(
            YeastRow(
                dataset=name.removesuffix("Dataset"),
                experiments=rec["n_experiments"],
                gene_level=c.gene_level,
                readout=c.readout,
                genotype=c.genotype,
                section=c.section,
                new=name not in prev,
                previous_experiments=prev[name]["n_experiments"] if name in prev else None,
            )
        )
    rows.sort(key=lambda r: (SECTION_ORDER.index(r.section), -r.experiments, r.dataset))
    return rows


def write_csv(rows: list[YeastRow]) -> None:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "experiments", "gene_level", "readout", "genotype", "section", "new", "previous_experiments"])
        for r in rows:
            w.writerow([r.dataset, r.experiments, r.gene_level, r.readout, r.genotype, r.section, r.new, r.previous_experiments if r.previous_experiments is not None else ""])

# Pandoc sizes a pipe table's columns by the dash counts of its separator line once any
# row is wider than the text block, so the separator carries the column weights; equal
# dashes gave the long dataset names a sixth of the width and ran them into the counts.
COLUMN_WEIGHTS = {"Dataset": 30, "Experiments": 10, "Gene-level": 8, "Readout": 26, "Genotype": 26, "New": 5}


def _weighted(markdown: str) -> str:
    sep = "| " + " | ".join(
        {"l": ":" + "-" * (COLUMN_WEIGHTS[c.header] - 1), "r": "-" * (COLUMN_WEIGHTS[c.header] - 1) + ":"}[c.align]
        for c in COLUMNS
    ) + " |"
    plain = "| " + " | ".join({"l": ":--", "r": "--:"}[c.align] for c in COLUMNS) + " |"
    if plain not in markdown:
        raise SystemExit("the markdown separator line changed; update _weighted")
    return markdown.replace(plain, sep)


def write_note(rows: list[YeastRow], current: dict, previous: dict, generated_at: datetime) -> None:
    table = PaperTable(
        columns=COLUMNS,
        rows=[
            Row(
                cells={
                    "Dataset": r.dataset,
                    "Experiments": f"{r.experiments:,}",
                    "Gene-level": "yes" if r.gene_level else "no",
                    "Readout": r.readout,
                    "Genotype": r.genotype,
                    "New": "new" if r.new else "",
                },
                section=r.section,
            )
            for r in rows
        ],
    )
    n = len(rows)
    n_gene = sum(r.gene_level for r in rows)
    n_new = sum(r.new for r in rows)
    grew = [r for r in rows if r.previous_experiments is not None and r.previous_experiments != r.experiments]
    total = sum(r.experiments for r in rows)
    rel = current["release"]
    prev_rel = previous["release"]
    stamp = generated_at.strftime("%Y-%m-%d %H:%M %Z")
    script_rel = SCRIPT.relative_to(REPO)
    front = read_frontmatter(NOTE_OUT, default_title="Yeast Datasets Table")
    body = f"""

## {generated_at:%Y.%m.%d} - Yeast datasets served in KG {current['version']}

<!-- GENERATED by {script_rel} from database/releases/{rel}.json; the readout and genotype
columns are the script's CURATED registry. Do not edit the table by hand. -->

**Table 1.** The {n} *S. cerevisiae* datasets the served knowledge graph carries, grouped by readout. *Experiments* is the dataset's experiment count in the release snapshot. *Gene-level* is yes when the readout is a property of a gene or a named locus (an expression level, a protein abundance, a copy number) and no when it is a property of the whole cell (growth, viability, morphology, a metabolite or product level). *New* marks a dataset absent from the previous release. Release `{rel}` (KG {current['version']}, torchcell {current['torchcell_tag']}, built {current['built_at'][:10]}, {current['n_nodes']:,} nodes over {len(current['organisms'])} organisms), compared with `{prev_rel}` (KG {previous['version']}). Generated {stamp} by `{script_rel}`; the PDF is `notes/assets/pdf-output/{PDF_OUT.name}`.

{_weighted(table.to_markdown(sectioned=True, heading_level=3))}

### Totals

{n} datasets, {total:,} experiments; {n_gene} gene-level, {n - n_gene} whole-cell; {n_new} new since `{prev_rel}`, and {len(grew)} already-served datasets gained records under the store-what-was-released loader fixes ({", ".join(f"{r.dataset} {r.previous_experiments:,} to {r.experiments:,}" for r in sorted(grew, key=lambda r: r.previous_experiments - r.experiments)[:4])}, and {max(len(grew) - 4, 0)} others).

Data: `experiments/database/results/{CSV_OUT.name}`. Script note: [[experiments.database.scripts.yeast_datasets_readout_table]].
"""
    NOTE_OUT.write_text(front + body)


def render_pdf() -> None:
    notes = REPO / "notes"
    subprocess.run(
        ["bash", str(PDF_SCRIPT), str(NOTE_OUT), str(notes), NOTE_OUT.stem],
        check=True,
        cwd=REPO,
    )
    if not PDF_OUT.exists():
        raise SystemExit(f"{PDF_OUT} was not written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=None, help="release id; default the newest committed snapshot")
    ap.add_argument("--previous", default=None, help="the release to mark new rows against; default the snapshot before --release")
    ap.add_argument("--pdf", action="store_true", help="also render the note to its PDF")
    args = ap.parse_args()
    on_disk = _releases_on_disk()
    release = args.release or on_disk[-1]
    previous = args.previous or on_disk[on_disk.index(release) - 1]
    current = _snapshot(release)
    prev = _snapshot(previous)
    rows = build_rows(current, prev)
    generated_at = datetime.now().astimezone()
    write_csv(rows)
    write_note(rows, current, prev, generated_at)
    print(f"{len(rows)} yeast datasets in {release} ({sum(r.new for r in rows)} new since {previous}); wrote {CSV_OUT.relative_to(REPO)} and {NOTE_OUT.relative_to(REPO)}")
    if args.pdf:
        render_pdf()
        print(f"wrote {PDF_OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
