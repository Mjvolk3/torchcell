# experiments/036-dataset-fixes-before-kg-build/scripts/keep_records_887_sd_zero_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.keep_records_887_sd_zero_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/keep_records_887_sd_zero_inventory
"""Measure what the two 2026-10-12 owner decisions keep, for #887 and the SD-of-zero rows.

The owner decided on 2026-10-12 that a released record is never dropped as a remedy, a
conflict or an unknown travels on the record as a typed flag or typed gap, and the
uncertainty a source releases is stored exactly as released ("if no uncertainty is
provided we can't do anything about it ... but if uncertainty exists we cannot be
missing it!"). This script measures, from the pinned bytes, everything that decision
rests on. It writes nothing to any store.

Xue 2025 (#887)
    the released genotype statement (both sheets of the pinned workbook, every cell),
    the allele search the decision asks for (RGR1, MED4, TFC7, truncation, DAmP, degron,
    conditional, hypomorph, partial), the per-gene record counts, the SGD null-mutant
    annotations of the three genes from the cached SGD locus records the essentiality
    store is built from, the essentiality store and Ohya 2005 membership, and the dev
    store's own records.

Released SDs of exactly 0 (#894, #895, #897, #930)
    per released table: the zero-SD rows by combined-mutant type, the released text of
    the SD and P-value cells on those rows; the SI column definitions and every SI line
    naming a standard deviation, a zero, or a "not computed" meaning; and, for the
    Kuzmin dev stores, how the zero rows are stored. The Costanzo 2016 store is read only
    with ``--costanzo-store`` (20.7 M records; another branch rebuilds it).

Writes ``experiments/036-dataset-fixes-before-kg-build/results/
keep_records_887_sd_zero_inventory.json``; ``--after`` instead censuses the two dev
stores PR #1054 rebuilds (records, typed gaps by field, uncertainty labels, released
zeros) into ``keep_records_887_sd_zero_stores_after.json``. Run from the repo root (under slurm: the
Costanzo tables are 2 GB)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/keep_records_887_sd_zero_inventory.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import pickle
import re
from collections import Counter
from collections.abc import Iterator
from typing import Any

import lmdb
import openpyxl
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.datasets.scerevisiae import xue2025

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
OUT_NAME = "keep_records_887_sd_zero_inventory.json"

# The three genes #887 names, with the ORF the loader stores for each (TFC7 since #886).
XUE_GENES = {"RGR1": "YLR071C", "MED4": "YOR174W", "TFC7": "YOR110W"}
# Allele-statement search terms the 2026-10-12 decision names, matched case-insensitively.
ALLELE_TERMS = [
    "RGR1",
    "MED4",
    "TFC7",
    "truncation",
    "DAmP",
    "degron",
    "conditional",
    "hypomorph",
    "partial",
]
# SI lines that would state what a released SD means: matched case-insensitively.
SD_TERMS = [
    "standard deviation",
    "not computed",
    "not calculated",
    "undefined",
    "zero",
    "single colony",
    "pseudocount",
]


class Frozen(BaseModel):
    """Strict, immutable result record."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FileRef(Frozen):
    """A file read by this script and the sha256 of its bytes."""

    path: str
    sha256: str


class VerbatimLine(Frozen):
    """A verbatim quote checked to be a substring of line ``line`` of ``file``."""

    file: FileRef
    line: int
    quote: str


class CellHit(Frozen):
    """A workbook cell whose text matched an allele search term."""

    sheet: str
    cell: str
    text: str
    terms: list[str]


class SgdNullInviable(Frozen):
    """One SGD ``phenotype_details`` entry: a null mutant annotated ``inviable``."""

    strain: str
    experiment: str
    reference: str
    pubmed_id: int | None


class XueGene(Frozen):
    """One of the three #887 genes: its records, its SGD annotations, its store rows."""

    common_name: str
    systematic_name: str
    records: int
    sgd_locus: FileRef
    sgd_null_inviable: list[SgdNullInviable]
    sgd_null_inviable_s288c: int
    essentiality_store_keys: list[str]
    in_ohya2005: bool


class XueInventory(Frozen):
    """#887: the release, the allele search, and the stored records."""

    workbook: FileRef
    mirror_manifest_files: list[str]
    sheets: dict[str, int]
    genotype_rows: int
    wild_type_rows: int
    records: int
    records_with_any_of_three: int
    records_with_rgr1_or_med4: int
    genotype_statement: list[CellHit]
    allele_terms: list[str]
    allele_hits: list[CellHit]
    genes: list[XueGene]
    store_records: int
    store_records_with_any_of_three: int
    store_perturbation_types_of_three: dict[str, int]


class ZeroSdTable(Frozen):
    """One released table: its zero-SD rows and the released cells on them."""

    citation_key: str
    file: FileRef
    sd_column: str
    rows: int
    zero_sd_rows: int
    zero_sd_rows_by_type: dict[str, int]
    zero_sd_cell_text: dict[str, int]
    p_value_on_zero_sd_rows: dict[str, int]


class ZeroSdStore(Frozen):
    """One dev store: how its released zero SDs are stored."""

    slug: str
    records: int
    zero_uncertainty_records: int
    uncertainty_types: dict[str, int]
    n_samples: dict[str, int]
    fitness_se: dict[str, int]
    fitness_std: dict[str, int]


class SiEvidence(Frozen):
    """What one paper and its SI say about the SD column.

    ``definitions`` are the verbatim column definitions in the SI; ``term_lines`` and
    ``paper_term_lines`` list every line of the SI and of the main text matching each of
    ``SD_TERMS``, so "the source states nothing else" is a measured statement.
    """

    citation_key: str
    file: FileRef
    definitions: list[VerbatimLine]
    term_lines: dict[str, list[int]]
    paper: FileRef
    paper_term_lines: dict[str, list[int]]


class StoreCensus(Frozen):
    """A rebuilt dev store: its count, typed gaps and uncertainty labels (``--after``)."""

    slug: str
    built_at: str
    torchcell_commit: str
    records: int
    gaps_by_field: dict[str, int]
    uncertainty_types: dict[str, int]
    zero_uncertainty_records: int


class Inventory(Frozen):
    """Everything this script measured."""

    xue2025: XueInventory
    zero_sd_tables: list[ZeroSdTable]
    zero_sd_stores: list[ZeroSdStore]
    si_evidence: list[SiEvidence]


def sha256_of(path: str) -> str:
    """Hex sha256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: str) -> FileRef:
    """Path plus the sha256 of its bytes."""
    return FileRef(path=path, sha256=sha256_of(path))


def lmdb_records(root: str) -> Iterator[dict[str, Any]]:
    """Every stored record of a dev store, read-only."""
    env = lmdb.open(
        osp.join(root, "processed", "lmdb"), readonly=True, lock=False, readahead=False
    )
    with env.begin() as txn:
        for _, value in txn.cursor():
            yield pickle.loads(value)
    env.close()


def workbook_cells(path: str) -> Iterator[tuple[str, str, str]]:
    """(sheet, cell coordinate, text) of every non-empty cell of a workbook."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for sheet in book.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None:
                    yield sheet.title, str(cell.coordinate), str(cell.value)
    book.close()


def sgd_null_inviable(path: str) -> list[SgdNullInviable]:
    """The ``null`` / ``inviable`` entries of a cached SGD locus record, any strain."""
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    return [
        SgdNullInviable(
            strain=entry["strain"]["display_name"],
            experiment=entry["experiment"]["display_name"],
            reference=entry["reference"]["display_name"],
            pubmed_id=entry["reference"].get("pubmed_id"),
        )
        for entry in record["phenotype_details"]
        if entry["mutant_type"] == "null"
        and entry["phenotype"]["display_name"] == "inviable"
    ]


def xue_inventory(data_root: str) -> XueInventory:
    """#887: the workbook, the allele search, SGD, the essentiality store, the dev store."""
    library = osp.join(data_root, "torchcell-library", "xue2025")
    path = osp.join(library, "data", xue2025.DATA_FILENAME)
    workbook = file_ref(path)
    if workbook.sha256 != xue2025.DATA_SHA256:
        raise ValueError(f"{path} is off the loader's pin {xue2025.DATA_SHA256}")
    with open(osp.join(library, "manifest.json"), encoding="utf-8") as handle:
        manifest_files = [entry["path"] for entry in json.load(handle)["files"]]

    cells = list(workbook_cells(path))
    sheets = dict(Counter(sheet for sheet, _, _ in cells))
    hits = []
    for sheet, coordinate, text in cells:
        terms = [term for term in ALLELE_TERMS if term.lower() in text.lower()]
        if terms:
            hits.append(CellHit(sheet=sheet, cell=coordinate, text=text, terms=terms))

    # The genotype statement: the code letter of each gene on the Abbreviations sheet
    # and the single-deletion strain row of the titer sheet ("R 4Δ" = RGR1 + baseline).
    code_of = {gene: code for code, gene in xue2025._EXPECTED_CODE_TO_GENE.items()}
    statement_text = {
        *(code_of[gene] for gene in XUE_GENES),
        *(f"{code_of[gene]} 4Δ" for gene in XUE_GENES),
        *XUE_GENES,
    }
    statement = [
        CellHit(sheet=sheet, cell=coordinate, text=text, terms=[])
        for sheet, coordinate, text in cells
        if text in statement_text
    ]

    raw = pd.read_excel(path, sheet_name=xue2025._TITER_SHEET, header=None)
    labels = [str(label).strip() for label in raw.iloc[1:, 0]]
    dataset = xue2025.FattyAcidXue2025Dataset.__new__(xue2025.FattyAcidXue2025Dataset)
    decoded = [
        dataset._decode_genotype(label, xue2025._EXPECTED_CODE_TO_GENE)
        for label in labels
    ]
    records = [genes for genes in decoded if genes]
    rgr1_med4 = sum(1 for genes in records if {"RGR1", "MED4"} & set(genes))

    store_root = osp.join(data_root, "data/torchcell/ffa_xue2025")
    store = list(lmdb_records(store_root))
    three = set(XUE_GENES.values())
    perturbation_types: Counter[str] = Counter()
    store_with_three = 0
    store_per_orf: Counter[str] = Counter()
    for item in store:
        perturbations = item["experiment"]["genotype"]["perturbations"]
        orfs = {p["systematic_gene_name"] for p in perturbations}
        store_with_three += bool(orfs & three)
        for p in perturbations:
            if p["systematic_gene_name"] in three:
                perturbation_types[p["perturbation_type"]] += 1
                store_per_orf[p["systematic_gene_name"]] += 1

    essential_keys: dict[str, list[str]] = {orf: [] for orf in three}
    essentiality_root = osp.join(data_root, "data/torchcell/gene_essentiality_sgd")
    env = lmdb.open(
        osp.join(essentiality_root, "processed", "lmdb"), readonly=True, lock=False
    )
    with env.begin() as txn:
        for key, value in txn.cursor():
            item = pickle.loads(value)
            for p in item["experiment"]["genotype"]["perturbations"]:
                if p["systematic_gene_name"] in three:
                    essential_keys[p["systematic_gene_name"]].append(key.decode())
    env.close()
    ohya_path = osp.join(
        data_root, "data/torchcell/scmd_ohya2005/preprocess/gene_set.json"
    )
    with open(ohya_path, encoding="utf-8") as handle:
        ohya = set(json.load(handle))

    genes = []
    for common, orf in XUE_GENES.items():
        locus = osp.join(data_root, "data/sgd/genome/genes", f"{orf}.json")
        annotations = sgd_null_inviable(locus)
        genes.append(
            XueGene(
                common_name=common,
                systematic_name=orf,
                records=sum(1 for genes_ in records if common in genes_),
                sgd_locus=file_ref(locus),
                sgd_null_inviable=annotations,
                sgd_null_inviable_s288c=sum(a.strain == "S288C" for a in annotations),
                essentiality_store_keys=sorted(essential_keys[orf], key=int),
                in_ohya2005=orf in ohya,
            )
        )
        if store_per_orf[orf] != genes[-1].records:
            raise ValueError(
                f"{orf}: the store holds {store_per_orf[orf]} perturbations, the "
                f"release {genes[-1].records}"
            )
    return XueInventory(
        workbook=workbook,
        mirror_manifest_files=manifest_files,
        sheets=sheets,
        genotype_rows=len(labels),
        wild_type_rows=sum(1 for genes_ in decoded if not genes_),
        records=len(records),
        records_with_any_of_three=sum(
            1 for genes_ in records if set(XUE_GENES) & set(genes_)
        ),
        records_with_rgr1_or_med4=rgr1_med4,
        genotype_statement=statement,
        allele_terms=ALLELE_TERMS,
        allele_hits=hits,
        genes=genes,
        store_records=len(store),
        store_records_with_any_of_three=store_with_three,
        store_perturbation_types_of_three=dict(perturbation_types),
    )


def zero_sd_rows(frame: pd.DataFrame, sd: str) -> pd.DataFrame:
    """Rows whose released SD text parses to exactly 0."""
    return frame[pd.to_numeric(frame[sd], errors="raise") == 0]


def table_entry(
    citation_key: str, path: str, frame: pd.DataFrame, sd: str, kind: str | None
) -> ZeroSdTable:
    """Count one released table's zero-SD rows, read as text so the cells are verbatim."""
    zero = zero_sd_rows(frame, sd)
    return ZeroSdTable(
        citation_key=citation_key,
        file=file_ref(path),
        sd_column=sd,
        rows=len(frame),
        zero_sd_rows=len(zero),
        zero_sd_rows_by_type=(
            {str(k): int(v) for k, v in zero[kind].value_counts().items()}
            if kind is not None
            else {"all": len(zero)}
        ),
        zero_sd_cell_text={str(k): int(v) for k, v in zero[sd].value_counts().items()},
        p_value_on_zero_sd_rows={
            str(k): int(v) for k, v in zero["P-value"].value_counts().items()
        },
    )


def zero_sd_tables(data_root: str) -> list[ZeroSdTable]:
    """The released tables of Kuzmin 2018, Kuzmin 2020 and Costanzo 2016."""
    base = osp.join(data_root, "data/torchcell")
    kind = "Combined mutant type"
    tables = []
    k2018 = osp.join(base, "dmf_kuzmin2018/raw/aao1729_data_s1.tsv")
    tables.append(
        table_entry(
            "kuzminSystematicAnalysisComplex2018",
            k2018,
            pd.read_csv(k2018, sep="\t", dtype=str),
            "Combined mutant fitness standard deviation",
            kind,
        )
    )
    for name in ("aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx"):
        path = osp.join(base, "dmf_kuzmin2020/raw", name)
        tables.append(
            table_entry(
                "kuzminExploringWholegenomeDuplicate2020",
                path,
                pd.read_excel(path, skiprows=1, dtype=str),
                "Double/triple mutant fitness standard deviation",
                kind,
            )
        )
    sd = "Double mutant fitness standard deviation"
    for name in ("SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"):
        path = osp.join(base, "dmf_costanzo2016/raw", name)
        chunks = [
            chunk
            for chunk in pd.read_csv(
                path, sep="\t", dtype=str, usecols=["P-value", sd], chunksize=2_000_000
            )
        ]
        tables.append(
            table_entry(
                "costanzoGlobalGeneticInteraction2016",
                path,
                pd.concat(chunks, ignore_index=True),
                sd,
                None,
            )
        )
    return tables


def store_entry(data_root: str, slug: str) -> ZeroSdStore:
    """How a fitness dev store holds the released zero SDs."""
    records = 0
    zero: list[dict[str, Any]] = []
    for item in lmdb_records(osp.join(data_root, "data/torchcell", slug)):
        records += 1
        phenotype = item["experiment"]["phenotype"]
        if phenotype["fitness_uncertainty"] == 0:
            zero.append(phenotype)
    return ZeroSdStore(
        slug=slug,
        records=records,
        zero_uncertainty_records=len(zero),
        uncertainty_types=dict(
            Counter(str(p["fitness_uncertainty_type"]) for p in zero)
        ),
        n_samples=dict(Counter(str(p["n_samples"]) for p in zero)),
        fitness_se=dict(Counter(str(p["fitness_se"]) for p in zero)),
        fitness_std=dict(Counter(str(p["fitness_std"]) for p in zero)),
    )


# The SI definitions of each SD column, verbatim (checked against the bytes below).
SD_DEFINITIONS: dict[str, tuple[str, list[str]]] = {
    "kuzminSystematicAnalysisComplex2018": (
        "si/si1.md",
        [
            "12) Combined mutant fitness standard deviation",
            "12. Combined mutant fitness standard deviation",
        ],
    ),
    "kuzminExploringWholegenomeDuplicate2020": (
        "si/si1.md",
        ["12. Combined mutant fitness standard deviation"],
    ),
    "costanzoGlobalGeneticInteraction2016": (
        "si/si1.md",
        ["Double mutant fitness Double mutant fitness standard deviation"],
    ),
    "baryshnikovaQuantitativeAnalysisFitness2010": (
        "si/si1.md",
        [
            "one that reflects the local variability in replicate colonies (based on "
            "$4 -$ 8 colonies for the corresponding double mutant)",
            "variance estimates derived from local double mutant colony "
            "reproducibility can occasionally dramatically underestimate the true "
            "variance",
        ],
    ),
}


def term_lines(path: str) -> dict[str, list[int]]:
    """Line numbers of ``path`` matching each of ``SD_TERMS``, case-insensitively."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().split("\n")
    return {
        term: [
            n
            for n, line in enumerate(lines, start=1)
            if re.search(re.escape(term), line, flags=re.IGNORECASE)
        ]
        for term in SD_TERMS
    }


def si_evidence(data_root: str) -> list[SiEvidence]:
    """Each SI's SD definitions (verbatim, by line); SI and paper lines on SD_TERMS."""
    out = []
    for key, (relative, quotes) in SD_DEFINITIONS.items():
        path = osp.join(data_root, "torchcell-library", key, relative)
        paper = osp.join(data_root, "torchcell-library", key, "paper.md")
        ref = file_ref(path)
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().split("\n")
        definitions: list[VerbatimLine] = []
        for quote in quotes:
            found = [n for n, line in enumerate(lines, start=1) if quote in line]
            if not found:
                raise ValueError(f"{key}: quote not in {path}: {quote!r}")
            definitions.extend(
                VerbatimLine(file=ref, line=n, quote=quote) for n in found
            )
        out.append(
            SiEvidence(
                citation_key=key,
                file=ref,
                definitions=definitions,
                term_lines=term_lines(path),
                paper=file_ref(paper),
                paper_term_lines=term_lines(paper),
            )
        )
    return out


#: The dev stores PR #1054 rebuilds (Additional Data S4), censused with ``--after``.
REBUILT_STORES = ["smf_kuzmin2018", "dmf_kuzmin2018"]
AFTER_NAME = "keep_records_887_sd_zero_stores_after.json"


def store_census(data_root: str, slug: str) -> StoreCensus:
    """Count the records, gaps and uncertainty labels of one rebuilt dev store."""
    root = osp.join(data_root, "data/torchcell", slug)
    with open(
        osp.join(root, "preprocess", "build_manifest.json"), encoding="utf-8"
    ) as h:
        manifest = json.load(h)
    records = 0
    zero = 0
    gaps: Counter[str] = Counter()
    types: Counter[str] = Counter()
    for item in lmdb_records(root):
        records += 1
        phenotype = item["experiment"]["phenotype"]
        for gap in phenotype.get("provenance_gaps") or []:
            gaps[str(gap["field"])] += 1
        types[str(phenotype.get("fitness_uncertainty_type"))] += 1
        zero += phenotype.get("fitness_uncertainty") == 0
    return StoreCensus(
        slug=slug,
        built_at=str(manifest["built_at"]),
        torchcell_commit=str(manifest["torchcell_commit"]),
        records=records,
        gaps_by_field=dict(gaps),
        uncertainty_types=dict(types),
        zero_uncertainty_records=zero,
    )


def main() -> None:
    """Measure, then write the results JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--costanzo-store",
        action="store_true",
        help="also read the 20.7 M-record dmf_costanzo2016 dev store",
    )
    parser.add_argument(
        "--after",
        action="store_true",
        help=f"census the stores PR #1054 rebuilds ({', '.join(REBUILT_STORES)}) "
        f"into {AFTER_NAME} instead of measuring the releases",
    )
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    if args.after:
        os.makedirs(RESULTS, exist_ok=True)
        out = osp.join(RESULTS, AFTER_NAME)
        census = [store_census(data_root, slug) for slug in REBUILT_STORES]
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps([c.model_dump() for c in census], indent=2, sort_keys=True)
            )
            handle.write("\n")
        print(f"wrote {out}")
        return
    slugs = ["dmf_kuzmin2018", "tmf_kuzmin2018", "dmf_kuzmin2020"]
    if args.costanzo_store:
        slugs.append("dmf_costanzo2016")
    inventory = Inventory(
        xue2025=xue_inventory(data_root),
        zero_sd_tables=zero_sd_tables(data_root),
        zero_sd_stores=[store_entry(data_root, slug) for slug in slugs],
        si_evidence=si_evidence(data_root),
    )
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, OUT_NAME)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(inventory.model_dump_json(indent=2))
        handle.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
