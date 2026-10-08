# experiments/036-dataset-fixes-before-kg-build/scripts/choe2019_release_shape.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.choe2019_release_shape]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/choe2019_release_shape
"""Measure what Choe 2019's release gives per sample, and what the schema can carry.

Row 47 of the bacterial schedule (``build_bacteria_candidate_datasets_table.py``,
``name="Choe 2019 genome-reduced ALE"``) carries a written schema need recorded as
"population allele-frequency genotype". This script settles, by measurement rather than
by reading the abstract, three questions:

1. WHAT THE RELEASE GIVES PER SAMPLE. The shape of all seven Supplementary Data
   workbooks and the Source Data workbook: the column semantics of the three variant
   tables (Supplementary Data 2, 3, 4), whether any column is a CLONE rather than a
   POPULATION, and how far a day-62 frequency is from a clone genotype.
2. WHETHER ANY EXISTING PERTURBATION LEAF CAN CARRY IT. Real pydantic construction
   attempts against every candidate leaf for one real variant row, with the exact
   exception each one raises recorded verbatim.
3. WHICH PHENOTYPES ARE STORABLE INDEPENDENTLY OF AN EVOLVED GENOTYPE. Every strain
   Supplementary Table 3 lists gets a writability verdict, and for the arms whose strains
   the schema CAN write, whether the released quantity has a home: the Fig. 2c growth
   rates of the two designed MS56 deletions against ``FitnessPhenotype``, the two Keio
   growth-rate percentages of the Supplementary Fig. 6 legend against the same class, and
   the RNA-seq RPKM against ``RNASeqExpressionPhenotype``'s required ``expression_count``
   (including a back-solve of integer read counts from RPKM times gene length, the
   Caglar 2017 ``back_solve_counts`` method).

CORRECTIONS THIS SCRIPT CARRIES OVER ITS FIRST RUN, each one measured:

- Supplementary Data 2 has 117 variant rows, not 118. The workbook carries a stray
  single-cell row twelve rows past the last call (a lone ``3`` at column J), which a
  row-is-non-empty filter counts as a variant and which also produced a phantom
  ``None`` mutation type. A row is now a variant only when it has BOTH a gene cell and a
  position cell.
- Supplementary Data 3 writes its intergenic calls as a bare ``-`` and Supplementary
  Data 4 as the bare word ``intergenic``, so a ``startswith("intergenic")`` test
  reported 0 intergenic rows for Data 3 when it has several.
- The Keio panel is Supplementary Fig. 6, not Supplementary Fig. 5 (Fig. 5 is the
  ``eMS57mutS+`` construction). The quote is the Fig. 6 legend.
- Supplementary Data 3 and 4 each carry a SECOND worksheet, 44 rows with no header and
  identical in both workbooks, which the first pass never opened.

Every file is sha256-verified against the library mirror's ``manifest.json`` before it is
read. Gene-name resolution runs against the deposited MG1655 GenBank annotation through
``bacteria_common.reconcile_locus_tags``, so the b-number coverage of the two expression
tables is measured, not assumed.

Writes ``results/choe2019_release_shape.json``,
``results/choe2019_variant_rows.csv`` (every row of Supplementary Data 2 with its
blocking reasons) and ``results/choe2019_expression_gene_resolution.csv``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/choe2019_release_shape.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
from collections import Counter
from typing import Any

import openpyxl
import pandas as pd
from dotenv import load_dotenv

from torchcell.datamodels.schema import (
    AlleleEdit,
    AllelePerturbation,
    BacterialBackgroundAllele,
    BacterialDeletionPerturbation,
    BacterialStrainBackground,
    FitnessPhenotype,
    Genotype,
    RNASeqExpressionPhenotype,
    SequenceVariantPerturbation,
)
from torchcell.datasets.bacteria_common import bacterial_genome, reconcile_locus_tags
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

CITATION_KEY = "choeAdaptiveLaboratoryEvolution2019"

#: si<N> -> the Supplementary item it is, from ``si/si3.md`` (MOESM3, "Description of
#: Additional Supplementary Files"), which is the only mapping the publisher releases.
SI_ROLES: dict[str, str] = {
    "si/si1.pdf": "Supplementary Information (Figs. 1-19, Tables 1-3)",
    "si/si2.pdf": "Peer Review File",
    "si/si3.pdf": "Description of Additional Supplementary Files",
    "si/si4.xlsx": "Supplementary Data 1: final readout of phenotype microarray",
    "si/si5.xlsx": "Supplementary Data 2: sequence variations, MS56 ALE",
    "si/si6.xlsx": "Supplementary Data 3: sequence variations, MG1655 ALE",
    "si/si7.xlsx": "Supplementary Data 4: sequence variations, 300 extra generations",
    "si/si8.xlsx": "Supplementary Data 5: peaks detected from ChIP-seq",
    "si/si9.xlsx": "Supplementary Data 6: expression level by RNA-seq (RPKM)",
    "si/si10.xlsx": "Supplementary Data 7: translation level (RPF) by Ribo-seq",
    "si/si11.pdf": "Reporting Summary",
    "si/si12.xlsx": "Source Data",
}


def mirror_root() -> str:
    """The library mirror directory of this citation key."""
    return osp.join(os.environ["DATA_ROOT"], "torchcell-library", CITATION_KEY)


def sha256_of(path: str) -> str:
    """sha256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_mirror() -> dict[str, dict[str, Any]]:
    """sha256-verify every file this script reads against the mirror manifest."""
    root = mirror_root()
    with open(osp.join(root, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    recorded = {record["path"]: record for record in manifest["files"]}
    verified: dict[str, dict[str, Any]] = {}
    for rel in [*SI_ROLES, "paper.md", "si/si1.md", "si/si2.md", "si/si3.md"]:
        record = recorded[rel]
        measured = sha256_of(osp.join(root, rel))
        if measured != record["sha256"]:
            raise ValueError(f"{rel}: sha256 {measured} != manifest {record['sha256']}")
        verified[rel] = {
            "role": SI_ROLES.get(rel, record["role"]),
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "source_url": record.get("source"),
        }
    return verified


def _one_line(error: Exception) -> str:
    """A pydantic error as one line, so the JSON stays readable."""
    return " | ".join(line.strip() for line in str(error).splitlines() if line.strip())


def sheet_rows(path: str, sheet: str | None = None) -> list[list[Any]]:
    """Every row of one worksheet as a list of lists (first sheet when unnamed)."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook[sheet] if sheet is not None else workbook.worksheets[0]
    rows = [list(row) for row in worksheet.iter_rows(values_only=True)]
    workbook.close()
    return rows


#: The markers the release uses for a cell that carries no number: an empty cell, the
#: dash of the "AA change" column for a non-coding call, and the ``NA`` of the RNA-seq
#: p-value column (measured: 50 of 3,457 rows).
MISSING_MARKERS: frozenset[Any] = frozenset({None, "", "-", "NA"})


def as_float(value: Any) -> float | None:
    """A cell as a float, or None for one of the release's missing markers."""
    if value in MISSING_MARKERS:
        return None
    return float(value)


# --------------------------------------------------------------------------- #
# 1. What the release gives per sample
# --------------------------------------------------------------------------- #
def measure_phenotype_microarray(root: str) -> dict[str, Any]:
    """Supplementary Data 1: the Biolog PM endpoint, per (plate, assay, nutrient)."""
    rows = sheet_rows(osp.join(root, "si/si4.xlsx"))
    columns = [rows[2][i] for i in range(3, 7)]
    body = [row for row in rows[3:] if row[0] is not None]
    plates = Counter(row[0] for row in body)
    assays = Counter(row[1] for row in body)
    keys = Counter((row[0], row[1], row[2]) for row in body)
    return {
        "sheet": "PM1",
        "strain_columns": columns,
        "nutrient_rows": len(body),
        "plates": dict(sorted(plates.items())),
        "assays": dict(sorted(assays.items())),
        "duplicate_plate_assay_nutrient_keys": [
            {"key": list(key), "n": n} for key, n in keys.items() if n > 1
        ],
        "distinct_nutrients": len({row[2] for row in body}),
        "mg1655_columns": [c for c in columns if str(c).startswith("MG")],
        "evolved_columns": [c for c in columns if str(c).startswith("eMS")],
        "non_numeric_cells": sum(
            1 for row in body for cell in row[3:7] if as_float(cell) is None
        ),
    }


#: A gene cell that names no locus. Supplementary Data 2 numbers its intergenic calls,
#: Supplementary Data 3 writes a bare dash and Supplementary Data 4 the bare word, so one
#: test has to cover all three spellings (the first pass tested only the first).
INTERGENIC_MARKERS: frozenset[str] = frozenset({"intergenic", "-"})


def is_intergenic(gene: str) -> bool:
    """True when a variant row's gene cell names no locus."""
    return gene.startswith("intergenic") or gene in INTERGENIC_MARKERS


def measure_variant_table(
    root: str, rel: str, *, header_rows: int, label: str
) -> dict[str, Any]:
    """One of the three variant tables: columns, row shapes, and frequency semantics.

    A variant row is one with BOTH a gene cell and a position cell. Filtering on
    "any cell is non-empty" instead counts the stray single-cell rows the workbooks
    carry below their tables: Supplementary Data 2 holds a lone ``3`` twelve rows past
    its last call, which is what made the first pass report 118 rows and a phantom
    ``None`` mutation type.
    """
    rows = sheet_rows(osp.join(root, rel))
    header = rows[1:header_rows]
    gene_column = 0 if rel == "si/si5.xlsx" else 1
    position_column = 1 if gene_column == 0 else 0
    type_column = 2 if rel == "si/si5.xlsx" else 4
    aa_column = 5
    body = [
        row
        for row in rows[header_rows:]
        if row[gene_column] is not None and row[position_column] is not None
    ]
    stray = [
        index
        for index, row in enumerate(rows[header_rows:], start=header_rows)
        if any(c is not None for c in row)
        and (row[gene_column] is None or row[position_column] is None)
    ]
    genes = [str(row[gene_column]) for row in body]
    types = Counter(str(row[type_column]) for row in body)
    with_aa = sum(1 for row in body if row[aa_column] not in (None, "", "-"))
    frequency_header = [c for c in (header[-1] if header else []) if c is not None]
    per_locus = Counter(gene for gene in genes if not is_intergenic(gene))
    return {
        "file": rel,
        "label": label,
        "header_rows": [[str(c) if c is not None else "" for c in r] for r in header],
        "variant_rows": len(body),
        "stray_rows_excluded": stray,
        "distinct_genes": len(set(genes)),
        "intergenic_rows": sum(1 for gene in genes if is_intergenic(gene)),
        "intergenic_spellings": sorted({gene for gene in genes if is_intergenic(gene)}),
        "mutation_type_histogram": dict(sorted(types.items())),
        "rows_with_amino_acid_change": with_aa,
        "loci_with_more_than_one_call": {
            gene: n for gene, n in sorted(per_locus.items()) if n > 1
        },
        "frequency_column_labels": [str(c) for c in frequency_header],
        "frequency_columns": len(frequency_header),
        "worksheets": measure_extra_worksheets(osp.join(root, rel)),
    }


def measure_extra_worksheets(path: str) -> dict[str, Any]:
    """Every worksheet of a variant workbook, including the ones with no header.

    Supplementary Data 3 and 4 each carry a second sheet of 44 unlabelled rows. Its
    content is measured rather than assumed: it is a working sheet, identical in both
    workbooks, with no header row of its own, so it is not an independent release.
    """
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out: dict[str, Any] = {}
    for worksheet in workbook.worksheets:
        rows = [list(row) for row in worksheet.iter_rows(values_only=True)]
        populated = [r for r in rows if any(c is not None for c in r)]
        out[worksheet.title] = {
            "populated_rows": len(populated),
            "max_column": worksheet.max_column,
            "first_row": [str(c) if c is not None else "" for c in rows[0]]
            if rows
            else [],
        }
    workbook.close()
    return out


def measure_day62_thresholding(root: str) -> dict[str, Any]:
    """How far the last ALE timepoint of Supplementary Data 2 is from a clone genotype.

    The flagship evolved strain eMS57 has no variant list of its own in the release, so
    the only route to a per-clone genotype is thresholding the final population
    frequency. This measures what that call would cost at several thresholds.
    """
    rows = sheet_rows(osp.join(root, "si/si5.xlsx"))
    days = [str(c) for c in rows[2][6:26]]
    body = [row for row in rows[3:] if row[0] is not None]
    final = [as_float(row[25]) for row in body]
    if any(value is None for value in final):
        raise ValueError("Supplementary Data 2 day-62 column has a non-numeric cell")
    values = [value for value in final if value is not None]
    return {
        "timepoint_labels_days": days,
        "timepoints": len(days),
        "last_timepoint_day": days[-1],
        "variant_rows": len(values),
        "rows_at_100_percent_on_last_day": sum(1 for v in values if v == 100.0),
        "rows_above_50_percent_on_last_day": sum(1 for v in values if v > 50.0),
        "rows_between_5_and_95_percent_on_last_day": sum(
            1 for v in values if 5.0 <= v <= 95.0
        ),
        "rows_at_zero_on_last_day": sum(1 for v in values if v == 0.0),
        "distinct_nonzero_frequencies_on_last_day": len(
            {v for v in values if v != 0.0}
        ),
    }


def measure_expression_table(
    root: str, rel: str, *, value_columns: list[tuple[str, int]]
) -> dict[str, Any]:
    """One expression table: gene count, sample columns, and value ranges."""
    rows = sheet_rows(osp.join(root, rel))
    body = [row for row in rows[3:] if row[0] is not None]
    genes = [str(row[0]) for row in body]
    per_column: dict[str, Any] = {}
    for name, index in value_columns:
        values = [as_float(row[index]) for row in body]
        present = [v for v in values if v is not None]
        per_column[name] = {
            "values": len(present),
            "missing": len(values) - len(present),
            "zeros": sum(1 for v in present if v == 0.0),
            "min": min(present),
            "max": max(present),
            "sum": sum(present),
        }
    return {
        "file": rel,
        "gene_rows": len(body),
        "distinct_gene_labels": len(set(genes)),
        "duplicate_gene_labels": [name for name, n in Counter(genes).items() if n > 1],
        "columns": per_column,
    }


def measure_source_data(root: str) -> dict[str, Any]:
    """The Source Data workbook: which figure panels release numbers, and for what."""
    path = osp.join(root, "si/si12.xlsx")
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets: dict[str, Any] = {}
    for worksheet in workbook.worksheets:
        rows = [list(row) for row in worksheet.iter_rows(values_only=True)]
        populated = [r for r in rows if any(c is not None for c in r)]
        sheets[worksheet.title] = {
            "title_cell": str(rows[0][1]) if len(rows[0]) > 1 else None,
            "populated_rows": len(populated),
            "max_column": worksheet.max_column,
            "has_numbers": any(
                isinstance(cell, (int, float))
                for row in rows
                for cell in row
                if cell is not None
            ),
        }
    workbook.close()
    return sheets


def measure_growth_rate_panels(root: str) -> dict[str, Any]:
    """The two Source Data panels that release a per-strain growth rate."""
    fig2c = sheet_rows(osp.join(root, "si/si12.xlsx"), "Fig2c")
    fig2g = sheet_rows(osp.join(root, "si/si12.xlsx"), "Fig2g")
    strains_2c = [
        {"strain": str(row[1]), "replicates": [as_float(c) for c in row[2:5]]}
        for row in fig2c[3:]
        if row[1] is not None
    ]
    rows_2g = [
        {
            "media": str(row[1]) if row[1] is not None else None,
            "strain": str(row[2]),
            "replicates": [as_float(c) for c in row[3:6]],
        }
        for row in fig2g[3:]
        if row[2] is not None
    ]
    return {
        "fig2c_title": str(fig2c[0][1]),
        "fig2c_strains": strains_2c,
        "fig2g_title": str(fig2g[0][1]),
        "fig2g_rows": rows_2g,
    }


# --------------------------------------------------------------------------- #
# 2. Whether any existing leaf can carry a called variant
# --------------------------------------------------------------------------- #
def probe_perturbation_leaves(root: str) -> dict[str, Any]:
    """Try to write ONE real Supplementary Data 2 row with every candidate leaf.

    The row is the ``ilvH`` G->T SNV at 82,242 (Gly21Cys), which reaches 100% in the
    population on day 27. It is the easiest case the table has: a coding SNV with a named
    gene and a stated amino-acid change. Every attempt is made for real and its exception
    recorded verbatim; nothing here is asserted from reading the class.
    """
    rows = sheet_rows(osp.join(root, "si/si5.xlsx"))
    row = next(r for r in rows[3:] if r[0] == "ilvH")
    released = {
        "gene": row[0],
        "position": row[1],
        "type": row[2],
        "ref": row[3],
        "allele": row[4],
        "aa_change": row[5],
        "allelic_frequency_percent_by_day": {
            str(rows[2][i]): as_float(row[i]) for i in range(6, 26)
        },
    }
    genome = bacterial_genome("ecoli", "MG1655")
    resolution = genome.resolve_gene_name(str(row[0]))
    locus_tag = resolution.systematic_name
    attempts: dict[str, Any] = {}

    def attempt(name: str, build: Any) -> None:
        try:
            model = build()
        except Exception as error:  # measurement: the message IS the finding
            attempts[name] = {
                "constructed": False,
                "error_type": type(error).__name__,
                "error": " | ".join(
                    line.strip() for line in str(error).splitlines() if line.strip()
                ),
            }
            return
        attempts[name] = {
            "constructed": True,
            "fields_without_a_home": sorted(
                set(released) - set(model.model_dump().keys())
            ),
        }

    attempt(
        "SequenceVariantPerturbation(gene_symbol)",
        lambda: SequenceVariantPerturbation(
            systematic_gene_name=str(row[0]),
            perturbed_gene_name=str(row[0]),
            strain_id="eMS57",
        ),
    )
    attempt(
        "SequenceVariantPerturbation(b_number)",
        lambda: SequenceVariantPerturbation(
            systematic_gene_name=str(locus_tag),
            perturbed_gene_name=str(row[0]),
            strain_id="eMS57",
        ),
    )
    attempt(
        "AllelePerturbation(b_number)",
        lambda: AllelePerturbation(
            systematic_gene_name=str(locus_tag), perturbed_gene_name=str(row[0])
        ),
    )
    attempt(
        "BacterialDeletionPerturbation(b_number)",
        lambda: BacterialDeletionPerturbation(
            systematic_gene_name=str(locus_tag),
            perturbed_gene_name=str(row[0]),
            gene_namespace="ecoli_k12_mg1655_bnumber",
        ),
    )
    attempt(
        "BacterialBackgroundAllele(functional_unknown)",
        lambda: BacterialBackgroundAllele(
            systematic_gene_name=str(locus_tag),
            gene_namespace="ecoli_k12_mg1655_bnumber",
            gene_name=str(row[0]),
            allele_name=f"{row[0]}({row[3]}{row[1]}{row[4]})",
            edit=AlleleEdit.sequence_variant,
            functional=None,  # type: ignore[arg-type]  # measurement: the refusal IS the finding
        ),
    )
    # Two calls in one gene within one population, which a background refuses.
    multi = [r for r in rows[3:] if r[0] is not None]
    gene_counts = Counter(str(r[0]) for r in multi)
    repeated = {g: n for g, n in gene_counts.items() if n > 1}
    attempt(
        "BacterialStrainBackground(two_alleles_one_locus)",
        lambda: BacterialStrainBackground(
            name="eMS57",
            reference_strain="MG1655",
            assembly_set="ecoli_K12_MG1655_ASM584v2",
            alleles=[
                BacterialBackgroundAllele(
                    systematic_gene_name=str(locus_tag),
                    gene_namespace="ecoli_k12_mg1655_bnumber",
                    gene_name=str(row[0]),
                    allele_name=f"{row[0]}_call_{i}",
                    edit=AlleleEdit.sequence_variant,
                    functional=False,
                    provenance_gaps=None,  # type: ignore[arg-type]  # measurement: the refusal IS the finding
                )
                for i in (1, 2)
            ],
            provenance_gaps=None,  # type: ignore[arg-type]  # measurement: the refusal IS the finding
        ),
    )
    del genome
    return {
        "probe_row": released,
        "gene_symbol_resolves_to": locus_tag,
        "attempts": attempts,
        "genes_with_more_than_one_call_in_supplementary_data_2": dict(
            sorted(repeated.items())
        ),
        "genotype_equality_rule": (
            "Genotype.__eq__ compares set(perturbations), so an evolved clone written "
            "with its parent's perturbations only is genotype-identical to its parent"
        ),
    }


# --------------------------------------------------------------------------- #
# 3. Which phenotypes are storable without an evolved genotype
# --------------------------------------------------------------------------- #
def gene_spans(genome: Any) -> dict[str, int]:
    """Locus tag -> gene-feature span in bp, from the deposited GenBank annotation."""
    spans: dict[str, int] = {}
    for feature in genome.db.features_of_type("gene"):
        tags = feature.attributes.get("locus_tag") or []
        if len(tags) != 1:
            continue
        spans[tags[0]] = int(feature.end) - int(feature.start) + 1
    return spans


def measure_rnaseq_loadability(root: str, results_dir: str) -> dict[str, Any]:
    """Can the wild-type MG1655 RNA-seq columns be written as an RNA-seq phenotype?

    ``RNASeqExpressionPhenotype`` requires BOTH ``expression_tpm`` and a per-gene
    integer ``expression_count``. Choe releases RPKM only. Two things are measured: the
    gene-label coverage against the MG1655 annotation, and whether the integer read count
    back-solves from ``RPKM x gene_length_kb x (N / 1e6)`` for a single scale factor, the
    way Caglar 2017's counts back-solved out of its VST table.
    """
    rows = sheet_rows(osp.join(root, "si/si9.xlsx"))
    body = [row for row in rows[3:] if row[0] is not None]
    labels = pd.Series([str(row[0]) for row in body], name="gene")
    genome = bacterial_genome("ecoli", "MG1655")
    stored, report = reconcile_locus_tags(
        genome, labels, label="choe2019 Supplementary Data 6 gene labels"
    )
    spans = gene_spans(genome)
    del genome
    frame = pd.DataFrame(
        {
            "released_label": labels,
            "stored_name": stored,
            "mg1655_1_rpkm": [as_float(row[1]) for row in body],
            "mg1655_2_rpkm": [as_float(row[2]) for row in body],
        }
    )
    frame["is_b_number"] = frame["stored_name"].str.fullmatch(r"b\d{4}")
    frame["gene_span_bp"] = frame["stored_name"].map(spans)
    frame.to_csv(
        osp.join(results_dir, "choe2019_expression_gene_resolution.csv"), index=False
    )

    resolved = frame[frame["is_b_number"] & frame["gene_span_bp"].notna()]
    # Back-solve: count_i = rpkm_i * span_kb_i * k. The smallest nonzero product is the
    # candidate unit (count == 1); test every integer multiple hypothesis up to 8.
    products = (
        resolved["mg1655_1_rpkm"] * resolved["gene_span_bp"] / 1000.0
    ).to_numpy()
    positive = sorted(float(v) for v in products if v > 0)
    back_solve: list[dict[str, Any]] = []
    for assumed_unit_count in range(1, 9):
        scale = assumed_unit_count / positive[0]
        residuals = [abs(v * scale - round(v * scale)) for v in positive]
        back_solve.append(
            {
                "assumed_count_of_smallest_product": assumed_unit_count,
                "implied_total_mapped_reads_millions": scale,
                "max_integer_residual": max(residuals),
                "fraction_within_0.01_of_an_integer": sum(
                    1 for r in residuals if r < 0.01
                )
                / len(residuals),
            }
        )

    tpm_sum = float(resolved["mg1655_1_rpkm"].sum())
    probe: dict[str, Any] = {}
    try:
        RNASeqExpressionPhenotype(expression_tpm={"b0001": 1.0}, expression_count={})
    except Exception as error:
        probe["empty_expression_count"] = _one_line(error)
    try:
        RNASeqExpressionPhenotype(expression_tpm={"b0001": 1.0})  # type: ignore[call-arg]
    except Exception as error:
        probe["omitted_expression_count"] = _one_line(error)
    try:
        RNASeqExpressionPhenotype(
            expression_tpm={"b0001": 1.0},
            expression_count={"b0001": 1.5},  # type: ignore[dict-item]
        )
    except Exception as error:
        probe["non_integer_expression_count"] = _one_line(error)

    return {
        "gene_rows": len(frame),
        "resolution_status_histogram": {
            status.value: n for status, n in report.status_histogram.items()
        },
        "resolution_layer_histogram": report.layer_histogram,
        "resolved_to_b_number": int(frame["is_b_number"].sum()),
        "unresolved_labels": sorted(
            frame.loc[~frame["is_b_number"], "released_label"].tolist()
        ),
        "b_numbers_with_a_gene_span": int(resolved.shape[0]),
        "rpkm_sum_mg1655_1": tpm_sum,
        "rpkm_sum_is_one_million": math.isclose(tpm_sum, 1e6, rel_tol=1e-3),
        "count_back_solve": back_solve,
        "expression_count_probe": probe,
    }


def measure_keio_fitness_arm(root: str) -> dict[str, Any]:
    """The BW25113 Keio single-deletion panel of Supplementary Fig. 6.

    (The first pass called this Supplementary Fig. 5, which is the ``eMS57mutS+``
    construction panel instead.) The only quantitative per-strain numbers the panel
    releases are the two growth-rate percentages its legend states. Both are built as
    real ``FitnessPhenotype`` records and both deletions as real
    ``BacterialDeletionPerturbation``s on the BW25113 namespace, so the claim that this
    arm is writable is a construction, not a reading.
    """
    genome = bacterial_genome("ecoli", "BW25113")
    records: dict[str, Any] = {}
    for symbol, percent in (("ydaS", 72.4), ("abgR", 69.8)):
        resolution = genome.resolve_gene_name(symbol)
        perturbation = BacterialDeletionPerturbation(
            systematic_gene_name=str(resolution.systematic_name),
            perturbed_gene_name=symbol,
            gene_namespace="ecoli_k12_bw25113_locus_tag",
            collection="single gene knockout collection (the Keio collection)",
        )
        phenotype = FitnessPhenotype(
            fitness=percent / 100.0, n_samples=None, sample_unit=None
        )
        records[symbol] = {
            "resolution_status": resolution.status.value,
            "stored_locus_tag": resolution.systematic_name,
            "genotype_len": len(Genotype(perturbations=[perturbation])),
            "fitness": phenotype.fitness,
        }
    del genome
    return {
        "quantified_strains": records,
        "panel_strains_total": 62,
        "panel_strains_with_a_released_number": len(records),
    }


# --------------------------------------------------------------------------- #
# 4. Per-strain writability, and the two growth panels
# --------------------------------------------------------------------------- #
#: Supplementary Table 3's strain rows, verbatim, with this measurement's verdict on
#: whether the schema can write the strain's genotype and why.
STRAIN_TABLE: tuple[dict[str, str], ...] = (
    {
        "strain": "MG1655",
        "genotype": "Laboratory E. coli, train K-12, substr. MG1655",
        "verdict": "writable",
        "why": "the reference strain; an unperturbed record is Genotype(perturbations=[])",
    },
    {
        "strain": "eMG1655",
        "genotype": "E. coli MG1655 adaptively evolved in M9glucose medium",
        "verdict": "blocked",
        "why": "evolved clone; Supplementary Data 3 gives per-replicate POPULATION "
        "allele frequencies, never a clone genotype",
    },
    {
        "strain": "MS56",
        "genotype": "E. coli MG1655 with large deletions MD1 toMD56",
        "verdict": "background only",
        "why": "writable as a BacterialStrainBackground with a verbatim "
        "genotype_statement and NO typed alleles, because this paper enumerates none of "
        "the 55 regions and defers them to its reference 4; not writable as a "
        "perturbation set, so an MG1655-referenced MS56 record would assert that MS56 is "
        "genotypically MG1655",
    },
    {
        "strain": "eMS57",
        "genotype": "E. coli MS56 adaptively evolved in M9 glucosemedium",
        "verdict": "blocked",
        "why": "evolved clone with NO variant list of its own; a genotype needs a "
        "day-62 frequency threshold the authors never applied",
    },
    {
        "strain": "eMS57mutS+",
        "genotype": "eMS57, puuP::mutS-kan",
        "verdict": "blocked",
        "why": "the puuP::mutS-kan knock-in is writable, but its parent eMS57 is not",
    },
    {
        "strain": "MS56 421 kb",
        "genotype": "MS56, (hycEDCBA-hypABCDE-fhlA-ygbA-mutS-pphB-ygbIJKLMN-rpoS)::kan",
        "verdict": "writable",
        "why": "21 named genes, each a BacterialDeletionPerturbation on the MG1655 "
        "b-number namespace with cassette 'kan', on the MS56 background",
    },
    {
        "strain": "MS56 drpoS",
        "genotype": "MS56, rpoS::kan",
        "verdict": "writable",
        "why": "one named gene, one BacterialDeletionPerturbation, on the MS56 "
        "background",
    },
    {
        "strain": "cspCWT / cspCmut",
        "genotype": "MS56, cspC::cspC-kan / MS56, cspC::cspC(G37A)-kanR",
        "verdict": "blocked",
        "why": "a DESIGNED single sequence variant with the base change released; the "
        "perturbation axis has no bacterial sequence-variant leaf, and "
        "BacterialBackgroundAllele needs a functional flag the paper does not determine",
    },
    {
        "strain": "ilvNWT / ilvNmut",
        "genotype": "MS56, ilvN::ilvN-kan / MS56, ilvN::ilvN(C202T)-kanR",
        "verdict": "blocked",
        "why": "same gap as cspC; the paper calls the allele a possible hitchhiker, so "
        "even its functional consequence is unresolved",
    },
    {
        "strain": "yifBWT / yifBmut",
        "genotype": "MS56, yifB::yifB-kan / MS56, yifB::yifB(1169C insertion)-kanR",
        "verdict": "blocked",
        "why": "same gap as cspC; a frameshift insertion at a stated position with no "
        "leaf to hold it",
    },
)


def measure_growth_panels(root: str) -> dict[str, Any]:
    """The Fig. 2c rates and the Fig. 2f ratios, and whether the two panels agree.

    Both panels name the strain pair eMS57/MS56, so a reader may take them for the same
    measurement. Measured here: they differ by a factor near 2, which is why nothing may
    be derived across them. Fig. 2c instead reproduces the paper's own prose claim that
    the two deletion strains recover to 80% of eMS57.
    """
    fig2c = sheet_rows(osp.join(root, "si/si12.xlsx"), "Fig2c")
    fig2f = sheet_rows(osp.join(root, "si/si12.xlsx"), "Fig2f")
    rates = {
        str(row[1]): [float(cell) for cell in row[2:5]]
        for row in fig2c[3:]
        if row[1] is not None
    }
    ratios = {
        str(row[1]): [float(cell) for cell in row[2:5]]
        for row in fig2f[3:]
        if row[1] is not None
    }
    means = {name: sum(values) / len(values) for name, values in rates.items()}
    ratio_means = {name: sum(v) / len(v) for name, v in ratios.items()}
    from_rates = means["eMS57"] / means["MS56"]
    released = ratio_means["eMS57/MS56"]
    return {
        "fig2c_title": str(fig2c[0][1]),
        "fig2c_replicates": rates,
        "fig2c_means": means,
        "fig2c_recovery_fraction_of_evolved": {
            name: means[name] / means["eMS57"] for name in ("Δ21 kb", "ΔrpoS")
        },
        "stated_recovery_fraction_of_evolved": 0.80,
        "fig2c_parent_over_wild_type": means["MS56"] / means["MG1655"],
        "fig2f_title": str(fig2f[0][1]),
        "fig2f_replicates": ratios,
        "fig2f_means": ratio_means,
        "evolved_over_parent_from_fig2c_rates": from_rates,
        "evolved_over_parent_released_by_fig2f": released,
        "panel_disagreement_factor": from_rates / released,
        "note": "the Fig. 2c sheet releases no unit cell; the records store a "
        "dimensionless ratio, so no unit is asserted",
    }


def measure_deleted_region(root: str) -> dict[str, Any]:
    """The 21 named genes against the pinned MG1655 annotation and the released span.

    Supplementary Table 3 names the genes, the Results name a 21 kb interval. Measured:
    the named genes are a contiguous run with no unlisted gene inside it, and the
    released interval is MS56's own coordinate system, not MG1655's.
    """
    del root
    symbols = [
        "hycE",
        "hycD",
        "hycC",
        "hycB",
        "hycA",
        "hypA",
        "hypB",
        "hypC",
        "hypD",
        "hypE",
        "fhlA",
        "ygbA",
        "mutS",
        "pphB",
        "ygbI",
        "ygbJ",
        "ygbK",
        "ygbL",
        "ygbM",
        "ygbN",
        "rpoS",
    ]
    genome = bacterial_genome("ecoli", "MG1655")
    tags = [str(genome.resolve_gene_name(symbol).systematic_name) for symbol in symbols]
    spans: dict[str, tuple[int, int]] = {}
    for feature in genome.db.features_of_type("gene"):
        locus = feature.attributes.get("locus_tag") or []
        if len(locus) == 1:
            spans[locus[0]] = (int(feature.start), int(feature.end))
    low = min(spans[tag][0] for tag in tags)
    high = max(spans[tag][1] for tag in tags)
    inside = sorted(
        tag for tag, (start, end) in spans.items() if start >= low and end <= high
    )
    del genome
    return {
        "named_symbols": symbols,
        "locus_tags": tags,
        "contiguous": tags == sorted(tags),
        "mg1655_window": [low, high],
        "mg1655_window_bp": high - low + 1,
        "genes_fully_inside_window": len(inside),
        "unlisted_genes_inside_window": [t for t in inside if t not in set(tags)],
        "released_ms56_window": [2038496, 2059460],
        "released_ms56_window_bp": 2059460 - 2038496 + 1,
        "ms56_to_mg1655_offset_bp": low - 2038496,
        "finding": "the released coordinates are MS56 coordinates, so no GenomicSpan "
        "against the pinned MG1655 assembly can carry them; the gene-keyed deletions "
        "are what a record can state",
    }


def probe_background_carriers() -> dict[str, Any]:
    """Can MS56 be stated at all, and can its unenumerated content be a typed gap?

    Both answers are constructed, not read: the background builds with an EMPTY allele
    list plus the verbatim genotype statement (the shape Girgis 2009 already uses), and a
    ``ProvenanceGap`` on ``alleles`` is REFUSED because the field defaults to ``[]``
    rather than ``None``. That refusal is why the absence lives in a ledger.
    """
    quote = (
        "MS56</td><td rowspan=1 colspan=1>E. coli MG1655 with large deletions MD1 "
        "toMD56</td><td rowspan=1 colspan=1>4</td>"
    )
    sourced = SourcedValue(
        value="MS56",
        provenance=Provenance(
            source_uri="si/si1.md",
            citation_key=CITATION_KEY,
            sha256="fcadf7faf5805d6a61e557cbf8440f17279664a96d81bf6cbec630a991a957bc",
        ),
        quote=quote,
    )
    out: dict[str, Any] = {}
    background = BacterialStrainBackground(
        name="MS56",
        reference_strain="MG1655",
        assembly_set="ecoli_K12_MG1655_ASM584v2",
        parents=["MG1655"],
        genotype_statement="E. coli MG1655 with large deletions MD1 toMD56",
        alleles=[],
        provenance=[sourced],
    )
    out["background_with_no_typed_alleles"] = {
        "constructed": True,
        "alleles": len(background.alleles),
        "genotype_statement": background.genotype_statement,
        "is_fully_sourced": background.is_fully_sourced,
    }
    try:
        BacterialStrainBackground(
            name="MS56",
            reference_strain="MG1655",
            assembly_set="ecoli_K12_MG1655_ASM584v2",
            genotype_statement="E. coli MG1655 with large deletions MD1 toMD56",
            provenance=[sourced],
            provenance_gaps=[
                ProvenanceGap(
                    field="alleles",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="the 55 deleted regions are never enumerated",
                )
            ],
        )
    except Exception as error:
        out["typed_gap_on_alleles"] = {
            "constructed": False,
            "error_type": type(error).__name__,
            "error": _one_line(error),
            "consequence": "the unenumerated 1.1 Mbp has no typed home; it is recorded "
            "in the loader's preprocess/genotype_gaps.json and in the note",
        }
    return out


def main() -> None:
    """Run every measurement and write the results beside this script."""
    load_dotenv()
    root = mirror_root()
    results_dir = osp.join(
        os.environ["EXPERIMENT_ROOT"], "036-dataset-fixes-before-kg-build", "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    verified = verify_mirror()
    variant_tables = [
        measure_variant_table(
            root, "si/si5.xlsx", header_rows=3, label="Supplementary Data 2 (MS56 ALE)"
        ),
        measure_variant_table(
            root,
            "si/si6.xlsx",
            header_rows=3,
            label="Supplementary Data 3 (MG1655 ALE)",
        ),
        measure_variant_table(
            root,
            "si/si7.xlsx",
            header_rows=4,
            label="Supplementary Data 4 (300 extra generations)",
        ),
    ]
    chip = sheet_rows(osp.join(root, "si/si8.xlsx"))
    results: dict[str, Any] = {
        "citation_key": CITATION_KEY,
        "doi": "10.1038/s41467-019-08888-6",
        "verified_files": verified,
        "phenotype_microarray": measure_phenotype_microarray(root),
        "variant_tables": variant_tables,
        "day62_thresholding": measure_day62_thresholding(root),
        "chip_seq_peaks": len([r for r in chip[2:] if r[0] is not None]),
        "rnaseq_table": measure_expression_table(
            root,
            "si/si9.xlsx",
            value_columns=[
                ("MG1655_1", 1),
                ("MG1655_2", 2),
                ("eMG1655_1", 3),
                ("eMG1655_2", 4),
                ("eMS57_1", 5),
                ("eMS57_2", 6),
                ("p_value_eMS57_over_MG1655", 7),
            ],
        ),
        "riboseq_table": measure_expression_table(
            root,
            "si/si10.xlsx",
            value_columns=[
                ("RPF_MG1655_1", 1),
                ("RPF_MG1655_2", 2),
                ("RPF_eMS57_1", 3),
                ("RPF_eMS57_2", 4),
                ("TE_MG1655", 5),
                ("TE_eMS57", 6),
            ],
        ),
        "source_data_sheets": measure_source_data(root),
        "growth_rate_panels": measure_growth_rate_panels(root),
        "growth_panels": measure_growth_panels(root),
        "deleted_region": measure_deleted_region(root),
        "strain_writability": list(STRAIN_TABLE),
        "background_carriers": probe_background_carriers(),
        "perturbation_leaf_probe": probe_perturbation_leaves(root),
        "rnaseq_loadability": measure_rnaseq_loadability(root, results_dir),
        "keio_fitness_arm": measure_keio_fitness_arm(root),
    }

    rows = sheet_rows(osp.join(root, "si/si5.xlsx"))
    days = [str(c) for c in rows[2][6:26]]
    ledger = []
    for row in rows[3:]:
        if row[0] is None or row[1] is None:
            continue
        gene = str(row[0])
        intergenic = is_intergenic(gene)
        reasons = ["population_allele_frequency_not_a_clone_genotype"]
        if intergenic:
            reasons.append("intergenic_call_has_no_locus_to_key_to")
        else:
            reasons.append("no_leaf_holds_position_ref_alt_aa_change_frequency")
        if row[5] in (None, "", "-"):
            reasons.append("functional_consequence_unknown")
        ledger.append(
            {
                "gene": gene,
                "position": row[1],
                "type": row[2],
                "ref": row[3],
                "allele": row[4],
                "aa_change": row[5],
                "max_frequency_percent": max(
                    (as_float(row[i]) or 0.0) for i in range(6, 26)
                ),
                "day62_frequency_percent": as_float(row[25]),
                "timepoints": len(days),
                "blocking_reasons": ";".join(reasons),
            }
        )
    pd.DataFrame(ledger).to_csv(
        osp.join(results_dir, "choe2019_variant_rows.csv"), index=False
    )
    results["variant_row_ledger_rows"] = len(ledger)

    out = osp.join(results_dir, "choe2019_release_shape.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
