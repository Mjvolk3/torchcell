# torchcell/datasets/ecoli/li2024
# [[torchcell.datasets.ecoli.li2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/li2024
# Test file: tests/torchcell/datasets/ecoli/test_li2024.py
"""D2Cell 2026: a provenance record of a secondary source this ontology does not admit.

Li et al. (bioRxiv 2024, doi:10.1101/2024.09.09.612023; published in Trends in
Biotechnology 2026; citation key ``liLeveragingLargeLanguage2024``) is row 60 of the
ranked bacterial candidate table, ``status="aggregation"``: a database of metabolic
engineering entries that a large language model extracted from published abstracts and
open-access full texts, plus the binary training set of the D2Cell-pred target model.

DECISION: NOT LOADED. This module registers no dataset. Unlike MCF2Chem 2023
(:mod:`torchcell.datasets.ecoli.cai2023`) the release IS a per-record artifact with a
per-record source DOI, so the row is settled by measuring what those records are, on
sha256-pinned bytes deposited in ``$DATA_ROOT/torchcell-raw/<key>/``.

**1. The database values are LLM transcriptions, not measurements.** Every row of
``Cell_factory_dataset_Qwen_110b.xlsx`` is the output of the Qwen1.5-110B relation
extraction step (:data:`RE_MODEL`) run over abstracts and full texts
(:data:`CORPUS`). The DOI names the paper a value was read FROM; no row carries the
sentence it was read from, so a stored number cannot be audited without mirroring the
source paper and re-reading it, which is the curation this ontology would then be doing
itself. The paper's own benchmark puts the NER precision at 85% (:data:`NER_PRECISION`),
and :func:`measure_database` finds the field misalignment that number predicts:
``parent strain`` cells holding a temperature, placeholders ("Not specified", "See
note", "Assumed similar") in condition fields, and Chinese text inside English fields.

**2. The training set is a label construction, not an observation.** The D2Cell-pred
E. coli split files carry a binary ``inf_label_01`` whose negatives are partly
designated by rule (:data:`NEGATIVE_DESIGNATION`) and partly FSEOF simulations
(:data:`SIMULATED_NEGATIVES`), so a ``0`` is not a measured null.
:func:`measure_training_split` counts the ``real data`` flag and the ``doi list``
column; the flag marks far fewer rows than the paper's 8,134 experimental entries.

**3. Four required titer fields have no honest value** (:data:`SCHEMA_BLOCKERS`): no
reference (parent-strain) titer column, free-text gene names with no locus or allele, a
``titer unit`` column that mixes concentrations with yields and percentages, and a
one-condition-per-row environment written as free text.

What a later curation could use is the per-record DOI: :func:`ecoli_source_dois` is the
lead list of primary papers, each of which would be loaded from its own release, never
from this transcription.

Script and committed results:
``experiments/036-dataset-fixes-before-kg-build/scripts/d2cell2026_release_inventory.py``.
Finding: [[torchcell.datasets.ecoli.li2024]].
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.data import verify_sha256
from torchcell.literature.manifest import (
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY: Final = "liLeveragingLargeLanguage2024"
PAPER_DOI: Final = "10.1101/2024.09.09.612023"
PAPER_TITLE: Final = "Leveraging large language models for metabolic engineering design"
PAPER_MD: Final = "paper.md"
PAPER_MD_SHA256: Final = (
    "ec236a1aec4a854c51aa088873981dd77eb8dc95991e49e0e5c5fd57c10ae8e7"
)
RAW_DIR_REL: Final = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL: Final = f"torchcell-library/{CITATION_KEY}"
RETRIEVED_AT: Final = "2026-10-10"

ZENODO_RECORD: Final = "https://zenodo.org/records/18240770"
#: The GitHub commit the split files were read at (HEAD on 2026-10-10).
GITHUB_COMMIT: Final = "db5f0a83f376b8f5849d686e31907861014c468e"
_GITHUB_RAW: Final = (
    f"https://raw.githubusercontent.com/LiLabTsinghua/D2Cell/{GITHUB_COMMIT}/"
    "Data/D2Cell-pred%20Data/Ecoli/"
)

DATABASE_FILE: Final = "Cell_factory_dataset_Qwen_110b.xlsx"
SPLIT_FILES: Final = (
    "ecoli_train_dataset.csv",
    "ecoli_valid_dataset.csv",
    "ecoli_test_dataset.csv",
)
ECOLI_ORGANISM: Final = "Escherichia coli"

PAPER: Final = Provenance(
    source_uri=PAPER_MD, citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
)


class RawFile(BaseModel):
    """One deposited file: its pinned bytes and how it was retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    relpath: str
    sha256: str
    bytes: int
    method: RetrievalMethod
    source_url: str
    description: str

    @property
    def retrieval(self) -> RetrievalRecord:
        """The re-runnable retrieval of these bytes."""
        return RetrievalRecord(
            method=self.method,
            source_url=self.source_url,
            retriever="torchcell.literature.retrieve.direct_url",
            params={"url": self.source_url},
            sha256=self.sha256,
            retrieved_at=RETRIEVED_AT,
        )


RAW_FILES: Final[tuple[RawFile, ...]] = (
    RawFile(
        name=DATABASE_FILE,
        relpath=f"data/{DATABASE_FILE}",
        sha256="ce4f08ed0104ea6eda5c17a434aefa4138cc11358e5e917915494eecf64adcde",
        bytes=11437401,
        method=RetrievalMethod.zenodo,
        source_url=f"{ZENODO_RECORD}/files/{DATABASE_FILE}?download=1",
        description="Zenodo 18240770 'D2Cell cell factory dataset' (2026-01-14), the "
        "record's only file (md5 202e5e260f9290e7271e1f4fa7c5f2ae as Zenodo lists it): "
        "one sheet of LLM-extracted entries, 39 columns, one source DOI per row",
    ),
    RawFile(
        name=SPLIT_FILES[0],
        relpath=f"data/{SPLIT_FILES[0]}",
        sha256="e99650cd75464c5c1dd376b6bc53391989742b8f266b553641858784bbc2747b",
        bytes=481622,
        method=RetrievalMethod.direct_url,
        source_url=_GITHUB_RAW + SPLIT_FILES[0],
        description="D2Cell-pred E. coli training split: (perturbation index, product "
        "index, binary label, real-data flag, doi list)",
    ),
    RawFile(
        name=SPLIT_FILES[1],
        relpath=f"data/{SPLIT_FILES[1]}",
        sha256="587ead72ea1368532cd5b146ded9370fe11f2587050a056949884eef8ccc3b60",
        bytes=59348,
        method=RetrievalMethod.direct_url,
        source_url=_GITHUB_RAW + SPLIT_FILES[1],
        description="D2Cell-pred E. coli validation split, same columns",
    ),
    RawFile(
        name=SPLIT_FILES[2],
        relpath=f"data/{SPLIT_FILES[2]}",
        sha256="d69335d4a1dc4259874bcac1e70ceba29ce09932a49a9b07ca1e72fb7b62d36f",
        bytes=61438,
        method=RetrievalMethod.direct_url,
        source_url=_GITHUB_RAW + SPLIT_FILES[2],
        description="D2Cell-pred E. coli test split, same columns",
    ),
)
RAW_FILES_BY_NAME: Final[dict[str, RawFile]] = {f.name: f for f in RAW_FILES}

# --------------------------------------------------------------------------- #
# What the release says about itself (verbatim, paper.md)
# --------------------------------------------------------------------------- #
DATABASE_SIZE: Final = SourcedValue(
    value={"entries": 29006, "products": 1210, "organisms": 751},
    provenance=PAPER.model_copy(update={"page": "Abstract"}),
    quote=(
        "We created a database containing over 29006 metabolic engineering entries, "
        "1210 products and 751 organisms."
    ),
)

CORPUS: Final = SourcedValue(
    value={"abstracts": 10000, "open_access_full_texts": 1340},
    provenance=PAPER.model_copy(
        update={"page": "Results, D2Cell assisted metabolic engineering database"}
    ),
    quote=(
        "we processed the metabolic engineering literature published from 2000 to "
        "2023 by various publisher groups, including more than 10000 abstracts and "
        "1340 openaccess full texts"
    ),
    note="Most entries were read from an abstract, not from a results table.",
)

RE_MODEL: Final = SourcedValue(
    value="Qwen1.5-110B-Chat",
    provenance=PAPER.model_copy(update={"page": "Methods, D2Cell-learn"}),
    quote=(
        "In the RE module, we use the Qwen1.5- 110B-Chat model to extract "
        "relationships between entities."
    ),
    note="The model that wrote every value of the deposited workbook (its file name).",
)

NER_PRECISION: Final = SourcedValue(
    value=0.85,
    provenance=PAPER.model_copy(update={"page": "Results, D2Cell-learn"}),
    quote="The Qwen-Lora model achieved the highest precision $( 8 5 \\% )$",
    note="Precision of entity recognition on the 873 annotated text segments.",
)

TEXT_ONLY: Final = SourcedValue(
    value="figures_and_tables_not_read",
    provenance=PAPER.model_copy(update={"page": "Discussion"}),
    quote=(
        "the methodological approach primarily concentrated on extracting information "
        "from textual content, failing to fully leverage the rich information embedded "
        "within figure illustrations and tabular data"
    ),
    note="The tables where primary papers report titers were not the source read.",
)

ECOLI_TRAINING_COUNTS: Final = SourcedValue(
    value={"experimental": 8134, "simulated": 11643, "products": 73},
    provenance=PAPER.model_copy(update={"page": "Results, D2Cell-pred"}),
    quote=(
        "The total dataset for $E .$ . coli consists of 8134 experimentally reported "
        "single- and double-gene modification data for 73 products and 11643 single "
        "gene modification data for 73 products simulated by GEM."
    ),
)

ECOLI_TRAINING_TOTAL: Final = SourcedValue(
    value=19777,
    provenance=PAPER.model_copy(update={"page": "Methods, dataset construction"}),
    quote=(
        "The dataset for E. coli, $S .$ cerevisiae, and C. glutamicum contain 19777, "
        "21021, and 5170 entries, respectively."
    ),
)

BINARY_LABEL: Final = SourcedValue(
    value="binary_enhancement_label",
    provenance=PAPER.model_copy(update={"page": "Methods, dataset construction"}),
    quote=(
        "a binary classification indicating the impact of the gene modification on the "
        "product production (1 denotes enhancement, 0 denotes no or negative impact)"
    ),
)

NEGATIVE_DESIGNATION: Final = SourcedValue(
    value="negatives_designated_by_rule",
    provenance=PAPER.model_copy(update={"page": "Methods, dataset construction"}),
    quote=(
        "Therefore, we designated gene modifications opposing those enhancing product "
        "production in the literature as negative samples."
    ),
    note="A 0 for the opposite modification is inferred, not reported by any paper.",
)

SIMULATED_NEGATIVES: Final = SourcedValue(
    value="fseof_non_targets_as_negatives",
    provenance=PAPER.model_copy(update={"page": "Methods, dataset construction"}),
    quote=(
        "Additionally, genes not predicted as targets by FSEOF were included in the "
        "negative dataset."
    ),
)

ENTRY_FIELDS: Final = SourcedValue(
    value="one_titer_per_entry_no_reference",
    provenance=PAPER.model_copy(
        update={"page": "Results, D2Cell assisted metabolic engineering database"}
    ),
    quote=(
        "Each entry includes details on the organism, product, product titer, genetic "
        "modification, medium setup, oxygen availability, volume, and temperature."
    ),
    note="One titer per entry; no parent-strain titer is part of an entry.",
)


class SchemaBlocker(BaseModel):
    """One required schema field the release has no honest value for."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    requirement: str
    evidence: str = Field(description="the measured or quoted reason, by name")


SCHEMA_BLOCKERS: Final[tuple[SchemaBlocker, ...]] = (
    SchemaBlocker(
        field="ProductTiterExperimentReference.phenotype_reference",
        requirement="a parent-strain titer read in the same CultureEnvironment",
        evidence="ENTRY_FIELDS: one titer per entry; the 'parent strain' column names "
        "a strain (or, measured, a temperature), never its titer",
    ),
    SchemaBlocker(
        field="ProductTiterPhenotype.titer_unit",
        requirement="a typed ConcentrationUnit",
        evidence="DatabaseMeasurement.titer_unit_counts: the 'titer unit' column mixes "
        "concentrations with yields (g/g, mol/mol), percentages and activities (U/mL)",
    ),
    SchemaBlocker(
        field="ProductTiterExperiment.genotype",
        requirement="typed bacterial perturbations, each an edit to genomic content",
        evidence="'knock out gene' / 'overexpress gene' / 'heterologous gene' are "
        "semicolon-joined gene names with no locus, allele, promoter or copy number",
    ),
    SchemaBlocker(
        field="every value",
        requirement="a verbatim quote of the source sentence with its sha256",
        evidence="RE_MODEL + TEXT_ONLY: no row carries the text it was extracted from",
    ),
)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
#: A parent-strain cell that is a temperature, the signature of a shifted field.
TEMPERATURE_CELL: Final = re.compile(r"^\s*\d+(\.\d+)?\s*°\s*C\s*$")
#: Placeholder phrases an extractor writes when the text had no value.
PLACEHOLDER: Final = re.compile(
    r"not specified|assumed|see note|same as|similar", re.IGNORECASE
)
CJK: Final = re.compile(r"[一-鿿]")
#: Condition fields the placeholder count reads.
CONDITION_FIELDS: Final = (
    "medium",
    "carbon source concentration",
    "time",
    "temperature",
    "product titer",
)
GENOTYPE_FIELDS: Final = ("knock out gene", "overexpress gene", "heterologous gene")


class DatabaseMeasurement(BaseModel):
    """What the deposited workbook holds, counted off its bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_rows_nonempty: int
    n_columns: int
    n_organisms: int
    n_ecoli_rows: int
    n_ecoli_source_dois: int = Field(description="distinct per-row source DOIs")
    n_ecoli_rows_without_doi: int
    n_ecoli_numeric_titer: int
    n_distinct_titer_units: int
    titer_unit_counts: dict[str, int]
    n_ecoli_parent_strain_missing: int
    n_ecoli_parent_strain_is_temperature: int
    n_ecoli_no_genotype: int
    n_ecoli_rows_with_placeholder: int
    n_ecoli_rows_with_cjk: int
    n_ecoli_rows_with_source_quote: int = Field(
        description="rows carrying the sentence the value was read from (no column)"
    )


def measure_database(frame: pd.DataFrame) -> DatabaseMeasurement:
    """Count the workbook's E. coli surface and its extraction faults (pure)."""
    rows = frame.dropna(how="all")
    ecoli = rows[rows["organism"] == ECOLI_ORGANISM]
    parent = ecoli["parent strain"]
    as_text = ecoli.astype(str).where(ecoli.notna(), "")
    placeholder = (
        as_text[list(CONDITION_FIELDS)]
        .apply(lambda col: col.str.contains(PLACEHOLDER))
        .any(axis=1)
    )
    cjk = as_text.apply(lambda col: col.str.contains(CJK)).any(axis=1)
    quote_columns = [c for c in ecoli.columns if "quote" in c.lower()]
    units = ecoli["titer unit"].dropna().astype(str).value_counts()
    return DatabaseMeasurement(
        n_rows_nonempty=len(rows),
        n_columns=rows.shape[1],
        n_organisms=int(rows["organism"].nunique()),
        n_ecoli_rows=len(ecoli),
        n_ecoli_source_dois=int(ecoli["DOI"].nunique()),
        n_ecoli_rows_without_doi=int(ecoli["DOI"].isna().sum()),
        n_ecoli_numeric_titer=int(
            pd.to_numeric(ecoli["titer value"], errors="coerce").notna().sum()
        ),
        n_distinct_titer_units=len(units),
        titer_unit_counts={str(k): int(v) for k, v in units.items()},
        n_ecoli_parent_strain_missing=int(parent.isna().sum()),
        n_ecoli_parent_strain_is_temperature=int(
            parent.dropna().astype(str).str.match(TEMPERATURE_CELL).sum()
        ),
        n_ecoli_no_genotype=int(ecoli[list(GENOTYPE_FIELDS)].isna().all(axis=1).sum()),
        n_ecoli_rows_with_placeholder=int(placeholder.sum()),
        n_ecoli_rows_with_cjk=int(cjk.sum()),
        n_ecoli_rows_with_source_quote=(
            int(ecoli[quote_columns].notna().any(axis=1).sum()) if quote_columns else 0
        ),
    )


def ecoli_source_dois(frame: pd.DataFrame) -> pd.DataFrame:
    """Per source DOI: its E. coli entry count and title, the curation lead list."""
    ecoli = frame.dropna(how="all")
    ecoli = ecoli[ecoli["organism"] == ECOLI_ORGANISM]
    grouped = ecoli.groupby("DOI").agg(
        n_entries=("organism", "size"),
        publication_year=("Publication Year", "first"),
        title=("Article Title", "first"),
    )
    return grouped.reset_index().rename(columns={"DOI": "doi"})


class SplitMeasurement(BaseModel):
    """What the D2Cell-pred E. coli split files hold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_rows: dict[str, int]
    n_total: int
    n_real_flagged: int = Field(description="rows whose 'real data' is 'yes'")
    n_simulated: int
    label_counts_real: dict[int, int]
    label_counts_simulated: dict[int, int]
    n_with_doi: int
    n_real_positive_with_doi: int
    stated_experimental: int = Field(description="ECOLI_TRAINING_COUNTS experimental")
    stated_total: int = Field(description="ECOLI_TRAINING_TOTAL")


def _label_counts(labels: pd.Series) -> dict[int, int]:
    """``{label: count}`` with plain ints."""
    counts = labels.value_counts()
    return dict(
        zip(map(int, counts.index.tolist()), map(int, counts.tolist()), strict=True)
    )


def measure_training_split(splits: Mapping[str, pd.DataFrame]) -> SplitMeasurement:
    """Count the split files' real/simulated partition and labels (pure)."""
    frame = pd.concat(list(splits.values()), ignore_index=True)
    real = frame["real data"].eq("yes")
    has_doi = frame["doi list"].notna()
    label = frame["inf_label_01"].astype(int)
    stated = ECOLI_TRAINING_COUNTS.value
    return SplitMeasurement(
        n_rows={name: len(df) for name, df in splits.items()},
        n_total=len(frame),
        n_real_flagged=int(real.sum()),
        n_simulated=int((~real).sum()),
        label_counts_real=_label_counts(label[real]),
        label_counts_simulated=_label_counts(label[~real]),
        n_with_doi=int(has_doi.sum()),
        n_real_positive_with_doi=int((real & has_doi & label.eq(1)).sum()),
        stated_experimental=int(stated["experimental"]),
        stated_total=int(ECOLI_TRAINING_TOTAL.value),
    )


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/liLeveragingLargeLanguage2024``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """This key's directory in the OCR library mirror."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def deposit_raw_mirror(
    *, sources: Mapping[str, str | Path], data_root: str | None = None
) -> Path:
    """Copy verified files into the raw mirror and write its ``manifest.json``.

    Idempotent by sha256: a mirror file with the pinned hash is left alone and a source
    or mirror file with any other hash raises.
    """
    missing = sorted(set(RAW_FILES_BY_NAME) - set(sources))
    if missing:
        raise KeyError(f"no source given for {missing}")
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for raw in RAW_FILES:
        src = Path(sources[raw.name])
        verify_sha256(src, raw.sha256)
        dest = root / raw.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            verify_sha256(dest, raw.sha256)
        else:
            shutil.copy2(src, dest)
        records.append(
            ArtifactRecord(
                path=raw.relpath,
                role="raw_data",
                bytes=dest.stat().st_size,
                sha256=raw.sha256,
                source=raw.source_url,
                original_filename=raw.name,
                retrieval=raw.retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=records,
        si_data_sources=[
            ZENODO_RECORD,
            f"https://github.com/LiLabTsinghua/D2Cell/tree/{GITHUB_COMMIT}",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def read_database(data_root: str | None = None) -> pd.DataFrame:
    """The pinned workbook, verified, every cell as an object."""
    path = raw_mirror_dir(data_root) / RAW_FILES_BY_NAME[DATABASE_FILE].relpath
    verify_sha256(path, RAW_FILES_BY_NAME[DATABASE_FILE].sha256)
    return pd.read_excel(path, dtype=object)


def read_splits(data_root: str | None = None) -> dict[str, pd.DataFrame]:
    """The three pinned split files, verified."""
    out: dict[str, pd.DataFrame] = {}
    for name in SPLIT_FILES:
        raw = RAW_FILES_BY_NAME[name]
        path = raw_mirror_dir(data_root) / raw.relpath
        verify_sha256(path, raw.sha256)
        out[name] = pd.read_csv(path, encoding="utf-8-sig")
    return out
