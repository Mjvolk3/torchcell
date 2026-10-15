# torchcell/datasets/ecoli/teteneva2024
# [[torchcell.datasets.ecoli.teteneva2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/teteneva2024
# Test file: tests/torchcell/datasets/ecoli/test_teteneva2024.py
r"""Teteneva 2024 lake water RB-TnSeq, E. coli K-12 W3110 RpoS+: Table S4, loaded.

Teteneva, Sanches-Medeiros and Sourjik 2024 (The ISME Journal 18(1):wrae096,
doi:10.1093/ismejo/wrae096, PMC11188689, PMID 38874171) is row 41 of the fifty bacterial
rows (issue #910, candidate rank 40): a randomly barcoded Tn5 library of E. coli K-12
W3110 RpoS+ grown in lake water. Supplementary Table S4 is the release: one row per
(water sample, gene), six gene fitness values per row (filtered and non-filtered water at
days 2, 4 and 8), 11,027 rows and 66,162 values (:func:`table_s4_inventory`).

THE RECORDS (:class:`RbTnseqTeteneva2024Dataset`). EVERY released value is a record,
66,162 of them: one ``EnvironmentResponseExperiment`` per (Table S4 row,
filtration, sampling day), the value as an ``EnvironmentResponsePhenotype``:

* the value is the gene fitness, a signed log2 ratio of barcode counts after incubation in
  the lake water over the day-0 culture, weighted over the gene's barcodes, normalized for
  chromosome position and centred on zero (:data:`BARCODE_FITNESS`,
  :data:`GENE_FITNESS_WEIGHTS`, :data:`CHROMOSOME_NORMALIZATION`, :data:`CENTERED`), and
  averaged over the technical replicates (:data:`AVERAGED`). No uncertainty is released,
  which is a typed gap (:data:`UNCERTAINTY_GAP`);
* ``n_samples`` is the number of technical replicates behind the mean: three, except two
  for lake water #1, filtered, day 4, whose replicate ``sample_10`` failed sequencing QC
  and was omitted (Table S3, :data:`QC_OMITTED`, :func:`read_table_s3`);
* genotype: one ``ReleasedIdentifierPerturbation`` per gene (the released-identifier
  leaf of #691), keyed by the gene symbol EXACTLY as Table S4's 'Gene name' gives it
  (``identifier_kind`` 'Gene name', the symbol also the strain label), its kind
  ``transposon_insertion`` and its cassette kanamycin as the Methods state them, with the
  W3110 host namespace. The data is the data (owner, 2026-10-12): no symbol is
  looked up in, resolved to or checked against any annotation. How the symbols land on
  the W3110 annotations is a separate, downstream artifact,
  ``experiments/database/scripts/teteneva2024_symbol_annotation.py``;
* environment: the lake water sample (#1 to #3 of Table S2, with its sampling date, pH
  and air and water temperature), filter-sterilized or not, as its own ``Media``; the
  sampling day as ``duration_hours`` (24 h per stated day); the incubation temperature is
  a typed gap, because the paper states only "room temperature";
* reference: the strain background ``W3110 RpoS+`` declared on the W3110 assembly set,
  which is named only as the strain's reference (RefSeq accession GCF_000010245.2, the
  "NCBI reference sequence: NC_007779.1" the authors mapped reads to), and the typical
  gene of the same environment, response 0 by the normalization.

The module also keeps what the gate-era branch recorded: the W3110 tier deposit
(:data:`W3110_TIER_DEPOSIT`), the PMC bucket listing (:data:`PMC_OA_DEPOSIT`) and the SI
deposit (:data:`SI_DEPOSIT`). The paper text is the PMC Article Datasets object
``PMC11188689.1.txt`` in the raw mirror (:func:`record_paper_and_texts`): the library
mirror held no Teteneva key when this loader was written (checked 2026-10-12), so every
paper quote cites that file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import os.path as osp
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import numpy as np
from dotenv import load_dotenv
from openpyxl import load_workbook
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    link_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.datamodels.media import LAKE_WATER
from torchcell.datamodels.schema import (
    AssayType,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    Publication,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    SampleUnit,
    StrainBackground,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    assembly_reference,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_PAPER_TEXT,
    ROLE_SI_DATA,
    ROLE_SI_PDF,
    ROLE_SI_TEXT,
    ArtifactRecord,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.manual_deposit import (
    DepositedFile,
    ManualDeposit,
    ScriptedEquivalent,
    extend_manifest,
)
from torchcell.literature.retrieve import pmc_cloud_object, pmc_cloud_url
from torchcell.sequence.genome.registry import ECOLI_K12_W3110
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paper identity
# --------------------------------------------------------------------------- #
#: The raw-mirror key the SI deposit created (#910, 2026-10-12). The library mirror held
#: no Teteneva key on 2026-10-12, so whether ``lit_sync`` exports this exact key from
#: Zotero item ``PUKZFNFZ`` is not yet measured; every quote here cites the raw mirror.
CITATION_KEY: Final = "tetenevaGenomewideScreenGenetic2024"
PAPER_DOI: Final = "10.1093/ismejo/wrae096"
PAPER_PMCID: Final = "PMC11188689"
PAPER_PMID: Final = "38874171"
PAPER_TITLE: Final = (
    "Genome-wide screen of genetic determinants that govern Escherichia coli growth "
    "and persistence in lake water"
)
#: Title, authors, journal and DOI as NCBI's ``esummary`` returned them on 2026-10-09;
#: the DOI resolved to this one PMC id through ``esearch`` on the same day.
PAPER_AUTHORS: Final = ("Teteneva N", "Sanches-Medeiros A", "Sourjik V")
PAPER_JOURNAL: Final = "The ISME journal 2024 Jan 8, volume 18, issue 1, wrae096"
RAW_ROOT_REL: Final = "torchcell-raw"

#: The row as the schedule generator recorded it, with the file that holds it. Both
#: counts are that script's measurement off the workbook with ``openpyxl``.
SCHEDULE_ROW: Final = 41
SCHEDULE_SOURCE: Final = (
    "experiments/database/scripts/build_bacteria_candidate_datasets_table.py"
)
SCHEDULE_FITNESS_VALUES: Final = 66_162
SCHEDULE_GENE_BY_SAMPLE_ROWS: Final = 11_027


class PmcDepositFile(BaseModel):
    """One object of this article's PMC open-access prefix, as the bucket listed it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(description="bucket key, passed straight to pmc_cloud_object")
    role: str = Field(description="paper | si_data | si_pdf | figure | metadata")


_OA_PREFIX: Final = f"{PAPER_PMCID}.1"

#: Every object under ``pmc-oa-opendata/PMC11188689.1/``, listed on 2026-10-09 with
#: ``GET /?list-type=2&prefix=PMC11188689``. NOTHING here was downloaded: the listing is
#: evidence that the retrieval is scriptable the moment the paper is curated, and
#: ``supplementary_table_s4_wrae096.xlsx`` is the workbook the row was counted off.
PMC_OA_DEPOSIT: Final = (
    PmcDepositFile(key=f"{_OA_PREFIX}/{_OA_PREFIX}.pdf", role="paper"),
    PmcDepositFile(key=f"{_OA_PREFIX}/{_OA_PREFIX}.xml", role="paper"),
    PmcDepositFile(key=f"{_OA_PREFIX}/{_OA_PREFIX}.txt", role="paper"),
    PmcDepositFile(key=f"{_OA_PREFIX}/{_OA_PREFIX}.json", role="metadata"),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_table_s1_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_table_s2_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_table_s3_wrae096.xlsx", role="si_data"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_table_s4_wrae096.xlsx", role="si_data"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_figure_s1_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_figure_s2_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_figure_s3_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_figure_s4_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(key=f"{_OA_PREFIX}/wrae096f1.jpg", role="figure"),
    PmcDepositFile(key=f"{_OA_PREFIX}/wrae096f2.jpg", role="figure"),
    PmcDepositFile(key=f"{_OA_PREFIX}/wrae096f3.jpg", role="figure"),
    PmcDepositFile(key=f"{_OA_PREFIX}/wrae096f4.jpg", role="figure"),
)
#: The one object a loader would read its values out of.
TABLE_S4_OA_KEY: Final = f"{_OA_PREFIX}/supplementary_table_s4_wrae096.xlsx"


# --------------------------------------------------------------------------- #
# The host: the deposited W3110 assembly set
# --------------------------------------------------------------------------- #
class TierMember(BaseModel):
    """One member of the W3110 set, as fetched, md5-matched and deposited."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    role: str = Field(description="genomes-tier role: annotation | sequence | index")
    bytes: int
    md5: str = Field(description="NCBI's md5checksums.txt value, matched on retrieval")
    sha256: str


class DepositedAssemblySet(BaseModel):
    """The W3110 assembly set as it was deposited into the genomes tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assembly_set: str
    organism: str = Field(description="'# Organism name' of the assembly report")
    strain: str
    assembly_name: str
    genbank_accession: str
    refseq_accession: str
    genbank_replicon: str
    refseq_replicon: str
    replicon_length_bp: int
    genbank_url: str
    refseq_url: str
    md5_checksum_files: dict[str, str] = Field(
        description="URL of each directory's md5checksums.txt -> sha256 of the copy used"
    )
    members: tuple[TierMember, ...]
    deposited_at: str


_NCBI: Final = "https://ftp.ncbi.nlm.nih.gov/genomes/all"
_GCA_DIR: Final = f"{_NCBI}/GCA/000/010/245/GCA_000010245.1_ASM1024v1"
_GCF_DIR: Final = f"{_NCBI}/GCF/000/010/245/GCF_000010245.2_ASM1024v1"

#: The GCA ``_assembly_report.txt`` of this set still names the 2006 RefSeq release,
#: ``GCF_000010245.1`` with RefSeq-Accn ``AC_000091.1``, while the current RefSeq
#: release is ``GCF_000010245.2`` with RefSeq-Accn ``NC_007779.1``. The accession PAIR
#: is therefore only readable off the GCF report, which is why that file is a member of
#: this set and of no other.
STALE_GCA_REFSEQ_ACCESSION: Final = "GCF_000010245.1"
STALE_GCA_REFSEQ_REPLICON: Final = "AC_000091.1"

W3110_TIER_DEPOSIT: Final = DepositedAssemblySet(
    assembly_set=ECOLI_K12_W3110,
    organism="Escherichia coli str. K-12 substr. W3110 (E. coli)",
    strain="K-12 W3110",
    assembly_name="ASM1024v1",
    genbank_accession="GCA_000010245.1",
    refseq_accession="GCF_000010245.2",
    genbank_replicon="AP009048.1",
    refseq_replicon="NC_007779.1",
    replicon_length_bp=4_646_332,
    genbank_url=f"{_GCA_DIR}/",
    refseq_url=f"{_GCF_DIR}/",
    md5_checksum_files={
        f"{_GCA_DIR}/md5checksums.txt": (
            "6674b4dec2356b88516648a3947cc8a6c7085b1c20b1af2982edd884a9e70321"
        ),
        f"{_GCF_DIR}/md5checksums.txt": (
            "71a7b12e78101e72ddbca9c018bcc933c81c0a8b14646ac423c588948b71dbee"
        ),
    },
    members=(
        TierMember(
            path="GCA_000010245.1_ASM1024v1_genomic.gbff.gz",
            role="annotation",
            bytes=3286152,
            md5="7c5e9f6b10f282bc9dc82ddcdc0bcc8d",
            sha256="7e22368bc1783fe3b56b07a7196dd0abe3104ef88cd795ad0970ab546418a157",
        ),
        TierMember(
            path="GCA_000010245.1_ASM1024v1_genomic.fna.gz",
            role="sequence",
            bytes=1381336,
            md5="7cc36bddd25647f60941a20c76571bac",
            sha256="57709b8e4bf6a66951db1779bb61a75fbca2db416aa3a1ba245365fdebff93a0",
        ),
        TierMember(
            path="GCA_000010245.1_ASM1024v1_genomic.gff.gz",
            role="annotation",
            bytes=287541,
            md5="f73b4df02eb07ccd0ab437d2f728d221",
            sha256="8aa0069b3715239d16103653b7becd27c5e59676f711d20854f43882c033bb6e",
        ),
        TierMember(
            path="GCA_000010245.1_ASM1024v1_protein.faa.gz",
            role="sequence",
            bytes=892689,
            md5="cc6b6bdc016bb04306214642419a8306",
            sha256="dfb59618a7c2c993859ad6b05c9c613a9ae1f7e2605cc7298ac286652ad33ff5",
        ),
        TierMember(
            path="GCA_000010245.1_ASM1024v1_feature_table.txt.gz",
            role="index",
            bytes=167217,
            md5="6f3ab693a5a99c1a76085a347a2b58ed",
            sha256="21b04994e343fa3bf892492f70180bc215110f4016ad463a47acce6a24112ec8",
        ),
        TierMember(
            path="GCA_000010245.1_ASM1024v1_assembly_report.txt",
            role="index",
            bytes=1283,
            md5="aeb8a80d69d84f7f69f752d2a6996d12",
            sha256="d3f6d0fc9bd9e8ef9479c9554239e8618a7d0bc464c362304b25385e646ce267",
        ),
        TierMember(
            path="GCF_000010245.2_ASM1024v1_genomic.gbff.gz",
            role="annotation",
            bytes=3452629,
            md5="526fb39d6930215f7c627b977536bfbd",
            sha256="3edf6662a5aac6a28f3550622f66328e88ae775da117cc0004eedb1f066b991e",
        ),
        TierMember(
            path="GCF_000010245.2_ASM1024v1_genomic.gff.gz",
            role="annotation",
            bytes=435483,
            md5="fa616f6df51b2db6d00514b8762306b6",
            sha256="1501e3787567e224ccd03e81b3467f00e1bad7db719b1e41db5200d46edce4f3",
        ),
        TierMember(
            path="GCF_000010245.2_ASM1024v1_gene_ontology.gaf.gz",
            role="annotation",
            bytes=156751,
            md5="508c8d4e53593b0986d5dc4a38cfd0a2",
            sha256="cd8de975cf31e31145c1254f6bf4f393ec275f56e1dba706f850bf1714b611a8",
        ),
        TierMember(
            path="GCF_000010245.2_ASM1024v1_assembly_report.txt",
            role="index",
            bytes=1200,
            md5="58588ab7b196dd61a04cdeb76d7836b6",
            sha256="98015272fe4acd3171174109f666137bd310885f580100e11c2ed1ae132b8c1c",
        ),
    ),
    deposited_at="2026-10-09",
)


# --------------------------------------------------------------------------- #
# The paper in Zotero, and in neither mirror's library tree
# --------------------------------------------------------------------------- #
#: The library mirror held no Teteneva key when this loader was written (listed on
#: 2026-10-12, after the filing and before the nightly lit_sync); the raw mirror holds
#: the SI deposit's key. So the paper text comes from the PMC bucket into the raw mirror
#: (:func:`record_paper_and_texts`).
LIBRARY_MIRROR_KEYS_MATCHING_TETENEVA: Final = 0
RAW_MIRROR_KEYS_MATCHING_TETENEVA: Final = 1
#: The Zotero item the owner filed on 2026-10-12 (#910), and its PDF attachment.
ZOTERO_ITEM: Final = "PUKZFNFZ"
ZOTERO_PDF_ATTACHMENT: Final = "DFFQ6A6W"

# --------------------------------------------------------------------------- #
# The owner's SI deposit (#910, 2026-10-12)
# --------------------------------------------------------------------------- #
SI_PAGE: Final = f"https://doi.org/{PAPER_DOI}"


def _si_file(
    name: str, role: str, sha256: str, size: int, arrived: str, holds: str
) -> DepositedFile:
    """One deposited SI file, with the PMC bucket object measured to equal it."""
    key = f"{_OA_PREFIX}/{name}"
    return DepositedFile(
        path=f"si/{name}",
        role=role,
        sha256=sha256,
        bytes=size,
        original_filename=name,
        arrived_at=f"2026-10-12T03:40:{arrived}-05:00",
        description=holds,
        scripted_equivalent=ScriptedEquivalent(
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": key},
            url=pmc_cloud_url(key),
            checked_at="2026-10-12",
        ),
    )


SI_DEPOSIT: Final = ManualDeposit(
    citation_key=CITATION_KEY,
    doi=PAPER_DOI,
    title=PAPER_TITLE,
    source_url=SI_PAGE,
    retrieval_command=(
        f"open {SI_PAGE} (The ISME Journal) in a browser, open the article's "
        "Supplementary data, download the five files, and copy them unchanged to si/; "
        "or, measured equivalent, fetch each from the PMC Article Datasets bucket "
        f"under {_OA_PREFIX}/ with torchcell.literature.retrieve.pmc_cloud_object"
    ),
    retrieved_by="the owner (mjvolk3), browser download",
    deposited_at="2026-10-12",
    issues=(910,),
    note="five of the article's supplementary files; the PMC bucket also lists "
    "Supplementary Table S1 and Figures S3 and S4, which were not downloaded. The "
    "DOI 10.1093/ismejo/wrae096 is the one NCBI esummary returned for PMC11188689 and "
    f"the one Zotero item {ZOTERO_ITEM} was filed under (#910); every file name "
    "carries the article id 'wrae096'",
    files=(
        _si_file(
            "supplementary_figure_s1_wrae096.pdf",
            ROLE_SI_PDF,
            "65bcac7a61e938a92b90838401e7a153e6ff5ec33ae57b5840950fb11efd5d82",
            354566,
            "57",
            "Supplementary Figure S1, hierarchical clustering of gene fitness, 1 page",
        ),
        _si_file(
            "supplementary_figure_s2_wrae096.pdf",
            ROLE_SI_PDF,
            "d88f962bd32bb756a0d16659e5b3208ad84b7dc620e21d747f0ff549d369f356",
            251076,
            "57",
            "Supplementary Figure S2, PCA of gene fitness per water sample, 1 page",
        ),
        _si_file(
            "supplementary_table_s2_wrae096.pdf",
            ROLE_SI_PDF,
            "c41e8b17ed4ecd8ddf420332e4d32a33aae13ae7b029feb275632a5baf5ab396",
            36151,
            "57",
            "Supplementary Table S2, date, pH, air and water temperature of five lake "
            "water samples, 1 page",
        ),
        _si_file(
            "supplementary_table_s3_wrae096.xlsx",
            ROLE_SI_DATA,
            "5cec5dd2f2b3fbc46a0012154b1864dda812326f12b1622391081e96995e323b",
            14056,
            "57",
            "Supplementary Table S3, sequencing QC of 63 samples (FastQC/MultiQC)",
        ),
        _si_file(
            "supplementary_table_s4_wrae096.xlsx",
            ROLE_SI_DATA,
            "678bbe6cecc2a11517fc11cbeb4d5a5ff88bca3e4aea1264bad336d5fccad0dc",
            1809613,
            "58",
            "Supplementary Table S4, gene fitness values, principal components and "
            "Euclidean distances per water sample",
        ),
    ),
)
TABLE_S4_RELPATH: Final = "si/supplementary_table_s4_wrae096.xlsx"
TABLE_S3_RELPATH: Final = "si/supplementary_table_s3_wrae096.xlsx"


class TableS4Inventory(BaseModel):
    """What Supplementary Table S4's DATA sheet holds, measured off the bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    header_rows: int
    gene_by_sample_rows: int
    rows_per_water_sample: dict[int, int]
    fitness_columns: tuple[str, ...]
    fitness_values: int
    identifier_column: str
    distinct_symbols: tuple[str, ...]
    other_columns: tuple[str, ...]


#: Table S4's DATA sheet: three header rows, then one row per (water sample, gene).
_S4_HEADER_ROWS: Final = 3
_S4_FITNESS: Final = slice(1, 7)
_S4_SYMBOL: Final = 7


def table_s4_inventory(path: str | Path) -> TableS4Inventory:
    """Measure Table S4: rows, water samples, the six fitness columns, the identifiers.

    The three header rows name the block ("gene fitness values", then "filtered" and
    "non-filtered", then "day 2/4/8"), so a fitness column is named
    ``<water>, <day>`` by carrying the merged cells to the right. Raises ``ValueError``
    when the headers are not that layout or a row carries no gene symbol.
    """
    workbook = load_workbook(io.BytesIO(Path(path).read_bytes()), read_only=True)
    rows = list(workbook["DATA"].iter_rows(values_only=True))
    workbook.close()
    top, water, day = rows[:_S4_HEADER_ROWS]
    if (top[0], top[1], top[_S4_SYMBOL]) != (
        "Lake water sample",
        "gene fitness values",
        "Gene name",
    ):
        raise ValueError(f"Table S4 header is {top}")
    columns: list[str] = []
    current = ""
    for w, d in zip(water[_S4_FITNESS], day[_S4_FITNESS], strict=True):
        current = str(w) if w is not None else current
        columns.append(f"{current}, {d}")
    body = rows[_S4_HEADER_ROWS:]
    if any(row[_S4_SYMBOL] is None for row in body):
        raise ValueError("a Table S4 row carries no gene symbol")
    per_sample = Counter(int(row[0]) for row in body)
    return TableS4Inventory(
        header_rows=_S4_HEADER_ROWS,
        gene_by_sample_rows=len(body),
        rows_per_water_sample=dict(sorted(per_sample.items())),
        fitness_columns=tuple(columns),
        fitness_values=sum(1 for r in body for c in r[_S4_FITNESS] if c is not None),
        identifier_column=str(top[_S4_SYMBOL]),
        distinct_symbols=tuple(sorted({str(row[_S4_SYMBOL]) for row in body})),
        other_columns=tuple(str(c) for c in top[_S4_SYMBOL + 1 :] if c is not None),
    )


#: Measured on the deposited bytes on 2026-10-12 by :func:`table_s4_inventory`.
TABLE_S4_GENE_BY_SAMPLE_ROWS: Final = 11_027
TABLE_S4_FITNESS_VALUES: Final = 66_162
TABLE_S4_ROWS_PER_WATER_SAMPLE: Final = {1: 3_675, 2: 3_679, 3: 3_673}
TABLE_S4_DISTINCT_SYMBOLS: Final = 3_691
TABLE_S4_FITNESS_COLUMNS: Final = (
    "filtered, day 2",
    "filtered, day 4",
    "filtered, day 8",
    "non-filtered, day 2",
    "non-filtered, day 4",
    "non-filtered, day 8",
)


# --------------------------------------------------------------------------- #
# The paper text and the derived texts every quote cites (raw mirror)
# --------------------------------------------------------------------------- #
#: The article's full text as PMC renders it, the anchor of every paper quote below. Not
#: OCR: the publisher's own text, so a quote matches character for character (thin and
#: no-break spaces included, written here as `` `` and ``\xa0``).
PAPER_TEXT: Final = f"{_OA_PREFIX}.txt"
PAPER_TEXT_RELPATH: Final = f"paper/{PAPER_TEXT}"
PAPER_TEXT_KEY: Final = f"{_OA_PREFIX}/{PAPER_TEXT}"
PAPER_TEXT_SHA256: Final = (
    "422bd7926fbeebfd6877e15f213e18914291b1f259a7086d66d66d0249353b2f"
)
PAPER_TEXT_BYTES: Final = 61_894
PAPER_TEXT_RETRIEVED_AT: Final = "2026-10-12"

TABLE_S2_PDF_RELPATH: Final = "si/supplementary_table_s2_wrae096.pdf"
TABLE_S2_PDF_SHA256: Final = (
    "c41e8b17ed4ecd8ddf420332e4d32a33aae13ae7b029feb275632a5baf5ab396"
)
#: Table S2's text layer as ``pdftotext -layout`` lays it out: the source of the water
#: sample rows (date, pH, air and water temperature).
TABLE_S2_TEXT_RELPATH: Final = "si/supplementary_table_s2_wrae096.layout.txt"
TABLE_S2_TEXT_SHA256: Final = (
    "78f0ba620a574aeec9f6315c3411ac5bf0453030d96676c94e2707d44eafbce1"
)
PDFTOTEXT_ARGS: Final[tuple[str, ...]] = ("-layout", "-enc", "UTF-8")
PDFTOTEXT_VERSION: Final = "pdftotext version 21.01.0"

TABLE_S3_SHA256: Final = (
    "5cec5dd2f2b3fbc46a0012154b1864dda812326f12b1622391081e96995e323b"
)
TABLE_S4_SHA256: Final = (
    "678bbe6cecc2a11517fc11cbeb4d5a5ff88bca3e4aea1264bad336d5fccad0dc"
)
#: Tables S3 and S4 rendered to text (:func:`render_workbook_texts`): both LEGEND sheets
#: and the Table S3 rows carrying the legend's red highlight, so a quote of either is
#: auditable; the workbooks themselves are binary.
WORKBOOK_TEXT_RELPATH: Final = "si/supplementary_tables_s3_s4.rendered.txt"
WORKBOOK_TEXT_SHA256: Final = (
    "b2f8ad50aa2ecedd091d65e0765b152bd8b5ef4ba11c1e4b54a1471ba47bd646"
)
#: The fill Table S3 marks its QC-failed sample with (Excel's "Light Red Fill").
QC_HIGHLIGHT_FILL: Final = "FFFFC7CE"


def paper_text_retrieval() -> RetrievalRecord:
    """The re-runnable retrieval of the paper text from the PMC Article Datasets bucket."""
    return RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=pmc_cloud_url(PAPER_TEXT_KEY),
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": PAPER_TEXT_KEY},
        sha256=PAPER_TEXT_SHA256,
        retrieved_at=PAPER_TEXT_RETRIEVED_AT,
    )


def pdftotext_version() -> str:
    """The ``pdftotext`` version line, recorded as the extraction's provenance."""
    result = subprocess.run(
        ["pdftotext", "-v"], capture_output=True, text=True, check=True
    )
    return (result.stderr or result.stdout).strip().split("\n")[0]


def render_table_s2(pdf_path: str | Path) -> str:
    """Table S2's text layer, ``pdftotext -layout -enc UTF-8``."""
    result = subprocess.run(
        ["pdftotext", *PDFTOTEXT_ARGS, str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _openpyxl_version() -> str:
    """The openpyxl version the workbook text was rendered with."""
    import openpyxl

    return str(openpyxl.__version__)


def _workbook(path: str | Path, *, read_only: bool) -> Any:
    """Open a workbook from its bytes (no file handle is left open)."""
    return load_workbook(io.BytesIO(Path(path).read_bytes()), read_only=read_only)


def highlighted_rows(table_s3: str | Path) -> list[tuple[Any, ...]]:
    """Table S3 DATA rows whose every non-empty cell carries :data:`QC_HIGHLIGHT_FILL`."""
    book = _workbook(table_s3, read_only=False)
    rows: list[tuple[Any, ...]] = []
    for row in book["DATA"].iter_rows(min_row=2):
        cells = [c for c in row if c.value is not None]
        fills = {
            c.fill.fgColor.rgb if c.fill is not None and c.fill.fill_type else None
            for c in cells
        }
        if cells and fills == {QC_HIGHLIGHT_FILL}:
            rows.append(tuple(c.value for c in cells))
    book.close()
    return rows


def render_workbook_texts(table_s3: str | Path, table_s4: str | Path) -> str:
    """Both LEGEND sheets, then Table S3's highlighted rows, as deterministic text.

    One line per legend row (non-empty cells, tab-joined), under a ``# workbook`` header;
    then one tab-joined line per Table S3 DATA row every cell of which carries the
    legend's red fill (:data:`QC_HIGHLIGHT_FILL`).
    """
    lines: list[str] = []
    for path in (table_s3, table_s4):
        book = _workbook(path, read_only=True)
        lines.append(f"# workbook: {Path(path).name}, sheet: LEGEND")
        for row in book["LEGEND"].iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None and str(c).strip()]
            if cells:
                lines.append("\t".join(cells))
        book.close()
    lines.append(
        f"# workbook: {Path(table_s3).name}, sheet: DATA, rows filled {QC_HIGHLIGHT_FILL}"
    )
    lines.extend("\t".join(str(v) for v in row) for row in highlighted_rows(table_s3))
    return "\n".join(lines) + "\n"


def _derived_record(
    relpath: str, sha256: str, size: int, source: str, processing: ProcessingRecord
) -> ArtifactRecord:
    """A derived-text manifest record."""
    return ArtifactRecord(
        path=relpath,
        role=ROLE_SI_TEXT,
        bytes=size,
        sha256=sha256,
        source=f"derived from {source}",
        original_filename=Path(relpath).name,
        processing=processing,
    )


def _write_pinned(dest: Path, data: bytes, sha256: str) -> None:
    """Write ``data`` to ``dest`` once; an existing file with another hash is refused."""
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise RuntimeError(f"{dest.name}: produced {got}, pinned {sha256}")
    if dest.exists():
        verify_sha256(str(dest), sha256)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def record_paper_and_texts(
    data_root: str | None = None, paper_text_source: str | Path | None = None
) -> Path:
    """Add the paper text and the two derived texts to the raw mirror's manifest.

    The paper text is fetched by its recorded retriever (``pmc_cloud_object``), or read
    from ``paper_text_source`` when given, and written only when it hashes to
    :data:`PAPER_TEXT_SHA256`. The Table S2 layout text and the workbook text are
    rendered from the deposited SI bytes and refused unless they hash to their pins;
    ``pdftotext`` must be the pinned version. Idempotent: a file already in place with
    its pinned hash is left alone.
    """
    root = raw_mirror_dir(data_root)
    paper = (
        pmc_cloud_object(PAPER_TEXT_KEY)
        if paper_text_source is None
        else Path(paper_text_source).read_bytes()
    )
    _write_pinned(root / PAPER_TEXT_RELPATH, paper, PAPER_TEXT_SHA256)
    version = pdftotext_version()
    if version != PDFTOTEXT_VERSION:
        raise RuntimeError(f"{version!r} is not the pinned {PDFTOTEXT_VERSION!r}")
    verify_sha256(str(root / TABLE_S2_PDF_RELPATH), TABLE_S2_PDF_SHA256)
    s2 = render_table_s2(root / TABLE_S2_PDF_RELPATH).encode("utf-8")
    _write_pinned(root / TABLE_S2_TEXT_RELPATH, s2, TABLE_S2_TEXT_SHA256)
    verify_sha256(str(root / TABLE_S3_RELPATH), TABLE_S3_SHA256)
    verify_sha256(str(root / TABLE_S4_RELPATH), TABLE_S4_SHA256)
    texts = render_workbook_texts(
        root / TABLE_S3_RELPATH, root / TABLE_S4_RELPATH
    ).encode("utf-8")
    _write_pinned(root / WORKBOOK_TEXT_RELPATH, texts, WORKBOOK_TEXT_SHA256)
    records = [
        ArtifactRecord(
            path=PAPER_TEXT_RELPATH,
            role=ROLE_PAPER_TEXT,
            bytes=len(paper),
            sha256=PAPER_TEXT_SHA256,
            source=pmc_cloud_url(PAPER_TEXT_KEY),
            original_filename=PAPER_TEXT,
            retrieval=paper_text_retrieval(),
        ),
        _derived_record(
            TABLE_S2_TEXT_RELPATH,
            TABLE_S2_TEXT_SHA256,
            len(s2),
            TABLE_S2_PDF_RELPATH,
            ProcessingRecord(
                processor="torchcell.datasets.ecoli.teteneva2024.render_table_s2",
                tool="pdftotext",
                version=PDFTOTEXT_VERSION,
                params={"args": list(PDFTOTEXT_ARGS)},
                input_sha256=[TABLE_S2_PDF_SHA256],
            ),
        ),
        _derived_record(
            WORKBOOK_TEXT_RELPATH,
            WORKBOOK_TEXT_SHA256,
            len(texts),
            f"{TABLE_S3_RELPATH} and {TABLE_S4_RELPATH}",
            ProcessingRecord(
                processor="torchcell.datasets.ecoli.teteneva2024.render_workbook_texts",
                tool="openpyxl",
                version=_openpyxl_version(),
                params={
                    "sheets": ["LEGEND", "LEGEND"],
                    "highlight_fill": QC_HIGHLIGHT_FILL,
                },
                input_sha256=[TABLE_S3_SHA256, TABLE_S4_SHA256],
            ),
        ),
    ]
    return extend_manifest(
        root, records, citation_key=CITATION_KEY, doi=PAPER_DOI, title=PAPER_TITLE
    )


def _data_root(data_root: str | None) -> Path:
    """``data_root`` or ``$DATA_ROOT`` (from the environment / ``.env``)."""
    if data_root is not None:
        return Path(data_root)
    load_dotenv()
    return Path(os.environ["DATA_ROOT"])


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/<CITATION_KEY>/``."""
    return _data_root(data_root) / RAW_ROOT_REL / CITATION_KEY


def raw_path(relpath: str, data_root: str | None = None) -> Path:
    """A file of this key's raw mirror."""
    return raw_mirror_dir(data_root) / relpath


# --------------------------------------------------------------------------- #
# Sourced values: the paper text, Table S2's text, the workbook text
# --------------------------------------------------------------------------- #
_PAPER_METHOD: Final = (
    "the article's full text as the PMC Article Datasets bucket serves it "
    f"({PAPER_TEXT_KEY}), mirrored verbatim in torchcell-raw"
)
_METHODS: Final = "Materials and methods"


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote of the pinned paper text."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_TEXT_RELPATH,
            citation_key=CITATION_KEY,
            sha256=PAPER_TEXT_SHA256,
            method=_PAPER_METHOD,
            page=page,
            retrieved=PAPER_TEXT_RETRIEVED_AT,
        ),
    )


def _derived(
    value: Any, quote: str, *, relpath: str, sha256: str, page: str, note: str | None
) -> SourcedValue:
    """Bind a value to a verbatim quote of a derived text of the SI deposit."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=relpath,
            citation_key=CITATION_KEY,
            sha256=sha256,
            method="text rendered from the owner's SI deposit (torchcell-raw)",
            page=page,
        ),
    )


# The strain and the library -------------------------------------------------- #
STRAIN = _paper(
    "W3110 RpoS+",
    "All strains used in this work were derived from the E. coli K-12\xa0W3110 RpoS+ "
    "strain [14]",
    page=f"{_METHODS}, 'Strains'",
)
BACKGROUND_CONSTRUCTION = _paper(
    "a derivative of the laboratory strain with the restored RpoS activity",
    "The E. coli K-12\xa0W3110 RpoS+ strain used in this study is a derivative of the "
    "laboratory strain with the restored RpoS activity.",
    page="Discussion",
    note="the restoring allele is not described in this paper (it cites ref. 14), so "
    "the background carries the genotype statement 'RpoS+' and no typed allele",
)
TRANSPOSON = _paper(
    "Tn5",
    "The transposon mutant library of E. coli W3110 RpoS+ [14] was constructed by "
    "inserting the randomly barcoded Tn5 system (described in detail in Wetmore "
    "et\xa0al. [11]) into the genome.",
    page=f"{_METHODS}, 'Construction and characterization of the transposon mutant "
    "library'",
)
CASSETTE = _paper(
    "kanamycin",
    "It consists of the set of plasmids carrying the transposase, the kanamycin "
    "resistance cassette, and the random 20-bp DNA barcodes [11].",
    page=f"{_METHODS}, 'Construction and characterization of the transposon mutant "
    "library'",
    note="the selectable marker the Tn5 carries into every insertion mutant",
)
REFERENCE_SEQUENCE = _paper(
    "NC_007779.1",
    "The reads containing the confirmed barcodes were then aligned to the E. coli "
    "K-12\xa0W3110 reference genome (NCBI reference sequence: NC_007779.1)",
    page=f"{_METHODS}, 'Construction and characterization of the transposon mutant "
    "library'",
    note="the RefSeq replicon of GCF_000010245.2, whose Y75_RS loci the records name "
    "(owner route #910)",
)
LIBRARY_INSERTIONS = _paper(
    {"unique_insertions": 430_849, "genes": 3_833},
    "The mapping resulted in 430 849 unique transposon insertions in 3833 genes.",
    page=f"{_METHODS}, 'Construction and characterization of the transposon mutant "
    "library'",
)
CODING_INSERTIONS_ONLY = _paper(
    "insertions within coding regions",
    "For the sake of simplicity, only insertions in the coding gene regions, which "
    "were assumed to inactivate the respective gene function, were considered for the "
    "assessment of the gene fitness effects.",
    page="Results, 'Experimental setup to test the growth and survival of E. coli in "
    "lake water'",
)

# The culture ------------------------------------------------------------------ #
PRE_CULTURE = _paper(
    {"medium": "TB", "kanamycin_ug_per_ml": 30, "temperature_c": 37, "od600": 1},
    "Pre-cultures were inoculated from whole 1\xa0ml glycerol stocks of the library "
    "into 50\xa0ml\xa0TB medium supplemented with kanamycin (30\xa0μg/ml) and "
    "grown at 37°C on a rotary shaker to an OD600 ~1.",
    page=f"{_METHODS}, 'Cultivation of transposon mutant library'",
    note="the step before the lake water; Environment states the lake water itself",
)
INCUBATION = _paper(
    {"dilution": "1:50", "temperature": "room temperature", "vessel": "24-well plate"},
    "These pre-cultures were then diluted 1:50 in lake water (~107 cells per "
    "millilitre) and incubated without antibiotics at room temperature in 24-well "
    "plates without shaking.",
    page=f"{_METHODS}, 'Cultivation of transposon mutant library'",
    note="'~107' is the text's rendering of ~10^7. No incubation temperature value is "
    "stated, so Environment.temperature is a typed gap (TEMPERATURE_GAP); no oxygen "
    "regime is stated, so aerobicity keeps the field default (it cannot be None), as "
    "goodall2018 and price2018 do",
)
SAMPLING_DAYS = _paper(
    (2, 4, 8),
    "Samples for further analysis were taken on Days 2, 4, and 8 of cultivation, with "
    "the sample of initial culture (Day 0) used as a control.",
    page=f"{_METHODS}, 'Cultivation of transposon mutant library'",
    note="the sampling day is stored as duration_hours = 24 x day; the source states "
    "days, not hours of the day",
)
REPLICATE_DESIGN = _paper(
    {"water_samples": 3, "technical_replicates": 3},
    "The experiment was performed with three independent lake water samples, each in "
    "three technical replicates.",
    page=f"{_METHODS}, 'Cultivation of transposon mutant library'",
)
TECHNICAL_REPLICATE_UNIT = _paper(
    SampleUnit.technical_replicate,
    "The biomass from four wells of a 24-well plate, each containing 2.5\xa0ml, was "
    "combined together to form one technical replicate.",
    page=f"{_METHODS}, 'Cultivation of transposon mutant library'",
)

# The statistic ------------------------------------------------------------------ #
BARCODE_FITNESS = _paper(
    MeasurementType.log2_ratio,
    "Briefly, we first calculated the barcode fitness, defined as the log2 ratio of "
    "barcode counts between the treatment sample (after the incubation in the lake "
    "water) and the control sample (the initial culture).",
    page=f"{_METHODS}, 'RB-TnSeq data analysis'",
)
GENE_FITNESS_WEIGHTS = _paper(
    "inverse Poisson strain variance",
    "Gene fitness was then calculated as a weighted average of the fitnesses of the "
    "barcodes belonging to the gene, with the weight being inversely proportional to "
    "the Poisson noise-based strain variance.",
    page=f"{_METHODS}, 'RB-TnSeq data analysis'",
)
CHROMOSOME_NORMALIZATION = _paper(
    {"running_median_window": 251, "mode_subtracted": True},
    "the running median of the gene fitness values (window size 251) across the "
    "chromosome and the mode of the gene fitness values were subtracted from the gene "
    "fitness values.",
    page=f"{_METHODS}, 'RB-TnSeq data analysis'",
)
CENTERED = _paper(
    0.0,
    "This normalization removes local biases and ensures that the gene fitness "
    "distribution is centred around zero.",
    page=f"{_METHODS}, 'RB-TnSeq data analysis'",
    note="the reference phenotype is therefore the typical gene of the same "
    "environment, response 0",
)
AVERAGED = _paper(
    "mean over technical replicates",
    "Gene fitness values were then averaged across the technical replicates.",
    page=f"{_METHODS}, 'RB-TnSeq data analysis'",
)
TABLE_S4_IS_THE_RELEASE = _paper(
    "Supplementary Table S4",
    "The full table of the gene fitness data is available as the Supplementary Table "
    "S4.",
    page="Data availability",
)
PMID_STATEMENT = _paper(PAPER_PMID, f"PMID: {PAPER_PMID}", page="Article information")

# The water ------------------------------------------------------------------------ #
WATER_SITE = _paper(
    "a man-made urban lake near Marburg, Germany",
    "It is a man-made urban lake that is used for rainwater collection with no known "
    "bacterial/chemical contamination.",
    page=f"{_METHODS}, 'Water sampling'",
)
WATER_RECORDED = _paper(
    ("sampling time", "air temperature", "water temperature", "pH"),
    "The sampling times, temperature of air and water, and pH were recorded "
    "(Supplementary Table S2).",
    page=f"{_METHODS}, 'Water sampling'",
)
WATER_FILTRATION = _paper(
    {"filter": "Steritop", "pore_size_um": 0.22},
    "Water samples were either used in their original state or filter-sterilized "
    "through the Steritop filter unit (Merck Millipore, Darmstadt, Germany) with a pore "
    "size of 0.22\xa0μm.",
    page=f"{_METHODS}, 'Water sampling'",
)
NON_FILTERED_PURPOSE = _paper(
    "the autochthonous microbiota stays in non-filtered water",
    "Samples were either filter-sterilized or left non-filtered to assess the impact "
    "of the autochthonous microbiota.",
    page="Results, 'Experimental setup to test the growth and survival of E. coli in "
    "lake water'",
)
THREE_WATER_SAMPLES = _paper(
    3,
    "The RB-TnSeq library cultures were incubated in three water samples collected at "
    "different times (also different from the samples used in the control "
    "experiments).",
    page="Results, 'Experimental setup to test the growth and survival of E. coli in "
    "lake water'",
)
WATER_SAMPLE_NUMBERING = _paper(
    (1, 2, 3),
    "PCA of changes in the abundance of individual gene mutations during incubation of "
    "E. coli in different indicated samples of filtered or non-filtered lake water "
    "#1(A), #2 (B), and #3 (C) (see Supplementary Table S2)",
    page="Figure 2 legend",
    note="the RB-TnSeq water samples are Table S2's #1 to #3 (the PCA of Fig. 2 is "
    "computed from Table S4, whose 'Lake water sample' column is 1 to 3); #4 and #5 "
    "are the control experiments of Fig. 1",
)
QC_OMITTED = _derived(
    "sample_10",
    "One sample that failed the QC is omitted from the further analysis and is "
    "highlighted in red.",
    relpath=WORKBOOK_TEXT_RELPATH,
    sha256=WORKBOOK_TEXT_SHA256,
    page="Supplementary Table S3, LEGEND sheet",
    note="the one Table S3 row filled red is sample_10 (biological replicate 1, day 4, "
    "filtered water, technical replicate 1), read off the cell fills by "
    "highlighted_rows",
)
QC_OMITTED_ROW = _derived(
    ("sample_10", 1, 4, "filtered water", 1),
    "sample_10\t1\t4\tfiltered water\t1",
    relpath=WORKBOOK_TEXT_RELPATH,
    sha256=WORKBOOK_TEXT_SHA256,
    page=f"Supplementary Table S3, DATA rows filled {QC_HIGHLIGHT_FILL}",
    note="sample id, biological replicate, sampling day, sample type, technical "
    "replicate; so lake water #1, filtered, day 4 is a mean of two technical "
    "replicates",
)
TABLE_S4_LEGEND = _derived(
    "gene fitness values per water sample",
    "Gene fitness values, principal component values, and Euclidean distance values "
    "calculated for the whole dataset.",
    relpath=WORKBOOK_TEXT_RELPATH,
    sha256=WORKBOOK_TEXT_SHA256,
    page="Supplementary Table S4, LEGEND sheet",
    note=None,
)

# The paper's statements about named genes (the L4 concordance rows) -------------- #
TOL_PAL_GENES = _paper(
    ("tolA", "pal", "lpp"),
    "with its most prominent representatives being mutations in genes encoding "
    "components of the Tol-Pal system (tolA, pal, and lpp).",
    page="Results, 'RB-TnSeq reveals two major groups of differentially enriched "
    "mutations in lake water'",
)
TOL_PAL_DEPLETED = _paper(
    "negative in filtered and non-filtered water",
    "This group represents mutants that have negative gene fitness values in "
    "non-filtered and filtered water (Supplementary Table S4) and thus have been "
    "strongly depleted.",
    page="Results, 'RB-TnSeq reveals two major groups of differentially enriched "
    "mutations in lake water'",
)
RPOS_GROUP_GENES = _paper(
    ("rhoL", "rpoS", "rbsR"),
    "The strongest selection within this group between the samples was observed for "
    "the rhoL, rpoS, and rbsR genes.",
    page="Results, 'RB-TnSeq reveals two major groups of differentially enriched "
    "mutations in lake water'",
)
RPOS_GROUP_FILTERED_ADVANTAGE = _paper(
    "positive in filtered water, not in non-filtered water",
    "These mutants show a fitness advantage (i.e. positive fitness values) in filtered "
    "water, but not in non-filtered water (Supplementary Fig. S1 and Supplementary "
    "Table S4).",
    page="Results, 'RB-TnSeq reveals two major groups of differentially enriched "
    "mutations in lake water'",
)
GLP_NON_FILTERED_ADVANTAGE = _paper(
    ("glpK", "glpF"),
    "several mutations were specifically beneficial in non-filtered water (glpK, glpF; "
    "highlighted in red in the Quadrant III of Fig. 2)",
    page="Results, 'RB-TnSeq reveals two major groups of differentially enriched "
    "mutations in lake water'",
)


# --------------------------------------------------------------------------- #
# Table S2: the water samples
# --------------------------------------------------------------------------- #
class WaterSample(BaseModel):
    """One lake water sample of Table S2, as its layout-text row states it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int
    date_verbatim: str = Field(description="Table S2's date, day.month.year as printed")
    sampled_on: date
    ph: float
    air_temperature_c: float
    water_temperature_c: float
    row: str = Field(description="the layout-text row, verbatim (the quote)")


_S2_ROW: Final = re.compile(
    r"^\s*(\d{1,2})\.(\d{2})\.(\d{4})\s+#(\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)
_S2_HEADER: Final = (
    "Date          Water sample    pH    Air temperature, °C   Water temperature, °C"
)


def parse_table_s2(text: str) -> tuple[WaterSample, ...]:
    """Every water sample row of Table S2's layout text, in table order.

    Refuses a text whose header is not Table S2's or whose numbered rows are not #1 to
    #5 in order. The date is printed day.month.year (``19.06.2022`` has no month 19).
    """
    if _S2_HEADER not in text:
        raise ValueError("Table S2 layout text has no header row")
    samples: list[WaterSample] = []
    for line in text.splitlines():
        match = _S2_ROW.match(line)
        if match is None:
            continue
        day, month, year, number, ph, air, water = match.groups()
        samples.append(
            WaterSample(
                number=int(number),
                date_verbatim=f"{day}.{month}.{year}",
                sampled_on=date(int(year), int(month), int(day)),
                ph=float(ph),
                air_temperature_c=float(air),
                water_temperature_c=float(water),
                row=line.strip(),
            )
        )
    if [s.number for s in samples] != [1, 2, 3, 4, 5]:
        raise ValueError(f"Table S2 rows are {[s.number for s in samples]}, not #1-#5")
    return tuple(samples)


#: The water samples Table S4 carries (WATER_SAMPLE_NUMBERING).
RBTNSEQ_WATER_SAMPLES: Final = (1, 2, 3)


def water_sample_value(sample: WaterSample) -> SourcedValue:
    """The Table S2 row of one water sample, bound to its layout-text quote."""
    return _derived(
        {
            "water_sample": sample.number,
            "sampled_on": sample.sampled_on.isoformat(),
            "pH": sample.ph,
            "air_temperature_c": sample.air_temperature_c,
            "water_temperature_c": sample.water_temperature_c,
        },
        sample.row,
        relpath=TABLE_S2_TEXT_RELPATH,
        sha256=TABLE_S2_TEXT_SHA256,
        page="Supplementary Table S2, 'Parameters of taken lake water samples'",
        note="the temperatures are of the lake at sampling, not of the incubation",
    )


# --------------------------------------------------------------------------- #
# Table S3: the sequencing samples and the replicate count behind each mean
# --------------------------------------------------------------------------- #
SampleType = Literal["control", "filtered water", "non-filtered water"]
Filtration = Literal["filtered", "non-filtered"]


class SequencingSample(BaseModel):
    """One Table S3 row: a sequenced sample and whether the legend's red marks it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    water_sample: int = Field(description="Table S3 'biological replicate'")
    day: int
    sample_type: SampleType
    technical_replicate: int
    qc_failed: bool = Field(description="the row carries the red QC-failure fill")


_S3_HEADER: Final = (
    "sample id",
    "biological replicate",
    "sampling day",
    "sample type",
    "technical replicate",
)


def read_table_s3(path: str | Path) -> tuple[SequencingSample, ...]:
    """Every Table S3 sample, with the QC failure read off the legend's red fill."""
    failed = {str(row[0]) for row in highlighted_rows(path)}
    book = _workbook(path, read_only=True)
    rows = list(book["DATA"].iter_rows(values_only=True))
    book.close()
    if tuple(rows[0][:5]) != _S3_HEADER:
        raise ValueError(f"Table S3 header is {rows[0][:5]}")
    return tuple(
        SequencingSample(
            sample_id=str(row[0]),
            water_sample=int(row[1]),
            day=int(row[2]),
            sample_type=row[3],
            technical_replicate=int(row[4]),
            qc_failed=str(row[0]) in failed,
        )
        for row in rows[1:]
        if row[0] is not None
    )


#: Measured on the deposited bytes on 2026-10-12: 63 samples, and the one QC failure.
TABLE_S3_SAMPLES: Final = 63
QC_FAILED_SAMPLES: Final = ("sample_10",)


def replicate_counts(
    samples: Sequence[SequencingSample],
) -> dict[tuple[int, Filtration, int], int]:
    """``(water sample, filtration, day) -> technical replicates kept`` (QC passed).

    The day-0 controls are not a condition of their own and are left out.
    """
    counts: Counter[tuple[int, Filtration, int]] = Counter()
    for sample in samples:
        if sample.sample_type == "control" or sample.qc_failed:
            continue
        filtration: Filtration = (
            "filtered" if sample.sample_type == "filtered water" else "non-filtered"
        )
        counts[(sample.water_sample, filtration, sample.day)] += 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# Table S4: the released values
# --------------------------------------------------------------------------- #
class FitnessColumn(BaseModel):
    """One of Table S4's six gene fitness columns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(description="0-based column index in the DATA sheet")
    filtration: Filtration
    day: int

    @property
    def label(self) -> str:
        """``'<filtration>, day <d>'``, the header as the inventory names it."""
        return f"{self.filtration}, day {self.day}"


#: Table S4's (filtration, day) per fitness column, left to right.
_COLUMN_ORDER: Final[tuple[tuple[Filtration, int], ...]] = (
    ("filtered", 2),
    ("filtered", 4),
    ("filtered", 8),
    ("non-filtered", 2),
    ("non-filtered", 4),
    ("non-filtered", 8),
)
FITNESS_COLUMNS: Final = tuple(
    FitnessColumn(index=1 + i, filtration=filtration, day=day)
    for i, (filtration, day) in enumerate(_COLUMN_ORDER)
)


class TableS4Row(BaseModel):
    """One Table S4 row: a gene in one water sample, its six values and its PCA."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    water_sample: int
    symbol: str
    description: str | None
    fitness: tuple[float, ...] = Field(description="in FITNESS_COLUMNS order")
    pc1: float
    pc2: float
    euclidean_distance: float


def read_table_s4(path: str | Path) -> tuple[TableS4Row, ...]:
    """Every Table S4 row, after :func:`table_s4_inventory` has checked the layout.

    The header must name the six columns exactly as :data:`FITNESS_COLUMNS`; a row with
    an empty fitness cell is refused (the inventory measured none).
    """
    inventory = table_s4_inventory(path)
    if inventory.fitness_columns != tuple(c.label for c in FITNESS_COLUMNS):
        raise ValueError(f"Table S4 fitness columns are {inventory.fitness_columns}")
    book = _workbook(path, read_only=True)
    body = list(book["DATA"].iter_rows(values_only=True))[_S4_HEADER_ROWS:]
    book.close()
    out: list[TableS4Row] = []
    for row in body:
        values = [row[c.index] for c in FITNESS_COLUMNS]
        if any(v is None for v in values):
            raise ValueError(
                f"Table S4 row {row[_S4_SYMBOL]} has an empty fitness cell"
            )
        out.append(
            TableS4Row(
                water_sample=int(row[0]),
                symbol=str(row[_S4_SYMBOL]),
                description=None if row[11] is None else str(row[11]),
                fitness=tuple(float(v) for v in values),
                pc1=float(row[8]),
                pc2=float(row[9]),
                euclidean_distance=float(row[10]),
            )
        )
    return tuple(out)


class PcaReproduction(BaseModel):
    """Table S4's released PC1, PC2 and distance against a recomputation, per sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    water_sample: int
    n_genes: int
    max_abs_pc1: float
    max_abs_pc2: float
    max_abs_distance: float


def reproduce_pca(rows: Sequence[TableS4Row]) -> tuple[PcaReproduction, ...]:
    """Recompute each water sample's PCA from its six fitness columns.

    Centred and scaled to unit sample variance (ddof 1), scores on the first two right
    singular vectors, each compared to the released PC up to sign; the distance is
    ``sqrt(PC1^2 + PC2^2)`` of the released PCs. Measured on 2026-10-12 to reproduce the
    release to 1e-13, so this checks that the six stored values of every row are the six
    the authors' PCA read.
    """
    out: list[PcaReproduction] = []
    for k in sorted({r.water_sample for r in rows}):
        sample = [r for r in rows if r.water_sample == k]
        x = np.array([r.fitness for r in sample])
        released = np.array([[r.pc1, r.pc2] for r in sample])
        y = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        _, _, vt = np.linalg.svd(y, full_matrices=False)
        scores = y @ vt.T[:, :2]
        gaps = [
            min(
                float(np.abs(scores[:, j] - released[:, j]).max()),
                float(np.abs(-scores[:, j] - released[:, j]).max()),
            )
            for j in range(2)
        ]
        distance = np.sqrt((released**2).sum(axis=1))
        out.append(
            PcaReproduction(
                water_sample=k,
                n_genes=len(sample),
                max_abs_pc1=gaps[0],
                max_abs_pc2=gaps[1],
                max_abs_distance=float(
                    np.abs(
                        distance - np.array([r.euclidean_distance for r in sample])
                    ).max()
                ),
            )
        )
    return tuple(out)


#: The tolerance the release's PCA must be reproduced to.
PCA_TOLERANCE: Final = 1e-9


# --------------------------------------------------------------------------- #
# The host, the media and the environments
# --------------------------------------------------------------------------- #
REFERENCE_STRAIN: Final = "W3110"
HOST_STRAIN_NAME: Final = str(STRAIN.value)
GENE_NAMESPACE: Final = STRAIN_GENE_NAMESPACES[REFERENCE_STRAIN]


def host_background() -> StrainBackground:
    """W3110 RpoS+ pinned to the W3110 set, with the restoring allele unstated.

    ``alleles`` is empty because the paper writes only "RpoS+" and "the restored RpoS
    activity" (it cites ref. 14 for the strain); that is the source's silence, not a
    claim that the strain equals the sequenced W3110.
    """
    return StrainBackground(
        name=HOST_STRAIN_NAME,
        reference_strain=REFERENCE_STRAIN,
        assembly_set="ecoli_K12_W3110_ASM1024v1",
        parents=None,
        construction=str(BACKGROUND_CONSTRUCTION.value),
        genotype_statement="RpoS+",
        alleles=[],
        provenance=[STRAIN, BACKGROUND_CONSTRUCTION],
    )


def host_reference(data_root: str | None = None) -> ReferenceGenome:
    """W3110 RpoS+ on ``ecoli_K12_W3110_ASM1024v1``, the RefSeq accession pinned."""
    return assembly_reference(
        REFERENCE_STRAIN, background=host_background(), data_root=data_root
    )


def lake_water_medium(sample: WaterSample, filtration: Filtration) -> Media:
    """One water sample, filter-sterilized or not, derived from the library's
    ``LAKE_WATER``.

    The water is the base's one component, ``intrinsically_undefined``: the paper reports
    no chemical or nutrient analysis of it, and a natural water is batch-variable by
    nature. Its sampling date, pH and lake temperatures are the Table S2 row
    (:func:`water_sample_value`). Non-filtered water keeps the lake's microbiota, which
    is the contrast the two media exist for (:data:`NON_FILTERED_PURPOSE`).
    """
    filtered = filtration == "filtered"
    state = "filter-sterilized 0.22 um" if filtered else "non-filtered"
    return Media(
        name=f"lake water #{sample.number} (sampled {sample.sampled_on.isoformat()}), "
        f"{state} (Teteneva 2024)",
        state="liquid",
        is_synthetic=False,
        base_medium="LAKE_WATER",
        components=list(LAKE_WATER.components),
        provenance=[
            water_sample_value(sample),
            WATER_SAMPLE_NUMBERING,
            WATER_FILTRATION if filtered else NON_FILTERED_PURPOSE,
        ],
    )


TEMPERATURE_GAP: Final = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=INCUBATION.provenance,
    note="the cultures were 'incubated without antibiotics at room temperature'; no "
    "value is stated. Table S2's air and water temperatures are of the lake at "
    "sampling, not of the incubation, so neither is carried here",
)
UNCERTAINTY_GAP: Final = ProvenanceGap(
    field="environment_response_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=TABLE_S4_RELPATH,
        citation_key=CITATION_KEY,
        sha256=TABLE_S4_SHA256,
        method="Supplementary Table S4 (DATA and LEGEND sheets) and the paper's "
        "Materials and methods",
        page="Supplementary Table S4",
    ),
    note="Table S4 releases the mean over technical replicates only: no SD, SE, t or "
    "p-value, and the per-replicate values are not released",
)


class Condition(BaseModel):
    """One of the 18 environments: a water sample, its filtration and the sampling day."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    water_sample: int
    filtration: Filtration
    day: int
    column: int = Field(description="index into TableS4Row.fitness")
    n_technical_replicates: int

    @property
    def screen_id(self) -> str:
        """``'lake water #<k>, <filtration>, day <d>'``."""
        return f"lake water #{self.water_sample}, {self.filtration}, day {self.day}"


def conditions(
    replicates: Mapping[tuple[int, Filtration, int], int],
) -> tuple[Condition, ...]:
    """The 18 conditions in water-sample then Table S4 column order.

    Every one must have at least one QC-passed technical replicate in Table S3.
    """
    out: list[Condition] = []
    for k in RBTNSEQ_WATER_SAMPLES:
        for j, column in enumerate(FITNESS_COLUMNS):
            key = (k, column.filtration, column.day)
            if replicates.get(key, 0) < 1:
                raise ValueError(f"no QC-passed technical replicate for {key}")
            out.append(
                Condition(
                    water_sample=k,
                    filtration=column.filtration,
                    day=column.day,
                    column=j,
                    n_technical_replicates=replicates[key],
                )
            )
    return tuple(out)


def build_environment(condition: Condition, sample: WaterSample) -> Environment:
    """The lake water, the sampling day in hours, and the temperature gap."""
    if sample.number != condition.water_sample:
        raise ValueError(f"{condition.screen_id} given water sample #{sample.number}")
    return Environment(
        media=lake_water_medium(sample, condition.filtration),
        temperature=None,
        duration_hours=24.0 * condition.day,
        provenance_gaps=[TEMPERATURE_GAP],
    )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
UNITS: Final = (
    "RB-TnSeq gene fitness, Teteneva 2024 Supplementary Table S4: log2 ratio of a "
    "gene's barcode counts after incubation in the lake water over the day-0 culture, "
    "normalized to total barcodes, weighted over the gene's barcodes (inverse Poisson "
    "strain variance), the chromosome running median (window 251) and the mode "
    "subtracted, then averaged over the technical replicates; the typical gene is 0"
)
UNITS_REFERENCE: Final = (
    "the typical gene of the same environment (the gene fitness distribution is "
    "centred on zero by the normalization)"
)
DESCRIPTION: Final = (
    "Randomly barcoded Tn5 insertion in a coding region, a gene-level call over the "
    "gene's barcoded insertion mutants (no per-strain barcode or mapped site released). "
    "Keyed by the gene symbol Supplementary Table S4 releases, verbatim: the release "
    "names no locus tag and no annotation lookup is made"
)
#: What ``systematic_gene_name`` is, as the source writes it: Table S4's column header.
IDENTIFIER_KIND: Final = "Gene name"


def publication() -> Publication:
    """This paper, by PubMed id and DOI."""
    return Publication(
        pubmed_id=PAPER_PMID,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PAPER_PMID}/",
        doi=PAPER_DOI,
        doi_url=f"https://doi.org/{PAPER_DOI}",
    )


def build_genotype(symbol: str) -> Genotype:
    """A Tn5 insertion in the gene Table S4 names, stored as released: the symbol is the
    identifier, the gene name and the strain label (a pooled release labels its row by
    the gene alone), the kind and cassette are the Methods'.
    """
    return Genotype(
        perturbations=[
            ReleasedIdentifierPerturbation(
                systematic_gene_name=symbol,
                perturbed_gene_name=symbol,
                gene_namespace=GENE_NAMESPACE,
                strain_label=symbol,
                identifier_kind=IDENTIFIER_KIND,
                released_kind=ReleasedKind.transposon_insertion,
                cassette=str(CASSETTE.value),
                description=DESCRIPTION,
            )
        ]
    )


def build_phenotype(value: float, condition: Condition) -> EnvironmentResponsePhenotype:
    """One gene's mean fitness in one condition, with its technical replicate count."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=value,
        n_samples=condition.n_technical_replicates,
        sample_unit=SampleUnit.technical_replicate,
        screen_id=condition.screen_id,
        units=UNITS,
        provenance_gaps=[UNCERTAINTY_GAP],
    )


def build_reference(
    dataset_name: str,
    environment: Environment,
    genome_reference: ReferenceGenome,
    condition: Condition,
) -> EnvironmentResponseExperimentReference:
    """The typical gene of the same environment: fitness 0 by the normalization."""
    return EnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=environment,
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            screen_id=condition.screen_id,
            units=UNITS_REFERENCE,
        ),
    )


class ConditionContext(BaseModel):
    """A condition with the environment and reference all its records share."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    condition: Condition
    environment: Environment
    reference: EnvironmentResponseExperimentReference


def condition_contexts(
    dataset_name: str,
    conds: Sequence[Condition],
    water: Mapping[int, WaterSample],
    genome_reference: ReferenceGenome,
) -> tuple[ConditionContext, ...]:
    """Each condition's environment and reference, built once."""
    out: list[ConditionContext] = []
    for condition in conds:
        environment = build_environment(condition, water[condition.water_sample])
        out.append(
            ConditionContext(
                condition=condition,
                environment=environment,
                reference=build_reference(
                    dataset_name, environment, genome_reference, condition
                ),
            )
        )
    return tuple(out)


def iter_records(
    dataset_name: str, contexts: Sequence[ConditionContext], rows: Sequence[TableS4Row]
) -> Iterator[
    tuple[
        EnvironmentResponseExperiment,
        EnvironmentResponseExperimentReference,
        Publication,
    ]
]:
    """Every record: condition by condition, then Table S4 row order. Every released
    value is a record.
    """
    genotypes = {symbol: build_genotype(symbol) for symbol in {r.symbol for r in rows}}
    paper = publication()
    for context in contexts:
        condition = context.condition
        for row in rows:
            if row.water_sample != condition.water_sample:
                continue
            yield (
                EnvironmentResponseExperiment(
                    dataset_name=dataset_name,
                    genotype=genotypes[row.symbol],
                    environment=context.environment,
                    phenotype=build_phenotype(row.fitness[condition.column], condition),
                ),
                context.reference,
                paper,
            )


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
class Inventory(BaseModel):
    """The value accounting the dendron note states: every released value is stored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table_s4_rows: int
    rows_per_water_sample: dict[int, int]
    released_values: int
    conditions: int
    replicates_per_condition: dict[str, int]
    symbols: int
    stored_records: int


def inventory(rows: Sequence[TableS4Row], conds: Sequence[Condition]) -> Inventory:
    """Count rows, conditions and records; the records must equal the released values."""
    stored = sum(1 for c in conds for r in rows if r.water_sample == c.water_sample)
    released = len(rows) * len(FITNESS_COLUMNS)
    if stored != released:
        raise RuntimeError(f"{stored} records for {released} released values")
    return Inventory(
        table_s4_rows=len(rows),
        rows_per_water_sample=dict(
            sorted(Counter(r.water_sample for r in rows).items())
        ),
        released_values=released,
        conditions=len(conds),
        replicates_per_condition={c.screen_id: c.n_technical_replicates for c in conds},
        symbols=len({r.symbol for r in rows}),
        stored_records=stored,
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
DATASET_ROOT_REL: Final = "data/torchcell/rbtnseq_teteneva2024_ecoli"

#: ``raw/`` name -> (raw-mirror path, pinned sha256): every file ``process`` reads.
LOADER_FILES: Final[dict[str, tuple[str, str]]] = {
    "table_s4.xlsx": (TABLE_S4_RELPATH, TABLE_S4_SHA256),
    "table_s3.xlsx": (TABLE_S3_RELPATH, TABLE_S3_SHA256),
    "table_s2.layout.txt": (TABLE_S2_TEXT_RELPATH, TABLE_S2_TEXT_SHA256),
}

#: What the build writes: every one of Table S4's 66,162 values (11,027 rows x 6).
EXPECTED_RECORDS: Final = 66_162
#: Records the environment-perturbed rule counts as unperturbed: every record of the
#: modal medium. Each record is a response to lake water against the day-0 culture;
#: the dataset has no non-lake-water condition, so the rule's elected baseline is one of
#: the six lake water media. Water #2 holds the most rows (3,679), and either of its two
#: media is the mode: 3 days x 3,679 = 11,037 records, declared.
EXPECTED_UNPERTURBED: Final = 11_037


class LoaderInputs(BaseModel):
    """Everything ``process`` reads, parsed."""

    model_config = ConfigDict(frozen=True)

    rows: tuple[TableS4Row, ...]
    samples: tuple[SequencingSample, ...]
    water: dict[int, WaterSample]
    conditions: tuple[Condition, ...]


def read_loader_inputs(paths: Mapping[str, str | Path]) -> LoaderInputs:
    """Parse the three loader files."""
    rows = read_table_s4(paths["table_s4.xlsx"])
    samples = read_table_s3(paths["table_s3.xlsx"])
    water = {
        s.number: s
        for s in parse_table_s2(Path(paths["table_s2.layout.txt"]).read_text())
    }
    return LoaderInputs(
        rows=rows,
        samples=samples,
        water=water,
        conditions=conditions(replicate_counts(samples)),
    )


@register_dataset
class RbTnseqTeteneva2024Dataset(ExperimentDataset):
    """Teteneva 2024, E. coli K-12 W3110 RpoS+ in lake water: Table S4 gene fitness."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[Literal["W3110"]] = "W3110"

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; no genome is read, the records keep Table S4's symbols."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return EnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return EnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Tables S4 and S3 and Table S2's layout text, linked by ``download``."""
        return list(LOADER_FILES)

    def download(self) -> None:
        """Link each pinned raw-mirror file into ``raw/`` after verifying its sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        for name, (relpath, pin) in LOADER_FILES.items():
            link_verified(raw_path(relpath), osp.join(self.raw_dir, name), pin)
        log.info("Teteneva 2024 raw files linked into %s", self.raw_dir)

    @post_process
    def process(self) -> None:
        """Build one record per (Table S4 row, condition) and the preprocess reports."""
        data_root = str(_data_root(None))
        verify_raw_files(
            self.raw_dir, {name: pin for name, (_, pin) in LOADER_FILES.items()}
        )
        inputs = read_loader_inputs(
            {name: osp.join(self.raw_dir, name) for name in LOADER_FILES}
        )
        contexts = condition_contexts(
            self.name, inputs.conditions, inputs.water, host_reference(data_root)
        )
        counts = inventory(inputs.rows, inputs.conditions)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        txn = env.begin(write=True)
        itxn = interned_env.begin(write=True)
        for experiment, reference, paper in tqdm(
            iter_records(self.name, contexts, inputs.rows),
            total=counts.stored_records,
            desc="Teteneva 2024",
        ):
            txn.put(
                f"{idx}".encode(),
                self._intern_record(experiment, reference, paper, itxn),
            )
            idx += 1
        itxn.commit()
        txn.commit()
        env.close()
        interned_env.close()
        if idx != counts.stored_records:
            raise RuntimeError(
                f"wrote {idx} records, inventory says {counts.stored_records}"
            )
        payloads: dict[str, Any] = {
            "inventory.json": counts.model_dump(mode="json"),
            "conditions.json": [c.model_dump(mode="json") for c in inputs.conditions],
            "sequencing_samples.json": [
                s.model_dump(mode="json") for s in inputs.samples
            ],
        }
        for name, payload in payloads.items():
            with open(osp.join(self.preprocess_dir, name), "w") as handle:
                json.dump(payload, handle, indent=2)
        log.info("Wrote %d Teteneva 2024 records", idx)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "Teteneva 2024 builds its records in process(); see iter_records"
        )


# --------------------------------------------------------------------------- #
# Verification (L0 to L4)
# --------------------------------------------------------------------------- #
VERIFY_PROVENANCE: Final = Provenance(
    source_uri=TABLE_S4_RELPATH,
    citation_key=CITATION_KEY,
    sha256=TABLE_S4_SHA256,
    method=(
        "Supplementary Table S4 gene fitness (log2 ratio against the day-0 culture, "
        "mean over technical replicates) per lake water sample, filtration and day, "
        "every value keyed by the released gene symbol verbatim; n_samples from Table "
        "S3 with the QC-failed sample_10 omitted"
    ),
    page="Supplementary Tables S3 and S4 (si/)",
    retrieved="2026-10-12",
)


class StoredValue(BaseModel):
    """The cells of one stored record the supplementary rows read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    screen_id: str
    symbol: str
    perturbed_gene_name: str
    perturbation_type: str
    identifier_kind: str
    strain_label: str
    gene_namespace: str
    value: float
    n_samples: int | None


def stored_values(records: Iterable[Mapping[str, Any]]) -> list[StoredValue]:
    """One :class:`StoredValue` per record."""
    out: list[StoredValue] = []
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        perturbations = record["experiment"]["genotype"]["perturbations"]
        if len(perturbations) != 1:
            raise ValueError(f"expected one perturbation, got {len(perturbations)}")
        leaf = perturbations[0]
        out.append(
            StoredValue(
                screen_id=str(phenotype["screen_id"]),
                symbol=str(leaf["systematic_gene_name"]),
                perturbed_gene_name=str(leaf["perturbed_gene_name"]),
                perturbation_type=str(leaf["perturbation_type"]),
                identifier_kind=str(leaf["identifier_kind"]),
                strain_label=str(leaf["strain_label"]),
                gene_namespace=str(leaf["gene_namespace"]),
                value=float(phenotype["environment_response"]),
                n_samples=phenotype["n_samples"],
            )
        )
    return out


def stored_symbols_are_released(
    values: Sequence[StoredValue], rows: Sequence[TableS4Row]
) -> LevelResult:
    """SUPPLEMENTARY L1: every stored gene identifier is a Table S4 symbol, verbatim.

    Each record is a released-strain leaf keyed by the symbol as released (the same
    string as its ``perturbed_gene_name`` and ``strain_label``, ``identifier_kind``
    'Gene name', the W3110 host namespace), and the stored symbols are exactly the
    release's.
    """
    released = {r.symbol for r in rows}
    stored = {v.symbol for v in values}
    off = sorted(
        {
            v.symbol
            for v in values
            if v.perturbation_type != "released_identifier"
            or v.identifier_kind != IDENTIFIER_KIND
            or v.strain_label != v.symbol
            or v.perturbed_gene_name != v.symbol
            or v.gene_namespace != GENE_NAMESPACE
        }
    )
    passed = not off and stored == released
    return LevelResult(
        level=Level.L1,
        name="stored_gene_identifiers_are_the_released_symbols",
        passed=passed,
        message=(
            f"SUPPLEMENTARY: {len(stored)} stored symbols against {len(released)} "
            f"released; {len(stored - released)} not released, "
            f"{len(released - stored)} released and not stored, {len(off)} not stored "
            "as a verbatim gene symbol of the W3110 host"
        ),
        details={
            "not_released": sorted(stored - released)[:20],
            "not_stored": sorted(released - stored)[:20],
            "not_verbatim": off[:20],
        },
    )


def stored_values_equal_release(
    values: Sequence[StoredValue],
    rows: Sequence[TableS4Row],
    conds: Sequence[Condition],
    expected: int,
) -> LevelResult:
    """SUPPLEMENTARY L2: every stored value IS its Table S4 cell, and every n_samples is
    its condition's QC-passed replicate count from Table S3.
    """
    by_screen = {c.screen_id: c for c in conds}
    cells = {(r.water_sample, r.symbol): r.fitness for r in rows}
    wrong_value: list[tuple[str, str]] = []
    wrong_n: list[tuple[str, str]] = []
    for v in values:
        condition = by_screen[v.screen_id]
        if cells[(condition.water_sample, v.symbol)][condition.column] != v.value:
            wrong_value.append((v.screen_id, v.symbol))
        if v.n_samples != condition.n_technical_replicates:
            wrong_n.append((v.screen_id, v.symbol))
    n_by_screen = Counter(v.screen_id for v in values)
    passed = not wrong_value and not wrong_n and len(values) == expected
    return LevelResult(
        level=Level.L2,
        name="stored_values_are_table_s4_cells",
        passed=passed,
        message=(
            f"SUPPLEMENTARY: {len(values)} stored values (expected {expected}); "
            f"{len(wrong_value)} differ from their Table S4 cell, {len(wrong_n)} carry "
            "an n_samples other than their condition's Table S3 replicate count"
        ),
        details={
            "wrong_value": wrong_value[:20],
            "wrong_n_samples": wrong_n[:20],
            "records_per_condition": dict(sorted(n_by_screen.items())),
            "replicates_per_condition": {
                c.screen_id: c.n_technical_replicates for c in conds
            },
        },
    )


def pca_result(reproduction: Sequence[PcaReproduction]) -> LevelResult:
    """SUPPLEMENTARY L4: the release's own PCA is reproduced from its six columns."""
    worst = max(
        max(r.max_abs_pc1, r.max_abs_pc2, r.max_abs_distance) for r in reproduction
    )
    return LevelResult(
        level=Level.L4,
        name="released_pca_reproduced_from_the_fitness_columns",
        passed=worst <= PCA_TOLERANCE and len(reproduction) == 3,
        message=(
            f"SUPPLEMENTARY: Table S4's PC1, PC2 and Euclidean distance recomputed from "
            f"its six fitness columns (centred, unit-variance PCA per water sample) "
            f"for {len(reproduction)} water samples; max abs difference {worst:.3g} "
            f"(tolerance {PCA_TOLERANCE})"
        ),
        details={"per_water_sample": [r.model_dump() for r in reproduction]},
    )


class ClaimCheck(BaseModel):
    """One of the paper's statements about named genes, against the stored values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: str
    quote: str
    genes: tuple[str, ...]
    holds: bool
    failures: tuple[str, ...]


def _value(
    values: Mapping[tuple[str, str], float], symbol: str, k: int, f: str, d: int
) -> float:
    return values[(f"lake water #{k}, {f}, day {d}", symbol)]


def paper_claims(values: Sequence[StoredValue]) -> tuple[ClaimCheck, ...]:
    """The paper's three sign statements about named genes, on the stored values.

    - Tol-Pal (tolA, pal, lpp): negative in every condition of every water sample.
    - rhoL, rpoS, rbsR: positive in filtered water at day 8, and filtered above
      non-filtered on average, in every water sample.
    - glpK, glpF: positive in non-filtered water at day 8, and above filtered at day 8,
      in every water sample.

    These test the filtration labels the records carry, which the PCA cannot see.
    """
    table = {(v.screen_id, v.symbol): v.value for v in values}
    days = (2, 4, 8)
    checks: list[ClaimCheck] = []
    failures = [
        f"{g} #{k} {f} d{d}"
        for g in ("tolA", "pal", "lpp")
        for k in RBTNSEQ_WATER_SAMPLES
        for f in ("filtered", "non-filtered")
        for d in days
        if not _value(table, g, k, f, d) < 0
    ]
    checks.append(
        ClaimCheck(
            claim="tol_pal_negative_everywhere",
            quote=TOL_PAL_DEPLETED.quote,
            genes=tuple(TOL_PAL_GENES.value),
            holds=not failures,
            failures=tuple(failures),
        )
    )
    failures = []
    for g in RPOS_GROUP_GENES.value:
        for k in RBTNSEQ_WATER_SAMPLES:
            filtered = [_value(table, g, k, "filtered", d) for d in days]
            non = [_value(table, g, k, "non-filtered", d) for d in days]
            if not (filtered[-1] > 0 and sum(filtered) > sum(non)):
                failures.append(f"{g} #{k}")
    checks.append(
        ClaimCheck(
            claim="rpos_group_filtered_advantage",
            quote=RPOS_GROUP_FILTERED_ADVANTAGE.quote,
            genes=tuple(RPOS_GROUP_GENES.value),
            holds=not failures,
            failures=tuple(failures),
        )
    )
    failures = [
        f"{g} #{k}"
        for g in GLP_NON_FILTERED_ADVANTAGE.value
        for k in RBTNSEQ_WATER_SAMPLES
        if not (
            _value(table, g, k, "non-filtered", 8) > 0
            and _value(table, g, k, "non-filtered", 8)
            > _value(table, g, k, "filtered", 8)
        )
    ]
    checks.append(
        ClaimCheck(
            claim="glp_non_filtered_advantage",
            quote=GLP_NON_FILTERED_ADVANTAGE.quote,
            genes=tuple(GLP_NON_FILTERED_ADVANTAGE.value),
            holds=not failures,
            failures=tuple(failures),
        )
    )
    return tuple(checks)


def claims_result(checks: Sequence[ClaimCheck]) -> LevelResult:
    """SUPPLEMENTARY L4: the paper's statements about named genes hold on the store."""
    return LevelResult(
        level=Level.L4,
        name="paper_statements_hold_on_the_stored_values",
        passed=all(c.holds for c in checks),
        message=(
            f"SUPPLEMENTARY: {sum(c.holds for c in checks)} of {len(checks)} of the "
            "paper's sign statements about named genes hold on the stored values "
            f"({', '.join(c.claim for c in checks)})"
        ),
        details={"checks": [c.model_dump() for c in checks]},
    )


def verify_build(
    dataset_root: str,
    *,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 verifier on a built tree and write its report.

    The gene universe is the release's own: the 3,691 symbols of Table S4, which the
    records carry verbatim. No genome is read and no name is resolved against an
    annotation (the symbol-to-annotation mapping is a separate artifact,
    ``experiments/database/scripts/teteneva2024_symbol_annotation.py``). Four
    SUPPLEMENTARY rows are added: the stored identifiers are the released symbols (L1),
    every stored value is its Table S4 cell with its Table S3 replicate count (L2), the
    release's PCA is reproduced from the six columns (L4), and the paper's sign
    statements about named genes hold on the store (L4).
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    root = str(_data_root(data_root))
    rows = read_table_s4(raw_path(TABLE_S4_RELPATH, root))
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name=RbTnseqTeteneva2024Dataset.__name__,
        provenance=VERIFY_PROVENANCE,
        expected_count=expected_count,
        sgd_genes={r.symbol for r in rows},
        gene_universe_label="Supplementary Table S4 released gene symbol",
        min_containment=1.0,
        resolve_gene_name=None,
        expected_unperturbed=EXPECTED_UNPERTURBED,
    )
    preprocess = osp.join(dataset_root, "preprocess")
    with open(osp.join(preprocess, "conditions.json")) as handle:
        conds = tuple(Condition.model_validate(c) for c in json.load(handle))
    values = stored_values(stream_records(dataset_root))
    report.add(stored_symbols_are_released(values, rows))
    report.add(stored_values_equal_release(values, rows, conds, expected_count))
    report.add(pca_result(reproduce_pca(rows)))
    report.add(claims_result(paper_claims(values)))
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def loader_report(data_root: str | None = None) -> dict[str, Any]:
    """Every number the dendron note states, recomputed from the pinned files."""
    root = str(_data_root(data_root))
    inputs = read_loader_inputs(
        {name: raw_path(relpath, root) for name, (relpath, _) in LOADER_FILES.items()}
    )
    return {
        "inventory": inventory(inputs.rows, inputs.conditions).model_dump(mode="json"),
        "pca": [r.model_dump() for r in reproduce_pca(inputs.rows)],
        "qc_failed": [s.sample_id for s in inputs.samples if s.qc_failed],
        "water_samples": [s.model_dump(mode="json") for s in inputs.water.values()],
    }


def main(argv: list[str] | None = None) -> None:
    """``record`` the paper text and derived texts, print the ``report``, or ``verify``
    the built dev store (L0 to L4).
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="mirror the paper text, derive texts")
    record.add_argument("--paper-text", default=None, help="a fetched copy to use")
    commands.add_parser("report", help="print the evidence as JSON")
    commands.add_parser("verify", help="verify the built dev store (L0 to L4)")
    args = parser.parse_args(argv)
    if args.command == "record":
        print(record_paper_and_texts(paper_text_source=args.paper_text))
    elif args.command == "report":
        print(json.dumps(loader_report(), indent=2))
    else:
        root = str(_data_root(None))
        print(verify_build(osp.join(root, DATASET_ROOT_REL), data_root=root).summary())


if __name__ == "__main__":
    main()
