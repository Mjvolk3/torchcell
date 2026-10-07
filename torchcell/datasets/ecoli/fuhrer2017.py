# torchcell/datasets/ecoli/fuhrer2017
# [[torchcell.datasets.ecoli.fuhrer2017]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/fuhrer2017
# Test file: tests/torchcell/datasets/ecoli/test_fuhrer2017.py
"""Fuhrer 2017 metabolome of the E. coli Keio deletion collection (FIA-TOF-MS ions).

Fuhrer, Zampieri, Sevin, Sauer and Zamboni 2017 (Mol Syst Biol 13:907,
doi:10.15252/msb.20167150) grew the Keio single-gene deletion strains on glucose
minimal medium with casein hydrolysate at 37 C, harvested them mid-exponentially and
measured 3,169 negative-mode and 4,365 positive-mode ions by flow-injection TOF mass
spectrometry. Each strain's released value per ion is a MODIFIED Z-SCORE summarized
over the strain's replicates; measured at build time, every ion of the released matrix
has median 0 and sample SD 1 over its 3,807 columns (``zscore_scale.json``), so the score
is standardized over exactly the released strains plus ``wt``.

DATA. The per-strain matrices are not in the publisher SI; the paper deposits them in
BioStudies S-BSST5 (``SOURCED_VALUES["data_availability"]``), a directly scriptable
HTTPS tree. The loader consumes ``zscore_neg.tsv`` / ``zscore_pos.tsv`` (rows ions,
columns strains), ``sample_id_zscore.xls`` (the column names), ``sample_id_all.xls``
(the raw matrix's column names, four per strain, read to COUNT replicates),
``neg_ionMz.xls`` / ``pos_ionMz.xls`` (each row's m/z), the study record
``S-BSST5.json`` (the files' own descriptions, read to check the design), and the
publisher's Table EV1 workbook (``si2.xlsx``, captured in the literature mirror), whose
sheet ``Table EV1A`` ties each strain name to its Keio JW id and Blattner number and whose
``Table EV1B`` lists the ions. The raw-intensity matrices and the KEGG annotation files
of S-BSST5 are NOT consumed and NOT mirrored.

STRAIN (``SOURCED_VALUES["background_strain"]``). The paper says the strains come "from
the KEIO knockout collection (Baba et al, 2006)" and never names the background; Baba
2006 (mirrored) does: "E. coli K-12 strain BW25113". Records pin
``assembly_reference("BW25113")`` and the ``ecoli_k12_bw25113_locus_tag`` namespace.

IDENTIFIERS. The deposit labels strains by GENE NAME, and on the BW25113 GenBank
annotation 11 kept names do not resolve to the locus of their own strain's JW id: five
resolve to another locus (``ecpD``, ``rarA``, ``rbn``, ``ykgB``, ``zupT``), four are
ambiguous and two retired (``identifier_reconciliation.json``, written at build time).
The strain's identity is its Keio JW id, which the BW25113 GenBank file carries as a
``gene_synonym``. So each name is joined to Table EV1A, and the JW id goes through
``reconcile_locus_tags`` on the BW25113 genome. A JW id
that annotation does not carry is kept as given by the reconciler and its record is
dropped (rule ``jw_id_not_on_a_bw25113_locus``); no cross-strain (MG1655 b-number
through ECK) mapping is used, because a derived mapping must be recorded on the record
and the deletion leaf has no field for it.

RECORDS DROPPED (rules + items in ``preprocess/dropped_records.json``): a deposit column
whose name matches MORE than one Table EV1A entry (``pooled_keio_entries``: the raw
matrix carries four columns per entry under that one name, so the released value
summarizes two or three Keio strains, 20 of them of different loci), and a JW id not on
a BW25113 locus. The ``wt`` column is the reference, not a record.

PHENOTYPE. ``MetabolitePhenotype`` keyed ``neg_NNNN`` / ``pos_NNNN`` (the 1-based ion
row, which is Table EV1B's ``Ion Index``; ``preprocess/ions.csv`` gives each key's m/z in
both files). ``n_replicates`` = the strain's raw-matrix columns over the two technical
injections: 2 independent clones per strain (``SOURCED_VALUES["n_biological_replicates"]``),
96 cultures for ``wt`` (192 columns, a back-solve; the paper states no WT count).
``metabolite_level_se`` is a typed gap: no per-strain uncertainty is released.
The reference is the measured ``wt`` profile on the same z-score scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import os.path as osp
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.media import M9_GLUCOSE_CASEIN_FUHRER2017
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    BacterialMetaboliteExperiment,
    BacterialMetaboliteExperimentReference,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    MetabolitePhenotype,
    Publication,
    StrainConstruction,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    LOCUS_TAG_PATTERNS,
    STRAIN_GENE_NAMESPACES,
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever
from torchcell.sequence.genome.base import GeneNameResolution
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
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
    audit_sourced_value,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "fuhrerGenomewideLandscapeGene2017"
PAPER_DOI = "10.15252/msb.20167150"
#: The PubMed id the BioStudies S-BSST5 record lists as its publication (accno).
PUBMED_ID = "28093455"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "ec87736bda4d30fa4c5eafa49a5d033bfff4f1ee3d641bbeb4c7f9e5d55f055b"
#: The MSB transaction report and author checklist (Expanded View 6), MinerU OCR.
SI6_MD = "si/si6.md"
SI6_MD_SHA256 = "58a5a9b04775791d2a33fb9d403f47079ddcc989c1124e388c0088ce8703f874"
#: Baba 2006, the Keio construction paper Fuhrer defers the collection to.
BABA_KEY = "babaConstructionEscherichiaColi2006"
BABA_SHA256 = "ca71475baf25d070562c70296d5aad8ec17a59d5b843f264b1e226362b6daf0d"

BIOSTUDIES_ACCESSION = "S-BSST5"
BIOSTUDIES_STUDY_URL = "https://www.ebi.ac.uk/biostudies/studies/S-BSST5"
BIOSTUDIES_ROOT = "https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/S-BSST0-99/S-BSST5"
DATA_RETRIEVED_AT = "2026-10-07"

STUDY_JSON = "S-BSST5.json"
ZSCORE_NEG = "zscore_neg.tsv"
ZSCORE_POS = "zscore_pos.tsv"
SAMPLE_ID_ZSCORE = "sample_id_zscore.xls"
SAMPLE_ID_ALL = "sample_id_all.xls"
NEG_ION_MZ = "neg_ionMz.xls"
POS_ION_MZ = "pos_ionMz.xls"
TABLE_EV1 = "si2.xlsx"
TABLE_EV1A_SHEET = "Table EV1A"
TABLE_EV1B_SHEET = "Table EV1B"
#: Table EV1A's header is its fourth row (three title/blank rows above it).
TABLE_EV1A_HEADER_ROW = 3

IonMode = Literal["neg", "pos"]
ION_MODES: tuple[IonMode, IonMode] = ("neg", "pos")
ZSCORE_FILES: dict[IonMode, str] = {"neg": ZSCORE_NEG, "pos": ZSCORE_POS}
ION_MZ_FILES: dict[IonMode, str] = {"neg": NEG_ION_MZ, "pos": POS_ION_MZ}


class RawFile(BaseModel):
    """One file the loader consumes: its pinned bytes and how they were retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str
    bytes: int
    description: str
    retrieval: RetrievalRecord

    @property
    def mirror_relpath(self) -> str:
        """Path inside the raw mirror (``data/<name>``)."""
        return f"data/{self.name}"


def _biostudies_file(
    name: str, sha256: str, size: int, description: str, *, under_files: bool = True
) -> RawFile:
    url = (
        f"{BIOSTUDIES_ROOT}/Files/{name}"
        if under_files
        else f"{BIOSTUDIES_ROOT}/{name}"
    )
    return RawFile(
        name=name,
        sha256=sha256,
        bytes=size,
        description=description,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.direct_url,
            source_url=url,
            retriever="torchcell.literature.retrieve.direct_url",
            params={"url": url},
            sha256=sha256,
            retrieved_at=DATA_RETRIEVED_AT,
        ),
    )


#: Table EV1 as the literature mirror captured it from the PMC Article Datasets bucket
#: (the retrieval record is copied from that key's ``manifest.json``).
_TABLE_EV1_KEY = "PMC5293155.1/MSB-13-907-s002.xlsx"
_TABLE_EV1_SHA256 = "cda4dc215a2026b0d63eae587fb3573b2b1375e9cfe1ae6de274bda231b27320"

RAW_FILES: tuple[RawFile, ...] = (
    _biostudies_file(
        STUDY_JSON,
        "65e4cb0e93049e14dbe45004ae38d6089d65f105b3d2d8ee95b8bbd8c57f77e4",
        7966,
        "BioStudies S-BSST5 study record: design and per-file descriptions",
        under_files=False,
    ),
    _biostudies_file(
        ZSCORE_NEG,
        "ecb93f2c5ce6854fa66db541cc624eef2afcc8a82a031fa9a992203a060ec8ee",
        90485873,
        "modified z-scores, negative mode; rows ions, columns strains",
    ),
    _biostudies_file(
        ZSCORE_POS,
        "6059a5550ec1e29ae734f5bf01547d83b22eee596fd0fbb6f7ab8c91a2a0f218",
        124636326,
        "modified z-scores, positive mode; rows ions, columns strains",
    ),
    _biostudies_file(
        SAMPLE_ID_ZSCORE,
        "511fc0d05b05d2389ee639f76514a5b4703c33b5f9d3b681da188579cc2fbc01",
        192512,
        "the z-score matrices' column names",
    ),
    _biostudies_file(
        SAMPLE_ID_ALL,
        "b420f178f1648b0d927ab372960728806f6bd4a93c1fa4ab0478389b87f21624",
        622592,
        "the raw matrices' column names (four per strain), read to count replicates",
    ),
    _biostudies_file(
        NEG_ION_MZ,
        "e3722fececd57daffe19186381be09c31d6bcd759b73b164888d19e33366211c",
        154112,
        "accurate m/z of each negative-mode row",
    ),
    _biostudies_file(
        POS_ION_MZ,
        "a9aeb65d7adab18a6f648470cea56f8761c988e1b50ceb10df59f0585f7a4f7f",
        203264,
        "accurate m/z of each positive-mode row",
    ),
    RawFile(
        name=TABLE_EV1,
        sha256=_TABLE_EV1_SHA256,
        bytes=1854096,
        description="Table EV1 (EV1A strains: JW id, Blattner id, gene name, Keio "
        "delivery status; EV1B ions: mode, ion index, m/z)",
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=f"https://pmc-oa-opendata.s3.amazonaws.com/{_TABLE_EV1_KEY}",
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": _TABLE_EV1_KEY},
            sha256=_TABLE_EV1_SHA256,
            retrieved_at="2026-10-07T11:44:03.836103+00:00",
        ),
    ),
)
#: ``{raw file name: pinned sha256}``, the build-time check of every consumed file.
DATA_SHA256: dict[str, str] = {f.name: f.sha256 for f in RAW_FILES}
RAW_FILES_BY_NAME: dict[str, RawFile] = {f.name: f for f in RAW_FILES}

#: S-BSST5 files deliberately not mirrored (the loader does not read them).
NOT_MIRRORED = (
    "S-BSST5 rawdata_neg_all.tsv / rawdata_pos_all.tsv (raw ion intensities, 4 columns "
    "per strain; 219 MB + 300 MB): not consumed; the replicate count is read from "
    "sample_id_all.xls instead",
    "S-BSST5 neg_kegg_all_3mD.xls / pos_kegg_all_3mD.xls (putative KEGG annotations, "
    "'refer to Table EV1B'): not consumed",
)


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the sha256-pinned Fuhrer ``paper.md``."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
        ),
    )


def _si6(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to the author checklist in Expanded View 6 (``si/si6.md``)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI6_MD,
            citation_key=CITATION_KEY,
            sha256=SI6_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page="EMBO reporting checklist, sample size",
        ),
    )


def _baba(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to Baba 2006, the Keio paper Fuhrer defers the collection to."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=BABA_KEY,
            sha256=BABA_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
        ),
    )


def _study(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to the BioStudies study record in the raw mirror.

    ``source_uri`` is relative to ``$DATA_ROOT/torchcell-raw/<citation_key>/``, not to
    the literature mirror; :func:`sourced_value_root` resolves which.
    """
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=RAW_FILES_BY_NAME[STUDY_JSON].mirror_relpath,
            citation_key=CITATION_KEY,
            sha256=DATA_SHA256[STUDY_JSON],
            method="BioStudies S-BSST5 study JSON (torchcell-raw mirror)",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_CULTURE_QUOTE = (
    "Culture volumes of $1 \\ \\mathrm { m l }$ were incubated in 96-deep well plates at "
    "$3 7 ^ { \\circ } \\mathrm { C }$ with shaking at $3 0 0 ~ \\mathrm { { r p m } }$ ."
)
_REPLICATE_SUMMARY_QUOTE = (
    "Modified $z$ -scores referring to technical and biological replicates are "
    "summarized into a unique median modified $z$ -score, and three additional mutants "
    "were removed from the dataset having inconsistent $z$ -score among the replicates "
    "(flhC, hrpB, and yhfW), resulting in a final data set with 3,807 mutants."
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "collection": _paper(
        "KEIO knockout collection",
        "Escherichia coli wild-type and 4,320 deletion mutants (Table EV1) from the KEIO "
        "knockout collection (Baba et al, 2006)",
        note="Methods, 'Biological samples'; the stored collection string is this "
        "quote's own wording",
    ),
    "background_strain": _baba(
        "BW25113",
        "The Keio collection is comprised of 3985 deletions in duplicate (7970 total) of "
        "E. coli K-12 strain BW25113 (Datsenko and Wanner, 2000)",
        note="the deferral target of Fuhrer's '(Baba et al, 2006)'; Fuhrer itself never "
        "names the background",
    ),
    "cassette": _baba(
        "kanamycin cassette flanked by FLP recognition target sites",
        "Open-reading frame coding regions were replaced with a kanamycin cassette "
        "flanked by FLP recognition target sites",
        note="Fuhrer does not say the cassette was excised, so the collection strain "
        "(kanamycin resistant, cassette in place) is what is recorded",
    ),
    "temperature_c": _paper(37.0, _CULTURE_QUOTE),
    "aerobic_shaking": _paper(
        "aerobic",
        _CULTURE_QUOTE,
        note="shaken 1 ml cultures in deep-well plates; the paper names an oxygen "
        "regime only for its separate anaerobic perturbation cultures",
    ),
    "harvest": _paper(
        "mid-exponential",
        "all samples were harvested during mid-exponential growth phase",
        note="harvest by growth phase, not by time, so duration_hours stays None",
    ),
    "n_biological_replicates": _paper(
        2,
        "For each knockout mutant, two different clones contained on separate plates "
        "in the library were separately processed on different days.",
        note="Methods, 'Reproducibility between biological replicates'. Counted per "
        "strain at build time as sample_id_all.xls columns / TECHNICAL_REPLICATES, which "
        "is 4 / 2 = 2 for every single-entry strain of the pinned deposit",
    ),
    "n_biological_replicates_checklist": _si6(
        2,
        "Two biological replicates (independent clones) were extracted and measured "
        "with two technicalreplicates.",
        note="the OCR joins 'technical replicates'; corroborates the Methods",
    ),
    "technical_replicates": _paper(
        2,
        "Metabolome extracts were prepared from cultures growing exponentially in "
        "mineral salts medium containing glucose and amino acids and analyzed in "
        "technical duplicates by non-targeted mass spectrometry.",
        note="technical duplicates are injections of one extract, not independent "
        "replicates, so n_replicates counts clones (raw columns / 2)",
    ),
    "design": _study(
        "2 biological x 2 technical",
        "2 Biological and 2 technical replicates",
        note="the study record's 'Experimental design' attribute; process() reads it "
        "and refuses any other value",
    ),
    "raw_columns_per_strain": _study(
        4,
        "Raw data matrix of negative ionization mode, rows correspond to ions, columns "
        "to genes, 4 columns per gene from technical and biological duplicates",
        note="the raw matrices' column names are sample_id_all.xls; a name with 8 or 12 "
        "columns there is a name shared by 2 or 3 Table EV1A entries",
    ),
    "zscore_matrix_layout": _study(
        "rows ions, columns strains, replicates averaged",
        "Zscore transformed data matrix of negative ionization mode, replicates are "
        "averaged, rows correspond to ions, columns to genes",
    ),
    "zscore_definition": _paper(
        "modified z-score: (intensity - median of the ion over the dataset) / std",
        "where $i$ and $j$ denote ion and samples, respectively, and median as well as "
        "standard deviation (std) refers to all intensities of ion i in the entire "
        "dataset.",
        note="the paper's per-sample definition. Measured on the deposit at build time "
        "(zscore_scale): every ion of the released matrix has median 0 and sample SD 1 "
        "over its 3,807 columns, wt included, so the stored score is standardized over "
        "exactly those columns; it is unitless and not centered on the wild type",
    ),
    "replicate_summary": _paper(
        "median of the replicate z-scores (paper) / average (deposit description)",
        _REPLICATE_SUMMARY_QUOTE,
        note="the paper says median, the S-BSST5 file description says 'replicates are "
        "averaged' (SOURCED_VALUES['zscore_matrix_layout']); the per-replicate scores "
        "are not released, so which one the deposit holds cannot be checked. Recorded, "
        "not resolved",
    ),
    "replicate_noise": _paper(
        2.765,
        "Ninety-nine percent of variability between biological replicates was "
        "estimated to be smaller than a $z$ -score of 2.765 (Fig EV2A).",
        note="a dataset-level noise statistic over all strains and ions, not a "
        "per-record uncertainty; it is not stored on any record",
    ),
    "ion_counts": _paper(
        {"neg": 3169, "pos": 4365},
        "Spectral data processing identified 3,169 and 4,365 distinct mass-tocharge "
        "$\\left( m / z \\right)$ features in negative and positive ionization mode, "
        "respectively.",
        note="process() requires the z-score matrices to have exactly these row counts",
    ),
    "data_availability": _paper(
        BIOSTUDIES_ACCESSION,
        "Raw data and modified $z$ -scores for positive and negative mode "
        "(tabseparated and excel files) can be downloaded from https://www.eb "
        "i.ac.uk/biostudies/, accession code: S-BSST5.",
    ),
    "no_internal_standards": _paper(
        "relative ion intensities only",
        "Notably, no internal standards were used because they are not suited to "
        "normalize thousands of mostly unknown chemical entities.",
        note="why the level is a relative score and no absolute concentration exists",
    ),
}

TEMPERATURE_C = SOURCED_VALUES["temperature_c"]
N_BIOLOGICAL_REPLICATES = SOURCED_VALUES["n_biological_replicates"]
TECHNICAL_REPLICATES: int = SOURCED_VALUES["technical_replicates"].value
RAW_COLUMNS_PER_STRAIN: int = SOURCED_VALUES["raw_columns_per_strain"].value
EXPERIMENTAL_DESIGN = SOURCED_VALUES["design"].quote
EXPECTED_ION_COUNTS: dict[str, int] = SOURCED_VALUES["ion_counts"].value
COLLECTION: str = SOURCED_VALUES["collection"].value
CASSETTE: str = SOURCED_VALUES["cassette"].value

MEASUREMENT_TYPE = "fia_tof_ms_ion_modified_z_score"
WILD_TYPE_COLUMN = "wt"
READY_STATUS = "ready to distribute"
#: Checklist item 4: below this fraction of JW ids resolving to BW25113 locus tags the
#: build stops. The BW25113 GenBank file carries 4,334 JW synonyms, so a fraction this low
#: would mean the wrong file or the wrong strain, not a handful of merged pseudogenes.
MIN_RESOLVED_FRACTION = 0.95
#: Minimum share of Table EV1B ions whose nearest deposit m/z is the same row; below it
#: the ``neg_NNNN`` / ``pos_NNNN`` keys would not mean Table EV1B's Ion Index.
MIN_ION_ALIGNMENT = 0.999

METABOLITE_LEVEL_SE_GAP = ProvenanceGap(
    field="metabolite_level_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=PAPER_MD,
        citation_key=CITATION_KEY,
        sha256=PAPER_MD_SHA256,
        method="full Methods read, plus every S-BSST5 file description in "
        f"{STUDY_JSON} (sha256 {DATA_SHA256[STUDY_JSON]})",
        page="Methods, 'Data normalization and calculation of differential ions' and "
        "'Reproducibility between biological replicates'",
    ),
    note="the deposit releases one summarized z-score per strain and ion; the "
    "per-replicate z-scores are not released, and the raw intensities cannot be "
    "re-normalized to them (plate drift, low-pass and harvest-OD LOWESS corrections "
    "with unreleased inputs). The only stated spread is dataset-level "
    "(SOURCED_VALUES['replicate_noise'])",
)

PUBLICATION = Publication(
    pubmed_id=PUBMED_ID,
    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
    doi=PAPER_DOI,
    doi_url=f"https://doi.org/{PAPER_DOI}",
)

FloatMatrix = npt.NDArray[np.float64]


def ion_key(mode: IonMode, index: int) -> str:
    """The metabolite key of one ion: mode and 1-based row (Table EV1B's Ion Index)."""
    return f"{mode}_{index:04d}"


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never a live URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/fuhrerGenomewideLandscapeGene2017``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/fuhrerGenomewideLandscapeGene2017``."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve_raw_files(
    dest_dir: str | Path, names: Sequence[str] | None = None
) -> dict[str, Path]:
    """Run each file's recorded retriever and write the verified bytes to ``dest_dir``.

    The recorded ``RetrievalRecord`` is what runs (``run_retriever``), so this is the
    re-runnable retrieval itself; a byte mismatch raises before anything is written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for raw in RAW_FILES:
        if names is not None and raw.name not in names:
            continue
        path = dest / raw.name
        write_verified(
            run_retriever(raw.retrieval),
            path,
            raw.sha256,
            raw.retrieval.source_url or raw.name,
        )
        out[raw.name] = path
    return out


def deposit_raw_mirror(
    *, sources: Mapping[str, str | Path], data_root: str | None = None
) -> Path:
    """Write the raw mirror from already-retrieved files plus its ``manifest.json``.

    ``sources`` maps every name in ``RAW_FILES`` to a local file. Idempotent by sha256:
    a mirror file with the pinned hash is left alone, and one with any other hash
    raises rather than being overwritten. Table EV1 (``si2.xlsx``) is the literature
    mirror's capture, recorded with the same retrieval.
    """
    missing = sorted(set(DATA_SHA256) - set(sources))
    if missing:
        raise KeyError(f"no source given for {missing}")
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for raw in RAW_FILES:
        src = Path(sources[raw.name])
        got = _sha256(src)
        if got != raw.sha256:
            raise RuntimeError(
                f"{src} sha256 mismatch: got {got}, expected {raw.sha256}"
            )
        dest = root / raw.mirror_relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != raw.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        records.append(
            ArtifactRecord(
                path=raw.mirror_relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=raw.sha256,
                source=raw.retrieval.source_url,
                retrieval=raw.retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title="Genomewide landscape of gene-metabolome associations in Escherichia coli",
        files=records,
        si_data_sources=[BIOSTUDIES_STUDY_URL, f"{BIOSTUDIES_ROOT}/Files/"],
        si_expected=list(NOT_MIRRORED),
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


def sourced_value_root(value: SourcedValue, data_root: str | None = None) -> Path:
    """The mirror root a ``SourcedValue``'s ``source_uri`` is relative to.

    The study record sits in the raw mirror (``data/...``); every other quote is from a
    paper in the literature mirror.
    """
    base = Path(data_root or _data_root())
    if value.provenance.source_uri.startswith("data/"):
        return base / "torchcell-raw"
    return base / "torchcell-library"


# --------------------------------------------------------------------------- #
# Parsing the released files
# --------------------------------------------------------------------------- #
class KeioEntry(BaseModel):
    """One Table EV1A row: a Keio strain and its delivery status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    jw_id: str
    blattner_id: str
    gene_name: str
    delivery_status: str


class UnmappedDepositColumnError(ValueError):
    """A z-score column whose name matches no Table EV1A entry.

    The pinned deposit has none besides ``wt``, the reference.
    """


class RawColumnCountError(ValueError):
    """A strain whose raw-matrix column count is not 4 per Table EV1A entry."""


class UndistributedKeioEntryError(ValueError):
    """A single-entry strain whose Keio status is not 'ready to distribute'.

    The pinned Table EV1A has none among the deposit's columns.
    """


class NonFiniteZScoreError(ValueError):
    """A z-score cell that is not a finite number (the pinned matrices have none)."""


def read_study_design(path: str | Path) -> str:
    """The ``Experimental design`` attribute of the BioStudies study JSON."""
    study = json.loads(Path(path).read_text(encoding="utf-8"))
    designs = [
        a["value"]
        for a in study["section"]["attributes"]
        if a["name"] == "Experimental design"
    ]
    if len(designs) != 1:
        raise ValueError(f"{path}: {len(designs)} 'Experimental design' attributes")
    return str(designs[0])


def read_table_ev1a(path: str | Path) -> list[KeioEntry]:
    """Table EV1A rows that carry a JW id (footnotes and blank rows have none)."""
    frame = pd.read_excel(
        path, sheet_name=TABLE_EV1A_SHEET, header=TABLE_EV1A_HEADER_ROW
    )
    frame = frame[frame["JW ID"].notna()]
    return [
        KeioEntry(
            jw_id=str(row["JW ID"]).strip(),
            blattner_id=str(row["Blattner ID"]).strip(),
            gene_name=str(row["Gene Name"]).strip(),
            delivery_status=str(row["Keio Delivery Statusb"]).strip(),
        )
        for _, row in frame.iterrows()
    ]


def read_table_ev1b_mz(path: str | Path) -> dict[IonMode, list[float]]:
    """Table EV1B's m/z per mode, ordered by Ion Index (which must run 1..n)."""
    frame = pd.read_excel(path, sheet_name=TABLE_EV1B_SHEET, header=0)
    frame = frame[frame["Ion Index"].notna()]
    out: dict[IonMode, list[float]] = {}
    for mode in ION_MODES:
        rows = frame[frame["Ionization Mode"] == mode]
        index = rows["Ion Index"].astype(int).tolist()
        if index != list(range(1, len(rows) + 1)):
            raise ValueError(
                f"Table EV1B {mode}: Ion Index does not run 1..{len(rows)}"
            )
        out[mode] = rows["m/z"].astype(float).tolist()
    return out


def read_name_column(path: str | Path) -> list[str]:
    """The single unnamed column of a BioStudies ``sample_id_*.xls`` / ``*_ionMz.xls``."""
    frame = pd.read_excel(path, header=None)
    if frame.shape[1] != 1:
        raise ValueError(f"{path}: expected one column, found {frame.shape[1]}")
    return [str(v).strip() for v in frame[0].tolist()]


def read_zscores(path: str | Path, n_rows: int, n_columns: int) -> FloatMatrix:
    """A headerless tab-separated z-score matrix, refusing a wrong shape or NaN/inf."""
    matrix = pd.read_csv(path, sep="\t", header=None, dtype=np.float64).to_numpy()
    if matrix.shape != (n_rows, n_columns):
        raise ValueError(
            f"{path}: shape {matrix.shape}, expected ({n_rows}, {n_columns})"
        )
    bad = np.argwhere(~np.isfinite(matrix))
    if bad.size:
        row, col = (int(v) for v in bad[0])
        raise NonFiniteZScoreError(
            f"{path}: {len(bad)} non-finite cells, first at row {row + 1}, column "
            f"{col + 1} ({matrix[row, col]!r})"
        )
    return matrix


class IonAlignment(BaseModel):
    """How one mode's deposit rows line up with Table EV1B's Ion Index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: IonMode
    n_ions: int
    nearest_row_agrees: int
    max_abs_mz_difference: float
    median_abs_mz_difference: float

    @property
    def agreement(self) -> float:
        """Share of Table EV1B ions whose nearest deposit m/z is the same row."""
        return self.nearest_row_agrees / self.n_ions


def ion_alignment(
    mode: IonMode, deposit_mz: Sequence[float], ev1b_mz: Sequence[float]
) -> IonAlignment:
    """Compare the deposit's m/z rows with Table EV1B, refusing a count mismatch or a
    nearest-row agreement below ``MIN_ION_ALIGNMENT``.
    """
    deposit = np.asarray(deposit_mz, dtype=np.float64)
    table = np.asarray(ev1b_mz, dtype=np.float64)
    if deposit.shape != table.shape:
        raise ValueError(
            f"{mode}: {deposit.size} deposit ions but {table.size} Table EV1B ions"
        )
    order = np.argsort(deposit, kind="stable")
    ordered = deposit[order]
    right = np.clip(np.searchsorted(ordered, table), 1, ordered.size - 1)
    left = right - 1
    closer_left = np.abs(table - ordered[left]) <= np.abs(ordered[right] - table)
    nearest = order[np.where(closer_left, left, right)]
    agrees = int((nearest == np.arange(table.size)).sum())
    diff = np.abs(deposit - table)
    result = IonAlignment(
        mode=mode,
        n_ions=int(table.size),
        nearest_row_agrees=agrees,
        max_abs_mz_difference=float(diff.max()),
        median_abs_mz_difference=float(np.median(diff)),
    )
    if result.agreement < MIN_ION_ALIGNMENT:
        raise ValueError(
            f"{mode}: only {agrees} of {table.size} Table EV1B ions are nearest their "
            f"own deposit row (< {MIN_ION_ALIGNMENT})"
        )
    return result


#: The deposit rounds to four decimals, so a standardized ion's column SD reads
#: 1.0000 to within rounding; anything further off is not the scale assumed here.
ZSCORE_SCALE_TOLERANCE = 1e-3


class ZScoreScale(BaseModel):
    """Per-ion location and spread of one mode's released matrix across its columns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: IonMode
    n_ions: int
    n_columns: int
    max_abs_column_median: float
    min_column_sd: float
    max_column_sd: float


def zscore_scale(mode: IonMode, matrix: FloatMatrix) -> ZScoreScale:
    """Check that every ion of the released matrix has median 0 and sample SD 1 over
    the deposit's own columns (``wt`` included), i.e. the stored score is standardized
    over exactly the released strain columns; refuse otherwise.
    """
    median = np.median(matrix, axis=1)
    sd = matrix.std(axis=1, ddof=1)
    scale = ZScoreScale(
        mode=mode,
        n_ions=int(matrix.shape[0]),
        n_columns=int(matrix.shape[1]),
        max_abs_column_median=float(np.abs(median).max()),
        min_column_sd=float(sd.min()),
        max_column_sd=float(sd.max()),
    )
    if (
        scale.max_abs_column_median > ZSCORE_SCALE_TOLERANCE
        or abs(scale.min_column_sd - 1.0) > ZSCORE_SCALE_TOLERANCE
        or abs(scale.max_column_sd - 1.0) > ZSCORE_SCALE_TOLERANCE
    ):
        raise ValueError(f"{mode}: released z-scores are not standardized: {scale}")
    return scale


# --------------------------------------------------------------------------- #
# Strain selection and the retention ledger
# --------------------------------------------------------------------------- #
class KeioStrain(BaseModel):
    """One deposit column kept as one Keio strain, before locus-tag reconciliation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: int
    deposit_name: str
    entry: KeioEntry
    raw_columns: int

    @property
    def n_biological(self) -> int:
        """Independent clones: raw columns over the technical injections per clone."""
        return self.raw_columns // TECHNICAL_REPLICATES


class DropRule(BaseModel):
    """One retention rule, the deposit columns it removed, and why."""

    rule: str
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    deposit_columns: int
    reference_columns: list[str]
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


class StrainSelection(BaseModel):
    """The deposit columns kept as candidate strains, the wild type, and the drops."""

    strains: list[KeioStrain]
    wt_column: int
    wt_raw_columns: int
    pooled: DropRule

    @property
    def wt_n_biological(self) -> int:
        """WT cultures: the ``wt`` raw columns over the technical injections."""
        return self.wt_raw_columns // TECHNICAL_REPLICATES


def select_strains(
    column_names: Sequence[str],
    entries: Sequence[KeioEntry],
    raw_column_counts: Mapping[str, int],
) -> StrainSelection:
    """Keep each deposit column that is exactly one Keio strain.

    ``wt`` is the reference. A name matching no Table EV1A entry, a raw column count
    other than 4 per entry, or a single entry that is not 'ready to distribute' is
    refused by name (the pinned deposit has none). A name matching several entries is
    dropped under ``pooled_keio_entries``: its released value summarizes more than one
    Keio strain.
    """
    by_name: dict[str, list[KeioEntry]] = {}
    for entry in entries:
        by_name.setdefault(entry.gene_name, []).append(entry)
    strains: list[KeioStrain] = []
    pooled_items: list[str] = []
    wt: tuple[int, int] | None = None
    for column, name in enumerate(column_names):
        raw = raw_column_counts.get(name, 0)
        if name == WILD_TYPE_COLUMN:
            if raw == 0 or raw % TECHNICAL_REPLICATES:
                raise RawColumnCountError(f"'wt' has {raw} raw columns")
            wt = (column, raw)
            continue
        matched = by_name.get(name, [])
        if not matched:
            raise UnmappedDepositColumnError(
                f"deposit column {column + 1} {name!r} matches no Table EV1A entry"
            )
        if raw != RAW_COLUMNS_PER_STRAIN * len(matched):
            raise RawColumnCountError(
                f"{name!r}: {raw} raw columns for {len(matched)} Table EV1A entries "
                f"(expected {RAW_COLUMNS_PER_STRAIN} each)"
            )
        if len(matched) > 1:
            loci = sorted({e.blattner_id for e in matched})
            pooled_items.append(
                f"{name}: "
                + ", ".join(
                    f"{e.jw_id} {e.blattner_id} ({e.delivery_status})" for e in matched
                )
                + ("; same locus" if len(loci) == 1 else f"; {len(loci)} loci")
            )
            continue
        entry = matched[0]
        if entry.delivery_status != READY_STATUS:
            raise UndistributedKeioEntryError(
                f"{name!r} ({entry.jw_id}) has Keio status {entry.delivery_status!r}"
            )
        strains.append(
            KeioStrain(column=column, deposit_name=name, entry=entry, raw_columns=raw)
        )
    if wt is None:
        raise UnmappedDepositColumnError("the deposit has no 'wt' column")
    return StrainSelection(
        strains=strains,
        wt_column=wt[0],
        wt_raw_columns=wt[1],
        pooled=DropRule(
            rule="pooled_keio_entries",
            description="the deposit name matches more than one Table EV1A entry and "
            "the raw matrix carries 4 columns per entry under that name, so the "
            "released value summarizes several Keio strains, not one genotype",
            n_records=len(pooled_items),
            items=pooled_items,
        ),
    )


class SymbolDisagreement(BaseModel):
    """A kept strain whose deposit NAME does not resolve to its JW id's locus.

    The name resolves to another locus, is ambiguous, or is retired on the BW25113
    annotation; the record keeps the JW id's locus either way.
    """

    deposit_name: str
    jw_id: str
    jw_locus: str
    name_resolution: str


class IdentifierLedger(BaseModel):
    """The JW reconciliation report plus what the deposit names would have said."""

    reconciliation: LocusTagReconciliation
    min_resolved_fraction: float
    symbol_disagreements: list[SymbolDisagreement]


def canonical_symbol(genome: EcoliK12Genome, tag: str) -> str:
    """The annotation's own gene symbol of ``tag`` when it resolves back to ``tag``,
    else the tag itself, so the stored pair always round-trips through the resolver.
    """
    symbol = genome.genbank.loci[tag].symbol
    if symbol is None:
        return tag
    resolution = genome.resolve_gene_name(symbol)
    return symbol if resolution.systematic_name == tag else tag


class ResolvedStrain(BaseModel):
    """A kept Keio strain with its BW25113 locus tag and the annotation's symbol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: KeioStrain
    locus_tag: str
    symbol: str


def resolve_strains(
    genome: EcoliK12Genome, strains: Sequence[KeioStrain], *, label: str
) -> tuple[list[ResolvedStrain], DropRule, IdentifierLedger]:
    """Reconcile each strain's JW id on ``genome`` and drop the ones off its namespace.

    Stops (``LocusTagResolutionError``) below ``MIN_RESOLVED_FRACTION``. A JW id the
    reconciler keeps as given (retired, ambiguous, or colliding with another JW id) is
    not a locus tag the deletion leaf accepts; its record is dropped with the reason.
    """
    jw = pd.Series([s.entry.jw_id for s in strains], dtype=object)
    stored, report = reconcile_locus_tags(genome, jw, label=label)
    report.require_resolved(MIN_RESOLVED_FRACTION)
    pattern = LOCUS_TAG_PATTERNS[report.gene_namespace]
    kept: list[ResolvedStrain] = []
    unplaced: list[tuple[KeioStrain, str, GeneNameResolution]] = []
    disagreements: list[SymbolDisagreement] = []
    for strain, tag in zip(strains, stored.tolist(), strict=True):
        name_res = genome.resolve_gene_name(strain.deposit_name)
        if pattern.match(tag) is None:
            jw_res = genome.resolve_gene_name(strain.entry.jw_id)
            reason = (
                "kept as given on collision"
                if strain.entry.jw_id in report.kept_on_collision
                else jw_res.status.value
            )
            unplaced.append((strain, reason, name_res))
            continue
        if name_res.systematic_name != tag:
            disagreements.append(
                SymbolDisagreement(
                    deposit_name=strain.deposit_name,
                    jw_id=strain.entry.jw_id,
                    jw_locus=tag,
                    name_resolution=f"{name_res.status.value} "
                    f"{name_res.systematic_name}",
                )
            )
        kept.append(
            ResolvedStrain(
                strain=strain, locus_tag=tag, symbol=canonical_symbol(genome, tag)
            )
        )
    held = {item.locus_tag for item in kept}
    dropped = [
        f"{strain.entry.jw_id} ({strain.deposit_name}, {strain.entry.blattner_id}): "
        f"{reason}; the name resolves {name_res.status.value} "
        f"{name_res.systematic_name}"
        + (", a kept record's locus" if name_res.systematic_name in held else "")
        for strain, reason, name_res in unplaced
    ]
    rule = DropRule(
        rule="jw_id_not_on_a_bw25113_locus",
        description="the strain's Keio JW id is not a gene_synonym of exactly one "
        "BW25113 GenBank locus (retired, ambiguous, or shared with another kept JW "
        "id), so no locus tag of the pinned namespace names its deletion; each item "
        "says what the deposit name resolves to and whether a kept record already "
        "holds that locus",
        n_records=len(dropped),
        items=dropped,
    )
    ledger = IdentifierLedger(
        reconciliation=report,
        min_resolved_fraction=MIN_RESOLVED_FRACTION,
        symbol_disagreements=disagreements,
    )
    return kept, rule, ledger


# --------------------------------------------------------------------------- #
# Record construction (pure, no files)
# --------------------------------------------------------------------------- #
def ion_keys(n_neg: int, n_pos: int) -> list[str]:
    """Every metabolite key, negative mode first, in deposit row order."""
    return [ion_key("neg", i) for i in range(1, n_neg + 1)] + [
        ion_key("pos", i) for i in range(1, n_pos + 1)
    ]


def metabolite_phenotype(
    keys: Sequence[str], values: Sequence[float], n_biological: int
) -> MetabolitePhenotype:
    """One strain's z-score profile; every ion carries the strain's clone count."""
    if len(keys) != len(values):
        raise ValueError(f"{len(keys)} keys for {len(values)} values")
    return MetabolitePhenotype(
        metabolite_level={k: float(v) for k, v in zip(keys, values, strict=True)},
        metabolite_level_se=None,
        n_replicates=dict.fromkeys(keys, n_biological),
        measurement_type=MEASUREMENT_TYPE,
        target_metabolite_ids=None,
        provenance_gaps=[METABOLITE_LEVEL_SE_GAP],
    )


def environment() -> Environment:
    """Fuhrer's screening medium at 37 C, shaken (aerobic)."""
    return Environment(
        media=M9_GLUCOSE_CASEIN_FUHRER2017,
        temperature=Temperature(value=TEMPERATURE_C.value),
        aerobicity=SOURCED_VALUES["aerobic_shaking"].value,
    )


def deletion_genotype(resolved: ResolvedStrain) -> Genotype:
    """The one Keio deletion, named by its BW25113 locus tag."""
    return Genotype(
        perturbations=[
            BacterialDeletionPerturbation(
                systematic_gene_name=resolved.locus_tag,
                perturbed_gene_name=resolved.symbol,
                gene_namespace=STRAIN_GENE_NAMESPACES["BW25113"],
                collection=COLLECTION,
                cassette=CASSETTE,
                construction=StrainConstruction(
                    strain_accession=resolved.strain.entry.jw_id
                ),
            )
        ]
    )


def build_experiment(
    dataset_name: str,
    resolved: ResolvedStrain,
    keys: Sequence[str],
    values: Sequence[float],
    env: Environment,
) -> BacterialMetaboliteExperiment:
    """The record of one Keio strain."""
    return BacterialMetaboliteExperiment(
        dataset_name=dataset_name,
        genotype=deletion_genotype(resolved),
        environment=env,
        phenotype=metabolite_phenotype(keys, values, resolved.strain.n_biological),
    )


def build_reference(
    dataset_name: str,
    genome_reference: AssemblyReferenceGenome,
    keys: Sequence[str],
    wt_values: Sequence[float],
    wt_n_biological: int,
    env: Environment,
) -> BacterialMetaboliteExperimentReference:
    """The measured ``wt`` profile on the same z-score scale, in the same medium."""
    return BacterialMetaboliteExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=env.model_copy(),
        phenotype_reference=metabolite_phenotype(keys, wt_values, wt_n_biological),
    )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class MetabolomeFuhrer2017Dataset(ExperimentDataset):
    """Genome-wide FIA-TOF-MS ion metabolome of the E. coli Keio deletion collection."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = "data/torchcell/metabolome_fuhrer2017",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is injected by the build entry points."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialMetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialMetaboliteExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Every consumed file, linked from the raw mirror."""
        return [raw.name for raw in RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking the manifest and sha256.

        The mirror plus ``DATA_SHA256`` is canonical; the BioStudies and PMC URLs are
        retrieval metadata that ``retrieve_raw_files`` re-runs, never a build input.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for raw in RAW_FILES:
            check_manifest_pin(
                raw.mirror_relpath,
                manifest_sha256(manifest, raw.mirror_relpath),
                raw.sha256,
            )
            src = raw_mirror_dir(data_root) / raw.mirror_relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, raw.name), raw.sha256)
        log.info("Fuhrer 2017 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _raw(self, name: str) -> str:
        return osp.join(self.raw_dir, name)

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache (a direct run);
        a genome of another assembly set is refused.
        """
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Parse the S-BSST5 matrices and Table EV1 into per-strain records + LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        design = read_study_design(self._raw(STUDY_JSON))
        if design != EXPERIMENTAL_DESIGN:
            raise ValueError(
                f"S-BSST5 design {design!r}, expected {EXPERIMENTAL_DESIGN!r}"
            )
        entries = read_table_ev1a(self._raw(TABLE_EV1))
        ev1b_mz = read_table_ev1b_mz(self._raw(TABLE_EV1))
        column_names = read_name_column(self._raw(SAMPLE_ID_ZSCORE))
        if len(set(column_names)) != len(column_names):
            raise ValueError(f"{SAMPLE_ID_ZSCORE} repeats a column name")
        raw_column_counts = Counter(read_name_column(self._raw(SAMPLE_ID_ALL)))

        alignments: list[IonAlignment] = []
        scales: list[ZScoreScale] = []
        matrices: dict[IonMode, FloatMatrix] = {}
        deposit_mz: dict[IonMode, list[float]] = {}
        for mode in ION_MODES:
            deposit_mz[mode] = [
                float(v) for v in read_name_column(self._raw(ION_MZ_FILES[mode]))
            ]
            if len(deposit_mz[mode]) != EXPECTED_ION_COUNTS[mode]:
                raise ValueError(
                    f"{ION_MZ_FILES[mode]}: {len(deposit_mz[mode])} ions, the paper "
                    f"states {EXPECTED_ION_COUNTS[mode]}"
                )
            alignments.append(ion_alignment(mode, deposit_mz[mode], ev1b_mz[mode]))
            matrices[mode] = read_zscores(
                self._raw(ZSCORE_FILES[mode]),
                EXPECTED_ION_COUNTS[mode],
                len(column_names),
            )
            scales.append(zscore_scale(mode, matrices[mode]))
        keys = ion_keys(EXPECTED_ION_COUNTS["neg"], EXPECTED_ION_COUNTS["pos"])
        stacked = np.vstack([matrices["neg"], matrices["pos"]])

        selection = select_strains(column_names, entries, raw_column_counts)
        genome = self._genome()
        resolved, unresolved, ledger = resolve_strains(
            genome, selection.strains, label=f"{self.name} Keio JW ids"
        )
        mutant_columns = len(column_names) - 1
        drop_log = DropLog(
            dataset=self.name,
            deposit_columns=len(column_names),
            reference_columns=[WILD_TYPE_COLUMN],
            source_records=mutant_columns,
            kept_records=len(resolved),
            dropped_records=mutant_columns - len(resolved),
            rules=[selection.pooled, unresolved],
        )
        if sum(r.n_records for r in drop_log.rules) != drop_log.dropped_records:
            raise RuntimeError("drop rules do not account for every dropped column")
        log.info(
            "Fuhrer 2017: %d deposit mutant columns -> %d records; %d pooled names, "
            "%d JW ids off the BW25113 namespace; JW statuses %s; %d deposit names "
            "do not resolve to their JW id's locus",
            mutant_columns,
            len(resolved),
            selection.pooled.n_records,
            unresolved.n_records,
            {s.value: n for s, n in ledger.reconciliation.status_histogram.items()},
            len(ledger.symbol_disagreements),
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_ledgers(
            drop_log, ledger, alignments, scales, deposit_mz, ev1b_mz, resolved
        )

        env = environment()
        reference = build_reference(
            self.name,
            assembly_reference(self.REFERENCE_STRAIN),
            keys,
            stacked[:, selection.wt_column].tolist(),
            selection.wt_n_biological,
            env,
        )
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, item in enumerate(tqdm(resolved, desc="fuhrer2017")):
                experiment = build_experiment(
                    self.name, item, keys, stacked[:, item.strain.column].tolist(), env
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        env_out.close()
        interned_env.close()
        log.info("Wrote %d Fuhrer 2017 metabolome experiments to LMDB", len(resolved))

    def _write_ledgers(
        self,
        drop_log: DropLog,
        ledger: IdentifierLedger,
        alignments: Sequence[IonAlignment],
        scales: Sequence[ZScoreScale],
        deposit_mz: Mapping[IonMode, Sequence[float]],
        ev1b_mz: Mapping[IonMode, Sequence[float]],
        resolved: Sequence[ResolvedStrain],
    ) -> None:
        """The drop log, identifier ledger, ion checks, ion and strain tables."""
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drop_log.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            ledger.model_dump_json(indent=2)
        )
        (out / "ion_alignment.json").write_text(
            json.dumps(
                [a.model_dump() | {"agreement": a.agreement} for a in alignments],
                indent=2,
            )
        )
        (out / "zscore_scale.json").write_text(
            json.dumps([scale.model_dump() for scale in scales], indent=2)
        )
        pd.DataFrame(
            [
                {
                    "key": ion_key(mode, i + 1),
                    "mode": mode,
                    "ion_index": i + 1,
                    "deposit_mz": deposit_mz[mode][i],
                    "table_ev1b_mz": ev1b_mz[mode][i],
                }
                for mode in ION_MODES
                for i in range(len(deposit_mz[mode]))
            ]
        ).to_csv(out / "ions.csv", index=False)
        pd.DataFrame(
            [
                {
                    "record": idx,
                    "deposit_column": r.strain.column + 1,
                    "deposit_name": r.strain.deposit_name,
                    "jw_id": r.strain.entry.jw_id,
                    "blattner_id": r.strain.entry.blattner_id,
                    "locus_tag": r.locus_tag,
                    "symbol": r.symbol,
                    "n_biological": r.strain.n_biological,
                }
                for idx, r in enumerate(resolved)
            ]
        ).to_csv(out / "strains.csv", index=False)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled by ``build_experiment``."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built LMDB
# --------------------------------------------------------------------------- #
VERIFIER_PROVENANCE = Provenance(
    source_uri=f"{BIOSTUDIES_STUDY_URL} (zscore_neg.tsv, zscore_pos.tsv)",
    citation_key=CITATION_KEY,
    method="FIA-TOF-MS ion modified z-scores per Keio strain; reference = deposit 'wt'",
    page="Mol Syst Biol 13:907; BioStudies S-BSST5",
)


def run_verification(data_root: str | None = None) -> VerificationReport:
    """Run the metabolite family verifier (L0-L3) plus the bacterial L4 containment
    and the provenance audit of every ``SOURCED_VALUES`` entry on the built dev LMDB,
    and write ``preprocess/verification_report.json``.
    """
    from torchcell.verification.metabolite import (
        metabolite_gene_set,
        verify_metabolite_dataset,
    )
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _write_report,
        load_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, "data/torchcell/metabolome_fuhrer2017")
    records = load_records(abs_root)
    drops = DropLog.model_validate_json(
        Path(abs_root, "preprocess", "dropped_records.json").read_text()
    )
    report = verify_metabolite_dataset(
        records,
        dataset_name="metabolome_fuhrer2017",
        provenance=VERIFIER_PROVENANCE,
        expected_count=drops.kept_records,
        reference_centered=False,
    )
    references = {
        json.dumps(r["reference"]["genome_reference"], sort_keys=True) for r in records
    }
    universe: set[str] = set()
    for ref in references:
        universe |= _gene_set_for_reference(json.loads(ref), base)
    deleted = metabolite_gene_set(records)
    missing = sorted(deleted - universe)
    report.add(
        LevelResult(
            level=Level.L4,
            name="gene_containment_bw25113_locus_tags",
            passed=not missing,
            message=f"{len(deleted) - len(missing)} of {len(deleted)} deleted loci are "
            "BW25113 GenBank gene rows",
            details={
                "n_deleted": len(deleted),
                "n_universe": len(universe),
                "missing_examples": missing[:20],
            },
        )
    )
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, sourced_value_root(value, base)))
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.fuhrer2017"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser(
        "deposit", help="retrieve (optional) and deposit the mirror"
    )
    deposit.add_argument("--download-dir", required=True)
    deposit.add_argument(
        "--retrieve",
        action="store_true",
        help="run every BioStudies retriever into --download-dir first",
    )
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        download = Path(args.download_dir)
        if args.retrieve:
            retrieve_raw_files(
                download, [r.name for r in RAW_FILES if r.name != TABLE_EV1]
            )
        sources: dict[str, str | Path] = {
            name: download / name for name in DATA_SHA256 if name != TABLE_EV1
        }
        sources[TABLE_EV1] = library_dir(data_root) / "si" / TABLE_EV1
        print(deposit_raw_mirror(sources=sources, data_root=data_root))
        return 0
    if args.command == "build":
        dataset = MetabolomeFuhrer2017Dataset(
            root=osp.join(data_root, "data/torchcell/metabolome_fuhrer2017")
        )
        print(f"len = {len(dataset)}")
        return 0
    report = run_verification(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
