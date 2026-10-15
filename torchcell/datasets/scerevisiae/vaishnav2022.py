# torchcell/datasets/scerevisiae/vaishnav2022
# [[torchcell.datasets.scerevisiae.vaishnav2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/vaishnav2022
# Test file: tests/torchcell/datasets/scerevisiae/test_vaishnav2022.py
"""Vaishnav 2022 pooled sort-seq expression of 71,158 promoter sequences (#991).

One record is ONE library sequence's measured expression in SC-Ura: an 80 bp variable
insert inside a 110 bp promoter scaffold driving a YFP reporter on a low-copy CEN
plasmid, measured by the gigantic parallel reporter assay (GPRA), a sort-seq massively
parallel reporter assay. The sequence is the genotype
(``VariablePromoterReporterPerturbation``) and the estimate is the phenotype
(``PooledSortSeqExpressionPhenotype``), the two classes #991 asked for.

RELEASE, and what this loader consumes
--------------------------------------
``GSE163045_MolEvol_seq_data_SCUra_only.splitByOrigID.meanEL.all.txt.gz`` (GEO
GSE163045, sha256 ``9193507a...``), four columns: ``origID``, ``seqID``, ``seq110``,
``ExpressionLevel``. MEASURED on the file (``vaishnav2022_release_inventory.py``):

- 75,487 rows, 71,158 distinct ``seqID`` values, 71,158 distinct ``seq110`` values.
- Within one ``seqID``: exactly one sequence and exactly one expression value, and no
  ``seqID`` mixes a value with a blank. The file name says what that means -- it is
  ``splitByOrigID``, one ROW per design label (``origID``) of a measurement whose own
  identity is ``seqID`` (the semicolon-joined set of the labels that asked for it;
  67,690 rows have ``origID == seqID``). One record per ``seqID``, therefore, and not
  one per row: a row-per-record store would hold 4,433 measurements more than once.
- 104 ``seqID`` groups (110 rows) carry the verbatim cell ``NA`` in
  ``ExpressionLevel``. The release LISTS them, so they are stored: their
  ``expression_level`` is ``None`` beside a typed ``ProvenanceGap`` naming the
  cell. Every listed ``seqID`` is a record, so the store holds 71,158, of which
  71,054 carry an estimate. Nothing outside the release decides what is stored.

``Random_testdata_defined_media.csv`` (Zenodo 4436477, sha256 ``cf31f667...``) re-releases
5,327 of those same measurements with three further columns. MEASURED: every row joins to
the GEO table by ``origID``, each maps to a DISTINCT ``seqID``, and ``meanEL`` equals the
GEO ``ExpressionLevel`` to 7.1e-15 (float round-trip, not a second measurement). It is
therefore not loaded as records of its own -- that would store 5,327 measurements twice --
but as the source of two released per-record values the GEO table does not carry:
``library_id`` (its ``expt`` token, ``NBT_S288CdU_SCUra`` on every row) and ``read_count``
(``proTotal``). Both are ``None`` on the other 65,831 records, which is what the release
states about them.

NOT loaded, with the reason, each named in the dataset note as a later admission:

- the two full random libraries, ``complex_media_training_data_Glu.txt`` (3.59 GB,
  30,722,376 sequences) and ``defined_media_training_data_SC_Ura.txt`` (2.47 GB,
  20,616,659) on Zenodo, and ``GSE163045_average_promoter_ELs_per_seq_N80_SD-Ura_Y8205_ALL
  .shuffled.txt.gz`` (0.67 GB) on GEO: a size decision for the owner, not a schema or a
  provenance one.
- ``Random_testdata_complex_media.txt`` (3,331 rows): the complex-medium (YPD) arm, whose
  full measured release is the 3.59 GB file above. Loading a 3,331-row slice of an arm
  whose superset is deferred would make the medium's coverage an artifact of this
  decision.
- ``si/si3.xlsx`` and ``S288C_reference_promoter_expression_in_both_media.tsv``: model
  PREDICTIONS, not measurements.

STRAIN, and the one inference this loader makes
-----------------------------------------------
The loaded file's rows are written against ``S288C::ura3``. Three pieces of evidence, the
third of which is an inference stated as one: the SI says the designed libraries were
measured in that strain ("For random libraries, the strain Y8205 was used, but later
experiments including the designed libraries were performed in S288C::ura3"); the Zenodo
``expt`` token on the 5,327 overlapping rows is ``NBT_S288CdU_SCUra`` (S288C dUra,
SC-Ura); and the Y8205 random training set is a SEPARATE GEO file whose own name carries
the strain (``..._SD-Ura_Y8205_ALL.shuffled.txt.gz``), which this loader does not read.
The GEO table itself carries no per-row strain column, so a per-record strain cannot be
read off the release; the assignment above is the file-level one.

MEDIUM. The paper names its defined medium "SD-Ura (Sunrise Science, 1703-500)" and the
method's deferral target (de Boer 2020, mirrored) names the same medium "SC-Ura". The
shared ``SC_URA`` object is used, as it already is for Smith 2016's differently-spelled
"SCM-Ura", and the source's verbatim spelling is in ``SOURCED_VALUES``.

WHAT IS NOT STORED, and why
---------------------------
``sdEL``, ``relativeSD`` and ``sdScale`` of the Zenodo table are NOT stored. MEASURED:
``sdEL == sdScale * relativeSD`` on all 5,327 rows to 2.0e-14, so ``sdEL`` is an
author-SCALED quantity, not a plain dispersion of the reads. Neither this paper's
Methods and SI nor de Boer 2020 (its deferral target) defines any of the three, so the
uncertainty TYPE cannot be labeled, and the schema admits no unlabeled uncertainty.
The uncertainty fields therefore carry a typed ``ProvenanceGap`` naming the columns. The
owner question in the dataset note is whether to ask the authors.

``n_samples`` and ``sample_unit`` are gapped for a different reason: the estimate
averages READS, not replicates ("the expression level was the average of the expression
bins in which it was observed, weighted by the number of times it was observed in each
bin"), and a read is not a replicate of the construct. The read count lives in
``read_count``, which is the release's own measure of how well determined the estimate is.

REFERENCE. The loaded files carry NO control-promoter arm: the library's internal control
is the constitutive RFP, and the released level is already a ratio against it ("Cells
were sorted by the log ratio of RFP to YFP signal"). The reference phenotype therefore
restates the record's own reading rather than inventing a baseline, as the untreated arm
of Mohiuddin 2022 does. The paper's measured native-promoter arm is in a Zenodo file this
loader does not read; it is named in the note.
"""

from __future__ import annotations

import argparse
import logging
import os
import os.path as osp
import pickle
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import ExperimentDataset, Organism, post_process, verify_raw_files
from torchcell.datamodels.media import SC_URA
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    PooledSortSeqExpressionExperiment,
    PooledSortSeqExpressionExperimentReference,
    PooledSortSeqExpressionPhenotype,
    Publication,
    ReferenceGenome,
    ReporterReadout,
    Temperature,
    VariablePromoterReporterPerturbation,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CITATION_KEY = "vaishnavEvolutionEvolvabilityEngineering2022"
PAPER_DOI = "10.1038/s41586-022-04506-6"
PAPER_PMID = "35264797"
PAPER_TITLE = "The evolution, evolvability and engineering of gene regulatory DNA"
DATASET_ROOT_REL = "data/torchcell/pooled_promoter_expression_vaishnav2022"

GEO_URL_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE163nnn/GSE163045/suppl/"
ZENODO_URL_BASE = "https://zenodo.org/api/records/4436477/files/"

MOLEVOL_NAME = "GSE163045_MolEvol_seq_data_SCUra_only.splitByOrigID.meanEL.all.txt.gz"
MOLEVOL_SHA256 = "9193507ae0500a8bf0f75388a9bb3e58dd21b84a519b8346d0b1995e242a36f2"
MOLEVOL_REL = f"geo/{MOLEVOL_NAME}"

TESTDATA_NAME = "Random_testdata_defined_media.csv"
TESTDATA_SHA256 = "cf31f667b99ab3ebcac45eda4ed42b483503db70558fcfd9c39a31eab651e0d9"
TESTDATA_REL = f"zenodo/{TESTDATA_NAME}"

ZENODO_README_NAME = "README.md"
ZENODO_README_SHA256 = (
    "b607160fe809c7433ae18f4d917fa206908cd542af12bd514ef154522969db47"
)
ZENODO_README_REL = "zenodo/README.md"

#: The raw files the loader reads, with the sha256 each build verifies.
RAW_FILES: dict[str, str] = {
    MOLEVOL_NAME: MOLEVOL_SHA256,
    TESTDATA_NAME: TESTDATA_SHA256,
}

#: The constant scaffold flanking the variable region, measured on all 75,487 rows of
#: the loaded table (every row starts with the first and ends with the second).
SCAFFOLD_5PRIME = "TGCATTTTTTTCACATC"
SCAFFOLD_3PRIME = "GGTTACGGCTGTT"
SCAFFOLD_NAME = "pTpA"
VARIABLE_REGION_LENGTH = 80

REPORTER_GENE = "YFP"
#: The reporter's identity: a CURIE in the namespace of the deposited dual-reporter
#: plasmid, since the added gene is heterologous and has no yeast systematic name.
REPORTER_IDENTIFIER = "addgene127546:YFP"
REPORTER_SOURCE_ORGANISM = "synthetic construct"
CONSTRUCT_NAME = "pTpA-YFP dual-reporter CEN plasmid (Addgene 127546)"
LOCALIZATION = "episomal_cen_plasmid"
STRAIN = "S288C::ura3"
EXPECTED_RECORDS = 71158
EXPECTED_ROWS = 75487
EXPECTED_UNMEASURED_SEQUENCES = 104

EXPRESSION_UNITS = (
    "read-weighted mean index of the 18 uniform log(YFP:RFP) sort bins the sequence was "
    "observed in (bin index, not an absolute intensity; the RFP ratio normalizes "
    "extrinsic noise, and a bin scale is comparable within its own sort only)"
)

_PAPER = Provenance(
    source_uri="paper.md",
    citation_key=CITATION_KEY,
    sha256="80aa5de145c36b3617809db881b313bc0be69267259e5296bb617efd8bd97f09",
    method="MinerU OCR of the published PDF (mirror)",
)
_DEBOER = Provenance(
    source_uri="paper.md",
    citation_key="deboerDecipheringEukaryoticGeneregulatory2020",
    sha256="9caa9d7d237a7493356ea309c710b9245750e05d8ad1265c9a3fa1112642ceb1",
    method="MinerU OCR of the published PDF (mirror); the method's deferral target",
)

#: Every schema value this loader hardcodes, each with the verbatim quote behind it.
SOURCED_VALUES: dict[str, SourcedValue] = {
    "construct": SourcedValue(
        value=CONSTRUCT_NAME,
        provenance=_PAPER,
        quote=(
            "cloned into the promoter of a YFP gene within a CEN plasmid (Addgene: "
            "127546)"
        ),
        note="the reporter, its plasmid and the episomal localization of the addition",
    ),
    "scaffold": SourcedValue(
        value=f"{SCAFFOLD_NAME} scaffold with an 80 bp variable region",
        provenance=_PAPER,
        quote=(
            "in which 80\u2009bp of DNA are embedded within a promoter construct and "
            "the associated expression is assayed in the S. cerevisiae"
        ),
        note=(
            "the variable region is 80 bp inside a constant construct; the loaded "
            "table's constant 17 bp and 13 bp flanks are that construct's pTpA "
            "scaffold, which Supplementary Table 3's pT-pA oligo defines"
        ),
    ),
    "strain": SourcedValue(
        value=STRAIN,
        provenance=_PAPER,
        quote=(
            "strain Y8205 for the training dataset of random sequences, and strain "
            "S288C::ura3 for all the rest of the sequences measured"
        ),
        note=(
            "the loaded table is the designed / evolution set, not the Y8205 random "
            "training set, which GEO releases as its own file"
        ),
    ),
    "media": SourcedValue(
        value="SC-Ura",
        provenance=_PAPER,
        quote=(
            "These expression measurements were carried out separately in defined "
            "medium lacking uracil (SD-Ura (Sunrise Science, 1703-500)) and complex "
            "medium (YPD: yeast extract, peptone, dextrose)."
        ),
        note=(
            "the loaded table is the defined-medium (SC-Ura) arm; de Boer 2020, the "
            "method's deferral target, names the same medium SC-Ura"
        ),
    ),
    "temperature": SourcedValue(
        value=30.0,
        provenance=_DEBOER,
        quote=(
            "All cultures were grown in a shaker incubator, at $3 0 ^ { \\circ } "
            "\\mathrm { C }$ and approximately $2 5 0 \\mathrm { r ."
        ),
        note=(
            "the pre-sort GPRA culture, in the paper this release's method defers to; "
            "the Vaishnav SI states the same sentence, and its own text states the "
            "same temperature for every culture (see temperature_paper). A shaking "
            "culture, so the environment is aerobic."
        ),
    ),
    "temperature_paper": SourcedValue(
        value=30.0,
        provenance=_PAPER,
        quote=(
            "Unless otherwise noted, yeast were grown at $3 0 ^ { \\circ } "
            "\\mathrm { C } ,$ , in an orbital shaker incubator at 225\u2009rpm."
        ),
        note="the same value stated by this paper itself, for every culture it reports",
    ),
    "expression_units": SourcedValue(
        value=EXPRESSION_UNITS,
        provenance=_PAPER,
        quote=(
            "For each sequence, the expression level was the average of the expression "
            "bins in which it was observed, weighted by the number of times it was "
            "observed in each bin."
        ),
        note="what the stored number IS, and what read_count counts",
    ),
    "bins": SourcedValue(
        value=18,
        provenance=_PAPER,
        quote=(
            "Cells were sorted into 18 uniformly sized expression bins covering the "
            "majority of the expression distribution."
        ),
        note="the scale the estimate is expressed on",
    ),
    "readout": SourcedValue(
        value=ReporterReadout.sort_seq_fluorescence.value,
        provenance=_DEBOER,
        quote=(
            "Expression levels for each promoter sequence were estimated as the "
            "weighted average of bins in which the promoter was observed."
        ),
        note=(
            "a pooled sort-seq estimate, not a per-well or per-cell reading, which is "
            "the ReporterReadout member #991 added"
        ),
    ),
    "internal_control": SourcedValue(
        value="log(YFP/RFP), the released ratio",
        provenance=_DEBOER,
        quote=(
            "dual reporter system expressing a constitutive red fluorescent protein "
            "(RFP) and a variable yellow fluorescent protein (YFP). log(YFP/RFP) "
            "measured using flow cytometry (see Methods) reports an expression signal "
            "that is integrated over several generations and normalized for extrinsic "
            "noise"
        ),
        note=(
            "why the reference restates the record: the control is inside the released "
            "number, and the loaded files carry no control-promoter arm"
        ),
    ),
}

_UNCERTAINTY_GAP_NOTE = (
    "the Zenodo defined-medium table releases sdEL, relativeSD and sdScale, and "
    "sdEL == sdScale * relativeSD on all 5,327 rows to 2.0e-14, so sdEL is an "
    "author-scaled quantity rather than a dispersion of the reads; neither this paper "
    "(Methods, Supplementary Notes) nor de Boer 2020, its method's deferral target, "
    "defines any of the three, so the uncertainty type cannot be labeled and no "
    "unlabeled uncertainty is stored"
)
_SAMPLE_GAP_NOTE = (
    "the estimate averages READS over sort bins, and a read is not a replicate of the "
    "construct; the release states no replicate design, and the read count is stored in "
    "read_count"
)


def phenotype_gaps() -> list[ProvenanceGap]:
    """The typed absences every record carries: the uncertainty and the replicate design."""
    return [
        ProvenanceGap(
            field="expression_level_uncertainty",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=_UNCERTAINTY_GAP_NOTE,
        ),
        ProvenanceGap(
            field="expression_level_uncertainty_type",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=_UNCERTAINTY_GAP_NOTE,
        ),
        ProvenanceGap(
            field="n_samples",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=_SAMPLE_GAP_NOTE,
        ),
        ProvenanceGap(
            field="sample_unit",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=_SAMPLE_GAP_NOTE,
        ),
    ]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/<citation key>/``, the deposit this loader reads."""
    base = data_root or os.environ["DATA_ROOT"]
    return Path(base) / "torchcell-raw" / CITATION_KEY


def _direct(url: str, sha: str, retrieved_at: str) -> RetrievalRecord:
    return RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=url,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": url},
        sha256=sha,
        retrieved_at=retrieved_at,
    )


def _deposit_one(
    root: Path, relpath: str, src: Path, url: str, retrieved_at: str
) -> ArtifactRecord:
    sha = sha256_file(src)
    dest = root / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256_file(dest) != sha:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(src, dest)
    return ArtifactRecord(
        path=relpath,
        role=ROLE_RAW_DATA,
        bytes=dest.stat().st_size,
        sha256=sha,
        source=url,
        retrieval=_direct(url, sha, retrieved_at),
    )


def deposit_raw_mirror(
    *, download_dir: str | Path, retrieved_at: str, data_root: str | None = None
) -> Path:
    """Write the raw mirror and its ``manifest.json`` from already-retrieved files.

    Run once after fetching the three objects into ``download_dir`` with the recorded
    direct URLs (the GEO listing is scriptable; the GEO ``acc.cgi`` page is not, which is
    why the ftp host is the recorded source). Idempotent by sha256; a file already in the
    mirror with other bytes raises rather than being overwritten.
    """
    root = raw_mirror_dir(data_root)
    download = Path(download_dir)
    plan = {
        MOLEVOL_REL: (download / MOLEVOL_NAME, GEO_URL_BASE + MOLEVOL_NAME),
        TESTDATA_REL: (
            download / TESTDATA_NAME,
            f"{ZENODO_URL_BASE}{TESTDATA_NAME}/content",
        ),
        ZENODO_README_REL: (
            download / ZENODO_README_NAME,
            f"{ZENODO_URL_BASE}{ZENODO_README_NAME}/content",
        ),
    }
    expected = {
        MOLEVOL_REL: MOLEVOL_SHA256,
        TESTDATA_REL: TESTDATA_SHA256,
        ZENODO_README_REL: ZENODO_README_SHA256,
    }
    files: list[ArtifactRecord] = []
    for relpath, (src, url) in plan.items():
        if sha256_file(src) != expected[relpath]:
            raise RuntimeError(
                f"{src} does not have the recorded sha256 {expected[relpath]}"
            )
        files.append(_deposit_one(root, relpath, src, url, retrieved_at))
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=files,
        si_data_sources=[GEO_URL_BASE, ZENODO_URL_BASE],
        si_expected=[MOLEVOL_NAME, TESTDATA_NAME, ZENODO_README_NAME],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def variable_region_span(sequence: str) -> tuple[int, int]:
    """1-based inclusive span of the variable region inside a released construct.

    The scaffold is constant in the loaded table, so the span is read off the flanks
    rather than assumed: a sequence that does not carry both, or whose variable region is
    not 80 bp, raises instead of being stored with a guessed span.
    """
    if not sequence.startswith(SCAFFOLD_5PRIME) or not sequence.endswith(
        SCAFFOLD_3PRIME
    ):
        raise RuntimeError(
            f"{sequence[:24]}... does not carry the pTpA scaffold flanks"
        )
    start = len(SCAFFOLD_5PRIME) + 1
    end = len(sequence) - len(SCAFFOLD_3PRIME)
    if end - start + 1 != VARIABLE_REGION_LENGTH:
        raise RuntimeError(
            f"variable region is {end - start + 1} bp, not {VARIABLE_REGION_LENGTH}"
        )
    return start, end


#: The verbatim cell the release writes where a listed sequence has no estimate.
RELEASED_ABSENT_CELL = "NA"
_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def _released_level(cell: str) -> float | None:
    """A release cell as a number, or None for the release's own absence marker.

    Read with no NA inference, so the cell is classified here and only here: the one
    marker the release uses (``NA``) is an absence the record carries as a typed gap,
    and any other non-numeric cell is a shape this loader has not seen and refuses.
    """
    if cell == RELEASED_ABSENT_CELL:
        return None
    if not _NUMBER.fullmatch(cell):
        raise RuntimeError(
            f"ExpressionLevel cell {cell!r} is neither a number nor "
            f"{RELEASED_ABSENT_CELL!r}"
        )
    return float(cell)


def read_measurements(
    molevol_path: str | Path, testdata_path: str | Path
) -> pd.DataFrame:
    """One row per released ``seqID``, as the release lists it, with its Zenodo columns.

    Every ``seqID`` the release lists is kept. A group whose rows carry ``NA`` keeps
    ``expression_level = None``; the record carries a typed gap for it. Enforced, not
    assumed: one sequence and one cell value per ``seqID`` (so no group mixes a number
    with ``NA``), every row's ``origID`` is a member of its ``seqID`` list (so one record
    per ``seqID`` drops no row's label), and every Zenodo row joins a distinct measured
    ``seqID`` with the same value.
    """
    table = pd.read_csv(molevol_path, sep="\t", keep_default_na=False, dtype=str)
    if list(table.columns) != ["origID", "seqID", "seq110", "ExpressionLevel"]:
        raise RuntimeError(
            f"unexpected columns in the GEO table: {list(table.columns)}"
        )
    if len(table) != EXPECTED_ROWS:
        raise RuntimeError(
            f"{len(table)} rows in the GEO table, expected {EXPECTED_ROWS}"
        )
    member = [
        orig in seq_id.split(";")
        for orig, seq_id in zip(table.origID, table.seqID, strict=True)
    ]
    if not all(member):
        raise RuntimeError("a row's origID is not in its own seqID list")
    groups = table.groupby("seqID")
    if groups.seq110.nunique().max() != 1:
        raise RuntimeError("a seqID carries more than one sequence")
    if groups.ExpressionLevel.nunique().max() != 1:
        raise RuntimeError("a seqID carries more than one ExpressionLevel cell")
    records = (
        table.groupby("seqID", sort=True)
        .agg(seq110=("seq110", "first"), cell=("ExpressionLevel", "first"))
        .reset_index()
    )
    records["expression_level"] = [_released_level(c) for c in records.cell]
    # pandas stores a None in a float column as NaN, so the absence is counted
    # and read back through isna(), never through an identity test.
    unmeasured = int(records.expression_level.isna().sum())
    if unmeasured != EXPECTED_UNMEASURED_SEQUENCES:
        raise RuntimeError(
            f"{unmeasured} seqID groups carry {RELEASED_ABSENT_CELL!r}, expected "
            f"{EXPECTED_UNMEASURED_SEQUENCES}"
        )
    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(f"{len(records)} records, expected {EXPECTED_RECORDS}")

    zenodo = pd.read_csv(testdata_path)
    measured = table[table.ExpressionLevel != RELEASED_ABSENT_CELL]
    joined = zenodo[["origID", "expt", "proTotal", "meanEL"]].merge(
        measured[["origID", "seqID", "ExpressionLevel"]], on="origID", how="left"
    )
    if joined.seqID.isna().any():
        raise RuntimeError("a Zenodo test-set row names an origID the GEO table lacks")
    if joined.seqID.duplicated().any():
        raise RuntimeError("two Zenodo test-set rows map to one seqID")
    drift = (joined.meanEL - joined.ExpressionLevel.astype(float)).abs().max()
    if drift > 1e-12:
        raise RuntimeError(
            f"the Zenodo test set disagrees with the GEO table by {drift} (not a "
            "float round-trip, so the two are different measurements)"
        )
    extra = joined.set_index("seqID")[["expt", "proTotal"]]
    out = records.join(extra, on="seqID")
    out["library_id"] = out["expt"].where(out["expt"].notna(), None)
    out["read_count"] = out["proTotal"]
    return out[["seqID", "seq110", "expression_level", "library_id", "read_count"]]


def environment() -> Environment:
    """SC-Ura at 30 C, shaking, with no environmental edit."""
    return Environment(
        media=SC_URA,
        temperature=Temperature(value=float(SOURCED_VALUES["temperature"].value)),
        aerobicity="aerobic",
    )


def genotype(sequence_id: str, sequence: str) -> Genotype:
    """The construct: a YFP reporter whose promoter carries this released sequence."""
    start, end = variable_region_span(sequence)
    return Genotype(
        perturbations=[
            VariablePromoterReporterPerturbation(
                systematic_gene_name=REPORTER_IDENTIFIER,
                gene_namespace="scerevisiae_s288c_sgd_systematic",
                perturbed_gene_name=REPORTER_GENE,
                source_organism=REPORTER_SOURCE_ORGANISM,
                is_heterologous=True,
                localization=LOCALIZATION,
                construct_name=CONSTRUCT_NAME,
                promoter_variant_id=sequence_id,
                promoter_sequence=sequence,
                scaffold_name=SCAFFOLD_NAME,
                variable_region_start=start,
                variable_region_end=end,
            )
        ]
    )


def absent_estimate_gap() -> ProvenanceGap:
    """The typed gap of a release row whose ExpressionLevel cell is ``NA``."""
    return ProvenanceGap(
        field="expression_level",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=Provenance(
            source_uri=f"$DATA_ROOT/torchcell-raw/{CITATION_KEY}/{MOLEVOL_REL}",
            citation_key=CITATION_KEY,
            sha256=MOLEVOL_SHA256,
            page="column ExpressionLevel",
        ),
        note=(
            f"the release lists this sequence as a row and writes "
            f"{RELEASED_ABSENT_CELL!r} in its ExpressionLevel cell: designed and listed, "
            "no estimate released"
        ),
    )


def phenotype(
    expression_level: float | None,
    sequence_id: str,
    library_id: str | None,
    read_count: int | None,
) -> PooledSortSeqExpressionPhenotype:
    """One sequence's released estimate, or the release's own absence as a typed gap."""
    gaps = phenotype_gaps()
    if expression_level is None:
        gaps.append(absent_estimate_gap())
    return PooledSortSeqExpressionPhenotype(
        expression_level=expression_level,
        sequence_id=sequence_id,
        library_id=library_id,
        read_count=read_count,
        readout=ReporterReadout.sort_seq_fluorescence,
        reporter_gene=REPORTER_GENE,
        expression_units=EXPRESSION_UNITS,
        provenance_gaps=gaps,
    )


@register_dataset
class PooledPromoterExpressionVaishnav2022Dataset(ExperimentDataset):
    """Vaishnav 2022 GPRA sort-seq expression of 71,158 promoter sequences in SC-Ura."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; no genome is needed (the genotype is a construct)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return PooledSortSeqExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return PooledSortSeqExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The GEO MolEvol table and the Zenodo defined-medium test set."""
        return list(RAW_FILES)

    def download(self) -> None:
        """Copy the two raw-mirror files into ``raw_dir`` and verify each sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        mirror = raw_mirror_dir()
        for name, relpath in (
            (MOLEVOL_NAME, MOLEVOL_REL),
            (TESTDATA_NAME, TESTDATA_REL),
        ):
            dest = Path(self.raw_dir) / name
            if dest.exists():
                continue
            src = mirror / relpath
            if not src.exists():
                raise RuntimeError(
                    f"raw mirror file not found: {src}. Deposit it first: python -m "
                    "torchcell.datasets.scerevisiae.vaishnav2022 deposit "
                    "--download-dir <dir>"
                )
            if sha256_file(src) != RAW_FILES[name]:
                raise RuntimeError(f"{src} does not have the recorded sha256")
            shutil.copy2(src, dest)
            log.info("copied %s (sha256 %s)", dest, RAW_FILES[name])

    @post_process
    def process(self) -> None:
        """Build one record per measured ``seqID`` and write the LMDB."""
        verify_raw_files(self.raw_dir, RAW_FILES)
        measurements = read_measurements(
            Path(self.raw_dir) / MOLEVOL_NAME, Path(self.raw_dir) / TESTDATA_NAME
        )
        env = environment()
        publication = Publication(
            pubmed_id=PAPER_PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PAPER_PMID}/",
            doi=PAPER_DOI,
            doi_url=f"https://doi.org/{PAPER_DOI}",
        )
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=STRAIN
        )
        pub_dump = publication.model_dump()
        env_dump = env.model_dump()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        lmdb_env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))
        with lmdb_env.begin(write=True) as txn:
            for index, row in enumerate(
                tqdm(
                    measurements.itertuples(index=False),
                    total=len(measurements),
                    desc="vaishnav2022",
                )
            ):
                sequence_id = str(row.seqID)
                read_count = (
                    None
                    if pd.isna(row.read_count)
                    else int(cast(float, row.read_count))
                )
                library_id = None if row.library_id is None else str(row.library_id)
                level = (
                    None
                    if pd.isna(row.expression_level)
                    else float(cast(float, row.expression_level))
                )
                measured = phenotype(level, sequence_id, library_id, read_count)
                experiment = PooledSortSeqExpressionExperiment(
                    dataset_name=self.name,
                    genotype=genotype(sequence_id, str(row.seq110)),
                    environment=env,
                    phenotype=measured,
                )
                reference = PooledSortSeqExpressionExperimentReference(
                    dataset_name=self.name,
                    genome_reference=genome_reference,
                    environment_reference=env.model_copy(),
                    phenotype_reference=measured,
                )
                txn.put(
                    f"{index}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": pub_dump,
                        }
                    ),
                )
        lmdb_env.close()
        measurements.assign(environment=str(env_dump["media"]["name"])).drop(
            columns=["seq110"]
        ).to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)
        log.info("Wrote %d Vaishnav 2022 sort-seq records to LMDB", len(measurements))

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing happens inside ``process`` for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is inline in ``process`` for this dataset."""
        raise NotImplementedError


def run_verification(data_root: str | None = None) -> Any:
    """Run the pooled sort-seq verifier (L0-L4) on the built dev LMDB."""
    from torchcell.verification.runners import (
        _write_report,
        verify_pooled_sort_seq_dataset,
    )

    base = data_root or os.environ["DATA_ROOT"]
    root = osp.join(base, DATASET_ROOT_REL)
    report = verify_pooled_sort_seq_dataset(
        "pooled_promoter_expression_vaishnav2022", base, abs_root=root
    )
    _write_report(report, osp.join(root, "preprocess"))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.scerevisiae.vaishnav2022"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser("deposit", help="deposit the retrieved files")
    deposit.add_argument("--download-dir", required=True)
    deposit.add_argument("--retrieved-at", default=datetime.now(UTC).date().isoformat())
    sub.add_parser("build", help="build the dev LMDB")
    sub.add_parser("verify", help="verify the built dev LMDB")
    args = parser.parse_args(argv)
    data_root = os.environ["DATA_ROOT"]
    if args.command == "deposit":
        root = deposit_raw_mirror(
            download_dir=args.download_dir, retrieved_at=args.retrieved_at
        )
        print(f"deposited {root}")
        return 0
    if args.command == "build":
        dataset = PooledPromoterExpressionVaishnav2022Dataset(
            root=osp.join(data_root, DATASET_ROOT_REL)
        )
        print(f"len = {len(dataset)}")
        return 0
    report = run_verification(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
