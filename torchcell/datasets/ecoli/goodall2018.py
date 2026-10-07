# torchcell/datasets/ecoli/goodall2018
# [[torchcell.datasets.ecoli.goodall2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/goodall2018
# Test file: tests/torchcell/datasets/ecoli/test_goodall2018.py
"""Goodall 2018 TraDIS gene essentiality of E. coli K-12 BW25113 (rank 11 of the fifty).

Goodall, Robinson, Johnston, Jabbari, Turner, Cunningham, Lund, Cole and Henderson 2018
(mBio 9:e02096-17, doi:10.1128/mBio.02096-17, citation key
``goodallEssentialGenomeEscherichia2018``) built a mini-Tn5 transposon library of about
3.7 million mutants in BW25113, sequenced the transposon junctions (TraDIS) and called
each protein-coding gene essential, non-essential or unclear from its INSERTION INDEX
(unique insertion sites in the CDS over the CDS length) by a two-mode mixture likelihood
ratio.

DATA. Two per-gene tables of the publisher SI, captured from the PMC Article Datasets
bucket by the literature mirror and copied into the raw mirror with the same retrieval
record: Table S1 (``si2.xlsx``) for the input library as plated (TL1 + TL2) and Table S4
(``si7.xlsx``) after outgrowth in LB broth (LB1 + LB2). Columns: ``Gene``, ``Insertion
Index Score``, ``Log Likelihood Ratio`` and one-hot ``Essential`` / ``Non-essential`` /
``Unclear``. The TraDIS reads (ENA PRJEB24436) are not consumed.

CONDITIONS. ``TL``: the mutants as selected on LB agar with chloramphenicol, the paper's
headline call (358 / 3,793 / 162, ``SOURCED_VALUES["tl_counts"]``, required at build
time). ``LB``: the same library after 5 or 6 generations in LB broth at 37 C with
shaking; these cells passed the TL selection first, which ``Environment`` cannot state.

STRAIN AND IDENTIFIERS. BW25113 (``SOURCED_VALUES["background_strain"]``), mapped to
CP009273.1, the replicon of the ``ecoli_K12_BW25113_ASM75055v1`` set (checked at build
time). ``Gene`` holds BW25113 GenBank gene symbols (``SOURCED_VALUES["gene_names"]``), so
``reconcile_locus_tags`` resolves them at the gene-symbol layer; a name not on exactly one
BW25113 locus (the multi-copy insertion-sequence symbols) is dropped, with its candidates
ledgered.

GENOTYPE. One ``TransposonInsertionPerturbation`` per record, at the GENE level: the
record stands for every insertion mutant of that gene in the pool. ``transposon`` is
"mini-Tn5"; ``barcode`` is None because TraDIS has no per-strain barcode (the inline index
marks a sequenced sample); ``insertion_position`` / ``insertion_strand`` are None because a
gene-level record aggregates many insertions and the tables release no position. The leaf
has no ``provenance_gaps`` field, so those absences are typed in
``PERTURBATION_FIELD_GAPS`` and written to ``preprocess/perturbation_field_gaps.json``.

PHENOTYPE. ``GeneEssentialityPhenotype.is_essential`` is the released call (True for
"essential", False for "non-essential"). The class has no field for an "unclear" call, the
insertion index, the likelihood ratio, ``n_samples`` or an uncertainty, so unclear rows
are dropped under a counted rule and every released value stays in
``preprocess/calls.csv``. The replicate structure is sourced anyway (two sequenced
samples per condition, pooled before calling; ``SOURCED_VALUES["tl_replicates"]``,
``["lb_replicates"]``); there is no per-gene uncertainty to type. The released calls
follow the PRINTED threshold of +/-3.6 exactly, not log2(12) = 3.585
(``SOURCED_VALUES["call_threshold"]``; enforced at build time).

REFERENCE. The unperturbed BW25113 parent in the same environment, viable
(``is_essential=False``): the library was constructed in it and selected on the TL plates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import os.path as osp
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import openpyxl
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
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import LB
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AssemblyReferenceGenome,
    BacterialGeneEssentialityExperiment,
    BacterialGeneEssentialityExperimentReference,
    Environment,
    Experiment,
    ExperimentReference,
    GeneEssentialityPhenotype,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    Publication,
    Temperature,
    TransposonInsertionPerturbation,
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
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.levels import l0_structural, l1_count
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
CITATION_KEY = "goodallEssentialGenomeEscherichia2018"
PAPER_DOI = "10.1128/mBio.02096-17"
PMCID = "PMC5821084"
#: The PubMed id of PMC5821084, read from the PMC id converter
#: (``pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=PMC5821084``) on 2026-10-07.
PUBMED_ID = "29463657"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "facfe1fac8ab6dab5cd0ecb88876b33616cc00de7c1fe68b60590f7a1883b429"

TABLE_S1 = "si2.xlsx"
TABLE_S4 = "si7.xlsx"
#: When the literature mirror captured both tables (copied from its ``manifest.json``).
DATA_RETRIEVED_AT = "2026-10-07T11:43:37.139541+00:00"
#: The BW25113 chromosome the reads were mapped to (``SOURCED_VALUES["reference_accession"]``).
PAPER_REPLICON = "CP009273.1"


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


def _pmc_file(name: str, key: str, sha256: str, size: int, description: str) -> RawFile:
    """A publisher SI file of the PMC Article Datasets bucket (``pmc_cloud``)."""
    return RawFile(
        name=name,
        sha256=sha256,
        bytes=size,
        description=description,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=f"https://pmc-oa-opendata.s3.amazonaws.com/{key}",
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": key},
            sha256=sha256,
            retrieved_at=DATA_RETRIEVED_AT,
        ),
    )


RAW_FILES: tuple[RawFile, ...] = (
    _pmc_file(
        TABLE_S1,
        f"{PMCID}.1/mbo001183726st1.xlsx",
        "db85743f0e28264751fb9adfe40d6ef1e333307c3a65cd315813066e57d3de15",
        257700,
        "Table S1: per-gene insertion index, log likelihood ratio and essentiality call "
        "of the input transposon library (TL1 and TL2 pooled)",
    ),
    _pmc_file(
        TABLE_S4,
        f"{PMCID}.1/mbo001183726st4.xlsx",
        "8a4c6eb83eafd4da8bfd1f2443acd4de4c1e5e5cb915b334a18725308f7eeb5f",
        270726,
        "Table S4: the same columns after outgrowth of the library in LB (LB1 and LB2 "
        "pooled)",
    ),
)
#: ``{raw file name: pinned sha256}``, the build-time check of every consumed file.
DATA_SHA256: dict[str, str] = {f.name: f.sha256 for f in RAW_FILES}
RAW_FILES_BY_NAME: dict[str, RawFile] = {f.name: f for f in RAW_FILES}

#: Released data deliberately not mirrored (the loader does not read it).
NOT_MIRRORED = (
    "ENA PRJEB24436 TraDIS reads (TL1, TL2, LB1, LB2): not consumed; per-insertion "
    "positions and orientations would be re-derived from them, and the loader reads only "
    "the released per-gene tables",
    "Tables S2 and S3 (si3.pdf, si6.pdf in the literature mirror): the Keio/PEC "
    "comparison gene lists and the discrepancy causes, not consumed",
)


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the sha256-pinned Goodall ``paper.md``."""
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


# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_LB_RESULTS_QUOTE = (
    "two independent samples of the transposon library were grown in Luria broth (LB) "
    "at $3 7 ^ { \\circ } \\mathsf { C }$ for 5 or 6 generations to an optical density "
    "at $6 0 0 ~ \\mathsf { n m }$ $( \\mathrm { O D } _ { 6 0 0 } )$ of 1.0 and were then "
    "sequenced."
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "background_strain": _paper(
        "BW25113",
        "E. coli K-12 strain BW25113, the parent strain of the Keio library, was used "
        "for construction of a transposon library.",
        note="Materials and Methods, 'Strains and plasmids'",
    ),
    "reference_accession": _paper(
        PAPER_REPLICON,
        "Trimmed, filtered sequences were then aligned to the reference genome E. coli "
        "BW25113 (accession no. CP009273.1), obtained from the NCBI genome repository "
        "(69).",
        note="CP009273.1 is the replicon of ecoli_K12_BW25113_ASM75055v1 "
        "(GCA_000750555.1); process() requires every BW25113 locus to sit on it",
    ),
    "gene_names": _paper(
        "BW25113 GenBank gene symbols",
        "Where gene names differed between databases, the BW25113 annotation was used.",
        note="so the Gene column is resolved at the gene-symbol layer of the BW25113 "
        "annotation, with no cross-strain (MG1655 b-number or ECK) mapping",
    ),
    "transposon": _paper(
        "mini-Tn5",
        "The main differences were that a mini-Tn 5 transposon coding for a "
        "chloramphenicol resistance cassette was used.",
        note="the OCR writes 'Tn 5'; the Results write 'A mini-Tn5 transposon'",
    ),
    "no_strain_barcode": _paper(
        "no per-strain barcode",
        "Raw data were checked for the presence of an inline index barcode to identify "
        "independently processed samples (Table 1).",
        note="the only barcode is a per-SAMPLE index; TraDIS counts transposon junctions, "
        "so TransposonInsertionPerturbation.barcode is None because none exists",
    ),
    "tl_growth": _paper(
        "overnight on selective medium",
        "A mini-Tn5 transposon with a chloramphenicol resistance cassette was "
        "transformed into competent cells and grown overnight on selective medium.",
        note="'overnight' states no hours, so duration_hours is gapped",
    ),
    "selection_medium": _paper(
        "LB agar supplemented with chloramphenicol",
        "Transposon mutants were selected by growth on LB agar supplemented with "
        "chloramphenicol.",
        note="the TL environment. Its LB is taken to be the LB the paper states for its "
        "broth (the LB Miller amounts of torchcell.datamodels.media.LB, which already "
        "cites this paper); no agar amount, chloramphenicol dose or incubation "
        "temperature is stated for the plates",
    ),
    "library_size": _paper(
        3_700_000,
        "Individual colonies were pooled to construct the initial library, estimated "
        "to consist of approximately 3.7 million mutants.",
    ),
    "tl_replicates": _paper(
        2,
        "DNA was extracted from two samples of the transposon library glycerol stock to "
        "generate TraDIS data referred to as TL1 and TL2 in the text.",
        note="two DNA extracts of one pooled stock; GeneEssentialityPhenotype has no "
        "n_samples field, so this is recorded here and in the note, not on records",
    ),
    "replicate_kind": _paper(
        "technical",
        "Correlation coefficients of gene insertion index scores for two sequenced "
        "technical replicates of the input transposon library (TL1 and TL2) (B) and "
        "following growth in LB (LB1 and LB2) (C).",
        note="the Fig. 1 legend calls both pairs technical replicates",
    ),
    "lb_replicates": _paper(
        2,
        "In addition, DNA was extracted from two independent cultures, LB1 and LB2, of "
        "the library grown in Luria broth (LB)",
        note="the Methods say independent cultures, the Results and Fig. 1 legend say "
        "technical replicates; recorded, not resolved",
    ),
    "tl_pooled": _paper(
        "TL1 and TL2 combined before calling",
        "The data were therefore combined to give a total of 8,279,309 sequences that "
        "were mapped to 901,383 unique insertion sites throughout the genome.",
        note="one call per gene from the pooled reads; no per-replicate call is "
        "released, so there is no per-gene spread to carry",
    ),
    "lb_pooled": _paper(
        "LB1 and LB2 combined before calling",
        "As there was a high correlation coefficient of 0.97 between the gene insertion "
        "index scores of each technical replicate (Fig. 1C), the data were combined to "
        "give a pool of 10,584,188 sequences.",
    ),
    "insertion_index": _paper(
        "unique insertion sites in the CDS / CDS length in bases",
        "To normalize for gene length, the number of unique insertion points within the "
        "CDS was divided by the CDS length in bases. This value was termed the "
        "insertion index score",
        note="released in the 'Insertion Index Score' column; not stored on records "
        "(no field), kept in preprocess/calls.csv",
    ),
    "cds_only": _paper(
        "protein-coding sequences",
        "CDS is defined as the protein coding sequence of a gene, inclusive of the start "
        "and stop codons.",
        note="RNA genes are not in the tables (Table S3: 'RNA genes not considered in "
        "our analysis')",
    ),
    "call_rule": _paper(
        "12-fold likelihood between the two modes",
        "genes were assigned as “essential” if they were 12 times more likely to be in "
        "the left mode than in the right mode, and “nonessential” if they were 12 times "
        "more likely to be in the right mode",
    ),
    "call_threshold": _paper(
        3.6,
        "Genes with log likelihood scores between the upper and lower $\\mathsf { l o g "
        "} _ { 2 } 1 2$ threshold values of 3.6 and $- 3 . 6 ,$ , respectively, were "
        "deemed “unclear.”",
        note="measured: every released call of both tables agrees with ratio < -3.6 "
        "essential, > 3.6 non-essential, else unclear; the exact log2(12) = 3.585 would "
        "move one unclear gene per table (ydhW in S1, grxD in S4) to non-essential. "
        "process() refuses any row that disagrees with 3.6",
    ),
    "tl_counts": _paper(
        {"essential": 358, "non_essential": 3793, "unclear": 162},
        "Using this approach, sufficient insertions were found in 3,793 genes for them "
        "to be classed as nonessential, 162 genes were situated between the two modes "
        "and classed as unclear, and 358 genes in the mutant library were identified as "
        "essential (Table S1).",
        note="process() requires Table S1 to carry exactly these counts",
    ),
    "essential_definition": _paper(
        "the CDS, or a portion of it, is required for growth under the tested conditions",
        "For the purposes of this study, we define a gene as essential if the "
        "transposon insertion data reveal that the protein coding sequence (CDS), or a "
        "portion of the CDS, is required for growth under the conditions tested here.",
    ),
    "left_mode": _paper(
        "essential or a very severe fitness cost",
        "which have a low number of transposon insertions, are either essential for "
        "survival or genes that, when disrupted, confer a very severe fitness cost "
        "(Fig. 1D).",
        note="what is_essential=True means for these records: the left (low-insertion) "
        "mode, which includes severe growth defects, not only lethality",
    ),
    "lb_table": _paper(
        "Table S4", "Insertion index scores were calculated as before (Table S4)."
    ),
    "lb_temperature_c": _paper(37.0, _LB_RESULTS_QUOTE),
    "lb_generations": _paper(
        "5 or 6",
        _LB_RESULTS_QUOTE,
        note="no per-culture value; the Methods sentence lost the number in the OCR "
        "('grown for generations'), so duration_generations is gapped rather than "
        "set to either end",
    ),
    "lb_shaking": _paper(
        "aerobic",
        "$3 7 ^ { \\circ } \\mathsf { C }$ with shaking until the culture reached an "
        "optical density at $6 0 0 \\ \\mathrm { n m } \\ ( \\mathrm { O D } _ { 6 0 0 } )$ "
        "of 1.0.",
        note="shaken broth cultures, harvested at OD600 1.0",
    ),
    "data_availability": _paper(
        "PRJEB24436",
        "TraDIS sequencing data are available from the European Nucleotide Archive "
        "under accession no. PRJEB24436.",
        note="the reads, not consumed (NOT_MIRRORED)",
    ),
}

TRANSPOSON: str = SOURCED_VALUES["transposon"].value
CALL_THRESHOLD: float = SOURCED_VALUES["call_threshold"].value
EXPECTED_TL_COUNTS: dict[str, int] = SOURCED_VALUES["tl_counts"].value
LB_TEMPERATURE_C: float = SOURCED_VALUES["lb_temperature_c"].value
LB_AEROBICITY: str = SOURCED_VALUES["lb_shaking"].value
SELECTION_MEDIUM_STATEMENT = SOURCED_VALUES["selection_medium"]
#: Checklist item 4: below this fraction of distinct Table S1/S4 gene names resolving to
#: one BW25113 locus the build stops. The paper states the names ARE BW25113 annotation
#: symbols, so a lower fraction would mean the wrong annotation, not a few multi-copy
#: insertion-sequence names (measured 4,260 of 4,269 = 0.9979).
MIN_RESOLVED_FRACTION = 0.99

_PAPER_LOOKED_IN = Provenance(
    source_uri=PAPER_MD,
    citation_key=CITATION_KEY,
    sha256=PAPER_MD_SHA256,
    method="full Results and Materials and Methods read",
)

TL_TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_LOOKED_IN,
    note="no incubation temperature is stated for the chloramphenicol selection "
    "plates; 37 C is stated only for the LB outgrowth and the beta-galactosidase "
    "cultures",
)
TL_DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_LOOKED_IN,
    note="the plates were 'grown overnight on selective medium'; no hours are stated",
)
LB_GENERATIONS_GAP = ProvenanceGap(
    field="duration_generations",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_LOOKED_IN,
    note="'5 or 6 generations' (Results), with no per-culture value; the Methods "
    "sentence's number is lost in the OCR. A generation count is part of the "
    "environment identity, so neither end is asserted",
)

_TABLES_LOOKED_IN = Provenance(
    source_uri=f"{RAW_DIR_REL}/data/{TABLE_S1}",
    citation_key=CITATION_KEY,
    sha256=DATA_SHA256[TABLE_S1],
    method="every column of Table S1 (Table S4 has the same six columns)",
)
#: Typed absences of the transposon leaf's coordinate fields. The leaf carries no
#: ``provenance_gaps`` slot, so they live here and in
#: ``preprocess/perturbation_field_gaps.json``, and every record's value is None.
PERTURBATION_FIELD_GAPS: tuple[ProvenanceGap, ...] = tuple(
    ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_TABLES_LOOKED_IN,
        note="a record is one GENE, standing for every insertion mutant of it in the "
        "pool, so no single insertion site exists for it; the tables release per-gene "
        "counts only, and per-insertion sites would come from the reads (ENA "
        "PRJEB24436), which are not consumed",
    )
    for field in ("insertion_position", "insertion_strand")
)

PUBLICATION = Publication(
    pubmed_id=PUBMED_ID,
    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
    doi=PAPER_DOI,
    doi_url=f"https://doi.org/{PAPER_DOI}",
)


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never a live URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/goodallEssentialGenomeEscherichia2018``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/goodallEssentialGenomeEscherichia2018``."""
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
    a mirror file with the pinned hash is left alone, and one with any other hash raises
    rather than being overwritten.
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
        title="The Essential Genome of Escherichia coli K-12",
        files=records,
        si_data_sources=[r.retrieval.source_url or r.name for r in RAW_FILES],
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


# --------------------------------------------------------------------------- #
# Parsing the released tables
# --------------------------------------------------------------------------- #
ConditionKey = Literal["TL", "LB"]
EssentialityCall = Literal["essential", "non_essential", "unclear"]
CALLS: tuple[EssentialityCall, ...] = ("essential", "non_essential", "unclear")
TABLE_COLUMNS = (
    "Gene",
    "Insertion Index Score",
    "Log Likelihood Ratio",
    "Essential",
    "Non-essential",
    "Unclear",
)


class ConditionSpec(BaseModel):
    """One released table: which condition it is and the title cell that says so."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: ConditionKey
    table: str
    title: str
    description: str


CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec(
        key="TL",
        table=TABLE_S1,
        title="Table S1. Essentiality classification for genes of the TL data",
        description="the input transposon library as selected on LB agar with "
        "chloramphenicol (TL1 and TL2 pooled)",
    ),
    ConditionSpec(
        key="LB",
        table=TABLE_S4,
        title="Table S4. Essentiality classification for genes following outgrowth "
        "in LB",
        description="the library after 5 or 6 generations in LB broth at 37 C with "
        "shaking (LB1 and LB2 pooled)",
    ),
)
CONDITIONS_BY_KEY: dict[ConditionKey, ConditionSpec] = {c.key: c for c in CONDITIONS}


class TableFormatError(ValueError):
    """A released table whose title, header or a cell is not what the loader reads."""


class CallRuleError(ValueError):
    """A released call that disagrees with the printed likelihood-ratio threshold."""


class CallCountError(ValueError):
    """Table S1's call counts differ from the counts the paper states."""


class DuplicateRecordError(ValueError):
    """Two kept rows of one condition on the same BW25113 locus."""


class GeneCall(BaseModel):
    """One table row: a gene's insertion index, likelihood ratio and released call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: ConditionKey
    row: int
    gene: str
    insertion_index: float
    log_likelihood_ratio: float
    call: EssentialityCall


def _number(value: Any, where: str) -> float:
    """A finite numeric cell (an int or a float, never a bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TableFormatError(f"{where}: {value!r} is not a number")
    if not math.isfinite(value):
        raise TableFormatError(f"{where}: {value!r} is not finite")
    return float(value)


def read_calls(path: str | Path, spec: ConditionSpec) -> list[GeneCall]:
    """Every data row of one released table, refusing an unexpected title, header,
    flag pattern (exactly one of the three call columns must be True) or number.

    ``row`` is 1-based over the data rows (the sheet row minus the title and header).
    openpyxl pads every row to the sheet's width, so a stray extra column fails the
    header check.
    """
    workbook = openpyxl.load_workbook(path, data_only=True)
    if len(workbook.worksheets) != 1:
        raise TableFormatError(f"{spec.table}: {len(workbook.worksheets)} sheets")
    rows = list(workbook.worksheets[0].iter_rows(values_only=True))
    if rows[0][0] != spec.title:
        raise TableFormatError(
            f"{spec.table}: title {rows[0][0]!r}, not {spec.title!r}"
        )
    if tuple(rows[1]) != TABLE_COLUMNS:
        raise TableFormatError(f"{spec.table}: header {rows[1]!r}")
    calls: list[GeneCall] = []
    for row, values in enumerate(rows[2:], start=1):
        where = f"{spec.table} data row {row}"
        gene, index, ratio, essential, non_essential, unclear = values
        if not isinstance(gene, str) or gene != gene.strip() or not gene:
            raise TableFormatError(f"{where}: gene {gene!r}")
        flags = (essential, non_essential, unclear)
        if not all(isinstance(flag, bool) for flag in flags) or sum(flags) != 1:
            raise TableFormatError(f"{where}: call flags {flags!r}")
        insertion_index = _number(index, where)
        if insertion_index < 0:
            raise TableFormatError(f"{where}: insertion index {insertion_index} < 0")
        calls.append(
            GeneCall(
                condition=spec.key,
                row=row,
                gene=gene,
                insertion_index=insertion_index,
                log_likelihood_ratio=_number(ratio, where),
                call=CALLS[flags.index(True)],
            )
        )
    return calls


def call_from_ratio(ratio: float, threshold: float) -> EssentialityCall:
    """The call a log likelihood ratio gets at ``threshold`` (negative = essential)."""
    if ratio < -threshold:
        return "essential"
    if ratio > threshold:
        return "non_essential"
    return "unclear"


class CallRuleCheck(BaseModel):
    """How one table's released calls compare with the printed threshold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: ConditionKey
    threshold: float
    n_rows: int
    n_agree: int
    disagreements: list[str]
    moved_at_exact_log2_12: list[str]


def call_rule_check(
    calls: Sequence[GeneCall], condition: ConditionKey, threshold: float
) -> CallRuleCheck:
    """Compare each released call with ``call_from_ratio`` at ``threshold``, and list the
    rows that the exact log2(12) cut would classify differently from ``threshold``.
    """
    exact = math.log2(12)
    disagreements = [
        f"{c.gene} (row {c.row}): ratio {c.log_likelihood_ratio} released {c.call}"
        for c in calls
        if call_from_ratio(c.log_likelihood_ratio, threshold) != c.call
    ]
    moved = [
        f"{c.gene} (row {c.row}): ratio {c.log_likelihood_ratio}"
        for c in calls
        if call_from_ratio(c.log_likelihood_ratio, exact)
        != call_from_ratio(c.log_likelihood_ratio, threshold)
    ]
    return CallRuleCheck(
        condition=condition,
        threshold=threshold,
        n_rows=len(calls),
        n_agree=len(calls) - len(disagreements),
        disagreements=disagreements,
        moved_at_exact_log2_12=moved,
    )


def call_counts(calls: Sequence[GeneCall]) -> dict[str, int]:
    """``{call: rows}`` over the three calls, zeros included."""
    counts = Counter(c.call for c in calls)
    return {call: counts[call] for call in CALLS}


# --------------------------------------------------------------------------- #
# Identifiers and the retention ledger
# --------------------------------------------------------------------------- #
DropRuleName = Literal[
    "symbol_not_on_one_bw25113_locus", "unclear_call_not_representable"
]
DROP_RULE_DESCRIPTIONS: dict[DropRuleName, str] = {
    "symbol_not_on_one_bw25113_locus": "the table's gene name does not resolve to exactly "
    "one BW25113 GenBank locus (ambiguous over several loci, retired, or shared with "
    "another resolving name), so no locus tag of the pinned namespace names the "
    "disrupted gene; each item gives the row and the candidates",
    "unclear_call_not_representable": "the released call is 'Unclear' (ratio within "
    "+/-3.6) and GeneEssentialityPhenotype stores only a boolean is_essential, so the "
    "call has no field; the row's values stay in preprocess/calls.csv",
}


class RowOutcome(BaseModel):
    """What happened to one table row: its locus and record, or the rule that dropped it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call: GeneCall
    locus_tag: str | None
    symbol: str | None
    drop_rule: DropRuleName | None
    reason: str | None


class DropRule(BaseModel):
    """One retention rule, the rows it removed, and why."""

    rule: DropRuleName
    description: str
    n_records: int
    items: list[str] = []


class ConditionLedger(BaseModel):
    """Per-condition counts: released calls and kept records."""

    condition: ConditionKey
    table: str
    source_rows: int
    released_calls: dict[str, int]
    kept_records: int
    kept_calls: dict[str, int]


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    conditions: list[ConditionLedger]
    rules: list[DropRule]


class IdentifierLedger(BaseModel):
    """The reconciliation report of every distinct table name, plus the stop threshold."""

    reconciliation: LocusTagReconciliation
    min_resolved_fraction: float


def canonical_symbol(genome: EcoliK12Genome, tag: str) -> str:
    """The annotation's own gene symbol of ``tag`` when it resolves back to ``tag``,
    else the tag itself, so the stored pair always round-trips through the resolver.
    """
    symbol = genome.genbank.loci[tag].symbol
    if symbol is None:
        return tag
    resolution = genome.resolve_gene_name(symbol)
    return symbol if resolution.systematic_name == tag else tag


def check_replicon(genome: EcoliK12Genome) -> None:
    """Refuse a genome whose loci are not all on the replicon the paper mapped to."""
    replicons = sorted({locus.replicon for locus in genome.genbank.loci.values()})
    if replicons != [PAPER_REPLICON]:
        raise ValueError(
            f"{genome.ASSEMBLY_SET} loci sit on {replicons}; the paper mapped to "
            f"{PAPER_REPLICON}"
        )


def _unplaced_reason(name: str, report: LocusTagReconciliation) -> str:
    """Why a kept-as-given name is on no single locus."""
    if name in report.ambiguous_kept:
        return "ambiguous over " + ", ".join(report.ambiguous_kept[name])
    if name in report.kept_on_collision:
        return "kept as given on collision with another resolving name"
    return "retired"


def resolve_calls(
    genome: EcoliK12Genome, calls: Sequence[GeneCall], *, label: str
) -> tuple[list[RowOutcome], LocusTagReconciliation]:
    """Reconcile every row's gene name on ``genome`` and decide each row's outcome.

    Stops (``LocusTagResolutionError``) below ``MIN_RESOLVED_FRACTION`` of distinct
    names. Rules, first match wins: a name off the BW25113 namespace
    (``symbol_not_on_one_bw25113_locus``), then an unclear call
    (``unclear_call_not_representable``). Kept rows carry the locus tag and the
    annotation's own symbol.
    """
    names = pd.Series([c.gene for c in calls], dtype=object)
    stored, report = reconcile_locus_tags(genome, names, label=label)
    report.require_resolved(MIN_RESOLVED_FRACTION)
    pattern = LOCUS_TAG_PATTERNS[report.gene_namespace]
    outcomes: list[RowOutcome] = []
    for call, tag in zip(calls, stored.tolist(), strict=True):
        if pattern.match(tag) is None:
            outcomes.append(
                RowOutcome(
                    call=call,
                    locus_tag=None,
                    symbol=None,
                    drop_rule="symbol_not_on_one_bw25113_locus",
                    reason=_unplaced_reason(call.gene, report),
                )
            )
            continue
        symbol = canonical_symbol(genome, tag)
        if call.call == "unclear":
            outcomes.append(
                RowOutcome(
                    call=call,
                    locus_tag=tag,
                    symbol=symbol,
                    drop_rule="unclear_call_not_representable",
                    reason=f"ratio {call.log_likelihood_ratio}",
                )
            )
            continue
        outcomes.append(
            RowOutcome(
                call=call, locus_tag=tag, symbol=symbol, drop_rule=None, reason=None
            )
        )
    seen: set[tuple[str, str]] = set()
    for outcome in outcomes:
        if outcome.drop_rule is not None or outcome.locus_tag is None:
            continue
        key = (outcome.call.condition, outcome.locus_tag)
        if key in seen:
            raise DuplicateRecordError(
                f"{outcome.call.condition}: two kept rows on {outcome.locus_tag}"
            )
        seen.add(key)
    return outcomes, report


def drop_rules(outcomes: Sequence[RowOutcome]) -> list[DropRule]:
    """One ``DropRule`` per rule name, in application order, with its row items."""
    rules: list[DropRule] = []
    for name, description in DROP_RULE_DESCRIPTIONS.items():
        items = [
            f"{o.call.condition} row {o.call.row} {o.call.gene}"
            + (f" ({o.locus_tag})" if o.locus_tag is not None else "")
            + f": {o.reason}"
            for o in outcomes
            if o.drop_rule == name
        ]
        rules.append(
            DropRule(
                rule=name, description=description, n_records=len(items), items=items
            )
        )
    return rules


def build_drop_log(dataset_name: str, outcomes: Sequence[RowOutcome]) -> DropLog:
    """The retention ledger of one build, refusing rules that miss a dropped row."""
    rules = drop_rules(outcomes)
    kept = [o for o in outcomes if o.drop_rule is None]
    conditions = [
        ConditionLedger(
            condition=spec.key,
            table=spec.table,
            source_rows=sum(1 for o in outcomes if o.call.condition == spec.key),
            released_calls=call_counts(
                [o.call for o in outcomes if o.call.condition == spec.key]
            ),
            kept_records=sum(1 for o in kept if o.call.condition == spec.key),
            kept_calls=call_counts(
                [o.call for o in kept if o.call.condition == spec.key]
            ),
        )
        for spec in CONDITIONS
    ]
    log_ = DropLog(
        dataset=dataset_name,
        source_records=len(outcomes),
        kept_records=len(kept),
        dropped_records=len(outcomes) - len(kept),
        conditions=conditions,
        rules=rules,
    )
    if sum(r.n_records for r in rules) != log_.dropped_records:
        raise RuntimeError("drop rules do not account for every dropped row")
    return log_


# --------------------------------------------------------------------------- #
# Environments and records (pure, no files)
# --------------------------------------------------------------------------- #
def selection_medium() -> Media:
    """The TL selection plates: the paper's LB, plus agar and chloramphenicol whose
    amounts the paper does not state (concentration None on both).

    ``LB_AGAR`` is not reused: its 2% (w/v) agar is Menasalvas 2025's and Schmidt
    2016's bench value, and asserting it for these plates would fabricate a number.
    The medium derives from the ``LB`` library key, so it joins there.
    """
    statement = SELECTION_MEDIUM_STATEMENT
    return Media(
        name="LB agar with chloramphenicol (Goodall 2018 transposon selection plates; "
        "agar and chloramphenicol amounts unstated)",
        state="solid",
        is_synthetic=False,
        base_medium="LB",
        components=[
            *LB.components,
            MediaComponent(
                compound=resolved_compound("agar"),
                role=MediaComponentRole.gelling_agent,
                concentration=None,
                provenance=[statement],
                note="no agar amount is stated for these plates",
            ),
            MediaComponent(
                compound=resolved_compound("chloramphenicol"),
                role=MediaComponentRole.selection_agent,
                concentration=None,
                provenance=[statement],
                note="selects for the transposon's chloramphenicol resistance "
                "cassette; no dose is stated",
            ),
        ],
        provenance=[statement],
    )


def environment(condition: ConditionKey) -> Environment:
    """The environment of one condition.

    TL: the selection plates; temperature and duration are typed gaps, and
    ``aerobicity`` keeps the field default (plates incubated in air are not stated, but
    the field cannot be None). LB: LB Miller broth at 37 C, shaken, with the generation
    count gapped.
    """
    if condition == "TL":
        return Environment(
            media=selection_medium(),
            provenance_gaps=[TL_TEMPERATURE_GAP, TL_DURATION_GAP],
        )
    return Environment(
        media=LB,
        temperature=Temperature(value=LB_TEMPERATURE_C),
        aerobicity=LB_AEROBICITY,
        provenance_gaps=[LB_GENERATIONS_GAP],
    )


def insertion_genotype(locus_tag: str, symbol: str) -> Genotype:
    """The gene-level transposon disruption of one BW25113 locus."""
    return Genotype(
        perturbations=[
            TransposonInsertionPerturbation(
                systematic_gene_name=locus_tag,
                perturbed_gene_name=symbol,
                gene_namespace=STRAIN_GENE_NAMESPACES["BW25113"],
                transposon=TRANSPOSON,
            )
        ]
    )


def build_experiment(
    dataset_name: str, outcome: RowOutcome, env: Environment
) -> BacterialGeneEssentialityExperiment:
    """The record of one kept row: essential is True, non-essential is False."""
    if (
        outcome.drop_rule is not None
        or outcome.locus_tag is None
        or outcome.symbol is None
    ):
        raise ValueError(f"row {outcome.call.row} was dropped; it has no record")
    return BacterialGeneEssentialityExperiment(
        dataset_name=dataset_name,
        genotype=insertion_genotype(outcome.locus_tag, outcome.symbol),
        environment=env,
        phenotype=GeneEssentialityPhenotype(
            is_essential=outcome.call.call == "essential"
        ),
    )


def build_reference(
    dataset_name: str, genome_reference: AssemblyReferenceGenome, env: Environment
) -> BacterialGeneEssentialityExperimentReference:
    """The unperturbed BW25113 parent in the same environment: viable."""
    return BacterialGeneEssentialityExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=env.model_copy(),
        phenotype_reference=GeneEssentialityPhenotype(is_essential=False),
    )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class GeneEssentialityGoodall2018Dataset(ExperimentDataset):
    """TraDIS essentiality calls of E. coli BW25113 genes, as plated and after LB."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = "data/torchcell/gene_essentiality_goodall2018",
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
        return BacterialGeneEssentialityExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialGeneEssentialityExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Every consumed file, linked from the raw mirror."""
        return [raw.name for raw in RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking the manifest and sha256.

        The mirror plus ``DATA_SHA256`` is canonical; the PMC bucket URLs are retrieval
        metadata that ``retrieve_raw_files`` re-runs, never a build input.
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
        log.info(
            "Goodall 2018 raw files linked into %s (sha256 verified)", self.raw_dir
        )

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
        """Parse Tables S1 and S4 into per-gene, per-condition records + LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        calls: list[GeneCall] = []
        checks: list[CallRuleCheck] = []
        for spec in CONDITIONS:
            table = read_calls(self._raw(spec.table), spec)
            check = call_rule_check(table, spec.key, CALL_THRESHOLD)
            if check.disagreements:
                raise CallRuleError(
                    f"{spec.table}: {len(check.disagreements)} calls disagree with "
                    f"+/-{CALL_THRESHOLD}: {check.disagreements[:5]}"
                )
            checks.append(check)
            calls.extend(table)
        tl_counts = call_counts([c for c in calls if c.condition == "TL"])
        if tl_counts != EXPECTED_TL_COUNTS:
            raise CallCountError(
                f"{TABLE_S1} counts {tl_counts}, the paper states {EXPECTED_TL_COUNTS}"
            )

        genome = self._genome()
        check_replicon(genome)
        outcomes, report = resolve_calls(
            genome, calls, label=f"{self.name} Table S1 and S4 gene names"
        )
        drop_log = build_drop_log(self.name, outcomes)
        kept = [o for o in outcomes if o.drop_rule is None]
        log.info(
            "Goodall 2018: %d table rows -> %d records; dropped %s; name statuses %s",
            drop_log.source_records,
            drop_log.kept_records,
            {r.rule: r.n_records for r in drop_log.rules},
            {s.value: n for s, n in report.status_histogram.items()},
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_ledgers(drop_log, report, checks, outcomes)

        environments = {spec.key: environment(spec.key) for spec in CONDITIONS}
        genome_reference = assembly_reference(self.REFERENCE_STRAIN)
        references = {
            key: build_reference(self.name, genome_reference, env)
            for key, env in environments.items()
        }
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, outcome in enumerate(tqdm(kept, desc="goodall2018")):
                key = outcome.call.condition
                experiment = build_experiment(self.name, outcome, environments[key])
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, references[key], PUBLICATION, itxn),
                )
        env_out.close()
        interned_env.close()
        log.info("Wrote %d Goodall 2018 essentiality experiments to LMDB", len(kept))

    def _write_ledgers(
        self,
        drop_log: DropLog,
        report: LocusTagReconciliation,
        checks: Sequence[CallRuleCheck],
        outcomes: Sequence[RowOutcome],
    ) -> None:
        """The drop log, identifier ledger, call-rule checks, per-row table (with each
        kept row's LMDB index) and the typed absences of the transposon leaf's
        coordinate fields.
        """
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drop_log.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            IdentifierLedger(
                reconciliation=report, min_resolved_fraction=MIN_RESOLVED_FRACTION
            ).model_dump_json(indent=2)
        )
        (out / "call_rule.json").write_text(
            json.dumps([check.model_dump() for check in checks], indent=2)
        )
        (out / "perturbation_field_gaps.json").write_text(
            json.dumps(
                [gap.model_dump(mode="json") for gap in PERTURBATION_FIELD_GAPS],
                indent=2,
            )
        )
        rows: list[dict[str, Any]] = []
        n_kept = 0
        for o in outcomes:
            record: int | None = None
            if o.drop_rule is None:
                record = n_kept
                n_kept += 1
            rows.append(
                {
                    "condition": o.call.condition,
                    "row": o.call.row,
                    "gene": o.call.gene,
                    "insertion_index": o.call.insertion_index,
                    "log_likelihood_ratio": o.call.log_likelihood_ratio,
                    "call": o.call.call,
                    "locus_tag": o.locus_tag,
                    "symbol": o.symbol,
                    "drop_rule": o.drop_rule,
                    "record": record,
                }
            )
        pd.DataFrame(rows).astype({"record": "Int64"}).to_csv(
            out / "calls.csv", index=False
        )

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
DATASET_ROOT_REL = "data/torchcell/gene_essentiality_goodall2018"
VERIFIER_PROVENANCE = Provenance(
    source_uri=f"https://doi.org/{PAPER_DOI} (Tables S1 and S4)",
    citation_key=CITATION_KEY,
    method="TraDIS insertion-index essentiality call per BW25113 gene and condition; "
    "reference = the unperturbed parent, viable",
    page="mBio 9:e02096-17, Tables S1 and S4",
)

Record = Mapping[str, Any]


def _validate_record(record: Record) -> None:
    """L0: the experiment and its reference validate as the assembly-pinned pair."""
    BacterialGeneEssentialityExperiment.model_validate(record["experiment"])
    BacterialGeneEssentialityExperimentReference.model_validate(record["reference"])


def record_conditions(records: Sequence[Record]) -> list[ConditionKey | None]:
    """Each record's condition: the one whose environment it stores, None if none."""
    dumps = [(spec.key, environment(spec.key).model_dump()) for spec in CONDITIONS]
    out: list[ConditionKey | None] = []
    for r in records:
        stored = r["experiment"]["environment"]
        out.append(next((key for key, dump in dumps if stored == dump), None))
    return out


def _l1_one_record_per_gene_and_condition(
    records: Sequence[Record], conditions: Sequence[ConditionKey | None]
) -> LevelResult:
    keys = Counter(
        (
            condition,
            r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"],
        )
        for r, condition in zip(records, conditions, strict=True)
    )
    unplaced = sum(n for (cond, _), n in keys.items() if cond is None)
    repeated = sorted(f"{c} {g} x{n}" for (c, g), n in keys.items() if n > 1)
    single = all(
        len(r["experiment"]["genotype"]["perturbations"]) == 1 for r in records
    )
    passed = not unplaced and not repeated and single
    return LevelResult(
        level=Level.L1,
        name="one_record_per_gene_and_condition",
        passed=passed,
        message=(
            f"{len(keys)} (condition, locus) pairs, one single-insertion record each"
            if passed
            else f"{unplaced} records match no condition environment; "
            f"{len(repeated)} pairs repeat; single-perturbation {single}"
        ),
        details={"unplaced": unplaced, "repeated": repeated[:20]},
    )


def _l2_calls_match_the_tables(
    records: Sequence[Record],
    conditions: Sequence[ConditionKey | None],
    raw_calls: Mapping[ConditionKey, Sequence[GeneCall]],
    resolve: Callable[[str], GeneNameResolution],
) -> LevelResult:
    """L2, independent of the build path: each record's call equals its table row's,
    found by the stored symbol, and the row's name resolves to the stored locus.
    """
    by_name: dict[tuple[str, str], list[GeneCall]] = {}
    for key, table in raw_calls.items():
        for call in table:
            by_name.setdefault((key, call.gene), []).append(call)
    problems: list[str] = []
    for r, condition in zip(records, conditions, strict=True):
        perturbation = r["experiment"]["genotype"]["perturbations"][0]
        symbol = perturbation["perturbed_gene_name"]
        rows = by_name.get((str(condition), symbol), [])
        if len(rows) != 1:
            problems.append(f"{condition} {symbol}: {len(rows)} table rows")
            continue
        (row,) = rows
        stored = r["experiment"]["phenotype"]["is_essential"]
        if row.call == "unclear" or stored != (row.call == "essential"):
            problems.append(f"{condition} {symbol}: stored {stored}, table {row.call}")
        resolution = resolve(row.gene)
        if resolution.systematic_name != perturbation["systematic_gene_name"] or (
            resolution.status
            not in (GeneNameStatus.RENAMED, GeneNameStatus.NON_GENE_FEATURE)
        ):
            problems.append(
                f"{condition} {symbol}: resolves {resolution.status.value} "
                f"{resolution.systematic_name}, stored "
                f"{perturbation['systematic_gene_name']}"
            )
    return LevelResult(
        level=Level.L2,
        name="calls_match_released_tables",
        passed=not problems,
        message=(
            f"{len(records)} records equal their Table S1/S4 row's call and locus"
            if not problems
            else f"{len(problems)} records disagree with the released tables"
        ),
        details={"n_problems": len(problems), "problems": problems[:20]},
    )


def _l3_call_rule(raw_calls: Mapping[ConditionKey, Sequence[GeneCall]]) -> LevelResult:
    checks = [
        call_rule_check(table, key, CALL_THRESHOLD) for key, table in raw_calls.items()
    ]
    passed = all(not c.disagreements for c in checks)
    return LevelResult(
        level=Level.L3,
        name="call_threshold_convention",
        passed=passed,
        message=", ".join(
            f"{c.condition}: {c.n_agree}/{c.n_rows} calls follow +/-{c.threshold}"
            for c in checks
        ),
        details={c.condition: c.model_dump() for c in checks},
    )


def _l3_tl_counts(raw_calls: Mapping[ConditionKey, Sequence[GeneCall]]) -> LevelResult:
    counts = call_counts(raw_calls["TL"])
    return LevelResult(
        level=Level.L3,
        name="table_s1_counts_equal_the_paper",
        passed=counts == EXPECTED_TL_COUNTS,
        message=f"Table S1 {counts}; the paper states {EXPECTED_TL_COUNTS}",
        details={"table": counts, "paper": EXPECTED_TL_COUNTS},
    )


def _l4_containment(records: Sequence[Record], universe: set[str]) -> LevelResult:
    disrupted = {
        p["systematic_gene_name"]
        for r in records
        for p in r["experiment"]["genotype"]["perturbations"]
    }
    missing = sorted(disrupted - universe)
    return LevelResult(
        level=Level.L4,
        name="gene_containment_bw25113_locus_tags",
        passed=bool(disrupted) and not missing,
        message=f"{len(disrupted) - len(missing)} of {len(disrupted)} disrupted loci "
        "are BW25113 GenBank gene rows",
        details={
            "n_disrupted": len(disrupted),
            "n_universe": len(universe),
            "missing_examples": missing[:20],
        },
    )


def verify_records(
    records: Sequence[Record],
    *,
    raw_calls: Mapping[ConditionKey, Sequence[GeneCall]],
    resolve: Callable[[str], GeneNameResolution],
    universe: set[str],
    expected_count: int,
    dataset_name: str = "gene_essentiality_goodall2018",
) -> VerificationReport:
    """The L0-L4 gate over built records, given the re-read tables, the BW25113
    resolver and the BW25113 gene universe.

    The shared rules (gap census, spelling, uncertainty, compound identity, media
    membership) run without a resolver: their canonical-name rule requires a CURRENT
    status, a pseudogene locus resolves NON_GENE_FEATURE, and the L2 check here accepts
    both a gene and a pseudogene by name.
    """
    from torchcell.verification.common import shared_rule_results

    report = VerificationReport(
        dataset_name=dataset_name, provenance=VERIFIER_PROVENANCE
    )
    conditions = record_conditions(records)
    report.add(l0_structural(records, _validate_record))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_one_record_per_gene_and_condition(records, conditions))
    report.add(_l2_calls_match_the_tables(records, conditions, raw_calls, resolve))
    report.add(_l3_call_rule(raw_calls))
    report.add(_l3_tl_counts(raw_calls))
    for result in shared_rule_results(records):
        report.add(result)
    report.add(_l4_containment(records, universe))
    return report


def run_verification(data_root: str | None = None) -> VerificationReport:
    """Verify the built dev-tree LMDB (L0-L4) plus the provenance audit of every
    ``SOURCED_VALUES`` entry, and write ``preprocess/verification_report.json``.
    """
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _write_report,
        load_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, DATASET_ROOT_REL)
    records = load_records(abs_root)
    drops = DropLog.model_validate_json(
        Path(abs_root, "preprocess", "dropped_records.json").read_text()
    )
    raw_calls: dict[ConditionKey, Sequence[GeneCall]] = {
        spec.key: read_calls(osp.join(abs_root, "raw", spec.table), spec)
        for spec in CONDITIONS
    }
    genome = bacterial_genome("ecoli", "BW25113", base)
    references = {
        json.dumps(r["reference"]["genome_reference"], sort_keys=True) for r in records
    }
    universe: set[str] = set()
    for ref in references:
        universe |= _gene_set_for_reference(json.loads(ref), base)
    report = verify_records(
        records,
        raw_calls=raw_calls,
        resolve=genome.resolve_gene_name,
        universe=universe,
        expected_count=drops.kept_records,
    )
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, Path(base) / "torchcell-library"))
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.goodall2018"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser("deposit", help="deposit the raw mirror")
    deposit.add_argument(
        "--retrieve-into",
        default=None,
        help="re-run every recorded PMC retrieval into this directory and deposit "
        "those bytes; without it the literature mirror's captured SI files are used",
    )
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        sources: dict[str, str | Path]
        if args.retrieve_into is not None:
            sources = dict(retrieve_raw_files(args.retrieve_into))
        else:
            sources = {
                name: library_dir(data_root) / "si" / name for name in DATA_SHA256
            }
        print(deposit_raw_mirror(sources=sources, data_root=data_root))
        return 0
    if args.command == "build":
        dataset = GeneEssentialityGoodall2018Dataset(
            root=osp.join(data_root, DATASET_ROOT_REL)
        )
        print(f"len = {len(dataset)}")
        return 0
    report = run_verification(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
