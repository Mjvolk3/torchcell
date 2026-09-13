# torchcell/datasets/scerevisiae/hillenmeyer2008
# [[torchcell.datasets.scerevisiae.hillenmeyer2008]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/hillenmeyer2008
# Test file: tests/torchcell/datasets/scerevisiae/test_hillenmeyer2008.py
"""Hillenmeyer 2008 "chemical genomic portrait of yeast" (FitDb): env x geno -> fitness.

Hillenmeyer et al. 2008 (Science 320:362; doi:10.1126/science.1150021; citation key
``hillenmeyerChemicalGenomicPortrait2008``) is the genome-scale HIP/HOP fitness
compendium: the heterozygous (HIP) and homozygous (HOP) barcoded deletion collections
competitively grown under ~400 chemical and environmental conditions, read out as a
per-strain fitness defect. It is the earlier, broader companion to the Hoepfner 2014
HIP-HOP atlas.

TWO DATASETS (the readouts are incomparable -- a log-ratio must never be mixed with a
z-score in one ``measurement_type`` -- so HET and HOM are SEPARATE dataset classes,
``HetHillenmeyer2008Dataset`` and ``HomHillenmeyer2008Dataset``, cf. the Smf/Dmf split):

- ``het.ratio_result_nm.pub`` -- HET (HIP) fitness-defect LOG-RATIO, 5984 strain rows x
  726 arrays, ``measurement_type=log2_ratio``. The SOM defines it as
  ``log-ratio_FD = log2(mu_i^c / x_i^t)``, mu_i^c the mean intensity of tag i across the
  matched control arrays and x_i^t its intensity in the treatment, averaged over the up
  and down tags. Positive = fitness defect (the strain is depleted under treatment).
- ``hom.z_result_nm.pub`` -- HOM (HOP) fitness-defect Z-SCORE, 4769 strain rows x 418
  arrays, ``measurement_type=z_score``, ``z_FD = (mu_i^c - x_i^t) / sigma_i^c`` with
  sigma_i^c the SD of tag i across the SAME matched control arrays. No HOM log-ratio
  matrix was archived, which is why the two classes carry different statistics.

REFERENCE. Both scores are defined against a MATCHED CONTROL SET, not against a single
global control, so the reference is emitted per control set rather than once per matrix:
"Each treatment array was compared to an associated control set of no-drug arrays that
matched (a) the deletion pool used in the experiment, (b) the number of growth
generations of the experiment, and (c) the scanner for the experiment (one of two
scanners)." ``het.txt`` / ``hom.txt`` release the per-array control-set assignment and
``het_controls.txt`` / ``hom_controls.txt`` list that set's control arrays, which is the
sample size of the z-score's denominator. The control-set id is stored on
``EnvironmentResponsePhenotype.screen_id``, so two arrays of one condition normalized
against different control sets stay distinct measurements instead of merging.

GENOTYPE (as in Hoepfner): the collections are diploid. HET = one of the two autosomal
copies deleted (KanMX) -> reduced dosage ->
``EngineeredCopyNumberPerturbation(copy_number=1, reference_copy_number=2, marker='KanMX')``
in a ``ReferenceGenome(ploidy='diploid')``, essential genes included. HOM = both copies
deleted -> ``KanMxDeletionPerturbation`` in a diploid.

ENVIRONMENT -- parsed from each column header
``filename:cond1:conc1:unit1:cond2:conc2:unit2:generations:pool:scanner``. cond1 is
classified per SOM Table S1 and cond2, when present, ALWAYS adds a second small molecule
(the double-drug arms), independently of how cond1 classified:

- ``<N> degrees C`` -> ``Environment.temperature`` (M2, no perturbation).
- ``pH<x>`` -> ``EnvironmentPhysicalPerturbation(factor=ph)``.
- a named-nutrient drop-out -> the DERIVED MEDIUM ``HILLENMEYER_DROPOUT_MEDIA[label]``
  (SC minus the nutrient), not a perturbation on YPD: a drop-out is only defined against
  a synthetic complete medium, and the derived medium is what joins ``SC`` / ``SC_URA``.
  ``vitamin drop-out control media`` is that series' CONTROL medium, so it maps to ``SC``.
- ``no drug irradiated`` -> ``EnvironmentPhysicalPerturbation(factor=radiation)``, dose
  not released. ``angelicin irradiated`` / ``psoralen irradiated`` are photo-activated
  crosslinkers: Table S1 files them under ALKYLATING small molecules, so they emit the
  compound at its dose AND the radiation factor.
- ``minimal media`` -> ``SD``; ``synthetic complete`` -> ``SC``; ``YP glycerol`` ->
  ``YP_GLYCEROL_LIQUID``. Table S1 calls all three a "media change".
- everything else -> ``SmallMoleculePerturbation(resolved_compound(label), dose)``.

Base medium is ``YPD_LIQUID`` (the shared library object), never a name-only
``Media(name="YPD")``. TEMPERATURE IS NOT DEFAULTED: the SOM never states the standard
growth temperature and defers the whole growth protocol to Pierce 2006 (Nat Methods
3:601), which is not mirrored, so every non-temperature condition carries
``temperature=None`` with a ``ProvenanceGap(deferred_pending_source_review)`` naming that
paper as the resolve target, rather than a silent 30 C.

DOSES are canonicalized within the molar family before they become part of the
environment identity: the release spells one dose two ways in five places (``sorbitol
1.5 m`` and ``sorbitol 1.5e+06 um`` at 15 generations; the same for streptozotocin,
mitomycin c and carboplatin), which without normalization becomes two environments for
one condition. Conversion is exact (``decimal.Decimal``) and picks the largest unit that
leaves the value >= 1.

DURATION. ``generations`` is the readout timepoint in doublings of competitive growth.
The sign is a PROTOCOL FLAG, not a negative duration: "A negative sign preceding the
number indicates that the pool was taken directly from the freezer, thawed, diluted and
grown in the condition. Absence of a negative sign indicates that the pool was thawed,
inoculated into YPD and grown overnight until log phase (OD600=2.0) (~10 generations of
recovery) before drug addition." ``duration_generations`` therefore stores the MAGNITUDE,
and the signed form survives verbatim inside the control-set id on ``screen_id`` (the
control set matches the treatment's signed generation count), so ``-5gen`` and ``5gen``
arrays are never merged.

DROPPED RECORDS (logged to ``preprocess/dropped_records.json``, three rules):
1. a condition naming a compound with no structure identifier -- the ten
   ``chemical diversity labs`` vendor codes, the Biomol catalog codes, activity-class
   labels (``phosphatase inhibitor``, ``tyrphostin``), genus names that do not pick a
   congener (``amphotericin``, ``latrunculin``, ``bisphenol``), ``FeCl4`` (which does not
   exist as written), the ``DMSO <n>%`` labels that fuse a vehicle dose into the name,
   and the ``<compound>, <n>ul total up/dn`` labels that fuse a hybridization volume into
   it. The audit trail is each row's ``unresolved_reason`` in the compound table.
2. ``37c, 45c`` -- a heat-shock CYCLE. ``PhysicalFactor`` has no ``heat_shock`` member and
   adding one would change a served class, and storing the peak 45 C would assert
   continuous growth at a temperature yeast does not grow at, so the single array is
   dropped rather than retyped.
3. the two ``minimal media:400:um`` arrays, whose 400 uM agent is named nowhere in the
   SOM or the release.

REPLICATE AGGREGATION. The atlas is per array and per barcode probe; the ontology record
is per (strain, environment, control set). Within one group each array contributes ONE
value -- the mean over that ORF's matrix rows -- and the record's value is the mean over
arrays with the sample SD across arrays as the uncertainty (``n_samples`` = arrays,
``sample_unit=biological_replicate``). Multiple rows of one ORF are NOT replicates: "The
number of strains exceeds the number of genes because some gene deletions were
constructed more than once, in different batches", so 59 het ORFs (128 rows) and 17 hom
ORFs (44 rows) average distinct constructions; the per-ORF row/batch map is written to
``preprocess/strain_batches.json``.

SUSPICIOUS BATCHES (the HIP background-mutation risk, stated by this paper itself). The
row id is ``ORF:batch``; the SOM names ten construction batches whose strains cluster on
a shared secondary genotype ("we believe this is due to unrelated artifacts, e.g. a
secondary site mutation in the parent strain used to make each batch") and reports "647
strains belonged to these suspicious batches, and we exclude these strains from most
analyses", adding that "the homozygous strains did not show this pattern". The batch
cannot be carried on the record without adding a field to a served perturbation class, so
the build writes ``preprocess/suspicious_batch_strains.json`` -- the ORFs, their batches,
the SOM quotes and the measured counts -- exactly as the Hoepfner background-mutation
purge list does.

GENE NAMES go through the shared ``SCerevisiaeGenome.resolve_gene_name`` first: a renamed
ORF is stored under its current systematic name, and a retired or non-gene id is dropped
with its reason to ``preprocess/dropped_genes.json`` instead of being discarded on a bare
count.

DATA SOURCE / RAW MIRROR. Live FitDb portals are DNS-dead and the Science SI 403s; the
Stanford static supplement survives only in the Internet Archive. The seven files this
loader and its provenance consume are deposited at
``$DATA_ROOT/torchcell-raw/hillenmeyerChemicalGenomicPortrait2008/data/`` with a
``manifest.json`` recording, per file, the exact ``web.archive.org/web/<TS>id_/`` URL,
the ``direct_url`` retriever, the sha256 and the retrieval date; ``download()`` links them
into ``raw/`` and verifies every sha256 against that manifest. See
``[[fitdb-hillenmeyer2008-wayback-data-source]]``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import (
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import (
    HILLENMEYER_DROPOUT_MEDIA,
    SC,
    SD,
    YP_GLYCEROL_LIQUID,
    YPD_LIQUID,
)
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    EngineeredCopyNumberPerturbation,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    SourceCheck,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "hillenmeyerChemicalGenomicPortrait2008"
DOI = "10.1126/science.1150021"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
#: sha256 of the mirrored OCR supplement every quote below is a literal substring of.
SOM_SHA256 = "cf4759f00083de78dd953b12dd66d4360a2f645321305f768403f26b451c1df0"
#: The SOM's declared deferral target for the growth protocol; NOT mirrored.
PIERCE2006 = "pierceGenomewideAnalysisBarcoded2006"

# Wayback-archived Stanford supplement. The timestamps are per file (the crawler hit
# them minutes apart) and come from the CDX index; re-running each URL today reproduces
# the deposited bytes exactly, which is what SOURCE_CHECKED_AT records.
WAYBACK_BASE = "http://chemogenomics.stanford.edu/supplements/global/download/data"
WAYBACK_TIMESTAMPS: dict[str, str] = {
    "het.ratio_result_nm.pub": "20151207003548",
    "hom.z_result_nm.pub": "20151207003024",
    "het.txt": "20151207063659",
    "hom.txt": "20151207011015",
    "het_controls.txt": "20151207003758",
    "hom_controls.txt": "20151207004537",
    "README_CEL_files.txt": "20151207004902",
}
#: Date the bytes were first retrieved into the dev tree (memory note
#: ``fitdb-hillenmeyer2008-wayback-data-source``; the staged files carry that mtime).
RAW_RETRIEVED_AT = "2026-07-11"

_SOM_PROVENANCE = Provenance(
    source_uri="paper.md", citation_key=CITATION_KEY, sha256=SOM_SHA256
)
_PIERCE_PROVENANCE = Provenance(
    source_uri="paper.pdf",
    citation_key=PIERCE2006,
    sha256="not-mirrored",
    method="S. E. Pierce et al., Nat Methods 3, 601 (Aug, 2006) -- ref (2) of the "
    "Hillenmeyer SOM, the declared source of the growth protocol",
)

# Verbatim SOM sentences (literal substrings of the pinned paper.md).
QUOTE_CONTROL_SET = (
    "Each treatment array was compared to an associated control set of no-drug arrays "
    "that matched (a) the deletion pool used in the experiment, (b) the number of growth "
    "generations of the experiment"
)
QUOTE_CONTROL_COUNT = (
    "combinations yielded 34 control sets, each composed of between 3 and 48 control "
    "arrays"
)
QUOTE_GENERATIONS_SIGN = (
    "A negative sign preceding the number indicates that the pool was taken directly "
    "from the freezer, thawed, diluted and grown in the condition."
)
QUOTE_PRESCREEN = (
    "those compounds/treatments that produced a measurable $1 0 { - } 1 5 \\%$ inhibition "
    "of wildtype growth (IC-15) were chosen for further full-genome screening"
)
QUOTE_PROTOCOL_DEFERRAL = (
    "The protocol for pooled, competitive growth of the deletion strains, genomic DNA "
    "purification and PCR, and tag hybridization follows Ref. (2)."
)
QUOTE_BATCH_IDS = (
    "All strains associated with batch $\\mathrm { c h r l } _ { - 1 }$ were made by a "
    "single laboratory or group. The gene names in our downloadable data include the "
    "strain-batch number."
)
QUOTE_RECONSTRUCTED = (
    "The number of strains exceeds the number of genes because some gene deletions were "
    "constructed more than once, in different batches."
)
QUOTE_SUSPICIOUS = (
    "we noted a few abnormally strong clusters whose genes were functionally unrelated "
    "but the strains in each cluster originated from a common deletion collection "
    "generation batch. We believe this is due to unrelated artifacts, e.g. a secondary "
    "site mutation in the parent strain used to make each batch."
)
QUOTE_SUSPICIOUS_COUNT = (
    "647 strains belonged to these suspicious batches, and we exclude these strains from "
    "most analyses."
)
QUOTE_SUSPICIOUS_HOM = "The homozygous strains did not show this pattern."
QUOTE_ARRAY_TYPE = (
    "The array type for all experiments was TAG3 (Affymetrix GenFlex Tag Array, Part No. "
    "510389)."
)

#: The ten construction batches the SOM names as carrying a shared secondary genotype.
SUSPICIOUS_BATCHES: tuple[str, ...] = (
    "chr00_1",
    "chr3_1",
    "chr00_5",
    "chr13_4",
    "chr12_4",
    "chr13_1b",
    "chr4_4",
    "chr13_5",
    "chr13_2",
    "chr7_5",
)

# --------------------------------------------------------------------------- #
# Matrices
# --------------------------------------------------------------------------- #
UNITS_HET = (
    "HIP fitness-defect log-ratio, log2(mean control intensity / treatment intensity) "
    "averaged over the up and down tags (Hillenmeyer 2008 scoring method 1); positive = "
    "fitness defect. Control = the matched no-drug control set named in screen_id "
    "(same deletion pool, generation count and scanner). Mean over the replicate arrays "
    "of that control set; an ORF with several construction batches is averaged within "
    "each array first"
)
UNITS_HOM = (
    "HOP fitness-defect z-score, (mean control intensity - treatment intensity) / SD of "
    "the control intensities (Hillenmeyer 2008 scoring method 2); positive = fitness "
    "defect. Control = the matched no-drug control set named in screen_id (same deletion "
    "pool, generation count and scanner). Mean over the replicate arrays of that control "
    "set; an ORF with several construction batches is averaged within each array first"
)


class _MatrixSpec(BaseModel):
    """One released score matrix and everything that is constant across it."""

    key: str
    filename: str
    keyfile: str
    controls_file: str
    measurement_type: MeasurementType
    collection: str
    strain: str
    units: str


MATRICES: dict[str, _MatrixSpec] = {
    "het": _MatrixSpec(
        key="het",
        filename="het.ratio_result_nm.pub",
        keyfile="het.txt",
        controls_file="het_controls.txt",
        measurement_type=MeasurementType.log2_ratio,
        collection="heterozygous",
        strain="heterozygous diploid deletion collection (Giaever 2002)",
        units=UNITS_HET,
    ),
    "hom": _MatrixSpec(
        key="hom",
        filename="hom.z_result_nm.pub",
        keyfile="hom.txt",
        controls_file="hom_controls.txt",
        measurement_type=MeasurementType.z_score,
        collection="homozygous",
        strain="homozygous diploid deletion collection (Giaever 2002)",
        units=UNITS_HOM,
    ),
}

# S288C R64 gene universe (systematic ORF + RNA-coding names). A resolved name must also
# be in this universe, which is the same set the L4 containment rule is scored against.
_SGD_GENE_FASTAS = (
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "orf_coding_all_R64-4-1_20230830.fasta",
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "rna_coding_R64-4-1_20230830.fasta",
)

_MOLAR_FAMILY: tuple[tuple[str, ConcentrationUnit, Decimal], ...] = (
    ("m", ConcentrationUnit.molar, Decimal(1)),
    ("mm", ConcentrationUnit.millimolar, Decimal("1e-3")),
    ("um", ConcentrationUnit.micromolar, Decimal("1e-6")),
    ("nm", ConcentrationUnit.nanomolar, Decimal("1e-9")),
)
_MOLAR_BY_TOKEN = {token: (unit, factor) for token, unit, factor in _MOLAR_FAMILY}
_OTHER_UNITS = {
    "ug/ml": ConcentrationUnit.ug_per_ml,
    "%": ConcentrationUnit.percent_v_v,
}

_TEMP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*degrees?\s*c", re.I)
_TEMP_CYCLE_RE = re.compile(r"^\s*(\d+)\s*c\s*,\s*(\d+)\s*c", re.I)
_PH_RE = re.compile(r"^\s*ph\s*([\d.]+)\s*$", re.I)
_IRRADIATED_SUFFIX = " irradiated"
_RADIATION_ONLY = "no drug irradiated"
#: condition label -> the shared medium it denotes (Table S1 "media change").
_MEDIA_SWAPS: dict[str, Media] = {
    "minimal media": SD,
    "synthetic complete": SC,
    "yp glycerol": YP_GLYCEROL_LIQUID,
}
# Missing-value tokens in the score matrices (upper-cased) -- skipped when aggregating.
_MISSING = {"", "NA", "NAN", "NULL"}

DROP_UNIDENTIFIABLE = "unidentifiable_agent"
DROP_HEAT_SHOCK_CYCLE = "heat_shock_cycle_not_representable"
DROP_UNNAMED_MEDIA_AGENT = "unnamed_agent_dosed_into_a_media_swap"


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_root() -> str:
    """``DATA_ROOT`` from the repo-root ``.env``."""
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/hillenmeyerChemicalGenomicPortrait2008``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def raw_relpaths() -> dict[str, str]:
    """Mirror-relative path of every deposited file, keyed by its released name."""
    return {name: f"data/{name}" for name in WAYBACK_TIMESTAMPS}


def wayback_url(name: str) -> str:
    """The exact archived URL whose bytes the mirror holds."""
    return f"http://web.archive.org/web/{WAYBACK_TIMESTAMPS[name]}id_/{WAYBACK_BASE}/{name}"


def deposit_raw_mirror(
    source_dir: str | Path,
    *,
    data_root: str | None = None,
    verify_source: bool = False,
    retrieved_at: str = RAW_RETRIEVED_AT,
) -> Path:
    """Deposit the Wayback-recovered supplement into the raw mirror with a manifest.

    ``source_dir`` holds the already-retrieved bytes (the dev staging directory). Copies
    are idempotent by sha256: an existing mirror file with a different hash raises rather
    than being overwritten. With ``verify_source`` the recorded retrieval is RE-RUN and
    each file's ``last_check`` records whether the archive still yields the same bytes;
    a mismatch raises, because a silently different archive copy is exactly what the
    sha256 anchor exists to catch.
    """
    root = raw_mirror_dir(data_root)
    rel = raw_relpaths()
    files: list[ArtifactRecord] = []
    for name, relpath in rel.items():
        src = Path(source_dir) / name
        if not src.exists():
            raise RuntimeError(f"required raw file missing from {source_dir}: {name}")
        sha = _sha256(src)
        check: SourceCheck | None = None
        if verify_source:
            from torchcell.literature.retrieve import direct_url

            produced = hashlib.sha256(direct_url(wayback_url(name))).hexdigest()
            if produced != sha:
                raise RuntimeError(
                    f"{name}: the archived URL now yields sha256 {produced}, the "
                    f"deposited bytes are {sha}"
                )
            check = SourceCheck(
                checked_at=date.today().isoformat(),
                produced_sha256=produced,
                matches=True,
            )
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != sha:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=sha,
                source=wayback_url(name),
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=wayback_url(name),
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": wayback_url(name)},
                    sha256=sha,
                    retrieved_at=retrieved_at,
                    last_check=check,
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title="The chemical genomic portrait of yeast: uncovering a phenotype for all genes",
        files=files,
        si_data_sources=[
            "http://web.archive.org/cdx/search/cdx?url=chemogenomics.stanford.edu/"
            "supplements/global*&output=text&fl=original,timestamp,statuscode,length",
            f"{WAYBACK_BASE}/ (Internet Archive; the live host is DNS-dead)",
        ],
        si_expected=[
            "het.ratio_result_nm.pub (HIP log-ratio matrix)",
            "hom.z_result_nm.pub (HOP z-score matrix)",
            "het.txt / hom.txt (array -> condition -> control set)",
            "het_controls.txt / hom_controls.txt (control-set membership)",
            "README_CEL_files.txt (key-file semantics)",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    return Manifest.model_validate_json(
        (raw_mirror_dir(data_root) / "manifest.json").read_text()
    )


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the Hillenmeyer raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #
def canonical_concentration(conc: str, unit: str) -> Concentration:
    """Parse one released dose, canonicalizing the molar family exactly.

    The release spells one molar dose two ways in five places (``1.5 m`` vs
    ``1.5e+06 um``), which would otherwise become two environments for one condition.
    Conversion runs in ``Decimal`` (so ``1.5e+06 uM`` and ``1.5 M`` land on the identical
    float) and picks the largest unit leaving the value >= 1. A dose with no unit is a
    ``DoseBasis.fixed`` dose, which is what the release means by an empty unit field.
    """
    value, token = conc.strip(), unit.strip().lower()
    if not value or not token:
        return Concentration(basis=DoseBasis.fixed)
    if token in _OTHER_UNITS:
        return Concentration(value=float(value), unit=_OTHER_UNITS[token])
    if token not in _MOLAR_BY_TOKEN:
        raise ValueError(f"unrecognized concentration unit {unit!r} (value {conc!r})")
    molar = Decimal(value) * _MOLAR_BY_TOKEN[token][1]
    for _, enum_unit, factor in _MOLAR_FAMILY:
        scaled = molar / factor
        if scaled >= 1:
            return Concentration(value=float(scaled), unit=enum_unit)
    return Concentration(
        value=float(molar / Decimal("1e-9")), unit=ConcentrationUnit.nanomolar
    )


class ConditionParse(BaseModel):
    """One parsed column header: its typed environment, or the rule that drops it."""

    model_config = {"arbitrary_types_allowed": True}

    media: Media
    temperature_c: float | None
    perturbations: list[SmallMoleculePerturbation | EnvironmentPhysicalPerturbation]
    drop_reason: str | None = None
    drop_detail: str | None = None


def _small_molecule(label: str, conc: str, unit: str) -> SmallMoleculePerturbation:
    """One dosed compound, resolved through the shared table by its SOURCE label."""
    return SmallMoleculePerturbation(
        compound=resolved_compound(label),
        concentration=canonical_concentration(conc, unit),
    )


def parse_condition(
    cond1: str, conc1: str, unit1: str, cond2: str, conc2: str, unit2: str
) -> ConditionParse:
    """Classify one column header into a typed environment (SOM Table S1).

    cond1 selects the branch; cond2, when present, ALWAYS appends a second small molecule
    (the double-drug arms). Splitting the two is what keeps ``pH7.5 + FK506`` from being
    stored as a plain pH stress and ``<compound> irradiated`` from being stored as bare
    radiation.
    """
    label = cond1.strip()
    low = label.lower()
    media: Media = YPD_LIQUID
    temperature_c: float | None = None
    perturbations: list[
        SmallMoleculePerturbation | EnvironmentPhysicalPerturbation
    ] = []
    drop_reason: str | None = None
    drop_detail: str | None = None

    if _TEMP_CYCLE_RE.match(label):
        drop_reason, drop_detail = DROP_HEAT_SHOCK_CYCLE, label
    elif (temp_match := _TEMP_RE.search(label)) is not None:
        temperature_c = float(temp_match.group(1))
    elif (ph_match := _PH_RE.match(label)) is not None:
        perturbations.append(
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.ph,
                magnitude=Concentration(
                    value=float(ph_match.group(1)), unit=ConcentrationUnit.ph
                ),
            )
        )
    elif label in HILLENMEYER_DROPOUT_MEDIA:
        media = HILLENMEYER_DROPOUT_MEDIA[label]
    elif low == _RADIATION_ONLY:
        perturbations.append(
            EnvironmentPhysicalPerturbation(factor=PhysicalFactor.radiation)
        )
    elif low.endswith(_IRRADIATED_SUFFIX):
        perturbations.append(
            _small_molecule(label[: -len(_IRRADIATED_SUFFIX)].strip(), conc1, unit1)
        )
        perturbations.append(
            EnvironmentPhysicalPerturbation(factor=PhysicalFactor.radiation)
        )
    elif low in _MEDIA_SWAPS:
        media = _MEDIA_SWAPS[low]
        if conc1.strip():
            drop_reason = DROP_UNNAMED_MEDIA_AGENT
            drop_detail = f"{label}: {conc1.strip()} {unit1.strip()}".strip()
    else:
        perturbations.append(_small_molecule(label, conc1, unit1))

    if cond2.strip():
        perturbations.append(_small_molecule(cond2.strip(), conc2, unit2))

    if drop_reason is None:
        unidentified = [
            perturbation.compound.name
            for perturbation in perturbations
            if isinstance(perturbation, SmallMoleculePerturbation)
            and not resolve_compound_identity(
                name=perturbation.compound.name
            ).identified
        ]
        if unidentified:
            drop_reason = DROP_UNIDENTIFIABLE
            drop_detail = ", ".join(unidentified)

    return ConditionParse(
        media=media,
        temperature_c=temperature_c,
        perturbations=perturbations,
        drop_reason=drop_reason,
        drop_detail=drop_detail,
    )


class ColumnSpec(BaseModel):
    """One matrix data column: its index, its environment and its grouping key."""

    model_config = {"arbitrary_types_allowed": True}

    index: int
    filename: str
    header: str
    control_set: str
    environment: Environment
    group_key: str
    drop_reason: str | None = None
    drop_detail: str | None = None


def _temperature_gap() -> ProvenanceGap:
    """The typed absence behind an unstated standard growth temperature."""
    return ProvenanceGap(
        field="temperature",
        reason=ProvenanceGapReason.deferred_pending_source_review,
        looked_in=_SOM_PROVENANCE,
        resolve_with=_PIERCE_PROVENANCE,
        note="the SOM states no growth temperature for the non-temperature conditions "
        "and defers the whole pooled-growth protocol to Pierce 2006, which is not "
        "mirrored; 30 C is the community default but this paper never states it",
    )


def build_environment(parse: ConditionParse, generations: str) -> Environment:
    """The typed environment of one column, with the generation MAGNITUDE and no default
    temperature.
    """
    duration = (
        abs(float(generations.replace("gen", ""))) if generations.strip() else None
    )
    gaps = [] if parse.temperature_c is not None else [_temperature_gap()]
    return Environment(
        media=parse.media,
        temperature=(
            Temperature(value=parse.temperature_c)
            if parse.temperature_c is not None
            else None
        ),
        perturbations=list(parse.perturbations),
        aerobicity="aerobic",
        duration_generations=duration,
        provenance_gaps=gaps,
    )


def read_control_set_map(path: str | Path) -> dict[str, str]:
    """``het.txt`` / ``hom.txt``: array filename -> the control set it was scored against."""
    mapping: dict[str, str] = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:3] != ["filename", "condition", "control_set"]:
            raise ValueError(f"unexpected key-file header in {path}: {header}")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            mapping[fields[0]] = fields[2]
    return mapping


def read_control_set_sizes(path: str | Path) -> dict[str, int]:
    """``*_controls.txt``: control set -> the number of control arrays it contains."""
    sizes: dict[str, int] = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:2] != ["control_set", "filename"]:
            raise ValueError(f"unexpected controls-file header in {path}: {header}")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            sizes[fields[0]] = sizes.get(fields[0], 0) + 1
    return sizes


def parse_columns(header: list[str], control_sets: dict[str, str]) -> list[ColumnSpec]:
    """Parse every data column into its environment, control set and grouping key."""
    specs: list[ColumnSpec] = []
    for index, raw in enumerate(header):
        if index == 0:
            continue  # the 'Orf' row-id column
        fields = raw.strip().split(":")
        while len(fields) < 10:
            fields.append("")
        filename, c1, co1, u1, c2, co2, u2, generations = fields[:8]
        if filename not in control_sets:
            raise ValueError(f"array {filename!r} has no control set in the key file")
        parse = parse_condition(c1, co1, u1, c2, co2, u2)
        environment = build_environment(parse, generations)
        control_set = control_sets[filename]
        # Group replicate arrays by the CANONICAL environment plus the control set the
        # score was computed against: two arrays of one condition normalized against
        # different control sets are different measurements, and the control set is also
        # what preserves the signed generation count the duration magnitude drops.
        group_key = json.dumps(
            [environment.model_dump(), control_set], sort_keys=True, default=str
        )
        specs.append(
            ColumnSpec(
                index=index,
                filename=filename,
                header=raw.strip(),
                control_set=control_set,
                environment=environment,
                group_key=group_key,
                drop_reason=parse.drop_reason,
                drop_detail=parse.drop_detail,
            )
        )
    return specs


def _load_sgd_genes(data_root: str) -> set[str]:
    """S288C R64 systematic-name universe from the ORF + RNA-coding FASTA headers."""
    genes: set[str] = set()
    for rel in _SGD_GENE_FASTAS:
        with open(osp.join(data_root, rel)) as handle:
            for line in handle:
                if line.startswith(">"):
                    genes.add(line[1:].split()[0])
    return genes


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class _Hillenmeyer2008Base(ExperimentDataset):
    """Shared FitDb HIP/HOP loader. HET (log2_ratio) and HOM (z_score) are SEPARATE
    datasets (incomparable readouts, distinct collections), one concrete subclass each.
    """

    _matrix_key: str = ""  # 'het' | 'hom' -- set by concrete subclasses

    def __init__(
        self,
        root: str,
        io_workers: int = 0,
        genome: Any | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset. ``genome`` supplies ``resolve_gene_name``; when it is
        omitted ``process`` opens the S288C genome read-only from ``DATA_ROOT``.
        """
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def spec(self) -> _MatrixSpec:
        """This subclass's released matrix and its constants."""
        return MATRICES[self._matrix_key]

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
        """The mirrored files this dataset reads: its matrix and its control-set tables."""
        spec = self.spec
        return [spec.filename, spec.keyfile, spec.controls_file]

    def download(self) -> None:
        """Link the manifest-listed mirror files into ``raw/`` and verify each sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        rel = raw_relpaths()
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in self.raw_file_names:
            src = raw_mirror_dir(data_root) / rel[name]
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            expected = manifest_sha256(manifest, rel[name])
            got = _sha256(src)
            if got != expected:
                raise RuntimeError(
                    f"{name} sha256 mismatch: got {got}, expected {expected}"
                )
            dest = osp.join(self.raw_dir, name)
            if not osp.exists(dest):
                os.symlink(src, dest)
        log.info(
            "Hillenmeyer 2008 %s raw files linked into %s (sha256 verified against the "
            "raw-mirror manifest)",
            self._matrix_key,
            self.raw_dir,
        )

    # ---- record builders ------------------------------------------------------
    def _genotype(self, orf: str) -> Genotype:
        """HET -> heterozygous engineered-CNV (copy 1 of 2); HOM -> homozygous deletion."""
        if self.spec.collection == "heterozygous":
            return Genotype(
                perturbations=[
                    EngineeredCopyNumberPerturbation(
                        systematic_gene_name=orf,
                        perturbed_gene_name=orf,
                        copy_number=1,
                        reference_copy_number=2,
                        marker="KanMX",
                    )
                ]
            )
        return Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                )
            ]
        )

    def _reference(
        self, control_set: str, n_control_arrays: int
    ) -> EnvironmentResponseExperimentReference:
        """The matched no-drug control set this control set's scores were computed against.

        ``control_set`` is the release's own id, ``<pool>::<scanner>::<generations>::tag3::
        YPD::dmso::0``: the deletion pool, the scanner, the SIGNED generation count, the
        array type, the medium and the vehicle. Its generation field is the reference's
        ``duration_generations``, so the reference differs from the treatment in the
        perturbation and in nothing else.
        """
        spec = self.spec
        generations = abs(float(control_set.split("::")[2]))
        environment = Environment(
            media=YPD_LIQUID,
            temperature=None,
            perturbations=[],
            aerobicity="aerobic",
            duration_generations=generations,
            provenance_gaps=[_temperature_gap()],
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain=spec.strain, ploidy="diploid"
            ),
            environment_reference=environment,
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=spec.measurement_type,
                assay_type=AssayType.pooled_competitive_growth_barcode,
                environment_response=0.0,
                n_samples=n_control_arrays,
                sample_unit=SampleUnit.biological_replicate,
                screen_id=control_set,
                units=(
                    f"no-drug control set {control_set!r}: {n_control_arrays} control "
                    "arrays on the same deletion pool, generation count and scanner, "
                    "run in YPD with a DMSO vehicle at concentration 0 as the release "
                    "spells it. The score is 0 by construction -- it is the control "
                    "mean this set's treatment arrays are scored against"
                ),
            ),
        )

    def _phenotype(
        self, values: list[float], control_set: str
    ) -> EnvironmentResponsePhenotype:
        """Mean over the replicate arrays of one group, with the across-array sample SD.

        A sample SD of exactly 0 is NOT reported as a dispersion. The matrices print four
        to six decimals, so two arrays agreeing to the last printed digit (measured: one
        (strain, environment, control set) group per matrix, ``YHR096C`` at 0.428325 on
        two het arrays and ``YAL068C`` at 4.7108 on two hom arrays) means the dispersion
        is below the release's resolution, not that it was measured to be zero. Storing 0
        would give those records a standard error of 0 and claim infinite precision, so
        the uncertainty is a typed absence instead.
        """
        n = len(values)
        mean = sum(values) / n
        sd = (
            math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
            if n >= 2
            else None
        )
        gaps: list[ProvenanceGap] = []
        if sd == 0.0:
            sd = None
            gaps.append(
                ProvenanceGap(
                    field="environment_response_uncertainty",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    looked_in=_SOM_PROVENANCE,
                    note=f"the {n} arrays of this group print the identical score to the "
                    "release's last decimal, so the replicate dispersion is below the "
                    "released precision rather than measured to be zero",
                )
            )
        return EnvironmentResponsePhenotype(
            measurement_type=self.spec.measurement_type,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=mean,
            n_samples=n,
            sample_unit=SampleUnit.biological_replicate,
            environment_response_uncertainty=sd,
            environment_response_uncertainty_type=(
                UncertaintyType.sample_sd if sd is not None else None
            ),
            screen_id=control_set,
            units=self.spec.units,
            provenance_gaps=gaps,
        )

    # ---- matrix reading -------------------------------------------------------
    def _resolver(self) -> Callable[[str], Any]:
        """``resolve_gene_name`` from the supplied genome, or a read-only S288C genome."""
        if self.genome is None:
            from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

            data_root = _data_root()
            # overwrite=False is mandatory: a rebuild here would race any other process
            # holding the same gffutils database.
            self.genome = SCerevisiaeGenome(
                genome_root=osp.join(data_root, "data/sgd/genome"),
                go_root=osp.join(data_root, "data/go"),
                overwrite=False,
            )
        resolver: Callable[[str], Any] = self.genome.resolve_gene_name
        return resolver

    def _read_rows(
        self, path: str, n_columns: int, sgd_genes: set[str]
    ) -> tuple[
        dict[str, list[list[float | None]]],
        dict[str, list[str]],
        dict[str, dict[str, str]],
    ]:
        """Stream the matrix into per-ORF value rows, resolving every row id to R64.

        The row id is ``ORF:batch``. The ORF goes through ``resolve_gene_name`` first, so
        a renamed strain is kept under the current systematic name instead of being
        discarded; a retired id or a non-gene locus is dropped with its reason. The batch
        is kept out of the record (no served class carries one) and returned for the
        strain/batch artifacts.
        """
        resolve = self._resolver()
        rows: dict[str, list[list[float | None]]] = {}
        batches: dict[str, list[str]] = {}
        dropped: dict[str, dict[str, str]] = {}
        resolved: dict[str, str | None] = {}
        with open(path) as handle:
            handle.readline()
            for line in handle:
                cells = line.rstrip("\n").split("\t")
                row_id = cells[0].strip().strip('"')
                source_orf, _, batch = row_id.partition(":")
                if source_orf not in resolved:
                    resolution = resolve(source_orf)
                    status = str(getattr(resolution.status, "value", resolution.status))
                    name = resolution.systematic_name
                    if status == "current" and name in sgd_genes:
                        resolved[source_orf] = name
                    elif (
                        status == "renamed"
                        and name is not None
                        and name in sgd_genes
                        and str(
                            getattr(resolve(name).status, "value", resolve(name).status)
                        )
                        == "current"
                    ):
                        resolved[source_orf] = name
                    else:
                        resolved[source_orf] = None
                        dropped[source_orf] = {
                            "status": status,
                            "resolved_to": str(name),
                            "in_sgd_fasta": str(name in sgd_genes),
                        }
                orf = resolved[source_orf]
                if orf is None:
                    continue
                values: list[float | None] = [None] * n_columns
                for index in range(1, min(n_columns, len(cells))):
                    cell = cells[index].strip()
                    if cell.upper() not in _MISSING:
                        values[index] = float(cell)
                rows.setdefault(orf, []).append(values)
                batches.setdefault(orf, []).append(batch)
        return rows, batches, dropped

    # ---- build ----------------------------------------------------------------
    @post_process
    def process(self) -> None:
        """Aggregate this matrix into per (strain, environment, control set) records."""
        spec = self.spec
        data_root = _data_root()
        sgd_genes = _load_sgd_genes(data_root)
        manifest = load_manifest(data_root)
        rel = raw_relpaths()
        matrix_path = osp.join(self.raw_dir, spec.filename)
        got = _sha256(matrix_path)
        expected = manifest_sha256(manifest, rel[spec.filename])
        if got != expected:
            raise RuntimeError(
                f"{spec.filename} sha256 mismatch at build time: got {got}, expected "
                f"{expected}"
            )

        control_sets = read_control_set_map(osp.join(self.raw_dir, spec.keyfile))
        control_sizes = read_control_set_sizes(
            osp.join(self.raw_dir, spec.controls_file)
        )
        with open(matrix_path) as handle:
            header = handle.readline().rstrip("\n").split("\t")
        columns = parse_columns(header, control_sets)

        kept_columns = [c for c in columns if c.drop_reason is None]
        dropped_columns = [c for c in columns if c.drop_reason is not None]
        groups: dict[str, list[ColumnSpec]] = {}
        for column in kept_columns:
            groups.setdefault(column.group_key, []).append(column)
        dropped_groups: dict[str, list[ColumnSpec]] = {}
        for column in dropped_columns:
            dropped_groups.setdefault(column.group_key, []).append(column)
        log.info(
            "Hillenmeyer2008 %s: %d arrays -> %d kept environments (%d arrays), "
            "%d dropped environments (%d arrays)",
            spec.key,
            len(columns),
            len(groups),
            len(kept_columns),
            len(dropped_groups),
            len(dropped_columns),
        )

        rows, batches, dropped_genes = self._read_rows(
            matrix_path, len(header), sgd_genes
        )
        references = {
            control_set: self._reference(control_set, control_sizes[control_set])
            for control_set in {c.control_set for c in kept_columns}
        }
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")
        group_list = [(members[0], members) for members in groups.values()]
        dropped_list = [(members[0], members) for members in dropped_groups.values()]
        dropped_record_counts: dict[str, int] = {}

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        batch_size = 250_000
        txn = env.begin(write=True)
        itxn = interned_env.begin(write=True)
        for orf, orf_rows in tqdm(rows.items(), desc=f"Hillenmeyer2008 {spec.key}"):
            genotype = self._genotype(orf)
            for head, members in group_list:
                values = self._array_values(orf_rows, members)
                if not values:
                    continue
                experiment = EnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=head.environment,
                    phenotype=self._phenotype(values, head.control_set),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(
                        experiment, references[head.control_set], publication, itxn
                    ),
                )
                idx += 1
                if idx % batch_size == 0:
                    itxn.commit()
                    txn.commit()
                    txn = env.begin(write=True)
                    itxn = interned_env.begin(write=True)
            for head, members in dropped_list:
                if self._array_values(orf_rows, members):
                    # keyed by the GROUP, not by the rule: one compound is dropped at
                    # several doses and control sets, and a per-rule key would count its
                    # records once per environment that shares the rule.
                    dropped_record_counts[head.group_key] = (
                        dropped_record_counts.get(head.group_key, 0) + 1
                    )
        itxn.commit()
        txn.commit()
        env.close()
        interned_env.close()

        self._write_drop_report(
            dropped_groups, dropped_record_counts, dropped_genes, idx
        )
        self._write_batch_report(batches)
        log.info(
            "Wrote %d Hillenmeyer2008 %s environment-response records to LMDB",
            idx,
            spec.key,
        )

    @staticmethod
    def _array_values(
        orf_rows: list[list[float | None]], members: list[ColumnSpec]
    ) -> list[float]:
        """One value per replicate ARRAY: the mean over this ORF's construction rows.

        The rows of one ORF are different physical strains, not replicates, so they are
        collapsed inside an array before the across-array replicate statistic is taken.
        """
        values: list[float] = []
        for column in members:
            cells = [
                row[column.index] for row in orf_rows if row[column.index] is not None
            ]
            if cells:
                values.append(sum(cells) / len(cells))  # type: ignore[arg-type]
        return values

    def _write_drop_report(
        self,
        dropped_groups: dict[str, list[ColumnSpec]],
        dropped_record_counts: dict[str, int],
        dropped_genes: dict[str, dict[str, str]],
        kept_records: int,
    ) -> None:
        """Write ``dropped_records.json`` / ``dropped_genes.json``: the rules and counts."""
        environments = []
        for members in dropped_groups.values():
            head = members[0]
            environments.append(
                {
                    "rule": head.drop_reason,
                    "detail": head.drop_detail,
                    "n_arrays": len(members),
                    "headers": [c.header for c in members],
                    "n_records_not_written": dropped_record_counts.get(
                        head.group_key, 0
                    ),
                }
            )
        by_rule: dict[str, dict[str, int]] = {}
        for entry in environments:
            rule = str(entry["rule"])
            bucket = by_rule.setdefault(
                rule, {"n_environments": 0, "n_arrays": 0, "n_records_not_written": 0}
            )
            bucket["n_environments"] += 1
            bucket["n_arrays"] += int(entry["n_arrays"])  # type: ignore[arg-type]
            bucket["n_records_not_written"] += int(entry["n_records_not_written"])  # type: ignore[arg-type]
        payload = {
            "dataset": self.name,
            "matrix": self.spec.filename,
            "kept_records": kept_records,
            "rules": {
                DROP_UNIDENTIFIABLE: "the condition names a compound the pinned "
                "compound-identity table resolves to no InChIKey, ChEBI id or PubChem "
                "CID; the per-name audit trail is that row's unresolved_reason",
                DROP_HEAT_SHOCK_CYCLE: "'37c, 45c' is a heat-shock cycle; PhysicalFactor "
                "has no heat_shock member (adding one changes a served class) and a "
                "steady 45 C would be false",
                DROP_UNNAMED_MEDIA_AGENT: "a dose is given but the agent is named "
                "nowhere in the SOM or the release",
            },
            "by_rule": by_rule,
            "environments": sorted(
                environments, key=lambda e: (str(e["rule"]), str(e["detail"]))
            ),
        }
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            json.dump(payload, handle, indent=2)
        with open(osp.join(self.preprocess_dir, "dropped_genes.json"), "w") as handle:
            json.dump(
                {
                    "rule": "a row id is kept only when resolve_gene_name reports it "
                    "CURRENT (or RENAMED onto a current name) and the resolved name is "
                    "in the SGD R64 ORF/RNA FASTA universe",
                    "n_dropped": len(dropped_genes),
                    "dropped": dict(sorted(dropped_genes.items())),
                },
                handle,
                indent=2,
            )

    def _write_batch_report(self, batches: dict[str, list[str]]) -> None:
        """Write the construction-batch artifacts, including the SOM's suspicious set.

        The batch cannot ride on the record (no served perturbation class carries a
        construction batch and adding the field would force a full rebuild), so the
        authoritative flag lives beside the build, as the Hoepfner background-mutation
        purge list does.
        """
        suspicious = set(SUSPICIOUS_BATCHES)
        flagged = {
            orf: sorted(set(orf_batches) & suspicious)
            for orf, orf_batches in batches.items()
            if set(orf_batches) & suspicious
        }
        multi = {
            orf: sorted(orf_batches)
            for orf, orf_batches in batches.items()
            if len(orf_batches) > 1
        }
        with open(osp.join(self.preprocess_dir, "strain_batches.json"), "w") as handle:
            json.dump(
                {
                    "quote_batch_ids": QUOTE_BATCH_IDS,
                    "quote_reconstructed": QUOTE_RECONSTRUCTED,
                    "citation_key": CITATION_KEY,
                    "sha256": SOM_SHA256,
                    "n_orfs": len(batches),
                    "n_rows": sum(len(v) for v in batches.values()),
                    "n_orfs_with_multiple_batches": len(multi),
                    "batches_by_orf": dict(sorted(batches.items())),
                },
                handle,
                indent=2,
            )
        with open(
            osp.join(self.preprocess_dir, "suspicious_batch_strains.json"), "w"
        ) as handle:
            json.dump(
                {
                    "quote": QUOTE_SUSPICIOUS,
                    "quote_count": QUOTE_SUSPICIOUS_COUNT,
                    "quote_homozygous": QUOTE_SUSPICIOUS_HOM,
                    "citation_key": CITATION_KEY,
                    "sha256": SOM_SHA256,
                    "suspicious_batches": list(SUSPICIOUS_BATCHES),
                    "collection": self.spec.collection,
                    "n_flagged_orfs": len(flagged),
                    "note": "these ORFs have at least one construction row in a batch "
                    "the SOM names as carrying a shared secondary genotype; the authors "
                    "excluded 647 heterozygous strains on this basis and state the "
                    "homozygous collection did not show the pattern. Records are SERVED "
                    "with the flag recorded here rather than dropped, because the "
                    "exclusion is an analysis choice, not a defect of the measurement",
                    "flagged_orfs": dict(sorted(flagged.items())),
                },
                handle,
                indent=2,
            )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "Hillenmeyer2008 builds its records in process(); see _array_values"
        )


@register_dataset
class HetHillenmeyer2008Dataset(_Hillenmeyer2008Base):
    """FitDb HIP (heterozygous, engineered-CNV) fitness-defect log2-ratio dataset."""

    _matrix_key = "het"

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_hillenmeyer2008_het",
        io_workers: int = 0,
        genome: Any | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HET (HIP) dataset."""
        super().__init__(root, io_workers, genome, transform, pre_transform, **kwargs)


@register_dataset
class HomHillenmeyer2008Dataset(_Hillenmeyer2008Base):
    """FitDb HOP (homozygous, KanMX deletion) fitness-defect z-score dataset."""

    _matrix_key = "hom"

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_hillenmeyer2008_hom",
        io_workers: int = 0,
        genome: Any | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HOM (HOP) dataset."""
        super().__init__(root, io_workers, genome, transform, pre_transform, **kwargs)


def main() -> None:
    """Build/load both datasets for interactive debugging."""
    data_root = _data_root()
    for cls, sub in (
        (HetHillenmeyer2008Dataset, "het"),
        (HomHillenmeyer2008Dataset, "hom"),
    ):
        root = osp.join(data_root, f"data/torchcell/env_chemgen_hillenmeyer2008_{sub}")
        dataset = cls(root=root)
        print(f"{sub}: len = {len(dataset)}")
        print(dataset[0])


if __name__ == "__main__":
    main()
