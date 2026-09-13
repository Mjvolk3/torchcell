# torchcell/datasets/scerevisiae/vanacloig2022
# [[torchcell.datasets.scerevisiae.vanacloig2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/vanacloig2022
# Test file: tests/torchcell/datasets/scerevisiae/test_vanacloig2022.py
"""Vanacloig-Pedros 2022 comparative chemical-genomic screen (env x geno -> response).

Vanacloig-Pedros et al. 2022 (FEMS Yeast Research, doi:10.1093/femsyr/foac036) profiled
the '3DeltaAlpha' drug-sensitized barcoded yeast deletion library ANAEROBICALLY against
plant-hydrolysate inhibitors at their IC30, in independent biological triplicate, alongside
matched inhibitor-free controls on the same plates. The readout is log2(inhibitor/control)
barcode abundance, a per-deletion fitness response.

READOUT PROVENANCE / RIGOR. The paper's published values are edgeR TMM+glmQLF PAIRED
logFCs whose exact reproduction needs R/edgeR plus the OUP Supplementary Table S1 (the
per-compound control pairing and the IC30 molar values), and academic.oup.com is not
scriptable, so Table S1 is not mirrored. GEO GSE186866 releases only the raw barcode-count
matrix, which IS scriptable and sha256-pinned. This loader therefore RECOMPUTES the paper's
DEFINED quantity from that canonical artifact: per-sample CPM normalization against the
library size summed over EVERY released barcode (so the stored value does not depend on
this loader's own retention rules),
then per gene ``log2((CPM_treated_rep + 1) / (CPM_control_batch + 1))`` for each replicate.
The stored response is the mean of the three replicate log2 ratios and the uncertainty is
their sample SD (SE = SD/sqrt(3)). It is NOT identical to the published edgeR logFC and
must not be treated as such; nothing mirrored here can check the stored number against a
published one, so L2 value fidelity checks range and finiteness only, never agreement.

CONTROL PAIRING. Each replicate is paired with the control columns of its OWN ``CG00n``
batch, because the paper's design is paired ("All 24-well plates contained control samples
with SynBase or SynBase + 1% DMSO lacking any inhibitor for paired analysis") and the
control log2 CPM measurably varies across the four batches. MMS is the one retained
compound the paper analyzed UNPAIRED, so its control is the mean of all 16 control columns;
its ``units`` string records that, which also keeps it distinguishable in the record key.

SOURCING. Every environment and phenotype number is a module-level ``SourcedValue``
carrying the verbatim quote plus the sha256 of the mirrored artifact it came from, or a
typed ``ProvenanceGap``. The medium is the shared ``SYNBASE`` library object (SynH3- minus
acetamide/sodium acetate/cellobiose, MSG for ammonium sulfate); its pH 5.0 rides as an
``EnvironmentPhysicalPerturbation`` whose ``agent`` is the HCl the same sentence names,
because ``Media`` has no pH field. The one field that is genuinely ``None`` on a dosed
perturbation, ``solvent``, carries a ``deferred_pending_source_review`` gap resolvable by
Table S1; the absent per-compound IC30 MOLAR value is not a second gap, because
``concentration`` is never None (``basis=IC30`` is what is known and is the schema's own
mechanism for a dose set to a target without a released number) and ``Concentration`` is
not itself a gap carrier.

RECORDS DROPPED (rule + count written to ``preprocess/dropped_records.json``): the DMSO
vehicle-control column (it is the denominator of the DMSO-delivered compounds, not a
treatment); compounds with no resolvable structure identifier (MBO, whose abbreviation the
paper contradicts itself on, and the two QUADRIS suspension doses); library rows whose ORF
is not a current R64 gene or is the legacy spelling of an ORF already in the pool; and
cells whose three replicate counts are ALL zero, where the CPM pseudocount would otherwise
manufacture a finite value with a sample SD of exactly 0.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import (
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import SYNBASE
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MarkerDeletionPerturbation,
    MeasurementType,
    NatMxDeletionPerturbation,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.gene_name_reconcile import default_genome
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "vanacloig-pedrosComparativeChemicalGenomic2022"
PAPER_DOI = "10.1093/femsyr/foac036"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

DATA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186866/suppl/"
    "GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz"
)
DATA_FILENAME = "GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz"
DATA_SHA256 = "e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a"
DATA_REL = f"data/{DATA_FILENAME}"
DATA_RETRIEVED_AT = "2026-09-13"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "0b5d938b54b8424fa08203a4357bc8f7c7dfae3fbe1a6d07d422848b92f37ba3"

CPM_PRIOR = 1.0  # CPM pseudocount; the all-zero cells it would fabricate are dropped


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned paper OCR mirror."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page="Methods, 'Strains and growth conditions' / 'Chemical genomic experiment'",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced environment + phenotype constants (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_GROWTH_QUOTE = (
    "Inoculated plates were grown in an anaerobic chamber (Coy Laboratory Products, "
    "Inc.), containing $1 \\% - 2 \\%$ $\\mathrm { H } _ { 2 }$ , $4 \\% - 5 \\%$ "
    "$\\mathrm { C O } _ { 2 }$ , and $9 0 \\% { - } 9 5 \\% \\mathrm { N } _ { 2 }$ at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ for $2 4 \\mathrm { { h } }$ , and then "
    "transferred into the identical fresh medium at $\\mathrm { O D } _ { 6 0 0 } = 0 . 1$ "
    "for another $2 4 \\mathrm { { h } }$ without shaking."
)

TEMPERATURE_C = _paper(30.0, _GROWTH_QUOTE)
AEROBICITY = _paper("anaerobic", _GROWTH_QUOTE)
DURATION_HOURS = _paper(
    48.0,
    _GROWTH_QUOTE,
    note="two consecutive 24 h anaerobic growth periods in identical fresh medium",
)
DURATION_GENERATIONS = _paper(
    6.5,
    "All cell cultures reached between 6.5 to 10 total cell doublings within the two "
    "$2 4 \\mathrm { ~ h ~ }$ growth periods.",
    note="the primary releases a RANGE and no per-condition value, and no companion "
    "statistic permits a back-solve, so the CLAUDE.md rule takes the conservative "
    "lower end (6.5 doublings) rather than the optimistic one",
)
MEDIUM_PH = _paper(
    5.0,
    "ammonium sulfate was replaced with ${ \\mathrm { ~ 1 ~ g / L ~ } }$ monosodium "
    "glutamate (MSG, Fisher Scientific) and adjusted to $\\mathrm { p H } ~ 5 . 0$ with "
    "HCl.",
    note="a medium-intrinsic pH; Media has no ph field (it sits in 36 served dataset "
    "closures, so adding one is a full rebuild), so it rides as a typed physical factor",
)
N_REPLICATES = _paper(
    3,
    "Growth of the yeast deletion library in all inhibitory and control conditions were "
    "performed in independent biological triplicate.",
)
ASSAY = _paper(
    AssayType.pooled_competitive_growth_barcode,
    "Barcode read counts were calculated from up-tag reads using custom python scripts.",
    note="a pooled library grown competitively and read out by amplifying each strain's "
    "UPTAG barcode",
)
LIBRARY_COLLECTION = _paper(
    "3DeltaAlpha drug-sensitive yeast deletion collection of 4309 mutants",
    "Saccharomyces cerevisiae strains used in the chemical genomics study belong to the "
    "‘3DeltaAlpha’ drug-sensitive yeast deletion collection of 4309 mutants",
)
BARCODE_IS_UPTAG = _paper(
    "uptag",
    "The library includes 4309 strains in which a non-essential gene is replaced with a "
    "unique DNA sequence (barcode) flanked by common sequences for barcode amplification.",
    note="the matrix's gene column is '<ORF>_<barcode>' and the counts are up-tag reads",
)
IC30_BASIS = _paper(
    DoseBasis.IC30,
    "Concentrations for each inhibitor used for the chemical genomics experiment with "
    "the yeast deletion library were determined based on estimated inhibition of "
    "${ \\sim } 3 0 \\%$ of growth $\\left( \\mathrm { I C } _ { 3 0 } \\right)$ in "
    "SynBase medium with the inhibitor relative to growth in SynBase medium lacking the "
    "inhibitor (Table S1, Supporting Information).",
    note="the per-compound molar values live in Table S1, which academic.oup.com does "
    "not serve to a script and which is therefore not mirrored; Concentration.value "
    "stays None and the IC30 basis carries the dose provenance",
)
BENOMYL_MMS_DOSE = _paper(
    {"benomyl_ug_per_ml": 10.0, "mms_percent": 0.01},
    "Benomyl and MMS concentrations were used as previously published (Piotrowski et al. "
    "2017), $1 0 ~ \\mathrm { u g / m L }$ and $0 . 0 1 \\%$ , respectively.",
    note="both doses were taken from Piotrowski 2017 rather than set to an IC30, so "
    "their basis is 'fixed'. The MMS percent is stored as a basis only: the primary "
    "writes '0.01%' with no v/v or w/v, and the deferral target (Piotrowski 2017) is "
    "not mirrored, so the unit would be a guess",
)
UNPAIRED_COMPOUNDS_QUOTE = (
    "Gene deletions with specific fitness contributions were identified using linear "
    "models in edgeR version 3.26.8 (Robinson et al. 2010), using TMM normalization and "
    "glmQLFit comparing paired treatment to control samples, except with MMS and QUADRIS "
    "compounds, which were unpaired."
)
PAIRED_CONTROL = _paper(
    "batch-matched",
    "All 24-well plates contained control samples with SynBase or SynBase $+ ~ 1 \\%$ "
    "DMSO lacking any inhibitor for paired analysis (see below).",
    note="each replicate is paired with the ControlN columns of its own CG00n batch; "
    "MMS keeps the pooled 16-column control because the paper analyzed it unpaired: "
    + UNPAIRED_COMPOUNDS_QUOTE,
)
VEHICLE_CONTROL = _paper(
    "DMSO",
    "Chemical compounds insoluble in water were dissolved in DMSO at 100X concentration "
    "so that the final concentration of DMSO in SynBase medium was $1 \\%$ $( \\mathrm "
    "{ v / v } )$ .",
    note="DMSO is the vehicle whose own column is a control, not an inhibitor; which "
    "compounds it delivered is in the unmirrored Table S1, so no per-compound Solvent "
    "can be asserted",
)
READOUT = _paper(
    MeasurementType.log2_ratio,
    "Results were presented in heat map figures as the $\\log _ { 2 }$ of the normalized "
    "read counts for inhibitor/control ratio.",
)
PH_AGENT = _paper(
    "hydrochloric acid",
    "adjusted to $\\mathrm { p H } ~ 5 . 0$ with HCl.",
    note="the acid that REALIZES the pH factor, carried on the physical perturbation's "
    "`agent` slot so the medium's pH joins on a compound entity",
)

#: The one unmirrored artifact that would close this dataset's recoverable gaps: the OUP
#: supplement holding the per-compound IC30 molar values AND which compounds were
#: delivered in DMSO. academic.oup.com returns 403 to a script, so it is not mirrored.
TABLE_S1 = Provenance(
    source_uri="https://doi.org/10.1093/femsyr/foac036 (Table S1, Supporting Information)",
    citation_key=CITATION_KEY,
    method="publisher supplementary table; academic.oup.com is not scriptable",
    page="Table S1",
)


def _solvent_gap() -> ProvenanceGap:
    """The typed absence of a per-compound vehicle.

    The vehicle is the field that is ACTUALLY ``None`` on the perturbation: the primary
    says water-insoluble compounds went in at 1% v/v DMSO but names them only in Table
    S1, so for any one compound it is unknown whether a vehicle was used at all. The
    DOSE is not a second gap: ``concentration`` is never None (an IC30 or fixed basis is
    always known), and ``Concentration`` is not itself a gap carrier, so the missing
    molar value is carried by ``basis`` -- the mechanism the schema documents for exactly
    this case.
    """
    return ProvenanceGap(
        field="solvent",
        reason=ProvenanceGapReason.deferred_pending_source_review,
        resolve_with=TABLE_S1,
        note=VEHICLE_CONTROL.quote
        + " Which compounds that covers is in Table S1, which is not mirrored, so the "
        "vehicle of any one compound is unknown rather than absent.",
    )


#: The constant 3DeltaAlpha background, deleted in EVERY library strain.
BACKGROUND_GENES = frozenset({"YGL013C", "YBL005W", "YDR011W"})

#: Compound column tokens that are NOT a treatment (the vehicle's own control column).
VEHICLE_CONTROL_TOKENS = frozenset({"DMSO"})

_SYSTEMATIC_RE = re.compile(
    r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
)
_SAMPLE_RE = re.compile(r"^(?P<compound>.+)_CG(?P<batch>\d+)_rep(?P<rep>\d+)$")
_CONTROL_RE = re.compile(r"^Control\d+_CG(?P<batch>\d+)$")

#: The one compound the paper analyzed UNPAIRED, so it keeps the pooled control.
UNPAIRED_COMPOUND_TOKENS = frozenset({"MMS"})

_PAIRED_UNITS = (
    "log2((CPM of the inhibitor replicate + 1) / (mean CPM of the SAME CG batch's "
    "inhibitor-free control columns + 1)), mean of 3 biological replicates; recomputed "
    "from the GEO GSE186866 raw up-tag counts, NOT the paper's edgeR logFC"
)
_POOLED_UNITS = (
    "log2((CPM of the inhibitor replicate + 1) / (mean CPM of all 16 inhibitor-free "
    "control columns + 1)), mean of 3 biological replicates; pooled rather than "
    "batch-matched because the paper analyzed this compound unpaired; recomputed from "
    "the GEO GSE186866 raw up-tag counts, NOT the paper's edgeR logFC"
)


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never the live GEO URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/vanacloig-pedrosComparativeChemicalGenomic2022``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deposit_raw_mirror(
    *,
    counts_path: str | Path,
    retrieved_at: str = DATA_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from an already-retrieved count matrix + its ``manifest.json``.

    Idempotent by sha256: an existing mirror file with the recorded hash is left alone and
    a differing one raises rather than being overwritten. The GEO supplementary URL is
    directly scriptable, so the recorded retrieval re-runs as-is.
    """
    root = raw_mirror_dir(data_root)
    retrieval = RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=DATA_URL,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": DATA_URL},
        sha256=DATA_SHA256,
        retrieved_at=retrieved_at,
    )
    got = _sha256(counts_path)
    if got != DATA_SHA256:
        raise RuntimeError(
            f"{counts_path} sha256 mismatch: got {got}, expected {DATA_SHA256}"
        )
    dest = root / DATA_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != DATA_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(counts_path, dest)
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=(
            "Comparative chemical genomic profiling across plant-based hydrolysate "
            "toxins reveals widespread antagonism in fitness contributions"
        ),
        files=[
            ArtifactRecord(
                path=DATA_REL,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=DATA_SHA256,
                source=DATA_URL,
                retrieval=retrieval,
            )
        ],
        si_data_sources=[
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186866",
            DATA_URL,
        ],
        si_expected=[
            "Table S1 (per-compound IC30 molar values + DMSO control pairing) -- "
            "academic.oup.com is not scriptable, so it is NOT mirrored",
            "Dataset2_mclust_cdt (clustered edgeR logFC matrix) -- same publisher gate",
        ],
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
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["compound", "library_row", "cell"]
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


class LibraryRows(BaseModel):
    """The library rows a build keeps, after the gene-name policy."""

    keep_mask: list[bool]
    systematic: list[str]
    common: list[str]
    barcode: list[str]
    dropped_retired: list[str]
    dropped_legacy_duplicate: list[str]


def _canonical_common_names(genome: SCerevisiaeGenome) -> dict[str, str]:
    """``systematic name -> the genome's own standard (common) name``.

    One spelling per gene is what keeps a perturbation from splitting into two graph
    nodes, and taking it from the genome rather than from the source is what makes the
    spelling identical across datasets. Only a standard name that resolves BACK to the
    gene is used, so the stored pair always round-trips through the resolver.
    """
    canonical: dict[str, str] = {}
    for standard in genome.feature_index["standard_to_ids"]:
        resolution = genome.resolve_gene_name(standard)
        if resolution.is_current_gene and resolution.systematic_name is not None:
            canonical.setdefault(resolution.systematic_name, standard)
    return canonical


def resolve_library_rows(
    orfs: pd.Series, barcodes: pd.Series, genome: SCerevisiaeGenome
) -> LibraryRows:
    """Map every library row onto a current R64 gene, or drop it with a typed reason.

    Two rules, both of which the L1 canonical-name and L4 current-genome gates enforce
    downstream. A row whose ORF no longer names a gene of the current genome (a retired
    2005-era ORF, or a feature that is not a gene) is dropped: it cannot be keyed to a
    gene entity. A row whose ORF is the LEGACY spelling of an ORF the same library also
    carries under its current name is dropped too, because remapping it would merge two
    physically distinct barcoded strains into one record key.
    """
    gene_set = {gene.upper() for gene in genome.gene_set}
    resolutions = {orf: genome.resolve_gene_name(orf) for orf in orfs.unique()}
    target: dict[str, str] = {}
    retired: list[str] = []
    for orf, resolution in resolutions.items():
        mapped = resolution.systematic_name
        if resolution.is_current_gene and mapped is not None and mapped in gene_set:
            target[orf] = mapped
        else:
            retired.append(orf)
    claimed = {orf for orf in target if orf in gene_set}
    legacy = sorted(
        orf for orf, mapped in target.items() if orf != mapped and mapped in claimed
    )
    canonical = _canonical_common_names(genome)
    keep_mask: list[bool] = []
    systematic: list[str] = []
    common: list[str] = []
    kept_barcodes: list[str] = []
    legacy_set = set(legacy)
    for orf, barcode in zip(orfs, barcodes, strict=True):
        mapped = target.get(orf)
        keep = mapped is not None and orf not in legacy_set
        keep_mask.append(keep)
        if not keep:
            continue
        assert mapped is not None
        systematic.append(mapped)
        common.append(canonical.get(mapped, mapped))
        kept_barcodes.append(barcode)
    if len(set(systematic)) != len(systematic):
        raise RuntimeError(
            "two retained library rows resolve to the same systematic gene; the "
            "legacy-duplicate rule did not separate them"
        )
    return LibraryRows(
        keep_mask=keep_mask,
        systematic=systematic,
        common=common,
        barcode=kept_barcodes,
        dropped_retired=sorted(retired),
        dropped_legacy_duplicate=legacy,
    )


@register_dataset
class EnvChemgenVanacloig2022Dataset(ExperimentDataset):
    """Anaerobic chemical-genomic env x geno -> log2(inhibitor/control) response screen."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_vanacloig2022",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (the genome is loaded lazily inside ``process``)."""
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
        """The GEO raw barcode-count matrix required before processing."""
        return [DATA_FILENAME]

    def download(self) -> None:
        """Link the manifest-listed mirror file into ``raw/`` and verify its sha256.

        The mirror + its recorded sha256 is canonical; the GEO URL is retrieval metadata
        that ``deposit_raw_mirror`` re-runs, never a live build dependency.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        expected = manifest_sha256(manifest, DATA_REL)
        src = raw_mirror_dir(data_root) / DATA_REL
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        got = _sha256(src)
        if got != expected:
            raise RuntimeError(
                f"{DATA_FILENAME} sha256 mismatch: got {got}, expected {expected}"
            )
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, DATA_FILENAME)
        if not osp.exists(dest):
            os.symlink(src, dest)
        log.info(
            "Vanacloig 2022 raw matrix linked into %s (sha256 verified)", self.raw_dir
        )

    def _load_matrix(self) -> pd.DataFrame:
        """Read the gzipped raw-count TSV into a DataFrame."""
        path = osp.join(self.raw_dir, DATA_FILENAME)
        with gzip.open(path, "rt") as handle:
            return pd.read_csv(handle, sep="\t")

    # ---- environment / phenotype builders ------------------------------------ #
    def _concentration(self, compound: str) -> Concentration:
        """The dose as the paper SET it: an IC30 target, or a published fixed dose."""
        if compound == "Benomyl":
            return Concentration(
                value=BENOMYL_MMS_DOSE.value["benomyl_ug_per_ml"],
                unit=ConcentrationUnit.ug_per_ml,
                basis=DoseBasis.fixed,
            )
        if compound == "MMS":
            return Concentration(basis=DoseBasis.fixed)
        return Concentration(basis=IC30_BASIS.value)

    def _base_environment(self, perturbations: list[Any]) -> Environment:
        """Anaerobic SynBase at pH 5.0 carrying ``perturbations`` on top."""
        return Environment(
            media=SYNBASE,
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=perturbations,
            aerobicity=AEROBICITY.value,
            duration_hours=DURATION_HOURS.value,
            duration_generations=DURATION_GENERATIONS.value,
        )

    def _ph(self) -> EnvironmentPhysicalPerturbation:
        """SynBase's stated pH, typed (``Media`` carries no pH field).

        ``agent`` is the acid the same sentence names as having set it, so the factor's
        realizing species is sourced rather than a silent None.
        """
        return EnvironmentPhysicalPerturbation(
            factor=PhysicalFactor.ph,
            magnitude=Concentration(value=MEDIUM_PH.value, unit=ConcentrationUnit.ph),
            agent=resolved_compound(PH_AGENT.value),
        )

    def _environment(self, compound: str) -> Environment:
        """The treated environment: SynBase + pH 5.0 + the inhibitor at its dose."""
        return self._base_environment(
            [
                SmallMoleculePerturbation(
                    compound=resolved_compound(compound),
                    concentration=self._concentration(compound),
                    provenance_gaps=[_solvent_gap()],
                ),
                self._ph(),
            ]
        )

    def _reference(self, compound: str) -> EnvironmentResponseExperimentReference:
        """The inhibitor-FREE control the log2 ratio is taken against.

        The reference environment is the one the denominator was measured in: SynBase at
        pH 5.0 with no inhibitor. Storing the treated environment here (the previous
        build) made the control for a compound contain that very compound.
        """
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="S288C"
            ),
            environment_reference=self._base_environment([self._ph()]),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=READOUT.value,
                assay_type=ASSAY.value,
                environment_response=0.0,
                units=self._units(compound),
            ),
        )

    def _units(self, compound: str) -> str:
        """The readout definition, which records WHICH control the ratio used."""
        return _POOLED_UNITS if compound in UNPAIRED_COMPOUND_TOKENS else _PAIRED_UNITS

    def _phenotype(
        self, compound: str, response: float, sd: float
    ) -> EnvironmentResponsePhenotype:
        """One log2-ratio response with its across-replicate sample SD."""
        return EnvironmentResponsePhenotype(
            measurement_type=READOUT.value,
            assay_type=ASSAY.value,
            environment_response=response,
            environment_response_uncertainty=sd,
            environment_response_uncertainty_type=UncertaintyType.sample_sd,
            n_samples=N_REPLICATES.value,
            sample_unit=SampleUnit.biological_replicate,
            units=self._units(compound),
        )

    def _genotype(self, systematic: str, common: str, barcode: str) -> Genotype:
        """The screened deletion (with its uptag barcode) plus the constant background."""
        return Genotype(
            perturbations=[
                BarcodedKanMxDeletionPerturbation(
                    systematic_gene_name=systematic,
                    perturbed_gene_name=common,
                    barcode=barcode,
                    collection=LIBRARY_COLLECTION.value,
                ),
                NatMxDeletionPerturbation(
                    systematic_gene_name="YGL013C", perturbed_gene_name="PDR1"
                ),
                MarkerDeletionPerturbation(
                    systematic_gene_name="YBL005W",
                    perturbed_gene_name="PDR3",
                    marker="KlURA3",
                ),
                MarkerDeletionPerturbation(
                    systematic_gene_name="YDR011W",
                    perturbed_gene_name="SNQ2",
                    marker="KlLEU2",
                ),
            ]
        )

    # ---- build ---------------------------------------------------------------- #
    @post_process
    def process(self) -> None:
        """Recompute per-(gene, compound) log2 responses from raw counts; write LMDB."""
        df = self._load_matrix()
        sample_cols = [c for c in df.columns if c not in ("gene", "std_name")]
        control_by_batch: dict[str, list[str]] = {}
        compound_cols: dict[str, list[str]] = {}
        for column in sample_cols:
            control = _CONTROL_RE.match(column)
            if control is not None:
                control_by_batch.setdefault(control.group("batch"), []).append(column)
                continue
            sample = _SAMPLE_RE.match(column)
            if sample is None:
                raise RuntimeError(f"unparseable sample column: {column!r}")
            compound_cols.setdefault(sample.group("compound"), []).append(column)
        if not control_by_batch:
            raise RuntimeError("no ControlN_CG* columns found in GSE186866 matrix")
        controls = [c for cols in control_by_batch.values() for c in cols]

        source_records = len(df) * len(compound_cols)
        rules: list[DropRule] = []

        # --- compound-level retention ----------------------------------------- #
        vehicle = sorted(set(compound_cols) & VEHICLE_CONTROL_TOKENS)
        unidentified = sorted(
            token
            for token in compound_cols
            if token not in VEHICLE_CONTROL_TOKENS
            and not resolve_compound_identity(name=token).identified
        )
        kept_compounds = sorted(set(compound_cols) - set(vehicle) - set(unidentified))

        # --- library-row retention --------------------------------------------- #
        split = df["gene"].astype(str).str.split("_", n=1)
        orfs = split.str[0]
        barcodes = split.str[1].fillna("")
        is_orf = orfs.map(lambda gene: bool(_SYSTEMATIC_RE.match(gene)))
        has_counts = ~df[sample_cols].isna().any(axis=1)
        not_background = ~orfs.isin(BACKGROUND_GENES)
        prefilter = is_orf & has_counts & not_background
        n_non_orf = int((~is_orf).sum())
        n_all_nan = int((is_orf & ~has_counts).sum())
        n_background = int((is_orf & has_counts & ~not_background).sum())

        genome = default_genome()
        library = resolve_library_rows(
            orfs[prefilter].reset_index(drop=True),
            barcodes[prefilter].reset_index(drop=True),
            genome,
        )
        row_keep = pd.Series(library.keep_mask, index=df.index[prefilter])
        keep = pd.Series(False, index=df.index)
        keep.loc[row_keep.index] = row_keep.to_numpy()
        n_rows = int(keep.sum())
        n_kept_compounds = len(kept_compounds)

        rules.append(
            DropRule(
                rule="vehicle_control_served_as_a_treatment",
                scope="compound",
                description=(
                    "the DMSO column is the vehicle the water-insoluble compounds were "
                    "delivered in and its own inhibitor-free control, not a treatment: "
                    + VEHICLE_CONTROL.quote
                ),
                n_records=len(vehicle) * len(df),
                items=vehicle,
            )
        )
        rules.append(
            DropRule(
                rule="compound_without_a_structure_identifier",
                scope="compound",
                description=(
                    "no InChIKey / PubChem CID / ChEBI id resolves for the source label, "
                    "so the compound entity cannot be encoded or joined; see the per-"
                    "token reason in compound_identity_inputs/vanacloig2022.txt"
                ),
                n_records=len(unidentified) * len(df),
                items=unidentified,
            )
        )
        rules.append(
            DropRule(
                rule="row_is_not_a_barcoded_orf_or_carries_no_counts",
                scope="library_row",
                description=(
                    "the gene column is not '<systematic ORF>_<barcode>', or every count "
                    "column is missing (a QC-dropped barcode), or the ORF is one of the "
                    "three constant 3DeltaAlpha background deletions"
                ),
                n_records=(n_non_orf + n_all_nan + n_background) * n_kept_compounds,
                items=[],
            )
        )
        rules.append(
            DropRule(
                rule="orf_is_not_a_current_genome_gene",
                scope="library_row",
                description=(
                    "the barcoded ORF does not resolve to a gene of the current R64 "
                    "annotation (retired ORF, or a non-gene feature), so no gene entity "
                    "exists to key the record to"
                ),
                n_records=len(library.dropped_retired) * n_kept_compounds,
                items=library.dropped_retired,
            )
        )
        rules.append(
            DropRule(
                rule="orf_is_a_legacy_spelling_of_another_library_orf",
                scope="library_row",
                description=(
                    "the ORF resolves to an ORF the SAME library also carries under its "
                    "current name; remapping would merge two physically distinct "
                    "barcoded strains into one record key, so the legacy row is dropped"
                ),
                n_records=len(library.dropped_legacy_duplicate) * n_kept_compounds,
                items=library.dropped_legacy_duplicate,
            )
        )
        log.info(
            "Vanacloig: %d compounds kept (%d vehicle, %d unidentified dropped); "
            "%d library rows kept (%d non-ORF, %d all-NaN, %d background, %d retired, "
            "%d legacy duplicates dropped)",
            n_kept_compounds,
            len(vehicle),
            len(unidentified),
            n_rows,
            n_non_orf,
            n_all_nan,
            n_background,
            len(library.dropped_retired),
            len(library.dropped_legacy_duplicate),
        )

        # --- normalization ----------------------------------------------------- #
        # The library size is a property of the SEQUENCED SAMPLE, so it is summed over
        # EVERY released barcode (NaN = a QC-dropped barcode contributing no reads)
        # BEFORE any retention rule is applied. Summing over the retained rows instead
        # would make every stored value depend on this loader's gene-name policy.
        col_idx = {column: i for i, column in enumerate(sample_cols)}
        library_sizes = np.nansum(df[sample_cols].to_numpy(dtype=np.float64), axis=0)
        kept = df.loc[keep].reset_index(drop=True)
        counts = kept[sample_cols].to_numpy(dtype=np.float64)
        cpm = counts / library_sizes * 1e6
        pooled_log = np.log2(
            cpm[:, [col_idx[c] for c in controls]].mean(axis=1) + CPM_PRIOR
        )
        batch_log = {
            batch: np.log2(cpm[:, [col_idx[c] for c in cols]].mean(axis=1) + CPM_PRIOR)
            for batch, cols in control_by_batch.items()
        }

        publication = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_all_zero_cells = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for compound in tqdm(kept_compounds, desc="Vanacloig compounds"):
                cols = compound_cols[compound]
                if len(cols) != N_REPLICATES.value:
                    raise RuntimeError(
                        f"{compound}: {len(cols)} replicate columns, expected "
                        f"{N_REPLICATES.value}"
                    )
                indices = [col_idx[c] for c in cols]
                if compound in UNPAIRED_COMPOUND_TOKENS:
                    control_log = np.repeat(pooled_log[:, None], len(cols), axis=1)
                else:
                    control_log = np.column_stack(
                        [
                            batch_log[_SAMPLE_RE.match(c).group("batch")]  # type: ignore[union-attr]  # every column matched above
                            for c in cols
                        ]
                    )
                log_rep = np.log2(cpm[:, indices] + CPM_PRIOR) - control_log
                response = log_rep.mean(axis=1)
                sd = log_rep.std(axis=1, ddof=1)
                all_zero = (counts[:, indices] == 0).all(axis=1)
                n_all_zero_cells += int(all_zero.sum())
                environment = self._environment(compound)
                reference = self._reference(compound)
                for row in range(n_rows):
                    if all_zero[row]:
                        continue
                    experiment = EnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=self._genotype(
                            library.systematic[row],
                            library.common[row],
                            library.barcode[row],
                        ),
                        environment=environment,
                        phenotype=self._phenotype(
                            compound, float(response[row]), float(sd[row])
                        ),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(experiment, reference, publication, itxn),
                    )
                    idx += 1
        env.close()
        interned_env.close()

        rules.append(
            DropRule(
                rule="all_three_replicate_counts_are_zero",
                scope="cell",
                description=(
                    "every replicate of this (strain, compound) cell has a raw count of "
                    "0, so no abundance was measured; the CPM pseudocount would turn a "
                    "below-detection observation into a finite log2 value whose sample "
                    "SD is exactly 0, i.e. fabricated infinite precision"
                ),
                n_records=n_all_zero_cells,
                items=[],
            )
        )
        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=idx,
            dropped_records=source_records - idx,
            rules=rules,
        )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        accounted = sum(rule.n_records for rule in rules)
        if accounted != drop_log.dropped_records:
            raise RuntimeError(
                f"drop accounting mismatch: rules total {accounted}, "
                f"{drop_log.dropped_records} records missing from the build"
            )
        log.info("Wrote %d Vanacloig environment-response experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/env_chemgen_vanacloig2022")
    dataset = EnvChemgenVanacloig2022Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])
    print(
        json.dumps(
            json.loads(
                Path(osp.join(root, "preprocess/dropped_records.json")).read_text()
            )["rules"],
            indent=2,
        )[:2000]
    )


if __name__ == "__main__":
    main()
