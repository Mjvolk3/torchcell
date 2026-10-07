# torchcell/datasets/ecoli/mutalik2020
# [[torchcell.datasets.ecoli.mutalik2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/mutalik2020
# Test file: tests/torchcell/datasets/ecoli/test_mutalik2020.py
"""Mutalik 2020 phage-resistance RB-TnSeq screen: raw mirror, sourcing, loader.

Mutalik et al. 2020 (PLoS Biology, doi:10.1371/journal.pbio.3000877) challenged the
*E. coli* K-12 BW25113 RB-TnSeq library (Wetmore 2015's KEIO_ML9) with 14 double-stranded
DNA phages at a range of multiplicities of infection, in planktonic and solid-agar pooled
competitive-growth assays, and read strain abundance out by Bar-seq. The gene-level
readout is "the normalized log2 change in the abundance of mutants in that gene".

The module carries three layers. It pins and deposits the raw artifacts, binds every
statistical value to a verbatim quote in a sha256-pinned mirror, parses the experiment
(environment) axis, and registers :class:`PhageRbTnseqMutalik2020Dataset`, one
``BacterialEnvironmentResponseExperiment`` per (gene, experiment). The dataset class
waited on ``PhagePerturbation``, the typed environment leaf for a bacteriophage: a virion
is none of the other three leaves (no InChIKey, not a scalar physical factor, not a
peptide / protein / antibody / toxin), and the dose is a multiplicity of infection, which
``ConcentrationUnit`` and ``DoseBasis`` cannot express. That leaf now exists, so every
record's environment names its phage and its dose and the 68 challenges keep 68
environment identities.

SUPERSET DECISION (plan checklist item 7). These experiments are NOT in the Fitness
Browser compendium, so the row is loaded on its own experiments and nothing is
de-duplicated away. Measured 2026-10-07: the July 2026 Fitness Browser archive
(figshare 10.6084/m9.figshare.32865896, ``db.StrainFitness.Keio``) carries 168 experiment
columns for orgId ``Keio``, from sets 1, 2, 5, 6 and 52, and none from Mutalik's sets 16,
19, 28 or 30; the February 2024 archive carries the same five sets; and Price 2018's
``expsUsed`` for ``Keio`` names only ``Keio_ML9_set1``, ``set2`` and ``set6``. Checked by
experiment name against the 162 successful ``Keio`` samples of the Price 2018 compendium
(the set that subsumes Wetmore 2015, rank 2): the intersection with these 99 experiments
is empty. The library is shared with Wetmore 2015 and Price 2018 (one pool, KEIO_ML9)
while the experiments are disjoint, which is exactly the case the superset rule is for.

WHAT THE RECORDS ARE. One record per (gene, experiment) over the 68 phage assays and 10
no-phage controls the paper's own ``Keio_exps_used.tab`` names: **286,344 records over
3,697 genes and 78 experiments** (measured, see ``EXPECTED_RECORDS``). The phenotype is
``EnvironmentResponsePhenotype`` (``measurement_type=log2_ratio``,
``assay_type=pooled_competitive_growth_barcode``) under
``BacterialEnvironmentResponseExperiment``, NOT ``FitnessPhenotype``: the value is a
signed log2 ratio, and 106,536 of the 250,960 released S1 Table cells (42.45%) are
negative, which ``FitnessPhenotype.validate_fitness`` would clamp to 0.0. The genotype is
one ``TransposonInsertionPerturbation`` per gene, and the environment of a challenge
carries exactly one ``PhagePerturbation``.

WHAT IS NOT STORED. The release carries a t-like statistic per (gene, experiment)
(``fit_t.tab``) beside the fitness and the estimated standard error, and the paper's hit
filter uses it ("fit >= 5; t >= 5; standard error = fit/t <= 2"). It is neither the
measurement nor an uncertainty, and ``EnvironmentResponsePhenotype`` has no slot for a
test statistic, so the file is not among the members the loader extracts. The release's
own per-experiment usability flag ``u`` is FALSE for 68 of the 78 kept experiments, which
is not a quality finding about this build: the paper states that under the strong
positive selection of a phage assay "our standard quality metrics reported earlier [64]
were not suitable", which is why the selection rule is the paper's own
``Keio_exps_used.tab`` and not ``u``. The flag is reported in ``assay_ledger.json``.

THE ASSAY FORMAT IS PART OF THE ENVIRONMENT, in three places a plain ``Environment`` can
hold (``BacterialEnvironmentResponseExperiment.environment`` is ``Environment``, not
``CultureEnvironment``, so a vessel cannot be stored):

1. the ``Media`` object and its ``state`` -- ``MUTALIK2020_LB_SM_BUFFER`` (liquid) for the
   58 planktonic challenges and the 8 liquid controls, ``MUTALIK2020_LB_AGAR_KAN``
   (solid) for the 10 solid-agar challenges and the 1 solid control, ``MUTALIK2020_LB``
   (liquid) for the one plain-LB control;
2. ``duration_hours`` -- 8.0 for the planktonic format ("for 8 hr"), a typed
   ``ProvenanceGap`` for the solid format, whose incubation the Methods state only as
   "overnight";
3. the phenotype's ``units`` and ``screen_id`` (``Keio:<expName>``), which name the format
   and keep two assays of the same phage at the same MOI in different formats distinct.

IDENTIFIER FINDING (plan checklist items 3 and 4). The assayed strain is BW25113 and the
released identifiers are MG1655 b-numbers on a 4,639,675 bp single-scaffold reference
(the ``g/Keio/genome.fna`` of the figshare release; U00096.3 is 4,641,652 bp and
CP009273.1 is 4,631,469 bp), so the fitness tables are BW25113 biology labeled in the
MG1655 namespace. The released fitness table is itself BW25113-consistent: ``araA``
(b0062), ``araB`` (b0063), ``rhaA`` (b3903) and ``rhaB`` (b3904), which BW25113 lacks,
carry no row, while ``lacZ``, ``hsdR`` and ``rph``, whose lesions leave the gene in
place, do. So the records pin BW25113 and remap every identifier through the ECK
accession the two GenBank annotations share (``eck_route``), which is a DERIVED mapping
and is reported rather than applied silently (plan D9). ``audit_identifiers`` scores all
three routes: the ECK join resolves 3,697 of the 3,716 genes that carry a value
(0.9949), the b-numbers resolve against the deposited MG1655 annotation at 0.998 (a
route that would misstate the strain) and the FEBA gene symbols resolve against BW25113
at 0.961 (the weakest). Every record says so on its own perturbation:
``identifier_mapping=DerivedIdentifierMapping(source_identifier=<b-number>,
route="eck_crosswalk")``. A gene with no one-to-one ECK pair has no storable BW25113
identifier, so it is DROPPED and counted (``DROP_NO_ECK_PAIR``, 19 genes / 1,471 cells),
the same rule the Price 2018 loader applies to the same release.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import os.path as osp
import re
import shutil
import tarfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import openpyxl
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    AssayType,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialReferenceStrain,
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    DerivedIdentifierMapping,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    PhagePerturbation,
    Publication,
    Temperature,
    TransposonInsertionPerturbation,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    eck_crosswalk,
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
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import (
    EckPair,
    EcoliK12BW25113Genome,
    EcoliK12Genome,
    EcoliK12MG1655Genome,
    EcoliK12StrainName,
)
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
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "mutalikHighthroughputMappingPhage2020"
PAPER_DOI = "10.1371/journal.pbio.3000877"
PAPER_TITLE = "High-throughput mapping of the phage resistance landscape in E. coli"
PMC_PREFIX = "PMC7553319.1"
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "c7aab1a1c4384a37f1f75f6ecafe2f538fc6ff22c7c39aceeebe988aba2e6cb5"

#: S12 Table in the literature mirror: the strain and phage inventory, which is where the
#: per-phage genome accession is released. It is NOT a raw-mirror artifact: the loader
#: reads no value out of it at build time, the eleven accessions are constants in
#: :data:`PHAGE_S12_ROWS`, each bound to its own row of these pinned bytes.
S12_TABLE_MD = "si/si20.xlsx"
S12_TABLE_SHA256 = "f3e9a7977dbfbfbfba23e1adb13f75c440a4c58ae531aeedc975ab71e29cce8a"

#: The method paper the RB-TnSeq fitness statistic defers to (mirrored, rank 2 of the
#: fifty), from which the replicate structure of a gene fitness value is sourced.
WETMORE_KEY = "wetmoreRapidQuantificationMutant2015"
WETMORE_PAPER_MD = "paper.md"
WETMORE_PAPER_MD_SHA256 = (
    "ca3e7ef27a22a2a28e52ccbe5fdbb60890fea93d03b18b46e1d2b77b033b3cdb"
)

#: The strain the library was built in. The released identifiers are MG1655 b-numbers;
#: see the module docstring's identifier finding.
REFERENCE_STRAIN: BacterialReferenceStrain = "BW25113"

RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
RETRIEVED_AT = "2026-10-07"

#: The 14 phages of the panel, as the released metadata spells them.
PHAGES: tuple[str, ...] = (
    "186",
    "CEV1",
    "CEV2",
    "LZ4",
    "N4",
    "P1",
    "P2",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "lambda cI857",
)


class RawArtifact(BaseModel):
    """One raw-mirror file: where it came from, how to re-retrieve it, and its hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rel: str = Field(description="Path relative to the raw-mirror directory.")
    sha256: str
    bytes: int
    method: RetrievalMethod
    retriever: str = Field(
        description="Registry key into torchcell.literature.retrieve"
    )
    params: dict[str, Any]
    source_url: str
    what: str = Field(description="What the file is, in one line.")


S1_TABLE_REL = "data/S1_Table_RB-TnSeq_K12.xlsx"
S13_TABLE_REL = "data/S13_Table_MOI.xlsx"
EXPS_USED_REL = "data/figshare/Keio_exps_used.tab"
FIGSHARE_README_REL = "data/figshare/README.txt"
TARBALL_REL = "data/figshare/RBTnSeq.tar.gz"

#: figshare's own md5 for the tarball (article 11413128 v1, file 20305338), which our
#: retrieval reproduced; the sha256 below is the canonical anchor.
TARBALL_FIGSHARE_MD5 = "8cd059e83ff7d6692255e37d028f2bde"

RAW_ARTIFACTS: tuple[RawArtifact, ...] = (
    RawArtifact(
        rel=S1_TABLE_REL,
        sha256="7db056cf673e0d3262222e41efef5e37373d42b04a88cdf04b4f27a887c54b4c",
        bytes=4126084,
        method=RetrievalMethod.pmc_cloud,
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": f"{PMC_PREFIX}/pbio.3000877.s009.xlsx"},
        source_url=pmc_cloud_url(f"{PMC_PREFIX}/pbio.3000877.s009.xlsx"),
        what="S1 Table: gene fitness per experiment for the K-12 RB-TnSeq screen "
        "(sheets 'planktonic assays1', 'planktonic assays2', 'Solid plate assays', "
        "'Keio_hits_fit5', 'Fig 2A data')",
    ),
    RawArtifact(
        rel=S13_TABLE_REL,
        sha256="86221e234567b2e87e07b9619f705ce7afcc47b4a82b617d03bf6d37238c6110",
        bytes=34539,
        method=RetrievalMethod.pmc_cloud,
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": f"{PMC_PREFIX}/pbio.3000877.s021.xlsx"},
        source_url=pmc_cloud_url(f"{PMC_PREFIX}/pbio.3000877.s021.xlsx"),
        what="S13 Table: plaque-forming units per ml, dilution and MOI per experiment "
        "(the Methods' designated MOI authority)",
    ),
    RawArtifact(
        rel=EXPS_USED_REL,
        sha256="f9e78846ef9a85e593d8ca201182beb5ddaf6ca1dbda69add440a1ad15fb0384",
        bytes=9982,
        method=RetrievalMethod.direct_url,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": "https://ndownloader.figshare.com/files/21662430"},
        source_url="https://ndownloader.figshare.com/files/21662430",
        what="which BW25113 experiments the paper's analysis used, with each one's "
        "time-zero set, gMean and maxFit",
    ),
    RawArtifact(
        rel=FIGSHARE_README_REL,
        sha256="ebf050b56ed1d6f99a12ffde3e049fb7fb7883b70d97fee9a62c3c4e0e5fdecd",
        bytes=922,
        method=RetrievalMethod.direct_url,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": "https://ndownloader.figshare.com/files/21662427"},
        source_url="https://ndownloader.figshare.com/files/21662427",
        what="the figshare deposit's own description of the tarball layout",
    ),
    RawArtifact(
        rel=TARBALL_REL,
        sha256="f760df837a6324a1f03c27321a9ba0c66c3f599a2e88ff1241b5e75fdee9d65c",
        bytes=915077703,
        method=RetrievalMethod.direct_url,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": "https://ndownloader.figshare.com/files/20305338"},
        source_url="https://ndownloader.figshare.com/files/20305338",
        what="the complete RB-TnSeq release the Data Availability statement names: the "
        "mapping reference and pool, and per analysis set the gene fitness, t-like "
        "statistic, estimated standard error, per-experiment metadata and quality",
    ),
)

#: Members of ``RBTnSeq.tar.gz`` the loader reads, each pinned by its own sha256, so a
#: re-packed archive cannot change a member silently. ``read_tarball_member`` verifies.
TARBALL_MEMBERS: dict[str, str] = {
    "g/Keio/genes.tab": (
        "66d6d07b0ad16779239592c0f27f423dffa7c6f427676bfb1619216d8532cc20"
    ),
    "g/Keio/genome.fna": (
        "a82899afe45debe75567d55dd7b0a7c6d77e5aae79fe6d45ba6f71bb87c91bb0"
    ),
    "html/Keio_ML9_set16_set19/exps": (
        "6c942aaf199ce02aa7db8e3dbf530aa145ab48babcde2637c3dcdf73d1484313"
    ),
    "html/Keio_ML9_set16_set19/fit_logratios.tab": (
        "aeddec415785baf9c8cb1352712f10752ad8b20d56dbb4a984049f5cd18c0786"
    ),
    "html/Keio_ML9_set16_set19/fit_t.tab": (
        "f44411c88e289265b15428a332b7c96d8f832e7d63094744b0ff0f35e41e4798"
    ),
    "html/Keio_ML9_set16_set19/fit_standard_error_obs.tab": (
        "6cdde7887b8853b165e14ca6ff4def11a39c5d0620224ed214b1d8882486bae8"
    ),
    "html/Keio_ML9_set16_set19/fit_quality.tab": (
        "6f71161f4570694b911d3541c59f6f599dd5e1d9c21434ce961314e9e56fada1"
    ),
    "html/Keio_ML9_set28_set29/exps": (
        "7109ab56408b9eec78d7dc37db54be5daa0eced7d4028936057510f889ec9bb3"
    ),
    "html/Keio_ML9_set28_set29/fit_logratios.tab": (
        "654e32cf692c2026d0d267432ec9e18c285e7cfd0bf79602f1e9da4c0388a8d7"
    ),
    "html/Keio_ML9_set28_set29/fit_t.tab": (
        "af7f09551f8c3e079969fe105ca35f995ccf584c0fbc769a2c76b621571449d5"
    ),
    "html/Keio_ML9_set28_set29/fit_standard_error_obs.tab": (
        "2596e321dd755870aee9a557700f3288957e5a9a08c966f6f3511bb40e290630"
    ),
    "html/Keio_ML9_set28_set29/fit_quality.tab": (
        "8cc9f56173c53e13424a31cc54abea1f31e468ce2ab1aa519e5a4f0788f43d48"
    ),
    "html/Keio_ML9_set30/exps": (
        "aa394390a64b9fd10cb2ec59bff038755045dfdaf18da91c663dd1d6fe7cd6dc"
    ),
    "html/Keio_ML9_set30/fit_logratios.tab": (
        "98bdbbd41908fbafff0f978da8ab90c1fb66a33206c9ec042d9fc490724215db"
    ),
    "html/Keio_ML9_set30/fit_t.tab": (
        "b4a2a8e34fbc998c72b0d945f0450b65bf81e8951fcfc696710f53fbb2e48c5d"
    ),
    "html/Keio_ML9_set30/fit_standard_error_obs.tab": (
        "50cf67f46e3e3e5662369855d53bdc08324287bb2f6ce50a78fd07af31b8b5e7"
    ),
    "html/Keio_ML9_set30/fit_quality.tab": (
        "fca1dd6441a869f1cb58f13a0d74c51e40f3bab0eeb978567a61424888305e27"
    ),
}

#: The analysis set each released fitness table belongs to, keyed by the ``setNN`` prefix
#: of an experiment name.
ANALYSIS_SETS: dict[str, str] = {
    "set16": "Keio_ML9_set16_set19",
    "set19": "Keio_ML9_set16_set19",
    "set28": "Keio_ML9_set28_set29",
    "set30": "Keio_ML9_set30",
}

#: The number of insertion strains behind the typical gene fitness value, sourced from
#: Wetmore 2015 Table 1 for this very library (see ``sourced_values()``).
MEDIAN_STRAINS_PER_GENE = 16


def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror lives under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/mutalikHighthroughputMappingPhage2020``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(rel: str) -> RawArtifact:
    """The pinned artifact at ``rel``."""
    for record in RAW_ARTIFACTS:
        if record.rel == rel:
            return record
    raise KeyError(f"{rel} is not a pinned raw artifact of {CITATION_KEY}")


# --------------------------------------------------------------------------- #
# Sourced values: every number bound to a verbatim quote in a pinned mirror
# --------------------------------------------------------------------------- #
def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in this paper's sha256-pinned OCR mirror."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=page,
        ),
    )


def _wetmore(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the Wetmore 2015 mirror (the deferral)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=WETMORE_PAPER_MD,
            citation_key=WETMORE_KEY,
            sha256=WETMORE_PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=page,
        ),
    )


def sourced_values() -> dict[str, SourcedValue]:
    """Every value this row needs from the literature, each with its verbatim quote.

    The keys are the record fields they justify. ``uncertainty_type`` and ``n_samples``
    are the two the modular-dataset rules single out: the released uncertainty is an
    estimated standard error of the gene fitness value (the paper's own filter writes it
    as fit over t), so it is ``UncertaintyType.standard_error`` and is used as-is, never
    divided again; the replicate unit behind it is the insertion strain, whose count the
    method paper gives for this exact library.
    """
    return {
        "reference_strain": _paper(
            "BW25113",
            "We used a previously constructed E. coli K-12 BW25113 RB-TnSeq library [64]",
            page="Results, 'Mapping genetic determinants of phage resistance'",
            note="reference 64 is Wetmore 2015, whose KEIO_ML9 pool this is",
        ),
        "strain_genotype": _paper(
            "K-12 lacI+rrnBT14 Delta(araB-D)567 Delta(rhaD-B)568 DeltalacZ4787(::rrnB-3) "
            "hsdR514 rph-1",
            "BW25113 (K-12 lacI+rrnBT14 $\\Delta$ (araB–D)567 Δ(rhaD–B)568 "
            "ΔlacZ4787(::rrnB-3) hsdR514 rph-1)",
            page="Methods, 'Bacterial strains and growth conditions'",
            note="the paper's own statement of the background; the deposited GenBank "
            "source feature states the same lesions",
        ),
        "library_size": _paper(
            {"insertions": 152018, "genes": 3728},
            "our RB-TnSeq library with 152,018 barcoded insertions in 3,728 genes",
            page="Results, 'RB-TnSeq identifies known receptors'",
        ),
        "experiment_counts": _paper(
            {"phage_assays": 68, "phages": 14, "no_phage_controls": 9},
            "In total, we performed 68 RB-TnSeq assays across 14 phages at varying "
            "multiplicity of infection (MOI) and 9 no-phage control assays (Methods)",
            page="Results, 'RB-TnSeq identifies known receptors'",
            note="the released Keio_exps_used.tab carries 68 phage assays and TEN "
            "no-phage controls (one of them plain LB without SM buffer); the text says "
            "nine, and read_experiment_axis reports what the file holds",
        ),
        "measurement_type": _paper(
            "log2_ratio",
            "which we define as the normalized log2 change in the abundance of mutants "
            "in that gene",
            page="Results, 'Mapping genetic determinants of phage resistance'",
            note="a signed log2 ratio, so the record is an EnvironmentResponsePhenotype "
            "and not a strictly positive FitnessPhenotype ratio",
        ),
        "gene_fitness_rule": _paper(
            "weighted average of the fitness of its strains",
            "The fitness value of each gene is the weighted average of the fitness of "
            "its strains.",
            page="Methods, 'Data processing and analysis of BarSeq reads'",
        ),
        "uncertainty_type": _paper(
            "standard_error",
            "we required that fit $\\ge 5 ; \\mathsf { t } \\ge 5 ;$ ; standard error "
            "$\\dot { \\bf \\varphi } = \\bf { f i t } / t \\le 2$",
            page="Methods, 'Data processing and analysis of BarSeq reads'",
            note="the uncertainty the release carries per (gene, experiment) is the "
            "estimated standard error of the gene fitness value "
            "(fit_standard_error_obs.tab), which the paper writes as fit over t; it is "
            "already an SE of the estimate, so UncertaintyType.standard_error and no "
            "division by sqrt(n)",
        ),
        "n_samples": _wetmore(
            MEDIAN_STRAINS_PER_GENE,
            "Median no. of strains per genec</td><td>16</td>",
            page="Table 1, column 'Escherichia coli BW25113' (library KEIO_ML9)",
            note="the deferral the RB-TnSeq statistic points at: one sample is an "
            "independent insertion strain, and footnote c restricts the count to "
            "'genes for which we report fitness estimates and only strains that were "
            "used to make those estimates'. This is the library median, not a "
            "per-record count; the per-record count is derivable from the deposited "
            "pool and strain tables, and SampleUnit has no insertion-strain member "
            "(see the PR body)",
        ),
        "moi_authority": _paper(
            S13_TABLE_REL,
            "Phage plaque-forming units/ ml and MOIs used in each experiment are "
            "listed in the S13 Table.",
            page="Methods, 'Bacterial strains and growth conditions'",
            note="the S1 Table column labels disagree with S13 for 11 experiments by a "
            "factor of 10 to 100; S13 is what the Methods designate",
        ),
        "medium": _paper(
            "LB",
            "we recovered a frozen aliquot of the E. coli K-12 RB-TnSeq library in "
            "lysogeny broth (LB [96]) to mid-log phase",
            page="Results, 'RB-TnSeq identifies known receptors'",
            note="the recipe is deferred to reference 96 (Bertani), which is not "
            "mirrored, so MEDIA_LIBRARY carries no Mutalik LB entry and the medium is "
            "an open gap rather than a borrowed recipe",
        ),
        "culture_liquid": _paper(
            {"vessel": "48-well microplate", "working_volume_ul": 700.0},
            "The mutant library experiments were grown in the wells of a 48-well "
            "microplate",
            page="Methods, 'Competitive growth experiments with RB-TnSeq library'",
        ),
        "culture_dilution": _paper(
            {"start_od600": 0.04, "medium_strength": "2X LB"},
            "we diluted the recovered mutant library stock to a starting OD600 of 0.04 "
            "in 2X LB media",
            page="Methods, 'Competitive growth experiments with RB-TnSeq library'",
            note="an equal volume of phage in dilution buffer is then added, so the "
            "assay medium is 1X LB plus SM buffer, which is what the released "
            "per-experiment metadata calls LB_plus_SM_buffer",
        ),
        "culture_solid": _paper(
            "LB agar supplemented with kanamycin",
            "then plated the mixture on LB agar supplemented with kanamycin plates and "
            "incubated at",
            page="Methods, 'Competitive growth experiments with RB-TnSeq library'",
        ),
        "kanamycin": _paper(
            {"value": 50.0, "unit": "ug/mL"},
            "inoculated into $2 5 \\mathrm { m l }$ of medium supplemented with "
            "kanamycin $( 5 0 \\mu \\mathrm { g } / \\mathrm { m l } )$",
            page="Methods, 'Competitive growth experiments with RB-TnSeq library'",
        ),
        "bl21_arm": _paper(
            "BL21-DE3",
            "We mapped E. coli BL21 Dub-seq library to E. coli BL21-DE3 genome "
            "sequence [88]",
            page="Methods, 'Construction of BL21 Dub-seq library'",
            note="the BL21 arm (S8 and S9 Tables) is out of scope here: BL21-DE3 is not "
            "one of the three deposited assembly sets, so a BL21 record has nothing to "
            "pin",
        ),
    }


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def deposit_raw_mirror(
    *,
    sources: Mapping[str, str | Path],
    retrieved_at: str = RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from already-retrieved files and record how to re-get them.

    ``sources`` maps every ``RawArtifact.rel`` to the local path the retrieval produced.
    Idempotent by sha256: a mirror file that already carries the recorded hash is left
    alone, and one that differs raises rather than being overwritten. Every recorded
    retrieval re-runs as-is through ``torchcell.literature.retrieve``.
    """
    missing = [record.rel for record in RAW_ARTIFACTS if record.rel not in sources]
    if missing:
        raise ValueError(
            f"deposit_raw_mirror needs a source for every artifact: {missing}"
        )
    root = raw_mirror_dir(data_root)
    files: list[ArtifactRecord] = []
    for record in RAW_ARTIFACTS:
        src = Path(sources[record.rel])
        got = _sha256(src)
        if got != record.sha256:
            raise RuntimeError(
                f"{src} sha256 mismatch: got {got}, expected {record.sha256}"
            )
        dest = root / record.rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != record.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        files.append(
            ArtifactRecord(
                path=record.rel,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=record.sha256,
                source=record.source_url,
                retrieval=RetrievalRecord(
                    method=record.method,
                    source_url=record.source_url,
                    retriever=record.retriever,
                    params=dict(record.params),
                    sha256=record.sha256,
                    retrieved_at=retrieved_at,
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=files,
        si_data_sources=[
            pmc_cloud_url(f"{PMC_PREFIX}/pbio.3000877.s009.xlsx"),
            pmc_cloud_url(f"{PMC_PREFIX}/pbio.3000877.s021.xlsx"),
            "https://doi.org/10.6084/m9.figshare.11413128",
        ],
        si_expected=[
            "S5 Table + figshare 10.6084/m9.figshare.11838879.v2 (Dub-seq gain-of-"
            "function arm) -- a multicopy genomic fragment is not one of the five "
            "bacterial perturbation leaves, so the GOF arm is a later revision",
            "S4 Table + figshare 10.6084/m9.figshare.11859216.v3 (MG1655 CRISPRi arm) "
            "-- a different library and strain, loaded with the CRISPRi rows",
            "S8 and S9 Tables (BL21 RB-TnSeq and Dub-seq) -- BL21-DE3 is not a "
            "deposited assembly set",
            "SRA BioProject PRJNA645443 (raw reads) -- the mirror keeps the processed "
            "fitness tables the loader consumes, not the reads",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    log.info("Mutalik 2020 raw mirror written to %s (%d files)", root, len(files))
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


def read_tarball_member(member: str, data_root: str | None = None) -> bytes:
    """Read one pinned member of the deposited figshare tarball, verifying its sha256.

    The tarball is the artifact the Data Availability statement names and the only place
    the per-record t-like statistic and estimated standard error are released. A member
    whose bytes do not match ``TARBALL_MEMBERS`` raises: a re-packed archive is detected,
    never followed.
    """
    expected = TARBALL_MEMBERS[member]
    path = raw_mirror_dir(data_root) / TARBALL_REL
    with tarfile.open(path, mode="r:gz") as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise RuntimeError(f"{member} is not a file in {path}")
        payload = handle.read()
    got = hashlib.sha256(payload).hexdigest()
    if got != expected:
        raise RuntimeError(
            f"{member} sha256 mismatch in {path}: got {got}, expected {expected}"
        )
    return payload


# --------------------------------------------------------------------------- #
# The experiment (environment) axis
# --------------------------------------------------------------------------- #
#: The experiment descriptions that are a no-phage control rather than a challenge.
CONTROL_DESCRIPTIONS = frozenset(
    {"LB_plus_SM_buffer", "LB", "No phage control ML9a", "NoPhageControl"}
)
TIME_ZERO_DESCRIPTION = "Time0"

AssayKind = Literal["time_zero", "no_phage_control", "phage"]

_PHAGE_IN_DESCRIPTION = re.compile(
    r"with\s*(?P<liquid>[A-Za-z0-9]+)_phage\s*(?P<liquid_moi>[0-9.eE-]+)\s*MOI"
    r"|^(?P<solid>[A-Za-z0-9]+)_phage_(?P<solid_moi>[0-9.eE-]+)_MOI$"
    r"|^(?P<p2>P2)\s+dilution\s+10-(?P<p2_exp>\d+)$"
)
_MOI_FORMULA = re.compile(r"=G(?P<row>\d+)\*J(?P=row)/I(?P=row)$")
_DILUTION_FORMULA = re.compile(r"=J(?P<row>\d+)\*(?P<factor>[0-9.]+)$")
#: The two constants the S13 Table's MOI formula divides by, as the sheet writes them.
_PHAGE_VOLUME_FORMULA = "=F{row}*0.35"
_ONE_OD_FORMULA = "=8*10^8"
_CELL_COUNT_FORMULA = "=0.04*0.35*8*10^8"
_ASSAY_VOLUME_ML = 0.35
_ONE_OD_CFU_PER_ML = 8e8
_ASSAY_OD600 = 0.04


class MoiRow(BaseModel):
    """One S13 Table row: the dose of one experiment, recomputed from its own inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exp_name: str
    library: str
    phage: str
    description: str
    pfu_per_ml: float
    dilution: float
    moi: float


def read_moi_workbook(path: str | Path) -> dict[str, MoiRow]:
    """The S13 Table MOI of every RB-TnSeq BW25113 experiment, keyed by experiment name.

    The sheet releases the MOI as a formula over its own inputs rather than as a number,
    so the formulas are checked against the forms the sheet uses and then evaluated:
    ``MOI = pfu/ml * 0.35 mL * dilution / (0.04 OD * 0.35 mL * 8e8 cfu/mL)``. A dilution
    cell that references the row above is resolved through the chain. A formula of any
    other shape raises, so a changed sheet is detected instead of mis-evaluated.
    """
    workbook = openpyxl.load_workbook(path, read_only=True)
    sheet = workbook["MOI_used_runs"]
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))
    dilutions: dict[int, float | None] = {}
    table: dict[str, MoiRow] = {}
    for index, values in enumerate(rows, start=2):
        record = dict(zip(header, values, strict=True))
        raw_dilution = record["phage dilution"]
        if isinstance(raw_dilution, str) and raw_dilution.startswith("="):
            match = _DILUTION_FORMULA.match(raw_dilution.replace(" ", ""))
            if match is None:
                raise ValueError(f"row {index}: unreadable dilution {raw_dilution!r}")
            previous = dilutions[int(match.group("row"))]
            if previous is None:
                raise ValueError(
                    f"row {index}: dilution chain starts from a blank cell"
                )
            dilution = previous * float(match.group("factor"))
        elif raw_dilution in (None, ""):
            dilution = None
        else:
            dilution = float(raw_dilution)
        dilutions[index] = dilution
        library = str(record["Library/assay"] or "").strip()
        pfu = record["pfu/ml"]
        if (
            not library.startswith("RBTnSeq-BW25113")
            or dilution is None
            or pfu in (None, "")
        ):
            continue
        for column, expected in (
            ("phage count for 350 ul", _PHAGE_VOLUME_FORMULA.format(row=index)),
            ("1 OD cfu/ml", _ONE_OD_FORMULA),
            ("Cell count at od 0.04, for 350 ul", _CELL_COUNT_FORMULA),
        ):
            if str(record[column]).replace(" ", "") != expected:
                raise ValueError(
                    f"row {index}: {column} is {record[column]!r}, not {expected!r}"
                )
        if _MOI_FORMULA.match(str(record["MOI"]).replace(" ", "")) is None:
            raise ValueError(f"row {index}: MOI is {record['MOI']!r}, not =G*J/I")
        pfu_per_ml = float(pfu)
        moi = (pfu_per_ml * _ASSAY_VOLUME_ML * dilution) / (
            _ASSAY_OD600 * _ASSAY_VOLUME_ML * _ONE_OD_CFU_PER_ML
        )
        exp_name = str(record["expName"])
        table[exp_name] = MoiRow(
            exp_name=exp_name,
            library=library,
            phage=str(record["Phage"]).strip(),
            description=str(record["expDescription"]),
            pfu_per_ml=pfu_per_ml,
            dilution=dilution,
            moi=moi,
        )
    return table


def read_moi_table(data_root: str | None = None) -> dict[str, MoiRow]:
    """:func:`read_moi_workbook` on the S13 Table of the deposited raw mirror."""
    return read_moi_workbook(raw_mirror_dir(data_root) / S13_TABLE_REL)


class Assay(BaseModel):
    """One experiment of the released analysis: its kind, its phage and its dose."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exp_name: str
    set_name: str
    analysis_set: str
    description: str
    kind: AssayKind
    phage: str | None = None
    moi: float | None = None
    moi_source: Literal["s13_table", "experiment_description"] | None = None
    time_zero_set: str
    g_mean: float
    max_fit: float


class ExperimentAxis(BaseModel):
    """The environment axis of the row, as the released experiment list states it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assays: tuple[Assay, ...]

    @property
    def time_zero(self) -> tuple[Assay, ...]:
        """The start samples every log ratio is taken against."""
        return tuple(a for a in self.assays if a.kind == "time_zero")

    @property
    def controls(self) -> tuple[Assay, ...]:
        """The no-phage control assays."""
        return tuple(a for a in self.assays if a.kind == "no_phage_control")

    @property
    def challenges(self) -> tuple[Assay, ...]:
        """The phage challenges."""
        return tuple(a for a in self.assays if a.kind == "phage")

    @property
    def phages(self) -> tuple[str, ...]:
        """The distinct phages challenged, sorted."""
        return tuple(sorted({a.phage for a in self.challenges if a.phage is not None}))

    @property
    def counts(self) -> dict[str, int]:
        """How many experiments of each kind, and how many distinct phages."""
        return {
            "time_zero": len(self.time_zero),
            "no_phage_control": len(self.controls),
            "phage": len(self.challenges),
            "phages": len(self.phages),
        }


def _assay_kind(description: str) -> AssayKind:
    """Which kind of experiment a released description names."""
    stripped = description.strip()
    if stripped == TIME_ZERO_DESCRIPTION:
        return "time_zero"
    if stripped in CONTROL_DESCRIPTIONS:
        return "no_phage_control"
    return "phage"


def _phage_and_moi_from_description(description: str) -> tuple[str, float]:
    """The phage and the dose a challenge's own description states."""
    match = _PHAGE_IN_DESCRIPTION.search(description.strip())
    if match is None:
        raise ValueError(f"unreadable phage challenge description: {description!r}")
    groups = match.groupdict()
    if groups["liquid"] is not None:
        return groups["liquid"], float(groups["liquid_moi"])
    if groups["solid"] is not None:
        return groups["solid"], float(groups["solid_moi"])
    return groups["p2"], 10.0 ** -int(groups["p2_exp"])


def parse_experiment_axis(
    path: str | Path, moi_table: Mapping[str, MoiRow]
) -> ExperimentAxis:
    """Parse the released experiment list into the row's environment axis.

    The dose of a challenge comes from the S13 Table, which the Methods designate; an
    experiment S13 does not carry falls back to the MOI its own description states, and
    says so in ``moi_source``. The phage name is taken from S13 where it has the row,
    because the descriptions spell the same phage several ways (``CI1857``,
    ``lambda1857``, ``I86``).
    """
    assays: list[Assay] = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            exp_name = row["expName"]
            set_match = re.match(r"(set\d+)", exp_name)
            if set_match is None:
                raise ValueError(f"unreadable experiment name: {exp_name!r}")
            set_name = set_match.group(1)
            kind = _assay_kind(row["expDescription"])
            phage: str | None = None
            moi: float | None = None
            source: Literal["s13_table", "experiment_description"] | None = None
            if kind == "phage":
                if exp_name in moi_table:
                    phage = moi_table[exp_name].phage
                    moi = moi_table[exp_name].moi
                    source = "s13_table"
                else:
                    phage, moi = _phage_and_moi_from_description(row["expDescription"])
                    source = "experiment_description"
            assays.append(
                Assay(
                    exp_name=exp_name,
                    set_name=set_name,
                    analysis_set=ANALYSIS_SETS[set_name],
                    description=row["expDescription"],
                    kind=kind,
                    phage=phage,
                    moi=moi,
                    moi_source=source,
                    time_zero_set=row["t0set"],
                    g_mean=float(row["gMean"]),
                    max_fit=float(row["maxFit"]),
                )
            )
    return ExperimentAxis(assays=tuple(assays))


def read_experiment_axis(data_root: str | None = None) -> ExperimentAxis:
    """:func:`parse_experiment_axis` on the experiment list of the deposited raw mirror."""
    return parse_experiment_axis(
        raw_mirror_dir(data_root) / EXPS_USED_REL, read_moi_table(data_root)
    )


# --------------------------------------------------------------------------- #
# Identifier audit
# --------------------------------------------------------------------------- #
class EckRoute(BaseModel):
    """Released b-number to BW25113 locus tag through the one-to-one ECK synonym join.

    This is the route the records take: the strain assayed is BW25113, so the pinned
    assembly is BW25113's and every identifier is remapped through the ECK accession the
    two GenBank annotations share. It is a DERIVED mapping and is reported, never
    applied silently (plan D9). ``disagreeing`` lists the mapped genes whose b-number and
    ``BW25113_`` number differ, which is why no string surgery relates the namespaces.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pairs: int = Field(description="One-to-one ECK pairs between the two annotations.")
    numeric_disagreements: int = Field(
        description="Pairs, across the whole join, whose two numbers differ."
    )
    requested: int = Field(description="Distinct released identifiers offered.")
    mapped: int
    unmapped: tuple[str, ...]
    disagreeing: tuple[tuple[str, str], ...]

    @property
    def fraction(self) -> float:
        """Share of the offered identifiers the ECK join resolves."""
        return self.mapped / self.requested


class IdentifierAudit(BaseModel):
    """How the released identifiers resolve against each deposited K-12 annotation.

    ``b_numbers_vs_mg1655`` reconciles the released ``sysName`` column against the
    MG1655 annotation the identifiers belong to, and ``symbols_vs_bw25113`` reconciles
    the released gene symbols against the annotation of the strain that was actually
    assayed; both are reported because they bound the two naive routes. ``eck_route`` is
    the route the loader takes, and it is the best of the three.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    genes: int = Field(description="Rows of the released FEBA gene table.")
    b_numbers_vs_mg1655: LocusTagReconciliation
    symbols_vs_bw25113: LocusTagReconciliation
    eck_route: EckRoute


def read_feba_gene_table(data_root: str | None = None) -> list[dict[str, str]]:
    """The released FEBA gene table: b-number, symbol, type and coordinates per gene."""
    payload = read_tarball_member("g/Keio/genes.tab", data_root)
    reader = csv.DictReader(io.StringIO(payload.decode()), delimiter="\t")
    return [{key: (value or "") for key, value in row.items()} for row in reader]


def measured_gene_ids(data_root: str | None = None) -> set[str]:
    """The released identifiers that actually carry a fitness value, over all three sets."""
    import pandas as pd

    measured: set[str] = set()
    for analysis_set in sorted(set(ANALYSIS_SETS.values())):
        payload = read_tarball_member(
            f"html/{analysis_set}/fit_logratios.tab", data_root
        )
        frame = pd.read_csv(
            io.StringIO(payload.decode()), sep="\t", dtype={"sysName": str}
        )
        measured |= set(frame["sysName"])
    return measured


def eck_mapping(
    mg1655_genome: EcoliK12MG1655Genome, bw25113_genome: EcoliK12BW25113Genome
) -> dict[str, EckPair]:
    """Each MG1655 b-number to the ECK pair that carries its BW25113 locus tag."""
    crosswalk = eck_crosswalk(mg1655_genome, bw25113_genome)
    return {pair.mg1655: pair for pair in crosswalk.pairs}


def eck_route(
    mg1655_genome: EcoliK12MG1655Genome,
    bw25113_genome: EcoliK12BW25113Genome,
    b_numbers: Iterable[str],
) -> EckRoute:
    """Map the released identifiers onto BW25113 locus tags and report what happened."""
    crosswalk = eck_crosswalk(mg1655_genome, bw25113_genome)
    by_b = {pair.mg1655: pair for pair in crosswalk.pairs}
    requested = sorted(set(b_numbers))
    mapped = [name for name in requested if name in by_b]
    return EckRoute(
        pairs=len(crosswalk.pairs),
        numeric_disagreements=len(crosswalk.numeric_disagreements),
        requested=len(requested),
        mapped=len(mapped),
        unmapped=tuple(name for name in requested if name not in by_b),
        disagreeing=tuple(
            (by_b[name].mg1655, by_b[name].bw25113)
            for name in mapped
            if not by_b[name].numerics_agree
        ),
    )


def audit_identifiers(
    mg1655_genome: EcoliK12MG1655Genome,
    bw25113_genome: EcoliK12BW25113Genome,
    data_root: str | None = None,
    measured: Iterable[str] | None = None,
) -> IdentifierAudit:
    """Reconcile the released identifiers against both deposited K-12 annotations.

    ``measured`` is the identifier set the ECK route is scored on; it defaults to the
    released gene table, and the loader passes ``measured_gene_ids()`` so the score is
    over the genes that actually carry a value.
    """
    import pandas as pd

    genes = read_feba_gene_table(data_root)
    b_numbers = pd.Series([row["sysName"] for row in genes])
    symbols = pd.Series(
        [row["name"] if row["name"].strip() else row["sysName"] for row in genes]
    )
    _, b_report = reconcile_locus_tags(
        mg1655_genome, b_numbers, label=f"{CITATION_KEY}/b-numbers-vs-MG1655"
    )
    _, symbol_report = reconcile_locus_tags(
        bw25113_genome, symbols, label=f"{CITATION_KEY}/symbols-vs-BW25113"
    )
    return IdentifierAudit(
        genes=len(genes),
        b_numbers_vs_mg1655=b_report,
        symbols_vs_bw25113=symbol_report,
        eck_route=eck_route(
            mg1655_genome,
            bw25113_genome,
            list(b_numbers) if measured is None else measured,
        ),
    )


# --------------------------------------------------------------------------- #
# Per-phage identity: what the Methods and the S12 Table state about each phage
# --------------------------------------------------------------------------- #
_S12_PAGE = "S12 Table, 'Strains_Supp Table 12' (phage block)"
_PHAGE_METHODS = "Methods, 'Bacteriophages and propagation'"


def _s12(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to one row of the sha256-pinned S12 Table in the paper mirror.

    The quote is the row as the xlsx row rendering writes it (cells joined by ``" | "``,
    empty cells skipped), which is how every other xlsx-sourced value in the tier is cut.
    """
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=S12_TABLE_MD,
            citation_key=CITATION_KEY,
            sha256=S12_TABLE_SHA256,
            method="row rendering of the publisher xlsx (torchcell-library mirror)",
            page=_S12_PAGE,
        ),
    )


PHAGE_GENOME_TYPE = _paper(
    "dsDNA",
    "We sourced 14 diverse E. coli phages with dsDNA genomes, belonging to Myoviridae, "
    "Podoviridae, and Siphoviridae families (within the order Caudovirales)",
    page="Results, 'RB-TnSeq identifies known receptors'",
    note="stated for the panel as a whole, so every PhagePerturbation of this row "
    "carries genome_type='dsDNA'. The same sentence names the three families "
    "COLLECTIVELY and neither the paper nor the S12 Table assigns one to a phage, "
    "which is why `family` is a typed gap rather than a guess",
)

PHAGE_PROPAGATION = _paper(
    {"default": "E. coli BW25113", "P2": "E. coli C", "N4": "E. coli W3350"},
    "All phages except P2 phage and N4 phage were propagated on E. coli BW25113 strain. "
    "To propagate P2 phage and N4 phage, we used E. coli C and E. coli W3350 strains, "
    "respectively.",
    page=_PHAGE_METHODS,
)

#: The strain each phage stock was propagated on, verbatim, from ``PHAGE_PROPAGATION``.
DEFAULT_PROPAGATION_HOST = "E. coli BW25113"
PHAGE_PROPAGATION_HOSTS: dict[str, str] = {"P2": "E. coli C", "N4": "E. coli W3350"}

#: Each phage's genome accession, keyed by the name the S13 Table gives it (which is the
#: name ``read_experiment_axis`` stores). ``value`` is ``None`` for the three the S12
#: Table itself calls "Not determined", so the absence is the source's own statement.
#: Two names differ between the two tables and the mapping is recorded in the note: S13's
#: ``P1`` is S12's ``P1vir`` (the paper: "a strictly virulent strain of P1 phage (P1vir)")
#: and S13's ``lambda cI857`` is S12's ``lambda c1857``.
PHAGE_S12_ROWS: dict[str, SourcedValue] = {
    "T2": _s12("MH751506.1", "T2 | Calendar Lab stock, UC Berkeley | MH751506.1"),
    "T3": _s12("NC_003298.1", "T3 | Arkin Lab stock, UC Berkeley | NC_003298.1"),
    "T4": _s12(
        "AF158101.6",
        "T4 | Elizabeth Kutter Lab, The evergreen state College, Olympia | AF158101.6",
    ),
    "T5": _s12(
        "NC_005859.1", "T5 | E coli Genetic Stock center, CGSC#: 12144 | NC_005859.1"
    ),
    "T6": _s12("MH550421.1", "T6 | ATCC 11303-B6 | MH550421.1"),
    "T7": _s12(
        "NC_001604.1", "T7 | E coli Genetic Stock center, CGSC#: 12146 | NC_001604.1"
    ),
    "N4": _s12("EF056009.1", "N4 | Lucia B. Rothman-Denes, Univ Chicago | EF056009.1"),
    "CEV1": _s12(
        None,
        "CEV1 | Elizabeth Kutter Lab, The evergreen state College, Olympia | "
        "Not determined",
        note="the sheet states the genome was not determined, so genome_accession is "
        "None because the source says so, not because we did not look",
    ),
    "CEV2": _s12(
        None,
        "CEV2 | Elizabeth Kutter Lab, The evergreen state College, Olympia | "
        "Not determined",
        note="as CEV1",
    ),
    "LZ4": _s12(
        None,
        "LZ4 | Elizabeth Kutter Lab, The evergreen state College, Olympia | "
        "Not determined",
        note="as CEV1",
    ),
    "P1": _s12(
        "NC_005856.1",
        "P1vir | Jason Gill Lab Texas A&M University | NC_005856.1",
        note="S13 writes the phage 'P1'; S12 writes the stock 'P1vir', which the "
        "Methods name as the phage used ('a strictly virulent strain of P1 phage "
        "(P1vir)'), so this row is that phage",
    ),
    "P2": _s12("AF063097.1", "P2 | Calendar Lab stock, UC Berkeley | AF063097.1"),
    "186": _s12("NC_001317.1", "186 | Calendar Lab stock, UC Berkeley | NC_001317.1"),
    "lambda cI857": _s12(
        "NC_001416.1",
        "lambda c1857 | Calendar Lab stock, UC Berkeley | NC_001416.1",
        note="S13 and the paper write the allele 'cI857' (capital I); the S12 Table "
        "writes 'c1857' (digit one) for the same stock, so the row is matched on the "
        "phage rather than on the string",
    ),
}


def _family_gap() -> ProvenanceGap:
    """No source assigns a viral family to an individual phage of the panel."""
    return ProvenanceGap(
        field="family",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page="Results, 'RB-TnSeq identifies known receptors'; Fig 1 caption; "
            "S12 Table",
        ),
        note="the paper names Myoviridae, Podoviridae and Siphoviridae for the panel of "
        "14 collectively and never per phage, and the S12 Table releases only the "
        "source and the genome accession, so no family is written on a phage",
    )


def _moi_gap(exp_name: str) -> ProvenanceGap:
    """A challenge the two released dose sources both leave without an MOI."""
    return ProvenanceGap(
        field="multiplicity_of_infection",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{S13_TABLE_REL}",
            citation_key=CITATION_KEY,
            sha256=artifact(S13_TABLE_REL).sha256,
            method="S13 Table sheet 'MOI_used_runs', the dose authority the Methods "
            "designate, plus the experiment's own released description",
            page=f"no row for {exp_name} and no MOI in its description",
            retrieved=RETRIEVED_AT,
        ),
        note="a phage challenge always has a dose, so an unstated MOI is this typed gap "
        "and never a value borrowed from a neighbouring assay of the same phage",
    )


def phage_perturbation(
    assay: Assay, *, titer_pfu_per_ml: float | None
) -> PhagePerturbation:
    """The phage of one challenge, at the dose the released sources state.

    ``name`` is the S13 Table's spelling (the experiment descriptions spell the same
    phage several ways), ``genome_type`` is the panel-wide ``dsDNA``,
    ``genome_accession`` comes from this phage's S12 Table row and is ``None`` for the
    three the sheet calls "Not determined", and ``host_of_propagation`` is the strain the
    Methods name for it. ``family`` and ``ncbi_taxid`` are not stated per phage: the
    family is a typed gap (:func:`_family_gap`), and no source names a taxon id at all.
    An assay with no stated MOI carries :func:`_moi_gap` instead of a borrowed dose.
    """
    if assay.phage is None:
        raise ValueError(f"{assay.exp_name} is a {assay.kind}, not a phage challenge")
    gaps = [_family_gap()]
    if assay.moi is None:
        gaps.append(_moi_gap(assay.exp_name))
    accession = (
        PHAGE_S12_ROWS[assay.phage].value if assay.phage in PHAGE_S12_ROWS else None
    )
    return PhagePerturbation(
        name=assay.phage,
        genome_type=str(PHAGE_GENOME_TYPE.value),
        genome_accession=accession,
        host_of_propagation=PHAGE_PROPAGATION_HOSTS.get(
            assay.phage, DEFAULT_PROPAGATION_HOST
        ),
        multiplicity_of_infection=assay.moi,
        titer_pfu_per_ml=titer_pfu_per_ml,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# Media: the three the release names, each an LB-based object with no invented amounts
# --------------------------------------------------------------------------- #
_CULTURE_PAGE = "Methods, 'Competitive growth experiments with RB-TnSeq library'"

LB_RECIPE_DEFERRAL = _paper(
    "LB",
    "we recovered a frozen aliquot of the E. coli K-12 RB-TnSeq library in lysogeny "
    "broth (LB [96]) to mid-log phase",
    page="Results, 'RB-TnSeq identifies known receptors'",
    note="the recipe is deferred to the paper's reference 96 (Bertani), which is not "
    "mirrored, so the LB lines below carry NO amounts. Borrowing MEDIA_LIBRARY's LB "
    "(Miller, 10 g/L NaCl) or LB_LENNOX (5 g/L, which is what Wetmore 2015 and Price "
    "2018 state in their OWN media tables) would assert a formulation Mutalik never "
    "gave; base_medium='LB' is the library key these objects join on",
)

SM_BUFFER = _paper(
    "SM buffer (Teknova), supplemented with 10 mM calcium chloride and magnesium "
    "sulphate",
    "SM buffer was supplemented with $1 0 \\mathrm { m M }$ calcium chloride and "
    "magnesium sulphate (Sigma).",
    page=_PHAGE_METHODS,
    note="the vendor buffer's own composition is not stated, so it is one "
    "composition_deferred line; the two supplements are named with their buffer "
    "concentration, which the assay then dilutes (see CULTURE_DILUTION)",
)

SM_BUFFER_VENDOR = _paper(
    "Teknova",
    "a 10-fold serial dilution of each phage in SM buffer (Teknova)",
    page=_PHAGE_METHODS,
    note="the buffer the phages are diluted in is the same commercial preparation the "
    "titering step uses",
)

PLANKTONIC_DURATION = _paper(
    8.0,
    "We grew the microplates in Tecan Infinite F200 readers with orbital shaking and "
    "OD600 readings every $1 5 \\mathrm { { m i n } }$ for $8 \\mathrm { { h r } }$ .",
    page=_CULTURE_PAGE,
    note="the planktonic format's exposure time, in hours",
)

SOLID_INCUBATION = _paper(
    "overnight",
    "then plated the mixture on LB agar supplemented with kanamycin plates and "
    "incubated at $3 7 ~ ^ { \\circ } \\mathrm { C }$ overnight",
    page=_CULTURE_PAGE,
    note="the solid format's exposure is stated as a word, not a number of hours, which "
    "is why Environment.duration_hours is a typed gap for the solid assays",
)

#: The analysis set whose released per-experiment metadata the plate dose is cut from.
_SOLID_SET = "Keio_ML9_set30"

SOLID_KANAMYCIN_DOSE = SourcedValue(
    value={"value": 50.0, "unit": "ug/ml"},
    quote="Kan\t50\tug/ml",
    note="the plate dose. The Methods sentence that names the plates says only "
    "'supplemented with kanamycin'; the 50 ug/ml for the ASSAY plates is the released "
    "per-experiment metadata's own Condition_2 / Concentration_2 / Units_2 columns, "
    "which process() checks on every solid assay it keeps. The Methods' 50 ug/ml "
    "(sourced_values()['kanamycin']) is the library RECOVERY culture, a different step, "
    "so it is not the citation for this value",
    provenance=Provenance(
        source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{TARBALL_REL}:html/{_SOLID_SET}/exps",
        citation_key=CITATION_KEY,
        sha256=TARBALL_MEMBERS[f"html/{_SOLID_SET}/exps"],
        method="tab-delimited per-experiment metadata of the deposited figshare "
        "RB-TnSeq release, read through read_tarball_member",
        page=f"html/{_SOLID_SET}/exps, the solid-plate rows",
        retrieved=RETRIEVED_AT,
    ),
)

ASSAY_MIXTURE = _paper(
    {"library_ul": 350.0, "phage_ul": 350.0, "well_ul": 700.0},
    "The mutant library experiments were grown in the wells of a 48-well microplate "
    "${ 7 0 0 \\mu \\mathrm { l } }$ per well)",
    page=_CULTURE_PAGE,
    note="with the preceding sentence's 350 uL of 2X library and 350 uL of diluted "
    "phage, this is what makes the planktonic culture's phage titer computable from "
    "the S13 Table's stock titer and dilution (see titer_in_culture)",
)


def _lb_lines(provenance: SourcedValue) -> list[MediaComponent]:
    """The three LB ingredients, named with no amounts (the recipe is deferred)."""
    return [
        MediaComponent(
            compound=resolved_compound("tryptone"),
            role=MediaComponentRole.complex_ingredient,
            concentration=None,
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[provenance],
            note="LB ingredient; Mutalik states no amount and defers the recipe",
        ),
        MediaComponent(
            compound=resolved_compound("yeast extract"),
            role=MediaComponentRole.complex_ingredient,
            concentration=None,
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[provenance],
            note="LB ingredient; Mutalik states no amount and defers the recipe",
        ),
        MediaComponent(
            compound=resolved_compound("sodium chloride"),
            role=MediaComponentRole.bulk_salt,
            concentration=None,
            provenance=[provenance],
            note="LB ingredient; Mutalik states no amount, and the Miller and Lennox "
            "formulations differ in exactly this component",
        ),
    ]


MUTALIK2020_LB = Media(
    name="LB, formulation not stated (Mutalik 2020), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="LB",
    components=_lb_lines(LB_RECIPE_DEFERRAL),
    provenance=[LB_RECIPE_DEFERRAL],
)
"""Plain liquid LB: the one no-phage control the release runs without SM buffer."""

MUTALIK2020_LB_SM_BUFFER = Media(
    name="LB with SM buffer (Teknova) plus 10 mM CaCl2 and MgSO4, formulation not "
    "stated (Mutalik 2020), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="LB",
    components=[
        *_lb_lines(LB_RECIPE_DEFERRAL),
        MediaComponent(
            compound=Compound(name="SM buffer (Teknova)"),
            role=MediaComponentRole.buffer,
            concentration=None,
            definition=ComponentDefinition.composition_deferred,
            provenance=[SM_BUFFER_VENDOR, SM_BUFFER],
            note="the phage dilution buffer. A commercial preparation whose composition "
            "neither the paper nor a mirrored protocol states, so it stays one deferred "
            "line rather than being expanded into guessed salts",
        ),
        MediaComponent(
            compound=resolved_compound("calcium chloride"),
            role=MediaComponentRole.bulk_salt,
            concentration=None,
            provenance=[SM_BUFFER],
            note="required for phage adsorption. The source states 10 mM in the BUFFER; "
            "the assay mixes an equal volume of buffer-diluted phage into 2X LB, so the "
            "concentration in the culture is not what the source states and no value is "
            "written here",
        ),
        MediaComponent(
            compound=resolved_compound("magnesium sulfate"),
            role=MediaComponentRole.bulk_salt,
            concentration=None,
            provenance=[SM_BUFFER],
            note="as calcium chloride: 10 mM in the buffer, diluted into the culture by "
            "an amount the source does not state for the final medium",
        ),
    ],
    provenance=[LB_RECIPE_DEFERRAL, SM_BUFFER, ASSAY_MIXTURE],
)
"""The planktonic assay medium, which the release's own metadata calls ``LB_plus_SM_buffer``."""

MUTALIK2020_LB_AGAR_KAN = Media(
    name="LB agar with kanamycin 50 ug/mL, formulation not stated (Mutalik 2020), solid",
    state="solid",
    is_synthetic=False,
    base_medium="LB",
    components=[
        *_lb_lines(LB_RECIPE_DEFERRAL),
        MediaComponent(
            compound=resolved_compound("agar"),
            role=MediaComponentRole.gelling_agent,
            concentration=None,
            provenance=[SOLID_INCUBATION],
            note="the plates are named 'LB agar' with no percentage; the 0.7% figure the "
            "paper gives is the top-agar overlay of the titering step, a different "
            "preparation, so it is not copied here",
        ),
        MediaComponent(
            compound=resolved_compound("kanamycin"),
            role=MediaComponentRole.selection_agent,
            concentration=Concentration(value=50.0, unit=ConcentrationUnit.ug_per_ml),
            provenance=[SOLID_INCUBATION, SOLID_KANAMYCIN_DOSE],
            note="the plates are 'LB agar supplemented with kanamycin' and the dose is "
            "the released metadata's own Condition_2, which process() checks on every "
            "solid assay it keeps",
        ),
    ],
    provenance=[LB_RECIPE_DEFERRAL, SOLID_INCUBATION],
)
"""The solid-agar assay medium, which the release's own metadata calls ``LB_agar``."""

#: The released per-experiment ``Media`` label to the object a record stores. A label
#: outside this map has no MEDIA_LIBRARY base to join on, so its assay is DROPPED and
#: counted (``DROP_MEDIUM_NOT_IN_LIBRARY``) rather than given an invented medium.
MEDIA_BY_LABEL: dict[str, Media] = {
    "LB": MUTALIK2020_LB,
    "LB_plus_SM_buffer": MUTALIK2020_LB_SM_BUFFER,
    "LB_agar": MUTALIK2020_LB_AGAR_KAN,
}

#: The kanamycin dose the solid plates carry, as the release's metadata writes it.
SOLID_KANAMYCIN = (50.0, "ug/ml")


# --------------------------------------------------------------------------- #
# The released per-experiment metadata (the `exps` member of each analysis set)
# --------------------------------------------------------------------------- #
#: The release's ``Aerobic_v_Anaerobic`` values to the schema's oxygen regimes. A value
#: outside this is refused, never defaulted to aerobic.
AEROBICITY_BY_LABEL: dict[str, str] = {
    "Aerobic": "aerobic",
    "Anaerobic": "anaerobic",
    "Microaerobic": "microaerobic",
}

#: The release's ``Liquid v. solid`` values to the ``Media.state`` they must agree with.
STATE_BY_LABEL: dict[str, str] = {"Liquid": "liquid", "Solid": "solid"}


class ReleasedAssay(BaseModel):
    """One row of an analysis set's ``exps`` member: what the culture actually was.

    The environment of a record is built from THIS, not from the paper's prose: the
    release states the medium, the format, the oxygen regime, the temperature and the
    second condition per experiment, and the prose states them for the two formats in
    general. Where the two can disagree, the per-experiment row is the one that names
    this experiment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    exp_name: str
    set_name: str
    media_label: str
    state: str = Field(description="'liquid' or 'solid', from 'Liquid v. solid'")
    growth_method: str
    shaking: str
    aerobicity: str
    group: str
    mutant_library: str
    temperature_c: float
    condition_2: str | None
    concentration_2: float | None
    units_2: str | None
    dropped_by_release: bool = Field(
        description="the release's own ``Drop`` flag on this experiment"
    )


def read_released_assays(paths: Mapping[str, str | Path]) -> dict[str, ReleasedAssay]:
    """Every experiment's released metadata, keyed by the ``setNNITNNN`` experiment name.

    ``paths`` maps an analysis-set name to its ``exps`` file. The experiment name is the
    set's ``setNN`` suffix plus the row's ``Index``, which is how the fitness tables name
    their columns and how ``Keio_exps_used.tab`` names its rows; a name two sets both
    claim is refused rather than silently overwritten.
    """
    assays: dict[str, ReleasedAssay] = {}
    for analysis_set, path in paths.items():
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                set_name = str(row["SetName"])
                exp_name = f"{set_name.removeprefix('Keio_ML9_')}{row['Index']}"
                if exp_name in assays:
                    raise ValueError(f"{exp_name} is released by two analysis sets")
                state = STATE_BY_LABEL.get(str(row["Liquid v. solid"]))
                if state is None:
                    raise ValueError(
                        f"{exp_name}: unreadable format {row['Liquid v. solid']!r}"
                    )
                aerobicity = AEROBICITY_BY_LABEL.get(str(row["Aerobic_v_Anaerobic"]))
                if aerobicity is None:
                    raise ValueError(
                        f"{exp_name}: unreadable oxygen regime "
                        f"{row['Aerobic_v_Anaerobic']!r}"
                    )
                if not str(row["Temperature"]).strip():
                    raise ValueError(f"{exp_name}: the release states no temperature")
                condition_2 = str(row["Condition_2"]).strip() or None
                assays[exp_name] = ReleasedAssay(
                    exp_name=exp_name,
                    set_name=analysis_set,
                    media_label=str(row["Media"]).strip(),
                    state=state,
                    growth_method=str(row["Growth Method"]).strip(),
                    shaking=str(row["Shaking"]).strip(),
                    aerobicity=aerobicity,
                    group=str(row["Group"]).strip(),
                    mutant_library=str(row["Mutant Library"]).strip(),
                    temperature_c=float(row["Temperature"]),
                    condition_2=condition_2,
                    concentration_2=(
                        float(row["Concentration_2"])
                        if str(row["Concentration_2"]).strip()
                        else None
                    ),
                    units_2=str(row["Units_2"]).strip() or None,
                    dropped_by_release=str(row["Drop"]).strip().upper() == "TRUE",
                )
    return assays


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def titer_in_culture(row: MoiRow | None, state: str) -> float | None:
    """The phage titer of a planktonic culture at the start of the assay, in pfu/mL.

    The S13 Table releases the STOCK titer and the dilution; the Methods state that
    350 uL of the diluted phage goes into 350 uL of 2X library for 700 uL per well
    (``ASSAY_MIXTURE``), so the culture carries ``pfu/ml * dilution / 2``. This is the
    same arithmetic over the same cells that ``read_moi_workbook`` evaluates for the MOI.

    Returns ``None`` for the SOLID format, whose final volume the Methods never state
    (the planktonic paragraph gives the two 350 uL volumes; the solid paragraph says only
    "the mixture"), and ``None`` for a challenge with no S13 row.
    """
    if row is None or state != "liquid":
        return None
    return row.pfu_per_ml * row.dilution / 2.0


def _solid_duration_gap() -> ProvenanceGap:
    """The solid format's exposure is stated as "overnight", not as a number of hours."""
    return ProvenanceGap(
        field="duration_hours",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=_CULTURE_PAGE,
        ),
        note=f"the plates were 'incubated at 37 C {SOLID_INCUBATION.value}'; the "
        "planktonic format's 8 hr is stated as a number and is carried on "
        "duration_hours, so the two formats differ in this field as well as in the "
        "medium, which is part of what keeps them distinct environments",
    )


def environment(
    assay: Assay, released: ReleasedAssay, moi_row: MoiRow | None
) -> Environment:
    """The culture one experiment was run in, from its own released metadata.

    The medium comes from the released ``Media`` label through :data:`MEDIA_BY_LABEL`;
    the temperature, the oxygen regime and the format come from the released row; the
    exposure is the planktonic 8 hr or a typed gap for the solid plates. A challenge
    carries exactly one ``PhagePerturbation`` and a no-phage control carries NONE: the
    controls replaced the phage with plain dilution buffer ("We also set up control
    'no-phage' competitive mutant fitness assays wherein we replaced phages with simply
    the phage dilution buffer"), so a zero-MOI phage would assert a challenge that did
    not happen. The buffer itself is in the medium, which is what makes a control's
    environment the honest reference for the challenges run beside it.
    """
    media = MEDIA_BY_LABEL[released.media_label]
    if media.state != released.state:
        raise ValueError(
            f"{assay.exp_name}: the release calls the medium {released.media_label!r} "
            f"and the format {released.state!r}, but {media.name!r} is "
            f"{media.state!r}"
        )
    if media is MUTALIK2020_LB_AGAR_KAN:
        dose = (released.concentration_2, released.units_2)
        if released.condition_2 != "Kan" or dose != SOLID_KANAMYCIN:
            raise ValueError(
                f"{assay.exp_name}: {media.name!r} carries kanamycin at "
                f"{SOLID_KANAMYCIN}, but the release states "
                f"{released.condition_2!r} at {dose}"
            )
    elif released.condition_2 is not None:
        raise ValueError(
            f"{assay.exp_name}: the release adds {released.condition_2!r} to "
            f"{released.media_label!r}, which {media.name!r} does not carry"
        )
    perturbations: list[EnvironmentPerturbationType] = (
        [
            phage_perturbation(
                assay, titer_pfu_per_ml=titer_in_culture(moi_row, media.state)
            )
        ]
        if assay.kind == "phage"
        else []
    )
    return Environment(
        media=media,
        temperature=Temperature(value=released.temperature_c),
        perturbations=perturbations,
        aerobicity=released.aerobicity,
        duration_hours=(
            float(PLANKTONIC_DURATION.value) if media.state == "liquid" else None
        ),
        provenance_gaps=[] if media.state == "liquid" else [_solid_duration_gap()],
    )


# --------------------------------------------------------------------------- #
# Genes: the ECK route from the released b-numbers to BW25113 locus tags
# --------------------------------------------------------------------------- #
#: The route must place at least this share of the measured genes; it places 3,697 of
#: 3,716 (0.9949) on the deposited annotations. Below it the build stops rather than
#: dropping more genes quietly.
MIN_ECK_ROUTE_FRACTION = 0.99

#: The library's transposon, from the method paper this row's statistic defers to.
TRANSPOSON = _wetmore(
    "Tn5 transpososome",
    "<td>Transposon</td><td>Tn5 transpososome</td>",
    page="Table 1, row 'Transposon'",
    note="Mutalik states no transposon and defers the library to reference 64 "
    "(Wetmore 2015), whose Table 1 names it for KEIO_ML9",
)

#: The sub-pool the release's own per-experiment metadata names, verbatim. Wetmore calls
#: the pool KEIO_ML9; every experiment of this row is run on ``Keio_ML9a``, which is what
#: the strain identity carries (a pooled measurement is pool-relative).
MUTANT_LIBRARY = "Keio_ML9a"

PERTURBATION_DESCRIPTION = (
    "Tn5 transposon insertion disrupting a gene; gene-level call over the gene's "
    "insertion strains (no per-strain barcode or mapped site released). Locus tag "
    "DERIVED: the release names an MG1655 b-number, mapped to this BW25113 locus "
    "through its one-to-one ECK pair (eck_crosswalk)"
)


class GeneMapping(BaseModel):
    """One released b-number placed on its BW25113 locus through a one-to-one ECK pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    b_number: str
    eck: str
    locus_tag: str
    perturbed_gene_name: str
    numerics_agree: bool


class MappingReport(BaseModel):
    """The ECK route over the measured genes, with the reconciler's histograms."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_measured_genes: int
    n_mapped: int
    mapped_fraction: float
    min_fraction: float
    unmapped: tuple[str, ...]
    numeric_disagreements: tuple[tuple[str, str, str], ...] = Field(
        description="(b-number, BW25113 tag, ECK) of mapped pairs whose numbers differ"
    )
    n_symbol_names: int = Field(
        description="records whose perturbed_gene_name is the BW25113 gene symbol"
    )
    n_tag_names: int = Field(
        description="records whose perturbed_gene_name falls back to the locus tag"
    )
    reconcile_status_histogram: dict[str, int]
    reconcile_layer_histogram: dict[str, int]


def stored_gene_name(
    symbol: str | None, tag: str, resolve: Callable[[str], Any]
) -> str:
    """The locus's gene symbol when it resolves back to that locus, else the tag.

    The stored common name then always resolves to the stored locus tag, which is what
    the verifier's canonical-gene-name rule checks.
    """
    if symbol:
        resolution = resolve(symbol)
        if resolution.systematic_name == tag and resolution.status in (
            GeneNameStatus.CURRENT,
            GeneNameStatus.RENAMED,
        ):
            return symbol
    return tag


def map_measured_genes(
    b_numbers: Sequence[str],
    mg1655: EcoliK12MG1655Genome,
    bw25113: EcoliK12BW25113Genome,
    *,
    label: str,
) -> tuple[dict[str, GeneMapping], MappingReport]:
    """Place the measured b-numbers on BW25113 locus tags and report what happened.

    The route is the one-to-one ECK synonym join the two deposited GenBank annotations
    share (:func:`eck_mapping`). A b-number the join does not place has no storable
    BW25113 identifier, so it is left out and named in the report; the build stops if the
    route places less than :data:`MIN_ECK_ROUTE_FRACTION` of the genes. The placed genes'
    ECK ids are then run through ``reconcile_locus_tags`` on BW25113, which must return
    the crosswalk's tag for every one.
    """
    by_b = eck_mapping(mg1655, bw25113)
    requested = sorted(set(b_numbers))
    placed = [name for name in requested if name in by_b]
    fraction = len(placed) / len(requested)
    if fraction < MIN_ECK_ROUTE_FRACTION:
        raise ValueError(
            f"{label}: the ECK route places {len(placed)} of {len(requested)} measured "
            f"genes ({fraction:.4f}), below {MIN_ECK_ROUTE_FRACTION}"
        )
    import pandas as pd

    stored, reconciliation = reconcile_locus_tags(
        bw25113,
        pd.Series([by_b[name].eck for name in placed]),
        label=f"{label} ECK ids",
    )
    differing = [
        name
        for name, tag in zip(placed, stored, strict=True)
        if tag != by_b[name].bw25113
    ]
    if differing:
        raise ValueError(f"{label}: reconciler and crosswalk disagree on {differing}")
    loci = bw25113.genbank.loci
    mapping = {
        name: GeneMapping(
            b_number=name,
            eck=by_b[name].eck,
            locus_tag=by_b[name].bw25113,
            perturbed_gene_name=stored_gene_name(
                loci[by_b[name].bw25113].symbol,
                by_b[name].bw25113,
                bw25113.resolve_gene_name,
            ),
            numerics_agree=by_b[name].numerics_agree,
        )
        for name in placed
    }
    n_symbol = sum(1 for m in mapping.values() if m.perturbed_gene_name != m.locus_tag)
    report = MappingReport(
        n_measured_genes=len(requested),
        n_mapped=len(mapping),
        mapped_fraction=fraction,
        min_fraction=MIN_ECK_ROUTE_FRACTION,
        unmapped=tuple(name for name in requested if name not in by_b),
        numeric_disagreements=tuple(
            (m.b_number, m.locus_tag, m.eck)
            for m in mapping.values()
            if not m.numerics_agree
        ),
        n_symbol_names=n_symbol,
        n_tag_names=len(mapping) - n_symbol,
        reconcile_status_histogram={
            status.value: n for status, n in reconciliation.status_histogram.items()
        },
        reconcile_layer_histogram=dict(reconciliation.layer_histogram),
    )
    return mapping, report


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
N_SAMPLES_GAP = ProvenanceGap(
    field="n_samples",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=Provenance(
        source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{TARBALL_REL}:html/<set>/fit_logratios.tab",
        citation_key=CITATION_KEY,
        sha256=artifact(TARBALL_REL).sha256,
        method="the per-gene tables of the deposited figshare RB-TnSeq release carry no "
        "strain-count column",
        page="html/<analysis set>/fit_logratios.tab and fit_standard_error_obs.tab",
        retrieved=RETRIEVED_AT,
    ),
    resolve_with=Provenance(
        source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{TARBALL_REL}:html/<set>/strain_fit.tab",
        citation_key=CITATION_KEY,
        method="the per-strain table of the same release, which names each strain's gene",
        page="html/<analysis set>/strain_fit.tab with g/Keio/pool",
    ),
    note="a gene fitness value is 'the weighted average of the fitness of its strains', "
    f"so one sample is an independent insertion strain. The library MEDIAN is {MEDIAN_STRAINS_PER_GENE} "
    "(Wetmore 2015 Table 1 for KEIO_ML9, sourced_values()['n_samples']) and a median "
    "over genes is not a per-record count, so it is NOT stored as n_samples. The stored "
    "uncertainty is already an SE of the estimate, so no n is needed to derive it",
)
SAMPLE_UNIT_GAP = ProvenanceGap(
    field="sample_unit",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=N_SAMPLES_GAP.looked_in,
    resolve_with=N_SAMPLES_GAP.resolve_with,
    note="travels with n_samples: one unit would be an independent insertion strain in "
    "a pooled library, which is none of SampleUnit's members (colony, screen, "
    "biological_replicate, technical_replicate, pooled)",
)
#: ``screen_id`` prefix: FEBA experiment names are unique only within an organism, so the
#: organism id joins them, exactly as the Price 2018 loader does for the same release.
ORG_ID = "Keio"

_UNITS_STEM = (
    "normalized log2(gene mutant barcode abundance at the end of the assay / abundance "
    "in the time-zero start sample), the weighted average over the gene's insertion "
    "strains"
)
UNITS_PLANKTONIC = (
    f"{_UNITS_STEM}; 8 h planktonic 48-well culture in LB plus phage dilution buffer, "
    "a typical gene = 0"
)
UNITS_SOLID = (
    f"{_UNITS_STEM}; overnight solid-agar plate assay, colonies scraped after plating "
    "the adsorbed mixture, a typical gene = 0"
)
UNITS_REFERENCE = (
    "the typical gene of this experiment (gene fitness is normalized to 0)"
)


def build_genotype(mapping: GeneMapping) -> Genotype:
    """A Tn5 insertion in one gene of ``Keio_ML9a``, with its derived mapping typed."""
    return Genotype(
        perturbations=[
            TransposonInsertionPerturbation(
                systematic_gene_name=mapping.locus_tag,
                perturbed_gene_name=mapping.perturbed_gene_name,
                gene_namespace=STRAIN_GENE_NAMESPACES[REFERENCE_STRAIN],
                identifier_mapping=DerivedIdentifierMapping(
                    source_identifier=mapping.b_number, route="eck_crosswalk"
                ),
                description=PERTURBATION_DESCRIPTION,
                transposon=str(TRANSPOSON.value),
                library_pool=MUTANT_LIBRARY,
            )
        ]
    )


def units_for(state: str) -> str:
    """The readout's human-readable definition, which names the assay format."""
    return UNITS_PLANKTONIC if state == "liquid" else UNITS_SOLID


def screen_id(exp_name: str) -> str:
    """``Keio:<expName>``: the released experiment, which is the screening run."""
    return f"{ORG_ID}:{exp_name}"


def build_phenotype(
    fitness: float, standard_error: float, *, exp_name: str, state: str
) -> EnvironmentResponsePhenotype:
    """One released (gene, experiment) cell with the release's own standard error."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=fitness,
        environment_response_uncertainty=standard_error,
        environment_response_uncertainty_type=UncertaintyType.standard_error,
        units=units_for(state),
        screen_id=screen_id(exp_name),
        provenance_gaps=[N_SAMPLES_GAP, SAMPLE_UNIT_GAP],
    )


def build_reference(
    dataset_name: str, assay_environment: Environment, *, exp_name: str, state: str
) -> BacterialEnvironmentResponseExperimentReference:
    """The typical gene of the same experiment: fitness 0 by the normalization."""
    return BacterialEnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=assembly_reference(REFERENCE_STRAIN),
        environment_reference=assay_environment,
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            units=UNITS_REFERENCE,
            screen_id=screen_id(exp_name),
        ),
    )


# --------------------------------------------------------------------------- #
# Retention: every rule, with its arithmetic
# --------------------------------------------------------------------------- #
DROP_MEDIUM_NOT_IN_LIBRARY = "medium_has_no_media_library_base"
DROP_NO_ECK_PAIR = "no_one_to_one_eck_pair"

DROP_RULES: dict[str, str] = {
    DROP_MEDIUM_NOT_IN_LIBRARY: "the released per-experiment Media label is not in "
    "MEDIA_BY_LABEL, so the assay's medium has no MEDIA_LIBRARY base to join on and a "
    "free-text medium joins nothing. The release names exactly LB, LB_plus_SM_buffer "
    "and LB_agar, so this rule drops nothing today; it is the refusal that keeps a new "
    "label from being given an invented recipe",
    DROP_NO_ECK_PAIR: "the release's MG1655 b-number has no one-to-one ECK pair with a "
    "BW25113 locus, so the record has no storable identifier in the namespace its "
    "assembly pin declares. Per gene in identifier_mapping.json",
}


class DropRule(BaseModel):
    """One retention rule and the records it removed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    scope: Literal["gene", "experiment"]
    description: str
    n_records: int
    items: tuple[str, ...] = ()


class DropLog(BaseModel):
    """The record-level arithmetic of a build: source, kept, and every rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: tuple[DropRule, ...]


class AssayLedger(BaseModel):
    """The experiment-level accounting, which record counts alone cannot show.

    The 21 start samples carry no fitness column by construction, so they are not source
    records at all; the release also carries far more fitness columns than the paper's
    analysis used. Both are selection, not record drops, and they are reported here
    rather than inside :class:`DropLog`, whose arithmetic is over cells.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    used_experiments: int
    time_zero: int
    record_bearing: int
    dropped_for_medium: tuple[str, ...]
    released_columns: int
    kind_census: dict[str, int]
    format_census: dict[str, int]
    media_census: dict[str, int]
    phage_census: dict[str, int]
    moi_source_census: dict[str, int]
    release_quality_flag: dict[str, int] = Field(
        description="the release's own per-experiment `u` usability flag over the kept "
        "experiments; the paper states its standard metrics are unsuitable here"
    )


# --------------------------------------------------------------------------- #
# raw/: the mirror files the build reads
# --------------------------------------------------------------------------- #
S13_NAME = "S13_Table_MOI.xlsx"
EXPS_USED_NAME = "Keio_exps_used.tab"

#: Mirror artifacts linked into ``raw/`` under these names.
LINKED_ARTIFACTS: dict[str, str] = {
    S13_NAME: S13_TABLE_REL,
    EXPS_USED_NAME: EXPS_USED_REL,
}

#: The per-analysis-set tables the build reads, as ``raw/`` path -> tarball member. A
#: ``.tar.gz`` has no random access, so these are extracted in ONE streaming pass at
#: download time instead of re-streaming 915 MB per member at build time.
RAW_MEMBERS: dict[str, str] = {
    f"{analysis_set}/{leaf}": f"html/{analysis_set}/{leaf}"
    for analysis_set in sorted(set(ANALYSIS_SETS.values()))
    for leaf in (
        "exps",
        "fit_logratios.tab",
        "fit_standard_error_obs.tab",
        "fit_quality.tab",
    )
}


def raw_pins() -> dict[str, str]:
    """``{raw/ path: sha256}`` for every file the build reads, from the module's pins."""
    pins = {name: artifact(rel).sha256 for name, rel in LINKED_ARTIFACTS.items()}
    pins.update(
        {raw_rel: TARBALL_MEMBERS[member] for raw_rel, member in RAW_MEMBERS.items()}
    )
    return pins


def extract_tarball_members(
    members: Mapping[str, str], dest_dir: str | Path, data_root: str | None = None
) -> None:
    """Extract pinned members of the deposited tarball in ONE pass, verifying each.

    ``members`` maps a destination path (relative to ``dest_dir``) to the tarball member
    it comes from. Each member's bytes are hashed against :data:`TARBALL_MEMBERS` before
    anything is written, and a member the archive does not carry raises, so a re-packed
    archive is detected rather than silently followed.
    """
    from torchcell.data import write_verified

    wanted = {member: rel for rel, member in members.items()}
    if len(wanted) != len(members):
        raise ValueError("two destinations claim the same tarball member")
    path = raw_mirror_dir(data_root) / TARBALL_REL
    found: set[str] = set()
    with tarfile.open(path, mode="r:gz") as archive:
        for info in archive:
            if info.name not in wanted:
                continue
            handle = archive.extractfile(info)
            if handle is None:
                raise RuntimeError(f"{info.name} is not a file in {path}")
            payload = handle.read()
            expected = TARBALL_MEMBERS[info.name]
            dest = Path(dest_dir) / wanted[info.name]
            dest.parent.mkdir(parents=True, exist_ok=True)
            write_verified(payload, dest, expected, f"{path}:{info.name}")
            found.add(info.name)
    missing = sorted(set(wanted) - found)
    if missing:
        raise RuntimeError(f"{path} does not carry {missing}")


def fitness_columns(columns: Iterable[str]) -> dict[str, str]:
    """``{experiment name: column label}`` for a released fitness table.

    The release labels a column ``"<expName> <expDescription>"``, and the first three
    columns (``locusId``, ``sysName``, ``desc``) are the gene key.
    """
    return {
        str(label).split(" ", 1)[0]: str(label)
        for label in columns
        if str(label) not in ("locusId", "sysName", "desc")
    }


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class PhageRbTnseqMutalik2020Dataset(ExperimentDataset):
    """Mutalik 2020 per-(gene, experiment) phage-challenge RB-TnSeq gene fitness."""

    #: The library is BW25113's KEIO_ML9; the released b-numbers are MG1655's and are
    #: remapped through the ECK join (see the module docstring's identifier finding).
    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = "data/torchcell/phage_rbtnseq_mutalik2020",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is the BW25113 genome the entry points inject."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The two linked mirror files and the twelve extracted tarball members."""
        return [*LINKED_ARTIFACTS, *RAW_MEMBERS]

    def download(self) -> None:
        """Link the mirror files into ``raw/`` and extract the pinned tarball members.

        The mirror plus the pins is canonical; the PMC and figshare URLs are retrieval
        metadata ``deposit_raw_mirror`` records, never a live build dependency. Every
        artifact is checked against the manifest and found on disk BEFORE ``raw/`` is
        created, so an incomplete mirror leaves no half-populated raw directory behind.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        mirror = raw_mirror_dir(data_root)
        for rel in (*LINKED_ARTIFACTS.values(), TARBALL_REL):
            pin = artifact(rel).sha256
            check_manifest_pin(rel, manifest_sha256(manifest, rel), pin)
            if not (mirror / rel).exists():
                raise RuntimeError(
                    f"required raw artifact missing from mirror: {mirror / rel}"
                )
        os.makedirs(self.raw_dir, exist_ok=True)
        for name, rel in LINKED_ARTIFACTS.items():
            link_verified(
                mirror / rel, osp.join(self.raw_dir, name), artifact(rel).sha256
            )
        extract_tarball_members(RAW_MEMBERS, self.raw_dir, data_root)
        log.info(
            "Mutalik 2020: %d mirror files linked and %d tarball members extracted "
            "into %s (sha256 verified)",
            len(LINKED_ARTIFACTS),
            len(RAW_MEMBERS),
            self.raw_dir,
        )

    def _bw25113(self) -> EcoliK12BW25113Genome:
        """The BW25113 genome: injected by the build entry points, or opened here."""
        if self.ecoli_genome is None:  # a direct run; the entry points inject it
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        genome = self.ecoli_genome
        if not isinstance(genome, EcoliK12BW25113Genome):
            raise TypeError(
                f"{type(self).__name__} needs the BW25113 genome, got "
                f"{type(genome).__name__}"
            )
        return genome

    def _kept_assays(
        self, axis: ExperimentAxis, released: Mapping[str, ReleasedAssay]
    ) -> tuple[list[Assay], list[str]]:
        """The record-bearing assays whose medium has a library base, and those dropped.

        The 21 start samples are excluded first: a time-zero sample is the denominator of
        every log ratio and the release gives it no fitness column.
        """
        kept: list[Assay] = []
        dropped: list[str] = []
        for assay in axis.assays:
            if assay.kind == "time_zero":
                continue
            if released[assay.exp_name].media_label not in MEDIA_BY_LABEL:
                dropped.append(assay.exp_name)
                continue
            kept.append(assay)
        return kept, dropped

    @post_process
    def process(self) -> None:
        """Build one record per (gene, kept experiment) and write the LMDB."""
        import pandas as pd

        verify_raw_files(self.raw_dir, raw_pins())
        moi_table = read_moi_workbook(osp.join(self.raw_dir, S13_NAME))
        axis = parse_experiment_axis(osp.join(self.raw_dir, EXPS_USED_NAME), moi_table)
        analysis_sets = sorted(set(ANALYSIS_SETS.values()))
        released = read_released_assays(
            {
                analysis_set: osp.join(self.raw_dir, analysis_set, "exps")
                for analysis_set in analysis_sets
            }
        )
        kept, dropped_for_medium = self._kept_assays(axis, released)
        by_set: dict[str, list[Assay]] = {name: [] for name in analysis_sets}
        for assay in kept:
            by_set[assay.analysis_set].append(assay)

        tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for analysis_set in analysis_sets:
            fitness = pd.read_csv(
                osp.join(self.raw_dir, analysis_set, "fit_logratios.tab"),
                sep="\t",
                dtype={"sysName": str},
            )
            errors = pd.read_csv(
                osp.join(self.raw_dir, analysis_set, "fit_standard_error_obs.tab"),
                sep="\t",
                dtype={"sysName": str},
            )
            if list(fitness["sysName"]) != list(errors["sysName"]):
                raise ValueError(
                    f"{analysis_set}: the fitness and standard-error tables do not "
                    "carry the same genes in the same order"
                )
            tables[analysis_set] = (fitness, errors)

        measured = sorted(
            {str(name) for fitness, _ in tables.values() for name in fitness["sysName"]}
        )
        bw25113 = self._bw25113()
        mg1655 = bacterial_genome("ecoli", "MG1655", _data_root())
        if not isinstance(mg1655, EcoliK12MG1655Genome):
            raise TypeError(f"expected the MG1655 genome, got {type(mg1655).__name__}")
        mapping, identifiers = map_measured_genes(
            measured, mg1655, bw25113, label=self.name
        )
        genotypes = {
            b_number: build_genotype(gene) for b_number, gene in mapping.items()
        }

        environments = {
            assay.exp_name: environment(
                assay, released[assay.exp_name], moi_table.get(assay.exp_name)
            )
            for assay in kept
        }
        references = {
            assay.exp_name: build_reference(
                self.name,
                environments[assay.exp_name],
                exp_name=assay.exp_name,
                state=released[assay.exp_name].state,
            )
            for assay in kept
        }
        publication = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        dropped_by_gene: Counter[str] = Counter()
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for analysis_set in analysis_sets:
                fitness, errors = tables[analysis_set]
                fit_columns = fitness_columns(fitness.columns)
                se_columns = fitness_columns(errors.columns)
                genes = [str(name) for name in fitness["sysName"]]
                for assay in tqdm(by_set[analysis_set], desc=analysis_set):
                    state = released[assay.exp_name].state
                    assay_environment = environments[assay.exp_name]
                    reference = references[assay.exp_name]
                    values = fitness[fit_columns[assay.exp_name]].to_numpy()
                    standard_errors = errors[se_columns[assay.exp_name]].to_numpy()
                    for b_number, value, standard_error in zip(
                        genes, values, standard_errors, strict=True
                    ):
                        genotype = genotypes.get(b_number)
                        if genotype is None:
                            dropped_by_gene[b_number] += 1
                            continue
                        experiment = BacterialEnvironmentResponseExperiment(
                            dataset_name=self.name,
                            genotype=genotype,
                            environment=assay_environment,
                            phenotype=build_phenotype(
                                float(value),
                                float(standard_error),
                                exp_name=assay.exp_name,
                                state=state,
                            ),
                        )
                        txn.put(
                            f"{idx}".encode(),
                            self._intern_record(
                                experiment, reference, publication, itxn
                            ),
                        )
                        idx += 1
        env.close()
        interned_env.close()

        self._write_reports(
            axis=axis,
            released=released,
            kept=kept,
            dropped_for_medium=dropped_for_medium,
            tables=tables,
            identifiers=identifiers,
            dropped_by_gene=dropped_by_gene,
            kept_records=idx,
        )
        log.info(
            "Mutalik2020: wrote %d records over %d genes and %d experiments",
            idx,
            len(mapping),
            len(kept),
        )

    def _write_reports(
        self,
        *,
        axis: ExperimentAxis,
        released: Mapping[str, ReleasedAssay],
        kept: Sequence[Assay],
        dropped_for_medium: Sequence[str],
        tables: Mapping[str, tuple[Any, Any]],
        identifiers: MappingReport,
        dropped_by_gene: Mapping[str, int],
        kept_records: int,
    ) -> None:
        """Write the retention ledger, the assay ledger and the identifier report."""
        import pandas as pd

        #: Source records are the cells of every record-bearing used experiment, the
        #: ones dropped for their medium INCLUDED, so the arithmetic below closes over
        #: both rules. The 21 start samples are not source records at all: the release
        #: gives a time-zero sample no fitness column (see :class:`AssayLedger`).
        def cells(exp_name: str) -> int:
            return len(tables[released[exp_name].set_name][0])

        source_records = sum(cells(a.exp_name) for a in kept) + sum(
            cells(name) for name in dropped_for_medium
        )
        rules = (
            DropRule(
                rule=DROP_MEDIUM_NOT_IN_LIBRARY,
                scope="experiment",
                description=DROP_RULES[DROP_MEDIUM_NOT_IN_LIBRARY],
                n_records=sum(cells(name) for name in dropped_for_medium),
                items=tuple(dropped_for_medium),
            ),
            DropRule(
                rule=DROP_NO_ECK_PAIR,
                scope="gene",
                description=DROP_RULES[DROP_NO_ECK_PAIR],
                n_records=sum(dropped_by_gene.values()),
                items=tuple(sorted(dropped_by_gene)),
            ),
        )
        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=kept_records,
            dropped_records=source_records - kept_records,
            rules=rules,
        )
        accounted = sum(rule.n_records for rule in rules)
        if accounted != drop_log.dropped_records:
            raise RuntimeError(
                f"drop accounting mismatch: rules total {accounted}, "
                f"{drop_log.dropped_records} records missing from the build"
            )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))

        quality: Counter[str] = Counter()
        for analysis_set in sorted(tables):
            frame = pd.read_csv(
                osp.join(self.raw_dir, analysis_set, "fit_quality.tab"), sep="\t"
            )
            flags = frame.set_index("name")["u"]
            for assay in kept:
                if assay.analysis_set == analysis_set:
                    quality[str(flags[assay.exp_name])] += 1
        ledger = AssayLedger(
            used_experiments=len(axis.assays),
            time_zero=len(axis.time_zero),
            record_bearing=len(kept),
            dropped_for_medium=tuple(dropped_for_medium),
            released_columns=sum(
                len(fitness_columns(tables[name][0].columns)) for name in tables
            ),
            kind_census=dict(Counter(a.kind for a in kept)),
            format_census=dict(Counter(released[a.exp_name].state for a in kept)),
            media_census=dict(Counter(released[a.exp_name].media_label for a in kept)),
            phage_census=dict(
                Counter(str(a.phage) for a in kept if a.phage is not None)
            ),
            moi_source_census=dict(
                Counter(str(a.moi_source) for a in kept if a.kind == "phage")
            ),
            release_quality_flag=dict(quality),
        )
        with open(osp.join(self.preprocess_dir, "assay_ledger.json"), "w") as handle:
            handle.write(ledger.model_dump_json(indent=2))
        with open(
            osp.join(self.preprocess_dir, "identifier_mapping.json"), "w"
        ) as handle:
            handle.write(identifiers.model_dump_json(indent=2))

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built tree
# --------------------------------------------------------------------------- #
#: The frozen record-count oracle, measured 2026-10-07 on the deposited release: the
#: mapped genes of each analysis set times its record-bearing used experiments.
EXPECTED_SET_CENSUS: dict[str, int] = {
    "Keio_ML9_set16_set19": 212686,
    "Keio_ML9_set28_set29": 33255,
    "Keio_ML9_set30": 40403,
}
EXPECTED_RECORDS = sum(EXPECTED_SET_CENSUS.values())

#: The frozen gene-set oracle: distinct BW25113 locus tags over the kept records.
EXPECTED_GENES = 3697

#: Records per kind of experiment: 68 phage challenges and 10 no-phage controls.
EXPECTED_KIND_CENSUS: dict[str, int] = {"phage": 249612, "no_phage_control": 36732}

#: Records per assay format, the split the 68 challenges must not collapse across.
EXPECTED_FORMAT_CENSUS: dict[str, int] = {"liquid": 245941, "solid": 40403}

VERIFY_PROVENANCE = Provenance(
    source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{TARBALL_REL}",
    citation_key=CITATION_KEY,
    sha256=artifact(TARBALL_REL).sha256,
    method=(
        "the deposited figshare RB-TnSeq release (10.6084/m9.figshare.11413128): one "
        "BacterialEnvironmentResponseExperiment per (gene, experiment), the normalized "
        "log2 change in the abundance of a gene's insertion mutants with the release's "
        "own estimated standard error; experiments selected by the paper's "
        "Keio_exps_used.tab, doses by the S13 Table"
    ),
    page="html/<analysis set>/fit_logratios.tab + fit_standard_error_obs.tab + exps",
    retrieved=RETRIEVED_AT,
)


def experiment_census(records: Iterable[Mapping[str, Any]]) -> list[LevelResult]:
    """SUPPLEMENTARY L1: the record census by analysis set, kind and assay format.

    The shared ``count`` row checks the dataset total, which one experiment's rows could
    cover for another's. These rows pin the three splits the selection and the
    environment identity turn on: a column read from the wrong analysis set, a control
    read as a challenge, and the planktonic / solid-agar split that must stay two
    environments rather than one.
    """
    by_set: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    by_format: Counter[str] = Counter()
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        exp_name = str(phenotype["screen_id"]).removeprefix(f"{ORG_ID}:")
        set_match = re.match(r"(set\d+)", exp_name)
        if set_match is None:
            raise ValueError(f"unreadable screen_id {phenotype['screen_id']!r}")
        by_set[ANALYSIS_SETS[set_match.group(1)]] += 1
        perturbations = record["experiment"]["environment"]["perturbations"]
        by_kind["phage" if perturbations else "no_phage_control"] += 1
        by_format[str(record["experiment"]["environment"]["media"]["state"])] += 1
    rows = []
    for name, observed, expected in (
        ("analysis_set_census", dict(by_set), EXPECTED_SET_CENSUS),
        ("experiment_kind_census", dict(by_kind), EXPECTED_KIND_CENSUS),
        ("assay_format_census", dict(by_format), EXPECTED_FORMAT_CENSUS),
    ):
        passed = observed == expected
        rows.append(
            LevelResult(
                level=Level.L1,
                name=name,
                passed=passed,
                message=(
                    f"SUPPLEMENTARY: {observed}"
                    if passed
                    else f"SUPPLEMENTARY: {observed} is not {expected}"
                ),
                details={"observed": observed, "expected": expected},
            )
        )
    return rows


def stored_tags_are_loci(
    tags: Iterable[str], resolve: Callable[[str], Any]
) -> LevelResult:
    """SUPPLEMENTARY L1: every stored tag resolves to itself as a locus of the assembly.

    The shared ``canonical_gene_names`` rule requires status ``current``, which a
    pseudogene locus never has (the bacterial resolver returns ``non_gene_feature``,
    naming the same tag). This row accepts a gene or a pseudogene locus that resolves to
    itself and counts the statuses; it is added beside the shared row, never in its place.
    """
    statuses: Counter[str] = Counter()
    elsewhere: list[str] = []
    ordered = sorted(set(tags))
    for tag in ordered:
        resolution = resolve(tag)
        statuses[str(resolution.status.value)] += 1
        if resolution.systematic_name != tag or resolution.status not in (
            GeneNameStatus.CURRENT,
            GeneNameStatus.NON_GENE_FEATURE,
        ):
            elsewhere.append(tag)
    return LevelResult(
        level=Level.L1,
        name="stored_tags_are_loci_of_the_pinned_assembly",
        passed=not elsewhere,
        message=(
            f"SUPPLEMENTARY: {len(ordered)} stored tags, statuses {dict(statuses)}; "
            f"{len(elsewhere)} do not resolve to themselves"
        ),
        details={
            "n_tags": len(ordered),
            "statuses": dict(statuses),
            "elsewhere": elsewhere[:20],
        },
    )


def verify_build(
    dataset_root: str,
    *,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 verifier on a built tree and write its report.

    The gene universe and the resolver are BW25113's (every GenBank locus, pseudogenes
    included), which is the assembly every record pins. The verifier is the STREAMING
    one: 286,344 records with a component-level medium on each environment are not
    materialized. Four SUPPLEMENTARY rows are appended -- the three censuses of
    :func:`experiment_census` and :func:`stored_tags_are_loci` -- and the shared
    ``pair_uniqueness`` and ``canonical_gene_names`` rows keep their own verdicts.

    This is the loader's own entry point rather than
    ``torchcell.verification.runners.run_environment_response``, which is yeast-only
    today (the same reason tong2020, borchert2024 and menasalvas2025 carry their own).
    The report is written to ``preprocess/verification_report.json``.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    genome = bacterial_genome("ecoli", REFERENCE_STRAIN, data_root)
    if not isinstance(genome, EcoliK12BW25113Genome):
        raise TypeError(f"expected the BW25113 genome, got {type(genome).__name__}")
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name=PhageRbTnseqMutalik2020Dataset.__name__,
        provenance=VERIFY_PROVENANCE,
        expected_count=expected_count,
        sgd_genes=set(genome.genbank.loci),
        min_containment=1.0,
        resolve_gene_name=genome.resolve_gene_name,
    )
    for row in experiment_census(stream_records(dataset_root)):
        report.add(row)
    tags = {
        str(perturbation["systematic_gene_name"])
        for record in stream_records(dataset_root)
        for perturbation in record["experiment"]["genotype"]["perturbations"]
    }
    report.add(stored_tags_are_loci(tags, genome.resolve_gene_name))
    if len(tags) != EXPECTED_GENES:
        report.add(
            LevelResult(
                level=Level.L1,
                name="gene_set_size",
                passed=False,
                message=f"SUPPLEMENTARY: {len(tags)} genes, not {EXPECTED_GENES}",
                details={"observed": len(tags), "expected": EXPECTED_GENES},
            )
        )
    else:
        report.add(
            LevelResult(
                level=Level.L1,
                name="gene_set_size",
                passed=True,
                message=f"SUPPLEMENTARY: {EXPECTED_GENES} distinct BW25113 locus tags",
                details={"observed": len(tags), "expected": EXPECTED_GENES},
            )
        )
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build the dataset and verify it, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/phage_rbtnseq_mutalik2020")
    dataset = PhageRbTnseqMutalik2020Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])
    for name in ("dropped_records.json", "assay_ledger.json"):
        print(
            json.dumps(
                json.loads(Path(root, "preprocess", name).read_text()), indent=2
            )[:3000]
        )
    print(verify_build(root, data_root=data_root).summary())


if __name__ == "__main__":
    main()
