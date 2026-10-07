# torchcell/datasets/ecoli/mutalik2020
# [[torchcell.datasets.ecoli.mutalik2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/mutalik2020
# Test file: tests/torchcell/datasets/ecoli/test_mutalik2020.py
"""Mutalik 2020 phage-resistance RB-TnSeq screen: raw mirror, sourcing, experiment axis.

Mutalik et al. 2020 (PLoS Biology, doi:10.1371/journal.pbio.3000877) challenged the
*E. coli* K-12 BW25113 RB-TnSeq library (Wetmore 2015's KEIO_ML9) with 14 double-stranded
DNA phages at a range of multiplicities of infection, in planktonic and solid-agar pooled
competitive-growth assays, and read strain abundance out by Bar-seq. The gene-level
readout is "the normalized log2 change in the abundance of mutants in that gene".

**This module is the provenance and sourcing layer, NOT a dataset loader.** It pins and
deposits the raw artifacts, binds every statistical value to a verbatim quote in a
sha256-pinned mirror, and parses the experiment (environment) axis. No
``ExperimentDataset`` subclass is registered, because the record cannot be written
honestly yet: the environment of every record IS a phage challenge, and the schema has no
typed environment perturbation for a phage. ``EnvironmentPerturbationType`` is
``SmallMoleculePerturbation | EnvironmentPhysicalPerturbation | BiologicPerturbation``,
whose leaves are an InChIKey-identified small molecule, a scalar physical factor
(``PhysicalFactor`` has no viral member) and a proteinaceous agent
(``BiologicAgentClass`` is peptide / protein / antibody / toxin). A virion is none of the
three, and the dose is a multiplicity of infection, which ``ConcentrationUnit`` and
``DoseBasis`` cannot express. Filing a phage under any existing leaf would mislabel the
agent, and leaving the phage out of the environment would collapse 68 phage challenges
onto one environment identity. The needed addition is stated in the PR body and in
[[torchcell.datasets.ecoli.mutalik2020]]; the dataset class lands on top of this module
once it exists.

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

WHAT THE RECORDS WILL BE. One record per (gene, experiment): 3,716 genes across the 68
phage assays and 10 no-phage controls of the released tables, 250,960 (gene, experiment)
cells in the S1 Table alone. The phenotype is ``EnvironmentResponsePhenotype``
(``measurement_type=log2_ratio``, ``assay_type=pooled_competitive_growth_barcode``) under
``BacterialEnvironmentResponseExperiment``, NOT ``FitnessPhenotype``: the value is a
signed log2 ratio, and 106,536 of the 250,960 released cells (42.45%) are negative, which
``FitnessPhenotype.validate_fitness`` would clamp to 0.0. The genotype is one
``TransposonInsertionPerturbation`` per gene.

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
at 0.961 (the weakest). The leaf has no slot for a derived mapping, so the field the
record needs to say so is named in the PR body.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
import re
import shutil
import tarfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import openpyxl
from pydantic import BaseModel, ConfigDict, Field

from torchcell.datamodels.schema import BacterialReferenceStrain
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    eck_crosswalk,
    reconcile_locus_tags,
)
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.sequence.genome.ecoli.k12 import (
    EckPair,
    EcoliK12BW25113Genome,
    EcoliK12MG1655Genome,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

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


def read_moi_table(data_root: str | None = None) -> dict[str, MoiRow]:
    """The S13 Table MOI of every RB-TnSeq BW25113 experiment, keyed by experiment name.

    The sheet releases the MOI as a formula over its own inputs rather than as a number,
    so the formulas are checked against the forms the sheet uses and then evaluated:
    ``MOI = pfu/ml * 0.35 mL * dilution / (0.04 OD * 0.35 mL * 8e8 cfu/mL)``. A dilution
    cell that references the row above is resolved through the chain. A formula of any
    other shape raises, so a changed sheet is detected instead of mis-evaluated.
    """
    path = raw_mirror_dir(data_root) / S13_TABLE_REL
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


def read_experiment_axis(data_root: str | None = None) -> ExperimentAxis:
    """Parse the released experiment list into the row's environment axis.

    The dose of a challenge comes from the S13 Table, which the Methods designate; an
    experiment S13 does not carry falls back to the MOI its own description states, and
    says so in ``moi_source``. The phage name is taken from S13 where it has the row,
    because the descriptions spell the same phage several ways (``CI1857``,
    ``lambda1857``, ``I86``).
    """
    path = raw_mirror_dir(data_root) / EXPS_USED_REL
    moi_table = read_moi_table(data_root)
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
