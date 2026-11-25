# torchcell/datasets/ecoli/fang2025
# [[torchcell.datasets.ecoli.fang2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/fang2025
# Test file: tests/torchcell/datasets/ecoli/test_fang2025.py
r"""Fang 2025: the two-round CRISPRi-FACS free-fatty-acid enrichment screen of E. coli.

Fang, Hao, Fan et al. 2025 (Nat Commun 16, doi:10.1038/s41467-025-58368-3, PMC11954867,
citation key ``fangGenomescaleCRISPRiScreen2025``) ran a genome-scale CRISPRi library
through two rounds of Nile-Red fluorescence-activated cell sorting and counted the
guides by NGS. This module serves the per-guide readout the paper releases:
:class:`CrispriGuideFfaEnrichmentFang2025Dataset`, one
``EnvironmentResponseExperiment`` per (guide, round) row of Supplementary
Data 6 that the authors' own read floor admits.

THE LIBRARY IS WANG 2018's, MEASURED ID FOR ID, AND THAT IS WHAT MAKES THIS LOADABLE.
The Results state "a previously published plasmid library containing 55,671 sgRNAs20 was
transformed into E. coli strain CF", and reference 20 is Wang 2018 (Nat Commun 9, 2475),
whose library ``CrispriGuideFitnessWang2018Dataset`` already serves. Measured on the
pinned bytes: the 55,671 ids Fang releases are EXACTLY Wang's 55,671 targeting guides,
zero in one and not the other, in both rounds; Fang releases none of Wang's 400
non-targeting controls. So the spacer, the b-number and the BLASTN cluster of every
record come from Wang 2018's own raw mirror under ITS own citation key -- the shape
Babu 2014 uses to read Butland 2008's array roster -- and the provenance chain is
Fang Results -> ref 20 -> Wang Supplementary Data 2 and 3.

A SECOND, INDEPENDENT PROOF OF THAT REUSE IS IN FANG'S OWN PRIMER TABLE, which is why
the short-form guide names in its figures are not a guess. Supplementary Data 3 names
174 primers in the ``<symbol>_<position>-for|rev`` short form Wang's Methods also uses
(``rsmE_9``), and every one of the 174 carries, after its 4-base cloning tag, either the
Wang spacer of ``<symbol><bNNNN>_<position>`` or its exact reverse complement: 174 of
174, no mismatch, case-insensitive on the symbol (Fang prints ``folk`` for ``folK``).
That is what resolves ``pcnB_956`` to ``pcnBb0143_956`` and gives round 2's background
knockdown a released spacer instead of a derived one.

THE RECORD IS A SIGNED LOG2 SORT ENRICHMENT, AND BOTH EQUATIONS REPRODUCE FROM THE
BYTES. Methods equation (1) normalizes ``(reads + 1) / total reads`` and equation (2)
takes ``log2(normalized AS / normalized BS)``. Recomputed per row from the released read
counts: equation (1) reproduces the released ``Normalized read`` columns EXACTLY (max
error 0.0) and equation (2) reproduces ``Fitness (all)`` to 1.8e-15 in both rounds. The
value is routinely negative, so it is an ``EnvironmentResponsePhenotype`` with
``measurement_type=log2_ratio`` and never a ``FitnessPhenotype``, which clamps
non-positive values.

``assay_type`` IS ``biosensor_readout``, NOT ``pooled_competitive_growth_barcode``, and
the paper says why. BS and AS are the SAME culture: one is sampled before the sort and
one after it, so no selective growth separates them -- "fitness reflects the relative
abundance of each sgRNA due solely to sorting, helping to minimize data noise from
cultivation." What does the separating is the Nile-Red fluorescence the sorter gates on,
"the top 1% of cells displaying the highest fluorescence intensity". The N20 spacer is
still the molecule counted, but the ASSAY is a fluorescence sort, and calling it
competitive growth would put this number on the same method axis as Wang 2018's five
growth selections, which it is not.

ROUND 2's GENOTYPE CARRIES TWO KNOCKDOWNS, BECAUSE THE REFERENCE CLASS CANNOT CARRY ONE.
Round 1 screens the library in strain CF; round 2 screens the same library in the pcnBi
strain, CF carrying the round-1 winner's ``pcnB_956`` guide on a pCF-pcnBi plasmid. That
repression is a real edit to the cell and it is present in every round-2 record, so it
is a ``CrisprInterferencePerturbation`` leaf on ``b0143`` beside the library
guide's. It is not pushed into the reference because
``EnvironmentResponseExperimentReference`` holds a genome, an environment and a
phenotype and has no genotype field, so a reference cannot express "the pcnBi strain" at
all. Consequence: round-1 records are single-knockdown and round-2 records are double,
and the pcnB leaf is the only perturbation this dataset writes that is not a library
guide.

THE HOST IS BACKGROUND, NOT PERTURBATION. CF is "an MG1655 (DE3) derivative with fadE
deletion) carrying the pCF plasmid for expression of dCas9 and the truncated fatty
acyl-ACP thioesterase TesA'". The dfadE lesion, the DE3 prophage and the pCF plasmid are
identical in every record of both rounds and in the reference, so they are a
``StrainBackground`` rather than a per-record edit. The paper prints no locus
tag for ``fadE`` (``grep PP_\|b0221`` finds none), so the allele is carried by GENE NAME
with its construction quoted and the locus tag left unasserted rather than derived.

RETENTION: 111,342 released rows become 15,708 records under TWO rules.

The Methods state "sgRNAs with fewer than 20 reads in each library were excluded from
the analysis to calculate sgRNA fitness", and ``si8.xlsx`` splits the fitness into a
qualified ``Fitness`` column plus one column per library that fell below the floor.
Applying the floor independently reproduces the qualified column with ZERO
disagreement in both rounds: round 1 keeps the 15,380 rows at >= 20 reads in all three
of its libraries (transformation, before sorting, after sorting) and round 2 keeps the
331 rows at >= 20 in both of its two. Every other row still carries a ``Fitness (all)``
value and it is NOT loaded, for the reason Wang 2018 gives for its own ``Bad`` rows: the
denominator is below the authors' stated robustness floor, they use none of these
numbers, and no phenotype field can carry the flag, so storing them unmarked would
present them as equal in quality to the 15,708.

That is the ONE rule. Every cluster member's b-number is stored as Wang 2018's
Supplementary Data 2 releases it, with the member's own symbol, and the loader reads no
annotation (the owner's rule of 2026-10-12: the data is the data, and the genome is a
separate layer joined at modeling time). Two of those b-numbers, ``b4700`` (``sokE``)
and ``b4590`` (``ybfK``), are ones GCA_000005845.2 no longer carries; before the rule
their 3 rows (2 in round 1, 1 in round 2) were dropped as ``retired_target`` and the two
members were pruned from the genotypes of the clusters that carried them beside a
current b-number. They are records now, and the L4 genome-reference rows report them
as findings (#1069). The per-round arithmetic and the per-library split of the
excluded rows are in ``preprocess/build_accounting.json``.

Round 2 keeping only 331 of 55,671 is not an error and not a drop rule: 55,334 of its
guides read below 20 AFTER sorting, which is what sorting the top 1% of a library
already shaped by a first round does. It is recorded because a reader seeing 331 will
otherwise suspect the parse.

THE PER-STRAIN FFA TITERS ARE A SECOND DATASET OF THIS MODULE, ONE RECORD PER CULTURE:
:class:`FfaTiterFang2025Dataset`, whose own section below this one carries its design.
Before 2026-11-25 they were NOT loaded, over the ``pcnBi-acrDi-fadR+`` strain's P_BAD
``fadR`` arm: the decline read "no bacterial gene-perturbation leaf types" an
overexpression, which is wrong. ``GeneAdditionPerturbation`` is the leaf for "an extra
native copy, carried on a plasmid", and Supplementary Data 2 states the plasmid as
``PBAD: fadR, ColE1, AmpR``, so the arm is a gene addition and needs no new leaf.

WHAT IS RELEASED AND DELIBERATELY NOT LOADED:

- **The 400 non-targeting controls.** Fang releases none of them, so unlike Wang 2018
  there is no control row to drop and no control median: equation (2) has no
  control-subtraction term, and the zero of this dataset's scale is "unchanged by
  sorting" rather than "equal to the non-targeting guides".
- **GEO GSE267827** (the screen reads), **GSE267710** (transcriptomics) and
  **PXD052390** (proteomics) are recorded and not deposited: no record reads reads, and
  the two omics sets are gene-level differential expression this class cannot hold.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import pickle
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tqdm import tqdm

from torchcell.data import Organism
from torchcell.data.experiment_dataset import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import M9_MODIFIED_FANG2025
from torchcell.datamodels.schema import (
    AssayType,
    BacterialAssemblySet,
    BacterialGeneNamespace,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    CrisprInterferencePerturbation,
    CultureEnvironment,
    DeletionStrainPerturbation,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    GeneAdditionPerturbation,
    GenePerturbationType,
    Genotype,
    Localization,
    MeasurementType,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainBackground,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import (
    BACTERIAL_ASSEMBLY_SETS,
    assembly_reference,
    bacterial_genome,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli.wang2018 import CITATION_KEY as WANG_CITATION_KEY
from torchcell.datasets.ecoli.wang2018 import CLUSTERS_FILE as WANG_CLUSTERS_FILE
from torchcell.datasets.ecoli.wang2018 import LIBRARY_FILE as WANG_LIBRARY_FILE
from torchcell.datasets.ecoli.wang2018 import GeneCluster, read_clusters, read_library
from torchcell.datasets.ecoli.wang2018 import load_manifest as wang_load_manifest
from torchcell.datasets.ecoli.wang2018 import manifest_sha256 as wang_manifest_sha256
from torchcell.datasets.ecoli.wang2018 import raw_mirror_dir as wang_raw_mirror_dir
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome
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
# Paper identity and the pinned artifacts
# --------------------------------------------------------------------------- #
DOI = "10.1038/s41467-025-58368-3"
PMCID = "PMC11954867"
TITLE = (
    "Genome-scale CRISPRi screen identifies pcnB repression conferring improved "
    "physiology for overproduction of free fatty acids in Escherichia coli"
)
CITATION_KEY = "fangGenomescaleCRISPRiScreen2025"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: PMC Article Datasets bucket prefix for this article's supplementary files.
PMC_PREFIX = f"{PMCID}.1"
#: Publisher file-name stem of every supplementary object.
ESM_STEM = "41467_2025_58368_MOESM"

#: The article OCR in the torchcell-library mirror: quoted for every sourced value,
#: never parsed here.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "93d4303ce7aad2cec712ad09fd41319e08f4b735dc8dfc69659d55e590f2ee3d"

#: The day the recorded ``pmc_cloud`` retrieval of the deposited file was last run and
#: reproduced its pin.
RAW_RETRIEVED_AT = "2026-11-25"

#: Accessions recorded and deliberately NOT deposited: the screen reads this loader does
#: not consume, and the two omics sets this class cannot hold.
GEO_SCREEN_ACCESSION = "GSE267827"
GEO_TRANSCRIPTOME_ACCESSION = "GSE267710"
PROTEOMEXCHANGE_ACCESSION = "PXD052390"

MG1655_STRAIN: Literal["MG1655"] = "MG1655"
MG1655_NAMESPACE: BacterialGeneNamespace = "ecoli_k12_mg1655_bnumber"
_ASSEMBLY_SET: TypeAdapter[BacterialAssemblySet] = TypeAdapter(BacterialAssemblySet)
MG1655_ASSEMBLY_SET: BacterialAssemblySet = _ASSEMBLY_SET.validate_python(
    BACTERIAL_ASSEMBLY_SETS[MG1655_STRAIN]
)

#: OCR page handles for the quotes below.
_PAGE_RESULTS_SCREEN = "Results, genome-scale CRISPRi screen"
_PAGE_METHODS_MEDIUM = "Methods, Strains, plasmids, and medium"
_PAGE_METHODS_CULTIVATION = "Methods, Cultivation and fermentation"
_PAGE_METHODS_SCREEN = "Methods, Genome-scale CRISPRi screen"
_PAGE_METHODS_NGS = "Methods, NGS library preparation and sequencing"
_PAGE_DATA_AVAILABILITY = "Data availability"


class RawFile(BaseModel):
    """One deposited supplementary file: its publisher object and its sha256 pin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    moesm: int
    data_number: int
    description: str
    sha256: str
    extension: str = "xlsx"

    @property
    def filename(self) -> str:
        """Publisher file name, which is also the name inside ``raw/``."""
        return f"{ESM_STEM}{self.moesm}_ESM.{self.extension}"

    @property
    def relpath(self) -> str:
        """Path inside the raw mirror."""
        return f"si/si_data/{self.filename}"

    @property
    def cloud_key(self) -> str:
        """Key of this object in the PMC Article Datasets bucket."""
        return f"{PMC_PREFIX}/{self.filename}"

    @property
    def url(self) -> str:
        """The recorded retrieval URL."""
        return f"https://pmc-oa-opendata.s3.amazonaws.com/{self.cloud_key}"


#: Supplementary Data 6, the source-data workbook: the only released file whose unit of
#: observation is a single sgRNA, and the only build input from THIS paper.
SCREEN_FILE = RawFile(
    moesm=8,
    data_number=6,
    description=(
        "Source data for every figure panel; sheets 'Figure 1' and 'Figure 5' hold the "
        "two rounds' per-guide read counts and fitness"
    ),
    sha256="04945e4651c05b1c6d6fa3dc057ae6bced9f370b2537068472354510d20a8a5a",
)
RAW_FILES: tuple[RawFile, ...] = (SCREEN_FILE,)

#: Supplementary Data 1: every strain of the study with its construction, verbatim
#: ("MG with plasmids pCF and pSg-gyrA_9"). The titer dataset's genotypes are parsed
#: from it, so the strain a titer row names is typed from the source's own sentence.
STRAIN_FILE = RawFile(
    moesm=3,
    data_number=1,
    description="Strains used in this study: strain, description, reference",
    sha256="523574e4406d9f1cfcf83e200241d76fadbe4f39c0ed3017a379937b8dacc264",
)
#: Supplementary Data 2: every plasmid with its parts, verbatim ("PLlacO-1: sgRNA
#: pcnB_956, pBR322, AmpR"). What turns a strain's plasmid list into perturbations.
PLASMID_FILE = RawFile(
    moesm=4,
    data_number=2,
    description="Plasmids used in this study: plasmid, description, reference",
    sha256="b7836a9ca6300960d498a8a088bef67976b648865d46e664ebffbcdb10dad40f",
)
#: Supplementary Data 3: the 205 primers, including the 174 ``<guide>-for|rev`` spacer
#: primers that tie each short-form guide name to its 20-mer.
PRIMER_FILE = RawFile(
    moesm=5,
    data_number=3,
    description="Primers used in this study: primer, sequence",
    sha256="9fbcf594c0670a43cd50579eb4e69bb9c92ac63ecd5b93ff119d6bce9abba669",
)
#: The titer dataset's inputs from THIS paper: the source-data workbook plus the three
#: tables that state what each strain is.
TITER_RAW_FILES: tuple[RawFile, ...] = (
    SCREEN_FILE,
    STRAIN_FILE,
    PLASMID_FILE,
    PRIMER_FILE,
)
#: Every file this paper's raw mirror holds, which is what ``deposit_raw_mirror``
#: deposits and records; ``RAW_FILES`` stays the screen dataset's own input list.
MIRROR_FILES: tuple[RawFile, ...] = (
    SCREEN_FILE,
    STRAIN_FILE,
    PLASMID_FILE,
    PRIMER_FILE,
)

#: Wang 2018's two build inputs, read from ITS raw mirror under ITS citation key: the
#: BLASTN cluster map and the guide-id -> 20-mer spacer library.
WANG_RAW_FILES: tuple[Any, ...] = (WANG_CLUSTERS_FILE, WANG_LIBRARY_FILE)
#: The library shape Wang 2018 states and this build re-checks.
N_TARGETING_GUIDES = 55671
N_CONTROL_GUIDES = 400
N_CLUSTERS = 4205
N_CLUSTER_MEMBERS = 4317


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the pinned ``paper.md`` OCR mirror."""
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


# --------------------------------------------------------------------------- #
# Verbatim quotes, cut from the pinned paper.md (the OCR keeps its LaTeX markup)
# --------------------------------------------------------------------------- #
_Q_LIBRARY = (
    "a previously published plasmid library containing 55,671 sgRNAs20 was transformed "
    "into E. coli strain CF (an MG1655 (DE3) derivative with fadE deletion) carrying "
    "the pCF plasmid for expression of dCas9 and the truncated fatty acyl-ACP "
    "thioesterase TesA′"
)
_Q_LIBRARY_REFERENCE = (
    "20. Wang, T. et al. Pooled CRISPR interference screening enables genome-scale "
    "functional genomics study in bacteria with superior performance. Nat. Commun. 9, "
    "2475 (2018)."
)
_Q_FADE_DELETION = (
    "The fadE and pcnB genes of E. coli MG1655(DE3) were deleted using the λ-RED "
    "recombineering system56."
)
_Q_SORT = (
    "The top $1 \\%$ of cells displaying the highest fluorescence intensity were sorted "
    "into the LB medium, collecting over eight thousand cells."
)
_Q_NILE_RED = (
    "The sample was then mixed with $2 0 \\mu \\ g \\mathrm { m L } ^ { - 1 }$ Nile Red "
    "(N121291, Aladdin, China) and incubated in the dark for 5 min for staining."
)
_Q_SORT_ONLY = (
    "Thus, fitness reflects the relative abundance of each sgRNA due solely to sorting, "
    "helping to minimize data noise from cultivation."
)
_Q_EQUATIONS = (
    "Reads per sgRNA $( \\mathrm { r e a d s } _ { \\mathrm { s g R N A } } + 1 )$ were "
    "normalized to the total number of reads per sample (total readstotal) to calculate "
    "the relative abundance of sgRNAs (normalized readssgRNA) (Eq. 1). Fitness of each "
    "sgRNA was calculated by dividing the normalized reads for this sgRNA in the cell "
    "library after sorting (AS) by the normalized reads in the cell library before "
    "sorting (BS) and subsequently taking the $\\log _ { 2 }$ value (Eq. 2)."
)
_Q_READ_FLOOR = (
    "sgRNAs with fewer than 20 reads in each library were excluded from the analysis to "
    "calculate sgRNA fitness."
)
_Q_SCREEN_CULTURE = (
    "The remaining part was re-inoculated into two $5 0 0 \\mathrm { m L }$ flasks "
    "containing $1 0 0 \\mathrm { m L }$ of modified M9 medium containing the "
    "corresponding antibiotic and cultivated with shaking at $3 0 ^ { \\circ } \\mathrm "
    "{ C }$ and $2 5 0 \\mathsf { r p m }$ ."
)
_Q_INDUCTION = (
    "Upon reaching an $\\mathrm { \\Gamma _ { 0 D _ { 6 0 0 } } }$ of about 1, cultures "
    "were induced with $\\bf { 1 m M }$ IPTG and allowed to grow for an additional $4 0 "
    "\\mathrm { h }$ ."
)
_Q_SAMPLE_NAMES = (
    "In the first round of screening, the NGS data of samples from the transformed cell "
    "library, the cultivated cell library before sorting (BS), and the cell library "
    "after sorting (AS) were named Tra1, Cul2, and Sor3, respectively. In the second "
    "round of screening, the NGS data of samples from BS and AS were named B-cul2 and "
    "Bsor3-P5, respectively."
)
_Q_ROUND_TWO_HOST = (
    "In the second round of screening, the pcnBi strain carrying the pCF-pcnBi plasmid "
    "was utilized for competent cell preparation and electroporation of the plasmid "
    "library."
)
_Q_PCNBI_ONE_PLASMID = (
    "pcnB repression using one plasmid (the pcnBi strain) obtains comparable FFAs "
    "production as using two plasmids (the pcnB_956 strain)."
)
_Q_DATA_AVAILABILITY = (
    "The NGS data and transcriptomics data generated in this study have been deposited "
    "in the NCBI GEO database under accession codes GSE267827 and GSE267710, "
    "respectively. The mass spectrometry proteomics data generated in this study have "
    "been deposited in the ProteomeXchange Consortium under accession code PXD052390."
)
_Q_MEDIUM = (
    "Modified M9 medium58 $\\mathrm { ( pH } 7 . 2 \\mathrm { ) }$ for tube and flask "
    "fermentation was prepared as follows:"
)

# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
TEMPERATURE_C = _paper(
    30.0, _Q_SCREEN_CULTURE, page=_PAGE_METHODS_SCREEN, note="both rounds"
)
INDUCER_MM = _paper(
    1.0,
    _Q_INDUCTION,
    page=_PAGE_METHODS_SCREEN,
    note="IPTG induces dCas9 and tesA' together, so the knockdown and the "
    "fatty-acid route switch on at the same moment",
)
SCREEN_HOURS = _paper(
    40.0,
    _Q_INDUCTION,
    page=_PAGE_METHODS_SCREEN,
    note="hours after induction, not total culture time",
)
SORT_FRACTION = _paper(0.01, _Q_SORT, page=_PAGE_METHODS_SCREEN)
READ_FLOOR = _paper(
    20,
    _Q_READ_FLOOR,
    page=_PAGE_METHODS_NGS,
    note="the authors' own floor; applied to EVERY library of a round, which "
    "reproduces the released qualified Fitness column exactly in both rounds",
)
N_REPLICATES = _paper(
    1,
    _Q_SAMPLE_NAMES,
    page=_PAGE_METHODS_NGS,
    note="one NGS sample per library per round (Tra1/Cul2/Sor3, then "
    "B-cul2/Bsor3-P5), so a guide's fitness is one number over one pooled library; "
    "the 'three biological replicates' of Statistics and reproducibility are the "
    "per-strain titer panels, not the screen",
)
SAMPLE_UNIT = _paper(
    "pooled",
    _Q_SCREEN_CULTURE,
    page=_PAGE_METHODS_SCREEN,
    note="the two flasks are a volume split of one transformed library and one BS "
    "sample is drawn from it, so the sample is the pool",
)
STATISTIC = _paper(
    "log2_ratio",
    _Q_EQUATIONS,
    page=_PAGE_METHODS_NGS,
    note="verified on the pinned bytes: equation (1) reproduces the released "
    "normalized-read columns exactly and equation (2) reproduces Fitness (all) to "
    "1.8e-15 in both rounds",
)
ASSAY = _paper(
    "biosensor_readout",
    _Q_SORT_ONLY,
    page=_PAGE_METHODS_NGS,
    note=f"BS and AS are the same culture and the sorter gates on Nile-Red "
    f"fluorescence ('{_Q_NILE_RED}'), so the enrichment is a biosensor readout and "
    "not a competitive-growth selection",
)
LIBRARY_PROVENANCE = _paper(
    WANG_CITATION_KEY,
    _Q_LIBRARY,
    page=_PAGE_RESULTS_SCREEN,
    note=f"reference 20 is Wang 2018 ('{_Q_LIBRARY_REFERENCE}'), whose "
    "Supplementary Data 2 and 3 supply every spacer, b-number and cluster here",
)
HOST_BACKGROUND = _paper(
    "CF",
    _Q_LIBRARY,
    page=_PAGE_RESULTS_SCREEN,
    note=f"constructed by lambda-RED ('{_Q_FADE_DELETION}'); identical in every "
    "record of both rounds and in the reference, so it is a background and not a "
    "perturbation",
)
ROUND_TWO_BACKGROUND_GUIDE = _paper(
    "pcnBb0143_956",
    _Q_ROUND_TWO_HOST,
    page=_PAGE_METHODS_SCREEN,
    note=f"the paper writes this guide 'pcnB_956' ('{_Q_PCNBI_ONE_PLASMID}'); the "
    "short form resolves to Wang 2018's id because all 174 short-form primer names of "
    "Fang Supplementary Data 3 carry the Wang spacer of their long-form id or its "
    "reverse complement, 174 of 174",
)

_UNCERTAINTY_GAP_NOTE = (
    "equations (1) and (2) give one number per guide from one pair of pooled read "
    "counts, so there is no per-guide spread to report and the release carries none; "
    "the paper's own uncertainty statements are the per-strain titer panels' SD over "
    "three biological replicates, a different measurement"
)
_FADE_LOCUS_GAP_NOTE = (
    "the paper names the gene and not its locus tag: no b-number appears anywhere in "
    "the mirrored bytes for fadE, and deriving b0221 would store a mapping the source "
    "never made, so the allele is carried by gene name"
)


def _uncertainty_gaps() -> list[ProvenanceGap]:
    """The three uncertainty fields, typed as absences rather than left silent."""
    return [
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_reported_by_primary,
            looked_in=Provenance(
                source_uri=PAPER_MD,
                citation_key=CITATION_KEY,
                sha256=PAPER_MD_SHA256,
                method="Methods, NGS data processing (equations 1 and 2)",
                page=_PAGE_METHODS_NGS,
            ),
            note=_UNCERTAINTY_GAP_NOTE,
        )
        for field in (
            "environment_response_uncertainty",
            "environment_response_uncertainty_type",
            "environment_response_se",
        )
    ]


# --------------------------------------------------------------------------- #
# The two rounds
# --------------------------------------------------------------------------- #
class Round(BaseModel):
    """One screening round: its sheet, its column layout and its measured retention.

    ``libraries`` names the NGS samples whose read counts the floor is applied to, in
    the sheet's own column order; round 1 has three and round 2 has two, which is why
    the floor is a per-round rule rather than one hard-coded test.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_id: str
    sheet: str
    panel: str
    host_strain: str
    libraries: tuple[str, ...]
    id_column: int
    read_columns: tuple[int, ...]
    fitness_column: int
    excluded_columns: tuple[int, ...]
    fitness_all_column: int
    carries_background_knockdown: bool
    floor_kept_rows: int
    kept_records: int


#: The two rounds, in the figure order the workbook uses. ``kept_records`` is measured on
#: the pinned bytes and asserted per round by the build.
ROUNDS: tuple[Round, ...] = (
    Round(
        round_id="round_1_cf",
        sheet="Figure 1",
        panel="Figure 1d",
        host_strain="CF",
        libraries=("transformation", "before_sorting", "after_sorting"),
        id_column=10,
        read_columns=(11, 12, 13),
        fitness_column=16,
        excluded_columns=(18, 19, 20),
        fitness_all_column=21,
        carries_background_knockdown=False,
        floor_kept_rows=15380,
        kept_records=15380,
    ),
    Round(
        round_id="round_2_pcnbi",
        sheet="Figure 5",
        panel="Figure 5c",
        host_strain="pcnBi",
        libraries=("before_sorting", "after_sorting"),
        id_column=10,
        read_columns=(11, 12),
        fitness_column=15,
        excluded_columns=(17, 18),
        fitness_all_column=19,
        carries_background_knockdown=True,
        floor_kept_rows=331,
        kept_records=331,
    ),
)

#: Totals over the two rounds, measured on the pinned bytes.
EXPECTED_RECORDS = 15711
EXPECTED_PERTURBATIONS = 16191
EXPECTED_GENES = 2630

#: A library guide id: ``<symbol><bNNNN>_<position>``.
GUIDE_ID_RE = re.compile(
    r"^(?P<token>(?P<symbol>[A-Za-z0-9]+?)(?P<bnumber>b\d{4}))_(?P<position>\d+)$"
)

DROP_RULES: list[str] = [
    "below_author_read_floor -- 95,962 rows (40,291 of round 1, 55,340 of round 2): at "
    "least one of the round's NGS libraries counted fewer than "
    f"{READ_FLOOR.value} reads for this guide, which is the authors' own exclusion "
    f"('{_Q_READ_FLOOR}'). The released workbook carries these rows' value in a "
    "Fitness (reads<20, ...) column rather than in the qualified Fitness column, and "
    "applying the floor independently reproduces that split with zero disagreement in "
    "both rounds; no phenotype field can carry the flag, so they are not stored",
    "no genome rule -- every cluster member's b-number is stored as released; b4700 "
    "(sokE) and b4590 (ybfK), which GCA_000005845.2 no longer carries, key 3 records "
    "(sokEb4700_32 and sokEb4700_33 in round 1, ybfKb4590_11 in round 2) and 4 "
    "cluster-member perturbations, and the L4 genome-reference rows report them",
]


class RoundRows(BaseModel):
    """One round's released rows, split by the floor, with both equations re-checked."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    round_id: str
    source_rows: int
    kept: pd.DataFrame
    excluded_rows: int
    excluded_by_library: dict[str, int]
    total_reads: dict[str, int]
    max_equation_1_error: float
    max_equation_2_error: float


def read_round(path: str | Path, round_: Round) -> RoundRows:
    """One round's sheet of Supplementary Data 6, verified and split by the read floor.

    Both Methods equations are recomputed from the released read counts and compared to
    the released columns, and the floor is cross-tabulated against which rows carry the
    qualified ``Fitness`` value. A formula that does not reproduce, or a floor that does
    not explain the column, stops the build: the stored number would then not be the
    number the authors define.

    Equation (1)'s residual is RELATIVE and equation (2)'s is ABSOLUTE, which is what
    their scales require: a normalized read is about 1e-5 on the real release, so an
    absolute bound on it would be a bound on the magnitude rather than on the agreement,
    while a log2 ratio is an O(1) number whose absolute error is the meaningful one.
    """
    import math

    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook[round_.sheet]
        guides: list[str] = []
        reads: dict[str, list[int]] = {name: [] for name in round_.libraries}
        released_normalized: list[tuple[float, float]] = []
        fitness: list[float | None] = []
        fitness_all: list[float] = []
        excluded_flags: list[tuple[bool, ...]] = []
        for row in sheet.iter_rows(min_row=3, values_only=True):
            guide = row[round_.id_column]
            if guide is None:
                continue
            guides.append(str(guide))
            for name, column in zip(round_.libraries, round_.read_columns, strict=True):
                reads[name].append(int(row[column]))
            released_normalized.append(
                (
                    float(row[round_.fitness_column - 2]),
                    float(row[round_.fitness_column - 1]),
                )
            )
            qualified = row[round_.fitness_column]
            fitness.append(None if qualified is None else float(qualified))
            fitness_all.append(float(row[round_.fitness_all_column]))
            excluded_flags.append(
                tuple(row[c] is not None for c in round_.excluded_columns)
            )
    finally:
        workbook.close()

    if len(set(guides)) != len(guides):
        raise ValueError(f"{round_.sheet} repeats a guide id")
    if len(guides) != N_TARGETING_GUIDES:
        raise ValueError(
            f"{round_.sheet} holds {len(guides)} guide rows, the library is "
            f"{N_TARGETING_GUIDES}"
        )
    totals = {name: sum(values) for name, values in reads.items()}
    max_equation_1 = 0.0
    max_equation_2 = 0.0
    for index in range(len(guides)):
        before = (reads["before_sorting"][index] + 1) / totals["before_sorting"]
        after = (reads["after_sorting"][index] + 1) / totals["after_sorting"]
        released_before, released_after = released_normalized[index]
        max_equation_1 = max(
            max_equation_1,
            abs(before / released_before - 1.0),
            abs(after / released_after - 1.0),
        )
        max_equation_2 = max(
            max_equation_2, abs(math.log2(after / before) - fitness_all[index])
        )
    if max_equation_1 > 1e-12 or max_equation_2 > 1e-12:
        raise ValueError(
            f"{round_.round_id}: Methods equations do not reproduce from the released "
            f"read counts (eq 1 error {max_equation_1:.3e}, eq 2 error "
            f"{max_equation_2:.3e})"
        )

    floor = int(READ_FLOOR.value)
    disagreements = 0
    keep: list[int] = []
    for index in range(len(guides)):
        passes = all(reads[name][index] >= floor for name in round_.libraries)
        if passes != (fitness[index] is not None):
            disagreements += 1
        if passes:
            keep.append(index)
    if disagreements:
        raise ValueError(
            f"{round_.round_id}: the {floor}-read floor disagrees with the released "
            f"Fitness column on {disagreements} rows"
        )
    if len(keep) != round_.floor_kept_rows:
        raise ValueError(
            f"{round_.round_id}: the floor keeps {len(keep)} rows, the pinned bytes "
            f"give {round_.floor_kept_rows}"
        )
    partition = Counter(excluded_flags)
    excluded_by_library = {
        name: sum(
            count for flags, count in partition.items() if flags[position] is True
        )
        for position, name in enumerate(round_.libraries)
    }
    kept_frame = pd.DataFrame(
        {"guide_id": [guides[i] for i in keep], "fitness": [fitness[i] for i in keep]}
    )
    return RoundRows(
        round_id=round_.round_id,
        source_rows=len(guides),
        kept=kept_frame,
        excluded_rows=len(guides) - len(keep),
        excluded_by_library=excluded_by_library,
        total_reads=totals,
        max_equation_1_error=max_equation_1,
        max_equation_2_error=max_equation_2,
    )


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the repo-root ``.env``."""
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/fangGenomescaleCRISPRiScreen2025``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _retrieval(raw: RawFile) -> RetrievalRecord:
    """The recorded ``pmc_cloud`` retrieval of one supplementary object."""
    return RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=raw.url,
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": raw.cloud_key},
        sha256=raw.sha256,
        retrieved_at=RAW_RETRIEVED_AT,
    )


def deposit_raw_mirror(
    sources: Mapping[str, str | Path] | None = None, *, data_root: str | None = None
) -> Path:
    """Deposit the consumed supplementary files and write ``manifest.json``.

    With ``sources=None`` the recorded retrieval runs (``pmc_cloud_object`` on the PMC
    Article Datasets bucket, which reproduced the pin on ``RAW_RETRIEVED_AT``);
    otherwise ``sources`` maps the file's ``relpath`` to an already-retrieved copy.
    The file is hashed against its pin BEFORE anything is written, so a refusal leaves
    no partial deposit. Idempotent by sha256: a matching mirror file is left alone and
    a differing one raises rather than being overwritten.
    """
    from torchcell.literature.retrieve import pmc_cloud_object

    root = raw_mirror_dir(data_root)
    staged: dict[str, Path] = {}
    for raw in MIRROR_FILES:
        if sources is not None:
            source = Path(sources[raw.relpath])
            digest = sha256_file(source)
            if digest != raw.sha256:
                raise RuntimeError(
                    f"{source} hashes to {digest}, the pin is {raw.sha256}"
                )
            staged[raw.relpath] = source
            continue
        incoming = root / f".{raw.filename}.incoming"
        incoming.parent.mkdir(parents=True, exist_ok=True)
        incoming.write_bytes(pmc_cloud_object(raw.cloud_key))
        digest = sha256_file(incoming)
        if digest != raw.sha256:
            incoming.unlink()
            raise RuntimeError(
                f"{raw.url} now yields {digest}, the pin is {raw.sha256}"
            )
        staged[raw.relpath] = incoming

    files: list[ArtifactRecord] = []
    for raw in MIRROR_FILES:
        dest = root / raw.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha256_file(dest) != raw.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(staged[raw.relpath], dest)
        source = staged[raw.relpath]
        if source.name.startswith(".") and source.name.endswith(".incoming"):
            source.unlink()
        files.append(
            ArtifactRecord(
                path=raw.relpath,
                role=ROLE_SI_DATA,
                bytes=dest.stat().st_size,
                sha256=raw.sha256,
                source=raw.url,
                original_filename=raw.filename,
                retrieval=_retrieval(raw),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=files,
        si_data_sources=[f"https://pmc-oa-opendata.s3.amazonaws.com/{PMC_PREFIX}/"],
        si_expected=[
            f"Supplementary Data {raw.data_number} ({raw.filename}): {raw.description}"
            for raw in MIRROR_FILES
        ]
        + [
            "Supplementary Data 1-3 (MOESM3-5, strains, plasmids and primers) ARE "
            "deposited here since 2026-11-25: the titer dataset parses the strain and "
            "plasmid tables for every genotype it stores and re-checks each spacer "
            "against the primer table",
            "Supplementary Information and the Reporting Summary (MOESM1, 2, 6, 7) are "
            "NOT deposited here: they are in the torchcell-library mirror, where the "
            "Methods quotes come from",
            "The guide spacers, b-numbers and BLASTN clusters are NOT deposited here: "
            "they are Wang 2018's Supplementary Data 2 and 3, read from the "
            f"{WANG_CITATION_KEY} raw mirror, because this screen reuses that library "
            "id for id",
            f"NCBI GEO {GEO_SCREEN_ACCESSION} holds the screen reads and is NOT "
            "deposited: no record reads reads",
            f"NCBI GEO {GEO_TRANSCRIPTOME_ACCESSION} and ProteomeXchange "
            f"{PROTEOMEXCHANGE_ACCESSION} hold the transcriptome and proteome and are "
            "NOT deposited: both are gene-level differential expression this class "
            "cannot hold",
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
    raise KeyError(f"{relpath} is not in the Fang 2025 raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Record construction
# --------------------------------------------------------------------------- #
def publication() -> Publication:
    """This paper, by DOI.

    ``pubmed_id`` is left ``None``: it is in none of the mirrored bytes nor in the
    bibliography store's entry for this key, so it is not sourced and is not asserted.
    """
    return Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")


def host_background() -> StrainBackground:
    """Strain CF: MG1655(DE3) dfadE carrying pCF, constant in every record.

    ``alleles`` is EMPTY and the lesion lives in ``genotype_statement`` verbatim, which
    is what the class documents for a source that "states only a genotype string".
    ``BackgroundAllele.systematic_gene_name`` is a required locus tag validated
    against the namespace, so typing the ``fadE`` deletion would mean storing ``b0221``,
    a tag that appears nowhere in the mirrored bytes; the gap is recorded on the
    background instead of being closed by a derivation the source never made. The DE3
    prophage and the pCF plasmid stay in the statement for the same reason: neither is
    given a locus tag, coordinates or a sequence.
    """
    return StrainBackground(
        name="CF",
        reference_strain=MG1655_STRAIN,
        assembly_set=MG1655_ASSEMBLY_SET,
        parents=["E. coli MG1655(DE3)"],
        construction="lambda-RED deletion of fadE in MG1655(DE3), then transformation "
        "with pCF (dCas9 under PTrc, truncated tesA' under PT7, p15A, CmR)",
        genotype_statement="an MG1655 (DE3) derivative with fadE deletion) carrying "
        "the pCF plasmid for expression of dCas9 and the truncated fatty acyl-ACP "
        "thioesterase TesA'",
        alleles=[],
        provenance=[
            _paper(str(HOST_BACKGROUND.value), _Q_LIBRARY, page=_PAGE_RESULTS_SCREEN),
            _paper("fadE deletion", _Q_FADE_DELETION, page=_PAGE_METHODS_MEDIUM),
        ],
    )


def host_reference(data_root: str | None = None) -> ReferenceGenome:
    """MG1655 pinned to ``GCA_000005845.2``, carrying the CF background.

    Unlike Wang 2018, whose host cassette has no locus tag and whose reference therefore
    asserts no background, CF's lesion IS a named gene with a quoted construction, so
    the background is asserted and the one unstated field is gapped inside it.
    """
    return assembly_reference(
        MG1655_STRAIN, background=host_background(), data_root=data_root
    )


def _inducer() -> SmallMoleculePerturbation:
    """The 1 mM IPTG that switches on dCas9 and tesA' together."""
    return SmallMoleculePerturbation(
        compound=resolved_compound("IPTG"),
        concentration=Concentration(
            value=float(INDUCER_MM.value), unit=ConcentrationUnit.millimolar
        ),
        solvent=None,
    )


def screen_environment() -> Environment:
    """The one culture both rounds are screened in, and both arms of each round.

    BS and AS are the same flask sampled before and after the sort, so the selective and
    the control arm are ONE environment object; what separates them is the sorter's
    fluorescence gate, which is a property of the measurement and not of the medium.
    """
    return Environment(
        media=M9_MODIFIED_FANG2025,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        perturbations=[_inducer()],
        aerobicity="aerobic",
        duration_hours=float(SCREEN_HOURS.value),
    )


_UNITS = (
    "log2(normalized reads after sorting / normalized reads before sorting) of one "
    "sgRNA, where normalized reads are (reads + 1) / total reads (Methods equations 1 "
    "and 2); positive = the knockdown is enriched among the brightest 1 percent of "
    "Nile-Red-stained cells, which the paper reports as correlating with FFA titer"
)
_UNITS_REFERENCE = (
    "a guide whose normalized abundance the sort does not change, which is exactly 0.0 "
    "by equation (2); there is no non-targeting control set in this release, so the "
    "zero is the unchanged-by-sorting point and not a control median"
)


def guide_phenotype(round_: Round, fitness: float) -> EnvironmentResponsePhenotype:
    """One guide's released sort enrichment in one round."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.biosensor_readout,
        environment_response=fitness,
        n_samples=int(N_REPLICATES.value),
        sample_unit=SampleUnit(str(SAMPLE_UNIT.value)),
        units=_UNITS,
        screen_id=round_.round_id,
        provenance_gaps=_uncertainty_gaps(),
    )


def build_reference(
    dataset_name: str, round_: Round, genome_reference: ReferenceGenome
) -> EnvironmentResponseExperimentReference:
    """The unsorted-library baseline of one round: enrichment 0 by the normalization."""
    return EnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=screen_environment(),
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.biosensor_readout,
            environment_response=0.0,
            n_samples=int(N_REPLICATES.value),
            sample_unit=SampleUnit(str(SAMPLE_UNIT.value)),
            units=_UNITS_REFERENCE,
            screen_id=round_.round_id,
            provenance_gaps=_uncertainty_gaps(),
        ),
    )


def _knockdown(
    bnumber: str, gene_name: str, spacer: str
) -> CrisprInterferencePerturbation:
    """One dCas9 knockdown leaf carrying its released 20-mer spacer."""
    return CrisprInterferencePerturbation(
        systematic_gene_name=bnumber,
        perturbed_gene_name=gene_name,
        gene_namespace=MG1655_NAMESPACE,
        identifier_mapping=None,
        crispr=CrisprConstruct(
            effector="dCas9", guide_sequence=spacer, n_guides=1, library_pool=None
        ),
    )


def build_genotype(
    targets: Sequence[tuple[str, str]],
    spacer: str,
    background: Sequence[tuple[str, str, str]] = (),
) -> Genotype:
    """One guide's knockdowns, plus the round's constant background knockdown.

    ``background`` is empty in round 1 and holds the pcnB repression in round 2, where
    every record is a DOUBLE knockdown: the library guide and the pcnBi plasmid's.
    """
    leaves: list[GenePerturbationType] = [
        _knockdown(bnumber, gene_name, spacer) for bnumber, gene_name in targets
    ]
    leaves += [
        _knockdown(bnumber, gene_name, background_spacer)
        for bnumber, gene_name, background_spacer in background
    ]
    return Genotype(perturbations=leaves)


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class RoundAccounting(BaseModel):
    """One round's retention arithmetic and its equation residuals."""

    model_config = ConfigDict(extra="forbid")

    round_id: str
    sheet: str
    panel: str
    host_strain: str
    libraries: list[str]
    total_reads: dict[str, int]
    source_rows: int
    dropped_below_read_floor: int
    dropped_by_library: dict[str, int]
    kept_records: int
    perturbations: int
    background_knockdowns: int
    max_equation_1_error: float
    max_equation_2_error: float

    def check(self) -> None:
        """The one drop count must account for every released row."""
        kept = self.source_rows - self.dropped_below_read_floor
        if kept != self.kept_records:
            raise ValueError(
                f"{self.round_id}: {self.source_rows} rows minus "
                f"{self.dropped_below_read_floor} below the floor is {kept}, not the "
                f"{self.kept_records} records written"
            )


class BuildAccounting(BaseModel):
    """Everything a reader needs to audit one build's retention arithmetic."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    reference_strain: str
    strain_background: str
    assembly_set: str
    library_citation_key: str
    library_guides: int
    library_targeting_guides: int
    library_non_targeting_guides: int
    clusters: int
    cluster_members: int
    rounds: list[RoundAccounting]
    source_rows: int
    kept_records: int
    perturbations: int
    distinct_genes: int
    read_floor: int
    drop_rules: list[str]
    accessions_recorded_not_loaded: dict[str, str]
    notes: list[str]

    def check(self) -> None:
        """Every round's arithmetic, and the totals over the rounds."""
        for round_ in self.rounds:
            round_.check()
        totals = (
            sum(r.source_rows for r in self.rounds),
            sum(r.kept_records for r in self.rounds),
            sum(r.perturbations for r in self.rounds),
        )
        if totals != (self.source_rows, self.kept_records, self.perturbations):
            raise ValueError(
                f"per-round totals {totals} do not sum to "
                f"{(self.source_rows, self.kept_records, self.perturbations)}"
            )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class CrispriGuideFfaEnrichmentFang2025Dataset(ExperimentDataset):
    """Fang 2025 per-guide CRISPRi-FACS enrichment across the paper's two rounds."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[Literal["MG1655"]] = "MG1655"

    def __init__(
        self,
        root: str = "data/torchcell/crispri_guide_ffa_enrichment_fang2025",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize. No genome parameter: the loader reads no annotation."""
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
        """This paper's screen workbook plus Wang 2018's two library files."""
        return [raw.filename for raw in RAW_FILES] + [
            raw.filename for raw in WANG_RAW_FILES
        ]

    def download(self) -> None:
        """Link both mirrors' files into ``raw/`` after verifying each against its manifest.

        Two mirrors, each read under its OWN citation key: this paper's screen workbook
        from ``torchcell-raw/fangGenomescaleCRISPRiScreen2025``, and the guide library
        from ``torchcell-raw/wangPooledCRISPRInterference2018``, because the screen
        reuses that library id for id.
        """
        data_root = _data_root()
        os.makedirs(self.raw_dir, exist_ok=True)
        manifest = load_manifest(data_root)
        root = raw_mirror_dir(data_root)
        for raw in RAW_FILES:
            recorded = manifest_sha256(manifest, raw.relpath)
            check_manifest_pin(raw.relpath, recorded, raw.sha256)
            source = root / raw.relpath
            if not source.exists():
                raise RuntimeError(
                    f"required raw artifact missing from mirror: {source}"
                )
            link_verified(source, osp.join(self.raw_dir, raw.filename), recorded)
        wang_manifest = wang_load_manifest(data_root)
        wang_root = wang_raw_mirror_dir(data_root)
        for raw in WANG_RAW_FILES:
            recorded = wang_manifest_sha256(wang_manifest, raw.relpath)
            check_manifest_pin(raw.relpath, recorded, raw.sha256)
            source = wang_root / raw.relpath
            if not source.exists():
                raise RuntimeError(
                    f"required raw artifact missing from mirror: {source}"
                )
            link_verified(source, osp.join(self.raw_dir, raw.filename), recorded)
        log.info("Fang 2025 and Wang 2018 artifacts linked into %s", self.raw_dir)

    def _pointer(self, obj: Any, hint: str, itxn: Any) -> Any:
        """``obj``'s dump, interned exactly as ``_intern_record`` interns it."""
        holder = {"value": obj.model_dump()}
        self._maybe_intern(holder, "value", obj, hint, itxn)
        return holder["value"]

    @post_process
    def process(self) -> None:
        """Build one record per kept (guide, round) row; write LMDB."""
        pins = {raw.filename: raw.sha256 for raw in RAW_FILES}
        pins.update({raw.filename: raw.sha256 for raw in WANG_RAW_FILES})
        verify_raw_files(self.raw_dir, pins)
        clusters = read_clusters(
            osp.join(self.raw_dir, WANG_CLUSTERS_FILE.filename),
            n_clusters=N_CLUSTERS,
            n_members=N_CLUSTER_MEMBERS,
        )
        spacers = read_library(
            osp.join(self.raw_dir, WANG_LIBRARY_FILE.filename),
            clusters,
            n_targeting=N_TARGETING_GUIDES,
            n_non_targeting=N_CONTROL_GUIDES,
        )

        # Every cluster member as Wang 2018's Supplementary Data 2 releases it: the
        # b-number and the member's own symbol, no annotation read.
        targets: dict[str, tuple[tuple[str, str], ...]] = {
            token: tuple((m.bnumber, m.symbol) for m in cluster.members)
            for token, cluster in clusters.items()
        }
        background = self._background_knockdown(targets, spacers)
        genome_reference = host_reference()
        pub = publication()
        environment = screen_environment()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        accounting: list[RoundAccounting] = []
        genes: set[str] = set()
        idx = 0
        for round_ in ROUNDS:
            rows = read_round(osp.join(self.raw_dir, SCREEN_FILE.filename), round_)
            reference = build_reference(self.name, round_, genome_reference)
            leaves = background if round_.carries_background_knockdown else ()
            genotypes: dict[str, dict[str, Any]] = {}
            perturbations = 0
            with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
                env_ptr = self._pointer(
                    environment, environment.media.stated_name, itxn
                )
                ref_ptr = self._pointer(reference, reference.dataset_name, itxn)
                pub_ptr = self._pointer(pub, "publication", itxn)
                first = True
                for guide_id, fitness in tqdm(
                    zip(rows.kept["guide_id"], rows.kept["fitness"], strict=True),
                    total=len(rows.kept),
                    desc=f"{self.name}:{round_.round_id}",
                ):
                    key = str(guide_id)
                    match = GUIDE_ID_RE.match(key)
                    if match is None:
                        raise RuntimeError(f"{key!r} is not a library guide id")
                    target = targets[str(match.group("token"))]
                    if key not in genotypes:
                        genotypes[key] = build_genotype(
                            target, spacers[key], leaves
                        ).model_dump()
                    genotype = genotypes[key]
                    phenotype = guide_phenotype(round_, float(fitness))
                    record = {
                        "experiment": {
                            "experiment_type": "environment_response",
                            "dataset_name": self.name,
                            "genotype": genotype,
                            "environment": env_ptr,
                            "phenotype": phenotype.model_dump(),
                        },
                        "reference": ref_ptr,
                        "publication": pub_ptr,
                    }
                    if first:
                        self._check_record(
                            record,
                            genotype,
                            environment,
                            phenotype,
                            reference,
                            pub,
                            itxn,
                        )
                        first = False
                    txn.put(f"{idx}".encode(), pickle.dumps(record))
                    idx += 1
                    perturbations += len(target) + len(leaves)
                    genes.update(bnumber for bnumber, _ in target)
                    genes.update(bnumber for bnumber, _, _ in leaves)
            kept = len(rows.kept)
            if kept != round_.kept_records:
                raise RuntimeError(
                    f"{round_.round_id}: wrote {kept} records, the pinned bytes give "
                    f"{round_.kept_records}"
                )
            accounting.append(
                RoundAccounting(
                    round_id=round_.round_id,
                    sheet=round_.sheet,
                    panel=round_.panel,
                    host_strain=round_.host_strain,
                    libraries=list(round_.libraries),
                    total_reads=rows.total_reads,
                    source_rows=rows.source_rows,
                    dropped_below_read_floor=rows.excluded_rows,
                    dropped_by_library=rows.excluded_by_library,
                    kept_records=kept,
                    perturbations=perturbations,
                    background_knockdowns=len(leaves) * kept,
                    max_equation_1_error=rows.max_equation_1_error,
                    max_equation_2_error=rows.max_equation_2_error,
                )
            )
        env.close()
        interned_env.close()

        totals = (idx, sum(r.perturbations for r in accounting), len(genes))
        expected = (EXPECTED_RECORDS, EXPECTED_PERTURBATIONS, EXPECTED_GENES)
        if totals != expected:
            raise RuntimeError(
                f"{self.name}: wrote (records, perturbations, genes) {totals}, the "
                f"pinned bytes give {expected}"
            )
        self._write_reports(clusters, spacers, accounting, genes, background)
        log.info(
            "Wrote %d %s records over %d distinct genes", idx, self.name, len(genes)
        )

    def _background_knockdown(
        self,
        targets: Mapping[str, tuple[tuple[str, str], ...]],
        spacers: Mapping[str, str],
    ) -> tuple[tuple[str, str, str], ...]:
        """Round 2's constant pcnB repression, as ``(b-number, gene name, spacer)``.

        The guide id is the long form of the paper's ``pcnB_956``; it must be a guide of
        the released library, or the short-form resolution this build rests on is wrong
        and the build stops rather than storing a derived spacer.
        """
        guide_id = str(ROUND_TWO_BACKGROUND_GUIDE.value)
        if guide_id not in spacers:
            raise RuntimeError(
                f"{self.name}: {guide_id!r} is not in the Wang 2018 library, so the "
                "paper's short-form pcnB_956 does not resolve"
            )
        match = GUIDE_ID_RE.match(guide_id)
        if match is None:
            raise RuntimeError(f"{guide_id!r} is not a library guide id")
        target = targets[str(match.group("token"))]
        if len(target) != 1:
            raise RuntimeError(
                f"{self.name}: {guide_id!r} targets {len(target)} genes; the pcnBi "
                "background is a single-gene repression"
            )
        bnumber, gene_name = target[0]
        return ((bnumber, gene_name, spacers[guide_id]),)

    def _check_record(
        self,
        record: dict[str, Any],
        genotype: dict[str, Any],
        environment: Environment,
        phenotype: EnvironmentResponsePhenotype,
        reference: EnvironmentResponseExperimentReference,
        pub: Publication,
        itxn: Any,
    ) -> None:
        """The fast-path record equals what ``_intern_record`` writes for the same objects."""
        experiment = EnvironmentResponseExperiment(
            dataset_name=self.name,
            genotype=Genotype.model_validate(genotype),
            environment=environment,
            phenotype=phenotype,
        )
        expected = pickle.loads(self._intern_record(experiment, reference, pub, itxn))
        if expected != record:
            raise AssertionError(
                f"{self.name}: assembled record differs from _intern_record's"
            )

    def _write_reports(
        self,
        clusters: Mapping[str, GeneCluster],
        spacers: Mapping[str, str],
        rounds: list[RoundAccounting],
        genes: set[str],
        background: tuple[tuple[str, str, str], ...],
    ) -> None:
        """Write the per-round retention table and the build accounting."""
        pd.DataFrame([r.model_dump(mode="json") for r in rounds]).to_csv(
            osp.join(self.preprocess_dir, "round_retention.csv"), index=False
        )
        accounting = BuildAccounting(
            dataset=self.name,
            reference_strain=self.REFERENCE_STRAIN,
            strain_background=str(HOST_BACKGROUND.value),
            assembly_set=BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN],
            library_citation_key=str(LIBRARY_PROVENANCE.value),
            library_guides=len(spacers),
            library_targeting_guides=N_TARGETING_GUIDES,
            library_non_targeting_guides=N_CONTROL_GUIDES,
            clusters=len(clusters),
            cluster_members=sum(len(c.members) for c in clusters.values()),
            rounds=rounds,
            source_rows=sum(r.source_rows for r in rounds),
            kept_records=sum(r.kept_records for r in rounds),
            perturbations=sum(r.perturbations for r in rounds),
            distinct_genes=len(genes),
            read_floor=int(READ_FLOOR.value),
            drop_rules=DROP_RULES,
            accessions_recorded_not_loaded={
                GEO_SCREEN_ACCESSION: "the screen reads; no record reads reads",
                GEO_TRANSCRIPTOME_ACCESSION: "gene-level differential expression",
                PROTEOMEXCHANGE_ACCESSION: "gene-level differential expression",
            },
            notes=[
                "one record is one kept (guide, round) row of Supplementary Data 6; the "
                "two rounds are kept apart by phenotype.screen_id",
                "the screened set is EXACTLY Wang 2018's 55,671 targeting guides, "
                "measured id for id in both rounds, and none of its 400 non-targeting "
                "controls is released, so this build reads that library's spacers, "
                "b-numbers and clusters from the Wang raw mirror",
                "round 2 adds one constant knockdown to every genotype, the pcnBi "
                f"plasmid's {background[0][1]} repression on {background[0][0]}, "
                "because the reference class has no genotype field and cannot express "
                "the pcnBi strain",
                "round 2 keeps only 331 of 55,671 because 55,334 of its guides read "
                "below the floor after sorting, which is what sorting the top 1 percent "
                "of an already-shaped library does; it is not a parse error",
                "the 13 FFAs (mg L-1) titer blocks of Supplementary Data 6, 165 "
                "labelled rows with three replicates and an SD each, are NOT loaded: "
                "pcnBi-acrDi-fadR+ carries a P_BAD-driven fadR OVEREXPRESSION that no "
                "bacterial gene-perturbation leaf types",
                "assay_type is biosensor_readout, not pooled_competitive_growth_"
                "barcode: BS and AS are the same culture and the sorter gates on "
                "Nile-Red fluorescence, so no growth separates the two arms",
            ],
        )
        accounting.check()
        with open(
            osp.join(self.preprocess_dir, "build_accounting.json"), "w"
        ) as handle:
            json.dump(accounting.model_dump(mode="json"), handle, indent=2)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "CrispriGuideFfaEnrichmentFang2025Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification (L0-L4)
# --------------------------------------------------------------------------- #
def verify_build(
    dataset_root: str,
    *,
    genome: EcoliK12Genome | None = None,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> Any:
    """Run the environment-response L0-L4 verifier on a built tree and write its report.

    The LMDB is streamed and never materialized. Every stored name is checked against
    the MG1655 genome its references pin: the resolver of the canonical-name rule, and
    as the L4 universe every GenBank locus of the assembly. The report is written to
    ``preprocess/verification_report.json``.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    if genome is None:
        genome = bacterial_genome("ecoli", MG1655_STRAIN, data_root)
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{SCREEN_FILE.relpath}",
            citation_key=CITATION_KEY,
            sha256=SCREEN_FILE.sha256,
            method=(
                "Supplementary Data 6 (MOESM8), sheets 'Figure 1' and 'Figure 5': one "
                "EnvironmentResponseExperiment per kept (sgRNA, round) row. "
                "measurement_type=log2_ratio, assay_type=biosensor_readout (BS and AS "
                "are the same culture; the FACSAria III gates on Nile-Red fluorescence "
                "and sorts the top 1 percent), environment_response = the released "
                "'Fitness' = log2(normalized reads in AS / normalized reads in BS) with "
                "normalized reads = (reads + 1) / total reads (Methods equations 1 and "
                "2, both reproduced from the released read counts: equation 1 exactly, "
                "equation 2 to 1.8e-15). n_samples=1 pooled library per arm per round, "
                "so no dispersion exists and the uncertainty fields are typed "
                "ProvenanceGaps. Genotype = one "
                "CrisprInterferencePerturbation per member of the guide's "
                "BLASTN cluster, effector dCas9, guide_sequence = the 20-mer spacer of "
                "Wang 2018 Supplementary Data 3 (the screened set is Wang's 55,671 "
                "targeting guides, measured id for id), plus in round 2 the constant "
                "pcnB repression of the pcnBi host (pcnBb0143_956). Reference = the "
                "unsorted library of the same round, response 0.0 by that "
                f"normalization. DROPPED: 95,631 rows below the authors' own "
                f"{READ_FLOOR.value}-read floor; every cluster member's b-number is "
                "stored as released (b4700 and b4590, which the pinned annotation no "
                "longer carries, are genome-reference findings)"
            ),
            page=(
                "Nat Commun 2025 16 (doi:10.1038/s41467-025-58368-3); Methods "
                "'Genome-scale CRISPRi screen' and 'NGS library preparation and "
                "sequencing'; Supplementary Data 6; "
                f"paper.md sha256={PAPER_MD_SHA256}"
            ),
            retrieved=RAW_RETRIEVED_AT,
        ),
        expected_count=expected_count,
        sgd_genes=set(genome.genbank.loci),
        resolve_gene_name=genome.resolve_gene_name,
    )
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


# --------------------------------------------------------------------------- #
# The per-strain FFA titers: one record per biological replicate culture
#
# WHAT A RECORD IS. Supplementary Data 6 lays every quantitative panel out as
# ``Strain | 1 | 2 | 3 | Mean | SD``, and the three numbered columns are the three
# cultures behind the panel's error bar: "Tube cultivations were performed in three
# biological replicates", "Quantitative data of three biological replicates are
# presented as mean +/- SD". Measured on the pinned bytes, the mean of the three
# columns reproduces the released ``Mean`` to 9.1e-13 and their sample SD reproduces
# the released ``SD`` to 3.9e-12 on all 144 titer rows, which is what proves the
# column alignment rather than assuming it. So one record is ONE culture: the released
# value verbatim, ``n_samples=1``, ``sample_unit=biological_replicate``,
# ``replicate_id`` the panel's own labels for the culture, its strain and its column
# number ("pcnBi 1"), and the uncertainty
# fields typed ``ProvenanceGap``s with reason ``not_applicable_single_measurement``
# (#1178). The released Mean and SD are a derivation over the stored cultures and are
# kept only in ``preprocess/titer_rows.csv``.
#
# THE SAME CULTURES ARE RE-PLOTTED ACROSS PANELS, AND THE STORE HOLDS THEM ONCE. 13
# ``FFAs (mg L-1)`` blocks release 144 strain rows over 116 distinct (strain, medium)
# pairs: the CF control appears in six panels and ``pcnBi`` in seven, with the same
# three numbers each time. A record is therefore keyed by (strain, medium, replicate),
# every panel that released it is recorded in ``preprocess/titer_rows.csv``, and the
# build refuses a key whose panels disagree by more than
# :data:`TITER_REPEAT_TOLERANCE`. Two repeats disagree at all: Figure S9b prints the
# ``pcnB-`` cultures rounded to seven significant figures where Figure S3a prints them
# in full (max 3.9e-4 mg/L), and the full-precision cell is the one stored.
#
# GENOTYPES COME FROM THE SOURCE'S OWN TWO TABLES, NOT FROM A HAND-WRITTEN MAP.
# Supplementary Data 1 states each strain as a sentence ("MG with plasmids pCF-pcnBi
# and pSg-acrD_2252") and Supplementary Data 2 states each plasmid's parts ("PLlacO-1:
# sgRNA acrD_2252, pBR322, AmpR"), so the build parses the sentence, looks each plasmid
# up and turns its parts into perturbations:
#
# - ``sgRNA <guide>`` -> a ``CrisprInterferencePerturbation`` per member of that guide's
#   Wang 2018 BLASTN cluster, carrying Wang's 20-mer spacer. The panels write the
#   short form (``acrD_2252``), and it resolves to Wang's ``<symbol><bNNNN>_<position>``
#   id on (symbol, position): measured on the pinned library, no short form of the
#   55,671 is ambiguous, and all 86 guides of the plasmid table resolve.
# - ``PBAD: fadR`` / ``PT7: fadR`` -> a ``GeneAdditionPerturbation``: an extra NATIVE
#   copy on its own replicon, ``is_heterologous=False``, the plasmid's own words in
#   ``replicon_statement``, and ``placement`` a typed gap (no construct sequence is
#   deposited). This is the leaf the earlier decline said did not exist.
# - ``MG1655(DE3) dfadE dpcnB`` -> a ``DeletionStrainPerturbation`` on pcnB, the
#   lambda-RED deletion the Methods state, with no cassette asserted.
#
# TWO BACKGROUNDS, BECAUSE THE pC FAMILY CARRIES NO tesA'. ``pCF`` is "PTrc: dCas9,
# PT7: tesA', p15A, CmR" and ``pC`` is "PTrc: dCas9, p15A, CmR": the ``CF-``,
# ``pcnBiF-`` and ``pcnB-F-`` strains of Figure S9b are the same host without the
# thioesterase, which is a different ``StrainBackground`` and a different reference.
# Each record points at the reference of ITS background: the released CF mean (746.8201
# mg/L, SD 20.4727, n = 3) for the pCF family and the released CF- mean (86.33773,
# SD 21.1876) for the pC family, both read out of the same panels and both with
# ``replicate_id=None``, since a reference states an aggregate. The three CF cultures
# and the three CF- cultures are records as well.
#
# THE PRODUCT IS A MIXTURE AND IS STORED AS ONE. "Total concentrations of FFAs were
# calculated as the sum of C12 to C18 (saturated and monounsaturated)", so no InChIKey
# exists for what was measured: ``resolved_compound("free fatty acids")`` returns the
# name with a typed gap on the structure, and the per-species titers a sibling dataset
# can hold (Xue 2025's five FFAs) are not what this release prints.
#
# WHAT IS NOT LOADED, AND WHY (42 of the 432 released values):
#
# - **The 10 alternate-host rows of Figures 4 and S11** (JM109(DE3), AD494(DE3),
#   BL21-trxB(DE3), OrigamiB(DE3), each as -CF and -pcnBi). Their genome is not a
#   deposited assembly: ``BacterialReferenceStrain`` has MG1655, BW25113, W3110, REL606
#   and KT2440, and a record needs the assembly its reference pins. Loading them needs
#   four new reference strains with deposited assemblies, which is a schema decision.
# - **The 2 glucose rows of Figure 4** (CF and pcnBi on "30 g/L glucose, specifically
#   used in Fig. 4"). That is a second medium, and ``torchcell.datamodels.media`` holds
#   only the glycerol recipe; a titer stored against the glycerol medium would state
#   the wrong carbon source.
# - **The 2 pcnB point-mutant rows of Figure S3a** (``pcnBmE`` = pcnB E108A,
#   ``pcnBmN`` = pcnB N138H). The allele name is the whole of what distinguishes the
#   two strains, and which leaf carries a named bacterial point allele
#   (``NamedAllelePerturbation``, documented for yeast, versus a site variant call) is
#   an owner decision rather than something the source settles.
#
# Nothing else is declined, and no released culture is averaged: the authors state "no
# data was excluded from the analysis" and withdraw no sample.
# --------------------------------------------------------------------------- #
_PAGE_METHODS_FERMENTATION = "Methods, Cultivation and fermentation"
_PAGE_METHODS_METABOLITE = "Methods, Metabolite extraction and analysis"
_PAGE_STATISTICS = "Statistics and reproducibility"
_PAGE_RESULTS_REVERSE = "Results, reverse engineering of the screen candidates"
_PAGE_RESULTS_FED_BATCH = "Results, FFAs production by fed-batch fermentation"

_Q_TUBE_REPLICATES = "Tube cultivations were performed in three biological replicates."
_Q_FLASK_REPLICATES = (
    "Flask cultivations followed the same procedures as glass-tube fermentations and "
    "were performed in three biological replicates."
)
_Q_STATISTICS = (
    "Quantitative data of three biological replicates are presented as mean $\\pm "
    "\\mathsf { S D }$ , as indicated in legends of figures."
)
_Q_NO_EXCLUSION = (
    "No statistical method was used to predetermine sample size, and no data was "
    "excluded from the analysis."
)
_Q_FFA_TOTAL = (
    "Total concentrations of FFAs were calculated as the sum of $\\mathbf { C } _ { 1 2 "
    "}$ to ${ \\sf C } _ { 1 8 }$ (saturated and monounsaturated)."
)
_Q_GC = (
    "The extracted FFAs were then determined using a Thermo Scientific TRACE 1300 gas "
    "chromatograph (GC) equipped with a TG-WaxMS A column $( 3 0 \\mathrm { m } \\times "
    "0 . 3 2 \\mathrm { m m } \\times 0 . 2 5 \\mu \\mathrm { m } )$ Thermo Fisher "
    "Scientific, USA)"
)
_Q_TUBE_FERMENTATION = (
    "Glass-tube fermentations were carried out in $5 \\mathsf { m l }$ of modified M9 "
    "medium."
)
_Q_ARABINOSE = (
    "When the $\\mathsf { P } _ { \\mathsf { B A D } }$ promoter was used, $1 0 \\mathrm "
    "{ m M }$ arabinose was utilized to induce gene expression"
)
_Q_CF_TITER = (
    "The CF strain, which did not undergo any genome modulation, produced $7 4 7 "
    "\\mathrm { m g L ^ { - 1 } }$ FFAs (Fig. 1e)."
)
_Q_FADR_OVEREXPRESSION = (
    "The results showed that the strain with fadR controlled by the $\\mathsf { P } _ { "
    "\\mathsf { B A D } }$ promoter (pcnBi-acrDi-fadR+ ) produced $5 1 0 5 \\mathrm { m "
    "g L ^ { - 1 } }$ FFAs"
)
_Q_GLUCOSE_MEDIUM = (
    "glycerol (or $3 0 { \\bf g } { \\bf L } ^ { - 1 }$ glucose, specifically used in "
    "Fig. 4)"
)
_Q_ALTERNATE_HOSTS = (
    "pcnB repression contributed to an increase of FFAs titer by $300 \\%$ , $53 \\%$ "
    ", $1 4 9 \\%$ , $1 9 9 \\%$ , and $2 6 4 \\%$ in MG1655(DE3) ΔfadE, "
    "JM109(DE3), AD494(DE3), BL21-trxB(DE3), and origami(DE3), respectively"
)
_Q_POINT_MUTANTS = (
    "we also constructed the pcnB deletion strain (ΔpcnB) and pcnB mutation "
    "strains (E108A and N138H) with decreased activity relative to the wild type25."
)

TITER_N_SAMPLES = _paper(
    1,
    _Q_STATISTICS,
    page=_PAGE_STATISTICS,
    note="one record is ONE of the three biological replicate cultures the panel "
    f"releases ('{_Q_TUBE_REPLICATES}'); the released mean and SD reproduce from the "
    "three stored values to 9.1e-13 and 3.9e-12 over all 144 rows, which is what fixes "
    "the column alignment",
)
TITER_SAMPLE_UNIT = _paper(
    "biological_replicate",
    _Q_TUBE_REPLICATES,
    page=_PAGE_METHODS_FERMENTATION,
    note=f"flask cultivations state the same design ('{_Q_FLASK_REPLICATES}')",
)
TITER_QUANTIFICATION = _paper(
    "GC-FID",
    _Q_GC,
    page=_PAGE_METHODS_METABOLITE,
    note="a Flame Ionization Detector against a heptadecanoic-acid internal standard; "
    "the figures' axis label is FFAs (mg L-1)",
)
TITER_PRODUCT = _paper(
    "free fatty acids",
    _Q_FFA_TOTAL,
    page=_PAGE_METHODS_METABOLITE,
    note="the measured quantity is a SUM over species, so no single structure answers "
    "to it and the compound layer returns the name with a typed gap on the InChIKey",
)
ARABINOSE_MM = _paper(
    10.0,
    _Q_ARABINOSE,
    page=_PAGE_METHODS_FERMENTATION,
    note="the inducer of the P_BAD fadR addition, and of nothing else in this dataset; "
    "the T7-driven fadR plasmid answers to the IPTG every record already carries",
)
NO_SAMPLE_WITHDRAWN = _paper(
    0,
    _Q_NO_EXCLUSION,
    page=_PAGE_STATISTICS,
    note="the authors withdraw no culture, so the only values this build does not "
    "store are the ones whose strain or medium the schema cannot state",
)

#: Product name, which the family verifier pins per dataset.
TITER_PRODUCT_NAME = "free fatty acids"
#: Agreement required between two panels that release the same culture: the rounding
#: of Figure S9b's ``pcnB-`` cells is 3.9e-4 mg/L, and anything larger is a different
#: measurement and stops the build.
TITER_REPEAT_TOLERANCE = 1e-3
#: Tolerance of the released-mean and released-SD reproduction over the three columns.
TITER_SUMMARY_TOLERANCE = 1e-9
#: Tolerance of the derived ``titer_se`` identity the family verifier checks.
TITER_SE_TOL = 1e-9

#: ``MG``, the host every stored strain is built from (Supplementary Data 1).
_HOST_MG = "MG"
#: The deletion host of the pcnB-deletion strains, as Supplementary Data 1 writes it.
_HOST_MG_DPCNB = "MG1655(DE3) ΔfadEΔpcnB"
#: The two background plasmids: with and without the tesA' thioesterase.
_PLASMID_PCF = "pCF"
_PLASMID_PC = "pC"

#: Strains whose host genome has no deposited assembly (Figure 4 / S11).
TITER_ALTERNATE_HOST_STRAINS: frozenset[str] = frozenset(
    {"JM-CF", "JM-pcnBi", "AD-CF", "AD-pcnBi", "BL-CF", "BL-pcnBi", "OB-CF", "OB-pcnBi"}
)
#: Strains carrying a named pcnB point allele (Figure S3a).
TITER_POINT_ALLELE_STRAINS: frozenset[str] = frozenset({"pcnBmE", "pcnBmN"})
#: The medium of the two Figure 4 rows this build does not store.
TITER_GLUCOSE_MEDIUM = "glucose"
#: The medium of every stored record, the one ``M9_MODIFIED_FANG2025`` states.
TITER_GLYCEROL_MEDIUM = "glycerol"


class TiterBlock(BaseModel):
    """One ``FFAs (mg L-1)`` block of Supplementary Data 6, by its measured geometry.

    ``header_row`` is the 0-based row holding ``Strain`` and the replicate headers
    ``1 2 3 Mean SD``; ``strain_column`` the column of the strain label and
    ``value_column`` the column of replicate 1. ``rows`` is the measured number of
    strain rows, asserted by the reader so a re-released workbook cannot grow or shrink
    a block silently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sheet: str
    panel: str
    header_row: int
    strain_column: int
    value_column: int
    rows: int
    substrate_column: int | None = None


#: Every ``FFAs (mg L-1)`` block of the workbook, measured on the pinned bytes: 13
#: blocks, 144 strain rows, 432 released culture values.
TITER_BLOCKS: tuple[TiterBlock, ...] = (
    TiterBlock(
        sheet="Figure 1",
        panel="Figure 1e",
        header_row=2,
        strain_column=23,
        value_column=24,
        rows=47,
    ),
    TiterBlock(
        sheet="Figure 1",
        panel="Figure 1g",
        header_row=2,
        strain_column=39,
        value_column=40,
        rows=9,
    ),
    TiterBlock(
        sheet="Figure 1",
        panel="Figure 1h",
        header_row=15,
        strain_column=39,
        value_column=40,
        rows=2,
    ),
    TiterBlock(
        sheet="Figure 4",
        panel="Figure 4",
        header_row=2,
        strain_column=2,
        value_column=3,
        rows=14,
        substrate_column=1,
    ),
    TiterBlock(
        sheet="Figure 5",
        panel="Figure 5d",
        header_row=2,
        strain_column=21,
        value_column=22,
        rows=37,
    ),
    TiterBlock(
        sheet="Figure 5",
        panel="Figure 5f",
        header_row=13,
        strain_column=30,
        value_column=31,
        rows=4,
    ),
    TiterBlock(
        sheet="Figure 5",
        panel="Figure 5g",
        header_row=21,
        strain_column=30,
        value_column=31,
        rows=2,
    ),
    TiterBlock(
        sheet="Figure 6",
        panel="Figure 6c",
        header_row=17,
        strain_column=1,
        value_column=2,
        rows=4,
    ),
    TiterBlock(
        sheet="Figure 7",
        panel="Figure 7a",
        header_row=2,
        strain_column=1,
        value_column=2,
        rows=3,
    ),
    TiterBlock(
        sheet="Figure S2",
        panel="Figure S2b",
        header_row=2,
        strain_column=1,
        value_column=7,
        rows=7,
    ),
    TiterBlock(
        sheet="Figure S3",
        panel="Figure S3a",
        header_row=2,
        strain_column=1,
        value_column=2,
        rows=4,
    ),
    TiterBlock(
        sheet="Figure S9",
        panel="Figure S9b",
        header_row=2,
        strain_column=1,
        value_column=2,
        rows=6,
    ),
    TiterBlock(
        sheet="Figure S12",
        panel="Figure S12",
        header_row=2,
        strain_column=1,
        value_column=2,
        rows=5,
    ),
)

#: Measured on the pinned bytes.
TITER_SOURCE_ROWS = 144
TITER_SOURCE_VALUES = 432
EXPECTED_TITER_CULTURES = 102
EXPECTED_TITER_RECORDS = 306


class TiterRow(BaseModel):
    """One released strain row of one panel: its three cultures, its mean and its SD."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    panel: str
    sheet: str
    row: int
    strain: str
    medium: str
    replicates: tuple[float, float, float]
    released_mean: float
    released_sd: float


def read_titer_blocks(path: str | Path) -> list[TiterRow]:
    """Every ``FFAs (mg L-1)`` row of Supplementary Data 6, with both summaries checked.

    Each block's header is matched before a number is read (``Strain``, then ``1 2 3
    Mean SD``), its row count is asserted against the pinned geometry, and the released
    ``Mean`` and ``SD`` are recomputed from the three columns. A summary that does not
    reproduce stops the build: the three columns would then not be the replicates the
    panel averaged, and a record would claim a culture it is not.
    """
    import statistics

    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows: list[TiterRow] = []
        for block in TITER_BLOCKS:
            sheet = workbook[block.sheet]
            # read_only sheets stream, so take the window this block needs once
            window = [
                row
                for index, row in enumerate(sheet.iter_rows(values_only=True))
                if block.header_row <= index <= block.header_row + block.rows
            ]
            header = window[0]
            if str(header[block.strain_column]).strip() != "Strain":
                raise ValueError(
                    f"{block.panel}: column {block.strain_column} of row "
                    f"{block.header_row} is {header[block.strain_column]!r}, not 'Strain'"
                )
            labels = [header[block.value_column + offset] for offset in range(5)]
            if [str(label).strip() for label in labels] != [
                "1",
                "2",
                "3",
                "Mean",
                "SD",
            ]:
                raise ValueError(f"{block.panel}: replicate headers are {labels}")
            substrate = TITER_GLYCEROL_MEDIUM
            for offset in range(block.rows):
                line = window[offset + 1]
                strain = line[block.strain_column]
                if strain is None:
                    raise ValueError(
                        f"{block.panel}: row {block.header_row + offset + 1} has no "
                        "strain label, the pinned geometry says it has one"
                    )
                if block.substrate_column is not None:
                    stated = line[block.substrate_column]
                    if stated is not None:
                        substrate = str(stated).strip().lower()
                values = tuple(
                    float(line[block.value_column + position]) for position in range(3)
                )
                mean = float(line[block.value_column + 3])
                deviation = float(line[block.value_column + 4])
                if abs(sum(values) / 3.0 - mean) > TITER_SUMMARY_TOLERANCE:
                    raise ValueError(
                        f"{block.panel} {strain}: the three columns mean "
                        f"{sum(values) / 3.0!r}, the released Mean is {mean!r}"
                    )
                if abs(statistics.stdev(values) - deviation) > TITER_SUMMARY_TOLERANCE:
                    raise ValueError(
                        f"{block.panel} {strain}: the three columns' sample SD is "
                        f"{statistics.stdev(values)!r}, the released SD is {deviation!r}"
                    )
                rows.append(
                    TiterRow(
                        panel=block.panel,
                        sheet=block.sheet,
                        row=block.header_row + offset + 1,
                        strain=str(strain).strip(),
                        medium=substrate,
                        replicates=values,  # type: ignore[arg-type]
                        released_mean=mean,
                        released_sd=deviation,
                    )
                )
    finally:
        workbook.close()
    if len(rows) != TITER_SOURCE_ROWS:
        raise ValueError(
            f"read {len(rows)} titer rows, the pinned bytes give {TITER_SOURCE_ROWS}"
        )
    return rows


def _read_two_column_table(path: str | Path, title_prefix: str) -> dict[str, str]:
    """``{name: description}`` of one ``Supplementary Data`` table, verbatim.

    The three tables share one layout: an empty first column, a title row, a header row
    and then ``name | description | reference``. The title is matched so a swapped file
    cannot be read as the wrong table.
    """
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook[workbook.sheetnames[0]]
        table: dict[str, str] = {}
        title_seen = False
        for row in sheet.iter_rows(min_col=2, max_col=3, values_only=True):
            name, description = row[0], row[1]
            if name is None:
                continue
            label = str(name).strip()
            if not title_seen:
                if label.startswith(title_prefix):
                    title_seen = True
                continue
            if description is None:
                continue
            table[label] = str(description).strip()
    finally:
        workbook.close()
    if not title_seen:
        raise ValueError(f"{path} does not carry a {title_prefix!r} title row")
    return table


def read_strain_table(path: str | Path) -> dict[str, str]:
    """Supplementary Data 1: ``{strain: description}``, each sentence verbatim."""
    return _read_two_column_table(path, "Supplementary Data 1.")


def read_plasmid_table(path: str | Path) -> dict[str, str]:
    """Supplementary Data 2: ``{plasmid: parts}``, each description verbatim."""
    return _read_two_column_table(path, "Supplementary Data 2.")


def read_primer_table(path: str | Path) -> dict[str, str]:
    """Supplementary Data 3: ``{primer: sequence}``, the spacer proof's own bytes."""
    return _read_two_column_table(path, "Supplementary Data 3.")


#: ``MG with plasmids pCF-pcnBi and pSg-acrD_2252`` -> the host and its plasmid list.
_STRAIN_RE = re.compile(r"^(?P<host>.+?) with plasmids? (?P<plasmids>.+)$")
#: A plasmid name inside a strain sentence.
_PLASMID_SPLIT_RE = re.compile(r"\s+and\s+")
#: ``PLlacO-1: sgRNA pcnB_956`` inside a plasmid description.
_GUIDE_RE = re.compile(r"sgRNA (?P<guide>[A-Za-z0-9]+_\d+)")
#: ``PBAD: fadR`` inside a plasmid description, an added native copy under a stated
#: promoter. It is read only on a NON-background plasmid, so ``PTrc: dCas9`` and
#: ``PT7: tesA'`` (the two parts every background plasmid carries) are never an
#: addition: those are the host, not a per-strain edit.
_ADDITION_RE = re.compile(
    r"(?P<promoter>P[A-Za-z0-9]+): (?P<gene>[A-Za-z][A-Za-z0-9]*)(?:,|$)"
)
#: ``MG1655(DE3) dfadE dpcnB`` -> the genes the host itself deletes.
_HOST_DELETION_RE = re.compile(r"\u0394(?P<gene>[A-Za-z][A-Za-z0-9]*)")
#: The background's own lesion, which lives on the ``StrainBackground`` and is not a
#: per-record perturbation.
_BACKGROUND_DELETION = "fadE"
#: A short-form guide name: ``<symbol>_<position>``.
_SHORT_GUIDE_RE = re.compile(r"^(?P<symbol>[A-Za-z0-9]+)_(?P<position>\d+)$")


class GeneAddition(BaseModel):
    """One added native gene: its symbol, its promoter and the plasmid carrying it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gene: str
    promoter: str
    plasmid: str
    plasmid_statement: str


class StrainSpec(BaseModel):
    """One strain as Supplementary Data 1 and 2 state it, resolved into its edits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: str
    statement: str
    host: str
    background_plasmid: str
    background_statement: str
    guides: tuple[str, ...]
    additions: tuple[GeneAddition, ...]
    deletions: tuple[str, ...]

    @property
    def carries_thioesterase(self) -> bool:
        """Whether the background plasmid expresses tesA' (``pCF``) or not (``pC``)."""
        return self.background_plasmid != _PLASMID_PC


def strain_spec(
    strain: str, strains: Mapping[str, str], plasmids: Mapping[str, str]
) -> StrainSpec:
    """Resolve one strain into its background, guides, additions and deletions.

    The release spells three things loosely and the lookups are case-insensitive for
    exactly that reason: the titer panels write ``folK_38`` where Supplementary Data 1
    writes ``folk_38``, and Supplementary Data 1's ``paaG_218`` row names its plasmid
    ``pSg-aaG_218`` where Supplementary Data 2 carries ``pSg-paaG_218`` (a dropped
    ``p``). A plasmid a strain names and the plasmid table does not carry stops the
    build unless the strain's own name completes it, which is what that one typo needs.
    """
    folded = {name.lower(): name for name in strains}
    key = folded.get(strain.lower())
    if key is None:
        raise ValueError(f"Supplementary Data 1 states no strain {strain!r}")
    statement = strains[key]
    match = _STRAIN_RE.match(statement)
    if match is None:
        raise ValueError(f"{strain!r}: cannot read the construction {statement!r}")
    host = match.group("host").strip()
    if host not in {_HOST_MG, _HOST_MG_DPCNB}:
        raise ValueError(f"{strain!r}: unexpected host {host!r}")
    plasmid_folded = {name.lower(): name for name in plasmids}
    names: list[str] = []
    for token in _PLASMID_SPLIT_RE.split(match.group("plasmids")):
        candidate = token.strip()
        resolved = plasmid_folded.get(candidate.lower())
        if resolved is None:
            # the ``pSg-aaG_218`` typo: the plasmid of a single-guide strain is named
            # after the strain, and that is the only completion this takes
            resolved = plasmid_folded.get(f"psg-{strain.lower()}")
        if resolved is None:
            raise ValueError(
                f"{strain!r}: Supplementary Data 2 states no plasmid {candidate!r}"
            )
        names.append(resolved)
    background = [name for name in names if name.startswith(_PLASMID_PCF)] or [
        name for name in names if name == _PLASMID_PC
    ]
    if len(background) != 1:
        raise ValueError(
            f"{strain!r}: plasmids {names} name no single dCas9 background plasmid"
        )
    guides: list[str] = []
    additions: list[GeneAddition] = []
    for name in names:
        description = plasmids[name]
        guides.extend(found for found in _GUIDE_RE.findall(description))
        if name == background[0]:
            continue
        for addition in _ADDITION_RE.finditer(description):
            additions.append(
                GeneAddition(
                    gene=addition.group("gene"),
                    promoter=addition.group("promoter"),
                    plasmid=name,
                    plasmid_statement=description,
                )
            )
    deletions = tuple(
        gene for gene in _HOST_DELETION_RE.findall(host) if gene != _BACKGROUND_DELETION
    )
    return StrainSpec(
        strain=strain,
        statement=statement,
        host=host,
        background_plasmid=background[0],
        background_statement=plasmids[background[0]],
        # duplicates are KEPT: pcnBi-pcnB_956 carries the same guide on two plasmids,
        # which is two knockdown leaves of one gene and not a parse error
        guides=tuple(guides),
        additions=tuple(additions),
        deletions=deletions,
    )


def resolve_short_guide(guide: str, spacers: Mapping[str, str]) -> str:
    """Fang's short guide name as Wang 2018's library id, by (symbol, position).

    ``acrD_2252`` -> ``acrDb2470_2252``. The map is built over the whole released
    library and refuses an ambiguous short form rather than picking one: measured on
    the pinned bytes, none of the 55,671 ids shares a (symbol, position) with another.
    """
    match = _SHORT_GUIDE_RE.match(guide)
    if match is None:
        raise ValueError(f"{guide!r} is not a short-form guide name")
    wanted = (match.group("symbol").lower(), match.group("position"))
    hits = []
    for candidate in spacers:
        parsed = GUIDE_ID_RE.match(candidate)
        if parsed is None:
            continue
        if (parsed.group("symbol").lower(), parsed.group("position")) == wanted:
            hits.append(candidate)
    if len(hits) != 1:
        raise ValueError(
            f"{guide!r} resolves to {hits} in the Wang 2018 library, not to one id"
        )
    return hits[0]


def check_spacer_against_primer(
    guide: str, spacer: str, primers: Mapping[str, str]
) -> str:
    """Fang's own primer for one guide, checked against Wang's spacer.

    Supplementary Data 3 names each spacer primer ``<guide>-for`` or ``<guide>-rev``
    and prints a 4-base cloning tag followed by the spacer or its reverse complement.
    Returns the primer name that matched; a guide whose primer matches neither stops
    the build, because the short-form resolution would then be a guess.
    """
    folded = {name.lower(): name for name in primers}
    for suffix in ("-for", "-rev"):
        name = folded.get(f"{guide.lower()}{suffix}")
        if name is None:
            continue
        tail = primers[name][4:].upper()
        complement = spacer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        if tail in {spacer, complement}:
            return name
    raise ValueError(
        f"{guide!r}: Supplementary Data 3 carries no primer whose 20-mer is the Wang "
        f"spacer {spacer!r} or its reverse complement"
    )


def tesa_free_background() -> StrainBackground:
    """The ``pC`` background: the CF host without the tesA' thioesterase.

    Supplementary Data 2 states ``pC`` as "PTrc: dCas9, p15A, CmR", which is ``pCF``
    minus its ``PT7: tesA'``, and the paper calls the three strains carrying it
    tesA'-free. ``alleles`` is empty for the same reason it is on
    :func:`host_background`: the fadE lesion carries no locus tag anywhere in the
    mirrored bytes.
    """
    return StrainBackground(
        name="MG with plasmid pC",
        reference_strain=MG1655_STRAIN,
        assembly_set=MG1655_ASSEMBLY_SET,
        parents=["E. coli MG1655(DE3)"],
        construction="lambda-RED deletion of fadE in MG1655(DE3), then transformation "
        "with pC (dCas9 under PTrc, p15A, CmR; no tesA')",
        genotype_statement="MG with plasmid pC",
        alleles=[],
        provenance=[
            _paper("CF-", _Q_LIBRARY, page=_PAGE_RESULTS_SCREEN),
            _paper("fadE deletion", _Q_FADE_DELETION, page=_PAGE_METHODS_MEDIUM),
        ],
    )


def titer_reference_genome(
    carries_thioesterase: bool, data_root: str | None = None
) -> ReferenceGenome:
    """MG1655 pinned to its assembly, carrying the background a record was built in."""
    background = host_background() if carries_thioesterase else tesa_free_background()
    return assembly_reference(MG1655_STRAIN, background=background, data_root=data_root)


def _arabinose() -> SmallMoleculePerturbation:
    """The 10 mM arabinose that induces the P_BAD fadR addition."""
    return SmallMoleculePerturbation(
        compound=resolved_compound("arabinose"),
        concentration=Concentration(
            value=float(ARABINOSE_MM.value), unit=ConcentrationUnit.millimolar
        ),
        solvent=None,
    )


def titer_environment(arabinose: bool = False) -> CultureEnvironment:
    """The fermentation every titer record shares, plus arabinose where P_BAD is used.

    ``culture_format`` is a typed gap, not a guess: the Methods describe glass-tube (5
    mL) and shake-flask (50 mL) fermentations in the same medium at the same
    temperature, induction and duration, and the only panel whose vessel the paper
    states is Figure 7a ("in tube fermentation"), whose control row carries the same
    three values as Figures 5g and 6c.
    """
    perturbations: list[Any] = [_inducer()]
    if arabinose:
        perturbations.append(_arabinose())
    return CultureEnvironment(
        media=M9_MODIFIED_FANG2025,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_hours=float(SCREEN_HOURS.value),
        culture_format=None,
        provenance_gaps=[
            ProvenanceGap(
                field="culture_format",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=Provenance(
                    source_uri=PAPER_MD,
                    citation_key=CITATION_KEY,
                    sha256=PAPER_MD_SHA256,
                    method="Methods, Cultivation and fermentation; every figure legend",
                    page=_PAGE_METHODS_FERMENTATION,
                ),
                note="the Methods state two formats for the same medium, temperature, "
                f"induction and duration ('{_Q_TUBE_FERMENTATION}' and "
                f"'{_Q_FLASK_REPLICATES}') and no titer panel but Figure 7a says which "
                "one it used",
            )
        ],
    )


def _titer_product() -> Any:
    """The product as a typed ``Compound``: a named mixture with a gapped structure."""
    return resolved_compound(TITER_PRODUCT_NAME)


_TITER_SINGLE_CULTURE_NOTE = (
    "this record is ONE of the panel's three biological replicate cultures, so it has "
    "no dispersion of its own; the released mean and SD over the three are a "
    "derivation over the three stored records and are kept in preprocess/titer_rows.csv"
)
_TITER_NOT_RELEASED_NOTE = (
    "no yield on substrate and no volumetric productivity is released for any tube or "
    "flask titer; the only ones the paper states are the fed-batch run's, a different "
    "fermentation this dataset does not hold"
)


def _titer_gaps(single_culture: bool) -> list[ProvenanceGap]:
    """The typed absences of one titer phenotype."""
    reason = (
        ProvenanceGapReason.not_applicable_single_measurement
        if single_culture
        else ProvenanceGapReason.not_reported_by_primary
    )
    gaps = [
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=_TITER_NOT_RELEASED_NOTE,
        )
        for field in (
            "product_yield",
            "product_yield_unit",
            "productivity",
            "productivity_unit",
        )
    ]
    if single_culture:
        gaps = [
            ProvenanceGap(field=field, reason=reason, note=_TITER_SINGLE_CULTURE_NOTE)
            for field in ("titer_uncertainty", "titer_uncertainty_type")
        ] + gaps
    return gaps


def titer_phenotype(titer_mg_per_l: float, replicate_id: str) -> ProductTiterPhenotype:
    """One culture's released FFA titer, stored verbatim.

    The release prints mg/L and ``ConcentrationUnit`` has no mg/L member; 1 mg/L is
    exactly 1 ug/mL, so the number is stored verbatim under the numerically identical
    unit, as the sibling titer loaders do.

    ``replicate_id`` is the panel's own two labels for the culture, the row's strain
    and the column's number ("pcnBi 1"), and the strain half is load-bearing rather
    than decorative: the schema states a guide without the plasmid carrying it, so
    ``pcnB_956`` (pCF + pSg-pcnB_956), ``pcnBi`` (the single pCF-pcnBi plasmid) and
    ``pcnBiF-`` (pC + pSg-pcnB_956) all carry ONE pcnB knockdown and differ only in
    plasmid architecture and background. Their released titers differ, so a bare "1"
    would key three different cultures the same way.
    """
    return ProductTiterPhenotype(
        product=_titer_product(),
        titer=titer_mg_per_l,
        titer_unit=ConcentrationUnit.ug_per_ml,
        n_samples=int(TITER_N_SAMPLES.value),
        sample_unit=SampleUnit.biological_replicate,
        replicate_id=replicate_id,
        quantification_method=str(TITER_QUANTIFICATION.value),
        provenance_gaps=_titer_gaps(single_culture=True),
    )


def titer_reference_phenotype(
    mean: float, deviation: float, n_samples: int
) -> ProductTiterPhenotype:
    """The control strain's released mean and SD, the aggregate a reference states."""
    return ProductTiterPhenotype(
        product=_titer_product(),
        titer=mean,
        titer_unit=ConcentrationUnit.ug_per_ml,
        titer_uncertainty=deviation,
        titer_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=n_samples,
        sample_unit=SampleUnit.biological_replicate,
        replicate_id=None,
        quantification_method=str(TITER_QUANTIFICATION.value),
        provenance_gaps=_titer_gaps(single_culture=False),
    )


def _added_gene(addition: GeneAddition, bnumber: str) -> GeneAdditionPerturbation:
    """One plasmid-borne extra copy of a native gene, with its promoter in the words
    the plasmid table uses.

    ``GeneAdditionPerturbation`` is the leaf for "an extra native copy, carried on a
    plasmid": the gene is PRESENT in the chromosome and a second copy arrives on its
    own replicon. ``placement`` is a typed gap because no construct sequence is
    deposited anywhere in the mirror, and the promoter lives in
    ``replicon_statement`` verbatim rather than in a field this leaf does not have,
    which is what keeps the P_BAD and the P_T7 fadR strains distinct.
    """
    return GeneAdditionPerturbation(
        systematic_gene_name=bnumber,
        perturbed_gene_name=addition.gene,
        gene_namespace=MG1655_NAMESPACE,
        identifier_mapping=None,
        source_organism="Escherichia coli",
        is_heterologous=False,
        localization=Localization.episomal_replicon,
        replicon_statement=addition.plasmid_statement,
        construct_name=addition.plasmid,
        placement=None,
        provenance_gaps=[
            ProvenanceGap(
                field="placement",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=Provenance(
                    source_uri=STRAIN_FILE.relpath,
                    citation_key=CITATION_KEY,
                    sha256=STRAIN_FILE.sha256,
                    method="Supplementary Data 1 and 2, the strain and plasmid tables",
                    page="Supplementary Data 2",
                ),
                note="the plasmid is stated only as its parts "
                f"('{addition.plasmid_statement}'); no construct sequence is deposited "
                "in the mirror and no accession is given for it",
            )
        ],
    )


def _host_deletion(gene: str, bnumber: str) -> DeletionStrainPerturbation:
    """One lambda-RED deletion the host sentence states, beyond the background's fadE.

    The only one the release carries is the pcnB of the ``pcnB-`` and ``pcnB-F-``
    strains ("MG1655(DE3) dfadE dpcnB with plasmid pCF"), and the gene is read from
    that sentence rather than named here.
    """
    return DeletionStrainPerturbation(
        systematic_gene_name=bnumber,
        perturbed_gene_name=gene,
        gene_namespace=MG1655_NAMESPACE,
        identifier_mapping=None,
        description="Gene deletion strain: the lambda-RED deletion the Methods state "
        f"('{_Q_FADE_DELETION}'), with no cassette asserted",
        collection=None,
        cassette=None,
        construction=None,
    )


class TiterStrainRecord(BaseModel):
    """One stored culture, with everything the preprocess ledger shows for it."""

    model_config = ConfigDict(extra="forbid")

    strain: str
    medium: str
    replicate_id: str
    titer_mg_per_l: float
    panels: list[str]
    released_mean: float
    released_sd: float
    statement: str
    background_plasmid: str
    guides: list[str]
    guide_ids: list[str]
    primers: list[str]
    additions: list[str]
    deletions: list[str]
    perturbations: int


class TiterAccounting(BaseModel):
    """What the titer build read, kept and declined, with the reason for each decline."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    source_blocks: int
    source_rows: int
    source_values: int
    distinct_cultures: int
    kept_records: int
    declined_values: int
    repeated_release_values: int
    declined: dict[str, list[str]]
    max_repeat_disagreement: float
    repeated_keys: int
    references: dict[str, dict[str, float]]
    notes: list[str]

    def check(self) -> None:
        """Every released value is a record, a named decline or a repeat of a record.

        A repeat is a cell a second panel prints for a culture this store already
        holds (the CF control is re-plotted in six panels), so the three sums have to
        close on the 432 released cells for the ledger to be auditable.
        """
        total = self.kept_records + self.declined_values + self.repeated_release_values
        if total != self.source_values:
            raise ValueError(
                f"{self.kept_records} stored + {self.declined_values} declined + "
                f"{self.repeated_release_values} repeated is not the "
                f"{self.source_values} released culture values"
            )
        if self.kept_records != 3 * self.distinct_cultures:
            raise ValueError(
                f"{self.kept_records} records over {self.distinct_cultures} cultures is "
                "not three replicates each"
            )


@register_dataset
class FfaTiterFang2025Dataset(ExperimentDataset):
    """Fang 2025 per-strain FFA titers, one record per biological replicate culture."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[Literal["MG1655"]] = "MG1655"

    def __init__(
        self,
        root: str = "data/torchcell/ffa_titer_fang2025",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize. No genome parameter: the loader reads no annotation."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProductTiterExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProductTiterExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The source-data workbook, the three tables, and Wang 2018's library."""
        return [raw.filename for raw in TITER_RAW_FILES] + [
            raw.filename for raw in WANG_RAW_FILES
        ]

    def download(self) -> None:
        """Link both mirrors' files into ``raw/`` after verifying each against its manifest."""
        data_root = _data_root()
        os.makedirs(self.raw_dir, exist_ok=True)
        manifest = load_manifest(data_root)
        root = raw_mirror_dir(data_root)
        for raw in TITER_RAW_FILES:
            recorded = manifest_sha256(manifest, raw.relpath)
            check_manifest_pin(raw.relpath, recorded, raw.sha256)
            source = root / raw.relpath
            if not source.exists():
                raise RuntimeError(
                    f"required raw artifact missing from mirror: {source}"
                )
            link_verified(source, osp.join(self.raw_dir, raw.filename), recorded)
        wang_manifest = wang_load_manifest(data_root)
        wang_root = wang_raw_mirror_dir(data_root)
        for raw in WANG_RAW_FILES:
            recorded = wang_manifest_sha256(wang_manifest, raw.relpath)
            check_manifest_pin(raw.relpath, recorded, raw.sha256)
            source = wang_root / raw.relpath
            if not source.exists():
                raise RuntimeError(
                    f"required raw artifact missing from mirror: {source}"
                )
            link_verified(source, osp.join(self.raw_dir, raw.filename), recorded)
        log.info("Fang 2025 titer artifacts linked into %s", self.raw_dir)

    def _declined(self, row: TiterRow) -> str | None:
        """Why this released row is not stored, or ``None`` when it is."""
        if row.strain in TITER_ALTERNATE_HOST_STRAINS:
            return "alternate_host_no_deposited_assembly"
        if row.medium == TITER_GLUCOSE_MEDIUM:
            return "glucose_medium_not_in_the_media_layer"
        if row.strain in TITER_POINT_ALLELE_STRAINS:
            return "named_pcnB_point_allele_leaf_undecided"
        return None

    @post_process
    def process(self) -> None:
        """Build one record per released culture of a statable strain; write LMDB."""
        pins = {raw.filename: raw.sha256 for raw in TITER_RAW_FILES}
        pins.update({raw.filename: raw.sha256 for raw in WANG_RAW_FILES})
        verify_raw_files(self.raw_dir, pins)
        clusters = read_clusters(
            osp.join(self.raw_dir, WANG_CLUSTERS_FILE.filename),
            n_clusters=N_CLUSTERS,
            n_members=N_CLUSTER_MEMBERS,
        )
        spacers = read_library(
            osp.join(self.raw_dir, WANG_LIBRARY_FILE.filename),
            clusters,
            n_targeting=N_TARGETING_GUIDES,
            n_non_targeting=N_CONTROL_GUIDES,
        )
        targets: dict[str, tuple[tuple[str, str], ...]] = {
            token: tuple((member.bnumber, member.symbol) for member in cluster.members)
            for token, cluster in clusters.items()
        }
        symbol_to_bnumber: dict[str, str] = {}
        for members in targets.values():
            for bnumber, symbol in members:
                symbol_to_bnumber.setdefault(symbol.lower(), bnumber)

        rows = read_titer_blocks(osp.join(self.raw_dir, SCREEN_FILE.filename))
        strains = read_strain_table(osp.join(self.raw_dir, STRAIN_FILE.filename))
        plasmids = read_plasmid_table(osp.join(self.raw_dir, PLASMID_FILE.filename))
        primers = read_primer_table(osp.join(self.raw_dir, PRIMER_FILE.filename))

        kept, declined, disagreement, repeats = self._collate(rows)
        ledger, references = self._write_records(
            kept, strains, plasmids, primers, targets, spacers, symbol_to_bnumber
        )
        self._write_titer_reports(
            ledger, references, declined, disagreement, repeats, len(rows)
        )

    def _collate(
        self, rows: Sequence[TiterRow]
    ) -> tuple[dict[tuple[str, str], list[TiterRow]], dict[str, list[str]], float, int]:
        """Group the released rows by culture, keeping the most precise repeat.

        A (strain, medium) pair released in several panels is ONE set of three
        cultures: the values agree cell for cell except for Figure S9b's rounding of
        ``pcnB-``, so the build checks the agreement, keeps the row whose cells carry
        the most digits and records every panel that released it.
        """
        kept: dict[tuple[str, str], list[TiterRow]] = {}
        declined: dict[str, list[str]] = {}
        for row in rows:
            reason = self._declined(row)
            if reason is not None:
                declined.setdefault(reason, []).append(
                    f"{row.panel}:{row.strain} ({row.medium})"
                )
                continue
            kept.setdefault((row.strain, row.medium), []).append(row)
        folded: dict[tuple[str, str], list[TiterRow]] = {}
        for (strain, medium), group in kept.items():
            folded.setdefault((strain.lower(), medium), []).extend(group)
        worst = 0.0
        repeats = 0
        for key, group in folded.items():
            if len(group) > 1:
                repeats += 1
            for position in range(3):
                values = [row.replicates[position] for row in group]
                spread = max(values) - min(values)
                worst = max(worst, spread)
                if spread > TITER_REPEAT_TOLERANCE:
                    raise RuntimeError(
                        f"{key}: replicate {position + 1} is released as {values} by "
                        f"{[row.panel for row in group]}, which disagree by more than "
                        f"{TITER_REPEAT_TOLERANCE}"
                    )
        return folded, declined, worst, repeats

    @staticmethod
    def _most_precise(group: Sequence[TiterRow]) -> TiterRow:
        """The repeat whose cells carry the most digits, which is the unrounded one."""
        return max(
            group, key=lambda row: sum(len(repr(value)) for value in row.replicates)
        )

    def _genotype(
        self,
        spec: StrainSpec,
        primers: Mapping[str, str],
        targets: Mapping[str, tuple[tuple[str, str], ...]],
        spacers: Mapping[str, str],
        symbol_to_bnumber: Mapping[str, str],
    ) -> tuple[Genotype, list[str], list[str]]:
        """One strain's genotype, with the guide ids and primers it was proved by."""
        leaves: list[GenePerturbationType] = []
        guide_ids: list[str] = []
        matched: list[str] = []
        for guide in spec.guides:
            guide_id = resolve_short_guide(guide, spacers)
            parsed = GUIDE_ID_RE.match(guide_id)
            if parsed is None:
                raise RuntimeError(f"{guide_id!r} is not a library guide id")
            spacer = spacers[guide_id]
            matched.append(check_spacer_against_primer(guide, spacer, primers))
            guide_ids.append(guide_id)
            for bnumber, symbol in targets[str(parsed.group("token"))]:
                leaves.append(_knockdown(bnumber, symbol, spacer))
        for addition in spec.additions:
            added = symbol_to_bnumber.get(addition.gene.lower())
            if added is None:
                raise RuntimeError(
                    f"{spec.strain}: no released b-number for the added gene "
                    f"{addition.gene!r}"
                )
            leaves.append(_added_gene(addition, added))
        for gene in spec.deletions:
            deleted = symbol_to_bnumber.get(gene.lower())
            if deleted is None:
                raise RuntimeError(f"{spec.strain}: no released b-number for {gene!r}")
            leaves.append(_host_deletion(gene, deleted))
        return Genotype(perturbations=leaves), guide_ids, matched

    def _write_records(
        self,
        kept: Mapping[tuple[str, str], list[TiterRow]],
        strains: Mapping[str, str],
        plasmids: Mapping[str, str],
        primers: Mapping[str, str],
        targets: Mapping[str, tuple[tuple[str, str], ...]],
        spacers: Mapping[str, str],
        symbol_to_bnumber: Mapping[str, str],
    ) -> tuple[list[TiterStrainRecord], dict[str, dict[str, float]]]:
        """Write every stored culture to LMDB and return the ledger and the references."""
        specs = {
            key: strain_spec(self._most_precise(group).strain, strains, plasmids)
            for key, group in kept.items()
        }
        references = self._references(kept, specs)
        pub = publication()
        environments = {
            False: titer_environment(arabinose=False),
            True: titer_environment(arabinose=True),
        }
        genomes = {
            True: titer_reference_genome(True),
            False: titer_reference_genome(False),
        }
        reference_objects = {
            carries: ProductTiterExperimentReference(
                dataset_name=self.name,
                genome_reference=genomes[carries],
                environment_reference=environments[False],
                phenotype_reference=titer_reference_phenotype(
                    references[plasmid]["mean"],
                    references[plasmid]["sd"],
                    int(references[plasmid]["n_samples"]),
                ),
            )
            for carries, plasmid in ((True, _PLASMID_PCF), (False, _PLASMID_PC))
        }

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        ledger: list[TiterStrainRecord] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for key in sorted(kept):
                group = kept[key]
                row = self._most_precise(group)
                spec = specs[key]
                genotype, guide_ids, matched = self._genotype(
                    spec, primers, targets, spacers, symbol_to_bnumber
                )
                arabinose = any(
                    addition.promoter == "PBAD" for addition in spec.additions
                )
                environment = environments[arabinose]
                reference = reference_objects[spec.carries_thioesterase]
                for position, value in enumerate(row.replicates, start=1):
                    replicate_id = f"{row.strain} {position}"
                    experiment = ProductTiterExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
                        environment=environment,
                        phenotype=titer_phenotype(float(value), replicate_id),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(experiment, reference, pub, itxn),
                    )
                    idx += 1
                    ledger.append(
                        TiterStrainRecord(
                            strain=row.strain,
                            medium=row.medium,
                            replicate_id=replicate_id,
                            titer_mg_per_l=float(value),
                            panels=sorted({member.panel for member in group}),
                            released_mean=row.released_mean,
                            released_sd=row.released_sd,
                            statement=spec.statement,
                            background_plasmid=spec.background_plasmid,
                            guides=list(spec.guides),
                            guide_ids=guide_ids,
                            primers=matched,
                            additions=[
                                f"{addition.promoter}: {addition.gene} on "
                                f"{addition.plasmid}"
                                for addition in spec.additions
                            ],
                            deletions=list(spec.deletions),
                            perturbations=len(genotype.perturbations),
                        )
                    )
        env.close()
        interned_env.close()
        if idx != EXPECTED_TITER_RECORDS:
            raise RuntimeError(
                f"{self.name}: wrote {idx} records, the pinned bytes give "
                f"{EXPECTED_TITER_RECORDS}"
            )
        if len(kept) != EXPECTED_TITER_CULTURES:
            raise RuntimeError(
                f"{self.name}: {len(kept)} distinct cultures, the pinned bytes give "
                f"{EXPECTED_TITER_CULTURES}"
            )
        log.info("Wrote %d %s records over %d strains", idx, self.name, len(kept))
        return ledger, references

    @staticmethod
    def _references(
        kept: Mapping[tuple[str, str], list[TiterRow]],
        specs: Mapping[tuple[str, str], StrainSpec],
    ) -> dict[str, dict[str, float]]:
        """The released control mean and SD of each background, from the same panels.

        ``CF`` is the control of the pCF family ("The CF strain, which did not undergo
        any genome modulation, produced 747 mg/L FFAs") and ``CF-`` of the pC family.
        Both are read out of the released rows rather than restated, so the reference
        is the source's own number.
        """
        references: dict[str, dict[str, float]] = {}
        for strain, plasmid in (("cf", _PLASMID_PCF), ("cf-", _PLASMID_PC)):
            key = (strain, TITER_GLYCEROL_MEDIUM)
            if key not in kept:
                raise RuntimeError(f"the released panels carry no {strain!r} control")
            row = FfaTiterFang2025Dataset._most_precise(kept[key])
            if specs[key].background_plasmid != plasmid:
                raise RuntimeError(
                    f"{strain!r} is stated on {specs[key].background_plasmid}, not "
                    f"{plasmid}"
                )
            if specs[key].guides or specs[key].additions or specs[key].deletions:
                raise RuntimeError(f"{strain!r} is not an unmodulated control strain")
            references[plasmid] = {
                "mean": row.released_mean,
                "sd": row.released_sd,
                "n_samples": 3.0,
            }
        return references

    def _write_titer_reports(
        self,
        ledger: Sequence[TiterStrainRecord],
        references: Mapping[str, dict[str, float]],
        declined: Mapping[str, list[str]],
        disagreement: float,
        repeats: int,
        source_rows: int,
    ) -> None:
        """Write the per-culture ledger and the build accounting."""
        pd.DataFrame([record.model_dump(mode="json") for record in ledger]).to_csv(
            osp.join(self.preprocess_dir, "titer_rows.csv"), index=False
        )
        accounting = TiterAccounting(
            dataset=self.name,
            source_blocks=len(TITER_BLOCKS),
            source_rows=source_rows,
            source_values=3 * source_rows,
            distinct_cultures=len({record.strain.lower() for record in ledger}),
            kept_records=len(ledger),
            declined_values=3 * sum(len(items) for items in declined.values()),
            repeated_release_values=(
                3 * source_rows
                - len(ledger)
                - 3 * sum(len(items) for items in declined.values())
            ),
            declined={reason: sorted(items) for reason, items in declined.items()},
            max_repeat_disagreement=disagreement,
            repeated_keys=repeats,
            references={
                plasmid: dict(values) for plasmid, values in references.items()
            },
            notes=[
                "one record is one biological replicate culture: the released value "
                "verbatim, n_samples=1, sample_unit=biological_replicate, replicate_id "
                "the panel's own labels for that culture, its strain and its column "
                "number ('pcnBi 1'); the strain half is what keeps pcnB_956, pcnBi and "
                "pcnBiF- apart, since the schema states a guide without the plasmid "
                "carrying it and all three carry one pcnB knockdown",
                "the released Mean and SD of every row reproduce from its three "
                f"columns within {TITER_SUMMARY_TOLERANCE}, which is what fixes the "
                "column alignment; they are a derivation over the stored records and "
                "live only in preprocess/titer_rows.csv",
                "a (strain, medium) pair released by several panels is ONE set of "
                "three cultures, stored once, with every panel that released it in the "
                f"ledger; the panels disagree by at most {disagreement} mg/L (Figure "
                "S9b rounds the pcnB- cells that Figure S3a prints in full)",
                "every genotype is parsed from Supplementary Data 1 and 2: the strain "
                "sentence names its plasmids and each plasmid's parts name its guides, "
                "its added gene and its promoter",
                "every guide's 20-mer is Wang 2018's, and every one is re-checked "
                "against the 4-base-tagged primer Fang's own Supplementary Data 3 "
                "carries for that guide name",
                "the fadR arm of pcnBi-acrDi-fadR+ and pcnBi-acrDi-fadRT7 is a "
                "GeneAdditionPerturbation (an extra native copy on its own replicon, "
                f"'{_Q_FADR_OVEREXPRESSION}'); the module's earlier decline, which "
                "said no leaf types a bacterial overexpression, was wrong",
                "the pC strains (CF-, pcnBiF-, pcnB-F-) carry a background without "
                "tesA' and a reference of their own, the released CF- mean",
                f"{3 * sum(len(items) for items in declined.values())} released values "
                "are not stored, and none of them is a replicate of a stored strain: "
                f"the alternate hosts of Figure 4 ('{_Q_ALTERNATE_HOSTS}') have no "
                f"deposited assembly, the glucose medium ('{_Q_GLUCOSE_MEDIUM}') is not "
                f"in the media layer, and the pcnB point mutants "
                f"('{_Q_POINT_MUTANTS}') need a leaf decision",
                f"the authors withdraw no culture ('{_Q_NO_EXCLUSION}')",
            ],
        )
        accounting.check()
        with open(
            osp.join(self.preprocess_dir, "titer_build_accounting.json"), "w"
        ) as handle:
            json.dump(accounting.model_dump(mode="json"), handle, indent=2)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError("FfaTiterFang2025Dataset builds records in process()")


def _titer_provenance() -> Provenance:
    """What a titer record is, for the verification report."""
    return Provenance(
        source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{SCREEN_FILE.relpath}",
        citation_key=CITATION_KEY,
        sha256=SCREEN_FILE.sha256,
        method=(
            "Supplementary Data 6, the 13 'FFAs (mg L-1)' blocks: one "
            "ProductTiterExperiment per released biological replicate culture of every "
            "strain whose genotype the schema states. titer = the released mg/L "
            "verbatim under the numerically identical ug/mL, n_samples=1, "
            "sample_unit=biological_replicate, replicate_id = the column header the "
            "panel prints; the uncertainty fields are typed ProvenanceGaps with reason "
            "not_applicable_single_measurement. Genotype parsed from Supplementary Data "
            "1 and 2: CrisprInterferencePerturbation per member of each guide's Wang "
            "2018 cluster with Wang's spacer, GeneAdditionPerturbation for the P_BAD "
            "and P_T7 fadR plasmids, DeletionStrainPerturbation for the lambda-RED "
            "pcnB deletion. Reference = the released control mean and SD of the "
            "record's own background (CF for pCF, CF- for pC). DECLINED: the 10 "
            "alternate-host rows (no deposited assembly), the 2 glucose rows (medium "
            "not in the media layer) and the 2 pcnB point-mutant rows (leaf undecided)"
        ),
        page=(
            "Nat Commun 2025 16 (doi:10.1038/s41467-025-58368-3); Methods "
            "'Cultivation and fermentation', 'Metabolite extraction and analysis' and "
            "'Statistics and reproducibility'; Supplementary Data 1, 2, 3 and 6; "
            f"paper.md sha256={PAPER_MD_SHA256}"
        ),
        retrieved=RAW_RETRIEVED_AT,
    )


def _l1_titer_panels_are_accounted(
    records: Sequence[dict[str, Any]], dataset_root: str
) -> LevelResult:
    """L1: the ledger accounts for every released row of every panel."""
    from torchcell.verification.levels import l1_count

    payload = json.loads(
        Path(
            osp.join(dataset_root, "preprocess/titer_build_accounting.json")
        ).read_text()
    )
    stored = int(payload["kept_records"])
    declined = int(payload["declined_values"])
    repeated = int(payload["repeated_release_values"])
    result = l1_count(len(records), stored)
    return result.model_copy(
        update={
            "name": "panels_are_accounted",
            "passed": result.passed
            and stored + declined + repeated == int(payload["source_values"])
            and int(payload["source_blocks"]) == len(TITER_BLOCKS),
            "message": (
                f"{len(records)} records, {stored} in the ledger, {declined} declined "
                f"values, {repeated} re-releases of a stored culture, "
                f"{payload['source_values']} released over "
                f"{payload['source_blocks']} blocks"
            ),
        }
    )


def _l3_three_cultures_per_strain(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L3: every stored strain carries its three numbered cultures, exactly once.

    The strain half of ``replicate_id`` is what keeps two strains of one genotype
    apart, so the rule reads it: each strain label must carry the labels ``1``, ``2``
    and ``3`` and no label may repeat.
    """
    from torchcell.verification.levels import l3_convention

    groups: dict[str, list[str]] = {}
    for record in records:
        label = str(record["experiment"]["phenotype"]["replicate_id"])
        strain, _, position = label.rpartition(" ")
        groups.setdefault(strain, []).append(position)
    bad = {
        strain: sorted(positions)
        for strain, positions in groups.items()
        if sorted(positions) != ["1", "2", "3"]
    }
    return l3_convention(
        "three_cultures_per_strain",
        not bad and len(groups) == EXPECTED_TITER_CULTURES,
        detail=(
            f"{len(groups)} stored strains, each of which must carry the panel's "
            f"cultures 1, 2 and 3 exactly once; {len(bad)} do not, and the pinned "
            f"bytes give {EXPECTED_TITER_CULTURES} strains"
        ),
    )


def _l4_titers_vs_released_workbook(
    records: Sequence[dict[str, Any]], data_root: str | None
) -> LevelResult:
    """L4: every stored titer is a released cell of the workbook, to the last digit."""
    root = raw_mirror_dir(data_root)
    released: set[float] = set()
    for row in read_titer_blocks(root / SCREEN_FILE.relpath):
        released.update(float(value) for value in row.replicates)
    stored = [float(record["experiment"]["phenotype"]["titer"]) for record in records]
    missing = [value for value in stored if value not in released]
    return LevelResult(
        level=Level.L4,
        name="titer_vs_released_source_data",
        passed=not missing,
        message=(
            f"{len(stored)} stored titers against {len(released)} released culture "
            f"values of Supplementary Data 6; {len(missing)} are not released cells"
        ),
        details={
            "n_stored": len(stored),
            "n_released": len(released),
            "missing": missing[:20],
        },
    )


def _l4_spacers_vs_released_primers(
    records: Sequence[dict[str, Any]], data_root: str | None
) -> LevelResult:
    """L4: every stored spacer is the one Fang's own primer table carries."""
    root = raw_mirror_dir(data_root)
    primers = read_primer_table(root / PRIMER_FILE.relpath)
    tails = set()
    for sequence in primers.values():
        tail = sequence[4:].upper()
        tails.add(tail)
        tails.add(tail.translate(str.maketrans("ACGT", "TGCA"))[::-1])
    spacers = {
        str(perturbation["crispr"]["guide_sequence"])
        for record in records
        for perturbation in record["experiment"]["genotype"]["perturbations"]
        if perturbation["perturbation_type"] == "crispr_interference"
    }
    missing = sorted(spacers - tails)
    return LevelResult(
        level=Level.L4,
        name="spacer_vs_released_primer_table",
        passed=not missing,
        message=(
            f"{len(spacers)} distinct stored spacers against the {len(primers)} primers "
            f"of Supplementary Data 3; {len(missing)} carry no primer"
        ),
        details={
            "n_spacers": len(spacers),
            "n_primers": len(primers),
            "missing": missing[:20],
        },
    )


def verify_titer_build(
    dataset_root: str, data_root: str | None = None
) -> VerificationReport:
    """Run the titer family's L0-L4 gate over the built store and write its report."""
    from torchcell.verification.product_titer import verify_product_titer_dataset
    from torchcell.verification.runners import load_records

    records = [dict(record) for record in load_records(dataset_root)]
    report = verify_product_titer_dataset(
        records,
        dataset_name="ffa_titer_fang2025",
        provenance=_titer_provenance(),
        expected_count=EXPECTED_TITER_RECORDS,
        titer_unit=ConcentrationUnit.ug_per_ml.value,
        titer_unit_detail=(
            "Supplementary Data 6 releases mg/L and ConcentrationUnit has no mg/L "
            "member; 1 mg/L == 1 ug/mL exactly, so the released number is stored "
            "verbatim under the numerically identical unit"
        ),
        se_tol=TITER_SE_TOL,
        pathway_gene_counts=(0,),
        product_names=(TITER_PRODUCT_NAME,),
    )
    report.add(_l1_titer_panels_are_accounted(records, dataset_root))
    report.add(_l3_three_cultures_per_strain(records))
    report.add(_l4_titers_vs_released_workbook(records, data_root))
    report.add(_l4_spacers_vs_released_primers(records, data_root))
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build the dataset under ``DATA_ROOT``, print its accounting and verify it."""
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/crispri_guide_ffa_enrichment_fang2025")
    genome = bacterial_genome("ecoli", MG1655_STRAIN, data_root)
    dataset = CrispriGuideFfaEnrichmentFang2025Dataset(root=root)
    print(f"len = {len(dataset)}")
    dataset.close_lmdb()
    accounting = json.loads(
        Path(osp.join(root, "preprocess/build_accounting.json")).read_text()
    )
    print(
        json.dumps(
            {
                key: accounting[key]
                for key in (
                    "source_rows",
                    "kept_records",
                    "perturbations",
                    "distinct_genes",
                    "read_floor",
                    "library_citation_key",
                )
            },
            indent=2,
        )
    )
    report = verify_build(root, genome=genome, data_root=data_root)
    print(report.summary())
    for result in report.results:
        flag = "PASS" if result.passed else "FAIL"
        print(f"  [{flag}] L{int(result.level)} {result.name}: {result.message}")


if __name__ == "__main__":
    main()
