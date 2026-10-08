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
``BacterialEnvironmentResponseExperiment`` per (guide, round) row of Supplementary
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
is a ``BacterialCrisprInterferencePerturbation`` leaf on ``b0143`` beside the library
guide's. It is not pushed into the reference because
``BacterialEnvironmentResponseExperimentReference`` holds a genome, an environment and a
phenotype and has no genotype field, so a reference cannot express "the pcnBi strain" at
all. Consequence: round-1 records are single-knockdown and round-2 records are double,
and the pcnB leaf is the only perturbation this dataset writes that is not a library
guide.

THE HOST IS BACKGROUND, NOT PERTURBATION. CF is "an MG1655 (DE3) derivative with fadE
deletion) carrying the pCF plasmid for expression of dCas9 and the truncated fatty
acyl-ACP thioesterase TesA'". The dfadE lesion, the DE3 prophage and the pCF plasmid are
identical in every record of both rounds and in the reference, so they are a
``BacterialStrainBackground`` rather than a per-record edit. The paper prints no locus
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

The second rule removes 3 more rows, 2 in round 1 and 1 in round 2, whose every cluster
member is a b-number ``GCA_000005845.2`` no longer carries: the ``sokEb4700`` and
``ybfKb4590`` singleton clusters, which Wang 2018 drops for the same reason at its own
scale. The per-round arithmetic and the per-library split of the excluded rows are in
``preprocess/build_accounting.json``.

Round 2 keeping only 331 of 55,671 is not an error and not a drop rule: 55,334 of its
guides read below 20 AFTER sorting, which is what sorting the top 1% of a library
already shaped by a first round does. It is recorded because a reader seeing 331 will
otherwise suspect the parse.

WHAT IS RELEASED AND DELIBERATELY NOT LOADED:

- **The per-strain FFA titers.** ``si8.xlsx`` carries 13 ``FFAs (mg L-1)`` blocks over
  165 labelled rows, each with three biological replicates, a mean and an SD, plus
  OD600, acetate, intracellular-FFA and RT-qPCR blocks. Those are a
  ``ProductTiterExperiment`` family with a RELEASED reference titer (the CF strain at
  746.8201 mg/L mean), so unlike three sibling titer rows this one would not be refused
  for want of a reference -- but the engineered strains include ``pcnBi-acrDi-fadR+``,
  whose ``fadR`` arm is a P_BAD-driven OVEREXPRESSION that no bacterial gene-perturbation
  leaf types, so loading them would either drop that strain or type an overexpression as
  something else. The titers are measured and recorded here, and the missing leaf is
  raised in the PR rather than forced.
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
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    BacterialCrisprInterferencePerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialGeneNamespace,
    BacterialStrainBackground,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    Environment,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    GenePerturbationType,
    Genotype,
    MeasurementType,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    BACTERIAL_ASSEMBLY_SETS,
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli.wang2018 import CITATION_KEY as WANG_CITATION_KEY
from torchcell.datasets.ecoli.wang2018 import CLUSTERS_FILE as WANG_CLUSTERS_FILE
from torchcell.datasets.ecoli.wang2018 import LIBRARY_FILE as WANG_LIBRARY_FILE
from torchcell.datasets.ecoli.wang2018 import (
    GeneCluster,
    read_clusters,
    read_library,
    stored_gene_names,
)
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
from torchcell.verification.report import Provenance
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
RAW_RETRIEVED_AT = "2026-10-08"

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
        kept_records=15378,
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
        kept_records=330,
    ),
)

#: Totals over the two rounds, measured on the pinned bytes.
EXPECTED_RECORDS = 15708
EXPECTED_PERTURBATIONS = 16187
EXPECTED_GENES = 2628

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
    "retired_target -- 3 rows (sokEb4700_32 and sokEb4700_33 in round 1, ybfKb4590_11 "
    "in round 2): every member of the guide's cluster is a b-number the pinned "
    "GCA_000005845.2 annotation no longer carries, so the L1 canonical-name rule and "
    "the L4 gene universe would both reject the stored name. Wang 2018 drops the same "
    "two singleton clusters, at its own scale (55 rows over five screens)",
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
    """Deposit the consumed supplementary file and write ``manifest.json``.

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
    for raw in RAW_FILES:
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
    for raw in RAW_FILES:
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
            for raw in RAW_FILES
        ]
        + [
            "Supplementary Data 1-3 (MOESM3-5, strains, plasmids and primers) are NOT "
            "deposited here: no record reads them, and the primer table is quoted from "
            "the torchcell-library mirror for the short-form guide-name proof",
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


def host_background() -> BacterialStrainBackground:
    """Strain CF: MG1655(DE3) dfadE carrying pCF, constant in every record.

    ``alleles`` is EMPTY and the lesion lives in ``genotype_statement`` verbatim, which
    is what the class documents for a source that "states only a genotype string".
    ``BacterialBackgroundAllele.systematic_gene_name`` is a required locus tag validated
    against the namespace, so typing the ``fadE`` deletion would mean storing ``b0221``,
    a tag that appears nowhere in the mirrored bytes; the gap is recorded on the
    background instead of being closed by a derivation the source never made. The DE3
    prophage and the pCF plasmid stay in the statement for the same reason: neither is
    given a locus tag, coordinates or a sequence.
    """
    return BacterialStrainBackground(
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


def host_reference(data_root: str | None = None) -> AssemblyReferenceGenome:
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
    dataset_name: str, round_: Round, genome_reference: AssemblyReferenceGenome
) -> BacterialEnvironmentResponseExperimentReference:
    """The unsorted-library baseline of one round: enrichment 0 by the normalization."""
    return BacterialEnvironmentResponseExperimentReference(
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
) -> BacterialCrisprInterferencePerturbation:
    """One dCas9 knockdown leaf carrying its released 20-mer spacer."""
    return BacterialCrisprInterferencePerturbation(
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
    dropped_retired_target: int
    kept_records: int
    perturbations: int
    background_knockdowns: int
    max_equation_1_error: float
    max_equation_2_error: float

    def check(self) -> None:
        """The two drop counts must account for every released row."""
        kept = (
            self.source_rows
            - self.dropped_below_read_floor
            - self.dropped_retired_target
        )
        if kept != self.kept_records:
            raise ValueError(
                f"{self.round_id}: {self.source_rows} rows minus "
                f"{self.dropped_below_read_floor} below the floor and "
                f"{self.dropped_retired_target} retired is {kept}, not the "
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
    retired_targets_dropped: list[str]
    reconciliation: LocusTagReconciliation
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

    REFERENCE_STRAIN: ClassVar[Literal["MG1655"]] = "MG1655"
    #: Measured on the pinned bytes: 4,313 of the 4,317 cluster-member b-numbers resolve
    #: to a locus of GCA_000005845.2 (0.9991). Below this the annotation or the release
    #: moved and the build stops rather than dropping more records.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 0.999

    def __init__(
        self,
        root: str = "data/torchcell/crispri_guide_ffa_enrichment_fang2025",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the MG1655 genome resolves the library's b-numbers."""
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

    def _genome(self) -> EcoliK12Genome:
        """The injected MG1655 genome, or one opened from the genomes tier."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

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

        genome = self._genome()
        bnumbers = sorted(
            {m.bnumber for cluster in clusters.values() for m in cluster.members}
        )
        stored, reconciliation = reconcile_locus_tags(
            genome, pd.Series(bnumbers), label=self.name
        )
        reconciliation.require_resolved(self.MIN_RESOLVED_FRACTION)
        if reconciliation.outside_namespace:
            raise RuntimeError(
                f"{self.name}: targets outside {MG1655_NAMESPACE}: "
                f"{reconciliation.outside_namespace}"
            )
        if list(stored) != bnumbers:
            raise RuntimeError(
                f"{self.name}: reconciliation remapped a b-number, which means the "
                "released identifiers are not this assembly's locus tags"
            )
        retired = frozenset(reconciliation.retired_kept) | frozenset(
            reconciliation.ambiguous_kept
        )
        names = stored_gene_names(genome, clusters.values())
        targets: dict[str, tuple[tuple[str, str], ...]] = {
            token: tuple(
                (m.bnumber, names[m.bnumber])
                for m in cluster.members
                if m.bnumber not in retired
            )
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
            dropped_retired = 0
            perturbations = 0
            with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
                env_ptr = self._pointer(environment, environment.media.name, itxn)
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
                    if not target:
                        dropped_retired += 1
                        continue
                    if key not in genotypes:
                        genotypes[key] = build_genotype(
                            target, spacers[key], leaves
                        ).model_dump()
                    genotype = genotypes[key]
                    phenotype = guide_phenotype(round_, float(fitness))
                    record = {
                        "experiment": {
                            "experiment_type": "bacterial_environment_response",
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
            kept = len(rows.kept) - dropped_retired
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
                    dropped_retired_target=dropped_retired,
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
        self._write_reports(
            clusters, spacers, reconciliation, accounting, genes, retired, background
        )
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
        reference: BacterialEnvironmentResponseExperimentReference,
        pub: Publication,
        itxn: Any,
    ) -> None:
        """The fast-path record equals what ``_intern_record`` writes for the same objects."""
        experiment = BacterialEnvironmentResponseExperiment(
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
        reconciliation: LocusTagReconciliation,
        rounds: list[RoundAccounting],
        genes: set[str],
        retired: frozenset[str],
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
            assembly_set=reconciliation.assembly_set,
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
            retired_targets_dropped=sorted(retired),
            reconciliation=reconciliation,
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
                "BacterialEnvironmentResponseExperiment per kept (sgRNA, round) row. "
                "measurement_type=log2_ratio, assay_type=biosensor_readout (BS and AS "
                "are the same culture; the FACSAria III gates on Nile-Red fluorescence "
                "and sorts the top 1 percent), environment_response = the released "
                "'Fitness' = log2(normalized reads in AS / normalized reads in BS) with "
                "normalized reads = (reads + 1) / total reads (Methods equations 1 and "
                "2, both reproduced from the released read counts: equation 1 exactly, "
                "equation 2 to 1.8e-15). n_samples=1 pooled library per arm per round, "
                "so no dispersion exists and the uncertainty fields are typed "
                "ProvenanceGaps. Genotype = one "
                "BacterialCrisprInterferencePerturbation per member of the guide's "
                "BLASTN cluster, effector dCas9, guide_sequence = the 20-mer spacer of "
                "Wang 2018 Supplementary Data 3 (the screened set is Wang's 55,671 "
                "targeting guides, measured id for id), plus in round 2 the constant "
                "pcnB repression of the pcnBi host (pcnBb0143_956). Reference = the "
                "unsorted library of the same round, response 0.0 by that "
                f"normalization. DROPPED: 95,631 rows below the authors' own "
                f"{READ_FLOOR.value}-read floor and 3 rows whose only target is a "
                "retired b-number (b4700, b4590)"
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


def main() -> None:
    """Build the dataset under ``DATA_ROOT``, print its accounting and verify it."""
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/crispri_guide_ffa_enrichment_fang2025")
    genome = bacterial_genome("ecoli", MG1655_STRAIN, data_root)
    dataset = CrispriGuideFfaEnrichmentFang2025Dataset(root=root, ecoli_genome=genome)
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
                    "retired_targets_dropped",
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
