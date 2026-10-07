# torchcell/datasets/ecoli/wang2018
# [[torchcell.datasets.ecoli.wang2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/wang2018
# Test file: tests/torchcell/datasets/ecoli/test_wang2018.py
r"""Wang 2018 pooled CRISPRi screening in E. coli K-12 MG1655.

Wang et al. 2018 (Nat Commun 9, 2475; doi:10.1038/s41467-018-04899-x) designed a
genome-scale sgRNA library of 55,671 guides plus 400 non-targeting controls against
``NC_000913.3`` and ran it through five pooled CRISPRi screens. This module serves the
per-guide readout the paper releases: :class:`CrispriGuideFitnessWang2018Dataset`, one
``BacterialEnvironmentResponseExperiment`` per (guide, screen) row of Supplementary
Data 6-10.

THE RECORD IS A SIGNED LOG2 RATIO, WHICH IS WHY IT IS NOT A ``FitnessPhenotype``. The
released statistic is the paper's ``sgRNA fitness``: equation (2) takes
``Log2((Read count)_selective / (Read count)_control)`` and equation (3) subtracts the
median of the non-targeting controls. It is routinely negative (the essentiality screen
spans -9.82 to +3.92), and ``FitnessPhenotype`` is a strictly positive ko/wt growth
ratio that CLAMPS non-positive values to 0 and whose verifier requires a 1.0 reference.
So the record is an ``EnvironmentResponsePhenotype`` with
``measurement_type=log2_ratio``, ``assay_type=pooled_competitive_growth_barcode`` (the
molecular barcode is the guide's own N20 spacer, amplified from the extracted plasmids
and counted by NGS), and the reference's response is 0.0 by that normalization.

THE ``Z score`` COLUMN CARRIES NO INFORMATION THE FITNESS DOES NOT, AND THE BUILD PROVES
IT. Equation (4) is ``Z = Fitness / sigma``, where sigma is the standard deviation of a
normal fitted to the 400 control guides' fitness, so sigma is one constant per screen.
Measured on the released bytes: ``fitness / Z`` is constant to 2.1e-12 across every row
of every screen (:data:`SCREENS` carries the five values, and
:func:`back_solved_sigma` recomputes them at build time and refuses a screen whose ratio
drifts). The Z score is therefore a deterministic rescaling and is not stored twice;
sigma is recorded in ``preprocess/build_accounting.json`` as the screen's null-dispersion
scale. It is NOT a per-record uncertainty: it is the spread of the NULL distribution, so
``environment_response_uncertainty`` stays a typed ``ProvenanceGap``.

WHAT ONE ROW IS, AND WHY A GUIDE CAN REPRESS SEVERAL GENES. The library's "cluster"
strategy groups genes whose sequences BLASTN-match into one unit and designs guides
against every member: "for genes with multiple copies in the genome, we used the BLASTN
program to categorize genes with highly similar sequences into clusters (Supplementary
Data 2) and designed sgRNAs to target all members of a cluster. Hence, genes in one
cluster are regarded as functionally identical." Supplementary Data 2 names 4,205
clusters (4,090 protein-coding, 115 ncRNA) over 4,317 member genes; 39 clusters hold
more than one member and the largest holds ten (the ``insH1`` IS5 transposase copies,
ahead of the eight 5S rRNA genes of ``rrfH``). One record is one (guide, screen) row and
carries one ``BacterialCrisprInterferencePerturbation`` per member of its guide's
cluster, so the build writes 242,294 knockdown perturbations over 240,481 records.

A SPACER CAN ALSO BE SHARED BETWEEN TWO CLUSTERS, which is why the stored strain identity
is (gene, spacer) and not the spacer alone. Measured on Supplementary Data 3: 25 spacers
appear twice and none more than twice, and every one of the 25 sits on two DIFFERENT
clusters over six pairs (``esrE``/``ubiJ``, ``hcaR``/``iroK``, ``hokB``/``mokB``,
``hokC``/``mokC``, ``rzoQ``/``rzpQ``, ``sgrS``/``sgrT``: overlapping or nested genes). No
pair falls inside one cluster, so no two records collide on (gene, spacer).

IDENTIFIERS COME OUT OF THE SOURCE'S OWN TOKENS. Every library id is
``<symbol><bNNNN>_<position>`` (``rsmE_9`` in the Methods is the tiling library's older
form; the genome-scale ids are ``gspKb3332_817``), and Supplementary Data 2 spells each
cluster member the same way. The b-number is therefore released, not derived: no ECK
crosswalk is used and ``identifier_mapping`` is ``None`` on every perturbation. The
fitness files' ``gene`` column is the cluster representative's bare symbol, and it
agrees with the id's symbol on all 247,348 rows. Reconciled against the pinned
``GCA_000005845.2`` annotation, 4,310 of the 4,317 member b-numbers are CURRENT, 3 are
pseudogene loci that resolve to themselves (``b4614``, ``b4643``, ``b4645``) and 4 are
RETIRED (``b4590`` ybfK, ``b4629`` ptwF, ``b4635`` pauD, ``b4700`` sokE).

``perturbed_gene_name`` IS THE ANNOTATION'S SYMBOL WHERE THE ANNOTATION HAS ONE, so one
gene carries one spelling across datasets. 253 of the source's symbols disagree with the
assembly's primary symbol for the same b-number (the source's ``acrS`` is the
annotation's ``envR``, its ``acuI`` is ``yhdH``); the b-number is identical in both, so
nothing is lost, and every source spelling is kept in
``preprocess/guide_library.csv``.

THE THREE DROP RULES, EACH WITH ITS ARITHMETIC. 247,348 released rows become 240,481
records:

- **1,942 non-targeting control rows** (``sgRNA`` matching ``NC_<n>``, ``gene`` column
  ``0``). A guide with no genomic target cannot be a
  ``BacterialCrisprInterferencePerturbation``, whose ``systematic_gene_name`` must be a
  locus tag, and there is no non-targeting-guide leaf to type it with. These rows are
  not discarded information: their median IS the zero of every stored fitness (equation
  3) and their fitted sigma IS each screen's Z scale, both of which the build checks and
  records. Adding a typed non-targeting leaf is raised in the PR, not taken here.
- **4,880 ``Quality == "Bad"`` rows**, 10 of them also controls. The flag is the paper's
  own: "We annotated the quality of the sgRNA fitness by checking the read counts for
  each sgRNA in the control condition. Those sgRNAs with <20 reads were eliminated from
  the following analysis to calculate the gene fitness." The denominator of the log2
  ratio is below the authors' stated robustness floor and they use none of these values.
  No phenotype field can carry that flag, so storing them unmarked would present them as
  equal in quality to the other 240,481. The two chemical-tolerance screens report no
  Bad row at all, because their control is the initial library, which the <20-read filter
  had already been applied to.
- **55 rows whose every target is a RETIRED b-number** (11 per screen: the singleton
  clusters ``ybfKb4590`` and ``sokEb4700``). ``GCA_000005845.2`` carries neither tag, so
  the L1 canonical-gene-name rule fails a retired stored name and the L4 universe does
  not contain it. ``b4629`` and ``b4635`` are retired too but no guide was designed
  against them.

A pseudogene locus is NOT a drop rule: the L1 rule passes a non-gene feature that
resolves to itself, and a repressed pseudogene is a real measurement. It happens that
none of the three (``sokAb4614``, ``pawZb4643``, ``psaAb4645``) carries a designed guide,
so the 4,218 measured genes hold no pseudogene; that is a property of the library's
off-target and GC filters, not of this loader, and
``preprocess/build_accounting.json`` records the empty set rather than omitting it.

WHAT IS NOT LOADED, AND THE SCHEMA FINDING BEHIND IT. Supplementary Data 5 is the paper's
GENE-level call: per cluster per screen, the median fitness of a position-selected guide
subset with a Mann-Whitney ``Z score``, an ``FDRvalue`` and an ``FPRvalue`` (4,142 +
4,004 x 4 = 20,158 rows). Those three statistics are the whole point of that table and
``EnvironmentResponsePhenotype`` has no significance field: only
``GeneInteractionPhenotype`` carries a p-value anywhere in the schema. Storing the gene
rows without them would keep a median and silently drop the hit call, so this module
loads the guide level only and the gap is a written finding in the PR rather than a
schema edit. Supplementary Data 4 (guides per gene) is not deposited either: it is
exactly the per-cluster count of Supplementary Data 3, verified identical for all 4,205
clusters.
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
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data.experiment_dataset import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import LB, MOPS_CASAMINO_WANG2018, MOPS_MINIMAL
from torchcell.datamodels.schema import (
    AssayType,
    AssemblyReferenceGenome,
    BacterialCrisprInterferencePerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialGeneNamespace,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    PhysicalFactor,
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
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence.genome.base import GeneNameStatus
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
DOI = "10.1038/s41467-018-04899-x"
PMCID = "PMC6018678"
TITLE = (
    "Pooled CRISPR interference screening enables genome-scale functional genomics "
    "study in bacteria with superior performance"
)
CITATION_KEY = "wangPooledCRISPRInterference2018"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: PMC Article Datasets bucket prefix for this article's supplementary files.
PMC_PREFIX = f"{PMCID}.1"
#: Publisher file-name stem of every supplementary object.
ESM_STEM = "41467_2018_4899_MOESM"

#: The article OCR in the torchcell-library mirror: quoted for every sourced value,
#: never parsed here.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "9980415d606835ab1ae3a1ad92a64784c822aad1ca01b514662d2fb21d1a3a55"

#: The day the recorded ``pmc_cloud`` retrieval of every deposited file was last run and
#: reproduced its pin.
RAW_RETRIEVED_AT = "2026-10-07"

#: BioProject of the raw reads, recorded and deliberately NOT deposited.
BIOPROJECT_ACCESSION = "PRJNA450392"
#: The authors' pipeline, recorded for provenance; this loader reimplements nothing.
CODE_REPOSITORY = (
    "https://github.com/zhangchonglab/CRISPRi-functional-genomics-in-prokaryotes"
)


class RawFile(BaseModel):
    """One deposited supplementary file: its publisher object and its sha256 pin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    moesm: int
    data_number: int
    description: str
    sha256: str

    @property
    def filename(self) -> str:
        """Publisher file name, which is also the name inside ``raw/``."""
        return f"{ESM_STEM}{self.moesm}_ESM.xlsx"

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


#: Supplementary Data 2: the BLASTN gene clusters the library's guides target.
CLUSTERS_FILE = RawFile(
    moesm=5,
    data_number=2,
    description="Gene clusters in genome-scale sgRNA library",
    sha256="8bb51873557bed15067f947053f84a9428883bc3acf771a6edf67a0a8740cdeb",
)
#: Supplementary Data 3: the genome-wide library, guide id -> 20-mer spacer.
LIBRARY_FILE = RawFile(
    moesm=6,
    data_number=3,
    description="E. coli genome-wide sgRNA library",
    sha256="78a05c25da94df158065c18a316d0d29869587e4c839ad3365c911503900ef62",
)


class Screen(BaseModel):
    """One pooled screen: its released file, its conditions and its Z scale.

    ``selective`` and ``control`` are Table 2's own cells, verbatim. ``nc_sigma`` is the
    standard deviation of the normal fitted to the 400 non-targeting guides' fitness
    (equation 4's denominator), back-solved from the released ``fitness / Z`` ratio and
    re-checked at build time by :func:`back_solved_sigma`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    screen_id: str
    raw: RawFile
    sheet: str
    phenotype_label: str
    selective: str
    control: str
    nc_sigma: float
    generations: float
    kept_records: int


#: The five screens of the genome-scale library, in Table 2's order. ``kept_records`` is
#: measured on the pinned bytes and asserted per screen by the build.
SCREENS: tuple[Screen, ...] = (
    Screen(
        screen_id="essentiality",
        raw=RawFile(
            moesm=9,
            data_number=6,
            description="Fitness scores for sgRNAs of essential gene",
            sha256="6500e9105fc6c7482794a558a6c69325196067dc89e295b8eda056ff7e284d48",
        ),
        sheet="essential genes",
        phenotype_label="Essentiality",
        selective="dCas9, LB",
        control="Empty plasmid, LB",
        nc_sigma=0.8490671688617482,
        generations=15.0,
        kept_records=52245,
    ),
    Screen(
        screen_id="auxotrophy",
        raw=RawFile(
            moesm=10,
            data_number=7,
            description="Fitness scores for sgRNAs of auxotrophy",
            sha256="3de9fd0aa6e14f543ae35703278558c10085ad63ee2e5f8be214dfc22b996a6e",
        ),
        sheet="auxotrophy in MOPS media",
        phenotype_label="Auxotrophy",
        selective="MOPS",
        control="LB",
        nc_sigma=1.0761377330569217,
        generations=5.0,
        kept_records=46207,
    ),
    Screen(
        screen_id="trp_biosynthesis",
        raw=RawFile(
            moesm=11,
            data_number=8,
            description="Fitness scores for sgRNAs of amino acid addition",
            sha256="4a322c70ac984e31cf7f931cec6a3f5af9bfcb3166d73e47f904306e1a4c2ae1",
        ),
        sheet="amino acid addition",
        phenotype_label="L-Trp biosynthesis",
        selective="0.5 g/L casamino acid, MOPS",
        control="LB",
        nc_sigma=0.7305571957056847,
        generations=5.0,
        kept_records=46207,
    ),
    Screen(
        screen_id="furfural_tolerance",
        raw=RawFile(
            moesm=12,
            data_number=9,
            description="Fitness scores for sgRNAs of furfural addition",
            sha256="b40c08b07d74b07f992929f866c6aa2f7cbff4e4059148597bf7d36e989df2a9",
        ),
        sheet="furfural tolerance",
        phenotype_label="Furfural tolerance",
        selective="0.4 g/L furfural, MOPS",
        control="initial, see Supplementary Fig. 7",
        nc_sigma=1.8472428072313638,
        generations=5.0,
        kept_records=47911,
    ),
    Screen(
        screen_id="isobutanol_tolerance",
        raw=RawFile(
            moesm=13,
            data_number=10,
            description="Fitness scores for sgRNAs of isobutanol addition",
            sha256="29658ef0c45af0b2ac43c33d0858741938a2939ec34c150f77d01bfebc10ba74",
        ),
        sheet="isobutanol tolerance",
        phenotype_label="Isobutanol tolerance",
        selective="4 g/L isobutanol, MOPS",
        control="initial, see Supplementary Fig. 7",
        nc_sigma=1.6023976401505866,
        generations=5.0,
        kept_records=47911,
    ),
)

#: Every file the loader reads, clusters and library first.
RAW_FILES: tuple[RawFile, ...] = (
    CLUSTERS_FILE,
    LIBRARY_FILE,
    *(screen.raw for screen in SCREENS),
)

# --------------------------------------------------------------------------- #
# Assembly, namespace and parsing
# --------------------------------------------------------------------------- #
MG1655_NAMESPACE: BacterialGeneNamespace = "ecoli_k12_mg1655_bnumber"
MG1655_STRAIN: Literal["MG1655"] = "MG1655"

#: A library id: the cluster representative's token plus the PAM position in the ORF.
GUIDE_ID_RE = re.compile(r"^(?P<token>.+?b\d{4})_(?P<position>\d+)$")
#: A cluster-member token: the source's gene symbol plus its b-number.
MEMBER_RE = re.compile(r"^(?P<symbol>.+?)(?P<bnumber>b\d{4})$")
#: A non-targeting control id.
CONTROL_ID_RE = re.compile(r"^NC_\d+$")
#: The ``gene`` cell of a non-targeting control row.
CONTROL_GENE_CELL = "0"
#: The ``Quality`` cell of a row whose control-condition read count is below 20.
BAD_QUALITY = "Bad"
#: Every ``Quality`` cell the release uses. A third value would be silently kept as
#: good by an equality test against ``BAD_QUALITY``, so the domain is checked.
QUALITY_VALUES: tuple[str, ...] = ("Good", BAD_QUALITY)

#: Columns of a Supplementary Data 6-10 sheet.
FITNESS_COLUMNS: tuple[str, ...] = (
    "sgRNA",
    "gene",
    "sgRNA fitness",
    "Z score",
    "Quality",
)
#: Columns of Supplementary Data 3.
LIBRARY_COLUMNS: tuple[str, ...] = ("sgRNAID", "nucleotide sequence")
#: Columns of either sheet of Supplementary Data 2.
CLUSTER_COLUMNS: tuple[str, ...] = ("Cluster", "Genes in cluster")
#: The two sheets of Supplementary Data 2 and the gene class each one holds.
CLUSTER_SHEETS: dict[str, str] = {
    "protein-coding genes": "protein_coding",
    "ncRNA-coding genes": "ncRNA_coding",
}

#: Spacer length of every designed guide, from the "20-mer" the Methods state.
SPACER_LENGTH = 20

# Counts measured on the pinned bytes; the build refuses anything else.
#: Guides in Supplementary Data 3, controls included.
N_LIBRARY_GUIDES = 56071
#: Gene-targeting guides of that library.
N_TARGETING_GUIDES = 55671
#: Non-targeting control guides of that library.
N_CONTROL_GUIDES = 400
#: Clusters in Supplementary Data 2.
N_CLUSTERS = 4205
#: Distinct member genes over those clusters.
N_CLUSTER_MEMBERS = 4317
#: Released (guide, screen) rows over the five screens.
EXPECTED_SOURCE_ROWS = 247348
#: Records the build writes.
EXPECTED_RECORDS = sum(screen.kept_records for screen in SCREENS)
#: Knockdown perturbations over those records (cluster members, retired ones removed).
EXPECTED_PERTURBATIONS = 242294
#: Distinct stored b-numbers over those perturbations.
EXPECTED_GENES = 4218

#: ``fitness / Z`` must be this flat within a screen for the sigma back-solve to hold.
SIGMA_RELATIVE_TOLERANCE = 1e-9

# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
_PAGE_METHODS_REAGENTS = "Methods, DNA manipulations and reagents"
_PAGE_METHODS_STRAINS = "Methods, Strain and plasmid construction"
_PAGE_METHODS_LIBRARY = "Methods, Design and preparation of the sgRNA library"
_PAGE_METHODS_SCREEN = "Methods, Screening experiments"
_PAGE_METHODS_NGS = "Methods, NGS data processing"
_PAGE_TABLE2 = "Table 2 The phenotypes studied in this work"
_PAGE_RESULTS_GENOMEWIDE = (
    "Results, Design and preparation of the genome-wide sgRNA library"
)
_PAGE_RESULTS_METABOLIC = "Results, Dissecting metabolic network via CRISPRi screening"
_PAGE_RESULTS_TOLERANCE = (
    "Results, Identification of genes carrying toxic chemical tolerance"
)


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


_Q_HOST = (
    "We transformed the sgRNA library by electroporation into E. coli strain MCm (a K12 "
    "MG1655 derivative with an integrated chloramphenicol-resistance cassette) carrying "
    "pdCas9-J23111"
)
_Q_MCM = (
    "E. coli strain $\\operatorname { M C m }$ , which was used in the screening "
    "experiments, was constructed by inserting a chloramphenicol expression cassette "
    "cloned from $\\mathsf { p K M 1 5 4 } ^ { 5 5 }$ (Addgene plasmid $\\# 1 3 0 3 6 )$ "
    "into the smf locus of wild-type E. coli K12 MG1655 by λ/RED recombineering56"
)
_Q_ANNOTATION = (
    "The E. coli K12 MG1655 genome sequence and relevant protein- or RNA-coding gene "
    "annotation of NC_000913.3 was used for the sgRNA library (20-mer) design"
)
_Q_DCAS9_PLASMID = (
    "The dCas9 expression plasmid was constructed by replacing the promoter and "
    "resistance marker region of Addgene plasmid $\\# 4 4 \\dot { 2 } 4 9 ^ "
    "{ \\bar { 2 } 4 }$ with a constitutive promoter (wild-type promoter for Cas9 from "
    "Streptococcus pyogenes)"
)
_Q_LIBRARY_SIZE = (
    "we designed a genome-scale CRISPRi sgRNA library consisting of 55,671 sgRNAs "
    "(Supplementary Data 3 and 4), as well as 400 negative control sgRNAs"
)
_Q_CLUSTERS = (
    "for genes with multiple copies in the genome, we used the BLASTN program to "
    "categorize genes with highly similar sequences into clusters (Supplementary Data "
    "2) and designed sgRNAs to target all members of a cluster. Hence, genes in one "
    "cluster are regarded as functionally identical"
)
_Q_GUIDE_NAMING = (
    "The sgRNAs were named “gene_p” according to the position (p) of the "
    "first guanine within the PAM region (NGG) in the coding region (e.g., rsmE_9, "
    "$\\mathrm { N } 2 0 =$ GTTCAGGATGATAAATGCGG)"
)
_Q_NEGATIVE_CONTROLS = (
    "we designed negative control sgRNAs by searching random 23-mers (with the format "
    "N20NGG) with the proper GC content to select those without any potential "
    "off-target candidates as identified by SeqMap"
)
_Q_REPLICATES = (
    "The library was independently transformed twice into either MCm/pdCas9-J23111 or "
    "MCm/pKanaNC, providing two biological replicates for each"
)
_Q_GEOMETRIC_MEAN = (
    "Subsequently, the read counts for each sgRNA in two biological replicates were "
    "averaged as the geometric mean"
)
_Q_ALL_TWO_REPLICATES = (
    "All experiments were carried out with two biological replicates"
)
_Q_FITNESS_EQUATIONS = (
    "we firstly calculated the fitness of each sgRNA for a phenotype (see Supplementary "
    "Fig. 7) by dividing the number of reads for this sgRNA in the corresponding "
    "selective condition by the number of reads in the relevant control condition and "
    "subsequently took the $\\log _ { 2 }$ value (equation 2). Then, the median of "
    "fitness for all negative control sgRNAs was determined and was used to normalize "
    "the fitness data for all sgRNAs, giving rise to the final sgRNA fitness data for "
    "this phenotype (equation 3)"
)
_Q_QUALITY = (
    "We annotated the quality of the sgRNA fitness by checking the read counts for each "
    "sgRNA in the control condition. Those sgRNAs with ${ < } 2 0$ reads were eliminated "
    "from the following analysis to calculate the gene fitness."
)
_Q_INITIAL_FILTER = (
    "Finally, sgRNAs with $< 2 0$ read counts in the initial library for each experiment "
    "were removed to increase statistical robustness (for the definition of the initial "
    "library for each experiment, see Supplementary Fig. 7)"
)
_Q_ZSCORE = (
    "we first fit the fitness for all negative control sgRNAs to a normal distribution, "
    "giving rise to the standard deviation $( \\sigma )$ . The $Z$ score for each sgRNA "
    "was then calculated by dividing the sgRNA fitness by the $\\sigma$ value (equation "
    "4)"
)
#: The two antibiotic doses straddle a paragraph break in the OCR, so the quote that is
#: verbatim in those bytes carries the break.
_Q_ANTIBIOTICS = (
    "Antibiotic concentrations for kanamycin and ampicillin were 50 and\n\n"
    "$1 0 0 \\mathrm { m g / L }$ , respectively."
)
_Q_MOPS = (
    "MOPS medium was prepared according to standard laboratory techniques53 "
    "$\\scriptstyle ( 1 0 \\mathrm { g } / \\mathrm { L }$ glucose). All cultures were "
    "carried out at $3 7 ^ { \\circ } \\mathrm { C }$"
)
_Q_OUTGROWTH = (
    "The remaining part of each sample was incubated with $1 0 0 \\mathrm { m L }$ LB "
    "broth (with kanamycin and ampicillin) in a $5 0 0 \\mathrm { - m L }$ flask with "
    "shaking at $3 7 ^ { \\circ } \\mathrm { C }$ until $\\mathrm { O D } _ { 6 0 0 } "
    "\\sim 1 . 0$ was reached $( \\sim 9 \\mathrm { h } )$ , allowing for around fifteen "
    "doublings"
)
_Q_SEED_MEDIA = (
    "used to seed cultures in 100 mL fresh medium in a $5 0 0 \\mathrm { - m L }$ flask "
    "with shaking (LB, MOPS, $\\mathbf { M O P S } + 0 . 5 \\ : \\mathbf { g } / \\mathrm "
    "{ L }$ casamino acid, $\\mathrm { M O P S } + 4 \\mathrm { g / L }$ isobutanol, "
    "$\\mathrm { M O P S } + 0 . 4 \\ : \\mathrm { g / L }$ furfural) with an initial "
    "$\\mathrm { O D } _ { 6 0 0 }$ of ${ \\sim } 0 . 0 3$ , thereby constructing two "
    "biological replicates for each tested phenotype"
)
_Q_FIVE_DOUBLINGS = (
    "we cultivated these cultures to $\\mathrm { O D } _ { 6 0 0 }$ of ${ \\sim } 1 . 0$ "
    ", thus allowing the cells to reproduce for around five doubling times for each "
    "experiment"
)
_Q_INITIAL_LIBRARY = (
    "The cultures representing the two replicates of MCm/pdCas9-J23111 with the sgRNA "
    "library were also mixed together, serving as the initial library for the following "
    "phenotypes to be tested (Supplementary Fig. 7)"
)
_Q_CASAMINO = (
    "we performed another screening with MOPS medium supplemented with $0 . 5 \\mathrm { "
    "g } / \\mathrm { L }$ casamino acid, which is composed of all amino acids except "
    "for tryptophan"
)
_Q_TOLERANCE_DOSES = (
    "We performed screening in MOPS medium with $0 . 4 \\mathrm { g / L }$ furfural or "
    "$4 \\mathrm { g / L }$ isobutanol to profile the chemical-tolerance profile in $E .$ "
    "coli at the genome level"
)
_Q_AMPLICON = (
    "Customized python scripts were then used to extract the 20-mer variable sequences "
    "from the raw NGS data via searching for the “GCACN20GTTT” 28_mer in the "
    "sequencing reads (and the reverse complementary sequence)"
)
_Q_SUBLIBRARIES = (
    "we divided the genome-scale sgRNA library into ten sublibraries according to the "
    "functions of the corresponding gene products (Supplementary Data 14). Customized "
    "barcode sequences were accordingly incorporated within the region flanking the N20 "
    "variable part of the library, enabling PCR amplification of these libraries "
    "separately from pooled DNA oligomers based on customized primers."
)
_Q_DATA_AVAILABILITY = (
    "NGS raw data of CRISPR screening results for the tiling library and genome-scale "
    "library can be accessed from the NCBI Short Read Archive with BioProject ID "
    "PRJNA450392"
)

REFERENCE_BACKGROUND = _paper(
    "MCm (K-12 MG1655 + chloramphenicol cassette in smf)",
    _Q_MCM,
    page=_PAGE_METHODS_STRAINS,
    note="the screening host is MG1655 carrying one marker cassette and two plasmids. "
    "AssemblyReferenceGenome.background is None and the cassette is NOT a perturbation: "
    "it is identical in all 240,481 records, the paper gives it no locus tag, and the "
    "library was designed against plain NC_000913.3, which is the assembly the stored "
    "b-numbers mean",
)
ANNOTATION = _paper(
    "NC_000913.3",
    _Q_ANNOTATION,
    page=_PAGE_METHODS_LIBRARY,
    note="the chromosome of GCA_000005845.2 (ASM584v2), which is the pinned MG1655 "
    "assembly set, so the released b-numbers need no crosswalk",
)
EFFECTOR = _paper(
    "dCas9",
    _Q_DCAS9_PLASMID,
    page=_PAGE_METHODS_STRAINS,
    note="the dead S. pyogenes Cas9 of pdCas9-J23111, written as the paper writes it; "
    "the Anderson promoter number is the expression level, not the effector identity",
)
N_REPLICATES = _paper(
    2,
    _Q_REPLICATES,
    page=_PAGE_METHODS_SCREEN,
    note=f"two independent transformations per arm, and every screen reuses them "
    f"('{_Q_ALL_TWO_REPLICATES}'); the two replicates' read counts are combined as a "
    f"geometric mean before the ratio is taken ('{_Q_GEOMETRIC_MEAN}'), so one record "
    "is one number over two biological replicates",
)
SAMPLE_UNIT = _paper(
    "biological_replicate",
    _Q_ALL_TWO_REPLICATES,
    page=_PAGE_METHODS_SCREEN,
    note="the source's own words for one replicate of a screen",
)
STATISTIC = _paper(
    "log2_ratio",
    _Q_FITNESS_EQUATIONS,
    page=_PAGE_METHODS_NGS,
    note="equations (2) and (3): log2(selective reads / control reads) minus the median "
    "of the non-targeting guides, which is why the reference response is exactly 0.0",
)
Z_SCALE = _paper(
    "fitness / sigma(non-targeting fitness)",
    _Q_ZSCORE,
    page=_PAGE_METHODS_NGS,
    note="equation (4). sigma is one constant per screen, so the released Z score is a "
    "deterministic rescaling of the stored fitness and is not stored again; the "
    "back-solved value per screen is Screen.nc_sigma",
)
QUALITY_RULE = _paper(
    "control-condition read count >= 20",
    _Q_QUALITY,
    page=_PAGE_METHODS_NGS,
    note=f"the drop rule for the 4,880 Bad rows. The two tolerance screens report no "
    f"Bad row because their control is the initial library, to which the same floor had "
    f"already been applied ('{_Q_INITIAL_FILTER}')",
)
LIBRARY_SIZE = _paper(
    {"targeting": N_TARGETING_GUIDES, "non_targeting": N_CONTROL_GUIDES},
    _Q_LIBRARY_SIZE,
    page=_PAGE_RESULTS_GENOMEWIDE,
    note="asserted against Supplementary Data 3 at build time",
)
CLUSTER_RULE = _paper(
    "one guide represses every member of its BLASTN cluster",
    _Q_CLUSTERS,
    page=_PAGE_RESULTS_GENOMEWIDE,
    note="why a record can carry more than one knockdown perturbation",
)
GUIDE_NAMING = _paper(
    "<gene><bnumber>_<PAM position>",
    _Q_GUIDE_NAMING,
    page=_PAGE_METHODS_LIBRARY,
    note="the Methods illustrate the tiling library's 'gene_p' form; the genome-scale "
    "ids carry the cluster representative's b-number between the two, which is where "
    "every stored systematic_gene_name comes from",
)
NON_TARGETING = _paper(
    N_CONTROL_GUIDES,
    _Q_NEGATIVE_CONTROLS,
    page=_PAGE_METHODS_LIBRARY,
    note="these guides have no genomic target, so no bacterial perturbation leaf can "
    "type them and their 1,942 rows are dropped; their median sets the zero of every "
    "stored fitness and their fitted sigma sets each screen's Z scale",
)
TEMPERATURE_C = _paper(
    37.0,
    _Q_MOPS,
    page=_PAGE_METHODS_REAGENTS,
    note="'All cultures were carried out at 37 C' covers every screen and control",
)
CARBON_SOURCE_G_PER_L = _paper(
    10.0,
    _Q_MOPS,
    page=_PAGE_METHODS_REAGENTS,
    note="the MOPS recipe is Neidhardt 1974 (the paper's reference 53), served as "
    "MEDIA_LIBRARY['MOPS_MINIMAL'], which is carbon-free; the stated glucose is this "
    "typed carbon-source factor so all four MOPS conditions share one base object",
)
KANAMYCIN_MG_PER_L = _paper(
    50.0,
    _Q_ANTIBIOTICS,
    page=_PAGE_METHODS_REAGENTS,
    note="stored as 50 ug/mL. The source prints mg/L, which is numerically identical to "
    "ug/mL (1 mg/L = 1 ug/mL exactly), and ConcentrationUnit.mg_per_l is a deliberately "
    "deferred member this loader does not add",
)
AMPICILLIN_MG_PER_L = _paper(
    100.0,
    _Q_ANTIBIOTICS,
    page=_PAGE_METHODS_REAGENTS,
    note="stored as 100 ug/mL, same unit identity as the kanamycin dose",
)
OUTGROWTH_HOURS = _paper(
    9.0,
    _Q_OUTGROWTH,
    page=_PAGE_METHODS_SCREEN,
    note="the LB culture that IS the essentiality screen and, mixed across the two "
    f"dCas9 replicates, the initial library of the two tolerance screens "
    f"('{_Q_INITIAL_LIBRARY}')",
)
OUTGROWTH_GENERATIONS = _paper(
    15.0,
    _Q_OUTGROWTH,
    page=_PAGE_METHODS_SCREEN,
    note="'around fifteen doublings', with the paper's own arithmetic beside it",
)
SCREEN_GENERATIONS = _paper(
    5.0,
    _Q_FIVE_DOUBLINGS,
    page=_PAGE_METHODS_SCREEN,
    note="the four re-seeded screens and their LB control culture; no wall-clock "
    "duration is stated for them, so duration_hours is a typed gap",
)
SELECTIVE_MEDIA = _paper(
    "LB, MOPS, MOPS + 0.5 g/L casamino acid, MOPS + 4 g/L isobutanol, MOPS + 0.4 g/L "
    "furfural",
    _Q_SEED_MEDIA,
    page=_PAGE_METHODS_SCREEN,
    note="the five seeded cultures; Table 2 pairs each phenotype with its selective and "
    "control condition",
)
TOLERANCE_DOSES = _paper(
    {"furfural": 0.4, "isobutanol": 4.0},
    _Q_TOLERANCE_DOSES,
    page=_PAGE_RESULTS_TOLERANCE,
    note="g/L in MOPS, the same doses Table 2's selective cells print",
)
CASAMINO_DOSE = _paper(
    0.5,
    _Q_CASAMINO,
    page=_PAGE_RESULTS_METABOLIC,
    note="g/L, a component of MEDIA_LIBRARY['MOPS_CASAMINO_WANG2018'] rather than an "
    "added compound: casamino acids is an acid hydrolysate of casein with no structure "
    "to resolve, so it belongs in the medium beside tryptone and yeast extract",
)
BARCODE = _paper(
    "the guide's own N20 spacer",
    _Q_AMPLICON,
    page=_PAGE_METHODS_NGS,
    note="why assay_type is pooled_competitive_growth_barcode: abundance is read out by "
    "amplifying and counting the variable 20-mer, which is the clone's barcode",
)
SINGLE_POOL = _paper(
    1,
    _Q_SUBLIBRARIES,
    page=_PAGE_METHODS_LIBRARY,
    note="CrisprConstruct.library_pool is None. The ten sublibraries are an "
    "oligo-amplification partition of ONE screened pool, not ten independently "
    "normalized screens, so recording them as a pool would split one measurement into "
    "ten strains for no reason",
)
RAW_READS = _paper(
    BIOPROJECT_ACCESSION,
    _Q_DATA_AVAILABILITY,
    page="Methods, Data availability",
    note="recorded, not deposited: no loader consumes reads",
)

#: The uncertainty fields no row carries. The released columns are a fitness and a Z
#: score, and the Z score's sigma is the NULL distribution's spread, not this
#: measurement's: the two replicates are combined as a geometric mean of read counts
#: BEFORE the ratio is taken, so no per-guide dispersion survives into the release.
_UNCERTAINTY_GAP_NOTE = (
    "the release carries a fitness and a Z score and no dispersion: the two biological "
    "replicates are combined as a geometric mean of read counts before the log2 ratio "
    "is taken, so no per-guide spread is published. The Z score's sigma is the standard "
    "deviation of the non-targeting guides' fitness, which is the null distribution's "
    "scale and not this record's uncertainty"
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
                method="Methods, NGS data processing (equations 1-4)",
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


def _duration_gap() -> ProvenanceGap:
    """``duration_hours`` of a re-seeded screen: stated in doublings, not hours."""
    return ProvenanceGap(
        field="duration_hours",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="Methods, Screening experiments",
            page=_PAGE_METHODS_SCREEN,
        ),
        note="the four re-seeded screens are dosed in doublings ('around five doubling "
        "times'), which duration_generations carries; no wall-clock time is stated for "
        "them, and the 9 h of the LB outgrowth belongs to a different culture",
    )


def _solvent_gap() -> ProvenanceGap:
    """``solvent`` of furfural or isobutanol: no vehicle is stated."""
    return ProvenanceGap(
        field="solvent",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="Methods, Screening experiments and Results, chemical tolerance",
            page=_PAGE_METHODS_SCREEN,
        ),
        note="both chemicals are stated as a g/L amount in MOPS with no vehicle named",
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
    """``$DATA_ROOT/torchcell-raw/wangPooledCRISPRInterference2018``."""
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
    """Deposit the seven consumed supplementary files and write ``manifest.json``.

    With ``sources=None`` the recorded retrieval runs for each file (``pmc_cloud_object``
    on the PMC Article Datasets bucket, which reproduced every pin on
    ``RAW_RETRIEVED_AT``); otherwise ``sources`` maps a file's ``relpath`` to an
    already-retrieved copy. Every file is hashed against its pin BEFORE anything is
    written, so a refusal leaves no partial deposit. Idempotent by sha256: a matching
    mirror file is left alone and a differing one raises rather than being overwritten.
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
            "Supplementary Data 4 (MOESM7, sgRNAs per gene) is NOT deposited: it is the "
            "per-cluster count of Supplementary Data 3, verified identical for all "
            f"{N_CLUSTERS} clusters, so it is derivable from a deposited file",
            "Supplementary Data 5 (MOESM8, gene fitness) is NOT deposited: it is the "
            "GENE-level call, whose FDRvalue and FPRvalue EnvironmentResponsePhenotype "
            "has no field for. Loading it would keep a median and drop the hit call, so "
            "the missing significance field is a written finding instead",
            "Supplementary Data 1, 11-14 (MOESM4, 14-17) are NOT deposited: the tiling "
            "library, the Keio essential-gene set, the other-microbe library designs and "
            "the sublibrary composition; no record reads them",
            "Supplementary Information PDFs (MOESM1-3) are NOT deposited here: they are "
            "in the torchcell-library mirror, where Table 2 and Supplementary Table 2 "
            "are quoted from",
            f"NCBI BioProject {BIOPROJECT_ACCESSION} holds the raw reads and is NOT "
            "deposited: no loader consumes reads",
            f"The authors' pipeline is at {CODE_REPOSITORY}; this loader reads the "
            "released tables and reimplements none of it",
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
    raise KeyError(f"{relpath} is not in the Wang 2018 raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
class ClusterMember(BaseModel):
    """One gene of one cluster, as Supplementary Data 2 spells it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str
    symbol: str
    bnumber: str


class GeneCluster(BaseModel):
    """One row of Supplementary Data 2: a guide target of one or more genes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str
    gene_class: str
    members: tuple[ClusterMember, ...]


def _member(token: str) -> ClusterMember:
    """Split a ``<symbol><bNNNN>`` token; an unparseable token is refused."""
    match = MEMBER_RE.match(token)
    if match is None:
        raise ValueError(f"{token!r} is not a <symbol><bNNNN> cluster member token")
    return ClusterMember(
        token=token, symbol=match.group("symbol"), bnumber=match.group("bnumber")
    )


def _sheet(path: str | Path, sheet: str, columns: Sequence[str]) -> pd.DataFrame:
    """Read one sheet as strings and require exactly ``columns``, in order."""
    frame = pd.read_excel(path, sheet_name=sheet, dtype=str)
    if tuple(frame.columns) != tuple(columns):
        raise ValueError(
            f"{osp.basename(str(path))}[{sheet!r}] has columns "
            f"{tuple(frame.columns)}, expected {tuple(columns)}"
        )
    return frame.dropna(how="all")


def read_clusters(
    path: str | Path,
    *,
    n_clusters: int = N_CLUSTERS,
    n_members: int = N_CLUSTER_MEMBERS,
) -> dict[str, GeneCluster]:
    """Supplementary Data 2 as ``cluster token -> GeneCluster``.

    The representative is the row's own ``Cluster`` token and must be its first member;
    a member in two clusters, or a cluster whose token is not a member, is refused,
    because the stored target set of a guide is read straight off this mapping. The two
    counts are the shape measured on the pinned bytes, parameterized only so a synthetic
    fixture can state its own.
    """
    clusters: dict[str, GeneCluster] = {}
    seen: dict[str, str] = {}
    for sheet, gene_class in CLUSTER_SHEETS.items():
        frame = _sheet(path, sheet, CLUSTER_COLUMNS)
        for token, members in zip(
            frame["Cluster"], frame["Genes in cluster"], strict=True
        ):
            parsed = tuple(_member(m) for m in str(members).split(","))
            if parsed[0].token != token:
                raise ValueError(
                    f"cluster {token!r} does not open with itself: {members!r}"
                )
            if token in clusters:
                raise ValueError(f"cluster {token!r} appears twice")
            for member in parsed:
                if member.token in seen:
                    raise ValueError(
                        f"{member.token!r} is in clusters {seen[member.token]!r} and "
                        f"{token!r}"
                    )
                seen[member.token] = str(token)
            clusters[str(token)] = GeneCluster(
                token=str(token), gene_class=gene_class, members=parsed
            )
    if (len(clusters), len(seen)) != (n_clusters, n_members):
        raise ValueError(
            f"Supplementary Data 2 parsed {len(clusters)} clusters over {len(seen)} "
            f"members, the paper's library is {n_clusters} over {n_members}"
        )
    return clusters


def read_library(
    path: str | Path,
    clusters: Mapping[str, GeneCluster],
    *,
    n_targeting: int = N_TARGETING_GUIDES,
    n_non_targeting: int = N_CONTROL_GUIDES,
) -> dict[str, str]:
    """Supplementary Data 3 as ``guide id -> 20-mer spacer``.

    Every gene-targeting id must name a cluster of ``clusters`` and every spacer must be
    a 20-mer, and the targeting / non-targeting split must be the library size the paper
    states. Non-targeting ids are kept in the mapping: the build counts them.
    """
    frame = _sheet(path, "sheet1", LIBRARY_COLUMNS)
    spacers = dict(
        zip(
            frame["sgRNAID"].astype(str),
            frame["nucleotide sequence"].astype(str),
            strict=True,
        )
    )
    if len(spacers) != len(frame):
        raise ValueError("Supplementary Data 3 repeats a guide id")
    bad = sorted(s for s in spacers.values() if len(s) != SPACER_LENGTH)
    if bad:
        raise ValueError(f"{len(bad)} spacers are not {SPACER_LENGTH}-mers: {bad[:5]}")
    targeting = [i for i in spacers if GUIDE_ID_RE.match(i) is not None]
    controls = [i for i in spacers if CONTROL_ID_RE.match(i) is not None]
    if len(targeting) + len(controls) != len(spacers):
        other = sorted(set(spacers) - set(targeting) - set(controls))
        raise ValueError(f"{len(other)} guide ids are neither form: {other[:5]}")
    if (len(targeting), len(controls)) != (n_targeting, n_non_targeting):
        raise ValueError(
            f"Supplementary Data 3 holds {len(targeting)} targeting and "
            f"{len(controls)} control guides, the paper states "
            f"{n_targeting} and {n_non_targeting}"
        )
    unknown = sorted(
        {
            str(GUIDE_ID_RE.match(i).group("token"))  # type: ignore[union-attr]  # matched above
            for i in targeting
        }
        - set(clusters)
    )
    if unknown:
        raise ValueError(f"{len(unknown)} guide tokens name no cluster: {unknown[:5]}")
    return spacers


class ScreenRows(BaseModel):
    """One screen's released rows, split by the drop rules, with its sigma re-checked."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    screen_id: str
    source_rows: int
    control_rows: int
    bad_rows: int
    control_and_bad_rows: int
    sigma: float
    candidates: pd.DataFrame


def back_solved_sigma(fitness: pd.Series, z_score: pd.Series) -> float:
    """``sigma`` of equation (4), back-solved from the released ``fitness / Z``.

    Equation (4) divides the fitness by one constant per screen, so the ratio is that
    constant on every row. Rows whose Z score is 0 carry no information about it and are
    skipped; a ratio that is not flat to ``SIGMA_RELATIVE_TOLERANCE`` means the released
    columns are not the equation the Methods state, so it is refused rather than averaged.
    """
    ratio = (fitness / z_score).replace([float("inf"), float("-inf")], pd.NA).dropna()
    if ratio.empty:
        raise ValueError("no row carries a non-zero Z score to back-solve sigma from")
    low, high = float(ratio.min()), float(ratio.max())
    middle = float(ratio.median())
    if high - low > SIGMA_RELATIVE_TOLERANCE * abs(middle):
        raise ValueError(
            f"fitness / Z spans [{low!r}, {high!r}], which is not one constant; "
            "equation (4) does not describe these columns"
        )
    return middle


def read_screen(path: str | Path, screen: Screen) -> ScreenRows:
    """One screen's sheet, with the control and Bad rows counted and removed.

    The ``gene`` cell of a non-targeting row must be the source's own ``0`` sentinel and
    a targeting row's cell must equal its id's symbol; both are checked, so a change in
    how the release names a row stops the build instead of being reinterpreted.
    """
    frame = _sheet(path, screen.sheet, FITNESS_COLUMNS)
    ids = frame["sgRNA"].astype(str)
    parsed = ids.str.extract(GUIDE_ID_RE)
    is_control = parsed["token"].isna()
    control_ids = ids[is_control]
    off_form = sorted(i for i in control_ids if CONTROL_ID_RE.match(i) is None)
    if off_form:
        raise ValueError(f"{len(off_form)} ids are neither form: {off_form[:5]}")
    control_cells = set(frame.loc[is_control, "gene"].astype(str))
    if control_cells not in ({CONTROL_GENE_CELL}, set()):
        raise ValueError(
            f"{screen.screen_id}: non-targeting rows name genes {sorted(control_cells)}, "
            f"not the source's {CONTROL_GENE_CELL!r} sentinel"
        )
    symbols = frame.loc[~is_control, "gene"].astype(str)
    token_symbols = parsed.loc[~is_control, "token"].map(
        lambda t: str(_member(str(t)).symbol)
    )
    disagree = int((symbols.to_numpy() != token_symbols.to_numpy()).sum())
    if disagree:
        raise ValueError(
            f"{screen.screen_id}: {disagree} rows whose gene cell is not their id's "
            "symbol"
        )
    fitness = pd.to_numeric(frame["sgRNA fitness"])
    z_score = pd.to_numeric(frame["Z score"])
    if fitness.isna().any() or z_score.isna().any():
        raise ValueError(f"{screen.screen_id}: a fitness or Z score is not a number")
    sigma = back_solved_sigma(fitness, z_score)
    if abs(sigma - screen.nc_sigma) > SIGMA_RELATIVE_TOLERANCE * abs(screen.nc_sigma):
        raise ValueError(
            f"{screen.screen_id}: back-solved sigma {sigma!r} is not the recorded "
            f"{screen.nc_sigma!r}"
        )
    quality = frame["Quality"].astype(str)
    unknown_quality = sorted(set(quality) - set(QUALITY_VALUES))
    if unknown_quality:
        raise ValueError(
            f"{screen.screen_id}: Quality cells {unknown_quality} are outside "
            f"{list(QUALITY_VALUES)}; a third value would be silently kept as good"
        )
    is_bad = quality == BAD_QUALITY
    candidates = pd.DataFrame(
        {
            "guide_id": ids[~is_control & ~is_bad].to_numpy(),
            "token": parsed.loc[~is_control & ~is_bad, "token"].to_numpy(),
            "fitness": fitness[~is_control & ~is_bad].to_numpy(),
        }
    )
    return ScreenRows(
        screen_id=screen.screen_id,
        source_rows=len(frame),
        control_rows=int(is_control.sum()),
        bad_rows=int(is_bad.sum()),
        control_and_bad_rows=int((is_control & is_bad).sum()),
        sigma=sigma,
        candidates=candidates,
    )


# --------------------------------------------------------------------------- #
# Record parts
# --------------------------------------------------------------------------- #
def publication() -> Publication:
    """This paper, by DOI.

    ``pubmed_id`` is left ``None``: the PubMed id is in none of the mirrored bytes nor in
    the bibliography store's entry for this key, and the NCBI id converter answered HTTP
    429 when asked, so it is not sourced and is not asserted.
    """
    return Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")


def host_reference(data_root: str | None = None) -> AssemblyReferenceGenome:
    """Plain MG1655, pinned to ``GCA_000005845.2``, with no asserted background.

    The screening host MCm carries a chloramphenicol cassette in ``smf`` and two
    plasmids, but the library was designed against plain ``NC_000913.3`` and the paper
    gives the cassette no locus tag. A ``BacterialStrainBackground`` here would assert a
    strain the stored b-numbers are not written against, so ``background`` is ``None``
    and the cassette is recorded in :data:`REFERENCE_BACKGROUND` and in the note.
    """
    return assembly_reference(MG1655_STRAIN, data_root=data_root)


def _antibiotics() -> list[EnvironmentPerturbationType]:
    """The kanamycin and ampicillin of the LB outgrowth, at the stated amounts."""
    return [
        SmallMoleculePerturbation(
            compound=resolved_compound(name),
            concentration=Concentration(value=dose, unit=ConcentrationUnit.ug_per_ml),
            solvent=None,
        )
        for name, dose in (
            ("kanamycin", float(KANAMYCIN_MG_PER_L.value)),
            ("ampicillin", float(AMPICILLIN_MG_PER_L.value)),
        )
    ]


def _carbon_source() -> EnvironmentPhysicalPerturbation:
    """The 10 g/L glucose of Wang's MOPS, as a typed carbon-source factor."""
    return EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.carbon_source,
        magnitude=Concentration(
            value=float(CARBON_SOURCE_G_PER_L.value), unit=ConcentrationUnit.g_per_l
        ),
        agent=resolved_compound("D-glucose"),
    )


def _stressor(name: str) -> SmallMoleculePerturbation:
    """Furfural or isobutanol at the g/L dose Table 2 prints."""
    doses: dict[str, float] = dict(TOLERANCE_DOSES.value)
    return SmallMoleculePerturbation(
        compound=resolved_compound(name),
        concentration=Concentration(value=doses[name], unit=ConcentrationUnit.g_per_l),
        solvent=None,
        provenance_gaps=[_solvent_gap()],
    )


def _outgrowth_environment() -> Environment:
    """The LB culture with kanamycin and ampicillin, grown ~9 h to OD600 ~1.0.

    This one culture is three things: the essentiality screen's selective arm
    (MCm/pdCas9-J23111), its control arm (MCm/pKanaNC, which differs by the PLASMID and
    not by the medium), and -- the two dCas9 replicates mixed -- the initial library the
    furfural and isobutanol screens are scored against.
    """
    return Environment(
        media=LB,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        perturbations=_antibiotics(),
        aerobicity="aerobic",
        duration_hours=float(OUTGROWTH_HOURS.value),
        duration_generations=float(OUTGROWTH_GENERATIONS.value),
    )


def _reseeded_environment(
    media: Any, perturbations: list[EnvironmentPerturbationType]
) -> Environment:
    """One of the five re-seeded screening cultures, dosed in doublings."""
    return Environment(
        media=media,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_hours=None,
        duration_generations=float(SCREEN_GENERATIONS.value),
        provenance_gaps=[_duration_gap()],
    )


def selective_environment(screen: Screen) -> Environment:
    """The environment Table 2 names as ``screen``'s selective condition."""
    if screen.screen_id == "essentiality":
        return _outgrowth_environment()
    if screen.screen_id == "auxotrophy":
        return _reseeded_environment(MOPS_MINIMAL, [_carbon_source()])
    if screen.screen_id == "trp_biosynthesis":
        return _reseeded_environment(MOPS_CASAMINO_WANG2018, [_carbon_source()])
    if screen.screen_id == "furfural_tolerance":
        return _reseeded_environment(
            MOPS_MINIMAL, [_carbon_source(), _stressor("furfural")]
        )
    if screen.screen_id == "isobutanol_tolerance":
        return _reseeded_environment(
            MOPS_MINIMAL, [_carbon_source(), _stressor("isobutanol")]
        )
    raise ValueError(f"{screen.screen_id!r} is not one of this paper's five screens")


def control_environment(screen: Screen) -> Environment:
    """The environment Table 2 names as ``screen``'s control condition.

    The LB outgrowth culture for the essentiality screen (whose ``Empty plasmid, LB``
    control arm differs by the PLASMID and not by the medium) and for the two tolerance
    screens (whose control is the initial library that culture became); a re-seeded LB
    culture for the two screens Table 2 controls on ``LB``. That re-seeded control
    carries no antibiotic: the Methods state kanamycin and ampicillin for the outgrowth
    and restate them for none of the five seeded cultures.
    """
    if screen.screen_id in (
        "essentiality",
        "furfural_tolerance",
        "isobutanol_tolerance",
    ):
        return _outgrowth_environment()
    if screen.screen_id in ("auxotrophy", "trp_biosynthesis"):
        return _reseeded_environment(LB, [])
    raise ValueError(f"{screen.screen_id!r} is not one of this paper's five screens")


_UNITS = (
    "log2(selective reads / control reads) of one sgRNA, minus the median of the 400 "
    "non-targeting control sgRNAs (Methods equations 2 and 3); negative = the knockdown "
    "is depleted under selection"
)
_UNITS_REFERENCE = (
    "the non-targeting control guides of the same screen, whose median IS the zero of "
    "equation (3), so the reference response is exactly 0.0"
)


def guide_phenotype(screen: Screen, fitness: float) -> EnvironmentResponsePhenotype:
    """One guide's released fitness in one screen."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=fitness,
        n_samples=int(N_REPLICATES.value),
        sample_unit=SampleUnit(str(SAMPLE_UNIT.value)),
        units=_UNITS,
        screen_id=screen.screen_id,
        provenance_gaps=_uncertainty_gaps(),
    )


def build_reference(
    dataset_name: str, screen: Screen, genome_reference: AssemblyReferenceGenome
) -> BacterialEnvironmentResponseExperimentReference:
    """The non-targeting baseline of one screen: fitness 0 by the normalization."""
    return BacterialEnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=control_environment(screen),
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            n_samples=int(N_REPLICATES.value),
            sample_unit=SampleUnit(str(SAMPLE_UNIT.value)),
            units=_UNITS_REFERENCE,
            screen_id=screen.screen_id,
            provenance_gaps=_uncertainty_gaps(),
        ),
    )


def build_genotype(targets: Sequence[tuple[str, str]], spacer: str) -> Genotype:
    """One guide repressing every ``(b-number, gene name)`` of its cluster."""
    return Genotype(
        perturbations=[
            BacterialCrisprInterferencePerturbation(
                systematic_gene_name=bnumber,
                perturbed_gene_name=gene_name,
                gene_namespace=MG1655_NAMESPACE,
                identifier_mapping=None,
                crispr=CrisprConstruct(
                    effector=str(EFFECTOR.value),
                    guide_sequence=spacer,
                    n_guides=1,
                    library_pool=None,
                ),
            )
            for bnumber, gene_name in targets
        ]
    )


def stored_gene_names(
    genome: EcoliK12Genome, clusters: Iterable[GeneCluster]
) -> dict[str, str]:
    """``b-number -> perturbed_gene_name``: the annotation's symbol where it round-trips.

    A symbol is used only when the genome resolves it back to the same b-number as a
    current gene, so the stored common name always resolves to the stored locus. Where
    the annotation names no symbol for a locus the source's own symbol is kept, which is
    what the four retired tags and the pseudogene loci rely on.
    """
    names: dict[str, str] = {}
    for cluster in clusters:
        for member in cluster.members:
            locus = genome.genbank.loci.get(member.bnumber)
            symbol = locus.symbol if locus is not None else None
            if symbol:
                resolution = genome.resolve_gene_name(symbol)
                if (
                    resolution.systematic_name == member.bnumber
                    and resolution.status
                    in (GeneNameStatus.CURRENT, GeneNameStatus.RENAMED)
                ):
                    names[member.bnumber] = str(symbol)
                    continue
            names[member.bnumber] = member.symbol
    return names


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class ScreenAccounting(BaseModel):
    """One screen's retention arithmetic and its back-solved Z scale."""

    model_config = ConfigDict(extra="forbid")

    screen_id: str
    phenotype_label: str
    selective_condition: str
    control_condition: str
    supplementary_data: int
    source_rows: int
    dropped_non_targeting: int
    dropped_bad_quality: int
    dropped_non_targeting_and_bad: int
    dropped_retired_target: int
    kept_records: int
    perturbations: int
    nc_sigma: float

    def check(self) -> None:
        """The four drop counts must account for every released row."""
        kept = (
            self.source_rows
            - self.dropped_non_targeting
            - self.dropped_bad_quality
            + self.dropped_non_targeting_and_bad
            - self.dropped_retired_target
        )
        if kept != self.kept_records:
            raise ValueError(
                f"{self.screen_id}: {self.source_rows} rows minus the drops is {kept}, "
                f"not the {self.kept_records} records written"
            )


class BuildAccounting(BaseModel):
    """Everything a reader needs to audit one build's retention arithmetic."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    reference_strain: str
    assembly_set: str
    library_guides: int
    library_targeting_guides: int
    library_non_targeting_guides: int
    clusters: int
    cluster_members: int
    multi_member_clusters: int
    largest_cluster: int
    screens: list[ScreenAccounting]
    source_rows: int
    kept_records: int
    perturbations: int
    distinct_genes: int
    retired_targets_dropped: list[str]
    pseudogene_targets_kept: list[str]
    source_symbol_disagreements: int
    reconciliation: LocusTagReconciliation
    drop_rules: list[str]
    notes: list[str]

    def check(self) -> None:
        """Every screen's arithmetic, and the totals over the screens."""
        for screen in self.screens:
            screen.check()
        totals = (
            sum(s.source_rows for s in self.screens),
            sum(s.kept_records for s in self.screens),
            sum(s.perturbations for s in self.screens),
        )
        if totals != (self.source_rows, self.kept_records, self.perturbations):
            raise ValueError(
                f"per-screen totals {totals} do not sum to "
                f"{(self.source_rows, self.kept_records, self.perturbations)}"
            )


DROP_RULES: list[str] = [
    "non_targeting: the sgRNA id is NC_<n> and the gene cell is the source's '0' "
    "sentinel. A guide with no genomic target cannot be a "
    "BacterialCrisprInterferencePerturbation and no non-targeting leaf exists; the rows "
    "still set the zero of equation (3) and each screen's Z sigma, both recorded here",
    "bad_quality: the Quality cell is 'Bad', the paper's own flag for a "
    "control-condition read count below 20, which it excludes from every downstream "
    "statistic; no phenotype field can carry the flag",
    "retired_target: every member of the guide's cluster is a b-number the pinned "
    "GCA_000005845.2 annotation no longer carries, so the L1 canonical-name rule and the "
    "L4 gene universe would both reject the stored name",
]


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class CrispriGuideFitnessWang2018Dataset(ExperimentDataset):
    """Wang 2018 per-guide CRISPRi fitness across the paper's five pooled screens."""

    REFERENCE_STRAIN: ClassVar[Literal["MG1655"]] = "MG1655"
    #: Measured on the pinned bytes: 4,313 of the 4,317 cluster-member b-numbers resolve
    #: to a locus of GCA_000005845.2 (0.9991). Below this the annotation or the release
    #: moved and the build stops rather than dropping more records.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 0.999

    def __init__(
        self,
        root: str = "data/torchcell/crispri_guide_fitness_wang2018",
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
        """The seven pinned supplementary files this dataset reads."""
        return [raw.filename for raw in RAW_FILES]

    def download(self) -> None:
        """Link the mirror's files into ``raw/`` after verifying each against the manifest."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        root = raw_mirror_dir(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for raw in RAW_FILES:
            recorded = manifest_sha256(manifest, raw.relpath)
            check_manifest_pin(raw.relpath, recorded, raw.sha256)
            source = root / raw.relpath
            if not source.exists():
                raise RuntimeError(
                    f"required raw artifact missing from mirror: {source}"
                )
            link_verified(source, osp.join(self.raw_dir, raw.filename), recorded)
        log.info("Wang 2018 artifacts linked into %s", self.raw_dir)

    def _genome(self) -> EcoliK12Genome:
        """The injected MG1655 genome, or one opened from the genomes tier.

        A genome of another K-12 assembly set is refused: a ``BW25113_`` tag is not a
        b-number, so resolving this release's identifiers against it would be nonsense.
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

    def _pointer(self, obj: Any, hint: str, itxn: Any) -> Any:
        """``obj``'s dump, interned exactly as ``_intern_record`` interns it."""
        holder = {"value": obj.model_dump()}
        self._maybe_intern(holder, "value", obj, hint, itxn)
        return holder["value"]

    @post_process
    def process(self) -> None:
        """Build one record per (guide, screen) row; write LMDB."""
        verify_raw_files(self.raw_dir, {raw.filename: raw.sha256 for raw in RAW_FILES})
        clusters = read_clusters(
            osp.join(self.raw_dir, CLUSTERS_FILE.filename),
            n_clusters=N_CLUSTERS,
            n_members=N_CLUSTER_MEMBERS,
        )
        spacers = read_library(
            osp.join(self.raw_dir, LIBRARY_FILE.filename),
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
        loci = set(genome.genbank.loci)
        names = stored_gene_names(genome, clusters.values())
        targets: dict[str, tuple[tuple[str, str], ...]] = {
            token: tuple(
                (m.bnumber, names[m.bnumber])
                for m in cluster.members
                if m.bnumber not in retired
            )
            for token, cluster in clusters.items()
        }
        genome_reference = host_reference()
        pub = publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        accounting: list[ScreenAccounting] = []
        genes: set[str] = set()
        idx = 0
        for screen in SCREENS:
            rows = read_screen(osp.join(self.raw_dir, screen.raw.filename), screen)
            environment = selective_environment(screen)
            reference = build_reference(self.name, screen, genome_reference)
            genotypes: dict[str, dict[str, Any]] = {}
            dropped_retired = 0
            perturbations = 0
            with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
                env_ptr = self._pointer(environment, environment.media.name, itxn)
                ref_ptr = self._pointer(reference, reference.dataset_name, itxn)
                pub_ptr = self._pointer(pub, "publication", itxn)
                first = True
                for guide_id, token, fitness in tqdm(
                    zip(
                        rows.candidates["guide_id"],
                        rows.candidates["token"],
                        rows.candidates["fitness"],
                        strict=True,
                    ),
                    total=len(rows.candidates),
                    desc=f"{self.name}:{screen.screen_id}",
                ):
                    target = targets[str(token)]
                    if not target:
                        dropped_retired += 1
                        continue
                    key = str(guide_id)
                    if key not in genotypes:
                        genotypes[key] = build_genotype(
                            target, spacers[key]
                        ).model_dump()
                    genotype = genotypes[key]
                    phenotype = guide_phenotype(screen, float(fitness))
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
                    perturbations += len(target)
                    genes.update(bnumber for bnumber, _ in target)
            kept = len(rows.candidates) - dropped_retired
            if kept != screen.kept_records:
                raise RuntimeError(
                    f"{screen.screen_id}: wrote {kept} records, the pinned bytes give "
                    f"{screen.kept_records}"
                )
            accounting.append(
                ScreenAccounting(
                    screen_id=screen.screen_id,
                    phenotype_label=screen.phenotype_label,
                    selective_condition=screen.selective,
                    control_condition=screen.control,
                    supplementary_data=screen.raw.data_number,
                    source_rows=rows.source_rows,
                    dropped_non_targeting=rows.control_rows,
                    dropped_bad_quality=rows.bad_rows,
                    dropped_non_targeting_and_bad=rows.control_and_bad_rows,
                    dropped_retired_target=dropped_retired,
                    kept_records=kept,
                    perturbations=perturbations,
                    nc_sigma=rows.sigma,
                )
            )
        env.close()
        interned_env.close()

        totals = (idx, sum(s.perturbations for s in accounting), len(genes))
        expected = (EXPECTED_RECORDS, EXPECTED_PERTURBATIONS, EXPECTED_GENES)
        if totals != expected:
            raise RuntimeError(
                f"{self.name}: wrote (records, perturbations, genes) {totals}, the "
                f"pinned bytes give {expected}"
            )
        self._write_reports(
            clusters, spacers, names, reconciliation, accounting, genes, retired, loci
        )
        log.info(
            "Wrote %d %s records over %d distinct genes", idx, self.name, len(genes)
        )

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
        names: Mapping[str, str],
        reconciliation: LocusTagReconciliation,
        screens: list[ScreenAccounting],
        genes: set[str],
        retired: frozenset[str],
        loci: set[str],
    ) -> None:
        """Write the guide-library table and the build accounting."""
        rows = [
            {
                "guide_id": guide_id,
                "spacer": spacer,
                "cluster_token": (
                    match.group("token")
                    if (match := GUIDE_ID_RE.match(guide_id))
                    else ""
                ),
                "pam_position": match.group("position") if match else "",
                "is_non_targeting": match is None,
            }
            for guide_id, spacer in spacers.items()
        ]
        for row in rows:
            cluster = clusters.get(str(row["cluster_token"]))
            row["gene_class"] = cluster.gene_class if cluster else ""
            row["cluster_members"] = (
                ";".join(m.bnumber for m in cluster.members) if cluster else ""
            )
            row["source_symbols"] = (
                ";".join(m.symbol for m in cluster.members) if cluster else ""
            )
            row["stored_gene_names"] = (
                ";".join(names[m.bnumber] for m in cluster.members) if cluster else ""
            )
        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "guide_library.csv"), index=False
        )
        sizes = Counter(len(c.members) for c in clusters.values())
        disagreements = sum(
            1
            for cluster in clusters.values()
            for m in cluster.members
            if names[m.bnumber] != m.symbol
        )
        accounting = BuildAccounting(
            dataset=self.name,
            reference_strain=self.REFERENCE_STRAIN,
            assembly_set=reconciliation.assembly_set,
            library_guides=len(spacers),
            library_targeting_guides=N_TARGETING_GUIDES,
            library_non_targeting_guides=N_CONTROL_GUIDES,
            clusters=len(clusters),
            cluster_members=sum(len(c.members) for c in clusters.values()),
            multi_member_clusters=sum(n for size, n in sizes.items() if size > 1),
            largest_cluster=max(sizes),
            screens=screens,
            source_rows=sum(s.source_rows for s in screens),
            kept_records=sum(s.kept_records for s in screens),
            perturbations=sum(s.perturbations for s in screens),
            distinct_genes=len(genes),
            retired_targets_dropped=sorted(retired),
            pseudogene_targets_kept=sorted(
                b
                for b in genes
                if b in loci
                and self._genome().resolve_gene_name(b).status
                is GeneNameStatus.NON_GENE_FEATURE
            ),
            source_symbol_disagreements=disagreements,
            reconciliation=reconciliation,
            drop_rules=DROP_RULES,
            notes=[
                "one record is one (guide, screen) row of Supplementary Data 6-10; the "
                "five screens are kept apart by phenotype.screen_id",
                "a guide targeting a multi-member BLASTN cluster carries one knockdown "
                "perturbation per member, which is why perturbations exceed records",
                "the released Z score is fitness / nc_sigma with nc_sigma constant per "
                "screen, so it is not stored twice; the back-solved value is on each "
                "screen here and the build refuses a ratio that is not flat",
                "Supplementary Data 5's gene-level FDR and FPR have no field on "
                "EnvironmentResponsePhenotype, so the gene level is not loaded and the "
                "missing significance field is a written finding",
                "the host MCm's chloramphenicol cassette in smf is constant in every "
                "record and carries no locus tag, so it is not a perturbation and the "
                "assembly reference asserts no background",
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
            "CrispriGuideFitnessWang2018Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification (L0-L4)
# --------------------------------------------------------------------------- #
#: The assembly pin every reference of this dataset must carry.
EXPECTED_ASSEMBLY_PIN = ("ecoli_K12_MG1655_ASM584v2", "GCA_000005845.2", "None")


class SupplementaryRows:
    """The three loader-specific verifier rows, accumulated in ONE pass.

    An accumulator rather than three functions over a materialized list, for the same
    reason the environment-response verifier streams: 240,481 records are not held in
    RAM to answer three counting questions. ``add`` folds one stored record in and
    ``results`` renders the rows in level order.
    """

    def __init__(self) -> None:
        """Start an empty accumulation for one dataset."""
        self.screen_counts: Counter[str] = Counter()
        self.effectors: Counter[str] = Counter()
        self.missing_spacers = 0
        self.pins: set[tuple[str, str, str]] = set()

    def add(self, record: Mapping[str, Any]) -> None:
        """Fold one ``{"experiment": ..., "reference": ...}`` record in."""
        experiment = record["experiment"]
        self.screen_counts[str(experiment["phenotype"]["screen_id"])] += 1
        for perturbation in experiment["genotype"]["perturbations"]:
            crispr = perturbation.get("crispr") or {}
            self.effectors[str(crispr.get("effector"))] += 1
            spacer = crispr.get("guide_sequence")
            if not isinstance(spacer, str) or len(spacer) != SPACER_LENGTH:
                self.missing_spacers += 1
        reference = record["reference"]["genome_reference"]
        self.pins.add(
            (
                str(reference.get("assembly_set")),
                str(reference.get("assembly_accession")),
                str(reference.get("background")),
            )
        )

    def add_all(self, records: Iterable[Mapping[str, Any]]) -> SupplementaryRows:
        """Fold a sequence or a stream of records in; returns self."""
        for record in records:
            self.add(record)
        return self

    def screen_coverage(self) -> LevelResult:
        """L1 SUPPLEMENTARY: every screen is present at its measured record count."""
        observed = dict(sorted(self.screen_counts.items()))
        expected = {screen.screen_id: screen.kept_records for screen in SCREENS}
        return LevelResult(
            level=Level.L1,
            name="every_screen_at_its_measured_record_count",
            passed=observed == expected,
            message=(f"SUPPLEMENTARY: {len(observed)} screens, records {observed}"),
            details={"observed": observed, "expected": expected},
        )

    def guide_spacers(self) -> LevelResult:
        """L1 SUPPLEMENTARY: every knockdown carries a 20-mer spacer and the effector.

        The spacer is what makes two guides of one gene two strains rather than two
        duplicates, so a record without it would silently collapse the L1 strain key.
        """
        total = sum(self.effectors.values())
        return LevelResult(
            level=Level.L1,
            name="every_knockdown_carries_its_20mer_spacer",
            passed=self.missing_spacers == 0
            and set(self.effectors) == {str(EFFECTOR.value)},
            message=(
                f"SUPPLEMENTARY: {total} knockdowns, {self.missing_spacers} without a "
                f"{SPACER_LENGTH}-mer spacer; effectors {dict(self.effectors)}"
            ),
            details={
                "n_perturbations": total,
                "n_missing": self.missing_spacers,
                "effectors": dict(self.effectors),
            },
        )

    def assembly_pin(self) -> LevelResult:
        """L3 SUPPLEMENTARY: every reference pins the same MG1655 GenBank assembly."""
        return LevelResult(
            level=Level.L3,
            name="assembly_pin_is_mg1655_genbank_with_no_asserted_background",
            passed=self.pins == {EXPECTED_ASSEMBLY_PIN},
            message=(
                f"SUPPLEMENTARY: {len(self.pins)} distinct assembly pin(s): "
                f"{sorted(self.pins)}"
            ),
            details={"pins": sorted(self.pins)},
        )

    def results(self) -> list[LevelResult]:
        """The three rows, in level order."""
        return [self.screen_coverage(), self.guide_spacers(), self.assembly_pin()]


def verify_build(
    dataset_root: str,
    *,
    genome: EcoliK12Genome | None = None,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 verifier on a built tree and write its report.

    The LMDB is streamed twice and never materialized: once for the shared verifier and
    once for :class:`SupplementaryRows`. Every name is checked against the MG1655 genome
    its references pin: the resolver of the canonical-name rule, and as the L4 universe
    every GenBank locus of the assembly (4,651, pseudogenes and RNA tags included). The
    three SUPPLEMENTARY rows are appended; the verifier's own rows keep their verdicts.
    The report is written to ``preprocess/verification_report.json``.
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
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/si/si_data/",
            citation_key=CITATION_KEY,
            sha256=LIBRARY_FILE.sha256,
            method=(
                "Supplementary Data 6-10 (one sheet each, MOESM9-13): one "
                "BacterialEnvironmentResponseExperiment per (sgRNA, screen) row. "
                "measurement_type=log2_ratio, assay_type="
                "pooled_competitive_growth_barcode (the barcode is the guide's own N20 "
                "spacer), environment_response = the released 'sgRNA fitness' = "
                "log2(selective reads / control reads) minus the median of the 400 "
                "non-targeting guides (equations 2 and 3). n_samples=2 biological "
                "replicates combined as a geometric mean of read counts before the "
                "ratio, so no dispersion survives and the uncertainty fields are typed "
                "ProvenanceGaps. Genotype = one "
                "BacterialCrisprInterferencePerturbation per member of the guide's "
                "BLASTN cluster (Supplementary Data 2), effector dCas9, guide_sequence "
                "= the 20-mer spacer of Supplementary Data 3 (MOESM6, sha256 "
                f"{LIBRARY_FILE.sha256}), library_pool None (one screened pool). "
                "Reference = the non-targeting baseline of the same screen, response "
                "0.0 by that normalization, on Table 2's control condition. DROPPED: "
                "1,942 non-targeting rows (no gene to type), 4,880 Quality=='Bad' rows "
                "(control-condition read count below the paper's own 20-read floor) and "
                "55 rows whose only target is a retired b-number (b4590, b4700)"
            ),
            page=(
                "Nat Commun 2018 9:2475 (doi:10.1038/s41467-018-04899-x); Table 2, "
                "Methods 'NGS data processing' and 'Screening experiments'; "
                f"Supplementary Data 2, 3 and 6-10; paper.md sha256={PAPER_MD_SHA256}"
            ),
            retrieved=RAW_RETRIEVED_AT,
        ),
        expected_count=expected_count,
        sgd_genes=set(genome.genbank.loci),
        resolve_gene_name=genome.resolve_gene_name,
    )
    for result in SupplementaryRows().add_all(stream_records(dataset_root)).results():
        report.add(result)
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build the dataset under ``DATA_ROOT``, print its accounting and verify it."""
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/crispri_guide_fitness_wang2018")
    genome = bacterial_genome("ecoli", MG1655_STRAIN, data_root)
    dataset = CrispriGuideFitnessWang2018Dataset(root=root, ecoli_genome=genome)
    print(f"len = {len(dataset)}")
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
                    "retired_targets_dropped",
                    "pseudogene_targets_kept",
                    "source_symbol_disagreements",
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
