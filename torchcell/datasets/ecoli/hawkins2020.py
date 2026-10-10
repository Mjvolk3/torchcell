# torchcell/datasets/ecoli/hawkins2020
# [[torchcell.datasets.ecoli.hawkins2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/hawkins2020
# Test file: tests/torchcell/datasets/ecoli/test_hawkins2020.py
r"""Hawkins 2020 mismatch-CRISPRi: per-sgRNA relative fitness of 317 E. coli essential genes.

Hawkins et al. 2020 (Cell Systems 11:523-535.e9, doi:10.1016/j.cels.2020.09.009,
citation key ``hawkinsMismatchCRISPRiRevealsCovarying2020``) titrated CRISPRi knockdown
by putting single mismatches into the guide's base-pairing region, so one essential gene
is targeted by a graded series of guides rather than by one switch.
:class:`MismatchCrispriFitnessHawkins2020Dataset` serves the E. coli arm of Table S3 as
one ``BacterialEnvironmentResponseExperiment`` per released sgRNA that carries a measured
relative fitness.

WHAT IS MEASURED AND WHAT IS PREDICTED, WHICH IS THE WHOLE POINT OF THIS ROW. Table S3's
E. coli sheet carries both, side by side, and this loader stores only the first:

- MEASURED, and stored: ``relative fitness (mean)`` and ``relative fitness (stddev)``,
  the mean and standard deviation over 4 biological replicates of one sgRNA's relative
  fitness after about 10 doublings of pooled competition.
- PREDICTED, and NOT stored: ``relative fitness (predicted)``, which despite its column
  name is the linear model's predicted sgRNA ACTIVITY, not a predicted fitness. Measured
  on the pinned bytes: it is exactly 1.0 for all 3,110 fully complementary spacers (the
  model's own definition, ``PREDICTED_ACTIVITY``) and spreads over -0.151 to 1.400 for
  the 33,180 singly mismatched ones. It is a model output at species-averaged R^2 0.56
  with an 11-fold cross-validation mean squared error of 0.10 +/- 0.08
  (``MODEL_FIT``), and a PRELIMINARY version of that same model chose the library's
  mismatches (``PRELIMINARY_MODEL``), so the dose is both imputed and entangled with the
  design. The schema has no field for a perturbation-level covariate that is a model
  prediction with its own error, so storing this number in any existing slot would
  present a prediction as a measurement. It is refused here and filed as a schema
  finding; every value stays recoverable in ``preprocess/guide_retention.csv``.

WHAT THE RECORD CARRIES INSTEAD OF THE DOSE. The guide's MISMATCH DESIGN, which is a
property of the construct we actually built and is therefore a measurement of the
genotype: the perturbation's ``description`` names the parent (fully complementary)
spacer, the 0-based mismatch position counted from the spacer's 5' end, and the base
substitution, and ``crispr.guide_sequence`` holds the 20-nt mismatched spacer itself. A
consumer that wants the dose can recompute the published linear model from the spacer
pair; a consumer that wants the measurement reads the phenotype.

RECORD = one released sgRNA with a measured fitness:

- GENOTYPE: one ``BacterialCrisprInterferencePerturbation`` keyed by the target gene's
  BW25113 locus tag, carrying the mismatched spacer on the shared ``crispr`` construct
  (``effector="dCas9"``, the S. pyogenes dCas9 of the Tn7att PlLac-O1 cassette).
- ENVIRONMENT: LB with 100 ug/mL ampicillin at 37 C, 1 mM IPTG inducing dCas9, over about
  10 doublings of exponential growth maintained by back dilution.
- PHENOTYPE: ``EnvironmentResponsePhenotype`` with
  ``measurement_type=relative_growth_rate``: the released number is the strain's
  doublings divided by the wild type's over the same time course, so 1.0 is wild-type
  growth and below 1.0 is slower (``FITNESS_DEFINITION``).

WHY ``EnvironmentResponsePhenotype`` AND NOT ``FitnessPhenotype``. 778 of the 24,149
stored values are negative, which is the release's own meaning for a clone actively
depleted from the pool rather than merely outgrown (``NEGATIVE_FITNESS``).
``FitnessPhenotype.validate_fitness`` clamps every non-positive value to 0.0, which would
erase exactly the strong-knockdown end of the dose axis this paper exists to measure.

THE HOST IS BW25113 AND THE RELEASE NAMES MG1655 b-NUMBERS. Every library strain was
built in wild-type BW25113 (``HOST_STRAIN``), while Table S3's ``locus_tag`` column is a
b-number. Each b-number is carried to its BW25113 locus through the one-to-one ECK
synonym join (``bacteria_common.eck_crosswalk``) and typed on the perturbation as a
``DerivedIdentifierMapping`` with ``route="eck_crosswalk"``, the Mutalik 2020 pattern. The
crosswalk is CHECKED rather than trusted: the workbook's own ``10-15 relfit (eco)`` sheet
names the same rows by ``BW25113_`` tag, and all 317 crossed tags agree with the authors'
own tag (measured, 0 disagreements;
``experiments/036-dataset-fixes-before-kg-build/scripts/hawkins2020_release_inventory.py``).

RECORDS DROPPED (rule + counts in ``preprocess/build_accounting.json``):

1. ``sgrna_has_no_released_fitness`` (12,104 rows): the released mean is blank, which the
   workbook's column legend defines as "blank values indicate that sgRNAs did not meet
   minimum read count (100 at t0) requirement" and the Methods state as the analysis's
   own floor (``READ_FLOOR``). There is no measurement to store.
2. ``target_gene_has_no_bw25113_locus`` (37 rows with a fitness, 99 rows in all): the
   single gene ``sokE`` (``b4700``), which is retired in GenBank ASM584v2 and carries no
   ECK synonym, so no BW25113 locus can key it. 317 of the 318 released genes resolve.

Retention arithmetic: 36,290 released (sgRNA, gene) rows = 24,149 stored + 12,104 with no
fitness + 37 on the unresolvable gene. The stored records cover 317 genes and 2,948 parent
spacers, of which 2,378 are themselves stored as the fully complementary member of their
series.

THE AUTHORS' OWN SERIES FILTER IS KEPT AS DATA, NOT APPLIED. The released
``family_retained`` flag is step 1 of the paper's curve pipeline: a series whose fully
complementary guide looks non-functional is excluded from the per-gene curves
(``FAMILY_FILTER``), which is why Table S3 and the curve tables (Table S5, Data S1) are
different populations. 4,863 of the 24,149 stored records sit in an excluded series. They
are stored, because the filter is about whether the series' PREDICTED activity scale is
trustworthy -- the quantity this loader already refuses -- and not about whether the
fitness was measured. The flag is kept per row in ``preprocess/guide_retention.csv`` so
the curve population is reconstructible.

THE 10-TO-15-DOUBLING WINDOW IS REFUSED, WITH THE MEASUREMENT THAT REFUSES IT. Table S3
also carries ``10-15 relfit (eco)``, "as 'relative fitness (eco data)', but computed from
10-doubling time point to extra 15-doubling time point". It is not stored:

- its ``relative fitness (stddev)`` column is byte-identical to the 10-doubling sheet's in
  all 22,313 rows where both are present, so the released dispersion does not describe
  that window's own measurements;
- its 1,000 non-targeting controls, which are what relative fitness is normalized to,
  spread 2.3-fold wider than the 10-doubling controls (SD 0.1935 against 0.0825, the
  latter reproducing the paper's stated noise floor to four digits);
- the two windows correlate at Pearson r 0.598 over the 18,780 sgRNAs both measure, so
  the late window is not a re-statement of the stored one either;
- and the Methods describe one 10-doubling design for E. coli and no 15-doubling
  sampling, so nothing in the mirrored text says how many replicates the late window has.

Storing it would mean attributing one window's dispersion to another window's values with
no sourced replicate count. The sheet stays in the raw mirror and the measurement is in
the inventory script's results JSON.

NOT A DUPLICATE OF THE OTHER E. coli CRISPRi ROWS (measured). Wang 2018 and Cui 2018 are
served (``CrispriGuideFitnessWang2018Dataset``, ``CrispriKnockdownCui2018Dataset``) and
overlap this release only on fully complementary spacers: 955 shared spacers against
Wang's essentiality screen (Pearson r 0.662) and 364 against each Cui screen (r 0.640 and
0.669), out of 24,149 stored records. The workbook's own two comparison columns agree with
that reading (957 rows against Wang at r 0.663, 290 against Rousset at r 0.663, and 0 of
either on a mismatched guide). 21,771 of the stored records are singly mismatched guides
that exist in no other release, the host is BW25113 rather than MG1655, and the paper
states the design difference the correlation reflects (``PRIOR_STUDIES``). Independent, so
it is loaded whole.

SOURCED VALUES: every schema value is a module-level ``SourcedValue`` anchored to the
sha256 of ``hawkinsMismatchCRISPRiRevealsCovarying2020/paper.md`` in the literature
mirror, or a typed ``ProvenanceGap``. The uncertainty is the released standard deviation
of 4 biological replicates (``N_REPLICATES``); 1,870 stored records release a mean with no
SD, and those carry typed gaps rather than a zero.

DATA SOURCE: Table S3 (``si/si4.xlsx``, publisher ``mmc4.xlsx``) from the literature
mirror, deposited into
``$DATA_ROOT/torchcell-raw/hawkinsMismatchCRISPRiRevealsCovarying2020/data/`` with its own
``manifest.json`` carrying the publisher retrieval record the library mirror already
holds. Tables S1, S2 and S4 to S12 are released and not consumed: S1 and S2 are the
FACS-seq GFP-knockdown measurements and the trained model parameters (a reporter assay and
a model, not a genotype-to-phenotype record of an essential gene), S4 is a gene list, S5
to S8 are the derived per-gene curve medians and their clusterings, and S9 to S12 are
oligonucleotides, strain tables and the B. subtilis murAA complementation measurements.
The B. subtilis sheets of Table S3 are the same release for another organism, which this
tier has no assembly for. Raw reads: SRA PRJNA574461 (``SRA_ACCESSION``), recorded as the
upstream accession and not mirrored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict
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
    AssemblyReferenceGenome,
    BacterialCrisprInterferencePerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialStrainBackground,
    ComponentDefinition,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    DerivedIdentifierMapping,
    Environment,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    GenePerturbationType,
    Genotype,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
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
from torchcell.sequence.genome.ecoli.k12 import (
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
# Paper identity and the pinned artifacts
# --------------------------------------------------------------------------- #
DOI = "10.1016/j.cels.2020.09.009"
TITLE = (
    "Mismatch-CRISPRi Reveals the Co-varying Expression-Fitness Relationships of "
    "Essential Genes in Escherichia coli and Bacillus subtilis"
)
CITATION_KEY = "hawkinsMismatchCRISPRiRevealsCovarying2020"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: The article OCR in the literature mirror: quoted here, never parsed.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "d252bf2492526b5c097a98affff8c01774576a26bc2d69d0f904b039807f5b4b"

#: Table S3 in the literature mirror, and the publisher object it came from.
TABLE_S3_LIBRARY_REL = "si/si4.xlsx"
TABLE_S3_FILENAME = "si4.xlsx"
TABLE_S3_REL = f"data/{TABLE_S3_FILENAME}"
TABLE_S3_SHA256 = "a9aa39f576e0d240e353e104e0cf94af7a2c5d9e8a375ef8d56cf07694d7412a"
TABLE_S3_BYTES = 9683583
TABLE_S3_PII = "S2405471220303665"
TABLE_S3_PUBLISHER_FILENAME = "mmc4.xlsx"
TABLE_S3_URL = (
    "https://ars.els-cdn.com/content/image/1-s2.0-"
    f"{TABLE_S3_PII}-{TABLE_S3_PUBLISHER_FILENAME}"
)
TABLE_S3_RETRIEVED_AT = "2026-10-07T11:42:04.661352+00:00"

#: The released sheet this loader reads, and the one it refuses.
SHEET_ECOLI = "relative fitness (eco data)"
SHEET_ECOLI_CONTROLS = "relative fitness (eco controls)"
SHEET_ECOLI_LATE = "10-15 relfit (eco)"
SHEET_ECOLI_LATE_CONTROLS = "10-15 relfit (eco controls)"

#: Every column of the E. coli sheet, in order; the build refuses any other header.
ECOLI_COLUMNS: tuple[str, ...] = (
    "variant",
    "original",
    "locus_tag",
    "pam",
    "offset",
    "gene",
    "relative fitness (mean)",
    "relative fitness (stddev)",
    "family_retained",
    "relative fitness (predicted)",
    "Log2FC from Wang et al., 2018",
    "relative fitness from Rousset et al., 2018",
)
COL_VARIANT = "variant"
COL_PARENT = "original"
COL_LOCUS = "locus_tag"
COL_GENE = "gene"
COL_MEAN = "relative fitness (mean)"
COL_SD = "relative fitness (stddev)"
COL_FAMILY = "family_retained"
COL_PREDICTED = "relative fitness (predicted)"

#: A 20-nt spacer over the four unambiguous bases; the build refuses anything else.
SPACER_RE = re.compile(r"^[ACGT]{20}$")
#: A b-number of the released MG1655 identifier column.
BNUMBER_RE = re.compile(r"^b\d{4}$")

#: The host the library strains were built in, and the namespace every record is keyed in.
REFERENCE_STRAIN_NAME: EcoliK12StrainName = "BW25113"
BW25113_NAMESPACE = STRAIN_GENE_NAMESPACES[REFERENCE_STRAIN_NAME]
#: The identifier namespace the RELEASE names its targets in.
MG1655_STRAIN_NAME: EcoliK12StrainName = "MG1655"

#: The one screen in this dataset: the V2 E. coli essential-gene library over 10 doublings.
SCREEN_ID = "eco_v2_essential_10_doublings"
#: The guide library every record's construct was screened in.
LIBRARY_POOL = "V2 E. coli essential-gene mismatch library (100 sgRNAs per gene)"

#: Frozen oracles, measured on the pinned bytes. The build refuses any other count.
EXPECTED_SOURCE_ROWS = 36290
EXPECTED_PARENT_SPACERS = 3110
EXPECTED_RELEASED_GENES = 318
EXPECTED_RECORDS = 24149
EXPECTED_GENES = 317
EXPECTED_STORED_PARENTS = 2948
#: Stored records whose released value is below 0 (active depletion from the pool) and
#: whose SD cell is blank. Both are declared oracles, so a regression in either fails.
EXPECTED_NEGATIVE = 778
EXPECTED_WITHOUT_SD = 1870

#: Drop reasons this loader declares; the counts are measured, not stated.
DROP_NO_FITNESS = "sgrna_has_no_released_fitness"
DROP_NO_LOCUS = "target_gene_has_no_bw25113_locus"
EXPECTED_DROPS: dict[str, int] = {DROP_NO_FITNESS: 12104, DROP_NO_LOCUS: 37}

#: The gene the release names that ASM584v2 retires, kept here so the drop is nameable.
UNRESOLVED_BNUMBER = "b4700"

#: SRA BioProject holding the reads every released fitness was computed from.
SRA_BIOPROJECT = "PRJNA574461"


# --------------------------------------------------------------------------- #
# Sourced values: every quote is a byte-exact substring of the pinned paper.md
# --------------------------------------------------------------------------- #
_OCR_METHOD = "MinerU OCR of the publisher PDF (torchcell-library mirror)"
_PAGE_ECOLI_STRAINS = (
    "STAR Methods, 'Escherichia coli Strain Construction and Growth Conditions'"
)
_PAGE_LIBRARY_DESIGN = "STAR Methods, 'sgRNA Plasmid Library Design'"
_PAGE_FITNESS_EXPERIMENTS = "STAR Methods, 'Relative Fitness Experiments'"
_PAGE_FITNESS_ANALYSIS = "STAR Methods, 'Relative Fitness Analysis'"
_PAGE_DETECTION_LIMITS = (
    "STAR Methods, 'Detection Limits of Relative Fitness Measurements'"
)
_PAGE_CURVE_ANALYSIS = (
    "STAR Methods, 'Expression-fitness Relationship Analysis Details'"
)
_PAGE_MICROBES = "STAR Methods, 'Microbes'"
_PAGE_RESULTS_FITNESS = (
    "Results, 'Measuring the Fitness of Libraries of Mismatched sgRNAs in E. coli and "
    "B. subtilis'"
)
_PAGE_RESULTS_MODEL = (
    "Results, 'A Species-Independent Linear Model Robustly Predicts Mismatched sgRNA "
    "Activity'"
)
_PAGE_FIGURE3 = "Figure 3 legend"
_PAGE_KEY_RESOURCES = "Key Resources Table"
_PAGE_DATA_AVAILABILITY = "Resource Availability, 'Data and Code Availability'"


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote of the pinned Hawkins 2020 OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method=_OCR_METHOD,
            page=page,
        ),
    )


HOST_STRAIN = _paper(
    REFERENCE_STRAIN_NAME,
    "All CRISPRi library strains were constructed in the wildtype BW25113 background by "
    "electroporating an sgRNA plasmid or plasmid pool (see ‘‘sgRNA plasmid "
    "construction’’) into a recipient strain encoding dcas9 (for essential gene "
    "knockdown libraries), or dcas9 and gfp or rfp (for GFP knockdown libraries), "
    "selecting for ampicillin resistance.",
    page=_PAGE_ECOLI_STRAINS,
    note="the host is BW25113, so every record is written against the BW25113 assembly "
    "although the release names its targets by MG1655 b-number",
)
HOST_CASSETTE = _paper(
    "CAG78830",
    "Tn7 transposition was used to integrate a dcas9 expression cassette into the Tn7att "
    "site using triparental mating of DAP(diaminopimelic acid)-dependent donors and "
    "selecting for gentamicin resistance in the absence of DAP, as previously described "
    "(Peters et al., 2019). The dcas9 expression cassette is modified from previously "
    "described versions (Peters et al., 2019), contains dcas9 from S. pyogenes (Qi et "
    "al., 2013) with a 3X Myc C-terminal tag, and is expressed from the IPTG-inducible "
    "promoter PlLac-O1 (Lutz and Bujard, 1997) and regulated by lacIq.",
    page=_PAGE_ECOLI_STRAINS,
    note="the recipient strain of every library record: BW25113 with a chromosomal "
    "IPTG-inducible dcas9 at the Tn7att site. The effector is S. pyogenes dCas9",
)
HOST_KEY_RESOURCE = _paper(
    "CAG78830",
    "Escherichia coli BW25113 Tn7att:PILac- 01-dcas9(Gent)</td><td>This study</td>"
    "<td>CAG78830</td></tr>",
    page=_PAGE_KEY_RESOURCES,
    note="the Key Resources row for the recipient strain; the OCR renders PlLac-O1 as "
    "'PILac- 01'. The quote is the HTML row the OCR holds, so it is auditable against "
    "the pinned bytes",
)
LIBRARY_DESIGN = _paper(
    (10, 10, 100),
    "For every gene in this set, ten non-overlapping fully complementary spacers were "
    "chosen on the non-template strand, as close to the start of the ORF as possible. For "
    "each fully complementary spacer, a set of 10 spacer variants was designed and "
    "ordered (for a total of 100 sgRNAs per gene): 1x the original fully complementary "
    "spacer, 9x single-mismatches (Figure S1).",
    page=_PAGE_LIBRARY_DESIGN,
    note="the V2 E. coli library: 10 parent spacers per gene, each with itself plus 9 "
    "singly mismatched variants. Measured on the pinned bytes: 2,956 of the 3,110 parent "
    "spacers carry exactly 10 rows, and every released E. coli variant differs from its "
    "parent in 0 or 1 position (3,110 at 0, 33,180 at 1)",
)
LIBRARY_DESIGN_GRADED = _paper(
    "graded knockdown",
    "Using our model of mismatched sgRNA activity, we designed a set of sgRNAs targeted "
    "to the essential gene complement of E. coli and B. subtilis ${ \\sim } 3 0 0$ genes "
    "in each species; Table S3) and predicted to have a range of activities. We generated "
    "large pooled libraries of strains in which each essential gene is targeted by 100 "
    "sgRNAs (10 fully matched guides, each with 9 singly mismatched variants; STAR "
    "Methods; Figure S1C).",
    page=_PAGE_RESULTS_FITNESS,
    note="what the mismatch design is for: a dose axis on an essential gene, which a "
    "deletion collection cannot carry",
)
ECOLI_CONDITIONS = _paper(
    (True, 1.0),
    "Fitness experiments for the E. coli V2 libraries were carried out in an identical "
    "manner to the B. subtilis fitness experiments with the following exceptions: all "
    "growth occurred in the presence of ampicillin, and induction was achieved with 1mM "
    "IPTG instead of $1 \\%$ xylose.",
    page=_PAGE_FITNESS_EXPERIMENTS,
    note="the E. coli culture is stated as the B. subtilis one with two changes: "
    "ampicillin throughout and 1 mM IPTG as the inducer. Everything else about the "
    "culture, the 10 doublings included, is sourced from the B. subtilis paragraph this "
    "sentence defers to",
)
AMPICILLIN = _paper(
    100.0,
    "Unless otherwise noted, all strain construction and growth assays for E. coli were "
    "done in LB medium and using antibiotic selection at the specified concentrations: "
    "ampicillin $( 1 0 0 \\mu \\ g / \\ m \\ l )$ , carbenicillin "
    "$( 5 0 \\mu \\ g / \\mathrm { m l } )$ , gentamicin "
    "$( 1 0 \\mu \\ g / \\mathrm { m l } )$ , chloramphenicol "
    "$( 2 5 \\mu \\ g / \\mathrm { m l } )$ , kanamycin "
    "$( 3 0 \\mathsf { u g } / \\mathsf { m l } )$ .",
    page=_PAGE_ECOLI_STRAINS,
    note="100 ug/mL ampicillin, the selection that maintains the sgRNA plasmid pJSHA77; "
    "it is a component of the medium rather than an environment perturbation, because "
    "every record and the reference carry it",
)
MEDIUM_AND_TEMPERATURE = _paper(
    ("LB", 37.0),
    "Escherichia coli strains were cultured in LB medium at 37C.",
    page=_PAGE_MICROBES,
    note="LB at 37 C. No formulation is printed anywhere in the paper, so the three LB "
    "ingredients are carried with NO amounts: the Miller and the Lennox recipes differ "
    "in exactly the NaCl this paper never states",
)
DOUBLINGS = _paper(
    10.0,
    "This culture was then grown to OD600 0.3 ${ \\sim } 5$ doublings), back-diluted to "
    "OD600 0.01 in $L B + 1 \\%$ xylose, and grown to OD600 0.3 (total ${ \\sim } 1 0$ "
    "doublings).",
    page=_PAGE_FITNESS_EXPERIMENTS,
    note="about 10 doublings of exponential growth, maintained by one back dilution. The "
    "quote is the B. subtilis paragraph, which ECOLI_CONDITIONS defers to for everything "
    "but the ampicillin and the inducer; the xylose it names is what 1 mM IPTG replaces "
    "in E. coli",
)
NORMALIZATION = _paper(
    1000,
    "by comparing its relative abundance (quantified by next-generation sequencing of the "
    "sgRNA spacers) to the relative abundance of 1,000 non-targeting sgRNAs at the start "
    "and end of each experiment",
    page=_PAGE_RESULTS_FITNESS,
    note="what the released number is relative to: the median of 1,000 non-targeting "
    "control sgRNAs in the same sample. Those controls are the phenotype_reference and "
    "are never a record, because a non-targeting guide perturbs no gene",
)
FITNESS_DEFINITION = _paper(
    1.0,
    "wild-type doublings over the time course of the experiment. Strains with a relative "
    "fitness of 1 grow as well as the wild-type does; lower values imply slower growth.",
    page=_PAGE_RESULTS_FITNESS,
    note="the released value is the strain's doublings divided by the wild type's over "
    "one time course, which is measurement_type relative_growth_rate: a dimensionless "
    "ratio of growth against a control measured in the same run, 1.0 = wild-type growth",
)
READ_FLOOR = _paper(
    100,
    "For each strain $( x )$ with at least 100 counts at $t _ { O }$ we calculate the "
    "relative fitness",
    page=_PAGE_FITNESS_ANALYSIS,
    note="the authors' own floor, already applied to the release: the workbook's column "
    "legend states a blank mean or SD as 'sgRNAs did not meet minimum read count (100 at "
    "t0) requirement'. This loader adds no filter of its own",
)
N_REPLICATES = _paper(
    4,
    "Finally, the relative fitness measurements of each sgRNA were averaged across "
    "samples (B. subtilis experiments: 6 replicates, E. coli experiments: 4 replicates) "
    "to calculate the final relative fitness value and standard deviation (Table S3).",
    page=_PAGE_FITNESS_ANALYSIS,
    note="4 biological replicates for E. coli, and the released SD is the standard "
    "deviation of those individual replicate values, which is why the uncertainty type is "
    "sample_sd with n_samples = 4",
)
NEGATIVE_FITNESS = _paper(
    0.0,
    "Many strains were abundant enough at the start of the experiment to allow accurate "
    "quantification of decreases greater than $2 ^ { 1 0 } \\sim$ 1,000-fold. These events "
    "(relative fitness $< 0$ ) represent active depletion from the pool.",
    page=_PAGE_DETECTION_LIMITS,
    note="a negative value is a meaning the release states, not a defect: 778 of the "
    "24,149 stored values are negative, which is why FitnessPhenotype (whose validator "
    "clamps non-positives to 0.0) cannot hold this readout",
)
NOISE_FLOOR = _paper(
    0.0825,
    "The standard deviation of relative fitness of our 1,000 non-targeting control sgRNAs "
    "was 0.0825 in E. coli",
    page=_PAGE_FIGURE3,
    note="the screen's own noise floor. Measured on the pinned bytes: the SD of the 897 "
    "released control means is 0.08254, reproducing the stated value to four digits. It "
    "is recorded here and is NOT stored as the reference's uncertainty: its samples are "
    "1,000 control STRAINS, which no SampleUnit member names, and reporting it as a "
    "replicate SD would mislabel it",
)
PREDICTED_ACTIVITY = _paper(
    "predicted sgRNA activity",
    "We next predicted the sgRNA activity of all sgRNAs using the model of sgRNA efficacy "
    "described above trained on the two species averaged GFP data also described above. "
    "Consistent with the definition of sgRNA activity above, fully complementary sgRNAs "
    "were assigned an sgRNA activity of 1.",
    page=_PAGE_CURVE_ANALYSIS,
    note="Table S3's 'relative fitness (predicted)' column is THIS quantity, a model "
    "output, not a predicted fitness. Measured on the pinned bytes: exactly 1.0 for all "
    "3,110 parent spacers, as this sentence defines. It is not stored; the schema has no "
    "field for a perturbation covariate that is a prediction with its own error",
)
MODEL_FIT = _paper(
    (0.56, 0.10, 0.08),
    "Despite the simplicity of this model, the effects of single mismatches were robustly "
    "predicted (Figures 1C and S4, species-averaged $\\mathsf { R } ^ { 2 } = 0 . 5 6$ , "
    "11-fold cross-validation mean squared error $= 0 . 1 0 \\pm 0 . 0 8 )$ )",
    page=_PAGE_RESULTS_MODEL,
    note="the error the refused dose would carry: R^2 0.56 with an 11-fold "
    "cross-validation mean squared error of 0.10 +/- 0.08",
)
PRELIMINARY_MODEL = _paper(
    "preliminary model",
    "For the design of all libraries using this strategy, a preliminary version of the "
    "linear model was used.",
    page=_PAGE_LIBRARY_DESIGN,
    note="the dose is entangled with the design: the library's mismatches were chosen by "
    "an earlier version of the model whose later version predicts their activity",
)
FAMILY_FILTER = _paper(
    "family_retained",
    "we surmised that the fully matched sgRNA was likely not functional and excluded its "
    "series from further analysis.",
    page=_PAGE_CURVE_ANALYSIS,
    note="the released family_retained flag is this exclusion, which is step 1 of the "
    "paper's CURVE pipeline. 4,863 of the 24,149 stored records sit in an excluded "
    "series; they are stored because the exclusion is about the series' predicted "
    "activity scale, not about whether the fitness was measured, and the flag is kept "
    "per row in preprocess/guide_retention.csv",
)
PRIOR_STUDIES = _paper(
    ("Rousset et al., 2018", "Wang et al., 2018"),
    "Our relative fitness values for fully complementary guides were correlated with "
    "previously reported measurements (Rousset et al., 2018; Wang et al., 2018) but had "
    "greatly expanded dynamic range (Figures S7D and S7E) due to differences in "
    "experimental design.",
    page=_PAGE_RESULTS_FITNESS,
    note="the paper's own reading of the overlap this loader measures: 955 fully "
    "complementary spacers shared with Wang 2018's essentiality screen at Pearson r "
    "0.662 and 364 with each Cui 2018 screen at r 0.640 and 0.669, out of 24,149 stored "
    "records, and 0 shared mismatched guides",
)
SRA_ACCESSION = _paper(
    SRA_BIOPROJECT,
    "All raw sequencing data are deposited in the Short Read Archive under accession "
    "PRJNA574461.",
    page=_PAGE_DATA_AVAILABILITY,
    note="the reads every released fitness was computed from; recorded as the upstream "
    "accession, not mirrored",
)

#: Every module-level sourced value, so the sourcing can be audited without the code.
SOURCED_VALUES: dict[str, SourcedValue] = {
    "host_strain": HOST_STRAIN,
    "host_cassette": HOST_CASSETTE,
    "host_key_resource": HOST_KEY_RESOURCE,
    "library_design": LIBRARY_DESIGN,
    "library_design_graded": LIBRARY_DESIGN_GRADED,
    "ecoli_conditions": ECOLI_CONDITIONS,
    "ampicillin_ug_per_ml": AMPICILLIN,
    "medium_and_temperature": MEDIUM_AND_TEMPERATURE,
    "duration_generations": DOUBLINGS,
    "normalization": NORMALIZATION,
    "fitness_definition": FITNESS_DEFINITION,
    "read_floor": READ_FLOOR,
    "n_samples": N_REPLICATES,
    "negative_fitness": NEGATIVE_FITNESS,
    "noise_floor": NOISE_FLOOR,
    "predicted_activity": PREDICTED_ACTIVITY,
    "model_fit": MODEL_FIT,
    "preliminary_model": PRELIMINARY_MODEL,
    "family_filter": FAMILY_FILTER,
    "prior_studies": PRIOR_STUDIES,
    "sra_accession": SRA_ACCESSION,
}


# --------------------------------------------------------------------------- #
# Medium, host background and environment
# --------------------------------------------------------------------------- #
def _lb_ingredients() -> list[MediaComponent]:
    """LB's three ingredients with NO amounts.

    The paper states "LB medium" and prints no formulation anywhere, and the Miller and
    Lennox recipes differ in exactly the NaCl it never states, so the identities are LB's
    definition and every amount is ``None``, which is what an unsourced amount means.
    """
    return [
        MediaComponent(
            compound=resolved_compound("tryptone"),
            role=MediaComponentRole.complex_ingredient,
            concentration=None,
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[MEDIUM_AND_TEMPERATURE],
            note="LB ingredient; no amount is stated by the paper",
        ),
        MediaComponent(
            compound=resolved_compound("yeast extract"),
            role=MediaComponentRole.complex_ingredient,
            concentration=None,
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[MEDIUM_AND_TEMPERATURE],
            note="LB ingredient; no amount is stated by the paper",
        ),
        MediaComponent(
            compound=resolved_compound("sodium chloride"),
            role=MediaComponentRole.bulk_salt,
            concentration=None,
            provenance=[MEDIUM_AND_TEMPERATURE],
            note="LB ingredient; no amount is stated by the paper, and the Miller and "
            "Lennox formulations differ in exactly this component",
        ),
    ]


HAWKINS2020_LB_AMPICILLIN = Media(
    name="LB with 100 ug/mL ampicillin, formulation not stated (Hawkins 2020 E. coli "
    "fitness experiments), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="LB",
    components=[
        *_lb_ingredients(),
        MediaComponent(
            compound=resolved_compound("ampicillin"),
            role=MediaComponentRole.selection_agent,
            concentration=Concentration(
                value=float(AMPICILLIN.value), unit=ConcentrationUnit.ug_per_ml
            ),
            provenance=[AMPICILLIN, ECOLI_CONDITIONS],
            note="maintains the sgRNA plasmid pJSHA77 throughout the competition; the "
            "paper states ampicillin for all E. coli growth and 100 ug/mL as its "
            "concentration",
        ),
    ],
    provenance=[MEDIUM_AND_TEMPERATURE, AMPICILLIN, ECOLI_CONDITIONS],
)
"""The E. coli fitness experiments' medium: LB with its selection, amounts unstated."""


def host_background() -> BacterialStrainBackground:
    """BW25113 with the chromosomal IPTG-inducible dcas9 cassette at Tn7att.

    ``alleles`` is empty and that is a statement: the cassette is a Tn7 transposition at
    the Tn7att site and the paper names no disrupted BW25113 locus, so there is no allele
    of a tagged gene to type. The sgRNA plasmid is not a background either -- it is the
    perturbation, and its spacer rides on the record's ``CrisprConstruct``.
    """
    return BacterialStrainBackground(
        name=str(HOST_CASSETTE.value),
        reference_strain=REFERENCE_STRAIN_NAME,
        assembly_set="ecoli_K12_BW25113_ASM75055v1",
        parents=[REFERENCE_STRAIN_NAME],
        construction="wild-type BW25113 with a dcas9 expression cassette (S. pyogenes "
        "dcas9, 3X Myc C-terminal tag, IPTG-inducible PlLac-O1, lacIq-regulated) "
        "integrated at the Tn7att site by Tn7 transposition",
        genotype_statement="Escherichia coli BW25113 Tn7att::PlLac-O1-dcas9(Gent)",
        alleles=[],
        provenance=[HOST_STRAIN, HOST_CASSETTE, HOST_KEY_RESOURCE],
    )


def host_reference(data_root: str | None = None) -> AssemblyReferenceGenome:
    """BW25113 pinned to its GenBank assembly, carrying the dcas9-cassette background."""
    return assembly_reference(
        REFERENCE_STRAIN_NAME, background=host_background(), data_root=data_root
    )


def _inducer() -> SmallMoleculePerturbation:
    """The 1 mM IPTG that switches the chromosomal dcas9 on."""
    return SmallMoleculePerturbation(
        compound=resolved_compound("IPTG"),
        concentration=Concentration(
            value=float(ECOLI_CONDITIONS.value[1]), unit=ConcentrationUnit.millimolar
        ),
    )


def screen_environment() -> Environment:
    """The one pooled competition every record and the reference share.

    LB with ampicillin at 37 C, 1 mM IPTG inducing dCas9, over about 10 doublings. The
    inducer is an ``Environment.perturbation`` and the ampicillin a medium component: the
    paper writes the ampicillin as a condition of all E. coli growth and the IPTG as the
    induction that realizes the knockdown.
    """
    return Environment(
        media=HAWKINS2020_LB_AMPICILLIN,
        temperature=Temperature(value=float(MEDIUM_AND_TEMPERATURE.value[1])),
        perturbations=[_inducer()],
        aerobicity="aerobic",
        duration_generations=float(DOUBLINGS.value),
    )


_UNITS = (
    "relative fitness: the number of doublings of one sgRNA's clone divided by the "
    "number of wild-type doublings over about 10 doublings of pooled competition, where "
    "wild type is the median of 1,000 non-targeting control sgRNAs in the same sample; "
    "1.0 = grows like the wild type, below 1.0 = slower, below 0 = actively depleted "
    "from the pool. Mean of 4 biological replicates"
)
_UNITS_REFERENCE = (
    "relative fitness 1.0, which is what the normalization to the median of 1,000 "
    "non-targeting control sgRNAs makes an unaffected clone by construction; the "
    "released control sgRNAs are not records, because a non-targeting guide perturbs no "
    "gene"
)

#: The uncertainty fields a record with no released SD cannot carry.
_UNCERTAINTY_FIELDS: tuple[str, ...] = (
    "environment_response_se",
    "environment_response_uncertainty",
    "environment_response_uncertainty_type",
)


def _missing_sd_gaps() -> list[ProvenanceGap]:
    """The typed absence of a dispersion for a released mean with a blank SD."""
    return [
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_reported_by_primary,
            note="the release carries this sgRNA's replicate mean and a blank "
            "'relative fitness (stddev)' cell; the workbook's legend ties a blank to the "
            "100-counts-at-t0 floor, so no dispersion is stored rather than a zero",
        )
        for field in _UNCERTAINTY_FIELDS
    ]


def _reference_gaps() -> list[ProvenanceGap]:
    """What the 1.0 baseline is not: a replicate set of its own."""
    return [
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_reported_by_primary,
            note="the 1.0 baseline is what the normalization to the median of 1,000 "
            "non-targeting control sgRNAs puts an unaffected clone at, not a measured "
            "replicate series. The screen's noise floor IS released (SD 0.0825 over "
            "the 1,000 controls, reproduced as 0.08254 on the pinned bytes) but its "
            "samples are control strains, which no SampleUnit member names",
        )
        for field in ("n_samples", "sample_unit", *_UNCERTAINTY_FIELDS)
    ]


def guide_phenotype(fitness: float, sd: float | None) -> EnvironmentResponsePhenotype:
    """One sgRNA's released relative fitness, with its replicate SD when released."""
    if sd is None:
        return EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.relative_growth_rate,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=fitness,
            n_samples=int(N_REPLICATES.value),
            sample_unit=SampleUnit.biological_replicate,
            units=_UNITS,
            screen_id=SCREEN_ID,
            provenance_gaps=_missing_sd_gaps(),
        )
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.relative_growth_rate,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=fitness,
        environment_response_uncertainty=sd,
        environment_response_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=int(N_REPLICATES.value),
        sample_unit=SampleUnit.biological_replicate,
        units=_UNITS,
        screen_id=SCREEN_ID,
        provenance_gaps=[],
    )


def reference_phenotype() -> EnvironmentResponsePhenotype:
    """The non-targeting baseline: relative fitness 1.0 by the normalization."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.relative_growth_rate,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=float(FITNESS_DEFINITION.value),
        units=_UNITS_REFERENCE,
        screen_id=SCREEN_ID,
        provenance_gaps=_reference_gaps(),
    )


def publication() -> Publication:
    """This paper, by DOI.

    No PubMed id is recorded: the literature mirror's manifest for this key carries the
    DOI and the title, and the paper's own pinned OCR prints no PMID, so asserting one
    would be an unsourced identifier.
    """
    return Publication(
        pubmed_id=None, pubmed_url=None, doi=DOI, doi_url=f"https://doi.org/{DOI}"
    )


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/hawkinsMismatchCRISPRiRevealsCovarying2020``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """The literature mirror of this citation key."""
    return Path(data_root or _data_root()) / "torchcell-library" / CITATION_KEY


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_retrieval(retrieved_at: str = TABLE_S3_RETRIEVED_AT) -> RetrievalRecord:
    """The recorded retrieval of Table S3: one Elsevier supplementary object.

    The literature mirror already holds this file under the digest the publisher served,
    with the retriever and its params recorded, so the raw mirror is written from those
    bytes rather than re-fetched. ``run_retriever`` on this record re-fetches them.
    """
    return RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=TABLE_S3_URL,
        retriever="torchcell.literature.retrieve.elsevier_mmc",
        params={"pii": TABLE_S3_PII, "filename": TABLE_S3_PUBLISHER_FILENAME},
        sha256=TABLE_S3_SHA256,
        retrieved_at=retrieved_at,
    )


#: The released files this loader does NOT consume, and why, for ``si_expected``.
NOT_CONSUMED: tuple[str, ...] = (
    "Table S1 (mmc2.xlsx) -- the FACS-seq GFP-knockdown measurements the mismatch model "
    "was trained on. NOT deposited: its unit of observation is a gfp-targeting sgRNA in "
    "a reporter strain, which is a reporter assay rather than a genotype-to-phenotype "
    "record of an essential gene",
    "Table S2 (mmc3.xlsx) -- the trained linear model's parameters and their standard "
    "errors. NOT deposited: a model, not a measurement",
    "Table S4 (mmc5.xlsx) -- the genes whose knockdown repeatedly gave negative relative "
    "fitness, with the B. subtilis lysis calls. NOT deposited: a gene list derived from "
    "Table S3, which this loader reads directly",
    "Tables S5 to S8 (mmc6, mmc7, mmc8, mmc9) -- the per-gene expression-fitness curve "
    "medians over 17 sliding activity bins, their functional-enrichment clusterings and "
    "the cross-species comparison. NOT deposited: every one of them is computed FROM the "
    "refused predicted activity, so each carries the model's error as well as the "
    "measurement's, and a curve bin is not a strain",
    "Table S9 (mmc10.xlsx) -- oligonucleotides and primers. NOT deposited: no "
    "measurement",
    "Tables S10 to S12 (mmc11, mmc12, mmc13) -- the strain tables, the B. subtilis murAA "
    "complementation knockdown measurements and the ampG / mpl per-gene fitness "
    "differences. NOT deposited: B. subtilis (no assembly in this tier) or, for S13's "
    "ampG and mpl sheets, per-GENE median differences rather than per-strain values",
    "Data S1 and S2 (mmc14.pdf, mmc15.pdf) -- the per-gene curve plots, and mmc1.pdf / "
    "mmc16.pdf the Supplementary figures and the accepted manuscript. Mirrored in the "
    "literature mirror as documents; not data this loader reads",
    "The B. subtilis sheets of Table S3 itself ('relative fitness (bsu data)', its "
    "controls, 'relative fitness (individual)' and 'v1 trimethoprim fitness scores') -- "
    "deposited with the workbook and not loaded: the organism has no assembly in the "
    "genomes tier, and the individual-isolate and trimethoprim sheets are B. subtilis "
    "dfrA measurements",
    "The '10-15 relfit (eco)' and '10-15 relfit (eco controls)' sheets -- deposited with "
    "the workbook and REFUSED, with the measurement that refuses them in the module "
    "docstring: the late window's released SD column is byte-identical to the "
    "10-doubling sheet's in all 22,313 shared rows, its controls spread 2.3-fold wider "
    "(SD 0.1935 against 0.0825) and the Methods describe no 15-doubling sampling for "
    "E. coli, so no replicate count can be sourced for it",
    f"SRA BioProject {SRA_BIOPROJECT} -- the raw reads behind every released value. NOT "
    "mirrored: this loader consumes the computed table, not the reads",
)


def deposit_raw_mirror(
    *, retrieved_at: str = TABLE_S3_RETRIEVED_AT, data_root: str | None = None
) -> Path:
    """Write the raw mirror (Table S3) from the literature mirror's bytes, plus a manifest.

    Idempotent by sha256: an existing mirror file with the pinned digest is left alone and
    a differing one raises rather than being overwritten. The bytes are verified BEFORE
    anything is written, so a refusal leaves no partial deposit.
    """
    src = library_dir(data_root) / TABLE_S3_LIBRARY_REL
    if not src.exists():
        raise RuntimeError(f"library mirror is missing {src}")
    got = _sha256(src)
    if got != TABLE_S3_SHA256:
        raise RuntimeError(f"{src} sha256 {got} is not the pinned {TABLE_S3_SHA256}")
    root = raw_mirror_dir(data_root)
    dest = root / TABLE_S3_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != TABLE_S3_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(src, dest)
    size = dest.stat().st_size
    if size != TABLE_S3_BYTES:
        raise RuntimeError(f"{dest} is {size} bytes, not the pinned {TABLE_S3_BYTES}")
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=[
            ArtifactRecord(
                path=TABLE_S3_REL,
                role=ROLE_RAW_DATA,
                bytes=size,
                sha256=TABLE_S3_SHA256,
                source=TABLE_S3_URL,
                original_filename=TABLE_S3_PUBLISHER_FILENAME,
                retrieval=table_retrieval(retrieved_at),
            )
        ],
        si_data_sources=[TABLE_S3_URL, f"https://doi.org/{DOI}"],
        si_expected=list(NOT_CONSUMED),
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


def _link_table(raw_dir: str) -> None:
    """Link Table S3 into ``raw/`` after checking it against the raw-mirror manifest."""
    data_root = _data_root()
    manifest = load_manifest(data_root)
    check_manifest_pin(
        TABLE_S3_REL, manifest_sha256(manifest, TABLE_S3_REL), TABLE_S3_SHA256
    )
    src = raw_mirror_dir(data_root) / TABLE_S3_REL
    if not src.exists():
        raise RuntimeError(f"required raw artifact missing from mirror: {src}")
    os.makedirs(raw_dir, exist_ok=True)
    link_verified(src, osp.join(raw_dir, TABLE_S3_FILENAME), TABLE_S3_SHA256)


# --------------------------------------------------------------------------- #
# The released table
# --------------------------------------------------------------------------- #
def read_ecoli_sheet(path: str | Path) -> pd.DataFrame:
    """Table S3's E. coli fitness sheet, with its header and shape checked."""
    frame = pd.read_excel(path, sheet_name=SHEET_ECOLI, engine="openpyxl")
    if tuple(frame.columns) != ECOLI_COLUMNS:
        raise RuntimeError(
            f"{SHEET_ECOLI} header is {tuple(frame.columns)!r}, not {ECOLI_COLUMNS!r}"
        )
    if len(frame) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(
            f"{SHEET_ECOLI} has {len(frame)} rows, not the pinned {EXPECTED_SOURCE_ROWS}"
        )
    return frame


class MismatchDesign(BaseModel):
    """One released sgRNA: its spacer, its parent spacer, and how the two differ.

    ``mismatch_index`` is 0-based from the spacer's 5' end, which is the position encoding
    the paper's own model uses ("the position of the mismatch (from 0 to 19, with 19 being
    PAM proximal)"). ``None`` on both mismatch fields is a fully complementary spacer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    spacer: str
    parent_spacer: str
    mismatch_index: int | None
    substitution: str | None

    @property
    def description(self) -> str:
        """The construct's mismatch design, which is what the perturbation records."""
        base = (
            "CRISPR interference (decreased expression) of a bacterial gene by locus "
            "tag; mismatch-CRISPRi guide"
        )
        if self.mismatch_index is None:
            return (
                f"{base}, fully complementary spacer {self.spacer} (the parent of its "
                "series)"
            )
        return (
            f"{base}, spacer {self.spacer}: a single mismatch to parent spacer "
            f"{self.parent_spacer} at 0-based position {self.mismatch_index} from the "
            f"5' end ({self.substitution}), which titrates knockdown below the parent's"
        )


def mismatch_design(spacer: str, parent: str) -> MismatchDesign:
    """The design of one released spacer against its parent, refusing anything else.

    The release's E. coli sheet holds fully complementary spacers and SINGLY mismatched
    ones only (measured: 3,110 at Hamming distance 0 and 33,180 at 1), which this
    function asserts rather than assumes: a double mismatch would need a description this
    class does not write.
    """
    if SPACER_RE.match(spacer) is None:
        raise RuntimeError(f"spacer {spacer!r} is not a 20-nt ACGT sequence")
    if SPACER_RE.match(parent) is None:
        raise RuntimeError(f"parent spacer {parent!r} is not a 20-nt ACGT sequence")
    positions = [
        index
        for index, (variant, original) in enumerate(zip(spacer, parent, strict=True))
        if variant != original
    ]
    if not positions:
        return MismatchDesign(
            spacer=spacer, parent_spacer=parent, mismatch_index=None, substitution=None
        )
    if len(positions) > 1:
        raise RuntimeError(
            f"spacer {spacer!r} differs from parent {parent!r} in {len(positions)} "
            "positions; the E. coli release carries 0 or 1"
        )
    (index,) = positions
    return MismatchDesign(
        spacer=spacer,
        parent_spacer=parent,
        mismatch_index=index,
        substitution=f"{parent[index]} to {spacer[index]}",
    )


# --------------------------------------------------------------------------- #
# Identifier resolution: released MG1655 b-number -> BW25113 locus tag
# --------------------------------------------------------------------------- #
class TargetResolution(BaseModel):
    """How one released b-number reached (or failed to reach) a BW25113 locus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    b_number: str
    mg1655_locus: str | None
    eck: str | None
    bw25113_locus: str | None
    symbol: str | None
    numerics_agree: bool | None


class TargetResolutionReport(BaseModel):
    """Every released gene's verdict, plus the MG1655 reconciliation it went through."""

    resolved: dict[str, TargetResolution]
    unresolved: dict[str, str]
    reconciliation: LocusTagReconciliation

    @property
    def stored(self) -> dict[str, TargetResolution]:
        """Only the genes that reached a BW25113 locus."""
        return {
            name: target
            for name, target in self.resolved.items()
            if target.bw25113_locus is not None
        }


def canonical_symbol(genome: EcoliK12BW25113Genome, locus: str) -> str | None:
    """The BW25113 annotation's own gene symbol for a locus, or ``None``."""
    exact, _ = genome.feature_index["symbol"]
    for symbol, loci in exact.items():
        if locus in {str(value) for value in loci}:
            return str(symbol)
    return None


def resolve_targets(
    b_numbers: Sequence[str],
    mg1655: EcoliK12MG1655Genome,
    bw25113: EcoliK12BW25113Genome,
) -> TargetResolutionReport:
    """Carry every released b-number to its BW25113 locus through the one-to-one ECK join.

    The release names MG1655 b-numbers for a BW25113 host, so each name is first
    reconciled against MG1655 and then crossed on its ECK synonym, the only published 1:1
    relation between the two namespaces (``bacteria_common.eck_crosswalk``). A name that
    MG1655 retires, or whose MG1655 locus has no ECK partner, reaches no BW25113 locus and
    is reported rather than guessed at.
    """
    for name in b_numbers:
        if BNUMBER_RE.match(name) is None:
            raise RuntimeError(f"released locus_tag {name!r} is not a b-number")
    names = pd.Series(sorted(set(b_numbers)), dtype=str)
    stored, report = reconcile_locus_tags(mg1655, names, label="hawkins2020 b-numbers")
    pair_of = {pair.mg1655: pair for pair in eck_crosswalk(mg1655, bw25113).pairs}
    resolved: dict[str, TargetResolution] = {}
    unresolved: dict[str, str] = {}
    for name, mg_locus in zip(names, stored, strict=True):
        if name in set(report.retired_kept):
            unresolved[str(name)] = (
                "retired in the pinned MG1655 assembly, so no locus and no ECK synonym"
            )
            continue
        pair = pair_of.get(str(mg_locus))
        if pair is None:
            unresolved[str(name)] = (
                f"MG1655 locus {mg_locus} carries no one-to-one ECK partner in BW25113"
            )
            continue
        resolved[str(name)] = TargetResolution(
            b_number=str(name),
            mg1655_locus=str(mg_locus),
            eck=pair.eck,
            bw25113_locus=pair.bw25113,
            symbol=canonical_symbol(bw25113, pair.bw25113),
            numerics_agree=pair.numerics_agree,
        )
    return TargetResolutionReport(
        resolved=resolved, unresolved=unresolved, reconciliation=report
    )


def build_genotype(
    design: MismatchDesign, target: TargetResolution, released_symbol: str
) -> Genotype:
    """One graded dCas9 knockdown of one essential gene.

    ``perturbed_gene_name`` is the BW25113 annotation's own symbol when it has one, so a
    gene carries one spelling across datasets, and the release's own symbol stays in
    ``preprocess/guide_retention.csv``. ``description`` carries the guide's mismatch
    design, which is the part of the graded dose that IS measured: what we built, rather
    than what the model predicts it does.
    """
    locus = target.bw25113_locus
    if locus is None:
        raise RuntimeError(
            f"{target.b_number} reached no BW25113 locus; it cannot key a perturbation"
        )
    leaf = BacterialCrisprInterferencePerturbation(
        systematic_gene_name=locus,
        perturbed_gene_name=target.symbol or released_symbol,
        gene_namespace=BW25113_NAMESPACE,
        identifier_mapping=DerivedIdentifierMapping(
            source_identifier=target.b_number, route="eck_crosswalk"
        ),
        description=design.description,
        crispr=CrisprConstruct(
            effector="dCas9",
            guide_sequence=design.spacer,
            n_guides=1,
            library_pool=LIBRARY_POOL,
        ),
    )
    perturbations: list[GenePerturbationType] = [leaf]
    return Genotype(perturbations=perturbations)


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
class RowVerdict(BaseModel):
    """One released row: the record it becomes, or the rule that drops it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spacer: str
    parent_spacer: str
    b_number: str
    released_symbol: str
    fitness: float | None
    sd: float | None
    predicted_activity: float
    family_retained: bool
    bw25113_locus: str | None
    drop_reason: str


def classify_row(
    row: Mapping[str, Any], targets: Mapping[str, TargetResolution]
) -> RowVerdict:
    """The verdict for one released row, by this loader's two declared drop rules.

    A row is stored when the release carries a fitness for it AND its target gene reached
    a BW25113 locus. The fitness rule comes first because it is the release's own floor:
    a blank mean is a measurement that was never made.
    """
    fitness = row[COL_MEAN]
    sd = row[COL_SD]
    b_number = str(row[COL_LOCUS])
    target = targets.get(b_number)
    reason = ""
    if pd.isna(fitness):
        reason = DROP_NO_FITNESS
    elif target is None:
        reason = DROP_NO_LOCUS
    return RowVerdict(
        spacer=str(row[COL_VARIANT]),
        parent_spacer=str(row[COL_PARENT]),
        b_number=b_number,
        released_symbol=str(row[COL_GENE]),
        fitness=None if pd.isna(fitness) else float(fitness),
        sd=None if pd.isna(sd) else float(sd),
        predicted_activity=float(row[COL_PREDICTED]),
        family_retained=bool(row[COL_FAMILY]),
        bw25113_locus=None if target is None else target.bw25113_locus,
        drop_reason=reason,
    )


def fitness_of(verdict: RowVerdict) -> float:
    """The stored value of a kept row; a kept row always carries one."""
    if verdict.fitness is None:
        raise RuntimeError(
            f"{verdict.spacer} has no released fitness and should have been dropped"
        )
    return verdict.fitness


class BuildAccounting(BaseModel):
    """Everything a reader needs to audit one build's retention arithmetic."""

    dataset: str
    released_rows: int
    released_genes: int
    released_parent_spacers: int
    kept_records: int
    dropped_rows: int
    dropped_rows_by_reason: dict[str, int]
    stored_genes: int
    stored_parent_spacers: int
    stored_fully_complementary: int
    stored_single_mismatch: int
    stored_without_sd: int
    stored_in_excluded_series: int
    stored_negative: int
    unresolved_genes: dict[str, str]
    reconciliation: LocusTagReconciliation
    notes: list[str] = []

    def check(self) -> None:
        """Nothing may vanish between the released sheet and the written store."""
        if self.kept_records + self.dropped_rows != self.released_rows:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_rows} "
                f"dropped != {self.released_rows} released rows"
            )
        total = sum(self.dropped_rows_by_reason.values())
        if total != self.dropped_rows:
            raise RuntimeError(
                f"{self.dataset}: the per-reason drops total {total}, not "
                f"{self.dropped_rows}"
            )
        if self.dropped_rows_by_reason != EXPECTED_DROPS:
            raise RuntimeError(
                f"{self.dataset}: drops {self.dropped_rows_by_reason} are not the "
                f"pinned {EXPECTED_DROPS}"
            )
        if (
            self.stored_fully_complementary + self.stored_single_mismatch
            != self.kept_records
        ):
            raise RuntimeError(
                f"{self.dataset}: {self.stored_fully_complementary} parent + "
                f"{self.stored_single_mismatch} mismatched != {self.kept_records}"
            )
        if (self.stored_negative, self.stored_without_sd) != (
            EXPECTED_NEGATIVE,
            EXPECTED_WITHOUT_SD,
        ):
            raise RuntimeError(
                f"{self.dataset}: {self.stored_negative} negative values and "
                f"{self.stored_without_sd} without an SD are not the pinned "
                f"{EXPECTED_NEGATIVE} and {EXPECTED_WITHOUT_SD}"
            )


def _write_accounting(accounting: BuildAccounting, preprocess_dir: str) -> None:
    """Validate and write ``preprocess/build_accounting.json``."""
    accounting.check()
    os.makedirs(preprocess_dir, exist_ok=True)
    with open(osp.join(preprocess_dir, "build_accounting.json"), "w") as handle:
        handle.write(accounting.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class MismatchCrispriFitnessHawkins2020Dataset(ExperimentDataset):
    """Hawkins 2020 per-sgRNA relative fitness of E. coli essential-gene knockdowns."""

    REFERENCE_STRAIN: ClassVar[Literal["BW25113"]] = "BW25113"
    #: Measured on the pinned bytes: 317 of the 318 released b-numbers reach a BW25113
    #: locus (0.9969). The floor sits below that so a small annotation move reports, and
    #: far above it so a large one stops the build.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 0.95

    def __init__(
        self,
        root: str = "data/torchcell/mismatch_crispri_fitness_hawkins2020",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the BW25113 genome keys the records, MG1655 the released names."""
        self.ecoli_genome = ecoli_genome
        self._mg1655_genome: EcoliK12MG1655Genome | None = None
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
        """Table S3, the released fitness workbook."""
        return [TABLE_S3_FILENAME]

    def download(self) -> None:
        """Link the pinned workbook into ``raw/`` after checking it against the manifest."""
        _link_table(self.raw_dir)
        log.info("Hawkins 2020 Table S3 linked into %s (sha256 verified)", self.raw_dir)

    def _genome(self) -> EcoliK12BW25113Genome:
        """The injected BW25113 genome, or one opened from the genomes tier."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        genome: EcoliK12Genome = self.ecoli_genome
        if not isinstance(genome, EcoliK12BW25113Genome):
            raise TypeError(
                f"{self.name} needs the BW25113 genome, got {type(genome).__name__}"
            )
        return genome

    def _mg1655(self) -> EcoliK12MG1655Genome:
        """The MG1655 genome the released b-numbers are reconciled against."""
        if self._mg1655_genome is None:
            opened = bacterial_genome("ecoli", MG1655_STRAIN_NAME)
            if not isinstance(opened, EcoliK12MG1655Genome):
                raise TypeError(
                    f"{self.name} resolves released names against MG1655, got "
                    f"{type(opened).__name__}"
                )
            self._mg1655_genome = opened
        genome = self._mg1655_genome
        if not isinstance(genome, EcoliK12MG1655Genome):
            raise TypeError(
                f"{self.name} resolves released names against MG1655, got "
                f"{type(genome).__name__}"
            )
        return genome

    @post_process
    def process(self) -> None:
        """Build one record per released sgRNA with a measured fitness; write LMDB."""
        verify_raw_files(self.raw_dir, {TABLE_S3_FILENAME: TABLE_S3_SHA256})
        frame = read_ecoli_sheet(osp.join(self.raw_dir, TABLE_S3_FILENAME))
        bw25113 = self._genome()
        report = resolve_targets(
            [str(name) for name in frame[COL_LOCUS]], self._mg1655(), bw25113
        )
        report.reconciliation.require_resolved(self.MIN_RESOLVED_FRACTION)
        released_genes = int(frame[COL_LOCUS].nunique())
        if released_genes != EXPECTED_RELEASED_GENES:
            raise RuntimeError(
                f"{released_genes} released genes, not the pinned "
                f"{EXPECTED_RELEASED_GENES}"
            )
        targets = report.stored
        verdicts = [
            classify_row(cast(Mapping[str, Any], row), targets)
            for row in frame.to_dict(orient="records")
        ]
        kept = [verdict for verdict in verdicts if not verdict.drop_reason]
        designs = {
            verdict.spacer: mismatch_design(verdict.spacer, verdict.parent_spacer)
            for verdict in kept
        }

        environment = screen_environment()
        reference = BacterialEnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=host_reference(),
            environment_reference=environment,
            phenotype_reference=reference_phenotype(),
        )
        pub = publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for verdict in tqdm(kept, desc="hawkins2020"):
                experiment = BacterialEnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=build_genotype(
                        designs[verdict.spacer],
                        targets[verdict.b_number],
                        verdict.released_symbol,
                    ),
                    environment=environment,
                    phenotype=guide_phenotype(fitness_of(verdict), verdict.sd),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                idx += 1
        env.close()
        interned_env.close()
        if idx != EXPECTED_RECORDS:
            raise RuntimeError(
                f"{idx} records written, not the pinned {EXPECTED_RECORDS}"
            )

        pd.DataFrame(
            [
                {
                    "spacer": verdict.spacer,
                    "parent_spacer": verdict.parent_spacer,
                    "b_number": verdict.b_number,
                    "released_symbol": verdict.released_symbol,
                    "bw25113_locus": verdict.bw25113_locus or "",
                    "relative_fitness_mean": verdict.fitness,
                    "relative_fitness_sd": verdict.sd,
                    "predicted_sgrna_activity_not_stored": verdict.predicted_activity,
                    "family_retained": verdict.family_retained,
                    "drop_reason": verdict.drop_reason,
                }
                for verdict in verdicts
            ]
        ).to_csv(osp.join(self.preprocess_dir, "guide_retention.csv"), index=False)

        stored_genes = {verdict.bw25113_locus for verdict in kept}
        stored_parents = {verdict.parent_spacer for verdict in kept}
        parents = sum(
            1 for verdict in kept if designs[verdict.spacer].mismatch_index is None
        )
        if len(stored_genes) != EXPECTED_GENES:
            raise RuntimeError(
                f"{len(stored_genes)} stored genes, not the pinned {EXPECTED_GENES}"
            )
        if len(stored_parents) != EXPECTED_STORED_PARENTS:
            raise RuntimeError(
                f"{len(stored_parents)} stored parent spacers, not the pinned "
                f"{EXPECTED_STORED_PARENTS}"
            )
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                released_rows=len(frame),
                released_genes=released_genes,
                released_parent_spacers=int(frame[COL_PARENT].nunique()),
                kept_records=idx,
                dropped_rows=len(frame) - idx,
                dropped_rows_by_reason={
                    reason: sum(
                        1 for verdict in verdicts if verdict.drop_reason == reason
                    )
                    for reason in EXPECTED_DROPS
                },
                stored_genes=len(stored_genes),
                stored_parent_spacers=len(stored_parents),
                stored_fully_complementary=parents,
                stored_single_mismatch=idx - parents,
                stored_without_sd=sum(1 for v in kept if v.sd is None),
                stored_in_excluded_series=sum(1 for v in kept if not v.family_retained),
                stored_negative=sum(
                    1 for v in kept if v.fitness is not None and v.fitness < 0
                ),
                unresolved_genes=report.unresolved,
                reconciliation=report.reconciliation,
                notes=[
                    "the released 'relative fitness (predicted)' column is the linear "
                    "model's predicted sgRNA ACTIVITY, not a predicted fitness, and it "
                    "is NOT stored: it is exactly 1.0 for every fully complementary "
                    "spacer by the model's own definition, carries an R^2 of 0.56 with "
                    "an 11-fold cross-validation mean squared error of 0.10 +/- 0.08, "
                    "and a preliminary version of the same model chose the library's "
                    "mismatches. Every value is kept in guide_retention.csv",
                    "what the record carries instead of the dose is the guide's "
                    "MISMATCH DESIGN, on the perturbation's description: the parent "
                    "spacer, the 0-based mismatch position from the 5' end and the base "
                    "substitution, with the 20-nt spacer itself on the CrisprConstruct",
                    "the host is BW25113 and the release names MG1655 b-numbers: every "
                    "stored locus tag is DERIVED through the one-to-one ECK synonym "
                    "join and typed as such on the perturbation. The crosswalk is "
                    "checked against the authors' own BW25113_ tags on the workbook's "
                    "10-15 sheet, which agree on all 317 crossed genes",
                    "the one gene that reaches no BW25113 locus is sokE (b4700), which "
                    "ASM584v2 retires and which carries no ECK synonym; 37 of its 99 "
                    "released rows carry a fitness and are dropped",
                    "the authors' family_retained flag is stored as data, not applied: "
                    "it is step 1 of the paper's CURVE pipeline, which excludes a "
                    "series whose parent guide looks non-functional, and it bears on "
                    "the refused predicted activity rather than on whether the fitness "
                    "was measured",
                    "the 10-to-15-doubling sheet is REFUSED: its released SD column is "
                    "byte-identical to the 10-doubling sheet's in all 22,313 shared "
                    "rows, its 1,000 non-targeting controls spread 2.3-fold wider (SD "
                    "0.1935 against 0.0825), the two windows correlate at r 0.598, and "
                    "the Methods describe no 15-doubling sampling for E. coli",
                    "negative values are kept: the release defines relative fitness "
                    "below 0 as active depletion from the pool, which is why this is an "
                    "EnvironmentResponsePhenotype and not a FitnessPhenotype",
                    "no record stores a non-targeting control guide: it perturbs no "
                    "gene, so the 1,000 controls are the phenotype_reference at "
                    "relative fitness 1.0 instead",
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Hawkins2020 mismatch-CRISPRi: %d records over %d genes and %d parent "
            "spacers (%d dropped)",
            idx,
            len(stored_genes),
            len(stored_parents),
            len(frame) - idx,
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4)
# --------------------------------------------------------------------------- #
class StoreSummary(BaseModel):
    """What one memory-bounded pass over a built store sees."""

    n_records: int
    loci: tuple[str, ...]
    n_spacers: int
    malformed_spacers: tuple[str, ...]
    designs_without_a_parent: tuple[str, ...]
    screens: dict[str, int]
    pins: tuple[tuple[str, str, str], ...]
    derived_from_a_b_number: int
    negative_responses: int


_DESIGN_RE = re.compile(
    r"mismatch-CRISPRi guide, (?:fully complementary spacer [ACGT]{20} \(the parent of "
    r"its series\)|spacer [ACGT]{20}: a single mismatch to parent spacer [ACGT]{20} at "
    r"0-based position \d{1,2} from the 5' end \([ACGT] to [ACGT]\), which titrates "
    r"knockdown below the parent's)$"
)


def summarize_store(records: Iterable[Mapping[str, Any]]) -> StoreSummary:
    """Accumulate the supplementary rows' inputs in one pass over the records."""
    loci: set[str] = set()
    spacers: set[str] = set()
    malformed: set[str] = set()
    undescribed: set[str] = set()
    screens: dict[str, int] = {}
    pins: set[tuple[str, str, str]] = set()
    derived = 0
    negative = 0
    n_records = 0
    for record in records:
        n_records += 1
        experiment = record["experiment"]
        for perturbation in experiment["genotype"]["perturbations"]:
            loci.add(str(perturbation["systematic_gene_name"]))
            spacer = str(perturbation["crispr"]["guide_sequence"])
            spacers.add(spacer)
            if SPACER_RE.match(spacer) is None:
                malformed.add(spacer)
            if _DESIGN_RE.search(str(perturbation["description"])) is None:
                undescribed.add(spacer)
            mapping = perturbation.get("identifier_mapping") or {}
            if (
                mapping.get("route") == "eck_crosswalk"
                and BNUMBER_RE.match(str(mapping.get("source_identifier"))) is not None
            ):
                derived += 1
        phenotype = experiment["phenotype"]
        screen = str(phenotype["screen_id"])
        screens[screen] = screens.get(screen, 0) + 1
        if float(phenotype["environment_response"]) < 0:
            negative += 1
        reference = record["reference"]["genome_reference"]
        background = reference.get("background") or {}
        pins.add(
            (
                str(reference.get("assembly_set")),
                str(reference.get("assembly_accession")),
                str(background.get("name")),
            )
        )
    return StoreSummary(
        n_records=n_records,
        loci=tuple(sorted(loci)),
        n_spacers=len(spacers),
        malformed_spacers=tuple(sorted(malformed)),
        designs_without_a_parent=tuple(sorted(undescribed)),
        screens=dict(sorted(screens.items())),
        pins=tuple(sorted(pins)),
        derived_from_a_b_number=derived,
        negative_responses=negative,
    )


def spacers_are_twenty_nt(summary: StoreSummary) -> LevelResult:
    """L2 SUPPLEMENTARY: every record's guide spacer is a 20-nt ACGT sequence.

    The spacer is this dataset's perturbation identity, and the mismatch it carries is
    the whole dose axis, so a malformed one is a record whose design cannot be read back.
    """
    return LevelResult(
        level=Level.L2,
        name="guide_spacers_are_twenty_nt_acgt",
        passed=not summary.malformed_spacers,
        message=(
            f"SUPPLEMENTARY: {summary.n_spacers} distinct spacers; "
            f"{len(summary.malformed_spacers)} malformed"
        ),
        details={
            "n_spacers": summary.n_spacers,
            "malformed": list(summary.malformed_spacers[:20]),
        },
    )


def designs_name_their_parent(summary: StoreSummary) -> LevelResult:
    """L2 SUPPLEMENTARY: every perturbation's description states its mismatch design.

    The description is where this dataset carries the part of the graded dose that is
    measured rather than predicted, so a record without it is a knockdown of unknown
    strength with no way to recover one.
    """
    return LevelResult(
        level=Level.L2,
        name="perturbation_descriptions_state_the_mismatch_design",
        passed=not summary.designs_without_a_parent,
        message=(
            f"SUPPLEMENTARY: {summary.n_spacers} distinct spacers; "
            f"{len(summary.designs_without_a_parent)} carry no readable design"
        ),
        details={"without_design": list(summary.designs_without_a_parent[:20])},
    )


def every_locus_is_derived_and_pinned(
    summary: StoreSummary, genome: EcoliK12BW25113Genome
) -> LevelResult:
    """L1 SUPPLEMENTARY: every stored locus is a BW25113 locus reached from a b-number.

    Two claims in one row, because they are one decision: the records are keyed in the
    BW25113 namespace the host lives in, and every one of those keys is DERIVED from the
    MG1655 b-number the release named, with the ECK route typed on the perturbation.
    """
    absent = [locus for locus in summary.loci if locus not in genome.genbank.loci]
    expected_pins = {
        ("ecoli_K12_BW25113_ASM75055v1", "GCA_000750555.1", str(HOST_CASSETTE.value))
    }
    passed = (
        not absent
        and summary.derived_from_a_b_number == summary.n_records
        and set(summary.pins) == expected_pins
    )
    return LevelResult(
        level=Level.L1,
        name="stored_loci_are_bw25113_loci_derived_from_the_released_b_number",
        passed=passed,
        message=(
            f"SUPPLEMENTARY: {len(summary.loci)} stored loci, {len(absent)} absent from "
            f"the BW25113 annotation; {summary.derived_from_a_b_number} of "
            f"{summary.n_records} perturbations carry an eck_crosswalk mapping from a "
            f"b-number; {len(summary.pins)} distinct (assembly, background) pins"
        ),
        details={
            "n_loci": len(summary.loci),
            "absent": absent[:20],
            "pins": [list(pin) for pin in summary.pins],
            "derived": summary.derived_from_a_b_number,
        },
    )


def one_screen_and_the_depleted_tail(summary: StoreSummary) -> LevelResult:
    """L3 SUPPLEMENTARY: one screen, and the negative tail the release defines is intact.

    ``EXPECTED_NEGATIVE`` of the stored values are below 0, which the release calls
    active depletion from the pool. A build that silently clamped them (the
    FitnessPhenotype behavior this dataset avoids) would show 0 here.
    """
    expected = {SCREEN_ID: EXPECTED_RECORDS}
    return LevelResult(
        level=Level.L3,
        name="one_screen_and_the_negative_tail_is_intact",
        passed=(
            summary.screens == expected
            and summary.negative_responses == EXPECTED_NEGATIVE
        ),
        message=(
            f"SUPPLEMENTARY: records per screen {summary.screens}; "
            f"{summary.negative_responses} negative relative fitnesses"
        ),
        details={"per_screen": summary.screens, "negative": summary.negative_responses},
    )


def verify_build(
    dataset_root: str,
    *,
    genome: EcoliK12BW25113Genome | None = None,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 verifier on a built tree and write its report.

    The LMDB is streamed and never materialized: the store holds 24,149 records, and the
    four supplementary rows come from a second such pass through
    :func:`summarize_store`. The L4 universe is every GenBank locus of the BW25113
    assembly the records pin.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    bw25113 = genome or bacterial_genome("ecoli", REFERENCE_STRAIN_NAME, data_root)
    if not isinstance(bw25113, EcoliK12BW25113Genome):
        raise TypeError(f"expected the BW25113 genome, got {type(bw25113).__name__}")
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name="MismatchCrispriFitnessHawkins2020Dataset",
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{TABLE_S3_REL}",
            citation_key=CITATION_KEY,
            sha256=TABLE_S3_SHA256,
            method=(
                "Table S3 (publisher mmc4.xlsx), sheet 'relative fitness (eco data)': "
                "one BacterialEnvironmentResponseExperiment per released sgRNA that "
                "carries a measured fitness. Label = the released 'relative fitness "
                "(mean)', the strain's doublings divided by the wild type's over about "
                "10 doublings of pooled competition, normalized to the median of 1,000 "
                "non-targeting control sgRNAs in the same sample, so "
                "measurement_type=relative_growth_rate and 1.0 is wild-type growth; "
                "778 stored values are below 0, which the release defines as active "
                "depletion from the pool. Uncertainty = the released 'relative fitness "
                "(stddev)' as sample_sd over n_samples=4 biological replicates, with "
                "typed gaps on the 1,870 records whose SD cell is blank. Genotype = one "
                "BacterialCrisprInterferencePerturbation on the target's BW25113 locus "
                "(DERIVED from the released MG1655 b-number through the one-to-one ECK "
                "synonym join), effector dCas9, guide_sequence = the released 20-nt "
                "mismatched spacer, and the perturbation's description naming the parent "
                "spacer, the 0-based mismatch position and the base substitution. The "
                "released 'relative fitness (predicted)' column is the linear model's "
                "predicted sgRNA ACTIVITY (exactly 1.0 for every parent spacer, R^2 "
                "0.56) and is NOT stored: a model prediction has no field on this "
                "schema, and it is filed as a schema finding. Environment: LB with 100 "
                "ug/mL ampicillin (formulation unstated by the source) at 37 C with 1 "
                "mM IPTG inducing dCas9 over about 10 doublings. Reference = the "
                "non-targeting controls' 1.0 by that normalization. DROPPED: 12,104 "
                "rows whose released mean is blank (the authors' own 100-counts-at-t0 "
                "floor) and 37 rows on sokE (b4700), which ASM584v2 retires and which "
                "carries no ECK synonym to BW25113"
            ),
            page=(
                "Cell Systems 2020 11:523-535.e9 (doi:10.1016/j.cels.2020.09.009); STAR "
                "Methods 'Relative Fitness Experiments', 'Relative Fitness Analysis' "
                f"and 'sgRNA Plasmid Library Design'; paper.md sha256={PAPER_MD_SHA256}"
            ),
            retrieved=TABLE_S3_RETRIEVED_AT,
        ),
        expected_count=expected_count,
        reference_unit_scaled=True,
        sgd_genes=set(bw25113.genbank.loci),
        resolve_gene_name=bw25113.resolve_gene_name,
    )
    summary = summarize_store(stream_records(dataset_root))
    report.add(every_locus_is_derived_and_pinned(summary, bw25113))
    report.add(spacers_are_twenty_nt(summary))
    report.add(designs_name_their_parent(summary))
    report.add(one_screen_and_the_depleted_tail(summary))
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build the dataset under ``DATA_ROOT``, print its accounting and verify it."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/mismatch_crispri_fitness_hawkins2020")
    genome = bacterial_genome("ecoli", REFERENCE_STRAIN_NAME, data_root)
    if not isinstance(genome, EcoliK12BW25113Genome):
        raise TypeError(f"expected the BW25113 genome, got {type(genome).__name__}")
    dataset = MismatchCrispriFitnessHawkins2020Dataset(root=root, ecoli_genome=genome)
    print(f"len = {len(dataset)}")
    accounting = json.loads(
        Path(osp.join(root, "preprocess/build_accounting.json")).read_text()
    )
    print(
        json.dumps(
            {
                key: accounting[key]
                for key in (
                    "released_rows",
                    "released_genes",
                    "kept_records",
                    "dropped_rows",
                    "dropped_rows_by_reason",
                    "stored_genes",
                    "stored_parent_spacers",
                    "stored_fully_complementary",
                    "stored_single_mismatch",
                    "stored_without_sd",
                    "stored_in_excluded_series",
                    "stored_negative",
                    "unresolved_genes",
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
