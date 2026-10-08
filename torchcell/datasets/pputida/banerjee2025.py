# torchcell/datasets/pputida/banerjee2025
# [[torchcell.datasets.pputida.banerjee2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/banerjee2025
# Test file: tests/torchcell/datasets/pputida/test_banerjee2025.py
"""Banerjee 2025 DIA proteome of the p-coumarate growth-coupling cutset in P. putida.

Banerjee et al. 2025 (npj Syst. Biol. Appl., doi:10.1038/s41540-024-00480-z) completed a
four-gene growth-coupling cutset for p-coumarate (p-CA) to glutamine to indigoidine in
the KT2440 chassis D1b_gf, found the fourth gene PP_0897 to be rate limiting, and
titrated it down with two promoter swaps instead of deleting it. One dataset class,
:class:`ProteomeBanerjee2025Dataset`, serves every released shotgun-proteome sample as a
``BacterialProteinAbundanceExperiment``.

SIX SAMPLES, READ FROM FOUR RELEASED WORKBOOKS, AND WHY SIX IS THE WHOLE RELEASE.
Supplementary Data S2 (``41540_2024_480_MOESM3_ESM.zip``) holds four t-test workbooks,
each with a ``Full t-test output`` sheet carrying, per protein, ``log2_mean_<group>`` and
``log2_std_<group>`` for the two groups it compares. Eleven such columns exist across the
four workbooks and they name SIX distinct (strain, medium) groups, because two groups are
each released twice:

- the promoter arm, 2,494 proteins per workbook: ``2370`` (the parental D1b_gf),
  ``2370_pJ_PP_0897`` and ``2370_0415pPP_0897``. ``log2_mean_2370`` /
  ``log2_std_2370`` appear in BOTH promoter workbooks and are equal CELL FOR CELL
  (measured: 0 of 2,494 differ), so the two comparisons share one control sample rather
  than carrying two.
- the cross-feeding arm, 2,470 proteins per workbook:
  ``2370_M9_pCA_alanine_malate``, ``2487_M9_pCA_alanine_malate`` and
  ``2487_M9_alanine_malate``. ``2487_M9_pCA_alanine_malate`` appears in both
  cross-feeding workbooks and is likewise equal cell for cell.

Both equalities are asserted at build time (:func:`assert_shared_groups_agree`): a
released revision that makes them differ means the arm no longer shares one sample, which
would change the record count, so the build stops rather than quietly writing seven.

THE RELEASED GROUP LABELS ARE STRAIN IDS, AND WHICH STRAIN EACH ONE IS WAS SETTLED BY
MEASUREMENT, NOT BY READING. Neither the paper nor the SI carries a table mapping
``2370`` or ``2487`` onto the Supplementary Table 1 strains. Three measurements on the
pinned bytes settle it, and the loader asserts all three
(:func:`assert_pp_0897_identifies_the_strains`):

1. PP_0897 is strongly DOWN in both promoter groups against ``2370``
   (log2 fold change -4.042, p 7.8e-5 for ``pJ``; -3.283, p 1.1e-5 for ``0415``), which
   is the titration the Results describe: "PP_0897 abundance was strongly reduced ~8
   fold as expected in both PP_0897 promoter titration strains".
2. PP_0897 in ``2487_M9_pCA_alanine_malate`` is 7.93 log2 units below
   ``2370_M9_pCA_alanine_malate`` (p 5.9e-4), i.e. effectively absent, which is a
   DELETION rather than a titration; and it is FLAT between the two ``2487`` media
   (log2 fold change 0.048, p 0.83), which is what a deleted gene must do.
   So ``2487`` is D1b_gf dPP_0897 and ``2370`` is the PP_0897-intact parent.
3. The counts of differentially abundant proteins reproduce the Results text exactly.
   The paper states "only 28 metabolic and 63 non-metabolic proteins exhibited distinct
   abundance levels in the PJ23109 strain and similarly, only 19 metabolic and 37
   non-metabolic proteins had different abundances in the PPP_0415 strain"; the ``pJ``
   workbook's significance sheets hold 20 + 71 = 91 = 28 + 63 rows and the ``0415``
   workbook's hold 24 + 32 = 56 = 19 + 37. The curated ``Promoter STrainsAnalysis``
   sheet holds 134 = 91 + 56 - 13 rows, and exactly 13 of them carry both arms' fold
   changes with matching sign, which is the paper's "Only 13 proteins showed matching
   amplitudes of varying abundance (i.e., both up or down)".

THE REPLICATE COUNT DIFFERS BY ARM, THE SOURCE APPEARS TO CONTRADICT ITSELF, AND THE
T STATISTIC DECIDES. The Methods say the proteomics cultures were "grown in triplicates",
while Supplementary Figure 4's caption says "Proteomics analysis was done using four
biological replicates" and "(n=4) in B". Back-solving each workbook's own
``t-test_stat`` from its released means and standard deviations resolves it: with
t = (m1 - m2) / sqrt((s1^2 + s2^2) / n), the promoter workbooks reproduce every released
t at n = 4 and the cross-feeding workbooks at n = 3, both to machine precision (mean
absolute deviation 2e-14 and 4e-14 over the first 400 rows; every other n in 2..6 is off
by 0.16 to 2.9). So the Methods sentence describes the cross-feeding arm and the
Supplementary Figure 4 caption the promoter arm, each is quoted on its own record, and
neither is guessed. :func:`assert_replicates_back_solve` re-runs the back-solve at build
time over every released row, and the ONE exception is asserted rather than excused: the
two merged-accession labels (``Pyrc`` and ``Ubid``, each filed twice under one name with
two different protein groups) back-solve to exactly 2n, because the released analysis
pooled both groups' replicates. Those are exactly the two labels the merge rule drops, so
no stored value comes from a row whose replicate count does not hold.

WHAT IS STORED, AND THE ONE DERIVED STEP IN THE KEY. ``protein_abundance`` is the
released ``log2_mean`` as given -- a log2 DIA-NN label-free quantity, replicate mean --
and ``protein_abundance_se`` is the released ``log2_std`` over sqrt(n), the standard
error of that log2 mean on the same scale. The released ``Protein`` column is a UniProt
gene label that is a bare locus tag for 1,376 to 1,381 of the proteins and a title-cased
symbol otherwise, and the four workbooks DISAGREE ON ITS CASE: one cross-feeding workbook
writes ``PP_0002`` where the other writes ``Pp_0002`` (1,376 labels differ, all of them
locus-tag-shaped, no other difference). A locus-tag-shaped label is therefore upper-cased
before anything else happens, which is what lets the two workbooks be joined at all and
also keeps six collision-held tags inside the namespace instead of dropping them. Every
label then goes through ``reconcile_locus_tags``, and a label that does not land on a
locus of the pinned assembly is dropped with its reason, as in the sibling de Siqueira
and Carruthers proteomes.

THREE QUANTITIES ARE IN THIS PAPER AND ONLY ONE IS A DATASET.

- The PROTEOME is released as numbers, and is this dataset.
- The INDIGOIDINE TITER is not. Figures 2C, 2D, 2E, Supplementary Figures 2, 3A and 8
  plot it in mg/L, in g/g and as a specific yield in mg/CFU, and nothing in the article,
  the Supplementary Information or either Supplementary Data archive releases a titer
  table: Supplementary Data S1 is the COBRA model, workspaces and flux distributions,
  Supplementary Data S2 is the proteomics above, and Supplementary Tables 1 to 4 are
  strains, plasmids, oligos and the BIOLOG grid. ``ProductTiterExperimentReference``
  requires a ``phenotype_reference`` and ``ProductTiterPhenotype.titer`` is a required
  float, so a titer family needs a released reference titer and there is none. The whole
  family is refused, which is what the sibling Yunus 2026 and Kang 2026 loaders do for
  the same reason (:data:`TITER_NOT_A_DATASET`).
- The GROWTH RATE is not released either. Figures 2B, 2F, 2G and Supplementary Figures
  1, 6, 7 and 8 are plates and growth curves; the only per-hour numbers anywhere in the
  paper (1.14/h, 0.43/h, 0.45/h) are flux-balance PREDICTIONS from the context-specific
  models, not measurements, so writing any of them as a growth phenotype would state a
  measurement nobody made (:data:`GROWTH_NOT_A_DATASET`).

THE BIOLOG GRID IS RELEASED AND IS DELIBERATELY NOT INGESTED HERE. Supplementary Table 4
releases, for strain D1b_gf dPP_0897, a measured tetrazolium ``BIOLOG OD595`` for 95
PM1-plate carbon sources beside the matched in-silico growth prediction, and the values
are signed (negative for a substrate that is not respired), so ``FitnessPhenotype`` is
wrong for them and ``EnvironmentResponsePhenotype`` is the right shape. It is not written
because ``MeasurementType`` has no member for a redox-dye respiration endpoint: the
closest, ``colony_size``, is colony size by its own definition, and ADDING a member would
change ``MeasurementType``'s schema and therefore the serialized closure of every served
``EnvironmentResponsePhenotype`` dataset, which forces a full knowledge-graph rebuild.
The grid is recorded as a measured, released, blocked-on-one-enum-member arm
(:data:`BIOLOG_NOT_A_DATASET`); the PM2A plate's OD values were never released at all,
only the narrative that "there were only two additional carbon sources where growth was
observed: dextrin and D,L-carnitine".

THE CHASSIS IS D1b_gf AND ITS MEDIUM COMPOSITION IS DEFERRED. Every record is written
against wild-type KT2440 carrying a ``BacterialStrainBackground`` named ``D1b_gf``, whose
four host-gene deletions the paper states with their locus tags (PP_1378/pcaT,
PP_1755/fumC-II, PP_0944/fumC-I and fleQ/PP_4373) and which is itself from reference 21.
The record's OWN edit is the perturbation: a ``PromoterReplacementPerturbation`` on
PP_0897 for the two promoter groups and a ``BacterialDeletionPerturbation`` on PP_0897
for the two ``2487`` groups, while the parental groups carry none.

THE THREE HETEROLOGOUS GENES ARE NOT TYPED, AND THAT IS THE SOURCE'S LIMIT. D1b_gf also
carries a chromosomally integrated indigoidine cassette, written by the paper and the SI
as ``PP_5402-intergenic::PBAD-Sc.bpsA,Bc.sfp,Ps.glnA``. ``GeneAdditionPerturbation``
requires ``source_organism`` with no default, and the abbreviations ``Sc.``, ``Bc.`` and
``Ps.`` are never expanded in this article or its SI -- "Bacillus subtilis" occurs only
in an unrelated reference title. The organisms are stated in the earlier indigoidine
papers (Banerjee 2020, reference 21), neither of which is in the literature mirror, so
expanding them is a deferral to an unmirrored source. The cassette is therefore kept
verbatim in the background's ``genotype_statement`` and the three genes are NOT written
as ``HeterologousPathwayPerturbation``; typing them would assert three organisms the
source does not state (:data:`PATHWAY_NOT_TYPED`).

TEMPERATURE IS A TYPED GAP. The shotgun-proteomics Methods state the medium, the vessel,
the inoculum and the harvest state but no incubation temperature. Every temperature this
paper does state elsewhere is 30 C, which is why the gap is recorded rather than filled:
``Environment.temperature`` is ``None`` with a ``ProvenanceGap`` naming the absence.

UNITS AND THE MEDIUM OBJECT. The carbon regime is the environment's variable, not a
medium per condition: ``M9_DEFERRED_BANERJEE2025`` is a carbon-free M9 whose SALT amounts
the Methods hand to reference 21 ("Engineered strains were grown in a modified M9 minimal
medium as previously described21"), and each condition's p-CA, D-alanine and L-malate are
``EnvironmentPhysicalPerturbation(factor=carbon_source)``, which is the de Siqueira
convention. ``p-CA`` is resolved through the compound table's ``p-coumaric acid`` row
(the same reading Lim 2022 records for its "coumarate"); ``D-alanine`` and ``L-malate``
have no curated row yet, so each carries a typed gap on ``inchikey`` and adding the two
rows is raised in the PR rather than done here.
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
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
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
    verify_sha256,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import M9_DEFERRED_BANERJEE2025
from torchcell.datamodels.schema import (
    AlleleEdit,
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    BacterialBackgroundAllele,
    BacterialDeletionPerturbation,
    BacterialGeneNamespace,
    BacterialProteinAbundanceExperiment,
    BacterialProteinAbundanceExperimentReference,
    BacterialReferenceStrain,
    BacterialStrainBackground,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    Genotype,
    PhysicalFactor,
    PromoterReplacementPerturbation,
    ProteinAbundancePhenotype,
    Publication,
)
from torchcell.datasets.bacteria_common import (
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
)
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1038/s41540-024-00480-z"
PMCID = "PMC11732973"
TITLE = (
    "Addressing genome scale design tradeoffs in Pseudomonas putida for bioconversion "
    "of an aromatic carbon source"
)
CITATION_KEY = "banerjeeAddressingGenomeScale2025"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: The PMC Article Datasets bucket prefix the SI archive was retrieved from.
PMC_PREFIX = f"{PMCID}.1"
#: Supplementary Data S2, the archive holding the proteomics workbooks.
ARCHIVE_FILENAME = "41540_2024_480_MOESM3_ESM.zip"
ARCHIVE_SHA256 = "09648ccbac2f12bc89bd5ea1582837a7b6e1fa771c3eede4929dc666120f6d0f"
ARCHIVE_URL = (
    f"https://pmc-oa-opendata.s3.amazonaws.com/{PMC_PREFIX}/{ARCHIVE_FILENAME}"
)
#: Supplementary Data S1, the COBRA archive. Named for the record, never retrieved here.
MODEL_ARCHIVE_FILENAME = "41540_2024_480_MOESM2_ESM.zip"
MODEL_ARCHIVE_SHA256 = (
    "cf9b5d1b7a33c6558d7cc54989f5e592fee51c0b624596a1b1c91be763d86b14"
)
SI_RETRIEVED_AT = "2026-10-08"

#: The article OCR in the torchcell-library mirror: every ``_paper`` quote's anchor.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "74d7040a0a7ec721f18e5fc49d855a5c9c6eaeee9ff7bf166ec675824e27c489"
#: The Supplementary Information OCR: every ``_si`` quote's anchor.
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "25cc7bbffa0e0d2ad89fe0c2748cd1966940f16e8dd8a1662a101ec1b55362b4"
#: The Supplementary Information PDF those bytes were OCR'd from.
SI1_PDF = "si/si1.pdf"
SI1_PDF_SHA256 = "e59a55234b50d5f64f5e941baa4d58c0e1f8fe8bb9ffe248a7594b55becb8ebf"
MINERU_VERSION = "2.7.6"

PRIDE_ACCESSION = "PXD050285"

KT2440_NAMESPACE: BacterialGeneNamespace = "pputida_kt2440_locus_tag"
KT2440_ASSEMBLY_SET: BacterialAssemblySet = "pputida_KT2440_ASM756v2"
WT_STRAIN: BacterialReferenceStrain = "KT2440"

#: The chassis every record is written against.
CHASSIS = "D1b_gf"
#: The PP_0897 locus the whole study turns on, and the four cutset genes with their
#: annotation symbols (``None`` where the annotation carries no symbol).
PP_0897 = "PP_0897"
CUTSET_LOCI: tuple[tuple[str, str, str], ...] = (
    ("PP_1378", "pcaT", "ΔPP_1378"),
    ("PP_1755", "fumC-II", "ΔPP_1755"),
    ("PP_0944", "fumC-I", "ΔPP_0944"),
    ("PP_4373", "fleQ", "ΔfleQ/ΔPP_4373"),
)
#: The locus whose promoter the weak variant borrows (``rpe`` in the annotation).
PP_0415 = "PP_0415"

# --------------------------------------------------------------------------- #
# The released groups
# --------------------------------------------------------------------------- #
ARM_PROMOTER = "promoter-variant"
ARM_CROSSFEED = "cross-feeding"

CONDITION_PCA = "M9 60 mM p-CA"
CONDITION_PCA_ALA_MAL = "M9 50 mM p-CA + 70 mM D-alanine + 70 mM L-malate"
CONDITION_ALA_MAL = "M9 D-alanine + L-malate"

STRAIN_PARENT = CHASSIS
STRAIN_PJ = "D1b_gf PJ23109-PP_0897"
STRAIN_0415 = "D1b_gf Ppp_0415-PP_0897"
STRAIN_DELETION = "D1b_gf ΔPP_0897"

WORKBOOK_PJ = "t-test_pJ_PP_0897_vs_Control_20230307-000644.xlsx"
WORKBOOK_0415 = "t-test_0415pPP_0897_vs_Control_20230425-000301.xlsx"
WORKBOOK_PAM_STRAINS = (
    "t-test_TEAM-2487_M9_pCA_alanine_malate_vs_TEAM-2370_M9_pCA_alanine_malate"
    "_20230626-224656.xlsx"
)
WORKBOOK_PAM_MEDIA = (
    "t-test_TEAM-2487_M9_pCA_alanine_malate_vs_TEAM-2487_M9_alanine_malate"
    "_20230626-224656.xlsx"
)
#: The curated analysis workbook. Read only as a build oracle, never as a value source.
WORKBOOK_ANALYSIS = "DataAnalysis.xlsx"

#: ``relpath -> (zip member, sha256)`` for every member deposited in the raw mirror.
RAW_MEMBERS: dict[str, tuple[str, str]] = {
    f"si/{WORKBOOK_PJ}": (
        f"Data S2/{WORKBOOK_PJ}",
        "02bb64f4f6cfe49507430478c41b83e28d52374cc728d052f581821ee575d80e",
    ),
    f"si/{WORKBOOK_0415}": (
        f"Data S2/{WORKBOOK_0415}",
        "53e36bec86b8c97cea094501a0ed6bfb9eef21da9d5eff519ccb8d71b42d21ed",
    ),
    f"si/{WORKBOOK_PAM_STRAINS}": (
        f"Data S2/{WORKBOOK_PAM_STRAINS}",
        "3817e0501952f93f2459bfcf9cfca8bb4a4a1cfb52dd3674c97ab61c11f71ea4",
    ),
    f"si/{WORKBOOK_PAM_MEDIA}": (
        f"Data S2/{WORKBOOK_PAM_MEDIA}",
        "cc4a05f1e19074a99e25e87ba2d5ef6b5f95d3d2bce8177e7f24e28277006104",
    ),
    f"si/{WORKBOOK_ANALYSIS}": (
        f"Data S2/{WORKBOOK_ANALYSIS}",
        "a888c57f06e47f3cf35b232607f2c36b4a2ecd7807b0109c84089038dcf405e1",
    ),
}

#: The sheet every value is read from.
FULL_SHEET = "Full t-test output"
#: The curated sheet the 134 / 13 oracle reads.
ANALYSIS_SHEET = "Promoter STrainsAnalysis"
#: The two significance sheets whose row counts reproduce the Results text.
SIGNIFICANCE_SHEETS: tuple[str, str] = (
    "p-val_Significant_UP",
    "p-val_Significant_DOWN",
)


class ProteomeGroup(BaseModel):
    """One released (strain, medium) proteome sample and where its two columns live."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    strain: str
    arm: str
    condition: str
    workbook: str

    @property
    def mean_column(self) -> str:
        """The released replicate-mean column of this group."""
        return f"log2_mean_{self.key}"

    @property
    def sd_column(self) -> str:
        """The released replicate standard-deviation column of this group."""
        return f"log2_std_{self.key}"


#: Every released sample, in the order the records are written.
PROTEOME_GROUPS: tuple[ProteomeGroup, ...] = (
    ProteomeGroup(
        key="2370",
        strain=STRAIN_PARENT,
        arm=ARM_PROMOTER,
        condition=CONDITION_PCA,
        workbook=WORKBOOK_PJ,
    ),
    ProteomeGroup(
        key="2370_pJ_PP_0897",
        strain=STRAIN_PJ,
        arm=ARM_PROMOTER,
        condition=CONDITION_PCA,
        workbook=WORKBOOK_PJ,
    ),
    ProteomeGroup(
        key="2370_0415pPP_0897",
        strain=STRAIN_0415,
        arm=ARM_PROMOTER,
        condition=CONDITION_PCA,
        workbook=WORKBOOK_0415,
    ),
    ProteomeGroup(
        key="2370_M9_pCA_alanine_malate",
        strain=STRAIN_PARENT,
        arm=ARM_CROSSFEED,
        condition=CONDITION_PCA_ALA_MAL,
        workbook=WORKBOOK_PAM_STRAINS,
    ),
    ProteomeGroup(
        key="2487_M9_pCA_alanine_malate",
        strain=STRAIN_DELETION,
        arm=ARM_CROSSFEED,
        condition=CONDITION_PCA_ALA_MAL,
        workbook=WORKBOOK_PAM_STRAINS,
    ),
    ProteomeGroup(
        key="2487_M9_alanine_malate",
        strain=STRAIN_DELETION,
        arm=ARM_CROSSFEED,
        condition=CONDITION_ALA_MAL,
        workbook=WORKBOOK_PAM_MEDIA,
    ),
)
EXPECTED_PROTEOME_RECORDS = len(PROTEOME_GROUPS)

#: The parental sample each arm's records are written against.
ARM_REFERENCE_KEY: dict[str, str] = {
    ARM_PROMOTER: "2370",
    ARM_CROSSFEED: "2370_M9_pCA_alanine_malate",
}
#: Replicates per arm, each quoted AND back-solved from the released t statistic.
ARM_REPLICATES: dict[str, int] = {ARM_PROMOTER: 4, ARM_CROSSFEED: 3}
#: The two columns released twice, with the workbook pair that must agree on them.
SHARED_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("2370", WORKBOOK_PJ, WORKBOOK_0415),
    ("2487_M9_pCA_alanine_malate", WORKBOOK_PAM_STRAINS, WORKBOOK_PAM_MEDIA),
)
#: Proteins per workbook, measured on the pinned bytes.
EXPECTED_PROTEIN_ROWS: dict[str, int] = {
    WORKBOOK_PJ: 2494,
    WORKBOOK_0415: 2494,
    WORKBOOK_PAM_STRAINS: 2470,
    WORKBOOK_PAM_MEDIA: 2470,
}
#: The Results text's differentially-abundant counts, per promoter workbook.
EXPECTED_SIGNIFICANT: dict[str, int] = {WORKBOOK_PJ: 91, WORKBOOK_0415: 56}
#: Rows of the curated sheet, and the subset carrying both arms' fold changes.
EXPECTED_ANALYSIS_ROWS = 134
EXPECTED_COMMON_PROTEINS = 13

MEASUREMENT_TYPE = "diann_lfq_log2_intensity_replicate_mean"
#: Measured on the pinned workbooks: 2,258 of 2,492 promoter labels (0.9061) and 2,230
#: of 2,468 cross-feeding labels (0.9036) resolve to a locus of this assembly. The
#: threshold sits just below the lower of the two; a real drop means the keying changed.
MIN_RESOLVED_FRACTION = 0.90

#: ``PP_####``, the shape whose CASE the four workbooks disagree on.
LOCUS_LABEL_RE = re.compile(r"PP_\d{4}", re.IGNORECASE)
#: Back-solve tolerance on the released t statistic (machine precision, not a fit).
T_STATISTIC_TOL = 1e-8
#: The merged-accession labels back-solve to TWICE their arm's replicate count: the
#: released analysis pooled both protein groups' replicates under the one label.
#: Measured on the pinned workbooks, exactly ``Pyrc`` and ``Ubid`` do this, and they are
#: exactly the two labels the merge rule drops.
MERGED_LABEL_REPLICATE_FACTOR = 2


# --------------------------------------------------------------------------- #
# Sourced values: every number below quotes sha256-pinned mirrored bytes
# --------------------------------------------------------------------------- #
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


def _si(value: Any, quote: str, *, page: str, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the pinned Supplementary Information OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI1_MD,
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method=(
                f"MinerU {MINERU_VERSION} OCR of the Supplementary Information PDF "
                f"({SI1_PDF}, sha256 {SI1_PDF_SHA256})"
            ),
            page=page,
        ),
    )


_METHODS_CULTIVATION = "Methods, 'Cultivation of Pseudomonas putida'"
_METHODS_RECOMBINEERING = "Methods, 'Construction of plasmids and targeted genomic mutants via recombineering'"
_METHODS_INDIGOIDINE = (
    "Methods, 'Indigoidine colorimetric quantification and specific indigoidine yield "
    "assessment'"
)
_METHODS_BIOLOG = "Methods, 'BIOLOG phenotype microarray'"
_METHODS_PROTEOMICS = "Methods, 'Shotgun proteomics analysis'"
_METHODS_MODELING = "Methods, 'Constraint-based modeling and simulations'"
_RESULTS_CUTSET = (
    "Results, 'A completed p-CA/indigoidine cutset and modulating PP_0897 gene "
    "expression impacts strain growth, production and proteome response'"
)
_RESULTS_CONTEXT = (
    "Results, 'Context-specific GSMM for strains with various growth coupling "
    "implementations explains the highly constrained design space'"
)
_RESULTS_SUPPLEMENTATION = (
    "Results, 'Metabolite supplementation in the completed cutset does not restore "
    "p-CA catabolism'"
)
_SI_TABLE1 = "Supplementary Table 1, 'Strains Used in This Study'"
_SI_TABLE4 = "Supplementary Table 4, 'List of BIOLOG metabolites tested in this study'"
_SI_FIGURE4 = "Supplementary Figure 4 caption"
_FIG2 = "Fig. 2 caption"

_Q_M9_DEFERRED = (
    "Engineered strains were grown in a modified M9 minimal medium as previously "
    "described21, and $\\boldsymbol { p }$ - CA (Sigma-Aldrich, Product No. C9008), "
    "L-malic acid sodium salt (Sigma-Aldrich, Product No. M1125) and D-alanine "
    "(Sigma-Aldrich, Product No. A7377) were used at the concentrations indicated in "
    "the figure legends."
)
_Q_PROTEOMICS_CULTURE = (
    "The D1b_gf strains designed in this study were grown in triplicates in M9 "
    "$6 0 \\ : \\mathrm { m M } \\ : p – \\mathrm { C } A$ , or M9 50 mM "
    "$\\boldsymbol { p }$ -CA supplemented with $7 0 \\mathrm { m M }$ D-alanine and "
    "$7 0 \\mathrm { m M L }$ -malate when indicated, using $1 0 \\mathrm { m L }$ "
    "culture tubes."
)
_Q_HARVEST = (
    "The strains were back-diluted to a starting $\\mathrm { O D } _ { 6 0 0 }$ of 0.05 "
    "from a saturated culture, and samples were harvested when cells reached mid-log "
    "phase $\\mathrm { ( O D } _ { 6 0 0 } = 0 . 8 \\mathrm { - } 1 _ { . }$ ) and "
    "stored at $- 8 0 ^ { \\circ } \\mathrm { C }$ until sample preparation."
)
_Q_INSTRUMENT = (
    "The resulting peptide samples were analyzed on an Agilent 1290 UHPLC system "
    "coupled to a Thermo Scientific Orbitrap Exploris 480 mass spectrometer for "
    "discovery proteomics69."
)
_Q_DIANN_DB = (
    "The database used in the DIA-NN search (library-free mode) is the latest Uniprot "
    "$P .$ . putida KT2440 proteome FASTA sequence plus the protein sequences of "
    "heterogeneous pathway genes and common proteomic contaminants."
)
_Q_FDR = (
    "Output main DIA-NN reports were filtered with a global "
    "$\\mathrm { F D R } = 0 . 0 1$ on both the precursor level and protein group level."
)
_Q_LFQ = (
    "A jupyter notebook written in Python executed label-free quantification (LFQ) "
    "data analysis on the DIA-NN peptide quantification report and the details of the "
    "analysis were described in the established protocol70."
)
_Q_TWO_THOUSAND = (
    "To characterize global perturbations on cell metabolism from reduced PP_0897 "
    "protein levels, we used proteomics to map proteins with differential abundance "
    "among the ${ \\sim } 2 0 0 0$ proteins quantified, comparing the promoter variants "
    "to the parental D1b_gf strain."
)
_Q_SIGNIFICANT_COUNTS = (
    "Only 28 metabolic and 63 non-metabolic proteins exhibited distinct abundance "
    "levels in the $\\mathrm { P } _ { J 2 3 1 0 9 }$ strain and similarly, only 19 "
    "metabolic and 37 non-metabolic proteins had different abundances in the "
    "$\\mathrm { P } _ { \\mathrm { P P } \\_ 0 4 1 5 }$ strain (Supplementary Fig. 4A)."
)
_Q_THIRTEEN_COMMON = (
    "Only 13 proteins showed matching amplitudes of varying abundance (i.e., both up "
    "or down)."
)
_Q_PROMOTER_SWAP = (
    "The endogenous PP_0897 promoter was replaced using recombineering with either a "
    "low activity pJ23109 promoter from the Anderson collection31 or the low abundance "
    "PP_0415 promoter sequence identified in Eng et al.21 (Fig. 1D)."
)
_Q_EIGHTFOLD_DOWN = (
    "PP_0897 abundance was strongly reduced ${ \\sim } 8$ fold as expected in both "
    "PP_0897 promoter titration strains (Supplementary Fig. 4B)."
)
_Q_CUTSET = (
    "The complete intervention design (cutset) requires deletion of four genes: an "
    "α-ketoglutarate/3-oxoadipate permease PP_1378; and three class I and II fumarate "
    "hydratases—PP_0897, fumC1/PP_0944, and fumC2/PP_1755."
)
_Q_D1BGF = (
    "We started with the strain $P$ . putida KT2440 $P _ { B A D }$ -Sc.bpsA, Bc.sfp, "
    "Ps.glnA ΔPP_1378 ΔPP_1755 ΔPP_0944 ΔfleQ21, i.e. Design 1b glnA ΔfleQ (see strain "
    "table, Supplementary Table 1, abbreviated and hereafter referred to as “D1b_gf”)."
)
_Q_SI_D1BGF_ROW = (
    "P. putida KT2440PP_5402-intergenic::PBAD-Sc.bpsA,Bc.sfp,Ps.glnAΔPP_1378 ΔPP_1755 "
    "ΔPP_0944ΔfleQ/ΔPP_4373 serial passaging via ALE(\"D1b_gf')"
)
_Q_SI_DELETION_ROW = (
    "P. putida KT2440PP_5402-intergenic::PBAD-Sc.bpsA,Bc.sfp,Ps.glnAΔPP_1378 ΔPP_1755 "
    "ΔPP_0944∆fleQ/∆PP_4373 serial passaging via ALEΔPP_0897 (D1b_gf ΔPP_0897)"
)
_Q_SI_PJ_ROW = (
    "P. putida KT2440PP_5402-intergenic::PBAD-Sc.bpsA,Bc.sfp,Ps.glnAΔPP_1378 ΔPP_1755 "
    "ΔPP_0944ΔfleQ/ΔPP_4373 serial passaging via ALEPJ23109-PP_0897"
)
_Q_SI_0415_ROW = (
    "P. putida KT2440PP_5402-intergenic::PBAD-Sc.bpsA,Bc.sfp,Ps.glnAΔPP_1378 ΔPP_1755 "
    "ΔPP_0944ΔfleQ/∆PP_4373 serial passaging via ALEPpp_0415-PP_0897"
)
_Q_N_FOUR = "Proteomics analysis was done using four biological replicates."
_Q_N_FOUR_ERRORBARS = (
    "(C) RB-TnSeq fitness profiling of differentially changed proteins common between "
    "both PP_0897 promoter mutants. Error bars represent mean "
    "$\\pm \\ : \\mathrm { S . D }$ . $( \\mathrm { n } { = } 4 )$ in B."
)
_Q_AEROBIC = (
    "Aerobic conditions with $\\boldsymbol { p }$ -coumarate $\\dot { P }$ -CA), which "
    "is represented as Trans-4-Hydroxycinnamate (T4hcinnm) in the GSMM, was used as "
    "the sole carbon source to model growth in $\\boldsymbol { p }$ -CA minimal medium "
    "conditions."
)
_Q_PAM_COMPOSITION = (
    "We determined a media composition with $5 0 \\mathrm { m M }$ "
    "$\\boldsymbol { p }$ -CA, $7 0 ~ \\mathrm { m M }$ L-malate and "
    "$7 0 \\mathrm { m M }$ Dalanine was optimal for restoring growth of the strain "
    "D1b_gf ΔPP_0897"
)
_Q_DATA_AVAILABILITY = (
    "The generated mass spectrometry proteomics data have been deposited to the "
    "ProteomeXchange Consortium via the PRIDE partner repository33 with the dataset "
    "identifier PXD050285. Any additional information required to reanalyze the data "
    "reported in this working paper is available from the lead contact upon request."
)
_Q_TITER_UNIT = (
    "The indigoidine titer $\\mathrm { ( m g / L ) }$ ) was normalized to the total "
    "number of viable cells (CFUs/L)."
)
_Q_FIG2_REPLICATES = (
    "Measurements are average of at least three independent replicates. Error bars "
    "represent mean $\\pm \\ : \\mathrm { S . D }$ . $( n = 4 )$ in (C–E). The shaded "
    "area represents mean $\\pm \\ : \\mathrm { S . D } _ { }$ . "
    "$\\left( n = 3 \\right)$ in $( \\mathbf { F } , \\mathbf { G } )$ ."
)
_Q_PREDICTED_GROWTH_RATE = (
    "The predicted maximum specific growth rate was $1 . 1 4 / \\mathrm { h } ,$ and "
    "the maximum glutamine flux was $8 . 7 7 \\mathrm { m m o l / g D C W / h }$"
)
_Q_BIOLOG_DYE = (
    "These plates use a tetrazolium dye to detect respiratory products rather than "
    "bulk changes in biomass formation rate as a proxy for cellular growth64,65"
)
_Q_BIOLOG_TABLE_HEADER = "BIOLOG™M OD595"
_Q_PM2A_NARRATIVE = (
    "In the case of the PM2A BIOLOG TM plate, there were only two additional carbon "
    "sources where growth was observed: dextrin and D,L-carnitine."
)

M9_DEFERRED = _paper(
    "modified M9 minimal medium, salts deferred to reference 21",
    _Q_M9_DEFERRED,
    page=_METHODS_CULTIVATION,
    note="the medium object M9_DEFERRED_BANERJEE2025 carries this deferral; the three "
    "named analytes are the environment's carbon regime, not medium components",
)
PROTEOMICS_CULTURE = _paper(
    (CONDITION_PCA, CONDITION_PCA_ALA_MAL),
    _Q_PROTEOMICS_CULTURE,
    page=_METHODS_PROTEOMICS,
    note="the two media the shotgun-proteomics Methods name. The released columns mark "
    "the supplemented one with a '_M9_pCA_alanine_malate' or '_M9_alanine_malate' "
    "suffix and leave the first unsuffixed, which is what assigns each group its "
    "medium; the unsuffixed and the suffixed parental samples also differ numerically "
    "(PP_0897 log2 mean 24.914 against 26.254), so they are different cultures",
)
PAM_COMPOSITION = _paper(
    {"p-CA_mM": 50.0, "L-malate_mM": 70.0, "D-alanine_mM": 70.0},
    _Q_PAM_COMPOSITION,
    page=_RESULTS_SUPPLEMENTATION,
    note="the Results state the same three doses the Methods state, independently",
)
PCA_MM = _paper(60.0, _Q_PROTEOMICS_CULTURE, page=_METHODS_PROTEOMICS)
HARVEST_STATE = _paper(
    "mid-log phase, OD600 0.8-1",
    _Q_HARVEST,
    page=_METHODS_PROTEOMICS,
    note="an OD endpoint, not a clock time, which is why duration_hours is gapped",
)
INSTRUMENT = _paper("Orbitrap Exploris 480", _Q_INSTRUMENT, page=_METHODS_PROTEOMICS)
DIANN_DATABASE = _paper(
    "Uniprot P. putida KT2440 proteome plus heterologous pathway genes and common "
    "proteomic contaminants",
    _Q_DIANN_DB,
    page=_METHODS_PROTEOMICS,
    note="the contaminants are why some released labels resolve to no locus of the "
    "assembly: they were never host proteins",
)
GLOBAL_FDR = _paper(0.01, _Q_FDR, page=_METHODS_PROTEOMICS)
QUANTIFICATION = _paper(
    "label-free quantification (LFQ) of the DIA-NN peptide report",
    _Q_LFQ,
    page=_METHODS_PROTEOMICS,
    note="what the stored log2 abundance IS: a log2 LFQ quantity, replicate mean",
)
PROTEINS_QUANTIFIED = _paper(
    2000,
    _Q_TWO_THOUSAND,
    page=_RESULTS_CUTSET,
    note="the paper's approximate figure. The released workbooks hold 2,494 proteins "
    "per promoter comparison and 2,470 per cross-feeding comparison, which is what the "
    "records carry",
)
SIGNIFICANT_COUNTS = _paper(
    {WORKBOOK_PJ: 91, WORKBOOK_0415: 56},
    _Q_SIGNIFICANT_COUNTS,
    page=_RESULTS_CUTSET,
    note="28 + 63 = 91 and 19 + 37 = 56, which the two workbooks' significance sheets "
    "reproduce exactly (20 + 71 and 24 + 32); this is what identifies which workbook "
    "is which promoter variant",
)
COMMON_PROTEINS = _paper(
    EXPECTED_COMMON_PROTEINS,
    _Q_THIRTEEN_COMMON,
    page=_RESULTS_CUTSET,
    note="the curated 'Promoter STrainsAnalysis' sheet holds 91 + 56 - 13 = 134 rows "
    "and exactly 13 of them carry both arms' fold changes with matching sign",
)
PROMOTER_PARTS = _paper(
    ("pJ23109", f"{PP_0415} promoter"),
    _Q_PROMOTER_SWAP,
    page=_RESULTS_CUTSET,
    note="both parts replace the ENDOGENOUS PP_0897 promoter, so the gene stays "
    "present and unedited: a PromoterReplacementPerturbation, not a deletion",
)
PROMOTER_DIRECTION = _paper(
    "decreased",
    _Q_EIGHTFOLD_DOWN,
    page=_RESULTS_CUTSET,
    note="measured on the released workbooks: PP_0897 log2 fold change -4.042 "
    "(pJ23109) and -3.283 (PP_0415 promoter) against the parental sample, p below 1e-4 "
    "in both, so the direction is decreased for both variants",
)
RECOMBINEERING = _paper(
    "markerless recombineering",
    "In-frame genomic deletions and promoter substitutions were generated exactly as "
    "described in ref. 60.",
    page=_METHODS_RECOMBINEERING,
    note="why no cassette is recorded on the deletion or the promoter swaps",
)
CUTSET = _paper(
    tuple(tag for tag, _, _ in CUTSET_LOCI[:3]) + (PP_0897,),
    _Q_CUTSET,
    page=_RESULTS_CUTSET,
    note="the paper's fumC1 / fumC2 are the annotation's fumC-I (PP_0944) and fumC-II "
    "(PP_1755); PP_0897 is the fourth, which this study completed",
)
CHASSIS_GENOTYPE = _paper(
    CHASSIS,
    _Q_D1BGF,
    page=_RESULTS_CUTSET,
    note="PP_4373 is named for fleQ in Supplementary Table 1 and resolves to PP_4373 "
    "in the pinned annotation",
)
CHASSIS_SI_ROW = _si(
    _Q_SI_D1BGF_ROW,
    _Q_SI_D1BGF_ROW,
    page=_SI_TABLE1,
    note="strain JBEI_233,088, attributed to 'Eng andBanerjee,2023' (reference 21). "
    "The integrated indigoidine cassette is kept verbatim here rather than typed; see "
    "PATHWAY_NOT_TYPED",
)
STRAIN_SI_ROWS: dict[str, SourcedValue] = {
    STRAIN_PARENT: CHASSIS_SI_ROW,
    STRAIN_DELETION: _si(
        _Q_SI_DELETION_ROW, _Q_SI_DELETION_ROW, page=_SI_TABLE1, note="JBEI_256,069"
    ),
    STRAIN_PJ: _si(_Q_SI_PJ_ROW, _Q_SI_PJ_ROW, page=_SI_TABLE1, note="JBEI_236,231"),
    STRAIN_0415: _si(
        _Q_SI_0415_ROW,
        _Q_SI_0415_ROW,
        page=_SI_TABLE1,
        note="JBEI_236,232. The OCR places this row under the Supplementary Table 2 "
        "heading; it is a strain row of Supplementary Table 1 that the PDF's table "
        "break split, which is why its page is recorded as Table 1",
    ),
}
PROMOTER_REPLICATES = _si(
    ARM_REPLICATES[ARM_PROMOTER],
    _Q_N_FOUR,
    page=_SI_FIGURE4,
    note="back-solved independently from the released t statistic: every row of both "
    "promoter workbooks reproduces t = (m1 - m2) / sqrt((s1^2 + s2^2) / n) at n = 4",
)
PROMOTER_REPLICATES_ERRORBARS = _si(
    ARM_REPLICATES[ARM_PROMOTER], _Q_N_FOUR_ERRORBARS, page=_SI_FIGURE4
)
CROSSFEED_REPLICATES = _paper(
    ARM_REPLICATES[ARM_CROSSFEED],
    _Q_PROTEOMICS_CULTURE,
    page=_METHODS_PROTEOMICS,
    note="'grown in triplicates'. Back-solved independently from the released t "
    "statistic: every row of both cross-feeding workbooks reproduces t at n = 3. The "
    "apparent conflict with Supplementary Figure 4's n = 4 is an ARM difference, not a "
    "contradiction: the Methods sentence describes the cross-feeding cultures and the "
    "figure caption the promoter cultures",
)
AEROBICITY = _paper(
    "aerobic",
    _Q_AEROBIC,
    page=_METHODS_MODELING,
    note="the culture Methods never state an oxygen regime; the modeling section "
    "states that the experimental scenario it simulates is aerobic, and the cultures "
    "are 10 mL tubes",
)
DATA_AVAILABILITY = _paper(
    PRIDE_ACCESSION,
    _Q_DATA_AVAILABILITY,
    page="Data availability",
    note="the raw DIA files. No loader consumes raw spectra, so the accession is "
    "retrieval metadata and Supplementary Data S2 is the released processed release",
)

#: Why no titer family is written, with the quote that makes the unit question moot.
TITER_NOT_A_DATASET = _paper(
    "no released indigoidine titer table",
    _Q_TITER_UNIT,
    page=_METHODS_INDIGOIDINE,
    note="the titer is defined and plotted (Figs. 2C-2E, Supplementary Figs. 2, 3A, 8) "
    "but never released as a number: Supplementary Data S1 is the COBRA model and flux "
    "distributions, Supplementary Data S2 is this proteomics release, and Supplementary "
    "Tables 1 to 4 are strains, plasmids, oligos and the BIOLOG grid. "
    "ProductTiterExperimentReference requires a phenotype_reference and "
    "ProductTiterPhenotype.titer is a required float, so the family is refused whole, "
    "as the sibling Yunus 2026 and Kang 2026 loaders refuse theirs",
)
#: Why no growth family is written: the only per-hour numbers are predictions.
GROWTH_NOT_A_DATASET = _paper(
    "no released growth-rate table",
    _Q_PREDICTED_GROWTH_RATE,
    page=_RESULTS_CONTEXT,
    note="1.14/h, 0.43/h and 0.45/h are flux-balance PREDICTIONS of the "
    "context-specific models. Every measured growth readout in the paper is a plate "
    "image or a growth curve (Figs. 2B, 2F, 2G, Supplementary Figs. 1, 6, 7, 8) with "
    f"no companion table; its replicate design is stated ({_Q_FIG2_REPLICATES!r}) but "
    "a design is not a measurement",
)
#: Why the released BIOLOG grid is a finding rather than a second dataset class.
BIOLOG_NOT_A_DATASET = _si(
    95,
    _Q_BIOLOG_TABLE_HEADER,
    page=_SI_TABLE4,
    note="Supplementary Table 4 releases a measured OD595 for 95 PM1-plate carbon "
    "sources on strain D1b_gf ΔPP_0897 beside the matched in-silico growth value. The "
    "readout is a tetrazolium redox-dye endpoint, not biomass "
    f"({_Q_BIOLOG_DYE!r}), and its values are signed, so FitnessPhenotype (which "
    "clamps non-positive values) is wrong and EnvironmentResponsePhenotype is the "
    "right shape. It is NOT written here because MeasurementType carries no member "
    "for a respiration endpoint and adding one would change the serialized closure of "
    "every served EnvironmentResponsePhenotype dataset, forcing a full "
    "knowledge-graph rebuild. The PM2A plate's OD values were never released, only "
    f"the narrative {_Q_PM2A_NARRATIVE!r}",
)
#: Why the three heterologous genes of the chassis cassette are not typed.
PATHWAY_NOT_TYPED = _paper(
    ("Sc.bpsA", "Bc.sfp", "Ps.glnA"),
    _Q_D1BGF,
    page=_RESULTS_CUTSET,
    note="GeneAdditionPerturbation requires source_organism with no default, and this "
    "article and its SI never expand 'Sc.', 'Bc.' or 'Ps.' ('Bacillus subtilis' occurs "
    "only in an unrelated reference title). The organisms are stated in the earlier "
    "indigoidine papers (Banerjee 2020 and reference 21), neither of which is in the "
    "literature mirror, so expanding them is a deferral to an unmirrored source. The "
    "cassette is kept verbatim in the background's genotype_statement instead",
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/banerjeeAddressingGenomeScale2025``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieval_record(
    member: str, sha256: str, retrieved_at: str = SI_RETRIEVED_AT
) -> RetrievalRecord:
    """The versioned retrieval of one member: ``zip_member`` over the PMC archive."""
    return RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=ARCHIVE_URL,
        retriever="torchcell.literature.retrieve.zip_member",
        params={
            "url": ARCHIVE_URL,
            "key": f"{PMC_PREFIX}/{ARCHIVE_FILENAME}",
            "member": member,
            "container_sha256": ARCHIVE_SHA256,
        },
        sha256=sha256,
        retrieved_at=retrieved_at,
    )


def deposit_raw_mirror(
    *,
    archive_path: str | Path,
    retrieved_at: str = SI_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the five consumed members of Supplementary Data S2 + ``manifest.json``.

    The archive must carry :data:`ARCHIVE_SHA256` and each member its pinned sha256.
    Idempotent by sha256: an existing mirror file with the recorded hash is left alone
    and a differing one raises rather than being overwritten. Every member is verified
    BEFORE anything is written, so a refusal leaves no partial deposit.
    """
    verify_sha256(archive_path, ARCHIVE_SHA256)
    root = raw_mirror_dir(data_root)
    files: list[ArtifactRecord] = []
    with zipfile.ZipFile(archive_path) as archive:
        payloads = {
            relpath: archive.read(member)
            for relpath, (member, _) in RAW_MEMBERS.items()
        }
    for relpath, (member, expected) in RAW_MEMBERS.items():
        got = hashlib.sha256(payloads[relpath]).hexdigest()
        if got != expected:
            raise RuntimeError(
                f"{member} sha256 mismatch: got {got}, expected {expected}"
            )
    for relpath, (member, expected) in RAW_MEMBERS.items():
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != expected:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            tmp = dest.with_suffix(dest.suffix + ".partial")
            tmp.write_bytes(payloads[relpath])
            shutil.move(str(tmp), str(dest))
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_SI_DATA,
                bytes=dest.stat().st_size,
                sha256=expected,
                source=ARCHIVE_URL,
                retrieval=retrieval_record(member, expected, retrieved_at),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=files,
        si_data_sources=[
            ARCHIVE_URL,
            "https://pmc-oa-opendata.s3.amazonaws.com/"
            f"{PMC_PREFIX}/{MODEL_ARCHIVE_FILENAME}",
            f"https://www.ebi.ac.uk/pride/archive/projects/{PRIDE_ACCESSION}",
        ],
        si_expected=[
            f"Supplementary Data S2 ({ARCHIVE_FILENAME}) -- the four t-test workbooks "
            "and the curated DataAnalysis workbook are deposited as members; the "
            "archive's volcano plots and heatmap PNGs are renderings of those same "
            "numbers and are not deposited",
            f"Supplementary Data S1 ({MODEL_ARCHIVE_FILENAME}, sha256 "
            f"{MODEL_ARCHIVE_SHA256}) -- NOT deposited: it is the COBRA Toolbox code, "
            "two MATLAB workspaces and the simulated flux distributions, and no "
            "measured phenotype is in it",
            f"Supplementary Information ({SI1_PDF}) -- NOT deposited here: it is a "
            "text PDF already mirrored and OCR'd in torchcell-library, and the values "
            "this loader takes from it are sourced constants quoting those bytes",
            f"PRIDE {PRIDE_ACCESSION} -- the raw DIA mass-spectrometry files. NOT "
            "deposited: no loader consumes raw spectra, and Supplementary Data S2 is "
            "the released processed release",
            "the indigoidine titers and the growth curves were never released as "
            "tables; see TITER_NOT_A_DATASET and GROWTH_NOT_A_DATASET",
            "Supplementary Table 4's BIOLOG OD595 grid is released, measured and not "
            "ingested; see BIOLOG_NOT_A_DATASET",
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


def _link_mirror_files(raw_dir: str, pins: Iterable[tuple[str, str, str]]) -> None:
    """Link each pinned mirror file into ``raw/`` after checking it against the manifest."""
    data_root = _data_root()
    manifest = load_manifest(data_root)
    os.makedirs(raw_dir, exist_ok=True)
    for relpath, filename, expected in pins:
        check_manifest_pin(relpath, manifest_sha256(manifest, relpath), expected)
        src = raw_mirror_dir(data_root) / relpath
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        link_verified(src, osp.join(raw_dir, filename), expected)


# --------------------------------------------------------------------------- #
# Released-file readers
# --------------------------------------------------------------------------- #
def canonical_label(label: str) -> str:
    """The stored spelling of one released ``Protein`` label.

    A locus-tag-shaped label is upper-cased and everything else is left verbatim. The
    four workbooks disagree on the case of exactly the locus-tag-shaped labels (1,376
    of them between the two cross-feeding workbooks, and no other difference), so
    without this the same protein is two keys and the two workbooks cannot be joined.
    Upper-casing also keeps the six collision-held tags inside the namespace.
    """
    stripped = label.strip()
    if LOCUS_LABEL_RE.fullmatch(stripped):
        return stripped.upper()
    return stripped


class ComparisonRow(BaseModel):
    """One protein row of a released workbook: both groups plus its released statistics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    accession: str
    description: str | None
    a_mean: float
    b_mean: float
    a_sd: float
    b_sd: float
    t_statistic: float
    p_value: float
    log2_fold_change: float


def _header_index(header: Sequence[Any], name: str, path: str) -> int:
    """Column index of ``name`` in a released header, refusing an absent column."""
    for index, cell in enumerate(header):
        if cell == name:
            return index
    raise RuntimeError(f"{path}: released header carries no {name!r} column")


def read_comparison(path: str, group_a: str, group_b: str) -> list[ComparisonRow]:
    """Every protein row of one workbook's ``Full t-test output`` sheet.

    ``group_a`` and ``group_b`` are the released group keys, in the order the workbook's
    ``log2_Fold_change_A/B`` subtracts them. A row whose statistics are not all present
    is still returned; the callers that need completeness say so themselves.
    """
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book[FULL_SHEET]
        stream = sheet.iter_rows(values_only=True)
        header = next(stream)
        columns = {
            "label": _header_index(header, "Protein", path),
            "accession": _header_index(header, "Protein.Group", path),
            "description": _header_index(header, "Protein.Description", path),
            "a_mean": _header_index(header, f"log2_mean_{group_a}", path),
            "b_mean": _header_index(header, f"log2_mean_{group_b}", path),
            "a_sd": _header_index(header, f"log2_std_{group_a}", path),
            "b_sd": _header_index(header, f"log2_std_{group_b}", path),
            "t_statistic": _header_index(header, "t-test_stat", path),
            "p_value": _header_index(header, "p-value", path),
            "log2_fold_change": _header_index(header, "log2_Fold_change_A/B", path),
        }
        rows: list[ComparisonRow] = []
        for row in stream:
            if row[columns["label"]] is None:
                continue
            values = {name: row[index] for name, index in columns.items()}
            if any(
                values[name] is None
                for name in ("a_mean", "b_mean", "a_sd", "b_sd", "accession")
            ):
                raise RuntimeError(
                    f"{path}: protein {values['label']!r} is missing a released "
                    "abundance or accession; a partial row would silently shrink a "
                    "sample"
                )
            rows.append(
                ComparisonRow(
                    label=canonical_label(str(values["label"])),
                    accession=str(values["accession"]),
                    description=(
                        None
                        if values["description"] is None
                        else str(values["description"])
                    ),
                    a_mean=float(values["a_mean"]),
                    b_mean=float(values["b_mean"]),
                    a_sd=float(values["a_sd"]),
                    b_sd=float(values["b_sd"]),
                    t_statistic=float(values["t_statistic"]),
                    p_value=float(values["p_value"]),
                    log2_fold_change=float(values["log2_fold_change"]),
                )
            )
    finally:
        book.close()
    if not rows:
        raise RuntimeError(f"{path}: the released sheet holds no protein row")
    return rows


#: ``workbook -> (group A, group B)`` in the order the workbook subtracts them.
WORKBOOK_GROUPS: dict[str, tuple[str, str]] = {
    WORKBOOK_PJ: ("2370_pJ_PP_0897", "2370"),
    WORKBOOK_0415: ("2370_0415pPP_0897", "2370"),
    WORKBOOK_PAM_STRAINS: ("2487_M9_pCA_alanine_malate", "2370_M9_pCA_alanine_malate"),
    WORKBOOK_PAM_MEDIA: ("2487_M9_pCA_alanine_malate", "2487_M9_alanine_malate"),
}


def read_workbooks(raw_dir: str) -> dict[str, list[ComparisonRow]]:
    """Every released workbook's rows, keyed by workbook filename."""
    return {
        workbook: read_comparison(osp.join(raw_dir, workbook), *groups)
        for workbook, groups in WORKBOOK_GROUPS.items()
    }


def group_cells(
    rows: Sequence[ComparisonRow], group: ProteomeGroup
) -> dict[str, tuple[float, float]]:
    """``label -> (log2 mean, log2 SD)`` for one group of one workbook."""
    a_key, _ = WORKBOOK_GROUPS[group.workbook]
    first = group.key == a_key
    cells: dict[str, tuple[float, float]] = {}
    for row in rows:
        value = (row.a_mean, row.a_sd) if first else (row.b_mean, row.b_sd)
        if row.label in cells and cells[row.label] != value:
            raise RuntimeError(
                f"{group.key}/{row.label}: two released rows disagree on the stored "
                "abundance; a repeated label would change the stored mean"
            )
        cells[row.label] = value
    return cells


# --------------------------------------------------------------------------- #
# Build oracles
# --------------------------------------------------------------------------- #
def assert_protein_counts(workbooks: dict[str, list[ComparisonRow]]) -> None:
    """Each workbook holds the protein count the pinned bytes held when this landed."""
    got = {name: len(rows) for name, rows in workbooks.items()}
    if got != EXPECTED_PROTEIN_ROWS:
        raise RuntimeError(
            f"released protein counts changed: {got} against {EXPECTED_PROTEIN_ROWS}"
        )


def assert_shared_groups_agree(workbooks: dict[str, list[ComparisonRow]]) -> None:
    """The two groups released twice are equal cell for cell across their workbooks.

    This is what makes the record count six rather than eight: a group released in two
    workbooks is ONE sample. A revision that makes the pair differ means the two
    comparisons no longer share a control, so the build stops.
    """
    for key, left, right in SHARED_GROUPS:
        group_left = next(
            g for g in PROTEOME_GROUPS if g.key == key and g.workbook == left
        )
        group_right = ProteomeGroup(
            key=key,
            strain=group_left.strain,
            arm=group_left.arm,
            condition=group_left.condition,
            workbook=right,
        )
        cells_left = group_cells(workbooks[left], group_left)
        cells_right = group_cells(workbooks[right], group_right)
        if set(cells_left) != set(cells_right):
            raise RuntimeError(
                f"{key}: {left} and {right} release different protein sets for the "
                "same sample"
            )
        differing = [
            label for label in cells_left if cells_left[label] != cells_right[label]
        ]
        if differing:
            raise RuntimeError(
                f"{key}: {len(differing)} cells differ between {left} and {right} "
                f"(first {differing[:5]}); the two comparisons no longer share one "
                "sample"
            )


def back_solved_replicates(row: ComparisonRow) -> float | None:
    """The ``n`` that reproduces one row's released t statistic, or ``None``.

    ``t = (mA - mB) / sqrt((sA^2 + sB^2) / n)``, so ``n`` is recoverable in closed form
    whenever the difference and the released t are both non-zero.
    """
    spread = row.a_sd**2 + row.b_sd**2
    difference = row.a_mean - row.b_mean
    if spread == 0.0 or difference == 0.0 or row.t_statistic == 0.0:
        return None
    return spread * (row.t_statistic / difference) ** 2


def merged_accession_labels(workbooks: dict[str, list[ComparisonRow]]) -> set[str]:
    """Released labels that file more than one protein group under one name."""
    accessions: dict[str, set[str]] = defaultdict(set)
    for rows in workbooks.values():
        for row in rows:
            accessions[row.label].add(row.accession)
    return {label for label, group in accessions.items() if len(group) > 1}


def assert_replicates_back_solve(
    workbooks: dict[str, list[ComparisonRow]],
) -> dict[str, int]:
    """Every released t statistic reproduces its arm's replicate count.

    Returns ``workbook -> rows checked``. The quoted counts (four for the promoter arm,
    three for the cross-feeding arm) are each confirmed here against the arithmetic of
    the released numbers, which is what resolves the source's two different statements.

    The merged-accession labels are the ONE exception and they are asserted, not
    excused: :data:`MERGED_LABEL_REPLICATE_FACTOR` times the arm's count, because the
    released analysis pooled both protein groups' replicates under the one label. Those
    are exactly the labels the merge rule drops, so the only rows whose replicate count
    does not hold are the rows that do not become stored values.
    """
    merged = merged_accession_labels(workbooks)
    checked: dict[str, int] = {}
    for workbook, rows in workbooks.items():
        group_a = WORKBOOK_GROUPS[workbook][0]
        arm = next(g.arm for g in PROTEOME_GROUPS if g.key == group_a)
        base = float(ARM_REPLICATES[arm])
        count = 0
        for row in rows:
            solved = back_solved_replicates(row)
            if solved is None:
                continue
            count += 1
            expected = (
                base * MERGED_LABEL_REPLICATE_FACTOR if row.label in merged else base
            )
            if abs(solved - expected) > T_STATISTIC_TOL * max(expected, 1.0):
                raise RuntimeError(
                    f"{workbook}/{row.label}: the released t statistic back-solves to "
                    f"n = {solved!r}, not the {expected} expected for the {arm} arm"
                    + (" (a merged-accession label)" if row.label in merged else "")
                )
        if count == 0:
            raise RuntimeError(f"{workbook}: no row could back-solve a replicate count")
        checked[workbook] = count
    return checked


def assert_pp_0897_identifies_the_strains(
    workbooks: dict[str, list[ComparisonRow]],
) -> dict[str, float]:
    """PP_0897's own released fold changes identify which strain each group is.

    Returns ``workbook -> PP_0897 log2 fold change``. The three facts asserted are the
    ones the module docstring states: PP_0897 is significantly DOWN in both promoter
    comparisons, far further down in the deletion comparison, and FLAT between the two
    media of the deletion strain, which is what a deleted gene must do.
    """
    folds: dict[str, float] = {}
    for workbook, rows in workbooks.items():
        hits = [row for row in rows if row.label == PP_0897]
        if len(hits) != 1:
            raise RuntimeError(
                f"{workbook}: {len(hits)} rows for {PP_0897}, which is the locus every "
                "strain assignment is read from"
            )
        folds[workbook] = hits[0].log2_fold_change
    for workbook in (WORKBOOK_PJ, WORKBOOK_0415):
        if not folds[workbook] < -2.0:
            raise RuntimeError(
                f"{workbook}: {PP_0897} log2 fold change {folds[workbook]} is not the "
                "strong reduction the promoter titration states"
            )
    deletion = folds[WORKBOOK_PAM_STRAINS]
    if not deletion < min(folds[WORKBOOK_PJ], folds[WORKBOOK_0415]):
        raise RuntimeError(
            f"{WORKBOOK_PAM_STRAINS}: {PP_0897} is only {deletion} below the parent, "
            "which is a titration rather than the deletion this group is assigned"
        )
    flat = folds[WORKBOOK_PAM_MEDIA]
    if abs(flat) > 0.5:
        raise RuntimeError(
            f"{WORKBOOK_PAM_MEDIA}: {PP_0897} changes by {flat} between the two media "
            "of one strain, which a deleted gene cannot do"
        )
    return folds


def released_significant_counts(raw_dir: str) -> dict[str, int]:
    """Rows of the two significance sheets of each promoter workbook."""
    counts: dict[str, int] = {}
    for workbook in EXPECTED_SIGNIFICANT:
        book = openpyxl.load_workbook(
            osp.join(raw_dir, workbook), read_only=True, data_only=True
        )
        try:
            total = 0
            for sheet_name in SIGNIFICANCE_SHEETS:
                sheet = book[sheet_name]
                stream = sheet.iter_rows(values_only=True)
                next(stream)
                total += sum(1 for row in stream if row[0] is not None)
            counts[workbook] = total
        finally:
            book.close()
    return counts


def assert_significant_counts_match_the_text(raw_dir: str) -> dict[str, int]:
    """The two promoter workbooks hold the counts the Results text states.

    This is the oracle that says WHICH workbook is WHICH promoter variant: the paper
    gives 28 + 63 for the PJ23109 strain and 19 + 37 for the PP_0415 strain, and only
    one assignment of the two workbooks reproduces both.
    """
    counts = released_significant_counts(raw_dir)
    if counts != EXPECTED_SIGNIFICANT:
        raise RuntimeError(
            f"released significance-sheet counts {counts} no longer reproduce the "
            f"Results text's {EXPECTED_SIGNIFICANT}"
        )
    return counts


def assert_curated_overlap(raw_dir: str) -> tuple[int, int]:
    """The curated sheet holds 134 rows of which exactly 13 carry both fold changes.

    91 + 56 - 13 = 134 reproduces the paper's own arithmetic, and the 13 are its "Only
    13 proteins showed matching amplitudes of varying abundance (i.e., both up or
    down)": every one of them is checked to have matching sign.
    """
    book = openpyxl.load_workbook(
        osp.join(raw_dir, WORKBOOK_ANALYSIS), read_only=True, data_only=True
    )
    try:
        sheet = book[ANALYSIS_SHEET]
        stream = sheet.iter_rows(values_only=True)
        header = next(stream)
        pj = _header_index(header, "log2FC pJ23109", WORKBOOK_ANALYSIS)
        weak = _header_index(header, "log2FC PP_0415p", WORKBOOK_ANALYSIS)
        rows = [row for row in stream if row[0] is not None]
    finally:
        book.close()
    both = [row for row in rows if row[pj] is not None and row[weak] is not None]
    mismatched = [row for row in both if (float(row[pj]) > 0) != (float(row[weak]) > 0)]
    if len(rows) != EXPECTED_ANALYSIS_ROWS:
        raise RuntimeError(
            f"{ANALYSIS_SHEET}: {len(rows)} rows, not the "
            f"{EXPECTED_ANALYSIS_ROWS} the paper's 91 + 56 - 13 implies"
        )
    if len(both) != EXPECTED_COMMON_PROTEINS:
        raise RuntimeError(
            f"{ANALYSIS_SHEET}: {len(both)} proteins carry both arms' fold changes, "
            f"not the {EXPECTED_COMMON_PROTEINS} the Results text states"
        )
    if mismatched:
        raise RuntimeError(
            f"{ANALYSIS_SHEET}: {len(mismatched)} of the common proteins change in "
            "OPPOSITE directions, against the text's 'both up or down'"
        )
    return len(rows), len(both)


# --------------------------------------------------------------------------- #
# Schema construction
# --------------------------------------------------------------------------- #
def publication() -> Publication:
    """This paper, by DOI. The mirror records no PubMed id, so none is invented."""
    return Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")


def chassis_background() -> BacterialStrainBackground:
    """D1b_gf: the four host-gene deletions every record inherits from reference 21.

    The integrated indigoidine cassette is in ``genotype_statement`` verbatim and NOT
    among the alleles: ``BacterialBackgroundAllele`` is an edit of a HOST locus, and
    this cassette sits in an intergenic site with three heterologous genes whose source
    organisms the paper never states (:data:`PATHWAY_NOT_TYPED`).
    """
    return BacterialStrainBackground(
        name=CHASSIS,
        reference_strain=WT_STRAIN,
        assembly_set=KT2440_ASSEMBLY_SET,
        parents=[WT_STRAIN],
        construction=(
            "four markerless in-frame deletions in KT2440 plus an indigoidine "
            "production cassette integrated at the PP_5402 intergenic site, then serial "
            "passaging via adaptive laboratory evolution; made in reference 21, not in "
            "this study"
        ),
        genotype_statement=str(CHASSIS_SI_ROW.value),
        alleles=[
            BacterialBackgroundAllele(
                systematic_gene_name=tag,
                gene_namespace=KT2440_NAMESPACE,
                gene_name=symbol,
                allele_name=allele,
                edit=AlleleEdit.full_deletion,
                functional=False,
                provenance=[CHASSIS_GENOTYPE, CHASSIS_SI_ROW, CUTSET]
                if tag != "PP_4373"
                else [CHASSIS_GENOTYPE, CHASSIS_SI_ROW],
            )
            for tag, symbol, allele in CUTSET_LOCI
        ],
        provenance=[CHASSIS_GENOTYPE, CHASSIS_SI_ROW, RECOMBINEERING],
    )


def record_reference_genome() -> AssemblyReferenceGenome:
    """KT2440 carrying the D1b_gf background: the reference of every record.

    One object for all six records, because the four strains differ from each other by
    their OWN PP_0897 edit, which is a perturbation in the genotype, not a background.
    """
    return assembly_reference(WT_STRAIN, background=chassis_background())


def promoter_perturbation(strain: str) -> PromoterReplacementPerturbation:
    """One promoter-variant strain's PP_0897 swap."""
    parts = PROMOTER_PARTS.value
    if not isinstance(parts, tuple):
        raise RuntimeError(f"PROMOTER_PARTS.value is not a pair: {parts!r}")
    if strain == STRAIN_PJ:
        promoter = str(parts[0])
    elif strain == STRAIN_0415:
        promoter = str(parts[1])
    else:
        raise RuntimeError(f"{strain!r} is not a promoter-variant strain of this paper")
    return PromoterReplacementPerturbation(
        systematic_gene_name=PP_0897,
        perturbed_gene_name=PP_0897,
        gene_namespace=KT2440_NAMESPACE,
        promoter_name=promoter,
        native_promoter="endogenous PP_0897 promoter",
        expression_direction=str(PROMOTER_DIRECTION.value),
        is_inducible=False,
    )


def deletion_perturbation() -> BacterialDeletionPerturbation:
    """The PP_0897 deletion that completes the cutset."""
    return BacterialDeletionPerturbation(
        systematic_gene_name=PP_0897,
        perturbed_gene_name=PP_0897,
        gene_namespace=KT2440_NAMESPACE,
    )


def strain_genotype(strain: str) -> Genotype:
    """One released strain's own edit on top of the D1b_gf chassis.

    The parental samples carry NO perturbation: everything they have is the background,
    and the three heterologous cassette genes are deliberately untyped
    (:data:`PATHWAY_NOT_TYPED`).
    """
    if strain == STRAIN_PARENT:
        return Genotype(perturbations=[])
    if strain == STRAIN_DELETION:
        return Genotype(perturbations=[deletion_perturbation()])
    return Genotype(perturbations=[promoter_perturbation(strain)])


#: Why the one undescribed medium's two substrates carry no dose.
UNDOSED_MEDIUM_NOTE = (
    "the released column 'log2_mean_2487_M9_alanine_malate' names a medium the Methods "
    "never describe: they name only the two media PROTEOMICS_CULTURE quotes, both of "
    "which contain p-CA. The substrate is therefore carried with no magnitude rather "
    "than borrowing the 70 mM dose stated for the p-CA-containing medium"
)


def _carbon(name: str, value: float | None) -> EnvironmentPhysicalPerturbation:
    """One carbon source of the carbon-free base M9, typed as media.py names it.

    ``M9_DEFERRED_BANERJEE2025`` carries no carbon source and its own provenance note
    says the loader carries the regime here, so each condition's substrates are the
    environment's variable rather than a medium object per condition. ``value`` is
    ``None`` for the one condition whose doses the source never states, and the
    perturbation then carries a typed gap on its own ``magnitude``.
    """
    return EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.carbon_source,
        magnitude=(
            None
            if value is None
            else Concentration(value=value, unit=ConcentrationUnit.millimolar)
        ),
        agent=resolved_compound(name),
        provenance_gaps=(
            []
            if value is not None
            else [
                ProvenanceGap(
                    field="magnitude",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note=UNDOSED_MEDIUM_NOTE,
                )
            ]
        ),
    )


#: The source label each carbon source is resolved under. ``p-coumaric acid`` is the
#: compound table's row for the paper's ``p-CA``, which is the reading Lim 2022 records
#: for its own "coumarate"; the other two have no curated row yet and carry typed gaps.
PCA_COMPOUND = "p-coumaric acid"
ALANINE_COMPOUND = "D-alanine"
MALATE_COMPOUND = "L-malate"


def proteome_environment(condition: str) -> Environment:
    """The M9 environment of one released proteomics condition.

    ``temperature`` is ``None`` with a gap: the shotgun-proteomics Methods state the
    medium, the vessel, the inoculum and the harvest state but never an incubation
    temperature. ``duration_hours`` is ``None`` with a gap too, because the cultures
    were harvested at a growth STATE rather than at a clock time.
    """
    doses = PAM_COMPOSITION.value
    if not isinstance(doses, dict):
        raise RuntimeError(f"PAM_COMPOSITION.value is not a mapping: {doses!r}")
    carbon: list[EnvironmentPerturbationType]
    if condition == CONDITION_PCA:
        carbon = [_carbon(PCA_COMPOUND, float(PCA_MM.value))]
    elif condition == CONDITION_PCA_ALA_MAL:
        carbon = [
            _carbon(PCA_COMPOUND, float(doses["p-CA_mM"])),
            _carbon(ALANINE_COMPOUND, float(doses["D-alanine_mM"])),
            _carbon(MALATE_COMPOUND, float(doses["L-malate_mM"])),
        ]
    elif condition == CONDITION_ALA_MAL:
        carbon = [_carbon(ALANINE_COMPOUND, None), _carbon(MALATE_COMPOUND, None)]
    else:
        raise RuntimeError(f"{condition!r} is not a released proteomics condition")
    gaps = [
        ProvenanceGap(
            field="temperature",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=(
                "the shotgun-proteomics Methods state no incubation temperature. Every "
                "temperature this paper does state elsewhere is 30 C, which is why the "
                "absence is recorded rather than filled in from a neighboring section"
            ),
        ),
        ProvenanceGap(
            field="duration_hours",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=(
                "the cultures were harvested at a growth state "
                f"({HARVEST_STATE.value}), so no clock time describes them"
            ),
        ),
    ]
    return Environment(
        media=M9_DEFERRED_BANERJEE2025,
        temperature=None,
        perturbations=carbon,
        aerobicity=str(AEROBICITY.value),
        duration_hours=None,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# Build accounting
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One reason a candidate did not become a record, with the items it removed."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    scope: str
    description: str
    n_records: int
    items: list[str]


class BuildAccounting(BaseModel):
    """Source rows in, records out, and every rule between, written to ``preprocess/``."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    source_rows: int
    candidate_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]
    reconciliation: LocusTagReconciliation
    notes: list[str]

    def check(self) -> None:
        """Candidates equal keeps plus drops, and each rule's count is non-negative."""
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_records} "
                f"dropped != {self.candidate_records} candidates"
            )
        for rule in self.rules:
            if rule.n_records < 0:
                raise RuntimeError(f"{rule.rule}: negative record count")


def _write_accounting(accounting: BuildAccounting, preprocess_dir: str) -> None:
    """Validate and write ``preprocess/build_accounting.json``."""
    accounting.check()
    os.makedirs(preprocess_dir, exist_ok=True)
    Path(preprocess_dir, "build_accounting.json").write_text(
        accounting.model_dump_json(indent=2)
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class ProteomeBanerjee2025Dataset(ExperimentDataset):
    """Banerjee 2025 DIA proteome: six released samples of the p-CA cutset strains.

    One record per released (strain, medium) sample. The two arms differ in their
    replicate count (four against three, both quoted and both back-solved) and in their
    released protein set, so each arm's records are written against that arm's own
    parental sample; ``measurement_type`` is one string for the whole dataset, which is
    what the shared ``verify_protein_dataset`` requires.
    """

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"

    def __init__(
        self,
        root: str = "data/torchcell/proteome_banerjee2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome resolves the released protein labels."""
        self.pputida_genome = pputida_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialProteinAbundanceExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The four t-test workbooks plus the curated analysis workbook."""
        return [osp.basename(relpath) for relpath in RAW_MEMBERS]

    def download(self) -> None:
        """Link every mirror member into ``raw/`` after verifying it against its pin."""
        _link_mirror_files(
            self.raw_dir,
            tuple(
                (relpath, osp.basename(relpath), sha256)
                for relpath, (_, sha256) in RAW_MEMBERS.items()
            ),
        )
        log.info("Banerjee 2025 Data S2 members linked into %s", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    @staticmethod
    def _phenotype(
        cells: dict[str, tuple[float, float]], n_replicates: int
    ) -> ProteinAbundancePhenotype:
        """One sample's abundances with SE = released log2 SD / sqrt(n).

        The released SD is the sample SD over this arm's replicates on the log2 scale,
        so SD / sqrt(n) is the standard error of the log2 mean on the same scale, which
        is the quantity the record stores.
        """
        if not cells:
            raise RuntimeError("a proteome sample carries no measured protein")
        root_n = math.sqrt(n_replicates)
        return ProteinAbundancePhenotype(
            protein_abundance={tag: mean for tag, (mean, _) in cells.items()},
            protein_abundance_se={tag: sd / root_n for tag, (_, sd) in cells.items()},
            n_replicates=dict.fromkeys(cells, n_replicates),
            measurement_type=MEASUREMENT_TYPE,
        )

    @post_process
    def process(self) -> None:
        """Build one record per released proteome sample."""
        pins = {
            osp.basename(relpath): sha256
            for relpath, (_, sha256) in RAW_MEMBERS.items()
        }
        verify_raw_files(self.raw_dir, pins)
        workbooks = read_workbooks(self.raw_dir)
        assert_protein_counts(workbooks)
        assert_shared_groups_agree(workbooks)
        back_solved = assert_replicates_back_solve(workbooks)
        pp_0897_folds = assert_pp_0897_identifies_the_strains(workbooks)
        significant = assert_significant_counts_match_the_text(self.raw_dir)
        analysis_rows, common_proteins = assert_curated_overlap(self.raw_dir)
        genome = self._genome()

        labels = sorted({row.label for rows in workbooks.values() for row in rows})
        stored, report = reconcile_locus_tags(
            genome, pd.Series(labels), label=self.name
        )
        report.require_resolved(MIN_RESOLVED_FRACTION)
        stored_by_label = dict(zip(labels, stored, strict=True))

        accessions: dict[str, set[str]] = defaultdict(set)
        descriptions: dict[str, str] = {}
        for rows in workbooks.values():
            for row in rows:
                accessions[row.label].add(row.accession)
                if row.description is not None:
                    descriptions.setdefault(row.label, row.description)
        outside = set(report.outside_namespace)
        merged = merged_accession_labels(workbooks)
        dropped_labels = sorted(
            {label for label in labels if stored_by_label[label] in outside} | merged
        )
        kept_labels = [label for label in labels if label not in set(dropped_labels)]
        if not kept_labels:
            raise RuntimeError(f"{self.name}: every protein label was dropped")

        samples: dict[str, dict[str, tuple[float, float]]] = {}
        for group in PROTEOME_GROUPS:
            cells = group_cells(workbooks[group.workbook], group)
            samples[group.key] = {
                stored_by_label[label]: value
                for label, value in cells.items()
                if label not in set(dropped_labels)
            }
        if len(samples) != EXPECTED_PROTEOME_RECORDS:
            raise RuntimeError(
                f"{self.name}: {len(samples)} samples, not the "
                f"{EXPECTED_PROTEOME_RECORDS} the pinned workbooks hold"
            )

        reference_genome = record_reference_genome()
        baselines = {
            arm: self._phenotype(samples[key], ARM_REPLICATES[arm])
            for arm, key in ARM_REFERENCE_KEY.items()
        }
        references = {
            arm: BacterialProteinAbundanceExperimentReference(
                dataset_name=self.name,
                genome_reference=reference_genome,
                environment_reference=proteome_environment(
                    next(
                        g.condition
                        for g in PROTEOME_GROUPS
                        if g.key == ARM_REFERENCE_KEY[arm]
                    )
                ),
                phenotype_reference=baselines[arm],
            )
            for arm in ARM_REFERENCE_KEY
        }
        pub = publication()

        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        sample_rows: list[dict[str, Any]] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for group in tqdm(PROTEOME_GROUPS, desc="banerjee2025-proteome"):
                cells = samples[group.key]
                n_replicates = ARM_REPLICATES[group.arm]
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=strain_genotype(group.strain),
                    environment=proteome_environment(group.condition),
                    phenotype=self._phenotype(cells, n_replicates),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, references[group.arm], pub, itxn),
                )
                sample_rows.append(
                    {
                        "group": group.key,
                        "strain": group.strain,
                        "arm": group.arm,
                        "condition": group.condition,
                        "workbook": group.workbook,
                        "mean_column": group.mean_column,
                        "sd_column": group.sd_column,
                        "n_proteins": len(cells),
                        "n_replicates": n_replicates,
                        "reference_group": ARM_REFERENCE_KEY[group.arm],
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        pd.DataFrame(sample_rows).to_csv(
            osp.join(self.preprocess_dir, "samples.csv"), index=False
        )
        pd.DataFrame(
            [
                {
                    "protein_label": label,
                    "stored_name": stored_by_label[label],
                    "accessions": ";".join(sorted(accessions[label])),
                    "description": descriptions.get(label),
                    "reason": (
                        "merged_accessions"
                        if label in merged
                        else "not_a_locus_of_the_pinned_assembly"
                    ),
                }
                for label in dropped_labels
            ]
        ).to_csv(
            osp.join(self.preprocess_dir, "dropped_protein_labels.csv"), index=False
        )
        Path(self.preprocess_dir, "build_oracles.json").write_text(
            json.dumps(
                {
                    "rows_back_solved_per_workbook": back_solved,
                    "replicates_per_arm": ARM_REPLICATES,
                    "pp_0897_log2_fold_change_per_workbook": pp_0897_folds,
                    "significant_proteins_per_promoter_workbook": significant,
                    "curated_sheet_rows": analysis_rows,
                    "proteins_changed_in_both_promoter_variants": common_proteins,
                },
                indent=2,
            )
        )
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=sum(len(rows) for rows in workbooks.values()),
                candidate_records=EXPECTED_PROTEOME_RECORDS,
                kept_records=idx,
                dropped_records=0,
                rules=[
                    DropRule(
                        rule="protein_label_is_not_a_locus_of_the_pinned_assembly",
                        scope="protein_label",
                        description=(
                            "the released label is a title-cased UniProt gene symbol "
                            "the GenBank annotation of this assembly carries no symbol "
                            "for, a symbol that collides with another label on one "
                            "locus, or one of the contaminants the DIA-NN database was "
                            f"built to include ({DIANN_DATABASE.quote}); it has no "
                            "gene node to key an abundance to. A UniProt-to-locus-tag "
                            "crosswalk in the genomes tier would recover the host "
                            "proteins, and is proposed in the PR"
                        ),
                        n_records=0,
                        items=sorted(
                            label for label in dropped_labels if label not in merged
                        ),
                    ),
                    DropRule(
                        rule="protein_label_merges_two_accessions",
                        scope="protein_label",
                        description=(
                            "the released sheets file two distinct protein groups "
                            "under one label, so the label names two proteins and its "
                            "abundance cannot be attributed to either"
                        ),
                        n_records=0,
                        items=sorted(merged),
                    ),
                ],
                reconciliation=report,
                notes=[
                    f"{idx} records, one per released (strain, medium) sample. Nothing "
                    "is dropped at the record level: every released group is written",
                    "the record count is six, not eight, because two groups are each "
                    f"released in two workbooks ({[key for key, _, _ in SHARED_GROUPS]}) "
                    "and agree cell for cell, so each is one sample",
                    "the four released group labels are strain ids the paper never "
                    "maps onto its strain table. PP_0897's own released fold changes "
                    f"settle it: {pp_0897_folds}. Strongly down in both promoter "
                    "comparisons, far further down against the parent in the "
                    "cross-feeding comparison, and flat between the two media of the "
                    "same strain, which only a deletion can be",
                    "the replicate count differs by arm and the source states both "
                    f"numbers: {ARM_REPLICATES}. Each is confirmed by back-solving the "
                    "released t statistic, over "
                    f"{back_solved} rows per workbook",
                    f"the promoter workbooks' significance sheets hold {significant}, "
                    "which reproduces the Results text's 28 + 63 and 19 + 37 exactly "
                    "and is what assigns each workbook its promoter variant; the "
                    f"curated sheet holds {analysis_rows} rows of which "
                    f"{common_proteins} carry both arms' fold changes with matching "
                    "sign, reproducing 91 + 56 - 13 and 'Only 13 proteins showed "
                    "matching amplitudes'",
                    "each record's phenotype_reference is its own ARM's parental "
                    f"sample ({ARM_REFERENCE_KEY}), because the two arms were searched "
                    "separately and release different protein sets (2,494 against "
                    "2,470). The released volcano for 2487_M9_alanine_malate is "
                    "against 2487_M9_pCA_alanine_malate rather than the parent; the "
                    "stored reference is still the arm's parental sample, which is the "
                    "strain baseline the schema field means",
                    "a parental sample is both a record and the reference it is "
                    "copied into: a measured sample is not spent by being used as a "
                    "baseline",
                    f"{len(dropped_labels)} of {len(labels)} protein LABELS are "
                    f"dropped, leaving {len(kept_labels)} candidates; each record "
                    "carries the subset its own arm released",
                    "every record's genome_reference is the same object: the four "
                    "strains differ by their own PP_0897 edit, which is a perturbation "
                    "in the genotype, and share the D1b_gf background",
                    "the chassis cassette's three heterologous genes are NOT typed: "
                    f"{PATHWAY_NOT_TYPED.note}",
                    f"no titer family is written: {TITER_NOT_A_DATASET.note}",
                    f"no growth family is written: {GROWTH_NOT_A_DATASET.note}",
                    f"the BIOLOG grid is released and not ingested: "
                    f"{BIOLOG_NOT_A_DATASET.note}",
                    "temperature is a typed gap on every record's environment: the "
                    "shotgun-proteomics Methods never state one",
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "banerjee2025 proteome: %d records over %d protein labels from %d released "
            "cells; %d labels dropped (%d outside the namespace, %d merged)",
            idx,
            len(kept_labels),
            sum(len(rows) for rows in workbooks.values()),
            len(dropped_labels),
            len(dropped_labels) - len(merged),
            len(merged),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _provenance() -> Provenance:
    """The released artifact every stored abundance was read from."""
    return Provenance(
        source_uri=f"si/{WORKBOOK_PJ}",
        citation_key=CITATION_KEY,
        sha256=RAW_MEMBERS[f"si/{WORKBOOK_PJ}"][1],
        method=(
            "Supplementary Data S2 'Full t-test output' log2_mean_<group> columns (the "
            "replicate-wise mean of a log2 DIA-NN LFQ quantity), with SE = "
            "log2_std_<group> / sqrt(n) at n = 4 for the promoter arm and n = 3 for the "
            "cross-feeding arm"
        ),
        page=f"{FULL_SHEET} of each deposited t-test workbook",
    )


def group_uniqueness_rule(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L1: one record per released (strain, medium) sample, and all six are present.

    The strain axis is the record's OWN edit, not ``genome_reference.strain``: every
    record shares the D1b_gf background, so the background alone would collapse all six.
    ``promoter_name`` is part of the key because the two promoter variants differ in
    nothing else -- same locus, same direction, same medium.
    """
    seen = Counter(
        (
            json.dumps(
                sorted(
                    (
                        p["perturbation_type"],
                        p["systematic_gene_name"],
                        p.get("promoter_name"),
                    )
                    for p in record["experiment"]["genotype"]["perturbations"]
                )
            ),
            json.dumps(
                sorted(
                    (
                        p["agent"]["name"],
                        None if p.get("magnitude") is None else p["magnitude"]["value"],
                    )
                    for p in record["experiment"]["environment"]["perturbations"]
                )
            ),
        )
        for record in records
    )
    duplicates = {key: n for key, n in seen.items() if n > 1}
    passed = len(seen) == EXPECTED_PROTEOME_RECORDS and not duplicates
    return LevelResult(
        level=Level.L1,
        name="group_uniqueness",
        passed=passed,
        message=(
            f"{len(seen)} distinct (strain edit, carbon regime) samples, one record each"
            if passed
            else f"{len(seen)} distinct samples with {len(duplicates)} duplicated"
        ),
        details={"n_samples": len(seen), "n_duplicated": len(duplicates)},
    )


def replicate_split_rule(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L2: a record's stored replicate count is one of the two sourced arm counts."""
    counts = Counter(
        n
        for record in records
        for n in record["experiment"]["phenotype"]["n_replicates"].values()
    )
    allowed = set(ARM_REPLICATES.values())
    passed = set(counts) == allowed
    return LevelResult(
        level=Level.L2,
        name="replicate_split",
        passed=passed,
        message=(
            f"every stored n_replicates is one of {sorted(allowed)}"
            if passed
            else f"stored replicate counts {sorted(counts)} are not {sorted(allowed)}"
        ),
        details={"counts": {str(k): v for k, v in counts.items()}},
    )


def assembly_pin_rule(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L3: every record is written against the pinned KT2440 assembly and D1b_gf."""
    sets = {
        record["reference"]["genome_reference"]["assembly_set"] for record in records
    }
    strains = {record["reference"]["genome_reference"]["strain"] for record in records}
    passed = sets == {KT2440_ASSEMBLY_SET} and strains == {CHASSIS}
    return LevelResult(
        level=Level.L3,
        name="assembly_pin",
        passed=passed,
        message=(
            f"all {len(records)} records pin {KT2440_ASSEMBLY_SET} / {CHASSIS}"
            if passed
            else f"assembly sets {sorted(sets)}, strains {sorted(strains)}"
        ),
        details={"assembly_sets": sorted(sets), "strains": sorted(strains)},
    )


def gene_containment_rule(
    records: Sequence[dict[str, Any]], loci: set[str]
) -> LevelResult:
    """L4: every stored protein key and every perturbed gene is a locus of the assembly."""
    keys: set[str] = set()
    for record in records:
        keys |= set(record["experiment"]["phenotype"]["protein_abundance"])
        keys |= {
            p["systematic_gene_name"]
            for p in record["experiment"]["genotype"]["perturbations"]
        }
        keys |= {
            allele["systematic_gene_name"]
            for allele in record["reference"]["genome_reference"]["background"][
                "alleles"
            ]
        }
    inside = keys & loci
    overlap = len(inside) / len(keys) if keys else 0.0
    return LevelResult(
        level=Level.L4,
        name="gene_containment_kt2440",
        passed=overlap == 1.0,
        message=(
            f"all {len(keys)} stored identifiers are loci of {KT2440_ASSEMBLY_SET}"
            if overlap == 1.0
            else f"{len(keys) - len(inside)} of {len(keys)} are not loci"
        ),
        details={"n_keys": len(keys), "n_inside": len(inside), "overlap": overlap},
    )


def stored_scale_rule(records: Sequence[dict[str, Any]], raw_dir: str) -> LevelResult:
    """L4: a stored abundance is the released ``log2_mean`` cell, byte for byte.

    Re-reads the deposited workbooks and re-joins every stored value to the cell it came
    from, including the SE, which must be the released SD over sqrt(n) of that record's
    arm.
    """
    workbooks = read_workbooks(raw_dir)
    genome_loci: dict[str, dict[str, tuple[float, float]]] = {
        group.key: group_cells(workbooks[group.workbook], group)
        for group in PROTEOME_GROUPS
    }
    checked = 0
    mismatched = 0
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        n = next(iter(phenotype["n_replicates"].values()))
        root_n = math.sqrt(n)
        candidates = [
            key
            for key, cells in genome_loci.items()
            if ARM_REPLICATES[next(g.arm for g in PROTEOME_GROUPS if g.key == key)] == n
            and len(cells) >= len(phenotype["protein_abundance"])
        ]
        matched = False
        for key in candidates:
            cells = genome_loci[key]
            values = {mean for mean, _ in cells.values()}
            if set(phenotype["protein_abundance"].values()) <= values:
                matched = True
                for tag, mean in phenotype["protein_abundance"].items():
                    checked += 1
                    se = phenotype["protein_abundance_se"][tag]
                    hit = [
                        (m, s)
                        for m, s in cells.values()
                        if m == mean and abs(s / root_n - se) <= 1e-12
                    ]
                    if not hit:
                        mismatched += 1
                break
        if not matched:
            mismatched += len(phenotype["protein_abundance"])
    passed = mismatched == 0 and checked > 0
    return LevelResult(
        level=Level.L4,
        name="stored_scale_is_the_released_log2_mean",
        passed=passed,
        message=(
            f"all {checked} stored abundances re-join a released log2_mean cell with "
            "SE = released SD / sqrt(n)"
            if passed
            else f"{mismatched} of {checked} stored values do not re-join"
        ),
        details={"n_checked": checked, "n_mismatched": mismatched},
    )


def verify_build(dataset_root: str, data_root: str | None = None) -> VerificationReport:
    """Run this module's L0-L4 gate over a built tree and write the report."""
    from torchcell.verification.protein import verify_protein_dataset
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    genome = bacterial_genome("pputida", "KT2440", data_root)
    report = verify_protein_dataset(
        records,
        dataset_name="proteome_banerjee2025",
        provenance=_provenance(),
        expected_count=EXPECTED_PROTEOME_RECORDS,
        allow_duplicate_orfs=True,
    )
    report.add(group_uniqueness_rule(records))
    report.add(replicate_split_rule(records))
    report.add(assembly_pin_rule(records))
    report.add(gene_containment_rule(records, set(genome.genbank.loci)))
    report.add(stored_scale_rule(records, osp.join(dataset_root, "raw")))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build the proteome dataset and verify it, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    genome = bacterial_genome("pputida", "KT2440", data_root)
    root = osp.join(data_root, "data/torchcell/proteome_banerjee2025")
    dataset = ProteomeBanerjee2025Dataset(root=root, pputida_genome=genome)
    print(f"ProteomeBanerjee2025Dataset: len = {len(dataset)}")
    dataset.close_lmdb()
    accounting = json.loads(
        Path(root, "preprocess", "build_accounting.json").read_text()
    )
    print(
        json.dumps(
            {
                key: accounting[key]
                for key in (
                    "source_rows",
                    "candidate_records",
                    "kept_records",
                    "dropped_records",
                    "notes",
                )
            },
            indent=2,
        )
    )
    report = verify_build(root, data_root)
    print(report.summary())


if __name__ == "__main__":
    main()
