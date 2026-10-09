# torchcell/datasets/pputida/carruthers2025
# [[torchcell.datasets.pputida.carruthers2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/carruthers2025
# Test file: tests/torchcell/datasets/pputida/test_carruthers2025.py
"""Carruthers 2025 CRISPRi isoprenol production campaign in P. putida KT2440.

Carruthers et al. 2025 (Nat Commun, doi:10.1038/s41467-025-66304-8; PMID 41390487) ran
six automated design-build-test-learn cycles over multiplexed CRISPRi arrays on an
engineered isoprenol-producing KT2440 chassis, with a paired global proteome panel.
This module serves every released readout, as three dataset classes because
``ExperimentDataset.transform_item`` validates against ONE ``experiment_class``:

- :class:`IsoprenolTiterCarruthers2025Dataset` -- ``ProductTiterExperiment``, one record
  per ``(CRISPRi construct, DBTL cycle)`` strain, titer in ``ug/mL`` with the sample SD
  over biological triplicates.
- :class:`ProteomeCarruthers2025Dataset` -- ``BacterialProteinAbundanceExperiment``, one
  record per released off-target-study proteome sample (Top3 DIA-NN abundances).
- :class:`ProteomeFoldChangeCarruthers2025Dataset` --
  ``BacterialProteinFoldChangeExperiment``, one record per released differential
  contrast COLUMN: 14 single-guide CRISPRi contrasts from ``Figure 5b`` plus the 2
  knockout contrasts the Fig. 6 caption names, of the 10 ``Figure 6b`` releases.

THE RELATIVE FAMILY IS NOT A MODE OF THE ABSOLUTE ONE. A Top3 level and a log2 ratio
are different measurements, and ``ProteinAbundancePhenotype`` has no field for the
p-value a differential release ships beside its ratio, so the fold changes are
``ProteinFoldChangePhenotype`` records with their own class, adapter and gate. What a
fold-change record's denominator IS comes from the source, never from this loader:
``Figure 5b``'s is "the non-target control strain", and the Source Data's own
``Supplementary Figure 9`` sheet holds the single cell ``See Figure 5b``, which is what
makes that caption a statement about this sheet. Two sheets are refused with their
measurements rather than guessed: eight of ``Figure 6b``'s ten columns, whose
denominators no mirrored statement names, and ``Supplementary Figure 10`` entirely,
because ``fold_change_scale`` is required and no mirrored statement names a scale or a
unit for it.

THE GENOTYPE, DECOMPOSED. The reference is the chassis ``IY1449b`` (KT2440 with
markerless in-frame deletions), carried as a ``BacterialStrainBackground`` on an
``AssemblyReferenceGenome`` pinned to ``pputida_KT2440_ASM756v2`` / ``GCA_000007565.2``.
Everything added on top of it is a perturbation in ``Genotype``:

- the five heterologous pIY670 pathway genes as ``HeterologousPathwayPerturbation``
  (IY1449b + pIY670 IS the production strain IY1452b), and
- each guide target as a ``BacterialCrisprInterferencePerturbation`` with its ``PP_``
  locus tag (``ProteomeCarruthers2025Dataset`` adds the chromosomal ``PP_0815``
  knockout its panel was built in as a ``BacterialDeletionPerturbation``).

THE CHASSIS, AND A SOURCE DISAGREEMENT KEPT RATHER THAN RESOLVED BY PREFERENCE. Two
mirrored statements of IY1449b's genotype do not match. Methods: "The selected chassis
strain, P. putida IY1449b, has the in-frame deletions dphaABC, dmvaB, dhbdH, and
4,538,575d86,812 (dzwf, dglZ, and dliuC)". The Fig. 2 caption: "P. putida IY1449b
(dphaABC, dmvaB, dhbdH, dldhA, dzwfB, dgntZ, and dliuC)". Both are recorded as
``provenance`` quotes on the background; what reconciles them is MEASURED, not chosen:

- The stated span 4,538,575 + 86,812 bp = 4,538,575..4,625,386 on GCA_000007565.2
  contains ``PP_4042`` (zwfB, 4,554,991..4,556,496), ``PP_4043`` (gntZ,
  4,556,493..4,557,476) and ``PP_4066`` (liuC, 4,590,930..4,591,745) entirely, while
  the symbol ``zwf`` alone resolves to ``PP_5351`` at 6,099,177 -- OUTSIDE the span. So
  the Methods' bare ``zwf`` is the caption's ``zwfB`` and its ``glZ`` is ``gntZ``.
- ``ldhA`` resolves to ``PP_1649`` at 1,840,974, outside the span, which is consistent
  with the Methods listing it nowhere and the caption listing it as its own deletion.
  It is typed as an allele on the caption's authority and that is said in the note.

TWO FIDELITY GAPS IN THAT BACKGROUND, STATED BECAUSE NEITHER IS REPRESENTABLE. The
86,812 bp span removes 57 annotated genes entirely while the source names three, so 54
removed loci carry no allele record. And ``phaC`` is not a gene symbol of this assembly
(``phaA`` -> ``PP_5003`` and ``phaB`` -> ``PP_5004`` resolve; ``phaC`` does not), so the
third gene of ``dphaABC`` gets no locus. Both survive verbatim in
``genotype_statement``; neither is a ``ProvenanceGap``, because a gap must name a field
that is ``None`` and these are missing ROWS, not missing fields.

TITER UNITS: THE SOURCE'S mg/L IS STORED VERBATIM AS ``ug/mL``. ``ConcentrationUnit``
has no ``mg/L`` member and 1 mg/L is exactly 1 ug/mL, so the released number is stored
unchanged under the numerically identical unit rather than divided by 1000. Adding
``mg_per_l`` to ``ConcentrationUnit`` would be the cleaner fix and is a served-closure
decision, so it is raised in the PR and not taken here.

NOTHING IS DROPPED FROM THE TITER FAMILY. Every non-control culture becomes a record.
The 90 control cultures (18 in DBTL0, 12 in each of DBTL1-6) are not records: they are
the per-cycle ``phenotype_reference``. The authors' CRISPRi proteomics filter
("pass filter?") is carried in ``preprocess/pass_filter.csv`` and NOT on the record:
it reports whether the designed knockdown was REALIZED, and no certainty axis exists to
type that on a perturbation. The paper is explicit that the filter shaped only model
training -- "no data was excluded in our analysis".

THE PROTEOME PANEL IS THE RELEASED ARM, NOT THE WHOLE CAMPAIGN. The 472-strain proteome
lives in seven PRIDE accessions of raw DIA files (PXD063733 DBTL0, PXD063737 DBTL1,
PXD063738 DBTL2, PXD063740 DBTL3, PXD063743 DBTL4, PXD063744 DBTL5, PXD063746 DBTL6)
and in a Dryad deposit of two processed CSVs. All nine locations are enumerated in the
raw mirror's ``si_data_sources``; none is deposited. The PRIDE accessions hold raw mass
spectra, which no loader here consumes, and the Dryad files are UNRETRIEVABLE by script:
``datadryad.org`` serves an Anubis JavaScript proof-of-work challenge on both the public
``/downloads/file_stream/<id>`` route (HTTP 200, challenge page) and the API download
route (HTTP 401, bearer token required), measured 2026-10-07. The manual recipe is in
``si_expected`` and in the note. What IS loaded is the only per-protein, per-replicate
abundance matrix in the Source Data file: 20 samples x 3 replicates x 1,501 protein
groups for the PP_0815 off-target study.

PROTEIN KEYS ARE RECONCILED, AND THE NON-HOST ONES ARE DROPPED BY A SOURCED RULE. The
sheet's ``Protein`` column mixes ``PP_`` tags with title-cased UniProt gene symbols, and
the search database was built to include them: "the latest P. putida KT2440 Uniprot
proteome FASTA sequences in addition to the protein sequences of heterologous proteins
and common proteomic contaminants". Measured on the released file, the organism
mnemonics in ``Protein.Names`` are PSEPK (89,460 rows) plus HUMAN (240), ENTFL (120),
and 60 each of KLEPN, PIG, ECOLI, YEAST, METMA and STRP1. ``reconcile_locus_tags``
resolves 1,425 of 1,501 keys (497 already locus tags, 928 through the gene-symbol
layer); the 76 it keeps as given are not loci of this assembly and are dropped from the
abundance map with every key listed. ``Apha`` is dropped too although it resolves: the
sheet files BOTH the heterologous ``APHA_ECOLI`` (P0AE22) and a native
``Q88C43_PSEPK`` under that one symbol, so the key names two proteins.

WHAT THE PROTEOME INDEPENDENTLY CONFIRMS ABOUT THE PATHWAY. Each pIY670 part token has
a measured UniProt entry in the same Source Data file, and its organism mnemonic agrees
with the token's own suffix: ``MvaSEf`` -> ``HMGCS_ENTFL``, ``MvaEEf`` -> ``Q9FD70_ENTFL``,
``MKMm`` -> ``Q8PW39_METMA``, ``PMDScHKQ`` -> ``MVD1_YEAST`` (which also agrees with the
Methods' "promiscuous mevalonate decarboxylase (PMD*) from S. cerevisiae"), ``AphA`` ->
``APHA_ECOLI``. ``source_organism`` is therefore sourced for all five, never inferred
from the suffix alone. The identifier stored is the VERBATIM pIY670 part token, because
that is what a mirrored file states; an organism-qualified name would be a rewrite.

FOUR BUILD-TIME CROSS-SOURCE ASSERTIONS, all measured to hold on the pinned bytes:

1. ``Figure 1D`` and ``Figure 4b`` are the same 1,506 rows in the same order with
   bit-identical titers (two independently exported sheets of one measurement).
2. ``Figure 3a``'s released per-target mean equals the mean over that target's DBTL0
   replicates for all 119 joinable targets (max |diff| 3.3e-8 mg/L).
3. Supplementary Data 1's ``Mean isoprenol titer (mg/L)`` agrees for all 118 of its
   targets to within its own 2-decimal rounding (max |diff| 0.005 mg/L).
4. ``Figure 2c``'s control titers equal ``Figure 4b``'s control rows exactly in DBTL1-6
   and to 3.3e-6 mg/L in DBTL0 (the sheets export DBTL0 at different precision).

The one cross-source DISAGREEMENT is kept as a finding: ``PP_3365`` is a DBTL0 guide
target in the Source Data but is absent from Supplementary Data 1's 120-row target
table, so that table covers 120 of the 121 targets actually screened.

COUNTS, AND WHY THEY DIFFER FROM THE ABSTRACT'S. The paper reports "472 unique strains
(125 single perturbations and 347 combinations) in triplicate"; 472 is the 1,416
non-control cultures divided by three. Grouping those cultures by (construct, cycle)
gives 465 strains, because seven of them carry SIX replicates (R1-R6) rather than three:
``PP_0812``, ``PP_0813``, ``PP_4678``, ``PP_4679`` in DBTL0 and ``PP_0814_PP_4192``,
``PP_0814_PP_4862``, ``PP_2137_PP_4189`` in DBTL1. 465 + 7 = 472 only under the
divide-by-three reading. This loader stores 465 strains and the replicate count each one
actually has.
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
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import openpyxl
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
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
from torchcell.datamodels.media import M9_NREL_CARRUTHERS2025
from torchcell.datamodels.schema import (
    AlleleEdit,
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    BacterialBackgroundAllele,
    BacterialCrisprInterferencePerturbation,
    BacterialDeletionPerturbation,
    BacterialGeneNamespace,
    BacterialProteinAbundanceExperiment,
    BacterialProteinAbundanceExperimentReference,
    BacterialProteinFoldChangeExperiment,
    BacterialProteinFoldChangeExperimentReference,
    BacterialStrainBackground,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    CultureEnvironment,
    CultureFormat,
    EndpointRule,
    Experiment,
    ExperimentReference,
    FoldChangeScale,
    GeneAdditionPerturbation,
    GenomicSpan,
    Genotype,
    HeterologousPathwayPerturbation,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    ProteinAbundancePhenotype,
    ProteinFoldChangePhenotype,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
    resolve_uniprot_accessions,
    uniprot_locus_crosswalk,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.verification.levels import l3_convention, l4_cross_source
from torchcell.verification.product_titer import verify_product_titer_dataset
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

DOI = "10.1038/s41467-025-66304-8"
PMID = "41390487"
PMCID = "PMC12748988"
TITLE = (
    "Automation and machine learning drive rapid optimization of isoprenol production "
    "in Pseudomonas putida"
)

CITATION_KEY = "carruthersAutomationMachineLearning2025"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: PMC Article Datasets bucket prefix of this article's open-access version.
PMC_PREFIX = f"{PMCID}.1"

#: Source Data file (Supplementary MOESM9): every released per-replicate number.
SOURCE_DATA_FILENAME = "41467_2025_66304_MOESM9_ESM.xlsx"
SOURCE_DATA_REL = f"si/{SOURCE_DATA_FILENAME}"
SOURCE_DATA_SHA256 = "1b3a7ab5f165386ba1c11e8873c397e7b03f5189274c0a423c5c60dd3616c1c7"
#: Supplementary Data 1 (MOESM4): the DBTL0 gene-target table, consumed only as the
#: cross-source oracle for the per-target mean titer.
TARGETS_FILENAME = "41467_2025_66304_MOESM4_ESM.xlsx"
TARGETS_REL = f"si/{TARGETS_FILENAME}"
TARGETS_SHA256 = "236c8dc5d3b18b7a459f69a6812efa534bba13cd49e20e1a89a01b7371d1e54b"
SI_RETRIEVED_AT = "2026-10-07"

#: Mirrored OCR / SI files every sourced value quotes (torchcell-library, not raw).
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "ca9a8a2593d2ae3ab3bacfb767e797ece2bdaa0228a4f798f1c38e1af73ef88d"
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "2f83cfecc539e6607e33c51c420fb0be95c335a494cba6ef485eb5cb627ea0a5"
#: Supplementary Data 3: the plasmid table that states pIY670's part composition.
PLASMIDS_XLSX = "si/si6.xlsx"
PLASMIDS_SHA256 = "e6704d6176d61c76f12243e8248bcff0767d7d57c6a0737da07d5574072c4759"

#: The seven PRIDE projects the campaign's raw DIA files were deposited to, one per
#: DBTL cycle. Enumerated in full because a seventh of a campaign is not the campaign.
PRIDE_ACCESSIONS: dict[str, str] = {
    "DBTL0": "PXD063733",
    "DBTL1": "PXD063737",
    "DBTL2": "PXD063738",
    "DBTL3": "PXD063740",
    "DBTL4": "PXD063743",
    "DBTL5": "PXD063744",
    "DBTL6": "PXD063746",
}
DRYAD_DOI = "10.5061/dryad.gtht76hzh"
#: The Dryad deposit, retrieved BY HAND on 2026-10-09 (issue #739) because
#: ``datadryad.org`` serves an Anubis JavaScript proof-of-work challenge to scripts.
#: These three files are the campaign proteome: the per-culture Top3 matrix that
#: :class:`CampaignProteomeCarruthers2025Dataset` reads, its per-culture metadata, and
#: the README whose column descriptions are the authority for what one number IS.
DRYAD_DIR_REL = "data/dryad"
DRYAD_VERSION = "v20250820"
DRYAD_ZIP_FILENAME = "doi_10_5061_dryad_gtht76hzh__v20250820.zip"
DRYAD_ZIP_SHA256 = "0c9e4feda2a0d15ebf481080f9dc2dccfaaa45238afe3f6a18f96ab7018468c0"
DRYAD_TOP3_FILENAME = (
    "CRISPRi_automation_Pputida_proteomic_Top3_peptide_quantification_method_data.csv"
)
DRYAD_TOP3_SHA256 = "3e9d75e573a3d09147ba99f114652ba41632c043fffc0b03f044e25a2e34c580"
DRYAD_METADATA_FILENAME = "CRISPRi_automation_Pputida_proteomic_metadata.csv"
DRYAD_METADATA_SHA256 = (
    "46bc06ca545e727ce18f194487ce0e7086b759faba7815e44015b3fa9da560c8"
)
DRYAD_README_FILENAME = "README.md"
DRYAD_README_SHA256 = "817bd5e0d349effe718acf792b339de42138985aa0c14a2246d0da8784547578"
DRYAD_TOP3_REL = f"{DRYAD_DIR_REL}/{DRYAD_TOP3_FILENAME}"
DRYAD_METADATA_REL = f"{DRYAD_DIR_REL}/{DRYAD_METADATA_FILENAME}"
DRYAD_README_REL = f"{DRYAD_DIR_REL}/{DRYAD_README_FILENAME}"
DRYAD_ZIP_REL = f"{DRYAD_DIR_REL}/{DRYAD_ZIP_FILENAME}"
DRYAD_RETRIEVED_AT = "2026-10-09"
#: The manual recipe, verbatim from the deposit's own ``DEPOSIT.md``, recorded as the
#: ``retrieval_command`` of every Dryad file's ``manual_browser`` retrieval record. It
#: is what a rebuild re-runs by hand before verifying the four sha256 digests above.
DRYAD_MANUAL_RECIPE = (
    "open https://doi.org/10.5061/dryad.gtht76hzh in a browser, solve the challenge, "
    'click "Download dataset", save the zip unchanged, then unzip it into this '
    "directory beside the zip."
)
#: Who produced those bytes and how, verbatim from ``DEPOSIT.md``.
DRYAD_RETRIEVED_BY = "the owner (mjvolk3), browser download"
DRYAD_DEPOSIT_NOTE = (
    "retrieval_method: manual_browser (datadryad.org serves an Anubis JavaScript "
    "proof-of-work challenge to scripts; measured 2026-10-07, issue #739). files: the "
    "arrival zip plus its three members, listed with sha256 in SHA256SUMS.txt. The zip "
    "is the bytes that arrived; the members are what loaders read."
)
#: The Dryad columns this loader keys on and the one it reads the measurement from.
DRYAD_KEY_COLUMNS: tuple[str, ...] = (
    "DBTL_Cycle",
    "Line_name",
    "Line",
    "Replicate",
    "Isoprenol_titer",
    "Isoprenol_titer_units",
)
#: The released titer unit of the Dryad table's own ``Isoprenol_titer_units`` column,
#: asserted to be the only value in it.
DRYAD_TITER_UNIT = "mg/L"
#: What one campaign-proteome number is: a PERCENT of the proteome, not the panel's raw
#: Top3 signal, so the two proteome families can never be silently pooled.
CAMPAIGN_MEASUREMENT_TYPE = "dia_nn_top3_percent_of_proteome_mean"
#: One record per ``(construct, DBTL cycle)`` strain of the campaign, the same identity
#: the titer family keys on.
EXPECTED_CAMPAIGN_PROTEOME_RECORDS = 465
#: The per-cycle control proteome references, DBTL0 to DBTL6.
EXPECTED_CAMPAIGN_REFERENCES = 7
#: Accessions of the Dryad matrix reaching exactly one KT2440 locus through the GOA
#: proteome crosswalk: 1,842 of 2,187, measured 2026-10-09. The floor sits just below
#: it; the 345 that do not are listed in ``preprocess/dropped_accessions.csv``.
CAMPAIGN_MIN_RESOLVED_FRACTION = 0.84
#: Accessions the Dryad matrix and ``SHEET_PROTEOME`` BOTH key, so that both the GOA
#: crosswalk and the sheet's own accession -> symbol route reach a locus for them.
#: Measured 2026-10-09 on the pinned bytes.
CAMPAIGN_CROSSWALK_SHARED = 1224
#: The accessions of those 1,224 the two routes send to DIFFERENT loci, as
#: ``accession -> (GOA locus, sheet locus)``, measured 2026-10-09. Pinned EXACTLY and
#: not under a tolerance: each one is a named symbol collision on the sheet's route,
#: explained in :func:`assert_campaign_accession_crosswalk`, and a third disagreement
#: would be a new fact about one of the two files rather than an accepted error rate.
CAMPAIGN_CROSSWALK_DISAGREEMENTS: dict[str, tuple[str, str]] = {
    "Q88G93": ("PP_3832", "PP_4472"),
    "Q88L01": ("PP_2137", "PP_2051"),
}

#: Source Data sheets this module reads.
SHEET_TITER = "Figure 4b"
SHEET_TITER_ALT = "Figure 1D"
SHEET_TARGET_MEANS = "Figure 3a"
SHEET_CONTROLS = "Figure 2c"
SHEET_PROTEOME = "Supplementary Figure 13abc"
#: The KO-against-CRISPRi comparison panel, and its row-for-row duplicate sheet.
SHEET_KO_PANEL = "Figure 6a"
SHEET_KO_PANEL_ALT = "Supplementary Figure 11"
#: The KO-plus-array panel: KO-only and CRISPRi-on-a-KO strains.
SHEET_KO_ARRAYS = "Figure 6d"
#: The titer arm of the PP_0815 off-target panel whose proteome is ``SHEET_PROTEOME``.
SHEET_OFFTARGET_TITER = "Supplementary Figure 13d"
#: The overexpression panel: titer and per-protein abundance across inducer levels.
SHEET_OVEREXPRESSION_TITER = "Supplementary Figure 12bd"
SHEET_OVEREXPRESSION_PROTEOME = "Supplementary Figure 12ac"
#: The per-cycle control proteome. NOT read -- see :data:`CONTROL_PROTEOME_DEFERRAL`.
SHEET_CONTROL_PROTEOME = "Supplementary Figure 15"
#: The released DIFFERENTIAL proteomics: per-protein log2 fold change with a p-value.
#: ``SHEET_FOLD_CHANGE`` is the single-guide CRISPRi panel, ``SHEET_FOLD_CHANGE_POINTER``
#: the sheet whose only cell points at it, and ``SHEET_KO_FOLD_CHANGE`` the KO panel.
SHEET_FOLD_CHANGE = "Figure 5b"
SHEET_FOLD_CHANGE_POINTER = "Supplementary Figure 9"
SHEET_KO_FOLD_CHANGE = "Figure 6b"
#: The best-array heatmap. NOT read -- see :data:`BEST_ARRAY_HEATMAP_REFUSAL`.
SHEET_BEST_ARRAY_HEATMAP = "Supplementary Figure 10"

KT2440_NAMESPACE: BacterialGeneNamespace = "pputida_kt2440_locus_tag"
#: The host species, as the deposited assembly report names it. An extra copy of a
#: NATIVE gene must declare THIS as its ``source_organism`` or the runner's
#: ``host_perturbed_gene_set`` would read it as a gene of another genome and skip it.
SPECIES = "Pseudomonas putida"
KT2440_ASSEMBLY_SET: BacterialAssemblySet = "pputida_KT2440_ASM756v2"
#: The single chromosome of GCA_000007565.2, as the assembly report names it.
KT2440_REPLICON = "AE015451.2"

#: Chassis strain labels. ``IY1449b`` is the background every record is written against;
#: ``IY1452b`` is that strain carrying pIY670 and is what the pathway perturbations make.
CHASSIS_STRAIN = "IY1449b"
PRODUCTION_STRAIN = "IY1452b"

#: 1-based inclusive span of the deletion the Methods state as ``4,538,575d86,812``.
SPAN_START = 4_538_575
SPAN_LENGTH = 86_812
SPAN_END = SPAN_START + SPAN_LENGTH - 1

#: A construct name is ``PP_`` tags joined by ``_``, optionally with a non-targeting
#: filler guide (``NT1`` / ``NT2``) occupying an array position.
LOCUS_TAG_RE = re.compile(r"PP_\d{4}")
REPLICATE_RE = re.compile(r"^(?P<base>.+)-R(?P<replicate>\d+)$")

#: The four dataset families this release serves, one per experiment class.
Family = Literal["titer", "proteome", "campaign_proteome", "fold_change"]
#: The product every titer record measures, as the compound layer canonicalizes it.
PRODUCT_NAME = "isoprenol"
#: One record per ``(construct, DBTL cycle)`` strain of ``Figure 4b``.
EXPECTED_CRISPRI_TITER_RECORDS = 465
#: The four Source Data panels below add 37 more: 12 + 4 + 19 + 2.
EXPECTED_PANEL_TITER_RECORDS = 37
#: One record per titer strain the release states a full genotype and environment for.
EXPECTED_TITER_RECORDS = EXPECTED_CRISPRI_TITER_RECORDS + EXPECTED_PANEL_TITER_RECORDS
#: One record per released ``SHEET_PROTEOME`` sample bar the non-targeting reference.
EXPECTED_PP0815_PROTEOME_RECORDS = 19
#: ``SHEET_OVEREXPRESSION_PROTEOME``'s two uninduced samples.
EXPECTED_OVEREXPRESSION_PROTEOME_RECORDS = 2
EXPECTED_PROTEOME_RECORDS = (
    EXPECTED_PP0815_PROTEOME_RECORDS + EXPECTED_OVEREXPRESSION_PROTEOME_RECORDS
)
#: One record per sgRNA target column of ``SHEET_FOLD_CHANGE``.
EXPECTED_FOLD_CHANGE_RECORDS_FIG5B = 14
#: One record per ``SHEET_KO_FOLD_CHANGE`` column the Fig. 6 caption names (see
#: :data:`KO_FOLD_CHANGE_CONTRASTS`); the other eight columns are ledgered.
EXPECTED_FOLD_CHANGE_RECORDS_FIG6B = 2
EXPECTED_FOLD_CHANGE_RECORDS = (
    EXPECTED_FOLD_CHANGE_RECORDS_FIG5B + EXPECTED_FOLD_CHANGE_RECORDS_FIG6B
)
#: What one stored fold change IS: a log2 ratio tested by the paired two-tailed
#: Student's t-test both captions and the Methods name.
FOLD_CHANGE_MEASUREMENT_TYPE = "dia_log2_fold_change_paired_two_tailed_t_test"
#: The released column suffixes. ``_log10_pval`` is UNSIGNED -log10(p), which
#: :func:`p_value_from_neg_log10` converts by exact arithmetic.
FOLD_CHANGE_SUFFIX = "_log2_FC"
NEG_LOG10_P_VALUE_SUFFIX = "_log10_pval"
#: Tolerance of ``-log10(10**-x) == x``. Measured over all 6,222 released p-values of
#: the two sheets: the maximum round-trip error is exactly 0.0.
P_VALUE_ROUND_TRIP_TOL = 1e-12
#: The significance ceiling both sheets are pre-filtered at, which the Fig. 5b caption
#: states. Measured: the largest released p-value is 0.04999950946964488.
FOLD_CHANGE_P_VALUE_CEILING = 0.05
#: The ``SHEET_KO_FOLD_CHANGE`` columns a verbatim statement names as a contrast.
KO_FOLD_CHANGE_CONTRASTS: tuple[str, ...] = ("PP_0812", "PP_0815")
#: The exact cell ``SHEET_FOLD_CHANGE_POINTER`` holds, which is what makes the
#: Supplementary Fig. 9 caption a statement about ``SHEET_FOLD_CHANGE``'s denominator.
FOLD_CHANGE_POINTER_CELL = "See Figure 5b"
#: What one Top3 number is, named so heterogeneous proteomics is never silently mixed.
PROTEOME_MEASUREMENT_TYPE = "dia_nn_top3_peptide_signal_mean"
#: The proteome panel's background: every sample is a derivative of the PP_0815 KO.
PROTEOME_BACKGROUND_DELETION = "PP_0815"
#: The panel's reference sample: the KO carrying a NON-targeting sgRNA.
PROTEOME_REFERENCE_SAMPLE = "JBEI_PP_0815_NT_48hr"
#: The panel's positive control: the same KO carrying the PP_0815 sgRNA.
PROTEOME_TARGET_SAMPLE = "JBEI_PP_0815_Target_48hr"
PROTEOME_SAMPLE_RE = re.compile(r"^JBEI_OTS_(?P<tag>PP_\d{4})(?:_\d)?(?:_P4)?_48hr$")

#: ``Figure 6a`` / ``Figure 6d`` arm labels. ``CRISPRi`` is the sgRNA-only strain and is
#: a re-export of ``Figure 4b``; the other arms are the genuinely new KO cultures.
ARM_CRISPRI = "CRISPRi"
ARM_KO_TARGET = "Target"
ARM_KO_NONTARGET = "Non-target"
ARM_KO_ONLY = "KO Only"
ARM_KO_PLUS_CRISPRI = "CRISPRi and KO"
#: ``Figure 6d``'s control line, measured to be three DBTL6 control cultures.
KO_ARRAY_CONTROL_LINE = "Control"
KO_ARRAY_CONTROL_CYCLE = 6
#: ``Figure 6d``'s ``KO Only`` lines end in ``_KO``; the combined lines add ``with``.
KO_ARRAY_LINE_RE = re.compile(r"^(?P<deleted>.+?)_KO(?: with (?P<knocked_down>.+))?$")
#: ``PP_0812-15`` names an inclusive locus-number range, not a single tag.
LOCUS_RANGE_RE = re.compile(r"^PP_(?P<start>\d{4})-(?P<end>\d{2})$")

#: ``Supplementary Figure 13d``'s reference and positive-control arm labels. The sheet
#: writes ``Non-Target`` where ``Figure 6a`` writes ``Non-target``.
OFFTARGET_REFERENCE_SAMPLE = "Non-Target"
OFFTARGET_TARGET_SAMPLE = "Target"
#: An off-target sample is a locus tag, optionally with a repeated-culture index.
OFFTARGET_SAMPLE_RE = re.compile(r"^(?P<tag>PP_\d{4})(?:_(?P<culture>\d))?$")
#: Supplementary Table 1 lists the off-target sgRNA targets; this is its row count.
OFFTARGET_CANDIDATE_TARGETS = 14
#: Cultures the two sheets' ``Target`` triplicates share. Measured: exactly one of six,
#: which is why neither triplicate is stored as the other (see :func:`_offtarget_groups`).
OFFTARGET_TARGET_SHARED_CULTURES = 1

#: ``Supplementary Figure 12bd`` / ``12ac`` label the RFP vector control ``Control``.
OVEREXPRESSION_CONTROL_STRAIN = "Control"
#: The released induction series. Only the 0 level is loaded: the inducer's UNIT is not
#: released anywhere in the mirror (see :data:`INDUCER_UNIT_GAP`).
OVEREXPRESSION_UNINDUCED_LEVEL = 0.0
OVEREXPRESSION_INDUCED_LEVELS: tuple[float, ...] = (
    31.25,
    62.5,
    125.0,
    250.0,
    500.0,
    1000.0,
)
#: ``Supplementary Figure 12ac``'s sample names: ``<label>_Condition_<n>`` / ``_Control``.
OVEREXPRESSION_SAMPLE_RE = re.compile(
    r"^(?P<label>pSTABL[12])_(?:Condition_(?P<condition>\d)|Control)$"
)
#: The two overexpressed operons, keyed by the plotting label the Source Data uses.
#: The LOCI are the Methods' and the Results' own, not the label's: the sheet writes
#: ``pSTABL2 (PP_2971-74)`` while five mirrored statements say ``PP_2791-94``, and
#: :func:`_assert_overexpression_operons` proves the correction against the measured
#: proteome of the same samples.
OVEREXPRESSION_OPERONS: dict[str, tuple[str, ...]] = {
    "pSTABL1": ("PP_2208", "PP_2209"),
    "pSTABL2": ("PP_2791", "PP_2792", "PP_2793", "PP_2794"),
}
#: The label the Source Data attaches to each, verbatim, typo included.
OVEREXPRESSION_SHEET_LABELS: dict[str, str] = {
    "pSTABL1": "pSTABL1 (PP_2208-09)",
    "pSTABL2": "pSTABL2 (PP_2971-74)",
}
#: ``Supplementary Figure 12ac``'s two Phn keys, and the locus each one's FUNCTION puts
#: it at on the pinned assembly. The sheet's own ``Protein.Description`` says the
#: opposite for these two rows; :func:`_assert_phn_crosswalk` pins both statements.
PHN_FUNCTION_CROSSWALK: dict[str, str] = {"Phnw": "PP_2209", "Phnx": "PP_2208"}
PHN_SHEET_DESCRIPTION: dict[str, str] = {"Phnw": "PP_2208", "Phnx": "PP_2209"}


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
            method="MinerU OCR of the Supplementary Information PDF (mirror)",
            page=page,
        ),
    )


def _plasmid(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim cell of Supplementary Data 3's plasmid table."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PLASMIDS_XLSX,
            citation_key=CITATION_KEY,
            sha256=PLASMIDS_SHA256,
            method="published Supplementary Data 3 workbook (mirror)",
            page="Supplementary Table 3: List of plasmids used in this study",
        ),
    )


def _source_data(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim cell of the Source Data workbook."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SOURCE_DATA_REL,
            citation_key=CITATION_KEY,
            sha256=SOURCE_DATA_SHA256,
            method="published Source Data workbook (raw mirror)",
            page=page,
        ),
    )


def _dryad_readme(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim sentence of the deposited Dryad ``README.md``.

    The README is the only released statement of what a campaign-proteome cell IS, and
    it is a manually retrieved artifact, so the provenance names the manual recipe as
    its method rather than claiming a scripted retrieval.
    """
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=DRYAD_README_REL,
            citation_key=CITATION_KEY,
            sha256=DRYAD_README_SHA256,
            method=(
                f"Dryad {DRYAD_DOI} {DRYAD_VERSION} README, retrieved by hand "
                f"({DRYAD_MANUAL_RECIPE})"
            ),
            page=page,
        ),
    )


_METHODS_STRAIN = "Methods, 'Automated transformation of P. putida'"
_METHODS_CULTURE = "Methods, 'Passaging and culturing of P. putida strains'"
_METHODS_STATS = "Methods, 'Statistics & reproducibility'"

_Q_CHASSIS = (
    "The selected chassis strain, P. putida IY1449b, has the in-frame deletions "
    "ΔphaABC, ΔmvaB, ΔhbdH, and 4,538,575Δ86,812 (Δzwf, ΔglZ, and ΔliuC) for improved "
    "isoprenol titers"
)
_Q_CHASSIS_FIG2 = (
    "Validated sgRNA arrays were then dispensed with electrocompetent P. putida "
    "IY1449b (ΔphaABC, ΔmvaB, ΔhbdH, ΔldhA, ΔzwfB, ΔgntZ, and ΔliuC) cells harboring "
    "pIY670 again using the ECHO 550."
)
_Q_SI_TABLE2 = "KT2440 ΔphaABC, ΔmvaB, ∆hbdH, 4,538,575 Δ86,812"
_Q_PATHWAY_PLASMID = (
    "made electrocompetent for transformation with pIY670, harboring the IPP-Bypass "
    "MVA pathway, and selected $( 5 0 \\mu \\mathrm { g } / \\mathrm { m L }$ kanamycin "
    "sulfate) to generate $\\mathbf { I Y 1 4 5 2 b ^ { 3 1 } }$ ."
)
_Q_PATHWAY = (
    "The base pathway (Fig. 1a) is an engineered mevalonate (MVA) pathway with a "
    "promiscuous mevalonate decarboxylase (PMD\\*) from S. cerevisiae capable of "
    "converting mevalonate monophosphate into isopentenyl monophosphate, thereby "
    "bypassing isopentenyl diphosphate (IPP-Bypass)"
)
_Q_PIY670 = "pRK2-Kan-araC-PBAD-MvaSEf-MvaEEf-TrpoH-Ptrc1-O-MKMm-PMDScHKQ-AphA"
_Q_PIY989 = "pRSF1010-Gm-NagR-pNagAa-dCas9-thrLABC-sfGFP"
_Q_CULTURE = (
    "Following adaptation, strains were inoculated in triplicate into $1 . 5 \\mathsf "
    "{ m l }$ of M9-NREL media in a 48-well BioLector flower plate without optodes and "
    "gas-permeable sealing foil (Beckman Coulter Life Sciences) to reduce evaporation."
)
_Q_TEMPERATURE = (
    "Cultures were grown at $2 4 ^ { \\circ } \\mathrm { C }$ and shaken at 1000 RPM "
    "without humidity control as batch experiments."
)
_Q_INDUCER = (
    "Isoprenol pathway genes were induced after 8 h by the addition of L-arabinose to a "
    "final concentration of ${ 2 \\mathrm { g } } / { \\mathrm { L } }$ . Following $4 8 "
    "\\mathrm { h }$ of production, cultures were transferred from the 48-well BioLector "
    "plate into a 96-well DWP"
)
_Q_MEDIUM = (
    "M9-NREL medium was selected owing to its prevalence as a baseline $P .$ . putida "
    "production medium."
)
_Q_TRIPLICATE = (
    "All strains were cultured as biological triplicates $\\left( n = 3 \\right)$ ."
)
_Q_ERRORBARS = (
    "With the exception of the box-and-whisker plot in Fig. 4d, all error bars represent "
    "standard deviation."
)
_Q_GCFID = (
    "Isoprenol from BioLector experiments was detected using gas "
    "chromatography-flame ionization detection (GC-FID; Agilent Technologies, Santa "
    "Clara, CA)."
)
_Q_CONTROL_N = (
    "In DBTL0, the control strain was cultured across plates $( n = 1 8 )$ while "
    "subsequent cycles had three control strains per plate $( n = 1 2 )$ )."
)
_Q_TOP3 = (
    "The Top3 method, which is the average MS signal response of the three most intense "
    "tryptic peptides of each identified protein, was used to plot the quantity of "
    "targeted proteins in the samples"
)
_Q_DIANN_DB = (
    "The database used in the DIA-NN search (library-free mode) included the latest P. "
    "putida KT2440 Uniprot proteome FASTA sequences in addition to the protein "
    "sequences of heterologous proteins and common proteomic contaminants."
)
_Q_OTS = (
    "To achieve this, IY1452b strains harboring overexpression candidates, along with "
    "IY1452b ΔPP_0815 strains expressing sgRNA for the various off-target candidates, "
    "were transformed, cultured, and characterized for isoprenol production and "
    "proteomics."
)
_Q_SI_TRIPLICATE = (
    "All strains were cultured in triplicate $( \\mathsf { n } = 3 )$ and error bars "
    "represent standard deviation."
)
_Q_SI_THREE_CULTURES = (
    "These strains were cultured three times owing to poor transformation efficiency."
)
_Q_SI_OTS_BACKGROUND = "sgRNA expressed in IY1449b ΔPP_0815"
_Q_NO_EXCLUSION = (
    "Data used to train the active learning model was filtered according to the method "
    "above; however, no data was excluded in our analysis."
)
# --- the four unstored titer panels and the overexpression proteome ------------
_Q_KO_PANEL_DESIGN = (
    "We sought to investigate the presence of off-target gene downregulation amongst "
    "our best-performing sgRNAs by comparing isoprenol titers between pairs of "
    "knockout (KO) strains harboring either a non-target sgRNA or the “target” sgRNA "
    "previously used to downregulate the KO gene"
)
_Q_KO_PANEL_FIG6A = (
    "a Comparison of mean isoprenol production between strains expressing either a "
    "PP_0815-targeting sgRNA or a non-targeting control sgRNA in a knockout "
    "background, demonstrating significant off-target effects of the PP_0815 sgRNA "
    "$\\left( n = 3 \\right)$ ."
)
_Q_KO_PANEL_SI11_TITLE = (
    "Supplementary Figure 11: Comparison of titers between KO strains harboring sgRNAs "
    "and their CRISPRi analogues."
)
_Q_KO_PANEL_SI11 = (
    "Most KO strains performed similarly with the non-target and target sgRNAs, "
    "indicating no obvious off-target effects. The ΔPP_2136, ΔPP_0812, and ΔPP_0813 "
    "strains showed higher performance than their CRISPRi analogues. ΔPP_0815 strains "
    "harboring the PP_0815 sgRNA showed clear increases in titer compared to the "
    "non-target sgRNA."
)
_Q_KO_0812_15_GUIDE = (
    "While most KO strains showed similar titers to their CRISPRi counterparts from "
    "DBTL0 (Supplementary Fig. 11), ΔPP_0815 and ΔPP_0812-15 harboring PP_0815 sgRNA "
    "produced significantly more isoprenol compared to those with non-targeting "
    "guides, indicating that an off-target gene was driving isoprenol production "
    "level (Fig. 6a)."
)
_Q_OXIDASE_COMPLEX = "PP_0815 (subunits of a terminal oxidase complex PP_0812-15)"
_Q_KO_CONSTRUCTION = (
    "Stable gene knockouts were generated from the parent strain IY1449b via a "
    "Cpf1-mediated repair63."
)
_Q_SI_TABLE2_KO_STRAINS = (
    "Supplementary Table 2: List of Pseudomonas putida strains constructed in this "
    "study"
)
_Q_KO_ARRAY_RESULT = (
    "Combining KOs with specific sgRNAs for PP_0528 and PP_0815 further improved titer "
    "to 4-fold that of the control $( 6 5 1 \\mathrm { m g / L } )$ and $12 \\%$ more "
    "isoprenol than the two-sgRNA array in a strain without KOs $( 5 8 0 \\mathrm { m "
    "g / L }$ , $p < 0 . 0 2 )$ , indicating the importance of modulating rather than "
    "deleting certain genes (e.g., off-target or essential) to recapitulate production "
    "phenotypes."
)
_Q_SI_FIG13D = (
    "d) Isoprenol titers of off-target sgRNA strains failed to recapitulate the "
    "observed titer of ΔPP_0815 harboring the PP_0815 sgRNA. At best, the off-target "
    "strains performed as well as the ΔPP_0815 strain harboring non-target sgRNA."
)
_Q_SI_TABLE1_OFFTARGETS = (
    "Supplementary Table 1: sgRNA targets selected to investigate PP_0815 off-target "
    "effects."
)
_Q_OVEREXPRESSION_OPERONS = (
    "These operons, PP_2208- PP_2209 and PP_2791-PP_2794, were amplified along with "
    "the DBTL3 Control Vector using Q5 DNA polymerase (NEB) and oligos with 20 bp ${ } "
    "^ { 5 ^ { \\prime } }$ overhangs."
)
_Q_OVEREXPRESSION_RESULTS = (
    "Specifically, we overexpressed two operons, PP_2791-94 (lvaA-D; levulinic acid "
    "degradation) and PP_2208-09 $( p h n W { \\cdot } X ;$ phosphonoacetalaldehyde "
    "hydrolase) on secondary plasmids and further downregulated 14 potential "
    "off-target candidates using CRISPRi."
)
_Q_OVEREXPRESSION_SALICYLATE = (
    "The vector amplicons, designed for salicylic acid induction of the inserted "
    "genes, were digested with dpnI (Thermo Fisher Scientific), assembled (NEBuilder "
    "HiFi Assembly Cloning Kit, NEB), and, as before, cloned into XL-1-blue competent "
    "cells."
)
_Q_OVEREXPRESSION_CULTURE = (
    "Strains harboring genes informed by Stabl were adapted to M9 medium and cultured "
    "for over $4 8 \\mathrm { h }$ in a Biolector Pro with an RFP control (JBx_266188) "
    "before GC-FID analysis."
)
_Q_OVEREXPRESSION_TRANSFORMED = (
    "Plasmid sequences were verified by whole plasmid sequencing (Primordium Labs) and "
    "ultimately transformed into IY1449b with pIY670 to evaluate the impact of "
    "titrated induction on isoprenol titer."
)
_Q_SI_FIG12_PANELS = (
    "a) Top3 protein abundance of PP_2208-09 (PhnW-X; phosphonoacetylaldehyde "
    "hydrolyase) under various inducer concentrations. Induced expression led to "
    "exceptionally high protein abundance. b) Uninduced expression of PP_2207-08 "
    "yielded marginally more isoprenol compared to an RFP control c) Top3 protein "
    "abundance of PP_2791-94 (levulinic acid degradation pathway LvaA-D) under various "
    "induction concentrations."
)
_Q_SI_FIG12_UNINDUCED = (
    "Uninduced expression, however, showed a $\\sim 1 0 \\%$ increase in titer "
    "compared to the RFP control."
)
_Q_SI_FIG10_LEVULINATE = (
    "Finally, phnX is a phosphonoacetaldehyde hydrolase while PP_2793-94 encodes "
    "proteins in the levulinate degradation pathway."
)
_Q_PSTABL1_PLASMID = "pRSF1010-Gm-NagR-PP_2791-94"
_Q_PSTABL2_PLASMID = "pRSF1010-Gm-NagR-PP_2208-09"
_Q_RFP_CONTROL_PLASMID = "pRSF1010-Gm-mCherry"
_Q_SI_FIG15_TITLE = (
    "Supplementary Figure 15: Levels of heterologous pathway proteins in the controls "
    "across all DBTL cycles as determined by the Top3 Method"
)
_Q_SI_FIG15 = (
    "Pathway proteins typically followed a similar rank-order trend with PMD\\* being "
    "the highest expressed followed by mvaE. MvaS was typically the lowest expression "
    "protein in the pathway. Owing to its toxicity, expression of dCas9 was kept "
    "comparatively low. Error bars represent standard deviation."
)

# --- the deposited Dryad campaign proteome -----------------------------------
CAMPAIGN_TOP3 = _dryad_readme(
    CAMPAIGN_MEASUREMENT_TYPE,
    "All other table values: (float): Percentage of the proteome for the specific "
    "protein as calculated by the Top3 peptide absolute protein quantification method "
    "as detailed in Ahrne et al. 2013 (DOI:10.1002/pmic.201300135), consisting of the "
    "average signal response of the three most intense tryptic peptides for each "
    "protein. When the protein is not detected or is detected with fewer than three "
    "peptides the field is left blank (nan).",
    page=(
        "README, 'Files and variables', "
        "CRISPRi_automation_Pputida_proteomic_Top3_peptide_quantification_method_data"
        ".csv"
    ),
    note="what one stored abundance IS: the mean, over the strain-cycle's replicates, "
    "of the released per-replicate PERCENT of the proteome. The panel family's "
    f"{PROTEOME_MEASUREMENT_TYPE} is the raw Top3 signal of a different sheet, so the "
    "two are different measurement_types and never pooled",
)
CAMPAIGN_BLANK_IS_ABSENT = _dryad_readme(
    "not_detected_or_fewer_than_three_peptides",
    "When the protein is not detected or is detected with fewer than three peptides "
    "the field is left blank (nan).",
    page=(
        "README, 'Files and variables', "
        "CRISPRi_automation_Pputida_proteomic_Top3_peptide_quantification_method_data"
        ".csv"
    ),
    note="the sourced rule for the blank cells: a blank is an ABSENT measurement, not "
    "a zero, so a blank replicate is excluded from that protein's mean and from its "
    "n_replicates, and a protein blank in every replicate of a strain-cycle carries no "
    "key in that record. A released 0 is a different cell and is kept verbatim",
)
CAMPAIGN_ACCESSION_KEYS = _dryad_readme(
    "uniprot_accession",
    "All other column headers: Uniprot accession IDs",
    page=(
        "README, 'Files and variables', "
        "CRISPRi_automation_Pputida_proteomic_Top3_peptide_quantification_method_data"
        ".csv"
    ),
    note="why this family needs a UniProt-accession crosswalk where the panel family "
    "needs reconcile_locus_tags: the campaign matrix is keyed by ACCESSION and the "
    "panel sheet by gene symbol / locus tag",
)
CAMPAIGN_TITER_IN_MG_PER_L = _dryad_readme(
    DRYAD_TITER_UNIT,
    "The file also contains the isoprenol titer measured in mg/L for each line.",
    page=(
        "README, 'Files and variables', "
        "CRISPRi_automation_Pputida_proteomic_Top3_peptide_quantification_method_data"
        ".csv"
    ),
    note="the Dryad table restates the Source Data's titer per CULTURE; it is read "
    "only as the cross-source oracle of the titer family and is never stored twice",
)
CAMPAIGN_LINE_COLUMN_RULE = _dryad_readme(
    "Line_name",
    "Line: (str) The line name with the replicate ID removed",
    page=(
        "README, 'Files and variables', "
        "CRISPRi_automation_Pputida_proteomic_Top3_peptide_quantification_method_data"
        ".csv"
    ),
    note="MEASURED DEVIATION FROM THIS SENTENCE, which is why the strain key is "
    "derived from Line_name and not from Line: in DBTL6 every Line cell carries a "
    "PRT1093_ prefix that its Line_name does not (177 non-control and 12 control rows, "
    "measured 2026-10-09), so Line is not Line_name minus the replicate id there. "
    "PRT1093 appears in no mirrored byte of this paper, so it is recorded and not "
    "typed as a perturbation",
)
CAMPAIGN_GENE_LIST_IS_THE_LINE = _dryad_readme(
    "Line",
    "Genes_targeted_for_CRISPRi:(str)A list of genes targeted for CRISPR interference "
    "(CRISPRi) for each line in the experiment.",
    page=(
        "README, 'Files and variables', "
        "CRISPRi_automation_Pputida_proteomic_metadata.csv"
    ),
    note="measured on the pinned bytes, the column is not a list but the Line cell "
    "verbatim (1,407 of 1,407 non-control rows, and the 12 DBTL6 control rows); it is "
    "blank for the 78 control rows of DBTL0-5. It is asserted against Line and carries "
    "nothing the construct name does not",
)

# --- the released differential proteomics (the fold-change family) -------------
_Q_FIG5B = (
    "b Statistically significant Log2(Fold-change) values (paired two-tailed "
    "Student’s $T -$ test, $p { < } 0 . 0 5 )$ for selected sgRNAs of electron "
    "transport chain complexes and selected proteins from the TCA cycle. POI "
    "$\\mathsf { L o g } _ { 2 } \\mathsf { F C }$ cells are bolded. Source data are "
    "provided in the Source Data file."
)
_Q_SI_FIG9_TITLE = (
    "Supplementary Figure 9: Heatmaps of relevant pathways among best performing "
    "sgRNAs compared to the non-target control strain"
)
_Q_FIG6B = (
    "b Volcano plots of two KO strains, ΔPP_0812 and ΔPP_0815, showing "
    "fold-change differences in global protein expression with a target vs. non-target "
    "sgRNA."
)
_Q_FIG6B_BOTH_DELETIONS = (
    "Furthermore, both deletion strains displayed broad changes in the proteome that "
    "were not reflected in other cytochrome ${ \\mathsf { b o } } _ { 3 }$ subunit "
    "deletions when comparing non-target and target sgRNAs (e.g., ΔPP_0812) "
    "(Fig. 6b)."
)
_Q_STATS_TTEST = (
    "Where applicable, statistical significance was determined using a paired "
    "Student’s T-test, where $p { < } 0 . 0 5$ ."
)
_Q_SI_FIG10_TITLE = (
    "Supplementary Figure 10: Heatmap of significantly changed proteins present in all "
    "25 best performing CRISPRi strain"
)
_Q_SI_FIG10 = (
    "Depicts the protein levels that were significantly changed (paired two-tailed "
    "Student’s T-test, $\\mathsf { p } < 0 . 0 5$ ) across 25 of the best "
    "performing sgRNA combinations."
)

CHASSIS_GENOTYPE = _paper(
    "KT2440 ΔphaABC, ΔmvaB, ΔhbdH, 4,538,575Δ86,812 (Δzwf, ΔglZ, ΔliuC)",
    _Q_CHASSIS,
    page=_METHODS_STRAIN,
    note="the Methods list; the Fig. 2 caption gives a DIFFERENT list for the same "
    "strain (ΔldhA present, zwf/glZ written zwfB/gntZ) and is recorded beside it",
)
CHASSIS_GENOTYPE_FIG2 = _paper(
    "KT2440 ΔphaABC, ΔmvaB, ΔhbdH, ΔldhA, ΔzwfB, ΔgntZ, ΔliuC",
    _Q_CHASSIS_FIG2,
    page="Fig. 2 caption",
    note="the only statement that names ΔldhA and that spells zwfB / gntZ; the stated "
    "86,812 bp span contains PP_4042 (zwfB), PP_4043 (gntZ) and PP_4066 (liuC) while "
    "the bare symbol zwf resolves to PP_5351 outside it, which is what identifies the "
    "Methods' zwf as zwfB",
)
CHASSIS_GENOTYPE_SI = _si(
    "KT2440 ΔphaABC, ΔmvaB, ∆hbdH, 4,538,575 Δ86,812",
    _Q_SI_TABLE2,
    page="Supplementary Table 2: List of Pseudomonas putida strains constructed in "
    "this study (IY1449b, JBx_273364)",
)
PRODUCTION_STRAIN_SOURCE = _paper(
    PRODUCTION_STRAIN,
    _Q_PATHWAY_PLASMID,
    page=_METHODS_STRAIN,
    note="IY1452b IS IY1449b carrying pIY670, so the pathway is a perturbation on the "
    "IY1449b background rather than part of the background",
)
PATHWAY_SOURCE = _paper(
    "isoprenol via the IPP-bypass mevalonate pathway",
    _Q_PATHWAY,
    page="Methods, 'Pathway overview'",
)
PIY670_PARTS = _plasmid(
    _Q_PIY670,
    _Q_PIY670,
    note="the part order is what the operon reading below is read from: araC-PBAD "
    "drives MvaSEf and MvaEEf up to the TrpoH terminator, and Ptrc1-O drives MKMm, "
    "PMDScHKQ and AphA. The source does not draw the operon boundaries in prose",
)
DCAS9_SOURCE = _plasmid(
    "dCas9",
    _Q_PIY989,
    note="the CRISPRi vector pIY989 / pDBTL3-6_Vector; the sgRNA array replaces the "
    "sfGFP placeholder. The spacer sequences are NOT released in any mirrored file "
    "(Supplementary Data 2's PP_*_gRNA oligos are the Cpf1 knockout guides, paired "
    "with PP_*_Repair oligos), so CrisprConstruct.guide_sequence stays None",
)
MEDIUM = _paper(
    "M9-NREL",
    _Q_MEDIUM,
    page=_METHODS_CULTURE,
    note="served as the shared media.M9_NREL_CARRUTHERS2025 object, whose Teknova "
    "T1001 trace-metal amount is an open gap recorded in media.py",
)
TEMPERATURE_C = _paper(24.0, _Q_TEMPERATURE, page=_METHODS_CULTURE)
DURATION_HOURS = _paper(48.0, _Q_INDUCER, page=_METHODS_CULTURE)
CULTURE_FORMAT = _paper(
    {
        "vessel": "48-well BioLector flower plate",
        "working_volume_ul": 1500.0,
        "shaking_rpm": 1000.0,
    },
    _Q_CULTURE,
    page=_METHODS_CULTURE,
    note="the vessel, 1.5 mL working volume, 1000 RPM shaking and 48 h endpoint are "
    "CultureEnvironment fields, and ProductTiterExperiment.environment is now annotated "
    "CultureEnvironment, so production_environment() carries them on the record; the "
    "proteome family's slot is still Environment, which drops them on dump",
)
AEROBICITY = _paper(
    "aerobic",
    _Q_CULTURE,
    page=_METHODS_CULTURE,
    note="a flower plate shaken at 1000 RPM under a gas-permeable seal, the standard "
    "aerobic configuration; the source never uses the word",
)
INDUCER_G_PER_L = _paper(2.0, _Q_INDUCER, page=_METHODS_CULTURE)
N_REPLICATES = _paper(
    3,
    _Q_TRIPLICATE,
    page=_METHODS_STATS,
    note="biological triplicates; seven (construct, cycle) strains carry six replicates "
    "in the released Source Data and store the count they actually have",
)
UNCERTAINTY = _paper(
    "sample_sd",
    _Q_ERRORBARS,
    page=_METHODS_STATS,
    note="the stored uncertainty is the sample SD across a strain's replicates, so "
    "SE = SD / sqrt(n); Fig. 4d is the one box-and-whisker panel and this loader reads "
    "no value from it",
)
QUANTIFICATION = _paper(
    "GC-FID", _Q_GCFID, page="Methods, 'Quantification of isoprenol using GC-FID'"
)
CONTROL_N = _paper(
    {"DBTL0": 18, "DBTL1-6": 12},
    _Q_CONTROL_N,
    page="Fig. 2 caption",
    note="the control cultures per cycle; they are the per-cycle phenotype_reference, "
    "not records",
)
TOP3 = _paper(
    PROTEOME_MEASUREMENT_TYPE,
    _Q_TOP3,
    page="Methods, 'Proteomics analysis'",
    note="what one stored abundance IS: the mean, over a sample's replicates, of the "
    "released per-replicate Top3 signal",
)
DIANN_DATABASE = _paper(
    "P. putida KT2440 UniProt proteome + heterologous proteins + common contaminants",
    _Q_DIANN_DB,
    page="Methods, 'Proteomics analysis'",
    note="the sourced reason non-host keys appear in the released abundance table and "
    "are dropped from the per-record abundance map",
)
PROTEOME_PANEL_BACKGROUND = _paper(
    f"{PRODUCTION_STRAIN} Δ{PROTEOME_BACKGROUND_DELETION}",
    _Q_OTS,
    page="Results, 'Investigating off-target effects by candidate sgRNAs via gene "
    "knockout'",
    note="the SI caption writes the same strain as the chromosomal designation "
    f"'{_Q_SI_OTS_BACKGROUND}'; both mean IY1449b ΔPP_0815 carrying pIY670, since "
    "Supplementary Fig. 13d reports isoprenol titers for these strains",
)
PROTEOME_N_REPLICATES = _si(3, _Q_SI_TRIPLICATE, page="Supplementary Figure 13 caption")
PROTEOME_REPEATED_CULTURES = _si(
    3,
    _Q_SI_THREE_CULTURES,
    page="Supplementary Figure 13 caption",
    note="why PP_0977 and PP_1638 appear as several samples: each is an independent "
    "culture of the same genotype and is kept as its own record, never averaged",
)
NO_EXCLUSION = _paper(
    True,
    _Q_NO_EXCLUSION,
    page=_METHODS_STATS,
    note="the authors' pass/fail CRISPRi filter shaped model training only, which is "
    "why every released culture becomes a record and the flag is carried in preprocess/",
)

KO_PANEL_DESIGN = _paper(
    {"arms": [ARM_KO_TARGET, ARM_KO_NONTARGET]},
    _Q_KO_PANEL_DESIGN,
    page="Results, 'Investigating off-target effects by candidate sgRNAs via gene "
    "knockout'",
    note="what the two new Figure 6a arms ARE: a KO strain carrying the sgRNA that "
    "previously knocked the SAME gene down (Target), and the same KO strain carrying "
    "a non-targeting sgRNA (Non-target), which perturbs no gene and is the reference. "
    "The sheet's third arm, CRISPRi, is the un-deleted knockdown strain and is a "
    "re-export of Figure 4b",
)
KO_PANEL_N = _paper(
    3,
    _Q_KO_PANEL_FIG6A,
    page="Fig. 6 caption",
    note="the Supplementary Fig. 11 caption restates it as "
    f"'{_Q_SI_TRIPLICATE}', and Supplementary Figure 11 is row-for-row identical to "
    "Figure 6a on the pinned bytes",
)
KO_MULTI_GENE_GUIDE = _paper(
    {"PP_0812-15": "PP_0815"},
    _Q_KO_0812_15_GUIDE,
    page="Results, 'Investigating off-target effects by candidate sgRNAs via gene "
    "knockout'",
    note="the ONE multi-gene KO background of Figure 6a whose Target sgRNA the "
    "released sheet does not name: no plasmid of Supplementary Data 3 carries a "
    "PP_0812-15 spacer and Supplementary Data 2's only PP_0812-15 entry is the Cpf1 "
    "PP_0812-15_Repair oligo, so this sentence is the sole statement of it. A new "
    "multi-gene background absent from this map stops the build",
)
OXIDASE_COMPLEX = _paper(
    {"PP_0812-15": ("PP_0812", "PP_0813", "PP_0814", "PP_0815")},
    _Q_OXIDASE_COMPLEX,
    page="Results, 'Rapid characterization of isoprenol production and gene "
    "downregulation'",
    note="what the hyphen in a PP_xxxx-yy designation means: an inclusive locus-number "
    "range. Supplementary Table 5 corroborates the four members by building "
    "IY1449b ΔPP_0812-15 as PP_0813-15 deleted from IY1449b ΔPP_0812",
)
KO_STRAIN_CONSTRUCTION = _paper(
    "Cpf1-mediated recombineering from IY1449b",
    _Q_KO_CONSTRUCTION,
    page="Methods, 'Generation of stable gene knockouts'",
    note="every KO background of Figure 6a and Figure 6d is a row of Supplementary "
    f"Table 2 ('{_Q_SI_TABLE2_KO_STRAINS}'); the JBEI part id of each is carried in "
    "preprocess/ko_backgrounds.csv rather than on the perturbation, so a deletion "
    "object built here is bit-identical to the PP_0815 deletion the proteome family "
    "already stores",
)
KO_ARRAY_RESULT = _paper(
    {"PP_0368_PP_0812-15_KO with PP_0528_PP_0815": 651.0, "PP_0528_PP_0815": 580.0},
    _Q_KO_ARRAY_RESULT,
    page="Results, 'Investigating off-target effects by candidate sgRNAs via gene "
    "knockout'",
    note="the Results text's own integer mg/L for one NEW Figure 6d record and one "
    "already-stored Figure 4b array, which is the cross-source oracle for this panel",
)
OFFTARGET_TITER_SOURCE = _si(
    OFFTARGET_CANDIDATE_TARGETS,
    _Q_SI_FIG13D,
    page="Supplementary Figure 13 caption, panel d",
    note="the titer arm of exactly the panel whose proteome is "
    f"{SHEET_PROTEOME}: 20 released groups, 18 off-target sgRNAs over the "
    f"{OFFTARGET_CANDIDATE_TARGETS} targets of Supplementary Table 1 "
    f"('{_Q_SI_TABLE1_OFFTARGETS}') plus the Target and Non-Target anchors",
)
OVEREXPRESSION_OPERON_SOURCE = _paper(
    OVEREXPRESSION_OPERONS,
    _Q_OVEREXPRESSION_OPERONS,
    page="Methods, 'Overexpression of candidate genes'",
    note="the loci are PP_2208-PP_2209 and PP_2791-PP_2794. The Source Data labels the "
    f"second strain '{OVEREXPRESSION_SHEET_LABELS['pSTABL2']}', a digit transposition "
    "contradicted by five mirrored statements: this sentence, the Results' "
    f"'{_Q_OVEREXPRESSION_RESULTS}', the Supplementary Fig. 12 caption's panels c and "
    "d, the Supplementary Fig. 10 caption's levulinate note, and Supplementary Data "
    "3's own plasmid composition. The measured proteome of the same samples proves it",
)
OVEREXPRESSION_NATIVE_COPIES = _paper(
    PRODUCTION_STRAIN,
    _Q_OVEREXPRESSION_TRANSFORMED,
    page="Methods, 'Overexpression of candidate genes'",
    note="the overexpression strains are IY1449b carrying pIY670 (so the five pathway "
    "perturbations stand) plus an extra plasmid-borne copy of a NATIVE operon, which "
    "is a GeneAdditionPerturbation with is_heterologous=False and the locus tag of the "
    "real gene",
)
PSTABL1_PLASMID = _plasmid(
    _Q_PSTABL1_PLASMID,
    _Q_PSTABL1_PLASMID,
    note="Supplementary Data 3 row pSTABL1 / JBx_273326. The plasmid NAMES are swapped "
    "between this table and the Source Data, which labels its pSTABL1 samples "
    f"'{OVEREXPRESSION_SHEET_LABELS['pSTABL1']}' and measures PP_2208 and PP_2209 in "
    "them. Neither name is stored on a perturbation: what is stored is the operon the "
    "sample's own measured proteome names",
)
PSTABL2_PLASMID = _plasmid(
    _Q_PSTABL2_PLASMID,
    _Q_PSTABL2_PLASMID,
    note="Supplementary Data 3 row pSTABL2 / JBx_273327; the mirror image of the "
    "swap recorded on PSTABL1_PLASMID",
)
RFP_CONTROL_PLASMID = _plasmid(
    _Q_RFP_CONTROL_PLASMID,
    _Q_RFP_CONTROL_PLASMID,
    note="Supplementary Data 3 row pTE519 / JBx_266188, the 'RFP control (JBx_266188)' "
    f"of '{_Q_OVEREXPRESSION_CULTURE}'. It is the overexpression panel's Control "
    "group, which is a phenotype_reference and therefore carries no genotype, so the "
    "mCherry marker is recorded here rather than typed as a perturbation",
)
OVEREXPRESSION_DURATION_GAP = _paper(
    None,
    _Q_OVEREXPRESSION_CULTURE,
    page="Methods, 'Overexpression of candidate genes'",
    note="the overexpression panel's own culturing sentence states the endpoint as "
    "'over 48 h' while the shared culturing Methods state 'Following 48 h of "
    "production', so duration_hours is a typed ProvenanceGap on these records rather "
    "than the shared 48.0. Everything else these records carry (M9-NREL, 24 C, the "
    "flower plate, the L-arabinose induction) the sentence leaves to the shared "
    "protocol and does not contradict",
)
INDUCER_UNIT_GAP = _paper(
    OVEREXPRESSION_INDUCED_LEVELS,
    _Q_OVEREXPRESSION_SALICYLATE,
    page="Methods, 'Overexpression of candidate genes'",
    note="why only the UNINDUCED arm of Supplementary Figure 12 is loaded. The inducer "
    "is salicylic acid and its concentrations are released as bare numbers in the "
    "'Inducer concentration' column with NO unit. Measured over the mirror: 'salicyl' "
    "occurs once in paper.md (this sentence, with no dose), 'inducer' once in si1.md "
    "(the Supplementary Fig. 12 caption's 'under various inducer concentrations', with "
    "no unit), and the dose series appears nowhere else in paper.md, si1.md, si2.md, "
    "si3.md or si8.md. The deferral was followed: Supplementary Data 3 refers the "
    "NagR-pNagAa vector to Supplementary Reference 1 (Yunus et al., mirrored as "
    "yunusPredictiveCRISPRmediatedGene2026), whose Methods state only an L-arabinose "
    "dose and no salicylate dose. ConcentrationUnit admits no unitless member and a "
    "Concentration needs a value WITH a unit, so the six induced levels cannot be "
    "typed; collapsing them to one dose basis would assert that one condition yielded "
    "six different titers. At level 0 no inducer was added and no unit is needed",
)
CONTROL_PROTEOME_DEFERRAL = _si(
    None,
    _Q_SI_FIG15,
    page="Supplementary Figure 15 caption",
    note=f"why {SHEET_CONTROL_PROTEOME} is NOT loaded here. Its six columns are the "
    "percent-of-total Top3 basis, established by elimination over the three columns "
    f"{SHEET_OVEREXPRESSION_PROTEOME} releases from the same pipeline: all 540 of its "
    "cells lie in 0.00037341..3.43366 with zero negatives, while Top_3pep_counts_mean "
    "runs 0..1.0027e8 and log10_%_abundance runs -2.5027..1.1373, leaving "
    "'%_of protein_abundance_Top3-method' (0.0031..13.7169) as the only basis whose "
    "range contains it; the caption's three rank claims also hold on that reading "
    "(column means PMD* 2.480 highest, mvaE 1.214 second, mvaS 0.589 lowest of the "
    "five pathway proteins, dCas9 0.034). That is a DIFFERENT measurement_type from "
    "the raw Top3 signal every record of this family stores, and "
    "verify_protein_dataset's measurement_type_consistent rule exists to stop exactly "
    "that mixing, so serving it needs its own dataset class and the full adapter gate. "
    "Two further blockers: five of its six keys are the pIY670 pathway tokens, which "
    "are not loci of the pinned assembly and so fail the runner's "
    "protein_and_perturbed_locus_containment_assembly, and the sixth names the dCas9 "
    "effector, which this release carries on CrisprConstruct rather than as a gene "
    "with a systematic name. Its 90 cultures are the per-cycle controls whose titers "
    "are already this dataset's phenotype_reference (18, 12, 12, 12, 12, 12, 12 "
    "measured, matching CONTROL_N), so nothing it holds is lost silently",
)

FOLD_CHANGE_SCALE = _paper(
    FoldChangeScale.log2.value,
    _Q_FIG5B,
    page="Fig. 5 caption",
    note="the caption names the scale of the stored number: Log2(Fold-change). The "
    f"column headers of {SHEET_FOLD_CHANGE} and {SHEET_KO_FOLD_CHANGE} say the same "
    "thing in the file ('<target>_log2_FC' beside '<target>_log10_pval')",
)
FOLD_CHANGE_TEST = _paper(
    FOLD_CHANGE_MEASUREMENT_TYPE,
    _Q_FIG5B,
    page="Fig. 5 caption",
    note="the test behind every stored p-value, stated identically by the Fig. 6c "
    f"caption and by the Methods' '{_Q_STATS_TTEST}'. One measurement_type for both "
    "sheets, because both report the same statistic of the same DIA-NN pipeline",
)
FOLD_CHANGE_P_VALUE_SCALE = _source_data(
    NEG_LOG10_P_VALUE_SUFFIX,
    "PP_0368_log10_pval",
    page=f"Source Data '{SHEET_FOLD_CHANGE}', header row",
    note="the released column is UNSIGNED -log10(p): measured over both sheets it runs "
    "1.30103425637741 to 19.3125325679514 with no negative value, and 10**-1.30103 = "
    "0.049999, so the sheets are pre-filtered at the caption's p<0.05. The stored "
    "number is the probability, recovered by the exact arithmetic p = 10**-x whose "
    "round trip :func:`p_value_from_neg_log10` asserts",
)
FOLD_CHANGE_REFERENCE_BASIS = _si(
    "the non-target control strain",
    _Q_SI_FIG9_TITLE,
    page="Supplementary Information contents, Supplementary Figure 9",
    note=f"the denominator of every {SHEET_FOLD_CHANGE} column, in the source's own "
    f"words. What binds this caption to that sheet is MEASURED, not assumed: the "
    f"Source Data's own '{SHEET_FOLD_CHANGE_POINTER}' sheet holds exactly one cell, "
    f"'{FOLD_CHANGE_POINTER_CELL}', so the figure this caption describes IS that "
    "sheet; :func:`assert_fold_change_pointer` stops the build if the cell changes",
)
KO_FOLD_CHANGE_REFERENCE_BASIS = _paper(
    "a non-targeting control sgRNA in a knockout background",
    _Q_KO_PANEL_FIG6A,
    page="Fig. 6 caption",
    note=f"the denominator of the two {SHEET_KO_FOLD_CHANGE} columns that are stored, "
    "in the source's own words. The Fig. 6a caption is the titer arm of exactly these "
    f"strain pairs, and the Fig. 6b caption states the proteome contrast itself: "
    f"'{_Q_FIG6B}'",
)
KO_FOLD_CHANGE_SOURCE = _paper(
    KO_FOLD_CHANGE_CONTRASTS,
    _Q_FIG6B,
    page="Fig. 6 caption",
    note=f"the only verbatim statement of a {SHEET_KO_FOLD_CHANGE} contrast, and it "
    "names two KO strains and which two. The sheet releases ten columns, so eight are "
    "ledgered unbuilt rather than given a guessed denominator (see "
    ":data:`KO_FOLD_CHANGE_UNSOURCED_REASONS`)",
)
BEST_ARRAY_HEATMAP_REFUSAL = _si(
    None,
    _Q_SI_FIG10,
    page="Supplementary Figure 10 caption",
    note=f"why {SHEET_BEST_ARRAY_HEATMAP} is NOT loaded. Measured on the pinned "
    "workbook: 26 keys x 25 strain columns, 650 cells with no empty cell, no p-value "
    "column at all, values signed from -7.05757655263963 to 3.92161053900316 with "
    "median -1.0118185429312652, and 24 of the 26 keys title-cased protein symbols "
    "(AcsA1, CyoA, ValS) rather than locus tags. ``fold_change_scale`` is a required "
    "field and no mirrored statement names a scale or a unit for this sheet: the "
    f"caption says only '{_Q_SI_FIG10_TITLE}', and its body says only that the values "
    "are 'protein levels that were significantly changed'. A log2 reading is a "
    "HYPOTHESIS the release does not support, and storing it would assert a scale the "
    "source never states, so the sheet is refused with its measurement recorded",
)


#: Every other ``SHEET_KO_FOLD_CHANGE`` column and the measurement that refuses it. The
#: sheet releases ten columns and the Fig. 6 caption names a contrast for two, so these
#: eight have no sourced denominator; each reason is a measurement on the pinned bytes.
KO_FOLD_CHANGE_UNSOURCED_REASONS: dict[str, str] = {
    "Control": (
        "the sheet's own label for a column no mirrored statement describes. 'Control' "
        "is also the name of the campaign's non-targeting control line in Figure 4b, "
        "so the header cannot even be read as a strain designation without choosing "
        "between a control strain and a contrast against one"
    ),
    "PP_0368": (
        "a Figure 6a KO-background label, but no mirrored statement names it as a "
        "Figure 6b contrast; the Fig. 6 caption names two KO strains and this is "
        "neither"
    ),
    "PP_0751": (
        "a Figure 6a KO-background label, but no mirrored statement names it as a "
        "Figure 6b contrast"
    ),
    "PP_0813": (
        "a Figure 6a KO-background label, but no mirrored statement names it as a "
        "Figure 6b contrast"
    ),
    "PP_0814": (
        "a Figure 6a KO-background label, but no mirrored statement names it as a "
        "Figure 6b contrast"
    ),
    "PP_0812_15": (
        "a Figure 6a KO-background label (the sheet writes with an underscore what "
        "Figure 6a writes PP_0812-15), and the nearest statement is the Results' "
        f"'{_Q_FIG6B_BOTH_DELETIONS}'. Reading that as this column means resolving the "
        "pronoun 'both deletion strains' to the previous sentence's pair against a "
        "caption that says 'two KO strains' and names which two, so the contrast would "
        "be inferred, not sourced"
    ),
    "PP_0751_PP_0812": (
        "not a KO background at all: measured on the pinned workbook, PP_0751_PP_0812 "
        "is a BUILT two-sgRNA CRISPRi array, released as a Figure 4b line name and "
        "listed in Supplementary Table 4's DBTL1-6 array list, so the column's "
        "denominator could be the campaign's non-targeting control or a KO carrying "
        "one, and the release states neither"
    ),
    "PP_1317_PP_0812": (
        "matches no released strain: measured on the pinned workbook it is absent from "
        "Figure 4b's line names and from Figure 6a's KO-background labels, and it is "
        "not in Supplementary Table 4's array list either, so neither the numerator "
        "strain nor the denominator can be sourced"
    ),
}

#: The five heterologous pIY670 genes. ``token`` is the verbatim part name in the
#: plasmid description, ``symbol`` the ``Protein`` id the released proteomics uses, and
#: ``entry`` that protein's UniProt entry name in the same file -- the mirrored evidence
#: for ``organism``. ``promoter`` is read from the part order (see :data:`PIY670_PARTS`).
PATHWAY_GENES: tuple[dict[str, str | None], ...] = (
    {
        "token": "MvaSEf",
        "symbol": "Mvas",
        "entry": "HMGCS_ENTFL",
        "accession": "Q9FD71",
        "organism": "Enterococcus faecalis",
        "promoter": "PBAD",
        "variant": None,
    },
    {
        "token": "MvaEEf",
        "symbol": "Mvae",
        "entry": "Q9FD70_ENTFL",
        "accession": "Q9FD70",
        "organism": "Enterococcus faecalis",
        "promoter": "PBAD",
        "variant": None,
    },
    {
        "token": "MKMm",
        "symbol": "Mvk",
        "entry": "Q8PW39_METMA",
        "accession": "Q8PW39",
        "organism": "Methanosarcina mazei",
        "promoter": "Ptrc1-O",
        "variant": None,
    },
    {
        "token": "PMDScHKQ",
        "symbol": "Mvd1",
        "entry": "MVD1_YEAST",
        "accession": "P32377",
        "organism": "Saccharomyces cerevisiae",
        "promoter": "Ptrc1-O",
        "variant": "HKQ",
    },
    {
        "token": "AphA",
        "symbol": "Apha",
        "entry": "APHA_ECOLI",
        "accession": "P0AE22",
        "organism": "Escherichia coli",
        "promoter": "Ptrc1-O",
        "variant": None,
    },
)

#: The chassis deletions this loader types as alleles: the source's own symbol, the
#: designation verbatim, which quote states it, and whether it sits in the stated span.
CHASSIS_ALLELES: tuple[dict[str, Any], ...] = (
    {"symbol": "phaA", "allele": "ΔphaABC", "in_span": False, "both_sources": True},
    {"symbol": "phaB", "allele": "ΔphaABC", "in_span": False, "both_sources": True},
    {"symbol": "mvaB", "allele": "ΔmvaB", "in_span": False, "both_sources": True},
    {"symbol": "hbdH", "allele": "ΔhbdH", "in_span": False, "both_sources": True},
    {"symbol": "ldhA", "allele": "ΔldhA", "in_span": False, "both_sources": False},
    {"symbol": "zwfB", "allele": "ΔzwfB", "in_span": True, "both_sources": False},
    {"symbol": "gntZ", "allele": "ΔgntZ", "in_span": True, "both_sources": False},
    {"symbol": "liuC", "allele": "ΔliuC", "in_span": True, "both_sources": True},
)
#: The designation of ``ΔphaABC`` whose gene the assembly does not annotate.
CHASSIS_UNMAPPED_SYMBOLS: tuple[str, ...] = ("phaC",)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/carruthersAutomationMachineLearning2025``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pmc_cloud_key(filename: str) -> str:
    """Bucket key of one supplementary file in the PMC Article Datasets bucket."""
    return f"{PMC_PREFIX}/{filename}"


def dryad_deposits() -> tuple[tuple[str, str, str], ...]:
    """``(relpath, role, sha256)`` of the four files the owner deposited by hand.

    The zip is the bytes that ARRIVED and the three members are what loaders read, so
    all four are recorded: a rebuild re-runs the recipe, gets the zip, and verifies
    every member. The digests are read HERE rather than captured in a module constant,
    so a test that re-points the pins at a synthetic deposit re-points this too.
    """
    return (
        (DRYAD_ZIP_REL, ROLE_RAW_DATA, DRYAD_ZIP_SHA256),
        (DRYAD_TOP3_REL, ROLE_RAW_DATA, DRYAD_TOP3_SHA256),
        (DRYAD_METADATA_REL, ROLE_RAW_DATA, DRYAD_METADATA_SHA256),
        (DRYAD_README_REL, ROLE_SI_DATA, DRYAD_README_SHA256),
    )


def dryad_artifact_records(root: Path) -> list[ArtifactRecord]:
    """The four manual-deposit records, verified in place under ``root``.

    The owner retrieved these bytes in a browser and deposited them, so this function
    never copies or fetches: it asserts each file is present with its pinned sha256 and
    describes it. An absent or altered file raises WITH the manual recipe, which is the
    only way to produce it again.
    """
    records: list[ArtifactRecord] = []
    for relpath, role, expected in dryad_deposits():
        path = root / relpath
        if not path.exists():
            raise RuntimeError(
                f"{path} is missing; the Dryad campaign proteome is a manual deposit. "
                f"MANUAL RECIPE: {DRYAD_MANUAL_RECIPE}"
            )
        verify_sha256(path, expected)
        records.append(
            ArtifactRecord(
                path=relpath,
                role=role,
                bytes=path.stat().st_size,
                sha256=expected,
                source=f"https://doi.org/{DRYAD_DOI}",
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.manual_browser,
                    source_url=f"https://doi.org/{DRYAD_DOI}",
                    retriever="manual",
                    params={
                        "retrieval_command": DRYAD_MANUAL_RECIPE,
                        "dataset": (
                            f"Dryad {DRYAD_DOI}, version of 2025-08-20 (the Dryad "
                            f"download is named `{DRYAD_ZIP_FILENAME}`)"
                        ),
                        "retrieved_by": DRYAD_RETRIEVED_BY,
                        "deposit_note": DRYAD_DEPOSIT_NOTE,
                        "deposit_record": f"{DRYAD_DIR_REL}/DEPOSIT.md",
                        "checksums": f"{DRYAD_DIR_REL}/SHA256SUMS.txt",
                    },
                    sha256=expected,
                    retrieved_at=DRYAD_RETRIEVED_AT,
                ),
            )
        )
    return records


def deposit_raw_mirror(
    *,
    source_data_path: str | Path,
    targets_path: str | Path,
    retrieved_at: str = SI_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror (Source Data + Supplementary Data 1) and its ``manifest.json``.

    Idempotent by sha256: an existing mirror file whose digest matches is left alone and
    a differing one raises rather than being overwritten. Both files are verified BEFORE
    anything is written, so a refusal leaves no partial deposit. Both come from the PMC
    Article Datasets bucket, which is directly scriptable: the recorded retrieval
    re-runs as-is and reproduced both pinned digests on ``retrieved_at``.
    """
    root = raw_mirror_dir(data_root)
    deposits = (
        (source_data_path, SOURCE_DATA_REL, SOURCE_DATA_FILENAME, SOURCE_DATA_SHA256),
        (targets_path, TARGETS_REL, TARGETS_FILENAME, TARGETS_SHA256),
    )
    for source, _, _, expected in deposits:
        verify_sha256(source, expected)
    files: list[ArtifactRecord] = []
    for source, relpath, filename, expected in deposits:
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != expected:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(source, dest)
        key = pmc_cloud_key(filename)
        url = f"https://pmc-oa-opendata.s3.amazonaws.com/{key}"
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_SI_DATA,
                bytes=dest.stat().st_size,
                sha256=expected,
                source=url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.pmc_cloud,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.pmc_cloud_object",
                    params={"key": key},
                    sha256=expected,
                    retrieved_at=retrieved_at,
                ),
            )
        )
    files.extend(dryad_artifact_records(root))
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=files,
        si_data_sources=[
            f"https://pmc-oa-opendata.s3.amazonaws.com/{pmc_cloud_key(SOURCE_DATA_FILENAME)}",
            f"https://pmc-oa-opendata.s3.amazonaws.com/{pmc_cloud_key(TARGETS_FILENAME)}",
            f"https://doi.org/{DRYAD_DOI}",
            *(
                f"https://www.ebi.ac.uk/pride/archive/projects/{accession}"
                for accession in PRIDE_ACCESSIONS.values()
            ),
        ],
        si_expected=[
            "Source Data (MOESM9) -- deposited; the per-replicate isoprenol titers "
            "(Figure 1D / Figure 4b) and the off-target-study proteome matrix "
            f"({SHEET_PROTEOME}) are what the two loaders consume",
            "Supplementary Data 1 (MOESM4) -- deposited; consumed only as the "
            "cross-source oracle for the per-target DBTL0 mean titer. It lists 120 "
            "targets while 121 were screened: PP_3365 is absent from it",
            "Supplementary Data 2-4 (MOESM5-7) -- oligonucleotides, plasmids and the "
            "DBTL1-6 CRISPRi array list. NOT deposited: no loader reads them, and the "
            "array list covers cycles 1 and 3-6 only (no DBTL2), so the Source Data's "
            "own cycle column is the authority. Supplementary Data 3 is quoted for "
            "pIY670's part composition from the torchcell-library mirror",
            "the CRISPRi sgRNA SPACER sequences were never released; Supplementary "
            "Data 2's PP_*_gRNA entries are the Cpf1 knockout guides",
            f"Dryad {DRYAD_DOI} {DRYAD_VERSION} -- the processed campaign proteomics "
            f"({DRYAD_TOP3_FILENAME}, 29,700,365 B; {DRYAD_METADATA_FILENAME}, 333,287 "
            f"B; {DRYAD_README_FILENAME}, 5,768 B). DEPOSITED "
            f"{DRYAD_RETRIEVED_AT} by hand under {DRYAD_DIR_REL}/, with the arrival "
            f"zip beside them, because datadryad.org serves an Anubis JavaScript "
            "proof-of-work challenge to scripts (measured 2026-10-07: "
            "/downloads/file_stream/<id> returns the challenge page with HTTP 200, "
            "/api/v2/files/<id>/download returns HTTP 401 'must have current bearer "
            f"token'). MANUAL RECIPE, re-run on rebuild: {DRYAD_MANUAL_RECIPE} The "
            "Top3 matrix and the metadata are what "
            "CampaignProteomeCarruthers2025Dataset consumes; the README is the "
            "authority for what one cell is",
            "the seven PRIDE projects "
            + ", ".join(f"{cycle} {acc}" for cycle, acc in PRIDE_ACCESSIONS.items())
            + " -- raw DIA mass-spectrometry files, enumerated above and not "
            "deposited: no loader consumes raw spectra",
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
# Genotype and environment builders
# --------------------------------------------------------------------------- #
def publication() -> Publication:
    """This paper, by PubMed id and DOI."""
    return Publication(
        pubmed_id=PMID,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
        doi=DOI,
        doi_url=f"https://doi.org/{DOI}",
    )


def chassis_background(genome: PPutidaKT2440Genome) -> BacterialStrainBackground:
    """IY1449b: KT2440 with the markerless in-frame deletions the sources state.

    Each allele's locus comes from the genome's own resolution of the SOURCE's symbol,
    never from a borrowed identifier. ``phaC`` is deliberately absent (see the module
    docstring), and so are the other 54 genes the stated 86,812 bp span removes.
    """
    span = GenomicSpan(
        chromosome=KT2440_REPLICON,
        start=SPAN_START,
        end=SPAN_END,
        assembly=KT2440_ASSEMBLY_SET,
    )
    for symbol in CHASSIS_UNMAPPED_SYMBOLS:
        resolved = genome.resolve_gene_name(symbol)
        if resolved.status is not GeneNameStatus.RETIRED:
            raise RuntimeError(
                f"{symbol!r} now resolves to {resolved.systematic_name} "
                f"({resolved.status}); it is documented as carrying no locus of this "
                "assembly and must become a typed allele"
            )
    alleles: list[BacterialBackgroundAllele] = []
    for entry in CHASSIS_ALLELES:
        symbol = str(entry["symbol"])
        resolution = genome.resolve_gene_name(symbol)
        if (
            resolution.status
            not in (
                GeneNameStatus.CURRENT,
                GeneNameStatus.RENAMED,
                GeneNameStatus.NON_GENE_FEATURE,
            )
            or resolution.systematic_name is None
        ):
            raise RuntimeError(
                f"chassis symbol {symbol!r} does not resolve to a {KT2440_ASSEMBLY_SET} "
                f"locus ({resolution.status}); the background cannot be typed"
            )
        source = CHASSIS_GENOTYPE if entry["both_sources"] else CHASSIS_GENOTYPE_FIG2
        alleles.append(
            BacterialBackgroundAllele(
                systematic_gene_name=resolution.systematic_name,
                gene_namespace=KT2440_NAMESPACE,
                gene_name=symbol,
                allele_name=str(entry["allele"]),
                edit=AlleleEdit.full_deletion,
                functional=False,
                deleted_span=span if entry["in_span"] else None,
                provenance=[source],
            )
        )
    return BacterialStrainBackground(
        name=CHASSIS_STRAIN,
        reference_strain="KT2440",
        assembly_set=KT2440_ASSEMBLY_SET,
        parents=["KT2440"],
        construction=(
            "markerless in-frame deletions in KT2440; the production strain "
            f"{PRODUCTION_STRAIN} is {CHASSIS_STRAIN} carrying pIY670, which this "
            "dataset records as HeterologousPathwayPerturbations rather than as part "
            "of the background"
        ),
        genotype_statement=_Q_SI_TABLE2,
        alleles=alleles,
        provenance=[
            CHASSIS_GENOTYPE,
            CHASSIS_GENOTYPE_FIG2,
            CHASSIS_GENOTYPE_SI,
            PRODUCTION_STRAIN_SOURCE,
        ],
    )


def chassis_reference(genome: PPutidaKT2440Genome) -> AssemblyReferenceGenome:
    """The assembly-pinned reference every record of this paper is written against."""
    return assembly_reference("KT2440", background=chassis_background(genome))


def pathway_perturbations() -> list[HeterologousPathwayPerturbation]:
    """The five heterologous pIY670 genes of the IPP-bypass mevalonate pathway.

    Each part token must appear in the quoted plasmid description, so the identifiers
    stay the source's own and a changed quote cannot drift away from them unnoticed.
    """
    parts = str(PIY670_PARTS.value)
    for gene in PATHWAY_GENES:
        token = str(gene["token"])
        if token not in parts:
            raise RuntimeError(
                f"pathway part {token!r} is not in the quoted pIY670 description "
                f"{parts!r}"
            )
    return [
        HeterologousPathwayPerturbation(
            systematic_gene_name=str(gene["token"]),
            perturbed_gene_name=str(gene["symbol"]),
            gene_namespace=KT2440_NAMESPACE,
            pathway_name=str(PATHWAY_SOURCE.value),
            source_organism=str(gene["organism"]),
            is_heterologous=True,
            localization="episomal_plasmid",
            construct_name="pIY670",
            variant=gene["variant"],
            promoter_name=gene["promoter"],
            copy_number=1.0,
        )
        for gene in PATHWAY_GENES
    ]


def crispri_perturbation(
    locus_tag: str, gene_name: str
) -> BacterialCrisprInterferencePerturbation:
    """One dCas9 knockdown of a KT2440 gene; the spacer was never released."""
    return BacterialCrisprInterferencePerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=gene_name,
        gene_namespace=KT2440_NAMESPACE,
        crispr=CrisprConstruct(
            effector=str(DCAS9_SOURCE.value), guide_sequence=None, n_guides=1
        ),
    )


def deletion_perturbation(
    locus_tag: str, gene_name: str
) -> BacterialDeletionPerturbation:
    """One chromosomal gene deletion of a Cpf1-built KO background.

    Built from exactly the three fields the proteome family's ``PP_0815`` deletion
    already carries, so the same genotype is the same object in both families: the
    JBEI registry part id of each background goes to ``preprocess/``, not here.
    """
    return BacterialDeletionPerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=gene_name,
        gene_namespace=KT2440_NAMESPACE,
    )


def native_copy_perturbation(
    locus_tag: str, gene_name: str
) -> GeneAdditionPerturbation:
    """An extra plasmid-borne copy of a NATIVE KT2440 operon member.

    ``is_heterologous`` is False and ``systematic_gene_name`` is the real locus tag, so
    the runner's ``host_perturbed_gene_set`` keeps it (``source_organism`` is the host's
    own species) and the containment gate checks it like any other host identifier.

    ``construct_name`` is deliberately ``None``: Supplementary Data 3 and the Source
    Data name the two overexpression plasmids in the OPPOSITE order (see
    :data:`PSTABL1_PLASMID`), so no plasmid name is asserted on a perturbation.

    NOT a ``HeterologousPathwayPerturbation``, although that class is the one carrying
    ``gene_namespace``. These operons are not part of the isoprenol pathway: typing
    them there would put them in a ``pathway_name`` they do not belong to and would
    make the family's ``heterologous_pathway_gene_counts`` rule accept 7 or 9 where it
    accepts 5 today, which is the rule that catches a production strain that lost its
    pathway. The cost is that ``GeneAdditionPerturbation`` declares no
    ``gene_namespace``, so the locus tag's namespace is recoverable from the record's
    ``genome_reference`` rather than stated on the leaf; that gap is raised in the PR.
    """
    return GeneAdditionPerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=gene_name,
        source_organism=SPECIES,
        is_heterologous=False,
        localization="episomal_plasmid",
        construct_name=None,
    )


def production_environment() -> CultureEnvironment:
    """M9-NREL at 24 C for 48 h in the flower plate, with the L-arabinose inducer.

    A ``CultureEnvironment``, since ``ProductTiterExperiment.environment`` is now
    annotated as one: a titer is read with its vessel, and the slot's narrowing is what
    makes :data:`CULTURE_FORMAT`'s vessel, working volume and shaking survive the dump
    (pydantic serializes by the DECLARED type, so the same object in an
    ``Environment``-typed slot loses them silently -- which is what the proteome family
    still does, and a test pins both halves).

    The production culture's kanamycin and gentamicin levels are stated only for the LB
    and passaging steps, so they are not recorded as doses here; that is said in the
    note rather than typed, because a ``ProvenanceGap`` must name a field that is
    ``None`` and ``perturbations`` is set.
    """
    if str(MEDIUM.value) not in M9_NREL_CARRUTHERS2025.name.replace(" ", "-"):
        raise RuntimeError(
            f"the served medium {M9_NREL_CARRUTHERS2025.name!r} is not the "
            f"{MEDIUM.value!r} the Methods name"
        )
    culture = CULTURE_FORMAT.value
    if not isinstance(culture, dict):
        raise RuntimeError(f"CULTURE_FORMAT.value is not a mapping: {culture!r}")
    return CultureEnvironment(
        media=M9_NREL_CARRUTHERS2025,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        culture_format=CultureFormat(
            vessel=str(culture["vessel"]),
            working_volume_ul=float(culture["working_volume_ul"]),
            shaking_rpm=float(culture["shaking_rpm"]),
            endpoint=EndpointRule.fixed_duration,
            provenance=[CULTURE_FORMAT],
        ),
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound("L-arabinose"),
                concentration=Concentration(
                    value=float(INDUCER_G_PER_L.value), unit=ConcentrationUnit.g_per_l
                ),
            )
        ],
        aerobicity=str(AEROBICITY.value),
        duration_hours=float(DURATION_HOURS.value),
    )


def overexpression_environment() -> CultureEnvironment:
    """The overexpression panel's environment: the shared protocol, endpoint gapped.

    The panel has its own one-sentence culturing statement, and it states the endpoint
    as "cultured for over 48 h" where the shared culturing Methods state "Following
    48 h of production". The two do not agree, so ``duration_hours`` is ``None`` with a
    typed ``ProvenanceGap`` rather than the shared 48.0. Everything else -- M9-NREL,
    24 C, the flower plate, the L-arabinose induction of pIY670 -- the sentence leaves
    to the shared protocol and does not contradict, so it is carried unchanged.

    Only the UNINDUCED cultures reach this environment: the inducer's unit is not
    released (:data:`INDUCER_UNIT_GAP`), and at level 0 no inducer was added, so the
    environment carries no salicylate dose and none is invented.
    """
    base = production_environment()
    return base.model_copy(
        update={
            "duration_hours": None,
            "provenance_gaps": [
                ProvenanceGap(
                    field="duration_hours",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note=str(OVEREXPRESSION_DURATION_GAP.note),
                )
            ],
        }
    )


def isoprenol_product() -> Any:
    """The product as a typed ``Compound`` through the shared compound-identity layer.

    ``isoprenol`` (3-methyl-3-buten-1-ol) has no row in the committed
    ``compound_identity_table.json``, so the resolver returns the honest typed absence:
    the canonical name with a ``ProvenanceGap`` on ``inchikey``. Curating the row needs
    a PubChem call against the committed input lists and is a human act, so it is raised
    in the PR rather than done here.
    """
    return resolved_compound("isoprenol")


def titer_phenotype(
    values: list[float], *, is_reference: bool
) -> ProductTiterPhenotype:
    """Mean titer over a strain's replicates with the sample SD, in ``ug/mL``.

    The released numbers are mg/L and 1 mg/L is exactly 1 ug/mL, so no arithmetic is
    applied to a source value. ``product_yield`` and ``productivity`` are typed
    absences: the campaign released neither.
    """
    n = len(values)
    if n < 2:
        raise RuntimeError(
            f"a titer record needs at least two replicates to carry a sample SD, got {n}"
        )
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    if not math.isfinite(mean) or not math.isfinite(sd):
        raise RuntimeError(f"non-finite titer statistics over {values!r}")
    return ProductTiterPhenotype(
        product=isoprenol_product(),
        titer=mean,
        titer_unit=ConcentrationUnit.ug_per_ml,
        titer_uncertainty=sd,
        titer_uncertainty_type=UncertaintyType(UNCERTAINTY.value),
        n_samples=n,
        sample_unit=SampleUnit.biological_replicate,
        quantification_method=str(QUANTIFICATION.value),
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=(
                    "the campaign reports titer only; neither a yield on substrate nor "
                    "a volumetric productivity is released for "
                    + ("the per-cycle control" if is_reference else "any strain")
                ),
            )
            for field in (
                "product_yield",
                "product_yield_unit",
                "productivity",
                "productivity_unit",
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Source Data readers
# --------------------------------------------------------------------------- #
def _sheet_rows(path: str, sheet: str) -> tuple[tuple[Any, ...], list[tuple[Any, ...]]]:
    """``(header, rows)`` of one Source Data sheet, read-only."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet not in book.sheetnames:
            raise RuntimeError(f"{osp.basename(path)} has no sheet {sheet!r}")
        stream = book[sheet].iter_rows(values_only=True)
        header = next(stream)
        return header, [row for row in stream if any(cell is not None for cell in row)]
    finally:
        book.close()


class TiterRow(BaseModel):
    """One released culture: its construct, DBTL cycle, replicate and titer."""

    construct_name: str
    cycle: int
    replicate: int
    is_control: bool
    titer_mg_per_l: float
    passed_filter: bool


def read_titer_rows(path: str) -> list[TiterRow]:
    """Every row of ``Figure 4b``, after asserting ``Figure 1D`` carries the same values.

    The two sheets are independent exports of one measurement; a mismatch means the
    pinned workbook is not the one this loader was written against, so it refuses.
    """
    _, rows = _sheet_rows(path, SHEET_TITER)
    _, alt = _sheet_rows(path, SHEET_TITER_ALT)
    if len(rows) != len(alt):
        raise RuntimeError(
            f"{SHEET_TITER} has {len(rows)} rows and {SHEET_TITER_ALT} has {len(alt)}"
        )
    for index, (a, b) in enumerate(zip(rows, alt, strict=True)):
        if str(a[0]) != str(b[0]) or float(a[3]) != float(b[3]):
            raise RuntimeError(
                f"{SHEET_TITER} and {SHEET_TITER_ALT} disagree at row {index}: "
                f"{a[0]!r}/{a[3]!r} vs {b[0]!r}/{b[3]!r}"
            )
    parsed: list[TiterRow] = []
    for line, cycle, is_control, titer, passed in rows:
        match = REPLICATE_RE.match(str(line))
        if match is None:
            raise RuntimeError(f"line name {line!r} has no -R<n> replicate suffix")
        parsed.append(
            TiterRow(
                construct_name=match.group("base"),
                cycle=int(cycle),
                replicate=int(match.group("replicate")),
                is_control=str(is_control) == "True",
                titer_mg_per_l=float(titer),
                passed_filter=str(passed) == "True",
            )
        )
    return parsed


def read_target_means(path: str) -> dict[str, float]:
    """``Figure 3a``'s released per-target mean DBTL0 titer, keyed by locus tag."""
    header, rows = _sheet_rows(path, SHEET_TARGET_MEANS)
    if list(header[:4]) != ["Strain", "Target", "cog_base_function", "Isoprenol mean"]:
        raise RuntimeError(f"{SHEET_TARGET_MEANS} header changed: {header!r}")
    return {str(row[1]).strip(): float(row[3]) for row in rows if row[1] is not None}


def read_control_titers(path: str) -> dict[int, list[float]]:
    """``Figure 2c``'s control titers per DBTL cycle, as the second control source."""
    header, rows = _sheet_rows(path, SHEET_CONTROLS)
    if list(header[:3]) != ["Line Name", "Cycle", "Titer"]:
        raise RuntimeError(f"{SHEET_CONTROLS} header changed: {header!r}")
    out: dict[int, list[float]] = defaultdict(list)
    for _, cycle, titer in rows:
        if titer is None:
            continue
        label = str(cycle)
        if not label.startswith("DBTL-"):
            raise RuntimeError(
                f"{SHEET_CONTROLS} cycle label {label!r} is not DBTL-<n>"
            )
        out[int(label.removeprefix("DBTL-"))].append(float(titer))
    return dict(out)


def read_si_target_means(path: str) -> dict[str, float]:
    """Supplementary Data 1's ``Mean isoprenol titer (mg/L)`` per locus number."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        stream = book.worksheets[0].iter_rows(values_only=True)
        next(stream)
        next(stream)
        header = next(stream)
        if str(
            header[0]
        ).strip() != "Locus number" or "Mean isoprenol titer" not in str(header[4]):
            raise RuntimeError(f"Supplementary Data 1 header changed: {header!r}")
        return {
            str(row[0]).strip(): float(row[4])
            for row in stream
            if row[0] is not None and row[4] is not None
        }
    finally:
        book.close()


class ProteomeRow(BaseModel):
    """One (sample, replicate, protein) cell of the released off-target proteome."""

    sample: str
    replicate: str
    protein: str
    accession: str
    entry_name: str
    description: str
    top3_signal: float


def read_proteome_rows(path: str) -> list[ProteomeRow]:
    """Every cell of the released off-target-study Top3 abundance matrix."""
    header, rows = _sheet_rows(path, SHEET_PROTEOME)
    expected = [
        "Protein.Group",
        "Protein.Names",
        "Protein",
        "Protein.Description",
        "Sample",
        "Replicate",
        "Top_3pep_counts_mean",
    ]
    if list(header[:7]) != expected:
        raise RuntimeError(f"{SHEET_PROTEOME} header changed: {header!r}")
    return [
        ProteomeRow(
            accession=str(row[0]).strip(),
            entry_name=str(row[1]).strip(),
            protein=str(row[2]).strip(),
            description=str(row[3]).strip(),
            sample=str(row[4]).strip(),
            replicate=str(row[5]).strip(),
            top3_signal=float(row[6]),
        )
        for row in rows
    ]


# --- the deposited Dryad campaign proteome ------------------------------------
class CampaignCulture(BaseModel):
    """One released culture of the campaign proteome: its identity and its titer.

    ``construct_name`` and ``replicate`` come from ``Line_name``, which the titer
    family also keys on, NOT from ``Line``: measured on the pinned bytes, ``Line``
    carries a ``PRT1093_`` prefix in DBTL6 that ``Line_name`` does not
    (:data:`CAMPAIGN_LINE_COLUMN_RULE`).
    """

    cycle: int
    construct_name: str
    replicate: int
    line_cell: str = Field(description="the 'Line' cell, verbatim")
    is_control: bool
    titer_mg_per_l: float


class CampaignProteome(BaseModel):
    """The deposited campaign proteome: its cultures, accessions and cells."""

    model_config = ConfigDict(extra="forbid")

    accessions: tuple[str, ...] = Field(
        description="the UniProt accession column headers, in released order"
    )
    cultures: tuple[CampaignCulture, ...]
    #: ``(culture index, accession) -> released percent-of-proteome``. Blank cells are
    #: ABSENT from this map by :data:`CAMPAIGN_BLANK_IS_ABSENT`; a released 0 is in it.
    cells: dict[tuple[int, str], float]
    blank_cells: int
    zero_cells: int


def read_campaign_proteome(path: str) -> CampaignProteome:
    """Read the deposited Top3 matrix: every culture, every accession, every cell.

    Refuses a header whose six leading columns are not :data:`DRYAD_KEY_COLUMNS`, a
    titer unit column holding anything but :data:`DRYAD_TITER_UNIT`, a ``Line_name``
    with no ``-R<n>`` suffix, and a repeated ``(cycle, Line_name)`` identity.
    """
    frame = pd.read_csv(path)
    header = tuple(str(column) for column in frame.columns)
    if header[: len(DRYAD_KEY_COLUMNS)] != DRYAD_KEY_COLUMNS:
        raise RuntimeError(
            f"{osp.basename(path)} leads with {header[:6]!r}, not {DRYAD_KEY_COLUMNS!r}"
        )
    units = set(frame["Isoprenol_titer_units"].astype(str))
    if units != {DRYAD_TITER_UNIT}:
        raise RuntimeError(
            f"{osp.basename(path)} states titer units {sorted(units)!r}; "
            f"{CAMPAIGN_TITER_IN_MG_PER_L.quote!r} says {DRYAD_TITER_UNIT}"
        )
    accessions = header[len(DRYAD_KEY_COLUMNS) :]
    cultures: list[CampaignCulture] = []
    seen: set[tuple[int, str, int]] = set()
    for cycle, line_name, line_cell in zip(
        frame["DBTL_Cycle"], frame["Line_name"], frame["Line"], strict=True
    ):
        match = REPLICATE_RE.match(str(line_name))
        if match is None:
            raise RuntimeError(f"Line_name {line_name!r} has no -R<n> replicate suffix")
        construct = match.group("base")
        replicate = int(match.group("replicate"))
        identity = (int(cycle), construct, replicate)
        if identity in seen:
            raise RuntimeError(f"{identity} appears twice in {osp.basename(path)}")
        seen.add(identity)
        cultures.append(
            CampaignCulture(
                cycle=int(cycle),
                construct_name=construct,
                replicate=replicate,
                line_cell=str(line_cell),
                is_control=construct.startswith(KO_ARRAY_CONTROL_LINE),
                titer_mg_per_l=0.0,
            )
        )
    titers = frame["Isoprenol_titer"].astype(float).tolist()
    cultures = [
        culture.model_copy(update={"titer_mg_per_l": float(titer)})
        for culture, titer in zip(cultures, titers, strict=True)
    ]
    block = frame[list(accessions)]
    values = block.to_numpy(dtype=float, na_value=math.nan)
    cells: dict[tuple[int, str], float] = {}
    blank = 0
    zero = 0
    for row_index in range(values.shape[0]):
        row = values[row_index]
        for column_index, accession in enumerate(accessions):
            value = float(row[column_index])
            if math.isnan(value):
                blank += 1
                continue
            if value == 0.0:
                zero += 1
            cells[(row_index, accession)] = value
    return CampaignProteome(
        accessions=accessions,
        cultures=tuple(cultures),
        cells=cells,
        blank_cells=blank,
        zero_cells=zero,
    )


class CampaignMetadataRow(BaseModel):
    """One row of the deposited per-culture metadata CSV, as the loader reads it."""

    model_config = ConfigDict(extra="forbid")

    cycle: int
    line_name: str
    line_cell: str
    genes_targeted: str | None = Field(
        description="the 'Genes_targeted_for_CRISPRi' cell, or None when it is blank"
    )
    values: dict[str, str] = Field(
        description="the environment columns, as strings, verbatim"
    )


#: The metadata columns this loader asserts the shared environment against, and the
#: value each must carry in every row. Every one is the same in all 1,497 rows
#: (measured 2026-10-09), which is what makes the file a cross-source oracle for
#: :func:`production_environment` rather than a per-culture variable.
CAMPAIGN_ENVIRONMENT_COLUMNS: dict[str, str] = {
    "Culture_volume": "1.5",
    "Culture_format": "48-well BioLector flower plate",
    "Growth_temperature_Celsius": "24",
    "Shaking_speed_rpm": "1000",
    "Media": "M9-NREL",
    "Carbon_source": "Glucose",
    "Carbon_source_concentration": "20",
    "Carbon_source_concentration_units": "g/L",
    "Inducer": "L-arabinose",
    "Inducer_concentration": "2",
    "Inducer_concentration_units": "g/L",
    "Induction_time_point": "8",
    "Induction_time_point_units": "hr",
    "Assay_type": "Proteomics, Isoprenol_titer",
    "Assay_time_point": "48",
    "Assay_time_point_units": "hr",
    "Organism": "Pseudomonas putida KT2440",
    "Strain_ID": "IY1449b",
}


def read_campaign_metadata(path: str) -> list[CampaignMetadataRow]:
    """Read the deposited per-culture metadata CSV.

    The file is UTF-8 with a BOM, so its first header cell is read by position rather
    than by name; every other column is read by name and a missing one refuses.
    """
    frame = pd.read_csv(path)
    frame = frame.rename(columns={frame.columns[0]: "DBTL_Cycle"})
    missing = sorted(
        set(CAMPAIGN_ENVIRONMENT_COLUMNS)
        | {"DBTL_Cycle", "Line_name", "Line", "Genes_targeted_for_CRISPRi"}
        - set(frame.columns)
    )
    absent = [column for column in missing if column not in frame.columns]
    if absent:
        raise RuntimeError(f"{osp.basename(path)} has no columns {absent}")
    rows: list[CampaignMetadataRow] = []
    for record in frame.to_dict("records"):
        gene_cell = record["Genes_targeted_for_CRISPRi"]
        rows.append(
            CampaignMetadataRow(
                cycle=int(record["DBTL_Cycle"]),
                line_name=str(record["Line_name"]),
                line_cell=str(record["Line"]),
                genes_targeted=(
                    None if gene_cell is None or pd.isna(gene_cell) else str(gene_cell)
                ),
                values={
                    column: str(record[column])
                    for column in CAMPAIGN_ENVIRONMENT_COLUMNS
                },
            )
        )
    return rows


class KoPanelRow(BaseModel):
    """One released culture of ``Figure 6a``: a KO background, an arm and a titer."""

    background: str
    arm: Literal["CRISPRi", "Target", "Non-target"]
    titer_mg_per_l: float


def read_ko_panel_rows(path: str) -> list[KoPanelRow]:
    """Every row of ``Figure 6a``, after asserting ``Supplementary Figure 11`` matches.

    The two sheets are one export of one panel: measured on the pinned workbook they
    are the same 105 rows in the same order, cell for cell. A disagreement means the
    workbook is not the one this loader was written against, so it refuses.
    """
    header, rows = _sheet_rows(path, SHEET_KO_PANEL)
    alt_header, alt = _sheet_rows(path, SHEET_KO_PANEL_ALT)
    expected = ["Strain", "Type", "Titer"]
    if list(header[:3]) != expected or list(alt_header[:3]) != expected:
        raise RuntimeError(
            f"{SHEET_KO_PANEL} / {SHEET_KO_PANEL_ALT} header changed: "
            f"{header!r} / {alt_header!r}"
        )
    if [tuple(row) for row in rows] != [tuple(row) for row in alt]:
        raise RuntimeError(
            f"{SHEET_KO_PANEL} and {SHEET_KO_PANEL_ALT} are not the same rows; they "
            "are one export of one panel on the pinned bytes"
        )
    return [
        KoPanelRow(
            background=str(background).strip(),
            arm=str(arm).strip(),  # type: ignore[arg-type]
            titer_mg_per_l=float(titer),
        )
        for background, arm, titer in rows
    ]


class KoArrayRow(BaseModel):
    """One released culture of ``Figure 6d``: a line name, an arm and a titer."""

    line_name: str
    arm: Literal["CRISPRi", "KO Only", "CRISPRi and KO"]
    replicate: str
    titer_mg_per_l: float


def read_ko_array_rows(path: str) -> list[KoArrayRow]:
    """Every row of ``Figure 6d``, the KO-only and CRISPRi-on-a-KO panel."""
    header, rows = _sheet_rows(path, SHEET_KO_ARRAYS)
    if list(header[:4]) != ["Line Name", "Type", "Replicate", "Isoprenol"]:
        raise RuntimeError(f"{SHEET_KO_ARRAYS} header changed: {header!r}")
    return [
        KoArrayRow(
            line_name=str(row[0]).strip(),
            arm=str(row[1]).strip(),  # type: ignore[arg-type]
            replicate=str(row[2]).strip(),
            titer_mg_per_l=float(row[3]),
        )
        for row in rows
    ]


class OffTargetTiterRow(BaseModel):
    """One released culture of ``Supplementary Figure 13d``."""

    sample: str
    replicate: str
    titer_mg_per_l: float


def read_offtarget_titer_rows(path: str) -> list[OffTargetTiterRow]:
    """Every row of ``Supplementary Figure 13d``, the off-target panel's titer arm."""
    header, rows = _sheet_rows(path, SHEET_OFFTARGET_TITER)
    if list(header[:3]) != ["Sample", "Replicate", "titer"]:
        raise RuntimeError(f"{SHEET_OFFTARGET_TITER} header changed: {header!r}")
    return [
        OffTargetTiterRow(
            sample=str(sample).strip(),
            replicate=str(replicate).strip(),
            titer_mg_per_l=float(titer),
        )
        for sample, replicate, titer in rows
    ]


class OverexpressionTiterRow(BaseModel):
    """One released culture of ``Supplementary Figure 12bd``."""

    strain: str
    replicate: str
    inducer_level: float
    titer_mg_per_l: float


def read_overexpression_titer_rows(path: str) -> list[OverexpressionTiterRow]:
    """Every row of ``Supplementary Figure 12bd``, the overexpression titer panel."""
    header, rows = _sheet_rows(path, SHEET_OVEREXPRESSION_TITER)
    if list(header[:4]) != [
        "Strain",
        "Replicate",
        "Inducer concentration",
        "Isoprenol",
    ]:
        raise RuntimeError(f"{SHEET_OVEREXPRESSION_TITER} header changed: {header!r}")
    return [
        OverexpressionTiterRow(
            strain=str(strain).strip(),
            replicate=str(replicate).strip(),
            inducer_level=float(inducer),
            titer_mg_per_l=float(titer),
        )
        for strain, replicate, inducer, titer in rows
    ]


class OverexpressionProteomeRow(BaseModel):
    """One (sample, replicate, protein) cell of ``Supplementary Figure 12ac``."""

    sample: str
    strain: str
    inducer_level: float
    replicate: str
    protein: str
    accession: str
    entry_name: str
    description: str
    top3_signal: float


def read_overexpression_proteome_rows(path: str) -> list[OverexpressionProteomeRow]:
    """Every cell of ``Supplementary Figure 12ac``'s Top3 abundance matrix.

    Same column set and same Top3 scale as :data:`SHEET_PROTEOME`, plus the ``Strain``
    and ``Inducer concentration`` columns the overexpression panel varies. The loader
    consumes ``Top_3pep_counts_mean``, so both proteome panels store one
    ``measurement_type``.
    """
    header, rows = _sheet_rows(path, SHEET_OVEREXPRESSION_PROTEOME)
    expected = [
        "Protein.Group",
        "Protein.Names",
        "Protein",
        "Protein.Description",
        "Sample",
        "Strain",
        "Inducer concentration",
        "Replicate",
        "Top_3pep_counts_mean",
    ]
    if list(header[:9]) != expected:
        raise RuntimeError(
            f"{SHEET_OVEREXPRESSION_PROTEOME} header changed: {header!r}"
        )
    return [
        OverexpressionProteomeRow(
            accession=str(row[0]).strip(),
            entry_name=str(row[1]).strip(),
            protein=str(row[2]).strip(),
            description=str(row[3]).strip(),
            sample=str(row[4]).strip(),
            strain=str(row[5]).strip(),
            inducer_level=float(row[6]),
            replicate=str(row[7]).strip(),
            top3_signal=float(row[8]),
        )
        for row in rows
    ]


class FoldChangeRow(BaseModel):
    """One released ``(protein key, contrast)`` log2 fold change and its -log10 p-value."""

    sheet: str
    contrast: str
    protein: str
    log2_fold_change: float
    neg_log10_p_value: float


def p_value_from_neg_log10(value: float) -> float:
    """``p = 10**-x`` for an UNSIGNED -log10 p-value, with the round trip asserted.

    The released column is -log10(p), so the probability is recovered by exact
    arithmetic rather than estimated. The inverse is checked on every value because a
    conversion that is not reversible is not the same number the source released.
    """
    if value < 0.0 or not math.isfinite(value):
        raise RuntimeError(
            f"a -log10 p-value must be a non-negative finite number, got {value!r}"
        )
    probability = float(10.0**-value)
    if not 0.0 < probability <= 1.0:
        raise RuntimeError(f"10**-{value} is {probability}, not a probability")
    back = -math.log10(probability)
    if abs(back - value) > P_VALUE_ROUND_TRIP_TOL:
        raise RuntimeError(
            f"the p-value conversion is not reversible: -log10(10**-{value}) = {back}"
        )
    if probability >= FOLD_CHANGE_P_VALUE_CEILING:
        raise RuntimeError(
            f"p = {probability} is not below the {FOLD_CHANGE_P_VALUE_CEILING} the "
            f"caption states the sheet is filtered at ('{_Q_FIG5B}')"
        )
    return probability


def _fold_change_column_pairs(
    header: Sequence[Any], sheet: str
) -> list[tuple[str, int, int]]:
    """``(contrast, fold-change column, p-value column)`` for each released contrast.

    The p-value column must be the fold-change column's immediate right neighbor and
    must carry the same contrast name, so a reordered or renamed export is refused
    rather than silently paired with the wrong statistic.
    """
    pairs: list[tuple[str, int, int]] = []
    for index, cell in enumerate(header):
        name = "" if cell is None else str(cell)
        if not name.endswith(FOLD_CHANGE_SUFFIX):
            continue
        contrast = name[: -len(FOLD_CHANGE_SUFFIX)]
        expected = f"{contrast}{NEG_LOG10_P_VALUE_SUFFIX}"
        neighbor = index + 1
        if neighbor >= len(header) or str(header[neighbor]) != expected:
            raise RuntimeError(
                f"{sheet}: column {name!r} is not followed by {expected!r}; the sheet's "
                "fold-change / p-value pairing changed"
            )
        pairs.append((contrast, index, neighbor))
    if not pairs:
        raise RuntimeError(f"{sheet}: no {FOLD_CHANGE_SUFFIX} column")
    return pairs


def read_fold_change_rows(path: str, sheet: str) -> list[FoldChangeRow]:
    """Every released fold-change cell of one sheet, long-form.

    A fold change and its p-value are released together or not at all, and the two
    sheets are ragged: a protein a contrast did not test is an empty pair of cells and
    becomes no row, never a 0 and never a 1. ``SHEET_KO_FOLD_CHANGE`` additionally
    carries 181 rows that hold a ``primary_name`` with no ``Locus Name`` and no value;
    they are skipped, and one that ever carries a value stops the build.
    """
    header, rows = _sheet_rows(path, sheet)
    pairs = _fold_change_column_pairs(header, sheet)
    out: list[FoldChangeRow] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = row[0]
        if key is None:
            if any(row[fc] is not None or row[pv] is not None for _, fc, pv in pairs):
                raise RuntimeError(
                    f"{sheet}: a row with no 'Locus Name' carries a value; the key "
                    "column cannot be dropped"
                )
            continue
        protein = str(key)
        for contrast, fc, pv in pairs:
            value, significance = row[fc], row[pv]
            if value is None:
                if significance is not None:
                    raise RuntimeError(
                        f"{sheet}/{contrast}/{protein}: a p-value with no fold change"
                    )
                continue
            if significance is None:
                raise RuntimeError(
                    f"{sheet}/{contrast}/{protein}: a fold change with no p-value"
                )
            identity = (contrast, protein)
            if identity in seen:
                raise RuntimeError(
                    f"{sheet}: {identity} appears twice; one contrast cannot give one "
                    "protein two fold changes"
                )
            seen.add(identity)
            out.append(
                FoldChangeRow(
                    sheet=sheet,
                    contrast=contrast,
                    protein=protein,
                    log2_fold_change=float(value),
                    neg_log10_p_value=float(significance),
                )
            )
    if not out:
        raise RuntimeError(f"{sheet}: no released fold change")
    return out


def fold_change_contrasts(path: str, sheet: str) -> tuple[str, ...]:
    """The sheet's contrast column names, in released order."""
    header, _ = _sheet_rows(path, sheet)
    return tuple(
        contrast for contrast, _, _ in _fold_change_column_pairs(header, sheet)
    )


def read_best_array_heatmap(
    path: str,
) -> tuple[tuple[str, ...], tuple[str, ...], list[float]]:
    """``SHEET_BEST_ARRAY_HEATMAP``'s keys, strain columns and values.

    Read ONLY to measure the refusal in :data:`BEST_ARRAY_HEATMAP_REFUSAL`; nothing
    this returns reaches a record.
    """
    header, rows = _sheet_rows(path, SHEET_BEST_ARRAY_HEATMAP)
    strains = tuple(str(cell) for cell in header[1:] if cell is not None)
    keys = tuple(str(row[0]) for row in rows)
    values = [float(cell) for row in rows for cell in row[1:] if cell is not None]
    return keys, strains, values


def assert_fold_change_pointer(path: str) -> str:
    """Prove the Supplementary Fig. 9 caption describes ``SHEET_FOLD_CHANGE``.

    The Source Data devotes a sheet to Supplementary Figure 9 and puts one cell in it.
    That cell is what makes the caption's "compared to the non-target control strain" a
    statement about this sheet's denominator rather than about a different figure, so
    it is asserted on the pinned bytes before a ``reference_basis`` is written.
    """
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if SHEET_FOLD_CHANGE_POINTER not in book.sheetnames:
            raise RuntimeError(
                f"{osp.basename(path)} has no sheet {SHEET_FOLD_CHANGE_POINTER!r}"
            )
        cells = [
            str(cell)
            for row in book[SHEET_FOLD_CHANGE_POINTER].iter_rows(values_only=True)
            for cell in row
            if cell is not None
        ]
    finally:
        book.close()
    if cells != [FOLD_CHANGE_POINTER_CELL]:
        raise RuntimeError(
            f"{SHEET_FOLD_CHANGE_POINTER} holds {cells!r}, not "
            f"[{FOLD_CHANGE_POINTER_CELL!r}]; the Supplementary Fig. 9 caption can no "
            f"longer be read as a statement about {SHEET_FOLD_CHANGE}"
        )
    return (
        f"{SHEET_FOLD_CHANGE_POINTER} holds exactly one cell, "
        f"{FOLD_CHANGE_POINTER_CELL!r}, so its caption "
        f"('{_Q_SI_FIG9_TITLE}') states {SHEET_FOLD_CHANGE}'s denominator"
    )


def assert_fold_change_contrasts_are_released_strains(path: str) -> list[str]:
    """Join every fold-change contrast column to the strains the release names.

    ``SHEET_FOLD_CHANGE``'s columns must all be single-guide CRISPRi constructs of the
    campaign, which is what licenses the genotype each record is written with; a column
    that is not stops the build. The ``SHEET_KO_FOLD_CHANGE`` columns are only MEASURED
    against the same universes, because the eight unstored ones are ledgered by that
    measurement rather than built.
    """
    titer_rows = read_titer_rows(path)
    constructs = {row.construct_name for row in titer_rows}
    ko_backgrounds = {row.background for row in read_ko_panel_rows(path)}
    proofs: list[str] = []
    singles = fold_change_contrasts(path, SHEET_FOLD_CHANGE)
    missing = [tag for tag in singles if tag not in constructs]
    if missing:
        raise RuntimeError(
            f"{SHEET_FOLD_CHANGE} columns {missing} are not released {SHEET_TITER} "
            "constructs, so the single-guide CRISPRi genotype is not sourced for them"
        )
    off_pattern = [tag for tag in singles if not LOCUS_TAG_RE.fullmatch(tag)]
    if off_pattern:
        raise RuntimeError(
            f"{SHEET_FOLD_CHANGE} columns {off_pattern} are not single PP_ locus tags"
        )
    proofs.append(
        f"all {len(singles)} {SHEET_FOLD_CHANGE} contrast columns are single PP_ locus "
        f"tags released as single-guide {SHEET_TITER} constructs"
    )
    for contrast in fold_change_contrasts(path, SHEET_KO_FOLD_CHANGE):
        normalized = contrast.replace("_15", "-15") if "_15" in contrast else contrast
        proofs.append(
            f"{SHEET_KO_FOLD_CHANGE} column {contrast!r}: "
            f"{SHEET_KO_PANEL} KO background {normalized in ko_backgrounds}, "
            f"{SHEET_TITER} line name {contrast in constructs}"
        )
    return proofs


def assert_best_array_heatmap_refusal(path: str) -> list[str]:
    """Measure, and record, why ``SHEET_BEST_ARRAY_HEATMAP`` is refused.

    A refusal is a measurement here, not an opinion: the sheet's shape, density, value
    range and key form are read off the pinned bytes, and the absence of any p-value
    column is proved by the header rather than assumed.
    """
    keys, strains, values = read_best_array_heatmap(path)
    header, _ = _sheet_rows(path, SHEET_BEST_ARRAY_HEATMAP)
    significance = [
        str(cell)
        for cell in header
        if cell is not None and str(cell).endswith(NEG_LOG10_P_VALUE_SUFFIX)
    ]
    if significance:
        raise RuntimeError(
            f"{SHEET_BEST_ARRAY_HEATMAP} now releases p-value columns {significance}; "
            "the recorded refusal is stale and must be re-measured"
        )
    tags = [key for key in keys if LOCUS_TAG_RE.fullmatch(key)]
    if len(values) != len(keys) * len(strains):
        raise RuntimeError(
            f"{SHEET_BEST_ARRAY_HEATMAP} is no longer dense: {len(values)} values for "
            f"{len(keys)} keys x {len(strains)} strain columns"
        )
    return [
        f"{SHEET_BEST_ARRAY_HEATMAP} refused: {len(keys)} keys x {len(strains)} strain "
        f"columns = {len(values)} values with no empty cell, no "
        f"{NEG_LOG10_P_VALUE_SUFFIX} column, values in "
        f"[{min(values)}, {max(values)}] with median {statistics.median(values)}, and "
        f"only {len(tags)} of {len(keys)} keys a PP_ locus tag",
        f"{SHEET_BEST_ARRAY_HEATMAP} refusal reason: {BEST_ARRAY_HEATMAP_REFUSAL.note}",
    ]


def expand_locus_designation(token: str) -> tuple[str, ...]:
    """The locus tags one KO designation names, e.g. ``PP_0812-15`` -> four tags.

    A bare ``PP_xxxx`` is itself; ``PP_xxxx-yy`` is the INCLUSIVE locus-number range
    whose end shares the start's leading digits, which is what the Results' "PP_0815
    (subunits of a terminal oxidase complex PP_0812-15)" names and what Supplementary
    Table 5 corroborates by building ``IY1449b ΔPP_0812-15`` as ``PP_0813-15`` deleted
    from ``IY1449b ΔPP_0812``. A range that does not ascend is a parse error, not a
    datum to accept.
    """
    if LOCUS_TAG_RE.fullmatch(token):
        return (token,)
    match = LOCUS_RANGE_RE.match(token)
    if match is None:
        raise RuntimeError(
            f"KO designation {token!r} is neither a PP_ locus tag nor a PP_xxxx-yy "
            "locus-number range"
        )
    start = int(match.group("start"))
    end = int(f"{match.group('start')[:2]}{match.group('end')}")
    if end <= start:
        raise RuntimeError(f"KO designation {token!r} does not ascend: {start}..{end}")
    return tuple(f"PP_{number:04d}" for number in range(start, end + 1))


def parse_ko_array_line(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(deleted designations, knocked-down designations)`` of a ``Figure 6d`` line.

    ``PP_0812-15_KO`` deletes one designation and knocks nothing down;
    ``PP_0368_PP_0812-15_KO with PP_0528_PP_0815`` deletes two and knocks two down.
    Anything else raises: a line name this loader cannot read is a changed release.
    """
    match = KO_ARRAY_LINE_RE.match(name)
    if match is None:
        raise RuntimeError(
            f"{SHEET_KO_ARRAYS} line name {name!r} is neither '<designations>_KO' nor "
            "'<designations>_KO with <designations>'"
        )
    deleted = _split_designations(match.group("deleted"))
    knocked = match.group("knocked_down")
    return deleted, _split_designations(knocked) if knocked else ()


def _split_designations(label: str) -> tuple[str, ...]:
    """``PP_0368_PP_0812-15`` -> ``('PP_0368', 'PP_0812-15')``."""
    parts = re.findall(r"PP_\d{4}(?:-\d{2})?", label)
    if "_".join(parts) != label:
        raise RuntimeError(
            f"{label!r} is not PP_ designations joined by '_': parsed {parts!r}"
        )
    return tuple(parts)


def parse_construct(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(locus tags, non-tag tokens)`` of a CRISPRi construct's released name.

    ``PP_0528_PP_0751_PP_0815`` is three targets; ``PP_1607_NT1`` is one target plus a
    non-targeting filler guide occupying an array position, which perturbs no gene.
    """
    tags = tuple(LOCUS_TAG_RE.findall(name))
    residue = LOCUS_TAG_RE.sub("", name).strip("_")
    extras = tuple(token for token in residue.split("_") if token)
    return tags, extras


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, what it removed, and the items it removed."""

    rule: str
    scope: Literal["culture", "strain", "protein_key", "protein_accession"]
    description: str
    n_records: int
    items: list[str] = []


class BuildAccounting(BaseModel):
    """Everything a reader needs to audit one build's retention arithmetic."""

    dataset: str
    source_rows: int
    control_rows: int
    candidate_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule] = []
    reconciliation: LocusTagReconciliation | None = None
    notes: list[str] = []

    def check(self) -> None:
        """The rules must account for every candidate record that was not written."""
        accounted = sum(rule.n_records for rule in self.rules)
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_records} "
                f"dropped != {self.candidate_records} candidates"
            )
        if accounted != self.dropped_records:
            raise RuntimeError(
                f"{self.dataset}: rules total {accounted}, {self.dropped_records} "
                "records are missing from the build"
            )


def _write_accounting(accounting: BuildAccounting, preprocess_dir: str) -> None:
    """Validate and write ``preprocess/build_accounting.json``."""
    accounting.check()
    os.makedirs(preprocess_dir, exist_ok=True)
    path = osp.join(preprocess_dir, "build_accounting.json")
    with open(path, "w") as handle:
        handle.write(accounting.model_dump_json(indent=2))


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


def _standard_names(genome: PPutidaKT2440Genome, tags: Iterable[str]) -> dict[str, str]:
    """Locus tag -> the genome's own gene symbol for it, falling back to the tag.

    The symbol is read from the annotation, so one gene carries one spelling across
    datasets; a locus the annotation gives no symbol keeps its tag as the common name.
    ``feature_index["symbol"]`` is ``(exact map, case-folded map)`` of name -> locus
    tags, so the exact map is inverted here.
    """
    exact, _ = genome.feature_index["symbol"]
    by_tag: dict[str, str] = {}
    for symbol, loci in exact.items():
        for locus in loci:
            by_tag.setdefault(str(locus), str(symbol))
    return {tag: by_tag.get(tag, tag) for tag in tags}


# --------------------------------------------------------------------------- #
# The four unstored titer panels of the Source Data file
#
# ``Figure 4b`` is the CRISPRi campaign and is what the 465 stored records are. Four
# further sheets release per-replicate titers of strains the campaign does NOT contain:
# chromosomal KO backgrounds, the off-target sgRNA panel, and the two overexpression
# strains. Each sheet also re-exports campaign values, so the first thing this section
# does is prove the partition, in both directions, before a record is built.
# --------------------------------------------------------------------------- #
class PanelTiterGroup(BaseModel):
    """One group of released cultures of a non-``Figure 4b`` panel: one titer record."""

    sheet: str
    group: str
    deletions: tuple[str, ...] = ()
    knockdowns: tuple[str, ...] = ()
    native_copies: tuple[str, ...] = ()
    titers_mg_per_l: tuple[float, ...]
    reference_key: str
    overexpression_environment: bool = False
    note: str | None = None


class PanelTiterReference(BaseModel):
    """One reference group of a panel. Two sheets exporting ONE group share one key."""

    key: str
    sheets: tuple[str, ...]
    group: str
    titers_mg_per_l: tuple[float, ...]
    note: str


class PanelTiterPlan(BaseModel):
    """Every titer record and reference the four panels contribute, with its proofs.

    ``stored_cycle_reference`` maps a reference key to a DBTL cycle of the ALREADY
    STORED per-cycle controls, for a panel whose own control rows are a measured
    re-export of those cultures. ``proofs`` are the measured cross-sheet statements the
    build asserted, written to ``preprocess/`` so the arithmetic stays auditable.
    """

    records: list[PanelTiterGroup]
    references: list[PanelTiterReference]
    stored_cycle_reference: dict[str, int] = {}
    proofs: list[str] = []
    drops: list[DropRule] = []


def _ordered(values: Iterable[float]) -> tuple[float, ...]:
    """A group's titers sorted, so a cross-sheet identity test is replicate-order free."""
    return tuple(sorted(values))


def _assert_panel_partition(
    reexported: Mapping[str, Sequence[float]],
    novel: Mapping[str, Sequence[float]],
    stored: Collection[float],
) -> list[str]:
    """Prove the partition: the CRISPRi arms are re-exports, the KO arms are new.

    Both directions are asserted, because each is a different failure. A ``CRISPRi``
    culture that stopped matching ``Figure 4b`` means the sheets have diverged and the
    re-export reading is wrong; a KO culture that started matching it means the
    revision is about to store one titer twice. Measured on the pinned workbook: 33 of
    33 ``Figure 6a`` and 99 of 99 ``Figure 6d`` CRISPRi cultures are stored values, and
    0 of the 190 cultures of the four new arms are.
    """
    proofs: list[str] = []
    stored_set = set(stored)
    for label, values in sorted(reexported.items()):
        missing = [value for value in values if value not in stored_set]
        if missing:
            raise RuntimeError(
                f"{label}: {len(missing)} of {len(values)} {ARM_CRISPRI} cultures are "
                f"NOT {SHEET_TITER} values, so the sheets have diverged: {missing[:5]}"
            )
        proofs.append(
            f"{label} {ARM_CRISPRI}: {len(values)} of {len(values)} cultures are "
            f"already-stored {SHEET_TITER} titers, so none is taken"
        )
    for label, values in sorted(novel.items()):
        collisions = [value for value in values if value in stored_set]
        if collisions:
            raise RuntimeError(
                f"{label}: {len(collisions)} of {len(values)} new cultures ALREADY "
                f"appear in {SHEET_TITER}, so taking them would store a titer twice: "
                f"{collisions[:5]}"
            )
        proofs.append(
            f"{label}: 0 of {len(values)} cultures appear in {SHEET_TITER}, so all are "
            "genuinely new"
        )
    return proofs


def _ko_panel_groups(
    rows: Sequence[KoPanelRow],
) -> tuple[list[PanelTiterGroup], list[PanelTiterReference], list[str]]:
    """``Figure 6a``: one record per KO background, its Non-target arm the reference.

    The design is the Results' own: "pairs of knockout (KO) strains harboring either a
    non-target sgRNA or the 'target' sgRNA previously used to downregulate the KO
    gene". So the Target arm is the KO plus a knockdown of the gene it deleted, and the
    Non-target arm is the same KO carrying a sgRNA that perturbs no gene, which is a
    ``phenotype_reference`` rather than a record. One background, ``PP_0812-15``,
    deletes four genes and its sgRNA is named only in the Results text
    (:data:`KO_MULTI_GENE_GUIDE`); a multi-gene background absent from that sourced map
    stops the build rather than getting a guessed guide.
    """
    arms: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        arms[row.background][row.arm].append(row.titer_mg_per_l)
    guides = dict(KO_MULTI_GENE_GUIDE.value)
    records: list[PanelTiterGroup] = []
    references: list[PanelTiterReference] = []
    proofs: list[str] = []
    for background, by_arm in sorted(arms.items()):
        deletions = expand_locus_designation(background)
        if len(deletions) == 1:
            knockdown = deletions[0]
        elif background in guides:
            knockdown = str(guides[background])
            proofs.append(
                f"{SHEET_KO_PANEL} {background}: deletes {len(deletions)} genes "
                f"{deletions} and its Target arm carries the {knockdown} sgRNA, which "
                "only the Results text names"
            )
        else:
            raise RuntimeError(
                f"{SHEET_KO_PANEL} background {background!r} deletes "
                f"{len(deletions)} genes and no mirrored statement names the sgRNA its "
                f"{ARM_KO_TARGET} arm carries; KO_MULTI_GENE_GUIDE must gain a sourced "
                "entry before it can be stored"
            )
        target = by_arm[ARM_KO_TARGET]
        non_target = by_arm[ARM_KO_NONTARGET]
        if not target or not non_target:
            raise RuntimeError(
                f"{SHEET_KO_PANEL} background {background!r} has {len(target)} "
                f"{ARM_KO_TARGET} and {len(non_target)} {ARM_KO_NONTARGET} cultures; "
                "the panel is pairs"
            )
        key = f"non_target:{background}"
        records.append(
            PanelTiterGroup(
                sheet=SHEET_KO_PANEL,
                group=f"{background}/{ARM_KO_TARGET}",
                deletions=deletions,
                knockdowns=(knockdown,),
                titers_mg_per_l=tuple(target),
                reference_key=key,
            )
        )
        references.append(
            PanelTiterReference(
                key=key,
                sheets=(SHEET_KO_PANEL,),
                group=f"{background}/{ARM_KO_NONTARGET}",
                titers_mg_per_l=tuple(non_target),
                note=(
                    f"the Δ{background} background carrying a NON-targeting sgRNA, "
                    "which perturbs no gene, so it is this record's reference rather "
                    "than a record of its own"
                ),
            )
        )
    return records, references, proofs


def _ko_array_groups(
    rows: Sequence[KoArrayRow], control_titers: Mapping[int, Sequence[float]]
) -> tuple[list[PanelTiterGroup], list[str]]:
    """``Figure 6d``: the KO-only and CRISPRi-on-a-KO records, on a STORED reference.

    The sheet's own ``Control`` triplicate is measured to be three of the twelve DBTL6
    control cultures ``Figure 4b`` already releases, so these records point at the
    stored DBTL6 ``phenotype_reference`` (n = 12) rather than at a three-culture copy
    of three of its members.
    """
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row.line_name, row.arm)].append(row.titer_mg_per_l)
    control = groups.pop((KO_ARRAY_CONTROL_LINE, ARM_CRISPRI), None)
    if control is None:
        raise RuntimeError(
            f"{SHEET_KO_ARRAYS} has no {KO_ARRAY_CONTROL_LINE!r} / {ARM_CRISPRI!r} "
            "row, so its reference cannot be matched to a stored cycle"
        )
    stored_cycle = set(control_titers[KO_ARRAY_CONTROL_CYCLE])
    foreign = [value for value in control if value not in stored_cycle]
    if foreign:
        raise RuntimeError(
            f"{SHEET_KO_ARRAYS}'s control cultures {foreign} are not DBTL"
            f"{KO_ARRAY_CONTROL_CYCLE} control titers of {SHEET_TITER}, so the panel's "
            "reference cannot be the stored per-cycle one"
        )
    proofs = [
        f"{SHEET_KO_ARRAYS} {KO_ARRAY_CONTROL_LINE}: all {len(control)} cultures are "
        f"DBTL{KO_ARRAY_CONTROL_CYCLE} control cultures of {SHEET_TITER}, so the "
        f"records reuse the stored DBTL{KO_ARRAY_CONTROL_CYCLE} reference "
        f"(n = {len(control_titers[KO_ARRAY_CONTROL_CYCLE])}) instead of storing three "
        "of its twelve members a second time"
    ]
    records: list[PanelTiterGroup] = []
    for (line, arm), values in sorted(groups.items()):
        if arm == ARM_CRISPRI:
            continue
        deleted, knocked = parse_ko_array_line(line)
        deletions = tuple(
            tag for token in deleted for tag in expand_locus_designation(token)
        )
        knockdowns = tuple(
            tag for token in knocked for tag in expand_locus_designation(token)
        )
        if (arm == ARM_KO_ONLY) != (not knockdowns):
            raise RuntimeError(
                f"{SHEET_KO_ARRAYS} line {line!r} is arm {arm!r} but parses to "
                f"{len(knockdowns)} knockdowns"
            )
        records.append(
            PanelTiterGroup(
                sheet=SHEET_KO_ARRAYS,
                group=f"{line}/{arm}",
                deletions=deletions,
                knockdowns=knockdowns,
                titers_mg_per_l=tuple(values),
                reference_key=f"stored_cycle:{KO_ARRAY_CONTROL_CYCLE}",
            )
        )
    return records, proofs


def _offtarget_groups(
    rows: Sequence[OffTargetTiterRow],
    ko_reference: PanelTiterReference,
    ko_target_arm: Sequence[float],
) -> tuple[list[PanelTiterGroup], PanelTiterReference, list[str]]:
    """``Supplementary Figure 13d``: the titer arm of the stored PP_0815 proteome panel.

    This is where the release's two internal duplications are settled, and both are
    settled by measurement rather than by preference.

    The ``Non-Target`` arm is bit-identical to ``Figure 6a``'s ``PP_0815`` /
    ``Non-target`` arm, so it is ONE group exported twice: it gets ONE reference
    object, shared with that sheet's record, and is never stored as two groups.

    The ``Target`` arm is NOT. The two sheets export triplicates that share exactly one
    of six values, and each of the other four occurs in its own sheet and nowhere else
    in the 31-sheet workbook. Both captions state ``n = 3``, so neither triplicate is
    stored as the other, neither is dropped, and the five distinct cultures are not
    pooled into an ``n = 5`` design no caption states: each sheet's released triplicate
    is its own record and carries the disagreement on it.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row.sample].append(row.titer_mg_per_l)
    reference_values = groups.pop(OFFTARGET_REFERENCE_SAMPLE, None)
    if reference_values is None:
        raise RuntimeError(
            f"{SHEET_OFFTARGET_TITER} has no {OFFTARGET_REFERENCE_SAMPLE!r} arm"
        )
    if _ordered(reference_values) != _ordered(ko_reference.titers_mg_per_l):
        raise RuntimeError(
            f"{SHEET_OFFTARGET_TITER}'s {OFFTARGET_REFERENCE_SAMPLE} arm "
            f"{_ordered(reference_values)} is not bit-identical to {SHEET_KO_PANEL}'s "
            f"Δ{PROTEOME_BACKGROUND_DELETION} {ARM_KO_NONTARGET} arm "
            f"{_ordered(ko_reference.titers_mg_per_l)}; the deduplication this loader "
            "documents no longer holds and the two groups must be re-decided"
        )
    reference = ko_reference.model_copy(
        update={
            "sheets": (SHEET_KO_PANEL, SHEET_OFFTARGET_TITER),
            "note": (
                f"{ko_reference.note}; {SHEET_OFFTARGET_TITER} exports the same "
                "triplicate bit for bit, so the two sheets share this one reference "
                "object and the group is stored once"
            ),
        }
    )
    proofs = [
        f"{SHEET_OFFTARGET_TITER} {OFFTARGET_REFERENCE_SAMPLE} == {SHEET_KO_PANEL} "
        f"{PROTEOME_BACKGROUND_DELETION}/{ARM_KO_NONTARGET} bit for bit "
        f"{_ordered(reference_values)}, stored once as one reference"
    ]
    target_values = groups.get(OFFTARGET_TARGET_SAMPLE)
    if target_values is None:
        raise RuntimeError(
            f"{SHEET_OFFTARGET_TITER} has no {OFFTARGET_TARGET_SAMPLE!r} arm"
        )
    overlap = sorted(set(target_values) & set(ko_target_arm))
    if len(overlap) != OFFTARGET_TARGET_SHARED_CULTURES:
        raise RuntimeError(
            f"{SHEET_OFFTARGET_TITER}'s {OFFTARGET_TARGET_SAMPLE} arm "
            f"{_ordered(target_values)} and {SHEET_KO_PANEL}'s "
            f"{PROTEOME_BACKGROUND_DELETION}/{ARM_KO_TARGET} arm "
            f"{_ordered(ko_target_arm)} share {len(overlap)} values, not the "
            f"{OFFTARGET_TARGET_SHARED_CULTURES} this loader documents; the "
            "disagreement has changed and must be re-decided before either is stored"
        )
    disagreement = (
        f"{SHEET_KO_PANEL} exports {_ordered(ko_target_arm)} for this strain and "
        f"{SHEET_OFFTARGET_TITER} exports {_ordered(target_values)}: two triplicates "
        f"sharing exactly the one value {overlap[0]}, with each of the other four "
        "occurring in its own sheet and nowhere else in the workbook. Both captions "
        "state n = 3, so neither triplicate is stored as the other, neither is "
        "dropped, and the five distinct cultures are not pooled into an n = 5 design "
        "no caption states"
    )
    proofs.append(
        f"{SHEET_OFFTARGET_TITER} {OFFTARGET_TARGET_SAMPLE} vs {SHEET_KO_PANEL} "
        f"{PROTEOME_BACKGROUND_DELETION}/{ARM_KO_TARGET}: {len(overlap)} shared value "
        f"({overlap[0]}), 2 disagreeing each way; both triplicates kept as their own "
        "record"
    )
    records: list[PanelTiterGroup] = []
    tags: set[str] = set()
    for sample, values in sorted(groups.items()):
        if sample == OFFTARGET_TARGET_SAMPLE:
            tag = PROTEOME_BACKGROUND_DELETION
            note: str | None = disagreement
        else:
            match = OFFTARGET_SAMPLE_RE.match(sample)
            if match is None:
                raise RuntimeError(
                    f"{SHEET_OFFTARGET_TITER} sample {sample!r} is neither "
                    f"{OFFTARGET_REFERENCE_SAMPLE!r}, {OFFTARGET_TARGET_SAMPLE!r} nor "
                    "a PP_xxxx[_n] off-target sgRNA strain"
                )
            tag = match.group("tag")
            note = _Q_SI_THREE_CULTURES if match.group("culture") is not None else None
            tags.add(tag)
        records.append(
            PanelTiterGroup(
                sheet=SHEET_OFFTARGET_TITER,
                group=sample,
                deletions=(PROTEOME_BACKGROUND_DELETION,),
                knockdowns=(tag,),
                titers_mg_per_l=tuple(values),
                reference_key=reference.key,
                note=note,
            )
        )
    if len(tags) != OFFTARGET_CANDIDATE_TARGETS:
        raise RuntimeError(
            f"{SHEET_OFFTARGET_TITER} screens {len(tags)} distinct off-target genes; "
            f"Supplementary Table 1 lists {OFFTARGET_CANDIDATE_TARGETS}"
        )
    proofs.append(
        f"{SHEET_OFFTARGET_TITER}: {len(records)} records over {len(tags)} distinct "
        f"off-target genes, which is Supplementary Table 1's "
        f"{OFFTARGET_CANDIDATE_TARGETS} targets plus the repeated cultures"
    )
    return records, reference, proofs


def _overexpression_titer_groups(
    rows: Sequence[OverexpressionTiterRow],
) -> tuple[list[PanelTiterGroup], PanelTiterReference, list[str], list[DropRule]]:
    """``Supplementary Figure 12bd``: the UNINDUCED arm only.

    The six induced levels are released as bare numbers whose UNIT appears nowhere in
    the mirror, so they cannot be typed as a dose and are dropped by a stated rule
    (:data:`INDUCER_UNIT_GAP`). At level 0 no inducer was added, so those cultures need
    no unit, and the uninduced arm is the one the paper's own claim rests on:
    "Uninduced expression, however, showed a ~10% increase in titer compared to the RFP
    control".
    """
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row.strain, row.inducer_level)].append(row.titer_mg_per_l)
    levels = {level for _, level in groups}
    expected = {OVEREXPRESSION_UNINDUCED_LEVEL, *OVEREXPRESSION_INDUCED_LEVELS}
    if levels != expected:
        raise RuntimeError(
            f"{SHEET_OVEREXPRESSION_TITER} releases levels {sorted(levels)}; this "
            f"loader was written against {sorted(expected)}"
        )
    labels = {strain for strain, _ in groups} - {OVEREXPRESSION_CONTROL_STRAIN}
    if labels != set(OVEREXPRESSION_SHEET_LABELS.values()):
        raise RuntimeError(
            f"{SHEET_OVEREXPRESSION_TITER} strain labels {sorted(labels)} are not the "
            f"pinned {sorted(OVEREXPRESSION_SHEET_LABELS.values())}"
        )
    by_label = {label: key for key, label in OVEREXPRESSION_SHEET_LABELS.items()}
    operons = dict(OVEREXPRESSION_OPERON_SOURCE.value)
    control = groups[(OVEREXPRESSION_CONTROL_STRAIN, OVEREXPRESSION_UNINDUCED_LEVEL)]
    reference = PanelTiterReference(
        key="overexpression_control",
        sheets=(SHEET_OVEREXPRESSION_TITER,),
        group=f"{OVEREXPRESSION_CONTROL_STRAIN}/{OVEREXPRESSION_UNINDUCED_LEVEL:g}",
        titers_mg_per_l=tuple(control),
        note=(
            f"the RFP vector control of '{_Q_OVEREXPRESSION_CULTURE}', carrying "
            f"{_Q_RFP_CONTROL_PLASMID} (Supplementary Data 3, pTE519 / JBx_266188). "
            f"The sheet releases it as one {OVEREXPRESSION_CONTROL_STRAIN!r} group of "
            f"{len(control)} uninduced cultures in two blocks and names no field that "
            "assigns a block to a plasmid, so all of them are this one reference"
        ),
    )
    records: list[PanelTiterGroup] = []
    dropped: list[str] = []
    for (label, level), values in sorted(groups.items()):
        if label == OVEREXPRESSION_CONTROL_STRAIN:
            continue
        if level != OVEREXPRESSION_UNINDUCED_LEVEL:
            dropped.append(f"{label}/{level:g}")
            continue
        records.append(
            PanelTiterGroup(
                sheet=SHEET_OVEREXPRESSION_TITER,
                group=f"{label}/{level:g}",
                native_copies=tuple(operons[by_label[label]]),
                titers_mg_per_l=tuple(values),
                reference_key=reference.key,
                overexpression_environment=True,
                note=_Q_SI_FIG12_UNINDUCED,
            )
        )
    drops = [
        DropRule(
            rule="inducer_concentration_has_no_released_unit",
            scope="strain",
            description=str(INDUCER_UNIT_GAP.note),
            n_records=len(dropped),
            items=sorted(dropped),
        )
    ]
    proofs = [
        f"{SHEET_OVEREXPRESSION_TITER}: {len(records)} uninduced records kept, "
        f"{len(dropped)} induced strain-level groups dropped because the inducer's "
        "unit is not released anywhere in the mirror"
    ]
    return records, reference, proofs, drops


def assert_phn_crosswalk(
    rows: Sequence[OverexpressionProteomeRow], stored_by_key: Mapping[str, str]
) -> list[str]:
    """Pin the one accession-to-locus crosswalk two pinned sources disagree on.

    ``Supplementary Figure 12ac`` is the only sheet of the release that puts a locus
    tag in ``Protein.Description``, and for its two Phn rows it puts the WRONG one. The
    pinned assembly annotates ``PP_2208`` as ``phnX`` with CDS product
    "phosphonoacetaldehyde hydrolase" and ``PP_2209`` as ``phnW`` with
    "2-aminoethylphosphonate--pyruvate transaminase"; the Source Data's own
    ``Protein.Names`` make ``Phnw`` the transaminase (``PHNW_PSEPK``, Q88KT0) and
    ``Phnx`` the hydrolase (``PHNX_PSEPK``, Q88KT1). Matching on the FUNCTION both
    files state therefore puts Q88KT0 at ``PP_2209`` and Q88KT1 at ``PP_2208``, which is
    what the symbol layer resolves and what the already-stored
    ``Supplementary Figure 13abc`` records are keyed by. This sheet's two description
    cells say the opposite, and they are the lone outlier.

    Both halves are asserted, so a corrected annotation or a corrected sheet stops the
    build instead of silently swapping two measurements of one operon.
    """
    described: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        described[row.protein].add(row.description)
    proofs: list[str] = []
    for key, locus in sorted(PHN_FUNCTION_CROSSWALK.items()):
        if stored_by_key.get(key) != locus:
            raise RuntimeError(
                f"{key!r} reconciles to {stored_by_key.get(key)!r}, not the {locus!r} "
                "its FUNCTION puts it at on the pinned assembly; the crosswalk this "
                "loader documents has changed"
            )
        sheet_says = PHN_SHEET_DESCRIPTION[key]
        if described[key] != {sheet_says}:
            raise RuntimeError(
                f"{SHEET_OVEREXPRESSION_PROTEOME} now describes {key!r} as "
                f"{sorted(described[key])}, not the {sheet_says!r} this loader records "
                "as contradicting the assembly; the disagreement must be re-decided"
            )
        proofs.append(
            f"{SHEET_OVEREXPRESSION_PROTEOME} {key}: keyed to {locus} on the function "
            f"both files state, while the sheet's Protein.Description cell says "
            f"{sheet_says}; the stored {SHEET_PROTEOME} records key it the same way"
        )
    return proofs


def assert_overexpression_operons(
    rows: Sequence[OverexpressionProteomeRow], stored_by_key: Mapping[str, str]
) -> list[str]:
    """Prove each overexpression label's operon from the proteome of its own samples.

    The Source Data labels one strain ``pSTABL2 (PP_2971-74)`` while the Methods, the
    Results, the Supplementary Fig. 12 caption's panels c and d, the Supplementary
    Fig. 10 caption and Supplementary Data 3 all say ``PP_2791-94``. The samples' own
    measured proteins settle it: nothing but an operon's members is quantified for it.
    """
    measured: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        match = OVEREXPRESSION_SAMPLE_RE.match(row.sample)
        if match is None:
            raise RuntimeError(
                f"{SHEET_OVEREXPRESSION_PROTEOME} sample {row.sample!r} is neither "
                "'<label>_Condition_<n>' nor '<label>_Control'"
            )
        measured[match.group("label")].add(stored_by_key[row.protein])
    proofs: list[str] = []
    for label, operon in sorted(OVEREXPRESSION_OPERONS.items()):
        if measured[label] != set(operon):
            raise RuntimeError(
                f"{SHEET_OVEREXPRESSION_PROTEOME}'s {label} samples quantify "
                f"{sorted(measured[label])}, not the operon {sorted(operon)} the "
                "Methods and the Results name"
            )
        sheet_label = OVEREXPRESSION_SHEET_LABELS[label]
        designations = re.findall(r"PP_\d{4}(?:-\d{2})?", sheet_label)
        labelled = {
            tag
            for designation in designations
            for tag in expand_locus_designation(designation)
        }
        proofs.append(
            f"{label}: its samples quantify exactly {sorted(operon)}, the operon the "
            f"Methods name; the sheet labels it {sheet_label!r}, which names "
            f"{sorted(labelled)} and so "
            + (
                "agrees"
                if labelled == set(operon)
                else "does NOT name those loci: it is the transposition the Methods, "
                "the Results, the Supplementary Fig. 12 caption body, the "
                "Supplementary Fig. 10 caption and Supplementary Data 3 all correct"
            )
        )
    return proofs


def overexpression_proteome_samples(
    rows: Sequence[OverexpressionProteomeRow],
) -> tuple[dict[str, str], list[str]]:
    """``{record sample: its Control sample}`` for the UNINDUCED arm, plus its proofs.

    One record per overexpression label, the uninduced condition, referenced against
    that label's own ``Control``. The six induced conditions are dropped for the same
    reason their titers are: the inducer's unit is not released
    (:data:`INDUCER_UNIT_GAP`).
    """
    levels: dict[str, float] = {}
    strains: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.sample in levels and levels[row.sample] != row.inducer_level:
            raise RuntimeError(
                f"{SHEET_OVEREXPRESSION_PROTEOME} sample {row.sample!r} carries two "
                "inducer levels"
            )
        levels[row.sample] = row.inducer_level
        strains[row.sample].add(row.strain)
    pairs: dict[str, str] = {}
    dropped: list[str] = []
    for sample, level in sorted(levels.items()):
        match = OVEREXPRESSION_SAMPLE_RE.match(sample)
        if match is None or match.group("condition") is None:
            continue
        if level != OVEREXPRESSION_UNINDUCED_LEVEL:
            dropped.append(sample)
            continue
        control = f"{match.group('label')}_Control"
        if control not in levels:
            raise RuntimeError(
                f"{SHEET_OVEREXPRESSION_PROTEOME} has no {control!r} for {sample!r}"
            )
        if levels[control] != OVEREXPRESSION_UNINDUCED_LEVEL:
            raise RuntimeError(
                f"{SHEET_OVEREXPRESSION_PROTEOME} control {control!r} is at inducer "
                f"level {levels[control]}, not {OVEREXPRESSION_UNINDUCED_LEVEL}"
            )
        if strains[control] != {OVEREXPRESSION_CONTROL_STRAIN}:
            raise RuntimeError(
                f"{SHEET_OVEREXPRESSION_PROTEOME} control {control!r} is strain "
                f"{sorted(strains[control])}, not {OVEREXPRESSION_CONTROL_STRAIN!r}"
            )
        pairs[sample] = control
    if len(pairs) != EXPECTED_OVEREXPRESSION_PROTEOME_RECORDS:
        raise RuntimeError(
            f"{SHEET_OVEREXPRESSION_PROTEOME} yields {len(pairs)} uninduced records; "
            f"this loader was written against "
            f"{EXPECTED_OVEREXPRESSION_PROTEOME_RECORDS}"
        )
    proofs = [
        f"{SHEET_OVEREXPRESSION_PROTEOME}: {len(pairs)} uninduced records "
        f"({', '.join(sorted(pairs))}) each against its own Control sample; "
        f"{len(dropped)} induced samples dropped because the inducer's unit is not "
        "released anywhere in the mirror"
    ]
    return pairs, proofs


def _overexpression_induced_samples(
    rows: Sequence[OverexpressionProteomeRow],
) -> list[str]:
    """``Supplementary Figure 12ac``'s induced samples, which carry an untypable dose."""
    return sorted(
        {
            row.sample
            for row in rows
            if row.inducer_level != OVEREXPRESSION_UNINDUCED_LEVEL
        }
    )


def _overexpression_cells(
    rows: Sequence[OverexpressionProteomeRow],
    sample: str,
    stored_by_key: Mapping[str, str],
) -> dict[str, list[float]]:
    """``{reconciled locus: its replicate Top3 signals}`` for one released sample.

    A repeated ``(sample, protein, replicate)`` raises: a doubled replicate would
    shrink the standard error the record carries.
    """
    cells: dict[str, list[float]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row.sample != sample:
            continue
        identity = (row.protein, row.replicate)
        if identity in seen:
            raise RuntimeError(
                f"{SHEET_OVEREXPRESSION_PROTEOME} {sample} {identity} appears twice; a "
                "repeated replicate would shrink the SE"
            )
        seen.add(identity)
        cells[stored_by_key[row.protein]].append(row.top3_signal)
    if not cells:
        raise RuntimeError(
            f"{SHEET_OVEREXPRESSION_PROTEOME} has no rows for sample {sample!r}"
        )
    return dict(cells)


def build_panel_titer_plan(
    path: str,
    *,
    stored_titers: Collection[float],
    control_titers: Mapping[int, Sequence[float]],
) -> PanelTiterPlan:
    """Read the four unstored titer panels and prove every cross-sheet claim.

    ``stored_titers`` is every ``Figure 4b`` value and ``control_titers`` its control
    cultures by DBTL cycle: the partition proof and the ``Figure 6d`` reference reuse
    are both joins against the already-stored campaign, so they are passed in rather
    than re-read.

    The record COUNT is not pinned here. It is pinned in the dataset's ``process()``,
    beside the sha256 check that makes it meaningful: a count assertion is a statement
    about one specific workbook, so it belongs where that workbook is verified.
    """
    ko_rows = read_ko_panel_rows(path)
    array_rows = read_ko_array_rows(path)
    offtarget_rows = read_offtarget_titer_rows(path)
    overexpression_rows = read_overexpression_titer_rows(path)

    proofs = _assert_panel_partition(
        reexported={
            SHEET_KO_PANEL: [
                row.titer_mg_per_l for row in ko_rows if row.arm == ARM_CRISPRI
            ],
            SHEET_KO_ARRAYS: [
                row.titer_mg_per_l for row in array_rows if row.arm == ARM_CRISPRI
            ],
        },
        novel={
            SHEET_KO_PANEL: [
                row.titer_mg_per_l for row in ko_rows if row.arm != ARM_CRISPRI
            ],
            SHEET_KO_ARRAYS: [
                row.titer_mg_per_l for row in array_rows if row.arm != ARM_CRISPRI
            ],
            SHEET_OFFTARGET_TITER: [row.titer_mg_per_l for row in offtarget_rows],
            SHEET_OVEREXPRESSION_TITER: [
                row.titer_mg_per_l for row in overexpression_rows
            ],
        },
        stored=stored_titers,
    )

    ko_records, ko_references, ko_proofs = _ko_panel_groups(ko_rows)
    array_records, array_proofs = _ko_array_groups(array_rows, control_titers)
    shared_key = f"non_target:{PROTEOME_BACKGROUND_DELETION}"
    by_key = {reference.key: reference for reference in ko_references}
    if shared_key not in by_key:
        raise RuntimeError(
            f"{SHEET_KO_PANEL} has no Δ{PROTEOME_BACKGROUND_DELETION} "
            f"{ARM_KO_NONTARGET} arm for {SHEET_OFFTARGET_TITER}'s reference to be "
            "deduplicated against"
        )
    ko_target_group = f"{PROTEOME_BACKGROUND_DELETION}/{ARM_KO_TARGET}"
    ko_target_arm = next(
        record.titers_mg_per_l
        for record in ko_records
        if record.group == ko_target_group
    )
    offtarget_records, shared_reference, offtarget_proofs = _offtarget_groups(
        offtarget_rows, by_key[shared_key], ko_target_arm
    )
    by_key[shared_key] = shared_reference
    (overexpression_records, overexpression_reference, overexpression_proofs, drops) = (
        _overexpression_titer_groups(overexpression_rows)
    )
    by_key[overexpression_reference.key] = overexpression_reference

    records = [*ko_records, *array_records, *offtarget_records, *overexpression_records]
    return PanelTiterPlan(
        records=records,
        references=[by_key[key] for key in sorted(by_key)],
        stored_cycle_reference={
            f"stored_cycle:{KO_ARRAY_CONTROL_CYCLE}": KO_ARRAY_CONTROL_CYCLE
        },
        proofs=[
            *proofs,
            *ko_proofs,
            *array_proofs,
            *offtarget_proofs,
            *overexpression_proofs,
        ],
        drops=drops,
    )


# --------------------------------------------------------------------------- #
# Family 1: isoprenol titer
# --------------------------------------------------------------------------- #
@register_dataset
class IsoprenolTiterCarruthers2025Dataset(ExperimentDataset):
    """Carruthers 2025 per-strain isoprenol titer across six DBTL cycles."""

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"
    #: Every guide target must resolve to a locus of the pinned assembly. Measured on
    #: the pinned workbook: 121 of 121 are current standard locus tags, so a value
    #: below 1.0 means the annotation or the released names moved and the build stops.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 1.0

    def __init__(
        self,
        root: str = "data/torchcell/isoprenol_titer_carruthers2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome resolves guide targets and chassis symbols."""
        self.pputida_genome = pputida_genome
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
        """The Source Data workbook and Supplementary Data 1's target table."""
        return [SOURCE_DATA_FILENAME, TARGETS_FILENAME]

    def download(self) -> None:
        """Link both mirror files into ``raw/`` after verifying each against its pin."""
        _link_mirror_files(
            self.raw_dir,
            (
                (SOURCE_DATA_REL, SOURCE_DATA_FILENAME, SOURCE_DATA_SHA256),
                (TARGETS_REL, TARGETS_FILENAME, TARGETS_SHA256),
            ),
        )
        log.info("Carruthers 2025 titer artifacts linked into %s", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    @post_process
    def process(self) -> None:
        """Build one titer record per (construct, cycle) strain; write LMDB."""
        verify_raw_files(
            self.raw_dir,
            {
                SOURCE_DATA_FILENAME: SOURCE_DATA_SHA256,
                TARGETS_FILENAME: TARGETS_SHA256,
            },
        )
        source_path = osp.join(self.raw_dir, SOURCE_DATA_FILENAME)
        rows = read_titer_rows(source_path)
        genome = self._genome()

        strains: dict[tuple[str, int], list[TiterRow]] = defaultdict(list)
        controls: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            if row.is_control:
                controls[row.cycle].append(row.titer_mg_per_l)
            else:
                strains[(row.construct_name, row.cycle)].append(row)

        self._assert_control_sources(source_path, controls)
        self._assert_target_means(source_path, strains)

        replicate_counts = Counter(len(group) for group in strains.values())
        modal, _ = replicate_counts.most_common(1)[0]
        if modal != int(N_REPLICATES.value):
            raise RuntimeError(
                f"the modal replicate count is {modal}, not the stated "
                f"{N_REPLICATES.value}; the replicate design changed"
            )

        plan = build_panel_titer_plan(
            source_path,
            stored_titers={row.titer_mg_per_l for row in rows},
            control_titers=controls,
        )
        if len(plan.records) != EXPECTED_PANEL_TITER_RECORDS:
            raise RuntimeError(
                f"the four Source Data panels build {len(plan.records)} records; the "
                f"sha256-pinned workbook holds {EXPECTED_PANEL_TITER_RECORDS}"
            )

        tags = sorted(
            {tag for name, _ in strains for tag in parse_construct(name)[0]}
            | {
                tag
                for record in plan.records
                for tag in (
                    *record.deletions,
                    *record.knockdowns,
                    *record.native_copies,
                )
            }
        )
        stored, report = reconcile_locus_tags(genome, pd.Series(tags), label=self.name)
        report.require_resolved(self.MIN_RESOLVED_FRACTION)
        if report.outside_namespace:
            raise RuntimeError(
                f"{self.name}: guide targets outside {KT2440_NAMESPACE}: "
                f"{report.outside_namespace}"
            )
        stored_by_tag = dict(zip(tags, stored, strict=True))
        common = _standard_names(genome, stored_by_tag.values())

        reference_genome = chassis_reference(genome)
        environment = production_environment()
        pathway = pathway_perturbations()
        references = {
            cycle: ProductTiterExperimentReference(
                dataset_name=self.name,
                genome_reference=reference_genome,
                environment_reference=environment.model_copy(),
                phenotype_reference=titer_phenotype(values, is_reference=True),
            )
            for cycle, values in sorted(controls.items())
        }
        if reference_genome.species != SPECIES:
            raise RuntimeError(
                f"the pinned assembly's species is {reference_genome.species!r}, not "
                f"{SPECIES!r}; a native extra copy declaring {SPECIES!r} would be read "
                "as a gene of another genome by the host-containment gate"
            )
        overexpression = overexpression_environment()
        panel_references: dict[str, ProductTiterExperimentReference] = {
            key: references[cycle] for key, cycle in plan.stored_cycle_reference.items()
        }
        for panel_reference in plan.references:
            panel_references[panel_reference.key] = ProductTiterExperimentReference(
                dataset_name=self.name,
                genome_reference=reference_genome,
                environment_reference=(
                    overexpression.model_copy()
                    if panel_reference.key == "overexpression_control"
                    else environment.model_copy()
                ),
                phenotype_reference=titer_phenotype(
                    list(panel_reference.titers_mg_per_l), is_reference=True
                ),
            )
        pub = publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        filter_rows: list[dict[str, Any]] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for (construct, cycle), group in tqdm(
                sorted(strains.items()), desc="carruthers2025-titer"
            ):
                tag_list, extras = parse_construct(construct)
                perturbations = [
                    crispri_perturbation(stored_by_tag[tag], common[stored_by_tag[tag]])
                    for tag in tag_list
                ]
                experiment = ProductTiterExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=[*pathway, *perturbations]),
                    environment=environment,
                    phenotype=titer_phenotype(
                        [row.titer_mg_per_l for row in group], is_reference=False
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, references[cycle], pub, itxn),
                )
                filter_rows.append(
                    {
                        "construct": construct,
                        "cycle": cycle,
                        "n_replicates": len(group),
                        "n_targets": len(tag_list),
                        "non_targeting_tokens": ";".join(extras),
                        "passed_crispri_filter": group[0].passed_filter,
                    }
                )
                idx += 1
            for record in tqdm(plan.records, desc="carruthers2025-titer-panels"):
                experiment = ProductTiterExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(
                        perturbations=[
                            *pathway,
                            *(
                                deletion_perturbation(
                                    stored_by_tag[tag], common[stored_by_tag[tag]]
                                )
                                for tag in record.deletions
                            ),
                            *(
                                native_copy_perturbation(
                                    stored_by_tag[tag], common[stored_by_tag[tag]]
                                )
                                for tag in record.native_copies
                            ),
                            *(
                                crispri_perturbation(
                                    stored_by_tag[tag], common[stored_by_tag[tag]]
                                )
                                for tag in record.knockdowns
                            ),
                        ]
                    ),
                    environment=(
                        overexpression
                        if record.overexpression_environment
                        else environment
                    ),
                    phenotype=titer_phenotype(
                        list(record.titers_mg_per_l), is_reference=False
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(
                        experiment, panel_references[record.reference_key], pub, itxn
                    ),
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(filter_rows).to_csv(
            osp.join(self.preprocess_dir, "pass_filter.csv"), index=False
        )
        pd.DataFrame(
            [
                {
                    "cycle": cycle,
                    "n_control_cultures": len(values),
                    "mean_titer_ug_per_ml": statistics.fmean(values),
                    "sd_titer_ug_per_ml": statistics.stdev(values),
                    "cv_percent": statistics.stdev(values)
                    / statistics.fmean(values)
                    * 100.0,
                }
                for cycle, values in sorted(controls.items())
            ]
        ).to_csv(osp.join(self.preprocess_dir, "cycle_controls.csv"), index=False)
        pd.DataFrame(
            [
                {
                    "sheet": record.sheet,
                    "group": record.group,
                    "n_replicates": len(record.titers_mg_per_l),
                    "deletions": ";".join(record.deletions),
                    "knockdowns": ";".join(record.knockdowns),
                    "native_copies": ";".join(record.native_copies),
                    "reference_key": record.reference_key,
                    "note": record.note or "",
                }
                for record in plan.records
            ]
        ).to_csv(osp.join(self.preprocess_dir, "source_data_panels.csv"), index=False)
        Path(osp.join(self.preprocess_dir, "panel_proofs.json")).write_text(
            json.dumps(plan.proofs, indent=2)
        )
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(rows),
                control_rows=sum(len(v) for v in controls.values()),
                candidate_records=len(strains)
                + len(plan.records)
                + sum(rule.n_records for rule in plan.drops),
                kept_records=idx,
                dropped_records=sum(rule.n_records for rule in plan.drops),
                rules=plan.drops,
                reconciliation=report,
                notes=[
                    "nothing is dropped from the CRISPRi campaign: every non-control "
                    f"{SHEET_TITER} culture is in a record, and the paper is explicit "
                    "that its CRISPRi filter shaped only model training "
                    f"({_Q_NO_EXCLUSION})",
                    "the control cultures are the per-cycle phenotype_reference, not "
                    f"records ({_Q_CONTROL_N})",
                    "the authors' per-strain pass/fail CRISPRi filter is in "
                    "preprocess/pass_filter.csv; it reports whether a designed "
                    "knockdown was realized, which no schema axis can type today",
                    f"{len(plan.records)} further records come from {SHEET_KO_PANEL}, "
                    f"{SHEET_KO_ARRAYS}, {SHEET_OFFTARGET_TITER} and "
                    f"{SHEET_OVEREXPRESSION_TITER}; their cross-sheet proofs are in "
                    "preprocess/panel_proofs.json and their genotypes in "
                    "preprocess/source_data_panels.csv",
                    *plan.proofs,
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Carruthers2025 titer: %d records = %d (construct, cycle) campaign strains "
            "from %d cultures (%d control) + %d from the four Source Data panels; "
            "replicate histogram %s; %d host loci",
            idx,
            len(strains),
            len(rows),
            sum(len(v) for v in controls.values()),
            len(plan.records),
            dict(sorted(Counter(len(g) for g in strains.values()).items())),
            len(tags),
        )

    def _assert_control_sources(
        self, source_path: str, controls: dict[int, list[float]]
    ) -> None:
        """``Figure 2c`` and ``Figure 4b`` must report the same control cultures.

        DBTL0 is exported at a different precision in the two sheets, so the tolerance
        is 1e-5 mg/L; the measured worst disagreement is 3.3e-6 in DBTL0 and 0 elsewhere.
        """
        expected_n = dict(CONTROL_N.value)
        for cycle, values in sorted(controls.items()):
            want = expected_n["DBTL0"] if cycle == 0 else expected_n["DBTL1-6"]
            if len(values) != want:
                raise RuntimeError(
                    f"DBTL{cycle} has {len(values)} control cultures; the Fig. 2 "
                    f"caption states {want}"
                )
        other = read_control_titers(source_path)
        if set(other) != set(controls):
            raise RuntimeError(
                f"{SHEET_CONTROLS} covers cycles {sorted(other)} and {SHEET_TITER} "
                f"{sorted(controls)}"
            )
        for cycle, values in sorted(controls.items()):
            mine, theirs = sorted(values), sorted(other[cycle])
            if len(mine) != len(theirs):
                raise RuntimeError(
                    f"cycle {cycle}: {len(mine)} control cultures in {SHEET_TITER}, "
                    f"{len(theirs)} in {SHEET_CONTROLS}"
                )
            worst = max(abs(a - b) for a, b in zip(mine, theirs, strict=True))
            if worst > 1e-5:
                raise RuntimeError(
                    f"cycle {cycle} control titers disagree by {worst} between "
                    f"{SHEET_TITER} and {SHEET_CONTROLS}"
                )

    def _assert_target_means(
        self, source_path: str, strains: dict[tuple[str, int], list[TiterRow]]
    ) -> None:
        """Both released per-target means must equal the mean over the DBTL0 replicates.

        ``Figure 3a`` is exported at full precision (tolerance 1e-6) and Supplementary
        Data 1 at two decimals (tolerance 0.005). Measured: 119 of 119 and 118 of 118
        joinable targets agree.
        """
        cycle0 = {
            name: [row.titer_mg_per_l for row in group]
            for (name, cycle), group in strains.items()
            if cycle == 0
        }
        oracles = (
            (SHEET_TARGET_MEANS, read_target_means(source_path), 1e-6),
            (
                "Supplementary Data 1",
                read_si_target_means(osp.join(self.raw_dir, TARGETS_FILENAME)),
                5e-3,
            ),
        )
        for label, released, tol in oracles:
            shared = sorted(set(released) & set(cycle0))
            if not shared:
                raise RuntimeError(f"{label} joins no DBTL0 construct by name")
            disagreements = [
                (tag, statistics.fmean(cycle0[tag]), released[tag])
                for tag in shared
                if abs(statistics.fmean(cycle0[tag]) - released[tag]) > tol
            ]
            if disagreements:
                raise RuntimeError(
                    f"{label}: {len(disagreements)} of {len(shared)} per-target means "
                    f"disagree with the replicate mean by more than {tol}: "
                    f"{disagreements[:5]}"
                )
            log.info(
                "%s cross-source check: all %d joinable targets agree within %g; %d of "
                "its %d targets do not join a DBTL0 construct name",
                label,
                len(shared),
                tol,
                len(set(released) - set(cycle0)),
                len(released),
            )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Family 2: the released off-target-study proteome
# --------------------------------------------------------------------------- #
@register_dataset
class ProteomeCarruthers2025Dataset(ExperimentDataset):
    """Carruthers 2025 Top3 proteome of the PP_0815 off-target-study strains."""

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"
    #: Measured on the pinned workbook: 1,425 of 1,501 protein keys (0.9494) resolve to
    #: a locus of this assembly. The threshold sits just below that, because the 76 that
    #: do not are the heterologous, marker and contaminant proteins the DIA-NN search
    #: database was built to include; a real drop below this means the keying changed.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 0.94

    def __init__(
        self,
        root: str = "data/torchcell/proteome_carruthers2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome resolves protein keys and chassis symbols."""
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
        """The Source Data workbook, which holds the released abundance matrix."""
        return [SOURCE_DATA_FILENAME]

    def download(self) -> None:
        """Link the Source Data workbook into ``raw/`` after verifying its pin."""
        _link_mirror_files(
            self.raw_dir, ((SOURCE_DATA_REL, SOURCE_DATA_FILENAME, SOURCE_DATA_SHA256),)
        )
        log.info("Carruthers 2025 proteome artifact linked into %s", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    @staticmethod
    def _aggregate(
        cells: dict[str, list[float]], sample: str
    ) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
        """Per-protein mean, SE and replicate count for one sample.

        A released 0 is the number the source reports for a protein with no Top3
        peptide signal in that replicate; it is kept, not imputed and not dropped, so
        ``n_replicates`` is the replicate count the sample actually has.
        """
        abundance: dict[str, float] = {}
        se: dict[str, float] = {}
        n_reps: dict[str, int] = {}
        for protein, values in cells.items():
            n = len(values)
            if n < 1:
                raise RuntimeError(f"{sample}/{protein}: no replicate values")
            abundance[protein] = statistics.fmean(values)
            n_reps[protein] = n
            se[protein] = (
                statistics.stdev(values) / math.sqrt(n) if n > 1 else float("nan")
            )
        return abundance, se, n_reps

    def _genotype(
        self,
        sample: str,
        pathway: list[HeterologousPathwayPerturbation],
        background_deletion: BacterialDeletionPerturbation,
        common: dict[str, str],
    ) -> Genotype:
        """The strain one proteome sample was taken from.

        Every sample is the chassis plus pIY670 plus the chromosomal ``PP_0815``
        knockout; the off-target samples add that sample's CRISPRi target, and the
        positive control adds the ``PP_0815`` guide itself. The non-targeting control is
        the reference and never reaches this method.
        """
        perturbations: list[Any] = [*pathway, background_deletion]
        if sample == PROTEOME_TARGET_SAMPLE:
            tag = PROTEOME_BACKGROUND_DELETION
        else:
            match = PROTEOME_SAMPLE_RE.match(sample)
            if match is None:
                raise RuntimeError(
                    f"proteome sample {sample!r} is neither the non-targeting control, "
                    "the PP_0815 target control, nor a JBEI_OTS_<tag>[_n][_P4]_48hr "
                    "off-target sample"
                )
            tag = match.group("tag")
        perturbations.append(crispri_perturbation(tag, common.get(tag, tag)))
        return Genotype(perturbations=perturbations)

    @post_process
    def process(self) -> None:
        """Build one protein-abundance record per released proteome sample; write LMDB."""
        verify_raw_files(self.raw_dir, {SOURCE_DATA_FILENAME: SOURCE_DATA_SHA256})
        rows = read_proteome_rows(osp.join(self.raw_dir, SOURCE_DATA_FILENAME))
        genome = self._genome()

        overexpression_rows = read_overexpression_proteome_rows(
            osp.join(self.raw_dir, SOURCE_DATA_FILENAME)
        )
        keys = sorted({row.protein for row in rows})
        stored, report = reconcile_locus_tags(genome, pd.Series(keys), label=self.name)
        report.require_resolved(self.MIN_RESOLVED_FRACTION)
        stored_by_key = dict(zip(keys, stored, strict=True))
        novel = sorted({row.protein for row in overexpression_rows} - set(keys))
        if novel:
            raise RuntimeError(
                f"{SHEET_OVEREXPRESSION_PROTEOME} quantifies {novel}, which "
                f"{SHEET_PROTEOME} does not, so the two panels would key one protein "
                "two ways; the reconciliation must cover both sheets"
            )
        proofs = [
            *assert_phn_crosswalk(overexpression_rows, stored_by_key),
            *assert_overexpression_operons(overexpression_rows, stored_by_key),
        ]
        overexpression_pairs, pair_proofs = overexpression_proteome_samples(
            overexpression_rows
        )
        proofs.extend(pair_proofs)

        accessions: dict[str, set[str]] = defaultdict(set)
        entries: dict[str, set[str]] = defaultdict(set)
        descriptions: dict[str, str] = {}
        for row in rows:
            accessions[row.protein].add(row.accession)
            entries[row.protein].add(row.entry_name)
            descriptions.setdefault(row.protein, row.description)

        outside = set(report.outside_namespace)
        merged = {key for key, group in accessions.items() if len(group) > 1}
        dropped_keys = sorted(outside | merged)
        kept_keys = [key for key in keys if key not in set(dropped_keys)]
        if not kept_keys:
            raise RuntimeError(f"{self.name}: every protein key was dropped")

        cells: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            if row.protein in set(dropped_keys):
                continue
            identity = (row.sample, row.protein, row.replicate)
            if identity in seen:
                raise RuntimeError(
                    f"{identity} appears twice; a repeated replicate would shrink the SE"
                )
            seen.add(identity)
            cells[row.sample][stored_by_key[row.protein]].append(row.top3_signal)

        if PROTEOME_REFERENCE_SAMPLE not in cells:
            raise RuntimeError(
                f"{self.name}: the non-targeting reference sample "
                f"{PROTEOME_REFERENCE_SAMPLE!r} is not in the released matrix"
            )
        ref_abundance, ref_se, ref_n = self._aggregate(
            cells[PROTEOME_REFERENCE_SAMPLE], PROTEOME_REFERENCE_SAMPLE
        )
        observed_reps = {
            n for sample in cells.values() for n in (len(v) for v in sample.values())
        }
        if observed_reps != {int(PROTEOME_N_REPLICATES.value)}:
            raise RuntimeError(
                f"the panel's replicate counts are {sorted(observed_reps)}; the "
                f"Supplementary Fig. 13 caption states {PROTEOME_N_REPLICATES.value}"
            )

        reference_genome = chassis_reference(genome)
        environment = production_environment()
        pathway = pathway_perturbations()
        background_deletion = BacterialDeletionPerturbation(
            systematic_gene_name=PROTEOME_BACKGROUND_DELETION,
            perturbed_gene_name=PROTEOME_BACKGROUND_DELETION,
            gene_namespace=KT2440_NAMESPACE,
        )
        common = _standard_names(
            genome, [PROTEOME_BACKGROUND_DELETION, *stored_by_key.values()]
        )
        reference = BacterialProteinAbundanceExperimentReference(
            dataset_name=self.name,
            genome_reference=reference_genome,
            environment_reference=environment.model_copy(),
            phenotype_reference=ProteinAbundancePhenotype(
                protein_abundance=ref_abundance,
                protein_abundance_se=ref_se,
                n_replicates=ref_n,
                measurement_type=str(TOP3.value),
            ),
        )
        pub = publication()

        samples = sorted(set(cells) - {PROTEOME_REFERENCE_SAMPLE})
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        sample_rows: list[dict[str, Any]] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for sample in tqdm(samples, desc="carruthers2025-proteome"):
                abundance, se, n_reps = self._aggregate(cells[sample], sample)
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=self._genotype(
                        sample, pathway, background_deletion, common
                    ),
                    environment=environment,
                    phenotype=ProteinAbundancePhenotype(
                        protein_abundance=abundance,
                        protein_abundance_se=se,
                        n_replicates=n_reps,
                        measurement_type=str(TOP3.value),
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                sample_rows.append(
                    {
                        "sheet": SHEET_PROTEOME,
                        "sample": sample,
                        "n_proteins": len(abundance),
                        "n_replicates": max(n_reps.values()),
                    }
                )
                idx += 1
            operons = dict(OVEREXPRESSION_OPERON_SOURCE.value)
            overexpression = overexpression_environment()
            for sample, control in tqdm(
                sorted(overexpression_pairs.items()),
                desc="carruthers2025-proteome-overexpression",
            ):
                label = str(OVEREXPRESSION_SAMPLE_RE.match(sample).group("label"))  # type: ignore[union-attr]
                abundance, se, n_reps = self._aggregate(
                    _overexpression_cells(overexpression_rows, sample, stored_by_key),
                    sample,
                )
                control_abundance, control_se, control_n = self._aggregate(
                    _overexpression_cells(overexpression_rows, control, stored_by_key),
                    control,
                )
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(
                        perturbations=[
                            *pathway,
                            *(
                                native_copy_perturbation(tag, common.get(tag, tag))
                                for tag in operons[label]
                            ),
                        ]
                    ),
                    environment=overexpression,
                    phenotype=ProteinAbundancePhenotype(
                        protein_abundance=abundance,
                        protein_abundance_se=se,
                        n_replicates=n_reps,
                        measurement_type=str(TOP3.value),
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(
                        experiment,
                        BacterialProteinAbundanceExperimentReference(
                            dataset_name=self.name,
                            genome_reference=reference_genome,
                            environment_reference=overexpression.model_copy(),
                            phenotype_reference=ProteinAbundancePhenotype(
                                protein_abundance=control_abundance,
                                protein_abundance_se=control_se,
                                n_replicates=control_n,
                                measurement_type=str(TOP3.value),
                            ),
                        ),
                        pub,
                        itxn,
                    ),
                )
                sample_rows.append(
                    {
                        "sheet": SHEET_OVEREXPRESSION_PROTEOME,
                        "sample": sample,
                        "n_proteins": len(abundance),
                        "n_replicates": max(n_reps.values()),
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(sample_rows).to_csv(
            osp.join(self.preprocess_dir, "samples.csv"), index=False
        )
        Path(osp.join(self.preprocess_dir, "panel_proofs.json")).write_text(
            json.dumps(proofs, indent=2)
        )
        pd.DataFrame(
            [
                {
                    "protein_key": key,
                    "accessions": ";".join(sorted(accessions[key])),
                    "entry_names": ";".join(sorted(entries[key])),
                    "description": descriptions[key],
                    "reason": (
                        "merged_accessions"
                        if key in merged
                        else "not_a_locus_of_the_pinned_assembly"
                    ),
                }
                for key in dropped_keys
            ]
        ).to_csv(osp.join(self.preprocess_dir, "dropped_protein_keys.csv"), index=False)
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(rows) + len(overexpression_rows),
                control_rows=1 + len(set(overexpression_pairs.values())),
                candidate_records=len(samples)
                + len(overexpression_pairs)
                + len(_overexpression_induced_samples(overexpression_rows)),
                kept_records=idx,
                dropped_records=len(
                    _overexpression_induced_samples(overexpression_rows)
                ),
                rules=[
                    DropRule(
                        rule="protein_key_is_not_a_locus_of_the_pinned_assembly",
                        scope="protein_key",
                        description=(
                            "the key names a heterologous pathway protein, the dCas9 "
                            "effector, a resistance marker or a proteomic contaminant, "
                            "which the DIA-NN search database was built to include "
                            f"({_Q_DIANN_DB}); it is no locus of "
                            f"{KT2440_ASSEMBLY_SET}, so it has no gene node to key an "
                            "abundance to"
                        ),
                        n_records=0,
                        items=sorted(outside),
                    ),
                    DropRule(
                        rule="protein_key_merges_two_accessions",
                        scope="protein_key",
                        description=(
                            "the released table files two distinct protein groups under "
                            "one symbol, so the key names two proteins and its "
                            "abundance cannot be attributed to either"
                        ),
                        n_records=0,
                        items=sorted(merged),
                    ),
                    DropRule(
                        rule="inducer_concentration_has_no_released_unit",
                        scope="strain",
                        description=str(INDUCER_UNIT_GAP.note),
                        n_records=len(
                            _overexpression_induced_samples(overexpression_rows)
                        ),
                        items=_overexpression_induced_samples(overexpression_rows),
                    ),
                ],
                reconciliation=report,
                notes=[
                    f"no SAMPLE is dropped; {len(dropped_keys)} of {len(keys)} protein "
                    f"KEYS are, leaving {len(kept_keys)} in every record's abundance map",
                    f"the non-targeting control {PROTEOME_REFERENCE_SAMPLE} is the "
                    "phenotype_reference, not a record",
                    "PP_0977, PP_1638 and PP_3416 appear as several samples because "
                    f"'{_Q_SI_THREE_CULTURES}'; each is an independent culture of the "
                    "same genotype and is kept as its own record, never averaged",
                    "a released Top3 value of 0 is kept verbatim; the source floors the "
                    "companion percent-abundance column at 1e-05 for those cells, and "
                    "this loader imputes nothing",
                    f"{len(overexpression_pairs)} further records come from "
                    f"{SHEET_OVEREXPRESSION_PROTEOME}, the second per-protein "
                    "per-replicate abundance matrix the Source Data carries. It reads "
                    "the same Top_3pep_counts_mean column on the same Top3 scale, so "
                    "both panels store one measurement_type; each record is referenced "
                    "against its own label's Control sample, whose key set it matches",
                    f"{SHEET_CONTROL_PROTEOME} is NOT loaded: {CONTROL_PROTEOME_DEFERRAL.note}",
                    *proofs,
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Carruthers2025 proteome: %d records = %d %s samples (+1 non-targeting "
            "reference) x %d protein keys + %d uninduced %s samples; %d keys dropped "
            "(%d outside the namespace, %d merged accessions) from %d released cells",
            idx,
            len(samples),
            SHEET_PROTEOME,
            len(kept_keys),
            len(overexpression_pairs),
            SHEET_OVEREXPRESSION_PROTEOME,
            len(dropped_keys),
            len(outside),
            len(merged),
            len(rows) + len(overexpression_rows),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Family 3: the deposited Dryad campaign proteome, paired to the 465 titer records
# --------------------------------------------------------------------------- #
class CampaignAggregate(BaseModel):
    """One strain-cycle's aggregated profile, before it becomes a phenotype."""

    model_config = ConfigDict(extra="forbid")

    abundance: dict[str, float]
    se: dict[str, float]
    n_replicates: dict[str, int]
    cultures: int = Field(description="cultures the strain-cycle has in the matrix")


def aggregate_campaign_profile(
    proteome: CampaignProteome,
    row_indices: Sequence[int],
    locus_of: Mapping[str, str],
    *,
    label: str,
) -> CampaignAggregate:
    """Mean, SE and replicate count per locus over one strain-cycle's cultures.

    A blank cell is an ABSENT measurement (:data:`CAMPAIGN_BLANK_IS_ABSENT`), so it is
    excluded from the protein's mean and from its ``n_replicates``, and a protein blank
    in every culture of the strain-cycle carries no key at all. A released 0 is a
    present measurement and is averaged in verbatim. The SE is ``SD / sqrt(n)`` over
    the present values, and ``nan`` for a protein present in exactly one culture, which
    is the same convention the panel family uses.
    """
    values: dict[str, list[float]] = defaultdict(list)
    for row_index in row_indices:
        for accession, locus in locus_of.items():
            cell = proteome.cells.get((row_index, accession))
            if cell is not None:
                values[locus].append(cell)
    if not values:
        raise RuntimeError(f"{label}: every protein is blank in every culture")
    abundance: dict[str, float] = {}
    se: dict[str, float] = {}
    n_replicates: dict[str, int] = {}
    for locus, present in values.items():
        n = len(present)
        abundance[locus] = statistics.fmean(present)
        n_replicates[locus] = n
        se[locus] = statistics.stdev(present) / math.sqrt(n) if n > 1 else float("nan")
    return CampaignAggregate(
        abundance=abundance, se=se, n_replicates=n_replicates, cultures=len(row_indices)
    )


def assert_campaign_rows_are_percentages(proteome: CampaignProteome) -> str:
    """Every released culture row sums to 100 within :data:`CAMPAIGN_ROW_SUM_TOL`.

    The strong form of :data:`CAMPAIGN_TOP3`'s "Percentage of the proteome": a row that
    did not sum to 100 would mean the released cells are not percentages and the whole
    family's unit would be wrong. Checked over ALL accessions, including the ones with
    no locus tag, because the percentage is of the whole measured proteome.
    """
    sums: list[float] = []
    for row_index in range(len(proteome.cultures)):
        sums.append(
            sum(
                value
                for accession in proteome.accessions
                if (value := proteome.cells.get((row_index, accession))) is not None
            )
        )
    low, high = min(sums), max(sums)
    if (
        abs(low - 100.0) > CAMPAIGN_ROW_SUM_TOL
        or abs(high - 100.0) > CAMPAIGN_ROW_SUM_TOL
    ):
        raise RuntimeError(
            f"released culture rows sum to {low:.4f}-{high:.4f} over all "
            f"{len(proteome.accessions)} accessions; {CAMPAIGN_TOP3.quote!r} says each "
            "cell is a percentage of the proteome, so a row sums to 100 within "
            f"{CAMPAIGN_ROW_SUM_TOL}"
        )
    return (
        f"L3: all {len(sums)} released culture rows sum to {low:.4f}-{high:.4f} over "
        f"the {len(proteome.accessions)} accessions, which is what makes the cells "
        "percentages of the proteome"
    )


def assert_campaign_titers_match_source_data(
    proteome: CampaignProteome, titer_rows: Sequence[TiterRow]
) -> str:
    """The Dryad matrix's titer column against the Source Data's, culture by culture.

    Two independently released files of one measurement, so the join is exact and any
    difference refuses. The Dryad file covers 1,497 of the Source Data's 1,506
    cultures; the nine it does not are named in the returned proof.
    """
    source = {
        (row.construct_name, row.cycle, row.replicate): row.titer_mg_per_l
        for row in titer_rows
    }
    missing: list[tuple[str, int, int]] = []
    worst = 0.0
    for culture in proteome.cultures:
        key = (culture.construct_name, culture.cycle, culture.replicate)
        if key not in source:
            raise RuntimeError(
                f"{DRYAD_TOP3_FILENAME} culture {key} is in no Source Data row; the "
                "two releases do not describe the same cultures"
            )
        worst = max(worst, abs(source[key] - culture.titer_mg_per_l))
    joined = {
        (culture.construct_name, culture.cycle, culture.replicate)
        for culture in proteome.cultures
    }
    missing = sorted(key for key in source if key not in joined)
    if worst != 0.0:
        raise RuntimeError(
            f"{DRYAD_TOP3_FILENAME} and {SHEET_TITER} disagree about a culture titer by "
            f"{worst} mg/L; they are two exports of one GC-FID measurement"
        )
    return (
        f"L4: the Dryad matrix's Isoprenol_titer equals {SHEET_TITER}'s titer EXACTLY "
        f"(max |diff| 0.0 mg/L) on all {len(joined)} cultures the two share; the "
        f"{len(missing)} Source Data cultures with no proteome row are "
        + ", ".join(
            f"{name}-R{replicate} (DBTL{cycle})" for name, cycle, replicate in missing
        )
    )


def assert_campaign_metadata(
    rows: Sequence[CampaignMetadataRow], proteome: CampaignProteome
) -> list[str]:
    """The metadata CSV against the matrix and against the shared environment.

    Three independent checks, each a cross-source statement about bytes this loader did
    not author: the metadata describes the same cultures as the matrix, its environment
    columns carry one value each and that value is what :func:`production_environment`
    serves, and its ``Genes_targeted_for_CRISPRi`` column is the ``Line`` cell verbatim
    rather than a gene list (:data:`CAMPAIGN_GENE_LIST_IS_THE_LINE`).
    """
    metadata_cultures = Counter((row.cycle, row.line_name) for row in rows)
    matrix_cultures = Counter(
        (culture.cycle, f"{culture.construct_name}-R{culture.replicate}")
        for culture in proteome.cultures
    )
    if metadata_cultures != matrix_cultures:
        raise RuntimeError(
            f"{DRYAD_METADATA_FILENAME} and {DRYAD_TOP3_FILENAME} describe different "
            "cultures"
        )
    deviating = [
        row.line_name for row in rows if row.values != CAMPAIGN_ENVIRONMENT_COLUMNS
    ]
    if deviating:
        raise RuntimeError(
            f"{DRYAD_METADATA_FILENAME} rows {deviating[:5]} carry environment values "
            "other than the one set every row was measured to carry; the campaign's "
            "environment is no longer a constant and production_environment() cannot "
            "stand for it"
        )
    environment = production_environment()
    culture_format = environment.culture_format
    if culture_format is None:
        raise RuntimeError("production_environment() carries no culture_format")
    if (
        environment.temperature is None
        or environment.duration_hours is None
        or culture_format.working_volume_ul is None
        or culture_format.shaking_rpm is None
    ):
        raise RuntimeError(
            "production_environment() carries no temperature, duration, working volume "
            f"or shaking speed, which the {DRYAD_METADATA_FILENAME} columns are "
            "asserted against"
        )
    expected: dict[str, float] = {
        "Growth_temperature_Celsius": float(environment.temperature.value),
        "Culture_volume": culture_format.working_volume_ul / 1000.0,
        "Shaking_speed_rpm": float(culture_format.shaking_rpm),
        "Assay_time_point": float(environment.duration_hours),
        "Inducer_concentration": float(INDUCER_G_PER_L.value),
    }
    for column, served in expected.items():
        released = float(CAMPAIGN_ENVIRONMENT_COLUMNS[column])
        if float(served) != released:
            raise RuntimeError(
                f"{DRYAD_METADATA_FILENAME} states {column}={released}, the served "
                f"environment {served}"
            )
    if CAMPAIGN_ENVIRONMENT_COLUMNS["Culture_format"] != str(
        CULTURE_FORMAT.value["vessel"]
    ):
        raise RuntimeError(
            f"{DRYAD_METADATA_FILENAME} states the vessel "
            f"{CAMPAIGN_ENVIRONMENT_COLUMNS['Culture_format']!r}, the Methods "
            f"{CULTURE_FORMAT.value['vessel']!r}"
        )
    gene_list_rows = 0
    for row in rows:
        if row.genes_targeted is None:
            if not row.line_name.startswith(KO_ARRAY_CONTROL_LINE):
                raise RuntimeError(
                    f"{DRYAD_METADATA_FILENAME} leaves Genes_targeted_for_CRISPRi blank "
                    f"for the non-control culture {row.line_name!r}"
                )
            continue
        if row.genes_targeted != row.line_cell:
            raise RuntimeError(
                f"{DRYAD_METADATA_FILENAME} row {row.line_name!r}: "
                f"Genes_targeted_for_CRISPRi {row.genes_targeted!r} is not its Line "
                f"cell {row.line_cell!r}"
            )
        gene_list_rows += 1
    prefixed = sorted(
        {
            culture.cycle
            for culture in proteome.cultures
            if culture.line_cell != f"{culture.construct_name}"
        }
    )
    return [
        f"L4: {DRYAD_METADATA_FILENAME} describes the same {len(rows)} cultures as "
        f"{DRYAD_TOP3_FILENAME}, and every one of its "
        f"{len(CAMPAIGN_ENVIRONMENT_COLUMNS)} environment columns carries one value in "
        "all of them; the five numeric ones equal what production_environment() serves "
        "(24 C, 1.5 mL, 1000 rpm, 48 h, 2 g/L L-arabinose) and the vessel string equals "
        "the Methods'",
        f"L4: Genes_targeted_for_CRISPRi is the Line cell verbatim in {gene_list_rows} "
        f"of {len(rows)} rows and blank in the other "
        f"{len(rows) - gene_list_rows}, all of them controls; it is therefore not a "
        "gene list and nothing is read from it",
        f"L1: the Line cell differs from the Line_name construct in DBTL{prefixed} "
        f"only, by the PRT1093_ prefix ({CAMPAIGN_LINE_COLUMN_RULE.note})",
    ]


def assert_campaign_accession_crosswalk(
    proteome: CampaignProteome,
    resolution: Any,
    panel_rows: Sequence[ProteomeRow],
    panel_locus_of: Mapping[str, str],
) -> list[str]:
    """The GOA accession crosswalk against the paper's own accession -> symbol route.

    The panel sheet carries both an accession (``Protein.Group``) and a gene key
    (``Protein``) for 1,426 accessions, so for every accession both files know, there
    are two independent routes to a locus. Measured 2026-10-09 on the pinned bytes:
    :data:`CAMPAIGN_CROSSWALK_SHARED` (1,224) of the campaign's accessions take both
    routes and 1,222 agree. The two that do not are
    :data:`CAMPAIGN_CROSSWALK_DISAGREEMENTS`, pinned by accession and by both loci, and
    the GOA route is the stored one, because each disagreement is a SYMBOL collision on
    the paper's route: the GOA file states the accession's own locus tags in its synonym
    column, while the panel route goes through a title-cased gene symbol that this
    assembly can carry at another locus.
    """
    panel_route: dict[str, str] = {}
    for row in panel_rows:
        locus = panel_locus_of.get(row.protein)
        if locus is not None:
            panel_route.setdefault(row.accession, locus)
    shared = sorted(set(resolution.resolved) & set(panel_route))
    disagreements = {
        accession: (resolution.resolved[accession], panel_route[accession])
        for accession in shared
        if resolution.resolved[accession] != panel_route[accession]
    }
    if len(shared) != CAMPAIGN_CROSSWALK_SHARED:
        raise RuntimeError(
            f"the Dryad matrix and {SHEET_PROTEOME} both key {len(shared)} accessions; "
            f"{CAMPAIGN_CROSSWALK_SHARED} were measured 2026-10-09, so one of the two "
            "files is not the one this loader was written against"
        )
    if disagreements != CAMPAIGN_CROSSWALK_DISAGREEMENTS:
        raise RuntimeError(
            f"the GOA crosswalk and {SHEET_PROTEOME}'s symbol route disagree on "
            f"{sorted(disagreements)}, not on the measured "
            f"{sorted(CAMPAIGN_CROSSWALK_DISAGREEMENTS)}; each disagreement is a named "
            "symbol collision, so a different set is a new fact about one of the two "
            "files and not an accepted error rate"
        )
    return [
        f"L4: of the {len(shared)} accessions the Dryad matrix and {SHEET_PROTEOME} "
        f"both key, {len(shared) - len(disagreements)} reach the same locus by the GOA "
        f"crosswalk and by the sheet's own accession -> symbol route; the "
        f"{len(disagreements)} that do not are "
        + "; ".join(
            f"{accession} (GOA {goa}, sheet {sheet})"
            for accession, (goa, sheet) in sorted(disagreements.items())
        ),
        "L1: the GOA route is the stored one. Each disagreement above is a symbol "
        "collision on the sheet's route, not a GOA defect: the GOA synonym column "
        "states the accession's own locus tags, so Q88G93 (CSRA_PSEPK) is PP_3832 and "
        "Q88L01 (FADA_PSEPK) is PP_2137, while the symbols csrA and fadA reach another "
        "locus of this assembly. Q877U6 carries BOTH PP_1157 and PP_3365 in the GOA "
        "file, so it names two genes and is dropped rather than keyed to either",
    ]


@register_dataset
class CampaignProteomeCarruthers2025Dataset(ExperimentDataset):
    """Carruthers 2025 campaign proteome: one record per CRISPRi strain-cycle.

    The paired feature set of :class:`IsoprenolTiterCarruthers2025Dataset`. Its 465
    records carry the SAME ``(construct, DBTL cycle)`` identity as the 465 titer
    records, so a proteome-conditioned baseline on the target dataset joins the two on
    the genotype. It reads the Dryad deposit of issue #739, not the Source Data: the
    Source Data's only abundance matrix is the 19-sample off-target panel that
    :class:`ProteomeCarruthers2025Dataset` serves, which is a different experiment on a
    different background and a different scale.
    """

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"
    #: 1,842 of 2,187 released accessions (0.8422) reach exactly one KT2440 locus
    #: through the GOA proteome crosswalk, measured 2026-10-09. The 345 that do not are
    #: the heterologous, marker and contaminant proteins the DIA-NN search database was
    #: built to include plus the host proteins UniProt-GOA carries no locus tag for;
    #: every one is listed in ``preprocess/dropped_accessions.csv``.
    MIN_RESOLVED_FRACTION: ClassVar[float] = CAMPAIGN_MIN_RESOLVED_FRACTION

    def __init__(
        self,
        root: str = "data/torchcell/campaign_proteome_carruthers2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome resolves accessions and guide-target tags."""
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
        """The Dryad matrix, its metadata, the README, and the Source Data workbook.

        The Source Data is here because two of this build's cross-source assertions
        read it: the titer column of the Dryad matrix is checked against ``Figure 4b``
        and the accession crosswalk against ``Supplementary Figure 13abc``.
        """
        return [
            DRYAD_TOP3_FILENAME,
            DRYAD_METADATA_FILENAME,
            DRYAD_README_FILENAME,
            SOURCE_DATA_FILENAME,
        ]

    def download(self) -> None:
        """Link the Dryad deposit and the Source Data into ``raw/`` after pin checks."""
        _link_mirror_files(
            self.raw_dir,
            (
                (DRYAD_TOP3_REL, DRYAD_TOP3_FILENAME, DRYAD_TOP3_SHA256),
                (DRYAD_METADATA_REL, DRYAD_METADATA_FILENAME, DRYAD_METADATA_SHA256),
                (DRYAD_README_REL, DRYAD_README_FILENAME, DRYAD_README_SHA256),
                (SOURCE_DATA_REL, SOURCE_DATA_FILENAME, SOURCE_DATA_SHA256),
            ),
        )
        log.info(
            "Carruthers 2025 campaign proteome linked into %s from the manual Dryad "
            "deposit",
            self.raw_dir,
        )

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    @post_process
    def process(self) -> None:
        """Build one record per CRISPRi strain-cycle, referenced to its cycle's control."""
        verify_raw_files(
            self.raw_dir,
            {
                DRYAD_TOP3_FILENAME: DRYAD_TOP3_SHA256,
                DRYAD_METADATA_FILENAME: DRYAD_METADATA_SHA256,
                DRYAD_README_FILENAME: DRYAD_README_SHA256,
                SOURCE_DATA_FILENAME: SOURCE_DATA_SHA256,
            },
        )
        source_path = osp.join(self.raw_dir, SOURCE_DATA_FILENAME)
        proteome = read_campaign_proteome(osp.join(self.raw_dir, DRYAD_TOP3_FILENAME))
        metadata = read_campaign_metadata(
            osp.join(self.raw_dir, DRYAD_METADATA_FILENAME)
        )
        genome = self._genome()

        proofs: list[str] = [
            assert_campaign_rows_are_percentages(proteome),
            assert_campaign_titers_match_source_data(
                proteome, read_titer_rows(source_path)
            ),
        ]
        proofs.extend(assert_campaign_metadata(metadata, proteome))

        crosswalk = uniprot_locus_crosswalk(genome)
        resolution = resolve_uniprot_accessions(
            crosswalk, proteome.accessions, label=self.name
        )
        resolution.require_resolved(self.MIN_RESOLVED_FRACTION)
        if resolution.collisions:
            raise RuntimeError(
                f"{self.name}: {resolution.collisions} name one locus from several "
                "accessions, so a stored abundance would be two proteins'"
            )
        panel_rows = read_proteome_rows(source_path)
        panel_keys = sorted({row.protein for row in panel_rows})
        panel_stored, panel_report = reconcile_locus_tags(
            genome, pd.Series(panel_keys), label=f"{self.name}-panel-oracle"
        )
        panel_outside = set(panel_report.outside_namespace)
        panel_locus_of = {
            key: stored
            for key, stored in zip(panel_keys, panel_stored, strict=True)
            if key not in panel_outside
        }
        proofs.extend(
            assert_campaign_accession_crosswalk(
                proteome, resolution, panel_rows, panel_locus_of
            )
        )
        locus_of = dict(resolution.resolved)

        by_strain: dict[tuple[str, int], list[int]] = defaultdict(list)
        controls: dict[int, list[int]] = defaultdict(list)
        for row_index, culture in enumerate(proteome.cultures):
            if culture.is_control:
                controls[culture.cycle].append(row_index)
            else:
                by_strain[(culture.construct_name, culture.cycle)].append(row_index)
        if len(by_strain) != EXPECTED_CAMPAIGN_PROTEOME_RECORDS:
            raise RuntimeError(
                f"the deposit groups into {len(by_strain)} strain-cycles; the titer "
                f"family stores {EXPECTED_CAMPAIGN_PROTEOME_RECORDS}"
            )
        if len(controls) != EXPECTED_CAMPAIGN_REFERENCES:
            raise RuntimeError(
                f"the deposit carries controls for {sorted(controls)}; the campaign ran "
                f"{EXPECTED_CAMPAIGN_REFERENCES} cycles"
            )

        tags = sorted(
            {tag for name, _ in by_strain for tag in parse_construct(name)[0]}
        )
        stored, report = reconcile_locus_tags(genome, pd.Series(tags), label=self.name)
        if report.outside_namespace:
            raise RuntimeError(
                f"{self.name}: guide targets outside {KT2440_NAMESPACE}: "
                f"{report.outside_namespace}"
            )
        stored_by_tag = dict(zip(tags, stored, strict=True))
        common = _standard_names(genome, stored_by_tag.values())

        reference_genome = chassis_reference(genome)
        environment = production_environment()
        pathway = pathway_perturbations()
        pub = publication()
        # The per-cycle control profile, from which each record's reference is PROJECTED
        # onto that record's own key set. The projection is not a choice: a record's key
        # set is the proteins its own cultures detected, and the shared protein gate
        # requires experiment and reference to be key-matched so that a per-protein
        # ratio against the control is defined everywhere. Measured 2026-10-09, every
        # one of the 465 records is a SUBSET of its cycle's control key set (0 records
        # with a key the control lacks, 0 of 658,405 record-keys lost), so the
        # projection discards nothing and invents nothing.
        controls_profile: dict[int, CampaignAggregate] = {}
        reference_rows: list[dict[str, Any]] = []
        for cycle, row_indices in sorted(controls.items()):
            aggregate = aggregate_campaign_profile(
                proteome, row_indices, locus_of, label=f"DBTL{cycle} control"
            )
            controls_profile[cycle] = aggregate
            reference_rows.append(
                {
                    "cycle": cycle,
                    "control_cultures": aggregate.cultures,
                    "n_proteins": len(aggregate.abundance),
                }
            )

        def cycle_reference(
            cycle: int, keys: Collection[str]
        ) -> BacterialProteinAbundanceExperimentReference:
            """The cycle's control profile over exactly ``keys``."""
            control = controls_profile[cycle]
            absent = sorted(set(keys) - set(control.abundance))
            if absent:
                raise RuntimeError(
                    f"DBTL{cycle}'s control measures none of {absent[:10]}, which a "
                    "record of that cycle does; the control cannot reference it"
                )
            return BacterialProteinAbundanceExperimentReference(
                dataset_name=self.name,
                genome_reference=reference_genome,
                environment_reference=environment.model_copy(),
                phenotype_reference=ProteinAbundancePhenotype(
                    protein_abundance={k: control.abundance[k] for k in keys},
                    protein_abundance_se={k: control.se[k] for k in keys},
                    n_replicates={k: control.n_replicates[k] for k in keys},
                    measurement_type=str(CAMPAIGN_TOP3.value),
                ),
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        strain_rows: list[dict[str, Any]] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for (construct, cycle), row_indices in tqdm(
                sorted(by_strain.items()), desc="carruthers2025-campaign-proteome"
            ):
                aggregate = aggregate_campaign_profile(
                    proteome, row_indices, locus_of, label=f"{construct} DBTL{cycle}"
                )
                tag_list, extras = parse_construct(construct)
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(
                        perturbations=[
                            *pathway,
                            *(
                                crispri_perturbation(
                                    stored_by_tag[tag], common[stored_by_tag[tag]]
                                )
                                for tag in tag_list
                            ),
                        ]
                    ),
                    environment=environment,
                    phenotype=ProteinAbundancePhenotype(
                        protein_abundance=aggregate.abundance,
                        protein_abundance_se=aggregate.se,
                        n_replicates=aggregate.n_replicates,
                        measurement_type=str(CAMPAIGN_TOP3.value),
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(
                        experiment,
                        cycle_reference(cycle, sorted(aggregate.abundance)),
                        pub,
                        itxn,
                    ),
                )
                strain_rows.append(
                    {
                        "construct": construct,
                        "cycle": cycle,
                        "n_cultures": aggregate.cultures,
                        "n_targets": len(tag_list),
                        "non_targeting_tokens": ";".join(extras),
                        "n_proteins": len(aggregate.abundance),
                        "max_n_replicates": max(aggregate.n_replicates.values()),
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(strain_rows).to_csv(
            osp.join(self.preprocess_dir, "strain_cycles.csv"), index=False
        )
        pd.DataFrame(reference_rows).to_csv(
            osp.join(self.preprocess_dir, "cycle_references.csv"), index=False
        )
        pd.DataFrame(
            [
                {"accession": accession, "reason": reason, "detail": detail}
                for accession, reason, detail in (
                    *(
                        (accession, "accession_names_several_loci", ";".join(loci))
                        for accession, loci in sorted(resolution.multi_locus.items())
                    ),
                    *(
                        (accession, "no_locus_tag_in_the_goa_proteome_file", "")
                        for accession in resolution.unmapped
                    ),
                )
            ]
        ).to_csv(osp.join(self.preprocess_dir, "dropped_accessions.csv"), index=False)
        Path(osp.join(self.preprocess_dir, "campaign_proofs.json")).write_text(
            json.dumps(proofs, indent=2)
        )
        key_sizes = Counter(row["n_proteins"] for row in strain_rows)
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(proteome.cultures),
                control_rows=sum(len(group) for group in controls.values()),
                candidate_records=len(by_strain),
                kept_records=idx,
                dropped_records=0,
                rules=[
                    DropRule(
                        rule="accession_has_no_locus_tag_in_the_goa_proteome_file",
                        scope="protein_accession",
                        description=(
                            "the deposited matrix is keyed by UniProt accession "
                            f"({CAMPAIGN_ACCESSION_KEYS.quote}) and the only mirrored "
                            "statement of accession -> locus tag is the assembly set's "
                            f"GOA proteome file {crosswalk.member} (sha256 "
                            f"{crosswalk.sha256}), which carries no locus tag for these "
                            "accessions. They are the heterologous pathway proteins, "
                            "the dCas9 effector, resistance markers and proteomic "
                            "contaminants the DIA-NN search database was built to "
                            f"include ({_Q_DIANN_DB}), plus host proteins UniProt-GOA "
                            "files no locus tag for; none has a gene node to key an "
                            "abundance to"
                        ),
                        n_records=0,
                        items=list(resolution.unmapped),
                    ),
                    DropRule(
                        rule="accession_names_several_loci",
                        scope="protein_accession",
                        description=(
                            "the GOA file gives the accession more than one locus tag, "
                            "so one abundance column stands for two genes and cannot be "
                            "attributed to either"
                        ),
                        n_records=0,
                        items=sorted(resolution.multi_locus),
                    ),
                ],
                reconciliation=report,
                notes=[
                    f"every one of the {EXPECTED_CAMPAIGN_PROTEOME_RECORDS} titer "
                    "strain-cycles gains a paired feature row: the deposit's "
                    f"{len(proteome.cultures)} cultures group into exactly the "
                    f"{len(by_strain)} non-control (construct, cycle) identities the "
                    "titer family stores, plus controls for all "
                    f"{EXPECTED_CAMPAIGN_REFERENCES} cycles",
                    "the nine Source Data cultures with no proteome row are R4-R6 of "
                    "PP_0814_PP_4192, PP_0814_PP_4862 and PP_2137_PP_4189 in DBTL1; "
                    "all three strain-cycles still have a record, from their R1-R3",
                    f"{len(locus_of)} of {len(proteome.accessions)} accessions are "
                    f"stored ({len(locus_of) / len(proteome.accessions):.4f}); "
                    f"{len(resolution.unmapped)} carry no locus tag in "
                    f"{crosswalk.member} and {len(resolution.multi_locus)} carry "
                    "several",
                    "a record's key set is the proteins present in at least one of its "
                    "cultures, so the sets differ by strain: "
                    f"{min(key_sizes)} to {max(key_sizes)} proteins, median "
                    f"{int(statistics.median([row['n_proteins'] for row in strain_rows]))}. "
                    f"{CAMPAIGN_BLANK_IS_ABSENT.note}",
                    "the controls are the per-cycle phenotype_reference and not "
                    f"records, as in the titer family: {CONTROL_N.note}. Each record's "
                    "reference is its cycle's control profile PROJECTED onto that "
                    "record's own key set, which the shared protein gate requires and "
                    "which discards nothing: every record's keys are a subset of its "
                    "cycle control's (0 of 658,405 record-keys lost, measured "
                    "2026-10-09)",
                    f"{proteome.blank_cells} of "
                    f"{len(proteome.cultures) * len(proteome.accessions)} released cells "
                    f"are blank and {proteome.zero_cells} are a released 0; the blank "
                    "is absent by the README's own rule and the 0 is kept verbatim",
                    *proofs,
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Carruthers2025 campaign proteome: %d records over %d accessions stored as "
            "%d loci, referenced to the %d per-cycle controls",
            idx,
            len(proteome.accessions),
            len(locus_of),
            len(controls_profile),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Family 3: the released differential proteomics (per-protein log2 fold change)
# --------------------------------------------------------------------------- #
@register_dataset
class ProteomeFoldChangeCarruthers2025Dataset(ExperimentDataset):
    """Carruthers 2025 per-protein log2 fold changes of the CRISPRi and KO contrasts.

    The RELATIVE sibling of :class:`ProteomeCarruthers2025Dataset`: that class stores an
    absolute Top3 level per strain, this one stores the ratio of a strain's proteome to
    a control's, which is a different measurement and so a different experiment class.
    One record per released contrast COLUMN:

    - ``Figure 5b`` (14 records): one single-guide CRISPRi strain per column, against
      the non-target control strain. Both the scale and the denominator are sourced.
    - ``Figure 6b`` (2 records): the two KO-plus-target-sgRNA strains the Fig. 6 caption
      names, against the same KO carrying a non-targeting sgRNA. The sheet releases ten
      columns; the other eight have no sourced denominator and are ledgered unbuilt,
      with the measurement that refused each one, in
      :data:`KO_FOLD_CHANGE_UNSOURCED_REASONS`.

    ``Supplementary Figure 10`` is refused entirely and its measurement recorded: see
    :data:`BEST_ARRAY_HEATMAP_REFUSAL`.

    THE REFERENCE IS NOT A MEASUREMENT. A fold change's denominator is the neutral value
    of its scale by definition, so each record's ``phenotype_reference`` is
    ``ProteinFoldChangePhenotype.neutral_reference()`` over the record's own keys, and
    experiment over reference reproduces the released number exactly. Nothing is
    imputed, and no absolute control level is invented.
    """

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"
    #: Measured on the pinned workbook over the UNION of both sheets' row keys: 1,395 of
    #: 1,414 (0.9866) resolve to a locus of this assembly. Per sheet it is 1,284 of
    #: 1,290 (Figure 5b) and 358 of 372 (Figure 6b). The 19 that do not are heterologous
    #: and contaminant keys (``EF_1364``, ``MM_1762``, ``SPy_1046``, ``YNR043W``,
    #: ``b4055 JW4015``, ``Q9FD70``, twelve ``A0A140F*`` UniProt accessions) plus the
    #: one two-tag key ``PP_1157 PP_3365``; the threshold sits just below the measured
    #: fraction, so a real drop means the keying changed.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 0.98

    def __init__(
        self,
        root: str = "data/torchcell/proteome_fold_change_carruthers2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome resolves protein keys and chassis symbols."""
        self.pputida_genome = pputida_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialProteinFoldChangeExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialProteinFoldChangeExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Source Data workbook, which holds both released contrast sheets."""
        return [SOURCE_DATA_FILENAME]

    def download(self) -> None:
        """Link the Source Data workbook into ``raw/`` after verifying its pin."""
        _link_mirror_files(
            self.raw_dir, ((SOURCE_DATA_REL, SOURCE_DATA_FILENAME, SOURCE_DATA_SHA256),)
        )
        log.info("Carruthers 2025 fold-change artifact linked into %s", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    @staticmethod
    def _phenotype(
        cells: Sequence[FoldChangeRow],
        stored_by_key: Mapping[str, str],
        dropped: Collection[str],
        *,
        reference_basis: str,
    ) -> ProteinFoldChangePhenotype:
        """One contrast column's fold changes, p-values and replicate counts.

        A key the reconciliation could not place as a locus of the pinned assembly is
        dropped from the map and ledgered; there is no gene node to key its ratio to.
        """
        fold_change: dict[str, float] = {}
        p_values: dict[str, float] = {}
        n_replicates: dict[str, int] = {}
        replicates = int(N_REPLICATES.value)
        for cell in cells:
            if cell.protein in dropped:
                continue
            tag = stored_by_key[cell.protein]
            if tag in fold_change:
                raise RuntimeError(
                    f"{cell.sheet}/{cell.contrast}: two released keys reconcile to "
                    f"{tag}, so one locus would carry two fold changes"
                )
            fold_change[tag] = cell.log2_fold_change
            p_values[tag] = p_value_from_neg_log10(cell.neg_log10_p_value)
            n_replicates[tag] = replicates
        if not fold_change:
            raise RuntimeError("a contrast with no resolved protein key")
        return ProteinFoldChangePhenotype(
            protein_fold_change=fold_change,
            fold_change_scale=FoldChangeScale(FOLD_CHANGE_SCALE.value),
            reference_basis=reference_basis,
            protein_fold_change_p_value=p_values,
            n_replicates=n_replicates,
            measurement_type=str(FOLD_CHANGE_TEST.value),
        )

    @staticmethod
    def _reference_phenotype(
        phenotype: ProteinFoldChangePhenotype,
    ) -> ProteinFoldChangePhenotype:
        """The denominator: the neutral value of this record's scale, per stored key."""
        neutral = phenotype.neutral_reference()
        return ProteinFoldChangePhenotype(
            protein_fold_change=neutral,
            fold_change_scale=phenotype.fold_change_scale,
            reference_basis=phenotype.reference_basis,
            n_replicates=dict(phenotype.n_replicates),
            measurement_type=phenotype.measurement_type,
        )

    @post_process
    def process(self) -> None:
        """Build one fold-change record per sourced contrast column; write LMDB."""
        verify_raw_files(self.raw_dir, {SOURCE_DATA_FILENAME: SOURCE_DATA_SHA256})
        path = osp.join(self.raw_dir, SOURCE_DATA_FILENAME)
        genome = self._genome()

        single_rows = read_fold_change_rows(path, SHEET_FOLD_CHANGE)
        ko_rows = read_fold_change_rows(path, SHEET_KO_FOLD_CHANGE)
        proofs = [
            assert_fold_change_pointer(path),
            *assert_fold_change_contrasts_are_released_strains(path),
            *assert_best_array_heatmap_refusal(path),
        ]

        keys = sorted({row.protein for row in (*single_rows, *ko_rows)})
        stored, report = reconcile_locus_tags(genome, pd.Series(keys), label=self.name)
        report.require_resolved(self.MIN_RESOLVED_FRACTION)
        stored_by_key = dict(zip(keys, stored, strict=True))
        dropped_keys = sorted(set(report.outside_namespace))
        kept_keys = [key for key in keys if key not in set(dropped_keys)]
        if not kept_keys:
            raise RuntimeError(f"{self.name}: every protein key was dropped")

        by_contrast: dict[tuple[str, str], list[FoldChangeRow]] = defaultdict(list)
        for row in (*single_rows, *ko_rows):
            by_contrast[(row.sheet, row.contrast)].append(row)

        single_contrasts = fold_change_contrasts(path, SHEET_FOLD_CHANGE)
        if len(single_contrasts) != EXPECTED_FOLD_CHANGE_RECORDS_FIG5B:
            raise RuntimeError(
                f"{SHEET_FOLD_CHANGE} releases {len(single_contrasts)} contrast "
                f"columns, not {EXPECTED_FOLD_CHANGE_RECORDS_FIG5B}"
            )
        ko_contrasts = fold_change_contrasts(path, SHEET_KO_FOLD_CHANGE)
        unsourced = [
            contrast
            for contrast in ko_contrasts
            if contrast not in KO_FOLD_CHANGE_CONTRASTS
        ]
        if set(unsourced) != set(KO_FOLD_CHANGE_UNSOURCED_REASONS):
            raise RuntimeError(
                f"{SHEET_KO_FOLD_CHANGE} releases unsourced columns {sorted(unsourced)}; "
                f"the ledger covers {sorted(KO_FOLD_CHANGE_UNSOURCED_REASONS)}. A new "
                "column needs its own sourced contrast or its own ledgered refusal"
            )
        missing = [
            contrast
            for contrast in KO_FOLD_CHANGE_CONTRASTS
            if contrast not in ko_contrasts
        ]
        if missing:
            raise RuntimeError(
                f"{SHEET_KO_FOLD_CHANGE} no longer releases {missing}, which the Fig. 6 "
                f"caption names ('{_Q_FIG6B}')"
            )

        reference_genome = chassis_reference(genome)
        environment = production_environment()
        pathway = pathway_perturbations()
        common = _standard_names(genome, [*single_contrasts, *KO_FOLD_CHANGE_CONTRASTS])
        pub = publication()

        plan: list[tuple[str, str, str, Genotype]] = []
        for contrast in single_contrasts:
            name = common.get(contrast, contrast)
            plan.append(
                (
                    SHEET_FOLD_CHANGE,
                    contrast,
                    str(FOLD_CHANGE_REFERENCE_BASIS.value),
                    Genotype(
                        perturbations=[*pathway, crispri_perturbation(contrast, name)]
                    ),
                )
            )
        for contrast in KO_FOLD_CHANGE_CONTRASTS:
            name = common.get(contrast, contrast)
            plan.append(
                (
                    SHEET_KO_FOLD_CHANGE,
                    contrast,
                    str(KO_FOLD_CHANGE_REFERENCE_BASIS.value),
                    Genotype(
                        perturbations=[
                            *pathway,
                            deletion_perturbation(contrast, name),
                            crispri_perturbation(contrast, name),
                        ]
                    ),
                )
            )
        if len(plan) != EXPECTED_FOLD_CHANGE_RECORDS:
            raise RuntimeError(
                f"{len(plan)} contrasts planned, not {EXPECTED_FOLD_CHANGE_RECORDS}"
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        contrast_rows: list[dict[str, Any]] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for sheet, contrast, basis, genotype in tqdm(
                plan, desc="carruthers2025-fold-change"
            ):
                phenotype = self._phenotype(
                    by_contrast[(sheet, contrast)],
                    stored_by_key,
                    dropped_keys,
                    reference_basis=basis,
                )
                experiment = BacterialProteinFoldChangeExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=environment,
                    phenotype=phenotype,
                )
                reference = BacterialProteinFoldChangeExperimentReference(
                    dataset_name=self.name,
                    genome_reference=reference_genome,
                    environment_reference=environment.model_copy(),
                    phenotype_reference=self._reference_phenotype(phenotype),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                contrast_rows.append(
                    {
                        "sheet": sheet,
                        "contrast": contrast,
                        "reference_basis": basis,
                        "n_proteins": len(phenotype.protein_fold_change),
                        "n_released": len(by_contrast[(sheet, contrast)]),
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(contrast_rows).to_csv(
            osp.join(self.preprocess_dir, "contrasts.csv"), index=False
        )
        Path(osp.join(self.preprocess_dir, "fold_change_proofs.json")).write_text(
            json.dumps(proofs, indent=2)
        )
        pd.DataFrame(
            [
                {"protein_key": key, "reason": "not_a_locus_of_the_pinned_assembly"}
                for key in dropped_keys
            ]
        ).to_csv(osp.join(self.preprocess_dir, "dropped_protein_keys.csv"), index=False)
        heatmap_keys, heatmap_strains, heatmap_values = read_best_array_heatmap(path)
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(single_rows) + len(ko_rows),
                control_rows=0,
                candidate_records=len(single_contrasts)
                + len(ko_contrasts)
                + len(heatmap_strains),
                kept_records=idx,
                dropped_records=len(unsourced) + len(heatmap_strains),
                rules=[
                    DropRule(
                        rule="protein_key_is_not_a_locus_of_the_pinned_assembly",
                        scope="protein_key",
                        description=(
                            "the key is a heterologous, contaminant or multi-tag "
                            f"identifier, no locus of {KT2440_ASSEMBLY_SET}, so it has "
                            "no gene node to key a fold change to"
                        ),
                        n_records=0,
                        items=dropped_keys,
                    ),
                    DropRule(
                        rule="ko_fold_change_contrast_has_no_sourced_denominator",
                        scope="strain",
                        description=(
                            f"{SHEET_KO_FOLD_CHANGE} releases ten contrast columns and "
                            f"the Fig. 6 caption names two ('{_Q_FIG6B}'). The other "
                            "eight are refused rather than given a guessed "
                            "denominator; the per-column measurement is in "
                            "KO_FOLD_CHANGE_UNSOURCED_REASONS and is listed here as "
                            "'<column>: <measurement>'"
                        ),
                        n_records=len(unsourced),
                        items=[
                            f"{contrast}: {KO_FOLD_CHANGE_UNSOURCED_REASONS[contrast]}"
                            for contrast in unsourced
                        ],
                    ),
                    DropRule(
                        rule="best_array_heatmap_states_no_scale",
                        scope="strain",
                        description=str(BEST_ARRAY_HEATMAP_REFUSAL.note),
                        n_records=len(heatmap_strains),
                        items=list(heatmap_strains),
                    ),
                ],
                reconciliation=report,
                notes=[
                    f"{len(single_contrasts)} {SHEET_FOLD_CHANGE} records + "
                    f"{len(KO_FOLD_CHANGE_CONTRASTS)} {SHEET_KO_FOLD_CHANGE} records; "
                    f"{len(dropped_keys)} of {len(keys)} protein KEYS are dropped, "
                    f"leaving {len(kept_keys)} across the two sheets",
                    f"the maps are RAGGED by release: {SHEET_FOLD_CHANGE} carries "
                    f"{len(single_rows)} fold changes over {len(single_contrasts)} "
                    f"columns and {SHEET_KO_FOLD_CHANGE} carries {len(ko_rows)} over "
                    f"{len(ko_contrasts)}, because both sheets are pre-filtered at the "
                    f"caption's p<{FOLD_CHANGE_P_VALUE_CEILING}; a protein a contrast "
                    "did not significantly change is simply not a key",
                    f"{SHEET_KO_FOLD_CHANGE} also releases "
                    "181 rows that carry a 'primary_name' with no 'Locus Name' and no "
                    "value; they are skipped, and one that ever carries a value stops "
                    "the build",
                    "the reference phenotype is neutral_reference(), 0.0 on the log2 "
                    "scale for every stored key: a fold change's denominator is its "
                    "scale's neutral value by definition, so nothing is imputed",
                    f"{SHEET_BEST_ARRAY_HEATMAP} is NOT loaded: its "
                    f"{len(heatmap_keys)} keys x {len(heatmap_strains)} strain columns "
                    f"are refused because no mirrored statement names their scale",
                    f"the released p-value column is {NEG_LOG10_P_VALUE_SUFFIX}, "
                    "unsigned -log10(p); the stored number is the probability "
                    "p = 10**-x and every conversion's round trip is asserted",
                    *proofs,
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Carruthers2025 fold change: %d records = %d %s contrasts + %d %s "
            "contrasts (%d of %d released %s columns refused); %d keys dropped from "
            "%d released cells",
            idx,
            len(single_contrasts),
            SHEET_FOLD_CHANGE,
            len(KO_FOLD_CHANGE_CONTRASTS),
            SHEET_KO_FOLD_CHANGE,
            len(unsourced),
            len(ko_contrasts),
            SHEET_KO_FOLD_CHANGE,
            len(dropped_keys),
            len(single_rows) + len(ko_rows),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# L0-L4 verification of a built tree. The shared family gates do L0-L3
# (``verify_product_titer_dataset`` / ``verify_protein_dataset``); the two rules below
# are this release's own, and the two L4 rows join the built store to a DIFFERENT
# released file from the one its loader read.
#
# ``run_product_titer`` and ``run_bacterial_protein_abundance`` in
# ``torchcell.verification.runners`` call :func:`verify_build` and add the host-aware
# gene-universe containment, so these levels run from ``run_all``.
# --------------------------------------------------------------------------- #
#: Strains the paper's own "472 unique strains" count reaches by dividing its 1,416
#: non-control cultures by three; seven of the 465 ``(construct, cycle)`` strains carry
#: SIX replicates rather than three, and 465 + 7 == 472.
PAPER_STRAIN_COUNT = 472
#: Protein keys a ``SHEET_PROTEOME`` record carries: 1,501 released minus the 77
#: dropped. A ``SHEET_OVEREXPRESSION_PROTEOME`` record carries only the operon its
#: strain overexpresses, so the two panels have different, pinned key-set sizes.
PROTEOME_KEYS_PER_RECORD = 1424
PROTEOME_KEY_SET_SIZES: tuple[int, ...] = (
    *(PROTEOME_KEYS_PER_RECORD,) * EXPECTED_PP0815_PROTEOME_RECORDS,
    *(len(operon) for operon in OVEREXPRESSION_OPERONS.values()),
)
#: The Results text prints the KO-plus-array titers to integer mg/L, inconsistently
#: rounded (651.860 printed 651, 579.534 printed 580), so the tolerance is that unit.
KO_ARRAY_TITER_TOL = 1.0
#: Records on the Δ PP_0815 background alone: ``Figure 6a``'s Target arm plus
#: ``Supplementary Figure 13d``'s 19. All share the one deduplicated reference.
OFFTARGET_SHARED_REFERENCE_RECORDS = 1 + 19
#: Tolerance of the derived standard error against ``SD / sqrt(n)``. The loader derives
#: it in float from the two stored numbers, so the identity holds to float noise.
TITER_SE_TOL = 1e-9
#: Supplementary Data 1 prints its per-target means to two decimals in mg/L.
SI_TARGET_MEAN_TOL = 5e-3
#: Targets of Supplementary Data 1's 120-row table that join a single-guide record. All
#: of them do: the RECORD-level join reaches ``PP_1607`` and ``PP_4194``, released only
#: under the filler names ``PP_1607_NT1`` / ``PP_4194_NT2`` whose filler guide is not a
#: perturbation, which a construct-name join cannot.
SI_TARGET_OVERLAP = 120
#: A campaign record's stored percentages sum to at most this. The cells ARE percentages
#: of the proteome, so their sum cannot exceed 100 by more than the release's own
#: rounding: measured 2026-10-09, the 1,497 released per-culture rows sum to
#: 99.9096-100.0128 over all 2,187 accessions.
CAMPAIGN_PERCENT_CEILING = 100.05
#: Tolerance of that per-CULTURE row sum against 100, asserted at build time on the
#: released bytes. It is the strong form of "these numbers are percentages"; the stored
#: per-record sum is necessarily lower, because the 345 accessions with no locus tag are
#: dropped and they carry proteome mass.
CAMPAIGN_ROW_SUM_TOL = 0.1
#: What the dropped accessions carry, measured 2026-10-09: 17.2% of the proteome on
#: average over the 1,497 cultures, so a stored record sums to about 83%. The spread is
#: WIDE and the reason is a sample-prep artifact rather than biology: the worst culture
#: (PP_1506_PP_4120-R3, DBTL6) is 63.1% porcine trypsin (P00761), leaving 23.6%, and the
#: stored per-record sums run 28.705-92.951. So this family carries no floor: a floor
#: would be a claim about contamination, not about the unit.
CAMPAIGN_DROPPED_MASS_SHARE = 0.172


def _titer_provenance() -> Provenance:
    """Where the titer family's numbers came from."""
    return Provenance(
        source_uri=SOURCE_DATA_REL,
        citation_key=CITATION_KEY,
        sha256=SOURCE_DATA_SHA256,
        method=(
            "Source Data sheet 'Figure 4b', column 'isoprenoli titer (mg/L)' (the typo "
            "is the source's), grouped by (construct, DBTL cycle); mean over the "
            "strain's biological replicates with their sample SD, stored verbatim as "
            "ug/mL since 1 mg/L == 1 ug/mL"
        ),
        page="Source Data 'Figure 4b'; Supplementary Data 1 as the cross-source oracle",
    )


def _proteome_provenance() -> Provenance:
    """Where the proteome family's numbers came from."""
    return Provenance(
        source_uri=SOURCE_DATA_REL,
        citation_key=CITATION_KEY,
        sha256=SOURCE_DATA_SHA256,
        method=(
            "Source Data sheet 'Supplementary Figure 13abc', column "
            "'Top_3pep_counts_mean' (DIA-NN Top3 signal), one record per released "
            "sample of the PP_0815 off-target panel; 77 of 1,501 protein keys dropped "
            "by a sourced rule"
        ),
        page="Source Data 'Supplementary Figure 13abc'",
    )


def _perturbation_types(record: Mapping[str, Any]) -> set[str]:
    """The ``perturbation_type`` values one record's genotype carries."""
    return {
        str(perturbation["perturbation_type"])
        for perturbation in record["experiment"]["genotype"]["perturbations"]
    }


def _is_campaign_record(record: Mapping[str, Any]) -> bool:
    """True for a ``Figure 4b`` CRISPRi strain: no deletion and no extra native copy.

    The four Source Data panels add chromosomal KO backgrounds and plasmid-borne native
    copies, and neither exists in the campaign. Every rule written against the
    campaign's own oracles -- the paper's 472, Supplementary Data 1's per-target means
    -- is scoped by this predicate, so a panel record can never be joined to an oracle
    that does not describe it.
    """
    return not ({"bacterial_deletion", "gene_addition"} & _perturbation_types(record))


def _l1_panel_partition(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L1: the store splits into the campaign's 465 and the panels' 37.

    This is the stored half of the partition the build proves on the released bytes:
    the campaign records carry only the pathway and CRISPRi perturbations, and every
    record that carries a deletion or an extra native copy came from one of the four
    panels.
    """
    campaign = sum(1 for record in records if _is_campaign_record(record))
    panels = len(records) - campaign
    passed = (
        campaign == EXPECTED_CRISPRI_TITER_RECORDS
        and panels == EXPECTED_PANEL_TITER_RECORDS
    )
    return LevelResult(
        level=Level.L1,
        name="campaign_and_panel_records_partition",
        passed=passed,
        message=(
            f"{campaign} {SHEET_TITER} campaign strains + {panels} Source Data panel "
            f"strains = {len(records)}"
        ),
        details={
            "n_campaign": campaign,
            "n_panel": panels,
            "expected_campaign": EXPECTED_CRISPRI_TITER_RECORDS,
            "expected_panel": EXPECTED_PANEL_TITER_RECORDS,
        },
    )


def _l4_ko_array_titer_vs_results_text(
    records: Sequence[dict[str, Any]],
) -> LevelResult:
    """L4: the KO-plus-array record against the titer the Results text prints for it.

    The Results state "Combining KOs with specific sgRNAs for PP_0528 and PP_0815
    further improved titer to 4-fold that of the control (651 mg/L) and 12% more
    isoprenol than the two-sgRNA array in a strain without KOs (580 mg/L, p < 0.02)".
    The first number is the ``Figure 6d`` record this revision adds and the second is a
    ``Figure 4b`` array already stored, so one prose sentence joins the new panel and
    the campaign at once. The text prints both to integer mg/L and is not consistent
    about rounding (651.860 printed 651, 579.534 printed 580), so the tolerance is the
    integer unit it prints in.
    """
    expected = dict(KO_ARRAY_RESULT.value)
    #: The exact (deleted, knocked down) split of each strain the sentence names, so a
    #: strain with the same gene UNION but a different split can never satisfy it: the
    #: whole point of the comparison is deletion against knockdown of the same loci.
    genotypes: dict[str, tuple[frozenset[str], frozenset[str]]] = {
        "PP_0368_PP_0812-15_KO with PP_0528_PP_0815": (
            frozenset({"PP_0368", "PP_0812", "PP_0813", "PP_0814", "PP_0815"}),
            frozenset({"PP_0528", "PP_0815"}),
        ),
        "PP_0528_PP_0815": (frozenset(), frozenset({"PP_0528", "PP_0815"})),
    }
    found: dict[str, list[float]] = defaultdict(list)
    for record in records:
        experiment = record["experiment"]
        split = (
            frozenset(
                perturbation["systematic_gene_name"]
                for perturbation in experiment["genotype"]["perturbations"]
                if perturbation["perturbation_type"] == "bacterial_deletion"
            ),
            frozenset(
                perturbation["systematic_gene_name"]
                for perturbation in experiment["genotype"]["perturbations"]
                if perturbation["perturbation_type"] == "bacterial_crispr_interference"
            ),
        )
        for label, genotype in genotypes.items():
            if split == genotype:
                found[label].append(float(experiment["phenotype"]["titer"]))
    wrong = {label: len(found[label]) for label in expected if len(found[label]) != 1}
    if wrong:
        raise AssertionError(
            f"the Results text names {sorted(wrong)} and the store holds {wrong} "
            "records with that exact deletion / knockdown split; each must be unique"
        )
    shared = [
        (label, found[label][0], float(expected[label])) for label in sorted(expected)
    ]
    return l4_cross_source(shared, tol=KO_ARRAY_TITER_TOL).model_copy(
        update={"name": "ko_array_titer_vs_results_text"}
    )


def _l1_strain_count_reconciles(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L1: the stored strain count plus its six-replicate strains is the paper's 472.

    The paper reports "472 unique strains (125 single perturbations and 347
    combinations) in triplicate", which is its 1,416 non-control cultures divided by
    three. Grouping by ``(construct, DBTL cycle)`` gives 465, because seven strains
    were cultured six times. This asserts the DOCUMENTED reconciliation rather than
    accepting either number: 465 + 7 == 472.

    Scoped to the CAMPAIGN records: the four Source Data panels are KO backgrounds and
    overexpression strains that the abstract's 472 does not count, so counting them
    here would silently break a rule about a different experiment.
    """
    campaign = [record for record in records if _is_campaign_record(record)]
    six = sum(
        1 for record in campaign if record["experiment"]["phenotype"]["n_samples"] == 6
    )
    total = len(campaign) + six
    return LevelResult(
        level=Level.L1,
        name="strain_count_reconciles_with_the_papers_472",
        passed=total == PAPER_STRAIN_COUNT,
        message=(
            f"{len(campaign)} campaign strains + {six} with six replicates = {total} "
            f"(the paper's {PAPER_STRAIN_COUNT})"
        ),
        details={
            "n_campaign_records": len(campaign),
            "n_records": len(records),
            "n_six_replicate_strains": six,
            "paper_strain_count": PAPER_STRAIN_COUNT,
        },
    )


def _l1_protein_key_sets(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L1: each record's protein key set is one of the panels', with the pinned count.

    The release carries TWO per-protein per-replicate matrices and they quantify
    different things: ``Supplementary Figure 13abc`` profiles 1,424 host loci for every
    one of its 19 samples, and ``Supplementary Figure 12ac`` quantifies only the operon
    a sample overexpresses, which is 2 loci for one label and 4 for the other. So the
    rule is not "one key set" but "the measured key-set sizes are exactly the panels'",
    which a record that silently lost or gained a protein still fails.
    """
    sizes = Counter(
        len(record["experiment"]["phenotype"]["protein_abundance"])
        for record in records
    )
    expected = Counter(PROTEOME_KEY_SET_SIZES)
    passed = sizes == expected
    return LevelResult(
        level=Level.L1,
        name="protein_key_set_sizes_are_the_panels",
        passed=passed,
        message=(
            f"{len(records)} records over key-set sizes {dict(sorted(sizes.items()))}"
            + ("" if passed else f"; expected {dict(sorted(expected.items()))}")
        ),
        details={
            # String keys, so the report round-trips through JSON unchanged.
            "key_set_sizes": {str(size): n for size, n in sorted(sizes.items())},
            "expected": {str(size): n for size, n in sorted(expected.items())},
        },
    )


def _l3_biological_triplicate(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L3: every (sample, protein) cell is a biological triplicate.

    Supplementary Fig. 13 caption, verbatim: "All strains were cultured in triplicate
    (n = 3) and error bars represent standard deviation."
    """
    counts = {
        int(n)
        for record in records
        for n in record["experiment"]["phenotype"]["n_replicates"].values()
    }
    return l3_convention(
        "every_sample_is_a_biological_triplicate",
        counts == {3},
        detail=(
            f"stored replicate counts {sorted(counts)}; Supplementary Fig. 13: 'All "
            "strains were cultured in triplicate (n = 3)'"
        ),
    )


def _l4_titer_vs_supplementary_data_1(
    records: Sequence[dict[str, Any]], data_root: str | None
) -> LevelResult:
    """L4: the store's single-guide titers against Supplementary Data 1's own means.

    The oracle is a DIFFERENT released file from the one the loader reads, so this
    joins the built records to an independent statement of the same measurement. Its
    means are printed to two decimals, hence the 0.005 mg/L tolerance. A target with
    several single-guide records (the same tag screened in more than one cycle) is
    joined on the record closest to the released mean, which is the strain the DBTL0
    table reports.

    Scoped to the CAMPAIGN records. Supplementary Data 1 reports the DBTL0 screen of
    single-guide CRISPRi strains, and three of the four panels also produce records
    with exactly one CRISPRi perturbation -- a KO background carrying the sgRNA of the
    gene it deleted. Those are a different strain with the same guide, so joining them
    to this oracle would be a false join that the nearest-record rule would then hide.
    """
    released = read_si_target_means(str(raw_mirror_dir(data_root) / TARGETS_REL))
    by_tag: dict[str, list[float]] = {}
    for record in records:
        if not _is_campaign_record(record):
            continue
        experiment = record["experiment"]
        targets = [
            perturbation["systematic_gene_name"]
            for perturbation in experiment["genotype"]["perturbations"]
            if perturbation["perturbation_type"] == "bacterial_crispr_interference"
        ]
        if len(targets) == 1:
            by_tag.setdefault(targets[0], []).append(experiment["phenotype"]["titer"])
    shared = [
        (tag, min(by_tag[tag], key=lambda titer: abs(titer - mean)), mean)
        for tag, mean in sorted(released.items())
        if tag in by_tag
    ]
    if len(shared) != SI_TARGET_OVERLAP:
        raise AssertionError(
            f"{len(shared)} of Supplementary Data 1's {len(released)} targets join a "
            f"single-guide record; all {SI_TARGET_OVERLAP} do on the pinned bytes"
        )
    return l4_cross_source(shared, tol=SI_TARGET_MEAN_TOL).model_copy(
        update={"name": "single_guide_titer_vs_supplementary_data_1"}
    )


def _l4_proteome_vs_released_sheet(
    records: Sequence[dict[str, Any]], data_root: str | None
) -> LevelResult:
    """L4: the store's PP_0815-target profile against the released sheet, re-read.

    Every stored abundance must be reproducible from the deposited bytes by the same
    aggregation, so a store that drifted from its source is caught per protein.
    """
    rows = read_proteome_rows(str(raw_mirror_dir(data_root) / SOURCE_DATA_REL))
    genome = bacterial_genome("pputida", "KT2440", data_root)
    keys = sorted({row.protein for row in rows})
    stored_keys, _ = reconcile_locus_tags(genome, pd.Series(keys), label="l4")
    key_map = dict(zip(keys, stored_keys, strict=True))
    cells: dict[str, list[float]] = {}
    for row in rows:
        if row.sample != PROTEOME_TARGET_SAMPLE:
            continue
        cells.setdefault(key_map[row.protein], []).append(row.top3_signal)
    target = next(
        record["experiment"]
        for record in records
        if any(
            perturbation["systematic_gene_name"] == PROTEOME_BACKGROUND_DELETION
            and perturbation["perturbation_type"] == "bacterial_crispr_interference"
            for perturbation in record["experiment"]["genotype"]["perturbations"]
        )
    )
    abundance = target["phenotype"]["protein_abundance"]
    shared = [
        (key, value, sum(cells[key]) / len(cells[key]))
        for key, value in sorted(abundance.items())
        if key in cells
    ]
    if len(shared) != len(abundance):
        raise AssertionError(
            f"{len(abundance) - len(shared)} stored proteins are not in the released "
            "sheet under their reconciled key"
        )
    return l4_cross_source(shared, tol=1e-6).model_copy(
        update={"name": "stored_target_profile_vs_released_sheet"}
    )


def _l3_panel_reference_is_stored_once(
    records: Sequence[dict[str, Any]],
) -> LevelResult:
    """L3: the Δ PP_0815 non-targeting reference is ONE object, not two copies.

    ``Figure 6a``'s ``PP_0815`` / ``Non-target`` arm and ``Supplementary Figure 13d``'s
    ``Non-Target`` arm are bit-identical, so the build shares one reference object
    between the records of both sheets. In the store that shows up as a single distinct
    reference phenotype across all of them: the ``Figure 6a`` Δ PP_0815 record, the
    ``Supplementary Figure 13d`` Target record and its 18 off-target records. Two
    distinct objects with the same numbers would mean the group was stored twice.
    """
    deltas = [
        record
        for record in records
        if {
            perturbation["systematic_gene_name"]
            for perturbation in record["experiment"]["genotype"]["perturbations"]
            if perturbation["perturbation_type"] == "bacterial_deletion"
        }
        == {PROTEOME_BACKGROUND_DELETION}
    ]
    distinct = {
        (
            record["reference"]["phenotype_reference"]["titer"],
            record["reference"]["phenotype_reference"]["titer_uncertainty"],
            record["reference"]["phenotype_reference"]["n_samples"],
        )
        for record in deltas
    }
    return l3_convention(
        "off_target_non_targeting_reference_is_stored_once",
        len(deltas) == OFFTARGET_SHARED_REFERENCE_RECORDS and len(distinct) == 1,
        detail=(
            f"{len(deltas)} records on the Δ{PROTEOME_BACKGROUND_DELETION} background "
            f"share {len(distinct)} distinct non-targeting reference phenotype(s); "
            f"{SHEET_KO_PANEL} and {SHEET_OFFTARGET_TITER} export that triplicate "
            "identically, so there must be exactly one"
        ),
    )


def _l4_panel_titers_vs_released_sheets(
    records: Sequence[dict[str, Any]], data_root: str | None
) -> LevelResult:
    """L4: every panel record's titer re-derived from the released sheets.

    The four panels are re-read from the deposited workbook and regrouped, so each
    stored mean is checked against the mean of the cultures the sheet releases for that
    group. This is what catches a record that drifted from its source, including the
    one group the two sheets disagree on: both triplicates must still be present, each
    under its own record.
    """
    path = str(raw_mirror_dir(data_root) / SOURCE_DATA_REL)
    rows = read_titer_rows(path)
    controls: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.is_control:
            controls[row.cycle].append(row.titer_mg_per_l)
    plan = build_panel_titer_plan(
        path,
        stored_titers={row.titer_mg_per_l for row in rows},
        control_titers=controls,
    )
    stored = Counter(
        round(float(record["experiment"]["phenotype"]["titer"]), 9)
        for record in records
        if not _is_campaign_record(record)
    )
    shared: list[tuple[str, float, float]] = []
    for group in plan.records:
        released = statistics.fmean(group.titers_mg_per_l)
        key = round(released, 9)
        if not stored[key]:
            raise AssertionError(
                f"{group.sheet} {group.group}: the released mean {released} is not a "
                "stored panel titer"
            )
        stored[key] -= 1
        shared.append((f"{group.sheet}:{group.group}", key, released))
    leftover = {value: count for value, count in stored.items() if count}
    if leftover:
        raise AssertionError(
            f"{sum(leftover.values())} stored panel titers match no released group: "
            f"{sorted(leftover)[:5]}"
        )
    return l4_cross_source(shared, tol=TITER_SE_TOL).model_copy(
        update={"name": "panel_titers_vs_released_sheets"}
    )


def _l4_overexpression_proteome_vs_released_sheet(
    records: Sequence[dict[str, Any]], data_root: str | None
) -> LevelResult:
    """L4: the overexpression records' abundances re-derived from the released sheet.

    Per protein, so a swapped pair inside an operon is caught: the ``Phnw`` / ``Phnx``
    crosswalk is the one place in this release where two pinned sources disagree about
    which locus a measured accession is, and this row re-runs that resolution from the
    bytes rather than trusting the store.
    """
    path = str(raw_mirror_dir(data_root) / SOURCE_DATA_REL)
    rows = read_overexpression_proteome_rows(path)
    genome = bacterial_genome("pputida", "KT2440", data_root)
    keys = sorted({row.protein for row in read_proteome_rows(path)})
    stored_keys, _ = reconcile_locus_tags(genome, pd.Series(keys), label="l4")
    key_map = dict(zip(keys, stored_keys, strict=True))
    assert_phn_crosswalk(rows, key_map)
    pairs, _ = overexpression_proteome_samples(rows)
    expected: dict[frozenset[str], dict[str, float]] = {}
    for sample in sorted(pairs):
        cells = _overexpression_cells(rows, sample, key_map)
        expected[frozenset(cells)] = {
            locus: statistics.fmean(values) for locus, values in cells.items()
        }
    shared: list[tuple[str, float, float]] = []
    for record in records:
        abundance = record["experiment"]["phenotype"]["protein_abundance"]
        released = expected.get(frozenset(abundance))
        if released is None:
            continue
        shared.extend(
            (locus, float(value), released[locus])
            for locus, value in sorted(abundance.items())
        )
    if len(shared) != sum(len(values) for values in expected.values()):
        raise AssertionError(
            f"{len(shared)} stored abundances join the {len(expected)} released "
            "overexpression samples; the panel is not fully represented"
        )
    return l4_cross_source(shared, tol=1e-6).model_copy(
        update={"name": "overexpression_proteome_vs_released_sheet"}
    )


def _campaign_provenance() -> Provenance:
    """Where the campaign-proteome family's numbers came from."""
    return Provenance(
        source_uri=DRYAD_TOP3_REL,
        citation_key=CITATION_KEY,
        sha256=DRYAD_TOP3_SHA256,
        method=(
            f"Dryad {DRYAD_DOI} {DRYAD_VERSION} {DRYAD_TOP3_FILENAME}, retrieved by "
            f"hand ({DRYAD_MANUAL_RECIPE}); the per-culture percent-of-proteome cells "
            "grouped by (construct, DBTL cycle) from Line_name, mean over the "
            "strain-cycle's cultures with SD / sqrt(n); blank cells are absent "
            "measurements and a released 0 is kept verbatim; UniProt accessions keyed "
            "to KT2440 locus tags through the assembly set's GOA proteome file"
        ),
        page=(
            "Dryad README, 'Files and variables'; Source Data 'Figure 4b' and "
            "'Supplementary Figure 13abc' as the cross-source oracles"
        ),
    )


def _l1_campaign_strain_cycles_pair_the_titers(
    records: Sequence[dict[str, Any]], data_root: str | None
) -> LevelResult:
    """L1: the stored genotypes are exactly the titer family's 465 campaign genotypes.

    The whole point of this family is to be the titer records' paired feature set, so
    the assertion is identity of the genotype KEY, not of a count: the sorted tuple of
    CRISPRi target loci of each record must be the same multiset the Source Data's
    non-control cultures group into.
    """

    def key(record: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(
            sorted(
                str(perturbation["systematic_gene_name"])
                for perturbation in record["experiment"]["genotype"]["perturbations"]
                if perturbation["perturbation_type"] == "bacterial_crispr_interference"
            )
        )

    genome = bacterial_genome("pputida", "KT2440", data_root)
    rows = read_titer_rows(str(raw_mirror_dir(data_root) / SOURCE_DATA_REL))
    constructs = sorted(
        {(row.construct_name, row.cycle) for row in rows if not row.is_control}
    )
    tags = sorted({tag for name, _ in constructs for tag in parse_construct(name)[0]})
    stored, _ = reconcile_locus_tags(genome, pd.Series(tags), label="l1-campaign")
    tag_map = dict(zip(tags, stored, strict=True))
    expected = Counter(
        tuple(sorted(tag_map[tag] for tag in parse_construct(name)[0]))
        for name, _ in constructs
    )
    observed = Counter(key(record) for record in records)
    passed = observed == expected
    return LevelResult(
        level=Level.L1,
        name="campaign_genotypes_are_the_titer_familys",
        passed=passed,
        message=(
            f"{len(records)} records over {len(observed)} distinct CRISPRi target sets; "
            f"the titer family's {len(constructs)} campaign strain-cycles give "
            f"{len(expected)}" + ("" if passed else "; the two no longer pair")
        ),
        details={
            "records": len(records),
            "titer_strain_cycles": len(constructs),
            "only_in_store": sorted(";".join(k) for k in (observed - expected)),
            "only_in_titers": sorted(";".join(k) for k in (expected - observed)),
        },
    )


def _l3_campaign_percent_of_proteome(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L3: no record's abundances sum past 100, because they ARE percentages.

    :data:`CAMPAIGN_TOP3` says each cell is a "Percentage of the proteome", so a
    record's sum is the share of the proteome its stored loci account for and cannot
    exceed 100 by more than the release's rounding. There is deliberately NO floor: the
    remainder is the 345 accessions with no locus tag, which carry
    :data:`CAMPAIGN_DROPPED_MASS_SHARE` of the proteome on average but 76.4% in the one
    culture that is mostly porcine trypsin, so a floor would assert a contamination
    level rather than a unit. The build-time
    :func:`assert_campaign_rows_are_percentages` is the strong form of this rule,
    checked on the released rows over ALL accessions; this one guards the store.
    """
    sums = [
        sum(record["experiment"]["phenotype"]["protein_abundance"].values())
        for record in records
    ]
    low, high = min(sums), max(sums)
    return l3_convention(
        "stored_abundances_are_percent_of_proteome",
        high <= CAMPAIGN_PERCENT_CEILING,
        detail=(
            f"per-record stored percentage sums to {low:.3f}-{high:.3f} over "
            f"{len(records)} records, under the {CAMPAIGN_PERCENT_CEILING} ceiling; the "
            "remainder of each is the dropped accessions' share of the proteome "
            f"({CAMPAIGN_DROPPED_MASS_SHARE:.1%} on average over the released cultures)"
        ),
    )


def _l4_campaign_profile_vs_deposited_matrix(
    records: Sequence[dict[str, Any]], data_root: str | None
) -> LevelResult:
    """L4: one stored profile re-derived from the deposited bytes, protein by protein.

    The strain re-read is the campaign's best, ``PP_4188`` is Yunus's; here it is the
    DBTL0 single-guide strain with the most cultures that is unambiguous to name, so
    the rule picks the first strain-cycle in sorted order deterministically.
    """
    root = raw_mirror_dir(data_root)
    proteome = read_campaign_proteome(str(root / DRYAD_TOP3_REL))
    genome = bacterial_genome("pputida", "KT2440", data_root)
    crosswalk = uniprot_locus_crosswalk(genome, data_root)
    resolution = resolve_uniprot_accessions(
        crosswalk, proteome.accessions, label="l4-campaign"
    )
    by_strain: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row_index, culture in enumerate(proteome.cultures):
        if not culture.is_control:
            by_strain[(culture.construct_name, culture.cycle)].append(row_index)
    construct, cycle = sorted(by_strain)[0]
    aggregate = aggregate_campaign_profile(
        proteome,
        by_strain[(construct, cycle)],
        resolution.resolved,
        label=f"{construct} DBTL{cycle}",
    )
    tags = sorted(parse_construct(construct)[0])
    stored, _ = reconcile_locus_tags(genome, pd.Series(tags), label="l4-campaign-tags")
    want = tuple(sorted(stored))
    match = next(
        (
            record
            for record in records
            if tuple(
                sorted(
                    str(perturbation["systematic_gene_name"])
                    for perturbation in record["experiment"]["genotype"][
                        "perturbations"
                    ]
                    if perturbation["perturbation_type"]
                    == "bacterial_crispr_interference"
                )
            )
            == want
        ),
        None,
    )
    if match is None:
        raise AssertionError(f"no stored record carries the CRISPRi target set {want}")
    abundance = match["experiment"]["phenotype"]["protein_abundance"]
    if set(abundance) != set(aggregate.abundance):
        raise AssertionError(
            f"{construct} DBTL{cycle}: the stored key set differs from the one the "
            "deposited matrix gives by "
            f"{sorted(set(abundance) ^ set(aggregate.abundance))[:10]}"
        )
    shared = [
        (locus, float(value), aggregate.abundance[locus])
        for locus, value in sorted(abundance.items())
    ]
    return l4_cross_source(shared, tol=1e-9).model_copy(
        update={"name": "stored_campaign_profile_vs_deposited_matrix"}
    )


def campaign_proteome_report(
    records: Sequence[dict[str, Any]], data_root: str | None = None
) -> VerificationReport:
    """The campaign-proteome family's L0-L4 report over already-loaded records."""
    from torchcell.verification.protein import verify_protein_dataset

    report = verify_protein_dataset(
        [dict(record) for record in records],
        dataset_name="campaign_proteome_carruthers2025",
        provenance=_campaign_provenance(),
        expected_count=EXPECTED_CAMPAIGN_PROTEOME_RECORDS,
        # The five pIY670 pathway tokens are in every record, as in the panel family.
        allow_duplicate_orfs=True,
    )
    report.add(_l1_campaign_strain_cycles_pair_the_titers(records, data_root))
    report.add(_l3_campaign_percent_of_proteome(records))
    report.add(_l4_campaign_profile_vs_deposited_matrix(records, data_root))
    return report


def _fold_change_provenance() -> Provenance:
    """Where the fold-change family's numbers came from."""
    return Provenance(
        source_uri=SOURCE_DATA_REL,
        citation_key=CITATION_KEY,
        sha256=SOURCE_DATA_SHA256,
        method=(
            f"Source Data sheets '{SHEET_FOLD_CHANGE}' (14 single-guide CRISPRi "
            f"contrasts) and '{SHEET_KO_FOLD_CHANGE}' (the 2 KO contrasts the Fig. 6 "
            "caption names, of 10 released), columns '<contrast>_log2_FC' and "
            "'<contrast>_log10_pval'; the fold change is stored verbatim on the log2 "
            "scale and the p-value as the probability p = 10**-x"
        ),
        page=(
            f"Source Data '{SHEET_FOLD_CHANGE}' and '{SHEET_KO_FOLD_CHANGE}'; the "
            f"Fig. 5 and Fig. 6 captions and '{SHEET_FOLD_CHANGE_POINTER}' as the "
            "statements of scale, test and denominator"
        ),
    )


def _fold_change_contrast(record: Mapping[str, Any]) -> tuple[str, str]:
    """``(sheet, contrast)`` of one stored fold-change record, read off its genotype.

    A record of the single-guide panel carries one CRISPRi leaf and no deletion; a
    record of the KO panel carries a deletion of the SAME locus beside it. That is the
    whole distinction, so neither sheet name nor contrast is stored twice.
    """
    knockdowns = sorted(
        str(perturbation["systematic_gene_name"])
        for perturbation in record["experiment"]["genotype"]["perturbations"]
        if perturbation["perturbation_type"] == "bacterial_crispr_interference"
    )
    deletions = sorted(
        str(perturbation["systematic_gene_name"])
        for perturbation in record["experiment"]["genotype"]["perturbations"]
        if perturbation["perturbation_type"] == "bacterial_deletion"
    )
    if len(knockdowns) != 1:
        raise AssertionError(
            f"a fold-change record carries {len(knockdowns)} CRISPRi leaves"
        )
    if not deletions:
        return SHEET_FOLD_CHANGE, knockdowns[0]
    if deletions != knockdowns:
        raise AssertionError(
            f"a KO fold-change record deletes {deletions} and knocks down {knockdowns}"
        )
    return SHEET_KO_FOLD_CHANGE, knockdowns[0]


def _released_fold_change_cells(
    data_root: str | None,
) -> tuple[dict[tuple[str, str, str], FoldChangeRow], dict[str, str]]:
    """Re-read both released sheets and reconcile their keys, for the L4 oracles."""
    path = str(raw_mirror_dir(data_root) / SOURCE_DATA_REL)
    rows = [
        *read_fold_change_rows(path, SHEET_FOLD_CHANGE),
        *read_fold_change_rows(path, SHEET_KO_FOLD_CHANGE),
    ]
    genome = bacterial_genome("pputida", "KT2440", data_root)
    keys = sorted({row.protein for row in rows})
    stored, _ = reconcile_locus_tags(genome, pd.Series(keys), label="l4-fold-change")
    key_map = dict(zip(keys, stored, strict=True))
    return (
        {(row.sheet, row.contrast, key_map[row.protein]): row for row in rows},
        key_map,
    )


def _l1_fold_change_contrast_coverage(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L1: the stored contrasts are exactly the sourced ones, per sheet."""
    observed: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sheet, contrast = _fold_change_contrast(record)
        observed[sheet].add(contrast)
    ko = sorted(observed[SHEET_KO_FOLD_CHANGE])
    holds = len(
        observed[SHEET_FOLD_CHANGE]
    ) == EXPECTED_FOLD_CHANGE_RECORDS_FIG5B and ko == sorted(KO_FOLD_CHANGE_CONTRASTS)
    return LevelResult(
        level=Level.L1,
        name="fold_change_contrast_coverage",
        passed=holds,
        message=(
            f"{len(observed[SHEET_FOLD_CHANGE])} {SHEET_FOLD_CHANGE} contrasts and "
            f"{SHEET_KO_FOLD_CHANGE} contrasts {ko}"
        ),
        details={
            "n_single_guide": len(observed[SHEET_FOLD_CHANGE]),
            "ko_contrasts": ko,
            "expected_single_guide": EXPECTED_FOLD_CHANGE_RECORDS_FIG5B,
            "expected_ko_contrasts": sorted(KO_FOLD_CHANGE_CONTRASTS),
        },
    )


def _l3_fold_change_triplicate(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L3: every protein's replicate count is the Methods' biological triplicate."""
    expected = int(N_REPLICATES.value)
    counts = {
        int(value)
        for record in records
        for value in record["experiment"]["phenotype"]["n_replicates"].values()
    }
    return l3_convention(
        "fold_change_biological_triplicate",
        counts == {expected},
        detail=(
            f"replicate counts {sorted(counts)}; the Methods state "
            f"'{_Q_TRIPLICATE}', so every fold change carries {expected}"
        ),
    )


def _l4_fold_change_vs_released_sheets(
    records: Sequence[dict[str, Any]], data_root: str | None
) -> LevelResult:
    """L4: every stored fold change is the released cell, re-read from the mirror."""
    cells, _ = _released_fold_change_cells(data_root)
    shared: list[tuple[str, float, float]] = []
    for record in records:
        sheet, contrast = _fold_change_contrast(record)
        for tag, value in sorted(
            record["experiment"]["phenotype"]["protein_fold_change"].items()
        ):
            row = cells.get((sheet, contrast, tag))
            if row is None:
                raise AssertionError(
                    f"{sheet}/{contrast}/{tag} is stored but is not in the released "
                    "sheet under its reconciled key"
                )
            shared.append((f"{sheet}/{contrast}/{tag}", value, row.log2_fold_change))
    return l4_cross_source(shared, tol=0.0).model_copy(
        update={"name": "stored_fold_changes_vs_released_sheets"}
    )


def _l4_fold_change_p_values_vs_released_sheets(
    records: Sequence[dict[str, Any]], data_root: str | None
) -> LevelResult:
    """L4: every stored p-value inverts to the released -log10 value exactly.

    The stored number is a probability and the released number is -log10 of it, so the
    oracle is the inverse of the loader's own conversion rather than a re-derivation.
    """
    cells, _ = _released_fold_change_cells(data_root)
    shared: list[tuple[str, float, float]] = []
    for record in records:
        sheet, contrast = _fold_change_contrast(record)
        p_values = record["experiment"]["phenotype"]["protein_fold_change_p_value"]
        for tag, probability in sorted(p_values.items()):
            row = cells[(sheet, contrast, tag)]
            shared.append(
                (
                    f"{sheet}/{contrast}/{tag}",
                    -math.log10(float(probability)),
                    row.neg_log10_p_value,
                )
            )
    return l4_cross_source(shared, tol=P_VALUE_ROUND_TRIP_TOL).model_copy(
        update={"name": "stored_p_values_invert_to_released_neg_log10"}
    )


def fold_change_report(
    records: Sequence[dict[str, Any]], data_root: str | None = None
) -> VerificationReport:
    """The fold-change family's L0-L4 report over already-loaded records."""
    from torchcell.verification.protein_fold_change import (
        verify_protein_fold_change_dataset,
    )

    report = verify_protein_fold_change_dataset(
        [dict(record) for record in records],
        dataset_name="proteome_fold_change_carruthers2025",
        provenance=_fold_change_provenance(),
        expected_count=EXPECTED_FOLD_CHANGE_RECORDS,
    )
    report.add(_l1_fold_change_contrast_coverage(records))
    report.add(_l3_fold_change_triplicate(records))
    report.add(_l4_fold_change_vs_released_sheets(records, data_root))
    report.add(_l4_fold_change_p_values_vs_released_sheets(records, data_root))
    return report


def titer_report(
    records: Sequence[dict[str, Any]], data_root: str | None = None
) -> VerificationReport:
    """The titer family's L0-L4 report over already-loaded records."""
    report = verify_product_titer_dataset(
        [dict(record) for record in records],
        dataset_name="isoprenol_titer_carruthers2025",
        provenance=_titer_provenance(),
        expected_count=EXPECTED_TITER_RECORDS,
        titer_unit=ConcentrationUnit.ug_per_ml.value,
        titer_unit_detail=(
            "Supplementary Data 1 and the Source Data release mg/L and "
            "ConcentrationUnit has no mg/L member; 1 mg/L == 1 ug/mL exactly, so the "
            "released number is stored verbatim under the numerically identical unit"
        ),
        se_tol=TITER_SE_TOL,
        pathway_gene_counts=(len(PATHWAY_GENES),),
        product_names=(PRODUCT_NAME,),
    )
    report.add(_l1_panel_partition(records))
    report.add(_l1_strain_count_reconciles(records))
    report.add(_l3_panel_reference_is_stored_once(records))
    report.add(_l4_titer_vs_supplementary_data_1(records, data_root))
    report.add(_l4_ko_array_titer_vs_results_text(records))
    report.add(_l4_panel_titers_vs_released_sheets(records, data_root))
    return report


def proteome_report(
    records: Sequence[dict[str, Any]], data_root: str | None = None
) -> VerificationReport:
    """The proteome family's L0-L4 report over already-loaded records."""
    from torchcell.verification.protein import verify_protein_dataset

    report = verify_protein_dataset(
        [dict(record) for record in records],
        dataset_name="proteome_carruthers2025",
        provenance=_proteome_provenance(),
        expected_count=EXPECTED_PROTEOME_RECORDS,
        # The panel repeats a genotype by design: PP_0977 and PP_1638 were "cultured
        # three times owing to poor transformation efficiency", each kept as its own
        # record, and the five pIY670 tokens plus the PP_0815 background are in every
        # record. Per-record uniqueness is what the L1 count asserts.
        allow_duplicate_orfs=True,
    )
    report.add(_l1_protein_key_sets(records))
    report.add(_l3_biological_triplicate(records))
    report.add(_l4_proteome_vs_released_sheet(records, data_root))
    report.add(_l4_overexpression_proteome_vs_released_sheet(records, data_root))
    return report


def verify_build(
    dataset_root: str, data_root: str | None = None, *, family: Family
) -> VerificationReport:
    """Run this release's L0-L4 gate over a built tree and write the report.

    ``family`` is ``"titer"``, ``"proteome"``, ``"campaign_proteome"`` or
    ``"fold_change"``. The report is written to
    ``<dataset_root>/preprocess/verification_report.json``.
    """
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    build = {
        "titer": titer_report,
        "proteome": proteome_report,
        "campaign_proteome": campaign_proteome_report,
        "fold_change": fold_change_report,
    }[family]
    report = build(records, data_root)
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build/load all three families for interactive debugging.

    Verification is NOT run here: the L4 oracles need the real raw mirror, and
    ``run_product_titer`` / ``run_bacterial_protein_abundance`` in
    ``torchcell.verification.runners`` are the entry points that run them.
    """
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    genome = bacterial_genome("pputida", "KT2440", data_root)
    for cls, rel in (
        (
            IsoprenolTiterCarruthers2025Dataset,
            "data/torchcell/isoprenol_titer_carruthers2025",
        ),
        (ProteomeCarruthers2025Dataset, "data/torchcell/proteome_carruthers2025"),
        (
            ProteomeFoldChangeCarruthers2025Dataset,
            "data/torchcell/proteome_fold_change_carruthers2025",
        ),
        (
            CampaignProteomeCarruthers2025Dataset,
            "data/torchcell/campaign_proteome_carruthers2025",
        ),
    ):
        root = osp.join(data_root, rel)
        dataset = cls(root=root, pputida_genome=genome)
        print(f"{cls.__name__}: len = {len(dataset)}")
        accounting = json.loads(
            Path(osp.join(root, "preprocess/build_accounting.json")).read_text()
        )
        print(
            json.dumps(
                {
                    key: accounting[key]
                    for key in (
                        "source_rows",
                        "control_rows",
                        "candidate_records",
                        "kept_records",
                        "dropped_records",
                        "notes",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
