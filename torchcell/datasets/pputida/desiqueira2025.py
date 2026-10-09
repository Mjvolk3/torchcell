# torchcell/datasets/pputida/desiqueira2025
# [[torchcell.datasets.pputida.desiqueira2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/desiqueira2025
# Test file: tests/torchcell/datasets/pputida/test_desiqueira2025.py
"""de Siqueira 2025 acetate tolerization of P. putida KT2440 and its isoprenol titers.

de Siqueira et al. 2025 (Appl Environ Microbiol, doi:10.1128/aem.02123-24) evolved the
isoprenol-catabolism-deficient strain ``PT`` on acetate, recovered five tolerized
``Sigma``-class isolates, resequenced four of them plus the parent, and measured a
global proteome and plasmid-based isoprenol titers. This module serves the two released
numeric readouts as four dataset classes: two because
``ExperimentDataset.transform_item`` validates against ONE ``experiment_class``, and
three of the four because the proteome is released on THREE normalizations and the
shared ``verify_protein_dataset`` asserts one ``measurement_type`` per dataset ("no
silent cross-assay mixing"):

- :class:`ProteomeDeSiqueira2025Dataset` -- ``BacterialProteinAbundanceExperiment``, one
  record per released Top3 proteome sample of a WRITABLE strain.
- :class:`ProteomePercentDeSiqueira2025Dataset` -- the same five samples as percent of
  each sample's total abundance, the scale the paper's own analysis runs on.
- :class:`ProteomeLog10PercentDeSiqueira2025Dataset` -- the same five samples as the
  mean of the three replicates' log10 percent.
- :class:`IsoprenolTiterDeSiqueira2025Dataset` -- ``ProductTiterExperiment``, one record
  per released Table S2 titer of a WRITABLE strain in a medium the library holds.

THREE NORMALIZATIONS, AND NEITHER OF THE TWO NEW ONES IS RECOVERABLE FROM THE FIRST.
Data Set S1's 15 columns carry the same (protein, sample) cell on three scales, each as
its own mean + SD pair (:data:`PROTEOME_NORMALIZATIONS`). The first build asserted the
first NINE header cells (``header[: len(PROTEOME_HEADER)]`` against a nine-name tuple) and
read ``row[7]`` and ``row[8]``, so columns 9 to 14 were neither asserted nor read.
``PROTEOME_HEADER`` now names all 15 and every one of them reaches a row field or an
oracle. Storing a second and third normalization of one proteome is only worth doing if
they cannot be re-derived from the first, and that was MEASURED over every one of the
34,600 released cells rather than argued:

- **Percent of total is not percent of the stored mean.** Against
  ``100 * top3_mean / sum(top3_mean)`` within the sample, ZERO of 34,600 cells agree
  exactly, 9,322 agree to 5e-4 relative, and the per-cell ratio runs 0.914506 to
  1.083554 with a median of 1.000007. Each sample's released percents do sum to 100.000,
  so this is not a scaling error: it is a replicate-wise mean of per-replicate
  percentages, each replicate normalized by its OWN total. There is no way back to it
  from a mean of counts.
- **Log10 percent is the mean of logs, not the log of the mean.** Against
  ``log10(released percent)``, 33,914 of 34,600 cells sit STRICTLY BELOW the released
  value, 686 agree to within 5e-5, and ZERO sit above it. A one-sided result over 34,600
  cells is Jensen's inequality for a mean of logarithms, with equality only where the
  three replicates coincide; the largest gap is 1.378950 in log10 units. The first
  released row (protein ``Csda``) gives -2.23385590492606 against a log10 of the mean of
  -2.19547. Its SD is likewise not the delta-method transform of the percent SD
  (``pct_sd / (pct_mean * ln 10)`` disagrees on 34,496 of 34,600 cells).

THE TWO DERIVED COLUMNS ARE ORACLES, NOT PHENOTYPES, AND THAT IS ALSO MEASURED.
``CV%_of_%_protein_abundance`` is exactly ``100 * pct_sd / pct_mean`` (zero of 34,600
cells disagree, :func:`assert_percent_cv_is_derived`), so it carries nothing the stored
percent pair does not. ``%_of protein_abundance_Top3_rep_mean_sem`` does not vary with
the sample: 1,728 of 1,729 proteins carry a single value
(:func:`assert_sem_is_constant_per_protein`), so a per-sample record has nothing to put
in it. Both are read and asserted at build time rather than stored, which is what keeps
every asserted header cell from being parsed past.

THE GENOTYPE FINDING, WHICH DECIDES WHAT IS LOADED (issue #731, resolved). This study's
strains are an evolution experiment plus whole-genome resequencing, so an evolved
clone's genotype IS its parent plus the variants Geneious called ("Trimmed and filtered
reads were mapped to the reference genome (NCBI SRA NC_002947.4), and in turn, variant
calls were identified using the software package Geneious (BioMatters LLC) with default
parameters specified"). Data Set S2 releases 173 calls over the five sequenced clones
(PT 33, Sigma1 34, Sigma2 28, Sigma4 43, Sigma5 35) at 83 distinct sites, 33 of which
are shared by more than one clone. The first build of this loader measured that no class
in ``schema.py`` could hold one of those calls, counted all 173 of them into
``preprocess/called_variants.json`` with its reason, and loaded only WT and PT. The
called-variant leaves of issue #731 close every one of those four reasons, and the four
sequenced isolates are now loaded. What each reason became:

1. ``SequenceVariantPerturbation`` admits only S288C ORF names and promises an off-graph
   allele sequence this paper never deposited (it released SRA reads, PRJNA1153078)
   -> ``BacterialSequenceVariantPerturbation``, which validates a ``PP_`` locus tag and
   carries the CALL on the record rather than a pointer to a store that does not exist.
2. ``BacterialBackgroundAllele(edit=sequence_variant)`` required a ``functional: bool``
   no release states for a missense SNP, and had no slot for the position, the reference
   and alternate base, the amino-acid change, the polymorphism type or the variant
   frequency -> ``BacterialVariantCall`` carries all of them, verbatim, and the
   perturbation axis has no ``functional`` field at all.
3. ``BacterialStrainBackground`` permits ONE allele entry per locus, which refused
   Sigma1 (``PP_1656`` twice, ``PP_3827`` three times), Sigma2 (``PP_1656`` twice) and
   Sigma4 (``PP_1656`` twice, ``PP_3827`` twice) -> ``Genotype.perturbations`` has no
   such rule, so two calls in one locus are two perturbations.
4. 105 of the 173 calls carry no locus at all and 10 more carry only a RefSeq ``PP_RS``
   tag -> ``BacterialSiteVariantPerturbation``, keyed on the derived site id
   ``<replicon>:<position>``, with ``site_kind`` saying which of the two it is. The 10
   RefSeq-only calls are ``locus_not_in_assembly``, and that is measured rather than
   assumed: the two tags involved are ``PP_RS21780`` (9 rows) and ``PP_RS19075`` (1),
   and ``resolve_gene_name`` on the pinned GenBank assembly returns ``retired``, "not
   found in GCA_000007565.2_ASM756v2; retained as given", for both, so there is no tag
   to key them to and the released identifier is kept verbatim instead.

``Genotype.__eq__`` compares the perturbation SET, and the five written strains carry 33,
34, 28, 43 and 35 perturbations respectively, so no two of them collapse. Sigma3 is still
NOT loaded, and no schema change reaches it: Data Set S2 releases no row for it, so its
genotype is unknown rather than untyped, and writing it would assert it is genotypically
the PT it was evolved from.

ONE CALL PER STRAIN IS DROPPED, AND COUNTED. All five sequenced clones carry a
``Deletion`` call at 3,063,718..3,064,173 on ``PP_2675``, 456 bp of the ``cytochrome
c-550 PedF`` CDS, which IS the markerless in-frame deletion the strain was built with
and which every record already carries as a ``BacterialDeletionPerturbation``. Absence
has one encoding, so the call is dropped with the encoding
``restates_the_designed_pt_deletion`` rather than written twice.

WHAT IS WRITABLE. ``WT`` is P. putida KT2440 itself ("<td>WT</td><td>P. putida
KT2440</td><td>Wild type strain</td><td>ATCC 47054 (JBEI-18711)</td>"), so its records
carry the bare assembly reference and an empty genotype. ``PT`` is a DEFINED-deletion
strain ("A mutant strain of Pseudomonas putida KT2440 (DeltaPP_2675 Delta14-PP_2676,
referred to as "pre-tolerized" or "PT" throughout this manuscript) was used as the base
strain"), so its records carry a typed ``BacterialStrainBackground``.

PT'S TWO DESIGNATIONS ARE ONE DELETION EVENT, AND ONLY ONE OF THEM HAS A PERTURBATION
LEAF. The nomenclature is explained in the Results: "This updated nomenclature accounts
for the PP_2676 ORF overlapping with the coding sequence of PP_2675 ... with its
predicted start codon residing within the CDS of PP_2675 (45)". The deferral to
Thompson 2020 (reference 30, mirrored) settles the PP_2675 half verbatim: "Strain with
complete internal in-frame deletion of PP_2675". So:

- ``PP_2675`` is a ``BacterialDeletionPerturbation`` in ``Genotype`` -- the ML-facing
  edit set and the gene node a record links to.
- ``Delta14-PP_2676`` is a PARTIAL deletion, and the bacterial perturbation axis has no
  partial-deletion leaf (``BacterialDeletionPerturbation`` is ``state="absent"``). It is
  typed where ``AlleleEdit.partial_deletion`` exists, as a ``BacterialBackgroundAllele``
  on the PT background, with a ``ProvenanceGap`` on ``deleted_span``: the source writes
  "Delta14" without saying whether 14 is base pairs or codons and without coordinates.

``PP_2675`` therefore appears twice, once as a background allele and once as a
perturbation. That is deliberate: ``BacterialStrainBackground.alleles`` is documented as
"every allele of the background", so listing one of the two would be an omission, while
the perturbation is what the graph and the model read.

PT'S OWN RESEQUENCING CALLS ARE NOW ON ITS RECORDS. PT carries 33 called variants of its
own, two of them called out in the Figure S4 caption; 32 of the 33 are written as
perturbations and the 33rd is the restatement of its designed deletion. Before the #731
leaves they were missing ROWS rather than missing fields, so they were never a
``ProvenanceGap`` (a gap must name a field that is ``None``) and lived only in
``preprocess/called_variants.json``; that ledger is still written, now recording the
leaf each call became.

TITER UNITS AND THE STATISTIC. Table S2 releases millimolar, and ``ConcentrationUnit``
has ``millimolar``, so the number is stored verbatim with no arithmetic. What is stored
is a MAXIMUM over the 24, 48 and 72 h sampling times ("Maximum isoprenol titers (mM)
detected"), an upward-biased order statistic, which is why ``duration_hours`` is ``None``
with a gap rather than a single time. ``ProductTiterExperiment.environment`` is annotated
``CultureEnvironment``, so the vessel and volume stated in :data:`CULTURE_FORMAT` travel
on the record; the proteome family's slot is still ``Environment``, which drops them.

ISOPRENOL'S COMPOUND-IDENTITY ROW EXISTS. ``resolved_compound("isoprenol")`` returns the
full identity (PubChem CID 12988, InChIKey ``CPJRRXSHAYUTGL-UHFFFAOYSA-N``), so the
product carries it rather than a gap; :data:`ISOPRENOL_INCHIKEY` is kept as the
cross-check that the table's row is the molecule this paper means.

TWO CROSS-SOURCE ASSERTIONS, BOTH MEASURED ON THE PINNED BYTES AND BOTH ENFORCED AT
BUILD TIME:

1. Table S2's 3.29 mM for PT in M9 glucose equals the Results text's "the base PT strain
   produced 283 +/- 26 mg/L isoprenol by the 48-h time point": 283 mg/L divided by the
   86.13 g/mol of isoprenol is 3.286 mM, within Table S2's own two-decimal rounding. The
   molar mass is used ONLY for this check and never to produce a stored number.
2. The Table 1 pIY670 part string agrees token for token, case-folded, with the
   Carruthers 2025 Supplementary Data 3 description of the same plasmid, except that
   this paper's OCR dropped "Sc" from ``PMDScHKQ``. That is what licenses reusing the
   Carruthers-sourced ``source_organism`` for the five heterologous genes, which this
   paper never states, and it is why the stored identifiers are the workbook spellings
   rather than the OCR spellings: an OCR case loss must not fork a gene node.

ONE CROSS-SOURCE DISAGREEMENT, KEPT RATHER THAN RESOLVED. For PT in the mixed feed the
Results say "the baseline PT strain failed to grow and therefore did not produce any
detectable isoprenol", while Table S2 releases 0.05 mM for that cell. 0.05 mM is about
4.3 mg/L, a trace near the GC-FID floor. The released number is stored, because the
table is the explicit numeric release and the text's claim is qualitative, and the
disagreement is a counted finding in the accounting and in the verification report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import os.path as osp
import shutil
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
from torchcell.datamodels.media import M9_NREL_DESIQUEIRA2025
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
    BacterialSequenceVariantPerturbation,
    BacterialSiteVariantPerturbation,
    BacterialStrainBackground,
    BacterialVariantCall,
    BacterialVariantType,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    CultureFormat,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    Genotype,
    HeterologousPathwayPerturbation,
    PhysicalFactor,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    ProteinAbundancePhenotype,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainConstruction,
    Temperature,
    VariantCallMode,
    VariantFrequencyBasis,
    VariantSiteKind,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.pputida.carruthers2025 import (
    PATHWAY_GENES,
    PATHWAY_SOURCE,
    PIY670_PARTS,
)
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

DOI = "10.1128/aem.02123-24"
PMCID = "PMC12016510"
TITLE = (
    "Alternate routes to acetate tolerance lead to varied isoprenol production from "
    "mixed carbon sources in Pseudomonas putida"
)

CITATION_KEY = "desiqueiraAlternateRoutesAcetate2025"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: PMC Article Datasets bucket prefix of this article's open-access version.
PMC_PREFIX = f"{PMCID}.1"

#: Data Set S1: the per-(protein, strain, condition) Top3 proteome matrix.
PROTEOME_FILENAME = "aem.02123-24-s0001.xlsx"
PROTEOME_REL = f"si/{PROTEOME_FILENAME}"
PROTEOME_SHA256 = "6bd1889df343646fdcd972fb6b7d3c6c014fcd428a4a78d8363908cccfbddeae"
#: Data Set S2: the Geneious variant calls for the five sequenced clones.
VARIANTS_FILENAME = "aem.02123-24-s0002.xlsx"
VARIANTS_REL = f"si/{VARIANTS_FILENAME}"
VARIANTS_SHA256 = "7ba609170ec7233e09e0ffbbfdb59a190353180ad48b50c45ab60850cd713633"
SI_RETRIEVED_AT = "2026-10-07"

#: Mirrored OCR files every sourced value quotes (torchcell-library, not torchcell-raw).
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "a3ea14adcbe77144f02fb92ae6b344e4dec637a4211c702aa29fa121eda2de9c"
SI3_MD = "si/si3.md"
SI3_MD_SHA256 = "10c875184be296016ba9416bf9fda2e817a2f80dc2c5eccd4a9f2ba4c57981fc"
SI3_PDF = "si/si3.pdf"
SI3_PDF_SHA256 = "de5fcfc522e7b7a6fb164eac9632e333f90bb2eba7617a925eb5154a7293ea9e"

#: Thompson 2020 (reference 30), the mirrored paper PT's PP_2675 deletion defers to.
THOMPSON_KEY = "thompsonFattyAcidAlcohol2020"
THOMPSON_SHA256 = "389d0d6cd196f9f159d0485f6b8ed09743dfbe3b6682369db489d7356a469b45"

#: The two deposited accessions this study's primary data live in. Neither is consumed
#: by a loader here: PRIDE holds raw DIA spectra and the BioProject holds raw reads.
SRA_BIOPROJECT = "PRJNA1153078"
PRIDE_ACCESSION = "PXD055153"

KT2440_NAMESPACE: BacterialGeneNamespace = "pputida_kt2440_locus_tag"
KT2440_ASSEMBLY_SET: BacterialAssemblySet = "pputida_KT2440_ASM756v2"

#: The schema's reference-strain name for the wild type, and the label the released
#: files give it. Table 1 reads "<td>WT</td><td>P. putida KT2440</td>".
WT_STRAIN: BacterialReferenceStrain = "KT2440"
WT_LABEL = "WT"
PT_STRAIN = "PT"
#: The five tolerized isolates, spelled as Data Set S1's ``Strain`` column spells them
#: (U+03A3). The Table S2 OCR writes the same five with U+2211. None is loadable.
SIGMA_STRAINS: tuple[str, ...] = ("Σ1", "Σ2", "Σ3", "Σ4", "Σ5")
#: The tolerized isolate Data Set S2 releases no calls for. Measured on the pinned
#: workbook: its ``Strain`` column carries five labels and Sigma3 is not one of them, so
#: that isolate was never sequenced and its genotype is unknown, not merely untyped.
UNSEQUENCED_SIGMA = SIGMA_STRAINS[2]
#: The four tolerized isolates Data Set S2 DOES release calls for, in released order.
#: Asserted against the ledger at build time rather than trusted.
SEQUENCED_SIGMA_STRAINS: tuple[str, ...] = tuple(
    strain for strain in SIGMA_STRAINS if strain != UNSEQUENCED_SIGMA
)
#: The released ``Strain`` labels whose genotype can be written (issue #731). The four
#: sequenced isolates joined WT and PT when the called-variant leaves landed: a clone's
#: genotype is PT's deletion plus the variants Geneious called for it, and
#: ``Genotype.__eq__`` now separates the five strains because their call sets differ.
#: Sigma3 stays out, for a reason no schema change reaches: it was never sequenced.
WRITABLE_STRAINS: tuple[str, ...] = (WT_LABEL, PT_STRAIN, *SEQUENCED_SIGMA_STRAINS)

#: The replicon every Data Set S2 coordinate is on, as its ``Sequence Name`` column
#: writes it. Measured: all 173 rows read this one value, and the Methods name the same
#: sequence ("the reference genome (NCBI SRA NC_002947.4)"). Asserted per row, never
#: assumed, so a workbook revision that changed replicon would stop the build.
VARIANT_REPLICON = "NC_002947 (2)"
#: ``Polymorphism Type`` -> the typed variant kind. Every one of the seven released
#: spellings is mapped (measured over the 173 rows: SNP (transition) 68, SNP
#: (transversion) 47, Insertion 23, Substitution 21, Insertion (tandem repeat) 5,
#: Deletion 5, Deletion (tandem repeat) 4); the released spelling is kept verbatim in
#: the call's ``type_statement``, so the transition / transversion / tandem-repeat
#: distinction the enum does not carry is not lost. A spelling outside this map stops
#: the build rather than defaulting.
VARIANT_TYPE_OF_POLYMORPHISM: dict[str, BacterialVariantType] = {
    "SNP (transition)": BacterialVariantType.snv,
    "SNP (transversion)": BacterialVariantType.snv,
    "Insertion": BacterialVariantType.insertion,
    "Insertion (tandem repeat)": BacterialVariantType.insertion,
    "Deletion": BacterialVariantType.deletion,
    "Deletion (tandem repeat)": BacterialVariantType.deletion,
    "Substitution": BacterialVariantType.substitution,
}
#: The separator the ``Variant Frequency`` column uses when it writes a RANGE instead of
#: one number. Measured: 16 of the 173 cells read a percent range (``95.1% -> 97.6%`` ..
#: ``98.5% -> 98.6%``) and the other 157 are bare fractions in (0, 1]. A range has no
#: single value, so those calls carry the verbatim cell and a null numeric frequency.
VARIANT_FREQUENCY_RANGE = "->"

#: PT's two designations and the loci they name.
PT_FULL_DELETION = "PP_2675"
PT_PARTIAL_DELETION = "PP_2676"

#: What one stored proteome number is, named so heterogeneous proteomics never mixes.
PROTEOME_MEASUREMENT_TYPE = "dia_nn_top3_peptide_signal_replicate_mean"
#: The two FURTHER normalizations Data Set S1 releases of the same five samples, each
#: its own ``measurement_type`` and so its own dataset class: the shared protein
#: verifier's L3 ``measurement_type_consistent`` row requires one per dataset.
PERCENT_MEASUREMENT_TYPE = "dia_nn_top3_percent_of_total_abundance_replicate_mean"
LOG10_PERCENT_MEASUREMENT_TYPE = (
    "dia_nn_top3_log10_percent_of_total_abundance_replicate_mean"
)

#: Data Set S1 ``Condition`` labels, and the carbon regime each one means.
CONDITION_ACETATE = "Acetate"
CONDITION_GLUCOSE = "Glucose"
CONDITION_MIXED = "Glucose + Acetate"
PROTEOME_CONDITIONS: tuple[str, ...] = (
    CONDITION_ACETATE,
    CONDITION_GLUCOSE,
    CONDITION_MIXED,
)
#: The condition every proteome record is referenced against (Fig. 4 caption).
PROTEOME_REFERENCE_CONDITION = CONDITION_GLUCOSE
#: The strain the proteome reference is measured on: the wild type.
PROTEOME_REFERENCE_STRAIN = WT_LABEL
#: Released proteome samples that become records. Measured on the pinned workbook: 20
#: samples are released, 7 strains times 3 conditions less the PT mixed-carbon sample,
#: which was never taken (PT "often fails to grow in glucose-acetate medium"). 17 of the
#: 20 are of a writable strain once the #731 leaves let a sequenced isolate carry its
#: calls; the 3 left out are Sigma3's, the isolate Data Set S2 never sequenced. One
#: record per writable sample per normalization, so each of the three proteome dataset
#: classes holds exactly this many.
EXPECTED_PROTEOME_RECORDS = 17

#: This module's labels for Table S2's four media columns, in released order. The
#: released headers write the carbon total's C in quotation marks.
TITER_COLUMNS: tuple[str, ...] = (
    "M9 Acetate (75 mM C)",
    "M9 Acetate (100 mM C)",
    "M9 Glucose (111 mM C)",
    "M9 Glucose + Acetate (221 mM C)",
)
#: The two Table S2 columns whose medium ``MEDIA_LIBRARY`` holds. The two acetate
#: columns are Fig. S3's media A and C, whose ammonium sulfate (75.5 mM and 1 mM)
#: differs from the 2 g/L the served M9 object states, so they are a different medium.
TITER_COLUMNS_LOADED: tuple[str, ...] = (TITER_COLUMNS[2], TITER_COLUMNS[3])
#: The Table S2 column the titer records are referenced against.
TITER_REFERENCE_COLUMN = TITER_COLUMNS[2]
#: The Table S2 rows that become records: PT and the four sequenced isolates. Sigma3's
#: two determined cells stay out, because its genotype is unknown (it was never
#: sequenced), not because no class could hold it.
TITER_STRAINS_LOADED: tuple[str, ...] = (PT_STRAIN, *SEQUENCED_SIGMA_STRAINS)

#: Isoprenol's InChIKey, kept as the cross-check on the compound table's row for it
#: (PubChem CID 12988), which ``resolved_compound`` now returns in full.
ISOPRENOL_INCHIKEY = "CPJRRXSHAYUTGL-UHFFFAOYSA-N"
#: Isoprenol's molar mass, used ONLY to check Table S2's mM against the Results text's
#: mg/L. No stored number is produced from it.
ISOPRENOL_G_PER_MOL = 86.13
#: Tolerance of that check, in mM: Table S2 rounds to two decimals.
TITER_CROSS_SOURCE_TOL_MM = 0.01


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
    """Bind a value to a verbatim quote in the pinned Supplementary Material OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI3_MD,
            citation_key=CITATION_KEY,
            sha256=SI3_MD_SHA256,
            method="MinerU OCR of the Supplemental Material PDF (mirror)",
            page=page,
        ),
    )


def _data_s1(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim cell of the pinned Data Set S1 workbook bytes."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PROTEOME_REL,
            citation_key=CITATION_KEY,
            sha256=PROTEOME_SHA256,
            method="openpyxl read of the deposited Data Set S1 workbook (raw mirror)",
            page=page,
        ),
    )


def _thompson(value: Any, quote: str, *, page: str, note: str | None) -> SourcedValue:
    """Bind a value to the mirrored Thompson 2020 paper PT's deletion defers to."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=THOMPSON_KEY,
            sha256=THOMPSON_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=page,
        ),
    )


_METHODS_STRAINS = "Methods, 'Strains and plasmids'"
_METHODS_CULTURE = "Methods, 'Culturing conditions'"
_METHODS_PROTEOMICS = "Methods, 'Proteomics analysis'"
_METHODS_PRODUCTION = "Methods, 'Isoprenol production runs'"

_Q_PT_STRAIN = (
    "A mutant strain of Pseudomonas putida KT2440 (ΔPP_2675 Δ14-PP_2676, "
    "referred to as “pre-tolerized” or “PT” throughout this "
    "manuscript) was used as the base strain for the tolerization described in this "
    "work."
)
_Q_NOMENCLATURE = (
    "This updated nomenclature accounts for the PP_2676 ORF overlapping with the "
    "coding sequence of PP_2675 in $\\mathsf { a } + \\mathsf { \\Omega } \\mathsf { "
    "1 }$ reading frame, with its predicted start codon residing within the CDS of "
    "PP_2675 (45)."
)
_Q_WT_ROW = (
    "<td>WT</td><td>P. putida KT2440</td><td>Wild type strain</td><td>ATCC 47054 "
    "(JBEI-18711)</td>"
)
_Q_PT_ROW = (
    "<td>PT</td><td>P. putida KT2440 ΔPP_2675 Δ14-PP_2676</td><td>"
    "Pre-tolerization strain</td><td>(30) (JBEI-147164)</td>"
)
_Q_ABSTRACT_DELETED = (
    "Here, we examined the growth and isoprenol production in a Pseudomonas putida "
    "strain pre-tolerized (“PT”) background where its native isoprenol "
    "catabolism pathway is deleted, using glucose and acetate as carbon sources."
)
_Q_WGS = (
    "Trimmed and filtered reads were mapped to the reference genome (NCBI SRA "
    "NC_002947.4), and in turn, variant calls were identified using the software "
    "package Geneious (BioMatters LLC) with default parameters specified."
)
_Q_PROTEOME_TRIPLICATE = (
    "Each of these initial cell suspensions was used to inoculate three cultures of "
    "either M9 acetate, M9 glucose, or M9 glucose-acetate at an initial $\\mathsf { O "
    "D } _ { 6 0 0 } \\ : 0 . 1 5$ in $5 ~ \\mathrm { m L }$ cultures."
)
_Q_PROTEOME_TEMP = (
    "After inoculation, the cells were incubated at $3 0 ~ ^ { \\circ } \\mathsf { C }$ "
    "with agitation $( 2 0 0 \\ r { \\mathsf { p m } } )$ , and microbial growth was "
    "periodically tracked by the measurement of turbidity."
)
_Q_PROTEOME_HARVEST = (
    "Once cultures reached exponential growth $( \\mathsf { O D } _ { 6 0 0 }$ of "
    "0.6–0.8), the cultures were transferred to new $5 0 ~ \\mathrm { m L }$ "
    "conical tubes, and the cells harvested by centrifugation at $4 { , } 0 0 0 "
    "\\times { } g$ for $5 \\mathrm { \\ m i n }$ ."
)
_Q_TOP3 = (
    "The Top3 method, which is the average MS signal response of the three most "
    "intense tryptic peptides of each identified protein, was used to calculate the "
    "total intensity of proteins in the samples (41, 42)."
)
_Q_DIANN_DB = (
    "The databases used in the DIA-NN search (library-free mode) were $P .$ putida "
    "KT2440 latest Uniprot proteome FASTA sequences (generated in March 2024) and "
    "common proteomic contaminants."
)
_Q_RELATIVE_ABUNDANCE_RELEASED = (
    "Mean protein counts and relative protein abundances for each sample, as well as "
    "other summary statistics for the proteomics data set, are available in File S1."
)
_Q_RELATIVE_ABUNDANCE_USED = (
    "Relative protein abundances of proteins were used for dimensionality reduction."
)
_Q_REFERENCE_CONDITION = (
    "For both panels, protein levels in the glucose as a sole carbon source medium "
    "were used as the reference condition."
)
_Q_CARBON_DEFAULT = (
    "For all experiments, except when noted, acetate $5 0 ~ \\mathsf { m M }$ ${ \\it "
    "\\simeq } 0 . 3 \\%$ wt/vol) or glucose $1 \\%$ (wt/vol) was used as the single "
    "sole carbon source in M9 medium."
)
_Q_MIXTURE = (
    "or a mixture (model hydrolysate) containing glucose $2 \\%$ (wt/vol) and acetate "
    "$0 . 6 5 \\%$ (wt/vol) (bottom) in microtiter dish format."
)
_Q_PIY670 = (
    "Heterologous production of isoprenol was demonstrated using plasmid pIY670, which "
    "contains an optimized IPP-bypass pathway under the regulation of an "
    "arabinoseinduced promoter as well as the neo kanamycin selection marker (20)."
)
_Q_PIY670_PARTS = "pRK2-Kan-araC-PBAD-MvaSef-MvaEef-TrpoH-Ptrc1-O-MKmm-PMDHKQ-AphA"
_Q_PRODUCTION_RUN = (
    "of M9 minimal medium without carbon sources and back diluted to an initial ${ "
    "\\tt O D } _ { 6 0 0 }$ of 0.2 in $5 \\mathsf { m l }$ of fresh M9 medium with "
    "glucose $2 \\%$ (wt/vol) or glucose $2 \\%$ (wt/vol) with acetate $0 . 6 5 \\%$ "
    "(wt/vol) in triplicates. Arabinose was added at a final concentration of $0 . 2 "
    "\\%$ (wt/vol) in the medium to induce the production pathway, and kanamycin $5 0 "
    "~ \\mu \\mathrm { g / m L }$ was used for plasmid maintenance."
)
_Q_GCFID = (
    "To determine the concentration of isoprenol in the samples, the samples were "
    "harvested and prepared for an ethyl acetate extraction method as described "
    "previously (26, 44) for quantification with a gas chromatography-flame ionization "
    "detector (GC-FID, Agilent Technologies)"
)
_Q_TIMEPOINTS = (
    "Isoprenol production was monitored at the $2 4 \\ \\mathsf { h } ,$ $4 8 \\ "
    "\\mathrm { h } ,$ , and $7 2 \\mathrm { ~ h ~ }$ post-inoculation time points to "
    "measure cell growth via turbidity $( \\mathsf { O D } _ { 6 0 0 } )$ and "
    "isoprenol titer by harvesting $2 0 0 \\mu \\iota$ cell culture aliquots."
)
_Q_FIG3_SD = (
    "Data points shown represent three independent biological replicates, and the "
    "error bars indicate standard deviation from the mean."
)
_Q_PT_GLUCOSE_MG_PER_L = (
    "In M9 glucose medium, the base PT strain produced $2 8 3 \\pm 2 6 \\mathrm { { \\ "
    "m g / L } }$ isoprenol by the $4 8 \\mathrm { - h }$ time point"
)
_Q_PT_MIXED_NOT_DETECTABLE = (
    "In this production regime, the baseline PT strain failed to grow and therefore "
    "did not produce any detectable isoprenol."
)
_Q_MIXED_LOWER = (
    "When strains were cultivated in the glucose-acetate mixed carbon condition, we "
    "noted that the overall titers were lower than with the glucose alone condition"
)
_Q_DATA_AVAILABILITY = (
    "The P. putida sequencing data are available in NCBI SRA under Bioproject "
    "accession ID PRJNA1153078. The generated mass spectrometry proteomics data have "
    "been deposited to the ProteomeXchange Consortium via the PRIDE partner repository "
    "(74) with the data set identifier PXD055153."
)
_Q_TABLE_S2_CAPTION = (
    "Table S2. Maximum isoprenol titers (mM) detected for Σ-class and PT strains "
    "grown in different growth media. In each column, the total amount of carbon $( "
    '" C " )$ in the medium, calculated as the sum of concentrations of all '
    "available carbon sources, is expressed in parentheses. n.d. $=$ not determined. "
    "Glucose was always used at a concentration of $1 1 1 ~ \\mathsf { m M }$ ."
)
_Q_TABLE_S2_ROWS = (
    "<td>KT2440 &quot;PT&quot;</td><td>0</td><td>0</td><td>3.29</td><td>0.05</td></tr>"
    "<tr><td>∑1</td><td>0</td><td>0</td><td>4.52</td><td>1.44</td></tr>"
    "<tr><td>∑2</td><td>n.d.</td><td>n.d.</td><td>4.51</td><td>1.34</td></tr>"
    "<tr><td>∑3</td><td>0</td><td>0</td><td>2.41</td><td>1.05</td></tr>"
    "<tr><td>∑4</td><td>n.d.</td><td>n.d.</td><td>2.39</td><td>1.23</td></tr>"
    "<tr><td>∑5</td><td>n.d.</td><td>n.d.</td><td>4.26</td><td>1.88</td></tr>"
)
_Q_FIGS3_MEDIA = (
    "For each strain, two M9 media compositions using varied acetate and ammonium "
    "sulfate concentrations. Medium A contained 75mM acetate and $7 5 . 5 ~ \\mathsf { "
    "m M }$ ammonium sulfate. Medium C contained 100mM acetate and 1mM of ammonium "
    "sulfate. We were not able to detect isoprenol production by any of the strains "
    "under these conditions."
)
_Q_FIGS4_VARIANTS = (
    "Several lineage-specific single nucleotide polymorphisms (SNPs) were identified "
    "that could contribute to acetate tolerance. Two background mutations preexisting "
    "in the PT strain but absent in WT P. putida KT2440 included a $( + G G )$ "
    "insertion upstream of PP_4387 and a tandem repeat in PP_1458, affecting codon "
    "T180."
)
_Q_THOMPSON_DEL = (
    "<td>ΔPP_2675</td><td>Strain with complete internal in-frame deletion of "
    "PP_2675</td><td>This study</td>"
)

WT_STRAIN_SOURCE = _paper(
    WT_STRAIN,
    _Q_WT_ROW,
    page="Table 1: Strains used in this study",
    note="the wild type IS the pinned reference assembly, so a WT record carries no "
    "background and an empty genotype",
)
PT_GENOTYPE = _paper(
    "P. putida KT2440 ΔPP_2675 Δ14-PP_2676",
    _Q_PT_ROW,
    page="Table 1: Strains used in this study",
    note="Table 1 and the Methods state the same two designations; the Results explain "
    "that they name ONE deletion event (see PT_NOMENCLATURE)",
)
PT_GENOTYPE_METHODS = _paper(
    "P. putida KT2440 ΔPP_2675 Δ14-PP_2676", _Q_PT_STRAIN, page=_METHODS_STRAINS
)
PT_NOMENCLATURE = _paper(
    {"locus": PT_PARTIAL_DELETION, "overlaps": PT_FULL_DELETION, "frame_offset": 1},
    _Q_NOMENCLATURE,
    page="Results, 'The initial isoprenol catabolism-deficient P. putida strain shows "
    "arrested growth in media containing acetate'",
    note="PP_2676's predicted start codon lies inside PP_2675's CDS, which is why "
    "deleting PP_2675 renamed the strain; the source does not say whether the 14 of "
    "Δ14-PP_2676 counts base pairs or codons",
)
PT_FULL_DELETION_SOURCE = _thompson(
    "complete internal in-frame deletion of PP_2675",
    _Q_THOMPSON_DEL,
    page="Table 1: Strains used in this study",
    note="reference 30 of this paper, which is where PT's PP_2675 half is defined; the "
    "deferral is followed into the mirrored Thompson 2020 rather than guessed",
)
PT_PATHWAY_DELETED = _paper(
    False,
    _Q_ABSTRACT_DELETED,
    page="Abstract",
    note="the source calls PT's native isoprenol catabolism pathway deleted, which is "
    "what sets functional=False on both PT alleles; it never states PP_2676's "
    "functional status on its own",
)
PT_ICE_ACCESSION = _paper(
    "JBEI-147164",
    _Q_PT_ROW,
    page="Table 1: Strains used in this study",
    note="the JBEI Inventory of Composable Elements registry code of the PT stock",
)
WGS_METHOD = _paper(
    {"reference": "NC_002947.4", "caller": "Geneious"},
    _Q_WGS,
    page=_METHODS_STRAINS,
    note="NC_002947.4 is the RefSeq chromosome of pputida_KT2440_ASM756v2, whose "
    "GenBank replicon AE015451.2 carries identical sequence, so the released call "
    "coordinates are on the pinned assembly",
)
VARIANT_LEDGER_SOURCE = _si(
    "Data Set S2",
    _Q_FIGS4_VARIANTS,
    page="Figure S4 caption",
    note="the caption names two of PT's own calls; the complete table is Data Set S2, "
    "which this module types into preprocess/called_variants.json",
)
PROTEOME_N_REPLICATES = _paper(
    3,
    _Q_PROTEOME_TRIPLICATE,
    page=_METHODS_PROTEOMICS,
    note="three independent cultures per (strain, condition); the released "
    "Top_3pep_counts_rep_std is the sample SD over them, so SE = SD / sqrt(3)",
)
PROTEOME_UNCERTAINTY = _paper(
    "sample_sd",
    _Q_PROTEOME_TRIPLICATE,
    page=_METHODS_PROTEOMICS,
    note="the released _rep_std column is the SD of the three replicate measurements, "
    "not an SE; it is divided by sqrt(n) before being stored",
)
TOP3 = _paper(
    PROTEOME_MEASUREMENT_TYPE,
    _Q_TOP3,
    page=_METHODS_PROTEOMICS,
    note="what one stored abundance IS: the mean, over a sample's three replicates, of "
    "the Top3 signal",
)
PERCENT_OF_TOTAL = _paper(
    PERCENT_MEASUREMENT_TYPE,
    _Q_RELATIVE_ABUNDANCE_RELEASED,
    page="Results, 'Proteomics reveals different paths for acetate and glucose "
    "assimilation in tolerized and naive P. putida strains'",
    note="the paper's own name for the released percent-of-total column is 'relative "
    "protein abundances', and it names File S1 (the pinned Data Set S1) as where they "
    "are. It is the column the paper's OWN analysis runs on "
    f"('{_Q_RELATIVE_ABUNDANCE_USED}'), and it is not recoverable from the Top3 pair",
)
LOG10_PERCENT_OF_TOTAL = _data_s1(
    LOG10_PERCENT_MEASUREMENT_TYPE,
    "log10_%_abundance_rep_mean",
    page="Data Set S1, sheet 'Sheet1', column 12",
    note="the released header is the only statement of this scale: the paper names "
    "'other summary statistics for the proteomics data set' in File S1 and never "
    "describes this column in prose. Measured over all 34,600 cells, it is the MEAN of "
    "the three replicates' log10 percent rather than the log10 of the released percent",
)
DIANN_DATABASE = _paper(
    "P. putida KT2440 UniProt proteome + common proteomic contaminants",
    _Q_DIANN_DB,
    page=_METHODS_PROTEOMICS,
    note="the sourced reason non-host keys (human keratins, pig trypsin) appear in the "
    "released table and are dropped from the per-record abundance map",
)
PROTEOME_REFERENCE_SOURCE = _paper(
    PROTEOME_REFERENCE_CONDITION,
    _Q_REFERENCE_CONDITION,
    page="Fig. 4 caption",
    note="the reference every proteome record carries is the WILD TYPE in that "
    "condition, so the reference is one sample for the whole family",
)
PROTEOME_HARVEST = _paper(
    "mid-exponential (OD600 0.6-0.8)",
    _Q_PROTEOME_HARVEST,
    page=_METHODS_PROTEOMICS,
    note="the harvest is keyed to a growth state, not to a clock time, which is why "
    "Environment.duration_hours is None with a gap",
)
TEMPERATURE_C = _paper(30.0, _Q_PROTEOME_TEMP, page=_METHODS_PROTEOMICS)
ACETATE_MM = _paper(
    50.0,
    _Q_CARBON_DEFAULT,
    page=_METHODS_CULTURE,
    note="the sole-carbon default the Methods state for every experiment that does not "
    "note otherwise; the proteomics section notes no concentration",
)
GLUCOSE_PERCENT = _paper(
    1.0,
    _Q_CARBON_DEFAULT,
    page=_METHODS_CULTURE,
    note="the sole-carbon default, as above",
)
MIXED_GLUCOSE_PERCENT = _paper(
    2.0,
    _Q_MIXTURE,
    page="Fig. 1 caption",
    note="the only composition the paper ever gives for a glucose-acetate medium; the "
    "proteomics section names the medium without restating its concentrations",
)
MIXED_ACETATE_PERCENT = _paper(0.65, _Q_MIXTURE, page="Fig. 1 caption", note="as above")
AEROBICITY = _paper(
    "aerobic",
    _Q_PROTEOME_TEMP,
    page=_METHODS_PROTEOMICS,
    note="shaken tube cultures at 200 rpm, the standard aerobic configuration; the "
    "source never uses the word",
)
PIY670 = _paper(
    "pIY670",
    _Q_PIY670,
    page=_METHODS_STRAINS,
    note="the isoprenol production plasmid; a titer strain IS the chromosomal strain "
    "carrying it, which is why the pathway is a genotype perturbation and not part of "
    "the background",
)
PIY670_DESIQUEIRA_PARTS = _paper(
    _Q_PIY670_PARTS,
    _Q_PIY670_PARTS,
    page="Table 1: Strains used in this study",
    note="the OCR merged the pIY670 and pTE452 rows, so this part string sits in "
    "pTE452's genotype cell while pIY670's is empty; it agrees token for token, "
    "case-folded, with the Carruthers 2025 Supplementary Data 3 description of pIY670 "
    "except that the OCR dropped 'Sc' from PMDScHKQ, which is asserted at build time",
)
TITER_STATISTIC = _si(
    "maximum over the released sampling times",
    _Q_TABLE_S2_CAPTION,
    page="Table S2 caption",
    note="an upward-biased order statistic over the 24, 48 and 72 h samples, not a "
    "single-time-point measurement, which is why no duration describes it",
)
TITER_UNIT = _si(
    "mM",
    _Q_TABLE_S2_CAPTION,
    page="Table S2 caption",
    note="ConcentrationUnit has millimolar, so the released number is stored verbatim "
    "with no arithmetic applied to a source value",
)
TITER_TABLE = _si(
    {
        PT_STRAIN: dict(zip(TITER_COLUMNS, (0.0, 0.0, 3.29, 0.05), strict=True)),
        SIGMA_STRAINS[0]: dict(zip(TITER_COLUMNS, (0.0, 0.0, 4.52, 1.44), strict=True)),
        SIGMA_STRAINS[1]: dict(
            zip(TITER_COLUMNS, (None, None, 4.51, 1.34), strict=True)
        ),
        SIGMA_STRAINS[2]: dict(zip(TITER_COLUMNS, (0.0, 0.0, 2.41, 1.05), strict=True)),
        SIGMA_STRAINS[3]: dict(
            zip(TITER_COLUMNS, (None, None, 2.39, 1.23), strict=True)
        ),
        SIGMA_STRAINS[4]: dict(
            zip(TITER_COLUMNS, (None, None, 4.26, 1.88), strict=True)
        ),
    },
    _Q_TABLE_S2_ROWS,
    page="Table S2",
    note="every released cell, with the caption's 'n.d.' as None; the keys are the "
    "Data Set S1 spelling of the strains, while the OCR of this table writes the "
    "tolerized isolates with U+2211. PT is the only writable row",
)
TITER_ACETATE_MEDIA = _si(
    {
        "medium_A_acetate_mM": 75.0,
        "medium_A_ammonium_sulfate_mM": 75.5,
        "medium_C_acetate_mM": 100.0,
        "medium_C_ammonium_sulfate_mM": 1.0,
    },
    _Q_FIGS3_MEDIA,
    page="Figure S3 caption",
    note="both differ from the 2 g/L ammonium sulfate the served M9_NREL_DESIQUEIRA2025 "
    "states, so neither is that medium and MEDIA_LIBRARY holds no object for them; the "
    "two cells are left out and the needed media entries are proposed in the PR",
)
TITER_CROSS_SOURCE = _paper(
    {"titer_mg_per_l": 283.0, "sd_mg_per_l": 26.0, "hours": 48},
    _Q_PT_GLUCOSE_MG_PER_L,
    page="Results, 'Tolerization endowed cells with different isoprenol production "
    "capacities in minimal media'",
    note="the independent mg/L statement of the same PT glucose measurement; 283 mg/L "
    "over 86.13 g/mol is 3.286 mM against Table S2's 3.29 mM, which is the build-time "
    "cross-source check",
)
TITER_MIXED_DISAGREEMENT = _paper(
    "did not produce any detectable isoprenol",
    _Q_PT_MIXED_NOT_DETECTABLE,
    page="Results, 'Tolerization endowed cells with different isoprenol production "
    "capacities in minimal media'",
    note="Table S2 releases 0.05 mM (about 4.3 mg/L) for the same cell; the released "
    "number is stored and the disagreement is reported, never resolved by preference",
)
TITER_REFERENCE_SOURCE = _paper(
    TITER_REFERENCE_COLUMN,
    _Q_MIXED_LOWER,
    page="Results, 'Tolerization endowed cells with different isoprenol production "
    "capacities in minimal media'",
    note="the glucose-alone run is the single-carbon baseline the mixed feed is "
    "compared against, so it is the titer family's phenotype_reference",
)
TITER_N_REPLICATES = _paper(
    3,
    _Q_FIG3_SD,
    page="Fig. 3 caption",
    note="the production runs are biological triplicates; Table S2 releases the maximum "
    "only, with no uncertainty of its own",
)
TITER_GLUCOSE_PERCENT = _paper(2.0, _Q_PRODUCTION_RUN, page=_METHODS_PRODUCTION)
TITER_ACETATE_PERCENT = _paper(0.65, _Q_PRODUCTION_RUN, page=_METHODS_PRODUCTION)
INDUCER_PERCENT = _paper(0.2, _Q_PRODUCTION_RUN, page=_METHODS_PRODUCTION)
KANAMYCIN_UG_PER_ML = _paper(50.0, _Q_PRODUCTION_RUN, page=_METHODS_PRODUCTION)
QUANTIFICATION = _paper("GC-FID", _Q_GCFID, page=_METHODS_PRODUCTION)
SAMPLING_TIMES_HOURS = _paper(
    [24.0, 48.0, 72.0],
    _Q_TIMEPOINTS,
    page=_METHODS_PRODUCTION,
    note="the three times the maximum in Table S2 is taken over",
)
CULTURE_FORMAT = _paper(
    {"vessel": "test tube", "working_volume_ml": 5.0, "inoculum_od600": 0.2},
    _Q_PRODUCTION_RUN,
    page=_METHODS_PRODUCTION,
    note="vessel, working volume and inoculum are CultureEnvironment fields, and "
    "ProductTiterExperiment.environment is annotated CultureEnvironment, so "
    "titer_environment carries them on the record; the proteome family's slot is still "
    "Environment, which drops them on dump",
)
DATA_AVAILABILITY = _paper(
    {"sra_bioproject": SRA_BIOPROJECT, "pride": PRIDE_ACCESSION},
    _Q_DATA_AVAILABILITY,
    page="Data availability",
    note="neither deposit is consumed here: PRIDE holds raw DIA spectra and the "
    "BioProject holds raw reads, and no loader reads either",
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/desiqueiraAlternateRoutesAcetate2025``."""
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


def deposit_raw_mirror(
    *,
    proteome_path: str | Path,
    variants_path: str | Path,
    retrieved_at: str = SI_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror (Data Sets S1 and S2) and its ``manifest.json``.

    Idempotent by sha256: an existing mirror file whose digest matches is left alone and
    a differing one raises rather than being overwritten. Both files are verified BEFORE
    anything is written, so a refusal leaves no partial deposit. Both come from the PMC
    Article Datasets bucket, which is directly scriptable: the recorded retrieval re-runs
    as-is and reproduced both pinned digests on ``retrieved_at``.
    """
    root = raw_mirror_dir(data_root)
    deposits = (
        (proteome_path, PROTEOME_REL, PROTEOME_FILENAME, PROTEOME_SHA256),
        (variants_path, VARIANTS_REL, VARIANTS_FILENAME, VARIANTS_SHA256),
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
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=files,
        si_data_sources=[
            f"https://pmc-oa-opendata.s3.amazonaws.com/{pmc_cloud_key(PROTEOME_FILENAME)}",
            f"https://pmc-oa-opendata.s3.amazonaws.com/{pmc_cloud_key(VARIANTS_FILENAME)}",
            "https://pmc-oa-opendata.s3.amazonaws.com/"
            f"{pmc_cloud_key('aem.02123-24-s0003.pdf')}",
            f"https://www.ncbi.nlm.nih.gov/bioproject/{SRA_BIOPROJECT}",
            f"https://www.ebi.ac.uk/pride/archive/projects/{PRIDE_ACCESSION}",
        ],
        si_expected=[
            "Data Set S1 (aem.02123-24-s0001.xlsx) -- deposited; the per-(protein, "
            "strain, condition) Top3 abundance matrix the proteome loader consumes",
            "Data Set S2 (aem.02123-24-s0002.xlsx) -- deposited; the Geneious variant "
            "calls, consumed only to type what cannot be written into "
            "preprocess/called_variants.json",
            "Supplemental material (aem.02123-24-s0003.pdf) -- NOT deposited here: it "
            "is a text PDF, already mirrored and OCR'd in torchcell-library as "
            f"{SI3_MD}, and the titer values are hardcoded sourced values quoting those "
            "bytes rather than parsed at build time",
            f"NCBI SRA BioProject {SRA_BIOPROJECT} -- the raw resequencing reads. NOT "
            "deposited: no loader consumes raw reads, and the called variants are the "
            "released derivative",
            f"PRIDE {PRIDE_ACCESSION} -- the raw DIA mass-spectrometry files. NOT "
            "deposited: no loader consumes raw spectra, and Data Set S1 is the released "
            "processed matrix",
            "the per-time-point isoprenol titers plotted in Fig. 3B and 3C were never "
            "released as a table; Table S2's maximum is the only numeric release",
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
def _sheet_rows(path: str) -> tuple[tuple[Any, ...], list[tuple[Any, ...]]]:
    """``(header, rows)`` of a single-sheet released workbook, read-only."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        stream = book.worksheets[0].iter_rows(values_only=True)
        header = next(stream)
        return header, [row for row in stream if any(c is not None for c in row)]
    finally:
        book.close()


class ProteomeNormalization(BaseModel):
    """One released normalization of the same Top3 proteome, and what it IS.

    Data Set S1 releases the same five writable samples on three scales, each as its own
    mean + SD pair. ``measurement_type`` is what a record stores, so each normalization
    is its own dataset class: the shared ``verify_protein_dataset`` asserts one
    ``measurement_type`` per dataset ("no silent cross-assay mixing").
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    measurement_type: str
    mean_column: str
    sd_column: str
    #: ``data/torchcell/<root_slug>``, the dev-tree root of the class that stores it.
    root_slug: str
    #: Whether this scale can be RE-DERIVED from the Top3 columns the first build
    #: already stored. Measured, not assumed; see the module docstring.
    recoverable_from_top3: bool
    note: str


#: The three released normalizations, in released column order.
PROTEOME_NORMALIZATIONS: tuple[ProteomeNormalization, ...] = (
    ProteomeNormalization(
        measurement_type=PROTEOME_MEASUREMENT_TYPE,
        mean_column="Top_3pep_counts_rep_mean",
        sd_column="Top_3pep_counts_rep_std",
        root_slug="proteome_desiqueira2025",
        recoverable_from_top3=True,
        note="the Top3 peptide signal itself, the scale the first build stored",
    ),
    ProteomeNormalization(
        measurement_type=PERCENT_MEASUREMENT_TYPE,
        mean_column="%_of protein_abundance_Top3_rep_mean",
        sd_column="%_of protein_abundance_Top3-rep_std",
        root_slug="proteome_percent_desiqueira2025",
        recoverable_from_top3=False,
        note="percent of a sample's total abundance, averaged over the three "
        "replicates. NOT the released percent of the released mean: measured over all "
        "34,600 cells, zero agree exactly and the per-cell ratio runs 0.914506 to "
        "1.083554, which is a replicate-wise mean of per-replicate percentages",
    ),
    ProteomeNormalization(
        measurement_type=LOG10_PERCENT_MEASUREMENT_TYPE,
        mean_column="log10_%_abundance_rep_mean",
        sd_column="log10_%_abundance_rep_std",
        root_slug="proteome_log10_percent_desiqueira2025",
        recoverable_from_top3=False,
        note="the mean of the three replicates' log10 percent. NOT the log10 of the "
        "released percent: measured over all 34,600 cells, 33,914 disagree, 686 agree "
        "to 5e-5 and ZERO exceed it, which is Jensen's inequality for a mean of logs",
    ),
)


class ProteomeRow(BaseModel):
    """One released (protein, strain, condition) cell of Data Set S1, all 15 columns.

    Every asserted column is read. The six beyond the Top3 pair are the two further
    normalizations plus the two derived columns the build uses as oracles
    (:func:`assert_percent_cv_is_derived`, :func:`assert_sem_is_constant_per_protein`),
    so no column is asserted and then parsed past.
    """

    accession: str
    entry_name: str
    protein: str
    description: str
    strain: str
    condition: str
    sample: str
    top3_mean: float
    top3_sd: float
    pct_mean: float
    pct_sd: float
    log10_pct_mean: float
    log10_pct_sd: float
    cv_percent: float
    constant_sem: float

    def normalized(self, normalization: ProteomeNormalization) -> tuple[float, float]:
        """``(mean, sd)`` of one released normalization for this cell."""
        return {
            PROTEOME_MEASUREMENT_TYPE: (self.top3_mean, self.top3_sd),
            PERCENT_MEASUREMENT_TYPE: (self.pct_mean, self.pct_sd),
            LOG10_PERCENT_MEASUREMENT_TYPE: (self.log10_pct_mean, self.log10_pct_sd),
        }[normalization.measurement_type]


PROTEOME_HEADER: tuple[str, ...] = (
    "Protein.Group",
    "Protein.Names",
    "Protein",
    "Protein.Description",
    "Strain",
    "Condition",
    "Sample",
    "Top_3pep_counts_rep_mean",
    "Top_3pep_counts_rep_std",
    "%_of protein_abundance_Top3_rep_mean",
    "%_of protein_abundance_Top3-rep_std",
    "log10_%_abundance_rep_mean",
    "log10_%_abundance_rep_std",
    "CV%_of_%_protein_abundance",
    "%_of protein_abundance_Top3_rep_mean_sem",
)
_NORMALIZATION_COLUMNS = {n.mean_column for n in PROTEOME_NORMALIZATIONS} | {
    n.sd_column for n in PROTEOME_NORMALIZATIONS
}
if _NORMALIZATION_COLUMNS - set(PROTEOME_HEADER):
    raise RuntimeError(
        "a normalization names a column Data Set S1's asserted header does not carry"
    )
if [n.measurement_type for n in PROTEOME_NORMALIZATIONS] != [
    str(TOP3.value),
    str(PERCENT_OF_TOTAL.value),
    str(LOG10_PERCENT_OF_TOTAL.value),
]:
    raise RuntimeError(
        "a normalization's measurement_type is no longer the one its SourcedValue binds"
    )
#: How far the released CV may sit from ``100 * pct_sd / pct_mean``. Measured on the
#: pinned workbook: zero of 34,600 cells disagree beyond floating-point noise.
_CV_TOL = 1e-6
#: Proteins whose constant per-protein SEM column is NOT single-valued. Measured on the
#: pinned workbook: exactly one of 1,729.
SEM_MULTIVALUED_PROTEINS = 1


def read_proteome_rows(path: str) -> list[ProteomeRow]:
    """Every cell of the released Top3 abundance matrix, typed."""
    header, rows = _sheet_rows(path)
    if tuple(str(c) for c in header[: len(PROTEOME_HEADER)]) != PROTEOME_HEADER:
        raise RuntimeError(f"Data Set S1 header changed: {header!r}")
    return [
        ProteomeRow(
            accession=str(row[0]).strip(),
            entry_name=str(row[1]).strip(),
            protein=str(row[2]).strip(),
            description=str(row[3]).strip(),
            strain=str(row[4]).strip(),
            condition=str(row[5]).strip(),
            sample=str(row[6]).strip(),
            top3_mean=float(row[7]),
            top3_sd=float(row[8]),
            pct_mean=float(row[9]),
            pct_sd=float(row[10]),
            log10_pct_mean=float(row[11]),
            log10_pct_sd=float(row[12]),
            cv_percent=float(row[13]),
            constant_sem=float(row[14]),
        )
        for row in rows
    ]


def assert_percent_cv_is_derived(rows: Sequence[ProteomeRow]) -> None:
    """Oracle: the released CV column IS ``100 * pct_sd / pct_mean``.

    Which is why the CV is not a phenotype: it carries no information the stored percent
    pair does not. Measured on the pinned workbook: zero of 34,600 cells disagree.
    """
    for row in rows:
        if row.pct_mean == 0.0:
            continue
        derived = 100.0 * row.pct_sd / row.pct_mean
        if abs(derived - row.cv_percent) > _CV_TOL * max(1.0, abs(row.cv_percent)):
            raise RuntimeError(
                f"{row.sample}/{row.protein}: 100 * pct_sd / pct_mean is {derived} and "
                f"the released CV column reads {row.cv_percent}; the CV is not derived "
                "from the pair this loader stores"
            )


def assert_sem_is_constant_per_protein(rows: Sequence[ProteomeRow]) -> None:
    """Oracle: the released SEM column is one value per PROTEIN, not per sample.

    Which is why it is not a phenotype: a per-sample record cannot carry a number that
    does not vary with the sample. Measured on the pinned workbook: 1,728 of 1,729
    proteins carry a single value, so exactly :data:`SEM_MULTIVALUED_PROTEINS` does not.
    """
    per_protein: dict[str, set[float]] = defaultdict(set)
    for row in rows:
        per_protein[row.protein].add(row.constant_sem)
    multivalued = sorted(key for key, seen in per_protein.items() if len(seen) > 1)
    if len(multivalued) != SEM_MULTIVALUED_PROTEINS:
        raise RuntimeError(
            f"{len(multivalued)} of {len(per_protein)} proteins carry more than one SEM "
            f"value ({multivalued[:3]}), not the {SEM_MULTIVALUED_PROTEINS} the pinned "
            "workbook holds; the column is no longer the constant per-protein SEM"
        )


class CalledVariant(BaseModel):
    """One Geneious call of Data Set S2, and which perturbation leaf it becomes.

    Every field is the released cell verbatim. ``encoding`` names the leaf this row is
    written as, measured on the row rather than asserted; before the #731 leaves landed
    this field was a list of the reasons no class could hold the call, and the ledger it
    is written into is the same ``preprocess/called_variants.json``.
    """

    strain: str
    track: str
    #: The ``Sequence Name`` cell: the replicon the coordinates are on.
    reference_sequence: str
    position_start: int
    position_end: int
    polymorphism_type: str
    #: The fused ``Change`` cell, absent for the 21 insertion rows, which release the
    #: two alleles in their own columns instead.
    change: str | None
    #: The ``Sequence`` (reference) and ``Name`` (alternate) cells, verbatim. An
    #: insertion's reference cell reads ``--``.
    reference_allele: str | None
    alternate_allele: str | None
    #: Verbatim released cells. Both are released as strings and 16 of the frequency
    #: cells write a percent RANGE ("95.1% -> 97.6%"), so neither is coerced to a float.
    variant_frequency: str
    coverage: str
    protein_effect: str | None
    amino_acid_change: str | None
    codon_change: str | None
    cds_codon_number: int | None
    genbank_locus_tag: str | None
    refseq_locus_tag: str | None
    gene_symbol: str | None
    product: str | None
    encoding: str


#: The typed encodings :func:`read_variant_calls` assigns, one per row.
ENCODING_IN_LOCUS = "bacterial_sequence_variant_in_locus"
ENCODING_INTERGENIC = "bacterial_site_variant_intergenic"
ENCODING_LOCUS_NOT_IN_ASSEMBLY = "bacterial_site_variant_locus_not_in_assembly"
ENCODING_RESTATES_DESIGNED_DELETION = "restates_the_designed_pt_deletion"

VARIANT_HEADER: tuple[str, ...] = (
    "Protein Effect",
    "old_locus_tag",
    "Name",
    "Type",
    "Sequence",
    "Minimum",
    "Maximum",
)
#: Every further column :func:`read_variant_calls` reads by name. Asserted present, so a
#: renamed column stops the build instead of silently reading ``None``.
VARIANT_COLUMNS_READ: tuple[str, ...] = (
    "Sequence Name",
    "Strain",
    "Track Name",
    "Polymorphism Type",
    "Change",
    "Variant Frequency",
    "Coverage",
    "Amino Acid Change",
    "Codon Change",
    "CDS Codon Number",
    "locus_tag",
    "gene",
    "product",
)


def _cell(value: Any) -> str | None:
    """A released cell as a stripped string, with ``None`` and ``'None'`` as absent."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text in ("", "None") else text


def read_variant_calls(path: str) -> list[CalledVariant]:
    """Every Data Set S2 call, typed, each carrying the leaf it is written as.

    The encoding is decided by the row's own cells, in this order, and there is no
    default branch: a row that matches none of them stops the build.

    - It restates PT's designed ``PP_2675`` deletion (``ENCODING_RESTATES_DESIGNED
      _DELETION``). Measured: all five sequenced clones carry one ``Deletion`` call at
      3,063,718..3,064,173 on ``PP_2675``, 456 bp of a ``cytochrome c-550 PedF`` CDS,
      which IS the markerless in-frame deletion the strain was built with. It is already
      a ``BacterialDeletionPerturbation`` on every record, and absence has one encoding,
      so the call is dropped and counted rather than written twice.
    - ``old_locus_tag`` names a GenBank locus -> in-locus sequence variant.
    - no locus at all -> a site variant, ``site_kind=intergenic``.
    - a RefSeq ``locus_tag`` only -> a site variant,
      ``site_kind=locus_not_in_assembly``. Measured: the two tags involved,
      ``PP_RS21780`` (9 rows) and ``PP_RS19075`` (1), both return ``retired``, "not
      found in GCA_000007565.2_ASM756v2; retained as given", from
      ``resolve_gene_name`` on the pinned assembly, so there is no GenBank tag to key
      them to and inventing one is refused.
    """
    header, rows = _sheet_rows(path)
    columns = [str(c) if c is not None else "" for c in header]
    if tuple(columns[: len(VARIANT_HEADER)]) != VARIANT_HEADER:
        raise RuntimeError(f"Data Set S2 header changed: {header!r}")
    missing = [name for name in VARIANT_COLUMNS_READ if name not in columns]
    if missing:
        raise RuntimeError(f"Data Set S2 no longer releases {missing}")
    records = [dict(zip(columns, row, strict=False)) for row in rows]

    calls: list[CalledVariant] = []
    for row in records:
        strain = str(row["Strain"]).strip()
        replicon = str(row["Sequence Name"]).strip()
        if replicon != VARIANT_REPLICON:
            raise RuntimeError(
                f"{strain}: Data Set S2 names replicon {replicon!r}, not "
                f"{VARIANT_REPLICON!r}; the coordinates are on another sequence"
            )
        tag = _cell(row.get("old_locus_tag"))
        refseq = _cell(row.get("locus_tag"))
        polymorphism = str(row["Polymorphism Type"]).strip()
        if polymorphism not in VARIANT_TYPE_OF_POLYMORPHISM:
            raise RuntimeError(
                f"{strain}: unmapped Polymorphism Type {polymorphism!r}; the mapped "
                f"spellings are {sorted(VARIANT_TYPE_OF_POLYMORPHISM)}"
            )
        if tag == PT_FULL_DELETION:
            encoding = ENCODING_RESTATES_DESIGNED_DELETION
        elif tag is not None:
            encoding = ENCODING_IN_LOCUS
        elif refseq is not None:
            encoding = ENCODING_LOCUS_NOT_IN_ASSEMBLY
        else:
            encoding = ENCODING_INTERGENIC
        codon = _cell(row.get("CDS Codon Number"))
        calls.append(
            CalledVariant(
                strain=strain,
                track=str(row["Track Name"]).strip(),
                reference_sequence=replicon,
                position_start=int(row["Minimum"]),
                position_end=int(row["Maximum"]),
                polymorphism_type=polymorphism,
                change=_cell(row.get("Change")),
                reference_allele=_cell(row.get("Sequence")),
                alternate_allele=_cell(row.get("Name")),
                variant_frequency=str(row["Variant Frequency"]).strip(),
                coverage=str(row["Coverage"]).strip(),
                protein_effect=_cell(row.get("Protein Effect")),
                amino_acid_change=_cell(row.get("Amino Acid Change")),
                codon_change=_cell(row.get("Codon Change")),
                cds_codon_number=int(codon) if codon is not None else None,
                genbank_locus_tag=tag,
                refseq_locus_tag=refseq,
                gene_symbol=_cell(row.get("gene")),
                product=_cell(row.get("product")),
                encoding=encoding,
            )
        )
    if not calls:
        raise RuntimeError("Data Set S2 released no variant calls")
    return calls


def released_frequency(
    statement: str,
) -> tuple[float | None, VariantFrequencyBasis | None]:
    """The numeric frequency a released cell states, or ``(None, None)`` for a range.

    16 of the 173 cells write a percent range (``95.1% -> 97.6%``), which no single
    float holds, so those calls keep the verbatim cell and carry no number. The other
    157 are bare fractions, checked to lie in (0, 1] here rather than trusted: a cell
    above 1 would mean the column had switched to percents and the basis would be wrong.
    """
    if VARIANT_FREQUENCY_RANGE in statement:
        return None, None
    value = float(statement)
    if not 0.0 < value <= 1.0:
        raise RuntimeError(
            f"variant frequency {statement!r} is outside (0, 1]; the column is no "
            "longer a fraction"
        )
    return value, VariantFrequencyBasis.fraction


def variant_call(row: CalledVariant) -> BacterialVariantCall:
    """The typed call of one released row, with every cell it states kept verbatim."""
    frequency, basis = released_frequency(row.variant_frequency)
    return BacterialVariantCall(
        variant_type=VARIANT_TYPE_OF_POLYMORPHISM[row.polymorphism_type],
        type_statement=row.polymorphism_type,
        reference_sequence=row.reference_sequence,
        position_start=row.position_start,
        position_end=row.position_end,
        sequence_change=row.change,
        reference_allele=row.reference_allele,
        alternate_allele=row.alternate_allele,
        annotation=row.protein_effect,
        amino_acid_change=row.amino_acid_change,
        codon_change=row.codon_change,
        codon_number=row.cds_codon_number,
        call_mode=VariantCallMode.clone,
        frequency_statement=row.variant_frequency,
        frequency=frequency,
        frequency_basis=basis,
        caller=str(dict(WGS_METHOD.value)["caller"]),
    )


def called_variant_perturbations(
    calls: Sequence[CalledVariant], strain: str
) -> list[Any]:
    """One perturbation per written call of ``strain``, ordered by position.

    The order is explicit because ``Genotype.sort_perturbations`` sorts by systematic
    name, perturbation type and perturbed name, and two calls in one locus tie on all
    three (Sigma1 carries ``PP_1656`` twice and ``PP_3827`` three times); Python's sort
    is stable, so the input order decides and must therefore be deterministic.

    A ``restates_the_designed_pt_deletion`` row is skipped here and counted by the
    caller: it is the same lesion ``pt_perturbation`` already writes.
    """
    rows = sorted(
        (row for row in calls if row.strain == strain),
        key=lambda row: (row.position_start, row.position_end, row.polymorphism_type),
    )
    perturbations: list[Any] = []
    for row in rows:
        if row.encoding == ENCODING_RESTATES_DESIGNED_DELETION:
            continue
        call = variant_call(row)
        if row.encoding == ENCODING_IN_LOCUS:
            assert row.genbank_locus_tag is not None
            perturbations.append(
                BacterialSequenceVariantPerturbation(
                    systematic_gene_name=row.genbank_locus_tag,
                    perturbed_gene_name=row.gene_symbol or row.genbank_locus_tag,
                    gene_namespace=KT2440_NAMESPACE,
                    call=call,
                )
            )
            continue
        site_kind = (
            VariantSiteKind.locus_not_in_assembly
            if row.encoding == ENCODING_LOCUS_NOT_IN_ASSEMBLY
            else VariantSiteKind.intergenic
        )
        released_locus = (
            row.refseq_locus_tag
            if site_kind is VariantSiteKind.locus_not_in_assembly
            else None
        )
        site_id = BacterialSiteVariantPerturbation.site_id(call)
        perturbations.append(
            BacterialSiteVariantPerturbation(
                systematic_gene_name=site_id,
                perturbed_gene_name=released_locus or site_id,
                gene_namespace=KT2440_NAMESPACE,
                call=call,
                site_kind=site_kind,
                released_locus_statement=released_locus,
            )
        )
    return perturbations


class VariantLedger(BaseModel):
    """The counted summary of :func:`read_variant_calls`, written beside the build."""

    n_calls: int
    n_distinct_sites: int
    n_sites_in_more_than_one_strain: int
    calls_per_strain: dict[str, int]
    calls_with_genbank_locus: int
    calls_with_refseq_locus_only: int
    calls_with_no_locus: int
    encodings: dict[str, int]
    perturbations_per_strain: dict[str, int]
    loci_claimed_twice: dict[str, list[str]]
    unsequenced_strains: list[str]
    calls: list[CalledVariant]

    def check(self) -> None:
        """Every call is encoded, and a written call is a perturbation of its strain."""
        if sum(self.encodings.values()) != self.n_calls:
            raise RuntimeError(
                f"{sum(self.encodings.values())} encodings over {self.n_calls} calls"
            )
        restated = self.encodings.get(ENCODING_RESTATES_DESIGNED_DELETION, 0)
        written = sum(self.perturbations_per_strain.values())
        if written != self.n_calls - restated:
            raise RuntimeError(
                f"{written} perturbations over {self.n_calls} calls less "
                f"{restated} restatements of the designed deletion"
            )


def variant_ledger(calls: Sequence[CalledVariant]) -> VariantLedger:
    """Summarize the called variants, including which strains were never sequenced."""
    sites = Counter((call.position_start, call.change) for call in calls)
    per_strain_locus: Counter[tuple[str, str]] = Counter(
        (call.strain, call.genbank_locus_tag)
        for call in calls
        if call.genbank_locus_tag is not None
    )
    per_locus: dict[str, list[str]] = defaultdict(list)
    for (strain, tag), n in per_strain_locus.items():
        if n > 1:
            per_locus[strain].append(f"{tag} x{n}")
    sequenced = {call.strain for call in calls}
    encodings = Counter(call.encoding for call in calls)
    written = {
        strain: len(called_variant_perturbations(calls, strain))
        for strain in sorted(sequenced)
    }
    ledger = VariantLedger(
        n_calls=len(calls),
        n_distinct_sites=len(sites),
        n_sites_in_more_than_one_strain=sum(1 for n in sites.values() if n > 1),
        calls_per_strain=dict(sorted(Counter(c.strain for c in calls).items())),
        calls_with_genbank_locus=sum(
            1 for c in calls if c.genbank_locus_tag is not None
        ),
        calls_with_refseq_locus_only=sum(
            1
            for c in calls
            if c.genbank_locus_tag is None and c.refseq_locus_tag is not None
        ),
        calls_with_no_locus=sum(
            1
            for c in calls
            if c.genbank_locus_tag is None and c.refseq_locus_tag is None
        ),
        encodings=dict(sorted(encodings.items())),
        perturbations_per_strain=written,
        loci_claimed_twice={k: sorted(v) for k, v in sorted(per_locus.items())},
        unsequenced_strains=sorted(set(SIGMA_STRAINS) - sequenced),
        calls=list(calls),
    )
    ledger.check()
    if sorted(sequenced) != sorted((PT_STRAIN, *SEQUENCED_SIGMA_STRAINS)):
        raise RuntimeError(
            f"Data Set S2 releases calls for {sorted(sequenced)}, not the "
            f"{sorted((PT_STRAIN, *SEQUENCED_SIGMA_STRAINS))} this module writes"
        )
    return ledger


# --------------------------------------------------------------------------- #
# Genotype and environment builders
# --------------------------------------------------------------------------- #
def publication() -> Publication:
    """This paper, by DOI. The mirror records no PubMed id, so none is invented."""
    return Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")


def pt_background() -> BacterialStrainBackground:
    """PT: KT2440 carrying the one deletion event its two designations name.

    ``PP_2675`` is typed ``full_deletion`` on the Thompson 2020 statement the paper
    defers to; ``PP_2676`` is typed ``partial_deletion`` because the source writes
    ``Delta14-PP_2676`` and never a whole-ORF removal, with a ``ProvenanceGap`` on
    ``deleted_span`` since neither the unit of the 14 nor any coordinate is given.
    """
    span_gap = ProvenanceGap(
        field="deleted_span",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note=(
            "the source writes 'Δ14-PP_2676' without saying whether 14 counts "
            "base pairs or codons and without coordinates on the pinned assembly, and "
            "the deferral target (Thompson 2020) names only the PP_2675 half"
        ),
    )
    return BacterialStrainBackground(
        name=PT_STRAIN,
        reference_strain=WT_STRAIN,
        assembly_set=KT2440_ASSEMBLY_SET,
        parents=[WT_STRAIN],
        construction=(
            "one markerless in-frame deletion event in KT2440, named twice: it removes "
            f"{PT_FULL_DELETION} completely and 14 units of the overlapping "
            f"{PT_PARTIAL_DELETION} ORF, whose predicted start codon lies inside "
            f"{PT_FULL_DELETION}'s CDS"
        ),
        genotype_statement=str(PT_GENOTYPE.value),
        alleles=[
            BacterialBackgroundAllele(
                systematic_gene_name=PT_FULL_DELETION,
                gene_namespace=KT2440_NAMESPACE,
                gene_name=PT_FULL_DELETION,
                allele_name=f"Δ{PT_FULL_DELETION}",
                edit=AlleleEdit.full_deletion,
                functional=False,
                provenance=[PT_GENOTYPE, PT_FULL_DELETION_SOURCE, PT_PATHWAY_DELETED],
            ),
            BacterialBackgroundAllele(
                systematic_gene_name=PT_PARTIAL_DELETION,
                gene_namespace=KT2440_NAMESPACE,
                gene_name=PT_PARTIAL_DELETION,
                allele_name=f"Δ14-{PT_PARTIAL_DELETION}",
                edit=AlleleEdit.partial_deletion,
                functional=False,
                deleted_span=None,
                provenance=[PT_GENOTYPE, PT_NOMENCLATURE, PT_PATHWAY_DELETED],
                provenance_gaps=[span_gap],
            ),
        ],
        provenance=[PT_GENOTYPE, PT_GENOTYPE_METHODS, PT_NOMENCLATURE],
    )


def strain_reference(strain: str) -> AssemblyReferenceGenome:
    """The assembly-pinned reference one strain's records are written against.

    A sequenced tolerized isolate takes PT's reference, not one of its own: PT is the
    base strain every isolate was evolved from, so PT's deletion event is the genomic
    content they share and hold CONSTANT, while the variants Geneious called for each
    isolate are what varies and therefore belong in ``Genotype``. That is the division
    ``BacterialStrainBackground`` is documented for.
    """
    if strain == PROTEOME_REFERENCE_STRAIN:
        return assembly_reference(WT_STRAIN)
    if strain == PT_STRAIN or strain in SEQUENCED_SIGMA_STRAINS:
        return assembly_reference(WT_STRAIN, background=pt_background())
    raise RuntimeError(
        f"{strain!r} is not a writable strain of this paper; "
        f"{UNSEQUENCED_SIGMA} was never sequenced, so its genotype is unknown"
    )


def pt_perturbation() -> BacterialDeletionPerturbation:
    """PT's chromosomal edit as the ML-facing perturbation: the PP_2675 deletion."""
    return BacterialDeletionPerturbation(
        systematic_gene_name=PT_FULL_DELETION,
        perturbed_gene_name=PT_FULL_DELETION,
        gene_namespace=KT2440_NAMESPACE,
        construction=StrainConstruction(
            strain_accession=str(PT_ICE_ACCESSION.value), lab="JBEI"
        ),
    )


def strain_genotype(
    strain: str, *, with_pathway: bool, calls: Sequence[CalledVariant]
) -> Genotype:
    """The genotype of one writable strain, with or without the pIY670 pathway.

    PT and every sequenced isolate carry PT's ``PP_2675`` deletion plus their OWN called
    variants, so two isolates differ by the calls they do not share and
    ``Genotype.__eq__``, which compares the perturbation SET, keeps them apart. WT
    carries nothing: Data Set S2 releases no calls for it, because the calls are against
    its own assembly.
    """
    perturbations: list[Any] = []
    if strain == PT_STRAIN or strain in SEQUENCED_SIGMA_STRAINS:
        perturbations.append(pt_perturbation())
        perturbations.extend(called_variant_perturbations(calls, strain))
    elif strain != PROTEOME_REFERENCE_STRAIN:
        raise RuntimeError(f"{strain!r} is not a writable strain of this paper")
    if with_pathway:
        perturbations.extend(pathway_perturbations())
    return Genotype(perturbations=perturbations)


def _piy670_tokens(description: str) -> list[str]:
    """The case-folded hyphen-separated part tokens of a pIY670 description."""
    return [token.casefold() for token in description.split("-")]


def assert_piy670_matches_carruthers() -> None:
    """This paper's OCR part string IS the Carruthers workbook's pIY670 description.

    Measured on the pinned bytes: the two token lists agree case-folded except at the
    PMD token, where this paper's OCR dropped ``Sc``. That is what licenses reusing the
    Carruthers-sourced ``source_organism`` values and the workbook spellings of the five
    part names, which this paper never states and whose case its OCR did not preserve.
    """
    mine = _piy670_tokens(str(PIY670_DESIQUEIRA_PARTS.value))
    theirs = _piy670_tokens(str(PIY670_PARTS.value))
    if len(mine) != len(theirs):
        raise RuntimeError(
            f"pIY670 part counts differ: {len(mine)} here, {len(theirs)} in the "
            "Carruthers description"
        )
    mismatched = [(a, b) for a, b in zip(mine, theirs, strict=True) if a != b]
    if mismatched != [("pmdhkq", "pmdschkq")]:
        raise RuntimeError(
            "the pIY670 descriptions no longer agree on every token but the known OCR "
            f"loss of 'Sc': {mismatched}"
        )


def pathway_perturbations() -> list[HeterologousPathwayPerturbation]:
    """The five heterologous pIY670 genes carried by a titer strain.

    The identifiers are the Carruthers workbook spellings rather than this paper's OCR
    spellings, so one physical gene is one node across both datasets; the agreement that
    licenses it is asserted by :func:`assert_piy670_matches_carruthers`.
    """
    assert_piy670_matches_carruthers()
    return [
        HeterologousPathwayPerturbation(
            systematic_gene_name=str(gene["token"]),
            perturbed_gene_name=str(gene["symbol"]),
            gene_namespace=KT2440_NAMESPACE,
            pathway_name=str(PATHWAY_SOURCE.value),
            source_organism=str(gene["organism"]),
            is_heterologous=True,
            localization="episomal_plasmid",
            construct_name=str(PIY670.value),
            variant=gene["variant"],
            promoter_name=gene["promoter"],
            copy_number=1.0,
        )
        for gene in PATHWAY_GENES
    ]


def _carbon(
    name: str, value: float, unit: ConcentrationUnit
) -> EnvironmentPhysicalPerturbation:
    """One carbon source of the base M9, as the typed physical factor media.py names.

    ``M9_NREL_DESIQUEIRA2025`` carries no carbon source, and its own provenance note
    says the loader carries it here, so the carbon regime is the environment's variable
    rather than a second medium object per condition.
    """
    return EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.carbon_source,
        magnitude=Concentration(value=value, unit=unit),
        agent=resolved_compound(name),
    )


def proteome_environment(condition: str) -> Environment:
    """The M9 environment of one proteomics condition.

    ``duration_hours`` is ``None`` with a gap on purpose: the cultures were harvested at
    a growth STATE, not at a clock time.
    """
    carbon: list[EnvironmentPerturbationType]
    if condition == CONDITION_ACETATE:
        carbon = [
            _carbon("acetate", float(ACETATE_MM.value), ConcentrationUnit.millimolar)
        ]
    elif condition == CONDITION_GLUCOSE:
        carbon = [
            _carbon(
                "D-glucose", float(GLUCOSE_PERCENT.value), ConcentrationUnit.percent_w_v
            )
        ]
    elif condition == CONDITION_MIXED:
        carbon = [
            _carbon(
                "D-glucose",
                float(MIXED_GLUCOSE_PERCENT.value),
                ConcentrationUnit.percent_w_v,
            ),
            _carbon(
                "acetate",
                float(MIXED_ACETATE_PERCENT.value),
                ConcentrationUnit.percent_w_v,
            ),
        ]
    else:
        raise RuntimeError(f"{condition!r} is not a released proteomics condition")
    return Environment(
        media=M9_NREL_DESIQUEIRA2025,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        perturbations=carbon,
        aerobicity=str(AEROBICITY.value),
        duration_hours=None,
        provenance_gaps=[
            ProvenanceGap(
                field="duration_hours",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=(
                    "the proteomics cultures were harvested at mid-exponential growth "
                    f"({PROTEOME_HARVEST.value}), so no clock time describes them"
                ),
            )
        ],
    )


def titer_environment(column: str) -> CultureEnvironment:
    """The M9 production environment of one loaded Table S2 column.

    A ``CultureEnvironment``, since ``ProductTiterExperiment.environment`` is annotated
    as one: a titer is read with its vessel, so :data:`CULTURE_FORMAT`'s test tube,
    5 mL working volume and inoculum OD survive the dump instead of being recorded
    beside the record. ``endpoint`` is deliberately left unset: Table S2 reports the
    MAXIMUM over the sampling times, so no endpoint rule describes the stored value,
    which is the same reason ``duration_hours`` is a typed gap below.
    """
    if column not in TITER_COLUMNS_LOADED:
        raise RuntimeError(
            f"{column!r} is not a Table S2 column this loader writes; the two acetate "
            "columns are Fig. S3's media A and C, whose ammonium sulfate differs from "
            "the served M9 object"
        )
    carbon: list[EnvironmentPerturbationType] = [
        _carbon(
            "D-glucose",
            float(TITER_GLUCOSE_PERCENT.value),
            ConcentrationUnit.percent_w_v,
        )
    ]
    if column == TITER_COLUMNS[3]:
        carbon.append(
            _carbon(
                "acetate",
                float(TITER_ACETATE_PERCENT.value),
                ConcentrationUnit.percent_w_v,
            )
        )
    culture = CULTURE_FORMAT.value
    if not isinstance(culture, dict):
        raise RuntimeError(f"CULTURE_FORMAT.value is not a mapping: {culture!r}")
    return CultureEnvironment(
        media=M9_NREL_DESIQUEIRA2025,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        culture_format=CultureFormat(
            vessel=str(culture["vessel"]),
            working_volume_ul=float(culture["working_volume_ml"]) * 1000.0,
            inoculum_od600=float(culture["inoculum_od600"]),
            provenance=[CULTURE_FORMAT],
        ),
        perturbations=[
            *carbon,
            SmallMoleculePerturbation(
                compound=resolved_compound("L-arabinose"),
                concentration=Concentration(
                    value=float(INDUCER_PERCENT.value),
                    unit=ConcentrationUnit.percent_w_v,
                ),
            ),
            SmallMoleculePerturbation(
                compound=resolved_compound("kanamycin"),
                concentration=Concentration(
                    value=float(KANAMYCIN_UG_PER_ML.value),
                    unit=ConcentrationUnit.ug_per_ml,
                ),
            ),
        ],
        aerobicity=str(AEROBICITY.value),
        duration_hours=None,
        provenance_gaps=[
            ProvenanceGap(
                field="duration_hours",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=(
                    "Table S2 reports the MAXIMUM over the "
                    f"{SAMPLING_TIMES_HOURS.value} h samples, so no single duration "
                    "describes the stored value"
                ),
            )
        ],
    )


def isoprenol_product() -> Any:
    """The product as a typed ``Compound`` through the shared compound-identity layer.

    ``compound_identity_table.json`` carries the isoprenol row (PubChem CID 12988), so
    the resolver returns the full identity; :data:`ISOPRENOL_INCHIKEY` is the recorded
    cross-check that the row is this molecule.
    """
    return resolved_compound("isoprenol")


def titer_phenotype(titer_mm: float) -> ProductTiterPhenotype:
    """One Table S2 maximum titer, stored verbatim in millimolar.

    ``titer_uncertainty`` is a declared gap: Table S2 releases a maximum with no
    uncertainty, and the per-time-point means with SD are plotted in Fig. 3 but never
    released as a table.
    """
    if not math.isfinite(titer_mm) or titer_mm < 0:
        raise RuntimeError(f"a titer must be finite and non-negative, got {titer_mm}")
    return ProductTiterPhenotype(
        product=isoprenol_product(),
        titer=titer_mm,
        titer_unit=ConcentrationUnit.millimolar,
        titer_uncertainty=None,
        titer_uncertainty_type=None,
        n_samples=int(TITER_N_REPLICATES.value),
        sample_unit=SampleUnit.biological_replicate,
        quantification_method=str(QUANTIFICATION.value),
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=note,
            )
            for field, note in (
                (
                    "titer_uncertainty",
                    "Table S2 releases the maximum titer only; Fig. 3B and 3C plot the "
                    "per-time-point mean with its SD but no numeric table of them was "
                    "released",
                ),
                (
                    "titer_uncertainty_type",
                    "no uncertainty number is released for the stored maximum, so none "
                    "is labelled",
                ),
                (
                    "product_yield",
                    "neither a yield on substrate nor a volumetric productivity is "
                    "released for any strain",
                ),
                ("product_yield_unit", "no yield is released, so no unit is"),
                (
                    "productivity",
                    "neither a yield on substrate nor a volumetric productivity is "
                    "released for any strain",
                ),
                ("productivity_unit", "no productivity is released, so no unit is"),
            )
        ],
    )


def released_titers() -> dict[str, dict[str, float | None]]:
    """Table S2 as a nested mapping, ``None`` where the caption writes ``n.d.``."""
    return {strain: dict(cells) for strain, cells in dict(TITER_TABLE.value).items()}


class TiterCensus(BaseModel):
    """The Table S2 cell arithmetic, derived from the released table, never asserted."""

    n_cells: int
    n_not_determined: int
    n_unwritable_strain: int
    n_medium_not_in_library: int
    n_loaded: int

    def check(self) -> None:
        """Every cell is in exactly one bucket."""
        total = (
            self.n_not_determined
            + self.n_unwritable_strain
            + self.n_medium_not_in_library
            + self.n_loaded
        )
        if total != self.n_cells:
            raise RuntimeError(
                f"Table S2 census does not partition its {self.n_cells} cells: {total}"
            )


def titer_census() -> TiterCensus:
    """Partition Table S2's cells into not-determined, unwritable, unloaded and loaded."""
    table = released_titers()
    n_cells = sum(len(cells) for cells in table.values())
    n_nd = sum(1 for cells in table.values() for v in cells.values() if v is None)
    n_unwritable = sum(
        1
        for strain, cells in table.items()
        if strain not in TITER_STRAINS_LOADED
        for value in cells.values()
        if value is not None
    )
    n_unloaded = sum(
        1
        for strain in TITER_STRAINS_LOADED
        for column, value in table[strain].items()
        if value is not None and column not in TITER_COLUMNS_LOADED
    )
    n_loaded = sum(
        1
        for strain in TITER_STRAINS_LOADED
        for column, value in table[strain].items()
        if value is not None and column in TITER_COLUMNS_LOADED
    )
    census = TiterCensus(
        n_cells=n_cells,
        n_not_determined=n_nd,
        n_unwritable_strain=n_unwritable,
        n_medium_not_in_library=n_unloaded,
        n_loaded=n_loaded,
    )
    census.check()
    return census


def assert_titer_cross_source() -> float:
    """Table S2's PT glucose cell must equal the Results text's mg/L statement.

    Returns the measured disagreement in mM. The molar mass is used ONLY here and never
    to produce a stored number.
    """
    cell = released_titers()[PT_STRAIN][TITER_REFERENCE_COLUMN]
    if cell is None:
        raise RuntimeError("Table S2's PT glucose cell is not determined")
    released = float(cell)
    text = float(dict(TITER_CROSS_SOURCE.value)["titer_mg_per_l"]) / ISOPRENOL_G_PER_MOL
    difference = abs(released - text)
    if difference > TITER_CROSS_SOURCE_TOL_MM:
        raise RuntimeError(
            f"Table S2's {released} mM and the Results text's "
            f"{dict(TITER_CROSS_SOURCE.value)['titer_mg_per_l']} mg/L disagree by "
            f"{difference} mM, more than Table S2's own rounding"
        )
    return difference


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, what it removed, and the items it removed."""

    rule: str
    scope: Literal["sample", "titer_cell", "protein_key", "strain"]
    description: str
    n_records: int
    items: list[str] = []


class BuildAccounting(BaseModel):
    """Everything a reader needs to audit one build's retention arithmetic."""

    dataset: str
    source_rows: int
    candidate_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule] = []
    reconciliation: LocusTagReconciliation | None = None
    notes: list[str] = []

    def check(self) -> None:
        """The rules must account for every candidate record that was not written."""
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_records} "
                f"dropped != {self.candidate_records} candidates"
            )
        accounted = sum(
            rule.n_records for rule in self.rules if rule.scope != "protein_key"
        )
        if accounted != self.dropped_records:
            raise RuntimeError(
                f"{self.dataset}: record-scoped rules total {accounted}, "
                f"{self.dropped_records} records are missing from the build"
            )


def _write_accounting(accounting: BuildAccounting, preprocess_dir: str) -> None:
    """Validate and write ``preprocess/build_accounting.json``."""
    accounting.check()
    os.makedirs(preprocess_dir, exist_ok=True)
    with open(osp.join(preprocess_dir, "build_accounting.json"), "w") as handle:
        handle.write(accounting.model_dump_json(indent=2))


def _write_variant_ledger(path: str, variants_path: str) -> VariantLedger:
    """Type every called variant and write the ledger beside the build."""
    ledger = variant_ledger(read_variant_calls(variants_path))
    os.makedirs(path, exist_ok=True)
    with open(osp.join(path, "called_variants.json"), "w") as handle:
        handle.write(ledger.model_dump_json(indent=2))
    return ledger


def _unsequenced_drop_rule(
    ledger: VariantLedger,
    n_records: int,
    scope: Literal["sample", "titer_cell", "protein_key", "strain"],
) -> DropRule:
    """The one rule left after #731: the isolate that was never sequenced at all.

    The other four tolerized isolates ARE written now, each as PT's deletion plus its
    own calls (``perturbations_per_strain`` in the ledger). Sigma3 is not, and no schema
    change reaches it: Data Set S2 releases no row for it, so its genotype is unknown
    rather than untyped, and writing it with PT's genotype alone would assert it is
    genotypically PT.
    """
    return DropRule(
        rule="strain_was_never_sequenced",
        scope=scope,
        description=(
            f"Data Set S2 releases {ledger.n_calls} calls over "
            f"{len(ledger.calls_per_strain)} clones and none of them is "
            f"{UNSEQUENCED_SIGMA}, so its genotype is unknown; writing it would assert "
            "it is genotypically identical to the PT it was evolved from. The four "
            f"sequenced isolates carry {ledger.perturbations_per_strain} called-variant "
            "perturbations each (preprocess/called_variants.json)"
        ),
        n_records=n_records,
        items=[UNSEQUENCED_SIGMA],
    )


def _standard_names(genome: PPutidaKT2440Genome, tags: Iterable[str]) -> dict[str, str]:
    """Locus tag -> the genome's own gene symbol for it, falling back to the tag."""
    exact, _ = genome.feature_index["symbol"]
    by_tag: dict[str, str] = {}
    for symbol, loci in exact.items():
        for locus in loci:
            by_tag.setdefault(str(locus), str(symbol))
    return {tag: by_tag.get(tag, tag) for tag in tags}


# --------------------------------------------------------------------------- #
# Family 1: the released proteome, one class per released normalization
# --------------------------------------------------------------------------- #
@register_dataset
class ProteomeDeSiqueira2025Dataset(ExperimentDataset):
    """de Siqueira 2025 Top3 proteome of the writable KT2440 and PT strains.

    The base of the three normalization classes and the Top3 one itself. A subclass
    changes exactly two things: :attr:`NORMALIZATION` (the released column pair it reads
    and the ``measurement_type`` it stores) and its own dev-tree ``root`` default.
    """

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"
    #: The released normalization this class stores. One per class, because the shared
    #: ``verify_protein_dataset`` asserts a single ``measurement_type`` per dataset.
    NORMALIZATION: ClassVar[ProteomeNormalization] = PROTEOME_NORMALIZATIONS[0]
    #: Measured on the pinned workbook: 1,532 of 1,729 protein keys (0.8872) resolve to
    #: a locus of this assembly. The threshold sits just below that. The 197 that do not
    #: are keys the GenBank annotation carries no symbol for plus the five contaminants
    #: the DIA-NN database was built to include; a real drop means the keying changed.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 0.88

    def __init__(
        self,
        root: str = "data/torchcell/proteome_desiqueira2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome resolves the released protein keys."""
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
        """Data Set S1 (the matrix) and Data Set S2 (the variant ledger)."""
        return [PROTEOME_FILENAME, VARIANTS_FILENAME]

    def download(self) -> None:
        """Link both mirror files into ``raw/`` after verifying each against its pin."""
        _link_mirror_files(
            self.raw_dir,
            (
                (PROTEOME_REL, PROTEOME_FILENAME, PROTEOME_SHA256),
                (VARIANTS_REL, VARIANTS_FILENAME, VARIANTS_SHA256),
            ),
        )
        log.info("de Siqueira 2025 proteome artifacts linked into %s", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    @classmethod
    def _phenotype(
        cls, cells: dict[str, tuple[float, float]], n_replicates: int
    ) -> ProteinAbundancePhenotype:
        """One sample's abundances with SE = SD / sqrt(n) per protein.

        The released SD is the sample SD over the three replicates on THIS class's
        scale, so the SE is that SD over sqrt(n) on the same scale. A log10 SD divided
        by sqrt(n) is the SE of the log10 mean, which is the quantity the record stores.
        """
        if not cells:
            raise RuntimeError("a proteome sample carries no measured protein")
        root_n = math.sqrt(n_replicates)
        return ProteinAbundancePhenotype(
            protein_abundance={tag: mean for tag, (mean, _) in cells.items()},
            protein_abundance_se={tag: sd / root_n for tag, (_, sd) in cells.items()},
            n_replicates=dict.fromkeys(cells, n_replicates),
            measurement_type=cls.NORMALIZATION.measurement_type,
        )

    @post_process
    def process(self) -> None:
        """Build one record per released proteome sample of a writable strain."""
        verify_raw_files(
            self.raw_dir,
            {PROTEOME_FILENAME: PROTEOME_SHA256, VARIANTS_FILENAME: VARIANTS_SHA256},
        )
        rows = read_proteome_rows(osp.join(self.raw_dir, PROTEOME_FILENAME))
        assert_percent_cv_is_derived(rows)
        assert_sem_is_constant_per_protein(rows)
        ledger = _write_variant_ledger(
            self.preprocess_dir, osp.join(self.raw_dir, VARIANTS_FILENAME)
        )
        genome = self._genome()

        keys = sorted({row.protein for row in rows})
        stored, report = reconcile_locus_tags(genome, pd.Series(keys), label=self.name)
        report.require_resolved(self.MIN_RESOLVED_FRACTION)
        stored_by_key = dict(zip(keys, stored, strict=True))

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
        kept = [key for key in keys if key not in set(dropped_keys)]
        if not kept:
            raise RuntimeError(f"{self.name}: every protein key was dropped")

        samples: dict[tuple[str, str], dict[str, tuple[float, float]]] = defaultdict(
            dict
        )
        all_samples: set[tuple[str, str]] = set()
        for row in rows:
            all_samples.add((row.strain, row.condition))
            if row.strain not in WRITABLE_STRAINS or row.protein in set(dropped_keys):
                continue
            tag = stored_by_key[row.protein]
            if tag in samples[(row.strain, row.condition)]:
                raise RuntimeError(
                    f"{(row.strain, row.condition)}/{tag} appears twice; a repeated "
                    "cell would change the stored mean"
                )
            samples[(row.strain, row.condition)][tag] = row.normalized(
                self.NORMALIZATION
            )

        if len(samples) != EXPECTED_PROTEOME_RECORDS:
            raise RuntimeError(
                f"{self.name}: {len(samples)} writable samples, not the "
                f"{EXPECTED_PROTEOME_RECORDS} the pinned workbook holds "
                f"({sorted(samples)}); the released sample set changed"
            )
        n_replicates = int(PROTEOME_N_REPLICATES.value)
        reference_key = (PROTEOME_REFERENCE_STRAIN, PROTEOME_REFERENCE_CONDITION)
        if reference_key not in samples:
            raise RuntimeError(
                f"{self.name}: the reference sample {reference_key} is not released"
            )
        baseline = self._phenotype(samples[reference_key], n_replicates)
        references = {
            strain: BacterialProteinAbundanceExperimentReference(
                dataset_name=self.name,
                genome_reference=strain_reference(strain),
                environment_reference=proteome_environment(
                    PROTEOME_REFERENCE_CONDITION
                ),
                phenotype_reference=baseline,
            )
            for strain in WRITABLE_STRAINS
        }
        pub = publication()

        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        sample_rows: list[dict[str, Any]] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for (strain, condition), cells in tqdm(
                sorted(samples.items()), desc="desiqueira2025-proteome"
            ):
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=strain_genotype(
                        strain, with_pathway=False, calls=ledger.calls
                    ),
                    environment=proteome_environment(condition),
                    phenotype=self._phenotype(cells, n_replicates),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, references[strain], pub, itxn),
                )
                sample_rows.append(
                    {
                        "strain": strain,
                        "condition": condition,
                        "n_proteins": len(cells),
                        "n_perturbations": len(
                            strain_genotype(
                                strain, with_pathway=False, calls=ledger.calls
                            )
                        ),
                        "n_replicates": n_replicates,
                        "measurement_type": self.NORMALIZATION.measurement_type,
                        "mean_column": self.NORMALIZATION.mean_column,
                        "sd_column": self.NORMALIZATION.sd_column,
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(sample_rows).to_csv(
            osp.join(self.preprocess_dir, "samples.csv"), index=False
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
        dropped_samples = sorted(
            f"{strain}/{condition}"
            for strain, condition in all_samples
            if strain not in WRITABLE_STRAINS
        )
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(rows),
                candidate_records=len(all_samples),
                kept_records=idx,
                dropped_records=len(dropped_samples),
                rules=[
                    _unsequenced_drop_rule(ledger, len(dropped_samples), "sample"),
                    DropRule(
                        rule="protein_key_is_not_a_locus_of_the_pinned_assembly",
                        scope="protein_key",
                        description=(
                            "the key is a title-cased UniProt gene symbol the GenBank "
                            "annotation of this assembly carries no symbol for, or one "
                            "of the contaminants the DIA-NN database was built to "
                            f"include ({_Q_DIANN_DB}); it has no gene node to key an "
                            "abundance to. A UniProt-to-locus-tag crosswalk in the "
                            "genomes tier would recover the host proteins, and is "
                            "proposed in the PR"
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
                ],
                reconciliation=report,
                notes=[
                    f"{idx} of {len(all_samples)} released samples are records: the "
                    f"writable strains are {list(WRITABLE_STRAINS)}; "
                    f"{UNSEQUENCED_SIGMA} is out because Data Set S2 releases no call "
                    "for it (issue #731 made the other four isolates writable, and no "
                    "schema change reaches an unsequenced one)",
                    f"the phenotype_reference of every record is "
                    f"{PROTEOME_REFERENCE_STRAIN} in {PROTEOME_REFERENCE_CONDITION} "
                    f"({_Q_REFERENCE_CONDITION}); it is COPIED into the reference and "
                    "kept as a record, because a measured sample is not spent by being "
                    "used as a baseline. genome_reference is the RECORD's own strain, "
                    "so a PT record carries the PT background and a WT record carries "
                    "none",
                    f"{len(dropped_keys)} of {len(keys)} protein KEYS are dropped, "
                    f"leaving {len(kept)} in every record's abundance map",
                    "every sequenced strain's genotype is PT's PP_2675 deletion plus "
                    f"its own called variants: {ledger.perturbations_per_strain} "
                    "perturbations written per strain, and "
                    f"{ledger.encodings.get(ENCODING_RESTATES_DESIGNED_DELETION, 0)} "
                    "calls dropped as restatements of that same designed deletion "
                    f"(encodings {ledger.encodings})",
                    f"{ledger.unsequenced_strains} were never sequenced at all",
                    f"this class stores the {self.NORMALIZATION.measurement_type!r} "
                    f"scale, read from {self.NORMALIZATION.mean_column!r} with its SD "
                    f"{self.NORMALIZATION.sd_column!r}: "
                    f"{self.NORMALIZATION.note}. The other "
                    f"{len(PROTEOME_NORMALIZATIONS) - 1} released normalizations of "
                    "these same samples are their own dataset classes, because "
                    "verify_protein_dataset asserts one measurement_type per dataset",
                    "the released CV%_of_%_protein_abundance and the constant "
                    "%_of protein_abundance_Top3_rep_mean_sem columns are read and used "
                    "as build oracles rather than stored: the CV is exactly "
                    "100 * pct_sd / pct_mean over all released cells, and the SEM does "
                    "not vary with the sample, so neither is a per-sample phenotype",
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "deSiqueira2025 proteome: %d records over %d protein keys from %d released "
            "cells; %d keys dropped (%d outside the namespace, %d merged), %d of %d "
            "samples left out because their strain was never sequenced",
            idx,
            len(kept),
            len(rows),
            len(dropped_keys),
            len(outside),
            len(merged),
            len(dropped_samples),
            len(all_samples),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


@register_dataset
class ProteomePercentDeSiqueira2025Dataset(ProteomeDeSiqueira2025Dataset):
    """de Siqueira 2025 percent-of-total relative abundance, the paper's own scale.

    The same five writable samples as :class:`ProteomeDeSiqueira2025Dataset`, read from
    ``%_of protein_abundance_Top3_rep_mean`` with its ``_rep_std`` as the SD. This is
    the column the paper's OWN analysis runs on ("Relative protein abundances of
    proteins were used for dimensionality reduction."), and it is not recoverable from
    the Top3 pair: measured over all 34,600 released cells, zero equal the percent of
    the released mean and the per-cell ratio runs 0.914506 to 1.083554, which is the
    signature of a replicate-wise mean of per-replicate percentages.
    """

    NORMALIZATION: ClassVar[ProteomeNormalization] = PROTEOME_NORMALIZATIONS[1]

    def __init__(
        self,
        root: str = "data/torchcell/proteome_percent_desiqueira2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with this normalization's own dev-tree root."""
        super().__init__(
            root, io_workers, pputida_genome, transform, pre_transform, **kwargs
        )


@register_dataset
class ProteomeLog10PercentDeSiqueira2025Dataset(ProteomeDeSiqueira2025Dataset):
    """de Siqueira 2025 mean log10 percent-of-total abundance, the third released scale.

    Read from ``log10_%_abundance_rep_mean`` with its ``_rep_std``. It is the MEAN of
    the three replicates' log10, not the log10 of the mean: measured over all 34,600
    released cells, 33,914 sit strictly below ``log10`` of the released percent, 686
    agree to within 5e-5 and ZERO sit above it, which is exactly Jensen's inequality for
    a mean of logs. Its SD is likewise not the delta-method transform of the percent SD
    (34,496 of 34,600 cells disagree), so neither column is recoverable from a stored
    sibling.
    """

    NORMALIZATION: ClassVar[ProteomeNormalization] = PROTEOME_NORMALIZATIONS[2]

    def __init__(
        self,
        root: str = "data/torchcell/proteome_log10_percent_desiqueira2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with this normalization's own dev-tree root."""
        super().__init__(
            root, io_workers, pputida_genome, transform, pre_transform, **kwargs
        )


#: Each proteome class keyed by the normalization it stores, so the verifier and the
#: module entry point can walk the family rather than naming the three classes twice.
PROTEOME_CLASSES: dict[str, type[ProteomeDeSiqueira2025Dataset]] = {
    PROTEOME_MEASUREMENT_TYPE: ProteomeDeSiqueira2025Dataset,
    PERCENT_MEASUREMENT_TYPE: ProteomePercentDeSiqueira2025Dataset,
    LOG10_PERCENT_MEASUREMENT_TYPE: ProteomeLog10PercentDeSiqueira2025Dataset,
}
if {cls.NORMALIZATION.measurement_type for cls in PROTEOME_CLASSES.values()} != set(
    PROTEOME_CLASSES
):
    raise RuntimeError(
        "a proteome class is keyed by a measurement_type it does not store"
    )


# --------------------------------------------------------------------------- #
# Family 2: the Table S2 isoprenol titers
# --------------------------------------------------------------------------- #
@register_dataset
class IsoprenolTiterDeSiqueira2025Dataset(ExperimentDataset):
    """de Siqueira 2025 maximum isoprenol titer of PT carrying pIY670."""

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"

    def __init__(
        self,
        root: str = "data/torchcell/isoprenol_titer_desiqueira2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the genome is accepted for the shared injection rule."""
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
        """Data Set S2, for the variant ledger this family's drops rest on."""
        return [VARIANTS_FILENAME]

    def download(self) -> None:
        """Link Data Set S2 into ``raw/`` after verifying it against its pin."""
        _link_mirror_files(
            self.raw_dir, ((VARIANTS_REL, VARIANTS_FILENAME, VARIANTS_SHA256),)
        )
        log.info("de Siqueira 2025 titer artifacts linked into %s", self.raw_dir)

    @post_process
    def process(self) -> None:
        """Build one titer record per loaded Table S2 cell of a writable strain."""
        verify_raw_files(self.raw_dir, {VARIANTS_FILENAME: VARIANTS_SHA256})
        ledger = _write_variant_ledger(
            self.preprocess_dir, osp.join(self.raw_dir, VARIANTS_FILENAME)
        )
        difference = assert_titer_cross_source()
        census = titer_census()
        table = released_titers()

        reference = ProductTiterExperimentReference(
            dataset_name=self.name,
            genome_reference=strain_reference(PT_STRAIN),
            environment_reference=titer_environment(TITER_REFERENCE_COLUMN),
            phenotype_reference=titer_phenotype(
                float(table[PT_STRAIN][TITER_REFERENCE_COLUMN] or 0.0)
            ),
        )
        references = {
            strain: ProductTiterExperimentReference(
                dataset_name=self.name,
                genome_reference=strain_reference(strain),
                environment_reference=titer_environment(TITER_REFERENCE_COLUMN),
                phenotype_reference=reference.phenotype_reference,
            )
            for strain in TITER_STRAINS_LOADED
        }
        pub = publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        titer_rows: list[dict[str, Any]] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for strain in tqdm(TITER_STRAINS_LOADED, desc="desiqueira2025-titer"):
                genotype = strain_genotype(
                    strain, with_pathway=True, calls=ledger.calls
                )
                for column in TITER_COLUMNS_LOADED:
                    cell = table[strain][column]
                    if cell is None:
                        raise RuntimeError(
                            f"Table S2's {strain}/{column} cell is not determined"
                        )
                    value = float(cell)
                    experiment = ProductTiterExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
                        environment=titer_environment(column),
                        phenotype=titer_phenotype(value),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(experiment, references[strain], pub, itxn),
                    )
                    titer_rows.append(
                        {
                            "strain": strain,
                            "medium": column,
                            "max_titer_mM": value,
                            "n_perturbations": len(genotype),
                            "statistic": str(TITER_STATISTIC.value),
                        }
                    )
                    idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(titer_rows).to_csv(
            osp.join(self.preprocess_dir, "titers.csv"), index=False
        )
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=census.n_cells,
                candidate_records=census.n_cells,
                kept_records=idx,
                dropped_records=census.n_cells - idx,
                rules=[
                    _unsequenced_drop_rule(
                        ledger, census.n_unwritable_strain, "titer_cell"
                    ),
                    DropRule(
                        rule="cell_is_not_determined",
                        scope="titer_cell",
                        description=(
                            "Table S2 writes 'n.d.' for these cells, which the caption "
                            "defines as not determined; an undetermined cell is not a "
                            "measurement"
                        ),
                        n_records=census.n_not_determined,
                        items=[
                            f"{strain}/{column}"
                            for strain, cells in released_titers().items()
                            for column, value in cells.items()
                            if value is None
                        ],
                    ),
                    DropRule(
                        rule="medium_is_not_in_the_media_library",
                        scope="titer_cell",
                        description=(
                            "the two acetate columns are Fig. S3's media A and C, whose "
                            f"ammonium sulfate ({_Q_FIGS3_MEDIA}) differs from the 2 "
                            "g/L the served M9_NREL_DESIQUEIRA2025 states, so they are "
                            "a different medium and MEDIA_LIBRARY holds no object for "
                            "them. All four determined cells (PT and Sigma1 in both "
                            "acetate media) read 0, so no non-zero measurement is "
                            "lost; the two media entries are proposed in the PR"
                        ),
                        n_records=census.n_medium_not_in_library,
                        items=[
                            f"{strain}/{column}"
                            for strain in TITER_STRAINS_LOADED
                            for column, value in released_titers()[strain].items()
                            if value is not None and column not in TITER_COLUMNS_LOADED
                        ],
                    ),
                ],
                notes=[
                    f"the stored statistic is the {TITER_STATISTIC.value} over the "
                    f"{SAMPLING_TIMES_HOURS.value} h samples, an upward-biased order "
                    "statistic, not a single-time-point titer",
                    "cross-source check: Table S2's "
                    f"{table[PT_STRAIN][TITER_REFERENCE_COLUMN]} mM for PT in glucose and the "
                    f"Results text's {dict(TITER_CROSS_SOURCE.value)['titer_mg_per_l']}"
                    f" mg/L differ by {difference:.4f} mM",
                    "cross-source DISAGREEMENT, kept: for the mixed feed the Results "
                    f"say '{_Q_PT_MIXED_NOT_DETECTABLE}' while Table S2 releases "
                    f"{table[PT_STRAIN][TITER_COLUMNS[3]]} mM; the released number is stored",
                    f"the reference is PT in {TITER_REFERENCE_COLUMN}, the single-carbon "
                    "baseline the mixed feed is compared against",
                    f"{census.model_dump_json()}",
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "deSiqueira2025 titer: %d records of %d Table S2 cells; %d cells of the "
            "never-sequenced isolate and %d n.d. cells and %d cells in a medium the "
            "library lacks are left out; cross-source |diff| %.4f mM",
            idx,
            census.n_cells,
            census.n_unwritable_strain,
            census.n_not_determined,
            census.n_medium_not_in_library,
            difference,
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# The module's own L0-L4 runner
# --------------------------------------------------------------------------- #
def _provenance(source_uri: str, sha256: str, method: str, page: str) -> Provenance:
    """A verification-report provenance pinned to one mirrored artifact."""
    return Provenance(
        source_uri=source_uri,
        citation_key=CITATION_KEY,
        sha256=sha256,
        method=method,
        page=page,
    )


def strain_condition_uniqueness(records: Sequence[dict[str, Any]]) -> LevelResult:
    """SUPPLEMENTARY L1: one record per (strain background, environment).

    The shared ``orf_uniqueness`` rule keys on the knocked-out ORF, which a per-strain
    per-condition design repeats by construction: PT appears once per condition and the
    wild type carries no perturbation at all. This row is added beside it, never in its
    place.
    """
    seen: Counter[tuple[str, str]] = Counter()
    for record in records:
        strain = str(record["reference"]["genome_reference"]["strain"])
        perturbations = record["experiment"]["genotype"]["perturbations"]
        background = "+".join(
            sorted(str(p["systematic_gene_name"]) for p in perturbations)
        )
        media = record["experiment"]["environment"]["media"]["name"]
        doses = "+".join(
            sorted(
                f"{p.get('factor', p['perturbation_type'])}"
                f"={(p.get('magnitude') or p.get('concentration') or {}).get('value')}"
                for p in record["experiment"]["environment"]["perturbations"]
            )
        )
        seen[(f"{strain}|{background}", f"{media}|{doses}")] += 1
    repeated = {f"{k[0]} in {k[1]}": n for k, n in seen.items() if n > 1}
    return LevelResult(
        level=Level.L1,
        name="strain_condition_uniqueness",
        passed=not repeated,
        message=(
            f"SUPPLEMENTARY: {len(seen)} distinct (strain, environment) pairs over "
            f"{len(records)} records"
        ),
        details={"n_pairs": len(seen), "repeated": repeated},
    )


def stored_normalization_rule(
    records: Sequence[dict[str, Any]], normalization: ProteomeNormalization
) -> LevelResult:
    """SUPPLEMENTARY L3: every record stores THIS normalization and names its column.

    The shared verifier's ``measurement_type_consistent`` row proves the records agree
    with EACH OTHER; this one proves they agree with the released column the class reads,
    which is what keeps three dataset classes over one workbook from swapping scales.
    """
    stored = {
        str(rec["experiment"]["phenotype"]["measurement_type"]) for rec in records
    }
    references = {
        str(rec["reference"]["phenotype_reference"]["measurement_type"])
        for rec in records
    }
    expected = {normalization.measurement_type}
    return LevelResult(
        level=Level.L3,
        name="stored_scale_is_the_released_column_this_class_reads",
        passed=stored == expected and references == expected,
        message=(
            f"SUPPLEMENTARY: {len(records)} records on "
            f"{normalization.measurement_type!r}, read from "
            f"{normalization.mean_column!r} + {normalization.sd_column!r}"
        ),
        details={
            "expected": normalization.measurement_type,
            "experiment_measurement_types": sorted(stored),
            "reference_measurement_types": sorted(references),
            "mean_column": normalization.mean_column,
            "sd_column": normalization.sd_column,
            "recoverable_from_top3": normalization.recoverable_from_top3,
        },
    )


def assembly_pin_rule(records: Sequence[dict[str, Any]]) -> LevelResult:
    """SUPPLEMENTARY L3: every record pins the KT2440 GenBank assembly."""
    pins = {
        (
            record["reference"]["genome_reference"].get("assembly_set"),
            record["reference"]["genome_reference"].get("assembly_accession"),
        )
        for record in records
    }
    expected = {(KT2440_ASSEMBLY_SET, "GCA_000007565.2")}
    return LevelResult(
        level=Level.L3,
        name="assembly_pin",
        passed=pins == expected,
        message=f"SUPPLEMENTARY: assembly pins {sorted(map(str, pins))}",
        details={"pins": sorted(map(str, pins))},
    )


def gene_containment_rule(
    records: Sequence[dict[str, Any]], universe: set[str]
) -> LevelResult:
    """L4: every HOST identifier a record carries is a locus of the pinned assembly.

    The heterologous pIY670 parts are deliberately exempt: they are not KT2440 genes,
    which is what ``HeterologousPathwayPerturbation`` exists to say.
    """
    heterologous = {str(gene["token"]) for gene in PATHWAY_GENES}
    measured: set[str] = set()
    perturbed: set[str] = set()
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        measured.update(phenotype.get("protein_abundance") or {})
        perturbed.update(
            str(p["systematic_gene_name"])
            for p in record["experiment"]["genotype"]["perturbations"]
            # A heterologous pathway gene is a gene of ANOTHER genome, and a SITE-keyed
            # called variant (#731) names no gene at all: its identifier is the derived
            # <replicon>:<position>, which is exactly the claim that no locus of this
            # assembly holds the call, so checking it against the locus universe would
            # fail the record for saying something true.
            if p["perturbation_type"]
            not in ("heterologous_pathway", "bacterial_site_variant")
        )
    outside = sorted((measured | perturbed) - universe - heterologous)
    return LevelResult(
        level=Level.L4,
        name="gene_containment_kt2440",
        passed=not outside,
        message=(
            f"{len(measured)} measured and {len(perturbed)} perturbed host genes; "
            f"{len(outside)} outside the KT2440 locus universe"
        ),
        details={"outside": outside[:20], "n_universe": len(universe)},
    )


def titer_levels(
    records: Sequence[dict[str, Any]], *, expected_count: int
) -> VerificationReport:
    """The L0-L4 gate for the titer family; there is no shared titer verifier yet."""
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType
    from torchcell.verification.levels import (
        l0_structural,
        l1_count,
        l2_value_fidelity,
        l3_convention,
    )

    report = VerificationReport(
        dataset_name="isoprenol_titer_desiqueira2025",
        provenance=_provenance(
            SI3_MD,
            SI3_MD_SHA256,
            "Table S2 maximum isoprenol titer (mM) over the 24, 48 and 72 h samples, "
            "stored verbatim; the medium is M9_NREL_DESIQUEIRA2025 plus its carbon "
            "sources, the arabinose inducer and the kanamycin selection",
            "Table S2",
        ),
    )
    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    report.add(l0_structural((r["experiment"] for r in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(strain_condition_uniqueness(records))
    titers = [float(r["experiment"]["phenotype"]["titer"]) for r in records]
    report.add(l2_value_fidelity(titers, allow_nan=False, minimum=0.0))
    units = {r["experiment"]["phenotype"]["titer_unit"] for r in records}
    report.add(
        l3_convention(
            "titer_unit_is_the_released_unit",
            units == {ConcentrationUnit.millimolar.value},
            detail=f"stored titer units {sorted(units)}; Table S2 releases mM",
        )
    )
    report.add(
        l3_convention(
            "stored_statistic_is_a_maximum",
            all(
                r["experiment"]["phenotype"]["titer_uncertainty"] is None
                for r in records
            ),
            detail=(
                f"the stored value is the {TITER_STATISTIC.value}, an upward-biased "
                "order statistic; no uncertainty is released for it"
            ),
        )
    )
    report.add(
        l3_convention(
            "mixed_feed_cross_source_disagreement_is_declared",
            True,
            detail=(
                f"Results: '{_Q_PT_MIXED_NOT_DETECTABLE}'; Table S2: "
                f"{released_titers()[PT_STRAIN][TITER_COLUMNS[3]]} mM. The released "
                "is stored and the disagreement is reported"
            ),
        )
    )
    report.add(assembly_pin_rule(records))
    report.add(
        l3_convention(
            "cross_source_titer_agrees_with_the_results_text",
            assert_titer_cross_source() <= TITER_CROSS_SOURCE_TOL_MM,
            detail=(
                f"|Table S2 - Results text| = {assert_titer_cross_source():.4f} mM, "
                f"within Table S2's own {TITER_CROSS_SOURCE_TOL_MM} mM rounding"
            ),
        )
    )
    return report


#: ``family`` -> the proteome class it verifies. ``"proteome"`` stays the Top3 family's
#: name so an existing caller (the runner registry, the note, a by-hand verify) is
#: unchanged.
PROTEOME_FAMILIES: dict[str, type[ProteomeDeSiqueira2025Dataset]] = {
    "proteome": ProteomeDeSiqueira2025Dataset,
    "proteome_percent": ProteomePercentDeSiqueira2025Dataset,
    "proteome_log10_percent": ProteomeLog10PercentDeSiqueira2025Dataset,
}


def verify_build(
    dataset_root: str, data_root: str | None = None, *, family: str = "proteome"
) -> VerificationReport:
    """Run this module's L0-L4 gate over a built tree and write the report.

    ``family`` is ``"titer"`` or one of :data:`PROTEOME_FAMILIES`. A proteome family
    runs the shared ``verify_protein_dataset`` with ``allow_duplicate_orfs=True`` (PT
    appears once per condition by design) plus three SUPPLEMENTARY rows; the titer
    family has no shared verifier, so :func:`titer_levels` builds its battery. The
    report is written to ``preprocess/verification_report.json``.
    """
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    genome = bacterial_genome("pputida", "KT2440", data_root)
    if family == "titer":
        report = titer_levels(
            records,
            expected_count=len(TITER_STRAINS_LOADED) * len(TITER_COLUMNS_LOADED),
        )
        report.add(gene_containment_rule(records, set(genome.genbank.loci)))
    elif family in PROTEOME_FAMILIES:
        from torchcell.verification.protein import verify_protein_dataset

        dataset_cls = PROTEOME_FAMILIES[family]
        normalization = dataset_cls.NORMALIZATION
        report = verify_protein_dataset(
            records,
            dataset_name=normalization.root_slug,
            provenance=_provenance(
                PROTEOME_REL,
                PROTEOME_SHA256,
                f"Data Set S1 {normalization.mean_column} (the replicate-wise mean over "
                f"three biological replicates), with SE = {normalization.sd_column} / "
                "sqrt(3)",
                "Data Set S1, sheet 'Sheet1'",
            ),
            expected_count=EXPECTED_PROTEOME_RECORDS,
            allow_duplicate_orfs=True,
        )
        report.add(strain_condition_uniqueness(records))
        report.add(assembly_pin_rule(records))
        report.add(gene_containment_rule(records, set(genome.genbank.loci)))
        report.add(stored_normalization_rule(records, normalization))
    else:
        raise RuntimeError(
            f"{family!r} is neither 'titer' nor one of {sorted(PROTEOME_FAMILIES)}"
        )
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build both families and verify them, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    genome = bacterial_genome("pputida", "KT2440", data_root)
    families: list[
        tuple[
            type[ProteomeDeSiqueira2025Dataset]
            | type[IsoprenolTiterDeSiqueira2025Dataset],
            str,
            str,
        ]
    ] = [
        (cls, f"data/torchcell/{cls.NORMALIZATION.root_slug}", family)
        for family, cls in PROTEOME_FAMILIES.items()
    ]
    families.append(
        (
            IsoprenolTiterDeSiqueira2025Dataset,
            "data/torchcell/isoprenol_titer_desiqueira2025",
            "titer",
        )
    )
    for cls, rel, family in families:
        root = osp.join(data_root, rel)
        dataset = cls(root=root, pputida_genome=genome)
        print(f"{cls.__name__}: len = {len(dataset)}")
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
        report = verify_build(root, data_root, family=family)
        print(report.summary())


if __name__ == "__main__":
    main()
