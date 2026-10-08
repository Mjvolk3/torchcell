# torchcell/datasets/ecoli/choe2019_growth_rate
# [[torchcell.datasets.ecoli.choe2019_growth_rate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/choe2019_growth_rate
# Test file: tests/torchcell/datasets/ecoli/test_choe2019_growth_rate.py
"""Choe 2019 genome-reduced ALE: the two designed-deletion arms, and nothing evolved.

Choe et al. 2019 (Nat Commun 10:935, doi:10.1038/s41467-019-08888-6) evolved the
genome-reduced *E. coli* MS56 for 807 generations in M9 glucose, resequenced the
population at 20 timepoints, isolated the clone eMS57, and profiled it against MG1655 by
RNA-seq, Ribo-seq, ChIP-seq, a Biolog phenotype microarray and intracellular metabolites.
Two dataset classes serve what is BOTH released as a number AND keyed to a strain whose
genotype this schema can write:

- :class:`GrowthRateChoe2019Dataset` -- ``BacterialFitnessExperiment``, two records:
  ``MS56 Delta21kb`` (21 gene deletions) and ``MS56 DeltarpoS`` (one), each against their
  isogenic parent MS56, from the Fig. 2c sheet of the Source Data workbook.
- :class:`TranscriptionFactorKnockoutChoe2019Dataset` --
  ``BacterialFitnessExperiment``, two records: the Keio BW25113 ``DeltaydaS`` and
  ``DeltaabgR`` single deletions, whose growth rates the Supplementary Fig. 6 legend
  states as a percentage of wild-type BW25113 in the same medium.

THE GENOTYPE FINDING: EVERY VARIANT THIS PAPER RELEASES IS A POPULATION ALLELE
FREQUENCY, AND NO LEAF HOLDS ONE. The three variant tables release 363 calls (117 in
Supplementary Data 2, 101 in Supplementary Data 3, 145 in Supplementary Data 4) and not
one of them is a clone genotype. Supplementary Data 2 gives a 20-timepoint allele
frequency per call, from day 0 to day 62, and the flagship evolved strain eMS57 has no
variant list of its own: a per-clone genotype would have to be produced by thresholding
the day-62 column, which leaves 11 calls at 100%, 30 above 50%, 23 between 5% and 95%
and 79 at zero. The authors never made that call; the lineage analysis they DID make
clusters 48 of those calls into three clonal lineages with five sub-lineages
("Sequence variants with allelic frequency > 0.8 once or > 0.5 at more than two time
points were subjected to hierarchical clustering"), and a lineage is a subpopulation, not
a strain. The ``Lineage Analysis`` column of Supplementary Data 2 marks INCLUSION in that
clustering, not membership of a clone.

Even if a clone genotype existed, no perturbation leaf could hold one of those rows, and
this was MEASURED on these bytes rather than assumed
(``experiments/036-dataset-fixes-before-kg-build/scripts/choe2019_release_shape.py``):

1. ``SequenceVariantPerturbation`` is the nearest class and it REFUSES the identifier.
   ``SequenceVariantPerturbation(systematic_gene_name="b0078", ...)`` raises
   ``Invalid systematic gene name format``: it inherits ``GenePerturbation``'s R64 ORF
   regex and has no ``gene_namespace`` field (issue #749).
2. No class has a slot for what a row RELEASES: position, reference base, alternate base,
   mutation type, amino-acid change, and a frequency per timepoint.
   ``BacterialBackgroundAllele.deleted_span`` is deletion-only and would lose all of it.
3. ``BacterialBackgroundAllele`` requires a non-optional ``functional: bool``
   (``Input should be a valid boolean``), so the unknown consequence of a missense SNV
   cannot be covered by a ``ProvenanceGap`` -- a gap must name a field that is ``None``.
4. ``BacterialStrainBackground`` permits ONE allele entry per locus, which refuses the
   seven loci carrying more than one call in Supplementary Data 2 alone (``yafF`` 5,
   ``ydjN`` 4, ``iscR`` 3, ``rrsH`` / ``tufA`` / ``yeaM`` 2 each, plus 13 rows labelled
   only ``intergenic``).
5. 22 of the 117 rows are intergenic and name no locus at all, so there is nothing to key
   to; inventing a flanking gene is exactly what must not be done.
6. ``Genotype.__eq__`` compares the perturbation SET, so a clone written with its
   parent's perturbations only would be genotype-identical to its parent.

THE DESIGNED POINT MUTANTS SHOW THE SAME GAP WITHOUT ANY EVOLVED-CLONE EXCUSE, WHICH IS
WHY THIS ROW MATTERS TO ISSUE #731. Supplementary Table 3 releases three strains the
authors BUILT on MS56, each a single exactly specified allele with its own isogenic
kan-marked wild-type-allele control: ``MS56, cspC::cspC(G37A)-kanR``,
``MS56, ilvN::ilvN(C202T)-kanR`` and ``MS56, yifB::yifB(1169C insertion)-kanR``. The
Fig. 2f sheet releases their growth rates as three paired mutant-over-parent ratios each.
The strains are designed, the alleles are given to the base, the ratios are released --
and not one record can be written, because the perturbation axis has no bacterial
sequence-variant leaf. That is a designed-strain gap, not an evolved-population one.

A SECOND REASON NOT TO MIX THE TWO GROWTH PANELS, measured here: Fig. 2c's rates give
eMS57/MS56 = 2.8159, while Fig. 2f releases eMS57/MS56 directly as 1.3990 (mean of three
sets). The two panels disagree by a factor of 2.013 on the same strain pair, so they are
not the same measurement and nothing may be derived across them. Fig. 2c is internally
consistent with the paper's own prose instead: it puts ``Delta21kb`` at 0.7763 and
``DeltarpoS`` at 0.7966 of eMS57, against a stated "recovered its growth rate to 80% of
that of eMS57", and ``check_recovery_fraction`` asserts that agreement on every build.

WHY THE REFERENCE IS MS56 AND NOT MG1655. Both deletion strains are "MS56, <lesion>::kan"
(Supplementary Table 3), so their isogenic parent is MS56 and the ratio a
``FitnessPhenotype`` means is to that parent. MS56 itself is carried as a
``BacterialStrainBackground`` whose ``genotype_statement`` is Supplementary Table 3's own
words and whose ``alleles`` list is EMPTY, which is the same statement the landed Girgis
2009 loader makes for MG1655 ``delta-lacZ``: this paper defers MS56's content to its
reference 4 as "large deletions MD1 to MD56" and enumerates no gene, the reduced strain
has no deposited assembly, and typing 55 regions nobody released would be an invention.
Two consequences are stated rather than typed:

- MS56's own Fig. 2c growth rate is NOT loaded. Written as a record it would need an
  empty genotype against an MG1655 reference, which asserts that MS56 is genotypically
  MG1655 while 1.1 Mbp of it is missing. The honest form of that record needs the 55
  regions, which this paper does not give.
- The background's unenumerated content has no TYPED home either. A ``ProvenanceGap`` on
  ``alleles`` is refused by ``ProvenanceGapMixin`` (``field 'alleles' has a
  ProvenanceGap but is not None``), because the field defaults to ``[]`` rather than
  ``None``. The absence therefore lives in ``preprocess/genotype_gaps.json``, in the
  note and here. That refusal is measured, not read off the class.

THE 21 DELETED GENES ARE THE SOURCE'S OWN LIST, AND THE RELEASED COORDINATES ARE NOT
USABLE. Supplementary Table 3 writes the large deletion as
``(hycEDCBA-hypABCDE-fhlA-ygbA-mutS-pphB-ygbIJKLMN-rpoS)::kan``, which expands to exactly
21 gene symbols; all 21 resolve on the pinned MG1655 assembly to the CONTIGUOUS run
``b2721`` to ``b2741``, and no unlisted gene lies inside that window
(``check_deleted_region``). The paper also gives coordinates, "genomic coordinates from
2,038,496 to 2,059,460 bp", and those are MS56 coordinates: the same genes sit at
2,844,762 to 2,867,551 on MG1655, an offset of 806,266 bp that is the upstream deletion
MS56 already carries. The released interval therefore belongs to an undeposited assembly
and is NOT stored as a span; the gene-keyed deletions are what the records carry.

WHAT ELSE IS RELEASED AND WHY IT IS NOT HERE. Every refusal is counted in
``preprocess/dropped_records.json``; the short form:

- RNA-seq (Supplementary Data 6, 3,457 genes x 6 samples) and Ribo-seq (Supplementary
  Data 7) are RPKM only. ``RNASeqExpressionPhenotype`` requires a per-gene integer
  ``expression_count`` alongside ``expression_tpm`` (``Field required``), and the count
  does not back-solve: over the 3,391 b-numbers with a gene span, no assumed total-read
  scale puts more than 3.4% of ``RPKM x span`` products within 0.01 of an integer. Only
  the raw reads under ENA PRJEB21199 could supply it. The MG1655 columns are wild type
  and would otherwise be writable, so this is a phenotype-class gap, not a genotype one.
- Ribosome-profiling translation level and translational efficiency have no phenotype
  class at all, and neither do the 839 sigma-70 ChIP-seq peaks of Supplementary Data 5.
- The Biolog phenotype microarray (Supplementary Data 1, 384 wells x 2 MG1655 + 2 eMS57
  columns) reads out "cellular respiration ... using an Omnilog instrument", an absolute
  endpoint dye-reduction signal. ``MeasurementType`` has no member for it, and it is
  neither a growth rate, a ratio, a z-score nor a colony size.
- Fig. 1a/1b/1d/1h release OD-over-time growth CURVES, not rates; no phenotype class
  holds a growth curve.
- Fig. 1g and Fig. 3d release intracellular metabolite levels for MG1655 and eMS57, and
  Supplementary Table 1 a fed-batch biomass and specific growth rate for the same pair.
  The only non-wild-type strain in any of them is the evolved clone, so the wild-type
  half would be a record whose fitness is 1.0 against itself.
- Fig. 2g's MG1655 valine arm releases no number at all (the cells did not grow), and
  eMS57 is the only other strain in the panel.

DATA. ``GrowthRateChoe2019Dataset`` consumes ``si12.xlsx`` (the Source Data workbook) and
the three variant tables ``si5.xlsx``, ``si6.xlsx`` and ``si7.xlsx``, which it reads only
to type each refused call into ``preprocess/called_variants.json``.
``TranscriptionFactorKnockoutChoe2019Dataset`` pins ``si1.pdf``, the Supplementary
Information whose Fig. 6 legend carries its two numbers; the numbers themselves are
``SourcedValue``s quoting that PDF's OCR, so the build reads no table for them. All five
files come from the raw mirror under the sha256 the library mirror recorded when the PMC
Article Datasets bucket was retrieved.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import os.path as osp
import shutil
import statistics
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

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
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    BacterialAssemblySet,
    BacterialDeletionPerturbation,
    BacterialFitnessExperiment,
    BacterialFitnessExperimentReference,
    BacterialStrainBackground,
    Concentration,
    ConcentrationUnit,
    DerivedIdentifierMapping,
    Environment,
    Experiment,
    ExperimentReference,
    FitnessPhenotype,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    Publication,
    SampleUnit,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import (
    BACTERIAL_ASSEMBLY_SETS,
    STRAIN_GENE_NAMESPACES,
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ROLE_SI_PDF,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

CITATION_KEY = "choeAdaptiveLaboratoryEvolution2019"
PAPER_DOI = "10.1038/s41467-019-08888-6"
PAPER_TITLE = "Adaptive laboratory evolution of a genome-reduced Escherichia coli"
PUBMED_ID = "30804555"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

GROWTH_DATASET_ROOT_REL = "data/torchcell/growth_rate_choe2019"
TF_DATASET_ROOT_REL = "data/torchcell/tf_knockout_growth_choe2019"

#: The two reference strains, one per dataset class.
MG1655: EcoliK12StrainName = "MG1655"
BW25113: EcoliK12StrainName = "BW25113"
MG1655_NAMESPACE = STRAIN_GENE_NAMESPACES[MG1655]
BW25113_NAMESPACE = STRAIN_GENE_NAMESPACES[BW25113]
#: The MG1655 assembly set, annotated so the background's Literal field accepts it;
#: ``test_the_assembly_set_literal_matches_the_registry`` pins it to
#: ``BACTERIAL_ASSEMBLY_SETS`` so the two can never drift.
MG1655_ASSEMBLY_SET: BacterialAssemblySet = "ecoli_K12_MG1655_ASM584v2"

#: ``paper.md`` and ``si/si1.md`` in the LIBRARY mirror: every quote's source.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "11a042209df0f0518197d594340ed4ab573d90ddc2574a37139451ae697f7713"
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "fcadf7faf5805d6a61e557cbf8440f17279664a96d81bf6cbec630a991a957bc"
#: The date the PMC Article Datasets bucket served every mirrored file.
RETRIEVED_AT = "2026-10-07T11:42:06.933682+00:00"
#: PMC bucket prefix of this article's supplementary objects.
PMC_PREFIX = "PMC6389913.1"
PMC_BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com"


class RawArtifact(BaseModel):
    """One mirrored file: its local name, its pin, and how it was retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="file name inside the dataset's ``raw/``")
    library_relpath: str = Field(description="path inside the LIBRARY mirror")
    sha256: str
    bytes: int
    role: str
    pmc_object: str = Field(description="object name inside the PMC bucket prefix")
    si_label: str = Field(description="what the publisher calls this file")

    @property
    def mirror_relpath(self) -> str:
        """Path inside the RAW mirror (``data/<name>``)."""
        return f"data/{self.name}"

    @property
    def source_url(self) -> str:
        """The PMC Article Datasets URL this file was retrieved from."""
        return f"{PMC_BUCKET_URL}/{PMC_PREFIX}/{self.pmc_object}"

    @property
    def retrieval(self) -> RetrievalRecord:
        """The retrieval record the library mirror already holds for this file."""
        return RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=self.source_url,
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": f"{PMC_PREFIX}/{self.pmc_object}"},
            sha256=self.sha256,
            retrieved_at=RETRIEVED_AT,
        )


SOURCE_DATA = RawArtifact(
    name="si12.xlsx",
    library_relpath="si/si12.xlsx",
    sha256="fcc30417afa1a986bfb72294fbd648c11abefe5ef0bed06627f2f7398b544705",
    bytes=157856,
    role=ROLE_RAW_DATA,
    pmc_object="41467_2019_8888_MOESM12_ESM.xlsx",
    si_label="Source Data",
)
VARIANTS_MS56 = RawArtifact(
    name="si5.xlsx",
    library_relpath="si/si5.xlsx",
    sha256="1315778a8810998afcdaa9475023e5c6d261fc72630c95e1e9072515a5e26856",
    bytes=25840,
    role=ROLE_RAW_DATA,
    pmc_object="41467_2019_8888_MOESM5_ESM.xlsx",
    si_label="Supplementary Data 2",
)
VARIANTS_MG1655 = RawArtifact(
    name="si6.xlsx",
    library_relpath="si/si6.xlsx",
    sha256="88426981ff35e8686d2184acebc0116a0890297fbbc8c296149bca1c525a9d4d",
    bytes=21699,
    role=ROLE_RAW_DATA,
    pmc_object="41467_2019_8888_MOESM6_ESM.xlsx",
    si_label="Supplementary Data 3",
)
VARIANTS_EXTRA = RawArtifact(
    name="si7.xlsx",
    library_relpath="si/si7.xlsx",
    sha256="6efdca5e32c55a2af76f742d7d8a59ac656f0003416711bc7a41b11bb163b18b",
    bytes=27916,
    role=ROLE_RAW_DATA,
    pmc_object="41467_2019_8888_MOESM7_ESM.xlsx",
    si_label="Supplementary Data 4",
)
SUPPLEMENTARY_PDF = RawArtifact(
    name="si1.pdf",
    library_relpath="si/si1.pdf",
    sha256="537dce71e89c465aef0e25f25c7b955fbc2ed6d51814e37b1b41e6af15cb3ec5",
    bytes=2906450,
    role=ROLE_SI_PDF,
    pmc_object="41467_2019_8888_MOESM1_ESM.pdf",
    si_label="Supplementary Information (Figs. 1-19, Tables 1-3)",
)

#: Every file either class links from the raw mirror.
RAW_FILES: tuple[RawArtifact, ...] = (
    SOURCE_DATA,
    VARIANTS_MS56,
    VARIANTS_MG1655,
    VARIANTS_EXTRA,
    SUPPLEMENTARY_PDF,
)
DATA_SHA256: dict[str, str] = {raw.name: raw.sha256 for raw in RAW_FILES}
RAW_BY_NAME: dict[str, RawArtifact] = {raw.name: raw for raw in RAW_FILES}

#: Supplementary Data 5, 6 and 7 are released and NOT mirrored here: no class in this
#: schema carries a ChIP-seq peak, a ribosome-protected-fragment RPKM or a translational
#: efficiency, and the RNA-seq RPKM cannot satisfy ``expression_count``.
NOT_MIRRORED: tuple[str, ...] = (
    "41467_2019_8888_MOESM4_ESM.xlsx",
    "41467_2019_8888_MOESM8_ESM.xlsx",
    "41467_2019_8888_MOESM9_ESM.xlsx",
    "41467_2019_8888_MOESM10_ESM.xlsx",
)

# --------------------------------------------------------------------------- #
# The Source Data sheet this loader reads
# --------------------------------------------------------------------------- #
GROWTH_SHEET = "Fig2c"
GROWTH_SHEET_TITLE = "Growth rate of MG1655, MS56, eMS57, and deletion strains"
RATIO_SHEET = "Fig2f"
RATIO_SHEET_TITLE = "Growth rate of MS56 after reconstruction of SNV"
#: The Fig. 2c sheet's own header row, verbatim and in order.
GROWTH_HEADERS: tuple[str, ...] = ("Strain", "Rep1", "Rep2", "Rep3")
#: The Fig. 2c strain labels, verbatim and in sheet order.
WILD_TYPE_LABEL = "MG1655"
PARENT_LABEL = "MS56"
EVOLVED_LABEL = "eMS57"
LARGE_DELETION_LABEL = "Δ21 kb"
RPOS_DELETION_LABEL = "ΔrpoS"
GROWTH_STRAIN_ORDER: tuple[str, ...] = (
    WILD_TYPE_LABEL,
    PARENT_LABEL,
    EVOLVED_LABEL,
    LARGE_DELETION_LABEL,
    RPOS_DELETION_LABEL,
)
#: The Fig. 2f row labels, verbatim and in sheet order (none is loadable).
RATIO_ROW_ORDER: tuple[str, ...] = (
    "eMS57/MS56",
    "ilvN*/ilvNWT",
    "cspC*/cspCWT",
    "yifB*/yifBWT",
)

#: The 21 gene symbols Supplementary Table 3's large-deletion genotype expands to, in
#: the order the genotype string writes them.
LARGE_DELETION_SYMBOLS: tuple[str, ...] = (
    "hycE",
    "hycD",
    "hycC",
    "hycB",
    "hycA",
    "hypA",
    "hypB",
    "hypC",
    "hypD",
    "hypE",
    "fhlA",
    "ygbA",
    "mutS",
    "pphB",
    "ygbI",
    "ygbJ",
    "ygbK",
    "ygbL",
    "ygbM",
    "ygbN",
    "rpoS",
)
RPOS_SYMBOL = "rpoS"
#: The replacement cassette both strains carry, verbatim from Supplementary Table 3.
KAN_CASSETTE = "kan"
#: The deletion method, one line, from the Methods section quoted below.
LAMBDA_RED_CONSTRUCTION = (
    "lambda recombination of MS56 with a kanamycin resistance cassette PCR amplified "
    "from pKD13, electroporated into MS56 carrying pKD46"
)
#: The MS56 parent and the two strains built on it, verbatim from Supplementary Table 3.
MS56_GENOTYPE_STATEMENT = "E. coli MG1655 with large deletions MD1 toMD56"
LARGE_DELETION_GENOTYPE_STATEMENT = (
    "MS56, (hycEDCBA-hypABCDE-fhlA-ygbA-mutS-pphB-ygbIJKLMN-rpoS)::kan"
)
RPOS_DELETION_GENOTYPE_STATEMENT = "MS56, rpoS::kan"

#: The Keio arm: the two BW25113 single deletions whose growth rate the Supplementary
#: Fig. 6 legend states, as a FRACTION of wild-type BW25113 in the same medium.
KEIO_COLLECTION = "single gene knockout collection (the Keio collection)"
KEIO_FITNESS: tuple[tuple[str, float], tuple[str, float]] = (
    ("ydaS", 0.724),
    ("abgR", 0.698),
)
#: Strains in the Supplementary Fig. 6 panel; only two carry a released number.
KEIO_PANEL_STRAINS = 62

#: Expected record counts, asserted at the end of each build.
GROWTH_EXPECTED_RECORDS = 2
GROWTH_EXPECTED_REFERENCES = 1
TF_EXPECTED_RECORDS = 2
TF_EXPECTED_REFERENCES = 1
#: Every deletion symbol must resolve; a renamed label stops the build.
MIN_RESOLVED_FRACTION = 1.0
#: The paper states the two deletion strains recover "to 80%" of eMS57; measured on the
#: released replicates they are 0.7763 and 0.7966, so 0.05 is the agreement window.
RECOVERY_TARGET = 0.80
RECOVERY_RTOL = 0.05
#: MS56 is "severe growth reduction in M9 minimal medium" relative to MG1655; measured,
#: its ratio is 0.3094, so the build refuses a sheet whose parent is not well below the
#: wild type (a swapped or re-exported row block).
MAX_PARENT_OVER_WILD_TYPE = 0.50
#: The two panels' eMS57/MS56 ratios disagree; the build records the factor and refuses
#: to proceed if they ever agree, which would mean the sheets changed meaning.
MIN_PANEL_DISAGREEMENT = 1.5

# --------------------------------------------------------------------------- #
# Verbatim quotes. Every one is a substring of the pinned ``paper.md`` or
# ``si/si1.md`` in the library mirror, OCR mangling included.
# --------------------------------------------------------------------------- #
_Q_MS56_DEFINITION = (
    "MS56 was created from the systematic deletion of 55 genomic regions of the "
    "wild-type E. coli MG1655. The 55 regions had a combined length of approximately "
    "1.1Mbp."
)
_Q_LARGE_DELETION = (
    "Notably, a large genomic region spanning $2 1 \\mathrm { k b }$ in length "
    "(genomic coordinates from 2,038,496 to 2,059,460 bp) was deleted spontaneously "
    "during ALE (Fig. 2a). The region contains 21 genes including rpoS"
)
_Q_RECOVERY = (
    "A single knockout of rpoS or deletion of the 21-kb region from MS56 recovered "
    "its growth rate to $8 0 \\%$ of that of eMS57; however, the deletion did not "
    "fully recover to the growth rate of eMS57 (Fig. 2c)."
)
_Q_CONSTRUCTION = (
    "Construction of MS56 Δ21kb and ΔrpoS strain. Target regions were "
    "knocked out using lambda recombination of $\\mathrm { M S } 5 6 ^ { 4 6 }$ . A "
    "kanamycin resistance cassette was PCR amplified from "
    "$\\mathsf { p K D 1 3 } ^ { 4 6 }$ using primers with homology to the target "
    "region."
)
_Q_MEDIA = (
    "Cells were grown in M9 glucose medium (47.75 $\\mathrm { m M }$ of "
    "${ \\mathrm { N a } } _ { 2 } { \\mathrm { H P O } } _ { 4 } ,$ , "
    "$2 2 . 0 4 \\mathrm { m M }$ of ${ \\mathrm { K H } } _ { 2 } "
    "{ \\mathrm { P O } } _ { 4 }$ , $8 . 5 6 \\mathrm { m M }$ of NaCl, "
    "$1 8 . 7 0 \\mathrm { m M }$ of $\\mathrm { N H _ { 4 } C l , }$ 2 mM of "
    "$\\mathrm { M g S O _ { 4 } }$ , $0 . 1 \\mathrm { m M }$ of "
    "$\\mathrm { C a C l } _ { 2 }$ , and $2 { \\bf g } 1 ^ { - 1 }$ of glucose), "
    "unless stated otherwise."
)
_Q_TRIPLICATED = (
    "All bacterial growth measurements were biologically triplicated and their "
    "differences were examined by two-sided t test of unequal variance "
    "(Welch’s t test)."
)
_Q_SEVERE_REDUCTION = (
    "Although MS56 exhibited a comparable growth rate to E. coli MG1655 in rich "
    "medium (Fig. 1a), it showed severe growth reduction in M9 minimal medium "
    "(Fig. 1b)."
)
_Q_MS56_ROW = (
    "MS56</td><td rowspan=1 colspan=1>E. coli MG1655 with large deletions MD1 "
    "toMD56</td><td rowspan=1 colspan=1>4</td>"
)
_Q_LARGE_DELETION_ROW = (
    "MS56 421 kb</td><td rowspan=1 colspan=1>MS56, (hycEDCBA-hypABCDE-fhlA-ygbA-"
    "mutS-pphB-ygbIJKLMN-rpoS)::kan</td>"
)
_Q_RPOS_ROW = "MS56 ΔrpoS</td><td rowspan=1 colspan=1>MS56, rpoS::kan</td>"
_Q_KEIO_PANEL = (
    "Supplementary Fig. 6. Growth curve of E. coli lacking one of the 62 "
    "transcription factors. Growth of 62 knock-out strains in M9 glucose medium was "
    "monitored in a 96-well plate on a Synergy H1 microplate reader (Bio-Tek). The "
    "plate was incubated at $3 7 ^ { \\circ } \\mathrm { C }$ with constant double "
    "orbital shaking $\\mathrm { ~ \\it ~ { ~ S ~ m m ~ } ~ }$ amplitude). WT: "
    "$E .$ coli K-12 strain BW25113. Deletion strains were obtained from single gene "
    "knockout collection (the Keio collection)."
)
_Q_KEIO_RATES = (
    "No strain showed significant growth retardation in M9 glucose medium, although "
    "the growth rates of the ΔydaS and ΔabgR strains were decreased to "
    "$7 2 . 4 \\%$ and $6 9 . 8 \\%$ of that of wild-type $E .$ . coli, respectively. "
    "Growth retardation of ΔydaS did not result from ydaS deletion1."
)
_Q_SIXTY_TWO_TFS = (
    "In addition, we would like to acknowledge that 63 transcription factors, composed "
    "of 28 families, were deleted in MS56."
)

PAPER = Provenance(
    source_uri=PAPER_MD, citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
)
SI1 = Provenance(source_uri=SI1_MD, citation_key=CITATION_KEY, sha256=SI1_MD_SHA256)


def _paper(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` quoting the pinned ``paper.md``."""
    return SourcedValue(value=value, provenance=PAPER, quote=quote, note=note)


def _si1(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` quoting the pinned ``si/si1.md``."""
    return SourcedValue(value=value, provenance=SI1, quote=quote, note=note)


SOURCED_VALUES: dict[str, SourcedValue] = {
    "parent_strain": _si1(PARENT_LABEL, _Q_MS56_ROW),
    "parent_genotype_statement": _si1(MS56_GENOTYPE_STATEMENT, _Q_MS56_ROW),
    "parent_regions_deleted": _paper(
        55,
        _Q_MS56_DEFINITION,
        note="the count and the 1.1 Mbp total are stated; no region's gene content is, "
        "so no allele of the background is typed",
    ),
    "parent_growth_reduction": _paper("severe", _Q_SEVERE_REDUCTION),
    "large_deletion_genotype": _si1(
        LARGE_DELETION_GENOTYPE_STATEMENT, _Q_LARGE_DELETION_ROW
    ),
    "rpos_deletion_genotype": _si1(RPOS_DELETION_GENOTYPE_STATEMENT, _Q_RPOS_ROW),
    "large_deletion_gene_count": _paper(
        21,
        _Q_LARGE_DELETION,
        note="the coordinates in the same sentence are MS56 coordinates, 806,266 bp "
        "upstream of where the same genes sit on MG1655, so they are not stored",
    ),
    "cassette": _paper(KAN_CASSETTE, _Q_CONSTRUCTION),
    "construction": _paper(LAMBDA_RED_CONSTRUCTION, _Q_CONSTRUCTION),
    "recovery_fraction_of_evolved": _paper(RECOVERY_TARGET, _Q_RECOVERY),
    "media_recipe": _paper("M9 glucose, 2 g/L glucose", _Q_MEDIA),
    "sample_unit": _paper(
        SampleUnit.biological_replicate.value,
        _Q_TRIPLICATED,
        note="one Fig. 2c Rep column is one independently grown culture",
    ),
    "keio_collection": _si1(KEIO_COLLECTION, _Q_KEIO_PANEL),
    "keio_reference_strain": _si1(BW25113, _Q_KEIO_PANEL),
    "keio_temperature_c": _si1(37.0, _Q_KEIO_PANEL),
    "keio_ydas_fitness": _si1(
        KEIO_FITNESS[0][1],
        _Q_KEIO_RATES,
        note="72.4% of wild-type BW25113 in the same M9 glucose medium; the legend's "
        "closing sentence states the retardation is not attributable to the ydaS "
        "deletion itself, which is a causal claim about the strain, not a correction "
        "to the measured rate",
    ),
    "keio_abgr_fitness": _si1(
        KEIO_FITNESS[1][1],
        _Q_KEIO_RATES,
        note="69.8% of wild-type BW25113 in the same M9 glucose medium",
    ),
    "keio_panel_size": _si1(KEIO_PANEL_STRAINS, _Q_KEIO_PANEL),
    "transcription_factors_deleted_in_parent": _paper(63, _Q_SIXTY_TWO_TFS),
}

_GAP_TEMPERATURE = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the paper states no cultivation temperature for the Fig. 2c growth-rate "
    "panel. Every temperature it does state is 37 C (the ALE flask, the pyruvate "
    "assay, the Supplementary Fig. 6 microplate), but none of those is this panel, so "
    "the field is left unset rather than carried across",
)
_GAP_CASSETTE_SEQUENCE = ProvenanceGap(
    field="cassette",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="Supplementary Table 3 writes the cassette as '::kan' and the Methods name "
    "pKD13, whose insert is FRT-kan-FRT; the paper never states which of the two the "
    "strain retains, so 'kan' is stored verbatim and the FRT flanks are not asserted",
)
_GAP_KEIO_REPLICATES = ProvenanceGap(
    field="n_samples",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the Supplementary Fig. 6 legend states two growth-rate percentages and no "
    "replicate count; the panel's curves are figure-only and the Source Data workbook "
    "holds no sheet for it",
)
_GAP_KEIO_UNCERTAINTY = ProvenanceGap(
    field="fitness_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="no dispersion is released for either percentage",
)
_GAP_KEIO_CONSTRUCTION = ProvenanceGap(
    field="construction",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="no plate, well or strain accession of either Keio strain is released",
)
#: ``BacterialDeletionPerturbation`` is not a ``ProvenanceGapMixin``, so a typed absence
#: on one of its fields is filed in ``preprocess/provenance_gaps.json`` instead.
GROWTH_PERTURBATION_GAPS: tuple[ProvenanceGap, ...] = (_GAP_CASSETTE_SEQUENCE,)
TF_PERTURBATION_GAPS: tuple[ProvenanceGap, ...] = (
    ProvenanceGap(
        field="cassette",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="the legend names the Keio collection but not its replacement cassette, "
        "which is a property of the collection stated by Baba 2006 and absent from "
        "this mirror",
    ),
    _GAP_KEIO_CONSTRUCTION,
)
#: The absence that has NO typed home, measured: ``BacterialStrainBackground.alleles``
#: defaults to ``[]``, and ``ProvenanceGapMixin`` refuses a gap on a field that is not
#: ``None``, so MS56's 55 unenumerated regions are recorded here and in the ledger.
BACKGROUND_ABSENCE = {
    "model": "BacterialStrainBackground",
    "field": "alleles",
    "strain": PARENT_LABEL,
    "why_no_typed_gap": "ProvenanceGapMixin requires a gapped field to be None; "
    "alleles defaults to [], so declaring the absence raises "
    "\"field 'alleles' has a ProvenanceGap but is not None\"",
    "what_is_missing": "the gene content of MS56's 55 deleted regions, ~1.1 Mbp",
    "where_it_lives_instead": "genotype_statement, verbatim from Supplementary Table 3",
    "source_defers_to": "reference 4 of this paper; MS56 has no deposited assembly",
}


# --------------------------------------------------------------------------- #
# The medium
# --------------------------------------------------------------------------- #
_MM = ConcentrationUnit.millimolar
_GL = ConcentrationUnit.g_per_l
_SALT = MediaComponentRole.bulk_salt


def _component(
    name: str, role: MediaComponentRole, value: float, unit: ConcentrationUnit
) -> MediaComponent:
    """One component of the M9 glucose medium, justified by the Methods recipe."""
    return MediaComponent(
        compound=resolved_compound(name),
        role=role,
        concentration=Concentration(value=value, unit=unit),
        provenance=[SOURCED_VALUES["media_recipe"]],
    )


M9_GLUCOSE_CHOE2019 = Media(
    name="M9 glucose (Choe 2019): 47.75 mM Na2HPO4, 22.04 mM KH2PO4, 8.56 mM NaCl, "
    "18.70 mM NH4Cl, 2 mM MgSO4, 0.1 mM CaCl2, 2 g/L glucose",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        _component("disodium hydrogen phosphate", _SALT, 47.75, _MM),
        _component("potassium dihydrogen phosphate", _SALT, 22.04, _MM),
        _component("sodium chloride", _SALT, 8.56, _MM),
        _component("ammonium chloride", MediaComponentRole.nitrogen_source, 18.70, _MM),
        _component("magnesium sulfate", _SALT, 2.0, _MM),
        _component("calcium chloride", _SALT, 0.1, _MM),
        _component("D-glucose", MediaComponentRole.carbon_source, 2.0, _GL),
    ],
    provenance=[SOURCED_VALUES["media_recipe"]],
)
"""Choe 2019's M9 glucose, built here because the library has no entry for this recipe."""


def growth_environment() -> Environment:
    """The Fig. 2c panel: M9 glucose, aerobic, temperature a declared gap."""
    return Environment(
        media=M9_GLUCOSE_CHOE2019,
        temperature=None,
        aerobicity="aerobic",
        provenance_gaps=[_GAP_TEMPERATURE],
    )


def keio_environment() -> Environment:
    """The Supplementary Fig. 6 panel: M9 glucose, 37 C, shaken 96-well plate."""
    return Environment(
        media=M9_GLUCOSE_CHOE2019,
        temperature=Temperature(value=SOURCED_VALUES["keio_temperature_c"].value),
        aerobicity="aerobic",
    )


def publication() -> Publication:
    """The paper every record cites."""
    return Publication(
        pubmed_id=PUBMED_ID,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
        doi=PAPER_DOI,
        doi_url=f"https://doi.org/{PAPER_DOI}",
    )


# --------------------------------------------------------------------------- #
# Reading the Source Data workbook
# --------------------------------------------------------------------------- #
class TableFormatError(RuntimeError):
    """The pinned workbook does not have the shape this module states."""


class StrainRates(BaseModel):
    """One Fig. 2c row: a strain's three released growth-rate replicates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: str = Field(description="row label, verbatim")
    row_number: int = Field(description="0-based row index in the sheet")
    replicates: tuple[float, ...]

    @property
    def n_samples(self) -> int:
        """Released replicate measurements of this strain's growth rate."""
        return len(self.replicates)

    @property
    def mean(self) -> float:
        """Mean of the released replicates."""
        return statistics.fmean(self.replicates)

    @property
    def sample_sd(self) -> float:
        """Sample (n-1) standard deviation of the released replicates."""
        return statistics.stdev(self.replicates)

    @property
    def standard_error(self) -> float:
        """Standard error of this strain's mean growth rate."""
        return self.sample_sd / math.sqrt(self.n_samples)


class GrowthPanel(BaseModel):
    """The Fig. 2c sheet: its five strain rows, keyed by their verbatim labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: tuple[StrainRates, ...]

    @property
    def by_strain(self) -> dict[str, StrainRates]:
        """``{strain label: rates}``."""
        return {row.strain: row for row in self.rows}

    def rates(self, strain: str) -> StrainRates:
        """One strain's released rates."""
        return self.by_strain[strain]


def _sheet_rows(path: str, sheet: str) -> list[list[Any]]:
    """Every row of one worksheet as a list of lists."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if sheet not in workbook.sheetnames:
        workbook.close()
        raise TableFormatError(f"{path}: no sheet {sheet!r}")
    worksheet = workbook[sheet]
    rows = [list(row) for row in worksheet.iter_rows(values_only=True)]
    workbook.close()
    return rows


def read_growth_panel(path: str) -> GrowthPanel:
    """Read the Fig. 2c sheet, checking its title, its headers and its strain order."""
    rows = _sheet_rows(path, GROWTH_SHEET)
    title = rows[0][1]
    if title != GROWTH_SHEET_TITLE:
        raise TableFormatError(f"{GROWTH_SHEET} title is {title!r}")
    headers = tuple(str(cell) for cell in rows[2][1:5])
    if headers != GROWTH_HEADERS:
        raise TableFormatError(f"{GROWTH_SHEET} headers are {headers}")
    parsed: list[StrainRates] = []
    for index, row in enumerate(rows[3:], start=3):
        if row[1] is None:
            continue
        replicates = tuple(float(cell) for cell in row[2:5])
        parsed.append(
            StrainRates(strain=str(row[1]), row_number=index, replicates=replicates)
        )
    order = tuple(row.strain for row in parsed)
    if order != GROWTH_STRAIN_ORDER:
        raise TableFormatError(f"{GROWTH_SHEET} strain order is {order}")
    return GrowthPanel(rows=tuple(parsed))


def read_reconstruction_ratios(path: str) -> dict[str, tuple[float, ...]]:
    """Read the Fig. 2f sheet: ``{row label: released ratios}``, for the ledger only."""
    rows = _sheet_rows(path, RATIO_SHEET)
    title = rows[0][1]
    if title != RATIO_SHEET_TITLE:
        raise TableFormatError(f"{RATIO_SHEET} title is {title!r}")
    out: dict[str, tuple[float, ...]] = {}
    for row in rows[3:]:
        if row[1] is None:
            continue
        out[str(row[1])] = tuple(float(cell) for cell in row[2:5])
    order = tuple(out)
    if order != RATIO_ROW_ORDER:
        raise TableFormatError(f"{RATIO_SHEET} row order is {order}")
    return out


class VariantTable(BaseModel):
    """One released variant table, read only so every refused call is typed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_name: str
    si_label: str
    rows: int
    distinct_genes: int
    intergenic_rows: int
    loci_with_several_calls: dict[str, int]
    frequency_columns: tuple[str, ...]


#: The three variant tables, with the column layout each one uses.
VARIANT_LAYOUTS: tuple[tuple[RawArtifact, int, int, int, int], ...] = (
    # (artifact, header rows, gene column, type column, first frequency column)
    (VARIANTS_MS56, 3, 0, 2, 6),
    (VARIANTS_MG1655, 3, 1, 4, 6),
    (VARIANTS_EXTRA, 4, 1, 4, 6),
)
#: A released cell that names no gene: Supplementary Data 2 numbers its intergenic
#: calls, Supplementary Data 3 writes a bare dash, Supplementary Data 4 writes the word.
_INTERGENIC_MARKERS = ("intergenic", "-")


def _is_intergenic(gene: str) -> bool:
    """True when a variant row's gene cell names no locus."""
    return gene.startswith("intergenic") or gene in _INTERGENIC_MARKERS


def read_variant_table(
    path: str, artifact: RawArtifact, *, header_rows: int, gene_column: int
) -> VariantTable:
    """Read one variant table's shape, discarding rows that name no variant.

    A variant row is one with BOTH a gene cell and a position cell; the workbooks carry
    stray single-cell rows below the table (Supplementary Data 2 holds a lone ``3``
    twelve rows past its last call) which a row-is-non-empty filter would count.
    """
    rows = _sheet_rows(path, "Sheet1")
    position_column = 1 if gene_column == 0 else 0
    body = [
        row
        for row in rows[header_rows:]
        if row[gene_column] is not None and row[position_column] is not None
    ]
    genes = [str(row[gene_column]) for row in body]
    counts: dict[str, int] = {}
    for gene in genes:
        if _is_intergenic(gene):
            continue
        counts[gene] = counts.get(gene, 0) + 1
    header = rows[header_rows - 1]
    return VariantTable(
        file_name=artifact.name,
        si_label=artifact.si_label,
        rows=len(body),
        distinct_genes=len(set(genes)),
        intergenic_rows=sum(1 for gene in genes if _is_intergenic(gene)),
        loci_with_several_calls={
            gene: n for gene, n in sorted(counts.items()) if n > 1
        },
        frequency_columns=tuple(
            str(cell) for cell in header[6:] if cell is not None and str(cell) != ""
        ),
    )


# --------------------------------------------------------------------------- #
# Cross-checks on the released numbers
# --------------------------------------------------------------------------- #
def check_recovery_fraction(panel: GrowthPanel) -> dict[str, Any]:
    """The two deletion strains reach the stated 80% of eMS57, on the released rates.

    The paper's own claim about this panel is the independent statistic: if a re-export
    reordered or rescaled the sheet, the prose would stop matching the bytes.
    """
    evolved = panel.rates(EVOLVED_LABEL).mean
    fractions = {
        label: panel.rates(label).mean / evolved
        for label in (LARGE_DELETION_LABEL, RPOS_DELETION_LABEL)
    }
    worst = max(abs(value - RECOVERY_TARGET) for value in fractions.values())
    if worst > RECOVERY_RTOL:
        raise TableFormatError(
            f"{GROWTH_SHEET}: deletion strains reach {fractions} of eMS57, which is "
            f"more than {RECOVERY_RTOL} from the stated {RECOVERY_TARGET}"
        )
    parent_ratio = panel.rates(PARENT_LABEL).mean / panel.rates(WILD_TYPE_LABEL).mean
    if parent_ratio > MAX_PARENT_OVER_WILD_TYPE:
        raise TableFormatError(
            f"{GROWTH_SHEET}: MS56 is {parent_ratio:.4f} of MG1655, which is not the "
            "severe reduction the paper states"
        )
    return {
        "stated_recovery_fraction_of_evolved": RECOVERY_TARGET,
        "measured_recovery_fraction_of_evolved": fractions,
        "tolerance": RECOVERY_RTOL,
        "parent_over_wild_type": parent_ratio,
        "max_parent_over_wild_type": MAX_PARENT_OVER_WILD_TYPE,
    }


def check_panels_disagree(
    panel: GrowthPanel, ratios: Mapping[str, Sequence[float]]
) -> dict[str, Any]:
    """Fig. 2c and Fig. 2f release different eMS57/MS56 ratios; record the factor.

    Both panels name the same strain pair, so a reader may assume they are the same
    measurement. They are not: the rates of Fig. 2c give 2.8159 and Fig. 2f releases
    1.3990. The build records the factor and refuses to proceed if the two ever agree,
    because that would mean one of the sheets changed meaning and the refusal of Fig. 2f
    would need rereading.
    """
    from_rates = panel.rates(EVOLVED_LABEL).mean / panel.rates(PARENT_LABEL).mean
    released = statistics.fmean(ratios[RATIO_ROW_ORDER[0]])
    factor = from_rates / released
    if factor < MIN_PANEL_DISAGREEMENT:
        raise TableFormatError(
            f"Fig. 2c gives eMS57/MS56 = {from_rates:.4f} and Fig. 2f releases "
            f"{released:.4f}; the factor {factor:.4f} is below the stated "
            f"{MIN_PANEL_DISAGREEMENT}, so the panels no longer disagree as measured"
        )
    return {
        "fig2c_evolved_over_parent_from_rates": from_rates,
        "fig2f_evolved_over_parent_released": released,
        "factor": factor,
        "min_disagreement": MIN_PANEL_DISAGREEMENT,
    }


def check_deleted_region(
    locus_tag: Mapping[str, str], genome: EcoliK12Genome
) -> dict[str, Any]:
    """The 21 named genes are a contiguous MG1655 run containing no unlisted gene.

    This is the cross-check that lets the gene list stand in for the released
    coordinates: Supplementary Table 3 names the genes and the Methods name a span, and
    on the pinned assembly the named genes span one uninterrupted block.
    """
    spans: dict[str, tuple[int, int]] = {}
    for feature in genome.db.features_of_type("gene"):
        tags = feature.attributes.get("locus_tag") or []
        if len(tags) == 1:
            spans[tags[0]] = (int(feature.start), int(feature.end))
    tags = [locus_tag[symbol] for symbol in LARGE_DELETION_SYMBOLS]
    missing = [tag for tag in tags if tag not in spans]
    if missing:
        raise TableFormatError(f"no gene feature for {missing}")
    low = min(spans[tag][0] for tag in tags)
    high = max(spans[tag][1] for tag in tags)
    inside = sorted(
        tag for tag, (start, end) in spans.items() if start >= low and end <= high
    )
    unlisted = [tag for tag in inside if tag not in set(tags)]
    if unlisted:
        raise TableFormatError(
            f"the MG1655 window {low}-{high} also contains {unlisted}, which "
            "Supplementary Table 3 does not list as deleted"
        )
    if len(inside) != len(LARGE_DELETION_SYMBOLS):
        raise TableFormatError(
            f"{len(inside)} genes inside the window, the source names "
            f"{len(LARGE_DELETION_SYMBOLS)}"
        )
    return {
        "named_symbols": list(LARGE_DELETION_SYMBOLS),
        "locus_tags": tags,
        "mg1655_window": [low, high],
        "mg1655_window_bp": high - low + 1,
        "released_ms56_window": [2038496, 2059460],
        "released_ms56_window_bp": 2059460 - 2038496 + 1,
        "ms56_to_mg1655_offset_bp": low - 2038496,
        "genes_inside_window": len(inside),
        "unlisted_genes_inside_window": 0,
        "note": "the released coordinates are MS56 coordinates, so they are not stored "
        "as a span against the pinned MG1655 assembly",
    }


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def parent_background() -> BacterialStrainBackground:
    """MS56, the genome-reduced parent both deletion strains were built on.

    ``alleles`` is empty and that is a statement, not an omission: this paper writes the
    genotype as "large deletions MD1 to MD56" and defers the content to its reference 4,
    MS56 has no deposited assembly, and ``AlleleEdit`` cannot say "55 regions somebody
    else listed". The verbatim genotype string is kept instead, and the untyped absence
    is filed in ``preprocess/genotype_gaps.json``.
    """
    return BacterialStrainBackground(
        name=PARENT_LABEL,
        reference_strain=MG1655,
        assembly_set=MG1655_ASSEMBLY_SET,
        parents=[WILD_TYPE_LABEL],
        construction="systematic deletion of 55 genomic regions of MG1655, about "
        "1.1 Mbp in total",
        genotype_statement=MS56_GENOTYPE_STATEMENT,
        alleles=[],
        provenance=[
            SOURCED_VALUES["parent_strain"],
            SOURCED_VALUES["parent_genotype_statement"],
            SOURCED_VALUES["parent_regions_deleted"],
        ],
    )


def fitness_phenotype(strain: StrainRates, parent: StrainRates) -> FitnessPhenotype:
    """One deletion strain's growth rate as a ratio to its isogenic parent MS56.

    ``fitness_uncertainty`` is the strain's sample standard deviation divided by the
    parent mean, which is exactly the sample standard deviation of the n released ratio
    observations, so the released statistic's kind and n survive the change of units.
    ``fitness_se`` is supplied as the delta-method standard error of the ratio, which
    also carries the parent's own spread and is therefore never the optimistic one.
    """
    fitness = strain.mean / parent.mean
    uncertainty = strain.sample_sd / parent.mean
    propagated = fitness * math.sqrt(
        (strain.standard_error / strain.mean) ** 2
        + (parent.standard_error / parent.mean) ** 2
    )
    conditioned = uncertainty / math.sqrt(strain.n_samples)
    if propagated < conditioned:
        raise RuntimeError(
            f"{strain.strain}: the propagated SE {propagated} is smaller than the "
            f"parent-conditioned {conditioned}"
        )
    return FitnessPhenotype(
        fitness=fitness,
        fitness_se=propagated,
        fitness_uncertainty=uncertainty,
        fitness_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=strain.n_samples,
        sample_unit=SampleUnit.biological_replicate,
    )


def parent_phenotype(parent: StrainRates) -> FitnessPhenotype:
    """MS56: the ratio of its own mean to itself, 1.0, with its own relative spread."""
    return FitnessPhenotype(
        fitness=1.0,
        fitness_uncertainty=parent.sample_sd / parent.mean,
        fitness_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=parent.n_samples,
        sample_unit=SampleUnit.biological_replicate,
    )


def build_growth_genotype(
    symbols: Sequence[str], locus_tag: Mapping[str, str]
) -> Genotype:
    """One strain's designed deletions, written against the pinned MG1655 assembly."""
    return Genotype(
        perturbations=[
            BacterialDeletionPerturbation(
                systematic_gene_name=locus_tag[symbol],
                perturbed_gene_name=symbol,
                gene_namespace=MG1655_NAMESPACE,
                identifier_mapping=DerivedIdentifierMapping(
                    source_identifier=symbol, route="gene_symbol"
                ),
                cassette=KAN_CASSETTE,
            )
            for symbol in symbols
        ]
    )


def build_keio_genotype(symbol: str, locus_tag: str) -> Genotype:
    """One Keio single deletion, written against the pinned BW25113 assembly."""
    return Genotype(
        perturbations=[
            BacterialDeletionPerturbation(
                systematic_gene_name=locus_tag,
                perturbed_gene_name=symbol,
                gene_namespace=BW25113_NAMESPACE,
                identifier_mapping=DerivedIdentifierMapping(
                    source_identifier=symbol, route="gene_symbol"
                ),
                collection=KEIO_COLLECTION,
            )
        ]
    )


def keio_phenotype(fitness: float) -> FitnessPhenotype:
    """One Keio strain's growth rate as the released fraction of wild-type BW25113."""
    return FitnessPhenotype(fitness=fitness, n_samples=None, sample_unit=None)


def resolve_symbols(
    symbols: Sequence[str], genome: EcoliK12Genome, *, label: str
) -> tuple[dict[str, str], LocusTagReconciliation]:
    """``{gene symbol: locus tag}`` plus the resolver report; nothing may be unresolved."""
    unique = sorted(set(symbols))
    stored, report = reconcile_locus_tags(genome, pd.Series(unique), label=label)
    report.require_resolved(MIN_RESOLVED_FRACTION)
    outside = set(report.outside_namespace)
    unresolved = [tag for tag in stored if tag in outside]
    if unresolved:
        raise RuntimeError(f"{label}: {unresolved} resolve to no locus of this strain")
    locus_tag = {symbol: str(tag) for symbol, tag in zip(unique, stored, strict=True)}
    if len(set(locus_tag.values())) != len(locus_tag):
        raise RuntimeError(f"{label}: two symbols share one locus tag")
    return locus_tag, report


# --------------------------------------------------------------------------- #
# The raw mirror (the loader reads the mirror, never a live URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/choeAdaptiveLaboratoryEvolution2019``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/choeAdaptiveLaboratoryEvolution2019``."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deposit_raw_mirror(data_root: str | None = None) -> Path:
    """Write the raw mirror from the library mirror's copies, plus ``manifest.json``.

    The library mirror already holds every file under the sha256 the PMC Article
    Datasets bucket served, with the retriever and its params recorded, so the raw
    mirror is written from those bytes rather than re-fetched. Idempotent by sha256: a
    mirror file with the pinned hash is left alone, one with any other hash raises.
    """
    library = library_dir(data_root)
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for raw in RAW_FILES:
        src = library / raw.library_relpath
        if not src.exists():
            raise RuntimeError(f"library mirror is missing {src}")
        got = _sha256(src)
        if got != raw.sha256:
            raise RuntimeError(f"{src} sha256 {got} != pinned {raw.sha256}")
        dest = root / raw.mirror_relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != raw.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        records.append(
            ArtifactRecord(
                path=raw.mirror_relpath,
                role=raw.role,
                bytes=dest.stat().st_size,
                sha256=raw.sha256,
                source=raw.source_url,
                retrieval=raw.retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=records,
        si_data_sources=[raw.source_url for raw in RAW_FILES],
        si_expected=list(NOT_MIRRORED),
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


def _link_raw(raw_dir: str, names: Sequence[str]) -> None:
    """Link each named mirror file into ``raw_dir`` after checking manifest + sha256."""
    data_root = _data_root()
    manifest = load_manifest(data_root)
    os.makedirs(raw_dir, exist_ok=True)
    for name in names:
        raw = RAW_BY_NAME[name]
        check_manifest_pin(
            raw.mirror_relpath,
            manifest_sha256(manifest, raw.mirror_relpath),
            raw.sha256,
        )
        src = raw_mirror_dir(data_root) / raw.mirror_relpath
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        link_verified(src, osp.join(raw_dir, raw.name), raw.sha256)


# --------------------------------------------------------------------------- #
# The designed-deletion dataset
# --------------------------------------------------------------------------- #
@register_dataset
class GrowthRateChoe2019Dataset(ExperimentDataset):
    """Choe 2019 Fig. 2c: the two MS56 designed deletions, as parent-relative fitness."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = MG1655

    def __init__(
        self,
        root: str = GROWTH_DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is injected by the build entry points."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialFitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialFitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Source Data workbook plus the three variant tables it refuses."""
        return [
            SOURCE_DATA.name,
            VARIANTS_MS56.name,
            VARIANTS_MG1655.name,
            VARIANTS_EXTRA.name,
        ]

    def _consumed_sha256(self) -> dict[str, str]:
        """``{name: sha256}`` of the files this loader reads."""
        return {name: DATA_SHA256[name] for name in self.raw_file_names}

    def download(self) -> None:
        """Link the mirrored files into ``raw/`` after checking manifest and sha256."""
        _link_raw(self.raw_dir, self.raw_file_names)
        log.info("Choe 2019 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    def _raw(self, name: str) -> str:
        return osp.join(self.raw_dir, name)

    @post_process
    def process(self) -> None:
        """Build the two designed-deletion fitness records and write the LMDB."""
        verify_raw_files(self.raw_dir, self._consumed_sha256())
        source = self._raw(SOURCE_DATA.name)
        panel = read_growth_panel(source)
        ratios = read_reconstruction_ratios(source)
        recovery = check_recovery_fraction(panel)
        disagreement = check_panels_disagree(panel, ratios)
        variants = tuple(
            read_variant_table(
                self._raw(artifact.name),
                artifact,
                header_rows=header_rows,
                gene_column=gene_column,
            )
            for artifact, header_rows, gene_column, _type_column, _first in (
                VARIANT_LAYOUTS
            )
        )

        genome = self._genome()
        locus_tag, report = resolve_symbols(
            LARGE_DELETION_SYMBOLS, genome, label=f"{self.name} deleted gene symbols"
        )
        region = check_deleted_region(locus_tag, genome)
        environment = growth_environment()
        reference_genome = assembly_reference(
            self.REFERENCE_STRAIN, background=parent_background()
        )
        parent = panel.rates(PARENT_LABEL)
        reference = BacterialFitnessExperimentReference(
            dataset_name=self.name,
            genome_reference=reference_genome,
            environment_reference=environment,
            phenotype_reference=parent_phenotype(parent),
        )
        pub = publication()
        strains: tuple[tuple[str, tuple[str, ...]], ...] = (
            (LARGE_DELETION_LABEL, LARGE_DELETION_SYMBOLS),
            (RPOS_DELETION_LABEL, (RPOS_SYMBOL,)),
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        rows: list[dict[str, Any]] = []
        index = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for label, symbols in tqdm(strains, desc="choe2019 growth rate"):
                rates = panel.rates(label)
                phenotype = fitness_phenotype(rates, parent)
                experiment = BacterialFitnessExperiment(
                    dataset_name=self.name,
                    genotype=build_growth_genotype(symbols, locus_tag),
                    environment=environment,
                    phenotype=phenotype,
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                index += 1
                rows.append(
                    {
                        "strain": label,
                        "deleted_genes": len(symbols),
                        "locus_tags": ";".join(locus_tag[s] for s in symbols),
                        "growth_rate_replicates": ";".join(
                            f"{value}" for value in rates.replicates
                        ),
                        "growth_rate_mean": rates.mean,
                        "parent_growth_rate_mean": parent.mean,
                        "n_samples": rates.n_samples,
                        "fitness": phenotype.fitness,
                        "fitness_uncertainty": phenotype.fitness_uncertainty,
                        "fitness_se": phenotype.fitness_se,
                    }
                )
        env.close()
        interned_env.close()

        if index != GROWTH_EXPECTED_RECORDS:
            raise RuntimeError(
                f"{index} records, the module states {GROWTH_EXPECTED_RECORDS}"
            )
        self._write_ledgers(
            panel, ratios, rows, report, recovery, disagreement, region, variants
        )
        log.info(
            "Choe 2019 Fig. 2c: %d designed-deletion records (+ %d MS56 reference); "
            "fitness %.6f to %.6f",
            index,
            GROWTH_EXPECTED_REFERENCES,
            min(row["fitness"] for row in rows),
            max(row["fitness"] for row in rows),
        )

    def _write_ledgers(
        self,
        panel: GrowthPanel,
        ratios: Mapping[str, Sequence[float]],
        rows: Sequence[Mapping[str, Any]],
        report: LocusTagReconciliation,
        recovery: Mapping[str, Any],
        disagreement: Mapping[str, Any],
        region: Mapping[str, Any],
        variants: Sequence[VariantTable],
    ) -> None:
        """The retention ledger, the refused calls, the gaps and the cross-checks."""
        out = Path(self.preprocess_dir)
        total_calls = sum(table.rows for table in variants)
        (out / "dropped_records.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "source_rows": len(panel.rows),
                    "kept_records": len(rows),
                    "reference_rows": [PARENT_LABEL],
                    "dropped_records": len(panel.rows) - len(rows) - 1,
                    "rules": [
                        {
                            "strain": EVOLVED_LABEL,
                            "reason": "evolved_clone_has_no_writable_genotype",
                            "detail": "eMS57 is a clone isolated from the day-62 "
                            "population and the release gives it no variant list; a "
                            "genotype would have to be produced by thresholding "
                            "population allele frequencies, a call the authors never "
                            "made, and no perturbation leaf holds a called variant",
                        },
                        {
                            "strain": PARENT_LABEL,
                            "reason": "parent_is_the_reference_not_a_record",
                            "detail": "MS56 is the isogenic parent both deletion "
                            "strains were built on, so it is the phenotype_reference "
                            "the ratio is taken against",
                        },
                        {
                            "strain": WILD_TYPE_LABEL,
                            "reason": "parent_genotype_not_enumerated_by_this_paper",
                            "detail": "an MG1655-referenced MS56 record would need "
                            "MS56's 55 deleted regions as perturbations; this paper "
                            "states only 'large deletions MD1 to MD56' and defers the "
                            "content to its reference 4, so the record would assert "
                            "that MS56 is genotypically MG1655",
                        },
                    ],
                    "notes": [
                        f"{len(panel.rows)} released Fig. 2c rows = {len(rows)} records "
                        f"+ 1 MS56 reference + 2 refused strains",
                        f"Fig. 2f releases three reconstructed single-allele strains "
                        f"({', '.join(RATIO_ROW_ORDER[1:])}) as paired "
                        "mutant-over-parent growth-rate ratios. Every one is a DESIGNED "
                        "strain with an exactly specified allele (cspC G37A, ilvN "
                        "C202T, yifB 1169C insertion) and none can be written: the "
                        "perturbation axis has no bacterial sequence-variant leaf "
                        "(issue #731), SequenceVariantPerturbation refuses a b-number "
                        "(issue #749), and BacterialBackgroundAllele requires a "
                        "non-optional functional flag the paper does not determine for "
                        "two of the three",
                        f"the three variant tables release {total_calls} calls, every "
                        "one a population allele frequency; each is typed in "
                        "called_variants.json with its blocking reasons",
                        "Supplementary Data 6 (RNA-seq) and 7 (Ribo-seq) are not "
                        "mirrored: RNASeqExpressionPhenotype requires a per-gene "
                        "integer expression_count and the release gives RPKM only, "
                        "which does not back-solve; translation level and translational "
                        "efficiency have no phenotype class",
                        "Supplementary Data 1 (Biolog phenotype microarray, 384 wells) "
                        "and Supplementary Data 5 (839 ChIP-seq peaks) are not "
                        "mirrored: an Omnilog respiration endpoint has no "
                        "MeasurementType member and a binding peak has no class",
                    ],
                    "reconciliation": report.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        (out / "called_variants.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "total_calls": total_calls,
                    "carrier": None,
                    "blocking_reasons": [
                        "population_allele_frequency_not_a_clone_genotype",
                        "no_perturbation_leaf_holds_a_bacterial_sequence_variant",
                        "sequence_variant_perturbation_refuses_a_locus_tag",
                        "bacterial_background_allele_requires_functional_bool",
                        "background_permits_one_allele_entry_per_locus",
                        "intergenic_call_has_no_locus_to_key_to",
                    ],
                    "tables": [table.model_dump(mode="json") for table in variants],
                },
                indent=2,
            )
        )
        (out / "genotype_gaps.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "untyped_absences": [BACKGROUND_ABSENCE],
                    "typed_gaps": {
                        "environment": [_GAP_TEMPERATURE.model_dump(mode="json")],
                        "perturbation": [
                            gap.model_dump(mode="json")
                            for gap in GROWTH_PERTURBATION_GAPS
                        ],
                    },
                    "note": "BacterialDeletionPerturbation is not a ProvenanceGapMixin, "
                    "so a typed absence on one of its fields is filed here",
                },
                indent=2,
            )
        )
        (out / "released_statistics_check.json").write_text(
            json.dumps(
                {
                    "recovery_fraction": dict(recovery),
                    "panel_disagreement": dict(disagreement),
                    "deleted_region": dict(region),
                    "fig2c_strain_means": {row.strain: row.mean for row in panel.rows},
                    "fig2f_released_ratio_means": {
                        label: statistics.fmean(values)
                        for label, values in ratios.items()
                    },
                },
                indent=2,
            )
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {
                    name: value.model_dump(mode="json")
                    for name, value in SOURCED_VALUES.items()
                },
                indent=2,
            )
        )
        pd.DataFrame(list(rows)).to_csv(out / "strains.csv", index=False)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "GrowthRateChoe2019Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# The Keio transcription-factor dataset
# --------------------------------------------------------------------------- #
@register_dataset
class TranscriptionFactorKnockoutChoe2019Dataset(ExperimentDataset):
    """Choe 2019 Supplementary Fig. 6: the two Keio deletions that carry a number."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = BW25113

    def __init__(
        self,
        root: str = TF_DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is injected by the build entry points."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialFitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialFitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Supplementary Information PDF whose Fig. 6 legend states both numbers.

        The PDF is pinned and sha256-verified rather than parsed: both values are
        ``SourcedValue``s quoting its OCR, which is the sanctioned shape for a number
        that lives in prose, and reading the OCR at build time is explicitly not what
        ``SourcedValue`` is for.
        """
        return [SUPPLEMENTARY_PDF.name]

    def _consumed_sha256(self) -> dict[str, str]:
        """``{name: sha256}`` of the files this loader pins."""
        return {name: DATA_SHA256[name] for name in self.raw_file_names}

    def download(self) -> None:
        """Link the mirrored SI PDF into ``raw/`` after checking manifest and sha256."""
        _link_raw(self.raw_dir, self.raw_file_names)
        log.info("Choe 2019 SI PDF linked into %s (sha256 verified)", self.raw_dir)

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Build the two Keio fitness records and write the LMDB."""
        verify_raw_files(self.raw_dir, self._consumed_sha256())
        genome = self._genome()
        symbols = [symbol for symbol, _ in KEIO_FITNESS]
        locus_tag, report = resolve_symbols(
            symbols, genome, label=f"{self.name} deletion symbols"
        )
        environment = keio_environment()
        reference = BacterialFitnessExperimentReference(
            dataset_name=self.name,
            genome_reference=assembly_reference(self.REFERENCE_STRAIN),
            environment_reference=environment,
            phenotype_reference=FitnessPhenotype(
                fitness=1.0, n_samples=None, sample_unit=None
            ),
        )
        pub = publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        rows: list[dict[str, Any]] = []
        index = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for symbol, fitness in tqdm(KEIO_FITNESS, desc="choe2019 TF knockouts"):
                phenotype = keio_phenotype(fitness)
                experiment = BacterialFitnessExperiment(
                    dataset_name=self.name,
                    genotype=build_keio_genotype(symbol, locus_tag[symbol]),
                    environment=environment,
                    phenotype=phenotype,
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                index += 1
                rows.append(
                    {
                        "strain": f"Δ{symbol}",
                        "gene_symbol": symbol,
                        "locus_tag": locus_tag[symbol],
                        "fitness": phenotype.fitness,
                    }
                )
        env.close()
        interned_env.close()

        if index != TF_EXPECTED_RECORDS:
            raise RuntimeError(
                f"{index} records, the module states {TF_EXPECTED_RECORDS}"
            )
        self._write_ledgers(rows, report)
        log.info(
            "Choe 2019 Supplementary Fig. 6: %d Keio records (+ %d BW25113 reference) "
            "out of a %d-strain panel",
            index,
            TF_EXPECTED_REFERENCES,
            KEIO_PANEL_STRAINS,
        )

    def _write_ledgers(
        self, rows: Sequence[Mapping[str, Any]], report: LocusTagReconciliation
    ) -> None:
        """The retention ledger, the typed gaps and the sourcing table."""
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "source_rows": KEIO_PANEL_STRAINS,
                    "kept_records": len(rows),
                    "reference_rows": [BW25113],
                    "dropped_records": KEIO_PANEL_STRAINS - len(rows),
                    "rules": [
                        {
                            "strains": KEIO_PANEL_STRAINS - len(rows),
                            "reason": "panel_releases_no_number_for_them",
                            "detail": "the legend states that no strain showed "
                            "significant growth retardation and names a percentage for "
                            "two of them; the other curves are figure-only and the "
                            "Source Data workbook holds no sheet for this panel. The "
                            "62 deleted transcription factors are not listed either, "
                            "so the unnamed strains could not be keyed to a locus even "
                            "if a number existed",
                        },
                        {
                            "strains": 1,
                            "reason": "dica_not_in_the_collection",
                            "detail": "the legend states the dicA single knockout was "
                            "not tested because only the double knockout is viable",
                        },
                    ],
                    "notes": [
                        "the legend's closing sentence states that the growth "
                        "retardation of the ydaS deletion strain does not result from "
                        "the ydaS deletion, citing Bindal 2017 on RacR; that is a "
                        "causal claim about the strain, so the measured rate is stored "
                        "and the claim is recorded here rather than altering the value",
                        "no replicate count, dispersion or absolute wild-type rate is "
                        "released for this panel, so n_samples, sample_unit and the "
                        "uncertainty fields are typed gaps",
                    ],
                    "reconciliation": report.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        (out / "provenance_gaps.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "carrier": "FitnessPhenotype / BacterialDeletionPerturbation",
                    "note": "neither leaf is a ProvenanceGapMixin, so these typed "
                    "absences are recorded here rather than on the record",
                    "gaps": [
                        gap.model_dump(mode="json")
                        for gap in (
                            _GAP_KEIO_REPLICATES,
                            _GAP_KEIO_UNCERTAINTY,
                            *TF_PERTURBATION_GAPS,
                        )
                    ],
                },
                indent=2,
            )
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {
                    name: value.model_dump(mode="json")
                    for name, value in SOURCED_VALUES.items()
                    if name.startswith("keio") or name.startswith("media")
                },
                indent=2,
            )
        )
        pd.DataFrame(list(rows)).to_csv(out / "strains.csv", index=False)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "TranscriptionFactorKnockoutChoe2019Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# L0-L4 verification
# --------------------------------------------------------------------------- #
def _verify(
    dataset_root: str,
    data_root: str | None,
    *,
    strain: EcoliK12StrainName,
    genome: EcoliK12Genome | None,
    expected_count: int,
    provenance: Provenance,
) -> VerificationReport:
    """Run the fitness L0-L4 gate over one built tree and write its report."""
    from torchcell.verification.fitness import verify_fitness_dataset
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    if genome is None:
        genome = bacterial_genome("ecoli", strain, data_root)
    report = verify_fitness_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=provenance,
        expected_count=expected_count,
        sgd_genes=set(genome.genbank.loci),
        gene_universe_label=f"{strain} ({genome.ASSEMBLY_SET})",
        resolve_gene_name=genome.resolve_gene_name,
    )
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def verify_growth_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    genome: EcoliK12Genome | None = None,
    expected_count: int = GROWTH_EXPECTED_RECORDS,
) -> VerificationReport:
    """L0-L4 over the designed-deletion tree."""
    return _verify(
        dataset_root,
        data_root,
        strain=MG1655,
        genome=genome,
        expected_count=expected_count,
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{SOURCE_DATA.mirror_relpath}",
            citation_key=CITATION_KEY,
            sha256=SOURCE_DATA.sha256,
            method=(
                "Source Data sheet 'Fig2c' Rep1-Rep3 growth rates: one "
                "BacterialFitnessExperiment per designed MS56 deletion strain, "
                "fitness = mean(strain replicates) / mean(MS56 replicates), "
                "fitness_se = the delta-method SE of that ratio"
            ),
            page=f"si12.xlsx sheets '{GROWTH_SHEET}' and '{RATIO_SHEET}'",
            retrieved=RETRIEVED_AT,
        ),
    )


def verify_tf_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    genome: EcoliK12Genome | None = None,
    expected_count: int = TF_EXPECTED_RECORDS,
) -> VerificationReport:
    """L0-L4 over the Keio transcription-factor tree."""
    return _verify(
        dataset_root,
        data_root,
        strain=BW25113,
        genome=genome,
        expected_count=expected_count,
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{LIBRARY_DIR_REL}/{SI1_MD}",
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method=(
                "the Supplementary Fig. 6 legend's two stated growth-rate percentages "
                "of wild-type BW25113, stored as the fitness ratio they are"
            ),
            page="Supplementary Fig. 6 legend",
            retrieved=RETRIEVED_AT,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: deposit the raw mirror, build a dev-tree LMDB, or verify a built one."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.choe2019_growth_rate"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("deposit", help="write the raw mirror from the library mirror")
    for name in ("build", "verify"):
        command = sub.add_parser(name, help=f"{name} a dev-tree LMDB")
        command.add_argument("--arm", choices=("growth", "tf", "both"), default="both")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    if args.command == "deposit":
        print(f"raw mirror at {deposit_raw_mirror(data_root)}")
        return 0

    arms: list[tuple[str, str, Any, Any]] = []
    if args.arm in ("growth", "both"):
        arms.append(
            (
                "growth",
                GROWTH_DATASET_ROOT_REL,
                GrowthRateChoe2019Dataset,
                verify_growth_build,
            )
        )
    if args.arm in ("tf", "both"):
        arms.append(
            (
                "tf",
                TF_DATASET_ROOT_REL,
                TranscriptionFactorKnockoutChoe2019Dataset,
                verify_tf_build,
            )
        )

    status = 0
    for label, root_rel, klass, verifier in arms:
        root = osp.join(data_root, root_rel)
        if args.command == "build":
            dataset = klass(root=root)
            print(f"{label}: len = {len(dataset)}")
            print(Path(root, "preprocess", "strains.csv").read_text())
            dataset.close_lmdb()
            continue
        report = verifier(root, data_root)
        print(f"==== {label} ====")
        print(report.summary())
        if not report.passed:
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
