# torchcell/datasets/ecoli/choe2019_growth_rate
# [[torchcell.datasets.ecoli.choe2019_growth_rate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/choe2019_growth_rate
# Test file: tests/torchcell/datasets/ecoli/test_choe2019_growth_rate.py
"""Choe 2019 genome-reduced ALE: every released growth row, each strain as released.

Choe et al. 2019 (Nat Commun 10:935, doi:10.1038/s41467-019-08888-6) evolved the
genome-reduced *E. coli* MS56 for 807 generations in M9 glucose, resequenced the
population at 20 timepoints, isolated the clone eMS57, and profiled it against MG1655 by
RNA-seq, Ribo-seq, ChIP-seq, a Biolog phenotype microarray and intracellular metabolites.
Four dataset classes serve what is BOTH released as a number AND keyed to a strain
whose genotype this schema can write:

- :class:`GrowthRateChoe2019Dataset` -- ``FitnessExperiment``, twelve records, one per
  released Fig. 2f set (#1175, 2026-11-24): the three reconstructed single-allele strains
  (``cspCmut``, ``ilvNmut``, ``yifBmut``, #1057), each against its own kan-marked
  wild-type-allele control, and the evolved clone eMS57 over MS56 (#1045 part 3), each
  ratio stored as the sheet releases it for ``Set1``, ``Set2`` and ``Set3``.
- :class:`AbsoluteGrowthRateChoe2019Dataset` -- ``EnvironmentResponseExperiment``,
  29 records, one per released culture (#1175, 2026-11-24): the Fig. 2c growth rates of
  MG1655, MS56, eMS57, ``MS56 Delta21kb`` (21 gene deletions) and ``MS56 DeltarpoS``
  (one), three cultures each, and the Fig. 2g growth rates of MG1655 and eMS57 in M9,
  M9 + valine and M9 + valine + isoleucine, every numeric cell of that sheet.
- :class:`MetaboliteChoe2019Dataset` -- ``MetaboliteExperiment``, 38 records, one per
  released culture cell (#1175, 2026-11-24): Fig. 1g's intracellular and extracellular
  pyruvate of two MG1655 and two eMS57 cultures at 4, 6 and 8 h after inoculation, and
  Fig. 3d's NADH/NAD+ and NADPH/NADP+ ratios (two cultures per strain) and ATP (three).
- :class:`TranscriptionFactorKnockoutChoe2019Dataset` --
  ``FitnessExperiment``, two records: the Keio BW25113 ``DeltaydaS`` and
  ``DeltaabgR`` single deletions, whose growth rates the Supplementary Fig. 6 legend
  states as a percentage of wild-type BW25113 in the same medium.

ONE RECORD PER RELEASED CULTURE (#1175, #1178, 2026-11-24). "All bacterial growth
measurements were biologically triplicated", and the Fig. 2c legend calls the three
values "three individual cultures shown in red circles". Until 2026-11-24 the Fig. 2c
rows were stored as mean(strain) / mean(parent) with a sample SD and a delta-method SE,
and the Fig. 2f rows as the mean of the three released ratios: eight loader-computed
aggregates. Now each released value is its own record with ``n_samples = 1``,
``replicate_id`` = the sheet's own column label (``Rep1``-``Rep3``, ``Set1``-``Set3``)
and ``not_applicable_single_measurement`` gaps on the dispersion fields. A Fig. 2c value
is an ABSOLUTE growth rate, so it is no longer divided by a loader-computed parent mean:
it is an ``EnvironmentResponsePhenotype`` of ``MeasurementType.growth_rate`` with the
released number verbatim, and the strain-over-parent ratio is a modeling-time
derivation over the records. A Fig. 2f value is a ratio the SOURCE releases per set, so
it stays a ``FitnessPhenotype`` ratio, one record per set. Each reference states an
aggregate: the background strain's own released cultures in the panel's base medium,
with their mean, sample SD and n, and ``replicate_id = None``.

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
   ``BackgroundAllele.deleted_span`` is deletion-only and would lose all of it.
3. ``BackgroundAllele`` requires a non-optional ``functional: bool``
   (``Input should be a valid boolean``), so the unknown consequence of a missense SNV
   cannot be covered by a ``ProvenanceGap`` -- a gap must name a field that is ``None``.
4. ``StrainBackground`` permits ONE allele entry per locus, which refuses the
   seven loci carrying more than one call in Supplementary Data 2 alone (``yafF`` 5,
   ``ydjN`` 4, ``iscR`` 3, ``rrsH`` / ``tufA`` / ``yeaM`` 2 each, plus 13 rows labelled
   only ``intergenic``).
5. 22 of the 117 rows are intergenic and name no locus at all, so there is nothing to key
   to; inventing a flanking gene is exactly what must not be done.
6. ``Genotype.__eq__`` compares the perturbation SET, so a clone written with its
   parent's perturbations only would be genotype-identical to its parent.

THE DESIGNED POINT MUTANTS ARE RECORDS (#1045 row 22, #1057). Supplementary Table 3
releases three strains the authors BUILT on MS56, each a single exactly specified allele
with its own isogenic kan-marked wild-type-allele control: ``cspCmut``
(``MS56, cspC::cspC(G37A)-kanR``) against ``cspCWT`` (``MS56, cspC::cspC-kan``),
``ilvNmut`` (``MS56, ilvN::ilvN(C202T)-kanR``) against ``ilvNWT``, and ``yifBmut``
(``MS56, yifB::yifB(1169C insertion)-kanR``) against ``yifBWT``. The Fig. 2f sheet
releases each mutant-over-control growth-rate ratio for three sets. Each is stored as
released, on ``ReleasedIdentifierPerturbation`` (``released_kind=sequence_allele``):
the gene symbol and the allele exactly as Supplementary Table 3 writes them, the strain
name, the marker as written, and no coordinate, because the release gives a CDS-relative
change and no replicon position. The reference of each is its control, a
``StrainBackground`` named as released whose ``genotype_statement`` is its
Supplementary Table 3 row; each record is ONE released set's ratio (n = 1, the Fig. 2f
legend's "three biological replicates"), and the reference is the control's 1.0.

A SECOND REASON NOT TO MIX THE TWO GROWTH PANELS, measured here: Fig. 2c's rates give
eMS57/MS56 = 2.8159, while Fig. 2f releases eMS57/MS56 directly as 1.3990 (mean of three
sets). The two panels disagree by a factor of 2.013 on the same strain pair, so they are
not the same measurement and nothing may be derived across them. Both are stored (#1045
part 3): the Fig. 2c cultures as absolute rates in
:class:`AbsoluteGrowthRateChoe2019Dataset` (``screen_id = "Fig2c"``), the Fig. 2f sets as
released ratios in :class:`GrowthRateChoe2019Dataset`. Fig. 2g's M9 arm disagrees with
Fig. 2c the same way (MG1655 0.30-0.31 against 0.58-0.60, eMS57 0.29-0.33 against
0.51), so its records carry ``screen_id = "Fig2g"`` and keep the two panels apart in
one environment.

THE EVOLVED CLONE IS A RECORD ON ITS RELEASED BACKGROUND (#1045 part 3). eMS57 is a
``StrainBackground`` named ``eMS57`` whose ``genotype_statement`` is its
Supplementary Table 3 row verbatim ("E. coli MS56 adaptively evolved in M9
glucosemedium"), parent MS56, and whose ``alleles`` list is empty for the reason below:
the release gives the clone no variant list. Its genotype is empty, because nothing is
edited on top of it; the strain is the background the reference genome names. Fig. 2c is internally
consistent with the paper's own prose instead: it puts ``Delta21kb`` at 0.7763 and
``DeltarpoS`` at 0.7966 of eMS57, against a stated "recovered its growth rate to 80% of
that of eMS57", and ``check_recovery_fraction`` asserts that agreement on every build.

WHY THE REFERENCE IS MS56 AND NOT MG1655. Both deletion strains are "MS56, <lesion>::kan"
(Supplementary Table 3), so their isogenic parent is MS56, and their reference is MS56's
own released Fig. 2c cultures on MS56's background: the record differs from its
reference by exactly the deletions. MS56 itself is carried as a
``StrainBackground`` whose ``genotype_statement`` is Supplementary Table 3's own
words and whose ``alleles`` list is EMPTY, which is the same statement the landed Girgis
2009 loader makes for MG1655 ``delta-lacZ``: this paper defers MS56's content to its
reference 4 as "large deletions MD1 to MD56" and enumerates no gene, the reduced strain
has no deposited assembly, and typing 55 regions nobody released would be an invention.
Two consequences are stated rather than typed:

- MS56's own Fig. 2c cultures are records (#1045 part 3) on MS56's named background,
  not an empty genotype against a bare MG1655 reference: the record's strain is the
  background its reference genome names, so it does not assert that MS56 is
  genotypically MG1655, and the 55 regions stay in the statement. The same holds for
  eMS57 on its background and for MG1655 on the bare assembly; each such record's
  reference is its own strain's released cultures in the panel's base medium.
- The background's unenumerated content has no TYPED home either. A ``ProvenanceGap`` on
  ``alleles`` is refused by ``ProvenanceGapMixin`` (``field 'alleles' has a
  ProvenanceGap but is not None``), because the field defaults to ``[]`` rather than
  ``None``. The absence therefore lives in ``preprocess/genotype_gaps.json``, in the
  note and here. That refusal is measured, not read off the class.

THE 21 DELETED GENES ARE THE SOURCE'S OWN LIST, AND THE RELEASED COORDINATES ARE NOT
USABLE. Supplementary Table 3 writes the large deletion as
``(hycEDCBA-hypABCDE-fhlA-ygbA-mutS-pphB-ygbIJKLMN-rpoS)::kan``, which expands to exactly
21 gene symbols. Each is stored as released (#1045 part 2): a
``ReleasedIdentifierPerturbation`` of ``released_kind=deletion`` keyed by the
symbol, the panel's strain label as the strain label; the loader reads no annotation.
What the pinned MG1655 assembly makes of them is the verification's genome-reference
row :data:`DELETED_REGION_RULE` (a finding, never a gate): measured, all 21 resolve to
the CONTIGUOUS run ``b2721`` to ``b2741`` and no unlisted gene lies inside that window
(:func:`check_deleted_region`). The paper also gives coordinates, "genomic coordinates
from 2,038,496 to 2,059,460 bp", and those are MS56 coordinates: the same genes sit at
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
- Fig. 1g's OD600 row (12 cells, one per culture and timepoint) is a growth-curve
  point, which no phenotype class holds; its pyruvate rows are records. Supplementary
  Table 1 releases a fed-batch biomass and specific growth rate for MG1655 and eMS57 as
  mean +- s.d. over a fermentation "repeated twice in different days"; that aggregate
  is not loaded yet (a different culture from the Fig. 2c flasks, and a biomass yield has
  no MeasurementType member).
- METABOLITE RECORDS (#1175, 2026-11-24). Each Fig. 1g and Fig. 3d cell is one record:
  the level verbatim, ``n_replicates = 1``, ``replicate_id`` the sheet's own label
  (``MG1655 Rep 1`` ... ``eMS57 Rep2`` for a Fig. 1g block, ``Rep1``-``Rep3`` for a
  Fig. 3d column), and ``measurement_type`` naming the compartment and unit
  (``intracellular_pyruvate_umol_per_1e9_cells``, ``extracellular_pyruvate_uM``,
  ``intracellular_atp_nmol_per_1e9_cells``) or the released ratio
  (``intracellular_nadh_to_nad_ratio``, ``intracellular_nadph_to_nadp_ratio``, keyed by
  the row label ``NADH/NAD+`` and ``NADPH/NADP+``). The three timepoints of one Fig. 1g
  culture are three environments (``duration_hours``). Fig. 3d's legend states "three
  biological replicates", and its sheet releases three ATP cells but two cells per ratio
  row per strain; every released cell is stored and none is invented.
  ``check_cofactor_fold_changes`` asserts that the two ratio rows reproduce the prose's
  "3.0 times lower" and "4.5 times higher" (measured 2.965 and 4.570).
- Fig. 2g releases a dash, not a number, in four cells: MG1655's M9 ``Rep3`` and all
  three of MG1655's M9 + valine cells (the paper: valine "inhibits cell growth" and
  only eMS57 "was completely resistant"). A dash is no released value, so those four
  cells are counted in the ledger and are not records; the other 14 cells are.

DATA. ``GrowthRateChoe2019Dataset`` consumes ``si12.xlsx`` (the Source Data workbook) and
the three variant tables ``si5.xlsx``, ``si6.xlsx`` and ``si7.xlsx``, which it reads only
to type each refused call into ``preprocess/called_variants.json``.
``AbsoluteGrowthRateChoe2019Dataset`` consumes ``si12.xlsx`` alone (sheets ``Fig2c``
and ``Fig2g``), and ``MetaboliteChoe2019Dataset`` the same workbook (sheets ``Fig1g`` and
``Fig3d``).
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
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    BacterialAssemblySet,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainBackground,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    assembly_reference,
    bacterial_genome,
    released_identifier_kind,
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

CITATION_KEY = "choeAdaptiveLaboratoryEvolution2019"
PAPER_DOI = "10.1038/s41467-019-08888-6"
PAPER_TITLE = "Adaptive laboratory evolution of a genome-reduced Escherichia coli"
PUBMED_ID = "30804555"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

GROWTH_DATASET_ROOT_REL = "data/torchcell/growth_rate_choe2019"
ABSOLUTE_DATASET_ROOT_REL = "data/torchcell/absolute_growth_rate_choe2019"
METABOLITE_DATASET_ROOT_REL = "data/torchcell/metabolite_choe2019"
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
#: The Fig. 2f row labels, verbatim and in sheet order; all but the first are records.
RATIO_ROW_ORDER: tuple[str, ...] = (
    "eMS57/MS56",
    "ilvN*/ilvNWT",
    "cspC*/cspCWT",
    "yifB*/yifBWT",
)
#: The Fig. 2f sheet's own header row, verbatim and in order.
RATIO_HEADERS: tuple[str, ...] = ("Mutated/Parental", "Set1", "Set2", "Set3")
#: The Fig. 2g sheet (#1175, 2026-11-24): growth rate per (medium, strain, culture).
SUPPLEMENT_SHEET = "Fig2g"
SUPPLEMENT_SHEET_TITLE = (
    "Growth rate of MG1655 and eMS57 in valine or isoleucin supplemented M9 glucose "
    "medium"
)
SUPPLEMENT_HEADERS: tuple[str, ...] = ("Media", "Strain", "Rep1", "Rep2", "Rep3")
BASE_MEDIUM_LABEL = "M9"
VALINE_LABEL = "M9 + valine"
VALINE_ISOLEUCINE_LABEL = "M9 + valine + isoleucine"
#: The Fig. 2g (medium, strain) rows, verbatim and in sheet order; the medium cell is
#: written once per block and blank on the second row of it.
SUPPLEMENT_ROW_ORDER: tuple[tuple[str, str], ...] = (
    (BASE_MEDIUM_LABEL, WILD_TYPE_LABEL),
    (BASE_MEDIUM_LABEL, EVOLVED_LABEL),
    (VALINE_LABEL, WILD_TYPE_LABEL),
    (VALINE_LABEL, EVOLVED_LABEL),
    (VALINE_ISOLEUCINE_LABEL, WILD_TYPE_LABEL),
    (VALINE_ISOLEUCINE_LABEL, EVOLVED_LABEL),
)
#: The compounds each Fig. 2g medium adds to M9 glucose, at the Methods' 0.15 mM.
SUPPLEMENTS: dict[str, tuple[str, ...]] = {
    BASE_MEDIUM_LABEL: (),
    VALINE_LABEL: ("L-valine",),
    VALINE_ISOLEUCINE_LABEL: ("L-valine", "L-isoleucine"),
}
SUPPLEMENT_MM = 0.15
#: The cell the Fig. 2g sheet writes where it releases no number.
NOT_RELEASED_CELL = "-"
#: The Fig. 1g sheet (#1175, 2026-11-24): one block per culture, three timepoints.
PYRUVATE_SHEET = "Fig1g"
PYRUVATE_SHEET_TITLE = "Intra and extracellular pyruvate level in MG1655 and eMS57"
#: The block labels, verbatim and in sheet order; each is the replicate_id it releases.
PYRUVATE_BLOCK_LABELS: tuple[str, ...] = (
    "MG1655 Rep 1",
    "MG1655 Rep 2",
    "eMS57 Rep1",
    "eMS57 Rep2",
)
#: The timepoint header of every block, verbatim, and the hours after inoculation.
PYRUVATE_TIMEPOINTS: tuple[tuple[str, float], ...] = (
    ("4h", 4.0),
    ("6h", 6.0),
    ("8h", 8.0),
)
PYRUVATE_OD_ROW = "Optical density (OD600nm)"
#: The two pyruvate rows of a block: (row label verbatim, measurement_type).
PYRUVATE_ROWS: tuple[tuple[str, str], ...] = (
    (
        "intracellular pyruvate (umol/109 cells)",
        "intracellular_pyruvate_umol_per_1e9_cells",
    ),
    ("extracellular pyruvate (uM)", "extracellular_pyruvate_uM"),
)
PYRUVATE_KEY = "pyruvate"
#: The Fig. 3d sheet: two ratio rows over Rep1-Rep2, one ATP row over Rep1-Rep3.
COFACTOR_SHEET = "Fig3d"
COFACTOR_SHEET_TITLE = "Intracellular NAD, NADP, and ATP level in MG1655 and eMS57"
#: (row label verbatim, measurement_type); the label is also the level's key.
COFACTOR_RATIO_ROWS: tuple[tuple[str, str], ...] = (
    ("NADH/NAD+", "intracellular_nadh_to_nad_ratio"),
    ("NADPH/NADP+", "intracellular_nadph_to_nadp_ratio"),
)
ATP_ROW = "ATP (nmol/109 cells)"
ATP_MEASUREMENT_TYPE = "intracellular_atp_nmol_per_1e9_cells"
ATP_KEY = "ATP"
METABOLITE_PROTOCOLS: frozenset[str] = frozenset(
    {
        *(measurement for _row, measurement in PYRUVATE_ROWS),
        *(measurement for _row, measurement in COFACTOR_RATIO_ROWS),
        ATP_MEASUREMENT_TYPE,
    }
)
#: The paper's own statement about Fig. 3d, the cross-check of the two ratio rows.
NADH_FOLD_LOWER = 3.0
NADPH_FOLD_HIGHER = 4.5
COFACTOR_RTOL = 0.05


class ReconstructedAllele(BaseModel):
    """One Fig. 2f reconstructed single-allele strain and its kan-marked control.

    Every field is a string Supplementary Table 3 or the Fig. 2f sheet prints, verbatim:
    the ratio row label, the gene symbol and the allele inside the mutant's genotype, the
    two strain names and their genotypes, and the marker as the mutant row writes it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ratio_label: str
    gene: str
    allele: str
    strain: str
    genotype_statement: str
    control_strain: str
    control_genotype_statement: str
    cassette: str

    @property
    def strain_quote(self) -> str:
        """The mutant's Supplementary Table 3 row, as ``si/si1.md`` carries it."""
        return (
            f"{self.strain}</td><td rowspan=1 colspan=1>{self.genotype_statement}</td>"
        )

    @property
    def control_quote(self) -> str:
        """The control's Supplementary Table 3 row, as ``si/si1.md`` carries it."""
        return (
            f"{self.control_strain}</td><td rowspan=1 colspan=1>"
            f"{self.control_genotype_statement}</td>"
        )


#: The three reconstructed alleles, in Fig. 2f row order (#1045 row 22, #1057).
RECONSTRUCTED_ALLELES: tuple[ReconstructedAllele, ...] = (
    ReconstructedAllele(
        ratio_label="ilvN*/ilvNWT",
        gene="ilvN",
        allele="ilvN(C202T)",
        strain="ilvNmut",
        genotype_statement="MS56, ilvN::ilvN(C202T)-kanR",
        control_strain="ilvNWT",
        control_genotype_statement="MS56, ilvN::ilvN-kan",
        cassette="kanR",
    ),
    ReconstructedAllele(
        ratio_label="cspC*/cspCWT",
        gene="cspC",
        allele="cspC(G37A)",
        strain="cspCmut",
        genotype_statement="MS56, cspC::cspC(G37A)-kanR",
        control_strain="cspCWT",
        control_genotype_statement="MS56, cspC::cspC-kan",
        cassette="kanR",
    ),
    ReconstructedAllele(
        ratio_label="yifB*/yifBWT",
        gene="yifB",
        allele="yifB(1169C insertion)",
        strain="yifBmut",
        genotype_statement="MS56, yifB::yifB(1169C insertion)-kanR",
        control_strain="yifBWT",
        control_genotype_statement="MS56, yifB::yifB-kanR",
        cassette="kanR",
    ),
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
#: The evolved clone, verbatim from Supplementary Table 3 (#1045 part 3).
EVOLVED_GENOTYPE_STATEMENT = "E. coli MS56 adaptively evolved in M9 glucosemedium"

#: The Keio arm: the two BW25113 single deletions whose growth rate the Supplementary
#: Fig. 6 legend states, as a FRACTION of wild-type BW25113 in the same medium.
KEIO_COLLECTION = "single gene knockout collection (the Keio collection)"
KEIO_FITNESS: tuple[tuple[str, float], tuple[str, float]] = (
    ("ydaS", 0.724),
    ("abgR", 0.698),
)
#: Strains in the Supplementary Fig. 6 panel; only two carry a released number.
KEIO_PANEL_STRAINS = 62

#: Expected record counts, asserted at the end of each build, measured on the pinned
#: workbook (#1175, 2026-11-24): Fig. 2f 4 rows x 3 sets; Fig. 2c 5 strains x 3
#: cultures; Fig. 2g 18 cells less the 4 dashes.
GROWTH_EXPECTED_RECORDS = 12
GROWTH_EXPECTED_REFERENCES = 4
ABSOLUTE_FIG2C_RECORDS = 15
ABSOLUTE_FIG2G_RECORDS = 14
ABSOLUTE_EXPECTED_RECORDS = ABSOLUTE_FIG2C_RECORDS + ABSOLUTE_FIG2G_RECORDS
#: Fig. 2c: MG1655, MS56 (also the deletions'), eMS57; Fig. 2g: MG1655, eMS57 in M9.
ABSOLUTE_EXPECTED_REFERENCES = 5
#: Records in M9 glucose with no supplement: all of Fig. 2c plus Fig. 2g's M9 arm.
ABSOLUTE_EXPECTED_UNPERTURBED = 20
#: Fig. 1g: 4 cultures x 3 timepoints x 2 pyruvate rows; Fig. 3d: 2 ratio rows x 4
#: cells plus 6 ATP cells.
METABOLITE_FIG1G_RECORDS = 24
METABOLITE_FIG3D_RECORDS = 14
METABOLITE_EXPECTED_RECORDS = METABOLITE_FIG1G_RECORDS + METABOLITE_FIG3D_RECORDS
#: One reference per (strain, measurement_type, environment): Fig. 1g 2 x 2 x 3,
#: Fig. 3d 2 x 3.
METABOLITE_EXPECTED_REFERENCES = 18
#: The Fig. 1g OD600 cells, a growth-curve readout no phenotype class holds.
FIG1G_OD_CELLS = 12
#: Fig. 2g cells that print a dash instead of a number.
FIG2G_DASH_CELLS = 4
TF_EXPECTED_RECORDS = 2
TF_EXPECTED_REFERENCES = 1
#: Every deletion symbol must resolve; a renamed label stops the build.
#: The verification's genome-reference row over the 21 named genes (#1045 part 2):
#: registered in ``torchcell.verification.report``, so it reports and never gates.
DELETED_REGION_RULE = "deleted_symbols_are_one_contiguous_run_of_the_pinned_assembly"
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
_Q_FIG2F = (
    "f Construction of the cspC point mutation on MS56 had a beneficial effect on "
    "fitness, whereas ilvN mutation showed no effect. The yifB mutation on MS56 "
    "decreased the growth rate. Error bars indicate s.d. of three biological replicates "
    "shown as red circles."
)
_Q_FIG2C = (
    "c Growth rates of MS56 with rpoS deletion or a large deletion compared to MG1655 "
    "and eMS57. Error bars indicate the s.d. of three individual cultures shown in red "
    "circles."
)
_Q_FIG2G = (
    "g Growth rate of MG1655 and eMS57 in response to valine supplementation which "
    "inhibits cell growth (white bar: MG1655, blue bar: eMS57, error bars indicate s.d. "
    "of three biological replicates shown as red circles). eMS57 was completely "
    "resistant to valine toxicity."
)
_Q_SUPPLEMENT = (
    "Isoleucine or valine was supplemented with a final concentration of "
    "$0 . 1 5 \\mathrm { m M }$ , when appropriate."
)
_Q_FIG1G = (
    "Intracellular pyruvate level is presented as mole pyruvate per "
    "$1 0 ^ { 9 }$ cells and extracellular pyruvate level was measured in molar "
    "concentration. Individual data points are shown as blue circles and error bars "
    "indicate the s.d. of two biological replicates."
)
_Q_FIG1G_TIME = (
    "g Intracellular and extracellular pyruvate concentrations for MG1655 and eMG57 at "
    "$4 , 6 ,$ and $^ { 8 \\mathrm { h } }$ after inoculation."
)
_Q_PYRUVATE_ASSAY = (
    "Intracellular and extracellular pyruvate assay. Cells were grown in M9 glucose "
    "medium at $3 7 ^ { \\circ } \\mathrm { C }$ ."
)
_Q_FIG3D = (
    "ATP intracellular level was decreased in eMS57. Red circles indicate three "
    "independent assays from biological replicates. Error bars indicate the s.d. of "
    "three biological replicates."
)
_Q_COFACTOR_FOLD = (
    "In eMS57, the NADH level was 3.0 times lower and the NADPH level was 4.5 times "
    "higher than MG1655."
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
_Q_EMS57_ROW = (
    "eMS57</td><td rowspan=1 colspan=1>E. coli MS56 adaptively evolved in M9 "
    "glucosemedium</td><td rowspan=1 colspan=1>This study</td>"
)
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
    "evolved_strain": _si1(EVOLVED_LABEL, _Q_EMS57_ROW),
    "evolved_genotype_statement": _si1(
        EVOLVED_GENOTYPE_STATEMENT,
        _Q_EMS57_ROW,
        note="the release gives the clone no variant list (Supplementary Data 2 is a "
        "population allele frequency per timepoint), so the statement is the whole "
        "genotype it states",
    ),
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
    "fig2c_replicate": _paper(
        "three individual cultures",
        _Q_FIG2C,
        note="each Fig. 2c Rep1-Rep3 cell is one culture's growth rate, so each is "
        "one record with n_samples 1 (#1175)",
    ),
    "fig2g_replicate": _paper(
        "three biological replicates",
        _Q_FIG2G,
        note="each numeric Fig. 2g Rep1-Rep3 cell is one culture's growth rate; four "
        "cells print a dash and release no number",
    ),
    "supplement_mm": _paper(SUPPLEMENT_MM, _Q_SUPPLEMENT),
    "fig1g_replicate": _paper(
        "two biological replicates",
        _Q_FIG1G,
        note="each Fig. 1g block (MG1655 Rep 1 ... eMS57 Rep2) is one culture sampled "
        "at 4, 6 and 8 h, so each (block, timepoint, row) cell is one record",
    ),
    "fig1g_hours_after_inoculation": _paper("4, 6 and 8 h", _Q_FIG1G_TIME),
    "pyruvate_temperature_c": _paper(37.0, _Q_PYRUVATE_ASSAY),
    "fig3d_replicate": _paper(
        "three biological replicates",
        _Q_FIG3D,
        note="the legend states three replicates for the panel, and the sheet releases "
        "three ATP cells but only two NADH/NAD+ and two NADPH/NADP+ cells per strain; "
        "every released cell is a record and none is invented",
    ),
    "cofactor_fold_changes": _paper(
        {"nadh_fold_lower": NADH_FOLD_LOWER, "nadph_fold_higher": NADPH_FOLD_HIGHER},
        _Q_COFACTOR_FOLD,
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
    "reconstruction_replicates": _paper(
        3,
        _Q_FIG2F,
        note="the Fig. 2f sheet's Set1-Set3 columns are the three biological replicates "
        "of each mutant-over-control ratio",
    ),
    **{
        f"{a.strain}_genotype": _si1(a.genotype_statement, a.strain_quote)
        for a in RECONSTRUCTED_ALLELES
    },
    **{
        f"{a.control_strain}_genotype": _si1(
            a.control_genotype_statement, a.control_quote
        )
        for a in RECONSTRUCTED_ALLELES
    },
}

_GAP_TEMPERATURE = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the paper states no cultivation temperature for the Fig. 2c, 2f and 2g "
    "growth-rate panels. Every temperature it does state is 37 C (the ALE flask, the "
    "pyruvate assay, the Supplementary Fig. 6 microplate), but none of those is these "
    "panels, so the field is left unset rather than carried across",
)
#: One culture or one set is one measurement, so a dispersion does not exist (#1178).
_SINGLE_MEASUREMENT_NOTE = (
    "the record is one released culture (Fig. 2c, Fig. 2g) or one released set "
    "(Fig. 2f), so no dispersion of its own exists; the dispersion the legends state "
    "is over the panel's three cultures, which the reference summarizes and the "
    "records of one genotype reproduce at modeling time"
)


def _single_measurement_gaps(fields: Sequence[str]) -> list[ProvenanceGap]:
    """``not_applicable_single_measurement`` on each dispersion field of an n = 1 record."""
    return [
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_applicable_single_measurement,
            note=_SINGLE_MEASUREMENT_NOTE,
        )
        for field in fields
    ]


_FITNESS_DISPERSION_FIELDS: tuple[str, ...] = (
    "fitness_se",
    "fitness_uncertainty",
    "fitness_uncertainty_type",
)
_RESPONSE_DISPERSION_FIELDS: tuple[str, ...] = (
    "environment_response_se",
    "environment_response_uncertainty",
    "environment_response_uncertainty_type",
)
_GAP_ASSAY = ProvenanceGap(
    field="assay_type",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the paper states that the growth measurements were biologically triplicated "
    "but neither the instrument nor the fit behind a Fig. 2c or Fig. 2g rate, so no "
    "AssayType member is sourced; liquid_od_growth would be a reading, not a "
    "released fact",
)
#: What the absolute readout's number is; the sheets print no unit.
ABSOLUTE_UNITS = (
    "growth rate as the Source Data sheet releases it; neither the sheet nor the "
    "legend prints a unit"
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
#: ``ReleasedIdentifierPerturbation`` is not a ``ProvenanceGapMixin``, so a typed
#: absence on one of its fields is filed in ``preprocess/provenance_gaps.json`` instead.
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
#: The absence that has NO typed home, measured: ``StrainBackground.alleles``
#: defaults to ``[]``, and ``ProvenanceGapMixin`` refuses a gap on a field that is not
#: ``None``, so MS56's 55 unenumerated regions are recorded here and in the ledger.
BACKGROUND_ABSENCE = {
    "model": "StrainBackground",
    "field": "alleles",
    "strain": PARENT_LABEL,
    "why_no_typed_gap": "ProvenanceGapMixin requires a gapped field to be None; "
    "alleles defaults to [], so declaring the absence raises "
    "\"field 'alleles' has a ProvenanceGap but is not None\"",
    "what_is_missing": "the gene content of MS56's 55 deleted regions, ~1.1 Mbp",
    "where_it_lives_instead": "genotype_statement, verbatim from Supplementary Table 3",
    "source_defers_to": "reference 4 of this paper; MS56 has no deposited assembly",
}
#: The same untyped absence for the evolved clone (#1045 part 3).
EVOLVED_ABSENCE = {
    "model": "StrainBackground",
    "field": "alleles",
    "strain": EVOLVED_LABEL,
    "why_no_typed_gap": BACKGROUND_ABSENCE["why_no_typed_gap"],
    "what_is_missing": "the clone's own mutations beyond MS56",
    "where_it_lives_instead": "genotype_statement, verbatim from Supplementary Table 3",
    "source_defers_to": "nothing: the release gives population allele frequencies "
    "(Supplementary Data 2-4), not a clone genotype",
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


def supplemented_environment(medium_label: str) -> Environment:
    """One Fig. 2g medium: M9 glucose plus each named amino acid at 0.15 mM.

    ``M9`` returns exactly :func:`growth_environment`, so the Fig. 2g M9 arm and Fig. 2c
    share one environment and are told apart by ``screen_id`` alone. Each supplement is
    a ``SmallMoleculePerturbation`` on the base medium, the Lamoureux 2023 convention
    for a compound added to a defined medium.
    """
    compounds = SUPPLEMENTS[medium_label]
    if not compounds:
        return growth_environment()
    return Environment(
        media=M9_GLUCOSE_CHOE2019,
        temperature=None,
        aerobicity="aerobic",
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound(name),
                concentration=Concentration(
                    value=SOURCED_VALUES["supplement_mm"].value, unit=_MM
                ),
            )
            for name in compounds
        ],
        provenance_gaps=[_GAP_TEMPERATURE],
    )


_GAP_METABOLITE_AEROBICITY = ProvenanceGap(
    field="aerobicity",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the pyruvate, NAD(P)H and ATP assay Methods state the medium and, for "
    "pyruvate, 37 C, but no vessel, agitation or oxygen regime for these cultures",
)
_GAP_COFACTOR_TEMPERATURE = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the NAD/NADH, NADP/NADPH and ATP assay Methods count cells and state no "
    "cultivation temperature; the medium is the Methods default, M9 glucose 'unless "
    "stated otherwise'",
)


def pyruvate_environment(hours: float) -> Environment:
    """The Fig. 1g culture at one sampling time: M9 glucose, 37 C, hours after inoculation.

    Each timepoint is its own environment (``duration_hours``), so the three samples of
    one culture are three records that differ in the environment, not three replicates.
    """
    return Environment(
        media=M9_GLUCOSE_CHOE2019,
        temperature=Temperature(value=SOURCED_VALUES["pyruvate_temperature_c"].value),
        aerobicity=None,
        duration_hours=hours,
        provenance_gaps=[_GAP_METABOLITE_AEROBICITY],
    )


def cofactor_environment() -> Environment:
    """The Fig. 3d cultures: M9 glucose, temperature and oxygen regime unstated."""
    return Environment(
        media=M9_GLUCOSE_CHOE2019,
        temperature=None,
        aerobicity=None,
        provenance_gaps=[_GAP_COFACTOR_TEMPERATURE, _GAP_METABOLITE_AEROBICITY],
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

    @property
    def labelled(self) -> tuple[tuple[str, float], ...]:
        """``(replicate label, rate)`` per released culture, labels from the header."""
        return tuple(zip(GROWTH_HEADERS[1:], self.replicates, strict=True))


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
    """Read the Fig. 2f sheet: ``{row label: released ratios}`` in ``Set1``-``Set3`` order."""
    rows = _sheet_rows(path, RATIO_SHEET)
    title = rows[0][1]
    if title != RATIO_SHEET_TITLE:
        raise TableFormatError(f"{RATIO_SHEET} title is {title!r}")
    headers = tuple(str(cell) for cell in rows[2][1:5])
    if headers != RATIO_HEADERS:
        raise TableFormatError(f"{RATIO_SHEET} headers are {headers}")
    out: dict[str, tuple[float, ...]] = {}
    for row in rows[3:]:
        if row[1] is None:
            continue
        out[str(row[1])] = tuple(float(cell) for cell in row[2:5])
    order = tuple(out)
    if order != RATIO_ROW_ORDER:
        raise TableFormatError(f"{RATIO_SHEET} row order is {order}")
    return out


class CultureRate(BaseModel):
    """One released Fig. 2g cell that carries a number: one culture's growth rate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    medium: str = Field(description="the medium label, verbatim")
    strain: str = Field(description="the strain label, verbatim")
    replicate_id: str = Field(description="the column header, verbatim (Rep1-Rep3)")
    rate: float


class DashCell(BaseModel):
    """One released Fig. 2g cell that prints a dash: no number is released for it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    medium: str
    strain: str
    replicate_id: str


class SupplementPanel(BaseModel):
    """The Fig. 2g sheet: every numeric cell, and every dash, in sheet order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rates: tuple[CultureRate, ...]
    dashes: tuple[DashCell, ...]

    def base_rates(self, strain: str) -> tuple[float, ...]:
        """One strain's released M9 rates: the reference of its supplemented records."""
        return tuple(
            cell.rate
            for cell in self.rates
            if cell.medium == BASE_MEDIUM_LABEL and cell.strain == strain
        )


def read_supplement_panel(path: str) -> SupplementPanel:
    """Read the Fig. 2g sheet, checking its title, its headers and its row order.

    The medium cell is written on the first row of each two-row block, so a blank one
    carries the medium above it. A cell is either a number or the release's dash; any
    other content stops the build rather than being coerced.
    """
    rows = _sheet_rows(path, SUPPLEMENT_SHEET)
    title = rows[0][1]
    if title != SUPPLEMENT_SHEET_TITLE:
        raise TableFormatError(f"{SUPPLEMENT_SHEET} title is {title!r}")
    headers = tuple(str(cell) for cell in rows[2][1:6])
    if headers != SUPPLEMENT_HEADERS:
        raise TableFormatError(f"{SUPPLEMENT_SHEET} headers are {headers}")
    medium: str | None = None
    order: list[tuple[str, str]] = []
    rates: list[CultureRate] = []
    dashes: list[DashCell] = []
    for row in rows[3:]:
        if row[2] is None:
            continue
        if row[1] is not None:
            medium = str(row[1])
        if medium is None:
            raise TableFormatError(
                f"{SUPPLEMENT_SHEET}: a strain row before any medium"
            )
        strain = str(row[2])
        order.append((medium, strain))
        for replicate_id, cell in zip(SUPPLEMENT_HEADERS[2:], row[3:6], strict=True):
            if cell == NOT_RELEASED_CELL:
                dashes.append(
                    DashCell(medium=medium, strain=strain, replicate_id=replicate_id)
                )
            elif isinstance(cell, int | float) and not isinstance(cell, bool):
                rates.append(
                    CultureRate(
                        medium=medium,
                        strain=strain,
                        replicate_id=replicate_id,
                        rate=float(cell),
                    )
                )
            else:
                raise TableFormatError(
                    f"{SUPPLEMENT_SHEET} ({medium}, {strain}, {replicate_id}) holds "
                    f"{cell!r}, neither a number nor the release's dash"
                )
    if tuple(order) != SUPPLEMENT_ROW_ORDER:
        raise TableFormatError(f"{SUPPLEMENT_SHEET} row order is {tuple(order)}")
    return SupplementPanel(rates=tuple(rates), dashes=tuple(dashes))


class MetaboliteValue(BaseModel):
    """One released Fig. 1g or Fig. 3d cell: one culture's level at one condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sheet: str
    strain: str = Field(description="MG1655 or eMS57, as the block or column names it")
    replicate_id: str = Field(description="the sheet's own label for the culture")
    measurement_type: str
    metabolite: str = Field(description="the level's key: pyruvate, NADH/NAD+, ...")
    hours: float | None = Field(
        default=None, description="hours after inoculation (Fig. 1g); None for Fig. 3d"
    )
    value: float


def _released_number(sheet: str, cell: Any, where: str) -> float:
    """A released numeric cell, refusing anything that is not a finite number."""
    if isinstance(cell, bool) or not isinstance(cell, int | float):
        raise TableFormatError(f"{sheet} {where} holds {cell!r}, not a number")
    value = float(cell)
    if not math.isfinite(value):
        raise TableFormatError(f"{sheet} {where} holds a non-finite {cell!r}")
    return value


def read_pyruvate_panel(path: str) -> tuple[tuple[MetaboliteValue, ...], int]:
    """Read the Fig. 1g sheet: every pyruvate cell, and the count of OD600 cells.

    Each block is one culture: its label row (``MG1655 Rep 1`` with the ``4h``, ``6h``,
    ``8h`` headers), an OD600 row, an intracellular and an extracellular pyruvate row.
    The block order, the headers and the row labels are checked verbatim.
    """
    rows = _sheet_rows(path, PYRUVATE_SHEET)
    title = rows[0][1]
    if title != PYRUVATE_SHEET_TITLE:
        raise TableFormatError(f"{PYRUVATE_SHEET} title is {title!r}")
    hours = dict(PYRUVATE_TIMEPOINTS)
    values: list[MetaboliteValue] = []
    od_cells = 0
    blocks: list[str] = []
    expected_rows = (PYRUVATE_OD_ROW, *(label for label, _m in PYRUVATE_ROWS))
    measurement = dict(PYRUVATE_ROWS)
    index = 0
    while index < len(rows):
        label = rows[index][1]
        if label is None or str(label) not in PYRUVATE_BLOCK_LABELS:
            if label is not None and index > 0:
                raise TableFormatError(f"{PYRUVATE_SHEET} row {index} is {label!r}")
            index += 1
            continue
        block = str(label)
        blocks.append(block)
        headers = tuple(str(cell) for cell in rows[index][2:5])
        if headers != tuple(header for header, _h in PYRUVATE_TIMEPOINTS):
            raise TableFormatError(f"{PYRUVATE_SHEET} {block} headers are {headers}")
        body = rows[index + 1 : index + 1 + len(expected_rows)]
        if tuple(str(row[1]) for row in body) != expected_rows:
            raise TableFormatError(f"{PYRUVATE_SHEET} {block} rows are out of order")
        strain = block.split(" ")[0]
        for row in body:
            row_label = str(row[1])
            for header, cell in zip(headers, row[2:5], strict=True):
                number = _released_number(
                    PYRUVATE_SHEET, cell, f"({block}, {row_label}, {header})"
                )
                if row_label == PYRUVATE_OD_ROW:
                    od_cells += 1
                    continue
                values.append(
                    MetaboliteValue(
                        sheet=PYRUVATE_SHEET,
                        strain=strain,
                        replicate_id=block,
                        measurement_type=measurement[row_label],
                        metabolite=PYRUVATE_KEY,
                        hours=hours[header],
                        value=number,
                    )
                )
        index += 1 + len(expected_rows)
    if tuple(blocks) != PYRUVATE_BLOCK_LABELS:
        raise TableFormatError(f"{PYRUVATE_SHEET} blocks are {tuple(blocks)}")
    return tuple(values), od_cells


def _strain_columns(
    header: Sequence[Any], replicates: Sequence[Any]
) -> list[tuple[str, str]]:
    """``(strain, replicate label)`` per column of a Fig. 3d block, strain carried right."""
    columns: list[tuple[str, str]] = []
    strain: str | None = None
    for name, replicate in zip(header, replicates, strict=True):
        if replicate is None:
            break
        if name is not None:
            strain = str(name)
        if strain not in (WILD_TYPE_LABEL, EVOLVED_LABEL):
            raise TableFormatError(f"{COFACTOR_SHEET} column strain is {strain!r}")
        columns.append((strain, str(replicate)))
    return columns


def read_cofactor_panel(path: str) -> tuple[MetaboliteValue, ...]:
    """Read the Fig. 3d sheet: the two ratio rows and the ATP row, every cell.

    The ratio block releases ``Rep1``-``Rep2`` per strain and the ATP block
    ``Rep1``-``Rep3``; both blocks are read by their own header rows, so a culture that
    is in one block and not the other is neither invented nor dropped.
    """
    rows = _sheet_rows(path, COFACTOR_SHEET)
    title = rows[0][1]
    if title != COFACTOR_SHEET_TITLE:
        raise TableFormatError(f"{COFACTOR_SHEET} title is {title!r}")
    if rows[2][1] != "Ratio" or rows[8][1] != "Concentration":
        raise TableFormatError(f"{COFACTOR_SHEET} block headers moved")
    values: list[MetaboliteValue] = []
    ratio_columns = _strain_columns(rows[2][2:], rows[3][2:])
    for offset, (row_label, measurement) in enumerate(COFACTOR_RATIO_ROWS):
        row = rows[4 + offset]
        if row[1] != row_label:
            raise TableFormatError(f"{COFACTOR_SHEET} row {4 + offset} is {row[1]!r}")
        for (strain, replicate), cell in zip(ratio_columns, row[2:], strict=False):
            values.append(
                MetaboliteValue(
                    sheet=COFACTOR_SHEET,
                    strain=strain,
                    replicate_id=replicate,
                    measurement_type=measurement,
                    metabolite=row_label,
                    value=_released_number(
                        COFACTOR_SHEET, cell, f"({row_label}, {strain}, {replicate})"
                    ),
                )
            )
    atp_columns = _strain_columns(rows[8][2:], rows[9][2:])
    if rows[10][1] != ATP_ROW:
        raise TableFormatError(f"{COFACTOR_SHEET} row 10 is {rows[10][1]!r}")
    for (strain, replicate), cell in zip(atp_columns, rows[10][2:], strict=False):
        values.append(
            MetaboliteValue(
                sheet=COFACTOR_SHEET,
                strain=strain,
                replicate_id=replicate,
                measurement_type=ATP_MEASUREMENT_TYPE,
                metabolite=ATP_KEY,
                value=_released_number(
                    COFACTOR_SHEET, cell, f"({ATP_ROW}, {strain}, {replicate})"
                ),
            )
        )
    return tuple(values)


def check_cofactor_fold_changes(values: Sequence[MetaboliteValue]) -> dict[str, Any]:
    """The two ratio rows reproduce the paper's "3.0 times lower" and "4.5 times higher".

    The prose is the independent statistic for the Fig. 3d sheet: a swapped strain block
    or a rescaled row would stop matching it.
    """

    def mean(measurement: str, strain: str) -> float:
        return statistics.fmean(
            v.value
            for v in values
            if v.measurement_type == measurement and v.strain == strain
        )

    nadh, nadph = (measurement for _row, measurement in COFACTOR_RATIO_ROWS)
    lower = mean(nadh, WILD_TYPE_LABEL) / mean(nadh, EVOLVED_LABEL)
    higher = mean(nadph, EVOLVED_LABEL) / mean(nadph, WILD_TYPE_LABEL)
    for measured, stated in ((lower, NADH_FOLD_LOWER), (higher, NADPH_FOLD_HIGHER)):
        if abs(measured - stated) > COFACTOR_RTOL * stated:
            raise TableFormatError(
                f"{COFACTOR_SHEET}: measured fold change {measured:.4f} is more than "
                f"{COFACTOR_RTOL:.0%} from the stated {stated}"
            )
    return {
        "stated_nadh_fold_lower": NADH_FOLD_LOWER,
        "measured_nadh_fold_lower": lower,
        "stated_nadph_fold_higher": NADPH_FOLD_HIGHER,
        "measured_nadph_fold_higher": higher,
        "relative_tolerance": COFACTOR_RTOL,
    }


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


def check_deleted_region(genome: EcoliK12Genome) -> dict[str, Any]:
    """Where the 21 named genes sit on the pinned MG1655 assembly; reports, never raises.

    The cross-check that lets the gene list stand in for the released coordinates:
    Supplementary Table 3 names the genes and the Methods name a span, and on the
    pinned assembly the named genes should span one uninterrupted block. It compares the
    release with OUR annotation, so it is a genome-reference finding (#1045 part 2):
    ``problems`` lists what the annotation disagrees with, and the build never reads it.
    """
    spans: dict[str, tuple[int, int]] = {}
    for feature in genome.db.features_of_type("gene"):
        feature_tags = feature.attributes.get("locus_tag") or []
        if len(feature_tags) == 1:
            spans[feature_tags[0]] = (int(feature.start), int(feature.end))
    resolved = {
        symbol: genome.resolve_gene_name(symbol).systematic_name
        for symbol in LARGE_DELETION_SYMBOLS
    }
    tags = [tag for tag in resolved.values() if tag is not None and tag in spans]
    problems: list[str] = []
    unplaced = [s for s, tag in resolved.items() if tag is None or tag not in spans]
    if unplaced:
        problems.append(f"no gene feature of the pinned assembly for {unplaced}")
    details: dict[str, Any] = {
        "named_symbols": list(LARGE_DELETION_SYMBOLS),
        "locus_tags": [resolved[s] for s in LARGE_DELETION_SYMBOLS],
        "released_ms56_window": [2038496, 2059460],
        "released_ms56_window_bp": 2059460 - 2038496 + 1,
        "note": "the released coordinates are MS56 coordinates, so they are not stored "
        "as a span against the pinned MG1655 assembly",
    }
    if tags:
        low = min(spans[tag][0] for tag in tags)
        high = max(spans[tag][1] for tag in tags)
        inside = sorted(
            tag for tag, (start, end) in spans.items() if start >= low and end <= high
        )
        unlisted = [tag for tag in inside if tag not in set(tags)]
        if unlisted:
            problems.append(
                f"the MG1655 window {low}-{high} also contains {unlisted}, which "
                "Supplementary Table 3 does not list as deleted"
            )
        details.update(
            {
                "mg1655_window": [low, high],
                "mg1655_window_bp": high - low + 1,
                "ms56_to_mg1655_offset_bp": low - 2038496,
                "genes_inside_window": len(inside),
                "unlisted_genes_inside_window": len(unlisted),
            }
        )
    details["problems"] = problems
    return details


def deleted_region_rule(genome: EcoliK12Genome) -> LevelResult:
    """L4 genome-reference row: :func:`check_deleted_region`, reported, never gating."""
    details = check_deleted_region(genome)
    problems = details["problems"]
    return LevelResult(
        level=Level.L4,
        name=DELETED_REGION_RULE,
        passed=not problems,
        message=(
            f"the {len(LARGE_DELETION_SYMBOLS)} released deletion symbols are one "
            f"contiguous run of {genome.ASSEMBLY_SET} with no unlisted gene inside"
            if not problems
            else "; ".join(problems)
        ),
        details=details,
    )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def parent_background() -> StrainBackground:
    """MS56, the genome-reduced parent both deletion strains were built on.

    ``alleles`` is empty and that is a statement, not an omission: this paper writes the
    genotype as "large deletions MD1 to MD56" and defers the content to its reference 4,
    MS56 has no deposited assembly, and ``AlleleEdit`` cannot say "55 regions somebody
    else listed". The verbatim genotype string is kept instead, and the untyped absence
    is filed in ``preprocess/genotype_gaps.json``.
    """
    return StrainBackground(
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


def evolved_background() -> StrainBackground:
    """eMS57, the clone isolated from the MS56 evolution, named and stated as released.

    ``alleles`` is empty for the reason MS56's is: the release gives no clone variant
    list, and a population allele frequency is not a clone genotype. Supplementary Table
    3's statement is kept verbatim and the untyped absence is filed in
    ``preprocess/genotype_gaps.json`` (#1045 part 3).
    """
    return StrainBackground(
        name=EVOLVED_LABEL,
        reference_strain=MG1655,
        assembly_set=MG1655_ASSEMBLY_SET,
        parents=[PARENT_LABEL],
        genotype_statement=EVOLVED_GENOTYPE_STATEMENT,
        alleles=[],
        provenance=[
            SOURCED_VALUES["evolved_strain"],
            SOURCED_VALUES["evolved_genotype_statement"],
        ],
    )


def released_ratio_phenotype(ratio: float, replicate_id: str) -> FitnessPhenotype:
    """One released Fig. 2f set: the mutant-over-control ratio the sheet prints.

    The ratio is the SOURCE's own per-set value, so it is stored verbatim with
    ``n_samples = 1`` and the set's label (#1175, policy item 6: a per-replicate ratio the
    release itself prints is stored as that ratio). A single set has no dispersion of
    its own, so the three dispersion fields are ``not_applicable_single_measurement``
    gaps; the mean and SD over the three sets are a modeling-time derivation.
    """
    if not math.isfinite(ratio):
        raise TableFormatError(f"non-finite released ratio {ratio!r}")
    return FitnessPhenotype(
        fitness=ratio,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        replicate_id=replicate_id,
        provenance_gaps=_single_measurement_gaps(_FITNESS_DISPERSION_FIELDS),
    )


def culture_rate_phenotype(
    rate: float, replicate_id: str, screen_id: str
) -> EnvironmentResponsePhenotype:
    """One released culture's ABSOLUTE growth rate, verbatim (Fig. 2c or Fig. 2g).

    ``screen_id`` names the panel, because Fig. 2c and Fig. 2g release the same strain
    in the same medium with values a factor near 2 apart (``check_panels_disagree`` for
    the Fig. 2c / Fig. 2f pair). ``replicate_id`` is the sheet's own column header.
    """
    if not math.isfinite(rate):
        raise TableFormatError(f"non-finite released rate {rate!r}")
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.growth_rate,
        environment_response=rate,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        units=ABSOLUTE_UNITS,
        screen_id=screen_id,
        replicate_id=replicate_id,
        provenance_gaps=[
            _GAP_ASSAY,
            *_single_measurement_gaps(_RESPONSE_DISPERSION_FIELDS),
        ],
    )


def culture_reference_phenotype(
    rates: Sequence[float], screen_id: str
) -> EnvironmentResponsePhenotype:
    """A background strain's released cultures in the panel's base medium, summarized.

    The only aggregate this class states: mean, sample SD and n over the released
    cultures, ``replicate_id = None`` because no one culture is what it reports, and the
    cultures it summarizes are themselves records. It needs two cultures for a sample
    SD; Fig. 2g's MG1655 M9 arm releases exactly two.
    """
    n = len(rates)
    if n < 2:
        raise TableFormatError(
            f"a reference needs two cultures for a sample SD, got {n}"
        )
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.growth_rate,
        environment_response=statistics.fmean(rates),
        environment_response_uncertainty=statistics.stdev(rates),
        environment_response_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=n,
        sample_unit=SampleUnit.biological_replicate,
        units=ABSOLUTE_UNITS,
        screen_id=screen_id,
        provenance_gaps=[_GAP_ASSAY],
    )


def metabolite_value_phenotype(value: MetaboliteValue) -> MetabolitePhenotype:
    """One released culture's metabolite level, verbatim, as the sheet prints it.

    ``measurement_type`` says what the number is: the compartment and the unit for
    pyruvate and ATP (the row label's own unit), and for the two Fig. 3d ratio rows the
    released reduced-over-oxidized ratio, keyed by the row label. One culture is one
    measurement, so ``metabolite_level_se`` is a ``not_applicable_single_measurement``
    gap.
    """
    return MetabolitePhenotype(
        metabolite_level={value.metabolite: value.value},
        n_replicates={value.metabolite: 1},
        measurement_type=value.measurement_type,
        replicate_id=value.replicate_id,
        provenance_gaps=_single_measurement_gaps(("metabolite_level_se",)),
    )


def metabolite_reference_phenotype(
    values: Sequence[MetaboliteValue],
) -> MetabolitePhenotype:
    """A strain's released cultures at one condition, summarized: mean, SE and n.

    The class carries a standard error, not an SD, so the summary's dispersion is the
    sample SD over sqrt(n). ``replicate_id`` is ``None``: the reference states the
    aggregate, and the cultures it summarizes are themselves records.
    """
    if len(values) < 2:
        raise TableFormatError(
            f"a metabolite reference needs two cultures, got {len(values)}"
        )
    (key,) = {v.metabolite for v in values}
    (measurement,) = {v.measurement_type for v in values}
    levels = [v.value for v in values]
    return MetabolitePhenotype(
        metabolite_level={key: statistics.fmean(levels)},
        metabolite_level_se={key: statistics.stdev(levels) / math.sqrt(len(levels))},
        n_replicates={key: len(levels)},
        measurement_type=measurement,
    )


def build_growth_genotype(symbols: Sequence[str], strain_label: str) -> Genotype:
    """One strain's designed deletions, each keyed by the symbol the release names."""
    return Genotype(
        perturbations=[
            ReleasedIdentifierPerturbation(
                systematic_gene_name=symbol,
                perturbed_gene_name=symbol,
                gene_namespace=MG1655_NAMESPACE,
                strain_label=strain_label,
                identifier_kind="gene symbol",
                released_kind=ReleasedKind.deletion,
                cassette=KAN_CASSETTE,
            )
            for symbol in symbols
        ]
    )


def reconstructed_genotype(allele: ReconstructedAllele) -> Genotype:
    """One reconstructed strain's allele, stored as Supplementary Table 3 releases it."""
    return Genotype(
        perturbations=[
            ReleasedIdentifierPerturbation(
                systematic_gene_name=allele.gene,
                perturbed_gene_name=allele.allele,
                gene_namespace=MG1655_NAMESPACE,
                strain_label=allele.strain,
                identifier_kind=released_identifier_kind(allele.gene),
                released_kind=ReleasedKind.sequence_allele,
                cassette=allele.cassette,
            )
        ]
    )


def control_background(allele: ReconstructedAllele) -> StrainBackground:
    """The kan-marked wild-type-allele control a reconstructed strain is a ratio to."""
    return StrainBackground(
        name=allele.control_strain,
        reference_strain=MG1655,
        assembly_set=MG1655_ASSEMBLY_SET,
        parents=[PARENT_LABEL],
        genotype_statement=allele.control_genotype_statement,
        alleles=[],
        provenance=[
            SOURCED_VALUES[f"{allele.control_strain}_genotype"],
            SOURCED_VALUES["parent_genotype_statement"],
        ],
    )


def control_phenotype() -> FitnessPhenotype:
    """A control: 1.0, the denominator of every released ratio; its spread is not."""
    return FitnessPhenotype(fitness=1.0, n_samples=None, sample_unit=None)


def build_keio_genotype(symbol: str) -> Genotype:
    """One Keio single deletion, keyed by the symbol the legend names."""
    return Genotype(
        perturbations=[
            ReleasedIdentifierPerturbation(
                systematic_gene_name=symbol,
                perturbed_gene_name=symbol,
                gene_namespace=BW25113_NAMESPACE,
                strain_label=f"Δ{symbol}",
                identifier_kind="gene symbol",
                released_kind=ReleasedKind.deletion,
                collection=KEIO_COLLECTION,
            )
        ]
    )


def keio_phenotype(fitness: float) -> FitnessPhenotype:
    """One Keio strain's growth rate as the released fraction of wild-type BW25113."""
    return FitnessPhenotype(fitness=fitness, n_samples=None, sample_unit=None)


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
# The released-ratio dataset (Fig. 2f)
# --------------------------------------------------------------------------- #
def _link_and_check(dataset: ExperimentDataset, names: Sequence[str]) -> None:
    """Link the mirrored files into ``raw/`` after checking manifest and sha256."""
    _link_raw(dataset.raw_dir, names)
    log.info("Choe 2019 raw files linked into %s (sha256 verified)", dataset.raw_dir)


@register_dataset
class GrowthRateChoe2019Dataset(ExperimentDataset):
    """Choe 2019 Fig. 2f: each released set's mutant-over-control growth-rate ratio."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = MG1655

    def __init__(
        self,
        root: str = GROWTH_DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the loader reads no annotation (#1045 part 2)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return FitnessExperimentReference

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
        _link_and_check(self, self.raw_file_names)

    def _raw(self, name: str) -> str:
        return osp.join(self.raw_dir, name)

    @post_process
    def process(self) -> None:
        """Build one record per released Fig. 2f set and write the LMDB (#1175)."""
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

        environment = growth_environment()
        pub = publication()

        def control(genome: ReferenceGenome) -> FitnessExperimentReference:
            return FitnessExperimentReference(
                dataset_name=self.name,
                genome_reference=genome,
                environment_reference=environment,
                phenotype_reference=control_phenotype(),
            )

        # (row label, strain label, reference strain, genotype, reference)
        arms: list[tuple[str, str, str, Genotype, FitnessExperimentReference]] = [
            (
                RATIO_ROW_ORDER[0],
                EVOLVED_LABEL,
                PARENT_LABEL,
                Genotype(perturbations=[]),
                control(
                    assembly_reference(
                        self.REFERENCE_STRAIN, background=evolved_background()
                    )
                ),
            ),
            *(
                (
                    allele.ratio_label,
                    allele.strain,
                    allele.control_strain,
                    reconstructed_genotype(allele),
                    control(
                        assembly_reference(
                            self.REFERENCE_STRAIN, background=control_background(allele)
                        )
                    ),
                )
                for allele in RECONSTRUCTED_ALLELES
            ),
        ]

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        rows: list[dict[str, Any]] = []
        index = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for label, strain, versus, genotype, reference in tqdm(
                arms, desc="choe2019 Fig. 2f sets"
            ):
                for replicate_id, ratio in zip(
                    RATIO_HEADERS[1:], ratios[label], strict=True
                ):
                    phenotype = released_ratio_phenotype(ratio, replicate_id)
                    experiment = FitnessExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
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
                            "panel": RATIO_SHEET,
                            "row_label": label,
                            "strain": strain,
                            "reference_strain": versus,
                            "replicate_id": replicate_id,
                            "fitness": phenotype.fitness,
                        }
                    )
        env.close()
        interned_env.close()

        if index != GROWTH_EXPECTED_RECORDS:
            raise RuntimeError(
                f"{index} records, the module states {GROWTH_EXPECTED_RECORDS}"
            )
        self._write_ledgers(panel, ratios, rows, recovery, disagreement, variants)
        log.info(
            "Choe 2019 Fig. 2f: %d records (%d references); fitness %.6f to %.6f",
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
        recovery: Mapping[str, Any],
        disagreement: Mapping[str, Any],
        variants: Sequence[VariantTable],
    ) -> None:
        """The retention ledger, the refused calls, the gaps and the cross-checks."""
        out = Path(self.preprocess_dir)
        total_calls = sum(table.rows for table in variants)
        released_sets = sum(len(values) for values in ratios.values())
        (out / "dropped_records.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "source_rows": len(ratios),
                    "source_values": released_sets,
                    "kept_records": len(rows),
                    "stored_by_replicate": {
                        replicate_id: sum(
                            1 for row in rows if row["replicate_id"] == replicate_id
                        )
                        for replicate_id in RATIO_HEADERS[1:]
                    },
                    "reference_rows": [a.control_strain for a in RECONSTRUCTED_ALLELES],
                    "dropped_records": released_sets - len(rows),
                    "rules": [],
                    "retired_rules": [
                        {
                            "reason": "evolved_clone_has_no_writable_genotype",
                            "retired": "no longer applied (#1045 part 3): eMS57 is "
                            "stored on a StrainBackground named eMS57, its "
                            "Supplementary Table 3 row the genotype statement",
                        },
                        {
                            "reason": "loader_mean_over_released_sets",
                            "retired": "no longer applied (#1175, 2026-11-24): each "
                            "released Fig. 2f set is its own record; the mean and "
                            "sample SD over the three sets are a modeling-time "
                            "derivation",
                        },
                        {
                            "reason": "fig2c_rates_as_parent_relative_fitness",
                            "retired": "no longer applied (#1175, 2026-11-24): the "
                            "Fig. 2c cultures are absolute growth rates in "
                            "AbsoluteGrowthRateChoe2019Dataset, not ratios over a "
                            "loader-computed parent mean in this class",
                        },
                    ],
                    "notes": [
                        f"{len(ratios)} released Fig. 2f rows x 3 sets = "
                        f"{released_sets} records: the three reconstructions and "
                        "eMS57/MS56, each set's ratio as the sheet prints it",
                        f"Fig. 2f releases three reconstructed single-allele strains "
                        f"({', '.join(RATIO_ROW_ORDER[1:])}) as mutant-over-control "
                        "growth-rate ratios for three sets. Since #1057 each is stored "
                        "on ReleasedIdentifierPerturbation (sequence_allele), the "
                        "gene and allele as Supplementary Table 3 writes them, against "
                        "its kan-marked wild-type-allele control",
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
                    "untyped_absences": [BACKGROUND_ABSENCE, EVOLVED_ABSENCE],
                    "typed_gaps": {
                        "environment": [_GAP_TEMPERATURE.model_dump(mode="json")],
                        "perturbation": [
                            gap.model_dump(mode="json")
                            for gap in GROWTH_PERTURBATION_GAPS
                        ],
                    },
                    "note": "ReleasedIdentifierPerturbation is not a "
                    "ProvenanceGapMixin, "
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
# The absolute growth-rate dataset (Fig. 2c, Fig. 2g)
# --------------------------------------------------------------------------- #
class CultureArm(BaseModel):
    """One released strain row of Fig. 2c or Fig. 2g and how its records are written.

    ``background`` names the strain the reference genome carries (the strain an empty
    genotype stands for, or the parent the deletions are made on); the reference is
    that background strain's own released cultures in the panel's base medium.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    panel: str
    strain: str
    medium: str
    background: str
    deleted_symbols: tuple[str, ...] = ()


@register_dataset
class AbsoluteGrowthRateChoe2019Dataset(ExperimentDataset):
    """Choe 2019 Fig. 2c and Fig. 2g: one absolute growth rate per released culture."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = MG1655

    def __init__(
        self,
        root: str = ABSOLUTE_DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the loader reads no annotation (#1045 part 2)."""
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
        """The Source Data workbook (sheets ``Fig2c`` and ``Fig2g``)."""
        return [SOURCE_DATA.name]

    def _consumed_sha256(self) -> dict[str, str]:
        """``{name: sha256}`` of the files this loader reads."""
        return {name: DATA_SHA256[name] for name in self.raw_file_names}

    def download(self) -> None:
        """Link the mirrored workbook into ``raw/`` after checking manifest and sha256."""
        _link_and_check(self, self.raw_file_names)

    def _genome(self, background: str) -> ReferenceGenome:
        """The reference genome a background label names (bare MG1655 for MG1655)."""
        if background == WILD_TYPE_LABEL:
            return assembly_reference(self.REFERENCE_STRAIN)
        if background == PARENT_LABEL:
            return assembly_reference(
                self.REFERENCE_STRAIN, background=parent_background()
            )
        if background == EVOLVED_LABEL:
            return assembly_reference(
                self.REFERENCE_STRAIN, background=evolved_background()
            )
        raise KeyError(f"no background is declared for {background!r}")

    @post_process
    def process(self) -> None:
        """Build one record per released culture of Fig. 2c and Fig. 2g (#1175)."""
        verify_raw_files(self.raw_dir, self._consumed_sha256())
        source = osp.join(self.raw_dir, SOURCE_DATA.name)
        panel = read_growth_panel(source)
        supplement = read_supplement_panel(source)
        recovery = check_recovery_fraction(panel)
        if len(supplement.dashes) != FIG2G_DASH_CELLS:
            raise TableFormatError(
                f"{SUPPLEMENT_SHEET} prints {len(supplement.dashes)} dashes, the "
                f"module states {FIG2G_DASH_CELLS}"
            )

        base = growth_environment()
        pub = publication()
        references: dict[tuple[str, str], EnvironmentResponseExperimentReference] = {}

        def reference(
            panel_id: str, background: str
        ) -> EnvironmentResponseExperimentReference:
            key = (panel_id, background)
            if key not in references:
                rates = (
                    panel.rates(background).replicates
                    if panel_id == GROWTH_SHEET
                    else supplement.base_rates(background)
                )
                references[key] = EnvironmentResponseExperimentReference(
                    dataset_name=self.name,
                    genome_reference=self._genome(background),
                    environment_reference=base,
                    phenotype_reference=culture_reference_phenotype(rates, panel_id),
                )
            return references[key]

        deletions = {
            LARGE_DELETION_LABEL: LARGE_DELETION_SYMBOLS,
            RPOS_DELETION_LABEL: (RPOS_SYMBOL,),
        }
        cells: list[tuple[CultureArm, str, float]] = []
        for row in panel.rows:
            arm = CultureArm(
                panel=GROWTH_SHEET,
                strain=row.strain,
                medium=BASE_MEDIUM_LABEL,
                background=PARENT_LABEL if row.strain in deletions else row.strain,
                deleted_symbols=deletions.get(row.strain, ()),
            )
            cells.extend(
                (arm, replicate_id, rate) for replicate_id, rate in row.labelled
            )
        for cell in supplement.rates:
            arm = CultureArm(
                panel=SUPPLEMENT_SHEET,
                strain=cell.strain,
                medium=cell.medium,
                background=cell.strain,
            )
            cells.append((arm, cell.replicate_id, cell.rate))

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        rows: list[dict[str, Any]] = []
        index = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for arm, replicate_id, rate in tqdm(cells, desc="choe2019 cultures"):
                phenotype = culture_rate_phenotype(rate, replicate_id, arm.panel)
                genotype = (
                    build_growth_genotype(arm.deleted_symbols, arm.strain)
                    if arm.deleted_symbols
                    else Genotype(perturbations=[])
                )
                experiment = EnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=supplemented_environment(arm.medium),
                    phenotype=phenotype,
                )
                record_reference = reference(arm.panel, arm.background)
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, record_reference, pub, itxn),
                )
                index += 1
                rows.append(
                    {
                        "panel": arm.panel,
                        "medium": arm.medium,
                        "strain": arm.strain,
                        "background": arm.background,
                        "deleted_genes": len(arm.deleted_symbols),
                        "replicate_id": replicate_id,
                        "growth_rate": rate,
                        "reference_growth_rate_mean": (
                            record_reference.phenotype_reference.environment_response
                        ),
                    }
                )
        env.close()
        interned_env.close()

        if index != ABSOLUTE_EXPECTED_RECORDS:
            raise RuntimeError(
                f"{index} records, the module states {ABSOLUTE_EXPECTED_RECORDS}"
            )
        if len(references) != ABSOLUTE_EXPECTED_REFERENCES:
            raise RuntimeError(
                f"{len(references)} references, the module states "
                f"{ABSOLUTE_EXPECTED_REFERENCES}"
            )
        self._write_ledgers(panel, supplement, rows, recovery)
        log.info(
            "Choe 2019 Fig. 2c + 2g: %d culture records (%d references)",
            index,
            len(references),
        )

    def _write_ledgers(
        self,
        panel: GrowthPanel,
        supplement: SupplementPanel,
        rows: Sequence[Mapping[str, Any]],
        recovery: Mapping[str, Any],
    ) -> None:
        """The retention ledger (every released cell accounted for) and the checks."""
        out = Path(self.preprocess_dir)
        fig2c_cells = sum(row.n_samples for row in panel.rows)
        fig2g_cells = len(supplement.rates) + len(supplement.dashes)
        (out / "dropped_records.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "source_cells": {
                        GROWTH_SHEET: fig2c_cells,
                        SUPPLEMENT_SHEET: fig2g_cells,
                    },
                    "kept_records": len(rows),
                    "stored_by_panel": {
                        panel_id: sum(1 for row in rows if row["panel"] == panel_id)
                        for panel_id in (GROWTH_SHEET, SUPPLEMENT_SHEET)
                    },
                    "stored_by_replicate": {
                        replicate_id: sum(
                            1 for row in rows if row["replicate_id"] == replicate_id
                        )
                        for replicate_id in GROWTH_HEADERS[1:]
                    },
                    "dropped_records": 0,
                    "unreleased_cells": [
                        dash.model_dump(mode="json") for dash in supplement.dashes
                    ],
                    "rules": [],
                    "notes": [
                        f"{len(panel.rows)} Fig. 2c strain rows x 3 cultures = "
                        f"{fig2c_cells} records, MG1655, MS56 and eMS57 included",
                        f"Fig. 2g releases {fig2g_cells} cells: {len(supplement.rates)} "
                        f"numbers are records and {len(supplement.dashes)} print a "
                        "dash, which is no released value (MG1655 M9 Rep3 and the "
                        "three MG1655 M9 + valine cells)",
                        "each record is one culture's absolute growth rate verbatim; "
                        "a ratio to a parent is a modeling-time derivation over these "
                        "records",
                        "each reference is the background strain's released cultures "
                        "in the panel's M9 glucose: mean, sample SD, n, replicate_id "
                        "None",
                        "each deletion is keyed by the gene symbol the release names "
                        "(#1045 part 2); where the pinned MG1655 assembly places them "
                        f"is the verification row {DELETED_REGION_RULE}",
                    ],
                },
                indent=2,
            )
        )
        (out / "released_statistics_check.json").write_text(
            json.dumps(
                {
                    "recovery_fraction": dict(recovery),
                    "fig2c_strain_means": {row.strain: row.mean for row in panel.rows},
                },
                indent=2,
            )
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {
                    name: value.model_dump(mode="json")
                    for name, value in SOURCED_VALUES.items()
                    if not name.startswith("keio")
                },
                indent=2,
            )
        )
        pd.DataFrame(list(rows)).to_csv(out / "cultures.csv", index=False)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "AbsoluteGrowthRateChoe2019Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# The metabolite dataset (Fig. 1g, Fig. 3d)
# --------------------------------------------------------------------------- #
@register_dataset
class MetaboliteChoe2019Dataset(ExperimentDataset):
    """Choe 2019 Fig. 1g and Fig. 3d: one metabolite level per released culture cell."""

    organism = Organism.ecoli
    #: MG1655 and eMS57 only: every genotype is empty, so the gene set is empty.
    has_gene_perturbations = False

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = MG1655

    def __init__(
        self,
        root: str = METABOLITE_DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the loader reads no annotation (#1045 part 2)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return MetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return MetaboliteExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Source Data workbook (sheets ``Fig1g`` and ``Fig3d``)."""
        return [SOURCE_DATA.name]

    def _consumed_sha256(self) -> dict[str, str]:
        """``{name: sha256}`` of the files this loader reads."""
        return {name: DATA_SHA256[name] for name in self.raw_file_names}

    def download(self) -> None:
        """Link the mirrored workbook into ``raw/`` after checking manifest and sha256."""
        _link_and_check(self, self.raw_file_names)

    def _genome(self, strain: str) -> ReferenceGenome:
        """Bare MG1655, or eMS57 on its released background."""
        if strain == WILD_TYPE_LABEL:
            return assembly_reference(self.REFERENCE_STRAIN)
        if strain == EVOLVED_LABEL:
            return assembly_reference(
                self.REFERENCE_STRAIN, background=evolved_background()
            )
        raise KeyError(f"no background is declared for {strain!r}")

    @staticmethod
    def _environment(value: MetaboliteValue) -> Environment:
        """The culture condition one value was measured in."""
        if value.sheet == PYRUVATE_SHEET:
            if value.hours is None:
                raise TableFormatError("a Fig. 1g value without its timepoint")
            return pyruvate_environment(value.hours)
        return cofactor_environment()

    @post_process
    def process(self) -> None:
        """Build one record per released Fig. 1g and Fig. 3d culture cell (#1175)."""
        verify_raw_files(self.raw_dir, self._consumed_sha256())
        source = osp.join(self.raw_dir, SOURCE_DATA.name)
        pyruvate, od_cells = read_pyruvate_panel(source)
        cofactors = read_cofactor_panel(source)
        folds = check_cofactor_fold_changes(cofactors)
        if od_cells != FIG1G_OD_CELLS:
            raise TableFormatError(
                f"{PYRUVATE_SHEET} releases {od_cells} OD600 cells, the module states "
                f"{FIG1G_OD_CELLS}"
            )
        values = (*pyruvate, *cofactors)
        groups: dict[tuple[str, str, str, float | None], list[MetaboliteValue]] = {}
        for value in values:
            key = (value.sheet, value.strain, value.measurement_type, value.hours)
            groups.setdefault(key, []).append(value)
        references = {
            key: MetaboliteExperimentReference(
                dataset_name=self.name,
                genome_reference=self._genome(members[0].strain),
                environment_reference=self._environment(members[0]),
                phenotype_reference=metabolite_reference_phenotype(members),
            )
            for key, members in groups.items()
        }
        pub = publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        rows: list[dict[str, Any]] = []
        index = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for value in tqdm(values, desc="choe2019 metabolites"):
                experiment = MetaboliteExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=[]),
                    environment=self._environment(value),
                    phenotype=metabolite_value_phenotype(value),
                )
                reference = references[
                    (value.sheet, value.strain, value.measurement_type, value.hours)
                ]
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                index += 1
                rows.append(value.model_dump(mode="json"))
        env.close()
        interned_env.close()

        if index != METABOLITE_EXPECTED_RECORDS:
            raise RuntimeError(
                f"{index} records, the module states {METABOLITE_EXPECTED_RECORDS}"
            )
        if len(references) != METABOLITE_EXPECTED_REFERENCES:
            raise RuntimeError(
                f"{len(references)} references, the module states "
                f"{METABOLITE_EXPECTED_REFERENCES}"
            )
        self._write_ledgers(rows, od_cells, folds)
        log.info(
            "Choe 2019 Fig. 1g + 3d: %d metabolite records (%d references)",
            index,
            len(references),
        )

    def _write_ledgers(
        self, rows: Sequence[Mapping[str, Any]], od_cells: int, folds: Mapping[str, Any]
    ) -> None:
        """The retention ledger (every released cell accounted for) and the checks."""
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "source_cells": {
                        PYRUVATE_SHEET: sum(
                            1 for row in rows if row["sheet"] == PYRUVATE_SHEET
                        )
                        + od_cells,
                        COFACTOR_SHEET: sum(
                            1 for row in rows if row["sheet"] == COFACTOR_SHEET
                        ),
                    },
                    "kept_records": len(rows),
                    "stored_by_measurement_type": {
                        measurement: sum(
                            1 for row in rows if row["measurement_type"] == measurement
                        )
                        for measurement in sorted(METABOLITE_PROTOCOLS)
                    },
                    "dropped_records": 0,
                    "not_records": [
                        {
                            "cells": od_cells,
                            "sheet": PYRUVATE_SHEET,
                            "row": PYRUVATE_OD_ROW,
                            "reason": "growth_curve_point_has_no_phenotype_class",
                            "detail": "the OD600 of each culture at 4, 6 and 8 h is a "
                            "point of a growth curve, which no phenotype class holds "
                            "(the Fig. 1a, 1b, 1d and 1h curves are refused for the "
                            "same reason)",
                        }
                    ],
                    "rules": [],
                    "notes": [
                        "each record is one released cell verbatim: one culture's "
                        "level at one condition, n_replicates 1, replicate_id the "
                        "sheet's own block or column label",
                        "Fig. 1g's three timepoints of one culture are three "
                        "environments (duration_hours 4, 6, 8 after inoculation), so "
                        "a culture is three records per pyruvate row",
                        "Fig. 3d releases two NADH/NAD+ and two NADPH/NADP+ cells per "
                        "strain and three ATP cells, against a legend that states "
                        "three biological replicates; every released cell is stored",
                        "each reference is the strain's own released cultures at the "
                        "same condition: mean, SE (sample SD / sqrt n), n, "
                        "replicate_id None",
                    ],
                },
                indent=2,
            )
        )
        (out / "released_statistics_check.json").write_text(
            json.dumps({"cofactor_fold_changes": dict(folds)}, indent=2)
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {
                    name: value.model_dump(mode="json")
                    for name, value in SOURCED_VALUES.items()
                    if name.startswith(
                        ("fig1g", "fig3d", "pyruvate", "cofactor", "media")
                    )
                },
                indent=2,
            )
        )
        pd.DataFrame(list(rows)).to_csv(out / "values.csv", index=False)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "MetaboliteChoe2019Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# The Keio transcription-factor dataset
# --------------------------------------------------------------------------- #
@register_dataset
class TranscriptionFactorKnockoutChoe2019Dataset(ExperimentDataset):
    """Choe 2019 Supplementary Fig. 6: the two Keio deletions that carry a number."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = BW25113

    def __init__(
        self,
        root: str = TF_DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the loader reads no annotation (#1045 part 2)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return FitnessExperimentReference

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

    @post_process
    def process(self) -> None:
        """Build the two Keio fitness records and write the LMDB."""
        verify_raw_files(self.raw_dir, self._consumed_sha256())
        environment = keio_environment()
        reference = FitnessExperimentReference(
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
                experiment = FitnessExperiment(
                    dataset_name=self.name,
                    genotype=build_keio_genotype(symbol),
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
                        "fitness": phenotype.fitness,
                    }
                )
        env.close()
        interned_env.close()

        if index != TF_EXPECTED_RECORDS:
            raise RuntimeError(
                f"{index} records, the module states {TF_EXPECTED_RECORDS}"
            )
        self._write_ledgers(rows)
        log.info(
            "Choe 2019 Supplementary Fig. 6: %d Keio records (+ %d BW25113 reference) "
            "out of a %d-strain panel",
            index,
            TF_EXPECTED_REFERENCES,
            KEIO_PANEL_STRAINS,
        )

    def _write_ledgers(self, rows: Sequence[Mapping[str, Any]]) -> None:
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
                        "each deletion is keyed by the gene symbol the legend names "
                        "(#1045 part 2); the loader reads no annotation",
                    ],
                },
                indent=2,
            )
        )
        (out / "provenance_gaps.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "carrier": "FitnessPhenotype / ReleasedIdentifierPerturbation",
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
    extra: Callable[[EcoliK12Genome], Sequence[LevelResult]] = lambda genome: (),
) -> VerificationReport:
    """Run the fitness L0-L4 gate over one built tree and write its report.

    ``extra`` adds a tree's own rows over the same genome (the growth tree's
    genome-reference region row, #1045 part 2).
    """
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
    for result in extra(genome):
        report.add(result)
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
    """L0-L4 over the released-ratio (Fig. 2f) tree."""
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
                "Source Data sheet 'Fig2f' Set1-Set3 mutant-over-control growth-rate "
                "ratios: one FitnessExperiment per released set, the ratio verbatim, "
                "n_samples 1, replicate_id the set label"
            ),
            page=f"si12.xlsx sheet '{RATIO_SHEET}'",
            retrieved=RETRIEVED_AT,
        ),
    )


def verify_absolute_growth_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    genome: EcoliK12Genome | None = None,
    expected_count: int = ABSOLUTE_EXPECTED_RECORDS,
) -> VerificationReport:
    """L0-L4 over the absolute growth-rate (Fig. 2c, Fig. 2g) tree.

    ``reference_centered=False`` selects the absolute branch: each reference states its
    own finite rate on the records' scale, and the branch refuses any record whose
    ``measurement_type`` is not an absolute one. ``expected_unperturbed`` is the 20
    records in M9 glucose with no supplement, an oracle the build must equal.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset,
    )
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    if genome is None:
        genome = bacterial_genome("ecoli", MG1655, data_root)
    report = verify_environment_response_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{SOURCE_DATA.mirror_relpath}",
            citation_key=CITATION_KEY,
            sha256=SOURCE_DATA.sha256,
            method=(
                "Source Data sheets 'Fig2c' and 'Fig2g' Rep1-Rep3 growth rates: one "
                "EnvironmentResponseExperiment (MeasurementType.growth_rate) per "
                "released culture, the rate verbatim, n_samples 1, replicate_id the "
                "column header, screen_id the sheet"
            ),
            page=f"si12.xlsx sheets '{GROWTH_SHEET}' and '{SUPPLEMENT_SHEET}'",
            retrieved=RETRIEVED_AT,
        ),
        expected_count=expected_count,
        reference_centered=False,
        expected_unperturbed=ABSOLUTE_EXPECTED_UNPERTURBED,
        sgd_genes=set(genome.genbank.loci),
        gene_universe_label=f"{MG1655} ({genome.ASSEMBLY_SET})",
        resolve_gene_name=genome.resolve_gene_name,
    )
    report.add(deleted_region_rule(genome))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def verify_metabolite_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    expected_count: int = METABOLITE_EXPECTED_RECORDS,
) -> VerificationReport:
    """L0-L3 over the metabolite (Fig. 1g, Fig. 3d) tree.

    The five readouts are declared protocols (``protocol_measurement_types``), so L1
    keys on (strain, protocol) and L3 requires each record's reference to share its
    protocol; ``environment_keyed`` adds the environment to the key, since one Fig. 1g
    culture is measured at three timepoints. ``reference_centered=False``: every level
    is absolute and every reference states its own finite value. ``data_root`` is
    unused (no genotype carries a gene) and kept for the shared call shape.
    """
    from torchcell.verification.metabolite import verify_metabolite_dataset
    from torchcell.verification.runners import load_records

    del data_root
    records = load_records(dataset_root)
    report = verify_metabolite_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{SOURCE_DATA.mirror_relpath}",
            citation_key=CITATION_KEY,
            sha256=SOURCE_DATA.sha256,
            method=(
                "Source Data sheets 'Fig1g' (pyruvate per culture block x 4, 6, 8 h) "
                "and 'Fig3d' (NADH/NAD+, NADPH/NADP+ Rep1-Rep2, ATP Rep1-Rep3): one "
                "MetaboliteExperiment per released cell, the level verbatim, "
                "n_replicates 1, replicate_id the sheet's label"
            ),
            page=f"si12.xlsx sheets '{PYRUVATE_SHEET}' and '{COFACTOR_SHEET}'",
            retrieved=RETRIEVED_AT,
        ),
        expected_count=expected_count,
        reference_centered=False,
        protocol_measurement_types=METABOLITE_PROTOCOLS,
        environment_keyed=True,
    )
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


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
        command.add_argument(
            "--arm",
            choices=("growth", "absolute", "metabolite", "tf", "all"),
            default="all",
        )
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    if args.command == "deposit":
        print(f"raw mirror at {deposit_raw_mirror(data_root)}")
        return 0

    arms: list[tuple[str, str, Any, Any]] = []
    if args.arm in ("growth", "all"):
        arms.append(
            (
                "growth",
                GROWTH_DATASET_ROOT_REL,
                GrowthRateChoe2019Dataset,
                verify_growth_build,
            )
        )
    if args.arm in ("absolute", "all"):
        arms.append(
            (
                "absolute",
                ABSOLUTE_DATASET_ROOT_REL,
                AbsoluteGrowthRateChoe2019Dataset,
                verify_absolute_growth_build,
            )
        )
    if args.arm in ("metabolite", "all"):
        arms.append(
            (
                "metabolite",
                METABOLITE_DATASET_ROOT_REL,
                MetaboliteChoe2019Dataset,
                verify_metabolite_build,
            )
        )
    if args.arm in ("tf", "all"):
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
            table = {"absolute": "cultures.csv", "metabolite": "values.csv"}.get(
                label, "strains.csv"
            )
            print(Path(root, "preprocess", table).read_text())
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
