# torchcell/datasets/pputida/lim2025
# [[torchcell.datasets.pputida.lim2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/lim2025
# Test file: tests/torchcell/datasets/pputida/test_lim2025.py
"""Lim 2025 isoprenol TALE of P. putida KT2440: the WRITABLE strains only.

Lim et al. 2025 (Metab. Eng., doi:10.1016/j.ymben.2025.05.007) ran a 16-lineage
tolerization adaptive-laboratory-evolution (TALE) campaign on KT2440 and two stacked
deletion strains, resequenced 46 evolved isolates, reverse-engineered seven deletion
strains from the convergent targets, and profiled the parent and two evolved isolates by
shotgun proteomics. Two dataset classes serve what is BOTH released as a number AND keyed
to a strain this schema can write:

- :class:`IsoprenolToleranceLim2025Dataset` -- ``BacterialEnvironmentResponseExperiment``,
  three records: IPL300 and IPL400 at the TALE starting dose of 4 g/L isoprenol
  (Supplementary Table 3's initial growth rates), and KT2440 ``dPP_3024`` at 6 g/L
  isoprenol (the one per-strain fold the Fig. 3A text states).
- :class:`ProteomeLim2025Dataset` -- ``BacterialProteinAbundanceExperiment``, three
  records: the parent IPL400 and the two evolved isolates the release quantifies
  (A10_F63_I1, A12_F53_I1) under 4 g/L isoprenol, each strain's Top3 log2 protein
  abundances referenced to its OWN unstressed proteome (2,361 / 2,361 / 2,332 keys).

THE CALLED VARIANTS ARE WRITTEN (issue #731), AND THE CLONE COLUMN IS THE CARRIER.
An evolved clone's genotype is its parent's genomic content plus the variants breseq
called from its resequencing. Supplementary Data 1 (``si/si2.xlsx``, sheet
``Fig 2B_Mutation List``) releases exactly that: 159 rows, each a position on
``AE015451`` -- the single replicon of the pinned ``pputida_KT2440_ASM756v2`` assembly --
with a mutation type (100 SNP, 47 DEL, 11 INS, 1 SUB), a sequence change (``G->A``,
``+C``, the delta-bp form of a deletion), the gene, gene pair or gene run it falls in,
and a detail field (87 amino-acid substitutions, 30 coding-span indels, 17 intergenic, 25
blank), crossed against 49 clone columns as 443 per-clone calls (431 at frequency 1, 12 at
0.9). The three leaves #731 added hold all of it, and the released row SHAPE decides
which (:func:`released_row_representation`, measured and partitioning all 159 rows):

1. ``Details`` starts ``intergenic`` (17 rows) -> ``BacterialSiteVariantPerturbation``
   with ``site_kind=intergenic``, keyed on the derived site id
   ``AE015451:<position>``, with both flanking loci resolved into
   ``flanking_systematic_gene_names`` and the ``Gene`` cell verbatim in
   ``flanking_gene_statement``. A locus tag there is refused by the leaf itself, so no
   neighbor is ever invented.
2. ``Mutation Type == DEL`` with an EMPTY ``Details`` (25 rows) ->
   ``BacterialSpanDeletionPerturbation``, one per covered locus, all sharing the event's
   ``span_designation``. breseq reports no coding offset exactly when the named loci lie
   wholly inside the deleted interval; 19 of the 25 name more than one locus (up to the
   53 of ``PP_3024``-``PP_5558``) and 6 name one, which is the same shape.
3. everything else (117 rows) -> ``BacterialSequenceVariantPerturbation`` on its one
   locus. ``PP_3415`` carries TWO of these within A12_F53_I1 (P293S at 3,866,001 and
   V46I at 3,866,742), which ``Genotype.perturbations`` admits: it has no
   one-entry-per-locus rule, unlike ``BacterialStrainBackground``.

THE ``Gene`` CELL IS USUALLY A SYMBOL, AND RESOLVING IT IS NOT OPTIONAL. Measured over
all 241 distinct ``Gene`` tokens of the sheet, 151 are already KT2440 locus tags and 90
resolve through the gene-symbol layer of the pinned assembly: ``ttgB`` is ``PP_1385``,
``adhP`` is ``PP_3839``, ``ivd,mccB,liuC,mccA`` are ``PP_4064``-``PP_4067``. Every stored
tag that differs from what the release named carries a
``DerivedIdentifierMapping(route="gene_symbol")``, and a token that does not land on
exactly one locus is a HARD ERROR with its name (measured: one such token, ``asd`` ->
``PP_1989`` or ``PP_1992``), never a guessed mapping.

NO END COORDINATE IS ASSERTED. The release gives a 1-based ``Position`` and a LENGTH
fused into ``Sequence Change`` (``D5,553 bp``, ``(CCAC)2->1``, ``2 bp->CG``), and never an
end. ``position_end`` therefore equals ``position_start`` on EVERY call, the released
extent stays verbatim in ``sequence_change``, and ``deleted_span`` is left as a None on
the span leaf. Synthesizing an end would mean parsing that string and choosing which side
of the position the length runs to, neither of which the source states.

A CALL THAT RESTATES A DESIGNED LESION IS DROPPED, AND THE DROP IS COUNTED. A call whose
resolved locus the record's own strain already carries as a designed
``BacterialDeletionPerturbation`` is not written a second time, so absence has one
encoding. Measured, and larger than it looks before the symbols are resolved: ``3063719
DEL PP_2675``, ``4362918 DEL adhP`` (``PP_3839``) and the four loci of ``4588139 DEL
ivd,mccB,liuC,mccA`` (``PP_4064``-``PP_4067``) restate IPL300's six designed deletions,
and ``1578244 DEL ttgB`` (``PP_1385``) restates IPL400's seventh. Per loaded record that
is 6 of IPL300's 9 candidate perturbations and 7 of the other three records' 11, 19 and
21. One consequence: every span-deletion row the four loaded clone columns carry is
entirely restated, so the span leaf's code path runs on all of them and ZERO span
perturbations survive onto a loaded record.

WHICH CLONE COLUMN IS WHICH STRAIN IS PROVEN, NOT ASSUMED
(:func:`assert_founder_columns`). An ``F0`` column is its lineage's founding starting
strain under the ALEdb convention the paper states, and Supplementary Table 3's row
groups put ALE 1-4 on KT2440, 5-8 on IPL300 and 9-12 on IPL400. Three measured facts have
to hold together or the build refuses: ``A1 F0 I1 R1`` carries ZERO calls (the matrix is
called against KT2440 WT, so its own founder must), ``A9 F0 I1 R1``'s 8 call positions are
a strict superset of ``A5 F0 I1 R1``'s 6, and what IPL400 adds is exactly ``ttgB`` and
``PP_4398`` -- the Results' own "IPL400 had the same mutations along with a mutation in
PP_4398".

WHAT STILL REFUSES, WITH COUNTS.

- The 16 LINEAGE final growth rates of Supplementary Table 3. Each is the average over a
  lineage's three LAST flasks, so its strain is the evolving POPULATION in that flask and
  not one of the 46 sequenced isolates. The matrix calls CLONES
  (``VariantCallMode.clone``) and releases no population allele frequency for a flask, so
  a population genotype would need a threshold the release never gives. The #731 leaves
  do not change this: they hold a clone's calls.
- The 46 evolved isolates as TOLERANCE records. They are writable now, but
  Supplementary Table 3 releases a growth rate per lineage and not per isolate, and
  Fig. 2A's per-isolate rates are a figure with no numbers. The two whose proteome IS
  released are records of :class:`ProteomeLim2025Dataset`.
- ``PP_2676``'s 14-codon N-terminal truncation on the PERTURBATION axis. The release
  gives no coordinate for it and no coding range, so it cannot be a
  ``BacterialSequenceVariantPerturbation`` of deletion type either; it stays a
  ``BacterialBackgroundAllele(partial_deletion)`` on the IPL400 background and a stated
  gap in ``preprocess/genotype_gaps.json``.
- A called variant on the strain-BACKGROUND axis. ``BacterialBackgroundAllele`` requires
  a non-optional ``functional: bool`` a call's consequence is unknown for, and
  ``BacterialStrainBackground`` permits one allele per locus, which refuses A12_F53_I1's
  two ``PP_3415`` calls. Each proteome record's ``genome_reference`` therefore carries
  IPL400's background, which for the two evolved isolates is a FLOOR on the genomic
  content of their own unstressed reference arm. The second
  ``preprocess/genotype_gaps.json`` entry states it.

Every one of the 159 rows is typed into ``preprocess/called_variants.json`` with the leaf
its shape maps to, and the per-record accounting (candidates, written, restated, resolved
identifiers) into ``preprocess/called_variant_perturbations.json``.


THE WRITABLE STRAINS, AND WHY ONLY THREE CARRY A NUMBER. IPL300
(``dPP_2675 dPP_3839 dPP_4064-dPP_4067``), IPL400 (the same plus ``dttgB``/``PP_1385``)
and the seven reverse-engineered KT2440 deletions are all expressible as
``BacterialDeletionPerturbation``s on the ``pputida_kt2440_locus_tag`` namespace, and the
eight pIY670 strains as ``HeterologousPathwayPerturbation``s. The campaign releases a
per-strain NUMBER for only three of them: Supplementary Table 3's initial growth rates
(IPL300, IPL400) and the 1.6-fold the Fig. 3A text gives for ``dPP_3024``. Figures 2A, 3A,
3B, 4D, 5A, 5B, 5D, 5E and S1C-F are the rest of the phenotyping and release no numbers at
all, so the other six deletion strains and all eight pIY670 strains are writable with
nothing to write about them. Every count is in ``preprocess/dropped_records.json``.

PP_2676'S TRUNCATION IS TYPED WHERE THE FRAMING ADMITS IT, AND GAPPED WHERE IT DOES NOT.
Supplementary Table 1 writes the parent genotype as "KT2440 with the complete internal
in-frame deletion of PP_2675 and partial truncation of the first fourteen amino acids of
PP_2676" -- one deletion event, since ``PP_2676``'s ORF overlaps ``PP_2675``'s CDS, and
the same physical lesion row 14's ``PT`` strain carries from the same Thompson 2020
strain. The bacterial PERTURBATION axis has no partial-deletion leaf
(``BacterialDeletionPerturbation`` is ``state="absent"``), so:

- :class:`ProteomeLim2025Dataset`, whose comparison is IPL400-stressed against
  IPL400-unstressed, states the whole strain on a ``BacterialStrainBackground``: seven
  ``full_deletion`` alleles plus ``PP_2676`` as a ``partial_deletion``, each with a typed
  ``deleted_span`` gap (no source gives coordinates). That is exactly how row 14 types
  its ``PT`` background, and it is where the truncation CAN be said.
- :class:`IsoprenolToleranceLim2025Dataset`, whose comparison is strain-against-KT2440,
  carries the reference-strain genome with NO background (the Wang 2015 pattern, because
  the reference ARM is the wild type) and the whole lesion set as perturbations. There
  the truncation has no home and stays a stated gap in
  ``preprocess/genotype_gaps.json``. A bacterial truncation / partial-deletion
  perturbation leaf is the fourth item of the additive proposal.

WHY A LOG2 RATIO AND NOT AN ABSOLUTE GROWTH RATE. ``MeasurementType.growth_rate`` exists,
but ``torchcell.verification.environment_response``'s L3 ``reference_zero`` requires a
numeric reference of exactly 0, which an absolute rate in h^-1 cannot be. Both released
quantities are already comparisons of one strain to KT2440 in the SAME medium at the SAME
isoprenol dose by the same liquid-OD assay, so ``log2_ratio`` is the faithful encoding and
the reference is KT2440 at log2(1) = 0 -- the Wang 2015 pattern for this same compound.
Raising the verifier to accept a declared non-zero numeric baseline is in the PR body.

THE FOUR TALE LINEAGES PER STRAIN ARE REPLICATES, SO THEY AGGREGATE. "Each TALE experiment
was conducted with four independent biological replicates in one high-throughput evolution
campaign", so ALE 1-4 (WT), 5-8 (IPL300) and 9-12 (IPL400) are four measurements of one
(strain, condition) and L1 requires them in one record with ``n_samples``. Each arm's
value is the mean of its four lineages' Supplementary Table 3 initial growth rates;
the per-lineage numbers stay in ``preprocess/table_s3.csv``.

UNCERTAINTY IS A TYPED GAP ON THE TOLERANCE RECORDS. Supplementary Table 3 reports no
dispersion for any lineage, and the stored value divides by the WT arm's mean, so a sample
SD over the four numerators would propagate only one of the two arms. Both
``environment_response_uncertainty`` and ``environment_response_se`` are typed gaps, as in
Wang 2015; ``n_samples = 4`` biological replicates is recorded.

THE PROTEOME'S n AND UNCERTAINTY TYPE ARE BACK-SOLVED, NOT ASSUMED. The sheets release a
per-arm ``log2_mean`` and ``log2_std`` and a ``t-test_stat``. Taking the std as the SAMPLE
SD and n = 3 per arm reproduces Welch's t for every kept row to 1.1e-12 (worst over 2,361
and 2,368 rows); taking it as a population SD misses by a median of 0.45 t units. The
released ``log2_Fold_change_A/B`` equals the difference of the two means to 9.3e-15. So
``protein_abundance_se = log2_std / sqrt(3)`` and ``n_replicates = 3``, which agrees with
the ProteomeXchange Sample Key's own ``R1,R2,R3`` and with "Samples were analyzed as three
independent biological replicates". The Welch identity is asserted at build time.

THREE CROSS-SOURCE ASSERTIONS, all measured to hold on the pinned bytes:

1. ``Proteome_A12F53I1vsIPL400_M9G``'s IPL400 columns are bit-identical to
   ``Proteome_A10F63I1vsIPL400_M9G``'s over all 2,367 rows, and the two ``G+4IP`` sheets
   agree exactly on their 2,338 shared loci (the A10 sheet carries 36 more). One IPL400
   measurement, exported four times; it becomes one record, never averaged.
2. Supplementary Table 3's ``Generations`` and ``CCD`` columns reproduce the Results'
   stated ranges: 143-353 generations and 1.33-3.32e12 CCD for the isoprenol arm,
   346-387 and 3.11-3.56e12 for the HCHO arm.
3. The HCHO arm's doses agree across units: Table 3's 1 mM and 9 mM are the Results'
   0.03 g/L and 0.27 g/L at formaldehyde's 30.026 g/mol.

SIX PROTEIN KEYS ARE DROPPED BY A MEASURED RULE. The sheets file three gene symbols
(``Ubid``, ``Pyrc``, ``Dapa``) under two paralogous locus tags each (PP_0548/PP_5213,
PP_1086/PP_4999, PP_1237/PP_2639) with DIFFERENT UniProt accessions but IDENTICAL means
and SDs, and those are exactly the rows whose released t does not reproduce at n = 3. One
protein group's statistics cannot be attributed to either paralog, so all six rows go, and
dropping them is what makes the Welch identity hold everywhere else. Seven further loci
(PP_0002, PP_0416, PP_0985, PP_2271, PP_3610, PP_3699, PP_5287 -- the seven the ``G+4IP``
sheet title-cases as ``Pp_0002``) are released under isoprenol but not in the unstressed
reference arm, so they have no key-matched reference value and are dropped from the record.

MEDIA. ``M9_NREL_LIM2025`` is this paper's own library entry (M9 at 4 g/L glucose) and
serves every loaded record. The three pIY670 production-culture proteome arms (12, 24 and
48 h) are at 20 g/L glucose, which has no ``MEDIA_LIBRARY`` entry; ``media.py`` is a
value-surface file this branch does not touch, so those three arms are NOT loaded and the
needed addition is in the PR body.

COMPOUND. ``isoprenol`` resolves through ``compound_identity_table.json``; its row landed
with the bacterial schema follow-ups (PR #729). :data:`ISOPRENOL_INCHIKEY` =
``CPJRRXSHAYUTGL-UHFFFAOYSA-N`` pins the identity this module was written against, and the
build stops if the table's row ever disagrees with it.

DATA. Both consumed files are the publisher's Elsevier supplements, deposited under
``$DATA_ROOT/torchcell-raw/limEvolutionguidedToleranceEngineering2025/data/``:
``si1.docx`` (Supplementary Notes, Figures and Tables 1-7) and ``si2.xlsx``
(Supplementary Data 1: the mutation matrix, the Table S4 gene lists, the ProteomeXchange
sample key and seven proteome comparison sheets). The three sequencing deposits are
RECORDED and not downloaded: SRA BioProject PRJNA1187681 (genome resequencing),
GEO GSE281392 (transcriptome) and PRIDE PXD054609 (raw DIA spectra). No loader consumes
reads or spectra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import os.path as osp
import re
import shutil
import statistics
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal
from xml.etree import ElementTree

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
    write_verified,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import M9_NREL_LIM2025
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AlleleEdit,
    AssayType,
    BacterialAssemblySet,
    BacterialBackgroundAllele,
    BacterialDeletionPerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialGeneNamespace,
    BacterialProteinAbundanceExperiment,
    BacterialProteinAbundanceExperimentReference,
    BacterialSequenceVariantPerturbation,
    BacterialSiteVariantPerturbation,
    BacterialSpanDeletionPerturbation,
    BacterialStrainBackground,
    BacterialVariantCall,
    BacterialVariantType,
    Compound,
    Concentration,
    ConcentrationUnit,
    DerivedIdentifierMapping,
    Environment,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    GenePerturbationType,
    Genotype,
    MeasurementType,
    ProteinAbundancePhenotype,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainConstruction,
    VariantCallMode,
    VariantFrequencyBasis,
    VariantSiteKind,
)
from torchcell.datasets.bacteria_common import (
    LOCUS_TAG_PATTERNS,
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
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
from torchcell.literature.provenance import run_retriever
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
    audit_sourced_value,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "limEvolutionguidedToleranceEngineering2025"
PAPER_DOI = "10.1016/j.ymben.2025.05.007"
PAPER_PII = "S1096717625000837"
PAPER_TITLE = (
    "Evolution-guided tolerance engineering of Pseudomonas putida KT2440 for production "
    "of the aviation fuel precursor isoprenol"
)
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
TOLERANCE_ROOT_REL = "data/torchcell/isoprenol_tolerance_lim2025"
PROTEOME_ROOT_REL = "data/torchcell/proteome_lim2025"

#: MinerU OCR of the publisher PDF in the literature mirror; every ``SourcedValue``
#: quotes a verbatim substring of exactly these bytes, which is what makes the
#: provenance audit meaningful (a docx or xlsx is a zip, so a quote cannot be audited
#: against its bytes -- SI statements are asserted against the EXTRACTED text at build
#: time instead, which is a stronger check because it refuses the build).
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "26b88d819d429f49cad4ebe85047cfd7354d532d84d584b0f00b222355772642"

SI1_DOCX = "si1.docx"
SI1_DOCX_SHA256 = "7bc748853a07d23e2bc0712a33cfe74c68b7009ad91ff52b1d98e1f875fbc2d3"
SI1_DOCX_BYTES = 5_492_159
SI2_XLSX = "si2.xlsx"
SI2_XLSX_SHA256 = "a3cfd6014cc611b206af9c7e7c8770c23189ae96986d23981da0b11236721612"
SI2_XLSX_BYTES = 2_769_528
SI_RETRIEVED_AT = "2026-10-07T11:44:27.078046+00:00"

#: Sequencing deposits: RECORDED, never downloaded. No loader consumes reads or spectra.
#: Thompson 2020 (reference 30), the construction paper every PT-lineage deletion of
#: this campaign defers to. Mirrored, and the same bytes the de Siqueira 2025 loader
#: quotes for the same lesion.
THOMPSON_KEY = "thompsonFattyAcidAlcohol2020"
THOMPSON_SHA256 = "389d0d6cd196f9f159d0485f6b8ed09743dfbe3b6682369db489d7356a469b45"

SRA_BIOPROJECT = "PRJNA1187681"
GEO_SERIES = "GSE281392"
PRIDE_ACCESSION = "PXD054609"
ALEDB_PROJECT = "Pputida_isoprenol_TALE"

#: Identity of the stressor, pinned and checked against the ``compound_identity_table.json`` row.
ISOPRENOL_LABEL = "isoprenol"
ISOPRENOL_INCHIKEY = "CPJRRXSHAYUTGL-UHFFFAOYSA-N"
ISOPRENOL_SMILES = "C=C(C)CCO"
ISOPRENOL_PUBCHEM_CID = 12988
#: Molar mass of formaldehyde (g/mol), used only to cross-check the HCHO arm's two
#: released dose units against each other.
FORMALDEHYDE_G_PER_MOL = 30.026

KT2440_NAMESPACE: BacterialGeneNamespace = "pputida_kt2440_locus_tag"
KT2440_ASSEMBLY_SET: BacterialAssemblySet = "pputida_KT2440_ASM756v2"
#: The single replicon every released mutation coordinate is given on.
KT2440_REPLICON = "AE015451"

#: ``{strain: its founding F0 clone column}``. An ``F0`` column is the lineage's
#: founding starting strain under the ALEdb convention the paper states, and Table 3's
#: own row groups put ALE 1-4 on KT2440, 5-8 on IPL300 and 9-12 on IPL400, so A1 F0 is
#: the WT, A5 F0 is IPL300 and A9 F0 is IPL400. :func:`assert_founder_columns` proves
#: the assignment on the bytes rather than resting on the convention.
FOUNDER_CLONE_COLUMNS: dict[str, str] = {
    "KT2440": "A1 F0 I1 R1",
    "IPL300": "A5 F0 I1 R1",
    "IPL400": "A9 F0 I1 R1",
}
#: ``{isolate label: its clone column}`` of the two evolved isolates the proteome
#: release quantifies. Both are end-point isolates of IPL400 lineages (ALE 10 and 12).
EVOLVED_ISOLATE_CLONE_COLUMNS: dict[str, str] = {
    "A10_F63_I1": "A10 F63 I1 R1",
    "A12_F53_I1": "A12 F53 I1 R1",
}
#: The starting strain both proteome isolates descend from, stated by the paper.
EVOLVED_ISOLATE_PARENT = "IPL400"

#: Sheets of ``si2.xlsx`` this module reads.
SHEET_MUTATIONS = "Fig 2B_Mutation List"
SHEET_SAMPLE_KEY = "ProteomeXchange Sample Key"
SHEET_PROTEOME_M9G = "Proteome_A10F63I1vsIPL400_M9G"
SHEET_PROTEOME_IPL = "Proteome_A10F63I1vsIPL400_G+4IP"
SHEET_PROTEOME_M9G_ALT = "Proteome_A12F53I1vsIPL400_M9G"
SHEET_PROTEOME_IPL_ALT = "Proteome_A12F53I1vsIPL400_G+4IP"
#: The three production-culture arms, enumerated so the exclusion is counted, not silent.
SHEETS_PIY670 = (
    "IPL400vsA10F63I1_pIY670_M9G_12h",
    "IPL400vsA10F63I1_pIY670_M9G_24h",
    "IPL400vsA10F63I1_pIY670_M9G_48h",
)
#: ``{sheet: (evolved-arm column suffix, IPL400 column suffix)}`` of the two loaded sheets.
PROTEOME_ARMS: dict[str, tuple[str, str]] = {
    SHEET_PROTEOME_M9G: ("A10_F63_I1_M9G", "IPL400_M9G"),
    SHEET_PROTEOME_IPL: ("A10F63I1_M9G+4IP", "IPL400_M9G+4IP"),
    SHEET_PROTEOME_M9G_ALT: ("A12_F53_I1_M9G", "IPL400_M9G"),
    SHEET_PROTEOME_IPL_ALT: ("A12F53I1_M9G+4IP", "IPL400_M9G+4IP"),
}
#: ``{strain: (unstressed sheet, isoprenol sheet, which column of each)}``. Each loaded
#: record is one strain's isoprenol arm referenced to its OWN unstressed arm, so a
#: record needs the same column of two sheets. IPL400 is the ``parent`` column of the
#: A10 pair (its columns are bit-identical across all four sheets, which
#: ``_assert_duplicate_exports`` re-proves); each evolved isolate is the ``test`` column
#: of its own pair.
PROTEOME_RECORD_SHEETS: dict[str, tuple[str, str, str]] = {
    "IPL400": (SHEET_PROTEOME_M9G, SHEET_PROTEOME_IPL, "parent"),
    "A10_F63_I1": (SHEET_PROTEOME_M9G, SHEET_PROTEOME_IPL, "test"),
    "A12_F53_I1": (SHEET_PROTEOME_M9G_ALT, SHEET_PROTEOME_IPL_ALT, "test"),
}
#: The ``Sample name`` the Sample Key sheet files each record's two arms under, so the
#: replicate token is read per arm rather than assumed from one arm.
PROTEOME_SAMPLE_NAMES: dict[str, tuple[str, str]] = {
    "IPL400": ("IPL400_M9G", "IPL400_M9G+4IP"),
    "A10_F63_I1": ("A10_F63_I1_M9G", "A10_F63_I1_M9G+4IP"),
    "A12_F53_I1": ("A12_F53_I1_M9G", "A12_F53_I1_M9G+4IP"),
}
#: What one stored proteome number is, named so heterogeneous proteomics never mixes.
PROTEOME_MEASUREMENT_TYPE = "dia_nn_top3_log2_mean"
PROTEOME_N_REPLICATES = 3
PROTEOME_REPLICATE_TOKEN = "R1,R2,R3"
#: Sample Key rows of the two loaded arms (its ``Sample name`` column, verbatim).
SAMPLE_KEY_M9G = "IPL400_M9G"
SAMPLE_KEY_IPL = "IPL400_M9G+4IP"
#: Welch's t at n = 3 per arm must reproduce to this, which it does to 1.1e-12.
WELCH_TOLERANCE = 1e-6
#: The released log2 fold change must equal the difference of the two means to this.
FOLD_CHANGE_TOLERANCE = 1e-9

LOCUS_TAG_RE = re.compile(r"^PP_\d{4}$")


class CompoundIdentityConflictError(ValueError):
    """The compound-identity table resolves isoprenol to another InChIKey."""


class TableExtractionError(ValueError):
    """The SI docx does not parse into Supplementary Table 3 as expected."""


class SheetExtractionError(ValueError):
    """A ``si2.xlsx`` sheet does not hold the columns or values this loader needs."""


class CrossSourceError(ValueError):
    """Two released statements of one quantity disagree on the pinned bytes."""


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the sha256-pinned Lim 2025 ``paper.md``."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_Q_IPL_GENOTYPES = (
    "IPL300 contains deletions in ΔPP_2675, ΔPP_4064–4067, and ΔPP_3839 while IPL400 "
    "also contains the same ΔPP_2675, ΔPP_4064–4067, and ΔPP_3839 deletions as well as an"
)
_Q_TTGB_LOCUS = (
    "the ttgB homolog (PP_1385) from DOT-T1E was identified using a simple BLASTn query "
    "and deleted in IPL400."
)
_Q_TALE_START = (
    "We performed high-throughput TALE experiments using the WT, IPL300, and IPL400 "
    "starting strains in glucose M9 minimal medium by gradually increasing the isoprenol "
    "concentration from $4 \\ g / \\mathrm { L }$ to $\\scriptstyle 8 \\ g / \\mathrm { L "
    "}$ in steps of $0 . 5 \\mathrm { - } 1 ~ \\mathrm { g / L }$ ."
)
_Q_FIG3A = (
    "(A) Relative growth rates of the reverse-engineered strains grown on "
    "$_ { 4 } \\ g / \\mathrm { L }$ glucose minimal medium supplemented with "
    "$_ 6 \\ g / \\mathrm { L }$ isoprenol."
)
_Q_PP3024_FOLD = (
    "The smaller ΔPP_3024 deletion essentially accounts for most of the tolerance "
    "improvement (1.6-fold) seen in the larger region mutant."
)
_Q_PROTEOME_CONDITIONS = (
    "Proteomic profiles of the evolved isolates grown on glucose $( 4 ~ \\mathrm { g } / "
    "\\mathrm { L } )$ minimal medium and supplemented with "
    "${ } _ { 4 } \\ { } _ { 8 } / \\mathrm { L }$ isoprenol (IPL) were compared to that "
    "of the starting strain IPL400 grown under similar conditions."
)


def _thompson(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to Thompson 2020, the construction paper the deletions defer to."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=THOMPSON_KEY,
            sha256=THOMPSON_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
        ),
    )


SOURCED_VALUES: dict[str, SourcedValue] = {
    "reference_strain": _paper(
        "KT2440",
        _Q_TALE_START,
        note="the three starting strains of the campaign; the WT is P. putida KT2440, "
        "which pins assembly set pputida_KT2440_ASM756v2 (GCA_000007565.2)",
    ),
    "ipl300_genotype": _paper(
        ("PP_2675", "PP_3839", "PP_4064", "PP_4065", "PP_4066", "PP_4067"),
        _Q_IPL_GENOTYPES,
        note="the en dash in 'PP_4064–4067' is the OCR of the range PP_4064 to PP_4067; "
        "Supplementary Table 1 writes the same set as 'KT2440 dPP_2675 d14-PP_2676 "
        "dPP_3839 dPP_4064-dPP_4067'. PP_2676's 14-codon truncation is a separate gap",
    ),
    "ipl400_extra_deletion": _paper(
        "PP_1385",
        _Q_TTGB_LOCUS,
        note="IPL400 is IPL300 plus this one deletion; the paper gives the locus tag "
        "beside the symbol, so no symbol resolution is relied on",
    ),
    "tale_isoprenol_start_g_per_l": _paper(
        4.0,
        _Q_TALE_START,
        note="the dose the Supplementary Table 3 initial growth rates were measured at; "
        "Table 3's own 'Starting concentration' column reads 4 g/L for all 12 isoprenol "
        "lineages and is checked against this at build time",
    ),
    "n_tale_replicates": _paper(
        4,
        "Each TALE experiment was conducted with four independent biological replicates "
        "in one high-throughput evolution campaign",
        note="ALE 1-4 (WT), 5-8 (IPL300) and 9-12 (IPL400) are four measurements of one "
        "(strain, condition), so they aggregate into one record with n_samples = 4",
    ),
    "growth_rate_method": _paper(
        "liquid_od_growth",
        "Maximum growth rates were calculated by plotting slope of ln "
        "$\\scriptstyle { \\mathrm { ( O D } } _ { 6 0 0 } { \\mathrm { ) } }$ vs. time "
        "during exponential growth.",
        note="an OD600 growth curve, which is AssayType.liquid_od_growth",
    ),
    "pp3024_fold": _paper(
        1.6,
        _Q_PP3024_FOLD,
        note="the relative growth rate of KT2440 dPP_3024 to the WT at 6 g/L isoprenol "
        "(Fig. 3A). Read as the mutant's own fold improvement, which the companion "
        "sentence corroborates: the two PP_3024 mutants average 1.7-fold, so 1.6 is below "
        "the pair mean, matching 'most of' rather than all of the improvement. The larger "
        "mutant's implied 1.8-fold is arithmetic on rounded numbers and is NOT stored",
    ),
    "pp3024_pair_mean_fold": _paper(
        1.7,
        "These two mutants displayed improved growth rates by 1.7-fold on average when "
        "grown in the presence of $6 g / \\mathrm { L }$ isoprenol",
        note="the two-mutant average that corroborates the 1.6-fold reading; asserted "
        "at build time to exceed it",
    ),
    "fig3a_isoprenol_g_per_l": _paper(
        6.0,
        _Q_FIG3A,
        note="the dose of the reverse-engineered panel, and its medium: the same 4 g/L "
        "glucose minimal medium as M9_NREL_LIM2025",
    ),
    "seven_deletion_strains": _paper(
        7,
        "A total of seven deletion strains were generated.",
        note="all seven are expressible as BacterialDeletionPerturbations; only "
        "dPP_3024 carries a released number",
    ),
    "proteome_conditions": _paper(
        (4.0, 4.0),
        _Q_PROTEOME_CONDITIONS,
        note="the two loaded proteome arms: 4 g/L glucose minimal medium, with and "
        "without 4 g/L isoprenol. The OCR garbles the second '4 g/L' to "
        "'{ } _ { 4 } \\ { } _ { 8 } / \\mathrm { L }'; the Sample Key sheet writes the "
        "same condition as 'M9 0.4% glucose + 4 g/L Isoprenol'",
    ),
    "proteome_measurement": _paper(
        PROTEOME_MEASUREMENT_TYPE,
        "The Top3 method, which is the average MS signal response of the three most "
        "intense tryptic peptides of each identified protein, was used to plot the "
        "quantity of the targeted proteins in the samples",
        note="the released columns are the log2 of that Top3 signal, per arm",
    ),
    "proteome_n_replicates": _paper(
        PROTEOME_N_REPLICATES,
        "Samples were analyzed as three independent biological replicates.",
        note="corroborates the Sample Key sheet's own 'R1,R2,R3' for both loaded arms, "
        "and the n = 3 that the Welch back-solve independently recovers",
    ),
    "proteome_search_database": _paper(
        "Uniprot P. putida KT2440 proteome",
        "The database used in the DIA-NN search (library-free mode) is the latest Uniprot "
        "P. putida KT2440 proteome FASTA sequence plus the protein sequences of "
        "heterogeneous pathway genes and common proteomic contaminants.",
        note="why the sheets' Locus tag column is already PP_ tags, and why a few keys "
        "are heterologous or contaminant proteins in the whole-campaign deposit",
    ),
    "preexisting_parent_mutations": _paper(
        ("PP_4986", "gacS", "yhjE", "PP_4398"),
        "the starting strains IPL300 and IPL400 also contained pre-existing mutations "
        "missing from the reference sequence: PP_4986, gacS, yhjE in IPL300, and then "
        "IPL400 had the same mutations along with a mutation in PP_4398.",
        note="the parents' own called variants, which the three #731 leaves now hold: "
        "the IPL300 record carries the A5 F0 I1 R1 column's calls and the IPL400 record "
        "the A9 F0 I1 R1 column's. The sentence is also what assert_founder_columns "
        "checks the two founder columns against",
    ),
    "evolved_isolate_parent": _paper(
        EVOLVED_ISOLATE_PARENT,
        "we chose two representative evolved end-point isolates (i.e., A10_F63_I1 and "
        "A12_F53_I1) which were derived from the same starting strain (IPL400) but "
        "contained mutations in different genes (Supplementary Table 5).",
        note="which parent the two proteome isolates descend from, stated by the paper "
        "rather than read off the lineage numbering; Supplementary Table 3's row groups "
        "put ALE 9-12 on IPL400, so ALE 10 and ALE 12 agree with it independently. Their "
        "designed deletions are therefore IPL400's seven, and their own calls come from "
        "the A10 F63 I1 R1 and A12 F53 I1 R1 clone columns",
    ),
    "mutation_total": _paper(
        (158, 73),
        "We identified a total of 158 unique mutations in 73 genetic regions (i.e., genes "
        "or intergenic regions between two genes, Supplementary Dataset 1).",
        note="the released matrix holds 159 rows on 159 distinct AE015451 positions, one "
        "more than the stated 158; the disagreement is reported, not reconciled",
    ),
    "resequenced_isolates": _paper(
        46,
        "whole-genome sequencing of evolved isolates for a total of 46 strains "
        "(including the HCHO tolerized isolates)",
        note="the matrix's 49 clone columns are these 46 plus the three F0 "
        "starting-strain clones (A1 F0 = WT, A5 F0 = IPL300, A9 F0 = IPL400)",
    ),
    "variant_caller": _paper(
        "breseq 0.33.1",
        "Raw files were processed by an in-lab analysis pipeline that utilizes Breseq "
        "(version 0.33.1)",
        note="what produced the per-clone calls; the raw reads are BioProject "
        f"{SRA_BIOPROJECT} and are recorded, never downloaded",
    ),
    "hcho_control": _paper(
        ("A13", "A16"),
        "We also tolerized the WT strain against formaldehyde stress as a control (HCHO, "
        "isolates A13-A16, Supplementary Table 3)",
        note="the HCHO arm's only strain is the WT, which is this dataset's reference, so "
        "the arm yields no record",
    ),
    "isoprenol_generations_range": _paper(
        (143, 353),
        "these P. putida KT2440 strains underwent approximately 143–353 generations, "
        "which is equivalent to 1.33 to $3 . 3 2 \\times 1 0 ^ { 1 2 }$ cumulative cell "
        "divisions (CCD)",
        note="recomputed from Supplementary Table 3's Generations and CCD columns at "
        "build time",
    ),
    "hcho_dose_range_g_per_l": _paper(
        (0.03, 0.27),
        "the concentration of HCHO was increased from $0 . 0 3 { \\gimel } / \\mathrm { L "
        "}$ to $0 . 2 7 ~ { \\ g / \\mathrm { L } }$ over 346–387 generations, "
        "corresponding to $3 . 1 1 \\times 1 0 ^ { 1 2 }$ to $3 . 5 6 \\times 1 0 ^ { 1 2 "
        "}$ CCDs",
        note="the OCR renders the first 'g/L' as '{ \\gimel } / \\mathrm { L }'. Table 3 "
        "states the same doses in mM (1 and 9); the two units are cross-checked at build "
        "time against formaldehyde's 30.026 g/mol",
    ),
    "aledb_naming": _paper(
        ("A", "F", "I"),
        "Each evolved isolate was named according to the ALEdb convention (Phaneuf et al., "
        "2019) that uses the ALE experiment number (A), flask number (F), and isolate "
        "number (I).",
        note="how the matrix's clone columns are keyed, and why an 'F0' column is the "
        "lineage's founding starting strain rather than an evolved isolate",
    ),
    "pp2675_construction": _thompson(
        "complete internal in-frame deletion",
        "Strain with complete internal in-frame deletion of PP_2675",
        note="Table 1 of the construction paper Lim 2025 defers to for the PT lesion; "
        "the same bytes the de Siqueira 2025 loader quotes for the same deletion",
    ),
    "pp3839_construction": _thompson(
        "complete internal in-frame deletion",
        "Strain with complete internal in-frame deletion of PP_3839",
        note="the second Thompson strain IPL300 was built from",
    ),
    "pp4064_construction": _thompson(
        "complete internal in-frame deletion",
        "Strain with complete internal in-frame deletion of the PP_4064-4067 operon",
        note="the four-gene operon deletion IPL300 and IPL400 both carry",
    ),
    "ipl_catabolism_abolished": _paper(
        False,
        "Both IPL300 and IPL400 were unable to grow with isoprenol as the sole carbon "
        "source in M9 media as they contained deletions of ΔPP_2675 and "
        "ΔPP_4064-ΔPP_4067, which were both shown to abolish growth individually on "
        "isoprenol when deleted",
        note="what licenses functional=False on the PP_2675 and PP_4064-PP_4067 "
        "background alleles. The source never states PP_2676's functional status on its "
        "own; its allele rests on the 'd14' designation and on the ORF overlap that "
        "makes the truncation a consequence of the PP_2675 deletion, and the allele's "
        "own note says so",
    ),
    "sra_bioproject": _paper(
        SRA_BIOPROJECT,
        f"Raw-read files were deposited to NCBI SRA with a Bio-Project number of "
        f"{SRA_BIOPROJECT}.",
        note="recorded, not downloaded: no loader consumes reads",
    ),
    "geo_series": _paper(
        GEO_SERIES,
        f"Raw read files were deposited at Gene Expression Omnibus with an accession "
        f"number of {GEO_SERIES}.",
        note="the WT isoprenol-vs-glucose transcriptome; recorded, not downloaded",
    ),
    "pride_accession": _paper(
        PRIDE_ACCESSION,
        "deposited to the ProteomeXchange Consortium via the PRIDE partner repository "
        f"with the dataset identifier {PRIDE_ACCESSION}",
        note="raw DIA spectra; the loaded abundances are the processed Supplementary "
        "Data 1 sheets, not the spectra",
    ),
}

IPL300_DELETIONS: tuple[str, ...] = tuple(SOURCED_VALUES["ipl300_genotype"].value)
IPL400_EXTRA_DELETION: str = SOURCED_VALUES["ipl400_extra_deletion"].value
TALE_DOSE_G_PER_L: float = float(SOURCED_VALUES["tale_isoprenol_start_g_per_l"].value)
PANEL_DOSE_G_PER_L: float = float(SOURCED_VALUES["fig3a_isoprenol_g_per_l"].value)
N_TALE_REPLICATES: int = int(SOURCED_VALUES["n_tale_replicates"].value)
PP3024_FOLD: float = float(SOURCED_VALUES["pp3024_fold"].value)
PP3024_PAIR_MEAN_FOLD: float = float(SOURCED_VALUES["pp3024_pair_mean_fold"].value)
_GENERATIONS = SOURCED_VALUES["isoprenol_generations_range"].value
ISOPRENOL_GENERATIONS_RANGE: tuple[int, int] = (
    int(_GENERATIONS[0]),
    int(_GENERATIONS[1]),
)
_HCHO_DOSES = SOURCED_VALUES["hcho_dose_range_g_per_l"].value
HCHO_DOSE_RANGE_G_PER_L: tuple[float, float] = (
    float(_HCHO_DOSES[0]),
    float(_HCHO_DOSES[1]),
)
#: Stated CCD ranges (x1e12) of the two arms, from the same two Results sentences the
#: generation ranges come from.
ISOPRENOL_CCD_RANGE = (1.33, 3.32)
HCHO_GENERATIONS_RANGE = (346, 387)
HCHO_CCD_RANGE = (3.11, 3.56)
HCHO_DOSE_RANGE_MM = (1.0, 9.0)

#: Checklist item 4: the loaded genotypes name 8 loci, every one of them written as a
#: PP_ tag by the paper itself, so anything below a complete resolution means the wrong
#: annotation rather than a few hard names.
MIN_RESOLVED_FRACTION = 1.0

_PAPER_LOOKED_IN = Provenance(
    source_uri=PAPER_MD,
    citation_key=CITATION_KEY,
    sha256=PAPER_MD_SHA256,
    method="full Materials and Methods, Results and figure captions read",
)
_SI1_LOOKED_IN = Provenance(
    source_uri=f"{RAW_DIR_REL}/data/{SI1_DOCX}",
    citation_key=CITATION_KEY,
    sha256=SI1_DOCX_SHA256,
    method="every Supplementary Table (1-7), Note 1 and Figure caption read from the "
    "WordprocessingML body (stdlib zipfile + ElementTree walk)",
)

TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_LOOKED_IN,
    note="no incubation temperature is stated for the TALE flasks, the tolerance assay "
    "or the proteome cultures; 30 C is stated only for the LB pre-culture of the "
    "pIY670 production runs",
)
TALE_DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_LOOKED_IN,
    note="the TALE passaged 'once or twice a day', so no fixed per-flask duration exists; "
    "the stored value is a mean over a lineage's three first flasks",
)
PANEL_DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_LOOKED_IN,
    note="Fig. 3A states the medium and the isoprenol dose but no growth duration; the "
    "readout is a maximum specific growth rate, not an end-point",
)
PROTEOME_DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_LOOKED_IN,
    note="'Cultures at the exponential growth phase were harvested by centrifugation'; "
    "no hours are stated for the M9G and M9G+4IP proteome arms",
)
RESPONSE_UNCERTAINTY_GAPS: tuple[ProvenanceGap, ...] = tuple(
    ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_SI1_LOOKED_IN,
        note="Supplementary Table 3 reports no dispersion for any lineage, and the stored "
        "value divides by the WT arm's mean, so a sample SD over the four numerators "
        "would propagate one arm only. The per-lineage numbers are in "
        "preprocess/table_s3.csv",
    )
    for field in ("environment_response_uncertainty", "environment_response_se")
)
PANEL_UNCERTAINTY_GAPS: tuple[ProvenanceGap, ...] = tuple(
    ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_PAPER_LOOKED_IN,
        note="Fig. 3A plots mean +/- SD but releases no numbers; the 1.6-fold is the "
        "only value its text states, and it carries no dispersion",
    )
    for field in ("environment_response_uncertainty", "environment_response_se")
)
PANEL_N_SAMPLES_GAP = ProvenanceGap(
    field="n_samples",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_LOOKED_IN,
    note="neither the Fig. 3A caption nor the Methods states a replicate count for the "
    "reverse-engineered tolerance panel",
)

#: What the paper releases that no loader here consumes, recorded in the raw manifest.
NOT_MIRRORED: tuple[str, ...] = (
    f"SRA BioProject {SRA_BIOPROJECT} (genome resequencing of the 46 evolved isolates "
    "and the three starting strains): RECORDED, not downloaded. The per-clone calls this "
    f"loader reads about are the processed matrix in {SI2_XLSX}; the campaign's own "
    f"mutation analysis is also public as ALEdb project {ALEDB_PROJECT}",
    f"GEO {GEO_SERIES} (WT transcriptome on glucose vs isoprenol as sole carbon source): "
    "RECORDED, not downloaded. No record of this loader is an expression record",
    f"PRIDE {PRIDE_ACCESSION} (raw DIA mass spectra of every proteome sample): RECORDED, "
    "not downloaded. The loaded abundances are the processed sheets of "
    f"{SI2_XLSX}; no loader consumes raw spectra",
    "paper.pdf: not consumed. Every quote is anchored in the literature mirror's "
    "paper.md, and every number comes from one of the two deposited supplements",
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
class RawFile(BaseModel):
    """One file the loader consumes: its pinned bytes and how they were retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str
    bytes: int
    description: str
    retrieval: RetrievalRecord

    @property
    def mirror_relpath(self) -> str:
        """Path inside the raw mirror (``data/<name>``)."""
        return f"data/{self.name}"


def _elsevier(filename: str, sha256: str) -> RetrievalRecord:
    """The recorded, re-runnable retrieval of one Elsevier multimedia component."""
    url = f"https://ars.els-cdn.com/content/image/1-s2.0-{PAPER_PII}-{filename}"
    return RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=url,
        retriever="torchcell.literature.retrieve.elsevier_mmc",
        params={"pii": PAPER_PII, "filename": filename},
        sha256=sha256,
        retrieved_at=SI_RETRIEVED_AT,
    )


RAW_FILES: tuple[RawFile, ...] = (
    RawFile(
        name=SI1_DOCX,
        sha256=SI1_DOCX_SHA256,
        bytes=SI1_DOCX_BYTES,
        description="Supplementary Information (mmc1.docx): Supplementary Note 1, "
        "Figures S1-S3 and Tables 1-7. Supplementary Table 3 is the consumed table -- "
        "per-lineage starting and ending stressor concentration, initial and final "
        "growth rate, passages, generations and cumulative cell divisions for all 16 "
        "TALE lineages",
        retrieval=_elsevier("mmc1.docx", SI1_DOCX_SHA256),
    ),
    RawFile(
        name=SI2_XLSX,
        sha256=SI2_XLSX_SHA256,
        bytes=SI2_XLSX_BYTES,
        description="Supplementary Data 1 (mmc2.xlsx): the 159-row per-clone mutation "
        "matrix (Fig 2B_Mutation List), the Table S4 gene lists, the ProteomeXchange "
        "sample key, and seven per-protein proteome comparison sheets. The two IPL400 "
        "arms of Proteome_A10F63I1vsIPL400_M9G and ..._G+4IP are the consumed columns",
        retrieval=_elsevier("mmc2.xlsx", SI2_XLSX_SHA256),
    ),
)
DATA_SHA256: dict[str, str] = {f.name: f.sha256 for f in RAW_FILES}
RAW_FILES_BY_NAME: dict[str, RawFile] = {f.name: f for f in RAW_FILES}


def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build trees live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/limEvolutionguidedToleranceEngineering2025``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve_raw_files(dest_dir: str | Path) -> dict[str, Path]:
    """Run each file's recorded retriever and write the verified bytes to ``dest_dir``.

    The recorded ``RetrievalRecord`` is what runs (``run_retriever``), so this IS the
    re-runnable retrieval; a byte mismatch raises before anything is written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for raw in RAW_FILES:
        path = dest / raw.name
        write_verified(
            run_retriever(raw.retrieval),
            path,
            raw.sha256,
            raw.retrieval.source_url or raw.name,
        )
        out[raw.name] = path
    return out


def deposit_raw_mirror(
    *, sources: Mapping[str, str | Path], data_root: str | None = None
) -> Path:
    """Write the raw mirror from already-retrieved files plus its ``manifest.json``.

    ``sources`` maps every name in :data:`RAW_FILES` to a local file. Idempotent by
    sha256: a mirror file with the pinned hash is left alone, and one with any other
    hash raises rather than being overwritten. Every source is verified BEFORE anything
    is written, so a refusal leaves no partial deposit.
    """
    missing = sorted(set(DATA_SHA256) - set(sources))
    if missing:
        raise KeyError(f"no source given for {missing}")
    for raw in RAW_FILES:
        got = _sha256(sources[raw.name])
        if got != raw.sha256:
            raise RuntimeError(
                f"{sources[raw.name]} sha256 mismatch: got {got}, expected {raw.sha256}"
            )
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for raw in RAW_FILES:
        dest = root / raw.mirror_relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != raw.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(sources[raw.name], dest)
        records.append(
            ArtifactRecord(
                path=raw.mirror_relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=raw.sha256,
                source=raw.retrieval.source_url,
                retrieval=raw.retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=records,
        si_data_sources=[
            raw.retrieval.source_url for raw in RAW_FILES if raw.retrieval.source_url
        ]
        + [
            f"https://www.ncbi.nlm.nih.gov/bioproject/{SRA_BIOPROJECT}",
            f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GEO_SERIES}",
            f"https://www.ebi.ac.uk/pride/archive/projects/{PRIDE_ACCESSION}",
        ],
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


def _link_mirror_files(raw_dir: str, data_root: str) -> None:
    """Link every consumed mirror file into ``raw/`` after checking manifest + sha256."""
    manifest = load_manifest(data_root)
    os.makedirs(raw_dir, exist_ok=True)
    for raw in RAW_FILES:
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
# Supplementary Table 3 (si1.docx)
# --------------------------------------------------------------------------- #
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
#: The header cells that identify Supplementary Table 3 among the SI's tables.
_S3_HEADER = (
    "ALE#",
    "Starting concentration",
    "Initial growth rate (h-1)",
    "Ending concentration",
    "Final growth rateb (h-1)",
    "Passages #",
    "Generations",
    "CCD (1012)",
)
#: ``{group label: (starting strain, stressor)}`` of Table 3's four row groups.
_S3_GROUPS: dict[str, tuple[str, str]] = {
    "Isoprenol TALE KT2440": ("KT2440", "isoprenol"),
    "Isoprenol TALE IPL300": ("IPL300", "isoprenol"),
    "Isoprenol TALE IPL400": ("IPL400", "isoprenol"),
    "HCHO TALE KT2440": ("KT2440", "formaldehyde"),
}
#: Total lineages, and lineages per group: four biological replicates each.
_S3_N_ROWS = 16
#: ``8*`` carries the footnote about the restarted lineage; the ALE number is the digits.
_S3_ALE_RE = re.compile(r"^(?P<number>\d+)\*?$")
_S3_DOSE_RE = re.compile(r"^(?P<value>[\d.]+)\s*(?P<unit>g/L|mM)$")


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    """Concatenated run text of one WordprocessingML paragraph."""
    return "".join(node.text or "" for node in paragraph.iter(_W + "t"))


def docx_body(path: str | Path) -> ElementTree.Element:
    """The ``w:body`` element of a .docx, read with the standard library only."""
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    body = root.find(_W + "body")
    if body is None:
        raise TableExtractionError(f"{path} has no WordprocessingML body")
    return body


def docx_text(path: str | Path) -> str:
    """Every paragraph and every table cell of a .docx as newline-joined text.

    This is what SI statements are asserted against at build time: a docx is a zip, so
    a quote cannot be audited against its bytes, and refusing the build when a quote has
    moved is stronger than a post-hoc audit.

    A table cell is emitted TWICE over: once as its own paragraphs (which is how the
    body walk sees them) and once as those paragraphs joined by a space, because Word
    splits a single visual cell such as ``KT2440 dPP_3024 (JBEI-235868)`` across two
    paragraphs and only the joined form carries the whole statement.
    """
    body = docx_body(path)
    lines: list[str] = []
    for paragraph in body.iter(_W + "p"):
        text = _paragraph_text(paragraph)
        if text.strip():
            lines.append(text)
    for cell in body.iter(_W + "tc"):
        joined = " ".join(
            _paragraph_text(paragraph) for paragraph in cell.findall(_W + "p")
        ).strip()
        if joined:
            lines.append(joined)
    return "\n".join(lines)


def docx_tables(path: str | Path) -> list[list[list[str]]]:
    """Every table of a .docx as rows of cell strings, in document order."""
    tables: list[list[list[str]]] = []
    for table in docx_body(path).iter(_W + "tbl"):
        rows: list[list[str]] = []
        for row in table.findall(_W + "tr"):
            rows.append(
                [
                    " ".join(_paragraph_text(p) for p in cell.findall(_W + "p")).strip()
                    for cell in row.findall(_W + "tc")
                ]
            )
        tables.append(rows)
    return tables


class TaleLineage(BaseModel):
    """One Supplementary Table 3 row: an ALE lineage and everything it releases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ale_number: int
    ale_label: str
    starting_strain: str
    stressor: str
    starting_dose: float
    starting_dose_unit: str
    initial_growth_rate: float
    ending_dose: float
    ending_dose_unit: str
    final_growth_rate: float
    passages: int
    generations: int
    ccd_1e12: float


def _dose(cell: str, *, field: str, ale: str) -> tuple[float, str]:
    """A ``4 g/L`` / ``1 mM`` dose cell as ``(value, unit)``."""
    match = _S3_DOSE_RE.match(cell.strip())
    if match is None:
        raise TableExtractionError(
            f"Supplementary Table 3, ALE {ale}: {field} {cell!r} is not a dose"
        )
    return float(match.group("value")), match.group("unit")


def parse_table_s3(tables: Sequence[Sequence[Sequence[str]]]) -> list[TaleLineage]:
    """Supplementary Table 3's 16 lineages, refusing any row that does not parse.

    The table is located by its header cells, not by position. The group label and the
    starting concentration are stated once per group and forward-filled, which is how
    the source writes them.
    """
    matches = [
        rows
        for rows in tables
        if rows and tuple(cell for cell in rows[0][1:]) == _S3_HEADER
    ]
    if len(matches) != 1:
        raise TableExtractionError(
            f"{len(matches)} SI tables carry Supplementary Table 3's header (expected 1)"
        )
    body = matches[0][1:]
    lineages: list[TaleLineage] = []
    group = ""
    dose: tuple[float, str] | None = None
    for row in body:
        if len(row) != len(_S3_HEADER) + 1:
            raise TableExtractionError(
                f"Supplementary Table 3 row has {len(row)} cells, expected "
                f"{len(_S3_HEADER) + 1}: {row!r}"
            )
        label, ale, start, initial, ending, final, passages, generations, ccd = row
        if label.strip():
            group = label.strip()
            dose = None
        if group not in _S3_GROUPS:
            raise TableExtractionError(
                f"Supplementary Table 3 row group {group!r} is not one of "
                f"{sorted(_S3_GROUPS)}"
            )
        number = _S3_ALE_RE.match(ale.strip())
        if number is None:
            raise TableExtractionError(
                f"Supplementary Table 3 ALE cell {ale!r} is not an ALE number"
            )
        if start.strip():
            dose = _dose(start, field="starting concentration", ale=ale)
        if dose is None:
            raise TableExtractionError(
                f"Supplementary Table 3 ALE {ale!r} states no starting concentration and "
                "none was forward-filled"
            )
        strain, stressor = _S3_GROUPS[group]
        end_value, end_unit = _dose(ending, field="ending concentration", ale=ale)
        lineages.append(
            TaleLineage(
                ale_number=int(number.group("number")),
                ale_label=ale.strip(),
                starting_strain=strain,
                stressor=stressor,
                starting_dose=dose[0],
                starting_dose_unit=dose[1],
                initial_growth_rate=float(initial),
                ending_dose=end_value,
                ending_dose_unit=end_unit,
                final_growth_rate=float(final),
                passages=int(passages),
                generations=int(generations),
                ccd_1e12=float(ccd),
            )
        )
    if len(lineages) != _S3_N_ROWS:
        raise TableExtractionError(
            f"Supplementary Table 3 parsed {len(lineages)} lineages, expected "
            f"{_S3_N_ROWS}"
        )
    numbers = [lineage.ale_number for lineage in lineages]
    if numbers != list(range(1, _S3_N_ROWS + 1)):
        raise TableExtractionError(f"Supplementary Table 3 ALE numbers are {numbers}")
    return lineages


def table_s3_digest(lineages: Sequence[TaleLineage]) -> str:
    """sha256 of the parsed lineages, independent of the document's row order."""
    payload = json.dumps(
        sorted((row.model_dump() for row in lineages), key=lambda r: r["ale_number"]),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: The parsed Supplementary Table 3 of :data:`SI1_DOCX_SHA256`. A changed digest means
#: the extraction or the source moved, and the build refuses rather than storing it.
TABLE_S3_SHA256 = "07e309b919552ee925bdc1c7d57b0e8fa0085da247b1fb425014db1b75a6c12c"

#: SI statements this loader relies on that live only in ``si1.docx``. A docx is a zip,
#: so ``audit_sourced_value`` cannot find a quote in its bytes; each of these is asserted
#: present in the EXTRACTED text at build time instead, which refuses the build on drift.
SI1_STATEMENTS: dict[str, str] = {
    "initial_growth_rate_definition": (
        "Average growth rate, observed in the three first flasks of each experiment."
    ),
    "final_growth_rate_definition": (
        "Average growth rate, observed in the three last flasks of each experiment."
    ),
    "ale8_restart": (
        "Has a lower passage than other lineages since the experiment was restarted from "
        "an early intermediate stock due to cross-contamination."
    ),
    "ipl300_genotype": "KT2440 ΔPP_2675 Δ14-PP_2676 ΔPP_3839 ΔPP_4064-∆PP_4067",
    "ipl400_genotype": (
        "KT2440 ΔPP_2675 Δ14-PP_2676 ΔPP_3839 ΔPP_4064-∆PP_4067 ΔttgB (PP_1385)"
    ),
    "pp2676_truncation": (
        "KT2440 with the complete internal in-frame deletion of PP_2675 and partial "
        "truncation of the first fourteen amino acids of PP_2676"
    ),
    "pp3024_strain": "KT2440 ΔPP_3024",
    "pp3024_accession": "KT2440 ΔPP_3024 (JBEI-235868)",
}
#: Accession of the one reverse-engineered strain that carries a released number.
PP3024_STRAIN_ACCESSION = "JBEI-235868"
PP3024_LOCUS = "PP_3024"
#: The locus PP_2676's 14-codon N-terminal truncation sits on. Present, not absent, and
#: with no stated functional consequence, so no leaf states it.
PP2676_LOCUS = "PP_2676"


# --------------------------------------------------------------------------- #
# Supplementary Data 1 (si2.xlsx)
# --------------------------------------------------------------------------- #
class ProteomeRow(BaseModel):
    """One proteome sheet row: a locus, its symbol, and both arms' log2 statistics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    locus_tag: str
    locus_tag_as_written: str
    symbol: str
    accession: str
    description: str
    test_mean: float
    test_sd: float
    parent_mean: float
    parent_sd: float
    t_statistic: float
    log2_fold_change: float


def _sheet_rows(path: str, sheet: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    """``(header, rows)`` of one ``si2.xlsx`` sheet, read-only, fully blank rows dropped.

    The filter is deliberately on the WHOLE row, not on its first cells: the mutation
    sheet states a region label once per region and leaves it empty on the region's
    other rows, so a leading-cell filter silently halves that table.
    """
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if sheet not in book.sheetnames:
        raise SheetExtractionError(f"{path} has no sheet {sheet!r}")
    worksheet = book[sheet]
    stream = worksheet.iter_rows(values_only=True)
    header = ["" if cell is None else str(cell) for cell in next(stream)]
    rows = [
        row
        for row in stream
        if any(cell is not None and str(cell).strip() != "" for cell in row)
    ]
    return header, rows


def read_proteome_sheet(path: str, sheet: str) -> list[ProteomeRow]:
    """One proteome comparison sheet, with its locus-tag column normalized to upper case.

    The ``G+4IP`` sheets title-case a handful of tags (``Pp_0002``); upper-casing the
    whole column is the one normalization applied, and the verbatim spelling is kept on
    the row so the rule is auditable.
    """
    test_suffix, parent_suffix = PROTEOME_ARMS[sheet]
    header, rows = _sheet_rows(path, sheet)
    index = {name: position for position, name in enumerate(header)}
    tag_column = "Locus tag" if "Locus tag" in index else "Locus Tag"
    required = [
        "Protein",
        tag_column,
        "Protein.Group",
        "Protein.Description",
        f"log2_mean_{test_suffix}",
        f"log2_mean_{parent_suffix}",
        f"log2_std_{test_suffix}",
        f"log2_std_{parent_suffix}",
        "t-test_stat",
        "log2_Fold_change_A/B",
    ]
    absent = [name for name in required if name not in index]
    if absent:
        raise SheetExtractionError(f"{sheet} is missing columns {absent}")
    out: list[ProteomeRow] = []
    for row in rows:
        if row[index["Protein"]] is None:
            continue
        written = str(row[index[tag_column]])
        tag = written.upper()
        if not LOCUS_TAG_RE.match(tag):
            raise SheetExtractionError(
                f"{sheet}: {written!r} is not a KT2440 locus tag"
            )
        out.append(
            ProteomeRow(
                locus_tag=tag,
                locus_tag_as_written=written,
                symbol=str(row[index["Protein"]]),
                accession=str(row[index["Protein.Group"]]),
                description=str(row[index["Protein.Description"]]),
                test_mean=float(row[index[f"log2_mean_{test_suffix}"]]),
                test_sd=float(row[index[f"log2_std_{test_suffix}"]]),
                parent_mean=float(row[index[f"log2_mean_{parent_suffix}"]]),
                parent_sd=float(row[index[f"log2_std_{parent_suffix}"]]),
                t_statistic=float(row[index["t-test_stat"]]),
                log2_fold_change=float(row[index["log2_Fold_change_A/B"]]),
            )
        )
    tags = [item.locus_tag for item in out]
    if len(set(tags)) != len(tags):
        raise SheetExtractionError(f"{sheet} files one locus tag twice")
    return out


def shared_symbol_loci(rows: Sequence[ProteomeRow]) -> list[str]:
    """Locus tags whose gene SYMBOL the sheet files under more than one locus.

    Measured on the pinned workbook: ``Ubid`` (PP_0548, PP_5213), ``Pyrc`` (PP_1086,
    PP_4999) and ``Dapa`` (PP_1237, PP_2639) carry identical means and SDs under two
    distinct UniProt accessions, and those six rows are exactly the ones whose released
    t statistic does not reproduce at n = 3. One protein group's statistics cannot be
    attributed to either paralog, so both rows of each symbol go.
    """
    counts = Counter(item.symbol for item in rows)
    shared = {symbol for symbol, n in counts.items() if n > 1}
    return sorted(item.locus_tag for item in rows if item.symbol in shared)


def welch_t(row: ProteomeRow, n: int) -> float:
    """Welch's t of the sheet's two arms, taking each SD as a SAMPLE SD over ``n``."""
    return (row.test_mean - row.parent_mean) / math.sqrt(
        (row.test_sd**2 + row.parent_sd**2) / n
    )


def assert_sheet_statistics(rows: Sequence[ProteomeRow], *, sheet: str) -> float:
    """Back-solve the replicate count and uncertainty type; return the worst residual.

    This is the measurement that makes ``n_replicates = 3`` and
    ``se = sd / sqrt(3)`` sourced rather than assumed: reading each ``log2_std`` as the
    SAMPLE SD of three replicates reproduces the released ``t-test_stat`` for every row,
    and the released ``log2_Fold_change_A/B`` equals the difference of the two means.
    """
    worst = 0.0
    for row in rows:
        if abs((row.test_mean - row.parent_mean) - row.log2_fold_change) > (
            FOLD_CHANGE_TOLERANCE
        ):
            raise CrossSourceError(
                f"{sheet}/{row.locus_tag}: released log2 fold change "
                f"{row.log2_fold_change} is not the difference of the two means"
            )
        denominator = math.sqrt(
            (row.test_sd**2 + row.parent_sd**2) / PROTEOME_N_REPLICATES
        )
        if denominator == 0.0:
            continue
        worst = max(worst, abs(welch_t(row, PROTEOME_N_REPLICATES) - row.t_statistic))
    if worst > WELCH_TOLERANCE:
        raise CrossSourceError(
            f"{sheet}: Welch's t at n = {PROTEOME_N_REPLICATES} with the reported std as "
            f"a sample SD misses the released t by up to {worst:.3g}; the replicate count "
            "or the uncertainty type is not what this loader stores"
        )
    return worst


def parent_arm(
    rows: Sequence[ProteomeRow], dropped: Iterable[str]
) -> dict[str, tuple[float, float]]:
    """``{locus tag: (log2 mean, log2 sample SD)}`` of the IPL400 arm of one sheet."""
    skip = set(dropped)
    return {
        row.locus_tag: (row.parent_mean, row.parent_sd)
        for row in rows
        if row.locus_tag not in skip
    }


def test_arm(
    rows: Sequence[ProteomeRow], dropped: Iterable[str]
) -> dict[str, tuple[float, float]]:
    """``{locus tag: (log2 mean, log2 sample SD)}`` of the EVOLVED arm of one sheet.

    The mirror of :func:`parent_arm`. Both columns are the arm's own absolute log2 Top3
    mean, not a ratio, which is what lets an evolved isolate's two arms be compared to
    each other exactly as IPL400's two are.
    """
    skip = set(dropped)
    return {
        row.locus_tag: (row.test_mean, row.test_sd)
        for row in rows
        if row.locus_tag not in skip
    }


def read_sample_key(path: str) -> dict[str, str]:
    """``{sample name: replicate token}`` of the ProteomeXchange Sample Key sheet."""
    _, rows = _sheet_rows(path, SHEET_SAMPLE_KEY)
    out: dict[str, str] = {}
    for row in rows:
        cells = ["" if cell is None else str(cell).strip() for cell in row]
        if len(cells) < 5 or cells[1] in {"", "Sample name"}:
            continue
        out[cells[1]] = cells[4]
    if not out:
        raise SheetExtractionError(f"{SHEET_SAMPLE_KEY} parsed no samples")
    return out


class MutationMatrix(BaseModel):
    """What the released per-clone variant matrix holds, as counts only.

    Counts only, written to ``preprocess/variant_accounting.json``: the matrix's shape
    independent of which clone columns a build loads, so a changed release shows up as a
    changed count rather than as silently different records. The calls themselves become
    perturbations through :func:`called_variant_perturbations`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sheet: str
    replicon: str
    n_rows: int
    n_distinct_positions: int
    n_clone_columns: int
    n_founder_columns: int
    n_evolved_columns: int
    n_calls: int
    call_frequencies: dict[str, int]
    mutation_types: dict[str, int]
    n_intergenic: int
    n_multi_locus: int = Field(
        description="rows naming more than one locus that are NOT intergenic (the large "
        "deletions); the intergenic rows also name two loci and are counted separately, "
        "so the two shapes do not overlap"
    )
    n_single_locus: int
    n_region_labels: int
    founder_call_counts: dict[str, int]


#: A clone column is ``A<ale> F<flask> I<isolate> R<replicate>``; ``F0`` is the lineage's
#: founding starting strain, not an evolved isolate (the ALEdb convention).
_CLONE_RE = re.compile(
    r"^A(?P<ale>\d+) F(?P<flask>\d+) I(?P<isolate>\d+) R(?P<rep>\d+)$"
)


#: ``{released Mutation Type cell: the schema's variant kind}``. The four cells are the
#: whole released vocabulary (measured on the pinned sheet: 100 SNP, 47 DEL, 11 INS,
#: 1 SUB over the 159 rows); an unknown cell raises a ``KeyError`` rather than being
#: read as anything.
VARIANT_TYPE_BY_STATEMENT: dict[str, BacterialVariantType] = {
    "SNP": BacterialVariantType.snv,
    "DEL": BacterialVariantType.deletion,
    "INS": BacterialVariantType.insertion,
    "SUB": BacterialVariantType.substitution,
}

#: The matrix cell is a within-clone FRACTION, not a percent: measured over its 443
#: calls, 431 cells read ``1`` and 12 read ``0.9``, so no cell exceeds 1.
CLONE_FREQUENCY_BASIS = VariantFrequencyBasis.fraction

VariantRepresentation = Literal[
    "bacterial_sequence_variant", "bacterial_site_variant", "bacterial_span_deletion"
]
"""Which #731 perturbation leaf a released row's SHAPE maps to.

The three values are the leaves' own ``perturbation_type`` literals, so the recorded
mapping cannot drift from the class that is written.
"""


def released_row_representation(
    *, is_intergenic: bool, mutation_type: str, detail: str
) -> VariantRepresentation:
    """The leaf one released row maps to, decided by the row's own cells.

    Three shapes, measured on the pinned sheet and partitioning all 159 rows:

    - ``Details`` starts ``intergenic`` (17 rows): the call sits between loci and the
      ``Gene`` cell names the two flanking ones, so it is keyed on its genomic SITE.
    - ``Mutation Type == DEL`` with an EMPTY ``Details`` (25 rows): breseq reports no
      coding offset, which is how it says the named loci lie wholly inside the deleted
      interval. 19 of the 25 name more than one locus (up to the 53 of
      ``PP_3024``-``PP_5558``) and 6 name one; both are the same shape, one span
      deletion event covering whole loci.
    - everything else (117 rows): one call inside one named locus. Measured: every row
      left after the first two tests names exactly one locus, which
      :func:`called_variant_perturbations` asserts rather than assumes.
    """
    if is_intergenic:
        return "bacterial_site_variant"
    kind = VARIANT_TYPE_BY_STATEMENT[mutation_type]
    if kind is BacterialVariantType.deletion and not detail.strip():
        return "bacterial_span_deletion"
    return "bacterial_sequence_variant"


class CalledVariant(BaseModel):
    """One released breseq call, verbatim, with the leaf its shape maps to.

    Every field is the released cell as the sheet writes it; nothing is normalized (the
    ``Details`` offsets use U+2011 non-breaking hyphens and keep them). ``representation``
    is the #731 leaf :func:`called_variant_perturbations` writes the call as, so
    ``preprocess/called_variants.json`` records the row-shape-to-class mapping on the
    real rows rather than in prose.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    region: str | None
    common_region: str | None
    replicon: str
    position: int
    mutation_type: str
    sequence_change: str
    gene_field: str
    loci: list[str]
    detail: str
    is_intergenic: bool
    clones: dict[str, str]
    representation: VariantRepresentation
    locus_seen_twice_in_a_clone: bool = Field(
        description="another call of the same clone sits in one of this row's loci "
        "(measured: 8 rows). The perturbation axis admits both, because "
        "Genotype.perturbations has no one-entry-per-locus rule; the strain-BACKGROUND "
        "axis does not, which is why a called variant is never a "
        "BacterialBackgroundAllele"
    )


def read_variant_calls(path: str) -> list[CalledVariant]:
    """Every released call as a typed row, with the #731 leaf its shape maps to."""
    header, rows = _sheet_rows(path, SHEET_MUTATIONS)
    clones = [name for name in header[9:] if name.strip()]
    body = [row for row in rows if str(row[3] or "").strip()]
    per_clone_locus: Counter[tuple[str, str]] = Counter()
    parsed: list[dict[str, Any]] = []
    for row in body:
        gene_field = str(row[6] or "")
        loci = [token.strip() for token in gene_field.split(",") if token.strip()]
        detail = str(row[8] or "")
        calls = {
            clone: str(row[9 + offset])
            for offset, clone in enumerate(clones)
            if row[9 + offset] is not None and str(row[9 + offset]).strip()
        }
        for clone in calls:
            for locus in loci:
                per_clone_locus[(clone, locus)] += 1
        parsed.append(
            {
                "region": str(row[0]).strip() or None,
                "common_region": str(row[1] or "").strip() or None,
                "replicon": str(row[2]),
                "position": int(row[3]),
                "mutation_type": str(row[4]),
                "sequence_change": str(row[5] or ""),
                "gene_field": gene_field,
                "loci": loci,
                "detail": detail,
                "is_intergenic": detail.startswith("intergenic"),
                "clones": calls,
            }
        )
    out: list[CalledVariant] = []
    for item in parsed:
        out.append(
            CalledVariant(
                representation=released_row_representation(
                    is_intergenic=item["is_intergenic"],
                    mutation_type=item["mutation_type"],
                    detail=item["detail"],
                ),
                locus_seen_twice_in_a_clone=any(
                    per_clone_locus[(clone, locus)] > 1
                    for clone in item["clones"]
                    for locus in item["loci"]
                ),
                **item,
            )
        )
    return out


def read_mutation_matrix(path: str) -> MutationMatrix:
    """Count every dimension of the released per-clone mutation matrix."""
    header, rows = _sheet_rows(path, SHEET_MUTATIONS)
    clones = [name for name in header[9:] if name.strip()]
    bad = [name for name in clones if _CLONE_RE.match(name) is None]
    if bad:
        raise SheetExtractionError(f"{SHEET_MUTATIONS}: unparsed clone columns {bad}")
    founders = [
        name
        for name in clones
        if int(_CLONE_RE.match(name).group("flask")) == 0  # type: ignore[union-attr]
    ]
    body = [row for row in rows if str(row[3] or "").strip()]
    replicons = {str(row[2]) for row in body}
    if replicons != {KT2440_REPLICON}:
        raise SheetExtractionError(
            f"{SHEET_MUTATIONS}: coordinates are on {sorted(replicons)}, not "
            f"{KT2440_REPLICON}"
        )
    frequencies: Counter[str] = Counter()
    founder_calls: Counter[str] = Counter({clone: 0 for clone in founders})
    for row in body:
        for offset, clone in enumerate(clones):
            cell = row[9 + offset]
            if cell is None or str(cell).strip() == "":
                continue
            frequencies[str(cell)] += 1
            if clone in founders:
                founder_calls[clone] += 1
    details = [str(row[8] or "") for row in body]
    genes = [str(row[6] or "") for row in body]
    return MutationMatrix(
        sheet=SHEET_MUTATIONS,
        replicon=KT2440_REPLICON,
        n_rows=len(body),
        n_distinct_positions=len({int(row[3]) for row in body}),
        n_clone_columns=len(clones),
        n_founder_columns=len(founders),
        n_evolved_columns=len(clones) - len(founders),
        n_calls=sum(frequencies.values()),
        call_frequencies=dict(frequencies),
        mutation_types=dict(Counter(str(row[4]) for row in body)),
        n_intergenic=sum(detail.startswith("intergenic") for detail in details),
        n_multi_locus=sum(
            "," in gene and not detail.startswith("intergenic")
            for gene, detail in zip(genes, details, strict=True)
        ),
        n_single_locus=sum("," not in gene for gene in genes),
        n_region_labels=len(
            {str(row[0]).strip() for row in body if str(row[0]).strip()}
        ),
        founder_call_counts=dict(founder_calls),
    )


# --------------------------------------------------------------------------- #
# The called variants as perturbations (issue #731)
# --------------------------------------------------------------------------- #
class FounderCheck(BaseModel):
    """The measured founder-column assignment, proved on the released matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    wt_column: str
    wt_n_calls: int
    ipl300_column: str
    ipl300_positions: tuple[int, ...]
    ipl400_column: str
    ipl400_positions: tuple[int, ...]
    ipl400_only_positions: tuple[int, ...]
    ipl400_only_genes: tuple[str, ...]


def assert_founder_columns(calls: Sequence[CalledVariant]) -> FounderCheck:
    """Prove which F0 column is which starting strain, and refuse a disagreement.

    Three independent facts have to hold together, and all three are measured on the
    pinned sheet: the matrix is called against KT2440 WT, so the WT founder column
    carries ZERO calls; IPL400 is IPL300 plus one deletion and one further pre-existing
    mutation, so the IPL400 founder's call positions must be a strict SUPERSET of the
    IPL300 founder's; and what IPL400 adds must be exactly the designed ``ttgB``
    deletion and the ``PP_4398`` mutation the Results name. Anything else means the
    column-to-strain assignment this loader writes the parents' calls under is wrong.
    """
    wt, ipl300, ipl400 = (
        FOUNDER_CLONE_COLUMNS["KT2440"],
        FOUNDER_CLONE_COLUMNS["IPL300"],
        FOUNDER_CLONE_COLUMNS["IPL400"],
    )
    positions = {
        column: tuple(sorted(call.position for call in calls if column in call.clones))
        for column in (wt, ipl300, ipl400)
    }
    if positions[wt]:
        raise CrossSourceError(
            f"{wt} carries {len(positions[wt])} calls; the matrix is called against "
            "KT2440 WT, so its own founder column must carry none and the "
            "column-to-strain assignment is wrong"
        )
    extra = tuple(sorted(set(positions[ipl400]) - set(positions[ipl300])))
    if not set(positions[ipl300]) < set(positions[ipl400]):
        raise CrossSourceError(
            f"{ipl400}'s calls are not a strict superset of {ipl300}'s "
            f"({positions[ipl300]} vs {positions[ipl400]}); IPL400 is IPL300 plus one "
            "deletion, so the founder assignment does not hold"
        )
    genes = tuple(
        call.gene_field
        for call in sorted(calls, key=lambda c: c.position)
        if call.position in extra
    )
    named = SOURCED_VALUES["preexisting_parent_mutations"].value[-1]
    designed = SOURCED_VALUES["ipl400_extra_deletion"].value
    if len(genes) != 2 or named not in genes:
        raise CrossSourceError(
            f"{ipl400} adds {genes} over {ipl300}; the Results state it adds the "
            f"{designed} (ttgB) deletion and a mutation in {named}"
        )
    return FounderCheck(
        wt_column=wt,
        wt_n_calls=0,
        ipl300_column=ipl300,
        ipl300_positions=positions[ipl300],
        ipl400_column=ipl400,
        ipl400_positions=positions[ipl400],
        ipl400_only_positions=extra,
        ipl400_only_genes=genes,
    )


def designed_deletion_tags(strain: str) -> tuple[str, ...]:
    """The locus tags one writable strain carries as a DESIGNED deletion."""
    if strain == "KT2440":
        return ()
    if strain == "IPL300":
        return IPL300_DELETIONS
    if strain in {"IPL400", *EVOLVED_ISOLATE_CLONE_COLUMNS}:
        return (*IPL300_DELETIONS, IPL400_EXTRA_DELETION)
    raise ValueError(f"{strain!r} is not a strain of this campaign")


def resolve_variant_loci(
    genome: PPutidaKT2440Genome, tokens: Sequence[str], *, label: str
) -> tuple[dict[str, str], LocusTagReconciliation]:
    """``{released Gene token: its KT2440 locus tag}``, refusing any token that misses.

    The ``Gene`` cell names a gene SYMBOL far more often than a locus tag (measured over
    all 241 distinct tokens of the sheet: 151 are already locus tags and 90 resolve
    through the gene-symbol layer; ``ttgB`` is ``PP_1385``, ``adhP`` is ``PP_3839``,
    ``ivd,mccB,liuC,mccA`` are ``PP_4064``-``PP_4067``). A token that does not land on
    exactly one locus of the pinned assembly is a hard error: there is no mapping to
    invent, and a guessed tag would key a variant onto the wrong gene. Measured on the
    whole sheet, one token is ambiguous (``asd`` -> ``PP_1989`` or ``PP_1992``), so a
    clone carrying that row is refused with its name rather than written.
    """
    stored, report = reconcile_locus_tags(genome, pd.Series(list(tokens)), label=label)
    report.require_resolved(MIN_RESOLVED_FRACTION)
    mapping = dict(zip(tokens, stored.tolist(), strict=True))
    pattern = LOCUS_TAG_PATTERNS[report.gene_namespace]
    outside = sorted(
        f"{released} -> {tag}"
        for released, tag in mapping.items()
        if pattern.match(tag) is None
    )
    if outside:
        raise RuntimeError(
            f"{label}: {outside} do not resolve to one {report.gene_namespace} locus, "
            "so no bacterial perturbation leaf accepts them and no mapping is invented"
        )
    return mapping, report


def _derived_mapping(released: str, locus_tag: str) -> DerivedIdentifierMapping | None:
    """How ``locus_tag`` was reached from the ``Gene`` cell, or None when they agree."""
    if released == locus_tag:
        return None
    return DerivedIdentifierMapping(source_identifier=released, route="gene_symbol")


def _clone_call(row: CalledVariant, clone_column: str) -> BacterialVariantCall:
    """One clone's call of one released row, with every cell kept verbatim.

    ``position_end`` equals ``position_start`` for EVERY row, including the multi-base
    ones. The release gives a 1-based ``Position`` and a LENGTH fused into ``Sequence
    Change`` (``Δ5,553 bp``, ``(CCAC)2→1``, ``2 bp→CG``) and never an end coordinate, so
    an end would have to be synthesized by parsing that string and asserting which side
    of the position the length runs to. Neither is released, so no end is asserted; the
    released extent stays verbatim in ``sequence_change``. ``deleted_span`` is left None
    on the span leaf for the same reason.

    ``reference_allele``, ``alternate_allele``, ``amino_acid_change``, ``codon_change``
    and ``codon_number`` stay None: the sheet fuses all of them into ``Sequence Change``
    and ``Details`` (``C→G``, ``G476A (GGT→GCT)``) rather than giving them their own
    columns, and those two cells are stored verbatim.
    """
    frequency = row.clones[clone_column]
    return BacterialVariantCall(
        variant_type=VARIANT_TYPE_BY_STATEMENT[row.mutation_type],
        type_statement=row.mutation_type,
        reference_sequence=row.replicon,
        position_start=row.position,
        position_end=row.position,
        sequence_change=row.sequence_change,
        annotation=row.detail or None,
        call_mode=VariantCallMode.clone,
        frequency_statement=frequency,
        frequency=float(frequency),
        frequency_basis=CLONE_FREQUENCY_BASIS,
        caller=SOURCED_VALUES["variant_caller"].value,
    )


class CalledVariantAccounting(BaseModel):
    """What one clone column's called variants became on the record that carries them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clone_column: str
    strain: str
    n_called_rows: int
    n_candidate_perturbations: int
    n_written: int
    n_sequence_variants: int
    n_intergenic_variants: int
    n_span_deletion_loci: int
    n_restating_a_designed_deletion: int
    restated_loci: tuple[str, ...]
    written_identifiers: tuple[str, ...]


def called_variant_perturbations(
    clone_column: str,
    *,
    strain: str,
    calls: Sequence[CalledVariant],
    loci: Mapping[str, str],
) -> tuple[list[GenePerturbationType], CalledVariantAccounting]:
    """One clone column's called variants as #731 perturbations, with the dedup counted.

    Row shape decides the leaf (:func:`released_row_representation`), and a span
    deletion becomes ONE perturbation PER COVERED LOCUS, all sharing the event's
    ``span_designation``. The list is ordered by ``position_start`` so that two calls in
    one locus (A12_F53_I1 carries ``PP_3415`` P293S at 3,866,001 and V46I at 3,866,742)
    land in a fixed order: ``Genotype.sort_perturbations`` keys on (name, type, perturbed
    name), which ties for those two, and ``sorted`` is stable, so the input order decides.

    DEDUPLICATION. A call whose resolved locus is one the record's own strain already
    carries as a designed ``BacterialDeletionPerturbation`` is DROPPED, because it
    restates a lesion the genotype already states and absence has one encoding. Measured
    on the pinned sheet, this is not a corner case: ``3063719 DEL PP_2675``, ``4362918
    DEL adhP`` (``PP_3839``) and the four loci of ``4588139 DEL ivd,mccB,liuC,mccA``
    (``PP_4064``-``PP_4067``) restate IPL300's six designed deletions, and ``1578244 DEL
    ttgB`` (``PP_1385``) restates IPL400's seventh. The drop is per LOCUS, so a span
    event keeps its full ``span_systematic_gene_names`` -- the event does remove them
    all -- and only stops writing a second perturbation for the designed one.
    """
    designed = frozenset(designed_deletion_tags(strain))
    rows = sorted(
        (call for call in calls if clone_column in call.clones),
        key=lambda call: call.position,
    )
    written: list[GenePerturbationType] = []
    restated: list[str] = []
    candidates = 0
    counts: Counter[VariantRepresentation] = Counter()
    for row in rows:
        call = _clone_call(row, clone_column)
        if row.representation == "bacterial_site_variant":
            candidates += 1
            counts[row.representation] += 1
            written.append(
                BacterialSiteVariantPerturbation(
                    systematic_gene_name=(
                        BacterialSiteVariantPerturbation.site_id(call)
                    ),
                    perturbed_gene_name=row.gene_field,
                    gene_namespace=KT2440_NAMESPACE,
                    call=call,
                    site_kind=VariantSiteKind.intergenic,
                    flanking_systematic_gene_names=tuple(
                        loci[token] for token in row.loci
                    ),
                    flanking_gene_statement=row.gene_field,
                )
            )
            continue
        if row.representation == "bacterial_span_deletion":
            tags = tuple(loci[token] for token in row.loci)
            designation = BacterialSpanDeletionPerturbation.designation(call)
            for released, tag in zip(row.loci, tags, strict=True):
                candidates += 1
                if tag in designed:
                    restated.append(tag)
                    continue
                counts[row.representation] += 1
                written.append(
                    BacterialSpanDeletionPerturbation(
                        systematic_gene_name=tag,
                        perturbed_gene_name=released,
                        gene_namespace=KT2440_NAMESPACE,
                        identifier_mapping=_derived_mapping(released, tag),
                        call=call,
                        span_designation=designation,
                        span_systematic_gene_names=tags,
                        deleted_span=None,
                    )
                )
            continue
        if len(row.loci) != 1:
            raise SheetExtractionError(
                f"{SHEET_MUTATIONS} position {row.position}: a call inside a locus "
                f"names {row.loci}, not one locus; its Details cell "
                f"({row.detail!r}) places it in a gene, so it cannot be a span"
            )
        candidates += 1
        released = row.loci[0]
        tag = loci[released]
        if tag in designed:
            restated.append(tag)
            continue
        counts[row.representation] += 1
        written.append(
            BacterialSequenceVariantPerturbation(
                systematic_gene_name=tag,
                perturbed_gene_name=released,
                gene_namespace=KT2440_NAMESPACE,
                identifier_mapping=_derived_mapping(released, tag),
                call=call,
            )
        )
    accounting = CalledVariantAccounting(
        clone_column=clone_column,
        strain=strain,
        n_called_rows=len(rows),
        n_candidate_perturbations=candidates,
        n_written=len(written),
        n_sequence_variants=counts["bacterial_sequence_variant"],
        n_intergenic_variants=counts["bacterial_site_variant"],
        n_span_deletion_loci=counts["bacterial_span_deletion"],
        n_restating_a_designed_deletion=len(restated),
        restated_loci=tuple(sorted(restated)),
        written_identifiers=tuple(p.systematic_gene_name for p in written),
    )
    if accounting.n_written + accounting.n_restating_a_designed_deletion != candidates:
        raise RuntimeError(
            f"{clone_column}: {accounting.n_written} written + "
            f"{accounting.n_restating_a_designed_deletion} restated != {candidates} "
            "candidate perturbations"
        )
    return written, accounting


def clone_variant_tokens(
    calls: Sequence[CalledVariant], clone_columns: Iterable[str]
) -> list[str]:
    """Every distinct ``Gene`` token the given clone columns need, in released order."""
    wanted = set(clone_columns)
    tokens: list[str] = []
    for call in sorted(calls, key=lambda c: c.position):
        if not wanted & set(call.clones):
            continue
        for token in call.loci:
            if token not in tokens:
                tokens.append(token)
    if not tokens:
        raise SheetExtractionError(
            f"{SHEET_MUTATIONS}: {sorted(wanted)} name no called locus"
        )
    return tokens


# --------------------------------------------------------------------------- #
# Cross-source assertions on Supplementary Table 3
# --------------------------------------------------------------------------- #
class RangeCheck(BaseModel):
    """One stated range, recomputed from Supplementary Table 3's own columns."""

    name: str
    stated: tuple[float, float]
    measured: tuple[float, float]


def range_checks(lineages: Sequence[TaleLineage]) -> list[RangeCheck]:
    """Recompute the Results' stated generation, CCD and HCHO-dose ranges, and refuse
    any disagreement.

    Three independent statements of Supplementary Table 3's content: the isoprenol arm's
    generation and CCD span, the HCHO arm's, and the HCHO arm's dose span given in g/L in
    the Results and in mM in the table.
    """
    isoprenol = [row for row in lineages if row.stressor == "isoprenol"]
    hcho = [row for row in lineages if row.stressor == "formaldehyde"]
    molar = FORMALDEHYDE_G_PER_MOL / 1000.0
    checks = [
        RangeCheck(
            name="isoprenol arm generations",
            stated=ISOPRENOL_GENERATIONS_RANGE,
            measured=(
                min(row.generations for row in isoprenol),
                max(row.generations for row in isoprenol),
            ),
        ),
        RangeCheck(
            name="isoprenol arm CCD (x1e12)",
            stated=ISOPRENOL_CCD_RANGE,
            measured=(
                min(row.ccd_1e12 for row in isoprenol),
                max(row.ccd_1e12 for row in isoprenol),
            ),
        ),
        RangeCheck(
            name="HCHO arm generations",
            stated=HCHO_GENERATIONS_RANGE,
            measured=(
                min(row.generations for row in hcho),
                max(row.generations for row in hcho),
            ),
        ),
        RangeCheck(
            name="HCHO arm CCD (x1e12)",
            stated=HCHO_CCD_RANGE,
            measured=(
                min(row.ccd_1e12 for row in hcho),
                max(row.ccd_1e12 for row in hcho),
            ),
        ),
        RangeCheck(
            name="HCHO arm dose, mM converted to g/L",
            stated=HCHO_DOSE_RANGE_G_PER_L,
            measured=(
                round(min(row.starting_dose for row in hcho) * molar, 2),
                round(max(row.ending_dose for row in hcho) * molar, 2),
            ),
        ),
    ]
    bad = [check for check in checks if check.stated != check.measured]
    if bad:
        raise CrossSourceError(
            "; ".join(
                f"{check.name}: stated {check.stated}, measured {check.measured}"
                for check in bad
            )
        )
    units = {row.starting_dose_unit for row in isoprenol} | {
        row.ending_dose_unit for row in isoprenol
    }
    if units != {"g/L"}:
        raise CrossSourceError(f"the isoprenol arm's doses are in {sorted(units)}")
    doses = {row.starting_dose for row in isoprenol}
    if doses != {TALE_DOSE_G_PER_L}:
        raise CrossSourceError(
            f"Supplementary Table 3's isoprenol starting doses are {sorted(doses)}, not "
            f"the {TALE_DOSE_G_PER_L} g/L the Results state"
        )
    if PP3024_FOLD >= PP3024_PAIR_MEAN_FOLD:
        raise CrossSourceError(
            f"the dPP_3024 fold {PP3024_FOLD} is not below the two-mutant mean "
            f"{PP3024_PAIR_MEAN_FOLD}, so the reading that it is 'most of' the "
            "improvement does not hold"
        )
    return checks


class ToleranceArm(BaseModel):
    """One starting strain's aggregated tolerance measurement at one isoprenol dose."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: str
    ale_labels: tuple[str, ...]
    dose_g_per_l: float
    growth_rates: tuple[float, ...]
    mean_growth_rate: float
    log2_ratio_to_wt: float
    n_samples: int


def tolerance_arms(
    lineages: Sequence[TaleLineage],
) -> tuple[ToleranceArm, list[ToleranceArm]]:
    """``(reference arm, record arms)`` from Supplementary Table 3's isoprenol lineages.

    KT2440's four lineages are the reference arm (log2(1) = 0); IPL300's and IPL400's are
    the records. Each arm's four lineages are the biological replicates the Results
    declare, so each becomes ONE record with ``n_samples = 4``.
    """
    by_strain: dict[str, list[TaleLineage]] = {}
    for row in lineages:
        if row.stressor != "isoprenol":
            continue
        by_strain.setdefault(row.starting_strain, []).append(row)
    counts = {strain: len(rows) for strain, rows in by_strain.items()}
    if set(counts) != {"KT2440", "IPL300", "IPL400"} or set(counts.values()) != {
        N_TALE_REPLICATES
    }:
        raise CrossSourceError(
            f"the isoprenol arm holds {counts}, not {N_TALE_REPLICATES} lineages for "
            "each of KT2440, IPL300 and IPL400"
        )

    def arm(strain: str, wt_mean: float | None) -> ToleranceArm:
        rows = sorted(by_strain[strain], key=lambda row: row.ale_number)
        rates = tuple(row.initial_growth_rate for row in rows)
        mean = statistics.fmean(rates)
        return ToleranceArm(
            strain=strain,
            ale_labels=tuple(row.ale_label for row in rows),
            dose_g_per_l=TALE_DOSE_G_PER_L,
            growth_rates=rates,
            mean_growth_rate=mean,
            log2_ratio_to_wt=0.0 if wt_mean is None else math.log2(mean / wt_mean),
            n_samples=len(rates),
        )

    reference = arm("KT2440", None)
    records = [
        arm(strain, reference.mean_growth_rate) for strain in ("IPL300", "IPL400")
    ]
    return reference, records


# --------------------------------------------------------------------------- #
# Record construction (pure, no files)
# --------------------------------------------------------------------------- #
PUBLICATION = Publication(
    pubmed_id=None,
    pubmed_url=None,
    doi=PAPER_DOI,
    doi_url=f"https://doi.org/{PAPER_DOI}",
)
RESPONSE_UNITS = (
    "log2(maximum specific growth rate of the strain / of KT2440 WT), both in the same "
    "4 g/L glucose M9 minimal medium at the same isoprenol dose"
)
MEASUREMENT_TYPE = MeasurementType.log2_ratio
ASSAY_TYPE = AssayType.liquid_od_growth


def isoprenol_compound() -> Compound:
    """Isoprenol through the shared compound-identity layer.

    The table's row (landed in PR #729) must carry :data:`ISOPRENOL_INCHIKEY` or the build
    stops; a table without the row would return the name with an ``inchikey`` gap.
    """
    compound = resolved_compound(ISOPRENOL_LABEL)
    if compound.inchikey is not None and compound.inchikey != ISOPRENOL_INCHIKEY:
        raise CompoundIdentityConflictError(
            f"the compound-identity table resolves {ISOPRENOL_LABEL!r} to "
            f"{compound.inchikey}, not {ISOPRENOL_INCHIKEY}"
        )
    return compound


def isoprenol_environment(
    dose_g_per_l: float, duration_gap: ProvenanceGap
) -> Environment:
    """Lim 2025's M9 at 4 g/L glucose, with isoprenol at ``dose_g_per_l``.

    A plain ``Environment``, not a ``CultureEnvironment``: ``Experiment.environment`` is
    annotated ``Environment`` and pydantic serializes by the DECLARED type, so a
    subclass's culture slots would be dumped away without an error. Temperature and
    duration are typed gaps -- the paper states neither for any loaded arm.
    """
    return Environment(
        media=M9_NREL_LIM2025,
        temperature=None,
        perturbations=[
            SmallMoleculePerturbation(
                compound=isoprenol_compound(),
                concentration=Concentration(
                    value=dose_g_per_l, unit=ConcentrationUnit.g_per_l
                ),
            )
        ],
        duration_hours=None,
        provenance_gaps=[TEMPERATURE_GAP, duration_gap],
    )


def unstressed_environment() -> Environment:
    """Lim 2025's M9 at 4 g/L glucose with nothing added: the proteome reference arm."""
    return Environment(
        media=M9_NREL_LIM2025,
        temperature=None,
        duration_hours=None,
        provenance_gaps=[TEMPERATURE_GAP, PROTEOME_DURATION_GAP],
    )


def deletion(
    locus_tag: str, symbol: str, *, construction: StrainConstruction | None = None
) -> BacterialDeletionPerturbation:
    """One markerless in-frame deletion on the KT2440 locus-tag namespace.

    ``cassette`` is None by design: "Scarless gene deletion was performed by following
    the conjugation protocol", so there is no replacement cassette to name. ``collection``
    is None because these are this study's own strains, not a catalogued collection.
    """
    return BacterialDeletionPerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=symbol,
        gene_namespace=KT2440_NAMESPACE,
        construction=construction,
    )


def ipl_background(
    strain: str, symbols: Mapping[str, str]
) -> BacterialStrainBackground:
    """The whole genome content IPL300 or IPL400 carries beyond the KT2440 assembly.

    Typed the way the de Siqueira 2025 loader types the same physical lesion (row 14's
    ``PT`` strain is ``dPP_2675 d14-PP_2676``, built from the same Thompson 2020
    strain): every designed deletion is a ``full_deletion`` allele, and ``PP_2676``'s
    14-codon N-terminal truncation is a ``partial_deletion`` allele, which is the one
    place it CAN be stated -- the bacterial perturbation axis has no partial-deletion
    leaf, so a perturbation there would assert ``state="absent"``.

    ``deleted_span`` is a typed gap on every allele: the sources give no coordinates,
    and the ``d14`` designation does not say whether 14 counts base pairs or codons.
    """
    if strain == "IPL300":
        deleted: tuple[str, ...] = IPL300_DELETIONS
    elif strain == "IPL400":
        deleted = (*IPL300_DELETIONS, IPL400_EXTRA_DELETION)
    else:
        raise ValueError(
            f"{strain!r} is not a stacked-deletion strain of this campaign"
        )
    construction = SOURCED_VALUES["pp2675_construction"]
    abolished = SOURCED_VALUES["ipl_catabolism_abolished"]
    span_gap = ProvenanceGap(
        field="deleted_span",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_SI1_LOOKED_IN,
        note="neither Lim 2025's Supplementary Table 1 nor the Thompson 2020 strain "
        "table gives coordinates for any of these deletions",
    )
    alleles = [
        BacterialBackgroundAllele(
            systematic_gene_name=tag,
            gene_namespace=KT2440_NAMESPACE,
            gene_name=symbols.get(tag, tag),
            allele_name=f"d{tag}",
            edit=AlleleEdit.full_deletion,
            functional=False,
            deleted_span=None,
            provenance=[construction, abolished],
            provenance_gaps=[span_gap],
        )
        for tag in sorted(deleted)
    ]
    alleles.append(
        BacterialBackgroundAllele(
            systematic_gene_name=PP2676_LOCUS,
            gene_namespace=KT2440_NAMESPACE,
            gene_name=symbols.get(PP2676_LOCUS, PP2676_LOCUS),
            allele_name=f"d14-{PP2676_LOCUS}",
            edit=AlleleEdit.partial_deletion,
            functional=False,
            deleted_span=None,
            provenance=[construction],
            provenance_gaps=[span_gap],
        )
    )
    return BacterialStrainBackground(
        name=strain,
        reference_strain="KT2440",
        assembly_set=KT2440_ASSEMBLY_SET,
        parents=["KT2440"],
        construction="scarless, markerless in-frame deletions in KT2440 by conjugation "
        "or Cpf1/RecT recombineering; the PP_2675 / PP_2676 pair is one deletion event "
        "carried over from the Thompson 2020 strain",
        genotype_statement=SI1_STATEMENTS[
            "ipl300_genotype" if strain == "IPL300" else "ipl400_genotype"
        ],
        alleles=sorted(alleles, key=lambda allele: allele.systematic_gene_name),
        provenance=[SOURCED_VALUES["ipl300_genotype"], construction],
    )


def strain_genotype(
    strain: str, symbols: Mapping[str, str], called: Sequence[GenePerturbationType] = ()
) -> Genotype:
    """The DESIGNED deletions of one writable strain, plus the calls it is given.

    KT2440 WT is the empty genotype (the reference). IPL300 is its six deletions, IPL400
    those plus ``PP_1385``, and an evolved isolate its parent's seven. ``called`` is the
    strain's own called variants as #731 perturbations, which the caller builds with
    :func:`called_variant_perturbations` (already deduplicated against these designed
    deletions) and passes explicitly; an empty sequence means no calls were given, which
    is the case for ``KT2440 dPP_3024``, a reverse-engineered strain that is not one of
    the 49 clone columns.
    """
    if strain == "KT2440":
        return Genotype(perturbations=list(called))
    if strain in {"IPL300", "IPL400", *EVOLVED_ISOLATE_CLONE_COLUMNS}:
        tags = designed_deletion_tags(strain)
    else:
        raise ValueError(f"{strain!r} is not a starting strain of this campaign")
    return Genotype(
        perturbations=[deletion(tag, symbols.get(tag, tag)) for tag in sorted(tags)]
        + list(called)
    )


def pp3024_genotype(symbols: Mapping[str, str]) -> Genotype:
    """KT2440 dPP_3024: the one reverse-engineered strain with a released number."""
    return Genotype(
        perturbations=[
            deletion(
                PP3024_LOCUS,
                symbols.get(PP3024_LOCUS, PP3024_LOCUS),
                construction=StrainConstruction(
                    strain_accession=PP3024_STRAIN_ACCESSION
                ),
            )
        ]
    )


def response_phenotype(
    value: float, *, n_samples: int | None, gaps: Sequence[ProvenanceGap]
) -> EnvironmentResponsePhenotype:
    """One strain's log2 growth-rate ratio to KT2440 in the same condition."""
    return EnvironmentResponsePhenotype(
        measurement_type=MEASUREMENT_TYPE,
        assay_type=ASSAY_TYPE,
        environment_response=value,
        n_samples=n_samples,
        sample_unit=None if n_samples is None else SampleUnit.biological_replicate,
        units=RESPONSE_UNITS,
        provenance_gaps=list(gaps),
    )


def abundance_phenotype(
    arm: Mapping[str, tuple[float, float]],
) -> ProteinAbundancePhenotype:
    """One arm's Top3 log2 abundances, with the SE derived from the released sample SD."""
    root_n = math.sqrt(PROTEOME_N_REPLICATES)
    return ProteinAbundancePhenotype(
        protein_abundance={tag: mean for tag, (mean, _) in arm.items()},
        protein_abundance_se={tag: sd / root_n for tag, (_, sd) in arm.items()},
        n_replicates=dict.fromkeys(arm, PROTEOME_N_REPLICATES),
        measurement_type=PROTEOME_MEASUREMENT_TYPE,
    )


# --------------------------------------------------------------------------- #
# The retention ledger
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, what it removed, and why."""

    rule: str
    scope: str
    description: str
    n_items: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    source_rows: int
    reference_rows: list[str]
    candidate_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]
    notes: list[str] = []

    def check(self) -> None:
        """Refuse a ledger whose record-scoped rules do not account for every drop."""
        accounted = sum(rule.n_items for rule in self.rules if rule.scope == "record")
        if accounted != self.dropped_records:
            raise RuntimeError(
                f"{self.dataset}: record-scoped rules account for {accounted} of "
                f"{self.dropped_records} dropped records"
            )
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_records} "
                f"dropped != {self.candidate_records} candidates"
            )


class GenotypeGap(BaseModel):
    """Genomic content a loaded strain carries that no class on the named axis states."""

    strain: str
    content: str
    reason: str
    source_quote: str


def standard_names(genome: PPutidaKT2440Genome, tags: Iterable[str]) -> dict[str, str]:
    """``{locus tag: annotation gene symbol}``, keeping the tag when it does not
    round-trip through the resolver, so a stored pair is always resolvable.
    """
    out: dict[str, str] = {}
    for tag in tags:
        symbol = genome.genbank.loci[tag].symbol
        if symbol is None:
            out[tag] = tag
            continue
        resolution = genome.resolve_gene_name(symbol)
        out[tag] = symbol if resolution.systematic_name == tag else tag
    return out


def genotype_loci() -> tuple[str, ...]:
    """Every locus this module writes or documents a gap on, as the paper spells it."""
    return (*IPL300_DELETIONS, IPL400_EXTRA_DELETION, PP3024_LOCUS, PP2676_LOCUS)


def reconcile_genotype_loci(
    genome: PPutidaKT2440Genome, *, label: str
) -> tuple[dict[str, str], LocusTagReconciliation]:
    """Place every genotype locus on the pinned assembly and report the histogram.

    Checklist item 4. The loci are PP_ tags the paper itself prints, so the threshold is
    a complete resolution: anything less means the wrong annotation, not a hard name.
    """
    tags = genotype_loci()
    stored, report = reconcile_locus_tags(genome, pd.Series(list(tags)), label=label)
    report.require_resolved(MIN_RESOLVED_FRACTION)
    pattern = LOCUS_TAG_PATTERNS[report.gene_namespace]
    outside = [tag for tag in stored.tolist() if pattern.match(tag) is None]
    if outside:
        raise RuntimeError(
            f"{label}: {outside} are not locus tags of {report.gene_namespace}, so no "
            "bacterial perturbation leaf accepts them"
        )
    return standard_names(genome, stored.tolist()), report


def genotype_gaps() -> list[GenotypeGap]:
    """The genomic content of the loaded strains that no class can state.

    The parents' own called variants were the first entry here and no longer are: the
    three #731 leaves hold them, and every loaded record built from a clone column now
    carries them as perturbations. What remains is the strain-BACKGROUND axis, where a
    called variant still has no carrier, and ``PP_2676``'s truncation.
    """
    return [
        GenotypeGap(
            strain="IPL300 and IPL400",
            content=f"{PP2676_LOCUS}'s 14-codon N-terminal truncation, on the "
            "PERTURBATION axis only",
            reason="the bacterial perturbation axis has no partial-deletion leaf "
            "(BacterialDeletionPerturbation is state='absent'), so a strain-against-"
            "KT2440 record cannot state it. It IS typed as a "
            "BacterialBackgroundAllele(partial_deletion) on the IPL400 background the "
            "proteome loader writes, which is the same typing the de Siqueira 2025 "
            "loader gives the same lesion",
            source_quote=SI1_STATEMENTS["pp2676_truncation"],
        ),
        GenotypeGap(
            strain=" and ".join(sorted(EVOLVED_ISOLATE_CLONE_COLUMNS)),
            content="each isolate's own called variants on the strain-BACKGROUND axis. "
            "The proteome record's genome_reference carries its parent IPL400's "
            "background (seven full_deletion alleles plus PP_2676), which is a FLOOR on "
            "the genomic content of the isolate's own unstressed reference arm",
            reason="BacterialBackgroundAllele requires a non-optional functional: bool, "
            "which a called variant's consequence is unknown for and a ProvenanceGap "
            "cannot cover (a gap must name a field that is None), and "
            "BacterialStrainBackground permits one allele entry per locus, which refuses "
            "A12_F53_I1's two PP_3415 calls. The calls ARE written, on the PERTURBATION "
            "axis, as the three #731 leaves on each record's genotype",
            source_quote=SOURCED_VALUES["evolved_isolate_parent"].quote,
        ),
    ]


# --------------------------------------------------------------------------- #
# Isoprenol tolerance of the writable strains
# --------------------------------------------------------------------------- #
@register_dataset
class IsoprenolToleranceLim2025Dataset(ExperimentDataset):
    """Isoprenol tolerance of IPL300, IPL400 and KT2440 dPP_3024 (Lim 2025)."""

    REFERENCE_STRAIN: ClassVar[str] = "KT2440"

    def __init__(
        self,
        root: str = TOLERANCE_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        pputida_genome: PPutidaKT2440Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``pputida_genome`` is injected by the build entry points."""
        self.pputida_genome = pputida_genome
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
        """Both consumed supplements, linked from the raw mirror."""
        return [raw.name for raw in RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking its manifest pin."""
        _link_mirror_files(self.raw_dir, _data_root())
        log.info("Lim 2025 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or the pinned assembly's default cache."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", "KT2440")
        expected = BACTERIAL_ASSEMBLY_SETS["KT2440"]
        if self.pputida_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.pputida_genome.ASSEMBLY_SET}"
            )
        return self.pputida_genome

    def _drop_log(
        self,
        matrix: MutationMatrix,
        kept: int,
        accounting: Sequence[CalledVariantAccounting],
    ) -> DropLog:
        """Every row of Supplementary Table 3 and every figure-only readout, counted."""
        founder = ", ".join(
            f"{clone} ({n} calls)"
            for clone, n in sorted(matrix.founder_call_counts.items())
        )
        return DropLog(
            dataset=self.name,
            source_rows=_S3_N_ROWS,
            reference_rows=[f"KT2440 ALE 1-{N_TALE_REPLICATES} (the log2(1) = 0 arm)"],
            candidate_records=kept + 2,
            kept_records=kept,
            dropped_records=2,
            rules=[
                DropRule(
                    rule="hcho_arm_has_no_strain_other_than_the_reference",
                    scope="record",
                    description="the formaldehyde arm tolerized only the WT, which is "
                    "this dataset's reference, so its aggregated initial growth rate "
                    "would be the reference against itself",
                    n_items=1,
                    items=["KT2440 at 1 mM formaldehyde (ALE 13-16)"],
                ),
                DropRule(
                    rule="final_growth_rate_is_an_evolved_population",
                    scope="record",
                    description="Supplementary Table 3's final growth rate is the "
                    "average over a lineage's three LAST flasks, so its strain is the "
                    "evolving POPULATION in that flask, not one of the 46 sequenced "
                    "isolates. The mutation matrix calls CLONES "
                    f"(VariantCallMode.clone; {matrix.n_calls} calls, "
                    f"{matrix.call_frequencies} by frequency), so it states no "
                    "population allele frequency for a flask, and a population genotype "
                    "would need a threshold on a frequency the release never gives for "
                    "one. This refusal is unchanged by the #731 leaves, which hold a "
                    "CLONE's calls",
                    n_items=1,
                    items=[f"{_S3_N_ROWS} final growth rates (one per lineage)"],
                ),
                DropRule(
                    rule="evolved_isolate_has_no_released_per_isolate_number",
                    scope="strain",
                    description="the 46 evolved isolates ARE writable now: each clone "
                    "column's calls become #731 perturbations on the parent's designed "
                    f"deletions ({matrix.n_rows} released rows, {matrix.n_calls} "
                    f"per-clone calls, {matrix.n_evolved_columns} evolved clone columns; "
                    f"{matrix.n_intergenic} rows keyed on a site and "
                    f"{matrix.n_multi_locus} spanning several loci). This dataset loads "
                    "none of them because Supplementary Table 3 releases a growth rate "
                    "per LINEAGE and not per isolate, and Fig. 2A's per-isolate rates "
                    "are a figure with no numbers; the two isolates whose proteome IS "
                    f"released are records of {ProteomeLim2025Dataset.__name__}",
                    n_items=matrix.n_evolved_columns,
                    items=[f"founder columns kept out of the evolved count: {founder}"],
                ),
                DropRule(
                    rule="call_restates_a_designed_deletion",
                    scope="perturbation",
                    description="a called variant whose resolved locus the record's own "
                    "strain already carries as a designed BacterialDeletionPerturbation "
                    "is dropped, so absence has one encoding. Measured on the loaded "
                    "records: PP_2675 (position 3063719), adhP/PP_3839 (4362918), the "
                    "four loci of ivd,mccB,liuC,mccA/PP_4064-PP_4067 (4588139) and, for "
                    "IPL400, ttgB/PP_1385 (1578244)",
                    n_items=sum(
                        row.n_restating_a_designed_deletion for row in accounting
                    ),
                    items=[
                        f"{row.strain} ({row.clone_column}): "
                        f"{row.n_restating_a_designed_deletion} of "
                        f"{row.n_candidate_perturbations} restate "
                        f"{list(row.restated_loci)}"
                        for row in accounting
                    ],
                ),
                DropRule(
                    rule="released_only_as_a_figure",
                    scope="readout",
                    description="the rest of the phenotyping releases no numbers: "
                    "Fig. 2A (16 strains x 2 isoprenol doses), Fig. 3A (the other six "
                    "reverse-engineered strains), Fig. 3B and 5A/5D (isoprenol titers), "
                    "Fig. 5B/5E (residual glucose), Fig. 4D and S3B (isoprenol "
                    "degradation), Fig. S1C-F (growth curves). The six other deletion "
                    "strains and all eight pIY670 strains are writable with nothing "
                    "released to write about them",
                    n_items=14,
                    items=[
                        "KT2440 dgnuR",
                        "KT2440 dttgB",
                        "KT2440 dttgB_L (dPP_1385-1395)",
                        "KT2440 dPP_1395",
                        "KT2440 dPP_3024-dPP_5558",
                        "KT2440 dmxtR (PP_1695)",
                        "the seven deletion strains carrying pIY670",
                        "KT2440 + pIY670",
                    ],
                ),
                DropRule(
                    rule="no_titer_is_released_per_writable_strain",
                    scope="readout",
                    description="the Results state isoprenol titers only as ranges over "
                    "several strains (60-70 mg/L for two evolved isolates, 280-400 mg/L "
                    "for 'their starting strain') or for derivatives of the evolved "
                    "isolate A10_F63_I1 (337 mg/L dfleQ, 370 mg/L dmvaB), so no "
                    "ProductTiterExperiment record is keyed to a writable strain",
                    n_items=0,
                ),
            ],
            notes=[
                "the three starting strains are KT2440, IPL300 and IPL400; KT2440 is the "
                "reference arm, so two of the three carry records",
                f"each record aggregates {N_TALE_REPLICATES} ALE lineages, which the "
                "Results declare to be independent biological replicates; the "
                "per-lineage numbers are in preprocess/table_s3.csv",
                "the dPP_3024 record is the only phenotype keyed to one of the seven "
                "reverse-engineered strains, and its 6 g/L dose makes it a different "
                "condition from the two TALE-dose records",
                f"the released matrix holds {matrix.n_rows} rows on "
                f"{matrix.n_distinct_positions} distinct positions while the Results "
                "state 158 unique mutations in 73 regions; the disagreement is reported, "
                "not reconciled",
            ],
        )

    @post_process
    def process(self) -> None:
        """Parse Supplementary Table 3, build the three records, write the LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        docx_path = osp.join(self.raw_dir, SI1_DOCX)
        xlsx_path = osp.join(self.raw_dir, SI2_XLSX)

        text = docx_text(docx_path)
        absent = sorted(
            name for name, quote in SI1_STATEMENTS.items() if quote not in text
        )
        if absent:
            raise TableExtractionError(
                f"{SI1_DOCX} no longer states {absent}; the quotes this loader relies on "
                "have moved and the build refuses rather than storing them"
            )
        lineages = parse_table_s3(docx_tables(docx_path))
        digest = table_s3_digest(lineages)
        if digest != TABLE_S3_SHA256:
            raise TableExtractionError(
                f"parsed Supplementary Table 3 sha256 {digest}, pinned {TABLE_S3_SHA256}"
            )
        checks = range_checks(lineages)
        reference_arm, record_arms = tolerance_arms(lineages)
        matrix = read_mutation_matrix(xlsx_path)
        variant_calls = read_variant_calls(xlsx_path)
        if len(variant_calls) != matrix.n_rows:
            raise SheetExtractionError(
                f"{len(variant_calls)} typed calls over {matrix.n_rows} released rows"
            )

        founder_check = assert_founder_columns(variant_calls)

        genome = self._genome()
        symbols, report = reconcile_genotype_loci(genome, label=self.name)
        log.info(
            "Lim 2025 tolerance: genotype loci %s",
            {status.value: n for status, n in report.status_histogram.items()},
        )
        founder_columns = [FOUNDER_CLONE_COLUMNS[arm.strain] for arm in record_arms]
        variant_loci, variant_report = resolve_variant_loci(
            genome,
            clone_variant_tokens(variant_calls, founder_columns),
            label=f"{self.name} called-variant loci",
        )
        called: dict[str, list[GenePerturbationType]] = {}
        accounting: list[CalledVariantAccounting] = []
        for arm in record_arms:
            perturbations, row_accounting = called_variant_perturbations(
                FOUNDER_CLONE_COLUMNS[arm.strain],
                strain=arm.strain,
                calls=variant_calls,
                loci=variant_loci,
            )
            called[arm.strain] = perturbations
            accounting.append(row_accounting)

        genome_reference = assembly_reference("KT2440")
        tale_environment = isoprenol_environment(TALE_DOSE_G_PER_L, TALE_DURATION_GAP)
        panel_environment = isoprenol_environment(
            PANEL_DOSE_G_PER_L, PANEL_DURATION_GAP
        )
        rows: list[dict[str, Any]] = []
        records: list[
            tuple[Genotype, Environment, EnvironmentResponsePhenotype, dict[str, Any]]
        ] = []
        for arm in record_arms:
            records.append(
                (
                    strain_genotype(arm.strain, symbols, called[arm.strain]),
                    tale_environment,
                    response_phenotype(
                        arm.log2_ratio_to_wt,
                        n_samples=arm.n_samples,
                        gaps=RESPONSE_UNCERTAINTY_GAPS,
                    ),
                    {
                        "strain": arm.strain,
                        "source": "Supplementary Table 3, initial growth rate",
                        "isoprenol_g_per_l": arm.dose_g_per_l,
                        "ale_labels": ";".join(arm.ale_labels),
                        "growth_rates": ";".join(f"{v}" for v in arm.growth_rates),
                        "mean_growth_rate": arm.mean_growth_rate,
                        "wt_mean_growth_rate": reference_arm.mean_growth_rate,
                        "log2_ratio_to_wt": arm.log2_ratio_to_wt,
                        "n_samples": arm.n_samples,
                        "clone_column": FOUNDER_CLONE_COLUMNS[arm.strain],
                        "n_called_perturbations": len(called[arm.strain]),
                    },
                )
            )
        records.append(
            (
                pp3024_genotype(symbols),
                panel_environment,
                response_phenotype(
                    math.log2(PP3024_FOLD), n_samples=None, gaps=PANEL_UNCERTAINTY_GAPS
                ),
                {
                    "strain": f"KT2440 d{PP3024_LOCUS}",
                    "source": "Fig. 3A text, relative growth rate",
                    "isoprenol_g_per_l": PANEL_DOSE_G_PER_L,
                    "ale_labels": "",
                    "growth_rates": "",
                    "mean_growth_rate": None,
                    "wt_mean_growth_rate": None,
                    "log2_ratio_to_wt": math.log2(PP3024_FOLD),
                    "n_samples": None,
                    # Not one of the 49 clone columns: a reverse-engineered KT2440
                    # deletion, never sequenced, so it has no called variants to carry.
                    "clone_column": "",
                    "n_called_perturbations": 0,
                },
            )
        )

        drop_log = self._drop_log(matrix, len(records), accounting)
        drop_log.check()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, (genotype, environment, phenotype, row) in enumerate(
                tqdm(records, desc="lim2025-tolerance")
            ):
                experiment = BacterialEnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=environment,
                    phenotype=phenotype,
                )
                reference = BacterialEnvironmentResponseExperimentReference(
                    dataset_name=self.name,
                    genome_reference=genome_reference,
                    environment_reference=environment.model_copy(),
                    phenotype_reference=response_phenotype(
                        0.0,
                        n_samples=reference_arm.n_samples,
                        gaps=RESPONSE_UNCERTAINTY_GAPS,
                    ),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
                rows.append({"record": index, **row})
        env.close()
        interned_env.close()

        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drop_log.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            report.model_dump_json(indent=2)
        )
        (out / "variant_accounting.json").write_text(matrix.model_dump_json(indent=2))
        (out / "called_variants.json").write_text(
            json.dumps([call.model_dump() for call in variant_calls], indent=2)
        )
        (out / "genotype_gaps.json").write_text(
            json.dumps([gap.model_dump() for gap in genotype_gaps()], indent=2)
        )
        (out / "called_variant_perturbations.json").write_text(
            json.dumps(
                {
                    "founder_check": founder_check.model_dump(),
                    "locus_reconciliation": json.loads(
                        variant_report.model_dump_json()
                    ),
                    "per_record": [row.model_dump() for row in accounting],
                },
                indent=2,
            )
        )
        (out / "extraction.json").write_text(
            json.dumps(
                {
                    "si1_docx_sha256": SI1_DOCX_SHA256,
                    "si2_xlsx_sha256": SI2_XLSX_SHA256,
                    "table_s3_sha256": digest,
                    "range_checks": [check.model_dump() for check in checks],
                    "si1_statements": SI1_STATEMENTS,
                },
                indent=2,
            )
        )
        pd.DataFrame(
            [
                {
                    "record": None,
                    "strain": reference_arm.strain,
                    "source": "Supplementary Table 3, initial growth rate",
                    "isoprenol_g_per_l": reference_arm.dose_g_per_l,
                    "ale_labels": ";".join(reference_arm.ale_labels),
                    "growth_rates": ";".join(
                        f"{v}" for v in reference_arm.growth_rates
                    ),
                    "mean_growth_rate": reference_arm.mean_growth_rate,
                    "wt_mean_growth_rate": reference_arm.mean_growth_rate,
                    "log2_ratio_to_wt": 0.0,
                    "n_samples": reference_arm.n_samples,
                    "clone_column": FOUNDER_CLONE_COLUMNS[reference_arm.strain],
                    "n_called_perturbations": 0,
                },
                *rows,
            ]
        ).astype({"record": "Int64"}).to_csv(out / "table_s3.csv", index=False)
        pd.DataFrame([row.model_dump() for row in lineages]).to_csv(
            out / "tale_lineages.csv", index=False
        )
        log.info(
            "Lim 2025 tolerance: %d records (IPL300, IPL400 at %g g/L isoprenol; "
            "d%s at %g g/L), reference KT2440 = log2(1) = 0; called variants written "
            "%s",
            len(records),
            TALE_DOSE_G_PER_L,
            PP3024_LOCUS,
            PANEL_DOSE_G_PER_L,
            {row.strain: row.n_written for row in accounting},
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# The parent strain's proteome under isoprenol
# --------------------------------------------------------------------------- #
@register_dataset
class ProteomeLim2025Dataset(ExperimentDataset):
    """IPL400 and its two evolved isolates under 4 g/L isoprenol (Lim 2025).

    Three records, one per strain, each the strain's Top3 log2 proteome under isoprenol
    referenced to its OWN unstressed arm. The two evolved isolates carry their called
    variants as #731 perturbations on top of their parent IPL400's designed deletions.
    """

    REFERENCE_STRAIN: ClassVar[str] = "KT2440"
    PARENT_STRAIN: ClassVar[str] = "IPL400"

    def __init__(
        self,
        root: str = PROTEOME_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        pputida_genome: PPutidaKT2440Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``pputida_genome`` is injected by the build entry points."""
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
        """Both consumed supplements, linked from the raw mirror."""
        return [raw.name for raw in RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking its manifest pin."""
        _link_mirror_files(self.raw_dir, _data_root())
        log.info("Lim 2025 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or the pinned assembly's default cache."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", "KT2440")
        expected = BACTERIAL_ASSEMBLY_SETS["KT2440"]
        if self.pputida_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.pputida_genome.ASSEMBLY_SET}"
            )
        return self.pputida_genome

    @staticmethod
    def _assert_duplicate_exports(path: str, dropped: Sequence[str]) -> dict[str, Any]:
        """The A12 sheets' IPL400 columns must be bit-identical to the A10 sheets'.

        Four sheets export the same IPL400 measurement; the two M9G sheets carry the same
        loci, and the A10 ``G+4IP`` sheet is a superset of the A12 one. Equality on every
        shared locus is what licenses storing ONE record rather than averaging four.
        """
        out: dict[str, Any] = {}
        for primary, alternate in (
            (SHEET_PROTEOME_M9G, SHEET_PROTEOME_M9G_ALT),
            (SHEET_PROTEOME_IPL, SHEET_PROTEOME_IPL_ALT),
        ):
            first = parent_arm(read_proteome_sheet(path, primary), dropped)
            second = parent_arm(read_proteome_sheet(path, alternate), dropped)
            shared = sorted(set(first) & set(second))
            if not shared:
                raise CrossSourceError(f"{primary} and {alternate} share no locus")
            differing = [tag for tag in shared if first[tag] != second[tag]]
            if differing:
                raise CrossSourceError(
                    f"{primary} and {alternate} disagree on IPL400 at "
                    f"{len(differing)} loci (e.g. {differing[:5]}); they are exports of "
                    "one measurement and must be bit-identical"
                )
            out[f"{primary}_vs_{alternate}"] = {
                "n_shared": len(shared),
                "n_primary_only": len(set(first) - set(second)),
                "n_alternate_only": len(set(second) - set(first)),
                "bit_identical_on_shared": True,
            }
        return out

    def _drop_log(
        self,
        *,
        n_rows: int,
        shared_symbols: Sequence[str],
        unreferenced: Mapping[str, Sequence[str]],
        unstressed_only: Mapping[str, Sequence[str]],
        kept_proteins: Mapping[str, int],
        accounting: Sequence[CalledVariantAccounting],
    ) -> DropLog:
        """Every arm of the proteome deposit and every dropped protein key, counted."""
        strains = list(PROTEOME_RECORD_SHEETS)
        return DropLog(
            dataset=self.name,
            source_rows=n_rows,
            reference_rows=[
                f"{strain} in M9 + 4 g/L glucose (its own baseline)"
                for strain in strains
            ],
            candidate_records=len(strains),
            kept_records=len(strains),
            dropped_records=0,
            rules=[
                DropRule(
                    rule="production_medium_has_no_media_library_entry",
                    scope="arm",
                    description="the pIY670 production cultures are M9 at 20 g/L glucose "
                    "with kanamycin and 2 g/L arabinose; MEDIA_LIBRARY carries Lim "
                    "2025's M9 at its default 4 g/L glucose only, and media.py is a "
                    "value-surface file this branch does not edit. Both arms of all "
                    "three timepoints are left out; the needed addition is in the PR",
                    n_items=6,
                    items=[
                        f"{sheet} ({strain})"
                        for sheet in SHEETS_PIY670
                        for strain in ("IPL400_pIY670", "A10F63I1_pIY670")
                    ],
                ),
                DropRule(
                    rule="unstressed_arm_is_the_reference_not_a_record",
                    scope="arm",
                    description="each strain in M9 + 4 g/L glucose is its own record's "
                    "phenotype_reference, which is where that strain's unstressed "
                    "abundances are stored",
                    n_items=len(strains),
                    items=[f"{strain} M9G" for strain in strains],
                ),
                DropRule(
                    rule="gene_symbol_filed_under_two_paralogous_loci",
                    scope="protein_key",
                    description="the sheets give one protein group's mean and SD to both "
                    "loci of three symbols (Ubid, Pyrc, Dapa) under distinct UniProt "
                    "accessions, and those rows are exactly the ones whose released t "
                    "statistic does not reproduce at n = 3; the statistics cannot be "
                    "attributed to either paralog. Measured: the same six loci on all "
                    "four consumed sheets",
                    n_items=len(shared_symbols),
                    items=list(shared_symbols),
                ),
                DropRule(
                    rule="no_key_matched_reference_abundance",
                    scope="protein_key",
                    description="the locus is quantified under isoprenol but not in that "
                    "strain's own unstressed arm, so the record would carry an abundance "
                    "with no reference value (the seven the G+4IP sheet title-cases as "
                    "'Pp_0002')",
                    n_items=sum(len(keys) for keys in unreferenced.values()),
                    items=[
                        f"{strain}: {list(keys)}"
                        for strain, keys in sorted(unreferenced.items())
                        if keys
                    ],
                ),
                DropRule(
                    rule="no_key_matched_isoprenol_abundance",
                    scope="protein_key",
                    description="the mirror case: the locus is quantified in the "
                    "strain's unstressed arm but not under isoprenol, so the record's "
                    "phenotype would have no value for a reference key. Measured: 0 for "
                    "IPL400 and A10_F63_I1, 29 for A12_F53_I1, whose G+4IP sheet carries "
                    "2,338 rows against the 2,367 of its M9G sheet",
                    n_items=sum(len(keys) for keys in unstressed_only.values()),
                    items=[
                        f"{strain}: {list(keys)}"
                        for strain, keys in sorted(unstressed_only.items())
                        if keys
                    ],
                ),
                DropRule(
                    rule="call_restates_a_designed_deletion",
                    scope="perturbation",
                    description="a called variant whose resolved locus the isolate's "
                    "parent already carries as a designed BacterialDeletionPerturbation "
                    "is dropped from its genotype, so absence has one encoding",
                    n_items=sum(
                        row.n_restating_a_designed_deletion for row in accounting
                    ),
                    items=[
                        f"{row.strain} ({row.clone_column}): "
                        f"{row.n_restating_a_designed_deletion} of "
                        f"{row.n_candidate_perturbations} restate "
                        f"{list(row.restated_loci)}"
                        for row in accounting
                    ],
                ),
            ],
            notes=[
                f"protein keys stored per record (phenotype and reference, key-matched "
                f"by construction): {dict(sorted(kept_proteins.items()))}",
                "the released log2 values are stored verbatim; nothing is exponentiated, "
                "imputed or rescaled",
                "the four comparison sheets export ONE IPL400 measurement, asserted "
                "bit-identical on every shared locus, so IPL400 becomes one record and "
                "is never averaged across the four exports",
                "each record compares one strain's isoprenol arm to its OWN unstressed "
                "arm, which is what the sheets' per-arm absolute log2 Top3 means support; "
                "the sheets' own evolved-vs-IPL400 ratio is not stored, because the "
                "difference of the two stored means IS it",
                "every record's genome_reference carries the IPL400 "
                "BacterialStrainBackground (seven full_deletion alleles plus PP_2676 as "
                "a partial_deletion); the same deletions also ride each genotype as the "
                "ML-facing edits, which is how the de Siqueira 2025 loader types its PT "
                "strain. For the two evolved isolates that background is a FLOOR on the "
                "reference arm's genomic content, because a called variant cannot be a "
                "BacterialBackgroundAllele; preprocess/genotype_gaps.json states it",
                "the called variants of each evolved isolate ARE written, on the "
                "perturbation axis: "
                + "; ".join(
                    f"{row.strain} {row.n_written} "
                    f"({row.n_sequence_variants} in a locus, "
                    f"{row.n_intergenic_variants} keyed on a site, "
                    f"{row.n_span_deletion_loci} span-deletion loci)"
                    for row in accounting
                ),
            ],
        )

    @post_process
    def process(self) -> None:
        """Build the three strains' isoprenol-stress proteome records, write the LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        path = osp.join(self.raw_dir, SI2_XLSX)

        sheets = sorted(
            {
                sheet
                for triple in PROTEOME_RECORD_SHEETS.values()
                for sheet in triple[:2]
            }
        )
        sheet_rows = {sheet: read_proteome_sheet(path, sheet) for sheet in sheets}
        shared_symbols = sorted(
            set().union(
                *(set(shared_symbol_loci(rows)) for rows in sheet_rows.values())
            )
        )
        skip = set(shared_symbols)
        residuals = {
            sheet: assert_sheet_statistics(
                [row for row in rows if row.locus_tag not in skip], sheet=sheet
            )
            for sheet, rows in sheet_rows.items()
        }
        duplicates = self._assert_duplicate_exports(path, shared_symbols)

        sample_key = read_sample_key(path)
        for samples in PROTEOME_SAMPLE_NAMES.values():
            for sample in samples:
                token = sample_key.get(sample)
                if token != PROTEOME_REPLICATE_TOKEN:
                    raise SheetExtractionError(
                        f"{SHEET_SAMPLE_KEY} gives {sample!r} replicates {token!r}, not "
                        f"{PROTEOME_REPLICATE_TOKEN!r}; the stored n_replicates would be "
                        "wrong"
                    )

        column = {"parent": parent_arm, "test": test_arm}
        arms: dict[str, tuple[dict[str, tuple[float, float]], ...]] = {}
        unreferenced: dict[str, list[str]] = {}
        unstressed_only: dict[str, list[str]] = {}
        for strain, (flat, stress, which) in PROTEOME_RECORD_SHEETS.items():
            read = column[which]
            baseline = read(sheet_rows[flat], shared_symbols)
            stressed = read(sheet_rows[stress], shared_symbols)
            unreferenced[strain] = sorted(set(stressed) - set(baseline))
            unstressed_only[strain] = sorted(set(baseline) - set(stressed))
            keys = sorted(set(baseline) & set(stressed))
            if not keys:
                raise SheetExtractionError(
                    f"{self.name}: {strain}'s two arms share no locus"
                )
            arms[strain] = (
                {tag: baseline[tag] for tag in keys},
                {tag: stressed[tag] for tag in keys},
            )

        genome = self._genome()
        all_keys = sorted(set().union(*(set(arm[0]) for arm in arms.values())))
        stored, report = reconcile_locus_tags(
            genome, pd.Series(all_keys), label=self.name
        )
        report.require_resolved(MIN_RESOLVED_FRACTION)
        if sorted(stored.tolist()) != all_keys:
            raise RuntimeError(
                f"{self.name}: reconciliation moved a released locus tag, which would "
                "re-key the abundance map"
            )
        symbols, genotype_report = reconcile_genotype_loci(
            genome, label=f"{self.name} genotype"
        )

        variant_calls = read_variant_calls(path)
        founder_check = assert_founder_columns(variant_calls)
        clone_columns = list(EVOLVED_ISOLATE_CLONE_COLUMNS.values())
        variant_loci, variant_report = resolve_variant_loci(
            genome,
            clone_variant_tokens(variant_calls, clone_columns),
            label=f"{self.name} called-variant loci",
        )
        called: dict[str, list[GenePerturbationType]] = {}
        accounting: list[CalledVariantAccounting] = []
        for isolate, clone_column in EVOLVED_ISOLATE_CLONE_COLUMNS.items():
            perturbations, row_accounting = called_variant_perturbations(
                clone_column, strain=isolate, calls=variant_calls, loci=variant_loci
            )
            called[isolate] = perturbations
            accounting.append(row_accounting)

        drop_log = self._drop_log(
            n_rows=sum(len(rows) for rows in sheet_rows.values()),
            shared_symbols=shared_symbols,
            unreferenced=unreferenced,
            unstressed_only=unstressed_only,
            kept_proteins={strain: len(arm[0]) for strain, arm in arms.items()},
            accounting=accounting,
        )
        drop_log.check()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        # Every record's reference strain is the PARENT's designed content: IPL400's
        # background is what CAN be said of an evolved isolate's background, since a
        # called variant is no BacterialBackgroundAllele (genotype_gaps.json states it).
        genome_reference = assembly_reference(
            "KT2440", background=ipl_background(self.PARENT_STRAIN, symbols)
        )
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, strain in enumerate(
                tqdm(list(PROTEOME_RECORD_SHEETS), desc="lim2025-proteome")
            ):
                baseline, stressed = arms[strain]
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=strain_genotype(
                        self.PARENT_STRAIN if strain == self.PARENT_STRAIN else strain,
                        symbols,
                        called.get(strain, []),
                    ),
                    environment=isoprenol_environment(
                        TALE_DOSE_G_PER_L, PROTEOME_DURATION_GAP
                    ),
                    phenotype=abundance_phenotype(stressed),
                )
                reference = BacterialProteinAbundanceExperimentReference(
                    dataset_name=self.name,
                    genome_reference=genome_reference,
                    environment_reference=unstressed_environment(),
                    phenotype_reference=abundance_phenotype(baseline),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        env.close()
        interned_env.close()

        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drop_log.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            report.model_dump_json(indent=2)
        )
        (out / "genotype_reconciliation.json").write_text(
            genotype_report.model_dump_json(indent=2)
        )
        (out / "genotype_gaps.json").write_text(
            json.dumps([gap.model_dump() for gap in genotype_gaps()], indent=2)
        )
        (out / "called_variant_perturbations.json").write_text(
            json.dumps(
                {
                    "founder_check": founder_check.model_dump(),
                    "evolved_isolate_parent": EVOLVED_ISOLATE_PARENT,
                    "parent_quote": SOURCED_VALUES["evolved_isolate_parent"].quote,
                    "locus_reconciliation": json.loads(
                        variant_report.model_dump_json()
                    ),
                    "per_record": [row.model_dump() for row in accounting],
                },
                indent=2,
            )
        )
        (out / "statistics_back_solve.json").write_text(
            json.dumps(
                {
                    "si2_xlsx_sha256": SI2_XLSX_SHA256,
                    "n_replicates": PROTEOME_N_REPLICATES,
                    "uncertainty_type": "sample_sd (SE = sd / sqrt(n))",
                    "welch_t_worst_residual": residuals,
                    "welch_tolerance": WELCH_TOLERANCE,
                    "duplicate_exports": duplicates,
                    "sample_key": {
                        sample: sample_key[sample]
                        for samples in PROTEOME_SAMPLE_NAMES.values()
                        for sample in samples
                    },
                },
                indent=2,
            )
        )
        pd.DataFrame(
            [
                {
                    "strain": strain,
                    "locus_tag": tag,
                    "log2_mean_unstressed": arms[strain][0][tag][0],
                    "log2_sd_unstressed": arms[strain][0][tag][1],
                    "log2_mean_isoprenol": arms[strain][1][tag][0],
                    "log2_sd_isoprenol": arms[strain][1][tag][1],
                }
                for strain in PROTEOME_RECORD_SHEETS
                for tag in arms[strain][0]
            ]
        ).to_csv(out / "lim2025_proteome.csv", index=False)
        log.info(
            "Lim 2025 proteome: %d records (%s under %g g/L isoprenol, each against its "
            "own unstressed arm) over %s protein keys; %d dropped as shared symbols. "
            "Called variants written %s",
            len(arms),
            ", ".join(PROTEOME_RECORD_SHEETS),
            TALE_DOSE_G_PER_L,
            {strain: len(arm[0]) for strain, arm in arms.items()},
            len(shared_symbols),
            {row.strain: row.n_written for row in accounting},
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of the built LMDBs
# --------------------------------------------------------------------------- #
TOLERANCE_PROVENANCE = Provenance(
    source_uri=f"https://doi.org/{PAPER_DOI} (Supplementary Table 3; Fig. 3A)",
    citation_key=CITATION_KEY,
    sha256=SI1_DOCX_SHA256,
    method="log2 of each starting strain's mean initial TALE growth rate over KT2440's, "
    "plus the Fig. 3A text's 1.6-fold for dPP_3024; reference = KT2440 (0)",
    page="Metab. Eng. 2025, Supplementary Table 3",
)
PROTEOME_PROVENANCE = Provenance(
    source_uri=f"https://doi.org/{PAPER_DOI} (Supplementary Data 1, proteome sheets)",
    citation_key=CITATION_KEY,
    sha256=SI2_XLSX_SHA256,
    method="IPL400's released log2 Top3 mean per locus with SE = log2_std / sqrt(3); "
    "reference = the same strain's unstressed arm",
    page="Metab. Eng. 2025, Supplementary Data 1",
)


#: A perturbation identifier of the site-keyed leaf: ``<replicon>:<1-based position>``.
#: It is deliberately not a gene, so the L4 gene-containment rows below partition the
#: stored identifiers on it rather than counting a site as a missing gene.
_SITE_ID_RE = re.compile(rf"^{re.escape(KT2440_REPLICON)}:\d+$")


def _assert_site_identifiers(
    report: VerificationReport, measured: Iterable[str], *, name: str
) -> set[str]:
    """Split stored identifiers into locus tags and site ids; check the site ids.

    A ``BacterialSiteVariantPerturbation`` is keyed on ``AE015451:<position>`` precisely
    because no locus tag of the pinned assembly holds the call, so asking the gene
    universe to contain one would fail by design. The site ids get their own L4 row --
    the replicon is the one the assembly carries and the position is inside it -- and
    only the locus tags go to the gene-containment row.
    """
    sites = sorted(tag for tag in measured if _SITE_ID_RE.match(tag))
    positions = [int(tag.split(":")[1]) for tag in sites]
    bad = [tag for tag, position in zip(sites, positions, strict=True) if position < 1]
    report.add(
        LevelResult(
            level=Level.L4,
            name=f"site_identifiers_{name}",
            passed=not bad,
            message=f"{len(sites) - len(bad)} of {len(sites)} site-keyed identifiers "
            f"are 1-based positions on {KT2440_REPLICON}",
            details={"n_sites": len(sites), "sites": sites, "malformed": bad},
        )
    )
    return {tag for tag in measured if not _SITE_ID_RE.match(tag)}


def _expected_count(abs_root: str) -> int:
    """The kept-record count the build itself wrote, as the L1 count oracle."""
    drops = DropLog.model_validate_json(
        Path(abs_root, "preprocess", "dropped_records.json").read_text()
    )
    return drops.kept_records


def _audit_sourced_values(report: VerificationReport, data_root: str) -> None:
    """Re-open every ``SourcedValue``'s pinned ``paper.md`` and confirm its quote."""
    library = Path(data_root) / "torchcell-library"
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, library))


def run_tolerance_verification(data_root: str | None = None) -> VerificationReport:
    """Run L0-L4 on the built tolerance LMDB and write its report."""
    from torchcell.verification.environment_response import (
        environment_response_gene_set,
        verify_environment_response_dataset,
    )
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _genome_for_reference,
        _write_report,
        load_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, TOLERANCE_ROOT_REL)
    records = load_records(abs_root)
    references = {
        json.dumps(rec["reference"]["genome_reference"], sort_keys=True)
        for rec in records
    }
    if len(references) != 1:
        raise ValueError(f"{len(references)} distinct genome references; expected 1")
    reference = json.loads(references.pop())
    genome = _genome_for_reference(reference, base)
    universe = _gene_set_for_reference(reference, base)
    # ``sgd_genes`` is deliberately NOT passed: it turns on a row whose message reads
    # "of N measured genes are S288C reference genes" over what are KT2440 locus tags,
    # which would be a mislabelled claim in the report. The containment itself is
    # asserted below against the pinned assembly's own gene set, and raising that
    # message to name the reference it was given is in the PR body.
    report = verify_environment_response_dataset(
        records,
        dataset_name=IsoprenolToleranceLim2025Dataset.__name__,
        provenance=TOLERANCE_PROVENANCE,
        expected_count=_expected_count(abs_root),
        resolve_gene_name=genome.resolve_gene_name,
    )
    deleted = _assert_site_identifiers(
        report, environment_response_gene_set(records), name="deleted_loci"
    )
    missing = sorted(deleted - universe)
    report.add(
        LevelResult(
            level=Level.L4,
            name="gene_containment_kt2440_locus_tags",
            passed=not missing,
            message=f"{len(deleted) - len(missing)} of {len(deleted)} deleted loci are "
            f"{KT2440_ASSEMBLY_SET} gene rows",
            details={
                "n_deleted": len(deleted),
                "n_universe": len(universe),
                "missing_examples": missing[:20],
            },
        )
    )
    _audit_sourced_values(report, base)
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def run_proteome_verification(data_root: str | None = None) -> VerificationReport:
    """Run L0-L4 on the built proteome LMDB and write its report."""
    from torchcell.verification.protein import protein_gene_set, verify_protein_dataset
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _write_report,
        load_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, PROTEOME_ROOT_REL)
    records = load_records(abs_root)
    references = {
        json.dumps(rec["reference"]["genome_reference"], sort_keys=True)
        for rec in records
    }
    if len(references) != 1:
        raise ValueError(f"{len(references)} distinct genome references; expected 1")
    universe = _gene_set_for_reference(json.loads(references.pop()), base)
    # ``allow_duplicate_orfs`` because this is not a knockout screen: the three records
    # are three STRAINS of one lineage, so each shares its parent IPL400's seven designed
    # deletions by construction, and the two evolved isolates additionally share the
    # founder calls IPL400 already carried. Record identity is the genotype as a whole,
    # which L1 ``count`` plus the distinct ``Genotype`` of each record carry; one record
    # per deleted ORF was never this dataset's shape.
    report = verify_protein_dataset(
        records,
        dataset_name=ProteomeLim2025Dataset.__name__,
        provenance=PROTEOME_PROVENANCE,
        expected_count=_expected_count(abs_root),
        allow_duplicate_orfs=True,
    )
    for name, measured in (
        (
            "deleted_loci",
            _assert_site_identifiers(
                report, protein_gene_set(records), name="deleted_loci"
            ),
        ),
        (
            "quantified_loci",
            {
                tag
                for rec in records
                for tag in rec["experiment"]["phenotype"]["protein_abundance"]
            },
        ),
    ):
        missing = sorted(measured - universe)
        report.add(
            LevelResult(
                level=Level.L4,
                name=f"gene_containment_kt2440_{name}",
                passed=not missing,
                message=f"{len(measured) - len(missing)} of {len(measured)} {name} are "
                f"{KT2440_ASSEMBLY_SET} gene rows",
                details={
                    "n_measured": len(measured),
                    "n_universe": len(universe),
                    "missing_examples": missing[:20],
                },
            )
        )
    _audit_sourced_values(report, base)
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def run_verification(data_root: str | None = None) -> tuple[VerificationReport, ...]:
    """Run L0-L4 on both built LMDBs."""
    base = data_root or _data_root()
    return (run_tolerance_verification(base), run_proteome_verification(base))


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDBs, or ``verify`` them."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.pputida.lim2025"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser(
        "deposit", help="retrieve (optional) and deposit the raw mirror"
    )
    deposit.add_argument("--download-dir", required=True)
    deposit.add_argument(
        "--retrieve",
        action="store_true",
        help="run the recorded Elsevier retriever into --download-dir first",
    )
    sub.add_parser("build", help="build (or load) both dev-tree LMDBs")
    sub.add_parser("verify", help="run L0-L4 on both built dev-tree LMDBs")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        download = Path(args.download_dir)
        if args.retrieve:
            retrieve_raw_files(download)
        sources: dict[str, str | Path] = {name: download / name for name in DATA_SHA256}
        print(deposit_raw_mirror(sources=sources, data_root=data_root))
        return 0
    if args.command == "build":
        genome = bacterial_genome("pputida", "KT2440", data_root)
        for cls, rel in (
            (IsoprenolToleranceLim2025Dataset, TOLERANCE_ROOT_REL),
            (ProteomeLim2025Dataset, PROTEOME_ROOT_REL),
        ):
            dataset = cls(root=osp.join(data_root, rel), pputida_genome=genome)
            print(f"{cls.__name__}: len = {len(dataset)}")
        return 0
    reports = run_verification(data_root)
    ok = True
    for report in reports:
        print(report.summary())
        ok = ok and report.passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
