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
- :class:`ProteomeLim2025Dataset` -- ``BacterialProteinAbundanceExperiment``, one record:
  the parent IPL400 under 4 g/L isoprenol, 2,361 Top3 log2 protein abundances, referenced
  to its own unstressed proteome.

THE GENOTYPE FINDING: NO CLASS HOLDS A CALLED VARIANT, SO NO EVOLVED CLONE IS LOADED.
An evolved clone's genotype is its parent's genomic content plus the variants breseq
called from its resequencing. Supplementary Data 1 (``si/si2.xlsx``, sheet
``Fig 2B_Mutation List``) releases exactly that: 159 rows, each a position on
``AE015451`` -- the single replicon of the pinned ``pputida_KT2440_ASM756v2`` assembly --
with a mutation type (100 SNP, 47 DEL, 11 INS, 1 SUB), a sequence change (``G->A``,
``+C``, ``D1 bp``), the gene, gene pair or gene run it falls in, and a detail field (87
amino-acid substitutions, 30 coding-span indels, 17 intergenic, 25 blank), crossed against
49 clone columns as 443 per-clone calls (431 at frequency 1, 12 at 0.9).

No perturbation leaf can hold one of those rows honestly, and this was MEASURED, not
assumed. These are the same reasons the de Siqueira 2025 loader (row 14, the other
P. putida evolved-WGS row) found for its own 173 calls, reached independently on these
bytes, and this loader MATCHES its treatment rather than inventing a second
representation:

1. ``SequenceVariantPerturbation`` is the nearest class and it REFUSES the identifier:
   ``SequenceVariantPerturbation(systematic_gene_name="PP_3415", ...)`` raises
   ``Invalid systematic gene name format`` -- it inherits ``GenePerturbation``'s R64 ORF
   regex and has no ``gene_namespace`` field.
2. Its contract is a dereferenceable allele SEQUENCE in an off-graph gene-keyed store
   (``sequence_source`` + ``sequence_ref``). The only sequence this paper deposits is the
   raw reads under BioProject PRJNA1187681; no per-gene allele store exists, so even a
   widened class would carry two None pointers.
3. No class has a slot for what a row RELEASES: replicon, position, reference base,
   alternate base, mutation type, amino-acid change, call frequency.
   ``BacterialBackgroundAllele.deleted_span`` is deletion-only and would lose all of it.
4. ``BacterialBackgroundAllele`` requires a non-optional ``functional: bool``, so the
   unknown functional consequence of a missense SNP cannot be covered by a
   ``ProvenanceGap`` -- a gap must name a field that is ``None``.
5. ``BacterialStrainBackground`` permits ONE allele entry per locus, which refuses the
   8 rows whose locus carries more than one call in a single clone. ``PP_3415`` alone
   carries three distinct variants across the campaign and TWO within A12_F53_I1 (P293S
   at 3,866,001 and V46I at 3,866,742, Supplementary Table 5).
6. ``Genotype.__eq__`` compares the perturbation SET, so a clone written with its
   parent's perturbations only would be genotype-identical to its parent, collapsing 46
   distinct strains onto three identities.
7. TWO SHAPES THIS ROW ADDS beyond row 14's. The 159 rows partition into 123 on one
   locus, 17 intergenic and 19 spanning several loci. An intergenic row's ``Gene`` field
   names the two FLANKING loci and there is no locus to key to -- inventing a neighbour
   is exactly what must not be done. A multi-locus row is a large deletion (up to the 53
   genes of ``PP_3024-PP_5558``) that no gene-keyed perturbation can state at all: it
   needs a span-level carrier, which row 14's point-variant-only table never required.

Every one of the 159 rows is typed into ``preprocess/called_variants.json`` with its own
``blocking_reasons``, the same file and vocabulary row 14 writes, so the additive proposal
in the PR body rests on the real rows. Nothing here works around the absence. Two
consequences are stated rather than typed:

- The 46 evolved isolates and every phenotype keyed to one of them are NOT loaded.
- IPL300 and IPL400 themselves carry called variants the resequencing found and this
  loader cannot write: "the starting strains IPL300 and IPL400 also contained pre-existing
  mutations missing from the reference sequence: PP_4986, gacS, yhjE in IPL300, and then
  IPL400 had the same mutations along with a mutation in PP_4398." The released matrix's
  F0 clones put 6 calls on IPL300 and 8 on IPL400. Their records therefore carry the
  DESIGNED deletions only, which is a floor on their genotype, not the whole of it. This
  is recorded as a ``ProvenanceGap`` is not available for it -- a gap must name a field
  that is None and ``perturbations`` is set -- so it lives in
  ``preprocess/genotype_gaps.json``, in the note and here.

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

COMPOUND. ``isoprenol`` still has no row in ``compound_identity_table.json``, so
``resolved_compound`` returns the name with an ``inchikey`` gap. Its identity,
:data:`ISOPRENOL_INCHIKEY` = ``CPJRRXSHAYUTGL-UHFFFAOYSA-N``, is recorded here and checked
against any row the table gains; curating that row is a separate, human change.

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
from typing import Any, ClassVar
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
    BacterialStrainBackground,
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    ProteinAbundancePhenotype,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainConstruction,
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

#: Identity of the stressor, recorded until ``compound_identity_table.json`` gains a row.
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
        note="the parents' own called variants, which no perturbation class can hold; "
        "their records carry the designed deletions only, a floor on the genotype",
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

    No record is built from it: no perturbation class holds a called base change at a
    coordinate on a pinned assembly (the module docstring states the finding). These
    counts are the counted reason, written to ``preprocess/variant_accounting.json``.
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


class CalledVariant(BaseModel):
    """One released breseq call, with the reason no perturbation class can hold it.

    The same typed row the de Siqueira 2025 loader writes for row 14's calls, so the
    two P. putida evolved-WGS rows state the blockage in one vocabulary. Nothing here
    becomes a record; the file is the counted evidence under the additive proposal.
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
    blocking_reasons: list[str]


#: The measured reasons no class holds one of these calls. Each row carries the subset
#: that applies to it; the first four apply to every row.
BLOCK_NO_BACTERIAL_VARIANT_LEAF = (
    "SequenceVariantPerturbation is the only variant-level leaf and its base validator "
    "admits only S288C ORF names, so a PP_ locus tag is refused (measured: it raises "
    "'Invalid systematic gene name format' on PP_3415)"
)
BLOCK_NO_ALLELE_SEQUENCE = (
    "SequenceVariantPerturbation promises a dereferenceable allele sequence "
    "(sequence_source + sequence_ref); this paper deposited raw reads "
    f"(BioProject {SRA_BIOPROJECT}) and no per-gene allele store"
)
BLOCK_NO_CALL_FIELDS = (
    "no class has a slot for what the row releases: replicon, position, reference and "
    "alternate base, mutation type, amino-acid change and call frequency. "
    "BacterialBackgroundAllele's deleted_span is deletion-only and would lose them"
)
BLOCK_FUNCTIONAL_REQUIRED = (
    "BacterialBackgroundAllele requires a non-optional functional: bool, so the unknown "
    "functional consequence of a called variant cannot be a ProvenanceGap (a gap must "
    "name a field that is None)"
)
BLOCK_GENOTYPE_COLLAPSE = (
    "Genotype.__eq__ compares the perturbation SET, so a clone written with its parent's "
    "perturbations only would be genotype-identical to its parent and the distinct "
    "strains would collapse onto one identity"
)
BLOCK_INTERGENIC = (
    "the call is intergenic: the Gene field names the two flanking loci, so there is no "
    "locus to key a gene-keyed perturbation to, and inventing a neighbour is exactly "
    "what must not be done"
)
BLOCK_MULTI_LOCUS = (
    "the call spans more than one locus (a large deletion), so one gene-keyed "
    "perturbation cannot state it; it needs a span-level carrier"
)
BLOCK_ONE_ALLELE_PER_LOCUS = (
    "BacterialStrainBackground permits one allele entry per locus, and this locus "
    "carries more than one call in a single clone"
)


def read_variant_calls(path: str) -> list[CalledVariant]:
    """Every released call as a typed row with its own blocking reasons."""
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
        reasons = [
            BLOCK_NO_BACTERIAL_VARIANT_LEAF,
            BLOCK_NO_ALLELE_SEQUENCE,
            BLOCK_NO_CALL_FIELDS,
            BLOCK_FUNCTIONAL_REQUIRED,
            BLOCK_GENOTYPE_COLLAPSE,
        ]
        if item["is_intergenic"]:
            reasons.append(BLOCK_INTERGENIC)
        if len(item["loci"]) > 1:
            reasons.append(BLOCK_MULTI_LOCUS)
        if any(
            per_clone_locus[(clone, locus)] > 1
            for clone in item["clones"]
            for locus in item["loci"]
        ):
            reasons.append(BLOCK_ONE_ALLELE_PER_LOCUS)
        out.append(CalledVariant(blocking_reasons=reasons, **item))
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

    With no table row the resolver returns the name with an ``inchikey`` gap; once a row
    exists, its InChIKey must be :data:`ISOPRENOL_INCHIKEY` or the build stops.
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


def strain_genotype(strain: str, symbols: Mapping[str, str]) -> Genotype:
    """The DESIGNED deletions of one writable strain.

    KT2440 WT is the empty genotype (the reference). IPL300 is its six deletions, IPL400
    those plus ``PP_1385``, and ``dPP_3024`` the single reverse-engineered deletion. None
    of them carries the called variants the resequencing found, which no class holds.
    """
    if strain == "KT2440":
        return Genotype(perturbations=[])
    if strain == "IPL300":
        tags: tuple[str, ...] = IPL300_DELETIONS
    elif strain == "IPL400":
        tags = (*IPL300_DELETIONS, IPL400_EXTRA_DELETION)
    else:
        raise ValueError(f"{strain!r} is not a starting strain of this campaign")
    return Genotype(
        perturbations=[deletion(tag, symbols.get(tag, tag)) for tag in sorted(tags)]
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
    """Genomic content a loaded strain carries that no perturbation class can state."""

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
    """The genomic content of the loaded strains that no class can state."""
    preexisting = SOURCED_VALUES["preexisting_parent_mutations"]
    return [
        GenotypeGap(
            strain="IPL300 and IPL400",
            content="the called variants the resequencing found in both parents "
            "(PP_4986, gacS, yhjE; IPL400 adds PP_4398). The released matrix puts 6 "
            "calls on the IPL300 founder clone and 8 on the IPL400 founder clone",
            reason="no perturbation leaf holds a called base change at a coordinate on "
            "a pinned assembly; SequenceVariantPerturbation refuses a PP_ tag and "
            "promises an allele sequence that was never deposited",
            source_quote=preexisting.quote,
        ),
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

    def _drop_log(self, matrix: MutationMatrix, kept: int) -> DropLog:
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
                    "average over a lineage's three LAST flasks, whose genotype is the "
                    "evolved population's: the parent plus the called variants no "
                    "perturbation class holds",
                    n_items=1,
                    items=[f"{_S3_N_ROWS} final growth rates (one per lineage)"],
                ),
                DropRule(
                    rule="evolved_clone_genotype_is_not_representable",
                    scope="strain",
                    description="no perturbation leaf holds a called base change at a "
                    f"coordinate on {KT2440_REPLICON}: "
                    f"{matrix.n_rows} released rows, {matrix.n_calls} per-clone calls, "
                    f"{matrix.n_evolved_columns} evolved clone columns. "
                    f"{matrix.n_intergenic} rows are intergenic (no locus to key to) and "
                    f"{matrix.n_multi_locus} name more than one locus",
                    n_items=matrix.n_evolved_columns,
                    items=[f"founder columns kept out of the evolved count: {founder}"],
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

        genome = self._genome()
        symbols, report = reconcile_genotype_loci(genome, label=self.name)
        log.info(
            "Lim 2025 tolerance: genotype loci %s",
            {status.value: n for status, n in report.status_histogram.items()},
        )

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
                    strain_genotype(arm.strain, symbols),
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
                },
            )
        )

        drop_log = self._drop_log(matrix, len(records))
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
                },
                *rows,
            ]
        ).astype({"record": "Int64"}).to_csv(out / "table_s3.csv", index=False)
        pd.DataFrame([row.model_dump() for row in lineages]).to_csv(
            out / "tale_lineages.csv", index=False
        )
        log.info(
            "Lim 2025 tolerance: %d records (IPL300, IPL400 at %g g/L isoprenol; "
            "d%s at %g g/L), reference KT2440 = log2(1) = 0",
            len(records),
            TALE_DOSE_G_PER_L,
            PP3024_LOCUS,
            PANEL_DOSE_G_PER_L,
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
    """IPL400's Top3 proteome under 4 g/L isoprenol, referenced to its own baseline."""

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
        unreferenced: Sequence[str],
        kept_proteins: int,
    ) -> DropLog:
        """Every arm of the proteome deposit and every dropped protein key, counted."""
        return DropLog(
            dataset=self.name,
            source_rows=n_rows,
            reference_rows=[
                f"{self.PARENT_STRAIN} in M9 + 4 g/L glucose (the baseline)"
            ],
            candidate_records=1,
            kept_records=1,
            dropped_records=0,
            rules=[
                DropRule(
                    rule="arm_is_an_evolved_isolate",
                    scope="arm",
                    description="the evolved-isolate columns of the four comparison "
                    "sheets are keyed to A10_F63_I1 and A12_F53_I1, whose genotype is "
                    "the parent plus called variants no perturbation class holds",
                    n_items=4,
                    items=[
                        "A10_F63_I1 M9G",
                        "A10_F63_I1 M9G+4IP",
                        "A12_F53_I1 M9G",
                        "A12_F53_I1 M9G+4IP",
                    ],
                ),
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
                    description="IPL400 in M9 + 4 g/L glucose is the record's "
                    "phenotype_reference, which is where its 2,361 abundances are stored",
                    n_items=1,
                    items=[f"{self.PARENT_STRAIN} M9G"],
                ),
                DropRule(
                    rule="gene_symbol_filed_under_two_paralogous_loci",
                    scope="protein_key",
                    description="the sheets give one protein group's mean and SD to both "
                    "loci of three symbols (Ubid, Pyrc, Dapa) under distinct UniProt "
                    "accessions, and those rows are exactly the ones whose released t "
                    "statistic does not reproduce at n = 3; the statistics cannot be "
                    "attributed to either paralog",
                    n_items=len(shared_symbols),
                    items=list(shared_symbols),
                ),
                DropRule(
                    rule="no_key_matched_reference_abundance",
                    scope="protein_key",
                    description="the locus is quantified under isoprenol but not in the "
                    "unstressed reference arm, so the record would carry an abundance "
                    "with no reference value (these are the seven the G+4IP sheet "
                    "title-cases as 'Pp_0002')",
                    n_items=len(unreferenced),
                    items=list(unreferenced),
                ),
            ],
            notes=[
                f"{kept_proteins} protein keys are stored on the record and on its "
                "reference, key-matched by construction",
                "the released log2 values are stored verbatim; nothing is exponentiated, "
                "imputed or rescaled",
                "the four comparison sheets export ONE IPL400 measurement, asserted "
                "bit-identical on every shared locus, so it becomes one record",
                "the record's genome_reference carries the full IPL400 "
                "BacterialStrainBackground (seven full_deletion alleles plus PP_2676 as "
                "a partial_deletion), because this comparison's reference strain IS "
                "IPL400; the same deletions also ride the genotype as the ML-facing "
                "edits, which is how the de Siqueira 2025 loader types its PT strain",
            ],
        )

    @post_process
    def process(self) -> None:
        """Build IPL400's isoprenol-stress proteome record and write the LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        path = osp.join(self.raw_dir, SI2_XLSX)

        baseline_rows = read_proteome_sheet(path, SHEET_PROTEOME_M9G)
        stressed_rows = read_proteome_sheet(path, SHEET_PROTEOME_IPL)
        shared_symbols = sorted(
            set(shared_symbol_loci(baseline_rows))
            | set(shared_symbol_loci(stressed_rows))
        )
        residuals = {
            SHEET_PROTEOME_M9G: assert_sheet_statistics(
                [
                    row
                    for row in baseline_rows
                    if row.locus_tag not in set(shared_symbols)
                ],
                sheet=SHEET_PROTEOME_M9G,
            ),
            SHEET_PROTEOME_IPL: assert_sheet_statistics(
                [
                    row
                    for row in stressed_rows
                    if row.locus_tag not in set(shared_symbols)
                ],
                sheet=SHEET_PROTEOME_IPL,
            ),
        }
        duplicates = self._assert_duplicate_exports(path, shared_symbols)

        sample_key = read_sample_key(path)
        for sample in (SAMPLE_KEY_M9G, SAMPLE_KEY_IPL):
            token = sample_key.get(sample)
            if token != PROTEOME_REPLICATE_TOKEN:
                raise SheetExtractionError(
                    f"{SHEET_SAMPLE_KEY} gives {sample!r} replicates {token!r}, not "
                    f"{PROTEOME_REPLICATE_TOKEN!r}; the stored n_replicates would be wrong"
                )

        baseline = parent_arm(baseline_rows, shared_symbols)
        stressed = parent_arm(stressed_rows, shared_symbols)
        unreferenced = sorted(set(stressed) - set(baseline))
        keys = sorted(set(baseline) & set(stressed))
        if not keys:
            raise SheetExtractionError(f"{self.name}: the two arms share no locus")
        baseline = {tag: baseline[tag] for tag in keys}
        stressed = {tag: stressed[tag] for tag in keys}

        genome = self._genome()
        stored, report = reconcile_locus_tags(genome, pd.Series(keys), label=self.name)
        report.require_resolved(MIN_RESOLVED_FRACTION)
        if sorted(stored.tolist()) != keys:
            raise RuntimeError(
                f"{self.name}: reconciliation moved a released locus tag, which would "
                "re-key the abundance map"
            )
        symbols, genotype_report = reconcile_genotype_loci(
            genome, label=f"{self.name} genotype"
        )

        drop_log = self._drop_log(
            n_rows=len(baseline_rows) + len(stressed_rows),
            shared_symbols=shared_symbols,
            unreferenced=unreferenced,
            kept_proteins=len(keys),
        )
        drop_log.check()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        experiment = BacterialProteinAbundanceExperiment(
            dataset_name=self.name,
            genotype=strain_genotype(self.PARENT_STRAIN, symbols),
            environment=isoprenol_environment(TALE_DOSE_G_PER_L, PROTEOME_DURATION_GAP),
            phenotype=abundance_phenotype(stressed),
        )
        reference = BacterialProteinAbundanceExperimentReference(
            dataset_name=self.name,
            genome_reference=assembly_reference(
                "KT2440", background=ipl_background(self.PARENT_STRAIN, symbols)
            ),
            environment_reference=unstressed_environment(),
            phenotype_reference=abundance_phenotype(baseline),
        )
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            txn.put(b"0", self._intern_record(experiment, reference, PUBLICATION, itxn))
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
                        SAMPLE_KEY_M9G: sample_key[SAMPLE_KEY_M9G],
                        SAMPLE_KEY_IPL: sample_key[SAMPLE_KEY_IPL],
                    },
                },
                indent=2,
            )
        )
        pd.DataFrame(
            [
                {
                    "locus_tag": tag,
                    "log2_mean_unstressed": baseline[tag][0],
                    "log2_sd_unstressed": baseline[tag][1],
                    "log2_mean_isoprenol": stressed[tag][0],
                    "log2_sd_isoprenol": stressed[tag][1],
                }
                for tag in keys
            ]
        ).to_csv(out / "ipl400_proteome.csv", index=False)
        log.info(
            "Lim 2025 proteome: 1 record (%s under %g g/L isoprenol) over %d protein "
            "keys; %d dropped as shared symbols, %d as unreferenced",
            self.PARENT_STRAIN,
            TALE_DOSE_G_PER_L,
            len(keys),
            len(shared_symbols),
            len(unreferenced),
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
    deleted = environment_response_gene_set(records)
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
    report = verify_protein_dataset(
        records,
        dataset_name=ProteomeLim2025Dataset.__name__,
        provenance=PROTEOME_PROVENANCE,
        expected_count=_expected_count(abs_root),
    )
    for name, measured in (
        ("deleted_loci", protein_gene_set(records)),
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
