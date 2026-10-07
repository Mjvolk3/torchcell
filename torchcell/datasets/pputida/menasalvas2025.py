# torchcell/datasets/pputida/menasalvas2025
# [[torchcell.datasets.pputida.menasalvas2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/menasalvas2025
# Test file: tests/torchcell/datasets/pputida/test_menasalvas2025.py
r"""Menasalvas 2025 biosensor-coupled CRISPRi selection in P. putida KT2440.

Menasalvas et al. 2025 (Sci Adv 11, eady2677; doi:10.1126/sciadv.ady2677; PMID
41134890) built an isoprenol biosensor, coupled it to growth by replacing its mCherry
reporter with ``pyrF``, and ran a pooled dCpf1 CRISPRi library through two rounds of
that selection on an isoprenol-producing KT2440 chassis. This module serves the one
per-gene readout the paper RELEASES:
:class:`IsoprenolSelectionMenasalvas2025Dataset`, a
``BacterialEnvironmentResponseExperiment`` per enriched knockdown target of
Supplementary Tables 1 and 2.

THE TITERS ARE NOT A RELEASED COLUMN, AND THIS LOADER DOES NOT INVENT ONE. The paper's
engineering arm is an isoprenol titer measured by GC-FID, and it is the readout a
production-campaign loader would want. Every per-strain titer in this paper lives in a
PLOTTED FIGURE PANEL (Figs. 4D, 4G, 6A, 6B and figs. S13, S14, S19, S20) and in no
table or data file. Measured on the deposit: Supplementary Data 1 carries metabolite
concentrations, the designed gRNA library, its read distribution, the lost-guide list,
the WGS polymorphisms and a ShinyGO enrichment; Supplementary Data 2 carries five
proteomics sheets; Data 3-5 carry AlphaFold output, a homolog co-occurrence analysis
and flow-cytometry raw files. None is a titer table. The only per-strain titers in the
mirrored bytes are prose, and all but one are approximations or ranges over several
strains ("produced up to $200 \mathrm{mg/}$ liter", "only produced $\sim 25 \mathrm{mg}$
/liter", "increased titers from 150 to $250 \mathrm{mg}$ /liter", "exceeded 850 to
$900 \mathrm{mg},$ 'liter ... in isolates TEAM-3185 and TEAM-3174"), so reading a
per-strain number off them would be fabrication, not sourcing. No
``ProductTiterPhenotype`` is therefore written, and the gap is carried in the raw
mirror's ``si_expected``, in this docstring and in the note.

A BIOSENSOR READOUT IS NOT A TITER, AND THE PHENOTYPE SAYS SO. What IS released per
gene is a CALL: the gene's guide was enriched in a pooled growth-coupled selection
where growth, not product, is the measured quantity. That is an environment response
with a qualitative outcome and no number, so the record is an
``EnvironmentResponsePhenotype`` with ``measurement_type=categorical``,
``assay_type=biosensor_readout``, ``category=ResponseCategory.enhanced`` (the shared
axis member whose definition is "measurably better than the reference ... a biosensor
signal above control") and ``category_label="enriched"``, the source's own word.
``environment_response`` is ``None`` with a typed ``ProvenanceGap``: the per-guide read
counts behind the call were never released, so there is no enrichment score to store.
This is the one-sided hit-list shape ``ResponseCategory`` already serves for
Auesukaree 2009's listed stress-sensitive mutants, used here with the opposite sign.

THE HOST, AND WHY ITS CHASSIS IS NOT ASSERTED. Every record's genotype carries the
three things the paper states for EVERY selection host: the native ``pyrF`` deletion
(``PP_1815``, from the strain table's "Pp KT2440 ΔPP_1815/pyrF"), the five integrated
isoprenol-pathway genes, and the plasmid-borne ``PpedF``-``pyrF`` reporter whose
``TERTU_1389`` open reading frame the Supplementary Methods name. What the paper does
NOT state is which strain each round used: "The pooled CRISPRi-∆pyrF selection regime
was applied in two sequential rounds using four $P _ { P } { _ { e d F } }$ -RBSpyrF
variants and four different strains, each with varied base isoprenol titers and
isoprenol activation thresholds". Four unnamed producer strains with four reporter
variants is not a genotype, and none of them appears in Supplementary Table 4, which
lists ``TEAM-862`` (a non-producing ΔpyrF strain) and the producer lineage but no
producing ΔpyrF strain. So ``AssemblyReferenceGenome.background`` is ``None``: the
reference is plain KT2440 and every stated edit is a perturbation. The chassis
elements that distinguish TEAM-2595 from TEAM-2777 (the ``PJ23100``-``PP_2666,PP_2665``
biosensor sensitization, the second ``PpedF``-RBS-mCherry copy, ``ΔPP_2664``,
``ΔPP_2675``, ``PJ23119``-``PP_1697``) are therefore absent from the records. They are
missing ROWS, not missing fields, so they are not a ``ProvenanceGap``; they are said
here, in ``preprocess/build_accounting.json`` and in the note.

WHICH LEAF EACH ENGINEERED CHANGE MAPS TO. The paper's campaign uses all four
perturbation modes the bacterial ontology types, and the strain table states them as
one string per strain. The mapping, settled once:

- a knockout (``ΔPP_2675``, ``ΔPP_2664``, ``ΔPP_1815/pyrF``, and the 15 validation
  deletions of Supplementary Table 4) -> ``BacterialDeletionPerturbation``;
- a knockdown guide (a dCpf1 CRISPRi target of the pooled library) ->
  ``BacterialCrisprInterferencePerturbation``, with ``n_guides=3`` from "three unique
  gRNAs per gene" and ``guide_sequence=None`` because the enriched spacer per gene was
  not released;
- a promoter change (``PJ23100``-``PP_2666,PP_2665``, ``PJ23119``-``PP_1697``) ->
  ``PromoterReplacementPerturbation``;
- an integrated pathway (``PP_5322intergenic::Pcv-mvaS,mvaE`` and
  ``PP_0871intergenic::Ptrc-MKmm,PMDHKQ,aphA``) -> ``HeterologousPathwayPerturbation``.

Only the first, second and fourth appear in THIS dataset's records, because the
promoter changes belong to the unstated chassis above.

THE PATHWAY IS THE SAME pIY670 CASSETTE CARRUTHERS 2025 CARRIES, INTEGRATED. The
plasmid table states pIY670 as "araC-pBAD-mvaS,mvaE ptrc-MKmm,PMDHKQ,aphARK2 kanR" and
the two integration vectors as "PP_5322intergenic::Pcv-mvaS,mvaE kanR sacB(integration
allelic exchange vector)" and "PP_0871intergenic::Ptrc-MKmm,PMDHKQ,aphA gntRsacB
(integration allelic exchange vector)". The five part tokens are therefore ``mvaS``,
``mvaE``, ``MKmm``, ``PMDHKQ`` and ``aphA``, stored verbatim, with
``localization="chromosomal_integration"`` and the integration locus each one's own
vector names -- this paper moved the pathway off the plasmid, which is the one
difference from the Carruthers records. ``source_organism`` is a typed
``ProvenanceGap``: Menasalvas defers the pathway's origin to reference 22 (Banerjee et
al.) and reference 24 (Kang et al.), neither of which is mirrored, and the three
``mvaS`` HOMOLOGS whose organisms this paper DOES name (Enterococcus faecalis,
Silicibacter pomeroyi, Staphylococcus aureus) are extra copies in the final producer
strains, not the integrated pathway's own ``mvaS``. Inferring "Mm" from ``MKmm`` is
exactly the suffix read Carruthers refused without independent evidence.

THE SELECTION TEMPERATURE IS A TYPED ABSENCE. The paper states 30 C for the conjugation
spot on LB agar, for petri-dish culture and for the production assays, and does not
state it for the 24-deep-well M9 selection plate. ``Environment.temperature`` is
``None`` with a ``ProvenanceGap`` rather than 30 C carried over from a neighboring
step.

RECORD COUNT, AND WHAT A ROW IS. 58 records: 28 from Supplementary Table 1 (the
lower-threshold first round) and 30 from Supplementary Table 2 (the higher-threshold
second round), matching the Methods' "choosing 28 targets for the first enrichment
analysis and 30 for the second analysis". One row is one SELECTED TARGET, which is
usually one locus tag but is an operon in one case: Table 1's "phaAZC-II /
PP_5003-PP_5005" is the three tags ``PP_5003``, ``PP_5004`` and ``PP_5005``, expanded
by integer enumeration of the stated endpoints and then checked against the annotation.
That record carries three ``BacterialCrisprInterferencePerturbation`` entries, one
guide repressing a polycistron, so the dataset holds 58 records over 60 distinct guide
targets. The two rounds share no target ("Analysis of gRNAs by sequencing showed no
overlap with the first set, as expected"), which the build asserts.

THE SELECTED SET IS NOT THE FULL ENRICHMENT, AND THE TABLES ARE A SUBSET BY DESIGN.
The released tables are the targets the authors PICKED from the enrichment for
follow-up: "Candidate genes from the gRNA enrichment were first grouped by function
using HMMer and COG to identify nonredundant cellular processes. t random, we picked
several from ach category to design new gRNA plasmids and recombineering oligos,
choosing 28 targets for the first enrichment analysis and 30 for the second analysis."
Enrichment is the measurement; the picking is curation. Every record is a gene whose
guide met the stated criterion ("particular gRNA was enriched ${ > } 5$ reads in one
biological replicate"), and no record claims the list is exhaustive. A guide-by-sample
abundance matrix was never released, so the unpicked enriched guides are unrecoverable.

GENE SYMBOLS COME FROM THE ANNOTATION, NOT FROM THE TABLE. The tables spell a gene
several ways ("sotB / PP_2428", "cmpX PP_2087", "hisQ |PP_4485", "relA PP_1656"), and
one spelling disagrees with the assembly: the source's "phaAZC-II" implies
``PP_5004`` is ``phaZ`` while GCA_000007565.2 annotates it ``phaB``.
``perturbed_gene_name`` is therefore the annotation's own symbol for the locus (falling
back to the tag where it has none) so one gene carries one spelling across datasets,
and the table's verbatim ``Gene``, ``Function`` and ``FunctionalCategory`` cells are
kept in ``preprocess/selected_targets.csv``.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data.experiment_dataset import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import M9_NREL_MOPS_MENASALVAS2025
from torchcell.datamodels.schema import (
    AssayType,
    AssemblyReferenceGenome,
    BacterialCrisprInterferencePerturbation,
    BacterialDeletionPerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialGeneNamespace,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    HeterologousPathwayPerturbation,
    MeasurementType,
    PhysicalFactor,
    Publication,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_SI_OCR,
    ROLE_SI_PDF,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.base import GeneNameStatus
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

# --------------------------------------------------------------------------- #
# Paper identity and the pinned artifacts
# --------------------------------------------------------------------------- #
DOI = "10.1126/sciadv.ady2677"
PMID = "41134890"
PMCID = "PMC12551699"
TITLE = (
    "Biosensor-driven strain engineering reveals key cellular processes for "
    "maximizing isoprenol production in Pseudomonas putida"
)
CITATION_KEY = "menasalvasBiosensordrivenStrainEngineering2025"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: PMC Article Datasets bucket prefix for this article's supplementary files.
PMC_PREFIX = f"{PMCID}.1"

#: The publisher Supplementary Information PDF: the released bytes the tables live in.
SI_PDF_FILENAME = "sciadv.ady2677_sm.pdf"
SI_PDF_REL = f"si/{SI_PDF_FILENAME}"
SI_PDF_SHA256 = "6d82d568b307655878306c236c3d6c04d4776ef73d9b8551d1b8181f2f0be02a"

#: The MinerU OCR of that PDF: the markdown this loader PARSES, pinned in its own right.
SI1_MD_FILENAME = "si1.md"
SI1_MD_REL = f"si/{SI1_MD_FILENAME}"
SI1_MD_SHA256 = "2afa42609d20500f80d5edbddf0bb41b88b4e7f1ea0abefb40fa4b969e4e5e8e"
MINERU_VERSION = "2.7.6"

#: The article OCR in the torchcell-library mirror, quoted but never parsed here.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "d14536948af5ba67d52362ae71fb804a2fac03a1f7ab152817a01df3aa92d080"

SI_RETRIEVED_AT = "2026-10-07"

#: Isoprenol's InChIKey, recorded for the curation step rather than used here. That step
#: has since happened (the table's isoprenol row, PubChem CID 12988), so
#: ``resolved_compound("isoprenol")`` now returns the full identity rather than a typed
#: gap on ``inchikey``; this constant stays as the cross-check that the row is the
#: molecule this paper means. THIS loader never calls the resolver -- a categorical
#: biosensor call names no product -- but a future titer loader for this paper would.
ISOPRENOL_INCHIKEY = "CPJRRXSHAYUTGL-UHFFFAOYSA-N"

#: The two data deposits the paper names, neither holding a titer table.
DRYAD_DOI = "10.5061/dryad.sbcc2frjq"
ZENODO_DOI = "10.5281/zenodo.17155686"
PRIDE_ACCESSION = "PXD061547"
BIOPROJECT_ACCESSION = "PRJNA1226229"

# --------------------------------------------------------------------------- #
# Assembly, namespace and parsing
# --------------------------------------------------------------------------- #
KT2440_NAMESPACE: BacterialGeneNamespace = "pputida_kt2440_locus_tag"
KT2440_STRAIN: Literal["KT2440"] = "KT2440"

#: The two Supplementary Tables this loader reads, and the round each one is.
ROUND_TABLES: tuple[tuple[int, str], ...] = ((1, "round-1"), (2, "round-2"))
#: ``screen_id`` per round, from each table's own title.
SCREEN_IDS: dict[int, str] = {
    1: "round-1-lower-isoprenol-threshold",
    2: "round-2-higher-isoprenol-threshold",
}
#: The ``Gene`` / ``Function`` / ``FunctionalCategory`` header both tables carry.
TABLE_HEADER: tuple[str, ...] = ("Gene", "Function", "FunctionalCategory")
#: Target counts the Methods state per round; the build refuses any other count.
EXPECTED_ROWS: dict[int, int] = {1: 28, 2: 30}
#: Total records == the two rounds' rows; one row is one selected target.
EXPECTED_RECORDS = sum(EXPECTED_ROWS.values())

LOCUS_TAG_RE = re.compile(r"PP_\d{4}")
#: ``PP_5003-PP_5005``: an operon written as its endpoint tags.
LOCUS_RANGE_RE = re.compile(r"PP_(\d{4})\s*-\s*PP_(\d{4})")
#: Separators the ``Gene`` cell uses between a symbol and its tag(s).
SYMBOL_STRIP = " /|,;:-"

#: The constant host background every record carries, excluded from the verifier's
#: strain and gene keys because it is identical in all 58 records.
HOST_BACKGROUND_NAMES: frozenset[str] = frozenset(
    {"PP_1815", "TERTU_1389", "mvaS", "mvaE", "MKmm", "PMDHKQ", "aphA"}
)

# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
_PAGE_METHODS_MEDIA = "Materials and Methods, Bacterial growth media and cultivation"
_PAGE_METHODS_SELECTION = (
    "Materials and Methods, Analysis of enriched gRNAs from the library under PpedF-pyrF"
    " selection"
)
_PAGE_METHODS_LIBRARY = (
    "Materials and Methods, gRNA library design, construction, and validation"
)
_PAGE_METHODS_STATS = "Materials and Methods, Statistical analysis"
_PAGE_METHODS_GC = (
    "Materials and Methods, Quantification of isoprenol by gas chromatography"
)
_PAGE_RESULTS_SELECTION = (
    "Results, CRISPRi-based selection using a growth-coupled isoprenol biosensor"
)
_PAGE_RESULTS_INTEGRATION = (
    "Results, Deploying a unified biosensor-producer strain to identify nonpathway "
    "bottlenecks"
)
_PAGE_SI_METHODS = "Supplementary Methods, Molecular biology"
_PAGE_SI_TABLE4 = "Supplementary Table 4. Strains Used in This Study"
_PAGE_SI_TABLE5 = "Supplementary Table 5. Plasmids Used in This Study"
_PAGE_SI_TABLES12 = "Supplementary Tables 1 and 2 (enriched gRNAs, rounds 1 and 2)"


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
            source_uri=SI1_MD_REL,
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method=(
                f"MinerU {MINERU_VERSION} OCR of the Supplementary Information PDF "
                "(raw mirror)"
            ),
            page=page,
        ),
    )


_Q_GROWTH_COUPLE = (
    "we coupled higher concentrations of isoprenol to cell growth by replacing mCherry "
    "with pyrF, a URA3 homolog from Teredinibacter turnerae T7901, under the pedF "
    "promoter linking activation of the biosensor to growth (instead of fluorescence)"
)
_Q_LIBRARY = (
    "The dCpf1/gRNA CRISPRi library contains \\~16,500 gRNAs targeting nearly all "
    "genes with three unique gRNAs per gene (data S1-2)"
)
_Q_GUIDES_PER_GENE = (
    "Three gRNAs targeting upstream of each of the 5591 coding sequences in the "
    "$P .$ putida KT2440 were designed using gRNASeqRET (81) (data S1-2)"
)
_Q_SELECTION_CULTURE = (
    "used to inoculate $1 . 5 \\mathrm { m l }$ of M9 medium kanamycin with or without "
    "$1 \\mu \\mathrm { M }$ crystal violet in 24-deep-well plates with four replicates "
    "from each conjugation"
)
_Q_GROW_24H = (
    "Samples were grown for 24 hours at which point we examined the cultures for growth"
)
_Q_CRITERIA = (
    "implicat genes were elected on the basis o thefollowing criteri: ( particular "
    "gRNA was enriched ${ > } 5$ reads in one biological replicate), (ii) if there are "
    "multiple gRNAs targeting the same gene, iii) gRNAs target genes functionally "
    "related (i.e., generation of a specific process) or targets in the same operon, "
    "and (iv) the repeated occurrence of gRNAs or gene targets across multiple "
    "replicates"
)
_Q_N_TARGETS = (
    "choosing 28 targets for the first enrichment analysis and 30 for the second "
    "analysis"
)
_Q_NO_OVERLAP = (
    "Analysis of gRNAs by sequencing showed no overlap with the first set, as expected "
    "(Fig. 5 and table S2)"
)
_Q_CRYSTAL_VIOLET = (
    "Crystal violet (Sigma-Aldrich, product no. 61135) was used at a concentration of "
    "$1 0 0 0 \\mathrm { n M }$ $( 1 ~ \\mu \\mathrm { M } )$ to induce production of "
    "the integrated isoprenol pathway"
)
_Q_M9 = (
    "At the 1X working concentration, M9 medium contains 47.9 mM "
    "${ \\mathrm { N a } } _ { 2 } { \\mathrm { H P O } } _ { 4 } ,$ 22 mM "
    "${ \\mathrm { K H } } _ { 2 } { \\mathrm { P O } } _ { 4 }$ 8.56 mM NaCl, 2 mM "
    "$\\mathrm { M g S O _ { 4 } , }$ $1 0 0 ~ \\mu \\mathrm { M } \\mathrm { \\ C a C l } _ { 2 }$"
    " with 1X trace metal solution (catalog no. T1001, Teknova Inc., Hollister, CA), "
    "$2 \\%$ glucose, $7 0 ~ \\mathrm { m M }$ "
    "$\\mathrm { ( N H _ { 4 } ) _ { 2 } S O _ { 4 } } ,$ and $3 0 \\mathrm { m M }$ "
    "Mops (Sigma-Aldrich, catalog no. M1254) adjusted to a $\\mathrm { p H }$ of 7.0"
)
_Q_KANAMYCIN = (
    "Kanamycin $ { 5 0 }  { \\mu \\mathrm { g / m L } } )$ ) or gentamicin "
    "$( 3 0 \\mu \\mathrm { g / m L }$ ) (Teknova Inc, Hollister, CA) was added to the "
    "appropriate medium as indicated for experiments requiring the selection of "
    "plasmids for both $E .$ . coli and $P .$ . putida"
)
_Q_PYRF_STRAIN = "Pp KT2440 ΔPP_1815/pyrF"
_Q_PYRF_REPORTER = (
    "The open reading frame for pyrF homolog purF (orotidine 5'phosphate decarboxylase, "
    "TERTU_1389, referred to simply as pyrF) and 300 bp downstream sequence from "
    "Teredinibacter turnerae T7901 was synthesized by Genewiz Ltd and assembled into a "
    "RSF1010 plasmid backbone immediately downstream of the pedF promoter sequence for "
    "cross-species pyrF complementation"
)
_Q_PATHWAY_SPLIT = (
    "We then replaced the original arabinose-inducible BAD promoter in the isoprenol "
    "pathway with a more economical crystal violetinducible promoter, PJEx1 (45), and "
    "split the pathway across two integration loci (Fig. 4B)"
)
_Q_PTE744 = (
    "PP_5322intergenic::Pcv-mvaS,mvaE kanR sacB(integration allelic exchange vector)"
)
_Q_PTE745 = (
    "PP_0871intergenic::Ptrc-MKmm,PMDHKQ,aphA gntRsacB (integration allelic exchange "
    "vector)"
)
_Q_PIY670 = "araC-pBAD-mvaS,mvaE ptrc-MKmm,PMDHKQ,aphARK2 kanR"
_Q_FOUR_STRAINS = (
    "The pooled CRISPRi-∆pyrF selection regime was applied in two sequential rounds "
    "using four $P _ { P } { _ { e d F } }$ -RBSpyrF variants and four different "
    "strains, each with varied base isoprenol titers and isoprenol activation "
    "thresholds (Materials and Methods)"
)
_Q_PICKED = (
    "Candidate genes from the gRNA enrichment were first grouped by function using "
    "HMMer and COG to identify nonredundant cellular processes. t random, we picked "
    "several from ach category to design new gRNA plasmids and recombineering oligos, "
    "choosing 28 targets for the first enrichment analysis and 30 for the second "
    "analysis"
)
_Q_SD = (
    "the error bars indicating the SD from the mean for isoprenol titer reflect are "
    "calculated using all data points shown in the figure panel"
)
_Q_GC_FID = (
    "Isoprenol quantification was performed using the gas chromatography-flame "
    "ionization detection (GC-FID 8890, Agilent Technologies, USA)"
)
_Q_TABLE1_TITLE = (
    "Supplementary Table 1. Lower pedF-RBS-pyrF Isoprenol Threshold: Enriched gRNAs."
)
_Q_TABLE2_TITLE = (
    "Supplementary Table 2. Second Round pedF-RBS-pyrF with Higher Isoprenol "
    "Threshold: Enriched gRNAs."
)
_Q_LB_AGAR_SPOT = (
    "spotted onto solid LB agar media and allowed to incubate overnight at "
    "$3 0 ^ { \\circ } \\mathrm { C }$"
)

READOUT = _paper(
    "biosensor_readout",
    _Q_GROWTH_COUPLE,
    page=_PAGE_RESULTS_SELECTION,
    note="growth under PpedF-driven pyrF complementation is the measured quantity; the "
    "isoprenol titer it proxies is NOT measured per clone in this assay",
)
EFFECTOR = _paper(
    "dCpf1/dCas12a",
    _Q_LIBRARY,
    page=_PAGE_RESULTS_SELECTION,
    note="the pooled library's dead-Cas effector, written as the paper writes it",
)
N_GUIDES = _paper(
    3,
    _Q_GUIDES_PER_GENE,
    page=_PAGE_METHODS_LIBRARY,
    note="the DESIGNED guides per gene; which of the three was enriched is not released",
)
N_REPLICATES = _paper(
    4,
    _Q_SELECTION_CULTURE,
    page=_PAGE_METHODS_SELECTION,
    note="four replicate selection cultures per conjugation; the enrichment criterion "
    f"calls them biological replicates ('{_Q_CRITERIA}')",
)
SAMPLE_UNIT = _paper(
    "biological_replicate",
    _Q_CRITERIA,
    page=_PAGE_METHODS_SELECTION,
    note="the criterion's own words for one replicate of the selection",
)
CALL_CRITERION = _paper(
    "enriched",
    _Q_CRITERIA,
    page=_PAGE_METHODS_SELECTION,
    note="the released call: a gene whose guide was enriched in the growth-coupled "
    "selection, with no per-guide read count released to score it",
)
N_TARGETS = _paper(
    dict(EXPECTED_ROWS),
    _Q_N_TARGETS,
    page=_PAGE_METHODS_SELECTION,
    note="28 rows in Supplementary Table 1 and 30 in Supplementary Table 2, which the "
    "build checks against the parsed tables",
)
ROUND_DISJOINT = _paper(
    True,
    _Q_NO_OVERLAP,
    page=_PAGE_RESULTS_SELECTION,
    note="asserted at build time over the parsed tables",
)
SELECTED_SUBSET = _paper(
    "picked_from_enrichment",
    _Q_PICKED,
    page=_PAGE_METHODS_SELECTION,
    note="the tables are the targets picked per functional category, not the full "
    "enriched set; no guide-by-sample matrix was released",
)
HOST_UNNAMED = _paper(
    "four unnamed producing dpyrF strains",
    _Q_FOUR_STRAINS,
    page=_PAGE_RESULTS_SELECTION,
    note="why AssemblyReferenceGenome.background is None: no selection host is named, "
    "and none appears in Supplementary Table 4",
)
MEDIUM = _paper(
    "M9 medium kanamycin",
    _Q_SELECTION_CULTURE,
    page=_PAGE_METHODS_SELECTION,
    note=f"the recipe is the Methods' NREL M9 ('{_Q_M9}'), served as "
    "MEDIA_LIBRARY['M9_NREL_MOPS_MENASALVAS2025']",
)
DURATION_HOURS = _paper(
    24.0,
    _Q_GROW_24H,
    page=_PAGE_METHODS_SELECTION,
    note="the selection plate is read at 24 h; the gRNA amplicon is prepared from it",
)
INDUCER_UM = _paper(
    1.0,
    _Q_CRYSTAL_VIOLET,
    page=_PAGE_METHODS_MEDIA,
    note="crystal violet induces the integrated isoprenol pathway, which is what makes "
    "the selection arm isoprenol-producing; the control arm omits it",
)
KANAMYCIN_UG_PER_ML = _si(
    50.0,
    _Q_KANAMYCIN,
    page=_PAGE_SI_METHODS,
    note="the dose the Supplementary Methods state for plasmid selection; the selection "
    "Methods name the antibiotic without restating the dose",
)
MEDIUM_PH = _paper(
    7.0,
    _Q_M9,
    page=_PAGE_METHODS_MEDIA,
    note="pH is not a Media field, so it rides as an EnvironmentPhysicalPerturbation",
)
PYRF_DELETION = _si(
    "PP_1815",
    _Q_PYRF_STRAIN,
    page=_PAGE_SI_TABLE4,
    note="the strain table's own locus for pyrF, which GCA_000007565.2 also annotates "
    "as pyrF",
)
PYRF_REPORTER = _si(
    "TERTU_1389",
    _Q_PYRF_REPORTER,
    page=_PAGE_SI_METHODS,
    note="the growth-coupled reporter: a Teredinibacter turnerae pyrF homolog on an "
    "RSF1010 plasmid downstream of the pedF promoter",
)
PATHWAY_NAME = _paper(
    "isoprenol pathway",
    _Q_PATHWAY_SPLIT,
    page=_PAGE_RESULTS_INTEGRATION,
    note="integrated at two loci under the crystal-violet-inducible PJEx1 (Pcv) and "
    "Ptrc promoters",
)
PATHWAY_LOCUS_A = _si(
    "PP_5322intergenic",
    _Q_PTE744,
    page=_PAGE_SI_TABLE5,
    note="integration locus of the Pcv-mvaS,mvaE half",
)
PATHWAY_LOCUS_B = _si(
    "PP_0871intergenic",
    _Q_PTE745,
    page=_PAGE_SI_TABLE5,
    note="integration locus of the Ptrc-MKmm,PMDHKQ,aphA half",
)
PATHWAY_PARTS = _si(
    _Q_PIY670,
    _Q_PIY670,
    page=_PAGE_SI_TABLE5,
    note="pIY670, the plasmid this paper's integrated pathway is derived from; the five "
    "part tokens stored by the loader are its own",
)
TITER_UNCERTAINTY = _paper(
    "sample_sd",
    _Q_SD,
    page=_PAGE_METHODS_STATS,
    note="the uncertainty type of the TITER readout, recorded because that readout is "
    "not released per strain; the selection call carries no dispersion at all",
)
TITER_METHOD = _paper(
    "GC-FID",
    _Q_GC_FID,
    page=_PAGE_METHODS_GC,
    note="how the unreleased per-strain titers were measured",
)
TABLE_TITLES = _si(
    {1: _Q_TABLE1_TITLE, 2: _Q_TABLE2_TITLE},
    f"{_Q_TABLE1_TITLE} {_Q_TABLE2_TITLE}",
    page=_PAGE_SI_TABLES12,
    note="each round's threshold, from its own table title; stored as screen_id",
)
PRESELECTION_ENVIRONMENT = _paper(
    "no crystal violet",
    _Q_SELECTION_CULTURE,
    page=_PAGE_METHODS_SELECTION,
    note="the control arm of the same selection: the same library in the same medium "
    "with the pathway inducer omitted, so no clone can be enriched by isoprenol",
)
CONJUGATION_TEMPERATURE = _paper(
    30.0,
    _Q_LB_AGAR_SPOT,
    page=_PAGE_METHODS_SELECTION,
    note="stated for the conjugation spot on LB agar and NOT for the deep-well M9 "
    "selection plate, which is why Environment.temperature is a typed absence",
)

#: The five integrated pathway genes. ``token`` is the verbatim pIY670 part name,
#: ``promoter`` and ``locus`` the integration vector that carries it.
PATHWAY_GENES: tuple[dict[str, str], ...] = (
    {"token": "mvaS", "promoter": "Pcv", "locus": "PP_5322intergenic"},
    {"token": "mvaE", "promoter": "Pcv", "locus": "PP_5322intergenic"},
    {"token": "MKmm", "promoter": "Ptrc", "locus": "PP_0871intergenic"},
    {"token": "PMDHKQ", "promoter": "Ptrc", "locus": "PP_0871intergenic"},
    {"token": "aphA", "promoter": "Ptrc", "locus": "PP_0871intergenic"},
)

#: The chassis edits this paper's producer lineage carries that THIS dataset's records
#: deliberately do not, because no selection host is named. Reported, never asserted.
UNASSERTED_CHASSIS: tuple[str, ...] = (
    "PJ23100-PP_2666,PP_2665 (promoter_replacement, biosensor sensitization)",
    "PP_5402intergenic::PpedF-RBS-mCherry (the fluorescent biosensor copy)",
    "PP_3159intergenic::PpedF-RBS-mCherry (the second biosensor copy)",
    "dPP_2664 (bacterial_deletion, stationary-phase biosensor activation)",
    "dPP_2675 (bacterial_deletion, blocks isoprenol catabolism)",
    "PJ23119-PP_1697 (promoter_replacement, TEAM-2777 only)",
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/menasalvasBiosensordrivenStrainEngineering2025``."""
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


def _ocr_processing() -> ProcessingRecord:
    """The MinerU run that turned the pinned SI PDF into the pinned markdown."""
    return ProcessingRecord(
        processor="torchcell.literature.ocr.ocr_pdf",
        tool="mineru",
        version=MINERU_VERSION,
        params={
            "backend": "pipeline",
            "lang": "en",
            "method": "auto",
            "dpi": 200,
            "images_dir": "images/si1",
        },
        input_sha256=[SI_PDF_SHA256],
    )


def deposit_raw_mirror(
    *,
    si_pdf_path: str | Path,
    si_md_path: str | Path,
    retrieved_at: str = SI_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror (the SI PDF and its OCR) and its ``manifest.json``.

    Idempotent by sha256: an existing mirror file whose digest matches is left alone
    and a differing one raises rather than being overwritten. Both files are verified
    BEFORE anything is written, so a refusal leaves no partial deposit. The PDF comes
    from the PMC Article Datasets bucket, which is directly scriptable and reproduced
    the pinned digest on ``retrieved_at``; the markdown is a DERIVED artifact of those
    exact bytes, so it carries a ``ProcessingRecord`` naming the MinerU run instead of
    a retrieval.
    """
    root = raw_mirror_dir(data_root)
    verify_sha256(si_pdf_path, SI_PDF_SHA256)
    verify_sha256(si_md_path, SI1_MD_SHA256)
    files: list[ArtifactRecord] = []
    for source, relpath, expected in (
        (si_pdf_path, SI_PDF_REL, SI_PDF_SHA256),
        (si_md_path, SI1_MD_REL, SI1_MD_SHA256),
    ):
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != expected:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(source, dest)
    key = pmc_cloud_key(SI_PDF_FILENAME)
    url = f"https://pmc-oa-opendata.s3.amazonaws.com/{key}"
    files.append(
        ArtifactRecord(
            path=SI_PDF_REL,
            role=ROLE_SI_PDF,
            bytes=(root / SI_PDF_REL).stat().st_size,
            sha256=SI_PDF_SHA256,
            source=url,
            original_filename=SI_PDF_FILENAME,
            retrieval=RetrievalRecord(
                method=RetrievalMethod.pmc_cloud,
                source_url=url,
                retriever="torchcell.literature.retrieve.pmc_cloud_object",
                params={"key": key},
                sha256=SI_PDF_SHA256,
                retrieved_at=retrieved_at,
            ),
        )
    )
    files.append(
        ArtifactRecord(
            path=SI1_MD_REL,
            role=ROLE_SI_OCR,
            bytes=(root / SI1_MD_REL).stat().st_size,
            sha256=SI1_MD_SHA256,
            source="mineru-ocr",
            processing=_ocr_processing(),
        )
    )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=files,
        si_data_sources=[
            url,
            f"https://doi.org/{DRYAD_DOI}",
            f"https://doi.org/{ZENODO_DOI}",
            f"https://www.ebi.ac.uk/pride/archive/projects/{PRIDE_ACCESSION}",
            f"https://www.ncbi.nlm.nih.gov/bioproject/{BIOPROJECT_ACCESSION}",
        ],
        si_expected=[
            "the Supplementary Information PDF -- deposited, with its MinerU "
            f"{MINERU_VERSION} OCR beside it. Supplementary Tables 1 and 2 (the "
            "enriched-gRNA target lists) are what the loader consumes; Supplementary "
            "Tables 3-6 (the optimization mutations, the strain table, the plasmid "
            "table and the recombineering gRNAs) are quoted for the genotype and the "
            "environment",
            "NO PER-STRAIN ISOPRENOL TITER IS RELEASED ANYWHERE. Every titer in this "
            "paper is a plotted figure panel (Figs. 4D, 4G, 6A, 6B and figs. S13, S14, "
            "S19, S20). Measured on the deposit: Supplementary Data 1 holds metabolite "
            "concentrations, the designed gRNA library, its read distribution, the "
            "lost-guide list, the WGS polymorphisms and a ShinyGO enrichment; "
            "Supplementary Data 2 holds five proteomics sheets. Neither is a titer "
            "table, so no ProductTiterPhenotype can be written for this paper",
            "NO PER-GUIDE ENRICHMENT MATRIX IS RELEASED. The pooled library is ~16,500 "
            "guides over 5,591 coding sequences and the deposit holds the designed "
            "sequences, the baseline read distribution and the missing-variant list, "
            "but no guide-by-sample abundance. The released per-round output is the "
            "picked target list of Supplementary Tables 1 and 2, which is what this "
            "loader stores as a categorical call",
            f"Dryad {DRYAD_DOI} -- Supplementary Data 1, 2, 4 and 5 "
            "(Data_Dryad_Supplementary_Data_Updated_2025-9-18_2.zip, 78,976,184 B; "
            "README.md, 18,978 B). NOT deposited: no loader reads them, and "
            "datadryad.org serves an Anubis JavaScript proof-of-work challenge, "
            "measured 2026-10-07 (/downloads/file_stream/<id> returns the challenge "
            "page with HTTP 200, /api/v2/files/<id>/download returns HTTP 401 'must "
            "have current bearer token'). MANUAL RECIPE: open "
            f"https://doi.org/{DRYAD_DOI} in a browser, solve the challenge, use "
            "'Download dataset', then deposit the two files under data/dryad/ with "
            "RetrievalMethod.manual_browser and the sha256 of the bytes that arrive",
            f"Zenodo {ZENODO_DOI} -- Supplementary Data 3, AlphaFold3 output. NOT "
            "deposited: no loader consumes structure predictions",
            f"PRIDE {PRIDE_ACCESSION} -- 97 raw proteomics files at 8, 24 and 48 h. "
            "NOT deposited: no loader consumes raw spectra",
            f"BioProject {BIOPROJECT_ACCESSION} -- three whole-genome resequencing "
            "runs. NOT deposited: no loader consumes reads",
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
# Supplementary-table readers
# --------------------------------------------------------------------------- #
_CAPTION_RE = re.compile(r"Supplementary Table (\d+)\.")
_TR_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _cells(row: str) -> list[str]:
    """The text of one OCR table row's cells, tags stripped and entities unescaped."""
    return [
        html.unescape(_TAG_RE.sub("", cell)).strip() for cell in _TD_RE.findall(row)
    ]


def read_table(markdown: str, number: int) -> list[tuple[str, str, str]]:
    """``(Gene, Function, FunctionalCategory)`` rows of one Supplementary Table.

    The OCR emits each table as a single ``<table>...</table>`` line following its
    caption line, so the caption nearest above a table is the one that names it. The
    header is checked rather than assumed: a changed export must fail, not be parsed
    into the wrong columns.
    """
    current: int | None = None
    for line in markdown.split("\n"):
        stripped = line.strip()
        match = _CAPTION_RE.search(stripped)
        if match is not None and stripped.lstrip("# ").startswith(
            "Supplementary Table"
        ):
            current = int(match.group(1))
        if not stripped.startswith("<table>") or current != number:
            continue
        rows = [_cells(row) for row in _TR_RE.findall(stripped)]
        if not rows or tuple(rows[0]) != TABLE_HEADER:
            raise RuntimeError(
                f"Supplementary Table {number} header is {rows[0] if rows else None!r}, "
                f"not {list(TABLE_HEADER)!r}"
            )
        body: list[tuple[str, str, str]] = []
        for cells in rows[1:]:
            if len(cells) != len(TABLE_HEADER):
                raise RuntimeError(
                    f"Supplementary Table {number} row has {len(cells)} cells: {cells!r}"
                )
            body.append((cells[0], cells[1], cells[2]))
        return body
    raise RuntimeError(f"Supplementary Table {number} is not in the pinned markdown")


class SelectedTarget(BaseModel):
    """One picked enrichment target: its round, its verbatim cells and its loci."""

    round_number: int
    gene_cell: str
    function: str
    functional_category: str
    locus_tags: tuple[str, ...]
    source_symbol: str | None


def parse_gene_cell(cell: str) -> tuple[tuple[str, ...], str | None]:
    """``(locus tags, the cell's own gene symbol)`` of one ``Gene`` cell.

    A cell names one locus (``PP_5007``, ``sotB / PP_2428``) or an operon written as
    its endpoint tags (``phaAZC-II / PP_5003-PP_5005``), which is expanded by integer
    enumeration of the endpoints. The symbol is whatever remains once the tags and the
    separators are removed, and is ``None`` when the cell is a bare tag.
    """
    residue = cell
    tags: list[str] = []
    for match in LOCUS_RANGE_RE.finditer(cell):
        first, last = int(match.group(1)), int(match.group(2))
        if last <= first:
            raise RuntimeError(f"locus range {match.group(0)!r} does not ascend")
        tags.extend(f"PP_{number:04d}" for number in range(first, last + 1))
        residue = residue.replace(match.group(0), " ")
    tags.extend(tag for tag in LOCUS_TAG_RE.findall(residue))
    residue = LOCUS_TAG_RE.sub(" ", residue)
    if not tags:
        raise RuntimeError(f"Gene cell {cell!r} names no PP_ locus tag")
    symbol = residue.strip(SYMBOL_STRIP).strip() or None
    ordered = sorted(dict.fromkeys(tags))
    if len(ordered) != len(tags):
        raise RuntimeError(f"Gene cell {cell!r} repeats a locus tag")
    return tuple(ordered), symbol


def read_selected_targets(path: str) -> list[SelectedTarget]:
    """Every picked target of both rounds, in table order.

    The per-round row count must equal the Methods' stated 28 and 30, and the two
    rounds must share no locus tag ("no overlap with the first set, as expected").
    """
    markdown = Path(path).read_text(encoding="utf-8")
    targets: list[SelectedTarget] = []
    per_round: dict[int, set[str]] = {}
    for number, _ in ROUND_TABLES:
        rows = read_table(markdown, number)
        if len(rows) != EXPECTED_ROWS[number]:
            raise RuntimeError(
                f"Supplementary Table {number} has {len(rows)} rows; the Methods state "
                f"{EXPECTED_ROWS[number]} ({_Q_N_TARGETS!r})"
            )
        tags_here: set[str] = set()
        for gene_cell, function, category in rows:
            tags, symbol = parse_gene_cell(gene_cell)
            tags_here.update(tags)
            targets.append(
                SelectedTarget(
                    round_number=number,
                    gene_cell=gene_cell,
                    function=function,
                    functional_category=category,
                    locus_tags=tags,
                    source_symbol=symbol,
                )
            )
        per_round[number] = tags_here
    shared = per_round[1] & per_round[2]
    if shared:
        raise RuntimeError(
            f"the two rounds share {sorted(shared)}; the paper states {_Q_NO_OVERLAP!r}"
        )
    return targets


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


def host_reference(data_root: str | None = None) -> AssemblyReferenceGenome:
    """Plain KT2440, pinned to its GenBank assembly, with no asserted background.

    ``background`` is ``None`` deliberately: the selection hosts are "four different
    strains" the paper never names, so a ``BacterialStrainBackground`` here would
    assert a chassis the source does not state. Every edit the source DOES state is a
    perturbation in :func:`host_perturbations`.
    """
    return assembly_reference(KT2440_STRAIN, data_root=data_root)


#: ``GeneAdditionPerturbation.source_organism`` is a REQUIRED ``str`` on a leaf that
#: carries no ``provenance_gaps`` field, so an unreported origin cannot be typed as an
#: absence. The honest value is therefore this explicit sentinel rather than an
#: organism read off the ``MKmm`` / ``PMDHKQ`` suffixes. Making the field nullable with
#: the gap mixin is raised in the PR, not taken here.
SOURCE_ORGANISM_UNREPORTED = "unreported"


def pathway_perturbations() -> list[HeterologousPathwayPerturbation]:
    """The five integrated isoprenol-pathway genes, by their verbatim part tokens.

    Each token must appear in the quoted pIY670 description, so a changed quote cannot
    drift away from the stored identifiers unnoticed. ``source_organism`` is
    :data:`SOURCE_ORGANISM_UNREPORTED`: this paper defers the pathway's origin to
    reference 22 (Banerjee et al., isoprenol in P. putida) and reference 24 (Kang et
    al., the IPP-bypass pathway in E. coli), neither of which is mirrored, and the three
    ``mvaS`` HOMOLOGS whose organisms it does name are extra copies in the final
    producer strains rather than the integrated pathway's own ``mvaS``.
    """
    parts = str(PATHWAY_PARTS.value)
    built: list[HeterologousPathwayPerturbation] = []
    for gene in PATHWAY_GENES:
        token = gene["token"]
        if token not in parts:
            raise RuntimeError(
                f"pathway part {token!r} is not in the quoted pIY670 description "
                f"{parts!r}"
            )
        built.append(
            HeterologousPathwayPerturbation(
                systematic_gene_name=token,
                perturbed_gene_name=token,
                gene_namespace=KT2440_NAMESPACE,
                pathway_name=str(PATHWAY_NAME.value),
                source_organism=SOURCE_ORGANISM_UNREPORTED,
                is_heterologous=True,
                localization="chromosomal_integration",
                integration_locus=gene["locus"],
                promoter_name=gene["promoter"],
                copy_number=1.0,
            )
        )
    return built


def reporter_perturbation() -> HeterologousPathwayPerturbation:
    """The plasmid-borne ``PpedF``-``pyrF`` reporter that makes the selection growth-coupled."""
    return HeterologousPathwayPerturbation(
        systematic_gene_name=str(PYRF_REPORTER.value),
        perturbed_gene_name="pyrF",
        gene_namespace=KT2440_NAMESPACE,
        pathway_name="PpedF-RBS-pyrF growth-coupled isoprenol biosensor",
        source_organism="Teredinibacter turnerae T7901",
        is_heterologous=True,
        localization="episomal_plasmid",
        construct_name="RSF1010 PpedF-RBS-pyrF",
        promoter_name="PpedF",
        copy_number=1.0,
    )


def pyrf_deletion() -> BacterialDeletionPerturbation:
    """``ΔPP_1815``: the native ``pyrF`` lesion the selection complements."""
    return BacterialDeletionPerturbation(
        systematic_gene_name=str(PYRF_DELETION.value),
        perturbed_gene_name="pyrF",
        gene_namespace=KT2440_NAMESPACE,
    )


def host_perturbations() -> list[Any]:
    """Every engineered edit the paper states for EVERY selection host.

    One list, built once per build and shared by all 58 records: the native ``pyrF``
    deletion, the five integrated pathway genes and the plasmid-borne reporter. The
    chassis edits that vary across the unnamed selection hosts are NOT here; see
    :data:`UNASSERTED_CHASSIS`.
    """
    built: list[Any] = [
        pyrf_deletion(),
        *pathway_perturbations(),
        reporter_perturbation(),
    ]
    names = {str(p.systematic_gene_name) for p in built}
    if names != HOST_BACKGROUND_NAMES:
        raise RuntimeError(
            f"the host background is {sorted(names)}, not the declared "
            f"{sorted(HOST_BACKGROUND_NAMES)}"
        )
    return built


def crispri_perturbation(
    locus_tag: str, gene_name: str
) -> BacterialCrisprInterferencePerturbation:
    """One dCpf1 knockdown of a KT2440 gene; the enriched spacer was never released."""
    return BacterialCrisprInterferencePerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=gene_name,
        gene_namespace=KT2440_NAMESPACE,
        crispr=CrisprConstruct(
            effector=str(EFFECTOR.value),
            guide_sequence=None,
            n_guides=int(N_GUIDES.value),
        ),
    )


def _medium_perturbations() -> list[Any]:
    """Kanamycin and the pH the Methods state, shared by both arms of the selection."""
    return [
        SmallMoleculePerturbation(
            compound=resolved_compound("kanamycin"),
            concentration=Concentration(
                value=float(KANAMYCIN_UG_PER_ML.value), unit=ConcentrationUnit.ug_per_ml
            ),
        ),
        EnvironmentPhysicalPerturbation(
            factor=PhysicalFactor.ph,
            magnitude=Concentration(
                value=float(MEDIUM_PH.value), unit=ConcentrationUnit.ph
            ),
        ),
    ]


def _environment(*, induced: bool) -> Environment:
    """The 24 h deep-well M9 selection culture, with or without the pathway inducer.

    ``temperature`` is a typed absence: the paper states 30 C for the conjugation spot
    and for every other P. putida step it specifies, and does NOT state it for this
    plate. Carrying 30 C over from a neighboring step would be an inference.
    """
    if M9_NREL_MOPS_MENASALVAS2025.base_medium != "M9":
        raise RuntimeError(
            f"the served medium {M9_NREL_MOPS_MENASALVAS2025.name!r} is not an M9 "
            f"derivative, so it is not the {MEDIUM.value!r} the Methods name"
        )
    perturbations = _medium_perturbations()
    if induced:
        perturbations.append(
            SmallMoleculePerturbation(
                compound=resolved_compound("crystal violet"),
                concentration=Concentration(
                    value=float(INDUCER_UM.value), unit=ConcentrationUnit.micromolar
                ),
            )
        )
    return Environment(
        media=M9_NREL_MOPS_MENASALVAS2025,
        temperature=None,
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_hours=float(DURATION_HOURS.value),
        provenance_gaps=[
            ProvenanceGap(
                field="temperature",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=str(CONJUGATION_TEMPERATURE.note),
            )
        ],
    )


#: The gaps every selection phenotype carries: no enrichment score and no dispersion.
_PHENOTYPE_GAP_NOTE = (
    "the pooled selection's per-guide read counts were never released, so the call has "
    "no score and no dispersion; the paper's only numeric isoprenol readout is a titer "
    f"measured by {TITER_METHOD.value} whose per-strain values live in figure panels "
    "alone"
)


def selection_phenotype(
    round_number: int, *, is_reference: bool
) -> EnvironmentResponsePhenotype:
    """The released call for one selected target, or the uninduced arm's baseline.

    A record's call is ``enhanced`` with the source's own label ``enriched``: the
    guide's clone outgrew the pool under ``PpedF``-driven ``pyrF`` complementation,
    which is a biosensor signal above control. The reference is the same library in the
    same medium with the pathway inducer omitted, where no clone can be enriched by
    isoprenol, so its call is the baseline ``no_change``.
    """
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.categorical,
        assay_type=AssayType.biosensor_readout,
        environment_response=None,
        category=(
            ResponseCategory.no_change if is_reference else ResponseCategory.enhanced
        ),
        category_label=(
            str(PRESELECTION_ENVIRONMENT.value)
            if is_reference
            else str(CALL_CRITERION.value)
        ),
        n_samples=int(N_REPLICATES.value),
        sample_unit=SampleUnit(str(SAMPLE_UNIT.value)),
        units=(
            "enriched-gRNA call from the pooled PpedF-pyrF growth-coupled selection "
            "(a guide enriched above 5 reads in one biological replicate)"
        ),
        screen_id=SCREEN_IDS[round_number],
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=_PHENOTYPE_GAP_NOTE,
            )
            for field in (
                "environment_response",
                "environment_response_uncertainty",
                "environment_response_uncertainty_type",
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class BuildAccounting(BaseModel):
    """Everything a reader needs to audit one build's retention arithmetic."""

    dataset: str
    source_rows: int
    candidate_records: int
    kept_records: int
    dropped_records: int
    distinct_targets: int
    reconciliation: LocusTagReconciliation | None = None
    notes: list[str] = []

    def check(self) -> None:
        """Nothing may vanish between the parsed tables and the written store."""
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_records} "
                f"dropped != {self.candidate_records} candidates"
            )
        if self.dropped_records:
            raise RuntimeError(
                f"{self.dataset}: {self.dropped_records} records dropped, but this "
                "dataset declares no drop rule"
            )


def _write_accounting(accounting: BuildAccounting, preprocess_dir: str) -> None:
    """Validate and write ``preprocess/build_accounting.json``."""
    accounting.check()
    os.makedirs(preprocess_dir, exist_ok=True)
    path = osp.join(preprocess_dir, "build_accounting.json")
    with open(path, "w") as handle:
        handle.write(accounting.model_dump_json(indent=2))


def _standard_names(genome: PPutidaKT2440Genome, tags: Iterable[str]) -> dict[str, str]:
    """Locus tag -> the annotation's own gene symbol, falling back to the tag.

    The symbol is read from the annotation so one gene carries one spelling across
    datasets; the table's own spelling is kept in ``preprocess/selected_targets.csv``.
    """
    exact, _ = genome.feature_index["symbol"]
    by_tag: dict[str, str] = {}
    for symbol, loci in exact.items():
        for locus in loci:
            by_tag.setdefault(str(locus), str(symbol))
    return {tag: by_tag.get(tag, tag) for tag in tags}


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class IsoprenolSelectionMenasalvas2025Dataset(ExperimentDataset):
    """Menasalvas 2025 enriched CRISPRi knockdowns from the growth-coupled selection."""

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"
    #: Every selected target must resolve to a locus of the pinned assembly. Measured
    #: on the pinned markdown: 60 of 60 are current standard locus tags, so a value
    #: below 1.0 means the annotation or the released tables moved and the build stops.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 1.0

    def __init__(
        self,
        root: str = "data/torchcell/isoprenol_selection_menasalvas2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome resolves the selected targets to loci."""
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
        """The pinned Supplementary Information OCR the tables are parsed from."""
        return [SI1_MD_FILENAME]

    def download(self) -> None:
        """Link the pinned OCR into ``raw/`` after verifying it against the manifest."""
        _link_mirror_files(
            self.raw_dir, ((SI1_MD_REL, SI1_MD_FILENAME, SI1_MD_SHA256),)
        )
        log.info("Menasalvas 2025 artifacts linked into %s", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    @post_process
    def process(self) -> None:
        """Build one record per selected enrichment target; write LMDB."""
        verify_raw_files(self.raw_dir, {SI1_MD_FILENAME: SI1_MD_SHA256})
        targets = read_selected_targets(osp.join(self.raw_dir, SI1_MD_FILENAME))
        if len(targets) != EXPECTED_RECORDS:
            raise RuntimeError(
                f"{len(targets)} selected targets parsed, not the stated "
                f"{EXPECTED_RECORDS}"
            )
        genome = self._genome()

        tags = sorted({tag for target in targets for tag in target.locus_tags})
        stored, report = reconcile_locus_tags(genome, pd.Series(tags), label=self.name)
        report.require_resolved(self.MIN_RESOLVED_FRACTION)
        if report.outside_namespace:
            raise RuntimeError(
                f"{self.name}: selected targets outside {KT2440_NAMESPACE}: "
                f"{report.outside_namespace}"
            )
        stored_by_tag = dict(zip(tags, stored, strict=True))
        common = _standard_names(genome, stored_by_tag.values())

        reference_genome = host_reference()
        induced = _environment(induced=True)
        control = _environment(induced=False)
        host = host_perturbations()
        references = {
            number: BacterialEnvironmentResponseExperimentReference(
                dataset_name=self.name,
                genome_reference=reference_genome,
                environment_reference=control.model_copy(),
                phenotype_reference=selection_phenotype(number, is_reference=True),
            )
            for number, _ in ROUND_TABLES
        }
        pub = publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        rows: list[dict[str, Any]] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for target in tqdm(targets, desc="menasalvas2025-selection"):
                knockdowns = [
                    crispri_perturbation(stored_by_tag[tag], common[stored_by_tag[tag]])
                    for tag in target.locus_tags
                ]
                experiment = BacterialEnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=[*host, *knockdowns]),
                    environment=induced,
                    phenotype=selection_phenotype(
                        target.round_number, is_reference=False
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(
                        experiment, references[target.round_number], pub, itxn
                    ),
                )
                rows.append(
                    {
                        "round": target.round_number,
                        "screen_id": SCREEN_IDS[target.round_number],
                        "gene_cell": target.gene_cell,
                        "source_symbol": target.source_symbol or "",
                        "locus_tags": ";".join(
                            stored_by_tag[tag] for tag in target.locus_tags
                        ),
                        "annotation_symbols": ";".join(
                            common[stored_by_tag[tag]] for tag in target.locus_tags
                        ),
                        "function": target.function,
                        "functional_category": target.functional_category,
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "selected_targets.csv"), index=False
        )
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(targets),
                candidate_records=len(targets),
                kept_records=idx,
                dropped_records=len(targets) - idx,
                distinct_targets=len(tags),
                reconciliation=report,
                notes=[
                    "nothing is dropped: every row of Supplementary Tables 1 and 2 is "
                    "a record, and the two rounds are kept apart by screen_id",
                    "no ProductTiterPhenotype is written: this paper releases no "
                    "per-strain isoprenol titer anywhere, only plotted figure panels",
                    "the record's readout is a biosensor GROWTH call, not a titer; "
                    f"category_label is the source's own word ({CALL_CRITERION.value!r})",
                    "the selection hosts are not named, so the reference is plain "
                    "KT2440 with background=None; the chassis edits NOT asserted here "
                    "are " + "; ".join(UNASSERTED_CHASSIS),
                    "the tables are the targets the authors PICKED per functional "
                    "category from the enrichment, not the full enriched set; no "
                    "guide-by-sample abundance matrix was released",
                    "Supplementary Table 1's 'phaAZC-II / PP_5003-PP_5005' is one "
                    "guide over a polycistron and is stored as one record with three "
                    "knockdown perturbations",
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Menasalvas2025 selection: %d records over %d distinct targets "
            "(round 1 %d, round 2 %d)",
            idx,
            len(tags),
            sum(1 for t in targets if t.round_number == 1),
            sum(1 for t in targets if t.round_number == 2),
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
def stored_tags_are_loci(
    records: Sequence[dict[str, Any]], genome: PPutidaKT2440Genome
) -> LevelResult:
    """L1 SUPPLEMENTARY: every stored knockdown target resolves to itself."""
    tags = sorted(
        {
            str(perturbation["systematic_gene_name"])
            for record in records
            for perturbation in record["experiment"]["genotype"]["perturbations"]
            if str(perturbation["perturbation_type"]) == "bacterial_crispr_interference"
        }
    )
    elsewhere = [
        tag
        for tag in tags
        if (resolution := genome.resolve_gene_name(tag)).systematic_name != tag
        or resolution.status
        not in (GeneNameStatus.CURRENT, GeneNameStatus.NON_GENE_FEATURE)
    ]
    return LevelResult(
        level=Level.L1,
        name="stored_targets_are_loci_of_the_pinned_assembly",
        passed=not elsewhere,
        message=(
            f"SUPPLEMENTARY: {len(tags)} stored knockdown targets; {len(elsewhere)} do "
            "not resolve to themselves"
        ),
        details={"n_targets": len(tags), "not_a_locus": elsewhere[:20]},
    )


def assembly_pin(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L3 SUPPLEMENTARY: every reference pins the same KT2440 GenBank assembly."""
    pins = {
        (
            str(record["reference"]["genome_reference"].get("assembly_set")),
            str(record["reference"]["genome_reference"].get("assembly_accession")),
            str(record["reference"]["genome_reference"].get("background")),
        )
        for record in records
    }
    expected = {("pputida_KT2440_ASM756v2", "GCA_000007565.2", "None")}
    return LevelResult(
        level=Level.L3,
        name="assembly_pin_is_kt2440_genbank_with_no_asserted_background",
        passed=pins == expected,
        message=(
            f"SUPPLEMENTARY: {len(pins)} distinct assembly pin(s): {sorted(pins)}"
        ),
        details={"pins": sorted(pins)},
    )


def host_background_present(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L3 SUPPLEMENTARY: every genotype carries the whole stated host background."""
    missing: list[int] = []
    for index, record in enumerate(records):
        names = {
            str(p["systematic_gene_name"])
            for p in record["experiment"]["genotype"]["perturbations"]
        }
        if not HOST_BACKGROUND_NAMES <= names:
            missing.append(index)
    return LevelResult(
        level=Level.L3,
        name="every_genotype_carries_the_stated_host_background",
        passed=not missing,
        message=(
            f"SUPPLEMENTARY: {len(records)} genotypes checked against "
            f"{sorted(HOST_BACKGROUND_NAMES)}; {len(missing)} incomplete"
        ),
        details={"n_records": len(records), "incomplete": missing[:20]},
    )


def verify_build(dataset_root: str, data_root: str | None = None) -> VerificationReport:
    """Run the environment-response verifier on a built tree and write its report.

    The constant host background is passed as ``background_genes`` so the L1 strain key
    and the L4 gene universe see only the SCREENED knockdown targets: the five pathway
    tokens and the reporter's ``TERTU_1389`` are heterologous identifiers with no locus
    in any assembly, and ``PP_1815`` is the same lesion in all 58 records. Three
    SUPPLEMENTARY rows are appended; the verifier's own rows keep their verdicts.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset,
    )
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    genome = bacterial_genome("pputida", KT2440_STRAIN, data_root)
    report = verify_environment_response_dataset(
        records,
        dataset_name="IsoprenolSelectionMenasalvas2025Dataset",
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{SI1_MD_REL}",
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method=(
                "Supplementary Tables 1 and 2, the picked enriched-gRNA targets of the "
                "two pooled PpedF-pyrF growth-coupled selection rounds; the readout is "
                "a categorical biosensor-growth call ('enriched'), NOT an isoprenol "
                "titer, because no per-strain titer and no per-guide read count were "
                f"released. n_samples={N_REPLICATES.value} biological replicate "
                "selection cultures; no dispersion is released, so the uncertainty "
                "fields are typed ProvenanceGaps"
            ),
            page=_PAGE_SI_TABLES12,
            retrieved=SI_RETRIEVED_AT,
        ),
        expected_count=EXPECTED_RECORDS,
        background_genes=HOST_BACKGROUND_NAMES,
        resolve_gene_name=genome.resolve_gene_name,
        sgd_genes=set(genome.genbank.loci),
    )
    report.add(stored_tags_are_loci(records, genome))
    report.add(assembly_pin(records))
    report.add(host_background_present(records))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build the dataset and verify it, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/isoprenol_selection_menasalvas2025")
    genome = bacterial_genome("pputida", KT2440_STRAIN, data_root)
    dataset = IsoprenolSelectionMenasalvas2025Dataset(root=root, pputida_genome=genome)
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
                    "candidate_records",
                    "kept_records",
                    "dropped_records",
                    "distinct_targets",
                    "notes",
                )
            },
            indent=2,
        )
    )
    report = verify_build(root, data_root)
    print(report.summary())
    for result in report.results:
        flag = "PASS" if result.passed else "FAIL"
        print(f"  [{flag}] L{int(result.level)} {result.name}: {result.message}")


if __name__ == "__main__":
    main()
