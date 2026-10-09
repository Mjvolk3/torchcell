# torchcell/datasets/pputida/menasalvas2025
# [[torchcell.datasets.pputida.menasalvas2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/menasalvas2025
# Test file: tests/torchcell/datasets/pputida/test_menasalvas2025.py
r"""Menasalvas 2025 biosensor-coupled CRISPRi selection in P. putida KT2440.

Menasalvas et al. 2025 (Sci Adv 11, eady2677; doi:10.1126/sciadv.ady2677; PMID
41134890) built an isoprenol biosensor, coupled it to growth by replacing its mCherry
reporter with ``pyrF``, and ran a pooled dCpf1 CRISPRi library through two rounds of
that selection on an isoprenol-producing KT2440 chassis. This module serves four
families, one per released readout:

- :class:`IsoprenolSelectionMenasalvas2025Dataset` -- a
  ``BacterialEnvironmentResponseExperiment`` per enriched knockdown target of
  Supplementary Tables 1 and 2 (58 records).
- :class:`ProteomeMenasalvas2025Dataset` -- a ``BacterialProteinAbundanceExperiment``
  per released sample of the Dryad deposit's Supplementary Data 2 sheet 4: the three
  designed strains in the growth and the production phase (6 records over 2,090 loci).
- :class:`MetaboliteGrowthPhaseMenasalvas2025Dataset` and
  :class:`MetaboliteProductionPhaseMenasalvas2025Dataset` -- a ``BacterialMetaboliteExperiment``
  per released strain of data S1-1's 48 ``Absolute`` rows, one dataset per growth phase
  because the two phases are two environments (3 records each).

THE DRYAD DEPOSIT IS A MANUAL ONE, AND THE RECIPE IS THE ONLY WAY BACK TO IT.
datadryad.org serves an Anubis JavaScript proof-of-work challenge to scripts, so the
owner retrieved ``10.5061/dryad.sbcc2frjq`` in a browser on 2026-10-09 and deposited
the arrival zip plus its two members in the raw mirror under ``data/dryad/``, each
pinned by sha256 and each carrying :data:`DRYAD_MANUAL_RECIPE` verbatim as its
``retrieval_command``. A loader extracts ONLY the two workbooks it reads, into its own
``preprocess/dryad/`` directory, and never writes the other 167 members (``__MACOSX/``
resource forks, ``.DS_Store``, an Excel lock file, ``.fcs`` flow files, ``.fastq``
reads, the AlphaFold ``.mp4`` movies and Supplementary Data 4's TSV).

THE DEPOSIT README IS LOAD-BEARING, NOT INCIDENTAL. It is the only mirrored byte that
states data S1-1's replicate count ("Average value from 3 biological replicates") and
the replicon data S1-5's bare ``position`` column is on ("Genomic Coordinates in P.
putida AE015451"). The article states the replicate count too, in its Fig. 7E caption,
but the MinerU OCR of the article DROPPED that sentence, so ``paper.md`` cannot be the
source for it.

THE STRAIN NAMING IS INCONSISTENT AND IS NOT RESOLVED HERE. Measured over the mirror:
the Results and every released data sheet name ``TEAM-3174`` and ``TEAM-3185``; the
shotgun-proteomics Methods name ``TEAM-3175`` and ``TEAM-3184``; the data-availability
statement names ``TEAM-3175`` and ``TEAM-3185``; and data S1-5 names ``TEAM-2595``,
``TEAM-3175`` and ``TEAM-3184``. Four distinct names for at most two improved strains,
and no mirrored byte states any identity between them. ``TEAM-2595`` is spelled
identically in data S1-5 and in every phenotype sheet, so its 75 called variants attach
with no assumption; the 509 calls of ``TEAM-3175`` and ``TEAM-3184`` are REFUSED
(:data:`WGS_CLONE_DISPOSITION`), because attaching them would assert an identity the
source never states.

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
import math
import os
import os.path as osp
import re
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from openpyxl import load_workbook
from pydantic import BaseModel, ConfigDict, Field
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
    BacterialMetaboliteExperiment,
    BacterialMetaboliteExperimentReference,
    BacterialProteinAbundanceExperiment,
    BacterialProteinAbundanceExperimentReference,
    BacterialSequenceVariantPerturbation,
    BacterialSiteVariantPerturbation,
    BacterialVariantCall,
    BacterialVariantType,
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
    MetabolitePhenotype,
    PhysicalFactor,
    ProteinAbundancePhenotype,
    Publication,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    VariantCallMode,
    VariantSiteKind,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ROLE_SI_DATA,
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
# The Dryad manual deposit (issue #788 item 1)
# --------------------------------------------------------------------------- #
#: Where the owner's browser download lives inside the raw mirror.
DRYAD_DIR_REL = "data/dryad"
DRYAD_VERSION = "v20250919"
#: The bytes that ARRIVED from datadryad.org, kept because they are what the recipe
#: reproduces; the two members inside it are what the loaders read.
DRYAD_ZIP_FILENAME = "doi_10_5061_dryad_sbcc2frjq__v20250919.zip"
DRYAD_ZIP_SHA256 = "67ae73c6f20058513daee837389aace1fc1fd5973840de0d0d886e9da6e4c19c"
#: The inner zip, left exactly as released (it carries ``__MACOSX/``, ``.DS_Store``, an
#: Excel lock file, ``.fcs``, ``.fastq`` and ``.mp4`` members beside the data).
DRYAD_INNER_ZIP_FILENAME = "Data_Dryad_Supplementary_Data_Updated_2025-9-18_2.zip"
DRYAD_INNER_ZIP_SHA256 = (
    "ca5c9a1b7d5aca6851df886b6c5b27c56884e7fde68f4f63880363e023038463"
)
#: The deposit's own README: the authoritative per-COLUMN description of every sheet,
#: and the only mirrored byte that states the metabolomics replicate count and the
#: replicon the WGS coordinates are on.
DRYAD_README_FILENAME = "README.md"
DRYAD_README_SHA256 = "07f1e8ed9981276341a67a7843eff4c0d408c1c4ee976e5fd08c1d214ebb9271"
DRYAD_ZIP_REL = f"{DRYAD_DIR_REL}/{DRYAD_ZIP_FILENAME}"
DRYAD_INNER_ZIP_REL = f"{DRYAD_DIR_REL}/{DRYAD_INNER_ZIP_FILENAME}"
DRYAD_README_REL = f"{DRYAD_DIR_REL}/{DRYAD_README_FILENAME}"
DRYAD_RETRIEVED_AT = "2026-10-09"
#: The manual recipe, VERBATIM from the deposit's own ``DEPOSIT.md``, recorded as the
#: ``retrieval_command`` of every Dryad file's ``manual_browser`` retrieval record. A
#: rebuild re-runs it by hand and then verifies the three sha256 digests above.
DRYAD_MANUAL_RECIPE = (
    "open https://doi.org/10.5061/dryad.sbcc2frjq in a browser, solve the challenge, "
    'click "Download dataset", save the zip unchanged, then unzip it into this '
    "directory beside the zip. The two members are the two files the raw-mirror "
    "`si_expected` entry names: `Data_Dryad_Supplementary_Data_Updated_2025-9-18_2.zip`"
    " (78,976,184 B) and `README.md` (18,978 B). The inner zip is left as released (it "
    "carries `__MACOSX/`, `.DS_Store` and a `~$...xlsx` Excel lock file beside the five "
    "Supplementary Data items); a loader extracts what it reads in `preprocess/`."
)
#: Who produced those bytes and how, verbatim from ``DEPOSIT.md``.
DRYAD_RETRIEVED_BY = "the owner (mjvolk3), browser download"
DRYAD_DEPOSIT_NOTE = (
    "retrieval_method: manual_browser (same Anubis wall as #739; measured 2026-10-07, "
    "issue #788 item 1). files: the arrival zip plus its two members, with sha256 in "
    "SHA256SUMS.txt. SHA256SUMS.txt pins THREE files, not four: this deposit's arrival "
    "zip holds two members, where Carruthers 2025's held three."
)
#: Our own records beside the bytes, named in every retrieval record's ``params``.
DRYAD_DEPOSIT_RECORD_REL = f"{DRYAD_DIR_REL}/DEPOSIT.md"
DRYAD_CHECKSUMS_REL = f"{DRYAD_DIR_REL}/SHA256SUMS.txt"

#: The single directory every real member of the inner zip sits under.
INNER_ZIP_ROOT = "Data Dryad Supplementary Data Updated 2025-9-18"
#: The ONLY members a loader extracts, by exact zip path. Everything else in the inner
#: zip is junk (``__MACOSX/`` resource forks, ``.DS_Store``, the ``~$...xlsx`` Excel
#: lock file) or bytes no loader reads (``.fcs`` flow files, ``.fastq`` reads, the
#: AlphaFold ``.mp4`` movies, and the ``fast.genomics`` co-occurrence TSV). Measured
#: 2026-10-09: 170 members, of which three are data this module reads.
SI_DATA_1_MEMBER = f"{INNER_ZIP_ROOT}/Menasalvas et al Supplementary Data 1.xlsx"
SI_DATA_2_MEMBER = f"{INNER_ZIP_ROOT}/Menasalvas et al Supplementary Data 2.xlsx"
#: Local basenames the extraction writes under ``preprocess/dryad/``. The loader
#: chooses them, so no released member name can steer a write outside that directory.
SI_DATA_1_BASENAME = "supplementary_data_1.xlsx"
SI_DATA_2_BASENAME = "supplementary_data_2.xlsx"
SI_DATA_1_SHA256 = "9cf26d19ae9199c3ce8d0c00ce747558b08e612e8a9ad00c4e810c9f9d3a9554"
SI_DATA_2_SHA256 = "d6fa89134445612c1f9c04a18390858204e15b69ac179dd44739630dd64eff32"
#: ``member -> (local basename, sha256)`` for the two workbooks.
EXTRACTED_MEMBERS: dict[str, tuple[str, str]] = {
    SI_DATA_1_MEMBER: (SI_DATA_1_BASENAME, SI_DATA_1_SHA256),
    SI_DATA_2_MEMBER: (SI_DATA_2_BASENAME, SI_DATA_2_SHA256),
}
#: Where an extraction lands, relative to a dataset root.
EXTRACT_SUBDIR = "dryad"

# --------------------------------------------------------------------------- #
# The released sheets this module reads
# --------------------------------------------------------------------------- #
#: Supplementary Data 1, sheet 1: data S1-1, the metabolite concentrations.
SHEET_METABOLITES = "Sheet 1 Metabolites"
#: Supplementary Data 1, sheet 5: data S1-5, the breseq polymorphism table.
SHEET_WGS = "Sheet 5 Pp WGS"
#: Supplementary Data 2, sheet 4: the growth/production-phase proteome of the three
#: designed strains. The one proteomics sheet whose samples ARE the designed strains.
SHEET_PROTEOME = "Sheet 4. GrowthProduction phase"
#: Supplementary Data 2, sheet 5: the same three strains carrying an ADDITIONAL
#: plasmid copy of the pathway. NOT loaded -- see :data:`SHEET_5_DEFERRAL`.
SHEET_PATHWAY_OVEREXPRESSION = "Sheet 5. Isoprenol pathway over"

#: 1-based row of the header in each sheet, measured on the pinned workbooks. The
#: readers CHECK the header rather than assume it, so a re-released workbook stops the
#: build instead of being parsed into the wrong columns.
METABOLITE_BANNER_ROW = 4
METABOLITE_HEADER_ROW = 5
PROTEOME_HEADER_ROW = 5
WGS_HEADER_ROW = 3

#: The banner cells above the two concentration blocks of data S1-1, verbatim.
METABOLITE_AVERAGE_BANNER = "Average Concentration (µM)"
METABOLITE_SPECIFIC_BANNER = "Specific Concentration (µM/OD600)"
#: The header of data S1-1, verbatim and in order (18 columns).
METABOLITE_HEADER: tuple[str, ...] = (
    "Metabolite",
    "Relative vs Absolute Concentration*",
    "TEAM-2595 GP",
    "TEAM-2595 PP",
    "TEAM-3174 GP",
    "TEAM-3174 PP",
    "TEAM-3185 GP",
    "TEAM-3185 PP",
    "Fold Change: 3174/2595 GP",
    "Fold Change: 3185/2595 GP",
    "TEAM-2595 GP",
    "TEAM-2595 PP",
    "TEAM-3174 GP",
    "TEAM-3174 PP",
    "TEAM-3185 GP",
    "TEAM-3185 PP",
    "Fold Change: 3174/2595 GP",
    "Fold Change: 3185/2595 GP",
)
#: The header of the proteomics sheets, verbatim and in order.
PROTEOME_HEADER: tuple[str, ...] = (
    "Protein.Group",
    "Protein.Names",
    "Protein",
    "Protein.Description",
    "Sample",
    "Replicate",
    "Counts_sum",
)
#: The header of data S1-5, verbatim and in order.
WGS_HEADER: tuple[str, ...] = (
    "Sample",
    "evidence",
    "position",
    "mutation",
    "annotation",
    "gene",
    "description",
)

#: The two growth stages, as the sheet's own column suffixes and the README name them.
Phase = Literal["growth", "production"]
PHASES: tuple[Phase, ...] = ("growth", "production")
#: data S1-1's suffix per phase ("GP = growth phase samples. PP = production phase
#: samples", from the deposit README's own column note).
METABOLITE_PHASE_SUFFIX: dict[Phase, str] = {"growth": "GP", "production": "PP"}
#: The three strains data S1-1 and the proteome sheet both name, in released order.
RELEASED_STRAINS: tuple[str, ...] = ("TEAM-2595", "TEAM-3174", "TEAM-3185")
#: The control every record references.
CONTROL_STRAIN = "TEAM-2595"
#: ``Sample`` cell of the proteome sheet -> ``(strain, phase)``, measured 2026-10-09.
PROTEOME_SAMPLE_RE = re.compile(r"^(?P<number>\d{4})_(?P<phase>growth|production)$")
#: ``Replicate`` cells the proteome sheet uses.
PROTEOME_REPLICATE_RE = re.compile(r"^R(?P<index>\d+)$")

#: The two released concentration classes of data S1-1's own second column.
CONCENTRATION_ABSOLUTE = "Absolute"
CONCENTRATION_RELATIVE = "Relative"
#: Measured on the pinned workbook: 48 ``Absolute`` rows and 10 ``Relative`` ones.
EXPECTED_METABOLITE_ROWS: dict[str, int] = {
    CONCENTRATION_ABSOLUTE: 48,
    CONCENTRATION_RELATIVE: 10,
}
#: One record per (strain, phase) of the three released strains, per phase dataset.
EXPECTED_METABOLITE_RECORDS = len(RELEASED_STRAINS)
#: One record per released proteome sample: three strains x two phases.
EXPECTED_PROTEOME_RECORDS = len(RELEASED_STRAINS) * len(PHASES)
#: Replicates the proteome sheet's own ``Replicate`` column holds per sample.
EXPECTED_PROTEOME_REPLICATES = 3
#: Rows data S1-5 releases, and the rows each named clone carries.
EXPECTED_WGS_ROWS = 584
EXPECTED_WGS_ROWS_PER_CLONE: dict[str, int] = {
    "TEAM-2595": 75,
    "TEAM-3175": 254,
    "TEAM-3184": 255,
}

#: What one number of the proteome sheet IS, named so no two proteomics datasets are
#: ever silently pooled. The deposit README calls the column only "Counts_Sum" and the
#: Methods name the acquisition (DIA on an Orbitrap Exploris 480), so the string says
#: SUM of DIA protein counts, which is neither Carruthers' Top3 peptide signal nor its
#: percent-of-proteome.
PROTEOME_MEASUREMENT_TYPE = "dia_protein_counts_sum_mean"
#: What one number of data S1-1 IS: an absolute concentration in micromolar, following
#: the ``intracellular_concentration_mM`` precedent of Mulleder 2016.
METABOLITE_MEASUREMENT_TYPE = "lc_ms_intracellular_concentration_uM"

#: The replicon data S1-5's bare ``position`` column is on, verbatim as the deposit
#: README writes it. There is NO replicon column in the sheet, so this is the only
#: statement of what a coordinate means.
WGS_REPLICON = "AE015451"
#: The variant caller, verbatim from the deposit README's sheet title.
WGS_CALLER = "breseq v 0.38.1"
#: The arrow U+2192 the ``mutation`` and ``gene`` cells fuse their fields with, and the
#: U+2190 that marks a reverse-strand gene. Written as codepoints so a byte-level
#: change in the release cannot pass unnoticed.
ARROW_RIGHT = "→"
ARROW_LEFT = "←"
#: The non-breaking space and non-breaking hyphen the cells are littered with.
NBSP = " "
NB_HYPHEN = "‑"

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


# --------------------------------------------------------------------------- #
# Sourced values of the deposited arms (issue #788 item 1)
# --------------------------------------------------------------------------- #
_PAGE_README_S1_1 = (
    "Dryad README.md, 'Supplementary Data 1 (Excel sheet):' -> 'Sheet 1. Metabolite "
    "concentrations from selected isoprenol producer strains.', column table"
)
_PAGE_README_S1_5 = (
    "Dryad README.md, 'Sheet 5. Illumina Genome Resequencing and Polymorphism Analysis "
    "of Selected Isoprenol Clones Analysis from breseq v 0.38.1', column table"
)
_PAGE_README_S2_4 = (
    "Dryad README.md, 'Sheet 4. Growth/Production phase Samples of High Isoprenol "
    "Producer Strains TEAM-3174 & 3185 Compared to TEAM-2595', column table"
)
_PAGE_README_S2_5 = (
    "Dryad README.md, 'Sheet 5. Plasmid-born augmentation of Isoprenol pathway "
    "overexpression in genomically integrated producer strains', column table"
)
_PAGE_METHODS_METABOLOMICS = "Materials and Methods, Metabolomics"
_PAGE_METHODS_PROTEOMICS = "Materials and Methods, Shotgun proteomics analysis"
_PAGE_METHODS_PRODUCTION = "Materials and Methods, Isoprenol production assays"
_PAGE_RESULTS_IMPROVED = (
    "Results, Functional genomics analyses reveal metabolic shifts in high isoprenol "
    "producers (opening paragraph)"
)
_PAGE_RESULTS_MVAS = (
    "Results, Combinatorial strain engineering of the highest isoprenol producers"
)
_PAGE_DATA_AVAILABILITY = "Data and materials availability"


def _readme(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the pinned Dryad ``README.md``.

    The deposit's own README is the per-COLUMN authority for every released sheet, and
    for two values it is the ONLY mirrored statement: the metabolomics replicate count
    (the MinerU OCR of the article DROPPED the Fig. 7E sentence that also carries it)
    and the replicon data S1-5's bare ``position`` column is on.
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
                "the Dryad deposit's own README, retrieved by hand in a browser "
                f"({RetrievalMethod.manual_browser.value}) and pinned in the raw mirror"
            ),
            page=page,
            retrieved=DRYAD_RETRIEVED_AT,
        ),
    )


_Q_README_AVERAGE = (
    "Average value from 3 biological replicates. Strain names are indicated with the "
    "TEAM-XXXX format. GP = growth phase samples. PP = production phase samples. Fold "
    "Change was calculated by the determining the ratio of concentrations from the "
    "indicated strain IDs in during growth phase (GP)"
)
_Q_README_SPECIFIC = (
    "The average metabolite concentration was normalized against the OD~600~ at the "
    "time of sample harvest."
)
_Q_METABOLITE_FOOTNOTE = (
    "*Absolute concentrations for metabolites are indicated where the concentration "
    "was determined using a standard curve calculated with the peak area of the "
    "authentic analyte. Relative concentrations are displayed where the concentration "
    "indicated was determined using response from a single chemical standard."
)
_Q_METABOLOMICS_METHOD = (
    "Metabolites were quantified via external calibration curves or chemical standards "
    "(for relative quantification)."
)
_Q_README_WGS_POSITION = "Genomic Coordinates in P. putida AE015451"
_Q_README_WGS_CALLER = (
    "Sheet 5. Illumina Genome Resequencing and Polymorphism Analysis of Selected "
    "Isoprenol Clones Analysis from breseq v 0.38.1"
)
_Q_README_WGS_EVIDENCE = (
    "RA = read alignment evidence. MJ = missing junction to reference sequence. JC = "
    "potential new junction."
)
_Q_README_PROTEIN_COLUMN = "Tertiary Protein.ID from UniPROT"
_Q_README_SHEET5 = (
    "Sheet 5. Plasmid-born augmentation of Isoprenol pathway overexpression in "
    "genomically integrated producer strains"
)
_Q_PROTEOMICS_HARVEST = (
    "TEAM-2595, TEAM-3175, and TEAM-3184 P. putida strains were grown for isoprenol "
    "production in deep-well plates. The log-phase samples were harvested when each "
    "strain reached an $\\mathrm { O D } _ { 6 0 0 }$ of 0.7 (roughly 8 to 14 hours "
    "postback dilution and induction) as monitored by a spectrophotometer. The "
    "production-phase samples were harvested at the 24-hour time point."
)
_Q_PRODUCTION_RUN = (
    "Production runs were performed in 24-deep-well plates $( 1 . 5 \\mathrm { m l } )$ "
    "with a gas-permeable film and M9 minimal media supplemented with "
    "$1 \\mu \\mathrm { M }$ crystal violet as an inducer."
)
_Q_ADAPTATION_TEMPERATURE = (
    "These plates were sealed with a gas-permeable film and incubated at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ with $1 0 0 0 \\mathrm { r p m }$ linear shaking "
    "and $7 0 \\%$ humidity."
)
#: The Results sentence that states both improved strains' genotypes. The MinerU OCR
#: breaks it across a blank line after "PP_3540/mvaB, and"; the quote joins the two
#: fragments with one space and changes nothing else.
_Q_IMPROVED_GENOTYPES = (
    "Our two highest producer strains TEAM-3185 and TEAM-3174 contain gene deletions "
    "in PP_2428, PP_4622, PP_3540/mvaB, and PP_4373/fleQ. TEAM-3174 also overexpresses "
    "mvaS and includes ΔPP_2710. TEAM-3185 lacks the mvaS overexpression, and "
    "ΔPP_2710 but PP_2074 is deleted."
)
_Q_MVAS_ORGANISM = (
    "We tested a small set of mvaS variants (Fig. 6A) and identified overexpression of "
    "Enterococcus faecalis mvaS as the most successful for isoprenol production "
    "improvement (Fig. 6A)."
)
_Q_FLEQ_SPAN = (
    "WGS identified 74 additional nonsynonymous single-nucleotide polymorphisms (SNPs) "
    "of unknown function in both improved producer strains, and a 26.1-kb deletion "
    "surrounding the fleQ/PP_4373 locus containing 24 flagellarelated genes. A "
    "comprehensive tabulation of all polymorphisms is described in data S1-5."
)
_Q_BIOSAMPLES = (
    "linked to BioSamples SAMN46924002 (strain TEAM-2595), SAMN46924003 (strain "
    "TEAM-3175), and SAMN46924004 (strain TEAM-3185)"
)
_Q_TABLE4_TEAM2595 = (
    "Pp KT2440 P3100-PP_266,PP_2665 PP_5402ntereicRBS-mCherry "
    "PP_5322intergenic::Pcv-mvaS,mvaEPP_0871intergenic::Ptrc-MKmm,PMDHKQ,ahpA"
)
_Q_TABLE4_TEAM2632 = (
    "Pp KT2440 P123100-PP_2666,PP_2665 PP_5402intergenic::PpedF-RBS-mCherry "
    "PP_5322intergenic::Pcv-mvaS,mvaEPP_0871intergenic::Ptrc-MKmm,PMDHKQ,ahpA"
    "PP_3159intergenic::PpedF-RBS-mCherry"
)
_Q_TABLE4_TEAM2777 = (
    "Pp KT2440 P123100-PP_2666,PP_2665 PP_5402intergenic:PpedF- RBS-mCherry "
    "PP_5322intergenic::Pcv-mvaS,mvaEPP_0871intergenic::Prc-MKmm,PMDHKQ,ahpA"
    "PP_3159intergenic::PpedF-RBS-mCherry ΔPP_ 2664 ΔPP_2675PJ23119-PP_1697"
)
_Q_TABLE4_TEAM3185 = "Pp TEAM-2777 ΔPP_ 2428 ΔPP_ 4622 ΔPP_ 3540 ΔPP_4373ΔPP_2074"
_Q_TABLE4_TEAM3174 = "ΔPP_2710 + PP_1117intergenic::Pcv-E. faecalis.mvaS"
_Q_TABLE4_TEAM3174_SECOND = (
    "PP_5464intergenic::Pcv-E. faecalis.mvaS Pp TEAM-2777 ΔPP_ 2428 "
    "ΔPP_ 4622 ΔPP_ 3540 ΔPP_4373"
)

METABOLITE_N_REPLICATES = _readme(
    3,
    _Q_README_AVERAGE,
    page=_PAGE_README_S1_1,
    note="the replicate count behind data S1-1's 'Average Concentration (uM)' column, "
    "stated per column by the deposit's own README. The article states it too, in the "
    "Fig. 7E caption, but the MinerU OCR of the article DROPPED that sentence, so this "
    "README is the only mirrored byte that carries it. No SD and no SE is released for "
    "any metabolite value anywhere in the mirror, so the phenotype carries n_replicates "
    "and no dispersion",
)
METABOLITE_UNCERTAINTY_ABSENT = _readme(
    None,
    _Q_README_AVERAGE,
    page=_PAGE_README_S1_1,
    note="the column note states the replicate count and NO dispersion statistic; the "
    "sheet releases 18 columns and none of them is an SD, SE, CV or n column (measured "
    "2026-10-09: two merged banners C4:J4 and K4:R4, no hidden rows or columns, no "
    "cell comments). metabolite_level_se is therefore None with a typed ProvenanceGap, "
    "not a zero and not a derived value",
)
METABOLITE_CLASS_FOOTNOTE = _readme(
    CONCENTRATION_ABSOLUTE,
    _Q_METABOLITE_FOOTNOTE,
    page=_PAGE_README_S1_1,
    note="the released class column's own footnote, which data S1-1 repeats verbatim in "
    "its row 66. An 'Absolute' value came off a standard curve of the authentic "
    "analyte; a 'Relative' one off a single chemical standard, so the two are on "
    "DIFFERENT scales and only the 48 Absolute rows are stored",
)
METABOLITE_SPECIFIC_DERIVED = _readme(
    METABOLITE_SPECIFIC_BANNER,
    _Q_README_SPECIFIC,
    page=_PAGE_README_S1_1,
    note="the second concentration block is the FIRST one divided by the harvest "
    "OD600, so storing both would store one measurement twice. The stored block is "
    f"{METABOLITE_AVERAGE_BANNER!r}: it is the quantity the LC-MS calibration produces, "
    "and the harvest OD600 it would be normalized by is not released per sample",
)
METABOLITE_QUANTIFICATION = _paper(
    "external calibration curve of the authentic analyte",
    _Q_METABOLOMICS_METHOD,
    page=_PAGE_METHODS_METABOLOMICS,
    note="how an Absolute concentration was obtained; the Methods state no replicate "
    "count, no SD and no SE, which is why the count is sourced from the deposit README",
)
PROTEOME_COLUMN = _readme(
    "Counts_Sum",
    _Q_README_PROTEIN_COLUMN,
    page=_PAGE_README_S2_4,
    note="the README describes the sheet's key column as a UniProt 'Tertiary "
    "Protein.ID' and leaves 'Counts_Sum' itself undescribed, so the measurement_type "
    f"names what the column IS as far as the mirror states it: {PROTEOME_MEASUREMENT_TYPE!r}"
    ", the per-sample MEAN over the released replicate Counts_sum values",
)
PROTEOME_HARVEST = _paper(
    {"growth": "OD600 0.7", "production": "24 h"},
    _Q_PROTEOMICS_HARVEST,
    page=_PAGE_METHODS_PROTEOMICS,
    note="the two harvest points, which are what makes growth phase and production "
    "phase two different environments. The growth-phase harvest is by OD600, not by "
    "clock time: its 'roughly 8 to 14 hours' is a RANGE no single duration_hours holds, "
    "so that environment's duration is a typed absence while the production phase's is "
    "24.0 h",
)
PRODUCTION_RUN = _paper(
    1.0,
    _Q_PRODUCTION_RUN,
    page=_PAGE_METHODS_PRODUCTION,
    note="the production culture both deposited arms were sampled from: M9 minimal "
    "medium with 1 uM crystal violet inducing the integrated isoprenol pathway",
)
PRODUCTION_TEMPERATURE_ABSENT = _paper(
    30.0,
    _Q_ADAPTATION_TEMPERATURE,
    page=_PAGE_METHODS_PRODUCTION,
    note="30 C is stated for the ADAPTATION plate and is NOT restated for the "
    "production run the proteome and metabolite samples come from, so "
    "Environment.temperature is a typed absence rather than a value carried over from "
    "the preceding step",
)
IMPROVED_GENOTYPES = _paper(
    {
        "shared": ("PP_2428", "PP_4622", "PP_3540", "PP_4373"),
        "TEAM-3174": ("PP_2710",),
        "TEAM-3185": ("PP_2074",),
    },
    _Q_IMPROVED_GENOTYPES,
    page=_PAGE_RESULTS_IMPROVED,
    note="the deletions of the two improved strains, stated in the Results. Checked "
    "against Supplementary Table 4's own strain rows, which state the same sets and "
    "additionally give TEAM-3174's two mvaS integration loci",
)
MVAS_ORGANISM = _paper(
    "Enterococcus faecalis",
    _Q_MVAS_ORGANISM,
    page=_PAGE_RESULTS_MVAS,
    note="the organism of the mvaS copies TEAM-3174 overexpresses, which Supplementary "
    "Table 4 writes into that strain's own genotype cell as 'Pcv-E. faecalis.mvaS'. It "
    "is NOT the organism of the INTEGRATED pathway's own mvaS, which stays "
    "SOURCE_ORGANISM_UNREPORTED",
)
TABLE4_CONTROL_CHASSIS = _si(
    _Q_TABLE4_TEAM2595,
    _Q_TABLE4_TEAM2595,
    page=_PAGE_SI_TABLE4,
    note="TEAM-2595's own strain row. The OCR mangles three part names in it "
    "('P3100-PP_266' for 'PJ23100-PP_2666', 'PP_5402ntereicRBS-mCherry' for "
    "'PP_5402intergenic::PpedF-RBS-mCherry', 'ahpA' for 'aphA'), which is why the "
    "stored identifiers are read from TEAM-2632's clean row instead and this quote is "
    "kept as the statement that TEAM-2595 carries those four elements and nothing else",
)
TABLE4_CLEAN_CHASSIS = _si(
    (
        "PP_5402intergenic",
        "PP_5322intergenic",
        "PP_0871intergenic",
        "PP_3159intergenic",
    ),
    _Q_TABLE4_TEAM2632,
    page=_PAGE_SI_TABLE4,
    note="TEAM-2632's row, the cleanest OCR of the same chassis, which is where the "
    "four integration loci are read from. TEAM-2595 carries the first three; the "
    "second PpedF-RBS-mCherry copy at PP_3159intergenic first appears in TEAM-2632 and "
    "is in every TEAM-2777 derivative",
)
TABLE4_TEAM2777_CHASSIS = _si(
    ("PP_2664", "PP_2675"),
    _Q_TABLE4_TEAM2777,
    page=_PAGE_SI_TABLE4,
    note="TEAM-2777's row: the base strain both improved strains derive from "
    "('Pp TEAM-2777 ...' in their own rows). The two deletions are stored; the "
    "PJ23119-PP_1697 promoter swap in the same row is NOT, because "
    "PromoterReplacementPerturbation requires expression_direction and no mirrored "
    "byte states a direction for it",
)
TABLE4_TEAM3185 = _si(
    ("PP_2428", "PP_4622", "PP_3540", "PP_4373", "PP_2074"),
    _Q_TABLE4_TEAM3185,
    page=_PAGE_SI_TABLE4,
    note="TEAM-3185's own strain row, which agrees with the Results sentence",
)
TABLE4_TEAM3174 = _si(
    ("PP_1117intergenic", "PP_5464intergenic"),
    f"{_Q_TABLE4_TEAM3174} | {_Q_TABLE4_TEAM3174_SECOND}",
    page=_PAGE_SI_TABLE4,
    note="TEAM-3174's two genotype cells, verbatim and separated by ' | '. Measured "
    "2026-10-09: the OCR merges this row with its rowspan neighbors, so the row's "
    "first fragment 'Pp TEAM-2777 dPP_2428 dPP_4622 dPP_3540 dPP_4373' sits in the "
    "cell above and the trailing 'Pp TEAM-2777 ...' of the second cell is the start of "
    "TEAM-3168's genotype. TEAM-3174's own statement is therefore 'dPP_2710 + "
    "PP_1117intergenic::Pcv-E. faecalis.mvaS PP_5464intergenic::Pcv-E. faecalis.mvaS', "
    "which is exactly what the Results sentence states independently",
)
WGS_REPLICON_SOURCED = _readme(
    WGS_REPLICON,
    _Q_README_WGS_POSITION,
    page=_PAGE_README_S1_5,
    note="data S1-5 has NO replicon column: its 'position' is a bare 1-based integer, "
    "and this README column note is the only mirrored statement of what it is a "
    "coordinate in. AE015451 is the single chromosome of GCA_000007565.2, and the "
    "value is stored verbatim as the README writes it (no version suffix)",
)
WGS_CALLER_SOURCED = _readme(
    WGS_CALLER,
    _Q_README_WGS_CALLER,
    page=_PAGE_README_S1_5,
    note="the caller and its version, from the deposit README's own sheet title",
)
WGS_EVIDENCE_LEGEND = _readme(
    {"RA": 569, "MC JC": 13, "JC": 2},
    _Q_README_WGS_EVIDENCE,
    page=_PAGE_README_S1_5,
    note="the README's legend for the evidence column, beside the counts measured on "
    "the pinned sheet. The legend defines 'MJ' while the sheet writes 'MC JC', which "
    "is why the released cell is kept verbatim and never mapped to an enum",
)
WGS_NAMING_DISAGREEMENT = _paper(
    {
        "results_and_every_data_sheet": ("TEAM-3174", "TEAM-3185"),
        "shotgun_proteomics_methods": ("TEAM-3175", "TEAM-3184"),
        "data_availability_biosamples": ("TEAM-3175", "TEAM-3185"),
        "data_S1_5": ("TEAM-3175", "TEAM-3184"),
    },
    f"{_Q_PROTEOMICS_HARVEST} | {_Q_BIOSAMPLES}",
    page=f"{_PAGE_METHODS_PROTEOMICS} | {_PAGE_DATA_AVAILABILITY}",
    note="FOUR distinct names for at most two improved strains, and no mirrored byte "
    "states any identity between them. The two quotes are joined by ' | ' and are "
    "unaltered. This is why data S1-5's TEAM-3175 and TEAM-3184 calls are REFUSED: "
    "attaching them to the TEAM-3174 and TEAM-3185 records would assert an identity "
    "the source never states. TEAM-2595 is spelled identically in data S1-5 and in "
    "every phenotype sheet, so its 75 calls attach with no assumption",
)
FLEQ_SPAN_NOT_IN_DATA = _paper(
    0,
    _Q_FLEQ_SPAN,
    page=_PAGE_RESULTS_IMPROVED,
    note="the sentence the deposit does NOT reproduce. Measured on data S1-5 "
    "(2026-10-09): 0 of its 584 rows name fleQ or PP_4373, and the largest released "
    "deletion is 'd1,867 bp' on PP_2664. The 26.1-kb span is therefore unsourceable "
    "from the released table and no BacterialSpanDeletionPerturbation is written; the "
    "sentence also says '74 additional' SNPs while the table holds 584 rows over 284 "
    "distinct positions for three clones, so the prose aggregate and the table are not "
    "the same statement",
)
SHEET_5_DEFERRAL = _readme(
    SHEET_PATHWAY_OVEREXPRESSION,
    _Q_README_SHEET5,
    page=_PAGE_README_S2_5,
    note="NOT loaded, and the reason is the arm's design rather than a missing field. "
    "Measured 2026-10-09: 55,080 rows, six samples TEAM_{2595,3174,3185}_{pIY670,"
    "pTE554} at four replicates each over 2,290 proteins, so it is the SAME three "
    "strains carrying an ADDITIONAL episomal copy of the five pathway genes the "
    "chromosome already holds. Every record's genotype would carry mvaS, mvaE, MKmm, "
    "PMDHKQ and aphA twice under two localizations, and the inducer the Supplementary "
    "Figure 18 caption states for it ('2% of arabinose') names no w/v or v/v basis "
    "where Supplementary Figure 20's caption does state '0.1% w/v'. Deferred as its "
    "own dataset with that measurement recorded, rather than folded into this sheet's "
    "records",
)
PATHWAY_ENZYME_ORGANISMS = _readme(
    {
        "Q9FD71": "HMGCS_ENTFL",
        "Q9FD70": "Q9FD70_ENTFL",
        "Q8PW39": "Q8PW39_METMA",
        "P32377": "MVD1_YEAST",
        "P0AE22": "APHA_ECOLI",
    },
    _Q_README_PROTEIN_COLUMN,
    page=_PAGE_README_S2_4,
    note="a FINDING of the proteome sheet, recorded and NOT acted on. Measured "
    "2026-10-09: nine of the sheet's 2,187 protein keys are not P. putida, and five of "
    "them are mevalonate-pathway enzymes whose UniProt entry names state their source "
    "organisms (Enterococcus faecalis HMG-CoA synthase and the acetyl-CoA "
    "acetyltransferase/HMG-CoA reductase, a Methanosarcina mazei mevalonate kinase, a "
    "Saccharomyces cerevisiae diphosphomevalonate decarboxylase and an Escherichia "
    "coli class B acid phosphatase). The sheet never says which pIY670 part token each "
    "one is, so SOURCE_ORGANISM_UNREPORTED stays on the integrated pathway and the "
    "mapping is raised for review rather than taken",
)
PROTEOME_CONTAMINANTS = _readme(
    ("P04264", "P13645", "P35527", "P00761"),
    _Q_README_PROTEIN_COLUMN,
    page=_PAGE_README_S2_4,
    note="the other four non-P. putida keys of the proteome sheet: three human "
    "keratins (type II cytoskeletal 1, type I cytoskeletal 10, type I cytoskeletal 9) "
    "and pig trypsin, which is the digestion enzyme the Methods add ('Overnight "
    "digestion with trypsin'). They are search-database entries, not host proteins, "
    "and no locus of the pinned assembly holds them",
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


def dryad_deposits() -> tuple[tuple[str, str, str], ...]:
    """``(relpath, role, sha256)`` of the three files the owner deposited by hand.

    The arrival zip is recorded beside its two members: a rebuild re-runs the recipe,
    gets the zip, and verifies everything inside it. The digests are read HERE rather
    than captured in a module constant, so a test that re-points the pins at a
    synthetic deposit re-points this too.

    SHA256SUMS.txt pins three files, not four. Carruthers 2025's deposit had four
    because its arrival zip held three members; this one holds two.
    """
    return (
        (DRYAD_ZIP_REL, ROLE_RAW_DATA, DRYAD_ZIP_SHA256),
        (DRYAD_INNER_ZIP_REL, ROLE_RAW_DATA, DRYAD_INNER_ZIP_SHA256),
        (DRYAD_README_REL, ROLE_SI_DATA, DRYAD_README_SHA256),
    )


def dryad_artifact_records(root: Path) -> list[ArtifactRecord]:
    """The three manual-deposit records, verified in place under ``root``.

    The owner retrieved these bytes in a browser and deposited them, so this function
    never copies and never fetches: it asserts each file is present with its pinned
    sha256 and describes it. An absent or altered file raises WITH the manual recipe,
    which is the only way to produce it again.
    """
    records: list[ArtifactRecord] = []
    for relpath, role, expected in dryad_deposits():
        path = root / relpath
        if not path.exists():
            raise RuntimeError(
                f"{path} is missing; the Menasalvas 2025 Dryad data is a manual "
                f"deposit. MANUAL RECIPE: {DRYAD_MANUAL_RECIPE}"
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
                            f"Dryad {DRYAD_DOI}, version of 2025-09-19 (the Dryad "
                            f"download is named `{DRYAD_ZIP_FILENAME}`)"
                        ),
                        "retrieved_by": DRYAD_RETRIEVED_BY,
                        "deposit_note": DRYAD_DEPOSIT_NOTE,
                        "deposit_record": DRYAD_DEPOSIT_RECORD_REL,
                        "checksums": DRYAD_CHECKSUMS_REL,
                    },
                    sha256=expected,
                    retrieved_at=DRYAD_RETRIEVED_AT,
                ),
            )
        )
    return records


def extract_dryad_members(
    inner_zip_path: str | Path, dest_dir: str | Path
) -> dict[str, Path]:
    """Extract ONLY the two workbooks this module reads, into ``dest_dir``.

    The released inner zip is untrusted input, so nothing here calls ``extractall``
    and no released member name ever reaches the filesystem: each wanted member is read
    by its exact zip path and written under a basename this module chose
    (:data:`EXTRACTED_MEMBERS`), which makes a traversing or absolute member name
    inert. The junk the deposit ships beside the data -- ``__MACOSX/`` resource forks,
    ``.DS_Store``, the ``~$...xlsx`` Excel lock file, the ``.fcs`` flow files, the
    ``.fastq`` reads and the AlphaFold ``.mp4`` movies -- is never written at all.

    Returns ``{member: written path}``. Idempotent by sha256: a file already present
    with its pinned digest is left alone, and one with a different digest raises.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    with zipfile.ZipFile(inner_zip_path) as archive:
        held = set(archive.namelist())
        missing = sorted(set(EXTRACTED_MEMBERS) - held)
        if missing:
            raise RuntimeError(
                f"{inner_zip_path} does not hold {missing}; the released member names "
                f"changed. MANUAL RECIPE: {DRYAD_MANUAL_RECIPE}"
            )
        for member, (basename, expected) in sorted(EXTRACTED_MEMBERS.items()):
            target = dest / basename
            if target.exists():
                if _sha256(target) != expected:
                    raise RuntimeError(
                        f"{target} exists with a different sha256; refusing"
                    )
            else:
                with archive.open(member) as source:
                    target.write_bytes(source.read())
                verify_sha256(target, expected)
            written[member] = target
    return written


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
    files.extend(dryad_artifact_records(root))
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
            "NO PER-STRAIN ISOPRENOL TITER IS RELEASED ANYWHERE, AND THAT IS STILL "
            "TRUE OF THE DEPOSITED BYTES. Every titer in this paper is a plotted "
            "figure panel (Figs. 4D, 4G, 6A, 6B and figs. S13, S14, S19, S20). "
            "Measured on the deposit now that it is mirrored: Supplementary Data 1 "
            "holds metabolite concentrations, the designed gRNA library, its read "
            "distribution, the lost-guide list, the WGS polymorphisms and a ShinyGO "
            "enrichment; Supplementary Data 2 holds five proteomics sheets. Neither is "
            "a titer table, so no ProductTiterPhenotype can be written for this paper. "
            "What IS loaded from the deposit: Supplementary Data 2 sheet 4 (the "
            "growth-phase and production-phase proteome of TEAM-2595, TEAM-3174 and "
            "TEAM-3185) by ProteomeMenasalvas2025Dataset, and Supplementary Data 1 "
            "sheet 1 (data S1-1, the 48 Absolute metabolite concentrations) by "
            "MetaboliteGrowthPhaseMenasalvas2025Dataset and "
            "MetaboliteProductionPhaseMenasalvas2025Dataset. Supplementary Data 1 "
            "sheet 5 (data S1-5, the breseq polymorphisms) is read for called-variant "
            "perturbations on the TEAM-2595 records only. NOT loaded: Supplementary "
            "Data 2 sheets 1-3 and 5, Supplementary Data 1 sheets 2-4 and the ShinyGO "
            "sheet, and Supplementary Data 4",
            "NO PER-GUIDE ENRICHMENT MATRIX IS RELEASED. The pooled library is ~16,500 "
            "guides over 5,591 coding sequences and the deposit holds the designed "
            "sequences, the baseline read distribution and the missing-variant list, "
            "but no guide-by-sample abundance. The released per-round output is the "
            "picked target list of Supplementary Tables 1 and 2, which is what this "
            "loader stores as a categorical call",
            f"Dryad {DRYAD_DOI} {DRYAD_VERSION} -- Supplementary Data 1, 2, 4 and 5. "
            f"DEPOSITED {DRYAD_RETRIEVED_AT} by the owner in a browser, because "
            "datadryad.org serves an Anubis JavaScript proof-of-work challenge to "
            "scripts, measured 2026-10-07 (/downloads/file_stream/<id> returns the "
            "challenge page with HTTP 200, /api/v2/files/<id>/download returns HTTP "
            f"401 'must have current bearer token'). Three files under {DRYAD_DIR_REL}/"
            f" with RetrievalMethod.manual_browser: the arrival zip "
            f"{DRYAD_ZIP_FILENAME} (78,995,492 B), its member "
            f"{DRYAD_INNER_ZIP_FILENAME} (78,976,184 B) and its member "
            f"{DRYAD_README_FILENAME} (18,978 B), each pinned by sha256 in "
            f"{DRYAD_CHECKSUMS_REL} with the recipe in {DRYAD_DEPOSIT_RECORD_REL}. The "
            "inner zip is kept exactly as released; a loader extracts only the two "
            "workbooks it reads into its own preprocess/ directory and never writes "
            "the 167 junk and unread members (__MACOSX/ resource forks, .DS_Store, a "
            "~$...xlsx Excel lock file, .fcs flow files, .fastq reads, .mp4 AlphaFold "
            "movies, the fast.genomics TSV of Supplementary Data 4). The README is "
            "load-bearing, not incidental: it is the only mirrored byte that states "
            "data S1-1's replicate count (the MinerU OCR of the article dropped the "
            "Fig. 7E sentence) and the replicon data S1-5's bare position column is on",
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
class DropRule(BaseModel):
    """One retention rule, what it removed, and the items it removed.

    ``n_records`` counts whole RECORDS a rule removed; a rule whose scope is a
    measurement KEY removes no record (every sample keeps its record, with fewer keys
    in its map) and carries ``n_records=0`` with the keys in ``items``.
    """

    rule: str
    scope: Literal["record", "sample", "metabolite_row", "protein_key", "variant_call"]
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
    distinct_targets: int
    rules: list[DropRule] = []
    reconciliation: LocusTagReconciliation | None = None
    notes: list[str] = []

    def check(self) -> None:
        """Nothing may vanish between the parsed source and the written store."""
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_records} "
                f"dropped != {self.candidate_records} candidates"
            )
        accounted = sum(rule.n_records for rule in self.rules)
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
# Deposited-workbook readers
# --------------------------------------------------------------------------- #
def _norm(value: Any) -> str:
    """A released cell as a plain string: no-break space and hyphen normalized.

    The deposited sheets are littered with U+00A0 and U+2011 (``cadA`` is written
    ``cadA‑I``), so every cell goes through this before it is matched. The
    VERBATIM cell is what reaches a ``quote=`` or a ``*_statement`` field; this is only
    for matching and for keys.
    """
    return str(value).replace(NBSP, " ").replace(NB_HYPHEN, "-").strip()


def _sheet_rows(
    path: str | Path, sheet: str, header_row: int, header: tuple[str, ...]
) -> list[tuple[Any, ...]]:
    """Every non-empty row of ``sheet`` below its CHECKED header.

    The header is asserted, never assumed: a re-released workbook that renamed or
    reordered a column stops the build instead of being read into the wrong fields.
    """
    workbook = load_workbook(path, read_only=True, data_only=True)
    if sheet not in workbook.sheetnames:
        raise RuntimeError(
            f"{path} holds {workbook.sheetnames!r}, not a sheet named {sheet!r}"
        )
    rows = list(workbook[sheet].iter_rows(values_only=True))
    workbook.close()
    if len(rows) < header_row:
        raise RuntimeError(
            f"{sheet!r} has {len(rows)} rows; the header is on row {header_row}"
        )
    got = tuple(
        _norm(cell) if cell is not None else "" for cell in rows[header_row - 1]
    )[: len(header)]
    if got != header:
        raise RuntimeError(
            f"{sheet!r} row {header_row} is {list(got)!r}, not {list(header)!r}"
        )
    return [row for row in rows[header_row:] if any(cell is not None for cell in row)]


class MetaboliteRow(BaseModel):
    """One released row of data S1-1, with both concentration blocks verbatim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metabolite: str = Field(
        description="the released name, normalized for use as a key"
    )
    metabolite_statement: str = Field(description="the released cell, verbatim")
    concentration_class: str = Field(description="'Absolute' or 'Relative', verbatim")
    #: ``(strain, phase) -> average micromolar``; a key is absent where the cell is
    #: blank, which is an ABSENT measurement. A released 0 is a measurement and is here.
    average_um: dict[tuple[str, str], float]
    #: the same keys for the derived ``Specific Concentration (uM/OD600)`` block, read
    #: only so the verifier can use it as an oracle; no record stores it.
    specific_um_per_od: dict[tuple[str, str], float]


def _metabolite_column_index() -> dict[tuple[str, str], tuple[int, int]]:
    """``(strain, phase) -> (average column, specific column)``, 0-based.

    Read off :data:`METABOLITE_HEADER` rather than hardcoded, and checked: the two
    blocks repeat the same six ``<strain> <GP|PP>`` headers, the first under
    ``Average Concentration`` and the second under ``Specific Concentration``.
    """
    index: dict[tuple[str, str], tuple[int, int]] = {}
    for strain in RELEASED_STRAINS:
        for phase in PHASES:
            label = f"{strain} {METABOLITE_PHASE_SUFFIX[phase]}"
            found = [i for i, name in enumerate(METABOLITE_HEADER) if name == label]
            if len(found) != 2:
                raise RuntimeError(
                    f"data S1-1's header names {label!r} {len(found)} times, not twice"
                )
            index[(strain, phase)] = (found[0], found[1])
    return index


def read_metabolite_rows(path: str | Path) -> tuple[list[MetaboliteRow], str]:
    """Every released row of data S1-1, and the sheet's own class footnote verbatim.

    The two banner cells above the concentration blocks are checked, so a release that
    changed the unit cannot be read as micromolar. The footnote is checked against
    :data:`_Q_METABOLITE_FOOTNOTE`, which is the same sentence the deposit README
    states for the class column, so the two sources must still agree.
    """
    workbook = load_workbook(path, read_only=True, data_only=True)
    if SHEET_METABOLITES not in workbook.sheetnames:
        raise RuntimeError(f"{path} has no sheet named {SHEET_METABOLITES!r}")
    rows = list(workbook[SHEET_METABOLITES].iter_rows(values_only=True))
    workbook.close()
    header = tuple(
        _norm(cell) if cell is not None else ""
        for cell in rows[METABOLITE_HEADER_ROW - 1]
    )[: len(METABOLITE_HEADER)]
    if header != METABOLITE_HEADER:
        raise RuntimeError(
            f"data S1-1 row {METABOLITE_HEADER_ROW} is {list(header)!r}, not "
            f"{list(METABOLITE_HEADER)!r}"
        )
    banner = rows[METABOLITE_BANNER_ROW - 1]
    index = _metabolite_column_index()
    average_first = min(pair[0] for pair in index.values())
    specific_first = min(pair[1] for pair in index.values())
    for column, expected in (
        (average_first, METABOLITE_AVERAGE_BANNER),
        (specific_first, METABOLITE_SPECIFIC_BANNER),
    ):
        got = _norm(banner[column]) if banner[column] is not None else ""
        if got != expected:
            raise RuntimeError(
                f"data S1-1 row {METABOLITE_BANNER_ROW} column {column + 1} is "
                f"{got!r}, not the released banner {expected!r}"
            )

    parsed: list[MetaboliteRow] = []
    footnote: str | None = None
    for row in rows[METABOLITE_HEADER_ROW:]:
        if all(cell is None for cell in row):
            continue
        name, concentration_class = row[0], row[1]
        if concentration_class is None:
            if footnote is not None:
                raise RuntimeError(
                    f"data S1-1 holds a second class-free row {_norm(name)!r}; the "
                    "sheet released exactly one, its class footnote"
                )
            footnote = _norm(name)
            continue
        released_class = _norm(concentration_class)
        if released_class not in EXPECTED_METABOLITE_ROWS:
            raise RuntimeError(
                f"data S1-1 row {_norm(name)!r} is class {released_class!r}, not one "
                f"of {sorted(EXPECTED_METABOLITE_ROWS)}"
            )
        average: dict[tuple[str, str], float] = {}
        specific: dict[tuple[str, str], float] = {}
        for key, (average_column, specific_column) in index.items():
            if row[average_column] is not None:
                average[key] = float(row[average_column])
            if row[specific_column] is not None:
                specific[key] = float(row[specific_column])
        parsed.append(
            MetaboliteRow(
                metabolite=_norm(name),
                metabolite_statement=str(name),
                concentration_class=released_class,
                average_um=average,
                specific_um_per_od=specific,
            )
        )
    if footnote is None:
        raise RuntimeError("data S1-1 no longer carries its class footnote row")
    if footnote != _Q_METABOLITE_FOOTNOTE:
        raise RuntimeError(
            f"data S1-1's class footnote is {footnote!r}, not the pinned "
            f"{_Q_METABOLITE_FOOTNOTE!r}"
        )
    counts = Counter(row.concentration_class for row in parsed)
    if counts != Counter(EXPECTED_METABOLITE_ROWS):
        raise RuntimeError(
            f"data S1-1 holds {dict(counts)}, not the measured "
            f"{EXPECTED_METABOLITE_ROWS}"
        )
    duplicates = sorted(
        name for name, n in Counter(row.metabolite for row in parsed).items() if n > 1
    )
    if duplicates:
        raise RuntimeError(f"data S1-1 names {duplicates} more than once")
    return parsed, footnote


class ProteomeRow(BaseModel):
    """One released row of the proteome sheet, every cell verbatim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protein_group: str
    protein_names: str
    protein: str
    protein_description: str
    sample: str
    replicate: str
    counts_sum: float


def read_proteome_rows(
    path: str | Path, sheet: str = SHEET_PROTEOME
) -> list[ProteomeRow]:
    """Every released row of one proteomics sheet of Supplementary Data 2."""
    rows = _sheet_rows(path, sheet, PROTEOME_HEADER_ROW, PROTEOME_HEADER)
    parsed = [
        ProteomeRow(
            protein_group=_norm(row[0]),
            protein_names=_norm(row[1]),
            protein=_norm(row[2]),
            protein_description=_norm(row[3]),
            sample=_norm(row[4]),
            replicate=_norm(row[5]),
            counts_sum=float(row[6]),
        )
        for row in rows
        if row[6] is not None
    ]
    blank = len(rows) - len(parsed)
    if blank:
        raise RuntimeError(
            f"{sheet!r} leaves {blank} of {len(rows)} Counts_sum cells blank; the "
            "pinned workbook leaves none, so an absent measurement has no encoding here"
        )
    if not parsed:
        raise RuntimeError(f"{sheet!r} released no rows")
    for row in parsed:
        if PROTEOME_REPLICATE_RE.match(row.replicate) is None:
            raise RuntimeError(
                f"{sheet!r} replicate cell {row.replicate!r} is not the released R<n> "
                "form"
            )
    return parsed


def proteome_sample_key(sample: str) -> tuple[str, Phase]:
    """``'3174_growth' -> ('TEAM-3174', 'growth')``.

    The sheet writes the strain as a bare four-digit number; the ``TEAM-`` prefix every
    other released file uses is restored here, and the result must be one of the three
    strains the sheet's own title row names.
    """
    match = PROTEOME_SAMPLE_RE.match(sample)
    if match is None:
        raise RuntimeError(
            f"proteome sample {sample!r} is not the released <number>_<phase> form"
        )
    strain = f"TEAM-{match.group('number')}"
    if strain not in RELEASED_STRAINS:
        raise RuntimeError(
            f"proteome sample {sample!r} names {strain}, which is not one of "
            f"{list(RELEASED_STRAINS)}"
        )
    phase: Phase = "growth" if match.group("phase") == "growth" else "production"
    return strain, phase


class ProteinKeyResolution(BaseModel):
    """What one proteome sheet's ``Protein`` keys resolved to, and what was dropped."""

    model_config = ConfigDict(extra="forbid")

    #: released ``Protein`` cell -> stored KT2440 locus tag, for the kept keys only.
    locus_of: dict[str, str]
    #: the five mevalonate-pathway enzymes, by their released ``Protein.Group``.
    pathway_enzymes: tuple[str, ...]
    #: the three human keratins and pig trypsin, by their released ``Protein.Group``.
    contaminants: tuple[str, ...]
    #: host keys whose name does not resolve to a locus tag of the pinned assembly.
    outside_namespace: tuple[str, ...]
    #: host keys the sheet files under more than one ``Protein.Group``.
    merged_keys: tuple[str, ...]
    reconciliation: LocusTagReconciliation


#: ``Protein.Names`` of a P. putida KT2440 entry ends with this UniProt organism code.
HOST_ORGANISM_CODE = "PSEPK"


def resolve_protein_keys(
    rows: Sequence[ProteomeRow], genome: PPutidaKT2440Genome, *, label: str
) -> ProteinKeyResolution:
    """Map the sheet's ``Protein`` keys to KT2440 locus tags, with every drop named.

    The sheet keys on a title-cased UniProt "tertiary Protein.ID" (``Pp_5426``,
    ``Xyla``), so the host keys go through :func:`reconcile_locus_tags`, which reaches
    a locus tag by locus tag first and by gene symbol second. Three kinds of key are
    then dropped, each with a measured reason and none with a default branch:

    - a key whose ``Protein.Names`` organism code is not ``PSEPK``. Measured
      2026-10-09: nine of 2,187, five of them mevalonate-pathway enzymes and four of
      them search-database contaminants. A non-host accession the module does not
      already name raises, so a new one is a finding rather than a silent drop.
    - a host key the sheet files under more than one ``Protein.Group`` (measured: four,
      ``Dapf``, ``Dapa``, ``Aroe`` and ``Asd``), whose abundance stands for two protein
      groups and cannot be attributed to either.
    - a host key that does not resolve to a locus tag of the pinned assembly (measured:
      86 of 2,178, 85 retired symbols and the ambiguous ``Asd``).
    """
    known_pathway = set(dict(PATHWAY_ENZYME_ORGANISMS.value))
    known_contaminants = set(PROTEOME_CONTAMINANTS.value)
    groups_by_key: dict[str, set[str]] = defaultdict(set)
    host_keys: set[str] = set()
    pathway: set[str] = set()
    contaminants: set[str] = set()
    for row in rows:
        organism = row.protein_names.rsplit("_", 1)[-1]
        if organism == HOST_ORGANISM_CODE:
            host_keys.add(row.protein)
            groups_by_key[row.protein].add(row.protein_group)
            continue
        if row.protein_group in known_pathway:
            pathway.add(row.protein_group)
        elif row.protein_group in known_contaminants:
            contaminants.add(row.protein_group)
        else:
            raise RuntimeError(
                f"{label}: {row.protein_group} ({row.protein_names}, "
                f"{row.protein_description!r}) is neither a {HOST_ORGANISM_CODE} entry "
                "nor one of the nine non-host entries measured on the pinned sheet; a "
                "new non-host protein is a finding, not a silent drop"
            )
    merged = tuple(
        sorted(key for key, groups in groups_by_key.items() if len(groups) > 1)
    )
    keys = sorted(host_keys)
    stored, report = reconcile_locus_tags(genome, pd.Series(keys), label=label)
    stored_by_key = dict(zip(keys, stored, strict=True))
    outside = set(report.outside_namespace)
    locus_of = {
        key: tag
        for key, tag in stored_by_key.items()
        if tag not in outside and key not in merged
    }
    if not locus_of:
        raise RuntimeError(f"{label}: no Protein key reached a locus of the assembly")
    return ProteinKeyResolution(
        locus_of=locus_of,
        pathway_enzymes=tuple(sorted(pathway)),
        contaminants=tuple(sorted(contaminants)),
        outside_namespace=tuple(sorted(outside)),
        merged_keys=merged,
        reconciliation=report,
    )


class ProteomeAggregate(BaseModel):
    """One proteome sample's per-locus mean, SE and replicate count."""

    model_config = ConfigDict(extra="forbid")

    abundance: dict[str, float]
    se: dict[str, float]
    n_replicates: dict[str, int]
    replicates: tuple[str, ...]


def aggregate_proteome_sample(
    rows: Sequence[ProteomeRow], sample: str, locus_of: Mapping[str, str], *, label: str
) -> ProteomeAggregate:
    """Mean, SE and replicate count per locus over one sample's released replicates.

    A released 0 is the number the sheet reports for a protein with no signal in that
    replicate; it is averaged in verbatim, never imputed and never dropped, so
    ``n_replicates`` is the replicate count the sample actually has. The SE is
    ``SD / sqrt(n)`` over the released values, and ``nan`` for a protein with exactly
    one replicate, which is the convention both Carruthers 2025 proteome families use.
    """
    values: dict[str, list[float]] = defaultdict(list)
    replicates: set[str] = set()
    for row in rows:
        if row.sample != sample:
            continue
        locus = locus_of.get(row.protein)
        if locus is None:
            continue
        values[locus].append(row.counts_sum)
        replicates.add(row.replicate)
    if not values:
        raise RuntimeError(f"{label}: sample {sample!r} measures no kept protein")
    abundance: dict[str, float] = {}
    se: dict[str, float] = {}
    n_replicates: dict[str, int] = {}
    for locus, present in values.items():
        n = len(present)
        abundance[locus] = statistics.fmean(present)
        n_replicates[locus] = n
        se[locus] = statistics.stdev(present) / math.sqrt(n) if n > 1 else float("nan")
    return ProteomeAggregate(
        abundance=abundance,
        se=se,
        n_replicates=n_replicates,
        replicates=tuple(sorted(replicates)),
    )


# --------------------------------------------------------------------------- #
# data S1-5: the breseq polymorphism table
# --------------------------------------------------------------------------- #
#: The typed encodings :func:`read_variant_calls` assigns, one per released row.
ENCODING_IN_LOCUS = "bacterial_sequence_variant_in_locus"
ENCODING_INTERGENIC = "bacterial_site_variant_intergenic"

#: ``gene`` cell forms, measured 2026-10-09 over all 584 rows: 395 name one locus with
#: a strand arrow and 189 name two flanking loci, and the split is 1:1 with the
#: ``annotation`` cell's own ``intergenic`` prefix.
GENE_ONE_LOCUS_RE = re.compile(rf"^(?P<name>\S+)\s*[{ARROW_RIGHT}{ARROW_LEFT}]$")
GENE_TWO_FLANKERS_RE = re.compile(
    rf"^(?P<left>.+?)\s*[{ARROW_RIGHT}{ARROW_LEFT}]\s*/\s*"
    rf"[{ARROW_RIGHT}{ARROW_LEFT}]\s*(?P<right>.+?)$"
)
#: ``mutation`` cell forms, measured over all 584 rows: 405 single-base substitutions,
#: 51 insertions, 17 delta-bp deletions, 94 tandem-repeat copy-number changes and 17
#: multi-base replacements. There is NO default branch.
MUT_SNV_RE = re.compile(rf"^[ACGT]{ARROW_RIGHT}[ACGT]$")
MUT_INSERTION_RE = re.compile(r"^\+[ACGT]+$")
MUT_DELETION_RE = re.compile(r"^Δ(?P<length>[\d,]+) bp$")
MUT_REPEAT_RE = re.compile(rf"^\([ACGT]+\)(?P<before>\d+){ARROW_RIGHT}(?P<after>\d+)$")
MUT_REPLACEMENT_RE = re.compile(rf"^\d+ bp{ARROW_RIGHT}(?:[ACGT]+|\d+ bp)$")
#: ``annotation`` prefixes, measured: 309 protein changes, 189 intergenic, 82 coding
#: offsets and 4 noncoding offsets.
ANNOTATION_INTERGENIC_PREFIX = "intergenic"


class CalledVariant(BaseModel):
    """One released row of data S1-5, and which perturbation leaf it becomes.

    Every field is the released cell verbatim or a typed reading of it. ``encoding``
    names the leaf the row is written as, decided by the row's own cells.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: str
    evidence: str
    position: int
    mutation: str
    annotation: str
    gene: str
    product: str
    variant_type: BacterialVariantType
    #: True where the released change spans more than one base, which is why the call's
    #: ``position_end`` is a typed absence rather than a computed coordinate.
    is_multibase: bool
    encoding: str
    #: the single released gene name, for an in-locus row.
    locus_statement: str | None
    #: the two released flanking names, left to right, for an intergenic row.
    flanking_statements: tuple[str, ...]


def classify_mutation(cell: str) -> tuple[BacterialVariantType, bool]:
    """``(variant type, spans more than one base)`` of one released ``mutation`` cell.

    Decided by the cell's own shape, with no default branch: an unseen form stops the
    build. A tandem-repeat cell (``(C)6→7``) is an insertion when the copy count
    rises and a deletion when it falls, which is what the two numbers state.
    """
    if MUT_SNV_RE.match(cell) is not None:
        return BacterialVariantType.snv, False
    if MUT_INSERTION_RE.match(cell) is not None:
        return BacterialVariantType.insertion, len(cell) > 2
    if (deletion := MUT_DELETION_RE.match(cell)) is not None:
        length = int(deletion.group("length").replace(",", ""))
        return BacterialVariantType.deletion, length > 1
    if (repeat := MUT_REPEAT_RE.match(cell)) is not None:
        before, after = int(repeat.group("before")), int(repeat.group("after"))
        if before == after:
            raise RuntimeError(
                f"tandem-repeat cell {cell!r} states no copy-number change"
            )
        kind = (
            BacterialVariantType.insertion
            if after > before
            else BacterialVariantType.deletion
        )
        return kind, abs(after - before) > 1
    if MUT_REPLACEMENT_RE.match(cell) is not None:
        return BacterialVariantType.substitution, True
    raise RuntimeError(
        f"data S1-5 mutation cell {cell!r} matches none of the five released forms "
        "(single-base substitution, insertion, delta-bp deletion, tandem-repeat "
        "copy-number change, multi-base replacement)"
    )


def read_variant_calls(path: str | Path) -> list[CalledVariant]:
    """Every row of data S1-5, typed, each carrying the leaf it is written as.

    The encoding is decided by the row's own cells, and the two branches are checked
    against each other rather than trusted: a ``gene`` cell naming two flanking loci
    must come with an ``annotation`` cell that begins ``intergenic``, and a cell naming
    one locus must not. Measured 2026-10-09 over all 584 rows: the agreement is exact,
    189 intergenic against 189 two-flanker cells.
    """
    rows = _sheet_rows(path, SHEET_WGS, WGS_HEADER_ROW, WGS_HEADER)
    calls: list[CalledVariant] = []
    for row in rows:
        strain = _norm(row[0])
        annotation = _norm(row[4])
        gene = _norm(row[5])
        mutation = _norm(row[3])
        variant_type, is_multibase = classify_mutation(mutation)
        is_intergenic = annotation.startswith(ANNOTATION_INTERGENIC_PREFIX)
        flankers = GENE_TWO_FLANKERS_RE.match(gene)
        single = GENE_ONE_LOCUS_RE.match(gene)
        if flankers is not None and single is None:
            if not is_intergenic:
                raise RuntimeError(
                    f"{strain} at {row[2]}: gene cell {gene!r} names two flanking loci "
                    f"while annotation {annotation!r} does not say intergenic"
                )
            encoding = ENCODING_INTERGENIC
            locus_statement = None
            flanking: tuple[str, ...] = (
                flankers.group("left").strip(),
                flankers.group("right").strip(),
            )
        elif single is not None:
            if is_intergenic:
                raise RuntimeError(
                    f"{strain} at {row[2]}: annotation {annotation!r} says intergenic "
                    f"while gene cell {gene!r} names one locus"
                )
            encoding = ENCODING_IN_LOCUS
            locus_statement = single.group("name")
            flanking = ()
        else:
            raise RuntimeError(
                f"{strain} at {row[2]}: gene cell {gene!r} is neither the released "
                "one-locus nor the released two-flanker form"
            )
        calls.append(
            CalledVariant(
                strain=strain,
                evidence=_norm(row[1]),
                position=int(row[2]),
                mutation=str(row[3]),
                annotation=str(row[4]),
                gene=str(row[5]),
                product=_norm(row[6]),
                variant_type=variant_type,
                is_multibase=is_multibase,
                encoding=encoding,
                locus_statement=locus_statement,
                flanking_statements=flanking,
            )
        )
    if not calls:
        raise RuntimeError("data S1-5 released no polymorphisms")
    return calls


#: Gaps every call of data S1-5 carries, because the sheet releases neither of them.
_CALL_GAP_FREQUENCY = (
    "data S1-5 releases no variant frequency and no coverage column: its seven columns "
    "are Sample, evidence, position, mutation, annotation, gene and description. breseq "
    "computes a frequency, but the deposited table does not carry it, so the call has "
    "no frequency and no basis"
)
#: Why every stored call has ``position_end == position_start``. This is a UNIFORM
#: convention, stated here and asserted by an L3 row, not a claim that every change is
#: one base long: 179 of the 584 released cells state a multi-base change.
_CALL_ONE_COORDINATE = (
    "data S1-5 releases ONE coordinate per call ('position') and no end coordinate, so "
    "position_end repeats position_start for every call and the released span stays "
    "verbatim in sequence_change (for example 'd1,227 bp') and in annotation (for "
    "example 'coding (1-1227/1227 nt)'). Deriving an end would assert breseq's "
    "coordinate convention for each of the five released change forms, and no mirrored "
    "byte states it. BacterialVariantCall.position_end is a required int, so the "
    "absence cannot be typed as a ProvenanceGap; it is stated here, in the note, and in "
    "the L3 row one_coordinate_per_call"
)


def variant_call(row: CalledVariant) -> BacterialVariantCall:
    """The typed call of one released row, with every cell it states kept verbatim."""
    gaps = [
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=_CALL_GAP_FREQUENCY,
        )
        for field in ("frequency_statement", "frequency", "frequency_basis")
    ]
    return BacterialVariantCall(
        variant_type=row.variant_type,
        type_statement=row.mutation,
        reference_sequence=str(WGS_REPLICON_SOURCED.value),
        position_start=row.position,
        position_end=row.position,
        sequence_change=row.mutation,
        annotation=row.annotation,
        call_mode=VariantCallMode.clone,
        caller=str(WGS_CALLER_SOURCED.value),
        provenance_gaps=gaps,
    )


def called_variant_perturbations(
    calls: Sequence[CalledVariant], strain: str, locus_of: Mapping[str, str]
) -> list[Any]:
    """One perturbation per released call of ``strain``, ordered by position.

    The order is explicit because ``Genotype.sort_perturbations`` sorts by systematic
    name, type and perturbed name, and two calls in one locus tie on all three; Python's
    sort is stable, so the input order decides and must be deterministic.

    An in-locus row becomes a :class:`BacterialSequenceVariantPerturbation` on the locus
    its ``gene`` cell names; an intergenic row becomes a
    :class:`BacterialSiteVariantPerturbation` keyed on its own site, carrying the two
    flanking loci as resolved tags and the released cell verbatim. The gene-containment
    exemption that follows is the site leaf's own: a site id is not a locus tag, so a
    gene-keyed consumer never reads the call as being inside a gene.
    """
    rows = sorted(
        (row for row in calls if row.strain == strain),
        key=lambda row: (row.position, row.mutation, row.gene),
    )
    perturbations: list[Any] = []
    for row in rows:
        call = variant_call(row)
        if row.encoding == ENCODING_IN_LOCUS:
            assert row.locus_statement is not None
            tag = locus_of.get(row.locus_statement)
            if tag is None:
                raise RuntimeError(
                    f"{strain} at {row.position}: data S1-5 names {row.locus_statement!r}"
                    " and no locus tag of the pinned assembly holds it; every one of "
                    "the 202 released single-locus names resolved on 2026-10-09, so "
                    "this is a new fact about the release or the annotation"
                )
            perturbations.append(
                BacterialSequenceVariantPerturbation(
                    systematic_gene_name=tag,
                    perturbed_gene_name=row.locus_statement,
                    gene_namespace=KT2440_NAMESPACE,
                    call=call,
                )
            )
            continue
        flanking = tuple(locus_of[name] for name in row.flanking_statements)
        perturbations.append(
            BacterialSiteVariantPerturbation(
                systematic_gene_name=BacterialSiteVariantPerturbation.site_id(call),
                perturbed_gene_name=", ".join(flanking),
                gene_namespace=KT2440_NAMESPACE,
                call=call,
                site_kind=VariantSiteKind.intergenic,
                flanking_systematic_gene_names=flanking,
                flanking_gene_statement=row.gene,
            )
        )
    return perturbations


class VariantLedger(BaseModel):
    """The counted summary of :func:`read_variant_calls`, written beside the build."""

    model_config = ConfigDict(extra="forbid")

    n_calls: int
    n_distinct_positions: int
    calls_per_clone: dict[str, int]
    encodings: dict[str, int]
    variant_types: dict[str, int]
    evidence: dict[str, int]
    #: clone -> the perturbations its calls were written as, for the clones that have a
    #: phenotype record under the SAME name. Empty for a refused clone.
    perturbations_per_strain: dict[str, int]
    #: the clones data S1-5 names that no phenotype sheet names, and why each is refused.
    refused_clones: dict[str, str]
    #: the largest released deletion, and whether the prose's 26.1-kb span is in here.
    largest_deletion_statement: str
    fleq_rows: int
    calls: list[CalledVariant]

    def check(self) -> None:
        """Every call is encoded, counted, and either written or refused by name."""
        if sum(self.encodings.values()) != self.n_calls:
            raise RuntimeError(
                f"{sum(self.encodings.values())} encodings over {self.n_calls} calls"
            )
        if sum(self.calls_per_clone.values()) != self.n_calls:
            raise RuntimeError(
                f"{sum(self.calls_per_clone.values())} clone rows over {self.n_calls} "
                "calls"
            )
        written = sum(self.perturbations_per_strain.values())
        refused = sum(
            self.calls_per_clone[clone]
            for clone in self.refused_clones
            if clone in self.calls_per_clone
        )
        if written + refused != self.n_calls:
            raise RuntimeError(
                f"{written} written + {refused} refused != {self.n_calls} calls"
            )
        if self.fleq_rows:
            raise RuntimeError(
                f"data S1-5 now holds {self.fleq_rows} fleQ/PP_4373 rows; the 26.1-kb "
                "span refusal was measured at 0 and must be re-read"
            )


def variant_ledger(
    calls: Sequence[CalledVariant],
    *,
    written: Mapping[str, int],
    refused: Mapping[str, str],
) -> VariantLedger:
    """Summarize data S1-5: what it holds, what was written and what was refused."""
    largest = max(
        (
            row
            for row in calls
            if MUT_DELETION_RE.match(_norm(row.mutation)) is not None
        ),
        key=lambda row: int(
            str(MUT_DELETION_RE.match(_norm(row.mutation)).group("length")).replace(  # type: ignore[union-attr]
                ",", ""
            )
        ),
        default=None,
    )
    return VariantLedger(
        n_calls=len(calls),
        n_distinct_positions=len({row.position for row in calls}),
        calls_per_clone=dict(sorted(Counter(row.strain for row in calls).items())),
        encodings=dict(sorted(Counter(row.encoding for row in calls).items())),
        variant_types=dict(
            sorted(Counter(str(row.variant_type) for row in calls).items())
        ),
        evidence=dict(sorted(Counter(row.evidence for row in calls).items())),
        perturbations_per_strain=dict(sorted(written.items())),
        refused_clones=dict(sorted(refused.items())),
        largest_deletion_statement=(
            _norm(largest.mutation) if largest is not None else "none"
        ),
        fleq_rows=sum(
            1 for row in calls if "fleQ" in row.gene or "PP_4373" in row.gene
        ),
        calls=list(calls),
    )


def _write_variant_ledger(ledger: VariantLedger, preprocess_dir: str) -> None:
    """Validate and write ``preprocess/called_variants.json``."""
    ledger.check()
    os.makedirs(preprocess_dir, exist_ok=True)
    path = osp.join(preprocess_dir, "called_variants.json")
    with open(path, "w") as handle:
        handle.write(ledger.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
# The designed strains of the deposited arms
# --------------------------------------------------------------------------- #
#: The verbatim part token of the fluorescent biosensor copy. mCherry's own source
#: organism is nowhere in the mirror, so it carries SOURCE_ORGANISM_UNREPORTED.
MCHERRY_TOKEN = "mCherry"
#: The mvaS copies TEAM-3174 overexpresses carry the SAME part token as the integrated
#: pathway's mvaS and are distinguished by their integration locus and their sourced
#: organism, which is what makes them extra copies rather than the same gene twice.
MVAS_TOKEN = "mvaS"

#: Chromosomal integrations per strain, as ``(token, promoter, locus)``. Sourced in
#: :data:`TABLE4_CLEAN_CHASSIS` (the loci), :data:`TABLE4_CONTROL_CHASSIS` (what
#: TEAM-2595 carries) and :data:`TABLE4_TEAM3174` (TEAM-3174's two mvaS loci).
STRAIN_INTEGRATIONS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "TEAM-2595": ((MCHERRY_TOKEN, "PpedF", "PP_5402intergenic"),),
    "TEAM-3174": (
        (MCHERRY_TOKEN, "PpedF", "PP_5402intergenic"),
        (MCHERRY_TOKEN, "PpedF", "PP_3159intergenic"),
        (MVAS_TOKEN, "Pcv", "PP_1117intergenic"),
        (MVAS_TOKEN, "Pcv", "PP_5464intergenic"),
    ),
    "TEAM-3185": (
        (MCHERRY_TOKEN, "PpedF", "PP_5402intergenic"),
        (MCHERRY_TOKEN, "PpedF", "PP_3159intergenic"),
    ),
}
#: Deletions per strain. TEAM-2595 has none; the two improved strains carry
#: TEAM-2777's two (:data:`TABLE4_TEAM2777_CHASSIS`) plus the four the Results sentence
#: states for both and their own fifth (:data:`IMPROVED_GENOTYPES`).
STRAIN_DELETIONS: dict[str, tuple[str, ...]] = {
    "TEAM-2595": (),
    "TEAM-3174": (
        "PP_2664",
        "PP_2675",
        "PP_2428",
        "PP_4622",
        "PP_3540",
        "PP_4373",
        "PP_2710",
    ),
    "TEAM-3185": (
        "PP_2664",
        "PP_2675",
        "PP_2428",
        "PP_4622",
        "PP_3540",
        "PP_4373",
        "PP_2074",
    ),
}
#: Every locus tag the three designed genotypes name, for the annotation lookup.
DESIGNED_STRAIN_TAGS: tuple[str, ...] = tuple(
    sorted({tag for tags in STRAIN_DELETIONS.values() for tag in tags})
)
#: Chassis elements of the three strains this module deliberately does NOT assert.
UNASSERTED_DESIGNED_CHASSIS: tuple[str, ...] = (
    "PJ23100-PP_2666,PP_2665 (all three strains) and PJ23119-PP_1697 (the two improved "
    "strains, via TEAM-2777): PromoterReplacementPerturbation requires "
    "expression_direction and no mirrored byte states a direction for either swap, so "
    "asserting one would state a consequence the source did not",
)


def mcherry_perturbation(locus: str) -> HeterologousPathwayPerturbation:
    """One integrated ``PpedF``-RBS-mCherry biosensor copy."""
    return HeterologousPathwayPerturbation(
        systematic_gene_name=MCHERRY_TOKEN,
        perturbed_gene_name=MCHERRY_TOKEN,
        gene_namespace=KT2440_NAMESPACE,
        pathway_name="PpedF-RBS-mCherry isoprenol biosensor",
        source_organism=SOURCE_ORGANISM_UNREPORTED,
        is_heterologous=True,
        localization="chromosomal_integration",
        integration_locus=locus,
        promoter_name="PpedF",
        copy_number=1.0,
    )


def mvas_overexpression_perturbation(locus: str) -> HeterologousPathwayPerturbation:
    """One extra integrated copy of ``Enterococcus faecalis`` ``mvaS`` (TEAM-3174)."""
    return HeterologousPathwayPerturbation(
        systematic_gene_name=MVAS_TOKEN,
        perturbed_gene_name=MVAS_TOKEN,
        gene_namespace=KT2440_NAMESPACE,
        pathway_name=str(PATHWAY_NAME.value),
        source_organism=str(MVAS_ORGANISM.value),
        is_heterologous=True,
        localization="chromosomal_integration",
        integration_locus=locus,
        promoter_name="Pcv",
        copy_number=1.0,
    )


def designed_perturbations(strain: str, common: Mapping[str, str]) -> list[Any]:
    """The whole stated genotype of one designed strain of the deposited arms.

    Every strain carries the five integrated isoprenol-pathway genes
    (:func:`pathway_perturbations`) and the ``PP_5402intergenic`` mCherry biosensor
    copy; the two improved strains add the second mCherry copy, TEAM-2777's two
    deletions, the four deletions the Results sentence states for both, and their own
    fifth, and TEAM-3174 adds its two ``E. faecalis`` ``mvaS`` copies. The two promoter
    replacements in the same Supplementary Table 4 rows are NOT here; see
    :data:`UNASSERTED_DESIGNED_CHASSIS`.
    """
    if strain not in STRAIN_INTEGRATIONS:
        raise RuntimeError(
            f"{strain} is not one of the three released strains "
            f"{list(RELEASED_STRAINS)}"
        )
    built: list[Any] = list(pathway_perturbations())
    for token, _, locus in STRAIN_INTEGRATIONS[strain]:
        if token == MCHERRY_TOKEN:
            built.append(mcherry_perturbation(locus))
        else:
            built.append(mvas_overexpression_perturbation(locus))
    for tag in STRAIN_DELETIONS[strain]:
        built.append(
            BacterialDeletionPerturbation(
                systematic_gene_name=tag,
                perturbed_gene_name=common.get(tag, tag),
                gene_namespace=KT2440_NAMESPACE,
            )
        )
    return built


# --------------------------------------------------------------------------- #
# The production culture both deposited arms were sampled from
# --------------------------------------------------------------------------- #
_PRODUCTION_TEMPERATURE_GAP = (
    "the production run's own temperature is not stated. 30 C is stated for the "
    "overnight LB culture, for the two M9 adaptation steps and for the conjugation "
    "spot, and is NOT restated for the production plate these samples come from, so "
    "carrying it over would be an inference"
)
_GROWTH_DURATION_GAP = (
    "the growth-phase sample is harvested by OPTICAL DENSITY, not by clock time: 'The "
    "log-phase samples were harvested when each strain reached an OD600 of 0.7 "
    "(roughly 8 to 14 hours postback dilution and induction)'. That is a 6 h range no "
    "single duration holds, and it differs per strain, so the duration is a typed "
    "absence while the harvest rule is recorded here"
)


def production_environment(phase: Phase) -> Environment:
    """The 24-deep-well M9 production culture, at one of its two harvest points.

    Growth phase and production phase are DIFFERENT environments and the difference is
    the harvest point the Methods state: OD600 0.7 for the log-phase sample and the
    24 h time point for the production-phase one. The medium, the inducer and the pH
    are shared; the temperature is a typed absence in both.
    """
    if phase not in PHASES:
        raise RuntimeError(f"{phase!r} is not one of {list(PHASES)}")
    if M9_NREL_MOPS_MENASALVAS2025.base_medium != "M9":
        raise RuntimeError(
            f"the served medium {M9_NREL_MOPS_MENASALVAS2025.name!r} is not an M9 "
            "derivative, so it is not the M9 minimal medium the production Methods name"
        )
    gaps = [
        ProvenanceGap(
            field="temperature",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=_PRODUCTION_TEMPERATURE_GAP,
        )
    ]
    if phase == "growth":
        gaps.append(
            ProvenanceGap(
                field="duration_hours",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=_GROWTH_DURATION_GAP,
            )
        )
    return Environment(
        media=M9_NREL_MOPS_MENASALVAS2025,
        temperature=None,
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound("crystal violet"),
                concentration=Concentration(
                    value=float(PRODUCTION_RUN.value), unit=ConcentrationUnit.micromolar
                ),
            ),
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.ph,
                magnitude=Concentration(
                    value=float(MEDIUM_PH.value), unit=ConcentrationUnit.ph
                ),
            ),
        ],
        aerobicity="aerobic",
        duration_hours=24.0 if phase == "production" else None,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# Phenotypes of the deposited arms
# --------------------------------------------------------------------------- #
_METABOLITE_SE_GAP = (
    "no SD and no SE is released for any metabolite value anywhere in the mirror. The "
    "sheet releases 18 columns -- the metabolite, its concentration class, six average "
    "concentrations, two fold changes, and the same six and two again normalized by "
    "OD600 -- and none of them is a dispersion or a per-record n (measured 2026-10-09: "
    "two merged banners C4:J4 and K4:R4, no hidden rows or columns, no cell comments). "
    "The replicate count IS released, per column, by the deposit README"
)


def metabolite_phenotype(
    levels: Mapping[str, float], *, n_replicates: int
) -> MetabolitePhenotype:
    """One strain-phase metabolite profile in micromolar, with no dispersion.

    ``metabolite_level_se`` is ``None`` with a typed gap rather than a zero map: the
    deposit releases an average and a replicate count and no dispersion at all.
    ``target_metabolite_ids`` is ``None``: those are Yeast9 ``s_NNNN`` ids and this is
    a P. putida dataset, so the mapping is deferred rather than invented.
    """
    return MetabolitePhenotype(
        metabolite_level=dict(levels),
        metabolite_level_se=None,
        n_replicates={key: n_replicates for key in levels},
        measurement_type=METABOLITE_MEASUREMENT_TYPE,
        target_metabolite_ids=None,
        provenance_gaps=[
            ProvenanceGap(
                field="metabolite_level_se",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=_METABOLITE_SE_GAP,
            )
        ],
    )


def proteome_phenotype(aggregate: ProteomeAggregate) -> ProteinAbundancePhenotype:
    """One proteome sample's profile: the per-locus mean of its released replicates."""
    return ProteinAbundancePhenotype(
        protein_abundance=aggregate.abundance,
        protein_abundance_se=aggregate.se,
        n_replicates=aggregate.n_replicates,
        measurement_type=PROTEOME_MEASUREMENT_TYPE,
    )


# --------------------------------------------------------------------------- #
# Shared build machinery for the deposited arms
# --------------------------------------------------------------------------- #
#: The clones data S1-5 names, mapped to the refusal that keeps their calls out of
#: every record, or ``None`` where the name is a phenotype sheet's name too.
WGS_CLONE_DISPOSITION: dict[str, str | None] = {
    "TEAM-2595": None,
    "TEAM-3175": (
        "data S1-5 names TEAM-3175 and no phenotype sheet does. The Results and every "
        "released data sheet name TEAM-3174, the shotgun-proteomics Methods name "
        "TEAM-3175 and TEAM-3184, and the data-availability statement names TEAM-3175 "
        "and TEAM-3185. No mirrored byte states that TEAM-3175 IS TEAM-3174, so its "
        "254 calls are not attached to any record"
    ),
    "TEAM-3184": (
        "data S1-5 names TEAM-3184 and no phenotype sheet does. Its 255 calls include "
        "a 924 bp deletion of the whole PP_2074 CDS, which Supplementary Table 4 "
        "states only for TEAM-3185, but no mirrored byte states that TEAM-3184 IS "
        "TEAM-3185, and the evidence is not even symmetric: TEAM-3175 carries no "
        "PP_2710 call, which Table 4 states only for TEAM-3174. The calls are not "
        "attached to any record"
    ),
}


class DepositBuild(BaseModel):
    """Everything the two deposited arms share, read once per build."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    data_1: Path
    data_2: Path


def _extract_for(dataset_root: str, raw_dir: str) -> DepositBuild:
    """Extract the two workbooks this module reads into ``<root>/preprocess/dryad``."""
    dest = osp.join(dataset_root, "preprocess", EXTRACT_SUBDIR)
    written = extract_dryad_members(osp.join(raw_dir, DRYAD_INNER_ZIP_FILENAME), dest)
    return DepositBuild(
        data_1=written[SI_DATA_1_MEMBER], data_2=written[SI_DATA_2_MEMBER]
    )


def _variant_perturbations_by_strain(
    data_1: Path, locus_of: Mapping[str, str]
) -> tuple[dict[str, list[Any]], VariantLedger]:
    """Called-variant perturbations for the clones a phenotype sheet also names.

    Only ``TEAM-2595`` qualifies: it is spelled identically in data S1-5 and in every
    phenotype sheet. ``TEAM-3175`` and ``TEAM-3184`` are refused by name, with the
    measurement in :data:`WGS_CLONE_DISPOSITION`.
    """
    calls = read_variant_calls(data_1)
    if len(calls) != EXPECTED_WGS_ROWS:
        raise RuntimeError(
            f"data S1-5 holds {len(calls)} rows, not the measured {EXPECTED_WGS_ROWS}"
        )
    per_clone = Counter(row.strain for row in calls)
    if dict(per_clone) != EXPECTED_WGS_ROWS_PER_CLONE:
        raise RuntimeError(
            f"data S1-5 names {dict(per_clone)}, not the measured "
            f"{EXPECTED_WGS_ROWS_PER_CLONE}"
        )
    unknown = sorted(set(per_clone) - set(WGS_CLONE_DISPOSITION))
    if unknown:
        raise RuntimeError(
            f"data S1-5 names {unknown}, which this module has no disposition for; a "
            "new clone name is a finding, not a silent drop"
        )
    attached = {
        strain: called_variant_perturbations(calls, strain, locus_of)
        for strain, refusal in WGS_CLONE_DISPOSITION.items()
        if refusal is None and strain in per_clone
    }
    refused = {
        strain: refusal
        for strain, refusal in WGS_CLONE_DISPOSITION.items()
        if refusal is not None
    }
    ledger = variant_ledger(
        calls,
        written={strain: len(rows) for strain, rows in attached.items()},
        refused=refused,
    )
    return attached, ledger


def _variant_locus_index(
    calls: Sequence[CalledVariant], genome: PPutidaKT2440Genome, *, label: str
) -> tuple[dict[str, str], LocusTagReconciliation]:
    """Every gene name data S1-5 states, resolved to a locus tag of the assembly.

    Both forms go through the same resolver: the single ``gene`` name of an in-locus
    row and the two flanking names of an intergenic one. Measured 2026-10-09: 202
    distinct single-locus names and 95 distinct flanking names, and all 297 resolve.
    """
    names = sorted(
        {row.locus_statement for row in calls if row.locus_statement is not None}
        | {name for row in calls for name in row.flanking_statements}
    )
    stored, report = reconcile_locus_tags(genome, pd.Series(names), label=label)
    index = dict(zip(names, stored, strict=True))
    outside = sorted(set(report.outside_namespace))
    if outside:
        raise RuntimeError(
            f"{label}: data S1-5 names {outside} and no locus tag of "
            f"{report.assembly_set} holds them; all 297 released names resolved on "
            "2026-10-09, so this is a new fact about the release or the annotation"
        )
    return index, report


# --------------------------------------------------------------------------- #
# The deposited proteome arm
# --------------------------------------------------------------------------- #
@register_dataset
class ProteomeMenasalvas2025Dataset(ExperimentDataset):
    """Menasalvas 2025 growth-phase and production-phase proteome of the three strains.

    One ``BacterialProteinAbundanceExperiment`` per released sample of Supplementary
    Data 2 sheet 4: the control TEAM-2595 and the two improved producers TEAM-3174 and
    TEAM-3185, each sampled in the growth phase and in the production phase. The
    reference of every record is the matching-phase TEAM-2595 profile, which is a real
    released sample rather than a synthetic baseline; the two TEAM-2595 records are
    therefore their own reference, which is what keeping every released sample as a
    record costs and is cheaper than dropping two samples.
    """

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"
    #: Host protein keys reaching a locus of the pinned assembly: 2,092 of 2,178,
    #: measured 2026-10-09. The floor sits just below it; the 86 that do not are listed
    #: in the build's drop rules.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 0.95

    def __init__(
        self,
        root: str = "data/torchcell/proteome_menasalvas2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome resolves the protein keys to loci."""
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
        """The deposited inner zip, the only raw artifact this dataset reads."""
        return [DRYAD_INNER_ZIP_FILENAME]

    def download(self) -> None:
        """Link the deposited inner zip into ``raw/`` after checking its manifest pin."""
        _link_mirror_files(
            self.raw_dir,
            ((DRYAD_INNER_ZIP_REL, DRYAD_INNER_ZIP_FILENAME, DRYAD_INNER_ZIP_SHA256),),
        )
        log.info("Menasalvas 2025 Dryad deposit linked into %s", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    @post_process
    def process(self) -> None:
        """Build one record per released proteome sample; write LMDB."""
        verify_raw_files(
            self.raw_dir, {DRYAD_INNER_ZIP_FILENAME: DRYAD_INNER_ZIP_SHA256}
        )
        extracted = _extract_for(self.root, self.raw_dir)
        rows = read_proteome_rows(extracted.data_2)
        genome = self._genome()
        resolution = resolve_protein_keys(rows, genome, label=self.name)
        resolution.reconciliation.require_resolved(self.MIN_RESOLVED_FRACTION)

        samples = sorted({row.sample for row in rows})
        by_key = {proteome_sample_key(sample): sample for sample in samples}
        if len(by_key) != len(samples):
            raise RuntimeError(f"{self.name}: two samples share a (strain, phase) key")
        expected_keys = {
            (strain, phase) for strain in RELEASED_STRAINS for phase in PHASES
        }
        if set(by_key) != expected_keys:
            raise RuntimeError(
                f"{self.name}: the sheet releases {sorted(by_key)}, not the measured "
                f"{sorted(expected_keys)}"
            )

        aggregates = {
            key: aggregate_proteome_sample(
                rows, sample, resolution.locus_of, label=self.name
            )
            for key, sample in by_key.items()
        }
        observed = {len(a.replicates) for a in aggregates.values()}
        if observed != {EXPECTED_PROTEOME_REPLICATES}:
            raise RuntimeError(
                f"{self.name}: the samples carry {sorted(observed)} replicates; the "
                f"sheet's own Replicate column holds {EXPECTED_PROTEOME_REPLICATES}"
            )

        common = _standard_names(
            genome, [*DESIGNED_STRAIN_TAGS, *sorted(set(resolution.locus_of.values()))]
        )
        variant_calls = read_variant_calls(extracted.data_1)
        locus_index, _ = _variant_locus_index(
            variant_calls, genome, label=f"{self.name}-variants"
        )
        attached, ledger = _variant_perturbations_by_strain(
            extracted.data_1, locus_index
        )
        reference_genome = host_reference()
        pub = publication()
        environments = {phase: production_environment(phase) for phase in PHASES}
        references = {
            phase: BacterialProteinAbundanceExperimentReference(
                dataset_name=self.name,
                genome_reference=reference_genome,
                environment_reference=environments[phase].model_copy(),
                phenotype_reference=proteome_phenotype(
                    aggregates[(CONTROL_STRAIN, phase)]
                ),
            )
            for phase in PHASES
        }

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        sample_rows: list[dict[str, Any]] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for (strain, phase), sample in tqdm(
                sorted(by_key.items()), desc="menasalvas2025-proteome"
            ):
                aggregate = aggregates[(strain, phase)]
                perturbations = [
                    *designed_perturbations(strain, common),
                    *attached.get(strain, []),
                ]
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=perturbations),
                    environment=environments[phase],
                    phenotype=proteome_phenotype(aggregate),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, references[phase], pub, itxn),
                )
                sample_rows.append(
                    {
                        "sample": sample,
                        "strain": strain,
                        "phase": phase,
                        "n_replicates": len(aggregate.replicates),
                        "n_proteins": len(aggregate.abundance),
                        "n_perturbations": len(perturbations),
                        "n_called_variants": len(attached.get(strain, [])),
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(sample_rows).to_csv(
            osp.join(self.preprocess_dir, "proteome_samples.csv"), index=False
        )
        _write_variant_ledger(ledger, self.preprocess_dir)
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(rows),
                candidate_records=len(by_key),
                kept_records=idx,
                dropped_records=len(by_key) - idx,
                distinct_targets=len(set(resolution.locus_of.values())),
                rules=_proteome_drop_rules(resolution),
                reconciliation=resolution.reconciliation,
                notes=[
                    f"no SAMPLE is dropped: all {len(by_key)} released samples of "
                    f"{SHEET_PROTEOME!r} are records, and the two TEAM-2595 records "
                    "are the matching-phase reference of every record including their "
                    "own",
                    "the record's number is the per-sample MEAN of the released "
                    f"Counts_sum replicates, named {PROTEOME_MEASUREMENT_TYPE!r} so it "
                    "is never pooled with Carruthers 2025's Top3 peptide signal or its "
                    "percent-of-proteome",
                    "growth phase and production phase are different environments: the "
                    f"harvest points are {dict(PROTEOME_HARVEST.value)} and the "
                    "growth-phase duration is a typed absence because its 'roughly 8 "
                    "to 14 hours' is a range",
                    f"data S1-5 attaches {len(attached.get(CONTROL_STRAIN, []))} called "
                    f"variants to each TEAM-2595 record and NOTHING to the other four: "
                    + "; ".join(sorted(ledger.refused_clones)),
                    "no BacterialSpanDeletionPerturbation is written: "
                    + str(FLEQ_SPAN_NOT_IN_DATA.note),
                    "chassis elements NOT asserted: "
                    + "; ".join(UNASSERTED_DESIGNED_CHASSIS),
                    "a finding recorded and not acted on: "
                    + str(PATHWAY_ENZYME_ORGANISMS.note),
                    str(SHEET_5_DEFERRAL.note),
                ],
            ),
            self.preprocess_dir,
        )

        log.info(
            "Menasalvas2025 proteome: %d records over %d loci (%d replicates each)",
            idx,
            len(set(resolution.locus_of.values())),
            EXPECTED_PROTEOME_REPLICATES,
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def _proteome_drop_rules(resolution: ProteinKeyResolution) -> list[DropRule]:
    """The three measured key-scope drop rules of the proteome build.

    Every rule carries ``n_records=0``: no SAMPLE is dropped, only measurement KEYS, so
    each record keeps its place with a smaller abundance map.
    """
    return [
        DropRule(
            rule="protein_key_is_not_a_host_protein",
            scope="protein_key",
            description=(
                "the key's Protein.Names organism code is not "
                f"{HOST_ORGANISM_CODE}, so no locus of {KT2440_STRAIN} holds it. Five "
                "are the mevalonate-pathway enzymes the DIA-NN search database was "
                "built to include and four are search-database contaminants (three "
                "human keratins and the pig trypsin the Methods digest with). The "
                "sheet never says which pIY670 part token each pathway enzyme is, so "
                "none is keyed to one: " + str(PATHWAY_ENZYME_ORGANISMS.note)
            ),
            n_records=0,
            items=sorted([*resolution.pathway_enzymes, *resolution.contaminants]),
        ),
        DropRule(
            rule="protein_key_merges_two_protein_groups",
            scope="protein_key",
            description=(
                "the sheet files two distinct Protein.Group accessions under one "
                "Protein key, so the key names two protein groups and its abundance "
                "cannot be attributed to either"
            ),
            n_records=0,
            items=list(resolution.merged_keys),
        ),
        DropRule(
            rule="protein_key_is_not_a_locus_of_the_pinned_assembly",
            scope="protein_key",
            description=(
                "the key is a host protein whose released name does not resolve to a "
                f"locus tag of {resolution.reconciliation.assembly_set}: a retired "
                "gene symbol, or an ambiguous one that names two loci. It has no gene "
                "node to key an abundance to"
            ),
            n_records=0,
            items=list(resolution.outside_namespace),
        ),
    ]


# --------------------------------------------------------------------------- #
# The deposited metabolite arm, one dataset per growth stage
# --------------------------------------------------------------------------- #
class _MetabolitePhaseDataset(ExperimentDataset):
    """Base of the two phase datasets of data S1-1; ``PHASE`` picks the block read.

    The two phases are separate datasets rather than one dataset of six records because
    ``verify_metabolite_dataset``'s L1 keys a record on its STRAIN and has no relaxation
    for a dataset with two records per strain, which one dataset of both phases would
    have. Splitting by phase keeps that gate meaningful instead of defeating it, and the
    two phases genuinely are two environments.
    """

    PHASE: ClassVar[Phase]
    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"
    #: Declared here because the concrete subclasses own ``__init__``.
    pputida_genome: PPutidaKT2440Genome | None

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialMetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialMetaboliteExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The deposited inner zip, the only raw artifact this dataset reads."""
        return [DRYAD_INNER_ZIP_FILENAME]

    def download(self) -> None:
        """Link the deposited inner zip into ``raw/`` after checking its manifest pin."""
        _link_mirror_files(
            self.raw_dir,
            ((DRYAD_INNER_ZIP_REL, DRYAD_INNER_ZIP_FILENAME, DRYAD_INNER_ZIP_SHA256),),
        )
        log.info("Menasalvas 2025 Dryad deposit linked into %s", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    @post_process
    def process(self) -> None:
        """Build one record per released strain of this phase; write LMDB."""
        verify_raw_files(
            self.raw_dir, {DRYAD_INNER_ZIP_FILENAME: DRYAD_INNER_ZIP_SHA256}
        )
        extracted = _extract_for(self.root, self.raw_dir)
        rows, footnote = read_metabolite_rows(extracted.data_1)
        absolute = [
            row for row in rows if row.concentration_class == CONCENTRATION_ABSOLUTE
        ]
        relative = [
            row for row in rows if row.concentration_class == CONCENTRATION_RELATIVE
        ]
        genome = self._genome()
        common = _standard_names(genome, DESIGNED_STRAIN_TAGS)
        variant_calls = read_variant_calls(extracted.data_1)
        locus_index, _ = _variant_locus_index(
            variant_calls, genome, label=f"{self.name}-variants"
        )
        attached, ledger = _variant_perturbations_by_strain(
            extracted.data_1, locus_index
        )

        levels: dict[str, dict[str, float]] = {}
        absent: dict[str, list[str]] = {}
        for strain in RELEASED_STRAINS:
            measured: dict[str, float] = {}
            missing: list[str] = []
            for row in absolute:
                value = row.average_um.get((strain, self.PHASE))
                if value is None:
                    missing.append(row.metabolite)
                else:
                    measured[row.metabolite] = value
            if not measured:
                raise RuntimeError(
                    f"{self.name}: {strain} measures no Absolute metabolite in the "
                    f"{self.PHASE} phase"
                )
            levels[strain] = measured
            absent[strain] = missing

        n_replicates = int(METABOLITE_N_REPLICATES.value)
        control = levels[CONTROL_STRAIN]
        reference_genome = host_reference()
        environment = production_environment(self.PHASE)
        pub = publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        strain_rows: list[dict[str, Any]] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for strain in tqdm(
                RELEASED_STRAINS, desc=f"menasalvas2025-metabolite-{self.PHASE}"
            ):
                measured = levels[strain]
                shared = {key: control[key] for key in measured if key in control}
                if not shared:
                    raise RuntimeError(
                        f"{self.name}: {strain} shares no metabolite with "
                        f"{CONTROL_STRAIN}, so the control cannot reference it"
                    )
                perturbations = [
                    *designed_perturbations(strain, common),
                    *attached.get(strain, []),
                ]
                experiment = BacterialMetaboliteExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=perturbations),
                    environment=environment,
                    phenotype=metabolite_phenotype(measured, n_replicates=n_replicates),
                )
                reference = BacterialMetaboliteExperimentReference(
                    dataset_name=self.name,
                    genome_reference=reference_genome,
                    environment_reference=environment.model_copy(),
                    phenotype_reference=metabolite_phenotype(
                        shared, n_replicates=n_replicates
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                strain_rows.append(
                    {
                        "strain": strain,
                        "phase": self.PHASE,
                        "n_metabolites": len(measured),
                        "n_absent_cells": len(absent[strain]),
                        "absent": ";".join(absent[strain]),
                        "n_perturbations": len(perturbations),
                        "n_called_variants": len(attached.get(strain, [])),
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(strain_rows).to_csv(
            osp.join(self.preprocess_dir, "metabolite_strains.csv"), index=False
        )
        _write_variant_ledger(ledger, self.preprocess_dir)
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(rows),
                candidate_records=len(RELEASED_STRAINS),
                kept_records=idx,
                dropped_records=len(RELEASED_STRAINS) - idx,
                distinct_targets=len(absolute),
                rules=[
                    DropRule(
                        rule="relative_concentration_is_on_another_scale",
                        scope="metabolite_row",
                        description=(
                            "the sheet's own class footnote, which the deposit README "
                            f"repeats: {footnote!r}. A Relative value came off a single "
                            "chemical standard rather than a standard curve of the "
                            "authentic analyte, so it is not a micromolar concentration "
                            "on the same scale as the Absolute rows and storing the two "
                            "together would mix two assays"
                        ),
                        n_records=0,
                        items=sorted(row.metabolite for row in relative),
                    ),
                    DropRule(
                        rule="metabolite_cell_is_blank_for_this_strain_and_phase",
                        scope="metabolite_row",
                        description=(
                            "a blank cell is an ABSENT measurement, so the metabolite "
                            "carries no key for that record. A released 0 is a present "
                            "measurement and IS stored"
                        ),
                        n_records=0,
                        items=sorted(
                            f"{strain}:{name}"
                            for strain, names in absent.items()
                            for name in names
                        ),
                    ),
                ],
                notes=[
                    f"one record per released strain of the {self.PHASE} phase; the "
                    f"reference is {CONTROL_STRAIN} of the SAME phase, restricted to "
                    "the metabolites that record also measures",
                    f"n_replicates is {n_replicates} for every metabolite, from the "
                    "deposit README's own column note; the article states it too, in "
                    "the Fig. 7E caption, which the MinerU OCR of the article dropped",
                    "no SD and no SE is released, so metabolite_level_se is None with "
                    "a typed gap: " + str(METABOLITE_UNCERTAINTY_ABSENT.note),
                    "the stored block is the average concentration, not the specific "
                    "one: " + str(METABOLITE_SPECIFIC_DERIVED.note),
                    f"data S1-5 attaches {len(attached.get(CONTROL_STRAIN, []))} called "
                    "variants to the TEAM-2595 record and NOTHING to the other two: "
                    + "; ".join(sorted(ledger.refused_clones)),
                    "chassis elements NOT asserted: "
                    + "; ".join(UNASSERTED_DESIGNED_CHASSIS),
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Menasalvas2025 metabolite (%s): %d records over %d Absolute metabolites",
            self.PHASE,
            idx,
            len(absolute),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


@register_dataset
class MetaboliteGrowthPhaseMenasalvas2025Dataset(_MetabolitePhaseDataset):
    """data S1-1's growth-phase (``GP``) absolute metabolite concentrations."""

    PHASE: ClassVar[Phase] = "growth"

    def __init__(
        self,
        root: str = "data/torchcell/metabolite_growth_menasalvas2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome names the deleted loci of each genotype."""
        self.pputida_genome = pputida_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)


@register_dataset
class MetaboliteProductionPhaseMenasalvas2025Dataset(_MetabolitePhaseDataset):
    """data S1-1's production-phase (``PP``) absolute metabolite concentrations."""

    PHASE: ClassVar[Phase] = "production"

    def __init__(
        self,
        root: str = "data/torchcell/metabolite_production_menasalvas2025",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome names the deleted loci of each genotype."""
        self.pputida_genome = pputida_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)


# --------------------------------------------------------------------------- #
# Verification of the deposited arms
# --------------------------------------------------------------------------- #
#: The harvest OD600 the growth-phase ``Specific Concentration`` block recovers
#: EXACTLY, measured 2026-10-09 over the 48 Absolute rows: every metabolite's
#: average/specific ratio agrees with these to within the 0.01 rounding of the average
#: column. The production-phase block does NOT recover a single OD600, which is why its
#: disagreements are pinned below instead.
RECOVERED_HARVEST_OD600: dict[tuple[str, str], float] = {
    ("TEAM-2595", "growth"): 2.4,
    ("TEAM-3174", "growth"): 1.2,
    ("TEAM-3185", "growth"): 2.7,
}
#: Absolute metabolites whose ``Specific Concentration`` cell is NOT the ``Average
#: Concentration`` cell divided by that strain-phase's single OD600, pinned EXACTLY
#: rather than under a tolerance: each is a named inconsistency in the released sheet,
#: and a new one would be a new fact about the release.
SPECIFIC_BLOCK_DISAGREEMENTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("TEAM-2595", "growth"): (),
    ("TEAM-3174", "growth"): (),
    ("TEAM-3185", "growth"): (),
    ("TEAM-2595", "production"): (
        "2-Methylcitrate",
        "ADP",
        "Citrate",
        "Malonate",
        "NADH",
        "Pyruvate",
    ),
    ("TEAM-3174", "production"): ("Methylmalonate",),
    ("TEAM-3185", "production"): ("NAD", "NADH"),
}
#: Loci Supplementary Note 2 names as measured in the proteomics dataset. Every one is
#: a key of every stored record, measured 2026-10-09. ``PP_4622``/``hmgR`` is NOT here:
#: Note 2 names it as the TARGETED DELETION behind the hmgABC change, not as a measured
#: protein, and it is absent from the sheet's resolved key set.
NOTE2_MEASURED_LOCI: tuple[str, ...] = (
    "PP_1816",
    "PP_2088",
    "PP_3511",
    "PP_3839",
    "PP_4401",
    "PP_4403",
    "PP_4619",
    "PP_4620",
    "PP_4621",
    "PP_5210",
)
#: ``PP_2088``/``SigX`` production-phase fold change over the control, as Supplementary
#: Note 2 states it per improved strain.
NOTE2_SIGX_FOLD: dict[str, float] = {"TEAM-3185": 34.0, "TEAM-3174": 20.0}
#: Relative tolerance on that fold. The SI states two ROUNDED integers computed on the
#: authors' own normalized intensities, while the store holds the ratio of the MEANS of
#: the released Counts_sum replicates, so the two statistics do not have to agree to
#: the digit. Measured 2026-10-09: 31.895 against 34 (6.2%) and 18.555 against 20
#: (7.2%), and the named statistic is the ratio of the per-sample means.
NOTE2_SIGX_FOLD_TOL = 0.10
_Q_NOTE2_SIGX = (
    "This gene (PP_2088, SigX, RNA polymerase sigma-70 factor) was upregulated during "
    "the production phase (34-fold in TEAM-3185 and 20-fold in TEAM-3174 (Figure 7B, "
    "Figure 7C), suggesting a nitrogen starvation response (56)."
)
_PAGE_SI_NOTE2 = (
    "Supplementary Note 2: Proteomics analysis of TEAM-3174 and TEAM-3185 compared to "
    "the starting strain TEAM-2595"
)
NOTE2_SIGX = _si(
    dict(NOTE2_SIGX_FOLD),
    _Q_NOTE2_SIGX,
    page=_PAGE_SI_NOTE2,
    note="the one per-protein number the Supplementary Information states for this "
    "proteome, used as the cross-source oracle of the built store. It lives in the "
    "Supplementary Information OCR, a DIFFERENT released file from the Dryad workbook "
    "the loader read",
)
#: The three metabolites the Results cite data S1-1 for and the sheet does not release.
#: Measured 2026-10-09: of the 58 released rows the only amino-acid-adjacent ones are
#: ``4-Aminobutyric acid`` and ``Glutamate``.
PROSE_METABOLITES_NOT_RELEASED: tuple[str, ...] = (
    "leucine",
    "phenylalanine",
    "tryptophan",
)
_Q_PROSE_AMINO_ACIDS = (
    "We observed a 2-fold increase in phenylalanine and a 15-fold increase in leucine "
    "concentrations comparing TEAM-3174 and TEAM-3185 to the base strain (Fig. 7E and "
    "data S1-1). In contrast, tryptophan concentrations showed inconsistent changes "
    "between the two producers as TEAM-3185 had no change whereas TEAM-3174 showed a "
    "$7 \\mathbf { x }$ decrease (Fig. 7E and data S1-1)."
)
PROSE_AMINO_ACIDS = _paper(
    PROSE_METABOLITES_NOT_RELEASED,
    _Q_PROSE_AMINO_ACIDS,
    page=_PAGE_RESULTS_IMPROVED,
    note="the Results cite data S1-1 for three amino-acid concentrations and data S1-1 "
    "releases none of them, so the claim is not sourceable from the deposit. The L4 "
    "row asserts exactly that, pinned: a released row for any of the three would be a "
    "new fact and must force a re-read",
)

#: Deletion signature -> released strain name, for reading a record's strain off its
#: own genotype. The signatures are disjoint by construction and the check asserts it.
STRAIN_OF_DELETION_SET: dict[frozenset[str], str] = {
    frozenset(tags): strain for strain, tags in STRAIN_DELETIONS.items()
}


def _record_strain(record: Mapping[str, Any]) -> str:
    """The released strain one record is of, read off its deletion set."""
    deleted = frozenset(
        str(p["systematic_gene_name"])
        for p in record["experiment"]["genotype"]["perturbations"]
        if str(p["perturbation_type"]) == "bacterial_deletion"
    )
    strain = STRAIN_OF_DELETION_SET.get(deleted)
    if strain is None:
        raise RuntimeError(
            f"a record's deletion set {sorted(deleted)} is none of the three released "
            f"strains' { ({s: sorted(t) for s, t in STRAIN_DELETIONS.items()}) }"
        )
    return strain


def _record_phase(record: Mapping[str, Any]) -> str:
    """The growth stage one record is of, read off its environment's duration.

    The production-phase harvest is the stated 24 h time point and the growth-phase one
    is a typed absence, so the two environments are distinguishable without a label.
    """
    duration = record["experiment"]["environment"].get("duration_hours")
    return "production" if duration is not None else "growth"


def _l1_proteome_sample_partition(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L1: exactly one record per (released strain, growth phase)."""
    keys = Counter((_record_strain(r), _record_phase(r)) for r in records)
    expected = {(strain, phase) for strain in RELEASED_STRAINS for phase in PHASES}
    duplicates = sorted(key for key, n in keys.items() if n > 1)
    passed = set(keys) == expected and not duplicates
    return LevelResult(
        level=Level.L1,
        name="one_record_per_strain_and_growth_phase",
        passed=passed,
        message=(
            f"{len(keys)} (strain, phase) keys over {len(records)} records; "
            f"{len(duplicates)} repeated, {len(expected - set(keys))} missing"
        ),
        details={
            "keys": sorted(f"{s}:{p}" for s, p in keys),
            "duplicated": [f"{s}:{p}" for s, p in duplicates],
            "missing": sorted(f"{s}:{p}" for s, p in expected - set(keys)),
        },
    )


def _l3_proteome_keys_are_one_locus_set(
    records: Sequence[dict[str, Any]],
) -> LevelResult:
    """L3: every record measures the SAME set of KT2440 locus tags, and only those."""
    key_sets = {
        frozenset(r["experiment"]["phenotype"]["protein_abundance"]) for r in records
    }
    shared = next(iter(key_sets)) if len(key_sets) == 1 else frozenset()
    not_a_tag = sorted(key for key in shared if LOCUS_TAG_RE.fullmatch(key) is None)
    passed = len(key_sets) == 1 and not not_a_tag
    return LevelResult(
        level=Level.L3,
        name="every_record_measures_the_same_kt2440_locus_set",
        passed=passed,
        message=(
            f"{len(key_sets)} distinct key set(s) over {len(records)} records, "
            f"{len(shared)} loci, {len(not_a_tag)} of them not a PP_ tag"
        ),
        details={
            "n_key_sets": len(key_sets),
            "n_loci": len(shared),
            "not_a_locus_tag": not_a_tag[:20],
        },
    )


def _l3_called_variants_only_on_the_control(
    records: Sequence[dict[str, Any]],
) -> LevelResult:
    """L3: called variants sit on the TEAM-2595 records and on no other.

    The refusal in :data:`WGS_CLONE_DISPOSITION` is a property of the written store, so
    it is checked there rather than only asserted in the loader: data S1-5's
    TEAM-3175 and TEAM-3184 calls must reach no record.
    """
    per_strain: dict[str, set[int]] = defaultdict(set)
    for record in records:
        strain = _record_strain(record)
        per_strain[strain].add(
            sum(
                1
                for p in record["experiment"]["genotype"]["perturbations"]
                if str(p["perturbation_type"])
                in ("bacterial_sequence_variant", "bacterial_site_variant")
            )
        )
    control = per_strain.get(CONTROL_STRAIN, set())
    others = {
        strain: sorted(counts)
        for strain, counts in per_strain.items()
        if strain != CONTROL_STRAIN and counts != {0}
    }
    expected = EXPECTED_WGS_ROWS_PER_CLONE[CONTROL_STRAIN]
    passed = control == {expected} and not others
    return LevelResult(
        level=Level.L3,
        name="called_variants_attach_only_to_the_identically_named_clone",
        passed=passed,
        message=(
            f"{CONTROL_STRAIN} records carry {sorted(control)} called variants "
            f"(expected {expected}); {len(others)} other strain(s) carry any"
        ),
        details={
            "control_counts": sorted(control),
            "expected": expected,
            "other_strains_with_variants": others,
            "refused": sorted(
                k for k, v in WGS_CLONE_DISPOSITION.items() if v is not None
            ),
        },
    )


def _l3_one_coordinate_per_call(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L3: every stored call has ``position_end == position_start``.

    data S1-5 releases one coordinate per call, so this is the written store's uniform
    convention and not a claim that every change is one base long. The row exists so a
    consumer reading an interval off these calls cannot mistake the convention for a
    measurement; the released span is in ``sequence_change`` and ``annotation``.
    """
    checked = 0
    wrong: list[dict[str, Any]] = []
    for record in records:
        for perturbation in record["experiment"]["genotype"]["perturbations"]:
            if str(perturbation["perturbation_type"]) not in (
                "bacterial_sequence_variant",
                "bacterial_site_variant",
            ):
                continue
            call = perturbation["call"]
            checked += 1
            if int(call["position_start"]) != int(call["position_end"]):
                wrong.append(
                    {
                        "start": call["position_start"],
                        "end": call["position_end"],
                        "change": call["sequence_change"],
                    }
                )
    return LevelResult(
        level=Level.L3,
        name="one_coordinate_per_call",
        passed=not wrong,
        message=(
            f"{checked} stored calls carry the release's single coordinate; "
            f"{len(wrong)} do not. {_CALL_ONE_COORDINATE}"
        ),
        details={"n_calls": checked, "worst": wrong[:10]},
    )


def _l4_genotypes_vs_supplementary_table_4(
    records: Sequence[dict[str, Any]],
) -> LevelResult:
    """L4 cross-source: each stored deletion set is Supplementary Table 4's own.

    The loader read the Dryad workbook; this row joins the written genotypes to the
    Supplementary Information OCR, a different released file, through Table 4's strain
    rows plus the Results sentence that states the same sets independently.
    """
    expected = {
        "TEAM-2595": frozenset(),
        "TEAM-3174": frozenset(
            (
                *TABLE4_TEAM2777_CHASSIS.value,
                *dict(IMPROVED_GENOTYPES.value)["shared"],
                *dict(IMPROVED_GENOTYPES.value)["TEAM-3174"],
            )
        ),
        "TEAM-3185": frozenset(
            (*TABLE4_TEAM2777_CHASSIS.value, *TABLE4_TEAM3185.value)
        ),
    }
    disagreements: list[dict[str, Any]] = []
    for record in records:
        strain = _record_strain(record)
        stored = frozenset(
            str(p["systematic_gene_name"])
            for p in record["experiment"]["genotype"]["perturbations"]
            if str(p["perturbation_type"]) == "bacterial_deletion"
        )
        if stored != expected[strain]:
            disagreements.append(
                {
                    "strain": strain,
                    "stored": sorted(stored),
                    "table_4": sorted(expected[strain]),
                }
            )
    return LevelResult(
        level=Level.L4,
        name="stored_deletion_sets_are_supplementary_table_4s",
        passed=not disagreements,
        message=(
            f"{len(records)} records checked against Supplementary Table 4's strain "
            f"rows; {len(disagreements)} disagree"
        ),
        details={
            "n_records": len(records),
            "expected": {k: sorted(v) for k, v in expected.items()},
            "worst": disagreements[:6],
        },
    )


def _l4_note2_loci_are_measured(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L4 cross-source: every locus Supplementary Note 2 names is a stored key."""
    absent: dict[str, list[str]] = {}
    for record in records:
        keys = set(record["experiment"]["phenotype"]["protein_abundance"])
        missing = sorted(set(NOTE2_MEASURED_LOCI) - keys)
        if missing:
            absent[f"{_record_strain(record)}:{_record_phase(record)}"] = missing
    return LevelResult(
        level=Level.L4,
        name="supplementary_note_2_loci_are_measured_in_every_record",
        passed=not absent,
        message=(
            f"{len(NOTE2_MEASURED_LOCI)} loci Supplementary Note 2 names as measured; "
            f"{len(absent)} of {len(records)} records miss any"
        ),
        details={"loci": list(NOTE2_MEASURED_LOCI), "missing_by_record": absent},
    )


def _l4_note2_sigx_fold(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L4 cross-source: ``PP_2088``'s production-phase fold is Note 2's 34x and 20x.

    The statistic is the ratio of the per-sample MEANS of the released ``Counts_sum``
    replicates; Note 2 states two rounded integers computed on the authors' own
    normalized intensities, so the row holds to a declared relative tolerance of
    :data:`NOTE2_SIGX_FOLD_TOL` and names both numbers in its details.
    """
    production = {
        _record_strain(record): record["experiment"]["phenotype"]["protein_abundance"]
        for record in records
        if _record_phase(record) == "production"
    }
    control = production.get(CONTROL_STRAIN, {}).get("PP_2088")
    observed: dict[str, float] = {}
    disagreements: list[dict[str, Any]] = []
    if control:
        for strain, stated in NOTE2_SIGX_FOLD.items():
            measured = production.get(strain, {}).get("PP_2088")
            if measured is None:
                disagreements.append({"strain": strain, "stored": None})
                continue
            fold = float(measured) / float(control)
            observed[strain] = fold
            if abs(fold - stated) / stated > NOTE2_SIGX_FOLD_TOL:
                disagreements.append(
                    {
                        "strain": strain,
                        "stored_fold": fold,
                        "note_2_fold": stated,
                        "relative": abs(fold - stated) / stated,
                    }
                )
    else:
        disagreements.append({"strain": CONTROL_STRAIN, "stored": control})
    return LevelResult(
        level=Level.L4,
        name="pp_2088_production_fold_is_supplementary_note_2s",
        passed=not disagreements,
        message=(
            "PP_2088 production-phase fold over "
            f"{CONTROL_STRAIN}: stored "
            f"{ {k: round(v, 3) for k, v in sorted(observed.items())} }, Note 2 states "
            f"{NOTE2_SIGX_FOLD} (relative tolerance {NOTE2_SIGX_FOLD_TOL})"
        ),
        details={
            "statistic": "ratio of the per-sample means of the released Counts_sum",
            "stored": observed,
            "note_2": dict(NOTE2_SIGX_FOLD),
            "tol": NOTE2_SIGX_FOLD_TOL,
            "worst": disagreements[:4],
        },
    )


def _l1_metabolite_strains(records: Sequence[dict[str, Any]]) -> LevelResult:
    """L1: exactly one record per released strain, all three present."""
    keys = Counter(_record_strain(record) for record in records)
    duplicates = sorted(strain for strain, n in keys.items() if n > 1)
    missing = sorted(set(RELEASED_STRAINS) - set(keys))
    return LevelResult(
        level=Level.L1,
        name="one_record_per_released_strain",
        passed=not duplicates and not missing,
        message=(
            f"{len(keys)} strains over {len(records)} records; "
            f"{len(duplicates)} repeated, {len(missing)} missing"
        ),
        details={"strains": sorted(keys), "duplicated": duplicates, "missing": missing},
    )


def _l3_metabolite_keys_are_absolute_rows(
    records: Sequence[dict[str, Any]], data_1: Path
) -> LevelResult:
    """L3: every stored metabolite is an ``Absolute`` row of data S1-1, and none is
    a ``Relative`` one.
    """
    rows, _ = read_metabolite_rows(data_1)
    allowed = {
        row.metabolite
        for row in rows
        if row.concentration_class == CONCENTRATION_ABSOLUTE
    }
    forbidden = {
        row.metabolite
        for row in rows
        if row.concentration_class == CONCENTRATION_RELATIVE
    }
    stored: set[str] = set()
    for record in records:
        stored |= set(record["experiment"]["phenotype"]["metabolite_level"])
    unknown = sorted(stored - allowed)
    leaked = sorted(stored & forbidden)
    return LevelResult(
        level=Level.L3,
        name="every_stored_metabolite_is_an_absolute_row",
        passed=not unknown and not leaked,
        message=(
            f"{len(stored)} stored metabolites over {len(allowed)} Absolute rows; "
            f"{len(unknown)} not released, {len(leaked)} from the {len(forbidden)} "
            "Relative rows"
        ),
        details={
            "n_stored": len(stored),
            "n_absolute": len(allowed),
            "not_released": unknown[:20],
            "relative_leaked": leaked[:20],
        },
    )


def _l4_prose_amino_acids_are_not_released(
    records: Sequence[dict[str, Any]],
) -> LevelResult:
    """L4 cross-source: the three metabolites the Results cite data S1-1 for are absent.

    The loader read the Dryad workbook; this row joins the written store to the
    article's own prose and asserts the measured disagreement between them, pinned. A
    released row for any of the three would make the row fail and force a re-read.
    """
    stored: set[str] = set()
    for record in records:
        stored |= {
            key.lower() for key in record["experiment"]["phenotype"]["metabolite_level"]
        }
    present = sorted(name for name in PROSE_METABOLITES_NOT_RELEASED if name in stored)
    return LevelResult(
        level=Level.L4,
        name="results_prose_amino_acids_are_not_in_data_s1_1",
        passed=not present,
        message=(
            f"the Results cite data S1-1 for {list(PROSE_METABOLITES_NOT_RELEASED)}; "
            f"{len(present)} of the three are stored, measured 0 on 2026-10-09"
        ),
        details={
            "prose_metabolites": list(PROSE_METABOLITES_NOT_RELEASED),
            "stored": present,
            "n_stored_metabolites": len(stored),
            "quote": PROSE_AMINO_ACIDS.quote,
        },
    )


def _l4_specific_block_recovers_the_harvest_od600(
    records: Sequence[dict[str, Any]], data_1: Path, phase: str
) -> LevelResult:
    """L4: the released ``Specific`` block, which no loader reads, against the store.

    The deposit README states the second block is the first "normalized against the
    OD600 at the time of sample harvest", so dividing a stored level by its released
    specific value must give ONE number per strain-phase. In the growth phase it does,
    exactly, and that number is the harvest OD600 the deposit never states as such
    (:data:`RECOVERED_HARVEST_OD600`). In the production phase it does not for a pinned
    set of metabolites (:data:`SPECIFIC_BLOCK_DISAGREEMENTS`), which is a property of
    the released sheet rather than of the build.
    """
    rows, _ = read_metabolite_rows(data_1)
    specific = {
        (row.metabolite, key): value
        for row in rows
        for key, value in row.specific_um_per_od.items()
    }
    findings: list[dict[str, Any]] = []
    recovered: dict[str, float] = {}
    for record in records:
        strain = _record_strain(record)
        levels = record["experiment"]["phenotype"]["metabolite_level"]
        implied: list[tuple[str, float, float]] = []
        for name, level in levels.items():
            denominator = specific.get((name, (strain, phase)))
            if denominator is None or denominator == 0.0 or float(level) == 0.0:
                continue
            implied.append((name, float(level) / denominator, 0.005 / denominator))
        if not implied:
            findings.append({"strain": strain, "reason": "no comparable metabolite"})
            continue
        median = statistics.median(value for _, value, _ in implied)
        recovered[strain] = median
        disagreeing = tuple(
            sorted(
                name
                for name, value, bound in implied
                if abs(value - median) > bound + 1e-9
            )
        )
        pinned = SPECIFIC_BLOCK_DISAGREEMENTS[(strain, phase)]
        if disagreeing != pinned:
            findings.append(
                {
                    "strain": strain,
                    "disagreeing": list(disagreeing),
                    "pinned": list(pinned),
                }
            )
        expected_od = RECOVERED_HARVEST_OD600.get((strain, phase))
        if expected_od is not None and abs(median - expected_od) > 1e-9:
            findings.append(
                {"strain": strain, "recovered_od600": median, "pinned": expected_od}
            )
    return LevelResult(
        level=Level.L4,
        name="specific_block_recovers_one_harvest_od600_per_strain",
        passed=not findings,
        message=(
            f"{len(records)} records against the released Specific block; recovered "
            f"OD600 { {k: round(v, 4) for k, v in sorted(recovered.items())} }, "
            f"{len(findings)} finding(s)"
        ),
        details={
            "recovered": recovered,
            "pinned_od600": {
                f"{s}:{p}": v for (s, p), v in RECOVERED_HARVEST_OD600.items()
            },
            "pinned_disagreements": {
                f"{s}:{p}": list(v)
                for (s, p), v in SPECIFIC_BLOCK_DISAGREEMENTS.items()
            },
            "worst": findings[:6],
        },
    )


def _deposit_provenance(what: str, page: str) -> Provenance:
    """Where a deposited arm's numbers came from."""
    return Provenance(
        source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{DRYAD_INNER_ZIP_REL}",
        citation_key=CITATION_KEY,
        sha256=DRYAD_INNER_ZIP_SHA256,
        method=what,
        page=page,
        retrieved=DRYAD_RETRIEVED_AT,
    )


def proteome_report(
    records: Sequence[dict[str, Any]], data_root: str | None = None
) -> VerificationReport:
    """The deposited proteome arm's L0-L4 report over already-loaded records."""
    from torchcell.verification.protein import verify_protein_dataset

    del data_root
    report = verify_protein_dataset(
        [dict(record) for record in records],
        dataset_name="proteome_menasalvas2025",
        provenance=_deposit_provenance(
            "Supplementary Data 2 sheet 4, the per-replicate Counts_sum of the three "
            "designed strains in the growth and production phases, averaged per sample "
            f"with SE = SD / sqrt(n) over n={EXPECTED_PROTEOME_REPLICATES} released "
            f"replicates; measurement_type {PROTEOME_MEASUREMENT_TYPE!r}",
            SHEET_PROTEOME,
        ),
        expected_count=EXPECTED_PROTEOME_RECORDS,
        # Three strains x two phases, so every genotype appears twice by design; the
        # (strain, phase) partition is what the L1 row below asserts.
        allow_duplicate_orfs=True,
    )
    report.add(_l1_proteome_sample_partition(records))
    report.add(_l3_proteome_keys_are_one_locus_set(records))
    report.add(_l3_called_variants_only_on_the_control(records))
    report.add(_l3_one_coordinate_per_call(records))
    report.add(_l4_genotypes_vs_supplementary_table_4(records))
    report.add(_l4_note2_loci_are_measured(records))
    report.add(_l4_note2_sigx_fold(records))
    return report


def metabolite_report(
    records: Sequence[dict[str, Any]], dataset_root: str, phase: str
) -> VerificationReport:
    """One deposited metabolite phase's L0-L4 report over already-loaded records."""
    from torchcell.verification.metabolite import verify_metabolite_dataset

    data_1 = Path(dataset_root) / "preprocess" / EXTRACT_SUBDIR / SI_DATA_1_BASENAME
    report = verify_metabolite_dataset(
        [dict(record) for record in records],
        dataset_name=f"metabolite_{phase}_menasalvas2025",
        provenance=_deposit_provenance(
            f"Supplementary Data 1 sheet 1 (data S1-1), the {phase}-phase "
            f"{METABOLITE_AVERAGE_BANNER} of the 48 Absolute rows, with "
            f"n_replicates={METABOLITE_N_REPLICATES.value} from the deposit README's "
            "own column note and NO released dispersion; measurement_type "
            f"{METABOLITE_MEASUREMENT_TYPE!r}",
            f"{SHEET_METABOLITES} ({METABOLITE_PHASE_SUFFIX[phase]})",  # type: ignore[index]
        ),
        expected_count=EXPECTED_METABOLITE_RECORDS,
        # Absolute micromolar concentrations, not centered scores, so the reference is
        # the control strain's own profile rather than an identical zero.
        reference_centered=False,
    )
    report.add(_l1_metabolite_strains(records))
    report.add(_l3_metabolite_keys_are_absolute_rows(records, data_1))
    report.add(_l3_called_variants_only_on_the_control(records))
    report.add(_l3_one_coordinate_per_call(records))
    report.add(_l4_prose_amino_acids_are_not_released(records))
    report.add(_l4_specific_block_recovers_the_harvest_od600(records, data_1, phase))
    return report


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


def selection_report(
    records: Sequence[dict[str, Any]], data_root: str | None = None
) -> VerificationReport:
    """Run the environment-response verifier over the selection arm's records.

    The constant host background is passed as ``background_genes`` so the L1 strain key
    and the L4 gene universe see only the SCREENED knockdown targets: the five pathway
    tokens and the reporter's ``TERTU_1389`` are heterologous identifiers with no locus
    in any assembly, and ``PP_1815`` is the same lesion in all 58 records. Three
    SUPPLEMENTARY rows are appended; the verifier's own rows keep their verdicts.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset,
    )

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
    return report


#: The four dataset families this module serves, one per experiment class and growth
#: stage. ``verify_build`` dispatches on this.
Family = Literal["selection", "proteome", "metabolite_growth", "metabolite_production"]


def verify_build(
    dataset_root: str, data_root: str | None = None, *, family: Family
) -> VerificationReport:
    """Run one family's L0-L4 gate over a built tree and write the report.

    ``family`` is ``"selection"`` (the enriched-gRNA calls of Supplementary Tables 1
    and 2), ``"proteome"`` (the deposited growth/production-phase proteome) or one of
    ``"metabolite_growth"`` / ``"metabolite_production"`` (the two phases of data
    S1-1). The report is written to
    ``<dataset_root>/preprocess/verification_report.json``.
    """
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    if family == "selection":
        report = selection_report(records, data_root)
    elif family == "proteome":
        report = proteome_report(records, data_root)
    elif family == "metabolite_growth":
        report = metabolite_report(records, dataset_root, "growth")
    elif family == "metabolite_production":
        report = metabolite_report(records, dataset_root, "production")
    else:
        raise RuntimeError(
            f"{family!r} is not one of the four families this module serves"
        )
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


#: ``family -> (dataset class, dev-tree root)``, the four stores this module builds.
FAMILY_BUILDS: dict[Family, tuple[type[ExperimentDataset], str]] = {
    "selection": (
        IsoprenolSelectionMenasalvas2025Dataset,
        "data/torchcell/isoprenol_selection_menasalvas2025",
    ),
    "proteome": (
        ProteomeMenasalvas2025Dataset,
        "data/torchcell/proteome_menasalvas2025",
    ),
    "metabolite_growth": (
        MetaboliteGrowthPhaseMenasalvas2025Dataset,
        "data/torchcell/metabolite_growth_menasalvas2025",
    ),
    "metabolite_production": (
        MetaboliteProductionPhaseMenasalvas2025Dataset,
        "data/torchcell/metabolite_production_menasalvas2025",
    ),
}


def main() -> None:
    """Build every family and verify it, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    genome = bacterial_genome("pputida", KT2440_STRAIN, data_root)
    for family, (cls, rel) in FAMILY_BUILDS.items():
        root = osp.join(data_root, rel)
        dataset = cls(root=root, pputida_genome=genome)  # type: ignore[call-arg]
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
        report = verify_build(root, data_root, family=family)
        print(report.summary())


if __name__ == "__main__":
    main()
