# torchcell/datasets/ecoli/mohiuddin2022
# [[torchcell.datasets.ecoli.mohiuddin2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/mohiuddin2022
# Test file: tests/torchcell/datasets/ecoli/test_mohiuddin2022.py
"""Mohiuddin 2022 promoter-GFP reporter library of E. coli under three antibiotics.

Mohiuddin, Massahi and Orman 2022 (Microbiology Spectrum 10:e02253-21,
doi:10.1128/spectrum.02253-21; citation_key
``mohiuddinHighthroughputScreeningPromoter2022``) grew a 1,930-well E. coli K-12 MG1655
promoter-GFP reporter collection in 96-well plates, dosed the early stationary phase with
ampicillin, ofloxacin or gentamicin, or left it untreated, and read GFP hourly from hour
two to hour ten. Supplemental File 2 releases the whole grid: a ``Raw Data`` sheet of
69,480 GFP readings (1,930 wells x 4 arms x 9 reads) and a ``Fold Change`` sheet of
52,110 treated-over-untreated ratios.

RECORD = one (arm, plate, well, read hour) ``PromoterActivityExperiment``:

- GENOTYPE: one ``HeterologousPathwayPerturbation``, the plasmid-borne reporter the
  well carries. ``promoter_name`` is the well's promoter, verbatim, which is what makes
  two wells of different promoters different strains; the reporter gene, the episomal
  localization and the native MG1655 origin of the promoter are constant across the
  library. Nothing in the chromosome is edited, so
  ``AssemblyReferenceGenome.background`` is ``None``: the host IS the reference.
- ENVIRONMENT: ``LB`` Miller at 37 C, aerobic, ``duration_hours`` = the read hour (hours
  after the 1:50 subculture into the assay plate), plus the arm's antibiotic as one
  ``SmallMoleculePerturbation`` ON THE READS FROM HOUR FIVE ONWARD ONLY. The dose went
  in at hour five, so a treated well's hour-two, hour-three and hour-four readings were
  taken before the drug existed in that well and carry no environmental edit. The
  placement is the paper's own statement of when the dose went in; the released fold
  changes corroborate it only weakly, sitting within 1.5% of 1.00 at those three hours
  (per-arm medians 1.009/1.015/1.006 at hour two, 1.004/1.004/1.010 at hour three and
  0.992/0.994/0.995 at hour four) and clearly moving only from hour seven (ampicillin
  1.062, rising to 1.579 at hour ten). Counts: 17,370 untreated reads, 17,370 pre-dose
  reads and 34,740 reads under a drug.
- PHENOTYPE: ``PromoterActivityPhenotype``, ``plate_reader_fluorescence``.
  ``promoter_activity`` is the released GFP number as printed, in the plate reader's own
  units; ``promoter_gene`` is the MG1655 locus the promoter label resolves to;
  ``well_id`` is the sheet's own arm, plate and well, which is what keeps the four arms'
  readings of one well L1-distinct. ``n_samples = 1``: the screen was run once.
- REFERENCE: the SAME well in the UNTREATED arm at the SAME read hour, which is the
  control reading the paper's own fold change divides by. An untreated record's
  reference is therefore itself, the ratio of a control to itself.

WHY A NEW PHENOTYPE FAMILY AND NOT AN EXISTING ONE. The measured entity is a promoter,
and the number is a reporter signal:

- ``EnvironmentResponsePhenotype`` is a fitness/growth response and its verifier
  requires an environmental edit on every record. Three of the four arms here are
  untreated or pre-dose, which that rule correctly refuses, and those are 32,160 of the
  69,480 readings: the control arm is not an artifact of the design, it is the
  denominator the paper publishes.
- the three expression families are genome-wide dicts of ONE strain's transcriptome. A
  reporter library inverts that shape: one promoter per strain across 1,809 strains, and
  the number is a proxy for transcription rather than a transcript count.
- the two phenotype families that existed with no consumer, ``FluxPhenotype`` and
  ``BacterialVisualScorePhenotype``, are a fitted net-flux map and an ordinal colony
  score. Neither is a reporter readout. Checked before the family was added.

THE FOLD CHANGE SHEET IS NOT STORED, BECAUSE IT IS EXACTLY DERIVABLE. The sheet's own
legend states the rule, and the build reproduces every one of its 52,110 cells from the
``Raw Data`` sheet to a maximum relative error of 2.2e-16, which is double-precision
round-off (``preprocess/extraction.json``, ``fold_change``). Storing it as well would
store one measurement twice. It is recoverable from ONE stored record, not two: a
treated record's reference carries the untreated reading of the same well at the same
hour, so ``promoter_activity / reference.promoter_activity`` IS the released fold
change. An L3 row re-checks that on the built store.

ONE READING PER TIME POINT, NOT A NINE-VECTOR. No phenotype in the schema carries an
ordered sequence; every vector-valued phenotype is a dict keyed by a biological entity
(gene, protein, metabolite, reaction), never by time. The repo's convention for repeated
measurements over time is one record per time point with the time on
``Environment.duration_hours``, which is what Caglar 2017 does with its ``growthTime_hr``
column, and it is also what makes the pre-dose reads expressible: three of the nine
reads of a treated well have a different environment from the other six, which a
nine-vector on one record could not say. The triage row's 7,720 instances at dim 9 count
(well x arm); 69,480 is the same grid counted as records, and is the number the row's own
note gives for the released readings.

IDENTIFIERS. The sheet names a well's promoter by the regulated gene's symbol, a
b-number or a gene synonym. ``reconcile_locus_tags`` resolves the 1,809 distinct labels
against MG1655 GCA_000005845.2 through its layers (24 as a locus tag, 1,481 as a gene
symbol, 266 as a gene synonym, 38 not found) under the retain-all policy that drops
nothing for a naming reason. Measured, 1,761 of the 1,809 end up naming one locus of the
assembly and 48 do not; those 48 are stored with ``promoter_gene = None`` and their
label kept verbatim, and they are:

- 38 the resolver does not find: the eight rRNA operon names ``rrnA`` to ``rrnH``, whose
  promoters are real but whose names are not gene names in this annotation; the
  ``insA``/``insB`` IS copies; retired symbols (``aniC``, ``dpbA``, ``fruL``,
  ``gatR_1``, ``groE``, ``gssA``, ``molR_2``, ``pagB``, ``phnQ``, ``rdoA``, ``smf_2``,
  ``somA``, ``tdcG_2``, ``yifM_2``, ``ygaQ``, ``yqaC``, ``yqaD``) and retired b-numbers;
- two ambiguous (``spr`` -> b2175 or b4043, ``ygaD`` -> b2676 or b2700);
- the six kept as given because another label reached the same locus, so the policy
  keeps both rather than merging two wells into one identity (``b0255`` and ``b0257``
  both reach b4587, ``b2639`` and ``b2641`` reach b2641, ``b4283`` and ``b4285`` reach
  b4623), and the retired b-numbers among them are already in the 38;
- ``Empty``, ``U66`` and ``U139``, one well on each of 20 plates. They name no E. coli
  gene. The collection's two backbones are pUA66 and pUA139 in the library's source
  paper (reference 28, Zaslaver 2006), so these read as the empty-vector and empty-well
  background controls, but THE PAPER DEFINES NONE OF THE THREE LABELS and no record
  asserts it: they are kept as 2,160 readings whose promoter resolves to nothing.

The full list and the resolution histogram are in
``preprocess/identifier_reconciliation.json``.

WHERE THE GENE LINKAGE LIVES, AND WHAT IT IS NOT. The only gene the genotype names is
the reporter, so this dataset's ``gene_set`` is ``{"gfp"}`` -- the same shape Foo 2014's
``rfp`` and Menasalvas 2025's reporter take. The measured gene identity is the
phenotype's ``promoter_gene`` property, which is a property match and not a graph edge
to the gene node. That is deliberate: making it an edge would mean writing the
promoter's gene as the perturbation's ``systematic_gene_name``, which asserts that the
gene was added to the strain, and what the plasmid carries is a copy of its PROMOTER
driving GFP. A promoter or regulatory-element node class is what would turn the property
into an edge, and this dataset is the first that would use one.

NO RECORDS ARE DROPPED. Every one of the 69,480 released readings is a finite positive
number (measured: zero non-numeric and zero missing cells; minimum 8.629, median 12.88,
maximum 692.3), the grid is complete, and no naming failure removes a well.

ARITHMETIC CHECKS RUN AT BUILD TIME (``preprocess/extraction.json``), each of which
stops the build:

1. ``grid_is_complete``: the ``Raw Data`` sheet is 1,930 rows and the four arms' blocks
   carry the same plate, well and promoter on every row, so an arm's reading is joined
   to the same well by position and not by a label that repeats.
2. ``fold_change_reproduced``: every ``Fold Change`` cell equals the treated reading over
   the untreated reading of the same well at the same hour; 52,110 of 52,110 within
   1e-9 relative, maximum 2.2e-16.
3. ``fold_change_rows_are_the_raw_rows``: the ``Fold Change`` sheet's three promoter
   columns are the ``Raw Data`` sheet's promoter column in the same order, which is the
   only thing that makes the sheet joinable at all (nine labels repeat, ``lacZ`` 22
   times).

DATA SOURCE. Supplemental File 2 (``spectrum02253-21_supp_2_seq5.xlsx``), with
Supplemental File 1 (the SI PDF) and the article's own PDF and plain text, all four from
the PMC Article Datasets bucket prefix ``PMC8865558.1`` (``pmc_cloud``, scriptable),
deposited in ``$DATA_ROOT/torchcell-raw/mohiuddinHighthroughputScreeningPromoter2022/``
with a ``manifest.json``.

NO LITERATURE MIRROR. This paper is in NEITHER Zotero library (measured: no item with
this DOI or title among the group library's 1,216 items or the personal library's
15,714), so ``scripts/lit_sync.py`` cannot capture it: the sync diffs Zotero against the
mirror and there is no item to diff. That is one step EARLIER than the known
missing-PDF-attachment failure, and adding the item to Zotero is the library owner's to
do. Every quote here is therefore anchored to ``paper/PMC8865558.1.txt``, the publisher's
own plain-text rendering of the article from the same PMC bucket, sha256-pinned in the
raw mirror beside the data, so the provenance chain is complete without Zotero.

NOT ASSERTED. The general Methods state that kanamycin at 50 ug/mL was used to maintain
plasmids, but the screening section names only LB for both the preculture and the assay
plate, so no selection agent is put on the medium. The paper calls the backbone "a
low-copy-number plasmid" and names the supplier, and never says which of the
collection's two backbones carries which promoter, so ``construct_name`` is left unset and the reporter's origin reads ``unreported``; a perturbation leaf has no gap field, so both absences are stated here and in ``SOURCED_VALUES`` rather than guessed.
The paper does not state whether the hour-five reading was taken immediately before or
immediately after the dose went in; the drug is placed on hour five by the paper's own
statement that treatment began at hour five and ran five hours to hour ten, and the
released hour-five fold changes sit near 1.00, which is consistent with either order.
The 1:50 subculture is hour zero of ``duration_hours``; the drug exposure length at a
read is the read hour minus five and is not restated as a second duration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import os.path as osp
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict
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
from torchcell.datamodels.media import LB
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AssemblyReferenceGenome,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    HeterologousPathwayPerturbation,
    PromoterActivityExperiment,
    PromoterActivityExperimentReference,
    PromoterActivityPhenotype,
    Publication,
    ReporterReadout,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_PAPER_PDF,
    ROLE_PAPER_TEXT,
    ROLE_RAW_DATA,
    ROLE_SI_PDF,
    ROLE_SI_TEXT,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.sequence.genome.bacterial import BacterialGenome
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
    audit_sourced_value,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "mohiuddinHighthroughputScreeningPromoter2022"
PAPER_DOI = "10.1128/spectrum.02253-21"
PAPER_TITLE = (
    "High-Throughput Screening of a Promoter Library Reveals New Persister "
    "Mechanisms in Escherichia coli"
)
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
DATASET_ROOT_REL = "data/torchcell/promoter_reporter_mohiuddin2022"

#: The PMC Article Datasets bucket prefix of this article (version 1).
PMC_PREFIX = "PMC8865558.1"
#: When the four files were retrieved and hashed before deposit.
RETRIEVED_AT = "2026-10-08"

DATA_FILE = "spectrum02253-21_supp_2_seq5.xlsx"
SI_FILE = "spectrum02253-21_supp_1_seq6.pdf"
PAPER_PDF_FILE = "PMC8865558.1.pdf"
PAPER_TEXT_FILE = "PMC8865558.1.txt"

#: Quote anchor: the publisher's plain text, inside the raw mirror (no Zotero item).
PAPER_TEXT_REL = f"paper/{PAPER_TEXT_FILE}"
PAPER_TEXT_SHA256 = "7a41a53aed473397704c67750e660ba3eff36e0273ba56742d938d565fe0fde7"

#: Quote anchor for the workbook's own legends. The workbook is binary, so a quote in it
#: is not auditable; this is the deterministic text rendering of its four legend rows,
#: deposited beside it with a ``ProcessingRecord`` naming the reader and pinning the
#: workbook's sha256 as its input.
LEGENDS_FILE = "spectrum02253-21_supp_2_seq5.legends.txt"
LEGENDS_REL = f"si/{LEGENDS_FILE}"
LEGENDS_SHA256 = "4fe99ca38a2fe93ce4fddf0b310851a9575ebfb1b3f23f605837cf148c2f8d29"

RAW_DATA_SHEET = "Raw Data"
FOLD_CHANGE_SHEET = "Fold Change"


class RawFile(BaseModel):
    """One file the mirror holds: its pinned bytes, role and how it was retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    relpath: str
    role: str
    sha256: str
    bytes: int
    description: str
    derived: bool = False

    @property
    def bucket_key(self) -> str:
        """The PMC Article Datasets bucket key the bytes came from."""
        return f"{PMC_PREFIX}/{self.name}"

    @property
    def source_url(self) -> str:
        """HTTPS URL of the bucket object."""
        return pmc_cloud_url(self.bucket_key)

    @property
    def retrieval(self) -> RetrievalRecord:
        """The re-runnable retrieval of these bytes."""
        return RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=self.source_url,
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": self.bucket_key},
            sha256=self.sha256,
            retrieved_at=RETRIEVED_AT,
        )


RAW_FILES: tuple[RawFile, ...] = (
    RawFile(
        name=DATA_FILE,
        relpath=f"data/{DATA_FILE}",
        role=ROLE_RAW_DATA,
        sha256="f0e2d7a8c3271eac5abee32b4d46ae997c5c003a07f1460e660abd15bba8a9e1",
        bytes=1311598,
        description="Supplemental File 2, Data Set S1: sheet 'Raw Data' holds 1,930 "
        "wells x 4 arms x 9 hourly GFP readings (69,480 numbers) with each well's "
        "plate, well and promoter; sheet 'Fold Change' holds the 52,110 "
        "treated-over-untreated ratios",
    ),
    RawFile(
        name=SI_FILE,
        relpath=f"si/{SI_FILE}",
        role=ROLE_SI_PDF,
        sha256="b96a9006bb652c1087bc11c87cc1a1471251cec89b342172b160b830b3f727d2",
        bytes=1045385,
        description="Supplemental File 1: Figures S1 to S8 and Tables S1 to S4. Table "
        "S1 is the hit list (28 ampicillin-induced and 38 ofloxacin-induced promoters "
        "with their genes), a call over this data set rather than a measurement; no "
        "per-record value is in it",
    ),
    RawFile(
        name=PAPER_PDF_FILE,
        relpath=f"paper/{PAPER_PDF_FILE}",
        role=ROLE_PAPER_PDF,
        sha256="3efeb5eaa4983668426a6d8305727b9d3572c83c369e3e63d025167166933002",
        bytes=4300189,
        description="The article PDF as PMC serves it. Mirrored because this paper is "
        "in neither Zotero library, so the literature mirror holds no paper.pdf for "
        "this citation key",
    ),
    RawFile(
        name=PAPER_TEXT_FILE,
        relpath=PAPER_TEXT_REL,
        role=ROLE_PAPER_TEXT,
        sha256=PAPER_TEXT_SHA256,
        bytes=73346,
        description="The article's full text as PMC renders it, the anchor of every "
        "quote in this loader that comes from the paper. Not OCR: it is the publisher's "
        "own text, so a quote matches character for character including the thin spaces "
        "before units",
    ),
    RawFile(
        name=LEGENDS_FILE,
        relpath=LEGENDS_REL,
        role=ROLE_SI_TEXT,
        sha256=LEGENDS_SHA256,
        bytes=579,
        description="The workbook's four legend rows, rendered to text by "
        "extract_sheet_legends so a quote in them is auditable; the workbook itself is "
        "binary. Carries the fold-change rule the sheet states",
        derived=True,
    ),
)
#: ``{raw file name: pinned sha256}``, the build-time check of every consumed file.
DATA_SHA256: dict[str, str] = {RAW_FILES[0].name: RAW_FILES[0].sha256}
RAW_FILES_BY_NAME: dict[str, RawFile] = {f.name: f for f in RAW_FILES}
#: The files that come off the PMC bucket; the legend text is derived from the workbook.
RETRIEVED_FILES: tuple[RawFile, ...] = tuple(f for f in RAW_FILES if not f.derived)

#: What the article releases that the loader deliberately does not consume.
NOT_MIRRORED = (
    "Supplemental File 2 sheet 'Fold Change': the ratio of the 'Raw Data' sheet's "
    "treated reading to its untreated reading of the same well at the same hour, "
    "reproduced by this build to 2.2e-16 over all 52,110 cells. It is recoverable from "
    "one stored record (the reference carries the untreated reading), so storing it "
    "would store one measurement twice",
    "Supplemental File 1 Table S1: the 28 ampicillin-responsive and 38 "
    "ofloxacin-responsive promoters at a two-fold cutoff. A hit call over this data "
    "set, not a measurement",
    "Supplemental File 1 Tables S2 to S4 and Figures S1 to S8: the Keio deletion "
    "follow-up, the persister assays, the MICs and the single-strain validations. "
    "Separate assays on separate strains, released as figures or as strain lists",
    "The seven article figures (spectrum.02253-21-f001.jpg to f007.jpg) in the same "
    "bucket prefix: no record is built from an image",
)

# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_RESULTS = "RESULTS, 'Identifying antibiotic-induced genes using a promoter library'"
_STRAINS = "MATERIALS AND METHODS, 'Bacterial strains and plasmids'"
_CHEMICALS = "MATERIALS AND METHODS, 'Chemicals, media, and growth conditions'"
_SCREEN = "MATERIALS AND METHODS, 'Screening the promoter library'"
_STATS = "MATERIALS AND METHODS, 'Statistical analysis'"


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned article text."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_TEXT_REL,
            citation_key=CITATION_KEY,
            sha256=PAPER_TEXT_SHA256,
            method="the publisher's own plain text of the article, from the PMC "
            "Article Datasets bucket (raw mirror; this paper is in neither Zotero "
            "library, so there is no OCR'd literature mirror to quote)",
            page=page,
        ),
    )


SOURCED_VALUES: dict[str, SourcedValue] = {
    "library_size": _paper(
        1930,
        "we screened an E. coli K-12 MG1655 library with more than 1900 promoters "
        "fused to a fast-folding green fluorescent protein (GFP) gene in a "
        "low-copy-number plasmid (28)",
        page=_RESULTS,
        note="the sheet releases 1,930 wells over 1,809 distinct promoter labels, "
        "which is the 'more than 1900 promoters' the text names",
    ),
    "reporter": _paper(
        "gfp",
        "fused to a fast-folding green fluorescent protein (GFP) gene in a "
        "low-copy-number plasmid (28)",
        page=_RESULTS,
        note="the paper names the reporter only as a fast-folding GFP and the "
        "backbone only as low-copy, deferring both to reference 28 (Zaslaver 2006)",
    ),
    "strain_source": _paper(
        "Escherichia coli K-12 MG1655",
        "The promoter strain collection of Escherichia coli MG1655 (28) in a 96-well "
        "plate format used for the screening assay was obtained from Horizon "
        "Discovery, Lafayette, CO, USA.",
        page=_STRAINS,
    ),
    "medium": _paper(
        "LB",
        "Overnight precultures were diluted 1:50 in LB medium in a new 96-well plate, "
        "sealed, and incubated as described above.",
        page=_SCREEN,
        note="the screening section names only LB for the preculture and the assay "
        "plate; the general methods state that kanamycin at 50 ug/mL was used to "
        "maintain plasmids, but not for these cultures, so no selection agent is put "
        "on this medium",
    ),
    "lb_formulation": _paper(
        "LB, Miller",
        "Standard Luria-Bertani (LB) broth was prepared by dissolving 5 g yeast "
        "extract, 10 g tryptone, and 10 g sodium chloride in 1 l deionized (DI).",
        page=_CHEMICALS,
        note="10 g/L tryptone, 5 g/L yeast extract and 10 g/L NaCl are the Miller "
        "amounts, which is the shared torchcell LB object",
    ),
    "temperature": _paper(
        37.0, "and cultured for 24 h at 37°C with shaking at 250 rpm", page=_SCREEN
    ),
    "aerobicity": _paper(
        "aerobic",
        "The plates were sealed with a sterile, oxygen-permeable membrane "
        "(Breathe-Easier, Cat# BERM-2000, VWR International)",
        page=_SCREEN,
        note="an oxygen-permeable seal with shaking at 250 rpm is an aerobic regime",
    ),
    "treatment": _paper(
        5.0,
        "At early stationary phase (t = 5 h), cells were treated with "
        "ampicillin (200 μg/mL), ofloxacin (5 μg/mL), and "
        "gentamicin (50 μg/mL) for 5 h.",
        page=_SCREEN,
        note="the dose went in at hour five and ran five hours to hour ten, so the "
        "hour-two to hour-four readings of a treated well are pre-dose",
    ),
    "read_interval": _paper(
        1.0,
        "Changes in GFP were measured hourly with a plate reader.",
        page=_RESULTS,
        note="the sheet's nine columns are t=2 to t=10, one per hour",
    ),
    "instrument": _paper(
        "plate reader",
        "GFP was measured with a Varioskan LUX Multimode Microplate Reader (Thermo "
        "Fisher, Waltham, MA, USA) at the indicated times with untreated cultures as "
        "a control (see Data set S1).",
        page=_SCREEN,
    ),
    "wavelengths": _paper(
        (485.0, 511.0),
        "The excitation and emission wavelengths for GFP measurement were "
        "485 nm and 511 nm, respectively.",
        page=_SCREEN,
    ),
    "n_samples": _paper(
        1,
        "High-throughput screening of the promoter library and the Keio collection "
        "was performed only once.",
        page=_STATS,
        note="the screen has ONE measurement per well, arm and hour: no replicate, no "
        "SD and no SE, which is why every uncertainty field is a typed gap. The N=3 "
        "and N=4 in the SI figure legends belong to the single-strain validation "
        "assays, not to this screen",
    ),
    "fold_change_rule": SourcedValue(
        value="treated / untreated",
        quote="Fold changes were calculated by taking the ratio of GFP values of "
        "antibiotic treated cultures to those of untreated cultures.",
        note="the sheet's own legend; the build reproduces every cell of it from the "
        "Raw Data sheet, so the sheet is not stored",
        provenance=Provenance(
            source_uri=LEGENDS_REL,
            citation_key=CITATION_KEY,
            sha256=LEGENDS_SHA256,
            method="the workbook's legend rows rendered to text by "
            "extract_sheet_legends (the workbook is binary, so a quote in it is not "
            "auditable); its ProcessingRecord pins the workbook's own sha256",
            page="Supplemental File 2, sheet 'Fold Change', row 2",
        ),
    ),
    "raw_sheet_legend": SourcedValue(
        value=4,
        quote="A library of E. coli MG1655 strains with promoter reporters was "
        "treated in the early stationary phase with ampicillin (200 µg/ml), "
        "ofloxacin (5 µg/ml), or gentamicin (50 µg/ml) for 5 h or left "
        "untreated. GFP was measured at the designated times (h).",
        note="the four arms and the hour unit of the nine columns, from the sheet's "
        "own legend",
        provenance=Provenance(
            source_uri=LEGENDS_REL,
            citation_key=CITATION_KEY,
            sha256=LEGENDS_SHA256,
            method="the workbook's legend rows rendered to text by "
            "extract_sheet_legends (the workbook is binary, so a quote in it is not "
            "auditable); its ProcessingRecord pins the workbook's own sha256",
            page="Supplemental File 2, sheet 'Raw Data', row 2",
        ),
    ),
}

# --------------------------------------------------------------------------- #
# The four arms and the nine reads
# --------------------------------------------------------------------------- #
#: The nine read hours, as the sheet's ``t=N`` column headers name them.
READ_HOURS: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
#: The hour the dose went in; reads from here on carry the arm's antibiotic.
TREATMENT_START_HOUR = 5.0


class ScreenArm(BaseModel):
    """One of the sheet's four arms: its block header, its drug and its dose."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    header: str
    compound: str | None = None
    dose: float | None = None

    @property
    def is_treated(self) -> bool:
        """Whether this arm received an antibiotic at all."""
        return self.compound is not None

    def perturbations(self, hour: float) -> list[SmallMoleculePerturbation]:
        """The arm's environmental edits at one read hour (empty before the dose)."""
        if self.compound is None or hour < TREATMENT_START_HOUR:
            return []
        assert self.dose is not None
        return [
            SmallMoleculePerturbation(
                compound=resolved_compound(self.compound),
                concentration=Concentration(
                    value=self.dose,
                    unit=ConcentrationUnit.ug_per_ml,
                    basis=DoseBasis.fixed,
                ),
            )
        ]


UNTREATED = ScreenArm(header="Untreated")
ARMS: tuple[ScreenArm, ...] = (
    UNTREATED,
    ScreenArm(header="Ampicillin Treatment", compound="ampicillin", dose=200.0),
    ScreenArm(header="Ofloxacin Treatment", compound="ofloxacin", dose=5.0),
    ScreenArm(header="Gentamicin Treatment", compound="gentamicin", dose=50.0),
)

# --------------------------------------------------------------------------- #
# Record constants
# --------------------------------------------------------------------------- #
GENE_NAMESPACE = STRAIN_GENE_NAMESPACES["MG1655"]
REPORTER_GENE = str(SOURCED_VALUES["reporter"].value)
READOUT = ReporterReadout.plate_reader_fluorescence
ACTIVITY_UNITS = (
    "GFP fluorescence of the well as the plate reader printed it (excitation 485 nm, "
    "emission 511 nm), not background-subtracted and not normalized to cell density"
)
PATHWAY_NAME = "promoter-GFP transcriptional reporter fusion on a low-copy plasmid"
REPORTER_SOURCE_ORGANISM = "unreported"
N_SAMPLES = int(SOURCED_VALUES["n_samples"].value)

PUBLICATION = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")

#: 1,930 wells x 4 arms x 9 read hours.
EXPECTED_WELLS = 1930
EXPECTED_RECORDS = EXPECTED_WELLS * len(ARMS) * len(READ_HOURS)
#: The 'Fold Change' sheet's cell count: three treated arms x nine hours x the wells.
EXPECTED_FOLD_CHANGE_CELLS = EXPECTED_WELLS * (len(ARMS) - 1) * len(READ_HOURS)
#: A reproduced fold change must match the released one to this relative tolerance.
FOLD_CHANGE_TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/mohiuddinHighthroughputScreeningPromoter2022``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_sheet_legends(workbook: str | Path) -> str:
    """The workbook's four legend rows as deterministic text.

    Rows 1 and 2 of each sheet are the release's own description of what the numbers
    are, and they are the only statement of the fold-change rule. The workbook is
    binary, so a quote in it cannot be audited; this rendering can, and its
    :class:`ProcessingRecord` names this function and pins the workbook's sha256.
    """
    import openpyxl

    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    lines: list[str] = []
    for sheet in (RAW_DATA_SHEET, FOLD_CHANGE_SHEET):
        lines.append(f"# sheet: {sheet}")
        work = book[sheet]
        rows = iter(work.iter_rows(min_row=1, max_row=2, values_only=True))
        for row in rows:
            lines.append("" if row[0] is None else str(row[0]))
    book.close()
    return "\n".join(lines) + "\n"


def legends_processing(workbook_sha256: str) -> ProcessingRecord:
    """How the legend text was produced, and from which bytes."""
    return ProcessingRecord(
        processor="torchcell.datasets.ecoli.mohiuddin2022.extract_sheet_legends",
        tool="openpyxl",
        version=_openpyxl_version(),
        params={"sheets": [RAW_DATA_SHEET, FOLD_CHANGE_SHEET], "rows": [1, 2]},
        input_sha256=[workbook_sha256],
    )


def _openpyxl_version() -> str:
    """The openpyxl version the legend text was rendered with."""
    import openpyxl

    return str(openpyxl.__version__)


def retrieve_raw_files(dest_dir: str | Path) -> dict[str, Path]:
    """Run each file's recorded retriever and write the verified bytes to ``dest_dir``.

    The recorded ``RetrievalRecord`` is what runs (``run_retriever``), so this IS the
    re-runnable retrieval; a byte mismatch raises before anything is written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for raw in RETRIEVED_FILES:
        path = dest / raw.name
        write_verified(run_retriever(raw.retrieval), path, raw.sha256, raw.source_url)
        out[raw.name] = path
    return out


def deposit_raw_mirror(
    *, sources: Mapping[str, str | Path], data_root: str | None = None
) -> Path:
    """Write the raw mirror from already-retrieved files plus its ``manifest.json``.

    ``sources`` maps every name in ``RAW_FILES`` to a local file. Idempotent by sha256: a
    mirror file with the pinned hash is left alone, and one with any other hash raises
    rather than being overwritten.
    """
    missing = sorted({f.name for f in RETRIEVED_FILES} - set(sources))
    if missing:
        raise KeyError(f"no source given for {missing}")
    root = raw_mirror_dir(data_root)
    workbook_sha256 = _sha256(sources[DATA_FILE])
    records: list[ArtifactRecord] = []
    for raw in RAW_FILES:
        dest = root / raw.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if raw.derived:
            text = extract_sheet_legends(sources[DATA_FILE])
            if dest.exists() and dest.read_text(encoding="utf-8") != text:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
            dest.write_text(text, encoding="utf-8")
            records.append(
                ArtifactRecord(
                    path=raw.relpath,
                    role=raw.role,
                    bytes=dest.stat().st_size,
                    sha256=_sha256(dest),
                    source=f"derived from data/{DATA_FILE}",
                    original_filename=raw.name,
                    processing=legends_processing(workbook_sha256),
                )
            )
            continue
        src = Path(sources[raw.name])
        got = _sha256(src)
        if got != raw.sha256:
            raise RuntimeError(
                f"{src} sha256 mismatch: got {got}, expected {raw.sha256}"
            )
        if dest.exists():
            if _sha256(dest) != raw.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        records.append(
            ArtifactRecord(
                path=raw.relpath,
                role=raw.role,
                bytes=dest.stat().st_size,
                sha256=raw.sha256,
                source=raw.source_url,
                original_filename=raw.name,
                retrieval=raw.retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=records,
        si_data_sources=[raw.source_url for raw in RETRIEVED_FILES],
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


# --------------------------------------------------------------------------- #
# Reading the workbook
# --------------------------------------------------------------------------- #
class SheetLayoutError(ValueError):
    """The workbook is not the layout this loader was written against."""


#: ``Raw Data``: the 0-based column of each arm's (plate, well, promoter, first hour).
_RAW_BLOCKS: dict[str, tuple[int, int, int, int]] = {
    "Untreated": (0, 1, 2, 3),
    "Ampicillin Treatment": (14, 15, 16, 17),
    "Ofloxacin Treatment": (28, 29, 30, 31),
    "Gentamicin Treatment": (42, 43, 44, 45),
}
#: ``Fold Change``: the 0-based column of each treated arm's (promoter, first hour).
_FC_BLOCKS: dict[str, tuple[int, int]] = {
    "Ampicillin Treatment": (0, 1),
    "Ofloxacin Treatment": (12, 13),
    "Gentamicin Treatment": (24, 25),
}
#: Row 6 (1-based) holds the column headers; data starts on row 7.
_HEADER_ROW = 6
_FIRST_DATA_ROW = 7
_HOUR_HEADERS = tuple(f"t={int(h)}" for h in READ_HOURS)


def _rows(path: str | Path, sheet: str) -> tuple[list[list[Any]], list[list[Any]]]:
    """The header rows (4 and 6) and the data rows of one sheet."""
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if book.sheetnames != [RAW_DATA_SHEET, FOLD_CHANGE_SHEET]:
        book.close()
        raise SheetLayoutError(
            f"expected sheets {[RAW_DATA_SHEET, FOLD_CHANGE_SHEET]}, got "
            f"{book.sheetnames}"
        )
    work = book[sheet]
    all_rows = [list(row) for row in work.iter_rows(values_only=True)]
    book.close()
    header = [all_rows[3], all_rows[_HEADER_ROW - 1]]
    return header, all_rows[_FIRST_DATA_ROW - 1 :]


def _check_hour_headers(header: Sequence[Any], first: int, what: str) -> None:
    """The nine columns after ``first`` are headed ``t=2`` to ``t=10``, in order."""
    got = tuple(str(header[first + i] or "").strip() for i in range(len(READ_HOURS)))
    if got != _HOUR_HEADERS:
        raise SheetLayoutError(
            f"{what}: expected hour headers {_HOUR_HEADERS}, got {got}"
        )


def read_raw_sheet(path: str | Path) -> pd.DataFrame:
    """The ``Raw Data`` sheet as one row per well, with the four arms' nine readings.

    Columns: ``plate``, ``well``, ``promoter`` and ``<arm header>|t=<hour>`` per arm and
    hour. The four blocks are joined BY POSITION, which the arm-key check below proves
    is the same well; joining by promoter label would be wrong, because nine labels
    repeat (``lacZ`` 22 times).
    """
    (block_header, column_header), data = _rows(path, RAW_DATA_SHEET)
    for arm, (plate_c, well_c, promoter_c, first_hour) in _RAW_BLOCKS.items():
        if str(block_header[first_hour + 3] or "").strip() != arm:
            raise SheetLayoutError(
                f"Raw Data: column {first_hour + 3} is not headed {arm!r}, got "
                f"{block_header[first_hour + 3]!r}"
            )
        for column, want in (
            (plate_c, "Plate_Number"),
            (well_c, "Well"),
            (promoter_c, "Promoter_Name"),
        ):
            if str(column_header[column] or "").strip() != want:
                raise SheetLayoutError(
                    f"Raw Data: column {column} is not {want!r}, got "
                    f"{column_header[column]!r}"
                )
        _check_hour_headers(column_header, first_hour, f"Raw Data {arm}")

    plate_c, well_c, promoter_c, _ = _RAW_BLOCKS["Untreated"]
    built: list[dict[str, Any]] = []
    for offset, row in enumerate(data):
        keys = {
            arm: (row[p], row[w], row[g]) for arm, (p, w, g, _) in _RAW_BLOCKS.items()
        }
        if len(set(keys.values())) != 1:
            raise SheetLayoutError(
                f"Raw Data row {offset + _FIRST_DATA_ROW}: the four arms name "
                f"different wells {keys}"
            )
        record: dict[str, Any] = {
            "plate": str(row[plate_c]),
            "well": str(row[well_c]),
            "promoter": str(row[promoter_c]),
        }
        for arm, (_, _, _, first_hour) in _RAW_BLOCKS.items():
            for index, hour in enumerate(READ_HOURS):
                value = row[first_hour + index]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise SheetLayoutError(
                        f"Raw Data row {offset + _FIRST_DATA_ROW}, {arm} t={int(hour)}: "
                        f"{value!r} is not a number"
                    )
                record[f"{arm}|t={int(hour)}"] = float(value)
        built.append(record)
    frame = pd.DataFrame(built)
    if len(frame) != EXPECTED_WELLS:
        raise SheetLayoutError(
            f"Raw Data: expected {EXPECTED_WELLS} wells, got {len(frame)}"
        )
    return frame


def read_fold_change_sheet(path: str | Path) -> pd.DataFrame:
    """The ``Fold Change`` sheet as one row per well, keyed by position."""
    (block_header, column_header), data = _rows(path, FOLD_CHANGE_SHEET)
    for arm, (promoter_c, first_hour) in _FC_BLOCKS.items():
        if str(block_header[first_hour + 3] or "").strip() != arm:
            raise SheetLayoutError(
                f"Fold Change: column {first_hour + 3} is not headed {arm!r}, got "
                f"{block_header[first_hour + 3]!r}"
            )
        if str(column_header[promoter_c] or "").strip() != "Promoter_Name":
            raise SheetLayoutError(
                f"Fold Change: column {promoter_c} is not 'Promoter_Name'"
            )
        _check_hour_headers(column_header, first_hour, f"Fold Change {arm}")

    built: list[dict[str, Any]] = []
    for row in data:
        record: dict[str, Any] = {}
        for arm, (promoter_c, first_hour) in _FC_BLOCKS.items():
            record[f"{arm}|promoter"] = str(row[promoter_c])
            for index, hour in enumerate(READ_HOURS):
                record[f"{arm}|t={int(hour)}"] = float(row[first_hour + index])
        built.append(record)
    frame = pd.DataFrame(built)
    if len(frame) != EXPECTED_WELLS:
        raise SheetLayoutError(
            f"Fold Change: expected {EXPECTED_WELLS} wells, got {len(frame)}"
        )
    return frame


# --------------------------------------------------------------------------- #
# Build-time checks
# --------------------------------------------------------------------------- #
class FoldChangeCheck(BaseModel):
    """Whether the released fold changes are the stored readings' own ratio."""

    model_config = ConfigDict(extra="forbid")

    n_cells: int
    n_within_tolerance: int
    tolerance: float
    max_relative_error: float
    promoter_order_matches: bool


def check_fold_change(raw: pd.DataFrame, fold: pd.DataFrame) -> FoldChangeCheck:
    """Reproduce every ``Fold Change`` cell from the ``Raw Data`` sheet.

    The sheet keys only by promoter name, and nine labels repeat, so the join is
    positional; the promoter columns agreeing with the raw sheet's own column, row for
    row, is what proves the position IS the well.
    """
    order_ok = all(
        (fold[f"{arm}|promoter"] == raw["promoter"]).all() for arm in _FC_BLOCKS
    )
    if not order_ok:
        raise SheetLayoutError(
            "Fold Change: the promoter columns are not the Raw Data sheet's promoter "
            "column in the same order, so no cell can be joined to a well"
        )
    worst = 0.0
    within = 0
    total = 0
    for arm in _FC_BLOCKS:
        for hour in READ_HOURS:
            released: npt.NDArray[np.float64] = fold[f"{arm}|t={int(hour)}"].to_numpy(
                dtype=np.float64
            )
            treated: npt.NDArray[np.float64] = raw[f"{arm}|t={int(hour)}"].to_numpy(
                dtype=np.float64
            )
            control: npt.NDArray[np.float64] = raw[f"Untreated|t={int(hour)}"].to_numpy(
                dtype=np.float64
            )
            computed = treated / control
            error = abs(computed - released) / abs(released)
            total += int(error.size)
            within += int((error <= FOLD_CHANGE_TOLERANCE).sum())
            worst = max(worst, float(error.max()))
    if total != EXPECTED_FOLD_CHANGE_CELLS:
        raise SheetLayoutError(
            f"Fold Change: compared {total} cells, expected "
            f"{EXPECTED_FOLD_CHANGE_CELLS}"
        )
    if within != total:
        raise SheetLayoutError(
            f"Fold Change: {total - within} of {total} cells are not the stored "
            f"readings' ratio within {FOLD_CHANGE_TOLERANCE} (worst {worst:.3g})"
        )
    return FoldChangeCheck(
        n_cells=total,
        n_within_tolerance=within,
        tolerance=FOLD_CHANGE_TOLERANCE,
        max_relative_error=worst,
        promoter_order_matches=order_ok,
    )


class PromoterLedger(BaseModel):
    """How the sheet's promoter labels map onto the pinned MG1655 annotation."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    n_wells: int
    n_labels: int
    n_labels_resolved: int
    unresolved_labels: list[str]
    reconciliation: LocusTagReconciliation
    labels_in_more_than_one_well: dict[str, int]


def resolve_promoters(
    raw: pd.DataFrame, genome: BacterialGenome[Any], *, label: str
) -> tuple[dict[str, str | None], PromoterLedger]:
    """Map every promoter label to an MG1655 locus tag, or to ``None``.

    ``reconcile_locus_tags`` is the shared retain-all policy: it returns the locus tag
    for a label that resolves to one and the label itself otherwise. A label that comes
    back unchanged AND is not a locus tag of the assembly resolved to nothing, so its
    ``promoter_gene`` is ``None`` rather than a non-gene string in a gene field.
    """
    stored, reconciliation = reconcile_locus_tags(genome, raw["promoter"], label=label)
    mapping: dict[str, str | None] = {}
    for source, resolved in zip(raw["promoter"], stored, strict=True):
        if source in mapping:
            continue
        mapping[source] = None if resolved == source else str(resolved)
    for source in list(mapping):
        if mapping[source] is None:
            resolution = genome.resolve_gene_name(source)
            if resolution.systematic_name == source and resolution.status.value in (
                "current",
                "non_gene_feature",
            ):
                mapping[source] = source
    counts = raw["promoter"].value_counts()
    return mapping, PromoterLedger(
        dataset=label,
        n_wells=len(raw),
        n_labels=len(mapping),
        n_labels_resolved=sum(1 for v in mapping.values() if v is not None),
        unresolved_labels=sorted(k for k, v in mapping.items() if v is None),
        reconciliation=reconciliation,
        labels_in_more_than_one_well={
            str(name): int(n) for name, n in counts.items() if n > 1
        },
    )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def reporter_genotype(promoter: str) -> Genotype:
    """The plasmid-borne promoter-GFP reporter the well carries.

    The chromosome is untouched: what the strain carries beyond the reference assembly
    is one episomal reporter plasmid, which is a ``GeneAdditionPerturbation`` leaf and
    not a background lesion. ``promoter_name`` is the measured promoter, verbatim, which
    is what makes two wells of different promoters two strains.

    ``construct_name`` is left unset and ``source_organism`` is
    :data:`REPORTER_SOURCE_ORGANISM`: the paper calls the backbone "a low-copy-number
    plasmid", defers the collection to reference 28 and never says which of the
    collection's two backbones carries which promoter, nor the reporter's organism of
    origin. A perturbation leaf carries no ``provenance_gaps`` field, so those two
    absences are recorded in this module's docstring and in ``SOURCED_VALUES`` instead
    of being guessed.
    """
    return Genotype(
        perturbations=[
            HeterologousPathwayPerturbation(
                systematic_gene_name=REPORTER_GENE,
                perturbed_gene_name=REPORTER_GENE,
                gene_namespace=GENE_NAMESPACE,
                pathway_name=PATHWAY_NAME,
                source_organism=REPORTER_SOURCE_ORGANISM,
                is_heterologous=True,
                localization="episomal_plasmid",
                promoter_name=promoter,
                copy_number=1.0,
            )
        ]
    )


def environment(arm: ScreenArm, hour: float) -> Environment:
    """LB Miller at 37 C, read at ``hour``, with the arm's drug once it is in."""
    return Environment(
        media=LB,
        temperature=Temperature(value=float(SOURCED_VALUES["temperature"].value)),
        perturbations=list(arm.perturbations(hour)),
        aerobicity=str(SOURCED_VALUES["aerobicity"].value),
        duration_hours=hour,
    )


def _activity_gaps() -> list[ProvenanceGap]:
    """The screen was run once, so there is no dispersion of any kind to carry."""
    note = (
        "the screen was performed only once, so a well, arm and hour has exactly one "
        "reading and the source reports no replicate, no SD and no SE; a dispersion "
        "computed here would be our statistic, not a released one"
    )
    return [
        ProvenanceGap(
            field=field, reason=ProvenanceGapReason.not_reported_by_primary, note=note
        )
        for field in (
            "promoter_activity_uncertainty",
            "promoter_activity_uncertainty_type",
        )
    ]


def well_id(arm: ScreenArm, plate: str, well: str) -> str:
    """The sheet's own identifier of one reading: its arm, plate and well."""
    return f"{arm.header}|{plate}|{well}"


def activity_phenotype(
    value: float, promoter: str, gene: str | None, identifier: str
) -> PromoterActivityPhenotype:
    """One well's GFP reading at one hour, as released."""
    return PromoterActivityPhenotype(
        promoter_activity=value,
        n_samples=N_SAMPLES,
        sample_unit=SampleUnit.biological_replicate,
        promoter_name=promoter,
        promoter_gene=gene,
        readout=READOUT,
        reporter_gene=REPORTER_GENE,
        activity_units=ACTIVITY_UNITS,
        well_id=identifier,
        provenance_gaps=_activity_gaps(),
    )


def build_experiment(
    dataset_name: str,
    genotype: Genotype,
    env: Environment,
    phenotype: PromoterActivityPhenotype,
) -> PromoterActivityExperiment:
    """The record of one well, in one arm, at one read hour."""
    return PromoterActivityExperiment(
        dataset_name=dataset_name,
        genotype=genotype,
        environment=env,
        phenotype=phenotype,
    )


def build_reference(
    dataset_name: str,
    genome_reference: AssemblyReferenceGenome,
    hour: float,
    phenotype: PromoterActivityPhenotype,
) -> PromoterActivityExperimentReference:
    """The same well in the untreated arm at the same hour: the fold change's divisor."""
    return PromoterActivityExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=environment(UNTREATED, hour),
        phenotype_reference=phenotype,
    )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class PromoterReporterMohiuddin2022Dataset(ExperimentDataset):
    """Mohiuddin 2022 hourly promoter-GFP readings under three antibiotics."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "MG1655"

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
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
        return PromoterActivityExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return PromoterActivityExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The one consumed workbook, linked from the raw mirror."""
        return [DATA_FILE]

    def download(self) -> None:
        """Link the mirrored workbook into ``raw/`` after checking manifest and sha256.

        The mirror plus ``DATA_SHA256`` is canonical; the PMC bucket URL is retrieval
        metadata ``retrieve_raw_files`` re-runs, never a build input.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        raw = RAW_FILES_BY_NAME[DATA_FILE]
        check_manifest_pin(
            raw.relpath, manifest_sha256(manifest, raw.relpath), raw.sha256
        )
        src = raw_mirror_dir(data_root) / raw.relpath
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        link_verified(src, osp.join(self.raw_dir, raw.name), raw.sha256)
        log.info(
            "Mohiuddin 2022 workbook linked into %s (sha256 verified)", self.raw_dir
        )

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache (a direct run)."""
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
        """Parse the workbook into per-(arm, well, hour) records; write LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        path = osp.join(self.raw_dir, DATA_FILE)
        raw = read_raw_sheet(path)
        fold = read_fold_change_sheet(path)
        fold_check = check_fold_change(raw, fold)
        genome = self._genome()
        promoter_gene, ledger = resolve_promoters(raw, genome, label=self.name)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_ledgers(ledger, fold_check)

        genome_reference = assembly_reference(self.REFERENCE_STRAIN)
        genotypes = {
            promoter: reporter_genotype(promoter) for promoter in promoter_gene
        }
        environments = {
            (arm.header, hour): environment(arm, hour)
            for arm in ARMS
            for hour in READ_HOURS
        }
        plates = raw["plate"].tolist()
        wells = raw["well"].tolist()
        promoters = raw["promoter"].tolist()
        readings = {
            (arm.header, hour): raw[f"{arm.header}|t={int(hour)}"].tolist()
            for arm in ARMS
            for hour in READ_HOURS
        }
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        idx = 0
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for position in tqdm(range(len(raw)), desc="mohiuddin2022"):
                promoter = promoters[position]
                gene = promoter_gene[promoter]
                genotype = genotypes[promoter]
                plate, well = plates[position], wells[position]
                for hour in READ_HOURS:
                    control = activity_phenotype(
                        readings[UNTREATED.header, hour][position],
                        promoter,
                        gene,
                        well_id(UNTREATED, plate, well),
                    )
                    reference = build_reference(
                        self.name, genome_reference, hour, control
                    )
                    for arm in ARMS:
                        phenotype = activity_phenotype(
                            readings[arm.header, hour][position],
                            promoter,
                            gene,
                            well_id(arm, plate, well),
                        )
                        experiment = build_experiment(
                            self.name,
                            genotype,
                            environments[arm.header, hour],
                            phenotype,
                        )
                        txn.put(
                            f"{idx}".encode(),
                            self._intern_record(
                                experiment, reference, PUBLICATION, itxn
                            ),
                        )
                        idx += 1
        env_out.close()
        interned_env.close()
        if idx != EXPECTED_RECORDS:
            raise RuntimeError(f"wrote {idx} records, the grid is {EXPECTED_RECORDS}")
        log.info(
            "Mohiuddin 2022: wrote %d records from %d wells over %d promoter labels, "
            "%d of which resolve to an MG1655 locus",
            idx,
            len(raw),
            ledger.n_labels,
            ledger.n_labels_resolved,
        )

    def _write_ledgers(
        self, ledger: PromoterLedger, fold_check: FoldChangeCheck
    ) -> None:
        """The promoter-identity ledger and the build's arithmetic checks."""
        out = Path(self.preprocess_dir)
        (out / "identifier_reconciliation.json").write_text(
            ledger.model_dump_json(indent=2)
        )
        (out / "extraction.json").write_text(
            json.dumps(
                {
                    "raw_sha256": DATA_SHA256,
                    "grid": {
                        "n_wells": ledger.n_wells,
                        "n_arms": len(ARMS),
                        "n_read_hours": len(READ_HOURS),
                        "n_records": EXPECTED_RECORDS,
                    },
                    "fold_change": fold_check.model_dump(),
                },
                indent=2,
            )
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled by ``build_experiment``."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built LMDB
# --------------------------------------------------------------------------- #
VERIFIER_PROVENANCE = Provenance(
    source_uri=(
        f"$DATA_ROOT/{RAW_DIR_REL}/data/{DATA_FILE} (Supplemental File 2, Data Set S1: "
        "sheets 'Raw Data' and 'Fold Change') + "
        f"{PAPER_TEXT_REL} (the publisher's article text, the quote anchor); raw "
        "mirror with manifest.json, pmc_cloud retrieval"
    ),
    citation_key=CITATION_KEY,
    sha256=RAW_FILES_BY_NAME[DATA_FILE].sha256,
    method=(
        "One PromoterActivityExperiment per (arm, plate, well, read hour) of the 'Raw "
        "Data' sheet: 1,930 wells x 4 arms x 9 hourly reads = 69,480 records, nothing "
        "dropped. promoter_activity is the released GFP number as printed, "
        "readout=plate_reader_fluorescence (Varioskan LUX, excitation 485 nm, emission "
        "511 nm), not background-subtracted and not OD-normalized. n_samples=1 with "
        "sample_unit biological_replicate: the paper states the screen 'was performed "
        "only once', so every uncertainty field is a typed gap. Genotype = one "
        "HeterologousPathwayPerturbation, the episomal promoter-GFP reporter, whose "
        "promoter_name is the well's promoter verbatim; the chromosome is untouched, so "
        "the MG1655 assembly reference carries no background. Environment = LB Miller "
        "at 37 C, aerobic, duration_hours = the read hour (hours after the 1:50 "
        "subculture), plus the arm's antibiotic at its stated dose (ampicillin 200, "
        "ofloxacin 5, gentamicin 50 ug/mL, DoseBasis.fixed) ON HOUR FIVE ONWARD ONLY, "
        "because the dose went in at hour five and the hour-two to hour-four reads of a "
        "treated well are pre-dose. Reference = the same well in the untreated arm at "
        "the same hour, which is the divisor of the paper's own fold change. The 'Fold "
        "Change' sheet is NOT stored: the build reproduces all 52,110 of its cells from "
        "the 'Raw Data' sheet to a maximum relative error of 2.2e-16, and a stored "
        "record's ratio to its own reference IS that cell. promoter_gene is the MG1655 "
        "locus the label resolves to for 1,761 of the 1,809 labels and None for the 48 "
        "that resolve to no single locus, which are kept with their label verbatim"
    ),
    page=(
        "Microbiology Spectrum 2022 10:e02253-21 "
        f"(doi:{PAPER_DOI}); Supplemental File 2 "
        f"sha256={RAW_FILES_BY_NAME[DATA_FILE].sha256}; "
        f"{PAPER_TEXT_FILE} sha256={PAPER_TEXT_SHA256}"
    ),
    retrieved=RETRIEVED_AT,
)


def fold_change_recoverable(records: Iterable[Mapping[str, Any]]) -> LevelResult:
    """L3: a record's ratio to its own reference is the released fold change.

    The released ``Fold Change`` sheet is not stored, so this is the row that proves it
    is not lost: for every treated record the stored reading over the stored reference
    reading reproduces the sheet's cell. The sheet is re-read from the raw mirror and
    joined positionally, the same join the build's own check uses.
    """
    raw_path = raw_mirror_dir() / RAW_FILES_BY_NAME[DATA_FILE].relpath
    raw = read_raw_sheet(raw_path)
    fold = read_fold_change_sheet(raw_path)
    released: dict[tuple[str, str, str, float], float] = {}
    for position in range(len(raw)):
        plate = raw["plate"][position]
        well = raw["well"][position]
        for arm_header in _FC_BLOCKS:
            for hour in READ_HOURS:
                released[arm_header, plate, well, hour] = float(
                    fold[f"{arm_header}|t={int(hour)}"][position]
                )
    worst = 0.0
    n_checked = 0
    missing = 0
    mismatched: list[dict[str, Any]] = []
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        arm_header, plate, well = str(phenotype["well_id"]).split("|")
        if arm_header not in _FC_BLOCKS:
            continue
        hour = float(record["experiment"]["environment"]["duration_hours"])
        key = (arm_header, plate, well, hour)
        if key not in released:
            missing += 1
            continue
        control = float(record["reference"]["phenotype_reference"]["promoter_activity"])
        computed = float(phenotype["promoter_activity"]) / control
        error = abs(computed - released[key]) / abs(released[key])
        worst = max(worst, error)
        n_checked += 1
        if error > FOLD_CHANGE_TOLERANCE and len(mismatched) < 20:
            mismatched.append(
                {"key": str(key), "computed": computed, "released": released[key]}
            )
    passed = not mismatched and missing == 0 and n_checked == EXPECTED_FOLD_CHANGE_CELLS
    return LevelResult(
        level=Level.L3,
        name="fold_change_recoverable_from_one_record",
        passed=passed,
        message=(
            f"{n_checked} treated records reproduce the released fold change, worst "
            f"relative error {worst:.3g}"
            if passed
            else f"{len(mismatched)} mismatched, {missing} unjoinable, {n_checked} of "
            f"{EXPECTED_FOLD_CHANGE_CELLS} checked"
        ),
        details={
            "n_checked": n_checked,
            "n_expected": EXPECTED_FOLD_CHANGE_CELLS,
            "n_unjoinable": missing,
            "max_relative_error": worst,
            "tolerance": FOLD_CHANGE_TOLERANCE,
            "mismatched": mismatched,
        },
    )


def predose_reads_carry_no_drug(records: Iterable[Mapping[str, Any]]) -> LevelResult:
    """L3: an environmental edit appears exactly on the treated reads from hour five.

    The dose went in at hour five, so this row is the record-level statement of that
    fact: a treated arm's reads before hour five carry no perturbation, its reads from
    hour five carry exactly one, and the untreated arm never carries any.
    """
    wrong: list[dict[str, Any]] = []
    counts = {"untreated": 0, "treated_predose": 0, "treated_dosed": 0}
    for record in records:
        experiment = record["experiment"]
        arm_header = str(experiment["phenotype"]["well_id"]).split("|")[0]
        hour = float(experiment["environment"]["duration_hours"])
        n_perturbations = len(experiment["environment"]["perturbations"])
        if arm_header == UNTREATED.header:
            expected = 0
            counts["untreated"] += 1
        elif hour < TREATMENT_START_HOUR:
            expected = 0
            counts["treated_predose"] += 1
        else:
            expected = 1
            counts["treated_dosed"] += 1
        if n_perturbations != expected and len(wrong) < 20:
            wrong.append(
                {
                    "arm": arm_header,
                    "hours": hour,
                    "n": n_perturbations,
                    "want": expected,
                }
            )
    return LevelResult(
        level=Level.L3,
        name="the_drug_is_on_the_reads_from_hour_five_only",
        passed=not wrong,
        message=(
            f"{counts['untreated']} untreated, {counts['treated_predose']} pre-dose "
            f"and {counts['treated_dosed']} dosed reads carry the right edits"
            if not wrong
            else f"{len(wrong)} reads carry the wrong number of environmental edits"
        ),
        details={**counts, "wrong": wrong},
    )


def run_verification(data_root: str | None = None) -> VerificationReport:
    """Run the promoter-activity verifier (L0-L4) on the built dev LMDB.

    The resolver is MG1655's, the host the records are written against. The store is
    streamed, so 69,480 records are never materialized at once. Three rows are appended
    to the family gate: the fold-change recovery, the dose placement, and the provenance
    audit of every ``SOURCED_VALUES`` entry. The report is written to
    ``preprocess/verification_report.json``.
    """
    from torchcell.verification.promoter_activity import (
        verify_promoter_activity_dataset_streaming,
    )
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _genome_for_reference,
        _write_report,
        stream_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, DATASET_ROOT_REL)
    first = next(iter(stream_records(abs_root)))
    reference = first["reference"]["genome_reference"]
    genome = _genome_for_reference(reference, base)
    universe = _gene_set_for_reference(reference, base)
    report = verify_promoter_activity_dataset_streaming(
        lambda: stream_records(abs_root),
        dataset_name=PromoterReporterMohiuddin2022Dataset.__name__,
        provenance=VERIFIER_PROVENANCE,
        expected_count=EXPECTED_RECORDS,
        resolve_gene_name=genome.resolve_gene_name,
    )
    report.add(
        promoters_are_in_the_host_gene_universe(stream_records(abs_root), universe)
    )
    report.add(fold_change_recoverable(stream_records(abs_root)))
    report.add(predose_reads_carry_no_drug(stream_records(abs_root)))
    mirror = Path(base) / "torchcell-raw"
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, mirror))
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def promoters_are_in_the_host_gene_universe(
    records: Iterable[Mapping[str, Any]], universe: set[str]
) -> LevelResult:
    """L4: every resolved promoter gene is a locus of the host's own gene universe."""
    genes: set[str] = set()
    n_null = 0
    for record in records:
        gene = record["experiment"]["phenotype"]["promoter_gene"]
        if gene is None:
            n_null += 1
            continue
        genes.add(str(gene))
    outside = sorted(genes - universe)
    return LevelResult(
        level=Level.L4,
        name="promoter_genes_in_the_host_gene_universe",
        passed=not outside,
        message=(
            f"{len(genes)} promoter genes of {len(universe)} MG1655 loci; "
            f"{n_null} records carry no gene"
            if not outside
            else f"{len(outside)} promoter genes are outside the host gene universe"
        ),
        details={
            "n_genes": len(genes),
            "n_universe": len(universe),
            "n_null_records": n_null,
            "outside": outside[:20],
        },
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.mohiuddin2022"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser(
        "deposit", help="retrieve (optional) and deposit the raw mirror"
    )
    deposit.add_argument("--download-dir", required=True)
    deposit.add_argument(
        "--retrieve",
        action="store_true",
        help="run the recorded PMC bucket retrievers into --download-dir first",
    )
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        download = Path(args.download_dir)
        if args.retrieve:
            retrieve_raw_files(download)
        sources: dict[str, str | Path] = {
            raw.name: download / raw.name for raw in RAW_FILES
        }
        print(deposit_raw_mirror(sources=sources, data_root=data_root))
        return 0
    if args.command == "build":
        dataset = PromoterReporterMohiuddin2022Dataset(
            root=osp.join(data_root, DATASET_ROOT_REL)
        )
        print(f"len = {len(dataset)}")
        return 0
    report = run_verification(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
