# torchcell/datasets/ecoli/ishii2007
# [[torchcell.datasets.ecoli.ishii2007]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/ishii2007
# Test file: tests/torchcell/datasets/ecoli/test_ishii2007.py
"""Ishii 2007 paired metabolome, proteome and 13C flux of 28 Keio chemostat cultures.

Ishii et al. 2007 (Science 316:593, doi:10.1126/science.1132067) ran four quantitative
layers on ONE set of glucose-limited chemostat cultures: "We collected data using these
multiple highthroughput analyses for wild-type $E$ coli K-12 and 24 single-gene
disruptants, which were selected from the Keio collection (13)". Three of those layers
are served here, one dataset class each, because
``verify_protein_dataset``-style L3 gates allow ONE ``measurement_type`` per dataset and
these are three different quantities on three different scales:

- :class:`MetabolomeIshii2007Dataset` -- CE-TOFMS intracellular concentrations in mM
  (``MetabolitePhenotype``), 28 records.
- :class:`ProteomeIshii2007Dataset` -- LC-MS/MS absolute abundances in
  mg-protein/g-dry-cell-weight (``ProteinAbundancePhenotype``), 28 records.
- :class:`FluxIshii2007Dataset` -- fitted 13C net fluxes as a percentage of the specific
  glucose uptake rate (``FluxPhenotype``), 28 records.

EACH ARM IS 24 DISRUPTANTS AT 0.2 h-1 PLUS THE WILD TYPE AT FOUR OTHER DILUTION RATES.
``Environment.dilution_rate_per_hour`` (#753) is set on every environment this module
builds, so the four wild-type cultures the release calls ``GR01``-``GR04`` are four
distinct environments and are records rather than a drop. Their rate comes off the
release's own label row, not off the ``GR`` numbering: the ``IDs`` sheet's Sample Name
column and every data sheet's name row both read ``WT, 0.1h-1``, ``WT, 0.4h-1``,
``WT, 0.5h-1`` and ``WT, 0.7h-1``, ``check_dilution_rate_arm`` asserts the two agree
column by column, and the set of parsed rates must equal the paper's arm
(0.1, 0.2, 0.4, 0.5, 0.7 h-1) minus the 0.2 h-1 reference. A wild-type culture carries
``Genotype(perturbations=[])``, so what distinguishes those four records from one
another, and from the reference, is the dilution rate alone.

THE FLUX ARM NEEDED NO SCHEMA CHANGE. ``FluxPhenotype`` already states that a flux map
is a FIT of a whole network and keeps the interval an interval; this release fits one
flux map per culture from that culture's own GC-MS mass-isotopomer distributions, which
is the second mirrored file, so the fit's INPUT travels with the record. The release
publishes no confidence bound at all, so ``net_flux_lower``, ``net_flux_upper`` and
``confidence_level`` are left ``None`` and typed as a gap rather than having one number
promoted to "the" statistic. ``flux phenotype`` is likewise an already-declared graph
node class with its own ``CellAdapter`` methods, so nothing in
``biocypher/config/torchcell_schema_config.yaml`` changes either.

RETRIEVAL: THE PUBLISHER IS BLOCKED, THE PAPER'S OWN CITED WEB SITE IS NOT. The row's
accession read "Science supporting online material; no repository accession found", and
the supporting online material is indeed unreachable by script: measured 2026-10-08,
``https://www.science.org/doi/suppl/10.1126/science.1132067`` answers HTTP 403 with a
Cloudflare JavaScript challenge ("Enable JavaScript and cookies to continue",
``window._cf_chl_opt``), the DOI has no PMCID ("Identifier not found in PMC"), and the
publisher's own free-access ``ijkey`` link printed on the project web site answers 403
too, so the block is at the edge and not an entitlement. The data is NOT in that
supplement, though: the paper's reference 21 is a project web site, and it serves the
complete quantitative release over plain HTTP with no challenge and no cookie. Both
consumed workbooks are therefore ``RetrievalMethod.direct_url`` and re-retrievable. What
stays retrieval-gated is the supplement's TEXT (Materials and Methods, SOM text, figs.
S1-S10, tables S1-S11), and the one thing the records need from it is the medium; the
manual recipe for it is written into the raw mirror's ``si_expected``.

THE MEDIUM IS THE ONE UNPINNED ENVIRONMENT VALUE, AND IT IS LABELLED AS A READING. No
mirrored byte names the medium: the paper body says only "in glucose-limited chemostat
cultures" and the project web site's workbooks state units, replicate structure and cell
parameters but no recipe. ``MEDIUM`` is therefore built in this module (never added to
the shared ``MEDIA_LIBRARY``) as a single ``composition_deferred`` component that defers
to the supplement, and ``is_synthetic=True`` is a READING, NOT A STATEMENT: the paper
says glucose is the growth-limiting substrate whose concentration the dilution rate sets,
and a complex medium's other carbon sources would relieve that limitation. It is
recorded in ``build_accounting.json`` under ``unpinned_environment_values`` and carried
as a ``ProvenanceGap`` on every environment. ``temperature`` is ``None`` for the same
reason, also as a typed gap. ``aerobicity="aerobic"`` is NOT a default left in place: it
is asserted at build time from the release's own ``Specific_Rates`` sheet, where every
loaded culture has a positive oxygen uptake rate.

RECORDS DROPPED, WITH THE ARITHMETIC. Each sheet's sample columns are classified in one
fixed order, so each dropped column has exactly one reason:

1. ``no_data_in_this_layer`` -- the column is empty in this sheet. ``KO05x`` (the first
   of the two pfkA cultures) in the Protein and Flux sheets, and ``GR04x`` in all three.
   ``GR04x`` is the release's second mRNA measurement of the 0.7 h-1 wild type ("Used
   for 2nd measurement of mRNAs."), so it is empty in all three served layers and is a
   fact about the bytes, not a schema limit.
2. ``reference_sample`` -- an ``RF`` column; the reference, not a record.
3. ``duplicate_culture_same_genotype_and_environment`` -- ``KO05x`` in the Metabolite
   sheet, where it does carry data. pfkA is the one disruptant cultured twice
   ("pfkA disruptant was cultured twice."), and both cultures are pfkA at 0.2 h-1, so
   they would collide on genotype AND environment. ``KO05`` is kept because it is the
   culture measured in all three served layers, which is what keeps the three datasets
   paired on one culture.

Metabolite: 35 columns - 1 empty - 5 reference - 1 duplicate = **28**.
Protein: 36 - 2 empty - 6 reference = **28**. Flux: 34 - 2 - 4 = **28**.

THE RULE THIS LOADER NO LONGER CARRIES is ``culture_not_batch``, which the landed
Schmidt 2016 and Lamoureux 2023 loaders state under the same name. It dropped
``GR01``-``GR04`` on the ground that ``Environment`` had no dilution-rate slot, so five
cultures would have shared one byte-identical environment. The slot exists, the four
cultures are loaded, and the rule is retired here rather than narrowed, because nothing
is left for it to name: every sample column of every served sheet is kept or named by
one of the three rules above. ``build_accounting.json`` records the retirement under
``retired_drop_rules``.

THE REFERENCE IS SERIES-MATCHED WHERE THE RELEASE STATES A SERIES, AND IT IS THE 0.2 h-1
WILD TYPE FOR EVERY RECORD. The workbook labels every sample with a measurement series
and documents which columns are the controls ("Column BL- | Control (Wild type, cultured
at a dilution rate of 0.2h-1)"), so each Metabolite and Protein record's
``experiment_reference`` is the reference column of its OWN series, not a pooled average.
The Samples table assigns that one control block to the whole sheet, the dilution-rate
block ("Column BA-BJ | Wild type, cultured at various dilution rates") included, and the
``IDs`` sheet puts each ``GR`` column in a series that has an ``RF`` column: Protein
series 4 for all four, Metabolite series 5 for ``GR01``-``GR03`` and series 4 for
``GR04``. So a dilution-rate record's reference is a REAL released control culture, and
it is a CROSS-ENVIRONMENT one: its ``environment_reference`` carries
``dilution_rate_per_hour=0.2`` while the record carries its own rate, which is what makes
the comparison the paper asks for ("To allow a comparison of the effects of these genetic
perturbations with the effects of environmental perturbations") visible in the stored
bytes instead of hidden in a matching denominator. The Flux sheet carries no series row,
so that arm uses ``RF03``, the one reference culture measured in all three served layers;
the other three reference fits are written to ``preprocess/flux_reference_fits.csv`` so
nothing is lost, and the choice is flagged in ``build_accounting.json``.

IDENTIFIERS, MEASURED. Every stored gene identifier is a BW25113 locus tag from the
pinned GenBank annotation through :func:`reconcile_locus_tags`, because the collection is
the Keio collection and Ishii never names its background: Baba 2006 (mirrored) does.
24 of 24 disruptant symbols resolve (23 through the gene-symbol layer, 1 -- ``gapC`` --
through a gene synonym onto a non-gene feature) with no collision and no ambiguity.
66 of the 67 quantified protein symbols resolve; ``gpmG`` is retired on this annotation
and carries NO value in any of the 36 Protein columns, so it never becomes a key, and
``check_gpmg_unused`` asserts that emptiness rather than trusting it. The four
dilution-rate records have no perturbation at all, so they contribute no identifier.

WHAT IS NOT LOADED, AND WHY. The qRT-PCR mRNA layer (85 transcripts, absolute copy
number per ug total RNA) has no home in the schema: the three expression phenotypes are
a microarray log2 ratio, an RNA-seq TPM that also requires raw mapped-read counts, and a
pseudobulk log2 ratio, and an absolute transcript copy number is none of them. Filing it
under any of them would misreport the assay, so it is left out and reported as a typed
schema gap. The genome-wide DNA-array arm (4,213 oligos, seven sample-vs-control ratio
blocks) would fit ``MicroarrayExpressionPhenotype`` but has no ``Bacterial`` experiment
pair to carry the assembly pin. Both arms' files are listed in the raw mirror's
``si_expected`` with their measured sizes, so a later revision retrieves them by the
same scriptable route.
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
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
import xlrd
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
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    BacterialDeletionPerturbation,
    BacterialMetaboliteExperiment,
    BacterialMetaboliteExperimentReference,
    BacterialProteinAbundanceExperiment,
    BacterialProteinAbundanceExperimentReference,
    ComponentDefinition,
    Compound,
    Environment,
    Experiment,
    ExperimentReference,
    FluxExperiment,
    FluxExperimentReference,
    FluxPhenotype,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    MetabolitePhenotype,
    ProteinAbundancePhenotype,
    Publication,
    SampleUnit,
    StrainConstruction,
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
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.report import Provenance, VerificationReport
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
CITATION_KEY = "ishiiMultipleHighThroughputAnalyses2007"
PAPER_DOI = "10.1126/science.1132067"
PUBMED_ID = "17379776"
TITLE = (
    "Multiple High-Throughput Analyses Monitor the Response of E. coli to Perturbations"
)
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "1d30256f408a939cf1cf400c03f36a124a8274627393602cabcb92cf72394065"
#: Baba 2006, the Keio construction paper Ishii defers the collection to.
BABA_KEY = "babaConstructionEscherichiaColi2006"
BABA_SHA256 = "ca71475baf25d070562c70296d5aad8ec17a59d5b843f264b1e226362b6daf0d"

#: The project web site the paper's reference 21 names; it serves the whole release.
PROJECT_SITE = "http://ecoli.iab.keio.ac.jp/"
#: The publisher supplement, reachable only through a browser (see the module docstring).
PUBLISHER_SUPPLEMENT_URL = "https://www.science.org/doi/suppl/10.1126/science.1132067"
DATA_RETRIEVED_AT = "2026-10-08"

QUANTITATIVE_FILE = "Quantitative_data.xls"
GC_MS_FILE = "Flux_GC-MS_data.xls"

SHEET_INFORMATION = "Information"
SHEET_IDS = "IDs"
SHEET_MRNA = "mRNA"
SHEET_PROTEIN = "Protein"
SHEET_METABOLITE = "Metabolite"
SHEET_FLUX = "Flux"
SHEET_RATES = "Specific_Rates"

#: The ``IDs`` sheet's roster: a header cell reading this in this column, then one row
#: per sample with the Sample ID here and the Sample Name in the next column.
IDS_SAMPLE_ID_HEADER = "Sample ID"
IDS_SAMPLE_ID_COLUMN = 1
#: How the release writes a dilution-rate culture's name, in the ``IDs`` sheet's Sample
#: Name column and in every data sheet's own name row: ``WT, 0.1h-1``.
DILUTION_RATE_NAME = re.compile(r"^WT, (?P<rate>[0-9]+(?:\.[0-9]+)?)h-1$")

#: ``Specific_Rates`` row labels this loader reads.
OXYGEN_UPTAKE_ROW = "Oxygen uptake rate (OUR)"
GLUCOSE_UPTAKE_ROW = "Glucose consumption rate"

#: The ``Metabolite`` sheet encodes the CE-TOFMS protocol as the fill colour of the name
#: cell, and the ``Information`` sheet prints the legend as three coloured swatches.
PROTOCOL_LEGEND_ROWS: tuple[tuple[int, str], ...] = (
    (78, "anion"),
    (79, "cation"),
    (80, "nucleotide"),
)
PROTOCOL_LEGEND_COLUMN = 1

#: Measured on the pinned workbook: protocol -> number of metabolite rows.
EXPECTED_PROTOCOL_ROWS: dict[str, int] = {"anion": 204, "cation": 314, "nucleotide": 61}
#: The one metabolite name the release uses twice (once per protocol).
DUPLICATE_METABOLITE_NAME = "Citrate"
#: The quantified protein the pinned BW25113 annotation has retired; it carries no value.
RETIRED_PROTEIN_SYMBOL = "gpmG"
#: Every disruptant symbol must land on a locus tag of the pinned namespace.
MIN_RESOLVED_FRACTION = 1.0

MEASUREMENT_TYPE_METABOLITE = "ce_tofms_intracellular_concentration_mm"
MEASUREMENT_TYPE_PROTEIN = "lc_ms_ms_absolute_mg_protein_per_g_dry_cell_weight"
MEASUREMENT_TYPE_FLUX = "c13_mfa_net_flux_percent_of_specific_glucose_uptake"

#: Records of the full build, MEASURED on the pinned workbook: the 24 disruptants at
#: 0.2 h-1 plus the wild type at 0.1, 0.4, 0.5 and 0.7 h-1.
EXPECTED_RECORDS: dict[str, int] = {
    "metabolome_ishii2007": 28,
    "proteome_ishii2007": 28,
    "flux_ishii2007": 28,
}
#: The flux reference fit this arm stores; the sheet carries no series row.
FLUX_REFERENCE_SAMPLE = "RF03"
#: Record-side abundances dropped because their own series reference did not detect that
#: protein, MEASURED over the pinned workbook: 1 of 1,349. ``verify_protein_dataset``'s
#: L3 requires a record's protein keys to EQUAL its reference's, so a protein the mutant
#: detected and the reference did not has no baseline and cannot be stored; the count is
#: pinned here so a corrected re-export that changes it stops the build instead of
#: silently dropping more.
EXPECTED_PROTEIN_KEYS_WITHOUT_REFERENCE = 1


class RawFile(BaseModel):
    """One file the loader consumes: its pinned bytes and how they were retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str
    bytes: int
    description: str

    @property
    def mirror_relpath(self) -> str:
        """Path inside the raw mirror (``data/<name>``)."""
        return f"data/{self.name}"

    @property
    def source_url(self) -> str:
        """The project web site URL these bytes came from."""
        return f"{PROJECT_SITE}{self.name}"

    @property
    def retrieval(self) -> RetrievalRecord:
        """The re-runnable retrieval of these bytes (a plain GET, no challenge)."""
        return RetrievalRecord(
            method=RetrievalMethod.direct_url,
            source_url=self.source_url,
            retriever="torchcell.literature.retrieve.direct_url",
            params={"url": self.source_url},
            sha256=self.sha256,
            retrieved_at=DATA_RETRIEVED_AT,
        )


RAW_FILES: tuple[RawFile, ...] = (
    RawFile(
        name=QUANTITATIVE_FILE,
        sha256="2b7663f505af2137d31697a4d316eb1f7ffd24274f73e4844d27d03348313888",
        bytes=824320,
        description="the targeted quantitative release, nine sheets: Information (units, "
        "replicate structure, cell parameters), IDs (the sample roster with culture "
        "dates and per-layer series), mRNA, Protein, Metabolite, Flux, Specific_Rates, "
        "Extracellular and Cell_composition. The three served datasets all read this "
        "one file",
    ),
    RawFile(
        name=GC_MS_FILE,
        sha256="7f185d05c19a610c8739d79dc310b73689c06ee9e37f5a2e64f52a98f67aa555",
        bytes=214016,
        description="the INPUT of the flux fit: 'Mass distributions of proteinogenic "
        "amino acids measured by GC-MS.', one sheet per culture (33 cultures, including "
        "pfkA_1, which has labeling data but no fitted flux column). Mirrored so a "
        "stored fitted flux names the data it was fitted to",
    ),
)
#: ``{raw file name: pinned sha256}``, the build-time check of every consumed file.
DATA_SHA256: dict[str, str] = {f.name: f.sha256 for f in RAW_FILES}
RAW_FILES_BY_NAME: dict[str, RawFile] = {f.name: f for f in RAW_FILES}

#: What the release holds that this build does not consume. Byte sizes measured by HEAD
#: on 2026-10-08; every one is the same scriptable plain GET as the two consumed files.
NOT_MIRRORED: tuple[str, ...] = (
    f"{PUBLISHER_SUPPLEMENT_URL} -- the Science supporting online material (Materials "
    "and Methods, SOM text, figs. S1-S10, tables S1-S11). NOT deposited: measured "
    "2026-10-08, science.org answers HTTP 403 with a Cloudflare JavaScript challenge "
    "('Enable JavaScript and cookies to continue', window._cf_chl_opt) for the "
    "supplement page, for the article PDF, and for the publisher's own free-access "
    "ijkey link printed on the project web site; the DOI has no PMCID ('Identifier not "
    "found in PMC'). MANUAL RECIPE: open "
    "https://www.science.org/doi/10.1126/science.1132067 in a browser, follow "
    "'Supplementary Materials', download the supporting-online-material PDF, then "
    "deposit it under si/ with RetrievalMethod.manual_browser and the sha256 of the "
    "bytes that arrive. It is the only source of the chemostat medium, the temperature "
    "and the flux-fitting procedure, which are typed gaps on the records until then",
    f"{PROJECT_SITE}Metabolome_UK_data.xls (17,547,264 B) -- the unknown-peak CE-TOFMS "
    "matrices (six sheets, 4,051-11,470 rows). NOT consumed: unidentified peaks have no "
    "metabolite identity to key on",
    f"{PROJECT_SITE}DNAArray_data.xls (2,789,376 B) -- the genome-wide DNA-microarray "
    "release, 4,213 oligos x seven sample-vs-control Control/Sample/Ratio blocks. NOT "
    "consumed: it would fit MicroarrayExpressionPhenotype, but there is no Bacterial "
    "microarray experiment pair to carry the assembly pin",
    f"{PROJECT_SITE}DNAArray_raw-data.zip (15,122,608 B) -- the array scanner output. "
    "NOT consumed: no loader consumes raw array data",
    f"{PROJECT_SITE}2D-DIGE_ratio_data.xls (1,842,688 B) and 2D-DIGE_ID_list.xls "
    "(114,176 B) -- the 2D-DIGE spot ratios (2,325 spots x 28 blocks) and the "
    "per-disruptant spot identifications. NOT consumed: the ratio table is keyed by gel "
    "spot, not by protein, and the paper calls this layer semiquantitative",
    "the qRT-PCR mRNA sheet of the consumed workbook (85 transcripts, copy "
    "number/ug-total RNA). The bytes ARE mirrored; no dataset class serves them, "
    "because no phenotype holds an absolute transcript abundance",
)


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the sha256-pinned Ishii ``paper.md``."""
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


def _baba(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to Baba 2006, the Keio paper Ishii defers the collection to."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=BABA_KEY,
            sha256=BABA_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
        ),
    )


def _workbook(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to the ``Information`` sheet of the pinned quantitative workbook.

    ``source_uri`` is relative to ``$DATA_ROOT/torchcell-raw/<citation_key>/``;
    :func:`sourced_value_root` resolves which mirror a quote belongs to.
    """
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=RAW_FILES_BY_NAME[QUANTITATIVE_FILE].mirror_relpath,
            citation_key=CITATION_KEY,
            sha256=DATA_SHA256[QUANTITATIVE_FILE],
            method="the project web site's quantitative workbook (torchcell-raw mirror)",
            page="sheet 'Information'",
        ),
    )


def _ids_sheet(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to the ``IDs`` sheet of the pinned quantitative workbook."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=RAW_FILES_BY_NAME[QUANTITATIVE_FILE].mirror_relpath,
            citation_key=CITATION_KEY,
            sha256=DATA_SHA256[QUANTITATIVE_FILE],
            method="the project web site's quantitative workbook (torchcell-raw mirror)",
            page="sheet 'IDs'",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_DISRUPTANT_QUOTE = (
    "We collected data using these multiple highthroughput analyses for wild-type $E$ "
    "coli K-12 and 24 single-gene disruptants, which were selected from the Keio "
    "collection (13) and cover most viable glycolysis and pentose phosphate pathway "
    "disruptants (i.e., galM, glk, pgm, pgi, pfkA, pfkB, fbp, fbaB, gapC, gpmA, gpmB, "
    "pykA, pykF, ppsA, zwf, pgl, gnd, rpe, rpiA, rpiB, tktA, tktB, talA, and talB)."
)
_CULTURE_QUOTE = (
    "The cells were grown at a single fixed dilution rate of 0.2 hours−1 in "
    "glucose-limited chemostat cultures, and wildtype cells cultured at the same "
    "specific growth rate were used as a reference sample for comparison."
)
_DILUTION_RATE_QUOTE = (
    "To allow a comparison of the effects of these genetic perturbations with the "
    "effects of environmental perturbations, wild-type cells were examined at several "
    "different dilution rates (0.1, 0.2, 0.4, 0.5, and $0 . 7 \\ \\mathrm { h o u r s } "
    "^ { - 1 }$ ."
)
_LIMITING_SUBSTRATE_QUOTE = (
    "In chemostat cultures, the concentration of growth-limiting substrate can be "
    "controlled by the dilution rate $( I 4 )$ . the dilution rate was thus varied in "
    "this study from an almost glucose-starved state to a nearly unlimited glucose "
    "supply."
)
_ASSAY_QUOTE = (
    "For more detailed analysis, in addition to qRT-PCR and metabolic flux analysis, we "
    "used capillary electrophoresis time-of-flight mass spectrometry (CE-TOFMS) for "
    "metabolome analysis (10, 11) and liquid chromatography tandem mass spectrometry "
    "(LC-MS/MS) for absolute quantification of proteins (12)."
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "disruptants": _paper(
        24,
        _DISRUPTANT_QUOTE,
        note="the 24 symbols this loader expects to find as sample names; process() "
        "refuses any other set",
    ),
    "dilution_rate_per_hour": _paper(
        0.2,
        _CULTURE_QUOTE,
        note="the dilution rate of every disruptant culture AND of every RF control "
        "column ('Control (Wild type, cultured at a dilution rate of 0.2h-1)'). It is "
        "stored on Environment.dilution_rate_per_hour (#753), so a record states the "
        "rate it was grown at instead of the rate being named only on the dataset",
    ),
    "dilution_rate_arm": _paper(
        [0.1, 0.2, 0.4, 0.5, 0.7],
        _DILUTION_RATE_QUOTE,
        note="the wild-type arm, all five rates of it loaded: 0.2 h-1 is the reference "
        "culture every record is compared against, and the other four are records of "
        "their own environment. check_dilution_rate_arm asserts that the rates parsed "
        "off the release's label row are exactly this list minus 0.2",
    ),
    "glucose_limited": _paper(
        "glucose",
        _LIMITING_SUBSTRATE_QUOTE,
        note="A READING, NOT A STATEMENT, for Media.is_synthetic: glucose being the "
        "sole growth-limiting substrate whose concentration the dilution rate sets "
        "requires a defined medium, since another carbon source would relieve the "
        "limitation. No mirrored byte names the medium or its recipe; the recipe is in "
        "the Cloudflare-gated supplement. Recorded in build_accounting.json under "
        "unpinned_environment_values",
    ),
    "assays": _paper(
        ["CE-TOFMS", "LC-MS/MS", "qRT-PCR", "metabolic flux analysis"],
        _ASSAY_QUOTE,
        note="the four quantitative layers; this module serves the CE-TOFMS metabolome, "
        "the LC-MS/MS proteome and the flux analysis",
    ),
    "aerobic": _paper(
        "aerobic",
        "The observed increase in the expression of phosphoenolpyruvate carboxylase "
        "(Ppc), decrease in the expression of phosphoenolpyruvate carboxykinase (PckA), "
        "and repression of glyoxylate shunt enzymes are also logical responses to "
        "enhance the tricarboxylic acid cycle flux (2), which corresponds to energy "
        "production under aerobic conditions and the components of which are important "
        "precursors in biomass production.",
        note="corroborated by the release itself and asserted at build time: every "
        "loaded culture has a positive oxygen uptake rate in the Specific_Rates sheet",
    ),
    "project_site": _paper(
        PROJECT_SITE,
        "21.http://lecoli.iab.keio.ac.jp",
        note="reference 21, which the body introduces as 'Additional information is "
        "available at the project Web site (21).' The OCR prefixes an 'l'; the PDF's "
        "own text layer (paper.pdf, sha256 "
        "a58d682321550961355783c58a8e05c65751efe41f211042a2890d925346af26) reads "
        "'21. http://ecoli.iab.keio.ac.jp', which is the live host both consumed "
        "workbooks were retrieved from",
    ),
    "targeted_component_counts": _paper(
        {"metabolites": 130, "proteins": 57, "mRNA transcripts": 85},
        "The numbers of individual components shown are metabolites, 130; proteins, 57; "
        "and mRNA transcripts, 85.",
        note="Fig. 1's counts are of components detected in MORE THAN HALF the samples, "
        "so they are not the sheet row counts (579 metabolite rows, 67 protein rows, 85 "
        "mRNA rows) and not a per-record key count; recorded, not asserted",
    ),
    "background_strain": _baba(
        "BW25113",
        "The Keio collection is comprised of 3985 deletions in duplicate (7970 total) "
        "of E. coli K-12 strain BW25113 (Datsenko and Wanner, 2000)",
        note="the deferral target of Ishii's '(13)'; Ishii itself says only 'E. coli "
        "K-12', which is the deliberately ambiguous label",
    ),
    "cassette": _baba(
        "kanamycin cassette flanked by FLP recognition target sites",
        "Open-reading frame coding regions were replaced with a kanamycin cassette "
        "flanked by FLP recognition target sites",
        note="Ishii does not say the cassette was excised, so the collection strain "
        "(kanamycin resistant, cassette in place) is what is recorded",
    ),
    "metabolite_unit": _workbook(
        "mM",
        "Metabolites (intracellular)",
        note="the Information sheet's Measurements table pairs this label with the unit "
        "mM; the paired cell is asserted at build time",
    ),
    "protein_unit": _workbook(
        "mg-protein/g-dry cell weight",
        "Proteins",
        note="the Information sheet's Measurements table pairs this label with this "
        "unit; the paired cell is asserted at build time",
    ),
    "flux_normalization": _workbook(
        "percent of the specific glucose uptake rate",
        "All fluxes are normalized to specific glucose uptake rate.",
        note="with the Measurements table's unit for Flux, '%-substrate uptake'. The "
        "Specific_Rates sheet's glucose consumption rate per culture is written to "
        "preprocess/specific_rates.csv so an absolute flux is derivable, but the stored "
        "number is the released normalized one",
    ),
    "flux_excluded_reaction": _workbook(
        "-",
        '"-" denotes reaction excluded from model.',
        note="a dash is stored as KEY ABSENCE on the phenotype, never as a zero flux",
    ),
    "flux_exchange_coefficient": _workbook(
        "Exch",
        '"Exch" denotes exchange coefficient of corresponding reaction.',
        note="the seven Exch. rows are NOT net fluxes and are not stored on "
        "FluxPhenotype; they go to preprocess/exchange_coefficients.csv, because no "
        "field holds a reversibility parameter",
    ),
    "protein_replicates": _workbook(
        2,
        'The CVs reported in columns C to BH in sheet "Protein" capture the technical '
        "variation (sample processing, digestion and quantification) in duplicate "
        "sample preparation and independent LC-MS/MS measurements performed in parallel "
        "from the same sampling of a single continous culture for each condition "
        "(column).",
        note="n_replicates for every protein key. The released statistic is a "
        "coefficient of variance in percent, so the stored SE is "
        "level * CV / 100 / sqrt(2); the sheet also says 'For each sample, measurements "
        "by LC-MS/MS were perfomed twice.' and reports 1, 2 or 3 peptides per protein, "
        "which could put the CV over as many as 6 values. With no per-record peptide "
        "column the CONSERVATIVE lower end of that range is taken (n = 2, the larger "
        "SE), per the range-resolution rule",
    ),
    "protein_peptides": _workbook(
        {"one": ["GpmG", "PfkA"], "two": ["LdhA", "Rpe", "RpiB", "SdhB", "Zwf"]},
        "For GpmG and PfkA, one peptide per protein was measured.",
        note="with 'For LdhA, Rpe, RpiB, SdhB and Zwf, two peptides per protein were "
        "measured.' and 'For other proteins, three peptides per protein were measured.' "
        "Written to preprocess/protein_peptides.csv; not stored on any record, because "
        "a peptide count is not a replicate count",
    ),
    "not_detected": _workbook(
        "not detected",
        'In the protein and metabolite sheet, blank cells denote "not detected" which '
        "means either not detectable or below detection limit when detectable.",
        note="a blank is stored as KEY ABSENCE, never as a zero level",
    ),
    "metabolite_protocols": _workbook(
        ["anion", "cation", "nucleotide"],
        "In the metabolite sheet, the colors of column A (metabolite names) denote "
        "measurement protocols.",
        note="the legend is three coloured swatches beside 'Method :Anion', 'Method: "
        "Cation' and 'Method: Nucleotide'; the loader reads the fill colours and refuses "
        "a workbook whose protocol row counts are not 204 / 314 / 61. The protocol is "
        "the separation method, not the quantity, so all three share one "
        "measurement_type",
    ),
    "cell_parameters": _workbook(
        {"volume_litres": "4.96x10-16", "dry_weight_grams": "2.8x10-13"},
        "To calculate metabolite concentrations, the following parameters were used:",
        note="paired in the sheet with 'Cell volume' 4.96x10-16 (L) and 'Cell dry "
        "weight' 2.8x10-13 (g), referenced to Neidhardt and Curtiss 1996. They are how "
        "the released mM concentration was computed; recorded, not stored",
    ),
    "pfka_cultured_twice": _workbook(
        "pfkA",
        "pfkA disruptant was cultured twice.",
        note="the two cultures are KO05x and KO05, both pfkA at 0.2 h-1, so one of them "
        "is dropped under duplicate_culture_same_genotype_and_environment",
    ),
    "tpia_washed_out": _workbook(
        "tpiA",
        "tpiA (a member of glycolysis pathway) disruptant could not be cultured because "
        "of reactor wash out at a dilution rate of 0.2h-1.",
        note="the intended disruptant that produced no culture and no data in any "
        "layer. Recorded so the panel's 24 is understood as 25 attempted",
    ),
    "gr04x_second_mrna_measurement": _ids_sheet(
        "GR04x",
        "Used for 2nd measurement of mRNAs.",
        note="the IDs sheet's Memo for GR04x, a SECOND wild-type culture at 0.7 h-1 "
        "(Culture Date 38693 against GR04's 38631), so GR04 and GR04x are the second "
        "pair of cultures this release grew at one genotype and one dilution rate. "
        "GR04x carries no value in the Protein, Metabolite or Flux sheets, so all three "
        "served layers drop it under no_data_in_this_layer; GR04 is named in "
        "PREFERRED_DUPLICATE_SAMPLES so the unserved mRNA layer, where both carry data, "
        "also resolves to one record per culture",
    ),
    "reference_columns": _workbook(
        "Control (Wild type, cultured at a dilution rate of 0.2h-1)",
        "Control (Wild type, cultured at a dilution rate of 0.2h-1)",
        note="the Samples table labels the trailing column block of every sheet this "
        "way, which is what makes the RF columns references rather than records",
    ),
}

DISRUPTANT_SYMBOLS: tuple[str, ...] = (
    "galM",
    "glk",
    "pgm",
    "pgi",
    "pfkA",
    "pfkB",
    "fbp",
    "fbaB",
    "gapC",
    "gpmA",
    "gpmB",
    "pykA",
    "pykF",
    "ppsA",
    "zwf",
    "pgl",
    "gnd",
    "rpe",
    "rpiA",
    "rpiB",
    "tktA",
    "tktB",
    "talA",
    "talB",
)
#: The release suffixes the two cultures of the one twice-grown disruptant ``pfkA_1``
#: and ``pfkA_2``; every other sample name is the bare gene symbol.
CULTURE_REPLICATE = re.compile(r"^(?P<symbol>[A-Za-z]+)_(?P<replicate>[0-9]+)$")
#: The only sample names allowed to carry that suffix, measured on the pinned workbook.
SUFFIXED_SAMPLE_NAMES: frozenset[str] = frozenset({"pfkA_1", "pfkA_2"})
#: Which culture is the record when one genotype was grown twice at one dilution rate.
#: ``KO05`` (``pfkA_2``) is the pfkA culture measured in ALL THREE served layers --
#: ``KO05x`` (``pfkA_1``) has metabolite and mRNA data only, and no oxygen uptake rate --
#: so keeping ``KO05`` is what makes the three datasets' pfkA records the same culture.
#: Column order alone would have picked ``KO05x``, which is why this is named rather than
#: implicit. ``GR04`` is the same situation on the wild type at 0.7 h-1: ``GR04x`` is a
#: second culture at that rate ("Used for 2nd measurement of mRNAs.", Culture Date 38693
#: against 38631) and is EMPTY in all three served sheets, so the served arms drop it
#: under ``no_data_in_this_layer`` and never reach the duplicate rule; naming ``GR04``
#: keeps the unserved mRNA layer, where both carry data, resolvable to one record.
PREFERRED_DUPLICATE_SAMPLES: frozenset[str] = frozenset({"KO05", "GR04"})

COLLECTION = "Keio collection"
CASSETTE: str = SOURCED_VALUES["cassette"].value
#: The dilution rate of the reference culture and of all 24 disruptant cultures.
DILUTION_RATE_PER_HOUR: float = SOURCED_VALUES["dilution_rate_per_hour"].value
#: Every rate the wild type was run at, the reference rate included.
DILUTION_RATE_ARM: tuple[float, ...] = tuple(SOURCED_VALUES["dilution_rate_arm"].value)
AEROBICITY: str = SOURCED_VALUES["aerobic"].value

PUBLICATION = Publication(
    pubmed_id=PUBMED_ID,
    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
    doi=PAPER_DOI,
    doi_url=f"https://doi.org/{PAPER_DOI}",
)

# --------------------------------------------------------------------------- #
# The medium, and the two environment gaps it leaves open
# --------------------------------------------------------------------------- #
_SUPPLEMENT_LOOKED_IN = Provenance(
    source_uri=PAPER_MD,
    citation_key=CITATION_KEY,
    sha256=PAPER_MD_SHA256,
    method="full body read, plus every sheet of both mirrored workbooks and every page "
    f"of {PROJECT_SITE}; the recipe is in the supplement at {PUBLISHER_SUPPLEMENT_URL}, "
    "which answers HTTP 403 with a Cloudflare JavaScript challenge (manual recipe in "
    "the raw mirror's si_expected)",
    page="body, 'The cells were grown at a single fixed dilution rate'",
)

MEDIUM = Media(
    name="glucose-limited chemostat medium, composition deferred to the Science "
    "supporting online material (Ishii 2007)",
    state="liquid",
    is_synthetic=True,
    base_medium=None,
    components=[
        MediaComponent(
            compound=Compound(
                name="glucose-limited chemostat medium (no recipe stated)"
            ),
            role=MediaComponentRole.other,
            concentration=None,
            definition=ComponentDefinition.composition_deferred,
            provenance=[SOURCED_VALUES["glucose_limited"]],
            defers_to=[
                "Ishii 2007 Supporting Online Material, Materials and Methods "
                f"({PUBLISHER_SUPPLEMENT_URL}); not mirrored"
            ],
            note="the paper names no medium and no component amount, and the project "
            "web site's workbooks state units, replicate structure and cell parameters "
            "but no recipe, so not one gram or millimolar figure is copied in from "
            "another E. coli chemostat study. The carbon source stays inside this "
            "deferred line: 'glucose-limited' names the limiting substrate, not a "
            "concentration",
        )
    ],
    provenance=[SOURCED_VALUES["glucose_limited"], SOURCED_VALUES["dilution_rate_arm"]],
)
"""Ishii 2007's chemostat medium, kept in this module and OUT of ``MEDIA_LIBRARY``.

The library holds recipes; this object holds the absence of one, and sharing it would
invite a second loader to attach amounts to it.
"""

TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_SUPPLEMENT_LOOKED_IN,
    note="the cultivation temperature is in the same unmirrored supplement, so "
    "temperature is None rather than a guessed 37 C",
)
FLUX_INTERVAL_GAP = ProvenanceGap(
    field="confidence_level",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=RAW_FILES_BY_NAME[QUANTITATIVE_FILE].mirror_relpath,
        citation_key=CITATION_KEY,
        sha256=DATA_SHA256[QUANTITATIVE_FILE],
        method="every cell of the Flux sheet and every sheet of the GC-MS workbook",
        page="sheets 'Flux' and 'Information'",
    ),
    note="the release publishes ONE number per reaction and no interval, so "
    "net_flux_lower, net_flux_upper and confidence_level are all None. The fit's input "
    f"is mirrored ({GC_MS_FILE}) and its procedure is in the unmirrored supplement; "
    "nothing here promotes a bound to 'the' statistic because there is no bound",
)
METABOLITE_SE_GAP = ProvenanceGap(
    field="metabolite_level_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=RAW_FILES_BY_NAME[QUANTITATIVE_FILE].mirror_relpath,
        citation_key=CITATION_KEY,
        sha256=DATA_SHA256[QUANTITATIVE_FILE],
        method="the Metabolite sheet has ONE column per sample and no CV column; the "
        "Information sheet puts the only released spread in 'Column AN, AO, AP', which "
        "is the average, SD and CV across the five reference samples",
        page="sheets 'Metabolite' and 'Information'",
    ),
    note="each loaded culture is one CE-TOFMS measurement of one chemostat, so "
    "n_replicates is 1 and no per-record SE exists. The reference-series dispersion is "
    "written to preprocess/metabolites.csv rather than copied onto every record",
)


def environment(dilution_rate_per_hour: float) -> Environment:
    """One culture's environment: the medium, aerobic, no temperature, this rate.

    The dilution rate is REQUIRED rather than defaulted, because it is the only field
    that distinguishes the five wild-type environments from one another: in a chemostat
    it is the controlled variable, and at steady state it equals the specific growth rate
    and sets the residual concentration of the growth-limiting substrate. The 24
    disruptant cultures and every RF control share 0.2 h-1; the four dilution-rate
    records carry 0.1, 0.4, 0.5 and 0.7 h-1 and are four different environments.

    There is NO ``ProvenanceGap`` on ``media``, and that is the mixin's rule rather than
    an omission: a field cannot both hold a value and declare itself missing. The
    deferral lives where it belongs, on :data:`MEDIUM`'s single
    ``composition_deferred`` component and its ``defers_to``, which is how
    ``SM_DEFERRED`` carries the same situation. ``is_synthetic`` being a reading is
    recorded in ``build_accounting.json`` under ``unpinned_environment_values``.
    """
    return Environment(
        media=MEDIUM,
        temperature=None,
        aerobicity=AEROBICITY,
        dilution_rate_per_hour=dilution_rate_per_hour,
        provenance_gaps=[TEMPERATURE_GAP],
    )


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never a live URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/ishiiMultipleHighThroughputAnalyses2007``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/ishiiMultipleHighThroughputAnalyses2007``."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve_raw_files(dest_dir: str | Path) -> dict[str, Path]:
    """Run each file's recorded retriever and write the verified bytes to ``dest_dir``.

    The recorded ``RetrievalRecord`` is what runs, so this IS the re-runnable retrieval;
    a byte mismatch raises before anything is written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for raw in RAW_FILES:
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
    missing = sorted(set(DATA_SHA256) - set(sources))
    if missing:
        raise KeyError(f"no source given for {missing}")
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for raw in RAW_FILES:
        src = Path(sources[raw.name])
        got = _sha256(src)
        if got != raw.sha256:
            raise RuntimeError(
                f"{src} sha256 mismatch: got {got}, expected {raw.sha256}"
            )
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
                role=ROLE_RAW_DATA,
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
        title=TITLE,
        files=records,
        si_data_sources=[PROJECT_SITE, PUBLISHER_SUPPLEMENT_URL],
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


def sourced_value_root(value: SourcedValue, data_root: str | None = None) -> Path:
    """The mirror root a ``SourcedValue``'s ``source_uri`` is relative to.

    The workbook quotes sit in the raw mirror (``data/...``); the paper quotes are from
    the literature mirror.
    """
    base = Path(data_root or _data_root())
    if value.provenance.source_uri.startswith("data/"):
        return base / "torchcell-raw"
    return base / "torchcell-library"


# --------------------------------------------------------------------------- #
# Reading the workbook
# --------------------------------------------------------------------------- #
SampleKind = Literal["disruptant", "dilution_rate", "reference"]


class SampleColumn(BaseModel):
    """One sample column of one sheet: who it is, which series, which column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(description="the sheet's own Sample ID, e.g. 'KO03'.")
    label: str = Field(
        description="the Sample ID, with the column letter appended when the sheet "
        "repeats that id. The Protein sheet carries RF03 in five separate series "
        "columns, so the drop ledger needs a per-COLUMN name to account for each one "
        "exactly once."
    )
    name: str = Field(description="the sheet's own sample name, e.g. 'pgm'.")
    kind: SampleKind
    column: int = Field(description="0-based column index of the value column.")
    series: str | None = Field(
        default=None,
        description="the sheet's Series ID cell verbatim ('1', '1*', ...); None for a "
        "sheet that carries no series row.",
    )


class DropReason(BaseModel):
    """One rule under which sample columns are not loaded, with its items."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    description: str
    sample_ids: tuple[str, ...] = Field(
        description="The SampleColumn labels this rule accounts for."
    )
    needed_addition: str | None = None


class DropLedger(BaseModel):
    """Every sample column of one sheet, accounted for exactly once."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    sheet: str
    sample_columns: int
    kept_records: int
    rules: tuple[DropReason, ...]

    def check(self) -> None:
        """Every column is either kept or named by exactly one rule."""
        named = [sid for rule in self.rules for sid in rule.sample_ids]
        if len(named) != len(set(named)):
            raise RuntimeError(f"{self.sheet}: a sample is named by two drop rules")
        if self.kept_records + len(named) != self.sample_columns:
            raise RuntimeError(
                f"{self.sheet}: {self.kept_records} kept + {len(named)} dropped != "
                f"{self.sample_columns} sample columns"
            )


def _text(sheet: Any, row: int, col: int) -> str:
    """One cell as stripped text (''), whatever its stored type."""
    if row >= sheet.nrows or col >= sheet.ncols:
        return ""
    value = sheet.cell_value(row, col)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _number(sheet: Any, row: int, col: int) -> float | None:
    """One cell as a float, or None when it is blank or a non-numeric marker."""
    if row >= sheet.nrows or col >= sheet.ncols:
        return None
    value = sheet.cell_value(row, col)
    if isinstance(value, float):
        return float(value)
    return None


def disruptant_symbol(name: str) -> str:
    """The gene symbol a sample name denotes, with the culture-replicate suffix off.

    ``pfkA_1`` and ``pfkA_2`` are the two cultures of ONE genotype, so the suffix must
    come off before the duplicate-culture rule can see them as the same strain. Any
    other suffixed name stops the build, because it would mean a second genotype was
    grown twice and the drop arithmetic no longer holds.
    """
    match = CULTURE_REPLICATE.match(name)
    if match is None:
        return name
    if name not in SUFFIXED_SAMPLE_NAMES:
        raise RuntimeError(
            f"sample name {name!r} carries a culture-replicate suffix; only "
            f"{sorted(SUFFIXED_SAMPLE_NAMES)} are known to"
        )
    return match.group("symbol")


def _sample_kind(sample_id: str) -> SampleKind:
    """Which arm a Sample ID belongs to, from the release's own KO/GR/RF prefixes."""
    if sample_id.startswith("KO"):
        return "disruptant"
    if sample_id.startswith("GR"):
        return "dilution_rate"
    if sample_id.startswith("RF"):
        return "reference"
    raise ValueError(f"unknown Ishii 2007 sample id {sample_id!r}")


def read_sample_columns(
    sheet: Any, *, paired: bool, series_row: int | None, name_row: int
) -> list[SampleColumn]:
    """Every sample column of one sheet, in sheet order.

    ``paired`` is True for the mRNA and Protein sheets, where each sample owns a
    ``Conc`` column and a ``CV`` column; the returned column index is the ``Conc`` one.
    A sample the layer did not measure carries a BLANK header pair instead (``GR04x`` in
    the Protein sheet), which is allowed here and caught as ``no_data_in_this_layer``.
    """
    found: list[tuple[str, int]] = []
    for col in range(1, sheet.ncols):
        sample_id = _text(sheet, 0, col)
        if not sample_id or sample_id in {
            "Sample ID",
            "RFs",
            "Ave",
            "SD",
            "CV",
            "Conc",
        }:
            continue
        found.append((sample_id, col))
    repeated = {
        sample_id
        for sample_id in {sid for sid, _ in found}
        if sum(1 for sid, _ in found if sid == sample_id) > 1
    }
    columns: list[SampleColumn] = [
        SampleColumn(
            sample_id=sample_id,
            label=(
                f"{sample_id}@{xlrd.colname(col)}"
                if sample_id in repeated
                else sample_id
            ),
            name=_text(sheet, name_row, col),
            kind=_sample_kind(sample_id),
            column=col,
            series=_text(sheet, series_row, col) or None
            if series_row is not None
            else None,
        )
        for sample_id, col in found
    ]
    if paired:
        for column in columns:
            header = _text(sheet, name_row + 1, column.column)
            statistic = _text(sheet, name_row + 1, column.column + 1)
            if (header, statistic) in {("Conc", "CV"), ("", "")}:
                continue
            raise RuntimeError(
                f"column {column.column} of {sheet.name} is {header!r}/{statistic!r}, "
                "expected 'Conc'/'CV' for a measured sample or a blank pair for a "
                "sample this layer did not measure"
            )
    return columns


def read_row_labels(sheet: Any, first_row: int) -> list[tuple[int, str]]:
    """The ``(row index, label)`` pairs of a sheet's first column, blanks skipped."""
    return [
        (row, _text(sheet, row, 0))
        for row in range(first_row, sheet.nrows)
        if _text(sheet, row, 0)
    ]


def read_ids_roster(book: Any) -> dict[str, str]:
    """``{Sample ID: Sample Name}`` from the ``IDs`` sheet's roster.

    This is the release's own documented roster of what every sample is, and it is where
    the dilution-rate arm's rates are read from: the ``GR`` numbering states nothing, the
    names do.
    """
    sheet = book.sheet_by_name(SHEET_IDS)
    headers = [
        row
        for row in range(sheet.nrows)
        if _text(sheet, row, IDS_SAMPLE_ID_COLUMN) == IDS_SAMPLE_ID_HEADER
    ]
    if len(headers) != 1:
        raise RuntimeError(
            f"the IDs sheet has {len(headers)} {IDS_SAMPLE_ID_HEADER!r} cells in column "
            f"{IDS_SAMPLE_ID_COLUMN}, expected exactly one; the roster cannot be read"
        )
    roster: dict[str, str] = {}
    for row in range(headers[0] + 1, sheet.nrows):
        sample_id = _text(sheet, row, IDS_SAMPLE_ID_COLUMN)
        if not sample_id:
            continue
        if sample_id in roster:
            raise RuntimeError(f"the IDs sheet lists {sample_id!r} twice")
        roster[sample_id] = _text(sheet, row, IDS_SAMPLE_ID_COLUMN + 1)
    return roster


def dilution_rate_from_name(name: str) -> float:
    """The dilution rate a released sample NAME states, in h^-1."""
    match = DILUTION_RATE_NAME.match(name)
    if match is None:
        raise RuntimeError(
            f"sample name {name!r} states no dilution rate; the release writes a "
            "dilution-rate culture as 'WT, <rate>h-1'"
        )
    return float(match.group("rate"))


def check_dilution_rate_arm(
    book: Any, columns: Sequence[SampleColumn]
) -> dict[str, float]:
    """``{column label: dilution rate}`` for one sheet's dilution-rate columns.

    MEASURED, never assumed from the ``GR`` ordering. The rate comes from the ``IDs``
    sheet's Sample Name, which the release writes for every sample, and the data sheet's
    OWN name row must agree with it character for character wherever it carries a name.
    It is blank for exactly the columns this layer did not measure (``GR04x`` in all
    three served sheets), which is the same blank header pair ``read_sample_columns``
    already allows and which ``no_data_in_this_layer`` accounts for. The set of rates
    found must equal the paper's arm minus the 0.2 h-1 reference rate, so a release that
    renumbered the arm, or whose two label rows disagree, stops the build here.
    """
    roster = read_ids_roster(book)
    rates: dict[str, float] = {}
    for column in columns:
        if column.kind != "dilution_rate":
            continue
        if column.sample_id not in roster:
            raise RuntimeError(
                f"{column.sample_id} is a sample column of {book.sheet_names()} but is "
                "not in the IDs sheet's roster, so its dilution rate has no label to be "
                "read from"
            )
        listed = roster[column.sample_id]
        if column.name and column.name != listed:
            raise RuntimeError(
                f"the IDs sheet calls {column.sample_id} {listed!r} and the sheet's own "
                f"name row calls it {column.name!r}; the dilution rate is read off that "
                "label, so the two must agree"
            )
        rates[column.label] = dilution_rate_from_name(listed)
    found = sorted(set(rates.values()))
    expected = sorted(set(DILUTION_RATE_ARM) - {DILUTION_RATE_PER_HOUR})
    if found != expected:
        raise RuntimeError(
            f"the dilution-rate columns state rates {found}, expected the paper's arm "
            f"minus the {DILUTION_RATE_PER_HOUR} h-1 reference rate ({expected})"
        )
    return rates


def culture_dilution_rate(
    column: SampleColumn, dilution_rates: Mapping[str, float]
) -> float:
    """The dilution rate one sample column was grown at, in h^-1.

    A dilution-rate column's rate comes from its own label through
    :func:`check_dilution_rate_arm`. A disruptant column's is 0.2 h-1 ("The cells were
    grown at a single fixed dilution rate of 0.2 hours-1"), and so is a reference
    column's ("Control (Wild type, cultured at a dilution rate of 0.2h-1)").
    """
    if column.kind == "dilution_rate":
        return dilution_rates[column.label]
    return DILUTION_RATE_PER_HOUR


def culture_identity(
    column: SampleColumn, dilution_rates: Mapping[str, float]
) -> tuple[str, float]:
    """What makes one sample column a distinct culture: its genotype and its rate.

    The genotype side is the disrupted gene symbol, or the empty string for a wild-type
    dilution-rate culture, which has no perturbation. Two columns collide only when both
    halves match, which is the pfkA pair and nothing else in this release.
    """
    symbol = "" if column.kind == "dilution_rate" else disruptant_symbol(column.name)
    return symbol, culture_dilution_rate(column, dilution_rates)


def read_measurements_unit(book: Any, label: str) -> str:
    """The unit the ``Information`` sheet's Measurements table pairs with ``label``."""
    sheet = book.sheet_by_name(SHEET_INFORMATION)
    for row in range(sheet.nrows):
        if _text(sheet, row, 0) == label:
            return _text(sheet, row, 1)
    raise RuntimeError(f"{label!r} is not a row of the Measurements table")


def read_protocol_legend(book: Any) -> dict[int, str]:
    """``{fill colour index: protocol}`` read off the ``Information`` sheet's swatches.

    The legend is three coloured cells beside the three ``Method:`` labels, so the
    colour-to-protocol map comes out of the workbook instead of being hardcoded.
    """
    sheet = book.sheet_by_name(SHEET_INFORMATION)
    legend: dict[int, str] = {}
    for row, protocol in PROTOCOL_LEGEND_ROWS:
        index = book.xf_list[
            sheet.cell_xf_index(row, PROTOCOL_LEGEND_COLUMN)
        ].background.pattern_colour_index
        if index in legend:
            raise RuntimeError(f"colour {index} labels two protocols")
        legend[index] = protocol
    return legend


def read_metabolite_rows(book: Any) -> list[tuple[int, str, str]]:
    """``(row, name, protocol)`` for every metabolite row, protocol from its fill colour."""
    sheet = book.sheet_by_name(SHEET_METABOLITE)
    legend = read_protocol_legend(book)
    rows: list[tuple[int, str, str]] = []
    for row, name in read_row_labels(sheet, 3):
        index = book.xf_list[
            sheet.cell_xf_index(row, 0)
        ].background.pattern_colour_index
        if index not in legend:
            raise RuntimeError(
                f"metabolite row {row} ({name!r}) has fill colour {index}, which the "
                f"Information sheet's legend does not name ({sorted(legend)})"
            )
        rows.append((row, name, legend[index]))
    counts: dict[str, int] = {}
    for _, _, protocol in rows:
        counts[protocol] = counts.get(protocol, 0) + 1
    if counts != EXPECTED_PROTOCOL_ROWS:
        raise RuntimeError(
            f"metabolite protocol row counts {counts}, expected {EXPECTED_PROTOCOL_ROWS}"
        )
    return rows


def metabolite_keys(rows: Sequence[tuple[int, str, str]]) -> dict[int, str]:
    """``{row: stored key}``, disambiguating only a name the release uses twice.

    Exactly one name is repeated across protocols, and appending the protocol to BOTH
    of its rows is what keeps a dict key from silently overwriting the other row.
    """
    seen: dict[str, int] = {}
    for _, name, _protocol in rows:
        seen[name] = seen.get(name, 0) + 1
    duplicated = sorted(name for name, n in seen.items() if n > 1)
    if duplicated != [DUPLICATE_METABOLITE_NAME]:
        raise RuntimeError(
            f"the Metabolite sheet repeats {duplicated}, expected only "
            f"[{DUPLICATE_METABOLITE_NAME!r}]"
        )
    return {
        row: (f"{name} ({protocol})" if seen[name] > 1 else name)
        for row, name, protocol in rows
    }


# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #
class ResolvedGene(BaseModel):
    """One released gene symbol and the BW25113 locus tag it is stored under."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    released_symbol: str
    locus_tag: str
    feature_type: str | None
    status: str


class IdentifierLedger(BaseModel):
    """What the reconciliation did to one panel of released symbols."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reconciliation: LocusTagReconciliation
    resolved: tuple[ResolvedGene, ...]
    unresolved: tuple[str, ...] = Field(
        description="Released symbols NOT stored as a locus tag of the pinned namespace."
    )


def resolve_symbols(
    genome: EcoliK12Genome, symbols: Sequence[str], *, label: str
) -> IdentifierLedger:
    """Map released gene symbols to the pinned BW25113 locus tags, keeping every name."""
    stored, report = reconcile_locus_tags(genome, pd.Series(list(symbols)), label=label)
    resolved: list[ResolvedGene] = []
    unresolved: list[str] = []
    for symbol, tag in zip(symbols, stored, strict=True):
        resolution = genome.resolve_gene_name(symbol)
        if tag == symbol:
            unresolved.append(symbol)
        resolved.append(
            ResolvedGene(
                released_symbol=symbol,
                locus_tag=str(tag),
                feature_type=resolution.feature_type,
                status=resolution.status.value,
            )
        )
    return IdentifierLedger(
        reconciliation=report, resolved=tuple(resolved), unresolved=tuple(unresolved)
    )


def require_full_resolution(ledger: IdentifierLedger, *, label: str) -> None:
    """Refuse a build where any symbol of a REQUIRED panel is off the namespace."""
    fraction = 1.0 - len(ledger.unresolved) / len(ledger.resolved)
    if fraction < MIN_RESOLVED_FRACTION:
        raise RuntimeError(
            f"{label}: {len(ledger.unresolved)} of {len(ledger.resolved)} symbols are "
            f"not BW25113 locus tags ({sorted(ledger.unresolved)}); every stored "
            "identifier must be a locus of the pinned assembly"
        )


# --------------------------------------------------------------------------- #
# Build-time checks on the released bytes
# --------------------------------------------------------------------------- #
def check_disruptant_panel(names: Sequence[str]) -> None:
    """The sheet's disruptant names are exactly the 24 the paper lists."""
    if sorted(names) != sorted(DISRUPTANT_SYMBOLS):
        raise RuntimeError(
            f"the sheet's disruptant names {sorted(names)} are not the paper's 24 "
            f"{sorted(DISRUPTANT_SYMBOLS)}"
        )


def check_units(book: Any) -> None:
    """The Information sheet still pairs each served layer with the unit we store in."""
    for key, label in (
        ("metabolite_unit", "Metabolites (intracellular)"),
        ("protein_unit", "Proteins"),
    ):
        expected = SOURCED_VALUES[key].value
        found = read_measurements_unit(book, label)
        if found != expected:
            raise RuntimeError(
                f"the Measurements table gives {label!r} as {found!r}, expected "
                f"{expected!r}; the stored measurement_type names the unit"
            )


def check_gpmg_unused(book: Any) -> int:
    """``gpmG`` is retired on the pinned annotation; assert it carries no value.

    Returns the number of Protein columns it is measured in, which must be zero. If a
    corrected re-export ever fills it, the build stops here instead of writing a key
    that is not a locus of the pinned assembly.
    """
    sheet = book.sheet_by_name(SHEET_PROTEIN)
    rows = {name: row for row, name in read_row_labels(sheet, 4)}
    if RETIRED_PROTEIN_SYMBOL not in rows:
        raise RuntimeError(
            f"{RETIRED_PROTEIN_SYMBOL!r} is no longer a Protein row; the retired-symbol "
            "check cannot be made"
        )
    row = rows[RETIRED_PROTEIN_SYMBOL]
    filled = sum(
        1
        for column in read_sample_columns(sheet, paired=True, series_row=1, name_row=2)
        if _number(sheet, row, column.column) is not None
    )
    if filled:
        raise RuntimeError(
            f"{RETIRED_PROTEIN_SYMBOL!r} now carries {filled} values but is retired on "
            "the pinned BW25113 annotation, so it has no locus tag to be stored under"
        )
    return filled


def read_specific_rates(book: Any) -> dict[str, dict[str, float]]:
    """``{sample id: {rate label: value}}`` from the ``Specific_Rates`` sheet."""
    sheet = book.sheet_by_name(SHEET_RATES)
    labels = read_row_labels(sheet, 2)
    out: dict[str, dict[str, float]] = {}
    for column in read_sample_columns(sheet, paired=False, series_row=None, name_row=1):
        rates = {
            label: value
            for row, label in labels
            if (value := _number(sheet, row, column.column)) is not None
        }
        if rates:
            out[column.sample_id] = rates
    return out


def check_aerobic(
    rates: Mapping[str, Mapping[str, float]], loaded: Sequence[str]
) -> None:
    """Every loaded culture has a positive oxygen uptake rate.

    This is what makes ``aerobicity="aerobic"`` a measured value on these records rather
    than the field default left in place.
    """
    missing = [sid for sid in loaded if OXYGEN_UPTAKE_ROW not in rates.get(sid, {})]
    if missing:
        raise RuntimeError(
            f"no oxygen uptake rate for {missing}; aerobicity cannot be asserted"
        )
    anaerobic = [sid for sid in loaded if rates[sid][OXYGEN_UPTAKE_ROW] <= 0.0]
    if anaerobic:
        raise RuntimeError(
            f"{anaerobic} have a non-positive oxygen uptake rate, so the stored "
            "aerobicity would be wrong"
        )


# --------------------------------------------------------------------------- #
# Column classification (one reason per dropped column, in a fixed order)
# --------------------------------------------------------------------------- #
DROP_NO_DATA = DropReason(
    rule="no_data_in_this_layer",
    description="the sample column is present in the sheet header and empty in every "
    "row, so this layer measured nothing for that culture",
    sample_ids=(),
)
DROP_REFERENCE = DropReason(
    rule="reference_sample",
    description="an RF column: the Samples table labels this block 'Control (Wild type, "
    "cultured at a dilution rate of 0.2h-1)', so it is the experiment_reference, not a "
    "record",
    sample_ids=(),
)
#: ``culture_not_batch`` WAS the third rule here: it dropped ``GR01``-``GR04``, the wild
#: type at 0.1, 0.4, 0.5 and 0.7 h-1, because ``Environment`` had no dilution-rate slot
#: and the five cultures would have shared one byte-identical environment. #753 added
#: ``Environment.dilution_rate_per_hour``, the four cultures are records, and the rule is
#: retired rather than narrowed because no sample column is left for it to name.
#: :data:`RETIRED_DROP_RULES` carries that into ``build_accounting.json``.
RETIRED_DROP_RULES: tuple[dict[str, str], ...] = (
    {
        "rule": "culture_not_batch",
        "retired_by": "Environment.dilution_rate_per_hour (#753)",
        "was_dropping": "GR01-GR04, the wild type at 0.1, 0.4, 0.5 and 0.7 h-1, in all "
        "three served sheets",
        "why_it_no_longer_applies": "the four cultures differ from the 0.2 h-1 "
        "reference and from one another in dilution rate, which is now a field of the "
        "environment, so they are four distinct environments and four records",
    },
)
DROP_DUPLICATE_CULTURE = DropReason(
    rule="duplicate_culture_same_genotype_and_environment",
    description="the second culture of the one disruptant grown twice ('pfkA "
    "disruptant was cultured twice.'). Both are pfkA at 0.2 h-1 in one medium, so they "
    "would carry the same genotype AND the same environment; the kept one is the "
    "culture measured in all three served layers, which keeps the three datasets "
    "paired on one culture",
    sample_ids=(),
)


def classify_columns(
    sheet: Any,
    columns: Sequence[SampleColumn],
    value_rows: Sequence[int],
    *,
    dataset: str,
    dilution_rates: Mapping[str, float],
) -> tuple[list[SampleColumn], DropLedger]:
    """Split one sheet's sample columns into records and drop rules.

    The order is fixed and each column gets exactly one reason: emptiness first (a fact
    about the bytes), then the reference block, then a duplicate culture of an
    already-kept genotype. A dilution-rate column is a RECORD, because its rate is part
    of its environment; the duplicate rule keys on (genotype, dilution rate) for the same
    reason, so two cultures only collide when both match.
    """
    empty: list[str] = []
    reference: list[str] = []
    duplicate: list[str] = []
    candidates: list[SampleColumn] = []
    for column in columns:
        if all(_number(sheet, row, column.column) is None for row in value_rows):
            empty.append(column.label)
        elif column.kind == "reference":
            reference.append(column.label)
        else:
            candidates.append(column)
    by_culture: dict[tuple[str, float], list[SampleColumn]] = {}
    for column in candidates:
        by_culture.setdefault(culture_identity(column, dilution_rates), []).append(
            column
        )
    kept_ids: set[str] = set()
    for culture, group in by_culture.items():
        if len(group) == 1:
            kept_ids.add(group[0].label)
            continue
        preferred = [c for c in group if c.sample_id in PREFERRED_DUPLICATE_SAMPLES]
        if len(preferred) != 1:
            raise RuntimeError(
                f"{culture} has {len(group)} cultures with data in {sheet.name} "
                f"({[c.label for c in group]}) and {len(preferred)} of them are named "
                "in PREFERRED_DUPLICATE_SAMPLES; exactly one culture of a genotype at "
                "one dilution rate can be the record"
            )
        kept_ids.add(preferred[0].label)
        duplicate.extend(c.label for c in group if c.label != preferred[0].label)
    kept = [column for column in candidates if column.label in kept_ids]
    ledger = DropLedger(
        dataset=dataset,
        sheet=sheet.name,
        sample_columns=len(columns),
        kept_records=len(kept),
        rules=(
            DROP_NO_DATA.model_copy(update={"sample_ids": tuple(empty)}),
            DROP_REFERENCE.model_copy(update={"sample_ids": tuple(reference)}),
            DROP_DUPLICATE_CULTURE.model_copy(update={"sample_ids": tuple(duplicate)}),
        ),
    )
    ledger.check()
    return kept, ledger


def normalize_series(series: str | None) -> str:
    """The series id with the release's unexplained trailing marker stripped.

    Three metabolite columns carry ``1*`` where every other column carries a bare
    number; the workbook prints no footnote for the asterisk, so it is stripped for
    matching and kept verbatim on the ``SampleColumn``.
    """
    if series is None:
        raise RuntimeError("this sheet carries no series row")
    return series.rstrip("*")


def reference_by_series(
    columns: Sequence[SampleColumn], kept: Sequence[SampleColumn]
) -> dict[str, SampleColumn]:
    """``{series: reference column}``, one per series, covering every kept record."""
    out: dict[str, SampleColumn] = {}
    for column in columns:
        if column.kind != "reference":
            continue
        series = normalize_series(column.series)
        if series in out:
            raise RuntimeError(
                f"series {series} has two reference columns ({out[series].column} and "
                f"{column.column}); the per-record reference would be ambiguous"
            )
        out[series] = column
    missing = sorted({normalize_series(c.series) for c in kept} - set(out), key=str)
    if missing:
        raise RuntimeError(f"no reference column for series {missing}")
    return out


# --------------------------------------------------------------------------- #
# Record construction (pure, no files)
# --------------------------------------------------------------------------- #
def deletion_genotype(resolved: ResolvedGene) -> Genotype:
    """The one Keio deletion of a disruptant, named by its BW25113 locus tag."""
    return Genotype(
        perturbations=[
            BacterialDeletionPerturbation(
                systematic_gene_name=resolved.locus_tag,
                perturbed_gene_name=resolved.released_symbol,
                gene_namespace=STRAIN_GENE_NAMESPACES["BW25113"],
                collection=COLLECTION,
                cassette=CASSETTE,
                construction=StrainConstruction(lab="Keio University"),
            )
        ]
    )


def wild_type_genotype() -> Genotype:
    """The wild type: no perturbation, so the dilution rate is the whole difference.

    The release's ``GR`` cultures are the unmodified K-12 strain at another dilution
    rate, so there is nothing to put in ``perturbations`` and nothing is invented. Two
    such records stay distinct because their environments do.
    """
    return Genotype(perturbations=[])


def metabolite_phenotype(levels: Mapping[str, float]) -> MetabolitePhenotype:
    """One culture's CE-TOFMS concentrations; one measurement, so one replicate each."""
    return MetabolitePhenotype(
        metabolite_level=dict(levels),
        metabolite_level_se=None,
        n_replicates=dict.fromkeys(levels, 1),
        measurement_type=MEASUREMENT_TYPE_METABOLITE,
        target_metabolite_ids=None,
        provenance_gaps=[METABOLITE_SE_GAP],
    )


def protein_phenotype(
    levels: Mapping[str, float], standard_errors: Mapping[str, float]
) -> ProteinAbundancePhenotype:
    """One culture's LC-MS/MS absolute abundances with the SE derived from the CV.

    ``standard_errors`` carries ``nan`` for a protein the sheet gives a level but no CV
    for, which is the release's own per-cell reporting gap and is not read as zero
    spread.
    """
    return ProteinAbundancePhenotype(
        protein_abundance=dict(levels),
        protein_abundance_se=dict(standard_errors),
        n_replicates=dict.fromkeys(levels, SOURCED_VALUES["protein_replicates"].value),
        measurement_type=MEASUREMENT_TYPE_PROTEIN,
    )


def flux_phenotype(net_flux: Mapping[str, float]) -> FluxPhenotype:
    """One culture's fitted 13C net-flux map, with no interval because none is released."""
    return FluxPhenotype(
        net_flux=dict(net_flux),
        net_flux_lower=None,
        net_flux_upper=None,
        confidence_level=None,
        measurement_type=MEASUREMENT_TYPE_FLUX,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        target_reaction_ids=None,
        provenance_gaps=[FLUX_INTERVAL_GAP],
    )


def standard_error_from_cv(level: float, cv_percent: float | None, n: int) -> float:
    """``level * cv / 100 / sqrt(n)``, or ``nan`` when the sheet reports no CV."""
    if cv_percent is None:
        return math.nan
    return abs(level) * cv_percent / 100.0 / math.sqrt(n)


# --------------------------------------------------------------------------- #
# The datasets
# --------------------------------------------------------------------------- #
class _Ishii2007Dataset(ExperimentDataset):
    """Shared raw-file handling for the three Ishii 2007 arms.

    All three read the SAME pinned workbook, so the mirror check, the genome pin and the
    ledger writing live here once; each subclass owns only its sheet and its phenotype.
    """

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"
    #: The dev-tree slug, which is what the record counts and the roots are keyed by;
    #: ``self.name`` is the CLASS name and is what lands on ``dataset_name``.
    SLUG: ClassVar[str] = ""

    def __init__(
        self,
        root: str,
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
    def raw_file_names(self) -> list[str]:
        """Every consumed file, linked from the raw mirror."""
        return [raw.name for raw in RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking the manifest and sha256.

        The mirror plus ``DATA_SHA256`` is canonical; the project-web-site URLs are
        retrieval metadata that ``retrieve_raw_files`` re-runs, never a build input.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for raw in RAW_FILES:
            check_manifest_pin(
                raw.mirror_relpath,
                manifest_sha256(manifest, raw.mirror_relpath),
                raw.sha256,
            )
            src = raw_mirror_dir(data_root) / raw.mirror_relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, raw.name), raw.sha256)
        log.info("Ishii 2007 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _raw(self, name: str) -> str:
        return osp.join(self.raw_dir, name)

    def _workbook(self) -> Any:
        """The pinned quantitative workbook, with cell formatting (the protocol legend)."""
        return xlrd.open_workbook(self._raw(QUANTITATIVE_FILE), formatting_info=True)

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

    def _disruptants(self, kept: Sequence[SampleColumn]) -> dict[str, ResolvedGene]:
        """``{released symbol: ResolvedGene}`` for the kept DISRUPTANT records.

        The dilution-rate records have no perturbation, so they are not in the panel and
        contribute no symbol; the panel check still sees exactly the paper's 24.
        """
        symbols = [
            disruptant_symbol(column.name)
            for column in kept
            if column.kind == "disruptant"
        ]
        check_disruptant_panel(symbols)
        ledger = resolve_symbols(
            self._genome(), symbols, label=f"{self.name} disruptant symbols"
        )
        require_full_resolution(ledger, label=f"{self.name} disruptant symbols")
        self._identifier_ledger = ledger
        return {gene.released_symbol: gene for gene in ledger.resolved}

    @staticmethod
    def _genotype_of(
        column: SampleColumn, genes: Mapping[str, ResolvedGene]
    ) -> Genotype:
        """The genotype of one kept culture: one Keio deletion, or the wild type."""
        if column.kind == "dilution_rate":
            return wild_type_genotype()
        return deletion_genotype(genes[disruptant_symbol(column.name)])

    def _write_common_ledgers(
        self,
        drops: DropLedger,
        rates: Mapping[str, Mapping[str, float]],
        dilution_rates: Mapping[str, float],
    ) -> None:
        """The drop ledger, the identifier ledger, the rates and the build accounting."""
        out = Path(self.preprocess_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dropped_records.json").write_text(drops.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            self._identifier_ledger.model_dump_json(indent=2)
        )
        pd.DataFrame(
            [
                {"sample_id": sid, **dict(sorted(values.items()))}
                for sid, values in sorted(rates.items())
            ]
        ).to_csv(out / "specific_rates.csv", index=False)
        (out / "build_accounting.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "reference_dilution_rate_per_hour": DILUTION_RATE_PER_HOUR,
                    "dilution_rates_loaded": sorted(
                        {DILUTION_RATE_PER_HOUR, *dilution_rates.values()}
                    ),
                    "dilution_rate_by_sample_column": dict(
                        sorted(dilution_rates.items())
                    ),
                    "retired_drop_rules": [dict(rule) for rule in RETIRED_DROP_RULES],
                    "sourced_values": {
                        key: value.model_dump(mode="json")
                        for key, value in SOURCED_VALUES.items()
                    },
                    "not_mirrored": list(NOT_MIRRORED),
                    "unpinned_environment_values": [
                        {
                            "field": "media",
                            "reading": "a synthetic (defined) glucose-limited medium "
                            "with its recipe deferred to the unmirrored supplement",
                            "why": str(SOURCED_VALUES["glucose_limited"].note),
                        },
                        {
                            "field": "temperature",
                            "reading": "none; left None",
                            "why": str(TEMPERATURE_GAP.note),
                        },
                    ],
                },
                indent=2,
            )
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for these datasets."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled by the per-arm builders."""
        raise NotImplementedError


@register_dataset
class MetabolomeIshii2007Dataset(_Ishii2007Dataset):
    """CE-TOFMS intracellular metabolite concentrations of 28 chemostat cultures.

    24 Keio disruptants at 0.2 h-1 plus the wild type at 0.1, 0.4, 0.5 and 0.7 h-1.
    """

    SLUG: ClassVar[str] = "metabolome_ishii2007"

    def __init__(
        self, root: str = "data/torchcell/metabolome_ishii2007", **kwargs: Any
    ) -> None:
        """Initialize at this arm's default dev-tree root."""
        super().__init__(root, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialMetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialMetaboliteExperimentReference

    @post_process
    def process(self) -> None:
        """Read the Metabolite sheet into one record per loaded disruptant culture."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        book = self._workbook()
        check_units(book)
        sheet = book.sheet_by_name(SHEET_METABOLITE)
        rows = read_metabolite_rows(book)
        keys = metabolite_keys(rows)
        columns = read_sample_columns(sheet, paired=False, series_row=1, name_row=2)
        dilution_rates = check_dilution_rate_arm(book, columns)
        kept, drops = classify_columns(
            sheet,
            columns,
            [row for row, _, _ in rows],
            dataset=self.name,
            dilution_rates=dilution_rates,
        )
        references = reference_by_series(columns, kept)
        rates = read_specific_rates(book)
        check_aerobic(rates, [column.sample_id for column in kept])
        genes = self._disruptants(kept)

        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_common_ledgers(drops, rates, dilution_rates)
        self._write_metabolite_ledger(
            sheet, rows, keys, kept, references, dilution_rates
        )

        reference_env = environment(DILUTION_RATE_PER_HOUR)
        genome_reference = assembly_reference(self.REFERENCE_STRAIN)
        reference_only = 0
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, column in enumerate(tqdm(kept, desc=self.name)):
                levels = self._levels(sheet, rows, keys, column.column)
                experiment = BacterialMetaboliteExperiment(
                    dataset_name=self.name,
                    genotype=self._genotype_of(column, genes),
                    environment=environment(
                        culture_dilution_rate(column, dilution_rates)
                    ),
                    phenotype=metabolite_phenotype(levels),
                )
                reference_column = references[normalize_series(column.series)]
                reference_levels = {
                    key: value
                    for key, value in self._levels(
                        sheet, rows, keys, reference_column.column
                    ).items()
                    if key in levels
                }
                if not reference_levels:
                    raise RuntimeError(
                        f"{column.label}: its series reference "
                        f"{reference_column.sample_id} shares no metabolite with it"
                    )
                reference = BacterialMetaboliteExperimentReference(
                    dataset_name=self.name,
                    genome_reference=genome_reference,
                    environment_reference=reference_env.model_copy(),
                    phenotype_reference=metabolite_phenotype(reference_levels),
                )
                reference_only += len(
                    self._levels(sheet, rows, keys, reference_column.column)
                ) - len(reference_levels)
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        env_out.close()
        interned_env.close()
        if len(kept) != EXPECTED_RECORDS[self.SLUG]:
            raise RuntimeError(
                f"{self.name}: wrote {len(kept)} records, the pinned workbook gives "
                f"{EXPECTED_RECORDS[self.SLUG]}"
            )
        log.info(
            "Wrote %d Ishii 2007 metabolome experiments to LMDB; %d reference "
            "baselines dropped as metabolites their record did not detect",
            len(kept),
            reference_only,
        )

    @staticmethod
    def _levels(
        sheet: Any,
        rows: Sequence[tuple[int, str, str]],
        keys: Mapping[int, str],
        column: int,
    ) -> dict[str, float]:
        """One column's concentrations; a blank is key absence, never a zero."""
        levels = {
            keys[row]: value
            for row, _, _ in rows
            if (value := _number(sheet, row, column)) is not None
        }
        if not levels:
            raise RuntimeError(f"column {column} of the Metabolite sheet is empty")
        return levels

    def _write_metabolite_ledger(
        self,
        sheet: Any,
        rows: Sequence[tuple[int, str, str]],
        keys: Mapping[int, str],
        kept: Sequence[SampleColumn],
        references: Mapping[str, SampleColumn],
        dilution_rates: Mapping[str, float],
    ) -> None:
        """Every metabolite row with its protocol and the released reference dispersion."""
        average, sd, cv = (
            [
                column
                for column in range(sheet.ncols)
                if _text(sheet, 2, column) == label
            ]
            for label in ("Ave", "SD", "CV")
        )
        if not (len(average) == len(sd) == len(cv) == 1):
            raise RuntimeError(
                "the Metabolite sheet does not carry exactly one Ave/SD/CV reference "
                "column trio, which the Information sheet names as 'Column AN, AO, AP'"
            )
        pd.DataFrame(
            [
                {
                    "key": keys[row],
                    "released_name": name,
                    "protocol": protocol,
                    "sheet_row": row + 1,
                    "reference_series_mean": _number(sheet, row, average[0]),
                    "reference_series_sd": _number(sheet, row, sd[0]),
                    "reference_series_cv_percent": _number(sheet, row, cv[0]),
                }
                for row, name, protocol in rows
            ]
        ).to_csv(Path(self.preprocess_dir) / "metabolites.csv", index=False)
        pd.DataFrame(
            [
                {
                    "sample_id": column.sample_id,
                    "perturbed_gene_symbol": culture_identity(column, dilution_rates)[
                        0
                    ],
                    "culture_name_verbatim": column.name,
                    "dilution_rate_per_hour": culture_dilution_rate(
                        column, dilution_rates
                    ),
                    "series_verbatim": column.series,
                    "reference_sample_id": references[
                        normalize_series(column.series)
                    ].sample_id,
                }
                for column in kept
            ]
        ).to_csv(Path(self.preprocess_dir) / "records.csv", index=False)


@register_dataset
class ProteomeIshii2007Dataset(_Ishii2007Dataset):
    """LC-MS/MS absolute protein abundances of 28 chemostat cultures.

    24 Keio disruptants at 0.2 h-1 plus the wild type at 0.1, 0.4, 0.5 and 0.7 h-1.
    """

    SLUG: ClassVar[str] = "proteome_ishii2007"

    def __init__(
        self, root: str = "data/torchcell/proteome_ishii2007", **kwargs: Any
    ) -> None:
        """Initialize at this arm's default dev-tree root."""
        super().__init__(root, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialProteinAbundanceExperimentReference

    @post_process
    def process(self) -> None:
        """Read the Protein sheet into one record per loaded disruptant culture."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        book = self._workbook()
        check_units(book)
        check_gpmg_unused(book)
        sheet = book.sheet_by_name(SHEET_PROTEIN)
        labels = read_row_labels(sheet, 4)
        columns = read_sample_columns(sheet, paired=True, series_row=1, name_row=2)
        dilution_rates = check_dilution_rate_arm(book, columns)
        kept, drops = classify_columns(
            sheet,
            columns,
            [row for row, _ in labels],
            dataset=self.name,
            dilution_rates=dilution_rates,
        )
        references = reference_by_series(columns, kept)
        rates = read_specific_rates(book)
        check_aerobic(rates, [column.sample_id for column in kept])
        genes = self._disruptants(kept)
        proteins = resolve_symbols(
            self._genome(),
            [name for _, name in labels],
            label=f"{self.name} quantified protein symbols",
        )
        keys = {
            row: gene.locus_tag
            for (row, _), gene in zip(labels, proteins.resolved, strict=True)
        }
        unresolved = set(proteins.unresolved)
        if unresolved != {RETIRED_PROTEIN_SYMBOL}:
            raise RuntimeError(
                f"protein symbols off the BW25113 namespace are {sorted(unresolved)}, "
                f"expected only {{{RETIRED_PROTEIN_SYMBOL!r}}}, which carries no value"
            )

        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_common_ledgers(drops, rates, dilution_rates)
        self._write_protein_ledger(labels, proteins, kept, references, dilution_rates)

        reference_env = environment(DILUTION_RATE_PER_HOUR)
        genome_reference = assembly_reference(self.REFERENCE_STRAIN)
        unbaselined: dict[str, list[str]] = {}
        reference_only = 0
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, column in enumerate(tqdm(kept, desc=self.name)):
                levels, errors = self._levels(sheet, labels, keys, column.column)
                reference_column = references[normalize_series(column.series)]
                ref_levels, ref_errors = self._levels(
                    sheet, labels, keys, reference_column.column
                )
                shared = set(levels) & set(ref_levels)
                dropped = sorted(set(levels) - shared)
                if dropped:
                    unbaselined[column.label] = dropped
                reference_only += len(set(ref_levels) - shared)
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=self._genotype_of(column, genes),
                    environment=environment(
                        culture_dilution_rate(column, dilution_rates)
                    ),
                    phenotype=protein_phenotype(
                        {k: levels[k] for k in shared}, {k: errors[k] for k in shared}
                    ),
                )
                reference = BacterialProteinAbundanceExperimentReference(
                    dataset_name=self.name,
                    genome_reference=genome_reference,
                    environment_reference=reference_env.model_copy(),
                    phenotype_reference=protein_phenotype(
                        {k: ref_levels[k] for k in shared},
                        {k: ref_errors[k] for k in shared},
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        env_out.close()
        interned_env.close()
        n_unbaselined = sum(len(keys_) for keys_ in unbaselined.values())
        if n_unbaselined != EXPECTED_PROTEIN_KEYS_WITHOUT_REFERENCE:
            raise RuntimeError(
                f"{n_unbaselined} measured abundances have no value in their own series "
                f"reference, the pinned workbook gives "
                f"{EXPECTED_PROTEIN_KEYS_WITHOUT_REFERENCE}: {unbaselined}"
            )
        (Path(self.preprocess_dir) / "abundances_without_reference.json").write_text(
            json.dumps(
                {
                    "record_keys_dropped": unbaselined,
                    "reference_baselines_dropped": reference_only,
                    "why": "verify_protein_dataset's L3 requires a record's protein "
                    "keys to EQUAL its reference's, so each record stores the proteins "
                    "its OWN series reference also detected. A blank cell is 'not "
                    "detected', so neither side's drop is a zero",
                },
                indent=2,
            )
        )
        if len(kept) != EXPECTED_RECORDS[self.SLUG]:
            raise RuntimeError(
                f"{self.name}: wrote {len(kept)} records, the pinned workbook gives "
                f"{EXPECTED_RECORDS[self.SLUG]}"
            )
        log.info("Wrote %d Ishii 2007 proteome experiments to LMDB", len(kept))

    @staticmethod
    def _levels(
        sheet: Any,
        labels: Sequence[tuple[int, str]],
        keys: Mapping[int, str],
        column: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """One column's abundances and the SEs derived from its CV column."""
        n: int = SOURCED_VALUES["protein_replicates"].value
        levels: dict[str, float] = {}
        errors: dict[str, float] = {}
        for row, _ in labels:
            level = _number(sheet, row, column)
            if level is None:
                continue
            key = keys[row]
            levels[key] = level
            errors[key] = standard_error_from_cv(
                level, _number(sheet, row, column + 1), n
            )
        if not levels:
            raise RuntimeError(f"column {column} of the Protein sheet is empty")
        return levels, errors

    def _write_protein_ledger(
        self,
        labels: Sequence[tuple[int, str]],
        proteins: IdentifierLedger,
        kept: Sequence[SampleColumn],
        references: Mapping[str, SampleColumn],
        dilution_rates: Mapping[str, float],
    ) -> None:
        """The protein panel with its locus tags, and the released peptide counts."""
        peptides: dict[str, int] = {}
        counts: dict[str, list[str]] = SOURCED_VALUES["protein_peptides"].value
        for symbol in counts["one"]:
            peptides[symbol.lower()] = 1
        for symbol in counts["two"]:
            peptides[symbol.lower()] = 2
        pd.DataFrame(
            [
                {
                    "sheet_row": row + 1,
                    "released_symbol": name,
                    "locus_tag": gene.locus_tag,
                    "resolution_status": gene.status,
                    "peptides_measured": peptides.get(name.lower(), 3),
                    "stored": name not in proteins.unresolved,
                }
                for (row, name), gene in zip(labels, proteins.resolved, strict=True)
            ]
        ).to_csv(Path(self.preprocess_dir) / "protein_peptides.csv", index=False)
        pd.DataFrame(
            [
                {
                    "sample_id": column.sample_id,
                    "perturbed_gene_symbol": culture_identity(column, dilution_rates)[
                        0
                    ],
                    "culture_name_verbatim": column.name,
                    "dilution_rate_per_hour": culture_dilution_rate(
                        column, dilution_rates
                    ),
                    "series_verbatim": column.series,
                    "reference_sample_id": references[
                        normalize_series(column.series)
                    ].sample_id,
                }
                for column in kept
            ]
        ).to_csv(Path(self.preprocess_dir) / "records.csv", index=False)


@register_dataset
class FluxIshii2007Dataset(_Ishii2007Dataset):
    """Fitted 13C net-flux maps of 28 chemostat cultures.

    24 Keio disruptants at 0.2 h-1 plus the wild type at 0.1, 0.4, 0.5 and 0.7 h-1.

    The reaction network is the release's own 43 net reactions, keyed by the reaction
    STRING the sheet prints ("G6P <-> F6P"), because the release names no model and no
    reaction ids; ``target_reaction_ids`` is therefore ``None`` and the constraint-based
    mapping is deferred rather than invented.
    """

    SLUG: ClassVar[str] = "flux_ishii2007"

    def __init__(
        self, root: str = "data/torchcell/flux_ishii2007", **kwargs: Any
    ) -> None:
        """Initialize at this arm's default dev-tree root."""
        super().__init__(root, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return FluxExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return FluxExperimentReference

    @post_process
    def process(self) -> None:
        """Read the Flux sheet into one fitted flux map per loaded disruptant culture."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        book = self._workbook()
        check_units(book)
        sheet = book.sheet_by_name(SHEET_FLUX)
        net_rows, exchange_rows = self._split_rows(sheet)
        columns = read_sample_columns(sheet, paired=False, series_row=None, name_row=1)
        dilution_rates = check_dilution_rate_arm(book, columns)
        kept, drops = classify_columns(
            sheet,
            columns,
            [row for row, _ in net_rows],
            dataset=self.name,
            dilution_rates=dilution_rates,
        )
        rates = read_specific_rates(book)
        check_aerobic(rates, [column.sample_id for column in kept])
        genes = self._disruptants(kept)
        reference_columns = [c for c in columns if c.kind == "reference"]
        stored_reference = self._stored_reference(reference_columns)
        self._check_fit_inputs(kept)

        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_common_ledgers(drops, rates, dilution_rates)
        self._write_flux_ledger(
            sheet, net_rows, exchange_rows, reference_columns, kept, dilution_rates
        )

        genome_reference = assembly_reference(self.REFERENCE_STRAIN)
        reference = FluxExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment(DILUTION_RATE_PER_HOUR),
            phenotype_reference=flux_phenotype(
                self._net_flux(sheet, net_rows, stored_reference.column)
            ),
        )
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, column in enumerate(tqdm(kept, desc=self.name)):
                experiment = FluxExperiment(
                    dataset_name=self.name,
                    genotype=self._genotype_of(column, genes),
                    environment=environment(
                        culture_dilution_rate(column, dilution_rates)
                    ),
                    phenotype=flux_phenotype(
                        self._net_flux(sheet, net_rows, column.column)
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        env_out.close()
        interned_env.close()
        if len(kept) != EXPECTED_RECORDS[self.SLUG]:
            raise RuntimeError(
                f"{self.name}: wrote {len(kept)} records, the pinned workbook gives "
                f"{EXPECTED_RECORDS[self.SLUG]}"
            )
        log.info("Wrote %d Ishii 2007 flux experiments to LMDB", len(kept))

    @staticmethod
    def _split_rows(sheet: Any) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        """The sheet's net-flux rows and its exchange-coefficient rows.

        ``Exch.`` rows are reversibility parameters of seven bidirectional reactions,
        not fluxes, so they never reach ``net_flux``.
        """
        net: list[tuple[int, str]] = []
        exchange: list[tuple[int, str]] = []
        for row, name in read_row_labels(sheet, 2):
            (exchange if name.startswith("Exch.") else net).append((row, name))
        if not net or not exchange:
            raise RuntimeError(
                f"the Flux sheet split into {len(net)} net and {len(exchange)} exchange "
                "rows; the Information sheet states both kinds are present"
            )
        return net, exchange

    @staticmethod
    def _stored_reference(reference_columns: Sequence[SampleColumn]) -> SampleColumn:
        """The one reference fit this arm stores, a DOCUMENTED REPRESENTATIVE choice.

        The Flux sheet carries no series row, so a record cannot be matched to the
        reference of its own series the way the other two arms are. Averaging the four
        released reference fits would be wrong: an average of four separate network fits
        is not a fit. ``RF03`` is stored because it is the ONE reference culture measured
        in all three served layers, which makes the three datasets' references the same
        culture; the other three are written to ``flux_reference_fits.csv`` and the
        choice is flagged for review in the PR.
        """
        by_id = {column.sample_id: column for column in reference_columns}
        if FLUX_REFERENCE_SAMPLE not in by_id:
            raise RuntimeError(
                f"{FLUX_REFERENCE_SAMPLE} is not a reference column of the Flux sheet "
                f"({sorted(by_id)}); the stored reference fit is gone"
            )
        return by_id[FLUX_REFERENCE_SAMPLE]

    @staticmethod
    def _fit_input_sheet(column: SampleColumn) -> str:
        """The GC-MS workbook's sheet name for one culture, as the release names it.

        MEASURED on the pinned workbook: the labeling sheets of the disruptant cultures
        are named by CULTURE NAME (``galM`` ... ``talB``, ``pfkA_1``, ``pfkA_2``) and the
        dilution-rate and reference sheets by SAMPLE ID (``GR01``-``GR04``,
        ``RF03``-``RF06``). A dilution-rate culture's name (``WT, 0.1h-1``) is not a
        sheet name in that workbook, so the id is what the lookup uses.
        """
        if column.kind == "dilution_rate":
            return column.sample_id
        return column.name

    def _check_fit_inputs(self, kept: Sequence[SampleColumn]) -> None:
        """Every stored fit's GC-MS labeling data is in the mirrored input workbook.

        A fitted flux is only honest to store with its inputs recorded, so the build
        refuses a record whose mass-isotopomer sheet is missing.
        """
        sheets = set(xlrd.open_workbook(self._raw(GC_MS_FILE)).sheet_names())
        missing = sorted(
            self._fit_input_sheet(column)
            for column in kept
            if self._fit_input_sheet(column) not in sheets
        )
        if missing:
            raise RuntimeError(
                f"{GC_MS_FILE} has no mass-distribution sheet for {missing}, so those "
                "fitted fluxes would be stored without the data they were fitted to"
            )

    @staticmethod
    def _net_flux(
        sheet: Any, net_rows: Sequence[tuple[int, str]], column: int
    ) -> dict[str, float]:
        """One column's fitted net fluxes; a '-' is key absence, not a zero flux."""
        fluxes = {
            name: value
            for row, name in net_rows
            if (value := _number(sheet, row, column)) is not None
        }
        if not fluxes:
            raise RuntimeError(f"column {column} of the Flux sheet is empty")
        return fluxes

    def _write_flux_ledger(
        self,
        sheet: Any,
        net_rows: Sequence[tuple[int, str]],
        exchange_rows: Sequence[tuple[int, str]],
        reference_columns: Sequence[SampleColumn],
        kept: Sequence[SampleColumn],
        dilution_rates: Mapping[str, float],
    ) -> None:
        """The reaction list, the exchange coefficients and all four reference fits."""
        out = Path(self.preprocess_dir)
        pd.DataFrame(
            [
                {"sheet_row": row + 1, "reaction": name, "kind": kind}
                for rows, kind in ((net_rows, "net_flux"), (exchange_rows, "exchange"))
                for row, name in rows
            ]
        ).to_csv(out / "reactions.csv", index=False)
        pd.DataFrame(
            [
                {"reaction": name}
                | {
                    column.sample_id: _number(sheet, row, column.column)
                    for column in list(kept) + list(reference_columns)
                }
                for row, name in exchange_rows
            ]
        ).to_csv(out / "exchange_coefficients.csv", index=False)
        pd.DataFrame(
            [
                {"reaction": name}
                | {
                    column.sample_id: _number(sheet, row, column.column)
                    for column in reference_columns
                }
                for row, name in net_rows
            ]
        ).to_csv(out / "flux_reference_fits.csv", index=False)
        pd.DataFrame(
            [
                {
                    "sample_id": column.sample_id,
                    "perturbed_gene_symbol": culture_identity(column, dilution_rates)[
                        0
                    ],
                    "culture_name_verbatim": column.name,
                    "dilution_rate_per_hour": culture_dilution_rate(
                        column, dilution_rates
                    ),
                    "reference_sample_id": FLUX_REFERENCE_SAMPLE,
                }
                for column in kept
            ]
        ).to_csv(out / "records.csv", index=False)


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built LMDB
# --------------------------------------------------------------------------- #
METABOLITE_PROVENANCE = Provenance(
    source_uri=f"{PROJECT_SITE}{QUANTITATIVE_FILE}",
    citation_key=CITATION_KEY,
    method="CE-TOFMS intracellular concentration in mM per chemostat culture (24 Keio "
    "disruptants at 0.2 h-1 and the wild type at 0.1, 0.4, 0.5 and 0.7 h-1); the "
    "reference is the wild-type 0.2 h-1 control column of the record's OWN series, "
    "which for a dilution-rate record is a cross-environment reference",
    page="Science 316:593; project web site v1.0.0, sheet 'Metabolite'",
)
PROTEIN_PROVENANCE = Provenance(
    source_uri=f"{PROJECT_SITE}{QUANTITATIVE_FILE}",
    citation_key=CITATION_KEY,
    method="LC-MS/MS absolute abundance in mg-protein/g-dry-cell-weight over 24 Keio "
    "disruptants at 0.2 h-1 and the wild type at 0.1, 0.4, 0.5 and 0.7 h-1; SE = "
    "level * CV / 100 / sqrt(2) over duplicate sample preparation and independent "
    "LC-MS/MS measurement; the reference is the 0.2 h-1 control of the record's own "
    "series",
    page="Science 316:593; project web site v1.0.0, sheet 'Protein'",
)
FLUX_PROVENANCE = Provenance(
    source_uri=f"{PROJECT_SITE}{QUANTITATIVE_FILE}",
    citation_key=CITATION_KEY,
    method="13C metabolic flux analysis over 24 Keio disruptants at 0.2 h-1 and the "
    "wild type at 0.1, 0.4, 0.5 and 0.7 h-1, net flux as a percentage of the specific "
    f"glucose uptake rate; the fit's input is {GC_MS_FILE}, no interval is released",
    page="Science 316:593; project web site v1.0.0, sheet 'Flux'",
)

DATASET_ROOTS: dict[str, str] = {
    "metabolome_ishii2007": "data/torchcell/metabolome_ishii2007",
    "proteome_ishii2007": "data/torchcell/proteome_ishii2007",
    "flux_ishii2007": "data/torchcell/flux_ishii2007",
}


def run_verification(name: str, data_root: str | None = None) -> VerificationReport:
    """Run one arm's family verifier plus the provenance audit on the built dev LMDB.

    The flux arm has no family verifier (no flux dataset has ever been served), so it
    gets the L0 structural gate, the L1 count and the L2 value check of the shared
    helpers directly, plus the bacterial L4 containment over its perturbed loci.

    The metabolome arm passes ``environment_keyed=True``: its four wild-type
    dilution-rate records share one empty genotype, so L1 uniqueness has to key on
    (strain, environment) rather than on the strain alone, which is what a record of a
    genotype IN an environment means.
    """
    from torchcell.verification.levels import l0_structural, l1_count, l2_value_fidelity
    from torchcell.verification.metabolite import (
        metabolite_gene_set,
        verify_metabolite_dataset,
    )
    from torchcell.verification.protein import verify_protein_dataset
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _write_report,
        host_perturbed_gene_set,
        load_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, DATASET_ROOTS[name])
    records = load_records(abs_root)
    if name == "metabolome_ishii2007":
        report = verify_metabolite_dataset(
            records,
            dataset_name=name,
            provenance=METABOLITE_PROVENANCE,
            expected_count=EXPECTED_RECORDS[name],
            reference_centered=False,
            environment_keyed=True,
        )
        measured = metabolite_gene_set(records)
        l4_name = "perturbed_gene_containment_bw25113_locus_tags"
    elif name == "proteome_ishii2007":
        report = verify_protein_dataset(
            records,
            dataset_name=name,
            provenance=PROTEIN_PROVENANCE,
            expected_count=EXPECTED_RECORDS[name],
            allow_duplicate_orfs=False,
        )
        measured = host_perturbed_gene_set(records)
        for record in records:
            measured |= set(record["experiment"]["phenotype"]["protein_abundance"])
        l4_name = "protein_and_perturbed_locus_containment_bw25113"
    else:
        from pydantic import TypeAdapter

        from torchcell.datamodels.schema import ExperimentType

        validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
        report = VerificationReport(dataset_name=name, provenance=FLUX_PROVENANCE)
        report.add(l0_structural((r["experiment"] for r in records), validate))
        report.add(l1_count(len(records), EXPECTED_RECORDS[name]))
        report.add(
            l2_value_fidelity(
                [
                    float(v)
                    for r in records
                    for v in r["experiment"]["phenotype"]["net_flux"].values()
                ],
                allow_nan=False,
            )
        )
        report.add(_l3_flux_interval(records))
        measured = host_perturbed_gene_set(records)
        l4_name = "perturbed_gene_containment_bw25113_locus_tags"

    universe: set[str] = set()
    for serialized in {
        json.dumps(r["reference"]["genome_reference"], sort_keys=True) for r in records
    }:
        universe |= _gene_set_for_reference(json.loads(serialized), base)
    missing = sorted(measured - universe)
    report.add(_containment_result(l4_name, measured, universe, missing))
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, sourced_value_root(value, base)))
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def _l3_flux_interval(records: Sequence[Mapping[str, Any]]) -> Any:
    """L3 for the flux arm: one measurement_type, and no half-stated interval.

    A record carrying one bound and not the other, or a bound with no confidence level,
    would be a flux reported as more precise than it is; this release carries neither,
    and that is what is asserted.
    """
    from torchcell.verification.report import Level, LevelResult

    types = {r["experiment"]["phenotype"]["measurement_type"] for r in records}
    half_stated = [
        idx
        for idx, r in enumerate(records)
        if (
            (r["experiment"]["phenotype"]["net_flux_lower"] is None)
            != (r["experiment"]["phenotype"]["net_flux_upper"] is None)
        )
        or (
            r["experiment"]["phenotype"]["net_flux_lower"] is not None
            and r["experiment"]["phenotype"]["confidence_level"] is None
        )
    ]
    passed = types == {MEASUREMENT_TYPE_FLUX} and not half_stated
    return LevelResult(
        level=Level.L3,
        name="flux_measurement_type_and_interval",
        passed=passed,
        message=f"{len(records)} records share measurement_type {sorted(types)}; "
        f"{len(half_stated)} carry a half-stated interval",
        details={"measurement_types": sorted(types), "half_stated": half_stated[:20]},
    )


def _containment_result(
    name: str, measured: set[str], universe: set[str], missing: Sequence[str]
) -> Any:
    """The L4 containment of every stored identifier in the pinned assembly's loci."""
    from torchcell.verification.report import Level, LevelResult

    return LevelResult(
        level=Level.L4,
        name=name,
        passed=not missing,
        message=f"{len(measured) - len(missing)} of {len(measured)} stored identifiers "
        "are BW25113 GenBank gene rows",
        details={
            "n_measured": len(measured),
            "n_universe": len(universe),
            "missing_examples": list(missing[:20]),
        },
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` a dev LMDB, or ``verify`` one."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.ishii2007"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser(
        "deposit", help="retrieve (optional) and deposit the raw mirror"
    )
    deposit.add_argument("--download-dir", required=True)
    deposit.add_argument(
        "--retrieve",
        action="store_true",
        help="run the recorded retriever for each file into --download-dir first",
    )
    build = sub.add_parser("build", help="build (or load) a dev-tree LMDB")
    build.add_argument("--dataset", choices=sorted(DATASET_ROOTS), required=True)
    verify = sub.add_parser("verify", help="run the verification on a built dev LMDB")
    verify.add_argument("--dataset", choices=sorted(DATASET_ROOTS), required=True)
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        download = Path(args.download_dir)
        if args.retrieve:
            retrieve_raw_files(download)
        print(
            deposit_raw_mirror(
                sources={name: download / name for name in DATA_SHA256},
                data_root=data_root,
            )
        )
        return 0
    if args.command == "build":
        classes: dict[str, type[_Ishii2007Dataset]] = {
            "metabolome_ishii2007": MetabolomeIshii2007Dataset,
            "proteome_ishii2007": ProteomeIshii2007Dataset,
            "flux_ishii2007": FluxIshii2007Dataset,
        }
        dataset = classes[args.dataset](
            root=osp.join(data_root, DATASET_ROOTS[args.dataset])
        )
        print(f"len = {len(dataset)}")
        dataset.close_lmdb()
        return 0
    report = run_verification(args.dataset, data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
