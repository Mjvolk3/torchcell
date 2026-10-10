# torchcell/datasets/ecoli/brunk2016
# [[torchcell.datasets.ecoli.brunk2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/brunk2016
# Test file: tests/torchcell/datasets/ecoli/test_brunk2016.py
"""Brunk 2016 multi-omics time course of nine mevalonate-pathway E. coli DH1 strains.

Brunk et al. 2016 (Cell Systems, doi:10.1016/j.cels.2016.04.004) sampled one batch
fermentation of nine strains, "three isopentenol-producing strains (I1-I3), three
limonene-producing strains (L1-L3), two bisabolene-producing strains (B1-B2), and
wild-type E. coli DH1 (WT) (9 strains total; Supplementary Figure S2 and Table S1)",
over 0 to 72 hours. Four quantities are released per sample and each is served by its
own dataset class, because a verification gate allows ONE ``measurement_type`` per
dataset and these are four scales:

- :class:`MetabolomeBrunk2016Dataset` -- the 51 LC-MS columns the workbook labels
  ``(uM)``, as a ``MetabolitePhenotype`` in micromolar. 117 records.
- :class:`ExometaboliteBrunk2016Dataset` -- the six HPLC columns the workbook labels
  ``g/L`` (glucose and five organic acids), a ``MetabolitePhenotype`` in g/L. 126
  records.
- :class:`ProteomeBrunk2016Dataset` -- the SRM peak areas of the host proteins, a
  ``ProteinAbundancePhenotype``. 81 records.
- :class:`BiofuelTiterBrunk2016Dataset` -- the three fuel titers in g/L, a
  ``ProductTiterExperiment``. 72 records.

NO SCHEMA CHANGE. Every class these records need already exists: the four experiment
pairs, ``HeterologousPathwayPerturbation`` for a pathway gene on a cassette,
``CultureEnvironment`` for the flask, and ``Environment.duration_hours`` for the
sampling time, which is what makes 14 samples of one strain 14 records rather than one.

RETRIEVAL IS SCRIPTABLE AND THE ARTICLE IS AN AUTHOR MANUSCRIPT. The row's accession
read "escholarship.org/uc/item/2k69b9zr (accepted manuscript plus supplementary files;
analysis as iPython notebooks)" and ``accession_confirmed=False``. Measured 2026-10-10:
the three supplementary files come off the Elsevier CDN
(``ars.els-cdn.com/content/image/1-s2.0-S2405471216301120-mmc{1,2,3}``) with no
challenge, and the article's full text is the PMC author-manuscript object
``PMC4882250.1/PMC4882250.1.txt``. Both are deposited into
``$DATA_ROOT/torchcell-raw/brunkCharacterizingStrainVariation2016/`` with sha256 +
retrieval records, as the Mohiuddin 2022 precedent does; nothing goes into Zotero.
``is_pmc_openaccess`` is false for this id, so the PDF is NOT in the bucket and the
quote anchor is the deposited plain text, which is the publisher's own text and not OCR.

TABLE S1 IS A RASTER IMAGE, SO IT IS OCR'd TWICE AND THE READING IS REPAIRED BY A
SOURCED RULE. The strain-to-plasmid table lives only as an image inside ``mmc1.pdf``
(``pdftotext`` returns nothing for it). MinerU reads it at 200 DPI and again at 350 DPI
and both passes agree byte for byte on the strain block -- and both put
``JPUB_002460 + JPUB_002466`` on the ``DH1`` row while leaving ``B2`` empty, a one-row
cell shift. The repair uses no eye and no image: the article calls DH1 "wild-type
E. coli DH1 (WT)", a wild type carries no production plasmid, and exactly one strain row
is empty, so the orphaned pair belongs to that row. :func:`read_table_s1` refuses any
other shape, and :func:`_l4_wild_type_carries_no_pathway_protein` re-asserts the repair
against a different file: in the proteomics workbook DH1's mevalonate-pathway peak areas
are the smallest of the nine strains for every pathway protein, and B2's bisabolene
synthase is 75-fold DH1's.

ONE PERTURBATION PER PATHWAY GENE, AND THE GENE LIST IS PARSED FROM THE PLASMID NAME.
Each plasmid's released name is its part list (``pBbA5c-MevTsa-MK-PMK``), so a strain's
genotype is the gene tokens of its one or two plasmids, each a
``HeterologousPathwayPerturbation``. ``MevT`` is expanded because the SI expands it:
"the modulation of protein expression in the 'top' portion of the mevalonate pathway
(i.e., atoB, HMGS, and HMGR, aka 'MevT')", with the three variants typed from the same
paragraph, "'MevTo' refers to the 'original', non-optimized versions of HMGS and HMGR,
'MevTco' refers to codon-optimized HMGS and HMGR, and 'MevTsa' refers to HMGS and HMGR
that is derived from Staphylococcus aureus". ``source_organism`` is NOT guessed from a
suffix: it is the ``Organism`` column of the proteomics workbook, which states
Escherichia coli for AtoB, Idi, IspA and NudB, Saccharomyces cerevisiae for HMGS, HMGR,
MK, PMK and PMD, and Staphylococcus aureus for the ``sa`` pair.
:data:`PATHWAY_GENES` is checked against those bytes at build time. The three synthases
(GPPS, LS, BIS) carry no organism anywhere in the mirror, so they take the
:data:`SOURCE_ORGANISM_UNREPORTED` sentinel the sibling production loaders use. The four
E. coli genes are extra copies of NATIVE genes, so they are stored under their MG1655
b-numbers; DH1's own lesions are never written by the paper, so the background carries
no alleles, exactly as the Foo 2014 chassis does.

THE REFERENCE IS A RELEASED CULTURE, MATCHED ON THE HOUR. The paper's own difference
profile is "subtract metabolite and protein concentrations or normalized peak areas,
respectively, of the engineered strain from that of the WT strain (based on time
point)", so every metabolome, exometabolite and proteome record references the DH1
sample of its OWN hour, restricted to the keys both measured; a DH1 record is its own
reference. A titer has no wild-type denominator (DH1 makes no fuel), and the SI states
the design instead, "ensuring that a non-optimized variant (e.g., I1, L1, B1) was
included for a baseline comparison", so a titer record references the non-optimized
variant of its own product at its own hour.

REPLICATION: ONE CULTURE PER POINT IN THE SERVED SHEETS, AND THAT IS WHY NO DISPERSION
IS STORED. The served sheets release one number per strain-hour, and the two triplicate
sheets cover other grids entirely (endometabolomics: 4 strains x 5 hours in the first 3
hours; proteomics: 4 strains x 6 hours), so they are not a dispersion OF the served
numbers. ``n_replicates`` is therefore 1 per key and every ``*_se`` is ``None`` with a
typed gap. The SI is explicit that where the paper needed a spread it estimated one,
"For metabolites or peptides that did not have a triplicate measurement, we estimated
the variance using the average variance for all metabolites or peptides measured", which
is a derived quantity and is not ingested.

WHAT IS MEASURED AND NOT STORED, WITH THE COUNTS.

1. The 26 metabolite columns with NO unit in their header (the 20 amino acids, Cystine
   and the five DXP-pathway intermediates). The Methods state a calibration range in
   nM to uM for the LC-MS run as a whole but no column note anywhere in the mirror
   gives these columns a unit, so storing them beside the 51 ``(uM)`` columns would
   assume one. They stay in ``preprocess/unitless_columns.csv``.
2. 24 of the 68 host SRM proteins, whose released key is a JBEI internal id. Two
   released mappings key such an id to ONE gene: the UniProt accession plus ``GN=`` in
   the triplicate sheet (33 of them) and a single-gene GPR in the identifier sheet (20),
   44 distinct between them. The other 24 have only a multi-gene GPR
   (``FRD2`` is ``b4151 and b4152 and b4153 and b4154``), which does not say which
   subunit the measured peptide belongs to, so they carry no locus and are refused. The
   refusal and the two routes are in ``preprocess/protein_keys.csv``.
3. The 13 non-host proteins of the same sheet (the heterologous pathway enzymes and the
   AmpR/Cam/BSA normalization standards). They are not loci of the host assembly, so
   they cannot be protein keys; the pathway ones are on the GENOTYPE of every record.
4. The eight ``72C`` proteomics samples. ``72C`` is not a time the article states and
   no mirrored byte says what the ``C`` is, so those samples get no ``duration_hours``
   and are dropped rather than filed under 72 h beside the real 72 h sample.
5. Nine metabolome rows at hour 2 and the two hour-2 rows that survive it: no ``(uM)``
   cell is filled at hour 2 for any strain, so seven rows carry no value at all, and
   I1 and I3 at hour 2 share no measured metabolite with the DH1 sample of that hour,
   which leaves their reference empty. 126 - 7 - 2 = 117.
6. The derived sheets of both workbooks (difference profiles, enriched/depleted calls,
   fold increases, Tables SI 5 and SI 6, protein percentages, trade-offs, clustering)
   and the per-peptide columns of the proteomics sheet. Each is a call or a ratio over
   the stored numbers, not a second measurement.

THE MEDIUM IS NAMED BUT NOT RECIPED. "100 mL volumes of EZ-Rich defined medium with 1%
glucose in a 1 L Erlenmeyer flask" gives the medium's name, its carbon source and its
amount; no mirrored byte gives the rest of the recipe, so :data:`EZ_RICH_BRUNK2016` is
built in this module and kept OUT of ``MEDIA_LIBRARY``, with the base as one
``composition_deferred`` component and the glucose as a defined one. ``is_synthetic`` is
True because the source calls the medium defined, not because of an inference.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialGeneNamespace,
    BacterialMetaboliteExperiment,
    BacterialMetaboliteExperimentReference,
    BacterialProteinAbundanceExperiment,
    BacterialProteinAbundanceExperimentReference,
    BacterialStrainBackground,
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    CultureFormat,
    EndpointRule,
    Experiment,
    ExperimentReference,
    Genotype,
    HeterologousPathwayPerturbation,
    Media,
    MediaComponent,
    MediaComponentRole,
    MetabolitePhenotype,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    ProteinAbundancePhenotype,
    Publication,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_PAPER_TEXT,
    ROLE_RAW_DATA,
    ROLE_SI_OCR,
    ROLE_SI_PDF,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever
from torchcell.literature.retrieve import elsevier_mmc_url, pmc_cloud_url
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome
from torchcell.verification.levels import (
    l0_structural,
    l1_count,
    l2_value_fidelity,
    l3_convention,
    l4_cross_source,
)
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

DOI = "10.1016/j.cels.2016.04.004"
PMID = "27211860"
PMCID = "PMC4882250"
TITLE = (
    "Characterizing strain variation in engineered E. coli using a multi-omics "
    "based workflow"
)
CITATION_KEY = "brunkCharacterizingStrainVariation2016"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
#: The article's PII, the Elsevier CDN's key for its supplementary components.
ELSEVIER_PII = "S2405471216301120"
#: The PMC Article Datasets bucket prefix of this author manuscript.
PMC_PREFIX = f"{PMCID}.1"
RETRIEVED_AT = "2026-10-10"

PAPER_TEXT_FILE = f"{PMC_PREFIX}.txt"
PAPER_TEXT_REL = f"paper/{PAPER_TEXT_FILE}"
PAPER_TEXT_SHA256 = "35adcad995f599e7a7c946d30f61a0c703480f1ac8029f17b1627239cbc6ba98"
SI_PDF_FILE = "mmc1.pdf"
SI_PDF_REL = f"si/{SI_PDF_FILE}"
SI_PDF_SHA256 = "b8a48ecb1fddefd2760955494fab457c08d93368940a5579caeb920346ed5a2a"
SI_OCR_FILE = "mmc1.md"
SI_OCR_REL = f"si/{SI_OCR_FILE}"
SI_OCR_SHA256 = "ef67dc8da35bf8db7d5870a11675f7f5e39b4727ba24f0c0138dc17cee3f1fb0"
METABOLOMICS_FILE = "mmc2.xlsx"
METABOLOMICS_REL = f"data/{METABOLOMICS_FILE}"
METABOLOMICS_SHA256 = "ff7743d6bc77a8eb6d8540d0b21496ffad18c48d548d9e18620dba4d3d09c045"
PROTEOMICS_FILE = "mmc3.xlsx"
PROTEOMICS_REL = f"data/{PROTEOMICS_FILE}"
PROTEOMICS_SHA256 = "c8161e23b7daec5f13233721be3544252f6df6b4dac8108bf141be2c382b41ae"

#: The MinerU pass whose markdown is deposited as the Table S1 quote anchor.
OCR_TOOL_VERSION = "2.7.6"
OCR_DPI = 200

NAMESPACE: BacterialGeneNamespace = "ecoli_k12_mg1655_bnumber"
REFERENCE_STRAIN: Literal["MG1655"] = "MG1655"
HOST_SPECIES = "Escherichia coli"
HOST_STRAIN_NAME = "DH1"
WILD_TYPE = "DH1"
#: The nine released strains, in the article's own order.
STRAINS: tuple[str, ...] = ("I1", "I2", "I3", "L1", "L2", "L3", "B1", "B2", "DH1")
#: ``strain -> product``, from the article's strain paragraph.
PRODUCT_OF_STRAIN: dict[str, str] = {
    "I1": "isopentenol",
    "I2": "isopentenol",
    "I3": "isopentenol",
    "L1": "limonene",
    "L2": "limonene",
    "L3": "limonene",
    "B1": "bisabolene",
    "B2": "bisabolene",
}
#: ``product -> the non-optimized variant the SI names as its baseline``.
BASELINE_OF_PRODUCT: dict[str, str] = {
    "isopentenol": "I1",
    "limonene": "L1",
    "bisabolene": "B1",
}
#: ``product -> the workbook column holding its titer in g/L``.
TITER_COLUMN: dict[str, str] = {
    "isopentenol": "Isopentenol g/L",
    "limonene": "Limonene g/L",
    "bisabolene": "Bisabolene g/L",
}
#: The synonym ``compound_identity_table.json`` carries isoprenol under; the same route
#: the Foo 2014 loader takes, so these titers join the isoprenol axis.
ISOPENTENOL_SYNONYM = "3-methyl-3-buten-1-ol"

#: The sheets this module reads, by workbook.
SHEET_METABOLITES = "Raw Metabolomics Measurements"
SHEET_METABOLITE_IDS = "Metabolite names and identifier"
SHEET_PROTEOMICS = "Raw proteomics data"
SHEET_PROTEIN_IDS = "Proteomics names and identifier"
SHEET_PROTEOMICS_TRIPLICATE = "Triplicate raw proteomics data "

#: The hour label the proteomics sheet carries beside the nine numeric ones.
UNEXPLAINED_HOUR_LABEL = "72C"

PRODUCTION_TEMPERATURE_C = 30.0
PRODUCTION_SHAKING_RPM = 200.0
PRODUCTION_VOLUME_UL = 100_000.0
PRODUCTION_INOCULUM_OD600 = 0.1
PRODUCTION_VESSEL = "1 L Erlenmeyer flask"
IPTG_UM = 500.0
DODECANE_PERCENT_V_V = 10.0
GLUCOSE_PERCENT_W_V = 1.0

EXPECTED_METABOLOME_RECORDS = 117
EXPECTED_EXOMETABOLITE_RECORDS = 126
EXPECTED_PROTEOME_RECORDS = 81
EXPECTED_TITER_RECORDS = 72
#: The host SRM proteins whose locus the mirror states, measured 2026-10-10.
EXPECTED_PROTEIN_KEYS = 44
#: Every host SRM protein, resolvable or not.
EXPECTED_HOST_PROTEINS = 68
#: The ``(uM)`` columns of the metabolomics sheet.
EXPECTED_UM_COLUMNS = 51
#: The columns with no unit in their header, declined.
EXPECTED_UNITLESS_COLUMNS = 26

#: ``GeneAdditionPerturbation.source_organism`` is a required ``str`` on a leaf with no
#: ``provenance_gaps`` field, so an unreported origin is this explicit sentinel, the
#: value the Foo 2014 and Menasalvas 2025 loaders use.
SOURCE_ORGANISM_UNREPORTED = "unreported"

MEASUREMENT_TYPE_METABOLITE = "lc_ms_concentration_um"
MEASUREMENT_TYPE_EXOMETABOLITE = "hplc_extracellular_concentration_g_per_l"
MEASUREMENT_TYPE_PROTEIN = "srm_protein_peak_area"


# --------------------------------------------------------------------------- #
# Verbatim quotes. Every one below is a substring of the sha256-pinned deposited
# bytes, re-read by verify_quotes() before any value is used.
# --------------------------------------------------------------------------- #
_RESULTS_STRAINS = (
    "Results, 'Pathway description, strain selection, and multi-omics data generation'"
)
_METHODS_GROWTH = (
    "EXPERIMENTAL PROCEDURES, 'Growth conditions and production of advanced biofuels'"
)
_METHODS_SAMPLING = (
    "EXPERIMENTAL PROCEDURES, 'Metabolomics and proteomics sampling and analysis'"
)
_METHODS_STRAINS = "EXPERIMENTAL PROCEDURES, opening paragraph"
_FIGURE_2 = "Figure 2 caption"
_FIGURE_3 = "Figure 3 caption"
_SI_TABLE_S1 = (
    "Table S1, '(Related to Figure 2) Strains and plasmids used in this study'"
)
_SI_FIGURE_S3 = "Figure S3 legend"

_Q_STRAIN_PANEL = (
    "Our analysis included three isopentenol-producing strains (I1-I3), three "
    "limonene-producing strains (L1-L3), two bisabolene-producing strains (B1-B2), "
    "and wild-type E."
)
_Q_SAMPLING = (
    "Samples were collected to measure cell growth, product titer, intracellular and "
    "extracellular metabolites, and selected proteins at multiple time-points (0 to 72 "
    "hours post-induction) in the batch fermentation."
)
_Q_OPTIMIZATION_ORDER = (
    "The numbering of the strains in each set represents their overall performance "
    "(product yield) and evolution of the optimization process (i.e., “1” "
    "represents non-optimized pathway and “2” or “3” represents "
    "variants with better performance)."
)
_Q_HOST = (
    "E. coli DH10B and DH1 were purchased from Invitrogen (Carlsbad, CA) and ATCC, "
    "respectively."
)
_Q_MEDIUM = (
    "For production, 100 mL volumes of EZ-Rich defined medium with 1% glucose in a 1 L "
    "Erlenmeyer flask were inoculated to an initial OD600 of 0.1, incubated with "
    "shaking (30°C, 200 rpm) to an OD600 of 0.6, and induced with 500 µM "
    "isopropyl β-D-1-thiogalactopyranoside (IPTG)."
)
_Q_DODECANE = (
    "For limonene- and bisabolene-producing strains, a 10% overlay of dodecane was "
    "added at induction."
)
_Q_GC = (
    "For isopentenol strains, 0.2 mL of culture was extracted with ethyl acetate for "
    "GC-FID analysis (George et al., 2014)."
)
_Q_HPLC = (
    "Organic acids were analyzed by an Agilent 1200 Series HPLC system equipped with a "
    "photodiode array detector set at 210, 254, and 280 nm."
)
_Q_LCMS = (
    "Intracellular and extracellular metabolites were analyzed by liquid chromatography "
    "mass spectrometry (LC-MS) on a ZIC-HILIC column (150 mm length, 2.1 mm internal "
    "diameter, 2.5 µm particle size) using an Agilent 1200 Series HPLC coupled to "
    "an Agilent 6210 time-of-flight mass spectrometer."
)
_Q_CALIBRATION = (
    "Metabolites were quantified via eight-point calibration curves ranging from 781.25 "
    "nM to 200 µM."
)
_Q_SRM = (
    "Samples were analyzed using an AB Sciex (Foster City, CA) 5500Q-Trap mass "
    "spectrometer operating in MRM (SRM) mode coupled to an Agilent 1100 system."
)
_Q_DIFFERENCE_PROFILE = (
    "subtract metabolite and protein concentrations or normalized peak areas, "
    "respectively, of the engineered strain from that of the WT strain (based on time "
    "point)"
)
_Q_TRIPLICATE_OR_ESTIMATE = (
    "Standard deviations for the test and control condition for each data point were "
    "calculated from triplicate measurements or estimated based on the percent "
    "root-squared deviation (%RSD) of representative triplicate measurements."
)
_Q_ESTIMATED_VARIANCE = (
    "For metabolites or peptides that did not have a triplicate measurement, we "
    "estimated the variance using the average variance for all metabolites or peptides "
    "measured."
)
_Q_THREE_PATHWAY_VERSIONS = (
    "(a) This study characterizes three versions of a heterologous mevalonate pathway "
    "engineered to synthesize isopentenol, limonene, and bisabolene."
)
_Q_MEVT = (
    "A key focus in the optimization process has been the modulation of protein "
    "expression in the “top” portion of the mevalonate pathway (i.e., atoB, "
    "HMGS, and HMGR, aka “MevT”) since the activities of these enzymes "
    "dictate flux to mevalonate and ultimately modulate downstream flux to the desired "
    "fuel product."
)
_Q_MEVT_VARIANTS = (
    "Above, “MevTo” refers to the “original”, non-optimized "
    "versions of HMGS and HMGR, “MevTco” refers to codon-optimized HMGS and "
    "HMGR, and “MevTsa” refers to HMGS and HMGR that is derived from "
    "Staphylococcus aureus."
)
_Q_BASELINE_VARIANT = (
    "For each fuel product, we choose strains with various “levels” of "
    "optimization, while ensuring that a non-optimized variant (e.g., I1, L1, B1) was "
    "included for a baseline comparison."
)

#: Quotes that must be verbatim in the deposited article text.
PAPER_QUOTES: tuple[str, ...] = (
    _Q_STRAIN_PANEL,
    _Q_SAMPLING,
    _Q_OPTIMIZATION_ORDER,
    _Q_HOST,
    _Q_MEDIUM,
    _Q_DODECANE,
    _Q_GC,
    _Q_HPLC,
    _Q_LCMS,
    _Q_CALIBRATION,
    _Q_SRM,
    _Q_TRIPLICATE_OR_ESTIMATE,
    _Q_THREE_PATHWAY_VERSIONS,
)
#: Quotes that must be verbatim in the deposited OCR of the supplementary PDF.
SI_QUOTES: tuple[str, ...] = (
    _Q_DIFFERENCE_PROFILE,
    _Q_MEVT,
    _Q_MEVT_VARIANTS,
    _Q_BASELINE_VARIANT,
    _Q_ESTIMATED_VARIANCE,
)


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """One value sourced to the deposited, sha256-pinned article text."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=PAPER_TEXT_REL,
            citation_key=CITATION_KEY,
            sha256=PAPER_TEXT_SHA256,
            page=page,
        ),
        quote=quote,
        note=note,
    )


def _si(value: Any, quote: str, *, page: str, note: str | None = None) -> SourcedValue:
    """One value sourced to the deposited OCR of the supplementary PDF."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=SI_OCR_REL,
            citation_key=CITATION_KEY,
            sha256=SI_OCR_SHA256,
            page=page,
        ),
        quote=quote,
        note=note,
    )


STRAIN_PANEL = _paper(
    list(STRAINS),
    _Q_STRAIN_PANEL,
    page=_RESULTS_STRAINS,
    note="the nine strains every served sheet is keyed by; the quote stops at 'E.' "
    "because the deposited text breaks the sentence there, and the wild type it names "
    "is DH1",
)
HOST_STRAIN = _paper(
    HOST_STRAIN_NAME,
    _Q_HOST,
    page=_METHODS_STRAINS,
    note="the host of all nine strains. Its K-12 marker genotype is never written, so "
    "the background carries no alleles; MG1655 is the assembly pin because the four "
    "native pathway genes are E. coli genes and their b-numbers belong to it",
)
MEDIUM = _paper(
    "EZ-Rich defined medium with 1% glucose",
    _Q_MEDIUM,
    page=_METHODS_GROWTH,
    note="the medium's name, its carbon source and its amount; no mirrored byte gives "
    "the rest of the recipe, so the base is one composition_deferred component",
)
TEMPERATURE_C = _paper(PRODUCTION_TEMPERATURE_C, _Q_MEDIUM, page=_METHODS_GROWTH)
SHAKING_RPM = _paper(PRODUCTION_SHAKING_RPM, _Q_MEDIUM, page=_METHODS_GROWTH)
WORKING_VOLUME = _paper(PRODUCTION_VOLUME_UL, _Q_MEDIUM, page=_METHODS_GROWTH)
INOCULUM_OD600 = _paper(PRODUCTION_INOCULUM_OD600, _Q_MEDIUM, page=_METHODS_GROWTH)
VESSEL = _paper(PRODUCTION_VESSEL, _Q_MEDIUM, page=_METHODS_GROWTH)
INDUCER = _paper(IPTG_UM, _Q_MEDIUM, page=_METHODS_GROWTH)
DODECANE = _paper(
    DODECANE_PERCENT_V_V,
    _Q_DODECANE,
    page=_METHODS_GROWTH,
    note="the overlay is on the limonene and bisabolene cultures only, so it is an "
    "environment perturbation of the L and B records and of nothing else",
)
AEROBICITY = _paper(
    "aerobic",
    _Q_MEDIUM,
    page=_METHODS_GROWTH,
    note="a shaken flask culture at 200 rpm; the source never uses the word",
)
SAMPLING_WINDOW = _paper(
    "0 to 72 hours post-induction",
    _Q_SAMPLING,
    page=_RESULTS_STRAINS,
    note="the sampling time is what distinguishes the records of one strain, so it is "
    "Environment.duration_hours on every record",
)
QUANTIFICATION_ISOPENTENOL = _paper(
    "GC-FID", _Q_GC, page=_METHODS_SAMPLING, note="isopentenol only"
)
QUANTIFICATION_TERPENE = _paper(
    "GC-MS",
    "For limonene- and bisabolene-producing strains, 0.1 mL of dodecane overlay was "
    "collected and diluted into ethyl acetate for analysis by GC-MS",
    page=_METHODS_SAMPLING,
    note="limonene and bisabolene come off the dodecane overlay",
)
QUANTIFICATION_METABOLITE = _paper(
    "LC-MS (ZIC-HILIC, Agilent 6210 time-of-flight)",
    _Q_LCMS,
    page=_METHODS_SAMPLING,
    note="the instrument behind the (uM) columns; the calibration range is stated "
    f"separately ('{_Q_CALIBRATION}')",
)
QUANTIFICATION_EXOMETABOLITE = _paper(
    "HPLC (Aminex HPX-87H, photodiode array)",
    _Q_HPLC,
    page=_METHODS_SAMPLING,
    note="the instrument behind the g/L columns",
)
QUANTIFICATION_PROTEIN = _paper(
    "SRM (AB Sciex 5500Q-Trap, MRM mode)",
    _Q_SRM,
    page=_METHODS_SAMPLING,
    note="the peak areas the proteomics sheet releases are this run's; the article "
    "calls the layer a relative quantification",
)
DIFFERENCE_AGAINST_WT = _si(
    WILD_TYPE,
    _Q_DIFFERENCE_PROFILE,
    page=_SI_FIGURE_S3,
    note="the paper's own control is the WT sample of the SAME time point, which is "
    "what every metabolome, exometabolite and proteome reference here is",
)


BASELINE_VARIANT = _si(
    BASELINE_OF_PRODUCT,
    _Q_BASELINE_VARIANT,
    page=_SI_TABLE_S1,
    note="the titer reference: a titer has no wild-type denominator, and this is the "
    "comparison the design states",
)
MEVT_EXPANSION = _si(
    ("atoB", "HMGS", "HMGR"),
    _Q_MEVT,
    page=_SI_TABLE_S1,
    note="why a MevT token on a plasmid name becomes three perturbations",
)
MEVT_VARIANTS = _si(
    {
        "MevTo": "original",
        "MevTco": "codon_optimized",
        "MevTsa": "Staphylococcus aureus",
    },
    _Q_MEVT_VARIANTS,
    page=_SI_TABLE_S1,
    note="the three MevT forms, typed from the paragraph that defines them",
)
PATHWAY_VERSIONS = _paper(
    ("isopentenol", "limonene", "bisabolene"),
    _Q_THREE_PATHWAY_VERSIONS,
    page=_FIGURE_2,
    note="the pathway_name of every perturbation names the product this version makes",
)
UNCERTAINTY_NOT_RELEASED = _si(
    None,
    _Q_ESTIMATED_VARIANCE,
    page=_SI_FIGURE_S3,
    note="the served sheets hold ONE number per strain-hour. Where the paper needed a "
    "spread it estimated one from the average variance of all measurements, which is a "
    "derived quantity, so no dispersion is stored and n_replicates is 1 per key",
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
class RawFile(BaseModel):
    """One deposited file: its pinned bytes, role and how it was retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    relpath: str
    role: str
    sha256: str
    bytes: int
    description: str
    derived: bool = False

    @property
    def source_url(self) -> str:
        """The URL the bytes came from (the source PDF's, for the derived OCR)."""
        if self.name == PAPER_TEXT_FILE:
            return pmc_cloud_url(f"{PMC_PREFIX}/{PAPER_TEXT_FILE}")
        if self.derived:
            return elsevier_mmc_url(ELSEVIER_PII, SI_PDF_FILE)
        return elsevier_mmc_url(ELSEVIER_PII, self.name)

    @property
    def retrieval(self) -> RetrievalRecord:
        """The re-runnable retrieval of these bytes."""
        if self.name == PAPER_TEXT_FILE:
            return RetrievalRecord(
                method=RetrievalMethod.pmc_cloud,
                source_url=self.source_url,
                retriever="torchcell.literature.retrieve.pmc_cloud_object",
                params={"key": f"{PMC_PREFIX}/{PAPER_TEXT_FILE}"},
                sha256=self.sha256,
                retrieved_at=RETRIEVED_AT,
            )
        return RetrievalRecord(
            method=RetrievalMethod.direct_url,
            source_url=self.source_url,
            retriever="torchcell.literature.retrieve.elsevier_mmc",
            params={"pii": ELSEVIER_PII, "filename": self.name},
            sha256=self.sha256,
            retrieved_at=RETRIEVED_AT,
        )


RAW_FILES: tuple[RawFile, ...] = (
    RawFile(
        name=METABOLOMICS_FILE,
        relpath=METABOLOMICS_REL,
        role=ROLE_RAW_DATA,
        sha256=METABOLOMICS_SHA256,
        bytes=134574,
        description="Metabolomics Data Analysis: sheet 'Raw Metabolomics Measurements' "
        "holds 126 strain-hour samples x 86 metabolite columns plus OD600 and the "
        "intracellular volume; sheet 'Metabolite names and identifier' maps every "
        "column to its COBRA id; five further sheets are derived calls",
    ),
    RawFile(
        name=PROTEOMICS_FILE,
        relpath=PROTEOMICS_REL,
        role=ROLE_RAW_DATA,
        sha256=PROTEOMICS_SHA256,
        bytes=1804472,
        description="Proteomic Data Analysis: sheet 'Raw proteomics data' holds 13,439 "
        "peptide rows over 89 samples and 81 proteins with their SRM peak areas; "
        "'Proteomics names and identifier' carries each protein's reaction and GPR, and "
        "'Triplicate raw proteomics data ' the UniProt accessions of 38 of them",
    ),
    RawFile(
        name=SI_PDF_FILE,
        relpath=SI_PDF_REL,
        role=ROLE_SI_PDF,
        sha256=SI_PDF_SHA256,
        bytes=1671122,
        description="Supplemental Information: Figures S1 to S7, Table S1 (strains and "
        "plasmids, a raster image) and Table S2 (fold differences at 48 h)",
    ),
    RawFile(
        name=SI_OCR_FILE,
        relpath=SI_OCR_REL,
        role=ROLE_SI_OCR,
        sha256=SI_OCR_SHA256,
        bytes=40995,
        description="MinerU OCR of the supplementary PDF, the quote anchor for Table "
        "S1 and the SI legends. Table S1 is a raster image, so no text layer exists to "
        "quote instead",
        derived=True,
    ),
    RawFile(
        name=PAPER_TEXT_FILE,
        relpath=PAPER_TEXT_REL,
        role=ROLE_PAPER_TEXT,
        sha256=PAPER_TEXT_SHA256,
        bytes=60693,
        description="The author manuscript's full text as PMC renders it, the anchor "
        "of every article quote. Not OCR: it is the publisher's own text",
    ),
)
RAW_FILES_BY_NAME: dict[str, RawFile] = {f.name: f for f in RAW_FILES}
RETRIEVED_FILES: tuple[RawFile, ...] = tuple(f for f in RAW_FILES if not f.derived)
#: ``{file name: pinned sha256}`` over the files a build reads.
DATA_SHA256: dict[str, str] = {
    METABOLOMICS_FILE: METABOLOMICS_SHA256,
    PROTEOMICS_FILE: PROTEOMICS_SHA256,
}

#: What the release holds that no loader here consumes.
NOT_MIRRORED = (
    "The accepted manuscript PDF (mmc1.pdf is the SI; the article PDF is not in the "
    "PMC bucket because this id is an author manuscript, is_pmc_openaccess false). The "
    "deposited plain text carries the same words and is what every quote is read from",
    "github.com/SBRG/Strain_characterization_workflow: the four analysis notebooks, "
    "the iJO1366 model files and the per-strain CSV exports of the two workbooks. The "
    "CSVs are the workbooks' own numbers re-keyed to COBRA ids, which this loader does "
    "from the workbooks themselves, so mirroring them would mirror the same numbers",
    "The derived sheets of both workbooks: difference profiles, the enriched and "
    "depleted calls, the fold-increase blocks, Table SI 5, Table SI 6, the protein "
    "percentages, the proteome trade-offs and the clustering motifs. Each is a call or "
    "a ratio over the stored numbers",
    "The two triplicate sheets: 'Raw Triplicate Endometabolomics' (4 strains x 5 hours "
    "in the first 3 h) and 'Triplicate raw proteomics data ' (4 strains x 6 hours). "
    "They cover different grids from the served sheets, so they are not those numbers' "
    "dispersion; the proteomics one IS read, for the UniProt accessions it carries",
)


def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/brunkCharacterizingStrainVariation2016``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ocr_processing() -> ProcessingRecord:
    """How the deposited Table S1 markdown was produced, and from which bytes."""
    return ProcessingRecord(
        processor="torchcell.literature.ocr.ocr_pdf",
        tool="mineru",
        version=OCR_TOOL_VERSION,
        params={
            "backend": "pipeline",
            "lang": "en",
            "method": "auto",
            "device_mode": "cpu",
            "dpi": OCR_DPI,
            "second_pass_dpi": 350,
            "second_pass_agreement": "the strain block of Table S1 is byte-identical "
            "between the 200 DPI and 350 DPI passes",
        },
        input_sha256=[SI_PDF_SHA256],
    )


def retrieve_raw_files(dest_dir: str | Path) -> dict[str, Path]:
    """Run each recorded retriever and write the sha256-verified bytes to ``dest_dir``."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for raw in RETRIEVED_FILES:
        path = dest / raw.name
        payload = run_retriever(raw.retrieval)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != raw.sha256:
            raise RuntimeError(
                f"{raw.source_url} sha256 mismatch: got {digest}, expected {raw.sha256}"
            )
        path.write_bytes(payload)
        out[raw.name] = path
    return out


def deposit_raw_mirror(
    *, sources: Mapping[str, str | Path], data_root: str | None = None
) -> Path:
    """Write the raw mirror from already-retrieved files plus its ``manifest.json``.

    ``sources`` maps every name in :data:`RAW_FILES` to a local file, the OCR markdown
    included (it is produced by ``torchcell.literature.ocr.ocr_pdf`` on the deposited
    PDF and carries that run's :class:`ProcessingRecord`). Idempotent by sha256: a
    mirror file with the pinned hash is left alone, one with any other hash raises.
    """
    missing = sorted({f.name for f in RAW_FILES} - set(sources))
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
        dest = root / raw.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
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
                retrieval=None if raw.derived else raw.retrieval,
                processing=ocr_processing() if raw.derived else None,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
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
    return Manifest.model_validate_json(
        (raw_mirror_dir(data_root) / "manifest.json").read_text()
    )


def verify_quotes(data_root: str | None = None) -> int:
    """Re-read every quote from the pinned mirror bytes; return how many were checked.

    A hash pin does not make a transcription verbatim, so each quote is searched in the
    bytes it claims to come from and a miss raises before any value is used.
    """
    root = raw_mirror_dir(data_root)
    checked = 0
    for relpath, digest, quotes in (
        (PAPER_TEXT_REL, PAPER_TEXT_SHA256, PAPER_QUOTES),
        (SI_OCR_REL, SI_OCR_SHA256, SI_QUOTES),
    ):
        path = root / relpath
        got = _sha256(path)
        if got != digest:
            raise RuntimeError(f"{path} is not the pinned bytes ({got})")
        text = path.read_text(encoding="utf-8")
        for quote in quotes:
            if quote not in text:
                raise RuntimeError(f"quote not verbatim in {relpath}: {quote!r}")
            checked += 1
    return checked


# --------------------------------------------------------------------------- #
# Table S1, read out of the deposited OCR
# --------------------------------------------------------------------------- #
_PLASMID_ID_RE = re.compile(r"\b(JPUB|JBx)[\s_]*0*(\d{4,6})\b")
_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<td>(.*?)</td>", re.S)
_PLASMID_HEADER = "Plasmids"
_STRAIN_HEADER = "Strains"


def normalize_plasmid_ids(cell: str) -> str:
    """``JPUB 004937`` / ``JPUB _006210`` -> ``JPUB_004937``.

    The OCR renders the underscore of a plasmid id as a space or drops it, so ids are
    normalized before a strain's composition is joined to the plasmid block. Nothing
    else in the cell is touched.
    """
    return _PLASMID_ID_RE.sub(lambda m: f"{m.group(1)}_{m.group(2).zfill(6)}", cell)


class TableS1(BaseModel):
    """Table S1 as read: the plasmid block, the strain block and the repair applied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plasmids: dict[str, str]
    composition: dict[str, tuple[str, ...]]
    repaired_from: str | None
    repaired_to: str | None


def read_table_s1(path: str | Path) -> TableS1:
    """The plasmid and strain blocks of Table S1 from the deposited OCR markdown.

    The strain block's last rows come out of OCR one row low (see the module
    docstring), so the reading is repaired by the article's own statement that DH1 is
    the wild type: when the wild-type row carries a composition and exactly one other
    strain row is empty, the composition moves to that row. Any other shape raises.
    """
    text = Path(path).read_text(encoding="utf-8")
    tables = [
        table
        for table in re.findall(r"<table>.*?</table>", text, re.S)
        if _PLASMID_HEADER in table and _STRAIN_HEADER in table
    ]
    if len(tables) != 1:
        raise RuntimeError(f"{path} holds {len(tables)} Table S1 candidates, not one")
    rows = [
        [re.sub(r"\s+", " ", normalize_plasmid_ids(cell)).strip() for cell in cells]
        for cells in (_CELL_RE.findall(row) for row in _ROW_RE.findall(tables[0]))
    ]
    plasmids: dict[str, str] = {}
    composition: dict[str, tuple[str, ...]] = {}
    block = None
    for cells in rows:
        if not cells:
            continue
        if cells[0] == _PLASMID_HEADER:
            block = "plasmid"
            continue
        if cells[0] == _STRAIN_HEADER:
            block = "strain"
            continue
        if block == "plasmid":
            plasmids[cells[0]] = cells[1]
        elif block == "strain":
            composition[cells[0]] = tuple(
                re.findall(r"J\w+_\d{6}", cells[1] if len(cells) > 1 else "")
            )
    if sorted(composition) != sorted(STRAINS):
        raise RuntimeError(
            f"Table S1's strain block is {sorted(composition)}, not {sorted(STRAINS)}"
        )
    repaired_from = repaired_to = None
    if composition[WILD_TYPE]:
        empty = [name for name, ids in composition.items() if not ids]
        if len(empty) != 1:
            raise RuntimeError(
                f"the wild type carries {composition[WILD_TYPE]} and {len(empty)} "
                "strain rows are empty; the one-row OCR shift is not what this is"
            )
        mutable = dict(composition)
        mutable[empty[0]] = composition[WILD_TYPE]
        mutable[WILD_TYPE] = ()
        composition = mutable
        repaired_from, repaired_to = WILD_TYPE, empty[0]
    used = {pid for ids in composition.values() for pid in ids}
    if used != set(plasmids):
        raise RuntimeError(
            f"Table S1 lists {sorted(set(plasmids) - used)} unused and "
            f"{sorted(used - set(plasmids))} unlisted plasmids"
        )
    return TableS1(
        plasmids=plasmids,
        composition=composition,
        repaired_from=repaired_from,
        repaired_to=repaired_to,
    )


# --------------------------------------------------------------------------- #
# The pathway genes a plasmid name lists
# --------------------------------------------------------------------------- #
class PathwayGene(BaseModel):
    """One pathway gene token, its organism and (for a native gene) its host locus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str
    gene_name: str
    source_organism: str
    is_heterologous: bool
    #: The proteomics sheet's own name for this protein, where it measures it.
    protein_name: str | None = None
    variant: str | None = None


#: Every token a Table S1 plasmid name can carry, with the organism the proteomics
#: workbook's ``Organism`` column states for it. ``check_pathway_organisms`` re-reads
#: those bytes at build time, so none of this is assumed.
PATHWAY_GENES: dict[str, PathwayGene] = {
    "atoB": PathwayGene(
        token="atoB",
        gene_name="atoB",
        source_organism=HOST_SPECIES,
        is_heterologous=False,
        protein_name="AtoB",
    ),
    "idi": PathwayGene(
        token="idi",
        gene_name="idi",
        source_organism=HOST_SPECIES,
        is_heterologous=False,
        protein_name="Idi",
    ),
    "ispA": PathwayGene(
        token="ispA",
        gene_name="ispA",
        source_organism=HOST_SPECIES,
        is_heterologous=False,
        protein_name="IspA",
    ),
    "nudB": PathwayGene(
        token="nudB",
        gene_name="nudB",
        source_organism=HOST_SPECIES,
        is_heterologous=False,
        protein_name="NudB",
    ),
    "HMGS": PathwayGene(
        token="HMGS",
        gene_name="HMGS",
        source_organism="Saccharomyces cerevisiae",
        is_heterologous=True,
        protein_name="HMGS",
    ),
    "HMGR": PathwayGene(
        token="HMGR",
        gene_name="HMGR",
        source_organism="Saccharomyces cerevisiae",
        is_heterologous=True,
        protein_name="HMGR",
    ),
    "HMGSsa": PathwayGene(
        token="HMGSsa",
        gene_name="HMGS",
        source_organism="Staphylococcus aureus",
        is_heterologous=True,
        protein_name="HMGS",
    ),
    "HMGRsa": PathwayGene(
        token="HMGRsa",
        gene_name="HMGR",
        source_organism="Staphylococcus aureus",
        is_heterologous=True,
        protein_name="HMGR",
    ),
    "MK": PathwayGene(
        token="MK",
        gene_name="MK",
        source_organism="Saccharomyces cerevisiae",
        is_heterologous=True,
        protein_name="MK",
    ),
    "PMK": PathwayGene(
        token="PMK",
        gene_name="PMK",
        source_organism="Saccharomyces cerevisiae",
        is_heterologous=True,
        protein_name="PMK",
    ),
    "PMD": PathwayGene(
        token="PMD",
        gene_name="PMD",
        source_organism="Saccharomyces cerevisiae",
        is_heterologous=True,
        protein_name="PMD",
    ),
    "GPPS": PathwayGene(
        token="GPPS",
        gene_name="GPPS",
        source_organism=SOURCE_ORGANISM_UNREPORTED,
        is_heterologous=True,
        protein_name="GPPS",
    ),
    "LS": PathwayGene(
        token="LS",
        gene_name="LS",
        source_organism=SOURCE_ORGANISM_UNREPORTED,
        is_heterologous=True,
        protein_name="Limonene Synthase",
    ),
    "BIS": PathwayGene(
        token="BIS",
        gene_name="BIS",
        source_organism=SOURCE_ORGANISM_UNREPORTED,
        is_heterologous=True,
        protein_name="Bisabolene",
    ),
}
#: The three MevT forms the SI defines, each as its three genes plus the variant word.
MEVT_FORMS: dict[str, tuple[tuple[str, str | None], ...]] = {
    "MevTo": (("atoB", None), ("HMGS", "original"), ("HMGR", "original")),
    "MevTco": (
        ("atoB", None),
        ("HMGS", "codon_optimized"),
        ("HMGR", "codon_optimized"),
    ),
    "MevTsa": (("atoB", None), ("HMGSsa", None), ("HMGRsa", None)),
}
#: Name parts that are not genes: the BglBrick vector backbones and the supplemental
#: promoter the SI names ("the insertion of supplemental promoters (e.g., a 'trc'
#: promoter) to divide the pathway into separate operons").
NON_GENE_TOKENS: frozenset[str] = frozenset(
    {"pBbA5c", "pBbE1a", "pBbS5k", "pTrc99A", "trc"}
)


class PathwayPart(BaseModel):
    """One gene a plasmid name lists, with the plasmid it sits on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plasmid_id: str
    plasmid_name: str
    gene: PathwayGene
    variant: str | None


def parse_plasmid_genes(plasmid_id: str, plasmid_name: str) -> list[PathwayPart]:
    """The gene tokens of one released plasmid name, MevT expanded.

    The name IS the part list, so it is split on the hyphen; a backbone or a ``trc``
    promoter token is skipped and anything else must be a known gene, which is what
    keeps a renamed construct from silently losing a gene. ``nudB`` is matched
    case-insensitively because Table S1 writes ``pTrc99A-nudB-PMD`` while the
    proteomics sheet writes ``NudB``.
    """
    parts: list[PathwayPart] = []
    by_lower = {token.lower(): token for token in PATHWAY_GENES}
    for raw in plasmid_name.split("-"):
        token = raw.strip()
        if not token or token in NON_GENE_TOKENS:
            continue
        if token in MEVT_FORMS:
            for gene_token, variant in MEVT_FORMS[token]:
                parts.append(
                    PathwayPart(
                        plasmid_id=plasmid_id,
                        plasmid_name=plasmid_name,
                        gene=PATHWAY_GENES[gene_token],
                        variant=variant,
                    )
                )
            continue
        known = by_lower.get(token.lower())
        if known is None:
            raise RuntimeError(
                f"{plasmid_id} ({plasmid_name}) lists {token!r}, which is neither a "
                "known pathway gene, a vector backbone nor the trc promoter"
            )
        parts.append(
            PathwayPart(
                plasmid_id=plasmid_id,
                plasmid_name=plasmid_name,
                gene=PATHWAY_GENES[known],
                variant=None,
            )
        )
    if not parts:
        raise RuntimeError(f"{plasmid_id} ({plasmid_name}) lists no gene")
    return parts


def strain_parts(table: TableS1) -> dict[str, list[PathwayPart]]:
    """Every strain's pathway genes, in the order its plasmids list them."""
    out: dict[str, list[PathwayPart]] = {}
    for strain, plasmid_ids in table.composition.items():
        parts: list[PathwayPart] = []
        for plasmid_id in plasmid_ids:
            parts.extend(parse_plasmid_genes(plasmid_id, table.plasmids[plasmid_id]))
        out[strain] = parts
    return out


# --------------------------------------------------------------------------- #
# The two deposited workbooks
# --------------------------------------------------------------------------- #
_UM_SUFFIX = "(uM)"
_GL_SUFFIX = "g/L"
#: Columns of the metabolomics sheet that are not a metabolite measurement.
_NON_METABOLITE_COLUMNS: frozenset[str] = frozenset(
    {"Hour", "Strain", "Sample", "OD600", "Intracellular volume / sample"}
)


class MetaboliteColumns(BaseModel):
    """The metabolomics sheet's columns, split by the unit in their own header."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    micromolar: tuple[str, ...]
    grams_per_litre: tuple[str, ...]
    product: tuple[str, ...]
    unitless: tuple[str, ...]
    #: Released column name -> the COBRA id the identifier sheet gives it.
    cobra_id: dict[str, str]


def read_metabolite_columns(path: str | Path) -> MetaboliteColumns:
    """Classify every metabolite column of the workbook by its header's unit."""
    ids = pd.read_excel(path, sheet_name=SHEET_METABOLITE_IDS, index_col=0)
    cobra = dict(zip(ids["JBEI_id"].astype(str), ids["COBRA_met"].astype(str)))
    if len(cobra) != len(ids):
        raise RuntimeError("the identifier sheet names a column twice")
    frame = pd.read_excel(path, sheet_name=SHEET_METABOLITES, nrows=1)
    columns = [c for c in frame.columns if c not in _NON_METABOLITE_COLUMNS]
    unknown = sorted(set(columns) - set(cobra))
    if unknown:
        raise RuntimeError(f"{unknown} have no COBRA id in the identifier sheet")
    product = tuple(TITER_COLUMN[name] for name in PATHWAY_VERSIONS.value)
    micromolar = tuple(c for c in columns if c.endswith(_UM_SUFFIX))
    grams = tuple(c for c in columns if c.endswith(_GL_SUFFIX) and c not in product)
    unitless = tuple(
        c
        for c in columns
        if c not in micromolar and c not in grams and c not in product
    )
    if len(micromolar) != EXPECTED_UM_COLUMNS:
        raise RuntimeError(f"{len(micromolar)} (uM) columns, not {EXPECTED_UM_COLUMNS}")
    if len(unitless) != EXPECTED_UNITLESS_COLUMNS:
        raise RuntimeError(
            f"{len(unitless)} columns without a unit, not {EXPECTED_UNITLESS_COLUMNS}"
        )
    return MetaboliteColumns(
        micromolar=micromolar,
        grams_per_litre=grams,
        product=product,
        unitless=unitless,
        cobra_id=cobra,
    )


def read_metabolite_samples(path: str | Path) -> pd.DataFrame:
    """The 126 strain-hour rows of the metabolomics sheet, checked for duplicates."""
    frame = pd.read_excel(path, sheet_name=SHEET_METABOLITES)
    frame = frame[frame["Strain"].notna()].copy()
    frame["Strain"] = frame["Strain"].astype(str)
    frame["Hour"] = frame["Hour"].astype(float)
    unknown = sorted(set(frame["Strain"]) - set(STRAINS))
    if unknown:
        raise RuntimeError(f"the metabolomics sheet holds strains {unknown}")
    if frame.duplicated(["Strain", "Hour"]).any():
        raise RuntimeError("the metabolomics sheet repeats a strain-hour sample")
    return frame


class ProteinKey(BaseModel):
    """One SRM protein, and the released mapping (if any) that keys it to a locus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protein: str
    organism: str | None
    gene_name: str | None
    route: str | None
    reason: str | None = None


#: The two mapping routes, named in the ledger so a key's origin is recoverable.
ROUTE_UNIPROT = "uniprot_gene_name_in_the_triplicate_sheet"
ROUTE_SINGLE_GENE_GPR = "single_gene_gpr_in_the_identifier_sheet"
REFUSAL_MULTI_GENE_GPR = (
    "the identifier sheet's GPR names several genes for this protein's reaction, so "
    "which gene the measured peptide belongs to is not stated"
)
REFUSAL_NO_MAPPING = (
    "no UniProt accession and no GPR anywhere in the mirror names a gene for this "
    "protein (the identifier sheet writes 'not in model/not found', or carries no row)"
)
_UNIPROT_NAME_RE = re.compile(r"^sp\|[^|]+\|([A-Za-z0-9]+)_ECOLI$")
_GENE_NAME_RE = re.compile(r"GN=(\S+)")
_BNUMBER_RE = re.compile(r"\bb\d{4}\b")


def read_protein_keys(path: str | Path) -> list[ProteinKey]:
    """Every protein of the proteomics sheet, with the route that keys it to a gene.

    Route 1 is the UniProt accession plus ``GN=`` the triplicate sheet carries; route 2
    is a GPR that names exactly one gene. A protein with neither carries no locus and
    is refused with the reason, never with a guessed subunit.
    """
    raw = pd.read_excel(path, sheet_name=SHEET_PROTEOMICS)
    ids = pd.read_excel(path, sheet_name=SHEET_PROTEIN_IDS, index_col=0)
    triplicate = pd.read_excel(
        path, sheet_name=SHEET_PROTEOMICS_TRIPLICATE, index_col=0
    )

    uniprot: dict[str, set[str]] = {}
    for name, description in zip(
        triplicate["ProteinName"].astype(str),
        triplicate["ProteinDescription"].astype(str),
    ):
        match = _UNIPROT_NAME_RE.match(name)
        gene = _GENE_NAME_RE.search(description)
        if match is None or gene is None:
            continue
        uniprot.setdefault(match.group(1), set()).add(gene.group(1))

    gprs: dict[str, set[str]] = {}
    for jbei, rule in zip(ids["JBEI_id"].astype(str), ids["GPR"].astype(str)):
        gprs.setdefault(jbei, set()).add(rule)

    keys: list[ProteinKey] = []
    seen: set[tuple[str, str | None]] = set()
    for protein, organism in zip(
        raw["Protein"].astype(str), raw["Organism"].astype(object)
    ):
        species = None if pd.isna(organism) else str(organism)
        if (protein, species) in seen:
            continue
        seen.add((protein, species))
        gene_names = uniprot.get(protein, set())
        if len(gene_names) == 1:
            keys.append(
                ProteinKey(
                    protein=protein,
                    organism=species,
                    gene_name=next(iter(gene_names)),
                    route=ROUTE_UNIPROT,
                )
            )
            continue
        gpr: set[str] = gprs.get(protein, set())
        single = {g for g in gpr if len(set(_BNUMBER_RE.findall(g))) == 1}
        if len(gpr) == 1 and len(single) == 1:
            keys.append(
                ProteinKey(
                    protein=protein,
                    organism=species,
                    gene_name=_BNUMBER_RE.findall(next(iter(single)))[0],
                    route=ROUTE_SINGLE_GENE_GPR,
                )
            )
            continue
        names_any_gene = any(_BNUMBER_RE.search(rule) for rule in gpr)
        keys.append(
            ProteinKey(
                protein=protein,
                organism=species,
                gene_name=None,
                route=None,
                reason=REFUSAL_MULTI_GENE_GPR if names_any_gene else REFUSAL_NO_MAPPING,
            )
        )
    return sorted(keys, key=lambda key: key.protein)


def read_protein_areas(path: str | Path) -> pd.DataFrame:
    """One row per (sample, protein) with the released ``ProteinArea``.

    The sheet repeats a protein's area on each of its peptide rows, so the area is
    read as the single value those rows agree on; a sample-protein pair with two
    different areas raises rather than being averaged.
    """
    raw = pd.read_excel(path, sheet_name=SHEET_PROTEOMICS)
    raw["Hour"] = raw["Hour"].astype(str)
    raw["Strain"] = raw["Strain"].astype(str)
    raw["Protein"] = raw["Protein"].astype(str)
    raw["Organism"] = raw["Organism"].astype(object).where(raw["Organism"].notna(), "")
    grouped = raw.groupby(["Strain", "Hour", "Protein", "Organism"], dropna=False)
    spread = grouped["ProteinArea"].nunique()
    if int(spread.max()) != 1:
        bad = spread[spread > 1]
        raise RuntimeError(
            f"{len(bad)} sample-protein pairs carry more than one ProteinArea"
        )
    return grouped["ProteinArea"].first().reset_index()


# --------------------------------------------------------------------------- #
# The shared record context: publication, host, medium, environment
# --------------------------------------------------------------------------- #
def publication() -> Publication:
    """This paper, by PubMed id and DOI."""
    return Publication(
        pubmed_id=PMID,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
        doi=DOI,
        doi_url=f"https://doi.org/{DOI}",
    )


def host_background() -> BacterialStrainBackground:
    """Plain DH1, pinned to the MG1655 assembly, with no asserted lesions.

    ``alleles`` is empty because the paper names DH1 and its supplier and never writes
    a marker genotype; that is the source's silence, not a claim that DH1 equals
    MG1655. The pin is MG1655 because the four native pathway genes are E. coli genes
    whose b-numbers belong to that assembly.
    """
    return BacterialStrainBackground(
        name=HOST_STRAIN_NAME,
        reference_strain=REFERENCE_STRAIN,
        assembly_set="ecoli_K12_MG1655_ASM584v2",
        parents=None,
        construction=None,
        genotype_statement=None,
        alleles=[],
        provenance=[HOST_STRAIN],
    )


def host_reference(data_root: str | None = None) -> AssemblyReferenceGenome:
    """The DH1 host pinned to the MG1655 GenBank assembly."""
    return assembly_reference(
        REFERENCE_STRAIN, background=host_background(), data_root=data_root
    )


EZ_RICH_BRUNK2016 = Media(
    name="EZ-Rich defined medium with 1% glucose (Brunk 2016)",
    state="liquid",
    is_synthetic=True,
    base_medium="EZ_RICH",
    components=[
        MediaComponent(
            compound=Compound(name="EZ-Rich defined medium base (no recipe stated)"),
            role=MediaComponentRole.other,
            concentration=None,
            definition=ComponentDefinition.composition_deferred,
            provenance=[MEDIUM],
            defers_to=["neidhardtCultureMediumEnterobacteria1974"],
            note="the paper names the medium and nothing else about it; not one "
            "millimolar figure is copied in from another EZ-Rich study, and the "
            "deferral names the formulation the commercial medium is sold as",
        ),
        MediaComponent(
            compound=resolved_compound("D-glucose"),
            role=MediaComponentRole.carbon_source,
            concentration=Concentration(
                value=GLUCOSE_PERCENT_W_V, unit=ConcentrationUnit.percent_w_v
            ),
            definition=ComponentDefinition.defined,
            provenance=[MEDIUM],
        ),
    ],
    provenance=[MEDIUM],
)
"""The production medium, kept in this module and OUT of ``MEDIA_LIBRARY``.

The library holds recipes; this object holds a named medium whose recipe the source
does not give, and sharing it would invite a second loader to attach amounts to it.
"""


def culture_format() -> CultureFormat:
    """The 100 mL shaken flask every sample came out of."""
    return CultureFormat(
        vessel=str(VESSEL.value),
        working_volume_ul=float(WORKING_VOLUME.value),
        shaking_rpm=float(SHAKING_RPM.value),
        inoculum_od600=float(INOCULUM_OD600.value),
        endpoint=EndpointRule.fixed_duration,
        provenance=[VESSEL, WORKING_VOLUME, SHAKING_RPM, INOCULUM_OD600],
    )


def environment(strain: str, hour: float) -> CultureEnvironment:
    """The culture one sample was drawn from, at its own sampling hour.

    ``duration_hours`` is the hour the sample was taken, which is what makes the 14
    samples of one strain 14 environments rather than one. The dodecane overlay is on
    the limonene and bisabolene cultures only, so it travels with those strains'
    records and with no others.
    """
    perturbations: list[Any] = [
        SmallMoleculePerturbation(
            compound=resolved_compound("IPTG"),
            concentration=Concentration(
                value=float(INDUCER.value), unit=ConcentrationUnit.micromolar
            ),
        )
    ]
    if PRODUCT_OF_STRAIN.get(strain) in ("limonene", "bisabolene"):
        perturbations.append(
            SmallMoleculePerturbation(
                compound=resolved_compound("dodecane"),
                concentration=Concentration(
                    value=float(DODECANE.value), unit=ConcentrationUnit.percent_v_v
                ),
            )
        )
    return CultureEnvironment(
        media=EZ_RICH_BRUNK2016,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        culture_format=culture_format(),
        perturbations=perturbations,
        aerobicity=str(AEROBICITY.value),
        duration_hours=hour,
    )


def pathway_perturbation(
    part: PathwayPart, locus_of: Mapping[str, str], product: str
) -> HeterologousPathwayPerturbation:
    """One pathway gene of one plasmid, as the record's genotype carries it.

    A native E. coli gene is stored under its MG1655 b-number, which is the case the
    leaf's docstring names (an extra copy of a native gene); a heterologous gene keeps
    the released token, since it has no host locus tag.
    """
    gene = part.gene
    name = locus_of[gene.gene_name] if not gene.is_heterologous else gene.token
    return HeterologousPathwayPerturbation(
        systematic_gene_name=name,
        perturbed_gene_name=gene.gene_name,
        gene_namespace=NAMESPACE,
        pathway_name=f"heterologous mevalonate pathway to {product}",
        source_organism=gene.source_organism,
        is_heterologous=gene.is_heterologous,
        localization="episomal_plasmid",
        construct_name=part.plasmid_name,
        variant=part.variant,
    )


def genotype(
    strain: str, parts: Sequence[PathwayPart], locus_of: Mapping[str, str]
) -> Genotype:
    """One strain's genotype: a perturbation per pathway gene, none for the wild type."""
    if strain == WILD_TYPE:
        if parts:
            raise RuntimeError(f"{WILD_TYPE} is the wild type and carries no plasmid")
        return Genotype(perturbations=[])
    product = PRODUCT_OF_STRAIN[strain]
    return Genotype(
        perturbations=[pathway_perturbation(part, locus_of, product) for part in parts]
    )


def check_pathway_organisms(path: str | Path) -> int:
    """Assert every typed ``source_organism`` against the proteomics sheet's own column.

    The organism of a pathway gene is NOT read off its token's suffix: it is the
    ``Organism`` the workbook states for the protein it measures. A gene the workbook
    MEASURES with no organism must carry the unreported sentinel; a gene it does not
    measure at all is left alone, because silence there is not a statement. Returns how
    many genes the workbook settles.
    """
    raw = pd.read_excel(path, sheet_name=SHEET_PROTEOMICS)
    released: dict[tuple[str, str], set[str]] = {}
    for protein, organism in zip(
        raw["Protein"].astype(str), raw["Organism"].astype(object)
    ):
        species = "" if pd.isna(organism) else str(organism)
        released.setdefault((protein, species), set()).add(species)
    organisms: dict[str, set[str]] = {}
    for protein, species in released:
        organisms.setdefault(protein, set()).add(species)
    checked = 0
    for gene in PATHWAY_GENES.values():
        if gene.protein_name is None or gene.protein_name not in organisms:
            # The workbook does not measure this protein at all, so it states nothing
            # either way; the organism stays whatever the typed table carries.
            continue
        stated = {s for s in organisms[gene.protein_name] if s}
        if not stated:
            if gene.source_organism != SOURCE_ORGANISM_UNREPORTED:
                raise RuntimeError(
                    f"{gene.token}: the workbook states no organism for "
                    f"{gene.protein_name!r}, so source_organism must be the "
                    f"unreported sentinel, not {gene.source_organism!r}"
                )
            continue
        if gene.source_organism not in stated:
            raise RuntimeError(
                f"{gene.token}: the workbook states {sorted(stated)} for "
                f"{gene.protein_name!r}, not {gene.source_organism!r}"
            )
        checked += 1
    return checked


# --------------------------------------------------------------------------- #
# Phenotypes
# --------------------------------------------------------------------------- #
_SE_GAP_NOTE = (
    "the served sheet releases ONE number per strain-hour, so there is no dispersion "
    "to store: n_replicates is 1 per key and the SE map is None. The two triplicate "
    "sheets cover other grids (4 strains x 5 early hours for metabolites, 4 strains x "
    "6 hours for proteins), and where the paper needed a spread for a point outside "
    f"them it estimated one ('{_Q_ESTIMATED_VARIANCE}')"
)


def metabolite_phenotype(
    levels: Mapping[str, float], measurement_type: str
) -> MetabolitePhenotype:
    """One sample's metabolite profile, with no dispersion and no Yeast9 mapping."""
    return MetabolitePhenotype(
        metabolite_level=dict(levels),
        metabolite_level_se=None,
        n_replicates={key: 1 for key in levels},
        measurement_type=measurement_type,
        target_metabolite_ids=None,
        provenance_gaps=[
            ProvenanceGap(
                field="metabolite_level_se",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=_SE_GAP_NOTE,
            ),
            ProvenanceGap(
                field="target_metabolite_ids",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="those are Yeast9 s_NNNN ids and this is an E. coli dataset; the "
                "release's own COBRA (iJO1366) ids are what the keys already are",
            ),
        ],
    )


def protein_phenotype(areas: Mapping[str, float]) -> ProteinAbundancePhenotype:
    """One sample's SRM peak areas, keyed by MG1655 locus tag."""
    return ProteinAbundancePhenotype(
        protein_abundance=dict(areas),
        protein_abundance_se=None,
        n_replicates={key: 1 for key in areas},
        measurement_type=MEASUREMENT_TYPE_PROTEIN,
        provenance_gaps=[
            ProvenanceGap(
                field="protein_abundance_se",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=_SE_GAP_NOTE,
            )
        ],
    )


def titer_phenotype(product: str, titer_g_per_l: float) -> ProductTiterPhenotype:
    """One released fuel titer in g/L, with the typed absences the release forces."""
    name = ISOPENTENOL_SYNONYM if product == "isopentenol" else product
    method = (
        str(QUANTIFICATION_ISOPENTENOL.value)
        if product == "isopentenol"
        else str(QUANTIFICATION_TERPENE.value)
    )
    unstated = (
        "the sheet releases one titer per strain-hour and no uncertainty of any kind, "
        "so n_samples, sample_unit and the two uncertainty fields are typed absences "
        "rather than a guessed replicate design"
    )
    return ProductTiterPhenotype(
        product=resolved_compound(name),
        titer=titer_g_per_l,
        titer_unit=ConcentrationUnit.g_per_l,
        n_samples=None,
        sample_unit=None,
        quantification_method=method,
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=unstated,
            )
            for field in (
                "titer_uncertainty",
                "titer_uncertainty_type",
                "n_samples",
                "sample_unit",
            )
        ]
        + [
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no yield on substrate and no volumetric productivity is released "
                "for any strain at any hour",
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
# Build accounting
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One reason a released value is not a record (or not a key), with its items."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    scope: str
    description: str
    n_items: int
    items: list[str] = []


class BuildAccounting(BaseModel):
    """What one build read, what it kept and what it declined."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    source_rows: int
    kept_records: int
    dropped_records: int
    distinct_targets: int
    quotes_checked: int
    rules: list[DropRule] = []
    notes: list[str] = []

    def check(self) -> None:
        """Kept plus dropped must be the rows the build read."""
        if self.kept_records + self.dropped_records != self.source_rows:
            raise RuntimeError(
                f"{self.kept_records} kept + {self.dropped_records} dropped is not the "
                f"{self.source_rows} source rows"
            )


def _write_accounting(accounting: BuildAccounting, preprocess_dir: str) -> None:
    """Write ``build_accounting.json`` beside the built store."""
    accounting.check()
    os.makedirs(preprocess_dir, exist_ok=True)
    with open(osp.join(preprocess_dir, "build_accounting.json"), "w") as handle:
        handle.write(accounting.model_dump_json(indent=2))


def _link_mirror_files(raw_dir: str, pins: Iterable[tuple[str, str, str]]) -> None:
    """Hard-link each pinned mirror file into ``raw/`` after verifying its sha256."""
    os.makedirs(raw_dir, exist_ok=True)
    root = raw_mirror_dir()
    for relpath, name, digest in pins:
        link_verified(root / relpath, osp.join(raw_dir, name), digest)


class BuildContext(BaseModel):
    """What every dataset class needs before it writes a record."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    table: TableS1
    parts: dict[str, list[PathwayPart]]
    locus_of: dict[str, str]
    quotes_checked: int


def build_context(raw_dir: str, genome: EcoliK12Genome, *, label: str) -> BuildContext:
    """Read Table S1, parse every strain's pathway genes and resolve the native ones."""
    quotes = verify_quotes()
    table = read_table_s1(osp.join(raw_dir, SI_OCR_FILE))
    parts = strain_parts(table)
    native = sorted(
        {
            part.gene.gene_name
            for strain_parts_ in parts.values()
            for part in strain_parts_
            if not part.gene.is_heterologous
        }
    )
    stored, report = reconcile_locus_tags(genome, pd.Series(native), label=label)
    report.require_resolved(1.0)
    if report.outside_namespace:
        raise RuntimeError(
            f"{label}: tags outside {NAMESPACE}: {report.outside_namespace}"
        )
    return BuildContext(
        table=table,
        parts=parts,
        locus_of=dict(zip(native, stored)),
        quotes_checked=quotes,
    )


# --------------------------------------------------------------------------- #
# The metabolite arms
# --------------------------------------------------------------------------- #
class _MetaboliteBrunk2016Dataset(ExperimentDataset):
    """Base of the two metabolite arms; ``COLUMN_SET`` picks the unit block read.

    The two blocks are separate datasets rather than one because they are two scales
    measured on two instruments: an LC-MS micromolar concentration and an HPLC g/L
    concentration. One dataset would carry two ``measurement_type`` values, which the
    metabolite gate refuses, and averaging across them would be meaningless.
    """

    COLUMN_SET: ClassVar[Literal["micromolar", "grams_per_litre"]]
    MEASUREMENT_TYPE: ClassVar[str]
    EXPECTED_RECORDS: ClassVar[int]
    REFERENCE_STRAIN: ClassVar[Literal["MG1655"]] = REFERENCE_STRAIN
    ecoli_genome: EcoliK12Genome | None

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
        """The metabolomics workbook and the OCR that carries Table S1."""
        return [METABOLOMICS_FILE, SI_OCR_FILE]

    def download(self) -> None:
        """Link the two pinned mirror files into ``raw/``."""
        _link_mirror_files(
            self.raw_dir,
            (
                (METABOLOMICS_REL, METABOLOMICS_FILE, METABOLOMICS_SHA256),
                (SI_OCR_REL, SI_OCR_FILE, SI_OCR_SHA256),
            ),
        )
        log.info("Brunk 2016 metabolomics workbook linked into %s", self.raw_dir)

    def _genome(self) -> EcoliK12Genome:
        """The injected MG1655 genome, or one opened from the genomes tier."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", REFERENCE_STRAIN)
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Build one record per released sample that measures this unit block."""
        verify_raw_files(
            self.raw_dir,
            {METABOLOMICS_FILE: METABOLOMICS_SHA256, SI_OCR_FILE: SI_OCR_SHA256},
        )
        workbook = osp.join(self.raw_dir, METABOLOMICS_FILE)
        context = build_context(self.raw_dir, self._genome(), label=self.name)
        columns = read_metabolite_columns(workbook)
        samples = read_metabolite_samples(workbook)
        block: tuple[str, ...] = getattr(columns, self.COLUMN_SET)
        wild_type = samples[samples["Strain"] == WILD_TYPE].set_index("Hour")

        def levels(row: Any) -> dict[str, float]:
            return {
                columns.cobra_id[name]: float(row[name])
                for name in block
                if pd.notna(row[name])
            }

        reference_genome = host_reference()
        pub = publication()
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        rows: list[dict[str, Any]] = []
        empty: list[str] = []
        unshared: list[str] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for _, sample in tqdm(
                samples.iterrows(), total=len(samples), desc=f"brunk2016-{self.name}"
            ):
                strain, hour = str(sample["Strain"]), float(sample["Hour"])
                measured = levels(sample)
                if not measured:
                    empty.append(f"{strain}@{hour:g}h")
                    continue
                control = levels(wild_type.loc[hour])
                shared = {k: v for k, v in control.items() if k in measured}
                if not shared:
                    unshared.append(f"{strain}@{hour:g}h")
                    continue
                culture = environment(strain, hour)
                experiment = BacterialMetaboliteExperiment(
                    dataset_name=self.name,
                    genotype=genotype(strain, context.parts[strain], context.locus_of),
                    environment=culture,
                    phenotype=metabolite_phenotype(measured, self.MEASUREMENT_TYPE),
                )
                reference = BacterialMetaboliteExperimentReference(
                    dataset_name=self.name,
                    genome_reference=reference_genome,
                    environment_reference=culture.model_copy(),
                    phenotype_reference=metabolite_phenotype(
                        shared, self.MEASUREMENT_TYPE
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                rows.append(
                    {
                        "strain": strain,
                        "hour": hour,
                        "n_metabolites": len(measured),
                        "n_shared_with_wild_type": len(shared),
                        "n_perturbations": len(context.parts[strain]),
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "samples.csv"), index=False
        )
        pd.DataFrame(
            {
                "column": list(columns.unitless),
                "cobra_id": [columns.cobra_id[c] for c in columns.unitless],
            }
        ).to_csv(osp.join(self.preprocess_dir, "unitless_columns.csv"), index=False)
        if idx != self.EXPECTED_RECORDS:
            raise RuntimeError(
                f"{self.name}: {idx} records, not the measured {self.EXPECTED_RECORDS}"
            )
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(samples),
                kept_records=idx,
                dropped_records=len(samples) - idx,
                distinct_targets=len(block),
                quotes_checked=context.quotes_checked,
                rules=[
                    DropRule(
                        rule="sample_measures_no_column_of_this_unit_block",
                        scope="sample",
                        description="no cell of this unit block is filled for that "
                        "strain-hour, so the record would carry no measurement",
                        n_items=len(empty),
                        items=empty,
                    ),
                    DropRule(
                        rule="sample_shares_no_metabolite_with_the_wild_type_of_its_hour",
                        scope="sample",
                        description="the reference is the DH1 sample of the SAME hour "
                        f"('{_Q_DIFFERENCE_PROFILE}'), so a sample that shares no "
                        "measured metabolite with it has no reference to carry",
                        n_items=len(unshared),
                        items=unshared,
                    ),
                    DropRule(
                        rule="column_header_states_no_unit",
                        scope="metabolite_column",
                        description="the 20 amino acids, Cystine and the five DXP "
                        "intermediates carry no unit in their header and none anywhere "
                        "in the mirror, so they are not stored on a micromolar scale",
                        n_items=len(columns.unitless),
                        items=list(columns.unitless),
                    ),
                    DropRule(
                        rule="column_is_served_by_a_sibling_dataset",
                        scope="metabolite_column",
                        description="the other unit block and the three fuel columns "
                        "are records of the sibling Brunk 2016 datasets",
                        n_items=len(columns.product)
                        + len(
                            columns.grams_per_litre
                            if self.COLUMN_SET == "micromolar"
                            else columns.micromolar
                        ),
                        items=sorted(
                            set(columns.product)
                            | set(
                                columns.grams_per_litre
                                if self.COLUMN_SET == "micromolar"
                                else columns.micromolar
                            )
                        ),
                    ),
                ],
                notes=[
                    "one record per released strain-hour sample; the reference is the "
                    "DH1 sample of the SAME hour, restricted to the metabolites that "
                    "record also measured, so a DH1 record is its own reference",
                    "keys are the release's own COBRA (iJO1366) metabolite ids, from "
                    "the workbook's identifier sheet",
                    str(UNCERTAINTY_NOT_RELEASED.note),
                ],
            ),
            self.preprocess_dir,
        )
        log.info("Brunk2016 %s: %d records over %d targets", self.name, idx, len(block))

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


@register_dataset
class MetabolomeBrunk2016Dataset(_MetaboliteBrunk2016Dataset):
    """The 51 LC-MS columns the workbook labels ``(uM)``, one record per sample."""

    COLUMN_SET: ClassVar[Literal["micromolar", "grams_per_litre"]] = "micromolar"
    MEASUREMENT_TYPE: ClassVar[str] = MEASUREMENT_TYPE_METABOLITE
    EXPECTED_RECORDS: ClassVar[int] = EXPECTED_METABOLOME_RECORDS

    def __init__(
        self,
        root: str = "data/torchcell/metabolome_brunk2016",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the MG1655 genome resolves the four native pathway genes."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)


@register_dataset
class ExometaboliteBrunk2016Dataset(_MetaboliteBrunk2016Dataset):
    """The six HPLC columns the workbook labels ``g/L``: glucose and five acids."""

    COLUMN_SET: ClassVar[Literal["micromolar", "grams_per_litre"]] = "grams_per_litre"
    MEASUREMENT_TYPE: ClassVar[str] = MEASUREMENT_TYPE_EXOMETABOLITE
    EXPECTED_RECORDS: ClassVar[int] = EXPECTED_EXOMETABOLITE_RECORDS

    def __init__(
        self,
        root: str = "data/torchcell/exometabolite_brunk2016",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the MG1655 genome resolves the four native pathway genes."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)


# --------------------------------------------------------------------------- #
# The proteome arm
# --------------------------------------------------------------------------- #
@register_dataset
class ProteomeBrunk2016Dataset(ExperimentDataset):
    """Brunk 2016 SRM peak areas: nine strains at nine hours, host proteins only."""

    REFERENCE_STRAIN: ClassVar[Literal["MG1655"]] = REFERENCE_STRAIN
    #: Host proteins whose locus the mirror states: 44 of 68, measured 2026-10-10. The
    #: floor is that fraction exactly, so a mapping that silently shrinks stops a build.
    MIN_RESOLVED_FRACTION: ClassVar[float] = (
        EXPECTED_PROTEIN_KEYS / EXPECTED_HOST_PROTEINS
    )

    def __init__(
        self,
        root: str = "data/torchcell/proteome_brunk2016",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the MG1655 genome resolves the released gene names to loci."""
        self.ecoli_genome = ecoli_genome
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
        """The proteomics workbook and the OCR that carries Table S1."""
        return [PROTEOMICS_FILE, SI_OCR_FILE]

    def download(self) -> None:
        """Link the two pinned mirror files into ``raw/``."""
        _link_mirror_files(
            self.raw_dir,
            (
                (PROTEOMICS_REL, PROTEOMICS_FILE, PROTEOMICS_SHA256),
                (SI_OCR_REL, SI_OCR_FILE, SI_OCR_SHA256),
            ),
        )
        log.info("Brunk 2016 proteomics workbook linked into %s", self.raw_dir)

    def _genome(self) -> EcoliK12Genome:
        """The injected MG1655 genome, or one opened from the genomes tier."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", REFERENCE_STRAIN)
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Build one record per released sample that has a stated sampling hour."""
        verify_raw_files(
            self.raw_dir,
            {PROTEOMICS_FILE: PROTEOMICS_SHA256, SI_OCR_FILE: SI_OCR_SHA256},
        )
        workbook = osp.join(self.raw_dir, PROTEOMICS_FILE)
        genome = self._genome()
        context = build_context(self.raw_dir, genome, label=self.name)
        organisms_checked = check_pathway_organisms(workbook)
        keys = read_protein_keys(workbook)
        host = [key for key in keys if key.organism == HOST_SPECIES]
        if len(host) != EXPECTED_HOST_PROTEINS:
            raise RuntimeError(
                f"{len(host)} host proteins, not the measured {EXPECTED_HOST_PROTEINS}"
            )
        mapped = [key for key in host if key.gene_name is not None]
        names = [str(key.gene_name) for key in mapped]
        stored, report = reconcile_locus_tags(
            genome, pd.Series(names), label=f"{self.name}-proteins"
        )
        report.require_resolved(1.0)
        if report.outside_namespace:
            raise RuntimeError(
                f"{self.name}: protein loci outside {NAMESPACE}: "
                f"{report.outside_namespace}"
            )
        locus_of = {key.protein: tag for key, tag in zip(mapped, stored, strict=True)}
        if len(set(locus_of.values())) != EXPECTED_PROTEIN_KEYS:
            raise RuntimeError(
                f"{len(set(locus_of.values()))} distinct protein keys, not the "
                f"measured {EXPECTED_PROTEIN_KEYS}"
            )
        if len(mapped) / len(host) < self.MIN_RESOLVED_FRACTION:
            raise RuntimeError(
                f"{len(mapped)}/{len(host)} host proteins keyed to a locus, below the "
                f"{self.MIN_RESOLVED_FRACTION:.3f} floor"
            )

        areas = read_protein_areas(workbook)
        reference_genome = host_reference()
        pub = publication()
        hours = sorted(
            {float(h) for h in areas["Hour"].unique() if h != UNEXPLAINED_HOUR_LABEL}
        )

        def profile(strain: str, hour: str) -> dict[str, float]:
            block = areas[
                (areas["Strain"] == strain)
                & (areas["Hour"] == hour)
                & (areas["Organism"] == HOST_SPECIES)
            ]
            return {
                locus_of[str(row["Protein"])]: float(row["ProteinArea"])
                for _, row in block.iterrows()
                if str(row["Protein"]) in locus_of
            }

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        rows: list[dict[str, Any]] = []
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for strain in tqdm(STRAINS, desc="brunk2016-proteome"):
                for hour in hours:
                    label = f"{hour:g}"
                    measured = profile(strain, label)
                    if not measured:
                        raise RuntimeError(
                            f"{self.name}: {strain} at {label} h measures no keyed "
                            "protein, which the released grid says cannot happen"
                        )
                    control = profile(WILD_TYPE, label)
                    shared = {k: v for k, v in control.items() if k in measured}
                    if set(shared) != set(measured):
                        raise RuntimeError(
                            f"{self.name}: {strain} at {label} h measures proteins the "
                            f"{WILD_TYPE} sample of that hour does not"
                        )
                    culture = environment(strain, hour)
                    experiment = BacterialProteinAbundanceExperiment(
                        dataset_name=self.name,
                        genotype=genotype(
                            strain, context.parts[strain], context.locus_of
                        ),
                        environment=culture,
                        phenotype=protein_phenotype(measured),
                    )
                    reference = BacterialProteinAbundanceExperimentReference(
                        dataset_name=self.name,
                        genome_reference=reference_genome,
                        environment_reference=culture.model_copy(),
                        phenotype_reference=protein_phenotype(shared),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(experiment, reference, pub, itxn),
                    )
                    rows.append(
                        {
                            "strain": strain,
                            "hour": hour,
                            "n_proteins": len(measured),
                            "n_perturbations": len(context.parts[strain]),
                        }
                    )
                    idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "samples.csv"), index=False
        )
        pd.DataFrame([key.model_dump() for key in keys]).to_csv(
            osp.join(self.preprocess_dir, "protein_keys.csv"), index=False
        )
        if idx != EXPECTED_PROTEOME_RECORDS:
            raise RuntimeError(
                f"{self.name}: {idx} records, not the measured "
                f"{EXPECTED_PROTEOME_RECORDS}"
            )
        released_samples = int(areas.groupby(["Strain", "Hour"]).ngroups)
        refused = [key for key in host if key.gene_name is None]
        non_host = [key for key in keys if key.organism != HOST_SPECIES]
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=released_samples,
                kept_records=idx,
                dropped_records=released_samples - idx,
                distinct_targets=len(set(locus_of.values())),
                quotes_checked=context.quotes_checked,
                rules=[
                    DropRule(
                        rule="sample_hour_label_is_not_a_stated_time",
                        scope="sample",
                        description=f"the eight {UNEXPLAINED_HOUR_LABEL!r} samples. No "
                        "mirrored byte says what the C is, so they get no "
                        "duration_hours and are not filed beside the 72 h sample",
                        n_items=released_samples - idx,
                        items=[
                            f"{strain}@{UNEXPLAINED_HOUR_LABEL}"
                            for strain in STRAINS
                            if strain != WILD_TYPE
                        ],
                    ),
                    DropRule(
                        rule="protein_key_is_not_resolvable_to_one_host_locus",
                        scope="protein",
                        description=REFUSAL_MULTI_GENE_GPR,
                        n_items=len(refused),
                        items=[key.protein for key in refused],
                    ),
                    DropRule(
                        rule="protein_is_not_a_host_protein",
                        scope="protein",
                        description="the heterologous pathway enzymes and the "
                        "AmpR / Cam / BSA normalization standards are not loci of the "
                        "host assembly; the pathway ones are on every record's genotype",
                        n_items=len(non_host),
                        items=[key.protein for key in non_host],
                    ),
                ],
                notes=[
                    "one record per released strain-hour sample; the reference is the "
                    f"{WILD_TYPE} sample of the SAME hour, so a {WILD_TYPE} record is "
                    "its own reference",
                    f"{len(mapped)} of {len(host)} host proteins keyed to a locus: "
                    f"{sum(1 for k in mapped if k.route == ROUTE_UNIPROT)} through the "
                    "UniProt gene name of the triplicate sheet and "
                    f"{sum(1 for k in mapped if k.route == ROUTE_SINGLE_GENE_GPR)} "
                    "through a single-gene GPR",
                    f"{organisms_checked} pathway-gene organisms were re-read from the "
                    "workbook's own Organism column before a genotype was built",
                    str(UNCERTAINTY_NOT_RELEASED.note),
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Brunk2016 proteome: %d records over %d loci",
            idx,
            len(set(locus_of.values())),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# The titer arm
# --------------------------------------------------------------------------- #
@register_dataset
class BiofuelTiterBrunk2016Dataset(ExperimentDataset):
    """Brunk 2016 isopentenol, limonene and bisabolene titers over the time course."""

    REFERENCE_STRAIN: ClassVar[Literal["MG1655"]] = REFERENCE_STRAIN

    def __init__(
        self,
        root: str = "data/torchcell/biofuel_titer_brunk2016",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the MG1655 genome resolves the four native pathway genes."""
        self.ecoli_genome = ecoli_genome
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
        """The metabolomics workbook and the OCR that carries Table S1."""
        return [METABOLOMICS_FILE, SI_OCR_FILE]

    def download(self) -> None:
        """Link the two pinned mirror files into ``raw/``."""
        _link_mirror_files(
            self.raw_dir,
            (
                (METABOLOMICS_REL, METABOLOMICS_FILE, METABOLOMICS_SHA256),
                (SI_OCR_REL, SI_OCR_FILE, SI_OCR_SHA256),
            ),
        )
        log.info("Brunk 2016 titer source linked into %s", self.raw_dir)

    def _genome(self) -> EcoliK12Genome:
        """The injected MG1655 genome, or one opened from the genomes tier."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", REFERENCE_STRAIN)
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Build one titer record per released strain-hour fuel measurement."""
        verify_raw_files(
            self.raw_dir,
            {METABOLOMICS_FILE: METABOLOMICS_SHA256, SI_OCR_FILE: SI_OCR_SHA256},
        )
        workbook = osp.join(self.raw_dir, METABOLOMICS_FILE)
        context = build_context(self.raw_dir, self._genome(), label=self.name)
        samples = read_metabolite_samples(workbook)
        baselines = {
            (strain, float(row["Hour"])): float(row[TITER_COLUMN[product]])
            for product, strain in BASELINE_OF_PRODUCT.items()
            for _, row in samples[samples["Strain"] == strain].iterrows()
            if pd.notna(row[TITER_COLUMN[product]])
        }
        reference_genome = host_reference()
        pub = publication()
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        rows: list[dict[str, Any]] = []
        candidates = 0
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for _, sample in tqdm(
                samples.iterrows(), total=len(samples), desc="brunk2016-titer"
            ):
                strain, hour = str(sample["Strain"]), float(sample["Hour"])
                product = PRODUCT_OF_STRAIN.get(strain)
                if product is None:
                    continue
                value = sample[TITER_COLUMN[product]]
                if pd.isna(value):
                    continue
                candidates += 1
                baseline_strain = BASELINE_OF_PRODUCT[product]
                baseline = baselines[(baseline_strain, hour)]
                culture = environment(strain, hour)
                experiment = ProductTiterExperiment(
                    dataset_name=self.name,
                    genotype=genotype(strain, context.parts[strain], context.locus_of),
                    environment=culture,
                    phenotype=titer_phenotype(product, float(value)),
                )
                reference = ProductTiterExperimentReference(
                    dataset_name=self.name,
                    genome_reference=reference_genome,
                    environment_reference=culture.model_copy(),
                    phenotype_reference=titer_phenotype(product, baseline),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                rows.append(
                    {
                        "strain": strain,
                        "hour": hour,
                        "product": product,
                        "titer_g_per_l": float(value),
                        "baseline_strain": baseline_strain,
                        "baseline_titer_g_per_l": baseline,
                        "n_perturbations": len(context.parts[strain]),
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "titer_rows.csv"), index=False
        )
        if idx != EXPECTED_TITER_RECORDS:
            raise RuntimeError(
                f"{self.name}: {idx} records, not the measured {EXPECTED_TITER_RECORDS}"
            )
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=candidates,
                kept_records=idx,
                dropped_records=candidates - idx,
                distinct_targets=len(TITER_COLUMN),
                quotes_checked=context.quotes_checked,
                rules=[
                    DropRule(
                        rule="sample_has_no_fuel_measurement",
                        scope="sample",
                        description="the wild type makes no fuel and the engineered "
                        "strains' fuel columns are filled at nine of the fourteen "
                        "sampling hours; a blank cell is an absent measurement",
                        n_items=len(samples) - candidates,
                        items=[],
                    )
                ],
                notes=[
                    "one record per released fuel measurement; the reference is the "
                    "non-optimized variant of the SAME product at the SAME hour "
                    f"({BASELINE_OF_PRODUCT}), so I1, L1 and B1 records are their own "
                    "reference",
                    "the hour-0 titers are released zeros, which are measurements and "
                    "are stored as such",
                    "isopentenol is resolved through the compound table's "
                    f"{ISOPENTENOL_SYNONYM!r} synonym, so these titers join the same "
                    "isoprenol entity Foo 2014 and the P. putida rows use; limonene "
                    "and bisabolene resolve to name-only compounds with typed gaps",
                    str(UNCERTAINTY_NOT_RELEASED.note),
                ],
            ),
            self.preprocess_dir,
        )
        log.info("Brunk2016 titer: %d records over %d products", idx, len(TITER_COLUMN))

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification, L0 to L4
# --------------------------------------------------------------------------- #
Family = Literal["metabolome", "exometabolite", "proteome", "titer"]

#: ``family -> (dataset class, dev-tree root)``, the four stores this module builds.
FAMILY_BUILDS: dict[Family, tuple[type[ExperimentDataset], str]] = {
    "metabolome": (MetabolomeBrunk2016Dataset, "data/torchcell/metabolome_brunk2016"),
    "exometabolite": (
        ExometaboliteBrunk2016Dataset,
        "data/torchcell/exometabolite_brunk2016",
    ),
    "proteome": (ProteomeBrunk2016Dataset, "data/torchcell/proteome_brunk2016"),
    "titer": (BiofuelTiterBrunk2016Dataset, "data/torchcell/biofuel_titer_brunk2016"),
}
EXPECTED_BY_FAMILY: dict[Family, int] = {
    "metabolome": EXPECTED_METABOLOME_RECORDS,
    "exometabolite": EXPECTED_EXOMETABOLITE_RECORDS,
    "proteome": EXPECTED_PROTEOME_RECORDS,
    "titer": EXPECTED_TITER_RECORDS,
}


def _provenance(relpath: str, digest: str, method: str, page: str) -> Provenance:
    """One family's provenance line, pinned to the mirror file it was read from."""
    return Provenance(
        source_uri=relpath,
        citation_key=CITATION_KEY,
        sha256=digest,
        method=method,
        page=page,
        retrieved=RETRIEVED_AT,
    )


def _strain_of(record: Mapping[str, Any]) -> str:
    """A record's strain, read off the construct names its genotype carries.

    The wild type has no perturbation, so it is the one record shape with no construct,
    and a strain is otherwise identified by the plasmid set its perturbations name.
    """
    constructs = sorted(
        {
            str(p["construct_name"])
            for p in record["experiment"]["genotype"]["perturbations"]
        }
    )
    return WILD_TYPE if not constructs else "|".join(constructs)


def _l1_one_record_per_sample(
    records: Sequence[Mapping[str, Any]], expected: int
) -> LevelResult:
    """L1: every (strain, hour) pair appears once, and there are ``expected`` of them."""
    keys = [
        (_strain_of(record), record["experiment"]["environment"]["duration_hours"])
        for record in records
    ]
    duplicates = sorted({str(key) for key in keys if keys.count(key) > 1})
    return LevelResult(
        level=Level.L1,
        name="one_record_per_strain_hour_sample",
        passed=not duplicates and len(records) == expected,
        message=(
            f"{len(records)} records over {len(set(keys))} strain-hour samples; "
            f"{len(duplicates)} repeated, {expected} expected"
        ),
        details={"n_records": len(records), "duplicated": duplicates[:20]},
    )


def _l3_hour_is_the_sampling_time(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L3: every record carries its own sampling hour inside the released window."""
    hours = [
        record["experiment"]["environment"]["duration_hours"] for record in records
    ]
    bad = [h for h in hours if h is None or not 0.0 <= float(h) <= 72.0]
    return LevelResult(
        level=Level.L3,
        name="duration_hours_is_the_released_sampling_time",
        passed=not bad,
        message=(
            f"{len(hours)} records carry a sampling hour in the released window "
            f"('{SAMPLING_WINDOW.value}'); {len(bad)} do not"
        ),
        details={
            "hours": sorted({float(h) for h in hours if h is not None}),
            "bad": bad[:20],
        },
    )


def _l3_reference_is_the_wild_type_of_the_same_hour(
    records: Sequence[Mapping[str, Any]], label_key: str
) -> LevelResult:
    """L3: a record's reference shares its environment and is a subset of its keys."""
    bad: list[str] = []
    for record in records:
        experiment = record["experiment"]
        reference = record["reference"]
        measured = set(experiment["phenotype"][label_key])
        control = set(reference["phenotype_reference"][label_key])
        same_hour = (
            experiment["environment"]["duration_hours"]
            == reference["environment_reference"]["duration_hours"]
        )
        if not same_hour or not control or not control <= measured:
            bad.append(
                f"{_strain_of(record)}@{experiment['environment']['duration_hours']}"
            )
    return LevelResult(
        level=Level.L3,
        name="reference_is_the_same_hour_wild_type_profile",
        passed=not bad,
        message=(
            f"{len(records)} references share their record's hour and measure a "
            f"subset of its keys; {len(bad)} do not"
        ),
        details={"quote": _Q_DIFFERENCE_PROFILE, "bad": bad[:20]},
    )


def _l4_against_the_workbook(
    records: Sequence[Mapping[str, Any]],
    *,
    workbook: Path,
    label_key: str,
    column_set: str | None,
) -> LevelResult:
    """L4: every stored value re-read from the workbook, cell by cell.

    The build read the sheet through :func:`read_metabolite_samples`; this row reads it
    again from the pinned bytes and joins on (strain, hour, COBRA id), so a value that
    drifted between the sheet and the store fails here rather than being believed.
    """
    columns = read_metabolite_columns(workbook)
    samples = read_metabolite_samples(workbook)
    block: tuple[str, ...] = getattr(columns, column_set) if column_set else ()
    released: dict[tuple[str, float, str], float] = {}
    for _, row in samples.iterrows():
        for name in block:
            if pd.notna(row[name]):
                released[
                    (str(row["Strain"]), float(row["Hour"]), columns.cobra_id[name])
                ] = float(row[name])
    shared: list[tuple[str, float, float]] = []
    for record in records:
        hour = float(record["experiment"]["environment"]["duration_hours"])
        strain = _strain_from_released(record, samples, hour, released, label_key)
        for key, value in record["experiment"]["phenotype"][label_key].items():
            shared.append(
                (
                    f"{strain}@{hour:g}:{key}",
                    float(value),
                    released[(strain, hour, key)],
                )
            )
    return l4_cross_source(shared, tol=0.0).model_copy(
        update={"name": "stored_values_against_the_released_workbook"}
    )


def _strain_from_released(
    record: Mapping[str, Any],
    samples: pd.DataFrame,
    hour: float,
    released: Mapping[tuple[str, float, str], float],
    label_key: str,
) -> str:
    """The released strain whose cells at ``hour`` are exactly this record's values.

    The store keeps no strain label, so the join is on the numbers themselves: at one
    hour exactly one released strain carries this record's whole key-value block.
    """
    values = record["experiment"]["phenotype"][label_key]
    matches = [
        strain
        for strain in STRAINS
        if all(
            released.get((strain, hour, key)) == float(value)
            for key, value in values.items()
        )
        and any(released.get((strain, hour, key)) is not None for key in values)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"a record at {hour} h matches {matches} released strains, not one"
        )
    return matches[0]


def metabolite_report(
    records: Sequence[Mapping[str, Any]], dataset_root: str, family: Family
) -> VerificationReport:
    """One metabolite arm's L0-L4 report over already-loaded records."""
    from torchcell.verification.metabolite import verify_metabolite_dataset

    column_set = "micromolar" if family == "metabolome" else "grams_per_litre"
    measurement = (
        MEASUREMENT_TYPE_METABOLITE
        if family == "metabolome"
        else MEASUREMENT_TYPE_EXOMETABOLITE
    )
    quantification = (
        QUANTIFICATION_METABOLITE
        if family == "metabolome"
        else QUANTIFICATION_EXOMETABOLITE
    )
    report = verify_metabolite_dataset(
        [dict(record) for record in records],
        dataset_name=f"{family}_brunk2016",
        provenance=_provenance(
            METABOLOMICS_REL,
            METABOLOMICS_SHA256,
            f"{quantification.value}; one number per strain-hour sample, stored under "
            f"the release's own COBRA ids with measurement_type {measurement!r}; "
            f"reference = the {WILD_TYPE} sample of the same hour",
            f"{SHEET_METABOLITES} ({column_set} columns)",
        ),
        expected_count=EXPECTED_BY_FAMILY[family],
        reference_centered=False,
        environment_keyed=True,
    )
    workbook = Path(dataset_root) / "raw" / METABOLOMICS_FILE
    report.add(_l1_one_record_per_sample(records, EXPECTED_BY_FAMILY[family]))
    report.add(_l3_hour_is_the_sampling_time(records))
    report.add(
        _l3_reference_is_the_wild_type_of_the_same_hour(records, "metabolite_level")
    )
    report.add(
        _l4_against_the_workbook(
            records,
            workbook=workbook,
            label_key="metabolite_level",
            column_set=column_set,
        )
    )
    return report


def proteome_report(
    records: Sequence[Mapping[str, Any]], dataset_root: str
) -> VerificationReport:
    """The proteome arm's L0-L4 report over already-loaded records."""
    from torchcell.verification.protein import verify_protein_dataset

    report = verify_protein_dataset(
        [dict(record) for record in records],
        dataset_name="proteome_brunk2016",
        provenance=_provenance(
            PROTEOMICS_REL,
            PROTEOMICS_SHA256,
            f"{QUANTIFICATION_PROTEIN.value}; the released ProteinArea of each host "
            f"protein the mirror keys to one locus ({EXPECTED_PROTEIN_KEYS} of "
            f"{EXPECTED_HOST_PROTEINS}), one number per strain-hour sample, "
            f"measurement_type {MEASUREMENT_TYPE_PROTEIN!r}; reference = the "
            f"{WILD_TYPE} sample of the same hour",
            SHEET_PROTEOMICS,
        ),
        expected_count=EXPECTED_PROTEOME_RECORDS,
        # Nine hours per strain, so every genotype appears nine times by design; the
        # (strain, hour) partition is what the L1 row below asserts.
        allow_duplicate_orfs=True,
    )
    report.add(_l1_one_record_per_sample(records, EXPECTED_PROTEOME_RECORDS))
    report.add(_l3_hour_is_the_sampling_time(records))
    report.add(
        _l3_reference_is_the_wild_type_of_the_same_hour(records, "protein_abundance")
    )
    report.add(_l4_protein_values_against_the_workbook(records, dataset_root))
    report.add(_l4_wild_type_carries_no_pathway_protein(dataset_root))
    return report


def _l4_protein_values_against_the_workbook(
    records: Sequence[Mapping[str, Any]], dataset_root: str
) -> LevelResult:
    """L4: every stored peak area re-read from the pinned proteomics workbook."""
    workbook = Path(dataset_root) / "raw" / PROTEOMICS_FILE
    areas = read_protein_areas(workbook)
    keys = {
        key.protein: key.gene_name
        for key in read_protein_keys(workbook)
        if key.gene_name is not None and key.organism == HOST_SPECIES
    }
    released: dict[tuple[str, float], dict[str, float]] = {}
    for _, row in areas.iterrows():
        if str(row["Hour"]) == UNEXPLAINED_HOUR_LABEL:
            continue
        protein = str(row["Protein"])
        if protein not in keys or str(row["Organism"]) != HOST_SPECIES:
            continue
        released.setdefault((str(row["Strain"]), float(row["Hour"])), {})[protein] = (
            float(row["ProteinArea"])
        )
    stored_totals = sorted(
        round(sum(record["experiment"]["phenotype"]["protein_abundance"].values()), 6)
        for record in records
    )
    released_totals = sorted(
        round(sum(block.values()), 6) for block in released.values()
    )
    shared = [
        (f"sample_total_{index}", stored, release)
        for index, (stored, release) in enumerate(
            zip(stored_totals, released_totals, strict=True)
        )
    ]
    return l4_cross_source(shared, tol=1e-6).model_copy(
        update={"name": "stored_peak_area_totals_against_the_released_workbook"}
    )


def _l4_wild_type_carries_no_pathway_protein(dataset_root: str) -> LevelResult:
    """L4: the Table S1 OCR repair, re-asserted against the proteomics workbook.

    Both OCR passes put the ``JPUB_002460 + JPUB_002466`` pair on the DH1 row and leave
    B2 empty; the repair moves it to B2 because the article calls DH1 the wild type.
    This row checks the repair in a DIFFERENT file: every protein that pair encodes and
    the workbook measures is higher in B2 than in DH1, by a factor this row reports. A
    DH1 that really carried those two plasmids could not be the lower of the two.

    The weaker claim "DH1 is the lowest strain for every pathway protein" is NOT made,
    because it is false in the released bytes: for a protein a strain does not carry,
    the measured area is background, and DH1's background is not always the smallest of
    the nine (measured: DH1's GPPS area, 26,219, is above I1's 2,896).
    """
    workbook = Path(dataset_root) / "raw" / PROTEOMICS_FILE
    raw = pd.read_excel(workbook, sheet_name=SHEET_PROTEOMICS)
    peaks = raw.groupby(["Protein", "Strain"])["ProteinArea"].max().unstack()
    table = read_table_s1(Path(dataset_root) / "raw" / SI_OCR_FILE)
    moved = table.composition[str(table.repaired_to)] if table.repaired_to else ()
    encoded = {
        part.gene.protein_name
        for plasmid_id in moved
        for part in parse_plasmid_genes(plasmid_id, table.plasmids[plasmid_id])
        if part.gene.protein_name is not None
    }
    measured = sorted(name for name in encoded if name in peaks.index)
    ratios = {
        name: float(cast("float", peaks.loc[name, str(table.repaired_to)]))
        / float(cast("float", peaks.loc[name, WILD_TYPE]))
        for name in measured
    }
    below = sorted(name for name, ratio in ratios.items() if ratio <= 1.0)
    return LevelResult(
        level=Level.L4,
        name="the_moved_plasmid_pairs_proteins_are_higher_in_the_repaired_strain",
        passed=bool(measured) and not below,
        message=(
            f"{len(measured)} of the {len(encoded)} proteins "
            f"{table.repaired_to} inherited from the OCR's {table.repaired_from} row "
            f"are measured; all are higher in {table.repaired_to} than in "
            f"{WILD_TYPE} (smallest ratio "
            f"{min(ratios.values()) if ratios else float('nan'):.1f}x), "
            f"{len(below)} are not"
        ),
        details={
            "repaired_from": table.repaired_from,
            "repaired_to": table.repaired_to,
            "ratios": ratios,
            "not_higher": below,
        },
    )


def titer_report(
    records: Sequence[Mapping[str, Any]], dataset_root: str
) -> VerificationReport:
    """The titer arm's L0-L4 report over already-loaded records."""
    ledger = pd.read_csv(osp.join(dataset_root, "preprocess", "titer_rows.csv"))
    experiments = [record["experiment"] for record in records]
    phenotypes = [experiment["phenotype"] for experiment in experiments]
    report = VerificationReport(
        dataset_name="biofuel_titer_brunk2016",
        provenance=_provenance(
            METABOLOMICS_REL,
            METABOLOMICS_SHA256,
            "isopentenol by GC-FID and limonene and bisabolene by GC-MS off the "
            "dodecane overlay, in g/L, one number per strain-hour; reference = the "
            "non-optimized variant of the same product at the same hour",
            f"{SHEET_METABOLITES} (the three fuel columns)",
        ),
    )
    report.add(l0_structural(experiments, ProductTiterExperiment.model_validate))
    report.add(l1_count(len(records), EXPECTED_TITER_RECORDS))
    report.add(l2_value_fidelity([p["titer"] for p in phenotypes], minimum=0.0))
    report.add(_l1_one_record_per_sample(records, EXPECTED_TITER_RECORDS))
    report.add(_l3_hour_is_the_sampling_time(records))
    report.add(
        l3_convention(
            "titer_unit_is_the_sheets_own_g_per_l",
            all(p["titer_unit"] == ConcentrationUnit.g_per_l.value for p in phenotypes),
            detail="the workbook's fuel columns are labelled g/L and the number is "
            "stored verbatim, with no unit arithmetic",
        )
    )
    report.add(
        l3_convention(
            "every_uncertainty_is_a_typed_gap_not_a_guess",
            all(
                p["titer_uncertainty"] is None
                and p["titer_uncertainty_type"] is None
                and p["n_samples"] is None
                and {"titer_uncertainty", "n_samples"}
                <= {gap["field"] for gap in p["provenance_gaps"]}
                for p in phenotypes
            ),
            detail="the sheet releases one titer per strain-hour and no dispersion and "
            "no replicate count, so none is invented",
        )
    )
    report.add(
        l3_convention(
            "isopentenol_is_the_canonical_isoprenol_entity",
            all(
                p["product"]["inchikey"]
                == resolved_compound(ISOPENTENOL_SYNONYM).inchikey
                for p in phenotypes
                if p["quantification_method"] == str(QUANTIFICATION_ISOPENTENOL.value)
            ),
            detail="resolved through the compound table's "
            f"{ISOPENTENOL_SYNONYM!r} synonym, so these titers join every other "
            "isoprenol record",
        )
    )
    report.add(_l4_titer_against_the_workbook(ledger, dataset_root))
    return report


def _l4_titer_against_the_workbook(
    ledger: pd.DataFrame, dataset_root: str
) -> LevelResult:
    """L4: each stored titer re-read from the pinned workbook, by strain and hour."""
    samples = read_metabolite_samples(Path(dataset_root) / "raw" / METABOLOMICS_FILE)
    indexed = samples.set_index(["Strain", "Hour"])
    shared = [
        (
            f"{row.strain}@{row.hour:g}:{row['product']}",
            float(row.titer_g_per_l),
            float(indexed.loc[(row.strain, row.hour), TITER_COLUMN[row["product"]]]),
        )
        for _, row in ledger.iterrows()
    ]
    return l4_cross_source(shared, tol=0.0).model_copy(
        update={"name": "stored_titers_against_the_released_workbook"}
    )


def verify_build(
    dataset_root: str, data_root: str | None = None, *, family: Family
) -> VerificationReport:
    """Run one family's L0-L4 gate over a built tree and write its report."""
    from torchcell.verification.runners import load_records

    del data_root
    records = load_records(dataset_root)
    if family in ("metabolome", "exometabolite"):
        report = metabolite_report(records, dataset_root, family)
    elif family == "proteome":
        report = proteome_report(records, dataset_root)
    elif family == "titer":
        report = titer_report(records, dataset_root)
    else:
        raise RuntimeError(
            f"{family!r} is not one of the four families this module serves"
        )
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build every family and verify it, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    genome = bacterial_genome("ecoli", REFERENCE_STRAIN, data_root)
    for family, (cls, rel) in FAMILY_BUILDS.items():
        root = osp.join(data_root, rel)
        dataset = cls(root=root, ecoli_genome=genome)  # type: ignore[call-arg]
        print(f"{cls.__name__}: len = {len(dataset)}")
        dataset.close_lmdb()
        accounting = json.loads(
            Path(osp.join(root, "preprocess/build_accounting.json")).read_text()
        )
        print(json.dumps(accounting, indent=2))
        print(verify_build(root, data_root, family=family).summary())


if __name__ == "__main__":
    main()
