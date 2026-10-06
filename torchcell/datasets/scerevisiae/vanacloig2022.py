# torchcell/datasets/scerevisiae/vanacloig2022
# [[torchcell.datasets.scerevisiae.vanacloig2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/vanacloig2022
# Test file: tests/torchcell/datasets/scerevisiae/test_vanacloig2022.py
"""Vanacloig-Pedros 2022 comparative chemical-genomic screen (env x geno -> response).

Vanacloig-Pedros et al. 2022 (FEMS Yeast Research, doi:10.1093/femsyr/foac036) profiled
the '3DeltaAlpha' drug-sensitized barcoded yeast deletion library ANAEROBICALLY against
34 conditions (Fig 1B: 33 inhibitors, mostly plant-hydrolysate toxins, and DMSO), in
independent biological triplicate, alongside matched inhibitor-free controls on the same
plates. The readout is log2(inhibitor/control) barcode abundance, a per-deletion fitness
response.

READOUT PROVENANCE / RIGOR. The paper's published values are edgeR glmQLFit PAIRED
logFCs on TMM-normalized counts ("using TMM normalization and glmQLFit comparing paired
treatment to control samples"). GEO GSE186866 releases only the raw barcode-count matrix,
which IS scriptable and sha256-pinned; the per-compound logFC table (Dataset2_mclust_cdt)
and Table S1 sit behind academic.oup.com, which is not scriptable. This loader recomputes
the paper's normalized ratio from the canonical counts:

- TMM scaling factors (``tmm_factors``: edgeR 3.26.8 ``calcNormFactors(method="TMM")``
  with its defaults, ``logratioTrim=0.3``, ``sumTrim=0.05``, ``doWeighting=TRUE``,
  ``Acutoff=-1e10``, reference = the sample whose upper-quartile fraction is closest to
  the mean) computed PER CONDITION over its three replicate columns and the control
  columns it is paired with, as the paper's per-compound edgeR fit does. The library
  size each factor multiplies is the column total over EVERY released barcode, before
  any retention rule, so the stored value does not depend on this loader's gene policy.
- per gene and replicate ``log2((TMM-CPM_rep + 1) / (mean TMM-CPM of the paired
  controls + 1))``; the response is the mean of the three, the uncertainty their sample
  SD (SE = SD/sqrt(3)). The pseudocount of 1 CPM is a loader choice the paper does not
  state (``PSEUDOCOUNT_GAP``); cells whose control mean sits near it carry a large SD,
  which is the low-count flag (no separate field).

It is NOT the published edgeR logFC (no dispersion shrinkage, no prior count) and must
not be treated as such; nothing mirrored carries the published numbers.

CONTROL PAIRING. Each replicate is paired with the control columns of its OWN ``CG00n``
batch, because the paper's design is paired and its comparison is "to the paired SynBase
medium control"; MMS is the one served condition the paper analyzed UNPAIRED, so its
control is the mean of all 16 control columns (its ``units`` string records that). DMSO
is served as a condition (1% v/v) paired the same way; which inhibitors were themselves
delivered in DMSO is in the unmirrored Table S1, so each inhibitor's ``solvent`` is a
typed gap.

STRAIN BACKGROUND (#500). The screened strains are the MATa meiotic progeny of the SGA
cross of query Y13206 (MATalpha pdr1::natMX pdr3::KlURA3 snq2::KlLEU2 can1::STE2pr-
Sp_his5 lyp1) to a MATa xxx::kanMX array (Piotrowski 2017, the library deferral). That
constant genome content rides on the reference's ``StrainReferenceGenome`` background;
each record's ``Genotype`` holds the ONE screened deletion.

RECORDS DROPPED (rule + count written to ``preprocess/dropped_records.json``): the 11
matrix tokens Fig 1B does not list (the paper never reports them); compounds with no
resolvable structure identifier; library rows whose ORF is not a barcoded ORF with
counts, is a locus the SGA selections fix in every strain, is not a current R64 gene, or
is the legacy spelling of an ORF already in the pool (those carry a typed
``ConstructedOrf`` in the ledger); and cells whose three replicate counts are ALL zero.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel
from scipy.stats import rankdata
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import (
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import SYNBASE
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    Concentration,
    ConcentrationUnit,
    ConstructedOrf,
    CultureEnvironment,
    CultureFormat,
    DoseBasis,
    EndpointRule,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MatingType,
    MeasurementType,
    PhysicalFactor,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainBackground,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    StrainReferenceGenome,
    Temperature,
    UncertaintyType,
    Zygosity,
)
from torchcell.datamodels.strain_background import (
    BRACHMANN_1998,
    pending_source_review,
    standard_allele,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.gene_name_reconcile import default_genome
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "vanacloig-pedrosComparativeChemicalGenomic2022"
PAPER_DOI = "10.1093/femsyr/foac036"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

DATA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186866/suppl/"
    "GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz"
)
DATA_FILENAME = "GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz"
DATA_SHA256 = "e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a"
DATA_REL = f"data/{DATA_FILENAME}"
DATA_RETRIEVED_AT = "2026-09-13"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "0b5d938b54b8424fa08203a4357bc8f7c7dfae3fbe1a6d07d422848b92f37ba3"

#: Fig 1B (the per-condition count of significant genes) as MinerU extracted it from the
#: publisher PDF; the 34 bar labels are read from this image, not from OCR text.
FIG_1B_IMAGE = (
    "images/8355ec6ee4cec3600bdfe3bb9305a3a5769788eb9bea45c0ec4e79453f8ba8b9.jpg"
)
FIG_1B_IMAGE_SHA256 = "f00ee21b185db86821ad11ba351f84841b38c78b5c0db22a9361ebc88b9c64cf"

PIOTROWSKI_KEY = "piotrowskiFunctionalAnnotationChemical2017"
PIOTROWSKI_SHA256 = "9314a0dd932c1b20b4dc4297de4ce9452b09f314bf100f05f2fe718fb3415fc1"
OHNUKI_KEY = "ohnukiHighthroughputPlatformYeast2022"
OHNUKI_SHA256 = "de2cad9b33c5e0f7e9ce7b7d56feb17a10dbb83de34f65330d6e35846b757ee1"

#: A float64 count / library-size array.
FloatArray = npt.NDArray[np.float64]

CPM_PRIOR = 1.0  # CPM pseudocount (PSEUDOCOUNT_GAP); all-zero cells are dropped

# edgeR 3.26.8 calcNormFactors(method="TMM") defaults, the version the paper names.
TMM_LOGRATIO_TRIM = 0.3
TMM_SUM_TRIM = 0.05
TMM_A_CUTOFF = -1e10
TMM_REFERENCE_QUANTILE = 0.75


def _paper(
    value: Any,
    quote: str,
    *,
    note: str | None = None,
    page: str = "Methods, 'Strains and growth conditions' / 'Chemical genomic experiment'",
) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned paper OCR mirror."""
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


def _piotrowski(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to Piotrowski 2017, the paper Vanacloig defers library methods to."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=PIOTROWSKI_KEY,
            sha256=PIOTROWSKI_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=page,
        ),
    )


def _ohnuki(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to Ohnuki 2022, which spells out the Y13206 query genotype."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=OHNUKI_KEY,
            sha256=OHNUKI_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page="Methods, strains (paper.md line 128)",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced environment + phenotype constants (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_GROWTH_QUOTE = (
    "Inoculated plates were grown in an anaerobic chamber (Coy Laboratory Products, "
    "Inc.), containing $1 \\% - 2 \\%$ $\\mathrm { H } _ { 2 }$ , $4 \\% - 5 \\%$ "
    "$\\mathrm { C O } _ { 2 }$ , and $9 0 \\% { - } 9 5 \\% \\mathrm { N } _ { 2 }$ at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ for $2 4 \\mathrm { { h } }$ , and then "
    "transferred into the identical fresh medium at $\\mathrm { O D } _ { 6 0 0 } = 0 . 1$ "
    "for another $2 4 \\mathrm { { h } }$ without shaking."
)

TEMPERATURE_C = _paper(30.0, _GROWTH_QUOTE)
AEROBICITY = _paper("anaerobic", _GROWTH_QUOTE)
DURATION_HOURS = _paper(
    48.0,
    _GROWTH_QUOTE,
    note="two consecutive 24 h anaerobic growth periods in identical fresh medium",
)
DURATION_GENERATIONS = _paper(
    6.5,
    "All cell cultures reached between 6.5 to 10 total cell doublings within the two "
    "$2 4 \\mathrm { ~ h ~ }$ growth periods.",
    note="the primary releases a RANGE and no per-condition value, and no companion "
    "statistic permits a back-solve, so the CLAUDE.md rule takes the conservative "
    "lower end (6.5 doublings) rather than the optimistic one",
)
MEDIUM_PH = _paper(
    5.0,
    "ammonium sulfate was replaced with ${ \\mathrm { ~ 1 ~ g / L ~ } }$ monosodium "
    "glutamate (MSG, Fisher Scientific) and adjusted to $\\mathrm { p H } ~ 5 . 0$ with "
    "HCl.",
    note="a medium-intrinsic pH; Media has no ph field (it sits in 36 served dataset "
    "closures, so adding one is a full rebuild), so it rides as a typed physical factor",
)
N_REPLICATES = _paper(
    3,
    "Growth of the yeast deletion library in all inhibitory and control conditions were "
    "performed in independent biological triplicate.",
)
ASSAY = _paper(
    AssayType.pooled_competitive_growth_barcode,
    "Barcode read counts were calculated from up-tag reads using custom python scripts.",
    note="a pooled library grown competitively and read out by amplifying each strain's "
    "UPTAG barcode",
)
LIBRARY_COLLECTION = _paper(
    "3DeltaAlpha drug-sensitive yeast deletion collection of 4309 mutants",
    "Saccharomyces cerevisiae strains used in the chemical genomics study belong to the "
    "‘3DeltaAlpha’ drug-sensitive yeast deletion collection of 4309 mutants",
)
BARCODE_IS_UPTAG = _paper(
    "uptag",
    "The library includes 4309 strains in which a non-essential gene is replaced with a "
    "unique DNA sequence (barcode) flanked by common sequences for barcode amplification.",
    note="the matrix's gene column is '<ORF>_<barcode>' and the counts are up-tag reads",
)
IC30_BASIS = _paper(
    DoseBasis.IC30,
    "Concentrations for each inhibitor used for the chemical genomics experiment with "
    "the yeast deletion library were determined based on estimated inhibition of "
    "${ \\sim } 3 0 \\%$ of growth $\\left( \\mathrm { I C } _ { 3 0 } \\right)$ in "
    "SynBase medium with the inhibitor relative to growth in SynBase medium lacking the "
    "inhibitor (Table S1, Supporting Information).",
    note="the per-compound molar values live in Table S1, which academic.oup.com does "
    "not serve to a script and which is therefore not mirrored; Concentration.value "
    "stays None and the IC30 basis carries the dose provenance",
)
BENOMYL_MMS_DOSE = _paper(
    {"benomyl_ug_per_ml": 10.0, "mms_percent": 0.01},
    "Benomyl and MMS concentrations were used as previously published (Piotrowski et al. "
    "2017), $1 0 ~ \\mathrm { u g / m L }$ and $0 . 0 1 \\%$ , respectively.",
    note="both doses were taken from Piotrowski 2017 rather than set to an IC30, so "
    "their basis is 'fixed'. Piotrowski 2017 (mirrored) states benomyl as 34.4 uM "
    "(BENOMYL_MOLAR), which is the stored value. The MMS percent is stored as a basis "
    "only: neither paper writes v/v or w/v for this 0.01% (Piotrowski 2017 names no "
    "MMS dose), so the unit would be a guess",
)
BENOMYL_MOLAR = _piotrowski(
    34.4,
    "Cultures were then spiked with either $3 4 . 4 \\mu \\mathrm { M }$ benomyl, "
    "$2 5 ~ \\mathrm { n M }$ micafungin, or a $1 \\%$ DMSO control.",
    page="Online Methods, signal-detection optimization (paper.md line 218)",
    note="the deferral target of Vanacloig's '10 ug/mL as previously published'. The "
    "same 34.4 uM appears at paper.md lines 25 (Fig 1c, BENOMYL_MOLAR_FIG1C) and 208 "
    "('34.4 uM benomyl'). Consistency check, not a source: 10 ug/mL / 290.32 g/mol "
    "(benomyl, C14H18N4O3) = 34.44 uM",
)
BENOMYL_MOLAR_FIG1C = _piotrowski(
    34.4,
    "at a concentration of $3 4 . 4 ~ \\mu \\mathrm { M }$ , the microtubule-binding "
    "compound benomyl showed a specific chemical-genetic interaction with TUB3",
    page="Results, drug-sensitized background (paper.md line 25)",
)
UNPAIRED_COMPOUNDS_QUOTE = (
    "Gene deletions with specific fitness contributions were identified using linear "
    "models in edgeR version 3.26.8 (Robinson et al. 2010), using TMM normalization and "
    "glmQLFit comparing paired treatment to control samples, except with MMS and QUADRIS "
    "compounds, which were unpaired."
)
NORMALIZATION = _paper(
    "TMM",
    UNPAIRED_COMPOUNDS_QUOTE,
    page="Methods, 'Chemical genomic data processing and functional analysis'",
    note="the loader computes edgeR 3.26.8's TMM factors in numpy (tmm_factors) per "
    "condition over its replicates and paired controls; glmQLFit's dispersion "
    "shrinkage is not reproduced, so the stored value is the normalized ratio, not the "
    "published logFC",
)
PAIRED_CONTROL = _paper(
    "batch-matched",
    "All 24-well plates contained control samples with SynBase or SynBase $+ ~ 1 \\%$ "
    "DMSO lacking any inhibitor for paired analysis (see below).",
    note="each replicate is paired with the ControlN columns of its own CG00n batch; "
    "MMS keeps the pooled 16-column control because the paper analyzed it unpaired: "
    + UNPAIRED_COMPOUNDS_QUOTE,
)
PAIRED_SYNBASE_CONTROL = _paper(
    "SynBase",
    "Linear modeling of quantitative barcode sequence counts identified genes whose "
    "deletion produced reproducible fitness effects in the presence of each inhibitor "
    "compared to the paired SynBase medium control (see Methods).",
    page="Results (paper.md line 103)",
    note="the paired control is inhibitor-free SynBase, so every condition, DMSO "
    "included, is paired against the ControlN columns rather than against the DMSO "
    "columns",
)
VEHICLE_CONTROL = _paper(
    "DMSO",
    "Chemical compounds insoluble in water were dissolved in DMSO at 100X concentration "
    "so that the final concentration of DMSO in SynBase medium was $1 \\%$ $( \\mathrm "
    "{ v / v } )$ .",
    note="which compounds DMSO delivered is in the unmirrored Table S1, so no "
    "per-compound Solvent can be asserted; DMSO itself is served as a condition "
    "(DMSO_DOSE)",
)
DMSO_DOSE = _paper(
    1.0,
    VEHICLE_CONTROL.quote,
    note="DMSO's own condition columns are the vehicle at its final 1% v/v with no "
    "inhibitor; Fig 1B lists DMSO as one of the 34 analyzed conditions (FIG_1B_CONDITIONS)",
)
FIG_1B_CONDITIONS = _paper(
    34,
    "Each colored strain depicts a different gene deletion strain, before and after "
    "exposure to one of 34 different inhibitors.",
    page="Figure 1 caption (paper.md line 89); bar labels read from the Fig 1B image "
    + FIG_1B_IMAGE
    + " (sha256 "
    + FIG_1B_IMAGE_SHA256
    + ")",
    note="the 34 Fig 1B bar labels, in figure order: Benomyl, Acetamide, "
    "4-OH-Benzoic Acid, 5-HMF, Benzoic Acid, Sinapic Acid, 2,2'-Dipyridyl, Coumaric "
    "Acid, BMIM-Cl, 4-OH-Benzaldehyde, DMSO, Vanillin, Isobutanol, Furfural, Cinnamic "
    "Acid, Vanillic Acid, 4-OH-Acetophenone, MMS, Syringaldehyde, Methylglyoxal, Azelaic "
    "Acid, Coumaroyl Amide, Feruloyl Amide, Syringic Acid, MBO, Acetovanillone, EMIM-Cl, "
    "Ferulic Acid, Acetosyringone, NAO, 2,6-Dimethylpyrazine, GVL, CV, EtOH. The matrix "
    "token of each is FIG_1B_TOKENS",
)
READOUT = _paper(
    MeasurementType.log2_ratio,
    "Results were presented in heat map figures as the $\\log _ { 2 }$ of the normalized "
    "read counts for inhibitor/control ratio.",
)
PH_AGENT = _paper(
    "hydrochloric acid",
    "adjusted to $\\mathrm { p H } ~ 5 . 0$ with HCl.",
    note="the acid that REALIZES the pH factor, carried on the physical perturbation's "
    "`agent` slot so the medium's pH joins on a compound entity",
)
CULTURE_VESSEL = _paper(
    {"vessel": "24-well plates (Falcon)", "working_volume_ul": 1500.0},
    "the pooled yeast gene-knockout library was inoculated into $1 . 5 \\mathrm { m L }$ "
    "of SynBase or SynBase $+ ~ 1 \\%$ DMSO containing individual inhibitors at their "
    "defined $\\mathrm { I C } _ { 3 0 }$ concentration in 24-well plates (Falcon) at an "
    "$\\mathrm { O D } _ { 6 0 0 } = 0 . 1$ .",
    note="also the inoculum OD600 of 0.1",
)
STATIC_CULTURE = _paper(
    0.0,
    _GROWTH_QUOTE,
    note="'without shaking' -> shaking_rpm 0.0; both 24 h periods start at OD600 0.1 "
    "and end at a fixed time (EndpointRule.fixed_duration)",
)

# --- MBO: the paper defines the abbreviation twice, as two different compounds ----- #
MBO_ABBREVIATION = _paper(
    "2-methyl-3-butyn-2-ol",
    "MBO : 2-Methyl-3-butyn-2-ol",
    page="Abbreviations (paper.md line 31)",
    note="the glossary entry, NOT adopted (MBO_IDENTITY_RULE); PubChem CID 8258",
)
MBO_IDENTITY = _paper(
    "2-methyl-3-buten-2-ol",
    "biofuel endproducts (ethanol, isobutanol and 2-methyl-3-buten-2-ol (MBO))",
    page="Results (paper.md line 103)",
    note="the adopted identity, PubChem CID 8257 (InChIKey HNVRRHSXBLFLIG-UHFFFAOYSA-N), "
    "under MBO_IDENTITY_RULE; it disagrees with MBO_ABBREVIATION (line 31)",
)
MBO_IS_A_BIOFUEL = _paper(
    "biofuel",
    "the two other biofuels included in our screen, IBA and MBO",
    page="Results (paper.md line 131)",
    note="the second in-text use; it names MBO a biofuel product of the screen",
)
MBO_IDENTITY_RULE = (
    "When the paper defines one abbreviation as two different structures, the in-text "
    "definition that names the compound where the experiment uses it outranks the "
    "one-line Abbreviations glossary. MBO is defined in the Results as "
    "'2-methyl-3-buten-2-ol (MBO)' among the 'biofuel endproducts' (line 103) and is "
    "called one of 'the two other biofuels included in our screen' (line 131); the "
    "glossary's '2-Methyl-3-butyn-2-ol' (line 31) is the single contrary statement. "
    "Hypothesis (no mirrored source checks it): the glossary line is a typo, since "
    "2-methyl-3-buten-2-ol is the hemiterpene alcohol produced as a biofuel and the "
    "butyn alkynol is not. Table S1 (not mirrored) would settle it. The adjudication "
    "is recorded on the identity row's input line "
    "(compound_identity_inputs/vanacloig2022.txt)."
)

#: The one unmirrored artifact that would close this dataset's recoverable gaps: the OUP
#: supplement holding the per-compound IC30 molar values AND which compounds were
#: delivered in DMSO. academic.oup.com returns 403 to a script, so it is not mirrored.
TABLE_S1 = Provenance(
    source_uri="https://doi.org/10.1093/femsyr/foac036 (Table S1, Supporting Information)",
    citation_key=CITATION_KEY,
    method="publisher supplementary table; academic.oup.com is not scriptable",
    page="Table S1",
)

#: The protocol Vanacloig's chemical-genomic method defers to ("as previously described
#: (Piotrowski et al. 2015) with modifications"), not mirrored: it would state how the
#: pool was grown before inoculation.
PIOTROWSKI_2015 = Provenance(
    source_uri="Piotrowski et al. 2015 (cited by Vanacloig-Pedros 2022 Methods for the "
    "chemical genomic protocol)",
    method="not mirrored; the pre-inoculation culture of the pooled library",
)

#: Zhang et al. 2019 (Front Microbiol 10:2596), the SynH3- recipe, not mirrored.
ZHANG_2019 = Provenance(
    source_uri="Zhang et al. 2019, Front Microbiol 10:2596 (SynH3- recipe)",
    method="not mirrored; the amino acids and supplements SynH3- carries",
)

#: SGD's locus history (merges and reannotations), not mirrored.
SGD_ORF_HISTORY = Provenance(
    source_uri="https://www.yeastgenome.org (locus history of the source ORF)",
    method="not mirrored; SGD locus-history notes record ORF merges and reannotations",
)

PSEUDOCOUNT_GAP = ProvenanceGap(
    field="pseudocount",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=PAPER_MD,
        citation_key=CITATION_KEY,
        sha256=PAPER_MD_SHA256,
        page="Methods, 'Chemical genomic data processing and functional analysis'",
    ),
    note="the paper presents 'the log2 of the normalized read counts for "
    "inhibitor/control ratio' and states no pseudocount; the loader adds 1 CPM "
    "(CPM_PRIOR) to both sides, a loader choice, so a cell whose control mean is near "
    "1 CPM has a pseudocount-dominated denominator and a large replicate SD",
)


def _solvent_gap() -> ProvenanceGap:
    """The typed absence of a per-compound vehicle.

    The vehicle is the field that is ACTUALLY ``None`` on the perturbation: the primary
    says water-insoluble compounds went in at 1% v/v DMSO but names them only in Table
    S1, so for any one compound it is unknown whether a vehicle was used at all. The
    DOSE is not a second gap: ``concentration`` is never None (an IC30 or fixed basis is
    always known), and ``Concentration`` is not itself a gap carrier, so the missing
    molar value is carried by ``basis`` -- the mechanism the schema documents for exactly
    this case.
    """
    return ProvenanceGap(
        field="solvent",
        reason=ProvenanceGapReason.deferred_pending_source_review,
        resolve_with=TABLE_S1,
        note=VEHICLE_CONTROL.quote
        + " Which compounds that covers is in Table S1, which is not mirrored, so the "
        "vehicle of any one compound is unknown rather than absent. The vehicle's own "
        "effect is served as the DMSO condition (dimethyl sulfoxide, 1% v/v).",
    )


# --------------------------------------------------------------------------- #
# Strain background (#500): the SGA MATa progeny of Y13206 x the MATa kanMX array
# --------------------------------------------------------------------------- #
QUERY_STRAIN = _piotrowski(
    "Y13206",
    "The MATα pdr1Δ::natMX pdr3Δ::KI.URA3 snq2Δ::KI.LEU2 (y13206) query strain carried "
    "the can1Δ::STEpr-SP_his5 and lypΔ SGA reporters.",
    page="Online Methods, genome-wide drug-sensitive collection (paper.md line 204)",
    note="the OCR writes K. lactis as 'KI.' and the reporters as 'STEpr-SP_his5' and "
    "'lypΔ'; the allele names stored are the standard spellings pdr3Δ::KlURA3, "
    "snq2Δ::KlLEU2, can1Δ::STE2pr-Sp_his5, lyp1Δ",
)
MATA_PROGENY = _piotrowski(
    MatingType.a,
    "The resulting spores were transferred to synthetic media lacking histidine and "
    "containing canavanine and thialysine to select for the MATa meiotic progeny.",
    page="Online Methods, genome-wide drug-sensitive collection (paper.md line 204)",
    note="the screened strains are MATa haploid progeny; Vanacloig's 'MATα' "
    "(paper.md line 54) describes the query strain",
)
ARRAY_KANMX = _piotrowski(
    "kanMX",
    "The MATα query strain was crossed to an ordered array of MATa xxxΔ::kanMX deletion "
    "mutants",
    page="Online Methods, genome-wide drug-sensitive collection (paper.md line 204)",
    note="the cassette of each screened deletion; Piotrowski names no kanMX version "
    "and no array strain background",
)
TRIPLE_SELECTION = _piotrowski(
    "pdr1Δ pdr3Δ snq2Δ xxxΔ",
    "Finally, these cells were transferred to synthetic media lacking uracil and leucine "
    "and containing G418 and NAT to select for the desired pdr1Δ pdr3Δ snq2Δ xxxΔ "
    "mutants.",
    page="Online Methods, genome-wide drug-sensitive collection (paper.md line 204)",
)
Y13206_GENOTYPE = _ohnuki(
    "Y13206",
    "The drug-hypersensitive yeast strain Y13206 (3Δ; MATα snq2Δ:: KlLEU2 pdr3Δ:: "
    "KlURA3 pdr1Δ:: NATMX can1Δan11:: 2iSp_his5 lyp1Δ his3Δ1 leu2Δ0 ura3Δ0 met15Δ LYS2)",
    note="the QUERY's full genotype; the OCR garbles the can1 token",
)
Y8835_GENOTYPE = _ohnuki(
    "Y8835",
    "its parent strain Y8835 (MATα ura3Δ0:: natMX4 can 1Δ:: STE2pr-Sp_his5 lyp1Δ "
    "his3Δ1 leu2Δ0 met15Δ0 LYS2)",
    note="the query's parent; it writes the can1 reporter legibly",
)

LIBRARY_STRAIN = "3DeltaAlpha SGA MATa progeny (Y13206 x MATa xxxΔ::kanMX array)"
_BY_AUXOTROPHY_NOTE = (
    "Ohnuki 2022 (mirrored) states this allele for the QUERY lineage only (Y13206 "
    "'his3Δ1 leu2Δ0 ura3Δ0 met15Δ', parent Y8835 'ura3Δ0:: natMX4 ... met15Δ0'); a "
    "haploid SGA segregant carries an unselected allele only when the array parent also "
    "does, and no mirrored source states the array's genotype (the standard "
    "BY4741-derived YKO string is Brachmann 1998)"
)


def library_background() -> StrainBackground:
    """The constant genome content of every screened strain beyond R64.

    Sourced: MATa (selected), haploid, can1Δ::STE2pr-Sp_his5 and lyp1Δ (the SGA
    reporters the query carried and the progeny were selected for), and the three
    drug-sensitizing cassette deletions (query alleles, selected for at the last step).
    Pending source review: the four BY auxotrophies, which only the query side states
    and which disagree between Y13206 and its parent at ura3 and met15.
    """
    reporter = [QUERY_STRAIN, Y8835_GENOTYPE]
    sensitizer = [QUERY_STRAIN, TRIPLE_SELECTION]
    alleles = [
        standard_allele("can1Δ::STE2pr-Sp_his5", Zygosity.haploid, provenance=reporter),
        standard_allele("lyp1Δ", Zygosity.haploid, provenance=reporter),
        standard_allele("pdr1Δ::natMX", Zygosity.haploid, provenance=sensitizer),
        standard_allele("pdr3Δ::KlURA3", Zygosity.haploid, provenance=sensitizer),
        standard_allele("snq2Δ::KlLEU2", Zygosity.haploid, provenance=sensitizer),
        *(
            standard_allele(
                name,
                Zygosity.haploid,
                resolve_with=BRACHMANN_1998,
                note=_BY_AUXOTROPHY_NOTE,
            )
            for name in ("his3Δ1", "leu2Δ0", "ura3Δ0", "met15Δ0")
        ),
    ]
    return StrainBackground(
        name=LIBRARY_STRAIN,
        parents=["Y13206", "MATa xxxΔ::kanMX deletion array"],
        construction=(
            "SGA: MATalpha query Y13206 crossed to the MATa xxxΔ::kanMX array; MATa "
            "meiotic progeny selected on -His +canavanine +thialysine, then -Ura +NAT, "
            "then -Ura -Leu +G418 +NAT (Piotrowski 2017 Online Methods)"
        ),
        mating_type=MATA_PROGENY.value,
        ploidy="haploid",
        alleles=alleles,
        provenance=[MATA_PROGENY, QUERY_STRAIN],
    )


#: Loci whose allele the SGA selections fix in every library strain (the three
#: sensitizing deletions and the two reporters). A library row screening one of them
#: would put a kanMX deletion on a locus the background already replaced, so it is
#: dropped (rule ``orf_is_a_selected_background_locus``).
SELECTED_BACKGROUND_LOCI = frozenset(
    {"YGL013C", "YBL005W", "YDR011W", "YEL063C", "YNL268W"}
)

#: The 34 Fig 1B conditions as matrix tokens (FIG_1B_CONDITIONS). Every other matrix
#: token is a condition the paper never reported.
FIG_1B_TOKENS = frozenset(
    {
        "Benomyl",
        "Acetamide",
        "4OHBenzoicAcid",
        "5HMF",
        "BenzoicAcid",
        "SinapicAcid",
        "22Dipyridyl",
        "CoumaricAcid",
        "BMIMCl",
        "4OHBenzaldehyde",
        "DMSO",
        "Vanillin",
        "IBA",
        "Furfural",
        "CinnamicAcid",
        "VanillicAcid",
        "4OHAcetophenone",
        "MMS",
        "Syringaldehyde",
        "Methylglyoxal",
        "AzelaicAcid",
        "CoumaroylAmide",
        "FeruloylAmide",
        "SyringicAcid",
        "MBO",
        "Acetovanillone",
        "EMIMCl",
        "FerulicAcid",
        "Acetosyringone",
        "NAO",
        "26Dimethylpyrazine",
        "GVL",
        "CV",
        "EtOH",
    }
)

_SYSTEMATIC_RE = re.compile(
    r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
)
_SAMPLE_RE = re.compile(r"^(?P<compound>.+)_CG(?P<batch>\d+)_rep(?P<rep>\d+)$")
_CONTROL_RE = re.compile(r"^Control\d+_CG(?P<batch>\d+)$")

#: The one served condition the paper analyzed UNPAIRED, so it keeps the pooled control.
UNPAIRED_COMPOUND_TOKENS = frozenset({"MMS"})

#: The vehicle served as its own condition (1% v/v, DMSO_DOSE).
DMSO_TOKEN = "DMSO"

_PAIRED_UNITS = (
    "log2((TMM-normalized CPM of the inhibitor replicate + 1) / (mean TMM-normalized "
    "CPM of the SAME CG batch's inhibitor-free control columns + 1)), mean of 3 "
    "biological replicates; TMM factors (edgeR 3.26.8 calcNormFactors defaults) "
    "computed per condition over its replicates and paired controls; the 1 CPM "
    "pseudocount is a loader choice the paper does not state; recomputed from the GEO "
    "GSE186866 raw up-tag counts, NOT the paper's edgeR glmQLFit logFC"
)
_POOLED_UNITS = (
    "log2((TMM-normalized CPM of the inhibitor replicate + 1) / (mean TMM-normalized "
    "CPM of all 16 inhibitor-free control columns + 1)), mean of 3 biological "
    "replicates; pooled rather than batch-matched because the paper analyzed this "
    "compound unpaired; TMM factors (edgeR 3.26.8 calcNormFactors defaults) computed "
    "over its replicates and the 16 controls; the 1 CPM pseudocount is a loader choice "
    "the paper does not state; recomputed from the GEO GSE186866 raw up-tag counts, NOT "
    "the paper's edgeR glmQLFit logFC"
)


# --------------------------------------------------------------------------- #
# TMM normalization (edgeR 3.26.8 calcNormFactors, method="TMM", defaults)
# --------------------------------------------------------------------------- #
def _tmm_factor(
    obs: FloatArray, ref: FloatArray, libsize_obs: float, libsize_ref: float
) -> float:
    """Port of edgeR's ``.calcFactorTMM``: one sample against the reference sample."""
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_frac = obs / libsize_obs
        ref_frac = ref / libsize_ref
        log_ratio = np.log2(obs_frac / ref_frac)
        abs_expr = (np.log2(obs_frac) + np.log2(ref_frac)) / 2
        variance = (libsize_obs - obs) / libsize_obs / obs + (
            libsize_ref - ref
        ) / libsize_ref / ref
    finite = np.isfinite(log_ratio) & np.isfinite(abs_expr) & (abs_expr > TMM_A_CUTOFF)
    log_ratio, abs_expr, variance = (
        log_ratio[finite],
        abs_expr[finite],
        variance[finite],
    )
    if np.max(np.abs(log_ratio)) < 1e-6:
        return 1.0
    n = len(log_ratio)
    lo_l = np.floor(n * TMM_LOGRATIO_TRIM) + 1
    hi_l = n + 1 - lo_l
    lo_s = np.floor(n * TMM_SUM_TRIM) + 1
    hi_s = n + 1 - lo_s
    rank_l = rankdata(log_ratio)
    rank_s = rankdata(abs_expr)
    keep = (rank_l >= lo_l) & (rank_l <= hi_l) & (rank_s >= lo_s) & (rank_s <= hi_s)
    weighted = np.nansum(log_ratio[keep] / variance[keep]) / np.nansum(
        1.0 / variance[keep]
    )
    return float(2.0 ** (0.0 if np.isnan(weighted) else weighted))


class TmmFactors(BaseModel):
    """TMM scaling factors for one condition's sample set, with the reference used."""

    columns: list[str]
    library_sizes: list[float]
    factors: list[float]
    reference_column: str


def tmm_factors(
    counts: FloatArray, library_sizes: FloatArray, columns: list[str]
) -> TmmFactors:
    """Port of edgeR 3.26.8 ``calcNormFactors(method="TMM")`` with its default arguments.

    ``counts`` is genes x samples with no missing value; all-zero rows are removed as
    edgeR does. The reference sample is the one whose 75th-percentile count fraction is
    closest to the mean of those fractions (``which.min``, so the first on a tie), or
    the largest ``sum(sqrt(counts))`` when the median fraction is below 1e-20. Each
    sample's factor is the precision-weighted mean log2 ratio against the reference
    over the genes surviving a 30% log-ratio trim and a 5% abundance trim; the factors
    are then scaled to a geometric mean of 1. Effective library size = library size x
    factor.
    """
    if np.isnan(counts).any():
        raise ValueError("TMM needs complete counts (edgeR: 'NA counts not permitted')")
    x = counts[(counts > 0).any(axis=1)]
    upper = np.quantile(x, TMM_REFERENCE_QUANTILE, axis=0) / library_sizes
    if np.median(upper) < 1e-20:
        reference = int(np.argmax(np.sqrt(x).sum(axis=0)))
    else:
        reference = int(np.argmin(np.abs(upper - upper.mean())))
    raw = np.array(
        [
            _tmm_factor(
                x[:, i],
                x[:, reference],
                float(library_sizes[i]),
                float(library_sizes[reference]),
            )
            for i in range(x.shape[1])
        ]
    )
    factors = raw / np.exp(np.mean(np.log(raw)))
    return TmmFactors(
        columns=list(columns),
        library_sizes=[float(v) for v in library_sizes],
        factors=[float(v) for v in factors],
        reference_column=columns[reference],
    )


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never the live GEO URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/vanacloig-pedrosComparativeChemicalGenomic2022``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deposit_raw_mirror(
    *,
    counts_path: str | Path,
    retrieved_at: str = DATA_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from an already-retrieved count matrix + its ``manifest.json``.

    Idempotent by sha256: an existing mirror file with the recorded hash is left alone and
    a differing one raises rather than being overwritten. The GEO supplementary URL is
    directly scriptable, so the recorded retrieval re-runs as-is.
    """
    root = raw_mirror_dir(data_root)
    retrieval = RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=DATA_URL,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": DATA_URL},
        sha256=DATA_SHA256,
        retrieved_at=retrieved_at,
    )
    got = _sha256(counts_path)
    if got != DATA_SHA256:
        raise RuntimeError(
            f"{counts_path} sha256 mismatch: got {got}, expected {DATA_SHA256}"
        )
    dest = root / DATA_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != DATA_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(counts_path, dest)
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=(
            "Comparative chemical genomic profiling across plant-based hydrolysate "
            "toxins reveals widespread antagonism in fitness contributions"
        ),
        files=[
            ArtifactRecord(
                path=DATA_REL,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=DATA_SHA256,
                source=DATA_URL,
                retrieval=retrieval,
            )
        ],
        si_data_sources=[
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186866",
            DATA_URL,
        ],
        si_expected=[
            "Table S1 (per-compound IC30 molar values + DMSO control pairing) -- "
            "academic.oup.com is not scriptable, so it is NOT mirrored",
            "Dataset2_mclust_cdt (clustered edgeR logFC matrix) -- same publisher gate",
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


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["compound", "library_row", "cell"]
    description: str
    n_records: int
    items: list[str] = []


class LegacyOrfStrain(BaseModel):
    """A dropped library strain built against an ORF the current genome merged away.

    ``constructed_orf`` is the typed record of what the strain physically deleted; its
    relation to the current gene and its deleted interval are gaps naming SGD's locus
    history. Kept in the ledger so the drop is a typed record, not a silent loss.
    """

    source_orf: str
    current_orf: str
    barcode: str
    constructed_orf: ConstructedOrf


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]
    legacy_orf_strains: list[LegacyOrfStrain] = []


class LibraryRows(BaseModel):
    """The library rows a build keeps, after the gene-name policy."""

    keep_mask: list[bool]
    systematic: list[str]
    common: list[str]
    barcode: list[str]
    dropped_retired: list[str]
    dropped_legacy_duplicate: list[str]
    legacy_target: dict[str, str] = {}


def _canonical_common_names(genome: SCerevisiaeGenome) -> dict[str, str]:
    """``systematic name -> the genome's own standard (common) name``.

    One spelling per gene is what keeps a perturbation from splitting into two graph
    nodes, and taking it from the genome rather than from the source is what makes the
    spelling identical across datasets. Only a standard name that resolves BACK to the
    gene is used, so the stored pair always round-trips through the resolver.
    """
    canonical: dict[str, str] = {}
    for standard in genome.feature_index["standard_to_ids"]:
        resolution = genome.resolve_gene_name(standard)
        if resolution.is_current_gene and resolution.systematic_name is not None:
            canonical.setdefault(resolution.systematic_name, standard)
    return canonical


def resolve_library_rows(
    orfs: pd.Series, barcodes: pd.Series, genome: SCerevisiaeGenome
) -> LibraryRows:
    """Map every library row onto a current R64 gene, or drop it with a typed reason.

    Two rules, both of which the L1 canonical-name and L4 current-genome gates enforce
    downstream. A row whose ORF no longer names a gene of the current genome (a retired
    2005-era ORF, or a feature that is not a gene) is dropped: it cannot be keyed to a
    gene entity. A row whose ORF is the LEGACY spelling of an ORF the same library also
    carries under its current name is dropped too, because remapping it would merge two
    physically distinct barcoded strains into one record key; ``legacy_target`` keeps
    the current ORF each legacy spelling resolves to.
    """
    gene_set = {gene.upper() for gene in genome.gene_set}
    resolutions = {orf: genome.resolve_gene_name(orf) for orf in orfs.unique()}
    target: dict[str, str] = {}
    retired: list[str] = []
    for orf, resolution in resolutions.items():
        mapped = resolution.systematic_name
        if resolution.is_current_gene and mapped is not None and mapped in gene_set:
            target[orf] = mapped
        else:
            retired.append(orf)
    claimed = {orf for orf in target if orf in gene_set}
    legacy = sorted(
        orf for orf, mapped in target.items() if orf != mapped and mapped in claimed
    )
    canonical = _canonical_common_names(genome)
    keep_mask: list[bool] = []
    systematic: list[str] = []
    common: list[str] = []
    kept_barcodes: list[str] = []
    legacy_set = set(legacy)
    for orf, barcode in zip(orfs, barcodes, strict=True):
        mapped = target.get(orf)
        keep = mapped is not None and orf not in legacy_set
        keep_mask.append(keep)
        if not keep:
            continue
        assert mapped is not None
        systematic.append(mapped)
        common.append(canonical.get(mapped, mapped))
        kept_barcodes.append(barcode)
    if len(set(systematic)) != len(systematic):
        raise RuntimeError(
            "two retained library rows resolve to the same systematic gene; the "
            "legacy-duplicate rule did not separate them"
        )
    return LibraryRows(
        keep_mask=keep_mask,
        systematic=systematic,
        common=common,
        barcode=kept_barcodes,
        dropped_retired=sorted(retired),
        dropped_legacy_duplicate=legacy,
        legacy_target={orf: target[orf] for orf in legacy},
    )


def legacy_orf_strain(
    source_orf: str, current_orf: str, barcode: str
) -> LegacyOrfStrain:
    """The typed ledger entry for a dropped legacy-spelling strain."""
    return LegacyOrfStrain(
        source_orf=source_orf,
        current_orf=current_orf,
        barcode=barcode,
        constructed_orf=ConstructedOrf(
            source_systematic_name=source_orf,
            relation=None,
            deleted_span=None,
            provenance_gaps=[
                pending_source_review(
                    "relation",
                    SGD_ORF_HISTORY,
                    f"{source_orf} resolves to {current_orf} in R64-4-1; whether it "
                    "was merged into it or reannotated is SGD locus history",
                ),
                pending_source_review(
                    "deleted_span",
                    SGD_ORF_HISTORY,
                    f"the interval the {source_orf} cassette replaced",
                ),
            ],
        ),
    )


@register_dataset
class EnvChemgenVanacloig2022Dataset(ExperimentDataset):
    """Anaerobic chemical-genomic env x geno -> log2(inhibitor/control) response screen."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_vanacloig2022",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (the genome is loaded lazily inside ``process``)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return StrainEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return StrainEnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The GEO raw barcode-count matrix required before processing."""
        return [DATA_FILENAME]

    def download(self) -> None:
        """Link the mirror file into ``raw/`` after verifying it against ``DATA_SHA256``.

        The mirror + the ``DATA_SHA256`` pin is canonical; the GEO URL is retrieval
        metadata that ``deposit_raw_mirror`` re-runs, never a live build dependency. The
        manifest is the retrieval record and must carry the pin, else
        ``ManifestPinMismatchError`` names both digests.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        check_manifest_pin(DATA_REL, manifest_sha256(manifest, DATA_REL), DATA_SHA256)
        src = raw_mirror_dir(data_root) / DATA_REL
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(src, osp.join(self.raw_dir, DATA_FILENAME), DATA_SHA256)
        log.info(
            "Vanacloig 2022 raw matrix linked into %s (sha256 verified)", self.raw_dir
        )

    def _load_matrix(self) -> pd.DataFrame:
        """Read the gzipped raw-count TSV into a DataFrame."""
        path = osp.join(self.raw_dir, DATA_FILENAME)
        with gzip.open(path, "rt") as handle:
            return pd.read_csv(handle, sep="\t")

    # ---- environment / phenotype builders ------------------------------------ #
    def _concentration(self, compound: str) -> Concentration:
        """The dose as the paper SET it: an IC30 target, or a published fixed dose."""
        if compound == "Benomyl":
            return Concentration(
                value=BENOMYL_MOLAR.value,
                unit=ConcentrationUnit.micromolar,
                basis=DoseBasis.fixed,
            )
        if compound == "MMS":
            return Concentration(basis=DoseBasis.fixed)
        if compound == DMSO_TOKEN:
            return Concentration(
                value=DMSO_DOSE.value,
                unit=ConcentrationUnit.percent_v_v,
                basis=DoseBasis.fixed,
            )
        return Concentration(basis=IC30_BASIS.value)

    def _culture_format(self) -> CultureFormat:
        """Static 1.5 mL 24-well cultures inoculated at OD600 0.1, read at fixed times."""
        return CultureFormat(
            vessel=CULTURE_VESSEL.value["vessel"],
            working_volume_ul=CULTURE_VESSEL.value["working_volume_ul"],
            shaking_rpm=STATIC_CULTURE.value,
            inoculum_od600=0.1,
            endpoint=EndpointRule.fixed_duration,
            provenance=[CULTURE_VESSEL, STATIC_CULTURE],
        )

    def _base_environment(self, perturbations: list[Any]) -> CultureEnvironment:
        """Anaerobic static SynBase at pH 5.0 carrying ``perturbations`` on top."""
        return CultureEnvironment(
            media=SYNBASE,
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=perturbations,
            aerobicity=AEROBICITY.value,
            duration_hours=DURATION_HOURS.value,
            duration_generations=DURATION_GENERATIONS.value,
            culture_format=self._culture_format(),
            pre_culture=None,
            auxotroph_supplements=None,
            provenance_gaps=[
                pending_source_review(
                    "pre_culture",
                    PIOTROWSKI_2015,
                    "Vanacloig: 'Chemical genomic experiments were performed as "
                    "previously described (Piotrowski et al. 2015) with modifications'; "
                    "how the pool was grown before inoculation is not stated",
                ),
                pending_source_review(
                    "auxotroph_supplements",
                    ZHANG_2019,
                    "the paper names no supplement for the library's auxotrophies; "
                    "which amino acids SynH3- carries is in Zhang 2019",
                ),
            ],
        )

    def _ph(self) -> EnvironmentPhysicalPerturbation:
        """SynBase's stated pH, typed (``Media`` carries no pH field).

        ``agent`` is the acid the same sentence names as having set it, so the factor's
        realizing species is sourced rather than a silent None.
        """
        return EnvironmentPhysicalPerturbation(
            factor=PhysicalFactor.ph,
            magnitude=Concentration(value=MEDIUM_PH.value, unit=ConcentrationUnit.ph),
            agent=resolved_compound(PH_AGENT.value),
        )

    def _compound(self, compound: str) -> SmallMoleculePerturbation:
        """The dosed condition: an inhibitor with a typed solvent gap, or DMSO itself."""
        if compound == DMSO_TOKEN:
            return SmallMoleculePerturbation(
                compound=resolved_compound(compound),
                concentration=self._concentration(compound),
            )
        return SmallMoleculePerturbation(
            compound=resolved_compound(compound),
            concentration=self._concentration(compound),
            provenance_gaps=[_solvent_gap()],
        )

    def _environment(self, compound: str) -> CultureEnvironment:
        """The treated environment: SynBase + pH 5.0 + the condition at its dose."""
        return self._base_environment([self._compound(compound), self._ph()])

    def _genome_reference(self) -> StrainReferenceGenome:
        """The library's typed SGA-progeny background (constant across records)."""
        return StrainReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain=LIBRARY_STRAIN,
            ploidy="haploid",
            background=library_background(),
        )

    def _reference(self, compound: str) -> StrainEnvironmentResponseExperimentReference:
        """The inhibitor-FREE control the log2 ratio is taken against.

        The reference environment is the one the denominator was measured in: SynBase at
        pH 5.0 with no inhibitor, grown in the same culture format.
        """
        return StrainEnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=self._genome_reference(),
            environment_reference=self._base_environment([self._ph()]),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=READOUT.value,
                assay_type=ASSAY.value,
                environment_response=0.0,
                units=self._units(compound),
            ),
        )

    def _units(self, compound: str) -> str:
        """The readout definition, which records WHICH control the ratio used."""
        return _POOLED_UNITS if compound in UNPAIRED_COMPOUND_TOKENS else _PAIRED_UNITS

    def _phenotype(
        self, compound: str, response: float, sd: float
    ) -> EnvironmentResponsePhenotype:
        """One log2-ratio response with its across-replicate sample SD."""
        return EnvironmentResponsePhenotype(
            measurement_type=READOUT.value,
            assay_type=ASSAY.value,
            environment_response=response,
            environment_response_uncertainty=sd,
            environment_response_uncertainty_type=UncertaintyType.sample_sd,
            n_samples=N_REPLICATES.value,
            sample_unit=SampleUnit.biological_replicate,
            units=self._units(compound),
        )

    def _genotype(self, systematic: str, common: str, barcode: str) -> Genotype:
        """The ONE screened deletion; the constant background is on the reference."""
        return Genotype(
            perturbations=[
                BarcodedKanMxDeletionPerturbation(
                    systematic_gene_name=systematic,
                    perturbed_gene_name=common,
                    barcode=barcode,
                    collection=LIBRARY_COLLECTION.value,
                    cassette=ARRAY_KANMX.value,
                )
            ]
        )

    # ---- build ---------------------------------------------------------------- #
    @post_process
    def process(self) -> None:
        """Recompute per-(gene, condition) log2 responses from raw counts; write LMDB."""
        verify_raw_files(self.raw_dir, {DATA_FILENAME: DATA_SHA256})
        df = self._load_matrix()
        sample_cols = [c for c in df.columns if c not in ("gene", "std_name")]
        control_by_batch: dict[str, list[str]] = {}
        compound_cols: dict[str, list[str]] = {}
        for column in sample_cols:
            control = _CONTROL_RE.match(column)
            if control is not None:
                control_by_batch.setdefault(control.group("batch"), []).append(column)
                continue
            sample = _SAMPLE_RE.match(column)
            if sample is None:
                raise RuntimeError(f"unparseable sample column: {column!r}")
            compound_cols.setdefault(sample.group("compound"), []).append(column)
        if not control_by_batch:
            raise RuntimeError("no ControlN_CG* columns found in GSE186866 matrix")
        controls = [c for cols in control_by_batch.values() for c in cols]

        source_records = len(df) * len(compound_cols)
        rules: list[DropRule] = []

        # --- compound-level retention ----------------------------------------- #
        unreported = sorted(set(compound_cols) - FIG_1B_TOKENS)
        unidentified = sorted(
            token
            for token in set(compound_cols) & FIG_1B_TOKENS
            if not resolve_compound_identity(name=token).identified
        )
        kept_compounds = sorted(
            set(compound_cols) - set(unreported) - set(unidentified)
        )

        # --- library-row retention --------------------------------------------- #
        split = df["gene"].astype(str).str.split("_", n=1)
        orfs = split.str[0]
        barcodes = split.str[1].fillna("")
        is_orf = orfs.map(lambda gene: bool(_SYSTEMATIC_RE.match(gene)))
        has_counts = ~df[sample_cols].isna().any(axis=1)
        not_background = ~orfs.isin(SELECTED_BACKGROUND_LOCI)
        prefilter = is_orf & has_counts & not_background
        n_non_orf = int((~is_orf).sum())
        n_all_nan = int((is_orf & ~has_counts).sum())
        background_rows = sorted(orfs[is_orf & has_counts & ~not_background])

        genome = default_genome()
        library = resolve_library_rows(
            orfs[prefilter].reset_index(drop=True),
            barcodes[prefilter].reset_index(drop=True),
            genome,
        )
        row_keep = pd.Series(library.keep_mask, index=df.index[prefilter])
        keep = pd.Series(False, index=df.index)
        keep.loc[row_keep.index] = row_keep.to_numpy()
        n_rows = int(keep.sum())
        n_kept_compounds = len(kept_compounds)
        prefiltered_barcode = dict(
            zip(orfs[prefilter], barcodes[prefilter], strict=True)
        )

        rules.append(
            DropRule(
                rule="compound_not_reported_by_the_paper",
                scope="compound",
                description=(
                    "the matrix token is not one of the 34 conditions Fig 1B lists ("
                    + FIG_1B_CONDITIONS.quote
                    + " Bar labels read from "
                    + FIG_1B_IMAGE
                    + ", sha256 "
                    + FIG_1B_IMAGE_SHA256
                    + "); the paper never reports these conditions or says why"
                ),
                n_records=len(unreported) * len(df),
                items=unreported,
            )
        )
        rules.append(
            DropRule(
                rule="compound_without_a_structure_identifier",
                scope="compound",
                description=(
                    "no InChIKey / PubChem CID / ChEBI id resolves for the source label, "
                    "so the compound entity cannot be encoded or joined; see the per-"
                    "token reason in compound_identity_inputs/vanacloig2022.txt"
                ),
                n_records=len(unidentified) * len(df),
                items=unidentified,
            )
        )
        rules.append(
            DropRule(
                rule="row_is_not_a_barcoded_orf_or_carries_no_counts",
                scope="library_row",
                description=(
                    "the gene column is not '<systematic ORF>_<barcode>', or every count "
                    "column is missing (a QC-dropped barcode)"
                ),
                n_records=(n_non_orf + n_all_nan) * n_kept_compounds,
                items=[],
            )
        )
        rules.append(
            DropRule(
                rule="orf_is_a_selected_background_locus",
                scope="library_row",
                description=(
                    "the screened ORF is a locus whose allele the SGA selections fix in "
                    "every library strain (pdr1Δ::natMX, pdr3Δ::KlURA3, snq2Δ::KlLEU2, "
                    "can1Δ::STE2pr-Sp_his5, lyp1Δ; Piotrowski 2017), so a kanMX "
                    "deletion of it contradicts the strain's own background"
                ),
                n_records=len(background_rows) * n_kept_compounds,
                items=background_rows,
            )
        )
        rules.append(
            DropRule(
                rule="orf_is_not_a_current_genome_gene",
                scope="library_row",
                description=(
                    "the barcoded ORF does not resolve to a gene of the current R64 "
                    "annotation (retired ORF, or a non-gene feature), so no gene entity "
                    "exists to key the record to"
                ),
                n_records=len(library.dropped_retired) * n_kept_compounds,
                items=library.dropped_retired,
            )
        )
        rules.append(
            DropRule(
                rule="orf_is_a_legacy_spelling_of_another_library_orf",
                scope="library_row",
                description=(
                    "the ORF resolves to an ORF the SAME library also carries under its "
                    "current name; remapping would merge two physically distinct "
                    "barcoded strains into one record key, so the legacy row is dropped "
                    "and recorded as a typed ConstructedOrf in legacy_orf_strains"
                ),
                n_records=len(library.dropped_legacy_duplicate) * n_kept_compounds,
                items=library.dropped_legacy_duplicate,
            )
        )
        legacy_strains = [
            legacy_orf_strain(orf, library.legacy_target[orf], prefiltered_barcode[orf])
            for orf in library.dropped_legacy_duplicate
        ]
        log.info(
            "Vanacloig: %d conditions kept (%d unreported, %d unidentified dropped); "
            "%d library rows kept (%d non-ORF, %d all-NaN, %d background loci, %d "
            "retired, %d legacy duplicates dropped)",
            n_kept_compounds,
            len(unreported),
            len(unidentified),
            n_rows,
            n_non_orf,
            n_all_nan,
            len(background_rows),
            len(library.dropped_retired),
            len(library.dropped_legacy_duplicate),
        )

        # --- normalization ----------------------------------------------------- #
        # The library size is a property of the SEQUENCED SAMPLE, so it is summed over
        # EVERY released barcode (NaN = a QC-dropped barcode contributing no reads)
        # BEFORE any retention rule is applied, and TMM runs over every complete row.
        col_idx = {column: i for i, column in enumerate(sample_cols)}
        all_counts = df[sample_cols].to_numpy(dtype=np.float64)
        library_sizes = np.nansum(all_counts, axis=0)
        complete = ~np.isnan(all_counts).any(axis=1)
        kept_counts = df.loc[keep, sample_cols].to_numpy(dtype=np.float64)

        publication = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_all_zero_cells = 0
        normalization: dict[str, TmmFactors] = {}
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for compound in tqdm(kept_compounds, desc="Vanacloig conditions"):
                cols = compound_cols[compound]
                if len(cols) != N_REPLICATES.value:
                    raise RuntimeError(
                        f"{compound}: {len(cols)} replicate columns, expected "
                        f"{N_REPLICATES.value}"
                    )
                if compound in UNPAIRED_COMPOUND_TOKENS:
                    paired = {c: controls for c in cols}
                else:
                    paired = {
                        c: control_by_batch[_SAMPLE_RE.match(c).group("batch")]  # type: ignore[union-attr]  # every column matched above
                        for c in cols
                    }
                control_set = sorted({c for group in paired.values() for c in group})
                members = control_set + cols
                member_idx = [col_idx[c] for c in members]
                factors = tmm_factors(
                    all_counts[np.ix_(complete, member_idx)],
                    library_sizes[member_idx],
                    members,
                )
                normalization[compound] = factors
                # TMM-normalized CPM of every member column over the kept rows.
                effective = np.asarray(factors.library_sizes) * np.asarray(
                    factors.factors
                )
                tmm_cpm = kept_counts[:, member_idx] / effective * 1e6
                position = {column: i for i, column in enumerate(members)}
                log_rep = np.column_stack(
                    [
                        np.log2(tmm_cpm[:, position[c]] + CPM_PRIOR)
                        - np.log2(
                            tmm_cpm[:, [position[k] for k in paired[c]]].mean(axis=1)
                            + CPM_PRIOR
                        )
                        for c in cols
                    ]
                )
                response = log_rep.mean(axis=1)
                sd = log_rep.std(axis=1, ddof=1)
                all_zero = (kept_counts[:, [col_idx[c] for c in cols]] == 0).all(axis=1)
                n_all_zero_cells += int(all_zero.sum())
                environment = self._environment(compound)
                reference = self._reference(compound)
                for row in range(n_rows):
                    if all_zero[row]:
                        continue
                    experiment = StrainEnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=self._genotype(
                            library.systematic[row],
                            library.common[row],
                            library.barcode[row],
                        ),
                        environment=environment,
                        phenotype=self._phenotype(
                            compound, float(response[row]), float(sd[row])
                        ),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(experiment, reference, publication, itxn),
                    )
                    idx += 1
        env.close()
        interned_env.close()

        rules.append(
            DropRule(
                rule="all_three_replicate_counts_are_zero",
                scope="cell",
                description=(
                    "every replicate of this (strain, compound) cell has a raw count of "
                    "0, so no abundance was measured; the CPM pseudocount would turn a "
                    "below-detection observation into a finite log2 value whose sample "
                    "SD is exactly 0, i.e. fabricated infinite precision"
                ),
                n_records=n_all_zero_cells,
                items=[],
            )
        )
        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=idx,
            dropped_records=source_records - idx,
            rules=rules,
            legacy_orf_strains=legacy_strains,
        )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        with open(
            osp.join(self.preprocess_dir, "normalization_factors.json"), "w"
        ) as handle:
            json.dump(
                {name: f.model_dump() for name, f in normalization.items()},
                handle,
                indent=2,
            )
        accounted = sum(rule.n_records for rule in rules)
        if accounted != drop_log.dropped_records:
            raise RuntimeError(
                f"drop accounting mismatch: rules total {accounted}, "
                f"{drop_log.dropped_records} records missing from the build"
            )
        log.info("Wrote %d Vanacloig environment-response experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/env_chemgen_vanacloig2022")
    dataset = EnvChemgenVanacloig2022Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])
    print(
        json.dumps(
            json.loads(
                Path(osp.join(root, "preprocess/dropped_records.json")).read_text()
            )["rules"],
            indent=2,
        )[:2000]
    )


if __name__ == "__main__":
    main()
