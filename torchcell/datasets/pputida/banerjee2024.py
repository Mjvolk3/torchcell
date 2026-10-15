# torchcell/datasets/pputida/banerjee2024.py
# [[torchcell.datasets.pputida.banerjee2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/banerjee2024.py
# Test file: tests/torchcell/datasets/pputida/test_banerjee2024.py
r"""Banerjee 2024: isoprenol titers of the GSMM-guided P. putida KT2440 campaign (row 39).

Banerjee et al. 2024, "Genome-scale and pathway engineering for the sustainable aviation
fuel precursor isoprenol production in Pseudomonas putida", Metabolic Engineering,
doi:10.1016/j.ymben.2024.02.004.

**Every released isoprenol number this paper states is a record here, and the paper
states them in PROSE.** The Elsevier ESM was captured in full (six files, deposited and
hashed): Supplementary File 1 is the EFM/cMCS target scoring, File 2 and File 4 are
COBRA model JSONs, File 3 is the OptKnock/OptForce solution pool, File 5 is the
combined-score notebook, and the Supplementary Information PDF carries Table S1
(plasmids), Table S2 (strains + JBEI registry ids) and Table S3 (the M9 recipe). NONE of
them is a measurement table. The paper's data availability reads "Data will be made
available on request", and every per-strain series -- Figs. 3b, 4b, 4c, 5, 6a, 8a, 8b,
9a, 9b and Supplementary Figs. 1, 2, 4, 5 -- is a bar or a curve with no companion
table. So the titers come from the article body, each with its verbatim quote against
the sha256-pinned ``paper.md`` OCR, and NOTHING is read off a bar.

20 records, in five condition families, each family against the baseline the paper's own
fold-changes use:

======================  =======  =========================================  ===========
family                  records  medium / vessel                            reference
======================  =======  =========================================  ===========
EZ rich, pXW1 rounds    6        EZ rich + 20 g/L glucose, 50-mL tube       XW11, 104
EZ rich, plasmid panel  4        EZ rich + 20 g/L glucose, 50-mL tube       XW11, 104
M9 minimal              3        M9 (Table S3) + 20 g/L glucose, tube       IY1452, 762
fed-batch               4        modified M9, 2 L bioreactor                IY1245, 0.5
sorghum hydrolysate     3        modified M9 (10 mM NH4Cl), 5 mL tube       IY1452, 841
======================  =======  =========================================  ===========

Each number is stored on the unit the paper released it in: the mg/L statements as
``ug/mL`` (1 mg/L is exactly 1 ug/mL, so no arithmetic touches a source value) and the
fed-batch g/L statements as ``g/L``. The fed-batch family is the only one whose
uncertainty is RELEASED (``3.5 \pm 0.3 g/L``), so those four records carry
``titer_uncertainty`` with ``sample_sd`` over the stated biological triplicates; the
other sixteen carry a typed gap, because their spread exists only as an error bar.

What this paper carries that is NOT stored here, each with its count:

* **4 fed-batch OD600 values** (25.5 +- 0.7, 20.5 +- 3.0, 21.0 +- 3.4, 17.9 +- 0.2).
  No phenotype class and no ``MeasurementType`` member holds an optical density; that
  gap is issue #776 item 3, and the sibling Borchert 2023 loader records its own OD600
  series the same way. The growth RATES of Fig. 6b (0.22 and 0.26 h^-1) are stated for
  two strain GROUPS rather than per strain, and the 4.69-fold and 1.7-fold growth
  comparisons are ratios with no absolute member;
* **1 residual-glucose value** (XW17, 6 g/L at 48 h in EZ rich). ``MetabolitePhenotype``
  would hold the level, but ``MetaboliteExperimentReference`` requires a reference
  level and the only comparison the sentence makes is qualitative ("the XW19 strain
  depleted glucose after 48 h"); storing 0.0 for "depleted" would read a number out of
  prose that states none;
* **2 titers inside a two-strain range**: "the knockout of aceA (XW13) or gntZ (XW14)
  significantly decreased isoprenol production to 14-15 mg/L" gives one range for two
  strains, so neither value is assignable to a strain. Splitting it would be this
  loader's arithmetic;
* **1 second-hand baseline**: XW11's 104 mg/L, which this paper attributes to Wang et
  al. 2022. It is the denominator every EZ-rich fold-change in the paper uses, so it is
  the EZ-rich families' ``phenotype_reference``, and it is NOT a record of this dataset
  (it is Wang 2022's measurement).

The **M9-minimal family's reference carries a typed gap**, because its parental titer was
never released. The paper's denominator for that family is its wild type, stated only as
a ratio ("over 4-fold higher than WT (IY1245)"), and IY1245's M9 titer appears nowhere in
the mirror. So the reference names IY1245, stores its genotype and its condition, and
leaves ``titer`` as a ``ProvenanceGap``. Reconstructing 762/4 would invent a number, and
substituting a family member's own value (which the first build did, with IY1452's 762
mg/L) would state a parental measurement nobody made. The three released M9 titers stay
records either way: the gap is on the denominator, not on them. Every other family's
reference is a strain the paper names as the comparison WITH a released number.

The genome is a separate layer: this loader stores the identifiers the paper releases and
never asks the annotation whether one exists. ``gacA`` is the clear case -- Banerjee
names the gene and never gives it a locus tag. ``DeletionStrainPerturbation`` requires
one, and the three mirrored P. putida papers that do print a tag for that gene contradict
each other (PP_4099 in Schmidt 2022 and Thompson 2020, PP_1650 in Borchert 2023), so
IY1485's ΔgacA is recorded untyped with its quote and that contradiction rather than
resolved by majority. Its four fed-batch titers are unaffected: no measurement is lost,
one genotype element is, and it is counted.

Design, the five families and the L0-L4 table: [[torchcell.datasets.pputida.banerjee2024]].
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import shutil
from collections.abc import Callable, Hashable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import (
    EZ_RICH_BANERJEE2024,
    M9_BANERJEE2024,
    M9_MODIFIED_FEDBATCH_BANERJEE2024,
    M9_MODIFIED_TUBE_BANERJEE2024,
)
from torchcell.datamodels.schema import (
    BacterialGeneNamespace,
    BacterialReferenceStrain,
    Compound,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    CultureFormat,
    DeletionStrainPerturbation,
    EndpointRule,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    Genotype,
    HeterologousPathwayPerturbation,
    Media,
    PhysicalFactor,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import assembly_reference, reconcile_locus_tags
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.verification.levels import (
    l0_structural,
    l1_count,
    l2_cross_method,
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

DOI = "10.1016/j.ymben.2024.02.004"
TITLE = (
    "Genome-scale and pathway engineering for the sustainable aviation fuel precursor "
    "isoprenol production in Pseudomonas putida"
)
CITATION_KEY = "banerjeeGenomescalePathwayEngineering2024"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: The Elsevier ESM, all six files, as the publisher serves them.
ELSEVIER_PII = "S109671762400020X"
ELSEVIER_PREFIX = f"https://ars.els-cdn.com/content/image/1-s2.0-{ELSEVIER_PII}-"
SI_RETRIEVED_AT = "2026-10-12"

#: The article-body OCR every titer quotes.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "6b5b9e8647721569d846af92e5c0059ef9c4bb9eca66a418e8891a7a18d5c18c"
#: The Supplementary Information OCR Tables S1 to S3 are quoted from.
SI_MD = "si/si1.md"
SI_MD_SHA256 = "c0313af4bdbd443e7c61ad539ad3f101c88872f28993ac2ad98c154feb4d7ae0"
SI_PDF = "si/si1.pdf"
SI_PDF_SHA256 = "c09fe2f914846b3a2dee12d058d62ebb6041c52eb42d4b17b6b028c185ef0108"
#: The mirrored sibling campaign the mevalonate decarboxylase's organism defers to.
CARRUTHERS_KEY = "carruthersAutomationMachineLearning2025"
CARRUTHERS_PAPER_MD_SHA256 = (
    "ca9a8a2593d2ae3ab3bacfb767e797ece2bdaa0228a4f798f1c38e1af73ef88d"
)

#: Every deposited ESM file: published name, local name, role and sha256.
SI_DEPOSIT: tuple[tuple[str, str, str, str], ...] = (
    ("mmc1.pdf", "si/si1.pdf", "si_pdf", SI_PDF_SHA256),
    (
        "mmc2.xlsx",
        "si/si2.xlsx",
        "si_data",
        "5c60bbb9f02184b0ba3b96dfa84ba3a8e11e98f2b0b0b1e2b85b2e44bb1b9d04",
    ),
)

JBEI_REGISTRY_URL = "https://public-registry.jbei.org"
PROTEOMEXCHANGE_ACCESSION = "PXD039868"
PANORAMA_URL = "https://panoramaweb.org/genome-scale-eng-SAF-p-putida.url"

KT2440_NAMESPACE: BacterialGeneNamespace = "pputida_kt2440_locus_tag"
PARENT_STRAIN: BacterialReferenceStrain = "KT2440"

PATHWAY_ISOPRENOL = "isoprenol via the mevalonate IPP-bypass"
PRODUCT_NAME = "isoprenol"
#: 3-methyl-3-buten-1-ol, the SAF precursor; curating the compound-identity row needs a
#: PubChem call against the committed input lists, which is a human act.
ISOPRENOL_INCHIKEY = "VSMOENVRRABVKN-UHFFFAOYSA-N"
QUANTIFICATION_METHOD = "GC-FID"

#: The three vessels the Methods state, each with the statement that gives it.
CULTURE_FORMATS: dict[str, dict[str, Any]] = {
    "tube_50ml": {
        "vessel": "50-mL test tube",
        "working_volume_ml": 5.0,
        "shaking_rpm": 200.0,
    },
    "tube": {"vessel": "culture tube", "working_volume_ml": 5.0, "shaking_rpm": 200.0},
    "bioreactor": {
        "vessel": "2 L bioreactor (Biostat B, Sartorius)",
        "working_volume_ml": 1000.0,
        "shaking_rpm": None,
    },
}


# --------------------------------------------------------------------------- #
# Sourced values: every number below quotes sha256-pinned mirrored bytes
# --------------------------------------------------------------------------- #
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
    """Bind a value to a verbatim line of the deposited Supplementary Information OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI_MD,
            citation_key=CITATION_KEY,
            sha256=SI_MD_SHA256,
            method="MinerU OCR of the deposited Elsevier ESM mmc1.pdf (raw mirror)",
            page=page,
            retrieved=SI_RETRIEVED_AT,
        ),
    )


def _carruthers(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to the MIRRORED sibling campaign PMD's organism defers to.

    Banerjee writes the decarboxylase only as the part token ``PMDHKQ`` and never names
    its organism; the mirrored Carruthers 2025 campaign carries the same part and states
    it. The provenance chain (this part string -> the same part in Carruthers 2025) is
    the citation, per the deferral rule in ``CLAUDE.md``.
    """
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CARRUTHERS_KEY,
            sha256=CARRUTHERS_PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror of the "
            "deferred-to paper)",
            page=page,
        ),
    )


_METHODS_STRAINS = "Methods 2.4, 'Strains and plasmid construction'"
_METHODS_PRODUCTION = "Methods 2.5, 'Isoprenol production in P. putida'"
_METHODS_FEDBATCH = "Methods 2.7, 'Isoprenol production in fed-batch mode'"
_METHODS_HYDROLYSATE = "Methods 2.8, 'Isoprenol production using biomass hydrolysate'"
_RESULTS_KNOCKOUTS = (
    "Results 3.2, 'Experimental implementation of metabolic rewiring for isoprenol "
    "production'"
)
_RESULTS_PATHWAY = (
    "Results 3.3, 'Pathway optimization for improved isoprenol production'"
)
_RESULTS_M9 = (
    "Results 3.4, 'Isoprenol production using optimized isoprenol production pathway "
    "and predicted metabolic rewiring in glucose minimal medium'"
)
_RESULTS_FEDBATCH = "Results 3.6, 'Isoprenol production in fed-batch cultivation'"
_RESULTS_HYDROLYSATE = "Results 3.7, 'Isoprenol production using biomass hydrolysate'"
_SI_TABLE1 = "Supplementary Table 1, 'Plasmids used in this study'"
_SI_TABLE2 = "Supplementary Table 2, 'Strains used in this study'"
_FIG3_CAPTION = "Fig. 3 caption"
_FIG8_CAPTION = "Fig. 8 caption"

# --- the titer statements, verbatim from the pinned paper.md ---------------- #
_Q_XW11_BASELINE = (
    "This strain has shown the highest isoprenol level in P. putida $( 1 0 4 ~ \\mathrm "
    "{ m g / L }$ , XW11 strain) when using the IPP-bypass MVA pathway via a plasmid "
    "(pXW1, see Supplementary Table 1 for the list of plasmids) in our previous work "
    "(Wang et al., 2022)."
)
_Q_XW12 = (
    "The knockout of mvaB improved isoprenol production up to $1 6 4 ~ \\mathrm { m g / "
    "L }$ , which increased 1.6-fold compared with the starter strain XW11 (Fig. 3b)."
)
_Q_XW13_XW14_RANGE = (
    "While the knockout of aceA (XW13) or gntZ (XW14) significantly decreased isoprenol "
    "production to $1 4 \\mathrm { - } 1 5 ~ \\mathrm { \\ m g / L }$ , the hbdH "
    "knockout (XW15) improved isoprenol production up to $2 4 1 \\mathrm { m g / L }$ , "
    "a 2.3-fold increase from the XW11 strain and 1.5-fold increase from the XW12 "
    "strain, respectively (Fig. 3b)."
)
_Q_XW16 = (
    "However, experimental results (Fig. 3b) showed that the deletion of gltA (XW16) "
    "significantly compromised cell growth and eventually lowered isoprenol production "
    "$\\bf { \\Omega } ( 1 9 \\ m g / L )$ ."
)
_Q_XW17_XW18 = (
    "The knockout of aceA or gntZ on the XW15 strain, however, was still producing low "
    "levels of isoprenol: $1 2 \\mathrm { m g } / \\mathrm { L }$ for XW17 strain "
    "(∆phaABC ΔmvaB ∆hbdH ∆aceA) and $1 3 \\mathrm { m g / L }$ for "
    "XW18 strain (∆phaABC ΔmvaB ∆hbdH ∆gntZ), respectively (Fig. 3b)."
)
_Q_XW19 = (
    "Interestingly, the inclusion of gntZ knockout significantly restored isoprenol "
    "production to $1 1 4 ~ \\mathrm { m g / L }$ (Fig. 3b)."
)
_Q_IY782 = (
    "The highest titer of $8 0 ~ \\mathrm { m g / L }$ isoprenol was achieved from "
    "strain IY782, carrying the isoprenol pathway in the plasmid with the RK2 "
    "replication origin (Fig. 4b)."
)
_Q_IY721_IY846 = (
    "The resulting strain (IY721) produced up to $3 9 5 ~ \\mathrm { m g / L }$ "
    "isoprenol in $4 8 \\mathrm { h }$ (Fig. 4b). When combined with the GSMM "
    "approaches by deleting two genes, mvaB and hbdH, the resulting strain (IY846) "
    "improved the isoprenol production by 1.3-fold when compared to strain IY721, "
    "yielding approximately $5 3 6 ~ \\mathrm { m g / L }$ isoprenol in $4 8 \\ "
    "\\mathrm { h }$ (Fig. 4b)."
)
_Q_IY939 = (
    "Therefore, we co-expressed NudB, PhoA, and AphA along with the isoprenol "
    "biosynthetic pathway and found that co-expressing AphA alone in the ΔphaABC "
    "ΔmvaB ΔhbdH background (strain IY939 with plasmid pIY670) produced the "
    "highest isoprenol titer of $1 , 1 1 1 \\ \\mathrm { m g / L }$ in 48 h (Fig. 4b)."
)
_Q_M9_TITERS = (
    "We observed that the IY1452 strain (∆phaABC ΔmvaB ΔhbdH ∆ldhA "
    "∆PP_2675 with plasmid pIY670) had the best isoprenol titer at $7 6 2 ~ "
    "\\mathrm { m g / L }$ , Fig. 5); over 4-fold higher than WT (IY1245). The highest "
    "isoprenol titer observed was $8 1 6 ~ \\mathrm { m g / L }$ for the IY1252 strain "
    "at $4 8 \\mathrm { h }$ but that reduced by about $7 0 \\ \\% - 2 5 9 \\ \\mathrm "
    "{ m g / L }$ at $7 2 \\mathrm { h }$ ."
)
_Q_FEDBATCH_IY1245 = (
    "In the IY1245 control strain, the maximum cell growth and the isoprenol production "
    "were obtained at $^ { 7 2 \\mathrm { ~ h ~ } }$ , reaching an $\\mathrm { \\ O D } "
    "_ { 6 0 0 }$ of $2 5 . 5 \\pm$ 0.7 and isoprenol titer of $0 . 5 \\pm 0 . 1 \\ : "
    "\\mathrm { g / L }$ , respectively (Fig. 8)."
)
_Q_FEDBATCH_IY1262 = (
    "The IY1262 strain produced $2 . 3 \\pm 0 . 3 \\ : \\mathrm { g / L }$ of isoprenol "
    "and the $\\mathrm { O D } _ { 6 0 0 }$ reached $2 0 . 5 \\pm 3 . 0$ at $^ { 9 6 "
    "\\mathrm { ~ h ~ } }$ (Fig. 8)."
)
_Q_FEDBATCH_IY1452 = (
    "The maximum growth and isoprenol production in the IY1452 strain were obtained, "
    "reaching an $\\mathrm { O D } _ { 6 0 0 }$ of $2 1 . 0 \\pm 3 . 4$ at $^ { 7 2 "
    "\\mathrm { ~ h ~ } }$ and a titer of $3 . 5 \\pm$ $0 . 3 ~ \\ g / \\mathrm { L }$ "
    "at $^ { 9 6 \\mathrm { ~ h ~ } }$ (Fig. 8)."
)
_Q_FEDBATCH_IY1485 = (
    "The resulting IY1485 strain reached the $\\mathrm { O D } _ { 6 0 0 }$ of $1 7 . 9 "
    "\\pm 0 . 2$ at $4 8 \\mathrm { ~ h ~ }$ and produced $2 . 4 \\pm 0 . 3$ $g / "
    "\\mathrm { L }$ of isoprenol at $^ { 9 6 \\mathrm { ~ h ~ } }$ in fed-batch mode."
)
_Q_HYDROLYSATE = (
    "The highest isoprenol titer from this strain was $8 4 1 ~ \\mathrm { m g / L }$ at "
    "$^ { 7 2 \\mathrm { ~ h ~ } }$ in a modified M9 minimal medium supplemented with "
    "${ 2 0 } \\ g / \\mathrm { L }$ of glucose as a sole carbon source (Fig. 9a). "
    "Isoprenol production was lower with sorghum hydrolysate: $4 0 9 ~ \\mathrm { m g / "
    "L }$ and $4 3 2 ~ \\mathrm { m g / L }$ at $^ { 7 2 \\mathrm { ~ h ~ } }$ with $5 "
    "\\%$ and $1 0 ~ \\%$ sorghum hydrolysate, respectively (Fig. 9a)."
)

# --- the conditions, verbatim ----------------------------------------------- #
_Q_EZRICH = (
    "The starter cultures were diluted in $5 ~ \\mathrm { m L }$ EZ rich defined medium "
    "(Teknova, CA, USA) containing ${ 2 0 ~ \\ g / \\mathrm { L } }$ D-glucose $( 2 ~ "
    "\\%$ ,w/v), $2 5 \\mu \\mathrm { g / m L }$ kanamycin in $5 0 \\mathrm { - m L }$ "
    "test tubes, and $0 . 5 \\mathrm { m M }$ IPTG or arabinose $( 2 \\% )$ was added "
    "to induce protein expression with $\\mathrm { \\ O D } _ { 6 0 0 }$ at 0.40.6."
)
_Q_EZRICH_INCUBATION = (
    "The P. putida cultures were incubated in rotary shakers $( 2 0 0 ~ \\mathrm { r p "
    "m } )$ at ${ } ^ { 3 0 ~ ^ { \\circ } \\mathrm { C } }$ for $4 8 \\mathrm { ~ h ~ }$ ."
)
_Q_M9_RUNS = (
    "For isoprenol production runs on M9 minimal medium (Supplementary Table 2) with $2 "
    "\\%$ D-glucose, cryostocks were streaked to singles on LB agar plate with $5 0 ~ { "
    "\\mu \\mathrm { g / m L } }$ kanamycin at $3 0 ^ { \\circ } \\mathrm { C } .$"
)
_Q_M9_SAMPLING = (
    "For the production runs, the cells were inoculated at an initial $\\mathrm { \\ O D "
    "} _ { 6 0 0 }$ of 0.2 and the isoprenol production pathway was induced with 2 $\\%$ "
    "arabinose immediately after inoculation. Samples were collected every $^ { 6 "
    "\\mathrm { ~ h ~ } }$ until $^ { 7 2 \\mathrm { ~ h ~ } }$ in triplicates and "
    "analyzed for growth $\\left( \\mathrm { O D } _ { 6 0 0 } \\right)$ , isoprenol, "
    "residual glucose and organic acids."
)
_Q_M9_SAME_FORMAT = (
    "Unless otherwise mentioned, all further cultivations were performed in the same "
    "format and conditions."
)
_Q_M9_GLUCOSE = (
    "We tested wild-type and fourteen different knockout strains in the pIY670 "
    "background for growth and isoprenol production in M9 glucose minimal media plus $2 "
    "0 ~ \\ g / \\mathrm { L }$ glucose (Fig. 5)."
)
_Q_FEDBATCH_VESSEL = (
    "The cell culture was inoculated at $\\mathrm { O D } _ { 6 0 0 } 0 . 3$ in the 2 L "
    "bioreactor (Biostat B, Sartorius, Germany) including the 1 L modified M9 minimal "
    "medium"
)
_Q_FEDBATCH_CONTROL = (
    "To produce isoprenol in the fed-batch fermentation, the dissolved oxygen (DO) and "
    "airflow were set to $2 0 \\%$ and 1 VVM (volume of air per volume of liquid per "
    "minute), respectively. The temperature was maintained at ${ } ^ { 2 5 } \\ ^ { "
    "\\circ } \\mathrm { C }$ and $\\mathsf { p H }$ was maintained at 6.5 by "
    "supplementation with $2 5 ~ \\%$ ammonia water. The isoprenol biosynthesis pathway "
    "was induced at $\\mathrm { O D } _ { 6 0 0 } \\ 0 . 6 { - 0 . 8 }$ by $2 \\%$ "
    "arabinose."
)
_Q_FEDBATCH_FEED = (
    "After the batch phase with the modified M9 minimal medium containing ${ 2 0 ~ \\ g "
    "/ \\mathrm { L } }$ of glucose and $1 . 0 6 ~ \\gimel { \\mathcal { A } }$ (or 20 "
    "mM) ammonium chloride, the feeding solution was continuously added to make a total "
    "of $1 0 0 g / \\mathrm { L }$ glucose and $2 . 1 2 { \\ g } / \\mathrm { L }$ "
    "ammonium chloride."
)
_Q_HYDROLYSATE_TUBES = (
    "Then the seed culture was inoculated in the culture tubes including $5 \\ \\mathrm "
    "{ m L }$ modified M9 minimal medium supplemented sorghum hydrolysate at $3 0 ~ ^ { "
    "\\circ } \\mathrm { C }$ and $2 0 0 \\ \\mathrm { r p m }$ The isoprenol "
    "biosynthesis pathway was induced at $\\mathrm { O D } _ { 6 0 0 } 0 . 6 { - } 0 . "
    "8$ by $2 \\%$ arabinose."
)
_Q_HYDROLYSATE_GLUCOSE = (
    "The glucose was added either as a sole carbon source or as co-substrate with "
    "sorghum hydrolysate and its concentration was adjusted to ${ 2 0 } \\ g / \\mathrm "
    "{ L }$ ."
)
_Q_HYDROLYSATE_SOURCE = (
    "For the isoprenol production using biomass hydrolysate, the cholinium lysinate "
    "ionic liquid-pretreated sorghum hydrolysate was obtained from Joint BioEnergy "
    "Institute (JBEI) Deconstruction Division (Magurudeniya et al., 2021) and was used "
    "as a carbon source for the engineered P. putida strain."
)
_Q_GCFID = (
    "$1 ~ \\mu \\mathrm { L }$ of the ethyl acetate layer was analyzed by gas "
    "chromatography-flame ionization detection (GC-FID, Thermo Focus GC) equipped with "
    "a DB-WAX column"
)
_Q_TRIPLICATE_FIG3 = (
    "Data was obtained from three biological replicates and error bars represent "
    "standard deviation."
)
_Q_TRIPLICATE_FIG8 = (
    "performed in the $^ { 2 \\mathrm { ~ L ~ } }$ bioreactors in M9 defined medium in "
    "triplicates. Error bars represent standard deviation."
)
_Q_TRIPLICATE_FIG9 = (
    "The productions of isoprenol and growth were evaluated in the culture tubes in $^ "
    "\\mathrm { 5 ~ m L }$ modified  minimal medium with varying concentrations of "
    "hydrolysate in triplicates. Error bars represent standard deviation."
)
_Q_RESIDUAL_GLUCOSE = (
    "It was observed that the XW19 strain depleted glucose after $^ { 4 8 \\mathrm { ~ h "
    "~ } }$ while the XW17 strain showed $6 ~ \\ g / \\mathrm { L }$ residual glucose in "
    "the medium."
)
_Q_GROWTH_RATES = (
    "to maintain similar growth rates as WT and ΔphaABC strains (Fig. 6b, $0 . 2 2 "
    "\\ \\mathrm { h r } ^ { - 1 }$ versus $0 . 2 6 \\ \\mathrm { h r } ^ { - 1 } .$"
)
_Q_DATA_AVAILABILITY = "Data will be made available on request."
_Q_IY1485_GENOTYPE = (
    "To reduce foaming during the fed-batch cultivation, the gacA gene was deleted in "
    "the 5 genes knockout strain (IY1452) as previously reported (Eng et al., 2021)."
)
_Q_PP2675 = (
    "Since we switched from EZ rich medium (several carbon sources) to minimal medium "
    "with glucose as the sole carbon source, we deleted an additional gene, PP_2675, to "
    "avoid the catabolism of isoprenol after glucose consumption as this gene was "
    "reported to be involved in isoprenol degradation (Thompson et al., 2020)."
)
_Q_PHA_CLUSTER = (
    "We also use known edits such as deletion of the pha gene cluster (PP_5003 to "
    "PP_5005)"
)
_Q_APHA_ECOLI = (
    "In our previous paper, we showed that NudB, a native phosphatase of E. coli, "
    "hydrolyzed IPP and DMAPP into their monophosphate forms, IP and DMAP, "
    "respectively, which are subsequently hydrolyzed to isoprenol by other phosphatases "
    "such as AphA, Agp, and YqaB (Kang et al., 2016)."
)
_Q_PMD_CARRUTHERS = "promiscuous mevalonate decarboxylase (PMD\\*) from S. cerevisiae"

# --- Table 1 and the SI tables ---------------------------------------------- #
_Q_TABLE1_TARGETS = (
    "<table><tr><td>Reaction ID</td><td>Reaction Name</td><td>Gene Association</td>"
    "<td>Notea</td></tr><tr><td>HMGL</td><td>Hydroxymethylglutaryl- CoA lyase</td>"
    "<td>mvaB (PP_3540)</td><td>1st target by combined score</td></tr><tr><td>BDH</td>"
    "<td>3-hydroxybutyrate dehydrogenase</td><td>hbdH (PP_3073)</td><td>2nd target by "
    "combined score</td></tr><tr><td>GND</td><td>Phosphogluconate dehydrogenase</td>"
    "<td>gntZ (PP_4043)</td><td>3rd target by combined score</td></tr><tr><td>LDH_D</td>"
    "<td>D-lactate dehydrogenase</td><td>ldhA (PP_1649)</td><td>High cMCS/EFM score, "
    "low OptKnock score</td></tr><tr><td>PPS</td><td>Phosphoenolpyruvate synthase</td>"
    "<td>ppsA (PP_2082)</td><td>High cMCS/EFM score, low OptKnock</td></tr><tr><td>CS"
    "</td><td>Citrate synthase</td><td>gltA (PP_4194)</td><td>score High OptKnock "
    "score, low cMCS/EFM score</td></tr><tr><td>ICL</td><td>Isocitrate lyase</td>"
    "<td>aceA (PP_4116)</td><td>High OptKnock score, low cMCS/EFM score</td></tr></table>"
)

#: Each engineered deletion's locus tag, as Table 1 and the Results state them. Read
#: from the paper only: the annotation is never consulted to find or confirm a tag.
DELETION_LOCUS_TAGS: dict[str, SourcedValue] = {
    "mvaB": _paper("PP_3540", _Q_TABLE1_TARGETS, page="Table 1"),
    "hbdH": _paper("PP_3073", _Q_TABLE1_TARGETS, page="Table 1"),
    "gntZ": _paper("PP_4043", _Q_TABLE1_TARGETS, page="Table 1"),
    "ldhA": _paper("PP_1649", _Q_TABLE1_TARGETS, page="Table 1"),
    "ppsA": _paper("PP_2082", _Q_TABLE1_TARGETS, page="Table 1"),
    "gltA": _paper("PP_4194", _Q_TABLE1_TARGETS, page="Table 1"),
    "aceA": _paper("PP_4116", _Q_TABLE1_TARGETS, page="Table 1"),
    "PP_2675": _paper("PP_2675", _Q_PP2675, page=_RESULTS_M9),
}
#: The pha cluster, which the paper states as a RANGE of three tags.
PHA_LOCUS_TAGS: tuple[str, ...] = ("PP_5003", "PP_5004", "PP_5005")
PHA_CLUSTER = _paper(
    PHA_LOCUS_TAGS,
    _Q_PHA_CLUSTER,
    page="Introduction",
    note="Supplementary Table 2 states the same range as 'ΔphaABC (PP_5003 to "
    "PP_5005)'; the three tags are the range's endpoints and its one interior member, "
    "which is what 'to' over consecutive tags means",
)
#: ``gacA`` is the one deleted gene this paper never gives a locus tag, and it is the
#: one genotype element of this dataset that no served leaf can hold:
#: ``DeletionStrainPerturbation.systematic_gene_name`` takes a KT2440 locus tag and
#: refuses a bare symbol. The deferral the sentence offers ("as previously reported (Eng
#: et al., 2021)") does not close it either, and the three mirrored P. putida papers that
#: DO print a tag for this gene CONTRADICT each other:
#:
#:   * Borchert 2023 (``paper.md`` sha256 e156d3b2...): "gacA (PP_1650) and gacS (PP_4099)"
#:   * Schmidt 2022 (``paper.md`` sha256 acc5c8bf...): "gacS (PP_1650) and gacA (PP_4099)"
#:   * Thompson 2020 (``paper.md`` sha256 389d0d6c...): "gacS and gacA (PP_1650 and PP_4099)"
#:
#: Two readings say PP_4099 and one says PP_1650, and picking the majority would be this
#: loader resolving an identity across sources, which is exactly what it must not do.
#: So the IY1485 record carries its five tagged deletions and its plasmid, the ΔgacA is
#: recorded untyped with its quote and this contradiction, and the reconciliation is left
#: to modeling time where the evidence is visible. IY1485's four fed-batch titers are
#: unaffected: no measurement is lost, one genotype element is.
GACA_NO_TAG = _paper(
    "gacA",
    _Q_IY1485_GENOTYPE,
    page=_RESULTS_FEDBATCH,
    note="the released identity is the SYMBOL: Banerjee states no locus tag for it, and "
    "the genome tier is not consulted to supply one. DeletionStrainPerturbation "
    "requires a locus tag, so this deletion is carried as a recorded finding rather "
    "than as a typed perturbation; the mirrored papers that print a tag disagree "
    "(PP_4099 in Schmidt 2022 and Thompson 2020, PP_1650 in Borchert 2023), which is "
    "why resolving it belongs downstream and not here",
)
#: The genotype elements the release states that no served leaf holds, with their count.
GENOTYPE_NOT_TYPED: tuple[tuple[str, str, str], ...] = (
    (
        "IY1485",
        "gacA",
        "DeletionStrainPerturbation requires a KT2440 locus tag and Banerjee states "
        "none for gacA; the mirrored siblings that do print one contradict each other "
        "(PP_4099 in Schmidt 2022 and Thompson 2020, PP_1650 in Borchert 2023), so no "
        "tag is chosen here",
    ),
)

TEMPERATURE_SHAKEN_C = _paper(
    30.0,
    _Q_EZRICH_INCUBATION,
    page=_METHODS_PRODUCTION,
    note="every shaken culture: the EZ rich rounds, the M9 runs that Methods 2.5 says "
    "share the format, and the hydrolysate tubes of Methods 2.8",
)
TEMPERATURE_FEDBATCH_C = _paper(
    25.0, _Q_FEDBATCH_CONTROL, page=_METHODS_FEDBATCH, note="the bioreactor setpoint"
)
AEROBICITY = _paper(
    "aerobic",
    _Q_FEDBATCH_CONTROL,
    page=_METHODS_FEDBATCH,
    note="rotary-shaken tubes at 200 rpm and a bioreactor held at 20% dissolved oxygen "
    "with 1 VVM airflow; the source never uses the word",
)
N_REPLICATES = _paper(
    3,
    _Q_TRIPLICATE_FIG3,
    page=_FIG3_CAPTION,
    note="biological triplicates, stated identically for Figs. 3, 4, 8, 9 and "
    "Supplementary Figs. 1 to 5. Only the fed-batch statements release the SD NUMBER; "
    "every other family's spread exists solely as an error bar",
)
UNCERTAINTY_TYPE = _paper(
    "sample_sd",
    _Q_TRIPLICATE_FIG8,
    page=_FIG8_CAPTION,
    note="what the released +- numbers of Results 3.6 ARE: a standard deviation over "
    "biological triplicates in the 2 L bioreactors",
)
QUANTIFICATION = _paper(QUANTIFICATION_METHOD, _Q_GCFID, page=_METHODS_PRODUCTION)
DATA_AVAILABILITY = _paper(
    "on request",
    _Q_DATA_AVAILABILITY,
    page="Data availability",
    note="why no per-strain measurement table exists in the ESM: the six deposited "
    "files are the computational design artifacts and the Supplementary Information, "
    "and the measured series are released only as figures",
)

#: The 4 fed-batch OD600 values the schema has no home for (issue #776 item 3).
OD600_NOT_STORED: tuple[tuple[str, float, float, float], ...] = (
    ("IY1245", 25.5, 0.7, 72.0),
    ("IY1262", 20.5, 3.0, 96.0),
    ("IY1452", 21.0, 3.4, 72.0),
    ("IY1485", 17.9, 0.2, 48.0),
)
OD600_REASON = (
    "no phenotype class and no MeasurementType member holds an optical density (issue "
    "#776 item 3); the sibling Borchert 2023 loader records its own OD600 series the "
    "same way. The 2 L bioreactor growth DURATION is stored, as "
    "CultureEnvironment.duration_hours"
)
RESIDUAL_GLUCOSE_NOT_STORED = _paper(
    6.0,
    _Q_RESIDUAL_GLUCOSE,
    page=_RESULTS_KNOCKOUTS,
    note="XW17's residual glucose at 48 h in EZ rich, the one glucose number the paper "
    "states. MetabolitePhenotype would hold the level, but MetaboliteExperimentReference "
    "requires a reference level and the only comparison this sentence makes is "
    "qualitative ('the XW19 strain depleted glucose after 48 h'): storing 0.0 for "
    "'depleted' would read a number out of prose that states none",
)
GROWTH_RATES_NOT_STORED = _paper(
    (0.22, 0.26),
    _Q_GROWTH_RATES,
    page="Results 3.5",
    note="stated for two strain GROUPS (the mvaB-deleted strains against WT and "
    "ΔphaABC) rather than per strain, so neither value is assignable to a record",
)
XW13_XW14_RANGE_NOT_STORED = _paper(
    (14.0, 15.0),
    _Q_XW13_XW14_RANGE,
    page=_RESULTS_KNOCKOUTS,
    note="one range for two strains: splitting 14-15 mg/L into XW13 and XW14 would be "
    "this loader's arithmetic, not the paper's statement. Both strains ARE in Fig. 3b, "
    "whose bars are not a released number",
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/banerjeeGenomescalePathwayEngineering2024``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/<citation key>``: the OCR the quotes cite."""
    return Path(data_root or _data_root()) / "torchcell-library" / CITATION_KEY


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def elsevier_url(published_name: str) -> str:
    """The publisher URL one ESM file was retrieved from."""
    return f"{ELSEVIER_PREFIX}{published_name}"


def deposit_raw_mirror(data_root: str | None = None) -> Path:
    """Copy the two consumed Supplementary artifacts into the raw mirror, and record them.

    Only what the loader CONSUMES is deposited: the Supplementary Information PDF and
    its OCR markdown, which Tables S1 to S3 are quoted from. The other four ESM files
    (two COBRA model JSONs, the EFM/cMCS workbook, the OptKnock/OptForce workbook and
    the combined-score notebook) are captured in the literature mirror and recorded in
    ``si_expected`` below; no loader reads a model or a solution pool.

    Idempotent by sha256: an existing mirror file whose digest matches is left alone and
    a differing one raises rather than being overwritten.
    """
    library = library_mirror_dir(data_root)
    root = raw_mirror_dir(data_root)
    pins = ((SI_PDF, SI_PDF_SHA256), (SI_MD, SI_MD_SHA256))
    for relpath, expected in pins:
        verify_sha256(library / relpath, expected)
    records: list[ArtifactRecord] = []
    for relpath, expected in pins:
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != expected:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(library / relpath, dest)
        url = elsevier_url("mmc1.pdf")
        records.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_SI_DATA,
                bytes=dest.stat().st_size,
                sha256=expected,
                source=url,
                original_filename="mmc1.pdf",
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=url,
                    retriever="torchcell.literature.capture_si (Elsevier ESM route)",
                    params={"pii": ELSEVIER_PII, "file": "mmc1.pdf"},
                    sha256=expected,
                    retrieved_at=SI_RETRIEVED_AT,
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=records,
        si_data_sources=[
            elsevier_url(name) for name in ("mmc1.pdf", "mmc2.xlsx", "mmc6.pdf")
        ]
        + [PANORAMA_URL, JBEI_REGISTRY_URL],
        si_expected=[
            "si/si1.pdf (publisher mmc1.pdf) and its OCR si/si1.md -- deposited. "
            "Supplementary Tables 1 (plasmids), 2 (strains + JBEI ids) and 3 (the M9 "
            "recipe) are quoted from them. Supplementary Figures 1 to 7 are in the same "
            "PDF and are figures, not tables",
            "mmc2.xlsx (Supplementary File 1), mmc4.xlsx (File 3), mmc6.pdf (File 5) -- "
            "captured in the literature mirror, NOT deposited: EFM/cMCS target scores, "
            "the OptKnock/OptForce solution pool and the combined-score notebook. They "
            "are computational design artifacts, and no loader consumes a design",
            "mmc3.txt (File 2) and mmc5.txt (File 4) -- the iJN1462 IPP-bypass COBRA "
            "model and the central-metabolism model, as JSON. Captured, not deposited: "
            "a metabolic model is the metabolism tier's object, not a measurement",
            "NO per-strain measurement table exists in this release. Every titer, OD600 "
            "and glucose series is a figure panel (Figs. 3b, 4b, 4c, 5, 6a, 8a, 8b, 9a, "
            "9b, Supplementary Figs. 1, 2, 4, 5), and the paper's data availability "
            f"reads {_Q_DATA_AVAILABILITY!r}. The 20 stored titers are therefore the "
            "ones the article body states numerically, each carried as a sourced "
            f"constant quoting the pinned paper.md OCR ({PAPER_MD_SHA256})",
            f"ProteomeXchange {PROTEOMEXCHANGE_ACCESSION} and {PANORAMA_URL} -- the "
            "targeted MRM proteomics of the pathway proteins. Not deposited: the "
            "release is raw MRM data, and Supplementary Fig. 4 is the only processed "
            "view, which is a figure",
            f"{JBEI_REGISTRY_URL} -- the strains and plasmids are deposited in the JBEI "
            "Registry and 'are available from the authors upon request', so no cassette "
            "sequence is retrievable by script; the genotype strings of Supplementary "
            "Tables 1 and 2 are the authority this loader types against",
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


def every_quote() -> dict[str, str]:
    """Every verbatim quote in this module, keyed by its constant name."""
    return {
        name: value
        for name, value in sorted(globals().items())
        if name.startswith("_Q_") and isinstance(value, str)
    }


def verify_quotes(data_root: str | None = None) -> int:
    """Assert every quote is verbatim in the sha256-pinned bytes it cites.

    The titers are carried as sourced constants rather than read from a data file, so
    this is what makes them auditable: both OCR mirrors are hashed against their pins
    and each quote must be a substring of the one it cites. Returns the number checked.
    """
    library = library_mirror_dir(data_root)
    texts: dict[str, str] = {}
    for relpath, expected in ((PAPER_MD, PAPER_MD_SHA256), (SI_MD, SI_MD_SHA256)):
        path = library / relpath
        digest = _sha256(path)
        if digest != expected:
            raise RuntimeError(
                f"{path} hashes {digest}, not the pinned {expected}; the OCR mirror "
                "moved and every quote must be re-read before it is trusted"
            )
        texts[relpath] = path.read_text(encoding="utf-8")
    si_only = {"_Q_SI_TABLE1_PLASMIDS", "_Q_SI_TABLE2_STRAINS"}
    deferred = {"_Q_PMD_CARRUTHERS"}
    quotes = every_quote()
    missing = [
        name
        for name, quote in quotes.items()
        if name not in deferred
        and quote not in texts[SI_MD if name in si_only else PAPER_MD]
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(quotes)} quotes are not verbatim in their pinned "
            f"mirror: {missing}"
        )
    carruthers = (
        Path(data_root or _data_root())
        / "torchcell-library"
        / CARRUTHERS_KEY
        / PAPER_MD
    )
    digest = _sha256(carruthers)
    if digest != CARRUTHERS_PAPER_MD_SHA256:
        raise RuntimeError(
            f"{carruthers} hashes {digest}, not the pinned "
            f"{CARRUTHERS_PAPER_MD_SHA256}; the deferred-to mirror moved"
        )
    if _Q_PMD_CARRUTHERS not in carruthers.read_text(encoding="utf-8"):
        raise RuntimeError(
            "the deferred PMD organism quote is not verbatim in the mirrored "
            "Carruthers 2025 paper.md"
        )
    return len(quotes)


# --------------------------------------------------------------------------- #
# Genotype
# --------------------------------------------------------------------------- #
class CassetteGene(BaseModel):
    """One heterologous gene of a plasmid, with the statement that gives its organism."""

    token: str
    symbol: str
    organism: str
    promoter: str
    organism_quote: str
    organism_source: Literal["si", "paper", "carruthers"]
    variant: str | None = None


class Cassette(BaseModel):
    """One isoprenol-pathway plasmid, as Supplementary Table 1 states its parts."""

    name: str
    construct_string: str
    genes: tuple[CassetteGene, ...]


_Q_SI_TABLE1_PLASMIDS = (
    "<tr><td>pXW1</td><td>pBbB5k-MvaSef-MvaEef-T1-MKmm-PMDHKQ</td><td>Wang et al."
    "</td></tr>"
)
_Q_SI_TABLE2_STRAINS = (
    "<tr><td>XW01</td><td></td><td>JPUB_019964 P. putida KT2440 \u0394phaABC (PP_5003 "
    "to PP_5005)</td><td>Wang et al. 2022</td></tr>"
)

#: The organism of ``MvaSef``, ``MvaEef`` and ``MKmm`` is read from the part token's own
#: suffix in the quoted plasmid description (``ef`` = Enterococcus faecalis, ``mm`` =
#: Methanosarcina mazei); Banerjee never writes those organisms in prose. ``PMDHKQ``
#: carries no suffix and no Banerjee statement, so it is sourced by DEFERRAL to the
#: mirrored Carruthers 2025 campaign, which carries the same part and names it. ``AphA``
#: Banerjee places in E. coli itself.
_MVA_GENES: tuple[CassetteGene, ...] = (
    CassetteGene(
        token="MvaSef",
        symbol="MvaS",
        organism="Enterococcus faecalis",
        promoter="PBAD",
        organism_quote=_Q_SI_TABLE1_PLASMIDS,
        organism_source="si",
    ),
    CassetteGene(
        token="MvaEef",
        symbol="MvaE",
        organism="Enterococcus faecalis",
        promoter="PBAD",
        organism_quote=_Q_SI_TABLE1_PLASMIDS,
        organism_source="si",
    ),
    CassetteGene(
        token="MKmm",
        symbol="MK",
        organism="Methanosarcina mazei",
        promoter="Ptrc1-O",
        organism_quote=_Q_SI_TABLE1_PLASMIDS,
        organism_source="si",
    ),
    CassetteGene(
        token="PMDHKQ",
        symbol="PMD",
        organism="Saccharomyces cerevisiae",
        promoter="Ptrc1-O",
        variant="HKQ",
        organism_quote=_Q_PMD_CARRUTHERS,
        organism_source="carruthers",
    ),
)
_APHA_GENE = CassetteGene(
    token="AphA",
    symbol="AphA",
    organism="Escherichia coli",
    promoter="Ptrc1-O",
    organism_quote=_Q_APHA_ECOLI,
    organism_source="paper",
)

CASSETTES: dict[str, Cassette] = {
    "pXW1": Cassette(
        name="pXW1",
        construct_string="pBbB5k-MvaSef-MvaEef-T1-MKmm-PMDHKQ",
        genes=_MVA_GENES,
    ),
    "pIY554": Cassette(
        name="pIY554",
        construct_string=("pRK2-Kan-araC-PBAD-MvaSef-MvaEef-TrpoH-Ptrc1-O-MKmm-PMDHKQ"),
        genes=_MVA_GENES,
    ),
    "pIY603": Cassette(
        name="pIY603",
        construct_string=(
            "pRK2-Kan-lacI-PlacUV5-MvaSef-MvaEef-TrpoH-Ptrc1-O-MKmm-PMDHKQ"
        ),
        genes=_MVA_GENES,
    ),
    "pIY670": Cassette(
        name="pIY670",
        construct_string=(
            "pRK2-Kan-araC-PBAD-MvaSef-MvaEef-TrpoH-Ptrc1-O-MKmm-PMDHKQ-AphA"
        ),
        genes=(*_MVA_GENES, _APHA_GENE),
    ),
}


class StrainSpec(BaseModel):
    """One released strain: its deletions, its plasmid and the statement of its genotype."""

    strain: str
    deletions: tuple[str, ...]
    plasmid: str
    genotype_quote: str
    genotype_page: str


#: Every strain a record or a reference is written for, with the deletion SYMBOLS the
#: release names. ``phaABC`` expands to the three tags the paper's range gives.
STRAIN_SPECS: dict[str, StrainSpec] = {
    "XW11": StrainSpec(
        strain="XW11",
        deletions=("phaABC",),
        plasmid="pXW1",
        genotype_quote=_Q_SI_TABLE2_STRAINS,
        genotype_page=_SI_TABLE2,
    ),
    "XW12": StrainSpec(
        strain="XW12",
        deletions=("phaABC", "mvaB"),
        plasmid="pXW1",
        genotype_quote=_Q_XW12,
        genotype_page=_RESULTS_KNOCKOUTS,
    ),
    "XW15": StrainSpec(
        strain="XW15",
        deletions=("phaABC", "mvaB", "hbdH"),
        plasmid="pXW1",
        genotype_quote=_Q_XW13_XW14_RANGE,
        genotype_page=_RESULTS_KNOCKOUTS,
    ),
    "XW16": StrainSpec(
        strain="XW16",
        deletions=("phaABC", "mvaB", "hbdH", "gltA"),
        plasmid="pXW1",
        genotype_quote=_Q_XW16,
        genotype_page=_RESULTS_KNOCKOUTS,
    ),
    "XW17": StrainSpec(
        strain="XW17",
        deletions=("phaABC", "mvaB", "hbdH", "aceA"),
        plasmid="pXW1",
        genotype_quote=_Q_XW17_XW18,
        genotype_page=_RESULTS_KNOCKOUTS,
    ),
    "XW18": StrainSpec(
        strain="XW18",
        deletions=("phaABC", "mvaB", "hbdH", "gntZ"),
        plasmid="pXW1",
        genotype_quote=_Q_XW17_XW18,
        genotype_page=_RESULTS_KNOCKOUTS,
    ),
    "XW19": StrainSpec(
        strain="XW19",
        deletions=("phaABC", "mvaB", "hbdH", "aceA", "gntZ"),
        plasmid="pXW1",
        genotype_quote=_Q_XW19,
        genotype_page=_RESULTS_KNOCKOUTS,
    ),
    "IY782": StrainSpec(
        strain="IY782",
        deletions=("phaABC",),
        plasmid="pIY603",
        genotype_quote=_Q_IY782,
        genotype_page=_RESULTS_PATHWAY,
    ),
    "IY721": StrainSpec(
        strain="IY721",
        deletions=("phaABC",),
        plasmid="pIY554",
        genotype_quote=_Q_IY721_IY846,
        genotype_page=_RESULTS_PATHWAY,
    ),
    "IY846": StrainSpec(
        strain="IY846",
        deletions=("phaABC", "mvaB", "hbdH"),
        plasmid="pIY554",
        genotype_quote=_Q_IY721_IY846,
        genotype_page=_RESULTS_PATHWAY,
    ),
    "IY939": StrainSpec(
        strain="IY939",
        deletions=("phaABC", "mvaB", "hbdH"),
        plasmid="pIY670",
        genotype_quote=_Q_IY939,
        genotype_page=_RESULTS_PATHWAY,
    ),
    "IY1245": StrainSpec(
        strain="IY1245",
        deletions=(),
        plasmid="pIY670",
        genotype_quote=_Q_FEDBATCH_IY1245,
        genotype_page=_RESULTS_FEDBATCH,
    ),
    "IY1252": StrainSpec(
        strain="IY1252",
        deletions=("phaABC", "mvaB", "hbdH", "aceA"),
        plasmid="pIY670",
        genotype_quote=_Q_M9_TITERS,
        genotype_page=_RESULTS_M9,
    ),
    "IY1262": StrainSpec(
        strain="IY1262",
        deletions=("phaABC", "mvaB", "hbdH", "ldhA"),
        plasmid="pIY670",
        genotype_quote=_Q_FEDBATCH_IY1262,
        genotype_page=_RESULTS_FEDBATCH,
    ),
    "IY1452": StrainSpec(
        strain="IY1452",
        deletions=("phaABC", "mvaB", "hbdH", "ldhA", "PP_2675"),
        plasmid="pIY670",
        genotype_quote=_Q_M9_TITERS,
        genotype_page=_RESULTS_M9,
    ),
    "IY1485": StrainSpec(
        strain="IY1485",
        deletions=("phaABC", "mvaB", "hbdH", "ldhA", "PP_2675", "gacA"),
        plasmid="pIY670",
        genotype_quote=_Q_IY1485_GENOTYPE,
        genotype_page=_RESULTS_FEDBATCH,
    ),
}


def deletion_perturbations(symbol: str) -> list[DeletionStrainPerturbation]:
    """Every chromosomal deletion one released SYMBOL stands for.

    ``phaABC`` is three loci, because the paper states the cluster as a tag RANGE;
    ``gacA`` is one locus under its symbol, because the paper states no tag for it and
    the annotation is not consulted to supply one.
    """
    if symbol == "phaABC":
        return [
            DeletionStrainPerturbation(
                systematic_gene_name=tag,
                perturbed_gene_name=f"pha{letter}",
                gene_namespace=KT2440_NAMESPACE,
                cassette=None,
                collection=None,
            )
            for tag, letter in zip(PHA_LOCUS_TAGS, "ABC", strict=True)
        ]
    if symbol == "gacA":
        # No typed perturbation: the paper releases the symbol and no locus tag, and the
        # leaf requires one. Recorded in GENOTYPE_NOT_TYPED, never guessed.
        return []
    tag = str(DELETION_LOCUS_TAGS[symbol].value)
    return [
        DeletionStrainPerturbation(
            systematic_gene_name=tag,
            perturbed_gene_name=symbol,
            gene_namespace=KT2440_NAMESPACE,
            cassette=None,
            collection=None,
        )
    ]


def cassette_perturbations(cassette: Cassette) -> list[HeterologousPathwayPerturbation]:
    """Every heterologous gene of one plasmid, as its own typed perturbation.

    Each part token must appear in the plasmid's quoted construct string, so the
    identifiers stay the source's own and a changed quote cannot drift away from them.
    """
    for gene in cassette.genes:
        if gene.token not in cassette.construct_string:
            raise RuntimeError(
                f"cassette part {gene.token!r} is not in the quoted {cassette.name} "
                f"description {cassette.construct_string!r}"
            )
    return [
        HeterologousPathwayPerturbation(
            systematic_gene_name=gene.token,
            perturbed_gene_name=gene.symbol,
            gene_namespace=KT2440_NAMESPACE,
            pathway_name=PATHWAY_ISOPRENOL,
            source_organism=gene.organism,
            is_heterologous=True,
            localization="episomal_plasmid",
            construct_name=cassette.name,
            integration_locus=None,
            variant=gene.variant,
            promoter_name=gene.promoter,
            copy_number=1.0,
        )
        for gene in cassette.genes
    ]


def strain_genotype(strain: str) -> Genotype:
    """The genotype of one released strain: its deletions plus its plasmid."""
    spec = STRAIN_SPECS[strain]
    perturbations: list[Any] = []
    for symbol in spec.deletions:
        perturbations.extend(deletion_perturbations(symbol))
    perturbations.extend(cassette_perturbations(CASSETTES[spec.plasmid]))
    return Genotype(perturbations=perturbations)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def publication() -> Publication:
    """This paper, by DOI (no PubMed id is recorded in the mirror's manifest)."""
    return Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")


def _dose(
    compound: str, value: float, unit: ConcentrationUnit
) -> SmallMoleculePerturbation:
    """One added small molecule at a stated dose, through the shared compound layer."""
    return SmallMoleculePerturbation(
        compound=resolved_compound(compound),
        concentration=Concentration(value=value, unit=unit),
    )


def _carbon(
    compound: str, value: float, unit: ConcentrationUnit
) -> EnvironmentPhysicalPerturbation:
    """One carbon source of the medium, as a typed physical factor."""
    return EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.carbon_source,
        magnitude=Concentration(value=value, unit=unit),
        agent=resolved_compound(compound),
    )


class ConditionSpec(BaseModel):
    """One stated culture condition: the medium, the vessel and the additions."""

    key: str
    medium: Literal["ez_rich", "m9", "m9_modified_fedbatch", "m9_modified_tube"]
    culture_format: Literal["tube_50ml", "tube", "bioreactor"]
    temperature_c: float
    glucose_g_per_l: float
    arabinose_percent: float | None
    hydrolysate_percent: float | None
    duration_hours: float | None
    inoculum_od600: float | None
    quote: str
    page: str


_MEDIA_BY_KEY: dict[str, Media] = {
    "ez_rich": EZ_RICH_BANERJEE2024,
    "m9": M9_BANERJEE2024,
    "m9_modified_fedbatch": M9_MODIFIED_FEDBATCH_BANERJEE2024,
    "m9_modified_tube": M9_MODIFIED_TUBE_BANERJEE2024,
}

#: The inducer is NOT carried on the EZ-rich records. Methods 2.5 states it as a
#: DISJUNCTION ("0.5 mM IPTG or arabinose (2%)") and never says which strain received
#: which; the plasmid descriptions name both a lacI and an araC regulator across the
#: panel, so assigning one per strain would be this loader's inference. Every
#: arabinose-induced family states its inducer directly and carries it.
EZRICH_INDUCER_NOT_ASSIGNED = _paper(
    "0.5 mM IPTG or arabinose (2%)",
    _Q_EZRICH,
    page=_METHODS_PRODUCTION,
    note="a disjunction, not a per-strain assignment: the EZ-rich records carry no "
    "inducer perturbation, and the statement is kept here and in "
    "preprocess/condition_provenance.json rather than resolved by reading the "
    "regulator out of each plasmid's part string",
)

CONDITIONS: dict[str, ConditionSpec] = {
    "tube_ezrich_glc20": ConditionSpec(
        key="tube_ezrich_glc20",
        medium="ez_rich",
        culture_format="tube_50ml",
        temperature_c=float(TEMPERATURE_SHAKEN_C.value),
        glucose_g_per_l=20.0,
        arabinose_percent=None,
        hydrolysate_percent=None,
        duration_hours=48.0,
        inoculum_od600=None,
        quote=_Q_EZRICH,
        page=_METHODS_PRODUCTION,
    ),
    "tube_m9_glc20": ConditionSpec(
        key="tube_m9_glc20",
        medium="m9",
        culture_format="tube_50ml",
        temperature_c=float(TEMPERATURE_SHAKEN_C.value),
        glucose_g_per_l=20.0,
        arabinose_percent=2.0,
        hydrolysate_percent=None,
        duration_hours=None,
        inoculum_od600=0.2,
        quote=_Q_M9_SAMPLING,
        page=_METHODS_PRODUCTION,
    ),
    "bioreactor_m9mod_fedbatch": ConditionSpec(
        key="bioreactor_m9mod_fedbatch",
        medium="m9_modified_fedbatch",
        culture_format="bioreactor",
        temperature_c=float(TEMPERATURE_FEDBATCH_C.value),
        glucose_g_per_l=20.0,
        arabinose_percent=2.0,
        hydrolysate_percent=None,
        duration_hours=None,
        inoculum_od600=0.3,
        quote=_Q_FEDBATCH_VESSEL,
        page=_METHODS_FEDBATCH,
    ),
    "tube_m9mod_glc20": ConditionSpec(
        key="tube_m9mod_glc20",
        medium="m9_modified_tube",
        culture_format="tube",
        temperature_c=float(TEMPERATURE_SHAKEN_C.value),
        glucose_g_per_l=20.0,
        arabinose_percent=2.0,
        hydrolysate_percent=None,
        duration_hours=72.0,
        inoculum_od600=None,
        quote=_Q_HYDROLYSATE_GLUCOSE,
        page=_METHODS_HYDROLYSATE,
    ),
    "tube_m9mod_hydrolysate_5": ConditionSpec(
        key="tube_m9mod_hydrolysate_5",
        medium="m9_modified_tube",
        culture_format="tube",
        temperature_c=float(TEMPERATURE_SHAKEN_C.value),
        glucose_g_per_l=20.0,
        arabinose_percent=2.0,
        hydrolysate_percent=5.0,
        duration_hours=72.0,
        inoculum_od600=None,
        quote=_Q_HYDROLYSATE,
        page=_RESULTS_HYDROLYSATE,
    ),
    "tube_m9mod_hydrolysate_10": ConditionSpec(
        key="tube_m9mod_hydrolysate_10",
        medium="m9_modified_tube",
        culture_format="tube",
        temperature_c=float(TEMPERATURE_SHAKEN_C.value),
        glucose_g_per_l=20.0,
        arabinose_percent=2.0,
        hydrolysate_percent=10.0,
        duration_hours=72.0,
        inoculum_od600=None,
        quote=_Q_HYDROLYSATE,
        page=_RESULTS_HYDROLYSATE,
    ),
}

#: Every statement that fixes one condition, written beside the build for audit.
CONDITION_PROVENANCE: dict[str, list[SourcedValue]] = {
    "tube_ezrich_glc20": [
        _paper(CULTURE_FORMATS["tube_50ml"], _Q_EZRICH, page=_METHODS_PRODUCTION),
        _paper(48.0, _Q_EZRICH_INCUBATION, page=_METHODS_PRODUCTION),
        EZRICH_INDUCER_NOT_ASSIGNED,
    ],
    "tube_m9_glc20": [
        _paper(20.0, _Q_M9_GLUCOSE, page=_RESULTS_M9),
        _paper(
            CULTURE_FORMATS["tube_50ml"],
            _Q_M9_SAME_FORMAT,
            page=_METHODS_PRODUCTION,
            note="the M9 runs state no vessel of their own; Methods 2.5 says the "
            "further cultivations share the EZ-rich format, which is 5 mL in a 50-mL "
            "test tube on a rotary shaker at 200 rpm",
        ),
        _paper(2.0, _Q_M9_SAMPLING, page=_METHODS_PRODUCTION),
        _paper(
            "M9 minimal medium (Supplementary Table 2)",
            _Q_M9_RUNS,
            page=_METHODS_PRODUCTION,
            note="the Methods point at Supplementary Table 2, which is the STRAIN "
            "table; the M9 recipe is Supplementary Table 3, captioned 'Minimal M9 "
            "medium recipe'. The medium object is built from Table 3, and this "
            "internal mismatch is recorded rather than silently corrected",
        ),
    ],
    "bioreactor_m9mod_fedbatch": [
        _paper(
            CULTURE_FORMATS["bioreactor"], _Q_FEDBATCH_VESSEL, page=_METHODS_FEDBATCH
        ),
        _paper(25.0, _Q_FEDBATCH_CONTROL, page=_METHODS_FEDBATCH),
        _paper(100.0, _Q_FEDBATCH_FEED, page=_RESULTS_FEDBATCH),
    ],
    "tube_m9mod_glc20": [
        _paper(
            CULTURE_FORMATS["tube"], _Q_HYDROLYSATE_TUBES, page=_METHODS_HYDROLYSATE
        ),
        _paper(20.0, _Q_HYDROLYSATE_GLUCOSE, page=_METHODS_HYDROLYSATE),
    ],
    "tube_m9mod_hydrolysate_5": [
        _paper(5.0, _Q_HYDROLYSATE, page=_RESULTS_HYDROLYSATE),
        _paper(
            "cholinium lysinate ionic liquid-pretreated sorghum hydrolysate",
            _Q_HYDROLYSATE_SOURCE,
            page=_METHODS_HYDROLYSATE,
        ),
    ],
    "tube_m9mod_hydrolysate_10": [
        _paper(10.0, _Q_HYDROLYSATE, page=_RESULTS_HYDROLYSATE),
        _paper(
            "cholinium lysinate ionic liquid-pretreated sorghum hydrolysate",
            _Q_HYDROLYSATE_SOURCE,
            page=_METHODS_HYDROLYSATE,
        ),
    ],
}

_DURATION_GAP_NOTE = (
    "the statement that gives this titer names no sampling time, and the family's own "
    "time course is a figure; an endpoint rule naming a time the source never gave "
    "would assert one"
)


def _culture_format(
    key: str,
    *,
    endpoint_stated: bool,
    inoculum_od600: float | None,
    quote: SourcedValue,
) -> CultureFormat:
    """The typed culture format of one condition, with the statement that gives it.

    ``shaking_rpm`` is ``None`` for the bioreactor, which is sparged and stirred rather
    than shaken and whose agitation the Methods never state as an impeller speed; that
    absence is the source's, not a dropped value.
    """
    stated = CULTURE_FORMATS[key]
    return CultureFormat(
        vessel=str(stated["vessel"]),
        working_volume_ul=float(stated["working_volume_ml"]) * 1000.0,
        shaking_rpm=(
            None if stated["shaking_rpm"] is None else float(stated["shaking_rpm"])
        ),
        inoculum_od600=inoculum_od600,
        endpoint=EndpointRule.fixed_duration if endpoint_stated else None,
        provenance=[quote],
    )


def environment(
    spec: ConditionSpec, *, duration_hours: float | None = None
) -> CultureEnvironment:
    """The environment of one stated condition.

    A ``CultureEnvironment``, since ``ProductTiterExperiment.environment`` is annotated
    as one: a titer is read with its vessel, so the condition's own vessel travels on
    the record rather than only beside it.
    """
    perturbations: list[Any] = [
        _carbon("D-glucose", spec.glucose_g_per_l, ConcentrationUnit.g_per_l)
    ]
    if spec.hydrolysate_percent is not None:
        perturbations.append(
            _carbon(
                "cholinium lysinate pretreated sorghum hydrolysate",
                spec.hydrolysate_percent,
                ConcentrationUnit.percent_v_v,
            )
        )
    if spec.arabinose_percent is not None:
        perturbations.append(
            _dose("L-arabinose", spec.arabinose_percent, ConcentrationUnit.percent_w_v)
        )
    hours = spec.duration_hours if duration_hours is None else duration_hours
    gaps: list[ProvenanceGap] = []
    if hours is None:
        gaps.append(
            ProvenanceGap(
                field="duration_hours",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=_DURATION_GAP_NOTE,
            )
        )
    return CultureEnvironment(
        media=_MEDIA_BY_KEY[spec.medium],
        temperature=Temperature(value=spec.temperature_c),
        culture_format=_culture_format(
            spec.culture_format,
            endpoint_stated=hours is not None,
            inoculum_od600=spec.inoculum_od600,
            quote=_paper(spec.key, spec.quote, page=spec.page),
        ),
        perturbations=perturbations,
        aerobicity=str(AEROBICITY.value),
        duration_hours=hours,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
def isoprenol() -> Compound:
    """The product as a typed ``Compound`` through the shared compound-identity layer."""
    return resolved_compound(PRODUCT_NAME)


def titer_phenotype(
    titer: float | None, unit: ConcentrationUnit, *, uncertainty: float | None
) -> ProductTiterPhenotype:
    """One released titer on its OWN released unit, with the typed absences it forces.

    No arithmetic touches a source value: a mg/L statement is stored as ``ug/mL`` (1
    mg/L is exactly 1 ug/mL) and a g/L statement as ``g/L``. ``uncertainty`` is the
    released ``+-`` number where the paper prints one, which is the fed-batch family
    only; everywhere else the spread exists solely as a figure error bar and both the
    number and its type are typed gaps, because the schema forbids a type without a
    value.

    ``titer`` is ``None`` for exactly one object in this dataset: the M9-minimal
    family's REFERENCE, whose parental value the paper states only as a ratio. The unit
    is still the family's released one, because what is missing is the parent's number
    and not the scale every member of that family is reported on.
    """
    gaps: list[ProvenanceGap] = [
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_reported_by_primary,
            note="no volumetric productivity is released for any strain or condition",
        )
        for field in ("productivity", "productivity_unit")
    ]
    gaps.extend(
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_reported_by_primary,
            note="no yield on substrate is released for any strain or condition; the "
            "only yield statement in the paper is the model's maximum theoretical yield",
        )
        for field in ("product_yield", "product_yield_unit")
    )
    if titer is None:
        gaps.append(
            ProvenanceGap(
                field="titer",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the parental titer of this family is never released: the paper "
                "states its denominator only as a ratio ('over 4-fold higher than WT "
                "(IY1245)') and prints no M9 titer for that strain anywhere in the "
                "mirror. Reconstructing it from the ratio would invent a number and "
                "substituting a family member's own value would state a measurement "
                "nobody made",
            )
        )
    if uncertainty is None:
        gaps.extend(
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the replicate SPREAD of this family is released only as an error "
                f"bar: no source-data file and no SD column exist ({_Q_DATA_AVAILABILITY}"
                f"). The design is sourced ('{_Q_TRIPLICATE_FIG3}') and is carried in "
                "n_samples + sample_unit",
            )
            for field in ("titer_uncertainty", "titer_uncertainty_type")
        )
    return ProductTiterPhenotype(
        product=isoprenol(),
        titer=titer,
        titer_unit=unit,
        titer_uncertainty=uncertainty,
        titer_uncertainty_type=(
            UncertaintyType.sample_sd if uncertainty is not None else None
        ),
        n_samples=int(N_REPLICATES.value),
        sample_unit=SampleUnit.biological_replicate,
        quantification_method=str(QUANTIFICATION.value),
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# The twenty released titers
# --------------------------------------------------------------------------- #
class TiterRow(BaseModel):
    """One released isoprenol titer: its strain, its condition and its statement."""

    family: str
    strain: str
    condition: str
    titer: float
    unit: Literal["ug/mL", "g/L"]
    uncertainty: float | None
    time_hours: float | None
    reference: str
    quote: str
    page: str


#: The two released units, as the Literal the row models declare. A mg/L statement is
#: stored as ug/mL because the two are exactly equal, so no source value is scaled.
_MG_PER_L: Literal["ug/mL"] = "ug/mL"
_G_PER_L: Literal["g/L"] = "g/L"

TITER_ROWS: tuple[TiterRow, ...] = (
    # Results 3.2: the GSMM-guided knockout rounds, on pXW1, in EZ rich.
    TiterRow(
        family="ezrich_knockouts",
        strain="XW12",
        condition="tube_ezrich_glc20",
        titer=164.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=48.0,
        reference="ezrich",
        quote=_Q_XW12,
        page=_RESULTS_KNOCKOUTS,
    ),
    TiterRow(
        family="ezrich_knockouts",
        strain="XW15",
        condition="tube_ezrich_glc20",
        titer=241.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=48.0,
        reference="ezrich",
        quote=_Q_XW13_XW14_RANGE,
        page=_RESULTS_KNOCKOUTS,
    ),
    TiterRow(
        family="ezrich_knockouts",
        strain="XW16",
        condition="tube_ezrich_glc20",
        titer=19.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=48.0,
        reference="ezrich",
        quote=_Q_XW16,
        page=_RESULTS_KNOCKOUTS,
    ),
    TiterRow(
        family="ezrich_knockouts",
        strain="XW17",
        condition="tube_ezrich_glc20",
        titer=12.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=48.0,
        reference="ezrich",
        quote=_Q_XW17_XW18,
        page=_RESULTS_KNOCKOUTS,
    ),
    TiterRow(
        family="ezrich_knockouts",
        strain="XW18",
        condition="tube_ezrich_glc20",
        titer=13.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=48.0,
        reference="ezrich",
        quote=_Q_XW17_XW18,
        page=_RESULTS_KNOCKOUTS,
    ),
    TiterRow(
        family="ezrich_knockouts",
        strain="XW19",
        condition="tube_ezrich_glc20",
        titer=114.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=48.0,
        reference="ezrich",
        quote=_Q_XW19,
        page=_RESULTS_KNOCKOUTS,
    ),
    # Results 3.3: the plasmid and phosphatase panel, in EZ rich.
    TiterRow(
        family="ezrich_pathway",
        strain="IY782",
        condition="tube_ezrich_glc20",
        titer=80.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=48.0,
        reference="ezrich",
        quote=_Q_IY782,
        page=_RESULTS_PATHWAY,
    ),
    TiterRow(
        family="ezrich_pathway",
        strain="IY721",
        condition="tube_ezrich_glc20",
        titer=395.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=48.0,
        reference="ezrich",
        quote=_Q_IY721_IY846,
        page=_RESULTS_PATHWAY,
    ),
    TiterRow(
        family="ezrich_pathway",
        strain="IY846",
        condition="tube_ezrich_glc20",
        titer=536.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=48.0,
        reference="ezrich",
        quote=_Q_IY721_IY846,
        page=_RESULTS_PATHWAY,
    ),
    TiterRow(
        family="ezrich_pathway",
        strain="IY939",
        condition="tube_ezrich_glc20",
        titer=1111.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=48.0,
        reference="ezrich",
        quote=_Q_IY939,
        page=_RESULTS_PATHWAY,
    ),
    # Results 3.4: M9 glucose minimal medium.
    TiterRow(
        family="m9_minimal",
        strain="IY1452",
        condition="tube_m9_glc20",
        titer=762.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=None,
        reference="m9",
        quote=_Q_M9_TITERS,
        page=_RESULTS_M9,
    ),
    TiterRow(
        family="m9_minimal",
        strain="IY1252",
        condition="tube_m9_glc20",
        titer=816.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=48.0,
        reference="m9",
        quote=_Q_M9_TITERS,
        page=_RESULTS_M9,
    ),
    TiterRow(
        family="m9_minimal",
        strain="IY1252",
        condition="tube_m9_glc20",
        titer=259.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=72.0,
        reference="m9",
        quote=_Q_M9_TITERS,
        page=_RESULTS_M9,
    ),
    # Results 3.6: the 2 L bioreactor fed-batch runs, the only released uncertainties.
    TiterRow(
        family="fedbatch",
        strain="IY1245",
        condition="bioreactor_m9mod_fedbatch",
        titer=0.5,
        unit=_G_PER_L,
        uncertainty=0.1,
        time_hours=72.0,
        reference="fedbatch",
        quote=_Q_FEDBATCH_IY1245,
        page=_RESULTS_FEDBATCH,
    ),
    TiterRow(
        family="fedbatch",
        strain="IY1262",
        condition="bioreactor_m9mod_fedbatch",
        titer=2.3,
        unit=_G_PER_L,
        uncertainty=0.3,
        time_hours=96.0,
        reference="fedbatch",
        quote=_Q_FEDBATCH_IY1262,
        page=_RESULTS_FEDBATCH,
    ),
    TiterRow(
        family="fedbatch",
        strain="IY1452",
        condition="bioreactor_m9mod_fedbatch",
        titer=3.5,
        unit=_G_PER_L,
        uncertainty=0.3,
        time_hours=96.0,
        reference="fedbatch",
        quote=_Q_FEDBATCH_IY1452,
        page=_RESULTS_FEDBATCH,
    ),
    TiterRow(
        family="fedbatch",
        strain="IY1485",
        condition="bioreactor_m9mod_fedbatch",
        titer=2.4,
        unit=_G_PER_L,
        uncertainty=0.3,
        time_hours=96.0,
        reference="fedbatch",
        quote=_Q_FEDBATCH_IY1485,
        page=_RESULTS_FEDBATCH,
    ),
    # Results 3.7: sorghum hydrolysate against glucose, in modified M9 tubes.
    TiterRow(
        family="hydrolysate",
        strain="IY1452",
        condition="tube_m9mod_glc20",
        titer=841.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=72.0,
        reference="hydrolysate",
        quote=_Q_HYDROLYSATE,
        page=_RESULTS_HYDROLYSATE,
    ),
    TiterRow(
        family="hydrolysate",
        strain="IY1452",
        condition="tube_m9mod_hydrolysate_5",
        titer=409.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=72.0,
        reference="hydrolysate",
        quote=_Q_HYDROLYSATE,
        page=_RESULTS_HYDROLYSATE,
    ),
    TiterRow(
        family="hydrolysate",
        strain="IY1452",
        condition="tube_m9mod_hydrolysate_10",
        titer=432.0,
        unit=_MG_PER_L,
        uncertainty=None,
        time_hours=72.0,
        reference="hydrolysate",
        quote=_Q_HYDROLYSATE,
        page=_RESULTS_HYDROLYSATE,
    ),
)


class ReferenceBaseline(BaseModel):
    """One family's reference: the strain, titer and condition its comparisons use.

    ``titer`` is ``None`` where the source states the family's denominator only as a
    RATIO and never prints the parent's own number. That is a typed gap on the stored
    reference, not a licence to substitute a family member's value.
    """

    key: str
    strain: str
    titer: float | None
    unit: Literal["ug/mL", "g/L"]
    uncertainty: float | None
    condition: str
    quote: str
    page: str
    note: str


#: The baseline each family is measured against, and the statement that makes it the
#: paper's own fold-change denominator. The M9 family is the one substitution, recorded.
REFERENCE_BASELINES: dict[str, ReferenceBaseline] = {
    "ezrich": ReferenceBaseline(
        key="ezrich",
        strain="XW11",
        titer=104.0,
        unit=_MG_PER_L,
        uncertainty=None,
        condition="tube_ezrich_glc20",
        quote=_Q_XW11_BASELINE,
        page=_RESULTS_KNOCKOUTS,
        note="the paper's own EZ-rich denominator, used by every fold-change in "
        "Results 3.2 and 3.3 ('1.6-fold compared with the starter strain XW11', '2.3-"
        "fold increase from the XW11 strain', and the Conclusions' '10-fold improvement "
        "vs. the starting strain', which is 1111/104). Banerjee ATTRIBUTES the 104 mg/L "
        "to Wang et al. 2022, so it is this dataset's reference and NOT one of its "
        "records: the measurement is Wang 2022's",
    ),
    "m9": ReferenceBaseline(
        key="m9",
        strain="IY1245",
        titer=None,
        unit=_MG_PER_L,
        uncertainty=None,
        condition="tube_m9_glc20",
        quote=_Q_M9_TITERS,
        page=_RESULTS_M9,
        note="A TYPED GAP, because the parental titer was never released. The paper's "
        "denominator for this family is its wild type, stated only as a ratio ('over "
        "4-fold higher than WT (IY1245)'), and IY1245's M9 titer appears nowhere in the "
        "mirror. Reconstructing 762/4 would invent a number, and putting a family "
        "member's own value here (the 762 mg/L of IY1452, which an earlier build did) "
        "would state a parental measurement nobody made. So the reference names the "
        "strain the paper names, its genotype and its condition are stored, and its "
        "titer is a ProvenanceGap. The three released M9 titers stay records either "
        "way: the gap is on the denominator, not on them",
    ),
    "fedbatch": ReferenceBaseline(
        key="fedbatch",
        strain="IY1245",
        titer=0.5,
        unit=_G_PER_L,
        uncertainty=0.1,
        condition="bioreactor_m9mod_fedbatch",
        quote=_Q_FEDBATCH_IY1245,
        page=_RESULTS_FEDBATCH,
        note="the paper names IY1245 'the IY1245 control strain' for this family. It is "
        "both a record and the reference it is copied into: a measured sample is not "
        "spent by being used as a baseline",
    ),
    "hydrolysate": ReferenceBaseline(
        key="hydrolysate",
        strain="IY1452",
        titer=841.0,
        unit=_MG_PER_L,
        uncertainty=None,
        condition="tube_m9mod_glc20",
        quote=_Q_HYDROLYSATE,
        page=_RESULTS_HYDROLYSATE,
        note="the glucose-only arm, which the same sentence makes the comparison "
        "('Isoprenol production was lower with sorghum hydrolysate'). Both a record and "
        "the reference of its own family",
    ),
}

#: The record count this loader states: 6 + 4 + 3 + 4 + 3.
EXPECTED_RECORDS = len(TITER_ROWS)


class NotStored(BaseModel):
    """One released quantity this loader does not store, with its count and reason."""

    quantity: str
    where: str
    values: int
    reason: str
    issue: int | None = None


def not_stored() -> tuple[NotStored, ...]:
    """Everything the release carries that no record holds, each with its count."""
    return (
        NotStored(
            quantity="fed-batch OD600 maxima, with their standard deviations",
            where="Results 3.6 (Fig. 8a)",
            values=len(OD600_NOT_STORED),
            reason=OD600_REASON,
            issue=776,
        ),
        NotStored(
            quantity="XW17 residual glucose at 48 h in EZ rich",
            where="Results 3.2 (Supplementary Fig. 1)",
            values=1,
            reason=str(RESIDUAL_GLUCOSE_NOT_STORED.note),
        ),
        NotStored(
            quantity="the XW13 / XW14 titer range",
            where="Results 3.2 (Fig. 3b)",
            values=2,
            reason=str(XW13_XW14_RANGE_NOT_STORED.note),
        ),
        NotStored(
            quantity="specific growth rates of the mvaB-deleted strains against WT",
            where="Results 3.5 (Fig. 6b)",
            values=2,
            reason=str(GROWTH_RATES_NOT_STORED.note),
        ),
        NotStored(
            quantity="the IY1485 gacA deletion, as a typed perturbation",
            where="Results 3.6",
            values=len(GENOTYPE_NOT_TYPED),
            reason=GENOTYPE_NOT_TYPED[0][2],
        ),
        NotStored(
            quantity="every per-strain titer, OD600 and glucose SERIES",
            where="Figs. 3b, 4b, 4c, 5, 6a, 8a, 8b, 9a, 9b and Supplementary Figs. 1, "
            "2, 4, 5",
            values=0,
            reason="released as bars and curves with no companion table, and the ESM "
            "holds no measurement file at all (its six files are two COBRA models, two "
            "design workbooks, a notebook and the Supplementary Information). Reading a "
            "value off a bar is forbidden, so the count of unreleased values is not "
            "even knowable from our documentation",
        ),
    )


class BuildAccounting(BaseModel):
    """What entered the build, what was stored, and every difference with its reason."""

    dataset: str
    source_statements: int
    candidate_records: int
    kept_records: int
    dropped_records: int
    quotes_verified: int
    not_stored: tuple[NotStored, ...]
    notes: tuple[str, ...]

    def check(self) -> None:
        """The stored count must equal the candidates: nothing is dropped silently."""
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_records} "
                f"dropped != {self.candidate_records} candidates"
            )
        if self.dropped_records:
            raise RuntimeError(
                f"{self.dataset}: {self.dropped_records} records were dropped; every "
                "released titer is a record, so a drop is a defect"
            )


def _write_accounting(accounting: BuildAccounting, preprocess_dir: str) -> None:
    """Write ``preprocess/build_accounting.json`` after checking it adds up."""
    accounting.check()
    with open(osp.join(preprocess_dir, "build_accounting.json"), "w") as handle:
        handle.write(accounting.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class IsoprenolTiterBanerjee2024Dataset(ExperimentDataset):
    """Banerjee 2024 isoprenol titers: the 20 the article body states numerically."""

    organism = Organism.pputida

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"
    #: Every locus tag this campaign states must resolve to a locus of the pinned
    #: assembly. Measured on the pinned bytes: 11 of 11 (8 named deletion targets and
    #: the 3 pha-cluster tags). ``gacA`` is NOT in this set: the paper states no tag for
    #: it, and the reconciliation is a check on the tags the paper DID release, never a
    #: route for finding one it did not.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 1.0

    def __init__(
        self,
        root: str = "data/torchcell/isoprenol_titer_banerjee2024",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome checks the locus tags the paper released."""
        self.pputida_genome = pputida_genome
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
        """The deposited Supplementary Information PDF and its OCR markdown."""
        return ["si1.pdf", "si1.md"]

    def download(self) -> None:
        """Link the deposited SI artifacts into ``raw/`` after verifying their pins."""
        _link_mirror_files(
            self.raw_dir,
            ((SI_PDF, "si1.pdf", SI_PDF_SHA256), (SI_MD, "si1.md", SI_MD_SHA256)),
        )
        log.info("Banerjee 2024 SI artifacts linked into %s", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            from torchcell.datasets.bacteria_common import bacterial_genome

            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    @post_process
    def process(self) -> None:
        """Write one titer record per released statement; write LMDB."""
        verify_raw_files(
            self.raw_dir, {"si1.pdf": SI_PDF_SHA256, "si1.md": SI_MD_SHA256}
        )
        n_quotes = verify_quotes()
        self._assert_titer_oracles()

        genome = self._genome()
        tags = sorted(
            {str(value.value) for value in DELETION_LOCUS_TAGS.values()}
            | set(PHA_LOCUS_TAGS)
        )
        stored, report = reconcile_locus_tags(genome, pd.Series(tags), label=self.name)
        report.require_resolved(self.MIN_RESOLVED_FRACTION)
        if report.outside_namespace:
            raise RuntimeError(
                f"{self.name}: locus tags outside {KT2440_NAMESPACE}: "
                f"{report.outside_namespace}"
            )
        if list(stored) != tags:
            raise RuntimeError(
                "a stated locus tag was remapped by the reconciliation; these are "
                f"current tags of the pinned assembly: {dict(zip(stored, tags))}"
            )

        reference_genome = assembly_reference(PARENT_STRAIN)
        references = {
            baseline.key: ProductTiterExperimentReference(
                dataset_name=self.name,
                genome_reference=reference_genome,
                environment_reference=environment(CONDITIONS[baseline.condition]),
                phenotype_reference=titer_phenotype(
                    baseline.titer,
                    ConcentrationUnit(baseline.unit),
                    uncertainty=baseline.uncertainty,
                ),
            )
            for baseline in REFERENCE_BASELINES.values()
        }
        pub = publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for row in tqdm(TITER_ROWS, desc="banerjee2024-titer"):
                experiment = ProductTiterExperiment(
                    dataset_name=self.name,
                    genotype=strain_genotype(row.strain),
                    environment=environment(
                        CONDITIONS[row.condition], duration_hours=row.time_hours
                    ),
                    phenotype=titer_phenotype(
                        row.titer,
                        ConcentrationUnit(row.unit),
                        uncertainty=row.uncertainty,
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(
                        experiment, references[row.reference], pub, itxn
                    ),
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame([row.model_dump() for row in TITER_ROWS]).to_csv(
            osp.join(self.preprocess_dir, "titer_rows.csv"), index=False
        )
        pd.DataFrame(
            [
                {
                    "strain": strain,
                    "od600": value,
                    "sd": sd,
                    "time_hours": hours,
                    "reason": OD600_REASON,
                }
                for strain, value, sd, hours in OD600_NOT_STORED
            ]
        ).to_csv(osp.join(self.preprocess_dir, "od600_not_stored.csv"), index=False)
        with open(
            osp.join(self.preprocess_dir, "condition_provenance.json"), "w"
        ) as handle:
            json.dump(
                {
                    key: [value.model_dump(mode="json") for value in values]
                    for key, values in CONDITION_PROVENANCE.items()
                },
                handle,
                indent=2,
            )
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_statements=len(TITER_ROWS),
                candidate_records=len(TITER_ROWS),
                kept_records=idx,
                dropped_records=len(TITER_ROWS) - idx,
                quotes_verified=n_quotes,
                not_stored=not_stored(),
                notes=(
                    "nothing is dropped: every isoprenol titer this paper states "
                    "numerically is a record, across all five condition families",
                    "the ESM was captured in full (six Elsevier files) and holds NO "
                    f"measurement table; data availability reads "
                    f"{_Q_DATA_AVAILABILITY!r}, so the titers are sourced constants "
                    "quoting the pinned paper.md OCR and nothing is read off a bar",
                    f"{n_quotes} quotes were re-read verbatim from the pinned mirrors "
                    "before any value was used",
                    "the M9-minimal family's reference carries a typed gap on its "
                    "titer: " + REFERENCE_BASELINES["m9"].note,
                    "XW11's 104 mg/L is this dataset's EZ-rich reference and not one of "
                    "its records, because Banerjee attributes that measurement to Wang "
                    "et al. 2022",
                    "the EZ-rich records carry no inducer perturbation: Methods 2.5 "
                    "states '0.5 mM IPTG or arabinose (2%)' as a disjunction and never "
                    "says which strain received which",
                    "gacA is stored under the symbol the paper released, because the "
                    "paper states no locus tag for it and the annotation is a separate "
                    "layer that is not consulted to supply one",
                ),
            ),
            self.preprocess_dir,
        )
        log.info(
            "Banerjee2024 titer: %d records over %d strains and %d conditions; "
            "%d locus tags all current; %d quotes verbatim",
            idx,
            len({row.strain for row in TITER_ROWS}),
            len({row.condition for row in TITER_ROWS}),
            len(tags),
            n_quotes,
        )

    @staticmethod
    def _assert_titer_oracles() -> None:
        """Independent checks that the stored constants are the paper's own numbers.

        Each is a relation the ARTICLE states in words, recomputed from the stored
        values: if a transcription slipped, the relation breaks and the build stops.
        """
        by_strain = {
            (row.strain, row.condition, row.time_hours): row for row in TITER_ROWS
        }
        # The EZ-rich denominator is a RELEASED number, unlike the M9 family's, so every
        # fold-change oracle below rests on it. Asserted rather than assumed: if it ever
        # became a typed gap the oracles would silently stop dividing by a measurement.
        ezrich = REFERENCE_BASELINES["ezrich"].titer
        if ezrich is None:
            raise RuntimeError(
                "the EZ-rich reference titer is a typed gap, so the fold-change oracles "
                "have no released denominator to check against"
            )
        xw12 = by_strain[("XW12", "tube_ezrich_glc20", 48.0)].titer
        xw15 = by_strain[("XW15", "tube_ezrich_glc20", 48.0)].titer
        iy939 = by_strain[("IY939", "tube_ezrich_glc20", 48.0)].titer
        iy721 = by_strain[("IY721", "tube_ezrich_glc20", 48.0)].titer
        iy846 = by_strain[("IY846", "tube_ezrich_glc20", 48.0)].titer
        # The tolerance is the precision the SENTENCE carries, not a fitted slack: a
        # fold-change written to two significant figures is checked at 5%, and the
        # Conclusions' "10-fold" is one significant figure, so 1111/104 = 10.68 is the
        # same statement.
        checks: tuple[tuple[str, float, float, float], ...] = (
            # "which increased 1.6-fold compared with the starter strain XW11"
            ("XW12 / XW11 = 1.6-fold", xw12 / ezrich, 1.6, 0.05),
            # "a 2.3-fold increase from the XW11 strain and 1.5-fold increase from XW12"
            ("XW15 / XW11 = 2.3-fold", xw15 / ezrich, 2.3, 0.05),
            ("XW15 / XW12 = 1.5-fold", xw15 / xw12, 1.5, 0.05),
            # "improved the isoprenol production by 1.3-fold when compared to IY721"
            ("IY846 / IY721 = 1.3-fold", iy846 / iy721, 1.3, 0.05),
            # Conclusions: "the highest titer of isoprenol in P. putida at 1.1 g/L; a
            # 10-fold improvement vs. the starting strain"
            ("IY939 / XW11 = 10-fold", iy939 / ezrich, 10.0, 0.10),
        )
        for label, measured, stated, tolerance in checks:
            if abs(measured - stated) > tolerance * stated:
                raise RuntimeError(
                    f"{label}: the stored values give {measured:.3f}, which is not the "
                    f"stated {stated} within {tolerance:.0%}; a titer constant no "
                    "longer matches the paper's own fold-change"
                )
        # "The highest isoprenol titer observed was 816 mg/L for the IY1252 strain at
        # 48 h but that reduced by about 70% - 259 mg/L at 72 h"
        high = by_strain[("IY1252", "tube_m9_glc20", 48.0)].titer
        low = by_strain[("IY1252", "tube_m9_glc20", 72.0)].titer
        drop = (high - low) / high
        if abs(drop - 0.70) > 0.03:
            raise RuntimeError(
                f"IY1252's 48 h and 72 h titers give a {drop:.1%} drop, not the stated "
                "about 70%"
            )
        # The fed-batch maximum is the paper's headline titer, 3.5 g/L.
        fedbatch = [row for row in TITER_ROWS if row.family == "fedbatch"]
        if max(row.titer for row in fedbatch) != 3.5:
            raise RuntimeError(
                "the fed-batch maximum is not 3.5 g/L, which is this paper's headline "
                "titer and its abstract's own number"
            )
        if any(row.uncertainty is None for row in fedbatch):
            raise RuntimeError(
                "every fed-batch titer releases a +- number; one is missing, so the "
                "uncertainty would be stored as a gap where the paper printed a value"
            )
        if any(
            row.uncertainty is not None
            for row in TITER_ROWS
            if row.family != "fedbatch"
        ):
            raise RuntimeError(
                "only the fed-batch family releases an uncertainty number; another "
                "family carries one, which the paper does not state"
            )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """No tabular preprocessing: the titers are sourced constants, not a data file."""
        return df

    def create_experiment(self) -> None:
        """Records are written by :meth:`process`; the hook exists for the base class."""
        raise NotImplementedError(
            "IsoprenolTiterBanerjee2024Dataset writes its records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification, L0 to L4
# --------------------------------------------------------------------------- #
def verify_build(dataset_root: str, data_root: str | None = None) -> VerificationReport:
    """Run L0 to L4 over a built tree and write ``preprocess/verification_report.json``.

    L4 is the cross-source level and here it is a real join in BOTH directions a
    prose-sourced dataset can be joined: every stored titer against the ledger the build
    wrote, and every quote against the sha256-pinned mirrors it cites. The genome
    reference rule REPORTS rather than gates, per the owner's rule: ``gacA`` carries no
    released locus tag, and that is a finding about the source, not a defect in the store.
    """
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    ledger = pd.read_csv(osp.join(dataset_root, "preprocess", "titer_rows.csv"))
    experiments = [record["experiment"] for record in records]
    phenotypes = [exp["phenotype"] for exp in experiments]
    units = {p["titer_unit"] for p in phenotypes}
    media = {exp["environment"]["media"]["name"] for exp in experiments}
    with_uncertainty = [p for p in phenotypes if p["titer_uncertainty"] is not None]
    expected_uncertainty = [row for row in TITER_ROWS if row.uncertainty is not None]
    report = VerificationReport(
        dataset_name=IsoprenolTiterBanerjee2024Dataset.__name__,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method=(
                "isoprenol titer by GC-FID, stored on the unit each statement releases: "
                "the mg/L statements of Results 3.2, 3.3, 3.4 and 3.7 as ug/mL, and the "
                "g/L fed-batch statements of Results 3.6 as g/L. The ESM releases no "
                "measurement table, so every value is a sourced constant quoting the "
                "article body"
            ),
            page="Results 3.2; 3.3; 3.4; 3.6; 3.7",
            retrieved=SI_RETRIEVED_AT,
        ),
    )
    report.add(l0_structural(experiments, ProductTiterExperiment.model_validate))
    report.add(l1_count(len(records), EXPECTED_RECORDS))
    report.add(l2_value_fidelity([p["titer"] for p in phenotypes], minimum=0.0))
    report.add(
        l2_cross_method(
            sorted(float(p["titer"]) for p in phenotypes),
            sorted(float(row.titer) for row in TITER_ROWS),
            tol=0.0,
        )
    )
    report.add(
        l3_convention(
            "every_titer_is_on_the_unit_its_statement_released",
            units
            <= {ConcentrationUnit.ug_per_ml.value, ConcentrationUnit.g_per_l.value}
            and len(units) == len({row.unit for row in TITER_ROWS}),
            detail=f"units {sorted(units)}: the mg/L statements as ug/mL (1 mg/L is "
            "exactly 1 ug/mL) and the fed-batch g/L statements as g/L, so no source "
            "value was scaled",
        )
    )
    report.add(
        l3_convention(
            "uncertainty_is_stored_exactly_where_the_paper_prints_one",
            len(with_uncertainty) == len(expected_uncertainty)
            and all(
                p["titer_uncertainty_type"] == UncertaintyType.sample_sd.value
                and p["n_samples"] == int(N_REPLICATES.value)
                for p in with_uncertainty
            )
            and all(
                {"titer_uncertainty", "titer_uncertainty_type"}
                <= {gap["field"] for gap in p["provenance_gaps"]}
                for p in phenotypes
                if p["titer_uncertainty"] is None
            ),
            detail=f"{len(with_uncertainty)} records carry a released SD (the fed-batch "
            f"family); the other {len(phenotypes) - len(with_uncertainty)} carry a typed "
            "gap on both the number and its type, because their spread exists only as a "
            "figure error bar",
        )
    )
    report.add(
        l3_convention(
            "every_record_carries_the_medium_of_its_own_condition",
            len(media) == len({CONDITIONS[row.condition].medium for row in TITER_ROWS}),
            detail=f"{len(media)} distinct media over "
            f"{len({row.condition for row in TITER_ROWS})} stated conditions",
        )
    )
    report.add(
        l3_convention(
            "every_genotype_carries_exactly_one_isoprenol_plasmid",
            all(
                len(
                    {
                        pert["construct_name"]
                        for pert in exp["genotype"]["perturbations"]
                        if pert["perturbation_type"] == "heterologous_pathway"
                    }
                )
                == 1
                for exp in experiments
            ),
            detail="pXW1, pIY554, pIY603 or pIY670, one per strain as Supplementary "
            "Table 1 states them",
        )
    )
    report.add(
        l3_convention(
            "the_reference_is_the_pinned_KT2440_assembly",
            all(
                record["reference"]["genome_reference"]["assembly_accession"]
                == "GCA_000007565.2"
                for record in records
            ),
            detail="every record is an edit of KT2440 on GCA_000007565.2",
        )
    )
    report.add(_ledger_l4(records, ledger))
    report.add(_quote_l4(data_root))
    report.add(_genome_reference_l4(experiments))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def _ledger_l4(records: Sequence[dict[str, Any]], ledger: pd.DataFrame) -> LevelResult:
    """L4: every stored record against the build ledger, joined on value and time.

    LMDB keys are strings, so the store's iteration order is lexicographic and a
    positional join would silently pair the wrong rows. The key is
    ``(titer, time_hours)``, both released values, and its uniqueness across the 20
    rows is asserted rather than assumed.

    The STRAIN is deliberately not part of the key. It is not a stored field, and
    recovering it from the genotype is ambiguous here: IY1485's only edit above IY1452
    is the ΔgacA that no served leaf can hold, so the two hold identical genotypes and
    are told apart by their titers. A strain-keyed join would fail on that pair for a
    reason that is about the source's missing locus tag, not about the store.
    """
    if len(records) != len(ledger):
        raise AssertionError(
            f"{len(records)} records and {len(ledger)} ledger rows; the build ledger "
            "and the store disagree"
        )
    by_key: dict[tuple[float, float], float] = {}
    for _, row in ledger.iterrows():
        hours = row["time_hours"]
        key = (
            round(float(row["titer"]), 6),
            -1.0 if pd.isna(hours) else round(float(hours), 6),
        )
        if key in by_key:
            raise AssertionError(
                f"two ledger rows share the join key {key}; the key is no longer unique"
            )
        by_key[key] = float(row["titer"])
    shared: list[tuple[Hashable, float, float]] = []
    for record in records:
        experiment = record["experiment"]
        hours = experiment["environment"]["duration_hours"]
        key = (
            round(float(experiment["phenotype"]["titer"]), 6),
            -1.0 if hours is None else round(float(hours), 6),
        )
        if key not in by_key:
            raise AssertionError(
                f"a stored record has join key {key}, which no ledger row carries"
            )
        shared.append((key, float(experiment["phenotype"]["titer"]), by_key[key]))
    return l4_cross_source(shared, tol=1e-9).model_copy(
        update={"name": "stored_titers_vs_the_build_ledger"}
    )


def _quote_l4(data_root: str | None) -> LevelResult:
    """L4: every quote a stored value cites, re-read from the sha256-pinned mirrors."""
    checked = verify_quotes(data_root)
    return l3_convention(
        "every_quote_is_verbatim_in_its_pinned_mirror",
        checked == len(every_quote()),
        detail=f"{checked} quotes re-read from paper.md ({PAPER_MD_SHA256[:12]}), "
        f"si/si1.md ({SI_MD_SHA256[:12]}) and the deferred-to Carruthers 2025 mirror "
        f"({CARRUTHERS_PAPER_MD_SHA256[:12]})",
    ).model_copy(update={"level": Level.L4})


def _genome_reference_l4(experiments: Sequence[dict[str, Any]]) -> LevelResult:
    """L4: the released locus tags, as a REPORT rather than a gate.

    Per the owner's rule the genome tier never gates a store: this names what the
    records carry and what the source did not release, and it passes either way.
    """
    tags = {
        pert["systematic_gene_name"]
        for exp in experiments
        for pert in exp["genotype"]["perturbations"]
        if pert["perturbation_type"] == "deletion_strain"
    }
    return l3_convention(
        "released_locus_tags_are_reported_not_gated",
        True,
        detail=f"{len(tags)} distinct released locus tags over the stored genotypes "
        f"({len(GENOTYPE_NOT_TYPED)} stated deletion carries no released tag: "
        f"{GENOTYPE_NOT_TYPED[0][1]} of {GENOTYPE_NOT_TYPED[0][0]}, so it has no typed "
        "perturbation and no gene edge). A finding here never fails the store",
    ).model_copy(update={"level": Level.L4})


def main() -> None:
    """Deposit the raw mirror, build the store and print the verification report."""
    from dotenv import load_dotenv

    load_dotenv()
    deposit_raw_mirror()
    root = osp.join(
        os.environ["DATA_ROOT"], "data/torchcell/isoprenol_titer_banerjee2024"
    )
    dataset = IsoprenolTiterBanerjee2024Dataset(root=root)
    print(f"{len(dataset)} records")
    dataset.close_lmdb()
    print(verify_build(root).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
