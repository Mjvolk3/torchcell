# torchcell/datasets/ecoli/foo2014
# [[torchcell.datasets.ecoli.foo2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/foo2014
# Test file: tests/torchcell/datasets/ecoli/test_foo2014.py
"""Foo 2014 isopentenol titers on an E. coli DH1 production chassis.

Foo et al. 2014 (mBio, doi:10.1128/mBio.01932-14) took the eight genes whose
overexpression shortened the isopentenol growth lag and expressed each one, singly, in
an isopentenol production strain. One dataset class,
:class:`IsopentenolTiterFoo2014Dataset`, serves every released titer as a
``ProductTiterExperiment``.

THE PRODUCT IS ISOPRENOL UNDER ITS OLDER NAME, AND THAT IS WHY THIS ROW MATTERS.
The paper's first sentence gives the IUPAC name outright, "(3-methyl-3-buten-1-ol) is
an important target compound", and ``compound_identity_table.json`` carries that string
as a synonym of ``isoprenol`` (InChIKey ``CPJRRXSHAYUTGL-UHFFFAOYSA-N``, PubChem CID
12988). The product is therefore resolved through that synonym, so these records join
the same compound entity every other isoprenol row uses rather than a name-matched stub:
``resolved_compound("isopentenol")`` returns an unresolved compound with a typed gap on
``inchikey``, which would have split the axis in two. Adding ``isopentenol`` to that
table's synonym list is a curation act and is raised in the PR, not taken here.

NINE RECORDS, AND THE REFERENCE TITER IS RELEASED. Table 1 gives the 48 h titer of the
eight tolerance strains, and its footnote b gives the control's: "Percent improvement in
titer is in comparison to result for PS+RFP at 48 h (834 5 mg liter-1)". That control is
a released number measured in the same experiment, so, unlike the sibling Yunus 2026 and
Kang 2026 isoprenol families, nothing here is refused for want of a denominator:
``ProductTiterExperimentReference.phenotype_reference`` carries 834 ug/mL. PS+RFP is also
a record of its own (Table S3 lists it among the nine production strains), exactly as
Kang 2026's baseline strain ``PIPA-AAT1`` is both the flask reference and a Table 1
record. The eight Table 1 titers are re-derivable from the control and the released
"Improvement in titer (%)" column, and an L4 level does that as a second reading of the
same table.

THE REPLICATE DESIGN IS SOURCED; THE UNCERTAINTY TYPE IS NOT RELEASED, SO THE NUMBER IS
NOT INGESTED. Table 1 footnote a states the design verbatim, "Averages from triplicates
after 48 h", and the Results restate it, "The production of isopentenol was quantified in
triplicate for all PS strains at 24, 48, and 72 h after 500 uM IPTG induction", so
``n_samples=3`` and ``sample_unit=biological_replicate`` are sourced. What the plus-minus
IS, a sample SD or a standard error, is stated NOWHERE. Measured over the whole mirror:
``paper.md``, the ``pdftotext`` layout of ``paper.pdf``, Text S1 (``si/si1.docx``), Tables
S1 to S4 (``si/si7.pdf``, ``si/si8.docx``, ``si/si9.docx``, ``si/si10.docx``) and the five
SI figure TIFFs carry exactly one statement of an error type, in the Fig. 3 caption:
"Error bars represent standard errors from at least 4 replicates" -- and that is the
maximum-growth-rate figure, a different measurement with a different replicate count, not
Table 1.

Back-solving does not break the tie either. The paper makes two statistical calls on this
table, that PS+SoxS and PS+NrdH "were statistically similar to that of PS+RFP" while the
other six increased the titer. Under a one-way ANOVA with Tukey HSD across the nine
strains -- the standard analysis of a nine-arm experiment -- BOTH readings reproduce all
eight calls, because pooling the nine spreads scales the critical comparison by the same
sqrt(3) that separates an SD reading from an SE reading. :data:`UNCERTAINTY_BACK_SOLVE`
records that measurement. ``UncertaintyType`` deliberately has no ``unknown`` member, so
the honest outcome is that ``titer_uncertainty`` and ``titer_uncertainty_type`` are typed
``ProvenanceGap``s on every record, with the plus-minus numbers kept per record in
``preprocess/titer_rows.csv`` so nothing released is lost.

ONE PERTURBATION PER RECORD, AND THE PRODUCTION PATHWAY IS BACKGROUND, NOT A GENOTYPE.
Table S3 names the chassis as "PS", "pJBEI-6830 + pJBEI-6833", and gives each plasmid's
composition, "pBbA5c-MevTsa-PMK-MK" and "pTrc99A-NudB-PMD". Those two plasmids are in
every one of the nine strains AND in the control, so they are the
:class:`BacterialStrainBackground`, carried verbatim in its ``genotype_statement`` and
``construction``. They are NOT typed as ``HeterologousPathwayPerturbation``, and the
reason is that the leaf's two required claims are unsourceable here:
``source_organism`` and ``is_heterologous`` are deferred whole to reference 6 (George et
al. 2014), which is not in the mirror, and the five tokens do not share an answer -- MevT
with its ``sa`` suffix is not an E. coli pathway while ``NudB`` is an E. coli gene, so a
single blanket value would state something the source does not. The five tokens stay
readable in the background strings; :data:`PRODUCTION_PATHWAY_NOT_TYPED` states the
decline in the build accounting.

What each record then carries is the one thing that distinguishes it: the pBbS5k-borne
gene, as a ``HeterologousPathwayPerturbation`` on the chassis. For the eight tolerance
genes that is an extra copy of a NATIVE gene -- the case the leaf's docstring names --
sourced to "Gene candidates to be tested for improving isopentenol tolerance were PCR
amplified using E. coli MG1655 genomic DNA", with the MG1655 b-number as
``systematic_gene_name``. For PS+RFP it is the reporter, whose origin the paper never
states, so ``source_organism`` is :data:`SOURCE_ORGANISM_UNREPORTED`.

DH1's OWN LESIONS ARE NOT STATED, AND ARE NOT INVENTED. The paper names the host once,
"For growth assays and isopentenol production, E. coli DH1 (ATCC 33849) was used", and
never writes its K-12 marker genotype. ``alleles`` is therefore empty: that is the
source's silence, not a claim that DH1 equals MG1655. MG1655 is the assembly pin because
it is the genome the overexpressed genes were amplified from and the namespace their
b-numbers belong to.

THE GEO SERIES IS OUT OF SCOPE FOR THIS CLASS. GSE53138 (GSM1282891 to GSM1282896,
platform GPL14649) holds six arrays of the MevT* strain, three with and three without
0.2% isopentenol. All six are ONE genotype, so the series is a transcriptome measured
against a dosed compound rather than a gene-perturbation record: a different experiment
class, a different reference, and a separate dataset if it is ever built. No raw reads
are read here, and no loader in this repo consumes raw reads. The same goes for Table S1's
microarray log2 and z scores, which are that series' processed summary on the same single
strain, and for Table S2's 40-candidate tolerance screen, whose growth-lag phenotype is
released only as the Fig. 4 and Fig. S2 curves with no number anywhere.

TITER UNITS: THE SOURCE'S mg/L IS STORED VERBATIM AS ``ug/mL``. ``ConcentrationUnit`` has
no ``mg/L`` member and 1 mg/L is exactly 1 ug/mL, so no arithmetic touches a source value.

THE ENVIRONMENT. ``ProductTiterExperiment.environment`` is annotated
``CultureEnvironment``, so the vessel travels on the record: MM9 (``MM9_FOO2014``,
already in ``torchcell.datamodels.media``), 30 C after induction, 5 mL of medium
inoculated to OD600 0.08, read at 48 h. The vessel type and the shaking speed of the
production culture are never stated and are typed gaps on ``CultureFormat``. Of the three
antibiotics the medium paragraph lists "where appropriate", only kanamycin is scoped to
this experiment by the source itself ("15 ug ml-1 for isopentenol production"), so it is
the only one typed; the sentence never says which plasmid carries which marker, and
:data:`ANTIBIOTICS_NOT_TYPED` records why the other two are left out.
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
from typing import Any, ClassVar, Literal
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    link_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import MM9_FOO2014
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialGeneNamespace,
    BacterialStrainBackground,
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
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    Publication,
    SampleUnit,
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
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome
from torchcell.verification.levels import (
    l0_structural,
    l1_count,
    l2_cross_method,
    l2_value_fidelity,
    l3_convention,
    l4_cross_source,
)
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1128/mBio.01932-14"
PMCID = "PMC4222104"
TITLE = (
    "Improving Microbial Biogasoline Production in Escherichia coli Using "
    "Tolerance Engineering"
)
CITATION_KEY = "fooImprovingMicrobialBiogasoline2014"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
#: The dev-tree root of this dataset, relative to ``DATA_ROOT``. It is the slug the
#: verification registry names and the slug the adapter conf is named after.
DATASET_SLUG = "isopentenol_titer_foo2014"
DEV_TREE_RELPATH = f"data/torchcell/{DATASET_SLUG}"

#: The PMC Article Datasets bucket prefix of this article's supplementary objects.
PMC_PREFIX = f"{PMCID}.1"
#: Table S3, "Description of plasmids and strains": the ONLY released data file this
#: loader consumes. It is the publisher's ``mbo005142049st3.docx``, mirrored as
#: ``si/si9.docx``.
SI_TABLE3_FILENAME = "si9.docx"
SI_TABLE3_REL = f"si/{SI_TABLE3_FILENAME}"
SI_TABLE3_SOURCE_FILENAME = "mbo005142049st3.docx"
SI_TABLE3_SHA256 = "f8196b5ad05ce7520694adeb0238181ffc9773c57843a1400e2f9f07320bb4d1"
SI_RETRIEVED_AT = "2026-10-07"

#: The OCR of the article body, in the literature mirror. Table 1 lives in the article,
#: not in any released data file, so every titer quote below cites these pinned bytes.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "b24baad46bdf488cf93a7e59c51eceb2af9ba4ad565459130897a6201cd62407"

GEO_SERIES = "GSE53138"
GEO_SAMPLES = ("GSM1282891", "GSM1282896")
GEO_PLATFORM = "GPL14649"

NAMESPACE: BacterialGeneNamespace = "ecoli_k12_mg1655_bnumber"
REFERENCE_STRAIN: Literal["MG1655"] = "MG1655"
#: Table S3's own name for the chassis every record is an edit of.
CHASSIS_STRAIN = "PS"
HOST_SPECIES = "Escherichia coli"
PRODUCT_SYNONYM = "3-methyl-3-buten-1-ol"
PRODUCT_SOURCE_NAME = "isopentenol"
QUANTIFICATION_METHOD = "gas chromatography"
PATHWAY_TOLERANCE = "isopentenol tolerance by single-gene overexpression"
PATHWAY_REPORTER = "rfp titer control"
PRODUCTION_TEMPERATURE_C = 30.0
PRODUCTION_DURATION_HOURS = 48.0
PRODUCTION_VOLUME_UL = 5000.0
PRODUCTION_INOCULUM_OD600 = 0.08
IPTG_UM = 500.0
KANAMYCIN_UG_PER_ML = 15.0
EXPECTED_RECORDS = 9

#: ``GeneAdditionPerturbation.source_organism`` is a required ``str`` on a leaf with no
#: ``provenance_gaps`` field, so an unreported origin cannot be typed as an absence. The
#: honest value is this explicit sentinel, the same one the Menasalvas 2025 loader uses.
SOURCE_ORGANISM_UNREPORTED = "unreported"


# --------------------------------------------------------------------------- #
# Verbatim quotes. Every one is a substring of the sha256-pinned mirror bytes, and
# verify_paper_quotes() (the article) and read_production_plasmids() +
# read_production_strains() (the deposited Table S3) re-read them before a value is used.
# --------------------------------------------------------------------------- #
_RESULTS_PRODUCTION = "Results, 'Eight tolerance genes ... isopentenol production'"
_METHODS_STRAINS = "Methods, 'Strains, plasmids, oligonucleotides, chemicals, media'"
_METHODS_CLONING = "Methods, 'Cloning and expression of genes'"
_METHODS_PRODUCTION = "Methods, 'Isopentenol production titer in whole culture'"
_TABLE1 = "Table 1, 'Isopentenol tolerance-enhancing genes'"
_SI_TABLE3 = "Table S3, 'Description of plasmids and strains'"

#: Table 1's rows, verbatim. The stored titer is PARSED out of these bytes, so a changed
#: OCR cannot drift away from the value without the quote check failing first.
_Q_TABLE1_ROWS: tuple[str, ...] = (
    "<tr><td>soxS</td><td>DNA-binding transcriptional dual regulator</td>"
    "<td>3.51, 2.81</td><td>838 ± 29</td><td>0</td><td>28, 42, 43</td></tr>",
    "<tr><td>nrdH</td><td>Glutaredoxin-like protein</td><td>2.17,2.08</td>"
    "<td>860 ± 3</td><td>3</td><td>44</td></tr>",
    "<tr><td>mdlB</td><td>Predicted multidrug ABC transporter</td><td>3.75, 2.31</td>"
    "<td>931 ± 16</td><td>12</td><td>25</td></tr>",
    "<tr><td>ibpA</td><td>Heat shock chaperone</td><td>4.80, 2.82</td>"
    "<td>967 ± 23</td><td>16</td><td>26, 27</td></tr>",
    "<tr><td>gidB</td><td>Glucose-inhibited division protein B</td><td>5.32, 2.64</td>"
    "<td>965 ± 1</td><td>16</td><td>29, 45, 46</td></tr>",
    "<tr><td>fpr</td><td>Ferredoxin-NADP reductase</td><td>2.30, 2.07</td>"
    "<td>989 ± 10</td><td>19</td><td>28</td></tr>",
    "<tr><td>yqhD</td><td>Alcohol dehydro-genase, NAD(P) dependent</td>"
    "<td>2.40, 2.39</td><td>994 ± 13</td><td>19</td><td>14, 30, 31</td></tr>",
    "<tr><td>metR</td><td>DNA-binding transcriptional activator</td><td>2.57,2.55</td>"
    "<td>1,290 ± 20</td><td>55</td><td>32, 33</td></tr>",
)
#: Footnote b. The ``\x05`` is what the OCR made of the PDF's plus-minus glyph in this
#: one line (``pdftotext`` reads the same cell as "834 ⫾ 5"); the quote is kept exactly
#: as the pinned bytes read, since a quote that is cleaned up is no longer a quote.
_Q_CONTROL_TITER = (
    "b Percent improvement in titer is in comparison to result for "
    "$\\mathrm { P S } + \\mathrm { R F P }$ at $4 8 \\mathrm { h }$ "
    "(834 \x05 5 mg liter-1)."
)
_Q_TRIPLICATE_FOOTNOTE = "a Averages from triplicates after $4 8 \\mathrm { { h } }$ ."
_Q_TRIPLICATE_RESULTS = (
    "The production of isopentenol was quantified in triplicate for all PS strains at "
    "24, 48, and $^ { 7 2 \\mathrm { ~ h ~ } }$ after $5 0 0 ~ \\mu \\mathrm { M }$ "
    "IPTG induction (Fig. 5)."
)
#: The ONLY error-type statement in the article or its SI, and it is about Fig. 3's
#: maximum growth rates, not about Table 1's titers.
_Q_ONLY_ERROR_TYPE_STATEMENT = (
    "Error bars represent standard errors from at least 4 replicates."
)
#: The drop cap of "Isopentenol" is missing from the OCR of this opening sentence, so
#: the quote starts after it.
_Q_PRODUCT_IUPAC = (
    "(3-methyl-3-buten-1-ol) is an important target compound as a precursor to "
    "pharmaceuticals"
)
_Q_HOST = (
    "For growth assays and isopentenol production, E. coli DH1 (ATCC 33849) was used."
)
_Q_CHASSIS = (
    "The isopentenol production strains all harbor plasmids pJBEI-6830 and "
    "pJBEI-6833 (6)."
)
_Q_PBBS5K = (
    "This strain was transformed with tolerance-conferring genes cloned into vector "
    "pBbS5k for production assays."
)
_Q_PROMOTER = "Both vectors have a lacUV5 promoter and kanamycin resistance marker"
_Q_AMPLIFIED_FROM_MG1655 = (
    "Gene candidates to be tested for improving isopentenol tolerance were PCR "
    "amplified using E. coli MG1655 genomic DNA"
)
_Q_EIGHT_GENES = (
    "Of the 40 candidates screened, 8 candidates were found to confer tolerance to "
    "exogenously added isopentenol: metR, ipbA, nrdH, soxS, mdlB, fpr, gidB, and yqhD."
)
_Q_VOLUME = (
    "diluted to an $\\mathrm { O D } _ { 6 0 0 }$ of ${ \\sim } 0 . 0 8$ in "
    "$5 \\mathrm { m l }$ of fresh medium"
)
_Q_INDUCTION = (
    "Expression of the production and tolerance genes was then simultaneously induced "
    "with $5 0 0 \\mu \\mathrm { M }$ IPTG."
)
_Q_TEMPERATURE = (
    "The cultures were subsequently grown at $3 0 ^ { \\circ } \\mathrm { C }$ with "
    "shaking for isopentenol production."
)
_Q_KANAMYCIN = (
    "kanamycin $1 5 \\mu \\mathrm { g } \\mathrm { m l } ^ { - 1 }$ for isopentenol "
    "production or $3 0 \\mu \\mathrm { g } \\mathrm { m l } ^ { - 1 }$ otherwise"
)
_Q_ANTIBIOTICS = (
    "Where appropriate, chloramphenicol $( 5 0 ~ \\mu \\mathrm { g } ~ \\mathrm "
    "{ m l ^ { - 1 } } ,$ ), carbenicillin $( 1 0 0 \\mu \\mathrm { g } \\mathrm "
    "{ m l ^ { - 1 } } ,$ , and kanamycin"
)
_Q_GC = (
    "analyzed by gas chromatography, as described previously (2), to quantify the "
    "isopentenol production titer"
)
_Q_STATISTICALLY_SIMILAR = (
    "At the 48-h time point, the isopentenol titer in the $\\mathrm { P S } + "
    "\\mathrm { S o x S }$ and $\\mathrm { P S } + \\mathrm { N r d H }$ strains were "
    "statistically similar to that of $\\mathrm { P S } + \\mathrm { R F P }$ ."
)
_Q_GEO = (
    "Complete data obtained in this work are available in the GEO database under "
    f"accession number {GEO_SERIES}"
)
_Q_MICROARRAY_SAMPLES = (
    "The strain was grown in MM9 with $1 \\%$ glucose, $2 \\mathrm { m M }$ $\\mathrm "
    "{ M g , }$ and $0 . 5 ~ \\mathrm { m M }$ IPTG and treated with $0 \\%$ or $0 . 2 "
    "\\%$ isopentenol added exogenously."
)

#: Every quote above that must be verbatim in the pinned ``paper.md``.
PAPER_QUOTES: tuple[str, ...] = (
    *_Q_TABLE1_ROWS,
    _Q_CONTROL_TITER,
    _Q_TRIPLICATE_FOOTNOTE,
    _Q_TRIPLICATE_RESULTS,
    _Q_ONLY_ERROR_TYPE_STATEMENT,
    _Q_PRODUCT_IUPAC,
    _Q_HOST,
    _Q_CHASSIS,
    _Q_PBBS5K,
    _Q_PROMOTER,
    _Q_AMPLIFIED_FROM_MG1655,
    _Q_EIGHT_GENES,
    _Q_VOLUME,
    _Q_INDUCTION,
    _Q_TEMPERATURE,
    _Q_KANAMYCIN,
    _Q_ANTIBIOTICS,
    _Q_GC,
    _Q_STATISTICALLY_SIMILAR,
    _Q_GEO,
    _Q_MICROARRAY_SAMPLES,
)


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """One value sourced to the sha256-pinned ``paper.md`` OCR of the article body."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            page=page,
        ),
        quote=quote,
        note=note,
    )


def _si(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """One value sourced to the deposited Table S3 document."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=SI_TABLE3_REL,
            citation_key=CITATION_KEY,
            sha256=SI_TABLE3_SHA256,
            page=_SI_TABLE3,
        ),
        quote=quote,
        note=note,
    )


CONTROL_TITER_MG_PER_L = _paper(
    834.0,
    _Q_CONTROL_TITER,
    page=_TABLE1,
    note="the released reference titer: PS+RFP at 48 h, the strain every 'Improvement "
    "in titer (%)' value of Table 1 is computed against. Its own plus-minus, 5, is "
    "released in the same parenthesis and is NOT ingested, for the reason the module "
    "docstring gives",
)
CONTROL_UNCERTAINTY_MG_PER_L = _paper(
    5.0,
    _Q_CONTROL_TITER,
    page=_TABLE1,
    note="carried in preprocess/titer_rows.csv beside the eight Table 1 plus-minus "
    "values; not stored on the phenotype, because its TYPE is not released",
)
N_REPLICATES = _paper(
    3,
    _Q_TRIPLICATE_FOOTNOTE,
    page=_TABLE1,
    note="restated in the Results for the same measurement ('"
    + _Q_TRIPLICATE_RESULTS
    + "'), so n_samples=3 and sample_unit=biological_replicate are sourced; the kind "
    "of the plus-minus is not",
)
UNCERTAINTY_TYPE_NOT_STATED = _paper(
    None,
    _Q_ONLY_ERROR_TYPE_STATEMENT,
    page="Fig. 3 caption",
    note="the ONLY error-type statement anywhere in the article or its SI, and it "
    "belongs to Fig. 3's maximum growth rates ('at least 4 replicates'), not to Table "
    "1's titers ('triplicates'). Measured across paper.md, the pdftotext layout of "
    "paper.pdf, Text S1 and Tables S1 to S4: no other sentence names a standard "
    "deviation, a standard error or a variance",
)
#: The back-solve the campaign's resolution ladder asks for, and its outcome: it does
#: not discriminate, so rule (1) is precluded and the value is not ingested.
UNCERTAINTY_BACK_SOLVE = _paper(
    "inconclusive",
    _Q_STATISTICALLY_SIMILAR,
    page=_RESULTS_PRODUCTION,
    note="one-way ANOVA + Tukey HSD over the nine strains at n=3 reproduces all eight "
    "of the paper's own calls (soxS and nrdH not separated from the control; the other "
    "six separated) under BOTH readings, because pooling the nine spreads scales the "
    "critical comparison by the same sqrt(3) that separates an SD reading from an SE "
    "reading. Welch per-pair without a multiplicity correction reproduces NEITHER "
    "reading's nrdH call. No companion statistic discriminates, so the type stays a gap",
)
PRODUCT_IDENTITY = _paper(
    PRODUCT_SYNONYM,
    _Q_PRODUCT_IUPAC,
    page="Introduction, first sentence",
    note="the paper's own IUPAC name for its product, and the synonym under which "
    "compound_identity_table.json carries isoprenol (CPJRRXSHAYUTGL-UHFFFAOYSA-N, "
    "PubChem CID 12988). Resolving through it is what puts this row on the isoprenol "
    "axis instead of on an unresolved 'isopentenol' stub",
)
HOST_STRAIN = _paper(
    "E. coli DH1 (ATCC 33849)",
    _Q_HOST,
    page=_METHODS_STRAINS,
    note="the host of every production strain. Its K-12 marker genotype is never "
    "written, so the background carries no alleles -- that is the source's silence, "
    "not a claim that DH1 equals MG1655",
)
CHASSIS_PLASMIDS = _paper(
    "pJBEI-6830 + pJBEI-6833",
    _Q_CHASSIS,
    page=_METHODS_STRAINS,
    note="Table S3 states the same pair as the composition of the strain it calls PS",
)
CHASSIS_COMPOSITION = _si(
    ("pBbA5c-MevTsa-PMK-MK", "pTrc99A-NudB-PMD"),
    "pBbA5c-MevTsa-PMK-MK",
    note="Table S3's 'Production plasmids' block: pJBEI-6830 is pBbA5c-MevTsa-PMK-MK "
    "and pJBEI-6833 is pTrc99A-NudB-PMD. Both strings are kept on the background's "
    "construction line; see PRODUCTION_PATHWAY_NOT_TYPED for why their five gene "
    "tokens are not typed as perturbations",
)
TOLERANCE_GENE_ORIGIN = _paper(
    HOST_SPECIES,
    _Q_AMPLIFIED_FROM_MG1655,
    page=_METHODS_CLONING,
    note="the eight overexpressed genes are extra copies of NATIVE E. coli genes, "
    "amplified from the MG1655 genome that is also this record's assembly pin",
)
PROMOTER = _paper(
    "lacUV5",
    _Q_PROMOTER,
    page=_METHODS_STRAINS,
    note="stated of pBbA5k and pBbS5k, the vector the production assays use",
)
TEMPERATURE_C = _paper(
    PRODUCTION_TEMPERATURE_C,
    _Q_TEMPERATURE,
    page=_METHODS_PRODUCTION,
    note="the production phase's own temperature, after the 37 C growth to OD600 0.4",
)
AEROBICITY = _paper(
    "aerobic",
    _Q_TEMPERATURE,
    page=_METHODS_PRODUCTION,
    note="a shaken 5 mL liquid culture; the source never uses the word",
)
WORKING_VOLUME = _paper(
    PRODUCTION_VOLUME_UL,
    _Q_VOLUME,
    page=_METHODS_PRODUCTION,
    note="5 mL of fresh MM9, inoculated to OD600 0.08. The vessel type and the shaking "
    "speed of this culture are never stated and are typed gaps on CultureFormat",
)
INDUCER = _paper(IPTG_UM, _Q_INDUCTION, page=_METHODS_PRODUCTION)
KANAMYCIN = _paper(
    KANAMYCIN_UG_PER_ML,
    _Q_KANAMYCIN,
    page=_METHODS_STRAINS,
    note="the one antibiotic the source scopes to this experiment by name",
)
QUANTIFICATION = _paper(
    QUANTIFICATION_METHOD,
    _Q_GC,
    page=_METHODS_PRODUCTION,
    note="the detector is not named; the method is deferred to reference 2, which is "
    "not in the mirror, so the stored string is what this paper states",
)
DURATION_HOURS = _paper(
    PRODUCTION_DURATION_HOURS,
    _Q_TRIPLICATE_FOOTNOTE,
    page=_TABLE1,
    note="Table 1 is the 48 h time point of the 24/48/72 h series",
)

#: Why the five production-pathway tokens are read but not typed.
PRODUCTION_PATHWAY_NOT_TYPED = (
    "the five tokens of pJBEI-6830 and pJBEI-6833 (MevTsa, PMK, MK, NudB, PMD) are "
    "named on the background's construction line and are NOT typed as "
    "HeterologousPathwayPerturbation. That leaf requires source_organism and "
    "is_heterologous, and this paper defers both to reference 6 (George et al. 2014, "
    "not mirrored). The tokens also do not share an answer -- MevT's 'sa' suffix is "
    "not an E. coli pathway while NudB is an E. coli gene -- so one blanket value "
    "would state what the source does not. They are identical in all nine strains and "
    "in the control, so nothing that distinguishes a record is lost"
)
#: Why only one of the three listed antibiotics is typed.
ANTIBIOTICS_NOT_TYPED = (
    "chloramphenicol (50 ug/ml) and carbenicillin (100 ug/ml) are listed in the same "
    "sentence as kanamycin but under 'Where appropriate', and the source never says "
    f"which plasmid carries which marker ('{_Q_ANTIBIOTICS}'). Only kanamycin is "
    "scoped to this experiment by the source itself, so it is the only one typed"
)
#: Why the deposited transcriptome is not a record of this class.
GEO_NOT_A_RECORD = (
    f"GEO {GEO_SERIES} ({GEO_SAMPLES[0]} to {GEO_SAMPLES[1]}, platform {GEO_PLATFORM}) "
    "holds six arrays of ONE strain, MevT*, three with and three without 0.2% "
    "isopentenol. That is an environment-response transcriptome on an unperturbed "
    "genotype, a different experiment class from a product titer, so it belongs to a "
    "separate dataset if it is ever built and is out of scope here. Table S1's "
    "microarray log2 and z scores are that series' processed summary on the same "
    "single strain and are out of scope for the same reason. No raw reads are read"
)
#: Why the 40-candidate tolerance screen is not a record.
TOLERANCE_SCREEN_NOT_A_RECORD = (
    "Table S2's 40-candidate tolerance screen and the eight-gene confirmation release "
    "no number: the growth-lag phenotype exists only as the Fig. 4 and Fig. S2 curves, "
    "and Table S2 is a gene list. Nothing numeric is dropped"
)


# --------------------------------------------------------------------------- #
# The deposited Table S3 document
# --------------------------------------------------------------------------- #
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = f"{{{_W_NS}}}"


def _text_of(element: Any) -> str:
    """Whitespace-normalized concatenation of every ``w:t`` run under an element."""
    return " ".join("".join(node.text or "" for node in element.iter(f"{_W}t")).split())


def _cell_text(cell: Any) -> str:
    """One table cell: its paragraphs joined by a single space, normalized."""
    return " ".join(
        part for part in (_text_of(para) for para in cell.findall(f"{_W}p")) if part
    ).strip()


def read_table_s3(path: str | Path) -> list[list[str]]:
    """The rows of Table S3, read from the deposited ``.docx`` with the stdlib.

    A ``.docx`` is a zip of OpenXML, so the table is ``word/document.xml``'s single
    ``w:tbl``. Refuses a document that holds anything other than exactly one table,
    which is what a replaced or re-numbered supplement would look like.
    """
    with ZipFile(path) as archive:
        document = archive.read("word/document.xml")
    body = ElementTree.fromstring(document).find(f"{_W}body")
    if body is None:
        raise RuntimeError(f"{path} has no w:body")
    tables = body.findall(f"{_W}tbl")
    if len(tables) != 1:
        raise RuntimeError(f"{path} holds {len(tables)} tables; Table S3 is one")
    return [
        [_cell_text(cell) for cell in row.findall(f"{_W}tc")]
        for row in tables[0].findall(f"{_W}tr")
    ]


class ProductionStrain(BaseModel):
    """One row of Table S3's ``Production strains`` block."""

    strain: str
    composition: str
    plasmid: str
    gene: str


#: The Table S3 heading that opens the block this loader reads.
_PRODUCTION_STRAINS_HEADING = "Production strains"
#: The chassis row that closes it: PS itself, which carries no pBbS5k plasmid.
_CHASSIS_ROW = "PS"
_PBBS5K = re.compile(r"^pJBEI-6830 \+ pJBEI-6833 \+ (pBbS5k-(\S+))$")


def read_production_strains(path: str | Path) -> list[ProductionStrain]:
    """The nine production strains of Table S3, in the order the table lists them.

    Each row must read ``pJBEI-6830 + pJBEI-6833 + pBbS5k-<gene>``: the shared chassis
    plus exactly one tolerance plasmid, which is the whole genotype axis of this
    dataset. A row that does not is a changed table, not a row to skip.
    """
    rows = read_table_s3(path)
    headings = [
        i for i, row in enumerate(rows) if row[0] == _PRODUCTION_STRAINS_HEADING
    ]
    if len(headings) != 1:
        raise RuntimeError(
            f"Table S3 holds {len(headings)} {_PRODUCTION_STRAINS_HEADING!r} headings; "
            "exactly one is expected"
        )
    strains: list[ProductionStrain] = []
    for row in rows[headings[0] + 1 :]:
        if row[0] == _CHASSIS_ROW:
            break
        match = _PBBS5K.match(row[1])
        if match is None:
            raise RuntimeError(
                f"Table S3 production strain {row[0]!r} reads {row[1]!r}, not "
                "'pJBEI-6830 + pJBEI-6833 + pBbS5k-<gene>'"
            )
        strains.append(
            ProductionStrain(
                strain=row[0],
                composition=row[1],
                plasmid=match.group(1),
                gene=match.group(2),
            )
        )
    return strains


#: The Table S3 heading that opens the two-row block naming what each production
#: plasmid is: pJBEI-6830 and pJBEI-6833, the pair every one of the nine strains and
#: the control carries.
_PRODUCTION_PLASMIDS_HEADING = "Production plasmids"


def read_production_plasmids(path: str | Path) -> dict[str, str]:
    """Table S3's ``Production plasmids`` block: plasmid name -> its composition.

    The block runs from its heading to the first row with an empty second cell, which
    in the deposited bytes is the blank separator before the ``Strains`` header.
    Refuses a block that is not exactly the two plasmids the Methods name, since a
    changed block is a changed chassis.
    """
    rows = read_table_s3(path)
    headings = [
        i for i, row in enumerate(rows) if row[0] == _PRODUCTION_PLASMIDS_HEADING
    ]
    if len(headings) != 1:
        raise RuntimeError(
            f"Table S3 holds {len(headings)} {_PRODUCTION_PLASMIDS_HEADING!r} headings; "
            "exactly one is expected"
        )
    plasmids: dict[str, str] = {}
    for row in rows[headings[0] + 1 :]:
        if not row[1]:
            break
        plasmids[row[0]] = row[1]
    expected = str(CHASSIS_PLASMIDS.value).split(" + ")
    if sorted(plasmids) != sorted(expected):
        raise RuntimeError(
            f"Table S3's {_PRODUCTION_PLASMIDS_HEADING!r} block names "
            f"{sorted(plasmids)}; the Methods state {sorted(expected)}"
        )
    return plasmids


def chassis_row(path: str | Path) -> list[str]:
    """Table S3's ``PS`` row: the chassis every record is an edit of."""
    for row in read_table_s3(path):
        if row[0] == _CHASSIS_ROW:
            return row
    raise RuntimeError(f"Table S3 in {path} has no {_CHASSIS_ROW!r} row")


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (both mirrors live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/fooImprovingMicrobialBiogasoline2014``."""
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


def pmc_cloud_key(filename: str = SI_TABLE3_SOURCE_FILENAME) -> str:
    """Bucket key of one supplementary file in the PMC Article Datasets bucket."""
    return f"{PMC_PREFIX}/{filename}"


def deposit_raw_mirror(
    *,
    table_s3_path: str | Path,
    retrieved_at: str = SI_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror (Table S3) and its ``manifest.json``.

    Exactly the file the loader consumes on its first successful build, no more: the
    other nine supplementary objects are listed in ``si_data_sources`` as the retrieval
    record of what the article released, and ``si_expected`` says of each why it is not
    deposited. Idempotent by sha256: a matching file is left alone and a differing one
    raises rather than being overwritten, and the source is verified BEFORE anything is
    written, so a refusal leaves no partial deposit.
    """
    verify_sha256(table_s3_path, SI_TABLE3_SHA256)
    root = raw_mirror_dir(data_root)
    dest = root / SI_TABLE3_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != SI_TABLE3_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(table_s3_path, dest)
    key = pmc_cloud_key()
    url = f"https://pmc-oa-opendata.s3.amazonaws.com/{key}"
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=[
            ArtifactRecord(
                path=SI_TABLE3_REL,
                role=ROLE_SI_DATA,
                bytes=dest.stat().st_size,
                sha256=SI_TABLE3_SHA256,
                source=url,
                original_filename=SI_TABLE3_SOURCE_FILENAME,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.pmc_cloud,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.pmc_cloud_object",
                    params={"key": key},
                    sha256=SI_TABLE3_SHA256,
                    retrieved_at=retrieved_at,
                ),
            )
        ],
        si_data_sources=[
            url,
            f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GEO_SERIES}",
        ],
        si_expected=[
            "si/si9.docx (publisher mbo005142049st3.docx, Table S3) -- deposited, and "
            "the only released data file this loader reads: its 'Production strains' "
            "block is the nine genotypes and its 'Production plasmids' block is the "
            "chassis composition",
            "Table 1's titer column lives in the article body, NOT in any released "
            "data file. It is carried as sourced constants quoting the sha256-pinned "
            f"paper.md OCR in the torchcell-library mirror ({PAPER_MD_SHA256}), and "
            "an L4 level re-reads those bytes",
            "Text S1 (mbo005142049s1.docx), Table S2 (mbo005142049st2.docx) and Table "
            "S4 (mbo005142049st4.docx) -- gene descriptions, the 40-candidate screen "
            "list and the cloning oligonucleotides. Mirrored in torchcell-library and "
            "read while sourcing, but no value of a record comes from them",
            "Table S1 (mbo005142049st1.pdf) and Figures S1 to S5 "
            "(mbo005142049sf1-sf5.tif) -- the microarray summary and the growth-curve "
            "figures. Out of scope: " + GEO_NOT_A_RECORD,
            "the per-replicate titers were NEVER released: there is no source-data "
            "file, and Table 1's plus-minus values are the only spread published. "
            "Nothing is fetchable here, which is why the uncertainty is a typed gap",
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


def verify_paper_quotes(data_root: str | None = None) -> int:
    """Assert every article quote is verbatim in the sha256-pinned ``paper.md``.

    Table 1's titer column is the one consumed value that is not in a released data
    file, so this is what makes it auditable: the OCR bytes are hashed, the hash is
    checked against :data:`PAPER_MD_SHA256`, and each quote must be a substring.
    Returns the number of quotes checked.
    """
    path = library_mirror_dir(data_root) / PAPER_MD
    digest = _sha256(path)
    if digest != PAPER_MD_SHA256:
        raise RuntimeError(
            f"{path} hashes {digest}, not the pinned {PAPER_MD_SHA256}; the OCR mirror "
            "moved and every quote must be re-read before it is trusted"
        )
    text = path.read_text(encoding="utf-8")
    missing = [quote for quote in PAPER_QUOTES if quote not in text]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(PAPER_QUOTES)} quotes are not verbatim in "
            f"{PAPER_MD}: {missing[:2]}"
        )
    return len(PAPER_QUOTES)


# --------------------------------------------------------------------------- #
# Table 1, parsed out of its own verbatim rows
# --------------------------------------------------------------------------- #
_TABLE1_CELLS = re.compile(r"<td>(.*?)</td>")


class Table1Row(BaseModel):
    """One Table 1 row: the gene, its 48 h titer, its plus-minus and its improvement."""

    gene: str
    description: str
    titer_mg_per_l: float
    uncertainty_mg_per_l: float
    improvement_percent: float
    quote: str


def parse_table1() -> list[Table1Row]:
    """Parse the eight Table 1 rows out of :data:`_Q_TABLE1_ROWS`.

    The stored titer comes from the same bytes the quote is checked against, so a value
    cannot drift away from its justification. The titer cell is ``<mean> ± <spread>``
    with a thousands comma on the one four-digit value.
    """
    rows: list[Table1Row] = []
    for quote in _Q_TABLE1_ROWS:
        cells = _TABLE1_CELLS.findall(quote)
        if len(cells) != 6:
            raise RuntimeError(f"Table 1 row {quote!r} has {len(cells)} cells, not 6")
        mean, _, spread = cells[3].partition(" ± ")
        rows.append(
            Table1Row(
                gene=cells[0],
                description=cells[1],
                titer_mg_per_l=float(mean.replace(",", "")),
                uncertainty_mg_per_l=float(spread),
                improvement_percent=float(cells[4]),
                quote=quote,
            )
        )
    return rows


TABLE1_ROWS: tuple[Table1Row, ...] = tuple(parse_table1())
#: Table 1's gene symbol for each Table S3 production strain. PS+RFP is the control and
#: has no Table 1 row: its titer is the one footnote b releases.
STRAIN_TO_GENE: dict[str, str] = {
    "PS+SoxS": "soxS",
    "PS+GidB": "gidB",
    "PS+NrdH": "nrdH",
    "PS+YqhD": "yqhD",
    "PS+IbpA": "ibpA",
    "PS+MetR": "metR",
    "PS+Fpr": "fpr",
    "PS+MdlB": "mdlB",
    "PS+RFP": "rfp",
}
CONTROL_STRAIN = "PS+RFP"
CONTROL_GENE = "rfp"
TOLERANCE_GENES: tuple[str, ...] = tuple(row.gene for row in TABLE1_ROWS)


# --------------------------------------------------------------------------- #
# The record parts
# --------------------------------------------------------------------------- #
def publication() -> Publication:
    """This paper, by DOI (the mirror's manifest records no PubMed id)."""
    return Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")


def isopentenol() -> Compound:
    """The product, resolved through the IUPAC synonym the paper itself states.

    Returns the canonical ``isoprenol`` row of ``compound_identity_table.json``, so a
    query for every isoprenol record reaches these titers.
    """
    compound = resolved_compound(PRODUCT_SYNONYM)
    if compound.inchikey is None:
        raise RuntimeError(
            f"{PRODUCT_SYNONYM!r} no longer resolves to a compound with an InChIKey; "
            "the product of this dataset is the isoprenol entity and nothing else"
        )
    return compound


def chassis_background(composition: str) -> BacterialStrainBackground:
    """The PS chassis: DH1 carrying the two production plasmids.

    ``alleles`` is empty because the paper states DH1 only by name and ATCC number and
    never writes its K-12 marker genotype; MG1655 is the pin because it is the genome
    the overexpressed genes were amplified from and the namespace their b-numbers
    belong to.
    """
    return BacterialStrainBackground(
        name=CHASSIS_STRAIN,
        reference_strain=REFERENCE_STRAIN,
        assembly_set="ecoli_K12_MG1655_ASM584v2",
        parents=[str(HOST_STRAIN.value)],
        construction=(
            f"{composition} ({CHASSIS_COMPOSITION.value[0]}; "
            f"{CHASSIS_COMPOSITION.value[1]})"
        ),
        genotype_statement=composition,
        alleles=[],
        provenance=[HOST_STRAIN, CHASSIS_PLASMIDS, CHASSIS_COMPOSITION],
    )


def chassis_reference(
    composition: str, data_root: str | None = None
) -> AssemblyReferenceGenome:
    """The PS chassis pinned to the MG1655 GenBank assembly."""
    return assembly_reference(
        REFERENCE_STRAIN,
        background=chassis_background(composition),
        data_root=data_root,
    )


def overexpression_perturbation(
    gene: str, locus_tag: str, plasmid: str
) -> HeterologousPathwayPerturbation:
    """The one pBbS5k-borne gene that distinguishes a production strain.

    For the eight tolerance genes this is an extra copy of a NATIVE gene, the case
    ``HeterologousPathwayPerturbation`` names explicitly, so ``systematic_gene_name`` is
    the MG1655 b-number and ``source_organism`` is the host. For the rfp control the
    reporter's origin is never stated, so it is
    :data:`SOURCE_ORGANISM_UNREPORTED`. ``copy_number`` is the leaf's default 1.0: Table
    S3 calls pBbS5k "a low-copy plasmid" and no number is released.
    """
    is_control = gene == CONTROL_GENE
    return HeterologousPathwayPerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=gene,
        gene_namespace=NAMESPACE,
        pathway_name=PATHWAY_REPORTER if is_control else PATHWAY_TOLERANCE,
        source_organism=(
            SOURCE_ORGANISM_UNREPORTED
            if is_control
            else str(TOLERANCE_GENE_ORIGIN.value)
        ),
        is_heterologous=is_control,
        localization="episomal_plasmid",
        construct_name=plasmid,
        promoter_name=str(PROMOTER.value),
    )


def genotype(gene: str, locus_tag: str, plasmid: str) -> Genotype:
    """The genotype of one production strain: the chassis plus one overexpressed gene."""
    return Genotype(
        perturbations=[overexpression_perturbation(gene, locus_tag, plasmid)]
    )


def _culture_format() -> CultureFormat:
    """The 5 mL production culture, with the two fields the source never states."""
    return CultureFormat(
        vessel=None,
        working_volume_ul=float(WORKING_VOLUME.value),
        shaking_rpm=None,
        inoculum_od600=PRODUCTION_INOCULUM_OD600,
        endpoint=EndpointRule.fixed_duration,
        provenance=[WORKING_VOLUME],
        provenance_gaps=[
            ProvenanceGap(
                field="vessel",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the production Methods state a 5 mL volume and shaking but never "
                "the vessel; the 24-well plates named elsewhere belong to the growth "
                "and tolerance assays, which are a different experiment",
            ),
            ProvenanceGap(
                field="shaking_rpm",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="'with shaking' for the production culture, with no speed; the "
                "225 rpm the Methods give belongs to the preculture and the growth "
                "assays",
            ),
        ],
    )


def environment() -> CultureEnvironment:
    """The single production condition every record and the control share.

    A ``CultureEnvironment``, since ``ProductTiterExperiment.environment`` is annotated
    as one. IPTG and kanamycin are the two typed additions; see
    :data:`ANTIBIOTICS_NOT_TYPED` for the two that are not.
    """
    return CultureEnvironment(
        media=MM9_FOO2014,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        culture_format=_culture_format(),
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound("IPTG"),
                concentration=Concentration(
                    value=float(INDUCER.value), unit=ConcentrationUnit.micromolar
                ),
            ),
            SmallMoleculePerturbation(
                compound=resolved_compound("kanamycin"),
                concentration=Concentration(
                    value=float(KANAMYCIN.value), unit=ConcentrationUnit.ug_per_ml
                ),
            ),
        ],
        aerobicity=str(AEROBICITY.value),
        duration_hours=float(DURATION_HOURS.value),
    )


def titer_phenotype(titer_mg_per_l: float) -> ProductTiterPhenotype:
    """One released 48 h titer in ``ug/mL``, with the typed absences the paper forces.

    The released numbers are mg/L and 1 mg/L is exactly 1 ug/mL, so no arithmetic is
    applied to a source value. ``titer_uncertainty`` and ``titer_uncertainty_type`` are
    ALWAYS gaps: the plus-minus number IS released, but what it IS is not, and
    ``UncertaintyType`` has no ``unknown`` member, so storing it would mean guessing
    between a sample SD and a standard error. The numbers stay in
    ``preprocess/titer_rows.csv``.
    """
    unstated = (
        "the plus-minus numbers ARE released (Table 1 and its footnote b) but their "
        "KIND is not: the only error-type statement in the article or its SI belongs "
        f"to Fig. 3's growth rates ('{_Q_ONLY_ERROR_TYPE_STATEMENT}'), and an ANOVA + "
        "Tukey back-solve on the paper's own significance calls does not discriminate "
        "an SD reading from an SE reading. The numbers are kept in "
        "preprocess/titer_rows.csv"
    )
    return ProductTiterPhenotype(
        product=isopentenol(),
        titer=titer_mg_per_l,
        titer_unit=ConcentrationUnit.ug_per_ml,
        n_samples=int(N_REPLICATES.value),
        sample_unit=SampleUnit.biological_replicate,
        quantification_method=str(QUANTIFICATION.value),
        provenance_gaps=[
            ProvenanceGap(
                field="titer_uncertainty",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=unstated,
            ),
            ProvenanceGap(
                field="titer_uncertainty_type",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=unstated,
            ),
            ProvenanceGap(
                field="product_yield",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no yield on substrate is released for any strain",
            ),
            ProvenanceGap(
                field="product_yield_unit",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no yield on substrate is released for any strain",
            ),
            ProvenanceGap(
                field="productivity",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no volumetric productivity is released for any strain",
            ),
            ProvenanceGap(
                field="productivity_unit",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no volumetric productivity is released for any strain",
            ),
        ],
    )


class BuildAccounting(BaseModel):
    """What the build read, what it kept and what it declined, written to preprocess."""

    dataset: str
    source_rows: int
    kept_records: int
    dropped_records: int
    paper_quotes_checked: int
    notes: list[str]

    def check(self) -> None:
        """Every source row is a record: this build declines nothing numeric."""
        if self.kept_records + self.dropped_records != self.source_rows:
            raise RuntimeError(
                f"{self.kept_records} kept + {self.dropped_records} dropped is not the "
                f"{self.source_rows} source rows"
            )


def _link_mirror_files(raw_dir: str, pins: Iterable[tuple[str, str, str]]) -> None:
    """Hard-link each pinned mirror file into ``raw/`` after verifying its sha256."""
    os.makedirs(raw_dir, exist_ok=True)
    root = raw_mirror_dir()
    for relpath, name, digest in pins:
        link_verified(root / relpath, osp.join(raw_dir, name), digest)


@register_dataset
class IsopentenolTiterFoo2014Dataset(ExperimentDataset):
    """Foo 2014 isopentenol titers: the eight tolerance strains and the RFP control."""

    #: Every b-number this campaign names must resolve to a locus of the pinned MG1655
    #: assembly. Measured on the pinned bytes: 8 of 8, so anything below 1.0 means the
    #: annotation or the released symbols moved and the build stops.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 1.0

    def __init__(
        self,
        root: str = DEV_TREE_RELPATH,
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the MG1655 genome resolves the eight overexpressed symbols."""
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
        """The deposited Table S3 document."""
        return [SI_TABLE3_FILENAME]

    def download(self) -> None:
        """Link Table S3 into ``raw/`` after verifying it against its pin."""
        _link_mirror_files(
            self.raw_dir, ((SI_TABLE3_REL, SI_TABLE3_FILENAME, SI_TABLE3_SHA256),)
        )
        log.info("Foo 2014 Table S3 linked into %s", self.raw_dir)

    def _genome(self) -> EcoliK12Genome:
        """The injected MG1655 genome, or one opened from the genomes tier."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", REFERENCE_STRAIN)
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Build one titer record per released production strain; write LMDB."""
        verify_raw_files(self.raw_dir, {SI_TABLE3_FILENAME: SI_TABLE3_SHA256})
        si_path = osp.join(self.raw_dir, SI_TABLE3_FILENAME)
        n_quotes = verify_paper_quotes()
        strains = read_production_strains(si_path)
        composition = chassis_row(si_path)[1]
        self._assert_table_s3_agrees_with_table1(
            strains, composition, read_production_plasmids(si_path)
        )

        genome = self._genome()
        stored, report = reconcile_locus_tags(
            genome, pd.Series(list(TOLERANCE_GENES)), label=self.name
        )
        report.require_resolved(self.MIN_RESOLVED_FRACTION)
        if report.outside_namespace:
            raise RuntimeError(
                f"{self.name}: locus tags outside {NAMESPACE}: "
                f"{report.outside_namespace}"
            )
        locus_tags = dict(zip(TOLERANCE_GENES, stored))

        titers = {row.gene: row for row in TABLE1_ROWS}
        shared_environment = environment()
        reference = ProductTiterExperimentReference(
            dataset_name=self.name,
            genome_reference=chassis_reference(composition),
            environment_reference=shared_environment,
            phenotype_reference=titer_phenotype(float(CONTROL_TITER_MG_PER_L.value)),
        )
        pub = publication()

        rows: list[dict[str, Any]] = []
        for strain in strains:
            gene = STRAIN_TO_GENE[strain.strain]
            is_control = gene == CONTROL_GENE
            row = None if is_control else titers[gene]
            rows.append(
                {
                    "strain": strain.strain,
                    "composition": strain.composition,
                    "plasmid": strain.plasmid,
                    "gene": gene,
                    "locus_tag": locus_tags.get(gene, gene),
                    "source": "Table 1 footnote b" if is_control else "Table 1",
                    "titer_mg_per_l": (
                        float(CONTROL_TITER_MG_PER_L.value)
                        if row is None
                        else row.titer_mg_per_l
                    ),
                    "uncertainty_mg_per_l": (
                        float(CONTROL_UNCERTAINTY_MG_PER_L.value)
                        if row is None
                        else row.uncertainty_mg_per_l
                    ),
                    "uncertainty_type": "not_released",
                    "improvement_percent": (
                        0.0 if row is None else row.improvement_percent
                    ),
                    "n_samples": int(N_REPLICATES.value),
                    "time_hours": float(DURATION_HOURS.value),
                    "quote": (_Q_CONTROL_TITER if row is None else row.quote),
                }
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for ledger_row in tqdm(rows, desc="foo2014-titer"):
                experiment = ProductTiterExperiment(
                    dataset_name=self.name,
                    genotype=genotype(
                        str(ledger_row["gene"]),
                        str(ledger_row["locus_tag"]),
                        str(ledger_row["plasmid"]),
                    ),
                    environment=shared_environment,
                    phenotype=titer_phenotype(float(ledger_row["titer_mg_per_l"])),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "titer_rows.csv"), index=False
        )
        accounting = BuildAccounting(
            dataset=self.name,
            source_rows=len(strains),
            kept_records=idx,
            dropped_records=len(strains) - idx,
            paper_quotes_checked=n_quotes,
            notes=[
                "nothing is dropped: every released isopentenol titer in the mirror is "
                "a record, the eight of Table 1 and the control of its footnote b",
                "PS+RFP is both a record and the reference phenotype, which is what "
                "the source does: its 834 mg/L is the denominator of every "
                "'Improvement in titer (%)' value",
                PRODUCTION_PATHWAY_NOT_TYPED,
                ANTIBIOTICS_NOT_TYPED,
                GEO_NOT_A_RECORD,
                TOLERANCE_SCREEN_NOT_A_RECORD,
                str(UNCERTAINTY_BACK_SOLVE.note),
                f"{n_quotes} article quotes were re-read verbatim from the pinned "
                "paper.md OCR before any value was used",
            ],
        )
        accounting.check()
        with open(
            osp.join(self.preprocess_dir, "build_accounting.json"), "w"
        ) as handle:
            handle.write(accounting.model_dump_json(indent=2))
        log.info(
            "Foo2014 titer: %d records over %d resolved loci; reference PS+RFP "
            "%.1f ug/mL",
            idx,
            len(locus_tags),
            float(CONTROL_TITER_MG_PER_L.value),
        )

    @staticmethod
    def _assert_table_s3_agrees_with_table1(
        strains: Sequence[ProductionStrain],
        composition: str,
        plasmids: Mapping[str, str],
    ) -> None:
        """The deposited strain table and the article's Table 1 must name one panel.

        Four checks on two independent places in the mirror: Table S3 lists exactly the
        nine strains :data:`STRAIN_TO_GENE` maps, their pBbS5k genes are Table 1's eight
        plus the control, the chassis row carries the pair the Methods state, and the
        two production plasmids read out of the deposited bytes are the compositions
        :data:`CHASSIS_COMPOSITION` quotes -- which is what makes that quote auditable,
        since no article quote covers it.
        """
        names = [strain.strain for strain in strains]
        if sorted(names) != sorted(STRAIN_TO_GENE):
            raise RuntimeError(
                f"Table S3 lists production strains {names}, not {sorted(STRAIN_TO_GENE)}"
            )
        genes = {strain.gene for strain in strains}
        expected = set(TOLERANCE_GENES) | {CONTROL_GENE}
        if genes != expected:
            raise RuntimeError(
                f"Table S3's pBbS5k genes are {sorted(genes)}; Table 1's eight plus the "
                f"control are {sorted(expected)}"
            )
        for strain in strains:
            if strain.gene != STRAIN_TO_GENE[strain.strain]:
                raise RuntimeError(
                    f"Table S3 pairs {strain.strain!r} with {strain.plasmid!r}, which "
                    f"is not the {STRAIN_TO_GENE[strain.strain]!r} plasmid"
                )
        if composition != str(CHASSIS_PLASMIDS.value):
            raise RuntimeError(
                f"Table S3's PS row reads {composition!r}; the Methods state "
                f"{CHASSIS_PLASMIDS.value!r}"
            )
        read = tuple(
            plasmids[name] for name in str(CHASSIS_PLASMIDS.value).split(" + ")
        )
        if read != tuple(CHASSIS_COMPOSITION.value):
            raise RuntimeError(
                f"Table S3 composes the production plasmids as {read}; "
                f"CHASSIS_COMPOSITION quotes {tuple(CHASSIS_COMPOSITION.value)}"
            )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification, L0 to L4, run from this module
# --------------------------------------------------------------------------- #
def _improvement_percent(titer: float, control: float) -> float:
    """The paper's own 'Improvement in titer (%)', rounded as the table prints it."""
    return float(round((titer - control) / control * 100.0))


def verify_build(dataset_root: str, data_root: str | None = None) -> VerificationReport:
    """Run L0 to L4 over a built tree and write ``preprocess/verification_report.json``.

    L4 is a genuine cross-source join: each stored titer is re-derived from the control
    titer and the released ``Improvement in titer (%)`` column, a DIFFERENT column of
    the same table from the one the values were read out of, and the genotype of each
    record is re-read from the deposited Table S3 bytes.
    """
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    ledger = pd.read_csv(osp.join(dataset_root, "preprocess", "titer_rows.csv"))
    experiments = [record["experiment"] for record in records]
    phenotypes = [exp["phenotype"] for exp in experiments]
    control = float(CONTROL_TITER_MG_PER_L.value)
    report = VerificationReport(
        dataset_name=IsopentenolTiterFoo2014Dataset.__name__,
        provenance=Provenance(
            source_uri=SI_TABLE3_REL,
            citation_key=CITATION_KEY,
            sha256=SI_TABLE3_SHA256,
            method=(
                "isopentenol titer in mg/L at 48 h by gas chromatography, stored "
                "verbatim as ug/mL: Table 1's eight tolerance strains (from the pinned "
                "paper.md OCR) and the PS+RFP control of its footnote b, with the nine "
                "genotypes read from the deposited Table S3"
            ),
            page=f"{_TABLE1}; {_SI_TABLE3}",
            retrieved=SI_RETRIEVED_AT,
        ),
    )
    report.add(l0_structural(experiments, ProductTiterExperiment.model_validate))
    report.add(l1_count(len(records), EXPECTED_RECORDS))
    report.add(l2_value_fidelity([p["titer"] for p in phenotypes], minimum=0.0))
    report.add(
        l2_cross_method(
            [float(len(exp["genotype"]["perturbations"])) for exp in experiments],
            [1.0] * len(experiments),
            tol=0.0,
        )
    )
    report.add(
        l3_convention(
            "titer_unit_is_the_sources_mg_per_l_as_ug_per_ml",
            all(
                p["titer_unit"] == ConcentrationUnit.ug_per_ml.value for p in phenotypes
            ),
            detail="1 mg/L == 1 ug/mL exactly, so the released number is stored verbatim",
        )
    )
    report.add(
        l3_convention(
            "every_uncertainty_is_a_typed_gap_not_a_guess",
            all(
                p["titer_uncertainty"] is None
                and p["titer_uncertainty_type"] is None
                and p["n_samples"] == int(N_REPLICATES.value)
                and p["sample_unit"] == SampleUnit.biological_replicate.value
                and {"titer_uncertainty", "titer_uncertainty_type"}
                <= {gap["field"] for gap in p["provenance_gaps"]}
                for p in phenotypes
            ),
            detail="the replicate design is sourced (n=3 biological replicates) and the "
            "plus-minus numbers are released, but their KIND is stated nowhere in the "
            "mirror and the back-solve does not discriminate",
        )
    )
    report.add(
        l3_convention(
            "the_product_is_the_canonical_isoprenol_entity",
            all(p["product"]["inchikey"] == isopentenol().inchikey for p in phenotypes),
            detail="resolved through the paper's own '3-methyl-3-buten-1-ol', so these "
            "titers join every other isoprenol record",
        )
    )
    report.add(
        l3_convention(
            "the_reference_is_the_released_PS_RFP_control_titer",
            all(
                record["reference"]["phenotype_reference"]["titer"] == control
                and record["reference"]["genome_reference"]["strain"] == CHASSIS_STRAIN
                and record["reference"]["genome_reference"]["assembly_accession"]
                == "GCA_000005845.2"
                for record in records
            ),
            detail=f"every record is an edit of PS on GCA_000005845.2 against the "
            f"{control} ug/mL PS+RFP control Table 1 footnote b releases",
        )
    )
    report.add(_improvement_l4(ledger, control))
    report.add(_genotype_l4(records, ledger, data_root))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def _improvement_l4(ledger: pd.DataFrame, control: float) -> Any:
    """L4: each stored titer against Table 1's own 'Improvement in titer (%)' column."""
    shared = [
        (str(gene), _improvement_percent(float(titer), control), float(improvement))
        for gene, titer, improvement in zip(
            ledger["gene"].tolist(),
            ledger["titer_mg_per_l"].tolist(),
            ledger["improvement_percent"].tolist(),
            strict=True,
        )
    ]
    return l4_cross_source(shared, tol=0.0).model_copy(
        update={"name": "titer_against_released_improvement_percent"}
    )


def _genotype_l4(
    records: Sequence[dict[str, Any]], ledger: pd.DataFrame, data_root: str | None
) -> Any:
    """L4: the stored plasmid of each record against the deposited Table S3 bytes."""
    released = {
        strain.gene: strain.plasmid
        for strain in read_production_strains(raw_mirror_dir(data_root) / SI_TABLE3_REL)
    }
    stored = {
        str(gene): str(
            records[index]["experiment"]["genotype"]["perturbations"][0][
                "construct_name"
            ]
        )
        for index, gene in enumerate(ledger["gene"].tolist())
    }
    shared = [
        (gene, 1.0, 1.0 if released.get(gene) == plasmid else 0.0)
        for gene, plasmid in stored.items()
    ]
    return l4_cross_source(shared, tol=0.0).model_copy(
        update={"name": "construct_against_deposited_table_s3"}
    )


def main() -> None:
    """Build the dataset into ``$DATA_ROOT`` and print its verification report."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, DEV_TREE_RELPATH)
    dataset = IsopentenolTiterFoo2014Dataset(root=root)
    print(f"len = {len(dataset)}")
    dataset.close_lmdb()
    accounting = json.loads(
        Path(osp.join(root, "preprocess/build_accounting.json")).read_text()
    )
    print(json.dumps(accounting, indent=2))
    print(verify_build(root, data_root).summary())


if __name__ == "__main__":
    main()
