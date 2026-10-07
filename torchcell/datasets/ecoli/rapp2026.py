# torchcell/datasets/ecoli/rapp2026
# [[torchcell.datasets.ecoli.rapp2026]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/rapp2026
# Test file: tests/torchcell/datasets/ecoli/test_rapp2026.py
"""Rapp 2026 metabolome of a metabolism-wide E. coli CRISPRi library (FI-MS features).

Rapp, Verhulsdonk, Garcke, Stadelmann, Farke, Trossmann, Kronenberger, Alvarado,
Petras and Link 2026 (Cell Systems, doi:10.1016/j.cels.2025.101518) sorted a pooled
CRISPRi library into an arrayed one covering all 1,515 genes of the iML1515 model,
induced dCas9 with anhydrotetracycline, grew every strain on M9 glucose at 37 C and
profiled 3,026 metabolite extracts by flow-injection mass spectrometry. Each strain's
released value per feature is a LINEAR FOLD CHANGE relative to the per-batch median,
and the stored value is the arithmetic mean of the strain's two plates. Measured at
build time, that mean reproduces the paper's own ``Mean_FC`` column exactly for all
1,385 accumulating strain-metabolite pairs of Table S5 (``mean_fc_crosscheck.json``),
so the stored statistic is the paper's, not a re-derivation of it.

DATA. The matrices are in the publisher SI, which the Elsevier CDN serves directly
(``RetrievalMethod.direct_url`` through ``retrieve.elsevier_mmc``). The loader consumes
five workbooks: Table S4 (``si5.xlsx``, rows annotated m/z features, columns the 3,026
samples; the values), Table S3 (``si4.xlsx``, one row per sample: target gene, b-number,
sampling OD, plate, well, replicate), Table S1 (``si2.xlsx``, the sgRNA of each library
gene including its 20-nt base-pairing region), Table S9 (``si10.xlsx``, the 802 isobaric
metabolites with their BiGG id, KEGG id, monoisotopic mass and neutral formula: the
metabolite-identity layer) and Table S5 (``si6.xlsx``, the accumulating pairs, read ONLY
to cross-check the stored mean against the paper's ``Mean_FC``). The MassIVE deposits
(MSV000098712 / MSV000098755 / MSV000098714) hold raw spectra, not these matrices, and
are NOT mirrored; neither are the SI tables the loader does not read (``NOT_MIRRORED``).

STRAIN. The host is ``YYdCas9``, which the key resources table writes
``BW25993 intC:tetR-dcas9-aadA lacY:ypet-cat`` and defers to Lawson 2017 (NOT in the
literature mirror, so the background's ``construction`` is a typed gap). BW25993 has no
deposited assembly set, and every identifier the paper releases is an iML1515
**b-number**, so records pin ``assembly_reference("MG1655", background=...)``: the
MG1655 GenBank assembly is the sha256-pinned bytes those b-numbers mean, and YYdCas9 is
carried as a ``BacterialStrainBackground`` whose ``parents`` name BW25993 and whose
``genotype_statement`` is the key-resources string verbatim. Measured: all 1,497
released b-numbers resolve on that annotation (1,495 current, 2 pseudogene loci).

RECORDS DROPPED (rules + items in ``preprocess/dropped_records.json``). A strain token
whose Table S3 b-number is the controls' ``b0000`` placeholder and which Table S1 carries
no sgRNA for (``no_target_gene_assigned``: ``argR``, one strain), and a strain whose
released b-number is not a locus tag of the pinned annotation but a ``/gene_synonym`` of
another locus (``b_number_remapped_by_the_annotation``: ``phnE`` ``b4104``, which the
MG1655 annotation carries as a synonym of the pseudogene ``b4583`` ``phnE1``). The second
is a SCHEMA finding, not a data one: storing ``b4583`` needs a
``DerivedIdentifierMapping``, and ``DerivedIdentifierRoute`` has no member for a retired
tag of the pinned strain's own namespace, so the mapping cannot be recorded on the
record and the record is dropped rather than remapped silently.

PHENOTYPE. ``MetabolitePhenotype`` keyed by Table S4's ``Abbr`` verbatim, which is the
iML1515 isobaric group's abbreviation plus the adduct (``frdp[M-H]-``), so two adducts of
one metabolite stay two measured features. ``n_replicates`` is 2 for every key (the two
independent plates); ``metabolite_level_se`` is the standard error of that mean,
``|r1 - r2| / 2``, derived from the two released replicate columns and in the same units
as the value. ``target_metabolite_ids`` maps a key to its Table S9 **BiGG** id wherever
the feature's abbreviation names exactly ONE metabolite; a feature whose abbreviation is
a MERGED isobaric set (FI-MS cannot separate equal masses) is absent from the map rather
than assigned one of its candidates, and ``preprocess/metabolite_identity.json`` lists
every such key with its full candidate set. A gap on the field is not expressible
beside a partial map (``ProvenanceGapMixin`` requires a gapped field to be ``None``), so
the uncovered keys are recorded in that ledger, the dendron note and the PR.
The reference is the measured profile of the 15 control strains (empty sgRNA) on the
same per-batch-median scale, with ``n_replicates`` 30.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Final

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
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    BacterialCrisprInterferencePerturbation,
    BacterialMetaboliteExperiment,
    BacterialMetaboliteExperimentReference,
    BacterialReferenceStrain,
    BacterialStrainBackground,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    MetabolitePhenotype,
    Publication,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    LOCUS_TAG_PATTERNS,
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
from torchcell.literature.retrieve import elsevier_mmc_url
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "rappMetabolomeColiCRISPRi2026"
PAPER_DOI = "10.1016/j.cels.2025.101518"
#: The article's Elsevier PII; the SI files are ``1-s2.0-<PII>-mmcN.xlsx`` on the CDN.
ELSEVIER_PII = "S2405471225003515"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "3c63b665c7e69d8e433f3b5a48a579a01956be919bc22669cbc789bf74d643e5"
DATA_RETRIEVED_AT = "2026-10-07T11:43:21.426570+00:00"

#: Table S1: the sgRNA of every library gene (``mmc2.xlsx``).
TABLE_S1 = "si2.xlsx"
TABLE_S1_SHEET = "Table_S1"
#: Table S3: one row per metabolome sample (``mmc4.xlsx``).
TABLE_S3 = "si4.xlsx"
TABLE_S3_SHEET = "Table_S3"
#: Table S4: the FI-MS fold-change matrix (``mmc5.xlsx``).
TABLE_S4 = "si5.xlsx"
TABLE_S4_SHEET = "Table_S4"
#: Table S5: the accumulating pairs, read only to cross-check the mean (``mmc6.xlsx``).
TABLE_S5 = "si6.xlsx"
TABLE_S5_SHEET = "TableS5"
#: Table S9: the 802 isobaric metabolites, the identity layer (``mmc10.xlsx``).
TABLE_S9 = "si10.xlsx"
TABLE_S9_SHEET = "TableS9"

FloatMatrix = npt.NDArray[np.float64]


class RawFile(BaseModel):
    """One file the loader consumes: its pinned bytes and how they were retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    member: str
    sha256: str
    bytes: int
    description: str
    retrieval: RetrievalRecord

    @property
    def mirror_relpath(self) -> str:
        """Path inside the raw mirror (``data/<name>``)."""
        return f"data/{self.name}"


def _elsevier_file(
    name: str, member: str, sha256: str, size: int, description: str
) -> RawFile:
    """One Elsevier SI workbook, retrieved by ``retrieve.elsevier_mmc``."""
    return RawFile(
        name=name,
        member=member,
        sha256=sha256,
        bytes=size,
        description=description,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.direct_url,
            source_url=elsevier_mmc_url(ELSEVIER_PII, member),
            retriever="torchcell.literature.retrieve.elsevier_mmc",
            params={"pii": ELSEVIER_PII, "filename": member},
            sha256=sha256,
            retrieved_at=DATA_RETRIEVED_AT,
        ),
    )


RAW_FILES: tuple[RawFile, ...] = (
    _elsevier_file(
        TABLE_S1,
        "mmc2.xlsx",
        "1057e6db44c589aeb816ed3de158766a09bae6d9611623d1cd7b24fd598d41a6",
        107114,
        "Table S1: every library gene's sgRNA (number, b-number, 20-nt base-pairing "
        "region, full oligo)",
    ),
    _elsevier_file(
        TABLE_S3,
        "mmc4.xlsx",
        "4a7fd186dabaa384baf2f8009843ba95fbf3a88851f61c82cb2b920d1c47079f",
        199690,
        "Table S3: one row per metabolome sample (target gene, b number, sampling OD, "
        "plate, well, replicate, sample id)",
    ),
    _elsevier_file(
        TABLE_S4,
        "mmc5.xlsx",
        "fdc5ac2c759ad82d5cfb5ee1bb3c59fea427a88279e1178ad8e894e072498b84",
        43827170,
        "Table S4: FI-MS fold changes relative to the per-batch median; rows annotated "
        "m/z features, columns the 3,026 samples",
    ),
    _elsevier_file(
        TABLE_S5,
        "mmc6.xlsx",
        "f3a03ddce7b413f5d10ed1ebe8ba4f20c0d32904acd29a1e08edd007893ad62c",
        318926,
        "Table S5: the accumulating strain-metabolite pairs with Mean_FC / R1_FC / "
        "R2_FC; read only to cross-check the stored mean",
    ),
    _elsevier_file(
        TABLE_S9,
        "mmc10.xlsx",
        "46aed36e634f52e239691bf490f698a24e342e46a650681ce50147a520d8e697",
        70584,
        "Table S9: the 802 isobaric metabolites with BiGG id, KEGG id, monoisotopic "
        "mass and neutral formula",
    ),
)
#: ``{raw file name: pinned sha256}``, the build-time check of every consumed file.
DATA_SHA256: dict[str, str] = {f.name: f.sha256 for f in RAW_FILES}
RAW_FILES_BY_NAME: dict[str, RawFile] = {f.name: f for f in RAW_FILES}

#: Released files deliberately not mirrored (the loader does not read them).
NOT_MIRRORED = (
    "MassIVE MSV000098712 (FI-MS), MSV000098755 (LC-MS/MS) and MSV000098714 (MetE "
    "targeted LC-MS/MS): raw spectra, not the released matrices this loader consumes",
    "mmc1.pdf (Figures S1-S12 + Data S3), mmc14.pdf (Data S1 parity plots), mmc16.pdf "
    "(transparent peer review) and mmc17.pdf (the article): captured in the literature "
    "mirror, not read here",
    "mmc3.xlsx (Table S2 growth curves), mmc7.xlsx (Table S6 LC-MS/MS spectra), "
    "mmc8.xlsx (Table S7 non-annotated features), mmc9.xlsx (Table S8 SIRIUS "
    "predictions), mmc11.xlsx (Table S10 iML1515 reactants), mmc12.xlsx (Table S11 "
    "EcoCyc pathways), mmc13.xlsx (Table S12 pathway reactants), mmc15.zip (Data S2 "
    "reference spectra): not consumed",
)


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the sha256-pinned Rapp ``paper.md``."""
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
# Sourced values (verbatim quotes of the pinned OCR, LaTeX markup included)
# --------------------------------------------------------------------------- #
_Q_LIBRARY = (
    "We created an arrayed CRISPRi library that targets all 1,515 genes in the iML1515 "
    "genome-scale model of E. coli metabolism (Figure 1A; Table S1)."
)
_Q_SYSTEM = (
    "The CRISPRi strains have an anhydrotetracycline (aTc)-inducible dCas9 on the "
    "genome and an sgRNA on a plasmid."
)
_Q_HOST = (
    "E. coli YYdCas9 strain19 was the wild-type strain used in this study. All strains "
    "in this study derive from the YYdCas9 strain and are listed in the key resources "
    "table."
)
_Q_GENOTYPE = (
    "<td>YYdCas9: BW25993 intC:tetR-dcas9-aadA lacY:ypet-cat</td><td>Lawson et "
    "al.19</td>"
)
_Q_CONTROL_WELL = (
    "Position A1 of each 96-deep-well plate contains the control strain, which "
    "expresses an empty sgRNA."
)
_Q_CONTROL_COUNT = (
    "Additionally, we collected metabolome samples from 15 replicates of a control "
    "strain carrying a non-targeting sgRNA. This strain was distributed across the "
    "cultivation plates."
)
_Q_REPLICATES = (
    "Each CRISPRi strain was sampled twice from two independent plates to test "
    "reproducibility."
)
_Q_OD_FILTER = (
    "Out of 1,515 CRISPRi strains, 17 did not reach the required OD (optical density "
    "at $6 0 0 \\ \\mathsf { n m } _ { \\rho } ^ { \\prime }$ ) range at the time of "
    "sampling $( 6 . 5 ~ \\mathsf { h }$ after induction) and were not further "
    "analyzed."
)
_Q_EXTRACTS = (
    "Next, we used FI-MS24,25 to screen the 3,026 metabolite extracts (1,498 CRISPRi "
    "strains and 15 control strains, each in biological duplicates)."
)
_Q_GUIDE_CHOICE = (
    "For inclusion in the final library, the sgRNA closest to the start codon was "
    "chosen for each gene."
)
_Q_MEDIA = (
    "Cultivations were performed with LB medium or M9 minimal medium with glucose as "
    "the sole carbon source $( 5 9 ^ { \\star } L ^ { - 1 } )$ . M9 medium contained "
    "(per liter): $7 . 5 2 \\ { \\tt g } \\ N a _ { 2 } { \\sf H P O } _ { 4 } \\ 2 ^ "
    "{ \\star } { \\sf H } _ { 2 } { \\sf O } ,$ 5 ${ \\mathfrak { g } } \\mathsf { K "
    "H } _ { 2 } { \\mathsf { P O } } _ { 4 }$ , 1.5 g $( N H _ { 4 } ) 2 _ { \\mathsf "
    "{ S } } O _ { 4 }$ , $0 . 5 { \\mathfrak { g } }$ NaCl. The following components "
    "were sterilized separately and then added (per liter of final medium): 1 mL 0.1 M "
    "${ \\mathsf { C a C l } } _ { 2 }$ , 1 mL 1 M ${ \\sf M g S O _ { 4 } }$ , $0 . 6 "
    "~ \\mathrm { m L } ~ 0 . 1$ M $\\mathsf { F e C l } _ { 3 }$ , 2 mL $1 . 4 ~ "
    "\\mathsf { m M }$ thiamine-HCl and $1 0 ~ \\mathrm { m L }$ trace salts solution. "
    "The trace salts solution contained (per liter): 180 mg $Z n S O _ { 4 }$ $7 ^ { "
    "\\star } \\mathsf { H } _ { 2 } \\mathsf { O }$ , $1 2 0 \\mathrm { \\ m g \\ C u "
    "C l } _ { 2 }$ $2 ^ { \\star } \\mathsf { H } _ { 2 } \\mathsf { O }$ , $1 2 0 "
    "\\mathrm { \\ m g \\ M n S O _ { 4 } ^ { \\star } H 2 O , }$ , $1 8 0 \\mathrm { "
    "\\ m g \\ C o C l _ { 2 } { ^ { \\star } 6 \\ H _ { 2 } O } }$ ."
)
_Q_SELECTION = (
    "LB, LB agar, and M9 media contained $1 0 0 ~ \\mu \\ g ^ { \\star } \\mathsf { m "
    "} \\mathsf { L } ^ { - 1 }$ ampicillin (Amp). Anhydrotetracycline (aTc) was added "
    "to a final concentration of $2 0 0 ~ \\mathsf { n M }$ to induce expression of the "
    "dCas9 protein in the YYdCas9 strain."
)
_Q_CULTURE = (
    "M9 cultures were diluted to $\\mathrm { O D } _ { 6 0 0 } 0 . 1$ in $^ { 1 , 0 5 0 "
    "\\mu \\ L }$ M9 medium in 96-deep-well plates and incubated for $6 . 5 \\mathsf { "
    "h }$ at $3 7 ^ { \\circ } \\mathrm { C }$ , 220 rpm."
)
_Q_PRECULTURE = (
    "Overnight cultures were prepared in M9 with $1 . 7 5 \\ : \\mathfrak { g } ^ { "
    "\\star } \\mathsf { L } ^ { - 1 }$ glucose."
)
_Q_ANNOTATION = (
    "For annotation to iML1515 metabolites, the findpeaks. $m$ function was used to "
    "pick m/z features in ${ \\mathsf { M } } { \\mathsf { S } } ^ { 1 }$ spectra with "
    "a peak height and prominence cutoff of 5000. Peaks were annotated to iML1515 "
    "metabolites with a mass tolerance of 0.003 Da. Because FI-MS analysis cannot "
    "distinguish compounds with the same mass, all isobaric metabolites were merged "
    "resulting in 802 metabolites with unique monoisotopic masses (Table S9)."
)
_Q_ADDUCTS = (
    "Metabolites with a monoisotopic mass below 1000 Da were annotated in their single "
    "protonated form $( [ M + H ] ^ { + } )$ or their deprotonated form ([M-H]- ). In "
    "addition, metabolites $> 1 0 0 0$ Da were additionally annotated in multiple "
    "charged states: $[ M + 2 H ] ^ { 2 + }$ , $\\left[ M + 3 H \\right] ^ { 3 + }$ "
    "and $[ M - 2 H ] ^ { 2 - }$ , $\\left[ M - 3 H \\right] ^ { 3 - }$ ."
)
_Q_NORMALIZATION = (
    "The intensity of the annotated $m / z$ features was used to calculate fold changes "
    "relative to the median on a per batch basis (Table S4)."
)
_Q_IMPUTATION = (
    "Samples with missing $m / z$ features were baseline imputed if at least one sample "
    "of the batch had an $m / z$ feature with a peak height and prominence of at least "
    "5000."
)
_Q_REPRODUCIBILITY = (
    "with only 37 strains showing a mean relative error greater than $40 \\%$ between "
    "replicates and with $9 9 \\%$ of the variability between biological replicates "
    "being smaller than a ${ \\mathsf { l o g } } _ { 2 }$ -fold change of 1.7 (Data "
    "S1)"
)
_Q_MG1655 = (
    "Pathways of E. coli K-12 substr. MG1655 were extracted from the EcoCyc database"
)
_Q_PLASMID = "<td>pgRNA-bacteria</td><td>Qi e al.20</td><td>Addgene plasmid #44251</td>"
_Q_DEPOSIT = (
    "Metabolome data have been deposited at the MASSIVE repository. FI-MS data are "
    "accessible with the accession number MSV000098712, LC-MS/ MS data with the "
    "accession number MSV000098755, and targeted LC-MS/MS data of MetE strain with "
    "accession number MSV000098714."
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "library_genes": _paper(
        1515,
        _Q_LIBRARY,
        note="Table S1 holds exactly these 1,515 genes; process() refuses another count",
    ),
    "effector": _paper(
        "dCas9",
        _Q_SYSTEM,
        note="the Cas effector of every CrisprConstruct; the key resources table writes "
        "the integrated cassette 'intC:tetR-dcas9-aadA' "
        "(SOURCED_VALUES['host_genotype'])",
    ),
    "host_strain": _paper(
        "YYdCas9",
        _Q_HOST,
        note="EXPERIMENTAL MODEL AND STUDY PARTICIPANT DETAILS, 'Strains and culture'",
    ),
    "host_genotype": _paper(
        "BW25993 intC:tetR-dcas9-aadA lacY:ypet-cat",
        _Q_GENOTYPE,
        note="the KEY RESOURCES TABLE row; BW25993 has no deposited assembly set, so "
        "the background's reference_strain is MG1655 (the namespace the released "
        "b-numbers are written in) and BW25993 is recorded as the parent",
    ),
    "gene_namespace_host": _paper(
        "MG1655",
        _Q_MG1655,
        note="METHOD DETAILS, 'E. coli metabolic pathways and reactants'; the released "
        "identifiers are iML1515 b-numbers and all 1,497 of them resolve on the MG1655 "
        "GenBank annotation (identifier_reconciliation.json)",
    ),
    "guide_choice": _paper(
        1,
        _Q_GUIDE_CHOICE,
        note="one sgRNA per gene, so every CrisprConstruct carries n_guides=1 and "
        "Table S1 holds one row per gene",
    ),
    "guide_plasmid": _paper(
        "pgRNA-bacteria (Addgene plasmid #44251)",
        _Q_PLASMID,
        note="the sgRNA vector; effector_plasmid_ref stays None until plasmids are "
        "first-class ('field now, plasmid later')",
    ),
    "control_sgrna": _paper(
        "empty sgRNA",
        _Q_CONTROL_WELL,
        note="position A1 of each of the 16 library plates; the Results call the same "
        "strain 'non-targeting' (SOURCED_VALUES['control_replicates'])",
    ),
    "control_replicates": _paper(
        15,
        _Q_CONTROL_COUNT,
        note="Table S4 carries exactly 15 'ctrlN' tokens (ctrl2 absent), each with two "
        "replicate columns, so the reference averages 30 samples",
    ),
    "n_biological_replicates": _paper(
        2,
        _Q_REPLICATES,
        note="two independent cultivation plates, so n_replicates counts plates; the "
        "two FI-MS injections per extract are the two polarities, not replication",
    ),
    "od_filter": _paper(
        17,
        _Q_OD_FILTER,
        note="1,515 - 17 = 1,498 analyzed strains. Table S1 holds 18 genes with no "
        "Table S4 sample and Table S4 holds one strain (argR) with no Table S1 sgRNA "
        "and no b-number, which is the one-strain difference; recorded, not resolved",
    ),
    "n_extracts": _paper(
        3026,
        _Q_EXTRACTS,
        note="process() requires Table S4 to carry exactly this many sample columns",
    ),
    "temperature_c": _paper(37.0, _Q_CULTURE),
    "duration_hours": _paper(
        6.5, _Q_CULTURE, note="induced main culture, sampled at the end of it"
    ),
    "aerobic_shaking": _paper(
        "aerobic",
        _Q_CULTURE,
        note="1,050 uL cultures shaken at 220 rpm in 96-deep-well plates; the paper "
        "names no oxygen-limited condition",
    ),
    "media_recipe": _paper(
        "M9 minimal medium with glucose as the sole carbon source (5 g/L)",
        _Q_MEDIA,
        note="the screen medium; MEDIA_LIBRARY has no entry for it (the deferral is "
        "recorded in [[torchcell.datamodels.media]]), so SCREEN_MEDIA is built here "
        "with base_medium='M9'",
    ),
    "glucose_preculture": _paper(
        1.75,
        _Q_PRECULTURE,
        note="the overnight preculture's glucose, NOT the measured culture's; the main "
        "culture is 'M9 medium', which the Media section defines at 5 g/L glucose",
    ),
    "selection_and_inducer": _paper(
        {"ampicillin_ug_per_ml": 100.0, "atc_nm": 200.0},
        _Q_SELECTION,
        note="ampicillin is a medium component (selection agent); aTc is the inducer "
        "and is a SmallMoleculePerturbation on the environment",
    ),
    "isobaric_metabolites": _paper(
        802,
        _Q_ANNOTATION,
        note="process() requires Table S9 to hold exactly 802 rows and Table S4's "
        "feature abbreviations to be exactly those 802",
    ),
    "adducts": _paper(
        ["[M+H]+", "[M-H]-", "[M+2H]2+", "[M+3H]3+", "[M-2H]2-", "[M-3H]3-"],
        _Q_ADDUCTS,
        note="MEASURED on Table S4: 802 [M+H]+, 802 [M-H]-, and 92 each of [M+3H]3+, "
        "[M-2H]2- and [M-3H]3-, totaling 1,880 feature rows. The released table carries "
        "NO [M+2H]2+ row although the Methods name it; recorded, not resolved",
    ),
    "normalization": _paper(
        "linear fold change relative to the per-batch median",
        _Q_NORMALIZATION,
        note="MEASURED at build time (batch_normalization.json): within each of the 6 "
        "batches, every populated feature's median over that batch's samples is exactly "
        "1.0, so the released value is a linear ratio and not a log",
    ),
    "imputation": _paper(
        "baseline imputed per batch",
        _Q_IMPUTATION,
        note="why a populated feature row is populated for every sample; MEASURED: of "
        "1,880 rows, 1,321 are finite in all 3,026 columns and 559 are empty in all of "
        "them, with no partial row",
    ),
    "replicate_noise": _paper(
        1.7,
        _Q_REPRODUCIBILITY,
        note="a dataset-level reproducibility statistic over all strains and features, "
        "not a per-record uncertainty; it is not stored on any record",
    ),
    "data_availability": _paper(
        ["MSV000098712", "MSV000098755", "MSV000098714"],
        _Q_DEPOSIT,
        note="the MassIVE deposits hold raw spectra; the matrices this loader consumes "
        "are the publisher SI workbooks, so the MassIVE files are NOT mirrored",
    ),
}

LIBRARY_GENES: Final[int] = SOURCED_VALUES["library_genes"].value
N_EXTRACTS: Final[int] = SOURCED_VALUES["n_extracts"].value
ISOBARIC_METABOLITES: Final[int] = SOURCED_VALUES["isobaric_metabolites"].value
N_BIOLOGICAL_REPLICATES: Final[int] = SOURCED_VALUES["n_biological_replicates"].value
CONTROL_REPLICATES: Final[int] = SOURCED_VALUES["control_replicates"].value
TEMPERATURE_C: Final[float] = SOURCED_VALUES["temperature_c"].value
DURATION_HOURS: Final[float] = SOURCED_VALUES["duration_hours"].value
CAS_EFFECTOR: Final[str] = SOURCED_VALUES["effector"].value
GUIDES_PER_GENE: Final[int] = SOURCED_VALUES["guide_choice"].value
HOST_STRAIN: Final[str] = SOURCED_VALUES["host_strain"].value
HOST_GENOTYPE: Final[str] = SOURCED_VALUES["host_genotype"].value

#: What the stored number is: the paper's own per-strain statistic (verified exactly
#: against Table S5's ``Mean_FC`` at build time).
MEASUREMENT_TYPE = "fi_ms_iml1515_feature_fold_change_vs_batch_median_mean_of_2_plates"

#: The strain whose GenBank assembly the released b-numbers mean; the host YYdCas9 is a
#: BW25993 derivative, and BW25993 has no deposited assembly set.
REFERENCE_STRAIN_NAME: BacterialReferenceStrain = "MG1655"
MG1655_ASSEMBLY_SET: BacterialAssemblySet = "ecoli_K12_MG1655_ASM584v2"
#: The strain the key resources table writes YYdCas9 as an edit of.
HOST_PARENT_STRAIN = "BW25993"
if BACTERIAL_ASSEMBLY_SETS[REFERENCE_STRAIN_NAME] != MG1655_ASSEMBLY_SET:
    raise RuntimeError(
        f"{REFERENCE_STRAIN_NAME}'s assembly set is "
        f"{BACTERIAL_ASSEMBLY_SETS[REFERENCE_STRAIN_NAME]!r}, not "
        f"{MG1655_ASSEMBLY_SET!r}"
    )

PUBLICATION = Publication(
    pubmed_id=None,
    pubmed_url=None,
    doi=PAPER_DOI,
    doi_url=f"https://doi.org/{PAPER_DOI}",
)
"""The mirrored article prints no PubMed id, so the DOI is the only identifier."""


# --------------------------------------------------------------------------- #
# The screen medium
#
# MEDIA_LIBRARY has no entry for Rapp's M9 ([[torchcell.datamodels.media]] records it
# as a deliberate deferral), and ``media.py`` is a value surface, so the medium is
# built here with ``base_medium="M9"`` (which resolves in the library, so the record
# still joins the M9 family). Every component quotes the one Methods recipe; the
# separately sterilized additions carry the stock-times-volume arithmetic in ``note``,
# and the identity is the anhydrous salt where the recipe weighs a hydrate.
# --------------------------------------------------------------------------- #
_GL = ConcentrationUnit.g_per_l
_MM = ConcentrationUnit.millimolar
_UM = ConcentrationUnit.micromolar
_UG_ML = ConcentrationUnit.ug_per_ml
_SALT = MediaComponentRole.bulk_salt
_TRACE = MediaComponentRole.trace_element


def _component(
    name: str,
    role: MediaComponentRole,
    value: float,
    unit: ConcentrationUnit,
    *,
    note: str | None = None,
) -> MediaComponent:
    """One component of the screen medium, justified by the Methods recipe quote."""
    return MediaComponent(
        compound=resolved_compound(name),
        role=role,
        concentration=Concentration(value=value, unit=unit),
        provenance=[SOURCED_VALUES["media_recipe"]],
        note=note,
    )


_HYDRATE = "the recipe weighs the hydrate; the typed identity is the anhydrous salt"
SCREEN_MEDIA = Media(
    name="M9 glucose (Rapp 2026): 7.52 g/L Na2HPO4 2H2O, 5 g/L KH2PO4, 1.5 g/L "
    "(NH4)2SO4, 0.5 g/L NaCl, 0.1 mM CaCl2, 1 mM MgSO4, 60 uM FeCl3, 2.8 uM "
    "thiamine-HCl, trace salts, 5 g/L glucose, 100 ug/mL ampicillin",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        _component("disodium hydrogen phosphate", _SALT, 7.52, _GL, note=_HYDRATE),
        _component("potassium dihydrogen phosphate", _SALT, 5.0, _GL),
        _component("ammonium sulfate", MediaComponentRole.nitrogen_source, 1.5, _GL),
        _component("sodium chloride", _SALT, 0.5, _GL),
        _component(
            "calcium chloride",
            _SALT,
            0.1,
            _MM,
            note="1 mL of a 0.1 M stock per liter of final medium",
        ),
        _component(
            "magnesium sulfate",
            _SALT,
            1.0,
            _MM,
            note="1 mL of a 1 M stock per liter of final medium",
        ),
        _component(
            "iron(III) chloride",
            _TRACE,
            60.0,
            _UM,
            note="0.6 mL of a 0.1 M stock per liter of final medium",
        ),
        _component(
            "thiamine hydrochloride",
            MediaComponentRole.vitamin,
            2.8,
            _UM,
            note="2 mL of a 1.4 mM stock per liter of final medium",
        ),
        _component(
            "zinc sulfate",
            _TRACE,
            1.8,
            _UG_ML,
            note="180 mg/L in the trace salts solution, added at 10 mL per liter; "
            + _HYDRATE,
        ),
        _component(
            "copper(II) chloride",
            _TRACE,
            1.2,
            _UG_ML,
            note="120 mg/L in the trace salts solution, added at 10 mL per liter; "
            + _HYDRATE,
        ),
        _component(
            "manganese sulfate",
            _TRACE,
            1.2,
            _UG_ML,
            note="120 mg/L in the trace salts solution, added at 10 mL per liter; "
            + _HYDRATE,
        ),
        _component(
            "cobalt(II) chloride",
            _TRACE,
            1.8,
            _UG_ML,
            note="180 mg/L in the trace salts solution, added at 10 mL per liter; "
            + _HYDRATE,
        ),
        _component("D-glucose", MediaComponentRole.carbon_source, 5.0, _GL),
        MediaComponent(
            compound=resolved_compound("ampicillin"),
            role=MediaComponentRole.selection_agent,
            concentration=Concentration(value=100.0, unit=_UG_ML),
            provenance=[SOURCED_VALUES["selection_and_inducer"]],
            note="maintains the sgRNA plasmid pgRNA-bacteria",
        ),
    ],
    provenance=[
        SOURCED_VALUES["media_recipe"],
        SOURCED_VALUES["selection_and_inducer"],
    ],
)
"""Rapp 2026's M9 glucose screen medium, built here because the library has no entry."""

ATC_INDUCTION = SmallMoleculePerturbation(
    compound=resolved_compound("anhydrotetracycline"),
    concentration=Concentration(value=200.0, unit=ConcentrationUnit.nanomolar),
    provenance_gaps=[],
)
"""The inducer that realizes every knockdown; dosed on top of the base medium."""


def environment() -> Environment:
    """Rapp's induced main culture: M9 glucose + 200 nM aTc, 37 C, shaken, 6.5 h."""
    return Environment(
        media=SCREEN_MEDIA,
        temperature=Temperature(value=TEMPERATURE_C),
        perturbations=[ATC_INDUCTION],
        aerobicity=SOURCED_VALUES["aerobic_shaking"].value,
        duration_hours=DURATION_HOURS,
    )


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never a live URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/rappMetabolomeColiCRISPRi2026``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/rappMetabolomeColiCRISPRi2026``."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve_raw_files(
    dest_dir: str | Path, names: Sequence[str] | None = None
) -> dict[str, Path]:
    """Run each file's recorded retriever and write the verified bytes to ``dest_dir``.

    The recorded ``RetrievalRecord`` is what runs (``run_retriever``), so this is the
    re-runnable retrieval itself; a byte mismatch raises before anything is written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for raw in RAW_FILES:
        if names is not None and raw.name not in names:
            continue
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

    ``sources`` maps every name in ``RAW_FILES`` to a local file. Idempotent by sha256:
    a mirror file with the pinned hash is left alone, and one with any other hash
    raises rather than being overwritten.
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
                source=raw.retrieval.source_url,
                retrieval=raw.retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title="The metabolome of an E. coli CRISPRi library identifies benefits of "
        "minimal metabolite levels and targets for engineering",
        files=records,
        si_data_sources=[
            elsevier_mmc_url(ELSEVIER_PII, raw.member) for raw in RAW_FILES
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


# --------------------------------------------------------------------------- #
# Parsing the released workbooks
# --------------------------------------------------------------------------- #
#: A Table S3 / Table S4 sample label: ``<gene>_R<replicate>_<injection>_B<batch>``.
SAMPLE_ID_PATTERN = re.compile(
    r"^(?P<gene>.+)_R(?P<replicate>\d+)_(?P<injection>ms[A-Za-z0-9]+)_B(?P<batch>\d+)$"
)
#: The control strain's tokens in Table S3 / Table S4 (``ctrl1`` .. ``ctrl16``).
CONTROL_TOKEN_PATTERN = re.compile(r"^ctrl\d+$")
#: Table S4's ``Abbr``: the iML1515 isobaric group's abbreviation plus the adduct.
FEATURE_PATTERN = re.compile(r"^(?P<abbreviation>.+?)(?P<adduct>\[M[^\]]*\][-+0-9]*)$")
#: The b-number Table S3 gives a strain with no assigned iML1515 target.
UNASSIGNED_B_NUMBER = "b0000"
#: Isobaric members of one Table S9 row are separated by this in ``Metabolite``.
ISOBARIC_SEPARATOR = "; "
#: Mass of the proton the annotation adds for ``[M+H]+`` (Da), BACK-SOLVED from the
#: pinned tables: Table S4's ``Mass`` minus Table S9's monoisotopic mass is this for
#: every one of the 802 single-protonated rows.
PROTON_MASS = 1.00728
#: Tolerance of that check; the tables round to four and six decimals respectively.
PROTON_MASS_TOLERANCE = 1e-4
#: Tolerance of the per-batch median check (the released values are full-precision
#: ratios to that median, so the median is 1 to floating-point).
BATCH_MEDIAN_TOLERANCE = 1e-9
#: Tolerance of the Table S5 ``Mean_FC`` cross-check; measured max difference is 0.0.
MEAN_FC_TOLERANCE = 1e-9
#: Checklist item 4: below this fraction of released b-numbers resolving to MG1655
#: locus tags the build stops. Measured on the pinned tables: 1.0.
MIN_RESOLVED_FRACTION = 0.99


class SampleId(BaseModel):
    """One Table S4 column / Table S3 row, parsed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    gene: str
    replicate: int
    injection: str
    batch: int


def parse_sample_id(column: str) -> SampleId:
    """Parse a sample label, refusing any other shape."""
    match = SAMPLE_ID_PATTERN.match(column)
    if match is None:
        raise ValueError(f"{column!r} is not a <gene>_R<n>_<injection>_B<n> sample id")
    return SampleId(
        column=column,
        gene=match["gene"],
        replicate=int(match["replicate"]),
        injection=match["injection"],
        batch=int(match["batch"]),
    )


class SampleRow(BaseModel):
    """One Table S3 row: a sample, its strain, its b-number and its sampling OD."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: SampleId
    b_number: str
    optical_density: float
    plate: str
    well: str


class Guide(BaseModel):
    """One Table S1 row: the sgRNA chosen for one library gene."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gene: str
    sgrna_id: str
    b_number: str
    spacer: str


class Metabolite(BaseModel):
    """One Table S9 row: an iML1515 isobaric group of unique monoisotopic mass.

    The group's ARITY is its id count, not its name count: ``BIGG``, ``KEGG`` and
    ``Abbreviation`` all carry one ``-``-joined token per merged metabolite and agree on
    all 802 rows, while the ``; ``-separated ``Metabolite`` string is a display field
    whose arity disagrees on exactly one row (``didp``, whose two names ``DIDP`` and
    ``2'-deoxyinosine-5'-diphosphate(3-)`` are one metabolite under one BiGG and one
    KEGG id). ``names`` is therefore kept verbatim and never counted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    abbreviation: str
    bigg: str
    names: tuple[str, ...]
    kegg: str
    monoisotopic_mass: float
    neutral_formula: str

    @property
    def n_isobaric(self) -> int:
        """Metabolites merged into this group (1 = one identity)."""
        return len(self.bigg_ids)

    @property
    def bigg_ids(self) -> tuple[str, ...]:
        """The group's BiGG ids, in the order Table S9 joins them."""
        return tuple(self.bigg.split("-"))

    @property
    def kegg_ids(self) -> tuple[str, ...]:
        """The group's KEGG ids, in the order Table S9 joins them."""
        return tuple(self.kegg.split("-"))


class Feature(BaseModel):
    """One Table S4 row: an annotated m/z feature of one metabolite group + adduct."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: int
    key: str
    abbreviation: str
    adduct: str
    mz: float
    kegg: str


class FeatureTable(BaseModel):
    """Table S4, parsed: its features, its sample columns and the value matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    features: tuple[Feature, ...]
    samples: tuple[SampleId, ...]
    matrix: FloatMatrix


class Accumulation(BaseModel):
    """One Table S5 row: a strain-metabolite pair and the paper's own ``Mean_FC``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gene: str
    key: str
    mean_fold_change: float


def read_guides(path: str | Path) -> dict[str, Guide]:
    """Table S1's one sgRNA per gene, refusing a repeated gene or a non-ACGT spacer."""
    frame = pd.read_excel(path, sheet_name=TABLE_S1_SHEET, header=0)
    guides: dict[str, Guide] = {}
    for _, row in frame.iterrows():
        spacer = str(row["base pairing region"]).strip().upper()
        if not re.fullmatch(r"[ACGT]+", spacer):
            raise ValueError(f"Table S1 {row['Gene']!r}: spacer {spacer!r} is not ACGT")
        guide = Guide(
            gene=str(row["Gene"]).strip(),
            sgrna_id=str(row["sgRNA Nr."]).strip(),
            b_number=str(row["b-Nr."]).strip(),
            spacer=spacer,
        )
        if guide.gene in guides:
            raise ValueError(f"Table S1 lists {guide.gene!r} more than once")
        guides[guide.gene] = guide
    if len(guides) != LIBRARY_GENES:
        raise ValueError(
            f"Table S1 holds {len(guides)} genes, the paper states {LIBRARY_GENES}"
        )
    return guides


def read_sample_rows(path: str | Path) -> dict[str, SampleRow]:
    """Table S3 keyed by sample id, refusing a repeated id."""
    frame = pd.read_excel(path, sheet_name=TABLE_S3_SHEET, header=0)
    rows: dict[str, SampleRow] = {}
    for _, row in frame.iterrows():
        sample = parse_sample_id(str(row["Sample ID"]).strip())
        if str(row["Target gene"]).strip() != sample.gene:
            raise ValueError(
                f"Table S3 {sample.column!r}: target gene "
                f"{str(row['Target gene']).strip()!r} is not the id's gene"
            )
        if sample.column in rows:
            raise ValueError(f"Table S3 lists {sample.column!r} more than once")
        rows[sample.column] = SampleRow(
            sample=sample,
            b_number=str(row["b number"]).strip(),
            optical_density=float(row["OD"]),
            plate=str(row["Plate ID"]).strip(),
            well=str(row["Well"]).strip(),
        )
    if len(rows) != N_EXTRACTS:
        raise ValueError(
            f"Table S3 holds {len(rows)} samples, the paper states {N_EXTRACTS}"
        )
    return rows


def read_metabolites(path: str | Path) -> dict[str, Metabolite]:
    """Table S9's 802 isobaric groups, keyed by the abbreviation Table S4 joins on."""
    frame = pd.read_excel(path, sheet_name=TABLE_S9_SHEET, header=0)
    metabolites: dict[str, Metabolite] = {}
    for _, row in frame.iterrows():
        metabolite = Metabolite(
            abbreviation=str(row["Abbreviation"]).strip(),
            bigg=str(row["BIGG"]).strip(),
            names=tuple(
                name.strip()
                for name in str(row["Metabolite"]).split(ISOBARIC_SEPARATOR)
            ),
            kegg=str(row["KEGG"]).strip(),
            monoisotopic_mass=float(row["Monoisotopic mass"]),
            neutral_formula=str(row["Neutral Formula"]).strip(),
        )
        if metabolite.abbreviation in metabolites:
            raise ValueError(
                f"Table S9 lists {metabolite.abbreviation!r} more than once"
            )
        arities = {
            len(metabolite.bigg_ids),
            len(metabolite.kegg_ids),
            len(metabolite.abbreviation.split("-")),
        }
        if len(arities) != 1:
            raise ValueError(
                f"Table S9 {metabolite.abbreviation!r}: the abbreviation, BiGG and "
                f"KEGG fields join {sorted(arities)} tokens, not the same number"
            )
        metabolites[metabolite.abbreviation] = metabolite
    if len(metabolites) != ISOBARIC_METABOLITES:
        raise ValueError(
            f"Table S9 holds {len(metabolites)} metabolites, the paper states "
            f"{ISOBARIC_METABOLITES}"
        )
    return metabolites


def read_feature_table(path: str | Path) -> FeatureTable:
    """Table S4's features, sample columns and values, refusing a repeated key."""
    frame = pd.read_excel(path, sheet_name=TABLE_S4_SHEET, header=0)
    label_columns = ["Abbr", "Metabolite", "Mass", "Kegg"]
    if list(frame.columns[: len(label_columns)]) != label_columns:
        raise ValueError(
            f"Table S4 starts with {list(frame.columns[:4])}, expected {label_columns}"
        )
    features: list[Feature] = []
    seen: set[str] = set()
    for row, (key, mz, kegg) in enumerate(
        zip(frame["Abbr"], frame["Mass"], frame["Kegg"], strict=True), start=1
    ):
        label = str(key).strip()
        match = FEATURE_PATTERN.match(label)
        if match is None:
            raise ValueError(f"Table S4 row {row}: {label!r} carries no [M..] adduct")
        if label in seen:
            raise ValueError(f"Table S4 lists feature {label!r} more than once")
        seen.add(label)
        features.append(
            Feature(
                row=row,
                key=label,
                abbreviation=match["abbreviation"],
                adduct=match["adduct"],
                mz=float(mz),
                kegg=str(kegg).strip(),
            )
        )
    samples = tuple(
        parse_sample_id(str(column)) for column in frame.columns[len(label_columns) :]
    )
    if len(samples) != N_EXTRACTS:
        raise ValueError(
            f"Table S4 holds {len(samples)} sample columns, the paper states "
            f"{N_EXTRACTS}"
        )
    return FeatureTable(
        features=tuple(features),
        samples=samples,
        matrix=frame.iloc[:, len(label_columns) :].to_numpy(dtype=np.float64),
    )


def read_accumulations(path: str | Path) -> list[Accumulation]:
    """Table S5's accumulating pairs, as (gene, feature key, the paper's Mean_FC)."""
    frame = pd.read_excel(path, sheet_name=TABLE_S5_SHEET, header=0)
    return [
        Accumulation(
            gene=str(row["Gene"]).strip(),
            key=f"{str(row['Metabolite Abbreviation']).strip()}"
            f"{str(row['Mode']).strip()}",
            mean_fold_change=float(row["Mean_FC"]),
        )
        for _, row in frame.iterrows()
    ]


# --------------------------------------------------------------------------- #
# Build-time checks on the released tables (each refuses, none falls back)
# --------------------------------------------------------------------------- #
class FeatureIdentity(BaseModel):
    """How one Table S4 feature joins Table S9's identity layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    abbreviation: str
    adduct: str
    mz: float
    monoisotopic_mass: float
    neutral_formula: str
    kegg: str
    bigg_ids: tuple[str, ...]
    names: tuple[str, ...]

    @property
    def n_isobaric(self) -> int:
        """Metabolites this feature's mass cannot separate (its BiGG id count)."""
        return len(self.bigg_ids)

    @property
    def target_metabolite_id(self) -> str | None:
        """The one BiGG id, or ``None`` when the feature is a merged isobaric set."""
        return self.bigg_ids[0] if self.n_isobaric == 1 else None


class IdentityLedger(BaseModel):
    """The metabolite-identity status of the features the loader stores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_features: int
    n_single_identity: int
    n_merged_isobaric: int
    n_metabolite_groups: int
    n_single_identity_groups: int
    adduct_histogram: dict[str, int]
    isobaric_size_histogram: dict[int, int]
    merged_candidates: dict[str, tuple[str, ...]]

    @property
    def target_metabolite_ids_covered(self) -> float:
        """Share of stored features carrying one BiGG id."""
        return self.n_single_identity / self.n_features


def join_identities(
    features: Sequence[Feature], metabolites: Mapping[str, Metabolite]
) -> list[FeatureIdentity]:
    """Join every Table S4 feature to its Table S9 row, refusing a mismatch.

    Checks that each feature's abbreviation is a Table S9 abbreviation, that Table S4's
    own ``Kegg`` string equals Table S9's, that every Table S9 row is used, and that a
    single-protonated feature's m/z is the monoisotopic mass plus
    :data:`PROTON_MASS`.
    """
    identities: list[FeatureIdentity] = []
    used: set[str] = set()
    for feature in features:
        metabolite = metabolites.get(feature.abbreviation)
        if metabolite is None:
            raise ValueError(
                f"Table S4 feature {feature.key!r} names abbreviation "
                f"{feature.abbreviation!r}, which Table S9 does not carry"
            )
        if feature.kegg != metabolite.kegg:
            raise ValueError(
                f"{feature.key!r}: Table S4 KEGG {feature.kegg!r} is not Table S9's "
                f"{metabolite.kegg!r}"
            )
        if feature.adduct == "[M+H]+":
            offset = feature.mz - metabolite.monoisotopic_mass
            if abs(offset - PROTON_MASS) > PROTON_MASS_TOLERANCE:
                raise ValueError(
                    f"{feature.key!r}: m/z minus the monoisotopic mass is {offset:.6f}, "
                    f"not the proton mass {PROTON_MASS}"
                )
        used.add(feature.abbreviation)
        identities.append(
            FeatureIdentity(
                key=feature.key,
                abbreviation=feature.abbreviation,
                adduct=feature.adduct,
                mz=feature.mz,
                monoisotopic_mass=metabolite.monoisotopic_mass,
                neutral_formula=metabolite.neutral_formula,
                kegg=metabolite.kegg,
                bigg_ids=metabolite.bigg_ids,
                names=metabolite.names,
            )
        )
    unused = sorted(set(metabolites) - used)
    if unused:
        raise ValueError(f"{len(unused)} Table S9 metabolites have no Table S4 feature")
    return identities


def identity_ledger(identities: Sequence[FeatureIdentity]) -> IdentityLedger:
    """Count the identity status of the features the loader stores."""
    adducts: dict[str, int] = {}
    sizes: dict[int, int] = {}
    merged: dict[str, tuple[str, ...]] = {}
    for identity in identities:
        adducts[identity.adduct] = adducts.get(identity.adduct, 0) + 1
        sizes[identity.n_isobaric] = sizes.get(identity.n_isobaric, 0) + 1
        if identity.target_metabolite_id is None:
            merged[identity.key] = identity.bigg_ids
    groups = {identity.abbreviation for identity in identities}
    single_groups = {
        identity.abbreviation
        for identity in identities
        if identity.target_metabolite_id is not None
    }
    return IdentityLedger(
        n_features=len(identities),
        n_single_identity=len(identities) - len(merged),
        n_merged_isobaric=len(merged),
        n_metabolite_groups=len(groups),
        n_single_identity_groups=len(single_groups),
        adduct_histogram=dict(sorted(adducts.items())),
        isobaric_size_histogram=dict(sorted(sizes.items())),
        merged_candidates=dict(sorted(merged.items())),
    )


class PopulatedFeatures(BaseModel):
    """Which Table S4 rows carry values, with the all-or-nothing rule asserted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_rows: int
    n_columns: int
    populated_rows: tuple[int, ...]
    n_empty_rows: int


def populated_features(table: FeatureTable) -> PopulatedFeatures:
    """The feature rows finite in EVERY sample column, refusing a partial row.

    Per-batch baseline imputation (``SOURCED_VALUES['imputation']``) makes a detected
    feature present for every sample; a row finite in some columns and not others would
    mean a key whose absence differs per strain, which the stored record cannot say.
    """
    matrix = table.matrix
    finite = np.isfinite(matrix)
    per_row = finite.sum(axis=1)
    partial = np.flatnonzero((per_row > 0) & (per_row < matrix.shape[1]))
    if partial.size:
        row = int(partial[0]) + 1
        raise ValueError(
            f"Table S4 has {partial.size} partially populated feature rows, first at "
            f"row {row} ({int(per_row[partial[0]])} of {matrix.shape[1]} values)"
        )
    return PopulatedFeatures(
        n_rows=int(matrix.shape[0]),
        n_columns=int(matrix.shape[1]),
        populated_rows=tuple(int(i) for i in np.flatnonzero(per_row > 0)),
        n_empty_rows=int((per_row == 0).sum()),
    )


class BatchNormalization(BaseModel):
    """The measured per-batch median of every populated feature, by batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch: int
    n_samples: int
    n_features: int
    min_median: float
    max_median: float


def batch_normalization(
    table: FeatureTable, rows: Sequence[int]
) -> list[BatchNormalization]:
    """Check that each batch's populated features have median 1 over that batch.

    This is what makes ``SOURCED_VALUES['normalization']`` a measurement rather than a
    reading of the Methods: a released value is a LINEAR ratio to the batch median, so
    the median must be exactly 1, not 0 (which a log scale would give).
    """
    matrix = table.matrix
    index = np.asarray(rows, dtype=np.int64)
    out: list[BatchNormalization] = []
    for batch in sorted({sample.batch for sample in table.samples}):
        columns = np.asarray(
            [i for i, s in enumerate(table.samples) if s.batch == batch], dtype=np.int64
        )
        medians = np.median(matrix[np.ix_(index, columns)], axis=1)
        result = BatchNormalization(
            batch=batch,
            n_samples=int(columns.size),
            n_features=int(index.size),
            min_median=float(medians.min()),
            max_median=float(medians.max()),
        )
        if max(abs(result.min_median - 1.0), abs(result.max_median - 1.0)) > (
            BATCH_MEDIAN_TOLERANCE
        ):
            raise ValueError(
                f"batch {batch}: populated-feature medians run "
                f"{result.min_median:.9f}..{result.max_median:.9f}, not 1.0, so the "
                "released values are not ratios to the batch median"
            )
        out.append(result)
    return out


class MeanFoldChangeCheck(BaseModel):
    """The stored mean against the paper's own ``Mean_FC`` column (Table S5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_pairs: int
    n_checked: int
    max_abs_difference: float
    tolerance: float


def check_mean_fold_change(
    table: FeatureTable,
    accumulations: Sequence[Accumulation],
    columns_by_gene: Mapping[str, tuple[int, ...]],
) -> MeanFoldChangeCheck:
    """Require the mean of a strain's two plates to be Table S5's ``Mean_FC``.

    Every Table S5 pair must be checkable: a pair naming a feature or a strain Table S4
    does not carry would mean the two tables disagree, which is refused rather than
    skipped.
    """
    matrix = table.matrix
    row_of = {feature.key: i for i, feature in enumerate(table.features)}
    worst = 0.0
    for pair in accumulations:
        row = row_of.get(pair.key)
        columns = columns_by_gene.get(pair.gene)
        if row is None or columns is None:
            raise ValueError(
                f"Table S5 pair ({pair.gene!r}, {pair.key!r}) is not in Table S4"
            )
        mean = float(matrix[row, list(columns)].mean())
        worst = max(worst, abs(mean - pair.mean_fold_change))
    result = MeanFoldChangeCheck(
        n_pairs=len(accumulations),
        n_checked=len(accumulations),
        max_abs_difference=worst,
        tolerance=MEAN_FC_TOLERANCE,
    )
    if worst > MEAN_FC_TOLERANCE:
        raise ValueError(
            f"the mean of the two plates differs from Table S5's Mean_FC by up to "
            f"{worst:.3g}, above {MEAN_FC_TOLERANCE}"
        )
    return result


# --------------------------------------------------------------------------- #
# Strain selection and the retention ledger
# --------------------------------------------------------------------------- #
class StrainSamples(BaseModel):
    """One strain token of Table S4: its two plate columns and its released b-number."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gene: str
    b_number: str
    columns: tuple[int, ...]
    rows: tuple[SampleRow, ...]

    @property
    def is_control(self) -> bool:
        """True for the empty-sgRNA control strain's tokens (``ctrlN``)."""
        return CONTROL_TOKEN_PATTERN.match(self.gene) is not None

    @property
    def has_target(self) -> bool:
        """True when Table S3 assigned this strain an iML1515 target gene."""
        return self.b_number != UNASSIGNED_B_NUMBER


class DropRule(BaseModel):
    """One retention rule, the strains it removed, and why."""

    rule: str
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    strain_tokens: int
    reference_tokens: list[str]
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


def group_samples(
    table: FeatureTable, rows: Mapping[str, SampleRow]
) -> list[StrainSamples]:
    """Group Table S4's columns into strains, refusing anything but two plates each.

    Table S4's columns must be exactly Table S3's sample ids, and each strain must
    carry replicates 1 and 2 under one b-number
    (``SOURCED_VALUES['n_biological_replicates']``).
    """
    columns = [sample.column for sample in table.samples]
    if set(columns) != set(rows):
        raise ValueError("Table S4's sample columns are not Table S3's sample ids")
    by_gene: dict[str, list[tuple[int, SampleRow]]] = {}
    for index, sample in enumerate(table.samples):
        by_gene.setdefault(sample.gene, []).append((index, rows[sample.column]))
    strains: list[StrainSamples] = []
    for gene, entries in by_gene.items():
        entries.sort(key=lambda item: item[1].sample.replicate)
        replicates = [row.sample.replicate for _, row in entries]
        if replicates != list(range(1, N_BIOLOGICAL_REPLICATES + 1)):
            raise ValueError(
                f"{gene!r} carries replicates {replicates}, expected "
                f"{list(range(1, N_BIOLOGICAL_REPLICATES + 1))}"
            )
        b_numbers = {row.b_number for _, row in entries}
        if len(b_numbers) != 1:
            raise ValueError(f"{gene!r} carries b-numbers {sorted(b_numbers)}")
        strains.append(
            StrainSamples(
                gene=gene,
                b_number=b_numbers.pop(),
                columns=tuple(index for index, _ in entries),
                rows=tuple(row for _, row in entries),
            )
        )
    strains.sort(key=lambda strain: strain.gene)
    return strains


class ResolvedStrain(BaseModel):
    """A kept CRISPRi strain with its MG1655 locus tag, symbol and guide."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: StrainSamples
    locus_tag: str
    symbol: str
    guide: Guide


class SymbolDisagreement(BaseModel):
    """A kept strain whose Table S3 gene SYMBOL does not resolve to its b-number's
    locus; the record keeps the b-number's locus either way.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    gene: str
    b_number: str
    locus_tag: str
    symbol_resolution: str


class IdentifierLedger(BaseModel):
    """The b-number reconciliation report plus what the released symbols would say."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reconciliation: LocusTagReconciliation
    min_resolved_fraction: float
    symbol_disagreements: list[SymbolDisagreement]


def canonical_symbol(genome: EcoliK12Genome, tag: str) -> str:
    """The annotation's own gene symbol of ``tag`` when it resolves back to ``tag``,
    else the tag itself, so the stored pair always round-trips through the resolver.
    """
    symbol = genome.genbank.loci[tag].symbol
    if symbol is None:
        return tag
    resolution = genome.resolve_gene_name(symbol)
    return symbol if resolution.systematic_name == tag else tag


def resolve_strains(
    genome: EcoliK12Genome,
    strains: Sequence[StrainSamples],
    guides: Mapping[str, Guide],
    *,
    label: str,
) -> tuple[list[ResolvedStrain], list[DropRule], IdentifierLedger]:
    """Settle every non-control strain: its target, its locus tag and its guide.

    Two retention rules, in order. ``no_target_gene_assigned`` drops a strain whose
    Table S3 b-number is the controls' placeholder and for which Table S1 holds no
    sgRNA, so neither the knocked-down locus nor the guide spacer is released.
    ``b_number_remapped_by_the_annotation`` drops a strain whose released b-number the
    pinned MG1655 annotation does not carry as a locus tag but as a ``/gene_synonym``
    of another locus: storing that locus needs a ``DerivedIdentifierMapping``, and
    ``DerivedIdentifierRoute`` has no member for a retired tag of the pinned strain's
    own namespace, so the record is dropped rather than remapped silently.

    Stops (``LocusTagResolutionError``) below :data:`MIN_RESOLVED_FRACTION`.
    """
    targeted = [strain for strain in strains if strain.has_target]
    unassigned = [strain for strain in strains if not strain.has_target]
    b_numbers = pd.Series([strain.b_number for strain in targeted], dtype=object)
    stored, report = reconcile_locus_tags(genome, b_numbers, label=label)
    report.require_resolved(MIN_RESOLVED_FRACTION)
    pattern = LOCUS_TAG_PATTERNS[report.gene_namespace]

    kept: list[ResolvedStrain] = []
    remapped: list[str] = []
    disagreements: list[SymbolDisagreement] = []
    for strain, tag in zip(targeted, stored.tolist(), strict=True):
        if tag != strain.b_number or pattern.match(tag) is None:
            resolution = genome.resolve_gene_name(strain.b_number)
            remapped.append(
                f"{strain.gene} ({strain.b_number}): the annotation carries it as a "
                f"{resolution.note}, so the record would store {tag}"
            )
            continue
        guide = guides.get(strain.gene)
        if guide is None:
            raise ValueError(
                f"{strain.gene!r} has b-number {strain.b_number} but no Table S1 sgRNA"
            )
        if guide.b_number != strain.b_number:
            raise ValueError(
                f"{strain.gene!r}: Table S1 b-number {guide.b_number} is not Table S3's "
                f"{strain.b_number}"
            )
        symbol_resolution = genome.resolve_gene_name(strain.gene)
        if symbol_resolution.systematic_name != tag:
            disagreements.append(
                SymbolDisagreement(
                    gene=strain.gene,
                    b_number=strain.b_number,
                    locus_tag=tag,
                    symbol_resolution=f"{symbol_resolution.status.value} "
                    f"{symbol_resolution.systematic_name}",
                )
            )
        kept.append(
            ResolvedStrain(
                strain=strain,
                locus_tag=tag,
                symbol=canonical_symbol(genome, tag),
                guide=guide,
            )
        )
    rules = [
        DropRule(
            rule="no_target_gene_assigned",
            description="Table S3 gives the strain the controls' "
            f"{UNASSIGNED_B_NUMBER} placeholder instead of an iML1515 b-number and "
            "Table S1 holds no sgRNA for it, so neither the knocked-down locus nor the "
            "guide spacer is released",
            n_records=len(unassigned),
            items=[
                f"{strain.gene} ({strain.b_number}, plate {strain.rows[0].plate} well "
                f"{strain.rows[0].well}): no Table S1 sgRNA"
                for strain in unassigned
            ],
        ),
        DropRule(
            rule="b_number_remapped_by_the_annotation",
            description="the released b-number is not a locus tag of the pinned "
            "MG1655 annotation but a /gene_synonym of another locus; recording that "
            "remap needs a DerivedIdentifierMapping, and DerivedIdentifierRoute has no "
            "member for a retired tag of the pinned strain's own namespace, so the "
            "record is dropped rather than remapped silently",
            n_records=len(remapped),
            items=remapped,
        ),
    ]
    ledger = IdentifierLedger(
        reconciliation=report,
        min_resolved_fraction=MIN_RESOLVED_FRACTION,
        symbol_disagreements=disagreements,
    )
    return kept, rules, ledger


# --------------------------------------------------------------------------- #
# Record construction (pure, no files)
# --------------------------------------------------------------------------- #
HOST_BACKGROUND_PROVENANCE = [
    SOURCED_VALUES["host_strain"],
    SOURCED_VALUES["host_genotype"],
    SOURCED_VALUES["gene_namespace_host"],
]
HOST_CONSTRUCTION_GAP = ProvenanceGap(
    field="construction",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=PAPER_MD,
        citation_key=CITATION_KEY,
        sha256=PAPER_MD_SHA256,
        method="full KEY RESOURCES TABLE and 'Strains and culture' read",
        page="EXPERIMENTAL MODEL AND STUDY PARTICIPANT DETAILS, 'Strains and culture'",
    ),
    note="this paper writes only the genotype string and cites Lawson 2017 for how "
    "YYdCas9 was built; Lawson 2017 is not in the literature mirror, so no step of "
    "the construction (nor BW25993's own lesions) is quotable here",
)


def host_background() -> BacterialStrainBackground:
    """``YYdCas9``: the dCas9-carrying host every strain of the library derives from.

    ``reference_strain`` is the strain whose namespace the records' identifiers are
    written in (MG1655 b-numbers), ``parents`` names BW25993, which the paper states
    and which has no deposited assembly set, and ``alleles`` is empty because the
    source states only a genotype string.
    """
    return BacterialStrainBackground(
        name=HOST_STRAIN,
        reference_strain=REFERENCE_STRAIN_NAME,
        assembly_set=MG1655_ASSEMBLY_SET,
        parents=[HOST_PARENT_STRAIN],
        construction=None,
        genotype_statement=HOST_GENOTYPE,
        alleles=[],
        provenance=HOST_BACKGROUND_PROVENANCE,
        provenance_gaps=[HOST_CONSTRUCTION_GAP],
    )


def metabolite_phenotype(
    keys: Sequence[str],
    replicates: Sequence[Sequence[float]],
    target_metabolite_ids: Mapping[str, str],
) -> MetabolitePhenotype:
    """One strain's profile: the mean of its plates, that mean's SE, and the identities.

    ``replicates`` holds one list of plate values per key. The level is their
    arithmetic mean, which is the paper's own ``Mean_FC``
    (:func:`check_mean_fold_change`), and the statistic is the standard error of that
    mean over the same plates.
    """
    if len(keys) != len(replicates):
        raise ValueError(f"{len(keys)} keys for {len(replicates)} replicate lists")
    level: dict[str, float] = {}
    standard_error: dict[str, float] = {}
    n_replicates: dict[str, int] = {}
    for key, values in zip(keys, replicates, strict=True):
        array = np.asarray(values, dtype=np.float64)
        level[key] = float(array.mean())
        standard_error[key] = float(array.std(ddof=1) / np.sqrt(array.size))
        n_replicates[key] = int(array.size)
    return MetabolitePhenotype(
        metabolite_level=level,
        metabolite_level_se=standard_error,
        n_replicates=n_replicates,
        measurement_type=MEASUREMENT_TYPE,
        target_metabolite_ids=dict(target_metabolite_ids),
    )


def crispri_genotype(resolved: ResolvedStrain) -> Genotype:
    """The one guide-directed knockdown, named by its MG1655 b-number."""
    return Genotype(
        perturbations=[
            BacterialCrisprInterferencePerturbation(
                systematic_gene_name=resolved.locus_tag,
                perturbed_gene_name=resolved.symbol,
                gene_namespace=STRAIN_GENE_NAMESPACES[REFERENCE_STRAIN_NAME],
                identifier_mapping=None,
                crispr=CrisprConstruct(
                    effector=CAS_EFFECTOR,
                    guide_sequence=resolved.guide.spacer,
                    n_guides=GUIDES_PER_GENE,
                    library_pool=None,
                    effector_plasmid_ref=None,
                ),
            )
        ]
    )


def build_experiment(
    dataset_name: str,
    resolved: ResolvedStrain,
    keys: Sequence[str],
    replicates: Sequence[Sequence[float]],
    target_metabolite_ids: Mapping[str, str],
    env: Environment,
) -> BacterialMetaboliteExperiment:
    """The record of one CRISPRi strain."""
    return BacterialMetaboliteExperiment(
        dataset_name=dataset_name,
        genotype=crispri_genotype(resolved),
        environment=env,
        phenotype=metabolite_phenotype(keys, replicates, target_metabolite_ids),
    )


def build_reference(
    dataset_name: str,
    genome_reference: AssemblyReferenceGenome,
    keys: Sequence[str],
    replicates: Sequence[Sequence[float]],
    target_metabolite_ids: Mapping[str, str],
    env: Environment,
) -> BacterialMetaboliteExperimentReference:
    """The measured control profile (empty sgRNA) on the same per-batch-median scale."""
    return BacterialMetaboliteExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=env.model_copy(),
        phenotype_reference=metabolite_phenotype(
            keys, replicates, target_metabolite_ids
        ),
    )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class MetabolomeRapp2026Dataset(ExperimentDataset):
    """FI-MS metabolome of a CRISPRi library covering every iML1515 gene of E. coli."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "MG1655"

    def __init__(
        self,
        root: str = "data/torchcell/metabolome_rapp2026",
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
        return BacterialMetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialMetaboliteExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Every consumed file, linked from the raw mirror."""
        return [raw.name for raw in RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking the manifest and sha256.

        The mirror plus ``DATA_SHA256`` is canonical; the Elsevier CDN URLs are
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
        log.info("Rapp 2026 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _raw(self, name: str) -> str:
        return osp.join(self.raw_dir, name)

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache (a direct run);
        a genome of another assembly set is refused.
        """
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
        """Parse the five SI workbooks into per-strain records + the LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        guides = read_guides(self._raw(TABLE_S1))
        sample_rows = read_sample_rows(self._raw(TABLE_S3))
        metabolites = read_metabolites(self._raw(TABLE_S9))
        table = read_feature_table(self._raw(TABLE_S4))
        identities = join_identities(table.features, metabolites)

        strains = group_samples(table, sample_rows)
        columns_by_gene = {strain.gene: strain.columns for strain in strains}
        populated = populated_features(table)
        batches = batch_normalization(table, populated.populated_rows)
        mean_check = check_mean_fold_change(
            table, read_accumulations(self._raw(TABLE_S5)), columns_by_gene
        )

        stored = [identities[row] for row in populated.populated_rows]
        ledger = identity_ledger(stored)
        keys = [identity.key for identity in stored]
        target_ids = {
            identity.key: identity.target_metabolite_id
            for identity in stored
            if identity.target_metabolite_id is not None
        }
        matrix = table.matrix[np.asarray(populated.populated_rows, dtype=np.int64), :]

        controls = [strain for strain in strains if strain.is_control]
        if len(controls) != CONTROL_REPLICATES:
            raise ValueError(
                f"Table S4 carries {len(controls)} control strains, the paper states "
                f"{CONTROL_REPLICATES}"
            )
        genome = self._genome()
        resolved, rules, identifiers = resolve_strains(
            genome,
            [strain for strain in strains if not strain.is_control],
            guides,
            label=f"{self.name} iML1515 b-numbers",
        )
        source_records = len(strains) - len(controls)
        drop_log = DropLog(
            dataset=self.name,
            strain_tokens=len(strains),
            reference_tokens=sorted(strain.gene for strain in controls),
            source_records=source_records,
            kept_records=len(resolved),
            dropped_records=source_records - len(resolved),
            rules=rules,
        )
        if sum(rule.n_records for rule in drop_log.rules) != drop_log.dropped_records:
            raise RuntimeError("drop rules do not account for every dropped strain")
        log.info(
            "Rapp 2026: %d strain tokens (%d control) -> %d records; drops %s; "
            "b-number statuses %s; %d released symbols do not resolve to their "
            "b-number's locus; %d of %d stored features carry one BiGG id",
            len(strains),
            len(controls),
            len(resolved),
            {rule.rule: rule.n_records for rule in drop_log.rules},
            {
                s.value: n
                for s, n in identifiers.reconciliation.status_histogram.items()
            },
            len(identifiers.symbol_disagreements),
            ledger.n_single_identity,
            ledger.n_features,
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_ledgers(
            drop_log,
            identifiers,
            ledger,
            batches,
            mean_check,
            populated,
            stored,
            resolved,
        )

        env = environment()
        control_columns = [column for strain in controls for column in strain.columns]
        reference = build_reference(
            self.name,
            assembly_reference(self.REFERENCE_STRAIN, background=host_background()),
            keys,
            matrix[:, control_columns].tolist(),
            target_ids,
            env,
        )
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, item in enumerate(tqdm(resolved, desc="rapp2026")):
                experiment = build_experiment(
                    self.name,
                    item,
                    keys,
                    matrix[:, list(item.strain.columns)].tolist(),
                    target_ids,
                    env,
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        env_out.close()
        interned_env.close()
        log.info("Wrote %d Rapp 2026 metabolome experiments to LMDB", len(resolved))

    def _write_ledgers(
        self,
        drop_log: DropLog,
        identifiers: IdentifierLedger,
        ledger: IdentityLedger,
        batches: Sequence[BatchNormalization],
        mean_check: MeanFoldChangeCheck,
        populated: PopulatedFeatures,
        stored: Sequence[FeatureIdentity],
        resolved: Sequence[ResolvedStrain],
    ) -> None:
        """The drop log, identifier ledger, identity ledger and the two check records."""
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drop_log.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            identifiers.model_dump_json(indent=2)
        )
        (out / "metabolite_identity.json").write_text(ledger.model_dump_json(indent=2))
        (out / "batch_normalization.json").write_text(
            json.dumps(
                {
                    "features": populated.model_dump(exclude={"populated_rows"}),
                    "n_populated_rows": len(populated.populated_rows),
                    "batches": [batch.model_dump() for batch in batches],
                },
                indent=2,
            )
        )
        (out / "mean_fc_crosscheck.json").write_text(
            mean_check.model_dump_json(indent=2)
        )
        pd.DataFrame(
            [
                {
                    "key": identity.key,
                    "abbreviation": identity.abbreviation,
                    "adduct": identity.adduct,
                    "mz": identity.mz,
                    "monoisotopic_mass": identity.monoisotopic_mass,
                    "neutral_formula": identity.neutral_formula,
                    "kegg": identity.kegg,
                    "bigg_ids": "-".join(identity.bigg_ids),
                    "n_isobaric": identity.n_isobaric,
                    "target_metabolite_id": identity.target_metabolite_id or "",
                    "names": ISOBARIC_SEPARATOR.join(identity.names),
                }
                for identity in stored
            ]
        ).to_csv(out / "metabolites.csv", index=False)
        pd.DataFrame(
            [
                {
                    "record": idx,
                    "gene": item.strain.gene,
                    "b_number": item.strain.b_number,
                    "locus_tag": item.locus_tag,
                    "symbol": item.symbol,
                    "sgrna_id": item.guide.sgrna_id,
                    "spacer": item.guide.spacer,
                    "plate": item.strain.rows[0].plate,
                    "well": item.strain.rows[0].well,
                    "batch": item.strain.rows[0].sample.batch,
                    "optical_density": ISOBARIC_SEPARATOR.join(
                        f"{row.optical_density:.6f}" for row in item.strain.rows
                    ),
                }
                for idx, item in enumerate(resolved)
            ]
        ).to_csv(out / "strains.csv", index=False)

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
DATASET_ROOT_REL = "data/torchcell/metabolome_rapp2026"
VERIFIER_PROVENANCE = Provenance(
    source_uri=f"torchcell-raw/{CITATION_KEY}/data/{TABLE_S4}",
    citation_key=CITATION_KEY,
    sha256=DATA_SHA256[TABLE_S4],
    method="FI-MS iML1515 feature fold changes relative to the per-batch median, mean "
    "of the strain's two plates; reference = the 15 empty-sgRNA control strains",
    page="Cell Syst 2026 Table S4 (mmc5.xlsx, sheet Table_S4)",
)


def run_verification(data_root: str | None = None) -> VerificationReport:
    """Run the metabolite family verifier (L0-L3), the bacterial L4 containment and
    the provenance audit of every ``SOURCED_VALUES`` entry on the built dev LMDB, and
    write ``preprocess/verification_report.json``.
    """
    from torchcell.verification.metabolite import (
        metabolite_gene_set,
        verify_metabolite_dataset,
    )
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _write_report,
        load_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, DATASET_ROOT_REL)
    records = load_records(abs_root)
    drops = DropLog.model_validate_json(
        Path(abs_root, "preprocess", "dropped_records.json").read_text()
    )
    report = verify_metabolite_dataset(
        records,
        dataset_name="metabolome_rapp2026",
        provenance=VERIFIER_PROVENANCE,
        expected_count=drops.kept_records,
        reference_centered=False,
    )
    references = {
        json.dumps(r["reference"]["genome_reference"], sort_keys=True) for r in records
    }
    universe: set[str] = set()
    for ref in references:
        universe |= _gene_set_for_reference(json.loads(ref), base)
    knocked_down = metabolite_gene_set(records)
    missing = sorted(knocked_down - universe)
    report.add(
        LevelResult(
            level=Level.L4,
            name="gene_containment_mg1655_b_numbers",
            passed=not missing,
            message=f"{len(knocked_down) - len(missing)} of {len(knocked_down)} "
            "knocked-down loci are MG1655 GenBank gene rows",
            details={
                "n_knocked_down": len(knocked_down),
                "n_universe": len(universe),
                "missing_examples": missing[:20],
            },
        )
    )
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, Path(base) / "torchcell-library"))
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(prog="python -m torchcell.datasets.ecoli.rapp2026")
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser("deposit", help="deposit the mirror from the SI workbooks")
    deposit.add_argument(
        "--download-dir",
        help="directory holding the retrieved workbooks; omitted means the literature "
        "mirror's si/ directory (the same bytes, already sha256-pinned there)",
    )
    deposit.add_argument(
        "--retrieve",
        action="store_true",
        help="run every Elsevier retriever into --download-dir first",
    )
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        download = (
            Path(args.download_dir)
            if args.download_dir
            else library_dir(data_root) / "si"
        )
        if args.retrieve:
            if not args.download_dir:
                raise SystemExit("--retrieve needs --download-dir")
            retrieve_raw_files(download)
        print(
            deposit_raw_mirror(
                sources={name: download / name for name in DATA_SHA256},
                data_root=data_root,
            )
        )
        return 0
    if args.command == "build":
        dataset = MetabolomeRapp2026Dataset(root=osp.join(data_root, DATASET_ROOT_REL))
        print(f"len = {len(dataset)}")
        return 0
    report = run_verification(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
