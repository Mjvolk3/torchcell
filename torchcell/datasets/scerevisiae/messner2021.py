# torchcell/datasets/scerevisiae/messner2021
# [[torchcell.datasets.scerevisiae.messner2021]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/messner2021
# Test file: tests/torchcell/datasets/scerevisiae/test_messner2021.py
"""Messner 2021 Scanning SWATH drug-response proteomes of wild-type yeast (Nat Biotechnol).

Messner et al. 2021 (Nat Biotechnol, doi:10.1038/s41587-021-00860-4, PMID 33767396) measured the
proteomes of one prototrophic BY4741 strain grown in synthetic minimal medium and treated
for one overnight culture with each of 16 drugs (statins, azoles, antifolates;
Supplementary Table 4) plus folic acid and FIN56, at 10 uM, against DMSO-treated controls.
``ProteomeMessner2021Dataset`` emits one ``ProteinAbundanceExperiment`` per drug-treated
proteome run: genotype = the reference strain (no perturbation), environment = the
medium plus the compound, phenotype = per-protein DIA-NN MaxLFQ quantities.

WHAT IS RELEASED (measured from the bytes, 2026-10-11). The paper's tables carry no protein
values. PRIDE PXD023613 holds the raw ``.wiff`` spectra and, for the drug screen, ONE
processed file: the DIA-NN 1.7.12 main report ``20200914_sswath_yeast_drugscreen.tsv``
(741,477,090 bytes, 1,296,112 precursor rows, 102 runs). Its runs are the 95 drug-plate
injections named ``<injection>_plate_2_wt_<well>_<agent>_<token>`` and seven pooled-digest
QC injections ``NA_NA_NA_QC_<n>_QC_NA``. The authors' own ``diann_outputfileinfo.txt``
names the report as the yeast drug screen. Both are deposited in the raw mirror
``$DATA_ROOT/torchcell-raw/messnerUltrafastProteomicsScanning2021/`` with per-file
provenance; the raw spectra are NOT (no loader consumes them; the accession is recorded).

THE PLATE (measured). Injection ``i`` is well ``96 + i``, and the 96 injections are four
blocks of 24 (18 agents then six DMSO), the paper's "blocked by replicates" design. The
block's agent order is Supplementary Table 4's row order once FIN56 and folic acid are set
aside, which is what maps the file tokens (``Rovustatin``, ``Pravastatine``,
``Lovastatine``) to the table's compounds; ``process`` refuses a release where it is not.
Injection 2 (block 1's fluvastatin well) is in neither the report nor the raw files, so
fluvastatin has three runs. The trailing number of a run name is not described anywhere in
the paper, the SI or PRIDE; it is kept verbatim in the run ledger and never interpreted.

THE VALUE. ``Genes.MaxLFQ.Unique``: DIA-NN's MaxLFQ quantity computed from proteotypic
precursors, linear, one value per (run, protein group). The paper's own rules define which
cells are quantities, applied in its order and reproduced from the bytes:

1. "Only proteins identified with proteotypic (that is, specific) peptides and 0.01 protein
   q-value were considered": ``Proteotypic == 1`` and ``Protein.Q.Value <= 0.01``
   (4,244 protein groups in 102 runs; every proteotypic group is one UniProt accession).
2. "proteins were considered only if detected in >50% of the samples": over the 102
   injections of the report, 2,200 proteins. The per-run mean after this rule is 1,976.4,
   the paper's "On average 1,980 unique proteins (1% FDR)" over its "103 injections"; the
   per-run mean without it is 2,279.3, so the rule is part of the stated proteome.
3. "samples were removed if they had <80% of the maximum identification number": after
   rule 2, ONE run falls below (``5_plate_2_wt_101_Pemetrexed_NA``, 1,096 identifications
   before rule 2 against a threshold of 1,706.4). Applied before rule 2 instead, the rule
   would remove 26 runs, which contradicts the paper's quadruplicates; with rule 3 as read
   here, the 16 Table S4 drugs keep 62 runs, the "n=62 for biological samples" of
   Supplementary Fig. 4b.

No protein is refused. P03872 (REP2), a 2-micron plasmid protein that no feature of the
S288C R64 GFF names, is stored under its released UniProt accession (the ledger lists it
under ``accessions_stored_as_released``); the data is the data, and a key the genome layer
cannot join is a modeling-time fact, not a reason to drop the measurement. Every other
accession is keyed by the ORF the SGD GFF cross-references give it (the Messner 2023
mapper); an accession the GFF does not carry is likewise stored under the accession.

THE REFERENCE. The paper's contrast is "drug-treated versus DMSO-treated", so the
reference is the 24 DMSO runs: per protein the mean, its standard error (SD / sqrt(n)) and
n = the DMSO runs that quantified it, restricted to the proteins each record measured (the
Messner 2023 shape, so reference keys equal experiment keys). Each drug run is one
acquisition: ``n_replicates = 1``, ``protein_abundance_se = None``.

ENVIRONMENT. Synthetic minimal liquid medium at 30 C ("incubated at 30 C"), with the
compound at 10 uM as a ``DoseBasis.fixed`` dose. The paper names the medium but prints no
recipe and cites none, so the medium is a ``composition_deferred`` object local to this
module (the ``SM_DEFERRED`` treatment of Zelezniak 2018), never Mulleder's grams. The
vehicle is recorded as DMSO at an unstated fraction, because the control is "DMSO-treated";
the reference environment carries DMSO itself as a fixed dose of unstated value.

GENOTYPE. "The auxotrophic S. cerevisiae strain BY4741 (his3, leu2, ura3, met15) was
rendered prototrophic by genomic knock-in of the missing genes": the reference is BY4741
and every record's genotype is empty. The four knock-ins are not modeled (as in the Messner
2023 and Mulleder loaders): ``ProteinAbundanceExperimentReference`` takes the base
``ReferenceGenome``, which has no background slot (issue #1032).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import os.path as osp
import pickle
import re
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import lmdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    Publication,
    ReferenceGenome,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.messner2023 import build_uniprot_to_orf_map
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import SourcedValue

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "messnerUltrafastProteomicsScanning2021"
DOI = "10.1038/s41587-021-00860-4"
#: The PubMed id PRIDE PXD023613's project record lists for this paper.
PUBMED_ID = "33767396"
TITLE = "Ultra-fast proteomics with Scanning SWATH"
PRIDE_ACCESSION = "PXD023613"
PRIDE_FTP = "https://ftp.pride.ebi.ac.uk/pride/data/archive/2021/03/PXD023613"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "07756a757d7378503f2b284c3d0ad2896446a9bee94435120ebf6715484b0e20"
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "30bb324267d8c64057cfcd629fbb6bc57b278d6ff40be8a55f7c4329ce90792d"

REPORT_FILENAME = "20200914_sswath_yeast_drugscreen.tsv"
REPORT_REL = f"data/pride/{REPORT_FILENAME}"
REPORT_SHA256 = "0416225401808bfb17ede25471f047cd6fb17897fd8d5e29cefc8d19a00afdfe"
REPORT_URL = f"{PRIDE_FTP}/{REPORT_FILENAME}"
INFO_FILENAME = "diann_outputfileinfo.txt"
INFO_REL = f"data/pride/{INFO_FILENAME}"
INFO_SHA256 = "008924ee01c4151eb74453731680a722cbd67101f45c862c3ae926b30b8f148f"
INFO_URL = f"{PRIDE_FTP}/{INFO_FILENAME}"
RAW_RETRIEVED_AT = "2026-10-11"

MEASUREMENT_TYPE = "swath_ms_diann_maxlfq_proteotypic_quantity"

#: Columns of the DIA-NN 1.7.12 main report the loader reads.
REPORT_COLUMNS = [
    "File.Name",
    "Protein.Group",
    "Genes",
    "Genes.MaxLFQ.Unique",
    "Protein.Q.Value",
    "Proteotypic",
]
#: The directory and suffix every ``File.Name`` of the report carries.
FILE_NAME_PREFIX = "H:\\Christoph\\sswath\\20200910_sswath_drugscreen_yeast\\"
FILE_NAME_SUFFIX = ".wiff.dia"

PROTEIN_Q_MAX = 0.01
DETECTION_FRACTION = 0.5  # a protein is kept when detected in MORE than this fraction
IDENTIFICATION_FRACTION = 0.8  # a run is removed below this fraction of the maximum
DOSE_UM = 10.0
TEMPERATURE_C = 30.0
BLOCK_SIZE = 24
WELL_OFFSET = 96
DMSO_TOKEN = "DMSO"

PLATE_RUN_RE = re.compile(
    r"^(?P<injection>\d+)_plate_(?P<plate>\d+)_wt_(?P<well>\d+)_"
    r"(?P<agent>.+)_(?P<token>-?[0-9.]+|NA)$"
)
QC_RUN_RE = re.compile(r"^NA_NA_NA_QC_(?P<qc>\d+)_QC_NA$")

#: Accessions the S288C R64 GFF names no feature for, with the reason they have no ORF
#: key; they are stored under the accession itself. Any other unmapped accession is
#: stored the same way and listed in the ledger beside these.
ACCESSIONS_WITHOUT_ORF: dict[str, str] = {
    "P03872": "REP2, a 2-micron plasmid protein: no feature of the S288C R64 GFF names "
    "it, so the record keys it by its released UniProt accession"
}

#: Measured from the release (2026-10-11); the build and ``verify_build`` assert them.
EXPECTED_RUNS = 102
EXPECTED_PLATE_RUNS = 95
EXPECTED_QC_RUNS = 7
EXPECTED_ABSENT_INJECTIONS = [2]
EXPECTED_OUTLIERS = ["5_plate_2_wt_101_Pemetrexed_NA"]
EXPECTED_RECORDS = 70
EXPECTED_REFERENCE_RUNS = 24
EXPECTED_TABLE_S4_RECORDS = 62  # Supplementary Fig. 4b: "n=62 for biological samples"
PAPER_MEAN_IDENTIFICATIONS = 1980  # "On average 1,980 unique proteins ... per run"
MIN_SGD_CONTAINMENT = 0.99
MIN_MESSNER2023_REFERENCE_PEARSON = 0.6
MESSNER2023_ROOT = "data/torchcell/proteome_messner2023"


# --------------------------------------------------------------------------- #
# Sourced statements (verbatim substrings of the mirrored OCR)
# --------------------------------------------------------------------------- #
def _paper_sv(value: object, quote: str, note: str | None = None) -> SourcedValue:
    """A SourcedValue pinned to the mirrored ``paper.md``."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=PAPER_MD, citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
        ),
        quote=quote,
        note=note,
    )


def _si_sv(value: object, quote: str, note: str | None = None) -> SourcedValue:
    """A SourcedValue pinned to the mirrored OCR of the Supplementary Information."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=SI1_MD, citation_key=CITATION_KEY, sha256=SI1_MD_SHA256
        ),
        quote=quote,
        note=note,
    )


STRAIN_QUOTE = (
    "The auxotrophic S. cerevisiae strain BY4741 (\u2206his3, \u2206leu2, \u2206ura3, "
    "\u2206met15) was rendered prototrophic by genomic knock-in of the missing genes."
)
STRAIN_SOURCE_QUOTE = (
    "Wild-type Saccharomyces cerevisiae strain BY4741 was obtained from EuroSCARF "
    "(Yo0000)."
)
MEDIUM_QUOTE = (
    "This prototrophic, wild-type strain was grown on agar plates containing synthetic "
    "minimal medium for 3 days. Subsequently, colonies were inoculated in synthetic "
    "minimal liquid medium $2 5 \\mathrm { m l } )$ ) and incubated at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ for 1 day."
)
DOSE_QUOTE = (
    "drugs were added to achieve a working concentration of $1 0 \\mu \\mathrm { M }$ "
    "(1 ml total volume per well). The yeast culture was incubated at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ and was grown overnight to exponential phase."
)
REPLICATE_QUOTE = (
    "Samples were measured in quadruplicate and pooled samples were repeatedly injected "
    "for quality control."
)
BLOCK_QUOTE = "Randomization drug response measurements were blocked by replicates."
DIANN_QUOTE = (
    "Raw data processing was carried out with DIA\u2013NN v.1.7.12 and with default "
    "settings in \u2018robust LC (high accuracy)\u2019 mode. Protein quantities were "
    "obtained using the MaxLFQ algorithm84 as implemented in either DIA\u2013NN (yeast "
    "drug screen)"
)
FILTER_QUOTE = (
    "For the analysis of yeast drug screen data, proteins were considered only if "
    "detected in ${ > } 5 0 \\%$ of the samples, and samples were removed if they had "
    "${ < } 8 0 \\%$ of the maximum identification number across samples. Only proteins "
    "identified with proteotypic (that is, specific) peptides and 0.01 protein $q$ "
    "-value were considered."
)
CONTRAST_QUOTE = (
    "The differential expression analysis (drug-treated versus DMSO-treated) for the "
    "yeast drug screen was done on the $\\log _ { 2 }$ -transformed protein quantities"
)
EXCLUSION_QUOTE = (
    "samples treated with folic acid and FIN56 were excluded from the analysis because "
    "these do not belong to the three studied drug classes."
)
MEAN_IDS_QUOTE = (
    "On average 1,980 unique proteins ( $1 \\%$ FDR) (Supplementary Fig. 4a) were "
    "identified per run."
)
INJECTIONS_QUOTE = (
    "Scanning SWATH identifies on average 1,980 protein groups $1 \\%$ FDR) in 103 "
    "injections of digested Saccharomyces cerevisiae (quality control and samples)."
)
BIOLOGICAL_N_QUOTE = "$\\scriptstyle { \\mathtt { n } } = 6 2$ for biological samples"
ACCESSION_QUOTE = "with the dataset identifier PXD023613"

#: Supplementary Table 4, in its row order: (Compound, Class). The OCR splits the table
#: in two at the page break; each row is quoted as its own cell pair.
TABLE_S4: list[tuple[str, str]] = [
    ("Rosuvastatin Calcium", "Statin"),
    ("Fluvastatin Sodium", "Statin"),
    ("Pyrimethamine", "Antifolate"),
    ("Pitavastatin Calcium", "Statin"),
    ("Pemetrexed Disodium Hydrate", "Antifolate"),
    ("Pralatrexate", "Antifolate"),
    ("Pravastatin Sodium", "Statin"),
    ("Clotrimazole", "Azole"),
    ("Miconazole Nitrate", "Azole"),
    ("Lovastatin", "Statin"),
    ("Ketoconazole", "Azole"),
    ("Atorvastatin", "Statin"),
    ("Methotrexate Disodium", "Antifolate"),
    ("Simvastatin", "Statin"),
    ("Uniconazole", "Azole"),
    ("Itraconazole", "Azole"),
]


def table_s4_quote(label: str, drug_class: str) -> str:
    """The OCR'd cell pair of one Supplementary Table 4 row."""
    return (
        f"<td rowspan=1 colspan=1>{label}</td><td rowspan=1 colspan=1>{drug_class}</td>"
    )


SOURCED: dict[str, SourcedValue] = {
    "strain": _paper_sv(
        "prototrophic BY4741 (his3, leu2, ura3, met15 restored by knock-in)",
        STRAIN_QUOTE,
        note="the four knock-ins are not modeled; the reference genome is BY4741",
    ),
    "strain_source": _paper_sv("BY4741 from EUROSCARF", STRAIN_SOURCE_QUOTE),
    "medium": _paper_sv(
        "synthetic minimal liquid medium, 30 C",
        MEDIUM_QUOTE,
        note="no recipe and no citation for the medium anywhere in the paper or SI",
    ),
    "dose": _paper_sv(
        "10 uM every agent; 30 C overnight to exponential phase",
        DOSE_QUOTE,
        note="the plate-wide statement, so it also covers folic acid and FIN56",
    ),
    "replicates": _paper_sv("quadruplicate", REPLICATE_QUOTE),
    "blocks": _paper_sv("four replicate blocks of 24 wells", BLOCK_QUOTE),
    "diann": _paper_sv("DIA-NN 1.7.12 MaxLFQ", DIANN_QUOTE),
    "filters": _paper_sv(
        "proteotypic, protein q <= 0.01, detected in > 50%, runs >= 80% of max ids",
        FILTER_QUOTE,
    ),
    "contrast": _paper_sv("reference = DMSO-treated runs", CONTRAST_QUOTE),
    "exclusion": _paper_sv(
        "folic acid and FIN56 were treated and measured",
        EXCLUSION_QUOTE,
        note="excluded from the paper's analysis for scope, not quality; served here",
    ),
    "mean_ids": _paper_sv(1980, MEAN_IDS_QUOTE),
    "injections": _si_sv(103, INJECTIONS_QUOTE),
    "biological_n": _si_sv(62, BIOLOGICAL_N_QUOTE),
    "accession": _paper_sv(PRIDE_ACCESSION, ACCESSION_QUOTE),
    **{
        f"table_s4_{index}": _si_sv(
            {"compound": label, "class": drug_class}, table_s4_quote(label, drug_class)
        )
        for index, (label, drug_class) in enumerate(TABLE_S4)
    },
}


# --------------------------------------------------------------------------- #
# Agents: the report's file tokens -> Supplementary Table 4 (or the Methods)
# --------------------------------------------------------------------------- #
class AgentSpec(BaseModel):
    """One treatment of the plate: its file token, its source label and its class."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str = Field(description="the agent token of the run names in the report")
    label: str = Field(description="the compound as the paper names it")
    drug_class: str | None = Field(
        description="Supplementary Table 4's Class; None for the two Methods agents"
    )


AGENTS: dict[str, AgentSpec] = {
    spec.token: spec
    for spec in [
        AgentSpec(
            token="Rovustatin", label="Rosuvastatin Calcium", drug_class="Statin"
        ),
        AgentSpec(token="Fluvastatin", label="Fluvastatin Sodium", drug_class="Statin"),
        AgentSpec(
            token="Pyrimethamine", label="Pyrimethamine", drug_class="Antifolate"
        ),
        AgentSpec(
            token="Pitavastatin", label="Pitavastatin Calcium", drug_class="Statin"
        ),
        AgentSpec(
            token="Pemetrexed",
            label="Pemetrexed Disodium Hydrate",
            drug_class="Antifolate",
        ),
        AgentSpec(token="Pralatrexate", label="Pralatrexate", drug_class="Antifolate"),
        AgentSpec(
            token="Pravastatine", label="Pravastatin Sodium", drug_class="Statin"
        ),
        AgentSpec(token="Clotrimazole", label="Clotrimazole", drug_class="Azole"),
        AgentSpec(token="Miconazole", label="Miconazole Nitrate", drug_class="Azole"),
        AgentSpec(token="Lovastatine", label="Lovastatin", drug_class="Statin"),
        AgentSpec(token="Ketoconazole", label="Ketoconazole", drug_class="Azole"),
        AgentSpec(token="Atorvastatin", label="Atorvastatin", drug_class="Statin"),
        AgentSpec(
            token="Methotrexate", label="Methotrexate Disodium", drug_class="Antifolate"
        ),
        AgentSpec(token="Simvastatin", label="Simvastatin", drug_class="Statin"),
        AgentSpec(token="FIN56", label="FIN56", drug_class=None),
        AgentSpec(token="Uniconazole", label="Uniconazole", drug_class="Azole"),
        AgentSpec(token="Itraconazole", label="Itraconazole", drug_class="Azole"),
        AgentSpec(token="Folic_acid", label="folic acid", drug_class=None),
    ]
}


class PlateLayoutError(ValueError):
    """The release's plate does not have the layout the token mapping rests on."""


def check_plate_layout(agent_tokens_by_injection: dict[int, str]) -> None:
    """Refuse a plate whose blocks do not repeat Supplementary Table 4's order.

    Every block of 24 injections holds the 18 agents in one fixed order followed by six
    DMSO wells; that order minus FIN56 and folic acid must be Table S4's row order, which
    is what ties each misspelled token to its compound. A missing injection (the release
    lacks injection 2) is allowed; a misplaced or unknown agent is not.
    """
    unknown = sorted(
        {t for t in agent_tokens_by_injection.values() if t not in AGENTS}
        - {DMSO_TOKEN}
    )
    if unknown:
        raise PlateLayoutError(f"agent tokens with no compound mapping: {unknown}")
    order = [*AGENTS, *([DMSO_TOKEN] * 6)]
    for injection, token in agent_tokens_by_injection.items():
        expected = order[(injection - 1) % BLOCK_SIZE]
        if token != expected:
            raise PlateLayoutError(
                f"injection {injection} is {token!r}; the block layout puts "
                f"{expected!r} there"
            )
    table_order = [AGENTS[t].label for t in AGENTS if AGENTS[t].drug_class is not None]
    if table_order != [label for label, _ in TABLE_S4]:
        raise PlateLayoutError("the plate order is not Supplementary Table 4's order")
    for spec in AGENTS.values():
        if (
            spec.drug_class is not None
            and (spec.label, spec.drug_class) not in TABLE_S4
        ):
            raise PlateLayoutError(f"{spec.label}: class differs from Table S4")


# --------------------------------------------------------------------------- #
# Medium: named, never stated
# --------------------------------------------------------------------------- #
SM_UNSTATED_MESSNER2021 = Media(
    name="SM (synthetic minimal, composition not stated by Messner 2021), liquid",
    state="liquid",
    is_synthetic=True,
    base_medium=None,
    components=[
        MediaComponent(
            compound=Compound(
                name="synthetic minimal (SM) medium, composition not stated"
            ),
            role=MediaComponentRole.other,
            definition=ComponentDefinition.composition_deferred,
            provenance=[SOURCED["medium"]],
            note="Messner 2021 names 'synthetic minimal medium' for the plates and the "
            "liquid culture and states no recipe, cites no protocol for it, and the SI "
            "adds none. The same lab's Mulleder 2016 and Messner 2023 print 6.7 g/L YNB "
            "without amino acids + 2% glucose, but that identification is nowhere "
            "stated here, so the grams are not copied in (the SM_DEFERRED treatment)",
        )
    ],
    provenance=[SOURCED["medium"]],
)


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
RunRole = Literal["experiment", "reference", "qc", "outlier"]


class RunEntry(BaseModel):
    """One run of the report and what the build did with it."""

    model_config = ConfigDict(extra="forbid")

    run: str
    kind: Literal["plate", "qc"]
    injection: int | None
    well: int | None
    block: int | None = Field(description="replicate block, (injection - 1) // 24")
    agent_token: str | None
    compound_label: str | None
    trailing_token: str | None = Field(
        description="the run name's last field, verbatim; its meaning is not stated"
    )
    n_identified: int = Field(description="proteins passing rule 1 (q, proteotypic)")
    n_after_detection_rule: int = Field(description="of those, kept by rule 2")
    role: RunRole
    record_index: int | None = None


class BuildLedger(BaseModel):
    """Counts and refusals of one build, written to ``preprocess/build_ledger.json``."""

    model_config = ConfigDict(extra="forbid")

    report_sha256: str
    n_report_rows: int
    n_runs: int
    n_plate_runs: int
    n_qc_runs: int
    absent_injections: list[int]
    n_proteins_identified: int = Field(description="rule 1, any run")
    n_proteins_kept: int = Field(description="rule 2")
    n_cells_rule1: int = Field(description="(run, protein) cells passing rule 1")
    n_cells_dropped_by_detection_rule: int
    n_cells_without_quantity: int = Field(
        description="rule-1 precursor rows whose Genes.MaxLFQ.Unique is empty"
    )
    max_identifications: int
    identification_threshold: float
    mean_identifications_after_detection_rule: float
    mean_identifications_rule1_only: float
    outliers: list[str]
    accessions_stored_as_released: dict[str, str] = Field(
        description="kept accessions with no ORF in the SGD GFF, keyed by the "
        "accession itself, each with the reason"
    )
    n_records: int
    n_reference_runs: int
    n_table_s4_records: int
    accession_to_orf: dict[str, str]
    runs: list[RunEntry]


# --------------------------------------------------------------------------- #
# Release parsing (pure; the tests drive these on synthetic reports)
# --------------------------------------------------------------------------- #
def run_name(file_name: str) -> str:
    """The run stem of one ``File.Name``: its Windows path and ``.wiff.dia`` removed."""
    if not (
        file_name.startswith(FILE_NAME_PREFIX) and file_name.endswith(FILE_NAME_SUFFIX)
    ):
        raise ValueError(f"unexpected File.Name {file_name!r}")
    return file_name[len(FILE_NAME_PREFIX) : -len(FILE_NAME_SUFFIX)]


def read_report(path: str | Path) -> pd.DataFrame:
    """The report's needed columns, with a ``run`` column; types are checked."""
    report = pd.read_csv(
        path,
        sep="\t",
        usecols=REPORT_COLUMNS,
        dtype={"File.Name": str, "Protein.Group": str, "Genes": str},
    )
    report["run"] = [run_name(name) for name in report["File.Name"]]
    return report


def identified_cells(report: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rule 1 cells: one quantity per (run, protein group); and the empty-quantity count.

    A proteotypic precursor names one protein group, and every precursor of a (run,
    group) carries the same ``Genes.MaxLFQ.Unique``; a release where they differ raises.
    """
    rule1 = report[
        (report["Proteotypic"] == 1) & (report["Protein.Q.Value"] <= PROTEIN_Q_MAX)
    ]
    multi = rule1["Protein.Group"].str.contains(";")
    if bool(multi.any()):
        raise ValueError(f"{int(multi.sum())} proteotypic rows name several proteins")
    empty = int(rule1["Genes.MaxLFQ.Unique"].isna().sum())
    rule1 = rule1[rule1["Genes.MaxLFQ.Unique"].notna()]
    grouped = rule1.groupby(["run", "Protein.Group"])["Genes.MaxLFQ.Unique"]
    distinct = grouped.nunique()
    if int((distinct > 1).sum()):
        raise ValueError(
            f"{int((distinct > 1).sum())} (run, protein) cells carry several quantities"
        )
    cells = grouped.first().reset_index()
    cells.columns = ["run", "accession", "quantity"]
    return cells, empty


def quantity_matrix(cells: pd.DataFrame, runs: list[str]) -> pd.DataFrame:
    """Proteins (rows) x runs (columns, every run of the report) of rule-1 quantities."""
    matrix = cells.pivot(index="accession", columns="run", values="quantity")
    return matrix.reindex(columns=runs).sort_index()


def detection_rule(matrix: pd.DataFrame) -> pd.Series:
    """Rule 2: a protein is kept when quantified in more than half of the runs."""
    return matrix.notna().mean(axis=1) > DETECTION_FRACTION


def identification_rule(counts: pd.Series) -> tuple[float, list[str]]:
    """Rule 3: the threshold and the runs below 80% of the maximum identifications."""
    threshold = IDENTIFICATION_FRACTION * float(counts.max())
    return threshold, sorted(str(r) for r in counts.index[counts < threshold])


class DmsoReference(BaseModel):
    """Per-protein mean, standard error and n over the DMSO runs."""

    model_config = ConfigDict(extra="forbid")

    mean: dict[str, float]
    se: dict[str, float]
    n: dict[str, int]


def dmso_reference(matrix: pd.DataFrame) -> DmsoReference:
    """Mean, SD / sqrt(n) (NaN for n = 1) and n of each protein over the given runs."""
    counts = matrix.notna().sum(axis=1)
    keep = counts[counts >= 1].index
    mean = matrix.mean(axis=1)
    std = matrix.std(axis=1, ddof=1)
    return DmsoReference(
        mean={str(k): float(mean[k]) for k in keep},
        se={
            str(k): float(std[k]) / math.sqrt(int(counts[k]))
            if int(counts[k]) > 1
            else float("nan")
            for k in keep
        },
        n={str(k): int(counts[k]) for k in keep},
    )


class MissingDmsoReferenceError(ValueError):
    """A drug run quantified a protein that no DMSO run quantified."""


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/messnerUltrafastProteomicsScanning2021``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for artifact in manifest.files:
        if artifact.path == relpath:
            return artifact.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


def deposit_raw_mirror(
    *,
    report_path: str | Path,
    info_path: str | Path,
    retrieved_at: str = RAW_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Copy the two retrieved PRIDE files into the raw mirror and write ``manifest.json``.

    Run once by hand after ``curl -f -o <file> <url>`` for each URL. Idempotent by
    sha256: a present file with the pinned hash is left alone, a differing one raises,
    and a source file off its pin raises before anything is written.
    """
    root = raw_mirror_dir(data_root)
    files: list[ArtifactRecord] = []
    for src, rel, pin, url in (
        (report_path, REPORT_REL, REPORT_SHA256, REPORT_URL),
        (info_path, INFO_REL, INFO_SHA256, INFO_URL),
    ):
        got = _sha256(src)
        if got != pin:
            raise RuntimeError(f"{src} sha256 {got} is not the pin {pin}")
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != pin:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        files.append(
            ArtifactRecord(
                path=rel,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=pin,
                source=url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": url},
                    sha256=pin,
                    retrieved_at=retrieved_at,
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=files,
        si_data_sources=[
            f"https://www.ebi.ac.uk/pride/archive/projects/{PRIDE_ACCESSION}",
            REPORT_URL,
            INFO_URL,
        ],
        si_expected=[
            f"{REPORT_FILENAME}: deposited; the DIA-NN 1.7.12 main report of the yeast "
            "drug screen (102 runs), the only processed file of the screen",
            f"{INFO_FILENAME}: deposited; the authors' index of the DIA-NN outputs, "
            "read to confirm the report is the yeast drug screen",
            f"PRIDE {PRIDE_ACCESSION} raw spectra (95 drug-plate and 7 QC .wiff/.wiff."
            "scan pairs, about 3.5 GB per .wiff.scan): NOT deposited; no loader "
            "consumes raw spectra and the report is the processed output",
            "20200623_yeastlib_sswath.tsv.speclib and "
            "yeast-uniprot-proteome-3AUP000002311-canonical.fasta: NOT deposited; the "
            "spectral library and search database behind the report, not read by the "
            "loader",
            "PXD023613_community_annotated.sdrf.tsv: NOT deposited; a community "
            "annotation whose 'biological replicate' column is the run names' trailing "
            "number times 100, not a replicate index",
            "Supplementary Information (MOESM1) and Reporting Summary (MOESM2) are "
            "quoted from the torchcell-library OCR mirror, not the raw mirror",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def report_is_the_drug_screen(info_text: str) -> bool:
    """True when the authors' index lists the report under 'Yeast drug screen'."""
    if "Yeast drug screen" not in info_text:
        return False
    after = info_text.split("Yeast drug screen", 1)[1].split()
    return bool(after) and after[0] == REPORT_FILENAME


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class ProteomeMessner2021Dataset(ExperimentDataset):
    """Scanning SWATH proteomes of wild-type yeast under 18 compounds vs DMSO (70 runs)."""

    organism = Organism.scerevisiae

    #: Every record is the reference strain itself (the perturbation is the compound),
    #: so the gene set is legitimately empty.
    has_gene_perturbations = False

    def __init__(
        self,
        root: str = "data/torchcell/proteome_messner2021",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (protein keys are systematic ORFs; no genome)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProteinAbundanceExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The DIA-NN report and the authors' index of DIA-NN outputs."""
        return [REPORT_FILENAME, INFO_FILENAME]

    def download(self) -> None:
        """Link the two mirror files into ``raw/`` after verifying each against its pin.

        The mirror plus the module pin is canonical; the PRIDE URL is retrieval metadata
        that ``deposit_raw_mirror`` re-runs, never a live dependency. The manifest must
        record the pin, else ``ManifestPinMismatchError`` names both digests.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for rel, pin, name in (
            (REPORT_REL, REPORT_SHA256, REPORT_FILENAME),
            (INFO_REL, INFO_SHA256, INFO_FILENAME),
        ):
            check_manifest_pin(rel, manifest_sha256(manifest, rel), pin)
            src = raw_mirror_dir(data_root) / rel
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, name), pin)

    @post_process
    def process(self) -> None:
        """Apply the paper's three rules to the report and write one record per run."""
        verify_raw_files(
            self.raw_dir, {REPORT_FILENAME: REPORT_SHA256, INFO_FILENAME: INFO_SHA256}
        )
        info_text = Path(self.raw_dir, INFO_FILENAME).read_text(encoding="utf-8-sig")
        if not report_is_the_drug_screen(info_text):
            raise ValueError(f"{INFO_FILENAME} does not list {REPORT_FILENAME}")
        report = read_report(osp.join(self.raw_dir, REPORT_FILENAME))
        ledger, rows = self.build_rows(report, build_uniprot_to_orf_map())

        os.makedirs(self.preprocess_dir, exist_ok=True)
        Path(self.preprocess_dir, "build_ledger.json").write_text(
            ledger.model_dump_json(indent=2)
        )
        pd.DataFrame([r.model_dump() for r in ledger.runs]).to_csv(
            osp.join(self.preprocess_dir, "runs.csv"), index=False
        )
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e10))
        with env.begin(write=True) as txn:
            for idx, row in enumerate(tqdm(rows)):
                experiment, reference, publication = self.create_experiment(row)
                txn.put(
                    f"{idx}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": publication.model_dump(),
                        }
                    ),
                )
        env.close()
        log.info("Messner 2021: wrote %d proteome records", len(rows))

    @staticmethod
    def build_rows(
        report: pd.DataFrame, up2orf: dict[str, str]
    ) -> tuple[BuildLedger, list[dict[str, Any]]]:
        """The ledger and the per-record rows (``create_experiment`` input) of a report."""
        runs = sorted(report["run"].unique())
        parsed: dict[str, re.Match[str]] = {}
        qc: list[str] = []
        for run in runs:
            plate = PLATE_RUN_RE.match(run)
            if plate is not None:
                parsed[run] = plate
            elif QC_RUN_RE.match(run):
                qc.append(run)
            else:
                raise ValueError(f"run name {run!r} is neither a plate nor a QC run")
        by_injection = {int(m["injection"]): m["agent"] for m in parsed.values()}
        check_plate_layout(by_injection)
        for run, m in parsed.items():
            if int(m["well"]) - int(m["injection"]) != WELL_OFFSET:
                raise PlateLayoutError(f"{run}: well is not injection + {WELL_OFFSET}")
        injections = sorted(by_injection)
        absent = sorted(set(range(1, injections[-1] + 1)) - set(injections))

        cells, n_empty = identified_cells(report)
        matrix = quantity_matrix(cells, runs)
        ids_rule1 = matrix.notna().sum()
        keep = detection_rule(matrix)
        kept = matrix[keep]
        ids_kept = kept.notna().sum()
        threshold, outliers = identification_rule(ids_kept)

        # A kept accession the GFF carries no ORF for keeps its released accession as
        # its key: nothing is dropped on the genome's account.
        stored_as_released = {
            a: ACCESSIONS_WITHOUT_ORF.get(
                a, "no feature of the S288C R64 GFF names this accession"
            )
            for a in kept.index
            if a not in up2orf
        }
        served = list(kept.index)
        accession_to_orf = {a: up2orf.get(a, a) for a in served}
        orf_counts = Counter(accession_to_orf.values())
        shared = sorted(o for o, c in orf_counts.items() if c > 1)
        if shared:
            raise RuntimeError(f"several accessions map to one ORF: {shared[:5]}")
        by_orf = kept.loc[served].rename(index=accession_to_orf)

        dmso_runs = sorted(
            (
                r
                for r, m in parsed.items()
                if m["agent"] == DMSO_TOKEN and r not in outliers
            ),
            key=lambda r: int(parsed[r]["injection"]),
        )
        reference = dmso_reference(by_orf[dmso_runs])
        drug_runs = sorted(
            (
                r
                for r, m in parsed.items()
                if m["agent"] != DMSO_TOKEN and r not in outliers
            ),
            key=lambda r: int(parsed[r]["injection"]),
        )
        rows: list[dict[str, Any]] = []
        record_index: dict[str, int] = {}
        for run in drug_runs:
            series = by_orf[run].dropna()
            abundance = {str(o): float(v) for o, v in series.items()}
            missing = sorted(set(abundance) - set(reference.mean))
            if missing:
                raise MissingDmsoReferenceError(
                    f"{run} quantified {len(missing)} protein(s) no DMSO run did: "
                    f"{missing[:5]}"
                )
            record_index[run] = len(rows)
            rows.append(
                {
                    "run": run,
                    "agent": AGENTS[parsed[run]["agent"]],
                    "abundance": dict(sorted(abundance.items())),
                    "reference": reference,
                }
            )

        entries: list[RunEntry] = []
        for run in runs:
            match = parsed.get(run)
            token = None if match is None else match["agent"]
            if match is None:
                role: RunRole = "qc"
            elif run in outliers:
                role = "outlier"
            elif token == DMSO_TOKEN:
                role = "reference"
            else:
                role = "experiment"
            entries.append(
                RunEntry(
                    run=run,
                    kind="qc" if match is None else "plate",
                    injection=None if match is None else int(match["injection"]),
                    well=None if match is None else int(match["well"]),
                    block=(
                        None
                        if match is None
                        else (int(match["injection"]) - 1) // BLOCK_SIZE
                    ),
                    agent_token=token,
                    compound_label=(
                        AGENTS[token].label
                        if token is not None and token != DMSO_TOKEN
                        else ("DMSO" if token == DMSO_TOKEN else None)
                    ),
                    trailing_token=None if match is None else match["token"],
                    n_identified=int(ids_rule1[run]),
                    n_after_detection_rule=int(ids_kept[run]),
                    role=role,
                    record_index=record_index.get(run),
                )
            )
        n_cells = int(matrix.notna().sum().sum())
        ledger = BuildLedger(
            report_sha256=REPORT_SHA256,
            n_report_rows=len(report),
            n_runs=len(runs),
            n_plate_runs=len(parsed),
            n_qc_runs=len(qc),
            absent_injections=absent,
            n_proteins_identified=len(matrix.index),
            n_proteins_kept=int(keep.sum()),
            n_cells_rule1=n_cells,
            n_cells_dropped_by_detection_rule=n_cells - int(kept.notna().sum().sum()),
            n_cells_without_quantity=n_empty,
            max_identifications=int(ids_kept.max()),
            identification_threshold=threshold,
            mean_identifications_after_detection_rule=float(ids_kept.mean()),
            mean_identifications_rule1_only=float(ids_rule1.mean()),
            outliers=outliers,
            accessions_stored_as_released=stored_as_released,
            n_records=len(rows),
            n_reference_runs=len(dmso_runs),
            n_table_s4_records=sum(1 for r in rows if r["agent"].drug_class),
            accession_to_orf=accession_to_orf,
            runs=entries,
        )
        return ledger, rows

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    @staticmethod
    def environment(perturbation: SmallMoleculePerturbation) -> Environment:
        """The medium at 30 C with one small molecule on top of it."""
        return Environment(
            media=SM_UNSTATED_MESSNER2021,
            temperature=Temperature(value=TEMPERATURE_C),
            perturbations=[perturbation],
        )

    @staticmethod
    def treatment(spec: AgentSpec) -> SmallMoleculePerturbation:
        """The agent at 10 uM, delivered in the DMSO vehicle of the control."""
        return SmallMoleculePerturbation(
            compound=resolved_compound(spec.label),
            concentration=Concentration(
                value=DOSE_UM, unit=ConcentrationUnit.micromolar, basis=DoseBasis.fixed
            ),
            solvent=Solvent(
                name="DMSO", percent=None, compound=resolved_compound("DMSO")
            ),
        )

    @staticmethod
    def vehicle() -> SmallMoleculePerturbation:
        """The DMSO control: a fixed dose whose fraction the paper does not state."""
        return SmallMoleculePerturbation(
            compound=resolved_compound("DMSO"),
            concentration=Concentration(basis=DoseBasis.fixed),
        )

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[
        ProteinAbundanceExperiment, ProteinAbundanceExperimentReference, Publication
    ]:
        """One drug run against the DMSO profile restricted to the proteins it measured."""
        abundance: dict[str, float] = row["abundance"]
        ref: DmsoReference = row["reference"]
        experiment = ProteinAbundanceExperiment(
            dataset_name=self.name,
            genotype=Genotype(perturbations=[]),
            environment=self.environment(self.treatment(row["agent"])),
            phenotype=ProteinAbundancePhenotype(
                protein_abundance=abundance,
                protein_abundance_se=None,
                n_replicates={orf: 1 for orf in abundance},
                measurement_type=MEASUREMENT_TYPE,
            ),
        )
        reference = ProteinAbundanceExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4741"
            ),
            environment_reference=self.environment(self.vehicle()),
            phenotype_reference=ProteinAbundancePhenotype(
                protein_abundance={k: ref.mean[k] for k in abundance},
                protein_abundance_se={k: ref.se[k] for k in abundance},
                n_replicates={k: ref.n[k] for k in abundance},
                measurement_type=MEASUREMENT_TYPE,
            ),
        )
        publication = Publication(
            pubmed_id=PUBMED_ID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        return experiment, reference, publication


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def sourced_values() -> list[SourcedValue]:
    """Every quote the records rest on: the module's and its medium's."""
    media = SM_UNSTATED_MESSNER2021
    return [
        *SOURCED.values(),
        *media.provenance,
        *(sv for c in media.components for sv in c.provenance),
    ]


def read_store(dataset_root: str) -> list[dict[str, Any]]:
    """Every record of a built store in index order (keys ``"0"``..``"n-1"``)."""
    env = lmdb.open(
        osp.join(dataset_root, "processed", "lmdb"), readonly=True, lock=False
    )
    with env.begin() as txn:
        n = txn.stat()["entries"]
        records = []
        for i in range(n):
            stored = txn.get(f"{i}".encode())
            assert stored is not None, f"record {i} of {n} missing from {dataset_root}"
            records.append(pickle.loads(stored))
    env.close()
    return records


def _result(
    level: Level, name: str, passed: bool, message: str, **details: Any
) -> LevelResult:
    return LevelResult(
        level=level, name=name, passed=passed, message=message, details=details
    )


def check_values_match_report(
    records: list[dict[str, Any]], ledger: BuildLedger, report: pd.DataFrame
) -> list[LevelResult]:
    """L2: every stored quantity and every reference mean re-read from the report bytes.

    An independent path over the raw rows (a dict keyed by run and accession, no matrix),
    so a pivot or an index slip in the build cannot hide here.
    """
    rule1 = report[
        (report["Proteotypic"] == 1)
        & (report["Protein.Q.Value"] <= PROTEIN_Q_MAX)
        & report["Genes.MaxLFQ.Unique"].notna()
    ]
    raw = {
        (run, acc): float(v)
        for run, acc, v in zip(
            rule1["run"], rule1["Protein.Group"], rule1["Genes.MaxLFQ.Unique"]
        )
    }
    orf_to_acc = {orf: acc for acc, orf in ledger.accession_to_orf.items()}
    by_index = {e.record_index: e for e in ledger.runs if e.record_index is not None}
    dmso = [e.run for e in ledger.runs if e.role == "reference"]
    bad_values = 0
    bad_reference = 0
    n_values = 0
    for index, record in enumerate(records):
        run = by_index[index].run
        values = record["experiment"]["phenotype"]["protein_abundance"]
        ref = record["reference"]["phenotype_reference"]["protein_abundance"]
        for orf, value in values.items():
            n_values += 1
            acc = orf_to_acc[orf]
            if raw.get((run, acc)) != value:
                bad_values += 1
            dmso_values = [raw[(r, acc)] for r in dmso if (r, acc) in raw]
            expected = float(np.mean(dmso_values))
            if not math.isclose(ref[orf], expected, rel_tol=1e-12):
                bad_reference += 1
    return [
        _result(
            Level.L2,
            "values_match_report",
            bad_values == 0,
            f"{n_values - bad_values}/{n_values} stored quantities equal the report's",
            n_values=n_values,
            n_bad=bad_values,
        ),
        _result(
            Level.L2,
            "reference_matches_report",
            bad_reference == 0,
            f"{n_values - bad_reference}/{n_values} reference means re-derived from "
            f"the {len(dmso)} DMSO runs",
            n_bad=bad_reference,
            n_dmso_runs=len(dmso),
        ),
    ]


def check_design(
    records: list[dict[str, Any]], ledger: BuildLedger
) -> list[LevelResult]:
    """L1: wild-type genotypes; one record per kept drug run; the measured plate."""
    genotypes = sum(1 for r in records if r["experiment"]["genotype"]["perturbations"])
    per_compound = Counter(
        r["experiment"]["environment"]["perturbations"][0]["compound"]["name"]
        for r in records
    )
    expected_per_label = Counter(
        e.compound_label
        for e in ledger.runs
        if e.role == "experiment" and e.compound_label is not None
    )
    expected = Counter(
        {resolved_compound(label).name: n for label, n in expected_per_label.items()}
    )
    return [
        _result(
            Level.L1,
            "wild_type_genotype",
            genotypes == 0,
            f"{len(records) - genotypes}/{len(records)} records carry no perturbation",
        ),
        _result(
            Level.L1,
            "runs_per_compound",
            per_compound == expected,
            f"{len(per_compound)} compounds, runs {dict(sorted(per_compound.items()))}",
            observed=dict(per_compound),
            expected=dict(expected),
        ),
        _result(
            Level.L1,
            "release_layout",
            (
                ledger.n_runs,
                ledger.n_plate_runs,
                ledger.n_qc_runs,
                ledger.absent_injections,
                ledger.outliers,
                ledger.n_reference_runs,
            )
            == (
                EXPECTED_RUNS,
                EXPECTED_PLATE_RUNS,
                EXPECTED_QC_RUNS,
                EXPECTED_ABSENT_INJECTIONS,
                EXPECTED_OUTLIERS,
                EXPECTED_REFERENCE_RUNS,
            ),
            f"{ledger.n_runs} runs ({ledger.n_plate_runs} plate, {ledger.n_qc_runs} "
            f"QC), absent injections {ledger.absent_injections}, outliers "
            f"{ledger.outliers}, {ledger.n_reference_runs} DMSO reference runs",
        ),
    ]


def check_paper_counts(
    records: list[dict[str, Any]], ledger: BuildLedger
) -> list[LevelResult]:
    """L4: the paper's own numbers, reproduced from the rebuilt proteome."""
    mean_ids = ledger.mean_identifications_after_detection_rule
    table_s4 = {resolved_compound(label).name for label, _ in TABLE_S4}
    n_s4 = sum(
        1
        for r in records
        if r["experiment"]["environment"]["perturbations"][0]["compound"]["name"]
        in table_s4
    )
    return [
        _result(
            Level.L4,
            "paper_mean_identifications",
            round(mean_ids, -1) == PAPER_MEAN_IDENTIFICATIONS,
            f"mean {mean_ids:.1f} proteins per run after rules 1-2 over "
            f"{ledger.n_runs} injections rounds to the paper's "
            f"{PAPER_MEAN_IDENTIFICATIONS} (rule 1 alone: "
            f"{ledger.mean_identifications_rule1_only:.1f})",
            mean=mean_ids,
        ),
        _result(
            Level.L4,
            "paper_biological_sample_count",
            n_s4 == EXPECTED_TABLE_S4_RECORDS,
            f"{n_s4} records of the 16 Table S4 drugs; Supplementary Fig. 4b states "
            f"n={EXPECTED_TABLE_S4_RECORDS} biological samples",
            n=n_s4,
        ),
    ]


def check_sgd_containment(
    records: list[dict[str, Any]], sgd_genes: set[str]
) -> LevelResult:
    """L4: every protein key is an S288C systematic name."""
    keys = {
        k for r in records for k in r["experiment"]["phenotype"]["protein_abundance"]
    }
    inside = keys & sgd_genes
    fraction = len(inside) / len(keys) if keys else 0.0
    return _result(
        Level.L4,
        "gene_containment_sgd",
        fraction >= MIN_SGD_CONTAINMENT,
        f"{fraction:.4f} of {len(keys)} protein keys are S288C systematic names "
        f"(>= {MIN_SGD_CONTAINMENT}); outside: {sorted(keys - sgd_genes)[:10]}",
        n_keys=len(keys),
        outside=sorted(keys - sgd_genes),
    )


def check_messner2023_reference(
    records: list[dict[str, Any]], messner2023_reference: dict[str, float]
) -> LevelResult:
    """L4: the DMSO profile agrees with the same lab's 388-run wild-type profile.

    Pearson r of log2 abundance over shared ORFs, against Messner 2023's HIS3 reference
    (a different instrument run, strain construction and quantification batch), so the
    bar is the ordering of protein amounts, r >= 0.6, set before it was measured.
    """
    ours: dict[str, float] = {}
    for record in records:
        ours.update(record["reference"]["phenotype_reference"]["protein_abundance"])
    shared = sorted(set(ours) & set(messner2023_reference))
    a = np.log2([ours[k] for k in shared])
    b = np.log2([messner2023_reference[k] for k in shared])
    r = float(np.corrcoef(a, b)[0, 1]) if len(shared) > 2 else float("nan")
    return _result(
        Level.L4,
        "messner2023_reference_agreement",
        r >= MIN_MESSNER2023_REFERENCE_PEARSON,
        f"Pearson r {r:.3f} of log2 DMSO vs Messner 2023 WT reference over "
        f"{len(shared)} shared ORFs (>= {MIN_MESSNER2023_REFERENCE_PEARSON})",
        r=r,
        n_shared=len(shared),
    )


def messner2023_reference_profile(
    data_root: str, uniprot_to_orf: dict[str, str] | None = None
) -> dict[str, float]:
    """The union of Messner 2023's per-record WT reference abundances (one profile),
    keyed by systematic ORF.

    Messner 2023 stores its protein keys as the released UniProt accessions (#1045) while
    this store keys by the ORF the DIA-NN report names, so the join lifts each accession
    through the SGD GFF cross-references (:func:`build_uniprot_to_orf_map`, a
    verification-time utility, never a record field); an accession the GFF does not
    carry keeps its released key and simply never matches.
    """
    from torchcell.verification.runners import stream_records

    mapping = (
        build_uniprot_to_orf_map(data_root)
        if uniprot_to_orf is None
        else uniprot_to_orf
    )
    profile: dict[str, float] = {}
    for record in stream_records(osp.join(data_root, MESSNER2023_ROOT)):
        abundances = record["reference"]["phenotype_reference"]["protein_abundance"]
        profile.update(
            {mapping.get(key, key): value for key, value in abundances.items()}
        )
    return profile


def check_sourced(library_root: str) -> LevelResult:
    """L3: every quote is still a verbatim substring of its sha256-pinned mirror file."""
    from torchcell.verification.sourced import audit_sourced_value

    results = [audit_sourced_value(sv, library_root) for sv in sourced_values()]
    failed = [r.details for r in results if not r.passed]
    return _result(
        Level.L3,
        "sourced_quotes_verbatim",
        not failed,
        f"{len(results) - len(failed)}/{len(results)} quotes verbatim in the mirror",
        failed=failed,
    )


def verify_build(
    dataset_root: str, *, data_root: str | None = None
) -> VerificationReport:
    """L0-L4 report for a built ``proteome_messner2021`` store."""
    from torchcell.verification.protein import verify_protein_dataset
    from torchcell.verification.runners import _sgd_gene_set

    root = data_root or _data_root()
    records = read_store(dataset_root)
    ledger = BuildLedger.model_validate_json(
        Path(dataset_root, "preprocess", "build_ledger.json").read_text()
    )
    report = verify_protein_dataset(
        records,
        dataset_name="proteome_messner2021",
        provenance=Provenance(
            source_uri=REPORT_URL,
            citation_key=CITATION_KEY,
            sha256=REPORT_SHA256,
            method=(
                "DIA-NN 1.7.12 Genes.MaxLFQ.Unique; proteotypic, protein q <= 0.01, "
                "detected in > 50% of runs, runs >= 80% of the maximum identifications; "
                "one acquisition per drug run vs the mean of 24 DMSO runs"
            ),
            page=f"PRIDE {PRIDE_ACCESSION} {REPORT_FILENAME}",
            retrieved=RAW_RETRIEVED_AT,
        ),
        expected_count=EXPECTED_RECORDS,
    )
    for result in check_design(records, ledger):
        report.add(result)
    raw = read_report(osp.join(dataset_root, "raw", REPORT_FILENAME))
    for result in check_values_match_report(records, ledger, raw):
        report.add(result)
    report.add(check_sourced(osp.join(root, "torchcell-library")))
    for result in check_paper_counts(records, ledger):
        report.add(result)
    report.add(check_sgd_containment(records, _sgd_gene_set(root)))
    report.add(
        check_messner2023_reference(records, messner2023_reference_profile(root))
    )
    return report


def main() -> None:
    """Deposit the raw mirror (``--deposit-from``) or build/load the store."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--deposit-from", help="directory holding the two retrieved PRIDE files"
    )
    args = parser.parse_args()
    if args.deposit_from:
        print(
            deposit_raw_mirror(
                report_path=osp.join(args.deposit_from, REPORT_FILENAME),
                info_path=osp.join(args.deposit_from, INFO_FILENAME),
            )
        )
        return
    dataset = ProteomeMessner2021Dataset(
        root=osp.join(_data_root(), "data/torchcell/proteome_messner2021")
    )
    print(f"proteome_messner2021: {len(dataset)} records")


if __name__ == "__main__":
    main()
