# torchcell/datasets/scerevisiae/peter2018
# [[torchcell.datasets.scerevisiae.peter2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/peter2018
# Test file: tests/torchcell/datasets/scerevisiae/test_peter2018.py
"""Peter 2018: growth of 971 natural S. cerevisiae isolates on 35 conditions.

Peter et al. 2018 (Nature 556:339, doi:10.1038/s41586-018-0030-5; citation key
``peterGenomeEvolution10112018``) sequenced 1,011 isolates and phenotyped 971 of them
by end-point colony growth on solid medium. The genome half is already the genomes-tier
set ``peter2018_1011_assemblies`` (Caudal 2024 and Bloom 2019 consume it); this module
serves the phenotype half.

RELEASE. ``phenoMatrix_35ConditionsNormalizedByYPD.tab.gz`` from the 1002 Yeast Genome
host, "growth ratio between 35 stress conditions and standard YPD medium at 30 C, for
971 isolates" (Data availability). 971 rows x 35 columns, no missing cell, every value
non-negative, 197 exact zeros. The YPD 30 C control is the denominator of every ratio
and is not released as a value, which is why the paper's "34,956 phenotypic
measurements that covered 36 traits" (971 x 36) is 33,985 released numbers (971 x 35).
The release headers are abbreviations (``YPDCAFEIN40``); each is mapped to its
Supplementary Table 2 row by content (compound and dose, or carbon source, or
temperature), and the verifier re-reads Table 2 from the mirrored workbook to check
every mapped row.

RECORD. One ``EnvironmentResponseExperiment`` per (isolate, condition): 33,985 records.
The phenotype is an ``EnvironmentResponsePhenotype`` with
``MeasurementType.condition_growth_ratio``, the isolate's end-point colony size on the
condition plate divided by its own colony size on YPD 30 C, so 1.0 is "grows as on the
control" and the reference states 1.0 (``RATIO_MEASUREMENT_TYPES``). Not
``FitnessPhenotype``: that class is a mutant-over-wild-type ratio and clamps a
non-positive value, while 197 released values are exactly 0. ``n_samples = 4`` colonies
is the conservative reading of "Each isolate was present in quadruplicates on the
corresponding matrix (interplate replicates) and at two different positions (intraplate
replicates)", which reads as either 4 or 4 x 2 = 8 colonies; the lower count is taken
(CLAUDE.md, resolving a range), and the aggregation statistic and any dispersion are not
released (typed gap on ``environment_response_se``).

GENOTYPE. Built by ``torchcell.sequence.natural_isolate`` (``NaturalIsolateGenotypes``),
the module every natural-isolate loader on this panel shares, so an isolate here is the
same ``Genotype`` as in the Caudal 2024, Muenzner 2024 and De Chiara 2022 stores:
``SequenceVariantPerturbation`` for every reference gene whose isolate allele differs from
the S288C R64 slice, ``NaturalGenePresencePerturbation`` for every non-reference pangenome
ORF the isolate carries, ``NaturalGeneAbsencePerturbation`` for every reference ORF it
lacks, ``DeletionPerturbation`` of HO on the 122 isolates Supplementary Table 1 marks
HO-deleted (#988), ``NaturalLossOfFunctionPerturbation`` per call of the two
loss-of-function matrices (#987), and ``Genotype.natural_isolate`` (Table S1 ploidy,
zygosity, aneuploidy, manipulation). One genotype per isolate, interned once in the LMDB
and shared by its 35 records. All 971 phenotyped isolates are in the 1,011-genome panel.

ENVIRONMENT. Media from Supplementary Table 2 (``si/si3.xls`` sheet ``Table S2``): YPD is
the library ``YPD_AGAR`` restated with Table 2's recipe (2% peptone, 1% yeast extract,
2% glucose, 2% agar, the same composition); the seven YP + 2% carbon-source plates are
loader-local media deriving from ``YP``. Chemical stresses are
``SmallMoleculePerturbation``s on YPD at Table 2's dose, with a typed gap on ``solvent``
(Table 2 states no vehicle). A bare ``%`` dose is ``ConcentrationUnit.percent`` (basis
not stated). Temperature lives on ``Environment.temperature`` (14, 40, 42 C for the
three temperature plates, 30 C otherwise: "Growth temperature is 30°C if not specified").
``duration_hours = 40`` from the Methods' one incubation sentence, which names 30 C; for
the three temperature plates that sentence's temperature is overridden by Table 2 and
whether the 40 h also held is not stated, so their duration is a typed gap.

RAW MIRROR. ``$DATA_ROOT/torchcell-raw/peterGenomeEvolution10112018/`` holds
``data/phenoMatrix_35ConditionsNormalizedByYPD.tab.gz`` and the host's own checksum file
``data/md5.txt`` (retrieved 2026-10-11, ``direct_url``); the release's md5 equals the
host's published md5. The two loss-of-function matrices the 2026-07-13 retrieval left
under ``$DATA_ROOT/data/peter2018/`` are registered into the genomes tier set
(``register_loss_of_function_matrices``), and so is the Supplementary Tables workbook
(``natural_isolate.register_supplementary_tables``); the genotype reads all three.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    compute_sha256_hash,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.data.experiment_dataset import canonical_json
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YP, YPD_AGAR, restated
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    DeletionPerturbation,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    NaturalGeneAbsencePerturbation,
    NaturalGenePresencePerturbation,
    NaturalIsolateDescriptor,
    NaturalLossOfFunctionPerturbation,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SequenceVariantPerturbation,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.registry import (
    PETER2018_1011,
    ROLE_MATRIX,
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
    assembly_set_dir,
    extend_assembly_set,
    load_genome_manifest,
)
from torchcell.sequence.natural_isolate import (
    FRAMESHIFT_NAME,
    GENOTYPE_FILES,
    LOF_NAME,
    NaturalIsolateGenotypes,
    link_genotype_files,
)
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
CITATION_KEY = "peterGenomeEvolution10112018"
PAPER_DOI = "10.1038/s41586-018-0030-5"
PAPER_TITLE = "Genome evolution across 1,011 Saccharomyces cerevisiae isolates"
DATA_HOST = "http://1002genomes.u-strasbg.fr/files/"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

PHENO_NAME = "phenoMatrix_35ConditionsNormalizedByYPD.tab.gz"
MD5_NAME = "md5.txt"
PHENO_REL = f"data/{PHENO_NAME}"
MD5_REL = f"data/{MD5_NAME}"
#: The host's own md5 of the release, as ``md5.txt`` lists it.
PHENO_PUBLISHED_MD5 = "479862335552a050e46de163d1ee5f72"
#: The release's sha256: the build-time pin (the raw-mirror manifest must record it).
PHENO_SHA256 = "7696369a6ad4ea98ed99708b2df59a2916ae34846f909648f241c1194c7347d9"

#: Library-mirror artifacts every quote below is read from (sha256 of the bytes).
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "21c49934d546f162b43025c85254bcd1cedcb80ab9a691935301b5962a6a12da"
SI_TABLES_XLS = "si/si3.xls"
SI_TABLES_XLS_SHA256 = (
    "b11de13b50bcf91bb2a40cdbd0f2f35372bdef6fc9a018d7538cf5e5eea7f273"
)
TABLE_S2_SHEET = "Table S2"
REPORTING_SUMMARY_MD = "si/si2.md"
REPORTING_SUMMARY_MD_SHA256 = (
    "ed8300228c7b1d2ec3ec9ac7f1d2bc5362f92e136f2435655eb8cdb4ff2fda46"
)

N_ISOLATES = 971
N_CONDITIONS = 35
EXPECTED_RECORDS = N_ISOLATES * N_CONDITIONS
#: The paper's measurement count: 971 isolates x 36 plates, the YPD 30 C control
#: included. The release carries the 35 ratios; the control is their denominator.
PAPER_MEASUREMENTS = 34956

#: What the isolate genotypes of this build carry, counted (``natural_isolate``).
GENOTYPE_LEDGER = "natural_isolate_genotype_ledger.json"
#: The classes an isolate genotype holds, built in ``torchcell.sequence.natural_isolate``.
#: Named here because the schema-impact closure of this loader is read from this file's
#: own imports: a change to any of them must re-stale this store.
GENOTYPE_CLASSES = (
    SequenceVariantPerturbation,
    NaturalGenePresencePerturbation,
    NaturalGeneAbsencePerturbation,
    DeletionPerturbation,
    NaturalLossOfFunctionPerturbation,
    NaturalIsolateDescriptor,
)

N_SAMPLES = 4
INCUBATION_HOURS = 40.0
CONTROL_TEMPERATURE_C = 30.0
#: The loss-of-function matrices of the 2026-07-13 retrieval (issue #66), legacy path.
LEGACY_PETER_DIR_REL = "data/peter2018"
LOF_PUBLISHED_MD5 = {
    LOF_NAME: "f3c4f2db7a5b2a9bf28039dbdd82e2b1",
    FRAMESHIFT_NAME: "6f8ec446283817be303e20f5661550b0",
}


def _sv(
    value: object,
    quote: str,
    *,
    uri: str = PAPER_MD,
    sha: str = PAPER_MD_SHA256,
    page: str | None = None,
    note: str | None = None,
) -> SourcedValue:
    """A SourcedValue pinned to a mirrored Peter 2018 artifact (quote + sha256)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=uri, citation_key=CITATION_KEY, sha256=sha, page=page
        ),
        quote=quote,
        note=note,
    )


def _table_s2_sv(
    value: object, quote: str, row: int, note: str | None = None
) -> SourcedValue:
    """A SourcedValue on one cell of Supplementary Table 2 (sheet row is 1-based)."""
    return _sv(
        value,
        quote,
        uri=SI_TABLES_XLS,
        sha=SI_TABLES_XLS_SHA256,
        page=f"sheet '{TABLE_S2_SHEET}' row {row}",
        note=note,
    )


# paper.md line 250 (Methods, "Phenotyping") and line 292 (Data availability).
ASSAY_QUOTE = (
    "Quantitative high-throughput phenotyping was performed using end-point colony "
    "growth on solid medium."
)
PRECULTURE_QUOTE = (
    "Strains were pregrown in flat-bottom 96-well microplates containing liquid YPD "
    "medium."
)
REPLICATE_QUOTE = (
    "Each isolate was present in quadruplicates on the corresponding matrix "
    "(interplate replicates) and at two different positions (intraplate replicates)."
)
INCUBATION_QUOTE = (
    "The plates were incubated at $3 0 ^ { \\circ } \\mathrm { C }$ for "
    "$4 0 \\mathrm { { h } }$ and were scanned at a resolution of 600 dpi and 16-bit "
    "grayscale."
)
COLONY_SIZE_QUOTE = (
    "Quantification of the colony size from plate images was performed using the "
    "software package Gitter72."
)
NORMALIZATION_QUOTE = (
    "Each value was normalized using the growth ratio between stress media and "
    "standard YPD medium at $3 0 ^ { \\circ } \\mathrm { C }$ (see ‘Data availability’)."
)
RELEASE_QUOTE = (
    "-phenoMatrix_35ConditionsNormalizedByYPD.tab.gz: growth ratio between 35 stress "
    "conditions and standard YPD medium at $3 0 ^ { \\circ } \\mathrm { C } ,$ for 971 "
    "isolates."
)
MEASUREMENT_COUNT_QUOTE = (
    "In total, we analysed 34,956 phenotypic measurements that covered 36 traits"
)
REPLICATION_QUOTE = "Experimental replication was not attempted in our study."
TABLE_S2_TEMPERATURE_QUOTE = "Growth temperature is 30°C if not specified"

PHENOTYPE_SOURCED_VALUES: dict[str, SourcedValue] = {
    "assay_type": _sv(
        AssayType.colony_size_array.value, ASSAY_QUOTE, page="paper.md line 250"
    ),
    "measurement_type": _sv(
        MeasurementType.condition_growth_ratio.value,
        RELEASE_QUOTE,
        page="paper.md line 292",
        note="the released number is the isolate's colony size on a condition plate "
        "over its colony size on YPD 30 C; NORMALIZATION_QUOTE (line 250) says the "
        "same in the Methods",
    ),
    "colony_size": _sv(
        "Gitter colony size", COLONY_SIZE_QUOTE, page="paper.md line 250"
    ),
    "n_samples": _sv(
        N_SAMPLES,
        REPLICATE_QUOTE,
        page="paper.md line 250",
        note="reads as 4 colonies (quadruplicates spanning plates and positions) or 4 x "
        "2 = 8 (four plates, two positions on each); the release gives neither the "
        "per-colony values nor the aggregation statistic, so nothing back-solves it and "
        "the conservative lower count is taken. The Reporting Summary's 'Experimental "
        "replication was not attempted in our study.' (si/si2.md line 30) concerns "
        "repeating the experiment, not the colonies within it",
    ),
    "duration_hours": _sv(INCUBATION_HOURS, INCUBATION_QUOTE, page="paper.md line 250"),
}

# --------------------------------------------------------------------------- #
# Media (Supplementary Table 2, sheet rows 5-15; quotes are the cell strings)
# --------------------------------------------------------------------------- #
_PCT = ConcentrationUnit.percent_w_v
_PERCENT_BASIS_NOTE = (
    "Table 2 writes '%' with no w/v or v/v basis; recorded as w/v, the media "
    "library's basis for every bare percentage"
)
YPD_ROW = 5
YPD_CELL = "2% bactopeptone; 1% yeast extract; 2% glucose; 2% agar "

PETER_YPD: Media = restated(
    YPD_AGAR,
    _table_s2_sv(
        "2% peptone, 1% yeast extract, 2% glucose, 2% agar",
        YPD_CELL,
        YPD_ROW,
        note="the same four rows and amounts as YPD_AGAR; 'bactopeptone' is the "
        "library's peptone",
    ),
)
"""Solid YPD, the control plate and the base of every chemical-stress plate."""


def _peter_yp_plus(carbon: str, cell: str, row: int) -> Media:
    """YP + 2% ``carbon`` + 2% agar, one Table 2 carbon-source plate."""
    statement = _table_s2_sv(f"2% {carbon}", cell, row, note=_PERCENT_BASIS_NOTE)
    return Media(
        name=f"YP + 2% {carbon}, 2% agar (Peter 2018 Table S2)",
        state="solid",
        is_synthetic=False,
        base_medium="YP",
        components=[
            *YP.components,
            MediaComponent(
                compound=resolved_compound(carbon),
                role=MediaComponentRole.carbon_source,
                concentration=Concentration(value=2.0, unit=_PCT),
                provenance=[statement],
            ),
            MediaComponent(
                compound=resolved_compound("agar"),
                role=MediaComponentRole.gelling_agent,
                concentration=Concentration(value=2.0, unit=_PCT),
                provenance=[
                    _table_s2_sv("2% agar", cell, row, note=_PERCENT_BASIS_NOTE)
                ],
            ),
        ],
        provenance=[statement],
    )


# --------------------------------------------------------------------------- #
# Conditions: release column -> Supplementary Table 2 row
# --------------------------------------------------------------------------- #
class ConditionSpec(BaseModel):
    """One released column, its Table 2 row (verbatim cells) and typed environment."""

    model_config = ConfigDict(extra="forbid")

    code: str
    table_s2_row: int
    table_s2_condition: str
    table_s2_media: str
    media: Media
    temperature_c: float
    perturbations: list[SmallMoleculePerturbation]

    @property
    def temperature_shifted(self) -> bool:
        """True for the three temperature plates (14, 40, 42 C)."""
        return self.temperature_c != CONTROL_TEMPERATURE_C


_SOLVENT_GAP_NOTE = (
    "Supplementary Table 2 states the final concentration only; no stock solvent or "
    "vehicle is given in the table or the Methods"
)


def _stress(
    label: str,
    value: float,
    unit: ConcentrationUnit,
    *,
    compound_name: str | None = None,
) -> SmallMoleculePerturbation:
    """A Table 2 chemical stress on YPD. ``compound_name`` is the table-resolvable name
    when Table 2's own spelling is not one (``6-azauracile``, ``methylviologen``,
    ``sodium metaarsenite``); the spelling itself stays in the verbatim cell.
    """
    return SmallMoleculePerturbation(
        compound=resolved_compound(compound_name or label),
        concentration=Concentration(value=value, unit=unit),
        provenance_gaps=[
            ProvenanceGap(
                field="solvent",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=_SOLVENT_GAP_NOTE,
            )
        ],
    )


_MM = ConcentrationUnit.millimolar
_M = ConcentrationUnit.molar
_UGML = ConcentrationUnit.ug_per_ml
_GL = ConcentrationUnit.g_per_l
_BARE_PCT = ConcentrationUnit.percent


def build_conditions() -> dict[str, ConditionSpec]:
    """The 35 released columns, keyed by the release header, in Table 2's order.

    Each entry pins the Table 2 row by its 1-based sheet row and both verbatim cells.
    The header-to-row mapping is made by content: the compound and dose, the carbon
    source, or the temperature the header abbreviates. Doses are typed as Table 2 writes
    them; ``hydroxyurea 30mg/ml`` is 30 g/L (mg/mL and g/L are the same unit).
    """

    def spec(
        code: str,
        row: int,
        condition: str,
        media_cell: str,
        *,
        media: Media = PETER_YPD,
        temperature_c: float = CONTROL_TEMPERATURE_C,
        perturbations: list[SmallMoleculePerturbation] | None = None,
    ) -> ConditionSpec:
        return ConditionSpec(
            code=code,
            table_s2_row=row,
            table_s2_condition=condition,
            table_s2_media=media_cell,
            media=media,
            temperature_c=temperature_c,
            perturbations=perturbations or [],
        )

    yp = "2% bactopeptone; 1% yeast extract; 2% {}; 2% agar"
    specs = [
        spec("YPD14", 6, "YPD, 14°C", YPD_CELL, temperature_c=14.0),
        spec("YPD40", 7, "YPD, 40°C", YPD_CELL, temperature_c=40.0),
        spec("YPD42", 8, "YPD, 42°C", YPD_CELL, temperature_c=42.0),
        spec("YPACETATE", 9, "YP acetate 2%", yp.format("acetate"), media=_peter_yp_plus("acetate", yp.format("acetate"), 9)),
        spec("YPETHANOL", 10, "YP EtOH 2%", yp.format("ethanol"), media=_peter_yp_plus("ethanol", yp.format("ethanol"), 10)),
        spec("YPGLYCEROL", 11, "YP glycerol 2%", yp.format("glycerol"), media=_peter_yp_plus("glycerol", yp.format("glycerol"), 11)),
        spec("YPSORBITOL", 12, "YP sorbitol 2%", yp.format("sorbitol"), media=_peter_yp_plus("sorbitol", yp.format("sorbitol"), 12)),
        spec("YPGALACTOSE", 13, "YP galactose 2%", yp.format("galactose"), media=_peter_yp_plus("galactose", yp.format("galactose"), 13)),
        spec("YPRIBOSE", 14, "YP ribose 2%", yp.format("ribose"), media=_peter_yp_plus("ribose", yp.format("ribose"), 14)),
        spec("YPXYLOSE", 15, "YP xylose 2%", yp.format("xylose"), media=_peter_yp_plus("xylose", yp.format("xylose"), 15)),
        spec("YPDFORMAMIDE4", 16, "YPD formamide 4%", "YPD; formamide 4%", perturbations=[_stress("formamide", 4.0, _BARE_PCT)]),
        spec("YPDFORMAMIDE5", 17, "YPD formamide 5%", "YPD; formamide 5%", perturbations=[_stress("formamide", 5.0, _BARE_PCT)]),
        spec("YPDETOH", 18, "YPD EtOH 15%", "YPD; ethanol 15%", perturbations=[_stress("ethanol", 15.0, _BARE_PCT)]),
        spec("YPDBENOMYL200", 19, "YPD benomyl 200 µg/ml", "YPD; benomyl 200µg/ml", perturbations=[_stress("benomyl", 200.0, _UGML)]),
        spec("YPDBENOMYL500", 20, "YPD benomyl 500 µg/ml", "YPD; benomyl 500µg/ml", perturbations=[_stress("benomyl", 500.0, _UGML)]),
        spec("YPDSDS", 21, "YPD SDS 0.2%", "YPD; SDS 0.2%", perturbations=[_stress("SDS", 0.2, _BARE_PCT, compound_name="sodium dodecyl sulfate")]),
        spec("YPDDMSO", 22, "YPD DMSO 6%", "YPD; DMSO 6%", perturbations=[_stress("DMSO", 6.0, _BARE_PCT)]),
        spec("YPDKCL2M", 23, "YPD KCl 2 M", "YPD; KCl 2M", perturbations=[_stress("KCl", 2.0, _M, compound_name="potassium chloride")]),
        spec("YPDNACL1M", 24, "YPD NaCl 1 M", "YPD; NaCl 1M", perturbations=[_stress("NaCl", 1.0, _M)]),
        spec("YPDNACL15M", 25, "YPD NaCl 1.5 M", "YPD; NaCl 1.5M", perturbations=[_stress("NaCl", 1.5, _M)]),
        spec("YPDCUSO410MM", 26, "YPD CuSO4 10 mM", "YPD; CuSO4 10mM", perturbations=[_stress("CuSO4", 10.0, _MM)]),
        spec("YPDLICL250MM", 27, "YPD LiCl 250 mM", "YPD; LiCl 250mM", perturbations=[_stress("LiCl", 250.0, _MM)]),
        spec("YPDCHX05", 28, "YPD CHX 0.5 µg/ml", "YPD; cycloheximide 0.5µg/ml", perturbations=[_stress("cycloheximide", 0.5, _UGML)]),
        spec("YPDCHX1", 29, "YPD CHX 1 µg/ml", "YPD; cycloheximide 1µg/ml ", perturbations=[_stress("cycloheximide", 1.0, _UGML)]),
        spec("YPDANISO10", 30, "YPD anisomycin 10 µg/ml", "YPD; anisomycin 10µg/ml", perturbations=[_stress("anisomycin", 10.0, _UGML)]),
        spec("YPDANISO20", 31, "YPD anisomycin 20 µg/ml", "YPD; anisomycin 20µg/ml", perturbations=[_stress("anisomycin", 20.0, _UGML)]),
        spec("YPDANISO50", 32, "YPD anisomycin 50 µg/ml", "YPD; anisomycin 50µg/ml", perturbations=[_stress("anisomycin", 50.0, _UGML)]),
        spec("YPDCAFEIN40", 33, "YPD caffeine 40 mM", "YPD; caffeine 40mM", perturbations=[_stress("caffeine", 40.0, _MM)]),
        spec("YPDCAFEIN50", 34, "YPD caffeine 50 mM", "YPD; caffeine 50mM", perturbations=[_stress("caffeine", 50.0, _MM)]),
        spec("YPD6AU", 35, "YPD 6AU 600 µg/ml", "YPD; 6-azauracile 600µg/ml", perturbations=[_stress("6-azauracile", 600.0, _UGML, compound_name="6-azauracil")]),
        spec("YPDNYSTATIN", 36, "YPD nystatin 10 µg/ml", "YPD; nystatin 10µg/ml ", perturbations=[_stress("nystatin", 10.0, _UGML)]),
        spec("YPDMV", 37, "YPD Mv 20 mM", "YPD; methylviologen 20mM", perturbations=[_stress("methylviologen", 20.0, _MM, compound_name="paraquat")]),
        spec("YPDFLUCONAZOLE", 38, "YPD fluconazole 20 µg/ml", "YPD; fluconazole 20µg/ml", perturbations=[_stress("fluconazole", 20.0, _UGML)]),
        spec("YPDHU", 39, "YPD hydroxyurea 30 mg/ml", "YPD; hydroxyurea 30mg/ml", perturbations=[_stress("hydroxyurea", 30.0, _GL)]),
        spec("YPDSODIUMMETAARSENITE", 40, "YPD sodium metaarsenite 2.5 mM", "YPD; sodium metaarsenite 2.5mM", perturbations=[_stress("sodium metaarsenite", 2.5, _MM, compound_name="sodium arsenite")]),
    ]  # fmt: skip
    conditions = {s.code: s for s in specs}
    if len(conditions) != N_CONDITIONS:
        raise ValueError(f"expected {N_CONDITIONS} conditions, built {len(conditions)}")
    return conditions


#: The control plate's Table 2 row (sheet row 5), the denominator of every ratio.
CONTROL_CONDITION = "YPD"


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/peterGenomeEvolution10112018``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def md5_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex md5 of a file (the host publishes md5, not sha256)."""
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


_MD5_LINE = re.compile(r"^MD5 \((?P<name>[^)]+)\) = (?P<md5>[0-9a-f]{32})$")


def read_published_md5(path: str | Path) -> dict[str, str]:
    """``{file name: md5}`` from the host's ``md5.txt`` (BSD ``md5`` output format).

    Every non-blank line must parse; a line that does not is refused rather than
    skipped, so a changed format cannot silently drop a checksum.
    """
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        match = _MD5_LINE.match(line.strip())
        if match is None:
            raise ValueError(f"{path}: unparseable md5 line {line!r}")
        out[match.group("name")] = match.group("md5")
    return out


def _direct_url_record(url: str, sha: str, retrieved_at: str) -> RetrievalRecord:
    return RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=url,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": url},
        sha256=sha,
        retrieved_at=retrieved_at,
    )


def deposit_raw_mirror(
    *,
    pheno_path: str | Path,
    md5_path: str | Path,
    retrieved_at: str,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from already-retrieved files and its ``manifest.json``.

    Run once by hand after ``curl -sSL -O http://1002genomes.u-strasbg.fr/files/<name>``
    for the release and ``md5.txt``. The release's md5 must equal the one ``md5.txt``
    publishes for it. Idempotent by sha256: an existing file with the recorded hash is
    left alone, a differing one raises.
    """
    published = read_published_md5(md5_path)
    if published[PHENO_NAME] != PHENO_PUBLISHED_MD5:
        raise ValueError(
            f"md5.txt lists {published[PHENO_NAME]} for {PHENO_NAME}, the loader pins "
            f"{PHENO_PUBLISHED_MD5}"
        )
    if md5_file(pheno_path) != PHENO_PUBLISHED_MD5:
        raise ValueError(f"{pheno_path} does not have the published md5")
    root = raw_mirror_dir(data_root)
    sources = {PHENO_REL: Path(pheno_path), MD5_REL: Path(md5_path)}
    files: list[ArtifactRecord] = []
    for relpath, src in sources.items():
        sha = sha256_file(src)
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha256_file(dest) != sha:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        url = DATA_HOST + osp.basename(relpath)
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=sha,
                source=url,
                retrieval=_direct_url_record(url, sha, retrieved_at),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=files,
        si_data_sources=[DATA_HOST],
        si_expected=[PHENO_NAME],
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


def register_loss_of_function_matrices(
    *,
    source_dir: str | Path,
    md5_path: str | Path,
    retrieved_at: str,
    data_root: str | None = None,
) -> str:
    """Copy the two loss-of-function matrices into ``peter2018_1011_assemblies``.

    ``source_dir`` holds the bytes as retrieved from the host; each file's md5 must equal
    the one the host's ``md5.txt`` publishes for it, and the file is then copied into the
    tier directory and appended to the set's manifest with a ``direct_url`` record
    (``extend_assembly_set`` re-hashes it against the disk). The legacy copies under
    ``$DATA_ROOT/data/peter2018/`` are left in place: experiment 018 reads them by path.
    """
    published = read_published_md5(md5_path)
    tier_dir = Path(assembly_set_dir(PETER2018_1011, data_root))
    records: list[ArtifactRecord] = []
    for name, md5 in LOF_PUBLISHED_MD5.items():
        src = Path(source_dir) / name
        if published[name] != md5:
            raise ValueError(
                f"md5.txt lists {published[name]} for {name}, pinned {md5}"
            )
        if md5_file(src) != md5:
            raise ValueError(f"{src} does not have the published md5 {md5}")
        dest = tier_dir / name
        sha = sha256_file(src)
        if dest.exists():
            if sha256_file(dest) != sha:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        url = DATA_HOST + name
        records.append(
            ArtifactRecord(
                path=name,
                role=ROLE_MATRIX,
                bytes=dest.stat().st_size,
                sha256=sha,
                source=url,
                retrieval=_direct_url_record(url, sha, retrieved_at),
            )
        )
    return extend_assembly_set(PETER2018_1011, records, data_root)


# --------------------------------------------------------------------------- #
# Release reading
# --------------------------------------------------------------------------- #
class ReleaseShapeError(ValueError):
    """The release does not have the shape the paper and this loader state."""


def read_release(
    path: str | Path, conditions: dict[str, ConditionSpec]
) -> pd.DataFrame:
    """Read the growth-ratio matrix: isolates x conditions, exact decimals.

    ``round_trip`` keeps every stored float equal to the release's decimal string, so
    the verifier's re-read compares bit for bit. Refuses a shape other than 971 x 35, a
    header set other than the condition table's, a duplicated isolate, a missing cell
    and a negative or non-finite value (a colony-size ratio is non-negative).
    """
    frame = pd.read_csv(
        path, sep="\t", index_col=0, compression="gzip", float_precision="round_trip"
    )
    frame.index = frame.index.astype(str)
    if frame.shape != (N_ISOLATES, N_CONDITIONS):
        raise ReleaseShapeError(f"release shape {frame.shape}, expected (971, 35)")
    if set(frame.columns) != set(conditions):
        raise ReleaseShapeError(
            f"release headers {sorted(set(frame.columns) ^ set(conditions))} differ "
            "from the condition table"
        )
    if not frame.index.is_unique:
        raise ReleaseShapeError("an isolate appears twice in the release")
    values = frame.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ReleaseShapeError(f"{int(np.isnan(values).sum())} missing cells")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ReleaseShapeError("a ratio is negative or non-finite")
    return frame


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
class ReleaseLedger(BaseModel):
    """Every released number, accounted for (``preprocess/release_ledger.json``)."""

    model_config = ConfigDict(extra="forbid")

    pheno_sha256: str
    n_isolates_released: int
    n_conditions_released: int
    n_values_released: int
    n_values_served: int
    refused: dict[str, int]
    n_zero_values: int
    paper_measurements: int
    control_values_not_released: int
    n_isolates_in_genome_panel: int
    n_isolates_in_caudal2024_store: int | None
    conditions: dict[str, int]

    @model_validator(mode="after")
    def _accounted(self) -> ReleaseLedger:
        if self.n_values_served + sum(self.refused.values()) != self.n_values_released:
            raise ValueError("served + refused must equal released")
        if self.n_values_released + self.control_values_not_released != (
            self.paper_measurements
        ):
            raise ValueError(
                "released + the unreleased control must equal the paper's count"
            )
        return self


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
UNITS = (
    "growth ratio: end-point colony size (Gitter) of the isolate on the condition "
    "plate divided by its colony size on the YPD 30 C control plate after 40 h "
    "(phenoMatrix_35ConditionsNormalizedByYPD); 1.0 = grows as on the control; the "
    "aggregation over colonies is not stated"
)
REFERENCE_UNITS = (
    "the same isolate on the YPD 30 C control plate, the denominator of the ratio: 1.0 "
    "by construction"
)


def condition_environment(spec: ConditionSpec) -> Environment:
    """A condition plate's environment: medium, temperature, stresses, 40 h incubation."""
    gaps: list[ProvenanceGap] = []
    duration: float | None = INCUBATION_HOURS
    if spec.temperature_shifted:
        duration = None
        gaps.append(
            ProvenanceGap(
                field="duration_hours",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the Methods' one incubation sentence reads 'incubated at 30 C for "
                f"40 h'; Table 2 moves this plate to {spec.temperature_c:g} C, and "
                "whether its incubation also lasted 40 h is not stated",
            )
        )
    return Environment(
        media=spec.media,
        temperature=Temperature(value=spec.temperature_c),
        perturbations=list(spec.perturbations),
        aerobicity="aerobic",
        duration_hours=duration,
        provenance_gaps=gaps,
    )


def control_environment() -> Environment:
    """The YPD 30 C control plate (Table 2 sheet row 5), every ratio's denominator."""
    return Environment(
        media=PETER_YPD,
        temperature=Temperature(value=CONTROL_TEMPERATURE_C),
        aerobicity="aerobic",
        duration_hours=INCUBATION_HOURS,
    )


def growth_phenotype(value: float) -> EnvironmentResponsePhenotype:
    """One released growth ratio."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.condition_growth_ratio,
        assay_type=AssayType.colony_size_array,
        environment_response=value,
        n_samples=N_SAMPLES,
        sample_unit=SampleUnit.colony,
        units=UNITS,
        provenance_gaps=[
            ProvenanceGap(
                field="environment_response_se",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the release carries one ratio per (isolate, condition); the "
                "per-colony sizes, the aggregation statistic and any dispersion are "
                "'available from the corresponding authors upon reasonable request', "
                "not released",
            )
        ],
    )


def shared_reference(dataset_name: str) -> EnvironmentResponseExperimentReference:
    """The one reference: S288C coordinates, the YPD 30 C plate, ratio 1.0.

    ``genome_reference`` is the coordinate system every isolate's perturbations are
    written against (as in ``caudal2024``); the ratio's denominator is the SAME isolate
    on the control plate, which is what makes the reference value 1.0 by construction.
    """
    return EnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=control_environment(),
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.condition_growth_ratio,
            assay_type=AssayType.colony_size_array,
            environment_response=1.0,
            units=REFERENCE_UNITS,
        ),
    )


def publication() -> Publication:
    """The Nature paper."""
    return Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")


@register_dataset
class IsolateGrowthPeter2018Dataset(ExperimentDataset):
    """Peter 2018: one record per (isolate, condition), 33,985 records."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/isolate_growth_peter2018",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Build or load the store under ``root``."""
        self.conditions = build_conditions()
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return EnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return EnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The release plus the six genomes-tier files the genotypes are built from."""
        return [PHENO_NAME, *GENOTYPE_FILES]

    def download(self) -> None:
        """Link the release and the genomes-tier files, each sha256-verified.

        The release is checked against ``PHENO_SHA256`` and the raw-mirror manifest must
        record that same digest (``ManifestPinMismatchError`` otherwise); the six tier
        files against the tier manifest.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        check_manifest_pin(
            PHENO_REL, manifest_sha256(manifest, PHENO_REL), PHENO_SHA256
        )
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(
            raw_mirror_dir(data_root) / PHENO_REL,
            osp.join(self.raw_dir, PHENO_NAME),
            PHENO_SHA256,
        )
        link_genotype_files(self.raw_dir, data_root)
        log.info("Peter 2018 raw files linked into %s (sha256 verified)", self.raw_dir)

    # ---- build ---------------------------------------------------------------
    @post_process
    def process(self) -> None:
        """Build 971 x 35 records, isolate-major, interning each isolate's genotype."""
        data_root = _data_root()
        peter = load_genome_manifest(PETER2018_1011, data_root)
        verify_raw_files(
            self.raw_dir,
            {
                PHENO_NAME: PHENO_SHA256,
                **{name: peter.record(name).sha256 for name in GENOTYPE_FILES},
            },
        )
        os.makedirs(self.preprocess_dir, exist_ok=True)
        release = read_release(osp.join(self.raw_dir, PHENO_NAME), self.conditions)
        isolates = sorted(release.index)
        genotypes = NaturalIsolateGenotypes(
            self.raw_dir, self.preprocess_dir, isolates, data_root
        )
        genotypes.write_ledger(osp.join(self.preprocess_dir, GENOTYPE_LEDGER))

        reference = shared_reference(self.name)
        pub = publication()
        environments = {c: condition_environment(s) for c, s in self.conditions.items()}
        served: dict[str, pd.Series[float]] = {
            code: release[code].astype("float64") for code in self.conditions
        }

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        gene_names: set[str] = set()
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for strain in tqdm(isolates, desc="isolates"):
                genotype = genotypes.genotype(strain)
                gene_names.update(genotype.systematic_gene_names)
                pointer = self._intern_genotype(genotype, strain, itxn)
                for code in self.conditions:
                    experiment = EnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
                        environment=environments[code],
                        phenotype=growth_phenotype(float(served[code][strain])),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._record_bytes(experiment, pointer, reference, pub, itxn),
                    )
                    idx += 1
        env.close()
        interned_env.close()
        if idx != EXPECTED_RECORDS:
            raise ValueError(f"wrote {idx} records, expected {EXPECTED_RECORDS}")
        self._gene_names = gene_names
        self._write_ledger(release, isolates, data_root)
        pd.DataFrame({"strain_id": isolates}).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )
        log.info("Wrote %d Peter 2018 records over %d isolates", idx, len(isolates))

    @staticmethod
    def _intern_genotype(genotype: Genotype, strain: str, itxn: Any) -> dict[str, str]:
        """Intern one isolate's genotype once; return the pointer its records carry.

        The digest and stored bytes are exactly what ``_maybe_intern`` would write
        (sha256 of the canonical JSON, the python-mode dump pickled), computed once per
        isolate instead of once per record.
        """
        digest = compute_sha256_hash(canonical_json(genotype))
        if itxn.get(digest.encode()) is None:
            itxn.put(digest.encode(), pickle.dumps(genotype.model_dump()))
        return {"$ref": digest, "name": strain}

    def _record_bytes(
        self,
        experiment: EnvironmentResponseExperiment,
        genotype_pointer: dict[str, str],
        reference: ExperimentReference,
        publication: Publication,
        itxn: Any,
    ) -> bytes:
        """Serialize one record with the genotype pointer and the interned constants."""
        exp = experiment.model_dump(exclude={"genotype"})
        exp["genotype"] = dict(genotype_pointer)
        rec: dict[str, Any] = {
            "experiment": exp,
            "reference": reference.model_dump(),
            "publication": publication.model_dump(),
        }
        self._maybe_intern(
            rec["experiment"],
            "environment",
            experiment.environment,
            experiment.environment.media.name,
            itxn,
        )
        self._maybe_intern(rec, "reference", reference, reference.dataset_name, itxn)
        self._maybe_intern(rec, "publication", publication, PAPER_DOI, itxn)
        return pickle.dumps(rec)

    def _write_ledger(
        self, release: pd.DataFrame, isolates: list[str], data_root: str
    ) -> None:
        caudal = osp.join(
            data_root, "data/torchcell/caudal_pantranscriptome2024/preprocess/data.csv"
        )
        in_caudal: int | None = None
        if osp.exists(caudal):
            built = set(pd.read_csv(caudal)["strain_id"].astype(str))
            in_caudal = len(set(isolates) & built)
        ledger = ReleaseLedger(
            pheno_sha256=PHENO_SHA256,
            n_isolates_released=release.shape[0],
            n_conditions_released=release.shape[1],
            n_values_released=int(release.size),
            n_values_served=EXPECTED_RECORDS,
            refused={},
            n_zero_values=int((release.to_numpy() == 0).sum()),
            paper_measurements=PAPER_MEASUREMENTS,
            control_values_not_released=N_ISOLATES,
            n_isolates_in_genome_panel=len(isolates),
            n_isolates_in_caudal2024_store=in_caudal,
            conditions={
                code: spec.table_s2_row for code, spec in self.conditions.items()
            },
        )
        with open(osp.join(self.preprocess_dir, "release_ledger.json"), "w") as handle:
            handle.write(ledger.model_dump_json(indent=2))

    # ---- gene set --------------------------------------------------------------
    def compute_gene_set(self) -> GeneSet:
        """Systematic names over the 971 interned genotypes, one read per isolate."""
        names: set[str] | None = getattr(self, "_gene_names", None)
        if names is None:
            names = set()
            self._load_interned()
            interned = self._interned or {}
            seen: set[str] = set()
            env = lmdb.open(
                osp.join(self.processed_dir, "lmdb"), readonly=True, lock=False
            )
            with env.begin() as txn:
                for _, value in txn.cursor():
                    ref = pickle.loads(value)["experiment"]["genotype"]["$ref"]
                    if ref in seen:
                        continue
                    seen.add(ref)
                    names.update(self.extract_systematic_gene_names(interned[ref]))
            env.close()
        return GeneSet(names)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing happens inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Records are assembled inside process()."""
        raise NotImplementedError(
            "IsolateGrowthPeter2018Dataset builds its records in process()"
        )


def verify_build(dataset_root: str, data_root: str | None = None) -> Any:
    """L0-L4 report for a built store (``build_dataset_lmdb --verify`` calls this).

    The family runner's per-dataset entry point over ``dataset_root``; the CLI writes
    the report to ``<dataset_root>/preprocess/verification_report.json``.
    """
    from torchcell.verification import runners

    return runners.verify_natural_isolate_dataset(
        "isolate_growth_peter2018", data_root or _data_root(), abs_root=dataset_root
    )


def main() -> None:
    """Build/load the dataset and print one record's shape."""
    data_root = _data_root()
    dataset = IsolateGrowthPeter2018Dataset(
        root=osp.join(data_root, "data/torchcell/isolate_growth_peter2018")
    )
    print(f"len = {len(dataset)}")
    record = dataset[0]
    print(
        len(record["experiment"].genotype.perturbations),
        record["experiment"].phenotype.environment_response,
    )
    print(json.dumps(record["reference"].model_dump(mode="json"))[:400])


if __name__ == "__main__":
    main()
