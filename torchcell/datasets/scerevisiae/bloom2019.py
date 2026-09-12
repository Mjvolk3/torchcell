# torchcell/datasets/scerevisiae/bloom2019
# [[torchcell.datasets.scerevisiae.bloom2019]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/bloom2019
# Test file: tests/torchcell/datasets/scerevisiae/test_bloom2019.py
"""Bloom 2019 segregant panels: 16 biparental crosses, 13,950 haploid segregants, 38
growth conditions (eLife 8:e49212, doi:10.7554/eLife.49212; citation key
``bloomRareVariantsContribute2019``).

The 50th dataset and the first ONTOLOGY extension through the incremental admission
gate: a segregant carries no gene edit, so its genotype is a ``SegregantGenotype``
(a haplotype mosaic against two sha256-pinned parent assemblies), a sibling of
``Genotype`` rather than a subclass. Design: ``[[plan.bloom2019-segregant-dataset.2026.09.11]]``,
``[[torchcell.datamodels.eqtl-data-model]]``.

GENOTYPE. The released per-cross matrices ``genotype_<cross>.tsv.gz`` hold one row per
segregant and one column per marker, coded 1 = parent 1, 2 = parent 2 (per
``cross_genotypes_README``). They are the R/qtl hard calls the authors' ``mapping.R``
wrote out (``g=pull.argmaxgeno(cross)``), so every block carries ``posterior=1.0`` and
the call method is quoted. Marker names are ``<chr>_<pos>_<ref>_<alt>_<index>``; the alt
field carries commas at multiallelic sites and ref/alt are multi-base at indels, so the
name is split on ``_`` into exactly five fields. Columns are grouped by chromosome in
karyotype order but positions are NOT monotone within a chromosome (10 to 85 backsteps
per chromosome on cross A), so the loader sorts markers by (chromosome, position) and
asserts the sort is a pure column permutation before run-length encoding each segregant's
calls per chromosome into ``HaplotypeBlock``s (start/end = the first and last marker
position of the run; inter-block gaps are unassigned). Re-expanding the blocks at the
cross's sorted marker positions reproduces the released row exactly; that round trip is
the L2 check. Cross A: min 55, median 83, mean 85.6, max 278 blocks per segregant.

PARENTS. From Figure 1 source data 1 (sheet ``Crosses and Strains``): the diploid parent,
the Peter 2018 strain id of parent 1, both parents' genotype strings, the cross code, the
magic-marker plasmid and the segregant count. Parents are pinned to the 1011-collection
assemblies tarball (``peterGenomeEvolution10112018/data/1011Assemblies.tar.gz``, mirrored
and sha256-pinned; member index beside it) by Peter id; BY is the S288C reference. The
xls writes strain 273614 as ``SACE_MAA`` while the assembly member index carries it as
``MAA``; the ``SACE_`` prefix is the same one that once dropped an isolate's variants
(issue #73), so the mapping is explicit and a parent absent from the index raises.

PHENOTYPE. ``phenotypes.tsv.gz`` has 13,950 rows x 40 trait columns; 38 are served.
Trait names are ``Condition;Concentration;PermutationGroup`` (``rr_fx.R``); the
trailing digit is the plate BATCH, not a replicate index. For 36 conditions the stored
value is ``residuals(lm(s.radius.mean ~ ctrl.s.radius.mean))`` against the same-cross,
same-batch, same-layout YPD plate (or YNB plate for the YNB conditions;
``process_images.R``), served as ``MeasurementType.control_regression_residual``; for
``YPD;;1`` and ``YNB;;1`` the value is the raw colony mean radius in image pixels
(``MeasurementType.colony_size``). ``YPD;;2`` and ``YPD;;3`` are the batch-2 and
batch-3 control plates: they are the regressors for those batches and are dropped from
every analysis in the source (``mapping.R``: "remove the results from the two
additional YPD replicates"), so they are not served as conditions. ``mapping.R`` averages
the duplicate plates and then mean-imputes missing cells per trait within a cross; the
release carries no missingness mask, so imputed cells cannot be identified (a
dataset-level limitation, not a per-record gap). The reference is 0 for every condition:
exact by construction for a residual, and the scale's physical zero (no colony) with a
``ProvenanceGap`` for the two absolute conditions, whose parental radii are not released.

ENVIRONMENT. Temperature lives on ``Environment.temperature`` (15, 30, 37 C for the YPD
temperature series); the 30 C of every other plate is a documented representative with a
``ProvenanceGap`` (the Methods state only the 48 h incubation, not its temperature). pH 3
and pH 8 are ``EnvironmentPhysicalPerturbation(factor=ph)``. Carbon sources, glycerol and
the two ethanol conditions are MEDIA (typed components, so the metabolism layer resolves
them to exchange reactions); the stress compounds are ``SmallMoleculePerturbation``s on
YPD. All plates are solid; the 48 h liquid YPD outgrowth is a pre-culture and is not
modeled. ``duration_hours=48`` from "Plates were incubated for 48 hr".

RAW MIRROR. ``$DATA_ROOT/torchcell-raw/bloomRareVariantsContribute2019/`` with a
``manifest.json`` (``torchcell.literature.manifest.Manifest``): ``data/`` holds the 18
zip members read from the authors' Dropbox share (``zip_member`` retriever, container
sha256 pinned) and the eLife xls; ``code/`` the four R files quoted above at commit
``c913c9ae``; ``paper/`` the eLife JATS XML and PDF. Exactly the files this loader and
its verifier read; the joint biallelic matrices and the parent VCF are not deposited.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import (
    YNB_GLUCOSE_SOLID,
    YP_ETHANOL,
    YP_FRUCTOSE,
    YP_GALACTOSE,
    YP_GLYCEROL,
    YP_LACTATE,
    YP_MALTOSE,
    YP_MANNOSE,
    YP_RAFFINOSE,
    YP_SUCROSE,
    YP_TREHALOSE,
    YP_XYLOSE,
    YPD,
    YPD_ETHANOL,
)
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    HaplotypeBlock,
    MeasurementType,
    Media,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SegregantGenotype,
    SegregantGrowthExperiment,
    SegregantGrowthExperimentReference,
    SegregantParent,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_PAPER_PDF,
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "bloomRareVariantsContribute2019"
PAPER_DOI = "10.7554/eLife.49212"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
PETER_DIR_REL = "torchcell-library/peterGenomeEvolution10112018/data"
ASSEMBLY_TAR = "1011Assemblies.tar.gz"
ASSEMBLY_TAR_SHA256 = "53540d095958ae8c32509c04485f0d2d0948069c7647f828698d611899a9b4da"
ASSEMBLY_INDEX = "1011Assemblies.tar.gz.member_index.tsv"
S288C_ASSEMBLY = "S288C_reference_genome_R64-4-1_20230830 (SGD; torchcell reference)"

DROPBOX_URL = (
    "https://www.dropbox.com/sh/jqm7a11zz9laytd/AABaE0EfQxLH6ounPhJ7yYWya?dl=1"
)
DROPBOX_ZIP_SHA256 = "78ded0dbef051878f322b26c9fa6bc8f1768a5fdaf7ef83362260d8dde7c0a5d"
XLS_URL = "https://cdn.elifesciences.org/articles/49212/elife-49212-fig1-data1-v2.xls"
XML_URL = "https://cdn.elifesciences.org/articles/49212/elife-49212-v2.xml"
PDF_URL = "https://cdn.elifesciences.org/articles/49212/elife-49212-v2.pdf"
GITHUB_REPO = "joshsbloom/yeast-16-parents"
GITHUB_COMMIT = "c913c9ae7fd237329f639de02e7ec511b048730f"

XLS_NAME = "elife-49212-fig1-data1-v2.xls"
XML_NAME = "elife-49212-v2.xml"
PDF_NAME = "elife-49212-v2.pdf"
PHENOTYPES_NAME = "phenotypes.tsv.gz"
README_NAME = "cross_genotypes_README"
CODE_FILES = {
    "mapping.R": "analysis/mapping.R",
    "mapping_fx.R": "analysis/mapping_fx.R",
    "process_images.R": "phenotyping/code/process_images.R",
    "rr_fx.R": "phenotyping/code/rr_fx.R",
}

# The 16 crosses in the README's order; cross code -> (parent 1, parent 2) as written.
CROSSES = [
    "375", "A", "376", "B", "377", "393", "381", "3008",
    "2999", "3000", "3001", "3049", "3003", "3004", "3043", "3028",
]  # fmt: skip
EXPECTED_SEGREGANTS = {
    "375": 876, "A": 951, "376": 709, "B": 953, "377": 841, "393": 865, "381": 944,
    "3008": 795, "2999": 936, "3000": 896, "3001": 648, "3049": 884, "3003": 899,
    "3004": 867, "3043": 943, "3028": 943,
}  # fmt: skip
N_SEGREGANTS = 13950
N_CONDITIONS = 38
EXPECTED_RECORDS = N_SEGREGANTS * N_CONDITIONS

# README parent label -> Peter 2018 strain id (None = BY, the S288C reference). The xls
# "Strain ID of Parent 1 in Peter et al. 2018" column supplies these; 273614 is written
# SACE_MAA there and MAA in the assembly member index.
PARENT_PETER_ID: dict[str, str | None] = {
    "BYa": None,
    "M22": "ADR",
    "RMx": "AAA",
    "YPS163a": "ACD",
    "YJM145x": "AAS",
    "CLIB413a": "ACG",
    "YJM978x": "ADG",
    "YJM454a": "ABH",
    "YPS1009x": "ADF",
    "I14a": "ACS",
    "Y10x": "ACI",
    "PW5a": "ADE",
    "273614xa": "MAA",
    "YJM981x": "ADI",
    "CBS2888a": "ABL",
    "CLIB219x": "ACQ",
}
# README label -> the strain-name prefix used in the xls "Parent 1"/"Parent 2" cells.
PARENT_XLS_PREFIX: dict[str, str] = {
    "BYa": "BY ",
    "M22": "M22 ",
    "RMx": "RM ",
    "YPS163a": "YPS163 ",
    "YJM145x": "YJM145 ",
    "CLIB413a": "Clib413 ",
    "YJM978x": "YJM978 ",
    "YJM454a": "YJM454 ",
    "YPS1009x": "YPS1009 ",
    "I14a": "I14 ",
    "Y10x": "Y10 ",
    "PW5a": "PW5 ",
    "273614xa": "273614 ",
    "YJM981x": "YJM981 ",
    "CBS2888a": "CBS2888 ",
    "CLIB219x": "Clib219 ",
}

CALL_METHOD = (
    "R/qtl argmax.geno hard call written by analysis/mapping.R "
    "(`g=pull.argmaxgeno(cross)`), released as genotype_<cross>.tsv.gz "
    "(1 = parent 1, 2 = parent 2)"
)
CALL_METHOD_QUOTE = "g=pull.argmaxgeno(cross)"

# Sourced from the eLife XML (Methods, "Phenotyping by endpoint colony growth").
DUPLICATE_QUOTE = (
    "Segregants were arrayed to 384-well liquid plates in duplicate with different "
    "plate positions across the duplicates."
)
INCUBATION_QUOTE = (
    "Plates were incubated for 48 hr and end-point growth was quantified by automated "
    "plate imaging using the colony arraying robot."
)
N_SAMPLES = 2
INCUBATION_HOURS = 48.0
# Documented representative: the Methods state the 48 h incubation but not its
# temperature; 30 C is carried with a ProvenanceGap on every non-temperature plate.
DEFAULT_TEMPERATURE_C = 30.0
NOT_SERVED = {"YPD;;2", "YPD;;3"}
# The control plate a residual was regressed on, by batch, and the served column that
# carries that plate's ENVIRONMENT: the batch-2 and batch-3 YPD plates are the same
# medium and temperature as the batch-1 plate (only the batch differs), so their
# environment is the YPD;;1 environment.
CONTROL_ENVIRONMENT_COLUMN = {
    "YPD;;1": "YPD;;1",
    "YPD;;2": "YPD;;1",
    "YPD;;3": "YPD;;1",
    "YNB;;1": "YNB;;1",
}

ph_unit = ConcentrationUnit.ph
_MM = ConcentrationUnit.millimolar
_UM = ConcentrationUnit.micromolar
_M = ConcentrationUnit.molar
_UGML = ConcentrationUnit.ug_per_ml
_PCT = ConcentrationUnit.percent_w_v
_PCT_VV = ConcentrationUnit.percent_v_v


class ConditionSpec(BaseModel):
    """One served phenotype column and its typed environment."""

    column: str
    media: Media
    temperature_c: float
    temperature_sourced: bool
    perturbations: list[SmallMoleculePerturbation | EnvironmentPhysicalPerturbation]
    measurement_type: MeasurementType
    control_column: str
    dose_quote: str | None = None


def _smp(name: str, value: float, unit: ConcentrationUnit) -> SmallMoleculePerturbation:
    return SmallMoleculePerturbation(
        compound=resolved_compound(name),
        concentration=Concentration(value=value, unit=unit),
    )


def _ph(value: float) -> EnvironmentPhysicalPerturbation:
    return EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.ph, magnitude=Concentration(value=value, unit=ph_unit)
    )


def _spec(
    column: str,
    media: Media,
    perturbations: list[Any],
    *,
    control: str,
    temperature_c: float = DEFAULT_TEMPERATURE_C,
    temperature_sourced: bool = False,
    measurement_type: MeasurementType = MeasurementType.control_regression_residual,
    dose_quote: str | None = None,
) -> ConditionSpec:
    return ConditionSpec(
        column=column,
        media=media,
        temperature_c=temperature_c,
        temperature_sourced=temperature_sourced,
        perturbations=perturbations,
        measurement_type=measurement_type,
        control_column=control,
        dose_quote=dose_quote,
    )


def build_conditions() -> dict[str, ConditionSpec]:
    """The 38 served columns of ``phenotypes.tsv.gz``, keyed by the header string.

    Doses come from the header when it carries one and from the xls ``Phenotypes`` sheet
    otherwise (copper sulfate, magnesium chloride, manganese sulfate carry no dose in the
    header; sorbitol is header-only). The control column is the same-batch YPD (or YNB)
    plate the residual was regressed on (``process_images.R``).
    """
    residual = MeasurementType.control_regression_residual
    absolute = MeasurementType.colony_size
    specs = [
        _spec("6-azauracil;50ug/mL;3", YPD, [_smp("6-azauracil", 50.0, _UGML)], control="YPD;;3", dose_quote="6-azauracil | 10 mg/mL | 20 mL | DMSO | 0.2 g | 1650 | 8.25 mL | 50 ug/mL"),
        _spec("Cadmium_Chloride;75uM;2", YPD, [_smp("cadmium chloride", 75.0, _UM)], control="YPD;;2", dose_quote="Cadmium chloride | 100 mM | 15 mL | H2O | 0.274 g | 1650 | 1.24 mL | 75 uM"),
        _spec("Caffeine;15mM;2", YPD, [_smp("caffeine", 15.0, _MM)], control="YPD;;2", dose_quote="Caffeine | 1650 | 4.8 g | 15mM"),
        _spec("Cobalt_Chloride;2mM;2", YPD, [_smp("cobalt chloride", 2.0, _MM)], control="YPD;;2", dose_quote="Cobalt Chloride | 1 M | 15 mL | H2O | 1.94 g | 1650 | 3.3 mL | 2 mM"),
        _spec("Congo_Red;75ug/mL;3", YPD, [_smp("Congo red", 75.0, _UGML)], control="YPD;;3", dose_quote="Congo Red | 5 mg/mL | 15 mL | H2O | 0.075 g | 1650 | 24.75 mL | 75 ug/mL"),
        _spec("Copper_Sulfate;;1", YPD, [_smp("copper sulfate", 6.0, _MM)], control="YPD;;1", dose_quote="Copper sulfate | 1 | M | H2O | 6mM | 12 mM | 0.012 | 1/2"),
        _spec("Diamide;1.5mM;3", YPD, [_smp("diamide", 1.5, _MM)], control="YPD;;3", dose_quote="Diamide | 125 mM | 15 mL | DMSO | 0.323 g | 1650 | 19.80 mL | 1.5 mM"),
        _spec("EGTA;5mM;3", YPD, [_smp("EGTA", 5.0, _MM)], control="YPD;;3", dose_quote="EGTA | 500 mM | 50mL | H2O | 9.51 g | 1650 | 16.5 mL | 5 mM"),
        _spec("EtOH;;1", YP_ETHANOL, [], control="YPD;;1"),
        _spec("EtOH_Glucose;;1", YPD_ETHANOL, [], control="YPD;;1"),
        _spec("Fluconazole;100uM;2", YPD, [_smp("fluconazole", 100.0, _UM)], control="YPD;;2", dose_quote="Fluconazole | 10 mM | 30 mL | DMSO | 0.0918 g | 250 | 2.5 mL | 100 uM"),
        _spec("Formamide;2%;2", YPD, [_smp("formamide", 2.0, _PCT_VV)], control="YPD;;2", dose_quote="Formamide | 1 | - | - | - | 250 | 5 mL | 0.02"),
        _spec("Fructose;;1", YP_FRUCTOSE, [], control="YPD;;1"),
        _spec("Galactose;;1", YP_GALACTOSE, [], control="YPD;;1"),
        _spec("Glycerol;;1", YP_GLYCEROL, [], control="YPD;;1"),
        _spec("Lactate;;1", YP_LACTATE, [_ph(6.0)], control="YPD;;1", dose_quote="Lactate | 20 | % | H2O, pH = 6 | YP"),
        _spec("Lithium_Chloride;100mM;2", YPD, [_smp("lithium chloride", 100.0, _MM)], control="YPD;;2", dose_quote="Lithium Chloride | 100mM"),
        _spec("Magnesium_Chloride;;1", YPD, [_smp("magnesium chloride", 100.0, _MM)], control="YPD;;1", dose_quote="Magnesium Chloride | 2.5 | M | H2O | 100mM | 200mM | 0.2 | 1/2"),
        _spec("Maltose;;1", YP_MALTOSE, [], control="YPD;;1"),
        _spec("Manganese_Sulfate;;1", YPD, [_smp("manganese sulfate", 10.0, _MM)], control="YPD;;1", dose_quote="Manganese Sulfate | 1.67 | M | H2O | 10mM | 100mM | 0.1 | 1/10"),
        _spec("Mannose;;1", YP_MANNOSE, [], control="YPD;;1"),
        _spec("Methotrexate;0.4mM;2", YPD, [_smp("methotrexate", 0.4, _MM)], control="YPD;;2", dose_quote="Methotrexate | 10 mM | 10 mL | DMSO | 0.0454 g | 250 | 10 mL | 0.4 mM"),
        _spec("Neomycin;5mg/mL;2", YPD, [_smp("neomycin", 5000.0, _UGML)], control="YPD;;2", dose_quote="Neomycin | 50 mg/mL | H2O | 5 mg/mL"),
        _spec("Paraquat;0.75mM;3", YPD, [_smp("paraquat", 0.75, _MM)], control="YPD;;3", dose_quote="Paraquat | 0.1 M | 30 mL | H2O | 0.7716 g | 1650 | 12.375 mL | 0.75 mM"),
        _spec("Raffinose;;1", YP_RAFFINOSE, [], control="YPD;;1"),
        _spec("SDS;0.075%;3", YPD, [_smp("sodium dodecyl sulfate", 0.075, _PCT)], control="YPD;;3", dose_quote="SDS | 0.1 | 100 mL | H2O | - | 1650 | 12.375 mL | 0.00075"),
        _spec("Sorbitol;1.5M;2", YPD, [_smp("sorbitol", 1.5, _M)], control="YPD;;2"),
        _spec("Sucrose;;1", YP_SUCROSE, [], control="YPD;;1"),
        _spec("Trehalose;;1", YP_TREHALOSE, [], control="YPD;;1"),
        _spec("Tunicamycin;3uM;3", YPD, [_smp("tunicamycin", 3.0, _UM)], control="YPD;;3", dose_quote="Tunicamycin | 1.5 mM | 15 mL | DMSO | 0.0189 g | 250 | 0.5 mL | 3 uM"),
        _spec("Xylose;;1", YP_XYLOSE, [], control="YPD;;1"),
        _spec("YNB;;1", YNB_GLUCOSE_SOLID, [], control="YNB;;1", measurement_type=absolute),
        _spec("YNB;ph3;1", YNB_GLUCOSE_SOLID, [_ph(3.0)], control="YNB;;1", dose_quote="pH = 3 | H2O | 3500 | YNB | 2% glucose"),
        _spec("YNB;ph8;1", YNB_GLUCOSE_SOLID, [_ph(8.0)], control="YNB;;1", dose_quote="pH = 8 | H2O | 3500 | YNB | 2% glucose"),
        _spec("YPD;;1", YPD, [], control="YPD;;1", measurement_type=absolute),
        _spec("YPD;15;1", YPD, [], control="YPD;;1", temperature_c=15.0, temperature_sourced=True, dose_quote="YPD 15"),
        _spec("YPD;37;1", YPD, [], control="YPD;;1", temperature_c=37.0, temperature_sourced=True, dose_quote="YPD 37"),
        _spec("Zeocin;25ug/mL;3", YPD, [_smp("zeocin", 25.0, _UGML)], control="YPD;;3", dose_quote="Zeocin | 100 mg/mL | - | H2O | - | 250 | 62.5 ul | 25 ug/ml"),
    ]  # fmt: skip
    conditions = {s.column: s for s in specs}
    if len(conditions) != N_CONDITIONS:
        raise ValueError(f"expected {N_CONDITIONS} conditions, built {len(conditions)}")
    n_residual = sum(1 for s in specs if s.measurement_type is residual)
    if n_residual != 36 or len(specs) - n_residual != 2:
        raise ValueError("expected exactly 36 residual and 2 absolute conditions")
    return conditions


# --------------------------------------------------------------------------- #
# Marker parsing + run-length encoding
# --------------------------------------------------------------------------- #
class Marker(BaseModel):
    """One parsed marker column name ``<chr>_<pos>_<ref>_<alt>_<index>``."""

    chromosome: str
    position: int
    ref: str
    alt: str
    index: int
    column: int  # position in the file's header order


def parse_marker(name: str, column: int) -> Marker:
    """Split a marker name into its five fields (alt may carry commas, ref/alt multi-base)."""
    parts = name.split("_")
    if len(parts) != 5:
        raise ValueError(f"marker name {name!r} does not have five '_' fields")
    chrom, pos, ref, alt, idx = parts
    return Marker(
        chromosome=chrom,
        position=int(pos),
        ref=ref,
        alt=alt,
        index=int(idx),
        column=column,
    )


def read_marker_names(path: str | Path) -> list[str]:
    """The marker column names of one released matrix (the id column excluded)."""
    header = pd.read_csv(path, sep="\t", compression="gzip", index_col=0, nrows=0)
    return [str(c) for c in header.columns]


def read_marker_matrix(path: str | Path) -> pd.DataFrame:
    """One released matrix as int8 (segregant id index x marker columns).

    ``dtype`` is given per marker column so the string id column is left alone; a
    scalar ``dtype="int8"`` would try to cast the ids and fail.
    """
    names = read_marker_names(path)
    return pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        index_col=0,
        dtype={name: "int8" for name in names},
    )


def sorted_markers(header: list[str]) -> list[Marker]:
    """Markers sorted by (chromosome karyotype order, position); a pure permutation."""
    markers = [parse_marker(name, i) for i, name in enumerate(header)]
    order = sorted(markers, key=lambda m: (_CHROM_ORDER[m.chromosome], m.position))
    if sorted(m.column for m in order) != list(range(len(header))):
        raise ValueError("marker sort is not a permutation of the header columns")
    return order


_CHROM_NAMES = [
    "chrI", "chrII", "chrIII", "chrIV", "chrV", "chrVI", "chrVII", "chrVIII", "chrIX",
    "chrX", "chrXI", "chrXII", "chrXIII", "chrXIV", "chrXV", "chrXVI",
]  # fmt: skip
_CHROM_ORDER = {name: i for i, name in enumerate(_CHROM_NAMES)}
CHROM_TO_GENOME_INDEX = {name: i + 1 for i, name in enumerate(_CHROM_NAMES)}


def encode_blocks(calls: np.ndarray, markers: list[Marker]) -> list[HaplotypeBlock]:
    """Run-length encode one segregant's sorted calls into per-chromosome blocks.

    ``calls`` is aligned with ``markers`` (already sorted by chromosome, position). A block
    ends at a parent change or a chromosome change; its ``start``/``end`` are the first and
    last marker positions of the run, and ``n_markers`` counts the markers it spans.
    """
    blocks: list[HaplotypeBlock] = []
    start = 0
    n = len(markers)
    for i in range(1, n + 1):
        if (
            i == n
            or calls[i] != calls[start]
            or markers[i].chromosome != markers[start].chromosome
        ):
            blocks.append(
                HaplotypeBlock(
                    chromosome=markers[start].chromosome,
                    start=markers[start].position,
                    end=markers[i - 1].position,
                    parent=int(calls[start]),  # type: ignore[arg-type]
                    posterior=1.0,
                    n_markers=i - start,
                )
            )
            start = i
    return blocks


def expand_blocks(blocks: list[HaplotypeBlock], markers: list[Marker]) -> np.ndarray:
    """Inverse of ``encode_blocks``: the int8 call at every sorted marker position."""
    out = np.zeros(len(markers), dtype=np.int8)
    i = 0
    for block in blocks:
        end = i + block.n_markers
        run = markers[i:end]
        if (
            not run
            or run[0].chromosome != block.chromosome
            or run[0].position != block.start
            or run[-1].position != block.end
        ):
            raise ValueError(
                f"block {block} does not align with the marker list at {i}"
            )
        out[i:end] = block.parent
        i = end
    if i != len(markers):
        raise ValueError(f"blocks cover {i} of {len(markers)} markers")
    return out


# --------------------------------------------------------------------------- #
# Raw-mirror deposit
# --------------------------------------------------------------------------- #
def _sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/bloomRareVariantsContribute2019``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def raw_relpaths() -> dict[str, str]:
    """Every file the loader or verifier reads, keyed by short name -> mirror-relative path."""
    rel = {
        PHENOTYPES_NAME: f"data/{PHENOTYPES_NAME}",
        README_NAME: f"data/{README_NAME}",
        XLS_NAME: f"data/{XLS_NAME}",
        XML_NAME: f"paper/{XML_NAME}",
        PDF_NAME: f"paper/{PDF_NAME}",
    }
    for cross in CROSSES:
        rel[f"genotype_{cross}.tsv.gz"] = f"data/genotype_{cross}.tsv.gz"
    for name in CODE_FILES:
        rel[name] = f"code/{name}"
    return rel


def deposit_raw_mirror(
    *,
    zip_members_dir: str | Path,
    xls_path: str | Path,
    xml_path: str | Path,
    pdf_path: str | Path,
    code_dir: str | Path,
    retrieved_at_data: str,
    retrieved_at_paper: str,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from already-retrieved files and its ``manifest.json``.

    Run once by hand after the retrievals in the module docstring; idempotent by sha256
    (an existing file with the recorded hash is left alone, a differing one raises). The
    Dropbox share re-zips on every request, so the container's sha256 is what pins the
    18 members; each member's own sha256 is recorded as its canonical anchor.
    ``retrieved_at_data`` dates the zip and xls retrievals, ``retrieved_at_paper`` the
    XML, PDF and R-file retrievals.
    """
    root = raw_mirror_dir(data_root)
    rel = raw_relpaths()
    sources: dict[str, tuple[Path, RetrievalRecord]] = {}

    def zip_record(member: str, sha: str) -> RetrievalRecord:
        return RetrievalRecord(
            method=RetrievalMethod.direct_url,
            source_url=DROPBOX_URL,
            retriever="torchcell.literature.retrieve.zip_member",
            params={
                "url": DROPBOX_URL,
                "member": member,
                "container_sha256": DROPBOX_ZIP_SHA256,
            },
            sha256=sha,
            retrieved_at=retrieved_at_data,
        )

    def url_record(
        url: str, sha: str, retrieved_at: str, **params: Any
    ) -> RetrievalRecord:
        return RetrievalRecord(
            method=RetrievalMethod.direct_url,
            source_url=url,
            retriever="torchcell.literature.retrieve.direct_url",
            params={"url": url, **params},
            sha256=sha,
            retrieved_at=retrieved_at,
        )

    members = [PHENOTYPES_NAME, README_NAME] + [f"genotype_{c}.tsv.gz" for c in CROSSES]
    for member in members:
        src = Path(zip_members_dir) / member
        sources[rel[member]] = (src, zip_record(member, _sha256(src)))
    sources[rel[XLS_NAME]] = (
        Path(xls_path),
        url_record(XLS_URL, _sha256(xls_path), retrieved_at_data),
    )
    sources[rel[XML_NAME]] = (
        Path(xml_path),
        url_record(XML_URL, _sha256(xml_path), retrieved_at_paper),
    )
    sources[rel[PDF_NAME]] = (
        Path(pdf_path),
        url_record(PDF_URL, _sha256(pdf_path), retrieved_at_paper),
    )
    for name, repo_path in CODE_FILES.items():
        src = Path(code_dir) / name
        url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_COMMIT}/{repo_path}"
        sources[rel[name]] = (
            src,
            url_record(
                url,
                _sha256(src),
                retrieved_at_paper,
                repo=GITHUB_REPO,
                commit=GITHUB_COMMIT,
                path=repo_path,
            ),
        )

    files: list[ArtifactRecord] = []
    for relpath, (src, record) in sources.items():
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != record.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        role = ROLE_PAPER_PDF if relpath.endswith(".pdf") else ROLE_RAW_DATA
        files.append(
            ArtifactRecord(
                path=relpath,
                role=role,
                bytes=dest.stat().st_size,
                sha256=record.sha256,
                source=record.source_url,
                retrieval=record,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title="Rare variants contribute disproportionately to quantitative trait variation in yeast",
        files=files,
        si_data_sources=[
            DROPBOX_URL,
            f"https://github.com/{GITHUB_REPO}/tree/{GITHUB_COMMIT}",
            XLS_URL,
            "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA549760/",
        ],
        si_expected=["Figure 1 source data 1 (elife-49212-fig1-data1-v2.xls)"],
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
# Parent table (xls) + README pairs
# --------------------------------------------------------------------------- #
class CrossInfo(BaseModel):
    """One cross from the README pairs and the xls ``Crosses and Strains`` sheet."""

    cross: str
    parent_1: str
    parent_2: str
    parent_1_genotype: str
    parent_2_genotype: str
    diploid_parent: str
    n_segregants_xls: int


def read_readme_pairs(path: str | Path) -> dict[str, tuple[str, str]]:
    """``cross_genotypes_README`` table: cross -> (Parent1, Parent2) verbatim."""
    pairs: dict[str, tuple[str, str]] = {}
    in_table = False
    for line in Path(path).read_text().splitlines():
        if line.startswith("cross\tParent1"):
            in_table = True
            continue
        if in_table:
            parts = [p for p in line.rstrip("\t").split("\t") if p]
            if len(parts) != 3:
                if parts:
                    raise ValueError(f"unexpected README table line: {line!r}")
                in_table = False
                continue
            pairs[parts[0]] = (parts[1], parts[2])
    if set(pairs) != set(CROSSES):
        raise ValueError(
            f"README crosses {sorted(pairs)} != expected {sorted(CROSSES)}"
        )
    return pairs


def read_cross_table(
    xls_path: str | Path, readme_path: str | Path
) -> dict[str, CrossInfo]:
    """Join the xls cross sheet to the README pairs; every field is checked, not assumed.

    The README defines the 1/2 allele coding ("The allele coding (1,2) corresponding to
    Parent 1 or Parent 2, is indicated in the table below"), so its order is
    authoritative. The xls ``Parent 1``/``Parent 2`` columns name the same two strains
    but not always in the same order (cross 375: xls M22/BY, README BYa/M22), so the
    xls is matched as an unordered pair and each genotype string is assigned to the
    README parent it names.
    """
    df = pd.read_excel(xls_path, sheet_name="Crosses and Strains", engine="xlrd")
    pairs = read_readme_pairs(readme_path)
    out: dict[str, CrossInfo] = {}
    for _, row in df.iterrows():
        cross = str(row["name of cross used in provided code"]).strip()
        p1, p2 = pairs[cross]
        cells = [str(row["Parent 1"]).strip(), str(row["Parent 2"]).strip()]
        g1 = [c for c in cells if c.startswith(PARENT_XLS_PREFIX[p1])]
        g2 = [c for c in cells if c.startswith(PARENT_XLS_PREFIX[p2])]
        if len(g1) != 1 or len(g2) != 1 or g1[0] == g2[0]:
            raise ValueError(
                f"cross {cross}: xls parents {cells!r} do not match README {p1}/{p2}"
            )
        out[cross] = CrossInfo(
            cross=cross,
            parent_1=p1,
            parent_2=p2,
            parent_1_genotype=g1[0],
            parent_2_genotype=g2[0],
            diploid_parent=str(row["Diploid Parent"]).strip(),
            n_segregants_xls=int(row["Number of Segregants Analyzed"]),
        )
    if set(out) != set(CROSSES):
        raise ValueError("xls cross sheet does not list every README cross")
    return out


def read_assembly_index(path: str | Path) -> dict[str, str]:
    """``1011Assemblies.tar.gz.member_index.tsv``: Peter id -> tar member path."""
    index: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        sid, member = line.rstrip("\n").split("\t")
        index[sid] = member
    return index


def build_parent(
    label: str, genotype_text: str, index: dict[str, str]
) -> SegregantParent:
    """A parent pinned to its assembly; a Peter id absent from the index raises."""
    peter = PARENT_PETER_ID[label]
    if peter is None:
        return SegregantParent(
            name=label,
            peter_strain_id=None,
            assembly_member=S288C_ASSEMBLY,
            assembly_sha256="S288C reference (SGD R64-4-1); see ReferenceGenome",
            engineered_background=genotype_text,
        )
    if peter not in index:
        raise RuntimeError(
            f"parent {label}: Peter id {peter} is not in the assembly member index"
        )
    return SegregantParent(
        name=label,
        peter_strain_id=peter,
        assembly_member=f"{ASSEMBLY_TAR}::{index[peter]}",
        assembly_sha256=ASSEMBLY_TAR_SHA256,
        engineered_background=genotype_text,
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class Bloom2019Dataset(ExperimentDataset):
    """Bloom 2019: one record per (segregant, condition), 530,100 records."""

    def __init__(
        self,
        root: str = "data/torchcell/bloom2019",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """A genome is REQUIRED (the gene set is the genes the mosaics span)."""
        self.genome = genome
        self.conditions = build_conditions()
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return SegregantGrowthExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return SegregantGrowthExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The data files (phenotypes, 16 genotype matrices, README, xls); code and paper
        are read from the mirror by the verifier, not symlinked here.
        """
        return [PHENOTYPES_NAME, README_NAME, XLS_NAME] + [
            f"genotype_{c}.tsv.gz" for c in CROSSES
        ]

    def download(self) -> None:
        """Symlink the manifest-listed mirror files into ``raw/`` and verify each sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        rel = raw_relpaths()
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in self.raw_file_names:
            src = raw_mirror_dir(data_root) / rel[name]
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            expected = manifest_sha256(manifest, rel[name])
            got = _sha256(src)
            if got != expected:
                raise RuntimeError(
                    f"{name} sha256 mismatch: got {got}, expected {expected}"
                )
            dest = osp.join(self.raw_dir, name)
            if not osp.exists(dest):
                os.symlink(src, dest)
        log.info("Bloom 2019 raw files linked into %s (sha256 verified)", self.raw_dir)

    # ---- environment / phenotype builders -------------------------------------
    def _environment(self, spec: ConditionSpec) -> Environment:
        # Temperature is on Environment.temperature (M2). The Methods give the 48 h
        # incubation but not its temperature, so DEFAULT_TEMPERATURE_C is a documented
        # representative (the Nadal-Ribelles precedent), flagged for review in the
        # module docstring and the dendron note; a ProvenanceGap cannot coexist with a
        # stored value, and the adapter and verifier both read the value.
        return Environment(
            media=spec.media,
            temperature=Temperature(value=spec.temperature_c),
            perturbations=list(spec.perturbations),
            aerobicity="aerobic",
            duration_hours=INCUBATION_HOURS,
        )

    def _units(self, spec: ConditionSpec) -> str:
        if spec.measurement_type is MeasurementType.colony_size:
            return (
                "end-point colony mean radius in image pixels (s.radius.mean, "
                "process_images.R), duplicate plates averaged and missing cells "
                "mean-imputed per trait within a cross (mapping.R); the control "
                "plate of its batch"
            )
        return (
            "residual of colony mean radius regressed on the same segregant's radius "
            f"on the matched control plate {spec.control_column} (same cross, batch and "
            "layout; residuals(lm(s.radius.mean ~ ctrl.s.radius.mean)), "
            "process_images.R), duplicate plates averaged and missing cells "
            "mean-imputed per trait within a cross (mapping.R)"
        )

    def _phenotype(
        self, spec: ConditionSpec, value: float
    ) -> EnvironmentResponsePhenotype:
        return EnvironmentResponsePhenotype(
            measurement_type=spec.measurement_type,
            assay_type=AssayType.colony_size_array,
            environment_response=value,
            n_samples=N_SAMPLES,
            sample_unit=SampleUnit.technical_replicate,
            units=self._units(spec),
            provenance_gaps=[
                ProvenanceGap(
                    field="environment_response_se",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="the release carries the duplicate-averaged value only",
                )
            ],
        )

    def _reference(self, spec: ConditionSpec) -> SegregantGrowthExperimentReference:
        gaps: list[ProvenanceGap] = []
        value = 0.0
        note = "residual reference: a segregant growing exactly as predicted from its control-plate growth has residual 0"
        if spec.measurement_type is MeasurementType.colony_size:
            note = (
                "absolute colony size: the parents' radii are not released; 0 is the "
                "scale's physical zero (no colony), flagged for review"
            )
            gaps.append(
                ProvenanceGap(
                    field="environment_response_se",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note=note,
                )
            )
        control_env = self._environment(self.control_spec(spec))
        return SegregantGrowthExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="S288C", ploidy="haploid"
            ),
            environment_reference=control_env,
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=spec.measurement_type,
                assay_type=AssayType.colony_size_array,
                environment_response=value,
                units=note,
                provenance_gaps=gaps,
            ),
        )

    def control_spec(self, spec: ConditionSpec) -> ConditionSpec:
        """The served condition whose environment is the control plate's environment."""
        return self.conditions[CONTROL_ENVIRONMENT_COLUMN[spec.control_column]]

    # ---- genotypes ------------------------------------------------------------
    def _iter_cross_genotypes(
        self, cross: str, info: CrossInfo, index: dict[str, str]
    ) -> Iterator[tuple[str, SegregantGenotype]]:
        """Stream one cross's matrix as int8, sort markers, and RLE every segregant."""
        path = osp.join(self.raw_dir, f"genotype_{cross}.tsv.gz")
        frame = read_marker_matrix(path)
        markers = sorted_markers([str(c) for c in frame.columns])
        order = [m.column for m in markers]
        values = frame.to_numpy(dtype=np.int8)[:, order]
        if not np.isin(values, (1, 2)).all():
            raise ValueError(f"genotype_{cross}: calls outside {{1, 2}}")
        if values.shape[0] != EXPECTED_SEGREGANTS[cross]:
            raise ValueError(
                f"genotype_{cross}: {values.shape[0]} segregants, expected {EXPECTED_SEGREGANTS[cross]}"
            )
        matrix_sha = manifest_sha256(self._manifest, f"data/genotype_{cross}.tsv.gz")
        p1 = build_parent(info.parent_1, info.parent_1_genotype, index)
        p2 = build_parent(info.parent_2, info.parent_2_genotype, index)
        self._markers[cross] = markers
        for row_id, calls in zip(frame.index.astype(str), values):
            blocks = encode_blocks(calls, markers)
            if not np.array_equal(expand_blocks(blocks, markers), calls):
                raise ValueError(f"{cross}/{row_id}: block round trip failed")
            yield (
                row_id,
                SegregantGenotype(
                    cross=cross,
                    segregant_id=row_id,
                    parent_1=p1,
                    parent_2=p2,
                    blocks=blocks,
                    call_method=CALL_METHOD,
                    marker_matrix_sha256=matrix_sha,
                ),
            )

    # ---- build ----------------------------------------------------------------
    @post_process
    def process(self) -> None:
        """Build 13,950 x 38 records, one cross at a time, interning each genotype."""
        import lmdb  # noqa: F401  (opened via _open_write_lmdb)

        data_root = _data_root()
        self._manifest = load_manifest(data_root)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        info = read_cross_table(
            osp.join(self.raw_dir, XLS_NAME), osp.join(self.raw_dir, README_NAME)
        )
        index = read_assembly_index(osp.join(data_root, PETER_DIR_REL, ASSEMBLY_INDEX))
        # round_trip: the stored value is the exact decimal the release prints, so the
        # verifier's re-read equals it bit for bit (the default parser can differ in
        # the last digit).
        phenotypes = pd.read_csv(
            osp.join(self.raw_dir, PHENOTYPES_NAME),
            sep="\t",
            compression="gzip",
            index_col=0,
            float_precision="round_trip",
        )
        phenotypes.index = phenotypes.index.astype(str)
        columns = [str(c) for c in phenotypes.columns]
        unknown = set(columns) - set(self.conditions) - NOT_SERVED
        if unknown:
            raise ValueError(
                f"phenotype columns not in the condition table: {sorted(unknown)}"
            )
        missing = set(self.conditions) - set(columns)
        if missing:
            raise ValueError(
                f"condition table columns absent from the release: {sorted(missing)}"
            )
        if phenotypes.shape[0] != N_SEGREGANTS:
            raise ValueError(
                f"{phenotypes.shape[0]} phenotype rows, expected {N_SEGREGANTS}"
            )

        references = {
            col: self._reference(spec) for col, spec in self.conditions.items()
        }
        environments = {
            col: self._environment(spec) for col, spec in self.conditions.items()
        }
        publication = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")
        self._markers: dict[str, list[Marker]] = {}

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        seen: set[str] = set()
        block_counts: dict[str, list[int]] = {}
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for cross in CROSSES:
                counts: list[int] = []
                for seg_id, genotype in tqdm(
                    self._iter_cross_genotypes(cross, info[cross], index),
                    total=EXPECTED_SEGREGANTS[cross],
                    desc=f"cross {cross}",
                ):
                    if seg_id in seen:
                        raise ValueError(
                            f"segregant id {seg_id} appears in two crosses"
                        )
                    seen.add(seg_id)
                    counts.append(genotype.n_blocks)
                    row = phenotypes.loc[seg_id]
                    for col, spec in self.conditions.items():
                        value = float(row[col])
                        if np.isnan(value):
                            raise ValueError(f"{seg_id}/{col}: NaN in the release")
                        experiment = SegregantGrowthExperiment(
                            dataset_name=self.name,
                            genotype=genotype,
                            environment=environments[col],
                            phenotype=self._phenotype(spec, value),
                        )
                        txn.put(
                            f"{idx}".encode(),
                            self._intern_record(
                                experiment, references[col], publication, itxn
                            ),
                        )
                        idx += 1
                block_counts[cross] = counts
        env.close()
        interned_env.close()
        if idx != EXPECTED_RECORDS:
            raise ValueError(f"wrote {idx} records, expected {EXPECTED_RECORDS}")
        with open(osp.join(self.preprocess_dir, "block_counts.json"), "w") as fh:
            json.dump(block_counts, fh)
        pd.DataFrame({"segregant_id": sorted(seen)}).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )
        log.info("Wrote %d Bloom 2019 records", idx)

    def _intern_record(
        self,
        experiment: Experiment,
        reference: ExperimentReference,
        publication: Publication,
        itxn: Any,
    ) -> bytes:
        """Also intern the genotype: one mosaic per segregant, not one per condition."""
        import pickle

        rec: dict[str, Any] = {
            "experiment": experiment.model_dump(),
            "reference": reference.model_dump(),
            "publication": publication.model_dump(),
        }
        genotype = experiment.genotype
        hint = getattr(genotype, "segregant_id", "genotype")
        self._maybe_intern(rec["experiment"], "genotype", genotype, hint, itxn)
        self._maybe_intern(
            rec["experiment"],
            "environment",
            experiment.environment,
            experiment.environment.media.name,
            itxn,
        )
        self._maybe_intern(rec, "reference", reference, reference.dataset_name, itxn)
        self._maybe_intern(
            rec, "publication", publication, publication.doi or "publication", itxn
        )
        return pickle.dumps(rec)

    # ---- gene set --------------------------------------------------------------
    @staticmethod
    def extract_systematic_gene_names(genotype: dict[str, Any]) -> list[str]:
        """Not applicable: a mosaic genotype has no gene-keyed perturbations."""
        raise NotImplementedError(
            "a SegregantGenotype carries no gene perturbations; Bloom2019Dataset.compute_gene_set "
            "returns the S288C genes its haplotype blocks span"
        )

    def compute_gene_set(self) -> GeneSet:
        """S288C genes overlapping at least one haplotype block of any cross.

        Computed once from the sorted marker tables kept during ``process`` (or rebuilt
        from ``raw/`` when loading a built store), so it never walks 530k records.
        """
        if self.genome is None:
            raise RuntimeError(
                "Bloom2019Dataset requires a genome to compute its gene set"
            )
        spans = self._marker_spans()
        gene_set = GeneSet()
        db = self.genome.db
        for gene_id in self.genome.gene_set:
            feature = db[gene_id]
            for start, end in spans.get(str(feature.chrom), []):
                if feature.start <= end and feature.end >= start:
                    gene_set.add(gene_id)
                    break
        if not gene_set:
            raise ValueError("no S288C gene overlaps any haplotype block")
        return gene_set

    def _marker_spans(self) -> dict[str, list[tuple[int, int]]]:
        """Per chromosome, the (first, last) marker positions of every cross."""
        spans: dict[str, list[tuple[int, int]]] = {}
        markers_by_cross = getattr(self, "_markers", None) or {}
        if not markers_by_cross:
            for cross in CROSSES:
                markers_by_cross[cross] = sorted_markers(
                    read_marker_names(
                        osp.join(self.raw_dir, f"genotype_{cross}.tsv.gz")
                    )
                )
        for markers in markers_by_cross.values():
            for chrom in _CHROM_NAMES:
                positions = [m.position for m in markers if m.chromosome == chrom]
                if positions:
                    spans.setdefault(chrom, []).append((min(positions), max(positions)))
        return spans

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing happens inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Records are assembled inside process(); see _iter_cross_genotypes."""
        raise NotImplementedError("Bloom2019Dataset builds its records in process()")


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    data_root = _data_root()
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    dataset = Bloom2019Dataset(
        root=osp.join(data_root, "data/torchcell/bloom2019"), genome=genome
    )
    print(f"len = {len(dataset)}")
    record = dataset[0]
    genotype = record["experiment"]["genotype"]
    print(
        "cross",
        genotype["cross"],
        "segregant",
        genotype["segregant_id"],
        "blocks",
        len(genotype["blocks"]),
    )
    print("phenotype", record["experiment"]["phenotype"]["environment_response"])


if __name__ == "__main__":
    main()
