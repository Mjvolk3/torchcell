# tests/torchcell/datasets/scerevisiae/test_wildenhain2015.py
# [[tests.torchcell.datasets.scerevisiae.test_wildenhain2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_wildenhain2015.py
"""Wildenhain 2015 loader: screen counting, the drop rule, the category mapping.

The end-to-end tests run on a synthetic AID-format CSV and a fake genome, so no build
tree and no network are needed. The audit test binds each module-level ``SourcedValue``
to its verbatim quote in the sha256-pinned mirror it names (the AID description lives in
the raw mirror, the paper OCR in the library mirror) and skips when they are absent.

2026.09.30 (Phase 12): the two-screen SD and SE are now exact (``stdev([-4.0, -6.0])`` is
``math.sqrt(2)`` to the bit and the schema's SE is sd / sqrt(2) = 1.0). Added: full
``model_dump()`` equality for the two-screen record (idx 2, YJR066W / vanillin, mean
-5.0, SD sqrt(2), n 2) and the single-screen edge record (idx 0, YAL001C / vanillin, its
two typed dispersion gaps) with their shared reference; the side files (records sorted by
(ORF, identity) give idx 0 YAL001C/CID 1183, 1 YAL001C/CID 702, 2 YJR066W/CID 1183,
3 YJR066W/CID 702; the reference index groups by compound, [[0, 2], [1, 3]]); the logged
counts (8 strain datapoints, 1 non-strain row, 5 cells; one cell whose every screen is
flagged non-replicate), including a blank line and a strain row with a blank z_score,
neither counted. Refusals with exact messages: an off-genome ORF, two identities with one
canonical name, and every ``download`` path (no manifest, a file missing from the mirror,
a digest mismatch naming the manifest's digest). ``deposit_raw_mirror`` is pinned on
synthetic files with the pins monkeypatched to their digests, and one build runs through
``download`` from that mirror.

2026.10.01 (issue #520): the datapoint key is the parsed z, so two z strings of equal
value (``-4.0`` / ``-4.00``) are one screen (n 1) rather than an SD-0 abort, and a missing
mirror manifest refuses with the deposit step instead of a bare ``FileNotFoundError``. A
non-finite or unparseable z refuses in ``_collapse_matrix`` naming the cell.

2026.10.02 (issue #504): the records are the strain-resolved family; the fixture's
``NULL / wild type`` row is now a served wild-type record (empty genotype, idx 0, so the
other indices shift by one); one reference (no compound, BY4741 background) serves every
record; the 33 essential-gene strains are pinned and served as conditional alleles with
typed gaps; ``NA/NNK1`` lands on YKL171W and TSCII / YGL11 / wtn01 are held under
``strain_label_unresolved``; the environment states the 96-well culture and 1.96 % v/v
DMSO; every phenotype carries a ``screen_id`` gap for the per-library normalization.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import math
import os
import os.path as osp
from pathlib import Path
from typing import Any

import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.identity import media_identity
from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    Compound,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    CultureFormat,
    EndpointRule,
    EnvironmentResponsePhenotype,
    Genotype,
    MatingType,
    MeasurementType,
    PreCulture,
    PreCultureSource,
    Publication,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    Temperature,
    UncertaintyType,
)
from torchcell.datamodels.strain_background import BRACHMANN_1998, GIAEVER_2002
from torchcell.datasets.scerevisiae import wildenhain2015 as w
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

_GENES = {"YJR066W", "YAL001C"}
_STANDARD = {"TOR1": ["YJR066W"], "TFC3": ["YAL001C"]}

_HEADER = [
    "PUBCHEM_RESULT_TAG", "PUBCHEM_SID", "PUBCHEM_CID",
    "PUBCHEM_EXT_DATASOURCE_SMILES", "PUBCHEM_ACTIVITY_OUTCOME",
    "PUBCHEM_ACTIVITY_SCORE", "PUBCHEM_ACTIVITY_URL", "PUBCHEM_ASSAYDATA_COMMENT",
    "orf", "sym", "raw OD read 1", "raw OD read 2", "normalized OD average",
    "z_score", "p_value", "non replicate", "cryptagen", "bioactivity",
]  # fmt: skip

_VANILLIN = "COC1=C(C=CC(=C1)C=O)O"


class _Resolution:
    def __init__(self, status: str, systematic: str | None) -> None:
        self.status = status
        self.systematic_name = systematic

    @property
    def is_current_gene(self) -> bool:
        return self.status in ("current", "renamed")


class _FakeGenome:
    """The slice of ``SCerevisiaeGenome`` the loader's gene-name policy reads."""

    gene_set = _GENES
    feature_index = {"standard_to_ids": _STANDARD}

    def resolve_gene_name(self, name: str) -> _Resolution:
        upper = name.upper()
        if upper in _GENES:
            return _Resolution("current", upper)
        for standard, ids in _STANDARD.items():
            if upper == standard:
                return _Resolution("renamed", ids[0])
        return _Resolution("retired", upper)


def _row(**kwargs: str) -> list[str]:
    row = dict.fromkeys(_HEADER, "")
    row.update(kwargs)
    return [row[name] for name in _HEADER]


def _rows() -> list[list[str]]:
    """One cell per scenario: the re-export duplicate, two screens, an SID-only compound."""
    return [
        # the SAME datapoint re-exported under two gene-symbol spellings -> ONE screen
        _row(
            PUBCHEM_SID="1",
            PUBCHEM_CID="1183",
            PUBCHEM_EXT_DATASOURCE_SMILES=_VANILLIN,
            PUBCHEM_ACTIVITY_OUTCOME="Inactive",
            orf="YAL001C",
            sym="TFC3",
            z_score="-0.5",
            **{"non replicate": "0"},
        ),
        _row(
            PUBCHEM_SID="1",
            PUBCHEM_CID="1183",
            PUBCHEM_EXT_DATASOURCE_SMILES=_VANILLIN,
            PUBCHEM_ACTIVITY_OUTCOME="Inactive",
            orf="YAL001C",
            sym="tfc3",
            z_score="-0.5",
            **{"non replicate": "0"},
        ),
        # two genuinely distinct screens of one cell -> n_samples 2 + a sample SD
        _row(
            PUBCHEM_SID="1",
            PUBCHEM_CID="1183",
            PUBCHEM_EXT_DATASOURCE_SMILES=_VANILLIN,
            PUBCHEM_ACTIVITY_OUTCOME="Active",
            orf="YJR066W",
            sym="TOR1",
            z_score="-4.0",
            bioactivity="sensitive",
            **{"non replicate": "0"},
        ),
        _row(
            PUBCHEM_SID="2",
            PUBCHEM_CID="1183",
            PUBCHEM_EXT_DATASOURCE_SMILES=_VANILLIN,
            PUBCHEM_ACTIVITY_OUTCOME="Active",
            orf="YJR066W",
            sym="Tor1",
            z_score="-6.0",
            bioactivity="sensitive",
            **{"non replicate": "1"},
        ),
        # PubChem's own no-call verdict
        _row(
            PUBCHEM_SID="3",
            PUBCHEM_CID="702",
            PUBCHEM_EXT_DATASOURCE_SMILES="CCO",
            PUBCHEM_ACTIVITY_OUTCOME="Inconclusive",
            orf="YJR066W",
            sym="TOR1",
            z_score="5.3",
            bioactivity="resistant",
            **{"non replicate": "0"},
        ),
        # screens that disagree on the released call
        _row(
            PUBCHEM_SID="4",
            PUBCHEM_CID="702",
            PUBCHEM_EXT_DATASOURCE_SMILES="CCO",
            PUBCHEM_ACTIVITY_OUTCOME="Active",
            orf="YAL001C",
            sym="TFC3",
            z_score="-4.2",
            bioactivity="sensitive",
            **{"non replicate": "0"},
        ),
        _row(
            PUBCHEM_SID="4",
            PUBCHEM_CID="702",
            PUBCHEM_EXT_DATASOURCE_SMILES="CCO",
            PUBCHEM_ACTIVITY_OUTCOME="Inactive",
            orf="YAL001C",
            sym="TFC3",
            z_score="-0.2",
            **{"non replicate": "0"},
        ),
        # no CID and no SMILES -> the compound cannot be encoded, cell dropped
        _row(
            PUBCHEM_SID="99",
            PUBCHEM_ACTIVITY_OUTCOME="Inactive",
            orf="YAL001C",
            sym="TFC3",
            z_score="0.1",
            **{"non replicate": "0"},
        ),
        # the BY4741 wild-type screen (#504): served as the EMPTY genotype
        _row(
            PUBCHEM_SID="1",
            PUBCHEM_CID="1183",
            PUBCHEM_ACTIVITY_OUTCOME="Inactive",
            orf="NULL",
            sym="wild type",
            z_score="-2.0",
            **{"non replicate": "0"},
        ),
    ]


@pytest.fixture()
def built(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build the synthetic dataset end to end into a temporary root."""
    monkeypatch.setattr(w, "default_genome", lambda: _FakeGenome())
    monkeypatch.setattr(
        w.EnvChemgenWildenhain2015Dataset, "download", lambda self: None
    )
    root = str(tmp_path / "env_chemgen_wildenhain2015")
    os.makedirs(osp.join(root, "raw"), exist_ok=True)
    with gzip.open(osp.join(root, "raw", w.DATA_FILENAME), "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        writer.writerow(["RESULT_TYPE"] * len(_HEADER))
        writer.writerows(_rows())
    with open(osp.join(root, "raw", w.AID_FILENAME), "w") as handle:
        json.dump({"placeholder": True}, handle)
    return w.EnvChemgenWildenhain2015Dataset(root=root)


def _by_key(dataset: Any) -> dict[tuple[str, str], dict[str, Any]]:
    """Records keyed by (screened ORF or ``wild type``, compound name)."""
    out = {}
    for i in range(len(dataset)):
        record = dataset[i]
        experiment = record["experiment"]
        perturbations = experiment["genotype"]["perturbations"]
        orf = perturbations[0]["systematic_gene_name"] if perturbations else "wild type"
        compound = experiment["environment"]["perturbations"][0]["compound"]["name"]
        out[(orf, compound)] = record
    return out


def _screen_gap() -> ProvenanceGap:
    return ProvenanceGap(
        field="screen_id",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="the release has no library, screen or plate column; screens were "
        "LOWESS- or DMSO-control-normalized by library (Sci Data lines 63-64), so a "
        "cell's normalization is not recoverable",
    )


def test_drop_rules_hold_unresolved_strains_then_unidentifiable_compounds(
    built: Any,
) -> None:
    """The fixture has no unresolved strain label, so rule 1 holds 0; rule 2 drops the
    SID-only compound's cell. 6 cells (5 ORF cells + the wild-type cell), 5 kept.
    """
    log = w.DropLog.model_validate_json(
        open(osp.join(built.root, "preprocess", "dropped_records.json")).read()
    )
    assert [rule.rule for rule in log.rules] == [
        "strain_label_unresolved",
        "compound_without_a_structure_identifier",
    ]
    assert log.rules[0].scope == "strain"
    assert log.rules[0].n_records == 0 and log.rules[0].items == []
    assert log.rules[1].items == ["SID 99"]
    assert log.rules[1].n_records == 1
    assert log.source_records == 6 and log.kept_records == 5
    assert len(built) == 5


def test_re_exported_duplicate_is_one_screen_not_two(built: Any) -> None:
    phenotype = _by_key(built)[("YAL001C", "vanillin")]["experiment"]["phenotype"]
    assert phenotype["n_samples"] == 1
    assert phenotype["sample_unit"] == SampleUnit.screen
    assert phenotype["environment_response"] == -0.5
    # n=1 has no dispersion, and that absence is typed rather than a silent None
    assert phenotype["environment_response_uncertainty"] is None
    assert [gap["field"] for gap in phenotype["provenance_gaps"]] == [
        "environment_response_uncertainty",
        "environment_response_se",
        "screen_id",
    ]


def test_two_screens_average_and_carry_a_sample_sd(built: Any) -> None:
    phenotype = _by_key(built)[("YJR066W", "vanillin")]["experiment"]["phenotype"]
    assert phenotype["n_samples"] == 2
    assert phenotype["environment_response"] == -5.0
    assert phenotype["environment_response_uncertainty"] == math.sqrt(2)
    assert (
        phenotype["environment_response_uncertainty_type"] == UncertaintyType.sample_sd
    )
    assert phenotype["environment_response_se"] == 1.0
    # the per-library normalization is the one typed gap a multi-screen cell carries
    assert phenotype["provenance_gaps"] == [_screen_gap().model_dump()]


def test_released_call_maps_onto_the_shared_category_axis(built: Any) -> None:
    records = _by_key(built)
    inactive = records[("YAL001C", "vanillin")]["experiment"]["phenotype"]
    assert inactive["category"] == ResponseCategory.no_change
    assert inactive["category_label"] == "Inactive"
    active = records[("YJR066W", "vanillin")]["experiment"]["phenotype"]
    assert active["category"] == ResponseCategory.sensitive
    assert active["category_label"] == "Active / sensitive"
    inconclusive = records[("YJR066W", "ethanol")]["experiment"]["phenotype"]
    assert inconclusive["category"] == ResponseCategory.not_determined
    assert inconclusive["category_label"] == "Inconclusive / resistant"
    disagree = records[("YAL001C", "ethanol")]["experiment"]["phenotype"]
    assert disagree["category"] == ResponseCategory.not_determined
    assert disagree["category_label"] == "Active / Inactive / sensitive"


def test_deletion_is_kanmx4_from_euroscarf_with_a_typed_barcode_gap(built: Any) -> None:
    deletion = _by_key(built)[("YJR066W", "vanillin")]["experiment"]["genotype"][
        "perturbations"
    ][0]
    assert deletion["perturbation_type"] == "barcoded_kanmx_deletion"
    assert deletion["perturbed_gene_name"] == "TOR1"  # the release also spells it Tor1
    assert deletion["collection"] == "Euroscarf deletion collection"
    assert deletion["cassette"] == "kanMX4"  # Giaever 2014, KANMX4_CASSETTE
    assert deletion["barcode"] is None  # the release publishes no barcode
    assert [gap["field"] for gap in deletion["provenance_gaps"]] == ["barcode"]
    assert (
        deletion["provenance_gaps"][0]["reason"]
        == ProvenanceGapReason.deferred_pending_source_review
    )


def test_wild_type_screen_is_the_empty_genotype_on_by4741(built: Any) -> None:
    """Contract (#504 finding 3): the released ``NULL / wild type`` rows are the BY4741
    screen, served with no perturbation; the background is the reference genome's.
    """
    record = _by_key(built)[("wild type", "vanillin")]
    assert record["experiment"]["genotype"] == {"perturbations": []}
    assert record["experiment"]["phenotype"]["environment_response"] == -2.0
    assert record["reference"]["genome_reference"]["strain"] == "BY4741"


def test_environment_is_a_static_96_well_sc_culture_at_20um_in_1_96pct_dmso(
    built: Any,
) -> None:
    environment = _by_key(built)[("YJR066W", "vanillin")]["experiment"]["environment"]
    assert environment["media"] == w.WILDENHAIN_SC.model_dump()
    assert environment["temperature"]["value"] == 30.0
    assert environment["duration_hours"] == 18.0
    assert [gap["field"] for gap in environment["provenance_gaps"]] == [
        "duration_generations",
        "auxotroph_supplements",
    ]
    culture = environment["culture_format"]
    assert {
        name: culture[name]
        for name in (
            "vessel",
            "working_volume_ul",
            "shaking_rpm",
            "inoculum_cells",
            "endpoint",
        )
    } == {
        "vessel": "96-well plate",
        "working_volume_ul": 100.0,
        "shaking_rpm": 0.0,
        "inoculum_cells": 50000.0,
        "endpoint": EndpointRule.until_control_saturation,
    }
    assert environment["pre_culture"]["source"] == PreCultureSource.overnight_culture
    assert [gap["field"] for gap in environment["pre_culture"]["provenance_gaps"]] == [
        "medium"
    ]
    perturbation = environment["perturbations"][0]
    assert perturbation["concentration"] == {"value": 20.0, "unit": "uM", "basis": None}
    assert perturbation["compound"]["inchikey"] == "MWOOGOJBHIARFG-UHFFFAOYSA-N"
    assert perturbation["compound"]["pubchem_cid"] == 1183
    assert perturbation["solvent"]["name"] == "DMSO"
    assert perturbation["solvent"]["compound"]["inchikey"] is not None
    assert perturbation["solvent"]["percent"] == 1.96


def test_dmso_fraction_is_2ul_into_a_100ul_culture() -> None:
    """Contract (#504 finding 7): 100 x 2 / (100 + 2) = 1.9608 -> 1.96 % v/v, from the
    two quoted volumes, not a hand-typed number.
    """
    assert w.DMSO_WORKING_STOCK_UL.value == 2.0
    assert w.WORKING_VOLUME_UL.value == 100.0
    assert w.DMSO_PERCENT_V_V == round(100.0 * 2.0 / 102.0, 2) == 1.96
    assert w.SOLVENT_PERCENT.value == 1.96


def test_wildenhain_sc_joins_the_shared_sc_and_drops_the_fungal_sentence() -> None:
    """The shared SC is imported by five loaders, so it is untouched; the local copy has
    the same composition (one ``media_identity``) and only the CGM's own medium quote.
    """
    assert media_identity(w.WILDENHAIN_SC) == media_identity(SC)
    assert w.WILDENHAIN_SC.base_medium == "SC"
    quotes = [sv.quote for sv in w.WILDENHAIN_SC.provenance] + [
        sv.quote for c in w.WILDENHAIN_SC.components for sv in c.provenance
    ]
    assert not any("fungal" in quote for quote in quotes)
    glucose = next(
        c for c in w.WILDENHAIN_SC.components if c.compound.name == "D-glucose"
    )
    assert [sv.provenance.citation_key for sv in glucose.provenance] == [
        "wildenhainSystematicChemicalgeneticChemicalchemical2016"
    ]


def test_reference_is_the_strains_own_center_without_compound(built: Any) -> None:
    """Contract (#504 finding 5): ONE reference, z = 0 at the screen center, in the
    culture with no compound, on the typed BY4741 background; its units say the baseline
    is the same strain, not BY4741 under the compound.
    """
    record = _by_key(built)[("YJR066W", "vanillin")]
    reference = record["reference"]
    assert reference["experiment_reference_type"] == "strain_environment_response"
    assert reference["phenotype_reference"]["environment_response"] == 0.0
    assert reference["environment_reference"]["perturbations"] == []
    environment = dict(record["experiment"]["environment"])
    environment["perturbations"] = []
    assert reference["environment_reference"] == environment
    units = reference["phenotype_reference"]["units"]
    assert "SAME strain's own screen center" in units
    assert "NOT BY4741 under the compound" in units
    assert reference == _by_key(built)[("wild type", "vanillin")]["reference"]


def test_background_is_by4741_sourced_to_sci_data_with_pending_constructions() -> None:
    genome = w.BY4741_GENOME
    background = genome.background
    assert (genome.strain, genome.ploidy) == ("BY4741", "haploid")
    assert background.mating_type == MatingType.a
    assert [a.allele_name for a in background.alleles] == [
        "his3Δ1",
        "leu2Δ0",
        "met15Δ0",
        "ura3Δ0",
    ]
    assert background.provenance == [w.BY4741_GENOTYPE]
    assert "MATa his3Δ1 leu2Δ0 met15Δ0 ura3Δ0" in w.BY4741_GENOTYPE.quote
    for allele in background.alleles:
        assert allele.provenance == [w.BY4741_GENOTYPE]
        assert [(g.field, g.resolve_with) for g in allele.provenance_gaps] == [
            ("deleted_span", BRACHMANN_1998)
        ]


def test_z_definition_is_the_n1_iqr_rule_and_units_say_iqr() -> None:
    assert "N ( 1 , I Q R )" in w.Z_SCORE_DEFINITION.quote
    assert "kernel density" not in w.Z_SCORE_DEFINITION.quote
    assert w.Z_SCORE_DEFINITION.provenance.citation_key == w.SCIDATA_CITATION_KEY
    assert "N(1, IQR) fit" in w.MEASUREMENT_UNITS
    assert "not unit variance" in w.MEASUREMENT_UNITS


def test_every_sourced_value_is_backed_by_a_verbatim_quote_in_its_mirror() -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT not set")
    library = osp.join(data_root, "torchcell-library")
    raw = osp.join(data_root, "torchcell-raw")
    if not all(
        osp.isdir(path)
        for path in (
            osp.join(library, w.CITATION_KEY),
            osp.join(library, w.SCIDATA_CITATION_KEY),
            osp.join(raw, w.CITATION_KEY),
        )
    ):
        pytest.skip("mirrors not mounted")
    values = [
        getattr(w, name)
        for name in dir(w)
        if isinstance(getattr(w, name), SourcedValue)
    ]
    assert len(values) >= 25
    for value in values:
        root = raw if value.provenance.source_uri.startswith("data/") else library
        assert audit_sourced_value(value, root).passed, value.quote


# ---- #504: essential-gene strains, non-ORF strain labels ------------------------- #
_ESSENTIAL_33 = {
    "YAR019C": "CDC15", "YBL105C": "PKC1", "YBR135W": "CKS1", "YBR136W": "MEC1",
    "YBR160W": "CDC28", "YDL017W": "CDC7", "YDL028C": "MPS1", "YDL108W": "KIN28",
    "YDL132W": "CDC53", "YDR052C": "DBF4", "YDR054C": "CDC34", "YER133W": "GLC7",
    "YFL009W": "CDC4", "YFL029C": "CAK1", "YFR003C": "YPI1", "YFR028C": "CDC14",
    "YIL147C": "SLN1", "YJR016C": "ILV3", "YKL193C": "SDS22", "YKL203C": "TOR2",
    "YMR001C": "CDC5", "YMR277W": "FCP1", "YNL006W": "LST8", "YNL161W": "CBK1",
    "YNL207W": "RIO2", "YNL222W": "SSU72", "YOL078W": "AVO1", "YOR119C": "RIO1",
    "YOR329C": "SCD5", "YPL153C": "RAD53", "YPL204W": "HRR25", "YPL209C": "IPL1",
    "YPR025C": "CCL1",
}  # fmt: skip


def test_the_33_essential_gene_strains_are_pinned() -> None:
    """Contract (#504 finding 1): exactly these 33 released ORFs (CDC28, TOR2, IPL1,
    RAD53, ...) are SGD-essential and are never served as kanMX nulls.
    """
    assert w.ESSENTIAL_GENE_ORFS == frozenset(_ESSENTIAL_33)
    assert len(w.ESSENTIAL_GENE_ORFS) == 33


def test_the_essential_set_is_the_intersection_with_the_sgd_essential_genes() -> None:
    """Recompute the pin from the two files it was derived from, when they are present."""
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT not set")
    gene_set = Path(data_root) / w.ESSENTIAL_GENE_SET_SOURCE.source_uri
    export = w.raw_mirror_dir(data_root) / w.DATA_REL
    if not gene_set.exists() or not export.exists():
        pytest.skip("essentiality build or raw mirror not present")
    essential = set(json.loads(gene_set.read_text()))
    with gzip.open(export, "rt", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        next(reader)
        orf_at = header.index("orf")
        orfs = {
            row[orf_at].strip()
            for row in reader
            if row and w._SYSTEMATIC_RE.match(row[orf_at].strip())
        }
    assert len(orfs) == 242
    assert orfs & essential == set(w.ESSENTIAL_GENE_ORFS)


class _PanelGenome(_FakeGenome):
    """The fake genome plus CDC28 (essential) and NNK1."""

    gene_set = _GENES | {"YBR160W", "YKL171W"}
    feature_index = {
        "standard_to_ids": {**_STANDARD, "CDC28": ["YBR160W"], "NNK1": ["YKL171W"]}
    }

    def resolve_gene_name(self, name: str) -> _Resolution:
        upper = name.upper()
        if upper in self.gene_set:
            return _Resolution("current", upper)
        ids = self.feature_index["standard_to_ids"].get(upper)
        if ids is not None:
            return _Resolution("renamed", ids[0])
        return _Resolution("retired", upper)


def _panel_row(orf: str, sym: str, sid: str, cid: str, z: str) -> list[str]:
    return _row(
        PUBCHEM_SID=sid,
        PUBCHEM_CID=cid,
        PUBCHEM_ACTIVITY_OUTCOME="Inactive",
        orf=orf,
        sym=sym,
        z_score=z,
        **{"non replicate": "0"},
    )


def _panel_rows() -> list[list[str]]:
    return [
        _panel_row("YBR160W", "CDC28", "1", "1183", "-1.0"),
        _panel_row("YKL171W", "NNK1", "2", "1183", "0.5"),
        _panel_row("NA", "NNK1", "3", "702", "-0.7"),
        _panel_row("NA", "TSCII", "4", "1183", "-0.3"),
        _panel_row("NA", "TSCII", "5", "702", "-0.4"),
        _panel_row("NULL", "YGL11", "6", "702", "0.2"),
        _panel_row("NULL", "wtn01", "7", "702", "0.1"),
    ]


def test_essential_gene_strain_is_a_conditional_allele_with_typed_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (#504 finding 1): CDC28 (YBR160W) is emitted, not dropped, as a
    ``ConditionalAllelePerturbation`` whose class and collection are pending-review gaps
    naming the strain tables, never as a kanMX null.
    """
    _write_raw(tmp_path, _panel_rows())
    dataset = _build(tmp_path, monkeypatch, _PanelGenome())
    records = _by_key(dataset)
    perturbation = records[("YBR160W", "vanillin")]["experiment"]["genotype"][
        "perturbations"
    ]
    assert len(perturbation) == 1
    conditional = perturbation[0]
    assert conditional["perturbation_type"] == "conditional_allele"
    assert conditional["perturbed_gene_name"] == "CDC28"
    assert conditional["allele_class"] is None
    assert conditional["collection"] is None
    assert [
        (g["field"], g["reason"], g["resolve_with"]["source_uri"])
        for g in conditional["provenance_gaps"]
    ] == [
        (
            field,
            ProvenanceGapReason.deferred_pending_source_review,
            "Sci Data 2016 Table 1 (available online only); Cell Systems 2015 Table S3",
        )
        for field in ("allele_class", "collection")
    ]
    nnk1 = records[("YKL171W", "vanillin")]["experiment"]["genotype"]["perturbations"]
    assert nnk1[0]["perturbation_type"] == "barcoded_kanmx_deletion"


def test_nnk1_rows_are_served_on_ykl171w_and_unresolved_labels_are_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (#504 finding 3): ``NA/NNK1`` lands on YKL171W (its own cell here, a
    compound the ORF-labelled rows do not cover); TSCII, YGL11 and wtn01 are held in
    the ledger under ``strain_label_unresolved`` with their cell and row counts.
    """
    _write_raw(tmp_path, _panel_rows())
    dataset = _build(tmp_path, monkeypatch, _PanelGenome())
    records = _by_key(dataset)
    assert sorted(records) == [
        ("YBR160W", "vanillin"),
        ("YKL171W", "ethanol"),
        ("YKL171W", "vanillin"),
    ]
    assert records[("YKL171W", "ethanol")]["experiment"]["phenotype"][
        "environment_response"
    ] == pytest.approx(-0.7)
    log = w.DropLog.model_validate_json(
        (tmp_path / "preprocess" / "dropped_records.json").read_text()
    )
    held = log.rules[0]
    assert held.rule == "strain_label_unresolved"
    assert held.n_records == 4
    assert held.items == [
        "NA/TSCII (2 cells, 2 rows)",
        "NULL/YGL11 (1 cells, 1 rows)",
        "NULL/wtn01 (1 cells, 1 rows)",
    ]
    assert log.source_records == 7 and log.kept_records == 3


def test_an_unlisted_non_orf_label_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_raw(tmp_path, [_panel_row("NA", "XYZ1", "1", "1183", "0.0")])
    with pytest.raises(RuntimeError) as info:
        _build(tmp_path, monkeypatch, _PanelGenome())
    assert str(info.value) == (
        "released row with orf='NA' sym='XYZ1' is neither a systematic ORF nor a "
        "listed NON_ORF_STRAIN_LABELS pair; a new strain label needs an explicit "
        "disposition, not a silent skip"
    )


def test_a_mapped_label_the_genome_does_not_resolve_to_its_orf_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the base fake genome, NNK1 resolves ``retired``, so the mapping refuses."""
    _write_raw(tmp_path, [_panel_row("NA", "NNK1", "1", "1183", "0.0")])
    with pytest.raises(RuntimeError) as info:
        _build(tmp_path, monkeypatch)
    assert str(info.value) == (
        "strain label NA/NNK1 is mapped to YKL171W, but the genome resolves 'NNK1' to "
        "'NNK1'"
    )


# ---- Full records, side files and logged counts (2026.09.30; #504 2026.10.02) ---- #
_NAME = "EnvChemgenWildenhain2015Dataset"


def _environment(compound: Compound | None) -> CultureEnvironment:
    """The hand-built culture environment; ``None`` is the no-compound reference."""
    perturbations: list[Any] = (
        []
        if compound is None
        else [
            SmallMoleculePerturbation(
                compound=compound,
                concentration=Concentration(
                    value=20.0, unit=ConcentrationUnit.micromolar
                ),
                solvent=Solvent(
                    name="DMSO",
                    percent=1.96,
                    compound=resolved_compound("dimethyl sulfoxide"),
                ),
            )
        ]
    )
    return CultureEnvironment(
        media=w.WILDENHAIN_SC,
        temperature=Temperature(value=30.0),
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_hours=18.0,
        culture_format=CultureFormat(
            vessel="96-well plate",
            working_volume_ul=100.0,
            shaking_rpm=0.0,
            inoculum_cells=50000.0,
            endpoint=EndpointRule.until_control_saturation,
            provenance=[
                w.CULTURE_VESSEL,
                w.WORKING_VOLUME_UL,
                w.STATIC_INCUBATION_RPM,
                w.INOCULUM_CELLS,
                w.ENDPOINT,
            ],
        ),
        pre_culture=PreCulture(
            source=PreCultureSource.overnight_culture,
            provenance=[w.PRE_CULTURE_SOURCE],
            provenance_gaps=[
                ProvenanceGap(
                    field="medium",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="'fresh overnight cultures' (Sci Data line 54); the overnight "
                    "medium is not stated",
                )
            ],
        ),
        provenance_gaps=[
            ProvenanceGap(
                field="duration_generations",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="an ~18 h liquid OD growth to control saturation doses exposure in "
                "hours, not doublings, and neither paper nor the AID protocol reports a "
                "doubling count",
            ),
            ProvenanceGap(
                field="auxotroph_supplements",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no supplement is named beside the medium: SC is complete, so the "
                "His/Leu/Met/Ura the BY4741 background needs are part of the medium's "
                "(shared SC) composition, which the lab does not state",
            ),
        ],
    )


def _vanillin() -> Compound:
    return resolved_compound("CID 1183", pubchem_cid=1183, smiles=_VANILLIN)


def _reference() -> dict[str, Any]:
    return StrainEnvironmentResponseExperimentReference(
        dataset_name=_NAME,
        genome_reference=w.BY4741_GENOME,
        environment_reference=_environment(None),
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            assay_type=AssayType.liquid_od_growth,
            environment_response=0.0,
            units=w.REFERENCE_UNITS,
        ),
    ).model_dump()


def _genotype(orf: str, common: str) -> Genotype:
    return Genotype(
        perturbations=[
            BarcodedKanMxDeletionPerturbation(
                systematic_gene_name=orf,
                perturbed_gene_name=common,
                collection="Euroscarf deletion collection",
                cassette="kanMX4",
                provenance_gaps=[
                    ProvenanceGap(
                        field="barcode",
                        reason=ProvenanceGapReason.deferred_pending_source_review,
                        resolve_with=GIAEVER_2002,
                        note="the release publishes no UPTAG/DNTAG; the YKO barcode "
                        "table gives them by ORF",
                    )
                ],
            )
        ]
    )


def test_two_screen_record_equals_the_hand_built_experiment(built: Any) -> None:
    """Record 3 = YJR066W / vanillin (record 0 is the wild-type screen, which sorts
    first): screens -4.0 and -6.0 give mean -5.0, sample SD sqrt(2), SE 1.0 (derived by
    the schema), n_samples 2 screens, ``Active / sensitive``; the release spells the gene
    ``Tor1`` once, the record stores TOR1.
    """
    environment = _environment(_vanillin())
    expected = StrainEnvironmentResponseExperiment(
        dataset_name=_NAME,
        genotype=_genotype("YJR066W", "TOR1"),
        environment=environment,
        phenotype=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            assay_type=AssayType.liquid_od_growth,
            environment_response=-5.0,
            category=ResponseCategory.sensitive,
            category_label="Active / sensitive",
            n_samples=2,
            sample_unit=SampleUnit.screen,
            units=w.MEASUREMENT_UNITS,
            environment_response_uncertainty=math.sqrt(2),
            environment_response_uncertainty_type=UncertaintyType.sample_sd,
            provenance_gaps=[_screen_gap()],
        ),
    ).model_dump()
    assert built[3]["experiment"] == expected
    assert built[3]["experiment"]["phenotype"]["environment_response_se"] == 1.0
    assert built[3]["reference"] == _reference()
    assert (
        built[3]["publication"]
        == Publication(
            doi="10.1016/j.cels.2015.12.003",
            doi_url="https://doi.org/10.1016/j.cels.2015.12.003",
        ).model_dump()
    )


def test_single_screen_record_equals_the_hand_built_experiment(built: Any) -> None:
    """Record 1 = YAL001C / vanillin: the re-exported duplicate collapses to one screen
    (z -0.5, ``Inactive``, n 1), so both dispersion fields carry the typed gap.
    """
    environment = _environment(_vanillin())
    note = (
        "one released screen for this (strain, compound) cell; a dispersion across "
        "screens is undefined at n=1 and the release carries no per-screen error"
    )
    expected = StrainEnvironmentResponseExperiment(
        dataset_name=_NAME,
        genotype=_genotype("YAL001C", "TFC3"),
        environment=environment,
        phenotype=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            assay_type=AssayType.liquid_od_growth,
            environment_response=-0.5,
            category=ResponseCategory.no_change,
            category_label="Inactive",
            n_samples=1,
            sample_unit=SampleUnit.screen,
            units=w.MEASUREMENT_UNITS,
            provenance_gaps=[
                *(
                    ProvenanceGap(
                        field=field,
                        reason=ProvenanceGapReason.not_reported_by_primary,
                        note=note,
                    )
                    for field in (
                        "environment_response_uncertainty",
                        "environment_response_se",
                    )
                ),
                _screen_gap(),
            ],
        ),
    ).model_dump()
    assert built[1]["experiment"] == expected
    assert built[1]["reference"] == _reference()


def test_side_files_hold_one_reference_for_every_record(built: Any) -> None:
    """The reference no longer carries the compound, so all five records share one."""
    preprocess = Path(built.preprocess_dir)
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YJR066W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2, 3, 4]]
    assert index[0]["reference"]["environment_reference"]["perturbations"] == []
    assert not (preprocess / "data.csv").exists()
    assert built.experiment_class is StrainEnvironmentResponseExperiment
    assert built.reference_class is StrainEnvironmentResponseExperimentReference


def _write_raw(root: Path, rows: list[list[str]], *, blank_line: bool = False) -> None:
    (root / "raw").mkdir(parents=True, exist_ok=True)
    with gzip.open(root / "raw" / w.DATA_FILENAME, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        writer.writerow(["RESULT_TYPE"] * len(_HEADER))
        writer.writerows(rows)
        if blank_line:
            handle.write("\r\n")
    (root / "raw" / w.AID_FILENAME).write_text("{}")


def _build(root: Path, monkeypatch: pytest.MonkeyPatch, genome: Any = None) -> Any:
    monkeypatch.setattr(w, "default_genome", lambda: genome or _FakeGenome())
    monkeypatch.setattr(
        w.EnvChemgenWildenhain2015Dataset, "download", lambda self: None
    )
    return w.EnvChemgenWildenhain2015Dataset(root=str(root))


def test_logged_counts_skip_blank_lines_and_blank_z_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Build (a): the fixture rows plus a strain row with an empty ``z_score`` and a
    trailing blank line. Neither is a datapoint, so the counts stay 9 datapoints (the
    nine fixture rows, one of them the wild-type screen's non-ORF label) and 6 cells;
    5 records are written (4 kanMX deletions + the wild type), 1 dropped for the SID-only
    compound, and no cell has every screen flagged ``non replicate`` (YJR066W / vanillin
    has one flagged screen of two). Build (b): the fixture with that cell's other screen
    flagged too, so the closing count is 1.
    """
    caplog.set_level(logging.INFO, logger=w.log.name)
    rows = [*_rows(), _row(PUBCHEM_SID="5", PUBCHEM_CID="702", orf="YJR066W")]
    _write_raw(tmp_path / "a", rows, blank_line=True)
    assert len(_build(tmp_path / "a", monkeypatch)) == 5
    messages = [r.getMessage() for r in caplog.records if r.name == w.log.name]
    assert messages[0] == (
        "Wildenhain2015: 9 datapoints (1 on a non-ORF strain label) -> 6 (strain, "
        "compound) cells"
    )
    assert messages[-1] == (
        "Wrote 5 Wildenhain2015 records (4 kanMX deletion, 0 conditional allele, 1 wild "
        "type; 0 held for an unresolved strain label; 1 dropped for an unidentifiable "
        "compound; 0 cells whose every screen is non-replicate flagged)"
    )
    caplog.clear()
    flagged = _rows()
    flagged[2][_HEADER.index("non replicate")] = "1"
    _write_raw(tmp_path / "b", flagged)
    _build(tmp_path / "b", monkeypatch)
    messages = [r.getMessage() for r in caplog.records if r.name == w.log.name]
    assert messages[-1] == (
        "Wrote 5 Wildenhain2015 records (4 kanMX deletion, 0 conditional allele, 1 wild "
        "type; 0 held for an unresolved strain label; 1 dropped for an unidentifiable "
        "compound; 1 cells whose every screen is non-replicate flagged)"
    )


def test_an_off_genome_orf_refuses_with_the_count_and_the_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """YDR001C is regex-valid but not in the fake genome (resolves ``retired``)."""
    rows = [*_rows(), _row(PUBCHEM_CID="702", orf="YDR001C", z_score="1.5")]
    _write_raw(tmp_path, rows)
    with pytest.raises(RuntimeError) as info:
        _build(tmp_path, monkeypatch)
    assert str(info.value) == (
        "1 released ORFs are not current R64 genes: ['YDR001C']; the release measured "
        "242 current genes when this loader was written, so a new drop rule is needed, "
        "not a silent skip"
    )


def _one_name_for_every_compound(self: Any, cell: w.MatrixCell) -> Compound:
    """A compound resolver that gives every identity the same canonical name."""
    return Compound(name="same", pubchem_cid=cell.pubchem_cid)


def test_two_identities_with_one_canonical_name_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the compound resolver returning one name for CID 1183 and CID 702, the two
    conditions would share a record key, so the build refuses.
    """
    _write_raw(tmp_path, _rows())
    monkeypatch.setattr(
        w.EnvChemgenWildenhain2015Dataset, "_compound", _one_name_for_every_compound
    )
    with pytest.raises(RuntimeError) as info:
        _build(tmp_path, monkeypatch)
    assert str(info.value) == (
        "two distinct compound identities resolve to the same canonical name, which "
        "would merge two conditions into one record key"
    )


def test_two_z_strings_of_equal_value_are_one_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #520): the datapoint key is the PARSED z, so ``-4.0`` and
    ``-4.00`` in one cell are the same released datapoint: one screen, z -4.0, n 1, and
    the single-screen dispersion gaps, instead of two screens with SD 0 that aborted the
    build. The pinned export has 0 cells with two z strings of equal value.
    """
    rows = [
        _row(
            PUBCHEM_CID="1183",
            PUBCHEM_ACTIVITY_OUTCOME="Inactive",
            orf="YAL001C",
            z_score=z,
            **{"non replicate": "0"},
        )
        for z in ("-4.0", "-4.00")
    ]
    _write_raw(tmp_path, rows)
    dataset = _build(tmp_path, monkeypatch)
    assert len(dataset) == 1
    phenotype = dataset[0]["experiment"]["phenotype"]
    assert phenotype["environment_response"] == -4.0
    assert phenotype["n_samples"] == 1
    assert [gap["field"] for gap in phenotype["provenance_gaps"]] == [
        "environment_response_uncertainty",
        "environment_response_se",
        "screen_id",
    ]


@pytest.mark.parametrize(
    ("z_scores", "message"),
    [
        (("nan", "nan"), "YAL001C/CID 1183: z_score 'nan' is not finite"),
        (("-4.0", "inf"), "YAL001C/CID 1183: z_score 'inf' is not finite"),
        (("-4.0", "n/a"), "YAL001C/CID 1183: z_score 'n/a' is not a number"),
    ],
)
def test_a_non_finite_or_unparseable_z_refuses_naming_the_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    z_scores: tuple[str, str],
    message: str,
) -> None:
    """Contract (issue #520 review): ``nan != nan``, so two ``nan`` rows would be two
    float keys of one cell, and an unparseable z would be a bare ``ValueError``. Both
    refuse in ``_collapse_matrix``, naming the cell and the raw string, before the store
    is opened. The pinned export has 0 of either.
    """
    rows = [
        _row(
            PUBCHEM_CID="1183",
            PUBCHEM_ACTIVITY_OUTCOME="Inactive",
            orf="YAL001C",
            z_score=z,
            **{"non replicate": "0"},
        )
        for z in z_scores
    ]
    _write_raw(tmp_path, rows)
    with pytest.raises(RuntimeError) as info:
        _build(tmp_path, monkeypatch)
    assert str(info.value) == message
    assert not (tmp_path / "processed" / "lmdb").exists()


def test_canonical_names_skip_standards_that_do_not_resolve_back() -> None:
    """``OLD1`` resolves ``retired`` and is skipped; ``TOR1`` and its second standard
    ``TORX`` both name YJR066W, and the first listed wins.
    """

    class _Genome(_FakeGenome):
        feature_index = {
            "standard_to_ids": {
                "TOR1": ["YJR066W"],
                "TORX": ["YJR066W"],
                "OLD1": ["YOL999W"],
                "TFC3": ["YAL001C"],
            }
        }

        def resolve_gene_name(self, name: str) -> _Resolution:
            if name == "TORX":
                return _Resolution("renamed", "YJR066W")
            return super().resolve_gene_name(name)

    genome: Any = _Genome()
    assert w._canonical_common_names(genome) == {"YJR066W": "TOR1", "YAL001C": "TFC3"}


# ---- Raw mirror, manifest, download (2026.09.30) --------------------------------- #
_CSV_BYTES = b"synthetic csv.gz bytes"
_AID_BYTES = b'{"synthetic": true}'


def _pin(monkeypatch: pytest.MonkeyPatch, csv_bytes: bytes, aid_bytes: bytes) -> None:
    monkeypatch.setattr(w, "DATA_SHA256", hashlib.sha256(csv_bytes).hexdigest())
    monkeypatch.setattr(w, "AID_SHA256", hashlib.sha256(aid_bytes).hexdigest())


def _sources(tmp_path: Path, csv_bytes: bytes, aid_bytes: bytes) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "1159580.csv.gz").write_bytes(csv_bytes)
    (source / "aid.json").write_bytes(aid_bytes)
    return source / "1159580.csv.gz", source / "aid.json"


def test_mirror_paths_and_relpaths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "/env/root")
    assert w.raw_mirror_dir("/given") == Path(
        "/given/torchcell-raw/wildenhainPredictionSynergismChemicalGenetic2015"
    )
    assert w.raw_mirror_dir() == Path(
        "/env/root/torchcell-raw/wildenhainPredictionSynergismChemicalGenetic2015"
    )
    assert w.raw_relpaths() == {
        "1159580.csv.gz": "data/1159580.csv.gz",
        "aid_1159580_description.json": "data/aid_1159580_description.json",
    }


def test_deposit_refuses_a_source_with_the_wrong_digest(tmp_path: Path) -> None:
    csv_path, aid_path = _sources(tmp_path, b"wrong", _AID_BYTES)
    with pytest.raises(RawSha256MismatchError) as info:
        w.deposit_raw_mirror(
            csv_path=csv_path, aid_path=aid_path, data_root=str(tmp_path / "dr")
        )
    got = hashlib.sha256(b"wrong").hexdigest()
    assert str(info.value) == (
        f"sha256 mismatch for {csv_path}: expected {w.DATA_SHA256}, observed {got}"
    )
    assert not (tmp_path / "dr").exists()


def test_deposit_refuses_a_later_source_before_writing_an_earlier_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both sources are verified before either is written: with the datapoint export
    on its (repointed) pin and the AID description off its pin, the refusal names the
    description and no mirror directory exists.
    """
    _pin(monkeypatch, _CSV_BYTES, _AID_BYTES)
    csv_path, aid_path = _sources(tmp_path, _CSV_BYTES, b"{}")
    with pytest.raises(RawSha256MismatchError) as info:
        w.deposit_raw_mirror(
            csv_path=csv_path, aid_path=aid_path, data_root=str(tmp_path / "dr")
        )
    assert str(info.value) == (
        f"sha256 mismatch for {aid_path}: expected "
        f"{hashlib.sha256(_AID_BYTES).hexdigest()}, "
        f"observed {hashlib.sha256(b'{}').hexdigest()}"
    )
    assert not (tmp_path / "dr").exists()


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): with both files already in ``raw/`` PyG skips
    ``download()``, so ``process()`` verifies them against ``DATA_SHA256`` and
    ``AID_SHA256`` first. The datapoint export off its pin raises
    ``RawSha256MismatchError`` naming it and both digests; no store is written.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(w, [w.DATA_FILENAME, w.AID_FILENAME])
    raw = staged.root / "raw" / w.DATA_FILENAME
    with pytest.raises(RawSha256MismatchError) as info:
        w.EnvChemgenWildenhain2015Dataset(root=str(staged.root))
    assert str(info.value) == (
        f"sha256 mismatch for {raw}: expected "
        "c461c679b63ac56045cef0f03ed9bcbb8e7f9c12146f1fc7cc8ac0c113188d64, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


def test_deposit_refuses_an_existing_mirror_file_with_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, _CSV_BYTES, _AID_BYTES)
    csv_path, aid_path = _sources(tmp_path, _CSV_BYTES, _AID_BYTES)
    dest = (
        w.raw_mirror_dir(str(tmp_path / "dr")) / "data" / "aid_1159580_description.json"
    )
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"{}")
    with pytest.raises(RuntimeError) as info:
        w.deposit_raw_mirror(
            csv_path=csv_path, aid_path=aid_path, data_root=str(tmp_path / "dr")
        )
    assert str(info.value) == f"{dest} exists with a different sha256; refusing"
    # the csv (processed first) was already copied before the AID file refused
    assert (dest.parent / "1159580.csv.gz").read_bytes() == _CSV_BYTES


def test_deposit_writes_both_files_and_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The csv's retrieval records the FTP container, its member and the container's
    sha256; the AID's is a PUG-REST GET. ``load_manifest`` reads it back and
    ``manifest_sha256`` answers per path, with a ``KeyError`` for an unlisted one.
    """
    _pin(monkeypatch, _CSV_BYTES, _AID_BYTES)
    csv_path, aid_path = _sources(tmp_path, _CSV_BYTES, _AID_BYTES)
    root = w.deposit_raw_mirror(
        csv_path=csv_path,
        aid_path=aid_path,
        retrieved_at="2026-09-13",
        data_root=str(tmp_path / "dr"),
    )
    assert root == w.raw_mirror_dir(str(tmp_path / "dr"))
    assert (root / "data" / "1159580.csv.gz").read_bytes() == _CSV_BYTES
    assert (root / "data" / "aid_1159580_description.json").read_bytes() == _AID_BYTES
    manifest = w.load_manifest(str(tmp_path / "dr"))
    ftp = "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/1159001_1160000.zip"
    aid = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1159580/description/JSON"
    csv_sha = hashlib.sha256(_CSV_BYTES).hexdigest()
    aid_sha = hashlib.sha256(_AID_BYTES).hexdigest()
    assert manifest == Manifest(
        citation_key="wildenhainPredictionSynergismChemicalGenetic2015",
        doi="10.1016/j.cels.2015.12.003",
        title="Prediction of Synergism from Chemical-Genetic Interactions by Machine "
        "Learning",
        files=[
            ArtifactRecord(
                path="data/1159580.csv.gz",
                role=ROLE_RAW_DATA,
                bytes=len(_CSV_BYTES),
                sha256=csv_sha,
                source=ftp,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=ftp,
                    retriever="torchcell.literature.retrieve.zip_member",
                    params={
                        "url": ftp,
                        "member": "1159001_1160000/1159580.csv.gz",
                        "container_sha256": "d1fd5dc2bf7c526ad9845e0a14ae9981256fb82"
                        "0aaf4228b48a3ba0724ee59b0",
                    },
                    sha256=csv_sha,
                    retrieved_at="2026-09-13",
                ),
            ),
            ArtifactRecord(
                path="data/aid_1159580_description.json",
                role=ROLE_RAW_DATA,
                bytes=len(_AID_BYTES),
                sha256=aid_sha,
                source=aid,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.pubchem_api,
                    source_url=aid,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": aid},
                    sha256=aid_sha,
                    retrieved_at="2026-09-13",
                ),
            ),
        ],
        si_data_sources=["https://pubchem.ncbi.nlm.nih.gov/bioassay/1159580", ftp, aid],
        si_expected=[
            "Cell Systems 2015 Tables S1/S2 (the four compound libraries) and Table S3 "
            "(the 195 original sentinel strains), and Sci Data 2016 Table 1 (all 242 "
            "sentinels of the extended CGM this AID releases; 'the number of sentinels "
            "has been increased from 195 to 242') -- not mirrored (cell.com and "
            "nature.com supplements are not scriptable), so the per-strain allele, "
            "collection and accession, which would resolve the 33 essential-gene "
            "strains and the TSCII / YGL11 / wtn01 labels, are typed gaps"
        ],
        provenance_complete=True,
        created_at=manifest.created_at,
    )
    assert w.manifest_sha256(manifest, "data/1159580.csv.gz") == csv_sha
    with pytest.raises(KeyError) as info:
        w.manifest_sha256(manifest, "data/other.csv")
    assert info.value.args == ("data/other.csv is not in the raw-mirror manifest",)
    # idempotent: the files are already at their pins, so a second deposit succeeds
    assert (
        w.deposit_raw_mirror(
            csv_path=csv_path, aid_path=aid_path, data_root=str(tmp_path / "dr")
        )
        == root
    )


def _bare(root: Path) -> Any:
    dataset = w.EnvChemgenWildenhain2015Dataset.__new__(
        w.EnvChemgenWildenhain2015Dataset
    )
    dataset.root = str(root)
    return dataset


def test_download_without_a_manifest_refuses_naming_the_deposit_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #520): with no mirror deposited, ``download`` refuses in
    ``load_manifest`` with the manifest path and the deposit step, not a bare
    ``FileNotFoundError``.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    with pytest.raises(RuntimeError) as info:
        _bare(tmp_path / "ds").download()
    manifest = w.raw_mirror_dir() / "manifest.json"
    assert str(info.value) == (
        f"raw-mirror manifest missing: {manifest}. Deposit the mirror with "
        "deposit_raw_mirror() first."
    )


def test_download_refuses_a_missing_file_and_a_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The expected digest comes from the MANIFEST (here the synthetic pins), not the
    module constant.
    """
    _pin(monkeypatch, _CSV_BYTES, _AID_BYTES)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    csv_path, aid_path = _sources(tmp_path, _CSV_BYTES, _AID_BYTES)
    mirror = w.deposit_raw_mirror(csv_path=csv_path, aid_path=aid_path)
    aid_mirror = mirror / "data" / "aid_1159580_description.json"
    aid_mirror.unlink()
    with pytest.raises(RuntimeError) as info:
        _bare(tmp_path / "ds").download()
    assert str(info.value) == f"required raw artifact missing from mirror: {aid_mirror}"
    aid_mirror.write_bytes(b"{}")
    with pytest.raises(RawSha256MismatchError) as info:
        _bare(tmp_path / "ds").download()
    got = hashlib.sha256(b"{}").hexdigest()
    assert str(info.value) == (
        f"sha256 mismatch for {aid_mirror}: expected "
        f"{hashlib.sha256(_AID_BYTES).hexdigest()}, observed {got}"
    )
    assert not (tmp_path / "ds" / "raw" / "aid_1159580_description.json").exists()


def test_build_links_the_mirror_into_raw_then_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``raw/``: ``download`` symlinks both verified mirror files, and the build then
    reads the linked csv (the ``_rows`` fixture, five records).
    """
    raw_source = tmp_path / "gz"
    _write_raw(raw_source, _rows())
    csv_bytes = (raw_source / "raw" / w.DATA_FILENAME).read_bytes()
    _pin(monkeypatch, csv_bytes, _AID_BYTES)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    csv_path, aid_path = _sources(tmp_path, csv_bytes, _AID_BYTES)
    mirror = w.deposit_raw_mirror(csv_path=csv_path, aid_path=aid_path)
    monkeypatch.setattr(w, "default_genome", lambda: _FakeGenome())
    root = tmp_path / "env_chemgen_wildenhain2015"
    dataset = w.EnvChemgenWildenhain2015Dataset(root=str(root))
    assert len(dataset) == 5
    for name, rel in w.raw_relpaths().items():
        assert os.readlink(root / "raw" / name) == str(mirror / rel)
    # a second download leaves the existing links in place
    dataset.download()
    assert os.readlink(root / "raw" / w.DATA_FILENAME) == str(
        mirror / "data" / "1159580.csv.gz"
    )


def test_inline_construction_hooks_are_inert(tmp_path: Path) -> None:
    dataset = _bare(tmp_path)
    frame = object()
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()


# Phase 24: the drop-accounting guard


def test_a_drop_log_that_disagrees_with_its_rules_refuses_after_writing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``built`` fixture rows with ``DropLog`` reporting one extra dropped record.

    The fixture holds 0 unresolved strains and drops 1 SID-only compound cell, so the
    rule total is 0 + 1 = 1 while the tampered log says 2: the exact mismatch message
    (this loader says ``rule total``, singular), the tampered 2 already on disk.
    """
    real = w.DropLog

    def inflated(**kwargs: Any) -> Any:
        kwargs["dropped_records"] += 1
        return real(**kwargs)

    monkeypatch.setattr(w, "DropLog", inflated)
    root = tmp_path / "env_chemgen_wildenhain2015"
    _write_raw(root, _rows())
    with pytest.raises(RuntimeError) as err:
        _build(root, monkeypatch)
    assert str(err.value) == (
        "drop accounting mismatch: rule total 1, 2 records missing from the build"
    )
    log = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert log["dropped_records"] == 2
