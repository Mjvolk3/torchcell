# tests/torchcell/datasets/scerevisiae/test_wildenhain2015.py
"""Wildenhain 2015 loader: screen counting, the drop rule, the category mapping.

The end-to-end tests run on a synthetic AID-format CSV and a fake genome, so no build
tree and no network are needed. The audit test binds each module-level ``SourcedValue``
to its verbatim quote in the sha256-pinned mirror it names (the AID description lives in
the raw mirror, the paper OCR in the library mirror) and skips when they are absent.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import os.path as osp
from typing import Any

import pytest

from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import ResponseCategory, SampleUnit, UncertaintyType
from torchcell.datasets.scerevisiae import wildenhain2015 as w
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

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
        # non-strain control rows are ignored
        _row(
            PUBCHEM_SID="1",
            PUBCHEM_CID="1183",
            orf="NULL",
            sym="wild type",
            z_score="-2.0",
        ),
    ]


@pytest.fixture()
def built(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build the synthetic dataset end to end into a temporary root."""
    import csv

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
    out = {}
    for i in range(len(dataset)):
        record = dataset[i]
        experiment = record["experiment"]
        orf = experiment["genotype"]["perturbations"][0]["systematic_gene_name"]
        compound = experiment["environment"]["perturbations"][0]["compound"]["name"]
        out[(orf, compound)] = record
    return out


def test_only_the_unidentifiable_compound_is_dropped(built: Any) -> None:
    log = w.DropLog.model_validate_json(
        open(osp.join(built.root, "preprocess", "dropped_records.json")).read()
    )
    assert [rule.rule for rule in log.rules] == [
        "compound_without_a_structure_identifier"
    ]
    assert log.rules[0].items == ["SID 99"]
    assert log.rules[0].n_records == 1
    assert log.source_records == 5 and log.kept_records == 4
    assert len(built) == 4


def test_re_exported_duplicate_is_one_screen_not_two(built: Any) -> None:
    phenotype = _by_key(built)[("YAL001C", "vanillin")]["experiment"]["phenotype"]
    assert phenotype["n_samples"] == 1
    assert phenotype["sample_unit"] == SampleUnit.screen
    assert phenotype["environment_response"] == -0.5
    # n=1 has no dispersion, and that absence is typed rather than a silent None
    assert phenotype["environment_response_uncertainty"] is None
    assert {gap["field"] for gap in phenotype["provenance_gaps"]} == {
        "environment_response_uncertainty",
        "environment_response_se",
    }


def test_two_screens_average_and_carry_a_sample_sd(built: Any) -> None:
    phenotype = _by_key(built)[("YJR066W", "vanillin")]["experiment"]["phenotype"]
    assert phenotype["n_samples"] == 2
    assert phenotype["environment_response"] == -5.0
    assert phenotype["environment_response_uncertainty"] == pytest.approx(math.sqrt(2))
    assert (
        phenotype["environment_response_uncertainty_type"] == UncertaintyType.sample_sd
    )
    assert phenotype["environment_response_se"] == pytest.approx(1.0)
    assert phenotype["provenance_gaps"] == []


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


def test_gene_name_is_the_genome_spelling_not_the_release_casing(built: Any) -> None:
    deletion = _by_key(built)[("YJR066W", "vanillin")]["experiment"]["genotype"][
        "perturbations"
    ][0]
    assert deletion["perturbation_type"] == "barcoded_kanmx_deletion"
    assert deletion["perturbed_gene_name"] == "TOR1"  # the release also spells it Tor1
    assert deletion["collection"] == w.COLLECTION.value
    assert deletion["barcode"] is None  # the release publishes no barcode


def test_environment_is_the_shared_sc_at_20um_in_dmso(built: Any) -> None:
    environment = _by_key(built)[("YJR066W", "vanillin")]["experiment"]["environment"]
    assert environment["media"] == MEDIA_LIBRARY["SC"].model_dump()
    assert environment["temperature"]["value"] == 30.0
    assert environment["duration_hours"] == 18.0
    assert [gap["field"] for gap in environment["provenance_gaps"]] == [
        "duration_generations"
    ]
    perturbation = environment["perturbations"][0]
    assert perturbation["concentration"] == {"value": 20.0, "unit": "uM", "basis": None}
    assert perturbation["compound"]["inchikey"] == "MWOOGOJBHIARFG-UHFFFAOYSA-N"
    assert perturbation["compound"]["pubchem_cid"] == 1183
    assert perturbation["solvent"]["name"] == "DMSO"
    assert perturbation["solvent"]["compound"]["inchikey"] is not None
    assert perturbation["solvent"]["percent"] is None


def test_reference_is_the_screen_center_in_the_same_environment(built: Any) -> None:
    record = _by_key(built)[("YJR066W", "vanillin")]
    reference = record["reference"]
    assert reference["phenotype_reference"]["environment_response"] == 0.0
    assert reference["genome_reference"]["strain"] == "BY4741"
    assert reference["environment_reference"] == record["experiment"]["environment"]
    assert "normalized-growth center" in reference["phenotype_reference"]["units"]


def test_every_sourced_value_is_backed_by_a_verbatim_quote_in_its_mirror() -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT not set")
    library = osp.join(data_root, "torchcell-library")
    raw = osp.join(data_root, "torchcell-raw")
    if not osp.isdir(osp.join(library, w.CITATION_KEY)) or not osp.isdir(
        osp.join(raw, w.CITATION_KEY)
    ):
        pytest.skip("mirrors not mounted")
    values = [
        getattr(w, name)
        for name in dir(w)
        if isinstance(getattr(w, name), SourcedValue)
    ]
    assert len(values) >= 12
    for value in values:
        root = raw if value.provenance.source_uri.startswith("data/") else library
        assert audit_sourced_value(value, root).passed
