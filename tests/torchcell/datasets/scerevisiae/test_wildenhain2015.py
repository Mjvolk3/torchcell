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
from torchcell.datamodels.media import MEDIA_LIBRARY, SC
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    MeasurementType,
    Publication,
    ReferenceGenome,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
    UncertaintyType,
)
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
    assert phenotype["environment_response_uncertainty"] == math.sqrt(2)
    assert (
        phenotype["environment_response_uncertainty_type"] == UncertaintyType.sample_sd
    )
    assert phenotype["environment_response_se"] == 1.0
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


# ---- Full records, side files and logged counts (2026.09.30) --------------------- #
_NAME = "EnvChemgenWildenhain2015Dataset"


def _environment(compound: Compound) -> Environment:
    return Environment(
        media=SC,
        temperature=Temperature(value=30.0),
        perturbations=[
            SmallMoleculePerturbation(
                compound=compound,
                concentration=Concentration(
                    value=20.0, unit=ConcentrationUnit.micromolar
                ),
                solvent=Solvent(
                    name="DMSO", compound=resolved_compound("dimethyl sulfoxide")
                ),
            )
        ],
        aerobicity="aerobic",
        duration_hours=18.0,
        provenance_gaps=[
            ProvenanceGap(
                field="duration_generations",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="an 18 h liquid OD growth to saturation doses exposure in hours, "
                "not doublings, and neither the paper nor the AID protocol reports a "
                "doubling count",
            )
        ],
    )


def _vanillin() -> Compound:
    return resolved_compound("CID 1183", pubchem_cid=1183, smiles=_VANILLIN)


def _reference(environment: Environment) -> dict[str, Any]:
    return EnvironmentResponseExperimentReference(
        dataset_name=_NAME,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=environment,
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            assay_type=AssayType.liquid_od_growth,
            environment_response=0.0,
            units=w.MEASUREMENT_UNITS
            + "; the reference 0 is the screen's own normalized-growth center by "
            "construction of the z-score, NOT a measured wild-type value",
        ),
    ).model_dump()


def _genotype(orf: str, common: str) -> Genotype:
    return Genotype(
        perturbations=[
            BarcodedKanMxDeletionPerturbation(
                systematic_gene_name=orf,
                perturbed_gene_name=common,
                collection="Euroscarf deletion collection",
            )
        ]
    )


def test_two_screen_record_equals_the_hand_built_experiment(built: Any) -> None:
    """Record 2 = YJR066W / vanillin: screens -4.0 and -6.0 give mean -5.0, sample SD
    sqrt(2), SE 1.0 (derived by the schema), n_samples 2 screens, ``Active /
    sensitive``; the release spells the gene ``Tor1`` once, the record stores TOR1.
    """
    environment = _environment(_vanillin())
    expected = EnvironmentResponseExperiment(
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
        ),
    ).model_dump()
    assert built[2]["experiment"] == expected
    assert built[2]["experiment"]["phenotype"]["environment_response_se"] == 1.0
    assert built[2]["reference"] == _reference(environment)
    assert (
        built[2]["publication"]
        == Publication(
            doi="10.1016/j.cels.2015.12.003",
            doi_url="https://doi.org/10.1016/j.cels.2015.12.003",
        ).model_dump()
    )


def test_single_screen_record_equals_the_hand_built_experiment(built: Any) -> None:
    """Record 0 = YAL001C / vanillin: the re-exported duplicate collapses to one screen
    (z -0.5, ``Inactive``, n 1), so both dispersion fields carry the typed gap.
    """
    environment = _environment(_vanillin())
    note = (
        "one released screen for this (strain, compound) cell; a dispersion across "
        "screens is undefined at n=1 and the release carries no per-screen error"
    )
    expected = EnvironmentResponseExperiment(
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
                ProvenanceGap(
                    field=field,
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note=note,
                )
                for field in (
                    "environment_response_uncertainty",
                    "environment_response_se",
                )
            ],
        ),
    ).model_dump()
    assert built[0]["experiment"] == expected
    assert built[0]["reference"] == _reference(environment)


def test_side_files_group_the_reference_index_by_compound(built: Any) -> None:
    preprocess = Path(built.preprocess_dir)
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YJR066W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 2], [1, 3]]
    compounds = [
        entry["reference"]["environment_reference"]["perturbations"][0]["compound"][
            "pubchem_cid"
        ]
        for entry in index
    ]
    assert compounds == [1183, 702]
    assert not (preprocess / "data.csv").exists()
    assert built.experiment_class is EnvironmentResponseExperiment
    assert built.reference_class is EnvironmentResponseExperimentReference


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
    trailing blank line. Neither is a datapoint nor a non-strain row, so the counts stay
    8 strain datapoints (nine fixture rows minus the NULL control), 1 non-strain row and
    5 cells, and no cell has every screen flagged ``non replicate`` (YJR066W / vanillin
    has one flagged screen of two). Build (b): the fixture with that cell's other screen
    flagged too, so the closing count is 1.
    """
    caplog.set_level(logging.INFO, logger=w.log.name)
    rows = [*_rows(), _row(PUBCHEM_SID="5", PUBCHEM_CID="702", orf="YJR066W")]
    _write_raw(tmp_path / "a", rows, blank_line=True)
    assert len(_build(tmp_path / "a", monkeypatch)) == 4
    messages = [r.getMessage() for r in caplog.records if r.name == w.log.name]
    assert messages[0] == (
        "Wildenhain2015: 8 strain datapoints (1 non-strain control rows) -> "
        "5 (ORF, compound) cells"
    )
    assert messages[-1] == (
        "Wrote 4 Wildenhain2015 records (1 dropped for an unidentifiable compound; 1 "
        "non-strain rows ignored; 0 cells whose every screen is non-replicate flagged)"
    )
    caplog.clear()
    flagged = _rows()
    flagged[2][_HEADER.index("non replicate")] = "1"
    _write_raw(tmp_path / "b", flagged)
    _build(tmp_path / "b", monkeypatch)
    messages = [r.getMessage() for r in caplog.records if r.name == w.log.name]
    assert messages[-1] == (
        "Wrote 4 Wildenhain2015 records (1 dropped for an unidentifiable compound; 1 "
        "non-strain rows ignored; 1 cells whose every screen is non-replicate flagged)"
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
            "Tables S1/S2 (the four compound libraries) and Table S3 (the 195 sentinel "
            "strains) -- cell.com supplementary files are not scriptable, so they are "
            "NOT mirrored and the 195-vs-242 strain split cannot be reconstructed"
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
    reads the linked csv (the ``_rows`` fixture, four records).
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
    assert len(dataset) == 4
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
