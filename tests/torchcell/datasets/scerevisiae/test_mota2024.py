# tests/torchcell/datasets/scerevisiae/test_mota2024.py
# [[tests.torchcell.datasets.scerevisiae.test_mota2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_mota2024.py
"""Mota 2024 loader: pH as a typed edit, the ordinal grade, and released-token storage.

The three spreadsheets are written synthetically with openpyxl, so the parser and the
placement by token form run for real without the raw mirror and without a genome.

A second, richer build (``_EDGE_ROWS``): each sheet opens with a title row and a blank
row before the ``Gene/ORF name`` header, so the header search is exercised.

    acid      row (token, score)      stored as (#1045)                 counted
    acetic    TFC3 +                  released symbol TFC3              yes
              EFG1 +, YGR272C ++      two records: the symbol EFG1 and  yes, yes
                                      the systematic name YGR272C
              RLM2 ++                 released symbol RLM2              yes
              YDL001W (blank score)   skipped                           no
              (blank token) +         skipped                           no
              YDL001W 0               the reference grade, skipped      no
              nbsp-only token +       empty after strip, skipped        no
    butyric   ALIASX +, YBR001C +     two records                       yes, yes
              RLM2 +, SBR2 +          two records                       yes, yes
    octanoic  YDL001W ++              kept                              yes
              RLM2 +                  kept                              yes

Records in sheet order: 0 TFC3 + (acetic), 1 EFG1 +, 2 YGR272C ++, 3 RLM2 ++, 4 ALIASX
+ (butyric), 5 YBR001C +, 6 RLM2 +, 7 SBR2 +, 8 YDL001W ++ (octanoic), 9 RLM2 +. The
ledger counts 10 rows, 10 records, 0 dropped, 3 under a systematic name and no repeated
token. Each acid has its own reference (the acid rides on the reference plate), so the
reference index is [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]. Also pinned: the download order
(mirror, then the ESM URL), the sha256 refusals with both digests, ``deposit_raw_mirror``
and its manifest, the missing-header refusal.

2026.10.01 (issue #520): a score outside ``0``, ``+``, ``++`` (``+++`` on the octanoic
sheet) refuses with a message naming the file, the symbol and the token, and a sheet
without the header row refuses with a message naming the file instead of a bare
``StopIteration``. All three sheets are parsed before ``processed/lmdb`` is opened, so
either refusal leaves no store and a retry refuses again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import openpyxl
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YPD_AGAR
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    ReleasedGeneKind,
    ReleasedGeneNamePerturbation,
    ResponseCategory,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.scerevisiae import mota2024 as m
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    SourceCheck,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

# (token, score) rows per acid; the header row the parser keys on is written first.
_ROWS: dict[str, list[tuple[str, str]]] = {
    "acetic": [("YAL001C", "+"), ("EFG1", "++"), ("YGR272C", "++"), ("RLM2", "+")],
    "butyric": [("YAL001C", "++"), ("RNR4", "+"), ("RNR4\xa0", "+")],
    "octanoic": [("EFG1", "+"), ("YGR272C", "++")],
}


def _write_sheets(raw: Path) -> None:
    for spec in m._ACID_SPECS:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["Gene/ORF name", "Encoded Protein Function", "Growth inhibition"])
        for token, score in _ROWS[spec["acid"]]:
            sheet.append([token, "synthetic", score])
        workbook.save(raw / spec["filename"])


@pytest.fixture
def built(tmp_path: Path) -> m.EnvChemgenMota2024Dataset:
    """A tiny end-to-end build over the synthetic spreadsheets."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_sheets(raw)
    return m.EnvChemgenMota2024Dataset(root=str(tmp_path))


def test_ph_is_a_typed_perturbation_and_the_medium_is_the_shared_object() -> None:
    spec = next(s for s in m._ACID_SPECS if s["acid"] == "acetic")
    dataset = m.EnvChemgenMota2024Dataset.__new__(m.EnvChemgenMota2024Dataset)
    environment = dataset._environment(spec)
    assert environment.media is YPD_AGAR
    assert environment.media.base_medium == "YPD"
    assert environment.media.name is not None and "pH" not in environment.media.name
    acid_entry, ph_entry = environment.perturbations
    acid = cast(SmallMoleculePerturbation, acid_entry)
    ph = cast(EnvironmentPhysicalPerturbation, ph_entry)
    assert acid.compound.chebi_id == "CHEBI:15366"
    assert acid.concentration.value == 75.0
    assert ph.factor is PhysicalFactor.ph
    assert ph.magnitude is not None
    assert ph.magnitude.unit is not None
    assert (ph.magnitude.value, ph.magnitude.unit.value) == (4.5, "pH")
    assert ph.agent is not None
    assert ph.agent.name == "hydrochloric acid"
    assert ph.agent.inchikey == "VEXZGXHMUGYJMC-UHFFFAOYSA-N"
    assert environment.duration_hours == 48.0


def test_duration_rule_is_the_scoring_anchor_not_the_photograph_range() -> None:
    assert m._DURATION.value == 48.0
    assert "$4 8 \\ \\mathrm { h }$ of incubation" in m._DURATION.quote
    assert "36-48 h" in (m._DURATION.note or "")


def test_the_ordinal_grade_is_stored_with_its_typed_call(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    grades = {}
    for i in range(len(built)):
        phenotype = built[i]["experiment"]["phenotype"]
        assert phenotype["measurement_type"] is MeasurementType.ordinal
        assert phenotype["assay_type"] is AssayType.spot_dilution
        grades[phenotype["category_label"]] = (
            phenotype["environment_response"],
            phenotype["category"],
        )
    assert grades == {
        "+": (1.0, ResponseCategory.reduced),
        "++": (2.0, ResponseCategory.severely_reduced),
    }
    reference = built[0]["reference"]["phenotype_reference"]
    assert reference["environment_response"] == 0.0
    assert reference["category"] is ResponseCategory.no_change
    assert reference["category_label"] == "0"


def test_unreported_replicate_design_is_a_typed_gap_not_a_silent_none(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    phenotype = built[0]["experiment"]["phenotype"]
    assert phenotype["n_samples"] is None and phenotype["sample_unit"] is None
    gapped = {gap["field"] for gap in phenotype["provenance_gaps"]}
    assert gapped == {"n_samples", "sample_unit"}
    reasons = {str(gap["reason"]) for gap in phenotype["provenance_gaps"]}
    assert reasons == {"not_reported_by_primary"}


def test_two_source_names_for_one_gene_are_two_records(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    """EFG1 and YGR272C (one gene since SGD merged the ORFs) are two released rows and
    so two records in every table they share; nothing is merged and no grade wins.
    """
    rows = {
        (
            p["systematic_gene_name"],
            built[i]["experiment"]["phenotype"]["category_label"],
        )
        for i in range(len(built))
        for p in built[i]["experiment"]["genotype"]["perturbations"]
    }
    assert {("EFG1", "++"), ("YGR272C", "++"), ("EFG1", "+")} <= rows
    log = m.DropLog.model_validate_json(
        Path(built.root, m._DROPPED_FILENAME).read_text()
    )
    assert log.rule == m.RETENTION_RULE and "no genome" in log.rule


def test_a_token_listed_twice_is_two_records_and_is_ledgered(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    """RNR4 (one with a trailing nbsp) is two butyric rows, two records and one
    ``RepeatedToken`` entry; the retired RLM2 is a record like any other.
    """
    log = m.DropLog.model_validate_json(
        Path(built.root, m._DROPPED_FILENAME).read_text()
    )
    assert (log.n_raw_rows, log.n_kept_records, log.n_dropped_records) == (9, 9, 0)
    assert (log.n_under_systematic_name, log.n_under_released_symbol) == (4, 5)
    assert log.repeated == [
        m.RepeatedToken(acid="butyric", token="RNR4", scores=["+", "+"])
    ]
    assert len(built) == 9
    stored = [
        p["systematic_gene_name"]
        for i in range(len(built))
        for p in built[i]["experiment"]["genotype"]["perturbations"]
    ]
    assert stored.count("RNR4") == 2 and stored.count("RLM2") == 1


def test_a_systematic_token_is_placed_and_a_symbol_is_released(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    """YAL001C goes on ``KanMxDeletionPerturbation``; EFG1 on
    ``ReleasedGeneNamePerturbation`` with the symbol verbatim, kind deletion, marker
    KanMX; the renamed-looking YGR272C stays as written on the placed leaf.
    """
    by_name = {
        p["systematic_gene_name"]: p
        for i in range(len(built))
        for p in built[i]["experiment"]["genotype"]["perturbations"]
    }
    assert by_name["YAL001C"]["perturbation_type"] == "kanmx_deletion"
    assert by_name["YGR272C"]["perturbation_type"] == "kanmx_deletion"
    assert "YGR271C-A" not in by_name
    efg1 = by_name["EFG1"]
    assert (
        efg1["perturbation_type"],
        efg1["released_gene_name"],
        efg1["released_kind"],
        efg1["marker"],
    ) == ("released_gene_name", "EFG1", "deletion", "KanMX")


# ---- A richer build: title rows, skipped rows, merges, multi-acid drops ---------- #
_EDGE_ROWS: dict[str, list[tuple[str | None, str | None]]] = {
    "acetic": [
        ("TFC3", "+"),
        ("EFG1", "+"),
        ("YGR272C", "++"),
        ("RLM2", "++"),
        ("YDL001W", None),
        (None, "+"),
        ("YDL001W", "0"),
        ("\xa0", "+"),
    ],
    "butyric": [("ALIASX", "+"), ("YBR001C", "+"), ("RLM2", "+"), ("SBR2", "+")],
    "octanoic": [("YDL001W", "++"), ("RLM2", "+")],
}


def _write_edge_sheets(raw: Path, header: str = "Gene/ORF name") -> None:
    for spec in m._ACID_SPECS:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        assert sheet is not None
        sheet.append([f"Table S{spec['n']}. {spec['acid']} acid susceptible mutants"])
        sheet.append([None, None, None])
        sheet.append([header, "Encoded Protein Function", "Growth inhibition"])
        for token, score in _EDGE_ROWS[spec["acid"]]:
            sheet.append([token, "synthetic", score])
        workbook.save(raw / spec["filename"])


@pytest.fixture
def edge_built(tmp_path: Path) -> m.EnvChemgenMota2024Dataset:
    root = tmp_path / "env_chemgen_mota2024"
    (root / "raw").mkdir(parents=True)
    _write_edge_sheets(root / "raw")
    return m.EnvChemgenMota2024Dataset(root=str(root))


_DATASET_NAME = "EnvChemgenMota2024Dataset"
_DOSE = {"acetic": 75.0, "butyric": 14.0, "octanoic": 0.30}


def _gaps() -> list[ProvenanceGap]:
    return [
        ProvenanceGap(
            field="n_samples", reason=ProvenanceGapReason.not_reported_by_primary
        ),
        ProvenanceGap(
            field="sample_unit", reason=ProvenanceGapReason.not_reported_by_primary
        ),
    ]


def _acid_plate(acid: str) -> Environment:
    return Environment(
        media=YPD_AGAR,
        temperature=Temperature(value=30.0),
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound(f"{acid} acid"),
                concentration=Concentration(
                    value=_DOSE[acid], unit=ConcentrationUnit.millimolar
                ),
            ),
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.ph,
                magnitude=Concentration(value=4.5, unit=ConcentrationUnit.ph),
                agent=resolved_compound("hydrochloric acid"),
            ),
        ],
        aerobicity="aerobic",
        duration_hours=48.0,
    )


def _ordinal(rank: float, category: ResponseCategory, label: str) -> Any:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.ordinal,
        assay_type=AssayType.spot_dilution,
        environment_response=rank,
        category=category,
        category_label=label,
        units=m.MEASUREMENT_UNITS,
        provenance_gaps=_gaps(),
    )


def _deletion(token: str) -> KanMxDeletionPerturbation | ReleasedGeneNamePerturbation:
    if token.startswith("Y"):
        return KanMxDeletionPerturbation(
            systematic_gene_name=token, perturbed_gene_name=token
        )
    return ReleasedGeneNamePerturbation(
        systematic_gene_name=token,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=token,
        released_gene_name=token,
        released_kind=ReleasedGeneKind.deletion,
        marker="KanMX",
    )


def _expected_record(
    acid: str, token: str, score: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    grade = {
        "+": (1.0, ResponseCategory.reduced),
        "++": (2.0, ResponseCategory.severely_reduced),
    }[score]
    environment = _acid_plate(acid)
    experiment = EnvironmentResponseExperiment(
        dataset_name=_DATASET_NAME,
        genotype=Genotype(perturbations=[_deletion(token)]),
        environment=environment,
        phenotype=_ordinal(grade[0], grade[1], score),
    )
    reference = EnvironmentResponseExperimentReference(
        dataset_name=_DATASET_NAME,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=environment,
        phenotype_reference=_ordinal(0.0, ResponseCategory.no_change, "0"),
    )
    return experiment.model_dump(), reference.model_dump()


def test_edge_build_record_order_and_stored_names(
    edge_built: m.EnvChemgenMota2024Dataset,
) -> None:
    rows = []
    for i in range(len(edge_built)):
        phenotype = edge_built[i]["experiment"]["phenotype"]
        (perturbation,) = edge_built[i]["experiment"]["genotype"]["perturbations"]
        rows.append(
            (
                perturbation["perturbation_type"],
                perturbation["systematic_gene_name"],
                phenotype["category_label"],
            )
        )
    assert rows == [
        ("released_gene_name", "TFC3", "+"),
        ("released_gene_name", "EFG1", "+"),
        ("kanmx_deletion", "YGR272C", "++"),
        ("released_gene_name", "RLM2", "++"),
        ("released_gene_name", "ALIASX", "+"),
        ("kanmx_deletion", "YBR001C", "+"),
        ("released_gene_name", "RLM2", "+"),
        ("released_gene_name", "SBR2", "+"),
        ("kanmx_deletion", "YDL001W", "++"),
        ("released_gene_name", "RLM2", "+"),
    ]


def test_released_symbol_record_equals_the_hand_built_experiment(
    edge_built: m.EnvChemgenMota2024Dataset,
) -> None:
    """Record 1: 75 mM acetic acid plus the pH 4.5 HCl edit on the YPD plate at 30 C for
    48 h; EFG1 ``+`` (rank 1.0, ``reduced``) under the symbol as released; the
    reference is BY4741 on the same acid plate at rank 0.0.
    """
    experiment, reference = _expected_record("acetic", "EFG1", "+")
    assert edge_built[1]["experiment"] == experiment
    assert edge_built[1]["reference"] == reference
    assert (
        edge_built[1]["publication"]
        == Publication(
            doi="10.1186/s12934-024-02309-0",
            doi_url="https://doi.org/10.1186/s12934-024-02309-0",
        ).model_dump()
    )


def test_systematic_token_records_equal_the_hand_built_experiments(
    edge_built: m.EnvChemgenMota2024Dataset,
) -> None:
    """Record 2: YGR272C ``++`` on the placed KanMX leaf, as written (SGD's merge into
    YGR271C-A is a modeling-time join); record 5: YBR001C ``+`` with 14 mM butyric
    acid; record 8: YDL001W ``++`` with 0.30 mM octanoic acid.
    """
    for index, acid, token, score in [
        (2, "acetic", "YGR272C", "++"),
        (5, "butyric", "YBR001C", "+"),
        (8, "octanoic", "YDL001W", "++"),
    ]:
        experiment, reference = _expected_record(acid, token, score)
        assert edge_built[index]["experiment"] == experiment
        assert edge_built[index]["reference"] == reference


def test_edge_ledger_equals_the_hand_built_accounting(
    edge_built: m.EnvChemgenMota2024Dataset,
) -> None:
    """Ten scored rows, ten records, nothing dropped, no token listed twice in one
    table (RLM2 appears once per table), three systematic names.
    """
    log = m.DropLog.model_validate_json(
        Path(edge_built.root, m._DROPPED_FILENAME).read_text()
    )
    assert log == m.DropLog(
        dataset=_DATASET_NAME,
        rule=m.RETENTION_RULE,
        n_raw_rows=10,
        n_kept_records=10,
        n_dropped_records=0,
        n_under_systematic_name=3,
        n_under_released_symbol=7,
        repeated=[],
    )


def test_unknown_grade_symbol_refuses_naming_file_symbol_and_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #520): a score cell outside ``0``, ``+``, ``++`` (here ``+++`` on
    YDL001W in the octanoic table, the LAST sheet, after acetic and butyric parse
    cleanly) refuses the build instead of being skipped before ``n_raw`` is counted. No
    ``processed/lmdb`` is left and a retry refuses with the same message. The pinned
    sheets carry 0 such cells.
    """
    monkeypatch.setitem(_EDGE_ROWS, "octanoic", [("YDL001W", "++"), ("YDL001W", "+++")])
    root = tmp_path / "ds"
    (root / "raw").mkdir(parents=True)
    _write_edge_sheets(root / "raw")
    assert (
        _refuse_twice(root)
        == [
            "12934_2024_2309_MOESM3_ESM.xlsx: score '+++' for 'YDL001W' is not one of the "
            "grade symbols ['+', '++', '0']"
        ]
        * 2
    )


def _refuse_twice(root: Path) -> list[str]:
    """Construct twice; return both refusal messages and assert no store was left.

    Every sheet is parsed before ``processed/lmdb`` is opened, so a refusal must leave
    no store behind; otherwise the retry would serve the empty store as a built dataset.
    """
    messages = []
    for _ in range(2):
        with pytest.raises(RuntimeError) as info:
            m.EnvChemgenMota2024Dataset(root=str(root))
        messages.append(str(info.value))
        assert not (root / "processed" / "lmdb").exists()
    return messages


def test_edge_side_files_one_reference_per_acid(
    edge_built: m.EnvChemgenMota2024Dataset,
) -> None:
    preprocess = Path(edge_built.preprocess_dir)
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "ALIASX",
        "EFG1",
        "RLM2",
        "SBR2",
        "TFC3",
        "YBR001C",
        "YDL001W",
        "YGR272C",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]
    assert not (preprocess / "data.csv").exists()
    assert edge_built.experiment_class is EnvironmentResponseExperiment
    assert edge_built.reference_class is EnvironmentResponseExperimentReference


def test_a_sheet_without_the_header_row_refuses_naming_the_file(tmp_path: Path) -> None:
    """Contract (issue #520): a renamed header column refuses with a message naming the
    first acid's file and the header it looked for, not a bare ``StopIteration``.
    """
    root = tmp_path / "ds"
    (root / "raw").mkdir(parents=True)
    _write_edge_sheets(root / "raw", header="Gene name")
    assert (
        _refuse_twice(root)
        == [
            "12934_2024_2309_MOESM1_ESM.xlsx: no row starts with the 'Gene/ORF name' "
            "header in the first sheet"
        ]
        * 2
    )


def test_inline_construction_hooks_are_inert() -> None:
    dataset = m.EnvChemgenMota2024Dataset.__new__(m.EnvChemgenMota2024Dataset)
    frame = object()
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()


# ---- download and the raw mirror ------------------------------------------------- #
_PAYLOADS = {"acetic": b"acetic xlsx", "butyric": b"butyric xlsx", "octanoic": b"oct"}


def _pinned_specs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The three specs with each pin replaced by its synthetic payload's digest."""
    specs = [
        {**spec, "sha256": hashlib.sha256(_PAYLOADS[spec["acid"]]).hexdigest()}
        for spec in m._ACID_SPECS
    ]
    monkeypatch.setattr(m, "_ACID_SPECS", specs)
    return specs


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_download_takes_the_mirror_first_then_the_esm_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acetic and butyric are in the mirror (copied, no request); octanoic is not, so
    ``MOESM3`` is fetched from the Springer ESM CDN with a browser User-Agent and a
    180 s timeout; every file then verifies against its pin.
    """
    specs = _pinned_specs(monkeypatch)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    si = data_root / "torchcell-raw" / m.CITATION_KEY / "si"
    si.mkdir(parents=True)
    for spec in specs[:2]:
        (si / spec["filename"]).write_bytes(_PAYLOADS[spec["acid"]])
    calls: list[tuple[str, str | None, int]] = []

    def urlopen(req: Any, timeout: int) -> _Response:
        calls.append((req.full_url, req.get_header("User-agent"), timeout))
        return _Response(_PAYLOADS["octanoic"])

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    dataset = m.EnvChemgenMota2024Dataset.__new__(m.EnvChemgenMota2024Dataset)
    dataset.root = str(tmp_path / "ds")
    dataset.download()
    assert calls == [
        (
            "https://static-content.springer.com/esm/art%3A10.1186%2Fs12934-024-02309-0"
            "/MediaObjects/12934_2024_2309_MOESM3_ESM.xlsx",
            "Mozilla/5.0",
            180,
        )
    ]
    raw = tmp_path / "ds" / "raw"
    assert {spec["acid"]: (raw / spec["filename"]).read_bytes() for spec in specs} == (
        _PAYLOADS
    )


def test_download_refuses_a_mirror_file_with_the_wrong_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    si = data_root / "torchcell-raw" / m.CITATION_KEY / "si"
    si.mkdir(parents=True)
    spec = m._ACID_SPECS[0]
    (si / spec["filename"]).write_bytes(b"not the SI")
    dataset = m.EnvChemgenMota2024Dataset.__new__(m.EnvChemgenMota2024Dataset)
    dataset.root = str(tmp_path / "ds")
    with pytest.raises(RawSha256MismatchError) as info:
        dataset.download()
    got = hashlib.sha256(b"not the SI").hexdigest()
    assert str(info.value) == (
        f"sha256 mismatch for {si / spec['filename']}: expected "
        "b23ad28141e70b307048fc69475aedd4e3cf880118ae9d0d806b6d9f91205e42, "
        f"observed {got}"
    )
    # The mirror file is hashed before the copy: nothing lands in raw/.
    assert list((tmp_path / "ds" / "raw").iterdir()) == []


def test_download_refuses_an_esm_payload_off_the_pin_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no mirror, a Springer ESM body whose digest is not the pin raises
    ``RawSha256MismatchError`` naming the URL and both digests before a byte is written.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout: _Response(b"a changed upload")
    )
    dataset = m.EnvChemgenMota2024Dataset.__new__(m.EnvChemgenMota2024Dataset)
    dataset.root = str(tmp_path / "ds")
    (tmp_path / "ds" / "raw").mkdir(parents=True)
    with pytest.raises(RawSha256MismatchError) as info:
        dataset.download()
    assert str(info.value) == (
        "sha256 mismatch for https://static-content.springer.com/esm/"
        "art%3A10.1186%2Fs12934-024-02309-0/MediaObjects/12934_2024_2309_MOESM1_ESM.xlsx: "
        "expected b23ad28141e70b307048fc69475aedd4e3cf880118ae9d0d806b6d9f91205e42, "
        f"observed {hashlib.sha256(b'a changed upload').hexdigest()}"
    )
    assert list((tmp_path / "ds" / "raw").iterdir()) == []


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): with the three spreadsheets already in ``raw/``
    PyG skips ``download()``, so ``process()`` verifies each against its pin first. The
    acetic-acid sheet off its pin raises ``RawSha256MismatchError`` naming it and both
    digests before a row is read; no store is written.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(m, [spec["filename"] for spec in m._ACID_SPECS])
    raw = staged.root / "raw" / "12934_2024_2309_MOESM1_ESM.xlsx"
    with pytest.raises(RawSha256MismatchError) as info:
        m.EnvChemgenMota2024Dataset(root=str(staged.root))
    assert str(info.value) == (
        f"sha256 mismatch for {raw}: expected "
        "b23ad28141e70b307048fc69475aedd4e3cf880118ae9d0d806b6d9f91205e42, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


def test_raw_mirror_dir_prefers_the_argument_then_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_ROOT", "/env/root")
    assert (
        m.raw_mirror_dir("/given") == "/given/torchcell-raw/motaSharedMoreSpecific2024"
    )
    assert m.raw_mirror_dir() == "/env/root/torchcell-raw/motaSharedMoreSpecific2024"


def test_deposit_refuses_a_source_with_the_wrong_digest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    spec = m._ACID_SPECS[0]
    (source / spec["filename"]).write_bytes(b"wrong")
    got = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(RawSha256MismatchError) as info:
        m.deposit_raw_mirror(
            source_dir=str(source),
            retrieved_at="2026-09-12T00:00:00+00:00",
            data_root=str(tmp_path / "dr"),
        )
    assert str(info.value) == (
        f"sha256 mismatch for {source / spec['filename']}: expected {spec['sha256']}, "
        f"observed {got}"
    )
    assert not (tmp_path / "dr").exists()


def test_deposit_refuses_a_later_source_before_writing_an_earlier_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every spreadsheet is verified before any is written: with acetic and butyric on
    their (repointed) pins and octanoic off its pin, the refusal names octanoic and no
    mirror directory exists, so a partial deposit never happens.
    """
    specs = _pinned_specs(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    for spec in specs[:2]:
        (source / spec["filename"]).write_bytes(_PAYLOADS[spec["acid"]])
    (source / specs[2]["filename"]).write_bytes(b"wrong")
    with pytest.raises(RawSha256MismatchError) as info:
        m.deposit_raw_mirror(
            source_dir=str(source),
            retrieved_at="2026-09-12T00:00:00+00:00",
            data_root=str(tmp_path / "dr"),
        )
    assert str(info.value) == (
        f"sha256 mismatch for {source / specs[2]['filename']}: expected "
        f"{specs[2]['sha256']}, observed {hashlib.sha256(b'wrong').hexdigest()}"
    )
    assert not (tmp_path / "dr").exists()


def test_deposit_refuses_an_existing_mirror_file_with_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = _pinned_specs(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    for spec in specs:
        (source / spec["filename"]).write_bytes(_PAYLOADS[spec["acid"]])
    dest = (
        tmp_path / "dr" / "torchcell-raw" / m.CITATION_KEY / "si" / specs[0]["filename"]
    )
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"tampered")
    with pytest.raises(RuntimeError) as info:
        m.deposit_raw_mirror(
            source_dir=str(source),
            retrieved_at="2026-09-12T00:00:00+00:00",
            data_root=str(tmp_path / "dr"),
        )
    assert str(info.value) == f"{dest} exists with a different sha256; refusing"


def _esm(n: int) -> str:
    return (
        "https://static-content.springer.com/esm/art%3A10.1186%2Fs12934-024-02309-0"
        f"/MediaObjects/12934_2024_2309_MOESM{n}_ESM.xlsx"
    )


def test_deposit_copies_the_three_files_and_writes_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``checked_at`` every retrieval carries a matching ``SourceCheck``; a second
    deposit without it (files already at their pins) rewrites the manifest with
    ``last_check`` None and leaves the files in place.
    """
    specs = _pinned_specs(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    for spec in specs:
        (source / spec["filename"]).write_bytes(_PAYLOADS[spec["acid"]])
    root = m.deposit_raw_mirror(
        source_dir=str(source),
        retrieved_at="2026-09-12T00:00:00+00:00",
        checked_at="2026-09-12T01:00:00+00:00",
        data_root=str(tmp_path / "dr"),
    )
    assert root == str(tmp_path / "dr" / "torchcell-raw" / m.CITATION_KEY)
    for spec in specs:
        assert (
            Path(root, "si", spec["filename"]).read_bytes() == _PAYLOADS[spec["acid"]]
        )
    manifest = Manifest.model_validate_json(Path(root, "manifest.json").read_text())

    def expected(checked: bool) -> Manifest:
        return Manifest(
            citation_key=m.CITATION_KEY,
            doi="10.1186/s12934-024-02309-0",
            title=(
                "Shared and more specific genetic determinants and pathways underlying "
                "yeast tolerance to acetic, butyric, and octanoic acids"
            ),
            library_id="6582362",
            zotero_item_key="4JMAVP2G",
            files=[
                ArtifactRecord(
                    path=f"si/{spec['filename']}",
                    role=ROLE_SI_DATA,
                    bytes=len(_PAYLOADS[spec["acid"]]),
                    sha256=spec["sha256"],
                    source=_esm(spec["n"]),
                    retrieval=RetrievalRecord(
                        method=RetrievalMethod.springer_esm,
                        source_url=_esm(spec["n"]),
                        retriever="torchcell.literature.retrieve.springer_esm",
                        params={"url": _esm(spec["n"])},
                        sha256=spec["sha256"],
                        retrieved_at="2026-09-12T00:00:00+00:00",
                        last_check=(
                            SourceCheck(
                                checked_at="2026-09-12T01:00:00+00:00",
                                produced_sha256=spec["sha256"],
                                matches=True,
                            )
                            if checked
                            else None
                        ),
                    ),
                )
                for spec in specs
            ],
            si_data_sources=[_esm(1), _esm(2), _esm(3)],
            si_expected=[
                "Additional file 1: Table S1 (acetic acid susceptible mutants)",
                "Additional file 2: Table S2 (butyric acid susceptible mutants)",
                "Additional file 3: Table S3 (octanoic acid susceptible mutants)",
            ],
            provenance_complete=True,
            created_at=manifest.created_at,
        )

    assert manifest == expected(checked=True)
    m.deposit_raw_mirror(
        source_dir=str(source),
        retrieved_at="2026-09-12T00:00:00+00:00",
        data_root=str(tmp_path / "dr"),
    )
    again = Manifest.model_validate_json(Path(root, "manifest.json").read_text())
    assert again == expected(checked=False).model_copy(
        update={"created_at": again.created_at}
    )
