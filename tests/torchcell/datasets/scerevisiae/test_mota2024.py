# tests/torchcell/datasets/scerevisiae/test_mota2024.py
# [[tests.torchcell.datasets.scerevisiae.test_mota2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_mota2024.py
"""Mota 2024 loader: pH as a typed edit, the ordinal grade, and the merge/drop rules.

The three spreadsheets are written synthetically with openpyxl, so the parser, the dedup
rule and the retention rule all run for real without the raw mirror.

2026.09.30 (Phase 12): a second, richer build (``_EDGE_ROWS``, explicit resolver
``_EdgeGenome``; an unknown token raises ``KeyError``). Each sheet opens with a title row
and a blank row before the ``Gene/ORF name`` header, so the header search is exercised.

    acid      row (token, score)      rule                              counted
    acetic    TFC3 +                  CURRENT YAL001C, kept as TFC3     yes
              EFG1 +, YGR272C ++      one gene YGR271C-A, ++ wins,      yes, yes
                                      canonical name EFG1 stored
              RLM2 ++                 RETIRED, dropped                  yes
              YDL001W (blank score)   skipped                           no
              (blank token) +         skipped                           no
              YDL001W 0               the reference grade, skipped      no
              YDL001W +++             not a grade symbol, skipped       no
              nbsp-only token +       empty after strip, skipped        no
    butyric   ALIASX +, YBR001C +     one gene YBR001C, tie; the        yes, yes
                                      attribute table's name is None,
                                      so the ORF is stored
              RLM2 +, SBR2 +          RETIRED, dropped                  yes, yes
    octanoic  YDL001W ++              kept                              yes
              RLM2 +                  RETIRED, dropped                  yes

Records (acid order, then ORF): 0 YAL001C/TFC3 + (acetic), 1 YGR271C-A/EFG1 ++
(acetic), 2 YBR001C/YBR001C + (butyric), 3 YDL001W/YDL001W ++ (octanoic). Ledger:
10 counted rows = 4 kept + 4 dropped (RLM2 in three acids, SBR2 once) + 2 merged. Each
acid has its own reference (the acid rides on the reference plate), so the reference
index is [[0, 1], [2], [3]]. Also pinned: the download order (mirror, then the ESM URL),
the sha256 refusals with both digests, ``deposit_raw_mirror`` and its manifest, the
missing-genome and missing-header refusals.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
from pathlib import Path
from typing import Any, cast

import openpyxl
import pandas as pd
import pytest

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
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

# EFG1 and YGR272C are the SI's two names for one gene; RLM2 is retired.
_RENAME = {"EFG1": "YGR271C-A", "YGR272C": "YGR271C-A", "RNR4": "YGR180C"}
_RETIRED = {"RLM2"}


class _StubGenome:
    """Resolver + attribute table, the two things the loader asks a genome for."""

    gene_attribute_table = pd.DataFrame(
        {"ID": ["YGR271C-A", "YGR180C", "YAL001C"], "gene": ["EFG1", "RNR4", "TFC3"]}
    )

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        upper = name.upper()
        if upper in _RENAME:
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.RENAMED,
                systematic_name=_RENAME[upper],
            )
        if upper in _RETIRED:
            return GeneNameResolution(
                input_name=name, status=GeneNameStatus.RETIRED, systematic_name=upper
            )
        return GeneNameResolution(
            input_name=name, status=GeneNameStatus.CURRENT, systematic_name=upper
        )


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
    return m.EnvChemgenMota2024Dataset(
        root=str(tmp_path), genome=cast(SCerevisiaeGenome, _StubGenome())
    )


def test_ph_is_a_typed_perturbation_and_the_medium_is_the_shared_object() -> None:
    spec = next(s for s in m._ACID_SPECS if s["acid"] == "acetic")
    dataset = m.EnvChemgenMota2024Dataset.__new__(m.EnvChemgenMota2024Dataset)
    environment = dataset._environment(spec)
    assert environment.media is YPD_AGAR
    assert environment.media.base_medium == "YPD"
    assert "pH" not in environment.media.name
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


def test_two_source_names_for_one_gene_merge_under_its_canonical_name(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    log = json.loads(open(osp.join(built.root, m._DROPPED_FILENAME)).read())
    merged = {(e["acid"], e["systematic_name"]): e for e in log["merged"]}
    acetic = merged[("acetic", "YGR271C-A")]
    assert acetic["source_tokens"] == ["EFG1", "YGR272C"]
    assert acetic["kept_score"] == "++"
    assert acetic["stored_gene_name"] == "EFG1"  # the genome's canonical common name
    # octanoic: EFG1 is + and YGR272C is ++, so the MORE SEVERE grade wins
    assert merged[("octanoic", "YGR271C-A")]["kept_score"] == "++"
    # butyric: the RNR4 source duplicate is one token twice, so the token survives
    butyric = merged[("butyric", "YGR180C")]
    assert butyric["source_tokens"] == ["RNR4"] and butyric["kept_score"] == "+"
    assert "MORE SEVERE" in log["dedup_rule"]


def test_retired_token_is_dropped_and_counted(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    log = json.loads(open(osp.join(built.root, m._DROPPED_FILENAME)).read())
    assert log["n_raw_rows"] == 9
    assert log["n_dropped_records"] == 1
    assert log["n_merged_records"] == 3
    assert len(built) == 5 == log["n_kept_records"]
    (dropped,) = log["dropped"]
    assert dropped["token"] == "RLM2" and dropped["status"] == "retired"
    assert dropped["acids"] == ["acetic"]


def test_renamed_systematic_looking_token_is_rekeyed(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    stored = {
        p["systematic_gene_name"]
        for i in range(len(built))
        for p in built[i]["experiment"]["genotype"]["perturbations"]
    }
    assert "YGR271C-A" in stored and "YGR272C" not in stored


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
        ("YDL001W", "+++"),
        ("\xa0", "+"),
    ],
    "butyric": [("ALIASX", "+"), ("YBR001C", "+"), ("RLM2", "+"), ("SBR2", "+")],
    "octanoic": [("YDL001W", "++"), ("RLM2", "+")],
}

_EDGE_RESOLUTIONS: dict[str, tuple[GeneNameStatus, str]] = {
    "TFC3": (GeneNameStatus.CURRENT, "YAL001C"),
    "EFG1": (GeneNameStatus.CURRENT, "YGR271C-A"),
    "YGR272C": (GeneNameStatus.RENAMED, "YGR271C-A"),
    "RLM2": (GeneNameStatus.RETIRED, "RLM2"),
    "SBR2": (GeneNameStatus.RETIRED, "SBR2"),
    "ALIASX": (GeneNameStatus.CURRENT, "YBR001C"),
    "YBR001C": (GeneNameStatus.CURRENT, "YBR001C"),
    "YDL001W": (GeneNameStatus.CURRENT, "YDL001W"),
}


class _EdgeGenome:
    """Explicit resolutions; YBR001C sits in the attribute table with no common name."""

    gene_attribute_table = pd.DataFrame(
        {
            "ID": ["YAL001C", "YGR271C-A", "YBR001C", "YDL001W"],
            "gene": ["TFC3", "EFG1", None, "RMD9"],
        }
    )

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        status, systematic = _EDGE_RESOLUTIONS[name]
        return GeneNameResolution(
            input_name=name, status=status, systematic_name=systematic
        )


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
    return m.EnvChemgenMota2024Dataset(
        root=str(root), genome=cast(SCerevisiaeGenome, _EdgeGenome())
    )


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


def _expected_record(
    acid: str, orf: str, stored: str, score: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    grade = {
        "+": (1.0, ResponseCategory.reduced),
        "++": (2.0, ResponseCategory.severely_reduced),
    }[score]
    environment = _acid_plate(acid)
    experiment = EnvironmentResponseExperiment(
        dataset_name=_DATASET_NAME,
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=stored
                )
            ]
        ),
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
                perturbation["systematic_gene_name"],
                perturbation["perturbed_gene_name"],
                phenotype["category_label"],
            )
        )
    assert rows == [
        ("YAL001C", "TFC3", "+"),
        ("YGR271C-A", "EFG1", "++"),
        ("YBR001C", "YBR001C", "+"),
        ("YDL001W", "YDL001W", "++"),
    ]


def test_merged_acetic_record_equals_the_hand_built_experiment(
    edge_built: m.EnvChemgenMota2024Dataset,
) -> None:
    """Record 1: 75 mM acetic acid plus the pH 4.5 HCl edit on the YPD plate at 30 C for
    48 h; EFG1 (+) and YGR272C (++) merge on YGR271C-A at the more severe ``++``
    (rank 2.0, ``severely_reduced``) under the canonical name EFG1; the reference is
    BY4741 on the same acid plate at rank 0.0.
    """
    experiment, reference = _expected_record("acetic", "YGR271C-A", "EFG1", "++")
    assert edge_built[1]["experiment"] == experiment
    assert edge_built[1]["reference"] == reference
    assert (
        edge_built[1]["publication"]
        == Publication(
            doi="10.1186/s12934-024-02309-0",
            doi_url="https://doi.org/10.1186/s12934-024-02309-0",
        ).model_dump()
    )


def test_tied_merge_without_a_common_name_stores_the_orf(
    edge_built: m.EnvChemgenMota2024Dataset,
) -> None:
    """Record 2: ALIASX and YBR001C tie at ``+`` on YBR001C, whose attribute-table name
    is None, so the ORF itself is stored; 14 mM butyric acid; octanoic record 3 carries
    0.30 mM.
    """
    experiment, reference = _expected_record("butyric", "YBR001C", "YBR001C", "+")
    assert edge_built[2]["experiment"] == experiment
    assert edge_built[2]["reference"] == reference
    experiment, reference = _expected_record("octanoic", "YDL001W", "YDL001W", "++")
    assert edge_built[3]["experiment"] == experiment
    assert edge_built[3]["reference"] == reference


def test_edge_drop_log_equals_the_hand_built_ledger(
    edge_built: m.EnvChemgenMota2024Dataset,
) -> None:
    """RLM2 is dropped in all three acids (one entry, three records), SBR2 once;
    merges are listed in acid order with their sorted tokens and scores.
    """
    log = m.DropLog.model_validate_json(
        Path(edge_built.root, m._DROPPED_FILENAME).read_text()
    )
    assert log == m.DropLog(
        dataset=_DATASET_NAME,
        rule=m.DROP_RULE,
        dedup_rule=m._DEDUP_RULE,
        n_raw_rows=10,
        n_kept_records=4,
        n_dropped_records=4,
        n_merged_records=2,
        dropped=[
            m.DroppedToken(
                token="RLM2",
                status="retired",
                acids=["acetic", "butyric", "octanoic"],
                n_records=3,
            ),
            m.DroppedToken(
                token="SBR2", status="retired", acids=["butyric"], n_records=1
            ),
        ],
        merged=[
            m.MergedGene(
                systematic_name="YGR271C-A",
                acid="acetic",
                source_tokens=["EFG1", "YGR272C"],
                source_scores=["+", "++"],
                kept_score="++",
                stored_gene_name="EFG1",
            ),
            m.MergedGene(
                systematic_name="YBR001C",
                acid="butyric",
                source_tokens=["ALIASX", "YBR001C"],
                source_scores=["+", "+"],
                kept_score="+",
                stored_gene_name="YBR001C",
            ),
        ],
    )


def test_unknown_grade_symbol_is_skipped_without_a_ledger_entry(
    edge_built: m.EnvChemgenMota2024Dataset,
) -> None:
    """Finding: a score cell outside ``0``, ``+``, ``++`` (here ``+++`` on YDL001W in the
    acetic table) is skipped before ``n_raw`` is counted, exactly like the blank score and
    the reference grade ``0``, so a malformed or new grade symbol vanishes with no count
    and no ledger entry. Pinned until an unknown symbol refuses or is ledgered.
    """
    acetic_orfs = {
        edge_built[i]["experiment"]["genotype"]["perturbations"][0][
            "systematic_gene_name"
        ]
        for i in range(2)
    }
    assert acetic_orfs == {"YAL001C", "YGR271C-A"}
    log = json.loads(Path(edge_built.root, m._DROPPED_FILENAME).read_text())
    assert "YDL001W" not in json.dumps(log["dropped"])
    assert log["n_raw_rows"] == 10


def test_edge_side_files_one_reference_per_acid(
    edge_built: m.EnvChemgenMota2024Dataset,
) -> None:
    preprocess = Path(edge_built.preprocess_dir)
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YDL001W",
        "YGR271C-A",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1], [2], [3]]
    assert not (preprocess / "data.csv").exists()
    assert edge_built.experiment_class is EnvironmentResponseExperiment
    assert edge_built.reference_class is EnvironmentResponseExperimentReference


def test_a_sheet_without_the_header_row_raises_a_bare_stop_iteration(
    tmp_path: Path,
) -> None:
    """Finding: ``_parse_acid`` finds the header with ``next(...)`` and no default, so a
    renamed header column surfaces as a bare ``StopIteration`` with no message naming the
    file. Pinned until the header search refuses with a message.
    """
    root = tmp_path / "ds"
    (root / "raw").mkdir(parents=True)
    _write_edge_sheets(root / "raw", header="Gene name")
    with pytest.raises(StopIteration) as info:
        m.EnvChemgenMota2024Dataset(
            root=str(root), genome=cast(SCerevisiaeGenome, _EdgeGenome())
        )
    assert info.value.args == ()


def test_parsing_without_a_genome_refuses() -> None:
    dataset = m.EnvChemgenMota2024Dataset.__new__(m.EnvChemgenMota2024Dataset)
    dataset.genome = None
    with pytest.raises(RuntimeError) as info:
        dataset._parse_acid(m._ACID_SPECS[0])
    assert str(info.value) == (
        "EnvChemgenMota2024Dataset requires a genome for gene-name resolution; "
        "inject SCerevisiaeGenome(...)"
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
    with pytest.raises(RuntimeError) as info:
        dataset.download()
    got = hashlib.sha256(b"not the SI").hexdigest()
    assert str(info.value) == (
        f"12934_2024_2309_MOESM1_ESM.xlsx sha256 mismatch: got {got}, "
        "expected b23ad28141e70b307048fc69475aedd4e3cf880118ae9d0d806b6d9f91205e42"
    )


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
    with pytest.raises(RuntimeError) as info:
        m.deposit_raw_mirror(
            source_dir=str(source),
            retrieved_at="2026-09-12T00:00:00+00:00",
            data_root=str(tmp_path / "dr"),
        )
    assert str(info.value) == (
        f"{source / spec['filename']} sha256 {got} != pinned {spec['sha256']}; "
        "refusing to deposit"
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
