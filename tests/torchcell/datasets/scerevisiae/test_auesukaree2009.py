# tests/torchcell/datasets/scerevisiae/test_auesukaree2009.py
# [[tests.torchcell.datasets.scerevisiae.test_auesukaree2009]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_auesukaree2009.py
"""Auesukaree 2009 loader: shared YPD, the unperturbed reference, the two adjudications.

All synthetic: the PDF table parse is monkeypatched and the genome is a stub resolver, so
nothing here needs the raw mirror or a built LMDB.

2026.09.30 (Phase 12): the second half ("Table parse and end-to-end build") drives the
real ``_parse_tables`` on a synthetic ``pdftotext -layout`` text layer (``_LAYOUT``,
returned by a module-local fake ``subprocess.run`` that records its argv), then the full
``process()`` under ``tmp_path`` with an explicit resolver (``_LayoutGenome``; an
unknown token is a test bug and raises ``KeyError``). The layout carries six captions,
a class row whose list wraps onto a continuation line, a trailing-comma cell, an
indented page-number line after a complete class (not a gene), prose before and after,
and a ``Table 7`` caption outside the 1-6 pattern. Parsed tokens and records:

    stress      listed tokens                        records (idx: ORF / stored name)
    ethanol     VMA2 YAL001C YBR127C YBL001C         0 YAL001C, 1 YBL001C, 2 YBR127C/VMA2
    methanol    FEN1 NOSUCHGENE                      3 YCR034W/ELO2
    1-propanol  PPA1                                 4 YHR026W/VMA16
    heat        YAL002W                              5 YAL002W (37 C, no perturbation)
    NaCl        YAL003W                              6 YAL003W (1 M sodium chloride)
    H2O2        YAL004W NOSUCHGENE                   7 YAL004W (5 mM hydrogen peroxide)

VMA2 is RENAMED onto YBR127C and the same table also lists YBR127C, so the ethanol
table's four tokens give three records and the first-seen name VMA2 is stored. Ledger:
11 listed tokens = 8 kept records + 2 dropped (NOSUCHGENE twice, RETIRED) + 1 collapsed
(the ethanol YBR127C token, ledgered since 2026.10.01, issue #520).
All eight records share the one non-stress reference, so the reference index is
[[0, ..., 7]]; ``gene_set.json`` is the eight ORFs sorted. Refusals pinned with their
exact messages: the missing raw mirror, a ``RawSha256MismatchError`` naming the file and
both digests (in ``download`` before the copy, in ``deposit_raw_mirror`` before any
mirror directory exists, and in ``process`` for a file placed in ``raw/`` by hand,
issue #518's sweep), an existing mirror file with other bytes,
a missing genome, a per-stress checksum miss, and a stress table absent from the PDF.

2026.10.01 (issue #520): a functional class whose parsed gene count differs from its
declared (N) refuses (short at table end, over at the next class row), and an in-table
collapse onto an already-claimed ORF is ledgered as a ``CollapsedToken``. Every table
is resolved before ``processed/lmdb`` is opened, so a refusal leaves no store and a retry
refuses again (asserted on the checksum miss and the missing table).
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YPD_AGAR
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Publication,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.scerevisiae import auesukaree2009 as a
from torchcell.literature.manifest import (
    ROLE_PAPER_PDF,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)

_TABLES = {
    "ethanol": ["VMA2", "YAL001C"],
    "methanol": ["FEN1"],
    "1-propanol": ["PPA1"],
    "heat": ["YAL002W"],
    "NaCl": ["YAL003W"],
    "H2O2": ["NOSUCHGENE"],
}


def _dataset() -> a.EnvChemgenAuesukaree2009Dataset:
    """An uninitialized instance: the methods under test read no build state."""
    return a.EnvChemgenAuesukaree2009Dataset.__new__(a.EnvChemgenAuesukaree2009Dataset)


@pytest.fixture
def built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> a.EnvChemgenAuesukaree2009Dataset:
    """A tiny end-to-end build over synthetic stress tables."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / a._PDF_FILENAME).write_bytes(b"")  # presence only: download() is skipped
    monkeypatch.setattr(a, "_EXPECTED_LISTED", {k: len(v) for k, v in _TABLES.items()})
    monkeypatch.setattr(
        a.EnvChemgenAuesukaree2009Dataset, "_parse_tables", lambda self: _TABLES
    )
    return a.EnvChemgenAuesukaree2009Dataset(root=str(tmp_path))


def test_media_is_the_shared_library_object_not_free_text() -> None:
    spec = next(s for s in a._STRESS_SPECS if s["stress"] == "ethanol")
    environment = _dataset()._environment(spec)
    # the PLATE member of the YPD family, not the bare join anchor: a spot assay is on
    # solid medium, so this dataset shares one media node with Mota 2024's plates
    assert environment.media is YPD_AGAR
    assert environment.media.state == "solid"
    assert environment.media.base_medium == "YPD"
    assert [c.compound.name for c in environment.media.components] == [
        "yeast extract",
        "peptone",
        "D-glucose",
        "agar",
    ]
    assert environment.duration_hours == 72.0
    assert environment.temperature is not None
    assert environment.temperature.value == 30.0


def test_heat_is_a_temperature_edit_with_no_perturbation_object() -> None:
    spec = next(s for s in a._STRESS_SPECS if s["stress"] == "heat")
    environment = _dataset()._environment(spec)
    assert environment.perturbations == []
    assert environment.temperature is not None
    assert environment.temperature.value == 37.0


def test_every_stress_compound_is_identified_at_its_sourced_dose() -> None:
    doses = {}
    for spec in a._STRESS_SPECS:
        if spec["kind"] != "small_molecule":
            continue
        environment = _dataset()._environment(spec)
        (entry,) = environment.perturbations
        perturbation = cast(SmallMoleculePerturbation, entry)
        assert perturbation.compound.inchikey is not None
        assert perturbation.concentration.unit is not None
        doses[perturbation.compound.name] = (
            perturbation.concentration.value,
            perturbation.concentration.unit.value,
        )
    assert doses == {
        "ethanol": (10.0, "percent_v/v"),
        "methanol": (16.0, "percent_v/v"),
        "1-propanol": (7.0, "percent_v/v"),
        "sodium chloride": (1.0, "M"),
        "hydrogen peroxide": (5.0, "mM"),
    }


def test_ambiguous_aliases_are_stored_as_the_symbols_the_paper_wrote(
    built: a.EnvChemgenAuesukaree2009Dataset,
) -> None:
    """PPA1 and FEN1 are aliases of two current genes each; no adjudication is made and
    each is a ``ReleasedIdentifierPerturbation`` carrying the symbol verbatim (#1045).
    """
    by_name = {
        p["systematic_gene_name"]: p
        for i in range(len(built))
        for p in built[i]["experiment"]["genotype"]["perturbations"]
    }
    for symbol in ("PPA1", "FEN1", "VMA2", "NOSUCHGENE"):
        assert by_name[symbol]["perturbation_type"] == "released_identifier"
        assert by_name[symbol]["systematic_gene_name"] == symbol
        assert by_name[symbol]["released_kind"] == "deletion"
        assert by_name[symbol]["cassette"] == "KanMX"
    assert by_name["YAL001C"]["perturbation_type"] == "kanmx_deletion"
    assert not hasattr(a, "_AMBIGUOUS_ADJUDICATIONS")


def test_category_is_typed_and_the_source_word_is_kept(
    built: a.EnvChemgenAuesukaree2009Dataset,
) -> None:
    phenotype = built[0]["experiment"]["phenotype"]
    assert phenotype["measurement_type"] is MeasurementType.categorical
    assert phenotype["assay_type"] is AssayType.spot_dilution
    assert phenotype["category"] is ResponseCategory.sensitive
    assert phenotype["category_label"] == "sensitive"
    reference = built[0]["reference"]["phenotype_reference"]
    assert reference["category"] is ResponseCategory.no_change
    assert reference["category_label"] == "tolerant"


def test_reference_environment_is_the_unperturbed_non_stress_plate(
    built: a.EnvChemgenAuesukaree2009Dataset,
) -> None:
    for i in range(len(built)):
        environment = built[i]["reference"]["environment_reference"]
        assert environment["perturbations"] == []
        assert environment["temperature"]["value"] == 30.0
        assert environment["media"] == YPD_AGAR.model_dump()
    assert "vs the SAME strain on the matched non-stress" in a.MEASUREMENT_UNITS


def test_no_listed_token_is_dropped(built: a.EnvChemgenAuesukaree2009Dataset) -> None:
    """Seven listed tokens, seven records; a token no annotation carries is a record."""
    assert len(built) == 7
    log = json.loads(open(osp.join(built.root, a._DROPPED_FILENAME)).read())
    assert log["n_listed_tokens"] == 7
    assert log["n_kept_records"] == 7
    assert log["n_dropped_records"] == 0
    assert (log["n_under_systematic_name"], log["n_under_released_symbol"]) == (3, 4)


# ---- Table parse and end-to-end build (2026.09.30) ------------------------------ #
_LAYOUT = "\n".join(
    [
        "J Appl Genet 50(3), 2009, pp. 301-310",
        "Table 1. Classification of genes whose deletions result in ethanol sensitivity",
        "Vacuolar function (3)       VMA2, YAL001C,",
        "                            YBR127C",
        "Unknown function (1)        YBL001C",
        "                            305",
        "Table 2. Classification of genes whose deletions result in methanol sensitivity",
        "Lipid metabolism (2)        FEN1, NOSUCHGENE",
        "",
        "Table 3. Classification of genes whose deletions result in 1-propanol "
        "sensitivity",
        "Vacuolar function (1)       PPA1",
        "Table 4. Classification of genes whose deletions result in heat sensitivity",
        "Unknown function (1)        YAL002W",
        "Table 5. Classification of genes whose deletions result in NaCl sensitivity",
        "Ion homeostasis (1)         YAL003W",
        "Footnote: genes are listed alphabetically within each class.",
        "Table 6. Classification of genes whose deletions result in H2O2 sensitivity",
        "Cell rescue (2)             YAL004W, NOSUCHGENE",
        "Table 7. Classification of genes whose deletions result in zinc sensitivity",
        "Metal (1)                   YAL005C",
        "",
    ]
)

_PARSED = {
    "ethanol": ["VMA2", "YAL001C", "YBR127C", "YBL001C"],
    "methanol": ["FEN1", "NOSUCHGENE"],
    "1-propanol": ["PPA1"],
    "heat": ["YAL002W"],
    "NaCl": ["YAL003W"],
    "H2O2": ["YAL004W", "NOSUCHGENE"],
}


class _Completed:
    """The two attributes of ``CompletedProcess`` the module reads."""

    def __init__(self, stdout: str, stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


def _fake_subprocess(
    monkeypatch: pytest.MonkeyPatch, stdout: str, stderr: str = ""
) -> list[list[str]]:
    """Replace the module's ``subprocess`` only (git in the build manifest stays real)."""
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> _Completed:
        assert kwargs == {"capture_output": True, "text": True, "check": True}
        calls.append(list(argv))
        return _Completed(stdout, stderr)

    monkeypatch.setattr(a, "subprocess", SimpleNamespace(run=run))
    return calls


def test_parse_tables_reads_the_layout_text_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Captions 1-6 open a table; a continuation line is read only while the class is
    short of its declared count (so the indented page number ``305`` after the complete
    ``Unknown function (1)`` row ends the ethanol table); a blank line, a footnote or the
    next caption end a table; ``Table 7`` is outside the caption pattern and never read.
    """
    calls = _fake_subprocess(monkeypatch, _LAYOUT)
    dataset = _dataset()
    dataset.root = str(tmp_path)
    assert dataset._parse_tables() == _PARSED
    assert calls == [
        ["pdftotext", "-layout", osp.join(str(tmp_path), "raw", "paper.pdf"), "-"]
    ]


_HEAT_CAPTION = (
    "Table 4. Classification of genes whose deletions result in heat sensitivity"
)


def test_a_class_short_of_its_declared_count_refuses_at_table_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #520): the parenthetical class count is a per-class
    self-checksum. The last class of a table, declaring (5) and listing two genes, refuses
    when the table ends, naming the stress, the class and both counts. All 67 classes of
    the pinned PDF match their declared count.
    """
    text = "\n".join(
        [_HEAT_CAPTION, "Unknown function (5)        YAL002W, YAL003W", ""]
    )
    _fake_subprocess(monkeypatch, text)
    dataset = _dataset()
    dataset.root = str(tmp_path)
    with pytest.raises(RuntimeError) as info:
        dataset._parse_tables()
    assert str(info.value) == (
        "heat: class 'Unknown function' declares 5 genes, parsed 2 "
        "(per-class table extraction self-checksum failed)"
    )


def test_a_class_over_its_declared_count_refuses_at_the_next_class_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A class row listing three genes against a declared (2) refuses as soon as the
    next class row opens, before that class is read.
    """
    text = "\n".join(
        [
            _HEAT_CAPTION,
            "Vacuolar function (2)        YAL002W, YAL003W, YAL004W",
            "Unknown function (1)        YAL005C",
            "",
        ]
    )
    _fake_subprocess(monkeypatch, text)
    dataset = _dataset()
    dataset.root = str(tmp_path)
    with pytest.raises(RuntimeError) as info:
        dataset._parse_tables()
    assert str(info.value) == (
        "heat: class 'Vacuolar function' declares 2 genes, parsed 3 "
        "(per-class table extraction self-checksum failed)"
    )


def test_tokenize_splits_on_commas_and_whitespace() -> None:
    assert a._tokenize("  VMA2, YAL001C,\tYBR127C ,, ") == [
        "VMA2",
        "YAL001C",
        "YBR127C",
    ]
    assert a._tokenize("   ") == []


_EXPECTED_LAYOUT_COUNTS = {k: len(v) for k, v in _PARSED.items()}
_DATASET_NAME = "EnvChemgenAuesukaree2009Dataset"


@pytest.fixture
def layout_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> a.EnvChemgenAuesukaree2009Dataset:
    """``process()`` end to end on ``_LAYOUT``; raw ``paper.pdf`` present, so no download."""
    root = tmp_path / "env_chemgen_auesukaree2009"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / a._PDF_FILENAME).write_bytes(b"%PDF-synthetic")
    monkeypatch.setattr(a, "_EXPECTED_LISTED", _EXPECTED_LAYOUT_COUNTS)
    _fake_subprocess(monkeypatch, _LAYOUT)
    return a.EnvChemgenAuesukaree2009Dataset(root=str(root))


def _phenotype(category: ResponseCategory, label: str) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.categorical,
        assay_type=AssayType.spot_dilution,
        category=category,
        category_label=label,
        n_samples=3,
        sample_unit=SampleUnit.biological_replicate,
        units=a.MEASUREMENT_UNITS,
    )


def _plate(temperature: float, perturbations: list[Any]) -> Environment:
    return Environment(
        media=YPD_AGAR,
        temperature=Temperature(value=temperature),
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_hours=72.0,
    )


def _deletion(token: str) -> KanMxDeletionPerturbation | ReleasedIdentifierPerturbation:
    if token.startswith("Y"):
        return KanMxDeletionPerturbation(
            systematic_gene_name=token, perturbed_gene_name=token
        )
    return ReleasedIdentifierPerturbation(
        systematic_gene_name=token,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=token,
        identifier_kind="gene name, as Tables 1-6 list it",
        released_kind=ReleasedKind.deletion,
        cassette="KanMX",
    )


def _record(token: str, environment: Environment) -> dict[str, Any]:
    return EnvironmentResponseExperiment(
        dataset_name=_DATASET_NAME,
        genotype=Genotype(perturbations=[_deletion(token)]),
        environment=environment,
        phenotype=_phenotype(ResponseCategory.sensitive, "sensitive"),
    ).model_dump()


_REFERENCE = EnvironmentResponseExperimentReference(
    dataset_name=_DATASET_NAME,
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="BY4742"
    ),
    environment_reference=_plate(30.0, []),
    phenotype_reference=_phenotype(ResponseCategory.no_change, "tolerant"),
)


def test_build_stores_one_record_per_listed_token_in_table_order(
    layout_built: a.EnvChemgenAuesukaree2009Dataset,
) -> None:
    """Stress order is ``_STRESS_SPECS`` (ethanol, methanol, 1-propanol, heat, NaCl,
    H2O2); within a stress the tokens keep their listed order; VMA2 and YBR127C (one
    gene) are two records, as are NOSUCHGENE's two listings in two tables.
    """
    assert len(layout_built) == 11
    pairs = []
    for i in range(11):
        (perturbation,) = layout_built[i]["experiment"]["genotype"]["perturbations"]
        pairs.append(
            (perturbation["perturbation_type"], perturbation["systematic_gene_name"])
        )
    assert pairs == [
        ("released_identifier", "VMA2"),
        ("kanmx_deletion", "YAL001C"),
        ("kanmx_deletion", "YBR127C"),
        ("kanmx_deletion", "YBL001C"),
        ("released_identifier", "FEN1"),
        ("released_identifier", "NOSUCHGENE"),
        ("released_identifier", "PPA1"),
        ("kanmx_deletion", "YAL002W"),
        ("kanmx_deletion", "YAL003W"),
        ("kanmx_deletion", "YAL004W"),
        ("released_identifier", "NOSUCHGENE"),
    ]


def test_ethanol_record_equals_the_hand_built_experiment(
    layout_built: a.EnvChemgenAuesukaree2009Dataset,
) -> None:
    """Record 0: 10 percent v/v ethanol on the YPD plate at 30 C for 72 h, the deletion
    under the released symbol VMA2, categorical ``sensitive`` over three biological
    replicates; the reference is the unperturbed 30 C plate. Record 2 is the same
    gene's systematic token YBR127C on the placed KanMX leaf.
    """
    ethanol = SmallMoleculePerturbation(
        compound=resolved_compound("ethanol"),
        concentration=Concentration(value=10.0, unit=ConcentrationUnit.percent_v_v),
    )
    assert layout_built[0]["experiment"] == _record("VMA2", _plate(30.0, [ethanol]))
    assert layout_built[2]["experiment"] == _record("YBR127C", _plate(30.0, [ethanol]))
    assert layout_built[0]["reference"] == _REFERENCE.model_dump()
    assert (
        layout_built[0]["publication"]
        == Publication(
            doi="10.1007/BF03195688", doi_url="https://doi.org/10.1007/BF03195688"
        ).model_dump()
    )


def test_heat_and_released_symbol_records_equal_the_hand_built_experiments(
    layout_built: a.EnvChemgenAuesukaree2009Dataset,
) -> None:
    """Record 7 (heat): the plate at 37 C with no perturbation object, yet its reference
    is the same 30 C plate as every other record. Record 4 (FEN1): 16 percent v/v
    methanol, stored under the symbol as released. Record 9 (H2O2): 5 mM hydrogen
    peroxide.
    """
    assert layout_built[7]["experiment"] == _record("YAL002W", _plate(37.0, []))
    assert layout_built[7]["reference"] == _REFERENCE.model_dump()
    methanol = SmallMoleculePerturbation(
        compound=resolved_compound("methanol"),
        concentration=Concentration(value=16.0, unit=ConcentrationUnit.percent_v_v),
    )
    assert layout_built[4]["experiment"] == _record("FEN1", _plate(30.0, [methanol]))
    peroxide = SmallMoleculePerturbation(
        compound=resolved_compound("hydrogen peroxide"),
        concentration=Concentration(value=5.0, unit=ConcentrationUnit.millimolar),
    )
    assert layout_built[9]["experiment"] == _record("YAL004W", _plate(30.0, [peroxide]))


def test_drop_log_accounts_for_every_listed_token(
    layout_built: a.EnvChemgenAuesukaree2009Dataset,
) -> None:
    """Every listed token is a record: ``n_listed_tokens`` (11) is ``n_kept_records``
    and nothing is dropped or collapsed. The pinned PDF has 525 listed tokens, 525
    records.
    """
    log = a.DropLog.model_validate_json(
        Path(layout_built.root, a._DROPPED_FILENAME).read_text()
    )
    assert log == a.DropLog(
        dataset=_DATASET_NAME,
        rule=a.RETENTION_RULE,
        n_listed_tokens=11,
        n_kept_records=11,
        n_dropped_records=0,
        n_under_systematic_name=6,
        n_under_released_symbol=5,
    )


def test_a_token_listed_twice_in_one_table_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One listed gene is one record, so a table that lists YAL001C twice refuses by
    name before the store is opened.
    """
    text = _LAYOUT.replace(
        "Unknown function (1)        YBL001C", "Unknown function (1)        YAL001C"
    )
    root = tmp_path / "ds"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / a._PDF_FILENAME).write_bytes(b"%PDF-synthetic")
    monkeypatch.setattr(a, "_EXPECTED_LISTED", _EXPECTED_LAYOUT_COUNTS)
    _fake_subprocess(monkeypatch, text)
    with pytest.raises(
        a.RepeatedTokenError, match="ethanol: the table lists 'YAL001C' twice"
    ):
        a.EnvChemgenAuesukaree2009Dataset(root=str(root))
    assert not (root / "processed" / "lmdb").exists()


def test_side_files_gene_set_and_one_shared_reference(
    layout_built: a.EnvChemgenAuesukaree2009Dataset,
) -> None:
    """``gene_set.json`` is the ten released names sorted; the reference index holds
    one reference with members 0 to 10; the loader writes no ``data.csv``.
    """
    preprocess = Path(layout_built.preprocess_dir)
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "FEN1",
        "NOSUCHGENE",
        "PPA1",
        "VMA2",
        "YAL001C",
        "YAL002W",
        "YAL003W",
        "YAL004W",
        "YBL001C",
        "YBR127C",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [list(range(11))]
    assert index[0]["reference"] == _REFERENCE.model_dump(mode="json")
    assert not (preprocess / "data.csv").exists()
    assert layout_built.experiment_class is EnvironmentResponseExperiment
    assert layout_built.reference_class is EnvironmentResponseExperimentReference


def _refuse_twice(root: Path) -> list[str]:
    """Construct twice; return both refusal messages and assert no store was left."""
    messages = []
    for _ in range(2):
        with pytest.raises(RuntimeError) as info:
            a.EnvChemgenAuesukaree2009Dataset(root=str(root))
        messages.append(str(info.value))
        assert not (root / "processed" / "lmdb").exists()
    return messages


def test_per_stress_checksum_miss_refuses_with_both_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Heat is the fourth stress, so three tables resolve before it fails. Since
    2026.10.01 (issue #520) every table resolves before the store is opened: no
    ``processed/lmdb`` is left behind and a retry refuses again, instead of serving the
    empty store as a built dataset.
    """
    root = tmp_path / "ds"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / a._PDF_FILENAME).write_bytes(b"%PDF-synthetic")
    monkeypatch.setattr(a, "_EXPECTED_LISTED", {**_EXPECTED_LAYOUT_COUNTS, "heat": 2})
    _fake_subprocess(monkeypatch, _LAYOUT)
    assert (
        _refuse_twice(root)
        == [
            "heat: parsed 1 listed genes, expected 2 (table extraction self-checksum "
            "failed)"
        ]
        * 2
    )


def test_a_stress_table_absent_from_the_pdf_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the Table 4 caption the heat table is missing; the message lists the five
    parsed stresses sorted (uppercase sorts before lowercase).
    """
    text = _LAYOUT.replace("in heat sensitivity", "in cold sensitivity")
    root = tmp_path / "ds"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / a._PDF_FILENAME).write_bytes(b"%PDF-synthetic")
    monkeypatch.setattr(a, "_EXPECTED_LISTED", _EXPECTED_LAYOUT_COUNTS)
    _fake_subprocess(monkeypatch, text)
    assert (
        _refuse_twice(root)
        == [
            "stress table not found in PDF: 'heat' (parsed: ['1-propanol', 'H2O2', "
            "'NaCl', 'cold', 'ethanol', 'methanol'])"
        ]
        * 2
    )


# ---- download and the raw mirror ------------------------------------------------- #
def test_build_without_raw_or_mirror_names_the_missing_mirror_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    src = data_root / "torchcell-raw" / a.CITATION_KEY / "paper" / "paper.pdf"
    with pytest.raises(RuntimeError) as info:
        a.EnvChemgenAuesukaree2009Dataset(root=str(tmp_path / "ds"))
    assert str(info.value) == (
        f"raw-mirror PDF not found: {src}. Deposit it with deposit_raw_mirror() from "
        "the library mirror's paper.pdf."
    )


def test_download_copies_from_the_mirror_then_refuses_a_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirrored = data_root / "torchcell-raw" / a.CITATION_KEY / "paper" / "paper.pdf"
    mirrored.parent.mkdir(parents=True)
    mirrored.write_bytes(b"not the pinned pdf")
    dataset = _dataset()
    dataset.root = str(tmp_path / "ds")
    with pytest.raises(RawSha256MismatchError) as info:
        dataset.download()
    got = hashlib.sha256(b"not the pinned pdf").hexdigest()
    assert str(info.value) == (
        f"sha256 mismatch for {mirrored}: expected {a._PDF_SHA256}, observed {got}"
    )
    # The mirror is hashed before the copy: nothing, not even a partial, lands in raw/.
    assert list((tmp_path / "ds" / "raw").iterdir()) == []


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract: with ``raw/paper.pdf`` present PyG skips ``download``, so ``process``
    verifies it against the pin first and raises ``RawSha256MismatchError`` before any
    table is parsed; no store is written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(a, ["paper.pdf"])
    raw = staged.root / "raw" / "paper.pdf"
    with pytest.raises(RawSha256MismatchError) as info:
        a.EnvChemgenAuesukaree2009Dataset(root=str(staged.root))
    assert str(info.value) == (
        f"sha256 mismatch for {raw}: expected {a._PDF_SHA256}, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


def test_raw_mirror_dir_prefers_the_argument_then_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_ROOT", "/env/root")
    assert a.raw_mirror_dir("/given") == (
        "/given/torchcell-raw/auesukareeGenomewideIdentificationGenes2009"
    )
    assert a.raw_mirror_dir() == (
        "/env/root/torchcell-raw/auesukareeGenomewideIdentificationGenes2009"
    )


def test_poppler_version_is_the_first_line_of_stderr_else_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_subprocess(
        monkeypatch, "", "pdftotext version 24.02.0\nCopyright 2005-2024 Poppler\n"
    )
    assert a.poppler_version() == "pdftotext version 24.02.0"
    assert calls == [["pdftotext", "-v"]]
    _fake_subprocess(monkeypatch, "pdftotext version 0.86.1\nmore\n", "")
    assert a.poppler_version() == "pdftotext version 0.86.1"


def test_deposit_refuses_a_source_with_the_wrong_digest(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"wrong bytes")
    got = hashlib.sha256(b"wrong bytes").hexdigest()
    with pytest.raises(RawSha256MismatchError) as info:
        a.deposit_raw_mirror(
            source_pdf=str(source),
            retrieved_at="2026-01-01T00:00:00+00:00",
            data_root=str(tmp_path / "dr"),
        )
    assert str(info.value) == (
        f"sha256 mismatch for {source}: expected {a._PDF_SHA256}, observed {got}"
    )
    # The source is checked before any mirror directory is created.
    assert not (tmp_path / "dr").exists()


def test_deposit_refuses_an_existing_mirror_file_with_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"the pinned bytes")
    monkeypatch.setattr(
        a, "_PDF_SHA256", hashlib.sha256(b"the pinned bytes").hexdigest()
    )
    dest = tmp_path / "dr" / "torchcell-raw" / a.CITATION_KEY / "paper" / "paper.pdf"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"something else")
    with pytest.raises(RuntimeError) as info:
        a.deposit_raw_mirror(
            source_pdf=str(source),
            retrieved_at="2026-01-01T00:00:00+00:00",
            data_root=str(tmp_path / "dr"),
        )
    assert str(info.value) == f"{dest} exists with a different sha256; refusing"
    assert dest.read_bytes() == b"something else"


def test_deposit_copies_the_pdf_and_writes_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the pin set to the synthetic bytes' digest, the PDF lands at
    ``paper/paper.pdf`` and ``manifest.json`` records the Zotero retrieval and the
    ``pdftotext -layout`` processing step with the (faked) poppler version.
    """
    payload = b"%PDF-1.4 synthetic"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(a, "_PDF_SHA256", digest)
    _fake_subprocess(monkeypatch, "", "pdftotext version 24.02.0\n")
    source = tmp_path / "paper.pdf"
    source.write_bytes(payload)
    root = a.deposit_raw_mirror(
        source_pdf=str(source),
        retrieved_at="2026-01-01T00:00:00+00:00",
        data_root=str(tmp_path / "dr"),
    )
    assert root == str(tmp_path / "dr" / "torchcell-raw" / a.CITATION_KEY)
    assert Path(root, "paper", "paper.pdf").read_bytes() == payload
    manifest = Manifest.model_validate_json(Path(root, "manifest.json").read_text())
    expected = Manifest(
        citation_key=a.CITATION_KEY,
        doi="10.1007/BF03195688",
        title=(
            "Genome-wide identification of genes involved in tolerance to various "
            "environmental stresses in Saccharomyces cerevisiae"
        ),
        library_id="6582362",
        zotero_item_key="IGDTEZJV",
        files=[
            ArtifactRecord(
                path="paper/paper.pdf",
                role=ROLE_PAPER_PDF,
                bytes=len(payload),
                sha256=digest,
                source="zotero:attachment:VIJCFVIA",
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.zotero_attachment,
                    source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2747848/",
                    retriever="torchcell.literature.zotero.ZoteroLibrary.download_artifact",
                    params={
                        "library_id": "6582362",
                        "zotero_item_key": "IGDTEZJV",
                        "attachment_key": "VIJCFVIA",
                        "citation_key": a.CITATION_KEY,
                    },
                    sha256=digest,
                    retrieved_at="2026-01-01T00:00:00+00:00",
                ),
                processing=ProcessingRecord(
                    processor="torchcell.datasets.scerevisiae.auesukaree2009."
                    "EnvChemgenAuesukaree2009Dataset._parse_tables",
                    tool="pdftotext",
                    version="pdftotext version 24.02.0",
                    params={"args": ["-layout"]},
                    input_sha256=[digest],
                ),
            )
        ],
        si_data_sources=[],
        si_expected=[
            "none -- the article Tables 1-6 ARE the data; this paper released no "
            "supplementary data file"
        ],
        provenance_complete=True,
        created_at=manifest.created_at,
    )
    assert manifest == expected
    # a second deposit finds the mirror file at the pin and leaves it in place
    source.write_bytes(payload)
    assert (
        a.deposit_raw_mirror(
            source_pdf=str(source),
            retrieved_at="2026-02-02T00:00:00+00:00",
            data_root=str(tmp_path / "dr"),
        )
        == root
    )
    assert Path(root, "paper", "paper.pdf").read_bytes() == payload
    again = Manifest.model_validate_json(Path(root, "manifest.json").read_text())
    assert again.model_dump()["files"][0]["retrieval"]["retrieved_at"] == (
        "2026-02-02T00:00:00+00:00"
    )


def test_inline_construction_hooks_are_inert() -> None:
    """``create_experiment`` is unreachable by design and ``preprocess_raw`` is identity."""
    dataset = _dataset()
    frame = object()
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()


# Phase 24: the verified-copy log line on the raw-mirror path


def test_download_copies_a_matching_raw_mirror_pdf_and_logs_the_verified_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The raw-mirror ``paper/paper.pdf`` is copied into ``raw_dir`` and logged.

    ``_PDF_SHA256`` is patched to ``sha256(b"%PDF-stub")``; the source is
    ``$DATA_ROOT/torchcell-raw/<CITATION_KEY>/paper/paper.pdf``. ``download`` runs
    unbound on a namespace with only ``raw_dir``. INFO line:
    ``Verified <raw_dir>/paper.pdf (sha256 <digest>)``.
    """
    import logging

    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    src = data_root / "torchcell-raw" / a.CITATION_KEY / "paper" / "paper.pdf"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"%PDF-stub")
    digest = hashlib.sha256(b"%PDF-stub").hexdigest()
    monkeypatch.setattr(a, "_PDF_SHA256", digest)
    raw_dir = tmp_path / "root" / "raw"
    with caplog.at_level(logging.INFO, logger=a.__name__):
        a.EnvChemgenAuesukaree2009Dataset.download(
            cast(Any, SimpleNamespace(raw_dir=str(raw_dir)))
        )
    dest = raw_dir / "paper.pdf"
    assert dest.read_bytes() == b"%PDF-stub"
    assert [r.getMessage() for r in caplog.records if r.name == a.__name__] == [
        f"Verified {dest} (sha256 {digest})"
    ]
