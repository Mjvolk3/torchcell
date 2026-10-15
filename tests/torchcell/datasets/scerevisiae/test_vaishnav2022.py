# tests/torchcell/datasets/scerevisiae/test_vaishnav2022.py
# [[tests.torchcell.datasets.scerevisiae.test_vaishnav2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_vaishnav2022.py
"""Vaishnav 2022 sort-seq loader and its verifier, end to end on synthetic files.

``DATA_ROOT`` points into ``tmp_path``. The raw mirror is written by the loader's own
``deposit_raw_mirror`` from synthetic retrievals, and the three shape pins stating the
REAL release's size (75,487 rows, 71,054 records, 104 unmeasured sequences) are
monkeypatched to the synthetic release's.

The synthetic release has five rows over four ``seqID`` groups, which exercises every
shape the real file has:

- ``A`` and ``B``: one row each, one measurement each.
- ``C;D``: TWO rows (one per design label) of ONE measurement, the ``splitByOrigID``
  shape, so the store must hold it once.
- ``E``: one row whose ``ExpressionLevel`` cell is the release's own ``NA``, stored
  as a record whose level is None beside a typed gap.

The Zenodo test set re-releases ``B`` with its ``expt`` token and read count, so exactly
one stored record carries ``library_id`` and ``read_count`` and the others carry neither.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from torchcell.datamodels.media import SC_URA
from torchcell.datamodels.schema import (
    PooledSortSeqExpressionPhenotype,
    ReporterReadout,
    SampleUnit,
    UncertaintyType,
    VariablePromoterReporterPerturbation,
)
from torchcell.datasets.scerevisiae import vaishnav2022 as v
from torchcell.verification import pooled_sort_seq as psq
from torchcell.verification.runners import stream_records

FLANK5 = v.SCAFFOLD_5PRIME
FLANK3 = v.SCAFFOLD_3PRIME


def _construct(letter: str) -> str:
    """A 110 bp construct: the real flanks around an 80 bp single-base insert."""
    return FLANK5 + letter * v.VARIABLE_REGION_LENGTH + FLANK3


SEQUENCES = {
    "A": _construct("A"),
    "B": _construct("C"),
    "C;D": _construct("G"),
    "E": _construct("T"),
}
EXPRESSION = {"A": 8.5, "B": 12.25, "C;D": 3.75}

MOLEVOL_ROWS = [
    ("A", "A", SEQUENCES["A"], EXPRESSION["A"]),
    ("B", "B", SEQUENCES["B"], EXPRESSION["B"]),
    ("C", "C;D", SEQUENCES["C;D"], EXPRESSION["C;D"]),
    ("D", "C;D", SEQUENCES["C;D"], EXPRESSION["C;D"]),
    ("E", "E", SEQUENCES["E"], "NA"),
]
ZENODO_ROWS = [("B", "NBT_S288CdU_SCUra", 647, EXPRESSION["B"])]


def _molevol_bytes() -> bytes:
    frame = pd.DataFrame(
        MOLEVOL_ROWS, columns=["origID", "seqID", "seq110", "ExpressionLevel"]
    )
    return gzip.compress(frame.to_csv(sep="\t", index=False).encode())


def _zenodo_bytes() -> bytes:
    frame = pd.DataFrame(ZENODO_ROWS, columns=["origID", "expt", "proTotal", "meanEL"])
    frame.insert(0, "", range(len(frame)))
    return frame.to_csv(index=False).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


MOLEVOL = _molevol_bytes()
ZENODO = _zenodo_bytes()
README = b"synthetic Zenodo README\n"


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A DATA_ROOT holding the deposited synthetic raw mirror, with the pins rebound."""
    root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(root))
    monkeypatch.setattr(v, "EXPECTED_ROWS", len(MOLEVOL_ROWS))
    monkeypatch.setattr(v, "EXPECTED_RECORDS", 4)
    monkeypatch.setattr(v, "EXPECTED_UNMEASURED_SEQUENCES", 1)
    monkeypatch.setattr(v, "MOLEVOL_SHA256", _sha(MOLEVOL))
    monkeypatch.setattr(v, "TESTDATA_SHA256", _sha(ZENODO))
    monkeypatch.setattr(v, "ZENODO_README_SHA256", _sha(README))
    monkeypatch.setattr(
        v, "RAW_FILES", {v.MOLEVOL_NAME: _sha(MOLEVOL), v.TESTDATA_NAME: _sha(ZENODO)}
    )
    download = tmp_path / "download"
    download.mkdir()
    (download / v.MOLEVOL_NAME).write_bytes(MOLEVOL)
    (download / v.TESTDATA_NAME).write_bytes(ZENODO)
    (download / v.ZENODO_README_NAME).write_bytes(README)
    v.deposit_raw_mirror(
        download_dir=download, retrieved_at="2026-10-12", data_root=str(root)
    )
    return root


@pytest.fixture
def dataset(data_root: Path) -> v.PooledPromoterExpressionVaishnav2022Dataset:
    """The built store over the synthetic release."""
    return v.PooledPromoterExpressionVaishnav2022Dataset(root=str(data_root / "store"))


def _records(
    dataset: v.PooledPromoterExpressionVaishnav2022Dataset,
) -> list[dict[str, Any]]:
    return [dataset[i] for i in range(len(dataset))]


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def test_raw_mirror_records_every_retrieval(data_root: Path) -> None:
    """Three deposited files, each with its direct-URL retrieval and its own sha256."""
    manifest_path = v.raw_mirror_dir(str(data_root)) / "manifest.json"
    from torchcell.literature.manifest import Manifest

    manifest = Manifest.model_validate_json(manifest_path.read_text())
    retrievals = [f.retrieval for f in manifest.files]
    assert all(r is not None for r in retrievals)
    assert [
        (f.path, f.role, f.retrieval.method.value)  # type: ignore[union-attr]
        for f in manifest.files
    ] == [
        (v.MOLEVOL_REL, "raw_data", "direct_url"),
        (v.TESTDATA_REL, "raw_data", "direct_url"),
        (v.ZENODO_README_REL, "raw_data", "direct_url"),
    ]
    assert [f.sha256 for f in manifest.files] == [
        _sha(MOLEVOL),
        _sha(ZENODO),
        _sha(README),
    ]
    assert [f.retrieval.source_url for f in manifest.files] == [  # type: ignore[union-attr]
        v.GEO_URL_BASE + v.MOLEVOL_NAME,
        f"{v.ZENODO_URL_BASE}{v.TESTDATA_NAME}/content",
        f"{v.ZENODO_URL_BASE}{v.ZENODO_README_NAME}/content",
    ]
    assert manifest.doi == v.PAPER_DOI


def test_deposit_refuses_a_file_whose_bytes_are_not_the_recorded_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retrieval that produced other bytes stops the deposit; nothing is written."""
    root = tmp_path / "root"
    monkeypatch.setattr(v, "MOLEVOL_SHA256", _sha(MOLEVOL))
    monkeypatch.setattr(v, "TESTDATA_SHA256", _sha(ZENODO))
    monkeypatch.setattr(v, "ZENODO_README_SHA256", _sha(README))
    download = tmp_path / "download"
    download.mkdir()
    (download / v.MOLEVOL_NAME).write_bytes(MOLEVOL + b"drift")
    (download / v.TESTDATA_NAME).write_bytes(ZENODO)
    (download / v.ZENODO_README_NAME).write_bytes(README)
    with pytest.raises(RuntimeError, match="recorded sha256"):
        v.deposit_raw_mirror(
            download_dir=download, retrieved_at="2026-10-12", data_root=str(root)
        )
    assert not (v.raw_mirror_dir(str(root)) / "manifest.json").exists()


# --------------------------------------------------------------------------- #
# Reading the release
# --------------------------------------------------------------------------- #
def test_read_measurements_collapses_the_split_rows_and_keeps_the_na_row(
    data_root: Path,
) -> None:
    """One row per LISTED seqID: the split pair collapses, the NA row is kept as absent.

    The release decides membership: a group whose cell is ``NA`` is listed, so it stays
    with no estimate, and the Zenodo columns appear only where the release has them.
    """
    mirror = v.raw_mirror_dir(str(data_root))
    frame = v.read_measurements(
        mirror / v.MOLEVOL_REL, mirror / v.TESTDATA_REL
    ).set_index("seqID")
    assert list(frame.index) == ["A", "B", "C;D", "E"]
    assert [frame.loc[key, "expression_level"] for key in ("A", "B", "C;D")] == [
        EXPRESSION["A"],
        EXPRESSION["B"],
        EXPRESSION["C;D"],
    ]
    assert pd.isna(frame.loc["E", "expression_level"])
    assert frame.loc["B", "library_id"] == "NBT_S288CdU_SCUra"
    assert frame.loc["B", "read_count"] == 647
    assert frame.loc["A", "library_id"] is None
    assert pd.isna(frame.loc["A", "read_count"])


def test_read_measurements_refuses_a_seqid_with_two_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seqID carrying two different cells is not one measurement split by label."""
    rows = [
        ("C", "C;D", SEQUENCES["C;D"], "3.75"),
        ("D", "C;D", SEQUENCES["C;D"], "4.0"),
    ]
    molevol = tmp_path / "molevol.txt.gz"
    molevol.write_bytes(
        gzip.compress(
            pd.DataFrame(rows, columns=["origID", "seqID", "seq110", "ExpressionLevel"])
            .to_csv(sep="\t", index=False)
            .encode()
        )
    )
    zenodo = tmp_path / "zenodo.csv"
    zenodo.write_bytes(ZENODO)
    monkeypatch.setattr(v, "EXPECTED_ROWS", len(rows))
    with pytest.raises(RuntimeError, match="more than one ExpressionLevel cell"):
        v.read_measurements(molevol, zenodo)


def test_read_measurements_refuses_a_zenodo_value_that_is_not_the_geo_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real disagreement means two measurements, not a float round-trip."""
    molevol = tmp_path / "molevol.txt.gz"
    molevol.write_bytes(MOLEVOL)
    zenodo = tmp_path / "zenodo.csv"
    frame = pd.DataFrame(
        [("B", "NBT_S288CdU_SCUra", 647, EXPRESSION["B"] + 0.5)],
        columns=["origID", "expt", "proTotal", "meanEL"],
    )
    zenodo.write_bytes(frame.to_csv(index=False).encode())
    monkeypatch.setattr(v, "EXPECTED_ROWS", len(MOLEVOL_ROWS))
    monkeypatch.setattr(v, "EXPECTED_RECORDS", 4)
    monkeypatch.setattr(v, "EXPECTED_UNMEASURED_SEQUENCES", 1)
    with pytest.raises(RuntimeError, match="disagrees with the GEO table"):
        v.read_measurements(molevol, zenodo)


@pytest.mark.parametrize(
    "sequence,match",
    [
        ("ACGT" * 10, "scaffold flanks"),
        (FLANK5 + "A" * 79 + FLANK3, "variable region is 79 bp"),
    ],
)
def test_variable_region_span_refuses_a_construct_it_cannot_locate(
    sequence: str, match: str
) -> None:
    """The span is read off the constant flanks, never assumed."""
    with pytest.raises(RuntimeError, match=match):
        v.variable_region_span(sequence)


def test_variable_region_span_is_the_eighty_base_insert() -> None:
    """1-based inclusive, 18 to 97 for a 110 bp construct."""
    assert v.variable_region_span(SEQUENCES["A"]) == (18, 97)


# --------------------------------------------------------------------------- #
# The built store
# --------------------------------------------------------------------------- #
def test_the_store_holds_one_record_per_listed_sequence(
    dataset: v.PooledPromoterExpressionVaishnav2022Dataset,
) -> None:
    """Four records: the split pair is stored once and the NA row is stored as absent."""
    assert len(dataset) == 4
    records = _records(dataset)
    assert [r["experiment"]["phenotype"]["sequence_id"] for r in records] == [
        "A",
        "B",
        "C;D",
        "E",
    ]
    assert [r["experiment"]["phenotype"]["expression_level"] for r in records] == [
        EXPRESSION["A"],
        EXPRESSION["B"],
        EXPRESSION["C;D"],
        None,
    ]
    assert [r["experiment"]["experiment_type"] for r in records] == [
        "pooled_sort_seq_expression"
    ] * 4


def test_the_na_row_is_a_record_whose_level_is_a_typed_gap(
    dataset: v.PooledPromoterExpressionVaishnav2022Dataset,
) -> None:
    """The release lists the row, so the record exists and its absence is typed.

    The gap names the field, the reason, the pinned file it was read from and the cell
    verbatim, so the absence is queryable rather than a row the loader dropped.
    """
    absent = next(
        r
        for r in _records(dataset)
        if r["experiment"]["phenotype"]["sequence_id"] == "E"
    )
    phenotype = absent["experiment"]["phenotype"]
    assert phenotype["expression_level"] is None
    gap = next(
        g for g in phenotype["provenance_gaps"] if g["field"] == "expression_level"
    )
    assert gap["reason"] == "not_reported_by_primary"
    assert gap["looked_in"]["sha256"] == v.MOLEVOL_SHA256
    assert v.RELEASED_ABSENT_CELL in gap["note"]
    assert (
        absent["experiment"]["genotype"]["perturbations"][0]["promoter_sequence"]
        == SEQUENCES["E"]
    )


def test_every_record_carries_its_sequence_on_the_genotype(
    dataset: v.PooledPromoterExpressionVaishnav2022Dataset,
) -> None:
    """The genotype is the construct: one perturbation holding the released bases."""
    for record in _records(dataset):
        perturbations = record["experiment"]["genotype"]["perturbations"]
        assert len(perturbations) == 1
        perturbation = perturbations[0]
        sequence_id = record["experiment"]["phenotype"]["sequence_id"]
        assert perturbation["perturbation_type"] == "variable_promoter_reporter"
        assert perturbation["promoter_variant_id"] == sequence_id
        assert perturbation["promoter_sequence"] == SEQUENCES[sequence_id]
        assert (
            perturbation["variable_region_start"],
            perturbation["variable_region_end"],
        ) == (18, 97)
        assert perturbation["scaffold_name"] == v.SCAFFOLD_NAME
        assert perturbation["localization"] == v.LOCALIZATION
        assert perturbation["systematic_gene_name"] == v.REPORTER_IDENTIFIER
        assert perturbation["state"] == "present"
        assert perturbation["mechanism_so_id"] == "SO:0000667"


def test_only_the_zenodo_rows_carry_a_library_and_a_read_count(
    dataset: v.PooledPromoterExpressionVaishnav2022Dataset,
) -> None:
    """A released value where the release has one, a typed None where it does not."""
    by_sequence = {
        r["experiment"]["phenotype"]["sequence_id"]: r["experiment"]["phenotype"]
        for r in _records(dataset)
    }
    assert by_sequence["B"]["library_id"] == "NBT_S288CdU_SCUra"
    assert by_sequence["B"]["read_count"] == 647
    for key in ("A", "C;D", "E"):
        assert by_sequence[key]["library_id"] is None
        assert by_sequence[key]["read_count"] is None


def test_the_uncertainty_is_a_typed_gap_on_every_record(
    dataset: v.PooledPromoterExpressionVaishnav2022Dataset,
) -> None:
    """SdEL is author-scaled and undefined, so nothing is stored and the gap says so."""
    for record in _records(dataset):
        phenotype = record["experiment"]["phenotype"]
        assert phenotype["expression_level_uncertainty"] is None
        assert phenotype["expression_level_uncertainty_type"] is None
        assert phenotype["expression_level_se"] is None
        assert phenotype["n_samples"] is None
        assert phenotype["sample_unit"] is None
        gaps = {gap["field"]: gap["reason"] for gap in phenotype["provenance_gaps"]}
        expected = {
            "expression_level_uncertainty",
            "expression_level_uncertainty_type",
            "n_samples",
            "sample_unit",
        }
        # The NA row carries one gap more, on the label itself.
        if phenotype["expression_level"] is None:
            expected = expected | {"expression_level"}
        assert set(gaps) == expected
        assert set(gaps.values()) == {"not_reported_by_primary"}


def test_the_environment_is_sc_ura_at_thirty_with_no_edit(
    dataset: v.PooledPromoterExpressionVaishnav2022Dataset,
) -> None:
    """One medium, one temperature, and no environmental perturbation anywhere."""
    for record in _records(dataset):
        environment = record["experiment"]["environment"]
        assert environment["media"]["name"] == SC_URA.name
        assert environment["temperature"]["value"] == 30.0
        assert environment["perturbations"] == []
        assert environment["aerobicity"] == "aerobic"


def test_the_reference_restates_the_reading_on_the_pinned_strain(
    dataset: v.PooledPromoterExpressionVaishnav2022Dataset,
) -> None:
    """No control-promoter arm is released, so the reference is the record's own reading."""
    for record in _records(dataset):
        reference = record["reference"]
        assert reference["experiment_reference_type"] == "pooled_sort_seq_expression"
        assert reference["genome_reference"]["strain"] == v.STRAIN
        assert reference["phenotype_reference"] == record["experiment"]["phenotype"]
        assert (
            reference["environment_reference"]["media"]["name"]
            == record["experiment"]["environment"]["media"]["name"]
        )


def test_the_publication_is_the_paper(
    dataset: v.PooledPromoterExpressionVaishnav2022Dataset,
) -> None:
    """Every record names the Nature paper by DOI and PubMed id."""
    for record in _records(dataset):
        assert record["publication"]["doi"] == v.PAPER_DOI
        assert record["publication"]["pubmed_id"] == v.PAPER_PMID


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
def _report(data_root: Path, dataset: Any) -> Any:
    from torchcell.verification.runners import POOLED_SORT_SEQ_DATASETS

    spec = dict(POOLED_SORT_SEQ_DATASETS["pooled_promoter_expression_vaishnav2022"])
    root = str(data_root / "store")
    records = list(stream_records(root))
    report = psq.verify_pooled_sort_seq_dataset_streaming(
        lambda: iter(records),
        dataset_name="synthetic",
        provenance=spec["provenance"],
        expected_count=len(dataset),
        maximum=spec["maximum"],
    )
    released: dict[str, float | None] = dict(EXPRESSION)
    released["E"] = None
    report.add(psq.l2_released_values(iter(records), released))
    report.add(psq.l4_read_count_coverage(iter(records)))
    return report


def test_the_verifier_passes_every_level_on_the_built_store(
    data_root: Path, dataset: v.PooledPromoterExpressionVaishnav2022Dataset
) -> None:
    """L0 to L4 pass, and each level's name is in the report exactly once."""
    report = _report(data_root, dataset)
    names = [result.name for result in report.results]
    assert names == [
        "structural",
        "count",
        "sequence_uniqueness",
        "value_fidelity",
        "units_declared",
        "sequence_is_the_genotype",
        "uncertainty_is_typed_or_gapped",
        "estimate_is_stated_or_gapped",
        "released_values",
        "read_count_coverage",
    ]
    stated = next(r for r in report.results if r.name == "estimate_is_stated_or_gapped")
    assert (stated.details["n_stated"], stated.details["n_gapped"]) == (3, 1)
    assert report.passed
    coverage = next(r for r in report.results if r.name == "read_count_coverage")
    assert coverage.details["n_with_read_count"] == 1
    assert coverage.details["smallest"] == 647


# --------------------------------------------------------------------------- #
# The schema classes the loader stands on
# --------------------------------------------------------------------------- #
def test_the_genotype_leaf_refuses_a_span_outside_the_sequence() -> None:
    """A span past the end of the sequence does not locate the insert."""
    with pytest.raises(ValueError, match="past the end of promoter_sequence"):
        VariablePromoterReporterPerturbation(
            systematic_gene_name=v.REPORTER_IDENTIFIER,
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name="YFP",
            source_organism="synthetic construct",
            is_heterologous=True,
            localization=v.LOCALIZATION,
            promoter_variant_id="A",
            promoter_sequence="ACGT",
            scaffold_name="pTpA",
            variable_region_start=2,
            variable_region_end=9,
        )


def test_the_phenotype_leaf_derives_the_se_from_a_typed_uncertainty() -> None:
    """A source that DOES label its dispersion gets the ML-facing SE for free."""
    phenotype = PooledSortSeqExpressionPhenotype(
        expression_level=8.5,
        expression_level_uncertainty=2.0,
        expression_level_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=4,
        sample_unit=SampleUnit.biological_replicate,
        sequence_id="A",
        readout=ReporterReadout.sort_seq_fluorescence,
        reporter_gene="YFP",
        expression_units="bins",
    )
    assert phenotype.expression_level_se == 1.0


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def test_every_sourced_value_quotes_a_text_artifact_of_the_mirror() -> None:
    """Each quote must be auditable: a TEXT artifact, named by citation key and sha256.

    The audit (`audit_sourced_value`) reads the source as text and looks for the quote, so
    a value anchored to a PDF could never pass it. Two of these values are sourced to de
    Boer 2020, the paper this release's method defers to, which is mirrored for that
    reason. This test checks the shape offline; the quotes themselves are re-read against
    the mirror by the verifier when the library is mounted.
    """
    assert set(v.SOURCED_VALUES) == {
        "construct",
        "scaffold",
        "strain",
        "media",
        "temperature",
        "temperature_paper",
        "expression_units",
        "bins",
        "readout",
        "internal_control",
    }
    for name, value in v.SOURCED_VALUES.items():
        provenance = value.provenance
        assert provenance.citation_key in {
            v.CITATION_KEY,
            "deboerDecipheringEukaryoticGeneregulatory2020",
        }, name
        assert provenance.sha256 and len(provenance.sha256) == 64, name
        assert provenance.source_uri.endswith(".md"), name
        assert value.quote.strip(), name


# --------------------------------------------------------------------------- #
# The family runner and the CLI
# --------------------------------------------------------------------------- #
def test_the_family_runner_verifies_the_store_it_is_pointed_at(
    data_root: Path,
    dataset: v.PooledPromoterExpressionVaishnav2022Dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``verify_pooled_sort_seq_dataset`` re-reads the store's own ``raw/`` for L2.

    This is the path the slurm build's ``--verify`` takes, so it is exercised here
    rather than only the streaming gate: the runner reads the release back out of the
    store it just built and compares every stored estimate to it.
    """
    from torchcell.verification import runners

    name = "pooled_promoter_expression_vaishnav2022"
    spec = dict(runners.POOLED_SORT_SEQ_DATASETS[name])
    spec["expected_count"] = len(dataset)
    monkeypatch.setitem(runners.POOLED_SORT_SEQ_DATASETS, name, spec)
    report = runners.verify_pooled_sort_seq_dataset(
        name, str(data_root), abs_root=str(data_root / "store")
    )
    assert report.passed, [r.message for r in report.results if not r.passed]
    released = next(r for r in report.results if r.name == "released_values")
    assert released.details["max_abs_delta"] == 0.0
    assert released.details["n_records"] == len(dataset)


def test_the_cli_builds_and_then_verifies_the_store(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``build`` then ``verify``: the two commands the slurm job and a rerun use."""
    from torchcell.verification import runners

    monkeypatch.setattr(v, "DATASET_ROOT_REL", "store")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    assert v.main(["build"]) == 0
    assert "len = 4" in capsys.readouterr().out

    name = "pooled_promoter_expression_vaishnav2022"
    spec = dict(runners.POOLED_SORT_SEQ_DATASETS[name])
    spec["expected_count"] = 4
    monkeypatch.setitem(runners.POOLED_SORT_SEQ_DATASETS, name, spec)
    assert v.main(["verify"]) == 0
    assert (data_root / "store" / "preprocess" / "verification_report.json").is_file()


def test_the_cli_deposit_writes_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``deposit`` is the one-time command that writes the raw mirror and its manifest."""
    root = tmp_path / "cli_root"
    monkeypatch.setenv("DATA_ROOT", str(root))
    monkeypatch.setattr(v, "MOLEVOL_SHA256", _sha(MOLEVOL))
    monkeypatch.setattr(v, "TESTDATA_SHA256", _sha(ZENODO))
    monkeypatch.setattr(v, "ZENODO_README_SHA256", _sha(README))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    download = tmp_path / "cli_download"
    download.mkdir()
    (download / v.MOLEVOL_NAME).write_bytes(MOLEVOL)
    (download / v.TESTDATA_NAME).write_bytes(ZENODO)
    (download / v.ZENODO_README_NAME).write_bytes(README)
    assert v.main(["deposit", "--download-dir", str(download)]) == 0
    assert "deposited" in capsys.readouterr().out
    assert (v.raw_mirror_dir(str(root)) / "manifest.json").is_file()
