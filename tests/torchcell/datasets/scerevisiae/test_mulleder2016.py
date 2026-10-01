# tests/torchcell/datasets/scerevisiae/test_mulleder2016.py
# [[tests.torchcell.datasets.scerevisiae.test_mulleder2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_mulleder2016.py
"""Hermetic build of the Mulleder 2016 amino-acid metabolome loader on a synthetic Table S3.

The workbook is written with openpyxl under the loader's ``.xls`` filename (pandas sniffs
the zip container and opens it with openpyxl) into ``<root>/raw/`` so PyG never calls
``download()``. No genome is involved: ORFs are validated with a regex.

``intracellular_concentration_mM`` sheet (ORF + the 19 amino acids in ``AMINO_ACIDS``
order):

    YAL001C     0.5, 1.0, ..., 9.5   (0.5 * (i + 1) for amino acid i)
    YBR001C     2.0 x 19
    YBR001C     3.0 x 19             duplicate ORF -> dropped, first row kept
    WT          1.0 x 19             not a systematic name -> skipped
    YLR287-A    1.0 x 19             malformed -> skipped
    " YCR001W " 1.0 x 19             stripped

``robust_summary_statistics`` sheet (amino acid, mean (mM), sd (mM)): mean 0.25 * (i + 1)
per amino acid; every record's reference is that table, n = 1 per amino acid, SE None.

2026.09.30 (Phase 14). Added, each on its own synthetic workbook under ``tmp_path``:

- the replicate counts as Findings: a ``data_raw`` sheet giving YAL001C three raw rows
  leaves its ``n_replicates`` at 1 on all 19 keys (issue #488: 191 released strains have
  two to four raw rows), and the reference, the MCD robust mean over the whole
  collection, also says n = 1 on all 19 keys (issue #489);
- the medium as a Finding: the record and its reference carry ``SM_AGAR`` (solid), not
  the liquid ``SM`` subculture the amino acids are extracted from (issue #143);
- the build ledger: four YBR001C rows, one blank ORF and one ``WT`` row log exactly
  "2 usable ORFs, 2 non-systematic ORF names skipped, 1 ORF collisions deduped" although
  three rows were dropped as collisions (Finding: the ledger counts ORFs, not rows);
- a blank concentration cell is stored as NaN in a served record (Finding) and a text
  cell (``n.d.``) aborts the build with the ``float()`` message;
- ``download()`` on a faked ``urlopen``: the request (URL, User-Agent, timeout 300), the
  sha256 refusal with its exact message and no file written, and the write on a match;
- ``process()`` verifying the raw workbook against ``DATA_SHA256`` before reading a
  sheet, so a file already in ``raw/`` (which ``download()`` returns early on) is refused
  at build time (issue #518's sweep);
- ``main()`` under a stubbed ``load_dotenv`` and a ``tmp_path`` ``DATA_ROOT``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import re
import urllib.request
from pathlib import Path
from typing import Any

import openpyxl
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.media import SM, SM_AGAR
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import mulleder2016 as m

_AA = m.AMINO_ACIDS
_LEVELS_YAL001C = [0.5 * (i + 1) for i in range(19)]
_CONC_ROWS: list[list[Any]] = [
    ["YAL001C", *_LEVELS_YAL001C],
    ["YBR001C", *([2.0] * 19)],
    ["YBR001C", *([3.0] * 19)],
    ["WT", *([1.0] * 19)],
    ["YLR287-A", *([1.0] * 19)],
    [" YCR001W ", *([1.0] * 19)],
]
_SUMMARY_MEANS = {aa: 0.25 * (i + 1) for i, aa in enumerate(_AA)}


def _write_workbook(
    raw: Path,
    conc_columns: list[str] = _AA,
    summary_means: dict[str, float] = _SUMMARY_MEANS,
    conc_rows: list[list[Any]] = _CONC_ROWS,
    raw_rows: list[list[Any]] | None = None,
) -> None:
    workbook = openpyxl.Workbook()
    conc = workbook.active
    conc.title = m._CONC_SHEET
    conc.append(["ORF", *conc_columns])
    for row in conc_rows:
        conc.append(row[: 1 + len(conc_columns)])
    summary = workbook.create_sheet(m._SUMMARY_SHEET)
    summary.append(["amino acid", "mean (mM)", "sd (mM)"])
    for aa, mean in summary_means.items():
        summary.append([aa, mean, 0.1])
    if raw_rows is not None:
        # the released workbook's per-injection sheet (identifier, ORF, batch, uM)
        data_raw = workbook.create_sheet("data_raw")
        data_raw.append(["identifier", "ORF", "batch", *_AA])
        for raw_row in raw_rows:
            data_raw.append(raw_row)
    workbook.save(raw / m.DATA_FILENAME)


def _root(tmp_path: Path, slug: str = "amino_acid_mulleder2016", **kwargs: Any) -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    _write_workbook(root / "raw", **kwargs)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> m.AminoAcidMulleder2016Dataset:
    return m.AminoAcidMulleder2016Dataset(root=str(_root(tmp_path)))


_ENVIRONMENT = Environment(media=SM_AGAR, temperature=Temperature(value=30))
_PUBLICATION = Publication(
    pubmed_id="27693354",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/27693354/",
    doi="10.1016/j.cell.2016.09.007",
    doi_url="https://doi.org/10.1016/j.cell.2016.09.007",
).model_dump()


def _experiment(orf: str, levels: list[float]) -> dict[str, Any]:
    return MetaboliteExperiment(
        dataset_name="AminoAcidMulleder2016Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                )
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=MetabolitePhenotype(
            metabolite_level=dict(zip(_AA, levels, strict=True)),
            metabolite_level_se=None,
            n_replicates=dict.fromkeys(_AA, 1),
            measurement_type="intracellular_concentration_mM",
            target_metabolite_ids=None,
        ),
    ).model_dump()


def _reference(means: dict[str, float]) -> dict[str, Any]:
    return MetaboliteExperimentReference(
        dataset_name="AminoAcidMulleder2016Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=MetabolitePhenotype(
            metabolite_level=means,
            metabolite_level_se=None,
            n_replicates=dict.fromkeys(means, 1),
            measurement_type="intracellular_concentration_mM",
        ),
    ).model_dump()


def test_three_records_n_one_with_the_robust_mean_reference(
    dataset: m.AminoAcidMulleder2016Dataset,
) -> None:
    """Six rows give three records (YAL001C, the first YBR001C, the stripped YCR001W);
    the WT and malformed names are skipped and the second YBR001C is a collision. Each
    amino acid is n = 1 with no SE on solid SM at 30 C; the reference is the summary mean.
    """
    assert len(dataset) == 3
    assert dataset[0]["experiment"] == _experiment("YAL001C", _LEVELS_YAL001C)
    assert dataset[1]["experiment"] == _experiment("YBR001C", [2.0] * 19)
    assert dataset[2]["experiment"] == _experiment("YCR001W", [1.0] * 19)
    assert dataset[0]["reference"] == _reference(_SUMMARY_MEANS)
    assert dataset[0]["publication"] == _PUBLICATION
    assert dataset[0]["experiment"]["environment"]["media"]["state"] == "solid"


def test_side_files(dataset: m.AminoAcidMulleder2016Dataset) -> None:
    """``data.csv`` is one row per kept ORF with the 19 concentrations; one reference
    covers all three records; the gene set is the three ORFs.
    """
    assert dataset.experiment_class is MetaboliteExperiment
    assert dataset.reference_class is MetaboliteExperimentReference
    preprocess = Path(dataset.root) / "preprocess"
    expected_rows = [
        ["YAL001C", *_LEVELS_YAL001C],
        ["YBR001C", *([2.0] * 19)],
        ["YCR001W", *([1.0] * 19)],
    ]
    assert (preprocess / "data.csv").read_text() == (
        ",".join(["orf", *_AA])
        + "\n"
        + "".join(",".join(str(v) for v in row) + "\n" for row in expected_rows)
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YCR001W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2]]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "amino_acid_mulleder2016"
    assert manifest["loader_class"] == "AminoAcidMulleder2016Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.mulleder2016"


def test_reference_takes_every_summary_row_not_only_the_19_amino_acids(
    tmp_path: Path,
) -> None:
    """Finding: ``_reference_levels`` is built from every row of the summary sheet, so an
    extra ``cysteine`` row (the amino acid the paper excludes) lands in the reference
    phenotype with n = 1 while no experiment measures it; the reference then has 20 keys
    against the experiment's 19.
    """
    means = {**_SUMMARY_MEANS, "cysteine": 0.125}
    root = _root(tmp_path, summary_means=means)
    dataset = m.AminoAcidMulleder2016Dataset(root=str(root))
    assert dataset[0]["reference"] == _reference(means)
    assert set(dataset[0]["experiment"]["phenotype"]["metabolite_level"]) == set(_AA)


def test_missing_concentration_column_raises(tmp_path: Path) -> None:
    root = _root(tmp_path, conc_columns=_AA[:-1])
    with pytest.raises(
        RuntimeError,
        match=re.escape("Table S3 concentration sheet missing columns: ['tyrosine']"),
    ):
        m.AminoAcidMulleder2016Dataset(root=str(root))


def test_missing_summary_amino_acid_raises(tmp_path: Path) -> None:
    means = {aa: v for aa, v in _SUMMARY_MEANS.items() if aa != "tyrosine"}
    root = _root(tmp_path, summary_means=means)
    with pytest.raises(
        RuntimeError, match=re.escape("summary sheet missing amino acids: ['tyrosine']")
    ):
        m.AminoAcidMulleder2016Dataset(root=str(root))


def test_download_leaves_a_present_file_to_the_build_check(
    dataset: m.AminoAcidMulleder2016Dataset,
) -> None:
    """An already present raw file makes ``download()`` return early: the bytes are left
    untouched and ``urlopen`` is never reached (the conftest guard would raise if it
    did). The pin is enforced on that file by ``process()``, asserted next.
    """
    dest = Path(dataset.root) / "raw" / m.DATA_FILENAME
    before = dest.read_bytes()
    assert hashlib.sha256(before).hexdigest() != m.DATA_SHA256
    dataset.download()
    assert dest.read_bytes() == before


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): with Table S3 already in ``raw/`` PyG skips
    ``download()``, so ``process()`` verifies it against ``DATA_SHA256`` first and
    raises ``RawSha256MismatchError`` naming it and both digests before a sheet is read;
    no store is written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(m, [m.DATA_FILENAME])
    raw = staged.root / "raw" / m.DATA_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        m.AminoAcidMulleder2016Dataset(root=str(staged.root))
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected {m.DATA_SHA256}, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


def test_n_replicates_is_one_even_for_a_strain_with_three_raw_rows(
    tmp_path: Path,
) -> None:
    """Finding (issue #488): ``n_replicates`` is the constant 1 on every amino acid
    (source line 249). The released workbook's ``data_raw`` sheet holds 2 to 4 raw rows
    for 191 of the 4,678 released strains, and the issue's diagnostic suggests their
    released value is the mean of those rows; the loader never opens ``data_raw``. Here
    YAL001C has three raw rows (batches 01, 02, 12) and its record still says n = 1 on all
    19 keys with no SE. Pinned until #488 sets ``n_replicates`` per record from the
    ``data_raw`` row count.
    """
    raw_rows = [
        [f"S{i:03d}", "YAL001C", batch, *([10.0 * (i + 1)] * 19)]
        for i, batch in enumerate(["01", "02", "12"])
    ] + [["S003", "YBR001C", "01", *([20.0] * 19)]]
    root = _root(tmp_path, raw_rows=raw_rows)
    dataset = m.AminoAcidMulleder2016Dataset(root=str(root))
    phenotype = dataset[0]["experiment"]["phenotype"]
    assert dataset[0]["experiment"]["genotype"]["perturbations"][0][
        "systematic_gene_name"
    ] == ("YAL001C")
    assert phenotype["n_replicates"] == dict.fromkeys(_AA, 1)
    assert phenotype["metabolite_level_se"] is None
    assert phenotype["metabolite_level"] == dict(zip(_AA, _LEVELS_YAL001C, strict=True))


def test_reference_is_the_population_robust_mean_but_says_n_one(
    dataset: m.AminoAcidMulleder2016Dataset,
) -> None:
    """Finding (issue #489): the reference phenotype is the ``robust_summary_statistics``
    ``mean (mM)`` column, the Minimum Covariance Determinant mean over the whole
    collection (0.25 * (i + 1) mM for amino acid i here: alanine 0.25, tyrosine 4.75),
    and it is stored with ``n_replicates = 1`` on all 19 keys and no SE, which describes
    one measurement, not a population estimate over 4,678 strains. The same reference is
    attached to all three records. Pinned until #489 decides how a population-statistic
    reference is represented.
    """
    for index in range(3):
        reference = dataset[index]["reference"]["phenotype_reference"]
        assert reference["metabolite_level"] == _SUMMARY_MEANS
        assert reference["metabolite_level"]["alanine"] == 0.25
        assert reference["metabolite_level"]["tyrosine"] == 4.75
        assert reference["n_replicates"] == dict.fromkeys(_AA, 1)
        assert reference["metabolite_level_se"] is None
        assert reference["measurement_type"] == "intracellular_concentration_mM"


def test_medium_is_sm_agar_not_the_liquid_subculture(
    dataset: m.AminoAcidMulleder2016Dataset,
) -> None:
    """Finding (issue #143, and the open question on ``SM_AGAR``'s docstring in
    ``torchcell/datamodels/media.py``): record and reference both carry ``SM_AGAR``,
    ``state="solid"`` with agar at 2% among its components, while the amino acids are
    extracted from the liquid SM subculture the spots inoculate (``SM``, same recipe
    without agar). #143 recorded this loader's solid SM as a bare stub; the recipe has
    since been sourced, so what remains pinned is the state. Pinned until #143 (or its
    successor) decides whether the record should carry ``SM``.
    """
    experiment_media = dataset[0]["experiment"]["environment"]["media"]
    reference_media = dataset[0]["reference"]["environment_reference"]["media"]
    assert experiment_media == SM_AGAR.model_dump()
    assert reference_media == SM_AGAR.model_dump()
    assert experiment_media != SM.model_dump()
    assert experiment_media["state"] == "solid"
    agar = experiment_media["components"][-1]
    assert (agar["compound"]["name"], agar["concentration"]["value"]) == ("agar", 2.0)
    assert dataset[0]["experiment"]["environment"]["temperature"]["value"] == 30.0


def test_ledger_counts_collided_orfs_not_dropped_rows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Six rows: YBR001C four times (2.0, 3.0, 4.0, 5.0), a blank ORF cell (read as NaN,
    stringified to ``"nan"``), ``WT``, and YAL001C. Kept: YBR001C (the first row, 2.0 on
    every key) and YAL001C, so two records. The ledger line counts the blank and ``WT``
    as two non-systematic names.

    Finding: the collision count is the number of distinct ORFs that collided (1), not
    the three rows discarded, so the ledger under-reports what the build dropped (source
    lines 182-193). Pinned until the ledger counts discarded rows.
    """
    rows: list[list[Any]] = [
        ["YBR001C", *([2.0] * 19)],
        ["YBR001C", *([3.0] * 19)],
        [None, *([1.0] * 19)],
        ["YBR001C", *([4.0] * 19)],
        ["WT", *([1.0] * 19)],
        ["YBR001C", *([5.0] * 19)],
        ["YAL001C", *_LEVELS_YAL001C],
    ]
    root = _root(tmp_path, conc_rows=rows)
    caplog.set_level(logging.INFO, logger=m.__name__)
    dataset = m.AminoAcidMulleder2016Dataset(root=str(root))
    assert len(dataset) == 2
    assert dataset[0]["experiment"] == _experiment("YBR001C", [2.0] * 19)
    assert dataset[1]["experiment"] == _experiment("YAL001C", _LEVELS_YAL001C)
    messages = [r.getMessage() for r in caplog.records if r.name == m.__name__]
    assert messages == [
        "Mulleder: 2 usable ORFs, 2 non-systematic ORF names skipped, "
        "1 ORF collisions deduped",
        "Wrote 2 Mulleder amino-acid experiments to LMDB",
    ]


def test_blank_concentration_is_served_as_nan(tmp_path: Path) -> None:
    """Finding: a blank concentration cell is read as NaN and ``float(nan)`` passes, so
    the record serves ``metabolite_level["tyrosine"] = nan`` with ``n_replicates`` 1 for
    it; ``MetabolitePhenotype`` does not reject a non-finite level. The other 18 keys
    keep their values. Pinned until the loader refuses or omits an unmeasured key.
    """
    rows: list[list[Any]] = [["YAL001C", *_LEVELS_YAL001C[:-1], None]]
    root = _root(tmp_path, conc_rows=rows)
    dataset = m.AminoAcidMulleder2016Dataset(root=str(root))
    level = dataset[0]["experiment"]["phenotype"]["metabolite_level"]
    assert math.isnan(level["tyrosine"])
    assert {aa: v for aa, v in level.items() if aa != "tyrosine"} == dict(
        zip(_AA[:-1], _LEVELS_YAL001C[:-1], strict=True)
    )
    assert dataset[0]["experiment"]["phenotype"]["n_replicates"]["tyrosine"] == 1


def test_text_concentration_aborts_the_build(tmp_path: Path) -> None:
    """A non-numeric cell reaches ``float(row[aa])`` (source line 186) and the build
    stops with Python's own message, naming the cell text; nothing is written to LMDB.
    """
    rows: list[list[Any]] = [["YAL001C", "n.d.", *_LEVELS_YAL001C[1:]]]
    root = _root(tmp_path, conc_rows=rows)
    with pytest.raises(
        ValueError, match=re.escape("could not convert string to float: 'n.d.'")
    ):
        m.AminoAcidMulleder2016Dataset(root=str(root))
    assert not (root / "processed" / "lmdb").exists()


class _Response(io.BytesIO):
    """The context-manager body ``urlopen`` returns."""

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _fake_urlopen(
    monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> list[tuple[urllib.request.Request, float]]:
    calls: list[tuple[urllib.request.Request, float]] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        calls.append((request, timeout))
        return _Response(payload)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


def test_download_refuses_a_payload_with_the_wrong_sha256(
    dataset: m.AminoAcidMulleder2016Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the raw file absent, ``download()`` requests ``DATA_URL`` once with a
    ``Mozilla/5.0`` User-Agent and a 300 s timeout, hashes the body, and refuses a
    mismatch with the exact message, writing nothing.
    """
    dest = Path(dataset.root) / "raw" / m.DATA_FILENAME
    dest.unlink()
    payload = b"not the Mendeley workbook"
    calls = _fake_urlopen(monkeypatch, payload)
    got = hashlib.sha256(payload).hexdigest()
    with pytest.raises(RawSha256MismatchError) as err:
        dataset.download()
    assert str(err.value) == (
        f"sha256 mismatch for {m.DATA_URL}: expected {m.DATA_SHA256}, observed {got}"
    )
    assert sorted(p.name for p in dest.parent.iterdir()) == []
    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == m.DATA_URL
    assert request.header_items() == [("User-agent", "Mozilla/5.0")]
    assert timeout == 300


def test_download_writes_a_payload_whose_sha256_matches_the_pin(
    dataset: m.AminoAcidMulleder2016Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the pin set to the payload's digest, the bytes land at ``raw/<filename>``
    unchanged.
    """
    dest = Path(dataset.root) / "raw" / m.DATA_FILENAME
    dest.unlink()
    payload = b"stand-in workbook bytes"
    monkeypatch.setattr(m, "DATA_SHA256", hashlib.sha256(payload).hexdigest())
    calls = _fake_urlopen(monkeypatch, payload)
    dataset.download()
    assert dest.read_bytes() == payload
    assert len(calls) == 1


def test_main_prints_length_and_first_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main()`` opens ``$DATA_ROOT/data/torchcell/amino_acid_mulleder2016`` and prints
    ``len = 3`` then record 0. The store is built first (its progress lines go to stdout
    and are discarded) so ``main()`` only loads it; ``load_dotenv`` is stubbed so the
    repo ``.env`` is never read and ``DATA_ROOT`` is ``tmp_path``.
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    root = _root(tmp_path / "data" / "torchcell")
    first = m.AminoAcidMulleder2016Dataset(root=str(root))[0]
    capsys.readouterr()
    m.main()
    assert capsys.readouterr().out == f"len = 3\n{first}\n"
