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
per amino acid; every record's reference is that table, SE None, and (since 2026.10.02,
issue #489) n = the number of released records on every key (3 here).

``data_raw`` sheet (identifier, ORF, batch, 19 uM columns; written by default since
2026.10.02): one row each for YAL001C, YBR001C, YCR001W plus one ``QC_001`` row with no
ORF, so every default record has n = 1 per amino acid (issue #488 counts these rows).

2026.09.30 (Phase 14). Added, each on its own synthetic workbook under ``tmp_path``:

- the replicate counts, Findings until 2026.10.02 and contracts since: three
  ``data_raw`` rows for YAL001C give n = 3 (issue #488), and the reference n is the
  number of strains the MCD robust mean summarizes (issue #489);
- the medium as a Finding: the record and its reference carry ``SM_AGAR`` (solid), not
  the liquid ``SM`` subculture the amino acids are extracted from (issue #143);
- the build ledger: four YBR001C rows, one blank ORF and one ``WT`` row log exactly
  "2 usable ORFs, 2 non-systematic ORF names skipped, 3 repeated-ORF rows dropped (1
  ORFs kept at their first row)" (2026.10.01, issue #528: it used to count the one ORF,
  not the three rows);
- a blank concentration cell and a text cell (``n.d.``) each raise
  ``InvalidConcentrationError`` with an exact message (2026.10.01, issue #528: the blank
  used to be served as NaN and the text to stop in Python's ``float()``). The pinned
  Table S3 has no repeated ORF, blank or text cell, so no stored record changes;
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
import os
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
_RAW_ROWS: list[list[Any]] = [
    ["S000", "YAL001C", "01", *([10.0] * 19)],
    ["S001", "YBR001C", "01", *([20.0] * 19)],
    ["S002", "YCR001W", "02", *([30.0] * 19)],
    ["QC_001", None, "01", *([5.0] * 19)],
]


def _write_workbook(
    raw: Path,
    conc_columns: list[str] = _AA,
    summary_means: dict[str, float] = _SUMMARY_MEANS,
    conc_rows: list[list[Any]] = _CONC_ROWS,
    raw_rows: list[list[Any]] = _RAW_ROWS,
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
    # the released workbook's per-injection sheet (identifier, ORF, batch, uM)
    data_raw = workbook.create_sheet(m._RAW_SHEET)
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


def _experiment(
    orf: str, levels: list[float], n: dict[str, int] | None = None
) -> dict[str, Any]:
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
            n_replicates=n if n is not None else dict.fromkeys(_AA, 1),
            measurement_type="intracellular_concentration_mM",
            target_metabolite_ids=None,
        ),
    ).model_dump()


def _reference(means: dict[str, float], n_strains: int = 3) -> dict[str, Any]:
    return MetaboliteExperimentReference(
        dataset_name="AminoAcidMulleder2016Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=MetabolitePhenotype(
            metabolite_level=means,
            metabolite_level_se=None,
            n_replicates=dict.fromkeys(means, n_strains),
            measurement_type="intracellular_concentration_mM",
        ),
    ).model_dump()


def test_three_records_n_one_with_the_robust_mean_reference(
    dataset: m.AminoAcidMulleder2016Dataset,
) -> None:
    """Six rows give three records (YAL001C, the first YBR001C, the stripped YCR001W);
    the WT and malformed names are skipped and the second YBR001C is a collision. Each
    amino acid is n = 1 (one ``data_raw`` row each) with no SE on solid SM at 30 C; the
    reference is the summary mean with n = 3, the strains it summarizes.
    """
    assert len(dataset) == 3
    assert dataset[0]["experiment"] == _experiment("YAL001C", _LEVELS_YAL001C)
    assert dataset[1]["experiment"] == _experiment("YBR001C", [2.0] * 19)
    assert dataset[2]["experiment"] == _experiment("YCR001W", [1.0] * 19)
    assert dataset[0]["reference"] == _reference(_SUMMARY_MEANS)
    assert dataset[0]["publication"] == _PUBLICATION
    assert dataset[0]["experiment"]["environment"]["media"]["state"] == "solid"


def test_side_files(dataset: m.AminoAcidMulleder2016Dataset) -> None:
    """``data.csv`` is one row per kept ORF with the 19 concentrations and its
    ``data_raw`` row count; one reference covers all three records; the gene set is the
    three ORFs.
    """
    assert dataset.experiment_class is MetaboliteExperiment
    assert dataset.reference_class is MetaboliteExperimentReference
    preprocess = Path(dataset.root) / "preprocess"
    expected_rows = [
        ["YAL001C", *_LEVELS_YAL001C, 1],
        ["YBR001C", *([2.0] * 19), 1],
        ["YCR001W", *([1.0] * 19), 1],
    ]
    assert (preprocess / "data.csv").read_text() == (
        ",".join(["orf", *_AA, "n_raw_rows"])
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
    phenotype (with the strain count, n = 3) while no experiment measures it; the
    reference then has 20 keys against the experiment's 19.
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


def test_n_replicates_counts_the_data_raw_rows_per_amino_acid_issue_488(
    tmp_path: Path,
) -> None:
    """Contract (issue #488): ``n_replicates`` is, per amino acid, the number of
    ``data_raw`` rows of the strain's ORF with a value in that column (it was the
    constant 1). YAL001C has three raw rows (batches 01, 02, 12, the last the paper's
    repeat screen), one of them blank for tyrosine, so n = 3 on 18 keys and 2 on
    tyrosine; YBR001C and YCR001W have one row each; the ``QC_001`` row (no ORF) counts
    for nobody. ``data.csv`` keeps the row count (3). The reference n stays the strain
    count, 3. SE stays None and the stored value is the released mM, not a raw uM.
    """
    raw_rows = [
        [f"S{i:03d}", "YAL001C", batch, *([10.0 * (i + 1)] * 18), tyrosine]
        for i, (batch, tyrosine) in enumerate([("01", 1.0), ("02", None), ("12", 3.0)])
    ] + [
        ["S003", "YBR001C", "01", *([20.0] * 19)],
        ["S004", "YCR001W", "01", *([20.0] * 19)],
        ["QC_001", None, "01", *([5.0] * 19)],
    ]
    root = _root(tmp_path, raw_rows=raw_rows)
    dataset = m.AminoAcidMulleder2016Dataset(root=str(root))
    expected_n = {**dict.fromkeys(_AA, 3), "tyrosine": 2}
    assert dataset[0]["experiment"] == _experiment(
        "YAL001C", _LEVELS_YAL001C, n=expected_n
    )
    assert dataset[1]["experiment"] == _experiment("YBR001C", [2.0] * 19)
    assert dataset[2]["experiment"] == _experiment("YCR001W", [1.0] * 19)
    assert dataset[0]["reference"] == _reference(_SUMMARY_MEANS, n_strains=3)
    rows = (root / "preprocess" / "data.csv").read_text().splitlines()
    assert [row.split(",")[-1] for row in rows] == ["n_raw_rows", "3", "1", "1"]


def test_a_released_orf_without_a_data_raw_row_is_refused(tmp_path: Path) -> None:
    """Contract (issue #488): a released ORF with no ``data_raw`` row has no count, so
    the build raises ``MissingRawMeasurementError`` naming it and writes no LMDB (the
    pinned workbook has 0 such ORFs).
    """
    raw_rows = [r for r in _RAW_ROWS if r[1] != "YCR001W"]
    root = _root(tmp_path, raw_rows=raw_rows)
    with pytest.raises(m.MissingRawMeasurementError) as excinfo:
        m.AminoAcidMulleder2016Dataset(root=str(root))
    assert str(excinfo.value) == (
        "Mulleder Table S3: released ORF YCR001W has no data_raw row"
    )
    assert not (root / "processed" / "lmdb").exists()


def test_an_amino_acid_blank_in_every_raw_row_is_refused(tmp_path: Path) -> None:
    """Contract (issue #488): a released ORF whose only ``data_raw`` row is blank for
    alanine has n = 0 there, which the schema cannot hold; the build raises
    ``MissingRawMeasurementError`` naming the ORF and the amino acid (0 such cells on the
    pinned workbook).
    """
    raw_rows = [
        ["S000", "YAL001C", "01", None, *([10.0] * 18)],
        *[r for r in _RAW_ROWS if r[1] != "YAL001C"],
    ]
    root = _root(tmp_path, raw_rows=raw_rows)
    with pytest.raises(m.MissingRawMeasurementError) as excinfo:
        m.AminoAcidMulleder2016Dataset(root=str(root))
    assert str(excinfo.value) == (
        "Mulleder Table S3: released ORF YAL001C has no data_raw value for ['alanine']"
    )


def test_reference_n_is_the_number_of_strains_the_robust_mean_summarizes_issue_489(
    tmp_path: Path,
) -> None:
    """Contract (issue #489): the reference phenotype is the ``robust_summary_statistics``
    ``mean (mM)`` column, the Minimum Covariance Determinant mean over the profiled
    strains (0.25 * (i + 1) mM for amino acid i here: alanine 0.25, tyrosine 4.75), and
    its ``n_replicates`` is the number of released strains counted from the concentration
    sheet on every key (it was 1). Four distinct ORFs here, so n = 4 on all 19 keys and
    the same reference on every record; SE stays None.
    """
    rows: list[list[Any]] = [
        [orf, *([1.0] * 19)] for orf in ["YAL001C", "YBR001C", "YCR001W", "YDR001C"]
    ]
    raw_rows: list[list[Any]] = [
        [f"S{i:03d}", row[0], "01", *([1.0] * 19)] for i, row in enumerate(rows)
    ]
    root = _root(tmp_path, conc_rows=rows, raw_rows=raw_rows)
    dataset = m.AminoAcidMulleder2016Dataset(root=str(root))
    for index in range(4):
        reference = dataset[index]["reference"]["phenotype_reference"]
        assert reference["metabolite_level"] == _SUMMARY_MEANS
        assert reference["metabolite_level"]["alanine"] == 0.25
        assert reference["metabolite_level"]["tyrosine"] == 4.75
        assert reference["n_replicates"] == dict.fromkeys(_AA, 4)
        assert reference["metabolite_level_se"] is None
        assert reference["measurement_type"] == "intracellular_concentration_mM"


def test_every_sourced_value_quote_is_verbatim_in_the_mirrored_paper() -> None:
    """The MCD estimator (line 387), the 4,678 profiled strains (line 395) and the 479
    strain repeat screen (line 341) are verbatim in the sha256-pinned ``paper.md``;
    skipped where the literature mirror is not mounted.
    """
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    paper = Path(data_root) / "torchcell-library" / m.CITATION_KEY / m.PAPER_MD
    if not paper.exists():
        pytest.skip("the literature mirror is not mounted on this machine")
    assert hashlib.sha256(paper.read_bytes()).hexdigest() == m.PAPER_MD_SHA256
    text = paper.read_text()
    assert sorted(m.SOURCED_VALUES) == [
        "n_profiled_strains",
        "reference_estimator",
        "repeat_screen",
    ]
    for key, sourced in m.SOURCED_VALUES.items():
        assert sourced.quote in text, key
        assert sourced.provenance.sha256 == m.PAPER_MD_SHA256
    assert m.SOURCED_VALUES["n_profiled_strains"].value == 4678
    assert m.SOURCED_VALUES["repeat_screen"].value == 479


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


def test_ledger_counts_dropped_rows_and_their_orfs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Six rows: YBR001C four times (2.0, 3.0, 4.0, 5.0), a blank ORF cell (read as NaN,
    stringified to ``"nan"``), ``WT``, and YAL001C. Kept: YBR001C (the first row, 2.0 on
    every key) and YAL001C, so two records. The ledger line counts the blank and ``WT``
    as two non-systematic names.

    Contract (issue #528): the ledger counts the three discarded YBR001C rows and the
    one ORF they belong to (it used to report only "1 ORF collisions deduped").
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
        "3 repeated-ORF rows dropped (1 ORFs kept at their first row)",
        "Mulleder: data_raw rows per released ORF {1: 2}; reference n_replicates 2 "
        "(strains the robust mean summarizes)",
        "Wrote 2 Mulleder amino-acid experiments to LMDB",
    ]


def test_blank_concentration_refuses_the_build_by_name(tmp_path: Path) -> None:
    """Contract (issue #528): a blank tyrosine cell raises
    ``InvalidConcentrationError`` naming the ORF, the amino acid and the cell, and
    nothing is written to LMDB (it used to be served as ``nan`` with ``n_replicates``
    1).
    """
    rows: list[list[Any]] = [["YAL001C", *_LEVELS_YAL001C[:-1], None]]
    root = _root(tmp_path, conc_rows=rows)
    with pytest.raises(m.InvalidConcentrationError) as excinfo:
        m.AminoAcidMulleder2016Dataset(root=str(root))
    assert str(excinfo.value) == (
        "Mulleder Table S3: YAL001C tyrosine concentration nan is not a finite number"
    )
    assert not (root / "processed" / "lmdb").exists()


def test_text_concentration_refuses_the_build_by_name(tmp_path: Path) -> None:
    """Contract (issue #528): a non-numeric cell raises ``InvalidConcentrationError``
    with the same message shape as a blank one, quoting the cell text (it used to stop
    with Python's ``could not convert string to float``); nothing is written to LMDB.
    """
    rows: list[list[Any]] = [["YAL001C", "n.d.", *_LEVELS_YAL001C[1:]]]
    root = _root(tmp_path, conc_rows=rows)
    with pytest.raises(m.InvalidConcentrationError) as excinfo:
        m.AminoAcidMulleder2016Dataset(root=str(root))
    assert str(excinfo.value) == (
        "Mulleder Table S3: YAL001C alanine concentration 'n.d.' is not a finite number"
    )
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
