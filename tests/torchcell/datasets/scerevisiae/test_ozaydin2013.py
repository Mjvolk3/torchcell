# tests/torchcell/datasets/scerevisiae/test_ozaydin2013.py
# [[tests.torchcell.datasets.scerevisiae.test_ozaydin2013]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_ozaydin2013.py
"""Build-smoke test for the Ozaydin 2013 beta-carotene visual-screen dataset.

Builds the dataset into a tmp root from the sha256-pinned library-mirror SI (no
network), then asserts record count, schema round-trip, the engineered genotype
(single KO + 3-gene carotenogenic cassette), synthetic media, and publication id.
Skipped unless ``--data --slow`` are given and the ``$DATA_ROOT`` mirror is present.

2026.09.30 (Phase 14). The smoke test's marks moved onto the test itself (the module
used to skip every test at import when the mirror was absent, and called
``load_dotenv()`` at import, which cannot change the sentinel ``DATA_ROOT`` the root
conftest sets first). Added hermetic tests on synthetic SI workbooks under
``tmp_path``; the per-row aggregation, the comment flags and the reference strain are
pinned in ``test_ozaydin2013_synthetic.py`` and not repeated here. Added:

- ``download()`` on a faked ``urlopen``: the request (URL, User-Agent, timeout 120);
  a 9,999-byte body refused as too small and a 10,000-byte body passing the size check
  and refused on sha256, each with its exact message and no file written; the write
  when the pin matches; a present file whose digest matches the pin returns without a
  request;
- the build ledger lines (6 ORFs total, 5 usable, 1 text-only, 1 malformed name) and
  the absence of the malformed line when every name is systematic;
- aggregation: the comment flags are OR-ed across replicates, and two equal scores
  give ``visual_score_min`` equal to the score (the schema documents None only for a
  single replicate);
- 2026.10.01 (issue #528): the record's strain is the strain its numeric scores were
  taken on, and numeric scores on two strains raise ``MixedStrainScoresError`` (the
  first row's strain used to win). On the pinned SI the per-ORF aggregate is
  identical before and after (4,975 ORFs);
- an out-of-scale color (6) refuses the build with the schema's message;
- the medium as a Finding: a free-text ``SC-URA`` stub with no components, not the
  sourced ``media.SC_URA`` of ``MEDIA_LIBRARY``;
- the YB/I/BTS1 cassette written out field by field in stored order (the sibling file
  compares against ``_carotenogenic_cassette()`` itself), with ``plasmid_contig_id``,
  ``locus_tag`` and ``integration_locus`` None as a Finding (no plasmid sequence store);
- ``main()`` under a stubbed ``load_dotenv`` and a ``tmp_path`` ``DATA_ROOT``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import os.path as osp
import re
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import Genotype, VisualScoreExperiment
from torchcell.datasets.scerevisiae import ozaydin2013 as m

_MIRROR_SI_REL = osp.join(
    "torchcell-library/ozaydinCarotenoidbasedPhenotypicScreen2013a/si",
    "1-s2.0-S109671761200081X-mmc1.xlsx",
)


@pytest.mark.data
@pytest.mark.slow
def test_ozaydin_build_smoke(tmp_path: Path) -> None:
    """Ozaydin dataset builds from the mirror and produces schema-valid records."""
    mirror_si = osp.join(os.environ["DATA_ROOT"], _MIRROR_SI_REL)
    if not osp.exists(mirror_si):
        pytest.skip(f"requires Ozaydin SI mirror at {mirror_si}")

    root = tmp_path / "carotenoid_ozaydin2013"
    (root / "raw").mkdir(parents=True)
    shutil.copy(mirror_si, root / "raw" / "1-s2.0-S109671761200081X-mmc1.xlsx")

    dataset = m.CarotenoidOzaydin2013Dataset(root=str(root))
    assert len(dataset) == 4474

    # dataset[i] returns the stored dicts; validating through the schema IS the
    # round-trip test (the stale on-disk LMDB failed exactly here on media.is_synthetic).
    record = dataset[0]
    exp = VisualScoreExperiment.model_validate(record["experiment"])
    dumped = exp.model_dump()
    assert type(exp).model_validate(dumped).model_dump() == dumped

    # genotype: one KO deletion + the 3-gene YB/I/BTS1 cassette
    assert isinstance(exp.genotype, Genotype)
    ptypes = Counter(p.perturbation_type for p in exp.genotype.perturbations)
    assert ptypes == {"kanmx_deletion": 1, "gene_addition": 3}

    # synthetic medium (the field whose absence made the stale on-disk LMDB fail)
    assert exp.environment.media.is_synthetic is True

    assert record["publication"]["pubmed_id"] == "22918085"


# --- hermetic tests on a synthetic SI ------------------------------------------ #

Row = tuple[str | None, str | None, Any, str | None]

_ROWS: list[Row] = [
    ("YAL001C", "BY4741", 2, None),
    ("YBR001C", "BY4741", "_", None),
    ("YCR001W", "BY4741", -3, None),
    ("YLR287-A", "BY4741", 1, None),
    ("YDR001C", "BY4741", 0, None),
    ("YER001W", "BY4741", 4, None),
    ("YFR001W", "BY4741", -1, None),
]


def _write_si(raw: Path, rows: list[Row]) -> Path:
    """The two released sheets with their released headers; TOP200 is empty."""
    workbook = openpyxl.Workbook()
    screen = workbook.active
    screen.title = "Color scores of all deletions"
    screen.append(["ORF name", "Strain", "Color", "Comment"])
    for row in rows:
        screen.append(list(row))
    top200 = workbook.create_sheet("Names and Functions of TOP200")
    top200.append(["ORF name", "Gene name", "Gene Function", "Assigned Category"])
    path = raw / m.CarotenoidOzaydin2013Dataset.si_filename
    workbook.save(path)
    return path


def _root(tmp_path: Path, rows: list[Row] = _ROWS) -> Path:
    root = tmp_path / "carotenoid_ozaydin2013"
    (root / "raw").mkdir(parents=True)
    _write_si(root / "raw", rows)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> m.CarotenoidOzaydin2013Dataset:
    return m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path)))


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


def _si_path(dataset: m.CarotenoidOzaydin2013Dataset) -> Path:
    return Path(dataset.root) / "raw" / m.CarotenoidOzaydin2013Dataset.si_filename


def test_download_refuses_a_body_under_ten_thousand_bytes(
    dataset: m.CarotenoidOzaydin2013Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the SI absent, one request goes to ``si_url`` with a ``Mozilla/5.0``
    User-Agent and a 120 s timeout; a 9,999-byte body is one byte under the floor
    (``len(data) < 10000``) and is refused before hashing, writing nothing.
    """
    dest = _si_path(dataset)
    dest.unlink()
    calls = _fake_urlopen(monkeypatch, b"x" * 9999)
    with pytest.raises(
        RuntimeError, match=re.escape("Ozaydin SI download too small: 9999 bytes")
    ):
        dataset.download()
    assert not dest.exists()
    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == (
        "https://ars.els-cdn.com/content/image/1-s2.0-S109671761200081X-mmc1.xlsx"
    )
    assert request.header_items() == [("User-agent", "Mozilla/5.0")]
    assert timeout == 120


def test_download_refuses_a_ten_thousand_byte_body_with_the_wrong_sha256(
    dataset: m.CarotenoidOzaydin2013Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exactly 10,000 bytes passes the size floor and fails the pin with
    ``RawSha256MismatchError`` naming the SI URL and both digests, writing nothing.
    """
    dest = _si_path(dataset)
    dest.unlink()
    payload = b"y" * 10000
    _fake_urlopen(monkeypatch, payload)
    digest = hashlib.sha256(payload).hexdigest()
    with pytest.raises(RawSha256MismatchError) as err:
        dataset.download()
    assert str(err.value) == (
        f"sha256 mismatch for {dataset.si_url}: expected {m._SI_SHA256}, "
        f"observed {digest}"
    )
    assert list(dest.parent.iterdir()) == []


def test_download_writes_a_body_matching_the_pin(
    dataset: m.CarotenoidOzaydin2013Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the pin set to the body's digest the bytes land unchanged at ``raw/``."""
    dest = _si_path(dataset)
    dest.unlink()
    payload = b"z" * 12345
    monkeypatch.setattr(m, "_SI_SHA256", hashlib.sha256(payload).hexdigest())
    calls = _fake_urlopen(monkeypatch, payload)
    dataset.download()
    assert dest.read_bytes() == payload
    assert len(calls) == 1


def test_download_accepts_a_present_file_matching_the_pin(
    dataset: m.CarotenoidOzaydin2013Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present SI whose digest equals the pin returns without a request (the fake
    records none) and is left byte for byte.
    """
    dest = _si_path(dataset)
    before = dest.read_bytes()
    monkeypatch.setattr(m, "_SI_SHA256", hashlib.sha256(before).hexdigest())
    calls = _fake_urlopen(monkeypatch, b"never served")
    dataset.download()
    assert calls == []
    assert dest.read_bytes() == before


def test_build_ledger_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Seven rows: YLR287-A is malformed (no W/C) and excluded by name; YBR001C has only
    the text score ``_``. The aggregate therefore holds 6 ORFs, 5 with a numeric color,
    1 excluded as text-only; the malformed line lists the name. A second build whose
    names are all systematic logs no malformed line.
    """
    caplog.set_level(logging.INFO, logger=m.__name__)
    dataset = m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path / "a")))
    assert len(dataset) == 5
    messages = [r.getMessage() for r in caplog.records if r.name == m.__name__]
    assert messages == [
        "Ozaydin: excluded 1 malformed ORF names (e.g. ['YLR287-A'])",
        "Ozaydin: 6 ORFs total, 5 usable (numeric color), 1 excluded (text-only)",
        "Wrote 5 Ozaydin visual-score experiments to LMDB",
    ]

    caplog.clear()
    clean = [row for row in _ROWS if row[0] != "YLR287-A"]
    m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path / "b", clean)))
    messages = [r.getMessage() for r in caplog.records if r.name == m.__name__]
    assert messages == [
        "Ozaydin: 6 ORFs total, 5 usable (numeric color), 1 excluded (text-only)",
        "Wrote 5 Ozaydin visual-score experiments to LMDB",
    ]


def test_replicates_on_one_strain_aggregate_and_or_the_flags(tmp_path: Path) -> None:
    """YAL001C scored twice on BY4741 (comment ``petite``, color 1; then ``tiny``, color
    3): the aggregate is max 3.0, min 1.0, n 2, flags petite and tiny, on BY4741.
    """
    rows: list[Row] = [
        ("YAL001C", "BY4741", 1, "petite"),
        ("YAL001C", "BY4741", 3, "tiny"),
    ]
    dataset = m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path, rows)))
    assert len(dataset) == 1
    phenotype = dataset[0]["experiment"]["phenotype"]
    assert (
        phenotype["visual_score"],
        phenotype["visual_score_min"],
        phenotype["n_replicates"],
    ) == (3.0, 1.0, 2)
    assert {k for k, v in phenotype["comment_annotations"].items() if v} == {
        "flag_petite",
        "flag_tiny",
    }
    assert dataset[0]["reference"]["genome_reference"]["strain"] == "BY4741"


def test_numeric_scores_on_two_strains_refuse_the_build_by_name(tmp_path: Path) -> None:
    """Contract (issue #528): YAL001C scored 1 on BY4730 and 3 on BY4741 raises
    ``MixedStrainScoresError`` naming the ORF and both strains, and nothing is written
    to LMDB (the first row's strain, BY4730, used to label a maximum scored on BY4741).
    """
    rows: list[Row] = [
        ("YAL001C", "BY4730", 1, "petite"),
        ("YAL001C", "BY4741", 3, "tiny"),
    ]
    root = _root(tmp_path, rows)
    with pytest.raises(m.MixedStrainScoresError) as excinfo:
        m.CarotenoidOzaydin2013Dataset(root=str(root))
    assert str(excinfo.value) == (
        "Ozaydin: YAL001C has numeric color scores on strains ['BY4730', 'BY4741']; "
        "one record carries one reference genome"
    )
    assert not (root / "processed" / "lmdb").exists()


def test_the_strain_is_the_one_the_score_was_taken_on(tmp_path: Path) -> None:
    """Contract (issue #528): a text-only BY4741 row (``pet``) before a BY4730 row
    scored 1 gives a record on BY4730 with score 1.0 and text ``pet`` (the first row's
    strain, BY4741, used to win). The released SI lists YML086C this way with the
    BY4730 row first, so its stored strain is BY4730 either way.
    """
    rows: list[Row] = [
        ("YAL001C", "BY4741", "pet", None),
        ("YAL001C", "BY4730", 1, None),
    ]
    dataset = m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path, rows)))
    assert len(dataset) == 1
    assert dataset[0]["reference"]["genome_reference"]["strain"] == "BY4730"
    phenotype = dataset[0]["experiment"]["phenotype"]
    assert (phenotype["visual_score"], phenotype["n_replicates"]) == (1.0, 1)
    assert phenotype["score_text"] == "pet"


def test_equal_replicates_report_a_min_equal_to_the_score(tmp_path: Path) -> None:
    """Two plates both scored 2: ``visual_score_min`` is 2.0, not None. The field is None
    only for a single numeric replicate (``len(scores) > 1``), whatever the spread.
    """
    rows: list[Row] = [("YAL001C", "BY4741", 2, None), ("YAL001C", "BY4741", 2, None)]
    dataset = m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path, rows)))
    phenotype = dataset[0]["experiment"]["phenotype"]
    assert (
        phenotype["visual_score"],
        phenotype["visual_score_min"],
        phenotype["n_replicates"],
    ) == (2.0, 2.0, 2)


def test_out_of_scale_color_refuses_the_build(tmp_path: Path) -> None:
    """A color of 6 is outside the declared -5..5 scale; ``VisualScorePhenotype``
    raises inside ``create_experiment`` and the build stops with the schema message.
    """
    rows: list[Row] = [("YAL001C", "BY4741", 6, None)]
    with pytest.raises(
        ValueError, match=re.escape("visual_score 6.0 outside scale [-5, 5]")
    ):
        m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path, rows)))


def test_medium_is_a_free_text_stub_not_the_library_sc_ura(
    dataset: m.CarotenoidOzaydin2013Dataset,
) -> None:
    """Finding: the screen medium is built inline as ``Media(name="SC-URA",
    state="solid", is_synthetic=True)`` (source line 361), a stub with no base medium,
    components, dropouts or provenance, while ``MEDIA_LIBRARY`` holds a sourced
    ``SC_URA`` (31 components, a uracil dropout). The stub's name matches no library
    entry, so the record cannot be joined to it. Record and reference carry the same
    stub at 30 C. Pinned until the loader uses a sourced SC-URA agar object.
    """
    stub = {
        "name": "SC-URA",
        "state": "solid",
        "is_synthetic": True,
        "base_medium": None,
        "components": [],
        "dropouts": [],
        "provenance": [],
    }
    experiment_env = dataset[0]["experiment"]["environment"]
    assert experiment_env["media"] == stub
    assert dataset[0]["reference"]["environment_reference"]["media"] == stub
    assert experiment_env["temperature"]["value"] == 30.0
    assert "SC-URA" not in {media.name for media in MEDIA_LIBRARY.values()}


def test_cassette_is_three_episomal_additions_without_a_sequence_pointer(
    dataset: m.CarotenoidOzaydin2013Dataset,
) -> None:
    """Record 0's genotype is the YAL001C deletion plus BTS1 = YPL069C (a native extra
    copy) and crtI and crtYB (heterologous, from *Xanthophyllomyces dendrorhous*), all
    ``episomal_2micron`` on construct ``YB/I/BTS1``, written out here field by field.
    ``Genotype`` stores perturbations sorted by systematic name, uppercase before
    lowercase, so the order is YAL001C, YPL069C, crtI, crtYB.

    Finding: ``plasmid_contig_id``, ``locus_tag`` and ``integration_locus`` are None on
    all three additions; the plasmid is physical-only (Euroscarf P30796) and the
    plasmid sequence store has not landed. Pinned until it does.
    """
    perturbations = dataset[0]["experiment"]["genotype"]["perturbations"]
    assert [
        (p["systematic_gene_name"], p["perturbation_type"]) for p in perturbations
    ] == [
        ("YAL001C", "kanmx_deletion"),
        ("YPL069C", "gene_addition"),
        ("crtI", "gene_addition"),
        ("crtYB", "gene_addition"),
    ]
    assert [
        (
            p["perturbed_gene_name"],
            p["source_organism"],
            p["is_heterologous"],
            p["localization"],
            p["construct_name"],
            p["plasmid_contig_id"],
            p["locus_tag"],
            p["integration_locus"],
        )
        for p in perturbations[1:]
    ] == [
        (
            "BTS1",
            "Saccharomyces cerevisiae",
            False,
            "episomal_2micron",
            "YB/I/BTS1",
            None,
            None,
            None,
        ),
        (
            "crtI",
            "Xanthophyllomyces dendrorhous",
            True,
            "episomal_2micron",
            "YB/I/BTS1",
            None,
            None,
            None,
        ),
        (
            "crtYB",
            "Xanthophyllomyces dendrorhous",
            True,
            "episomal_2micron",
            "YB/I/BTS1",
            None,
            None,
            None,
        ),
    ]


def test_main_prints_length_and_first_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main()`` opens ``$DATA_ROOT/data/torchcell/carotenoid_ozaydin2013`` and prints
    ``len = 5`` then record 0. The store is built first (its progress output is
    discarded), ``load_dotenv`` is stubbed and ``DATA_ROOT`` is ``tmp_path``.
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    root = _root(tmp_path / "data" / "torchcell")
    first = m.CarotenoidOzaydin2013Dataset(root=str(root))[0]
    capsys.readouterr()
    m.main()
    assert capsys.readouterr().out == f"len = 5\n{first}\n"


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): with the SI already in ``raw/`` PyG skips
    ``download()``, so ``process()`` verifies it against ``_SI_SHA256`` first and raises
    ``RawSha256MismatchError`` naming it and both digests before a row is read; no store
    is written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(m, [m.CarotenoidOzaydin2013Dataset.si_filename])
    raw = staged.root / "raw" / m.CarotenoidOzaydin2013Dataset.si_filename
    with pytest.raises(RawSha256MismatchError) as err:
        m.CarotenoidOzaydin2013Dataset(root=str(staged.root))
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected "
        "4818726e352ead3cb739fd9becf08a0c04d14b8a8761732184214344447507f0, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed
