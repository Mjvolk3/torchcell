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
- the build ledger line (7 released rows, 5 records, 1 text-only row and 1 row
  deferred to the released-identifier leaf) and ``dropped_records.json``;
- one record per released row (2026.10.12, #1045): two plates of one ORF are two
  records with their own scores, flags and strains, never a maximum; a text-only row
  is listed in the ledger with its text, and the strain is the row's own cell;
- an out-of-scale color (6) refuses the build with the schema's message;
- the medium (issue #622, 2026.10.02): the sourced ``OZAYDIN_SC_URA_AGAR`` plate (the
  library ``SC_URA`` made solid, agar amount an open gap), no longer a free-text stub;
- the YB/I/BTS1 cassette written out field by field in stored order (the sibling file
  compares against ``_carotenogenic_cassette()`` itself), with ``placement`` a typed
  ``ProvenanceGap`` and ``integration_locus`` None (P4, 2026.10.15: the paper deposits
  no map and names a JBEI registry handle, so the absence is stated, not silent);
- ``main()`` under a stubbed ``load_dotenv`` and a ``tmp_path`` ``DATA_ROOT``.
"""

from __future__ import annotations

import hashlib
import io
import json
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
#: The five released position cells of one row: record no., Batch, Plate, Row, Col.
Position = tuple[int, str, int, str, int]

_ROWS: list[Row] = [
    ("YAL001C", "BY4741", 2, None),
    ("YBR001C", "BY4741", "_", None),
    ("YCR001W", "BY4741", -3, None),
    ("YLR287-A", "BY4741", 1, None),
    ("YDR001C", "BY4741", 0, None),
    ("YER001W", "BY4741", 4, None),
    ("YFR001W", "BY4741", -1, None),
]


def _write_si(
    raw: Path, rows: list[Row], positions: list[Position] | None = None
) -> Path:
    """The two released sheets with their released headers; TOP200 is empty.

    Every row carries the sheet's nine columns. ``positions`` gives the five position
    cells per row; by default row ``i`` is record ``338 + i`` of batch ``01_1`` at plate
    1, row A, column ``i + 1``, one distinct position per row as the release has.
    """
    workbook = openpyxl.Workbook()
    screen = workbook.active
    screen.title = "Color scores of all deletions"
    screen.append(
        [
            "record no.",
            "ORF name",
            "Strain",
            "Batch",
            "Plate",
            "Row",
            "Col",
            "Comment",
            "Color",
        ]
    )
    if positions is None:
        positions = [(338 + i, "01_1", 1, "A", i + 1) for i in range(len(rows))]
    for (orf, strain, color, comment), (record, batch, plate, prow, col) in zip(
        rows, positions, strict=True
    ):
        screen.append([record, orf, strain, batch, plate, prow, col, comment, color])
    top200 = workbook.create_sheet("Names and Functions of TOP200")
    top200.append(["ORF name", "Gene name", "Gene Function", "Assigned Category"])
    path = raw / m.CarotenoidOzaydin2013Dataset.si_filename
    workbook.save(path)
    return path


def _root(
    tmp_path: Path, rows: list[Row] = _ROWS, positions: list[Position] | None = None
) -> Path:
    root = tmp_path / "carotenoid_ozaydin2013"
    (root / "raw").mkdir(parents=True)
    _write_si(root / "raw", rows, positions)
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
    """Seven rows: YLR287-A is not ORF-shaped and is stored on the released-gene-name
    leaf (#1045); YBR001C has only the text score ``_`` and is listed in the ledger. Six
    rows are records. ``dropped_records.json`` carries both lists verbatim.
    """
    caplog.set_level(logging.INFO, logger=m.__name__)
    dataset = m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path / "a")))
    assert len(dataset) == 6
    messages = [r.getMessage() for r in caplog.records if r.name == m.__name__]
    assert messages == [
        "Ozaydin: 7 released rows, 6 records (one per scored row); 1 text-only rows "
        "listed in the ledger, 1 names on the released-identifier leaf",
        "Wrote 6 Ozaydin visual-score experiments to LMDB",
    ]
    ledger = json.loads(
        (Path(dataset.root) / "preprocess" / "dropped_records.json").read_text()
    )
    assert (ledger["source_rows"], ledger["kept_records"]) == (7, 6)
    assert [(r["rule"], r["n_records"]) for r in ledger["rules"]] == [
        ("orf_cell_is_blank", 0),
        ("stored_on_released_gene_name_leaf", 0),
        ("color_cell_is_text_not_a_score", 1),
    ]
    assert ledger["rules"][1]["items"] == ["YLR287-A"]
    assert ledger["rules"][2]["items"] == [
        {
            "orf": "YBR001C",
            "strain": "BY4741",
            "text": "_",
            "record_no": "339",
            "batch": "01_1",
            "plate": "1",
            "row": "A",
            "col": "2",
        }
    ]


def test_two_plates_of_one_orf_are_two_records_with_their_own_flags(
    tmp_path: Path,
) -> None:
    """YAL001C scored twice on BY4741 (comment ``petite``, color 1; then ``tiny``, color
    3): two records in row order, each n = 1 with no minimum, each carrying its own
    row's flag (the previous build stored one record at the maximum with both flags).
    """
    rows: list[Row] = [
        ("YAL001C", "BY4741", 1, "petite"),
        ("YAL001C", "BY4741", 3, "tiny"),
    ]
    dataset = m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path, rows)))
    assert len(dataset) == 2
    got = []
    for i in range(2):
        phenotype = dataset[i]["experiment"]["phenotype"]
        got.append(
            (
                phenotype["visual_score"],
                phenotype["visual_score_min"],
                phenotype["n_replicates"],
                {k for k, v in phenotype["comment_annotations"].items() if v},
            )
        )
    assert got == [(1.0, None, 1, {"flag_petite"}), (3.0, None, 1, {"flag_tiny"})]
    assert dataset[0]["reference"]["genome_reference"]["strain"] == "BY4741"


def test_scores_on_two_strains_are_two_records_each_on_its_own_strain(
    tmp_path: Path,
) -> None:
    """YAL001C scored 1 on BY4730 and 3 on BY4741: two records, each referenced to the
    strain its row names (the previous build refused the pair by name).
    """
    rows: list[Row] = [
        ("YAL001C", "BY4730", 1, "petite"),
        ("YAL001C", "BY4741", 3, "tiny"),
    ]
    dataset = m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path, rows)))
    assert [
        dataset[i]["reference"]["genome_reference"]["strain"] for i in range(2)
    ] == ["BY4730", "BY4741"]


def test_a_text_only_row_is_ledgered_and_the_scored_row_keeps_its_strain(
    tmp_path: Path,
) -> None:
    """A text-only BY4741 row (``pet``) before a BY4730 row scored 1: one record on
    BY4730 with score 1.0 and no text; the ``pet`` row is in the ledger with its ORF and
    strain.
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
    assert phenotype["score_text"] is None
    ledger = json.loads(
        (Path(dataset.root) / "preprocess" / "dropped_records.json").read_text()
    )
    assert ledger["rules"][2]["items"] == [
        {
            "orf": "YAL001C",
            "strain": "BY4741",
            "text": "pet",
            "record_no": "338",
            "batch": "01_1",
            "plate": "1",
            "row": "A",
            "col": "1",
        }
    ]


def test_equal_scores_on_two_plates_are_two_records(tmp_path: Path) -> None:
    """Two plates both scored 2: two records, score 2.0 each, no minimum, n = 1."""
    rows: list[Row] = [("YAL001C", "BY4741", 2, None), ("YAL001C", "BY4741", 2, None)]
    dataset = m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path, rows)))
    assert len(dataset) == 2
    for i in range(2):
        phenotype = dataset[i]["experiment"]["phenotype"]
        assert (
            phenotype["visual_score"],
            phenotype["visual_score_min"],
            phenotype["n_replicates"],
        ) == (2.0, None, 1)


def test_out_of_scale_color_refuses_the_build(tmp_path: Path) -> None:
    """A color of 6 is outside the declared -5..5 scale; ``VisualScorePhenotype``
    raises inside ``create_experiment`` and the build stops with the schema message.
    """
    rows: list[Row] = [("YAL001C", "BY4741", 6, None)]
    with pytest.raises(
        ValueError, match=re.escape("visual_score 6.0 outside scale [-5, 5]")
    ):
        m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path, rows)))


def test_medium_is_the_sourced_sc_ura_agar_plate(
    dataset: m.CarotenoidOzaydin2013Dataset,
) -> None:
    """Issue #622: the screen medium was a free-text ``Media(name="SC-URA",
    state="solid", is_synthetic=True)`` stub with no components or provenance. It is now
    ``OZAYDIN_SC_URA_AGAR``: the library ``SC_URA`` (base ``SC``, uracil dropped) made
    solid, plus an agar row with NO concentration (the paper prints no agar amount),
    quoting Methods line 42. Record and reference carry the same medium at 30 C.
    """
    media = m.OZAYDIN_SC_URA_AGAR
    experiment_env = dataset[0]["experiment"]["environment"]
    assert experiment_env["media"] == media.model_dump()
    assert dataset[0]["reference"]["environment_reference"]["media"] == (
        media.model_dump()
    )
    assert experiment_env["temperature"]["value"] == 30.0
    assert (media.state, media.is_synthetic, media.base_medium) == ("solid", True, "SC")
    assert [c.name for c in media.dropouts] == ["uracil"]
    assert len(media.components) == len(MEDIA_LIBRARY["SC_URA"].components) + 1
    agar = media.components[-1]
    assert (agar.compound.name, agar.concentration) == ("agar", None)
    assert "agar" in media.open_gaps
    assert media.provenance[-1].quote == m.SCORING_PLATE_QUOTE
    assert agar.provenance[0].quote == (
        "spotted onto selective medium agar plates (SC-URA)"
    )


def test_cassette_is_three_episomal_additions_without_a_sequence_pointer(
    dataset: m.CarotenoidOzaydin2013Dataset,
) -> None:
    """Record 0's genotype is the YAL001C deletion plus BTS1 = YPL069C (a native extra
    copy) and crtI and crtYB (heterologous, from *Xanthophyllomyces dendrorhous*), all
    on construct ``YB/I/BTS1``, written out here field by field. ``Genotype`` stores
    perturbations sorted by systematic name, uppercase before lowercase, so the order is
    YAL001C, YPL069C, crtI, crtYB.

    P4 (2026.10.15): ``localization`` is the typed ``episomal_replicon`` and the paper's
    own word for the backbone, ``YEplac195``, is the verbatim ``replicon_statement``;
    the former ``episomal_2micron`` token asserted a replicon class this paper never
    writes. ``placement`` is ``None`` beside a ``ProvenanceGap`` reading
    ``not_released_by_primary``: the plasmid is a JBEI registry deposit with no released
    sequence, so the absence is a counted statement rather than a silent None.
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
            p["replicon_statement"],
            p["placement"],
            [gap["field"] for gap in p["provenance_gaps"]],
            p["integration_locus"],
        )
        for p in perturbations[1:]
    ] == [
        (
            "BTS1",
            "Saccharomyces cerevisiae",
            False,
            "episomal_replicon",
            "YB/I/BTS1",
            "YEplac195",
            None,
            ["placement"],
            None,
        ),
        (
            "crtI",
            "Xanthophyllomyces dendrorhous",
            True,
            "episomal_replicon",
            "YB/I/BTS1",
            "YEplac195",
            None,
            ["placement"],
            None,
        ),
        (
            "crtYB",
            "Xanthophyllomyces dendrorhous",
            True,
            "episomal_replicon",
            "YB/I/BTS1",
            "YEplac195",
            None,
            ["placement"],
            None,
        ),
    ]


def test_main_prints_length_and_first_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main()`` opens ``$DATA_ROOT/data/torchcell/carotenoid_ozaydin2013`` and prints
    ``len = 6`` then record 0. The store is built first (its progress output is
    discarded), ``load_dotenv`` is stubbed and ``DATA_ROOT`` is ``tmp_path``.
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    root = _root(tmp_path / "data" / "torchcell")
    first = m.CarotenoidOzaydin2013Dataset(root=str(root))[0]
    capsys.readouterr()
    m.main()
    assert capsys.readouterr().out == f"len = 6\n{first}\n"


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


def test_identical_scores_at_two_collection_positions_are_two_distinct_records(
    tmp_path: Path,
) -> None:
    """Two released rows of YAL049C that agree in strain, color and comment and differ
    only in the collection position (record 356, batch ``01_1``, plate 1, well B3; record
    7729, batch ``chr00_23``, plate 75, well A1; the real pair of the release) are two
    records whose experiment dumps differ, each carrying its own row's position as the
    deletion's ``construction``. Before lane Q (2026-10-15) the record carried none of
    the five cells, the two dumps were byte-identical, and the graph, whose Experiment
    id is the content hash, kept one node for the two measurements: 31 such pairs in
    the released sheet, 4,576 nodes for 4,607 records. The regression is the
    inequality of the two digests.
    """
    rows: list[Row] = [("YAL049C", "BY4741", 0, None), ("YAL049C", "BY4741", 0, None)]
    positions: list[Position] = [
        (356, "01_1", 1, "B", 3),
        (7729, "chr00_23", 75, "A", 1),
    ]
    dataset = m.CarotenoidOzaydin2013Dataset(root=str(_root(tmp_path, rows, positions)))
    assert len(dataset) == 2
    dumps = [dataset[i]["experiment"] for i in range(2)]
    digests = [hashlib.sha256(json.dumps(d).encode("utf-8")).hexdigest() for d in dumps]
    assert digests[0] != digests[1]
    deletions = [
        next(
            p
            for p in d["genotype"]["perturbations"]
            if p["perturbation_type"] == "kanmx_deletion"
        )
        for d in dumps
    ]
    assert [d["construction"] for d in deletions] == [
        {
            "strain_accession": "356",
            "lab": None,
            "batch": "01_1",
            "plate": "1",
            "well": "B3",
        },
        {
            "strain_accession": "7729",
            "lab": None,
            "batch": "chr00_23",
            "plate": "75",
            "well": "A1",
        },
    ]
    # Everything else the two records carry is the same: the position is the whole
    # difference, as it is in the release.
    for d, deletion in zip(dumps, deletions, strict=True):
        deletion["construction"] = None
    assert dumps[0] == dumps[1]


def test_a_sheet_without_the_position_columns_is_refused(tmp_path: Path) -> None:
    """The five position columns are released facts the record must carry, so a sheet
    that lacks them stops the build naming them, never a record without a position.
    """
    root = tmp_path / "carotenoid_ozaydin2013"
    (root / "raw").mkdir(parents=True)
    workbook = openpyxl.Workbook()
    screen = workbook.active
    screen.title = "Color scores of all deletions"
    screen.append(["ORF name", "Strain", "Color", "Comment"])
    screen.append(["YAL001C", "BY4741", 1, None])
    workbook.create_sheet("Names and Functions of TOP200").append(
        ["ORF name", "Gene name", "Gene Function", "Assigned Category"]
    )
    workbook.save(root / "raw" / m.CarotenoidOzaydin2013Dataset.si_filename)
    with pytest.raises(
        KeyError, match=re.escape("['record no.', 'Batch', 'Plate', 'Row', 'Col']")
    ):
        m.CarotenoidOzaydin2013Dataset(root=str(root))
