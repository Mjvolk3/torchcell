# tests/torchcell/datasets/scerevisiae/test_cachera2023.py
# [[tests.torchcell.datasets.scerevisiae.test_cachera2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_cachera2023.py
"""Cachera 2023 betaxanthin CRI-SPA loader.

The last two tests build from the sha256-pinned library-mirror CSV with a real
``SCerevisiaeGenome`` and are data-gated (``--data --slow``). Everything above them is
hermetic.

2026.09.30 (Phase 14): a synthetic ``GA1_2_4_6.csv`` under ``tmp_path`` in the released
27-column layout (a blank index header, ``Unnamed: 0``, ``gene``, then the 24
``<metric>.<24|48>_<mean|std|count>`` columns; only the three
``corrected_mean_intensity.24_*`` columns are consumed, every other cell is 9.0), and
a stub genome (``_StubGenome``: a hand-written resolution table, every unknown name
RETIRED, and ``feature_index["standard_to_ids"]`` listing AAC1, APL2, AAD3, SDH7 and
FLO8). Twelve rows, with (level, std, count):

0. ``0`` (-0.22, 1.2, 87) -> control, dropped
1. AAC1 (0.13, 1.2, 16) -> record 0, YMR056C / AAC1, SE 1.2 / sqrt(16) = 0.3, n 16
2. APL2 (0.25, blank, 1) -> record 1, YKL135C / APL2, SE NaN (count 1), n 1
3. BUD16 (blank, blank, 0) -> NaN level, dropped
4. WT (0.35, 0.68, 12) -> RETIRED, unresolved
5. ACN9 (0.5, 0.8, 4) -> RENAMED to YDR511W, record 2 stored as SDH7 (the genome's
   standard name, not the source spelling); SE 0.8 / 2 = 0.4, n 4
6. FLO8 (-0.4, 1.6, 64) -> NON_GENE_FEATURE YER109C, record 3; FLO8 is not a current
   gene so the canonical map has no entry and the id is stored twice; SE 1.6 / 8 = 0.2
7. SDH7 (0.9, 0.1, 4) -> CURRENT YDR511W, an ORF collision with row 5, dropped
8. blank gene (0.9, 0.1, 4) -> ``str(nan)`` is ``"nan"``, a control, dropped
9. AMB (0.7, 0.2, 4) -> AMBIGUOUS, unresolved
10. `` AAD3 `` (0.0, 0.5, blank) -> stripped to AAD3, record 4, YCR107W; a blank count
    reads as 1, so SE is NaN and n 1 even though a std is given; 0.0 is a value

11. XYZ9 (0.1, 0.1, 4) -> RETIRED, unresolved

So 5 records, 3 control/NaN rows (0, 3, 8), 3 unresolved names (AMB, WT, XYZ9) and 1
collision (YDR511W). Every record shares one reference (the Btx
background at level 0.0), so the index is [[0, 1, 2, 3, 4]]; the gene set holds the five
varying ORFs plus the four cassette identifiers ``CYP76AD1``, ``DOD``, ``YBR249C``,
``YPR060C`` (nine names, sorted).

Findings pinned (source lines in ``cachera2023.py``), citing issue #509 (the paper's
screen plate is YPD-G418, it states no temperature, and the score is an HSV colony-color
value): every record stores ``Media(name="SC", is_synthetic=True)`` at 30 C (lines
339-342) with ``measurement_type`` ``cri_spa_corrected_fluorescence_intensity_24h``
(line 65). A blank count reads as one colony (line 256), so a released std is
discarded. The dataset gene set carries the non-S288C names ``CYP76AD1`` and ``DOD``
(the cassette additions, ``extract_systematic_gene_names`` takes every perturbation).

The sha256 contract (issue #528, fixed): ``process`` verifies the CSV in ``raw/`` against
``DATA_SHA256`` before reading a row, so a file placed there by hand is refused; the
tests that build from the synthetic CSV run under the conftest recorder in place of that
byte check.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import os.path as osp
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from tests.torchcell.conftest import require_trusted_genome_database
from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.schema import (
    Environment,
    GeneAdditionPerturbation,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import cachera2023 as c
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
)

_CURRENT = {"AAC1": "YMR056C", "APL2": "YKL135C", "AAD3": "YCR107W", "SDH7": "YDR511W"}
_RENAME = {"ACN9": "YDR511W"}
_NON_GENE = {"FLO8": ("YER109C", "blocked_reading_frame")}
_AMBIGUOUS = {"AMB": ["YAL001C", "YAL002W"]}


class _StubGenome:
    """The two things the loader asks a genome for, with a hand-written answer table."""

    feature_index: dict[str, dict[str, list[str]]] = {
        "standard_to_ids": {
            name: [] for name in ["AAC1", "APL2", "AAD3", "SDH7", "FLO8"]
        }
    }

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        if name in _CURRENT:
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.CURRENT,
                systematic_name=_CURRENT[name],
            )
        if name in _RENAME:
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.RENAMED,
                systematic_name=_RENAME[name],
            )
        if name in _NON_GENE:
            systematic, feature = _NON_GENE[name]
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.NON_GENE_FEATURE,
                systematic_name=systematic,
                feature_type=feature,
            )
        if name in _AMBIGUOUS:
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.AMBIGUOUS,
                systematic_name=None,
                candidates=_AMBIGUOUS[name],
            )
        return GeneNameResolution(
            input_name=name, status=GeneNameStatus.RETIRED, systematic_name=name
        )


# The released header of GA1_2_4_6.csv, verbatim (27 columns, the first one blank).
_HEADER = (
    ",Unnamed: 0,gene,area.24_mean,area.24_std,area.24_count,mean_intensity.24_mean,"
    "mean_intensity.24_std,mean_intensity.24_count,area.48_mean,area.48_std,"
    "area.48_count,mean_intensity.48_mean,mean_intensity.48_std,"
    "mean_intensity.48_count,corrected_area.24_mean,corrected_area.24_std,"
    "corrected_area.24_count,corrected_mean_intensity.24_mean,"
    "corrected_mean_intensity.24_std,corrected_mean_intensity.24_count,"
    "corrected_area.48_mean,corrected_area.48_std,corrected_area.48_count,"
    "corrected_mean_intensity.48_mean,corrected_mean_intensity.48_std,"
    "corrected_mean_intensity.48_count"
).split(",")

# (gene, level, std, count); None is a blank cell.
_ROWS: list[tuple[str | None, float | None, float | None, float | None]] = [
    ("0", -0.22, 1.2, 87.0),
    ("AAC1", 0.13, 1.2, 16.0),
    ("APL2", 0.25, None, 1.0),
    ("BUD16", None, None, 0.0),
    ("WT", 0.35, 0.68, 12.0),
    ("ACN9", 0.5, 0.8, 4.0),
    ("FLO8", -0.4, 1.6, 64.0),
    ("SDH7", 0.9, 0.1, 4.0),
    (None, 0.9, 0.1, 4.0),
    ("AMB", 0.7, 0.2, 4.0),
    (" AAD3 ", 0.0, 0.5, None),
    ("XYZ9", 0.1, 0.1, 4.0),
]


def _cell(value: float | str | None) -> str:
    return "" if value is None else str(value)


def _write_csv(path: Path) -> None:
    """The released layout, written by hand so blank cells stay blank."""
    # the loader reads these three by name; they sit at the released positions 18-20
    assert _HEADER.index(c._LEVEL) == 18 and _HEADER.index(c._COUNT) == 20
    lines = [",".join(_HEADER)]
    for i, (gene, level, std, count) in enumerate(_ROWS):
        cells: dict[str, str] = {name: "9.0" for name in _HEADER[3:]}
        cells[c._LEVEL] = _cell(level)
        cells[c._STD] = _cell(std)
        cells[c._COUNT] = _cell(count)
        lines.append(
            ",".join([str(i), str(i), _cell(gene)] + [cells[n] for n in _HEADER[3:]])
        )
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> c.BetaxanthinCachera2023Dataset:
    """The twelve-row CSV of the module docstring, built end to end under ``tmp_path``."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    (tmp_path / "raw").mkdir()
    _write_csv(tmp_path / "raw" / c.DATA_FILENAME)
    return c.BetaxanthinCachera2023Dataset(
        root=str(tmp_path), genome=cast(SCerevisiaeGenome, _StubGenome())
    )


def _cassette() -> list[GeneAdditionPerturbation]:
    return [
        GeneAdditionPerturbation(
            systematic_gene_name=systematic,
            perturbed_gene_name=name,
            source_organism=organism,
            is_heterologous=heterologous,
            localization="chromosomal_integration",
            integration_locus="XII-5",
            construct_name="Btx-cassette",
            variant=variant,
        )
        for systematic, name, organism, heterologous, variant in [
            ("CYP76AD1", "CYP76AD1", "Beta vulgaris", True, None),
            ("DOD", "DOD", "Mirabilis jalapa", True, None),
            ("YBR249C", "ARO4", "Saccharomyces cerevisiae", False, "K229L"),
            ("YPR060C", "ARO7", "Saccharomyces cerevisiae", False, "G141S"),
        ]
    ]


def _environment() -> Environment:
    return Environment(
        media=Media(name="SC", state="solid", is_synthetic=True),
        temperature=Temperature(value=30),
    )


def _experiment(
    orf: str, name: str, level: float, se: float, n: int
) -> MetaboliteExperiment:
    return MetaboliteExperiment(
        dataset_name="BetaxanthinCachera2023Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=name
                ),
                *_cassette(),
            ]
        ),
        environment=_environment(),
        phenotype=MetabolitePhenotype(
            metabolite_level={"betaxanthin": level},
            metabolite_level_se={"betaxanthin": se},
            n_replicates={"betaxanthin": n},
            measurement_type="cri_spa_corrected_fluorescence_intensity_24h",
            target_metabolite_ids=None,
        ),
    )


def _reference() -> dict[str, Any]:
    return MetaboliteExperimentReference(
        dataset_name="BetaxanthinCachera2023Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=_environment(),
        phenotype_reference=MetabolitePhenotype(
            metabolite_level={"betaxanthin": 0.0},
            metabolite_level_se=None,
            n_replicates={"betaxanthin": 1},
            measurement_type="cri_spa_corrected_fluorescence_intensity_24h",
        ),
    ).model_dump()


_PUBLICATION = Publication(
    pubmed_id="37572348",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/37572348/",
    doi="10.1093/nar/gkad656",
    doi_url="https://doi.org/10.1093/nar/gkad656",
).model_dump()


def _without_se(experiment: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Split off the SE (NaN never compares equal) and return the rest of the record."""
    rest = json.loads(json.dumps(experiment, default=str))
    se = rest["phenotype"]["metabolite_level_se"].pop("betaxanthin")
    return float(se), rest


def test_build_keeps_five_orfs_in_row_order_with_genome_spellings(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Rows 1, 2, 5, 6, 10 kept; ACN9 is stored as SDH7, FLO8's id is stored twice."""
    got = []
    for i in range(len(built)):
        experiment = built[i]["experiment"]
        (deletion,) = [
            p
            for p in experiment["genotype"]["perturbations"]
            if p["perturbation_type"] == "kanmx_deletion"
        ]
        phenotype = experiment["phenotype"]
        got.append(
            (
                deletion["systematic_gene_name"],
                deletion["perturbed_gene_name"],
                phenotype["metabolite_level"]["betaxanthin"],
                phenotype["n_replicates"]["betaxanthin"],
            )
        )
    assert got == [
        ("YMR056C", "AAC1", 0.13, 16),
        ("YKL135C", "APL2", 0.25, 1),
        ("YDR511W", "SDH7", 0.5, 4),
        ("YER109C", "YER109C", -0.4, 64),
        ("YCR107W", "AAD3", 0.0, 1),
    ]


def test_full_row_record_equals_the_hand_built_experiment(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Record 0 (AAC1): SE = 1.2 / sqrt(16) = 0.3; experiment, reference, publication."""
    record = built[0]
    assert (
        record["experiment"]
        == _experiment("YMR056C", "AAC1", 0.13, 0.3, 16).model_dump()
    )
    assert record["reference"] == _reference()
    assert record["publication"] == _PUBLICATION


def test_renamed_and_non_gene_records_carry_their_exact_se(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """ACN9 -> YDR511W / SDH7 with SE 0.8 / 2 = 0.4; FLO8 -> YER109C with 1.6 / 8 = 0.2."""
    assert (
        built[2]["experiment"]
        == _experiment("YDR511W", "SDH7", 0.5, 0.4, 4).model_dump()
    )
    assert (
        built[3]["experiment"]
        == _experiment("YER109C", "YER109C", -0.4, 0.2, 64).model_dump()
    )


def test_one_colony_record_has_a_nan_se_and_n_one(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Record 1 (APL2, count 1, blank std): SE is NaN, never 0, and ``n_replicates`` 1."""
    se, rest = _without_se(built[1]["experiment"])
    assert math.isnan(se)
    _, expected = _without_se(
        _experiment("YKL135C", "APL2", 0.25, float("nan"), 1).model_dump()
    )
    assert rest == expected


def test_a_blank_count_discards_the_released_std(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Finding: row 10 has std 0.5 and a blank count; the count reads as 1 (line 256),
    so the SE is NaN and the std is thrown away rather than the row being refused. The
    padded gene cell `` AAD3 `` is stripped and resolves, and a level of 0.0 is kept as
    a value. Pinned until a blank count is refused or recorded as a provenance gap.
    """
    se, rest = _without_se(built[4]["experiment"])
    assert math.isnan(se)
    _, expected = _without_se(
        _experiment("YCR107W", "AAD3", 0.0, float("nan"), 1).model_dump()
    )
    assert rest == expected


def test_environment_and_readout_disagree_with_the_paper_issue_509(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Finding (issue #509): every record and its reference store SC, synthetic, solid,
    at 30 C (lines 339-342) with a fluorescence ``measurement_type`` (line 65), where the
    paper's final screen plate is YPD-G418, no temperature is stated, and the score is an
    HSV colony-color value. Pinned until #509 lands a sourced YPD + G418 medium, a
    temperature gap and a color-score measurement type.
    """
    environments = {
        json.dumps(built[i]["experiment"]["environment"], sort_keys=True)
        for i in range(len(built))
    }
    assert environments == {json.dumps(_environment().model_dump(), sort_keys=True)}
    record = built[0]
    media = record["experiment"]["environment"]["media"]
    assert (media["name"], media["state"], media["is_synthetic"]) == (
        "SC",
        "solid",
        True,
    )
    assert record["experiment"]["environment"]["temperature"]["value"] == 30.0
    assert record["reference"]["environment_reference"] == _environment().model_dump()
    assert {
        built[i]["experiment"]["phenotype"]["measurement_type"]
        for i in range(len(built))
    } == {"cri_spa_corrected_fluorescence_intensity_24h"}


def test_every_record_shares_one_reference_and_the_gene_set_has_the_cassette(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Finding: the gene set is every perturbation's ``systematic_gene_name``, so the
    heterologous ``CYP76AD1`` and ``DOD`` sit beside the S288C ids. Pinned until the gene
    set is restricted to the genome's ids or the cassette carries separate identifiers.
    """
    index = built.experiment_reference_index
    assert index is not None
    (entry,) = index
    assert entry.member_indices == [0, 1, 2, 3, 4]
    assert entry.reference.model_dump() == _reference()
    preprocess = Path(built.preprocess_dir)
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "CYP76AD1",
        "DOD",
        "YBR249C",
        "YCR107W",
        "YDR511W",
        "YER109C",
        "YKL135C",
        "YMR056C",
        "YPR060C",
    ]
    stored = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [e["member_indices"] for e in stored] == [[0, 1, 2, 3, 4]]


def test_preprocess_csv_holds_the_kept_rows_exactly(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """``preprocess/data.csv``: orf, common, level, se, n; a NaN SE is a blank cell."""
    text = (Path(built.preprocess_dir) / "data.csv").read_text()
    assert text.splitlines() == [
        "orf,common,level,se,n",
        "YMR056C,AAC1,0.13,0.3,16",
        "YKL135C,APL2,0.25,,1",
        "YDR511W,SDH7,0.5,0.4,4",
        "YER109C,YER109C,-0.4,0.2,64",
        "YCR107W,AAD3,0.0,,1",
    ]


def test_the_drop_summary_is_logged_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Rows 0, 3, 8 are control/NaN; AMB, WT, XYZ9 unresolved; SDH7 collides with ACN9.

    ``XYZ9`` and ``WT`` are RETIRED and ``AMB`` is AMBIGUOUS: only CURRENT, RENAMED and
    NON_GENE_FEATURE are kept (lines 212-222). The list is logged sorted.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    (tmp_path / "raw").mkdir()
    _write_csv(tmp_path / "raw" / c.DATA_FILENAME)
    with caplog.at_level(logging.INFO, logger=c.log.name):
        dataset = c.BetaxanthinCachera2023Dataset(
            root=str(tmp_path), genome=cast(SCerevisiaeGenome, _StubGenome())
        )
    assert len(dataset) == 5
    messages = [r.getMessage() for r in caplog.records if r.name == c.log.name]
    assert (
        "Cachera: 5 usable ORFs, 3 control/NaN rows, 3 unresolved names dropped "
        "['AMB', 'WT', 'XYZ9'], 1 ORF collisions deduped"
    ) in messages
    assert "Wrote 5 Cachera betaxanthin experiments to LMDB" in messages


def test_process_refuses_without_a_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    (tmp_path / "raw").mkdir()
    _write_csv(tmp_path / "raw" / c.DATA_FILENAME)
    with pytest.raises(RuntimeError) as excinfo:
        c.BetaxanthinCachera2023Dataset(root=str(tmp_path))
    assert str(excinfo.value) == (
        "Cachera2023 requires an injected SCerevisiaeGenome to resolve common "
        "gene names to systematic ORF ids (source uses common names)."
    )


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #528): with the CSV already in ``raw/`` PyG skips ``download``,
    so ``process`` verifies it against ``DATA_SHA256`` first and raises
    ``RawSha256MismatchError`` naming the file and both digests; no store is written and
    the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(c, [c.DATA_FILENAME])
    raw = staged.root / "raw" / c.DATA_FILENAME
    with pytest.raises(RawSha256MismatchError) as excinfo:
        c.BetaxanthinCachera2023Dataset(
            root=str(staged.root), genome=cast(SCerevisiaeGenome, _StubGenome())
        )
    assert str(excinfo.value) == (
        f"sha256 mismatch for {raw}: expected {c.DATA_SHA256}, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


class _Response:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._data


def _bare(tmp_path: Path) -> c.BetaxanthinCachera2023Dataset:
    """An uninitialized instance whose ``raw_dir`` is ``tmp_path/raw`` (not created)."""
    dataset = c.BetaxanthinCachera2023Dataset.__new__(c.BetaxanthinCachera2023Dataset)
    dataset.root = str(tmp_path)
    return dataset


def _fake_urlopen(
    monkeypatch: pytest.MonkeyPatch, data: bytes, calls: list[tuple[str, str, int]]
) -> None:
    def urlopen(req: Any, timeout: int) -> _Response:
        calls.append((req.full_url, req.get_header("User-agent"), timeout))
        return _Response(data)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_download_refuses_a_response_under_ten_thousand_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """9999 bytes is under the floor; the request carries the URL, UA and 120 s timeout."""
    calls: list[tuple[str, str, int]] = []
    _fake_urlopen(monkeypatch, b"x" * 9999, calls)
    dataset = _bare(tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        dataset.download()
    assert str(excinfo.value) == "CRI-SPA download too small: 9999 bytes"
    assert calls == [(c.DATA_URL, "Mozilla/5.0", 120)]
    assert os.listdir(tmp_path / "raw") == []


def test_download_refuses_a_response_off_the_pin_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"y" * 10000
    _fake_urlopen(monkeypatch, data, [])
    dataset = _bare(tmp_path)
    with pytest.raises(RawSha256MismatchError) as excinfo:
        dataset.download()
    assert str(excinfo.value) == (
        f"sha256 mismatch for {c.DATA_URL}: expected {c.DATA_SHA256}, "
        f"observed {hashlib.sha256(data).hexdigest()}"
    )
    assert os.listdir(tmp_path / "raw") == []


def test_download_writes_a_response_on_the_pin_then_accepts_it_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the pin set to the payload's digest the bytes land verbatim; a second call
    finds the file and returns without a request (``process`` verifies it at build).
    """
    data = b"z" * 10000
    monkeypatch.setattr(c, "DATA_SHA256", hashlib.sha256(data).hexdigest())
    calls: list[tuple[str, str, int]] = []
    _fake_urlopen(monkeypatch, data, calls)
    dataset = _bare(tmp_path)
    dataset.download()
    assert (tmp_path / "raw" / c.DATA_FILENAME).read_bytes() == data
    dataset.download()
    assert len(calls) == 1


def test_items_retype_through_the_metabolite_classes_and_the_hooks_are_inert(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """``transform_item`` rebuilds a stored item as a ``MetaboliteExperiment`` with its
    reference, dumping to exactly the stored dictionaries (``experiment_dataset.py``
    lines 638 to 641); the raw file list is the one released CSV, and ``preprocess_raw``
    is the documented identity.
    """
    item = built[0]
    typed = built.transform_item(item)
    assert type(typed["experiment"]) is MetaboliteExperiment
    assert type(typed["reference"]) is MetaboliteExperimentReference
    assert typed["experiment"].model_dump() == item["experiment"]
    assert typed["reference"].model_dump() == item["reference"]
    assert built.raw_file_names == ["GA1_2_4_6.csv"]
    frame = pd.DataFrame({"a": [1]})
    assert built.preprocess_raw(frame) is frame


def test_main_builds_under_data_root_with_the_genome_it_constructs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` builds ``$DATA_ROOT/data/torchcell/betaxanthin_cachera2023`` with a genome
    read from ``$DATA_ROOT/data/sgd/genome`` and ``data/go`` (``overwrite=False``), then
    prints the length and record 0.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    genome_kwargs: list[dict[str, Any]] = []

    class _MainGenome(_StubGenome):
        def __init__(self, **kwargs: Any) -> None:
            genome_kwargs.append(kwargs)

    monkeypatch.setattr(c, "SCerevisiaeGenome", _MainGenome)
    root = tmp_path / "data" / "torchcell" / "betaxanthin_cachera2023"
    (root / "raw").mkdir(parents=True)
    _write_csv(root / "raw" / c.DATA_FILENAME)
    c.main()
    assert genome_kwargs == [
        {
            "genome_root": osp.join(str(tmp_path), "data/sgd/genome"),
            "go_root": osp.join(str(tmp_path), "data/go"),
            "overwrite": False,
        }
    ]
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == "len = 5"
    rebuilt = c.BetaxanthinCachera2023Dataset(
        root=str(root), genome=cast(SCerevisiaeGenome, _StubGenome())
    )
    assert lines[-1] == str(rebuilt[0])


# --- data-gated: the real mirror CSV and the real SGD genome -----------------------

_DATA_ROOT = os.environ["DATA_ROOT"]
_MIRROR_CSV = osp.join(
    _DATA_ROOT,
    "torchcell-library/cacheraCRISPAHighthroughputMethod2023/si",
    "GA1_2_4_6.csv",
)
_GENOME_DIR = osp.join(_DATA_ROOT, "data/sgd/genome")
_GO_DIR = osp.join(_DATA_ROOT, "data/go")
_needs_mirror = pytest.mark.skipif(
    not (osp.exists(_MIRROR_CSV) and osp.isdir(_GENOME_DIR) and osp.isdir(_GO_DIR)),
    reason="requires Cachera CSV mirror + SGD genome at $DATA_ROOT (absent in CI)",
)


@pytest.mark.data
@pytest.mark.slow
@_needs_mirror
def test_cachera_build_smoke(tmp_path: Path) -> None:
    """Cachera dataset builds from the mirror + genome and yields schema-valid records."""
    root = tmp_path / "betaxanthin_cachera2023"
    (root / "raw").mkdir(parents=True)
    shutil.copy(_MIRROR_CSV, root / "raw" / "GA1_2_4_6.csv")

    require_trusted_genome_database(_GENOME_DIR)
    genome = SCerevisiaeGenome(
        genome_root=_GENOME_DIR, go_root=_GO_DIR, overwrite=False
    )
    dataset = c.BetaxanthinCachera2023Dataset(root=str(root), genome=genome)
    # 4,788 raw rows minus 28 control/NaN rows, 11 unresolvable names (2 ambiguous, 9
    # retired) and 30 ORF collisions. The earlier 4735 predated the layered resolver.
    assert len(dataset) == 4719

    # dataset[i] returns the stored dicts; validating through the schema IS the
    # round-trip test (the stale on-disk LMDB failed exactly here on media.is_synthetic).
    record = dataset[0]
    exp = MetaboliteExperiment.model_validate(record["experiment"])
    dumped = exp.model_dump()
    assert type(exp).model_validate(dumped).model_dump() == dumped

    # genotype: one KO deletion + the 4-gene Btx-cassette (natMX marker omitted)
    assert isinstance(exp.genotype, Genotype)
    ptypes = Counter(p.perturbation_type for p in exp.genotype.perturbations)
    assert ptypes == {"kanmx_deletion": 1, "gene_addition": 4}

    # synthetic medium (the field whose absence made the stale on-disk LMDB fail)
    assert exp.environment.media.is_synthetic is True

    assert record["publication"]["pubmed_id"] == "37572348"


@pytest.mark.data
@pytest.mark.slow
@_needs_mirror
def test_cachera_varying_deletion_carries_the_canonical_common_name(
    tmp_path: Path,
) -> None:
    """The varying KO stores a common name, not a second copy of the ORF id.

    Issue #195: the CRI-SPA file is common-named, the resolver turns those names into
    systematic ids, and the loader used to store the id in both fields. It now stores the
    genome's own standard name (the L1 ``canonical_gene_names`` spelling shared with the
    Smith, Mormino and Lian loaders), falling back to the id for an ORF with no
    round-tripping standard name. The nine genes a pre-resolver build dropped are the
    check that both halves are present.
    """
    root = tmp_path / "betaxanthin_cachera2023"
    (root / "raw").mkdir(parents=True)
    shutil.copy(_MIRROR_CSV, root / "raw" / "GA1_2_4_6.csv")

    require_trusted_genome_database(_GENOME_DIR)
    genome = SCerevisiaeGenome(
        genome_root=_GENOME_DIR, go_root=_GO_DIR, overwrite=False
    )
    dataset = c.BetaxanthinCachera2023Dataset(root=str(root), genome=genome)

    # (systematic id -> stored common name) for the varying deletion of every record.
    ko_names: dict[str, str] = {}
    for i in range(len(dataset)):
        exp = MetaboliteExperiment.model_validate(dataset[i]["experiment"])
        assert isinstance(exp.genotype, Genotype)
        kos = [
            p
            for p in exp.genotype.perturbations
            if p.perturbation_type == "kanmx_deletion"
        ]
        assert len(kos) == 1
        ko_names[kos[0].systematic_gene_name] = kos[0].perturbed_gene_name

    # The nine genes the pre-resolver build dropped, with the common name the file gives.
    for systematic, common in {
        "YEL024W": "RIP1",
        "YHL011C": "PRS3",
        "YKL148C": "SDH1",
        "YML022W": "APT1",
        "YML100W": "TSL1",
        "YMR205C": "PFK2",
        "YNL129W": "NRK1",
        "YPR047W": "MSF1",
        "YPR128C": "ANT1",
    }.items():
        assert ko_names[systematic] == common

    # Most ORFs have a standard name, so most records must differ between the two fields;
    # an ORF with no round-tripping standard name legitimately stores the id in both.
    n_common = sum(1 for sysn, name in ko_names.items() if name != sysn)
    assert n_common == 3930
    assert len(ko_names) - n_common == 789

    # The spelling comes from the genome, not from the 2023-era source column, so a name
    # SGD has since superseded is stored under its current standard name.
    assert ko_names["YDR511W"] == "SDH7"  # source column says ACN9
    assert ko_names["YAL046C"] == "BOL3"  # source column says AIM1

    # Every (systematic, perturbed) pair stays unique, so the L1 ORF-uniqueness check
    # cannot be collapsed by two source spellings mapping onto one standard name.
    assert len({(s, n) for s, n in ko_names.items()}) == len(dataset)

    # The fixed Btx-cassette keeps the same convention (ARO4 at YBR249C, ARO7 at YPR060C).
    exp0 = MetaboliteExperiment.model_validate(dataset[0]["experiment"])
    assert isinstance(exp0.genotype, Genotype)
    additions = {
        p.systematic_gene_name: p.perturbed_gene_name
        for p in exp0.genotype.perturbations
        if p.perturbation_type == "gene_addition"
    }
    assert additions["YBR249C"] == "ARO4"
    assert additions["YPR060C"] == "ARO7"
