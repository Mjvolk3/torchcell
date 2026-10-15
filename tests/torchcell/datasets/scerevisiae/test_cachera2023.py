# tests/torchcell/datasets/scerevisiae/test_cachera2023.py
# [[tests.torchcell.datasets.scerevisiae.test_cachera2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_cachera2023.py
"""Cachera 2023 betaxanthin CRI-SPA loader.

The last two tests build from the sha256-pinned library-mirror CSV and are data-gated
(``--data --slow``). Everything above them is hermetic.

A synthetic ``GA1_2_4_6.csv`` under ``tmp_path`` in the released 27-column layout (a
blank index header, ``Unnamed: 0``, ``gene``, then the 24 ``<metric>.<24|48>_<mean|std|
count>`` columns; only the three ``corrected_mean_intensity.24_*`` columns are consumed,
every other cell is 9.0). No genome: every row is stored under the name its ``gene``
cell releases (#1045). Thirteen rows, with (level, std, count):

0. ``0`` (-0.22, 1.2, 87) -> an empty plate position, no record
1. AAC1 (0.13, 1.2, 16) -> record 0 on the released-gene-name leaf, SE 0.3, n 16
2. APL2 (0.25, blank, 1) -> record 1, SE NaN (count 1), n 1
3. BUD16 (blank, blank, 0) -> blank level, no record
4. WT (0.35, 0.68, 12) -> the control row, no record
5. ACN9 (0.5, 0.8, 4) -> record 2 under ``ACN9`` as released (never renamed to SDH7)
6. FLO8 (-0.4, 1.6, 64) -> record 3 under ``FLO8``; SE 1.6 / 8 = 0.2
7. SDH7 (0.9, 0.1, 4) -> record 4 under ``SDH7``, a record of its own beside ACN9
8. blank gene (0.9, 0.1, 4) -> ``str(nan)`` is ``"nan"``, no record
9. AMB (0.7, 0.2, 4) -> record 5 under ``AMB`` (an ambiguous alias is still a name)
10. `` AAD3 `` (0.0, 0.5, blank) -> stripped to AAD3, record 6; a blank count reads as
    1, so SE is NaN and n 1 even though a std is given; 0.0 is a value
11. XYZ9 (0.1, 0.1, 4) -> record 7 under ``XYZ9`` (a name no annotation carries)
12. YBR001C (0.3, 1.2, 16) -> record 8 on ``KanMxDeletionPerturbation``, both names
    the ORF, SE 0.3

So 9 records (8 under a released symbol, 1 under a systematic name), 3 rows without a
gene or a level (0, 3, 8) and 1 control row (WT), ledgered in
``preprocess/dropped_records.json``. Every record shares one reference (the Btx
background at level 0.0), so the index is [[0..8]]; the gene set holds the nine varying
names plus the four cassette identifiers.

Issue #509 (fixed 2026.10.02): every record and its reference store the paper's final
screen plate, ``CACHERA_YPD_G418`` (library YPD root + 2% agar + 200 ug/ml G418, each
quoted from the mirror OCR), ``temperature=None`` with a typed
``not_reported_by_primary`` gap, and ``measurement_type``
``cri_spa_corrected_hsv_yellowness_24h`` (the score is an HSV colony-color value, line 58
of ``paper.md``). Findings still pinned: a blank count reads as one colony, so a released
std is discarded. The dataset gene set carries the non-S288C names ``CYP76AD1`` and
``DOD`` (the cassette additions, ``extract_systematic_gene_names`` takes every
perturbation).

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
from typing import Any

import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.schema import (
    Environment,
    GeneAdditionPerturbation,
    Genotype,
    KanMxDeletionPerturbation,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    ReleasedGeneKind,
    ReleasedGeneNamePerturbation,
)
from torchcell.datasets.scerevisiae import cachera2023 as c

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
    ("YBR001C", 0.3, 1.2, 16.0),
]


def _cell(value: float | str | None) -> str:
    return "" if value is None else str(value)


def _write_csv(
    path: Path,
    rows: list[tuple[str | None, float | None, float | None, float | None]] = _ROWS,
) -> None:
    """The released layout, written by hand so blank cells stay blank."""
    # the loader reads these three by name; they sit at the released positions 18-20
    assert _HEADER.index(c._LEVEL) == 18 and _HEADER.index(c._COUNT) == 20
    lines = [",".join(_HEADER)]
    for i, (gene, level, std, count) in enumerate(rows):
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
    """The thirteen-row CSV of the module docstring, built end to end under ``tmp_path``."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    (tmp_path / "raw").mkdir()
    _write_csv(tmp_path / "raw" / c.DATA_FILENAME)
    return c.BetaxanthinCachera2023Dataset(root=str(tmp_path))


def _cassette() -> list[GeneAdditionPerturbation]:
    return [
        GeneAdditionPerturbation(
            systematic_gene_name=systematic,
            gene_namespace="scerevisiae_s288c_sgd_systematic",
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
        media=c.CACHERA_YPD_G418, temperature=None, provenance_gaps=[c.TEMPERATURE_GAP]
    )


def _deletion(gene: str) -> KanMxDeletionPerturbation | ReleasedGeneNamePerturbation:
    if gene.startswith("Y"):
        return KanMxDeletionPerturbation(
            systematic_gene_name=gene, perturbed_gene_name=gene
        )
    return ReleasedGeneNamePerturbation(
        systematic_gene_name=gene,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=gene,
        released_gene_name=gene,
        released_kind=ReleasedGeneKind.deletion,
        marker="KanMX",
    )


def _experiment(gene: str, level: float, se: float, n: int) -> MetaboliteExperiment:
    return MetaboliteExperiment(
        dataset_name="BetaxanthinCachera2023Dataset",
        genotype=Genotype(perturbations=[_deletion(gene), *_cassette()]),
        environment=_environment(),
        phenotype=MetabolitePhenotype(
            metabolite_level={"betaxanthin": level},
            metabolite_level_se={"betaxanthin": se},
            n_replicates={"betaxanthin": n},
            measurement_type="cri_spa_corrected_hsv_yellowness_24h",
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
            measurement_type="cri_spa_corrected_hsv_yellowness_24h",
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


def test_build_keeps_every_named_row_in_row_order_as_released(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Nine records under the released names, ACN9 and SDH7 both kept, never renamed."""
    got = []
    for i in range(len(built)):
        experiment = built[i]["experiment"]
        (deletion,) = [
            p
            for p in experiment["genotype"]["perturbations"]
            if p["perturbation_type"] in ("kanmx_deletion", "released_gene_name")
        ]
        phenotype = experiment["phenotype"]
        got.append(
            (
                deletion["perturbation_type"],
                deletion["systematic_gene_name"],
                deletion["perturbed_gene_name"],
                phenotype["metabolite_level"]["betaxanthin"],
                phenotype["n_replicates"]["betaxanthin"],
            )
        )
    assert got == [
        ("released_gene_name", "AAC1", "AAC1", 0.13, 16),
        ("released_gene_name", "APL2", "APL2", 0.25, 1),
        ("released_gene_name", "ACN9", "ACN9", 0.5, 4),
        ("released_gene_name", "FLO8", "FLO8", -0.4, 64),
        ("released_gene_name", "SDH7", "SDH7", 0.9, 4),
        ("released_gene_name", "AMB", "AMB", 0.7, 4),
        ("released_gene_name", "AAD3", "AAD3", 0.0, 1),
        ("released_gene_name", "XYZ9", "XYZ9", 0.1, 4),
        ("kanmx_deletion", "YBR001C", "YBR001C", 0.3, 16),
    ]


def test_full_row_record_equals_the_hand_built_experiment(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Record 0 (AAC1): SE = 1.2 / sqrt(16) = 0.3; experiment, reference, publication.

    The varying deletion is ``ReleasedGeneNamePerturbation``: the symbol verbatim in
    ``systematic_gene_name`` and ``released_gene_name``, kind deletion, marker KanMX, no
    released id.
    """
    record = built[0]
    assert record["experiment"] == _experiment("AAC1", 0.13, 0.3, 16).model_dump()
    assert record["reference"] == _reference()
    assert record["publication"] == _PUBLICATION
    (deletion,) = [
        p
        for p in record["experiment"]["genotype"]["perturbations"]
        if p["perturbation_type"] == "released_gene_name"
    ]
    assert (
        deletion["released_gene_name"],
        deletion["released_kind"],
        deletion["marker"],
        deletion["released_gene_id"],
    ) == ("AAC1", "deletion", "KanMX", None)


def test_renamed_and_non_gene_names_are_stored_as_released(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """ACN9 (SGD has since renamed it SDH7) and FLO8 (a blocked reading frame) are
    stored under the names the file gives, with SE 0.8 / 2 = 0.4 and 1.6 / 8 = 0.2;
    the file's own SDH7 row is a fourth record beside ACN9, and the systematic-named
    row goes on the placed KanMX leaf.
    """
    assert built[2]["experiment"] == _experiment("ACN9", 0.5, 0.4, 4).model_dump()
    assert built[3]["experiment"] == _experiment("FLO8", -0.4, 0.2, 64).model_dump()
    assert built[4]["experiment"] == _experiment("SDH7", 0.9, 0.05, 4).model_dump()
    assert built[8]["experiment"] == _experiment("YBR001C", 0.3, 0.3, 16).model_dump()


def test_one_colony_record_has_a_nan_se_and_n_one(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Record 1 (APL2, count 1, blank std): SE is NaN, never 0, and ``n_replicates`` 1."""
    se, rest = _without_se(built[1]["experiment"])
    assert math.isnan(se)
    _, expected = _without_se(_experiment("APL2", 0.25, float("nan"), 1).model_dump())
    assert rest == expected


def test_a_blank_count_discards_the_released_std(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Finding: row 10 has std 0.5 and a blank count; the count reads as 1, so the SE is
    NaN and the std is thrown away rather than the row being refused. The padded gene
    cell `` AAD3 `` is stripped, and a level of 0.0 is kept as a value. Pinned until a
    blank count is refused or recorded as a provenance gap.
    """
    se, rest = _without_se(built[6]["experiment"])
    assert math.isnan(se)
    _, expected = _without_se(_experiment("AAD3", 0.0, float("nan"), 1).model_dump())
    assert rest == expected


def test_environment_and_readout_match_the_paper_issue_509(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Issue #509: every record and its reference store the final CRI-SPA screen plate
    (solid YPD + 2% agar + 200 ug/ml G418, complex), no temperature with a typed
    ``not_reported_by_primary`` gap, and an HSV yellowness ``measurement_type``. The
    loader used to store synthetic SC at 30 C with a fluorescence measurement type.
    """
    environments = {
        json.dumps(built[i]["experiment"]["environment"], sort_keys=True)
        for i in range(len(built))
    }
    assert environments == {json.dumps(_environment().model_dump(), sort_keys=True)}
    record = built[0]
    env = record["experiment"]["environment"]
    media = env["media"]
    assert (media["state"], media["is_synthetic"], media["base_medium"]) == (
        "solid",
        False,
        "YPD",
    )
    assert media["name"].startswith("YPD + G418")
    assert [
        (
            comp["compound"]["name"],
            comp["role"],
            comp["concentration"]["value"],
            comp["concentration"]["unit"],
        )
        for comp in media["components"]
    ] == [
        ("yeast extract", "complex_ingredient", 1.0, "percent_w/v"),
        ("peptone", "complex_ingredient", 2.0, "percent_w/v"),
        ("D-glucose", "carbon_source", 2.0, "percent_w/v"),
        ("agar", "gelling_agent", 2.0, "percent_w/v"),
        ("G418 (geneticin)", "selection_agent", 200.0, "ug/mL"),
    ]
    assert media["components"][-1]["provenance"][0]["quote"] == (
        "For selection on solid medium, plates were supplemented with "
        "$2 0 0 ~ \\mathrm { { m g / l } }$ geneticin (G418)"
    )
    assert env["temperature"] is None
    assert [(g["field"], g["reason"]) for g in env["provenance_gaps"]] == [
        ("temperature", "not_reported_by_primary")
    ]
    assert record["reference"]["environment_reference"] == _environment().model_dump()
    assert {
        built[i]["experiment"]["phenotype"]["measurement_type"]
        for i in range(len(built))
    } | {
        built[i]["reference"]["phenotype_reference"]["measurement_type"]
        for i in range(len(built))
    } == {"cri_spa_corrected_hsv_yellowness_24h"}


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
    assert entry.member_indices == list(range(9))
    assert entry.reference.model_dump() == _reference()
    preprocess = Path(built.preprocess_dir)
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "AAC1",
        "AAD3",
        "ACN9",
        "AMB",
        "APL2",
        "CYP76AD1",
        "DOD",
        "FLO8",
        "SDH7",
        "XYZ9",
        "YBR001C",
        "YBR249C",
        "YPR060C",
    ]
    stored = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [e["member_indices"] for e in stored] == [list(range(9))]


def test_preprocess_csv_holds_the_kept_rows_exactly(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """``preprocess/data.csv``: gene, level, se, n; a NaN SE is a blank cell."""
    text = (Path(built.preprocess_dir) / "data.csv").read_text()
    assert text.splitlines() == [
        "gene,level,se,n",
        "AAC1,0.13,0.3,16",
        "APL2,0.25,,1",
        "ACN9,0.5,0.4,4",
        "FLO8,-0.4,0.2,64",
        "SDH7,0.9,0.05,4",
        "AMB,0.7,0.1,4",
        "AAD3,0.0,,1",
        "XYZ9,0.1,0.05,4",
        "YBR001C,0.3,0.3,16",
    ]


def test_the_ledger_and_the_log_account_for_every_released_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Rows 0, 3, 8 name no gene or carry no level and row 4 is the WT control; the
    ledger lists them verbatim with their rule, and every other row is a record.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    (tmp_path / "raw").mkdir()
    _write_csv(tmp_path / "raw" / c.DATA_FILENAME)
    with caplog.at_level(logging.INFO, logger=c.log.name):
        dataset = c.BetaxanthinCachera2023Dataset(root=str(tmp_path))
    assert len(dataset) == 9
    messages = [r.getMessage() for r in caplog.records if r.name == c.log.name]
    assert (
        "Cachera: 9 records (1 under a systematic name, 8 under a released symbol), "
        "3 rows without a gene or a level, 1 control rows ['WT']"
    ) in messages
    assert "Wrote 9 Cachera betaxanthin experiments to LMDB" in messages
    ledger = c.DropLog.model_validate_json(
        (Path(dataset.preprocess_dir) / "dropped_records.json").read_text()
    )
    assert ledger == c.DropLog(
        dataset="BetaxanthinCachera2023Dataset",
        source_records=13,
        kept_records=9,
        kept_under_systematic_name=1,
        kept_under_released_symbol=8,
        dropped_records=4,
        rules=[
            c.DropRule(
                rule="gene_cell_is_0_or_blank_or_level_is_blank",
                description=c._RULE_DESCRIPTIONS[c.RULE_NO_GENE_OR_LEVEL],
                n_records=3,
                items=["0", "BUD16", "nan"],
            ),
            c.DropRule(
                rule="control_row",
                description=c._RULE_DESCRIPTIONS[c.RULE_CONTROL_ROW],
                n_records=1,
                items=["WT"],
            ),
        ],
    )


def test_a_repeated_gene_token_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two rows naming AAC1 are two claims on one strain: refused by name, no store."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    (tmp_path / "raw").mkdir()
    _write_csv(tmp_path / "raw" / c.DATA_FILENAME, _ROWS + [("AAC1", 0.2, 0.2, 4.0)])
    with pytest.raises(c.RepeatedGeneError, match="names 'AAC1' on two rows"):
        c.BetaxanthinCachera2023Dataset(root=str(tmp_path))
    assert not (tmp_path / "processed" / "lmdb").exists()


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
        c.BetaxanthinCachera2023Dataset(root=str(staged.root))
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


def test_main_builds_under_data_root_without_a_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` builds ``$DATA_ROOT/data/torchcell/betaxanthin_cachera2023`` with no
    genome, then prints the length and record 0.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    root = tmp_path / "data" / "torchcell" / "betaxanthin_cachera2023"
    (root / "raw").mkdir(parents=True)
    _write_csv(root / "raw" / c.DATA_FILENAME)
    c.main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == "len = 9"
    rebuilt = c.BetaxanthinCachera2023Dataset(root=str(root))
    assert lines[-1] == str(rebuilt[0])


# --- data-gated: the real mirror CSV ------------------------------------------------

_DATA_ROOT = os.environ["DATA_ROOT"]
_MIRROR_CSV = osp.join(
    _DATA_ROOT,
    "torchcell-library/cacheraCRISPAHighthroughputMethod2023/si",
    "GA1_2_4_6.csv",
)
_needs_mirror = pytest.mark.skipif(
    not osp.exists(_MIRROR_CSV),
    reason="requires the Cachera CSV mirror at $DATA_ROOT (absent in CI)",
)


@pytest.mark.data
@pytest.mark.slow
@_needs_mirror
def test_cachera_build_stores_every_named_row_as_released(tmp_path: Path) -> None:
    """The pinned file (4,788 rows) builds to 4,759 records with no genome: 1,167 under
    a systematic name on the KanMX leaf (the mixed-case YAL043C-a and YML009c among
    them, upper-cased) and 3,592 under a released symbol on the released-gene-name
    leaf; 28 rows name no gene or carry no level and one is the WT control. ACN9 and
    AIM1 stay as written (SGD has since renamed them), and the strand-less ``YLR287-A``
    is a released symbol, not a malformed id. Measured by slurm array 4604 task 1 and
    by this test on the final loader.
    """
    root = tmp_path / "betaxanthin_cachera2023"
    (root / "raw").mkdir(parents=True)
    shutil.copy(_MIRROR_CSV, root / "raw" / "GA1_2_4_6.csv")
    dataset = c.BetaxanthinCachera2023Dataset(root=str(root))
    assert len(dataset) == 4759

    record = dataset[0]
    exp = MetaboliteExperiment.model_validate(record["experiment"])
    dumped = exp.model_dump()
    assert type(exp).model_validate(dumped).model_dump() == dumped
    assert isinstance(exp.genotype, Genotype)
    ptypes = Counter(p.perturbation_type for p in exp.genotype.perturbations)
    assert ptypes in (
        {"kanmx_deletion": 1, "gene_addition": 4},
        {"released_gene_name": 1, "gene_addition": 4},
    )
    assert exp.environment.media.is_synthetic is False
    assert exp.environment.temperature is None
    assert record["publication"]["pubmed_id"] == "37572348"

    names: dict[str, str] = {}
    for i in range(len(dataset)):
        (deletion,) = [
            p
            for p in dataset[i]["experiment"]["genotype"]["perturbations"]
            if p["perturbation_type"] != "gene_addition"
        ]
        names[deletion["systematic_gene_name"]] = deletion["perturbation_type"]
    assert len(names) == 4759
    assert Counter(names.values()) == {
        "released_gene_name": 3592,
        "kanmx_deletion": 1167,
    }
    assert names["YAL043C-A"] == "kanmx_deletion"
    assert names["ACN9"] == "released_gene_name"
    assert names["AIM1"] == "released_gene_name"
    assert names["YLR287-A"] == "released_gene_name"
    assert "SDH7" not in names and "BOL3" not in names

    ledger = c.DropLog.model_validate_json(
        (Path(dataset.preprocess_dir) / "dropped_records.json").read_text()
    )
    assert (ledger.source_records, ledger.kept_records, ledger.dropped_records) == (
        4788,
        4759,
        29,
    )
    assert [(r.rule, r.n_records) for r in ledger.rules] == [
        ("gene_cell_is_0_or_blank_or_level_is_blank", 28),
        ("control_row", 1),
    ]


_PAPER_MD = osp.join(_DATA_ROOT, "torchcell-library", c.CITATION_KEY, c.PAPER_MD)


@pytest.mark.data
@pytest.mark.skipif(
    not osp.exists(_PAPER_MD), reason="requires the Cachera paper.md mirror"
)
def test_every_sourced_quote_is_on_its_line_of_the_pinned_paper_issue_509() -> None:
    """Each issue #509 quote is a verbatim substring of the pinned OCR, on its line."""
    data = Path(_PAPER_MD).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c.PAPER_MD_SHA256
    lines = data.decode("utf-8").split("\n")
    found = {
        key: [n for n, line in enumerate(lines, 1) if sv.quote in line]
        for key, sv in c.SOURCED_VALUES.items()
    }
    assert found == {
        "measurement_type": [58],
        "screen_medium": [117],
        "readout_medium": [58],
        "ypd_recipe": [31],
        "agar": [31],
        "g418": [31],
    }
    assert not [
        line
        for line in lines
        if "\u00b0" in line or "\\circ" in line or "temperature" in line.lower()
    ]
