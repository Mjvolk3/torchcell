# tests/torchcell/verification/test_induction_timecourse.py
# [[tests.torchcell.verification.test_induction_timecourse]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_induction_timecourse.py
"""Rule-level tests of ``torchcell.verification.induction_timecourse``.

The whole verifier runs end to end on a synthetic store in
``tests/torchcell/datasets/scerevisiae/test_hackett2020.py``; this module pins the
helpers whose failure modes that run does not reach: the quote audit across its three
source shapes (text, gzip, the Table EV1 workbook), the GEO title parser on the title
spellings the real series uses (``_t0_rep``, ``__rep_t0``, a system token), and each of
the three layer-nesting rules.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from torchcell.datasets.scerevisiae import hackett2020 as h
from torchcell.verification import induction_timecourse as v
from torchcell.verification.report import Level, Provenance
from torchcell.verification.sourced import SourcedValue

CK = h.CITATION_KEY


def _sv(uri: str, data: bytes, quote: str) -> SourcedValue:
    return SourcedValue(
        value="x",
        provenance=Provenance(
            source_uri=uri, citation_key=CK, sha256=hashlib.sha256(data).hexdigest()
        ),
        quote=quote,
    )


def test_audit_reads_text_gzip_and_the_strain_workbook(tmp_path: Path) -> None:
    library = tmp_path / "library"
    (library / CK / "si").mkdir(parents=True)
    paper = b"Cells were grown at 30 C. Estradiol was added."
    (library / CK / "paper.md").write_bytes(paper)
    strains = tmp_path / "strains.xlsx"
    pd.DataFrame(
        [
            ["Strain", "Genotype", "Type", "Source"],
            ["SMY1", "KanMX-Z3EVpr-ACA1", "TF", "x"],
        ]
    ).to_excel(strains, header=False, index=False, sheet_name=h.STRAINS_SHEET)
    workbook = strains.read_bytes()
    (library / CK / "si" / "si2.xls").write_bytes(workbook)
    data_root = tmp_path / "data"
    geo = gzip.compress(
        b'!Sample_growth_protocol\t"at 7.5 mg/L methionine."\n', mtime=0
    )
    geo_rel = f"{h.RAW_DIR_REL}/data/geo.txt.gz"
    (data_root / h.RAW_DIR_REL / "data").mkdir(parents=True)
    (data_root / geo_rel).write_bytes(geo)
    good = [
        _sv("paper.md", paper, "Estradiol was added."),
        _sv("paper.md", paper, "Estradiol was added."),  # a repeat is checked once
        _sv(geo_rel, geo, "at 7.5 mg/L methionine."),
        _sv("si/si2.xls", workbook, "SMY1\tKanMX-Z3EVpr-ACA1\tTF\tx"),
    ]
    result = v.audit_sourced(good, library_root=library, data_root=data_root)
    assert (result.level, result.passed, result.details["n_checked"]) == (
        Level.L3,
        True,
        3,
    )
    bad = [
        _sv("paper.md", paper, "Cells were grown at 37 C."),
        _sv("paper.md", b"other bytes", "Cells were grown"),
        _sv("si/absent.md", b"", "x"),
    ]
    result = v.audit_sourced(bad, library_root=library, data_root=data_root)
    assert not result.passed
    assert result.details["failures"] == [
        {"source": "paper.md", "quote": "Cells were grown at 37 C."},
        {"source": "paper.md", "problem": "missing or sha256 drift"},
        {"source": "si/absent.md", "problem": "missing or sha256 drift"},
    ]
    assert result.details["failed_sources"] == ["paper.md", "si/absent.md"]


def _matrix(path: Path, titles: list[str], strains: list[str]) -> None:
    lines = [
        "!Sample_title\t" + "\t".join(f'"{t}"' for t in titles),
        "!Sample_characteristics_ch1\t" + "\t".join(f'"strain: {s}"' for s in strains),
        "!Sample_characteristics_ch1\t"
        + "\t".join('"background: s288c"' for _ in titles),
        "!series_matrix_table_begin",
        '"ID_REF"\t"GSM1"',
        '"after the table"\t"1"',
    ]
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0))


def test_geo_samples_parses_the_real_title_spellings(tmp_path: Path) -> None:
    path = tmp_path / "m.txt.gz"
    _matrix(
        path,
        [
            "SMY57_HAP1_t0",
            "SMY57_HAP1_t0_rep",
            "SMY26_MSN2__rep_t0",
            "SMY257_GCN4_4ZEV_t2.5",
        ],
        ["SMY57", "SMY57", "SMY26", "SMY257"],
    )
    assert v.geo_samples([path]) == {
        ("SMY57", 0.0, "SMY57_HAP1_t0"): 1,
        ("SMY57", 0.0, "SMY57_HAP1_t0_rep"): 1,
        ("SMY26", 0.0, "SMY26_MSN2__rep_t0"): 1,
        ("SMY257", 2.5, "SMY257_GCN4_4ZEV_t2.5"): 1,
    }


def test_geo_samples_refuses_an_unparseable_title_and_a_short_strain_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.txt.gz"
    _matrix(path, ["SMY57_HAP1_early"], ["SMY57"])
    with pytest.raises(ValueError, match="unparseable GEO title"):
        v.geo_samples([path])
    _matrix(path, ["SMY57_HAP1_t0", "SMY57_HAP1_t5"], ["SMY57"])
    with pytest.raises(ValueError, match="2 titles, 1 strains"):
        v.geo_samples([path])


def _layers(
    cleaned: float, thresholded: float, selected: float, shrunken: float
) -> dict[str, Any]:
    return {
        "log2_cleaned": {"G": cleaned},
        "log2_thresholded": {"G": thresholded},
        "log2_selected": {"G": selected},
        "log2_shrunken": {"G": shrunken},
    }


@pytest.mark.parametrize(
    ("phen", "rule"),
    [
        (_layers(0.5, 0.5, 0.0, 0.2), "shrunken"),
        (_layers(0.5, 0.0, 0.5, 0.0), "selected"),
        (_layers(0.5, 0.4, 0.0, 0.0), "thresholded"),
    ],
)
def test_check_nesting_names_the_broken_rule(phen: dict[str, Any], rule: str) -> None:
    bad: list[dict[str, Any]] = []
    v._check_nesting(phen, "TC", 5.0, bad)
    assert bad == [{"tc": "TC", "t": 5.0, "gene": "G", "rule": rule}]


def test_check_nesting_accepts_nested_layers_and_skips_ungapped_ones() -> None:
    bad: list[dict[str, Any]] = []
    v._check_nesting(_layers(0.5, 0.5, 0.5, 0.25), "TC", 5.0, bad)
    v._check_nesting(
        {"log2_cleaned": {"G": 1.0}, "log2_thresholded": None}, "TC", 5.0, bad
    )
    assert bad == []


def test_record_sourced_values_skips_the_layers() -> None:
    sv = _sv("paper.md", b"x", "quote").model_dump()
    record = {
        "experiment": {
            "environment": {"media": {"provenance": [sv]}, "log2_cleaned": {"q": sv}}
        },
        "reference": {"genome_reference": {"background": {"provenance": [sv, sv]}}},
    }
    found = list(v._record_sourced_values(record))
    assert [s.quote for s in found] == ["quote", "quote", "quote"]
