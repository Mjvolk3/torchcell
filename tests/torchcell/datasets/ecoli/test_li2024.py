# tests/torchcell/datasets/ecoli/test_li2024.py
# [[tests.torchcell.datasets.ecoli.test_li2024]]
"""Hermetic tests of the D2Cell 2026 provenance record (no dataset is registered)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.datasets.ecoli import li2024
from torchcell.literature.manifest import Manifest, RetrievalMethod

COLUMNS = [
    "DOI",
    "organism",
    "titer value",
    "titer unit",
    "parent strain",
    "knock out gene",
    "overexpress gene",
    "heterologous gene",
    "medium",
    "carbon source concentration",
    "time",
    "temperature",
    "product titer",
    "Publication Year",
    "Article Title",
]


def _row(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {c: None for c in COLUMNS}
    base.update(kw)
    return base


def _database() -> pd.DataFrame:
    rows = [
        _row(
            DOI="10.1/a",
            organism="Escherichia coli",
            **{
                "titer value": 1.5,
                "titer unit": "g/L",
                "parent strain": "BW25113",
                "knock out gene": "pta;ackA",
                "medium": "M9",
                "Publication Year": 2020,
                "Article Title": "A",
            },
        ),
        _row(
            DOI="10.1/a",
            organism="Escherichia coli",
            **{
                "titer value": "(3.6 g)",
                "titer unit": "g",
                "parent strain": "37 °C",
                "overexpress gene": "dxs",
                "time": "Assumed similar",
                "Publication Year": 2020,
                "Article Title": "A",
            },
        ),
        _row(
            DOI="10.1/b",
            organism="Escherichia coli",
            **{
                "titer value": 0.2,
                "titer unit": "g/L",
                "medium": "等效发酵条件",
                "Publication Year": 2021,
                "Article Title": "B",
            },
        ),
        _row(
            DOI="10.1/c",
            organism="Saccharomyces cerevisiae",
            **{"titer value": 4.0, "titer unit": "g/L"},
        ),
        _row(),
    ]
    return pd.DataFrame(rows, columns=COLUMNS, dtype=object)


def test_no_dataset_registered() -> None:
    assert not any(
        cls.__module__ == li2024.__name__ for cls in dataset_registry.values()
    )


def test_measure_database_counts_ecoli_surface_and_faults() -> None:
    m = li2024.measure_database(_database())
    assert m.n_rows_nonempty == 4
    assert m.n_columns == len(COLUMNS)
    assert m.n_organisms == 2
    assert m.n_ecoli_rows == 3
    assert m.n_ecoli_source_dois == 2
    assert m.n_ecoli_rows_without_doi == 0
    assert m.n_ecoli_numeric_titer == 2
    assert m.n_distinct_titer_units == 2
    assert m.titer_unit_counts == {"g/L": 2, "g": 1}
    assert m.n_ecoli_parent_strain_missing == 1
    assert m.n_ecoli_parent_strain_is_temperature == 1
    assert m.n_ecoli_no_genotype == 1
    assert m.n_ecoli_rows_with_placeholder == 1
    assert m.n_ecoli_rows_with_cjk == 1
    assert m.n_ecoli_rows_with_source_quote == 0


def test_measure_database_counts_a_quote_column_when_present() -> None:
    frame = _database()
    frame["source quote"] = ["s", None, None, None, None]
    assert li2024.measure_database(frame).n_ecoli_rows_with_source_quote == 1


def test_ecoli_source_dois_is_the_per_doi_lead_list() -> None:
    leads = li2024.ecoli_source_dois(_database())
    assert leads.to_dict("records") == [
        {"doi": "10.1/a", "n_entries": 2, "publication_year": 2020, "title": "A"},
        {"doi": "10.1/b", "n_entries": 1, "publication_year": 2021, "title": "B"},
    ]


def test_measure_training_split_partitions_real_and_simulated() -> None:
    cols = [
        "pert index",
        "product index",
        "inf_label_01",
        "product",
        "doi list",
        "real data",
    ]
    train = pd.DataFrame(
        [
            ["[1]", 0, 1, "succ_c", None, "yes"],
            ["[2]", 0, 0, "succ_c", "['x']", "yes"],
            ["[3]", 1, 1, "ac_c", None, None],
        ],
        columns=cols,
    )
    test = pd.DataFrame([["[4]", 1, 0, "ac_c", None, None]], columns=cols)
    m = li2024.measure_training_split({"train": train, "test": test})
    assert m.n_rows == {"train": 3, "test": 1}
    assert m.n_total == 4
    assert m.n_real_flagged == 2
    assert m.n_simulated == 2
    assert m.label_counts_real == {1: 1, 0: 1}
    assert m.label_counts_simulated == {1: 1, 0: 1}
    assert m.n_with_doi == 1
    assert m.n_real_positive_with_doi == 0
    assert m.stated_experimental == 8134
    assert m.stated_total == 19777


def test_raw_files_record_rerunnable_retrievals() -> None:
    names = [f.name for f in li2024.RAW_FILES]
    assert names == [li2024.DATABASE_FILE, *li2024.SPLIT_FILES]
    for raw in li2024.RAW_FILES:
        record = raw.retrieval
        assert record.retriever == "torchcell.literature.retrieve.direct_url"
        assert record.params == {"url": raw.source_url}
        assert record.sha256 == raw.sha256
    assert li2024.RAW_FILES[0].method is RetrievalMethod.zenodo
    assert all(li2024.GITHUB_COMMIT in f.source_url for f in li2024.RAW_FILES[1:])


def _fake_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    fakes = []
    for raw in li2024.RAW_FILES:
        path = tmp_path / "src" / raw.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw.name.encode())
        digest = hashlib.sha256(raw.name.encode()).hexdigest()
        fakes.append(raw.model_copy(update={"sha256": digest, "bytes": len(raw.name)}))
        sources[raw.name] = path
    monkeypatch.setattr(li2024, "RAW_FILES", tuple(fakes))
    monkeypatch.setattr(li2024, "RAW_FILES_BY_NAME", {f.name: f for f in fakes})
    return sources


def test_deposit_writes_verified_files_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _fake_raw(tmp_path, monkeypatch)
    root = li2024.deposit_raw_mirror(sources=sources, data_root=str(tmp_path))
    assert root == tmp_path / li2024.RAW_DIR_REL
    manifest = Manifest.model_validate_json((root / "manifest.json").read_text())
    assert manifest.citation_key == li2024.CITATION_KEY
    assert manifest.doi == li2024.PAPER_DOI
    assert [f.path for f in manifest.files] == [f.relpath for f in li2024.RAW_FILES]
    assert all(f.retrieval is not None for f in manifest.files)
    # idempotent on a second deposit
    li2024.deposit_raw_mirror(sources=sources, data_root=str(tmp_path))
    data = json.loads((root / "manifest.json").read_text())
    assert len(data["files"]) == 4


def test_deposit_refuses_missing_and_drifted_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _fake_raw(tmp_path, monkeypatch)
    with pytest.raises(KeyError):
        li2024.deposit_raw_mirror(
            sources={li2024.DATABASE_FILE: sources[li2024.DATABASE_FILE]},
            data_root=str(tmp_path),
        )
    sources[li2024.DATABASE_FILE].write_bytes(b"drift")
    with pytest.raises(RawSha256MismatchError):
        li2024.deposit_raw_mirror(sources=sources, data_root=str(tmp_path))


def test_readers_verify_and_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / li2024.RAW_DIR_REL / "data"
    root.mkdir(parents=True)
    _database().to_excel(root / li2024.DATABASE_FILE, index=False)
    for name in li2024.SPLIT_FILES:
        (root / name).write_text(
            "﻿pert index,inf_label_01,doi list,real data\n[1],1,,yes\n", encoding="utf-8"
        )
    fakes = [
        raw.model_copy(
            update={
                "sha256": hashlib.sha256((root / raw.name).read_bytes()).hexdigest()
            }
        )
        for raw in li2024.RAW_FILES
    ]
    monkeypatch.setattr(li2024, "RAW_FILES_BY_NAME", {f.name: f for f in fakes})
    frame = li2024.read_database(str(tmp_path))
    assert len(frame) == 4  # pandas drops the trailing all-empty row on read
    splits = li2024.read_splits(str(tmp_path))
    assert list(splits) == list(li2024.SPLIT_FILES)
    assert list(splits[li2024.SPLIT_FILES[0]].columns)[0] == "pert index"


def test_paths_resolve_under_data_root(tmp_path: Path) -> None:
    assert li2024.raw_mirror_dir(str(tmp_path)) == tmp_path / li2024.RAW_DIR_REL
    assert li2024.library_dir(str(tmp_path)) == tmp_path / li2024.LIBRARY_DIR_REL


def test_schema_blockers_name_the_titer_fields() -> None:
    fields = [b.field for b in li2024.SCHEMA_BLOCKERS]
    assert "ProductTiterExperimentReference.phenotype_reference" in fields
    assert "ProductTiterPhenotype.titer_unit" in fields
    assert "ProductTiterExperiment.genotype" in fields
