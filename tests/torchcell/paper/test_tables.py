# tests/torchcell/paper/test_tables.py
"""Unit tests for the reusable paper-table primitives."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import lmdb

from torchcell.paper.tables import (
    Column,
    PaperTable,
    Row,
    SignalCache,
    default_phenotype_bytes,
    human_bytes,
    read_frontmatter,
    stream_gzip_signal,
    tex_escape,
)


def test_human_bytes_matches_paper_style() -> None:
    assert human_bytes(584) == "584 B"
    assert human_bytes(5_099) == "5.1 KB"
    assert human_bytes(101_000) == "101 KB"
    assert human_bytes(913_971) == "914 KB"
    assert human_bytes(2_024_641) == "2 MB"
    assert human_bytes(22_548_096) == "22.5 MB"


def test_tex_escape() -> None:
    assert tex_escape("n×1") == r"n$\times$1"
    assert tex_escape("A & B") == r"A \& B"
    assert tex_escape("~6000") == r"\textasciitilde{}6000"
    assert tex_escape("β-carotene") == r"$\beta$-carotene"
    assert tex_escape("smf_costanzo") == r"smf\_costanzo"


def _make_lmdb(tmp: Path, phenotypes: list[dict[str, Any]]) -> Path:
    """Write a torchcell-shaped LMDB with the given phenotype dicts."""
    d = tmp / "processed" / "lmdb"
    d.mkdir(parents=True)
    env = lmdb.open(str(d), map_size=10_000_000)
    with env.begin(write=True) as txn:
        for i, ph in enumerate(phenotypes):
            rec = {"experiment": {"phenotype": ph}, "reference": {}, "publication": {}}
            txn.put(str(i).encode(), pickle.dumps(rec))
    env.close()
    return d


def test_stream_gzip_signal_matches_nonstreaming(tmp_path: Path) -> None:
    phenotypes = [{"fitness": i * 0.5, "se": 0.01, "label": "smf"} for i in range(200)]
    d = _make_lmdb(tmp_path, phenotypes)

    n, nbytes = stream_gzip_signal(d, log_every=0)
    assert n == 200

    # The streamed size must equal a single gzip over the concatenated payloads,
    # iterated in LMDB key order (the cursor's order).
    blob = bytearray()
    env = lmdb.open(str(d), readonly=True, lock=False)
    with env.begin() as txn:
        for _, v in txn.cursor():
            blob += default_phenotype_bytes(pickle.loads(v))
    env.close()
    assert nbytes == len(gzip.compress(bytes(blob), 6))


def test_signal_cache_roundtrip_and_invalidation(tmp_path: Path) -> None:
    d = _make_lmdb(tmp_path, [{"x": 1}, {"x": 2}])
    cache_path = tmp_path / "cache.json"

    cache = SignalCache.load(cache_path)
    n1, b1, hit1 = cache.get_or_compute("ds", d, label="ds")
    assert (n1, hit1) == (2, False)

    # Reload from disk -> same values, now a cache hit (no recompute).
    cache2 = SignalCache.load(cache_path)
    n2, b2, hit2 = cache2.get_or_compute("ds", d, label="ds")
    assert (n2, b2, hit2) == (n1, b1, True)


def test_paper_table_markdown_sectioned() -> None:
    cols = [
        Column(header="Dataset", align="l"),
        Column(header="Signal (gzip)", align="r"),
    ]
    rows = [
        Row(
            section="Fitness",
            cells={"Dataset": "Costanzo 2016 smf", "Signal (gzip)": "101 KB"},
        ),
        Row(
            section="Metabolite",
            cells={"Dataset": "Mülleder 2016", "Signal (gzip)": "914 KB"},
        ),
    ]
    md = PaperTable(columns=cols, rows=rows).to_markdown(sectioned=True)
    assert "### Fitness" in md
    assert "### Metabolite" in md
    assert "| :-- | --: |" in md
    assert "| Costanzo 2016 smf | 101 KB |" in md


def test_paper_table_latex_sectioned() -> None:
    cols = [
        Column(header="Dataset", align="l"),
        Column(header="Signal (gzip)", align="r"),
    ]
    rows = [
        Row(
            section="Fitness",
            cells={"Dataset": "Costanzo 2016 dmf", "Signal (gzip)": "5 MB"},
        )
    ]
    tex = PaperTable(columns=cols, rows=rows).to_latex(caption="Cap", label="tab:x")
    assert r"\begin{table*}[t]" in tex
    assert r"\begin{tabular}{@{}l r@{}}" in tex
    assert r"\multicolumn{2}{@{}l}{\textbf{Fitness}} \\" in tex
    assert r"Costanzo 2016 dmf & 5 MB \\" in tex
    assert r"\label{tab:x}" in tex


def test_read_frontmatter(tmp_path: Path) -> None:
    note = tmp_path / "n.md"
    note.write_text("---\nid: abc\ntitle: T\n---\n\nbody\n")
    assert read_frontmatter(note) == "---\nid: abc\ntitle: T\n---"
    assert "title: Missing" in read_frontmatter(
        tmp_path / "nope.md", default_title="Missing"
    )
