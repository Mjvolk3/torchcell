# tests/torchcell/paper/test_tables.py
"""Unit tests for the reusable paper-table primitives."""

from __future__ import annotations

import gzip
import pickle
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lmdb
import pytest

import torchcell.paper.tables as tables
from torchcell.paper.tables import (
    Column,
    PaperTable,
    Row,
    SignalCache,
    default_phenotype_bytes,
    human_bytes,
    instance_bytes,
    phenotype_descriptor,
    read_first_record,
    read_frontmatter,
    scientific,
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


def test_scientific_bytes() -> None:
    c = scientific(103_398)
    assert c.md == "1.0×10⁵"
    assert c.tex == r"$1.0\times10^{5}$"
    assert scientific(584).md == "5.8×10²"  # smallest -> still a positive exponent
    assert scientific(127_797_526).md == "1.3×10⁸"  # Costanzo dmf


def _record(graph_level: str, label_name: str, value: Any) -> dict[str, Any]:
    return {
        "experiment": {
            "phenotype": {
                "graph_level": graph_level,
                "label_name": label_name,
                label_name: value,
            }
        }
    }


def test_phenotype_descriptor() -> None:
    # scalar label -> scalar; global stays global
    assert phenotype_descriptor(_record("global", "fitness", 0.9)) == (
        "scalar",
        "global",
    )
    # length-1 vector is still a scalar
    assert phenotype_descriptor(
        _record("metabolism", "metabolite_level", {"betaxanthin": 1.2})
    ) == ("scalar", "bipartite node")
    # multi-element vector reports its dimensionality
    assert phenotype_descriptor(
        _record("node", "expression_log2_ratio", [0.0] * 6169)
    ) == ("vector (6169)", "node")
    # metabolism graph_level maps to bipartite node
    assert phenotype_descriptor(
        _record("metabolism", "metabolite_level", {"a": 1, "b": 2})
    ) == ("vector (2)", "bipartite node")


def test_paper_table_cell_md_tex_divergence() -> None:
    cols = [Column(header="Dataset", align="l"), Column(header="Signal", align="r")]
    rows = [Row(cells={"Dataset": "X", "Signal": scientific(103_398)})]
    t = PaperTable(columns=cols, rows=rows)
    assert "1.0×10⁵" in t.to_markdown(sectioned=False)  # unicode in md
    assert r"$1.0\times10^{5}$" in t.to_latex(
        caption="c", label="tab:x"
    )  # math in latex


def test_paper_table_footer_totals() -> None:
    cols = [Column(header="Dataset", align="l"), Column(header="Instances", align="r")]
    rows = [Row(section="A", cells={"Dataset": "d1", "Instances": "10"})]
    footer = Row(bold=True, cells={"Dataset": "Total (1 datasets)", "Instances": "10"})
    t = PaperTable(columns=cols, rows=rows, footer=footer)
    tex = t.to_latex(caption="c", label="tab:x")
    # footer is preceded by a rule and bolded
    assert r"\midrule" in tex.rsplit("d1", 1)[1]  # a midrule after the body
    assert r"\textbf{Total (1 datasets)}" in tex
    assert r"\textbf{10}" in tex
    md = t.to_markdown(sectioned=True)
    assert "### Total" in md
    assert "| **Total (1 datasets)** | **10** |" in md


def test_read_frontmatter(tmp_path: Path) -> None:
    note = tmp_path / "n.md"
    note.write_text("---\nid: abc\ntitle: T\n---\n\nbody\n")
    assert read_frontmatter(note) == "---\nid: abc\ntitle: T\n---"
    assert "title: Missing" in read_frontmatter(
        tmp_path / "nope.md", default_title="Missing"
    )


# --------------------------------------------------------------------------- #
# 2026.10.06 (Phase 21): exact renderings and the remaining branches.
# --------------------------------------------------------------------------- #
def test_scientific_zero_negative_and_small_values() -> None:
    """Non-positive values render as ``0``; 0.05 has exponent -2 (superscript minus)."""
    assert scientific(0) == tables.Cell(md="0", tex="0")
    assert scientific(-3.0) == tables.Cell(md="0", tex="0")
    assert scientific(0.05) == tables.Cell(md="5.0×10⁻²", tex=r"$5.0\times10^{-2}$")
    assert scientific(1) == tables.Cell(md="1.0×10⁰", tex=r"$1.0\times10^{0}$")


def test_scientific_mantissa_can_round_up_to_ten() -> None:
    """Finding: the exponent is taken BEFORE the mantissa is rounded, so a value just
    below a power of ten renders with a two-digit mantissa: 99,999 is
    ``floor(log10) = 4``, ``9.9999 -> "10.0"``, giving ``10.0×10⁴`` instead of
    ``1.0×10⁵`` (likewise 9.96 -> ``10.0×10⁰``). The docstring promises
    identically-shaped magnitudes. Pinned until the exponent is recomputed after
    rounding (tables.py:107-108). Reach:
    experiments/database/scripts/render_supported_datasets_table.py renders these cells
    into paper/nature-biotech/sections/datasets_table.tex; no current value there has a
    10.0 mantissa.
    """
    assert scientific(99_999) == tables.Cell(md="10.0×10⁴", tex=r"$10.0\times10^{4}$")
    assert scientific(9.96).md == "10.0×10⁰"
    assert scientific(9.94).md == "9.9×10⁰"


def test_instance_bytes_is_the_sorted_experiment_json() -> None:
    rec = {
        "experiment": {"phenotype": {"b": 1, "a": 2}, "genotype": {"x": [1]}},
        "reference": {"r": 0},
    }
    assert (
        instance_bytes(rec)
        == b'{"genotype": {"x": [1]}, "phenotype": {"a": 2, "b": 1}}'
    )


def test_stream_gzip_signal_logs_every_n_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three records, ``log_every=2``: one progress line at n=2. Fake clock: t0=100,
    then 101 for both reads inside the log call, so rate 2/1 = 2 per second, 1 s.
    """
    d = _make_lmdb(tmp_path, [{"x": 1}, {"x": 2}, {"x": 3}])
    clock = iter([100.0, 101.0, 101.0])
    monkeypatch.setattr(tables, "time", SimpleNamespace(time=lambda: next(clock)))
    lines: list[str] = []
    n, _ = stream_gzip_signal(d, log_every=2, label="ds", log=lines.append)
    assert n == 3
    assert lines == ["    [ds] 2 records (2/s, 1s)"]


def test_read_first_record_returns_the_first_key(tmp_path: Path) -> None:
    """Keys are ``b"0"``, ``b"1"``: the cursor's first is ``b"0"``."""
    d = _make_lmdb(tmp_path, [{"fitness": 0.25}, {"fitness": 0.75}])
    assert read_first_record(d) == {
        "experiment": {"phenotype": {"fitness": 0.25}},
        "reference": {},
        "publication": {},
    }


def test_phenotype_descriptor_interaction_roles() -> None:
    """``edge``/``hyperedge`` come from the perturbation count, overriding graph_level;
    an absent graph_level renders the placeholder U+2014. A one-perturbation
    interaction record would read ``hyperedge`` (``n_genes == 2`` is the only edge
    test); no stored interaction has fewer than two perturbations.
    """

    def rec(level: str | None, n: int) -> dict[str, Any]:
        return {
            "experiment": {
                "phenotype": {"graph_level": level, "label_name": "gi", "gi": 0.1},
                "genotype": {"perturbations": [{}] * n},
            }
        }

    assert phenotype_descriptor(rec("edge", 2)) == ("scalar", "edge")
    assert phenotype_descriptor(rec("edge", 3)) == ("scalar", "hyperedge")
    assert phenotype_descriptor(rec("hyperedge", 2)) == ("scalar", "edge")
    assert phenotype_descriptor(rec("hyperedge", 1)) == ("scalar", "hyperedge")
    assert phenotype_descriptor(rec(None, 1)) == ("scalar", "\u2014")


def _table() -> PaperTable:
    cols = [Column(header="Dataset", align="l"), Column(header="N & n", align="r")]
    rows = [
        Row(section="Fitness", cells={"Dataset": "smf_a", "N & n": "1"}),
        Row(section=None, cells={"Dataset": "loose", "N & n": "2"}),
        Row(section="Fitness", cells={"Dataset": "dmf", "N & n": "3"}),
        Row(section="Expr", cells={"Dataset": "kem", "N & n": ""}, bold=True),
    ]
    return PaperTable(
        columns=cols,
        rows=rows,
        footer=Row(bold=True, cells={"Dataset": "Total", "N & n": "6"}),
    )


def test_to_latex_full_rendering_with_sections_comment_and_footnote() -> None:
    r"""Sections in first-seen order (Fitness, None, Expr); ``\addlinespace`` before
    every named section but the first; the None section has no heading; a bold row
    keeps an empty cell empty; the footer follows a ``\midrule``.
    """
    tex = _table().to_latex(
        caption="Cap",
        label="tab:x",
        header_comment="line one\nline two",
        footnote="fn.",
    )
    assert tex == "\n".join(
        [
            "% line one",
            "% line two",
            r"\begin{table*}[t]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\caption{Cap}",
            r"\label{tab:x}",
            r"\begin{tabular}{@{}l r@{}}",
            r"\toprule",
            r"Dataset & N \& n \\",
            r"\midrule",
            r"\multicolumn{2}{@{}l}{\textbf{Fitness}} \\",
            r"smf\_a & 1 \\",
            r"dmf & 3 \\",
            r"loose & 2 \\",
            r"\addlinespace",
            r"\multicolumn{2}{@{}l}{\textbf{Expr}} \\",
            r"\textbf{kem} &  \\",
            r"\midrule",
            r"\textbf{Total} & \textbf{6} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\\[2pt]{\footnotesize fn.}",
            r"\end{table*}",
            "",
        ]
    )


def test_to_latex_unsectioned_keeps_row_order() -> None:
    tex = _table().to_latex(
        caption="C",
        label="L",
        sectioned=False,
        table_env="table",
        position="h",
        size="small",
        colsep_pt=2,
    )
    body = tex.split("\\midrule\n", 1)[1]
    assert body.startswith(
        "smf\\_a & 1 \\\\\nloose & 2 \\\\\ndmf & 3 \\\\\n\\textbf{kem} &  \\\\\n"
    )
    assert tex.startswith(
        "\\begin{table}[h]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{2pt}\n"
    )
    assert tex.endswith("\\end{table}\n")


def test_to_markdown_full_rendering() -> None:
    md = _table().to_markdown(heading_level=2)
    assert md == "\n".join(
        [
            "## Fitness",
            "",
            "| Dataset | N & n |",
            "| :-- | --: |",
            "| smf_a | 1 |",
            "| dmf | 3 |",
            "",
            "| Dataset | N & n |",
            "| :-- | --: |",
            "| loose | 2 |",
            "",
            "## Expr",
            "",
            "| Dataset | N & n |",
            "| :-- | --: |",
            "| **kem** |  |",
            "",
            "## Total",
            "",
            "| Dataset | N & n |",
            "| :-- | --: |",
            "| **Total** | **6** |",
        ]
    )


@pytest.mark.parametrize("text", ["---\nid: x\nno close\n", "id: x\n---\n"])
def test_read_frontmatter_falls_back_without_a_closed_leading_block(
    tmp_path: Path, text: str
) -> None:
    note = tmp_path / "n.md"
    note.write_text(text)
    assert read_frontmatter(note, default_title="T") == "---\ntitle: T\ndesc: ''\n---"


def test_signal_cache_recomputes_on_a_changed_fingerprint(tmp_path: Path) -> None:
    d = _make_lmdb(tmp_path, [{"x": 1}])
    cache = SignalCache.load(tmp_path / "c.json")
    cache.entries["ds"] = tables._CacheEntry(key="0:0", n=99, bytes=99)
    n, _, hit = cache.get_or_compute("ds", d, log=lambda s: None)
    assert (n, hit) == (1, False)
    assert re.fullmatch(r"\d+:\d+", cache.entries["ds"].key)
