# torchcell/paper/tables.py
# [[torchcell.paper.tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/paper/tables.py
# Test file: tests/torchcell/paper/test_tables.py
"""Reusable primitives for generating paper tables in BOTH markdown and LaTeX
from a single in-code source of truth, plus data-derived columns such as a
streaming-gzip "signal" (a Kolmogorov-complexity proxy) computed directly from a
built LMDB.

Why this exists: paper tables were being hand-transcribed from markdown into
LaTeX, which drifts and hides errors. The reliable pattern is: define columns +
rows once (pydantic), compute any data-derived cells from the source of truth
(the LMDB), then render markdown and LaTeX from the same object. This module is
table-agnostic so multiple paper tables can reuse it.

Building blocks:
  - ``human_bytes`` / ``tex_escape``            -- formatting helpers
  - ``stream_gzip_signal``                      -- bounded-memory gzip size over an LMDB
  - ``SignalCache``                             -- memoize signals, keyed by LMDB fingerprint
  - ``Column`` / ``Row`` / ``PaperTable``       -- render the same table to md + LaTeX
"""

from __future__ import annotations

import json
import math
import pickle
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import lmdb
from pydantic import BaseModel, Field

# --- formatting -----------------------------------------------------------


def human_bytes(nbytes: int) -> str:
    """Format a byte count the way a paper table reads: 0.6 KB, 101 KB, 1.9 MB.

    Uses SI (1000) units to match the gzip-size convention already in the note.
    """
    kb = nbytes / 1000
    if kb < 1:
        return f"{nbytes} B"
    if kb < 10:
        return f"{kb:.1f}".rstrip("0").rstrip(".") + " KB"
    if kb < 1000:
        return f"{kb:.0f} KB"
    mb = kb / 1000
    return f"{mb:.1f}".rstrip("0").rstrip(".") + " MB"


_TEX_REPLACEMENTS = {
    "&": r"\&",
    "%": r"\%",
    "_": r"\_",
    "#": r"\#",
    "$": r"\$",
    "×": r"$\times$",
    "~": r"\textasciitilde{}",
    "β": r"$\beta$",
    "±": r"$\pm$",
    "≈": r"$\approx$",
}


def tex_escape(s: str) -> str:
    """Escape the LaTeX specials that show up in dataset table cells.

    Note ``$`` is escaped too, so pass already-formed math (e.g. ``n×1``) using
    the unicode source characters (``×``) and let this build the math, rather
    than embedding raw ``$...$`` yourself.
    """
    for k, v in _TEX_REPLACEMENTS.items():
        s = s.replace(k, v)
    return s


class Cell(BaseModel):
    r"""A table cell that renders differently in markdown vs LaTeX.

    Use when a value needs markup that differs between the two targets -- e.g.
    scientific notation, where markdown wants unicode superscripts (``1.3×10⁵``)
    and LaTeX wants math mode (``$1.3\times10^{5}$``). ``tex`` is emitted RAW
    (not escaped); plain ``str`` cells are tex-escaped as usual.
    """

    md: str
    tex: str


_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def scientific(value: float) -> Cell:
    """Format a number in scientific notation (unit-free) for a table cell.

    Renders identically-shaped magnitudes so a column is scannable and aligns:
    ``584 -> 5.8×10²``, ``103398 -> 1.0×10⁵``, ``127797526 -> 1.3×10⁸``. Mantissa
    is 2 significant figures. The UNIT belongs in the column header (e.g. bytes),
    not the cell -- keeping cells numeric is what lets them right-align cleanly.
    """
    if value <= 0:
        return Cell(md="0", tex="0")
    e = math.floor(math.log10(value))
    mant = f"{value / 10**e:.1f}"
    return Cell(
        md=f"{mant}×10{str(e).translate(_SUPERSCRIPT)}",
        tex=rf"${mant}\times10^{{{e}}}$",
    )


# --- gzip signal over an LMDB (streaming, bounded memory) -----------------


def default_phenotype_bytes(record: dict[str, Any]) -> bytes:
    """Extract the serialized phenotype from a torchcell LMDB record.

    Records are ``{"experiment": {..., "phenotype": {...}}, ...}``. Sorting
    keys makes the signal deterministic across builds.
    """
    ph = record["experiment"]["phenotype"]
    return json.dumps(ph, sort_keys=True, default=str).encode()


def instance_bytes(record: dict[str, Any]) -> bytes:
    """Serialize the full stored INSTANCE (perturbation + environment + phenotype).

    This counts the stored PERTURBATION -- the edit that defines the genotype,
    whether a single gene deletion or thousands of natural gene-presence entries
    that amount to an entirely new genome (referenced by ``sequence_uri`` +
    ``sequence_sha256``) -- alongside the environment and the measured phenotype.
    It never includes the shared reference genome (that lives external, as a
    pointer, and is not stored per-instance); only each instance's delta from it.
    Constant record metadata (experiment_type, dataset_name) compresses away.
    """
    return json.dumps(record["experiment"], sort_keys=True, default=str).encode()


def stream_gzip_signal(
    lmdb_dir: str | Path,
    *,
    extract: Callable[[dict[str, Any]], bytes] = default_phenotype_bytes,
    level: int = 6,
    log_every: int = 1_000_000,
    label: str = "",
    log: Callable[[str], None] = print,
) -> tuple[int, int]:
    """Return ``(n_records, gzip_bytes)`` for an LMDB in a single streaming pass.

    Byte-equivalent to gzip-compressing the concatenation of ``extract(record)``
    over every record, but via ``zlib.compressobj`` so memory stays bounded --
    this is what lets 20M+ record LMDBs (tens of GB) get a real value instead of
    "pending". ``extract`` is pluggable so the same machinery can measure other
    per-record payloads for future tables.
    """
    lmdb_dir = str(lmdb_dir)
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, max_readers=2048)
    comp = zlib.compressobj(level, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    total = 0
    n = 0
    t0 = time.time()
    try:
        with env.begin() as txn:
            for _, v in txn.cursor():
                total += len(comp.compress(extract(pickle.loads(v))))
                n += 1
                if log_every and n % log_every == 0:
                    rate = n / max(time.time() - t0, 1e-9)
                    log(
                        f"    [{label}] {n:,} records ({rate:,.0f}/s, {time.time() - t0:.0f}s)"
                    )
        total += len(comp.flush())
    finally:
        env.close()
    return n, total


class DatasetSignalRecord(BaseModel):
    """One dataset's computed row -- the raw-data artifact the DB report emits.

    This is the source-of-truth output: a ``build`` step scans each LMDB and
    writes a list of these to JSON; ``render`` (table) and ``plot`` (scatter)
    steps consume ONLY the JSON, never re-reading an LMDB. ``instances`` is the
    dataset length (record count); ``signal_bytes`` is the gzip proxy.
    """

    section: str
    name: str
    genotypes: str
    env: str
    phenotype: str
    instances: int
    shape: str
    graph_role: str
    signal_bytes: int
    built: bool


def read_first_record(lmdb_dir: str | Path) -> dict[str, Any]:
    """Return the first record of an LMDB (for schema-derived table columns)."""
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            _, v = next(iter(txn.cursor()))
            record: dict[str, Any] = pickle.loads(v)
            return record
    finally:
        env.close()


# The schema stores a ``graph_level`` per phenotype; display it against the Cell
# Graph Transformer's structure -- the gene multigraph + the bipartite metabolic
# network. ``metabolism`` -> ``bipartite node`` (a metabolite in the bipartite
# layer). Interactions are refined edge-vs-hyperedge by gene count below.
GRAPH_ROLE_MAP = {"metabolism": "bipartite node"}
_INTERACTION_LEVELS = {"edge", "hyperedge"}


def phenotype_descriptor(record: dict[str, Any]) -> tuple[str, str]:
    """Derive ``(shape, graph_role)`` for one record from its stored instance.

    ``shape`` is the shape of a SINGLE phenotype instance: ``scalar`` or
    ``vector (D)`` (a length-1 vector is a scalar). ``graph_role`` is where the
    label sits in the cell graph: ``global`` / ``node`` / ``bipartite node``
    (from ``graph_level``), or -- for a gene interaction -- ``edge`` when it
    relates two gene nodes and ``hyperedge`` when it relates three or more,
    derived from the number of perturbations (so a digenic interaction reads
    ``edge`` and a trigenic one ``hyperedge``, regardless of the stored
    ``graph_level``). Everything is read from the record, never hand-authored.
    """
    ph = record["experiment"]["phenotype"]
    graph_level = ph.get("graph_level")
    role = GRAPH_ROLE_MAP.get(graph_level, graph_level or "—")
    if graph_level in _INTERACTION_LEVELS:
        n_genes = len(record["experiment"]["genotype"]["perturbations"])
        role = "edge" if n_genes == 2 else "hyperedge"
    label = ph.get(ph.get("label_name"))
    if isinstance(label, (dict, list)) and len(label) > 1:
        shape = f"vector ({len(label)})"
    else:
        shape = "scalar"
    return shape, role


def lmdb_fingerprint(lmdb_dir: str | Path) -> str:
    """Fingerprint an LMDB's ``data.mdb`` (mtime+size) so a rebuild invalidates
    any cached signal. Raises if the file is absent (caller checks existence).
    """
    st = (Path(lmdb_dir) / "data.mdb").stat()
    return f"{int(st.st_mtime)}:{st.st_size}"


class _CacheEntry(BaseModel):
    key: str
    n: int
    bytes: int


class SignalCache(BaseModel):
    """JSON-backed memo of ``(n, bytes)`` per id, keyed by LMDB fingerprint.

    Reuse across runs so only rebuilt datasets recompute. ``entries`` maps a
    stable id (e.g. a dataset subpath) to its last computed signal + the LMDB
    fingerprint it was computed from.
    """

    path: Path
    entries: dict[str, _CacheEntry] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> SignalCache:
        """Load an existing cache JSON, or return an empty cache for a new path."""
        path = Path(path)
        if path.exists():
            raw = json.loads(path.read_text())
            entries = {k: _CacheEntry(**v) for k, v in raw.items()}
            return cls(path=path, entries=entries)
        return cls(path=path)

    def save(self) -> None:
        """Persist the cache to its JSON path (sorted for a stable diff)."""
        payload = {k: v.model_dump() for k, v in sorted(self.entries.items())}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def get_or_compute(
        self,
        cache_id: str,
        lmdb_dir: str | Path,
        *,
        extract: Callable[[dict[str, Any]], bytes] = default_phenotype_bytes,
        label: str = "",
        log: Callable[[str], None] = print,
    ) -> tuple[int, int, bool]:
        """Return ``(n, bytes, from_cache)``, computing + persisting on a miss."""
        fp = lmdb_fingerprint(lmdb_dir)
        hit = self.entries.get(cache_id)
        if hit is not None and hit.key == fp:
            return hit.n, hit.bytes, True
        n, nbytes = stream_gzip_signal(lmdb_dir, extract=extract, label=label, log=log)
        self.entries[cache_id] = _CacheEntry(key=fp, n=n, bytes=nbytes)
        self.save()
        return n, nbytes, False


# --- generic table (one source of truth -> markdown + LaTeX) --------------

Align = Literal["l", "c", "r"]
_MD_ALIGN = {"l": ":--", "c": ":-:", "r": "--:"}


class Column(BaseModel):
    """A table column. ``header`` is also the key used to look up a row's cell."""

    header: str
    align: Align = "l"


class Row(BaseModel):
    r"""One table row. ``cells`` maps column header -> display string. ``section``
    optionally groups rows (a subheading in md, a ``\multicolumn`` rule in LaTeX).
    ``bold`` emphasizes the whole row (e.g. a totals footer).
    """

    cells: dict[str, str | Cell]
    section: str | None = None
    bold: bool = False


class PaperTable(BaseModel):
    r"""Columns + rows rendered identically to markdown and LaTeX.

    Rows may carry a ``section``; sections render as H3 headings (markdown) or
    bold ``\multicolumn`` group rows (LaTeX), preserving first-seen order.
    """

    columns: list[Column]
    rows: list[Row]
    footer: Row | None = None

    def _cell_md(self, row: Row, col: Column) -> str:
        v = row.cells.get(col.header, "")
        s = v.md if isinstance(v, Cell) else v
        return f"**{s}**" if row.bold and s else s

    def _cell_tex(self, row: Row, col: Column) -> str:
        v = row.cells.get(col.header, "")
        s = v.tex if isinstance(v, Cell) else tex_escape(v)
        return rf"\textbf{{{s}}}" if row.bold and s else s

    def _sections(self) -> list[str | None]:
        seen: list[str | None] = []
        for r in self.rows:
            if r.section not in seen:
                seen.append(r.section)
        return seen

    # -- markdown --
    def to_markdown(self, *, sectioned: bool = True, heading_level: int = 3) -> str:
        """Render as GitHub-flavored markdown; sections become H3 headings."""
        headers = [c.header for c in self.columns]
        aligns = [_MD_ALIGN[c.align] for c in self.columns]
        hbar = "| " + " | ".join(headers) + " |"
        abar = "| " + " | ".join(aligns) + " |"

        def body(rows: list[Row]) -> list[str]:
            return [
                "| " + " | ".join(self._cell_md(r, c) for c in self.columns) + " |"
                for r in rows
            ]

        if not sectioned or self._sections() == [None]:
            return "\n".join([hbar, abar, *body(self.rows)])

        out: list[str] = []
        hashes = "#" * heading_level
        for sec in self._sections():
            srows = [r for r in self.rows if r.section == sec]
            if sec is not None:
                out.append(f"{hashes} {sec}")
                out.append("")
            out.extend([hbar, abar, *body(srows), ""])
        if self.footer is not None:
            out.extend([f"{hashes} Total", "", hbar, abar, *body([self.footer]), ""])
        return "\n".join(out).rstrip("\n")

    # -- latex --
    def to_latex(
        self,
        *,
        caption: str,
        label: str,
        sectioned: bool = True,
        table_env: str = "table*",
        position: str = "t",
        size: str = "footnotesize",
        colsep_pt: int = 4,
        footnote: str | None = None,
        header_comment: str | None = None,
    ) -> str:
        """Render as a booktabs LaTeX table; sections become bold multicolumn rows."""
        colspec = "@{}" + " ".join(c.align for c in self.columns) + "@{}"
        ncol = len(self.columns)
        out: list[str] = []
        if header_comment:
            out.extend(f"% {ln}" for ln in header_comment.splitlines())
        out.append(rf"\begin{{{table_env}}}[{position}]")
        out.append(r"\centering")
        out.append(rf"\{size}")
        out.append(rf"\setlength{{\tabcolsep}}{{{colsep_pt}pt}}")
        out.append(rf"\caption{{{caption}}}")
        out.append(rf"\label{{{label}}}")
        out.append(rf"\begin{{tabular}}{{{colspec}}}")
        out.append(r"\toprule")
        out.append(" & ".join(tex_escape(c.header) for c in self.columns) + r" \\")
        out.append(r"\midrule")

        def emit(rows: list[Row]) -> None:
            for r in rows:
                out.append(
                    " & ".join(self._cell_tex(r, c) for c in self.columns) + r" \\"
                )

        if not sectioned or self._sections() == [None]:
            emit(self.rows)
        else:
            first = True
            for sec in self._sections():
                srows = [r for r in self.rows if r.section == sec]
                if sec is not None:
                    if not first:
                        out.append(r"\addlinespace")
                    out.append(
                        rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textbf{{{tex_escape(sec)}}}}} \\"
                    )
                emit(srows)
                first = False

        if self.footer is not None:
            out.append(r"\midrule")
            emit([self.footer])
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        if footnote:
            out.append(rf"\\[2pt]{{\footnotesize {footnote}}}")
        out.append(rf"\end{{{table_env}}}")
        out.append("")
        return "\n".join(out)


# --- dendron note frontmatter ---------------------------------------------


def read_frontmatter(path: str | Path, *, default_title: str = "Untitled") -> str:
    """Return the leading dendron ``---...---`` block of a note (verbatim), or a
    minimal one if absent -- so a regenerated note keeps its dendron identity.
    """
    path = Path(path)
    if path.exists():
        text = path.read_text()
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                return text[: end + 4]
    return f"---\ntitle: {default_title}\ndesc: ''\n---"
