# experiments/database/scripts/build_bacteria_discovery_queue.py
# [[experiments.database.expansion-bacteria]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/build_bacteria_discovery_queue
r"""The 300-row bacterial discovery queue, typed and hash-pinned.

A sibling of ``build_bacteria_candidate_datasets_table.py`` and deliberately NOT the
same thing. That script holds the CURATED list: every headline figure traces to a
source fetched by hand, and a row is admitted only once its genotype and condition
axes are known. This script holds the QUEUE behind it: 200 *E. coli* and 100
*P. putida* publications found by a metadata sweep, none of whose record counts have
been verified.

The distinction is the point, so it is enforced rather than annotated. A queue row
carries no instance count, because the source asserts none: every cell of the
spreadsheet's ``Instances`` column reads "Verify from supplement". Promoting a row
into the curated table means fetching the paper, counting the axes and writing a
``Candidate``; nothing here is a substitute for that, and a queue row must never be
cited as though its scale were known.

PROVENANCE. The input is one spreadsheet, stored beside this script's output and
pinned by sha256. It is the canonical artifact; its own provenance is recorded in
``SOURCE_NOTE`` because it was produced outside this repository and handed over as a
file, so there is no retrieval command that would reproduce it. The sha256 is checked
on every run and a mismatch is a hard failure, because a silently edited queue would
change which rows look promotable without changing any committed code.

WHAT THE SOURCE ASSERTS, verbatim from its own Overview sheet:
  "Important: this is a 300-candidate discovery queue, not 300 download-verified
  datasets. 'Verify from supplement' is intentional; accession, row count, genotype
  reconstruction, and licensing require campaign-level audit."
That sentence is the reason this file exists separately.

THE OVERLAP with the curated list is computed here, never asserted. The curated
script's ``ranked()`` is imported and called, so the rows matched on DOI are the rows
that script prints today rather than a JSON dump of an older run. Two sets come out of
it: every curated row (``promoted_as`` and ``curated_rank``), and the fifty
recommended builds, its first ``TRANCHE_2`` rows, which ``queue-rows.tex`` removes.

Emits:
  - <results>/candidates/bacteria_discovery_queue.json                     (machine-readable)
  - notes-tex/database/database-expansion-bacteria/tables/queue.tex        (per-class counts)
  - notes-tex/database/database-expansion-bacteria/tables/queue-iso.tex    (the isoprenol rows)
  - notes-tex/database/database-expansion-bacteria/tables/queue-rows.tex   (every row but the fifty)

Run from the repo root:
  python experiments/database/scripts/build_bacteria_discovery_queue.py
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

# The curated sibling. Run as documented, this script's own directory is sys.path[0],
# so the import resolves to the file beside it; importing it builds the curated rows
# and writes nothing, because its outputs are written only by its main().
import build_bacteria_candidate_datasets_table as curated
import pandas as pd
from pydantic import BaseModel, Field, field_validator

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
# The spreadsheet lives under inputs/ rather than results/ because results/ is
# untracked on the rule that everything in it is regenerable, and this file is not:
# it arrived as a file, so there is no retrieval command and the stored copy is the
# only canonical form. See experiments/database/inputs/README.md.
INPUTS = SCRIPT.parent.parent / "inputs"
TEX_DIR = REPO / "notes-tex" / "database" / "database-expansion-bacteria" / "tables"
XLSX = INPUTS / "ecoli-putida-300-discovery-queue.xlsx"
JSON_OUT = RESULTS / "candidates" / "bacteria_discovery_queue.json"

SOURCE_LINE = "%% SOURCE: experiments/database/scripts/build_bacteria_discovery_queue.py"

# The pin. Checked on every run; a mismatch fails rather than warns.
XLSX_SHA256 = "8596b65706a239377e89e16c50ca41ed61cfe643d5f2b2bb46a22e7a83ed6f44"

SOURCE_NOTE = (
    "ecoli-putida-300-discovery-queue.xlsx, 83,359 bytes, sha256 "
    "8596b657..., received 2026-10-02 as a file rather than fetched from a "
    "URL, so there is no retrieval command and the stored copy is the only "
    "canonical form. Its Overview sheet states the sweep method: "
    "'Candidate discovery used Europe PMC publication metadata/API, followed "
    "by rule-based scope filtering and prioritization.' Its own generation "
    "date is given as 2026-10-02."
)


# ---------------------------------------------------------------------------
# Vocabulary. The source's own categories, typed so a changed spreadsheet
# heading fails loudly instead of silently adding a category.
# ---------------------------------------------------------------------------

Organism = Literal["E. coli", "P. putida"]

# The source's portfolio classes, with its own definitions quoted in the document.
Portfolio = Literal["Joint", "ME breadth", "Scale anchor"]

# The source's isoprenol axis. "direct" is its "Direct", meaning the study measures
# isoprenol or isopentenol itself; "adjacent" is its "Adjacent isoprenoid/terpenoid".
# Mapped to our own words so the curated table and the queue share one vocabulary.
Relevance = Literal["direct", "adjacent", "none"]

Wave = Literal["Wave 1", "Wave 2", "Wave 3", "Reserve"]


class QueueRow(BaseModel):
    """One publication in the discovery queue.

    No instance count and no genotype count, by construction: the source asserts
    neither, and inventing one here would make an unverified row indistinguishable
    from a curated one. ``promoted_as`` is the only field this repository adds, and
    it is filled in when a row graduates into the curated table.
    """

    rank: int = Field(ge=1)
    organism: Organism
    title: str
    year: int
    portfolio: Portfolio
    # The source's "Product / readout" column as given. Often a title fragment cut
    # mid-word by the sweep's own extraction ("ing of Pseudomonas putida for ..."),
    # and kept that way, because repairing it would put words in the source's mouth.
    product: str
    wave: Wave
    relevance: Relevance
    priority_score: float
    doi: str
    pmid: str | None
    pmcid: str | None
    authors: str
    url: str
    open_access: bool
    # Set when this row is already a curated Candidate, by DOI match. The queue is
    # then a superset of the curated list and the overlap is measured, not claimed.
    promoted_as: str | None = None
    # That Candidate's 1-based rank in the curated table, so the fifty recommended
    # builds (rank <= TRANCHE_2) separate from the ranked reserve below the cut.
    curated_rank: int | None = Field(default=None, ge=1)

    @field_validator("title", "product")
    @classmethod
    def strip_markup(cls, v: str) -> str:
        """The source carries HTML italics inside titles; LaTeX cannot take them."""
        return re.sub(r"</?i>|</?sub>|</?sup>", "", v).strip()

    @property
    def recommended(self) -> bool:
        """Whether this row is one of the fifty recommended builds."""
        return self.curated_rank is not None and self.curated_rank <= curated.TRANCHE_2


class Queue(BaseModel):
    """The whole queue, with the counts the document reports."""

    source: str
    sha256: str
    n_rows: int
    rows: list[QueueRow]

    @property
    def by_organism(self) -> dict[str, int]:
        """Row count per host."""
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.organism] = out.get(r.organism, 0) + 1
        return out

    @property
    def promoted(self) -> list[QueueRow]:
        """Rows already present anywhere in the curated table, reserve included."""
        return [r for r in self.rows if r.promoted_as]


# ---------------------------------------------------------------------------
# Reading the source.
# ---------------------------------------------------------------------------

RELEVANCE_MAP: dict[str, Relevance] = {
    "Direct": "direct",
    "Adjacent isoprenoid/terpenoid": "adjacent",
}


def norm_doi(text: object) -> str:
    """The bare DOI, lowercased, so it can serve as a join key against the curated list."""
    if not isinstance(text, str):
        return ""
    m = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", text)
    return m.group(1).lower().rstrip(".") if m else ""


def check_pin() -> str:
    """Verify the stored spreadsheet against its recorded hash, or fail."""
    if not XLSX.exists():
        raise FileNotFoundError(
            f"The pinned queue spreadsheet is missing: {XLSX}. It is the canonical "
            "artifact and there is no URL to re-fetch it from."
        )
    got = hashlib.sha256(XLSX.read_bytes()).hexdigest()
    if got != XLSX_SHA256:
        raise ValueError(
            f"sha256 mismatch on {XLSX.name}: recorded {XLSX_SHA256}, found {got}. "
            "The stored artifact changed. Record a new version rather than "
            "overwriting the hash."
        )
    return got


def curated_dois() -> dict[str, tuple[str, int]]:
    """DOI to (curated row name, curated rank), from the curated script's live order.

    Every curated row carries a DOI in its ``url``; one that did not could never be
    matched, so a missing DOI is a hard failure rather than a silently unmatched row.
    A DOI shared by two curated rows fails too, since the later row would otherwise
    overwrite the earlier one's rank and could move a queue row across the cut.
    """
    rows, _moves = curated.ranked()
    out: dict[str, tuple[str, int]] = {}
    for i, c in enumerate(rows, start=1):
        doi = norm_doi(c.url)
        if not doi:
            raise ValueError(f"curated row {c.name!r} has no DOI in its url: {c.url}")
        if doi in out:
            raise ValueError(f"curated rows {out[doi][0]!r} and {c.name!r} share DOI {doi}")
        out[doi] = (c.name, i)
    return out


def read_queue() -> Queue:
    sha = check_pin()
    promoted = curated_dois()
    rows: list[QueueRow] = []
    book = pd.ExcelFile(XLSX)
    for sheet, organism in (("E coli", "E. coli"), ("P putida", "P. putida")):
        frame = book.parse(sheet)
        for _, r in frame.iterrows():
            doi = norm_doi(r["DOI"])
            pmid = r["PMID"]
            pmcid = r["PMCID"]
            # A row with no DOI ("" from norm_doi) cannot match a curated row.
            match = promoted.get(doi) if doi else None
            rows.append(
                QueueRow(
                    rank=int(r["Rank"]),
                    organism=organism,  # type: ignore[arg-type]
                    title=str(r["Dataset / campaign"]),
                    year=int(r["Year"]),
                    portfolio=str(r["Portfolio class"]),  # type: ignore[arg-type]
                    product=str(r["Product / readout"]),
                    wave=str(r["Wave"]),  # type: ignore[arg-type]
                    relevance=RELEVANCE_MAP.get(str(r["Isoprenol relevance"]), "none"),
                    priority_score=float(r["Priority score"]),
                    doi=doi,
                    pmid=None if pd.isna(pmid) else str(int(float(pmid))),
                    pmcid=None if pd.isna(pmcid) else str(pmcid),
                    authors=str(r["Authors"]),
                    url=str(r["Source URL"]),
                    open_access=str(r["Access"]).startswith("Open"),
                    promoted_as=match[0] if match else None,
                    curated_rank=match[1] if match else None,
                )
            )
    return Queue(source=SOURCE_NOTE, sha256=sha, n_rows=len(rows), rows=rows)


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "%% GENERATED FILE -- do not hand-edit.\n" + SOURCE_LINE + "\n" + body
    )
    print(f"Wrote {path.relative_to(REPO)}")


# One escape for both documents' tables: the curated script's, which also covers the
# braces, caret and tilde that this file's own escape used to miss.
esc = curated.tex_escape

# Characters the queue's titles carry that the document's Latin Modern text font has
# no glyph for. Without this map TeX drops them with only a log warning, which is how
# the queue's "β-Alanine" first printed as "-Alanine". Applied AFTER escaping,
# because the replacements are themselves TeX. Every one renders the same character,
# so the source's words are unchanged.
UNICODE_TEX = {
    "\u03b1": r"$\alpha$",
    "\u03b2": r"$\beta$",
    "\u0394": r"$\Delta$",
    "\u03c9": r"$\omega$",
    "\u2011": "-",  # non-breaking hyphen
    "\u2009": r"\,",  # thin space
    "\u00a0": "~",  # no-break space
}


def source_tex(text: str) -> str:
    """Escape source text and map the characters the text font lacks."""
    out = esc(text)
    for a, b in UNICODE_TEX.items():
        out = out.replace(a, b)
    return out


def render_queue(q: Queue) -> str:
    """Per-class counts, and how much of the queue is already curated."""
    classes: list[Portfolio] = ["Joint", "ME breadth", "Scale anchor"]
    body = [
        r"\begingroup",
        r"\footnotesize",
        r"\begin{longtable}{@{}L{34mm} r r r r r@{}}",
        r"\caption[]{The discovery queue by class and host. \emph{Curated} counts rows "
        r"already present in Table~\ref{tab:bfinal}, matched on DOI, so the overlap is "
        r"measured rather than asserted. \emph{Direct} and \emph{Adjacent} are the "
        r"source's isoprenol axis. No row carries an instance count, because the source "
        r"asserts none.}",
        r"\label{tab:bqueue}\\",
        r"\toprule",
        r"\textbf{Class} & \textbf{\emph{E. coli}} & \textbf{\emph{P. putida}} & "
        r"\textbf{Curated} & \textbf{Direct} & \textbf{Adjacent} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for k in classes:
        sel = [r for r in q.rows if r.portfolio == k]
        body.append(
            f"{esc(k)} & "
            f"{len([r for r in sel if r.organism == 'E. coli'])} & "
            f"{len([r for r in sel if r.organism == 'P. putida'])} & "
            f"{len([r for r in sel if r.promoted_as])} & "
            f"{len([r for r in sel if r.relevance == 'direct'])} & "
            f"{len([r for r in sel if r.relevance == 'adjacent'])} \\\\"
        )
    body.append(r"\midrule")
    body.append(
        r"\textbf{Total} & "
        f"\\textbf{{{q.by_organism.get('E. coli', 0)}}} & "
        f"\\textbf{{{q.by_organism.get('P. putida', 0)}}} & "
        f"\\textbf{{{len(q.promoted)}}} & "
        f"\\textbf{{{len([r for r in q.rows if r.relevance == 'direct'])}}} & "
        f"\\textbf{{{len([r for r in q.rows if r.relevance == 'adjacent'])}}} \\\\"
    )
    body += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(body) + "\n"


def render_iso(q: Queue) -> str:
    """Every isoprenol-relevant row in the queue, with its curation state.

    The table the request turns on, so it prints all of them rather than a count:
    which isoprenol papers exist, which are already curated, and which are the
    remaining work.
    """
    sel = sorted(
        [r for r in q.rows if r.relevance != "none"],
        key=lambda r: (r.relevance != "direct", r.organism, -r.priority_score),
    )
    body = [
        r"\begin{landscape}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}L{12mm} L{13mm} r L{104mm} L{38mm}@{}}",
        r"\caption[]{Every isoprenol-relevant publication the sweep returned, the "
        r"source's own flag. \emph{Direct} measures isoprenol or isopentenol itself; "
        r"\emph{adjacent} measures another isoprenoid or terpenoid, so it carries the "
        r"precursor pathway without the product. \emph{Curated} names the row in "
        r"Table~\ref{tab:bfinal} when the paper is already there, and is otherwise the "
        r"remaining work.}",
        r"\label{tab:bqueueiso}\\",
        r"\toprule",
        r"\textbf{Flag} & \textbf{Host} & \textbf{Year} & \textbf{Title} & "
        r"\textbf{Curated} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{@{}l}{\footnotesize\emph{Table~\ref{tab:bqueueiso}, continued}}\\",
        r"\toprule",
        r"\textbf{Flag} & \textbf{Host} & \textbf{Year} & \textbf{Title} & "
        r"\textbf{Curated} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for r in sel:
        title = r.title
        if len(title) > 150:
            title = title[:147] + "..."
        # Quoted, and that is load-bearing rather than decorative: these are the
        # publishers' own titles, so British spellings in them must survive. The
        # style gate skips ``...'' spans for exactly that reason, and an
        # unquoted title would be Americanized into a falsified citation.
        body.append(
            f"{r.relevance} & \\emph{{{esc(r.organism)}}} & {r.year} & "
            f"``{source_tex(title)}'' & {esc(r.promoted_as) if r.promoted_as else '--'} \\\\"
        )
    body += [r"\end{longtable}", r"\endgroup", r"\end{landscape}"]
    return "\n".join(body) + "\n"


# The two generic labels the source uses where it extracted no product. Counted so
# the caption can say how much of the Product column is a placeholder.
GENERIC_PRODUCTS = ("Genome-scale/omics phenotype", "Metabolic-engineering phenotype")


class RowsCounts(BaseModel):
    """The figures the row table's caption and the document's prose quote.

    Computed once, printed by ``main`` and written into the caption from the same
    object, so a number in the PDF cannot disagree with the number on the console.
    """

    n_queue: int
    n_removed: int
    removed: dict[str, int]
    kept: dict[str, int]
    n_kept: int
    n_reserve: int
    n_no_doi: int
    n_generic_product: int
    n_title_fragment: int
    n_midword: int


def cut_midword(product: str, title: str) -> bool:
    """Whether a product phrase taken from the title starts or ends inside a word."""
    i = title.find(product)
    j = i + len(product)
    return (i > 0 and title[i - 1].isalnum()) or (j < len(title) and title[j].isalnum())


def rows_counts(q: Queue) -> RowsCounts:
    """Count what the row table removes, keeps and carries as placeholder text."""
    kept = [r for r in q.rows if not r.recommended]
    removed = [r for r in q.rows if r.recommended]
    hosts: list[Organism] = ["E. coli", "P. putida"]
    fragments = [
        r
        for r in kept
        if r.product not in GENERIC_PRODUCTS
        and r.product != r.title
        and r.product in r.title
    ]
    return RowsCounts(
        n_queue=q.n_rows,
        n_removed=len(removed),
        removed={h: sum(r.organism == h for r in removed) for h in hosts},
        kept={h: sum(r.organism == h for r in kept) for h in hosts},
        n_kept=len(kept),
        n_reserve=sum(r.curated_rank is not None for r in kept),
        n_no_doi=sum(not r.doi for r in kept),
        n_generic_product=sum(r.product in GENERIC_PRODUCTS for r in kept),
        # A product that is a substring of the title but not the whole title is the
        # sweep's extraction cutting a phrase out of it, sometimes mid-word.
        n_title_fragment=len(fragments),
        n_midword=sum(cut_midword(r.product, r.title) for r in fragments),
    )


def doi_link(r: QueueRow) -> str:
    """The DOI as a resolvable link, or the source record's own URL when it has none."""
    if r.doi:
        return r"\href{https://doi.org/" + r.doi + "}{" + curated.breakable(r.doi) + "}"
    return curated.link_tex(r.url)


def render_rows(q: Queue, n: RowsCounts) -> str:
    """Every queue row except the fifty recommended builds, matched on DOI.

    Ordered by host, then by the source's own rank within the host, so a row can be
    found from the spreadsheet. Rows already in the ranked reserve of the curated
    table stay in, marked, because only the fifty were removed.
    """
    sel = sorted(
        [r for r in q.rows if not r.recommended], key=lambda r: (r.organism, r.rank)
    )
    hdr = (
        r"\textbf{Rank} & \textbf{Host} & \textbf{Dataset or campaign} & "
        r"\textbf{Year} & \textbf{Class} & \textbf{Product or readout} & "
        r"\textbf{Isoprenol} & \textbf{DOI} \\"
    )
    ec, pp = "E. coli", "P. putida"
    caption = (
        r"\caption[]{The discovery queue less the fifty recommended builds: every row of "
        rf"the {n.n_queue}-publication sweep except the {n.n_removed} whose DOI matches "
        rf"one of rows 1--{curated.TRANCHE_2} of Table~\ref{{tab:bfinal}} "
        rf"({n.removed[ec]} \org{{E. coli}}, {n.removed[pp]} \org{{P. putida}}), "
        rf"leaving {n.n_kept}: {n.kept[ec]} \org{{E. coli}} and {n.kept[pp]} "
        r"\org{P. putida}. The match is on the DOI, lowercased and without its "
        r"resolver prefix. No row carries a verified instance count, because the "
        r"source asserts none. \emph{Rank} is the source's rank within its host, "
        r"and a superscript \textbf{R} marks the "
        rf"{n.n_reserve} rows already curated in the ranked reserve below the cut. "
        r"\emph{Class} is the source's portfolio class (Sec.~\ref{sec:queue}) and "
        r"\emph{Isoprenol} its isoprenol flag. \emph{Product or readout} is the "
        rf"source's field as given: {n.n_generic_product} rows carry one of its two "
        rf"generic labels and {n.n_title_fragment} a phrase taken from the title, "
        rf"{n.n_midword} of those cut mid-word. "
        + rf"Where the source gives no DOI ({n.n_no_doi} "
        + ("row" if n.n_no_doi == 1 else "rows")
        + "), the DOI column links the source record instead.}"
    )
    body = [
        r"\begin{landscape}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{longtable}{@{}r@{\hspace{4pt}} L{14mm} L{96mm} r L{18mm} "
        r"L{45mm} L{14mm} L{38mm}@{}}",
        caption,
        r"\label{tab:bqueuerows}\\",
        r"\toprule",
        hdr,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{@{}l}{\footnotesize\emph{Table~\ref{tab:bqueuerows}, "
        r"continued}}\\",
        r"\toprule",
        hdr,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    prev: str | None = None
    for r in sel:
        if prev is not None and r.organism != prev:
            body.append(r"\midrule")
        prev = r.organism
        mark = r"\,\textsuperscript{\textbf{R}}" if r.curated_rank is not None else ""
        flag = {"direct": "direct", "adjacent": "adjacent", "none": "--"}[r.relevance]
        # \sourcetext, not quotation marks: titles and the product field are the
        # source's words, so the style gate must skip them (a British spelling in
        # a publisher's title is correct there), and quote marks on every cell of a
        # table this long would be noise.
        body.append(
            " & ".join(
                [
                    f"{r.rank}{mark}",
                    curated.org_tex(r.organism),
                    r"\sourcetext{" + source_tex(r.title) + "}",
                    str(r.year),
                    esc(r.portfolio),
                    r"\sourcetext{" + source_tex(r.product) + "}",
                    flag,
                    doi_link(r),
                ]
            )
            + r" \\"
        )
        body.append(r"\addlinespace[2pt]")
    body += [r"\end{longtable}", r"\endgroup", r"\end{landscape}"]
    return "\n".join(body) + "\n"


def main() -> None:
    q = read_queue()
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(q.model_dump_json(indent=2) + "\n")
    print(f"Wrote {JSON_OUT.relative_to(REPO)}")
    write(TEX_DIR / "queue.tex", render_queue(q))
    write(TEX_DIR / "queue-iso.tex", render_iso(q))
    n = rows_counts(q)
    write(TEX_DIR / "queue-rows.tex", render_rows(q, n))

    direct = [r for r in q.rows if r.relevance == "direct"]
    adjacent = [r for r in q.rows if r.relevance == "adjacent"]
    print(
        f"{q.n_rows} queue rows "
        f"({q.by_organism.get('E. coli', 0)} E. coli, "
        f"{q.by_organism.get('P. putida', 0)} P. putida); "
        f"{len(q.promoted)} already curated"
    )
    print(
        f"isoprenol: {len(direct)} direct "
        f"({len([r for r in direct if r.promoted_as])} curated), "
        f"{len(adjacent)} adjacent "
        f"({len([r for r in adjacent if r.promoted_as])} curated)"
    )
    # The figures the row table's caption carries and sections/queue.tex quotes.
    print(
        f"queue-rows: {n.n_removed} of {n.n_queue} removed as the fifty recommended "
        f"builds ({n.removed['E. coli']} E. coli, {n.removed['P. putida']} P. putida); "
        f"{n.n_kept} kept ({n.kept['E. coli']} E. coli, {n.kept['P. putida']} P. putida), "
        f"{n.n_reserve} of them already in the curated reserve"
    )
    print(
        f"queue-rows: {n.n_no_doi} without a DOI, {n.n_generic_product} with a generic "
        f"product label, {n.n_title_fragment} with a product taken from the title "
        f"({n.n_midword} cut mid-word)"
    )


if __name__ == "__main__":
    main()
