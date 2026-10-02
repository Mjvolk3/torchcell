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

Emits:
  - <results>/candidates/bacteria_discovery_queue.json          (machine-readable)
  - notes-tex/database-expansion-bacteria/tables/queue.tex      (per-class counts)
  - notes-tex/database-expansion-bacteria/tables/queue-iso.tex  (the isoprenol rows)

Run from the repo root:
  python experiments/database/scripts/build_bacteria_discovery_queue.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

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
TEX_DIR = REPO / "notes-tex" / "database-expansion-bacteria" / "tables"
XLSX = INPUTS / "ecoli-putida-300-discovery-queue.xlsx"
JSON_OUT = RESULTS / "candidates" / "bacteria_discovery_queue.json"
CURATED_JSON = RESULTS / "candidates" / "bacteria_candidate_datasets.json"

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

    @field_validator("title")
    @classmethod
    def strip_markup(cls, v: str) -> str:
        """The source carries HTML italics inside titles; LaTeX cannot take them."""
        return re.sub(r"</?i>|</?sub>|</?sup>", "", v).strip()


class Queue(BaseModel):
    """The whole queue, with the counts the document reports."""

    source: str
    sha256: str
    n_rows: int
    rows: list[QueueRow]

    @property
    def by_organism(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.organism] = out.get(r.organism, 0) + 1
        return out

    @property
    def promoted(self) -> list[QueueRow]:
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


def curated_dois() -> dict[str, str]:
    """DOI to curated row name, so the overlap is computed rather than asserted."""
    if not CURATED_JSON.exists():
        return {}
    payload = json.loads(CURATED_JSON.read_text())
    return {norm_doi(c["url"]): c["name"] for c in payload["candidates"]}


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
            rows.append(
                QueueRow(
                    rank=int(r["Rank"]),
                    organism=organism,  # type: ignore[arg-type]
                    title=str(r["Dataset / campaign"]),
                    year=int(r["Year"]),
                    portfolio=str(r["Portfolio class"]),  # type: ignore[arg-type]
                    wave=str(r["Wave"]),  # type: ignore[arg-type]
                    relevance=RELEVANCE_MAP.get(str(r["Isoprenol relevance"]), "none"),
                    priority_score=float(r["Priority score"]),
                    doi=doi,
                    pmid=None if pd.isna(pmid) else str(int(float(pmid))),
                    pmcid=None if pd.isna(pmcid) else str(pmcid),
                    authors=str(r["Authors"]),
                    url=str(r["Source URL"]),
                    open_access=str(r["Access"]).startswith("Open"),
                    promoted_as=promoted.get(doi),
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


def esc(text: str) -> str:
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
    ):
        text = text.replace(a, b)
    return text


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
            f"``{esc(title)}'' & {esc(r.promoted_as) if r.promoted_as else '--'} \\\\"
        )
    body += [r"\end{longtable}", r"\endgroup", r"\end{landscape}"]
    return "\n".join(body) + "\n"


def main() -> None:
    q = read_queue()
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(q.model_dump_json(indent=2) + "\n")
    print(f"Wrote {JSON_OUT.relative_to(REPO)}")
    write(TEX_DIR / "queue.tex", render_queue(q))
    write(TEX_DIR / "queue-iso.tex", render_iso(q))

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


if __name__ == "__main__":
    main()
