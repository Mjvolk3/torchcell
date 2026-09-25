# experiments/database/scripts/build_bacteria_candidate_datasets_table.py
# [[experiments.database.expansion-bacteria]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/build_bacteria_candidate_datasets_table
r"""The bacterial candidate list: *E. coli* and *P. putida* datasets to ingest next.

The sibling of ``build_candidate_datasets_table.py``, which ranks the next fifty
*S. cerevisiae* datasets. This is the CURATION for the two bacterial hosts, held as
data: nothing here can be recomputed from a store, so the judgment itself is the
artifact and it lives in a committed script.

ORDERING RULE, and it differs from the yeast list on purpose. The yeast list bands
by what a row supplies to a planned Perturb-seq campaign, then ranks inside a band.
Here the requested rule is scale and density directly: rows sort by MEASUREMENTS,
meaning instances times phenotype dimensionality, descending. That single quantity
is what "most instances and highest density" resolves to, because a scalar screen
of n records contributes n numbers while a proteome panel of n records contributes
n times the panel depth. Instances break a measurement tie and tier breaks that,
so the rule is total and reproducible.

STAGING. The build is planned in two tranches, 20 rows then 30:
  tranche 1  the first 20, with a per-organism quota of 10 and 10, so neither host
             waits behind the other's larger screens
  tranche 2  the next 30 by the same global rule
The quota is applied AFTER ranking and is reported row by row, so the effect of the
quota on the order is auditable rather than folded into a score.

Emits, off the same records:
  - notes-tex/database-expansion-bacteria/tables/final.tex       (the ranked list)
  - notes-tex/database-expansion-bacteria/tables/sources.tex     (citation, link, data)
  - notes-tex/database-expansion-bacteria/tables/counts.tex      (per-class, per-organism)
  - notes-tex/database-expansion-bacteria/tables/summary.tex     (summary statistics)
  - notes-tex/database-expansion-bacteria/tables/analogs.tex     (yeast analog per row)
  - notes-tex/database-expansion-bacteria/tables/schema.tex      (what the schema needs)
  - notes-tex/database-expansion-bacteria/tables/excluded.tex    (considered and dropped)
  - <results>/candidates/bacteria_candidate_datasets.json        (machine-readable dump)

Run from the repo root:
  python experiments/database/scripts/build_bacteria_candidate_datasets_table.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
TEX_DIR = REPO / "notes-tex" / "database-expansion-bacteria" / "tables"
JSON_OUT = RESULTS / "candidates" / "bacteria_candidate_datasets.json"

SOURCE_LINE = (
    "%% SOURCE: experiments/database/scripts/build_bacteria_candidate_datasets_table.py"
)

# The database holds 51 schematized and L0-L4-verified S. cerevisiae datasets and no
# bacterial dataset at all. This list is the bacterial tranche: a target of 50 rows,
# built in two waves.
YEAST_BUILT = 51
BACTERIA_BUILT = 0
TARGET_COUNT = 50

# The two tranches the build week is planned against.
TRANCHE_1 = 20
TRANCHE_2 = 50
# Per-organism quota inside tranche 1, so both hosts start together.
QUOTA_1 = {"E. coli": 10, "P. putida": 10}

# ---------------------------------------------------------------------------
# Vocabulary. Defined here so it is defined before use in the document, and so a
# typo becomes a validation error rather than a silently novel category.
# ---------------------------------------------------------------------------

Organism = Literal["E. coli", "P. putida"]

Klass = Literal[
    "Fitness / chemical genomics",
    "Transposon fitness",
    "CRISPR library screen",
    "Genetic interaction",
    "Production campaign",
    "Combinatorial design",
    "Tolerance / robustness",
    "Transcriptome",
    "Proteome",
    "Metabolome / flux",
    "Translation / turnover",
    "Multi-omics campaign",
    "Modality / backbone",
    "Aggregation / support",
]

# How the total genomic content of one strain would be reconstructed. The same hard
# gate the yeast list applies: a row with no route to a sequence cannot train a
# genotype-to-phenotype model and is excluded rather than ranked low. The bases are
# named per host because the reference differs, and because a b-number and a PP_
# locus tag are different namespaces that must not be silently merged.
SeqBasis = Literal[
    "MG1655-KO",  # cataloged single deletion in a sequenced reference (Keio)
    "MG1655-KO x KO",  # a constructed double mutant; both loci cataloged
    "MG1655+transposon",  # mapped insertion site, barcoded
    "MG1655+guide",  # genome unedited; the perturbation is a cassette plus a guide
    "KT2440-KO",
    "KT2440+transposon",
    "KT2440+guide",
    "KT2440+promoter",  # reference plus a designed promoter cassette
    "engineered-chassis",  # named production strain plus heterologous cassettes
    "engineered-chassis+RBS",  # the chassis plus a designed ribosome binding site
    "engineered-chassis+promoter",
    "evolved-WGS",  # a resequenced evolved clone; unsequenced populations are excluded
    "reference-only",  # wild type; the environment carries the perturbation
]

Basis = Literal["reported", "product", "estimate"]

# Ingestion state. Every row here starts as a candidate because no bacterial dataset
# is built, but two other states matter: a corpus that re-serves other papers is an
# "aggregation" and its records are not net new until it is split by source, and a
# row whose per-record values are not released is "blocked".
Status = Literal["candidate", "blocked", "aggregation"]

# How well a row's numbers and citation were checked. "sourced" means every headline
# figure traces to a source fetched this pass; "recall" means the dataset is real and
# the description sound, but the counts and accession need confirming before a loader.
Confidence = Literal["sourced", "recall"]


class Synergy(BaseModel):
    """What joining this row to another buys, and on what key.

    A synergy is only real if the two datasets share an addressable axis, so the
    join key is a required field rather than a remark. For this list the keys are
    the Keio collection, a named transposon library, a guide library over one gene
    set, a condition panel, a product pathway, or a single gene.
    """

    partner: str
    partner_status: Literal["supported", "candidate"]
    join: str
    yields: str


class Analog(BaseModel):
    """The already-built S. cerevisiae dataset a bacterial row mirrors.

    Recorded per row because the request was for bacterial data that matches the
    schema already in use. A row with a named yeast analog needs no new phenotype
    class: the loader writes the same record type against a different reference
    genome. A row with no analog is naming a phenotype the substrate has never
    held, which is a larger piece of work and is flagged as such.
    """

    dataset: str
    why: str


class Candidate(BaseModel):
    """One bacterial dataset proposed for ingestion."""

    name: str
    organism: Organism
    citation: str
    url: str
    klass: Klass
    tier: int = Field(ge=1, le=4)
    genotypes_n: int | None
    genotypes: str
    env_n: int | None
    env: str
    instances_n: int | None
    instances_basis: Basis
    phenotype: str
    dim: int = 1  # phenotype vector length; 1 for a scalar label
    dim_basis: Basis = "reported"
    seq_basis: SeqBasis
    modality: str
    why: str
    accession: str
    accession_confirmed: bool = False
    status: Status = "candidate"
    confidence: Confidence = "sourced"
    analog: Analog | None = None
    synergy: list[Synergy] = Field(default_factory=list)
    time_axis: str = ""
    # What the schema must gain before this row can be written, empty when the
    # existing classes already cover it apart from the shared bacterial blockers.
    schema_need: str = ""

    @property
    def measurements(self) -> int | None:
        """Instances times phenotype dimensionality, the quantity rows are ranked on.

        Ranking on instances alone silently prefers a scalar fitness screen over a
        vector-valued omics panel of the same size: a 22-condition proteome is 22
        instances but roughly 50,000 numbers. Density is the second half of the
        requested ordering and this is where it enters.
        """
        return None if self.instances_n is None else self.instances_n * self.dim

    @property
    def sort_key(self) -> tuple[float, float, int]:
        """Measurements descending, then instances descending, then tier.

        Total and reproducible: no two rows can tie on all three unless they are
        the same size in every respect, and the log keeps a 10^8 row from
        overflowing the comparison.
        """
        return (
            -math.log10(max(self.measurements or 1, 1)),
            -math.log10(max(self.instances_n or 1, 1)),
            self.tier,
        )


class Excluded(BaseModel):
    """A dataset considered and dropped, with the rule that dropped it."""

    name: str
    reason: str
    rule: Literal["no-sequence", "off-host", "not-a-dataset", "no-per-record-data"]


class SchemaNeed(BaseModel):
    """One change the schema needs before any bacterial row can be written.

    Separated from the per-row ``schema_need`` because these are shared: every row
    in the table waits on them, so they are the critical path rather than a
    per-dataset cost. ``evidence`` is how the blocker was established, and for the
    two hard ones it is a command that was actually run.
    """

    what: str
    where: str
    blocks: str
    evidence: str
    additive: bool  # True when the change adds a class and moves no served closure


# ---------------------------------------------------------------------------
# The shared blockers. Established by running the validators, not by reading them:
# see the dated section of notes/experiments.database.expansion-bacteria.md.
# ---------------------------------------------------------------------------

SCHEMA_NEEDS: list[SchemaNeed] = [
    SchemaNeed(
        what=(
            "A bacterial gene-identifier namespace on GenePerturbation, so a "
            "b-number or a PP_ locus tag validates as a systematic gene name"
        ),
        where="torchcell/datamodels/schema.py, GenePerturbation.validate_sys_gene_name",
        blocks="every row in the table that perturbs a named gene",
        evidence=(
            "The validator admits only Y[A-P][LR]NNN[WC], QNNNN and YNC[A-Q]NNNN[WC]. "
            "Constructing a DeletionPerturbation with b0002, PP_0001, ECK0002 or thrA "
            "raises 'Invalid systematic gene name format'; YAL001C is accepted"
        ),
        additive=True,
    ),
    SchemaNeed(
        what=(
            "A genomes-tier assembly set per host, so a strain's sequence resolves "
            "the way a yeast strain's does"
        ),
        where="torchcell/sequence/genome/registry.py and torchcell/sequence/genome/",
        blocks="the sequence basis of every row, and so the sequence-reconstruction gate",
        evidence=(
            "The tier defines two assembly sets, both S. cerevisiae "
            "(sgd_S288C_R64-4-1_20230830 and peter2018_1011_assemblies), and the "
            "genome package holds only a scerevisiae subpackage"
        ),
        additive=True,
    ),
    SchemaNeed(
        what="Nothing on the reference side",
        where="torchcell/datamodels/schema.py, ReferenceGenome",
        blocks="nothing",
        evidence=(
            "species and strain are free strings, so ReferenceGenome("
            "species='Escherichia coli', strain='MG1655') and the KT2440 equivalent "
            "both validate unchanged"
        ),
        additive=True,
    ),
]


# ---------------------------------------------------------------------------
# Candidates. Figures marked confidence="sourced" trace to a source fetched this
# pass; "recall" rows carry counts that need confirming before a loader is written.
# ---------------------------------------------------------------------------

CANDIDATES: list[Candidate] = []

EXCLUDED: list[Excluded] = []


# ---------------------------------------------------------------------------
# Rendering. The same helpers and table shapes the yeast document uses, so the two
# read as one family and a reader who knows one table can read the other.
# ---------------------------------------------------------------------------


def tex_escape(s: str) -> str:
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def link_tex(url: str) -> str:
    """A clickable, short display form of a source URL.

    The full URL is too wide for a table column and mostly boilerplate, so the
    common prefixes collapse to a label while the href keeps the real target.
    Break points go after every slash, since a bare host path is one unbreakable
    token to TeX.
    """
    shown = url
    for prefix, label in (
        ("https://doi.org/", "doi:"),
        ("https://pubmed.ncbi.nlm.nih.gov/", "PMID "),
        ("https://pmc.ncbi.nlm.nih.gov/articles/", ""),
        ("https://www.ncbi.nlm.nih.gov/pmc/articles/", ""),
        ("https://www.", ""),
        ("https://", ""),
    ):
        if shown.startswith(prefix):
            shown = label + shown[len(prefix) :]
            break
    shown = shown.rstrip("/")
    body = tex_escape(shown).replace("/", r"/\allowbreak ")
    return r"\href{" + url + r"}{" + body + r"}"


def status_tex(status: str) -> str:
    """Marker for a row that is not a plain candidate."""
    return {
        "candidate": "",
        "blocked": r"\,\textsuperscript{\textbf{B}}",
        "aggregation": r"\,\textsuperscript{\textbf{A}}",
    }[status]


def seq_tex(basis: str) -> str:
    """Sequence-basis label with break points at + and -.

    "engineered-chassis+RBS" is one unbreakable token to TeX, so a narrow column
    cannot wrap it and the row runs off the text block.
    """
    return (
        tex_escape(basis).replace("+", r"+\allowbreak ").replace("-", r"-\allowbreak ")
    )


def org_tex(organism: str) -> str:
    """Host abbreviation, italic per house style for an organism name."""
    return r"\org{" + tex_escape(organism) + r"}"


def sci(n: int | None) -> str:
    """A count as an order-of-magnitude figure, the way the built table reports it."""
    if n is None:
        return "--"
    if n < 1000:
        return f"{n:,}"
    exp = int(math.floor(math.log10(n)))
    mant = n / 10**exp
    return f"${mant:.1f}\\times 10^{{{exp}}}$"


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "%% GENERATED FILE -- do not hand-edit.\n" + SOURCE_LINE + "\n" + body
    )
    print(f"Wrote {path.relative_to(REPO)}")


def ranked() -> tuple[list[Candidate], list[tuple[str, int, int]]]:
    """Rank by measurements, then apply the tranche-1 per-organism quota.

    Returns the ordered rows and the quota moves as (name, from_rank, to_rank), so
    the document can report every row the quota promoted rather than hiding the
    reordering inside a score.
    """
    rows = sorted(CANDIDATES, key=lambda c: c.sort_key)
    if not rows:
        return rows, []

    # Tranche 1 by the global rule alone, then corrected to the quota. A host whose
    # screens are all smaller would otherwise not appear in the first tranche at
    # all, and the request was to start both hosts together.
    moves: list[tuple[str, int, int]] = []
    for organism, quota in QUOTA_1.items():
        while True:
            have = [c for c in rows[:TRANCHE_1] if c.organism == organism]
            if len(have) >= quota:
                break
            below = [c for c in rows[TRANCHE_1:] if c.organism == organism]
            if not below:
                break
            promote = below[0]
            # Displace the weakest row of the OTHER host that is over its own quota,
            # so a promotion never costs a row the quota is protecting.
            others = [
                c
                for c in rows[:TRANCHE_1]
                if c.organism != organism
                and len([d for d in rows[:TRANCHE_1] if d.organism == c.organism])
                > QUOTA_1.get(c.organism, 0)
            ]
            if not others:
                break
            demote = max(others, key=lambda c: c.sort_key)
            old = rows.index(promote) + 1
            rows.remove(promote)
            rows.insert(rows.index(demote), promote)
            rows.remove(demote)
            rows.insert(TRANCHE_1, demote)
            moves.append((promote.name, old, rows.index(promote) + 1))
    return rows, moves


def render_final(rows: list[Candidate]) -> str:
    """The ranked list, every statistic the ranking used beside the reason."""
    cols = (
        r"@{}r@{\hspace{3pt}} L{38mm} L{13mm} L{10mm} L{19mm} L{17mm} r@{\hspace{4pt}} "
        r"r@{\hspace{4pt}} L{19mm} L{74mm}@{}"
    )
    hdr = (
        r"\textbf{\#} & \textbf{Dataset (class; phenotype)} & \textbf{Host} & "
        r"\textbf{Tier} & \textbf{Genotypes} & \textbf{Env} & \textbf{Inst.} & "
        r"\textbf{Meas.} & \textbf{Sequence basis} & \textbf{Why} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{"""
        + cols
        + r"""}
\caption[]{The bacterial candidates, ranked by \emph{Meas.} descending, which is
instances times phenotype dimensionality (Sec.~\ref{sec:rule}). \emph{Genotypes} and
\emph{Env} are the perturbation and condition axes. \emph{Inst.} is
genotype$\times$environment records, $\dagger$ where it is the product of the two axes
rather than a reported count and $\ddagger$ where it is an order-of-magnitude estimate.
\emph{Sequence basis} is the route to each strain's total genomic content; a row with no
route is excluded (Table~\ref{tab:bexcluded}). Superscript \textbf{B} marks a row whose
per-record values are not released, \textbf{A} a corpus that re-serves other papers and
is not net new until split by source. Citations and data locations are in
Table~\ref{tab:bsources}; the yeast dataset each row mirrors is in
Table~\ref{tab:banalogs}.}
\label{tab:bfinal}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{10}{@{}l}{\footnotesize\emph{Table~\ref{tab:bfinal}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    lines = []
    dividers = {
        TRANCHE_1 + 1: (
            r"End of tranche 1. Rows 1--"
            + str(TRANCHE_1)
            + r" are the first builds, "
            + str(QUOTA_1["E. coli"])
            + r" per host."
        ),
        TRANCHE_2 + 1: (
            r"End of tranche 2. Rows "
            + str(TRANCHE_1 + 1)
            + r"--"
            + str(TRANCHE_2)
            + r" complete the fifty; rows below are the ranked reserve."
        ),
    }
    for i, c in enumerate(rows, start=1):
        if i in dividers:
            lines.append(
                r"\midrule \multicolumn{10}{@{}l}{\textbf{"
                + dividers[i]
                + r"}}\\ \midrule"
            )
        mark = {"reported": "", "product": r"$\dagger$", "estimate": r"$\ddagger$"}[
            c.instances_basis
        ]
        dataset = (
            r"\textbf{"
            + tex_escape(c.name)
            + r"}"
            + status_tex(c.status)
            + r"\newline {\scriptsize "
            + tex_escape(c.klass)
            + "; "
            + tex_escape(c.phenotype)
            + "}"
        )
        lines.append(
            " & ".join(
                [
                    str(i),
                    dataset,
                    org_tex(c.organism),
                    str(c.tier),
                    tex_escape(c.genotypes),
                    tex_escape(c.env),
                    sci(c.instances_n) + mark,
                    sci(c.measurements),
                    seq_tex(c.seq_basis),
                    tex_escape(c.why),
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_sources(rows: list[Candidate]) -> str:
    hdr = (
        r"\textbf{\#} & \textbf{Dataset, citation and link} & "
        r"\textbf{Modality and readout} & \textbf{Data} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}r@{\hspace{4pt}} L{86mm} L{74mm} L{78mm}@{}}
\caption[]{Sources for Table~\ref{tab:bfinal}, in the same order. \emph{Data} is where
the per-record values live; an entry marked unconfirmed was not fetched live and must be
checked before a loader is written. Every link is clickable.}
\label{tab:bsources}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{4}{@{}l}{\footnotesize\emph{Table~\ref{tab:bsources}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    lines = []
    for i, c in enumerate(rows, start=1):
        cite = (
            r"\textbf{"
            + tex_escape(c.name)
            + r"}\newline "
            + tex_escape(c.citation)
            + r"\newline "
            + link_tex(c.url)
        )
        modality = tex_escape(c.modality) + r"\newline {\scriptsize " + (
            tex_escape(c.time_axis) if c.time_axis else "endpoint or steady state"
        ) + "}"
        acc = tex_escape(c.accession).replace("/", r"/\allowbreak ")
        if not c.accession_confirmed:
            acc += r"\newline {\scriptsize (unconfirmed)}"
        lines.append(" & ".join([str(i), cite, modality, acc]) + r" \\")
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_analogs(rows: list[Candidate]) -> str:
    """The built yeast dataset each bacterial row mirrors, and what the schema needs.

    The column that answers whether a row fits the existing record types: a row
    with a named analog writes the same record against a different reference, and
    a row without one is naming a phenotype the substrate has never held.
    """
    hdr = (
        r"\textbf{\#} & \textbf{Bacterial row} & \textbf{Built yeast analog} & "
        r"\textbf{Why it is the analog} & \textbf{What the schema still needs} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}r@{\hspace{4pt}} L{40mm} L{40mm} L{74mm} L{74mm}@{}}
\caption[]{What each bacterial row maps onto in the supported set. A named analog means
the loader writes a record type the schema already holds, against a different reference
genome, so the work is the retrieval and the provenance rather than a new phenotype
class. \emph{What the schema still needs} is per-row and excludes the two blockers every
row shares (Table~\ref{tab:bschema}); a dash means those two are the whole cost.}
\label{tab:banalogs}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{5}{@{}l}{\footnotesize\emph{Table~\ref{tab:banalogs}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    lines = []
    for i, c in enumerate(rows, start=1):
        analog = r"\textbf{" + tex_escape(c.analog.dataset) + r"}" if c.analog else "--"
        why = tex_escape(c.analog.why) if c.analog else "No built row measures this."
        lines.append(
            " & ".join(
                [
                    str(i),
                    tex_escape(c.name) + " " + org_tex(c.organism),
                    analog,
                    why,
                    tex_escape(c.schema_need) if c.schema_need else "--",
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_schema() -> str:
    """The blockers every row shares, with how each was established."""
    head = r"""\begingroup
\footnotesize
\begin{longtable}{@{}L{52mm} L{46mm} L{72mm}@{}}
\caption[]{What the schema needs before any bacterial row can be written, and how each
was established. The first two are the critical path: every row in
Table~\ref{tab:bfinal} waits on them, and both are additive, so they add classes and
move no served closure. \emph{Evidence} is the result of running the validator, not a
reading of it.}
\label{tab:bschema}\\
\toprule
Change & Where & Evidence \\
\midrule
\endfirsthead
\toprule
Change & Where & Evidence \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    lines = []
    for s in SCHEMA_NEEDS:
        lines.append(
            " & ".join(
                [
                    r"\textbf{" + tex_escape(s.what) + r"}",
                    r"\file{" + tex_escape(s.where) + r"}",
                    tex_escape(s.evidence) + ". Blocks: " + tex_escape(s.blocks) + ".",
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_excluded() -> str:
    head = r"""\begingroup
\footnotesize
\begin{longtable}{@{}L{58mm} L{28mm} L{83mm}@{}}
\caption[]{Considered and dropped. \emph{no-sequence} is the hard gate: without a route
to the strain's genomic content there is no genotype to map a phenotype from.
\emph{no-per-record-data} is a real experiment whose released form is a figure or a
summary rather than per-strain values.}
\label{tab:bexcluded}\\
\toprule
Dataset or group & Rule & Reason \\
\midrule
\endfirsthead
\toprule
Dataset or group & Rule & Reason \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    lines = []
    for e in EXCLUDED:
        lines.append(
            " & ".join([tex_escape(e.name), tex_escape(e.rule), tex_escape(e.reason)])
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_counts(rows: list[Candidate]) -> str:
    """Class by tranche and host by tranche, off the same ordering."""
    head = r"""\begin{table}[H]\centering
\small
\caption[]{Candidates by class and by host, split at the two tranche lines.
\emph{Genotypes}, \emph{Instances} and \emph{Meas.} sum the per-row axes over all
tranches; a row with no count contributes nothing, so every total is a lower bound.}
\label{tab:bcounts}
\begin{tabular}{@{}l r r r r r r@{}}
\toprule
 & Tr. 1 & Tr. 2 & Reserve & Genotypes & Instances & Meas. \\
\midrule
"""

    def block(key: str, values: list[str]) -> list[str]:
        out = []
        for v in values:
            members = [(i, c) for i, c in enumerate(rows, 1) if getattr(c, key) == v]
            if not members:
                continue
            t1 = [c for i, c in members if i <= TRANCHE_1]
            t2 = [c for i, c in members if TRANCHE_1 < i <= TRANCHE_2]
            rest = [c for i, c in members if i > TRANCHE_2]
            g = sum(c.genotypes_n or 0 for _i, c in members)
            n = sum(c.instances_n or 0 for _i, c in members)
            m = sum(c.measurements or 0 for _i, c in members)
            out.append(
                f"{tex_escape(v)} & {len(t1)} & {len(t2)} & {len(rest)} & "
                f"{g:,} & {sci(n)} & {sci(m)} \\\\"
            )
        return out

    klasses = sorted({c.klass for c in rows})
    lines = block("klass", klasses)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{@{}l}{\emph{The same rows, by host}}\\")
    lines += block("organism", ["E. coli", "P. putida"])
    g_all = sum(c.genotypes_n or 0 for c in rows)
    n_all = sum(c.instances_n or 0 for c in rows)
    m_all = sum(c.measurements or 0 for c in rows)
    lines.append(r"\midrule")
    lines.append(
        f"Total & {min(TRANCHE_1, len(rows))} & "
        f"{max(min(TRANCHE_2, len(rows)) - TRANCHE_1, 0)} & "
        f"{max(len(rows) - TRANCHE_2, 0)} & "
        f"{g_all:,} & {sci(n_all)} & {sci(m_all)} \\\\"
    )
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def render_summary(rows: list[Candidate]) -> str:
    """Summary statistics over the fifty, split at the tranche-1 line."""
    fifty = rows[:TRANCHE_2]
    t1 = fifty[:TRANCHE_1]
    t2 = fifty[TRANCHE_1:]

    def count(pred: Any) -> str:
        a = sum(1 for c in t1 if pred(c))
        b = sum(1 for c in t2 if pred(c))
        g = sum(c.genotypes_n or 0 for c in fifty if pred(c))
        n = sum(c.instances_n or 0 for c in fifty if pred(c))
        m = sum(c.measurements or 0 for c in fifty if pred(c))
        return f"{a} & {b} & {g:,} & {sci(n)} & {sci(m)} \\\\"

    def block(title: str, items: list[tuple[str, Any]]) -> list[str]:
        out = [r"\midrule", r"\multicolumn{6}{@{}l}{\emph{" + title + r"}}\\"]
        for label, pred in items:
            if sum(1 for c in fifty if pred(c)) == 0:
                continue
            out.append(tex_escape(label) + " & " + count(pred))
        return out

    klasses = sorted({c.klass for c in fifty})
    bases = sorted({c.seq_basis for c in fifty})
    lines: list[str] = []
    lines += block(
        "By host",
        [(o, (lambda c, o=o: c.organism == o)) for o in ("E. coli", "P. putida")],
    )
    lines += block(
        "By tier", [(f"tier {t}", (lambda c, t=t: c.tier == t)) for t in (1, 2, 3, 4)]
    )
    lines += block("By class", [(k, (lambda c, k=k: c.klass == k)) for k in klasses])
    lines += block(
        "By sequence basis", [(b, (lambda c, b=b: c.seq_basis == b)) for b in bases]
    )
    lines += block(
        "Other attributes",
        [
            ("carries a time axis", lambda c: bool(c.time_axis)),
            ("mirrors a built yeast dataset", lambda c: c.analog is not None),
            ("needs a phenotype class the substrate lacks", lambda c: c.analog is None),
            (
                "has a join to a built dataset",
                lambda c: any(s.partner_status == "supported" for s in c.synergy),
            ),
            ("figures sourced this pass", lambda c: c.confidence == "sourced"),
            ("figures from recall, to confirm", lambda c: c.confidence == "recall"),
            ("data location confirmed live", lambda c: c.accession_confirmed),
            ("per-record values not released", lambda c: c.status == "blocked"),
            ("a corpus, not net new until split", lambda c: c.status == "aggregation"),
            ("instances a reported count", lambda c: c.instances_basis == "reported"),
            (
                "instances a product of the axes",
                lambda c: c.instances_basis == "product",
            ),
            ("instances an estimate", lambda c: c.instances_basis == "estimate"),
            ("dimensionality reported, not estimated", lambda c: c.dim_basis == "reported"),
        ],
    )
    lines.append(r"\midrule")
    lines.append("Total & " + count(lambda c: True))
    n_joins = sum(len(c.synergy) for c in fifty)
    n_built = sum(
        1 for c in fifty for s in c.synergy if s.partner_status == "supported"
    )
    head = (
        r"""\begin{table}[H]\centering
\small
\caption[]{Summary of the fifty, split at the tranche-1 line. \emph{Tr. 1} and
\emph{Tr. 2} count rows; \emph{Genotypes}, \emph{Instances} and \emph{Meas.} sum the row
axes over all fifty and are lower bounds, since a row with no count contributes nothing.
The fifty name """
        + str(n_joins)
        + r""" joins, """
        + str(n_built)
        + r""" of them to a dataset already built.}
\label{tab:bsummary}
\begin{tabular}{@{}l r r r r r@{}}
\toprule
 & Tr. 1 & Tr. 2 & Genotypes & Instances & Meas. \\
"""
    )
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def main() -> None:
    rows, quota_moves = ranked()
    if len(rows) < TARGET_COUNT:
        raise SystemExit(
            f"only {len(rows)} candidates; the target is {TARGET_COUNT} rows"
        )

    write(TEX_DIR / "final.tex", render_final(rows))
    write(TEX_DIR / "sources.tex", render_sources(rows[:TRANCHE_2]))
    write(TEX_DIR / "analogs.tex", render_analogs(rows[:TRANCHE_2]))
    write(TEX_DIR / "schema.tex", render_schema())
    write(TEX_DIR / "counts.tex", render_counts(rows))
    write(TEX_DIR / "summary.tex", render_summary(rows))
    write(TEX_DIR / "excluded.tex", render_excluded())

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps(
            {
                "yeast_built": YEAST_BUILT,
                "bacteria_built": BACTERIA_BUILT,
                "target_count": TARGET_COUNT,
                "tranche_1": TRANCHE_1,
                "tranche_2": TRANCHE_2,
                "quota_1": QUOTA_1,
                "n_candidates": len(rows),
                "quota_moves": quota_moves,
                "schema_needs": [s.model_dump() for s in SCHEMA_NEEDS],
                "candidates": [c.model_dump() for c in rows],
                "excluded": [e.model_dump() for e in EXCLUDED],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {JSON_OUT.relative_to(REPO)}")

    for name, old, new in quota_moves:
        print(f"quota promoted {name!r} from rank {old} to {new}")
    n_ec = sum(c.organism == "E. coli" for c in rows[:TRANCHE_1])
    n_pp = sum(c.organism == "P. putida" for c in rows[:TRANCHE_1])
    print(
        f"{len(rows)} candidates; tranche 1 = {TRANCHE_1} "
        f"({n_ec} E. coli, {n_pp} P. putida), tranche 2 ends at {TRANCHE_2}"
    )
    n_analog = sum(c.analog is not None for c in rows[:TRANCHE_2])
    n_sourced = sum(c.confidence == "sourced" for c in rows[:TRANCHE_2])
    print(
        f"of the fifty: {n_analog} mirror a built yeast dataset, "
        f"{n_sourced} carry figures sourced this pass"
    )


if __name__ == "__main__":
    main()
