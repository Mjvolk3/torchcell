# experiments/019-perturb-seq-costing/scripts/render_tex_tables.py
# [[experiments.019-perturb-seq-costing.scripts.render_tex_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/render_tex_tables
"""Emit the review's tables as LaTeX fragments for notes-tex/microbe-perturb-seq/.

Companion to ``render_tables.py`` (which emits the markdown/landscape-PDF form).
Both read the same constants, so the typeset document and the note cannot
disagree about a number.

Every emitted file carries a ``%% SOURCE:`` header naming this script, which is
what ``notes-tex/common/check_doc.py`` looks for when it audits provenance.

Run:  python experiments/019-perturb-seq-costing/scripts/render_tex_tables.py
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv

import cost_data as CD
import cost_model as CM
import design_equation as DE
import figure_sources as FS
import glossary as GL
import method_data as MD
import read_structure as RS
import uiuc_core_data as UC
from cost_per_cell_table import CITE, library_prep_table, loaded_table

load_dotenv()
OUT = osp.join(
    os.environ["WORKSPACE_DIR"], "notes-tex", "microbe-perturb-seq", "tables"
)
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "019-perturb-seq-costing", "results"
)

HEADER = (
    "%% GENERATED FILE -- do not hand-edit.\n"
    "%% SOURCE: experiments/019-perturb-seq-costing/scripts/render_tex_tables.py\n"
)


def esc(s: str) -> str:
    """Escape the LaTeX specials that occur in our table cells.

    ``^``, ``~``, ``<`` and ``>`` were added after a plain-ASCII prose field
    containing "n >= (z_a+z_b)^2" halted the build with "Missing $ inserted".
    Source fields are written as readable ASCII and are not expected to know
    about TeX, so the escaper has to cover everything TeX treats specially --
    catching only the obvious five leaves a build failure waiting for whoever
    writes the next quote containing a caret.
    """
    for a, b in (
        ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
        ("^", r"\textasciicircum{}"), ("~", r"\textasciitilde{}"),
        ("<", r"\textless{}"), (">", r"\textgreater{}"),
    ):
        s = s.replace(a, b)
    return s


def emit(name: str, body: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(osp.join(OUT, f"{name}.tex"), "w") as f:
        f.write(HEADER + body.rstrip() + "\n")
    print(f"  {name}.tex")


def table(spec: str, header: str, rows: list[str], caption: str,
          label: str, size: str = r"\footnotesize", note: str = "") -> str:
    """A booktabs table wrapped so it sits inline where it is \\input.

    Default size is \\footnotesize: at \\small these tables read visibly larger
    than the manuscript's, which is the house reference for how a table should
    sit on the page.

    ``note`` is folded into the CAPTION rather than set as a grey block under the
    table. Column glosses and provenance flags are things the reader needs
    *before* reading the rows, and a note underneath is read last if at all; it
    also detached visually from the table it belonged to. Pass plain prose --
    the wrapping and styling belong here, not at each call site.

    \\caption[]{...} with the EMPTY optional argument throughout: notes contain
    \\url and \\file, which are \\DeclareUrlCommand-based and therefore fragile,
    and a caption is a moving argument written to the .lot.
    """
    body = caption if not note else caption + " " + note
    return "\n".join(
        [
            r"\begin{table}[htbp]\centering",
            size,
            rf"\caption[]{{{body}}}\label{{{label}}}",
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *[r + r" \\" for r in rows],
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


# --- t0 (glossary) -----------------------------------------------------------
def t0() -> None:
    """The vocabulary table, Sec. 2.1.

    A ``longtable`` rather than the usual ``table``+``tabular`` float, because
    this one is a reference and must be allowed to break across pages: a 28-row
    float would either be dumped at the end of the document or overflow the page
    silently. longtable also repeats the header on each page, which is what a
    lookup table needs.

    The ``where`` field is emitted as a real \\cref, so every glossary entry is a
    live link into the section that treats it, and a wrong label is a build-time
    undefined reference rather than a dead pointer.
    """
    lines = [
        r"\begingroup\footnotesize",
        # p{} widths sum to 0.95\textwidth, leaving room for \tabcolsep. The
        # definition column takes the slack: term names are short and bounded,
        # definitions are not.
        # LTleft/LTright at 0pt make the table flush-left and full width rather
        # than centred.
        r"\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}",
        # The caption is set OUTSIDE the longtable with \captionof, not inside it
        # with \caption. tcdoc loads the `caption` package, and `caption` breaks
        # longtable's in-table caption: the caption box is sized independently of
        # the column widths and lands ~518 pt past the text block, whatever the
        # caption text says. Verified minimally -- dropping `caption` from the
        # preamble is the only thing that fixes the in-table form, and neither
        # \LTcapwidth, ltcaption, nor \captionsetup[longtable]{width=...} helps.
        # \captionof{table} keeps correct numbering, a working \label, and the
        # document's caption styling.
        #
        # \captionof[]{...} -- the EMPTY optional argument is still required:
        # \file is a \DeclareUrlCommand and therefore fragile, and a caption is a
        # moving argument (written to the .lot), so the unoptioned form halts with
        # "\url used in a moving argument". Same fix as \tcfig in tcdoc.sty.
        r"\begingroup\captionsetup{type=table}",
        r"\captionof{table}[]{Controlled vocabulary, alphabetical. The last column "
        r"links to the section that treats each term in depth, so an entry is a "
        r"pointer rather than a replacement for the argument. Generated from "
        r"\file{experiments/019-perturb-seq-costing/scripts/glossary.py}.}"
        r"\label{tab:glossary}",
        r"\endgroup",
        # Term column at 0.21: at 0.19 the longest bold terms ("Combinatorial
        # split-pool barcoding", "Source plate / working plate") filled it to the
        # last point and read as though they were running into the definition
        # column. The slack comes out of the definition column, which has the
        # most room to give.
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.21\textwidth}"
        r">{\raggedright\arraybackslash}p{0.62\textwidth}"
        r">{\raggedright\arraybackslash}p{0.11\textwidth}@{}}",
        r"\toprule",
        r"Term & Definition & See \\",
        r"\midrule",
        r"\endfirsthead",
        # Continuation header, so a reader landing mid-table still has columns.
        r"\multicolumn{3}{@{}l}{\footnotesize\itshape "
        r"Table~\ref{tab:glossary}, continued}\\",
        r"\toprule",
        r"Term & Definition & See \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    # One flat alphabetical run, no group headings. A heading row leaves the
    # Definition column empty, and the first entry under it then reads as though
    # its definition has slipped a line -- flagged twice in review. See the
    # ordering note in glossary.py for why teaching-order lost to lookup-order.
    #
    # \addlinespace between EVERY row, not just at group boundaries: at 36 rows
    # of wrapped text the entries ran together and the eye could not find where
    # one definition ended. 5 pt, not the 2.5 pt tried first -- the ask was "1.5
    # space", i.e. half a line ADDED, and at \footnotesize the baseline is about
    # 9.5 pt. 2.5 pt is a quarter line and reads as no gap at all between rows
    # that are themselves several lines of wrapped text.
    for t in GL.alphabetical():
        name = rf"\textbf{{{esc(t.term)}}}"
        if t.abbrev:
            # Abbreviation is NOT escaped: some are math ($k$).
            name += rf"\newline {{\color{{tcgray}}({t.abbrev})}}"
        lines.append(rf"{name} & {t.definition} & \cref{{{t.where}}} \\")
        lines.append(r"\addlinespace[5pt]")
    lines += [r"\end{longtable}", r"\endgroup"]
    emit("t0-glossary", "\n".join(lines))


# --- t1 ----------------------------------------------------------------------
def t1() -> None:
    rows = [
        (r"\org{S.\ cerevisiae}", "Total mRNA molecules per cell",
         r"\textbf{20{,}000--60{,}000}", r"Jariani 2020 \texttt{(md:31)}"),
        ("", "Working figure (Brettner)", r"$\sim$30{,}000",
         r"Brettner 2024 \texttt{(md:70)}"),
        ("", "Total mRNA molecules per cell", r"40{,}000--60{,}000",
         r"Jackson 2020~\citep{jacksonGeneRegulatoryNetwork2020}"),
        ("", "Implied by 3{,}000 genes $\\times$ 3.5 molecules", r"$\sim$10{,}500",
         r"Nadal-Ribelles 2024 \texttt{(md:107)}"),
        # The primary behind the row above, and it is a MEASUREMENT of the same
        # quantity the review rounds to 3.5 -- worth carrying separately because
        # it is also the second factor in Nadal-Ribelles' derived depth in
        # Table 4, and a reader should be able to find it without the review.
        ("", "Mean UMIs per detected gene, per cell", "3.46",
         r"Nadal-Ribelles 2019~\citep{nadal-ribellesSensitiveHighthroughputSinglecell2019}"),
        ("", "Capture rate, 10x 3$'$ v2", r"3--5\%", r"Jackson 2020~\citep{jacksonGeneRegulatoryNetwork2020}"),
        ("", "Total RNA per cell ($\\sim$85\\% rRNA)", "0.7--1 pg",
         r"Nadal-Ribelles 2024 \texttt{(md:49)}"),
        ("", "Protein-coding genes", r"$\sim$6{,}000", "SGD / R64"),
        # External rows are marked with a superscript letter and resolved to a
        # URL in a note under the table. The point of the flag is that someone
        # will need to go and verify the number, so the link has to be there --
        # but inline URLs would blow the column width out, hence the note.
        (r"\org{E.\ coli}", r"Total mRNA per cell, \textbf{rich (LB)}",
         rf"\textbf{{$\sim${MD.ECOLI_MRNA_PER_CELL_RICH:,}}}".replace(",", "{,}"),
         r"Bartholom\"aus 2016$^{a}$"),
        ("", "Total mRNA per cell, minimal medium",
         rf"$\sim${MD.ECOLI_MRNA_PER_CELL_MINIMAL:,}".replace(",", "{,}"),
         r"Bartholom\"aus 2016$^{a}$"),
        # No longer flagged: Taniguchi is now in the mirror and the collection,
        # so it is an ordinary sourced row rather than an outside reference.
        ("", "Total mRNA per cell, medium growth rate", r"$\sim$3{,}000",
         r"Taniguchi 2010~\citep{taniguchiQuantifyingColiProteome2010}"),
        ("", "rRNA share of all transcripts", r"\textbf{$>$95\%}",
         r"Brandner 2025 \texttt{(md:71)}"),
        ("", "Protein-coding genes", r"$\sim$4{,}400", "K-12 MG1655"),
        ("Mammalian", "Total mRNA molecules per cell", "50{,}000--300{,}000",
         r"Jariani 2020 \texttt{(md:31)}"),
    ]
    note = (
        r"$^{a}$ BNID 112795, "
        r"\url{https://bionumbers.hms.harvard.edu/bionumber.aspx?id=112795} -- "
        r"the one row here that is \emph{not} in the torchcell-library mirror. "
        r"No mirrored bacterial paper states a total mRNA content, only what its "
        r"assay detected, which is a different and $\sim$30$\times$ smaller "
        r"quantity. Every other row is mirror-backed and separately citable. "
        r"\textbf{The yeast rows disagree and are left that way.} Jariani et "
        r"al.'s 20{,}000 floor, Jackson et al.'s 40{,}000--60{,}000 and the "
        r"$\sim$10{,}500 implied by the review's own 3{,}000 genes $\times$ 3.5 "
        r"molecules span six-fold. The document works at "
        r"$\sim$30{,}000; because $p_j$ enters \cref{eq:design} only through the "
        r"shot-noise term, and \cref{sec:design-equation} finds both candidate "
        r"platforms firmly shot-noise limited, this choice does move the "
        r"cells-per-perturbation numbers roughly in proportion."
    )
    emit(
        "t1-transcript-content",
        table(
            "llrl",
            r"Organism & Quantity & Value & Source",
            [" & ".join(r) for r in rows],
            "mRNA content per cell. This is the denominator for every "
            "capture-efficiency claim in the document.",
            "tab:transcript-content",
            note=note,
        ),
    )


# --- t2 ----------------------------------------------------------------------
# Display-only shortenings so every cell fits on one line. The full strings stay
# in method_data.py; this is presentation, not data.
SHORT_PLATFORM = {
    "Fluidigm C1 / SMART-Seq v4": "Fluidigm C1",
    "FACS + STRT-seq (5' end)": "FACS + STRT-seq",
    "10x Chromium 3' v2": "10x 3$'$ v2",
    "10x Chromium 3' v2 + in-droplet zymolyase": "10x 3$'$ v2 (in-drop zym.)",
    "yeastDrop-Seq (table-top)": "yeastDrop-Seq",
    "mDrop-seq": "mDrop-seq",
    "10x Chromium 3' v2 + v3": "10x 3$'$ v2\\,+\\,v3",
    "10x Chromium ATAC + plate preindexing": "10x ATAC + preindex.",
    "yeast-optimized SPLiT-seq": "SPLiT-seq (yeast)",
    "microSPLiT": "microSPLiT",
    "mapSPLiT (microSPLiT + GBC)": "mapSPLiT",
    "ProBac-seq (probe-based, 10x)": "ProBac-seq",
    "BacDrop (pre-indexed droplet)": "BacDrop",
}
SHORT_STUDY = {
    "Brettner 2024 (SPLiT-seq)": "Brettner 2024",
    "Dohn 2021 (mDrop-seq)": "Dohn 2021",
    "microSPLiT (B. subtilis)": "Kuchina 2021",
    "mapSPLiT (E. coli, CRISPRi/a)": "Brandner 2025",
    "ProBac-seq (B. subtilis)": "McNulty 2023",
    "BacDrop (K. pneumoniae)": "Ma 2023",
}


# The separator that opens the non-microbial block of the landscape table. The
# `table()` helper appends the row terminator, so this is written as a rule
# followed by a spanning italic label and nothing else -- a break a reader
# cannot skim past, which a blank line would be.
_COMPARATOR_RULE = (
    r"\midrule \multicolumn{8}{@{}l}{\itshape "
    r"For comparison, not a microbial study:}"
)


def t2() -> None:
    # Microbial rows first, then a rule, then the comparators. Sorting them into
    # one block would let a mammalian cell count be read as microbial evidence,
    # which is the whole reason `Method.microbial` exists.
    rows = []
    ordered = [m for m in MD.METHODS if m.microbial] + [
        m for m in MD.METHODS if not m.microbial
    ]
    for m in ordered:
        if not m.microbial and rows and rows[-1] != _COMPARATOR_RULE:
            rows.append(_COMPARATOR_RULE)
        umi = "--"
        if m.mrna_umis_per_cell is not None:
            umi = f"{m.mrna_umis_per_cell:,.0f}"
            if m.mrna_umis_low and m.mrna_umis_high:
                umi = f"{m.mrna_umis_low:,.0f}--{m.mrna_umis_high:,.0f}"
        gene = "--"
        if m.genes_per_cell is not None:
            gene = f"{m.genes_per_cell:,.0f}"
            if m.genes_low and m.genes_high:
                gene = f"{m.genes_low:,.0f}--{m.genes_high:,.0f}"
        rows.append(
            " & ".join(
                [
                    esc(m.study)
                    + (r"$^{\dagger}$" if m.secondhand else "")
                    + (r"$^{\ddagger}$"
                       if m.spread_kind == "across_experiments" else ""),
                    SHORT_PLATFORM.get(m.platform, esc(m.platform)),
                    m.isolation.replace("_", "-"),
                    f"{m.cells_profiled:,}".replace(",", "{,}"),
                    umi.replace(",", "{,}"),
                    gene.replace(",", "{,}"),
                    "yes" if m.has_perturbation_readout else "--",
                    # Basis column: a stated value and a value we derived must
                    # be distinguishable at a glance (same rule as t3/t4).
                    {"reported": "rep.", "midpoint": "midpt.",
                     "median_of": "med-of", "weighted": "wtd.",
                     "derived": "deriv."}[m.value_basis],
                ]
            )
        )
    note = (
        r"$\dagger$ values taken from the Nadal-Ribelles review's summary table "
        r"because the primary paper is not in our mirror, so they are second-hand "
        r"and should be verified before being cited. "
        r"$\ddagger$ the quoted UMI range spans separate \emph{experiments} of "
        r"deliberately differing depth rather than conditions within one run. "
        r"\textbf{No range in the UMI column is a dispersion statistic over "
        r"cells}; all are min--max ranges of per-condition or per-experiment "
        r"summaries. \textbf{A dash in the UMI column is deliberate}: Gasch et "
        r"al.\ used a protocol that carries no UMI and Ma et al.\ report only "
        r"genes, so for those two no molecule count exists. Datlinger et al.\ is "
        r"a different case worth distinguishing: a per-cell depth was measured, "
        r"but the article defers its performance metrics to a supplementary "
        r"table held behind the journal's paywall and absent from the mirror, so "
        r"the value exists and is not in hand. No value is invented in either "
        r"case, and where a measured gene count exists it appears under "
        r"Genes/cell instead. "
        r"\textbf{The row below the rule is not microbial.} scifi-RNA-seq was "
        r"run on human and mouse material only; it is tabulated because "
        r"\cref{sec:scifi} and the budget both turn on it, and it is separated "
        r"because a mammalian cell count is not evidence about microbes. Its "
        r"151{,}788 is a recovery through quality control from one Chromium "
        r"channel, not the 1.53-million-nuclei loading demonstration, which is a "
        r"feasibility ceiling. It is absent from \cref{fig:landscape} for want of "
        r"a depth coordinate; the throughput move it demonstrates is drawn there "
        r"as an arrow instead. "
        r"\textbf{Basis} states how the single plotted value was obtained: "
        r"\emph{reported} = stated by the source; \emph{midpoint} = midpoint of a "
        r"reported range, a declared convention where no per-condition value "
        r"exists; \emph{median-of} = median of the reported per-condition values; "
        r"\emph{weighted} = cell-weighted mean of reported per-dataset values; "
        r"\emph{derived} = computed from two quantities the source states "
        r"separately, never printing the one needed here. "
        r"``Pert.'' = has a perturbation readout."
    )
    emit(
        "t2-method-landscape",
        table(
            # All-l columns: every cell sits on one line. Wrapped platform names
            # made the rows ragged and hard to scan; the fix is shorter labels
            # (see SHORT_PLATFORM) plus \scriptsize, not p{} boxes.
            r"lllrrrcl",
            r"Study & Platform & Isolation & Cells & mRNA UMIs/cell & Genes/cell & Pert. & Basis",
            rows,
            "Published single-cell RNA-seq runs in yeast and other microbes, "
            "plus one non-microbial comparator below the rule.",
            "tab:landscape",
            size=r"\scriptsize",
            note=note,
        ),
    )


# --- t3, t4 ------------------------------------------------------------------
def t3() -> None:
    df = library_prep_table()
    rows = []
    for _, r in df.iterrows():
        u = r["usd_per_cell"]
        s = f"{u:,.2f}" if u >= 1 else (f"{u:.4f}" if u >= 0.01 else f"{u:.5f}")
        rows.append(
            " & ".join(
                [
                    esc(r["method"]),
                    rf"\org{{{esc(r['organism'])}}}",
                    r["isolation"],
                    rf"\${s}",
                    r["sourced_or_derived"],
                    esc(r["citation"].replace("*", "")),
                ]
            )
        )
    emit(
        "t3-cost-per-cell",
        table(
            "lllrll",
            r"Method & Organism & Isolation & \$ / cell & Basis & Source",
            rows,
            r"Cost per cell, \emph{library preparation only}. ``Quoted'' means the "
            r"source states a per-cell rate; ``derived'' means a total divided by a "
            r"cell count the same source supplies. Every row excludes sequencing.",
            "tab:cost-per-cell",
            size=r"\footnotesize",
        ),
    )


def t4() -> None:
    df = loaded_table()
    # Short platform labels -- the full names push this past the text block.
    short = {
        "SPLiT-seq (Brettner, as published)": "SPLiT-seq, as published",
        "SPLiT-seq + rRNA depletion": "SPLiT-seq + rRNA depl.",
        "10x Chromium X (GEM-X 3')": "10x Chromium X",
        "10x + scifi preindexing (projected)": "10x + scifi preindex.",
    }
    rows = [
        " & ".join(
            [
                esc(short.get(r["platform"], r["platform"]))
                + (r"$^{\ast}$" if r["projected"] else ""),
                f"\\${r['reagent_usd_per_cell_processed']:.4f}",
                f"\\textbf{{\\${r['loaded_usd_per_usable_cell']:.4f}}}",
                f"{r['hidden_multiplier']}$\\times$",
                f"{r['sequencing_share_pct']}\\%",
            ]
        )
        for _, r in df.iterrows()
    ]
    emit(
        "t4-cost-per-cell-loaded",
        table(
            "lrrrr",
            r"Platform & Reagents \$/cell proc. & Loaded \$/usable cell & Hidden mult. & Seq.\ share",
            rows,
            "Cost per cell end to end: 6{,}000 genes at 250 cells/gene, per cell "
            "that survives QC \\emph{and} carries an identified guide. "
            "$\\ast$ projected rather than measured: the scifi row changes one "
            "parameter of the 10x row, cells recovered per channel, and is "
            "unmeasured in yeast (\\cref{sec:scifi}).",
            "tab:cost-loaded",
        ),
    )


# --- t5 ----------------------------------------------------------------------
def t5() -> None:
    rows = []
    for i in CD.BRETTNER_ITEMS:
        scale = i.scaling.replace("_", " ")
        if i.runs_covered:
            scale = rf"\textbf{{one-time, {i.runs_covered} runs}}"
        elif i.scaling == "per_run":
            scale = r"\textbf{per run}"
        elif i.scaling == "per_sublibrary":
            scale = "per sublibrary"
        rows.append(f"{esc(i.name)} & \\${i.usd:,.2f} & {scale}".replace(",", "{,}"))
    emit(
        "t5-brettner-items",
        table(
            "lrl",
            r"Item & Cost & Scaling",
            rows,
            r"Brettner 2024 SPLiT-seq line items \texttt{(paper.md:183)}. The "
            r"start-up cost is one purchase order -- three plates of custom IDT "
            r"oligos, good for 215 runs.",
            "tab:brettner-items",
        ),
    )


# --- t6 ----------------------------------------------------------------------
def t6() -> None:
    rows = []
    cells = [45_000, 480_000, 3_000_000, 12_000_000]

    def rate_cells(B: int) -> str:
        return " & ".join(f"{100 * MD.collision_rate(c, B):.2f}\\%" for c in cells)

    for rounds in (3, 4):
        for sub in (1, 24, 96):
            B = MD.barcode_space(rounds=rounds, sublibraries=sub)
            bold = rounds == 4 and sub == 24
            fmt = (lambda s: rf"\textbf{{{s}}}") if bold else (lambda s: s)
            rows.append(
                " & ".join(
                    [fmt(f"split-pool, {rounds} rounds"), fmt(str(sub)),
                     fmt(f"{B:,}".replace(",", "{,}")), rate_cells(B)]
                )
            )
    # The preindexed-droplet scheme, for comparison on the same axis. Its space
    # is a PRODUCT of a plate dimension and a droplet dimension, so it does not
    # fit the 96^R x S form above and gets its own row rather than being forced
    # into those columns. Sublibraries are "--" because there is no post-ligation
    # split: the second dimension is the droplet itself.
    B_scifi = MD.SCIFI_ROUND1_WELLS_USED * MD.SCIFI_ROUND2_BARCODES
    rows.append(
        " & ".join(
            [r"scifi, $384\times$droplet", "--",
             f"{B_scifi:,}".replace(",", "{,}"), rate_cells(B_scifi)]
        )
    )
    emit(
        "t6-barcode",
        table(
            "lrrrrrr",
            r"Scheme & Sublib. & Barcode space & @45k & @480k & @3M & @12M cells",
            rows,
            r"Barcode collision rate, $100(1-((B-1)/B)^{C-1})$. For split-pool "
            r"$B=96^{R}\times S$; the sublibrary index $S$ is a real barcode "
            r"dimension, because cells are split into sublibraries \emph{after} "
            r"the last in-cell ligation. The final row is the preindexed-droplet "
            r"scheme of \cref{sec:scifi}, where $B$ is the 384 round1 wells times "
            r"the 737{,}280 round2 barcodes the Chromium ATAC reagents supply. "
            r"\textbf{That row is an upper bound on its own performance.} The "
            r"formula assumes cells are spread uniformly over the space, which is "
            r"true of plate wells and not of droplets; Datlinger et al.\ replace "
            r"the uniform assumption with a zero-inflated Poisson fit for exactly "
            r"this reason, so read the row as ``the space is not the binding "
            r"constraint'' rather than as a predicted collision rate.",
            "tab:barcode",
        ),
    )


# --- t7 ----------------------------------------------------------------------
def t7() -> None:
    target = CM.PSEUDOBULK_TIERS["standard (Brettner's own 500-cell heuristic)"]
    depths = [
        ("SPLiT-seq as published", 410),
        ("SPLiT-seq + rRNA depletion", 861),
        ("10x v2 (Jariani)", 894),
        ("10x v3 (Jackson)", 2000),
    ]
    rows = []
    for name, d in depths:
        n = CM.cells_per_perturbation(d, target)
        binding = "biological floor" if n == CM.CELLS_FLOOR else "sequencing depth"
        rows.append(
            f"{esc(name)} & {d:,} & \\textbf{{{n}}} & {binding} & {n * 6000:,}".replace(
                ",", "{,}"
            )
        )
    emit(
        "t7-cells-per-perturbation",
        table(
            "lrrlr",
            r"Method & mRNA UMIs/cell & Cells/pert. & Binding constraint & Cells for 6{,}000 genes",
            rows,
            r"Cells per perturbation at the standard pseudobulk tier "
            r"($2\times10^{5}$ mRNA UMIs). The 100-cell biological floor and the "
            r"depth requirement are independent; the larger binds.",
            "tab:cells-per-pert",
        ),
    )


# --- t8 ----------------------------------------------------------------------
def t8() -> None:
    df = pd.read_csv(osp.join(RESULTS, "genome_scale_budgets.csv"))
    df["reagents"] = df.protocol_usd + df.sublibrary_usd
    short = {
        "SPLiT-seq (Brettner, as published)": "SPLiT-seq",
        "SPLiT-seq + rRNA depletion": "SPLiT-seq + depl.",
        "10x Chromium X (GEM-X 3')": "10x Chromium X",
        "10x + scifi preindexing (projected)": "10x + scifi preindex.",
    }
    rows = []
    for _, r in df.iterrows():
        rows.append(
            " & ".join(
                [
                    str(r["cells_per_gene"]),
                    short[r["platform"]] + (r"$^{\ast}$" if r["projected"] else ""),
                    f"{r['usable_cells']:,}".replace(",", "{,}"),
                    f"{r['sequenced_cells']:,}".replace(",", "{,}"),
                    str(r["n_batches"]),
                    f"{r['read_pairs_billions']:.1f}",
                    f"\\${r['reagents']:,.0f}".replace(",", "{,}"),
                    f"\\${r['sequencing_usd']:,.0f}".replace(",", "{,}"),
                    f"\\textbf{{\\${r['recurring_usd']:,.0f}}}".replace(",", "{,}"),
                ]
            )
        )
    note = (
        r"\textbf{Cells per target gene} = cells carrying a guide against that gene, "
        r"pooled over all six of its guides -- not cells per guide and not cells per "
        r"plasmid. \textbf{Analysed} = cells surviving QC \emph{and} assigned to a "
        r"guide; \textbf{sequenced} = cells whose reads were paid for, which is "
        r"larger because $\sim$75\% of split-pool cells are discarded after "
        r"sequencing. \textbf{Protocol runs} is a count of wet-lab repetitions of "
        r"the barcoding workflow (split-pool) or of 10x channels, \emph{not} a cost. "
        r"All sequencing is priced on NovaSeq X 25B PE150 at the 8+ lane rate, the "
        r"cheapest per-read option the core offers. Excludes labor, oligo pool "
        r"synthesis, strain construction, and the $\sim$\$10{,}000 one-time "
        r"split-pool start-up. "
        r"$\ast$ \textbf{The scifi row is a projection, not a measurement.} It "
        r"holds every 10x parameter fixed and changes one: cells recovered per "
        r"channel becomes Datlinger et al.'s measured 151{,}788 in place of the "
        r"UIUC baseline of 20{,}000. So it answers what the droplet budget looks "
        r"like if a channel goes 7.6 times further and nothing else changes. "
        r"Per-cell depth in yeast is unmeasured for this chemistry and is held "
        r"optimistically at the 10x value; no scifi run in any microbe has been "
        r"published (\cref{sec:scifi})."
    )
    emit(
        "t8-budgets",
        table(
            "rlrrrrrrr",
            r"Cells per & Platform & Cells & Cells & Protocol & Read pairs & "
            r"Reagents & Sequencing & Total \\"
            "\n"
            r"target gene & & analyzed & sequenced & runs & (billions) & & &",
            rows,
            r"Genome-scale CRISPRi Perturb-seq budget: 6{,}000 target genes, six "
            r"guides per gene, one environment, UIUC rates effective 2026-08-01.",
            "tab:budgets",
            size=r"\footnotesize",
            note=note,
        ),
    )


# --- t9 ----------------------------------------------------------------------
def t9() -> None:
    opts = sorted(
        UC.NOVASEQ_X + UC.MISEQ_I100, key=lambda o: o.usd_per_million_read_pairs
    )
    rows = [
        " & ".join(
            [
                esc(o.label),
                o.instrument,
                f"\\${o.usd_per_lane:,.0f}".replace(",", "{,}"),
                f"{o.read_pairs_per_lane / 1e6:,.0f}M".replace(",", "{,}"),
                f"\\${o.usd_per_million_read_pairs:.2f}",
            ]
        )
        for o in opts
    ]
    emit(
        "t9-sequencing",
        table(
            "llrrr",
            r"Option & Instrument & \$ / lane & Read pairs / lane & \$ per M pairs",
            rows,
            r"UIUC Carver Biotechnology Center sequencing rates, effective "
            r"2026-08-01, on-campus. External users pay $+30.8\%$. The core has no "
            r"NextSeq and no NovaSeq 6000.",
            "tab:sequencing",
        ),
    )


# --- t10 ---------------------------------------------------------------------
def t10() -> None:
    rows = []
    for k in (1, 2, 3, 5, 8, 10):
        n = CM.cells_for_main_effects_kplex(250, 6000, k)
        pairs_6000 = CM.cells_for_all_pairs(6000, k)
        pairs_200 = CM.cells_for_all_pairs(200, k)
        fmt = lambda v: ("--" if math.isinf(v) else f"{v:,.0f}".replace(",", "{,}"))
        rows.append(
            " & ".join(
                [
                    str(k),
                    f"{n:,}".replace(",", "{,}"),
                    fmt(pairs_6000),
                    fmt(pairs_200),
                    ("--" if k < 2 else f"{CM.max_targets_for_pairs(300_000, k):.0f}"),
                ]
            )
        )
    note = (
        r"\textbf{Plex $k$} = guide RNAs delivered per cell, so $k$ target genes are "
        r"knocked down in that cell. \textbf{Main effects} = the average effect of "
        r"one gene, pooling every cell that carries a guide against it whatever else "
        r"it carries. \textbf{Pairs} are pairs of \emph{target genes}, not plasmids: "
        r"a cell informs a pair only if it happens to carry guides against both. "
        r"Column 2 targets 250-cells-per-gene precision over 6{,}000 genes; columns "
        r"3--4 apply Eq.~\ref{eq:pairs} at Yao's 400 cells per pair; column 5 "
        r"inverts it -- how many target genes can have \emph{all} their pairs "
        r"powered inside one 300{,}000-cell run."
    )
    emit(
        "t10-multiplex",
        table(
            "rrrrr",
            r"Guides per & Cells for & Cells for all pairs & Cells for all pairs & "
            r"Genes fully paired \\"
            "\n"
            r"cell ($k$) & main effects & among 6{,}000 genes & among 200 genes & "
            r"in 300k cells",
            rows,
            "What multiplexing buys, and what it does not.",
            "tab:multiplex",
            note=note,
        ),
    )


# --- t11 ---------------------------------------------------------------------
# Display-only shortenings; the full descriptions live in read_structure.py.
SHORT_READ = {
    "cDNA": "cDNA",
    # Order is load-bearing, so the short form keeps it: UMI first (it rides in
    # on the round-3 oligo), BC1 last (it is on the round-1 RT primer, nearest
    # the cDNA). See read_structure.py for the STARsolo offsets that fix this.
    "10 nt UMI + BC3 + BC2 + BC1 (with linkers)": "UMI+BC3+BC2+BC1",
    "16 nt cell barcode + 12 nt UMI": "barcode + UMI",
    "12 nt cell barcode + 8 nt UMI": "barcode + UMI",
}
SHORT_IDX = {
    "6 nt single, or dual 8 nt -- the sublibrary index (barcode round 4)":
        "6 or dual 8 nt (= BC round 4)",
    "dual 10 nt (i7 + i5), unique dual indices": "dual 10 nt",
    "single 8 nt": "single 8 nt",
}

def t11() -> None:
    rows = []
    for c in RS.READ_CONFIGS:
        short = c.method.split("(")[0].strip().split("/")[0].strip()
        rows.append(
            " & ".join(
                [
                    rf"\textbf{{{esc(short)}}}",
                    f"{c.read1_nt} nt {esc(SHORT_READ.get(c.read1_content, c.read1_content))}",
                    f"{c.read2_nt} nt {esc(SHORT_READ.get(c.read2_content, c.read2_content))}",
                    esc(SHORT_IDX.get(c.index_nt, c.index_nt)),
                    str(c.min_kit_cycles),
                ]
            )
        )
    emit(
        "t11-read-structure",
        table(
            # One line per cell; the content is short enough once the parenthetical
            # glosses are stripped in the row builder above.
            r"lllll",
            r"Method & Read 1 & Read 2 & Index & Cycles",
            rows,
            r"Read configurations. Both families require paired-end runs but put "
            r"the cell barcode on \emph{opposite} reads. For SPLiT-seq the index is "
            r"barcode round 4, not merely a sample tag, so unique dual indices are "
            r"mandatory.",
            "tab:read-structure",
            size=r"\footnotesize",
        ),
    )


def t12() -> None:
    """Parameters of the design equation, and which are actually known.

    The point of this table is the STATUS column. Three of the seven inputs are
    assumed, and the section's whole argument is that those three are what stand
    between a structure and a predictor -- so the table has to make "measured"
    and "assumed" impossible to confuse.
    """
    # Delta is no longer assumed, and this table said it was long after
    # effect_size_analysis.py had measured it -- the figure of Sec. 4.7 was
    # already drawn at the measured value while this table still printed the
    # nominal two-fold beside the word "assumed". Both now read from the same
    # constant, so the two cannot drift apart again.
    delta = DE.DELTA_MEASURED
    A = DE.power_coefficient(delta)
    rows = [
        (r"$A(\Delta)$", "power coefficient, Eq.~(\\ref{eq:design})",
         f"{A:.1f}", "derived",
         r"at the measured $\Delta$; scales as $\Delta^{-2}$"),
        (r"$\Delta$", "log$_2$ fold change to resolve", f"{delta:.2f}",
         "measured", r"median responder at a 1.25$\times$ cut, "
                     r"\cref{tab:effect-size}"),
        (r"$p_j$", "transcriptome share of target gene",
         f"{DE.P_TYPICAL:.1e}".replace("e-0", r"$\times10^{-") + r"}$",
         "derived", r"3.5 molecules / 30{,}000 mRNA, \cref{tab:transcript-content}"),
        (r"$\varphi_j$", "biological overdispersion between cells", "0.5--10",
         r"\textbf{assumed}", "swept; sets the floor. Not yet measured"),
        # Source cells are set in an `l` column, so they must fit on one line.
        # The preindexed caveat lives in the caption note instead; spelling it
        # out here pushed the table 92 pt past the text block.
        (r"$d$", "sequencing depth, mRNA UMIs per cell", "410--2{,}000",
         "measured", r"per platform, \cref{tab:landscape}"),
        (r"$\rho_2$", "variance inflation, second order", "4",
         "quoted", "Yao et al.; the 400-cells-per-pair constant"),
        (r"$q$", "per-guide detection probability", "--",
         r"\textbf{assumed}", r"needs the construct of \cref{sec:guide-capture}"),
    ]
    emit(
        "t12-design-parameters",
        table(
            "llrll",
            r"Symbol & Quantity & Value & Status & Source",
            [" & ".join(r) for r in rows],
            "Inputs to the design equation. Two of the seven are assumed rather "
            "than measured, and those two are what separate a structure from a "
            "predictor.",
            "tab:design-params",
            note=r"\textbf{Status}: \emph{measured} = from data we hold; "
                 r"\emph{quoted} = stated by a source; \emph{derived} = computed "
                 r"from measured values; \textbf{assumed} = a placeholder, and "
                 r"every number downstream of it inherits that status. "
                 r"One qualification on $d$: the 410--2{,}000 range is measured "
                 r"per platform, but the preindexed droplet of \cref{sec:scifi} "
                 r"has no published per-cell depth in any microbe and is held at "
                 r"the 10x value throughout, so for that platform alone $d$ is "
                 r"\textbf{assumed}.",
        ),
    )


def t14() -> None:
    """Measured response to a gene deletion, from the expression compendia.

    Replaces the nominal two-fold the rest of the document assumed. The column
    that matters is the last one: the median response among responding genes is
    ~1.34-fold in all three datasets, and A goes as Delta^-2.
    """
    df = pd.read_csv(osp.join(RESULTS, "effect_size_summary.csv"))
    label = {
        "kemmeren2014_single": r"Kemmeren 2014, single deletion",
        "sameith2015_single": r"Sameith 2015, single deletion",
        "sameith2015_double": r"Sameith 2015, \textbf{double} deletion",
    }
    rows = []
    for _, r in df.sort_values(["n_perturbations", "dataset"]).iterrows():
        med = r["median_abs_log2fc_responders"]
        rows.append(" & ".join([
            label[r["dataset"]],
            f"{int(r['n_strains']):,}".replace(",", "{,}"),
            f"{r['median_n_resp_1.25x']:.0f}",
            f"{r['median_n_resp_1.5x']:.0f}",
            f"{r['median_n_resp_2.0x']:.0f}",
            f"\\textbf{{{med:.2f}}}",
            f"{2**med:.2f}$\\times$",
        ]))
    emit(
        "t14-effect-size",
        table(
            "lrrrrrr",
            r"Compendium & Strains & \multicolumn{3}{c}{Median responding genes at} & "
            r"\multicolumn{2}{c}{Median $|\Delta|$, responders} \\"
            "\n"
            r" & & 1.25$\times$ & 1.5$\times$ & 2$\times$ & log$_2$ & fold",
            rows,
            "Measured response of the transcriptome to a gene deletion.",
            "tab:effect-size",
            note=r"\textbf{``Responding'' means $|\log_2$ FC$| > \log_2 1.25$}, "
                 r"that is, a gene whose expression moved by more than 1.25-fold "
                 r"in either direction relative to wild type, on the array "
                 r"platform of the source compendium. There is no significance "
                 r"test behind it and none is available per gene per strain: the "
                 r"criterion is a magnitude cut on the reported log ratio, applied "
                 r"identically to all three datasets. "
                 r"A single deletion moves a few hundred genes by that criterion "
                 r"but only $\sim$10--15 by 2$\times$, so designing at two-fold "
                 r"targets a few percent of the response. "
                 r"\textbf{The 1.34$\times$ in the last column is conditioned on "
                 r"that cut and is not independent of it.} The $|\Delta|$ "
                 r"distribution falls off monotonically (\cref{fig:effect-size}a), so "
                 r"the median of whatever upper tail is selected sits just above "
                 r"the cut that selected it: the same statistic is 1.16$\times$ at "
                 r"a 1.1$\times$ cut, 1.36$\times$ at 1.25$\times$, 1.73$\times$ "
                 r"at 1.5$\times$ and 2.57$\times$ at 2$\times$ "
                 r"(\cref{fig:effect-size}c, right axis). What the column establishes "
                 r"is therefore not a natural effect size but a consequence of "
                 r"choosing 1.25$\times$, and the choice is what to argue with. "
                 r"It matters because the power coefficient $A$ scales as "
                 r"$|\Delta|^{-2}$, so at 1.34$\times$ every cell requirement is "
                 r"$\sim$5.6 times the two-fold figure. "
                 r"Doubling the perturbation count roughly doubles the responder "
                 r"count at 1.25$\times$ and trebles it at 2$\times$, without "
                 r"shifting the median response -- more perturbations make a cell "
                 r"noisier, not its individual effects larger. Generated from "
                 r"\file{experiments/019-perturb-seq-costing/scripts/effect_size_analysis.py}.",
        ),
    )


# t13 (external references) is RETIRED. All eight works it listed are now in the
# microbe-perturb-seq Zotero collection and carry numbered \citep citations in
# Sec. 4.7, which is exactly the condition the table existed to make visible.
# Do not restore it: a DOI table for works we hold would re-hide the citations
# it was invented to expose.



# --- t16 (compression parameters) --------------------------------------------
def t16() -> None:
    """n, q and r for compressed Perturb-seq, and how well each is known.

    The companion to t12. That table asks what the design equation needs and
    which of its inputs are assumed; this one asks the same of the compressed
    screen, and the answer is different in an instructive way -- two of the three
    are measured here for the first time, and the one that is not is a choice
    rather than a gap.
    """
    with open(osp.join(RESULTS, "compression_summary.json")) as fh:
        c = json.load(fh)
    q_lo, q_hi = c["q_iqr"]
    rows = [
        (r"$n$", "targets in the library", "200--6{,}000",
         r"\textbf{a choice}",
         r"the panel of \cref{sec:multiplex}, or the genome"),
        (r"$q$", "genes moved per perturbation",
         f"{c['q_median_genes_moved_per_perturbation']:.0f}",
         "measured",
         rf"IQR {q_lo:.0f}--{q_hi:.0f} over {c['n_strains']:,} deletions"
         .replace(",", "{,}")),
        (r"$r$", "components in the effect matrix",
         f"{c['rank_90pct']}", "measured",
         rf"{c['rank_50pct']} at 50\% of variance, "
         rf"{c['rank_95pct']} at 95\%; no knee"),
        (r"$(q{+}r)\log n$", "composite samples, unit constant",
         f"{(c['q_median_genes_moved_per_perturbation'] + c['rank_90pct']) * math.log(6000):,.0f}"
         .replace(",", "{,}"),
         "derived", r"at $n=6{,}000$; $0.65n$, so compression barely pays"),
        ("slope", "observed / additive, double deletions",
         f"{c['additivity_median_slope']:.2f}", "measured",
         rf"median over {c['additivity_n_testable_doubles']} Sameith doubles"),
    ]
    emit(
        "t16-compression-parameters",
        table(
            "llrll",
            r"Symbol & Quantity & Value & Status & Basis",
            [" & ".join(r) for r in rows],
            "Parameters of the compressed-screen bound, measured from the "
            "expression compendia.",
            "tab:compression-params",
            note=r"Yao et al.\ give the sample requirement as "
                 r"$O((q+r)\log n)$ without a constant, so the fourth row is a "
                 r"\emph{shape} evaluated at unit constant and not a budget; what "
                 r"survives the unknown constant is that the requirement is "
                 r"nearly flat in $n$ while the conventional one is linear. "
                 r"\textbf{The last row is the assumption, not a parameter.} "
                 r"Guide-pooling requires effects to combine additively and "
                 r"argues that interactions cancel; in yeast they do not cancel, "
                 r"they buffer, and a double produces about 60\% of the sum of "
                 r"its singles. Sameith et al.'s pairs were chosen because they "
                 r"were expected to interact, so 0.62 is a lower bound on "
                 r"additivity rather than an estimate of it "
                 r"(\cref{sec:compression}). Generated from "
                 r"\file{experiments/019-perturb-seq-costing/scripts/compression_analysis.py}.",
        ),
    )


# --- t15 (figure provenance) -------------------------------------------------
def t15() -> None:
    """Where every number drawn by hand in Figs. 1--4 comes from.

    A matplotlib panel needs no such table: its script read the data. A draw.io
    canvas has numbers typed into it, and nothing in the exported PDF connects
    them to a source. This is that connection, and it is rendered rather than
    kept in a comment so a reader can check a figure without our library.
    """
    lines = [
        r"\begingroup\scriptsize",
        r"\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}",
        r"\begingroup\captionsetup{type=table}",
        r"\captionof{table}[]{Provenance for every number drawn by hand in "
        r"Figs.~\ref{fig:methods-map}--\ref{fig:scifi}. The matplotlib figures "
        r"are omitted because their generating scripts read the data directly. A "
        r"\textbf{note} marks a number that needed a judgement call or that was "
        r"corrected in review; an entry with no citation key is not backed by the "
        r"library mirror and should not be quoted. Generated from "
        r"\file{experiments/019-perturb-seq-costing/scripts/figure_sources.py}.}"
        r"\label{tab:figure-provenance}",
        r"\endgroup",
        # Column boundaries are EXPLICIT @{\hspace{...}} rather than the
        # default \tabcolsep, and the widths are unbalanced on purpose. With
        # 0.20/0.15/0.60 the Element column was exactly filled by its longest
        # entries ("FACS into plates throughput") while the As-drawn column ran
        # half empty, so column 1 read as though it were touching column 2 while
        # column 2 floated in whitespace. Moving width from 2 to 1 and setting a
        # real gutter fixes both; the gutter is stated in \textwidth so it
        # survives a geometry change, which \tabcolsep does not.
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.235\textwidth}"
        r"@{\hspace{0.025\textwidth}}>{\raggedright\arraybackslash}p{0.145\textwidth}"
        r"@{\hspace{0.025\textwidth}}>{\raggedright\arraybackslash}p{0.57\textwidth}@{}}",
        r"\toprule",
        r"Element & As drawn & Source \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{3}{@{}l}{\scriptsize\itshape "
        r"Table~\ref{tab:figure-provenance}, continued}\\",
        r"\toprule",
        r"Element & As drawn & Source \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    cur = None
    for r in FS.RECORDS:
        if r.figure != cur:
            cur = r.figure
            lines.append(
                r"\multicolumn{3}{@{}l}{\bfseries "
                rf"{esc(cur)}.drawio}} \\*[1pt]"
            )
        src = rf"``{esc(r.quote)}''"
        if r.citation_key:
            loc = rf"\,\texttt{{(md:{r.line})}}" if r.line else ""
            src += rf" \citep{{{r.citation_key}}}{loc}"
        else:
            src += r" \tcflagext"
        if r.note:
            src += rf" {{\color{{tcgray}}\emph{{Note:}} {esc(r.note)}}}"
        lines.append(
            rf"{esc(r.element)} & \texttt{{{esc(r.value)}}} & {src} \\"
        )
        lines.append(r"\addlinespace[2.5pt]")
    lines += [r"\end{longtable}", r"\endgroup"]
    emit("t15-figure-provenance", "\n".join(lines))


def main() -> None:
    print(f"writing LaTeX tables -> {OUT}")
    for fn in (t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t14,
               t15, t16):
        fn()
    print("done")


if __name__ == "__main__":
    main()
