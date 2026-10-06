# experiments/037-yeastract/scripts/yeastract_publication_tables
# [[experiments.037-yeastract.scripts.yeastract_publication_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/037-yeastract/scripts/yeastract_publication_tables

r"""
LaTeX tables for notes-tex/yeastract-publications.

Reads the manifest written by `yeastract_publication_manifest.py`
(experiments/037-yeastract/results/yeastract_publications.csv), takes the SPELL
study count from experiments/015-spell/results/spell_publications.csv, and
writes every file under notes-tex/yeastract-publications/tables/:

    counts.tex        \\newcommand macros for each number the prose quotes
    t1-summary.tex    papers and rows by relation to the SPELL archive
    t2-sizes.tex      papers and rows by the number of rows a paper contributes
    t3-leverage.tex   papers outside SPELL, most rows first, to LEVERAGE_PCT of
                      the rows outside SPELL
    t4-all.tex        every paper, most rows first, with its links

Usage:
    python experiments/037-yeastract/scripts/yeastract_publication_tables.py
"""

import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

SCRIPT = "experiments/037-yeastract/scripts/yeastract_publication_tables.py"
CSV_PATH = osp.join(
    EXPERIMENT_ROOT, "037-yeastract", "results", "yeastract_publications.csv"
)
SPELL_CSV = osp.join(EXPERIMENT_ROOT, "015-spell", "results", "spell_publications.csv")
TABLES_DIR = osp.join(
    osp.dirname(EXPERIMENT_ROOT), "notes-tex", "yeastract-publications", "tables"
)
GENERATED = f"%% GENERATED FILE -- do not hand-edit.\n%% SOURCE: {SCRIPT}\n"

MAX_AUTHORS = 6
# The leverage table stops at the paper that brings the cumulative share of
# rows outside SPELL to this percentage.
LEVERAGE_PCT = 95
COVERAGE_PCTS = (50, 80, 90, 95, 99)
SIZE_BINS = [
    ("1", 1, 1),
    ("2--5", 2, 5),
    ("6--10", 6, 10),
    ("11--100", 11, 100),
    ("101--1,000", 101, 1000),
    ("1,001--10,000", 1001, 10000),
    ("more than 10,000", 10001, None),
]
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="

# PubMed returns Unicode; the documents are typeset in T1 Latin Modern, so each
# character is written as its LaTeX form. An unmapped character is an error.
UNICODE_TEX = {
    "\xa0": " ",
    "Á": r"\'A",
    "Ö": r"\"O",
    "Ü": r"\"U",
    "Š": r"\v{S}",
    "Ž": r"\v{Z}",
    "á": r"\'a",
    "ä": r"\"a",
    "æ": r"\ae{}",
    "ç": r"\c{c}",
    "è": r"\`e",
    "é": r"\'e",
    "ê": r"\^e",
    "ë": r"\"e",
    "í": r"\'{\i}",
    "ï": r"\"{\i}",
    "ñ": r"\~n",
    "ò": r"\`o",
    "ó": r"\'o",
    "ô": r"\^o",
    "õ": r"\~o",
    "ö": r"\"o",
    "ø": r"\o{}",
    "ú": r"\'u",
    "ü": r"\"u",
    "ć": r"\'c",
    "č": r"\v{c}",
    "ł": r"\l{}",
    "ń": r"\'n",
    "ş": r"\c{s}",
    "š": r"\v{s}",
    "ž": r"\v{z}",
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "ρ": r"$\rho$",
}
ASCII_TEX = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "_": r"\_",
    "#": r"\#",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}


def tex_escape(s: str) -> str:
    out = []
    for ch in s:
        if ch in ASCII_TEX:
            out.append(ASCII_TEX[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append(UNICODE_TEX[ch])
    return "".join(out)


def href(url: str, shown: str) -> str:
    # Angle brackets occur in SICI-form DOIs and are percent-encoded in a URL.
    url = url.replace("<", "%3C").replace(">", "%3E")
    return r"\href{" + url.replace("%", r"\%").replace("#", r"\#") + "}{" + shown + "}"


def breakable(s: str) -> str:
    """Escape, then allow a line break after each separator in an id."""
    body = tex_escape(s)
    for sep in ("/", ".", r"\_", "-", ":", ";", ")"):
        body = body.replace(sep, sep + r"\allowbreak ")
    return body


def dash(n: int) -> str:
    return f"{n:,}" if n else "--"


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}"


def label_tex(row: pd.Series) -> str:
    if not row.pubmed_found:
        return r"\textbf{PMID " + row.pmid + "}"
    return r"\textbf{" + tex_escape(f"{row.first_author} {row.year}") + "}"


def authors_tex(authors: str) -> str:
    names = authors.split("; ")
    if len(names) > MAX_AUTHORS:
        names = names[:3] + ["et al."] + names[-1:]
    return tex_escape(", ".join(names))


def citation_tex(row: pd.Series) -> str:
    if not row.pubmed_found:
        return label_tex(row) + r"\newline PubMed returns no record for this id."
    title = row.title if row.title.endswith((".", "?", "!")) else row.title + "."
    where = str(row.year)
    if row.volume:
        where += f";{row.volume}"
        if row.issue:
            where += f"({row.issue})"
    if row.pages:
        where += f":{row.pages}"
    return (
        label_tex(row)
        + r"\newline "
        + r"\sourcetext{"
        + authors_tex(row.authors)
        + ". "
        + tex_escape(title)
        + " "
        + r"\emph{"
        + tex_escape(row.journal)
        + "}} "
        + breakable(where).replace(r"-\allowbreak ", "--")
        + "."
    )


def title_tex(row: pd.Series) -> str:
    if not row.pubmed_found:
        return citation_tex(row)
    return (
        label_tex(row)
        + r"\newline "
        + r"\sourcetext{"
        + tex_escape(row.title.rstrip("."))
        + ". "
        + r"\emph{"
        + tex_escape(row.journal)
        + "}} "
        + str(row.year)
        + "."
    )


def links_tex(row: pd.Series) -> str:
    parts = []
    if row.doi:
        parts.append(href(row.doi_url, "doi:" + breakable(row.doi)))
    parts.append(href(row.pubmed_url, "PMID " + row.pmid))
    if row.pmcid:
        parts.append(href(row.pmc_url, row.pmcid))
    return r"\newline ".join(parts)


def link_tex(row: pd.Series) -> str:
    return href(row.doi_url, "doi") if row.doi else href(row.pubmed_url, "PubMed")


def spell_label(study_name: str) -> str:
    """`Hu_2007_PMID_17417638` -> `Hu 2007`."""
    return tex_escape(study_name.split("_PMID_")[0].replace("_", " "))


def spell_tex(row: pd.Series) -> str:
    """The paper's place in the SPELL archive: its GEO series, or its absence."""
    if row.in_spell:
        if row.spell_geo:
            return href(GEO_URL + row.spell_geo, row.spell_geo)
        return "no GEO"
    if row.reanalysis_of_spell_study:
        shown = "via " + spell_label(row.reanalysis_of_spell_study)
        if row.reanalysis_of_spell_geo:
            geo = row.reanalysis_of_spell_geo
            shown += r"\newline " + href(GEO_URL + geo, geo)
        return shown
    return "--"


def assay_tex(row: pd.Series) -> str:
    if not row.top_assay:
        return "--"
    return r"\sourcetext{" + tex_escape(row.top_assay) + "}"


def write(name: str, body: str) -> None:
    path = osp.join(TABLES_DIR, name)
    with open(path, "w") as f:
        f.write(GENERATED + body)
    print(path)


def macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def number(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def counts_tex(df: pd.DataFrame) -> str:
    total = df.n_rows.sum()
    same = df[df.in_spell]
    reanalysis = df[df.relation == "reanalysis"]
    outside = df[~df.in_spell]
    absent = df[df.relation == "absent"]
    found = df[df.pubmed_found]
    top = outside.iloc[0], outside.iloc[1]
    leverage = outside[outside.outside_rank <= leverage_cutoff(outside)]
    values = {
        "ytNRows": number(total),
        "ytNPapers": number(len(df)),
        "ytNTfs": number(df.attrs["n_tfs"]),
        "ytNTargets": number(df.attrs["n_targets"]),
        "ytNPairs": number(df.attrs["n_pairs"]),
        "ytNDirectRows": number(df.n_direct.sum()),
        "ytNIndirectRows": number(df.n_indirect.sum()),
        "ytNEvidenceNaRows": number(df.n_evidence_na.sum()),
        "ytYearMin": str(found.year.min()),
        "ytYearMax": str(found.year.max()),
        "ytNNoPubmed": number((~df.pubmed_found).sum()),
        "ytNDoi": number((df.doi != "").sum()),
        "ytNNoDoi": number((found.doi == "").sum()),
        "ytNPmc": number((df.pmcid != "").sum()),
        "ytNSpellPapers": number(len(same)),
        "ytNSpellRows": number(same.n_rows.sum()),
        "ytPctSpellRows": pct(same.n_rows.sum(), total),
        "ytNSpellIndirectRows": number(same.n_indirect.sum()),
        "ytNSpellDirectRows": number(same.n_direct.sum()),
        "ytNSpellNoGeo": number((same.spell_geo == "").sum()),
        "ytNReanalysisPapers": number(len(reanalysis)),
        "ytNReanalysisRows": number(reanalysis.n_rows.sum()),
        "ytPctReanalysisRows": pct(reanalysis.n_rows.sum(), total),
        "ytNOutsidePapers": number(len(outside)),
        "ytNOutsideRows": number(outside.n_rows.sum()),
        "ytPctOutsideRows": pct(outside.n_rows.sum(), total),
        "ytNAbsentPapers": number(len(absent)),
        "ytNAbsentRows": number(absent.n_rows.sum()),
        "ytPctAbsentRows": pct(absent.n_rows.sum(), total),
        "ytTopOneRows": number(top[0].n_rows),
        "ytTopOnePct": pct(top[0].n_rows, total),
        "ytTopTwoRows": number(top[1].n_rows),
        "ytTopTwoPct": pct(top[1].n_rows, total),
        "ytLeveragePct": str(LEVERAGE_PCT),
        "ytNLeveragePapers": number(len(leverage)),
        "ytNLeverageRows": number(leverage.n_rows.sum()),
        "ytNLeverageIndirectPapers": number(
            (leverage.n_indirect > leverage.n_direct).sum()
        ),
        "ytNLeverageDirectPapers": number(
            (leverage.n_direct >= leverage.n_indirect).sum()
        ),
        "ytNTailPapers": number(len(outside) - len(leverage)),
        "ytNTailRows": number(outside.n_rows.sum() - leverage.n_rows.sum()),
        "ytNOneRowPapers": number((df.n_rows == 1).sum()),
        "ytNFiveRowPapers": number((df.n_rows <= 5).sum()),
        "ytNFiveRowRows": number(df[df.n_rows <= 5].n_rows.sum()),
        "ytNNoAssayPapers": number((df.top_assay == "").sum()),
    }
    spell = pd.read_csv(SPELL_CSV, dtype=str, keep_default_na=False).set_index("pmid")
    values["ytNSpellStudies"] = number(len(spell))
    (source,) = reanalysis.reanalysis_of_pmid
    values["ytReanalysisSourceConditions"] = number(int(spell.n_conditions[source]))
    values["ytReanalysisSourceGeo"] = spell.geo_accession[source]
    for share in COVERAGE_PCTS:
        name = {
            50: "Fifty",
            80: "Eighty",
            90: "Ninety",
            95: "NinetyFive",
            99: "NinetyNine",
        }
        values[f"ytNPapersTo{name[share]}"] = number(
            (outside.outside_cum_rows < share / 100 * outside.n_rows.sum()).sum() + 1
        )
    return "\n".join(macro(k, v) for k, v in values.items()) + "\n"


def leverage_cutoff(outside: pd.DataFrame) -> int:
    """Rank of the paper that brings cumulative rows outside SPELL to LEVERAGE_PCT."""
    below = outside.outside_cum_rows < LEVERAGE_PCT / 100 * outside.n_rows.sum()
    return int(below.sum()) + 1


def summary_tex(df: pd.DataFrame) -> str:
    total = df.n_rows.sum()
    found = df[df.pubmed_found]
    rows = []
    for label, sub in (
        ("same PMID in SPELL", df[df.relation == "same"]),
        ("reanalysis of a SPELL study", df[df.relation == "reanalysis"]),
        ("not in SPELL", df[df.relation == "absent"]),
        (r"\textbf{all}", df),
    ):
        if label.startswith(r"\textbf"):
            rows.append(r"\midrule")
        years = found.year[found.pmid.isin(sub.pmid)]
        rows.append(
            f"{label} & {len(sub):,} & {sub.n_rows.sum():,} & "
            f"{pct(sub.n_rows.sum(), total)} & {sub.n_direct.sum():,} & "
            f"{sub.n_indirect.sum():,} & {(sub.doi != '').sum():,} & "
            f"{(sub.pmcid != '').sum():,} & {years.min()}--{years.max()} \\\\"
        )
    return (
        r"""\begin{table}[H]\centering
\footnotesize
\caption[]{Papers and rows in the YEASTRACT+ flat file by relation to the SPELL
archive. \emph{Rows} counts flat-file rows credited to the papers and \emph{\%}
is their share of all rows. \emph{Direct} and \emph{indirect} count rows by the
evidence label in column 7 of the file; rows labeled \texttt{N/A} are in
neither. \emph{DOI} and \emph{PMC id} count the papers whose PubMed record
carries one. \emph{Years} is the range of publication years.}\label{tab:summary}
\begin{tabular}{lrrrrrrrl}
\toprule
& papers & rows & \% & direct & indirect & DOI & PMC id & years \\
\midrule
"""
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def sizes_tex(df: pd.DataFrame) -> str:
    total = df.n_rows.sum()
    rows = []
    for label, low, high in SIZE_BINS:
        sub = df[(df.n_rows >= low) & (df.n_rows <= (high or df.n_rows.max()))]
        rows.append(
            f"{label} & {len(sub):,} & {sub.n_rows.sum():,} & "
            f"{pct(sub.n_rows.sum(), total)} & {dash(sub.n_direct.sum())} & "
            f"{dash(sub.n_indirect.sum())} & {dash(int((sub.relation != 'absent').sum()))} \\\\"
        )
    rows.append(r"\midrule")
    rows.append(
        rf"\textbf{{all}} & {len(df):,} & {total:,} & 100.0 & {df.n_direct.sum():,} & "
        f"{df.n_indirect.sum():,} & {(df.relation != 'absent').sum():,} \\\\"
    )
    return (
        r"""\begin{table}[H]\centering
\footnotesize
\caption[]{Papers by the number of flat-file rows each contributes. \emph{Rows}
sums over the papers in the band and \emph{\%} is the share of all rows.
\emph{Direct} and \emph{indirect} count rows by evidence label. \emph{SPELL}
counts the papers in the band that are in the SPELL archive, by PMID or as a
reanalysis; a dash is zero.}\label{tab:sizes}
\begin{tabular}{lrrrrrr}
\toprule
rows per paper & papers & rows & \% & direct & indirect & SPELL \\
\midrule
"""
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def longtable(
    colspec: str, header: str, ncols: int, caption: str, label: str, rows: list[str]
) -> str:
    return (
        r"\begin{landscape}"
        + "\n"
        + r"\begingroup"
        + "\n"
        + r"\footnotesize"
        + "\n"
        + r"\setlength{\tabcolsep}{4pt}"
        + "\n"
        + r"\renewcommand{\arraystretch}{1.1}"
        + "\n"
        + r"\begin{longtable}{"
        + colspec
        + "}\n"
        + r"\caption[]{"
        + caption
        + "}\n"
        + r"\label{"
        + label
        + r"}\\"
        + "\n"
        + "\\toprule\n"
        + header
        + " \\\\\n\\midrule\n\\endfirsthead\n"
        + rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\footnotesize\emph{{Table~\ref{{{label}}}, continued}}}}\\"
        + "\n\\toprule\n"
        + header
        + " \\\\\n\\midrule\n\\endhead\n"
        + "\\bottomrule\n\\endfoot\n"
        + "\n\\addlinespace[3pt]\n".join(rows)
        + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def leverage_tex(df: pd.DataFrame) -> str:
    outside = df[~df.in_spell]
    sub = outside[outside.outside_rank <= leverage_cutoff(outside)]
    outside_rows = outside.n_rows.sum()
    rows = [
        f"{r.outside_rank} & {r.number} & {title_tex(r)} & {r.n_rows:,} & "
        f"{pct(r.outside_cum_rows, outside_rows)} & {r.n_tfs:,} & {r.n_pairs:,} & "
        f"{dash(r.n_direct)} & {dash(r.n_indirect)} & {r.n_conditions:,} & "
        f"{assay_tex(r)} & {spell_tex(r)} & {link_tex(r)} \\\\"
        for r in sub.itertuples()
    ]
    return longtable(
        colspec=r"@{}r r@{\hspace{5pt}} L{84mm} r r r r r r r L{38mm} L{20mm} l@{}",
        header=(
            r"\textbf{rank} & \textbf{\#} & \textbf{Publication} & \textbf{rows} & "
            r"\textbf{cum.\ \%} & \textbf{TFs} & \textbf{pairs} & \textbf{direct} & "
            r"\textbf{indirect} & \textbf{cond.} & \textbf{top assay} & "
            r"\textbf{SPELL} & \textbf{link}"
        ),
        ncols=13,
        caption=(
            r"Papers whose PMID is not in the SPELL archive, most rows first, down "
            r"to the paper that brings the cumulative share to "
            + str(LEVERAGE_PCT)
            + r" percent. \emph{\#} is the row number in Table~\ref{tab:all}. "
            r"\emph{cum.\ \%} is the cumulative share of the rows credited to "
            r"papers outside SPELL. \emph{TFs} is the number of distinct "
            r"transcription factors and \emph{pairs} the number of distinct "
            r"factor-target pairs in the paper's rows. \emph{Direct} and "
            r"\emph{indirect} count rows by evidence label; a dash is zero. "
            r"\emph{cond.} is the number of distinct free-text conditions and "
            r"\emph{top assay} the most frequent value of the assay column. "
            r"\emph{SPELL} names the archive study a reanalysis draws on "
            r"(Sec.~\ref{sec:spell})."
        ),
        label="tab:leverage",
        rows=rows,
    )


def all_tex(df: pd.DataFrame) -> str:
    rows = [
        f"{r.number} & {citation_tex(r)} & {links_tex(r)} & {spell_tex(r)} & "
        f"{r.n_rows:,} & {r.n_tfs:,} & {r.n_pairs:,} & {dash(r.n_direct)} & "
        f"{dash(r.n_indirect)} & {r.n_conditions:,} & {assay_tex(r)} \\\\"
        for r in df.itertuples()
    ]
    return longtable(
        colspec=r"@{}r@{\hspace{5pt}} L{82mm} L{40mm} L{20mm} r r r r r r L{34mm}@{}",
        header=(
            r"\textbf{\#} & \textbf{Publication} & \textbf{Links} & \textbf{SPELL} & "
            r"\textbf{rows} & \textbf{TFs} & \textbf{pairs} & \textbf{direct} & "
            r"\textbf{indirect} & \textbf{cond.} & \textbf{top assay}"
        ),
        ncols=11,
        caption=(
            r"Every paper credited in the YEASTRACT+ flat file, most rows first; "
            r"ties are ordered by PMID. The bold label is the first author and "
            r"year. \emph{Links} gives the DOI, the PubMed record and, where one "
            r"exists, the PubMed Central copy. \emph{SPELL} is the GEO series of "
            r"the SPELL study with the same PMID, \emph{no GEO} where that study "
            r"names none, \emph{via} a study where the paper reanalyzes it, and a "
            r"dash where the paper is not in the archive. \emph{rows} is the "
            r"number of flat-file rows credited to the paper, \emph{TFs} the "
            r"distinct transcription factors and \emph{pairs} the distinct "
            r"factor-target pairs in them. \emph{Direct} and \emph{indirect} count "
            r"rows by evidence label; a dash is zero. \emph{cond.} is the number "
            r"of distinct free-text conditions and \emph{top assay} the most "
            r"frequent value of the assay column, a dash where that value is "
            r"empty. Every link is clickable."
        ),
        label="tab:all",
        rows=rows,
    )


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    for col in ("pubmed_found", "in_spell"):
        df[col] = df[col].map({"True": True, "False": False})
    for col in [c for c in df.columns if c.startswith("n_")]:
        df[col] = df[col].astype(int)
    df["year"] = pd.to_numeric(df.year).astype("Int64")
    df["pmid_number"] = df.pmid.astype(int)
    df = df.sort_values(["n_rows", "pmid_number"], ascending=[False, True]).reset_index(
        drop=True
    )
    df["number"] = df.index + 1
    df["relation"] = "absent"
    df.loc[df.reanalysis_of_spell_study != "", "relation"] = "reanalysis"
    df.loc[df.in_spell, "relation"] = "same"
    outside = ~df.in_spell
    df["outside_rank"] = outside.cumsum().where(outside, 0)
    df["outside_cum_rows"] = df.n_rows.where(outside, 0).cumsum().where(outside, 0)
    return df


def flatfile_totals() -> dict[str, int]:
    """Whole-file distinct counts, which do not sum over papers."""
    from yeastract_publication_manifest import read_flatfile

    flat = read_flatfile()
    return {
        "n_tfs": flat.tf.nunique(),
        "n_targets": flat.target.nunique(),
        "n_pairs": len(flat[["tf", "target"]].drop_duplicates()),
    }


def main() -> None:
    df = load()
    df.attrs.update(flatfile_totals())

    os.makedirs(TABLES_DIR, exist_ok=True)
    write("counts.tex", counts_tex(df))
    write("t1-summary.tex", summary_tex(df))
    write("t2-sizes.tex", sizes_tex(df))
    write("t3-leverage.tex", leverage_tex(df))
    write("t4-all.tex", all_tex(df))


if __name__ == "__main__":
    main()
