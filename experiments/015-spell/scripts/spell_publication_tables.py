# experiments/015-spell/scripts/spell_publication_tables
# [[experiments.015-spell.scripts.spell_publication_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/015-spell/scripts/spell_publication_tables

"""
LaTeX tables for notes-tex/spell-publications.

Reads the manifest written by `spell_publication_manifest.py`
(experiments/015-spell/results/spell_publications.csv) and the deletion
coverage written by `spell_knockout_coverage.py`
(experiments/015-spell/results/spell_knockout_coverage.csv), and writes every
file under notes-tex/spell-publications/tables/:

    counts.tex          \\newcommand macros for each number the prose quotes
    t1-summary.tex      studies, datasets and conditions by GEO status
    t2-knockout.tex     studies with a deletion call, most deleted genes first
    t3-gene-named.tex   studies whose headers are bare gene names
    t4-no-geo.tex       the studies with no GEO accession
    t5-all.tex          every study with its links: studies that add a
                        perturbed genotype first, then most conditions first

Usage:
    python experiments/015-spell/scripts/spell_publication_tables.py
"""

import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

SCRIPT = "experiments/015-spell/scripts/spell_publication_tables.py"
CSV_PATH = osp.join(EXPERIMENT_ROOT, "015-spell", "results", "spell_publications.csv")
COVERAGE_PATH = osp.join(
    EXPERIMENT_ROOT, "015-spell", "results", "spell_knockout_coverage.csv"
)
TABLES_DIR = osp.join(
    osp.dirname(EXPERIMENT_ROOT), "notes-tex", "spell-publications", "tables"
)
GENERATED = f"%% GENERATED FILE -- do not hand-edit.\n%% SOURCE: {SCRIPT}\n"

MAX_AUTHORS = 6

# PubMed returns Unicode; the documents are typeset in T1 Latin Modern, so each
# character is written as its LaTeX form. An unmapped character is an error.
UNICODE_TEX = {
    "\xa0": " ",
    "Á": r"\'A",
    "Ç": r"\c{C}",
    "Ö": r"\"O",
    "Ü": r"\"U",
    "á": r"\'a",
    "ã": r"\~a",
    "ä": r"\"a",
    "å": r"\aa{}",
    "ç": r"\c{c}",
    "è": r"\`e",
    "é": r"\'e",
    "í": r"\'{\i}",
    "ñ": r"\~n",
    "ó": r"\'o",
    "ô": r"\^o",
    "ö": r"\"o",
    "ø": r"\o{}",
    "ú": r"\'u",
    "ü": r"\"u",
    "ć": r"\'c",
    "č": r"\v{c}",
    "ě": r"\v{e}",
    "İ": r"\.I",
    "ı": r"\i{}",
    "Š": r"\v{S}",
    "ť": r"\v{t}",
    "Δ": r"$\Delta$",
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "ζ": r"$\zeta$",
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
    return r"\href{" + url.replace("%", r"\%").replace("#", r"\#") + "}{" + shown + "}"


def breakable(s: str) -> str:
    """Escape, then allow a line break after each separator in an id or filename."""
    body = tex_escape(s)
    for sep in ("/", ".", r"\_", "-"):
        body = body.replace(sep, sep + r"\allowbreak ")
    return body


def authors_tex(authors: str) -> str:
    names = authors.split("; ")
    if len(names) > MAX_AUTHORS:
        names = names[:3] + ["et al."] + names[-1:]
    return tex_escape(", ".join(names))


def citation_tex(row: pd.Series) -> str:
    title = row.title if row.title.endswith((".", "?", "!")) else row.title + "."
    where = row.pubdate[:4]
    if row.volume:
        where += f";{row.volume}"
        if row.issue:
            where += f"({row.issue})"
    if row.pages:
        where += f":{row.pages}"
    label = f"{row.first_author} {row.folder_year}"
    return (
        r"\textbf{" + tex_escape(label) + r"}\newline "
        + r"\sourcetext{" + authors_tex(row.authors) + ". "
        + tex_escape(title) + " "
        + r"\emph{" + tex_escape(row.journal) + "}} "
        + breakable(where).replace(r"-\allowbreak ", "--") + "."
    )


def links_tex(row: pd.Series) -> str:
    parts = []
    if row.doi:
        parts.append(href(row.doi_url, "doi:" + breakable(row.doi)))
    parts.append(href(row.pubmed_url, "PMID " + row.pmid))
    if row.pmcid:
        parts.append(href(row.pmc_url, row.pmcid))
    return r"\newline ".join(parts)


def geo_tex(row: pd.Series) -> str:
    if not row.has_geo:
        return r"\textbf{none}"
    series = [row.geo_accession] + [
        g for g in row.geo_series_in_pcl_names.split("|") if g and g != row.geo_accession
    ]
    return r"\newline ".join(
        href(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={g}", g) for g in series
    )


def channels_tex(row: pd.Series) -> str:
    return row.channels.replace("|", ", ")


def write(name: str, body: str) -> None:
    path = osp.join(TABLES_DIR, name)
    with open(path, "w") as f:
        f.write(GENERATED + body)
    print(path)


def union(column: pd.Series) -> set[str]:
    return {g for genes in column for g in genes.split("|") if g}


def title_tex(row: pd.Series) -> str:
    label = f"{row.first_author} {row.folder_year}"
    return (
        r"\textbf{" + tex_escape(label) + r"}\newline "
        + r"\sourcetext{" + tex_escape(row.title.rstrip(".")) + ". "
        + r"\emph{" + tex_escape(row.journal) + "}} " + row.pubdate[:4] + "."
    )


def counts_tex(df: pd.DataFrame) -> str:
    geo, nogeo = df[df.has_geo], df[~df.has_geo]
    extra = df[df.has_geo & (df.geo_series_in_pcl_names.str.count(r"\|") > 0)]
    nogeo_pcl = [p for files in nogeo.pcl_files for p in files.split("|")]
    values = {
        "spellNStudies": len(df),
        "spellNDatasets": df.n_datasets.sum(),
        "spellNConditions": df.n_conditions.sum(),
        "spellYearMin": df.folder_year.min(),
        "spellYearMax": df.folder_year.max(),
        "spellNDoi": (df.doi != "").sum(),
        "spellNNoDoi": (df.doi == "").sum(),
        "spellNPmc": (df.pmcid != "").sum(),
        "spellNNoPmc": (df.pmcid == "").sum(),
        "spellNGeo": len(geo),
        "spellNGeoDatasets": geo.n_datasets.sum(),
        "spellNGeoConditions": geo.n_conditions.sum(),
        "spellNGeoPmc": (geo.pmcid != "").sum(),
        "spellGeoYearMin": geo.folder_year.min(),
        "spellGeoYearMax": geo.folder_year.max(),
        "spellNGeoMultiSeries": len(extra),
        "spellNNoGeo": len(nogeo),
        "spellNNoGeoDatasets": nogeo.n_datasets.sum(),
        "spellNNoGeoConditions": nogeo.n_conditions.sum(),
        "spellNNoGeoPmc": (nogeo.pmcid != "").sum(),
        "spellNNoGeoNoPmc": (nogeo.pmcid == "").sum(),
        "spellNoGeoYearMin": nogeo.folder_year.min(),
        "spellNoGeoYearMax": nogeo.folder_year.max(),
        "spellNNoGeoPclPrefixed": sum(p.startswith("2010.") for p in nogeo_pcl),
        "spellNNoGeoTwoChannel": (nogeo.channels == "2").sum(),
        "spellNKoStudies": (df.n_ko_genes > 0).sum(),
        "spellNKoConditions": df.n_ko_conditions.sum(),
        "spellNKoGenes": len(union(df.ko_genes)),
        "spellNKoGenesNew": len(union(df.ko_genes_not_in_kemmeren)),
        "spellNKoStudiesNew": (df.n_ko_genes_not_in_kemmeren > 0).sum(),
        "spellNKoStudiesTenPlus": (df.n_ko_genes >= 10).sum(),
        "spellNKoStudiesNoGeo": (nogeo.n_ko_genes > 0).sum(),
        "spellNNamedStudies": (df.n_named_conditions > 0).sum(),
        "spellNNamedConditions": df.n_named_conditions.sum(),
        "spellNNamedGenes": len(union(df.named_genes)),
        "spellNNamedStudiesNoGeo": (nogeo.n_named_conditions > 0).sum(),
        "spellNNewStudies": (df.n_new_genotypes > 0).sum(),
        "spellNNewGenotypes": df.n_new_genotypes.sum(),
        "spellNNewKoGenotypes": df.n_ko_genotypes_new.sum(),
        "spellNNewKoStudies": (df.n_ko_genotypes_new > 0).sum(),
        "spellNNewNamedGenotypes": df.n_named_genotypes_new.sum(),
        "spellNNewStudiesFivePlus": (df.n_new_genotypes >= 5).sum(),
    }
    lines = [rf"\newcommand{{\{k}}}{{{int(v):,}}}".replace(",", "{,}") for k, v in values.items()]
    # Years are not thousands-separated.
    lines = [
        rf"\newcommand{{\{k}}}{{{int(v)}}}" if "Year" in k else line
        for (k, v), line in zip(values.items(), lines)
    ]
    return "\n".join(lines) + "\n"


def summary_tex(df: pd.DataFrame) -> str:
    rows = []
    for label, sub in (
        ("GEO accession in README", df[df.has_geo]),
        (r"no GEO accession (\texttt{GEO ID: N/A})", df[~df.has_geo]),
        (r"\textbf{all}", df),
    ):
        if label.startswith(r"\textbf"):
            rows.append(r"\midrule")
        rows.append(
            f"{label} & {len(sub):,} & {sub.n_datasets.sum():,} & "
            f"{sub.n_conditions.sum():,} & {(sub.doi != '').sum():,} & "
            f"{(sub.pmcid != '').sum():,} & "
            f"{sub.folder_year.min()}--{sub.folder_year.max()} \\\\"
        )
    return (
        r"""\begin{table}[H]\centering
\footnotesize
\caption[]{Studies in the SPELL archive by whether the study README names a GEO
series. \emph{PCL datasets} counts the dataset rows in the READMEs, one per PCL
matrix file. \emph{Conditions} sums the \texttt{\# conditions} field over those
rows. \emph{DOI} and \emph{PMC id} count the studies whose PubMed record carries
one. \emph{Years} is the range of publication years in the folder names.}\label{tab:summary}
\begin{tabular}{lrrrrrl}
\toprule
& studies & PCL datasets & conditions & DOI & PMC id & years \\
\midrule
"""
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def longtable(colspec: str, header: str, ncols: int, caption: str, label: str, rows: list[str]) -> str:
    return (
        r"\begin{landscape}" + "\n" + r"\begingroup" + "\n" + r"\footnotesize" + "\n"
        + r"\setlength{\tabcolsep}{4pt}" + "\n"
        + r"\renewcommand{\arraystretch}{1.1}" + "\n"
        + r"\begin{longtable}{" + colspec + "}\n"
        + r"\caption[]{" + caption + "}\n"
        + r"\label{" + label + r"}\\" + "\n"
        + "\\toprule\n" + header + " \\\\\n\\midrule\n\\endfirsthead\n"
        + rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\footnotesize\emph{{Table~\ref{{{label}}}, continued}}}}\\"
        + "\n\\toprule\n" + header + " \\\\\n\\midrule\n\\endhead\n"
        + "\\bottomrule\n\\endfoot\n"
        + "\n\\addlinespace[3pt]\n".join(rows)
        + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def all_tex(df: pd.DataFrame) -> str:
    rows = [
        f"{r.number} & {citation_tex(r)} & {links_tex(r)} & {geo_tex(r)} & "
        f"{bold_dash(r.n_new_genotypes)} & {r.n_conditions:,} & "
        f"{dash(r.n_ko_conditions)} & {dash(r.n_ko_genes)} & "
        f"{dash(r.n_named_genes)} & {r.n_datasets} & {channels_tex(r)} \\\\"
        for r in df.itertuples()
    ]
    return longtable(
        colspec=r"@{}r@{\hspace{5pt}} L{102mm} L{42mm} L{20mm} r r r r r r r@{}",
        header=(
            r"\textbf{\#} & \textbf{Publication} & \textbf{Links} & \textbf{GEO} & "
            r"\textbf{new} & \textbf{cond.} & \textbf{KO cond.} & "
            r"\textbf{KO genes} & \textbf{named} & \textbf{PCL} & \textbf{ch.}"
        ),
        ncols=11,
        caption=(
            r"Every study in the SPELL archive. The studies that add a perturbed "
            r"genotype come first, most new genotypes first; the rest follow, most "
            r"conditions first. \emph{new} is the number of genotypes, deletion or "
            r"gene-named, found in neither Kemmeren 2014 nor Sameith 2015 "
            r"(Sec.~\ref{sec:new}); a dash is zero. The bold label is the "
            r"folder's first author and "
            r"year. \emph{Links} gives the DOI, the PubMed record and, where one exists, "
            r"the PubMed Central copy. \emph{GEO} is the series named in the README, "
            r"followed by any further series that appear in the study's PCL filenames; "
            r"\textbf{none} marks a README with \texttt{GEO ID: N/A}, and those rows "
            r"are repeated in Table~\ref{tab:no-geo}. \emph{cond.} is the number of "
            r"conditions summed over the study's PCL matrices. \emph{KO cond.} and "
            r"\emph{KO genes} are the deletion conditions and distinct deleted genes "
            r"estimated from the column headers (Sec.~\ref{sec:knockout}), and "
            r"\emph{named} is the number of genes in headers that are bare gene names; "
            r"a dash is zero. \emph{PCL} is the number of dataset matrices and "
            r"\emph{ch.} the microarray channel count the README reports. Every link "
            r"is clickable."
        ),
        label="tab:all",
        rows=rows,
    )


def dash(n: int) -> str:
    return f"{n:,}" if n else "--"


def bold_dash(n: int) -> str:
    return rf"\textbf{{{n:,}}}" if n else "--"


def knockout_tex(df: pd.DataFrame) -> str:
    sub = df[df.n_ko_genes > 0].sort_values(
        ["n_ko_genes", "n_ko_conditions", "study_name"], ascending=[False, False, True]
    )
    rows = [
        f"{r.number} & {title_tex(r)} & {r.n_ko_genes:,} & "
        f"{dash(r.n_ko_genotypes_new)} & {r.n_ko_genotypes:,} & "
        f"{r.n_ko_conditions:,} & {r.n_conditions:,} & {geo_tex(r)} & "
        + (href(r.doi_url, "doi") if r.doi else href(r.pubmed_url, "PubMed"))
        + " \\\\"
        for r in sub.itertuples()
    ]
    return longtable(
        colspec=r"@{}r@{\hspace{5pt}} L{128mm} r r r r r L{20mm} l@{}",
        header=(
            r"\textbf{\#} & \textbf{Publication} & \textbf{KO genes} & "
            r"\textbf{new} & \textbf{genotypes} & \textbf{KO cond.} & "
            r"\textbf{cond.} & \textbf{GEO} & \textbf{link}"
        ),
        ncols=9,
        caption=(
            r"Studies with at least one deletion call, most deleted genes first. "
            r"\emph{\#} is the row number in Table~\ref{tab:all}. \emph{KO genes} is "
            r"the number of distinct genes a deletion marker is attached to in the "
            r"study's column headers. \emph{genotypes} counts distinct sets of deleted "
            r"genes, so a double mutant is one genotype, and \emph{new} is how many "
            r"of those genotypes are in neither Kemmeren 2014 nor Sameith 2015 "
            r"(Sec.~\ref{sec:new}); a dash is zero. \emph{KO cond.} is the number of conditions "
            r"with a call and \emph{cond.} the study's total. The calls are read "
            r"from header text and are estimates (Sec.~\ref{sec:knockout})."
        ),
        label="tab:knockout",
        rows=rows,
    )


def gene_named_tex(df: pd.DataFrame) -> str:
    sub = df[df.n_named_conditions > 0].sort_values(
        ["n_named_genes", "study_name"], ascending=[False, True]
    )
    rows = [
        f"{r.number} & \\textbf{{{tex_escape(r.first_author)} {r.folder_year}}} & "
        f"{r.n_named_genes:,} & {dash(r.n_named_genotypes_new)} & "
        f"{r.n_named_conditions:,} & {r.n_conditions:,} & "
        f"{geo_tex(r)} & "
        + (href(r.doi_url, "doi") if r.doi else href(r.pubmed_url, "PubMed"))
        + " \\\\"
        for r in sub.itertuples()
    ]
    return (
        r"""\begin{table}[H]\centering
\footnotesize
\caption[]{Studies with conditions labeled by gene name alone. \emph{\#} is the
row number in Table~\ref{tab:all}. \emph{genes} counts the distinct genes named,
\emph{new} the gene-named genotypes in neither Kemmeren 2014 nor Sameith 2015,
\emph{named cond.} the conditions labeled that way and \emph{cond.} the study's
total. The header does not say how the gene was perturbed, so these are not
counted as deletions.}\label{tab:gene-named}
\begin{tabular}{rlrrrrll}
\toprule
\# & study & genes & new & named cond. & cond. & GEO & link \\
\midrule
"""
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def no_geo_tex(df: pd.DataFrame) -> str:
    rows = []
    for r in df[~df.has_geo].itertuples():
        pcl = r"\newline ".join(breakable(p) for p in r.pcl_files.split("|"))
        tags = tex_escape(r.tags.replace("|", "; "))
        rows.append(
            f"{r.number} & {citation_tex(r)} & {links_tex(r)} & "
            f"{{\\ttfamily\\scriptsize {pcl}}} & {tags} & {r.n_conditions:,} & "
            f"{dash(r.n_ko_genes)} & {channels_tex(r)} \\\\"
        )
    return longtable(
        colspec=r"@{}r@{\hspace{5pt}} L{82mm} L{40mm} L{60mm} L{28mm} r r r@{}",
        header=(
            r"\textbf{\#} & \textbf{Publication} & \textbf{Links} & "
            r"\textbf{PCL files in the archive} & \textbf{SPELL tags} & "
            r"\textbf{cond.} & \textbf{KO genes} & \textbf{ch.}"
        ),
        ncols=8,
        caption=(
            r"The studies whose README reads \texttt{GEO ID: N/A}, in the order of "
            r"Table~\ref{tab:all}. \emph{\#} is the row number in Table~\ref{tab:all} and "
            r"\emph{KO genes} the estimate of Sec.~\ref{sec:knockout}. \emph{PCL files in the archive} lists "
            r"each matrix SPELL distributes for the study, which is the only data "
            r"location the archive gives for these rows. \emph{SPELL tags} are the "
            r"topic tags in the README dataset table. A row with a PMC link has a "
            r"full text in PubMed Central to search for the data location; a row "
            r"without one does not."
        ),
        label="tab:no-geo",
        rows=rows,
    )


def main() -> None:
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    df["has_geo"] = df.has_geo.map({"True": True, "False": False})
    for col in ("folder_year", "n_datasets", "n_conditions"):
        df[col] = df[col].astype(int)
    coverage = pd.read_csv(COVERAGE_PATH, keep_default_na=False)
    df = df.merge(coverage, on="study_name", validate="one_to_one")
    assert len(df) == len(coverage)
    assert (df.n_conditions == df.n_conditions_in_pcl).all()
    df["n_new_genotypes"] = df.n_ko_genotypes_new + df.n_named_genotypes_new
    df = df.sort_values(
        ["n_new_genotypes", "n_conditions", "n_ko_genes", "study_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    df["number"] = df.index + 1

    os.makedirs(TABLES_DIR, exist_ok=True)
    write("counts.tex", counts_tex(df))
    write("t1-summary.tex", summary_tex(df))
    write("t2-knockout.tex", knockout_tex(df))
    write("t3-gene-named.tex", gene_named_tex(df))
    write("t4-no-geo.tex", no_geo_tex(df))
    write("t5-all.tex", all_tex(df))


if __name__ == "__main__":
    main()
