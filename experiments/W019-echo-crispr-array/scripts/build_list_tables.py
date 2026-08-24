# experiments/W019-echo-crispr-array/scripts/build_list_tables.py
# [[experiments.W019-echo-crispr-array.scripts.build_list_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/build_list_tables
"""Emit the bench-facing strain build list as LaTeX tables and as CSV.

Five tables for notes-tex/w019-strain-build-list/, all re-derived from the pinned
CSVs rather than transcribed:

  t1-existing-singles   the 12 singles that already exist, measured vs published
  t2-existing-doubles   the 13 doubles that already exist, measured vs published
  t3-new-doubles        the 25 doubles to construct, D01-D25
  t4-new-triples        the 20 triples to construct, T01-T20
  t5-plate              what goes on the measurement plate, and the well budget

Each table is built ONCE as a list of records and rendered from that one
structure into both forms, so the .tex and the .csv cannot disagree. The CSV is
the data form: no LaTeX, missing values empty rather than a dash, ORF and common
name in separate columns, and the lab attribution that the .tex carries in its
`\\cmidrule` group headers moved into the column NAMES (`ours_fitness` vs
`costanzo_smf`), which is the only place a flat file can hold it.

ID assignment, and it is the part that has to be reproducible:

  D ids  the 25 new doubles sorted lexicographically by (gene1, gene2)
  T ids  the 20 triples sorted by `routes` ascending, then by the design ranking
         carried in triple_design_rank_sampling_selection.csv

`routes` is how many of a triple's three pairs are already built, so it is the
number of distinct crosses that can produce it from strains in hand. A triple at
`routes` 1 has a single way in and is scheduled first.

Inputs (all committed, none written by this script)
  results/run4_measured_summary_singles.csv
  results/run4_measured_summary_doubles.csv
  results/run4_measured_summary_gaps.csv
  results/triple_design_rank_sampling_selection.csv    strategy == "capped"

Outputs
  notes-tex/w019-strain-build-list/tables/t{1..5}-*.tex
  results/build_list_tables/t{1..5}-*.csv

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/build_list_tables.py
"""

from __future__ import annotations

import itertools
import os
import os.path as osp

import pandas as pd

EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
REPO = osp.dirname(osp.dirname(EXP_DIR))
RESULTS = osp.join(EXP_DIR, "results")
OUT_TEX = osp.join(REPO, "notes-tex", "w019-strain-build-list", "tables")
# A subdirectory of its own: results/ already holds about ninety files, and a
# bare `t1-...csv` sitting among them says nothing about which document it backs.
OUT_CSV = osp.join(RESULTS, "build_list_tables")

SINGLES = osp.join(RESULTS, "run4_measured_summary_singles.csv")
DOUBLES = osp.join(RESULTS, "run4_measured_summary_doubles.csv")
GAPS = osp.join(RESULTS, "run4_measured_summary_gaps.csv")
SELECTION = osp.join(RESULTS, "triple_design_rank_sampling_selection.csv")

# The .tex files carry a provenance header; the CSVs deliberately do not. A
# comment line in a CSV is a parsing hazard, and the run summary prints the path
# of every file written instead.
HEADER = (
    "%% GENERATED FILE -- do not hand-edit.\n"
    "%% SOURCE: experiments/W019-echo-crispr-array/scripts/build_list_tables.py\n"
)

RULES = (r"\midrule", r"\addlinespace", r"\cmidrule")

# Plate constants. A 384-well plate, six wells left empty so the plate's
# orientation is unambiguous when it is read, and twenty given to the wild-type
# reference every other strain is normalized against.
PLATE_WELLS = 384
ORIENTATION_BLANKS = 6
WT_WELLS = 20

# name -> paths and row counts, filled by emit() and checked at the end of main().
WRITTEN: dict[str, dict] = {}


def table(spec: str, head: list[str], rows: list[str], caption: str, label: str,
          size: str = r"\footnotesize") -> str:
    """A booktabs table that sits inline where it is \\input.

    `head` is a list of pre-built header lines so a table can carry a grouping
    row plus \\cmidrule above the column names. `\\caption[]{}` with the empty
    optional argument throughout, because captions here contain \\file, which is
    \\DeclareUrlCommand-based and therefore fragile in a moving argument.
    """
    # [H], not [htbp]. Each of these is the subject of the two sentences directly
    # above it, and letting LaTeX float them produced a float page carrying
    # Tables 1 and 2 with half a page of white between, detached from the prose
    # that says what to do with them. `float` is loaded by main.tex.
    return "\n".join([
        r"\begin{table}[H]\centering",
        size,
        rf"\caption[]{{{caption}}}\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        *head,
        r"\midrule",
        # A row that IS a rule carries no `\\`. Appending one to \midrule emits an
        # empty table line, which booktabs renders as a stray gap under the rule.
        # Matched on the rule commands, not on a leading backslash: real rows open
        # with \textbf often enough that the looser test would break them.
        *[r if r.startswith(RULES) else r + r" \\" for r in rows],
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def emit(name: str, records: list[dict], spec: str, head: list[str],
         rows: list[str], caption: str, label: str,
         size: str = r"\footnotesize", dtypes: dict[str, str] | None = None) -> None:
    """Write one table in both forms from the one set of records.

    `rows` must have been derived from `records`, never from the source CSVs a
    second time. That is the whole guarantee: there is no path by which the
    typeset table and the data file can carry different numbers.
    """
    os.makedirs(OUT_TEX, exist_ok=True)
    os.makedirs(OUT_CSV, exist_ok=True)

    tex_path = osp.join(OUT_TEX, f"{name}.tex")
    with open(tex_path, "w") as f:
        f.write(HEADER + table(spec, head, rows, caption, label, size).rstrip() + "\n")

    df = pd.DataFrame(records)
    if dtypes:
        df = df.astype(dtypes)
    csv_path = osp.join(OUT_CSV, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    WRITTEN[name] = {
        "tex": tex_path,
        "csv": csv_path,
        "n_tex": sum(1 for r in rows if not r.startswith(RULES)),
        "n_records": len(records),
        "n_cols": df.shape[1],
    }
    print(f"  wrote {osp.relpath(tex_path, REPO)}")
    print(f"  wrote {osp.relpath(csv_path, REPO)}")


def val(v: object) -> float | None:
    """A source cell as a number, or None so the CSV cell comes out empty."""
    return None if pd.isna(v) else float(v)


def text(v: object) -> str | None:
    return None if pd.isna(v) or str(v) == "" else str(v)


def tex_num(v: float | None) -> str:
    return "--" if v is None else f"{v:.4f}"


def tex_gene(rec: dict, key: str) -> str:
    """`YBR203W (COS111)` for the typeset table, from the two CSV columns."""
    name = rec.get(f"{key}_common")
    return f"{rec[key]} ({name})" if name else rec[key]


def pairs(triple: frozenset[str]) -> list[frozenset[str]]:
    return [frozenset(p) for p in itertools.combinations(sorted(triple), 2)]


def load():
    singles = pd.read_csv(SINGLES)
    doubles = pd.read_csv(DOUBLES)
    gaps = pd.read_csv(GAPS)
    sel = pd.read_csv(SELECTION)
    sel = sel[sel.strategy == "capped"].reset_index(drop=True)

    # ORF -> common name, taken from the measured singles table so there is one
    # definition of every name in play and none of it is retyped here.
    common = {r.orf: ("" if pd.isna(r.common) else str(r.common))
              for r in singles.itertuples()}

    built = {}
    for r in doubles.itertuples():
        g1, g2 = (x.strip() for x in r.pair.split("+"))
        built[frozenset((g1, g2))] = r.id
    return singles, doubles, gaps, sel, common, built


def derive(sel: pd.DataFrame, built: dict[frozenset[str], str]):
    """The 25 new doubles and the 20 triples, with their D and T ids."""
    triples = [(frozenset(t.split(" + ")), int(rk))
               for t, rk in zip(sel.triple, sel["rank"], strict=True)]
    assert len(triples) == 20, len(triples)

    new_doubles = {p for t, _ in triples for p in pairs(t)} - set(built)
    assert len(new_doubles) == 25, len(new_doubles)

    d_rows = []
    for i, p in enumerate(sorted(sorted(x) for x in new_doubles), start=1):
        d_rows.append({"id": f"D{i:02d}", "gene1": p[0], "gene2": p[1]})

    t_rows = []
    ordered = sorted(
        triples, key=lambda tr: (len([p for p in pairs(tr[0]) if p in built]), tr[1])
    )
    for i, (t, _rank) in enumerate(ordered, start=1):
        have = sorted((sorted(p) for p in pairs(t) if p in built))
        parent = frozenset(have[0])          # lexicographic first built parent
        third = sorted(t - parent)[0]
        t_rows.append({
            "id": f"T{i:02d}",
            "genes": sorted(t),
            "parent": parent,
            "third": third,
            "routes": len(have),
        })

    n_single_route = sum(1 for r in t_rows if r["routes"] == 1)
    assert n_single_route == 15, n_single_route
    assert [r["id"] for r in t_rows[:15]] == [f"T{i:02d}" for i in range(1, 16)]
    assert all(r["routes"] == 2 for r in t_rows[15:])
    return d_rows, t_rows, new_doubles


def plate_budget(n_singles: int, n_built_doubles: int, n_new_doubles: int,
                 n_triples: int):
    """Strain count and well budget, computed rather than asserted."""
    n_strains = n_singles + n_built_doubles + n_new_doubles + n_triples + 1
    free = PLATE_WELLS - ORIENTATION_BLANKS - WT_WELLS
    per_strain = free // n_strains
    non_wt = n_strains - 1
    spare = free - per_strain * non_wt
    return {
        "n_strains": n_strains,
        "free": free,
        "per_strain": per_strain,
        "non_wt": non_wt,
        "spare": spare,
    }


# --- t1 ----------------------------------------------------------------------
def t1(singles: pd.DataFrame) -> None:
    records = [
        {
            "id": r.id,
            "orf": r.orf,
            "common": text(r.common),
            "ours_fitness": val(r.fitness),
            "ours_boot_se": val(r.boot_se),
            "ours_plate_sd": val(r.across_plate_sd),
            "costanzo_smf": val(r.costanzo_smf),
            "costanzo_se": val(r.costanzo_se),
        }
        for r in singles.itertuples()
    ]
    head = [
        r"& & & \multicolumn{3}{c}{Ours, run 4} & \multicolumn{2}{c}{Costanzo 2016} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
        r"id & ORF & common & fitness & boot SE & plate SD & SMF & SE \\",
    ]
    rows = [
        " & ".join([
            rec["id"], rec["orf"], rec["common"] or "--",
            tex_num(rec["ours_fitness"]), tex_num(rec["ours_boot_se"]),
            tex_num(rec["ours_plate_sd"]), tex_num(rec["costanzo_smf"]),
            tex_num(rec["costanzo_se"]),
        ])
        for rec in records
    ]
    caption = (
        "Single knockouts that already exist, all twelve measured in run 4. "
        "The three columns under \\emph{Ours, run 4} are this lab's measurement; "
        "the two under \\emph{Costanzo 2016} are the published single-mutant "
        "fitness and its standard error."
    )
    emit("t1-existing-singles", records, "lllrrrrr", head, rows, caption,
         "tab:existing-singles")


# --- t2 ----------------------------------------------------------------------
def t2(doubles: pd.DataFrame, triples: list[dict],
       built: dict[frozenset[str], str]) -> None:
    parent_ids = {built[r["parent"]] for r in triples}
    records = []
    for r in doubles.itertuples():
        g1, g2 = (x.strip() for x in r.pair.split("+"))
        records.append({
            "id": r.id,
            "gene1": g1,
            "gene2": g2,
            "ours_fitness": val(r.fitness),
            "ours_boot_se": val(r.boot_se),
            "ours_plate_sd": val(r.across_plate_sd),
            "costanzo_dmf": val(r.costanzo_dmf),
            "costanzo_sd": val(r.costanzo_dmf_sd),
            "tier": r.tier,
            "is_parent": r.id in parent_ids,
        })
    head = [
        r"& & \multicolumn{3}{c}{Ours, run 4} & \multicolumn{2}{c}{Costanzo 2016} "
        r"& & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
        r"id & pair & fitness & boot SE & plate SD & DMF & SD & tier & parent \\",
    ]
    rows = [
        " & ".join([
            rec["id"], f"{rec['gene1']} + {rec['gene2']}",
            tex_num(rec["ours_fitness"]), tex_num(rec["ours_boot_se"]),
            tex_num(rec["ours_plate_sd"]), tex_num(rec["costanzo_dmf"]),
            tex_num(rec["costanzo_sd"]), rec["tier"],
            "yes" if rec["is_parent"] else "no",
        ])
        for rec in records
    ]
    caption = (
        "Double knockouts that already exist, all thirteen measured in run 4. "
        "Column attribution is as in Table~\\ref{tab:existing-singles}, with the "
        "published double-mutant fitness and its standard deviation under "
        "\\emph{Costanzo 2016}. \\emph{parent} says whether the pair is the "
        "starting strain for one of the triples in "
        "Table~\\ref{tab:new-triples}. A dash in the Costanzo columns means no one "
        "has published the pair."
    )
    emit("t2-existing-doubles", records, "llrrrrrll", head, rows, caption,
         "tab:existing-doubles")


# --- t3 ----------------------------------------------------------------------
def t3(d_rows: list[dict], common: dict[str, str]) -> None:
    records = [
        {
            "id": r["id"],
            "gene1": r["gene1"],
            "gene1_common": common.get(r["gene1"]) or None,
            "gene2": r["gene2"],
            "gene2_common": common.get(r["gene2"]) or None,
        }
        for r in d_rows
    ]
    # One strain per row, running down the page. The two-blocks-side-by-side form is
    # shorter on paper but it is read at the bench against a rack, where scanning a
    # single column of IDs beats saving half a page.
    head = [r"ID & gene 1 & gene 2 \\"]
    rows = [" & ".join([rec["id"], tex_gene(rec, "gene1"), tex_gene(rec, "gene2")])
            for rec in records]
    caption = (
        "The 25 double knockouts to construct, one per row. Each row names the two "
        "singles to cross, both of them in Table~\\ref{tab:existing-singles}."
    )
    emit("t3-new-doubles", records, "lll", head, rows, caption, "tab:new-doubles")


# --- t4 ----------------------------------------------------------------------
def t4(t_rows: list[dict], common: dict[str, str]) -> None:
    records = []
    for r in t_rows:
        g1, g2, g3 = r["genes"]
        p1, p2 = sorted(r["parent"])
        records.append({
            "id": r["id"],
            "gene1": g1, "gene1_common": common.get(g1) or None,
            "gene2": g2, "gene2_common": common.get(g2) or None,
            "gene3": g3, "gene3_common": common.get(g3) or None,
            "build_from_gene1": p1,
            "build_from_gene1_common": common.get(p1) or None,
            "build_from_gene2": p2,
            "build_from_gene2_common": common.get(p2) or None,
            "third_single": r["third"],
            "third_single_common": common.get(r["third"]) or None,
            "routes": r["routes"],
        })
    head = [
        r"ID & gene 1 & gene 2 & gene 3 & build from & $\times$ third single "
        r"& routes \\"
    ]
    rows = [
        " & ".join([
            rec["id"], tex_gene(rec, "gene1"), tex_gene(rec, "gene2"),
            tex_gene(rec, "gene3"),
            f"{rec['build_from_gene1']} + {rec['build_from_gene2']}",
            tex_gene(rec, "third_single"), str(rec["routes"]),
        ])
        for rec in records
    ]
    caption = (
        "The 20 triple knockouts to construct, in build order. "
        "\\emph{build from} names an existing double by ORF and "
        "\\emph{third single} the remaining gene; crossing the two gives the "
        "triple. \\emph{routes} counts how many different ways the bench can make "
        "the triple from strains that already exist, and is defined in "
        "Sec.~\\ref{sec:terms}."
    )
    emit("t4-new-triples", records, "lllllll", head, rows, caption,
         "tab:new-triples", size=r"\scriptsize")


# --- t5 ----------------------------------------------------------------------
def t5(budget: dict, n_singles: int, n_built: int, n_new: int,
       n_triples: int) -> None:
    per = budget["per_strain"]
    strain_wells = per * budget["non_wt"] + WT_WELLS
    # `kind` separates the five strain groups from the two reserved blocks and the
    # two aggregates, so a reader summing the `strains` column cannot double count.
    records = [
        {"category": "singles_existing", "kind": "strain_group",
         "label": "singles, already built", "strains": n_singles,
         "wells": n_singles * per},
        {"category": "doubles_existing", "kind": "strain_group",
         "label": "doubles, already built", "strains": n_built,
         "wells": n_built * per},
        {"category": "doubles_new", "kind": "strain_group",
         "label": "doubles, to construct", "strains": n_new, "wells": n_new * per},
        {"category": "triples_new", "kind": "strain_group",
         "label": "triples, to construct", "strains": n_triples,
         "wells": n_triples * per},
        {"category": "wild_type", "kind": "strain_group",
         "label": "wild type, BY4741", "strains": 1, "wells": WT_WELLS},
        {"category": "total_strains", "kind": "subtotal", "label": "total",
         "strains": budget["n_strains"], "wells": strain_wells},
        {"category": "orientation_blanks", "kind": "reserved",
         "label": "orientation blanks", "strains": None,
         "wells": ORIENTATION_BLANKS},
        {"category": "spare", "kind": "reserved", "label": "spare",
         "strains": None, "wells": budget["spare"]},
        {"category": "plate", "kind": "total", "label": "plate", "strains": None,
         "wells": PLATE_WELLS},
    ]
    # The typeset table cross-references the three build tables; the CSV keeps the
    # bare label, since a `\ref` is a rendering and would be noise in a data file.
    xref = {
        "doubles_existing": r"~(Table~\ref{tab:existing-doubles})",
        "doubles_new": r"~(Table~\ref{tab:new-doubles})",
        "triples_new": r"~(Table~\ref{tab:new-triples})",
    }
    head = [r"on the plate & strains & wells \\"]
    rows = []
    for rec in records:
        if rec["kind"] in ("subtotal", "total"):
            name = rf"\textbf{{{rec['label']}}}"
            strains = ("" if rec["strains"] is None
                       else rf"\textbf{{{rec['strains']}}}")
            wells = (rf"\textbf{{{rec['wells']}}}" if rec["kind"] == "total"
                     else str(rec["wells"]))
            rows.append(r"\midrule")
        else:
            name = rec["label"] + xref.get(rec["category"], "")
            strains = "--" if rec["strains"] is None else str(rec["strains"])
            wells = str(rec["wells"])
        rows.append(f"{name} & {strains} & {wells}")
    caption = (
        f"What goes on the measurement plate. {budget['n_strains']} strains at "
        f"{per} wells each, one 384-well layout, one pick per strain. Wells per "
        f"strain is the whole number of times {budget['n_strains']} strains divide "
        f"the {budget['free']} wells left after {ORIENTATION_BLANKS} are kept empty "
        f"to fix the plate's orientation and {WT_WELLS} are given to the wild-type "
        f"reference, which gives {per}. Every existing double is re-measured "
        f"alongside the new strains, at no cost in construction work or in wells "
        f"per strain."
    )
    emit("t5-plate", records, "lrr", head, rows, caption, "tab:plate",
         size=r"\small", dtypes={"strains": "Int64", "wells": "int64"})


def check_written(expected_rows: dict[str, int]) -> None:
    """Every .tex has its .csv, and the two agree on how many rows they hold.

    The CSV is re-read from disk rather than trusted from memory, so a write that
    silently produced nothing is caught here rather than by a reader next week.
    """
    print()
    for name, info in sorted(WRITTEN.items()):
        assert osp.exists(info["tex"]), f"{name}: no .tex written"
        assert osp.exists(info["csv"]), f"{name}: no .csv written"
        df = pd.read_csv(info["csv"])
        n_disk = len(df)
        assert n_disk == info["n_records"] == info["n_tex"], (
            f"{name}: {info['n_tex']} tex rows, {info['n_records']} records, "
            f"{n_disk} csv rows on disk"
        )
        assert df.shape[1] == info["n_cols"], name
        want = expected_rows[name]
        assert n_disk == want, f"{name}: expected {want} rows, wrote {n_disk}"
        print(f"  {name}: {n_disk} rows x {df.shape[1]} cols, tex and csv agree")


def main() -> None:
    singles, doubles, gaps, sel, common, built = load()
    d_rows, t_rows, new_doubles = derive(sel, built)

    n_singles = len({g for t in sel.triple for g in t.split(" + ")})
    budget = plate_budget(n_singles, len(built), len(new_doubles), len(t_rows))
    assert budget["n_strains"] == 70, budget
    assert budget["per_strain"] == 5, budget
    # SPH1 riding along costs no wells per strain.
    assert plate_budget(n_singles + 1, len(built), len(new_doubles),
                        len(t_rows))["per_strain"] == 5

    print("build list tables")
    t1(singles)
    t2(doubles, t_rows, built)
    t3(d_rows, common)
    t4(t_rows, common)
    t5(budget, n_singles, len(built), len(new_doubles), len(t_rows))

    check_written({
        "t1-existing-singles": len(singles),
        "t2-existing-doubles": len(doubles),
        "t3-new-doubles": len(new_doubles),
        "t4-new-triples": len(t_rows),
        "t5-plate": 9,
    })

    never_built = gaps[gaps.kind == "double, designed but never built"].strain.tolist()
    print()
    print(f"  existing: {len(singles)} singles, {len(doubles)} doubles, 0 triples")
    print(f"  to construct: {len(new_doubles)} doubles (D01-D{len(d_rows):02d}), "
          f"{len(t_rows)} triples (T01-T{len(t_rows):02d})")
    print(f"  routes 1: {sum(1 for r in t_rows if r['routes'] == 1)} triples, "
          f"routes 2: {sum(1 for r in t_rows if r['routes'] == 2)}")
    print(f"  plate: {n_singles} singles + {len(built) + len(new_doubles)} doubles "
          f"+ {len(t_rows)} triples + WT = {budget['n_strains']} strains, "
          f"{budget['per_strain']} wells each")
    print(f"  well budget: {PLATE_WELLS} - {ORIENTATION_BLANKS} blanks - {WT_WELLS} "
          f"WT = {budget['free']}; {budget['free']} // {budget['n_strains']} = "
          f"{budget['per_strain']}; spare {budget['spare']}")
    print(f"  designed but never built: {', '.join(never_built)}")


if __name__ == "__main__":
    main()
