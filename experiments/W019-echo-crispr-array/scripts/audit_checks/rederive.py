# experiments/W019-echo-crispr-array/scripts/verify_triple_build_list.py  (splice-in)
"""A5 adversarial re-derivation checks, group `rederive`.

Splices into `verify_triple_build_list.py`. Covers what the existing 61 checks do not:
input-file integrity of the k200 predictions, rank-tie determinism, whether the `capped`
fill rule is order-independent, the five non-capped rows of the strategy table, the two
note-prose lists that name the wrong IDs, the wells-per-strain arithmetic against the real
384 layout, and the "nothing extra" half of tau closure.

Uses the host module's DATA_ROOT-free constants and helpers: STRAINS, TARGETS, SELECTION,
SUMMARY, BUILD_LIST, RATIONALE, GENES, BLOCKED, NO_TRIGENIC_DATA, orf, pairs.

New constants, all defined inside run():
    WELLS       = 384    physical wells on the destination plate
    N_BLANKS    = 6      orientation-resolving blanks, from generate_picklist.py
    N_PLATE     = 65     strains on plate, closure + WT
    TIE_PRED    = 0.42041015625   the one tied prediction inside the 39-target basis
    STRAT_ZERO_DATA = corrected zero-data-triple counts for all six strategies
"""

from __future__ import annotations

import itertools
import re

import pandas as pd


def run(check) -> None:
    WELLS = 384
    N_BLANKS = 6
    N_PLATE = 65
    TIE_PRED = 0.42041015625
    STRAT_ZERO_DATA = {"rank": 17, "count": 15, "balanced": 15,
                       "uniform": 11, "capped": 6, "no_ylr": 9}

    # ---------------------------------------------------------------- shared re-derivation
    raw = pd.read_csv(TARGETS)  # noqa: F821
    raw_triples = [
        (frozenset((r.gene1.strip(), r.gene2.strip(), r.gene3.strip())),
         (r.gene1.strip(), r.gene2.strip(), r.gene3.strip()), float(r.prediction))
        for r in raw.itertuples()
    ]
    in_panel = [(s, p) for s, _, p in raw_triples if s <= GENES]  # noqa: F821
    # Python's sorted() is stable, so this is the file order broken only by prediction.
    basis = sorted(
        [(s, p) for s, p in in_panel if not (BLOCKED <= s)],  # noqa: F821
        key=lambda x: -x[1],
    )
    rank_of = {s: i for i, (s, _) in enumerate(basis, 1)}

    sheet = pd.read_csv(STRAINS)  # noqa: F821
    kind = sheet["#"].astype(str)
    built = {frozenset((orf(r.KO1), orf(r.KO2)))  # noqa: F821
             for r, m in zip(sheet.itertuples(), kind.str.match(r"d\d"), strict=True) if m}
    built_singles = {orf(r.KO1)  # noqa: F821
                     for r, m in zip(sheet.itertuples(), kind.str.match(r"s\d"), strict=True)
                     if m}

    sel_df = pd.read_csv(SELECTION)  # noqa: F821
    by_strategy = {
        m: [frozenset(t.split(" + ")) for t in g.triple]
        for m, g in sel_df.groupby("strategy")
    }
    sel = by_strategy["capped"]
    plate_doubles = {p for t in sel for p in pairs(t)}  # noqa: F821
    new_doubles = plate_doubles - built

    def has_parent(t):
        return any(p in built for p in pairs(t))  # noqa: F821

    bl = open(BUILD_LIST).read()  # noqa: F821
    rt = open(RATIONALE).read()  # noqa: F821

    def table_rows(text):
        out = []
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("|") and s.endswith("|") and set(s) - set("|:- "):
                out.append([c.strip() for c in s[1:-1].split("|")])
        return out

    d_id = {}
    for c in table_rows(bl):
        if len(c) == 6:
            for blk in (c[0:3], c[3:6]):
                if re.fullmatch(r"D\d\d\*?", blk[0]):
                    d_id[blk[0].rstrip("*")] = frozenset(
                        (orf(blk[1]), orf(blk[2])))  # noqa: F821

    # ---- A. prediction-file integrity ------------------------------------------------
    print("\nA. PREDICTION-FILE INTEGRITY")
    check("rederive", len(raw_triples) == 52,
          "the k200 target file holds 52 prediction rows", len(raw_triples))
    out_of_panel = sorted({g for _, t, _ in raw_triples for g in t} - GENES)  # noqa: F821
    check("rederive", out_of_panel == ["YIL174W"],
          "the only out-of-panel gene in the prediction file is YIL174W", out_of_panel)
    check("rederive", len(in_panel) == 39 and len(raw_triples) - len(in_panel) == 13,
          "39 rows are subsets of the eleven genes and 13 are dropped for YIL174W",
          f"{len(in_panel)} kept, {len(raw_triples) - len(in_panel)} dropped")
    check("rederive", len({s for s, _, _ in raw_triples}) == len(raw_triples),
          "no two prediction rows collapse to the same gene SET",
          len({s for s, _, _ in raw_triples}))
    check("rederive", all(len(s) == 3 for s, _, _ in raw_triples),
          "every prediction row names three distinct genes",
          sorted({len(s) for s, _, _ in raw_triples}))
    check("rederive", all(list(t) == sorted(t) for _, t, _ in raw_triples),
          "every prediction row is already in lexicographic gene order", "52 of 52")
    check("rederive", not [s for s, _ in in_panel if BLOCKED <= s],  # noqa: F821
          "the blocked-pair filter removes 0 rows, so it blocks 0 of the 39 targets", 0)

    # ---- B. rank determinism ----------------------------------------------------------
    print("\nB. RANK DETERMINISM")
    tied = [s for s, p in basis if p == TIE_PRED]
    check("rederive", len(tied) == 2 and sorted(rank_of[s] for s in tied) == [34, 35],
          "exactly one prediction tie inside the basis, and it decides ranks 34 and 35",
          sorted(rank_of[s] for s in tied))
    check("rederive",
          len({p for _, p in basis}) == 38,
          "the 39 basis targets carry 38 distinct prediction values",
          len({p for _, p in basis}))
    check("rederive",
          rank_of[frozenset(("YBR203W", "YGL087C", "YPL046C"))] == 34,
          "a STABLE sort puts YBR203W + YGL087C + YPL046C at rank 34, as the notes print",
          rank_of[frozenset(("YBR203W", "YGL087C", "YPL046C"))])
    # The tie is between a clean constructible target (always selected) and a flagged one
    # ranked below the cap, so no ordering of the tie can change the selection.
    tie_clean = [s for s in tied if not (s & NO_TRIGENIC_DATA)]  # noqa: F821
    check("rederive",
          len(tie_clean) == 1 and tie_clean[0] in sel
          and all(s not in sel for s in tied if s not in tie_clean),
          "the tie cannot change the selection: its clean member is always in, "
          "its flagged member always out", "selection invariant to the tie")

    # ---- C. where the flagged block ends ----------------------------------------------
    print("\nC. ZERO-DATA BLOCK EXTENT")
    first_clean = min(rank_of[s] for s, _ in basis if not (s & NO_TRIGENIC_DATA))  # noqa: F821
    check("rederive", first_clean == 12,
          "ranks 1-11 ALL touch a zero-data gene; the first clean target is rank 12",
          f"first clean rank {first_clean}")
    check("rederive", "ranks 1--10" not in rt and "ranks 1-10" not in rt,
          "the rationale no longer understates the flagged block as ranks 1-10",
          "corrected to 1-11")

    # ---- D. inventory hazards the existing pass does not cover -------------------------
    print("\nD. INVENTORY HAZARDS")
    check("rederive", (kind == "WT").any(),
          "the order sheet carries a WT row, the reference every fitness is divided by",
          "WT present")
    cells = [str(c) for c in sheet["KO1"].tolist() + sheet["KO2"].tolist()
             if isinstance(c, str) and c.strip() not in ("", "'-")]
    check("rederive", all(re.fullmatch(r"Y[A-P][LR]\d{3}[WC](-[A-Z])?", orf(c))  # noqa: F821
                          for c in cells),
          "every KO cell parses to a well-formed systematic ORF", f"{len(cells)} cells")
    n_single_rows = int(kind.str.match(r"s\d").sum())
    n_double_rows = int(kind.str.match(r"d\d").sum())
    check("rederive",
          len(built_singles) == n_single_rows and len(built) == n_double_rows,
          "no duplicated single ORF and no duplicated double pair in the sheet",
          f"{n_single_rows} single rows, {n_double_rows} double rows")
    check("rederive", not [p for p in built if len(p) != 2],
          "no built double names the same ORF twice", "13 of 13 distinct pairs")
    no_dbl = sorted(GENES - {g for p in built for g in p})  # noqa: F821
    check("rederive", no_dbl == ["YKL033W-A", "YLR104W"],
          "TWO panel genes have zero built doubles: YKL033W-A and YLR104W", no_dbl)

    # ---- E. capped determinism ---------------------------------------------------------
    print("\nE. CAPPED DETERMINISM")
    ranked = [(s, p) for s, p in basis if has_parent(s)]
    flagged_pool = [x for x in ranked if x[0] & NO_TRIGENIC_DATA]  # noqa: F821
    clean_pool = [x for x in ranked if not (x[0] & NO_TRIGENIC_DATA)]  # noqa: F821
    check("rederive", len(clean_pool) == 14,
          "exactly 14 constructible clean targets exist, so the 14-triple fill is forced",
          len(clean_pool))
    head = [s for s, _ in flagged_pool[:6]]
    check("rederive", sorted(rank_of[s] for s in head) == [1, 2, 3, 4, 5, 6],
          "the 6 capped flagged slots are ranks 1-6, all distinct in prediction",
          sorted(rank_of[s] for s in head))
    # rank-order fill vs the script's fewest-new-doubles fill
    rank_fill = head + [s for s, _ in clean_pool[:14]]
    have = {p for s in head for p in pairs(s)} | set(built)  # noqa: F821
    cost_fill, pool = list(head), list(clean_pool)
    while len(cost_fill) < 20 and pool:
        best = min(pool, key=lambda d: (
            len([p for p in pairs(d[0]) if p not in have]), -d[1]))  # noqa: F821
        have |= set(pairs(best[0]))  # noqa: F821
        cost_fill.append(best[0])
        pool.remove(best)
    check("rederive", set(rank_fill) == set(cost_fill) == set(sel),
          "rank-order fill and fewest-new-doubles fill give the IDENTICAL 20, "
          "so `capped` is order-independent here", "3-way identical")
    nd_rank = {p for s in rank_fill for p in pairs(s)} - built  # noqa: F821
    check("rederive", len(nd_rank) == len(new_doubles) == 25,
          "both fills cost the same 25 new doubles",
          f"rank fill {len(nd_rank)}, shipped {len(new_doubles)}")
    check("rederive",
          min(5, len(flagged_pool)) + len(clean_pool) == 19
          and min(6, len(flagged_pool)) + len(clean_pool) == 20,
          "cap 5 tops out at 19 parallel triples; cap 6 reaches exactly 20",
          f"cap5={min(5, len(flagged_pool)) + len(clean_pool)}, "
          f"cap6={min(6, len(flagged_pool)) + len(clean_pool)}")

    # ---- F. the five non-capped strategy rows ------------------------------------------
    print("\nF. STRATEGY TABLE, ALL SIX ROWS")
    obs_zero = {m: sum(1 for t in v if t & NO_TRIGENIC_DATA)  # noqa: F821
                for m, v in by_strategy.items()}
    check("rederive", obs_zero == STRAT_ZERO_DATA,
          "zero-data triple counts per strategy are rank 17, count 15, balanced 15, "
          "uniform 11, capped 6, no_ylr 9", obs_zero)
    check("rederive", obs_zero["no_ylr"] == 9,
          "no_ylr eliminates YLR312C-B but still puts 9 of 20 triples on YER079W, "
          "so it is NOT a zero-extrapolation design", obs_zero["no_ylr"])
    serial = {m: sorted(rank_of[t] for t in v if not has_parent(t))
              for m, v in by_strategy.items()}
    check("rederive", serial["no_ylr"] == [26]
          and all(not v for m, v in serial.items() if m != "no_ylr"),
          "no_ylr alone selects a triple with no built parent (rank 26), so its 40-strain "
          "cost is not a one-wave build", serial)
    # every strategy row of the note table, recomputed from the selection CSV
    strat_tbl = {}
    for c in table_rows(rt):
        if len(c) == 11 and c[0].strip("* ") in STRAT_ZERO_DATA:
            strat_tbl[c[0].strip("* ")] = [x.strip("* ") for x in c]
    check("rederive", len(strat_tbl) == 6,
          "the rationale strategy table parses to six strategy rows", sorted(strat_tbl))
    bad = []
    for m, v in by_strategy.items():
        genes = {g for t in v for g in t}
        dbl = {p for t in v for p in pairs(t)}  # noqa: F821
        want = [str(len(v)), str(len(dbl - built)), str(len(dbl - built) + len(v)),
                str(len(genes) + len(dbl) + len(v) + 1)]
        if strat_tbl[m][1:5] != want:
            bad.append((m, strat_tbl[m][1:5], want))
    check("rederive", not bad,
          "every strategy row's tri / new-dbl / construct / measure recomputes",
          bad or "6 of 6")
    bad = [(m, strat_tbl[m][8], str(STRAT_ZERO_DATA[m])) for m in STRAT_ZERO_DATA
           if strat_tbl[m][8] != str(STRAT_ZERO_DATA[m])]
    check("rederive", not bad,
          "the strategy table's zero-data column carries a real number for all six rows",
          bad or "6 of 6")
    bad = [(m, strat_tbl[m][9]) for m in STRAT_ZERO_DATA
           if strat_tbl[m][9] != ("no" if m == "no_ylr" else "yes")]
    check("rederive", not bad,
          "the strategy table's one-wave column marks no_ylr as the only non-parallel "
          "build", bad or "6 of 6")

    # ---- G. note prose that names IDs --------------------------------------------------
    print("\nG. NOTE PROSE ID LISTS")
    truly_ylr = sorted(i for i, p in d_id.items() if "YLR104W" in p)
    m = re.search(r"pair with YLR104W \(([^)]*)\)", bl)
    listed = sorted(x.strip() for x in m.group(1).split(",")) if m else []
    truly_ykl = sorted(i for i, p in d_id.items() if "YKL033W-A" in p)
    m2 = re.search(r"six with YKL033W-A\s*\n?\s*\(([^)]*)\)", bl)
    listed_ykl = sorted(x.strip() for x in m2.group(1).split(",")) if m2 else []
    check("rederive", listed_ykl == truly_ykl == ["D04", "D09", "D18", "D19", "D20", "D21"],
          "the YKL033W-A bullet names exactly the six new doubles that contain it, the "
          "second panel gene with no built double",
          f"note {listed_ykl} vs computed {truly_ykl}")
    check("rederive", listed == truly_ylr == ["D06", "D11", "D17", "D23", "D24", "D25"],
          "the YLR104W bullet names exactly the six new doubles that contain YLR104W",
          f"note {listed} vs computed {truly_ylr}")
    truly_kl = sorted(i for i, p in d_id.items() if "YKL033W-A" in p)
    check("rederive", truly_kl == ["D04", "D09", "D18", "D19", "D20", "D21"],
          "YKL033W-A likewise gains six new doubles, its first coverage after the "
          "failed cross", truly_kl)
    truly_312 = sorted(i for i, p in d_id.items() if "YLR312C-B" in p)
    m = re.search(r"It appears in (D\d\d(?:, D\d\d)*)", bl)
    check("rederive", m is not None
          and sorted(x.strip() for x in m.group(1).split(",")) == truly_312,
          "the YLR312C-B bullet names exactly its five new doubles", truly_312)
    clean_orphan = sorted(rank_of[s] for s, _ in basis
                          if not (s & NO_TRIGENIC_DATA) and not has_parent(s))  # noqa: F821
    check("rederive", clean_orphan == [32, 37],
          "the CLEAN shortfall is 2 targets, ranks 32 and 37; rank 26 is flagged, "
          "not clean", clean_orphan)
    check("rederive", "ranks 26, 32 and 37 have none" not in rt,
          "the rationale no longer lists rank 26 among the clean-16 shortfall",
          "corrected")

    # ---- H. wells and tubes ------------------------------------------------------------
    print("\nH. PLATE ARITHMETIC")
    check("rederive", N_PLATE * 6 > WELLS and WELLS // N_PLATE == 5,
          "5 wells per strain is FORCED: 65 x 6 = 390 exceeds 384 with any reserve",
          f"{N_PLATE * 6} > {WELLS}")
    used = (N_PLATE - 1) * 5 + N_BLANKS
    check("rederive", used <= WELLS,
          "64 non-WT strains at 5 wells plus the 6 orientation blanks fit on one plate",
          f"{used} of {WELLS} wells, {WELLS - used} left for WT")
    check("rederive", WELLS - used == 58,
          "the note allocates 5 wells to each of the 64 non-WT strains and is silent on "
          "the remaining 58, which are the WT block", WELLS - used)
    summ = pd.read_csv(SUMMARY).set_index("strategy")  # noqa: F821
    check("rederive", int(summ.loc["capped", "tubes_1pick"]) == N_PLATE,
          "one pick per strain is 65 tubes", int(summ.loc["capped", "tubes_1pick"]))
    check("rederive", "135 at two" not in rt,
          "two picks per strain is 2 x 65 = 130 tubes, not 135", 2 * N_PLATE)

    # ---- I. tau closure, the 'nothing extra' half --------------------------------------
    print("\nI. TAU CLOSURE, NOTHING EXTRA")
    closure = {("S", g) for t in sel for g in t} \
        | {("D", p) for p in plate_doubles} | {("T", t) for t in sel}
    check("rederive", len(closure) + 1 == N_PLATE,
          "the closure of the 20 taus is exactly 64 strains, plus WT = 65", len(closure) + 1)
    unused = built - plate_doubles
    check("rederive", len(unused) == 5,
          "5 of the 13 existing doubles serve no selected tau and are correctly OFF plate",
          sorted(sorted(p) for p in unused))
    check("rederive", "YLR313C" not in {g for t in sel for g in t},
          "YLR313C is on no plate strain, so the 12th built single is correctly excluded",
          "excluded")
    check("rederive",
          all(len([p for p in pairs(t) if p in plate_doubles]) == 3  # noqa: F821
              and len(t & built_singles) == 3 for t in sel),
          "all 6 supporting terms of every tau are drawn from the SAME 65-strain plate",
          "20 of 20")
    surv = [t for t in sel if not (t & NO_TRIGENIC_DATA)]  # noqa: F821
    check("rederive", len({g for t in surv for g in t}) == 9,
          "the drop-the-two contingency needs 9 singles, not 11",
          len({g for t in surv for g in t}))
    surv_dbl = {p for t in surv for p in pairs(t)}  # noqa: F821
    check("rederive", len(surv_dbl) == 23 and 9 + 23 + 14 + 1 == 47,
          "the contingency plate is 9 singles + 23 doubles + 14 triples + WT = 47, "
          "not the 65 the note's plate table implies", len(surv_dbl) + 9 + 14 + 1)


if __name__ == "__main__":
    import os
    import os.path as osp

    from dotenv import load_dotenv

    load_dotenv()
    _EXP = osp.join(
        os.environ.get(
            "W019_ROOT",
            "/home/michaelvolk/Documents/projects/torchcell.worktrees/audit/"
            "w019-trigenic-round",
        ),
        "experiments/W019-echo-crispr-array",
    )
    _REPO = osp.dirname(osp.dirname(_EXP))
    g = globals()
    g["STRAINS"] = osp.join(
        _EXP, "data/run4_doubles_2026-08-06/Single-and-Double-KO-Strains-List-Order.csv")
    g["TARGETS"] = osp.join(
        _REPO, "experiments/010-kuzmin-tmi/results/inference_3",
        "top_k_constructible_panel12_k200.csv")
    g["SELECTION"] = osp.join(
        _EXP, "results/triple_design_rank_sampling_selection.csv")
    g["SUMMARY"] = osp.join(_EXP, "results/triple_design_rank_sampling_summary.csv")
    g["BUILD_LIST"] = osp.join(
        _REPO, "notes/experiments.W019-echo-crispr-array.build-list.md")
    g["RATIONALE"] = osp.join(
        _REPO, "notes/experiments.W019-echo-crispr-array.next-strains-to-construct.md")
    g["GENES"] = frozenset((
        "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W", "YKL033W-A",
        "YLL012W", "YLR104W", "YLR312C-B", "YPL046C", "YPL081W"))
    g["BLOCKED"] = frozenset(("YKL033W-A", "YJR060W"))
    g["NO_TRIGENIC_DATA"] = frozenset(("YER079W", "YLR312C-B"))
    g["orf"] = lambda cell: str(cell).split(" ")[0].strip()
    g["pairs"] = lambda t: [frozenset(p) for p in itertools.combinations(sorted(t), 2)]

    tally = {"n": 0, "fail": 0}

    def _check(group: str, ok: bool, claim: str, observed: object) -> None:
        tally["n"] += 1
        tally["fail"] += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {claim}  [{observed}]")

    run(_check)
    print(f"\n{tally['n']} rederive checks, {tally['fail']} FAIL")
