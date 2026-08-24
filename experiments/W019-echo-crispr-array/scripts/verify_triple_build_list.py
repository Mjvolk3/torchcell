# experiments/W019-echo-crispr-array/scripts/verify_triple_build_list.py
# [[experiments.W019-echo-crispr-array.scripts.verify_triple_build_list]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/verify_triple_build_list
"""Independent audit of the capped triple build list.

`triple_design_rank_sampling.py` CHOOSES the round; this script CHECKS it. It does not
import that script -- it re-derives the strain inventory, the target basis and the
selection consequences from the pinned inputs, then asserts every number the two bench
notes state.

The load-bearing check is tau-computability. A trigenic interaction

    tau_abc = f_abc - f_ab f_c - f_ac f_b - f_bc f_a + 2 f_a f_b f_c

consumes seven measured fitnesses, so a triple is worthless unless ALL THREE of its
doubles and ALL THREE of its singles are on the same plate. Selecting 20 triples whose
doubles are not all present would produce 20 fitness numbers and zero tau values. Check 5
is what rules that out.

The Kuzmin pass counts per-gene records directly from the LMDBs to confirm which panel
genes the model could have learned anything about. It reads ~1.4M records and takes a
couple of minutes.

Outputs
  results/verify_triple_build_list_checks.csv     one row per assertion
  results/verify_triple_build_list_kuzmin.csv     per-gene trigenic + digenic counts

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/verify_triple_build_list.py
"""

from __future__ import annotations

import itertools
import os
import os.path as osp
import pickle
import re
import sys

import lmdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
REPO = osp.dirname(osp.dirname(EXP_DIR))
RESULTS = osp.join(EXP_DIR, "results")
NOTES = osp.join(REPO, "notes")

STRAINS = osp.join(
    EXP_DIR, "data/run4_doubles_2026-08-06/Single-and-Double-KO-Strains-List-Order.csv"
)
TARGETS = osp.join(
    REPO, "experiments/010-kuzmin-tmi/results/inference_3",
    "top_k_constructible_panel12_k200.csv",
)
SELECTION = osp.join(RESULTS, "triple_design_rank_sampling_selection.csv")
SUMMARY = osp.join(RESULTS, "triple_design_rank_sampling_summary.csv")
CONSTRUCTION = osp.join(RESULTS, "triple_build_construction_check.csv")
BUILD_LIST = osp.join(NOTES, "experiments.W019-echo-crispr-array.build-list.md")
RATIONALE = osp.join(NOTES, "experiments.W019-echo-crispr-array.next-strains-to-construct.md")

GENES = frozenset((
    "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W", "YKL033W-A",
    "YLL012W", "YLR104W", "YLR312C-B", "YPL046C", "YPL081W",
))
BLOCKED = frozenset(("YKL033W-A", "YJR060W"))
NO_TRIGENIC_DATA = frozenset(("YER079W", "YLR312C-B"))
KUZMIN_ORDER = [
    "YLL012W", "YBR203W", "YPL046C", "YJR060W", "YGL087C", "YPL081W",
    "YDR057W", "YLR104W", "YKL033W-A", "YER079W", "YLR312C-B",
]

CHECKS: list[dict] = []


def check(group: str, ok: bool, claim: str, observed: object) -> None:
    CHECKS.append({"group": group, "pass": bool(ok), "claim": claim,
                   "observed": str(observed)})
    print(f"  {'PASS' if ok else 'FAIL'}  {claim}  [{observed}]")


def orf(cell: object) -> str:
    return str(cell).split(" ")[0].strip()


def pairs(triple):
    return [frozenset(p) for p in itertools.combinations(sorted(triple), 2)]


def read_inventory():
    sheet = pd.read_csv(STRAINS)
    kind = sheet["#"].astype(str)
    singles = {orf(r.KO1)
               for r, m in zip(sheet.itertuples(), kind.str.match(r"s\d"), strict=True) if m}
    doubles = {frozenset((orf(r.KO1), orf(r.KO2))): str(r[1])
               for r, m in zip(sheet.itertuples(), kind.str.match(r"d\d"), strict=True) if m}
    triples = [r for r, m in zip(sheet.itertuples(), kind.str.match(r"t\d"), strict=True) if m]
    return singles, doubles, triples


def read_targets():
    df = pd.read_csv(TARGETS).sort_values("prediction", ascending=False)
    out = []
    for r in df.itertuples():
        t = frozenset((r.gene1, r.gene2, r.gene3))
        if t <= GENES and not (BLOCKED <= t):
            out.append({"triple": t, "prediction": r.prediction})
    for i, d in enumerate(out, start=1):
        d["rank"] = i
    return out


def kuzmin_counts():
    """Per-gene record counts straight from the LMDBs, split by perturbation order."""
    counts = {g: {"trigenic": 0, "digenic": 0} for g in KUZMIN_ORDER}
    totals = {}
    for name, col in (("tmi_kuzmin2018", "trigenic"), ("tmi_kuzmin2020", "trigenic"),
                      ("dmi_kuzmin2018", "digenic"), ("dmi_kuzmin2020", "digenic")):
        path = osp.join(DATA_ROOT, "data/torchcell", name, "processed/lmdb")
        env = lmdb.open(path, readonly=True, lock=False, subdir=True)
        n = 0
        with env.begin() as tx:
            for _, v in tx.cursor():
                n += 1
                rec = pickle.loads(v)
                genes = {p["systematic_gene_name"]
                         for p in rec["experiment"]["genotype"]["perturbations"]}
                for g in genes & set(KUZMIN_ORDER):
                    counts[g][col] += 1
        env.close()
        totals[name] = n
        print(f"    {name}: {n} records")
    return counts, totals


def main() -> None:
    singles, doubles, built_triples = read_inventory()
    built = set(doubles)
    targets = read_targets()
    rank_of = {d["triple"]: d["rank"] for d in targets}
    pred_of = {d["triple"]: d["prediction"] for d in targets}

    sel_df = pd.read_csv(SELECTION)
    sel = [frozenset(s.split(" + ")) for s in sel_df[sel_df.strategy == "capped"].triple]

    new_doubles = {p for t in sel for p in pairs(t)} - built
    plate_doubles = {p for t in sel for p in pairs(t)}
    reused = plate_doubles & built

    # ---- 1. inventory ----------------------------------------------------------------
    print("\n1. INVENTORY")
    check("inventory", len(singles) == 12, "12 singles built", len(singles))
    check("inventory", len(built) == 13, "13 doubles built", len(built))
    check("inventory", len(built_triples) == 0, "0 triples built", len(built_triples))
    check("inventory", GENES <= singles, "all 11 prediction genes are built singles",
          sorted(GENES - singles) or "none missing")
    check("inventory", singles - GENES == {"YLR313C"},
          "the one built non-node single is YLR313C (SPH1)", sorted(singles - GENES))
    check("inventory", BLOCKED not in built,
          "the blocked pair YKL033W-A x YJR060W is not built", "absent")
    check("inventory", not [p for p in built if "YLR104W" in p],
          "YLR104W (LCL2) has zero built doubles", 0)
    check("inventory", not [p for p in built if not set(p) <= singles],
          "every built double has both parent singles built", "all 13")

    # ---- 2. target basis --------------------------------------------------------------
    print("\n2. TARGET BASIS")
    check("basis", len(targets) == 39, "39 in-basis targets", len(targets))
    check("basis", all(d["triple"] & NO_TRIGENIC_DATA for d in targets if d["rank"] <= 10),
          "ranks 1-10 all touch a zero-trigenic-data gene", "10 of 10")
    orphan = sorted(d["rank"] for d in targets
                    if not any(p in built for p in pairs(d["triple"])))
    check("basis", orphan == [26, 32, 37], "targets with no built parent are 26, 32, 37",
          orphan)
    check("basis", all("YLR104W" in d["triple"] for d in targets if d["rank"] in orphan),
          "every no-parent target contains YLR104W", "3 of 3")
    within_ten = [d for d in targets if "YLR104W" not in d["triple"]]
    check("basis",
          len(within_ten) == 31
          and all(any(p in built for p in pairs(d["triple"])) for d in within_ten),
          "set-cover guarantee: 31 of 31 within-TEN targets have a built parent",
          len(within_ten))
    clean = [d for d in targets if not (d["triple"] & NO_TRIGENIC_DATA)]
    check("basis", len(clean) == 16, "16 of 39 targets are clean", len(clean))
    check("basis",
          len([d for d in clean if any(p in built for p in pairs(d["triple"]))]) == 14,
          "14 clean targets have a built parent",
          len([d for d in clean if any(p in built for p in pairs(d["triple"]))]))

    # ---- 3. selection -----------------------------------------------------------------
    print("\n3. THE CAPPED SELECTION")
    check("selection", len(sel) == 20 == len(set(sel)), "20 distinct triples", len(sel))
    check("selection", all(t in rank_of for t in sel),
          "every selected triple is one of the 39 targets", "20 of 20")
    check("selection", sum(1 for t in sel if t & NO_TRIGENIC_DATA) == 6,
          "exactly 6 triples touch a zero-data gene (the cap)",
          sum(1 for t in sel if t & NO_TRIGENIC_DATA))
    check("selection", all(any(p in built for p in pairs(t)) for t in sel),
          "every triple has >=1 built parent double, so the wave is parallel", "20 of 20")
    single_route = [t for t in sel if len([p for p in pairs(t) if p in built]) == 1]
    check("selection", len(single_route) == 15,
          "15 triples have exactly one parent route (build these first)",
          len(single_route))

    # ---- 4. new doubles ---------------------------------------------------------------
    print("\n4. DOUBLES")
    check("doubles", len(new_doubles) == 25, "25 new doubles", len(new_doubles))
    check("doubles", len(reused) == 8, "8 of the 13 existing doubles are re-measured",
          len(reused))
    check("doubles", len(plate_doubles) == 33, "33 distinct doubles on plate",
          len(plate_doubles))
    check("doubles", BLOCKED not in new_doubles,
          "the blocked pair is not among the new doubles", "absent")
    check("doubles", not [p for p in new_doubles if not set(p) <= singles],
          "every new double has both parent singles built", "25 of 25")

    # ---- 5. tau computability ---------------------------------------------------------
    print("\n5. TAU COMPUTABILITY")
    have = built | new_doubles
    unscorable = [sorted(t) for t in sel
                  if [p for p in pairs(t) if p not in have] or not t <= singles]
    check("tau", not unscorable,
          "after the build every triple has all 3 doubles and all 3 singles measured",
          unscorable or "20 of 20 scorable")
    closure = {g for t in sel for g in t} | plate_doubles | set(sel)
    check("tau", len({g for t in sel for g in t}) == 11,
          "the 11 plate singles are exactly the 11 basis genes",
          len({g for t in sel for g in t}))
    check("tau", len(closure) + 1 == 65,
          "plate = 11 singles + 33 doubles + 20 triples + WT = 65", len(closure) + 1)
    check("tau", len(new_doubles) + len(sel) == 45, "to construct = 45",
          len(new_doubles) + len(sel))
    check("tau", (378 - 28) // (len(closure)) == 5, "5 wells per strain",
          (378 - 28) // len(closure))

    # ---- 6. drop-the-two contingency --------------------------------------------------
    print("\n6. DROP-THE-TWO CONTINGENCY")
    surv_t = [t for t in sel if not (t & NO_TRIGENIC_DATA)]
    surv_d = {p for t in surv_t for p in pairs(t)} - built
    clean_d = {p for p in new_doubles if not (p & NO_TRIGENIC_DATA)}
    check("contingency", len(surv_t) == 14, "14 triples survive", len(surv_t))
    check("contingency", len(surv_d) == 17, "17 new doubles are still needed", len(surv_d))
    check("contingency", len(clean_d) == 18,
          "18 new doubles contain neither flagged gene", len(clean_d))
    check("contingency", {frozenset(("YLR104W", "YPL081W"))} == clean_d - surv_d,
          "YLR104W + YPL081W is the one clean double orphaned by the drop",
          [sorted(p) for p in clean_d - surv_d])
    check("contingency", len([p for p in new_doubles if p & NO_TRIGENIC_DATA]) == 7,
          "7 of 25 new doubles touch a zero-data gene",
          len([p for p in new_doubles if p & NO_TRIGENIC_DATA]))

    # ---- 7. notes vs computed ---------------------------------------------------------
    print("\n7. NOTE TABLES vs COMPUTED")
    bl = open(BUILD_LIST).read()
    rt = open(RATIONALE).read()

    # Tables are parsed by splitting rows on the pipe, not by regex over the whole file:
    # the D table is two ID blocks per row and both tables carry the zero-data flag as a
    # `*` suffix on the ID, neither of which a single-pass regex reads reliably.
    def table_rows(text):
        rows = []
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("|") and s.endswith("|") and set(s) - set("|:- "):
                rows.append([c.strip() for c in s[1:-1].split("|")])
        return rows

    def split_id(cell):
        return (cell[:-1], True) if cell.endswith("*") else (cell, False)

    d_rows, t_rows = [], []
    for c in table_rows(bl):
        if len(c) == 6:
            for blk in (c[0:3], c[3:6]):
                if re.fullmatch(r"D\d\d\*?", blk[0]):
                    did, star = split_id(blk[0])
                    d_rows.append((did, star, frozenset((orf(blk[1]), orf(blk[2])))))
        elif len(c) == 7 and re.fullmatch(r"T\d\d\*?", c[0]):
            tid, star = split_id(c[0])
            t_rows.append((tid, star, c[1], c[2], c[3], c[4], c[5], c[6]))

    check("notes", len(d_rows) == 25, "build-list has 25 D-rows", len(d_rows))
    check("notes", {p for _, _, p in d_rows} == new_doubles,
          "build-list D-rows equal the computed new doubles", "identical sets")
    d_marked = {i for i, s, _ in d_rows if s}
    d_truly = {i for i, _, p in d_rows if p & NO_TRIGENIC_DATA}
    check("notes", d_marked == d_truly
          and d_marked == {"D07", "D10", "D11", "D13", "D19", "D22", "D23"},
          "D stars mark exactly the 7 zero-data doubles", sorted(d_marked))

    check("notes", len(t_rows) == 20, "build-list has 20 T-rows", len(t_rows))
    check("notes",
          {frozenset((orf(a), orf(b), orf(c))) for _, _, a, b, c, _, _, _ in t_rows} == set(sel),
          "build-list T-rows equal the capped selection", "identical sets")
    bad = []
    for tid, _, a, b, c, bfrom, third, routes in t_rows:
        t = frozenset((orf(a), orf(b), orf(c)))
        parent = frozenset(orf(x) for x in bfrom.split(" + "))
        if parent not in built or not parent < t:
            bad.append(("parent", tid))
        if orf(third) != sorted(t - parent)[0]:
            bad.append(("third", tid))
        if routes != str(len([p for p in pairs(t) if p in built])):
            bad.append(("routes", tid))
    check("notes", not bad,
          "every T-row build-from, third single and routes count is correct",
          bad or "20 of 20")

    t_marked = {i for i, s, *_ in t_rows if s}
    t_truly = {i for i, _, a, b, c, *_ in t_rows
               if frozenset((orf(a), orf(b), orf(c))) & NO_TRIGENIC_DATA}
    check("notes", t_marked == t_truly == {f"T0{i}" for i in range(1, 7)},
          "T stars mark exactly T01-T06, the zero-data triples", sorted(t_marked))

    tri_tbl = re.findall(r"\| (\d+) \| (0\.\d+) \| ([A-Z0-9\- +]+?) \| (\d) \|", rt)
    check("notes", len(tri_tbl) == 20, "rationale 20-triple table has 20 rows", len(tri_tbl))
    bad = []
    for rk, pr, trp, nd in tri_tbl:
        t = frozenset(trp.strip().split(" + "))
        if t not in set(sel) or int(rk) != rank_of[t] \
                or abs(float(pr) - pred_of[t]) > 5e-5 \
                or int(nd) != len([p for p in pairs(t) if p not in built]):
            bad.append(trp)
    check("notes", not bad, "every rank, prediction and new-double count is correct",
          bad or "20 of 20")

    dbl_tbl = re.findall(r"\| ([A-Z0-9\-]+ \+ [A-Z0-9\-]+) \| ([\d, ]+) \|", rt)
    check("notes", len(dbl_tbl) == 25, "rationale 25-double table has 25 rows", len(dbl_tbl))
    bad = []
    for dstr, serves in dbl_tbl:
        p = frozenset(dstr.split(" + "))
        claimed = {int(x) for x in serves.replace(" ", "").split(",")}
        if p not in new_doubles or claimed != {rank_of[t] for t in sel if p in pairs(t)}:
            bad.append(dstr)
    check("notes", not bad, "every new double's serves-triples list is correct",
          bad or "25 of 25")

    # sorted so the CSV report is byte-stable: set iteration order is not
    freq = {}
    for t in sel:
        for g in sorted(t):
            freq[g] = freq.get(g, 0) + 1
    freq = dict(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])))
    line = re.search(r"\n(YBR203W \d+ .*?YER079W \d+)\n", rt).group(1)
    noted = {k: int(v) for k, v in re.findall(r"([A-Z0-9\-]+) (\d+)", line)}
    check("notes", noted == freq, "gene-participation line matches", freq)
    check("notes", len(freq) == 11 and min(freq.values()) == 2 and max(freq.values()) == 8,
          "all 11 genes covered, range 2-8",
          f"{len(freq)} genes, {min(freq.values())}-{max(freq.values())}")

    summ = pd.read_csv(SUMMARY).set_index("strategy").loc["capped"]
    check("notes",
          (int(summ.triples), int(summ.new_doubles), int(summ.construct_total),
           int(summ.measure_plus_wt), int(summ.wells_per_strain)) == (20, 25, 45, 65, 5)
          and abs(float(summ.mean_prediction) - 0.5300) < 5e-5,
          "summary CSV capped row matches the note",
          f"{int(summ.construct_total)} construct, {float(summ.mean_prediction):.4f} mean pred")

    cc = pd.read_csv(CONSTRUCTION)
    cc = cc[cc.strategy == "capped"]
    check("notes", len(cc) == 20 and (cc.built_parents >= 1).all()
          and int((cc.built_parents == 1).sum()) == 15,
          "construction-check CSV agrees: 20 rows, all with a parent, 15 single-route",
          f"{len(cc)} rows, {int((cc.built_parents == 1).sum())} single-route")

    # ---- 8. Kuzmin training coverage --------------------------------------------------
    print("\n8. KUZMIN TRAINING COVERAGE (reading LMDBs)")
    counts, totals = kuzmin_counts()
    kdf = pd.DataFrame(
        [{"gene": g, "trigenic": counts[g]["trigenic"], "digenic": counts[g]["digenic"]}
         for g in KUZMIN_ORDER])
    kdf.to_csv(osp.join(RESULTS, "verify_triple_build_list_kuzmin.csv"), index=False)
    print(kdf.to_string(index=False))
    zero = {g for g in KUZMIN_ORDER if counts[g]["trigenic"] == 0}
    check("kuzmin", zero == set(NO_TRIGENIC_DATA),
          "YER079W and YLR312C-B are exactly the zero-trigenic-record genes", sorted(zero))
    check("kuzmin", {g for g in KUZMIN_ORDER if counts[g]["digenic"] == 0} == set(NO_TRIGENIC_DATA),
          "the same two genes are also absent from Kuzmin digenic",
          sorted({g for g in KUZMIN_ORDER if counts[g]["digenic"] == 0}))
    noted_tbl = dict(re.findall(r"\| (?:\*\*)?([A-Z0-9\-]+)(?:\*\*)?(?: \([A-Z0-9]+\))? \| (?:\*\*)?(\d+)(?:\*\*)? \|", rt))
    check("kuzmin",
          all(int(noted_tbl[g]) == counts[g]["trigenic"] for g in KUZMIN_ORDER if g in noted_tbl),
          "the note's trigenic column matches the LMDBs",
          f"{len(noted_tbl)} rows compared")

    # ---- report -----------------------------------------------------------------------
    rep = pd.DataFrame(CHECKS)
    rep.to_csv(osp.join(RESULTS, "verify_triple_build_list_checks.csv"), index=False)
    n_fail = int((~rep["pass"]).sum())
    print(f"\n{len(rep)} checks, {n_fail} FAIL")
    for r in rep[~rep["pass"]].itertuples():
        print(f"  FAIL [{r.group}] {r.claim} -> {r.observed}")
    print(f"wrote -> {RESULTS}/verify_triple_build_list_checks.csv")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
