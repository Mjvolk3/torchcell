# experiments/W019-echo-crispr-array/scripts/run4_measured_summary.py
# [[experiments.W019-echo-crispr-array.scripts.run4_measured_summary]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/run4_measured_summary
"""What run 4 actually measured for the strains that already exist, and what is missing.

The build list says which strains exist; this says what each one scored, and it is the
table the bench sheet pastes. Three things it is built to make unmissable:

  1. Coverage. 12 of 12 singles and 13 of 13 built doubles were measured. The 14th
     DESIGNED double was never built.
  2. Reference coverage, which is NOT the same thing. A strain can be measured here and
     still have nothing published to compare against, and two do.
  3. Fitness only. Run 4's epsilon is NOT reportable -- the round's denominator rests on
     one WT colony and the double/single ratio is 0.758 where a multiplicative model
     wants ~1.07 (see [[experiments.W019-echo-crispr-array.run4-handoff]]). Emitting eps
     next to fitness here would invite it to be read as a result, so it is left out and
     named instead.

Outputs
  results/run4_measured_summary_singles.csv
  results/run4_measured_summary_doubles.csv
  results/run4_measured_summary_gaps.csv     one row per missing thing, with the reason

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/run4_measured_summary.py
"""

from __future__ import annotations

import os.path as osp

import pandas as pd

EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
REPO = osp.dirname(osp.dirname(EXP_DIR))
RESULTS = osp.join(EXP_DIR, "results")

BOOTSTRAP = osp.join(RESULTS, "run4_strain_bootstrap.csv")
SINGLES_REF = osp.join(RESULTS, "reference_smf_12panel.csv")
DOUBLES_REF = osp.join(
    REPO, "experiments/010-kuzmin-tmi/results/construction_validation_doubles.csv"
)
STRAINS = osp.join(
    EXP_DIR, "data/run4_doubles_2026-08-06/Single-and-Double-KO-Strains-List-Order.csv"
)

COMMON = {"YBR203W": "COS111", "YDR057W": "YOS9", "YGL087C": "MMS2", "YJR060W": "CBF1",
          "YLL012W": "YEH1", "YLR104W": "LCL2", "YLR313C": "SPH1", "YPL046C": "ELC1",
          "YPL081W": "RPS9A"}
MEASURED_ROUND = "run 4"


def orf(cell: object) -> str:
    return str(cell).split(" ")[0].strip()


def read_order_sheet():
    sheet = pd.read_csv(STRAINS)
    kind = sheet["#"].astype(str)
    singles, doubles = [], []
    for r, is_s, is_d in zip(sheet.itertuples(), kind.str.match(r"s\d"),
                             kind.str.match(r"d\d"), strict=True):
        if is_s:
            singles.append((str(r[1]), orf(r.KO1)))
        if is_d:
            doubles.append((str(r[1]), orf(r.KO1), orf(r.KO2)))
    return singles, doubles


def main() -> None:
    boot = pd.read_csv(BOOTSTRAP).set_index("strain")
    sref = pd.read_csv(SINGLES_REF).set_index("orf")
    # keyed by unordered pair; a frozenset is not usable as a pandas index label
    dref = {frozenset((r.gene1, r.gene2)): r for r in pd.read_csv(DOUBLES_REF).itertuples()}
    singles, doubles = read_order_sheet()

    gaps = []

    srows = []
    for sid, g in singles:
        m = boot.loc[g]
        # Every run-4 single has a published Costanzo SMF; build_reference_smf.py asserts
        # the panel covers the plate, so a KeyError here means they have diverged again.
        srows.append({
            "id": sid, "orf": g, "common": COMMON.get(g, ""),
            "n_plates": int(m.n_plates), "fitness": round(float(m.fitness), 4),
            "boot_se": round(float(m.boot_se), 4),
            "across_plate_sd": round(float(m.across_plate_sd), 4),
            "costanzo_smf": round(float(sref.loc[g, "costanzo_smf"]), 4),
            "costanzo_se": round(float(sref.loc[g, "costanzo_se"]), 4),
        })

    drows = []
    for did, a, b in doubles:
        key = frozenset((a, b))
        name = f"{a}+{b}"
        m = boot.loc[name] if name in boot.index else boot.loc[f"{b}+{a}"]
        r = dref[key]
        ref = r.DmfCostanzo2016_fitness
        drows.append({
            "id": did, "pair": " + ".join(sorted((a, b))),
            "n_plates": int(m.n_plates), "fitness": round(float(m.fitness), 4),
            "boot_se": round(float(m.boot_se), 4),
            "across_plate_sd": round(float(m.across_plate_sd), 4),
            "tier": r.tier,
            "costanzo_dmf": None if pd.isna(ref) else round(float(ref), 4),
            # Costanzo's own uncertainty on the double, carried alongside the value so a
            # reader is never left comparing our error bar against a bare number. This is
            # their colony-level sample SD (`DmfCostanzo2016_std`), NOT the `se` column of
            # the same file, which is that SD divided by the colony count and is what
            # their epsilon p-values are built on. Different statistics, so the column is
            # named for the one it actually holds.
            "costanzo_dmf_sd": (None if pd.isna(r.DmfCostanzo2016_std)
                                else round(float(r.DmfCostanzo2016_std), 4)),
        })
        if pd.isna(ref):
            gaps.append({
                "kind": "double, no published reference", "strain": " + ".join(sorted((a, b))),
                "reason": f"tier '{r.tier}': the pair has no measurement in Costanzo "
                          "2016, Kuzmin 2018 or Kuzmin 2020, so there is nothing to "
                          "compare against",
            })

    # Which double failed to construct is DERIVED, not recorded. The designed set is the
    # tiered rows of construction_validation_doubles.csv, the pairs picked to enable
    # triples; the built set is the d-rows of the run-4 order sheet. Exactly one pair is
    # in the first and not the second, and nothing was built that was not designed, so the
    # two lists pin the failure between them without anyone having to write it down.
    designed = {frozenset((r.gene1, r.gene2))
                for r in pd.read_csv(DOUBLES_REF).itertuples() if not pd.isna(r.tier)}
    built = {frozenset((a, b)) for _, a, b in doubles}
    never_built = designed - built
    assert not built - designed, f"built but never designed: {sorted(built - designed)}"
    assert len(never_built) == 1, f"expected 1 unbuilt design, got {sorted(never_built)}"
    blocked = sorted(next(iter(never_built)))
    gaps.append({
        "kind": "double, designed but never built", "strain": " + ".join(blocked),
        "reason": f"the only pair in the {len(designed)}-double design that is absent from "
                  f"the {len(built)} measured in {MEASURED_ROUND}. The transformation gave "
                  "no colonies. Both parent singles exist and were measured",
    })
    gaps.append({
        "kind": "triples, none exist", "strain": "--",
        "reason": "run 4 carried singles and doubles only; no triple has been constructed "
                  "in any round. The 20 in the build list are the first",
    })

    s = pd.DataFrame(srows)
    d = pd.DataFrame(drows)
    g = pd.DataFrame(gaps)
    s.to_csv(osp.join(RESULTS, "run4_measured_summary_singles.csv"), index=False)
    d.to_csv(osp.join(RESULTS, "run4_measured_summary_doubles.csv"), index=False)
    g.to_csv(osp.join(RESULTS, "run4_measured_summary_gaps.csv"), index=False)

    print(f"singles measured {len(s)} of {len(singles)} built | "
          f"with a published SMF {int(s.costanzo_smf.notna().sum())}")
    print(f"doubles measured {len(d)} of {len(doubles)} built (14 designed) | "
          f"with a published DMF {int(d.costanzo_dmf.notna().sum())}")
    print("triples 0\n")

    def md(df, cols, headers):
        w = [max(len(h), *(len(fmt(r[c])) for _, r in df.iterrows()))
             for c, h in zip(cols, headers, strict=True)]
        out = ["| " + " | ".join(h.ljust(x) for h, x in zip(headers, w, strict=True)) + " |",
               "|" + "|".join(":" + "-" * (x + 1) for x in w) + "|"]
        for _, r in df.iterrows():
            out.append("| " + " | ".join(fmt(r[c]).ljust(x)
                                         for c, x in zip(cols, w, strict=True)) + " |")
        return "\n".join(out)

    def fmt(v):
        return "--" if pd.isna(v) or v is None or v == "" else str(v)

    # Every column is prefixed with whose number it is. Reporting a Costanzo value next
    # to an unlabeled `boot SE` invites the SE to be read as theirs; it is ours.
    print(md(s, ["id", "orf", "common", "fitness", "boot_se", "across_plate_sd",
                 "costanzo_smf", "costanzo_se"],
             ["id", "ORF", "common", "ours: fitness", "ours: boot SE", "ours: plate SD",
              "Costanzo SMF", "Costanzo SE"]))
    print()
    print(md(d, ["id", "pair", "fitness", "boot_se", "across_plate_sd", "costanzo_dmf",
                 "costanzo_dmf_sd", "tier"],
             ["id", "pair", "ours: fitness", "ours: boot SE", "ours: plate SD",
              "Costanzo DMF", "Costanzo SD", "tier"]))
    print()
    print(g.to_string(index=False))
    print(f"\nwrote -> {RESULTS}/run4_measured_summary_*.csv")


if __name__ == "__main__":
    main()
