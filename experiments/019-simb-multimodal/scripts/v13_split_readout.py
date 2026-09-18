# experiments/019-simb-multimodal/scripts/v13_split_readout.py
# [[experiments.019-simb-multimodal.scripts.v13_split_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/v13_split_readout
"""Read the v13 split round: how much the partition moves the score, and what 90/10 buys.

Reads every run of W&B project ``torchcell_019_expr_v13`` (24 runs: H_ref and H_concat on
split seeds 0 to 3, two init seeds each, plus split 0 with the test records folded into
train). The statistic is the strand's ``roll_max``, the maximum of a centered 5-epoch
rolling mean of ``val/expression/pearson_per_feature``, read at the MATCHED epoch (the
lowest epoch any run has reached) so no arm is scored on more epochs than another; the
unmatched roll_max and the last value are carried for reference. Four reads:

  1. per partition, the score across both readouts and both seeds, and the spread of
     those partition means against the spread within a partition;
  2. H_concat minus H_ref paired within card and seed (12 pairs);
  3. split 0 at 90/10 minus split 0 at 80/10/10, same validation strains, same seeds;
  4. the linear baselines on the same partitions (results/expression_baselines_split),
     validation-selected B2 and B3 on ProtT5, next to the trained arms.

Also carries the test score at the best-validation checkpoint where a run has finished
(``test/expression/pearson_per_feature`` in the run summary). Partial runs are partial:
the matched epoch is printed with every number. Writes results/v13_split_readout.json.

Parameterized by ``--round`` for the rounds that reuse the design: v14 (the proteome,
``val/proteome``, baselines in results/baselines_split_fig3_proteome) and v15 (weight decay,
1e-2 against the reference in read 2).

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/v13_split_readout.py --round v13
    python experiments/019-simb-multimodal/scripts/v13_split_readout.py --round v14
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import re
from typing import Any

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

from torchcell.utils.paths import experiment_results_dir  # noqa: E402

# The rounds that share the split design. `arm_re` group 1 is the contrast level (readout
# for v13/v14, the decay level for v15), group 2 the split; `ref` names the reference level
# and `alt` the level contrasted against it in read 2.
ROUNDS: dict[str, dict[str, Any]] = {
    "v13": {
        "project": "zhao-group/torchcell_019_expr_v13",
        "phenotype": "expression",
        "prefix": "V_",
        "arm_re": r"V_(ref|concat)_(s\d+(?:_90)?)",
        "ref": "ref",
        "alt": "concat",
        "splits": ["s0", "s0_90", "s1", "s2", "s3"],
        "baselines_dir": "expression_baselines_split",
        "out": "v13_split_readout.json",
    },
    "v14": {
        "project": "zhao-group/torchcell_019_prot_v14",
        "phenotype": "proteome",
        "prefix": "P_",
        "arm_re": r"P_(ref|concat)_(s\d+)",
        "ref": "ref",
        "alt": "concat",
        "splits": ["s0", "s1", "s2", "s3"],
        "baselines_dir": "baselines_split_fig3_proteome",
        "out": "v14_proteome_readout.json",
    },
    "v15": {
        "project": "zhao-group/torchcell_019_expr_v15",
        "phenotype": "expression",
        "prefix": "W_",
        "arm_re": r"W_(ref|wd1e2|wd1e1)_(s\d+)",
        "ref": "ref",
        "alt": "wd1e2",
        "alt_extra": ["wd1e1"],
        "splits": ["s1", "s2"],
        "baselines_dir": "expression_baselines_split",
        "out": "v15_wd_readout.json",
    },
}
_args = argparse.ArgumentParser(description=__doc__)
_args.add_argument("--round", choices=sorted(ROUNDS), default="v13")
ROUND = ROUNDS[_args.parse_args().round]
PROJECT = ROUND["project"]
KEY = f"val/{ROUND['phenotype']}/pearson_per_feature"
TEST_KEY = f"test/{ROUND['phenotype']}/pearson_per_feature"
# NMSE (per-feature squared error over that feature's variance, 1.0 = predict each gene's
# mean) is carried beside the Pearson because the two separate ordering from magnitude:
# the Pearson can rise while the NMSE passes 1.0 as the head commits to bolder predictions.
NMSE_KEY = f"val/{ROUND['phenotype']}/nmse"
SPLITS: list[str] = ROUND["splits"]
REF, ALT = ROUND["ref"], ROUND["alt"]
WINDOW = 5
PLATEAU = 0.05  # a roll_max below this at the matched epoch is a run that never trained


def roll_max_le(h: pd.DataFrame, epoch: int) -> float:
    hh = h[h["epoch"] <= epoch]
    return float(hh[KEY].rolling(WINDOW, center=True).mean().max())


def main() -> None:
    api = wandb.Api()
    rows: list[dict[str, Any]] = []
    hist: dict[str, pd.DataFrame] = {}
    for r in api.runs(PROJECT):
        arm = r.config.get("arm") or next(
            t for t in r.tags if t.startswith(ROUND["prefix"])
        )
        m = re.fullmatch(ROUND["arm_re"], arm)
        if m is None:
            raise ValueError(f"{r.id}: arm {arm} does not parse")
        h = r.history(keys=["epoch", KEY, NMSE_KEY], samples=20000, pandas=True)
        h = h.dropna(subset=[KEY]).sort_values("epoch").reset_index(drop=True)
        if h.empty:
            continue
        roll = h[KEY].rolling(WINDOW, center=True).mean()
        hist[r.id] = h
        peak_i = int(roll.idxmax())
        rows.append(
            {
                "id": r.id,
                "arm": arm,
                "readout": m.group(1),
                "split": m.group(2),
                "seed": int(r.config["seed"]),
                "state": r.state,
                "epoch": int(h["epoch"].max()),
                "last": float(h[KEY].iloc[-1]),
                "roll_max": float(roll.max()),
                "roll_max_epoch": int(h["epoch"].iloc[peak_i]),
                "nmse_at_peak": float(h[NMSE_KEY].iloc[peak_i]),
                "nmse_min": float(h[NMSE_KEY].min()),
                "nmse_min_epoch": int(h["epoch"].iloc[int(h[NMSE_KEY].idxmin())]),
                "nmse_last": float(h[NMSE_KEY].iloc[-1]),
                "test_at_best_val": (
                    float(r.summary[TEST_KEY]) if TEST_KEY in r.summary else None
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values(["split", "readout", "seed"])
    matched = int(df["epoch"].min())
    df["roll_max_matched"] = [roll_max_le(hist[i], matched) for i in df["id"]]

    out: dict[str, Any] = {
        "generated_by": "experiments/019-simb-multimodal/scripts/v13_split_readout.py",
        "project": PROJECT,
        "statistic": f"max of a centered {WINDOW}-epoch rolling mean of {KEY}",
        "matched_epoch": matched,
        "n_runs": int(len(df)),
        "runs": df.to_dict(orient="records"),
    }
    print(f"{len(df)} runs; matched epoch {matched}")
    pd.set_option("display.width", 220)
    print(
        df[
            [
                "arm",
                "seed",
                "state",
                "epoch",
                "roll_max_matched",
                "roll_max",
                "roll_max_epoch",
                "last",
                "nmse_at_peak",
                "nmse_min",
                "nmse_min_epoch",
                "nmse_last",
                "test_at_best_val",
                "id",
            ]
        ].to_string(index=False)
    )

    # 1. partition spread at the matched epoch (80/10/10 splits only)
    per_split: dict[str, Any] = {}
    for s in [x for x in SPLITS if not x.endswith("_90")]:
        sub = df[(df["split"] == s) & (df["roll_max_matched"] >= PLATEAU)][
            "roll_max_matched"
        ]
        ok = df[(df["split"] == s) & (df["roll_max_matched"] >= PLATEAU)]
        per_split[s] = {
            "mean": float(sub.mean()),
            "sd_within": float(sub.std(ddof=1)),
            "n": int(len(sub)),
            "nmse_at_peak_mean": float(ok["nmse_at_peak"].mean()),
            "nmse_min_mean": float(ok["nmse_min"].mean()),
            "nmse_last_mean": float(ok["nmse_last"].mean()),
        }
    means = np.array([per_split[s]["mean"] for s in per_split])
    within = np.array([per_split[s]["sd_within"] for s in per_split])
    out["partition"] = {
        "per_split": per_split,
        "sd_between_partitions": float(means.std(ddof=1)),
        "range_between_partitions": float(means.max() - means.min()),
        "sd_within_partition_pooled": float(np.sqrt((within**2).mean())),
    }
    print(
        "\n1. partition (mean over both readouts and seeds at the matched epoch, plateau runs excluded):"
    )
    for s, v in per_split.items():
        print(
            f"   {s}: {v['mean']:.4f} (within sd {v['sd_within']:.4f}, n={v['n']})"
            f"  nmse at peak {v['nmse_at_peak_mean']:.3f}, min {v['nmse_min_mean']:.3f},"
            f" last {v['nmse_last_mean']:.3f}"
        )
    print(
        f"   between-partition sd {out['partition']['sd_between_partitions']:.4f}, "
        f"range {out['partition']['range_between_partitions']:.4f}; "
        f"pooled within-partition sd {out['partition']['sd_within_partition_pooled']:.4f}"
    )

    # 2. paired alt - ref within card and seed, for the primary alt and any extra alt
    def _section2(alt: str, key: str) -> None:
        pairs: list[dict[str, Any]] = []
        for s in SPLITS:
            ref = df[(df["split"] == s) & (df["readout"] == REF)].set_index("seed")
            con = df[(df["split"] == s) & (df["readout"] == alt)].set_index("seed")
            for seed in sorted(set(ref.index) & set(con.index)):
                pairs.append(
                    {
                        "split": s,
                        "seed": int(seed),
                        "ref": float(ref.loc[seed, "roll_max_matched"]),
                        "concat": float(con.loc[seed, "roll_max_matched"]),
                        "diff": float(
                            con.loc[seed, "roll_max_matched"]
                            - ref.loc[seed, "roll_max_matched"]
                        ),
                    }
                )

        # A run that never left the plateau (roll_max under PLATEAU) is a training failure,
        # not a readout measurement; the pair it sits in is reported and then set aside.
        def _pair_stats(ps: list[dict[str, Any]]) -> dict[str, Any]:
            d = np.array([p["diff"] for p in ps])
            return {
                "pairs": ps,
                "mean": float(d.mean()),
                "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
                "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
                if len(d) > 1
                else None,
                "n_positive": int((d > 0).sum()),
                "n": int(len(d)),
            }

        clean = [p for p in pairs if min(p["ref"], p["concat"]) >= PLATEAU]
        out[f"{key}_minus_ref"] = _pair_stats(pairs)
        out[f"{key}_minus_ref_excluding_plateau"] = _pair_stats(clean)
        out["plateau_runs"] = df[df["roll_max_matched"] < PLATEAU][
            ["arm", "seed", "id", "roll_max_matched"]
        ].to_dict(orient="records")
        print(
            f"\n2. {alt} minus {REF}, paired within split and seed at the matched epoch:"
        )
        for p in pairs:
            flag = (
                "  (plateau run in pair)"
                if min(p["ref"], p["concat"]) < PLATEAU
                else ""
            )
            print(
                f"   {p['split']:<6} seed {p['seed']}: {p['diff']:+.4f} (ref {p['ref']:.4f}, concat {p['concat']:.4f}){flag}"
            )
        for label, c in [
            ("all pairs", out[f"{key}_minus_ref"]),
            ("excluding plateau pairs", out[f"{key}_minus_ref_excluding_plateau"]),
        ]:
            if c["n"] == 0:
                print(f"   {label}: no pairs yet")
                continue
            sd = f"{c['sd']:.4f}" if c["sd"] is not None else "--"
            t = f"{c['t']:+.2f}" if c["t"] is not None else "--"
            print(
                f"   {label}: mean {c['mean']:+.4f}, sd {sd}, t {t}, {c['n_positive']}/{c['n']} positive"
            )

    _section2(ALT, "concat")
    for extra in ROUND.get("alt_extra", []):
        _section2(extra, extra)

    # 3. 90/10 minus 80/10/10 on split 0, same seeds
    fold: list[dict[str, Any]] = []
    for readout in sorted(df["readout"].unique()):
        a = df[(df["split"] == "s0") & (df["readout"] == readout)].set_index("seed")
        b = df[(df["split"] == "s0_90") & (df["readout"] == readout)].set_index("seed")
        for seed in sorted(set(a.index) & set(b.index)):
            fold.append(
                {
                    "readout": readout,
                    "seed": int(seed),
                    "80_10_10": float(a.loc[seed, "roll_max_matched"]),
                    "90_10": float(b.loc[seed, "roll_max_matched"]),
                    "diff": float(
                        b.loc[seed, "roll_max_matched"]
                        - a.loc[seed, "roll_max_matched"]
                    ),
                }
            )
    fd = np.array([p["diff"] for p in fold])
    out["fold90_minus_80"] = {
        "pairs": fold,
        "mean": float(fd.mean()) if len(fd) else None,
        "sd": float(fd.std(ddof=1)) if len(fd) > 1 else None,
        "n": int(len(fd)),
    }
    print(
        "\n3. split 0, test folded into train (90/10) minus 80/10/10, same val strains and seed:"
    )
    for p in fold:
        print(
            f"   {p['readout']:<6} seed {p['seed']}: {p['diff']:+.4f} (80/10/10 {p['80_10_10']:.4f}, 90/10 {p['90_10']:.4f})"
        )
    if len(fd):
        print(f"   mean {fd.mean():+.4f} over {len(fd)} pairs")

    # 4. linear baselines on the same partitions
    res_dir = experiment_results_dir("019-simb-multimodal", __file__)
    base: dict[str, Any] = {}
    for s, name in [
        ("s0", "seed0"),
        ("s1", "seed1"),
        ("s2", "seed2"),
        ("s3", "seed3"),
        ("s0_90", "seed0_fold90"),
    ]:
        path = osp.join(res_dir, ROUND["baselines_dir"], f"{name}.json")
        if not osp.exists(path):
            continue
        with open(path) as f:
            b = json.load(f)
        b2 = b["B2_bilinear"]["by_embedding"]["prot_T5_all"]["selected_on_val"]
        b3 = b["B3_neighbor_average"]["by_embedding"]["prot_T5_all"]["selected_on_val"]
        base[s] = {
            "B2_prot_T5_val": b2["val_pearson_per_feature"],
            "B2_prot_T5_test": b2.get("test_pearson_per_feature"),
            "B3_prot_T5_val": b3["val_pearson_per_feature"],
            "B3_prot_T5_test": b3.get("test_pearson_per_feature"),
        }
    out["baselines"] = base
    print("\n4. trained arms against the linear baselines on the same partition (val):")
    for s in SPLITS:
        arms = df[(df["split"] == s) & (df["roll_max_matched"] >= PLATEAU)]
        if s not in base or arms.empty:
            continue
        ref = arms[arms["readout"] == REF]["roll_max_matched"].mean()
        con = arms[arms["readout"] == ALT]["roll_max_matched"].mean()
        print(
            f"   {s:<6} {REF} {ref:.4f}  {ALT} {con:.4f}  |  B2 ProtT5 {base[s]['B2_prot_T5_val']:.4f}  "
            f"B3 ProtT5 {base[s]['B3_prot_T5_val']:.4f}"
        )

    dst = osp.join(res_dir, ROUND["out"])
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
