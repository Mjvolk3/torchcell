# experiments/035-env-chemgen-vanacloig-cgt/scripts/compare_models.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.compare_models]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/compare_models
"""Every scored model beside two references, paired on the same held-out compounds.

Reads the ladder (``results/ladder/<tag>_scores.csv``) and every factorized sweep
(``results/factorized/*/<name>_scores.csv``). A row of the output is one model on one
fold seed (or every fold seed pooled, ``fold_seed == -1``), target and compound subset:
its median and mean Spearman over the held-out compounds it scored, and its PAIRED
difference from each reference on exactly those compounds, with two bootstrap 95%
intervals for the mean difference.

``ci_low`` / ``ci_high`` resample the compound-evaluations. With every fold seed pooled
that treats the three evaluations of one compound as independent, which they are not, so
the interval is too narrow there. ``cluster_ci_low`` / ``cluster_ci_high`` resample the
COMPOUNDS, each carrying its mean difference over the fold seeds it was scored on, and
``compounds_improved`` counts the compounds whose mean difference is positive. On a single
fold seed the two intervals are the same thing. Quote the cluster interval for pooled rows.

REFERENCES, both from the ladder and both nested (no test compound touches a choice):

``ridge``     ``krr`` over the linear kernel of standardized FCFP4 counts, the ridge map
              of 031 with its penalty chosen by leave-one-compound-out.
``selected``  the ladder's whole-pipeline pick, per fold.

SUBSETS: ``all`` is the 41 served compounds; ``published`` is the 32 the paper reports,
since the nine unreported ones have replicate reliability near or below zero
(``vanacloig_data.UNREPORTED_COMPOUNDS``, issue #501).

ARM AGAINST ARM: ``PAIRS`` names the comparisons between two trained arms that the note
quotes (two environment layers against one, 150 epochs against 50, ...). Each is the same
paired difference, first arm minus second, on the compound-evaluations both scored, every
fold seed pooled, with both intervals.

Writes ``results/compare_models.csv`` and ``results/compare_pairs.csv``.
"""

from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
import re
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(__file__))
from vanacloig_data import RESULTS_SUBDIR, UNREPORTED_COMPOUNDS  # noqa: E402

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(
    EXPERIMENT_ROOT, "035-env-chemgen-vanacloig-cgt", "results", RESULTS_SUBDIR
)
KEY = ["fold_seed", "compound", "target"]
N_BOOT = 2000
COLUMNS = [
    "name",
    "fold_seed",
    "compounds",
    "spearman_median",
    "spearman_mean",
    "vs_ridge_mean_diff",
    "vs_ridge_ci_low",
    "vs_ridge_ci_high",
    "vs_ridge_wins",
]


#: (first arm, second arm): the paired difference is first minus second
PAIRS = [
    ("r9_operator:op_L1_lam0:ensemble", "r9_control:bil_L1_lam0:ensemble"),
    ("r10_envenc:enc_L1_lam0:ensemble", "r9_control:bil_L1_lam0:ensemble"),
    ("r13_scale:bilz_L1_lam0:ensemble", "r9_control:bil_L1_lam0:ensemble"),
    ("r14_envenc2:enc2_L1_lam0:ensemble", "r10_envenc:enc_L1_lam0:ensemble"),
    ("r16_envenc_e150:enc_L1_lam0_e150:seed0", "r10_envenc:enc_L1_lam0:seed0"),
]


def bootstrap_mean(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(diff, size=(N_BOOT, len(diff)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def bootstrap_compounds(
    paired: pd.DataFrame, rng: np.random.Generator
) -> tuple[float, float, int, int]:
    """The interval resampling compounds, and how many compounds improve on average."""
    per_compound = (
        (paired["spearman"] - paired["spearman_ref"])
        .groupby(paired["compound"])
        .mean()
        .to_numpy()
    )
    low, high = bootstrap_mean(per_compound, rng)
    return low, high, int((per_compound > 0).sum()), len(per_compound)


def load_scores(ladder_tag: str) -> pd.DataFrame:
    ladder = pd.read_csv(osp.join(RESULTS, "ladder", f"{ladder_tag}_scores.csv"))
    ladder["name"] = np.where(
        ladder["model"] == "selected",
        "ladder:selected",
        "ladder:" + ladder["model"] + "|" + ladder["kernel"],
    )
    models = [ladder[KEY + ["name", "spearman", "ceiling"]]]
    for path in sorted(glob.glob(osp.join(RESULTS, "factorized", "*", "*_scores.csv"))):
        sweep = osp.basename(osp.dirname(path))
        if sweep.startswith("smoke"):
            continue
        # a round repeated per fold seed is the sweep <round>_fs<seed>, and its "b"
        # continuation (r6b_mix_fs1) belongs with the round it continues (r6_mix)
        sweep = re.sub(r"_fs\d$", "", sweep)
        sweep = re.sub(r"^(r\d+)b_", r"\1_", sweep)
        # round 8 is split over cards and chained jobs (r8_small_a, r8_deep_c, ...)
        sweep = re.sub(r"^r8_(small|deep|mid)_[a-z]$", "r8", sweep)
        sweep = re.sub(r"^(r14_envenc2)_[a-z]$", r"\1", sweep)
        d = pd.read_csv(path)
        # a config repeated per fold seed is named <name>_fs<seed>; pool it under <name>
        # a config run one fold per process is named <name>_f<fold>; pool it the same way
        d["name"] = f"{sweep}:" + d["name"].str.replace(
            r"_f\d$", "", regex=True
        ).str.replace(r"_fs\d$", "", regex=True)
        d["name"] = d["name"] + ":" + d["member"]
        models.append(d[KEY + ["name", "spearman", "ceiling"]])
    scores = pd.concat(models, ignore_index=True)
    scores["subset"] = np.where(
        scores["compound"].isin(UNREPORTED_COMPOUNDS), "unreported", "published"
    )
    return scores


def summarize(
    g: pd.DataFrame, references: dict[str, pd.DataFrame], rng: np.random.Generator
) -> dict[str, float | int]:
    """Median, mean, fraction of ceiling, and the paired differences from each reference."""
    g = g.dropna(subset=["spearman"])
    row: dict[str, float | int] = {
        "compounds": len(g),
        "spearman_median": g["spearman"].median(),
        "spearman_mean": g["spearman"].mean(),
        "fraction_of_ceiling_median": (
            g["spearman"] / g["ceiling"].where(g["ceiling"] > 0)
        ).median(),
    }
    for ref_name, ref in references.items():
        paired = g.merge(ref, on=KEY, suffixes=("", "_ref")).dropna(
            subset=["spearman_ref"]
        )
        diff = (paired["spearman"] - paired["spearman_ref"]).to_numpy()
        if len(diff) == 0:
            continue
        low, high = bootstrap_mean(diff, rng)
        c_low, c_high, improved, distinct = bootstrap_compounds(paired, rng)
        row |= {
            f"vs_{ref_name}_mean_diff": float(diff.mean()),
            f"vs_{ref_name}_ci_low": low,
            f"vs_{ref_name}_ci_high": high,
            f"vs_{ref_name}_wins": int((diff > 0).sum()),
            f"vs_{ref_name}_paired": len(diff),
            f"vs_{ref_name}_cluster_ci_low": c_low,
            f"vs_{ref_name}_cluster_ci_high": c_high,
            f"vs_{ref_name}_compounds_improved": improved,
            f"vs_{ref_name}_distinct_compounds": distinct,
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-tag", default="ladder_r2")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    scores = load_scores(args.ladder_tag)
    references = {
        "ridge": scores[scores["name"] == "ladder:krr|linear:fcfp4_count"],
        "selected": scores[scores["name"] == "ladder:selected"],
    }
    rows = []
    for view in ("all", "published"):
        table = scores if view == "all" else scores[scores["subset"] == "published"]
        for (name, fold_seed, target), g in table.groupby(
            ["name", "fold_seed", "target"]
        ):
            rows.append(
                {"name": name, "fold_seed": fold_seed, "target": target, "subset": view}
                | summarize(g, references, rng)
            )
        # every fold seed pooled: 3 x 41 compound-evaluations, paired within fold seed
        for (name, target), g in table.groupby(["name", "target"]):
            rows.append(
                {"name": name, "fold_seed": -1, "target": target, "subset": view}
                | summarize(g, references, rng)
            )
    out = pd.DataFrame(rows).sort_values(
        ["subset", "target", "fold_seed", "spearman_median"],
        ascending=[True, True, True, False],
    )
    out.to_csv(osp.join(RESULTS, "compare_models.csv"), index=False)
    pair_rows = []
    for first, second in PAIRS:
        for view in ("all", "published"):
            table = scores if view == "all" else scores[scores["subset"] == "published"]
            table = table[table["target"] == "centered"]
            a, b = table[table["name"] == first], table[table["name"] == second]
            if len(a) == 0 or len(b) == 0:
                continue
            pair_rows.append(
                {"first": first, "second": second, "subset": view}
                | summarize(a, {"second": b}, rng)
            )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(osp.join(RESULTS, "compare_pairs.csv"), index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 200)
    for view in ("all", "published"):
        print(f"== {view} compounds, centered target")
        shown = out[
            (out["target"] == "centered")
            & (out["subset"] == view)
            & (out["fold_seed"] == -1)
        ]
        print(shown[COLUMNS].head(40).round(3).to_string(index=False))
    print("== arm against arm, centered target, every fold seed pooled")
    print(pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
