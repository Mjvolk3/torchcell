# experiments/035-env-chemgen-vanacloig-cgt/scripts/compare_models.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.compare_models]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/compare_models
"""Every scored model beside two references, paired on the same held-out compounds.

Reads the ladder (``results/ladder/<tag>_scores.csv``) and every factorized sweep
(``results/factorized/*/<name>_scores.csv``). A row of the output is one model on one
fold seed and target: its median and mean Spearman over the held-out compounds it
scored, and its PAIRED difference from each reference on exactly those compounds, with
a bootstrap 95% interval over compounds for the mean difference.

REFERENCES, both from the ladder and both nested (no test compound touches a choice):

``ridge``     ``krr`` over the linear kernel of standardized FCFP4 counts, the ridge map
              of 031 with its penalty chosen by leave-one-compound-out.
``selected``  the ladder's whole-pipeline pick, per fold.

Writes ``results/compare_models.csv``.
"""

from __future__ import annotations

import argparse
import glob
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "035-env-chemgen-vanacloig-cgt", "results")
KEY = ["fold_seed", "compound", "target"]
N_BOOT = 2000


def bootstrap_mean(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(diff, size=(N_BOOT, len(diff)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-tag", default="ladder_r2")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    ladder = pd.read_csv(osp.join(RESULTS, "ladder", f"{args.ladder_tag}_scores.csv"))
    ladder["name"] = np.where(
        ladder["model"] == "selected",
        "ladder:selected",
        "ladder:" + ladder["model"] + "|" + ladder["kernel"],
    )
    models = [ladder[KEY + ["name", "spearman", "ceiling"]]]
    for path in sorted(glob.glob(osp.join(RESULTS, "factorized", "*", "*_scores.csv"))):
        sweep = osp.basename(osp.dirname(path))
        if sweep == "smoke":
            continue
        d = pd.read_csv(path)
        d["name"] = f"{sweep}:" + d["name"] + ":" + d["member"]
        models.append(d[KEY + ["name", "spearman", "ceiling"]])
    scores = pd.concat(models, ignore_index=True)

    references = {
        "ridge": scores[scores["name"] == "ladder:krr|linear:fcfp4_count"],
        "selected": scores[scores["name"] == "ladder:selected"],
    }
    rows = []
    for (name, fold_seed, target), g in scores.groupby(["name", "fold_seed", "target"]):
        g = g.dropna(subset=["spearman"])
        row = {
            "name": name,
            "fold_seed": fold_seed,
            "target": target,
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
            row |= {
                f"vs_{ref_name}_mean_diff": float(diff.mean()),
                f"vs_{ref_name}_ci_low": low,
                f"vs_{ref_name}_ci_high": high,
                f"vs_{ref_name}_wins": int((diff > 0).sum()),
                f"vs_{ref_name}_paired": len(diff),
            }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(
        ["target", "fold_seed", "spearman_median"], ascending=[True, True, False]
    )
    out.to_csv(osp.join(RESULTS, "compare_models.csv"), index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 200)
    cols = [
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
    print(
        out[out["target"] == "centered"][cols].head(60).round(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
