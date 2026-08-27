# experiments/019-simb-multimodal/scripts/project_census.py
# [[experiments.019-simb-multimodal.scripts.project_census]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/project_census
"""A census of every 019-023 W&B project: how many runs, which metrics, how many strains.

WHY THIS IS SEPARATE FROM THE LEADERBOARD. `pull_round_leaderboards.py` needs one history
request per run to find a maximum, which over 2,187 runs does not finish against a
rate-limited API. This script reads only each run's SUMMARY, which arrives in one paginated
query per project, so it covers ALL projects in minutes. It answers a different and, for
several claims, more important set of questions:

  * How much work exists, and where. Any retrospective quoting "the best score" has to say
    which projects it looked at.
  * WHICH METRIC NAME each project used. The honest per-feature validation correlation was
    `val/per_gene/pearson_per_gene` for expression and `val/global/pearson_per_gene` for
    morphology before the rename to `val/<head>/pearson_per_feature`. A reader of only the
    new names sees nothing in the eleven oldest projects.
  * HOW MANY STRAINS each project trained on. This is the load-bearing one. It is what
    establishes that every morphology project trained on 1,161 strains out of a 4,718-strain
    screen, and that the betaxanthin rounds differ (4,235 against 3,698) because one pins a
    competitor's test genes out of training, so their scores are not comparable.

`n_train_supervised` is read from at most `SAMPLE_RUNS` runs per project, because it is a
property of the build rather than of the run and reading every run to confirm a constant
would cost the saving this script exists to make. The distribution actually observed is
reported, so a project where it is NOT constant is visible rather than averaged away.

Run from repo root (needs network; W&B login):
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/project_census.py
"""

from __future__ import annotations

import collections
import json
import os
import os.path as osp
import re

import pandas as pd
import wandb
from dotenv import load_dotenv

ENTITY = "zhao-group"
EXPERIMENT = "019-simb-multimodal"
PROJECT_PATTERN = r"torchcell_(019|020|021|022|023)"

# Summaries are read for every run; these fields are sampled from the first SAMPLE_RUNS
# because they describe the dataset build rather than the individual run.
SAMPLE_RUNS = 30

# Which strand each project belongs to, so the census groups the way the document does.
STRAND_OF = {
    "expr": "expression",
    "morph": "morphology",
    "expr_morph": "expression_morphology_joint",
    "betaxanthin": "betaxanthin",
    "beta_carotene": "beta_carotene",
    "mulleder19": "amino_acid",
    "bx_m19": "betaxanthin_amino_acid_joint",
    "metabolism": "betaxanthin",
    "cgt_multitask": "expression",
}


def strand_for(project: str) -> str:
    # Longest key first so `expr_morph` is not matched by `expr`.
    for key in sorted(STRAND_OF, key=len, reverse=True):
        if key in project:
            return STRAND_OF[key]
    return "other"


def census(api: wandb.Api) -> list[dict[str, object]]:
    projects = [
        p.name for p in api.projects(ENTITY) if re.search(PROJECT_PATTERN, p.name)
    ]
    rows = []
    for project in sorted(projects):
        runs = list(api.runs(f"{ENTITY}/{project}", per_page=500))
        metrics: collections.Counter[str] = collections.Counter()
        n_train: collections.Counter[int] = collections.Counter()
        states: collections.Counter[str] = collections.Counter()
        for run in runs:
            states[run.state] += 1
        for run in runs[:SAMPLE_RUNS]:
            for key in run.summary.keys():
                if key.startswith("val/") and ("pearson" in key or "spearman" in key):
                    metrics[key] += 1
            value = run.summary.get("n_train_supervised") or run.config.get(
                "n_train_supervised"
            )
            if value is not None:
                n_train[int(value)] += 1
        rows.append(
            {
                "strand": strand_for(project),
                "project": project,
                "n_runs": len(runs),
                "states": dict(states),
                "metric_keys": [k for k, _ in metrics.most_common(8)],
                "n_train_observed": dict(n_train.most_common(5)),
                "n_train_sampled_from": min(len(runs), SAMPLE_RUNS),
            }
        )
        print(f"{project:45s} {len(runs):4d} runs", flush=True)
    return rows


def main() -> None:
    load_dotenv()
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    results_dir = osp.join(experiment_root, EXPERIMENT, "results")
    os.makedirs(results_dir, exist_ok=True)

    api = wandb.Api(timeout=60)
    rows = census(api)
    frame = pd.DataFrame(rows)
    frame.to_csv(osp.join(results_dir, "project_census.csv"), index=False)

    by_strand: dict[str, dict[str, object]] = {}
    for strand, group in frame.groupby("strand"):
        n_train_values = sorted(
            {v for d in group["n_train_observed"] for v in d}  # noqa: C416
        )
        by_strand[str(strand)] = {
            "n_projects": int(len(group)),
            "n_runs": int(group["n_runs"].sum()),
            "n_train_values_seen": n_train_values,
            "projects": list(group["project"]),
        }
    payload = {
        "n_projects": int(len(frame)),
        "n_runs_total": int(frame["n_runs"].sum()),
        "by_strand": by_strand,
        "per_project": rows,
    }
    with open(osp.join(results_dir, "project_census.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps({k: v for k, v in payload.items() if k != "per_project"}, indent=2))


if __name__ == "__main__":
    main()
