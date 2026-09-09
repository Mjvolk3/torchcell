# experiments/019-simb-multimodal/scripts/wandb_run_index.py
# [[experiments.019-simb-multimodal.scripts.wandb_run_index]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/wandb_run_index
"""Index every W&B run behind the expression document, and emit its hyperlinked table.

Reads the SAME result files the document's numbers come from, so a run cannot appear in
a table without appearing here, and vice versa:

  - results/short_budget_spread.json        the eight long-budget v9 mask-schedule arms
  - results/round_leaderboards.csv          the objective round (tag `stage-launch`)
  - results/mech_round_readout.csv          the mechanism round, one run per arm and seed
  - results/pearson_round_readout.csv       the metric-aligned round
  - results/v10_grid_factorial.csv          the v10 grid, one run per cell and seed

Writes results/wandb_run_index.json (pydantic records) and
notes-tex/019-simb-multimodal-expression/tables/wandb_runs_019.tex: one row per round and
arm (or grid cell), each entry `seed: id` linked to its run page. Run URLs are built from
the project name and run id; the entity is the leaderboard puller's.

Run from the repository root:

    python experiments/019-simb-multimodal/scripts/wandb_run_index.py
"""

from __future__ import annotations

import importlib.util
import json
import os.path as osp
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "experiments" / "019-simb-multimodal" / "results"
TABLE_OUT = (
    REPO_ROOT / "notes-tex" / "019-simb-multimodal-expression" / "tables" / "wandb_runs_019.tex"
)
INDEX_OUT = RESULTS / "wandb_run_index.json"

_SPEC = importlib.util.spec_from_file_location(
    "_plb", osp.join(osp.dirname(osp.abspath(__file__)), "pull_round_leaderboards.py")
)
_PLB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLB)
ENTITY = _PLB.ENTITY
V9 = "torchcell_019_expr_v9"
V10 = "torchcell_019_expr_v10"

# Display order of rounds and, within a round, of arms.
ROUNDS = ("incumbent", "objective", "mechanism", "pearson", "v10")
ROUND_LABEL = {
    "incumbent": "v9 mask arms",
    "objective": "objective",
    "mechanism": "mechanism",
    "pearson": "metric-aligned",
    "v10": "v10 grid",
}
ARM_ORDER = {
    "objective": ["Q_point", "Q_crps", "Q_laplace"],
    "mechanism": ["R_ref", "R_basis64", "R_pergene", "R_pergene_basis64"],
    "pearson": ["Q_pearson", "Q_pearson_mse", "Q_pearson_b64"],
}


class RunRecord(BaseModel):
    """One W&B run behind the document: which round and arm it belongs to, and its page."""

    round: str
    arm: str
    seed: int
    project: str
    wandb_run_id: str
    wandb_url: str
    epochs: int


class RunIndex(BaseModel):
    """The whole index: entity, project pages, and every run record."""

    entity: str
    projects: dict[str, str]
    n_runs: int
    runs: list[RunRecord]


def url(project: str, run_id: str) -> str:
    return f"https://wandb.ai/{ENTITY}/{project}/runs/{run_id}"


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def load_runs() -> list[RunRecord]:
    runs: list[RunRecord] = []
    lb = pd.read_csv(RESULTS / "round_leaderboards.csv")
    lb = lb[lb.project == V9].set_index("run_id")

    spread = json.loads((RESULTS / "short_budget_spread.json").read_text())
    for rid in spread["run_ids"]:
        tags = lb.loc[rid, "tags"].split(",")
        arm = [t for t in tags if t.startswith("M_")][0]
        runs.append(RunRecord(round="incumbent", arm=arm, seed=int(lb.loc[rid, "seed"]),
                              project=V9, wandb_run_id=rid, wandb_url=url(V9, rid),
                              epochs=int(lb.loc[rid, "epochs"])))

    # Objective round: the launch-stage runs of 1,000 epochs or more. Each arm and seed has
    # two entries, the fresh run and its 2026-09-05 resume; both are listed.
    obj = lb[lb.tags.str.contains("stage-launch", na=False) & (lb.epochs >= 1000)]
    for rid, r in obj.iterrows():
        arm = [t for t in r.tags.split(",") if t.startswith("Q_")][0]
        runs.append(RunRecord(round="objective", arm=arm, seed=int(r.seed), project=V9,
                              wandb_run_id=rid, wandb_url=url(V9, rid), epochs=int(r.epochs)))

    for fname, rnd in (("mech_round_readout.csv", "mechanism"),
                       ("pearson_round_readout.csv", "pearson")):
        t = pd.read_csv(RESULTS / fname)
        for _, r in t.iterrows():
            runs.append(RunRecord(round=rnd, arm=r.arm, seed=int(r.seed), project=V9,
                                  wandb_run_id=r.run_id, wandb_url=url(V9, r.run_id),
                                  epochs=int(r.n_epochs)))

    grid = pd.read_csv(RESULTS / "v10_grid_factorial.csv")
    for _, r in grid.iterrows():
        runs.append(RunRecord(round="v10", arm=r.cell_name, seed=int(r.seed), project=V10,
                              wandb_run_id=r.run_id, wandb_url=url(V10, r.run_id),
                              epochs=int(r.n_epochs)))

    def key(r: RunRecord) -> tuple:
        order = ARM_ORDER.get(r.round)
        arm_rank = order.index(r.arm) if order else (
            int(r.arm.split("_")[0][1:]) if r.round == "v10" else r.arm)
        return (ROUNDS.index(r.round), arm_rank, r.seed, -r.epochs)

    return sorted(runs, key=key)


def write_table(index: RunIndex) -> None:
    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/wandb_run_index.py",
        "%% from results/short_budget_spread.json, round_leaderboards.csv,",
        "%% mech_round_readout.csv, pearson_round_readout.csv, v10_grid_factorial.csv.",
        "%% Do not edit by hand; rerun the script.",
        "%% SOURCE: the result files named above (run ids), URLs built from project and id.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[W\\&B runs behind this document]{Every run behind the numbers in this "
        f"document, {index.n_runs} in all, in projects "
        + " and ".join(
            f"\\href{{{u}}}{{\\texttt{{{tex_escape(p)}}}}}" for p, u in index.projects.items()
        )
        + f" (entity \\texttt{{{tex_escape(index.entity)}}}). One row per round and arm; "
        "each entry is a seed and its run id, linked to the run page, with the final epoch "
        "in brackets. The objective round lists each fresh run and its resume. The v10 "
        "grid names cells by embedding, trunk, readout and weight decay.}",
        "  \\label{tab:wandb-runs-019}",
        "  \\begin{tabular}{@{}llp{0.60\\textwidth}@{}}",
        "    \\hline",
        "    Round & Arm & Seed: run [epochs] \\\\",
        "    \\hline",
    ]
    for rnd in ROUNDS:
        rows = [r for r in index.runs if r.round == rnd]
        arms = []
        for r in rows:
            if r.arm not in arms:
                arms.append(r.arm)
        for i, arm in enumerate(arms):
            entries = " ".join(
                f"{r.seed}: \\href{{{r.wandb_url}}}{{\\texttt{{{r.wandb_run_id}}}}} "
                f"[{r.epochs:,}]"
                for r in rows if r.arm == arm
            )
            label = tex_escape(ROUND_LABEL[rnd]) if i == 0 else ""
            lines.append(f"    {label} & \\texttt{{{tex_escape(arm)}}} & {entries} \\\\")
        lines.append("    \\hline")
    lines += ["  \\end{tabular}", "\\end{table}", ""]
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text("\n".join(lines))


def main() -> None:
    runs = load_runs()
    projects = {p: f"https://wandb.ai/{ENTITY}/{p}" for p in (V9, V10)}
    index = RunIndex(entity=ENTITY, projects=projects, n_runs=len(runs), runs=runs)
    INDEX_OUT.write_text(index.model_dump_json(indent=2))
    write_table(index)
    by_round = {r: sum(1 for x in runs if x.round == r) for r in ROUNDS}
    print(f"{len(runs)} runs indexed: {by_round}")
    print(f"wrote {INDEX_OUT}")
    print(f"wrote {TABLE_OUT}")


if __name__ == "__main__":
    main()
