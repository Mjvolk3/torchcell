# experiments/025-solid-growth/scripts/wandb_run_index_025.py
# [[experiments.025-solid-growth.scripts.wandb_run_index_025]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/wandb_run_index_025
"""Index every W&B run behind the additive-baselines document, and emit its linked table.

Every transformer number in notes-tex/025-additive-baselines traces to a training run:
the three 010 checkpoints on arm R's split, the 025 replication (GH 1598), the arm Q
run in the 010 configuration (GH 1640), and the three IGB sequence-embedding runs on arm
Q. The registry below names each once, with the configuration it ran and what that
configuration turns on, and this script verifies each run against the W&B API (state,
epochs logged, best validation Pearson from the run's history), writes
``results/wandb_run_index_025.json`` and the hyperlinked LaTeX table
``notes-tex/025-additive-baselines/tables/t5-wandb-runs.tex``. Ids are never typed into
the document; they come from here.

Run from the repository root (needs W&B credentials):

    python experiments/025-solid-growth/scripts/wandb_run_index_025.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import wandb
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "experiments" / "025-solid-growth" / "results"
TABLE_OUT = (
    REPO_ROOT / "notes-tex" / "025-additive-baselines" / "tables" / "t5-wandb-runs.tex"
)
INDEX_OUT = RESULTS / "wandb_run_index_025.json"
ENTITY = "zhao-group"
P010 = "torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer"
P025 = "torchcell_025-solid-growth_equivariant_cell_graph_transformer"
VAL_KEY = "val/gene_interaction/Pearson"


class RunSpec(BaseModel):
    """One row of the registry: what ran, where, and under which configuration."""

    label: str
    arm: str
    project: str
    run_id: str
    job: str
    config: str
    gene_input: str
    schedule: str
    readout: str
    graph_penalty: str
    scored_on_test: str


class RunRecord(RunSpec):
    """A registry entry filled in with the W&B run's URL, state, and best validation score."""

    url: str
    state: str
    created: str
    n_epochs_logged: int
    best_epoch: int
    val_pearson_best: float


class RunIndex(BaseModel):
    """The resolved run registry written to disk: a count and the ordered records."""

    n_runs: int
    runs: list[RunRecord]


# The registry. Order is the order of the table. The 010 checkpoints are the arm R
# reference and were scored on the 010 test split by their own evaluation runs; 1598 and
# 1640 have no test evaluation; 1640's epoch 7 checkpoint was scored on CPU afterwards
# (score_cgt_checkpoint_cpu.py); the IGB runs finished 30 epochs with no test evaluation.
REGISTRY = [
    RunSpec(
        label="CGT M01",
        arm="R (010 split)",
        project=P010,
        run_id="lzs9pcj3",
        job="IGB, 010",
        config="equivariant_cell_graph_transformer_cabbi_002",
        gene_input="learnable table",
        schedule="cosine, 30-epoch first cycle",
        readout="mean",
        graph_penalty="KL, layer 1, all nine heads",
        scored_on_test="yes, 010 eval run",
    ),
    RunSpec(
        label="CGT M02",
        arm="R (010 split)",
        project=P010,
        run_id="yv4r30bi",
        job="IGB, 010",
        config="equivariant_cell_graph_transformer_cabbi_002",
        gene_input="learnable table",
        schedule="cosine, 30-epoch first cycle",
        readout="mean",
        graph_penalty="KL, layer 1, all nine heads",
        scored_on_test="yes, 010 eval run",
    ),
    RunSpec(
        label="CGT M03",
        arm="R (010 split)",
        project=P010,
        run_id="c7671wgj",
        job="IGB, 010",
        config="equivariant_cell_graph_transformer_cabbi_002",
        gene_input="learnable table",
        schedule="cosine, longer first cycle",
        readout="mean",
        graph_penalty="KL, layer 1, all nine heads",
        scored_on_test="yes, 010 eval run",
    ),
    RunSpec(
        label="GH 1598",
        arm="R",
        project=P025,
        run_id="0yw7moue",
        job="GilaHyper 1598",
        config="cgt_s0_r_kl_000",
        gene_input="learnable table",
        schedule="cosine, 010 schedule",
        readout="sum",
        graph_penalty="KL, corrected edge normalization",
        scored_on_test="no",
    ),
    RunSpec(
        label="GH 1640",
        arm="Q",
        project=P025,
        run_id="327csnlk",
        job="GilaHyper 1640",
        config="cgt_s0_q_kl_004",
        gene_input="learnable table",
        schedule="cosine, 010 schedule",
        readout="sum",
        graph_penalty="KL, corrected edge normalization",
        scored_on_test="epoch 7 checkpoint, CPU",
    ),
    RunSpec(
        label="IGB 2391132",
        arm="Q",
        project=P025,
        run_id="s1vx2zgw",
        job="IGB mmli 2391132",
        config="cgt_s0_q_kl_emb_017",
        gene_input="four-region sequence composite",
        schedule="constant 2.5e-4, 30 epochs",
        readout="perturbed CLS",
        graph_penalty="KL, corrected edge normalization",
        scored_on_test="no",
    ),
    RunSpec(
        label="IGB 2391133",
        arm="Q",
        project=P025,
        run_id="8aa08xx0",
        job="IGB mmli 2391133",
        config="cgt_s0_q_kl_calm_020",
        gene_input="CaLM codon embedding",
        schedule="constant 2.5e-4, 30 epochs",
        readout="perturbed CLS",
        graph_penalty="KL, corrected edge normalization",
        scored_on_test="no",
    ),
    RunSpec(
        label="IGB 2391134",
        arm="Q",
        project=P025,
        run_id="pmkzwwzw",
        job="IGB mmli 2391134",
        config="cgt_s0_q_kl_prot_021",
        gene_input="ProtT5 protein embedding",
        schedule="constant 2.5e-4, 30 epochs",
        readout="perturbed CLS",
        graph_penalty="KL, corrected edge normalization",
        scored_on_test="no",
    ),
]


def tex(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&")


def short_config(name: str) -> str:
    """The distinguishing tail of a config name; the full name is in the index json."""
    for prefix in ("equivariant_cell_graph_transformer_", "cgt_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def fetch(spec: RunSpec, api: wandb.Api) -> RunRecord:
    run = api.run(f"{ENTITY}/{spec.project}/{spec.run_id}")
    hist = pd.DataFrame(run.scan_history(keys=["epoch", VAL_KEY])).dropna()
    if hist.empty:
        raise SystemExit(f"{spec.run_id}: no validation history under {VAL_KEY}")
    i = int(hist[VAL_KEY].idxmax())
    return RunRecord(
        **spec.model_dump(),
        url=f"https://wandb.ai/{ENTITY}/{spec.project}/runs/{spec.run_id}",
        state=run.state,
        created=run.created_at[:10],
        n_epochs_logged=int(hist["epoch"].nunique()),
        best_epoch=int(hist.loc[i, "epoch"]),
        val_pearson_best=float(hist.loc[i, VAL_KEY]),
    )


def write_table(index: RunIndex) -> None:
    lines = [
        "%% GENERATED by experiments/025-solid-growth/scripts/wandb_run_index_025.py",
        "%% SOURCE: results/wandb_run_index_025.json, each row verified against the W&B API. Do not edit by hand.",
        "\\begin{tabular}{@{}llll>{\\raggedright\\arraybackslash}p{19mm}>{\\raggedright\\arraybackslash}p{24mm}rr>{\\raggedright\\arraybackslash}p{15mm}@{}}",
        "\\toprule",
        "Run & W\\&B & Arm & Config & Gene input & Schedule, readout & Epochs & Val max (ep.) & Test \\\\",
        "\\midrule",
    ]
    for r in index.runs:
        link = f"\\href{{{r.url}}}{{\\texttt{{{r.run_id}}}}}"
        lines.append(
            f"{tex(r.label)} & {link} & {tex(r.arm)} & \\texttt{{{tex(short_config(r.config))}}} & {tex(r.gene_input)} & "
            f"{tex(r.schedule)}, {tex(r.readout)} & {r.n_epochs_logged} & {r.val_pearson_best:.3f} ({r.best_epoch}) & "
            f"{tex(r.scored_on_test)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text("\n".join(lines))
    print(f"wrote {TABLE_OUT}")


def main() -> None:
    api = wandb.Api(timeout=120)
    runs = [fetch(spec, api) for spec in REGISTRY]
    index = RunIndex(n_runs=len(runs), runs=runs)
    INDEX_OUT.write_text(json.dumps(index.model_dump(), indent=2))
    print(f"wrote {INDEX_OUT}")
    for r in runs:
        print(
            f"{r.label:<12s} {r.run_id} {r.state:<9s} epochs {r.n_epochs_logged:>3d} best {r.val_pearson_best:.4f} @ {r.best_epoch}"
        )
    write_table(index)


if __name__ == "__main__":
    main()
