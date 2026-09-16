# experiments/025-solid-growth/scripts/disjoint_embedding_wandb_report.py
# [[experiments.025-solid-growth.scripts.disjoint_embedding_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/disjoint_embedding_wandb_report
"""Publish a W&B report for the query-pair-disjoint embedding arms.

One run set filtered to the `split_Q` tag (the disjoint-split arms under the constant-rate
protocol), line plots of validation Pearson, validation point loss, train Pearson and the
graph penalty against epoch, and the readout table from
results/disjoint_embedding_readout.csv (written by disjoint_embedding_readout.py). Rank
1 to 3 runs of each DDP job carry no validation metrics and so do not draw. Re-running
creates a new report version at a new URL; the URL is printed.
"""

import os
import os.path as osp

import pandas as pd
import wandb_workspaces.reports.v2 as wr
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
ENTITY = "zhao-group"
PROJECT = "torchcell_025-solid-growth_equivariant_cell_graph_transformer"

LABELS = {
    "cgt_s0_q_kl_ctrl_016": "learnable table, control, 30 ep",
    "cgt_s0_q_kl_emb_017": "composite (promoter + CaLM + ProtT5 + terminator), 30 ep",
    "cgt_s0_q_kl_rand_018": "random 1,000-vector, matched control, 30 ep",
    "cgt_s0_q_kl_embfit_027": "composite + fitness head, 30 ep",
    "cgt_s0_q_kl_calm_020": "CaLM alone, 30 ep",
    "cgt_s0_q_kl_prot_021": "ProtT5 alone, 30 ep",
    "cgt_s0_q_kl_emb_022": "composite, 100 ep",
    "cgt_s0_q_kl_calm_023": "CaLM alone, 100 ep",
    "cgt_s0_q_kl_prot_024": "ProtT5 alone, 100 ep",
    "cgt_s0_q_kl_fudt_026": "promoter + terminator alone, 100 ep",
    "cgt_s0_q_kl_004": "learnable table, cosine schedule (job 1640), scale mark",
}


def main() -> None:
    df = pd.read_csv(osp.join(RESULTS_DIR, "disjoint_embedding_readout.csv"))
    line_titles = {
        row[
            "run_id"
        ]: f"{LABELS.get(row['config'], row['config'])} (seed {row['seed']})"
        for _, row in df.iterrows()
    }
    cols = [
        "config",
        "seed",
        "budget_epochs",
        "epochs_logged",
        "val_pearson_max",
        "val_pearson_max_epoch",
        "val_pearson_epoch29",
        "val_pearson_mean_ep10_29",
        "val_pearson_mean_ep60_99",
        "val_point_loss_epoch29",
        "train_pearson_last",
        "graph_reg_loss_last",
    ]
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df[cols].iterrows():
        md += (
            "| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |\n"
        )

    runset = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="disjoint split (split_Q), constant rate",
        filters="Tags in ['split_Q']",
    )

    def plot(title, y, log_y=False):
        return wr.LinePlot(
            title=title,
            x="epoch",
            y=[y],
            log_y=log_y,
            line_titles=line_titles,
            legend_position="south",
            smoothing_type="none",
            layout=wr.Layout(w=12, h=8),
        )

    report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="025 disjoint split: sequence embeddings against the learnable table",
        description=(
            "Query-pair-disjoint split, constant rate 2.5e-4, 30 or 100 epochs, one seed, "
            "embedding side parameter-matched to the 6,607 x 180 learnable table."
        ),
        blocks=[
            wr.H1("Design"),
            wr.P(
                "Every arm is the 010 soft-KL transformer on the 025 S0 triples with the "
                "query-pair-disjoint split (420 held-out pairs), AdamW 2.5e-4 with no "
                "schedule, normalizer fit on train, the CLS through the perturbation "
                "operator, interaction head only. The arms differ in the per-gene input: "
                "a free 180-vector (control) or a fixed sequence embedding projected to "
                "180 by a 2-layer MLP whose width matches the table's parameter count."
            ),
            wr.H1("Readout (disjoint_embedding_readout.py)"),
            wr.P(
                "max is the max over epochs run (biased upward); epoch 29 is the "
                "protocol's fixed reading; mean 10 to 29 is a window average."
            ),
            wr.MarkdownBlock(md),
            wr.H1("Curves"),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    plot(
                        "val/gene_interaction/Pearson", "val/gene_interaction/Pearson"
                    ),
                    plot("val/point_loss (z-scored MSE)", "val/point_loss"),
                    plot(
                        "train/gene_interaction/Pearson",
                        "train/gene_interaction/Pearson",
                    ),
                    plot(
                        "train/graph_reg_loss (log)", "train/graph_reg_loss", log_y=True
                    ),
                    plot("val/cls_pert_strain_sd", "val/cls_pert_strain_sd"),
                ],
            ),
        ],
    )
    report.save()
    print(report.url)


if __name__ == "__main__":
    main()
