# experiments/035-env-chemgen-vanacloig-cgt/scripts/train_vanacloig_cgt.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.train_vanacloig_cgt]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/train_vanacloig_cgt
"""Can the cell graph transformer predict Vanacloig 2022 for a compound it has not seen?

One run is one arm on one compound-cold fold. The cell graph transformer is used
unchanged: it encodes the 6,607 gene tokens, applies the perturbation operator for the
strain's four deletions (the queried gene and the three host deletions), and returns the
class token and the perturbed gene embeddings. The readout is built here, outside the
model, so no core module changes for a preliminary question:

    z_S = sum of the perturbed embeddings over the strain's four genes
    z_E = MLP(log1p(compound fingerprint))
    y   = MLP([h_CLS ; z_S ; z_E])

THREE ARMS, selected by ``arm``:

``cgt_genes``      [h_CLS ; z_S] and no environment. The floor: with no compound input
                   the best it can do is each gene's mean, which scores zero on the
                   centered target by construction.
``cgt_env``        [h_CLS ; z_S ; z_E]. The question.
``embedding_env``  [z_S ; z_E] with z_S summed from a plain embedding table, no
                   transformer and no graphs. It says whether the encoder earns anything
                   over a lookup table on this task.

THE SCORE is ``vanacloig_data.score_compounds``: per held-out compound, Spearman over
its genes, raw and centered, beside the compound's ceiling. The epoch reported is the
one with the best VALIDATION mean centered Spearman; the test compounds are scored at
every epoch for the curve and are never used to choose.

Writes ``results/runs/<arm>/fold<k>_seed<s>_{scores,history}.csv`` and logs to W&B.
"""

from __future__ import annotations

import os
import os.path as osp
import sys
import time
from types import SimpleNamespace
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import HeteroData

from torchcell.data.cell_data import to_cell_data
from torchcell.data.neo4j_cell import create_graph_from_gene_set
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

sys.path.insert(0, osp.dirname(__file__))
from vanacloig_data import (  # noqa: E402
    HOST_GENES,
    Fold,
    VanacloigCells,
    load_cells,
    make_folds,
    score_compounds,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
WANDB_MODE = os.getenv("WANDB_MODE")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "035-env-chemgen-vanacloig-cgt", "results")
ARMS = ("cgt_genes", "cgt_env", "embedding_env")


class EnvironmentReadout(nn.Module):
    """A scalar readout over the strain summary and, in two arms, the compound."""

    def __init__(
        self,
        arm: str,
        encoder: CellGraphTransformer | None,
        gene_num: int,
        hidden: int,
        feature_dim: int,
        dropout: float,
    ) -> None:
        """``encoder`` None selects the embedding table; ``arm`` decides the inputs."""
        super().__init__()
        self.arm = arm
        self.encoder = encoder
        self.table = nn.Embedding(gene_num, hidden) if encoder is None else None
        uses_environment = arm != "cgt_genes"
        self.environment = (
            nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )
            if uses_environment
            else None
        )
        width = hidden * ((2 if encoder is not None else 1) + int(uses_environment))
        self.head = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, cell_graph: HeteroData, strain: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``strain`` is [batch, 4] gene indices, ``features`` [batch, feature_dim]."""
        size, order = strain.shape
        if self.encoder is not None:
            batch = {
                "gene": SimpleNamespace(
                    perturbation_indices=strain.reshape(-1),
                    perturbation_indices_batch=torch.arange(
                        size, device=strain.device
                    ).repeat_interleave(order),
                )
            }
            _, out = self.encoder(cell_graph, batch)
            rows = torch.arange(size, device=strain.device).unsqueeze(1)
            z_s = out["H_genes_pert"][rows, strain].sum(dim=1)
            parts = [out["h_CLS"].unsqueeze(0).expand(size, -1), z_s]
            penalty = out["graph_reg_loss"]
        else:
            assert self.table is not None
            parts = [self.table(strain).sum(dim=1)]
            penalty = torch.zeros((), device=strain.device)
        if self.environment is not None:
            parts.append(self.environment(torch.log1p(features)))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1), penalty


def build_cell_graph(graph_names: list[str]) -> HeteroData:
    """The wildtype cell graph over the S288C gene set, as the dataset class builds it."""
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)
    assert multigraph is not None, "no gene graphs were named"
    multigraph.graphs["base"] = create_graph_from_gene_set(genome.gene_set)
    return to_cell_data(multigraph)


def strain_indices(cells: VanacloigCells, node_ids: list[str]) -> torch.Tensor:
    """[n_cells, 4]: the queried gene and the three host genes, as cell-graph indices."""
    position = {gene: i for i, gene in enumerate(node_ids)}
    query = np.array([position[g] for g in cells.genes])[cells.gene_of_cell]
    host = np.array([position[g] for g in HOST_GENES])
    return torch.tensor(
        np.concatenate([query[:, None], np.tile(host, (len(query), 1))], axis=1),
        dtype=torch.long,
    )


def cells_of(cells: VanacloigCells, compounds: list[int]) -> torch.Tensor:
    return torch.tensor(
        np.flatnonzero(np.isin(cells.compound_of_cell, compounds)), dtype=torch.long
    )


@torch.no_grad()
def predict(
    model: EnvironmentReadout,
    cell_graph: HeteroData,
    strain: torch.Tensor,
    features: torch.Tensor,
    index: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(index), batch_size):
        rows = index[start : start + batch_size]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y, _ = model(cell_graph, strain[rows].to(device), features[rows].to(device))
        out.append(y.float().cpu().numpy())
    return np.concatenate(out)


def evaluate(
    cells: VanacloigCells,
    fold: Fold,
    standardized: np.ndarray,
    mean: float,
    sd: float,
    has_environment: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score the validation and test compounds from a prediction for EVERY cell."""
    prediction = cells.matrix(standardized * sd + mean)
    frames = []
    summary: dict[str, float] = {}
    for split, compounds in (("val", fold.val), ("test", fold.test)):
        scores = score_compounds(cells, prediction, fold.train, compounds).assign(
            split=split
        )
        if not has_environment:
            # with no compound input the prediction is one value per gene, so its
            # centered form is numerical noise around zero and carries no score
            scores.loc[scores["target"] == "centered", ["spearman", "pearson"]] = np.nan
        frames.append(scores)
        for target, g in scores.groupby("target"):
            summary[f"{split}/{target}_spearman_mean"] = float(g["spearman"].mean())
            summary[f"{split}/{target}_spearman_median"] = float(g["spearman"].median())
            summary[f"{split}/{target}_pearson_median"] = float(g["pearson"].median())
    return pd.concat(frames, ignore_index=True), summary


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    assert cfg.arm in ARMS, f"arm must be one of {ARMS}, got {cfg.arm!r}"
    config: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if cfg.arm != "embedding_env" or cfg.gpu else "cpu")

    cells = load_cells(cfg.data.cell_table, cfg.data.embedding)
    fold = make_folds(
        len(cells.compounds), cfg.folds.n_folds, cfg.folds.n_val, cfg.folds.seed
    )[cfg.fold]
    cell_graph = build_cell_graph(list(cfg.cell_dataset.graphs))
    strain = strain_indices(cells, list(cell_graph["gene"].node_ids))
    features = torch.tensor(cells.compound_features, dtype=torch.float32)[
        torch.tensor(cells.compound_of_cell)
    ]
    index = {
        "train": cells_of(cells, fold.train),
        "val": cells_of(cells, fold.val),
        "test": cells_of(cells, fold.test),
    }
    train_y = cells.response[index["train"].numpy()]
    mean, sd = float(train_y.mean()), float(train_y.std(ddof=1))
    target = torch.tensor((cells.response - mean) / sd, dtype=torch.float32)
    print(
        f"arm {cfg.arm} fold {fold.fold} seed {cfg.seed}: "
        f"{len(fold.train)} train / {len(fold.val)} val / {len(fold.test)} test compounds, "
        f"{len(index['train']):,} / {len(index['val']):,} / {len(index['test']):,} cells",
        flush=True,
    )

    encoder = None
    if cfg.arm != "embedding_env":
        encoder = CellGraphTransformer(
            gene_num=cfg.model.gene_num,
            hidden_channels=cfg.model.hidden_channels,
            num_transformer_layers=cfg.model.num_transformer_layers,
            num_attention_heads=cfg.model.num_attention_heads,
            cell_graph=cell_graph,
            graph_regularization_config=config["model"]["graph_regularization"],
            perturbation_head_config=config["model"]["perturbation_head"],
            dropout=cfg.model.dropout,
            graph_reg_lambda=cfg.loss.graph_reg_lambda,
            node_embeddings=None,
            learnable_embedding_config=config["model"]["learnable_embedding"],
        )
    model = EnvironmentReadout(
        arm=cfg.arm,
        encoder=encoder,
        gene_num=cfg.model.gene_num,
        hidden=cfg.model.hidden_channels,
        feature_dim=features.shape[1],
        dropout=cfg.model.dropout,
    ).to(device)
    cell_graph = cell_graph.to(device)

    name = f"{cfg.arm}_fold{fold.fold}_seed{cfg.seed}"
    run = wandb.init(
        mode=WANDB_MODE,
        project=cfg.wandb.project,
        group=cfg.arm,
        name=name,
        tags=list(cfg.wandb.tags),
        config=config,
        dir=osp.join(DATA_ROOT, "wandb-experiments", "035-env-chemgen-vanacloig-cgt"),
    )
    run_dir = osp.join(RESULTS_DIR, cfg.results_subdir, cfg.arm)
    checkpoint_dir = osp.join(
        DATA_ROOT,
        "models/checkpoints/035-env-chemgen-vanacloig-cgt",
        cfg.results_subdir,
    )
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(run.dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )
    steps_per_epoch = int(np.ceil(len(index["train"]) / cfg.trainer.batch_size))
    if cfg.trainer.limit_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.trainer.limit_train_batches)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.optimizer.lr,
        total_steps=cfg.trainer.max_epochs * steps_per_epoch,
        pct_start=cfg.optimizer.warmup_fraction,
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    history = []
    best = {"score": -np.inf, "epoch": -1}
    best_scores = pd.DataFrame()
    for epoch in range(cfg.trainer.max_epochs):
        model.train()
        started = time.time()
        order = index["train"][torch.randperm(len(index["train"]), generator=generator)]
        loss_sum, penalty_sum = 0.0, 0.0
        for step in range(steps_per_epoch):
            rows = order[
                step * cfg.trainer.batch_size : (step + 1) * cfg.trainer.batch_size
            ]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                prediction, penalty = model(
                    cell_graph, strain[rows].to(device), features[rows].to(device)
                )
                mse = nn.functional.mse_loss(
                    prediction.float(), target[rows].to(device)
                )
                loss = mse + cfg.loss.graph_reg_lambda * penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            loss_sum += float(mse)
            penalty_sum += float(penalty)
        train_seconds = time.time() - started

        started = time.time()
        everything = predict(
            model,
            cell_graph,
            strain,
            features,
            torch.arange(len(target)),
            cfg.trainer.eval_batch_size,
            device,
        )
        standardized = {split: everything[index[split].numpy()] for split in index}
        scores, summary = evaluate(
            cells, fold, everything, mean, sd, has_environment=cfg.arm != "cgt_genes"
        )
        eval_seconds = time.time() - started
        row = {
            "epoch": epoch,
            "train/mse": loss_sum / steps_per_epoch,
            "train/graph_penalty": penalty_sum / steps_per_epoch,
            "val/mse": float(
                np.mean((standardized["val"] - target[index["val"]].numpy()) ** 2)
            ),
            "test/mse": float(
                np.mean((standardized["test"] - target[index["test"]].numpy()) ** 2)
            ),
            "lr": scheduler.get_last_lr()[0],
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            **summary,
        }
        history.append(row)
        wandb.log(row)
        print(
            f"epoch {epoch:3d}  train mse {row['train/mse']:.4f}  "
            f"penalty {row['train/graph_penalty']:.4f}  val mse {row['val/mse']:.4f}  "
            f"val centered rho {summary['val/centered_spearman_mean']:+.4f}  "
            f"test centered rho (median) {summary['test/centered_spearman_median']:+.4f}  "
            f"test raw rho (median) {summary['test/raw_spearman_median']:+.4f}  "
            f"train {train_seconds:.0f} s  eval {eval_seconds:.0f} s",
            flush=True,
        )

        # cgt_genes has no compound input, so its centered score is undefined (a constant
        # per gene after centering is noise around zero); it is selected on the raw score
        selector = "raw" if cfg.arm == "cgt_genes" else "centered"
        score = summary[f"val/{selector}_spearman_mean"]
        if np.isfinite(score) and score > best["score"]:
            best = {"score": score, "epoch": epoch}
            best_scores = scores.assign(
                arm=cfg.arm,
                fold=fold.fold,
                seed=cfg.seed,
                epoch=epoch,
                selected_on=f"val/{selector}_spearman_mean",
                n_train_compounds=len(fold.train),
            )
            torch.save(model.state_dict(), osp.join(checkpoint_dir, f"{name}.pt"))
        pd.DataFrame(history).to_csv(
            osp.join(run_dir, f"fold{fold.fold}_seed{cfg.seed}_history.csv"),
            index=False,
        )
        best_scores.to_csv(
            osp.join(run_dir, f"fold{fold.fold}_seed{cfg.seed}_scores.csv"), index=False
        )

    test = best_scores[best_scores["split"] == "test"]
    final = {
        f"final/test_{target_name}_spearman_median": float(g["spearman"].median())
        for target_name, g in test.groupby("target")
    }
    final["final/epoch"] = best["epoch"]
    wandb.log(final)
    print(f"selected epoch {best['epoch']} on validation: {final}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
