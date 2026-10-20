# experiments/030-solid-growth-multi/scripts/fixed_token_readout_030.py
# [[experiments.030-solid-growth-multi.scripts.fixed_token_readout_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/fixed_token_readout_030
r"""Score a trained 030 token checkpoint on the pinned triples under every token.

The per-entry dataset token tells the readout which source dataset a row came from.
On the trigenic validation set that is the screen (Kuzmin 2018 or Kuzmin 2020), and the
2018 pool was selected for negative digenic interactions, so the token can carry the
screen's selection shift into the pooled Pearson. This script measures how much: every
validation (or test) triple is read once under its own token, the number the training
curve reports, and once under each token of the vocabulary held fixed for every row,
the number a de novo triple would get. One encoder pass per batch; the token enters at
the readout only (``CellGraphTransformer.entry_readout``), so the fixed-token reads are
free.

Per-row predictions go to ``results/fixed_token_readout_030_<tag>_rows.parquet`` and the
summary to ``results/fixed_token_readout_030_<tag>.json``: pooled Pearson and Spearman,
within-screen Pearson (both series centered per screen, as the lower-order ladder
defines it), per-screen Pearson, and the mean prediction per screen under each token.
Genotypes are reduced to one value exactly as the trainer does (``_policy_reduce``: rows
of the lowest precedence rank present, averaged).

    python experiments/030-solid-growth-multi/scripts/fixed_token_readout_030.py \\
        --config cgt_030_s3_r_tok_embfit_003 --ckpt <...>/u28pgzde-last.ckpt --tag r_s0_ep183
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from scipy.stats import pearsonr, spearmanr
from torch_geometric.transforms import Compose

from torchcell.losses.point_dist_graph_reg import PointDistGraphReg
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.trainers.int_transformer_cell import (
    RegressionTask,
    token_precedence_rank,
)
from torchcell.transforms.coo_regression_to_classification import (
    COOInverseCompose,
    COOLabelNormalizationTransform,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import (  # noqa: E402
    EXPERIMENT,
    build_dataset,
    index_sha256,
    load_json_artifact,
    make_data_module,
    resolve_arm,
    results_dir,
)

SCREEN_OF_TOKEN = {
    "TmiKuzmin2018Dataset": "kuzmin2018",
    "TmiKuzmin2020Dataset": "kuzmin2020",
}


class TokenRead(BaseModel):
    """The scores of one way of reading the pinned triples."""

    token: str = Field(description="'own' or the vocabulary name held fixed")
    n_genotypes: int
    pearson: float
    spearman: float
    rmse: float
    pearson_within_screen: float
    pearson_kuzmin2018: float
    pearson_kuzmin2020: float
    mean_pred_kuzmin2018: float
    mean_pred_kuzmin2020: float


class FixedTokenReadout(BaseModel):
    """What one checkpoint measured on one split under every token."""

    config_name: str
    checkpoint: str
    checkpoint_epoch: int
    split: str
    seed: int
    n_rows: int
    n_genotypes: int
    n_genotypes_both_screens: int
    mean_target_kuzmin2018: float
    mean_target_kuzmin2020: float
    dataset_vocabulary: list[str]
    reads: list[TokenRead]


def _score(y: np.ndarray, yhat: np.ndarray, screen: np.ndarray) -> dict[str, float]:
    """Pooled, within-screen and per-screen scores of ``yhat`` against ``y``."""
    assert np.isfinite(yhat).all(), "a prediction is not finite"
    yc = y.copy()
    hc = yhat.copy()
    per: dict[str, float] = {}
    means: dict[str, float] = {}
    for s in ("kuzmin2018", "kuzmin2020"):
        m = screen == s
        means[s] = float(yhat[m].mean()) if m.any() else float("nan")
        yc[m] -= y[m].mean()
        hc[m] -= yhat[m].mean()
        per[s] = float(pearsonr(y[m], yhat[m])[0]) if m.sum() >= 3 else float("nan")
    return {
        "pearson": float(pearsonr(y, yhat)[0]),
        "spearman": float(spearmanr(y, yhat)[0]),
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "pearson_within_screen": float(pearsonr(yc, hc)[0]),
        "pearson_kuzmin2018": per["kuzmin2018"],
        "pearson_kuzmin2020": per["kuzmin2020"],
        "mean_pred_kuzmin2018": means["kuzmin2018"],
        "mean_pred_kuzmin2020": means["kuzmin2020"],
    }


def policy_reduce(rows: pd.DataFrame, vocabulary: list[str]) -> pd.DataFrame:
    """One row per genotype: the rows of its lowest precedence rank, averaged.

    The trainer's ``_policy_reduce`` in pandas. The screen of the reduced genotype is
    the screen of the kept rows (a triple in both screens is read as Kuzmin 2018).
    """
    rank = np.array([token_precedence_rank(n) for n in vocabulary])
    rows = rows.assign(rank=rank[rows["token"].to_numpy()])
    best = rows.groupby("genotype")["rank"].transform("min")
    kept = rows[rows["rank"] == best]
    pred_cols = [c for c in kept.columns if c.startswith("pred_")]
    agg = kept.groupby("genotype").agg(
        {
            "target": "mean",
            "order": "first",
            "token": "first",
            **{c: "mean" for c in pred_cols},
        }
    )
    agg["screen"] = [SCREEN_OF_TOKEN[vocabulary[int(t)]] for t in agg["token"]]
    return agg.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="cgt_030_s3_r_tok_embfit_003")
    ap.add_argument("--ckpt", required=True, help="a Lightning *-last.ckpt of the arm")
    ap.add_argument("--tag", required=True, help="suffix of the result files")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-batches", type=int, default=0, help="debug: stop early")
    args = ap.parse_args()

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    conf_dir = osp.abspath(osp.join(osp.dirname(__file__), "..", "conf"))
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg_omega = compose(config_name=args.config)
    cfg: dict[str, Any] = OmegaConf.to_container(cfg_omega, resolve=True)  # type: ignore[assignment]

    arm = resolve_arm(cfg["subset"])
    train_sha = index_sha256(arm.train_records())
    stats_file = load_json_artifact(str(cfg["transforms"]["fit_stats"]))
    if stats_file["train_index_sha256"] != train_sha:
        raise ValueError(
            "the normalization constants were fitted on another training set"
        )
    print(
        f"arm: pool={len(arm.pool)} pinned={ ({k: len(v) for k, v in arm.pinned.items()}) } "
        f"vocabulary={len(arm.dataset_vocabulary)}",
        flush=True,
    )

    with open(
        osp.join(experiment_root, EXPERIMENT, "queries/001_multi_measurement.cql")
    ) as f:
        query = f.read()
    phenotype_labels = list(cfg["cell_dataset"]["phenotype_labels"])
    t0 = time.time()
    dataset, _genome, _graph, node_embeddings = build_dataset(
        data_root=data_root,
        dataset_root=osp.join(data_root, cfg["dataset"]["root_rel"]),
        query=query,
        graph_names=list(cfg["cell_dataset"]["graphs"]),
        node_embedding_names=list(cfg["cell_dataset"].get("node_embeddings", [])),
        phenotype_labels=phenotype_labels,
        dataset_vocabulary=arm.dataset_vocabulary,
    )
    print(f"dataset: {len(dataset)} records in {time.time() - t0:.0f}s", flush=True)
    norm_transform = COOLabelNormalizationTransform(
        dataset,
        cfg["transforms"]["forward_transform"]["normalization"],
        fit_stats=stats_file["stats"],
    )
    dataset.transform = Compose([norm_transform])
    inverse_transform = COOInverseCompose([norm_transform])

    dm_cfg = dict(cfg["data_module"])
    dm_cfg["val_batch_size"] = args.val_batch_size
    dm_cfg["num_workers"] = args.num_workers
    dm_cfg["persistent_workers"] = False
    data_module = make_data_module(
        dataset, arm, args.seed, dm_cfg, ["perturbation_indices", "phenotype_values"]
    )
    loader = (
        data_module.val_dataloader()
        if args.split == "val"
        else data_module.test_dataloader()
    )
    assert not isinstance(loader, list)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = cfg["model"]
    token_cfg = dict(model_cfg["dataset_token"])
    token_cfg["vocab_size"] = len(arm.dataset_vocabulary)
    loss_cfg = cfg["regression_task"]["loss"]
    model = CellGraphTransformer(
        gene_num=model_cfg["gene_num"],
        hidden_channels=model_cfg["hidden_channels"],
        num_transformer_layers=model_cfg["num_transformer_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        cell_graph=dataset.cell_graph,
        graph_regularization_config=model_cfg["graph_regularization"],
        perturbation_head_config=model_cfg["perturbation_head"],
        dropout=model_cfg["dropout"],
        graph_reg_lambda=float(loss_cfg["graph_regularization"]["lambda"]),
        node_embeddings=node_embeddings,
        learnable_embedding_config=model_cfg.get("learnable_embedding"),
        attention_mask_config=model_cfg.get("attention_mask"),
        heads_config=model_cfg["heads"],
        perturb_cls=bool(model_cfg.get("perturb_cls", False)),
        perturbation_head_cls=str(model_cfg.get("perturbation_head_cls", "wildtype")),
        dataset_token=token_cfg,
    )
    task = RegressionTask(
        model=model,
        cell_graph=dataset.cell_graph,
        optimizer_config=cfg["regression_task"]["optimizer"],
        lr_scheduler_config=cfg["regression_task"]["lr_scheduler"],
        batch_size=cfg["data_module"]["batch_size"],
        clip_grad_norm=cfg["regression_task"]["clip_grad_norm"],
        clip_grad_norm_max_norm=cfg["regression_task"]["clip_grad_norm_max_norm"],
        plot_sample_ceiling=cfg["regression_task"]["plot_sample_ceiling"],
        loss_func=PointDistGraphReg(
            point_estimator=loss_cfg.get("point_estimator"),
            distribution_loss=loss_cfg.get("distribution_loss"),
            graph_regularization=loss_cfg.get("graph_regularization"),
            buffer=loss_cfg.get("buffer"),
            ddp=loss_cfg.get("ddp"),
        ),
        grad_accumulation_schedule=cfg["regression_task"]["grad_accumulation_schedule"],
        device=device,
        inverse_transform=inverse_transform,
        plot_every_n_epochs=cfg["regression_task"]["plot_every_n_epochs"],
        plot_edge_recovery_every_n_epochs=cfg["regression_task"][
            "plot_edge_recovery_every_n_epochs"
        ],
        plot_transformer_diagnostics_every_n_epochs=cfg["regression_task"][
            "plot_transformer_diagnostics_every_n_epochs"
        ],
        execution_mode="inference",
        fitness_lambda=float(cfg["regression_task"]["fitness_lambda"]),
        gradient_probe_epochs=cfg["regression_task"].get("gradient_probe_epochs"),
        per_order_metrics=False,
        per_entry=True,
        dataset_vocabulary=arm.dataset_vocabulary,
        essentiality_eval=None,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = task.load_state_dict(ckpt["state_dict"], strict=False)
    bad = [k for k in missing if k.startswith("model.")] + list(unexpected)
    if bad:
        raise ValueError(f"checkpoint does not match the model: {bad[:10]}")
    print(
        f"loaded {args.ckpt} (epoch {ckpt['epoch']}, step {ckpt['global_step']}); "
        f"non-model keys not in the checkpoint: {missing}",
        flush=True,
    )
    task = task.to(device).eval()

    vocabulary = arm.dataset_vocabulary
    n_tokens = len(vocabulary)
    frames: list[pd.DataFrame] = []
    offset = 0
    t0 = time.time()
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"),
    ):
        for bi, batch in enumerate(loader):
            if args.limit_batches and bi >= args.limit_batches:
                break
            batch = batch.to(device)
            gene = batch["gene"]
            batch_size = task._get_batch_size(batch)
            preds, reps = task(
                batch,
                return_attention=False,
                entry_batch=gene.phenotype_values_batch,
                entry_dataset=gene.phenotype_dataset_indices,
            )
            rows, _vals, orig, toks, sel = task._entry_rows(batch, "gene_interaction")
            own = task._inverse_scalar(preds[sel].float(), "gene_interaction").view(-1)
            order = task._perturbation_order(batch, batch_size)[rows]
            frame = {
                "genotype": (offset + rows).cpu().numpy(),
                "token": toks.cpu().numpy(),
                "order": order.cpu().numpy(),
                "target": orig.float().cpu().numpy(),
                "pred_own": own.cpu().numpy(),
            }
            for t in range(n_tokens):
                fixed_pred, _ = task.model.entry_readout(
                    reps, batch, rows, torch.full_like(toks, t)
                )
                frame[f"pred_{vocabulary[t]}"] = (
                    task._inverse_scalar(fixed_pred.float(), "gene_interaction")
                    .view(-1)
                    .cpu()
                    .numpy()
                )
            frames.append(pd.DataFrame(frame))
            offset += batch_size
            if bi % 20 == 0:
                print(
                    f"batch {bi}/{len(loader)} rows={sum(len(f) for f in frames):,} {time.time() - t0:.0f}s",
                    flush=True,
                )

    rows_df = pd.concat(frames, ignore_index=True)
    assert set(rows_df["order"].unique()) == {3}, f"orders {rows_df['order'].unique()}"
    screens = {vocabulary[int(t)] for t in rows_df["token"].unique()}
    assert screens <= set(SCREEN_OF_TOKEN), (
        f"unexpected tokens on the triples: {screens}"
    )
    out_rows = osp.join(
        results_dir(), f"fixed_token_readout_030_{args.tag}_rows.parquet"
    )
    rows_df.to_parquet(out_rows, index=False)

    reduced = policy_reduce(rows_df, vocabulary)
    both = rows_df.groupby("genotype")["token"].nunique()
    y = reduced["target"].to_numpy()
    screen = reduced["screen"].to_numpy()
    reads = [
        TokenRead(
            token="own",
            n_genotypes=len(reduced),
            **_score(y, reduced["pred_own"].to_numpy(), screen),
        )
    ]
    for name in vocabulary:
        reads.append(
            TokenRead(
                token=name,
                n_genotypes=len(reduced),
                **_score(y, reduced[f"pred_{name}"].to_numpy(), screen),
            )
        )
    summary = FixedTokenReadout(
        config_name=args.config,
        checkpoint=args.ckpt,
        checkpoint_epoch=int(ckpt["epoch"]),
        split=args.split,
        seed=args.seed,
        n_rows=len(rows_df),
        n_genotypes=len(reduced),
        n_genotypes_both_screens=int((both > 1).sum()),
        mean_target_kuzmin2018=float(y[screen == "kuzmin2018"].mean()),
        mean_target_kuzmin2020=float(y[screen == "kuzmin2020"].mean()),
        dataset_vocabulary=vocabulary,
        reads=reads,
    )
    out_json = osp.join(results_dir(), f"fixed_token_readout_030_{args.tag}.json")
    with open(out_json, "w") as f:
        json.dump(summary.model_dump(), f, indent=2)
    print(
        f"\n{summary.split}: {summary.n_rows:,} rows, {summary.n_genotypes:,} genotypes, "
        f"{summary.n_genotypes_both_screens} in both screens"
    )
    print(
        f"mean target 2018 {summary.mean_target_kuzmin2018:+.4f}  2020 {summary.mean_target_kuzmin2020:+.4f}"
    )
    print(
        f"{'token':<40} {'pearson':>8} {'within':>8} {'r2018':>8} {'r2020':>8} {'mu2018':>8} {'mu2020':>8}"
    )
    for r in reads:
        print(
            f"{r.token:<40} {r.pearson:8.4f} {r.pearson_within_screen:8.4f} {r.pearson_kuzmin2018:8.4f} "
            f"{r.pearson_kuzmin2020:8.4f} {r.mean_pred_kuzmin2018:+8.4f} {r.mean_pred_kuzmin2020:+8.4f}"
        )
    print(f"wrote {out_rows}\nwrote {out_json}")


if __name__ == "__main__":
    main()
