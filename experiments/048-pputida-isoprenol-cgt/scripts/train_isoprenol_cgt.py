# experiments/048-pputida-isoprenol-cgt/scripts/train_isoprenol_cgt.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.train_isoprenol_cgt]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/train_isoprenol_cgt
"""Train the P. putida isoprenol CGT with the metabolic module.

The entry point for experiment 048. The data objects it consumes are assembled by
``pputida_cgt_data.py`` beside this file; this file is the model, the loss, the
splits it trains on and the table it reports. Run it with ``--print-data`` to see
the cell graph, one record and one batch before any training starts, which is the
quickest way to check what the model is actually being handed.

Two split regimes, both named in the MVP contract
([[experiments.048-pputida-isoprenol-cgt.p-putida-cgt-mvp-goal]]):

``--split design``
    Random 0.8/0.1/0.1 over CRISPRi **designs**, never over records. A design
    measured in two DBTL cycles is two records, so a record-level split would
    put the same guide array in train and test.

``--split cycle``
    Train on cycles 0 to 4, select on cycle 5, test on cycle 6. This is the
    "predict the next cycle from the previous ones" arm. Cycle 6 is the
    campaign's last and widest cycle, so cycle 5's score is reported beside it
    rather than instead of it.

``--fidelity-tier`` restricts training to cycles 3 to 6, the post-repair tier.
Carruthers reports 40% failure in DBTL2 from low dCas9 expression, a vector
repair that removed a redundant linker, a 25% fidelity improvement, and that
their own models after DBTL3 were trained only on post-repair data. Whether that
restriction helps us is a measurement, not an assumption, so it is an arm.

Every run reports the lower-order nulls on the same table as the model: the
train-mean constant, ridge on the guide indicator, and gradient-boosted trees on
the same indicator. On a 465-record campaign with 401 designs, a transformer that
does not beat boosted trees on a multi-hot guide vector has not earned its
parameters. That has happened before on this stack.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from pputida_cgt_data import (  # noqa: E402
    build_datamodule,
    build_dataset,
    campaign_table,
    cycle_splits,
    design_splits,
)

from torchcell.metabolism.bigg_gem import BiggGEM  # noqa: E402
from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.flux_layer import FluxLayer, FluxLayerConfig  # noqa: E402
from torchcell.models.cell_graph_transformer_metabolism import (  # noqa: E402
    CellGraphTransformerMetabolism,
)

load_dotenv()

log = logging.getLogger(__name__)

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt", "results")

HEAD_NAME = "titer"

#: The isoprenol route's own metabolites, in the pinned isoprenol-augmented
#: iJN1463. These are the ``precursor_indices`` the mechanistic head reads: the
#: mevalonate/IPP-bypass intermediates from acetyl-CoA to secreted isoprenol.
#:
#:   2 accoa_c -> aacoa_c -> hmgcoa_c -> mva_c -> 5pmva_c -> 3mbut3e1yp_c
#:   -> ip_c -> ip_p -> ip_e
#:
#: Acetyl-CoA is kept because redirecting central carbon into it is what the
#: campaign's CRISPRi targets are for. The energy and redox currency of those
#: same reactions (atp, adp, coa, h, h2o, pi, nadp, nadph, co2) is left out: it
#: is shared with the whole network, so its turnover says nothing specific about
#: this route.
ISOPRENOL_ROUTE_METABOLITES: tuple[str, ...] = (
    "accoa_c",
    "aacoa_c",
    "hmgcoa_c",
    "mva_c",
    "5pmva_c",
    "3mbut3e1yp_c",
    "ip_c",
    "ip_p",
    "ip_e",
)

#: The isoprenol secretion reaction. Its flux is the GEM's own statement of
#: product formation, and the dense term of the mechanistic head can read it.
ISOPRENOL_EXCHANGE = "EX_ip_e"


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch, and make cudnn deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def titer_targets(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Each sample's titer as ``[B, 1]``, with a mask of which rows carry one.

    The values are averaged per sample through ``phenotype_values_batch``, not
    assigned. The difference is not cosmetic: three samples of this build
    aggregate more than one record (one pair and two triples, all Source Data
    panel strains), and plain indexed assignment would keep whichever replicate
    came last while the null baselines are fit on the group mean, so the model
    and its baselines would be scored against different numbers. The panel
    strains are excluded from every campaign split, so this is latent today and
    would have become a silent defect the moment a second phenotype or the panel
    records entered a split.

    The mask exists because a zero is a legitimate titer here: the record table's
    minimum is 0.00 ug/mL, so a zero-filled row cannot be read as missing.
    """
    gene = batch["gene"]
    values = gene.phenotype_values.float()
    rows = gene.phenotype_values_batch.long()
    size = int(batch.num_graphs)
    totals = torch.zeros(size, device=values.device)
    counts = torch.zeros(size, device=values.device)
    totals.index_add_(0, rows, values)
    counts.index_add_(0, rows, torch.ones_like(values))
    mask = counts > 0
    out = torch.zeros(size, device=values.device)
    out[mask] = totals[mask] / counts[mask]
    return out.unsqueeze(1), mask.unsqueeze(1)


def guide_indicator(dataset: Any, indices: list[int]) -> np.ndarray:
    """A multi-hot matrix over genes: 1 where a sample's guide targets that gene.

    This is the feature the lower-order nulls get. It is exactly the information
    the CGT receives about the genotype, minus the graph and the sequence, which
    is what makes the comparison a fair accounting of what the structure buys.
    """
    n_genes = int(dataset.cell_graph["gene"].num_nodes)
    matrix = np.zeros((len(indices), n_genes), dtype=np.float32)
    for row, index in enumerate(indices):
        sample = dataset[index]
        matrix[row, sample["gene"].perturbation_indices.numpy()] = 1.0
    return matrix


def lower_order_nulls(
    dataset: Any, splits: dict[str, list[int]], targets: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Score the train-mean constant, ridge and boosted trees on the indicator."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    features = {name: guide_indicator(dataset, idx) for name, idx in splits.items()}
    out: dict[str, dict[str, float]] = {}

    mean = float(targets["train"].mean())
    out["train_mean"] = {
        split: float(np.mean((targets[split] - mean) ** 2)) for split in ("val", "test")
    }
    out["train_mean"].update({f"{s}_pearson": float("nan") for s in ("val", "test")})

    for name, model in (
        ("ridge", Ridge(alpha=1.0)),
        ("boosted_trees", HistGradientBoostingRegressor(max_iter=300, random_state=0)),
    ):
        model.fit(features["train"], targets["train"])
        scores: dict[str, float] = {}
        for split in ("val", "test"):
            prediction = model.predict(features[split])
            scores[split] = float(np.mean((targets[split] - prediction) ** 2))
            scores[f"{split}_pearson"] = float(
                pearsonr(targets[split], prediction).statistic
            )
        out[name] = scores
    return out


def build_flux_layer(
    dataset: Any, hidden: int, model_id: str, device: torch.device
) -> tuple[FluxLayer, list[int], dict[str, Any]]:
    """Build the flux layer from the pinned isoprenol GEM.

    This is where the metabolism stops being decoration. Without a flux layer the
    cell graph still CARRIES the reaction and metabolite stores, but on this
    branch nothing in the forward pass reads them, so the metabolic module would
    contribute nothing to a prediction. With one, the gene representation sets a
    per-gene availability, availability becomes a per-reaction capacity through
    the GPR, and a feasible flux vector comes out, which the head then reads.

    ``with_thermo=False`` because the thermodynamic tables are yeast-GEM files:
    the loader reads ``model_metDeltaG.csv`` and ``model_rxnDeltaG.csv`` from a
    yeast-GEM checkout, and no such table ships with a BiGG model. Supplying
    P. putida deltaG values is a sourcing job, not a default to invent, so the
    thermodynamic term is off and said to be off.

    ``model_gene_ids`` is the cell graph's node id list, in token order, because
    the layer maps its own GEM genes onto transformer tokens by position.
    """
    gem_source = BiggGEM(model_id)
    gem = build_gem_tensors(gem_source.model, with_thermo=False)
    gene_ids = list(dataset.cell_graph["gene"].node_ids)
    layer = FluxLayer(
        gem=gem, model_gene_ids=gene_ids, config=FluxLayerConfig(hidden_dim=hidden)
    ).to(device)

    met_index = {name: i for i, name in enumerate(gem.met_ids)}
    missing = [m for m in ISOPRENOL_ROUTE_METABOLITES if m not in met_index]
    if missing:
        raise ValueError(
            f"{model_id} does not carry the isoprenol route metabolites {missing}; "
            f"the mechanistic head cannot name a precursor the model lacks"
        )
    precursors = [met_index[m] for m in ISOPRENOL_ROUTE_METABOLITES]
    info = {
        "model_id": model_id,
        "n_reactions": gem.n_reactions,
        "n_metabolites": gem.n_metabolites,
        "biomass_reaction": gem.rxn_ids[gem.biomass_index],
        "n_exchanges": 0 if gem.exchange_indices is None else len(gem.exchange_indices),
        "n_catalytic_units": gem.catalytic_units.n_units,
        "n_multigene_units": gem.catalytic_units.n_multigene_units,
        "n_reactions_with_gpr": gem.catalytic_units.n_reactions_with_gpr,
        "gem_genes": len(gem.catalytic_units.gene_ids),
        "isoprenol_exchange_index": gem.rxn_ids.index(ISOPRENOL_EXCHANGE),
        "precursor_metabolites": list(ISOPRENOL_ROUTE_METABOLITES),
        "thermo": False,
    }
    return layer, precursors, info


def build_model(
    dataset: Any,
    hidden: int,
    layers: int,
    heads: int,
    dropout: float,
    device: torch.device,
    flux_layer: FluxLayer | None = None,
    precursor_indices: list[int] | None = None,
) -> CellGraphTransformerMetabolism:
    """Build the CGT with one scalar titer head.

    ``use_gene_pool`` and ``use_pert_pool`` are both left on, and that is not
    cosmetic. ``ProductScalarHead`` broadcasts the CLS token across the batch, so
    the CLS contributes the same vector to every strain in a batch; with both
    pools off the head is a constant predictor by construction. The strain enters
    through the mean over the perturbed gene representation and through the pool
    over the guide-targeted genes.

    ``attention_mask_config`` is left ``None``. The yeast backbone names nine
    relations and gives one attention head to each; this cell graph has six
    STRING channels and no physical, regulatory or TFLink layer, because no
    bacterial counterpart of those exists in the tier yet. Naming a relation the
    graph does not carry raises, so the mask stays off until the channel set is
    settled.
    """
    if flux_layer is not None:
        if precursor_indices is None:
            raise ValueError(
                "a flux_layer was supplied without precursor_indices; the "
                "mechanistic head has to be told which metabolites it reads"
            )
        head_spec: dict[str, Any] = {
            "kind": "flux_scalar",
            "output_dim": 1,
            "param_dim": 1,
            "precursor_indices": precursor_indices,
            "use_dense_flux": True,
        }
    else:
        head_spec = {
            "kind": "scalar",
            "output_dim": 1,
            "param_dim": 1,
            "use_gene_pool": True,
            "use_pert_pool": True,
            "dropout": dropout,
        }
    return CellGraphTransformerMetabolism(
        gene_num=int(dataset.cell_graph["gene"].num_nodes),
        hidden_channels=hidden,
        num_transformer_layers=layers,
        num_attention_heads=heads,
        cell_graph=dataset.cell_graph,
        dropout=dropout,
        node_embeddings=dataset.node_embedding_datasets,
        learnable_embedding_config={"enabled": False},
        perturbation_head_config={"enabled": False},
        heads_config={HEAD_NAME: head_spec},
        flux_layer=flux_layer,
    ).to(device)


def evaluate(
    model: nn.Module,
    cell_graph: Any,
    loader: Any,
    device: torch.device,
    mean: float,
    std: float,
) -> dict[str, float]:
    """Return MSE in standardized units plus Pearson and Spearman in mg/L."""
    model.eval()
    predictions: list[float] = []
    observed: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target, mask = titer_targets(batch)
            if not bool(mask.all()):
                raise ValueError(
                    "a batch carried a sample with no titer; every record of "
                    "this build has one, so this is a collation change"
                )
            _, reps = model(cell_graph, batch)
            prediction = reps["head_outputs"][HEAD_NAME]
            predictions.extend(prediction.view(-1).cpu().tolist())
            observed.extend(target.view(-1).cpu().tolist())
    predicted_raw = np.asarray(predictions) * std + mean
    observed_raw = np.asarray(observed)
    standardized = (observed_raw - mean) / std
    return {
        "mse": float(np.mean((standardized - np.asarray(predictions)) ** 2)),
        "pearson": float(pearsonr(observed_raw, predicted_raw).statistic),
        "spearman": float(spearmanr(observed_raw, predicted_raw).statistic),
        "n": len(observed),
    }


def main() -> None:
    """Build the data, train, and write the result table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("design", "cycle"), default="design")
    parser.add_argument("--fidelity-tier", action="store_true")
    parser.add_argument("--holdout-cycle", type=int, default=6)
    parser.add_argument("--val-cycle", type=int, default=5)
    parser.add_argument("--embeddings", default="one_hot")
    parser.add_argument("--no-metabolism", action="store_true")
    parser.add_argument("--flux", action="store_true")
    parser.add_argument("--gem-model", default="pputida_isoprenol")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-data", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--tag", default="mvp")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(RESULTS, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_names = tuple(n for n in args.embeddings.split(",") if n)
    dataset = build_dataset(
        embedding_names=embedding_names, use_metabolism=not args.no_metabolism
    )
    table = campaign_table(dataset)

    if args.split == "design":
        splits = design_splits(table, seed=args.seed)
    else:
        splits = cycle_splits(
            table, holdout_cycle=args.holdout_cycle, val_cycle=args.val_cycle
        )

    if args.fidelity_tier:
        tier = set(table.loc[table.cycle >= 3, "index"])
        splits["train"] = [i for i in splits["train"] if i in tier]

    module = build_datamodule(
        dataset,
        splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        cache_tag=f"{args.split}_{args.tag}",
    )

    cell_graph = dataset.cell_graph.to(device)
    flux_layer = None
    precursors = None
    flux_info: dict[str, Any] | None = None
    if args.flux:
        if args.no_metabolism:
            raise SystemExit(
                "--flux needs the metabolic stores in the cell graph; drop "
                "--no-metabolism"
            )
        flux_layer, precursors, flux_info = build_flux_layer(
            dataset, args.hidden, args.gem_model, device
        )
        log.info(f"flux layer: {flux_info}")
    model = build_model(
        dataset,
        args.hidden,
        args.layers,
        args.heads,
        args.dropout,
        device,
        flux_layer=flux_layer,
        precursor_indices=precursors,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Named for what the model IS, because the alternative is a table that says
    # "CGT + met" for a model with no metabolism in it. Measured: with no flux
    # layer, deleting every reaction and metabolite store from the cell graph
    # moves the head output by 5.2e-08, which is exactly the same-input
    # nondeterminism floor. The metabolic stores are carried but unread, because
    # the blocks that would read them are not on this branch, so the only way
    # metabolism enters a prediction here is through the flux layer.
    model_label = "CGT + flux met" if args.flux else "CGT, pooled head"
    metabolism_role = (
        "flux_layer reads the GEM; the cell graph's rmr and gpr stores are "
        "carried but not read by the forward pass"
        if args.flux
        else "carried in the cell graph and NOT read by the forward pass"
    )

    if args.print_data:
        print("\n=== cell graph ===")
        print(dataset.cell_graph)
        print("\n=== dataset[0] ===")
        print(dataset[0])
        print("\n=== one training batch ===")
        print(next(iter(module.train_dataloader())))
        print(f"\ntrainable parameters {n_params:,}")

    targets_by_split = {
        name: np.asarray(
            [float(table.loc[table["index"] == i, "titer"].iloc[0]) for i in idx]
        )
        for name, idx in splits.items()
    }
    mean = float(targets_by_split["train"].mean())
    std = float(targets_by_split["train"].std())

    nulls = lower_order_nulls(dataset, splits, targets_by_split)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project="torchcell_048-pputida-isoprenol-cgt",
            name=f"{args.split}-{args.tag}-seed{args.seed}",
            config=vars(args)
            | {"n_params": n_params, "splits": {k: len(v) for k, v in splits.items()}},
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()

    best = {"val_mse": float("inf"), "epoch": -1}
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        seen = 0
        for batch in module.train_dataloader():
            batch = batch.to(device)
            raw, mask = titer_targets(batch)
            if not bool(mask.all()):
                raise ValueError(
                    "a training batch carried a sample with no titer; every "
                    "record of this build has one"
                )
            target = (raw - mean) / std
            _, reps = model(cell_graph, batch)
            prediction = reps["head_outputs"][HEAD_NAME]
            loss = loss_fn(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(batch.num_graphs)
            seen += int(batch.num_graphs)

        train_mse = total / max(seen, 1)
        val = evaluate(model, cell_graph, module.val_dataloader(), device, mean, std)
        row = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val["mse"],
            "val_pearson": val["pearson"],
            "val_spearman": val["spearman"],
        }
        history.append(row)
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "loss/train": train_mse,
                    "loss/val": val["mse"],
                    "val/pearson": val["pearson"],
                    "val/spearman": val["spearman"],
                },
                step=epoch,
            )
        if val["mse"] < best["val_mse"]:
            best = {"val_mse": val["mse"], "epoch": epoch, **val}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best["epoch"] >= args.patience:
            log.info(
                f"stopping at epoch {epoch}, no val improvement for {args.patience}"
            )
            break
        if epoch % 10 == 0:
            log.info(
                f"epoch {epoch:>3}  train_mse {train_mse:.4f}  "
                f"val_mse {val['mse']:.4f}  val_r {val['pearson']:+.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, cell_graph, module.test_dataloader(), device, mean, std)
    val = evaluate(model, cell_graph, module.val_dataloader(), device, mean, std)

    report = {
        "args": vars(args),
        "device": str(device),
        "n_params": n_params,
        "model_label": model_label,
        "metabolism_role": metabolism_role,
        "val_is_selection_set": True,
        "mse_units": "standardized by the train split mean and sd",
        "flux": flux_info,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "train_titer_mean": mean,
        "train_titer_std": std,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "seconds": time.time() - started,
        "cgt": {"val": val, "test": test},
        "lower_order_nulls": nulls,
        "history": history,
    }
    out = osp.join(RESULTS, f"train_{args.split}_{args.tag}_seed{args.seed}.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== result table ===")
    print("MSE is in standardized units; val is the SELECTION set, not held out.")
    print(f"{'model':<22}{'val MSE':>10}{'val r':>9}{'test MSE':>10}{'test r':>9}")
    print(
        f"{model_label:<22}{val['mse']:>10.4f}{val['pearson']:>+9.3f}"
        f"{test['mse']:>10.4f}{test['pearson']:>+9.3f}"
    )
    for name, scores in nulls.items():
        val_r = scores.get("val_pearson", float("nan"))
        test_r = scores.get("test_pearson", float("nan"))
        print(
            f"{name:<22}{scores['val'] / std**2:>10.4f}{val_r:>+9.3f}"
            f"{scores['test'] / std**2:>10.4f}{test_r:>+9.3f}"
        )
    print(f"\nbest epoch {best['epoch']} of {len(history)} run")
    if run is not None:
        for split_name, scores in (("val", val), ("test", test)):
            for metric, value in scores.items():
                run.summary[f"{split_name}/{metric}"] = value
        run.summary["best_epoch"] = best["epoch"]
        run.finish()
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
