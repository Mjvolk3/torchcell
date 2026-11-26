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
train-mean constant and ORDINARY LEAST SQUARES on the multi-hot guide vector.
OLS is the standard baseline across this program's experiments because it has no
hyperparameter: ridge's alpha and a tree ensemble's depth and iteration count are
choices somebody makes, so a model that beats them invites the question of
whether the baseline was tuned as hard as the model. OLS removes the question.
On a 465-record campaign with 401 designs, a transformer that does not beat a
linear fit on the guide identity has not earned its parameters, and that has
happened before on this stack.
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
    """Score the train-mean constant and ORDINARY LEAST SQUARES on the indicator.

    **The baseline is OLS and nothing else, on purpose.** Ridge and
    gradient-boosted trees used to be here, and both carry settings somebody
    chose: ridge's ``alpha`` and the trees' ``max_iter`` and depth. Every model
    comparison then invites the question of whether the baseline was tuned as
    hard as the model, and on a 465-record campaign that question has no cheap
    answer. OLS has no hyperparameter at all, so the comparison is the same
    comparison in every experiment and nobody has to defend a baseline's
    configuration.

    **The indicator is restricted to the columns that vary in training.** It is
    built over all 5,729 genes, of which only the campaign's ~121 CRISPRi
    targets are ever nonzero, so the full matrix has 5,729 columns against 347
    training rows and OLS is underdetermined: ``lstsq`` would return a
    minimum-norm solution, which is a regularizer by the back door and exactly
    the thing OLS was chosen to avoid. A column that is constant across the
    training rows carries no information about the fit, so dropping it changes
    no prediction and makes the problem well posed. The surviving count is
    reported so a reader can see the design matrix's shape.

    Every null is scored with the same :func:`regression_metrics` the model is, so
    the trial's objective (Spearman) is comparable across the whole table and not
    only the MSE column. The constant predictor has no rank information at all, so
    its Spearman and Pearson are undefined rather than zero: a zero would read as
    "uncorrelated", and the truth is that the statistic does not exist for a
    sequence with no variance. They come back as NaN.

    The flat ``val``/``test``/``val_pearson``/``test_pearson`` keys are kept beside
    the nested metric dicts because ``summarize_arms.py`` reads them, and the three
    arm groups already measured wrote that shape.
    """
    from sklearn.linear_model import LinearRegression

    features = {name: guide_indicator(dataset, idx) for name, idx in splits.items()}
    informative = features["train"].std(axis=0) > 0.0
    features = {name: matrix[:, informative] for name, matrix in features.items()}
    train_mean = float(targets["train"].mean())
    train_sd = float(targets["train"].std())
    out: dict[str, dict[str, Any]] = {}

    def record(name: str, predictions: dict[str, np.ndarray]) -> None:
        scores: dict[str, Any] = {}
        for split, prediction in predictions.items():
            if float(np.std(prediction)) == 0.0:
                metrics = regression_metrics(
                    targets[split], prediction, train_mean, train_sd
                )
                metrics["spearman"] = float("nan")
                metrics["pearson"] = float("nan")
            else:
                metrics = regression_metrics(
                    targets[split], prediction, train_mean, train_sd
                )
            scores[f"{split}_metrics"] = metrics
            scores[split] = metrics["mse_raw"]
            scores[f"{split}_pearson"] = metrics["pearson"]
            scores[f"{split}_spearman"] = metrics["spearman"]
            if split == "test":
                scores["test_predictions"] = [float(x) for x in prediction]
        out[name] = scores

    record(
        "train_mean",
        {split: np.full_like(targets[split], train_mean) for split in ("val", "test")},
    )
    model = LinearRegression()
    model.fit(features["train"], targets["train"])
    record("ols", {split: model.predict(features[split]) for split in ("val", "test")})
    out["ols"]["n_features"] = int(informative.sum())
    out["ols"]["n_train"] = int(len(targets["train"]))
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


def graph_regularization(
    dataset: Any, layer: int, lam: float, n_layers: int, n_heads: int
) -> dict[str, Any]:
    """One STRING channel per attention head, regularized on one block.

    The KL term pulls head *k*'s gene-gene attention row toward the row-normalized
    adjacency of channel *k*, so the six STRING channels and the six attention
    heads line up one to one. That alignment is why ``--heads`` is pinned to the
    channel count whenever this is on: a seventh channel would have no head and
    would be silently dropped by the backbone, which skips a channel it cannot
    place.

    ``layer`` is 0-based and must not be 0. The regularizer can only shape
    attention a later block still reads, and on a two-block stack block 0's
    attention is what block 1 consumes, so regularizing block 0 constrains the
    representation the model is still free to overwrite. Measured on the 031
    campaign: every one-block cell reported ``graph_reg_loss`` exactly 0.0 at
    every epoch, because the config named block 1 and the stack had only block 0.
    This function refuses that configuration instead of reproducing it.
    """
    channels = [
        rel
        for src, rel, dst in dataset.cell_graph.edge_types
        if src == "gene" and dst == "gene"
    ]
    if layer <= 0 or layer >= n_layers:
        raise ValueError(
            f"graph regularization asks for block {layer} of a {n_layers}-block "
            f"stack; it needs a block index in 1..{n_layers - 1} so a later "
            f"block reads the attention it shapes. Raise --layers or lower "
            f"--graph-reg-layer."
        )
    if n_heads != len(channels):
        raise ValueError(
            f"graph regularization maps one STRING channel to one attention "
            f"head, but this cell graph has {len(channels)} gene-gene channels "
            f"and the model has {n_heads} heads. Set --heads {len(channels)}."
        )
    return {
        "regularized_heads": {
            channel: {"layer": [layer], "head": index, "lambda": lam}
            for index, channel in enumerate(channels)
        },
        "row_sampling_rate": 1.0,
    }


def build_model(
    dataset: Any,
    hidden: int,
    layers: int,
    heads: int,
    dropout: float,
    device: torch.device,
    flux_layer: FluxLayer | None = None,
    precursor_indices: list[int] | None = None,
    graph_reg_config: dict[str, Any] | None = None,
    graph_reg_lambda: float = 0.0,
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
    model = CellGraphTransformerMetabolism(
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
        graph_regularization_config=graph_reg_config,
        graph_reg_lambda=graph_reg_lambda,
    ).to(device)
    if model.adjacency_matrices is not None:
        # The backbone keeps the adjacencies in a plain dict, not as registered
        # buffers, so they stay on the CPU and the loss copies every one of them
        # to the GPU on every step. At 5,729 genes a single channel is a dense
        # 5,729 x 5,729 float32, 131 MB, so six channels move 787 MB per step
        # over PCIe for no reason. Moving them once, in place so the layers'
        # shared reference follows, is the whole fix. In place also matters for
        # correctness of the backbone's own edge normalizer, which reads the same
        # dict.
        for channel, matrix in model.adjacency_matrices.items():
            model.adjacency_matrices[channel] = matrix.to(device)
    return model


def regression_metrics(
    observed_raw: np.ndarray, predicted_raw: np.ndarray, mean: float, std: float
) -> dict[str, float]:
    """The full metric set, in raw ug/mL plus the standardized MSE.

    Spearman is the objective of this trial, so it is first, but a rank
    correlation alone cannot say whether the predictions are on the right scale:
    an affine transform of the predictions leaves Spearman and Pearson untouched
    and moves RMSE, MAE and NMSE. Both families are therefore reported together
    on every split and every epoch.

    ``mse`` stays in standardized units because that is the quantity the loss
    minimizes and the one the lower-order nulls are reported in, so it is the
    only metric here that is comparable to the printed null table. Everything
    else is in ug/mL, the unit the campaign reports titer in.

    ``nmse`` divides by the variance of the OBSERVED values of that split, so 1.0
    is the score of predicting that split's own mean. It is not the same as
    ``mse``, whose standardization uses the TRAIN mean and sd: a split whose
    titers are more spread out than the train split's can have nmse below 1 and
    standardized mse above it. ``r2`` is ``1 - nmse`` by construction and is kept
    because it is what a reader expects to see; it is not a second measurement.
    """
    residual = observed_raw - predicted_raw
    variance = float(np.var(observed_raw))
    mse_raw = float(np.mean(residual**2))
    standardized_observed = (observed_raw - mean) / std
    standardized_predicted = (predicted_raw - mean) / std
    return {
        "spearman": float(spearmanr(observed_raw, predicted_raw).statistic),
        "pearson": float(pearsonr(observed_raw, predicted_raw).statistic),
        "mse": float(np.mean((standardized_observed - standardized_predicted) ** 2)),
        "mse_raw": mse_raw,
        "rmse": float(np.sqrt(mse_raw)),
        "mae": float(np.mean(np.abs(residual))),
        "nmse": mse_raw / variance,
        "r2": 1.0 - mse_raw / variance,
        "observed_sd": float(np.sqrt(variance)),
        "predicted_sd": float(np.std(predicted_raw)),
        "n": len(observed_raw),
    }


def predict(
    model: nn.Module,
    cell_graph: Any,
    loader: Any,
    device: torch.device,
    mean: float,
    std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(observed, predicted)`` in ug/mL, in loader order.

    Saved alongside the metrics so a comparison against the nulls can be
    bootstrapped on the records both models actually scored. Without the
    predictions the only uncertainty available on a fixed split is the spread
    over initializations, which says nothing about the test set being 59
    records, and a seed standard deviation quietly standing in for a sampling
    error is how a 0.11 gap gets called significant.
    """
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
    return np.asarray(observed), np.asarray(predictions) * std + mean


def evaluate(
    model: nn.Module,
    cell_graph: Any,
    loader: Any,
    device: torch.device,
    mean: float,
    std: float,
) -> dict[str, float]:
    """Run the model over a loader and score it with every metric."""
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
    return regression_metrics(np.asarray(observed), predicted_raw, mean, std)


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
    parser.add_argument(
        "--graph-reg-lambda",
        type=float,
        default=0.0,
        help=(
            "weight on the KL term that pulls each head's attention toward one "
            "STRING channel's adjacency; 0 turns the regularizer off. The "
            "backbone applies it once per channel and divides by that channel's "
            "own edge density, so the value means the same thing whatever the "
            "channel count"
        ),
    )
    parser.add_argument("--graph-reg-layer", type=int, default=1)
    parser.add_argument(
        "--no-decay-bias",
        action="store_true",
        help=(
            "exclude bias and normalization parameters from weight decay. Off "
            "by default so the arms measured before 2026-11-24 stay comparable, "
            "but suspected to matter here: the flux head reads the raw flux "
            "vector over 2,958 reactions, whose magnitudes are large and "
            "unnormalized, so it needs a large bias to center its output. "
            "Measured on the design grid, the share of test MSE that is a "
            "constant offset is 0 to 14 percent for every pooled cell and 12 to "
            "86 percent for every flux cell, and the two flux cells at weight "
            "decay 1e-2 are offset by 224 and 235 ug/mL against 37 to 125 for "
            "the four at 1e-4. Decaying the bias toward zero is the mechanism "
            "this flag tests"
        ),
    )
    parser.add_argument(
        "--select-on",
        choices=("val_mse", "val_spearman"),
        default="val_mse",
        help="the checkpoint selection rule; the trial's objective is Spearman",
    )
    parser.add_argument(
        "--project",
        default="torchcell_048-pputida-isoprenol-cgt",
        help="wandb project; the low-data trial logs to its own project",
    )
    parser.add_argument("--group", default=None, help="wandb group for this sweep")
    parser.add_argument(
        "--results-subdir",
        default=None,
        help="write the report under results/<subdir>/ instead of results/",
    )
    args = parser.parse_args()
    results_dir = (
        RESULTS
        if args.results_subdir is None
        else osp.join(RESULTS, args.results_subdir)
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(results_dir, exist_ok=True)
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
    graph_reg_config = None
    if args.graph_reg_lambda > 0.0:
        graph_reg_config = graph_regularization(
            dataset,
            layer=args.graph_reg_layer,
            lam=args.graph_reg_lambda,
            n_layers=args.layers,
            n_heads=args.heads,
        )
        log.info(
            "graph regularization: block %d, lambda %g, channels %s",
            args.graph_reg_layer,
            args.graph_reg_lambda,
            ", ".join(graph_reg_config["regularized_heads"]),
        )
    model = build_model(
        dataset,
        args.hidden,
        args.layers,
        args.heads,
        args.dropout,
        device,
        flux_layer=flux_layer,
        precursor_indices=precursors,
        graph_reg_config=graph_reg_config,
        graph_reg_lambda=args.graph_reg_lambda,
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
            project=args.project,
            group=args.group,
            name=f"{args.split}-{args.tag}-seed{args.seed}",
            config=vars(args)
            | {"n_params": n_params, "splits": {k: len(v) for k, v in splits.items()}},
        )

    if args.no_decay_bias:
        # A parameter is exempt when it is one-dimensional, which covers every
        # bias, every LayerNorm weight and the flux layer's scalar parameters,
        # and keeps every weight MATRIX decayed. Shape is the right test rather
        # than the name: the heads are built by the model, not here, so a name
        # filter would silently miss a head added later.
        decayed = [p for p in model.parameters() if p.requires_grad and p.dim() > 1]
        exempt = [p for p in model.parameters() if p.requires_grad and p.dim() <= 1]
        groups: list[dict[str, Any]] = [
            {"params": decayed, "weight_decay": args.weight_decay},
            {"params": exempt, "weight_decay": 0.0},
        ]
        log.info(
            "weight decay on %d parameter tensors, exempt on %d",
            len(decayed),
            len(exempt),
        )
        optimizer = torch.optim.AdamW(groups, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    loss_fn = nn.MSELoss()

    # The selection rule, as a sign and a key, so the loop reads the same for
    # both. Spearman is maximized and MSE is minimized, and the trial's stated
    # objective is Spearman, so running it on MSE has to be an explicit choice.
    select_key = "spearman" if args.select_on == "val_spearman" else "mse"
    select_sign = -1.0 if args.select_on == "val_spearman" else 1.0

    best = {"score": float("inf"), "epoch": -1}
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float]] = []
    graph_reg_seen = 0.0
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        reg_total = 0.0
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
            mse = loss_fn(prediction, target)
            # The backbone already scaled the KL term by graph_reg_scale, so it
            # is added as it comes back. It is reported separately from the MSE
            # at every epoch because the two have no common unit and a
            # regularizer that quietly dominates the objective looks, from the
            # loss curve alone, exactly like a model that will not fit.
            reg = reps["graph_reg_loss"]
            loss = mse + reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(mse.detach()) * int(batch.num_graphs)
            reg_total += float(reg.detach()) * int(batch.num_graphs)
            seen += int(batch.num_graphs)

        train_mse = total / max(seen, 1)
        train_reg = reg_total / max(seen, 1)
        graph_reg_seen = max(graph_reg_seen, abs(train_reg))
        train = evaluate(
            model, cell_graph, module.train_dataloader(), device, mean, std
        )
        val = evaluate(model, cell_graph, module.val_dataloader(), device, mean, std)
        row = {
            "epoch": epoch,
            "train_mse": train_mse,
            "graph_reg_loss": train_reg,
            **{f"train_{k}": v for k, v in train.items()},
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(row)
        if run is not None:
            # train and val on one axis for every metric, which is the only way
            # a 375-record fit shows its overfitting as it happens.
            logged = {"epoch": epoch, "loss/train": train_mse, "loss/val": val["mse"]}
            logged["loss/graph_reg"] = train_reg
            for split_name, scores in (("train", train), ("val", val)):
                for metric, value in scores.items():
                    if metric != "n":
                        logged[f"{split_name}/{metric}"] = value
            run.log(logged, step=epoch)
        score = select_sign * val[select_key]
        if score < best["score"]:
            best = {"score": score, "epoch": epoch, **val}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best["epoch"] >= args.patience:
            log.info(
                f"stopping at epoch {epoch}, no val improvement for {args.patience}"
            )
            break
        if epoch % 10 == 0:
            log.info(
                f"epoch {epoch:>3}  train_mse {train_mse:.4f}  "
                f"val_mse {val['mse']:.4f}  val_rho {val['spearman']:+.3f}  "
                f"val_r {val['pearson']:+.3f}  reg {train_reg:.4f}"
            )

    if args.graph_reg_lambda > 0.0 and graph_reg_seen == 0.0:
        raise ValueError(
            "graph regularization was switched on and graph_reg_loss was 0.0 at "
            "every epoch, so no channel was ever regularized and the run would "
            "have been reported as a graph-regularized result. This is exactly "
            "how the 031 campaign recorded 378 unregularized cells as "
            "regularized ones. Check that --graph-reg-layer names a block the "
            "stack has."
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, cell_graph, module.test_dataloader(), device, mean, std)
    val = evaluate(model, cell_graph, module.val_dataloader(), device, mean, std)
    test_observed, test_predicted = predict(
        model, cell_graph, module.test_dataloader(), device, mean, std
    )
    # The nulls are fit on the same splits in the same order, so one observed
    # vector serves the model and every null and the bootstrap can resample
    # record positions once and index all of them with it.
    if len(test_observed) != len(targets_by_split["test"]):
        raise ValueError(
            f"the test loader yielded {len(test_observed)} records and the "
            f"split holds {len(targets_by_split['test'])}; the saved "
            f"predictions would not line up with the nulls' and a paired "
            f"bootstrap over them would be comparing different records"
        )

    report = {
        "args": vars(args),
        "device": str(device),
        "n_params": n_params,
        "model_label": model_label,
        "metabolism_role": metabolism_role,
        "selection_rule": args.select_on,
        "graph_reg": (
            None
            if graph_reg_config is None
            else {
                "lambda": args.graph_reg_lambda,
                "layer": args.graph_reg_layer,
                "channels": list(graph_reg_config["regularized_heads"]),
                "max_epoch_loss": graph_reg_seen,
            }
        ),
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
        "test_indices": [int(i) for i in splits["test"]],
        "test_observed": [float(x) for x in test_observed],
        "test_predicted": [float(x) for x in test_predicted],
        "lower_order_nulls": nulls,
        "history": history,
    }
    out = osp.join(results_dir, f"train_{args.split}_{args.tag}_seed{args.seed}.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== result table ===")
    print(
        f"selection: {args.select_on}. MSE standardized; rho is Spearman, r is "
        f"Pearson, RMSE and MAE in ug/mL. val is the SELECTION set, not held out."
    )
    header = (
        f"{'model':<22}{'val rho':>9}{'val r':>8}{'val MSE':>9}"
        f"{'test rho':>10}{'test r':>8}{'test MSE':>10}{'test RMSE':>10}"
        f"{'test MAE':>9}{'test NMSE':>10}"
    )
    print(header)

    def table_row(label: str, v: dict[str, float], t: dict[str, float]) -> None:
        print(
            f"{label:<22}{v['spearman']:>+9.3f}{v['pearson']:>+8.3f}"
            f"{v['mse']:>9.4f}{t['spearman']:>+10.3f}{t['pearson']:>+8.3f}"
            f"{t['mse']:>10.4f}{t['rmse']:>10.1f}{t['mae']:>9.1f}"
            f"{t['nmse']:>10.3f}"
        )

    table_row(model_label, val, test)
    for name, scores in nulls.items():
        table_row(name, scores["val_metrics"], scores["test_metrics"])
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
