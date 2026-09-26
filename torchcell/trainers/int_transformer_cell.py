"""Lightning training module for the transformer cell gene-interaction model."""

import logging
from typing import Any, cast

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch_geometric.data import HeteroData
from torchmetrics import MeanSquaredError, MetricCollection, PearsonCorrCoef

from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.mle_dist_supcr import MleDistSupCR
from torchcell.losses.mle_wasserstein import MleWassSupCR
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg
from torchcell.viz import genetic_interaction_score
from torchcell.viz.visual_graph_degen import VisGraphDegen
from torchcell.viz.visual_regression import Visualization

log = logging.getLogger(__name__)

# The phenotype name of each (label, perturbation order): single-, double- and
# triple-mutant fitness, and digenic and trigenic interaction. Per-order metrics are
# logged once under ``order<k>`` and once under this name, so a chart can be read as
# ``train/gene_interaction/tmi/Pearson`` without a lookup. A single carries no
# interaction label, so order 1 has no interaction name.
ORDER_PHENOTYPE_NAMES: dict[str, dict[int, str]] = {
    "fitness": {1: "smf", 2: "dmf", 3: "tmf"},
    "gene_interaction": {2: "dmi", 3: "tmi"},
}


def token_precedence_rank(dataset_name: str) -> int:
    """Rank of a source dataset under ``LabelPolicy``'s default precedence, low first.

    Kuzmin 2018 before Kuzmin 2020 before Costanzo 2016 before a converted 0 (SGD
    essentiality, SynthLethDB), the order ``torchcell.data.label_policy.LabelPolicy``
    ranks source keys in. The validation reduction of the per-entry path keeps, per
    genotype and label, the rows of the lowest rank present and averages them, so its
    target is the policy's choice (plain mean where the policy would weight
    same-source replicates by inverse variance; the deviation is measured by
    ``make_pinned_eval_table_030.py``). ``Synthetic`` names are the smoke test's
    fictitious second token and rank last. Any other name is a configuration error.
    """
    if "Kuzmin2018" in dataset_name:
        return 0
    if "Kuzmin2020" in dataset_name:
        return 1
    if "Costanzo2016" in dataset_name:
        return 2
    if "GeneEssentialitySgd" in dataset_name or "SynthLethality" in dataset_name:
        return 3
    if dataset_name.startswith("Synthetic"):
        return 4
    raise ValueError(f"no precedence rank for dataset {dataset_name!r}")


class RegressionTask(L.LightningModule):
    """Lightning module training the transformer cell model on gene interactions."""

    # Metric collections are populated via setattr in __init__; declare their
    # types here so attribute access (reset/items/indexing) type-checks.
    train_metrics: MetricCollection
    val_metrics: MetricCollection
    test_metrics: MetricCollection
    train_transformed_metrics: MetricCollection
    val_transformed_metrics: MetricCollection
    test_transformed_metrics: MetricCollection
    _cell_graph_device: torch.device

    def __init__(
        self,
        model: nn.Module,
        cell_graph: torch.Tensor,
        optimizer_config: dict[str, Any],
        lr_scheduler_config: dict[str, Any] | None,
        batch_size: int | None = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        plot_sample_ceiling: int = 1000,
        plot_every_n_epochs: int = 10,
        plot_transformer_diagnostics_every_n_epochs: int = 10,
        plot_edge_recovery_every_n_epochs: int = 10,
        loss_func: nn.Module | None = None,
        grad_accumulation_schedule: dict[int, int] | None = None,
        device: str = "cuda",
        inverse_transform: nn.Module | None = None,
        execution_mode: str = "training",  # "training" or "dataloader_profiling"
        fitness_lambda: float | None = None,
        gradient_probe_epochs: list[int] | None = None,
        per_order_metrics: bool = False,
        per_entry: bool = False,
        dataset_vocabulary: list[str] | None = None,
        essentiality_eval: dict[str, Any] | None = None,
    ):
        """Set up the model, cloned cell graph, loss, metrics, and execution mode.

        ``per_entry`` trains one row per STORED ENTRY instead of one per genotype, for
        a build that keeps every source measurement (030). The model must carry a
        ``dataset_token``; the batch must carry ``phenotype_values_batch`` and
        ``phenotype_dataset_indices``; ``dataset_vocabulary`` is the token list the
        processor emitted against. Losses are masked MSE over entry rows. Training
        metrics are per entry. Validation and test report the 025-comparable number,
        one value per pinned genotype per label: the entries of the highest-precedence
        source present (Kuzmin 2018, then Kuzmin 2020, then Costanzo, converted zeros
        last, the order of ``LabelPolicy``) averaged, scored under that source's own
        token; beside it the per-entry, per-token and cross-token (Kuzmin 2018 rows
        under the 2020 token and back) Pearson. ``_coo_label``'s conflict rule is not
        used on this path: two values of one label under different tokens are two
        targets, which is the point.

        ``essentiality_eval`` scores a second validation loader (``dataloader_idx``
        1) of held-out single-deletion genotypes: ``{"token_smf": <name>, "token_sgd":
        <name>, "released": {node_index: 0/1}, "matched": {node_index: 0/1}}``. Each
        genotype's fitness is read under the Costanzo single token (the reported
        score) and the SGD token (a diagnostic), and AUROC of ``-fitness`` against
        the essential label is logged per set at epoch end after an all-gather, so
        every rank logs the same four keys.

        ``per_order_metrics`` adds, for every stage, the same MSE / RMSE / Pearson
        collections split by perturbation order (the number of perturbed genes in the
        record: 1, 2, 3), logged as ``<stage>/gene_interaction/order<k>/<metric>`` and,
        on the joint path, ``<stage>/fitness/order<k>/<metric>``, for the orders that
        received samples in the epoch. Each value is logged a second time under the
        phenotype name of that (label, order) from ``ORDER_PHENOTYPE_NAMES``
        (``<stage>/fitness/smf|dmf|tmf/<metric>``, ``<stage>/gene_interaction/dmi|tmi/
        <metric>``), as are the record counts. An arm whose training pool mixes doubles and
        triples (the closure and whole-build arms) otherwise reports one training
        Pearson over dmi and tmi together; this separates the trigenic fit from the
        digenic one. Validation and test on the pinned trigenic splits are order 3
        only, so there the order-3 line equals the overall one.

        ``gradient_probe_epochs`` lists epochs at whose FIRST training batch the
        gradient of each weighted loss term (point, distribution, graph penalty,
        fitness) is taken separately against every parameter and its global norm
        logged as ``probe/grad_norm/<term>``, with the norm of the summed loss as
        ``probe/grad_norm/total``. Panel c of the graph-regularization figure: where
        the gradient comes from, by epoch. A term without a graph (lambda 0, the hard
        mask) logs 0. Costs one extra backward per term on one batch per listed epoch.

        ``fitness_lambda`` switches on the JOINT fitness objective. ``None`` (the
        default) is the single-label path, bit-for-bit what 010 and the 025
        replication trained: ``phenotype_values`` IS the gene-interaction vector.
        A float means the batch carries two scalar labels per record in COO form
        (``phenotype_labels: [fitness, gene_interaction]``), the model was built with a
        ``global`` head (``heads_config``), and the loss becomes
        ``L_gi + fitness_lambda * MSE(global_head, fitness)`` on the normalized
        scale. Requires ``follow_batch=["phenotype_values"]`` in the data module so the
        COO values carry their batch row.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.execution_mode = execution_mode
        self.fitness_lambda = fitness_lambda
        self.gradient_probe_epochs: set[int] = set(
            int(e) for e in (gradient_probe_epochs or [])
        )
        # Clone cell_graph to avoid modifying the dataset's original cell_graph
        # This is necessary for pin_memory compatibility in DataLoader
        self.cell_graph = cell_graph.clone()
        self.inverse_transform = inverse_transform
        self.loss_func = loss_func

        # Initialize gradient accumulation
        self.current_accumulation_steps = 1
        if self.hparams["grad_accumulation_schedule"] is not None:
            # Get the accumulation steps for epoch 0
            self.current_accumulation_steps = self.hparams[
                "grad_accumulation_schedule"
            ].get(0, 1)

        reg_metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(squared=True),
                "RMSE": MeanSquaredError(squared=False),
                "Pearson": PearsonCorrCoef(),
            }
        )

        # Create metrics for each stage
        for stage in ["train", "val", "test"]:
            metrics_dict = reg_metrics.clone(prefix=f"{stage}/gene_interaction/")
            setattr(self, f"{stage}_metrics", metrics_dict)

            # Add metrics operating in transformed space
            transformed_metrics = reg_metrics.clone(
                prefix=f"{stage}/transformed/gene_interaction/"
            )
            setattr(self, f"{stage}_transformed_metrics", transformed_metrics)

            # Fitness metrics exist only on the joint path, so a single-label run's
            # module list and checkpoint are unchanged.
            if fitness_lambda is not None:
                setattr(
                    self,
                    f"{stage}_fitness_metrics",
                    reg_metrics.clone(prefix=f"{stage}/fitness/"),
                )
                setattr(
                    self,
                    f"{stage}_transformed_fitness_metrics",
                    reg_metrics.clone(prefix=f"{stage}/transformed/fitness/"),
                )

        # Per-perturbation-order metrics (opt-in, see __init__). Orders 1 to 3 cover
        # every record of the 025 build; a count per order says which ones to log.
        self.per_order_metrics = per_order_metrics
        self.metric_orders = (1, 2, 3)
        # Records fed to each per-order collection this epoch, keyed by the ModuleDict
        # attribute name; the interaction and fitness collections count separately
        # because a single-gene record carries fitness but no interaction label.
        self._order_counts: dict[str, dict[int, int]] = {
            f"{stage}_order{kind}": dict.fromkeys(self.metric_orders, 0)
            for stage in ("train", "val", "test")
            for kind in ("_metrics", "_fitness_metrics")
        }
        if per_order_metrics:
            for stage in ("train", "val", "test"):
                setattr(
                    self,
                    f"{stage}_order_metrics",
                    nn.ModuleDict(
                        {
                            str(k): reg_metrics.clone(
                                prefix=f"{stage}/gene_interaction/order{k}/"
                            )
                            for k in self.metric_orders
                        }
                    ),
                )
                if fitness_lambda is not None:
                    setattr(
                        self,
                        f"{stage}_order_fitness_metrics",
                        nn.ModuleDict(
                            {
                                str(k): reg_metrics.clone(
                                    prefix=f"{stage}/fitness/order{k}/"
                                )
                                for k in self.metric_orders
                            }
                        ),
                    )

        # === Per-entry rows with the source-dataset token (see __init__) ===
        self.per_entry = bool(per_entry)
        self.dataset_vocabulary: list[str] = list(dataset_vocabulary or [])
        self._token_rank: torch.Tensor | None = None
        self._crosstoken_pair: tuple[int, int] | None = None
        self._entry_counts: dict[str, dict[str, int]] = {
            stage: {"gene_interaction": 0, "fitness": 0}
            for stage in ("train", "val", "test")
        }
        self._token_counts: dict[str, dict[str, list[int]]] = {}
        if self.per_entry:
            if not self.dataset_vocabulary:
                raise ValueError("per_entry needs the dataset_vocabulary")
            self.register_buffer(
                "_token_rank_buffer",
                torch.tensor(
                    [token_precedence_rank(n) for n in self.dataset_vocabulary],
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self._token_rank = cast(torch.Tensor, self._token_rank_buffer)
            names = self.dataset_vocabulary
            if "TmiKuzmin2018Dataset" in names and "TmiKuzmin2020Dataset" in names:
                self._crosstoken_pair = (
                    names.index("TmiKuzmin2018Dataset"),
                    names.index("TmiKuzmin2020Dataset"),
                )
            per_entry_labels = ["gene_interaction"] + (
                ["fitness"] if fitness_lambda is not None else []
            )
            for stage in ("train", "val", "test"):
                # Original-scale metrics over every entry row (train's primary metric,
                # val/test's secondary one beside the policy-reduced collection).
                setattr(
                    self,
                    f"{stage}_entry_metrics",
                    nn.ModuleDict(
                        {
                            label: reg_metrics.clone(prefix=f"{stage}/entries/{label}/")
                            for label in per_entry_labels
                        }
                    ),
                )
                # One Pearson per (label, token); every key is logged on every rank in
                # vocabulary order, NaN when the rank saw no row of that token.
                setattr(
                    self,
                    f"{stage}_token_metrics",
                    nn.ModuleDict(
                        {
                            label: nn.ModuleDict(
                                {str(i): PearsonCorrCoef() for i in range(len(names))}
                            )
                            for label in per_entry_labels
                        }
                    ),
                )
                self._token_counts[stage] = {
                    label: [0] * len(names) for label in per_entry_labels
                }
                setattr(self, f"{stage}_crosstoken_metric", PearsonCorrCoef())
        # === Essentiality holdout (second validation loader) ===
        self.essentiality_eval: dict[str, Any] | None = None
        self._ess_buffer: dict[str, list[torch.Tensor]] = {
            "node": [],
            "smf": [],
            "sgd": [],
        }
        if essentiality_eval is not None:
            if not self.per_entry:
                raise ValueError("essentiality_eval needs per_entry=True")
            for key in ("token_smf", "token_sgd", "released", "matched"):
                if key not in essentiality_eval:
                    raise ValueError(f"essentiality_eval lacks {key!r}")
            for key in ("token_smf", "token_sgd"):
                if essentiality_eval[key] not in self.dataset_vocabulary:
                    raise ValueError(
                        f"essentiality_eval[{key!r}]={essentiality_eval[key]!r} is "
                        "not in the dataset vocabulary"
                    )
            self.essentiality_eval = {
                "token_smf": self.dataset_vocabulary.index(
                    essentiality_eval["token_smf"]
                ),
                "token_sgd": self.dataset_vocabulary.index(
                    essentiality_eval["token_sgd"]
                ),
                "released": {
                    int(k): int(v) for k, v in essentiality_eval["released"].items()
                },
                "matched": {
                    int(k): int(v) for k, v in essentiality_eval["matched"].items()
                },
            }

        # Separate accumulators for train, validation, and test samples
        self.train_samples: dict[str, Any] = {
            "true_values": [],
            "predictions": [],
            "latents": {},
        }
        self.val_samples: dict[str, Any] = {
            "true_values": [],
            "predictions": [],
            "latents": {},
        }
        self.test_samples: dict[str, Any] = {
            "true_values": [],
            "predictions": [],
            "latents": {},
        }
        self.automatic_optimization = False

        # Edge recovery metric accumulators (for validation)
        self.edge_recovery_ks = [8, 32, 128, 320]
        self.reset_edge_recovery_accumulators()

        # Attention diagnostic accumulators (for validation)
        self.attention_stats_accumulators: dict[
            int, dict[str, float | int]
        ] = {}  # {layer_idx: {"entropy_sum": float, "effective_rank_sum": float, "top5_sum": float, "top10_sum": float, "top50_sum": float, "count": int}}
        self.gradient_norms: dict[int, float] = {}  # {layer_idx: norm_value}
        #: Rows masked by `_coo_label` because they carried two values of one label.
        self.coo_conflict_rows: dict[str, int] = {}

        # Residual update accumulators (Tier 1 - very cheap)
        self.residual_update_accumulators: dict[
            int, dict[str, float | int]
        ] = {}  # {layer_idx: {"sum_ratio": float, "count": int}}

    def _coo_label(
        self, batch: HeteroData, label: str, batch_size: int, original: bool
    ) -> torch.Tensor:
        """One value per batch row for ``label`` from the COO phenotype fields.

        Returns ``[batch_size, 1]`` with NaN where a row carries no value of that
        label. Rows come from ``phenotype_values_batch`` (``follow_batch``), NOT from
        ``phenotype_sample_indices``, which indexes experiments WITHIN a genotype and
        is not offset across the batch.

        A row carrying TWO values of one label is a data conflict, not a duplicate to
        average: in the 025 build three single-deletion genotypes (YPL212C, YHL047C,
        YKL117W) carry a measured Costanzo 2016 fitness near 1.0 beside a SynthLethDB
        single-gene record at 0.0 (found by the S3 closure cell, IGB job 2408791,
        2026-09-17). Such a row is masked (NaN, so it contributes no loss and no metric)
        and counted in ``coo_conflict_rows``; the mean of 1.0 and 0.0 would be a value
        nobody measured. Until 2026-09-17 this raised, which took the whole DDP job down
        through an NCCL timeout on the other ranks.
        """
        gene = batch["gene"]
        types = gene.phenotype_types
        if isinstance(types[0], list):
            types = types[0]
        values = gene.phenotype_values_original if original else gene.phenotype_values
        sel = gene.phenotype_type_indices == types.index(label)
        rows = gene.phenotype_values_batch[sel]
        vals = values[sel]
        out = torch.full(
            (batch_size,), float("nan"), device=vals.device, dtype=vals.dtype
        )
        counts = torch.bincount(rows, minlength=batch_size)
        conflict = counts > 1
        if bool(conflict.any()):
            n_conflict = int(conflict.sum())
            self.coo_conflict_rows[label] = (
                self.coo_conflict_rows.get(label, 0) + n_conflict
            )
            if self.coo_conflict_rows[label] == n_conflict:
                print(
                    f"[coo] {label}: {n_conflict} batch row(s) carry more than one "
                    "value and are masked out of the loss and every metric "
                    "(counted in coo_conflict_rows)",
                    flush=True,
                )
            keep = ~conflict[rows]
            rows, vals = rows[keep], vals[keep]
        out[rows] = vals
        return out.unsqueeze(1)

    # ------------------------------------------------------------ per-entry rows
    def _entry_rows(
        self, batch: HeteroData, label: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Entry rows of ``label``: (genotype row, value, original value, token, mask).

        ``mask`` selects the rows of ``label`` among ALL entry rows of the batch, so a
        per-entry model output ``[E, 1]`` is indexed by it. Placeholder rows (a
        genotype with no value, NaN) are dropped.
        """
        gene = batch["gene"]
        types = gene.phenotype_types
        if isinstance(types[0], list):
            types = types[0]
        sel = gene.phenotype_type_indices == types.index(label)
        values = gene.phenotype_values
        sel = sel & ~torch.isnan(values)
        original = (
            gene.phenotype_values_original
            if hasattr(gene, "phenotype_values_original")
            else values
        )
        return (
            gene.phenotype_values_batch[sel],
            values[sel],
            original[sel],
            gene.phenotype_dataset_indices[sel],
            sel,
        )

    def _policy_reduce(
        self,
        rows: torch.Tensor,
        values: torch.Tensor,
        preds: torch.Tensor,
        tokens: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One (target, prediction) per genotype from its highest-precedence source.

        Returns ``(targets [B], preds [B], present [B])``; genotypes without a row of
        this label are absent. Rows of the lowest precedence rank present in a
        genotype are kept and averaged (see ``token_precedence_rank``).
        """
        assert self._token_rank is not None
        rank = self._token_rank.to(tokens.device)[tokens]
        big = torch.iinfo(rank.dtype).max
        best = torch.full((batch_size,), big, dtype=rank.dtype, device=rank.device)
        best = best.scatter_reduce(0, rows, rank, reduce="amin")
        keep = rank == best[rows]
        rows_k, vals_k, preds_k = rows[keep], values[keep], preds[keep]
        counts = torch.bincount(rows_k, minlength=batch_size).to(values.dtype)
        present = counts > 0
        denom = counts.clamp(min=1)
        target = torch.zeros(batch_size, dtype=values.dtype, device=values.device)
        target = target.index_add(0, rows_k, vals_k) / denom
        pred = torch.zeros(batch_size, dtype=preds.dtype, device=preds.device)
        pred = pred.index_add(0, rows_k, preds_k) / denom
        return target, pred, present

    def _update_order_metrics_rows(
        self,
        stage: str,
        attr: str,
        order: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Per-order collections fed with rows whose order is given per row."""
        if not self.per_order_metrics:
            return
        collections = getattr(self, attr)
        for k in self.metric_orders:
            sel = order == k
            n = int(sel.sum())
            if n == 0:
                continue
            collections[str(k)].update(preds[sel].view(-1), targets[sel].view(-1))
            self._order_counts[attr][k] += n

    def _update_token_metrics(
        self,
        stage: str,
        label: str,
        tokens: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        collections = getattr(self, f"{stage}_token_metrics")[label]
        for t in torch.unique(tokens).tolist():
            sel = tokens == t
            collections[str(int(t))].update(preds[sel].view(-1), targets[sel].view(-1))
            self._token_counts[stage][label][int(t)] += int(sel.sum())

    def _log_per_entry_epoch_metrics(self, stage: str) -> None:
        """Log the per-entry, per-token, cross-token and count keys of ``stage``."""
        if not self.per_entry:
            return
        for label, collection in getattr(self, f"{stage}_entry_metrics").items():
            for key, value in self._compute_metrics_safely(collection).items():
                self.log(key, value, sync_dist=True)
            collection.reset()
            self.log(
                f"{stage}/n_entries/{label}",
                float(self._entry_counts[stage][label]),
                sync_dist=True,
                reduce_fx="sum",
            )
            self._entry_counts[stage][label] = 0
        for label, per_token in getattr(self, f"{stage}_token_metrics").items():
            for i, name in enumerate(self.dataset_vocabulary):
                metric = per_token[str(i)]
                # Every key on every rank, in vocabulary order: a token this rank
                # never saw logs NaN rather than being skipped.
                value = (
                    metric.compute()
                    if self._token_counts[stage][label][i] >= 2
                    else torch.tensor(float("nan"))
                )
                self.log(f"{stage}/token/{name}/{label}/Pearson", value, sync_dist=True)
                self.log(
                    f"{stage}/token/{name}/{label}/n_entries",
                    float(self._token_counts[stage][label][i]),
                    sync_dist=True,
                    reduce_fx="sum",
                )
                metric.reset()
                self._token_counts[stage][label][i] = 0
        if self._crosstoken_pair is not None and stage != "train":
            metric = getattr(self, f"{stage}_crosstoken_metric")
            for key, value in self._compute_metrics_safely(
                MetricCollection({"Pearson": metric})
            ).items():
                self.log(
                    f"{stage}/crosstoken/gene_interaction/{key}", value, sync_dist=True
                )
            metric.reset()

    def _inverse_scalar(self, predictions: torch.Tensor, label: str) -> torch.Tensor:
        """Map ``[B, 1]`` normalized predictions of ``label`` back to the label scale."""
        if self.inverse_transform is None:
            return predictions
        batch_size = predictions.size(0)
        temp_data = HeteroData()
        temp_data["gene"].phenotype_values = predictions.squeeze(1)
        temp_data["gene"].phenotype_type_indices = torch.zeros(
            batch_size, dtype=torch.long, device=predictions.device
        )
        temp_data["gene"].phenotype_sample_indices = torch.arange(
            batch_size, device=predictions.device
        )
        temp_data["gene"].phenotype_types = [label]
        inv = self.inverse_transform(temp_data)["gene"]["phenotype_values"]
        return cast(torch.Tensor, inv).reshape(batch_size, 1)

    def _gradient_norm(self, term: torch.Tensor) -> float:
        """Global L2 norm of d(term)/d(params); 0 for a term with no graph."""
        if not term.requires_grad:
            return 0.0
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        total = torch.zeros((), device=term.device)
        for g in grads:
            if g is not None:
                total = total + g.detach().float().pow(2).sum()
        return float(total.sqrt())

    def _log_gradient_probe(
        self, terms: dict[str, torch.Tensor], total_loss: torch.Tensor, batch_size: int
    ) -> None:
        """Log ``probe/grad_norm/<term>`` for each weighted term and for their sum.

        Each backward is taken with ``retain_graph=True`` so the optimizer's own
        backward on ``total_loss`` still runs afterwards; the ``.grad`` fields are not
        touched (``torch.autograd.grad`` returns the gradients rather than
        accumulating them).
        """
        norms = {name: self._gradient_norm(t) for name, t in terms.items()}
        norms["total"] = self._gradient_norm(total_loss)
        # The SLURM log carries the numbers too, so a run whose W&B history is lost
        # (offline sync, a killed job) still has its probe.
        print(
            f"gradient probe epoch {self.current_epoch} rank {self.global_rank}: "
            + ", ".join(f"{k}={v:.4g}" for k, v in norms.items())
        )
        for name, value in norms.items():
            self.log(
                f"probe/grad_norm/{name}", value, batch_size=batch_size, sync_dist=True
            )
        if norms["point"] > 0:
            self.log(
                "probe/grad_ratio/graph_reg_to_point",
                norms.get("graph_reg", 0.0) / norms["point"],
                batch_size=batch_size,
                sync_dist=True,
            )
        self.log("probe/epoch", float(self.current_epoch), sync_dist=True)

    def _fitness_step(
        self,
        batch: HeteroData,
        representations: dict[str, Any],
        batch_size: int,
        stage: str,
    ) -> torch.Tensor:
        """Weighted fitness MSE on the ``global`` head; also feeds the fitness metrics."""
        fit_pred = representations["head_outputs"]["global"]  # [B, 1]
        fit_vals = self._coo_label(batch, "fitness", batch_size, original=False)
        mask = ~torch.isnan(fit_vals)
        fitness_loss = nn.functional.mse_loss(fit_pred[mask], fit_vals[mask])
        self.log(
            f"{stage}/fitness_loss", fitness_loss, batch_size=batch_size, sync_dist=True
        )
        getattr(self, f"{stage}_transformed_fitness_metrics").update(
            fit_pred[mask].view(-1), fit_vals[mask].view(-1)
        )
        fit_orig = self._coo_label(batch, "fitness", batch_size, original=True)
        inv_fit = self._inverse_scalar(fit_pred.detach(), "fitness")
        getattr(self, f"{stage}_fitness_metrics").update(
            inv_fit[mask].view(-1), fit_orig[mask].view(-1)
        )
        self._update_order_metrics(
            stage,
            f"{stage}_order_fitness_metrics",
            batch,
            batch_size,
            inv_fit,
            fit_orig,
            mask,
        )
        assert self.fitness_lambda is not None
        return self.fitness_lambda * fitness_loss

    def _perturbation_order(self, batch: HeteroData, batch_size: int) -> torch.Tensor:
        """Number of perturbed genes per batch row, ``[batch_size]`` long."""
        return torch.bincount(
            batch["gene"].perturbation_indices_batch, minlength=batch_size
        )

    def _update_order_metrics(
        self,
        stage: str,
        attr: str,
        batch: HeteroData,
        batch_size: int,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Feed ``preds``/``targets`` rows of each perturbation order to its metrics.

        ``mask`` is the ``[batch_size, 1]`` label-present mask already used for the
        pooled collection; ``attr`` names the ModuleDict (``<stage>_order_metrics`` or
        ``<stage>_order_fitness_metrics``).
        """
        if not self.per_order_metrics:
            return
        order = self._perturbation_order(batch, batch_size)
        present = mask.view(-1)
        collections = getattr(self, attr)
        for k in self.metric_orders:
            sel = present & (order == k)
            n = int(sel.sum())
            if n == 0:
                continue
            collections[str(k)].update(preds[sel].view(-1), targets[sel].view(-1))
            self._order_counts[attr][k] += n

    def _log_order_epoch_metrics(self, stage: str) -> None:
        """Compute, log and reset the per-order collections of ``stage``."""
        if not self.per_order_metrics:
            return
        attrs = [f"{stage}_order_metrics"]
        if self.fitness_lambda is not None:
            attrs.append(f"{stage}_order_fitness_metrics")
        for attr in attrs:
            collections = getattr(self, attr)
            label = (
                "gene_interaction" if attr == f"{stage}_order_metrics" else "fitness"
            )
            for k in self.metric_orders:
                collection = collections[str(k)]
                order_key = f"order{k}"
                name = ORDER_PHENOTYPE_NAMES[label].get(k)
                if self._order_counts[attr][k] > 0:
                    for key, value in self._compute_metrics_safely(collection).items():
                        self.log(key, value, sync_dist=True)
                        if name is not None:
                            self.log(
                                key.replace(f"/{order_key}/", f"/{name}/"),
                                value,
                                sync_dist=True,
                            )
                collection.reset()
                count = float(self._order_counts[attr][k])
                self.log(
                    f"{stage}/n_records/{label}/{order_key}",
                    count,
                    sync_dist=True,
                    reduce_fx="sum",
                )
                if name is not None:
                    self.log(
                        f"{stage}/n_records/{label}/{name}",
                        count,
                        sync_dist=True,
                        reduce_fx="sum",
                    )
                self._order_counts[attr][k] = 0

    def _log_fitness_epoch_metrics(self, stage: str) -> None:
        """Compute, log and reset the two fitness metric collections of ``stage``."""
        if self.fitness_lambda is None:
            return
        for name in (
            f"{stage}_fitness_metrics",
            f"{stage}_transformed_fitness_metrics",
        ):
            collection = getattr(self, name)
            for key, value in self._compute_metrics_safely(collection).items():
                self.log(key, value, sync_dist=True)
            collection.reset()

    def _get_batch_size(self, batch: HeteroData) -> int:
        """Get batch size from batch, handling different batch structures."""
        if hasattr(batch["gene"], "x"):
            return int(batch["gene"].x.size(0))
        elif hasattr(batch["gene"], "perturbation_indices"):
            # For Perturbation processor, count unique batch indices
            if hasattr(batch["gene"], "perturbation_indices_batch"):
                return int(batch["gene"].perturbation_indices_batch.max().item() + 1)
            else:
                # Fallback: assume batch size from perturbation_indices
                return int(batch["gene"].perturbation_indices.size(0))
        elif hasattr(batch["gene"], "phenotype_values"):
            return int(batch["gene"].phenotype_values.size(0))
        else:
            # Last resort fallback
            return 1

    def reset_edge_recovery_accumulators(self) -> None:
        """Reset accumulators for edge recovery metrics."""
        self.edge_recovery_accumulators: dict[str, Any] = {}

    def _plot_edge_recovery_metrics(self) -> None:
        """Create and log matplotlib visualizations for edge recovery metrics."""
        from torchcell.viz.graph_recovery import GraphRecoveryVisualization

        # Initialize visualization
        vis = GraphRecoveryVisualization(base_dir=self.trainer.default_root_dir)

        # Prepare recall metrics (keys now include layer/head: "graph_L0_H1")
        recall_metrics = {}
        for metric_key, acc in self.edge_recovery_accumulators.items():
            if acc["count_nodes_deg"] > 0:
                recall_metrics[metric_key] = (
                    acc["sum_recall_deg"] / acc["count_nodes_deg"]
                )

        # Prepare precision metrics
        precision_metrics: dict[str, dict[int, float]] = {}
        for metric_key, acc in self.edge_recovery_accumulators.items():
            precision_metrics[metric_key] = {}
            for k in self.edge_recovery_ks:
                if acc["count_nodes_prec"][k] > 0:
                    precision_metrics[metric_key][k] = (
                        acc["sum_prec"][k] / acc["count_nodes_prec"][k]
                    )

        # Prepare edge-mass alignment metrics
        edge_mass_metrics = {}
        for metric_key, acc in self.edge_recovery_accumulators.items():
            if acc["count_batches"] > 0:
                edge_mass_metrics[metric_key] = (
                    acc["sum_edge_mass"] / acc["count_batches"]
                )

        # Create aggregate plots
        if recall_metrics:
            vis.plot_edge_recovery_recall(
                recall_metrics, self.current_epoch, None, stage="val"
            )

        if precision_metrics:
            vis.plot_edge_recovery_precision(
                precision_metrics,
                self.edge_recovery_ks,
                self.current_epoch,
                None,
                stage="val",
            )

        if edge_mass_metrics:
            vis.plot_edge_mass_alignment(
                edge_mass_metrics, self.current_epoch, None, stage="val"
            )

        # Create per-graph plots
        if recall_metrics and precision_metrics:
            vis.plot_edge_recovery_per_graph(
                recall_metrics,
                precision_metrics,
                self.edge_recovery_ks,
                self.current_epoch,
                None,
                stage="val",
            )

    def _accumulate_edge_recovery_metrics(
        self, attention_weights_list: list[torch.Tensor], batch_idx: int
    ) -> None:
        """Compute edge recovery metrics from attention weights.

        Args:
            attention_weights_list: List of attention tensors per layer [batch, heads, N, N]
            batch_idx: Current batch index
        """
        # Only compute if model has graph regularization enabled
        if not hasattr(self.model, "regularized_head_config") or not hasattr(
            self.model, "adjacency_matrices"
        ):
            # Removed debug print - not a warning, just informational
            # if batch_idx == 0:  # Print once per epoch
            #     print(
            #         "Edge recovery: Model missing regularized_head_config or adjacency_matrices attributes"
            #     )
            return

        if (
            self.model.regularized_head_config is None
            or self.model.adjacency_matrices is None
        ):
            # Removed debug print - not a warning, just informational
            # if batch_idx == 0:  # Print once per epoch
            #     print(
            #         "Edge recovery: Model has None for regularized_head_config or adjacency_matrices"
            #     )
            return

        # For each regularized graph, expand multi-layer configs
        regularized_head_config = cast(
            dict[str, Any], self.model.regularized_head_config
        )
        adjacency_matrices = cast(
            dict[str, torch.Tensor], self.model.adjacency_matrices
        )
        for graph_name, config in regularized_head_config.items():
            # Handle both single layer (int) and multiple layers (list)
            layer_config = config["layer"]
            layers = [layer_config] if isinstance(layer_config, int) else layer_config
            head_idx = config["head"]

            # Process each layer separately
            for layer_idx in layers:
                # Skip if this layer's attention not available
                if layer_idx >= len(attention_weights_list):
                    continue

                # Get attention for this graph's layer/head: [batch, heads, N, N]
                attn = attention_weights_list[layer_idx]
                attn_for_head = attn[:, head_idx, :, :]  # [batch, N, N]

                # Get true adjacency for this graph from dict
                if graph_name not in adjacency_matrices:
                    continue
                adj_true = adjacency_matrices[graph_name].to(
                    attn_for_head.device
                )  # [N, N]

                # Average attention across batch dimension
                attn_avg = attn_for_head.mean(dim=0)  # [N, N]

                # Create unique key for this (graph, layer, head) combination
                metric_key = f"{graph_name}_L{layer_idx}_H{head_idx}"

                # Initialize accumulators for this combination if needed
                if metric_key not in self.edge_recovery_accumulators:
                    self.edge_recovery_accumulators[metric_key] = {
                        "sum_recall_deg": 0.0,
                        "count_nodes_deg": 0,
                        "sum_prec": {k: 0.0 for k in self.edge_recovery_ks},
                        "count_nodes_prec": {k: 0 for k in self.edge_recovery_ks},
                        "sum_edge_mass": 0.0,  # NEW: Edge-mass alignment
                        "count_batches": 0,  # NEW: Number of batches
                        "graph_name": graph_name,  # Store original graph name
                        "layer": layer_idx,  # Store layer index
                        "head": head_idx,  # Store head index
                        "degree_correlation_sum": 0.0,
                        "degree_corr_count": 0,
                    }
                elif (
                    "degree_correlation_sum"
                    not in self.edge_recovery_accumulators[metric_key]
                ):
                    # Add degree fields if accumulator exists but was created by degree bias
                    self.edge_recovery_accumulators[metric_key][
                        "degree_correlation_sum"
                    ] = 0.0
                    self.edge_recovery_accumulators[metric_key]["degree_corr_count"] = 0

                acc = self.edge_recovery_accumulators[metric_key]

                # NEW: Compute edge-mass alignment (fraction of attention on true edges)
                edge_mask = (adj_true > 0).float()  # [N, N] binary mask
                total_attn_val = attn_avg.sum().item()
                edge_attn_val = (attn_avg * edge_mask).sum().item()
                edge_mass_fraction = (
                    (edge_attn_val / total_attn_val) if total_attn_val > 0 else 0.0
                )

                acc["sum_edge_mass"] += edge_mass_fraction
                acc["count_batches"] += 1

                # Get number of nodes
                N = attn_avg.size(0)

                # Pre-compute topk for ALL nodes at once (vectorized - prevents memory leak)
                # This replaces N*5 topk calls with 1 topk call per batch
                k_max = max(self.edge_recovery_ks + [N])
                topk_values, topk_indices = torch.topk(
                    attn_avg, min(k_max, N), dim=-1
                )  # [N, k_max]

                # For each node
                for i in range(N):
                    # Get true neighbors from adjacency
                    true_neighbors = (adj_true[i] > 0).nonzero(as_tuple=True)[0]
                    degree_i = len(true_neighbors)

                    # Skip isolated nodes
                    if degree_i == 0:
                        continue

                    # Use pre-computed topk results (no new allocations)
                    top_indices_i = topk_indices[i]  # [k_max]

                    # === Recall@degree ===
                    k_deg = min(degree_i, N)
                    top_deg_indices = top_indices_i[:k_deg]
                    hits_deg = torch.isin(top_deg_indices, true_neighbors).sum().item()
                    recall_deg_i = hits_deg / degree_i

                    acc["sum_recall_deg"] += recall_deg_i
                    acc["count_nodes_deg"] += 1

                    # === Precision@k for each k ===
                    for k in self.edge_recovery_ks:
                        k_eff = min(k, N)
                        top_k_indices = top_indices_i[:k_eff]
                        hits_k = torch.isin(top_k_indices, true_neighbors).sum().item()
                        prec_k_i = hits_k / k_eff

                        acc["sum_prec"][k] += prec_k_i
                        acc["count_nodes_prec"][k] += 1

                # Clean up GPU memory after processing all nodes
                if torch.cuda.is_available():
                    del topk_values, topk_indices
                    torch.cuda.empty_cache()

    def _accumulate_attention_diagnostics(
        self, attention_weights_list: list[torch.Tensor], batch_idx: int
    ) -> None:
        """Compute attention diagnostics (entropy, effective rank, top-k concentration) from attention weights.

        Args:
            attention_weights_list: List of attention tensors per layer [batch, heads, N, N]
            batch_idx: Current batch index
        """
        if attention_weights_list is None or len(attention_weights_list) == 0:
            return

        for layer_idx, attn in enumerate(attention_weights_list):
            # attn shape: [batch, heads, N, N]
            # Average across batch and heads for diagnostics
            attn_avg = attn.mean(dim=(0, 1))  # [N, N]

            # Compute entropy: -sum(p * log(p))
            entropy = -(attn_avg * torch.log(attn_avg + 1e-10)).sum(dim=-1).mean()

            # Compute effective rank: exp(entropy)
            effective_rank = torch.exp(entropy)

            # Compute top-k concentration: fraction of attention mass in top-k positions
            top5_vals, _ = torch.topk(attn_avg, k=min(5, attn_avg.shape[-1]), dim=-1)
            top10_vals, _ = torch.topk(attn_avg, k=min(10, attn_avg.shape[-1]), dim=-1)
            top50_vals, _ = torch.topk(attn_avg, k=min(50, attn_avg.shape[-1]), dim=-1)

            concentration_top5 = top5_vals.sum(dim=-1).mean()
            concentration_top10 = top10_vals.sum(dim=-1).mean()
            concentration_top50 = top50_vals.sum(dim=-1).mean()

            # NEW: Max row weight (one-hot detection)
            max_weights = attn_avg.max(dim=-1)[0]  # [N]
            avg_max_weight = max_weights.mean()

            # NEW: Column-sum concentration (sink collapse detection)
            col_sums = attn_avg.sum(dim=0)  # [N] - attention received per gene
            col_sums_normalized = col_sums / (
                col_sums.sum() + 1e-10
            )  # Normalize to prob dist
            col_entropy = -(
                col_sums_normalized * torch.log(col_sums_normalized + 1e-10)
            ).sum()
            max_col_sum = col_sums.max()

            # Initialize accumulator if needed
            if layer_idx not in self.attention_stats_accumulators:
                self.attention_stats_accumulators[layer_idx] = {
                    "entropy_sum": 0.0,
                    "effective_rank_sum": 0.0,
                    "top5_sum": 0.0,
                    "top10_sum": 0.0,
                    "top50_sum": 0.0,
                    "max_row_weight_sum": 0.0,
                    "col_entropy_sum": 0.0,
                    "max_col_sum_sum": 0.0,
                    "count": 0,
                }

            # Accumulate
            acc = self.attention_stats_accumulators[layer_idx]
            acc["entropy_sum"] += entropy.item()
            acc["effective_rank_sum"] += effective_rank.item()
            acc["top5_sum"] += concentration_top5.item()
            acc["top10_sum"] += concentration_top10.item()
            acc["top50_sum"] += concentration_top50.item()
            acc["max_row_weight_sum"] += avg_max_weight.item()
            acc["col_entropy_sum"] += col_entropy.item()
            acc["max_col_sum_sum"] += max_col_sum.item()
            acc["count"] += 1

    def reset_attention_diagnostics(self) -> None:
        """Reset attention diagnostic accumulators."""
        self.attention_stats_accumulators = {}
        self.gradient_norms = {}

    def _accumulate_residual_updates(
        self, x_in: torch.Tensor, x_out: torch.Tensor, layer_idx: int
    ) -> None:
        """Track layer update magnitude relative to input (Tier 1 - very cheap).

        Args:
            x_in: [batch, N+1, d] input to layer
            x_out: [batch, N+1, d] output from layer
            layer_idx: Current transformer layer index
        """
        # Compute ratio: ||x_out - x_in|| / ||x_in||
        update_norm = (
            (x_out - x_in).norm(dim=-1).mean()
        )  # Average over batch and sequence
        input_norm = x_in.norm(dim=-1).mean()
        ratio = (update_norm / (input_norm + 1e-10)).item()

        # Initialize accumulator if needed
        if layer_idx not in self.residual_update_accumulators:
            self.residual_update_accumulators[layer_idx] = {
                "sum_ratio": 0.0,
                "count": 0,
            }

        # Accumulate
        acc = self.residual_update_accumulators[layer_idx]
        acc["sum_ratio"] += ratio
        acc["count"] += 1

    def _accumulate_degree_bias(
        self,
        attention_weights: torch.Tensor,
        graph_name: str,
        layer_idx: int,
        head_idx: int,
    ) -> None:
        """Compute correlation between graph degree and attention received (Tier 2 - medium cost).

        Args:
            attention_weights: [batch, heads, N, N] gene-gene attention weights
            graph_name: Name of the biological graph
            layer_idx: Current transformer layer index
            head_idx: Attention head index
        """
        # Only for graph-regularized layers
        if graph_name not in cast(dict[str, Any], self.model.regularized_head_config):
            return

        # Get adjacency matrix from dict
        adjacency_matrices = cast(
            dict[str, torch.Tensor], self.model.adjacency_matrices
        )
        if graph_name not in adjacency_matrices:
            return
        adj_true = adjacency_matrices[graph_name].to(attention_weights.device)  # [N, N]
        degrees = adj_true.sum(dim=-1).detach().cpu().numpy()  # [N]

        # Average attention across batch
        attn_avg = attention_weights.mean(dim=0)  # [heads, N, N]

        # Extract specific head
        attn_head = attn_avg[head_idx]  # [N, N]

        # Column sums = attention received per node
        col_sums = attn_head.sum(dim=0).detach().cpu().numpy()  # [N]

        # Spearman correlation
        from scipy.stats import spearmanr

        degree_corr, _ = spearmanr(degrees, col_sums)

        # Store in edge recovery accumulator (same tier)
        # IMPORTANT: Use same key format as edge recovery (_L{}_H{} not _layer{}_head{})
        key = f"{graph_name}_L{layer_idx}_H{head_idx}"
        if key not in self.edge_recovery_accumulators:
            # Initialize with complete structure to avoid KeyError in logging
            self.edge_recovery_accumulators[key] = {
                "sum_recall_deg": 0.0,
                "count_nodes_deg": 0,
                "sum_prec": {k: 0.0 for k in self.edge_recovery_ks},
                "count_nodes_prec": {k: 0 for k in self.edge_recovery_ks},
                "sum_edge_mass": 0.0,
                "count_batches": 0,
                "graph_name": graph_name,
                "layer": layer_idx,
                "head": head_idx,
                "degree_correlation_sum": 0.0,
                "degree_corr_count": 0,
            }
        elif "degree_correlation_sum" not in self.edge_recovery_accumulators[key]:
            # Add degree fields if accumulator exists but was created by edge recovery
            self.edge_recovery_accumulators[key]["degree_correlation_sum"] = 0.0
            self.edge_recovery_accumulators[key]["degree_corr_count"] = 0

        self.edge_recovery_accumulators[key]["degree_correlation_sum"] += degree_corr
        self.edge_recovery_accumulators[key]["degree_corr_count"] += 1

    def _plot_attention_diagnostics(self) -> None:
        """Create and log attention diagnostic visualizations."""
        from torchcell.viz.transformer_diagnostics import TransformerDiagnostics

        # Initialize visualization
        vis = TransformerDiagnostics(base_dir=self.trainer.default_root_dir)

        # Prepare attention stats
        attention_stats = {}
        for layer_idx, acc in self.attention_stats_accumulators.items():
            if acc["count"] > 0:
                attention_stats[layer_idx] = {
                    "entropy": acc["entropy_sum"] / acc["count"],
                    "effective_rank": acc["effective_rank_sum"] / acc["count"],
                    "top5": acc["top5_sum"] / acc["count"],
                    "top10": acc["top10_sum"] / acc["count"],
                    "top50": acc["top50_sum"] / acc["count"],
                    "max_row_weight": acc["max_row_weight_sum"] / acc["count"],
                    "col_entropy": acc["col_entropy_sum"] / acc["count"],
                    "max_col_sum": acc["max_col_sum_sum"] / acc["count"],
                }

        # Prepare residual update ratios
        residual_ratios = {}
        for layer_idx, acc in self.residual_update_accumulators.items():
            if acc["count"] > 0:
                residual_ratios[layer_idx] = acc["sum_ratio"] / acc["count"]

        # Plot if we have stats
        if attention_stats:
            vis.plot_attention_diagnostics(
                attention_stats,
                residual_ratios=residual_ratios if residual_ratios else None,
                gradient_norms=self.gradient_norms if self.gradient_norms else None,
                num_epochs=self.current_epoch,
                stage="val",
            )

    def forward(
        self, batch: HeteroData, return_attention: bool = False, **model_kwargs: Any
    ) -> Any:
        """Run a forward pass, optionally returning attention diagnostics.

        ``model_kwargs`` reach the model unchanged (the per-entry path passes
        ``entry_batch`` and ``entry_dataset``).
        """
        # Get device from batch - handle different batch structures
        if hasattr(batch["gene"], "x"):
            batch_device = batch["gene"].x.device
        elif hasattr(batch["gene"], "perturbation_indices"):
            batch_device = batch["gene"].perturbation_indices.device
        elif hasattr(batch["gene"], "phenotype_values"):
            batch_device = batch["gene"].phenotype_values.device
        else:
            # Fallback to model device
            batch_device = next(self.model.parameters()).device

        if (
            not hasattr(self, "_cell_graph_device")
            or self._cell_graph_device != batch_device
        ):
            self.cell_graph = self.cell_graph.to(batch_device)
            self._cell_graph_device = batch_device

        # Return all outputs from the model
        return self.model(
            self.cell_graph, batch, return_attention=return_attention, **model_kwargs
        )

    def _ensure_no_unused_params_loss(self) -> torch.Tensor | int:
        """Add a dummy loss to ensure all parameters are used in backward pass."""
        dummy_loss: torch.Tensor | int = 0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is None:
                dummy_loss = dummy_loss + 0.0 * param.sum()
        return dummy_loss

    def _is_scheduled(self, freq: int | None) -> bool:
        """Is this epoch one of every ``freq``, with a non-positive freq meaning never.

        ``0`` is the natural way to switch a diagnostic off in config, and one of the two
        edge-recovery call sites already read it that way (`is not None and > 0`). The
        other five sites went straight into ``(epoch + 1) % freq``, so
        ``plot_edge_recovery_every_n_epochs: 0`` raised ZeroDivisionError from
        ``_shared_step`` on the first VALIDATION batch. That is 3 minutes into a run,
        after the model, the dataset and the sanity check have all succeeded, and it
        killed a 12 h 4-GPU job (1599) that was otherwise correct.

        Routing every site through here makes 0 mean "never" everywhere rather than
        "never in one place and crash in five".
        """
        return freq is not None and freq > 0 and (self.current_epoch + 1) % freq == 0

    def _shared_step(
        self, batch: HeteroData, batch_idx: int, stage: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # DataLoader profiling mode: Skip model forward, create dummy loss
        if self.execution_mode == "dataloader_profiling":
            # Execute all batch preparation (moving to device happens in forward())
            # Get device from batch - handle different batch structures
            if hasattr(batch["gene"], "x"):
                batch_device = batch["gene"].x.device
            elif hasattr(batch["gene"], "perturbation_indices"):
                batch_device = batch["gene"].perturbation_indices.device
            elif hasattr(batch["gene"], "phenotype_values"):
                batch_device = batch["gene"].phenotype_values.device
            else:
                batch_device = next(self.model.parameters()).device

            # Ensure cell_graph is on correct device
            if (
                not hasattr(self, "_cell_graph_device")
                or self._cell_graph_device != batch_device
            ):
                self.cell_graph = self.cell_graph.to(batch_device)
                self._cell_graph_device = batch_device

            # Create trivial loss that touches ALL model parameters (required for DDP)
            # This ensures no "unused parameters" error in DDP mode
            loss = torch.zeros((), device=batch_device, requires_grad=True)
            for param in self.model.parameters():
                if param.requires_grad:
                    loss = loss + (param * 0.0).sum()

            # Log minimal metrics
            batch_size = self._get_batch_size(batch)
            self.log(
                f"{stage}/dataloader_profile_loss",
                loss,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                f"{stage}/dataloader_profile_batch_size",
                float(batch_size),
                batch_size=batch_size,
                sync_dist=True,
            )

            return loss, None, None

        if self.per_entry:
            return self._shared_step_per_entry(batch, batch_idx, stage)

        # Normal training/validation/test execution
        # Get model outputs - only request attention weights during validation on diagnostic epochs
        if stage == "val":
            # Check if ANY diagnostic tier is scheduled for this epoch
            is_diagnostic_epoch = self._is_scheduled(
                self.hparams.get("plot_transformer_diagnostics_every_n_epochs", 10)
            ) or self._is_scheduled(
                self.hparams.get("plot_edge_recovery_every_n_epochs", 10)
            )
            # The attention the diagnostics read is computed on the wild-type graph
            # before any perturbation, and validation runs in eval mode, so it is the
            # same tensor for every validation batch of the epoch. One batch carries
            # it; the other batches take the fused path. Returning it for all 37 batches
            # of a 256-record validation held eight [1, 9, 6608, 6608] fp32 matrices
            # (11.7 GiB) beside every batch's per-record tensors, the pass that OOM'd
            # Delta 22034666 and 22055149 at epoch 10 (2026-09-15); the diagnostic
            # values themselves were 37 identical samples averaged.
            return_attention = is_diagnostic_epoch and batch_idx == 0

            # Debug: Log attention storage decision (only on rank 0, once per epoch)
            # Commented out to reduce output clutter - these are debug messages, not warnings
            # if batch_idx == 0 and self.trainer.global_rank == 0:
            #     if return_attention:
            #         print(f"Epoch {self.current_epoch}: Storing attention for diagnostics")
            #     else:
            #         print(f"Epoch {self.current_epoch}: Skipping attention storage")
        else:
            return_attention = (
                False  # Never during training (graph reg doesn't need storage)
            )

        predictions, representations = self(batch, return_attention=return_attention)

        # In validation stage, compute edge recovery metrics and attention diagnostics
        if stage == "val":
            if (
                "attention_weights" in representations
                and representations["attention_weights"] is not None
            ):
                attention_weights_list = representations["attention_weights"]

                # TIER 1: Cheap transformer diagnostics (attention stats, entropy, residual ratios)
                if self._is_scheduled(
                    self.hparams.get("plot_transformer_diagnostics_every_n_epochs", 10)
                ):
                    self._accumulate_attention_diagnostics(
                        attention_weights_list, batch_idx
                    )

                # TIER 2: Edge recovery (already has separate frequency control)
                if self._is_scheduled(
                    self.hparams.get("plot_edge_recovery_every_n_epochs", 10)
                ):
                    self._accumulate_edge_recovery_metrics(
                        attention_weights_list, batch_idx
                    )

                # TIER 1: Accumulate residual update ratios (from model)
                if (
                    "residual_update_ratios" in representations
                    and representations["residual_update_ratios"]
                ):
                    for layer_idx, ratio in enumerate(
                        representations["residual_update_ratios"]
                    ):
                        if layer_idx not in self.residual_update_accumulators:
                            self.residual_update_accumulators[layer_idx] = {
                                "sum_ratio": 0.0,
                                "count": 0,
                            }
                        acc = self.residual_update_accumulators[layer_idx]
                        acc["sum_ratio"] += ratio
                        acc["count"] += 1

                # TIER 2: Medium cost - Degree-bias correlation
                # Iterate through regularized heads to compute degree-bias
                if (
                    hasattr(self.model, "regularized_head_config")
                    and self.model.regularized_head_config
                ):
                    for graph_name, config in cast(
                        dict[str, Any], self.model.regularized_head_config
                    ).items():
                        layer_spec = config["layer"]
                        head_idx = config["head"]

                        # Handle both single int and list of ints for layer
                        layer_indices = (
                            layer_spec if isinstance(layer_spec, list) else [layer_spec]
                        )

                        # Process each layer
                        for layer_idx in layer_indices:
                            if layer_idx < len(attention_weights_list):
                                self._accumulate_degree_bias(
                                    attention_weights_list[layer_idx],
                                    graph_name,
                                    layer_idx,
                                    head_idx,
                                )

            # Removed debug print - not a warning, just informational
            # elif batch_idx == 0:  # Print once per validation epoch
            #     print(
            #         "Edge recovery: Model not returning 'attention_weights' in representations"
            #     )

        # Ensure predictions has correct shape (batch_size, 1)
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0).unsqueeze(0)  # Make it [1, 1]
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)  # Make it [batch_size, 1]

        batch_size = predictions.size(0)

        # Get target values - now in COO format
        if self.fitness_lambda is not None:
            # Joint path: two labels per record, so phenotype_values is [2B] and the
            # gene-interaction column has to be picked out by type. The fitness column
            # is read in _fitness_step below.
            gene_interaction_vals = self._coo_label(
                batch, "gene_interaction", batch_size, original=False
            )
            gene_interaction_orig = (
                self._coo_label(batch, "gene_interaction", batch_size, original=True)
                if hasattr(batch["gene"], "phenotype_values_original")
                else gene_interaction_vals
            )
        else:
            # For gene interaction dataset, phenotype_values directly contains the values
            gene_interaction_vals = batch["gene"].phenotype_values

            # For original values, check if there's a phenotype_values_original
            if hasattr(batch["gene"], "phenotype_values_original"):
                gene_interaction_orig = batch["gene"].phenotype_values_original
            else:
                gene_interaction_orig = gene_interaction_vals

        # Handle tensor shape
        if gene_interaction_vals.dim() == 0:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_vals.dim() == 1:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(1)

        # Handle tensor shape
        if gene_interaction_orig.dim() == 0:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_orig.dim() == 1:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(1)

        # Get latent representations from model
        # Backward compatibility: new models return H_genes_pert, old models return z_p
        z_p = representations.get("z_p")
        if z_p is None:
            z_p = representations.get("H_genes_pert")
        H_genes = representations.get("H_genes")  # Pre-perturbation gene embeddings
        H_genes_pert = representations.get(
            "H_genes_pert"
        )  # Post-perturbation gene embeddings
        h_CLS = representations.get("h_CLS")  # CLS token

        # Log CLS token norm
        if h_CLS is not None:
            cls_norm = h_CLS.norm(p=2, dim=-1).mean()
            self.log(
                f"{stage}/cls_token_norm",
                cls_norm,
                batch_size=batch_size,
                sync_dist=True,
            )

        # Perturbed CLS: across-strain spread of the token the whole-cell head reads. The
        # wild-type token measures 0.0 here by construction; this is the number that says
        # the operator actually moved the CLS per strain.
        h_CLS_pert = representations.get("h_CLS_pert")
        if h_CLS_pert is not None and h_CLS_pert.shape[0] > 1:
            self.log(
                f"{stage}/cls_pert_strain_sd",
                h_CLS_pert.detach().float().std(dim=0).mean(),
                batch_size=batch_size,
                sync_dist=True,
            )

        # Residual Update Ratio: Measure how much the transformer changes the embeddings
        if stage == "val" and H_genes is not None and H_genes_pert is not None:
            # Compute the ratio of update magnitude to input magnitude
            residual_norm = (H_genes_pert - H_genes).norm(p=2)
            input_norm = H_genes.norm(p=2)
            residual_ratio = residual_norm / (
                input_norm + 1e-8
            )  # Avoid division by zero

            self.log(
                f"{stage}/residual_update_ratio",
                residual_ratio.item(),
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
            )

        # Calculate loss based on loss function type
        if self.loss_func is None:
            raise ValueError("No loss function provided")

        if isinstance(self.loss_func, LogCoshLoss):
            # For LogCoshLoss, just pass predictions and targets
            loss = self.loss_func(predictions, gene_interaction_vals)
        elif isinstance(self.loss_func, PointDistGraphReg):
            # For PointDistGraphReg, pass predictions, targets, and representations
            # Returns (total_loss, loss_dict) with all components
            total_loss, loss_dict = self.loss_func(
                predictions,
                gene_interaction_vals,
                representations,
                epoch=self.current_epoch,
            )
            loss = total_loss

            # Log all loss components
            if isinstance(loss_dict, dict):
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            self.log(
                                f"{stage}/{key}",
                                value.item(),
                                batch_size=batch_size,
                                sync_dist=True,
                            )
                    elif isinstance(value, (int, float)):
                        self.log(
                            f"{stage}/{key}",
                            value,
                            batch_size=batch_size,
                            sync_dist=True,
                        )
        else:
            # For ICLoss or other custom losses that might use z_p
            # Check if loss function accepts epoch parameter (for MleDistSupCR and MleWassSupCR)
            if z_p is not None:
                if isinstance(self.loss_func, (MleDistSupCR, MleWassSupCR)):
                    loss_output = self.loss_func(
                        predictions,
                        gene_interaction_vals,
                        z_p,
                        epoch=self.current_epoch,
                    )
                else:
                    loss_output = self.loss_func(
                        predictions, gene_interaction_vals, z_p
                    )
            else:
                loss_output = self.loss_func(predictions, gene_interaction_vals)

            # Handle if loss_func returns a tuple (for ICLoss)
            if isinstance(loss_output, tuple):
                loss = loss_output[0]  # First element is the loss
                loss_dict = loss_output[1] if len(loss_output) > 1 else {}

                # Log additional loss components if available
                if isinstance(loss_dict, dict):
                    for key, value in loss_dict.items():
                        if isinstance(value, torch.Tensor):
                            # Handle multi-dimensional tensors
                            if value.numel() == 1:
                                # Single element tensor - log as scalar
                                self.log(
                                    f"{stage}/{key}",
                                    value.item(),
                                    batch_size=batch_size,
                                    sync_dist=True,
                                )
                            elif value.numel() > 1:
                                # Multi-element tensor - log each element separately
                                for i in range(value.numel()):
                                    self.log(
                                        f"{stage}/{key}_{i}",
                                        value[i].item(),
                                        batch_size=batch_size,
                                        sync_dist=True,
                                    )
                            # Skip empty tensors
                        elif isinstance(value, (int, float)):
                            # Handle scalar values
                            self.log(
                                f"{stage}/{key}",
                                value,
                                batch_size=batch_size,
                                sync_dist=True,
                            )
            else:
                loss = loss_output

        # Add graph regularization loss if present (for transformer models)
        # IMPORTANT: Skip if using PointDistGraphReg, as it already includes graph_reg in total
        if "graph_reg_loss" in representations and not isinstance(
            self.loss_func, PointDistGraphReg
        ):
            graph_reg_loss = representations["graph_reg_loss"]
            loss = loss + graph_reg_loss
            self.log(
                f"{stage}/graph_reg_loss",
                graph_reg_loss,
                batch_size=batch_size,
                sync_dist=True,
            )

        # Joint fitness objective on the global head (opt-in, see __init__).
        fitness_term: torch.Tensor | None = None
        if self.fitness_lambda is not None:
            fitness_term = self._fitness_step(batch, representations, batch_size, stage)
            loss = loss + fitness_term

        # Per-term gradient norms on the first batch of a probe epoch (see __init__).
        if (
            stage == "train"
            and batch_idx == 0
            and self.current_epoch in self.gradient_probe_epochs
            and isinstance(self.loss_func, PointDistGraphReg)
        ):
            terms = dict(self.loss_func.last_terms)
            if fitness_term is not None:
                terms["fitness"] = fitness_term
            self._log_gradient_probe(terms, loss, batch_size)

        # Add dummy loss for unused parameters
        dummy_loss = self._ensure_no_unused_params_loss()
        loss = loss + dummy_loss

        # Log the loss
        self.log(f"{stage}/loss", loss, batch_size=batch_size, sync_dist=True)

        # Log z_p norm if available
        if z_p is not None:
            z_p_norm = z_p.norm(p=2, dim=-1).mean()
            self.log(
                f"{stage}/z_p_norm", z_p_norm, batch_size=batch_size, sync_dist=True
            )

        # Log gate weights if available
        if "gate_weights" in representations:
            gate_weights = representations["gate_weights"]
            # Average gate weights across batch
            avg_gate_weights = gate_weights.mean(dim=0)
            self.log(
                f"{stage}/gate_weight_global",
                avg_gate_weights[0],
                batch_size=batch_size,
                sync_dist=True,
            )
            # Only log local weight if it exists (when local predictor is enabled)
            if avg_gate_weights.size(0) > 1:
                self.log(
                    f"{stage}/gate_weight_local",
                    avg_gate_weights[1],
                    batch_size=batch_size,
                    sync_dist=True,
                )

        # Update transformed metrics
        mask = ~torch.isnan(gene_interaction_vals)
        if mask.sum() > 0:
            transformed_metrics = getattr(self, f"{stage}_transformed_metrics")
            transformed_metrics.update(
                predictions[mask].view(-1), gene_interaction_vals[mask].view(-1)
            )

        # Handle inverse transform if available
        inv_predictions = predictions.clone()
        if hasattr(self, "inverse_transform") and self.inverse_transform is not None:
            # Create a temp HeteroData object with predictions in COO format
            temp_data = HeteroData()

            # Create COO format data for predictions
            batch_size = predictions.size(0)
            device = predictions.device
            temp_data["gene"].phenotype_values = predictions.squeeze()
            temp_data["gene"].phenotype_type_indices = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            temp_data["gene"].phenotype_sample_indices = torch.arange(
                batch_size, device=device
            )
            temp_data["gene"].phenotype_types = ["gene_interaction"]

            # Apply the inverse transform
            inv_data = self.inverse_transform(temp_data)

            # Extract the inversed predictions
            inv_gene_int = inv_data["gene"]["phenotype_values"]

            # Handle tensor shape
            if isinstance(inv_gene_int, torch.Tensor):
                if inv_gene_int.dim() == 0:
                    inv_predictions = inv_gene_int.unsqueeze(0).unsqueeze(0)
                elif inv_gene_int.dim() == 1:
                    inv_predictions = inv_gene_int.unsqueeze(1)
                else:
                    inv_predictions = inv_gene_int

        # Update metrics with original scale values
        mask = ~torch.isnan(gene_interaction_orig)
        if mask.sum() > 0:
            metrics = getattr(self, f"{stage}_metrics")
            metrics.update(
                inv_predictions[mask].view(-1), gene_interaction_orig[mask].view(-1)
            )
            self._update_order_metrics(
                stage,
                f"{stage}_order_metrics",
                batch,
                batch_size,
                inv_predictions,
                gene_interaction_orig,
                mask,
            )

        # Collect samples for visualization
        if stage == "train" and self._is_scheduled(self.hparams["plot_every_n_epochs"]):
            current_count = sum(t.size(0) for t in self.train_samples["true_values"])
            if current_count < self.hparams["plot_sample_ceiling"]:
                remaining = self.hparams["plot_sample_ceiling"] - current_count
                if batch_size > remaining:
                    idx = torch.randperm(batch_size)[:remaining]
                    self.train_samples["true_values"].append(
                        gene_interaction_orig[idx].detach()
                    )
                    self.train_samples["predictions"].append(
                        inv_predictions[idx].detach()
                    )

                    # Collect latent representations for UMAP visualization
                    if "latents" not in self.train_samples:
                        self.train_samples["latents"] = {}

                    # Collect H_pooled (combines h_CLS with perturbed genes, pooled)
                    if h_CLS is not None and H_genes_pert is not None:
                        if "H_pooled" not in self.train_samples["latents"]:
                            self.train_samples["latents"]["H_pooled"] = []
                        # Pool immediately: mean([h_CLS, H_genes_pert]) -> [batch, d]
                        h_CLS_batched = (
                            h_CLS.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
                        )  # [batch, 1, d]
                        H_combined = torch.cat(
                            [h_CLS_batched, H_genes_pert], dim=1
                        )  # [batch, N+1, d]
                        H_pooled = H_combined.mean(dim=1)  # [batch, d]
                        self.train_samples["latents"]["H_pooled"].append(
                            H_pooled[idx].detach().cpu()
                        )
                else:
                    self.train_samples["true_values"].append(
                        gene_interaction_orig.detach()
                    )
                    self.train_samples["predictions"].append(inv_predictions.detach())

                    # Collect latent representations for UMAP visualization
                    if "latents" not in self.train_samples:
                        self.train_samples["latents"] = {}

                    # Collect H_pooled (combines h_CLS with perturbed genes, pooled)
                    if h_CLS is not None and H_genes_pert is not None:
                        if "H_pooled" not in self.train_samples["latents"]:
                            self.train_samples["latents"]["H_pooled"] = []
                        # Pool immediately: mean([h_CLS, H_genes_pert]) -> [batch, d]
                        h_CLS_batched = (
                            h_CLS.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
                        )  # [batch, 1, d]
                        H_combined = torch.cat(
                            [h_CLS_batched, H_genes_pert], dim=1
                        )  # [batch, N+1, d]
                        H_pooled = H_combined.mean(dim=1)  # [batch, d]
                        self.train_samples["latents"]["H_pooled"].append(
                            H_pooled.detach().cpu()
                        )
        elif stage == "val" and self._is_scheduled(self.hparams["plot_every_n_epochs"]):
            # Only collect validation samples on epochs we'll plot, respecting ceiling
            current_count = sum(t.size(0) for t in self.val_samples["true_values"])
            if current_count < self.hparams["plot_sample_ceiling"]:
                remaining = self.hparams["plot_sample_ceiling"] - current_count
                if batch_size > remaining:
                    idx = torch.randperm(batch_size)[:remaining]
                    self.val_samples["true_values"].append(
                        gene_interaction_orig[idx].detach()
                    )
                    self.val_samples["predictions"].append(
                        inv_predictions[idx].detach()
                    )

                    # Collect latent representations for UMAP visualization
                    if "latents" not in self.val_samples:
                        self.val_samples["latents"] = {}

                    # Collect H_pooled (combines h_CLS with perturbed genes, pooled)
                    if h_CLS is not None and H_genes_pert is not None:
                        if "H_pooled" not in self.val_samples["latents"]:
                            self.val_samples["latents"]["H_pooled"] = []
                        # Pool immediately: mean([h_CLS, H_genes_pert]) -> [batch, d]
                        h_CLS_batched = (
                            h_CLS.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
                        )  # [batch, 1, d]
                        H_combined = torch.cat(
                            [h_CLS_batched, H_genes_pert], dim=1
                        )  # [batch, N+1, d]
                        H_pooled = H_combined.mean(dim=1)  # [batch, d]
                        self.val_samples["latents"]["H_pooled"].append(
                            H_pooled[idx].detach().cpu()
                        )
                else:
                    self.val_samples["true_values"].append(
                        gene_interaction_orig.detach()
                    )
                    self.val_samples["predictions"].append(inv_predictions.detach())

                    # Collect latent representations for UMAP visualization
                    if "latents" not in self.val_samples:
                        self.val_samples["latents"] = {}

                    # Collect H_pooled (combines h_CLS with perturbed genes, pooled)
                    if h_CLS is not None and H_genes_pert is not None:
                        if "H_pooled" not in self.val_samples["latents"]:
                            self.val_samples["latents"]["H_pooled"] = []
                        # Pool immediately: mean([h_CLS, H_genes_pert]) -> [batch, d]
                        h_CLS_batched = (
                            h_CLS.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
                        )  # [batch, 1, d]
                        H_combined = torch.cat(
                            [h_CLS_batched, H_genes_pert], dim=1
                        )  # [batch, N+1, d]
                        H_pooled = H_combined.mean(dim=1)  # [batch, d]
                        self.val_samples["latents"]["H_pooled"].append(
                            H_pooled.detach().cpu()
                        )
        elif stage == "test":
            # For test, always collect samples (no epoch check since test runs once)
            self.test_samples["true_values"].append(gene_interaction_orig.detach())
            self.test_samples["predictions"].append(inv_predictions.detach())

            # Collect latent representations for UMAP visualization
            if "latents" not in self.test_samples:
                self.test_samples["latents"] = {}

            # Collect H_pooled (combines h_CLS with perturbed genes, pooled)
            if h_CLS is not None and H_genes_pert is not None:
                if "H_pooled" not in self.test_samples["latents"]:
                    self.test_samples["latents"]["H_pooled"] = []
                # Pool immediately: mean([h_CLS, H_genes_pert]) -> [batch, d]
                h_CLS_batched = (
                    h_CLS.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
                )  # [batch, 1, d]
                H_combined = torch.cat(
                    [h_CLS_batched, H_genes_pert], dim=1
                )  # [batch, N+1, d]
                H_pooled = H_combined.mean(dim=1)  # [batch, d]
                self.test_samples["latents"]["H_pooled"].append(H_pooled.detach().cpu())

        return loss, predictions, gene_interaction_orig

    def _interaction_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        representations: dict[str, Any],
        stage: str,
        batch_size: int,
    ) -> torch.Tensor:
        """The configured interaction loss on ``[N, 1]`` rows, components logged.

        The per-entry path's counterpart of the inline block of ``_shared_step``:
        PointDistGraphReg (point + distribution + graph prior, its dict logged),
        LogCosh, or a plain module; the graph prior is added once, by whichever of
        the two owns it.
        """
        if self.loss_func is None:
            raise ValueError("No loss function provided")
        if isinstance(self.loss_func, PointDistGraphReg):
            loss, loss_dict = self.loss_func(
                predictions, targets, representations, epoch=self.current_epoch
            )
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.log(
                        f"{stage}/{key}",
                        value.item(),
                        batch_size=batch_size,
                        sync_dist=True,
                    )
                elif isinstance(value, (int, float)):
                    self.log(
                        f"{stage}/{key}", value, batch_size=batch_size, sync_dist=True
                    )
            return cast(torch.Tensor, loss)
        loss = self.loss_func(predictions, targets)
        if isinstance(loss, tuple):
            loss = loss[0]
        if "graph_reg_loss" in representations:
            graph_reg_loss = representations["graph_reg_loss"]
            loss = loss + graph_reg_loss
            self.log(
                f"{stage}/graph_reg_loss",
                graph_reg_loss,
                batch_size=batch_size,
                sync_dist=True,
            )
        return cast(torch.Tensor, loss)

    def _per_entry_label_metrics(
        self,
        batch: HeteroData,
        stage: str,
        label: str,
        preds_norm: torch.Tensor,
        targets_norm: torch.Tensor,
        targets_orig: torch.Tensor,
        rows: torch.Tensor,
        tokens: torch.Tensor,
        batch_size: int,
        order_attr: str,
        metrics_attr: str,
        transformed_attr: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Feed every metric collection of ``label`` from its entry rows.

        Returns the original-scale (prediction, target) pairs used for the primary
        collection: every entry row in training, the policy-reduced one per genotype
        in validation and test.
        """
        self._entry_counts[stage][label] += int(rows.numel())
        preds_orig = self._inverse_scalar(preds_norm.detach(), label)
        # transformed (normalized-scale) and per-entry original-scale collections
        getattr(self, transformed_attr).update(
            preds_norm.detach().view(-1), targets_norm.view(-1)
        )
        getattr(self, f"{stage}_entry_metrics")[label].update(
            preds_orig.view(-1), targets_orig.view(-1)
        )
        self._update_token_metrics(
            stage, label, tokens, preds_orig.view(-1), targets_orig.view(-1)
        )
        order_all = self._perturbation_order(batch, batch_size)
        if stage == "train":
            primary_pred, primary_target = preds_orig.view(-1), targets_orig.view(-1)
            order = order_all[rows]
        else:
            target_r, pred_r, present = self._policy_reduce(
                rows, targets_orig.view(-1), preds_orig.view(-1), tokens, batch_size
            )
            primary_pred, primary_target = pred_r[present], target_r[present]
            order = order_all[present]
        getattr(self, metrics_attr).update(primary_pred, primary_target)
        self._update_order_metrics_rows(
            stage, order_attr, order, primary_pred, primary_target
        )
        return primary_pred, primary_target

    def _shared_step_per_entry(
        self, batch: HeteroData, batch_idx: int, stage: str
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Loss and metrics on ENTRY rows under the source-dataset token."""
        gene = batch["gene"]
        batch_size = self._get_batch_size(batch)
        entry_batch = gene.phenotype_values_batch
        entry_dataset = gene.phenotype_dataset_indices
        predictions, representations = self(
            batch,
            return_attention=False,
            entry_batch=entry_batch,
            entry_dataset=entry_dataset,
        )  # predictions [E, 1]

        rows, vals, orig, toks, sel = self._entry_rows(batch, "gene_interaction")
        pred_gi = predictions[sel]  # [E_gi, 1]
        loss = self._interaction_loss(
            pred_gi, vals.unsqueeze(1), representations, stage, batch_size
        )
        gi_pred_primary, gi_target_primary = self._per_entry_label_metrics(
            batch,
            stage,
            "gene_interaction",
            pred_gi,
            vals,
            orig,
            rows,
            toks,
            batch_size,
            f"{stage}_order_metrics",
            f"{stage}_metrics",
            f"{stage}_transformed_metrics",
        )

        # Cross-token ablation: the Kuzmin 2018 rows re-read under the 2020 token and
        # the 2020 rows under the 2018 token, on the encoder outputs already computed.
        if self._crosstoken_pair is not None and stage != "train":
            a, b = self._crosstoken_pair
            swap = (toks == a) | (toks == b)
            if bool(swap.any()):
                swapped = toks.clone()
                swapped[toks == a] = b
                swapped[toks == b] = a
                with torch.no_grad():
                    cross_pred, _ = cast(Any, self.model).entry_readout(
                        representations, batch, rows[swap], swapped[swap]
                    )
                    cross_orig = self._inverse_scalar(cross_pred, "gene_interaction")
                getattr(self, f"{stage}_crosstoken_metric").update(
                    cross_orig.view(-1), orig[swap].view(-1)
                )

        fitness_term: torch.Tensor | None = None
        if self.fitness_lambda is not None:
            fit_all = representations["head_outputs"]["global"]  # [E, 1]
            rows_f, vals_f, orig_f, toks_f, sel_f = self._entry_rows(batch, "fitness")
            pred_f = fit_all[sel_f]
            fitness_loss = nn.functional.mse_loss(pred_f, vals_f.unsqueeze(1))
            self.log(
                f"{stage}/fitness_loss",
                fitness_loss,
                batch_size=batch_size,
                sync_dist=True,
            )
            self._per_entry_label_metrics(
                batch,
                stage,
                "fitness",
                pred_f,
                vals_f,
                orig_f,
                rows_f,
                toks_f,
                batch_size,
                f"{stage}_order_fitness_metrics",
                f"{stage}_fitness_metrics",
                f"{stage}_transformed_fitness_metrics",
            )
            fitness_term = self.fitness_lambda * fitness_loss
            loss = loss + fitness_term

        if (
            stage == "train"
            and batch_idx == 0
            and self.current_epoch in self.gradient_probe_epochs
            and isinstance(self.loss_func, PointDistGraphReg)
        ):
            terms = dict(self.loss_func.last_terms)
            if fitness_term is not None:
                terms["fitness"] = fitness_term
            self._log_gradient_probe(terms, loss, batch_size)

        loss = loss + self._ensure_no_unused_params_loss()
        self.log(f"{stage}/loss", loss, batch_size=batch_size, sync_dist=True)
        h_CLS_pert = representations.get("h_CLS_pert")
        if h_CLS_pert is not None and h_CLS_pert.shape[0] > 1:
            self.log(
                f"{stage}/cls_pert_strain_sd",
                h_CLS_pert.detach().float().std(dim=0).mean(),
                batch_size=batch_size,
                sync_dist=True,
            )

        samples = getattr(self, f"{stage}_samples")
        if stage == "test" or self._is_scheduled(self.hparams["plot_every_n_epochs"]):
            if (
                sum(t.size(0) for t in samples["true_values"])
                < self.hparams["plot_sample_ceiling"]
            ):
                samples["true_values"].append(gi_target_primary.detach().view(-1, 1))
                samples["predictions"].append(gi_pred_primary.detach().view(-1, 1))
        return loss, gi_pred_primary.view(-1, 1), gi_target_primary.view(-1, 1)

    def _essentiality_step(self, batch: HeteroData) -> None:
        """Buffer each held-out single's fitness under the Costanzo and SGD tokens."""
        assert self.essentiality_eval is not None
        batch_size = self._get_batch_size(batch)
        order = self._perturbation_order(batch, batch_size)
        if bool((order != 1).any()):
            raise ValueError(
                "the essentiality loader must hold single-deletion genotypes only"
            )
        device = batch["gene"].perturbation_indices.device
        entry_batch = torch.arange(batch_size, device=device)
        smf = torch.full(
            (batch_size,), self.essentiality_eval["token_smf"], device=device
        )
        sgd = torch.full(
            (batch_size,), self.essentiality_eval["token_sgd"], device=device
        )
        with torch.no_grad():
            _, representations = self(
                batch,
                return_attention=False,
                entry_batch=entry_batch,
                entry_dataset=smf,
            )
            fit_smf = representations["head_outputs"]["global"].view(-1)
            _, fit_sgd_out = cast(Any, self.model).entry_readout(
                representations, batch, entry_batch, sgd
            )
            assert fit_sgd_out is not None
        self._ess_buffer["node"].append(
            batch["gene"].perturbation_indices.detach().cpu()
        )
        self._ess_buffer["smf"].append(fit_smf.detach().float().cpu())
        self._ess_buffer["sgd"].append(fit_sgd_out.view(-1).detach().float().cpu())

    def _log_essentiality_epoch(self) -> None:
        """AUROC of ``-fitness`` against the essential label, per set and token."""
        if self.essentiality_eval is None:
            return
        from torchmetrics.functional.classification import binary_auroc

        local = {
            k: (torch.cat(v) if v else torch.zeros(0))
            for k, v in self._ess_buffer.items()
        }
        self._ess_buffer = {"node": [], "smf": [], "sgd": []}
        gathered: list[dict[str, torch.Tensor]] = [local]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            holder: list[Any] = [None] * world
            torch.distributed.all_gather_object(holder, local)
            gathered = cast(list[dict[str, torch.Tensor]], holder)
        node = torch.cat([g["node"] for g in gathered]).long()
        preds = {
            token: torch.cat([g[token] for g in gathered]) for token in ("smf", "sgd")
        }
        for set_name in ("released", "matched"):
            labels_map = self.essentiality_eval[set_name]
            keep = torch.tensor([int(n) in labels_map for n in node.tolist()])
            y = torch.tensor(
                [labels_map[int(n)] for n in node[keep].tolist()], dtype=torch.long
            )
            for token in ("smf", "sgd"):
                score = -preds[token][keep]
                value = (
                    binary_auroc(score, y)
                    if y.numel() >= 2 and y.min() != y.max()
                    else torch.tensor(float("nan"))
                )
                self.log(f"val_ess/auroc_{set_name}_{token}", value, sync_dist=False)
            self.log(f"val_ess/n_{set_name}", float(y.numel()), sync_dist=False)

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Run a manual-optimization training step with gradient accumulation."""
        loss, _, _ = self._shared_step(batch, batch_idx, "train")

        # Model profiling mode: Skip optimizer step to isolate model compute
        if self.execution_mode == "model_profiling":
            return loss

        # Get batch size using helper method
        batch_size = self._get_batch_size(batch)

        # Normal training: Run optimizer
        if self.hparams["grad_accumulation_schedule"] is not None:
            loss = loss / self.current_accumulation_steps
        opt = cast(LightningOptimizer, self.optimizers())
        self.manual_backward(loss)
        if (
            self.hparams["grad_accumulation_schedule"] is None
            or (batch_idx + 1) % self.current_accumulation_steps == 0
        ):
            if self.hparams["clip_grad_norm"]:
                nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=self.hparams["clip_grad_norm_max_norm"]
                )
            opt.step()
            opt.zero_grad()
        self.log(
            "learning_rate",
            cast(LightningOptimizer, self.optimizers()).param_groups[0]["lr"],
            batch_size=batch_size,
            sync_dist=True,
        )
        # Log effective batch size when using gradient accumulation
        if self.hparams["grad_accumulation_schedule"] is not None:
            # Get world size for DDP
            world_size = 1
            if hasattr(self.trainer, "strategy") and hasattr(
                self.trainer.strategy, "_strategy_name"
            ):
                if self.trainer.strategy._strategy_name == "ddp":
                    import torch.distributed as dist

                    if dist.is_initialized():
                        world_size = dist.get_world_size()

            effective_batch_size = (
                batch_size * self.current_accumulation_steps * world_size
            )
            self.log(
                "effective_batch_size",
                effective_batch_size,
                batch_size=batch_size,
                sync_dist=True,
            )
        # print(f"Loss: {loss}")
        return loss

    def validation_step(
        self, batch: HeteroData, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor | None:
        """Run the validation shared step and periodically free CUDA cache.

        ``dataloader_idx`` 1 is the essentiality holdout loader (see
        ``CellDataModule.extra_val_indices``); it is scored, not trained or plotted.
        """
        if dataloader_idx == 1:
            self._essentiality_step(batch)
            return None
        if dataloader_idx > 1:
            raise ValueError(f"unexpected validation dataloader_idx {dataloader_idx}")
        loss, _, _ = self._shared_step(batch, batch_idx, "val")

        # Defragment GPU memory every 50 batches to prevent OOM from fragmentation
        if batch_idx > 0 and batch_idx % 50 == 0:
            torch.cuda.empty_cache()

        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Run the shared step for the test stage and return the loss."""
        loss, _, _ = self._shared_step(batch, batch_idx, "test")
        return loss

    def _compute_metrics_safely(
        self, metrics_dict: MetricCollection
    ) -> dict[str, torch.Tensor]:
        results: dict[str, torch.Tensor] = {}
        for metric_name, metric in metrics_dict.items():
            try:
                results[metric_name] = metric.compute()
            except ValueError as e:
                if any(
                    msg in str(e)
                    for msg in [
                        "Needs at least two samples",
                        "No samples to concatenate",
                    ]
                ):
                    continue
                raise e
        return results

    def _plot_samples(self, samples: dict[str, Any], stage: str) -> None:
        if not samples["true_values"]:
            return

        true_values = torch.cat(samples["true_values"], dim=0)
        predictions = torch.cat(samples["predictions"], dim=0)

        # Process latents if they exist
        latents = {}
        if "latents" in samples and samples["latents"]:
            for k, v in samples["latents"].items():
                if v:  # Check if the list is not empty
                    # Ensure all tensors are 2D before concatenating (defensive check)
                    tensors_2d = [t.unsqueeze(0) if t.ndim == 1 else t for t in v]
                    latents[k] = torch.cat(tensors_2d, dim=0)

        max_samples = self.hparams["plot_sample_ceiling"]
        if true_values.size(0) > max_samples:
            idx = torch.randperm(true_values.size(0))[:max_samples]
            true_values = true_values[idx]
            predictions = predictions[idx]
            for key in latents:
                latents[key] = latents[key][idx]

        # Use Visualization for enhanced plotting
        vis = Visualization(
            base_dir=self.trainer.default_root_dir, max_points=max_samples
        )

        loss_name = (
            self.loss_func.__class__.__name__ if self.loss_func is not None else "Loss"
        )

        # Ensure data is in the correct format for visualization
        # For gene interactions, we only need a single dimension
        if true_values.dim() == 1:
            true_values = true_values.unsqueeze(1)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)

        # Collect sample-level latent representations for UMAP visualization
        sample_latents = {}

        # Use pre-computed H_pooled (already combines h_CLS with H_genes_pert)
        if "H_pooled" in latents:
            sample_latents["H_pooled"] = latents["H_pooled"]

        # Use our updated visualize_model_outputs method
        vis.visualize_model_outputs(
            predictions,
            true_values,
            sample_latents,
            loss_name,
            self.current_epoch,
            None,
            stage=stage,
        )

        # Log oversmoothing metrics on latent spaces if available
        if "z_p" in latents:
            smoothness = VisGraphDegen.compute_smoothness(latents["z_p"])
            wandb.log({f"{stage}/oversmoothing_z_p": smoothness.item()})

        # Log genetic interaction box plot
        if torch.any(~torch.isnan(true_values)):
            # For gene interactions, values are in the first dimension
            fig_gi = genetic_interaction_score.box_plot(
                true_values[:, 0].cpu(), predictions[:, 0].cpu()
            )
            wandb.log({f"{stage}/gene_interaction_box_plot": wandb.Image(fig_gi)})
            plt.close(fig_gi)

    def on_train_epoch_end(self) -> None:
        """Log metrics, plot samples, step the scheduler, and clear CUDA cache."""
        # Peak GPU memory of the epoch, so the margin to the card is a logged number
        # rather than something read off an OOM traceback after the fact.
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 2**30
            reserved_gb = torch.cuda.max_memory_reserved() / 2**30
            print(
                f"epoch {self.current_epoch} rank {self.global_rank}: peak allocated "
                f"{peak_gb:.2f} GiB, peak reserved {reserved_gb:.2f} GiB"
            )
            self.log("train/cuda_peak_allocated_gb", peak_gb, sync_dist=True)
            self.log("train/cuda_peak_reserved_gb", reserved_gb, sync_dist=True)
            torch.cuda.reset_peak_memory_stats()
        # Log training metrics
        computed_metrics = self._compute_metrics_safely(self.train_metrics)
        for name, value in computed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.train_metrics.reset()

        # Compute and log transformed metrics
        transformed_metrics = self._compute_metrics_safely(
            self.train_transformed_metrics
        )
        for name, value in transformed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.train_transformed_metrics.reset()
        self._log_fitness_epoch_metrics("train")
        self._log_order_epoch_metrics("train")
        self._log_per_entry_epoch_metrics("train")

        # Plot training samples
        if (
            self._is_scheduled(self.hparams["plot_every_n_epochs"])
            and self.train_samples["true_values"]
        ):
            self._plot_samples(self.train_samples, "train_sample")
            # Reset the sample containers
            self.train_samples = {"true_values": [], "predictions": [], "latents": {}}

        # Step the scheduler when using manual optimization
        sch = self.lr_schedulers()
        if sch is not None:
            # Lightning returns a list of schedulers even if there's only one
            active_sch: LRScheduler | ReduceLROnPlateau
            if isinstance(sch, list) and len(sch) > 0:
                active_sch = sch[0]
            else:
                active_sch = cast("LRScheduler | ReduceLROnPlateau", sch)
            # Manual-optimization schedulers stepped here are epoch-interval
            # LRSchedulers, never ReduceLROnPlateau (which Lightning drives via
            # its monitor); narrow so step() needs no metric argument.
            assert not isinstance(active_sch, ReduceLROnPlateau)
            active_sch.step()

        # CRITICAL: Clear GPU memory at end of training epoch
        # This ensures validation starts with maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_epoch_start(self) -> None:
        """Update gradient accumulation steps and clear training sample buffers."""
        # Update gradient accumulation steps based on current epoch
        if self.hparams["grad_accumulation_schedule"] is not None:
            for epoch_threshold in sorted(
                self.hparams["grad_accumulation_schedule"].keys()
            ):
                # Convert epoch_threshold to int if it's a string
                epoch_threshold_int = (
                    int(epoch_threshold)
                    if isinstance(epoch_threshold, str)
                    else epoch_threshold
                )
                if self.current_epoch >= epoch_threshold_int:
                    self.current_accumulation_steps = self.hparams[
                        "grad_accumulation_schedule"
                    ][epoch_threshold]
            print(
                f"Epoch {self.current_epoch}: Using gradient accumulation steps = {self.current_accumulation_steps}"
            )

        # ALWAYS clear sample containers at epoch start to prevent memory accumulation
        # (unconditional - fixes OOM from stale samples persisting across epochs)
        self.train_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_validation_epoch_start(self) -> None:
        """Free CUDA memory, reset diagnostics, and clear validation sample buffers."""
        # CRITICAL: Aggressively clear GPU memory before validation starts
        # This prevents OOM when transitioning from training to validation
        # Training state (optimizer, gradients, cached activations) can fragment memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronize to ensure all pending operations complete before validation
            torch.cuda.synchronize()

        # Reset edge recovery accumulators and attention diagnostics
        self.reset_edge_recovery_accumulators()
        self.reset_attention_diagnostics()

        # ALWAYS clear sample containers at validation start to prevent memory accumulation
        # (unconditional - fixes OOM from stale samples persisting across epochs)
        self.val_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_test_epoch_start(self) -> None:
        """Clear test sample buffers at the start of the test epoch."""
        # Always clear sample containers for test (test runs only once)
        self.test_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_validation_epoch_end(self) -> None:
        """Log and reset validation metrics and plot validation samples periodically."""
        # Peak GPU memory of the validation pass. On a diagnostic epoch the model
        # returns every layer's [1, heads, N+1, N+1] attention, so validation, not
        # training, is where a run is closest to the card (Delta 22034666 and 22055149
        # OOM'd at the epoch-10 diagnostic validation, 2026-09-15).
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 2**30
            print(
                f"val epoch {self.current_epoch} rank {self.global_rank}: peak "
                f"allocated {peak_gb:.2f} GiB"
            )
            self.log("val/cuda_peak_allocated_gb", peak_gb, sync_dist=True)
            torch.cuda.reset_peak_memory_stats()
        # Log validation metrics
        computed_metrics = self._compute_metrics_safely(self.val_metrics)
        for name, value in computed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.val_metrics.reset()

        # Compute and log transformed metrics
        transformed_metrics = self._compute_metrics_safely(self.val_transformed_metrics)
        for name, value in transformed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.val_transformed_metrics.reset()
        self._log_fitness_epoch_metrics("val")
        self._log_order_epoch_metrics("val")
        self._log_per_entry_epoch_metrics("val")
        self._log_essentiality_epoch()

        # Log edge recovery metrics (now includes layer and head info)
        for metric_key, acc in self.edge_recovery_accumulators.items():
            # Recall@degree
            if acc["count_nodes_deg"] > 0:
                recall_deg = acc["sum_recall_deg"] / acc["count_nodes_deg"]
                self.log(
                    f"val_edge_recovery/{metric_key}/recall_at_deg",
                    recall_deg,
                    sync_dist=True,
                )

            # Precision@k for each k
            for k in self.edge_recovery_ks:
                if acc["count_nodes_prec"][k] > 0:
                    prec_k = acc["sum_prec"][k] / acc["count_nodes_prec"][k]
                    self.log(
                        f"val_edge_recovery/{metric_key}/precision_k{k}",
                        prec_k,
                        sync_dist=True,
                    )

        # TIER 1: Plot basic transformer diagnostics (cheap - controlled by separate frequency)
        if not self.trainer.sanity_checking and self._is_scheduled(
            self.hparams.get("plot_transformer_diagnostics_every_n_epochs", 10)
        ):
            if self.attention_stats_accumulators:
                self._plot_attention_diagnostics()

        # TIER 2: Plot edge recovery + degree-bias (medium cost - less frequent)
        if not self.trainer.sanity_checking and self._is_scheduled(
            self.hparams.get("plot_edge_recovery_every_n_epochs", 10)
        ):
            if self.edge_recovery_accumulators:
                self._plot_edge_recovery_metrics()  # Includes degree-bias

        # Reset all accumulators
        self.reset_edge_recovery_accumulators()
        self.reset_attention_diagnostics()

        # Explicit GPU memory cleanup after validation epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Plot validation samples
        if (
            not self.trainer.sanity_checking
            and self._is_scheduled(self.hparams["plot_every_n_epochs"])
            and self.val_samples["true_values"]
        ):
            self._plot_samples(self.val_samples, "val_sample")
            # Reset the sample containers
            self.val_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_test_epoch_end(self) -> None:
        """Log and reset test metrics and plot test samples."""
        # Log test metrics
        computed_metrics = self._compute_metrics_safely(self.test_metrics)
        for name, value in computed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.test_metrics.reset()

        # Compute and log transformed metrics
        transformed_metrics = self._compute_metrics_safely(
            self.test_transformed_metrics
        )
        for name, value in transformed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.test_transformed_metrics.reset()
        self._log_fitness_epoch_metrics("test")
        self._log_order_epoch_metrics("test")
        self._log_per_entry_epoch_metrics("test")

        # Plot test samples
        if self.test_samples["true_values"]:
            self._plot_samples(self.test_samples, "test_sample")
            # Reset the sample containers
            self.test_samples = {"true_values": [], "predictions": [], "latents": {}}

    def configure_optimizers(self) -> Any:  # Lightning accepts optimizer or config dict
        """Build the optimizer and learning-rate scheduler from hparams config."""
        optimizer_config = self.hparams["optimizer_config"]
        optimizer_class = getattr(torch.optim, optimizer_config["type"])
        optimizer_params = {k: v for k, v in optimizer_config.items() if k != "type"}
        if "learning_rate" in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")
        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        # If no lr_scheduler_config is provided, return just the optimizer
        lr_scheduler_config = self.hparams["lr_scheduler_config"]
        if lr_scheduler_config is None:
            return optimizer

        # Handle different scheduler types
        scheduler_type = lr_scheduler_config.get("type", "ReduceLROnPlateau")
        scheduler_params = {k: v for k, v in lr_scheduler_config.items() if k != "type"}

        scheduler: LRScheduler | ReduceLROnPlateau
        if scheduler_type == "CosineAnnealingWarmupRestarts":
            # Import the custom scheduler
            from torchcell.scheduler.cosine_annealing_warmup import (
                CosineAnnealingWarmupRestarts,
            )

            scheduler = CosineAnnealingWarmupRestarts(optimizer, **scheduler_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_params
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            # Default to ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/gene_interaction/MSE",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
