# experiments/030-solid-growth-multi/scripts/arm_030.py
# [[experiments.030-solid-growth-multi.scripts.arm_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/arm_030
"""The arm definition shared by every 030 per-entry run: records, split, tokens, holdout.

Four scripts read the same committed artifacts and must agree on what they mean: the
training script, the split-cache warmer, the normalization-stats writer and the smoke
check. Each of them calls into here rather than re-deriving the pool, so a mismatch
between "the records that trained" and "the records the constants were fitted on" is
not possible by construction.

An arm is:

- ``subset.indices``: the record pool (S3, ``subset_S3_indices.json.gz``).
- ``subset.exclude``: the essentiality holdout (``essentiality_holdout_030.json.gz``);
  its ``excluded_record_indices`` leave the pool and are served as the second
  validation loader ``val_ess`` (``CellDataModule.extra_val_indices``).
- ``subset.split_file`` / ``split_key``: the pinned 010 random split over the triples,
  carried to 030 by gene-set identity; with ``unpinned_to_train`` every other pool
  record trains.
- ``subset.vocabulary``: the sorted source-dataset names of the build (the token
  vocabulary, ``subset_definitions_030_summary.json``), optionally extended by the
  smoke test's synthetic name.

``train_records`` is the exact training set under these rules and ``index_sha256`` its
fingerprint; the normalization-stats file records the fingerprint of the set it was
fitted on and the training script refuses a mismatch.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import os.path as osp
from collections.abc import Iterable, Mapping
from typing import Any

from lightning.pytorch.callbacks import Callback
from pydantic import BaseModel, Field

EXPERIMENT = "030-solid-growth-multi"


def results_dir() -> str:
    """``experiments/030-solid-growth-multi/results`` under ``EXPERIMENT_ROOT``."""
    return osp.join(os.environ["EXPERIMENT_ROOT"], EXPERIMENT, "results")


def load_json_artifact(name: str) -> Any:
    """Read a committed ``results/`` artifact, gzip-compressed or plain by extension."""
    path = osp.join(results_dir(), name)
    if name.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def index_sha256(indices: Iterable[int]) -> str:
    """Fingerprint of a record-index set (sorted, comma-joined, sha256)."""
    payload = ",".join(map(str, sorted(set(int(i) for i in indices)))).encode()
    return hashlib.sha256(payload).hexdigest()


class EssentialitySet(BaseModel):
    """One held-out gene set of the essentiality holdout, labels by gene name."""

    genes: list[str]
    labels: dict[str, int]
    record_indices: list[int]


class Arm030(BaseModel):
    """The resolved arm: pool, split, holdout and token vocabulary."""

    subset_name: str
    split_file: str
    split_key: str
    exclude_name: str
    unpinned_to_train: bool
    pool: list[int] = Field(description="subset minus the excluded holdout records")
    pinned: dict[str, list[int]]
    excluded: list[int]
    released: EssentialitySet
    matched: EssentialitySet
    dataset_vocabulary: list[str]

    def train_records(self) -> list[int]:
        """Records that train: the pinned train plus, with ``unpinned_to_train``,
        every pool record the pin does not hold out.
        """
        pool = set(self.pool)
        if self.unpinned_to_train:
            held = set(self.pinned["val"]) | set(self.pinned["test"])
            return sorted(pool - held)
        return sorted(set(self.pinned["train"]) & pool)

    def realized(self, split: str) -> set[int]:
        """The records the data module must place in ``split``."""
        if split == "train":
            return set(self.train_records())
        return set(self.pinned[split]) & set(self.pool)

    def essentiality_node_labels(
        self, node_ids: list[str]
    ) -> dict[str, dict[int, int]]:
        """``{"released": {node_index: label}, "matched": {...}}`` over the cell graph."""
        position = {gene: i for i, gene in enumerate(node_ids)}
        out: dict[str, dict[int, int]] = {}
        for name, held in (("released", self.released), ("matched", self.matched)):
            missing = [g for g in held.labels if g not in position]
            if missing:
                raise ValueError(
                    f"{name}: {len(missing)} holdout genes are not cell-graph nodes, "
                    f"first {missing[:5]}"
                )
            out[name] = {position[g]: int(v) for g, v in held.labels.items()}
        return out


def resolve_arm(
    subset_cfg: Mapping[str, Any], extra_vocabulary: Iterable[str] = ()
) -> Arm030:
    """Read the arm's artifacts and check them against each other.

    Raises when the holdout overlaps the pinned val or test sets, when an excluded
    record is outside the pool, or when the vocabulary is not sorted and unique.
    """
    subset_name = str(subset_cfg["indices"])
    split_file = str(subset_cfg["split_file"])
    split_key = str(subset_cfg["split_key"])
    exclude_name = str(subset_cfg["exclude"])
    vocab_name = str(subset_cfg["vocabulary"])

    subset = [int(i) for i in load_json_artifact(subset_name)]
    pinned_raw = load_json_artifact(split_file)[split_key]
    pinned = {k: [int(i) for i in pinned_raw[k]] for k in ("train", "val", "test")}
    holdout = load_json_artifact(exclude_name)
    excluded = [int(i) for i in holdout["excluded_record_indices"]]
    vocabulary = list(load_json_artifact(vocab_name)["dataset_vocabulary"])
    if vocabulary != sorted(set(vocabulary)):
        raise ValueError("dataset_vocabulary must be sorted and unique")
    for name in extra_vocabulary:
        if name in vocabulary:
            raise ValueError(f"{name!r} is already a source dataset")
        vocabulary.append(str(name))

    subset_set = set(subset)
    excluded_set = set(excluded)
    outside = excluded_set - subset_set
    if outside:
        raise ValueError(f"{len(outside)} excluded records are outside the subset")
    held = set(pinned["val"]) | set(pinned["test"])
    if excluded_set & held:
        raise ValueError("the essentiality holdout overlaps the pinned val/test")

    return Arm030(
        subset_name=subset_name,
        split_file=split_file,
        split_key=split_key,
        exclude_name=exclude_name,
        unpinned_to_train=bool(subset_cfg.get("unpinned_to_train", False)),
        pool=sorted(subset_set - excluded_set),
        pinned=pinned,
        excluded=sorted(excluded_set),
        released=EssentialitySet(**holdout["released"]),
        matched=EssentialitySet(**holdout["matched"]),
        dataset_vocabulary=vocabulary,
    )


class NormalizationStats030(BaseModel):
    """Per-label normalization constants fitted on the arm's training ENTRY rows."""

    build: str
    entries_parquet: str
    subset_name: str
    split_file: str
    exclude_name: str
    unpinned_to_train: bool
    n_train_records: int
    train_index_sha256: str
    n_rows: dict[str, int]
    stats: dict[str, dict[str, float]]


def build_dataset(
    data_root: str,
    dataset_root: str,
    query: str,
    graph_names: list[str],
    node_embedding_names: list[str],
    phenotype_labels: list[str],
    dataset_vocabulary: list[str],
    rank: int = 0,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """The 030 dataset with the per-entry processor: (dataset, genome, graph, embeddings).

    The same construction for training and for the cache warmer; the deduplicator is
    ``None`` because the 030 build merged nothing, and the processed store is what is
    read (PyG skips ``process`` when ``processed/`` exists).
    """
    from torchcell.data import GenotypeAggregator, Neo4jCellDataset
    from torchcell.data.graph_processor import Perturbation
    from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.graph.graph import build_gene_multigraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    if rank:
        genome_root = osp.join(data_root, f"data/sgd/genome_{rank}")
        go_root = osp.join(data_root, f"data/go/go_{rank}")
    else:
        genome_root = osp.join(data_root, "data/sgd/genome")
        go_root = osp.join(data_root, "data/go")
    genome = SCerevisiaeGenome(
        genome_root=genome_root, go_root=go_root, overwrite=False
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)
    node_embeddings = NodeEmbeddingBuilder.build(
        embedding_names=node_embedding_names,
        data_root=data_root,
        genome=genome,
        graph=graph,
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=gene_multigraph,
        node_embeddings=node_embeddings,
        converter=None,
        deduplicator=None,
        aggregator=GenotypeAggregator,
        graph_processor=Perturbation(dataset_vocabulary=dataset_vocabulary),
        transform=None,
        phenotype_labels=phenotype_labels,
    )
    return dataset, genome, gene_multigraph, node_embeddings


def make_data_module(
    dataset: Any,
    arm: Arm030,
    seed: int,
    dm_cfg: Mapping[str, Any],
    follow_batch: list[str],
) -> Any:
    """``CellDataModule`` over the arm: pool, pinned split, holdout loader."""
    from torchcell.datamodules import CellDataModule

    data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset.root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        index_subset=arm.pool,
        pinned_split_indices=arm.pinned,
        unpinned_to_train=arm.unpinned_to_train,
        extra_val_indices={"val_ess": arm.excluded},
        batch_size=int(dm_cfg["batch_size"]),
        random_seed=seed,
        num_workers=int(dm_cfg["num_workers"]),
        pin_memory=bool(dm_cfg["pin_memory"]),
        prefetch=bool(dm_cfg["prefetch"]),
        prefetch_factor=int(dm_cfg["prefetch_factor"]),
        persistent_workers=bool(dm_cfg["persistent_workers"]),
        follow_batch=follow_batch,
        val_batch_size=dm_cfg.get("val_batch_size"),
    )
    data_module.setup()
    for split in ("train", "val", "test"):
        realized = set(getattr(data_module.index, split))
        requested = arm.realized(split)
        assert realized == requested, (
            f"{split}: realized {len(realized)} records, the arm defines {len(requested)}"
        )
    return data_module


def smoke_twins(arm: Arm030, smoke_cfg: Mapping[str, Any]) -> dict[str, str]:
    """``{source: twin}`` for the smoke: ``smoke.sources`` -> ``Synthetic<source>``."""
    twins = {str(src): f"Synthetic{src}" for src in smoke_cfg["sources"]}
    for src, twin in twins.items():
        if src not in arm.dataset_vocabulary or twin not in arm.dataset_vocabulary:
            raise ValueError(
                f"smoke source {src!r} or twin {twin!r} not in the vocabulary"
            )
    return twins


class SmokeResult(BaseModel):
    """What one smoke run measured after training (plan decision 10)."""

    config_name: str
    seed: int
    wandb_run_id: str
    twins: dict[str, str] = Field(description="source dataset -> its synthetic twin")
    delta: float
    control: bool
    n_batches: int
    n_rows: int = Field(description="real gene_interaction entry rows scored")
    mean_diff: float = Field(
        description="mean of pred(synthetic token) - pred(own token)"
    )
    sd_diff: float = Field(description="sd of that difference over rows")
    mse_own: float = Field(
        description="normalized MSE of real rows under their own token"
    )
    mse_swapped: float = Field(
        description="normalized MSE of real rows under the synthetic token"
    )
    mse_all_rows: float = Field(
        description="normalized MSE over every gene_interaction row, real and clone, under its own token"
    )
    mean_resid_real: float = Field(description="mean(pred - target) over real rows")
    mean_resid_clone: float = Field(description="mean(pred - target) over clone rows")
    callback_metrics: dict[str, float]


def run_smoke_check(
    task: Any,
    loader: Any,
    arm: Arm030,
    config_name: str,
    seed: int,
    smoke_cfg: Mapping[str, Any],
    n_batches: int,
    callback_metrics: Mapping[str, float],
    wandb_run_id: str,
) -> SmokeResult:
    """Score the trained model on validation batches under both tokens.

    For every real ``gene_interaction`` row of a twinned source: its prediction under
    its own token, and the prediction of the same genotype under the source's twin
    token from the same encoder pass (``entry_readout``). The report script turns these into the three PASS/FAIL
    criteria; this function only measures.
    """
    import torch

    twins = smoke_twins(arm, smoke_cfg)
    lookup = torch.full((len(arm.dataset_vocabulary),), -1, dtype=torch.long)
    for src, twin in twins.items():
        lookup[arm.dataset_vocabulary.index(src)] = arm.dataset_vocabulary.index(twin)
    delta = float(smoke_cfg["delta"])
    device = next(task.parameters()).device
    task.eval()
    diffs: list[torch.Tensor] = []
    sq_own: list[torch.Tensor] = []
    sq_swapped: list[torch.Tensor] = []
    sq_all: list[torch.Tensor] = []
    resid_real: list[torch.Tensor] = []
    resid_clone: list[torch.Tensor] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = batch.to(device)
            gene = batch["gene"]
            preds, reps = task(
                batch,
                return_attention=False,
                entry_batch=gene.phenotype_values_batch,
                entry_dataset=gene.phenotype_dataset_indices,
            )
            rows, vals, _, toks, sel = task._entry_rows(batch, "gene_interaction")
            pred = preds[sel].view(-1)
            is_clone = gene.phenotype_synthetic[sel]
            twin_tok = lookup.to(toks.device)[toks]
            # real rows of a twinned source: the population the clones were made from
            own = (~is_clone) & (twin_tok >= 0)
            resid = pred - vals
            sq_all.append(resid.pow(2)[own | is_clone])
            resid_real.append(resid[own])
            resid_clone.append(resid[is_clone])
            own_rows = rows[own]
            own_pred = pred[own]
            own_vals = vals[own]
            swapped_tok = twin_tok[own]
            swapped_pred, _ = task.model.entry_readout(
                reps, batch, own_rows, swapped_tok
            )
            swapped_pred = swapped_pred.view(-1)
            diffs.append(swapped_pred - own_pred)
            sq_own.append((own_pred - own_vals).pow(2))
            sq_swapped.append((swapped_pred - own_vals).pow(2))
    diff = torch.cat(diffs).float()
    return SmokeResult(
        config_name=config_name,
        seed=seed,
        wandb_run_id=wandb_run_id,
        twins=twins,
        delta=delta,
        control=bool(smoke_cfg.get("control", False)),
        n_batches=min(n_batches, len(diffs)),
        n_rows=int(diff.numel()),
        mean_diff=float(diff.mean()),
        sd_diff=float(diff.std()),
        mse_own=float(torch.cat(sq_own).float().mean()),
        mse_swapped=float(torch.cat(sq_swapped).float().mean()),
        mse_all_rows=float(torch.cat(sq_all).float().mean()),
        mean_resid_real=float(torch.cat(resid_real).float().mean()),
        mean_resid_clone=float(torch.cat(resid_clone).float().mean()),
        callback_metrics=dict(callback_metrics),
    )


class SmokeTrajectory(Callback):
    """Log the smoke measurements at every validation epoch end.

    Job 2863's token run measured pred(synthetic) - pred(own) of 0.93 for a 0.3 offset
    after 1,500 steps with nothing recorded in between, so it could not say whether the
    readout was converging on the offset or sitting past it. A few validation batches
    per epoch under both tokens answer that (``smoke/mean_diff`` and friends over
    epochs), at the cost of ``n_batches`` extra forward passes per epoch.
    """

    def __init__(
        self, arm: Arm030, smoke_cfg: Mapping[str, Any], n_batches: int
    ) -> None:
        """Keep the arm, the smoke config and the per-epoch batch budget."""
        super().__init__()
        self.arm = arm
        self.smoke_cfg = dict(smoke_cfg)
        self.n_batches = int(n_batches)

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Measure on the pinned validation loader and log under ``smoke/``."""
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        loaders = trainer.val_dataloaders
        loader = loaders[0] if isinstance(loaders, list) else loaders
        result = run_smoke_check(
            task=pl_module,
            loader=loader,
            arm=self.arm,
            config_name="trajectory",
            seed=0,
            smoke_cfg=self.smoke_cfg,
            n_batches=self.n_batches,
            callback_metrics={},
            wandb_run_id="",
        )
        values = {
            f"smoke/{k}": float(v)
            for k, v in result.model_dump().items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and k not in ("seed", "n_batches")
        }
        pl_module.log_dict(values, rank_zero_only=True)
        print(
            f"smoke epoch {trainer.current_epoch}: mean_diff {result.mean_diff:+.4f} "
            f"sd_diff {result.sd_diff:.4f} mse_own {result.mse_own:.4f} "
            f"mse_swapped {result.mse_swapped:.4f} mse_all {result.mse_all_rows:.4f} "
            f"resid real {result.mean_resid_real:+.4f} clone {result.mean_resid_clone:+.4f}",
            flush=True,
        )
