# experiments/027-betaxanthin-metabolic/scripts/train_bx.py
# [[experiments.027-betaxanthin-metabolic.scripts.train_bx]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/train_bx.py

r"""Betaxanthin-only training cell for the metabolic-module decision.

WHAT THIS DECIDES
-----------------
Whether the enzyme-constrained thermodynamic flux layer, specialized to the betaxanthin
route, predicts the Cachera screen better than (a) the same CGT with no metabolic module and
(b) the published Flux Cone Learning baseline (Merzbacher 2025, RandomForest_Resampled,
Spearman +0.0391 on the 639 pinned test genes).

WHY THIS FILE EXISTS RATHER THAN A 026 OR A 019 RUN
----------------------------------------------------
026's ``train_flux.py`` has the flux layer but the wrong protocol: two heads, a re-rolled
split per seed, and a headline number that is a MAXIMUM over epochs of the same validation
curve used to pick the epoch. 019's ``train_cgt_multitask.py`` has the right protocol --
pinned Merzbacher test split, pinned partition, checkpoint-by-validation, test pass, per-gene
dump -- but it never passes ``flux_layer=`` to ``CellGraphTransformerMetabolism``, so it
cannot build the module at all.

So this file takes 026's flux construction VERBATIM by import (``ARMS``, ``build_dataset``,
``build_flux_layer``, ``extract_targets``, ``BETAXANTHIN_PRECURSORS``) and reimplements only
the loop, where the protocol lives. Nothing about the flux layer is redefined here; if 026's
arm registry changes, this experiment follows it.

THE THREE PROTOCOL CHANGES, each fixing a measured defect
----------------------------------------------------------
1. **One head.** 026 trained betaxanthin and the Mulleder 19 amino acids jointly, so a
   betaxanthin gain could have come from the co-head. The FCL comparison is betaxanthin-only,
   so the training must be too. Mulleder-only genotypes still arrive in the batch; their
   betaxanthin mask is False and they contribute no loss.
2. **The split is pinned; the seed varies weight initialization only.** 026 let ``seed`` set
   both, which is why its paired arm gap carried an across-seed SD of 0.0303 with real labels
   and 0.0722 with permuted ones. 019 measured the same nuisance directly: between-seed SD
   0.0444 against a within-seed across-arm SD of 0.0058.
3. **Selection and reporting use DISJOINT data.** The epoch is chosen by the argmax of a
   centered 5-epoch moving average of validation Pearson; the number reported is the Spearman
   on the pinned test genes at that epoch. A max over epochs of validation is an upward-biased
   order statistic whose bias grows with the epoch budget -- with 353 validation measurements
   its null width is 0.0535, and every arm 026 reported sat inside one null width of every
   other and of its own permuted-label control.

Run one cell by hand from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
        --arm flux_anchored --seed 0 --epochs 2 --smoke
"""

import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import pearsonr, spearmanr

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
CONF_DIR = osp.join(EXP_DIR, "conf")
RESULTS_DIR = osp.join(EXP_DIR, "results")
#: Per-gene test predictions land beside 020's, so the existing FCL head-to-head scripts
#: (`plot_merzbacher_comparison.py`, `evaluate_merzbacher_head_to_head.py`) read them without
#: modification. Re-deriving the FCL labels here would fork their provenance.
TEST_PRED_DIR = os.getenv(
    "TC_TEST_PRED_DIR", osp.join(DATA_ROOT, "test-predictions-027")
)

# 026 owns the flux layer's construction; this experiment owns only the protocol.
#
# `TC_FLUX_SCRIPTS_DIR` exists for one situation and it is worth naming: 027 was authored in a
# worktree whose branch predates the 026 landing, so `$EXPERIMENT_ROOT/026-metabolism-flux/`
# there has no `train_flux.py` and `torchcell/metabolism/` has no `flux_layer.py`. The
# override lets a smoke test run against the primary checkout. On Delta and after the rebase
# it is unset and the default path is the right one.
FLUX_SCRIPTS_DIR = os.getenv(
    "TC_FLUX_SCRIPTS_DIR", osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts")
)
assert osp.isfile(osp.join(FLUX_SCRIPTS_DIR, "train_flux.py")), (
    f"026's train_flux.py is not at {FLUX_SCRIPTS_DIR}. This branch predates the 026 "
    "landing; rebase onto main, or set TC_FLUX_SCRIPTS_DIR for a local smoke test."
)
sys.path.insert(0, FLUX_SCRIPTS_DIR)
from train_flux import (  # noqa: E402
    ARMS,
    BETAXANTHIN_PRECURSORS,
    build_flux_layer,
    extract_targets,
)

# 026's `build_dataset` is deliberately NOT imported. It opens the dataset with two graphs
# and no node embeddings, which is the configuration every weak betaxanthin run used
# (0.04-0.16 val Pearson). The 0.43 runs used nine graphs and prot_T5. See README.md,
# "Design review before launch".
from torchcell.data import (  # noqa: E402
    DeletionKeyedGenotypeAggregator,
    MeanExperimentDeduplicator,
)
from torchcell.data.graph_processor import Perturbation  # noqa: E402
from torchcell.data.neo4j_cell import Neo4jCellDataset  # noqa: E402
from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.graph import SCerevisiaeGraph, build_gene_multigraph  # noqa: E402
from torchcell.losses.distributional import gaussian_crps  # noqa: E402
from torchcell.models.cell_graph_transformer_metabolism import (  # noqa: E402
    CellGraphTransformerMetabolism,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

HEAD = "betaxanthin"

# 026's module globals came in with its `build_dataset`; this file defines its own, from
# the same config fields the arms already declare.
DATASET_ROOT = osp.join(DATA_ROOT, "data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer")
QUERY_PATH = osp.join(
    os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal/queries/fig6_pigment_transfer.cql"
)

# The nine graphs and the protein embedding that separate the 0.43 runs from the 0.13 ones.
# Copied from experiments/019-simb-multimodal/conf/gh_metabolism_000.yaml rather than
# retyped, so the backbone is the one that was actually measured.
STRONG_GRAPHS: list[str] = [
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
]
STRONG_NODE_EMBEDDINGS: tuple[str, ...] = ("prot_T5_all",)

# The model infers its input width by indexing the embedding dataset, so it needs the same
# objects the dataset was opened with, not just their names. Built once and held here
# because `build_dataset` owns the genome and graph they are constructed from.
_NODE_EMBEDDINGS: dict[str, Any] = {}


def build_dataset() -> "Neo4jCellDataset":
    """Open the pigment dataset with the nine-graph, prot_T5 backbone. Never rebuilds."""
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        # NOT the default. `overwrite` defaults to True, which calls
        # `gffutils.create_db(force=True)` and so UNLINKS and rebuilds the 15 MB
        # `data/sgd/genome/data.db` on every construction. With 8 workers per node all
        # building the genome at once, seven of them open the file inside another's rebuild
        # window and die on `FileNotFoundError: .../data.db` -- which reads as a missing
        # file and is in fact a write race. This killed 7 of 8 workers on Delta node
        # gpub071 (21813442_0) and every worker of three tasks in array 21797666. The
        # database is a build artifact and is never edited, so nothing here should rebuild
        # it; the older experiments all pass overwrite=False for the same reason.
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    _NODE_EMBEDDINGS.update(
        NodeEmbeddingBuilder.build(
            embedding_names=list(STRONG_NODE_EMBEDDINGS),
            data_root=DATA_ROOT,
            genome=genome,
            graph=graph,
        )
    )
    with open(QUERY_PATH) as handle:
        query = handle.read()
    return Neo4jCellDataset(
        root=DATASET_ROOT,
        query=query,
        gene_set=genome.gene_set,
        graphs=build_gene_multigraph(graph=graph, graph_names=STRONG_GRAPHS),
        incidence_graphs=None,
        node_embeddings=_NODE_EMBEDDINGS,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=DeletionKeyedGenotypeAggregator,
        graph_processor=Perturbation(),
    )


class ScoringConfig(BaseModel):
    """The scoring rule, fixed before the first launch."""

    model_config = ConfigDict(extra="forbid")

    primary: str
    select_metric: str
    select_smoothing_window: int
    aggregate: str


class BaselineConfig(BaseModel):
    """The published numbers the decision is made against, read from 020's results JSON."""

    model_config = ConfigDict(extra="forbid")

    fcl_rf_test_spearman: float
    fcl_rf_test_auc_high: float
    cgt_020_v4_test_spearman_mean: float
    cgt_020_v4_test_spearman_sd: float


class DataConfig(BaseModel):
    """Dataset location, the external test split to pin, and the partition seed."""

    model_config = ConfigDict(extra="forbid")

    dataset_subdir: str
    query_path: str
    pinned_test_split_file: str
    split_seed: int


class ModelConfig(BaseModel):
    """The 026 incumbent hyperparameters, identical across all six arms."""

    model_config = ConfigDict(extra="forbid")

    hidden: int
    layers: int
    attention_heads: int
    batch_size: int
    lr: float
    weight_decay: float


class TrainConfig(BaseModel):
    """Epoch ceiling and the two knobs the launcher may override."""

    model_config = ConfigDict(extra="forbid")

    epochs: int
    early_stopping: bool
    num_workers: int


class FluxConfig(BaseModel):
    """The kcat-coverage floor that stops a flux arm running on default parameters."""

    model_config = ConfigDict(extra="forbid")

    min_kcat_experimental_fraction: float


class BxConfig(BaseModel):
    """``conf/base.yaml``, validated.

    ``extra="forbid"`` throughout, so a typo in a YAML key is an error at load rather than a
    silently ignored hyperparameter -- the failure mode that makes two arms differ by
    something nobody chose.
    """

    model_config = ConfigDict(extra="forbid")

    scoring: ScoringConfig
    baselines: BaselineConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    flux: FluxConfig


class ArmConfig(BaseModel):
    """``conf/arm_<name>.yaml``: the only two things an arm is allowed to change."""

    model_config = ConfigDict(extra="forbid")

    base_arm: str
    permute_train_targets: bool


class BxCell(BaseModel):
    """One training run, fully specified. Serialized into the results file verbatim."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    base_arm: str
    seed: int
    permute_train_targets: bool
    epochs: int
    hidden: int
    layers: int
    attention_heads: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    split_seed: int
    select_smoothing_window: int = Field(ge=1)
    # RECIPE SWITCHES, added 2026-09-09 after the collapse audit. Both default to what the
    # 144-cell grid ran, so a grid cell deserializes unchanged; both are ON in the 019 run
    # that reached 0.434 (job 1344), and the grid had dropped them without recording it.
    crps: bool = Field(
        default=False,
        description=(
            "Score the scalar head as a Gaussian CRPS with a (mu, log_sigma) output. The "
            "grid's loop sliced log_sigma away before the loss and fell back to MSE."
        ),
    )
    graph_reg: bool = Field(
        default=False,
        description=(
            "Pass 019's graph-structure regularizer (lambda 6.5e-5, one attention head per "
            "graph) to the model. The grid never passed it, so graph_reg_loss was 0."
        ),
    )
    abort_on_collapse: bool = Field(
        default=False,
        description=(
            "Stop the run when the validation prediction spread stays under 1 percent of "
            "the target spread past epoch 10, which is the collapse the grid ran 200 epochs "
            "of without noticing."
        ),
    )
    aux_mulleder: bool = Field(
        default=False,
        description=(
            "Train the 19-column Mulleder amino-acid head jointly, as the 019 run that "
            "reached 0.434 did (active_heads: [betaxanthin, mulleder19]). The 027 protocol "
            "dropped it to make the comparison betaxanthin-only, and the four recipe "
            "diagnostics of 2026-09-09 collapsed with and without CRPS and the graph "
            "regularizer, which leaves the co-head as the remaining recipe difference. "
            "Selection and scoring stay betaxanthin-only."
        ),
    )


    attention_mask: bool = Field(
        default=False,
        description=(
            "Hard graph masking: burn each of the nine relations into one attention head "
            "at layer 1, as the 020 Delta grid (delta_grid_betaxanthin.yaml) did. The 019 "
            "KL regularizer was replaced by this after the July runs."
        ),
    )
    learnable_embedding: bool = Field(
        default=False,
        description=(
            "MEMORIZATION CONTROL. A free per-gene vector lets the model fit the training "
            "targets by lookup (019 measured train r 0.86 with it), so a run that cannot "
            "fit even with it has a dead signal path, not weak features."
        ),
    )
    pert_heads: int = Field(
        default=6,
        description=(
            "Perturbation-head attention heads. 6 is what 019's gh_metabolism_000 and the "
            "027 grid used; 9 is what the 020 grid that trained used. Must divide hidden."
        ),
    )
    # WAVE 5 (2026-09-10). Three knobs on the module itself, all defaulting to what waves 1
    # to 4 ran, so an earlier result file deserializes unchanged.
    kcat_tier: str = Field(
        default="measured",
        description=(
            "'measured' resolves kcat from the Open Enzyme Database and defaults the rest "
            "(148 of 3,728 catalytic units measured, 3,580 default). 'predicted' inserts the "
            "sequence-predictor tier between them (148 measured, 3,410 predicted, 170 "
            "default) using the predictors the within-class study admits for yeast."
        ),
    )
    thermo_source: str = Field(
        default="shipped",
        description=(
            "Which route prices the standard reaction energies: 'shipped' (the GEM's own "
            "metabolite table summed over S, 3,505 of 4,131 reactions), 'equilibrator' "
            "(2,741) or 'union' (3,722)."
        ),
    )
    constraint_warmup_epochs: int = Field(
        default=0,
        description=(
            "Ramp the flux layer's constraint weights linearly from 0 over this many epochs."
        ),
    )
    constraint_scale: float = Field(
        default=1.0,
        description="Final multiplier on the whole weighted constraint sum after warmup.",
    )
    constraint_delay_epochs: int = Field(
        default=0,
        description=(
            "Hold the constraint scale at exactly zero for this many epochs before the ramp "
            "begins, so the module is applied to a head that has already left the plateau."
        ),
    )
    head_normalize: bool = Field(
        default=False,
        description=(
            "Divide the flux head's two inputs by the layer's working flux magnitude before "
            "reading them. MEASURED 2026-09-10 (diagnose_head_scale.py, results/"
            "head_scale_diagnosis.json): unnormalized, the flux head starts at output mean "
            "+7.030 against a target standardized to mean 0 sd 1, while the pooled head "
            "starts at -0.573. The head spends its first epochs unlearning a seven-sigma "
            "offset. Off by default so every wave-1 to wave-5 run deserializes unchanged."
        ),
    )


AUX_HEAD = "mulleder19"
AUX_WIDTH = 19

#: The 020 grid's hard mask: relation -> attention head at layer 1, in edge-type names.
ATTENTION_MASK_CONFIG: dict[str, Any] = {
    "enabled": True,
    "layers": [1],
    "head_graphs": {
        0: "physical_interaction",
        1: "regulatory_interaction",
        2: "tflink",
        3: "string12_0_neighborhood",
        4: "string12_0_fusion",
        5: "string12_0_cooccurence",
        6: "string12_0_coexpression",
        7: "string12_0_experimental",
        8: "string12_0_database",
    },
}


#: The 019 graph-structure prior, copied from gh_metabolism_000.yaml: one regularized
#: attention head per graph at layer 1, lambda 6.5e-5, a quarter of the rows sampled.
GRAPH_REG_LAMBDA = 6.5e-5
GRAPH_REG_CONFIG: dict[str, Any] = {
    "graph_reg_lambda": GRAPH_REG_LAMBDA,
    "graph_reg_layer": [1],
    "row_sampling_rate": 0.25,
    "regularized_heads": {
        name: {"layer": [1], "head": i, "lambda": GRAPH_REG_LAMBDA}
        for i, name in enumerate(STRONG_GRAPHS)
    },
}
#: Collapse detector threshold: prediction spread over target spread on the validation
#: split. The grid's cells sat at 8e-3 and below. The patience is NOT ten epochs: on the
#: current code the mean plateau is the normal start and lasts 117 to 157 epochs before the
#: spread appears (020 Delta grid, W&B history of runs w7fuxsvr, azh3tstz, 74sde13z,
#: dhmloet1), while the July 26 code left it at epoch 23 (run gilahyper-1346). A ten-epoch
#: abort, which the 2026-09-09 diagnostics ran with, kills every run that would have
#: escaped; the 027 grid's 200 epochs on Delta were also inside that window for most seeds.
COLLAPSE_SD_RATIO = 0.01
COLLAPSE_PATIENCE_EPOCHS = int(os.getenv("COLLAPSE_PATIENCE_EPOCHS", "200"))


def _head_params(pred: torch.Tensor) -> torch.Tensor:
    """``[B, param_dim]`` from a scalar head, whichever shape it emitted.

    A distributional scalar head emits ``[B, 1, param_dim]`` and a point head ``[B, 1]``.
    The grid's loop did ``pred[..., 0]`` on the former, which keeps mu and throws away
    log_sigma, so the CRPS branch of the loss was unreachable. Squeezing the feature axis
    keeps both parameters.
    """
    return pred.squeeze(1) if pred.dim() == 3 else pred


def arm_names() -> list[str]:
    """Every arm this experiment defines, in a FIXED order.

    The order is load-bearing: the shard rule assigns worker ``w`` the setting at index
    ``w % n_settings``, so a reordering silently reassigns work between workers mid-experiment.
    Sorted, so it cannot drift with filesystem order.
    """
    return sorted(
        f[len("arm_") : -len(".yaml")]
        for f in os.listdir(CONF_DIR)
        if f.startswith("arm_") and f.endswith(".yaml")
    )


def load_config() -> BxConfig:
    with open(osp.join(CONF_DIR, "base.yaml")) as fh:
        return BxConfig(**yaml.safe_load(fh))


def build_cell(
    arm: str,
    seed: int,
    cfg: BxConfig,
    epochs: int | None = None,
    crps: bool = False,
    graph_reg: bool = False,
    abort_on_collapse: bool = False,
    aux_mulleder: bool = False,
    attention_mask: bool = False,
    pert_heads: int = 6,
    layers: int | None = None,
    learnable_embedding: bool = False,
    kcat_tier: str = "measured",
    thermo_source: str = "shipped",
    constraint_warmup_epochs: int = 0,
    constraint_scale: float = 1.0,
    constraint_delay_epochs: int = 0,
    head_normalize: bool = False,
) -> BxCell:
    """Merge ``base.yaml`` with one arm file into a fully specified cell."""
    with open(osp.join(CONF_DIR, f"arm_{arm}.yaml")) as fh:
        armcfg = ArmConfig(**yaml.safe_load(fh))
    assert armcfg.base_arm in ARMS, (
        f"arm {arm!r} names base_arm={armcfg.base_arm!r}, which is not in 026's ARMS "
        f"registry ({sorted(ARMS)}). The registry is the single source of truth for what "
        "the flux layer does; this file never redefines it."
    )
    return BxCell(
        arm=arm,
        base_arm=armcfg.base_arm,
        seed=seed,
        permute_train_targets=armcfg.permute_train_targets,
        epochs=epochs if epochs is not None else cfg.train.epochs,
        hidden=cfg.model.hidden,
        layers=layers if layers is not None else cfg.model.layers,
        attention_heads=cfg.model.attention_heads,
        batch_size=cfg.model.batch_size,
        num_workers=int(os.getenv("NUM_WORKERS", str(cfg.train.num_workers))),
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        split_seed=cfg.data.split_seed,
        select_smoothing_window=cfg.scoring.select_smoothing_window,
        crps=crps,
        graph_reg=graph_reg,
        abort_on_collapse=abort_on_collapse,
        aux_mulleder=aux_mulleder,
        attention_mask=attention_mask,
        pert_heads=pert_heads,
        learnable_embedding=learnable_embedding,
        kcat_tier=kcat_tier,
        thermo_source=thermo_source,
        constraint_warmup_epochs=constraint_warmup_epochs,
        constraint_scale=constraint_scale,
        constraint_delay_epochs=constraint_delay_epochs,
        head_normalize=head_normalize,
    )


def resolve_pinned_test(dataset: Any, cfg: BxConfig) -> tuple[set[int], dict[str, Any]]:
    """Merzbacher 2025's 639 test ORFs -> record indices in our build.

    Lifted from ``train_cgt_multitask._pinned_test_indices`` including its two guards, both
    of which were found the hard way there:

    * the mapping goes through ``is_any_deletion_gene_index``, NOT
      ``is_any_perturbed_gene_index``. The pigment cassettes are emitted as perturbations on
      every strain and three of their members are native ORFs (ARO4/YBR249C and ARO7/YPR060C
      in Cachera's betaxanthin cassette), two of which are in Merzbacher's test list. Under
      the perturbed-gene index the pin resolved 639 genes to 4,885 of 4,930 records and left
      28 training examples -- and trained anyway.
    * a requested gene absent from the built LMDB is COUNTED AND REPORTED. The Cachera build
      is stale w.r.t. the shared name resolver (issue #195) and the split file names the 10
      it loses; a comparison that quietly shrank its own test set would be reporting on a
      different gene set than it claims.
    """
    path = osp.join(EXPERIMENT_ROOT, cfg.data.pinned_test_split_file)
    with open(path) as fh:
        payload = json.load(fh)
    genes = payload["split"]["test"]
    gene_index = dataset.is_any_deletion_gene_index
    indices: set[int] = set()
    found: list[str] = []
    absent: list[str] = []
    for gene in genes:
        hits = gene_index.get(gene)
        if hits:
            found.append(gene)
            indices.update(int(h) for h in hits)
        else:
            absent.append(gene)
    frac = len(indices) / len(dataset)
    assert frac < 0.5, (
        f"pinned test set is {frac:.1%} of the dataset ({len(indices)}/{len(dataset)}). "
        "639 genes should select a small minority of records; this means the gene index is "
        "resolving cassette membership rather than deletions."
    )
    report = {
        "path": path,
        "requested": len(genes),
        "found": len(found),
        "absent": absent,
        "records": len(indices),
        "fraction_of_dataset": frac,
    }
    print(
        f"[pinned-split] {len(found)}/{len(genes)} genes present -> {len(indices)} records "
        f"pinned to TEST ({frac:.1%} of {len(dataset)}); {len(absent)} absent (issue #195)",
        flush=True,
    )
    return indices, report


def _params_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU copy of the TRAINABLE parameters only.

    Not ``state_dict()``: the flux layer registers the stoichiometric matrix and the
    thermodynamic tables as buffers, and the selection ring holds five snapshots, so copying
    buffers would carry ~46 MB of constant S matrix five times per worker for nothing.
    """
    return {k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters()}


def _restore(model: nn.Module, snap: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(snap[name].to(param.device))


def _select_epoch(val_curve: list[float], window: int) -> tuple[int, str]:
    """Epoch chosen by argmax of a CENTERED moving average of validation Pearson.

    Returns ``(epoch, rule)``. The rule string is written into the result so a short run --
    a canary, or a cell the wall clock truncated -- is never silently scored by a different
    rule than the one the design fixed.
    """
    finite = [v if np.isfinite(v) else -np.inf for v in val_curve]
    if len(finite) < window:
        return int(np.argmax(finite)), "raw_argmax_short_run"
    half = window // 2
    smoothed = [
        float(np.mean(finite[i - half : i + half + 1]))
        for i in range(half, len(finite) - half)
    ]
    return int(np.argmax(smoothed)) + half, f"centered_ma_{window}"


def _masked_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson on a 1-D pair, with the RELATIVE variance guard 026 arrived at.

    An absolute floor is the wrong test: a head collapsed onto a constant emits a residual
    spread of order 1e-11, which clears any absolute floor and then makes ``corrcoef`` divide
    by a number indistinguishable from zero, yielding NaN -- reported as "metric unavailable"
    when what happened is the documented mean-collapse failure. Scaling by the target's spread
    makes "collapsed" scale-free, and a collapsed head scores 0.0, which is what it earned.
    """
    finite = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[finite], true[finite]
    if p.size < 10 or t.std() < 1e-12 or p.std() < 1e-8 * t.std():
        return 0.0
    r = float(pearsonr(p, t)[0])
    return r if np.isfinite(r) else 0.0


def _masked_spearman(pred: np.ndarray, true: np.ndarray) -> float:
    finite = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[finite], true[finite]
    if p.size < 10 or t.std() < 1e-12 or p.std() < 1e-8 * t.std():
        return 0.0
    rho = float(spearmanr(p, t)[0])
    return rho if np.isfinite(rho) else 0.0


def _top_k_enrichment(
    pred: np.ndarray, true: np.ndarray, k: int, top_frac: float = 0.1
) -> dict[str, float]:
    """Of the k genes the model ranks highest, how many are truly in the top ``top_frac``.

    The imbalance-immune half of the 020 recipe, and the metric the task is actually about:
    accuracy on a 67%-majority problem actively selects against finding high producers.
    ``top_frac`` defines "high" on OUR measured values so the metric needs no external label
    file; the 3-class head-to-head against Merzbacher's released labels is a separate step,
    run by 020's own scripts on the per-gene dump this file writes.
    """
    finite = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[finite], true[finite]
    if p.size < k:
        return {"hits": float("nan"), "enrichment": float("nan")}
    n_high = max(1, int(round(top_frac * t.size)))
    high = set(np.argsort(-t)[:n_high].tolist())
    top = np.argsort(-p)[:k].tolist()
    hits = sum(1 for i in top if i in high)
    base = n_high / t.size
    return {
        "hits": float(hits),
        "enrichment": float((hits / k) / base) if base else float("nan"),
    }


def run_cell(
    cell: BxCell,
    cfg: BxConfig,
    dataset: Any,
    pinned_test: set[int],
    pinned_report: dict[str, Any],
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
    dump_predictions: bool = True,
) -> dict[str, Any]:
    """Train one cell, select an epoch on validation, score it on the pinned test genes."""
    torch.manual_seed(cell.seed)
    np.random.seed(cell.seed)
    # 026's registry stays the single source of truth for WHAT the module does; the cell adds
    # only the parameter-source and schedule knobs on top of it. A copy, so a second cell in
    # the same process cannot inherit the first one's overrides.
    arm = {
        **ARMS[cell.base_arm],
        "kcat_tier": cell.kcat_tier,
        "thermo_source": cell.thermo_source,
        "constraint_warmup_epochs": cell.constraint_warmup_epochs,
        "constraint_final_scale": cell.constraint_scale,
        "constraint_delay_epochs": cell.constraint_delay_epochs,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gene_ids = list(dataset.cell_graph["gene"].node_ids)
    flux_layer, flux_meta = build_flux_layer(arm, gene_ids, cell.hidden)

    if arm["flux"]:
        frac = float(flux_meta["kcat_experimental_fraction"])
        # NO FALLBACK. `resolve_kcat_table` catches a missing Open Enzyme Database mirror and
        # returns an empty record list, which replaces every measured kcat with the organism
        # default -- an "enzyme-constrained" arm with no enzyme constraints, running to
        # completion and reporting a plausible number. The mirror is 512 KB and is NOT on
        # Delta as of 2026-09-03; see README.md, "Prerequisites".
        assert frac >= cfg.flux.min_kcat_experimental_fraction, (
            f"kcat experimental fraction is {frac:.4f}, below the floor "
            f"{cfg.flux.min_kcat_experimental_fraction}. The Open Enzyme Database mirror at "
            f"$DATA_ROOT/data/enzyme_kinetics/open_enzyme_database/scerevisiae is missing or "
            "empty, so every kcat has silently become the organism default."
        )

    head_specs: dict[str, dict[str, Any]] = {
        # Betaxanthin and the Mulleder metabolome both carry the label `metabolite_level`;
        # WIDTH is what separates them. Declaring only the width-1 spec supervises betaxanthin
        # alone, and a Mulleder-only genotype simply matches nothing and arrives masked False.
        HEAD: {"label": "metabolite_level", "width": 1}
    }
    param_dim = 2 if cell.crps else 1
    if arm["flux"]:
        met_index = flux_meta["met_index"]
        heads_config = {
            HEAD: {
                "kind": "flux_scalar",
                "output_dim": 1,
                "param_dim": param_dim,
                "precursor_indices": [
                    met_index[m] for m in BETAXANTHIN_PRECURSORS if m in met_index
                ],
                "normalize_inputs": cell.head_normalize,
            }
        }
    else:
        heads_config = {HEAD: {"kind": "scalar", "output_dim": 1, "param_dim": param_dim}}
    if cell.aux_mulleder:
        # The co-head reads the pooled representation whatever the arm, so the contrast
        # between arms stays on the betaxanthin readout alone.
        head_specs[AUX_HEAD] = {"label": "metabolite_level", "width": AUX_WIDTH}
        heads_config[AUX_HEAD] = {"kind": "vector", "output_dim": AUX_WIDTH}

    graph_reg_kwargs: dict[str, Any] = (
        {
            "graph_regularization_config": GRAPH_REG_CONFIG,
            "graph_reg_lambda": GRAPH_REG_LAMBDA,
        }
        if cell.graph_reg
        else {}
    )
    model = CellGraphTransformerMetabolism(
        cell_graph=dataset.cell_graph,
        gene_num=len(gene_ids),
        hidden_channels=cell.hidden,
        num_transformer_layers=cell.layers,
        num_attention_heads=cell.attention_heads,
        dropout=0.1,
        heads_config=heads_config,
        # OFF. With it on, the model learns a free per-gene vector and stops needing the
        # protein embedding, which 020 measured at 0.279 (off) against 0.151 (on) and
        # 019 recorded as making generalization impossible by construction.
        learnable_embedding_config={
            "enabled": cell.learnable_embedding,
            "size": cell.hidden,
            "preprocessor": {"num_layers": 2, "dropout": 0.1},
        },
        # 6, from 019's conf, or 9 as the 020 grid ran. The default is 8, and 90 is not
        # divisible by 8, so the perturbation head's cross-attention refuses to build.
        perturbation_head_config={"num_heads": cell.pert_heads, "dropout": 0.1},
        attention_mask_config=ATTENTION_MASK_CONFIG if cell.attention_mask else None,
        # The model reads cell_graph["gene"].x only when it is TOLD it has precomputed
        # embeddings; without this it looks for a learnable table that is now disabled and
        # raises "No gene embeddings available".
        node_embeddings=_NODE_EMBEDDINGS,
        flux_layer=flux_layer,
        **graph_reg_kwargs,
    ).to(device)

    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, cfg.data.dataset_subdir, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cell.batch_size,
        # THE PINNED PARTITION. `random_seed` selects train/val/test, so it is held at
        # `split_seed` for every arm and every seed; `cell.seed` reached torch above and
        # varies weight INITIALIZATION only.
        random_seed=cell.split_seed,
        num_workers=cell.num_workers,
        pin_memory=True,
        prefetch=False,
        follow_batch=["perturbation_indices", "phenotype_values"],
        pinned_test_indices=pinned_test,
    )
    dm.setup()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cell.lr, weight_decay=cell.weight_decay
    )
    # CRPS, not MSE. The head emits (mu, log_sigma) and is scored as a distribution;
    # 019 measured 0.413 -> 0.434 from this single change, and 020 measured crps 0.275
    # against point 0.172 across its sweep.
    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape[-1] == 2:
            mu, log_sigma = prediction[..., 0], prediction[..., 1]
            # `gaussian_crps` takes sigma, not log sigma; the clamp keeps exp finite.
            sigma = torch.exp(log_sigma.clamp(-6.0, 3.0))
            return gaussian_crps(mu, sigma, target.squeeze(-1)).unsqueeze(-1)
        return nn.functional.mse_loss(prediction, target, reduction="none")

    # Target standardization from the TRAIN split only. A metric computed against
    # test-informed statistics is not a held-out metric.
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in head_specs}
    for batch in dm.train_dataloader():
        for name, (t, m) in extract_targets(batch, head_specs).items():
            chunks[name].append(torch.where(m, t, torch.nan))
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, parts in chunks.items():
        cat = torch.cat(parts, dim=0)
        mu = torch.nanmean(cat, dim=0)
        centered = cat - mu
        s = torch.nanmean(centered * centered, dim=0).sqrt().clamp(min=1e-6)
        stats[name] = (mu.to(device), s.to(device))
    mean, sd = stats[HEAD]

    window = cell.select_smoothing_window
    half = window // 2
    #: ``(epoch, trainable-parameter snapshot)``, `window` deep. Keyed by epoch rather than
    #: indexed by position: a run whose validation curve is entirely non-finite selects epoch
    #: 0, which position arithmetic would resolve to a snapshot the ring no longer holds.
    ring: deque[tuple[int, dict[str, torch.Tensor]]] = deque(maxlen=window)
    val_curve: list[float] = []
    history: list[dict[str, Any]] = []
    best_smoothed = -np.inf
    best_snap: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    t0 = time.time()

    for epoch in range(cell.epochs):
        model.train()
        train_loss, train_data_loss, n_batches = 0.0, 0.0, 0
        # THE CONSTRAINT RAMP, set once per epoch before any batch. Epoch 0 carries no
        # physics at all when a warmup is configured, so the data term is the only thing
        # moving the head while it is on the mean plateau.
        constraint_scale = 1.0
        if model.flux_layer is not None:
            constraint_scale = model.flux_layer.set_constraint_epoch(epoch)
        train_constraint_loss = 0.0
        feas: dict[str, float] = {}
        for batch in dm.train_dataloader():
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, reps = model(dataset.cell_graph.to(device), batch)
            t, m = extract_targets(batch, head_specs)[HEAD]
            t, m = t.to(device), m.to(device)
            if cell.permute_train_targets:
                # THE NULL: target and mask permuted TOGETHER, so the genotype-to-phenotype
                # association is destroyed while the target distribution and the missingness
                # rate are untouched. Validation and test are never permuted.
                perm = torch.randperm(t.shape[0], device=device)
                t, m = t[perm], m[perm]
            z = (t - mean) / sd
            pred = _head_params(reps["head_outputs"][HEAD])
            data_loss = (loss_fn(pred, z) * m).sum() / m.sum().clamp(min=1)
            loss = data_loss + reps["graph_reg_loss"]
            if cell.aux_mulleder:
                # Masked per-feature MSE on the standardized amino-acid vector, weight 1,
                # as 019's head_weights had it. Never permuted: the null is on betaxanthin.
                t_aux, m_aux = extract_targets(batch, head_specs)[AUX_HEAD]
                t_aux, m_aux = t_aux.to(device), m_aux.to(device)
                mu_aux, sd_aux = stats[AUX_HEAD]
                z_aux = (t_aux - mu_aux) / sd_aux
                p_aux = reps["head_outputs"][AUX_HEAD]
                if p_aux.dim() == 3:
                    p_aux = p_aux[..., 0]
                aux_loss = (
                    nn.functional.mse_loss(p_aux, z_aux, reduction="none") * m_aux
                ).sum() / m_aux.sum().clamp(min=1)
                loss = loss + aux_loss
                feas["aux_loss"] = feas.get("aux_loss", 0.0) + float(aux_loss.detach())
            train_data_loss += float(data_loss.detach())
            if "flux" in reps:
                # `reps` carries "flux" only when the model was built WITH a layer, so this
                # cannot be None here. Asserted rather than type-ignored: if it ever were
                # None the constraint terms would silently vanish and the arm would still
                # report a plausible number.
                assert model.flux_layer is not None
                constraint_term = model.flux_layer.constraint_loss(reps["flux"])
                loss = loss + constraint_term
                train_constraint_loss += float(constraint_term.detach())
                for k, v in reps["flux"].items():
                    if v.dim() == 0 and k.startswith(
                        ("c_", "feas_", "g_diss", "protein_used")
                    ):
                        feas[k] = feas.get(k, 0.0) + float(v.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += float(loss.detach())
            n_batches += 1

        preds, trues = _predict(
            model, dm.val_dataloader(), dataset, head_specs, mean, sd
        )
        val_r = _masked_pearson(preds, trues)
        val_curve.append(val_r)
        ring.append((epoch, _params_snapshot(model)))
        # THE COLLAPSE DIAGNOSTIC. Pearson and Spearman are scale-free, so a head that
        # emits a constant plus jitter still returns a number; the spread ratio is what
        # says whether there is a prediction to correlate at all.
        observed = np.isfinite(trues)
        pred_sd_ratio = float(
            np.std(preds[observed]) / max(np.std(trues[observed]), 1e-12)
        )
        if (
            cell.abort_on_collapse
            and epoch >= COLLAPSE_PATIENCE_EPOCHS
            and pred_sd_ratio < COLLAPSE_SD_RATIO
        ):
            raise SystemExit(
                f"[{cell.arm} s{cell.seed}] COLLAPSED: validation prediction spread is "
                f"{pred_sd_ratio:.2e} of the target spread at epoch {epoch}. Stopping "
                "rather than spending the budget on a constant predictor."
            )

        # The centered window can only score epoch `t - half` once epoch `t` exists, which is
        # exactly why the ring is `window` deep: the snapshot for that epoch is still in it.
        if len(val_curve) >= window:
            centered_epoch = epoch - half
            score = float(np.mean(val_curve[-window:]))
            if score > best_smoothed:
                best_smoothed = score
                best_snap = ring[-(half + 1)][1]
                best_epoch = centered_epoch

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_batches, 1),
            # The data term alone, which is the one comparable with `val_loss`.
            "train_data_loss": train_data_loss / max(n_batches, 1),
            "val_loss": _masked_loss(
                model, dm.val_dataloader(), dataset, head_specs, mean, sd, loss_fn
            ),
            "val_pearson": val_r,
            "val_pred_sd_ratio": pred_sd_ratio,
            "n_val": int(np.isfinite(trues).sum()),
            "graph_reg_loss": float(reps["graph_reg_loss"]),
            # The two numbers the warmup experiment is read on, logged BESIDE the spread
            # ratio so the epoch at which the physics binds and the epoch at which the head
            # leaves the plateau are on the same axis.
            "constraint_scale": constraint_scale,
            "train_constraint_loss": train_constraint_loss / max(n_batches, 1),
        }
        for k, v in feas.items():
            row[k] = v / max(n_batches, 1)
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
        print(
            f"[{cell.arm} s{cell.seed}] ep {epoch:3d} loss {row['train_loss']:.4f} "
            f"data {row['train_data_loss']:.4f} cstr {row['train_constraint_loss']:.4f} "
            f"scale {constraint_scale:.3f} "
            f"val_loss {row['val_loss']:.4f} val_r {val_r:+.4f} "
            f"sd_ratio {pred_sd_ratio:.4f} best_ep {best_epoch}",
            flush=True,
        )

    sel_epoch, sel_rule = _select_epoch(val_curve, window)
    snap_by_epoch = dict(ring)
    if best_snap is not None and sel_rule.startswith("centered_ma"):
        # The ring selects online (it must -- a centered window needs a snapshot that is
        # about to fall out of memory) and `_select_epoch` recomputes offline from the saved
        # curve. They are two implementations of ONE rule, so they must agree, and a silent
        # divergence would mean the reported epoch is not the epoch that was scored.
        assert sel_epoch == best_epoch, (
            f"selection disagreement: online ring picked {best_epoch}, offline recompute "
            f"picked {sel_epoch}. The ring and `_select_epoch` must implement one rule."
        )
        _restore(model, best_snap)
    else:
        # A run shorter than the smoothing window (a canary, or a wall-clock truncation)
        # falls to the raw argmax, and `select_rule` in the result says so.
        assert sel_epoch in snap_by_epoch, (
            f"selected epoch {sel_epoch} has no snapshot; the ring holds "
            f"{sorted(snap_by_epoch)}. This means every validation epoch was non-finite."
        )
        _restore(model, snap_by_epoch[sel_epoch])

    test = _score_test(model, dm, dataset, head_specs, mean, sd, cell, dump_predictions)
    return {
        "cell": cell.model_dump(),
        "select_epoch": sel_epoch,
        "select_rule": sel_rule,
        "select_val_pearson": val_curve[sel_epoch] if val_curve else float("nan"),
        "test": test,
        "history": history,
        "flux_meta": {k: v for k, v in flux_meta.items() if k != "met_index"},
        "pinned_split": pinned_report,
        "n_parameters": model.num_parameters,
        "wall_time_s": time.time() - t0,
    }


def _predict(
    model: nn.Module,
    loader: Any,
    dataset: Any,
    head_specs: dict[str, dict[str, Any]],
    mean: torch.Tensor,
    sd: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Denormalized predictions and targets over a loader; unobserved rows become NaN."""
    device = next(model.parameters()).device
    model.eval()
    ps: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            t, m = extract_targets(batch, head_specs)[HEAD]
            # The point prediction is mu, the first parameter, whatever the head emits.
            pred = _head_params(reps["head_outputs"][HEAD])[:, :1]
            ps.append((pred * sd + mean).cpu().numpy().reshape(-1))
            ts.append(torch.where(m, t, torch.nan).cpu().numpy().reshape(-1))
    return np.concatenate(ps), np.concatenate(ts)


def _masked_loss(
    model: nn.Module,
    loader: Any,
    dataset: Any,
    head_specs: dict[str, dict[str, Any]],
    mean: torch.Tensor,
    sd: torch.Tensor,
    loss_fn: Any,
) -> float:
    """The DATA-FIT loss on a held-out loader, in the same units as training.

    Separate from ``_predict`` because that returns the point prediction only: it slices
    ``pred[..., 0]`` and discards ``log_sigma``, so a CRPS cannot be reconstructed from what
    it hands back. The loss is computed on normalized targets, exactly as in training.

    This is the DATA term alone. The training loop's ``train_loss`` also carries the graph
    regularizer and, on a flux arm, the constraint terms, and neither depends on the labels,
    so adding them to a validation number would make the two curves incomparable in a way
    that is invisible on a plot. ``train_data_loss`` is logged beside it for that comparison.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, count = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            t, m = extract_targets(batch, head_specs)[HEAD]
            t, m = t.to(device), m.to(device)
            z = (t - mean) / sd
            pred = _head_params(reps["head_outputs"][HEAD])
            total += float((loss_fn(pred, z) * m).sum())
            count += float(m.sum())
    model.train(was_training)
    return total / max(count, 1.0)


def _score_test(
    model: nn.Module,
    dm: Any,
    dataset: Any,
    head_specs: dict[str, dict[str, Any]],
    mean: torch.Tensor,
    sd: torch.Tensor,
    cell: BxCell,
    dump_predictions: bool,
) -> dict[str, Any]:
    """The prespecified test metrics, on the pinned genes and on the whole test split.

    ``test_dataloader`` is built with ``shuffle=False``, so batches arrive in the order of
    ``test_dataset.indices`` -- that is what lets a running cursor map row -> record -> gene.
    The gene name comes from the DELETION index for the reason spelled out in
    :func:`resolve_pinned_test`.
    """
    order = [int(i) for i in dm.test_dataset.indices]
    preds, trues = _predict(model, dm.test_dataloader(), dataset, head_specs, mean, sd)
    assert len(preds) == len(order), (
        f"test pass covered {len(preds)}/{len(order)} records -- a short count is a silent "
        "truncation of the comparison set."
    )

    idx_to_genes: dict[int, list[str]] = {}
    for gene, rows in dataset.is_any_deletion_gene_index.items():
        for row in rows:
            idx_to_genes.setdefault(int(row), []).append(str(gene))

    pin_path = osp.join(
        EXPERIMENT_ROOT, "020-cachera-betaxanthin/results/merzbacher_nested_split.json"
    )
    with open(pin_path) as fh:
        pinned_genes = set(json.load(fh)["split"]["test"])

    records = []
    is_pinned = np.zeros(len(order), dtype=bool)
    for i, rec in enumerate(order):
        genes = idx_to_genes.get(rec, [])
        is_pinned[i] = any(g in pinned_genes for g in genes)
        records.append(
            {
                "record_index": rec,
                "genes": genes,
                "pred": float(preds[i]),
                "target": None if not np.isfinite(trues[i]) else float(trues[i]),
                "pinned": bool(is_pinned[i]),
            }
        )

    def block(sel: np.ndarray) -> dict[str, Any]:
        p, t = preds[sel], trues[sel]
        obs = np.isfinite(t)
        return {
            "n": int(obs.sum()),
            "spearman": _masked_spearman(p, t),
            "pearson": _masked_pearson(p, t),
            **{f"top{k}": _top_k_enrichment(p[obs], t[obs], k) for k in (10, 25, 50)},
        }

    out: dict[str, Any] = {
        "pinned": block(is_pinned),
        "all": block(np.ones(len(order), dtype=bool)),
    }

    if dump_predictions:
        os.makedirs(TEST_PRED_DIR, exist_ok=True)
        path = osp.join(TEST_PRED_DIR, f"{cell.arm}_s{cell.seed}.json")
        with open(path, "w") as fh:
            json.dump(
                {
                    "experiment": "027-betaxanthin-metabolic",
                    "cell": cell.model_dump(),
                    "head": HEAD,
                    "n_test_records": len(order),
                    "metrics": out,
                    "predictions": records,
                },
                fh,
            )
        out["dump_path"] = path
        print(f"[test-dump] wrote {path} ({len(records)} records)", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=arm_names())
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Skip the per-gene dump. For a local build/shape check, never for a result.",
    )
    parser.add_argument("--crps", action="store_true", help="Gaussian CRPS on the head.")
    parser.add_argument(
        "--graph-reg", action="store_true", help="019's graph-structure regularizer."
    )
    parser.add_argument(
        "--abort-on-collapse", action="store_true",
        help="Stop once the validation prediction spread stays under 1%% of the target's.",
    )
    parser.add_argument(
        "--aux-mulleder", action="store_true",
        help="Train the 19-amino-acid Mulleder head jointly, as the 019 recipe did.",
    )
    parser.add_argument(
        "--attention-mask", action="store_true",
        help="The 020 grid's hard graph masking, one relation per attention head.",
    )
    parser.add_argument("--pert-heads", type=int, default=6)
    parser.add_argument("--learnable-embedding", action="store_true",
                        help="Memorization control: a free per-gene vector.")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument(
        "--kcat-tier", choices=("measured", "predicted"), default="measured",
        help="Insert the sequence-predictor kcat/K_M tier between measured and default.",
    )
    parser.add_argument(
        "--thermo-source", choices=("shipped", "equilibrator", "union"), default="shipped",
        help="Which route prices the standard reaction energies.",
    )
    parser.add_argument(
        "--constraint-warmup-epochs", type=int, default=0,
        help="Ramp the flux constraint weights linearly from 0 over this many epochs.",
    )
    parser.add_argument(
        "--constraint-scale", type=float, default=1.0,
        help="Final multiplier on the weighted constraint sum after warmup.",
    )
    parser.add_argument(
        "--constraint-delay-epochs", type=int, default=0,
        help="Hold the constraint scale at zero for this many epochs before ramping.",
    )
    parser.add_argument(
        "--head-normalize", action="store_true",
        help="Read the flux head's inputs in units of the layer's working flux magnitude.",
    )
    parser.add_argument(
        "--wandb-group", default=None,
        help="Log this cell to W&B under the given group (project from WANDB_PROJECT).",
    )
    parser.add_argument(
        "--tag", default="", help="Suffix for the result file and the W&B run name."
    )
    args = parser.parse_args()

    cfg = load_config()
    cell = build_cell(
        args.arm, args.seed, cfg, epochs=args.epochs, crps=args.crps,
        graph_reg=args.graph_reg, abort_on_collapse=args.abort_on_collapse,
        aux_mulleder=args.aux_mulleder, attention_mask=args.attention_mask,
        pert_heads=args.pert_heads, layers=args.layers,
        learnable_embedding=args.learnable_embedding,
        kcat_tier=args.kcat_tier, thermo_source=args.thermo_source,
        constraint_warmup_epochs=args.constraint_warmup_epochs,
        constraint_scale=args.constraint_scale,
        constraint_delay_epochs=args.constraint_delay_epochs,
        head_normalize=args.head_normalize,
    )
    dataset = build_dataset()
    print(f"dataset: {len(dataset)} aggregated genotypes", flush=True)
    pinned, report = resolve_pinned_test(dataset, cfg)

    run = None
    if args.wandb_group:
        import wandb

        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "torchcell_027_bxfx"),
            group=args.wandb_group,
            name=f"{cell.arm}{args.tag}-s{cell.seed}",
            tags=["027", "bxfx", cell.arm, f"seed{cell.seed}", args.wandb_group],
            config=cell.model_dump(),
            mode=os.getenv("WANDB_MODE", "online"),
        )
    result = run_cell(
        cell, cfg, dataset, pinned, report, dump_predictions=not args.smoke,
        on_epoch=(lambda row: wandb.log(row, step=int(row["epoch"]))) if run else None,
    )
    if run is not None:
        pin = result["test"]["pinned"]
        run.summary.update(
            {
                "test_spearman_pinned": pin["spearman"],
                "test_pearson_pinned": pin["pearson"],
                "n_test_pinned": pin["n"],
                "select_epoch": result["select_epoch"],
                "select_val_pearson": result["select_val_pearson"],
                "wall_time_s": result["wall_time_s"],
            }
        )
        result["wandb_run_id"] = run.id
        result["wandb_url"] = run.url
        run.finish()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = args.out or f"single_{cell.arm}{args.tag}_s{cell.seed}.json"
    path = osp.join(RESULTS_DIR, out)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(
        f"wrote {path}\n"
        f"  select_epoch {result['select_epoch']} ({result['select_rule']})\n"
        f"  pinned test spearman {result['test']['pinned']['spearman']:+.4f} "
        f"(n={result['test']['pinned']['n']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
