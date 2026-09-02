# experiments/026-metabolism-flux/scripts/train_flux.py
# [[experiments.026-metabolism-flux.scripts.train_flux]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/train_flux.py

r"""Diagnostic runs for the enzyme-constrained thermodynamic flux layer.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/train_flux.py --arms pooled,flux_anchored

Two phenotypes, both single-deletion, both already built into
``fig6_pigment_transfer``:

* **betaxanthin** (Cachera 2023) -- 4,735 strains, a heterologous product whose precursor
  is tyrosine. Noise ceiling ``r = 0.914``.
* **mulleder19** (Mulleder 2016) -- 4,678 strains x 19 intracellular amino acids in mM.
  Every one of the 19 resolves to a cytosolic yeast-GEM metabolite, so the flux head reads
  the model's own turnover at exactly the species that were measured.

WHY A PURPOSE-BUILT LOOP RATHER THAN THE 019 HARNESS
-----------------------------------------------------
The 019 multitask trainer is 3,264 lines and its loss is assembled from head outputs plus
one graph-regularization scalar. The flux layer adds **five** more loss terms and, more
importantly, five per-epoch **feasibility diagnostics** that are the actual object of this
experiment -- a flux vector that fits betaxanthin while violating mass balance has not
learned metabolism. Threading those through the big harness would put the diagnostics in
the position of an afterthought. Here they are first-class and logged every epoch.

THE ARMS
--------
Each arm changes exactly one thing relative to the one above it, so a difference is
attributable:

=================  ==========================================================
arm                what it adds
=================  ==========================================================
``pooled``         baseline: the existing MLP-over-pooled-tokens readout
``flux_off``       flux layer, GPR availability, mass balance. No thermo, no kcat
``flux_free``      + learned potential (loop-freedom, no tabulated energies)
``flux_anchored``  + tabulated delta_f G', enzyme capacity, budget, dissipation
=================  ==========================================================

``pooled`` -> ``flux_off`` asks whether routing the prediction through a stoichiometric
network helps at all. ``flux_free`` -> ``flux_anchored`` asks whether the **measured**
energies carry information beyond the structural loop-freedom that any potential gives.
That second contrast is the one that decides whether the thermodynamic table earns its
place, and it is why the two thermo modes are separate arms rather than one flag.
"""

import argparse
import json
import os
import os.path as osp
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

from torchcell.data import DeletionKeyedGenotypeAggregator, MeanExperimentDeduplicator
from torchcell.data.graph_processor import Perturbation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.datamodules import CellDataModule
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.metabolism.constraints import (
    ThermoMode,
    build_gem_tensors,
    null_space_basis,
)
from torchcell.metabolism.flux_layer import FluxLayer, FluxLayerConfig
from torchcell.metabolism.parameters import molecular_weight_table, resolve_kcat_table
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.models.cell_graph_transformer_metabolism import (
    CellGraphTransformerMetabolism,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
DATASET_ROOT = osp.join(
    DATA_ROOT, "data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer"
)
QUERY_PATH = osp.join(
    EXPERIMENT_ROOT, "019-simb-multimodal/queries/fig6_pigment_transfer.cql"
)
OED_MIRROR = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
)

#: The 19 Mulleder columns in the key-sorted order the Perturbation graph processor emits
#: (``sorted(metabolite_level)``), paired with the cytosolic yeast-GEM species each one is.
#: Resolving these is what the Mulleder loader deferred as ``target_metabolite_ids``.
AMINO_ACID_TO_GEM_METABOLITE = {
    "alanine": "s_0955",
    "arginine": "s_0965",
    "asparagine": "s_0969",
    "aspartate": "s_0973",
    "glutamate": "s_0991",
    "glutamine": "s_0999",
    "glycine": "s_1003",
    "histidine": "s_1006",
    "isoleucine": "s_1016",
    "leucine": "s_1021",
    "lysine": "s_1025",
    "methionine": "s_1029",
    "phenylalanine": "s_1032",
    "proline": "s_1035",
    "serine": "s_1039",
    "threonine": "s_1045",
    "tryptophan": "s_1048",
    "tyrosine": "s_1051",
    "valine": "s_1056",
}

#: Betaxanthin's route is tyrosine -> L-DOPA -> betalamic acid, and the Cachera cassette
#: additionally carries feedback-resistant ARO4 and ARO7, i.e. it deregulates the shikimate
#: pathway upstream of tyrosine. The named precursors are therefore cytosolic tyrosine and
#: the two aromatic-pathway intermediates the cassette's alleles act on.
BETAXANTHIN_PRECURSORS = ["s_1051", "s_0188", "s_1032"]

ARMS: dict[str, dict[str, Any]] = {
    "pooled": {"flux": False},
    "flux_off": {
        "flux": True,
        "thermo_mode": ThermoMode.OFF,
        "use_enzyme_capacity": False,
        "use_protein_budget": False,
        "use_dissipation_limit": False,
    },
    "flux_free": {
        "flux": True,
        "thermo_mode": ThermoMode.FREE,
        "use_enzyme_capacity": False,
        "use_protein_budget": False,
        "use_dissipation_limit": False,
    },
    "flux_anchored": {
        "flux": True,
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
    },
    # The exactness-budget arm: mass balance exact by construction, box soft.
    "flux_nullspace": {
        "flux": True,
        "parameterization": "nullspace",
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
    },
}


def build_dataset() -> Neo4jCellDataset:
    """Open the already-built pigment/metabolome dataset. Never rebuilds."""
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    with open(QUERY_PATH) as f:
        query = f.read()
    return Neo4jCellDataset(
        root=DATASET_ROOT,
        query=query,
        gene_set=genome.gene_set,
        graphs=build_gene_multigraph(graph=graph, graph_names=["physical", "regulatory"]),
        incidence_graphs=None,
        node_embeddings={},
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=DeletionKeyedGenotypeAggregator,
        graph_processor=Perturbation(),
    )


def build_flux_layer(
    arm: dict[str, Any], model_gene_ids: list[str], hidden_dim: int
) -> tuple[FluxLayer | None, dict[str, Any]]:
    """Construct the flux layer for one arm, with real MW and OED-resolved kcat."""
    if not arm["flux"]:
        return None, {}
    source = YeastGEM()
    gem = build_gem_tensors(source.model, model_dir=source.model_dir)
    units = gem.catalytic_units
    mw = molecular_weight_table(source.model_dir, units.gene_ids)
    kcat = resolve_kcat_table(units, source.model_dir, OED_MIRROR)
    parameterization = arm.get("parameterization", "box")
    cfg = FluxLayerConfig(
        hidden_dim=hidden_dim,
        parameterization=parameterization,
        thermo_mode=arm["thermo_mode"],
        use_enzyme_capacity=arm["use_enzyme_capacity"],
        use_protein_budget=arm["use_protein_budget"],
        use_dissipation_limit=arm["use_dissipation_limit"],
        stochastic=bool(arm.get("stochastic", False)),
        # The physics terms are dimensionless but not automatically COMMENSURATE with the
        # data term. Measured at init on a real batch, the unweighted constraint sum is
        # ~255 against a data loss of ~2, so at weight 1 the model spends every parameter
        # on feasibility and none on the phenotype. These weights put the two within an
        # order of magnitude of each other at initialization.
        weights={
            "balance": 0.05,
            "thermo": 0.5,
            "capacity": 1.0,
            "budget": 0.5,
            "dissipation": 0.05,
            "parsimony": 1.0e-3,
            "thermo_prior": 1.0e-2,
            "box": 0.5,
        },
    )
    basis = None
    if parameterization == "nullspace":
        basis = null_space_basis(
            gem.s,
            cache_path=osp.join(
                DATA_ROOT, "data/torchcell/yeast-GEM/null_space_basis_9_0_2.npy"
            ),
        )
    layer = FluxLayer(
        gem,
        model_gene_ids,
        config=cfg,
        kcat_per_s=kcat.values,
        molecular_weight_kda=mw.values,
        null_space=basis,
    )
    meta = {
        "coverage": layer.coverage_report(),
        "kcat_experimental_fraction": float(kcat.experimental_coverage.fraction),
        "mw_experimental_fraction": float(mw.experimental_coverage.fraction),
        "met_index": {m: i for i, m in enumerate(gem.met_ids)},
    }
    return layer, meta


def extract_targets(
    batch: Any, head_specs: dict[str, dict[str, Any]]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Pull ``(target, mask)`` per head out of the COO phenotype encoding.

    THREE COLLATION FACTS, and getting any of them wrong yields a plausible number
    rather than an error. This decode follows the one validated in the 019 harness
    (``train_cgt_multitask.py:_extract_targets_and_masks``).

    1. **The batch row is ``phenotype_values_batch``, not ``phenotype_sample_indices``.**
       ``phenotype_sample_indices`` indexes the EXPERIMENT within a genotype group and is
       not offset across the batch, so every graph's values collide at 0, 1, 2. Using it as
       the row index writes a 64-strain batch's targets into the first three rows.
    2. **``phenotype_types`` collates to a list-of-lists**, one name list per genotype, and
       the type index is LOCAL to its graph. A value's name is
       ``phenotype_types[b][type_idx]``.
    3. **The dict keys are dropped, so two heads sharing a label are separated only by
       group SIZE.** Betaxanthin and the Mulleder metabolome are both a
       ``MetabolitePhenotype`` and both arrive as ``metabolite_level``. Within one
       genotype they are two experiments, hence two groups under
       ``phenotype_sample_indices``, of size 1 and 19. **Their order is not fixed** --
       measured over 1,200 records, the 1-value betaxanthin group comes first in only
       ~11 % of the co-measured genotypes -- so splitting a 20-value block by position is
       wrong roughly nine times in ten. Grouping and matching on width is what makes the
       assignment identity-based.
    """
    gene = batch["gene"]
    values = gene.phenotype_values
    type_idx = gene.phenotype_type_indices
    val_batch = gene.phenotype_values_batch
    samp_idx = gene.phenotype_sample_indices
    raw_types = gene["phenotype_types"]
    per_graph_types: list[list[str]] = (
        [list(raw_types)]
        if len(raw_types) > 0 and isinstance(raw_types[0], str)
        else [list(t) for t in raw_types]
    )
    batch_size = int(batch.num_graphs)

    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, spec in head_specs.items():
        label = spec["label"]
        width = int(spec["width"])
        target = values.new_zeros(batch_size, width)
        mask = torch.zeros(batch_size, width, dtype=torch.bool, device=values.device)
        for b in range(batch_size):
            sel_b = val_batch == b
            if not bool(sel_b.any()):
                continue
            names_b = per_graph_types[b]
            tb = type_idx[sel_b].tolist()
            keep = torch.tensor(
                [names_b[t] == label for t in tb],
                dtype=torch.bool,
                device=values.device,
            )
            if not bool(keep.any()):
                continue
            cand_vals = values[sel_b][keep]
            cand_samp = samp_idx[sel_b][keep]
            groups = [
                cand_vals[cand_samp == s]
                for s in cand_samp.unique(sorted=True).tolist()
            ]
            matched = [g for g in groups if g.numel() == width]
            if not matched:
                continue
            target[b] = torch.stack(matched, dim=0).mean(dim=0)
            mask[b] = True
        out[name] = (target, mask)
    return out


def masked_pearson(
    pred: np.ndarray, true: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    """Per-feature Pearson over observed rows, plus the fraction of predictions that are
    not finite.

    **Non-finite predictions are excluded and COUNTED, not silently propagated.** The
    earlier version tested ``p.std() < 1e-12`` first, and a NaN fails that comparison, so a
    single NaN prediction turned the whole feature's correlation into NaN and the arm's
    metric into NaN with no indication of why. Worse, it was indistinguishable from a
    collapsed constant prediction, which the same branch reports as exactly 0.0: the
    flux_nullspace seed-1234 run alternated between ``nan`` and ``0.0000`` across epochs
    and neither value said which failure was happening.

    Returning the non-finite fraction alongside the score makes the distinction visible,
    which matters because "the model predicted a constant" and "the model produced NaN" are
    different defects with different fixes.

    Returns:
        ``(mean per-feature Pearson, fraction of observed predictions that are non-finite)``.
    """
    scores: list[float] = []
    n_obs = 0
    n_bad = 0
    for c in range(pred.shape[1]):
        sel = mask[:, c]
        if sel.sum() < 10:
            continue
        p, t = pred[sel, c], true[sel, c]
        finite = np.isfinite(p) & np.isfinite(t)
        n_obs += int(finite.size)
        n_bad += int((~finite).sum())
        p, t = p[finite], t[finite]
        # RELATIVE variance guard. An absolute floor of 1e-12 is the wrong test: a head
        # that has collapsed onto a constant does not emit a bitwise-identical value, it
        # emits one with a residual spread of order 1e-11, which clears an absolute floor
        # and then makes `corrcoef` divide by a number indistinguishable from zero. The
        # result is NaN, reported as if the metric were unavailable, when what actually
        # happened is the documented mean-collapse failure mode. Measured on the
        # flux_nullspace seed-1234 rerun: 8 of 20 epochs returned NaN and 3 returned
        # exactly 0.0, and both are the same event seen through different sides of the
        # floor. Scaling by the target's spread makes "collapsed" scale-free.
        if p.size < 10 or t.std() < 1e-12 or p.std() < 1e-8 * t.std():
            scores.append(0.0)
            continue
        score = float(np.corrcoef(p, t)[0, 1])
        scores.append(score if np.isfinite(score) else 0.0)
    return (
        float(np.mean(scores)) if scores else float("nan"),
        (n_bad / n_obs) if n_obs else 0.0,
    )


def run_arm(
    arm_name: str,
    seed: int,
    dataset: Neo4jCellDataset,
    args: argparse.Namespace,
    arm_override: dict[str, Any] | None = None,
    return_model: bool = False,
) -> dict[str, Any]:
    """Train one arm at one seed and return its metrics and feasibility trace.

    Args:
        arm_name: Key into :data:`ARMS`, or a label when ``arm_override`` is given.
        seed: Split seed and weight-init seed, held identical across arms.
        dataset: The already-built pigment/metabolome dataset.
        args: Parsed hyperparameters.
        arm_override: An arm dict used instead of the registry entry, for one-off
            configurations such as the stochastic sampler.
        return_model: Include the trained model under ``"model"``. The sampling demo needs
            a FITTED posterior; sampling an untrained prior would make its width comparison
            against flux variability analysis meaningless.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    arm = arm_override if arm_override is not None else ARMS[arm_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gene_ids = list(dataset.cell_graph["gene"].node_ids)
    flux_layer, flux_meta = build_flux_layer(arm, gene_ids, args.hidden)

    aa_keys = sorted(AMINO_ACID_TO_GEM_METABOLITE)
    # Both phenotypes carry the label `metabolite_level`; `width` is what tells them apart.
    head_specs: dict[str, dict[str, Any]] = {
        "betaxanthin": {"label": "metabolite_level", "width": 1},
        "mulleder19": {"label": "metabolite_level", "width": len(aa_keys)},
    }

    heads_config: dict[str, Any] = {}
    if arm["flux"]:
        met_index = flux_meta["met_index"]
        heads_config["betaxanthin"] = {
            "kind": "flux_scalar",
            "output_dim": 1,
            "precursor_indices": [
                met_index[m] for m in BETAXANTHIN_PRECURSORS if m in met_index
            ],
        }
        heads_config["mulleder19"] = {
            "kind": "flux_metabolite",
            "output_dim": 19,
            "metabolite_indices": [
                met_index[AMINO_ACID_TO_GEM_METABOLITE[k]] for k in aa_keys
            ],
        }
    else:
        heads_config["betaxanthin"] = {"kind": "scalar", "output_dim": 1}
        heads_config["mulleder19"] = {"kind": "vector", "output_dim": 19}

    model = CellGraphTransformerMetabolism(
        cell_graph=dataset.cell_graph,
        gene_num=len(gene_ids),
        hidden_channels=args.hidden,
        num_transformer_layers=args.layers,
        num_attention_heads=4,
        dropout=0.1,
        heads_config=heads_config,
        learnable_embedding_config={
            "enabled": True,
            "size": args.hidden,
            "preprocessor": {"num_layers": 2, "dropout": 0.1},
        },
        flux_layer=flux_layer,
    ).to(device)

    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATASET_ROOT, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=args.batch_size,
        random_seed=seed,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch=False,
        follow_batch=["perturbation_indices", "phenotype_values"],
    )
    dm.setup()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss(reduction="none")

    # Target standardization from the TRAIN split only: a metric computed against
    # test-informed statistics is not a held-out metric.
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    acc: dict[str, list[torch.Tensor]] = {k: [] for k in head_specs}
    for batch in dm.train_dataloader():
        for name, (t, m) in extract_targets(batch, head_specs).items():
            acc[name].append(torch.where(m, t, torch.nan))
    for name, chunks in acc.items():
        cat = torch.cat(chunks, dim=0)
        mean = torch.nanmean(cat, dim=0)
        centered = cat - mean
        var = torch.nanmean(centered * centered, dim=0)
        stats[name] = (mean.to(device), var.sqrt().clamp(min=1e-6).to(device))

    history: list[dict[str, Any]] = []
    best = {"val_betaxanthin": -np.inf, "val_mulleder19": -np.inf, "epoch": -1}
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        feas: dict[str, float] = {}
        for batch in dm.train_dataloader():
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, reps = model(dataset.cell_graph.to(device), batch)
            head_outputs = reps["head_outputs"]
            targets = extract_targets(batch, head_specs)
            loss = torch.zeros((), device=device)
            for name in head_specs:
                t, m = targets[name]
                t, m = t.to(device), m.to(device)
                mean, sd = stats[name]
                z = (t - mean) / sd
                pred = head_outputs[name]
                if pred.dim() == 3:
                    pred = pred[..., 0]
                per = loss_fn(pred, z) * m
                denom = m.sum().clamp(min=1)
                loss = loss + per.sum() / denom
            loss = loss + reps["graph_reg_loss"]
            if "flux" in reps:
                flux = reps["flux"]
                loss = loss + model.flux_layer.constraint_loss(flux)
                for k, v in flux.items():
                    # Scalars only: `c_j` is [B, r] reaction availability, not a residual,
                    # and shares the `c_` prefix with the constraint terms.
                    if v.dim() == 0 and k.startswith(
                        ("c_", "feas_", "g_diss", "protein_used")
                    ):
                        feas[k] = feas.get(k, 0.0) + float(v.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += float(loss.detach())
            n_batches += 1

        model.eval()
        preds: dict[str, list[np.ndarray]] = {k: [] for k in head_specs}
        trues: dict[str, list[np.ndarray]] = {k: [] for k in head_specs}
        masks: dict[str, list[np.ndarray]] = {k: [] for k in head_specs}
        with torch.no_grad():
            for batch in dm.val_dataloader():
                batch = batch.to(device)
                _, reps = model(dataset.cell_graph.to(device), batch)
                targets = extract_targets(batch, head_specs)
                for name in head_specs:
                    t, m = targets[name]
                    mean, sd = stats[name]
                    pred = reps["head_outputs"][name]
                    if pred.dim() == 3:
                        pred = pred[..., 0]
                    preds[name].append((pred * sd + mean).cpu().numpy())
                    trues[name].append(t.cpu().numpy())
                    masks[name].append(m.cpu().numpy())

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_batches, 1),
        }
        for name in head_specs:
            score, nonfinite = masked_pearson(
                np.concatenate(preds[name]),
                np.concatenate(trues[name]),
                np.concatenate(masks[name]),
            )
            row[f"val_{name}"] = score
            row[f"nonfinite_{name}"] = nonfinite
        for k, v in feas.items():
            row[k] = v / max(n_batches, 1)
        history.append(row)
        # `nan > x` is False, so a NaN epoch can never become the best. That is the
        # intended behavior: an epoch whose metric could not be computed is not a result.
        if row["val_betaxanthin"] > best["val_betaxanthin"]:
            best = {
                "val_betaxanthin": row["val_betaxanthin"],
                "val_mulleder19": row["val_mulleder19"],
                "epoch": epoch,
            }
        # Print from `row`, which holds the per-batch MEANS; `feas` holds the running sums
        # and printing it reports a violation fraction of 23.8, which is not a fraction.
        print(
            f"[{arm_name} s{seed}] ep {epoch:3d} loss {row['train_loss']:.4f} "
            f"bx {row['val_betaxanthin']:+.4f} aa {row['val_mulleder19']:+.4f} "
            + " ".join(
                f"{k.removeprefix('feas_')} {v:.3g}"
                for k, v in row.items()
                if k.startswith("feas_")
            ),
            flush=True,
        )

    out: dict[str, Any] = {
        "arm": arm_name,
        "seed": seed,
        "best": best,
        "final": history[-1] if history else {},
        "history": history,
        "flux_meta": {k: v for k, v in flux_meta.items() if k != "met_index"},
        "n_parameters": model.num_parameters,
        "wall_time_s": time.time() - t0,
    }
    if return_model:
        out["model"] = model
    return out


def main() -> None:
    """Run the requested arms across the requested seeds, checkpointing after each."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="pooled,flux_off,flux_free,flux_anchored")
    parser.add_argument("--seeds", default="42,7,1234")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--out", default="flux_arms.json")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = build_dataset()
    print(f"dataset: {len(dataset)} aggregated genotypes", flush=True)

    out_path = osp.join(RESULTS_DIR, args.out)
    runs: list[dict[str, Any]] = []
    # SEED-MAJOR, and the ordering is load-bearing under a deadline. Arm-major finishes
    # every seed of arm 1 before starting arm 2, so an interrupted sweep yields complete
    # data for some arms and none for others, which supports no comparison at all.
    # Seed-major finishes one seed of EVERY arm first, so any prefix of the run is a
    # usable, balanced experiment and each further pass just adds a replicate.
    for seed in [int(s) for s in args.seeds.split(",")]:
        for arm_name in args.arms.split(","):
            runs.append(run_arm(arm_name, seed, dataset, args))
            with open(out_path, "w") as f:
                json.dump({"args": vars(args), "runs": runs}, f, indent=2, default=str)
            print(f"checkpointed {len(runs)} runs -> {out_path}", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
