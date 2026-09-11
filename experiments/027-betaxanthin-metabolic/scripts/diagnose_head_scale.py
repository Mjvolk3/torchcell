# experiments/027-betaxanthin-metabolic/scripts/diagnose_head_scale.py
# [[experiments.027-betaxanthin-metabolic.scripts.diagnose_head_scale]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/diagnose_head_scale.py

r"""Where does each scalar head START, against the standardized target it is scored on?

THE OBSERVATION THIS EXISTS TO EXPLAIN. With the constraint weight at exactly zero (epoch 0
of the wave-5 warmup cells, jobs 1663 to 1665), the flux arms' training data loss is 8.5 to
14.3 while the pooled arm on the same recipe runs a validation data loss of 0.49 (job 1634).
The constraint terms cannot be the cause at that epoch, because there are none. The
remaining candidate is the readout: the target is standardized to mean 0, sd 1, and a head
whose untrained output has sd 10 pays a CRPS of order 10 before it has learned anything.

WHAT IS MEASURED, on CPU, at initialization, on one real training batch:

* the raw output of each head (mu channel), mean and sd across the batch;
* the standardized target's mean and sd on the same batch, which is what the head is
  compared against;
* the inputs each head reads, so the number can be attributed rather than guessed: the
  hidden state for the pooled head, and the flux vector and precursor turnover for the flux
  head, with the contribution of each of the flux head's two terms separated.

Run from the repo root::

    EXPERIMENT_ROOT=$PWD/experiments PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/diagnose_head_scale.py

Writes ``results/head_scale_diagnosis.json``.
"""

import json
import os
import os.path as osp
import sys
from typing import Any, cast

import torch
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()

EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
SCRIPTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts")

sys.path.insert(0, SCRIPTS)
import train_bx  # noqa: E402
from train_bx import HEAD, build_cell, build_dataset, load_config  # noqa: E402


class HeadScale(BaseModel):
    """One head's output scale at initialization, against the target it is scored on."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    head_kind: str
    out_mean: float
    out_sd: float
    target_mean: float
    target_sd: float
    ratio_sd_to_target: float
    detail: dict[str, float]


def _stats(x: torch.Tensor) -> tuple[float, float]:
    return float(x.mean()), float(x.std())


def measure(arm_name: str, dataset: Any, cfg: Any, batch: Any) -> HeadScale:
    """Build one arm's model at init and read its head's raw output on ``batch``."""
    from train_flux import ARMS, build_flux_layer  # noqa: E402

    torch.manual_seed(2700)
    cell = build_cell(arm_name, 2700, cfg, epochs=1, crps=True, graph_reg=True)
    gene_ids = list(dataset.cell_graph["gene"].node_ids)
    arm = {
        **ARMS[cell.base_arm],
        "kcat_tier": cell.kcat_tier,
        "thermo_source": cell.thermo_source,
    }
    flux_layer, flux_meta = build_flux_layer(arm, gene_ids, cell.hidden)

    head_specs = {HEAD: {"label": "metabolite_level", "width": 1}}
    if arm["flux"]:
        met_index = flux_meta["met_index"]
        heads_config = {
            HEAD: {
                "kind": "flux_scalar",
                "output_dim": 1,
                "param_dim": 2,
                "precursor_indices": [
                    met_index[m]
                    for m in train_bx.BETAXANTHIN_PRECURSORS
                    if m in met_index
                ],
            }
        }
    else:
        heads_config = {HEAD: {"kind": "scalar", "output_dim": 1, "param_dim": 2}}

    from torchcell.models.cell_graph_transformer_metabolism import (  # noqa: E402
        CellGraphTransformerMetabolism,
    )

    model = CellGraphTransformerMetabolism(
        cell_graph=dataset.cell_graph,
        gene_num=len(gene_ids),
        hidden_channels=cell.hidden,
        num_transformer_layers=cell.layers,
        num_attention_heads=cell.attention_heads,
        dropout=0.1,
        heads_config=heads_config,
        learnable_embedding_config={
            "enabled": False,
            "size": cell.hidden,
            "preprocessor": {"num_layers": 2, "dropout": 0.1},
        },
        perturbation_head_config={"num_heads": cell.pert_heads, "dropout": 0.1},
        attention_mask_config=None,
        node_embeddings=train_bx._NODE_EMBEDDINGS,
        flux_layer=flux_layer,
    )
    model.eval()
    with torch.no_grad():
        _, reps = model(dataset.cell_graph, batch)
    raw = reps["head_outputs"][HEAD]
    mu = raw.squeeze(1)[:, 0] if raw.dim() == 3 else raw[:, 0]
    out_mean, out_sd = _stats(mu)

    detail: dict[str, float] = {}
    if arm["flux"]:
        head = getattr(model, f"{HEAD}_head")
        v = reps["flux"]["v"]
        turnover = model.flux_layer.turnover(v)
        omega = turnover[:, head.precursor_indices].clamp(min=0.0)
        with torch.no_grad():
            dense_term = head.dense(v)[:, 0]
            prec_term = head.precursor(torch.log1p(omega))[:, 0]
        detail["v_abs_mean"] = float(v.abs().mean())
        detail["v_sd"] = float(v.std())
        detail["v_abs_max"] = float(v.abs().max())
        detail["n_reactions"] = float(v.shape[1])
        detail["log1p_omega_mean"] = float(torch.log1p(omega).mean())
        detail["dense_term_sd"] = float(dense_term.std())
        detail["precursor_term_sd"] = float(prec_term.std())
        detail["dense_weight_sd"] = float(head.dense.weight.std())
    else:
        detail["hidden_cls_sd"] = float(reps["h_CLS"].std()) if "h_CLS" in reps else -1.0

    return HeadScale(
        arm=arm_name,
        head_kind=str(heads_config[HEAD]["kind"]),
        out_mean=out_mean,
        out_sd=out_sd,
        target_mean=0.0,
        target_sd=1.0,
        ratio_sd_to_target=out_sd,
        detail=detail,
    )


def main() -> None:
    cfg = load_config()
    dataset = build_dataset()
    print(f"dataset: {len(dataset)} aggregated genotypes", flush=True)

    from torchcell.data import (  # noqa: E402
        DeletionKeyedGenotypeAggregator,  # noqa: F401
    )
    from torchcell.datamodules import CellDataModule  # noqa: E402

    pinned, _ = train_bx.resolve_pinned_test(dataset, cfg)
    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(
            os.environ["DATA_ROOT"], cfg.data.dataset_subdir, "data_module_cache"
        ),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cfg.model.batch_size,
        random_seed=cfg.data.split_seed,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    rows = [measure(name, dataset, cfg, batch) for name in ("pooled", "flux_capacity")]
    payload = {
        "note": (
            "Raw head output at INITIALIZATION on one real training batch, against a target "
            "standardized to mean 0 sd 1. A head that starts at sd s pays a CRPS of order s "
            "before it has learned anything."
        ),
        "heads": [r.model_dump() for r in rows],
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = osp.join(RESULTS, "head_scale_diagnosis.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)

    for row in rows:
        print(
            f"{row.arm:16s} {row.head_kind:12s} out mean {row.out_mean:+.4f} "
            f"sd {row.out_sd:.4f} (target sd 1.0)",
            flush=True,
        )
        for k, v in row.detail.items():
            print(f"    {k:22s} {v:.6g}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
