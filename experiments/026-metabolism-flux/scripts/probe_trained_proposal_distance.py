# experiments/026-metabolism-flux/scripts/probe_trained_proposal_distance.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_trained_proposal_distance
r"""How far is a TRAINED head's proposal from the feasible set?

THE GAP THIS CLOSES. ``probe_warmstart.py`` shows that the existing K = 100 alternation
clears the turnover-thresholded balance rule when it starts within about one percent of a
feasible flux, and fails at 2.0 from a uniform random point of the box. Both of those are
synthetic starts. Nothing had measured where a trained checkpoint's own proposal sits
between them, and the whole case for a warm-started head rests on it being far rather than
near.

WHAT IS MEASURED. The layer already carries its pre-projection flux out of ``forward`` as
``v_pre_projection`` on the ``fluxforce`` channel, so no re-derivation is needed: the
statistic is

.. math::
    \mathrm{rel} = \frac{\lVert v_{\mathrm{pre}} - P(v_{\mathrm{pre}})\rVert}
                        {\lVert v_{\mathrm{pre}}\rVert},

per genotype, with :math:`P` the layer's own ``_project_feasible`` at the cell's own K,
into the cell's own box. ``probe_warmstart.py`` records the same ratio beside each of its
``rel`` rows, so the band a checkpoint lands in is read off that table rather than assumed.

A head whose proposal is near the polytope makes the warm start nearly free; one whose
proposal is an O(1) distance away makes the warm start the whole experiment.

Run from the repo root, on CPU::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_trained_proposal_distance.py \
        --result wave8_dens_d2p0_boxforce_k100_e80_s2700.json --genotypes 32

Writes ``experiments/026-metabolism-flux/results/probe_trained_proposal_distance_<stem>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
BX_SCRIPTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts")
BX_RESULTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
sys.path.insert(0, BX_SCRIPTS)

from train_bx import (  # noqa: E402
    BxCell,
    build_dataset,
    build_model_for_cell,
    load_config,
    resolve_pinned_test,
)

from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.metabolism.flux_layer import FluxLayer  # noqa: E402

FLUX_ACTIVE_RELATIVE_THRESHOLD = 1.0e-6
TURNOVER_THRESHOLD = 1.0e-3

#: The bands of ``probe_warmstart.py``'s table, keyed by the `rel` of the proposal that
#: produced them. Filled from that script's JSON when it is present, so the placement of a
#: checkpoint is read off a committed measurement rather than a transcribed number.
WARMSTART_JSON = "probe_warmstart.json"


def _bands() -> list[dict[str, object]]:
    """The K = 100 rows of ``probe_warmstart.py``, with their projection displacement."""
    path = osp.join(RESULTS_DIR, WARMSTART_JSON)
    if not osp.isfile(path):
        return []
    with open(path) as handle:
        payload = json.load(handle)
    return [
        row
        for row in payload["rows"]
        if "K=100" in str(row["tag"]) and "projection_displacement_rel" in row
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="wave8_dens_d2p0_boxforce_k100_e80_s2700.json",
        help="A 027 result file under its results/. The default is job 1784.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--genotypes", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(osp.join(BX_RESULTS, args.result)) as handle:
        result = json.load(handle)
    cell = BxCell(**result["cell"])
    cfg = load_config()
    ckpt_path = args.checkpoint or result["checkpoint_path"]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["cell"]["seed"] == cell.seed, (
        f"checkpoint is for seed {checkpoint['cell']['seed']}, result for {cell.seed}."
    )

    dataset = build_dataset()
    pinned, _ = resolve_pinned_test(dataset, cfg)
    device = torch.device("cpu")
    model, _, _, _ = build_model_for_cell(cell, cfg, dataset, device)
    assert model.flux_layer is not None, f"arm {cell.arm!r} has no flux layer."
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(checkpoint["parameters"][name].to(param.device))
    model.eval()
    layer = model.flux_layer
    assert isinstance(layer, FluxLayer)

    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, cfg.data.dataset_subdir, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cell.batch_size,
        random_seed=cell.split_seed,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()

    scale = float(layer.config.flux_scale)
    collected: dict[str, list[np.ndarray]] = {}
    seen = 0
    cell_graph = dataset.cell_graph.to(device)
    with torch.no_grad():
        for batch in dm.all_dataloader():
            batch = batch.to(device)
            _, reps = model(cell_graph, batch)
            flux = reps["flux"]
            assert "v_pre_projection" in flux, (
                f"the arm {cell.arm!r} on channel "
                f"{layer.config.parameterization!r} does not carry v_pre_projection. "
                "Only the fluxforce and warmstart channels emit a pre-projection flux; "
                "there is nothing to measure the projection's displacement against."
            )
            v_pre = flux["v_pre_projection"]
            v = flux["v"]
            residual = layer._s_matmul(v)[:, layer.independent_rows].abs()
            omega = layer.turnover(v)[:, layer.independent_rows]
            ratio = residual / (omega + layer.config.balance_epsilon)
            active = omega > TURNOVER_THRESHOLD * scale
            pre_norm = v_pre.norm(dim=1)
            per = {
                "proposal_distance_rel": (
                    (v - v_pre).norm(dim=1) / pre_norm.clamp(min=1e-12)
                ),
                "v_pre_norm": pre_norm,
                "v_norm": v.norm(dim=1),
                "projection_distance_abs": (v - v_pre).norm(dim=1),
                "balance_active_max": ratio.masked_fill(~active, float("-inf")).max(
                    dim=1
                ).values,
                "balance_active_rows": active.sum(dim=1).to(ratio.dtype),
                "flux_active_frac": (
                    (v.abs() > FLUX_ACTIVE_RELATIVE_THRESHOLD * scale).float().mean(dim=1)
                ),
                "v_pre_active_frac": (
                    (v_pre.abs() > FLUX_ACTIVE_RELATIVE_THRESHOLD * scale)
                    .float()
                    .mean(dim=1)
                ),
            }
            for key, value in per.items():
                collected.setdefault(key, []).append(value.detach().cpu().numpy())
            seen += int(batch.num_graphs)
            if seen >= args.genotypes:
                break
    stacked = {k: np.concatenate(v)[: args.genotypes] for k, v in collected.items()}

    def summary(x: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(x)),
            "mean": float(np.mean(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    bands = _bands()
    measured = float(np.median(stacked["proposal_distance_rel"]))
    placement = None
    if bands:
        nearest = min(
            bands,
            key=lambda row: abs(
                float(row["projection_displacement_rel"]) - measured
            ),
        )
        placement = {
            "nearest_band_tag": nearest["tag"],
            "nearest_band_projection_displacement_rel": float(
                nearest["projection_displacement_rel"]
            ),
            "nearest_band_balance_active_max": float(nearest["balance_active_max"]),
            "bands": [
                {
                    "tag": row["tag"],
                    "projection_displacement_rel": float(
                        row["projection_displacement_rel"]
                    ),
                    "balance_active_max": float(row["balance_active_max"]),
                }
                for row in bands
            ],
        }

    payload = {
        "definitions": {
            "proposal_distance_rel": (
                "||v_pre - P(v_pre)|| / ||v_pre|| per genotype, with P the layer's own "
                "_project_feasible at this cell's K and box. The fraction of the proposal "
                "the projection discards."
            ),
            "placement": (
                "the row of probe_warmstart.json whose own projection displacement is "
                "closest to the measured median, which is the `rel` band this checkpoint "
                "sits in."
            ),
        },
        "result_file": args.result,
        "checkpoint": ckpt_path,
        "cell": result["cell"],
        "select_epoch": int(result["select_epoch"]),
        "n_genotypes": int(stacked["proposal_distance_rel"].size),
        "genotype_selection": (
            "the first genotypes of the all-split loader in dataset order; the proposal "
            "distance is a property of the head and the box rather than of the ranking"
        ),
        "projection_steps": int(layer.projection_steps),
        "projection_end": str(layer.projection_end),
        "flux_scale": scale,
        "statistics": {k: summary(v) for k, v in stacked.items()},
        "placement": placement,
    }
    stem = osp.splitext(args.result)[0]
    out_path = osp.join(
        RESULTS_DIR, args.out or f"probe_trained_proposal_distance_{stem}.json"
    )
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["statistics"], indent=2))
    if placement is not None:
        print(f"placement: {placement['nearest_band_tag']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
