# experiments/027-betaxanthin-metabolic/scripts/bench_flux_draws.py
# [[experiments.027-betaxanthin-metabolic.wave8-sampling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/bench_flux_draws.py

r"""What K latent draws per genotype cost, on the real yeast-GEM, before a GPU is booked.

THE ONLY THING K CHANGES is the flux layer. The genotype chain, the box and the reaction
energies are computed once and repeated, and the transformer backbone in front of the layer
never sees the extra rows at all, so the added cost is exactly the layer's forward and
backward on ``B*K`` rows instead of ``B``. This times that, at the real width (4,131
reactions, a 1,538-dimensional null basis) and the real batch size, for K in a list.

Reported per K: the median wall time of one forward plus backward, the ratio to K = 1, and
the peak resident bytes of the flux tensors the expansion allocates. Times are CPU times and
are NOT a prediction of the GPU ratio, which is measured from the first submitted pair's
epoch times; a CPU that is already compute-bound at K = 1 overstates the slope. The number
this decides is whether the expansion is linear in K, which it should be, or worse, which
would mean something is being recomputed per draw.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/bench_flux_draws.py

Writes ``results/wave8_draw_cost.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import os.path as osp
import time
from typing import Any

import torch
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.timestamp import timestamp

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
RESULTS = osp.join(EXP_DIR, "results")


class DrawCost(BaseModel):
    """One K: what the forward plus backward costs and what it allocates."""

    model_config = ConfigDict(extra="forbid")

    draws: int
    batch: int
    rows: int = Field(description="B * draws, the width the layer runs at.")
    seconds_median: float
    seconds_min: float
    ratio_to_one_draw: float
    flux_tensor_mib: float = Field(
        description=(
            "Bytes of ONE [rows, n_reactions] float32 tensor, in MiB. The projection loop "
            "holds a handful of these at once, so it is the unit the memory scales in."
        )
    )


def _build_layer(draws: int, hidden: int, latent_stochastic: bool) -> Any:
    """The plain projection layer wave 8's arm (a) runs, at the given draw count."""
    spec = importlib.util.spec_from_file_location(
        "train_flux_for_bench",
        osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts", "train_flux.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arm = {
        **module.ARMS["flux_projection"],
        "kcat_tier": "predicted",
        "thermo_source": "union",
        "constraint_warmup_epochs": 20,
        "constraint_final_scale": 0.0,
    }
    layer, _ = module.build_flux_layer(arm, [], hidden)
    assert layer is not None
    layer.config.latent_stochastic = latent_stochastic
    layer.config.draws = draws
    return layer


def bench(draws: int, batch: int, hidden: int, repeats: int) -> DrawCost:
    """Time one forward plus backward of the layer at this draw count."""
    torch.manual_seed(0)
    layer = _build_layer(draws, hidden, latent_stochastic=True)
    layer.train()
    n_genes = 64
    h = torch.randn(batch, n_genes, hidden, requires_grad=True)
    context = torch.randn(batch, hidden, requires_grad=True)
    pert = torch.zeros(0, dtype=torch.long)
    pert_batch = torch.zeros(0, dtype=torch.long)

    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = layer(h, context, pert, pert_batch)
        (out["v"].square().mean() + layer.constraint_loss(out)).backward()
        times.append(time.perf_counter() - start)
        layer.zero_grad(set_to_none=True)
        h.grad = None
        context.grad = None
    times.sort()
    rows = batch * draws
    return DrawCost(
        draws=draws,
        batch=batch,
        rows=rows,
        seconds_median=times[len(times) // 2],
        seconds_min=times[0],
        ratio_to_one_draw=1.0,
        flux_tensor_mib=rows * int(layer.n_reactions) * 4 / (1024.0**2),
    )


def main() -> None:
    """Benchmark every requested K and write the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, nargs="*", default=[1, 4, 16])
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=90)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="wave8_draw_cost.json")
    args = parser.parse_args()

    rows: list[DrawCost] = []
    for k in args.draws:
        row = bench(k, args.batch, args.hidden, args.repeats)
        rows.append(row)
        print(
            f"draws={k:3d} rows={row.rows:5d} median {row.seconds_median:7.3f} s "
            f"flux tensor {row.flux_tensor_mib:7.1f} MiB",
            flush=True,
        )
    base = rows[0].seconds_median
    for row in rows:
        row.ratio_to_one_draw = row.seconds_median / base

    payload = {
        "generated_at": timestamp(),
        "device": "cpu",
        "note": (
            "CPU forward plus backward of the flux layer alone. The backbone is unaffected "
            "by K, so this is the whole of the added cost; the GPU ratio is measured from "
            "the submitted pairs' epoch times."
        ),
        "rows": [row.model_dump() for row in rows],
    }
    out_path = osp.join(RESULTS, args.out)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
