# experiments/042-hpc-smoke-tests/scripts/smoke_train.py
# [[experiments.042-hpc-smoke-tests.scripts.smoke_train]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/042-hpc-smoke-tests/scripts/smoke_train
"""GPU + W&B smoke test for a freshly installed torchcell environment.

Proves four things on one accelerator, in order, and exits non-zero at the first one
that fails:

1. the environment imports (torch, torch_geometric, lightning, wandb, torchcell);
2. a GPU is visible (CUDA or Intel XPU) and a scatter-add runs on it, through
   torch_scatter where it is installed and through torch's native index_add_ otherwise;
3. a training loop on synthetic data lowers the loss on that device;
4. every step is logged to W&B online, and on CUDA a Lightning Trainer with WandbLogger
   runs as well (Lightning has no XPU accelerator, so Aurora logs the manual loop only).

Synthetic data on purpose: this measures the machine, not a dataset, so no store has to
cross before the first job. Writes a JSON summary beside the job output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import torch
from torch import nn


def pick_device(allow_cpu: bool) -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda", 0), torch.cuda.get_device_name(0)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu", 0), torch.xpu.get_device_name(0)
    if allow_cpu:
        return torch.device("cpu"), platform.processor() or "cpu"
    raise SystemExit("no GPU visible: torch.cuda and torch.xpu both report unavailable")


def scatter_add(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> tuple[torch.Tensor, str]:
    if importlib.util.find_spec("torch_scatter") is not None:
        from torch_scatter import scatter_add as ts_scatter_add

        return ts_scatter_add(src, index, dim=0, dim_size=dim_size), "torch_scatter"
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    return out.index_add_(0, index, src), "torch.index_add_"


class PoolMLP(nn.Module):
    """Node features -> graph pooling by scatter-add -> regression head."""

    def __init__(self, width: int) -> None:
        """Two-layer node encoder of the given width and a scalar head."""
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(16, width), nn.GELU(), nn.Linear(width, width)
        )
        self.head = nn.Linear(width, 1)
        self.scatter_impl = "unset"

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor, n_graphs: int
    ) -> torch.Tensor:
        """Encode nodes, pool them per graph with a scatter-add, predict one value per graph."""
        h = self.enc(x)
        pooled, self.scatter_impl = scatter_add(h, batch, n_graphs)
        return self.head(pooled).squeeze(-1)


def make_batch(
    device: torch.device, gen: torch.Generator, n_graphs: int, nodes_per_graph: int
):
    x = torch.randn(n_graphs * nodes_per_graph, 16, generator=gen).to(device)
    batch = torch.arange(n_graphs).repeat_interleave(nodes_per_graph).to(device)
    # Target: a fixed linear function of the summed node features, so the loss can fall.
    w = torch.linspace(-1, 1, 16).to(device)
    y = (x @ w).view(n_graphs, nodes_per_graph).sum(1) / nodes_per_graph
    return x, batch, y


def manual_loop(run, device: torch.device, steps: int, width: int) -> dict:
    gen = torch.Generator().manual_seed(0)
    model = PoolMLP(width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = last = float("nan")
    t0 = time.perf_counter()
    for step in range(steps):
        x, batch, y = make_batch(device, gen, n_graphs=256, nodes_per_graph=64)
        loss = nn.functional.mse_loss(model(x, batch, 256), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
        run.log({"manual/loss": last, "manual/step": step}, step=step)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "first_loss": first,
        "last_loss": last,
        "steps": steps,
        "steps_per_s": steps / elapsed,
        "scatter_impl": model.scatter_impl,
    }


def lightning_loop(run, steps: int, width: int) -> dict:
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger
    from torch.utils.data import DataLoader, TensorDataset

    gen = torch.Generator().manual_seed(1)
    x = torch.randn(4096, 16, generator=gen)
    y = x @ torch.linspace(-1, 1, 16)

    class Reg(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(16, width), nn.GELU(), nn.Linear(width, 1)
            )

        def training_step(self, batch, _idx):
            xb, yb = batch
            loss = nn.functional.mse_loss(self.net(xb).squeeze(-1), yb)
            self.log("lightning/loss", loss, on_step=True, prog_bar=False)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    logger = WandbLogger(experiment=run)
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=steps,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=5,
    )
    trainer.fit(Reg(), DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True))
    last = trainer.callback_metrics.get("lightning/loss")
    return {
        "steps": trainer.global_step,
        "last_loss": float(last) if last is not None else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--machine", required=True, help="polaris | aurora | perlmutter | ..."
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--project", default="torchcell_042_hpc_smoke")
    p.add_argument("--out", type=Path, required=True, help="JSON summary path")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="run on CPU when no GPU is visible (script validation only; a GPU smoke test never passes this)",
    )
    args = p.parse_args()

    import lightning
    import torch_geometric
    import wandb

    import torchcell

    device, device_name = pick_device(args.allow_cpu)
    info = {
        "machine": args.machine,
        "hostname": socket.gethostname(),
        "job_id": os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__,
        "lightning": lightning.__version__,
        "wandb": wandb.__version__,
        "torchcell_file": torchcell.__file__,
        "device": str(device),
        "device_name": device_name,
        "wandb_mode": os.environ.get("WANDB_MODE", "online"),
    }
    print(json.dumps(info, indent=2), flush=True)

    run = wandb.init(
        project=args.project,
        name=f"{args.machine}-{info['job_id'] or 'local'}",
        group=args.machine,
        config=info,
        tags=["042", "smoke", args.machine, device.type],
    )
    result = {"info": info, "manual": manual_loop(run, device, args.steps, args.width)}
    if device.type == "cuda":
        result["lightning"] = lightning_loop(run, args.steps // 3, args.width)
    else:
        result["lightning"] = {"skipped": f"no Lightning accelerator for {device.type}"}
    run.summary.update(
        {
            "manual/first_loss": result["manual"]["first_loss"],
            "manual/last_loss": result["manual"]["last_loss"],
            "manual/steps_per_s": result["manual"]["steps_per_s"],
            "scatter_impl": result["manual"]["scatter_impl"],
        }
    )
    result["wandb_url"] = run.url
    run.finish()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

    if not result["manual"]["last_loss"] < result["manual"]["first_loss"]:
        raise SystemExit("loss did not fall: training did not run")
    print(f"SMOKE OK {args.machine} {device_name} {run.url}", flush=True)


if __name__ == "__main__":
    main()
