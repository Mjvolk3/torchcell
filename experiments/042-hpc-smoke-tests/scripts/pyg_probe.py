# experiments/042-hpc-smoke-tests/scripts/pyg_probe.py
# [[experiments.042-hpc-smoke-tests.scripts.pyg_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/042-hpc-smoke-tests/scripts/pyg_probe
"""Does PyG, and does torchcell's model code, run on this accelerator?

`smoke_train.py` proved the environment, the GPU and the W&B path with a plain MLP. This
probe asks the question that matters for a real torchcell run, check by check, never
stopping at the first failure:

- `torch_geometric.utils.scatter` with every reduce, forward and backward, on the device;
- the pooling and batching utilities (`global_*_pool`, `to_dense_batch`, `softmax`);
- message-passing layers forward and backward on a random graph (GCN, SAGE, GATv2, GIN,
  TransformerConv) and a `HeteroConv` over a two-type graph;
- bf16 autocast on the device;
- `torch_scatter` import;
- an import of every module under `torchcell.models` and of `torchcell.data.graph_processor`,
  recording the exception class and first line when one fails.

Every check lands in the JSON summary and in the W&B run summary as `probe/<name>`. The exit
code is non-zero when a PyG-core check fails; a torchcell import failure is reported, not
fatal, because on XPU the `torch_scatter` imports are the expected result.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import pkgutil
import platform
import socket
import sys
import traceback
from collections.abc import Callable
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


class Probe:
    """Collects named checks; each returns a short detail string or raises."""

    def __init__(self, device: torch.device) -> None:
        """Checks run on `device`; results accumulate in `self.results` by check name."""
        self.device = device
        self.results: dict[str, dict[str, str]] = {}

    def run(self, name: str, fn: Callable[[], str], core: bool = True) -> None:
        """Run one check, record ok or the first line of its exception; `core` marks PyG checks."""
        try:
            detail = fn()
            synchronize(self.device)
            self.results[name] = {"status": "ok", "detail": detail, "core": str(core)}
            print(f"ok    {name}: {detail}", flush=True)
        except Exception as exc:  # noqa: BLE001 - the probe's purpose is to record every failure
            first = (
                f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            )
            self.results[name] = {"status": "fail", "detail": first, "core": str(core)}
            print(f"FAIL  {name}: {first}", flush=True)
            if os.environ.get("PROBE_TRACE"):
                traceback.print_exc()


def random_graph(device: torch.device, n: int = 512, e: int = 4096, f: int = 32):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(n, f, generator=gen).to(device).requires_grad_(True)
    edge_index = torch.randint(0, n, (2, e), generator=gen).to(device)
    return x, edge_index


def scatter_checks(p: Probe) -> None:
    from torch_geometric.utils import scatter

    x, edge_index = random_graph(p.device)
    for reduce in ("sum", "mean", "max", "min", "mul"):

        def fn(reduce=reduce):
            out = scatter(
                x[edge_index[0]],
                edge_index[1],
                dim=0,
                dim_size=x.shape[0],
                reduce=reduce,
            )
            out.sum().backward()
            return f"out {tuple(out.shape)} grad {tuple(x.grad.shape)}"

        p.run(f"scatter_{reduce}", fn)
        x.grad = None


def util_checks(p: Probe) -> None:
    from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
    from torch_geometric.utils import softmax, to_dense_batch

    x, edge_index = random_graph(p.device)
    batch = torch.arange(8, device=p.device).repeat_interleave(x.shape[0] // 8)
    p.run("global_add_pool", lambda: str(tuple(global_add_pool(x, batch).shape)))
    p.run("global_mean_pool", lambda: str(tuple(global_mean_pool(x, batch).shape)))
    p.run("global_max_pool", lambda: str(tuple(global_max_pool(x, batch).shape)))
    p.run("to_dense_batch", lambda: str(tuple(to_dense_batch(x, batch)[0].shape)))
    p.run(
        "softmax_by_index",
        lambda: str(tuple(softmax(x[edge_index[0], 0], edge_index[1]).shape)),
    )


def conv_checks(p: Probe) -> None:
    from torch_geometric.nn import (
        GATv2Conv,
        GCNConv,
        GINConv,
        SAGEConv,
        TransformerConv,
    )

    x, edge_index = random_graph(p.device)
    f = x.shape[1]
    layers = {
        "GCNConv": GCNConv(f, 64),
        "SAGEConv": SAGEConv(f, 64),
        "GATv2Conv": GATv2Conv(f, 16, heads=4),
        "GINConv": GINConv(
            nn.Sequential(nn.Linear(f, 64), nn.ReLU(), nn.Linear(64, 64))
        ),
        "TransformerConv": TransformerConv(f, 16, heads=4),
    }
    for name, layer in layers.items():
        layer = layer.to(p.device)

        def fn(layer=layer):
            out = layer(x, edge_index)
            out.sum().backward()
            grads = sum(1 for q in layer.parameters() if q.grad is not None)
            return f"out {tuple(out.shape)}, {grads} param grads"

        p.run(f"conv_{name}", fn)
        x.grad = None


def hetero_check(p: Probe) -> None:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv

    def fn() -> str:
        gen = torch.Generator().manual_seed(1)
        data = HeteroData()
        data["gene"].x = torch.randn(300, 16, generator=gen)
        data["reaction"].x = torch.randn(120, 16, generator=gen)
        data["gene", "ppi", "gene"].edge_index = torch.randint(
            0, 300, (2, 1500), generator=gen
        )
        data["gene", "gpr", "reaction"].edge_index = torch.stack(
            [
                torch.randint(0, 300, (400,), generator=gen),
                torch.randint(0, 120, (400,), generator=gen),
            ]
        )
        data = data.to(p.device)
        conv = HeteroConv(
            {
                ("gene", "ppi", "gene"): SAGEConv(16, 32),
                ("gene", "gpr", "reaction"): SAGEConv((16, 16), 32),
            },
            aggr="sum",
        ).to(p.device)
        out = conv(data.x_dict, data.edge_index_dict)
        sum(v.sum() for v in out.values()).backward()
        return ", ".join(f"{k} {tuple(v.shape)}" for k, v in out.items())

    p.run("HeteroConv_SAGE", fn)


def autocast_check(p: Probe) -> None:
    def fn() -> str:
        a = torch.randn(1024, 1024, device=p.device)
        with torch.autocast(device_type=p.device.type, dtype=torch.bfloat16):
            out = a @ a
        return f"dtype {out.dtype}"

    p.run("autocast_bf16", fn)


def import_checks(p: Probe) -> None:
    # Enumerate the model modules from the package DIRECTORY, not by importing the package:
    # torchcell.models/__init__ imports its models, so on a machine without torch_scatter
    # that import is itself the first failure to record (Aurora job 8938786 died here).
    spec = importlib.util.find_spec("torchcell")
    models_dir = Path(spec.submodule_search_locations[0]) / "models"
    model_mods = sorted(m.name for m in pkgutil.iter_modules([str(models_dir)]))

    p.run(
        "import_torch_scatter",
        lambda: importlib.import_module("torch_scatter").__version__,
        core=False,
    )
    p.run(
        "import_torchcell.models",
        lambda: importlib.import_module("torchcell.models").__name__,
        core=False,
    )
    p.run(
        "import_torchcell.data.graph_processor",
        lambda: importlib.import_module("torchcell.data.graph_processor").__name__,
        core=False,
    )
    for mod in model_mods:
        p.run(
            f"import_torchcell.models.{mod}",
            lambda mod=mod: importlib.import_module(f"torchcell.models.{mod}").__name__,
            core=False,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--machine", required=True)
    ap.add_argument("--project", default="torchcell_042_hpc_smoke")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-cpu", action="store_true", help="script validation only")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    import torch_geometric

    device, device_name = pick_device(args.allow_cpu)
    info = {
        "machine": args.machine,
        "hostname": socket.gethostname(),
        "job_id": os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "device": str(device),
        "device_name": device_name,
    }
    print(json.dumps(info, indent=2), flush=True)

    p = Probe(device)
    scatter_checks(p)
    util_checks(p)
    conv_checks(p)
    hetero_check(p)
    autocast_check(p)
    import_checks(p)

    core_fail = [
        k for k, v in p.results.items() if v["status"] == "fail" and v["core"] == "True"
    ]
    other_fail = [
        k
        for k, v in p.results.items()
        if v["status"] == "fail" and v["core"] == "False"
    ]
    result = {
        "info": info,
        "checks": p.results,
        "core_failures": core_fail,
        "import_failures": other_fail,
    }

    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.project,
            name=f"pyg-probe-{args.machine}-{info['job_id'] or 'local'}",
            group=args.machine,
            config=info,
            tags=["042", "pyg-probe", args.machine, device.type],
        )
        run.summary.update({f"probe/{k}": v["status"] for k, v in p.results.items()})
        run.summary.update(
            {"core_failures": len(core_fail), "import_failures": len(other_fail)}
        )
        result["wandb_url"] = run.url
        run.finish()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    n_ok = sum(1 for v in p.results.values() if v["status"] == "ok")
    print(
        f"PROBE {args.machine} {device_name}: {n_ok}/{len(p.results)} ok, "
        f"core failures {core_fail}, import failures {other_fail}",
        flush=True,
    )
    if core_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
