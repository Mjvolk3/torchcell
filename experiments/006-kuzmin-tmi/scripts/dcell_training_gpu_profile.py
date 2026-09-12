# experiments/006-kuzmin-tmi/scripts/dcell_training_gpu_profile.py
# [[experiments.006-kuzmin-tmi.scripts.dcell_training_gpu_profile]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/dcell_training_gpu_profile
"""Per-operation GPU profile of one DCell training step (panel e of FigS-dcell-training).

The cluster run behind Fig. 2d (wandb ``eni948by``, four GPUs, batch 600 per GPU,
bf16-mixed) was never profiled (``profiler = None`` in ``dcell.py``), and the CPU stand-in
of ``dcell_training_cpu_profile.py`` cannot separate the two hypotheses for where its
72 s per optimizer step go: kernel launches from the per-subsystem Python loop, or data
loading. This script measures one training step of the same architecture on one GPU of
the gilahyper workstation, under ``torch.profiler`` with CPU and CUDA activities, at the
cluster's per-GPU batch of 600 and the same precision (``torch.autocast`` bf16 on the
forward and loss, fp32 parameters and optimizer, as Lightning's ``bf16-mixed``).

Everything else is shared with the CPU script and imported from it: the model is the
trained baseline's exact architecture rebuilt on the frozen filtered GO DAG
(``results/dcell_model/``; parameter count checked against ``dcell_model_size.csv``), the
batch is synthetic with the training script's exact layout (per strain one copy of the
59,986-row state table with three random genes zeroed), and one step is forward,
``DCellLoss`` (auxiliary losses on), backward, gradient clipping at 10, AdamW. The batch
is built once and lives on the GPU, so the step contains NO data loading: what is
measured is the compute side of the step alone. If the wall-clock of that step is far
below the cluster's 72 s, the remainder is loading, collation and DDP; if it matches, the
step itself is the cost.

Two quantities per phase (the CPU script's exclusive regions: gene-state gather,
child-output concatenation, subsystem modules, heads, forward loop overhead, loss,
backward, clipping, optimizer):
  host_ms    CPU-side time inside the region: Python plus the cost of issuing kernels
  device_ms  GPU time of the kernels the region launched (the profiler's inclusive
             device time of the region)
and per step: kernel launches (``cudaLaunchKernel`` events), ``aten::`` leaf ops, the
wall-clock with the GPU synchronized, and the fraction of that wall-clock the GPU spent
running kernels. A step whose device time is a small fraction of its wall-clock and whose
host time tracks the launch count is launch-bound.

``--checkpoint`` loads the state dict of a saved checkpoint of the cluster run into the
rebuilt model with ``strict=True`` before profiling; it is a check that the profiled
architecture is the trained one (every parameter name and shape must match), not a change
to the measurement.

Outputs (results/dcell_training/):
  dcell_training_gpu_profile.csv      host and device time per phase at the headline batch,
                                      with GPU model, torch version, precision, launches
  dcell_training_gpu_profile_ops.csv  launches, leaf ops, wall-clock and device time per
                                      batch size of the sweep
Panel: $ASSET_IMAGES_DIR/006-kuzmin-tmi/dcell_training_gpu_profile.{svg,png}

Run from the repo root on a machine with one free GPU (``gh_dcell_training_gpu_profile.slurm``):
    python experiments/006-kuzmin-tmi/scripts/dcell_training_gpu_profile.py [--checkpoint PATH]
    python experiments/006-kuzmin-tmi/scripts/dcell_training_gpu_profile.py --from-csv
"""

import argparse
import os
import os.path as osp
import platform
import sys
import time
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from dcell_training_cpu_profile import (  # noqa: E402
    ALPHA,
    CLIP_NORM,
    GRAY,
    LR,
    MODEL_RESULTS,
    PHASES,
    PURPLE,
    RESULTS,
    SEED,
    SUBSYSTEM_MIN,
    SUBSYSTEM_RATIO,
    WEIGHT_DECAY,
    attach_regions,
    box,
    cell_graph_from_frozen_dag,
    save,
    synthetic_batch,
)
from dcell_training_cpu_profile import BATCH_SWEEP as CPU_BATCH_SWEEP

from torchcell.losses.dcell import DCellLoss  # noqa: E402
from torchcell.models.dcell import DCell  # noqa: E402
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in  # noqa: E402

PROFILE_CSV = osp.join(RESULTS, "dcell_training_gpu_profile.csv")
OPS_CSV = osp.join(RESULTS, "dcell_training_gpu_profile_ops.csv")

# The cluster run's per-GPU batch (conf/dcell_kuzmin2018_tmi_mmli_001.yaml: batch_size 600,
# precision bf16-mixed). The sweep spans the CPU script's sizes up to the cluster batch so
# the launch count's slope in the batch can be compared with the CPU measurement.
HEADLINE_BATCH = 600
BATCH_SWEEP = [8, 64, 256, 600]
assert CPU_BATCH_SWEEP[1] == BATCH_SWEEP[0], "the sweeps share batch 8 so the two profiles can be joined"
WARMUP_STEPS = 2
TIMED_STEPS = 3
DARK_PURPLE, DARK_GRAY = PLOT_PALETTE[8], PLOT_PALETTE[11]
FORWARD_PHASES = ("gather", "concat", "subsystem", "heads", "loop")


def train_step(model, loss_fn, opt, batch) -> None:
    """One optimizer step as Lightning runs it under ``precision: bf16-mixed``: autocast
    on the forward and the loss, backward and clipping on fp32 gradients, AdamW on fp32
    parameters. No scaler: bf16 has fp32's exponent range.
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with record_function("step/forward"):
            pred, out = model(None, batch)
        with record_function("step/loss"):
            loss, _ = loss_fn(pred, out, batch["gene"].phenotype_values)
    with record_function("step/backward"):
        loss.backward()
    with record_function("step/clip_grad"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    with record_function("step/optimizer"):
        opt.step()
        opt.zero_grad(set_to_none=True)


def load_checkpoint_into(model: DCell, path: str) -> dict:
    """Load a Lightning checkpoint's model weights into the rebuilt DCell, strictly."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    prefix = "model."
    stripped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not stripped:
        raise SystemExit(f"no keys with prefix {prefix!r} in {path}; keys start with {sorted(sd)[:3]}")
    # The DCell that trained the run carried one extra registered tensor, ``dummy``, a
    # placeholder parameter with no role in the forward pass that the current model no
    # longer declares. It is the only key outside the architecture; every subsystem, head
    # and BatchNorm tensor must match by name and shape, which strict loading enforces.
    dropped = sorted(k for k in stripped if k == "dummy")
    for k in dropped:
        del stripped[k]
    model.load_state_dict(stripped, strict=True)
    return {"checkpoint": osp.basename(path), "checkpoint_epoch": int(ckpt["epoch"]),
            "checkpoint_global_step": int(ckpt["global_step"]), "checkpoint_tensors": len(stripped),
            "checkpoint_dropped_keys": ",".join(dropped)}


# ----------------------------------------------------------------------------- the profile
REGION_PREFIXES = ("step/", "dcell/")


def region_totals(prof) -> tuple[dict[str, float], dict[str, float]]:
    """Inclusive host (CPU) and device (CUDA kernel) time in ms per record_function region.

    With CUDA activity on, the profiler reports every ``record_function`` range twice under
    the same key: the CPU-side range, and a GPU-side ``gpu_user_annotation`` spanning the
    kernels it launched. Only the CPU-side entry is read: its ``cpu_time_total`` is the host
    time inside the region and its ``device_time_total`` the kernel time attributed to it.
    """
    host, device = {}, {}
    for ev in prof.key_averages():
        if ev.device_type.name == "CPU" and ev.key.startswith(REGION_PREFIXES):
            host[ev.key] = ev.cpu_time_total / 1e3
            device[ev.key] = ev.device_time_total / 1e3
    return host, device


def kernel_time_total_ms(prof) -> float:
    """Total GPU kernel time of the step: the device-side events that are kernels (or
    memcpy/memset), excluding the GPU-side copies of the region annotations.
    """
    return sum(
        ev.self_device_time_total
        for ev in prof.key_averages()
        if ev.device_type.name == "CUDA" and not ev.key.startswith(REGION_PREFIXES)
    ) / 1e3


def op_counts(prof) -> dict[str, int]:
    """Per step: kernel launches and ``aten::`` leaf ops, total and by top-level region."""

    def step_region(ev) -> str:
        while ev is not None and not ev.name.startswith("step/"):
            ev = ev.cpu_parent
        return ev.name.split("/")[1] if ev is not None else "other"

    counts: dict[str, int] = {"launches": 0, "aten_total": 0, "aten_leaf": 0}
    names: set[str] = set()
    for ev in prof.events():
        if ev.device_type.name != "CPU":
            continue
        if ev.name.startswith("cudaLaunchKernel"):
            counts["launches"] += 1
            key = f"launches_{step_region(ev)}"
            counts[key] = counts.get(key, 0) + 1
            continue
        if not ev.name.startswith("aten::"):
            continue
        counts["aten_total"] += 1
        names.add(ev.name)
        if not any(ch.name.startswith("aten::") for ch in ev.cpu_children):
            counts["aten_leaf"] += 1
            key = f"leaf_{step_region(ev)}"
            counts[key] = counts.get(key, 0) + 1
    counts["distinct_ops"] = len(names)
    return counts


def timed_steps(model, loss_fn, opt, batch, n: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        train_step(model, loss_fn, opt, batch)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def run_profile(checkpoint: str | None, batches: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    headline = batches[-1]
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this script measures the GPU step")
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    cell_graph = cell_graph_from_frozen_dag()
    model = DCell(cell_graph, min_subsystem_size=SUBSYSTEM_MIN, subsystem_ratio=SUBSYSTEM_RATIO, output_size=1)
    n_params = model.num_parameters
    expected = int(pd.read_csv(osp.join(MODEL_RESULTS, "dcell_model_size.csv"))["params_total"].iloc[0])
    if n_params["total"] != expected:
        raise SystemExit(f"rebuilt DCell has {n_params['total']} parameters, frozen DAG implies {expected}")
    ckpt_meta = load_checkpoint_into(model, checkpoint) if checkpoint else {}
    model = model.to(device)
    attach_regions(model)
    loss_fn = DCellLoss(alpha=ALPHA, use_auxiliary_losses=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    meta = {
        "torch_version": torch.__version__,
        "gpu_model": torch.cuda.get_device_name(device),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 1),
        "cuda_version": torch.version.cuda,
        "host": platform.node(),
        "python_version": platform.python_version(),
        "date": date.today().isoformat(),
        "params_total": n_params["total"],
        "subsystems": n_params["num_subsystems"],
        "precision": "bf16-mixed (autocast bf16 forward and loss, fp32 parameters)",
        "device": "cuda",
        **ckpt_meta,
    }
    print(pd.Series(meta).to_string(), flush=True)

    ops_rows, phase_df = [], None
    for bs in batches:
        batch = synthetic_batch(cell_graph, bs, gen).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(WARMUP_STEPS):
            train_step(model, loss_fn, opt, batch)
        wall = timed_steps(model, loss_fn, opt, batch, TIMED_STEPS)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            train_step(model, loss_fn, opt, batch)
            torch.cuda.synchronize()
        wall_prof = time.perf_counter() - t0
        t0 = time.perf_counter()
        counts = op_counts(prof)
        host, dev = region_totals(prof)
        kernel_ms = kernel_time_total_ms(prof)
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        row = {
            "batch_size": bs,
            "state_rows": int(batch["gene_ontology"].go_gene_strata_state.shape[0]),
            "step_wall_s": wall,
            "step_wall_s_profiled": wall_prof,
            "kernel_time_s": kernel_ms / 1e3,
            "kernel_busy_frac": kernel_ms / 1e3 / wall,
            "forward_host_ms": host["step/forward"],
            "forward_device_ms": dev["step/forward"],
            "backward_host_ms": host["step/backward"],
            "backward_device_ms": dev["step/backward"],
            "peak_memory_gb": peak_gb,
            **counts,
            **meta,
        }
        ops_rows.append(row)
        print(
            f"batch {bs}: {wall:.2f} s/step ({wall_prof:.1f} s profiled, {time.perf_counter() - t0:.0f} s to parse); "
            f"{counts['launches']:,} launches, {counts['aten_leaf']:,} leaf ops; kernels busy "
            f"{kernel_ms / 1e3:.2f} s = {100 * kernel_ms / 1e3 / wall:.0f}% of the step; peak {peak_gb:.1f} GB",
            flush=True,
        )
        if bs == headline:
            def split(R):
                return {
                    "gather": R["dcell/gene_state_gather"],
                    "concat": R["dcell/prepare_input"] - R["dcell/gene_state_gather"],
                    "subsystem": R["dcell/subsystem"],
                    "heads": R["dcell/head"],
                    "loop": R["step/forward"] - R["dcell/prepare_input"] - R["dcell/subsystem"] - R["dcell/head"],
                    "loss": R["step/loss"],
                    "backward": R["step/backward"],
                    "clip": R["step/clip_grad"],
                    "optimizer": R["step/optimizer"],
                }
            h, d = split(host), split(dev)
            step_host = sum(host[k] for k in ("step/forward", "step/loss", "step/backward", "step/clip_grad", "step/optimizer"))
            assert abs(sum(h.values()) - step_host) < 1e-6, "host phases must partition the step"
            # Backward kernels are launched by the autograd engine's own thread, outside the
            # CPU-side ``step/backward`` range, so the profiler attributes them to no region.
            # They are the only kernels launched outside a region in the step, so backward
            # takes the kernel time the other regions do not account for.
            attributed = sum(v for k, v in d.items() if k != "backward")
            d["backward"] = kernel_ms - attributed
            assert d["backward"] >= -1e-6, (kernel_ms, attributed)
            rows = [
                {"phase": k, "label": lab, "host_ms": h[k], "device_ms": d[k],
                 "host_share": h[k] / step_host, "device_share": d[k] / kernel_ms}
                for k, lab in PHASES
            ]
            phase_df = pd.DataFrame(rows)
            for k, v in {**{c: row[c] for c in row if c not in counts or c in ("launches", "aten_leaf", "distinct_ops")},
                         "step_host_ms": step_host, "step_device_ms": sum(d.values())}.items():
                phase_df[k] = v
        del prof, batch
        torch.cuda.empty_cache()
    ops = pd.DataFrame(ops_rows)
    os.makedirs(RESULTS, exist_ok=True)
    phase_df.to_csv(PROFILE_CSV, index=False)
    ops.to_csv(OPS_CSV, index=False)
    print(phase_df[["label", "host_ms", "device_ms", "host_share", "device_share"]].to_string())
    return phase_df, ops


# ----------------------------------------------------------------------------- the panel
PHASE_LABELS = {
    "gather": "Forward: gene-state gather",
    "concat": "Forward: child-output concatenation",
    "subsystem": "Forward: subsystem layers",
    "heads": "Forward: root and auxiliary heads",
    "loop": "Forward: Python loop overhead",
    "loss": "Loss (root + 2,654 auxiliary MSEs)",
    "backward": "Backward",
    "clip": "Gradient clipping",
    "optimizer": "AdamW step",
}


def panel_gpu_profile(prof: pd.DataFrame, ops: pd.DataFrame):
    """Horizontal bars per phase of one step at the cluster batch: share of the step's
    host time (light) with the share of GPU kernel time overlaid (dark). Shares, not
    seconds: CUDA tracing slows the traced step several-fold. The environment, launch
    count and wall-clock are in the figure caption, read from the frozen CSVs.
    """
    w = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(48)))
    fig.subplots_adjust(left=0.44, right=0.97, bottom=0.18, top=0.90)
    y = np.arange(len(prof))[::-1]
    light = [PURPLE if k in FORWARD_PHASES else GRAY for k in prof["phase"]]
    dark = [DARK_PURPLE if k in FORWARD_PHASES else DARK_GRAY for k in prof["phase"]]
    ax.barh(y, 100 * prof["host_share"], color=light, edgecolor="black", lw=0.5, height=0.65, zorder=3)
    ax.barh(y, 100 * prof["device_share"], color=dark, edgecolor="black", lw=0.5, height=0.32, zorder=4)
    for yi, (_, r) in zip(y, prof.iterrows()):
        ax.text(100 * max(r["host_share"], r["device_share"]) + 2, yi,
                f"{100 * r['host_share']:.0f}% | {100 * r['device_share']:.0f}%", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([PHASE_LABELS[k] for k in prof["phase"]])
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Share of one training step (%): host | GPU kernels")
    m = prof.iloc[0]
    ax.set_title(f"One training step on one GPU, batch {int(m['batch_size'])}", fontsize=6, pad=3)
    ax.grid(axis="x", color="#D0D0D0", lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, fc=PURPLE, ec="black", lw=0.5), plt.Rectangle((0, 0), 1, 1, fc=DARK_PURPLE, ec="black", lw=0.5)],
        labels=["host: Python + kernel launches", "GPU: kernels running"],
        loc="lower right", frameon=True, fontsize=6, handlelength=1.0, handleheight=0.8, borderaxespad=0.3,
    )
    box(ax)
    save(fig, "dcell_training_gpu_profile")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true", help="re-render the panel from the frozen CSVs")
    ap.add_argument("--checkpoint", default=None, help="Lightning .ckpt of the cluster run; loaded strictly as an architecture check")
    ap.add_argument("--batches", default=",".join(map(str, BATCH_SWEEP)),
                    help="comma-separated batch sizes; the last is the headline (default the full sweep)")
    args = ap.parse_args()
    if args.from_csv:
        prof, ops = pd.read_csv(PROFILE_CSV), pd.read_csv(OPS_CSV)
    else:
        prof, ops = run_profile(args.checkpoint, [int(b) for b in args.batches.split(",")])
    panel_gpu_profile(prof, ops)


if __name__ == "__main__":
    main()
