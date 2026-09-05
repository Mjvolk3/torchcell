# experiments/006-kuzmin-tmi/scripts/dcell_training_cpu_profile.py
# [[experiments.006-kuzmin-tmi.scripts.dcell_training_cpu_profile]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/dcell_training_cpu_profile
"""Per-operation CPU profile of one DCell training step (panel e of FigS-dcell-training).

No per-operation trace of the cluster GPU runs exists (``profiler = None`` in
``experiments/006-kuzmin-tmi/scripts/dcell.py``), so this script measures where a training
step spends its time on the CPU of a workstation. The absolute times do not transfer to the
GPU; what transfers is the STRUCTURE of the step: how many ops it launches and what share
of the work is the per-subsystem Python loop, which is what the caption reports.

The model is the trained baseline's exact architecture: ``torchcell.models.dcell.DCell``
built on the frozen filtered GO DAG (``results/dcell_model/go_terms_final.csv``,
``go_edges_final.csv``, ``go_annotations_final.csv``, the DAG that
``dcell_model_go_stats.py`` measured and checked against the wandb-logged parameter
count), with ``subsystem_output_min = 20`` and ``subsystem_output_max_mult = 0.3`` as in
``conf/dcell_kuzmin2018_tmi_mmli_001.yaml``. The batch is synthetic but has exactly the
layout the training script feeds the model (``DCellGraphProcessor`` + ``follow_batch =
["go_gene_strata_state"]``): per strain one copy of the 59,986-row ``(term, gene, stratum,
state)`` table with the rows of three random genes zeroed, a ``go_gene_strata_state_ptr``,
the gene batch vector, and one target. One step is forward, ``DCellLoss`` (auxiliary
losses on, alpha 0.3), backward, gradient clipping at 10, AdamW (lr 1e-3, weight decay
1e-6), profiled with ``torch.profiler`` (CPU activity, record_shapes) after warmup steps.

Phases are exclusive regions of the step: gene-state gather (the per-term, per-sample
indexing in ``_extract_gene_states_for_term``), child-output concatenation (the rest of
``_prepare_term_input``), the subsystem modules (Linear, BatchNorm1d, tanh), the root and
auxiliary heads, the remaining Python overhead of the forward loop, loss, backward,
clipping, optimizer. Regions are attached from outside the model (forward hooks on the
subsystem and head modules, wrappers on the two bound methods), so the model code is
untouched. Op counts are ``aten::`` events in the profiler trace; a leaf op has no
``aten::`` child and is the unit that becomes one kernel launch on a GPU.

Outputs (results/dcell_training/):
  dcell_training_cpu_profile.csv      phase and subphase CPU time of the profiled step at the
                                      headline batch, with torch version, CPU model, threads
  dcell_training_cpu_profile_ops.csv  op counts and step wall-clock per batch size
Panel: $ASSET_IMAGES_DIR/006-kuzmin-tmi/dcell_training_cpu_profile.{svg,png}

Run from the repo root:
    python experiments/006-kuzmin-tmi/scripts/dcell_training_cpu_profile.py            # profile + render
    python experiments/006-kuzmin-tmi/scripts/dcell_training_cpu_profile.py --from-csv # re-render only
"""

import argparse
import os
import os.path as osp
import platform
import subprocess
import sys
import time
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.data import HeteroData

from torchcell.losses.dcell import DCellLoss
from torchcell.models.dcell import DCell
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "svg.fonttype": "none",
        "savefig.bbox": None,
    }
)

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set in the environment"

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
EXP_DIR = osp.dirname(SCRIPT_DIR)
MODEL_RESULTS = osp.join(EXP_DIR, "results", "dcell_model")
RESULTS = osp.join(EXP_DIR, "results", "dcell_training")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "006-kuzmin-tmi")
PROFILE_CSV = osp.join(RESULTS, "dcell_training_cpu_profile.csv")
OPS_CSV = osp.join(RESULTS, "dcell_training_cpu_profile_ops.csv")

# The trained run's model and optimizer settings (conf/dcell_kuzmin2018_tmi_mmli_001.yaml).
SUBSYSTEM_MIN, SUBSYSTEM_RATIO = 20, 0.3
ALPHA, LR, WEIGHT_DECAY, CLIP_NORM = 0.3, 1e-3, 1e-6, 10.0
GENES_PER_STRAIN = 3
HEADLINE_BATCH = 8
BATCH_SWEEP = [2, 8, 32]  # op counts and wall-clock (BatchNorm in train mode needs >= 2); breakdown at HEADLINE_BATCH
WARMUP_STEPS = 2
TIMED_STEPS = 3
SEED = 0

ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]

PHASES = [
    ("gather", "Gene-state gather (per term, per strain)"),
    ("concat", "Child-output concatenation"),
    ("subsystem", "Subsystems: Linear, BatchNorm, tanh"),
    ("heads", "Root and auxiliary heads"),
    ("loop", "Forward loop overhead (Python)"),
    ("loss", "Loss (root + 2,654 auxiliary MSEs)"),
    ("backward", "Backward"),
    ("clip", "Gradient clipping"),
    ("optimizer", "AdamW step and zero_grad"),
]


# ----------------------------------------------------------------------------- the model
def cell_graph_from_frozen_dag() -> HeteroData:
    """Rebuild the ``cell_graph`` fields DCell reads, from the frozen DAG CSVs.

    Mirrors ``torchcell.data.cell_data._process_gene_ontology``: terms indexed in sorted
    order, genes indexed in sorted order over the 6,607-gene reference, hierarchy edges
    child -> parent, strata from the frozen table (``compute_strata`` output, verified by
    ``dcell_model_go_stats.py --dag-only``), and the unperturbed (term, gene, stratum,
    state) table with state 1.
    """
    terms = pd.read_csv(osp.join(MODEL_RESULTS, "go_terms_final.csv"))
    edges = pd.read_csv(osp.join(MODEL_RESULTS, "go_edges_final.csv"))
    ann = pd.read_csv(osp.join(MODEL_RESULTS, "go_annotations_final.csv"))
    genes = pd.read_csv(osp.join(MODEL_RESULTS, "go_genes_final.csv"))
    go_nodes = sorted(terms["term"])
    go_idx = {t: i for i, t in enumerate(go_nodes)}
    gene_nodes = sorted(genes["gene"])
    gene_idx = {g: i for i, g in enumerate(gene_nodes)}
    stratum = dict(zip(terms["term"], terms["stratum"]))

    hd = HeteroData()
    hd["gene"].num_nodes = len(gene_nodes)
    hd["gene"].node_ids = gene_nodes
    hd["gene_ontology"].num_nodes = len(go_nodes)
    hd["gene_ontology"].node_ids = go_nodes
    hd["gene_ontology", "is_child_of", "gene_ontology"].edge_index = torch.tensor(
        [[go_idx[c] for c in edges["child"]], [go_idx[p] for p in edges["parent"]]], dtype=torch.long
    )
    counts = torch.zeros(len(go_nodes), dtype=torch.long)
    for t in ann["term"]:
        counts[go_idx[t]] += 1
    hd["gene_ontology"].term_gene_counts = counts
    strata = torch.tensor([stratum[t] for t in go_nodes], dtype=torch.long)
    hd["gene_ontology"].strata = strata
    hd["gene_ontology"].stratum_to_terms = {
        int(s): torch.tensor(sorted(int(i) for i in torch.where(strata == s)[0]), dtype=torch.long)
        for s in strata.unique().tolist()
    }
    state = torch.zeros((len(ann), 4), dtype=torch.float)
    state[:, 0] = torch.tensor([go_idx[t] for t in ann["term"]], dtype=torch.float)
    state[:, 1] = torch.tensor([gene_idx[g] for g in ann["gene"]], dtype=torch.float)
    state[:, 2] = torch.tensor([stratum[t] for t in ann["term"]], dtype=torch.float)
    state[:, 3] = 1.0
    hd["gene_ontology"].go_gene_strata_state = state
    return hd


def synthetic_batch(cell_graph: HeteroData, batch_size: int, gen: torch.Generator) -> HeteroData:
    """A batch with the training script's layout: per strain one copy of the state table
    with the rows of ``GENES_PER_STRAIN`` random genes zeroed, the ptr, the gene batch
    vector, and one target per strain."""
    base = cell_graph["gene_ontology"].go_gene_strata_state
    n_rows, n_genes = base.shape[0], cell_graph["gene"].num_nodes
    copies = []
    for _ in range(batch_size):
        s = base.clone()
        deleted = torch.randperm(n_genes, generator=gen)[:GENES_PER_STRAIN]
        s[torch.isin(s[:, 1].long(), deleted), 3] = 0.0
        copies.append(s)
    b = HeteroData()
    b["gene"].num_nodes = batch_size * n_genes
    b["gene"].x = torch.zeros((batch_size * n_genes, 1))
    b["gene"].batch = torch.arange(batch_size).repeat_interleave(n_genes)
    b["gene"].phenotype_values = 0.05 * torch.randn(batch_size, generator=gen)
    b["gene_ontology"].go_gene_strata_state = torch.cat(copies, dim=0)
    b["gene_ontology"].go_gene_strata_state_ptr = torch.arange(0, (batch_size + 1) * n_rows, n_rows)
    return b


def attach_regions(model: DCell) -> None:
    """Profiler regions from outside the model: hooks on the subsystem and head modules,
    wrappers on the two bound methods of the forward loop."""

    def region_hooks(module: torch.nn.Module, name: str) -> None:
        stack: list = []

        def pre(m, inp):
            ctx = record_function(name)
            ctx.__enter__()
            stack.append(ctx)

        def post(m, inp, out):
            stack.pop().__exit__(None, None, None)

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    for m in model.subsystems.values():
        region_hooks(m, "dcell/subsystem")
    for m in model.linear_heads.values():
        region_hooks(m, "dcell/head")

    prepare, extract = model._prepare_term_input, model._extract_gene_states_for_term

    def prepare_wrapped(*a, **k):
        with record_function("dcell/prepare_input"):
            return prepare(*a, **k)

    def extract_wrapped(*a, **k):
        with record_function("dcell/gene_state_gather"):
            return extract(*a, **k)

    model._prepare_term_input = prepare_wrapped
    model._extract_gene_states_for_term = extract_wrapped


def train_step(model, loss_fn, opt, batch) -> None:
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


# ----------------------------------------------------------------------------- the profile
def region_totals(prof) -> dict[str, float]:
    """Inclusive CPU time (ms) summed over every event of each record_function region."""
    return {ev.key: ev.cpu_time_total / 1e3 for ev in prof.key_averages() if ev.key.startswith(("step/", "dcell/"))}


def subsystem_split(prof) -> dict[str, float]:
    """Inside the subsystem regions: inclusive ms of the Linear, BatchNorm and tanh calls."""
    out = {"aten::linear": 0.0, "aten::batch_norm": 0.0, "aten::tanh": 0.0}
    for ev in prof.events():
        if ev.name == "dcell/subsystem":
            for ch in ev.cpu_children:
                if ch.name in out:
                    out[ch.name] += ch.time_range.elapsed_us() / 1e3
    return out


def op_counts(prof) -> dict[str, int]:
    """aten:: events per step, total and leaf (no aten:: child), by top-level step region."""
    def step_region(ev) -> str:
        while ev is not None and not ev.name.startswith("step/"):
            ev = ev.cpu_parent
        return ev.name.split("/")[1] if ev is not None else "other"

    counts: dict[str, int] = {"aten_total": 0, "aten_leaf": 0}
    names: set[str] = set()
    for ev in prof.events():
        if not ev.name.startswith("aten::"):
            continue
        counts["aten_total"] += 1
        names.add(ev.name)
        leaf = not any(ch.name.startswith("aten::") for ch in ev.cpu_children)
        if leaf:
            counts["aten_leaf"] += 1
            key = f"leaf_{step_region(ev)}"
            counts[key] = counts.get(key, 0) + 1
    counts["distinct_ops"] = len(names)
    return counts


def cpu_model() -> str:
    if platform.system() == "Darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return platform.processor()


def run_profile() -> tuple[pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    cell_graph = cell_graph_from_frozen_dag()
    model = DCell(cell_graph, min_subsystem_size=SUBSYSTEM_MIN, subsystem_ratio=SUBSYSTEM_RATIO, output_size=1)
    n_params = model.num_parameters
    expected = int(pd.read_csv(osp.join(MODEL_RESULTS, "dcell_model_size.csv"))["params_total"].iloc[0])
    if n_params["total"] != expected:
        raise SystemExit(f"rebuilt DCell has {n_params['total']} parameters, frozen DAG implies {expected}")
    attach_regions(model)
    loss_fn = DCellLoss(alpha=ALPHA, use_auxiliary_losses=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    meta = {
        "torch_version": torch.__version__,
        "cpu_model": cpu_model(),
        "num_threads": torch.get_num_threads(),
        "python_version": platform.python_version(),
        "date": date.today().isoformat(),
        "params_total": n_params["total"],
        "subsystems": n_params["num_subsystems"],
        "dtype": "float32",
        "device": "cpu",
    }
    print(pd.Series(meta).to_string())

    ops_rows, phase_df = [], None
    for bs in BATCH_SWEEP:
        batch = synthetic_batch(cell_graph, bs, gen)
        for _ in range(WARMUP_STEPS):
            train_step(model, loss_fn, opt, batch)
        t0 = time.perf_counter()
        for _ in range(TIMED_STEPS):
            train_step(model, loss_fn, opt, batch)
        wall = (time.perf_counter() - t0) / TIMED_STEPS
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            train_step(model, loss_fn, opt, batch)
        wall_prof = time.perf_counter() - t0
        counts = op_counts(prof)
        R = region_totals(prof)
        ops_rows.append(
            {
                "batch_size": bs,
                "state_rows": int(batch["gene_ontology"].go_gene_strata_state.shape[0]),
                "step_wall_s": wall,
                "step_wall_s_profiled": wall_prof,
                "forward_ms": R["step/forward"],
                "backward_ms": R["step/backward"],
                **counts,
                **meta,
            }
        )
        print(f"batch {bs}: {wall:.2f} s/step unprofiled, {wall_prof:.2f} s profiled; "
              f"{counts['aten_total']} aten ops ({counts['aten_leaf']} leaf, {counts['distinct_ops']} distinct)")
        if bs == HEADLINE_BATCH:
            ms = {
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
            total = sum(ms.values())
            step_total = R["step/forward"] + R["step/loss"] + R["step/backward"] + R["step/clip_grad"] + R["step/optimizer"]
            assert abs(total - step_total) < 1e-6, "phases must partition the step"
            rows = [{"phase": k, "subphase": "", "label": lab, "cpu_ms": ms[k], "share": ms[k] / total} for k, lab in PHASES]
            for op, v in subsystem_split(prof).items():
                rows.append({"phase": "subsystem", "subphase": op, "label": op, "cpu_ms": v, "share": v / total})
            phase_df = pd.DataFrame(rows)
            for k, v in {"batch_size": bs, "step_cpu_ms": total, "step_wall_s": wall,
                         "step_wall_s_profiled": wall_prof, **counts, **meta}.items():
                phase_df[k] = v
    ops = pd.DataFrame(ops_rows)
    os.makedirs(RESULTS, exist_ok=True)
    phase_df.to_csv(PROFILE_CSV, index=False)
    ops.to_csv(OPS_CSV, index=False)
    print(phase_df[["label", "cpu_ms", "share"]].to_string())
    return phase_df, ops


# ----------------------------------------------------------------------------- the panel
def box(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_edgecolor("black")


def save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"{name}.svg"))
    fig.savefig(osp.join(IMG_DIR, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"saved {osp.join(IMG_DIR, name)}.{{svg,png}}")


def panel_cpu_profile(prof: pd.DataFrame, ops: pd.DataFrame):
    """Horizontal bars: exclusive CPU time per phase of one profiled step at the headline
    batch, share labels at the bar ends, the environment and op count in the panel."""
    w = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(48)))
    fig.subplots_adjust(left=0.47, right=0.97, bottom=0.18, top=0.82)
    main = prof[prof["subphase"] == ""].reset_index(drop=True)
    y = np.arange(len(main))[::-1]
    colors = [PURPLE if k in ("gather", "concat", "subsystem", "heads", "loop") else GRAY for k in main["phase"]]
    ax.barh(y, main["cpu_ms"] / 1e3, color=colors, edgecolor="black", lw=0.5, height=0.65, zorder=3)
    xmax = main["cpu_ms"].max() / 1e3
    for yi, (_, r) in zip(y, main.iterrows()):
        ax.text(r["cpu_ms"] / 1e3 + 0.02 * xmax, yi, f"{r['share'] * 100:.0f}%", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(main["label"])
    ax.set_xlim(0, xmax * 1.22)
    ax.set_xlabel("CPU time in one training step (s)")
    ax.grid(axis="x", color="#D0D0D0", lw=0.4)
    ax.set_axisbelow(True)
    m = main.iloc[0]
    head = ops[ops["batch_size"] == int(m["batch_size"])].iloc[0]
    # Leaf ops per added strain: the slope of leaf ops against batch over the sweep (the
    # per-strain gather loop is the only batch-dependent op count).
    slope = np.polyfit(ops["batch_size"], ops["aten_leaf"], 1)[0]
    fig.text(
        0.01, 0.985,
        f"CPU profile, not the GPU run: {m['cpu_model']}, torch {m['torch_version']}, {m['dtype']}, batch {int(m['batch_size'])}\n"
        f"{int(head['aten_leaf']):,} leaf ops per step ({int(head['distinct_ops'])} distinct), "
        f"+{slope:,.0f} per added strain (batch {ops['batch_size'].min()} to {ops['batch_size'].max()})\n"
        f"{m['step_cpu_ms'] / 1e3:.1f} s recorded CPU time under the profiler; {m['step_wall_s']:.1f} s per step without it",
        ha="left", va="top", fontsize=6, linespacing=1.15,
    )
    box(ax)
    save(fig, "dcell_training_cpu_profile")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true", help="re-render the panel from the frozen CSVs")
    args = ap.parse_args()
    if args.from_csv:
        prof, ops = pd.read_csv(PROFILE_CSV, keep_default_na=False), pd.read_csv(OPS_CSV)
    else:
        prof, ops = run_profile()
    panel_cpu_profile(prof, ops)


if __name__ == "__main__":
    main()
