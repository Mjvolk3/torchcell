# experiments/031-betaxanthin-module/scripts/target_go_models.py
# [[experiments.031-betaxanthin-module.scripts.target_go_models]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_go_models

r"""DCell's ontology hierarchy and a GO multi-hot perceptron on the two targets, on CPU.

The essentiality study read DCell's hierarchy (``dcell_style_baseline.py``: one small
subsystem per GO term wired up the DAG, a shared gene input per term, auxiliary heads) and
a graph convolution over the GO multi-hot as the annotation-only family beside the
transformer. Neither has a target head. This script gives the two targets that family on
the CPU ladder's split, so the reads sit in the same column as ``target_store_baselines.py``:

* ``dcell_style_shared``: the hierarchy and subsystems of the essentiality script, the
  root's state read by a linear head of ``n_out`` outputs (one for betaxanthin, nineteen
  for the amino acids), every term's scalar head an auxiliary regression of the
  standardized target (its mean over observed columns for the amino acids) at
  alpha 0.3, batch normalization over the training rows in every pass;
* ``dcell_published_input``: the same hierarchy with DCell's own gene input, a weight
  vector per (gene, term) annotation pair; under the split by gene a held-out gene's
  columns are never trained, so this arm is the published network's read of an unseen
  gene and the control for what the shared input buys;
* ``go_mlp``: the gene's propagated GO multi-hot through one hidden layer of 256 and a
  linear head, the perceptron the GO graph convolution reduces to on one deletion per
  instance; and the same perceptron over the HiGHS parsimonious store row
  (``store_mlp``), over ProtT5 (``prot_mlp``), and over their concatenations
  (``go_store_mlp``, ``go_prot_mlp``, ``go_prot_store_mlp``), which separate the
  annotation, the sequence and the flux families on one model.

The instance is a genotype: its GO multi-hot is the OR over its deleted genes' rows of the
``go_term_hot`` store, columns with no training support zeroed (the essentiality rule).
Full batch, Adam at 1e-3, a masked squared error on the targets standardized on the
training genes, the epoch selected by a five-epoch moving average of the validation
per-feature Pearson (the ladder's criterion), the pinned test genes scored once at that
epoch: the Spearman (mean over the 19 columns for the amino acids) and, for betaxanthin,
the top-50 enrichment of high producers. Six seeds, and the mean prediction over seeds
scored the same way. Writes ``results/go_models_<target>.json`` in the target's study.
Run from the repo root::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_go_models.py \
        --target betaxanthin --workers 1 --threads 16
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from scipy.stats import spearmanr
from torch import nn

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
for study in ("031-betaxanthin-module", "028-gene-essentiality", "027-betaxanthin-metabolic"):
    sys.path.insert(0, osp.join(EXPERIMENT_ROOT, study, "scripts"))
import train_bx  # noqa: E402
from dcell_style_baseline import (  # noqa: E402
    ALPHA,
    GO_DIR,
    Config,
    DCellStyle,
    Hierarchy,
    set_threads,
    smoothed_argmax,
)
from linear_baselines_bx import (  # noqa: E402
    TARGETS,
    embedding_matrix,
    open_dataset,
    per_feature_pearson,
    read_targets,
    record_genes,
)
from target_store_baselines import STUDY, arm_targets, load_store, top50_enrichment  # noqa: E402

from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

ARMS = ("dcell_style_shared", "dcell_published_input", "go_mlp", "store_mlp", "prot_mlp", "go_store_mlp", "go_prot_mlp", "go_prot_store_mlp")
#: The two hierarchy arms: the shared per-term gene input, and DCell's published input,
#: one weight vector per (gene, term) annotation pair. Under a split by gene the
#: published input's columns for a held-out gene receive no gradient, so that arm reads
#: an unseen gene through its random initial columns; it is the control that says what
#: the shared input buys (2026-09-25, the user's request for DCell proper sized as the
#: essentiality study sized it, width 8 and 20).
DCELL_ARMS = ("dcell_style_shared", "dcell_published_input")
#: The perceptron arms' input blocks: the GO multi-hot (binary), the HiGHS parsimonious
#: store row (log(1+|v|), standardized on the training genes; a gene the store does not
#: cover reads the wild-type row) and ProtT5 (standardized), concatenated.
ARM_INPUTS = {
    "go_mlp": ("go",),
    "store_mlp": ("store",),
    "prot_mlp": ("prot",),
    "go_store_mlp": ("go", "store"),
    "go_prot_mlp": ("go", "prot"),
    "go_prot_store_mlp": ("go", "prot", "store"),
}
STORE_SPEC = {"betaxanthin": "YPD_bx_pfba", "amino_acids": "SM_gem"}
HIDDEN_MLP = 256


class SeedRead(BaseModel):
    """One arm at one seed: the selected epoch and the pinned-test read."""

    model_config = ConfigDict(extra="forbid")
    arm: str
    seed: int
    select_epoch: int
    val_pearson: float
    test_pinned_spearman: float
    test_pinned_top50_enrichment: float | None
    test_all_spearman: float
    n_parameters: int
    seconds: float
    pinned_pred: list[list[float]]


class ArmRead(BaseModel):
    """One arm over its seeds, and the seed ensemble."""

    model_config = ConfigDict(extra="forbid")
    arm: str
    seeds: list[SeedRead]
    single_mean_spearman: float
    single_sd_spearman: float
    single_mean_top50: float | None
    ensemble_spearman: float
    ensemble_top50: float | None


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    columns: list[str]
    n_rows: dict[str, int]
    n_pinned_test: int
    n_go_columns: int
    n_columns_with_train_support: int
    epochs: int
    width: int
    arms: list[ArmRead]
    seconds: float


class TargetDCell(DCellStyle):
    """The essentiality hierarchy with an ``n_out`` head on the root's state."""

    def __init__(self, hierarchy: Hierarchy, membership: torch.Tensor, widths: np.ndarray,
                 order: str, n_out: int, shared: bool = True) -> None:
        super().__init__(hierarchy, membership, widths, order, shared=shared)
        self.out_head = nn.Linear(self.width, n_out)

    def forward_target(self, stat_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(instances x terms auxiliary scalars, instances x n_out prediction)."""
        gene = self.gene_input()
        states: list[torch.Tensor] = []
        for lo, hi, e_lo, e_hi, parent_pos, sources in self.blocks:
            pre = gene[:, lo:hi] + self.bias[lo:hi]
            if e_hi > e_lo:
                below = torch.cat([states[c].index_select(1, pos) for c, pos in sources], dim=1)
                message = torch.einsum("beh,ehk->bek", below, self.edge_weight[e_lo:e_hi])
                pre = pre.index_add(1, parent_pos, message)
            if self.order == "tanh_bn":
                state = self.normalize(torch.tanh(pre), stat_rows, lo, hi)
            else:
                state = torch.tanh(self.normalize(pre, stat_rows, lo, hi))
            states.append(state * cast(torch.Tensor, self.unit_mask)[lo:hi])
        state = torch.cat(states, dim=1)
        aux = (state * self.head_weight[None]).sum(dim=-1) + self.head_bias[None]
        return aux, self.out_head(state[:, -1, :])


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arm: str
    seed: int
    epochs: int
    width: int


# worker state, inherited by fork
_X: np.ndarray | None = None
_Y: np.ndarray | None = None
_ROWS: dict[str, np.ndarray] | None = None
_PINNED: np.ndarray | None = None
_HIER: Hierarchy | None = None
_BLOCKS: dict[str, np.ndarray] | None = None


def masked_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    obs = torch.isfinite(y)
    return ((pred - torch.nan_to_num(y)) ** 2)[obs].mean()


def run_task(task: Task) -> SeedRead:
    assert _X is not None and _Y is not None and _ROWS is not None and _PINNED is not None
    t0 = time.time()
    torch.manual_seed(task.seed)
    x = torch.from_numpy(_X)
    y = torch.from_numpy(_Y)
    rows = {k: torch.from_numpy(v) for k, v in _ROWS.items()}
    n_out = _Y.shape[1]
    y_train = y[rows["train"]]
    mu = torch.nanmean(y_train, dim=0)
    sd = torch.tensor([float(torch.std(y_train[torch.isfinite(y_train[:, j]), j])) for j in range(n_out)])
    y_std = (y - mu) / sd
    if task.arm in DCELL_ARMS:
        assert _HIER is not None
        net: nn.Module = TargetDCell(_HIER, x, _HIER.widths(Config(width=task.width)), "tanh_bn", n_out,
                                     shared=task.arm == "dcell_style_shared")
    else:
        assert _BLOCKS is not None
        x = torch.from_numpy(np.concatenate([_BLOCKS[b] for b in ARM_INPUTS[task.arm]], axis=1))
        net = nn.Sequential(nn.Linear(x.shape[1], HIDDEN_MLP), nn.ReLU(), nn.Linear(HIDDEN_MLP, n_out))
    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    obs_val = np.isfinite(_Y[_ROWS["val"]])
    val_curve: list[float] = []
    test_preds: list[np.ndarray] = []
    aux_target = torch.nanmean(y_std, dim=1)  # the per-instance mean over observed columns
    for epoch in range(task.epochs + 1):
        if task.arm in DCELL_ARMS:
            aux, out = cast(TargetDCell, net).forward_target(rows["train"])
        else:
            aux, out = None, net(x)
        pred = (out.detach() * sd + mu).numpy()
        val_curve.append(float(per_feature_pearson(pred[_ROWS["val"]], _Y[_ROWS["val"]], obs_val)))
        test_preds.append(pred[_ROWS["test"]].copy())
        if epoch == task.epochs:
            break
        opt.zero_grad()
        loss = masked_mse(out[rows["train"]], y_std[rows["train"]])
        if aux is not None:
            tgt = aux_target[rows["train"]]
            fin = torch.isfinite(tgt)
            loss = loss + ALPHA * ((aux[rows["train"]][fin] - tgt[fin][:, None]) ** 2).mean()
        loss.backward()
        opt.step()
    # The five-epoch moving average returns five values for a shorter curve (a smoke
    # test), so the window is used only once the curve is at least as long as it.
    e = smoothed_argmax(val_curve) if len(val_curve) >= 5 else int(np.argmax(val_curve))
    assert len(test_preds) == len(val_curve) == task.epochs + 1 and 0 <= e < len(test_preds), (
        len(test_preds), len(val_curve), e, val_curve[:5])
    pinned = test_preds[e][_PINNED]
    truth = _Y[_ROWS["test"]][_PINNED]
    rhos = []
    for j in range(n_out):
        fin = np.isfinite(truth[:, j])
        rhos.append(float(spearmanr(pinned[fin, j], truth[fin, j]).correlation))
    all_rhos = []
    truth_all = _Y[_ROWS["test"]]
    for j in range(n_out):
        fin = np.isfinite(truth_all[:, j])
        all_rhos.append(float(spearmanr(test_preds[e][fin, j], truth_all[fin, j]).correlation))
    top50 = top50_enrichment(pinned[:, 0], truth[:, 0])[0] if n_out == 1 else None
    read = SeedRead(
        arm=task.arm,
        seed=task.seed,
        select_epoch=e,
        val_pearson=val_curve[e],
        test_pinned_spearman=float(np.mean(rhos)),
        test_pinned_top50_enrichment=top50,
        test_all_spearman=float(np.mean(all_rhos)),
        n_parameters=sum(p.numel() for p in params),
        seconds=time.time() - t0,
        pinned_pred=pinned.tolist(),
    )
    print(
        f"{read.arm:20s} s{read.seed} sel {e:3d} val {read.val_pearson:+.3f} pinned "
        f"{read.test_pinned_spearman:+.4f}" + (f" top50 {top50:.2f}" if top50 is not None else "")
        + f" ({read.seconds:.0f}s)",
        flush=True,
    )
    return read


def main() -> None:
    global _X, _Y, _ROWS, _PINNED, _HIER, _BLOCKS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=sorted(STUDY))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(2700, 2706)))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--width", type=int, default=8)
    # MEASURED 2026-09-24: a forked pool of two workers sat idle for 77 min with no cell
    # finished (the parent had initialized torch's thread pool before forking, the
    # deadlock the fork warning names); in-process with 16 threads a DCell cell takes
    # minutes, so one worker is the default.
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    started = time.time()
    torch.set_num_threads(args.threads)
    spec = TARGETS[args.target]
    columns = spec["columns"]

    cfg = train_bx.load_config()
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    dataset = open_dataset(spec, genome, graph)
    pinned_idx, _ = train_bx.resolve_pinned_test(dataset, cfg)
    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, spec["dataset_subdir"], "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cfg.model.batch_size,
        random_seed=cfg.data.split_seed,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned_idx,
    )
    dm.setup()
    base = osp.join(DATA_ROOT, spec["dataset_subdir"])
    splits = {
        name: [int(i) for i in getattr(dm, f"{name}_dataset").indices]
        for name in ("train", "val", "test")
    }
    idx_to_genes = record_genes(dataset)
    with open(osp.join(EXPERIMENT_ROOT, cfg.data.pinned_test_split_file)) as fh:
        pinned_genes = set(json.load(fh)["split"]["test"])
    hierarchy = Hierarchy(GO_DIR)
    gene_index = {g: i for i, g in enumerate(hierarchy.gene_ids)}
    ys: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    rows: dict[str, list[int]] = {}
    pinned_mask: list[bool] = []
    perts_test: list[list[str]] = []
    for name, indices in splits.items():
        values = read_targets(base, indices, spec["label"], columns)
        genes = [idx_to_genes.get(i, []) for i in indices]
        has = np.isfinite(values).any(axis=1) & np.array([len(g) > 0 for g in genes])
        rows[name] = []
        for i in np.where(has)[0]:
            gs = genes[i]
            hit = [gene_index[g] for g in gs if g in gene_index]
            row = hierarchy.membership[hit].max(axis=0) if hit else np.zeros(hierarchy.membership.shape[1], np.float32)
            rows[name].append(len(xs))
            xs.append(row)
            ys.append(values[i])
            if name == "test":
                pinned_mask.append(any(g in pinned_genes for g in gs))
                perts_test.append(list(gs))
        print(f"  {name}: {int(has.sum())} genotypes with a measurement", flush=True)
    x = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.float32)
    support = x[rows["train"]].sum(axis=0) > 0
    x[:, ~support] = 0.0
    pinned = np.array(pinned_mask)
    # The identity check the ladder makes, against the study's reference arm.
    theirs = arm_targets(args.target)
    test_y = y[rows["test"]]
    shared = 0
    for i, gs in enumerate(perts_test):
        if pinned[i] and len(gs) == 1 and gs[0] in theirs:
            theirs_vec = np.array([np.nan if v is None else float(v) for v in np.atleast_1d(theirs[gs[0]])])
            assert np.allclose(test_y[i], theirs_vec, equal_nan=True, atol=1e-5), gs[0]
            shared += 1
    assert shared >= 500, shared
    print(f"  identity check: {shared} pinned targets match the dump; {int(support.sum())} GO columns with train support", flush=True)
    _X, _Y, _HIER, _PINNED = x, y, hierarchy, pinned
    _ROWS = {k: np.array(v) for k, v in rows.items()}
    # The perceptron arms' other blocks, per genotype: the store row of the deleted gene
    # (the first gene the store covers, else the wild-type row) and ProtT5 (the mean over
    # the genotype's genes), each standardized on the training rows.
    all_genes = [gs for name in ("train", "val", "test") for gs in
                 [idx_to_genes.get(i, []) for i in splits[name]]]
    kept_genes = []
    for name, indices in splits.items():
        values = read_targets(base, indices, spec["label"], columns)
        genes = [idx_to_genes.get(i, []) for i in indices]
        has = np.isfinite(values).any(axis=1) & np.array([len(g) > 0 for g in genes])
        kept_genes.extend(genes[i] for i in np.where(has)[0])
    del all_genes
    store_rows, store_wild, _ = load_store({"store": STORE_SPEC[args.target], "block": "v_deletion", "flux": True})
    store = np.stack([next((store_rows[g] for g in gs if g in store_rows), store_wild) for gs in kept_genes]).astype(np.float32)
    emb = embedding_matrix(("prot_T5_all",), DATA_ROOT, genome, graph)
    dim_emb = len(next(iter(emb.values())))
    prot = np.stack([np.mean([emb[g] for g in gs if g in emb], axis=0) if any(g in emb for g in gs) else np.zeros(dim_emb)
                     for gs in kept_genes]).astype(np.float32)

    def standardized(block: np.ndarray) -> np.ndarray:
        tr = block[_ROWS["train"]]
        m, sd = tr.mean(axis=0), tr.std(axis=0)
        sd[sd == 0] = 1.0
        return ((block - m) / sd).astype(np.float32)

    _BLOCKS = {"go": x, "store": standardized(store), "prot": standardized(prot)}
    print(f"  blocks: go {x.shape[1]}, store {store.shape[1]}, prot {prot.shape[1]}", flush=True)
    tasks = [Task(arm=a, seed=s, epochs=args.epochs, width=args.width) for a in args.arms for s in args.seeds]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, initializer=set_threads, initargs=(args.threads,)) as pool:
            reads = list(pool.map(run_task, tasks))
    else:
        reads = [run_task(t) for t in tasks]
    truth = y[_ROWS["test"]][pinned]
    arms: list[ArmRead] = []
    for arm in args.arms:
        seeds = [r for r in reads if r.arm == arm]
        mean_pred = np.mean([np.array(r.pinned_pred) for r in seeds], axis=0)
        rhos = []
        for j in range(y.shape[1]):
            fin = np.isfinite(truth[:, j])
            rhos.append(float(spearmanr(mean_pred[fin, j], truth[fin, j]).correlation))
        top50s = [r.test_pinned_top50_enrichment for r in seeds if r.test_pinned_top50_enrichment is not None]
        arms.append(
            ArmRead(
                arm=arm,
                seeds=seeds,
                single_mean_spearman=float(np.mean([r.test_pinned_spearman for r in seeds])),
                single_sd_spearman=float(np.std([r.test_pinned_spearman for r in seeds], ddof=1)) if len(seeds) > 1 else 0.0,
                single_mean_top50=float(np.mean(top50s)) if top50s else None,
                ensemble_spearman=float(np.mean(rhos)),
                ensemble_top50=top50_enrichment(mean_pred[:, 0], truth[:, 0])[0] if y.shape[1] == 1 else None,
            )
        )
        print(
            f"{arm}: single {arms[-1].single_mean_spearman:+.4f} (sd {arms[-1].single_sd_spearman:.4f})"
            + (f" top50 {arms[-1].single_mean_top50:.2f}" if arms[-1].single_mean_top50 is not None else "")
            + f"; ensemble {arms[-1].ensemble_spearman:+.4f}"
            + (f" top50 {arms[-1].ensemble_top50:.2f}" if arms[-1].ensemble_top50 is not None else ""),
            flush=True,
        )
    report = Report(
        target=args.target,
        columns=list(columns),
        n_rows={k: len(v) for k, v in rows.items()},
        n_pinned_test=int(pinned.sum()),
        n_go_columns=int(x.shape[1]),
        n_columns_with_train_support=int(support.sum()),
        epochs=args.epochs,
        width=args.width,
        arms=arms,
        seconds=time.time() - started,
    )
    out = args.out or osp.join(EXPERIMENT_ROOT, STUDY[args.target], "results", f"go_models_{args.target}.json")
    # Arms already in the file and not rerun are kept, so a cheap arm can be added
    # without retraining the hierarchy; a rerun arm replaces its earlier read.
    if osp.exists(out):
        with open(out) as fh:
            previous = Report.model_validate_json(fh.read())
        assert previous.epochs == report.epochs and previous.width == report.width, "settings differ from the file's"
        kept = [a for a in previous.arms if a.arm not in {b.arm for b in report.arms}]
        order = {a: i for i, a in enumerate(ARMS)}
        report.arms = sorted(kept + report.arms, key=lambda a: order[a.arm])
        report.seconds += previous.seconds
    with open(out, "w") as fh:
        fh.write(report.model_dump_json(indent=2))
    print(f"wrote {out} in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
