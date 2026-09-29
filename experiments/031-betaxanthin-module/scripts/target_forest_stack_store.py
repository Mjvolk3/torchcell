# experiments/031-betaxanthin-module/scripts/target_forest_stack_store.py
# [[experiments.031-betaxanthin-module.scripts.target_forest_stack_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_forest_stack_store

r"""A forest's out-of-fold prediction from the GO neighbor mean as a one-column inductive store.

The GO neighbor mean (``target_graph_feature_store.py``) reads betaxanthin at 0.279 by
random forest on the pinned split, but the transformer's affine slot reads the same
columns at 0.145 (linear) and 0.217 (through 256): the read is nonlinear in the columns
and the slot does not shape it the way trees do (rounds 226 to 228). Stacking hands the
model the forest's read itself. Every gene's row is one number, the forest's prediction
of the standardized betaxanthin target from the gene's neighbor-mean row: OUT OF FOLD
for the training genes (five folds over the trainer's train split, so a training gene's
value never comes from a forest that saw its own label), and from the forest fit on the
whole train split for every other gene of the genome (validation, pinned and unlabeled
alike). The forest is the ladder's (300 trees, square-root features, leaf size selected
on the validation genes), its inputs the base store's columns as stored. INDUCTIVE: the
forest reads genotype features only, so the row exists for any gene with neighbors, and
no pinned gene's label is read anywhere. The store is nonnegative-shifted by the
training minimum so the trainer's ``log1p`` read stays monotone, as the label store is.
Writes ``ess_solve_encodings_<out>.npz`` and ``results/target_forest_stack_store_<out>.json``.
Run under slurm (``gh_target_forest_stack_store.slurm``)::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_forest_stack_store.py \
        --target betaxanthin --base YPD_bx_gonbr --out YPD_bx_gonbrforest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import time

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts"))
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "031-betaxanthin-module", "scripts"))
import train_bx  # noqa: E402
from linear_baselines_bx import TARGETS, open_dataset, read_targets, record_genes  # noqa: E402
from target_store_baselines import CSV_TARGETS, FOREST_TREES, STUDY  # noqa: E402

from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
LEAF_GRID = (1, 3, 5, 10)
FOLDS = 5


class StackProvenance(BaseModel):
    model_config = {"extra": "forbid"}

    method: str
    target: str
    base_store: str
    base_sha256: str
    pinned_split_file: str
    split_seed: int
    n_train_genes: int
    n_val_genes: int
    n_pinned_genes_excluded: int
    folds: int
    n_trees: int
    min_samples_leaf: int
    val_pearson: float
    val_spearman: float
    oof_train_spearman: float
    shift: float
    n_genes: int
    labels_read: str
    python_hash_seed: str
    flux_sha256: str


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=sorted(STUDY))
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-dir", default=TABLE_DIR)
    args = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0"
    started = time.time()
    path = osp.join(args.out_dir, f"ess_solve_encodings_{args.out}.npz")
    assert not osp.exists(path), f"{path} exists; stores are never overwritten"
    spec = CSV_TARGETS.get(args.target) or TARGETS[args.target]
    columns = list(spec["columns"])
    assert len(columns) == 1, "one-column targets only"

    base_path = osp.join(TABLE_DIR, f"ess_solve_encodings_{args.base}.npz")
    z = np.load(base_path, allow_pickle=False)
    base_rows = {str(g): v.astype(np.float64) for g, v in zip(z["genes"], z["v_deletion"], strict=True)}
    base_wild = z["v_wild_type"].astype(np.float64)
    with open(base_path, "rb") as fh:
        base_sha = hashlib.sha256(fh.read()).hexdigest()

    cfg = train_bx.load_config()
    genome = SCerevisiaeGenome(genome_root=osp.join(DATA_ROOT, "data/sgd/genome"), go_root=osp.join(DATA_ROOT, "data/go"), overwrite=False)
    graph = SCerevisiaeGraph(sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"), string_root=osp.join(DATA_ROOT, "data/string"), tflink_root=osp.join(DATA_ROOT, "data/tflink"), genome=genome)
    dataset = open_dataset(spec, genome, graph)
    pinned, _ = train_bx.resolve_pinned_test(dataset, cfg)
    dm = CellDataModule(dataset=dataset, cache_dir=osp.join(DATA_ROOT, spec["dataset_subdir"], "data_module_cache"), split_indices=["phenotype_label_index", "perturbation_count_index"], batch_size=cfg.model.batch_size, random_seed=cfg.data.split_seed, num_workers=0, pin_memory=False, prefetch=False, follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"], pinned_test_indices=pinned)
    dm.setup()
    base = osp.join(DATA_ROOT, spec["dataset_subdir"])
    idx_to_genes = record_genes(dataset)
    with open(osp.join(EXPERIMENT_ROOT, cfg.data.pinned_test_split_file)) as fh:
        pinned_genes = set(json.load(fh)["split"]["test"])
    labeled: dict[str, dict[str, float]] = {}
    for name in ("train", "val"):
        indices = [int(i) for i in getattr(dm, f"{name}_dataset").indices]
        values = read_targets(base, indices, spec["label"], columns)
        rows: dict[str, float] = {}
        for i, v in zip(indices, values, strict=True):
            gs = idx_to_genes.get(i, [])
            if len(gs) == 1 and np.isfinite(v[0]) and gs[0] in base_rows:
                assert gs[0] not in pinned_genes
                rows[gs[0]] = float(v[0])
        labeled[name] = rows
    n_pinned_excluded = len({idx_to_genes[i][0] for i in [int(i) for i in dm.test_dataset.indices] if len(idx_to_genes.get(i, [])) == 1 and idx_to_genes[i][0] in pinned_genes})
    tr_genes = sorted(labeled["train"]); va_genes = sorted(labeled["val"])
    x_tr = np.stack([base_rows[g] for g in tr_genes]); y_tr = np.array([labeled["train"][g] for g in tr_genes])
    x_va = np.stack([base_rows[g] for g in va_genes]); y_va = np.array([labeled["val"][g] for g in va_genes])
    mu, sd = y_tr.mean(), y_tr.std()
    z_tr, z_va = (y_tr - mu) / sd, (y_va - mu) / sd
    print(f"  train {len(tr_genes)} val {len(va_genes)} genes with a label and a base row; pinned excluded {n_pinned_excluded}", flush=True)

    def forest(leaf: int, seed: int = 0) -> RandomForestRegressor:
        return RandomForestRegressor(n_estimators=FOREST_TREES, max_features="sqrt", min_samples_leaf=leaf, n_jobs=args.threads, random_state=seed)

    best_leaf, best_val = None, -np.inf
    for leaf in LEAF_GRID:
        p = forest(leaf).fit(x_tr, z_tr).predict(x_va)
        v = float(pearsonr(p, z_va)[0])
        print(f"  leaf {leaf}: val Pearson {v:+.4f} Spearman {spearmanr(p, z_va)[0]:+.4f}", flush=True)
        if v > best_val:
            best_leaf, best_val = leaf, v
    assert best_leaf is not None
    full = forest(best_leaf).fit(x_tr, z_tr)
    val_pred = full.predict(x_va)
    oof = np.zeros(len(tr_genes))
    for k, (a, b) in enumerate(KFold(FOLDS, shuffle=True, random_state=0).split(x_tr)):
        oof[b] = forest(best_leaf, seed=k).fit(x_tr[a], z_tr[a]).predict(x_tr[b])
    oof_rho = float(spearmanr(oof, z_tr)[0])
    print(f"  selected leaf {best_leaf}; out-of-fold train Spearman {oof_rho:+.4f}; val Spearman {spearmanr(val_pred, z_va)[0]:+.4f}", flush=True)

    genes = sorted(genome.gene_set)
    values = np.zeros(len(genes), dtype=np.float64)
    oof_by_gene = dict(zip(tr_genes, oof, strict=True))
    rest = [g for g in genes if g not in oof_by_gene]
    x_rest = np.stack([base_rows.get(g, base_wild) for g in rest])
    rest_pred = dict(zip(rest, full.predict(x_rest), strict=True))
    for i, g in enumerate(genes):
        values[i] = oof_by_gene[g] if g in oof_by_gene else rest_pred[g]
    shift = float(min(oof.min(), values.min()))
    v_del = (values - shift).astype(np.float32)[:, None]
    wild = np.array([float(full.predict(base_wild[None, :])[0]) - shift], np.float32)
    prov = StackProvenance(
        method="random-forest stacking: out-of-fold prediction for the train genes, full-train prediction for every other gene; written by target_forest_stack_store.py",
        target=args.target, base_store=args.base, base_sha256=base_sha, pinned_split_file=cfg.data.pinned_test_split_file, split_seed=int(cfg.data.split_seed),
        n_train_genes=len(tr_genes), n_val_genes=len(va_genes), n_pinned_genes_excluded=n_pinned_excluded, folds=FOLDS, n_trees=FOREST_TREES, min_samples_leaf=int(best_leaf),
        val_pearson=float(best_val), val_spearman=float(spearmanr(val_pred, z_va)[0]), oof_train_spearman=oof_rho, shift=shift, n_genes=len(genes),
        labels_read="train and validation genes' betaxanthin (validation for the leaf size only); no pinned gene's label", python_hash_seed=os.environ["PYTHONHASHSEED"],
        flux_sha256=hashlib.sha256(v_del.tobytes()).hexdigest())
    np.savez(path, genes=np.array(genes), v_deletion=v_del, v_wild_type=wild, rxn_ids=np.array([f"forest_{args.base}_{columns[0]}"]), provenance=np.array(prov.model_dump_json()))
    out = osp.join(EXPERIMENT_ROOT, STUDY[args.target], "results", f"target_forest_stack_store_{args.out}.json")
    with open(out, "w") as fh:
        json.dump({"generated_by": "experiments/031-betaxanthin-module/scripts/target_forest_stack_store.py", "store": path, "shape": list(v_del.shape), "provenance": prov.model_dump(), "leaf_grid": list(LEAF_GRID), "seconds": time.time() - started}, fh, indent=2)
    print(path); print(out)


if __name__ == "__main__":
    main()
