# experiments/010-kuzmin-tmi/scripts/dango_full_dataset_inference.py
# [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_inference]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/dango_full_dataset_inference

"""Score the served DANGO checkpoints on the experiment-006 validation and test splits.

Fills panels e, f, g of FigS-dango-full-dataset
(``paper/nature-biotech/sections/si-note-dango-full.tex``). One-shot producer of
frozen result files in ``experiments/010-kuzmin-tmi/results/``.

The 006 LMDB is not read (the current schema may refuse it, see
``score_010_checkpoints_directly.py``). Records come from the build's side files:
``label_df.parquet`` (record id, gene_interaction), ``is_any_perturbed_gene_index.json``
(perturbed genes per record, exactly three each) and the seed-42 split cache. The
model's gene index space is ``sorted(genome.gene_set)``, because the 006 training script
built ``Neo4jCellDataset(gene_set=genome.gene_set)`` and ``to_cell_data`` orders nodes
by the sorted base graph. The six STRING channels are rebuilt with
``build_gene_multigraph`` and ``to_cell_data`` as the dataset did. The checkpoints carry
no label transform, so predictions are in raw interaction units.

Correctness is checked, not assumed: the validation Pearson and MSE are recomputed and
asserted against the values the training run logged at the checkpoint's own epoch
(``results/dango_full_dataset_history.csv``, DDP-synced torchmetrics). A wrong gene
order or a wrong edge set fails that check rather than producing a plausible number.

ReZero ablation. ``HyperSAGNN`` applies two attention layers ``x <- beta_l * O_l(attn(x)) + x``
to the integrated embeddings ``e_g`` of the perturbed genes, then scores each gene by
``w . (dyn_g - ReLU(W_s e_g + b_s))^2 + b`` and averages over the genes of a genotype.
With ``beta1 = beta2 = 0`` both layers return their input, so ``dyn_g = e_g`` and the
prediction is the mean of three per-gene scores ``s(g)``, with no term that depends on
two genes jointly. The script asserts this by scoring every gene alone (a singleton set)
and comparing the triple prediction with the mean of its singleton scores. The size of
the residual branch is reported per layer as ``||beta_l O_l(.)||_F / ||x_l||_F`` over the
3 x 64 block of each record (``x_l`` is the layer input), read through forward hooks on
``Q_l`` and ``O_l``.

Meta-embedding channel weights are the softmax over channels of
``meta_embedding.attention_mlp`` applied to each channel's embedding, recomputed per gene
and asserted to reproduce ``meta_embedding``'s output. They depend on the gene, not on
the genotype.
"""

import argparse
import copy
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr
from sortedcontainers import SortedDict
from torch_geometric.data import HeteroData

from torchcell.data.cell_data import to_cell_data
from torchcell.data.neo4j_cell import create_graph_from_gene_set
from torchcell.graph import GeneMultiGraph, SCerevisiaeGraph, build_gene_multigraph
from torchcell.models.dango import Dango
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
BUILD_DIR = osp.join(DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
CKPT_ROOT = osp.join(DATA_ROOT, "models", "checkpoints")

CHANNELS = ["neighborhood", "fusion", "cooccurence", "coexpression", "experimental", "database"]
RUNS = {
    "014mprap": {
        "string": "string9_1",
        "dir": "compute-3-3-1941704_ff8554028c424f578dc4f0402ce2e9b0e3835f61c413ac5fd18af2c1f67f2e96",
        "ckpts": {
            "minmse": "014mprap-best-epoch=81-val/gene_interaction/MSE=0.0032.ckpt",
            "last": "014mprap-last.ckpt",
        },
    },
    "9jpfy547": {
        "string": "string12_0",
        "dir": "compute-3-3-1940775_dd2b1fa1419e340a913e53235fd7abc19c4c0ac8367559e94a03a0f02d6283f3",
        "ckpts": {"minmse": "9jpfy547-best-epoch=53-val/gene_interaction/MSE=0.0033.ckpt"},
    },
}
WEIGHT_CKPTS = [("014mprap", "minmse"), ("9jpfy547", "minmse")]
HIDDEN, HEADS = 64, 4  # conf/dango_kuzmin2018_tmi_string9_1.yaml, model block
BATCH = 1024
PEARSON_TOL = 0.003  # numpy over the split vs DDP-synced torchmetrics
MSE_RTOL = 0.01


def load_records() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    """Perturbed gene names per record (as vocabulary positions), labels, splits, vocabulary."""
    label_df = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    with open(osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")) as f:
        gene_index = json.load(f)
    with open(osp.join(BUILD_DIR, "data_module_cache", "index_seed_42.json")) as f:
        split_index = json.load(f)

    record_ids = np.sort(label_df["index"].to_numpy())
    y = label_df.set_index("index").loc[record_ids, "gene_interaction"].to_numpy()
    id_to_row = {int(r): i for i, r in enumerate(record_ids)}

    gene_names = sorted(gene_index.keys())
    row_genes = np.full((len(record_ids), 3), -1, dtype=np.int64)
    fill = np.zeros(len(record_ids), dtype=np.int8)
    for j, gene in enumerate(gene_names):
        for rid in gene_index[gene]:
            row = id_to_row[int(rid)]
            row_genes[row, fill[row]] = j
            fill[row] += 1
    assert (fill == 3).all()
    splits = {
        name: np.array([id_to_row[int(r)] for r in ids], dtype=np.int64)
        for name, ids in split_index.items()
    }
    return row_genes, y, splits, gene_names


def build_cell_graph(genome: SCerevisiaeGenome, graph: SCerevisiaeGraph, names: list[str]) -> HeteroData:
    """The wildtype cell graph with the run's six STRING channels, genome base graph."""
    multigraph = build_gene_multigraph(graph=graph, graph_names=names)
    graphs_dict = SortedDict(multigraph.graphs.copy())
    graphs_dict["base"] = create_graph_from_gene_set(GeneSet(genome.gene_set))
    cell_graph = to_cell_data(GeneMultiGraph(graphs=graphs_dict), incidence_graphs=None)
    for name in names:
        assert ("gene", name, "gene") in cell_graph.edge_types, (name, cell_graph.edge_types)
    return cell_graph


def load_model(path: str, names: list[str], device: torch.device) -> tuple[Dango, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["hyper_parameters"]["forward_transform"] is None
    sd = ckpt["state_dict"]
    assert all(k.startswith("model.") for k in sd), [k for k in sd if not k.startswith("model.")]
    state = {k[len("model.") :]: v for k, v in sd.items()}
    convs = {k.split(".")[2] for k in state if k.startswith("pretrain_model.layer1_convs.")}
    assert convs == set(names), (convs, names)
    model = Dango(
        gene_num=state["pretrain_model.gene_embedding.weight"].shape[0],
        edge_types=names,
        hidden_channels=HIDDEN,
        num_heads=HEADS,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), int(ckpt["epoch"])


@torch.no_grad()
def score(model: Dango, cell_graph: HeteroData, sets: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    """Predictions for [n, k] gene-index sets and per-set residual-branch ratios per layer."""
    hs = model.hyper_sagnn
    cap: dict[str, torch.Tensor] = {}
    hooks = [
        hs.Q1.register_forward_hook(lambda m, i, o: cap.__setitem__("x1", i[0])),
        hs.O1.register_forward_hook(lambda m, i, o: cap.__setitem__("o1", o)),
        hs.Q2.register_forward_hook(lambda m, i, o: cap.__setitem__("x2", i[0])),
        hs.O2.register_forward_hook(lambda m, i, o: cap.__setitem__("o2", o)),
    ]
    n_all, k = sets.shape
    out = {key: np.empty(n_all, dtype=np.float64) for key in ("pred", "ratio1", "ratio2")}
    for start in range(0, n_all, BATCH):
        chunk = sets[start : start + BATCH]
        n = chunk.shape[0]
        batch = HeteroData()
        batch.num_graphs = n
        batch["gene"].perturbation_indices = torch.from_numpy(chunk.reshape(-1)).to(device)
        batch["gene"].perturbation_indices_batch = torch.arange(n, device=device).repeat_interleave(k)
        preds, _ = model(cell_graph, batch)
        out["pred"][start : start + n] = preds.double().cpu().numpy()
        for layer, beta in ((1, hs.beta1), (2, hs.beta2)):
            num = (beta * cap[f"o{layer}"]).pow(2).sum(1).view(n, k).sum(1).sqrt()
            den = cap[f"x{layer}"].pow(2).sum(1).view(n, k).sum(1).sqrt()
            out[f"ratio{layer}"][start : start + n] = (num / den).double().cpu().numpy()
    for h in hooks:
        h.remove()
    return out


@torch.no_grad()
def channel_weights(model: Dango, cell_graph: HeteroData) -> np.ndarray:
    """Per-gene softmax weights over the STRING channels, [num_genes, 6]."""
    emb = model.pretrain_model(cell_graph)["embeddings"]
    stacked = torch.stack([emb[e] for e in model.pretrain_model.edge_types], dim=1)
    w = torch.softmax(model.meta_embedding.attention_mlp(stacked).squeeze(-1), dim=1)
    integrated = model.meta_embedding(emb)
    assert torch.allclose((stacked * w.unsqueeze(-1)).sum(1), integrated, atol=1e-5)
    return w.double().cpu().numpy()


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "n": int(y.size),
        "pearson": float(pearsonr(y, p)[0]),
        "spearman": float(spearmanr(y, p)[0]),
        "mse": float(np.mean((y - p) ** 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda")
    device = torch.device(parser.parse_args().device)

    row_genes, y, splits, vocab = load_records()
    history = pd.read_csv(osp.join(RESULTS_DIR, "dango_full_dataset_history.csv"))
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    print(f"records {len(y)}; " + ", ".join(f"{s} {len(v)}" for s, v in splits.items()))
    for split in ("val", "test"):
        np.save(osp.join(RESULTS_DIR, f"dango_record_rows_{split}.npy"), splits[split])
        np.save(osp.join(RESULTS_DIR, f"dango_labels_{split}.npy"), y[splits[split]])

    summary: dict[str, object] = {"n_records": int(len(y)), "split_sizes": {s: int(len(v)) for s, v in splits.items()}}
    check_rows, ablation_rows = [], []
    for run, spec in RUNS.items():
        names = [f"{spec['string']}_{c}" for c in CHANNELS]
        cell_graph = build_cell_graph(genome, graph, names).to(device)
        node_ids = list(cell_graph["gene"].node_ids)
        node_to_idx = {g: i for i, g in enumerate(node_ids)}
        remap = np.array([node_to_idx[g] for g in vocab], dtype=np.int64)
        triples = remap[row_genes]
        edges = {n: int(cell_graph["gene", n, "gene"].num_edges) for n in names}
        print(f"\n{run}: cell graph {len(node_ids)} genes, edges (with self loops) {edges}")
        summary[f"{run}_graph"] = {"num_genes": len(node_ids), "edges_with_self_loops": edges}

        for tag, rel in spec["ckpts"].items():
            model, epoch = load_model(osp.join(CKPT_ROOT, spec["dir"], rel), names, device)
            assert model.pretrain_model.gene_num == len(node_ids)
            hs = model.hyper_sagnn
            beta1, beta2 = float(hs.beta1), float(hs.beta2)
            rec = history[(history.run_id == run) & (history.epoch == epoch)]
            assert len(rec) == 1, (run, epoch, len(rec))
            rec_p, rec_mse = float(rec.val_pearson.iloc[0]), float(rec.val_mse.iloc[0])
            key = f"{run}_{tag}"
            res = {}
            for split in ("val", "test"):
                sel = splits[split]
                res[split] = score(model, cell_graph, triples[sel], device)
                p = res[split]["pred"]
                np.save(osp.join(RESULTS_DIR, f"dango_predictions_{key}_{split}.npy"), p)
                m = metrics(y[sel], p)
                row = {"run": run, "ckpt": tag, "epoch": epoch, "split": split} | m
                if split == "val":
                    row |= {
                        "recorded_val_pearson": rec_p,
                        "recorded_val_mse": rec_mse,
                        "delta_pearson": m["pearson"] - rec_p,
                        "rel_delta_mse": (m["mse"] - rec_mse) / rec_mse,
                    }
                check_rows.append(row)
                print(f"  {tag} (epoch {epoch}) {split}: " + "  ".join(f"{k} {v:.6f}" for k, v in m.items() if k != "n"))
            cv = check_rows[-2]
            print(f"    recorded val pearson {rec_p:.6f} mse {rec_mse:.7f}; delta pearson {cv['delta_pearson']:+.2e}, rel delta mse {cv['rel_delta_mse']:+.2e}")
            assert abs(cv["delta_pearson"]) < PEARSON_TOL, cv
            assert abs(cv["rel_delta_mse"]) < MSE_RTOL, cv

            # ReZero ablation on val: both residual gates set to zero.
            sel = splits["val"]
            ablated = copy.deepcopy(model)
            ablated.hyper_sagnn.beta1.data.zero_()
            ablated.hyper_sagnn.beta2.data.zero_()
            abl = score(ablated, cell_graph, triples[sel], device)
            p0 = abl["pred"]
            np.save(osp.join(RESULTS_DIR, f"dango_predictions_{key}_val_beta0.npy"), p0)
            singles = score(ablated, cell_graph, np.arange(len(node_ids))[:, None], device)["pred"]
            additive_gap = float(np.abs(singles[triples[sel]].mean(1) - p0).max())
            assert additive_gap < 1e-5, additive_gap
            p_tr = res["val"]["pred"]
            between = float(pearsonr(p_tr, p0)[0])
            r1, r2 = res["val"]["ratio1"], res["val"]["ratio2"]
            common = {"run": run, "ckpt": tag, "epoch": epoch, "beta1": beta1, "beta2": beta2}
            for arm, p, ratios in (("trained", p_tr, (r1, r2)), ("beta_zero", p0, (abl["ratio1"], abl["ratio2"]))):
                ablation_rows.append(
                    common
                    | {"arm": arm}
                    | metrics(y[sel], p)
                    | {
                        "pearson_vs_trained_pred": float(pearsonr(p_tr, p)[0]),
                        "residual_ratio_l1_mean": float(ratios[0].mean()),
                        "residual_ratio_l1_median": float(np.median(ratios[0])),
                        "residual_ratio_l2_mean": float(ratios[1].mean()),
                        "residual_ratio_l2_median": float(np.median(ratios[1])),
                        "max_abs_gap_to_singleton_mean": additive_gap if arm == "beta_zero" else float("nan"),
                    }
                )
            print(
                f"    beta {beta1:.4f}/{beta2:.4f}; beta=0 val pearson {ablation_rows[-1]['pearson']:.6f} "
                f"mse {ablation_rows[-1]['mse']:.6f}; pearson(trained, beta0) {between:.6f}; "
                f"residual ratio mean l1 {r1.mean():.4f} l2 {r2.mean():.4f}; additive gap {additive_gap:.1e}"
            )

            if (run, tag) in WEIGHT_CKPTS:
                w = channel_weights(model, cell_graph)
                perturbed = np.isin(node_ids, vocab)
                val_counts = np.bincount(triples[sel].reshape(-1), minlength=len(node_ids))
                wdf = pd.DataFrame(w, columns=CHANNELS)
                wdf.insert(0, "n_val_records", val_counts)
                wdf.insert(0, "is_perturbed_in_006", perturbed)
                wdf.insert(0, "gene", node_ids)
                wdf.to_csv(osp.join(RESULTS_DIR, f"dango_meta_embedding_weights_{key}.csv"), index=False)
                slot_mean = (w * val_counts[:, None]).sum(0) / val_counts.sum()
                means: dict[str, object] = {
                    "all_genes": dict(zip(CHANNELS, w.mean(0).tolist())),
                    "perturbed_genes": dict(zip(CHANNELS, w[perturbed].mean(0).tolist())),
                    "val_gene_slots": dict(zip(CHANNELS, slot_mean.tolist())),
                    "n_perturbed_genes": int(perturbed.sum()),
                }
                summary[f"{key}_channel_weight_means"] = means
                for label in ("all_genes", "perturbed_genes", "val_gene_slots"):
                    print(f"    channel weight mean, {label}: " + "  ".join(f"{c} {v:.4f}" for c, v in means[label].items()))

    check = pd.DataFrame(check_rows)
    ablation = pd.DataFrame(ablation_rows)
    check.to_csv(osp.join(RESULTS_DIR, "dango_full_dataset_inference_check.csv"), index=False)
    ablation.to_csv(osp.join(RESULTS_DIR, "dango_rezero_ablation.csv"), index=False)
    summary["check"] = check.astype(object).where(check.notna(), None).to_dict(orient="records")
    summary["rezero_ablation"] = ablation.astype(object).where(ablation.notna(), None).to_dict(orient="records")
    with open(osp.join(RESULTS_DIR, "dango_full_dataset_inference_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + check.to_string(index=False))
    print("\n" + ablation.to_string(index=False))


if __name__ == "__main__":
    main()
