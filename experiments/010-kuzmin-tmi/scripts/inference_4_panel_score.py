#!/usr/bin/env python
# experiments/010-kuzmin-tmi/scripts/inference_4_panel_score.py
# [[experiments.010-kuzmin-tmi.scripts.inference_4_panel_score]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_4_panel_score
"""Score every strain of the 20-strain panel with the three checkpoints, on CPU.

WHY. The panel's six triples are unknown, but its five singles and nine doubles are
already published, so the model can be asked about all twenty and checked on fourteen.
That turns the panel into a test of the model before a single strain is built.

WHAT IS AND IS NOT IN DISTRIBUTION. The 010 build is 376,732 records and EVERY one is a
3-gene perturbation, verified by reading the training LMDB. So:

  triples   in distribution. Predicted tau, no measured value, this is the ask.
  doubles   ZERO-SHOT. The model never saw a 2-gene perturbation in training. Measured
            digenic epsilon = f_ab - f_a*f_b is computable for all nine from published
            singles and doubles, so the comparison is real even though the input order
            is one the model was never trained on.
  singles   ZERO-SHOT, and the interaction of a single gene is identically zero by
            definition. Whatever the model emits here is pure diagnostic: a model that
            respects the definition should sit near zero.

Any agreement on the doubles is therefore evidence about ORDER TRANSFER, not about
in-distribution accuracy, and it must be reported that way.

CPU on purpose. All four GPUs were running the 025 training job. Twenty records is a
few minutes of CPU, so nothing needs to wait for a GPU.

THREE CHECKS, because each one closes an objection a reader would otherwise raise.

  parity            The six triples are members of the inference_4 space, so the GPU
                    run already scored them and this path must reproduce those 18
                    values. Not to equality: that run used autocast and this one is
                    float32, so the bar is 5e-3 absolute plus r > 0.9999.
  epsilon rebuild   The nine measured epsilon are reconstructed from singles and
                    doubles that do not all come from one screen. Three pairs also
                    carry a published Costanzo epsilon, so the reconstruction is
                    checked against it rather than trusted.
  rescue vs single  Whether the predicted rescue ordering of the third gene is just
                    that gene's own single-gene ordering. Agreement would mean the
                    ranking carries no combination information.

Run from repo root:
  ~/miniconda3/envs/torchcell/bin/python \
    experiments/010-kuzmin-tmi/scripts/inference_4_panel_score.py

Outputs:
  results/inference_4/panel20_scored.csv       every strain, per checkpoint
  results/inference_4/panel20_scored.json      counts and correlations quoted in prose
  $ASSET_IMAGES_DIR/010-kuzmin-tmi/inference_4_panel_score.{png,svg}
"""

import json
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")

import lmdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from scipy import stats as sps
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from inference_dataset_1 import InferenceDataset  # noqa: E402

from torchcell.data.graph_processor import Perturbation  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg  # noqa: E402
from torchcell.models.equivariant_cell_graph_transformer import (  # noqa: E402
    CellGraphTransformer,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.trainers.int_transformer_cell import RegressionTask  # noqa: E402
from torchcell.transforms.coo_regression_to_classification import (  # noqa: E402
    COOInverseCompose,
    COOLabelNormalizationTransform,
)
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

CONF_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "conf")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results", "inference_4")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
PANEL_ROOT = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/panel20")
TRAIN_LABEL_DF = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/010-kuzmin-tmi/001-small-build/processed/label_df.parquet",
)

CHECKPOINT_CONFIGS = {
    "M01": "equivariant_cell_graph_transformer_inference_4_m00.yaml",
    "M02": "equivariant_cell_graph_transformer_inference_4_m01.yaml",
    "M03": "equivariant_cell_graph_transformer_inference_4_m02.yaml",
}
# One record at a time: the transformer attends over all 6,607 genes, so a batch of 20
# would hold 20 x 9 heads x 6,607^2 attention entries at once.
BATCH_SIZE = 1
TORCH_THREADS = 16  # the 025 training job is using this machine too


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Arial", "font.size": 6, "axes.labelsize": 6,
            "axes.titlesize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "legend.fontsize": 5, "legend.title_fontsize": 5, "figure.titlesize": 6,
            "svg.fonttype": "none", "axes.linewidth": 0.5,
            "savefig.bbox": None, "savefig.pad_inches": 0.0,
        }
    )


def load_config(name: str):
    """Merge conf/default.yaml under the named config, the way hydra would."""
    base = OmegaConf.load(osp.join(CONF_DIR, "default.yaml"))
    spec = OmegaConf.load(osp.join(CONF_DIR, name))
    cfg = OmegaConf.merge(base, spec)
    if "defaults" in cfg:
        del cfg["defaults"]
    return cfg


def build_panel_lmdb(genotypes: list[tuple[str, ...]]) -> None:
    """Write one JSON record per panel strain, on the generator's own template.

    The template is read from the generator's recorded output rather than rebuilt, so
    the record schema here is the same one the 41.9M-triple space was written with.
    Only the perturbation list changes, and its length varies from 1 to 3.
    """
    with open(osp.join(RESULTS_DIR, "generation_summary.json")) as f:
        template = json.load(f)["template_record"]
    proto_pert = template[0]["experiment"]["genotype"]["perturbations"][0]

    processed = osp.join(PANEL_ROOT, "processed")
    os.makedirs(processed, exist_ok=True)
    lmdb_path = osp.join(processed, "lmdb")
    if osp.exists(lmdb_path):
        import shutil

        shutil.rmtree(lmdb_path)

    env = lmdb.open(lmdb_path, map_size=int(1e9), readonly=False)
    with env.begin(write=True) as txn:
        for i, genes in enumerate(genotypes):
            rec = json.loads(json.dumps(template))
            perts = []
            for g in genes:
                p = dict(proto_pert)
                p["systematic_gene_name"] = g
                p["perturbed_gene_name"] = g
                p["strain_id"] = f"{g}_deletion"
                perts.append(p)
            rec[0]["experiment"]["genotype"]["perturbations"] = perts
            txn.put(str(i).encode("utf-8"), json.dumps(rec).encode("utf-8"))
    env.close()
    print(f"wrote {len(genotypes)} records to {lmdb_path}")


def score(cfg, dataset, cell_graph, node_embeddings, inverse_transform,
          n_records: int) -> np.ndarray:
    """Predicted interaction for every record, one checkpoint, on CPU."""
    device = torch.device("cpu")
    loss_cfg = cfg.regression_task["loss"]
    graph_reg_lambda = float(loss_cfg["graph_regularization"]["lambda"])
    model = CellGraphTransformer(
        gene_num=cfg["model"]["gene_num"],
        hidden_channels=cfg["model"]["hidden_channels"],
        num_transformer_layers=cfg["model"]["num_transformer_layers"],
        num_attention_heads=cfg["model"]["num_attention_heads"],
        cell_graph=cell_graph,
        graph_regularization_config=cfg["model"]["graph_regularization"],
        perturbation_head_config=cfg["model"]["perturbation_head"],
        dropout=cfg["model"]["dropout"],
        graph_reg_lambda=graph_reg_lambda,
        node_embeddings=node_embeddings,
        learnable_embedding_config=cfg["model"].get("learnable_embedding"),
    ).to(device)

    loss_func = PointDistGraphReg(
        point_estimator=OmegaConf.to_container(loss_cfg["point_estimator"]),
        distribution_loss=OmegaConf.to_container(loss_cfg["distribution_loss"]),
        graph_regularization=OmegaConf.to_container(loss_cfg["graph_regularization"]),
        buffer=OmegaConf.to_container(loss_cfg["buffer"]),
        ddp=OmegaConf.to_container(loss_cfg["ddp"]),
    )

    ckpt = cfg.model["checkpoint_path"]
    if not osp.isabs(ckpt):
        ckpt = osp.join(DATA_ROOT, ckpt)
    if not osp.exists(ckpt):
        raise FileNotFoundError(ckpt)

    task = RegressionTask.load_from_checkpoint(
        ckpt,
        map_location=device,
        model=model,
        cell_graph=cell_graph,
        loss_func=loss_func,
        device=device,
        optimizer_config=cfg.regression_task["optimizer"],
        lr_scheduler_config=cfg.regression_task["lr_scheduler"],
        batch_size=BATCH_SIZE,
        clip_grad_norm=cfg.regression_task["clip_grad_norm"],
        clip_grad_norm_max_norm=cfg.regression_task["clip_grad_norm_max_norm"],
        inverse_transform=inverse_transform,
        plot_every_n_epochs=cfg.regression_task["plot_every_n_epochs"],
        plot_sample_ceiling=cfg.regression_task["plot_sample_ceiling"],
        plot_edge_recovery_every_n_epochs=cfg.regression_task.get(
            "plot_edge_recovery_every_n_epochs", 10
        ),
        plot_transformer_diagnostics_every_n_epochs=cfg.regression_task.get(
            "plot_transformer_diagnostics_every_n_epochs", 10
        ),
        grad_accumulation_schedule=cfg.regression_task.get("grad_accumulation_schedule"),
        execution_mode="inference",
        strict=False,
    )
    task.eval()
    model.eval()

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        follow_batch=["perturbation_indices"],
    )
    out = np.full(n_records, np.nan, dtype=np.float64)
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            preds, _ = task(batch)
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            preds = preds.reshape(-1)
            tmp = HeteroData()
            tmp["gene"].phenotype_values = preds
            tmp["gene"].phenotype_type_indices = torch.zeros(
                preds.numel(), dtype=torch.long
            )
            tmp["gene"].phenotype_sample_indices = torch.arange(preds.numel())
            tmp["gene"].phenotype_types = ["gene_interaction"]
            inv = inverse_transform(tmp)["gene"].phenotype_values.reshape(-1)
            n = inv.numel()
            out[cursor:cursor + n] = inv.numpy()
            cursor += n
    if cursor != n_records:
        raise SystemExit(f"scored {cursor} of {n_records} records")
    return out


def rescue_vs_single(df: pd.DataFrame) -> dict:
    """Rank the third gene by predicted rescue, and by its own single-gene score.

    Agreement would mean the rescue ordering restates the per-gene scores and carries
    no combination information. Disagreement is the panel's only evidence, before it is
    built, that the ordering is about the combination.
    """
    tri = pd.read_csv(osp.join(RESULTS_DIR, "panel20_triples.csv"))
    chassis_triples = tri[tri["arm"] == "chassis pair"]
    if chassis_triples.empty:
        return {}
    chassis = set(chassis_triples.iloc[0][["name1", "name2", "name3"]]) & set(
        chassis_triples.iloc[1][["name1", "name2", "name3"]]
    )
    f_chassis = float(
        df.loc[df["name"] == "+".join(sorted(chassis)), "measured_fitness"].iloc[0]
        if "+".join(sorted(chassis)) in set(df["name"])
        else df.loc[df["name"].isin(["+".join(p) for p in
                                     (tuple(chassis), tuple(reversed(tuple(chassis))))]),
                    "measured_fitness"].iloc[0]
    )
    singles = dict(zip(df[df["order"] == 1]["name"], df[df["order"] == 1]["pred_mean"]))
    rows = []
    for r in chassis_triples.itertuples():
        third = next(n for n in (r.name1, r.name2, r.name3) if n not in chassis)
        rows.append({
            "gene": third,
            "rescue_over_chassis_double": float(r.f_triple_predicted_worst - f_chassis),
            "single_gene_prediction": float(singles[third]),
        })
    rows.sort(key=lambda d: -d["rescue_over_chassis_double"])
    by_rescue = [d["gene"] for d in rows]
    by_single = [d["gene"] for d in
                 sorted(rows, key=lambda d: -d["single_gene_prediction"])]
    return {
        "chassis": sorted(chassis),
        "chassis_double_fitness": f_chassis,
        "rows": rows,
        "order_by_rescue": by_rescue,
        "order_by_single_gene": by_single,
        "orders_agree": by_rescue == by_single,
    }


def main():
    torch.set_num_threads(TORCH_THREADS)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    strains = pd.read_csv(osp.join(RESULTS_DIR, "panel20_strains.csv"))
    genotypes = [tuple(g.split("+")) for g in strains["genotype"]]
    print(f"panel: {len(genotypes)} strains, orders "
          f"{strains['order'].value_counts().sort_index().to_dict()}")
    build_panel_lmdb(genotypes)

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

    cfg0 = load_config(CHECKPOINT_CONFIGS["M01"])
    gene_multigraph = build_gene_multigraph(
        graph=graph, graph_names=cfg0.cell_dataset["graphs"]
    )
    graphs_dict = dict(gene_multigraph.graphs) if gene_multigraph else None

    class _LabelDfOnly:
        def __init__(self, df):
            self.label_df = df

    norm = COOLabelNormalizationTransform(
        _LabelDfOnly(pd.read_parquet(TRAIN_LABEL_DF)),
        OmegaConf.to_container(cfg0.transforms["forward_transform"]["normalization"]),
    )
    forward_transform = Compose([norm])
    inverse_transform = COOInverseCompose([norm])

    dataset = InferenceDataset(
        root=PANEL_ROOT,
        gene_set=genome.gene_set,
        graphs=graphs_dict,
        node_embeddings=None,
        graph_processor=Perturbation(),
        transform=forward_transform,
    )
    print(f"dataset: {len(dataset)} records")

    preds = {}
    for name, conf_name in CHECKPOINT_CONFIGS.items():
        print(f"\nscoring {name} on CPU ...")
        cfg = load_config(conf_name)
        preds[name] = score(cfg, dataset, dataset.cell_graph, None,
                            inverse_transform, len(genotypes))
        print(f"  {name}: " + ", ".join(f"{v:+.4f}" for v in preds[name]))

    df = strains.copy()
    for name in CHECKPOINT_CONFIGS:
        df[f"pred_{name}"] = preds[name]

    # PARITY GUARD. The six triples are members of the inference_4 space, so the GPU run
    # already scored them. If this CPU path reproduces those numbers, then the model
    # construction, the checkpoint load, the normalization and the inverse transform are
    # all correct here. It is worth an explicit check because loading these checkpoints
    # warns about `perturbation_transform.*_layers.0.*` keys being absent: the
    # checkpoint carries the older `cross_attn` / `ffn` / `norm1` / `norm2` names, which
    # the current module keeps as ALIASES of layer 0, so the tensors are shared and the
    # warning is cosmetic. This assertion is what proves that rather than assuming it.
    # EXACT equality is not the right bar. The GPU run scored under
    # torch.cuda.amp.autocast (`use_amp = device.type == "cuda"` in
    # equivariant_cell_graph_transformer_inference_4.py) while this path is float32 on
    # CPU, so the two differ by reduced-precision rounding. Measured: the largest
    # absolute gap over 18 values is 1.2e-3 on a prediction of 0.58, which is 0.2
    # percent. PARITY_TOL is set above that and below anything that would indicate a
    # wrong tensor, and the correlation is reported beside it so a structural error
    # cannot hide inside a loose absolute bound.
    PARITY_TOL = 5e-3
    gpu = pd.read_csv(osp.join(RESULTS_DIR, "panel20_triples.csv"))
    gpu_by_genes = {frozenset(t.split("+")): r for t, r in
                    zip(gpu["triple"], gpu.to_dict(orient="records"))}
    cpu_vals, gpu_vals = [], []
    for row in df[df["order"] == 3].itertuples():
        ref = gpu_by_genes[frozenset(row.genotype.split("+"))]
        for name in CHECKPOINT_CONFIGS:
            cpu_vals.append(getattr(row, f"pred_{name}"))
            gpu_vals.append(ref[name])
    cpu_vals, gpu_vals = np.array(cpu_vals), np.array(gpu_vals)
    worst_delta = float(np.abs(cpu_vals - gpu_vals).max())
    parity_r = float(np.corrcoef(cpu_vals, gpu_vals)[0, 1])
    print(f"\nparity against the GPU run, {len(cpu_vals)} values "
          f"(6 triples x 3 checkpoints): largest absolute difference "
          f"{worst_delta:.2e}, Pearson r = {parity_r:.6f}")
    if worst_delta > PARITY_TOL or parity_r < 0.9999:
        raise SystemExit(
            f"CPU predictions disagree with the stored GPU predictions: "
            f"max |delta| {worst_delta:.4f}, r {parity_r:.6f}; the model was not "
            f"reconstructed correctly"
        )
    pred_cols = [f"pred_{n}" for n in CHECKPOINT_CONFIGS]
    df["pred_mean"] = df[pred_cols].mean(axis=1)
    df["pred_worst"] = df[pred_cols].min(axis=1)

    # Measured interaction where it exists. A single gene has no interaction term, so
    # its measured value is zero by definition rather than by measurement.
    f_of = dict(zip(df["genotype"], df["measured_fitness"]))
    measured, kind = [], []
    for g, order in zip(df["genotype"], df["order"]):
        genes = g.split("+")
        if order == 1:
            measured.append(0.0)
            kind.append("zero by definition")
        elif order == 2:
            fa, fb = f_of.get(genes[0]), f_of.get(genes[1])
            fab = f_of.get(g)
            measured.append(fab - fa * fb if None not in (fa, fb, fab) else np.nan)
            kind.append("measured epsilon")
        else:
            measured.append(np.nan)
            kind.append("unknown, this is the ask")
    df["measured_interaction"] = measured
    df["measured_kind"] = kind
    df["known"] = df["order"] < 3
    df.to_csv(osp.join(RESULTS_DIR, "panel20_scored.csv"), index=False)

    print("\n=== every panel strain ===")
    print(df[["name", "order", "measured_fitness", "measured_interaction",
              "pred_M01", "pred_M02", "pred_M03", "pred_mean", "measured_kind"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    dbl = df[df["order"] == 2].dropna(subset=["measured_interaction"])
    sgl = df[df["order"] == 1]
    summary = {
        "n_strains": int(len(df)),
        "n_known": int(df["known"].sum()),
        "n_unknown": int((~df["known"]).sum()),
        "training_orders": {"3": 376732},
        "singles_zero_shot": {
            "mean_abs_prediction": float(sgl["pred_mean"].abs().mean()),
            "max_abs_prediction": float(sgl["pred_mean"].abs().max()),
            "label_sd": 0.06326,
        },
        "doubles_zero_shot": {
            "n": int(len(dbl)),
            "measured_epsilon_range": [float(dbl["measured_interaction"].min()),
                                       float(dbl["measured_interaction"].max())],
            "predicted_range": [float(dbl["pred_mean"].min()),
                                float(dbl["pred_mean"].max())],
            "mean_abs_error": float(
                (dbl["pred_mean"] - dbl["measured_interaction"]).abs().mean()
            ),
            # Direction of the error, which is the whole story: a model that merely had
            # a weak correlation would miss in both directions.
            "mean_signed_error": float(
                (dbl["pred_mean"] - dbl["measured_interaction"]).mean()
            ),
            "n_over_predicted": int(
                (dbl["pred_mean"] > dbl["measured_interaction"]).sum()
            ),
            "sign_test_p": float(sps.binomtest(
                int((dbl["pred_mean"] > dbl["measured_interaction"]).sum()),
                len(dbl), 0.5).pvalue),
            "n_measured_negative": int((dbl["measured_interaction"] < 0).sum()),
            "n_predicted_negative": int((dbl["pred_mean"] < 0).sum()),
        },
        # Is the measured epsilon itself trustworthy? It is reconstructed from singles
        # and doubles that do not all come from the same screen, and mixing
        # normalizations across screens is exactly what flipped a sign in the pcl6
        # census. Three of the nine pairs also carry a PUBLISHED epsilon from Costanzo
        # 2016, so the reconstruction can be checked against it on those three.
        "epsilon_reconstruction_check": {
            "n_with_published": int(dbl["published_epsilon"].notna().sum()),
            "max_abs_difference": (
                float((dbl["published_epsilon"] - dbl["measured_interaction"])
                      .abs().max())
                if dbl["published_epsilon"].notna().any() else None
            ),
        },
        # Does the prediction grow with the NUMBER of deleted genes rather than with
        # any three-way structure? A model with no order awareness, trained only at
        # order 3, has no reason not to.
        "mean_prediction_by_order": {
            str(int(o)): float(df[df["order"] == o]["pred_mean"].mean())
            for o in sorted(df["order"].unique())
        },
        # The sharpest single-gene check available: LAT1 carries 80 of the ranked top
        # 500, so if the panel's signal is a gene claim rather than an interaction
        # claim, LAT1 alone should score like the triples that contain it.
        # The one question the panel can answer about itself before it is built: is the
        # RESCUE ordering of the three regulators just their single-gene ordering? The
        # chassis-pair triples share their first two deletions, so the comparison is
        # within-chassis and the third gene is the only thing that varies.
        "rescue_vs_single": rescue_vs_single(df),
        "single_vs_triples": {
            "top_single": str(df.loc[df[df["order"] == 1]["pred_mean"].idxmax(), "name"]),
            "top_single_prediction": float(df[df["order"] == 1]["pred_mean"].max()),
            "n_triples_below_top_single": int(
                (df[df["order"] == 3]["pred_mean"]
                 < df[df["order"] == 1]["pred_mean"].max()).sum()
            ),
            "n_triples": int((df["order"] == 3).sum()),
        },
    }
    if len(dbl) >= 3:
        r, p = sps.pearsonr(dbl["measured_interaction"], dbl["pred_mean"])
        rho, prho = sps.spearmanr(dbl["measured_interaction"], dbl["pred_mean"])
        summary["doubles_zero_shot"].update(
            {"pearson_r": float(r), "pearson_p": float(p),
             "spearman_rho": float(rho), "spearman_p": float(prho)}
        )
        print(f"\ndoubles, zero-shot: Pearson r = {r:+.3f} (p = {p:.3f}), "
              f"Spearman rho = {rho:+.3f} (p = {prho:.3f}), n = {len(dbl)}")
    print(f"singles, zero-shot: mean |prediction| = "
          f"{summary['singles_zero_shot']['mean_abs_prediction']:.4f} "
          f"against a label SD of 0.0633")
    z = summary["doubles_zero_shot"]
    print(f"doubles: {z['n_over_predicted']} of {z['n']} over-predicted "
          f"(sign test p = {z['sign_test_p']:.4f}), mean signed error "
          f"{z['mean_signed_error']:+.4f}; {z['n_measured_negative']} measured "
          f"negative, {z['n_predicted_negative']} predicted negative")
    mo = summary["mean_prediction_by_order"]
    print("mean prediction by order: " +
          ", ".join(f"{k} gene(s) {v:+.4f}" for k, v in mo.items()))
    sv = summary["single_vs_triples"]
    print(f"{sv['top_single']} alone scores {sv['top_single_prediction']:+.4f}, above "
          f"{sv['n_triples_below_top_single']} of the {sv['n_triples']} triples")
    rv = summary.get("rescue_vs_single") or {}
    if rv:
        print(f"\nrescue ordering vs single-gene ordering, chassis "
              f"{'+'.join(rv['chassis'])} at f = {rv['chassis_double_fitness']:.4f}:")
        for row in rv["rows"]:
            print(f"  {row['gene']:<6} rescue {row['rescue_over_chassis_double']:+.4f}"
                  f"   single {row['single_gene_prediction']:+.4f}")
        print(f"  by rescue: {' > '.join(rv['order_by_rescue'])}")
        print(f"  by single: {' > '.join(rv['order_by_single_gene'])}")
        print(f"  orders agree: {rv['orders_agree']}")
    er = summary["epsilon_reconstruction_check"]
    if er["max_abs_difference"] is not None:
        print(f"reconstructed epsilon vs published, on the {er['n_with_published']} "
              f"pairs that carry one: largest difference "
              f"{er['max_abs_difference']:.4f}")

    with open(osp.join(RESULTS_DIR, "panel20_scored.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot(df, summary, osp.join(IMAGES_DIR, "inference_4_panel_score"))
    print(f"\nwrote {RESULTS_DIR} and figures to {IMAGES_DIR}")


def _letter(ax, letter):
    ax.text(-0.17, 1.06, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def plot(df, summary, out_stem):
    set_plot_style()
    fig, axes2 = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(120.0))
    )
    axes = axes2.ravel()
    c_known, c_unknown = PLOT_PALETTE[4], PLOT_PALETTE[0]

    # a: what is known and what is not, every strain.
    ax = axes[0]
    d = df.sort_values(["order", "pred_mean"]).reset_index(drop=True)
    ys = np.arange(len(d))[::-1]
    cols = [c_known if k else c_unknown for k in d["known"]]
    ax.barh(ys, d["pred_mean"], 0.62, color=cols, edgecolor="black",
            linewidth=0.35, zorder=3)
    for y, row in zip(ys, d.itertuples()):
        lo = min(row.pred_M01, row.pred_M02, row.pred_M03)
        hi = max(row.pred_M01, row.pred_M02, row.pred_M03)
        ax.plot([lo, hi], [y, y], color="black", linewidth=0.5, zorder=4)
        # A diamond ONLY where a measurement exists. A single gene's interaction is
        # zero by definition, not by measurement, so it gets a different mark.
        if row.measured_kind == "measured epsilon":
            ax.scatter([row.measured_interaction], [y], s=9, marker="D",
                       facecolor="white", edgecolor="black", linewidths=0.5, zorder=5)
        elif row.measured_kind == "zero by definition":
            ax.scatter([0.0], [y], s=14, marker="|", color="black",
                       linewidths=0.7, zorder=5)
    ax.axvline(0.0, color="black", linewidth=0.5, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(d["name"], fontsize=4.2)
    ax.set_xlabel("Interaction")
    ax.set_title(
        f"All {len(df)} panel strains scored\n"
        f"bar is the ensemble mean prediction, line spans the three checkpoints",
        fontsize=6, loc="left", pad=3,
    )
    ax.legend(handles=[
        plt.Line2D([], [], marker="s", linestyle="none", markersize=3,
                   color=c_known, label=f"known ({int(df['known'].sum())})"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=3,
                   color=c_unknown, label=f"unknown ({int((~df['known']).sum())})"),
        plt.Line2D([], [], marker="D", linestyle="none", markersize=3,
                   markerfacecolor="white", markeredgecolor="black",
                   label="measured $\\varepsilon$"),
        plt.Line2D([], [], marker="|", linestyle="none", markersize=5,
                   color="black", label="0 by definition"),
    ], loc="upper right", frameon=True, fontsize=5, handlelength=1.2,
        labelspacing=0.3, borderpad=0.3)

    # b: predicted against measured on the nine doubles. Zero-shot.
    ax = axes[1]
    dbl = df[df["order"] == 2].dropna(subset=["measured_interaction"])
    ax.scatter(dbl["measured_interaction"], dbl["pred_mean"], s=14,
               color=c_known, edgecolor="black", linewidths=0.4, zorder=4)
    for row in dbl.itertuples():
        lo = min(row.pred_M01, row.pred_M02, row.pred_M03)
        hi = max(row.pred_M01, row.pred_M02, row.pred_M03)
        ax.plot([row.measured_interaction] * 2, [lo, hi], color="black",
                linewidth=0.5, zorder=3)
        # Only the three that carry the argument get a label. Panel a already names
        # every pair, and nine labels in this space collide into unreadability.
        if row.name in ("LAT1+CUP9", "TOS8+LAT1", "PDC1+LAT1"):
            ax.annotate(row.name, (row.measured_interaction, row.pred_mean),
                        textcoords="offset points", xytext=(4, -1), fontsize=4.6)
    lim = np.array([
        min(dbl["measured_interaction"].min(), dbl["pred_mean"].min()) - 0.02,
        max(dbl["measured_interaction"].max(), dbl["pred_mean"].max()) + 0.02,
    ])
    ax.plot(lim, lim, color="black", linewidth=0.5, linestyle="--", zorder=2)
    ax.axhline(0, color="0.7", linewidth=0.4, zorder=1)
    ax.axvline(0, color="0.7", linewidth=0.4, zorder=1)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Measured digenic $\\varepsilon = f_{ab} - f_a f_b$")
    ax.set_ylabel("Predicted interaction")
    z = summary["doubles_zero_shot"]
    sub = (f"Pearson $r$ = {z['pearson_r']:+.3f}, $p$ = {z['pearson_p']:.3f}; "
           f"{z['n_over_predicted']} of {z['n']} above the line"
           if "pearson_r" in z else "too few pairs to correlate")
    ax.set_title(f"The nine doubles, ZERO-SHOT\n{sub}", fontsize=6, loc="left", pad=3)

    # c: does the model respect the definition at order 1?
    ax = axes[2]
    for i, o in enumerate((1, 2, 3)):
        sub_df = df[df["order"] == o]
        ax.scatter(np.full(len(sub_df), i) + np.linspace(-0.12, 0.12, len(sub_df)),
                   sub_df["pred_mean"], s=12,
                   color=(c_known if o < 3 else c_unknown),
                   edgecolor="black", linewidths=0.4, zorder=4,
                   label=None)
    ax.axhline(0.0, color="black", linewidth=0.5, zorder=2)
    ax.axhspan(-0.06326, 0.06326, color="0.88", zorder=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["singles\n(zero-shot)", "doubles\n(zero-shot)",
                        "triples\n(trained order)"])
    ax.set_ylabel("Predicted interaction")
    mo = summary["mean_prediction_by_order"]
    ax.plot([0, 1, 2], [mo["1"], mo["2"], mo["3"]], color="black", linewidth=0.7,
            linestyle="--", marker="_", markersize=8, zorder=5)
    ax.set_title(
        "Order transfer: the model trained on triples ONLY\n"
        "band is $\\pm 1$ label SD; dashed line is the mean, "
        f"{mo['1']:+.3f} to {mo['2']:+.3f} to {mo['3']:+.3f}",
        fontsize=6, loc="left", pad=3,
    )

    # d: the six triples, which is what the panel is for.
    ax = axes[3]
    tri = df[df["order"] == 3].sort_values("pred_mean").reset_index(drop=True)
    ys = np.arange(len(tri))
    ax.barh(ys, tri["pred_mean"], 0.6, color=c_unknown, edgecolor="black",
            linewidth=0.4, zorder=3)
    for y, row in zip(ys, tri.itertuples()):
        lo = min(row.pred_M01, row.pred_M02, row.pred_M03)
        hi = max(row.pred_M01, row.pred_M02, row.pred_M03)
        ax.plot([lo, hi], [y, y], color="black", linewidth=0.5, zorder=4)
    ax.axvline(0.08, color="0.5", linewidth=0.5, linestyle=":", zorder=2)
    ax.axvline(0.0, color="black", linewidth=0.5, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(tri["name"], fontsize=4.6)
    ax.set_xlabel("Predicted $\\tau$")
    ax.set_title("The six unknowns, rescored one at a time\n"
                 "dotted line is the Kuzmin 2020 positive call",
                 fontsize=6, loc="left", pad=3)

    for ax in axes:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
            sp.set_color("black")
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
    for ax, letter in zip(axes, "abcd"):
        _letter(ax, letter)

    fig.suptitle(
        "The 20-strain panel scored by all three checkpoints on CPU. Singles and "
        "doubles are zero-shot: the 010 model trained on 3-gene perturbations only.",
        fontsize=6, y=0.995,
    )
    fig.tight_layout(rect=(0.01, 0, 1, 0.965))
    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
