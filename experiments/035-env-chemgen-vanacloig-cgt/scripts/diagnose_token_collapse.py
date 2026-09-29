# experiments/035-env-chemgen-vanacloig-cgt/scripts/diagnose_token_collapse.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.diagnose_token_collapse]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/diagnose_token_collapse
"""Do the cell graph transformer's strain summaries still tell strains apart?

Round 1 of the ``cgt_env`` arm trained to a loss the embedding table beats easily, and
its within-compound raw Spearman stayed near zero: the model predicted about one value per
compound and did not learn which genes are sick. This measures where the identity of a
strain is lost, at initialization and in every saved checkpoint:

``H_genes``   the encoder's gene tokens. Reported as the share of their total squared
              norm that varies ACROSS genes (1 means every token distinct, 0 means every
              token identical), and the mean cosine between random token pairs.
``z_S``       the readout's strain summary, the sum of the four perturbed embeddings.
              Same two statistics across strains.
``y``         the prediction under one compound, its standard deviation over strains
              beside its standard deviation over compounds for one strain.

The embedding-table arm is measured on ``z_S`` and ``y`` as the contrast. Writes
``results/token_collapse.csv``.
"""

from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, osp.dirname(__file__))
from train_vanacloig_cgt import (  # noqa: E402
    EnvironmentReadout,
    build_cell_graph,
    strain_indices,
)
from vanacloig_data import load_cells  # noqa: E402

from torchcell.models.equivariant_cell_graph_transformer import (  # noqa: E402
    CellGraphTransformer,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXPERIMENT = osp.join(EXPERIMENT_ROOT, "035-env-chemgen-vanacloig-cgt")
CHECKPOINTS = osp.join(DATA_ROOT, "models/checkpoints/035-env-chemgen-vanacloig-cgt")
N_SAMPLE = 512


def spread(x: torch.Tensor) -> tuple[float, float]:
    """Share of the squared norm that varies across rows, and mean pairwise cosine."""
    x = x.double()
    total = float((x**2).sum())
    across = float(((x - x.mean(dim=0)) ** 2).sum())
    unit = torch.nn.functional.normalize(x, dim=1)
    cosine = unit @ unit.T
    off = cosine[~torch.eye(len(x), dtype=torch.bool)]
    return across / total, float(off.mean())


def build(arm: str, cfg: dict, cell_graph, feature_dim: int) -> EnvironmentReadout:
    encoder = None
    if arm != "embedding_env":
        encoder = CellGraphTransformer(
            gene_num=cfg["model"]["gene_num"],
            hidden_channels=cfg["model"]["hidden_channels"],
            num_transformer_layers=cfg["model"]["num_transformer_layers"],
            num_attention_heads=cfg["model"]["num_attention_heads"],
            cell_graph=cell_graph,
            graph_regularization_config=cfg["model"]["graph_regularization"],
            perturbation_head_config=cfg["model"]["perturbation_head"],
            dropout=cfg["model"]["dropout"],
            graph_reg_lambda=0.0,
            node_embeddings=None,
            learnable_embedding_config=cfg["model"]["learnable_embedding"],
        )
    return EnvironmentReadout(
        arm=arm,
        encoder=encoder,
        gene_num=cfg["model"]["gene_num"],
        hidden=cfg["model"]["hidden_channels"],
        feature_dim=feature_dim,
        dropout=cfg["model"]["dropout"],
    )


@torch.no_grad()
def measure(
    model: EnvironmentReadout,
    cell_graph,
    strain: torch.Tensor,
    features: torch.Tensor,
    compound_features: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(0)
    rows = torch.tensor(rng.choice(len(strain), N_SAMPLE, replace=False))
    batch_strain = strain[rows]
    out: dict[str, float] = {}
    if model.encoder is not None:
        size, order = batch_strain.shape
        batch = {"gene": type("G", (), {})()}
        batch["gene"].perturbation_indices = batch_strain.reshape(-1)
        batch["gene"].perturbation_indices_batch = torch.arange(size).repeat_interleave(
            order
        )
        _, rep = model.encoder(cell_graph, batch)
        genes = rep["H_genes"]
        sample = torch.tensor(rng.choice(len(genes), N_SAMPLE, replace=False))
        out["H_genes_across_share"], out["H_genes_mean_cosine"] = spread(genes[sample])
        z_s = rep["H_genes_pert"][torch.arange(size).unsqueeze(1), batch_strain].sum(1)
        out["h_CLS_norm"] = float(rep["h_CLS"].norm())
    else:
        z_s = model.table(batch_strain).sum(dim=1)
    out["z_S_across_share"], out["z_S_mean_cosine"] = spread(z_s)
    out["z_S_mean_norm"] = float(z_s.norm(dim=1).mean())

    one = compound_features[0].expand(N_SAMPLE, -1)
    y_strains, _ = model(cell_graph, batch_strain, one)
    first = batch_strain[:1].expand(len(compound_features), -1)
    y_compounds, _ = model(cell_graph, first, compound_features)
    out["y_sd_over_strains"] = float(y_strains.std())
    out["y_sd_over_compounds"] = float(y_compounds.std())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    raw = OmegaConf.load(osp.join(EXPERIMENT, "conf", "default.yaml"))
    raw.pop("hydra")
    cfg = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(cfg, dict)
    torch.manual_seed(0)

    cells = load_cells(cfg["data"]["cell_table"], cfg["data"]["embedding"])
    cell_graph = build_cell_graph(list(cfg["cell_dataset"]["graphs"]))
    strain = strain_indices(cells, list(cell_graph["gene"].node_ids))
    compound_features = torch.tensor(cells.compound_features, dtype=torch.float32)
    features = compound_features[torch.tensor(cells.compound_of_cell)]

    rows = []
    for arm in ("cgt_env", "embedding_env"):
        torch.manual_seed(0)
        model = build(arm, cfg, cell_graph, compound_features.shape[1])
        rows.append(
            {"arm": arm, "checkpoint": "initialization"}
            | measure(model, cell_graph, strain, features, compound_features)
        )
    for path in sorted(glob.glob(osp.join(CHECKPOINTS, "round1_*", "*.pt"))):
        arm = "embedding_env" if "embedding_env" in path else "cgt_env"
        model = build(arm, cfg, cell_graph, compound_features.shape[1])
        model.load_state_dict(torch.load(path, map_location="cpu"))
        name = osp.join(osp.basename(osp.dirname(path)), osp.basename(path))
        print(f"measuring {name}", flush=True)
        rows.append(
            {"arm": arm, "checkpoint": name}
            | measure(model, cell_graph, strain, features, compound_features)
        )
    table = pd.DataFrame(rows)
    table.to_csv(osp.join(EXPERIMENT, "results", "token_collapse.csv"), index=False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
