# experiments/043-expression-proteome-morphology/scripts/model_math_043_counts.py
# [[experiments.043-expression-proteome-morphology.scripts.model_math_043_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/model_math_043_counts.py
"""Numbers of the model-math mermaid note: parameter counts and the store's shapes.

Instantiates ``CellModel043`` exactly as ``train_043.build_model`` does for the default
config (``conf/train_043_default.yaml``, arm ``joint_9``) with vocabularies read from the
prepared store, sums ``p.numel()`` per block for each loss arm, and reads from the store the
marks per record and the measured index columns per dataset. Writes
``results/model_math_043_counts.json``, the source of every count in
[[experiments.043-expression-proteome-morphology.mermaid.model-math]].

The frozen tables and the graph mask are non-persistent buffers (no parameters), so the
model is built with two-row stand-in tables and no mask; the parameter count is unchanged.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)

import train_043 as T  # noqa: E402

from torchcell.losses.hl_gauss import HLGaussConfig  # noqa: E402
from torchcell.models.cell_model_043 import (  # noqa: E402
    ENCODER_DIMS,
    CellModel043,
    CellModel043Config,
)

STORE = "prep_v3_routeA"


def main() -> None:
    """Count, then write the JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    exp = osp.dirname(HERE)
    prep = osp.join(
        data_root,
        "data/torchcell/experiments/043-expression-proteome-morphology/train_043_prep",
        STORE,
    )
    with open(osp.join(exp, "conf/train_043_default.yaml")) as f:
        mcfg = yaml.safe_load(f)["model"]
    with open(osp.join(prep, "meta.json")) as f:
        meta = json.load(f)
    z = np.load(osp.join(prep, "store.npz"))
    n_parents = int(z["mark_parent"].max()) + 1
    shorts = meta["datasets"]
    G = len(meta["genes"])

    out: dict[str, object] = {
        "store": STORE,
        "n_genes": G,
        "n_envs": len(meta["env_keys"]),
        "n_parents": n_parents,
        "n_datasets_joint_9": len(T.JOINT_9),
        "params": {},
    }
    for loss in ("hlgauss", "pinball", "mse"):
        mc = CellModel043Config(
            hidden_dim=mcfg["hidden_dim"],
            num_heads=mcfg["num_heads"],
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
            mask_layers=(),
            head_graphs={int(k): v for k, v in mcfg["head_graphs"].items()},
            fourier_frequencies=mcfg["fourier_frequencies"],
            operator_ffn_mult=mcfg["operator_ffn_mult"],
            n_parents=n_parents,
            n_datasets=len(T.JOINT_9),
            n_envs=len(meta["env_keys"]),
            exp_type_classes=T.EXP_TYPE_CLASSES,
            exp_type_mode=mcfg["exp_type_mode"],
            exp_type_paths=T.EXP_TYPE_PATHS,
            type_token_dim=mcfg["type_token_dim"],
            env_token_dim=mcfg["env_token_dim"],
            dataset_token_dim=mcfg["dataset_token_dim"],
            head_hidden=mcfg["head_hidden"],
            loss=loss,
            hl_gauss=HLGaussConfig(),
            num_quantiles=19,
        )
        tables = [torch.zeros(2, d, dtype=torch.float16) for d in ENCODER_DIMS]
        model = CellModel043(
            mc, tables, torch.zeros(G, dtype=torch.long), torch.zeros(G), None
        )
        blocks = {
            name: sum(p.numel() for p in mod.parameters())
            for name, mod in model.named_children()
        }
        for name, p in model.named_parameters(recurse=False):
            blocks[name] = p.numel()
        blocks["trunk_per_layer"] = sum(
            p.numel() for p in model.trunk.layers[0].parameters()
        )
        blocks["total"] = sum(p.numel() for p in model.parameters())
        out["params"][loss] = blocks  # type: ignore[index]
        print(loss, blocks)

    n_marks = np.diff(z["mark_ptr"])
    per_ds: dict[str, dict[str, int | str]] = {}
    for i, s in enumerate(shorts):
        sel = z["rec_dataset"] == i
        row: dict[str, int | str] = {
            "records": int(sel.sum()),
            "marks_max": int(n_marks[sel].max()),
            "marks_median": int(np.median(n_marks[sel])),
        }
        if T.SPECS[s].level == "gene":
            Y = z["Y"][z["rec_yrow"][sel]]
            fin = np.isfinite(Y)
            row["level"] = "gene"
            row["index_columns_measured_any_record"] = int(fin.any(0).sum())
            row["index_columns_measured_median_record"] = int(np.median(fin.sum(1)))
        else:
            row["level"] = "record"
        per_ds[s] = row
    out["datasets"] = per_ds
    arm = [shorts.index(d) for d in (*T.JOINT_9, "sameith_dm")]
    out["marks_max_joint_9"] = int(n_marks[np.isin(z["rec_dataset"], arm)].max())
    path = osp.join(exp, "results/model_math_043_counts.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(path)


if __name__ == "__main__":
    main()
