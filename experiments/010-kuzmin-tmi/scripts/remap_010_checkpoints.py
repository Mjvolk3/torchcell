# experiments/010-kuzmin-tmi/scripts/remap_010_checkpoints.py
# [[experiments.010-kuzmin-tmi.scripts.remap_010_checkpoints]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/remap_010_checkpoints

"""Make the 010 checkpoints loadable by the current model code.

``EquivariantPerturbationTransform`` was refactored from a single fixed block to
a ``ModuleList`` of blocks after the 010 runs finished, so the 010 checkpoints
store

    model.perturbation_transform.cross_attn.*      norm1.*  norm2.*  ffn.N.*

while the current class expects

    model.perturbation_transform.cross_attn_layers.0.*   norm1_layers.0.*  ...

The current class still exposes ``.cross_attn``, ``.ffn``, ``.norm1`` and
``.norm2`` as aliases onto element 0 of those lists, so the two names denote the
same parameters and the rename is one-to-one with identical shapes. Nothing else
in the state dict changes.

This writes a remapped copy of each checkpoint under
``$DATA_ROOT/models/checkpoints-remapped/`` and leaves the originals untouched.

The remap is only trustworthy if the remapped model reproduces the metrics the
original run recorded, so run the eval afterwards and check the reported val and
test Pearson against ``results/prediction_calibration_stats.csv``. A silent
mis-mapping would still load and would still produce plausible-looking numbers.
"""

import os
import os.path as osp
import re
import shutil

import torch
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
CKPT_ROOT = osp.join(DATA_ROOT, "models", "checkpoints")
OUT_ROOT = osp.join(DATA_ROOT, "models", "checkpoints-remapped")

# group directory -> checkpoint file, for the three 010 best-Pearson checkpoints
CHECKPOINTS = {
    "compute-3-3-2027905_a1260b50c3d74b6b7acea919b89416feb6fc957b3023c9ac866f9"
    "0378df82625": "lzs9pcj3-best-pearson-epoch=24-val/gene_interaction/"
    "Pearson=0.4520.ckpt",
    "compute-3-3-2027907_a1260b50c3d74b6b7acea919b89416feb6fc957b3023c9ac866f9"
    "0378df82625": "yv4r30bi-best-pearson-epoch=25-val/gene_interaction/"
    "Pearson=0.4472.ckpt",
    "compute-3-3-2036902_bd9e6c666ea1c0e7d1bbb6321fbc4d3bd5f60f100d6dc0e0288cd"
    "97e366fc15e": "c7671wgj-best-pearson-epoch=24-val/gene_interaction/"
    "Pearson=0.4619.ckpt",
}

PREFIX = "model.perturbation_transform."


def remap_key(key: str) -> str:
    """Rename a single-block perturbation-transform key onto the ModuleList."""
    if not key.startswith(PREFIX):
        return key
    tail = key[len(PREFIX) :]
    for old, new in (
        ("cross_attn.", "cross_attn_layers.0."),
        ("norm1.", "norm1_layers.0."),
        ("norm2.", "norm2_layers.0."),
    ):
        if tail.startswith(old):
            return PREFIX + new + tail[len(old) :]
    m = re.match(r"ffn\.(\d+)\.(.+)", tail)
    if m:
        return f"{PREFIX}ffn_layers.0.{m.group(1)}.{m.group(2)}"
    return key


def main() -> None:
    for group, rel in CHECKPOINTS.items():
        src = osp.join(CKPT_ROOT, group, rel)
        dst = osp.join(OUT_ROOT, group, rel)
        os.makedirs(osp.dirname(dst), exist_ok=True)
        ckpt = torch.load(src, map_location="cpu", weights_only=False)

        state = ckpt["state_dict"]
        # The current class keeps `.cross_attn` etc. as attributes aliasing element
        # 0 of the ModuleLists, so its own state_dict carries BOTH spellings of
        # those twelve tensors. Loading is strict, so the checkpoint must too:
        # add the list spelling and keep the alias rather than replacing it.
        added = {}
        for k, v in state.items():
            new = remap_key(k)
            if new != k:
                added[new] = v
        for new, v in added.items():
            assert new not in state, f"{new} already present, remap would overwrite"
        state.update(added)
        ckpt["state_dict"] = state
        torch.save(ckpt, dst)
        print(f"{rel.split('-')[0]}: added {len(added)} aliased keys -> {dst}")

    # The eval script resolves DATA_ROOT/models/checkpoints/<path>, so expose the
    # remapped tree under that name for a run that points DATA_ROOT here.
    print(f"\nremapped tree: {OUT_ROOT}")
    print("point the eval at it by swapping checkpoints/ for checkpoints-remapped/")
    if not osp.exists(osp.join(OUT_ROOT, "README.txt")):
        with open(osp.join(OUT_ROOT, "README.txt"), "w") as f:
            f.write(
                "Key-renamed copies of the 010 best-Pearson checkpoints, written by\n"
                "experiments/010-kuzmin-tmi/scripts/remap_010_checkpoints.py.\n"
                "Parameters are identical to the originals; only the\n"
                "perturbation_transform key names change. Verify any run against\n"
                "results/prediction_calibration_stats.csv before trusting it.\n"
            )
    shutil.copystat(CKPT_ROOT, OUT_ROOT)


if __name__ == "__main__":
    main()
