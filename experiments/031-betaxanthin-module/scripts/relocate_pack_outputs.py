# experiments/031-betaxanthin-module/scripts/relocate_pack_outputs.py
# [[experiments.031-betaxanthin-module.scripts.relocate_pack_outputs]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/relocate_pack_outputs

"""Move a pack's outputs from the essentiality study's trees into the study they belong to.

A pack submitted without ``ESS_STUDY`` runs under the launcher's default,
``028-gene-essentiality``: the trainer writes its result file into that study's
``results/``, the test-prediction dump into ``$DATA_ROOT/test-predictions-028`` and the
checkpoint into ``$DATA_ROOT/checkpoints-028``. The readouts of the module studies look
for the result file in their own ``results/`` and open the dump at the path the result
records, so a mis-routed pack reads as "no cells yet". This script moves the three files
of every seed of a stem into the named study's trees and rewrites the two recorded paths
in the result file, leaving a ``relocated_from`` record beside them. The W&B run of a
mis-routed cell stays in the essentiality project; the ``wandb_run_id`` in the result is
untouched. Nothing is copied: a file that already exists at the destination raises.

    ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/relocate_pack_outputs.py \
        --stem w203_aadelbxscore_h12l1p1_e200 --study 032-amino-acid-module
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import shutil

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
SOURCE_STUDY = "028-gene-essentiality"


class Relocation(BaseModel):
    stem: str
    seed: int
    result_from: str
    result_to: str
    dump_from: str
    dump_to: str
    checkpoint_from: str
    checkpoint_to: str


def _move(src: str, dst: str) -> None:
    if osp.exists(dst):
        raise FileExistsError(dst)
    os.makedirs(osp.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


def relocate(stem: str, study: str) -> list[Relocation]:
    src_results = osp.join(EXPERIMENT_ROOT, SOURCE_STUDY, "results")
    dst_results = osp.join(EXPERIMENT_ROOT, study, "results")
    src_num, dst_num = SOURCE_STUDY.split("-")[0], study.split("-")[0]
    paths = sorted(glob.glob(osp.join(src_results, f"{stem}_s*.json")))
    if not paths:
        raise FileNotFoundError(osp.join(src_results, f"{stem}_s*.json"))
    done: list[Relocation] = []
    for result_from in paths:
        with open(result_from) as fh:
            result = json.load(fh)
        seed = int(result_from.rsplit("_s", 1)[1].removesuffix(".json"))
        dump_from = result["test"]["dump_path"]
        ckpt_from = result["checkpoint_path"]
        assert f"test-predictions-{src_num}" in dump_from, dump_from
        assert f"checkpoints-{src_num}" in ckpt_from, ckpt_from
        dump_to = dump_from.replace(f"test-predictions-{src_num}", f"test-predictions-{dst_num}")
        ckpt_to = ckpt_from.replace(f"checkpoints-{src_num}", f"checkpoints-{dst_num}")
        result_to = osp.join(dst_results, osp.basename(result_from))
        rec = Relocation(
            stem=stem, seed=seed, result_from=result_from, result_to=result_to,
            dump_from=dump_from, dump_to=dump_to, checkpoint_from=ckpt_from, checkpoint_to=ckpt_to,
        )
        _move(dump_from, dump_to)
        _move(ckpt_from, ckpt_to)
        result["test"]["dump_path"] = dump_to
        result["checkpoint_path"] = ckpt_to
        result["relocated_from"] = rec.model_dump()
        if osp.exists(result_to):
            raise FileExistsError(result_to)
        with open(result_to, "w") as fh:
            json.dump(result, fh, indent=1)
        os.remove(result_from)
        done.append(rec)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--study", required=True, choices=["031-betaxanthin-module", "032-amino-acid-module"])
    args = ap.parse_args()
    for rec in relocate(args.stem, args.study):
        print(f"seed {rec.seed}: {rec.result_from} -> {rec.result_to}")


if __name__ == "__main__":
    main()
