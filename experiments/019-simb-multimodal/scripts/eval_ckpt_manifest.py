# experiments/019-simb-multimodal/scripts/eval_ckpt_manifest.py
# [[experiments.019-simb-multimodal.scripts.eval_ckpt_manifest]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/eval_ckpt_manifest
"""Write the checkpoint-evaluation manifest for a round from its W&B runs.

One row per run: the training config, the arm, the init seed, the W&B run id, the run
group (which names the checkpoint directory under `$DATA_ROOT/models/checkpoints/`), the
best-validation epoch and value as W&B recorded them, and the checkpoint file found on
this machine. `gh_eval_ckpt_predictions.slurm` reads the manifest one row per array task,
so a round's test reads need no hand-written case table; the W&B value in the row is what
the eval's validation pass must reproduce.

    python experiments/019-simb-multimodal/scripts/eval_ckpt_manifest.py --round v13 v14

Writes `results/eval_ckpt_manifest_<round>.tsv` per round. Rows whose checkpoint is not on
this machine are written with an empty path and reported, so the pull that is missing is
named rather than silently skipped.
"""

import argparse
import glob
import os
import os.path as osp

import wandb
from dotenv import load_dotenv

ROUNDS: dict[str, dict[str, str]] = {
    "v13": {
        "project": "zhao-group/torchcell_019_expr_v13",
        "config": "cgt_expr_v13_split",
        "metric": "val/expression/pearson_per_feature",
        "arm_prefix": "V_",
    },
    "v14": {
        "project": "zhao-group/torchcell_019_prot_v14",
        "config": "cgt_expr_v14_proteome",
        "metric": "val/proteome/pearson_per_feature",
        "arm_prefix": "P_",
    },
}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", nargs="+", choices=sorted(ROUNDS), required=True)
    args = parser.parse_args()
    ck_root = osp.join(os.environ["DATA_ROOT"], "models", "checkpoints")
    exp_root = os.environ.get(
        "EXPERIMENT_ROOT", osp.join(osp.dirname(osp.abspath(__file__)), "..", "..")
    )
    api = wandb.Api(timeout=120)
    for name in args.round:
        spec = ROUNDS[name]
        rows = []
        missing = []
        for run in api.runs(spec["project"]):
            tags = list(run.tags)
            arms = [t for t in tags if t.startswith(spec["arm_prefix"])]
            seeds = [t for t in tags if t.startswith("seed") and t[4:].isdigit()]
            if len(arms) != 1 or len(seeds) != 1:
                raise ValueError(f"{run.id}: arm tags {arms}, seed tags {seeds}")
            group = str(run.group)
            hist = run.history(
                keys=["epoch", spec["metric"]], samples=20000, pandas=True
            )
            hist = hist.dropna(subset=[spec["metric"]])
            peak = hist.loc[hist[spec["metric"]].idxmax()]
            found = sorted(
                glob.glob(osp.join(ck_root, group, "*best-metric-epoch=*.ckpt"))
            )
            ckpt = found[-1] if found else ""
            if not ckpt:
                missing.append(f"{arms[0]} seed{seeds[0][4:]} {run.id} {group}")
            rows.append(
                (
                    spec["config"],
                    arms[0],
                    seeds[0][4:],
                    run.id,
                    group,
                    str(int(peak["epoch"])),
                    f"{float(peak[spec['metric']]):.6f}",
                    ckpt,
                )
            )
        rows.sort(key=lambda r: (r[1], int(r[2])))
        out = osp.join(
            exp_root, "019-simb-multimodal", "results", f"eval_ckpt_manifest_{name}.tsv"
        )
        with open(out, "w") as fh:
            fh.write("config\tarm\tseed\trun_id\tgroup\tbest_epoch\tbest_val\tckpt\n")
            for r in rows:
                fh.write("\t".join(r) + "\n")
        print(
            f"{name}: {len(rows)} runs, {len(rows) - len(missing)} checkpoints found -> {out}"
        )
        for m in missing:
            print(f"  missing checkpoint: {m}")


if __name__ == "__main__":
    main()
