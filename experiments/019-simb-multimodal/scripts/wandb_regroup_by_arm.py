# experiments/019-simb-multimodal/scripts/wandb_regroup_by_arm.py
# [[experiments.019-simb-multimodal.scripts.wandb_regroup_by_arm]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/wandb_regroup_by_arm
"""Set each run's W&B group to its ARM, so one group page shows an arm across splits and seeds.

WHY. The 019 trainer writes `group = <host>-<jobid>_<hash>`, unique per run, so every group
page holds one run and there is no link that shows "the joint arm". The review unit is the
arm. This rewrites the group on the server to the arm family (`J_joint`, not `J_joint_s1`),
leaving the run NAME untouched: the name is `run_<host>-<jobid>_<hash>`, which is the
checkpoint directory, and `eval_ckpt_manifest.py` reads it from there.

RE-RUN AFTER EVERY `wandb sync` of a live run. A sync replays the offline run record, which
carries the original group. The script is idempotent and prints the group URL per arm.

    python experiments/019-simb-multimodal/scripts/wandb_regroup_by_arm.py --round v16
"""

import argparse
import json
import os.path as osp
import re

import wandb

ENTITY = "zhao-group"
# round -> (project, arm-tag regex whose group 1 is the arm family)
ROUNDS: dict[str, tuple[str, str]] = {
    "v16": ("torchcell_019_prot_v16", r"(J_(?:ref|expr|joint))_s\d+"),
    "v17": ("torchcell_019_expr_v17", r"(L_(?:ref|self|prop2))_s\d+"),
    "v18": ("torchcell_019_expr_v18", r"(Y_(?:ref|k0|ctx))_s\d+"),
}
RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", required=True, choices=sorted(ROUNDS))
    args = ap.parse_args()
    project, pattern = ROUNDS[args.round]
    arm_re = re.compile(pattern)

    groups: dict[str, list[str]] = {}
    changed = 0
    for run in wandb.Api(timeout=60).runs(f"{ENTITY}/{project}"):
        arms = [m.group(1) for t in run.tags if (m := arm_re.fullmatch(t))]
        if len(arms) != 1:
            raise ValueError(f"{run.id}: expected one arm tag, found {arms} in {run.tags}")
        if not run.name.startswith("run_"):
            raise ValueError(f"{run.id}: name {run.name!r} no longer names the checkpoint dir")
        if run.group != arms[0]:
            run.group = arms[0]
            run.update()
            changed += 1
        groups.setdefault(arms[0], []).append(run.id)

    out = {
        arm: {
            "url": f"https://wandb.ai/{ENTITY}/{project}/groups/{arm}",
            "run_ids": sorted(ids),
        }
        for arm, ids in sorted(groups.items())
    }
    path = osp.join(RESULTS, f"wandb_groups_{args.round}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{changed} runs regrouped")
    for arm, rec in out.items():
        print(f"{arm}  n={len(rec['run_ids'])}")
        print(rec["url"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
