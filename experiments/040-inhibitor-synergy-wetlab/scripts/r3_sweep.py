# experiments/040-inhibitor-synergy-wetlab/scripts/r3_sweep.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.r3_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/r3_sweep
r"""Write round 3's sweep files and the Delta submission script, one GPU per job.

Rounds 1 and 2 varied one field at a time around one recipe at 50 epochs. The W&B
history of the multi-source arms shows every auxiliary loss still falling at epoch 50
(Hoepfner 0.55 to 0.85, Hillenmeyer 0.9) while the single-source Vanacloig loss is flat
at 0.25, so a multi-source arm has been compared against a converged single-source one.
Round 3 trains longer, holds out part of every auxiliary source so each source carries
its own held-out score and held-out loss curve, and spends the budget on the edges of
the design space rather than on seeds.

Each arm becomes one sweep file of one or more configs; each config is one compound-cold
(or gene-cold) fold with three seeds. Jobs are packed so one job is one GPU and one
process, holding the folds of one arm, and no job is estimated past ``LIMIT_HOURS``.

    python experiments/040-inhibitor-synergy-wetlab/scripts/r3_sweep.py --out-dir \\
        experiments/040-inhibitor-synergy-wetlab/conf/mixture

Writes ``r3_<arm>.yaml`` per arm, ``r3_submit_delta.sh`` (the sbatch lines, one per job),
and ``results/r3_plan.csv`` (one row per job with its arm, folds and cost estimate).
"""

from __future__ import annotations

import argparse
import os
import os.path as osp

import pandas as pd
import yaml

EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")

#: Measured single-GPU cost of one config (one fold, three seeds) at 50 epochs, in hours,
#: from the round-1 and round-2 W&B runtimes: 0.78 h for one Vanacloig source on a
#: GilaHyper RTX 6000 Ada, 2.07 h for Vanacloig plus Hoepfner on a Delta A40, 2.84 h for
#: three sources on an A40. The A40 is taken as 1.4 times slower than the Ada, so these
#: are A40 hours and a GilaHyper job finishes sooner.
HOURS_PER_CONFIG_50: dict[int, float] = {1: 1.1, 2: 2.1, 3: 2.9, 4: 3.6}
#: Those are three-seed figures; this round is a SCREEN and runs one seed per fold, so an
#: arm's cost is a third of them per fold. The seed ensemble is what round 4 spends on the
#: directions this round says move.
SEEDS_PER_FOLD = 1
#: The folds a screening arm runs. Three of the five compound-cold folds cover 17 of the
#: 28 compounds, enough to order arms; a surviving arm runs all five in round 4.
SCREEN_FOLDS: list[int] = [0, 1, 2]
#: Every job asks for the partition limit rather than its estimate. The estimates below
#: are of a regime that has not converged, so they are the WORST kind of estimate to size
#: a job by, and a job that ends early costs nothing while one that is cut off loses its
#: unwritten folds.
LIMIT_HOURS = 12.0
#: What one job should aim to take; folds beyond this go to another job of the same arm.
TARGET_HOURS = 10.5

SOURCES: dict[str, list[str]] = {
    "van": ["vanacloig"],
    "vanhop": ["vanacloig", "hoepfner_hop"],
    "vanhophet": ["vanacloig", "hoepfner_hop", "hillenmeyer_het"],
    "vanhophetwil": ["vanacloig", "hoepfner_hop", "hillenmeyer_het", "wildenhain"],
    "vanwil": ["vanacloig", "wildenhain"],
}

#: The round's shared recipe: 038 round 10's converged settings, the host head trained on
#: the public anchors and the ex21 titration, the combination plate never trained on.
BASE: dict[str, object] = {
    "cgt_layers": 1,
    "cgt_lambda": 0.0,
    "eval_epochs": 5,
    # THE PROTOCOL CHANGED FOR THIS ROUND, and the first 150-epoch run is why. Under
    # ``fit_on: pool`` the fold's validation compounds are inside the training set, so the
    # logged validation score is in-sample: it climbs to 0.88 while the held-out score
    # peaks at 0.304 near epoch 85 and falls to 0.274 by 115, and the Vanacloig held-out
    # loss bottoms at 0.629 near epoch 35 and rises to 0.717 (run erueq4ns). Keeping
    # ``select: last`` at 150 epochs would therefore report an overfitted model on every
    # arm. ``fit_on: train`` holds the four validation compounds out so the selection
    # signal is honest, and ``select: val`` stops at that signal's best step. The cost is
    # four fitted compounds; the alternative is a round whose every arm is read past its
    # own optimum. The bridge arm keeps round 1's pool-and-last protocol so the two rounds
    # still join.
    "select": "val",
    "fit_on": "train",
    "gene_batch": 128,
    "env_chunk": 16,
    "host_batch": 16,
    "cond_batch": 24,
    "seeds": list(range(SEEDS_PER_FOLD)),
    "host_train": "anchors+ex21",
    "dose": "off",
    "env_layers": 1,
    "lambda_host": 1.0,
    "checkpoint": True,
}


def arm(
    name: str,
    sources: str,
    epochs: int,
    *,
    fold_seed: int = 0,
    split: str = "compound",
    aux_holdout: str | None = None,
    **overrides: object,
) -> dict[str, object]:
    """One arm: its source set, its length, and whatever it changes from ``BASE``.

    An arm with an auxiliary source holds out part of it by default (by compound), so the
    source has a held-out score; an arm whose question is the dose holds out doses.
    """
    names = SOURCES[sources]
    if aux_holdout is None:
        aux_holdout = "compound" if len(names) > 1 else "none"
    assert len(names) > 1 or aux_holdout == "none", f"{name}: no auxiliary source"
    return {
        "arm": name,
        "sources": names,
        "epochs": epochs,
        "fold_seed": fold_seed,
        "split": split,
        "aux_holdout": aux_holdout,
        "overrides": overrides,
    }


def arms() -> list[dict[str, object]]:
    """The round's arms, grouped by the question each one answers."""
    out: list[dict[str, object]] = []

    # 1. Does combining data help, at a length where every source has converged? The
    #    150-epoch row is the paired comparison; 300 says whether 150 is itself enough.
    for sources in ("van", "vanhop", "vanhophet", "vanhophetwil", "vanwil"):
        out.append(arm(f"{sources}_e150", sources, 150))
    # 300 epochs on the heaviest source set only: the held-out loss is logged every five
    # epochs inside every 150-epoch arm, so saturation is read from those curves and this
    # arm is the check that reading is right rather than the way it is measured
    out.append(arm("vanhophet_e300", "vanhophet", 300))
    # the round-1 arm's own length AND its protocol, so the round-1 reading can be placed
    # on this round's axis; the only arm here that keeps pool-and-last
    out.append(arm("vanhop_e50", "vanhop", 50, fit_on="pool", select="last"))

    # 2. How the source enters: one learned token per source (the round-1 encoding), the
    #    typed attribute tokens, a per-source linear head, or both.
    out.append(
        arm("vanhophet_e150_attr", "vanhophet", 150, source_encoding="attributes")
    )
    out.append(arm("vanhophet_e150_head", "vanhophet", 150, source_head=True))
    out.append(
        arm(
            "vanhophet_e150_attr_head",
            "vanhophet",
            150,
            source_encoding="attributes",
            source_head=True,
        )
    )

    # 3. How much the auxiliary sources should count, and how many of their conditions a
    #    step sees.
    out.append(arm("vanhophet_e150_aw0p3", "vanhophet", 150, aux_weight=0.3))
    out.append(arm("vanhophet_e150_cb96", "vanhophet", 150, cond_batch=96))

    # 4. Dose. The public screens carry 83 Hoepfner and 66 Hillenmeyer compounds at two or
    #    more molar doses, so a held-out DOSE of a seen compound is a direct test of the
    #    dose channel; the compound hold-out twin says whether FiLM costs anything on the
    #    compound axis.
    out.append(
        arm("vanhop_e150_film_dose", "vanhop", 150, aux_holdout="dose", dose="film")
    )
    out.append(
        arm(
            "vanhophet_e150_film_dose",
            "vanhophet",
            150,
            aux_holdout="dose",
            dose="film",
        )
    )
    out.append(arm("vanhop_e150_film", "vanhop", 150, dose="film"))

    # 5. Gene-cold: a held-out deletion strain, every compound fitted. The in-house
    #    deletion assay is the only validation for a prediction on a gene outside the
    #    Vanacloig panel, so this is where that claim starts.
    for sources in ("van", "vanhop", "vanhophet"):
        out.append(arm(f"{sources}_e150_gene", sources, 150, split="gene"))

    # 6. Capacity, on the three-source set at the round's length.
    out.append(arm("vanhophet_e150_L2", "vanhophet", 150, env_layers=2))
    out.append(arm("vanhophet_e150_cgt2", "vanhophet", 150, cgt_layers=2))
    out.append(arm("vanhophet_e150_wide", "vanhophet", 150, dim=128, hidden=512))

    # 7. Fold seeds: the gene-level score moved 0.276 to 0.363 across partitions on one
    #    arm while nested ridge moved 0.017, so the two headline arms are repeated on the
    #    other two partitions.
    for sources in ("van", "vanhop"):
        for fold_seed in (1, 2):
            out.append(
                arm(f"{sources}_e150_fs{fold_seed}", sources, 150, fold_seed=fold_seed)
            )
    return out


def configs(spec: dict[str, object], folds: list[int]) -> list[dict[str, object]]:
    """One config per fold of ``spec``."""
    out = []
    for fold in folds:
        config = dict(BASE)
        config |= {
            "name": f"mix_{spec['arm']}_f{fold}",
            "sources": spec["sources"],
            "epochs": spec["epochs"],
            "fold_seed": spec["fold_seed"],
            "split": spec["split"],
            "aux_holdout": spec["aux_holdout"],
            "folds": [fold],
        }
        config |= spec["overrides"]  # type: ignore[arg-type]
        out.append(config)
    return out


def hours(spec: dict[str, object], n_folds: int) -> float:
    """Estimated single-A40 hours for ``n_folds`` folds of ``spec``."""
    per = (
        HOURS_PER_CONFIG_50[len(spec["sources"])]
        * int(spec["epochs"])
        / 50
        * SEEDS_PER_FOLD
        / 3
    )
    if spec["overrides"].get("cond_batch") == 96:  # type: ignore[union-attr]
        per *= 1.6
    if (
        spec["overrides"].get("env_layers") == 2
        or spec["overrides"].get("cgt_layers") == 2
    ):  # type: ignore[union-attr]
        per *= 1.4
    if spec["overrides"].get("dim") == 128:  # type: ignore[union-attr]
        per *= 1.3
    return per * n_folds


def pack(spec: dict[str, object]) -> list[list[int]]:
    """The arm's five folds split into jobs that each stay near ``TARGET_HOURS``."""
    per_fold = hours(spec, 1)
    per_job = max(1, int(TARGET_HOURS // per_fold))
    folds = list(SCREEN_FOLDS)
    return [folds[i : i + per_job] for i in range(0, len(folds), per_job)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=osp.join(EXPERIMENT, "conf", "mixture"))
    parser.add_argument("--account", default="bfjt-delta-gpu")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    rows, lines = [], []
    for spec in arms():
        groups = pack(spec)
        for k, folds in enumerate(groups):
            suffix = "" if len(groups) == 1 else f"_p{k}"
            sweep = f"r3_{spec['arm']}{suffix}"
            path = osp.join(args.out_dir, f"{sweep}.yaml")
            estimate = hours(spec, len(folds))
            limit = LIMIT_HOURS
            header = (
                f"# experiments/040-inhibitor-synergy-wetlab/conf/mixture/{sweep}.yaml\n"
                f"#\n"
                f"# ROUND 3, ARM {spec['arm']}, folds {folds} of fold seed"
                f" {spec['fold_seed']}, written by scripts/r3_sweep.py.\n"
                f"# Sources: {', '.join(spec['sources'])}."
                f" Split: {spec['split']}. Auxiliary hold-out: {spec['aux_holdout']}."
                f" Epochs: {spec['epochs']}.\n"
                f"# ESTIMATE, not a measurement: {estimate:.1f} A40 hours for this file"
                f" (round-1 runtimes scaled by epochs and sources); the job asks for"
                f" {limit:.1f} h.\n"
                f"#\n"
                f"#   sbatch --job-name=040-{sweep} --time={int(limit)}:00:00"
                f" --export=ALL,SWEEP=conf/mixture/{sweep}.yaml,PARALLEL=1 \\\n"
                f"#     experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm\n"
            )
            with open(path, "w") as f:
                f.write(header)
                yaml.safe_dump(
                    configs(spec, folds), f, sort_keys=False, default_flow_style=False
                )
            lines.append(
                f"sbatch --job-name=040-{sweep} --account={args.account}"
                f" --qos={args.account} --time={int(limit)}:00:00"
                f" --export=ALL,SWEEP=conf/mixture/{sweep}.yaml,PARALLEL=1 \\\n"
                f"  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm"
            )
            rows.append(
                {
                    "job": len(rows) + 1,
                    "sweep": sweep,
                    "arm": spec["arm"],
                    "sources": " ".join(spec["sources"]),
                    "n_sources": len(spec["sources"]),
                    "epochs": spec["epochs"],
                    "split": spec["split"],
                    "aux_holdout": spec["aux_holdout"],
                    "fold_seed": spec["fold_seed"],
                    "folds": " ".join(str(f) for f in folds),
                    "configs": len(folds),
                    "estimated_hours": round(estimate, 2),
                    "time_limit_hours": int(limit),
                    "overrides": ", ".join(
                        f"{k}={v}"
                        for k, v in spec["overrides"].items()  # type: ignore[union-attr]
                    ),
                }
            )

    plan = pd.DataFrame(rows)
    plan.to_csv(osp.join(RESULTS, "r3_plan.csv"), index=False)
    script = osp.join(args.out_dir, "r3_submit_delta.sh")
    with open(script, "w") as f:
        f.write(
            "#!/bin/bash\n"
            "# experiments/040-inhibitor-synergy-wetlab/conf/mixture/r3_submit_delta.sh\n"
            "#\n"
            "# ROUND 3 SUBMISSION, written by scripts/r3_sweep.py. One GPU and one\n"
            "# process per job, no dependencies and no -preempt partition, so every job\n"
            "# is eligible the moment it is submitted.\n"
            "#\n"
            "#   ssh delta\n"
            "#   cd /scratch/bbub/mjvolk3/torchcell-040\n"
            "#   mkdir -p experiments/040-inhibitor-synergy-wetlab/slurm/output\n"
            "#   bash experiments/040-inhibitor-synergy-wetlab/conf/mixture/r3_submit_delta.sh\n"
            "#\n"
            "# An account at its limit: rerun with ACCOUNT=<other>-delta-gpu, which sets\n"
            "# the matching QOS as well (an account change alone leaves the old QOS and\n"
            "# the job is rejected as InvalidQOS).\n"
            "set -eu\n"
            f'ACCOUNT="${{ACCOUNT:-{args.account}}}"\n'
            "cd /scratch/bbub/mjvolk3/torchcell-040\n"
            "mkdir -p experiments/040-inhibitor-synergy-wetlab/slurm/output\n\n"
        )
        for line in lines:
            f.write(
                line.replace(
                    f"--account={args.account}", '--account="$ACCOUNT"'
                ).replace(f"--qos={args.account}", '--qos="$ACCOUNT"')
                + "\n\n"
            )
    os.chmod(script, 0o755)

    print(plan.to_string(index=False))
    print(
        f"\n{len(plan)} jobs, {plan['configs'].sum()} configs, "
        f"{plan['estimated_hours'].sum():.0f} estimated A40 hours, "
        f"{plan['time_limit_hours'].sum()} hours requested"
    )
    print(f"wrote {script}")
    print(f"wrote {osp.join(RESULTS, 'r3_plan.csv')}")


if __name__ == "__main__":
    main()
