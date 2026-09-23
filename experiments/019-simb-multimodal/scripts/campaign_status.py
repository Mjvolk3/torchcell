# experiments/019-simb-multimodal/scripts/campaign_status.py
# [[experiments.019-simb-multimodal.scripts.campaign_status]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/campaign_status
"""One table of every 019 job across every cluster: round, wave, arms, epoch, progress.

WHY. The campaign runs on three machines and five partitions at once, and `squeue` answers
a different question on each: it names the launcher, not the round. On 2026-09-20 five
concurrent arrays all read `019-wave5-igb`, so the only way to tell the joint round from
the locality round was to remember which job id was which. This reads every cluster, maps
each array task to the round and arms it is actually running, and pulls the current epoch
out of the run's own log, so one command answers "what is training right now".

WHERE THE FACTS COME FROM, in order of authority:
  * `squeue` for state, partition, elapsed and remaining walltime;
  * the job NAME for round and wave, parsed as `019-<round>w<N>-<stage>` (the convention
    documented in igb_expr_wave5.slurm); a job that predates it falls back to the stage
    recorded in its log header, and failing that reports the round as unknown rather than
    guessing;
  * the per-run log file names for the arms on each card (one file per run, named
    `<jobname>_<jobid>_<task>_<arm>_seed<k>.out`);
  * the tail of the newest log for the current epoch, read with `tail -c`, never by
    reading the whole file (these reach tens of megabytes and the login node is not a
    place to cat them).

The budget per round is read from the round's config, so "epoch 1168 of 1200" is the
config's number rather than a remembered one.

    python experiments/019-simb-multimodal/scripts/campaign_status.py
    python experiments/019-simb-multimodal/scripts/campaign_status.py --no-epochs   # faster

Each cluster is ONE ssh call. Nothing here runs on a compute node and nothing computes.
"""

import argparse
import re
import subprocess
from dataclasses import dataclass, field

# A cluster: how to reach it, whose jobs to list, and where its 019 slurm logs live.
# `ssh` None means the local machine.
CLUSTERS: list[dict[str, str | None]] = [
    {
        "name": "IGB",
        "ssh": "mjvolk3@biologin.igb.illinois.edu",
        "user": "mjvolk3",
        "logs": "/home/a-m/mjvolk3/projects/torchcell/experiments/019-simb-multimodal/slurm/output",
    },
    {
        "name": "GilaHyper",
        "ssh": None,
        "user": "michaelvolk",
        "logs": "/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output",
    },
]

# Stage -> (round label, config, epoch budget). The budget is what the config sets; a
# stage missing here still lists, with an unknown budget, rather than being dropped.
STAGES: dict[str, tuple[str, str, int]] = {
    "split": ("v13 split", "cgt_expr_v13_split", 6000),
    "proteome": ("v14 proteome", "cgt_expr_v14_proteome", 2000),
    "wd": ("v15 weight decay", "cgt_expr_v15_wd", 6000),
    # 500 in wave 1; wave 2 continues every run from its epoch-499 checkpoint to 1,200.
    "joint": ("v16 joint", "cgt_expr_v16_joint", 1200),
    "locality": ("v17 locality", "cgt_expr_v17_locality", 1200),
    "hygiene": ("v18 hygiene", "cgt_expr_v18_hygiene", 1200),
}
NAME_RE = re.compile(r"019-(?P<round>v\d+)w(?P<wave>\d+)-(?P<stage>[a-z_]+)")
EPOCH_RE = re.compile(r"Epoch (\d+)")


@dataclass
class Task:
    """One slurm array task, with what the logs say it is actually running."""

    cluster: str
    job: str
    partition: str
    state: str
    elapsed: str
    left: str
    name: str
    reason: str
    arms: list[str] = field(default_factory=list)
    epoch: int | None = None

    @property
    def round_label(self) -> str:
        """Round and wave from the job name, or the raw name when it predates the convention."""
        m = NAME_RE.fullmatch(self.name)
        if m:
            stage = m.group("stage")
            label = STAGES.get(stage, (m.group("round"), "", 0))[0]
            return f"{label} w{m.group('wave')}"
        return self.name

    @property
    def budget(self) -> int | None:
        """The round's configured epoch budget, or None when the stage is unrecognized."""
        m = NAME_RE.fullmatch(self.name)
        if m and m.group("stage") in STAGES:
            return STAGES[m.group("stage")][2]
        return None


def run(ssh: str | None, cmd: str) -> str:
    argv = (
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", ssh, cmd]
        if ssh
        else ["bash", "-lc", cmd]
    )
    p = subprocess.run(argv, capture_output=True, text=True)
    return p.stdout


def collect(cluster: dict[str, str | None], want_epochs: bool) -> list[Task]:
    user, logs, ssh = cluster["user"], cluster["logs"], cluster["ssh"]
    # ONE ssh: the queue, then for each of my array tasks the per-run log names and the
    # last epoch line of the newest one. Built as a single remote script so a cluster
    # costs one round trip rather than one per task.
    remote = f"""
squeue -u {user} -h -o '%i|%P|%T|%M|%L|%j|%r' | sed 's/^/Q|/'
{
        ""
        if not want_epochs
        else f'''
for t in $(squeue -u {user} -h -t RUNNING -o '%i'); do
  fs=$(ls {logs}/*_${{t}}_*.out 2>/dev/null)
  [ -z "$fs" ] && continue
  arms=$(for f in $fs; do basename "$f" | sed -E "s/.*_${{t}}_//; s/\\.out$//"; done | paste -sd, -)
  newest=$(ls -t $fs | head -1)
  ep=$(tail -c 6000 "$newest" | tr '\\r' '\\n' | grep -oE 'Epoch [0-9]+' | tail -1)
  echo "E|$t|$arms|$ep"
done
'''
    }
"""
    out = run(ssh, remote)
    tasks: dict[str, Task] = {}
    extra: dict[str, tuple[str, str]] = {}
    for line in out.splitlines():
        if line.startswith("Q|"):
            f = line[2:].split("|")
            if len(f) < 7:
                continue
            tasks[f[0]] = Task(
                cluster=str(cluster["name"]),
                job=f[0],
                partition=f[1],
                state=f[2],
                elapsed=f[3],
                left=f[4],
                name=f[5],
                reason=f[6],
            )
        elif line.startswith("E|"):
            f = line[2:].split("|")
            if len(f) >= 3:
                extra[f[0]] = (f[1], f[2] if len(f) > 2 else "")
    for jid, (arms, ep) in extra.items():
        if jid in tasks:
            tasks[jid].arms = [a for a in arms.split(",") if a]
            m = EPOCH_RE.search(ep)
            tasks[jid].epoch = int(m.group(1)) if m else None
    return list(tasks.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-epochs", action="store_true", help="skip the per-run log reads"
    )
    ap.add_argument("--all", action="store_true", help="include jobs that are not 019")
    args = ap.parse_args()

    rows: list[Task] = []
    for c in CLUSTERS:
        rows += collect(c, not args.no_epochs)
    if not args.all:
        rows = [t for t in rows if "019" in t.name or t.name.startswith("019")]
    order = {"RUNNING": 0, "PENDING": 1}
    rows.sort(key=lambda t: (order.get(t.state, 2), t.cluster, t.round_label, t.job))

    w = max([len(t.round_label) for t in rows] + [12])
    print(
        f"{'CLUSTER':<10} {'JOB':<15} {'PART':<7} {'ROUND':<{w}} {'STATE':<8} {'EPOCH':>12} {'ELAPSED':>11}  ARMS / REASON"
    )
    for t in rows:
        b = t.budget
        ep = f"{t.epoch} of {b}" if t.epoch and b else (str(t.epoch) if t.epoch else "")
        tail = (
            ", ".join(t.arms) if t.arms else (t.reason if t.state != "RUNNING" else "")
        )
        print(
            f"{t.cluster:<10} {t.job:<15} {t.partition:<7} {t.round_label:<{w}} "
            f"{t.state:<8} {ep:>12} {t.elapsed:>11}  {tail}"
        )
    run_n = sum(1 for t in rows if t.state == "RUNNING")
    n_runs = sum(len(t.arms) for t in rows if t.state == "RUNNING")
    print(f"\n{run_n} tasks running ({n_runs} runs), {len(rows) - run_n} queued")


if __name__ == "__main__":
    main()
