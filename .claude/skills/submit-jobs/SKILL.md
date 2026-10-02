---
name: submit-jobs
description: The compute submission rules for GilaHyper, the IGB Biocluster and NCSA Delta. Read in full before ANY sbatch, srun, scontrol, scancel, or launcher edit on any cluster, and before estimating queue wait or budget. Covers login-node policy, sandbox flags, accounts and balances, partitions and their caps, memory and time sizing, job naming, staggering versus dependencies, checkpoint continuation, worktree pinning, and the post-submit checks.
---

# Submit Jobs

Every job we launch goes through these rules. They were each learned by losing a card, a
day, a batch, or goodwill with an admin; the dated memory behind each one is in
`~/.claude/projects/-home-michaelvolk-Documents-projects-torchcell/memory/` and this file
is the consolidated, current version. **When a new rule is learned, add it here first**, then
to memory. A rule that lives only in a conversation is lost by the next session.

**Usage:** `/submit-jobs` before a launch, or read this file whenever a task involves
`sbatch`, `srun`, `scontrol`, `scancel`, a `.slurm` launcher, a submitter script, a queue
estimate, or a service-unit budget.

## 1. Rules that hold on every cluster

1. **Never compute on a login node.** Login nodes are for `sbatch`, `squeue`, `sacct`,
   `scontrol`, `scancel`, `git`, `rsync`, and the one-shot `wandb sync`. Anything that imports
   torch, scans logs, walks a filesystem, or runs longer than a few seconds is a job. IGB
   locks accounts for this and has emailed warnings twice (a `grep` loop over `.out` files;
   a bare `wandb sync` of a backlog).
2. **A launcher is a committed script in `experiments/<id>/scripts/`**, named for its
   cluster (`gh_*.slurm`, `igb_*.slurm`, `delta_*.slurm`), with its submitter beside it. A
   one-off query or probe is a small `.slurm` in the session scratchpad with its output
   there; it is still a job.
3. **`sbatch` copies the script at submit time.** A fix to a `.slurm` file reaches only jobs
   submitted after it. Every PENDING job submitted before the fix is cancelled and
   resubmitted; `scontrol hold` + fix + `release` does not work. Configs and Python code
   under the project root are read at RUN time, so those changes do take effect for queued
   jobs, which is exactly why the next rule exists.
4. **Pin the code a job runs.** Never advance a worktree under a running job, and never
   point a multi-day job at a checkout another session may `git checkout` (Delta grid
   21796128 lost all 64 runs that way). On GilaHyper submit through
   `freeze_and_submit.sh` (rsyncs a snapshot to `$DATA_ROOT/job-snapshots/`); on Delta use a
   detached worktree under `/scratch/bbub/mjvolk3/torchcell.worktrees/<name>` at a recorded
   commit. Spawned DataLoader workers re-import the live code every epoch, so an in-progress
   edit can kill a RUNNING cell.
5. **Name the job** `<experiment>-<round>w<N>-<stage>` with `-J` on the sbatch line (the
   `#SBATCH --job-name` in the file is a placeholder). `<N>` counts submissions of that round
   on that cluster. Rename in place with `scontrol update JobId=<id> JobName=...`.
6. **Size `--mem` and `--cpus-per-task` from a MEASURED peak, about 2x**, and pass them on
   the sbatch line, not only in the header; a submitter's env knob can be documented and
   inert (`ESS_MEM` was, three packs died OOM). After every submit confirm the request took:
   `sacct -j <id> -o JobName,ReqMem,ReqCPUS,Timelimit -n`.
7. **Give an honest `--time`.** The time limit decides the queue wait on Delta and the
   backfill placement on GilaHyper. Measure the previous round's wall time with `sacct -o
   Elapsed` and set the limit at about 1.3x that, never a round number chosen for comfort.
   Users may lower a running job's limit (`scontrol update JobId=<id> TimeLimit=...`), never
   raise it.
8. **Submit everything eligible at once. No dependency chains and no `--begin` stagger,
   with one exception.** `--dependency=after:<id>+30` lanes make each job eligible only
   after its predecessor STARTS, so a lane advances one job per queue wait; the 025 round-2
   campaign finished 13 of 80 jobs in 4.5 days that way. A `--begin` stagger is milder but
   can still only delay a job, and the stage-in contention it would guard against has never
   been measured (six concurrent 554 GB copies at 290 MB/s are 1.7 GB/s on Lustre, and
   fairshare bounds how many of our jobs start together). Submit the whole batch and let
   the scheduler parallelize it. The one legitimate dependency is **checkpoint
   continuation**: a run that outlives
   the partition's wall limit is split into legs chained `afterok:<previous leg>`, and the
   next leg resumes BOTH the checkpoint and the W&B run (`wandb.init(id=<run id>,
   resume="must")`, or record `resumed_from` so the readout can stitch the legs). A
   continuation that opens a new W&B run is a bug. Queued chains are cleared in place with
   `scontrol update JobId=<id> Dependency= StartTime=now+<m>minutes` (job ids, scripts and
   age priority survive); a cancelled predecessor counts as started and releases its
   successors.
9. **One sbatch per Bash call** in this harness. On GilaHyper the call also needs
   `dangerouslyDisableSandbox: true` (the sandbox blocks the slurm daemon), and a compound
   command that bundles `cd`, `chmod` or a loop with the sbatch is denied even then. A
   submitter script that loops over sbatch is fine on Delta and IGB.
10. **Poll with `squeue`, never `pgrep -f`** (it matches its own poller shell). Slurm jobs
    are not harness-tracked; a background `until [ -z "$(squeue -h -u $USER)" ]` loop or a
    `ScheduleWakeup` is how a session gets re-invoked.
11. **Compute nodes on IGB and Delta have no outbound network.** `WANDB_MODE=offline`, then
    one-shot `wandb sync` from the login node, through the per-run wrapper
    `experiments/019-simb-multimodal/scripts/igb_login_wandb_sync.sh`, never a bare glob.
    Read training progress from W&B on GilaHyper, never from logs on a login node.
12. **Report runs with W&B links, one URL per line**, and name the job id, the statistic
    and the sample size for every number quoted. A cancelled or in-flight job has no
    result; say the epoch it reached.
13. **Before a batch submission, audit it** the way the 80-job 025 round was audited:
    compose every config override, check every path the launcher reads exists on that
    cluster, `sbatch --test-only` one line, run the real-size smoke, and only then submit.
    A wrong batch costs the whole wait, not one job.

## 2. GilaHyper (our node: 128 CPUs, 513 GB, 4 GPUs, `sched/backfill`, every job priority 1)

- `sbatch`/`squeue`/`scancel` need `dangerouslyDisableSandbox: true`, one command per call.
- **Memory budget:** the database build reserves 256 GB; metabolic-side work (026 to 035)
  is capped at half the node, 64 CPUs and 256 GB, across everything it runs at once. Add up
  what is allocated first: `squeue -u michaelvolk -o "%j %C %m %t"`. Measured peaks: a 027
  training cell 5.4 GB (request 12g), a CPU audit 10 GB (16g), a KG full build 123 to 148 GB
  (256G as a reservation). Read a running job's peak from
  `/sys/fs/cgroup/system.slice/gilahyper_slurmstepd.scope/job_<id>/memory.peak`.
- **GPU cards:** jobs from this work may occupy only the physical cards the user names
  (currently 0 and 1, which are slurm IDX 3 and 2; slurm's index is the REVERSE of
  nvidia-smi's). Inside a job the card is always `CUDA_VISIBLE_DEVICES=0`, so the launcher
  cannot pick; hold pending packs and release one only when a free slurm index is in the
  allowed set. Card memory: a width-36 plain cell 3.3 GB, a KL graph-regularized cell 12 to
  15 GB (three per 46 GB card, twelve killed seven with OOM).
- **Backfill:** a pending GPU pack at the head of the queue holds a reservation on the node;
  a CPU job starts beside the running packs only if its `--time` ends before that
  reservation. Give CPU jobs a short honest limit; `squeue --start` shows the plan.
  `scontrol update NumCPUs` does not fix a pending job; cancel and resubmit.
- **Epilog takes about 6 minutes** per job (`COMPLETING`, nvidia-smi walk). Prefer fewer,
  fuller packs; do not cancel or resubmit on seeing a "free" card under a COMPLETING job.
- **Packs** (028/031/032): `ESS_STUDY=<study> ESS_PACK=<pack> ESS_JOB_NAME=<name>
  ESS_MEM=36g ESS_CPUS=16 bash experiments/028-gene-essentiality/scripts/freeze_and_submit.sh
  experiments/028-gene-essentiality/scripts/gh_train_essentiality_pack.slurm`. Without
  `ESS_STUDY` the outputs land in the 028 trees.
- **DDP:** `torchrun --nnodes=1 --nproc_per_node=4 --rdzv-backend=c10d
  --rdzv-endpoint=127.0.0.1:$PORT --rdzv-id=$SLURM_JOB_ID --local-addr=127.0.0.1`; the bare
  hostname does not route. Activate conda with `source ~/miniconda3/etc/profile.d/conda.sh`,
  not `bin/activate` (it eats `$1`).
- **Anything that touches the served Neo4j store is a slurm job**, including one-off
  verification queries; never look nodes up by id outside an increment.
- `trash` is not installed here; use `mv` into the deprecated tree.

## 3. IGB Biocluster (`biologin`, lab partitions `mmli` and `cabbi`, stock `gpu`)

- Login node: the fewest possible commands. `squeue`/`sacct` state only, `sbatch`,
  `scancel`, `rsync`, the wandb sync wrapper. No `find`, `du`, `grep -r`, `wandb.Api()`,
  python, or repeated polls. A log that must be read is bounded (`tail -c 20000`).
- **Partitions:** `mmli` is ours (zero measured wait, 4-GPU whole-node jobs); `cabbi` is
  shared and holds other users' 15- and 30-day jobs (measured p75 wait 45 h, p90 109 h);
  `gpu` has six A40s and an admin cap of **three GPU jobs at once for us**, so every array
  there is `--array=<range>%3` and all our gpu-partition jobs count toward the three.
- **`--time` on cabbi is at most `5-00:00:00`** (user rule, not enforced by the scheduler).
  A round that needs longer goes to `gpu` or is split into checkpoint-continuation legs.
- `DATA_ROOT=/home/a-m/mjvolk3/scratch/torchcell`; singularity `rockylinux_9.sif` + conda.
  Dataloader workers: several (local disk, the risk is starvation), the opposite of Delta.
- Interactive work gets off the login node with `srun --pty -p mmli --gres=gpu:1 /bin/bash`.
- The fleet across clusters is read with
  `experiments/019-simb-multimodal/scripts/campaign_status.py`, one ssh per cluster.

## 4. NCSA Delta (`dt-login01`, Duo-gated)

- **The Delta strategy is parallelism** (user, 2026-10-01: "trying to get the speed up
  through parallelization... on Delta this will most often be our strategy"). Delta has
  about 200 four-GPU nodes and a long per-job wait, so a campaign's wall time is one queue
  wait plus one run when every job is eligible at submission, and N queue waits when they
  are chained or staggered. Submit the whole batch at once, with no dependencies and no
  `--begin`, and let the scheduler spread it; never trade eligibility for tidiness.
- **Access from this harness goes through the user's open ssh in tmux pane
  `torchcell:8.3`.** Short commands: `tmux send-keys -t torchcell:8.3 'echo START; <cmd>;
  echo END' Enter`, then `tmux capture-pane -p -t torchcell:8.3 -S -200 | sed -n
  '/START/,/END/p'`. Long commands fail ("not in a mode"): write the text to the scratchpad,
  `tmux load-buffer <file>`, `tmux paste-buffer -d -t torchcell:8.3`, `send-keys Enter`.
- **Always, before anything else on Delta, run `accounts` and read recent `sacct`** (user
  rule, 2026-10-02). The balance says what each account can still carry and the recent jobs
  say which account has been burning its fairshare, how long our runs actually take, and
  what failed, all of which the submission depends on:

  ```bash
  accounts
  sacct -u $USER -S $(date -d '7 days ago' +%F) -X -o JobID,JobName%40,Account,Partition,State,Elapsed,Timelimit,Start -n | tail -60
  ```

  Quote the balance and the measured Elapsed in the plan, not a remembered number. `bbtp`
  is gone; the default is `bfjt-delta-gpu`; `bbhh` is overdrawn; `bbub` is low. All four GPU
  accounts can submit to every GPU partition (A40x4, A100x4, A100x8, H200x8, MI100x8); the
  H200 rate is higher. There is no DeltaAI allocation on any of these projects (it would
  show as a separate `accounts` line). Charge is in service units per GPU-hour.
- **Priority is fairshare, and fairshare is per ACCOUNT.** Live scheduler config read
  2026-10-02: Fair Tree with weights fairshare 10,000, age 1,000 (maxes after 2 days),
  partition 1,000, job size and QOS 0; usage decays with a one-day half-life. The account is
  ranked first, then the user inside it, so burning one account down (bfjt after the 025
  rounds: fairshare 0.064, about 640 points) does not touch the others (bbub and bflt about
  3,000 points, the head of the gpuA40x4 queue was 2,336). **Spread a large batch across the
  accounts we may charge**, and re-account a pending job without resubmitting:
  `scontrol update JobId=<id> Account=<account>` (priority recalculates within minutes).
  Balances are per account (a 20 h A40 node job is about 40 SU), and bflt and bgcg are
  other projects' awards, so charging training to them needs the PI's consent; bbub
  (multimodal ML) is the natural second account for 025. Hypothesis, untested: an account
  that carries a few node-days of our usage drops toward bfjt's share within days.
- **Partitions and measured waits** (all users, 7 days to 2026-10-01, 4-GPU jobs; re-measure
  with `python3 ~/qwait.py <since> gpuA40x4,gpuA100x4 -a` on the login node before quoting):

  | partition, time limit | median wait | p75 | p90 |
  |---|---|---|---|
  | gpuA40x4, 20 h or more | 23.6 h | 46 h | 79 h |
  | gpuA40x4, 12 h or less | 7.3 h | 11 h | 32 h |
  | gpuA100x4, 20 h or more | 47.7 h | 85 h | 227 h |
  | gpuA100x4, 12 h or less | 0.2 h | 7 h | 28 h |
  | gpuA40x4-preempt, any | 1.6 h | 3 h | 14 h |

  So a run that fits in 12 h belongs in the 12 h class, and an A100x4 12 h probe of a 025
  config is the untested lever. `gpuA40x4` gives 32 CPUs per 4 GPUs; the partition limit is
  48 h.
- **Read `experiments/019-simb-multimodal/scripts/delta_grid_common.sh` before writing any
  Delta launcher.** `DATA_ROOT=/scratch/bbub/mjvolk3/torchcell` (not `/work/hdd`); the repo
  is `/projects/bbub/mjvolk3/torchcell` and jobs run from a detached worktree under
  `/scratch/bbub/mjvolk3/torchcell.worktrees/`; call the env python directly
  (`/work/hdd/bbub/miniconda3/envs/torchcell/bin/python`, no `conda activate`);
  `PYTHONUNBUFFERED=1`; DataLoader workers 0 (spawned workers re-import from Lustre and
  stall in `cl_sync_io_wait`); parent import off Lustre is about 54 min cold; `sbatch
  --test-only` before spending anything; a preflight that fails in seconds when a path is
  missing (`Neo4jCellDataset` would otherwise try to rebuild with no Neo4j).
- **The 025 launcher stages the 554 GB LMDB to node-local NVMe** (about 30 min at 290 MB/s;
  reading it from /scratch costs 63 min per epoch against 19). Submissions are not
  staggered for it (`delta_submit_round2.sh`, generated by `graph_reg_round2_plan.py`,
  submits all runs eligible at once); if stage-in contention is ever suspected, measure
  the copy time of concurrent starts from the job logs before adding any delay.
- Measured 025 wall times: 30 epochs of ctrl_013 on 4 A40s take 13.4 to 19.6 h (11 round-1b
  runs), so the 24 h limit is honest and 12 h is not reachable on A40s.
- `WANDB_MODE=offline`; sync from the login node afterwards.

## 5. Checklists

**Before submitting**

- [ ] The launcher and submitter are committed in `experiments/<id>/scripts/`, and the job
      root is a pinned snapshot or detached worktree at a recorded commit.
- [ ] On Delta, `accounts` and the 7-day `sacct` were run and read first: balance, which
      account is burned down, measured Elapsed, recent failures.
- [ ] Account chosen and its balance read (Delta); partition cap respected (IGB `gpu` %3,
      cabbi 5 days; GilaHyper allowed cards and the half-node cap).
- [ ] `-J <exp>-<round>w<N>-<stage>`, `--time` from measured wall time, `--mem` and
      `--cpus-per-task` from measured peaks, all on the sbatch line.
- [ ] No `--dependency` unless it is a checkpoint continuation that resumes the W&B run;
      no `--begin` stagger; the whole batch is eligible at submission.
- [ ] Every config override composes; every path the launcher reads exists on that cluster;
      `sbatch --test-only` passes; the real-size smoke passed at this commit.
- [ ] `WANDB_MODE=offline` where the compute node has no network.

**After submitting**

- [ ] `sacct -j <id> -o JobName,ReqMem,ReqCPUS,Timelimit -n` shows what was asked.
- [ ] `squeue -u $USER -o "%i %j %r %S"`: the reason is Priority, Resources, BeginTime or
      None, not Dependency.
- [ ] The job id, account, commit and submission log path are written into the
      experiment's dendron note, and the weekly child note gets the item.
- [ ] Polling is a `squeue` loop or a wakeup, not `pgrep`, and not a loop on an IGB login
      node.
