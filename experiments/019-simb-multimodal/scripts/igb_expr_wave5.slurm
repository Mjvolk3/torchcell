#!/bin/bash
# experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
# [[experiments.019-simb-multimodal.scripts.igb_expr_wave5]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/igb_expr_wave5
#
# IGB half of WAVE 5. Resources come from the sbatch COMMAND LINE (CLI beats in-file
# directives), because mmli and cabbi need different partitions, memory and packing:
#
#   # canary -- prove the container + env + data path before committing a wave. ~5 min.
#   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=8 --mem=60g --time=00:30:00 \
#       --array=0 --export=ALL,W5_STAGE=canary \
#       experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
#
#   # H2 on mmli -- 4 x A100 80GB, 2 runs per GPU, 24 runs over 3 rounds.
#   # --cpus-per-task=16 is load-bearing: it is what gives 7 dataloader workers per run.
#   sbatch -p mmli --gres=gpu:1 --cpus-per-task=16 --mem=100g --time=20:00:00 \
#       --array=0-11%4 --export=ALL,W5_STAGE=h2 \
#       experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
#
#   # probes on cabbi -- 2 free RTX6000, ONE run per GPU, 8 runs over 4 rounds.
#   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=8 --mem=60g --time=20:00:00 \
#       --array=0-7%2 --export=ALL,W5_STAGE=probe \
#       experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
#
# PACKING DIFFERS BY PARTITION BECAUSE VRAM DOES, and this is not a tuning choice.
# One run measured 20,026 MiB. An A100 80GB therefore takes 3 (60 GB, ~20 GB headroom).
#
# CORRECTION 2026-08-31: the cabbi cards are RTX 6000 **Ada**, 48 GB, not 24 GB -- read off
# W&B device metadata on compute-3-3 ("memory_total": 48305799168). The earlier claim that
# cabbi "takes exactly ONE -- packing two there would OOM" was wrong, and it halved the
# usable throughput of the larger partition. Two runs fit at 20 GB each with 8 GB headroom.
# Packing is still a per-stage decision rather than an automatic win: a campaign that needs
# LONG individual runs (the expression curve does not turn before ~10,000 epochs) is better
# served by one run per GPU, since packing shortens every run to raise aggregate throughput.
# Packing is
# uniform WITHIN a stage, which is what the protocol requires: a paired reference is only a
# valid denominator for arms that ran beside it under identical conditions. It is NOT uniform
# ACROSS stages, so an mmli arm is never compared to a cabbi arm. Each stage co-launches its
# own W_ref for exactly this reason (_007 pooled A100 with RTX 6000 Ada runs; means 0.0242
# vs 0.0182).
#
# NO REQUEUE CHAIN, DELIBERATELY. The _007 launchers derived `--time` from a hardcoded
# DEADLINE minus 120 min of slack and also passed `--deadline`; when a job then waited in
# queue longer than the slack, its TimeLimit no longer fit and Slurm killed it instantly in
# DEADLINE state at 0 s elapsed, silently -- that is what happened to 2323142. Both
# partitions are MaxTime=UNLIMITED, so a plain fixed walltime removes the whole failure mode.
#
#SBATCH -N 1
#SBATCH --ntasks=1
# NAME EVERY SUBMISSION `019-<round>w<N>-<stage>` (2026-09-20). This default is a
# placeholder: `squeue` showed five concurrent 019 arrays all reading `019-wave5-igb`, so
# nothing but the job id said which round or which half of a round a card was holding, and
# the log file names (`%x_%A_%a.out`) inherited the same ambiguity. Override it at submit:
#
#   sbatch -J 019-v17w2-locality -p gpu ... --array=6-11 --export=ALL,W5_STAGE=locality ...
#
# `<round>` is the config version (v16, v17), `<N>` counts SUBMISSIONS of that round on one
# cluster (a round split across partitions or resubmitted after a cancellation gets w1, w2,
# ...), and `<stage>` repeats W5_STAGE so the name survives being read without the launch
# line. `campaign_status.py` parses this and is the cross-cluster view; a job that predates
# the convention can be renamed in place with
# `scontrol update JobId=<id> JobName=019-v16w1-joint`.
# WALLTIME RULE (2026-09-21): every submission to `-p cabbi` passes `--time` of at most
# 5-00:00:00. The scheduler does not enforce it (it accepted a 10-day v15 job), so the
# submit line must. The `gpu` partition has no such limit.
# GPU PARTITION CAP (IGB HPC admin, 2026-09-21): the `gpu` partition has six A40s in total
# and we may hold at most THREE at once. Every array to `-p gpu` is submitted
# `--array=<range>%3`, and other gpu-partition jobs of ours count toward the three.
#SBATCH --job-name=019-wave5-igb
#SBATCH --signal=B:USR1@180
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out

set -euo pipefail

OUT_DIR=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output
mkdir -p "$OUT_DIR"

# PROJECT_ROOT resolves from THIS SCRIPT'S OWN LOCATION, so the code that runs is always the
# checkout the submitted script came from. It used to default to a named branch worktree,
# which meant submitting the script from anywhere else silently trained a different branch's
# source -- an un-attributable run that looks completely normal in the log.
#
# `torchcell` is pip-installed editable against /home/a-m/mjvolk3/projects/torchcell/torchcell,
# so PYTHONPATH must still point at PROJECT_ROOT or the import resolves to whatever branch
# that checkout happens to be on (memory: worktree-build-needs-pythonpath).
# W5_PROJECT_ROOT still overrides, for the deliberate case of running one checkout's script
# against another's source.
#
# BASH_SOURCE DOES NOT WORK UNDER sbatch, and this cost a canary (2368284, all three tasks
# FAILED at 27 s with "PROJECT_ROOT=/var has no torchcell/ package dir"). Slurm COPIES the
# batch script into its spool directory before executing it, so ${BASH_SOURCE[0]} is
# /var/spool/slurmd/job.../slurm_script and `../../..` resolves to /var -- never the
# checkout. The self-resolution is still correct when the script is run directly (bash
# path/to/script), which is how it was tested.
#
# SLURM_SUBMIT_DIR preserves the ORIGINAL intent exactly: it is the directory sbatch was
# invoked from, i.e. the checkout whose script was submitted. It is preferred when set,
# BASH_SOURCE is the fallback for a direct run, and W5_PROJECT_ROOT still overrides both.
_W5_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${W5_PROJECT_ROOT:-}" ]; then
  PROJECT_ROOT="$W5_PROJECT_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/torchcell" ]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="$(cd "$_W5_SELF/../../.." && pwd)"
fi
if [ ! -d "$PROJECT_ROOT/torchcell" ]; then
  echo "FATAL: PROJECT_ROOT=$PROJECT_ROOT has no torchcell/ package dir." >&2
  echo "       Submit this script from inside a torchcell checkout, or set W5_PROJECT_ROOT." >&2
  exit 1
fi
# The container lives in the main checkout and is untracked, so it is referenced by absolute
# path rather than expected inside the worktree.
SIF="${W5_SIF:-/home/a-m/mjvolk3/projects/torchcell/rockylinux_9.sif}"
ARM_SH="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh"

STAGE="${W5_STAGE:?set W5_STAGE=canary|h2|probe|long|wave6 via --export=ALL,W5_STAGE=...}"

# SEEDS ARE INIT-ONLY. cgt_expr_011 pins data_module.split_seed=0, so every run reads the
# SAME index_seed_0.json and these vary weight initialization only. That also removes the old
# constraint that seeds be drawn from {0,1,2,42}: concurrent tasks can no longer race to
# build a split cache, because there is only ever one split.
# Per-run override sets for the scaling sweeps, indexed like JOBS; empty for other stages.
SWEEP_OV=()
case "$STAGE" in
  # H2 -- does mixing genes AFTER the perturbation tell other genes how to express? The
  # encoder runs on the wildtype graph at batch 1, so H_genes is strain-independent and
  # nothing routes between gene tokens once the perturbation is applied. 9 paired replicates
  # per arm, because the prior evidence (E0_perceiver_on -0.0302, GEARS_crossgene -0.0243)
  # was n=1 on seed 0 alone and carries no significance.
  # PACKING IS 2, NOT 3, AND THAT IS A MEASURED CORRECTION. The first mmli submission
  # (2324287) ran 3 runs/GPU with --cpus-per-task=12, giving each run only (12-2)/3 = 3
  # dataloader workers against GilaHyper's 7. After 14 minutes all 12 runs were at epoch 1-2
  # while GH reached ~23 and cabbi (1 run/GPU, 6 workers) reached 5 -- and the A100s read 0%
  # utilization at 25 GB, i.e. idle waiting on data rather than compute-bound. So the limit was
  # never VRAM (3 x 20 GB fits an 80 GB card comfortably); it was worker starvation, made worse
  # by LMDB reads over GPFS. At that rate 400 epochs x 3 rounds needed ~50 h against a 20 h
  # walltime. Submit with --cpus-per-task=16 so this yields (16-2)/2 = 7 workers, matching GH.
  #
  # X_mix64 IS DROPPED to buy the replicates back. It was a capacity variant of X_mix, whereas
  # W_ref / X_mix / X_mix_rezero are the actual H2 question: is the channel useful, and does
  # "not useful" separate from "never opened" (rezero starts it closed). 3 arms x 8 replicates
  # = 24 runs at 2/GPU = 3 rounds, which fits.
  h2)     ARMS=(W_ref X_mix X_mix_rezero); SEEDS=(0 1 2 3 4 5 6 7)
          RUNS_PER_GPU=2
          EXTRA_OV="trainer.max_epochs=300 trainer.train_eval_every=5" ;;
  # PROBES -- lr below the frozen 3e-4 (the _007 top 12 cluster at 1e-4 while the wave-3
  # ladder only swept 3e-4/1e-3/3e-3), and dropout off on a model that underfits.
  probe)  ARMS=(W_ref P_lr1e4 P_lr3e5 H1_nodrop); SEEDS=(0 1)
          RUNS_PER_GPU=1
          EXTRA_OV="trainer.max_epochs=400 trainer.train_eval_every=5" ;;
  # LONG -- the H1 question taken to its limit on ONE A100. The 1500-epoch GH runs showed
  # (a) eval-mode TRAIN pearson still rising at 0.64-0.72 with max == last in 4 of 4 runs, and
  # (b) val pearson dipping around epoch 200-300 and then climbing past its early peak to a
  # project-best 0.1980 at epoch 1367. Neither had plateaued, so the cap was still the binding
  # constraint rather than the model. 10000 epochs is deliberately far past where anything is
  # expected to move; it can be cancelled the moment the curves flatten or turn over, and
  # ModelCheckpoint now keeps best-by-loss AND best-by-metric AND last, so cancelling costs
  # nothing.
  #
  # Reference for reading the train side: the replicate-based expression ceiling is 0.775
  # (mean_g sqrt(rho_g), notes/experiments.019-simb-multimodal.scripts.expression_ceiling_replicate).
  # traineval already sits at 0.64-0.72, i.e. the model is approaching the point where it fits
  # TRAIN as well as the experiment reproduces itself -- past that, additional train fit is
  # fitting measurement noise by construction.
  #
  # train_eval_every=10 rather than 5: the eval-mode pass costs ~65 s on its own (it sweeps
  # 4074 train records against validation's 534), so at every=5 it would add ~13 s to every
  # epoch and cost ~36 h across 10000 of them. At every=10 that halves and still yields 1000
  # curve points.
  long)   ARMS=(W_ref); SEEDS=(0 1)
          RUNS_PER_GPU=2
          EXTRA_OV="trainer.max_epochs=10000 trainer.train_eval_every=10" ;;
  # WAVE 6 -- the pair-term rank ladder plus regularization, 12 arms, ONE seed, no replicates.
  # 3 runs/GPU x 4 A100 = 12 slots. Measured 23.51 s/epoch at this packing and batch 32
  # (jobs 1431-1434), so a 48 h walltime reaches ~7,350 of the 10,000-epoch cap -- the runs
  # are EXPECTED to hit the wall, which is fine because ModelCheckpoint retains best-by-loss,
  # best-by-metric and last. The noise floor is carried from wave 5's measured across-init
  # sd of ~0.006 (n=8, fixed split) rather than spent on in-wave replicates.
  wave6)  ARMS=(V_ref V_sink V_basis16 V_basis32 V_basis64 V_film V_hadamard V_hadamard_add
                V_drop2 V_drop3 V_wd1e4 V_wd1e2); SEEDS=(0)
          RUNS_PER_GPU=3; STAGE_CONFIG=cgt_expr_012
          EXTRA_OV="" ;;
  # ---------------------------------------------------------------- LAUNCH (2026.08.31)
  # The objective round. Three challenger heads x 3 replicates = 9 runs at 2/GPU = 5 tasks,
  # scored against the eight existing quantile replicates rather than a co-launched
  # reference. That is the ONE case where an out-of-wave reference is valid: those eight runs
  # share this config, this budget and this hardware class, and they are the quantity whose
  # spread (sd 0.0222) sizes the whole design.
  #
  # 9,900 IS THE TARGET, NOT THE EXPECTATION. Measured packed wall clock is ~46.7 s/epoch on
  # A100 and ~47.4 on Ada, so a 5-day job reaches ~9,100-9,250 epochs and the runs are
  # EXPECTED to hit the wall. That is fine and it is why the declared scoring window is
  # roll_max over epochs <= 9,000, which every run clears: ModelCheckpoint retains
  # best-by-loss, best-by-metric and last.
  #
  # train_eval_every=10 matches the incumbent's own long run, so the eval-mode train curve is
  # comparable to it rather than to a differently-instrumented arm.
  launch) ARMS=(Q_crps Q_laplace Q_point); SEEDS=(0 1 2)
          RUNS_PER_GPU=2; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.max_epochs=9900 trainer.train_eval_every=10" ;;
  # ---------------------------------------------------------------- PEARSON (2026.09.06)
  # The metric-aligned objective: train on 1 - per-feature Pearson (Q_pearson) or on that
  # plus an MSE anchor (Q_pearson_mse). Two arms x 2 seeds = 4 runs at 2/GPU = 2 tasks, sized
  # to the two cabbi cards free on 2026-09-06. Same budget, packing and scoring window as
  # `launch` so the runs are read against the same n=8 quantile baseline (roll_max over
  # epochs <= 9,000). Why this exists is in gh_expr_008_arm.sh at the Q_pearson arm.
  #
  #   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=8 --mem=60g --time=5-00:00:00 \
  #       --array=0-1 --export=ALL,W5_STAGE=pearson,TORCHCELL_SOURCE_GIT_HASH=$H,\
  #   TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  # SEED 2 ADDED 2026-09-06 after tasks 0-1 (seeds 0-1, job 2378262) were already running:
  # a third cabbi card freed up. JOBS is seed-major, so tasks 0-1 are unchanged and task 2
  # is exactly (Q_pearson:2, Q_pearson_mse:2). Submit it alone with --array=2.
  pearson) ARMS=(Q_pearson Q_pearson_mse); SEEDS=(0 1 2)
          RUNS_PER_GPU=2; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.max_epochs=9900 trainer.train_eval_every=10" ;;
  # PEARSON_B64 -- the same pure loss with each gene's per-batch correlation estimated over
  # 64 strains instead of 32. ONE run per card: batch 64 roughly doubles activation memory
  # over the measured ~20 GB at batch 32, and 2 x ~40 GB does not fit a 48 GB Ada. Two seeds
  # on the two remaining free cabbi cards (2026-09-06).
  #
  #   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=8 --mem=60g --time=5-00:00:00 \
  #       --array=0-1 --export=ALL,W5_STAGE=pearson_b64,... igb_expr_wave5.slurm
  pearson_b64)
          ARMS=(Q_pearson_b64); SEEDS=(0 1)
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.max_epochs=9900 trainer.train_eval_every=10" ;;
  canary_pearson)
          ARMS=(Q_pearson Q_pearson_mse); SEEDS=(0)
          RUNS_PER_GPU=2; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.fast_dev_run=true" ;;
  # ---------------------------------------------------------------- LISTMLE (2026.09.08)
  # The ranking objective (Q_listmle, Q_listmle_mse), 2 arms x 2 seeds = 4 runs at 2/GPU =
  # 2 tasks. max_epochs 6,000 rather than 9,900: every packed five-day task so far has died
  # of host memory between days 3.5 and 4.6 (61 GB against 60 GB requested), the Pearson
  # round's peaks came by epoch 3,200, and 6,000 packed epochs is about three days. Scored
  # by roll_max at matched budgets against the Pearson round and the v9 long-budget arms.
  #
  #   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=8 --mem=60g --time=5-00:00:00 \
  #       --array=0-1 --export=ALL,W5_STAGE=listmle,TORCHCELL_SOURCE_GIT_HASH=$H,\
  #   TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  listmle) ARMS=(Q_listmle Q_listmle_mse); SEEDS=(0 1)
          RUNS_PER_GPU=2; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.max_epochs=6000 trainer.train_eval_every=10" ;;
  # LISTMLE_B64 -- the pure ranking loss with 64-strain lists, one run per card, on the
  # idle A40s of the `gpu` partition (2026-09-08; cabbi had two free cards, taken by the
  # packed stage above).
  #
  #   sbatch -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=60g --time=5-00:00:00 \
  #       --array=0-1 --export=ALL,W5_STAGE=listmle_b64,... igb_expr_wave5.slurm
  listmle_b64)
          ARMS=(Q_listmle_b64); SEEDS=(0 1)
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.max_epochs=6000 trainer.train_eval_every=10" ;;
  canary_listmle)
          ARMS=(Q_listmle Q_listmle_mse); SEEDS=(0)
          RUNS_PER_GPU=2; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.fast_dev_run=true" ;;
  # RESUME -- the incumbent past its own 9,900. Not an arm: one run, one GPU pair, testing
  # only whether THIS draw's curve turns. With sd 0.0222 a single continuation cannot
  # establish saturation, and the plan says so.
  #
  # max_epochs IS ABSOLUTE IN LIGHTNING, so 19,000 means "9,100 more", and the train script
  # raises if max_epochs <= the checkpoint's epoch rather than exiting instantly with a
  # confusing success. W5_RESUME_CKPT must be an absolute path readable on the compute node.
  #
  # PARAMETERIZED 2026.09.01, because restarting a casualty is now a routine operation rather
  # than a one-off. The dataloader file-descriptor bug (commit f1bfa95b) killed two DIFFERENT
  # runs on cabbi a day into their budgets: the incumbent continuation AND launch arm Q_point
  # seed 2. Restarting the second from its checkpoint was impossible while this stage
  # hardcoded W_ref/seed 0, so the arm and seed now default to the incumbent's and are
  # overridable. Defaults reproduce the original invocation exactly.
  #
  #   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=8 --mem=60g --time=5-00:00:00 --array=0 \
  #     --export=ALL,W5_STAGE=resume,W5_RESUME_ARM=Q_point,W5_RESUME_SEED=2,\
  # W5_RESUME_EPOCHS=9900,W5_RESUME_CKPT=/abs/path/last.ckpt,... igb_expr_wave5.slurm
  resume) ARMS=("${W5_RESUME_ARM:-W_ref}"); SEEDS=("${W5_RESUME_SEED:-0}")
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.max_epochs=${W5_RESUME_EPOCHS:-19000} trainer.train_eval_every=10 trainer.resume_ckpt_path=${W5_RESUME_CKPT:?set W5_RESUME_CKPT=/abs/path/to/last.ckpt for the resume stage}" ;;
  # RESUME_MANY -- restart a whole PACKED round from per-run checkpoints. `resume` above
  # carries ONE checkpoint and therefore one run per GPU, which cannot restore a round that
  # was running two runs per card: mmli held 8 objective-round runs on 4 A100s, and
  # relaunching those unpacked would have halved the round to two sequential rounds of four.
  #
  # The checkpoint is per RUN, not per stage, so it cannot ride in EXTRA_OV like every other
  # override. It comes from a MAP FILE instead, one `arm seed ckpt_path` record per line,
  # blank and #-comment lines ignored. A file rather than a packed env var on purpose: it is
  # an artifact that can be read back later to say exactly which run resumed from which
  # checkpoint, which is the only record of where a restarted round came from.
  #
  # Run order in the file IS the packing order, so lines 1-2 share the first GPU, 3-4 the
  # second, and so on. Keep co-resident runs different arms, as the seed-major builder below
  # does, rather than putting two replicates of one arm on the same card.
  #
  #   sbatch -p mmli --gres=gpu:1 --cpus-per-task=16 --mem=100g --time=5-00:00:00 \
  #     --array=0-3 --export=ALL,W5_STAGE=resume_many,W5_RESUME_MAP=/abs/map.tsv,\
  # W5_RESUME_EPOCHS=9900,... igb_expr_wave5.slurm
  #
  # CONTINUING A FINISHED ROUND (2026.09.23, the joint round v16). The map may carry a FOURTH
  # column, the W&B id of the run the checkpoint came from; it is written into the new run's
  # config as `wandb.resumed_from`, so the two segments of one trajectory can be stitched by
  # the readout. The continuation is a NEW W&B run on purpose: the train script can also
  # continue the SAME run id (`wandb.resume_id`), but that path has never been exercised in
  # offline mode, where `wandb sync` of a second segment with the same id could overwrite the
  # first segment's history rather than append to it. `W5_RESUME_CONFIG` names the round's
  # config, since the arm's overrides and the round's epoch budget both compose from it.
  resume_many)
          MAP="${W5_RESUME_MAP:?set W5_RESUME_MAP=/abs/path/to/map.tsv for resume_many}"
          [[ -r "$MAP" ]] || { echo "W5_RESUME_MAP is not readable: $MAP" >&2; exit 1; }
          RESUME_JOBS=(); RESUME_CKPTS=(); RESUME_FROM=()
          while read -r _arm _seed _ck _from _rest; do
            [[ -z "${_arm:-}" || "${_arm:0:1}" == "#" ]] && continue
            [[ -n "${_seed:-}" && -n "${_ck:-}" ]] || {
              echo "malformed W5_RESUME_MAP line for arm '$_arm'" >&2; exit 1; }
            RESUME_JOBS+=("$_arm:$_seed"); RESUME_CKPTS+=("$_ck"); RESUME_FROM+=("${_from:-}")
          done < "$MAP"
          [[ "${#RESUME_JOBS[@]}" -gt 0 ]] || { echo "W5_RESUME_MAP has no records" >&2; exit 1; }
          ARMS=(); SEEDS=()
          RUNS_PER_GPU="${W5_RESUME_PACK:-2}"; STAGE_CONFIG="${W5_RESUME_CONFIG:-cgt_expr_v9_mask}"
          EXTRA_OV="trainer.max_epochs=${W5_RESUME_EPOCHS:-9900} trainer.train_eval_every=10" ;;
  # ---------------------------------------------------------------- MECHANISM (2026.09.01)
  # Pair-term and READOUT mechanisms, replicated, on the four cabbi cards the objective
  # round left free. 4 arms x 2 seeds = 8 runs at 2/GPU = 4 tasks.
  #
  # BUDGET. Measured cabbi throughput is ~29.5 s/epoch solo (from point seed 2 in the
  # objective round), so ~45 s/epoch packed at the measured 1.53x. 8,500 epochs is then
  # ~4.4 days, inside a 5-day wall with margin, and the arms COMPLETE rather than being
  # killed mid-run. Scored as roll_max over epochs <= 8,000, which every run clears.
  #
  # THIS IS A SCREEN, NOT A DECISION, and the arithmetic says so plainly: at sd 0.0222 a
  # 2-versus-2 paired contrast against the in-round R_ref resolves only ~0.062. It can
  # therefore find a LARGE readout effect and nothing subtler. Any arm that moves gets
  # replicated before it is believed -- which is exactly what wave 6 failed to do.
  mech)   ARMS=(R_ref R_basis64 R_pergene R_pergene_basis64); SEEDS=(0 1)
          RUNS_PER_GPU=2; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.max_epochs=8500 trainer.train_eval_every=10" ;;
  # Canary for the mechanism round: R_pergene is NEW CODE (a per-gene output weight in
  # PerGeneHead), so it must be proven to construct and step before four days are committed.
  canary_mech)
          ARMS=(R_pergene R_pergene_basis64); SEEDS=(0)
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.fast_dev_run=true" ;;
  # ---------------------------------------------------------- PER-GENE REPLICATES (2026.09.09)
  # RR_ref vs RR_pergene (same configs as R_ref / R_pergene, see gh_expr_008_arm.sh), fresh
  # seeds 2-4 at 1,000 epochs, 2 runs per card so each task is one seed's PAIR on one card:
  # the contrast is paired within task, and the card is a block. 1,000 epochs is 12 h for
  # the reference and about 16 h for the per-gene arm at the packed rates measured in the
  # mechanism round (84 and 64 epochs per hour), so a 30-hour wall clears it and the host
  # memory growth that killed the five-day packed tasks (61 GB by day 3.5) is not reached.
  # Three tasks: two on the cabbi cards freed by the batch-64 Pearson runs (2026-09-09), the
  # third when the anchored-Pearson task ends. Submit task 2 alone with --array=2.
  #
  #   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=8 --mem=60g --time=1-06:00:00 \
  #       --array=0-1 --export=ALL,W5_STAGE=pergene_rep,TORCHCELL_SOURCE_GIT_HASH=$H,\
  #   TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  pergene_rep)
          ARMS=(RR_ref RR_pergene); SEEDS=(2 3 4)
          RUNS_PER_GPU=2; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.max_epochs=1000 trainer.train_eval_every=10" ;;
  canary_pergene_rep)
          ARMS=(RR_ref RR_pergene); SEEDS=(2)
          RUNS_PER_GPU=2; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.fast_dev_run=true" ;;
  # ------------------------------------------------------------ READOUT ROUND (2026.09.10)
  # Eight readouts on the E_full input under pinball (conf/cgt_expr_v12_head.yaml), 4 seeds,
  # FOUR runs per 48 GB card (W&B system metrics: ~9 GB per batch-32 run), seed-major so a
  # card holds four arms of one seed. Eight tasks: two on cabbi, six on the `gpu` A40s.
  # Budget 1,400 epochs (v11's) inside a two-day wall. Host memory: two packed runs read
  # 21 GB at day 1.9, so four are given 110-120 GB.
  #
  #   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=12 --mem=120g --time=2-00:00:00 \
  #       --array=0-1 --export=ALL,W5_STAGE=head,TORCHCELL_SOURCE_GIT_HASH=$H,\
  #   TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  #   sbatch -p gpu --gres=gpu:1 --cpus-per-task=16 --mem=110g --time=2-00:00:00 \
  #       --array=2-7 --export=ALL,W5_STAGE=head,TORCHCELL_SOURCE_GIT_HASH=$H,\
  #   TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  head)   ARMS=(H_ref H_linear H_pergene H_gears H_basis64 H_pergene_basis64 H_concat H_state)
          SEEDS=(0 1 2 3)
          RUNS_PER_GPU=4; STAGE_CONFIG=cgt_expr_v12_head
          EXTRA_OV="" ;;
  # ------------------------------------------------------------- SPLIT ROUND (2026.09.13)
  # H_ref and H_concat on four partitions plus split 0 at 90/10 (conf/cgt_expr_v13_split.yaml).
  # The run list is EXPLICIT rather than an arm x seed product, because the arm carries the
  # split and each card holds one split's two arms for two seeds: every ref/concat contrast
  # is paired within card and seed. Six tasks, four runs each, all on the `gpu` A40s.
  # 6,000 epochs; --cpus-per-task=12 gives (12-2)/4 = 2 workers per run and 120 GB per task
  # is half a 250 GB node, both chosen to hold host memory below what killed the five-day
  # packed tasks. No walltime tuning: the partition has no limit and the round should end
  # by finishing.
  #
  #   H=$(git rev-parse HEAD); D=$(git diff HEAD | sha256sum | cut -d" " -f1)
  #   sbatch -p gpu --gres=gpu:1 --cpus-per-task=12 --mem=120g --time=10-00:00:00 \
  #       --array=0-5 --export=ALL,W5_STAGE=split,TORCHCELL_SOURCE_GIT_HASH=$H,\
  #   TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  split)  JOBS=(
            V_ref_s0:0 V_concat_s0:0 V_ref_s0:1 V_concat_s0:1
            V_ref_s0:2 V_concat_s0:2 V_ref_s0:3 V_concat_s0:3
            V_ref_s0_90:0 V_concat_s0_90:0 V_ref_s0_90:1 V_concat_s0_90:1
            V_ref_s1:0 V_concat_s1:0 V_ref_s1:1 V_concat_s1:1
            V_ref_s2:0 V_concat_s2:0 V_ref_s2:1 V_concat_s2:1
            V_ref_s3:0 V_concat_s3:0 V_ref_s3:1 V_concat_s3:1
          )
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=4; STAGE_CONFIG=cgt_expr_v13_split
          EXTRA_OV="" ;;
  # ------------------------------------------------------------- PROTEOME ROUND (2026.09.15)
  # The split round's design on the Messner knockout proteome (conf/cgt_expr_v14_proteome.yaml):
  # P_ref and P_concat on split seeds 0 to 3, two init seeds, four runs per card, one split
  # per card. Four tasks on cabbi (RTX 6000 Ada, 48 GB); the fifth free card is left for the
  # other group's pending job. 2,000 epochs at ~2.9x the steps per epoch of Kemmeren; the
  # dataset must be mirrored to IGB first (sync_igb_fig3_core.sh fig3_proteome).
  #
  #   H=$(git rev-parse HEAD); D=$(git diff HEAD | sha256sum | cut -d" " -f1)
  #   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=12 --mem=120g --time=10-00:00:00 \
  #       --array=0-3 --export=ALL,W5_STAGE=proteome,TORCHCELL_SOURCE_GIT_HASH=$H,\
  #   TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  proteome) JOBS=(
            P_ref_s0:0 P_concat_s0:0 P_ref_s0:1 P_concat_s0:1
            P_ref_s1:0 P_concat_s1:0 P_ref_s1:1 P_concat_s1:1
            P_ref_s2:0 P_concat_s2:0 P_ref_s2:1 P_concat_s2:1
            P_ref_s3:0 P_concat_s3:0 P_ref_s3:1 P_concat_s3:1
          )
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=4; STAGE_CONFIG=cgt_expr_v14_proteome
          EXTRA_OV="" ;;
  # ------------------------------------------------------------- WEIGHT-DECAY ROUND (2026.09.15)
  # Strong AdamW weight decay (1e-2, 1e-1) against the incumbent's 1e-8 on split seeds 1 and
  # 2, two init seeds, 6,000 epochs (conf/cgt_expr_v15_wd.yaml). Twelve runs on three cabbi
  # cards; the contrast is paired within split and seed, and the run list is laid out so
  # each card mixes arms rather than holding one arm.
  #
  #   H=$(git rev-parse HEAD); D=$(git diff HEAD | sha256sum | cut -d" " -f1)
  #   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=12 --mem=120g --time=10-00:00:00 \
  #       --array=0-2 --export=ALL,W5_STAGE=wd,TORCHCELL_SOURCE_GIT_HASH=$H,\
  #   TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  wd)     JOBS=(
            W_ref_s1:0 W_wd1e2_s1:0 W_wd1e1_s1:0 W_ref_s1:1
            W_wd1e2_s1:1 W_wd1e1_s1:1 W_ref_s2:0 W_wd1e2_s2:0
            W_wd1e1_s2:0 W_ref_s2:1 W_wd1e2_s2:1 W_wd1e1_s2:1
          )
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=4; STAGE_CONFIG=cgt_expr_v15_wd
          EXTRA_OV="" ;;
  # ------------------------------------------------------------- JOINT ROUND (2026.09.17)
  # The proteome head with the transcriptome as a second per-gene head on the same trunk,
  # against the proteome-only reference AND the expression-only reference on the same
  # partition, split seeds 0 to 2, two init seeds, 500 epochs (conf/cgt_expr_v16_joint.yaml).
  # Eighteen runs as six tasks of three: the three arms of one (split, seed) share a card,
  # so every contrast is paired within card, split and seed. Tasks are split-major; send
  # any subset to whichever partition has cards (cabbi as v14 ends, the gpu A40s as v13
  # ends) with the same stage and a different --array.
  #
  #   H=$(git rev-parse HEAD); D=$(git diff HEAD | sha256sum | cut -d" " -f1)
  #   sbatch -p cabbi --gres=gpu:1 --cpus-per-task=12 --mem=120g --time=4-00:00:00 \
  #       --array=0-5 --export=ALL,W5_STAGE=joint,TORCHCELL_SOURCE_GIT_HASH=$H,\
  #   TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  joint)  JOBS=(
            J_ref_s0:0 J_expr_s0:0 J_joint_s0:0
            J_ref_s0:1 J_expr_s0:1 J_joint_s0:1
            J_ref_s1:0 J_expr_s1:0 J_joint_s1:0
            J_ref_s1:1 J_expr_s1:1 J_joint_s1:1
            J_ref_s2:0 J_expr_s2:0 J_joint_s2:0
            J_ref_s2:1 J_expr_s2:1 J_joint_s2:1
          )
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=3; STAGE_CONFIG=cgt_expr_v16_joint
          EXTRA_OV="" ;;
  # ------------------------------------------------ DECONFOUNDED JOINT ROUND (2026.09.27, v19)
  # The v16 question with the arms differing in the active loss terms only, on the both-label
  # rows, every head unmasked, twelve split seeds x one init seed, 1,200 epochs
  # (conf/cgt_expr_v19_joint_clean.yaml, which also carries the pre-registered test). 36 runs
  # as twelve tasks of three, the three arms of one partition per card. Tasks are
  # partition-major; queue the first six on cabbi (<= 5 days) and the last six on the gpu
  # partition under %3, chained after whatever of ours is already queued there, so the two
  # throttles never add. K_perm arrives as its own stage when the trainer can permute labels.
  #
  #   H=$(git rev-parse HEAD); D=$(git diff HEAD | sha256sum | cut -d" " -f1)
  #   sbatch -J 019-v19w1-joint_clean -p cabbi --gres=gpu:1 --cpus-per-task=12 --mem=120g \
  #       --time=3-00:00:00 --array=0-5 --export=ALL,W5_STAGE=joint_clean,\
  #   TORCHCELL_SOURCE_GIT_HASH=$H,TORCHCELL_SOURCE_DIFF_SHA256=$D \
  #       experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  #   sbatch -J 019-v19w1-joint_clean -p gpu --gres=gpu:1 --cpus-per-task=12 --mem=120g \
  #       --time=3-00:00:00 --array=6-11%3 --dependency=afterany:<pending gpu array> ... (same)
  joint_clean) JOBS=()
          for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
            JOBS+=("K_prot_s${s}:0" "K_expr_s${s}:0" "K_joint_s${s}:0")
          done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=3; STAGE_CONFIG=cgt_expr_v19_joint_clean
          EXTRA_OV="" ;;
  # ------------------------------------------- CONDITIONED CROSS-MODAL ROUND (2026.10.03, v20)
  # One modality revealed in full as input, the other predicted, and the same with another
  # strain's measurement revealed (conf/cgt_expr_v20_conditioned.yaml). 48 runs as twelve
  # tasks of four, the four arms of one partition per card; the unconditioned controls are
  # v19's K_expr and K_prot. Every task is eligible at submission: no dependency, no stagger.
  #
  #   H=$(git rev-parse HEAD); D=$(git diff HEAD | sha256sum | cut -d" " -f1)
  #   sbatch -J 019-v20w1-conditioned -p cabbi --gres=gpu:1 --cpus-per-task=16 --mem=160g \
  #       --time=<1.3x the canary-scaled wall time> --array=0-11 \
  #       --export=ALL,W5_STAGE=conditioned,TORCHCELL_SOURCE_GIT_HASH=$H,\
  #   TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  # ------------------------------------- v22 BATCH-128 CANDIDATE DEFAULT, MORE SEEDS (2026.10.07)
  # The v22 fast round on GilaHyper read batch 128 at the unscaled rate 3e-4 as +0.013 over
  # the batch-32 reference on split seeds 0 to 2 (3 of 3 above). This stage runs the same arm
  # on split seeds 4 to 11 so the contrast pairs with the relaunched v21 S_ref runs on Delta
  # (same config, same store, same split seeds, batch 32). Three runs per A40 (12.2 GB each),
  # splits held in memory, W&B project torchcell_019_expr_v22, tags as the v22 launcher.
  #
  #   H=$(git rev-parse HEAD); D=$(git diff HEAD | sha256sum | cut -d" " -f1)
  #   sbatch -J 019-v22w1-b128seeds -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=80g \
  #       --time=14:00:00 --array=0-2 --export=ALL,W5_STAGE=v22_b128_seeds,\
  #   TORCHCELL_SOURCE_GIT_HASH=$H,TORCHCELL_SOURCE_DIFF_SHA256=$D \
  #       experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  v22_b128_seeds) JOBS=()
          for s in 4 5 6 7 8 9 10 11; do JOBS+=("S_ref_s${s}:0"); done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=3; STAGE_CONFIG=cgt_expr_v21_small
          EXTRA_OV="trainer.max_epochs=1200 data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true wandb.project=torchcell_019_expr_v22" ;;
  # THE SCALING SWEEPS (2026-10-09, notes-tex/modeling/scaling-laws section 6): the data
  # axis and the parameter axis of the expression head on the both-label store, on the
  # fast recipe with the mask schedule on (the one v21 arm with zero collapses in twelve
  # seeds) at batch 128 and the unscaled rate. One run per card (RTX6000, 24 GB; batch 128
  # at width 90 took 12.2 GB on an A40), one task per run, split seeds 0 to 2 as the
  # replicates. Both sweeps share the 1,200-epoch budget, so the small data fractions
  # take fewer optimizer steps (one step per epoch at 1/16); the per-epoch validation
  # curve is what the readout reads, and the step mismatch is stated in the document.
  #   scale_d  train_fraction 1, 1/2, 1/4, 1/8, 1/16 of the 1,103 both-label training rows,
  #            nested (train_cgt_multitask.py `train_fraction`); validation and test untouched.
  #   scale_n  hidden width 45, 90, 180, 360 at six layers (nine heads divide all four),
  #            learning rate scaled as 1/width from 3e-4 at 90 (hypothesis, not mu-P: the
  #            width-180 arm of v22 collapsed at the width-90 rate). Width 360 is last so an
  #            out-of-memory there costs the sweep nothing else.
  #   sbatch -J 019-scale1w1-d -p cabbi --gres=gpu:1 --cpus-per-task=5 --mem=60g \
  #       --time=2-00:00:00 --array=0-14%3 --export=ALL,W5_STAGE=scale_d,... igb_expr_wave5.slurm
  #   sbatch -J 019-scale1w1-n ... --array=0-11%3 --export=ALL,W5_STAGE=scale_n,...
  scale_d) JOBS=(); SWEEP_OV=()
          for f in 1 0.5 0.25 0.125 0.0625; do for s in 0 1 2; do
            JOBS+=("S_mask_s${s}:0")
            SWEEP_OV+=("+data_module.train_fraction=${f} wandb.tags=[scaling,D_frac${f},cabbi,split${s}]")
          done; done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_expr_v21_small
          EXTRA_OV="trainer.max_epochs=1200 data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true wandb.project=torchcell_019_scaling" ;;
  # THE MORPHOLOGY ROUND v24 (2026-10-10, conf/cgt_morph_v24.yaml): can the model predict
  # single-deletion morphology (the 116 moving CalMorph features, read off the perturbed
  # CLS), and does revealed expression help. Arm-major order so the M_cls tasks write the
  # split caches for seeds 4 to 11 before a conditioned task of the same seed starts (the
  # IGB fig3_core mirror carries seeds 0 to 3 only). One run per card.
  #   sbatch -J 019-morph1w1-cls -p cabbi --gres=gpu:1 --cpus-per-task=5 --mem=60g \
  #       --time=1-12:00:00 --array=0-35%8 --export=ALL,W5_STAGE=morph_v24,... igb_expr_wave5.slurm
  #   sbatch -J 019-morph1w1-pool -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=80g \
  #       --time=1-12:00:00 --array=0-2%3 --export=ALL,W5_STAGE=morph_v24_pool,... igb_expr_wave5.slurm
  morph_v24) JOBS=("M_cls_s0:0" "CM_expr_s0:0" "CM_exprperm_s0:0")  # seed 0 first: every arm fails fast
          for arm in M_cls CM_expr CM_exprperm; do for s in 1 2 3 4 5 6 7 8 9 10 11; do
            JOBS+=("${arm}_s${s}:0")
          done; done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_morph_v24
          EXTRA_OV="trainer.max_epochs=1200 data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true" ;;
  # The proteome-revealed arms on the fig3_morph_proteome store (built 2026-10-10 from the
  # v1.6.2 worktree, mirrored to IGB). Seed 0 of each arm first, then arm-major so MP_cls
  # writes every split cache before a conditioned task of the same seed starts.
  #   sbatch -J 019-morph1w3-prot -p cabbi --gres=gpu:1 --cpus-per-task=5 --mem=60g \
  #       --time=1-12:00:00 --array=0-35%8 --export=ALL,W5_STAGE=morph_v24_prot,... igb_expr_wave5.slurm
  morph_v24_prot) JOBS=("MP_cls_s0:0" "CM_prot_s0:0" "CM_protperm_s0:0")
          for arm in MP_cls CM_prot CM_protperm; do for s in 1 2 3 4 5 6 7 8 9 10 11; do
            JOBS+=("${arm}_s${s}:0")
          done; done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_morph_v24
          EXTRA_OV="trainer.max_epochs=1200 data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true" ;;
  # The expression arms again with the pooled readout (see the arm script): 24 tasks.
  #   sbatch -J 019-morph1w3-exprpool -p cabbi ... --array=0-23%8 --export=ALL,W5_STAGE=morph_v24_expr_pool,...
  morph_v24_expr_pool) JOBS=()
          for arm in CMP_expr CMP_exprperm; do for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
            JOBS+=("${arm}_s${s}:0")
          done; done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_morph_v24
          EXTRA_OV="trainer.max_epochs=1200 data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true" ;;
  morph_v24_pool) JOBS=("M_pool_s0:0" "M_pool_s1:0" "M_pool_s2:0")
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_morph_v24
          EXTRA_OV="trainer.max_epochs=1200 data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true" ;;
  # BUDGET (2026-10-12, user): every v25 stage runs MORPH_EPOCHS epochs, default 500. The
  # first 22 cells at 1,200 epochs showed that the loss finishes learning by epoch 50,
  # an escape from the plateau happens by epoch 400 and the validation peak of an escaped
  # run sits at epochs 400 to 600 before it overfits, so 1,200 epochs was idle GPU. The
  # registered window of v25 is epochs 400 to 499 (morph_v25_readout.py); the 1,200-epoch
  # cells keep their 1,000 to 1,199 reading as a check that the two windows rank alike.
  # THE INDUCTIVE ROUND v25 (2026-10-11, project torchcell_019_morph_v25): the genotype-only
  # morphology grid on fig3_core. Readout {M_cls, M_pool, M_zs, M_xattn} x lr {3e-4, 1e-3,
  # 3e-3} x split seeds 0 to 3 (48 tasks, init seed 0), then the plateau-escape replicates
  # of M_cls (init seeds 1 and 2 at every lr and split, 24 tasks), then M_cls at lr 3e-3
  # with a 100-epoch warmup and cosine decay (4 tasks). 76 one-card tasks, no dependencies;
  # the split caches for seeds 0 to 3 are on the IGB mirror. Split seed 0 of every cell
  # comes first so each arm fails fast.
  #   sbatch -J 019-morph2w1-inductive -p cabbi --gres=gpu:1 --cpus-per-task=5 --mem=60g \
  #       --time=0-16:00:00 --array=0-75%8 --export=ALL,W5_STAGE=morph_v25,... igb_expr_wave5.slurm
  morph_v25) JOBS=(); SWEEP_OV=()
          for s in 0 1 2 3; do for lr in 3e-4 1e-3 3e-3; do for arm in M_cls M_pool M_zs M_xattn; do
            JOBS+=("${arm}_s${s}:0")
            SWEEP_OV+=("regression_task.optimizer.lr=${lr} wandb.tags=[v25,${arm},lr${lr},split${s},init0]")
          done; done; done
          for s in 0 1 2 3; do for lr in 3e-4 1e-3 3e-3; do for init in 1 2; do
            JOBS+=("M_cls_s${s}:${init}")
            SWEEP_OV+=("regression_task.optimizer.lr=${lr} wandb.tags=[v25,M_cls,lr${lr},split${s},init${init}]")
          done; done; done
          for s in 0 1 2 3; do
            JOBS+=("M_cls_s${s}:0")
            SWEEP_OV+=("regression_task.optimizer.lr=3e-3 regression_task.lr_scheduler.type=CosineAnnealingWarmupRestarts regression_task.lr_scheduler.first_cycle_steps=1200 regression_task.lr_scheduler.warmup_steps=100 regression_task.lr_scheduler.max_lr=3e-3 regression_task.lr_scheduler.min_lr=1e-5 wandb.tags=[v25,M_cls,lr3e-3,warmup100,split${s},init0]")
          done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_morph_v24
          EXTRA_OV="trainer.max_epochs=${MORPH_EPOCHS:-500} data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true wandb.project=torchcell_019_morph_v25" ;;
  # v25b (2026-10-11, same project): the levers the betaxanthin, trigenic and environment
  # campaigns found, transplanted. Loss: the 19-quantile pinball (the default) against MSE
  # (`dist=point`) on M_cls and M_zs at two lrs. Graph prior: the hard mask at layer 1
  # replaced by the KL graph regularization at lambda 1e-2 and 1 (010, 025 and 035 did not
  # train without it). Schedule: lr 1e-4 as the low control, and warm restarts (peak 1e-3,
  # 50-epoch warmup, 300-epoch cycles). A stacked cell: MSE + KL 1e-2 + lr 1e-3 on M_zs.
  # 36 one-card tasks, split seeds 0 to 3, init seed 0.
  #   sbatch -J 019-morph2w2-levers -p cabbi --gres=gpu:1 --cpus-per-task=5 --mem=60g \
  #       --time=0-16:00:00 --array=0-35%8 --export=ALL,W5_STAGE=morph_v25b,... igb_expr_wave5.slurm
  morph_v25b) JOBS=(); SWEEP_OV=()
          for s in 0 1 2 3; do for arm in M_cls M_zs; do for lr in 3e-4 1e-3; do
            JOBS+=("${arm}_s${s}:0")
            SWEEP_OV+=("multitask.dist=point regression_task.optimizer.lr=${lr} wandb.tags=[v25b,${arm},mse,lr${lr},split${s},init0]")
          done; done; done
          for s in 0 1 2 3; do for lam in 1e-2 1; do
            JOBS+=("M_cls_s${s}:0")
            SWEEP_OV+=("model.attention_mask.enabled=false model.graph_regularization.graph_reg_lambda=${lam} wandb.tags=[v25b,M_cls,graphkl${lam},lr3e-4,split${s},init0]")
          done; done
          for s in 0 1 2 3; do
            JOBS+=("M_cls_s${s}:0")
            SWEEP_OV+=("regression_task.optimizer.lr=1e-4 wandb.tags=[v25b,M_cls,lr1e-4,split${s},init0]")
            JOBS+=("M_cls_s${s}:0")
            SWEEP_OV+=("regression_task.optimizer.lr=1e-3 regression_task.lr_scheduler.type=CosineAnnealingWarmupRestarts regression_task.lr_scheduler.first_cycle_steps=300 regression_task.lr_scheduler.cycle_mult=1.0 regression_task.lr_scheduler.warmup_steps=50 regression_task.lr_scheduler.max_lr=1e-3 regression_task.lr_scheduler.min_lr=1e-5 wandb.tags=[v25b,M_cls,restarts300,lr1e-3,split${s},init0]")
            JOBS+=("M_zs_s${s}:0")
            SWEEP_OV+=("multitask.dist=point model.attention_mask.enabled=false model.graph_regularization.graph_reg_lambda=1e-2 regression_task.optimizer.lr=1e-3 wandb.tags=[v25b,M_zs,stack,mse,graphkl1e-2,lr1e-3,split${s},init0]")
          done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_morph_v24
          EXTRA_OV="trainer.max_epochs=${MORPH_EPOCHS:-500} data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true wandb.project=torchcell_019_morph_v25" ;;
  # v25c (2026-10-11, same project): the loss axis beyond MSE. Morphology trains the
  # 19-quantile pinball; the global heads of the other campaigns trained MSE or CRPS, and
  # the metric is Pearson per feature, so the Gaussian CRPS and the pearson_mse objective
  # (torchcell/losses/distributional.py) run on M_cls and M_zs at lr 1e-3, split seeds 0
  # to 3: 16 one-card tasks.
  #   sbatch -J 019-morph2w3-loss -p cabbi --gres=gpu:1 --cpus-per-task=5 --mem=60g \
  #       --time=0-16:00:00 --array=0-15%8 --export=ALL,W5_STAGE=morph_v25c,... igb_expr_wave5.slurm
  morph_v25c) JOBS=(); SWEEP_OV=()
          for s in 0 1 2 3; do for arm in M_cls M_zs; do for dist in crps pearson_mse; do
            JOBS+=("${arm}_s${s}:0")
            SWEEP_OV+=("multitask.dist=${dist} regression_task.optimizer.lr=1e-3 wandb.tags=[v25c,${arm},${dist},lr1e-3,split${s},init0]")
          done; done; done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_morph_v24
          EXTRA_OV="trainer.max_epochs=${MORPH_EPOCHS:-500} data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true wandb.project=torchcell_019_morph_v25" ;;
  # v26 (2026-10-12, same project, torchcell/losses/latent_align.py): shaping the latent
  # space by the labels. The cosine similarity of two strains' latent vectors is pulled
  # toward the Pearson correlation of their 116-feature morphology vectors, weighted
  # lambda beside the pinball loss. The marginal solution (every strain the same vector)
  # pays this term from the first step, which the data loss never did (0.195 to 0.187
  # across an escape). M_cls shapes the perturbed CLS at lambda 0.01, 0.1 and 1; M_zs
  # shapes [CLS || z_S] at 0.1 and 1; one M_cls cell at lambda 1 without batch centering.
  # lr 3e-4 (the only lr that escaped in v25), split seeds 0 to 3: 24 one-card tasks.
  # Read with morph_v25_readout.py (window epochs 400 to 499; tags align<lambda>, src<source>).
  #   sbatch -J 019-morph2w5-align -p cabbi --gres=gpu:1 --cpus-per-task=5 --mem=60g \
  #       --time=0-08:00:00 --array=0-23%8 --export=ALL,W5_STAGE=morph_v26,... igb_expr_wave5.slurm
  morph_v26) JOBS=(); SWEEP_OV=()
          LA="multitask.latent_align.enabled=true multitask.latent_align.head=global regression_task.optimizer.lr=3e-4"
          for s in 0 1 2 3; do
            for lam in 0.1 1; do
              JOBS+=("M_cls_s${s}:0")
              SWEEP_OV+=("$LA multitask.latent_align.lambda=${lam} multitask.latent_align.source=cls wandb.tags=[v26,M_cls,align${lam},srccls,lr3e-4,split${s},init0]")
              JOBS+=("M_zs_s${s}:0")
              SWEEP_OV+=("$LA multitask.latent_align.lambda=${lam} multitask.latent_align.source=cls_zs wandb.tags=[v26,M_zs,align${lam},srccls_zs,lr3e-4,split${s},init0]")
            done
            JOBS+=("M_cls_s${s}:0")
            SWEEP_OV+=("$LA multitask.latent_align.lambda=0.01 multitask.latent_align.source=cls wandb.tags=[v26,M_cls,align0.01,srccls,lr3e-4,split${s},init0]")
            JOBS+=("M_cls_s${s}:0")
            SWEEP_OV+=("$LA multitask.latent_align.lambda=1 multitask.latent_align.source=cls multitask.latent_align.center=false wandb.tags=[v26,M_cls,align1,srccls,nocenter,lr3e-4,split${s},init0]")
          done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_morph_v24
          EXTRA_OV="trainer.max_epochs=${MORPH_EPOCHS:-500} data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true wandb.project=torchcell_019_morph_v25" ;;
  # v26b (2026-10-12 evening, same project; read of 113 v25 cells in morph_v25_readout.json):
  # the plateau breaks under a gene-token readout with MSE (M_zs + MSE 0.147, 4 of 4 escaped
  # by epoch 200), under the graph KL on M_cls (4 of 4), and under a LOWER lr on M_cls
  # (lr 1e-4: 0.141, 3 of 4, still rising at epoch 499); a higher lr never breaks it (0 of
  # 23 cells). This stage combines and replicates: M_zs + MSE init seeds 1 and 2 (the
  # headline's replicates); M_xattn + MSE; M_zs + MSE with the alignment term at 0.1;
  # M_zs + MSE and M_cls + MSE at lr 1e-4, and M_cls pinball at lr 1e-4, the three lr 1e-4
  # cells at 1,000 epochs (MORPH_EPOCHS_LR1E4, window 900 to 999 beside 400 to 499).
  # Split seeds 0 to 3: 28 one-card tasks, the 500-epoch cells first.
  #   sbatch -J 019-morph2w6-combine -p cabbi --gres=gpu:1 --cpus-per-task=5 --mem=60g \
  #       --time=0-14:00:00 --array=0-27%8 --export=ALL,W5_STAGE=morph_v26b,... igb_expr_wave5.slurm
  morph_v26b) JOBS=(); SWEEP_OV=()
          E1="trainer.max_epochs=${MORPH_EPOCHS_LR1E4:-1000}"
          for s in 0 1 2 3; do for init in 1 2; do
            JOBS+=("M_zs_s${s}:${init}")
            SWEEP_OV+=("multitask.dist=point regression_task.optimizer.lr=3e-4 wandb.tags=[v26b,M_zs,mse,lr3e-4,split${s},init${init}]")
          done; done
          for s in 0 1 2 3; do
            JOBS+=("M_xattn_s${s}:0")
            SWEEP_OV+=("multitask.dist=point regression_task.optimizer.lr=3e-4 wandb.tags=[v26b,M_xattn,mse,lr3e-4,split${s},init0]")
            JOBS+=("M_zs_s${s}:0")
            SWEEP_OV+=("multitask.dist=point regression_task.optimizer.lr=3e-4 multitask.latent_align.enabled=true multitask.latent_align.head=global multitask.latent_align.lambda=0.1 multitask.latent_align.source=cls_zs wandb.tags=[v26b,M_zs,mse,align0.1,srccls_zs,lr3e-4,split${s},init0]")
          done
          for s in 0 1 2 3; do
            JOBS+=("M_zs_s${s}:0")
            SWEEP_OV+=("$E1 multitask.dist=point regression_task.optimizer.lr=1e-4 wandb.tags=[v26b,M_zs,mse,lr1e-4,split${s},init0]")
            JOBS+=("M_cls_s${s}:0")
            SWEEP_OV+=("$E1 multitask.dist=point regression_task.optimizer.lr=1e-4 wandb.tags=[v26b,M_cls,mse,lr1e-4,split${s},init0]")
            JOBS+=("M_cls_s${s}:0")
            SWEEP_OV+=("$E1 regression_task.optimizer.lr=1e-4 wandb.tags=[v26b,M_cls,lr1e-4,e1000,split${s},init0]")
          done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_morph_v24
          EXTRA_OV="trainer.max_epochs=${MORPH_EPOCHS:-500} data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true wandb.project=torchcell_019_morph_v25" ;;
  # prot_v26 (2026-10-12, project torchcell_019_prot_v26): the same term on the knockout
  # proteome. The S_prot arm (v21 small trunk: ProtT5, width 90, six layers, per-gene
  # head) on the FULL fig3_proteome store (Messner 2023, every proteome genotype, not the
  # both-label store of v21) at batch 128, the masked path off (mask_schedule null, so the
  # alignment runs on the plain step), 600 epochs; the registered window is epochs 500 to
  # 599, declared here before any cell is read. The latent is z_S, the pool over the
  # strain's perturbed gene tokens (no perturbed CLS on this trunk), the label vector the
  # measured proteins. Four reference tasks first (one per split, so each writes its
  # split cache alone under %4), then lambda 0.1 and 1: 12 one-card tasks.
  #   sbatch -J 019-prot2w1-align -p cabbi --gres=gpu:1 --cpus-per-task=5 --mem=60g \
  #       --time=0-20:00:00 --array=0-11%4 --export=ALL,W5_STAGE=prot_v26,... igb_expr_wave5.slurm
  prot_v26) JOBS=(); SWEEP_OV=()
          PS="multitask.mask_schedule=null cell_dataset.query_file=fig3_proteome.cql cell_dataset.dataset_tag=fig3_proteome cell_dataset.require_modalities=[protein_abundance] regression_task.optimizer.lr=3e-4"
          for s in 0 1 2 3; do
            JOBS+=("S_prot_s${s}:0")
            SWEEP_OV+=("$PS wandb.tags=[v26,S_prot,ref,lr3e-4,split${s},init0]")
          done
          for lam in 0.1 1; do for s in 0 1 2 3; do
            JOBS+=("S_prot_s${s}:0")
            SWEEP_OV+=("$PS multitask.latent_align.enabled=true multitask.latent_align.lambda=${lam} multitask.latent_align.source=zs multitask.latent_align.head=per_gene wandb.tags=[v26,S_prot,align${lam},srczs,lr3e-4,split${s},init0]")
          done; done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_expr_v21_small
          EXTRA_OV="trainer.max_epochs=${PROT_EPOCHS:-600} data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true wandb.project=torchcell_019_prot_v26" ;;
  scale_n) JOBS=(); SWEEP_OV=()
          for wl in 45:6e-4 90:3e-4 180:1.5e-4 360:7.5e-5; do for s in 0 1 2; do
            JOBS+=("S_mask_s${s}:0")
            SWEEP_OV+=("model.hidden_channels=${wl%%:*} regression_task.optimizer.lr=${wl##*:} wandb.tags=[scaling,N_w${wl%%:*},cabbi,split${s}]")
          done; done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_expr_v21_small
          EXTRA_OV="trainer.max_epochs=1200 data_module.batch_size=128 +data_module.materialize=true data_module.persistent_workers=true wandb.project=torchcell_019_scaling" ;;
  conditioned) JOBS=()
          for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
            JOBS+=("C_expr_s${s}:0" "C_exprperm_s${s}:0" "C_prot_s${s}:0" "C_protperm_s${s}:0")
          done
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=4; STAGE_CONFIG=cgt_expr_v20_conditioned
          EXTRA_OV="" ;;
  # Canary for the conditioned round: the four arms of partition 0 for two real epochs, which
  # takes the conditioning step through train, validation, the eval-mode train pass, the
  # checkpoint and the held-out test pass. fast_dev_run cannot do this: it writes no
  # checkpoint, so `run_test` fails on ckpt_path="best" after a fit that passed (2423179).
  canary_conditioned) JOBS=("C_expr_s0:0" "C_exprperm_s0:0" "C_prot_s0:0" "C_protperm_s0:0")
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=4; STAGE_CONFIG=cgt_expr_v20_conditioned
          EXTRA_OV="trainer.max_epochs=2 trainer.train_eval_every=1 wandb.project=torchcell_019_prot_v20_canary" ;;
  # ------------------------------------------------------- PERTURBATION-LOCALITY ROUND (2026.09.18)
  # The v13 reference against the hop-0 self-indicator and 2-hop graph propagation of the
  # deletion, split seeds 0 to 3, init seeds 0 to 2, 1,200 epochs (conf/cgt_expr_v17_locality.yaml).
  # 36 runs as twelve tasks of three: the three arms of one (split, seed) share a card, so
  # every contrast is paired within card, split and seed. Tasks are split-major, so
  # `--array=0-2` is split 0 and any subset of tasks can be sent to whichever partition
  # has cards (the gpu A40s as v13 ends; cabbi as v14 ends).
  #
  #   H=$(git rev-parse HEAD); D=$(git diff HEAD | sha256sum | cut -d" " -f1)
  #   sbatch -p gpu --gres=gpu:1 --cpus-per-task=12 --mem=120g --time=4-00:00:00 \
  #       --array=0-11 --export=ALL,W5_STAGE=locality,TORCHCELL_SOURCE_GIT_HASH=$H,\
  #   TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  locality) JOBS=(
            L_ref_s0:0 L_self_s0:0 L_prop2_s0:0
            L_ref_s0:1 L_self_s0:1 L_prop2_s0:1
            L_ref_s0:2 L_self_s0:2 L_prop2_s0:2
            L_ref_s1:0 L_self_s1:0 L_prop2_s1:0
            L_ref_s1:1 L_self_s1:1 L_prop2_s1:1
            L_ref_s1:2 L_self_s1:2 L_prop2_s1:2
            L_ref_s2:0 L_self_s2:0 L_prop2_s2:0
            L_ref_s2:1 L_self_s2:1 L_prop2_s2:1
            L_ref_s2:2 L_self_s2:2 L_prop2_s2:2
            L_ref_s3:0 L_self_s3:0 L_prop2_s3:0
            L_ref_s3:1 L_self_s3:1 L_prop2_s3:1
            L_ref_s3:2 L_self_s3:2 L_prop2_s3:2
          )
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=3; STAGE_CONFIG=cgt_expr_v17_locality
          EXTRA_OV="" ;;
  # ------------------------------------------------------ HYGIENE AND READOUT ROUND (2026.09.20)
  # The v13 reference against the masked objective switched off and against a per-gene
  # affine row over the strain context, split seeds 0 to 3, init seeds 0 to 2, 1,200
  # epochs (conf/cgt_expr_v18_hygiene.yaml). 36 runs as twelve tasks of three, the three
  # arms of one (split, seed) per card. Tasks are split-major, so any subset can go to
  # whichever partition has cards.
  #
  #   H=$(git rev-parse HEAD); D=$(git diff HEAD | sha256sum | cut -d" " -f1)
  #   sbatch -J 019-v18w1-hygiene -p cabbi --gres=gpu:1 --cpus-per-task=12 --mem=120g \
  #       --time=2-00:00:00 --array=0-11 --export=ALL,W5_STAGE=hygiene,\
  #   TORCHCELL_SOURCE_GIT_HASH=$H,TORCHCELL_SOURCE_DIFF_SHA256=$D \
  #       experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
  hygiene) JOBS=(
            Y_ref_s0:0 Y_k0_s0:0 Y_ctx_s0:0
            Y_ref_s0:1 Y_k0_s0:1 Y_ctx_s0:1
            Y_ref_s0:2 Y_k0_s0:2 Y_ctx_s0:2
            Y_ref_s1:0 Y_k0_s1:0 Y_ctx_s1:0
            Y_ref_s1:1 Y_k0_s1:1 Y_ctx_s1:1
            Y_ref_s1:2 Y_k0_s1:2 Y_ctx_s1:2
            Y_ref_s2:0 Y_k0_s2:0 Y_ctx_s2:0
            Y_ref_s2:1 Y_k0_s2:1 Y_ctx_s2:1
            Y_ref_s2:2 Y_k0_s2:2 Y_ctx_s2:2
            Y_ref_s3:0 Y_k0_s3:0 Y_ctx_s3:0
            Y_ref_s3:1 Y_k0_s3:1 Y_ctx_s3:1
            Y_ref_s3:2 Y_k0_s3:2 Y_ctx_s3:2
          )
          JOBS_EXPLICIT=1
          RUNS_PER_GPU=3; STAGE_CONFIG=cgt_expr_v18_hygiene
          EXTRA_OV="" ;;
  # CANARY -- one fast_dev_run to prove container + conda + PYTHONPATH + LMDB path before a
  # wave is committed to the queue. Exits in minutes.
  canary) ARMS=(X_mix); SEEDS=(0)
          RUNS_PER_GPU=1
          EXTRA_OV="trainer.fast_dev_run=true" ;;
  # Canary for the launch round specifically: proves the three NEW dist heads construct and
  # step, which the X_mix canary cannot, since a head that fails to build fails at model
  # construction and would otherwise surface only after five days of queue.
  canary_launch)
          ARMS=(Q_crps Q_laplace Q_point); SEEDS=(0)
          RUNS_PER_GPU=1; STAGE_CONFIG=cgt_expr_v9_mask
          EXTRA_OV="trainer.fast_dev_run=true" ;;
  *) echo "unknown W5_STAGE '$STAGE' (expected canary|canary_launch|canary_mech|canary_pearson|canary_listmle|canary_pergene_rep|h2|probe|long|wave6|launch|pearson|pearson_b64|listmle|listmle_b64|resume|resume_many|mech|pergene_rep|head|split|proteome|wd|joint|locality|hygiene|joint_clean)" >&2; exit 1 ;;
esac

# SEED-MAJOR, so each task's co-resident runs are DIFFERENT arms rather than three copies of
# one. The metric is unaffected either way (the budget is a fixed epoch count, so a slower
# neighbour costs wall time, not score), but arm-blocked packing aligns "what you share a GPU
# with" perfectly with the arm axis, and there is no reason to leave that correlation in.
#
# resume_many is the one stage that does NOT build its run list from an arm x seed product:
# its runs, their order and their checkpoints all come from the map file, and reordering them
# here would break the JOBS[i] <-> RESUME_CKPTS[i] pairing that assigns each run its own
# checkpoint.
if [[ "$STAGE" == "resume_many" ]]; then
  JOBS=("${RESUME_JOBS[@]}")
elif [[ "${JOBS_EXPLICIT:-0}" == "1" ]]; then
  # The stage wrote its own run list above (split round); nothing to build.
  :
else
  JOBS=()
  for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
      JOBS+=("$arm:$seed")
    done
  done
fi

TASK="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"
START=$((TASK * RUNS_PER_GPU))
if [[ "$START" -ge "${#JOBS[@]}" ]]; then
  echo "task $TASK starts past the end of stage '$STAGE' (${#JOBS[@]} runs)" >&2
  exit 1
fi

# Reserve 2 CPUs for overhead, then split the rest across co-resident runs. Giving each the
# full allocation oversubscribes the cores and makes the interference non-uniform across tasks.
NUM_WORKERS=$(((SLURM_CPUS_PER_TASK - 2) / RUNS_PER_GPU))
[[ "$NUM_WORKERS" -lt 1 ]] && NUM_WORKERS=1

echo "=========================================================="
echo "host          : $(hostname)"
echo "partition     : ${SLURM_JOB_PARTITION:-unknown}"
echo "job           : ${SLURM_ARRAY_JOB_ID:-none} task=${TASK}"
echo "stage         : $STAGE  (runs ${START}..$((START + RUNS_PER_GPU - 1)) of ${#JOBS[@]})"
echo "project root  : $PROJECT_ROOT"
echo "gpus          : ${CUDA_VISIBLE_DEVICES:-unset}  ($RUNS_PER_GPU runs co-resident)"
echo "num_workers   : $NUM_WORKERS  (of $SLURM_CPUS_PER_TASK cpus)"
echo "=========================================================="

# `set -u` MUST be relaxed around the module/bashrc bootstrap. IGB's /etc/bashrc line 12
# references BASHRCSOURCED before defining it, so under `set -u` sourcing it aborts the job
# with "BASHRCSOURCED: unbound variable" -- which is exactly how the first canary (2324269)
# died, before reaching python at all. Cluster init scripts are not ours to make -u clean, so
# the strictness is dropped for the bootstrap and restored immediately after.
set +u
module purge
source ~/.bashrc
module load singularity
set -u

# ONE `singularity exec` PER RUN, not one container wrapping the whole loop. The loop lives in
# the OUTER shell so every variable (arm, seed, paths, worker count) is expanded here and the
# container receives a fully-formed command -- a loop nested inside `bash -c "..."` has to
# escape a bash array through several quoting layers, which is how a launcher silently runs the
# wrong arm. Concurrent singularity execs on one node are just processes and share the GPU
# exactly as concurrent python processes would.
#
# WANDB IS OFFLINE: compute nodes have no internet. Runs land under
# $DATA_ROOT/wandb-experiments/<group>/wandb/offline-run-* and are synced from the LOGIN node
# afterwards with scripts/wandb_sync_offline.sh (memory: cluster-rules-igb-delta).
CONTAINER_ENV="source \$HOME/miniconda3/bin/activate && conda activate torchcell"
CONTAINER_ENV="$CONTAINER_ENV && export PYTHONPATH='$PROJECT_ROOT'"
CONTAINER_ENV="$CONTAINER_ENV && export WANDB_MODE=offline WANDB__SERVICE_WAIT=600"
CONTAINER_ENV="$CONTAINER_ENV && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CONTAINER_ENV="$CONTAINER_ENV && export TORCHCELL_CONFIG=${STAGE_CONFIG:-cgt_expr_011}"
# SOURCE PROVENANCE MUST BE RESOLVED AT SUBMIT TIME, ON THE LOGIN NODE.
#
# Two canaries died establishing where git actually lives here. 2324270: the train script
# shelled out to `git rev-parse` and rockylinux_9.sif has no git -> FileNotFoundError('git').
# 2324272: moving that call into this launcher failed too, because **IGB COMPUTE NODES HAVE NO
# GIT EITHER** -- verified directly, `which git` on compute-3-3 returns nothing while
# /usr/bin/sha256sum is present. Only the login node has it.
#
# So the submitting shell exports the values and they ride in through --export:
#
#   H=$(git -C "$WT" rev-parse HEAD)
#   D=$(git -C "$WT" diff HEAD | sha256sum | cut -d" " -f1)
#   sbatch ... --export=ALL,W5_STAGE=h2,TORCHCELL_SOURCE_GIT_HASH=$H,TORCHCELL_SOURCE_DIFF_SHA256=$D ...
#
# The `command -v git` branch below is kept so the same script still self-resolves on a host
# that DOES have git (GilaHyper). If neither path yields a hash the run records "unavailable"
# and says so loudly -- an unattributable run must not be pooled with attributable ones.
SRC_HASH="${TORCHCELL_SOURCE_GIT_HASH:-}"
SRC_DIFF="${TORCHCELL_SOURCE_DIFF_SHA256:-}"
if [[ -z "$SRC_HASH" ]] && command -v git >/dev/null 2>&1; then
  SRC_HASH="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
  SRC_DIFF="$(git -C "$PROJECT_ROOT" diff HEAD | sha256sum | cut -d" " -f1)"
fi
if [[ -n "$SRC_HASH" && -n "$SRC_DIFF" ]]; then
  echo "source        : $SRC_HASH (diff ${SRC_DIFF:0:12})"
  CONTAINER_ENV="$CONTAINER_ENV && export TORCHCELL_SOURCE_GIT_HASH='$SRC_HASH'"
  CONTAINER_ENV="$CONTAINER_ENV && export TORCHCELL_SOURCE_DIFF_SHA256='$SRC_DIFF'"
else
  echo "source        : UNAVAILABLE -- no git here and no hash exported at submit time"
fi

PIDS=()
LABELS=()
for k in $(seq 0 $((RUNS_PER_GPU - 1))); do
  IDX=$((START + k))
  [[ "$IDX" -ge "${#JOBS[@]}" ]] && continue
  JOB="${JOBS[$IDX]}"
  ARM="${JOB%%:*}"
  SEED="${JOB##*:}"
  LOG="$OUT_DIR/019-wave5-igb_${SLURM_ARRAY_JOB_ID}_${TASK}_${ARM}_seed${SEED}.out"
  # Per-RUN overrides. Only resume_many has any: its checkpoint differs between the two runs
  # sharing this GPU, so it cannot travel in the stage-wide EXTRA_OV.
  RUN_OV=""
  # The v22 readout keys runs by the tags [v22,<arm>,split<N>]; the arm script's own tag
  # list comes earlier on the command line, so this one wins.
  if [[ "$STAGE" == "v22_b128_seeds" ]]; then
    RUN_OV=" wandb.tags=[v22,F_b128lr1,igb,split${ARM##*_s}]"
  fi
  # The scaling sweeps carry one override set per run (fraction, or width and rate),
  # indexed like JOBS.
  if [[ -n "${SWEEP_OV[$IDX]:-}" ]]; then
    RUN_OV="$RUN_OV ${SWEEP_OV[$IDX]}"
  fi
  if [[ "$STAGE" == "resume_many" ]]; then
    RUN_OV=" trainer.resume_ckpt_path=${RESUME_CKPTS[$IDX]}"
    [[ -n "${RESUME_FROM[$IDX]}" ]] && RUN_OV="$RUN_OV +wandb.resumed_from=${RESUME_FROM[$IDX]}"
    echo "  resuming $ARM seed=$SEED from ${RESUME_CKPTS[$IDX]} (W&B run ${RESUME_FROM[$IDX]:-none})"
  fi
  echo "  launching $ARM seed=$SEED -> $LOG"
  # STAGGERED START. Four runs constructing the dataset in the same second raced on the
  # gene_set.json write (job 2397304: three of 24 runs died at startup on the temp-file
  # rename). The writer now uses per-process temp names; the stagger keeps the LMDB open,
  # the split cache read and the W&B init from piling up as well. 30 s per run is nothing
  # against a multi-day budget.
  [[ "$k" -gt 0 ]] && sleep $((k * 30))
  singularity exec --nv "$SIF" bash -c "
    # -u omitted on purpose: conda's activate scripts reference unset variables, the same
    # class of failure as /etc/bashrc above. -e and pipefail still catch real errors.
    set -eo pipefail
    $CONTAINER_ENV
    export TORCHCELL_EXTRA_OVERRIDES='$EXTRA_OV$RUN_OV trainer.early_stopping.enabled=false data_module.num_workers=$NUM_WORKERS'
    echo \"python: \$(which python)\"
    echo \"overrides: \$TORCHCELL_EXTRA_OVERRIDES\"
    bash '$ARM_SH' '$ARM' '$SEED'
  " > "$LOG" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$ARM:$SEED")
done

# Fail the task if ANY co-resident run fails: a half-completed task leaves an arm missing a
# replicate, which breaks the pairing for that replicate.
STATUS=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  OK     ${LABELS[$i]}"
  else
    echo "  FAILED ${LABELS[$i]}" >&2
    STATUS=1
  fi
done
exit "$STATUS"
