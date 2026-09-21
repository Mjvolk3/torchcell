---
id: ina71ass9fe07p9xo8zprn3
title: Campaign_status
desc: ''
updated: 1789954695906
created: 1789954695906
---

## 2026.09.20 - One table of every 019 job across every cluster

The campaign runs on three machines and five partitions at once, and `squeue` names the launcher rather than the round. On this date five concurrent arrays all read `019-wave5-igb`, so the only thing distinguishing the joint round from the locality round on the same partition was the job id. Two changes fix that.

### The naming convention

Every submission is named `019-<round>w<N>-<stage>`: the config version, the submission index of that round on that cluster, and the stage the launcher was given. Pass it with `-J` at submit time, since the `#SBATCH --job-name` line in `igb_expr_wave5.slurm` is a static placeholder.

```bash
sbatch -J 019-v17w2-locality -p gpu --gres=gpu:1 --cpus-per-task=12 --mem=120g \
    --time=4-00:00:00 --array=6-11 \
    --export=ALL,W5_STAGE=locality,TORCHCELL_SOURCE_GIT_HASH=$H,TORCHCELL_SOURCE_DIFF_SHA256=$D \
    experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
```

The name also lands in the log file names, since the output pattern is `%x_%A_%a.out`. A job that predates the convention is renamed in place with `scontrol update JobId=<id> JobName=019-v16w1-joint`; a single array task takes `JobId=<id>_[6-11]`, which is how v17's held second wave became `w2` while its running first wave stayed `w1`.

### The status view

`campaign_status.py` reads every cluster in one ssh call each and prints round, wave, state, current epoch against the round's configured budget, and the arms each card is holding. Round and wave come from the job name, the arms from the per-run log file names, and the epoch from a `tail -c` of the newest log, never from reading a whole file, since these reach tens of megabytes and the read happens on a login node.

```
CLUSTER    JOB             PART    ROUND               STATE           EPOCH     ELAPSED  ARMS / REASON
IGB        2401479_0       cabbi   v15 weight decay w1 RUNNING  3224 of 6000  3-13:07:03  W_ref_s1_seed0, W_ref_s1_seed1, W_wd1e1_s1_seed0, W_wd1e2_s1_seed0
IGB        2409262_0       gpu     v17 locality w1     RUNNING  1176 of 1200  1-05:15:21  L_prop2_s0_seed0, L_ref_s0_seed0, L_self_s0_seed0
IGB        2409562_[0-5]   gpu     v16 joint w1        PENDING                      0:00  Resources
IGB        2409262_[6-11]  gpu     v17 locality w2     PENDING                      0:00  JobHeldUser
```

`--no-epochs` skips the log reads when only the queue matters. `--all` keeps jobs whose names do not mention 019. Adding a round means one line in `STAGES` mapping its stage to a label and its configured epoch budget; adding a cluster means one entry in `CLUSTERS` with an ssh target and a log directory.

Related: [[experiments.019-simb-multimodal.scripts.igb_expr_wave5]], [[experiments.019-simb-multimodal.conf.cgt_expr_v16_joint]], [[experiments.019-simb-multimodal.conf.cgt_expr_v17_locality]].
