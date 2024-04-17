---
id: t99fo336hplyhhrlkebk6w4
title: '16'
desc: ''
updated: 1713325608116
created: 1713323777077
---

## Running the Sweep

We can't even get to running the script on Delta during interactive session, which has become prohibitively expensive.

```bash
 *  Executing task in folder torchcell: srun --account=bbub-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=1 --cpus-per-task=1 --mem=62g --pty bash 

srun: job 3416475 queued and waiting for resources
source activate
srun: job 3416475 has been allocated resources
conda activate 
source activate
conda activate 
Tue Apr 16 21:44:22 CDT 2024 - Starting to source .bashrc
Tue Apr 16 21:44:22 CDT 2024 - Sourcing global definitions...
Tue Apr 16 21:44:22 CDT 2024 - Global definitions sourced.
Tue Apr 16 21:44:22 CDT 2024 - Setting up user-specific environment...
Tue Apr 16 21:44:22 CDT 2024 - User-specific environment set.
Tue Apr 16 21:44:22 CDT 2024 - Initializing Conda...
^C[mjvolk3@gpub001 torchcell]$ source ~/.bashrc
Tue Apr 16 21:45:04 CDT 2024 - Starting to source .bashrc
Tue Apr 16 21:45:04 CDT 2024 - Sourcing global definitions...
Tue Apr 16 21:45:04 CDT 2024 - Global definitions sourced.
Tue Apr 16 21:45:04 CDT 2024 - Setting up user-specific environment...
Tue Apr 16 21:45:04 CDT 2024 - User-specific environment set.
Tue Apr 16 21:45:04 CDT 2024 - Initializing Conda...
Tue Apr 16 21:47:18 CDT 2024 - Conda initialized.
Tue Apr 16 21:47:18 CDT 2024 - .bashrc sourced successfully.
(base) [mjvolk3@gpub001 torchcell]$ source ~/.bashrc
Tue Apr 16 21:48:24 CDT 2024 - Starting to source .bashrc
Tue Apr 16 21:48:24 CDT 2024 - Sourcing global definitions...
^C
(base) [mjvolk3@gpub001 torchcell]$ ^C
(base) [mjvolk3@gpub001 torchcell]$ conda activate /projects/bbub/miniconda3/envs/torchcell

(torchcell) [mjvolk3@gpub001 torchcell]$ 
(torchcell) [mjvolk3@gpub001 torchcell]$ wandb sweep experiments/smf-dmf-tmf-001/conf/deep_set-sweep_12.yaml
^CTraceback (most recent call last):
  File "/projects/bbub/miniconda3/envs/torchcell/bin/wandb", line 5, in <module>
    from wandb.cli.cli import cli
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/__init__.py", line 27, in <module>
    from wandb import sdk as wandb_sdk
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/__init__.py", line 25, in <module>
    from .artifacts.artifact import Artifact
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/artifacts/artifact.py", line 46, in <module>
    from wandb.apis.normalize import normalize_exceptions
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/apis/__init__.py", line 43, in <module>
    from .internal import Api as InternalApi  # noqa
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/apis/internal.py", line 3, in <module>
    from wandb.sdk.internal.internal_api import Api as InternalApi
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/internal/internal_api.py", line 35, in <module>
    from wandb_gql import Client, gql
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/vendor/gql-0.2.0/wandb_gql/__init__.py", line 1, in <module>
^Cobject address  : 0x7f6d9a911cc0
object refcount : 2
object type     : 0x878c80
object type name: KeyboardInterrupt
object repr     : KeyboardInterrupt()
lost sys.stderr
^C
(torchcell) [mjvolk3@gpub001 torchcell]$ ^C
(torchcell) [mjvolk3@gpub001 torchcell]$ python experiments/smf-dmf-tmf-001/deep_set.py
slurmstepd: error: *** STEP 3416475.0 ON gpub001 CANCELLED AT 2024-04-16T22:14:41 DUE TO TIME LIMIT ***
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
srun: error: Timed out waiting for job step to complete
 *  Terminal will be reused by tasks, press any key to close it. 
 ```

## Running Only Deep Set

```bash
Running "module reset". Resetting modules to system default. The following $MODULEPATH directories have been removed: None
Tue Apr 16 22:01:52 CDT 2024 - Starting to source .bashrc
Tue Apr 16 22:01:52 CDT 2024 - Sourcing global definitions...
Tue Apr 16 22:01:53 CDT 2024 - Global definitions sourced.
Tue Apr 16 22:01:53 CDT 2024 - Setting up user-specific environment...
Tue Apr 16 22:01:53 CDT 2024 - User-specific environment set.
Tue Apr 16 22:01:53 CDT 2024 - Initializing Conda...
Tue Apr 16 22:04:19 CDT 2024 - Conda initialized.
Tue Apr 16 22:04:19 CDT 2024 - .bashrc sourced successfully.
/projects/bbub/mjvolk3/torchcell
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              64
On-line CPU(s) list: 0-63
Thread(s) per core:  1
Core(s) per socket:  64
Socket(s):           1
NUMA node(s):        4
Vendor ID:           AuthenticAMD
CPU family:          25
Model:               1
Model name:          AMD EPYC 7763 64-Core Processor
Stepping:            1
CPU MHz:             2445.384
BogoMIPS:            4890.76
Virtualization:      AMD-V
L1d cache:           32K
L1i cache:           32K
L2 cache:            512K
L3 cache:            32768K
NUMA node0 CPU(s):   0-15
NUMA node1 CPU(s):   16-31
NUMA node2 CPU(s):   32-47
NUMA node3 CPU(s):   48-63
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr wbnoinvd amd_ppin brs arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload vgif v_spec_ctrl umip pku ospke vaes vpclmulqdq rdpid overflow_recov succor smca fsrm
Tue Apr 16 22:04:20 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A40                     On  | 00000000:C7:00.0 Off |                    0 |
|  0%   28C    P8              21W / 300W |      0MiB / 46068MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
MemTotal:       263842004 kB
MemFree:        180708136 kB
MemAvailable:   213966264 kB
Buffers:            2116 kB
Cached:         41367216 kB
SwapCached:            0 kB
Active:         25525848 kB
Inactive:       38380116 kB
Active(anon):    4720232 kB
Inactive(anon): 31360000 kB
Active(file):   20805616 kB
Inactive(file):  7020116 kB
Unevictable:       92968 kB
Mlocked:           89896 kB
SwapTotal:             0 kB
SwapFree:              0 kB
Dirty:                48 kB
Writeback:             0 kB
AnonPages:      22432696 kB
Mapped:           748364 kB
Shmem:          13543600 kB
KReclaimable:    7211264 kB
Slab:           17565748 kB
SReclaimable:    7211264 kB
SUnreclaim:     10354484 kB
KernelStack:       19040 kB
PageTables:       146200 kB
NFS_Unstable:          0 kB
Bounce:                0 kB
WritebackTmp:          0 kB
CommitLimit:    131921000 kB
Committed_AS:   82482264 kB
VmallocTotal:   34359738367 kB
VmallocUsed:      451060 kB
VmallocChunk:          0 kB
Percpu:            68864 kB
HardwareCorrupted:     0 kB
AnonHugePages:  11433984 kB
ShmemHugePages:        0 kB
ShmemPmdMapped:        0 kB
FileHugePages:         0 kB
FilePmdMapped:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB
DirectMap4k:    29523892 kB
DirectMap2M:    238641152 kB
DirectMap1G:           0 kB

Currently Loaded Modules:
  1) gcc/11.4.0      3) cuda/11.8.0         5) slurm-env/0.1
  2) openmpi/4.1.6   4) cue-login-env/1.0   6) default-s11

 

wandb: Using wandb-core version 0.17.0b10 as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: Currently logged in as: mjvolk3 (zhao-group). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.16.6
wandb: Run data is saved locally in /projects/bbub/mjvolk3/torchcell/wandb/run-20240416_224146-g3nok1rk
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run volcanic-elevator-201
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_test
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_test/runs/g3nok1rk
/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer. For large datasets, this may lead to large memory footprint.
  warnings.warn(*args, **kwargs)  # noqa: B028
data/go/go.obo: fmt(1.2) rel(2024-01-17) 45,869 Terms
[2024-04-16 22:43:26,402][__main__][INFO] - cuda
GPU available: True (cuda), used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/setup.py:187: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loggers/wandb.py:389: There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`.
[2024-04-16 22:43:59,319][torchcell.datamodules.cell][INFO] - Loading cached indices from /scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/smf-dmf-tmf_1e02/data_module_cache/cached_indices.json

  | Name          | Type             | Params
---------------------------------------------------
0 | model         | ModuleDict       | 72.0 K
1 | loss          | MSEListMLELoss   | 0     
2 | loss_node     | MSELoss          | 0     
3 | train_metrics | MetricCollection | 0     
4 | val_metrics   | MetricCollection | 0     
5 | test_metrics  | MetricCollection | 0     
6 | pearson_corr  | PearsonCorrCoef  | 0     
7 | spearman_corr | SpearmanCorrCoef | 0     
---------------------------------------------------
72.0 K    Trainable params
0         Non-trainable params
72.0 K    Total params
0.288     Total estimated model params size (MB)
SLURM auto-requeueing enabled. Setting signal handlers.

Sanity Checking: |          | 0/? [00:00<?, ?it/s]
Sanity Checking:   0%|          | 0/2 [00:00<?, ?it/s]
Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028


Sanity Checking DataLoader 0:  50%|█████     | 1/2 [00:17<00:17,  0.06it/s]

Sanity Checking DataLoader 0: 100%|██████████| 2/2 [00:17<00:00,  0.11it/s]
                                                                           
/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loops/fit_loop.py:293: The number of training batches (39) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.

Training: |          | 0/? [00:00<?, ?it/s]
Training:   0%|          | 0/39 [00:00<?, ?it/s]
Epoch 0:   0%|          | 0/39 [00:00<?, ?it/s] 

Epoch 0:   3%|▎         | 1/39 [00:07<04:53,  0.13it/s]
Epoch 0:   3%|▎         | 1/39 [00:07<04:53,  0.13it/s, v_num=k1rk]

Epoch 0:   5%|▌         | 2/39 [00:08<02:29,  0.25it/s, v_num=k1rk]
Epoch 0:   5%|▌         | 2/39 [00:08<02:29,  0.25it/s, v_num=k1rk]

Epoch 0:   8%|▊         | 3/39 [00:08<01:37,  0.37it/s, v_num=k1rk]
Epoch 0:   8%|▊         | 3/39 [00:08<01:37,  0.37it/s, v_num=k1rk]

Epoch 0:  10%|█         | 4/39 [00:08<01:11,  0.49it/s, v_num=k1rk]
Epoch 0:  10%|█         | 4/39 [00:08<01:11,  0.49it/s, v_num=k1rk]

Epoch 0:  13%|█▎        | 5/39 [00:08<00:55,  0.61it/s, v_num=k1rk]
Epoch 0:  13%|█▎        | 5/39 [00:08<00:55,  0.61it/s, v_num=k1rk]

Epoch 0:  15%|█▌        | 6/39 [00:08<00:45,  0.73it/s, v_num=k1rk]
Epoch 0:  15%|█▌        | 6/39 [00:08<00:45,  0.73it/s, v_num=k1rk]

Epoch 0:  18%|█▊        | 7/39 [00:08<00:37,  0.85it/s, v_num=k1rk]
Epoch 0:  18%|█▊        | 7/39 [00:08<00:37,  0.85it/s, v_num=k1rk]

Epoch 0:  21%|██        | 8/39 [00:08<00:32,  0.96it/s, v_num=k1rk]
Epoch 0:  21%|██        | 8/39 [00:08<00:32,  0.96it/s, v_num=k1rk]

Epoch 0:  23%|██▎       | 9/39 [00:08<00:27,  1.08it/s, v_num=k1rk]
Epoch 0:  23%|██▎       | 9/39 [00:08<00:27,  1.08it/s, v_num=k1rk]

Epoch 0:  26%|██▌       | 10/39 [00:08<00:24,  1.19it/s, v_num=k1rk]
Epoch 0:  26%|██▌       | 10/39 [00:08<00:24,  1.19it/s, v_num=k1rk]

Epoch 0:  28%|██▊       | 11/39 [00:08<00:21,  1.31it/s, v_num=k1rk]
Epoch 0:  28%|██▊       | 11/39 [00:08<00:21,  1.31it/s, v_num=k1rk]

Epoch 0:  31%|███       | 12/39 [00:08<00:19,  1.42it/s, v_num=k1rk]
Epoch 0:  31%|███       | 12/39 [00:08<00:19,  1.42it/s, v_num=k1rk]

Epoch 0:  33%|███▎      | 13/39 [00:08<00:17,  1.53it/s, v_num=k1rk]
Epoch 0:  33%|███▎      | 13/39 [00:08<00:17,  1.53it/s, v_num=k1rk]

Epoch 0:  36%|███▌      | 14/39 [00:08<00:15,  1.64it/s, v_num=k1rk]
Epoch 0:  36%|███▌      | 14/39 [00:08<00:15,  1.64it/s, v_num=k1rk]

Epoch 0:  38%|███▊      | 15/39 [00:08<00:13,  1.75it/s, v_num=k1rk]
Epoch 0:  38%|███▊      | 15/39 [00:08<00:13,  1.75it/s, v_num=k1rk]

Epoch 0:  41%|████      | 16/39 [00:08<00:12,  1.86it/s, v_num=k1rk]
Epoch 0:  41%|████      | 16/39 [00:08<00:12,  1.86it/s, v_num=k1rk]

Epoch 0:  44%|████▎     | 17/39 [00:08<00:11,  1.96it/s, v_num=k1rk]
Epoch 0:  44%|████▎     | 17/39 [00:08<00:11,  1.96it/s, v_num=k1rk]

Epoch 0:  46%|████▌     | 18/39 [00:08<00:10,  2.07it/s, v_num=k1rk]
Epoch 0:  46%|████▌     | 18/39 [00:08<00:10,  2.07it/s, v_num=k1rk]

Epoch 0:  49%|████▊     | 19/39 [00:08<00:09,  2.17it/s, v_num=k1rk]
Epoch 0:  49%|████▊     | 19/39 [00:08<00:09,  2.17it/s, v_num=k1rk]

Epoch 0:  51%|█████▏    | 20/39 [00:08<00:08,  2.28it/s, v_num=k1rk]
Epoch 0:  51%|█████▏    | 20/39 [00:08<00:08,  2.28it/s, v_num=k1rk]

Epoch 0:  54%|█████▍    | 21/39 [00:08<00:07,  2.38it/s, v_num=k1rk]
Epoch 0:  54%|█████▍    | 21/39 [00:08<00:07,  2.38it/s, v_num=k1rk]

Epoch 0:  56%|█████▋    | 22/39 [00:08<00:06,  2.48it/s, v_num=k1rk]
Epoch 0:  56%|█████▋    | 22/39 [00:08<00:06,  2.48it/s, v_num=k1rk]

Epoch 0:  59%|█████▉    | 23/39 [00:10<00:07,  2.27it/s, v_num=k1rk]
Epoch 0:  59%|█████▉    | 23/39 [00:10<00:07,  2.27it/s, v_num=k1rk]

Epoch 0:  62%|██████▏   | 24/39 [00:10<00:06,  2.36it/s, v_num=k1rk]
Epoch 0:  62%|██████▏   | 24/39 [00:10<00:06,  2.36it/s, v_num=k1rk]

Epoch 0:  64%|██████▍   | 25/39 [00:10<00:05,  2.45it/s, v_num=k1rk]
Epoch 0:  64%|██████▍   | 25/39 [00:10<00:05,  2.45it/s, v_num=k1rk]

Epoch 0:  67%|██████▋   | 26/39 [00:10<00:05,  2.54it/s, v_num=k1rk]
Epoch 0:  67%|██████▋   | 26/39 [00:10<00:05,  2.54it/s, v_num=k1rk]

Epoch 0:  69%|██████▉   | 27/39 [00:10<00:04,  2.63it/s, v_num=k1rk]
Epoch 0:  69%|██████▉   | 27/39 [00:10<00:04,  2.63it/s, v_num=k1rk]

Epoch 0:  72%|███████▏  | 28/39 [00:10<00:04,  2.71it/s, v_num=k1rk]
Epoch 0:  72%|███████▏  | 28/39 [00:10<00:04,  2.71it/s, v_num=k1rk]

Epoch 0:  74%|███████▍  | 29/39 [00:10<00:03,  2.80it/s, v_num=k1rk]
Epoch 0:  74%|███████▍  | 29/39 [00:10<00:03,  2.80it/s, v_num=k1rk]

Epoch 0:  77%|███████▋  | 30/39 [00:10<00:03,  2.79it/s, v_num=k1rk]
Epoch 0:  77%|███████▋  | 30/39 [00:10<00:03,  2.79it/s, v_num=k1rk]

Epoch 0:  79%|███████▉  | 31/39 [00:10<00:02,  2.86it/s, v_num=k1rk]
Epoch 0:  79%|███████▉  | 31/39 [00:10<00:02,  2.86it/s, v_num=k1rk]

Epoch 0:  82%|████████▏ | 32/39 [00:10<00:02,  2.94it/s, v_num=k1rk]
Epoch 0:  82%|████████▏ | 32/39 [00:10<00:02,  2.94it/s, v_num=k1rk]

Epoch 0:  85%|████████▍ | 33/39 [00:10<00:01,  3.02it/s, v_num=k1rk]
Epoch 0:  85%|████████▍ | 33/39 [00:10<00:01,  3.02it/s, v_num=k1rk]

Epoch 0:  87%|████████▋ | 34/39 [00:10<00:01,  3.10it/s, v_num=k1rk]
Epoch 0:  87%|████████▋ | 34/39 [00:10<00:01,  3.10it/s, v_num=k1rk]

Epoch 0:  90%|████████▉ | 35/39 [00:10<00:01,  3.18it/s, v_num=k1rk]
Epoch 0:  90%|████████▉ | 35/39 [00:10<00:01,  3.18it/s, v_num=k1rk]

Epoch 0:  92%|█████████▏| 36/39 [00:11<00:00,  3.26it/s, v_num=k1rk]
Epoch 0:  92%|█████████▏| 36/39 [00:11<00:00,  3.26it/s, v_num=k1rk]

Epoch 0:  95%|█████████▍| 37/39 [00:11<00:00,  3.34it/s, v_num=k1rk]
Epoch 0:  95%|█████████▍| 37/39 [00:11<00:00,  3.34it/s, v_num=k1rk]

Epoch 0:  97%|█████████▋| 38/39 [00:11<00:00,  3.42it/s, v_num=k1rk]
Epoch 0:  97%|█████████▋| 38/39 [00:11<00:00,  3.42it/s, v_num=k1rk]

Epoch 0: 100%|██████████| 39/39 [00:11<00:00,  3.50it/s, v_num=k1rk]
Epoch 0: 100%|██████████| 39/39 [00:11<00:00,  3.50it/s, v_num=k1rk]

Validation: |          | 0/? [00:00<?, ?it/s][A

Validation:   0%|          | 0/5 [00:00<?, ?it/s][A

Validation DataLoader 0:   0%|          | 0/5 [00:00<?, ?it/s][A


Validation DataLoader 0:  20%|██        | 1/5 [00:00<00:01,  2.41it/s][A


Validation DataLoader 0:  40%|████      | 2/5 [00:00<00:00,  3.13it/s][A


Validation DataLoader 0:  60%|██████    | 3/5 [00:00<00:00,  4.49it/s][A


Validation DataLoader 0:  80%|████████  | 4/5 [00:00<00:00,  5.71it/s][A


Validation DataLoader 0: 100%|██████████| 5/5 [00:00<00:00,  6.84it/s][A

                                                                      [A
Epoch 0: 100%|██████████| 39/39 [00:23<00:00,  1.69it/s, v_num=k1rk]
Epoch 0: 100%|██████████| 39/39 [00:23<00:00,  1.69it/s, v_num=k1rk]panic: json unmarshal error: json: invalid character 'N' looking for beginning of value: NaN, items: key:"val/pearson" value_json:"NaN"

goroutine 86 [running]:
github.com/wandb/wandb/core/pkg/observability.(*CoreLogger).CaptureFatalAndPanic(...)
 /project/core/pkg/observability/logging.go:142
github.com/wandb/wandb/core/pkg/filestream.(*FileStream).streamHistory(0xc0004c3a70, 0x0?)
 /project/core/pkg/filestream/loop_process.go:83 +0x245
github.com/wandb/wandb/core/pkg/filestream.(*FileStream).processRecord(0xc0004c3a70, 0xb054e8?)
 /project/core/pkg/filestream/loop_process.go:31 +0xdd
github.com/wandb/wandb/core/pkg/filestream.(*FileStream).loopProcess(0xc0004c3a70, 0xc0004962a0)
 /project/core/pkg/filestream/loop_process.go:58 +0x279
github.com/wandb/wandb/core/pkg/filestream.(*FileStream).Start.func1()
 /project/core/pkg/filestream/filestream.go:194 +0x25
created by github.com/wandb/wandb/core/pkg/filestream.(*FileStream).Start in goroutine 14
 /project/core/pkg/filestream/filestream.go:193 +0xfd

Epoch 0:   0%|          | 0/39 [00:00<?, ?it/s, v_num=k1rk]         
Epoch 1:   0%|          | 0/39 [00:00<?, ?it/s, v_num=k1rk]/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028


Epoch 1:   3%|▎         | 1/39 [00:00<00:26,  1.46it/s, v_num=k1rk]
Epoch 1:   3%|▎         | 1/39 [00:00<00:26,  1.46it/s, v_num=k1rk]

Epoch 1:   5%|▌         | 2/39 [00:00<00:15,  2.42it/s, v_num=k1rk]
Epoch 1:   5%|▌         | 2/39 [00:00<00:15,  2.42it/s, v_num=k1rk]

Epoch 1:   8%|▊         | 3/39 [00:01<00:13,  2.62it/s, v_num=k1rk]
Epoch 1:   8%|▊         | 3/39 [00:01<00:13,  2.62it/s, v_num=k1rk]

Epoch 1:  10%|█         | 4/39 [00:01<00:10,  3.30it/s, v_num=k1rk]
Epoch 1:  10%|█         | 4/39 [00:01<00:10,  3.30it/s, v_num=k1rk]

Epoch 1:  13%|█▎        | 5/39 [00:01<00:08,  3.98it/s, v_num=k1rk]
Epoch 1:  13%|█▎        | 5/39 [00:01<00:08,  3.98it/s, v_num=k1rk]

Epoch 1:  15%|█▌        | 6/39 [00:01<00:07,  4.61it/s, v_num=k1rk]
Epoch 1:  15%|█▌        | 6/39 [00:01<00:07,  4.60it/s, v_num=k1rk]

Epoch 1:  18%|█▊        | 7/39 [00:01<00:07,  4.20it/s, v_num=k1rk]
Epoch 1:  18%|█▊        | 7/39 [00:01<00:07,  4.20it/s, v_num=k1rk]

Epoch 1:  21%|██        | 8/39 [00:01<00:06,  4.69it/s, v_num=k1rk]
Epoch 1:  21%|██        | 8/39 [00:01<00:06,  4.69it/s, v_num=k1rk]

Epoch 1:  23%|██▎       | 9/39 [00:01<00:05,  5.16it/s, v_num=k1rk]
Epoch 1:  23%|██▎       | 9/39 [00:01<00:05,  5.16it/s, v_num=k1rk]

Epoch 1:  26%|██▌       | 10/39 [00:01<00:05,  5.61it/s, v_num=k1rk]
Epoch 1:  26%|██▌       | 10/39 [00:01<00:05,  5.61it/s, v_num=k1rk]

Epoch 1:  28%|██▊       | 11/39 [00:01<00:04,  6.04it/s, v_num=k1rk]
Epoch 1:  28%|██▊       | 11/39 [00:01<00:04,  6.04it/s, v_num=k1rk]Error executing job with overrides: []
Traceback (most recent call last):
  File "/projects/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/deep_set.py", line 205, in main
    trainer.fit(task, data_module)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 544, in fit
    call._call_and_handle_interrupt(
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 44, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 580, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 989, in _run
    results = self._run_stage()
              ^^^^^^^^^^^^^^^^^
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1035, in _run_stage
    self.fit_loop.run()
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loops/fit_loop.py", line 202, in run
    self.advance()
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loops/fit_loop.py", line 359, in advance
    self.epoch_loop.run(self._data_fetcher)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loops/training_epoch_loop.py", line 136, in run
    self.advance(data_fetcher)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loops/training_epoch_loop.py", line 268, in advance
    trainer._logger_connector.update_train_step_metrics()
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py", line 155, in update_train_step_metrics
    self.log_metrics(self.metrics["log"])
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py", line 109, in log_metrics
    logger.log_metrics(metrics=scalar_metrics, step=step)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning_utilities/core/rank_zero.py", line 43, in wrapped_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loggers/wandb.py", line 425, in log_metrics
    self.experiment.log(dict(metrics, **{"trainer/global_step": step}))
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/wandb_run.py", line 420, in wrapper
    return func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/wandb_run.py", line 371, in wrapper_fn
    return func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/wandb_run.py", line 361, in wrapper
    return func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/wandb_run.py", line 1838, in log
    self._log(data=data, step=step, commit=commit)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/wandb_run.py", line 1602, in _log
    self._partial_history_callback(data, step, commit)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/wandb_run.py", line 1474, in _partial_history_callback
    self._backend.interface.publish_partial_history(
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/interface/interface.py", line 602, in publish_partial_history
    self._publish_partial_history(partial_history)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/interface/interface_shared.py", line 89, in _publish_partial_history
    self._publish(rec)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/interface/interface_sock.py", line 51, in _publish
    self._sock_client.send_record_publish(record)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/lib/sock_client.py", line 221, in send_record_publish
    self.send_server_request(server_req)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/lib/sock_client.py", line 155, in send_server_request
    self._send_message(msg)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/lib/sock_client.py", line 152, in _send_message
    self._sendall_with_error_handle(header + data)
  File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/wandb/sdk/lib/sock_client.py", line 130, in _sendall_with_error_handle
    sent = self._sock.send(data)
           ^^^^^^^^^^^^^^^^^^^^^
BrokenPipeError: [Errno 32] Broken pipe

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
wandb: While tearing down the service manager. The following error has occurred: [Errno 32] Broken pipe

Epoch 1:  28%|██▊       | 11/39 [00:04<00:11,  2.51it/s, v_num=k1rk]
```
