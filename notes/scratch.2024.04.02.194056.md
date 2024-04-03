---
id: zet0wjaifeyt9q9qqz6hicn
title: '194056'
desc: ''
updated: 1712106667489
created: 1712104858960
---
```bash
 *  Executing task in folder torchcell: srun --account=bbub-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=1 --cpus-per-task=1 --mem=62g --pty bash 

srun: job 3347856 queued and waiting for resources
srun: job 3347856 has been allocated resources
conda activate /projects/bbub/miniconda3/envs/torchcell
GpuFreq=control_disabled
conda activate /projects/bbub/miniconda3/envs/torchcell
(base) [mjvolk3@gpub001 torchcell]$ conda activate /projects/bbub/miniconda3/envs/torchcell
(torchcell) [mjvolk3@gpub001 torchcell]$ module reset
Running "module reset". Resetting modules to system default. The following $MODULEPATH directories have been removed: None
(torchcell) [mjvolk3@gpub001 torchcell]$ source ~/.bashrc

(base) [mjvolk3@gpub001 torchcell]$ 
(base) [mjvolk3@gpub001 torchcell]$ cd /projects/bbub/mjvolk3/torchcell
(base) [mjvolk3@gpub001 torchcell]$ pwd
/projects/bbub/mjvolk3/torchcell
(base) [mjvolk3@gpub001 torchcell]$ 
(base) [mjvolk3@gpub001 torchcell]$ lscpu
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
CPU MHz:             2445.331
BogoMIPS:            4890.66
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
(base) [mjvolk3@gpub001 torchcell]$ nvidia-smi 
Tue Apr  2 18:44:20 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A40                     On  | 00000000:85:00.0 Off |                    0 |
|  0%   29C    P8              21W / 300W |      0MiB / 46068MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+

(base) [mjvolk3@gpub001 torchcell]$ 
(base) [mjvolk3@gpub001 torchcell]$ cat /proc/meminfo
MemTotal:       263842016 kB
MemFree:        247333696 kB
MemAvailable:   248562548 kB
Buffers:            2112 kB
Cached:         13478672 kB
SwapCached:            0 kB
Active:          4773196 kB
Inactive:        9228352 kB
Active(anon):    4381932 kB
Inactive(anon):  7263900 kB
Active(file):     391264 kB
Inactive(file):  1964452 kB
Unevictable:       85620 kB
Mlocked:           82548 kB
SwapTotal:             0 kB
SwapFree:              0 kB
Dirty:               340 kB
Writeback:             0 kB
AnonPages:        604004 kB
Mapped:           339104 kB
Shmem:          11125068 kB
KReclaimable:     424300 kB
Slab:            1537144 kB
SReclaimable:     424300 kB
SUnreclaim:      1112844 kB
KernelStack:       18224 kB
PageTables:        13316 kB
NFS_Unstable:          0 kB
Bounce:                0 kB
WritebackTmp:          0 kB
CommitLimit:    131921008 kB
Committed_AS:   13350416 kB
VmallocTotal:   34359738367 kB
VmallocUsed:      334476 kB
VmallocChunk:          0 kB
Percpu:            56320 kB
HardwareCorrupted:     0 kB
AnonHugePages:    391168 kB
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
DirectMap4k:    22267828 kB
DirectMap2M:    171448320 kB
DirectMap1G:    74448896 kB
(base) [mjvolk3@gpub001 torchcell]$ module list  # job documentation and metadata

Currently Loaded Modules:
  1) gcc/11.4.0   2) openmpi/4.1.6   3) cuda/11.8.0   4) cue-login-env/1.0   5) slurm-env/0.1   6) default-s11

 

(base) [mjvolk3@gpub001 torchcell]$ conda activate /projects/bbub/miniconda3/envs/torchcell
(torchcell) [mjvolk3@gpub001 torchcell]$ echo "Starting Embedding Computation..."
Starting Embedding Computation...
(torchcell) [mjvolk3@gpub001 torchcell]$ python experiments/embeddings/compute_nucleotide_transformer_embeddings.py

Starting main...
wandb: Currently logged in as: mjvolk3 (zhao-group). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.16.5
wandb: Run data is saved locally in /projects/bbub/mjvolk3/torchcell/wandb/run-20240402_184756-crtwl8wy
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run wobbly-frost-4
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_embeddings
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_embeddings/runs/crtwl8wy/workspace
data/go/go.obo: fmt(1.2) rel(2023-07-27) 46,356 Terms
event: 0
starting model_name: nt_window_5979
Processing...
Done!
/scratch/bbub/mjvolk3/torchcell/data/scerevisiae/nucleotide_transformer_embed/processed/nt_window_5979.pt
Downloading InstaDeepAI/nucleotide-transformer-2.5b-multi-species model to /projects/bbub/mjvolk3/torchcell/torchcell/models/pretrained_LLM/nucleotide_transformer/InstaDeepAI/nucleotide-transformer-2.5b-multi-species...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [01:21<00:00, 40.76s/it]
Download finished.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.58s/it]
100%|███████████████████████████████████████████████████████████████████████████████████| 6607/6607 [00:52<00:00, 125.52it/s]
 52%|███████████████████████████████████████████▋                                        | 3439/6607 [20:57<19:17,  2.74it/s]srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
slurmstepd: error: *** STEP 3347856.0 ON gpub001 CANCELLED AT 2024-04-02T19:13:08 DUE TO TIME LIMIT ***
 53%|████████████████████████████████████████████▊                                       | 3521/6607 [21:27<18:47,  2.74it/s]srun: error: Timed out waiting for job step to complete
 *  Terminal will be reused by tasks, press any key to close it. 
```