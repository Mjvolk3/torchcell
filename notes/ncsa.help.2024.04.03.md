---
id: d9b45j7ffn9g89u7f9x03sn
title: '03'
desc: ''
updated: 1712162856134
created: 1712160610661
---
Hi there,

I have been trying to run the following slurm script and keep getting an out of memory error.

## Python script that is being run

First it might be nice to see at least the main function of the python script so I can explain the discrepancy between batch and interactive.

```python
def main():
    from dotenv import load_dotenv
    import wandb
    from time import sleep
    print("Starting main...")
    wandb.init(mode="online", project="torchcell_embeddings")
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
 
    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
    model_names = [
        "nt_window_5979",
        "nt_window_5979_max",
        "nt_window_3utr_5979",
        "nt_window_3utr_5979_undersize",
        "nt_window_5utr_5979",
        "nt_window_5utr_5979_undersize",
        "nt_window_3utr_300",
        "nt_window_3utr_300_undersize",
        "nt_window_5utr_1000",
        "nt_window_5utr_1000_undersize",
    ]
    event = 0
    for model_name in model_names:
        print(f"event: {event}")
        print(f"starting model_name: {model_name}")
        wandb.log({"event": event})
        dataset = NucleotideTransformerDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embed"),
            genome=genome,
            model_name=model_name,
        )
        print(f"Completed Dataset for {model_name}: {dataset}")
        event += 1
 

if __name__ == "__main__":
    main()
```

The python script iterates over model names and computes embeddings with a transformer model. Before any of that happens we should see a print statement “Starting main…”

## Running program with batch script.

SLURM script:

```bash
#!/bin/bash
#SBATCH --mem=64 # up to 256 gb cpu... 256 doesn't work, 128 does... for cpu but not for gpu? 64 works for gpu.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=embed
#SBATCH --time=18:00:00      # hh:mm:ss for the job, 48 hr max.
#SBATCH --constraint="projects&scratch"
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
# #SBATCH --output=/dev/null
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/embeddings/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/embeddings/slurm/output/%x_%j.out
 

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
source ~/.bashrc
cd /projects/bbub/mjvolk3/torchcell
pwd
lscpu
nvidia-smi
cat /proc/meminfo
#module load anaconda3_cpu
module list  # job documentation and metadata
conda activate /projects/bbub/miniconda3/envs/torchcell
echo "Starting Embedding Computation..."
python experiments/embeddings/compute_nucleotide_transformer_embeddings.py
```

OOM Error:

```bash
Running "module reset". Resetting modules to system default. The following $MODULEPATH directories have been removed: None
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
CPU MHz:             2872.056
BogoMIPS:            4890.55
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
Tue Apr  2 18:47:56 2024      
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A40                     On  | 00000000:C7:00.0 Off |                    0 |
|  0%   27C    P8              21W / 300W |      0MiB / 46068MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                        
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
MemTotal:       263842016 kB
MemFree:        218140496 kB
MemAvailable:   229309904 kB
Buffers:            2112 kB
Cached:         24071372 kB
SwapCached:            0 kB
Active:         10882484 kB
Inactive:       32111940 kB
Active(anon):    4463988 kB
Inactive(anon): 26280152 kB
Active(file):    6418496 kB
Inactive(file):  5831788 kB
Unevictable:       87732 kB
Mlocked:           84660 kB
SwapTotal:             0 kB
SwapFree:              0 kB
Dirty:             21068 kB
Writeback:             0 kB
AnonPages:      18977052 kB
Mapped:          2287128 kB
Shmem:          11823108 kB
KReclaimable:     491704 kB
Slab:            1378484 kB
SReclaimable:     491704 kB
SUnreclaim:       886780 kB
KernelStack:       25440 kB
PageTables:        79100 kB
NFS_Unstable:          0 kB
Bounce:                0 kB
WritebackTmp:          0 kB
CommitLimit:    131921008 kB
Committed_AS:   35035416 kB
VmallocTotal:   34359738367 kB
VmallocUsed:      363948 kB
VmallocChunk:          0 kB
Percpu:            64256 kB
HardwareCorrupted:     0 kB
AnonHugePages:  11958272 kB
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
DirectMap4k:     4603828 kB
DirectMap2M:    136683520 kB
DirectMap1G:    126877696 kB
 
Currently Loaded Modules:
  1) gcc/11.4.0      3) cuda/11.8.0         5) slurm-env/0.1
  2) openmpi/4.1.6   4) cue-login-env/1.0   6) default-s11
 
 
Starting Embedding Computation...
/var/spool/slurmd/job3347872/slurm_script: line 34: 1657575 Killed                  python experiments/embeddings/compute_nucleotide_transformer_embeddings.py
slurmstepd: error: Detected 1 oom_kill event in StepId=3347872.batch. Some of the step tasks have been OOM Killed.
```

Since In the batch setting I redirect stderror and stdout to the out file so if it prints we should see it. Since we do not see the print statement from python I assume that we are getting the OOM before we even get to the main function. I have attempted to scale up resources and this does not solve the issue.

## Running the script on an interactive GPU.

When we the batch script (line by line) on the interactive GPU with I believe to be equivalent settings. We see the print statement “starting main…”, then we see the rest of the logging, print statements, progress bars, etc.

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

My thoughts are that somehow my interactive node does not reflect the settings of the batch script. I then considered that we get the OOM when we try to load the model, but you can see for this script that we print “Starting main…” before models load so it cannot be that.

Any assistance in trouble shooting this would be appreciated thanks.

Sincerely,

Michael Volk
Ph.D. Candidate
Zhao Research Group
Chemical and Biomolecular Engineering
University of Illinois Urbana-Champaign

## 2024.04.03 - Potential Follow Up - Smaller Loop on Interactive GPU

```bash
ode=1 --tasks=1 --tasks-per-node=1 --cpus-per-task=1 --mem=62g --pty bash 

srun: job 3351852 queued and waiting for resources
srun: job 3351852 has been allocated resources
conda activate /projects/bbub/miniconda3/envs/torchcell
GpuFreq=control_disabled
conda activate /projects/bbub/miniconda3/envs/torchcell
(base) [mjvolk3@gpub002 torchcell]$ conda activate /projects/bbub/miniconda3/envs/torchcell
(torchcell) [mjvolk3@gpub002 torchcell]$ python experiments/embeddings/compute_nucleotide_transformer_embeddings.py
Starting main...
wandb: Currently logged in as: mjvolk3 (zhao-group). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.16.5
wandb: Run data is saved locally in /projects/bbub/mjvolk3/torchcell/wandb/run-20240403_112427-nkpx4kx1
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run solar-deluge-18
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_embeddings
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_embeddings/runs/nkpx4kx1/workspace
Downloading http://current.geneontology.org/ontology/go.obo
data/go/go.obo: fmt(1.2) rel(2024-01-17) 45,869 Terms
event: 0
starting model_name: nt_window_5979
Processing...
Done!
/scratch/bbub/mjvolk3/torchcell/data/scerevisiae/nucleotide_transformer_embed/processed/nt_window_5979.pt
Completed Dataset for nt_window_5979: NucleotideTransformerDataset(11)
event: 1
starting model_name: nt_window_5979_max
Processing...
Done!
/scratch/bbub/mjvolk3/torchcell/data/scerevisiae/nucleotide_transformer_embed/processed/nt_window_5979_max.pt
Completed Dataset for nt_window_5979_max: NucleotideTransformerDataset(11)
event: 2
starting model_name: nt_window_three_prime_5979
Processing...
Done!
/scratch/bbub/mjvolk3/torchcell/data/scerevisiae/nucleotide_transformer_embed/processed/nt_window_three_prime_5979.pt
Downloading InstaDeepAI/nucleotide-transformer-2.5b-multi-species model to /projects/bbub/mjvolk3/torchcell/torchcell/models/pretrained_LLM/nucleotide_transformer/InstaDeepAI/nucleotide-transformer-2.5b-multi-species...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [01:00<00:00, 30.32s/it]
Download finished.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.72s/it]
4it [00:00, 53.85it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:21<00:00,  4.28s/it]
Completed Dataset for nt_window_three_prime_5979: NucleotideTransformerDataset(5)
event: 3
starting model_name: nt_window_five_prime_5979
Processing...
Done!
/scratch/bbub/mjvolk3/torchcell/data/scerevisiae/nucleotide_transformer_embed/processed/nt_window_five_prime_5979.pt
Downloading InstaDeepAI/nucleotide-transformer-2.5b-multi-species model to /projects/bbub/mjvolk3/torchcell/torchcell/models/pretrained_LLM/nucleotide_transformer/InstaDeepAI/nucleotide-transformer-2.5b-multi-species...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.49s/it]
Download finished.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.31s/it]
4it [00:00, 51.98it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.58it/s]
Completed Dataset for nt_window_five_prime_5979: NucleotideTransformerDataset(5)
event: 4
starting model_name: nt_window_three_prime_300
Processing...
Done!
/scratch/bbub/mjvolk3/torchcell/data/scerevisiae/nucleotide_transformer_embed/processed/nt_window_three_prime_300.pt
Downloading InstaDeepAI/nucleotide-transformer-2.5b-multi-species model to /projects/bbub/mjvolk3/torchcell/torchcell/models/pretrained_LLM/nucleotide_transformer/InstaDeepAI/nucleotide-transformer-2.5b-multi-species...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.32s/it]
Download finished.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.12s/it]
4it [00:00, 56.61it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.72it/s]
Completed Dataset for nt_window_three_prime_300: NucleotideTransformerDataset(5)
event: 5
starting model_name: nt_window_five_prime_1003
Processing...
Done!
/scratch/bbub/mjvolk3/torchcell/data/scerevisiae/nucleotide_transformer_embed/processed/nt_window_five_prime_1003.pt
Downloading InstaDeepAI/nucleotide-transformer-2.5b-multi-species model to /projects/bbub/mjvolk3/torchcell/torchcell/models/pretrained_LLM/nucleotide_transformer/InstaDeepAI/nucleotide-transformer-2.5b-multi-species...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.38s/it]
Download finished.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.09s/it]
4it [00:00, 49.65it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.70it/s]
Completed Dataset for nt_window_five_prime_1003: NucleotideTransformerDataset(5)
wandb: / 0.022 MB of 0.022 MB uploaded
wandb: Run history:
wandb: event ▁▂▄▅▇█
wandb: 
wandb: Run summary:
wandb: event 5
wandb: 
wandb: 🚀 View run solar-deluge-18 at: https://wandb.ai/zhao-group/torchcell_embeddings/runs/nkpx4kx1/workspace
wandb: Synced 5 W&B file(s), 0 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20240403_112427-nkpx4kx1/logs
(torchcell) [mjvolk3@gpub002 torchcell]$ 
```

## 2024.04.03 - Memory Formatting is MB

Should have review the slurm docs early. I found this by checking the interactive srun that appends `g` to denote GB. This explains the discrepancy.
