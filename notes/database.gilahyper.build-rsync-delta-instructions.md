---
id: nhm52r608rqgj8b51v1q3je
title: Build Rsync Delta Instructions
desc: ''
updated: 1765931284752
created: 1765319911784
---
## Database Build and Sync to Production

This protocol documents the complete workflow for building the TorchCell Neo4j knowledge graph on GilaHyper and syncing it to Delta NCSA for production use.

### Overview

| Step | Script                                       | Resources     | Purpose               |
|------|----------------------------------------------|---------------|-----------------------|
| 1    | `gilahyper_build-slurm_docker.slurm`         | 64 CPU, 500GB | Build KG              |
| 2    | `scancel`                                    | -             | Free resources        |
| 3    | `gilahyper_run_container-slurm_docker.slurm` | 8 CPU, 64GB   | Run container         |
| 4    | `export_database.sh`                         | -             | Export dump           |
| 5a   | `ssh -fN dt-login.delta...`                  | -             | Persistent connection |
| 5b   | `rsync_transfer_gh_to_delta.slurm`           | 1 CPU, 4GB    | Transfer to Delta     |
| 6    | `scancel`                                    | -             | Clean up              |

### Step 1: Build the Knowledge Graph

```bash
sbatch database/slurm/scripts/gilahyper_build-slurm_docker.slurm
```

**What it does:**

- Starts Neo4j container with 64 CPUs, 500GB RAM
- Installs latest `torchcell`, `biocypher` from GitHub
- Builds knowledge graph using config: `torchcell/knowledge_graphs/conf/tmi_tmf_kg.yaml`
- Bulk imports data into Neo4j
- Runs until cancelled (cleanup trap removes container)

**Monitor:**

```bash
squeue -u $USER
tail -f /scratch/projects/torchcell-scratch/database/slurm/output/<job_id>_build_database.out
```

**Duration:** Several hours depending on dataset size.

***

### Step 2: Cancel Build Job (Free Resources)

Once the build completes (check log for "Knowledge graph creation completed"):

```bash
scancel <build_job_id>
```

**What happens:**

- Cleanup trap stops and removes the container
- Data persists in `/scratch/projects/torchcell-scratch/database/data/`
- Frees up 64 CPUs and 500GB RAM

***

### Step 3: Start Container for Export

```bash
sbatch database/slurm/scripts/gilahyper_run_container-slurm_docker.slurm
```

**What it does:**

- Starts lightweight container (8 CPUs, 64GB RAM)
- Mounts existing data (no rebuild)
- Neo4j ready in ~30-60 seconds

**Monitor:**

```bash
squeue -u $USER
# Wait for "Neo4j is ready and responding!" in output
```

***

### Step 4: Export the Database

```bash
bash database/scripts/export_database.sh
```

**What it does:**

- Stops database for consistent export
- Creates dump: `torchcell_YYYYMMDD_HHMMSS.dump`
- Updates symlink: `torchcell_latest.dump`
- Restarts database

**Output location:** `/scratch/projects/torchcell-scratch/database/export/`

**Verify:**

```bash
ls -la /scratch/projects/torchcell-scratch/database/export/
```

***

### Step 5: Transfer to Delta

Delta requires two-factor authentication. SSH multiplexing is configured in `~/.ssh/config` with `ControlPersist 4h` (connection stays open 4 hours after last use).

#### 5a. Open persistent SSH connection (2FA once)

```bash
ssh -O check dt-login.delta.ncsa.illinois.edu || ssh -fN dt-login.delta.ncsa.illinois.edu
# Complete 2FA once, then all subsequent SSH/rsync reuses this connection
```

#### 5b. Submit transfer job

```bash
sbatch database/slurm/scripts/rsync_transfer_gh_to_delta.slurm
```

**What it does:**

- Resolves `torchcell_latest.dump` symlink to actual timestamped file
- Transfers via rsync with checksum verification
- Creates `torchcell_latest.dump` symlink on Delta
- Verifies MD5 checksums match

**Destination:** `/projects/bbub/mjvolk3/torchcell/database/import/`

**Monitor:**

```bash
tail -f /scratch/projects/torchcell-scratch/database/slurm/output/<job_id>_rsync_to_delta.out
```

***

### Step 6: Clean Up

Cancel the run container job:

```bash
scancel <run_job_id>
```

Close SSH master connection (optional):

```bash
ssh -O exit dt-login.delta.ncsa.illinois.edu
```

***

### Troubleshooting

#### Container doesn't exist for export

The build job was cancelled/ended. Run Step 3 to start the container.

#### SSH "Host key verification failed"

SSH to Delta interactively once to add the host key:

```bash
ssh mjvolk3@dt-login.delta.ncsa.illinois.edu
# Type 'yes' when prompted
```

#### Transfer shows old date

The symlink points to an old dump. Re-run `export_database.sh` (Step 4).

#### Rsync transfers symlink (30 bytes) instead of file

The script should resolve symlinks automatically. If not, check that `readlink -f` works on the source path.

***

### File Locations

| Description          | Path                                                                |
|----------------------|---------------------------------------------------------------------|
| Build script         | `database/slurm/scripts/gilahyper_build-slurm_docker.slurm`         |
| Run container script | `database/slurm/scripts/gilahyper_run_container-slurm_docker.slurm` |
| Export script        | `database/scripts/export_database.sh`                               |
| Transfer script      | `database/slurm/scripts/rsync_transfer_gh_to_delta.slurm`           |
| SLURM output         | `/scratch/projects/torchcell-scratch/database/slurm/output/`        |
| Export dumps         | `/scratch/projects/torchcell-scratch/database/export/`              |
| Neo4j data           | `/scratch/projects/torchcell-scratch/database/data/`                |
| Delta import         | `/projects/bbub/mjvolk3/torchcell/database/import/`                 |
| KG config            | `torchcell/knowledge_graphs/conf/tmi_tmf_kg.yaml`                   |
