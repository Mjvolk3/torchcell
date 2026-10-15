---
id: iu1hc4hdk2pgrouxq7l0qey
title: Gilahyper_generate_adapter_array-slurm_docker
desc: ''
updated: 1792055272817
created: 1792055272817
---

## 2026.10.15 - One array task per adapter, with per-task recovery

`database/slurm/scripts/gilahyper_generate_adapter_array-slurm_docker.slurm` generates ONE adapter's CSVs per array task, in its own container `tc-neo4j-gen-<i>`, into `$SERVE_ROOT/biocypher-out/$OUT_NAME/adapters/<iii>_<Adapter>/`. It is the generation stage of the live rebuild under `GENERATION_MODE=array` ([[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]) and the vehicle of the Hoepfner measurement ([[experiments.database.hoepfner-generation-study]]). The host-side logic is [[torchcell.knowledge_graphs.generation_array]], the container-side entry [[torchcell.knowledge_graphs.generate_adapter]].

- **Submission is through the conductor** ([[database.scripts.kg_generation_array_conductor]]), never a bare `sbatch --array`: the conductor creates the output directory through a container (biocypher-out is uid 7474's), records `adapter_order.json` and `dev_fence.txt` there, and submits one array per tier of [[torchcell.knowledge_graphs.conf.generation_tiers.yaml]] with that tier's `--cpus-per-task`, `--mem`, `--time` and `%N`. A task refuses when the order or the fence is absent.
- **The container limits are the tier's**, not the allocation's: `--cpus=$GEN_CPUS --memory=${GEN_MEM_GB}g`, with `SLURM_CPUS_PER_TASK=$GEN_CPUS` inside so the adapter's worker count is the tier's cpus. Docker runs outside the slurm cgroup; the slurm `--mem` is a reservation and the docker limit is the cap, and both come from the same tier row (`MaxMemPerCPU=4100` rounds the allocation's CPU count up for the 160 G and 320 G tiers; the container keeps the tier's).
- **The fence is re-checked per task.** `generation_array fence` re-digests every mapped store's build manifest and diffs against the conductor's `dev_fence.txt` before the container starts, so a store rebuilt under the array stops the next task at once instead of after the whole generation.
- **DONE is the generator's last write; the task only verifies it.** A directory without a marker is incomplete whatever it holds; a previous incomplete attempt is moved aside as `<dir>.incomplete.<ts>` (never written into), and a directory that already carries a marker ends the task with nothing to do.
- **Recovery** is the header's first paragraph: `generation_array missing --out-dir <out> ...` prints the indices without a marker and the exact per-tier `sbatch --array=<ids>%N` lines; then `generation_array merge`. Under the live rebuild, `RESUME_FROM=array` does both.
- **A cancelled task's container outlives it** (the live rebuild's lesson, skill section 0): `docker rm -f tc-neo4j-gen-<i>` before re-running that index; the task refuses while the name exists.

Logs: `$BUILD_ROOT/database/slurm/output/<array job>_<i>_gen_adapter.out` (the task) and `<array job>_<i>_generate.log` (the container's stream, which is also the `--generate-log` of `kg_build_adapter_resources.py`).

Tests: `tests/torchcell/database/test_generate_adapter_array_slurm.py` pins the container sizing, the marker contract, the fence re-check, the documented recovery and the conductor's steps.

## 2026.10.15 - GEN_IMAGE: the task runs the prebuilt image, or refuses

With `GEN_IMAGE` set (the conductor's resolved tag), the task's preflight checks that the
tag is the one `generation_image tag --build-commit $BUILD_COMMIT` names AND that the
image's own `org.torchcell.build_commit` label is that commit, then runs the container
from it with no pip stage. A tag that is gone or re-tagged fails the task; there is no
install fallback, because a silent fallback would tie the CSVs to a commit the image name
no longer matches. With `GEN_IMAGE` unset the container runs the same three installs in
the same order as before.

Measured over the 47 tasks of job 4974: 11.0 s median from the `1 GENERATE` line to the
generator's first logged line, against 278 s on the install path in job 4919_64
([[experiments.database.bacterial-generation-profile]]).
