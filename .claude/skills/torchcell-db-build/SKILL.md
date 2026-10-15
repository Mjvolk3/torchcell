---
name: torchcell-db-build
description: Run, watch, resume and release a full torchcell knowledge-graph build on GilaHyper (the live rebuild under slurm plus docker), then ship it to Taiga and swap Radiant onto it. Carries the lessons of every full build since KG 3.0 -- the pairing tag, the comma split, the leftover container, the 12 h limit, the container that outlives its job, the resource telemetry and the memory-throttled adapters. Use for any `live_rebuild_kg` submission, check-in, failure or release step.
---

# torchcell-db-build

The full build is `database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm`
([[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]): a slurm job that
runs an ephemeral generation container (`tc-neo4j-gen`, reads the dev-tree LMDBs
read-only, writes CSVs to `/db/database/biocypher-out/<ts>`), a build container
(`tc-neo4j-build`, `neo4j-admin import` into `/db/database-next`), a validation, a
directory swap under `tc-neo4j-readonly`, and the stamp (manifest, KgRelease node,
release snapshot). Releasing and serving are `scripts/kg_release.sh`
([[scripts.kg_release]]). The older in-place build (`gilahyper_first_build`,
`build_database` jobs) has its own skill, `/monitor-tcdb-build`; it is superseded for
full builds.

Every number below is measured on a named job. The KG 4.0 generation, job 4901
(2026-10-13 22:27 CDT, 170 datasets, three organisms, v1.7.4), is the calibration run.

Before any build that follows a schema change, the change itself must have passed `/db-constitution` ([[database.constitution]]): the build is principle 12's gate, and it rebuilds whatever a correct change moves.

## 0. The one fact that shapes everything: docker is outside slurm

The containers are started by the docker daemon, so **they are not in the job's cgroup**.
Consequences, all observed:

- `sacct` reports `TotalCPU 00:00:00` and no `MaxRSS` for a build job. Slurm never
  accounted the build's real use. The accounting lives in the container's own telemetry
  (section 5), not in slurm.
- **`scancel` and a time limit kill the job, not the container.** Job 4899 was cancelled
  and its `tc-neo4j-gen` kept running; job 4900 then died in 11 s with
  "tc-neo4j-gen already exists". Job 4901 hit its 12 h limit at 10:27 CDT 2026-10-14 and
  its generation kept writing CSVs for hours afterwards.
- **A dead job is a resource clash.** Once slurm frees the job's 64 CPUs and 256 G, it
  schedules other work onto them while the container still uses them (job 4901's
  container ran at a mean 20 cores and 213 G with no slurm reservation behind it). After
  any job death, decide at once: keep the container and resume (section 4), or
  `docker rm -f tc-neo4j-gen` so the node is honest again. Never leave it undecided.
- `--rm` on the generation container removes it, **and its log**, the moment it exits.
  The job's own capture (`docker run ... | tee <job>_generate.log`) dies with the job,
  so a surviving container's output must be captured by hand (section 4).

## 1. Before submitting (the preflight the script cannot do for you)

Run from the primary checkout on `main`, clean and equal to `origin/main`.

1. **Every mapped dev store is fresh.** `python -m torchcell.database.build_dataset_lmdb
   --list-stale --include-private --organisms scerevisiae,ecoli,pputida` prints nothing.
   A stale set is rebuilt by the memory-tiered sweep (`gilahyper_build_dataset_lmdbs_array.slurm`
   with `--retire-existing --verify`; each store build is single-threaded, memory is the
   binding constraint, peaks measured in `store_rss_recent.tsv`: Nadal-Ribelles 77 G,
   Wildenhain 39 G, Mutalik 37 G, Hale 36 G). The sweep keeps `*.superseded.<ts>` copies
   that are never pruned: on 2026-10-13 2,083 of them (1.9 T) filled `/scratch` to 100%
   mid-sweep. Prune to the newest per store before a sweep, and check `df` first.
2. **Served counts do not decrease.** `python scripts/served_count_drift.py` after the
   sweep; a decreased count is a dropped record and stops the build.
3. **The release pair exists.** The build commit must carry a package tag. A `DB(kg)`
   commit (a dated entry in `notes/versioning.md`) lands through the merge queue;
   semantic-release pushes the `vX.Y.Z` bump commit; the build runs from a worktree
   checked out AT that commit:
   `BUILD_COMMIT=$(git rev-parse 'v1.7.4^{commit}')` -- **the tag's commit, never the
   annotated tag object** (job 4898 refused on the object).
4. **Multi-member sets go through the environment, not `--export`.** sbatch splits the
   export list on commas: `--export=ALL,KG_ORGANISMS=scerevisiae,ecoli,pputida` gave job
   4899 a yeast-only build (70 datasets). Use
   `KG_ORGANISMS=scerevisiae,ecoli,pputida INCLUDE_PRIVATE=1 sbatch --export=ALL,TORCHCELL_SRC=<wt>,BUILD_COMMIT=<sha> <script>`.
5. **No leftover containers.** `docker ps -a --filter name=tc-neo4j`; remove
   `tc-neo4j-gen` and `tc-neo4j-build` from a dead job (section 0).
6. **`--time` against the size.** The script header carries the limit; compare it with
   the last run of this size (section 5). 51 datasets generated in 3 h 05 min (job 3297);
   170 datasets needed more than 12 h of generation (job 4901, killed at 12 h at adapter
   65 of 170). The header is 48 h since PR #1102.
7. **Space.** `/db` needs `MIN_FREE_GB` (1500) and will take about 2x the CSV volume
   (CSVs plus the imported store; 4.0: 380 G of CSV, store about 300 G); `/bulk` takes the
   CSV archive, the backup work directory (the whole store) and the artifact; the old
   store stays on `/db` as `data.superseded.<ts>` for rollback.
8. **Node priority.** The build has it (owner, 2026-10-13): hold pending GPU and screen
   jobs of ours while the sweep and the build run, release them after.

Then, within a minute of submitting, read the job's `config` line and the preflight's
"N mapped datasets (organisms)" line. For 4.0 that is three organisms and 170 datasets.

## 2. Watching a running build

The `.out` is dominated by tqdm bars; grep it.

```bash
L=/scratch/projects/torchcell/database/slurm/output/<job>_live_rebuild_kg.out
grep -n -E '^-{10,}' $L                                   # stage boundaries
echo "adapters done: $(grep -c 'Writing edges for adapter' $L)/N"
grep -E 'Writing (nodes|edges) for adapter' $L | tail -n 2
grep -n -E 'Traceback|ERROR|FAIL|Killed|No space|refus|already exists' $L | tail
```

Stage 1 has two parts: a parallel per-dataset part (64 cores busy), then the **serial
adapter pass**, one `Writing nodes for adapter: X` then `Writing edges for adapter: X`
per dataset, each a chunked fold ("all node types (chunked) (N methods folded)"). The
big adapters dominate it; the chunk count of a pass is the record count over the
"single-pass chunk ... -> R records" line, and the completed bars inside the pass give
the rate:

```bash
n=$(grep -n 'Writing nodes for adapter: EnvChemgenHoepfner2014Adapter' $L | cut -d: -f1)
tail -n +$n $L | tr '\r' '\n' | grep -a -c -E '100%\|[█ ]+\| [0-9]+/[0-9]+ \['
```

**Watch with a stat-offset poll, not `tail -f`.** A `tail -n 0 -f <out> | grep` Monitor
delivered nothing through a 30 min window in which a matching line was appended
(2026-10-14; the same pipeline worked on a plain file; cause not isolated). The shape
that works, and that also reports the job's end:

```bash
F=<out>; off=$(stat -c %s $F)
while true; do sleep 20; new=$(stat -c %s $F)
  if [ "$new" -gt "$off" ]; then tail -c +$((off+1)) $F | grep -a -E '<pattern>' | cut -c1-200; off=$new; fi
  if ! squeue -h -j <job> | grep -q .; then echo "JOB LEFT THE QUEUE"; exit 0; fi
done
```

Re-arm it at every 30 min expiry. Do not poll with `pgrep` inside the container.

## 3. Expected durations (job 4901, 64 CPUs, 256 G container limit)

| phase | measured | note |
|---|---|---|
| preflight (freshness walk, fence digest) | about 1 min | 170 stores |
| per-dataset parallel part | about 5 h 25 min | ends when the first "Writing nodes for adapter" appears |
| serial adapter pass, adapters 1 to 64 | 5 h 45 min | Vaishnav 2022 1 h 43 min, Hale 36 min, the Costanzo pair 23 min each |
| Hoepfner 2014 alone | 4 h 14 min nodes + 2 h 14 min edges | 30.3 M experiments; section 6 |
| import + validate + swap + stamp | 51 datasets: 7 min import (job 3297) | 4.0: expected an hour or two; the resume job measures it |

Costanzo DMF (20.7 M experiments) took 9 min of nodes and 12 min of edges, so a record
count alone does not predict a pass; the resolved record size and the memory behaviour
do.

## 4. When the job dies

Read `sacct -j <job> -X -o State,ExitCode,Elapsed` and the `.out` tail first. Then:

- **Generation container still running** (`docker ps --filter name=tc-neo4j-gen`) and the
  CSVs worth keeping: capture its remaining log NOW, before `--rm` removes it:
  `nohup docker logs -f --since <UTC time the job died>Z tc-neo4j-gen > <OUTDIR>/<job>_generate_after_timeout.log 2>&1 &`
  (**the trailing `Z` is required**; without it docker reads local time and captures
  nothing). When the container exits, append that file to `<job>_generate.log` and
  resume at the import:
  `KG_ORGANISMS=<set> INCLUDE_PRIVATE=1 sbatch --export=ALL,TORCHCELL_SRC=<wt>,BUILD_COMMIT=<sha>,RESUME_JOB=<job>,RESUME_FROM=generation <script>`.
  The resume checks the generate log ends in `GENERATION_CALL=`, re-digests the dev
  tree against the old job's fence, and runs import, validate, swap and stamp.
- **Died after `IMPORT DONE in`:** `RESUME_JOB=<job>` (default `RESUME_FROM=import`)
  restarts the build container and continues from the post-import restart.
- **Died in generation and the CSVs are not worth keeping:** `docker rm -f tc-neo4j-gen`,
  fix the cause, resubmit. Known causes: a dev store rebuilt under the running job (the
  fence catches it after the fact; keep the tree quiet), a `GenomeRootNotFoundError`
  (bacterial genome caches must be mounted from the dev tree), an OOM-killed container
  (`docker inspect --format '{{.State.OOMKilled}}'`).
- **Died after the swap:** the old store is at `/db/database/data.superseded.<ts>`;
  rollback is stop the container, reverse the two `mv`, relaunch.

## 5. Resource tracking (what exists, where it is)

- **Container cgroup telemetry, every 5 s:** `biocypher-out/<ts>/telemetry/resource_samples.csv`
  (`torchcell/build_telemetry.py`, [[torchcell.build_telemetry]]): cores in use over the
  interval (every process), resident, peak and anonymous memory, the parent process's
  cores alone, the process count, and the adapter + method + phase current at that
  instant. `phase_timings.json` beside it at the end: one row per phase with seconds,
  CPU core-seconds and peak memory. The job's `.out` gets one summary line,
  `Generation wall N s, peak memory X GiB`.
- **Per-adapter wall times** are in the generate log (`Writing nodes` / `Writing edges`
  timestamps, UTC) and in W&B project `tcdb` (`<Adapter>_write_nodes_time`,
  `_write_edges_time`, `_n_nodes`, `_n_edges`, the `phase_timings` table).
- **Pool recycles** (`pool recycled at N chunks: cgroup memory at F of its limit`) are in
  the generate log; their rate is the memory-throttling signal.
- **The per-adapter table** that joins all of this:
  `experiments/database/scripts/kg_build_adapter_resources.py --telemetry <csv> --generate-log <log> --label <job>`
  writes `experiments/database/results/kg_build_adapter_resources_<job>.{json,md}`
  ([[experiments.database.scripts.kg_build_adapter_resources]]). Run it on every full
  build and put the table in the build's dated note section; it is the input for sizing
  the next build (`--time`, memory) and for choosing what to optimize.
- Slurm's own accounting is blind to the containers (section 0); do not size from `sacct`.
- Store-build sizing for the sweep: `store_rss_recent.tsv` from the array logs joined
  with `sacct MaxRSS` (those jobs run in the slurm cgroup, so sacct is right there).

## 6. The memory-throttled adapter (Hoepfner 2014, job 4901, measured)

From `resource_samples.csv` folded per adapter:

| adapter, phase | hours | cores mean | cores max | parent cores | mem mean | mem max | recycles |
|---|---|---|---|---|---|---|---|
| Hoepfner node | 4.24 | 19.4 | 64.6 | 0.75 | 213 G | 256 G | 139 |
| Hoepfner edge | 2.25 | 21.8 | 64.5 | 0.61 | 213 G | 256 G | 49 |
| Costanzo DMF node | 0.15 | 60.8 | 64.1 | 0.32 | 75 G | 122 G | 0 |
| Costanzo DMI edge | 0.23 | 60.8 | 64.2 | 0.23 | 158 G | 167 G | 0 |
| Vaishnav node | 1.02 | 1.1 | 25.9 | 0.98 | 110 G | 165 G | 0 |

Hoepfner's records resolve to 39 KB each (Costanzo's 22 KB), the 128 MiB chunk budget
cuts its chunks to 3,446 records (about 8,800 chunks per pass), and the container sat
at a mean 213 G against a 256 G limit with the pool recycled every two minutes (the
pool stops taking chunks above `pool_memory_fraction` 0.8 of the limit, 205 G). So the
pass ran at a third of Costanzo's cores: **memory-throttled, not compute-bound, and not
parent-bound** (the parent used under one core). Vaishnav is the other shape: one core,
no pool, a per-record cost in the adapter itself.

Levers, untested, in the order to try: (a) raise the container memory for the build
(the node has 513 G; 256 G was chosen to leave room for four GPU packs, which the build's
node priority already displaces); (b) a per-adapter `single_pass_chunk_budget_bytes`
and worker count for the 39 KB-record adapters so the in-flight set
(`process_workers + 2` chunks) fits under the recycle threshold; (c) profile what in
the Hoepfner record resolves to 39 KB. Measure each against the table above.

## 7. After the swap: release, ship, Radiant

In order, from the primary checkout (`R3` the pinned release, `R4` the new one from
`make ops`):

```bash
bash scripts/kg_release.sh backup          # detached in the serving container; wait for "exit 0" in /bulk/kg-releases/$R4/backup.log
bash scripts/kg_release.sh archive
bash scripts/kg_release.sh ship $R4        # rsync onto Taiga zhao5 through the Radiant VM's NFS mount, SHA256SUMS verified there
bash scripts/kg_release.sh keep $R4 "KG X.0, served on Radiant"
bash scripts/kg_release.sh deploy $R4 --from-archive-mount --dry-run && bash scripts/kg_release.sh deploy $R4 --from-archive-mount
bash scripts/kg_release.sh serve-swap --latest $R4 --pinned $R3 --dry-run && bash scripts/kg_release.sh serve-swap --latest $R4 --pinned $R3
bash scripts/ops.sh sync && make ops
```

The pinned release's restore onto zhao5 (`deploy $R3 --from-archive-mount`) has no
downtime and can run while the build is still generating (3.0 took 478 s on
2026-10-14). The serve-swap is the only Radiant downtime (one stop, one start).
Rollback: `serve-point --latest $R3 --pinned $R3`, or `serve-rollback <tc-neo4j-pre-ts>`.

Then: commit the release snapshot (`DB(kg): <release> snapshot`), regenerate the compat
page, run `kg_build_adapter_resources.py` on the run, write the dated section in the
script's note, and only then remove the build worktree (a landing deletes a worktree
under a running job: check `scontrol show job | grep WorkDir` first).

## 8. Lessons, dated

- 2026-10-09, jobs 3546/3547: bacterial genome caches must be mounted from the dev tree;
  a store rebuilt under a running generation mixes schema versions (the fence).
- 2026-10-13, 4898: `BUILD_COMMIT` must be the tag's commit. 4899: comma sets through
  the environment. 4900: a cancelled job's container must be removed first.
- 2026-10-13: the sweep's superseded copies filled `/scratch`; prune before sweeping.
- 2026-10-14, 4901: `--time` sized on 51 datasets killed a 170-dataset build; the
  container outlived the job; `tail -f` monitors missed lines; `RESUME_FROM=generation`
  and the `IMPORT DONE` marker (which the post-import resume had required since job 2032
  but nothing wrote) landed in PR #1102.
- 2026-10-14: Hoepfner is memory-throttled at the 256 G limit, not CPU-bound (section 6).
