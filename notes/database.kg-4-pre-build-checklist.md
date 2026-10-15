---
id: unqociauy85y0od7n1ywdig
title: Kg 4 Pre Build Checklist
desc: ''
updated: 1792084491351
created: 1792084491351
---

## 2026.10.15 - The KG 4.0 pre-build checklist

What the build driver executes, in order, with the source of every number and every
command. Nothing here has run: this is the plan the measurements of lanes A to Q support
([[scratch.2026.10.15.ontology-unification-decision]],
[[database.kg-4-pre-build-verification]]). The release note the build ships with is
[[database.kg-4-release-note]]; the version and pairing rules are in [[versioning]].

KG 4.0 carries **ten** schema-change records, not the four this checklist was first
written for: the owner folded P5 (the strain background) into this rebuild, lane Q added
the deletion `construction` field, lane R corrected the three audit flags and lane T
gave a construct span its parts. All ten are integrated on `ontology-unify` at
`21456d696`. Step 0 below is what holds the build until the branch lands.

Read [[database.constitution]] and the `/torchcell-db-build` skill before step 1.

### 0. Gates that must be green before anything is submitted

| # | step | how | source |
|---|---|---|---|
| 0.1 | **DONE** Lane Q's two fixes are on the branch (Ozaydin's 31 byte-identical records, Caudal's out-of-vocabulary `host_genome`) | `git log origin/ontology-unify` reads `958a4f4be` | integration 2, 2026-10-15 |
| 0.2 | **DONE** Lane U's P5 integrated: the shared strain-background base and the merged `BackgroundAllele` in `schema.py`, `BacterialGeneNamespace` and `BACTERIAL_LOCUS_TAG_PATTERNS` gone, `StrainReferenceGenome.background` collapsed onto the base and still required | `ls database/schema_changes/2026-10-15-*` reads ten; head `21456d696` | coordinator, 2026-10-15: P5 is in this rebuild, not the next |
| 0.3 | `ontology-unify` landed on `main`, with all ten schema-change records and this release prep | merge queue | lane P verdict: the branch is safe to sweep and build |
| 0.4 | **DONE** `tests/torchcell/datamodels/test_schema_change_records.py` AND `scripts/check_schema_change_records.py` pass with NO `TORCHCELL_SCHEMA_RECORD_ACK`, with the ten records owning 146 of 146 moved symbols, one owner each | pytest plus the tripwire | integration 2: 4 passed, tripwire "every schema change vs origin/main is recorded" |
| 0.5 | **DONE on the branch head**, to be re-run on the landed head: ruff, format, strict mypy, the FULL `tests/torchcell` tree | pre-commit plus pytest | integration 2 on `21456d696`: 15,913 passed, 1,164 skipped, 10 xfailed, 1 failed in 15 min 38 s; the one failure is `test_s288c_synthetic.py::test_concurrent_processes_keep_the_original_pair_over_an_earlier_one`, a four-real-subprocess race over a hot SQLite journal that passes 5 of 5 in isolation and 262 of 262 with its whole file, did not fail in the earlier full run of this same tree, and lives in a file no lane on this branch touches; mypy clean over 549 changed files |
| 0.6 | **DONE** The four supported queries parse and name no retired label, re-run after P5, R and T | `PYTHONPATH=. python experiments/database/scripts/kg4_release_prep_checks.py queries` | integration 2: all four clean, result byte-identical to lane M's |
| 0.7 | **DONE** The release note's `pending` P5 rows carry the record's tool-computed numbers, and the ten records' scope is the one measured on the integrated head | read `database/schema_changes/2026-10-15-*.json` | [[database.kg-4-release-note]], the ten-record table |
| 0.8 | Lanes V, S and W closed out or deferred in writing (V: the remaining content-id collisions; S: the generation profile; W: the environment-identity verdict) | check-in log | integration 2, 2026-10-15: all three still out when the ten records were consolidated |

### 1. Cut the pairing release, and build only from its tag

The build gate in `gilahyper_live_rebuild-slurm_docker.slurm` refuses to start unless the
build commit carries a `vX.Y.Z` tag and the fingerprint checkout sits on it, so this step
is not optional.

1. Push one commit to `main` whose subject starts `REL(kg):` (minor, so `1.7.5` becomes
   `1.8.0`; [[versioning]] holds the tag map and the reasoning).
2. Wait for `semantic-release.yaml` to push its bump commit and the `v1.8.0` tag, then
   `git fetch --tags`. The tag sits on the bot's bump commit, not on the `REL` commit.
3. Confirm with `git describe --tags --exact-match <bump commit>` and
   `grep version pyproject.toml`.
4. Check out that tag into the build worktree (`TORCHCELL_SRC`), clean tree. Never build
   from the primary checkout, which other sessions pull forward.

No landing of any kind goes to `main` between this tag and the end of step 8: every
landing re-stales the dev stores, which is what makes the sweep the last step before the
build (decision note, stage ordering).

### 2. Hold the node

The sweep is memory bound and the owner's ruling is that a ready KG rebuild has node
priority (owner, 2026-10-13: "i just want this to have priority when it is ready to
build so we can pause other gpus runs etc."). Hold every pending job of the user that is
not part of the sweep, leave running jobs to finish, and tell the other sessions to
submit nothing above 24 G during the window.

```bash
HELD=/scratch/projects/torchcell/database/slurm/sweep_held_$(date +%s).txt; : > "$HELD"
for j in $(squeue -h -u "$USER" -t PD -o "%i %j" | grep -v -E ' kg4-sweep' | awk '{print $1}'); do
  scontrol hold "$j" && echo "$j" >> "$HELD"
done
```

**The two helper scripts the 2026-10-13 sweep used are NOT in the repository.**
`sweep_conductor.sh` and `release_held.sh` live only in a session scratchpad
(`/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/ebf6929f-c5a0-46a3-a925-7ca89fee3498/scratchpad/`),
so the driver either copies them out or runs the loop above and releases by hand in step
12. Holding GPU packs idles a GPU for the hold's duration, so ask the owner first
(memory: `gilahyper-fifo-hold-release-to-reorder`).

### 3. Prune the superseded store copies on `/scratch`

Measured 2026-10-15: `/scratch` is **78 percent full with 1.6 T free**, and **405
`*superseded*` directories hold 669 G** under
`/scratch/projects/torchcell-scratch/data/torchcell/`. Each is a previous `processed/` or
`preprocess/` that an earlier `--retire-existing` renamed aside, and the sweep in step 4
creates one more per store.

Move them, never delete them, and `trash` is not installed on this host (memory:
`trash-not-installed-on-gilahyper`):

```bash
DEPRECATED_DIR=/scratch/projects/torchcell-deprecated \
  bash scripts/deprecate.sh <path> "superseded store copy, pruned before the KG 4.0 sweep"
```

The graveyard must be the sibling `/scratch/projects/torchcell-deprecated`, never a path
inside `$DATA_ROOT`; `scripts/deprecate.sh` refuses the latter.

### 4. Sweep all 170 dev stores, memory-tiered, as the last step before the build

```bash
CLASSES=/scratch/projects/torchcell/database/slurm/stale_classes_kg4.txt
PYTHONPATH=$TORCHCELL_SRC python -m torchcell.database.build_dataset_lmdb \
    --list-stale --include-private > "$CLASSES"
sbatch --array=1-$(wc -l < "$CLASSES")%4 --export=ALL,CLASSES_FILE="$CLASSES" \
    database/slurm/scripts/gilahyper_build_dataset_lmdbs_array.slurm
```

Every store reads stale against the landed branch, which is correct: the records
move every closure, and lane A's `MANIFEST_SCHEMA_VERSION = 2` adds
`BuildManifest.perturbation_leaves`, so a version 1 manifest is stale on its own.

The tier sizes, from the 2026-10-13 arrays (4685, 4686, 4687, 4866, 4897; 194 task logs)
and the decision note's overestimated sizing table. The committed array script is a
single tier (16 CPU, 64 G, 12 h); the tiers below are what the driver should submit as
three arrays off three class lists:

| tier | stores | `--cpus` | `--mem` | `--time` | concurrency | measured basis |
|---|---:|---:|---:|---:|---|---|
| `s` | 122 | 4 | 32 G | 8 h | `%6` | longest 52 min at 20 G |
| `l` | 41 plus 27 reruns | 8 | 96 G | 24 h | `%3` | longest 4 h 32 min at 64 G (the Costanzo pair and Hale) |
| `xl` | 1 | 8 | 160 G | 24 h | with `l` | 9 min at 120 G |

Lane P's nine stores confirm the tiers hold with at least **3x memory margin** (largest
store 10.0 GB peak resident against the 32 G small tier). Calendar floor **7 to 11 h**,
set by the longest `l` store plus the `l` tier at `%3`. Tasks are independent, so recovery
is re-running the named task alone.

One adapter to watch that lane P's sample did not cover: **Mutalik 2020** (286,344
records, 37 G store-build resident on 2026-10-13) stays in the standard tier, so check it
against 64 G on the first array.

### 5. Prove zero stale before the build reads anything

```bash
PYTHONPATH=$TORCHCELL_SRC python -m torchcell.database.build_dataset_lmdb \
    --list-stale --include-private --organisms scerevisiae,ecoli,pputida
```

It must print **nothing**. Then confirm every manifest is version 2 and carries
`perturbation_leaves` (lane A), and record the count drift as the data-is-the-data check:

```bash
PYTHONPATH=$TORCHCELL_SRC python scripts/served_count_drift.py
```

Exit 1 on any dataset that DECREASED. Expected on 2026-10-15: 19 of 51 served datasets
increased, 32 unchanged, 0 decreased ([[database.kg-4-release-note]] carries the table).
The organism set here must be the same set passed to the build as `KG_ORGANISMS`, because
the preflight walks only that set's map.

### 6. Generation

```bash
KG_ORGANISMS=scerevisiae,ecoli,pputida sbatch --export=ALL \
    database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm
```

`KG_ORGANISMS` goes in the environment, not in `--export`, because `sbatch` splits that
list on commas (memory: `live-rebuild-submit-pitfalls`). `INCLUDE_PRIVATE` defaults to 1,
which is right for the in-house store. Give the job at least 30 h; the header reserves
48 h after job 4901 was killed at the old 12 h limit.

**The one measurement that decides the shape of this step: memory, not compute.** The
Hoepfner 2014 adapter ran **4,558 s (1 h 16 min) at 320 G with 4 pool recycles** in job
4919_64, against **6 h 29 min at the 256 G limit with 139 recycles** in job 4901: a
**5.1x speedup from memory alone**, with identical row counts. Neither variant beat it
(chunk 256 MiB, job 4920_64: 297.9 GiB and 59 recycles, no gain; 32 workers, job 4921_64:
4,795 s). So the baseline configuration and the 320 G reservation are what the build uses,
and the generation tiers in `torchcell/knowledge_graphs/conf/generation_tiers.yaml` are
set from it: `hoepfner` 64 CPU / 320 G / 36 h, `large` 32 / 160 G / 16 h for the 13 named
adapters above 5 M records, `standard` 16 / 64 G / 8 h for the rest.

`GENERATION_MODE` defaults to `serial` (one container, one serial loop, no per-adapter
checkpoint, a failure at adapter 124 restarting from adapter 1). Array mode is what gives
the owner's rewind-to-10 recovery:

```bash
GENERATION_MODE=array KG_ORGANISMS=scerevisiae,ecoli,pputida sbatch --export=ALL \
    database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm
```

It submits one task per adapter per tier, prints `ARRAY_OUT` and `ARRAY_JOBS`, resubmits
itself with `--dependency=afterany` and exits so the node is not held by a waiting job.
Startup is 15.8 s mean on the prebuilt image against 255.9 s on the install path, which
saves about 11.3 h of CPU time and 2.8 h of wall clock across 170 tasks; a same-commit
rebuild of the image is free on the layer cache, so a resume pays no image time (lane C2).

**Recovery, the command to keep at hand:**

```bash
PYTHONPATH=. python -m torchcell.knowledge_graphs.generation_array missing
sbatch --export=ALL,RESUME_JOB=<array-mode job>,RESUME_FROM=array \
    database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm
```

`missing` prints the slots without a `DONE` marker and the per-tier re-run lines; the
resume re-runs exactly those and chains a new follow-up. A follow-up never re-runs a task
itself, so a recurring failure cannot loop. The merge step verifies every directory has
its `DONE`, re-digests the dev-tree fence, joins the part files per label and proves every
duplicate content id byte-identical. The duplicate check keeps **full scope**: an
Experiment id is the record alone with no dataset component, so the same record released
by two sources collides across directories; only `Dataset` and `ExperimentMemberOf` cannot
recur, which is 2.7 percent of bytes. Worst case measured (two full copies, every row
duplicated, contended disk): 1 h 55 min at 43.9 GB peak resident, 0 conflicts.

### 7. The CSV endpoint gate, before `neo4j-admin import`

The import branch runs it in every mode
(`torchcell.knowledge_graphs.csv_endpoint_gate`, walking `adapters/*/` as one output).
It must be read, not merely run.

- It checks that every edge endpoint exists, that it resolves to a node of a **label the
  graph schema declares** as that edge's `source` or `target`, and that no id appears
  under two labels. The label half is lane N's addition and is what caught the defect
  lane A's existence check could not see.
- Cost, measured on the 383 G job 4901 partial: **1,473 s and 40.1 GB peak resident**
  with dataset resolution on, against 1,281 s and 30.6 GB without it.
- On lane P's nine generated directories: **2,683,077 node rows in 20 labels, 7,972,416
  edge rows in 13 labels, 0 dangling, 0 mislabeled, 0 id collisions**, in 19 s.
- The number to insist on: **0 dangling `PerturbationMemberOf` rows**. The 4,233 dangling
  rows on the partial were attributed and are a partial-run artifact (the sink buffers
  1 MiB per part file and flushes in `finish()`, which the Shiver 2016 failure never
  reached), not a lane gap; a complete build has no excuse for a nonzero count.
- `Mentions` is the one expected undeclared edge label.

### 8. Import, validate, swap

The slurm script does these in order, and each is a gate (script header):

1. **import** into a fresh `$NEXT_ROOT/data` on the same `/db` filesystem, in a build
   container with no host ports.
2. **validate** the new store before anything served is touched: every mapped dataset has
   a `Dataset` node and its live `ExperimentMemberOf` count equals its CSV rows.
3. **swap**: stop both containers, `mv $SERVE_ROOT/data` to `data.superseded.<ts>`,
   `mv $NEXT_ROOT/data` into place, relaunch `tc-neo4j-readonly`. Downtime is two renames
   plus one Neo4j start.
4. **record**: `kg_manifest bootstrap` (with `--organisms` and, in array mode,
   `--merge-report`), `kg_manifest artifact-refs`, `releases stamp`, the `latest` and
   `pinned` aliases, CSVs archived to `/bulk`.

Disk: the preflight refuses below `MIN_FREE_GB` (1500). Measured 2026-10-15: `/db` has
**2.4 T free at 68 percent**, against about 380 G of CSV and a store on the order of
200 G, so the rollback copy is affordable.

### 9. Commit the release, then re-validate what reads it

1. Commit `database/releases/<release>.json` and `.closures.json` with a `DB(kg):`
   subject (never `RELEASE(kg)`, which is not an allowed tag and bumped nothing in
   October).
2. `python scripts/kg_compat_page.py` and commit the page. `--check` runs in CI
   (`docs.yaml`, job `query-drift`) and fails while the page is stale.
3. Re-validate the four supported queries, which is what records their new
   `dataset_composite`:

   ```bash
   for q in amino_acid_betaxanthin essentiality_smf expression_proteome_morphology solid_growth_025; do
     PYTHONPATH=. python -m torchcell.knowledge_graphs.supported_queries validate "$q" --release <new release>
   done
   ```

   `validate` resolves a `composite_changed` drift and refuses every other kind, so a
   `contract_changed` that survives the rebuild is a real problem: read it, do not
   validate around it.
4. Check the pairing end to end: `python -m torchcell.knowledge_graphs.releases --repo .
   compat --version latest` and the pairs table of
   `docs/source/database/compatibility.md` must name `v1.8.0`.

### 10. Ship the artifact

```bash
bash scripts/kg_release.sh backup     # online backup into $ARCHIVE_ROOT/<release>/
bash scripts/kg_release.sh archive    # kg_manifest.json + release.json + SHA256SUMS
bash scripts/kg_release.sh ship       # rsync to Taiga, verify SHA256SUMS there
bash scripts/kg_release.sh keep <release> "KG 4.0, the ontology unification"
```

About 1 h in total (decision note). `backup` writes the whole store before compressing,
so it must land on the big `/bulk` RAID and never on `/db`. On GilaHyper these run under
slurm, never as a shell process (memory: `gilahyper-db-work-under-slurm`):

```bash
sbatch -J kg-ship -c 4 --mem=16G --time=4:00:00 --wrap "bash scripts/kg_release.sh ship"
```

Prove the artifact restores before it is trusted as the only copy:
`kg_release.sh restore-test <release>`.

### 11. Swap Radiant onto it

```bash
bash scripts/kg_release.sh deploy <release> --from-archive-mount --pin
bash scripts/kg_release.sh serve-swap --latest <release> --pinned <release>
```

The restore reads and writes only on the Taiga mount and runs in a detached container, so
a dropped ssh connection does not stop it. The new database is created, checked against
`release.json` (release id, version, every dataset's experiment count, the node total),
given the committed snapshot's package pairing, set **READ ONLY**, and `latest`,
`torchcell` and the default database move to it. Auth is off on that host, so a new
database needs READ ONLY set by hand (memory: `radiant-served-db-contents-and-io-fault`).

### 12. Rollback, and release the held jobs

| failure | undo | source |
|---|---|---|
| The new store is wrong after the GilaHyper swap | the reverse pair of `mv`: `data.superseded.<ts>` back to `$SERVE_ROOT/data`. The old store stays on disk until someone removes it | live rebuild header, `swap` |
| Radiant serves the wrong release but already holds the right one | `kg_release.sh serve-point --latest <release>`, the alias-level rollback | `kg_release.sh` usage |
| The Radiant swap itself must be undone | `kg_release.sh serve-rollback <kept container>`: the pre-swap container was kept as `<name>-pre-<timestamp>` with its store | `kg_release.sh` usage |
| A generation task failed | step 6's `RESUME_FROM=array` recovery; the other directories are untouched | live rebuild header |
| One dev store failed to sweep | re-run that array task alone | sweep recovery, 2026-10-13 (`4866_11` failed, re-run as `4897`) |

Release what step 2 held, once the large sweep tier and the generation arrays have left
the queue:

```bash
for j in $(cat "$HELD"); do scontrol release "$j"; done
```

Then post the artifact pointer line for the run (memory: `ops-artifact-pointer-line`) and
check `squeue` rather than `pgrep` for anything still ours (memory:
`gilahyper-poll-squeue-never-pgrep`).

### What this checklist does not cover

- **The KG 3.0 defect disclosure cannot be written into the 4.0 store's own metadata.**
  `releases snapshot --note` is accepted only with `--torchcell-version`, which is refused
  when the manifest already records a version, and a build from a tagged commit always
  records one; `kg_manifest bootstrap` takes no note argument. The disclosure therefore
  lives in [[database.kg-4-release-note]], in the `v1.8.0` GitHub release body and on the
  compatibility page. Putting it in `events[].note` needs a code change, which is not in
  this release.
- **The `perturbation of genome` edge is declared and off by default.** Measuring it is a
  decision for the first build, not a step here.
- **Ozaydin 2013's 31 byte-identical records** are a loader gap the build would otherwise
  bake in, and lane Q owns it; step 0.1 is the gate.
- **P5 is now IN this rebuild** (lane U), so it is a step-0 gate rather than future work.
  What the plan grouped with P5 and the coordinator did not name for lane U stays open and
  needs lane U's own answer: the `CalMorphPhenotype` and `BacterialMorphologyPhenotype`
  rename, whose names are backwards, and the `BW25113_BACKGROUND_GENOTYPE` and `_LESIONS`
  constants, one strain's verbatim genotype string living in `schema.py` and read by 84
  loader closures. A rename of a served phenotype class after 4.0 is another full rebuild,
  so if it is to happen it should happen here; that is the owner's call, not the driver's.
