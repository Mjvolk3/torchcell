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

KG 4.0 carries **twelve** schema-change records, not the four this checklist was first
written for: the owner folded P5 (the strain background) into this rebuild, lane Q added
the deletion `construction` field, lane R corrected the three audit flags, lane T gave a
construct span its parts, lane V put the curated reference on an essentiality call and
lane X typed a medium's composition. All twelve are integrated on `ontology-unify` at
`77facd140` (integration 3, which also folded lane W, a verification fix carrying no
schema symbol and so no record). Step 0 below is what holds the build until the branch
lands, and the GO/NO-GO page at the end of this note is the one-page version of it.

Read [[database.constitution]] and the `/torchcell-db-build` skill before step 1.

### 0. Gates that must be green before anything is submitted

| # | step | how | source |
|---|---|---|---|
| 0.1 | **DONE** Lane Q's two fixes are on the branch (Ozaydin's 31 byte-identical records, Caudal's out-of-vocabulary `host_genome`) | `git log origin/main..ontology-unify` reads "the deletion strain's collection position is on the record" | integration 2, re-based at integration 3 |
| 0.2 | **DONE** Lane U's P5 integrated: the shared strain-background base and the merged `BackgroundAllele` in `schema.py`, `BacterialGeneNamespace` and `BACTERIAL_LOCUS_TAG_PATTERNS` gone, `StrainReferenceGenome.background` collapsed onto the base and still required | `ls database/schema_changes/2026-10-15-*` reads twelve; head `77facd140` | coordinator, 2026-10-15: P5 is in this rebuild, not the next |
| 0.3 | `ontology-unify` landed on `main`, with all twelve schema-change records and this release prep | merge queue | lane P verdict: the branch is safe to sweep and build |
| 0.4 | **DONE** `tests/torchcell/datamodels/test_schema_change_records.py` AND `scripts/check_schema_change_records.py` pass with NO `TORCHCELL_SCHEMA_RECORD_ACK`, with the twelve records owning 149 of 149 moved symbols, one owner each and none exempt | pytest plus the tripwire | integration 3: 4 passed, tripwire "every schema change vs origin/main is recorded" |
| 0.5 | **DONE on the branch head**, to be re-run on the landed head: ruff, format, strict mypy, the FULL `tests/torchcell` tree | pre-commit plus pytest | integration 3 on `77facd140`, numbers in the GO/NO-GO page below; `ruff check torchcell tests` and `ruff format --check` clean, strict mypy clean over the 555 changed gate-scope files |
| 0.6 | **DONE** The four supported queries parse and name no retired label, re-run on the final head after V, W and X | `PYTHONPATH=$PWD python experiments/database/scripts/kg4_release_prep_checks.py queries --worktree` | integration 3: all four parse, 0 missing labels, 0 missing relationship types, 0 missing properties, and the only drift is the same ten `contract_changed` details lane M measured from `main`. The output file is regenerable and deliberately untracked, so it is re-run rather than diffed |
| 0.7 | **DONE** The release note's `pending` P5 rows carry the record's tool-computed numbers, and the twelve records' scope is the one measured on the integrated head | read `database/schema_changes/2026-10-15-*.json` | [[database.kg-4-release-note]], the twelve-record table |
| 0.8 | **DONE** Lanes V, S, W and X all closed out in writing (V: the remaining content-id collisions and the manifest guard; S: the bacterial generation profile, landed on `main`; W: the environment-identity verdict; X: the media composition fidelity) | check-in log | integration 3, 2026-10-15: all four folded in, S on `main` underneath the branch |
| 0.9 | **ONE OWNER DECISION OPEN, and it is a label rather than a change**: lane V's essentiality record carries `ESSENTIALITY_EVIDENCE_FIELDS[rename]`, which `SIGN_OFF_KINDS` makes a sign-off kind, although the dict's 6 keys are identical before and after and only one key's tuple gained a name | measured in the GO/NO-GO page below | integration 3 verified it is a `binding_members` rendering artifact |

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

Measured 2026-10-15, re-measured at integration 3: `/scratch` is **78 percent full with
1.6 T free**, and **407 `*superseded*` directories hold 669 G** under
`/scratch/projects/torchcell-scratch/`. Each is a previous `processed/` or `preprocess/`
that an earlier `--retire-existing` renamed aside, and the sweep in step 4 creates one
more per store.

`/db` is **68 percent full with 2.4 T free**, and 414 G of it is the Hoepfner generation
study's three output directories
(`/db/database/biocypher-out/hoepfner-study-2026-10-15_04-20-20`, `-04-21-22-chunk256`
and `-04-21-22-w32`, 138 G each, measured at integration 3). The study is CLOSED and its
numbers are in [[experiments.database.hoepfner-generation-study]], so the directories are
reproducible output rather than evidence, and reclaiming them is the owner's call. The
job 4901 partial is a further 383 G (`biocypher-out/2026-10-14_03-53-33`) and an earlier
partial 170 G; the 4.0 build writes about 380 G of CSV, so `/db` does not gate the build
either way.

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
move every closure, and the manifest schema is now at **version 3**, lane A's
`perturbation_leaves` plus lane V's `content_id_collisions`, so a version 1 or 2 manifest
is stale on its own.

The tier sizes, from the 2026-10-13 arrays (4685, 4686, 4687, 4866, 4897; 194 task logs)
and the decision note's overestimated sizing table. The committed array script is a
single tier (16 CPU, 64 G, 12 h); the tiers below are what the driver should submit as
three arrays off three class lists:

| tier | stores | `--cpus` | `--mem` | `--time` | concurrency | measured basis |
|---|---:|---:|---:|---:|---|---|
| `s` | 122 | 4 | 32 G | 8 h | `%6` | longest 52 min at 20 G |
| `l` | 41 plus 27 reruns | 8 | 96 G | 24 h | `%2` | longest 4 h 32 min at 64 G (the Costanzo pair and Hale) |
| `xl` | 1 | 8 | 160 G | 24 h | before `s` | 9 min at 120 G |

**The 384 G cap (owner, 2026-10-15: "apply the 384 cap").** The build reserves at most 384 G at any moment so the metabolic side keeps 129 G, its GPUs and its slurm jobs at 32 G or less beside it. In the sweep that is `l` at `%2` (192 G) beside `s` at `%6` (192 G), with the single `xl` task (160 G) run before `s` starts. The `l` tier is 22.3 h of measured task time over 40 stores (arrays 4686 and 4866), so `%2` puts it at about 11.2 h against 7.4 h at `%3`; the sweep's calendar is **12 to 14 h** under the cap instead of 9 to 13 h. The generation array is capped the same way in `torchcell/knowledge_graphs/conf/generation_tiers.yaml`: `large` 2 x 160 G beside `standard` 1 x 64 G, and the Hoepfner adapter in `large` once lane Y lands.

Lane P's nine stores confirm the tiers hold with at least **3x memory margin** (largest
store 10.0 GB peak resident against the 32 G small tier). Calendar floor **7 to 11 h**,
set by the longest `l` store plus the `l` tier at `%3` (`%2` under the 384 G cap, about +4 h). Tasks are independent, so recovery
is re-running the named task alone.

**Add lane V's collision guard to this step, not to generation.** The manifest now hashes
every record's resolved experiment dump in the same single pass as the leaf walk, which
lane V measured at about **+48 min on Hoepfner's `write_build_manifest` and +30 min per
Costanzo store, roughly 1 h 48 min across the three largest**. All three are `l`-tier
stores running at `%3`, so the guard lands on the tier that already sets the calendar: the
7 to 11 h floor becomes roughly **9 to 13 h**. The digest form is bounded rather than
convenient (whole 32-byte digests in one buffer plus the keys as a blob with offsets,
grouped by a stable numpy argsort, about 1.5 GB for Hoepfner's 30.3 M records against
roughly 6.5 GB for the dict-and-list form), so the tier memory does not move.

**The guard REFUSES at the rebuild preflight, so a surprise here stops the build rather
than being reported.** `build_dataset_lmdb.mapped_store_collisions` runs in the live
rebuild's preflight right beside the lane walk, and the only written escape is
`EXPECTED_CONTENT_ID_COLLISIONS`, which names `env_chemgen_mota2024` at exactly 2 and
nothing else. If the sweep surfaces a collision in a store nobody has scanned, the
preflight names it and the build does not start; that is the intended behavior and not a
failure of the sweep.

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
The conductor resolves the image once and passes it to every task as `GEN_IMAGE`, which is
lane S's hook on `main` at `0a238c967`.

**The array's calendar, now measured rather than estimated (lane S, job 4974, 141 tasks
completed, 47 adapters profiled).** Every one of the 170 tasks is charged a number measured
somewhere, and only the arithmetic is projected:

| tier | tasks | concurrency | measured tasks | task seconds | projected wall | basis |
|---|---:|---:|---:|---:|---:|---|
| `standard` | 157 | 4 | 47 | 14,368 | 1.00 h | 47 this run, 86 the job 4901 fold, 24 bounded above by one 5 s sampling interval |
| `large` | 12 | 2 | 1 | 18,759 | 2.61 h | 1 at this tier in another run, 11 from the 4901 fold |
| `hoepfner` | 1 | 1 | 1 | 4,569 | 1.27 h | job 4919_64's 320 G baseline |

The three arrays are submitted together and run concurrently, so **the calendar is the
largest row, about 2.6 h, not the 4.88 h serial sum**, and the bacterial tail that lane S
existed to measure is not the cost: 157 of 170 tasks project to 1.00 h.

**Critical path adapter: `PooledPromoterExpressionVaishnav2022Adapter` at 6,175 s
(1.72 h).** It is the longest single task of the array, longer than Hoepfner's 4,558 s, and
the `large` tier cannot go below it however the other 11 are packed. Job 4901 measured it
at 1.1 cores mean of 64 with 0 recycles at a mean 110 G, so it is neither core-bound nor
memory-throttled but **parent-bound**, and the `large` tier's 32 CPUs will not shorten it.

Two caveats lane S labeled, both of which can only make the array slower:

- **97 of the 170 charges come from the job 4901 serial container**, which gave every
  adapter 64 CPUs and a cumulative heap. `DmfCostanzo2016Adapter`, `DmiCostanzo2016Adapter`
  and `Hale2024Adapter` ran at 60.8, 60.0 and 51.4 cores mean there and get 32 in the
  `large` tier, so the understatement is bounded at about 2x on three tasks of the tier
  that sets the calendar: a `large` tier at 2x its charge is **5.2 h rather than 2.6 h**.
- **Pool recycling is the one thing that could invalidate it.** Every task of lane S's run
  recycled 0 times, and the Hoepfner study measured a 5.1x penalty once a container crosses
  0.8 of its memory limit, so an adapter approaching its tier is multiples slower rather
  than a few percent. That is the whole argument for the one tier change below.

**The one tier change lane S proposed is NOT in the committed table, and that is a
one-line owner decision.** `RbTnseqBorchert2024Adapter` (1,466,920 records, the largest
count of the bacterial tail) peaked at **33.93 GiB of its 64 G standard container, 53.0
percent**, with 0 recycles at 14.4 of 16 cores in 522 s; the 0.8 recycling threshold of a
64 G container is 51.2 G, which that run came within 17 GiB of on a quiet filesystem, and
the branch is 6.6 percent heavier than `main` on the one adapter both sides measured
(Nichols, 219 s and 12.5 GiB on the branch against 205.7 s and 11.73 GiB on `main`), which
puts it at about 56.5 percent. `torchcell/knowledge_graphs/conf/generation_tiers.yaml`
still lists 12 `large` adapters and Borchert is not among them, so the build as configured
runs it at `standard`. Moving it is one line and costs the array about 4 min (522 s at
`%2`); leaving it accepts a 1.9x margin on the adapter nearest its ceiling.

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
- **Lane W carries no record and needs none.** Its `_stored_identity_projection` change is
  in `torchcell/verification/`, moves no schema symbol and rebuilds nothing; what it moves
  is the L1 duplicate allowance, which is a verification property of a store and not a
  served value.
- **P5 is now IN this rebuild** (lane U), so it is a step-0 gate rather than future work.
  The two items the plan grouped with P5 and did not name for lane U are **done, by lane R**,
  and they are in this rebuild rather than costing a second one: `BacterialMorphologyPhenotype`
  became `MorphologyPhenotype` while `CalMorphPhenotype` KEPT its name (the program name
  already states what it is specific to, and `YeastMorphologyPhenotype` would be false), and
  the `BW25113_BACKGROUND_GENOTYPE` and `_LESIONS` constants moved to the genomes tier beside
  the assembly set whose GenBank note they come from. Lane R also corrected the audit's radius
  on the second one: the constants sit in 1 and 0 loader closures on THIS branch, not the 84
  measured on `main` before the unification, so the defect was the location and not the reach.

## 2026.10.15 - GO/NO-GO for the KG 4.0 sweep and build, one page

Written at the end of integration 3, which folded the last three lanes (V, W, X) onto
`ontology-unify`. The sweep is the first irreversible step of this program, so this page
is the whole decision in one place: what is ready, what it costs, what is still unknown.
Every number is measured and says where; nothing here is an estimate dressed as a result.

### The verdict

**GO on the gates. The branch is ready and nothing in it is waiting on more work.** What
is not DONE is the landing, and it has to come first rather than alongside: the branch
lands on `main` behind the `REL(kg):` bump commit and its tag (step 1 and step 0.3 above),
because every landing after the sweep re-stales the dev stores and the build gate refuses
an untagged build commit. Three smaller decisions are listed below and none of them blocks
a start.

### The branch

| | |
|---|---|
| branch | `ontology-unify`, draft PR #1112 |
| gate head | `77facd140` (the head this page's gate numbers were measured on) |
| rebased onto | `main` at `b123a9d56` (lanes A, B, C, C2, N, P and S all underneath) |
| commits ahead of `main` | 37 (36 at the gate head plus this note's own commit), linear, no merge commits, no duplicated commits |
| schema-change records | **12**, all dated 2026-10-15, ten signed and two needing no sign-off |
| lane PRs stacked in | #1113 #1115 #1116 #1117 #1120 #1121 #1122 #1123 #1124 #1125 #1126 #1127 |
| schema symbols moved | **149**, each named by exactly one record, none exempt |
| version pairing | `1.8.0`, prepared and NOT cut; the tag must point at the semantic-release bump commit the `REL` push creates |

### Gate state

| gate | state | measured |
|---|---|---|
| `ruff check` + `ruff format --check` at gate scope (`torchcell`, `tests`, `experiments/016+`) | **green** | clean, 1,313 files formatted; 21 findings remain in frozen `experiments/006`, `009`, `010`, where `main` has 44 |
| strict mypy over the branch's 555 changed gate-scope files | **green** | `Success: no issues found` |
| full `tests/torchcell` | **green, 0 failed** | **15,943 passed, 1,165 skipped, 10 xfailed, 0 failed** in 15 min 43 s. Integration 3 ran the tree TWICE: on `f157f6e5a` it was 8 failed / 15,935 passed, every failure one cause (two stale fake manifests against lane V's `content_id_collisions`), fixed in `77facd140`, and the re-run is clean. The known flake `test_s288c_synthetic.py::test_concurrent_processes_keep_the_original_pair_over_an_earlier_one` did NOT reproduce in either run |
| `scripts/check_schema_change_records.py` (the tripwire) | **green, NO ack** | "every schema change vs origin/main is recorded"; liveness verified by perturbing a record and seeing 3 problems |
| `tests/torchcell/datamodels/test_schema_change_records.py` | **green, NO ack** | 4 passed |
| `schema_change check` | **green** | 31 records carry every change since `eb07bd213ca1` |
| loader-to-lane exhaustiveness | **green** | lane P: 9 of 9 built stores, 170 of 170 mapped |
| CSV endpoint gate on real generated CSVs | **green** | lane P, nine directories: 2,683,077 node rows in 20 labels, 7,972,416 edge rows in 13 labels, **0 dangling, 0 mislabeled, 0 id collisions**, 19 s |
| `host_genome` equals the genome node's `assembly_set` | **green on real data** | lane P, every perturbation row of every adapter; this is the join the whole unification was for |
| supported-query revalidation | **green** | re-run on this head: all four parse, 0 missing labels, 0 missing relationship types, 0 missing properties; the only drift is the same ten `contract_changed` details lane M measured from `main`, so it predates this work |
| `TORCHCELL_SCHEMA_ACK=1`, `TORCHCELL_QUERY_DRIFT_ACK=1` | **expected** | the rebuild gate and the KG 3.0 query drift; both are the point of this branch |
| `TORCHCELL_SCHEMA_RECORD_ACK` | **not set anywhere** | every lane needed it; the integrated branch does not |

### What the compute costs, measured

| step | calendar | basis |
|---|---|---|
| prune `/scratch` | minutes | 407 directories, `scripts/deprecate.sh`, a rename on the same filesystem |
| sweep, 170 dev stores | **9 to 13 h**, 12 to 14 h under the 384 G cap (`l` at `%2`) | the 2026-10-13 arrays' 7-to-11 h floor (longest `l` store 4 h 32 min, `l` tier at `%3`) plus lane V's collision guard at about +48 min on Hoepfner and +30 min per Costanzo store, roughly 1 h 48 min on the tier that sets the floor |
| zero-stale read | minutes | `--list-stale` must print nothing |
| generation, per-adapter array | **about 2.6 h**, about 4 h under the 384 G cap (`standard` at `%1`), and 5.2 h if the three core-bound `large` adapters run at 2x their serial-container charge | lane S, job 4974, 141 tasks completed: `standard` 157 tasks at `%4` project 1.00 h, `large` 12 at `%2` project 2.61 h, `hoepfner` 1 projects 1.27 h; the three arrays run concurrently so the calendar is the largest row, not the 4.88 h sum |
| critical path inside it | **1.72 h, `PooledPromoterExpressionVaishnav2022Adapter`** | 6,175 s, longer than Hoepfner's 4,558 s; 1.1 cores mean of 64 at a mean 110 G with 0 recycles, so **parent-bound** and 32 CPUs will not shorten it |
| merge plus the duplicate check | 10 min to 2 h | worst case measured at two full copies with every row duplicated on a contended disk: 1 h 55 min, 43.9 GB peak resident, 0 conflicts. Production hypothesis, untested: 10 to 30 min at 16 processes on a quiet disk |
| CSV endpoint gate on the full build | about 25 min | 1,473 s and 40.1 GB max resident on the 383 G job 4901 partial with dataset resolution on |
| import, validate, swap | 1 to 3 h | KG 3.0 imported 100 M node rows and 361 M edge rows in 7 min; 4.0 writes about 380 G of CSV and its import is **unmeasured** |
| ship plus swap Radiant | about 1 h | `kg_release.sh backup/archive/ship` |

**Total, about 14 to 25 h of node time** (the rows added at their low and their high ends:
9 + 2.6 + 0.2 + 0.4 + 1 + 1 against 13 + 5.2 + 2 + 0.4 + 3 + 1), dominated by the sweep.
The sweep and the generation array both have per-task recovery, so a failure costs the
failed task and not the step. Hold the GPU packs from the sweep's start until the import
finishes, and keep any other job under 24 G during the window.

### Disk, measured at integration 3

| | |
|---|---|
| `/scratch` | 7.0 T, **78 percent full, 1.6 T free**; **407 `*superseded*` directories hold 669 G** and are reclaimable |
| `/db` | 7.3 T, **68 percent full, 2.4 T free** |
| Hoepfner study output on `/db` | **414 G** in three directories of 138 G each; the study is CLOSED and reproducible, so this is the owner's call |
| job 4901 partial on `/db` | 383 G, plus an earlier partial at 170 G |
| served store and its predecessor | 150 G plus 135 G (`data.superseded.2026-10-06_04-41-16`, the rollback copy) |
| what 4.0 needs | about 380 G of CSV on `/db`, plus one new `processed/` per store on `/scratch` |

`/db` does not gate the build at 2.4 T free. `/scratch` is the one to watch: the sweep
writes a new `processed/` per store and retires the old one aside, and the three largest
stores are 126 GB (Hoepfner) and 78 GB each (the Costanzo pair). Pruning the 669 G first
is not strictly required but it is the cheap way to stop watching.

### Two corrections to the lane reports, so the numbers are read right

- **`YeastPhenomeDataset` is NOT in the 169-entry `dataset_adapter_map`**, measured at
  integration 3. So lane X's headline figures (474,837 of 524,682 records carrying a
  truncated medium name, 6 tokens collapsing onto `SC`, and the 14,508 declared duplicate
  pairs that lanes W and X together take to 0) are DEV-store fidelity facts, not served-store
  facts. What lane X moves in the served store is `Media.dropout_roles` on every closure,
  which is why its record is `full_kg_rebuild` over 51 served datasets, and
  `gene_essentiality_sgd`'s "YPD with 2% glucose" carbon source across its 1,329 served
  records. That makes the fix better rather than smaller: the defect is closed before that
  store is served for the first time, which is the whole point of fixing it now.
- **`experiments/database/results/kg4_release_prep_checks.json` is not tracked**, so the
  earlier claim that a re-run was "byte-identical to the one lane M committed" was imprecise:
  nothing was committed. The check was re-run on this head instead, and its result is in the
  gate table above.

### One thing to do before the sweep, found while measuring

**A leftover lane W process is holding 117 GB of the node's 513 G and has produced
nothing.** PID 2830717, started 13:56 CDT, 2 h 55 min elapsed and 2.6 h of CPU at
integration 3:

```
python experiments/database/scripts/environment_response_identity_outcomes.py   --dataset env_chemgen_hoepfner2014 --data-root /scratch/projects/torchcell-scratch
```

It is reading `env_chemgen_hoepfner2014/processed/lmdb/data.mdb` and its log holds nothing
but import warnings. **Lane W's verdict does not depend on it**: lane W re-keyed 15 stores
end to end (9,602,177 records) and stated that Hoepfner's 30.3 M follows by monotonicity, so
it concluded without this arm and left the process running. It is squatting 22 percent of
the node's memory, it is the reason integration 3's own full-tree re-run ran several times
slower than the first, and the sweep's budget was 480 G of 513 G with everything else held, now 384 G under the cap.
**Reap it before the sweep**, and if its result is wanted, re-run it after the build on a
quiet node rather than against the budget.

### Three decisions for the owner, none of them blocking

1. **Lane V's `rename` kind on the essentiality record is a tooling artifact, verified.**
   The dict has 6 keys before and 6 after, identical sets; one key's tuple gained one name;
   `binding_members` renders a dict member as `"key: value"`, so a value edit can only read
   as one lost plus one gained, and all three kinds admissible for a "lost members" reason
   (`rename`, `removal`, `enum_member_removed`) are sign-off kinds, so no honest label
   avoids the sign-off. The record was left as lane V wrote it, signed with a quote that
   says exactly this. **Confirm the reading, or tell it to re-label.** The durable fix is in
   `schema_impact` and would move every dict-binding record's kinds, so it is its own change.
2. **Reclaim the 669 G of superseded copies and the 414 G of Hoepfner study output, or not.**
   Neither is evidence any more (the study's numbers are written up and reproducible), and
   neither blocks the build.
3. **`RbTnseqBorchert2024Adapter`'s tier.** Lane S proposed moving it from `standard` to
   `large` and `torchcell/knowledge_graphs/conf/generation_tiers.yaml` still lists 12 `large`
   adapters without it, so the build as configured runs it at `standard`. It peaked at
   **33.93 GiB of 64 G, 53.0 percent**, with 0 recycles in 522 s, and the branch is about
   6.6 percent heavier than `main` on the one adapter both sides measured, which puts it near
   56.5 percent. The 0.8 recycling threshold of a 64 G container is 51.2 G. Moving it is one
   line and costs the array about 4 min; leaving it accepts a 1.9x margin on the adapter
   closest to its ceiling.

### What is still unknown, stated plainly

- **The 4.0 import is unmeasured.** KG 3.0's 7 min is the only datum and 4.0 writes about
  380 G of CSV against 3.0's smaller set. The 1-to-3 h figure is a hypothesis.
- **The per-adapter generation array has never run end to end.** Lane C's array, lane C2's
  image and lane S's 141-task measurement all ran, but no full 170-task array has. The merge
  step's duplicate check has only its synthetic worst case and lane P's nine directories.
- **97 of the 170 generation charges come from job 4901's single cumulative container**, not
  from a per-adapter container. Three of them are core-bound there (`DmfCostanzo2016Adapter`
  60.8 cores mean, `DmiCostanzo2016Adapter` 60.0, `Hale2024Adapter` 51.4) and get 32 CPUs in
  the array, which is the whole reason the `large` tier could be 5.2 h instead of 2.6 h.
- **24 adapters are charged one 5 s sampling interval**, which bounds them above rather than
  estimating them. 29 of lane S's 47 also finished inside two intervals, so the shape holds.
- **The bacterial tail has never been generated from the branch.** Lane S measured it at
  `main`, because a main-built dev store cannot be unpickled by branch code (`ReleasedGeneKind`
  retired), and the transfer is calibrated on one adapter: Nichols at 219 s and 12.5 GiB on
  the branch against 205.7 s and 11.73 GiB on `main`, so about 6.6 percent slower and heavier.
  That is one ratio, not a fitted model. The sweep is what makes a branch measurement possible
  at all, which is why it comes first.
- **`Mutalik 2020` is the sweep task to watch**: 286,344 records and 37 G store-build resident
  on 2026-10-13 against the 64 G standard tier, and lane P's nine-store sample did not cover it.
- **The three dev stores that can no longer be unpickled** (`betaxanthin_cachera2023`,
  `carotenoid_ozaydin2013`, `crispri_mormino2022`) are expected, not a defect: `ReleasedGeneKind`
  retired and those are exactly the served datasets the records name. The sweep rebuilds them.
- **A collision the sweep surfaces stops the build**, by design: the preflight refuses on any
  undeclared `content_id_collisions` and the only written escape names `EnvChemgenMota2024Dataset`
  at exactly 2. 179 stores and 102,263,636 records have been scanned, so a surprise is unlikely
  rather than impossible.
- **The sweep procedure as executed on 2026-10-13 is not reproducible from the repository**:
  `sweep_conductor.sh` and `release_held.sh` exist only in a session scratchpad (lane M). That
  is a provenance gap of exactly the kind the constitution exists to prevent, and it should be
  committed before the 4.0 sweep rather than after.
- **Two retired bacterial graph labels and every content id change**, so no KG 3.0 query that
  names them and no id-based comparison survives. That is intended and disclosed in
  [[database.kg-4-release-note]].
- **Lane T's compound-span shape is the one breaking design in this program decided by an
  agent rather than by an owner ruling**, and lane T said so in its own sign-off. It is right
  by the INSDC grammar and round-trips 570 of 570 features byte for byte, but it deserves a
  look before the branch lands.

### The irreversibility boundary

Nothing before `neo4j-admin database import` touches the served store. The sweep rewrites
the dev tree (recoverable: every store rebuilds from its sha256-pinned raw mirror, which is
what the 9-to-13 h is), the generation array writes CSV beside it, and the import builds a
NEW store on `/db` which only then swaps in, leaving the old one as `data.superseded.<ts>`
for rollback. So the honest reading is that the expensive irreversible thing is **time**,
not data: a sweep that has to be redone costs most of a day.
