---
id: qi08dln0ueuc03xk5b6xxgm
title: Generation_array
desc: ''
updated: 1792055272817
created: 1792055272817
---

## 2026.10.15 - The per-adapter generation array, its merge and its duplicate check

`torchcell/knowledge_graphs/generation_array.py` is the host-side half of the per-adapter KG generation (Path U, lane C of [[scratch.2026.10.15.ontology-unification-decision]]). The serial generation (`create_scerevisiae_kg_small`, one container, one loop over 170 adapters, no checkpoint) restarts from adapter 1 when adapter 124 fails; job 4901 lost 17 h 38 min that way. The array runs one slurm task per adapter ([[database.slurm.scripts.gilahyper_generate_adapter_array-slurm_docker]], the task entry [[torchcell.knowledge_graphs.generate_adapter]]), each into `biocypher-out/<ts>/adapters/<iii>_<Adapter>/` with a `DONE` marker written last, and this module turns the 170 directories back into one import.

### Order

Index `i` is position `i` of `build_adapter_map(include_private, organisms)`: the public map, then the private map under `--include-private`, narrowed to the organism set. That is the order the serial build walks, so `EnvChemgenHoepfner2014Adapter` is index 64 (job 4901's "adapter 65 of 170"). `adapter_order` is the one place that maps an index to a dataset and an adapter; the conductor writes it to `<out>/adapter_order.json` and every task and the merge read the recorded list, so the array and its merge cannot disagree on membership.

### Tiers

[[torchcell.knowledge_graphs.conf.generation_tiers.yaml]] keys adapter names to a tier (`cpus`, `mem_gb`, `hours`, `concurrency`). A slurm array carries one resource spec, so `tier_submissions` emits one `sbatch --array=<ids>%N --cpus-per-task --mem --time` per tier, with `GEN_CPUS` and `GEN_MEM_GB` in the task environment as the container's `--cpus` and `--memory`. Multi-member values (`KG_ORGANISMS`, hydra overrides) travel in the environment with `--export=ALL`, never inside `--export`'s list, which sbatch splits on commas (job 4899).

### The DONE protocol and recovery

`DoneMarker` (index, classes, dataset length, node and edge counts, wall time, peak memory, package version, build commit) is the generator's last write, after `sink.finish()` and `bc.write_import_call()`. `missing_slots` lists the slots whose directory has no marker (a marker naming another slot is an error, not a DONE), and `recovery_commands` prints the exact per-tier sbatch lines that re-run them:

```bash
python -m torchcell.knowledge_graphs.generation_array missing --out-dir /db/database/biocypher-out/<ts> \
    --out-name <ts> --torchcell-src <checkout> --build-commit <sha> --include-private --organisms scerevisiae,ecoli,pputida
```

### Merge

`merge_outputs` refuses while any directory lacks its marker (naming the indices), checks that every label's header is byte-identical across the directories carrying it, runs the duplicate check, and writes one `<out>/neo4j-admin-import-call.sh`: per label the first directory's header and the comma-joined part files of every directory carrying it, nodes before relationships. The command prefix (delimiter, quote, `--overwrite-destination`, `--skip-bad-relationships`, `--skip-duplicate-nodes`) is transplanted from a per-directory call BioCypher wrote, so the flags are the BioCypher config's; the merge refuses a config without `skip_duplicate_nodes: true`. One argument must stay under the kernel's 128 KiB single-argument limit; a label carried by all 170 directories is about 15 KB. `merge_report.json` carries the per-label counts, and `kg_manifest bootstrap --merge-report` records them in the build manifest (`GenerationRecord`).

### Duplicates

BioCypher and `FastCsvSink` deduplicate within one process; across directories content-id nodes recur (genome, environment, media, temperature, perturbation, genotype, experiment reference, interned constant). The check hashes every row of every part file (a 128-bit blake2b of the key, a 64-bit blake2b of the whole line; the key is the node id or `start\x00end` for an edge), sort-merges per label with numpy, and treats a key seen in two or more directories as a duplicate. Node rows: `--skip-duplicate-nodes=true` keeps the first row neo4j sees, and the check proves every later row is byte-identical. Edge rows: neo4j has no skip for duplicate relationships, so the later directories' rows of a duplicate key are dropped into a rewritten copy under `<out>/merged/<dir>/` that the import call references (the adapter directories are never modified). A duplicate that is not byte-identical refuses the merge with the label, the key and the files (`DuplicateConflict`); a key repeated inside one directory is what that adapter wrote and is kept. A row whose tab count differs from the header's is parsed with the csv module (a tab inside a quoted value), so the fast split is exact where it is used.

Cost: the dated section below (the two-copy worst case measured on the job 4901 partial, the production estimate, and the scope decision).

Tests: `tests/torchcell/knowledge_graphs/test_generation_array.py` (the tier sizes and that every named adapter is mapped, the order against `build_adapter_map`, the sbatch plan and the dry run, the marker protocol, the merge on synthetic directories with and without DONE, identical and non-identical duplicates for nodes and edges, the header check, the quoted-tab parse).

## 2026.10.15 - Container startup cost: 278 s per task, so the array wants a prebuilt image

Measured on the Hoepfner baseline task (job 4919, task 64, `4919_64_hoepfner_base_hoepfner.out` and `4919_64_generate.log`): `docker run` at 04:25:22 CDT, the first generator line (`Generating adapter 64 ...`) at 09:30:00 UTC, which is 04:30:00 CDT: **278 s of startup before any record is read**. The container runs the three pip installs the live rebuild's generation container runs (`torchcell` from GitHub at `BUILD_COMMIT` through a blob-filtered clone of the whole repository, `biocypher==0.15.2`, CaLM from GitHub); the torchcell clone and install took about four of those minutes (the CaLM clone, the last step, began at about 04:29 by the log's mtime).

In the serial generation that cost is paid once. In the array it is paid once per task: 170 tasks x 278 s = 13.1 h of container time, and for the standard tier, whose adapters take minutes of real work each, startup is the largest term: 157 standard tasks at `%4` spend about 3 h of calendar on startup alone, on top of 170 full-repository clones from GitHub in the window. The large and Hoepfner tiers absorb it (hours of work each).

Recommendation, not built in this lane: the conductor builds ONE image tagged at the build commit before submitting the arrays (`docker build` from `$IMAGE` with the same three installs baked in, tagged `tc-neo4j-gen:<BUILD_COMMIT[:12]>`), every task runs that image with no pip stage (the array script's container command becomes the generator call alone when `GEN_IMAGE` is set), and the conductor removes the tag after the merge. The fence and the `git diff --quiet $BUILD_COMMIT HEAD` check stay as they are; the image tag carrying the commit is what ties the CSVs to the code. Startup then drops to the container's own start (seconds), and the array's standard tier runs at the adapters' real cost.

Lane C2 measured the prebuilt image on 2026-10-15: install path 255.9 s from container start to the generator's first line (239 s of it pip installs; the adapter itself generating in 1.5 s), image path 17.3, 17.0 and 13.1 s across three runs, cold image build 216 to 259 s, and a same-commit rebuild free on the layer cache, so re-running failed adapters pays no image time. Those numbers are the design's basis; this lane's 278 s on job 4919 agrees with the install path.

## 2026.10.15 - Duplicate-check cost: the two-copy worst case, the production estimate, the scope

**Worst case, measured.** Two symlinked copies of the job 4901 partial (`/db/database/biocypher-out/2026-10-14_03-53-33`: 124 adapters' rows, 49 labels with rows, 410.6 GB and 813 M rows per copy; headers synthesized from each part's first row, `DONE` and a per-directory call written by the harness; `merge --workers 16 --no-rewrite`, so 2 parts per label and 2 scanning processes): every one of the 813 M keys is a cross-directory duplicate and every row is byte-identical. 0 conflicts, 812,821,484 duplicate rows counted, wall **1 h 54 min 57 s**, peak RSS **43.9 GB** (`/usr/bin/time -v`), 6,807 s summed over the labels. The largest labels: `Experiment` 2 x 95.9 M rows (2 x 261 GB) 1,312 s, `GenotypeMemberOf` 1,213 s, `ExperimentReferenceOf` 1,055 s, `PhenotypeMemberOf` 1,013 s, `ExperimentMemberOf` 669 s, `EnvironmentMemberOf` 357 s, `Mentions` 349 s, `PerturbationMemberOf` 245 s, `Genotype` 214 s. Per scanning process the rate was about 200 MB/s on the 2.7 KB `Experiment` rows and 270 k rows/s on 150-byte edge rows (`EnvironmentMemberOf`); the four edge labels of the same row count that took 1,000 to 1,200 s instead of 357 s ran while the Hoepfner study (jobs 4919 and 4920, 64 cores) wrote its CSVs to the same NVMe, so the bound is contended and conservative. The first run of the check, before the per-group work was vectorized with `reduceat`, did not finish the `Experiment` label in 25 min: a Python loop over 96 M duplicate groups is what made the worst case pathological, and the vectorized check's cost is the sort, not the number of recurring keys.

**Production, estimated (hypothesis until the first array build measures it).** The build hashes each row once (one copy, not two), over 170 parts per label with 16 scanning processes instead of 2, on a quiet `/db`. At the measured per-process rates the hashing of 410 GB and 813 M rows is 3 to 6 min with 16 processes if the NVMe sustains about 2 GB/s, the parent's per-label sort-merge adds about a minute per 100 M-row label (eight of them), and the peak memory is one label's rows (at most about 100 M rows x 24 bytes plus the sort's index and copies, about 10 GB; the 44 GB peak above held 192 M). So 10 to 30 min inside the 64 G merge reservation for the 4.0 volume (170 adapters, about 1 B rows); the merge logs the seconds per label (`LabelDuplicates.seconds`) so the first build replaces this estimate with a measurement.

**Scope.** The proposal was to check only the labels whose ids are content hashes, on the reasoning that an `Experiment` row belongs to exactly one adapter. Against the adapter that does not hold: `CellAdapter._experiment_node` gives the node the sha256 of `json.dumps(record.model_dump())`, the fully inlined record with no dataset component, and the dataset is attached only by the `ExperimentMemberOf` edge (`_experiment_to_dataset_edge`), so two datasets that release a byte-identical experiment record write the same `Experiment` id into two directories (the sink's `dup_experiments` set exists for duplicated experiment ids inside one build; job 4919's Hoepfner directory alone had none). Every node label but `Dataset` is content-addressed the same way (`_genotype_node`, the phenotype nodes, environment, media, perturbation, experiment reference, interned constant), and every edge key but `ExperimentMemberOf`'s (experiment -> dataset) can recur with its endpoints. The labels that cannot recur across directories are therefore `Dataset` and `ExperimentMemberOf`, 2.7 % of the bytes and 669 of the 6,807 label-seconds above (10 %); excluding `Experiment` as well would save 64 % of the bytes and 29 % of the label-seconds, at the price of importing a non-identical duplicate unchecked if two loaders ever emit one. The check keeps the full scope: the production cost is the full number above, and the two exclusions are not worth a rule the adapter does not back.

## 2026.10.15 - The prebuilt image is wired in, and the standard tier's peak is measured

Lane C2's image ([[torchcell.knowledge_graphs.generation_image]]) landed as a module with
no caller; its three hooks are now in place, so every task of an array can start in the
generator instead of installing torchcell first.

- The **conductor** takes `GEN_IMAGE_MODE` (`auto`, the default, or `image` or
  `install`) and resolves it ONCE before any array is submitted: `auto` takes the image
  when the tag already exists, `image` builds it when it is absent and refuses if it
  cannot, `install` leaves every task to its own pip installs. The resolved tag is
  passed to `generation_array submit --gen-image`.
- `--gen-image` rides the task environment as `GEN_IMAGE` with `--export=ALL`, beside
  `GEN_CPUS` and `GEN_MEM_GB`.
- The **array task** re-reads the tag's own `org.torchcell.build_commit` label in its
  preflight and refuses when it is not `BUILD_COMMIT`, so a removed or re-tagged image
  never serves another commit and never falls back on the install path. With `GEN_IMAGE`
  unset the task runs the same three installs it always did.
- The **live rebuild** removes the tag after the merge succeeds, which is the point at
  which every task carried its marker; a failed merge or a missing marker exits earlier
  and leaves the tag, so a re-run of the failed tasks pays no image time
  (`teardown_decision`).

Measured on the first array to use it, job 4974 (47 tasks): **container start to the
generator's first line is a median 11.0 s and a mean 11.2 s, range 10 to 14 s**, against
255.9 s on the install path (lane C2) and 278 s in job 4919_64. Cold image build 434 s on
a busy node. Over the real 170-task array that is 11.6 h of container CPU for one 7 min
build.

Job 4974 also closed the standard tier's unmeasured per-task peak, which this module's
tier table flagged in its own comment: 47 adapters of the bacterial tail generated at
16 CPU and 64 G, **one of them past half its tier** (`RbTnseqBorchert2024Adapter`, 33.93
GiB, 53.0 percent, proposed for `large`) and nothing else past 22 percent. Per-adapter
table, the Mutalik 2020 verdict, the endpoint gate's twelve dangling Carruthers rows and
the full-array projection: [[experiments.database.bacterial-generation-profile]].

## 2026.10.16 - Variants: duplicate keys that differ only in non-identity fields

The KG 4.0 follow-up (job 5408, build commit 7b56568db) failed at the merge in 16 s: `DuplicateConflict: Environment: 116 duplicate ids map to rows that are not byte-identical across directories`. The duplicate check had only ever run on two copies of one pooled directory (the worst case in [[database.kg-4-pre-build-verification]]), where every recurring row is byte-identical by construction; a pooled single-process build deduplicated across adapters first-seen inside BioCypher, so a shared node carrying one source's narrative was never visible. With one directory per adapter it is.

Measured on `/db/database/biocypher-out/2026-10-16_13-12-52` (scratch classification over every `Environment` part, 21,297 rows in 170 parts): 445 ids recur across directories, 116 differ byte-wise, and every one of the 116 differs only in fields the identity design excludes. 112 differ in `serialized_data.provenance_gaps` (a duration gap carrying each source's citation key, artifact, page and note: Borchert 2023 against Borchert 2024 13 ids, Borchert 2024 against the Fitness Browser putida store 86, the rest Caglar 2017, Lamoureux 2023, Schmidt 2016, Lim 2025, Price 2018 against Wetmore 2015 and the Costanzo, Ohya, Kemmeren and Messner groups one each), 3 in `media.provenance[]` quotes (which paper's sentence states the recipe), and 1 in the free-text `media.name` (`SC` against `SC (FFA production)`, Xue 2025 beside Kemmeren, Sameith and Lopez). The `EnvironmentPerturbation` label has 5 such ids, Nichols 2011 against Shiver 2016, differing in the plain `compound_name` column; the compound is addressed by its InChIKey. The `FitnessPhenotype` label (8,314,489 rows, 397,933 ids recurring across its 18 directories) has 1 such id, a fitness of exactly 1.0 with no SE that Kuzmin 2018 stores as a measured phenotype (`preferred_id` `phenotype_<id>`) and Kuzmin 2020 and Choe 2019 also write as the reference phenotype (`preferred_id` `fitness phenotype`); `preferred_id` is BioCypher's id-namespace hint, not content. Job 5568 (`gen_adapter_mergecheck`, 32 cpu, 160 G) ran the duplicate check over all 61 labels of the output in 18 min 25 s: these four labels are the only ones with a conflicting key (`/scratch/projects/torchcell/database/slurm/merge_conflict_scan_7b56568db.json`), so the rule below reconciles the whole KG 4.0 generation with 0 refusals.

`environment_identity` drops `provenance_gaps`; `media_identity` drops `name`, `provenance` and every component's `provenance`, `note` and `defers_to` (`torchcell/datamodels/identity.py`). Two rows of one id are therefore one entity by the adapter's own identity, and what they disagree on is who stated it. The merge now reads back every conflicting key (`reconcile_conflicts`) and compares each later directory's row with the lowest-index directory's by `row_differences`: a plain column in `NON_IDENTITY_COLUMNS` (`name`, `compound_name`, `preferred_id`) or a JSON leaf whose path passes through a key in `NON_IDENTITY_JSON_KEYS` (`provenance`, `provenance_gaps`, `note`, `defers_to`, `name`) is a non-identity difference; anything else (a value, a unit, a composition, a column that is JSON in one row and text in the other) is an identity difference and refuses the merge naming the label, the key, the directories and the paths. A key with only non-identity differences is a *variant*: the lowest-index directory's row is the one imported (`--skip-duplicate-nodes` keeps the first row neo4j sees and the call lists directories in index order; an edge's later rows are dropped into the rewritten part as before), `merge_report.json` lists every variant key with its directories and differing paths (`LabelDuplicates.variants`), and the build manifest's `GenerationRecord` carries `node_variants` and `edge_variants` per label. Over 100,000 conflicting keys in one label (`MAX_VARIANT_KEYS`) refuses without reading them back: that is a broken identity, not a provenance pattern.

What the store loses under this rule is exactly what it lost under the pooled build: the per-source gap note and quote on a shared environment or medium node, which stay in each dataset's LMDB and in the report. Attaching per-source provenance to a shared content node is a representation question for the schema (a per-dataset membership edge property, or a provenance node), recorded here as a follow-up and not decided by the merge.

Run on the real `Environment` and `EnvironmentPerturbation` labels from the fix worktree: 116 and 5 conflicting keys, 116 and 5 variants, 0 refused, 1.4 s and 0.2 s.

### Deep dive on the 130 variant keys (job 5569)

`variant_deep_dive.py` (scratchpad, slurm job 5569, 8 cpus, 64 G, read-only) ran `check_label_duplicates` and `reconcile_conflicts` on the four conflicting labels of `/db/database/biocypher-out/2026-10-16_13-12-52` and grouped every variant by the directory pair and the differing paths. Result file: `/scratch/projects/torchcell/database/slurm/variant_deep_dive_7b56568db.json`. 130 of 130 reconcile, 0 refused.

Four families, all measured on the real rows:

| label | keys | directory pairs | what differs |
|---|---|---|---|
| Environment | 116 | Borchert 2024 vs FitnessBrowser putida 86, Borchert 2023 vs 2024 13, Caglar 2017 siblings 5, Lamoureux 2023 siblings 6, Price 2018 vs Wetmore 2015, Costanzo family, Ohya vs Ohnuki, O'Duibhir vs Caudal, Messner vs Boocock, Hillenmeyer vs Parsons, Kemmeren family | `provenance_gaps` on `duration_hours` / `duration_generations` (which paper was looked in, its page, its note), or the medium's `provenance` quotes (the nested `Media.provenance[i]`) |
| Media | 8 | the SC, YPD liquid, YPD solid, SM, SC minus lysine, DM500 media shared across 2 to 13 datasets | `provenance[i]` quotes (each paper's own sentence naming the medium), one `name` (Hillenmeyer "lysine dropout" vs Parsons "synthetic complete media lacking lysine") |
| EnvironmentPerturbation | 5 | Shiver 2016 vs Nichols 2011 | `compound_name` case only: acriflavine vs Acriflavine, amoxicillin vs Amoxicillin, ampicillin vs Ampicillin |
| FitnessPhenotype | 1 | Kuzmin 2018 vs Kuzmin 2020, Choe 2019 | `preferred_id`: `phenotype_<id>` where the fitness 1.0 record is an experiment phenotype vs `fitness phenotype` where it is the reference phenotype |

In every Environment row the identity fields agree (temperature, `duration_hours` null, `duration_generations` null, the same `Media` identity); the rows differ in how each loader documented the absence. Three of the families are the same paper documented differently across sibling loaders (Caglar 072 and 075 carry the duration gap, 074 does not; Costanzo 054 carries a `deferred_pending_source_review` gap with no `looked_in`, 000 and 053 carry none; Lamoureux 104 cites the `time` column, 102 and 103 cite the methods). Those are loader inconsistencies to align later, not source disagreements. The rest are genuinely different papers sharing one physical condition: the 86 Borchert 2024 vs FitnessBrowser keys are the Price 2018 RCH2 and LB media at 30 C with the gap note citing Borchert's release in one and the Fitness Browser deposit in the other.

## 2026.10.17 - The import call sizes its read buffer from the longest row

Follow-up job 5581 (BUILD_COMMIT 8fef5b45e, v1.8.1) merged in 17 min and passed the endpoint gate, then `neo4j-admin database import full` failed before writing a node: `IllegalStateException: Weird input data, no newline character in the whole buffer 4194304` (rerun with `--verbose` inside `tc-neo4j-build`). neo4j's CSV chunker needs a newline inside each read buffer, and its default `--read-buffer-size` is 4 MiB. A `wc -L` scan of every part file over 4 MiB (job 5582, 579 files, `/scratch/projects/torchcell/database/slurm/longest_row_scan_8fef5b45e.txt`) found seven files with a longer row, all `InternedConstant` parts of the Peter 2018 pan-genome datasets (Caudal 2024, De Chiara 2022 x3, Muenzner 2024 x2, Peter 2018): an interned isolate genotype that lists every present pan-genome ORF, longest 5,443,661 bytes. The 3.0 import (2026-10-06, no read-buffer flag) never saw one; these genotypes are new to 4.0.

The merge now measures the longest row of every label in the hash pass it already makes (`scan_part` returns it, `LabelDuplicates.longest_row_bytes`) and writes `--read-buffer-size=<bytes>` into the merged call: neo4j's 4 MiB until a row passes 2 MiB, then twice the longest row rounded up to a MiB (`read_buffer_bytes`; 5,443,661 gives 11 MiB). `MergeReport.longest_row_bytes` / `read_buffer_bytes` and the manifest's `GenerationRecord` record the pair. Memory cost at 64 import threads is under a gigabyte.

## 2026.10.17 - Duplicate node rows are dropped by the merge, not by neo4j

Follow-up job 5590 (v1.8.2, read buffer 11 MiB) imported 165,826,579 nodes in 11 min 17 s and then aborted in "Prepare ID mapper": `Neo4j-admin aborted the import because 1063 bad entries were found, which exceeds the set fault tolerance (1000)`, every entry `Id '<sha>' is defined more than once in group 'global id space'` (`import.report`, 1000 lines, saved to the session scratchpad). `--skip-duplicate-nodes=true` makes a repeated id non-fatal, but neo4j still counts each one as a bad entry against `--bad-tolerance`, whose default is 1000, and the merged CSVs carry about 23 M cross-directory duplicate node rows (Genotype 22,172,281, FitnessPhenotype 404,548, Perturbation 113,280, CrisprConstruct 49,324, ...). The pooled 3.0 build never met this because BioCypher deduplicated inside its one process.

The merge now drops cross-directory duplicate node rows into rewritten parts under `merged/`, exactly as it already did for edges (`check_label_duplicates` computes `drop_rows` for both kinds). `--skip-duplicate-nodes=true` stays as the guard for a key an adapter repeated inside one directory, which the check now counts as `repeated_in_directory_rows` (kept as the adapter wrote it, expected zero, and printed by the CLI when present). Raising `--bad-tolerance` to the duplicate count was the alternative; it would have logged 23 M lines to `import.report` and left the import's "optimized for fault-free data" path, and the report would no longer name real problems.
