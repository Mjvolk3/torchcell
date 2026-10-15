---
id: 0fxo2qpmfsb7wlx9za6nzud
title: Content_id_collision_scan
desc: ''
updated: 1792084087254
created: 1792084087254
---

## 2026.10.15 - Every dev store asked whether two of its records share one Experiment content id (lane Q)

`experiments/database/scripts/content_id_collision_scan.py` reads every store under `$DATA_ROOT/data/torchcell/<slug>/processed/lmdb` (superseded and deprecated copies skipped), resolves each record's interned pointers as the adapter does, hashes `json.dumps(record["experiment"])` and counts, per store, the records, the distinct digests, the collision groups, their size histogram and up to five example groups. It is the question lane P's `n_distinct_experiments` answered for nine stores ([[database.kg-4-pre-build-verification]], finding 2), asked of all of them, because a content-id collision inside one dataset is the defect class lane N fixed for interned constants ([[torchcell.datamodels.interned_constant]]): the CSV sink deduplicates node ids, so the second record is in the store and not in the graph, with no error. The shared tree was written by `main` under the pre-unification schema, so a retired enum class (`ReleasedGeneKind`) is unpickled as its plain value through the tolerant unpickler lane P used; `json.dumps` of the dump is unchanged by that.

Run read-only on the shared dev tree (`--data-root /scratch/projects/torchcell-scratch --workers 6 --skip-larger-than-gb 40`, the host, 2026-10-15); result `experiments/database/results/content_id_collision_scan.json`. The three stores over 40 GB (`dmf_costanzo2016` and `dmi_costanzo2016` at 77.8 GB, `env_chemgen_hoepfner2014` at 125.7 GB) were not scanned by this run; their `_5e5` subsets were, with no collision.

### Four stores collide, not one

| store | served in 3.0 | records | distinct | groups (sizes) | lost in the graph | what tells the colliding rows apart in the release |
|---|---|---|---|---|---|---|
| `carotenoid_ozaydin2013` | yes | 4,607 | 4,576 | 31 (2) | 31 | the strain's collection position (`record no.`, `Batch`, `Plate`, `Row`, `Col`); fixed by lane Q, [[torchcell.datasets.scerevisiae.ozaydin2013]] |
| `gene_essentiality_sgd` | yes | 1,329 | 1,311 | 17 (16 of 2, 1 of 3) | 18 | two SGD annotations of one gene under different references: records 1147 and 1148 (YOR122C, inviable null) carry PubMed 12140549 and 20413493 on their `Publication`, identical experiment and reference; the graph keeps both Publication nodes and one Experiment node |
| `env_chemgen_mota2024` | yes | 1,289 | 1,287 | 2 (2) | 2 | the SI lists RNR4 twice in Table S1 and in Table S2 (`RNR4` and `RNR4` with a trailing non-breaking space, the same `+` score); two released rows with no distinguishing cell once the name is stripped |
| `growth_rate_lamoureux2023` | no (bacterial, not in 3.0) | 354 | 340 | 13 (12 of 2, 1 of 3) | 14 | two PRECISE-1K samples in one condition with the same growth rate (records 144 and 146 are `p1k_00225` and `p1k_00227`, `preprocess/record_samples.json`); the sample id is not on the record, the references differ |

Three of the four are pre-existing on `main` (the shared tree is main-built) and three are served in KG 3.0, so the served graph is already short 18 SGD essentiality annotations, 2 Mota rows and 31 Ozaydin measurements. The class of defect is one: the record does not carry the released fact that distinguishes two measurements, and the content id makes them one node. The discriminator differs per source (a collection position, an annotation's reference, a duplicated released row, a sample id), so each is its own loader decision under "the data is the data": Ozaydin's is landed here; the other three are not changed by lane Q and are reported for the owner. Every other scanned store has 0 collision groups.

Totals of the finished run: 176 stores scanned, 30,543,838 records hashed, 4 stores with a collision (63 collision groups, 65 records in a store and not in its graph), 172 stores with none, Hale 2024's 8,692,635 segregant records among them (0 groups, 1,424 s). Skipped by stated reason: the three stores over 40 GB and the two `showcase_*` query dumps.

## 2026.10.15 - The three stores lane Q skipped, scanned under slurm: all three lose nothing (lane V)

Lane Q's run left the three largest dev stores unscanned with the size as the stated
reason (`--skip-larger-than-gb 40`). They are the three biggest datasets in the graph, so
an unmeasured collision in them could have dwarfed everything found; this is that
measurement. `experiments/database/scripts/content_id_collision_scan.slurm` runs the same
scanner on ONE store per job (16 CPU, 64 G, 8 h, `--workers 1`), read-only on the shared
dev tree, writing a per-store result JSON so two jobs never write one file. Hoepfner and
Costanzo dmf ran together; dmi ran after Hoepfner with `--dependency=afterany`, which is
the at-most-two-at-once the shared node allows.

| store | records | distinct | collision groups | records lost in the graph | wall | max RSS | job |
|---|---|---|---|---|---|---|---|
| `dmf_costanzo2016` | 20,705,612 | 20,705,612 | 0 | **0** | 1,792 s | 67.1 GB | 4983 |
| `dmi_costanzo2016` | 20,705,612 | 20,705,612 | 0 | **0** | 1,864 s | 67.0 GB | 4984 |
| `env_chemgen_hoepfner2014` | 30,308,574 | 30,308,574 | 0 | **0** | 4,115 s | 67.2 GB | 4982 |

**The three largest datasets lose nothing.** Every record of each store has its own
`Experiment` content id, so the collisions are confined to the four stores lane Q named
and the dev tree is now measured end to end: 179 stores and
102,263,636 records hashed across the two runs
(30,543,838 in lane Q's 176, 71,719,798 here).

The scanner was left as lane Q wrote it, with no streaming rewrite, because the
reservation was enough and that is measured rather than assumed: `/usr/bin/time -v`
reports a maximum resident set of 67.1 GB, 67.0 GB and 67.2 GB against the 64 GiB
(67.11 GB) reservation, with no OOM kill on any of the three. The number sits at the
cgroup's limit because most of it is the LMDB mmap's page cache, which the kernel
reclaims under pressure rather than OOM-killing for; the Python heap is the digest dict,
about 4.5 GB for Hoepfner's 30.3 M records. The build-manifest walk that now measures the
same number on every store ([[torchcell.provenance.build_manifest]]) does use a bounded
form, because it runs inside every dataset build rather than in a job sized for it.

Throughput, for sizing a later run: 11,556 records/s for Costanzo dmf (1,791.7 s),
11,108 for dmi (1,864.1 s) and 7,365 for Hoepfner (4,114.9 s), the last two slowed by
sharing the NVMe with the other scan for part of their run.
