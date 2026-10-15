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
