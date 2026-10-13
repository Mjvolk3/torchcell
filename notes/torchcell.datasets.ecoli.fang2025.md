---
id: yntifzp80gyim7p5qkta8ax
title: Fang2025
desc: ''
updated: 1791848119118
created: 1791848119118
---

## 2026.10.12 - Every cluster member stored as Wang 2018 released it (genome-gate reversal)

Module: `torchcell/datasets/ecoli/fang2025.py` (`CrispriGuideFfaEnrichmentFang2025Dataset`); earlier build history is in [[torchcell.datasets.ecoli.wang2018]] and the sweep note [[database.genome-gating-sweep.2026.10.12]] (row 244).

The owner's rule of 2026-10-12: the loader reads no annotation. `reconcile_locus_tags`, `MIN_RESOLVED_FRACTION`, the `ecoli_genome` constructor parameter and the `retired_target` drop rule are gone. Every member of a guide's BLASTN cluster is a knockdown leaf keyed by the b-number and symbol Wang 2018's Supplementary Data 2 releases for it (`ClusterMember.bnumber`, `ClusterMember.symbol`), including `b4700` (`sokE`) and `b4590` (`ybfK`), which GCA_000005845.2 no longer carries: their 3 rows above the read floor (`sokEb4700_32` and `sokEb4700_33` in round 1, `ybfKb4590_11` in round 2) are records, and the two members are no longer pruned from the genotypes of the clusters that carry them beside a current b-number. The L4 genome-reference rows report them as findings (#1069). `stored_gene_names` (the annotation's symbol where it round-trips) is no longer used: the symbol is the release's own.

Pins, measured on the pinned bytes: `EXPECTED_RECORDS` 15,708 -> 15,711, `EXPECTED_PERTURBATIONS` 16,187 -> 16,191, `EXPECTED_GENES` 2,628 -> 2,630; round 1 keeps 15,380 records and round 2 331. `build_accounting.json` drops `retired_targets_dropped` and `reconciliation`; `RoundAccounting` drops `dropped_retired_target`.

Released vs stored: 111,342 released (guide, round) rows = 15,711 records + 95,631 rows below the authors' own 20-read floor (the one remaining rule, the release's own exclusion); 0 dropped on a genome ground.

### 2026.10.12 - The per-round pins follow the restored records

Job 4464 failed at `round_1_cf: wrote 15380 records, the pinned bytes give 15378`: `EXPECTED_RECORDS` moved to 15,711 with the reversal but each `Round.kept_records` still carried its retired-target subtraction. A round's kept records are now exactly its `floor_kept_rows`, 15,380 and 331, which is the 15,711 total.
