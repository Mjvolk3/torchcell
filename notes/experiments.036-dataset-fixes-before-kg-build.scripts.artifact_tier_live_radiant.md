---
id: 9kvnkfpir2dhrat1vf1g5z4
title: Artifact_tier_live_radiant
desc: ''
updated: 1792391602219
created: 1792391602219
---

## 2026.10.19 - The wiring diagram run against the public hosts

`experiments/036-dataset-fixes-before-kg-build/scripts/artifact_tier_live_radiant.py` is [[database.tc-data-endpoint.mermaid.access-options]] run for real: a process with no `TC_DATA_URL`, no key and an empty artifact data root queries Radiant's served graph for six Caudal 2024 records, the raw stage's gate checks every pointer against the public tc-data, the distinct pointers are listed from the records, and one file is materialized into the cache and verified. Results in `experiments/036-dataset-fixes-before-kg-build/results/artifact_tier_live_radiant.json`.

Measured 2026.10.19 (torchcell 1.8.4 against release 2026.10.17-4295f7ca, KG 4.0, paired with v1.8.3): 6 records, query plus gate 17.3 s (the gate resolved 4 distinct files once each, no download; the first attempt the same morning took 372 s while three of those files were still missing on Taiga), 9 distinct pointers over 4 genomes-tier files (two tarball members per isolate plus the Peter 2018 tables), and `1011LossOfFunction.xls.gz` materialized at 281,760 bytes, sha256 verified, in 0.11 s. Before this run the panel's `artifacts` line read 3/8 because five pointed files had never been shipped; `scripts/ship_artifact_tiers.sh genomes` put all ten sets on Taiga (checksums OK) and the line went to 7/8, the eighth being the Cachera plasmid in the library tier ([[database.tc-data-endpoint]]).
