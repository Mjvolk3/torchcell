---
id: 6ngbxuumnz7miwxwycbuo4t
title: Prebuild_verify_served_dump
desc: ''
updated: 1792076426664
created: 1792076426664
---

## 2026.10.15 - Read-only dump of the served KG 3.0 records of the lane P sample

`experiments/database/scripts/prebuild_verify_served_dump.py` streams, per dataset,
every experiment under the `Dataset` node through its `ExperimentMemberOf` edges (no
property index needed) and writes one JSONL row per experiment (id, pointered blob, the
labels of the nodes its `GenotypeMemberOf` edges reach) plus one file of every
`InternedConstant` node, so the `$ref` pointers resolve offline. Every statement is a
`MATCH` in a read transaction (driver 4.4: `read_transaction`). It runs under slurm
through `prebuild_verify_served_dump.slurm`, never from the login shell. Job 4934
(2026-10-15): 21,020 interned constants in 12 s, then 1,539 + 1,312 + 943 + 4,719 +
4,474 + 7,053 served experiments in about a second each, exactly the release counts of
`database/releases/2026.10.06-4b293d34.json`. Consumed by
[[experiments.database.scripts.prebuild_verify_stores]]; results in
[[database.kg-4-pre-build-verification]].
