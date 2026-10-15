---
id: igssc8tygsvz4juok3urmil
title: Csv_endpoint_gate
desc: ''
updated: 1792055089554
created: 1792055089554
---

## 2026.10.15 - The gate before `neo4j-admin import`

Stage 1 of the ontology unification, guard (c3) of [[scratch.2026.10.15.ontology-audit-C_serving]] section 4, made the mandatory check per [[scratch.2026.10.15.ontology-review-3-serving]] finding 8: every config under `biocypher/config/` sets `skip_bad_relationships: true`, so flipping it alone only moves a dangling-row failure into the import after the whole generation. The flag stays as it is; this module is the hard stop.

`python -m torchcell.knowledge_graphs.csv_endpoint_gate <biocypher-out dir> [--header-dir DIR] [--no-resolve-datasets] [--report PATH]` loads the `:ID` column of every node CSV into one id table (a 64-hex content id packed to its 32 bytes, any other id as its bytes), streams every edge CSV's `:START_ID` and `:END_ID` against it, and reports every endpoint with no node grouped by edge label and side. Where a gap exists it walks the known end of each dangling row along the `member of` ladder to a `Dataset` node, the ladder read off a 1000-row sample of each edge file (the yaml's edge names are not the file labels: `publication mentions experiment` is the `Mentions` file), so the report names the dataset whose records produced the row. Exit 1 on any gap. Rows are read as bytes and split on the tab: BioCypher writes no multiline field and a node id is the first field, which is what lets the 261 G `Experiment` file be read without parsing its blobs.

Wired into `database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm` inside the import branch, after the dev fence and before the build container starts, writing `<job>_csv_endpoint_gate.{txt,json}` beside the other job artifacts; a failure exits the job with "the fix is in the data, regenerate after it". The expected-counts block after the import stays as the live-store counterpart.

Measured on the job 4901 partial (`/db/database/biocypher-out/2026-10-14_03-53-33`, 383 G of CSV from 124 of 170 adapters, 47.3 M `PerturbationMemberOf` rows, read-only, headers borrowed through `--header-dir` because the killed generation never wrote them): 1281 s wall (21 min 21 s under `/usr/bin/time`, one process at `nice 19`, 30.6 G maximum resident set for 188.5 M node ids), over 188,497,777 node rows in 47 labels and 624,323,707 edge rows in 14 labels (47,251,386 `PerturbationMemberOf`, 95,916,158 `Experiment` nodes); the 261 G `Experiment` file alone took about 5 min at 0.8 GB/s. That partial dangles by construction (its `Dataset`, `Genome` and `Temperature` node files are empty, written only after the last adapter), so the measurement ran with `--no-resolve-datasets`; a production run resolves the few gaps it finds through two more edge-file passes.

Tests: `tests/torchcell/knowledge_graphs/test_csv_endpoint_gate.py` on a synthetic directory (the Carruthers shape resolved to its dataset through `GenotypeMemberOf -> ExperimentMemberOf`, an unresolvable dangling `Dataset` end, both sides dangling, borrowed headers, a non-first id column parsed as csv, a wrong field count refused, the CLI's exit code and JSON).
