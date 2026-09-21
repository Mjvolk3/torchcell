---
id: 4zqjigp66boxfqxzdgl70hy
title: 030 Solid Growth Multi
desc: ''
updated: 1789962867830
created: 1789962867830
---

## 2026.09.20 - Why 030, what it is, and where it stands

**Principle.** The query decides what the dataset is: the gene universe, the phenotypes, the
medium, the record shape. It does not decide which measurement of a genotype counts, because that
is a modeling assumption and belongs with the run, beside the split and the seed. A build that has
already chosen has thrown the alternatives away, and the store is the expensive thing to make.

030 = the 025 query (all alleles) + the 029 build (no deduplication) + the record both lacked.

### What made it necessary

The S3 closure recompute found that the 025 build's own fitness reproduces its own trigenic scores
at r 0.230. Three audits then established, from the raw tables and the supplementary text, that the
published score is

    tau_ijk = f_ijk - f_ij f_k - eps_ik - eps_jk

the tau-SGA model with the single-mutant query fitness at 1, reproducing the released value on
99.98 percent of Kuzmin 2018 rows and 99.56 percent of 2020 rows at the rounding floor of the
published columns. Every term is a record except f_ij, the double-mutant query strain's own
fitness, which every trigenic row reports. Dropping that one term alone collapses the
reconstruction to r 0.499 and r 0.479. Detail and the SI quotes:
[[experiments.025-solid-growth.s3-closure]].

### Stages and where each stands

| stage | what | state |
|---|---|---|
| 1 loaders | both Dmf classes emit one record per distinct double-mutant query strain, 172 and 201 | LANDED on main as 4c4a4f950 (PR #412) |
| 2 dev stores | rebuild dmf_kuzmin2018 and dmf_kuzmin2020 | running 2026-09-20 22:53 |
| 3 served graph | admission check, then incremental import | BLOCKED, see below |
| 4 query and build | `queries/001_multi_measurement.cql`, `scripts/query.py`, `scripts/gh_query_build_001.slurm` | written, BLOCKED on disk |
| 5 label policy | pydantic, hashed, applied at read time | delegated to a parallel session |

### Stage 3 is blocked by a policy guard, not by schema drift

`kg_manifest admit` on both datasets returns BLOCKED with every drift check clean: served schema
drift 0, graph schema 0 changed and 0 added, adapter drift none, value surface unchanged. The sole
reason is `[BLOCK] <Dataset> is already in the served store`. The gate is built to ADD a dataset,
and these two are already served. It is a guard, not a mechanical limit: incremental import matches
incoming nodes to existing ones through the uniqueness constraint on the content-addressed id and
runs with `--skip-duplicate-nodes=true`, and `filter_existing_edges` rewrites the edge files without
rows already in the live store. So the mechanics for extending a served dataset exist. Two ways
forward, both needing a decision: teach the gate to admit a served dataset whose records are a
superset (and measure how long the edge filter takes over 410k and 632k records), or take the full
knowledge-graph rebuild, which would also land issue #410.

**A related trap.** The staleness detector will NOT flag this change. It fingerprints the contract
of the schema symbols a loader reaches, excluding docstrings, methods and field descriptions, so a
loader that emits MORE records under the same contract still reads `dev LMDB: fresh`. The two dev
stores had to be cleared by hand. The reference index also caches under `preprocess/`, not
`processed/`, and clearing only `processed/` makes the rebuild fail its coverage assertion, because
the stale index covers the old record count.

### Stage 4 is blocked on disk

The build needs about 3.7 TB of stage copies and /db has 523 GB free. The 025 and 029 intermediates
are 4.5 TB, both archived and verified under `/bulk/experiments/*-intermediates/`; deleting them is
the user's action. `gh_query_build_001.slurm` refuses to start without the space rather than dying
on ENOSPC at hour 20, and asks for 320 GB of memory on 80 CPUs, scaled from 029's measured 155 GB
peak at 26.8M records.

### What the query is, and what it is not

Fifteen blocks, not seventeen. The new measurement needed no new block, because it went into the
existing Dmf datasets rather than a new class: the value is double-mutant fitness, and a second
class for the same quantity would have been a provenance lie carried forever. The only change from
the 025 query is the 029 SynthLethDB rule, exactly two perturbation NODES, which drops the records
that list one gene twice and so put two fitness values into one genotype.
