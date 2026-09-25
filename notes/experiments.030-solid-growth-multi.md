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

## 2026.09.25 - Build 001 complete, and the trigenic identity with the query strain's double

Build 001 finished 2026-09-24 21:06 CDT (slurm 2791, 1 d 6 h 58 m at 256 GB and 63 CPUs, peak
resident 258 GB, which is page cache filling the request; the build process held 55 GB). The raw
query (slurm 2747) returned 43,820,352 records in 2 h 48 min plus 7 h of index passes;
aggregation produced 13,525,088 groups; the processed store is 955 GB at
`/db/experiments/030-solid-growth-multi-001-multi-build`. Stage times measured on 2791:
conversion 3 h 53 (the forked, batched converter at 4,630 records/s), aggregation 8 h 46, processed
copy 1 h 05, experiment types 2 h 33, label table 8 h 41, then three index passes of about 2 h 40
each. The index stage writes no progress line; its passes are dated from the files it drops in
`processed/`. The raw, conversion and aggregation stores are archived on /bulk (slurm 2820, 3.1 TB
to 15.5 GB, sha256 and exact decompressed byte counts in `archive.log`); the first archive attempt
aborted after its test had passed because zstd 1.5.5 reports a 1 TB stream in GiB and the script
parsed for bytes, so `scripts/archive_build_intermediates.sh` now counts the bytes itself.

### The identity closes on Kuzmin 2018 where the query strain's double is in the build

`scripts/closure_recompute_030.py` scans the store (every triple, the 739,236 doubles inside them,
every single, each entry with its strain identifier) and scores the published identity
tau = f_ijk - f_ij f_k - eps_ik - eps_jk under the Kuzmin-first policy with the triple's own year
promoted, on identical records two ways: f_ij chosen by `LabelPolicy.select_double`, which prefers
the entry whose strain token equals the triple's query strain, and f_ij with those entries
excluded, which is the 029 reading. Query and array roles resolve for 376,732 of 376,732 triples.
The build carries 372 query-strain double entries (172 Kuzmin 2018, 200 Kuzmin 2020), and because
each query strain was screened against hundreds of array genes they cover 86,111 of 91,111
Kuzmin 2018 triples and 255,296 of 300,187 Kuzmin 2020 triples. Results in
`results/closure_030_by_screen.csv`, the LaTeX table `results/t10-030-closure.tex` (slurm 2819 scan,
2834 analyze; smoke test on the 029 store, slurm 2818, reproduced its published 0.511).

| screen, stratum | n | 029 reading | with the query strain's double |
|---|---|---|---|
| Kuzmin 2018, deletion arrays (029's stratum) | 57,877 | 0.511 (029: 0.511 on 57,451) | 0.778 |
| Kuzmin 2018, all triples | 91,111 | 0.410 | 0.729 |
| Kuzmin 2018, query double in the build | 86,111 | 0.435 | 0.951, rmse 0.016, median abs residual 3.6e-5 |
| Kuzmin 2020, deletion arrays | 250,062 | 0.325 (029: 0.325 on 236,225) | 0.543 |
| Kuzmin 2020, all triples | 298,785 | 0.337 | 0.543 |
| Kuzmin 2020, query double in the build | 255,064 | 0.338 | 0.660, rmse 0.068, median abs residual 0.032 |

Readings. (1) On the 029 stratum with the query-strain entries excluded, 030 reproduces 029 to
three decimals on both screens, so the extra alleles of the 025 query and the larger store change
nothing on that reading. (2) On Kuzmin 2018 the one record closes the gap: where the query strain's
double is in the build the identity reproduces the released score with a median absolute residual
of 3.6e-5, that is exactly for most triples, and r 0.951 against 0.985 measured on the raw tables;
the remaining distance is a tail of residuals, not a shift (slope 0.98). (3) On Kuzmin 2020 the
record helps, 0.338 to 0.660 with the error halved, but the residual is 0.032 at the median, so
most 2020 triples are still not reproduced exactly; something else in the 2020 computation differs
from what the policy selects. Hypothesis (untested): the control terms eps_ik, eps_jk and f_k are
chosen from the pair's pooled entries regardless of the ARRAY strain, since a double's two
perturbations carry two strain identifiers and the policy matches on the query token only, whereas
the within-screen recompute of [[experiments.025-solid-growth.s3-closure]] matched the same array
strain identifier; the 2020 screen mixes deletion and temperature-sensitive array alleles of one
gene and both years' screens of one pair. Testing it needs the scan to keep both strain identifiers
of a double and an array-strain match for the control terms.

The dendron note of the script is
[[experiments.030-solid-growth-multi.scripts.closure_recompute_030]]; the comparison table goes
into `notes-tex/025-s3-closure` as Table 10.
