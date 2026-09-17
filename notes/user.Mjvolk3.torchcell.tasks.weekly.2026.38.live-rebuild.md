---
id: 9px7t2zinj4dhjnftj2lqpo
title: live-rebuild
desc: ''
updated: 1789369983852
created: 1789369983852
---

## 2026.09.14

- [x] Dataset sweep before the full rebuild: 1,092 tests pass (the known Cachera count assertion fails), the L0-L4 verifier sweep runs one dataset per process (job 1843), and the schema-contract freshness check finds 28 of the 50 mapped dev stores STALE (built in July under the pre-media-library schema; drift on `Compound`, `DoseBasis`, `Environment`, `Media`, `MediaComponent`, `Phenotype`) and 8 with no build manifest at all. Only the 14 stores the serve-all-50 campaign rebuilt read fresh.
- [x] 36 stale or unmanifested stores queued for rebuild from their raw files under the current schema (jobs 1847-1860, 1864-1865, 1873-1882, 1885-1894; the first ten attempts failed on 7474-owned `preprocess/` files left by earlier staging, chowned to the dev user and resubmitted).
- [x] Post-rebuild gates queued: freshness of all 50 + L0-L4 sweep (job 1895) and the all-50 adapter rehearsal at 1,000 records per dataset (job 1896), then the live full rebuild + swap (job 1897) [[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]

## 2026.09.15

- [x] Gates re-run after PR #374 landed and the worktree vanished (1952 verify: 50/50 fresh, 33 PASS, Xue 2025 L4 gene_containment FAIL is pre-existing; 1962 rehearsal PASS after the `genotype member of` input_label and `crispr construct` is_a schema fixes, PR #377). Build held for Cooper 2010, landed as the 51st dataset (PR #382).
- [x] Live rebuild attempts: 1996 (32 CPU / 200 G) OOM-killed at Costanzo dmf with 206 G anonymous memory; 2031 (320 G) pended behind GPU jobs, cancelled; 2032 (24 CPU / 256 G) submitted 19:38 CDT.

## 2026.09.17

- [x] Job 2032 generated all 51 datasets in 28 h 40 min and imported 99,723,455 nodes / 361,895,230 relationships in 17 min, then FAILED at the post-import `neo4j stop` (PID 1 of the build container; container exited, `neo4j start` failed). Store intact and cleanly checkpointed under `/db/database-next/data` (682 G); CSVs 634 G at `/db/database/biocypher-out/2026-09-16_00-44-53`.
- [ ] Fix: restart the container instead of the process, `RESUME_JOB=<old job>` mode that skips generation and import, defaults 24 CPU / 256 G with the measured envelope in the header [[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]. Resume of 2032 submitted after landing; validation, swap and manifest bootstrap recorded here on completion.
