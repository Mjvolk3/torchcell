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
- [ ] Post-rebuild gates queued: freshness of all 50 + L0-L4 sweep (job 1895) and the all-50 adapter rehearsal at 1,000 records per dataset (job 1896), then the live full rebuild + swap (job 1897) [[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]
