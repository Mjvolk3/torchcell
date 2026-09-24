---
id: iwieqze1ae069druc5ovm1m
title: Runners
desc: ''
updated: 1783563836060
created: 1783563836060
---

## 2026.07.08 - Runners: the importable harness that runs every family's L0-L4 gate and writes reports

This module is the executable top of the framework: it knows WHICH built LMDBs exist, pins each one's `Provenance` and expected-count oracle, loads its records, dispatches to the matching per-family verifier, adds the cross-source L4 checks, and writes a `verification_report.json` beside each dataset. It was moved out of `scripts/` into `src` (`d457b5d8`) so the runners can be imported, tested, and reused -- it REPLACES the old `scripts/verify_expression_datasets.py` + `scripts/verify_morphology_datasets.py`, which could only be run, never composed.

- The per-family dataset registries (expression WS5, morphology WS6, visual-score WS7, metabolite WS8, protein/metabolite WS9, rnaseq WS10) are the single place each source's provenance + count oracle is declared -- provenance travels with the run, not the loader.
- Owns the L4 cross-source assertions that no single verifier can make: expression datasets share one platform gene universe; deletion screens are gene-contained in Ohya's morphology set; RNA-seq genes are contained in the S288C SGD reference. Per-family verifiers only expose the gene-set key; the runner joins them.
- Reads the Ohya LMDB from the (possibly read-only) KG-build tree but WRITES reports to the writable `data/torchcell/...` tree -- decoupling verification output from the build user's ownership.
- `main`/`run_all` returns a shell exit code, so the whole abstract's data can gate CI. Uses the checks in [[torchcell.verification.levels]] and the models in [[torchcell.verification.report]].

## 2026.09.14 - S288C gene universe and the Bloom member index from the genomes tier

`SGD_GENE_FASTAS` holds filenames now and `_sgd_gene_set` resolves them from the tier
set `sgd_S288C_R64-4-1_20230830` (7,146 names, unchanged); the Bloom entry's
`assembly_index` is resolved from `peter2018_1011_assemblies`. `_genome` still passes
`genome_root=data/sgd/genome` as the cache root with `overwrite=False`. See
[[torchcell.sequence.genome.registry]].

## 2026.09.23 - Issue 92 closed by rebuilds: all 11 datasets pass L0 again

Issue #92 reported L0-structural failing on 11/11 pre-2026-07-14 datasets: `Media.is_synthetic`
became a required field with no default in `1cf60cdc` (2026-07-14), so every LMDB built before
that date stored a `media` dict the current `ExperimentType` union no longer accepts, and
`l0_structural` rejected 100% of records with
`loc: ('Experiment', 'environment', 'media', 'is_synthetic') type: missing`.

All 11 dev-tree stores have since been rebuilt (LMDB `data.mdb` mtime 2026-09-14), and the
verifiers now pass every level on every one of them. Re-run today with
`PYTHONPATH=. python scripts/verify_datasets.py`, which narrows the family registries in this
module to exactly the 11 slugs, calls `run_expression`, `run_morphology`,
`run_morphology_ohnuki`, `run_ohnuki_morphology`, `run_visual_score` and `run_metabolite`, and
tabulates the `verification_report.json` each one wrote:

| dataset | records | L0 | L1 | L2 | L3 | L4 | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| microarray_kemmeren2014 | 1484 | PASS | PASS | PASS | PASS | PASS | PASS |
| sm_microarray_sameith2015 | 82 | PASS | PASS | PASS | PASS | PASS | PASS |
| dm_microarray_sameith2015 | 72 | PASS | PASS | PASS | PASS | - | PASS |
| scmd_ohya2005 | 4718 | PASS | PASS | PASS | PASS | PASS | PASS |
| scmd_ohnuki2018 | 1112 | PASS | PASS | PASS | PASS | PASS | PASS |
| scmd_ohnuki2022 | 1979 | PASS | PASS | PASS | PASS | PASS | PASS |
| carotenoid_ozaydin2013 | 4474 | PASS | PASS | PASS | PASS | PASS | PASS |
| betaxanthin_cachera2023 | 4719 | PASS | PASS | PASS | PASS | PASS | PASS |
| amino_acid_mulleder2016 | 4678 | PASS | PASS | PASS | PASS | PASS | PASS |
| metabolite_zelezniak2018 | 95 | PASS | PASS | PASS | PASS | PASS | PASS |
| metabolite_dasilveira2014 | 127 | PASS | PASS | PASS | PASS | PASS | PASS |

`dm_microarray_sameith2015` shows `-` at L4 because it is the reference gene universe the
expression L4 compares the other two microarray datasets against, so no L4 result is recorded
on it; its L0-L3 all pass. Every other dataset carries an L4 gene-containment or gene-universe
check. Reports written 2026.09.23 19:07-19:09 under
`$DATA_ROOT/data/torchcell/<slug>/preprocess/verification_report.json` (`DATA_ROOT` =
`/scratch/projects/torchcell-scratch`), which is the writable dev tree, never `database/`:

- `microarray_kemmeren2014`, `sm_microarray_sameith2015`, `dm_microarray_sameith2015`
- `scmd_ohya2005`, `scmd_ohnuki2018`, `scmd_ohnuki2022`
- `carotenoid_ozaydin2013`
- `betaxanthin_cachera2023`, `amino_acid_mulleder2016`, `metabolite_zelezniak2018`,
  `metabolite_dasilveira2014`

**No back-compat shim was added, and that is deliberate.** Issue #92's option 2 was a
`model_validator(mode="before")` on `Media` that defaults or infers `is_synthetic` when the
stored dict omits it. Nothing needs it: `is_synthetic` is still required with no default and
`Media` carries no before-validator, so L0 passing on the stored records is itself the proof
that every rebuilt store writes the field. Adding the shim now would be a silent fallback that
lets a genuinely pre-2026-07-14 store keep loading while reporting green, which is exactly the
staleness the L0 gate exists to surface. `python -m torchcell.provenance.build_manifest`
independently reads all 11 as `fresh`; the datasets it still reads `[STALE]` on the media
closure (the four `dmf/dmi_costanzo2016` slices) are the Costanzo rebuild, tracked separately.
