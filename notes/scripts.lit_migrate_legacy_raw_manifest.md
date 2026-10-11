---
id: czxx4cgu0o5c1hb2dif34xn
title: Lit_migrate_legacy_raw_manifest
desc: ''
updated: 1791697424705
created: 1791697424705
---

## 2026.10.11 - Regenerating the Zhang 2020 raw manifest (#923)

`$DATA_ROOT/torchcell-raw/zhangCombiningMechanisticMachine2020/manifest.json` was written by hand on 2026-09-14 (commit 6d5aab86a). It was the only one of 746 mirror manifests (672 library + 74 raw, measured 2026-10-11 with `python -m torchcell.literature.manifest validate`) that failed `Manifest`: 20 errors, four per file on each of its five files.

```text
20 validation errors for Manifest
files.0.retrieval.retriever
  Field required [type=missing, ...]
files.0.retrieval.sha256
  Field required [type=missing, ...]
files.0.retrieval.retrieval_command
  Extra inputs are not permitted [type=extra_forbidden, ...]
files.0.retrieval.pinned_commit
  Extra inputs are not permitted [type=extra_forbidden, ...]
... the same four for files.1 to files.4
```

### What was regenerated and what was kept

The script rebuilds the manifest by `build_manifest`, so every file's `sha256`, `bytes` and `role` comes from the bytes on disk. All five measured `sha256`/`bytes` equal the legacy pins (the script refuses otherwise).

| field | outcome |
|---|---|
| `sha256`, `bytes` | re-derived; identical to the legacy record for all 5 files |
| `role` | re-derived; `data/README.md` is now `raw_data` (the hand-written record said `readme`, which is not a role `_role_for` produces) |
| `doi`, `title`, per-file `source` | kept |
| `retrieval.method`, `source_url`, `retrieved_at` | kept (`direct_url`, the commit-pinned GitHub raw URL, `2026-09-14`) |
| `retrieval.retriever` + `params` | new: `torchcell.literature.retrieve.direct_url`, `{"url": source_url}` |
| `retrieval.sha256` | the measured file hash |
| `retrieval.last_check` | new: the retriever was re-run on 2026-10-11 and produced the recorded sha256 for 5 of 5 files (`matches: true`) |
| `pinned_commit` | checked to be inside every `source_url`, and kept as `si_data_sources = ["https://github.com/sorpet/Zhang_and_Petersen_et_al_2019/tree/f95f7d103c4a9990446ad6ff7b1a883a0b11db07"]` |
| `retrieval_command` (`gh api ... \| base64 -d`), per-file `description`, `data_availability_note` | not representable in the schema; dropped from the live manifest, preserved in the graveyard copy below |

The retrieval provenance is known, so `provenance_complete` stays `true`; no retrieval was recorded as unknown.

The hand-written file (sha256 `2dc427ce098d0911742c4627b6fbab4aed488c5ef68e0cb8131f197bcc8bcedf`) was moved, not deleted, to `/scratch/projects/torchcell-deprecated/2026-10-11_004056__manifest.json/manifest.json`. The regenerated manifest has sha256 `c0e12b2fd1d4599d7382503210d3050146b8933bc27dce495fb7290f764e505b`. Its `data_availability_note`, verbatim:

> The paper's statement names JBEI ICE (public-registry.jbei.org) for genotypes and JBEI EDD (public-edd.jbei.org) for time series under study 'Zhang and Petersen, et al. 2019', and a GitHub copy. public-edd.jbei.org requires a manually approved account, so the GitHub copy is the retrievable source and is what is mirrored here; the deposit README states it is a copy of the same ICE and EDD records.

Command run:

```bash
python scripts/lit_migrate_legacy_raw_manifest.py \
  --key-dir $DATA_ROOT/torchcell-raw/zhangCombiningMechanisticMachine2020 \
  --legacy /scratch/projects/torchcell-deprecated/2026-10-11_004056__manifest.json/manifest.json \
  --check-source
```

After it, `validate` reports `torchcell-library: 672 valid, 0 failing` and `torchcell-raw: 74 valid, 0 failing`.
