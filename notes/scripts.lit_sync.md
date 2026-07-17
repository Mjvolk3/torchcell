---
id: eynip7c910cv2ewolssj3r8
title: Lit_sync
desc: ''
updated: 1784301556855
created: 1784301556855
---

`scripts/lit_sync.py` -- nightly mirror + MinerU OCR of new papers across the
Zotero collections we track. Generalizes the former database-only
`scripts/lit_sync_database.py` to any collection; the cron now runs it over
`database` + `paper`. Engine: `torchcell.literature.sync`
([[torchcell.literature.sync]]). Served read-through by the tc-lit endpoint
([[literature-keyed-endpoint]]), so captured papers appear with no restart.

## 2026.07.17 - Paper collection added to the literature sync

### What changed

- `torchcell/literature/sync.py`: added collection-parameterized
  `sync_collection(lib, collection, ...)`, `plan_collection_sync(...)`,
  `_collection_items(...)`, and a `sync_collections(...)` helper over
  `DEFAULT_COLLECTIONS = ("database", "paper")`. The old
  `sync_database` / `plan_database_sync` remain as thin back-compat shims.
- `scripts/lit_sync.py`: new CLI (`--collection` repeatable, default both;
  `--dry-run` / `--no-ocr` / `--limit`). Self-flocks on
  `/tmp/torchcell-lit-sync.lock`; writes one JSON report per collection under
  `torchcell-library/_sync_reports/`. Supersedes `lit_sync_database.py`.
- `scripts/crontab.txt` + live crontab (GilaHyper): the 03:30 job now runs
  `lit_sync.py --limit 10` (was `lit_sync_database.py`). `--limit` is per
  collection. Log: `/tmp/torchcell-lit-sync.log`.

The mirror is a flat, collection-agnostic namespace
(`torchcell-library/<citation_key>/`), so a key in both collections is captured
once (the second collection pass sees it `present`); no server change was needed
to serve `paper` papers -- only the capture side was collection-bound.

### Paper collection run (18 items)

- **2 present** (already mirrored via the `database` sync):
  `cacheraCRISPAHighthroughputMethod2023`,
  `ozaydinCarotenoidbasedPhenotypicScreen2013a`.
- **1 unsupported** -- `stephanopoulosMetabolicEngineeringPrinciples1998` (a
  textbook: has a PDF but **no DOI**, so `capture_by_doi` has no join key).
  Needs a DOI in Zotero or a hand-run.
- **15 eligible** (DOI + PDF). **14 captured + OCR'd**; **1 blocked** ->
  `domenzainComputationalBiologyPredicts2025` (see below).

### domenzain blocked: Zotero attachment has no stored bytes

The paper the run was kicked off for (PNAS,
DOI `10.1073/pnas.2417322122`) could **not** be captured. Item `APMHNV69` has:

- `488HDKPX` -- "Full Text PDF", `linkMode=imported_url`, `contentType=pdf`,
  **`md5=None`**; `GET .../items/488HDKPX/file` -> **404**.
- `EDMV6BDC` -- an `imported_file` **ZIP** (SI/data), also **`md5=None`**,
  `/file` -> **404**.
- `CF6N9H5C` -- a note.

`md5=None` + `/file` 404 means the **file bytes were never uploaded to the Zotero
*group* file storage** -- only URL-import stubs exist. The recorded source URL
(`https://www.pnas.org/doi/pdf/10.1073/pnas.2417322122`) returns **HTTP 403** to a
scripted fetch, so there is no legitimate scriptable path to the bytes.
`capture_by_doi` correctly refused rather than fabricate. (Live md5 re-queried per
the standing lesson -- this is a genuine gap, not a false "missing".)

**Fix (user, in Zotero):** open the domenzain item in the **group** library,
attach the actual PDF (and ideally the SI) as **stored/imported files** (drag them
in so they upload to group storage), and sync. Once `md5` is populated and
`/file` returns `302 -> S3`, the next `lit_sync.py` pass (or the nightly cron)
captures + OCRs it automatically like the other 14.

### Zotero dup-twins (hygiene, non-blocking)

`kuzminSystematicAnalysisComplex2018a` and `costanzoGlobalGeneticInteraction2016a`
in `paper` are suffixed-key twins of the already-mirrored `...2018` / `...2016`
(from the `database` build). They capture into separate `...a` dirs -- duplicate
OCR of the same DOI. Merge the twin entries in Zotero so the sync stops mirroring
both. Same class of issue noted in [[lit-sync-database-nightly]].
