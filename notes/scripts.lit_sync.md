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

## 2026.08.15 - microbe-perturb-seq collection added to the literature sync

A third Zotero collection, **`microbe-perturb-seq`** (key `FE8DQKUH`) -- single-cell
and pooled perturbation-screen *methods* papers -- was added to the TorchCell group
and is now mirrored + OCR'd like `database` and `paper`.

`DEFAULT_COLLECTIONS` in [[torchcell.literature.sync]] is now
`('database', 'paper', 'microbe-perturb-seq')`, so the nightly cron picks up new
drops into the collection with no further change. The mirror stays flat and
collection-agnostic, so the keyed endpoint ([[torchcell.literature.server]]) serves
them with no restart.

### Capture run (2026.08.16 UTC)

```bash
python scripts/lit_sync.py --collection microbe-perturb-seq
```

`microbe-perturb-seq: 15 items | captured=14 present=1` -- **zero failures, zero
unsupported**: every item carried both a DOI and a stored PDF attachment, so the
whole collection was scriptable on the first pass (contrast the `paper` collection,
which surfaced a no-DOI textbook and the `domenzain` no-bytes attachment).

The 14 newly captured keys, all `provenance_complete=True`:

| citation key | DOI |
| --- | --- |
| `nadal-ribellesRiseSinglecellTranscriptomics2024` | 10.1002/yea.3934 |
| `yaoScalableGeneticScreening2024` | 10.1038/s41587-023-01964-9 |
| `dixitPerturbSeqDissectingMolecular2016` | 10.1016/j.cell.2016.11.038 |
| `jarianiNewProtocolSinglecell2020` | 10.7554/eLife.55320 |
| `urbonaiteYeastoptimizedSinglecellTranscriptomics2021` | 10.1038/s42003-021-02320-w |
| `fulcherParallelMeasurementTranscriptomes2024` | 10.1038/s41467-024-54099-z |
| `gaisserHighthroughputSinglecellTranscriptomics2024` | 10.1038/s41596-024-01007-w |
| `brettnerUltraHighthroughputMassively2024` | 10.1002/yea.3927 |
| `leonavicieneRNACytometrySinglecells2023` | 10.1093/nar/gkac918 |
| `leonavicieneMultistepProcessingSingle2020` | 10.1039/d0lc00660b |
| `baronasHighthroughputSinglecellOmics2026` | 10.1126/science.ady7227 |
| `brandnerPooledSinglecellCRISPRa2025` | 10.64898/2025.12.20.695731 |
| `larsonCRISPRInterferenceCRISPRi2013` | 10.1038/nprot.2013.132 |
| `sunGenomescaleCRISPRiScreening2023` | 10.1016/j.engmic.2023.100089 |

OCR markdown ranges 44--191 KB per paper (MinerU on `cuda`); only
`yaoScalableGeneticScreening2024` carried SI PDFs (5), the rest had a single PDF
attachment in Zotero. Mirror went 83 -> 97 citation-key directories; the endpoint's
unauthenticated `/health` reports `n_keys` 98 (97 keys + `_sync_reports`) with **no
restart**, confirming the read-through path.

### Refreshed the one already-present key

`nadal-ribellesSinglecellResolvedGenotypephenotype2025` was already mirrored from the
`database` sync, so the pass classified it `present` and left it alone -- but its
manifest still recorded `collections: ['database']`, which is now stale (the item sits
in both collections). Refreshed in place with
`backfill_key(..., force=True)` ([[torchcell.literature.backfill]]) ->
`collections: ['microbe-perturb-seq', 'database']`, same 10 files, still
`provenance_complete=True`.

**General lesson:** `sync_collection` short-circuits on `manifest.json` existing, so a
paper that gains a *new* collection membership after capture keeps a stale
`collections` list. The mirror is collection-agnostic, so nothing about serving or
retrieval breaks -- only that provenance field drifts. A `force` backfill is the fix;
worth a periodic sweep if collection membership is ever used to select papers.
