---
id: 8aka9vbgl9nh132lawkhzs1
title: Capture_si
desc: ''
updated: 1791364441588
created: 1791364441588
---

## 2026.10.07 - Publisher supplementary files into mirrored keys

The nightly sync ([[torchcell.literature.sync]]) mirrors what Zotero holds, which for most papers is the article PDF alone. The 50 bacterial papers filed on 2026-10-07 under `database/Escherichia-coli` and `database/Pseudomonas-putida` have no SI attachment in Zotero, and their data tables live in the publisher's supplementary files. This module fetches those files into keys the sync has already captured. Command line: [[scripts.lit_capture_si]]. Tests: [[tests.torchcell.literature.test_capture_si]].

### What a capture writes

For each resolved file not already recorded:

- the bytes at `<key>/si/si<N>.<ext>`, with `N` continuing after the largest `si<N>` index on disk or in the manifest, so OCR (`ocr_artifact` globs `si/si*.pdf`) and the manifest role rules (`si_pdf` for a PDF, `si_data` for anything else under `si/`) apply unchanged;
- an `ArtifactRecord` appended to `manifest.json` with the sha256, `source` (the URL), `original_filename` (the publisher's name; a new optional `FileRecord` field, omitted from the JSON when unset so older manifests serialize byte-identically), and a `RetrievalRecord` naming the retriever's dotted path and params, so `provenance.check_source` can re-run it;
- the source URL appended to `si_data_sources`.

The file is written under a hidden `.part` name and renamed into place, and the manifest is rewritten after every file, so an interrupted run never leaves an unrecorded `si<N>` file. The manifest is appended to, never rebuilt: a forced `backfill_key` rebuild (also step 5 of [[torchcell.literature.reocr_si]]) would drop every `retrieval` record and `original_filename`, because `build_manifest` scans files and knows nothing of how they were fetched.

### Routes, in order

The first route that yields files is used; every route tried is listed in the report as `<route>:<status>`.

| route | how files are listed | how a file is fetched | measured 2026-10-07 |
|---|---|---|---|
| `pmc_cloud` | PMCID from the NCBI ID converter; newest `PMC<id>.<v>/` prefix of the `pmc-oa-opendata` bucket; `<supplementary-material>` media in its JATS XML | `retrieve.pmc_cloud_object(key)` | 37 of the 50 bacterial papers, all listed files hosted |
| `springer` (Crossref member 297: Nature, BMC, EMBO) | ESM links on `link.springer.com/article/<doi>` | `retrieve.springer_esm(url)` on `static-content.springer.com` | 2 of 50 (nmeth.1239, s41586-018-0124-0) |
| `plos` (340) | supplementary objects in the article XML; file name from the 302 `Location` | `retrieve.plos_supplementary(journal, object_doi)` | all 5 PLOS papers went through PMC first; route verified on pbio.2005130 (30 files) |
| `elsevier` (78) | PII from Crossref `alternative-id`; HEAD probe of `ars.els-cdn.com/.../1-s2.0-<PII>-mmc<N>.<ext>` over 43 extensions until two consecutive numbers miss | `retrieve.elsevier_mmc(pii, filename)` | 7 of 50 (two of them author manuscripts whose PMC copy lists the files but hosts none) |
| `elife` (4374) | `additionalFiles` and figure `sourceData` from `api.elifesciences.org` | `retrieve.direct_url(url)` | route verified on eLife 05224 (8 files); none of the 50 |
| `asm`, `acs`, `aaas` (235, 316, 221) | `suppl_file` links on the Atypon `/doi/suppl/<doi>` page | `retrieve.direct_url(url)` | HTTP 403 for every page tried |
| `wiley` (311) | `downloadSupplement` links on the article page | `retrieve.direct_url(url)` | HTTP 403 |
| OUP (286), bioRxiv (246), anything else | none | none | manual by design |

Findings that shaped the routes:

- The PMC OA web service (`oa.fcgi`, behind the old `retrieve.pmc_oa_api`) answers HTTP 404 for every id tried; the PMC Article Datasets bucket on AWS replaces it. The bucket holds OA and author-manuscript articles; an author manuscript's XML lists its supplements but the bucket holds none of them, so those fall through to the publisher.
- A PMC JATS that lists no supplementary material does not prove the paper has none: `sameithHighresolutionGeneExpression2015` (PMC4690272) lists none in PMC and four ESM files on Springer. A PMC `none_listed` therefore always goes on to the publisher.
- `link.springer.com` serves a JavaScript challenge page with HTTP 200 to a browser-like User-Agent from this client and the article to `torchcell-literature (+https://github.com/Mjvolk3/torchcell)`, so discovery identifies itself as a script. A page that never names the DOI is treated as a block.
- The Elsevier CDN serves files directly while ScienceDirect and cell.com answer 403. The probe cannot tell "no SI" from an SI with an unprobed extension, so an empty probe is `manual`, never `none_listed`; a number with no file between two found numbers is reported in `unretrieved`.
- Where both exist, the PMC bucket bytes and the publisher bytes agree: Caudal 2024 `MOESM3_ESM.xlsx` from the bucket has the sha256 already recorded for the Springer ESM copy (`753e17d6...`), and the PMC and publisher file counts matched for three yeast keys (17, 30, 4).

### Outcomes

| outcome | meaning |
|---|---|
| `captured` | every file the route listed was fetched and recorded |
| `would_capture` | dry run: the planned `si/si<N>` paths and source URLs, nothing written |
| `partial` | some listed files could not be fetched (a 401/403/429 on download, an Elsevier numbering hole); `unretrieved` names them and `manual_url` says where to get them |
| `present` | every resolved file is already recorded, or the key holds SI from another path (a Zotero SI attachment, Dryad, a hand retrieval: any `si_pdf`/`si_data` record without `original_filename`), which is left alone so the publisher copy is not duplicated |
| `none_listed` | an enumerating route (Springer page, PLOS XML, eLife API, an Atypon page that loads) lists no supplementary file |
| `manual` | blocked or no scripted route; `manual_url` is the page a person opens |
| `no_mirror_dir` | the key has no `manifest.json` yet (the paper sync has not captured it) |
| `failed` | anything else raised; `error` says what, and the batch continues |

Idempotency is per file: a candidate whose retriever and params match a recorded `retrieval` is never downloaded again, so a rerun after a full capture is `present` with no object request, and a rerun after a `partial` or `failed` run fetches only what is missing.

### The manual path

A `manual` key is the "manual-once, deposit, reproducible via the mirror" case of the provenance rules: open `manual_url` in a browser, download the supplementary files, and deposit them into `<key>/si/` with a `RetrievalRecord` of method `manual_browser` whose command is the recipe (URL plus the click path). This module has no deposit command yet; until it does, a deposit is a hand step recorded the same way the raw-data mirror records its manual files. The 2026-10-07 dry run leaves four of the 50 bacterial papers manual, all blocked by HTTP 403:

- `ishiiMultipleHighThroughputAnalyses2007` (Science, no PMCID): `https://www.science.org/doi/suppl/10.1126/science.1132067`
- `tianRedirectingMetabolicFlux2019` (ACS Synth Biol, no PMCID): `https://pubs.acs.org/doi/suppl/10.1021/acssynbio.8b00429`
- `thompsonFattyAcidAlcohol2020` (AEM, PMC7580535 not in the bucket): `https://journals.asm.org/doi/suppl/10.1128/AEM.01665-20`
- `schmidtNitrogenMetabolismPseudomonas2022` (AEM, PMC9004399 not in the bucket): `https://journals.asm.org/doi/suppl/10.1128/aem.02430-21`

### Validation on 2026-10-07

- Dry run over the two collections against the real mirror: 50 `no_mirror_dir` at 09:06 UTC (the paper sync had not captured any of them yet), then 7 `would_capture` and 43 `no_mirror_dir` at 09:18 UTC once the bacterial sync had mirrored its first keys; the 7 resolved exactly as on the stub keys (pmc_cloud 6, elsevier 1, 84 files).
- Dry run over stub keys for the same 50 DOIs (scratch mirror holding only a manifest per key): 46 `would_capture` (pmc_cloud 37, elsevier 7, springer 2; 377 files, about 726 MB through the PMC route alone), 4 `manual`.
- Dry run on mirrored yeast keys: `kemmerenLargeScaleGeneticPerturbations2014` elsevier 5, `ohnukiHighdimensionalSinglecellPhenotyping2018` pmc_cloud 30, `sameithHighresolutionGeneExpression2015` springer 4, `oduibhirCellCyclePopulation2014` pmc_cloud 17, `mullederFunctionalMetabolomicsDescribes2016` pmc_cloud 4, `caudalPantranscriptomeRevealsLarge2024` present (SI from the issue #598 script).
- One real capture into a scratch copy of `sameithHighresolutionGeneExpression2015`: four files written as `si/si1.xlsx` to `si/si4.xlsx`, each record verified by `verify_artifact` and re-fetched to a matching sha256 by `check_source`; a second run was `present` with no download. The real mirror's manifest was unchanged (same sha256 before and after).
