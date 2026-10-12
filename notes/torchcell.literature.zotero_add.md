---
id: yvk84xrx97n9lu4926d5562
title: Zotero_add
desc: ''
updated: 1791795107677
created: 1791795107677
---

## 2026.10.12 - The sanctioned agent write path into the database collections

Owner policy of 2026-10-12, recorded in the repo `CLAUDE.md` (Provenance & Reproducibility): in the TorchCell group library (6582362) the `database/<Organism>` collections exist to collect each dataset's primary paper, from the publisher first, and an agent may file into them. This module is the only code that does so; `scripts/zotero_add_paper.py` ([[scripts.zotero_add_paper]]) is its command line.

**Allowed:** create an item for a dataset's primary paper with its PDF attached, in the named `database/<Organism>` collection; attach SI PDFs to that item; attach the article PDF to an existing item that has none (`--attach-to`), when the item sits in the named collection or in the `database` root, where the owner files dataset papers too (its collection membership is left as it is).

**Refused (`RefusedError`, raised before the first write):**

- a collection that is not exactly `database/<Name>` (the `database` root, `paper`, notes-tex and topic collections), or a library that is not group 6582362;
- a DOI Crossref does not know, or whose Crossref first author (case- and accent-insensitive) or title substring differs from `--first-author` / `--title-contains`;
- a DOI the group library already holds (`ZoteroLibrary.find_item_by_doi`), unless `--attach-to` names that very item, which must also sit in the named collection or the `database` root;
- a second article PDF on an item that has one, and any file whose MD5 an existing attachment already has;
- a file that is not a PDF, a raw-mirror copy whose bytes no longer hash to its manifest, a raw-mirror record without its retrieval, and a DOI two raw keys claim.

A metadata-only item cannot result: every retrieval and check runs before the item is created, and if the upload of its PDF fails the item this run created is deleted again and the error re-raised. After the uploads, the MD5 Zotero stores for each attachment is compared with the bytes sent.

**Article PDF routes, first verified PDF wins:**

| route | what | recorded as |
|---|---|---|
| `publisher` | Crossref `application/pdf` links, then the member's PDF URL (Elsevier `pdfft`, Springer `content/pdf`, ASM/ACS/AAAS `/doi/pdf`, Wiley `pdfdirect`, PLOS `printable`) | `direct_url` retrieval of that URL |
| `raw_mirror` | the `paper_pdf` a `$DATA_ROOT/torchcell-raw/<key>/manifest.json` with this DOI records, re-hashed | that file's recorded retrieval, unchanged (no re-download) |
| `pmc_cloud` | `<PMCID>.<v>/<PMCID>.<v>.pdf` of the newest version in the PMC Article Datasets bucket | `pmc_cloud` retrieval, retriever `pmc_cloud_object` |
| `local_file` | `--pdf`, a browser download, with `--pdf-source-url` | `manual_browser`, retriever `manual` |

A response counts as a PDF only with `%PDF-` in its first KiB and `%%EOF` in its last 2 KiB, so an HTML interstitial with status 200 (the `nature.com` redirect, PMC's proof-of-work page) is reported `not_pdf`, never attached. Each attachment's Zotero note carries its `RetrievalRecord` JSON. The article attachment is titled `Full Text PDF` and SI attachments `Supplementary material: <file>`, which `_is_main_article` / `_is_si_attachment` in [[torchcell.literature.zotero]] classify as main and SI, so `lit_sync` writes the article to `paper.pdf`.

**SI costs something.** An SI PDF attached in Zotero is mirrored by `lit_sync` as `si/si<N>.pdf` without an `original_filename`, and `capture_si.foreign_si` then reports the key `present` and never fetches the publisher's supplementary files (data tables included) for it ([[torchcell.literature.capture_si]]). Attach SI only when the whole SI is PDFs or the data tables are captured another way.

Measured 2026-10-12 from GilaHyper (scratch probes, both User-Agents): ScienceDirect `pdfft`, `cell.com` `showPdf`, `academic.oup.com` `article-pdf`, `journals.asm.org/doi/pdf` and `science.org/doi/pdf` answer HTTP 403; `nature.com/articles/<id>.pdf` answers 200 with the HTML article for a non-subscriber; Europe PMC's `?pdf=render` answers 403; `api.elsevier.com` needs an API key the repo `.env` does not hold. The PMC bucket holds an article PDF for open-access articles and none for NIH author manuscripts (txt, xml and json only, e.g. `PMC3060659.1`, `PMC4882250.1`, `PMC4006352.1`, `PMC9804519.1`).

Tests: [[tests.torchcell.literature.test_zotero_add]], 78 cases, hermetic (an `httpx.MockTransport` that raises on any unscripted URL, and a recording Zotero whose write methods only log). Line coverage of this module 97% (`coverage run --branch`, 2026-10-12).
