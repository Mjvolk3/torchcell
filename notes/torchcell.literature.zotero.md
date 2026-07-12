---
id: 693wv83p8a8vbiu4g2c16xz
title: Zotero
desc: ''
updated: 1783563777251
created: 1783563777251
---

## 2026.07.08 - DOI-keyed access to the canonical-PDF library as a mirror backstop

This module exists to make the Zotero group library (canonical PDFs only) programmatically usable as a provenance backstop: given a paper's DOI, find its item, enumerate its PDF attachments, and lay them down in the deterministic on-disk artifact layout (`<data_root>/torchcell-library/<citation_key>/paper.pdf` + `si/si*.pdf`). It is the fallback path for the un-scriptable sources (nature.com, PMC file downloads) that [[torchcell.literature.retrieve]] cannot fetch directly, and the source of the citation_key that keys every artifact directory.

- DOI lookup is a full `everything(items())` scan matched on each item's `data["DOI"]`, because Zotero's quick search does not index the DOI field -- reliability over cleverness for a curated library.
- Distinguishes main article vs SI attachments so `paper.pdf` and `si/` are populated correctly; citation key is taken from Zotero's own field/`extra` so the directory stays byte-identical to what the user sees (deferring to [[torchcell.literature.citation_keys]] only as a last resort).
- Retry-hardened against the flaky Zotero web API: transient 5xx / transport errors back off and retry; a 404 is terminal, never silently degraded.
- Feeds [[torchcell.literature.capture]] and, via per-file sources/md5, [[torchcell.literature.manifest]].
