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

## 2026.10.01 - Every collection listing is paged (issue #563)

Previous behavior: `list_collections` called `zot.collections()` without `everything(...)`, and pyzotero returns at most `limit=100` rows per request, so `collection_key` searched only the first 100 collections. `collection_tree` paged its own listing but resolved its root through that unpaged lookup. When the personal `torchcell` root left the first page on 2026-09-21, the nightly `scripts/lit_bib_store.py` failed with `Zotero collection 'torchcell' not found` (`/tmp/torchcell-lit-bib-store.log` lists exactly 100 names with `torchcell-topics` present and `torchcell` absent), and the same lookup failed the `lit_sync.py` personal roots and the annotations job. With `create_if_missing=True` a name past page one would have been created a second time (no caller passed it).

Fix: `list_collections` reads `everything(collections())`; `collection_key` and `collection_tree` resolve names through one helper, `_resolve_name`, over that full listing (`collection_tree` now makes one paged listing, not two). Two collections sharing a name are now refused with a `ValueError` naming each candidate's key and parent path ("Address it by collection key"), replacing first-match-wins. Every nightly name lookup (group `database`, `paper`, `microbe-perturb-seq`; personal `torchcell`) resolves to exactly one collection in the live check, so the refusal does not change the nightly.

Live read-only check (2026-10-01, paged listings only): group library 4 collections, personal library 122 collections, personal `torchcell` = `ICDCVSL6`; personal `thesis` is not in the library at all (a missing collection, not a paging miss).

Tests: `test_list_collections_reads_every_page`, `test_collection_key_finds_name_past_first_page`, `test_collection_key_create_if_missing_finds_page_two_and_creates_nothing`, `test_collection_key_refuses_ambiguous_name`, `test_collection_tree_root_and_children_past_first_page` in [[tests.torchcell.literature.test_zotero]]; each fails on the previous code.
