---
id: zlbv953cawt8jadvtehi0n9
title: Test_zotero
desc: ''
updated: 1790562770705
created: 1790562770705
---

## 2026.09.27 - The Zotero read paths on a read-only fake client

Twenty-eight tests with the pyzotero `Zotero` client replaced by `tests/torchcell/literature/_fake_zotero.py`, a read-only stand-in with no write methods, so any write attempt fails rather than imitating success; `random.uniform` and `dotenv.load_dotenv` are stubbed for `_smoke_test`. Module coverage 98%. Deliberately not tested: `collection_key(create_if_missing=True)` (zotero.py lines 236 to 239), which writes to Zotero. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.10.01 - Paging and ambiguous names (issue #563)

`_fake_zotero.FakeZot` now pages like pyzotero 1.13: at most 100 rows per read, a `next` link in `links`, `follow()` for the next page, and `everything()` following it. Five tests added against a 150-collection library whose target sits at position 120: the full listing in API order with exactly one `follow`, `collection_key` resolving `TCROOT01`, `create_if_missing=True` returning `TCROOT01` with only read calls recorded, an exact ambiguity refusal naming `M1 under torchcell/torchcell-topics; M2 under torchcell/notes-tex; M3 at the top level`, and `collection_tree` finding a root at 120 and its child at 149. `create_if_missing=True` is now exercised only on a name that exists; the create branch itself stays untested because it writes.
