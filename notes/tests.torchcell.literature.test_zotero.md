---
id: zlbv953cawt8jadvtehi0n9
title: Test_zotero
desc: ''
updated: 1790562770705
created: 1790562770705
---

## 2026.09.27 - The Zotero read paths on a read-only fake client

Twenty-eight tests with the pyzotero `Zotero` client replaced by `tests/torchcell/literature/_fake_zotero.py`, a read-only stand-in with no write methods, so any write attempt fails rather than imitating success; `random.uniform` and `dotenv.load_dotenv` are stubbed for `_smoke_test`. Module coverage 98%. Deliberately not tested: `collection_key(create_if_missing=True)` (zotero.py lines 236 to 239), which writes to Zotero. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
