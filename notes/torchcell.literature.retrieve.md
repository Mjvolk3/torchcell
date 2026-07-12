---
id: yguljxozeafhzsmcmrrhsw3
title: Retrieve
desc: ''
updated: 1783563749063
created: 1783563749063
---

## 2026.07.08 - Retrieval as versioned source code, not a shell string

This module exists so that HOW an artifact was fetched is versioned, testable source code rather than an unreproducible one-off command. Each retriever is a named function registered by dotted path; a `RetrievalRecord` in [[torchcell.literature.manifest]] stores that path plus its params, and [[torchcell.literature.provenance]] resolves and re-runs it. This is what turns "we downloaded it from a URL once" into a recipe that a future rebuild can execute and sha256-verify.

- Encodes the verified retrieval reality: Springer ESM (`static-content.springer.com`) and the PMC OA API are directly scriptable and get real retrievers; nature.com (auth redirect) and PMC file downloads (JS proof-of-work) are NOT, so they route through Zotero (or the future Radiant VM endpoint).
- No silent partials: non-2xx raises, and `pmc_oa_api` raises when an id is outside the redistributable OA subset so the caller falls back to the mirror instead of capturing an interstitial page.
- The `RETRIEVERS` registry is the single source of truth mapping a record's key to the callable.
