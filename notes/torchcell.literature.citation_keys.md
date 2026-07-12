---
id: g2v0ocazd65zvecawply5it
title: Citation_keys
desc: ''
updated: 1783563713786
created: 1783563713786
---

## 2026.07.08 - A deterministic key so every paper's artifact directory is reproducible

This module exists to give each paper one stable, reproducible identity string that names its artifact directory (`torchcell-library/<citation_key>/`). The key is a pure function of the metadata the Zotero web API returns (authors, date, title), so it reproduces byte-identically on every run and host -- the property the whole mirror relies on to address each paper's directory deterministically. It mirrors Better BibTeX's default format so our generated keys match what the user already sees in Zotero.

- `generate_citation_key`: `{firstauthorlastname}{TitleWords}{YYYY}`, matching Better BibTeX's `[auth:lower][Title:skipwords:nopunct:fold:condense=''][year]`.
- Transliterates non-ASCII surnames/title words via `unidecode` (so `Müller` -> `muller`, not the lossy `mller`) and guarantees a letter-initial, `[A-Za-z0-9]`-only key via the `unknown` fallback.
- Used by [[torchcell.literature.zotero]] only as the last resort, after an item's native `citationKey` / `extra` field, so a stored key always wins for directory-naming fidelity.
