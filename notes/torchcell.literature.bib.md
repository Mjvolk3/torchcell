---
id: itydct12tnlmwh0jtg0giun
title: Bib
desc: ''
updated: 1786915849982
created: 1786915849982
---

## 2026.08.16 - Zotero-generated bibliography (replaces hand-editing)

`notes/assets/bib/bib.bib` is now **generated**, not hand-edited. This module pulls
BibTeX from the TorchCell Zotero group over the **Web API via pyzotero**
(`format="bibtex"`) and merges it into the two managed bibliographies. CLI wrapper:
[[scripts.lit_bib]].

### Why the Web API and not Better BibTeX

`paper/nature-biotech/zotero_export_bib.py` already exports the manuscript's
`references.bib` through the **Better BibTeX local endpoint** (`localhost:23119`).
That needs Zotero desktop + BBT *running*, so it only works on a machine with the app
open. This module uses the Web API instead, so it runs headless — GilaHyper, cron, or
the Mac — which is what "pull it with the zot library, never hand-edit" requires. The
two exporters coexist: BBT owns `references.bib` (manuscript), this owns `bib.bib`
(Dendron notes).

### The entry key IS the mirror key

The Web API's BibTeX export emits Zotero 7's native `citationKey`, which is exactly
what `_resolve_citation_key` uses to name mirror directories. Verified against the
`microbe-perturb-seq` collection: the key sets are identical in both directions
(`ours - theirs = theirs - ours = ∅`). So `@brettnerUltraHighthroughputMassively2024`
in a note and `<DATA_ROOT>/torchcell-library/brettnerUltraHighthroughputMassively2024/`
are the same string, with no mapping table.

### Three things the naive version would have got wrong

**1. The export is 46% attachment stubs.** A whole-library `items(format="bibtex")`
pull returns **185** entries, but only **102** are citable works — the other 83 are PDF
*attachments*, exported as `@misc{noauthor_notitle_nodate...}` with no author, title, or
date. `fetch_bibtex_entries` therefore intersects the export against
`citable_citation_keys` (a JSON pass filtered on `NON_CITABLE_ITEM_TYPES`) rather than
trusting it wholesale.

**2. The merge must be add-only, not replace.** `bib.bib` predates the Zotero group:
of its 485 keys, only **23** are in the group and **462** are not (older exports from a
personal library). A wholesale regeneration would delete those 462 and break every note
citing one. `merge_bib_entries` never drops a key, and `sync_bib_file` raises rather
than write a file that shrank.

**3. Letting Zotero overwrite shared entries is a silent regression.** The 23 shared
entries are **biblatex**-style locally (`date`, `journaltitle`, `shortjournal`,
`langid`) while the API export is **BibTeX**-style (`year`, `journal`, `language`).
Nothing citation-critical is lost, but `shortjournal` is what the Nature CSL uses for
abbreviated journal names. Rendering all 23 through pandoc + `nature.csl` measured the
cost of each policy:

| policy | render vs current | note |
| --- | --- | --- |
| **add-only** (default) | **byte-identical** | new keys appended; existing untouched |
| field-wise (`update_existing=True`) | 13 lines differ | keeps `shortjournal`; author/URL differences remain |
| Zotero-wins wholesale | 33 lines differ | `Nat Methods` → `Nature Methods`, biorxiv URL lost |

Hence add-only is the default and `--update-existing` is opt-in. All 23 resolve under
every policy — the difference is cosmetic, not broken citations.

### Two latent bugs this surfaced

- **`notes/assets/bib/bib.bib` was unparseable by pandoc.** It died at line 8188 on
  `@online{yun$On$ConnectionsAre2020,` — a `$` in a citation key aborts the whole file,
  so *every* citation in it silently failed. The publish twin had been hand-fixed to
  `yunOnConnectionsAre2020`, which is why publishing worked and the assets copy did not.
  `read_bib_entries` now sanitizes invalid key characters (logged at WARNING). Nothing
  cited the key, so the rename is safe.
- **The two copies had drifted.** They differed by exactly that one key. Generating both
  from one pull converges them — they are now byte-identical (same md5).

### First run

485 → **563** entries in both files: 78 added, 23 unchanged (add-only), 462 preserved,
1 duplicate key collapsed. Cross-checked every `@key` cited anywhere in `notes/`:
**0 regressions**, 1 newly resolvable (`zhangCombiningMechanisticMachine2020`, cited in
[[data-factory]] but previously missing from the bib).

### Not wired to cron, deliberately

`bib.bib` is git-tracked, so a nightly regeneration would leave the primary checkout
dirty and fight the worktree discipline in `CLAUDE.md`. Run it by hand after
[[scripts.lit_sync]] captures new papers.
