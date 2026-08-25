---
id: gmce5xfibw71b0svd2mu6pl
title: Zotero Capture Gaps
desc: ''
updated: 1787671516472
created: 1787671516472
---

## 2026.08.23 - Zotero items blocking capture into the OCR mirror

Open backlog. Each entry needs a fix in Zotero or in `torchcell/literature/sync.py` before the
nightly capture can reach it. Sync engine and mirror layout are described in
[[paper.literature-ocr-ingestion]].

Audit of the 602 wanted citation keys (torchcell group 101 UNION personal `torchcell`
collection tree 569) against `$DATA_ROOT/torchcell-library/`. Result: 577 fully OCR'd,
1 present without `paper.md`, 24 absent.

`lit_sync.py` captures an item only when it has **both a DOI and a PDF attachment**
(`torchcell/literature/sync.py:136`). Everything below fails that gate, so no amount of
re-running the nightly job will pick these up. Each needs a fix in Zotero.

### Issue 1: DOI present, no PDF attachment (14)

Attach the PDF to the Zotero item and the next sync captures + OCRs it.

| citation key | DOI | collection path |
|---|---|---|
| alcantarHighthroughputSyntheticBiology2024 | 10.1016/j.molcel.2024.05.025 | torchcell, torchcell/torchcell-queue/torchcell-in |
| angelerBiologicalSystemsSymphonies2023 | 10.1002/bies.202300113 | torchcell, torchcell/torchcell-queue/torchcell-in |
| chaiReviewComputationalApproaches2014 | 10.1016/j.compbiomed.2014.02.011 | torchcell, torchcell/torchcell-topics/gene-regulatory-network |
| fornasieroGeneralizingDefinitionProtein2023 | 10.1002/bies.202300059 | torchcell, torchcell/torchcell-queue/torchcell-in |
| gombertMathematicalModellingMetabolism2000 | 10.1016/S0958-1669(00)00079-3 | torchcell |
| hollmannTabPFNTransformerThat2023 | 10.48550/arXiv.2207.01848 | torchcell/torchcell-queue/torchcell-in |
| huangUniGNNUnifiedFramework2021 | 10.48550/arXiv.2105.00956 | torchcell |
| kumarProductionLevulinicAcid2020 | 10.1016/j.matpr.2020.04.749 | torchcell, torchcell/torchcell-queue/torchcell-in, torchcell/torchcell-topics/production-targets |
| petriPetriNet2008 | 10.4249/scholarpedia.6477 | torchcell |
| porterFluidProteinFold2023 | 10.1002/bies.202300057 | torchcell, torchcell/torchcell-queue/torchcell-in |
| shiMicroRNAsPlayRegulatory2023 | 10.1002/bies.202200187 | torchcell, torchcell/torchcell-queue/torchcell-in |
| singhGmx_qkAutomatedProtein2023 | 10.1021/acs.jcim.3c00341 | torchcell |
| tangGraphGPTGraphInstruction2024 | 10.1145/3626772.3657775 | torchcell, torchcell/torchcell-queue/torchcell-in |
| yeCurrentStatusTrends2023 | 10.1002/bies.202200242 | torchcell, torchcell/torchcell-queue/torchcell-in |
  
Nine of these sit in `torchcell-queue/torchcell-in`, so the inbox is the main source of
the backlog. Six of the fourteen are BioEssays reviews.

### Issue 2: no DOI and no PDF attachment (5)

These need a DOI added as well as a PDF. Two are `webpage` items, which may never be
capturable and might belong out of the mirror's scope entirely.

| citation key | itemType | collection path |
|---|---|---|
| GenomicFactorsShape | journalArticle | torchcell |
| GraphAttentionNetworks | webpage | torchcell |
| alamSelfinhibitoryNatureMetabolic2017 | journalArticle | torchcell |
| caoRelationalMultitaskLearning2023 | journalArticle | torchcell |
| fuchsFabianFuchs | webpage | torchcell/torchcell-topics/deep-set |

`GenomicFactorsShape`, `GraphAttentionNetworks`, and `fuchsFabianFuchs` also carry
placeholder citation keys rather than the usual author-title-year form, which suggests
the items were never filled in from a real record.

### Issue 3: PDFs present, no DOI (1)

| citation key | itemType | pdfs | collection path |
|---|---|---|---|
| volkMultiplexedCRISPRiPerturbseq2026 | report | 11 | torchcell/notes-tex, torchcell/notes-tex/microbe-perturb-seq |

Our own report, so there is no DOI to assign and it will never satisfy a DOI-keyed
capture. If we want it mirrored, capture has to key off the attachment instead of the
DOI, since `capture_by_doi` is the only path `lit_sync` uses.

### Issue 4: mirrored but never OCR'd (1)

`ohyaHighdimensionalLargescalePhenotyping2005a` has a directory holding
`annotations.json`, `annotations.md`, `data/`, and `manifest.json`, but no `paper.pdf`
and no `paper.md`. Because `_is_mirrored()` tests only for `manifest.json`, the sync
counts it as done and will never revisit it. MinerU has not run on this paper.

### Issue 5: group-side absent (2)

- `ozaydinCarotenoidbasedPhenotypicScreen2013` was reported captured on the 2026.08.23
  03:31 run, but the directory written was `zaydnCarotenoidBasedPhenotypicScreenYeast2013`,
  which does not match the Zotero key. `_is_mirrored()` will miss it again tonight, so it
  re-downloads and re-OCRs on every pass.
- `chaoPredictingDynamicExpression2025a` absent. Attachment status not checked, it was
  outside the personal-tree pass.

### Not an issue, just unwired (RESOLVED 2026.08.23)

Two personal-tree papers were fully capturable and missing only because
`lit_sync.py` never read the personal library:

- `macoskoHighlyParallelGenomewide2015`, 10.1016/j.cell.2015.05.002, 1 pdf
- `rosenbergSinglecellProfilingDeveloping2018`, 10.1126/science.aam8999, 1 pdf

Both in `torchcell/torchcell-topics/microbe-perturb-seq`. Both are now captured and
OCR'd, on branch `feat/lit-sync-personal-tree` (commit c3a555ca), which walks the
personal tree alongside the group collections. The endpoint reports 583 keys, up
from 581. They are also the 2 entries `lit_bib.py --dry-run` wants to add to
`bib.bib`, which is still pending since that script has no cron.

The nightly cron invokes `scripts/lit_sync.py` from the primary checkout, so it keeps
the old group-only behavior until that branch lands on `main`.

### How this was derived

Ad hoc audit scripts in the session scratchpad, not committed. They walk the personal
tree with `_collection_tree(user, $ZOTERO_USER_ROOT_COLLECTION)` and
`citable_citation_keys`, the same functions `lit_bib.py` uses, so keys match the mirror
directory naming by construction, then diff against `library_root($DATA_ROOT)` and query
`pdf_attachments` per item. If this list is worth keeping, the check should move into a
committed script, most naturally a `--scope personal` mode on `plan_collection_sync`.
