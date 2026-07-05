---
id: l8393oimoopk2ab7dmwakw7
title: Ozaydin2013
desc: ''
updated: 1783210882206
created: 1783210882206
---

## 2026.07.04 - WS7 build: Ozaydin beta-carotene visual screen

Roadmap [[plan.schematization-ingestion-roadmap.2026.06.23]] WS7. The abstract's
β-carotene case. Maps to the new `VisualScorePhenotype` (WS4).

### Source + provenance (full extraction machinery exercised)

- Paper + SI found in Zotero (group `6582362`) by DOI `10.1016/j.ymben.2012.07.010`.
  Zotero holds the **PDF only**; the data SI is NOT in Zotero.
- **SI xlsx** (`1-s2.0-S109671761200081X-mmc1.xlsx`, 367 KB, sha256 `4818726e…`)
  fetched from the **Elsevier ESM** direct URL (`ars.els-cdn.com/...mmc1.xlsx`).
- **paper.pdf** pulled from Zotero via `torchcell.literature.ZoteroLibrary`; **MinerU
  OCR** (`swanki-mineru` env) → `paper.md` to capture the strain construction.
- All three in `$DATA_ROOT/torchcell-library/ozaydinCarotenoidbasedPhenotypicScreen2013a/`
  with a `manifest.json` (role + sha256 + retrieval per file).

### The screen (from OCR'd paper.md — corrects the iBF note)

Base strain **BY4741** (Open Biosystems YKO collection) transformed with plasmid
**YB/I/BTS1** = `YEplac195 TDH3p-crtYB-CYC1t; TDH3p-crtI-CYC1t; TDH3p-BTS1-CYC1t`
(Verwaal 2007): crtYB + crtI from *X. dendrorhous* + an extra copy of the **native
GGPP synthase BTS1** (NOT crtE — the iBF note said CrtE; the OCR shows BTS1). Colony
color on a **-5..+5** scale (WT carrying the plasmid = 0) is a visual proxy for
carotenoid (β-carotene) accumulation. Scored on SC-URA agar, 30 °C. This heterologous
cassette is the background a future CBM adds on top of Yeast9; it is captured in
paper.md so the CBM is buildable later (per user: capture-complete, modeling-light).

### Dataset (`CarotenoidOzaydin2013Dataset`)

Per-ORF aggregation of SI Sheet 1: `visual_score = max` numeric color across replicate
plates, `visual_score_min = min`, `n_replicates = count`, QC flags parsed from the
Comment free-text. Sheet 2 (TOP200) merged for gene name/function/category + `in_top200`.
Base strain (**BY4741 / BY4730**) captured per record. `target_metabolite_id` left
`None` (Yeast9 mapping deferred).

**4474 records** (ORFs with a numeric color). Excluded, never silently: text-only rows
(`pet`/`tiny`/`_`) and **malformed ORF names** (e.g. `YLR287-A` missing the W/C — NOT
guess-repaired). Score distribution matches the paper (0 = 2020 majority, spanning
-5..+5). L0-L4 all PASS (`torchcell/verification/visual_score.py`); L4 gene overlap
with the Ohya deletion collection = **0.946**.

### Schema note (WS4)

`VisualScorePhenotype` (schema.py): ordinal `visual_score` within a declared
`[score_scale_min, score_scale_max]`, `score_semantics`, `target_product`
(+ optional Yeast9 `target_metabolite_id`), `n_replicates`, `qc_flags`, `score_text`.
A defensive fix to `MicroarrayExpressionPhenotype.n_replicates` (raise a clean
ValueError on non-dict, not crash on `.items()`) was needed so pydantic union
resolution can skip that member when a scalar-`n_replicates` phenotype is the match.
