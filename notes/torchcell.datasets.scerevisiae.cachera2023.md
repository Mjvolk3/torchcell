---
id: sfk463pvqf0q3yk8urthewq
title: Cachera2023
desc: ''
updated: 1783223952046
created: 1783223952046
---

## 2026.07.05 - WS8 build: Cachera CRI-SPA betaxanthin

Roadmap [[plan.schematization-ingestion-roadmap.2026.06.23]] WS8. The abstract's
betaxanthin case. First `MetabolitePhenotype` dataset (WS4).

### What the data is (from OCR'd paper.md)

Cachera 2023 (NAR, `10.1093/nar/gkad656`) is a METHOD paper (CRI-SPA). It transfers
**four betaxanthin-biosynthesis genes** into each strain of the ~4800-strain YKO
collection and reads out per-colony **betaxanthin** (a yellow, naturally fluorescent
plant metabolite) by image analysis. The "CRI-SPA score" is a corrected/normalized
colony fluorescence intensity -- a QUANTITATIVE proxy for betaxanthin level. Because it
is population-centered it can be negative. So this is a `MetabolitePhenotype`
(continuous), NOT the ordinal VisualScore of Ozaydin -- the data shape picked the schema.

### Source (data is on GitHub, not the PDF)

The paper's **Data Availability** points to the CRI-SPA GitHub repo. Zotero holds only
the PDF (OCR'd → paper.md). The genome-wide per-gene data lives at
`github.com/pc2912/CRI-SPA_repo` (the OCR dropped the `_repo` suffix). We ingest
`GA1_2_4_6.csv` (gene-level corrected+filtered, replicates 1/2/4/6, 4788 rows) using
the 24 h `corrected_mean_intensity` (mean/std/count) → betaxanthin level + SE + n. All
mirrored under `torchcell-library/cacheraCRISPAHighthroughputMethod2023/` with a
sha256 `manifest.json` (paper.pdf, paper.md, si/GA1_2_4_6.csv).

### Dataset (`BetaxanthinCachera2023Dataset`)

Source gene names are COMMON (AAC1), so an injected `SCerevisiaeGenome` resolves them
to systematic ORFs (same pattern as Sameith). **4735 records** (one per ORF). Excluded,
all logged: 28 control/NaN rows (incl. 'WT'), 5 unresolved common names, 20
common-name→same-ORF collisions (deduped, keep first). L0-L4 all PASS
(`torchcell/verification/metabolite.py`); L4 gene overlap with the Ohya deletion
collection = **0.982**.

### Schema (WS4) `MetabolitePhenotype`

`metabolite_level: dict[metabolite_id -> float]` (Yeast9 `s_NNNN` where native, or a
product name for heterologous betaxanthin), `metabolite_level_se`, `n_replicates`
(per metabolite), `measurement_type` (what the number IS -- here
`cri_spa_corrected_fluorescence_intensity_24h`, so assays are never silently mixed),
optional `target_metabolite_ids` for Yeast9/CBM linkage (None -- deferred).
