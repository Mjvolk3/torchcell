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

## 2026.07.05 - Follow-up Fixes + Cassette Provenance + Perturbation Design

### Done + committed (branch `fix/ws8-ozaydin-cachera-followups`)

- **PubMed ID.** `Publication` now carries `pubmed_id="37572348"` + `pubmed_url`.
  PMID<->DOI (`10.1093/nar/gkad656`) confirmed via NCBI E-utilities (title "CRI-SPA:
  a high-throughput method...", Nucleic Acids Research 2023). Not guessed.
- Cachera has no `qc_flags` field (that was Ozaydin's `VisualScorePhenotype`); no
  change to `MetabolitePhenotype` here. The canonical LMDB still has the OLD
  `pubmed_id=None` publication and should be rebuilt in place when this lands (needs
  the injected `SCerevisiaeGenome` to map common gene names).

### Btx-cassette definition -- sourced from OUR mirror (provenance-first)

From `torchcell-library/cacheraCRISPAHighthroughputMethod2023/paper.md` (MinerU OCR):

- **Btx-cassette** (paper calls it a "five-gene" cassette; the abstract's "four
  genes that enable betaxanthin production" = the 4 non-marker genes):
  - Heterologous plant genes: **CYP76AD1** (cytochrome P450) + **DOD** (DOPA
    4,5-dioxygenase) -- the two genes strictly required for betaxanthin (ref 25).
  - Native yeast genes, feedback-resistant mutant alleles: **ARO4^K229L**
    (DAHP synthase, YBR249C) + **ARO7^G141S** (chorismate mutase, YPR060C) -- relieve
    shikimate-pathway negative feedback.
  - Selectable marker: **natMX** (not a betaxanthin gene).
- **Localization = chromosomal_integration** at expression site **XII-5** (BY-Btx
  made by NotI-excised insert of **pBTX2** into BY4741; CD-Btx from pBTX1 targets
  XII-5). Contrast Ozaydin's episomal 2micron plasmid.
- **Full sequence (#4, external):** plasmids **pBTX1 / pBTX2** in Supplementary
  Table S2 and the CRI-SPA repo **github.com/pc2912/CRI-SPA_repo** (also the source
  of the ingested `GA1_2_4_6.csv`). Our mirror has the composition + locus; the
  plasmid sequences are the remaining external dig.

### Heterologous-cassette perturbation -- DESIGN SIGN-OFF PENDING

Same schema gap + three-option decision as documented in
`[[torchcell.datasets.scerevisiae.ozaydin2013]]` (recommendation = per-gene
`GeneAdditionPerturbation` with a `localization` field). Cachera exercises the
`chromosomal_integration` localization + a `locus="XII-5"` value, and the native
feedback-resistant alleles (ARO4^K229L / ARO7^G141S) are the concrete case for the
"native extras" sub-decision (new addition type flagged native vs reuse
`AllelePerturbation`).

### 2026.07.05 - Plasmid availability (answer: YES, GenBank maps provided)

Verified by web research (do they provide the plasmid so it can be downloaded?):

- **Data Availability (paper, verbatim):** "All plasmid maps are available for
  download as GenBank files." So pBTX1 / pBTX2 (+ the CRI-SPA vectors) DO exist as
  downloadable GenBank maps -- the raw artifact the "plasmid-seq + feature
  annotation -> collapse to GeneAddition" path needs.
- **Location:** NOT in the GitHub repo `github.com/pc2912/CRI-SPA_repo` (that has
  only data CSVs + analysis notebooks, no `.gb/.fasta/.dna`). The maps are in the
  **OUP supplementary archive**:
  `https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/nar/51/17/10.1093_nar_gkad656/1/gkad656_supplemental_files.zip`
  (contains Suppl. Tables S1 strains / S2 plasmids / S3 primers + figures/methods).
- **Retrieval method = manual_browser (un-scriptable).** The silverchair CDN URL is
  a CloudFront SIGNED url: bare `curl` returns HTTP 403 `MissingKey` (needs a
  Key-Pair-Id token from an academic.oup.com browser session). So this is the
  nature.com-class case in our provenance notes: manual-once download -> deposit to
  `torchcell-library/cacheraCRISPAHighthroughputMethod2023/si/` + sha256 -> then
  reproducible via our mirror. No Addgene ID / GenBank accession is given for pBTX1/2.
- Upstream heterologous parts ARE on Addgene (e.g. `pL0-BvCYP76AD1` #162529, a MoClo
  L0 part; DOD from the DeLoache 2015 betaxanthin biosensor lineage), useful as
  cross-checks but they are PARTS, not the assembled pBTX1/pBTX2.
