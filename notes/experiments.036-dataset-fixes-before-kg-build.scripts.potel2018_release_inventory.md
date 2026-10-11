---
id: 4c7b2w5cpy8mk8546wqj0o3
title: Potel2018_release_inventory
desc: ''
updated: 1791694634970
created: 1791694634970
---

## 2026.10.10 - Potel 2018 release inventory, raw-mirror deposit and gate verdict

Candidate row "Potel 2018 phosphoproteome" (*E. coli*, "Proteome"): Potel CM, Lin MH, Heck AJR, Lemeer S. Widespread bacterial protein histidine phosphorylation revealed by mass spectrometry-based proteomics. Nature Methods 2018;15:187-190, doi:10.1038/nmeth.4580, PMID 29377012. ProteomeXchange PXD008369 is hosted by jPOST as JPST000369; PRIDE answers "The project accession is not in the database".

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/potel2018_release_inventory.py`. Results: `experiments/036-dataset-fixes-before-kg-build/results/potel2018_release_inventory.json`. Verdict: `database/candidates/potelWidespreadBacterialProtein2018.json`. Issues: #921 (schema), #922 (walled article, by hand), #910 (W3110 route, comment), #923 (an off-schema raw manifest that crashes the gate's DOI lookup).

### Gate verdict

`candidate-gate gate --row "Potel 2018 phosphoproteome" --write`, outcome **`blocked:ingest`**. The stop rule leaves G3 to G5 unmeasured in the stored verdict; the third column is what each gate returns when called directly on this deposit (not stored).

| gate | stored | evaluated directly |
|---|---|---|
| G1 primary | pass (`primary`, no aggregation marker) | same |
| G2 host route | **blocked:ingest**: W3110 -> `ecoli_K12_W3110_ASM1024v1` resolves but no genome class reads it (#910) | same |
| G3 deposit | unmeasured | pass: 11 files in raw, all intact; its "1,052,331 worksheet rows" is 1,048,356 declared rows of Supplementary Table 1, which has 35 rows with content |
| G4 overlap | unmeasured | pass: no key held by another module, 0 mentions of the DOI or PXD008369 |
| G5 schema | unmeasured | gap #921: no `Phenotype` subclass holds a site-level phosphorylation intensity |

Before the row was corrected, the first run said G2 pass (MG1655). That token came from a search snippet; no deposited byte names MG1655 (`strain_tokens_in_deposited_texts`: 0 in the SI, 0 in the reporting summary, 0 in the dissertation, which names W3110 three times). The row's `why` and `accession` now state the measured host, counts and issues; its numeric fields are unchanged so nothing re-ranks, as in the Li 2014 correction.

### Mirror route

Not in Zotero and not in `torchcell-library/`; not in PMC or Europe PMC (no PMCID, `isOpenAccess: N`). nature.com serves the subscription landing page for both the article and its PDF URL. The release is deposited in `$DATA_ROOT/torchcell-raw/potelWidespreadBacterialProtein2018/` (re-run with `--deposit`; idempotent):

| file | role | retrieval | sha256 (prefix) |
|---|---|---|---|
| `data/..._MOESM4_ESM.xlsx` (Supplementary Table 2) | raw_data | `retrieve.springer_esm` | `f69ca61a` |
| `si/..._MOESM1_ESM.pdf` (Supplementary Text and Figures) | si_pdf | `retrieve.springer_esm` | `f4cc85a6` |
| `si/..._MOESM2_ESM.pdf` (Life Sciences Reporting Summary) | si_pdf | `retrieve.springer_esm` | `7ab7cabe` |
| `si/..._MOESM3/5/6/7_ESM.xlsx` (Supplementary Tables 1, 3, 5; Fig. 1 source data) | si_data | `retrieve.springer_esm` | `e6e4ef28`, `c2ebaef8`, `31a986e2`, `61ff4b38` |
| `paper/thesis/Potel.pdf` (dissertation, Utrecht handle 1874/370166) | paper_pdf | `retrieve.direct_url` (DSpace REST) | `4252703b` as held |
| `si/..._MOESM1/2_ESM.pdftotext.txt`, `paper/thesis/Potel.pdftotext.txt` | si_text, paper_text | derived, `pdftotext -layout` 21.01.0, input sha256 pinned | `4dee5c2a`, `9951a1ac`, `05a50657` |
| `data/..._MOESM4_ESM.legends.txt`, `si/..._MOESM6_ESM.legends.txt` | si_text | derived, `extract_sheet_legends` (openpyxl 3.1.5) | `a85a3a15`, `ea531578` |

The dissertation is the quote anchor for the Methods: its Chapter V is this paper ("Chapter V: Widespread bacterial protein histidine phosphorylation revealed by mass spectrometry-based proteomics"). The Utrecht repository wraps every download in a generated cover, so two downloads differed in 134 bytes while their text renderings were identical. The script therefore pins the TEXT (`05a50657...`), checks it on every run, and treats the deposited PDF bytes as canonical as held; a re-run of the retriever is expected to give different PDF bytes. The repository's copy of the article itself (handle 1874/362159) is the October 2017 submitted version with no Online Methods.

jPOST is FTP-only (`ftp://ftp.jpostdb.org/JPST000369/`), which `httpx` cannot fetch, so nothing from it is deposited; `--network` records its listing: 18 `.raw` files (17.4 GB), a 254 MB MaxQuant `txt` archive, a 13.8 GB `_all.zip`, nine `evidence.txt` files, `mqpar.xml` and the jPOST XML. No loader consumes any of them.

### Host, medium and replicate design

From the dissertation's Chapter V Methods (each quote checked to occur in Chapter V by the running page header; the growth paragraph recurs verbatim in Chapter VI):

- "E. coli strain W3110 was grown in M9 minimal medium, consisting of M9 salts", "supplemented with either additional 0.5% (w/v) glucose or 0.5% glycerol", "with vigorous shaking at 37°C. Cells were collected by centrifugation at mid-log phase (OD600= 0.6/0.5 for glucose / glycerol-fed cells) and stationary phase (OD600= 1.2/1.0 for glucose/ glycerol-fed cells) (Figure S12)".
- "Fe3+-IMAC was performed in technical triplicates for each biological groups".
- Search: "reviewed E. coli K12 database (Uniprot, March 2016)", MaxQuant 1.5.5.1.

**The release contradicts itself on replicates.** SI Figures 3 and 6: "The same UV profiles were observed for the n=12 biological replicates." The reporting summary: "we made technical triplicates (after digestion, before enrichment) of each" condition and "Results were reproduced under 4 biological conditions, each consisting of technical" triplicates. Two sources against one say four biological conditions, each in technical triplicate; the decision is flagged in #921 for the owner (the #800 pattern), not taken here.

**The dissertation is not byte-for-byte the published release.** Its t-test used "a false discovery rate (FDR) of less than 0.05"; the published Supplementary Table 5 legend says "(t-test, with FDR ≤ 0.02 and S0=0.1)". The published Online Methods are walled (#922), so the strain, medium, temperature and harvest values above are sourced from the reprint and are not yet confirmed against the published text.

### What was released

Supplementary Table 2 is the only per-record artifact: 2,129 sites x 28 columns, no duplicate (accession, residue, position).

| measure | value |
|---|---|
| residues | S 1,220, T 501, Y 162, H 246 |
| localization probability >= 0.75 | S 964, T 340, Y 143, H 135 (min 0.206) |
| pHis sites / distinct accession strings | 246 / 173, matching Chapter V's "246 pHis sites on 173 histidine phosphoproteins" and "135 ... Class I pHis sites (localization confidence ≥ 75%)" |
| distinct accession strings | 781; 42 rows name more than one gene and position, 10 of them more than one accession |
| per-sample intensity columns | 12 = (glucose, glycerol) x (exp, stat) x 3; 25,548 cells, 12,095 are 0, none negative |
| nonzero per sample | 897 (glucose_exp3) to 1,319 (glycerol_stat3); nonzero range 2.7e5 to 3.8e10, median 1.2e7 |
| sites 0 in all twelve | 245 (244 also have a total of 0) |
| total `Intensity` vs sum of twelve | equal 278, larger 1,130, smaller 477: not derivable from the twelve |

The twelve columns map one-to-one onto the twelve `OR13_..._Ecoli_<carbon>_<phase>_2mg_IMAC_R<n>.raw` files in jPOST; the six OR10/OR11 runs are the protocol comparison of the Fig. 1 source data (phosphopeptides per run: optimized 1,149/1,017/1,263; standard plus precipitation 333/442/331; standard 137/75/69).

Supplementary Table 5 (1,469 sites, glycerol exponential vs stationary) is derived from Table 2: all 6,513 joinable cells with a nonzero Table 2 intensity equal its log2 (max difference 5.1e-5), and 2,103 cells are imputations where Table 2 is 0. Its only new content is the `+` flag on 594 sites. 33 of its keys are not Table 2 keys because Table 2 lists those sites on rows with more than one position and Table 5 gives one (`unmatched_keys_by_table_2_row`): 9 on multi-accession rows and 24 on one-accession rows naming two genes, 22 of them `P0CE47` (`tufA;tufB`, positions such as `313;313`). Supplementary Tables 1 and 3 are synthetic-peptide standards (35 rows with content) and a comparison with an antibody study (250 sites); neither is a measurement of this study's cells.

### Identifier route

Gene symbols of the 2,087 single-gene rows (772 distinct) through `reconcile_locus_tags`: 756 resolve on `ecoli_K12_MG1655_ASM584v2` (0.9793) and 754 on `ecoli_K12_BW25113_ASM75055v1` (0.9767). The MG1655 misses: `bioD1`, `ccdB`, `corC`, `rmlA1`, `sopA`, `traD`, `traG`, `traI`, `traJ`, `traM`, `traP`, `traQ`, `traY`, `trbI`, `trbJ`, `yuaY`. Hypothesis, not checked: the `tra`/`trb`/`sopA`/`ccdB` symbols are F-plasmid genes that the UniProt K-12 reviewed set carries, which W3110 (F-) should not express. Neither K-12 set is the host's; the W3110 set is not readable (#910).

### Schema fit (G5)

21 `Phenotype` subclasses; none has a field naming a site, residue or modification (`schema_fit`). `ProteinAbundancePhenotype` holds a protein abundance keyed by protein. Proposed leaf and its three open decisions (replicate field, shared-peptide key grammar, waiting for #910): #921. Two yeast rows also name phosphosite abundance (Leutert 2023, Grossbach 2022); their shapes are not measured.

### Duplication by content (G4)

Read-only over the six served proteome dev stores. All are `bacterial_protein_abundance`, keyed by protein, with **0 site-shaped keys**; verdict: independent. Shared proteins (MG1655 b-numbers or BW25113 tags, by each store's namespace): Schmidt 2016 708 of 2,329, SRM sets 32 of 41 and 26 of 31, Ishii 2007 49 of 59, Brunk 2016 40 of 65, Mori 2021 691 of 2,073. Every store's build manifest carries `torchcell_dirty: true`, so no values were compared; the gate's G4-value refuses a dirty store by design.

### Stored records

0. No loader, adapter, KG entry, dev store or schema change, so there is no L0-L4 table and no schema-impact verdict.
