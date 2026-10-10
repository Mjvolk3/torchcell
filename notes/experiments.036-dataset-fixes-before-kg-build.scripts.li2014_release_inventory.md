---
id: 386xiztqxgr7kqiuqfc7ty6
title: Li2014_release_inventory
desc: ''
updated: 1791613925349
created: 1791613925349
---

## 2026.10.10 - Li 2014 release inventory, raw-mirror deposit and schema refusal

Candidate row 61 (*E. coli*, "Translation / turnover"): Li GW, Burkhardt D, Gross C, Weissman JS. Quantifying absolute protein synthesis rates reveals principles underlying allocation of cellular resources. Cell 2014;157:624-635, doi:10.1016/j.cell.2014.02.033, PMC4006352 (NIH author manuscript), PII S0092867414002323.

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/li2014_release_inventory.py`. Results: `experiments/036-dataset-fixes-before-kg-build/results/li2014_release_inventory.json`. Schema finding: issue #857.

### Mirror route

The paper is not in the literature mirror and nothing was added to Zotero. The release is deposited in the raw mirror `$DATA_ROOT/torchcell-raw/liQuantifyingAbsoluteProtein2014/` with a `manifest.json` (re-run with `--deposit`; idempotent by sha256):

| file | role | retrieval | sha256 (prefix) |
|---|---|---|---|
| `data/...-mmc1.xlsx` (Table S1) | raw_data | `retrieve.elsevier_mmc`, direct_url | `f493023e` |
| `si/...-mmc2.xlsx` .. `mmc5.xlsx` (Tables S2-S5) | si_data | `retrieve.elsevier_mmc` | `ffd05c21`, `28a68899`, `877ba750`, `47b501c9` |
| `si/...-mmc6.pdf` (published article + Extended Experimental Procedures) | si_pdf | `retrieve.elsevier_mmc` | `a96e712d` |
| `si/...-mmc6.pdftotext.txt` | si_text | derived, `pdftotext -layout` 21.01.0, input sha256 pinned | in manifest |
| `paper/PMC4006352.1.txt`, `.xml` | paper_text | `retrieve.pmc_cloud_object` | `b3cc01fb`, `03ac4643` |

The PMC bucket prefix `PMC4006352.1/` holds exactly the `.json`, `.txt` and `.xml`; no PDF and no supplement, so the SI comes from the Elsevier CDN.

### What was released

- Strain, from the SI text: "E. coli K-12 strain MG1655 was used for this study." The candidate row's "Strain designation is unconfirmed" is therefore wrong; MG1655 is in the genomes tier.
- Media: three, not the row's two. Table S1's columns are `MOPS complete`, `MOPS minimal`, `MOPS complete without methionine`; the SI: "All cultures were based on MOPS media with 0.2% glucose (Teknova), with either", and doubling times "21.5 ± 0.4 min in fully supplemented MOPS media, 26.5 ± 1.1 min in the methionine" dropout, 56.3 min in minimal.
- Statistic: absolute synthesis rate, "where ki has the unit of molecules per generation." No degradation rate, half-life, interval or SE is released for any protein.
- Table S1: 4,095 gene rows, 4,095 unique names; 8 merged-pair rows (`tufA+tufB`, `gadA+gadB`, ...) and the frameshift split `dnaXgamma` / `dnaXtau`.

| medium | plain integer cells | `[n]` bracketed cells | plain range | bracketed range |
|---|---|---|---|---|
| MOPS complete | 3,041 | 1,054 | 2 to 1,191,641 | 0 to 86 |
| MOPS minimal | 3,362 | 733 | 0 to 619,492 | 0 to 36 |
| MOPS complete without methionine | 2,241 | 1,854 | 0 to 669,430 | 0 to 1,161 |

The bracket is defined nowhere in the paper or SI text. It is back-solved from a count: the main text says "we evaluated 3,041 genes which account for >96% of total proteins synthesized" and "All of these genes have >128 ribosome footprint fragments sequenced", and exactly 3,041 MOPS-complete cells are plain. It is a read-count gate, not a value threshold: 683 plain MOPS-complete values are at or below the largest bracketed value (86), and the methionine-dropout column, with the most bracketed cells, has bracketed values up to 1,161.

- Replicates: GEO GSE53767 has four samples, all MG1655: one ribosome-profiling sample in rich defined medium, two in minimal medium, one mRNA-seq in rich medium, each released only as pooled `.wig` tracks. No GEO sample is for the methionine-dropout medium. The paper's "error of less than 1.3-fold across biological replicates" has no per-gene replicate column behind it.
- Tables S2, S3 and S5 are curated literature tables (copy numbers from 21 labs, complex stoichiometries, TF properties), not measurements of this study. Sheet `TableS4` (mmc4) carries rich-medium mRNA RPKM and translation efficiency; the SI text calls it "Table S5", a numbering slip in the source.

### Identifier route

`reconcile_locus_tags` on `ecoli_K12_MG1655_ASM584v2`: 4,059 of 4,087 single names resolve to one b-number (0.9931); 3,751 through the gene-symbol layer and 317 through a synonym. Not resolved: 19 retired (the `ins*` IS elements, `dnaXgamma`/`dnaXtau`, ...), 5 ambiguous (`ade`, `gtrA`, `rffT`, `spr`, `ygaD`), 4 kept on collision (`ecpD`, `mscM`, `yagW`, `yjeP`).

### Schema fit: nothing loads

`ProteinTurnoverPhenotype` requires a non-empty `degradation_rate` and accepts `synthesis_rate` only on keys that also carry a degradation rate; the script constructs a synthesis-only phenotype and records the validator's answer, `degradation_rate cannot be empty`. Inventing a degradation rate (zero, or the dilution rate) would be fabrication. `ProteinAbundancePhenotype` is refused too: the conversion is the authors' assumption ("For stable proteins, ki is also the copy number"), and the main text says the rate is "an upper bound for the protein levels for the small subset of proteins that are rapidly degraded". **Stored records: 0.** No loader, adapter, KG entry or dev store is added; no schema change, so there is no schema-impact verdict to record.

Keys a synthesis-rate record would carry once a class exists (#857 proposes an additive `ProteinSynthesisRatePhenotype`):

| medium | loadable keys | bracketed (below 128 footprints) | merged pair | name not one b-number |
|---|---|---|---|---|
| MOPS complete | 3,025 | 1,054 | 2 | 14 |
| MOPS minimal | 3,346 | 733 | 3 | 13 |
| MOPS complete without methionine | 2,226 | 1,854 | 1 | 14 |

### Duplication against Gupta 2024

Gupta 2024's dev store (`protein_turnover_gupta2024`, 13 records, 3,260 b-numbers) shares 3,202 b-numbers with Li's 4,059. Against Gupta's one batch wild-type record (MOPS glucose minimal, 42 min, 2,553 keys), on the plain Li cells:

| Li medium | n shared | Spearman (synthesis vs degradation rate) | Pearson (log10 synthesis vs degradation rate) |
|---|---|---|---|
| MOPS complete | 2,395 | -0.096 | -0.042 |
| MOPS minimal | 2,484 | -0.116 | -0.036 |
| MOPS complete without methionine | 1,972 | -0.055 | -0.010 |

Verdict: independent. The two releases measure different quantities of different strains (MG1655 vs NCM3722), and the rank agreement is near zero.
