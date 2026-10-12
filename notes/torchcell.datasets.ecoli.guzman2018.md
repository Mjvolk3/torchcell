---
id: mm6lw8quj5onaf8px7cdisd
title: Guzman2018
desc: ''
updated: 1791848349323
created: 1791848349323
---

## 2026.10.12 - Loaded from its own GEO release, the compendium as a cross-check

Guzmán et al. 2018 (BMC Syst Biol 12:143, doi:10.1186/s12918-018-0653-z, `guzmanReframingGeneEssentiality2018`), issue #964. `RnaseqGuzman2018Dataset`, slug `rnaseq_guzman2018`, 14 records, one per column of GEO `GSE117303_deseq2-tpm.csv.gz`.

**Raw mirror.** `$DATA_ROOT/torchcell-raw/guzmanReframingGeneEssentiality2018/`: the PMC plain text (`paper/PMC6296033.1.txt`, sha256 `392b5f3d...`, the quote source), Additional file 4 (`si/12918_2018_653_MOESM4_ESM.xlsx`, `97e587ad...`), the TPM matrix (`590b94e9...`) and the series matrix (`5a1c437a...`), each with its retrieval recorded. The paper is filed in Zotero `database/Escherichia-coli` (item `MBW5S5AB`).

**What is stored.** TPM verbatim under the release's own gene ids: 4,329 per library, 106 of them `_1` / `_2` split ids the annotation does not carry. `expression_count` is a typed gap: the counts the Methods describe were never deposited (the schema change of this PR makes the field value-or-gap). Genotypes: two wild-type libraries carry none; each deletion-strain library carries its Keio deletion and every Additional file 4 call for its DNA sample (`FPS <gene> A<n> F<n> I<n> R1`), 12 calls over six strains. The GEO sample record names each library `Clone` or `Population`, which is the call mode; frequencies are fractions ("Note: Mutation frequencies are reported as a fraction of the population containing the detected mutation (0-1.0)"). In-gene calls are keyed on their released gene through the `gene_symbol` route; intergenic calls are site-keyed on `CP009273.1:<position>` with the released flanking cell verbatim. The one `MOB` call (IS5 at 638,004, "+4 bp") is typed an insertion at a zero-length interval, with the released cells kept.

**Environment.** M9, 2 g/L D-glucose, aerobic, 37 C; 50 mg/L kanamycin A for the deletion strains, none for the wild type. The Methods' media paragraph says 25 mg/L for the growth characterizations; the RNA-seq paragraph and GEO's growth protocol say 50 mg/L for these libraries, which is stored.

**Reference.** Wild-type BW25113 without kanamycin, TPM the mean of the two released wild-type libraries, counts a typed gap.

**Cross-check against PRECISE-1K, a finding.** Matched by the compendium's `sample_id` (`fps__bw_glc__1`, `fps__fps_thrA_ale1__1`, ...) and the one-to-one ECK pairs, Pearson r of log2(TPM + 1) per library is 0.27 to 0.55 over 4,081 genes, far below the 0.95 the issue proposed, and each study library correlates about equally with every compendium library of the project. The study's own two wild-type libraries correlate at r = 0.87 with each other. Hypothesis (untested): the released "deseq2-tpm" values are a different normalization than PRECISE-1K's re-quantification, or the matrix's rows are not aligned the way its labels say. Stored as released; the compendium still does not serve these libraries.

**#960 (Guzmán 2018, bioRxiv 10.1101/310946)** is not loaded: no scriptable PDF route (manual browser download), and its release is per-library Cufflinks FPKM files, a measurement this leaf does not hold.
