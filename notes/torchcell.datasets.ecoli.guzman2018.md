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

## 2026.11.24 - Per-replicate review: one record per library, replicate_id set

Review under the #1178 per-replicate policy, part of #1171. The store already held one record per biological replicate; the only change is that each record now names its replicate.

**What a replicate is.** One culture tube, one RNA extraction, one sequenced library, its own GEO sample and BioSample. Quotes from the pinned PMC text (`paper/PMC6296033.1.txt`, sha256 `392b5f3d016ddeb5158fb8bb22b3e61af7d6b65c62fdd2653432a58a7fc46b71`), RNA sequencing section: line 279, "The pre-cultures were split into two 17 mL tubes (biological duplicates) for each experimental condition."; line 281, "For each condition examined, biological duplicate samples were sequenced and analyzed." The Fig. 3 legend (line 205) and the DESeq2 paragraph (line 281) compare "biological duplicate replicates" against wild type "biological duplicate replicates". No technical level exists (one library per tube) and no sample is withdrawn: the series matrix (`5a1c437a...`) lists 14 GSMs (`GSM3290757`-`GSM3290770`, each with its own BioSample) and the TPM matrix (`590b94e9...`) carries 14 columns.

**Counts, measured.** 7 conditions (wild type plus six deletion-strain clone or population samples) x 2 libraries = 14 records, `_R1` 7 and `_R2` 7. Records before 14, after 14.

**Change.** `RNASeqExpressionPhenotype.replicate_id` is the release's TPM column label verbatim (`wt_R1`, `thrA_A1_F5_I1_R1`, ...), which prints the replicate label. The reference stays the mean of the two wild-type libraries with `replicate_id=None`; both wild-type libraries remain records. The leaf has no dispersion or n field, so no `ProvenanceGap` for a single measurement is needed. New sourced value `duplicate_tubes`, new L1 rule `replicate_id_is_release_column` (each record keyed one-to-one on a released column, its TPM that column's own vector). Docstring fix: the read totals are a mean of 9.8 million and a range of 6.8 to 17 million per the Methods, not "2-to-17-million".

**Scratch build and checks.** `$DATA_ROOT/tmp/scratch/per-replicate-guzman2018/rnaseq_guzman2018`, verifier PASS (L0 14 records, L1 count 14, replicate groups 7 x 2, replicate ids 14 one-to-one; L4 PRECISE-1K min r 0.2702 unchanged, a finding). Compared with the dev store: every record equal apart from `replicate_id`, max |TPM difference| 0.0; the mean over each (genotype, environment) group equals the old group mean (max difference 0.0); the mean of the two wild-type records equals the stored reference (max difference 0.0). The dev store under `$DATA_ROOT/data/torchcell/rnaseq_guzman2018` predates this change and fails the new L1 rule until rebuilt.

**Not changed.** The growth triplicates of Table 1 and Fig. 1b (Additional file 2 is a figure PDF) are not loaded by this dataset and release no per-replicate table. The one DNA sample per (experiment, flask, isolate) in Additional file 4 means both libraries of a condition carry the same calls, which is the release's design, not a merge.
