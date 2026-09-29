---
id: og63bwsmuttkslvoqz9n350
title: Showcase verification 2026.09.29
desc: 'Independent re-verification of Kemmeren 2014, Sameith 2015, Messner 2023, Mulleder 2016 and Ohya 2005 against the papers, SI, GEO and the dev LMDBs'
updated: 1790725348966
created: 1790725348966
---

## 2026.09.29 - Scope and verdicts

Five read-only Fable 5.1 agents, one per dataset, graded every previously recorded claim about Kemmeren 2014, Sameith 2015, Messner 2023, Mulleder 2016 and Ohya 2005 (loader docstrings and comments, dataset and test notes, issues #72, #143, #271 and #459, PR #460, memory) on 2026-09-29. Sources: the literature mirror (`$DATA_ROOT/torchcell-library/`), publisher and PMC/Europe PMC pages and SI, GEO SOFT records fetched live, MetaboLights MTBLS434, ProteomeXchange/PRIDE/MassIVE, the SCMD and CalMorph portals, SSBD, and the dev-tree LMDBs under `$DATA_ROOT/data/torchcell/`. Nothing under the repo or `$DATA_ROOT` was written. Track T1 of [[plan.data-release-program.2026.09.29]] (Decision 9: this note is the source of what the group 2 showcase pages may state); issue #465.

The raw reports are committed verbatim (one provenance line added under each title); every number below comes from them, and each report names the scratch script behind each measurement:

- [kemmeren2014.md](assets/verification/2026.09.29/kemmeren2014.md)
- [sameith2015.md](assets/verification/2026.09.29/sameith2015.md)
- [messner2023.md](assets/verification/2026.09.29/messner2023.md)
- [mulleder2016.md](assets/verification/2026.09.29/mulleder2016.md)
- [ohya2005.md](assets/verification/2026.09.29/ohya2005.md)

| dataset | claims | confirmed | refuted | partly | unverifiable | consequence for the served records |
|---|---:|---:|---:|---:|---:|---|
| Kemmeren 2014 (`microarray_kemmeren2014`, 1484) | 8 | 4 | 1 | 3 | 0 | sign right on 1448, wrong or cancelled on 36 (fixed on PR #460); reference `n_replicates` 400/28 is not a count the paper defines (#484) |
| Sameith 2015 (SM 82, DM 72) | 7 | 3 | 1 | 3 | 0 | 4 DM strains wrong (fixed on PR #460); wrong PubMed ID on all 154 (#478); 45 of 82 SM records mix in double-mutant arrays (#479) |
| Messner 2023 (`proteome_messner2023`, 4699) | 20 | 13 | 1 | 5 | 1 | values, reference and SE correct; `perturbed_gene_name` wrong on 156 (numeric) and 1,182 (lowercase ORF) records (#485) |
| Mulleder 2016 (`amino_acid_mulleder2016`, 4678) | 8 | 4 | 1 | 3 | 0 | values correct; `n_replicates` understated on 191 (#488); reference `n_replicates` misdescribes a population mean on all 4678 (#489) |
| Ohya 2005 (`scmd_ohya2005`, 4718) | 8 | 4 | 0 | 4 | 0 | no value wrong; the values are the Suzuki 2018 CalMorph 1.2 re-analysis of the 2005 images, not the 2005 numbers (#491) |
| total | 51 | 28 | 4 | 18 | 1 | |

Issues filed from these reports (label `torchcell-lib`; `before-next-kg-build` where a served record's content is wrong):

- Sameith: #478 PubMed ID (before build), #479 SM records absorb double-mutant arrays (before build), #480 unsourced KanMX/NatMX marker assignment, #481 loader quotations and comments
- Kemmeren and Sameith: #482 "four measurements" is 2 arrays x 2 spots, #483 GEO `VALUE` orientation and `strain` field unreliable
- Kemmeren: #484 reference `n_replicates` (before build)
- Messner: #485 `perturbed_gene_name` (before build), #486 documentation and the unrecorded 8 h duration
- Mulleder: #487 read Table S3 from the Cell SI, raw-mirror record, MTBLS434; #488 `n_replicates` for 191 strains (before build); #489 reference `n_replicates` (before build); #490 `download()` skips the hash
- Ohya: #491 values are the Suzuki 2018 re-analysis, #492 cite SSBD, #493 cell-count companion files not mirrored, #494 loader comments, #495 lit sync treats the `2005a` raw-data mirror as a broken capture (corrects #271), #496 four strains the paper flags carry no qc flag

Already tracked and not refiled: the Kemmeren channel fix and the Sameith strain fix (PR #460, #459), the two-probes-per-ORF loss (#459 comment), the SM medium state question (#143 and [[torchcell.datamodels.media-components]]). Corrections to PR #460's own claims are posted on the PR.

## Kemmeren 2014

Report: [kemmeren2014.md](assets/verification/2026.09.29/kemmeren2014.md). Totals: CONFIRMED 4, REFUTED 1, PARTLY 3, UNVERIFIABLE 0.

### Claim-by-claim

- C1. "On every array `label_ch1` is Cy5 and `label_ch2` is Cy3; `source_name_ch1/ch2` name the reference pool in exactly one channel; the reference pool is in Cy5 on "-a" arrays and Cy3 on "-b" arrays." CONFIRMED, with seven named exceptions whose GEO metadata is consistent. Raw SOFT, 3061 of 3061 arrays; independently, the deleted gene's own probes read negative under GEO's labels on 2458 of 2479 ORF-matched arrays (0.9915, median -2.60) (`kemmeren_soft_channels.py`, `kemmeren_full_pass.py`).
- C2. GEO's `VALUE` column follows one convention on all arrays (main note: "VALUE = log2(Cy3/Cy5) = log2(refpool/deletion)"). REFUTED as a single convention: VALUE is log2(deletion/refpool) on 2169 of 2633 deletion arrays and log2(refpool/deletion) on 464; the `#VALUE` label is wrong on 465 (`kemmeren_full_pass.py`, every array, all 15552 rows). Neither loader reads VALUE.
- C3. The old title rule read the wrong channel on most arrays and a second negation made the served sign right on most records. CONFIRMED. The rule's column is the reference on 2594 arrays (sign right after negation) and the true deletion on 39: 29 "-d" arrays, `pfk1-del-2-c1`, `[hs1991] ctf8-del-3-g`, `[hs1991] lsm12-del-3-e`, the six rcm1/rps0b/rps21a arrays and `[hs1991] yil014c-a-del-1-b`; 36 affected records, all in the dev LMDB (`kemmeren_full_pass.py`).
- C4. The paper's design (dye-swap pairs, independent cultures, common reference, limma M and p). CONFIRMED for the paper (paper.md lines 130, 307, 388, 390, 394); the loaders' paraphrase of "four measurements" is wrong (finding 5).
- C5. GPL11232 has two probes per ORF. PARTLY: 6128 ORFs occur twice and 12 four times in the platform table (fetched 2026-09-29); the paper gives no reason and no averaging sentence ("Each gene is represented twice on the microarray, resulting in four measurements per mutant."); both loaders keep the last row per ORF.
- C6. Strain from Table S1 mating type, MATa deletions are BY4741, "YCR087C-A absent from GEO", "YDR443C twice in Table S1". PARTLY: the strain claims are confirmed (Table S1 1474 MATalpha, 10 MATa; paper line 225); YCR087C-A is in GEO as `[hs1990] lug1-del-3-a/-b` (GSM1106470/1) and in the dev LMDB (self log2 -3.024 over 2 arrays); YDR443C occurs once (`gene` = SSN2).
- C7. Responsive / non-responsive split, 1484 mutants, 6127 genes. PARTLY: 700 + 784 is confirmed (GEO series titles, Table S1); every dev LMDB record carries 6169 ORFs; 6127 is the graph-mapped subset (`notes/plan.simb-2026-multimodal-cgt.2026.07.21.md` line 629), not a loader or paper number.
- C8. In the dev LMDB the deleted gene's own log2 ratio is negative on 97.0% of records, median -2.48. CONFIRMED: 1434 of 1479 (0.9696), median -2.4787; 45 records are >= 0, 31 of them among the 36 title-rule records; the other 14 are also positive or near zero under GEO's channels, the paper's "Deleted Gene Not Down" case (`kemmeren_lmdb_self_ratio.py`, `kemmeren_followup.py`).

### New findings

1. GEO's `VALUE` is log2(refpool/deletion) on 464 of 2633 deletion arrays (232 per series); the label disagrees with the numbers on 465 (462 "normalized log2 ratio (Cy3/Cy5)" arrays in the `[hs1990]`/`[hs1991]` resubmission batches hold log2(Cy5/Cy3); two "test/ref" arrays, including GSM630089 `abc1-del-2-a`, hold ref/test; GSM1107527 `[hs1991] opi3-del-1-a` is not an exact copy of a signal ratio). The main note's "VALUE Column Convention" is false on most arrays. #483.
2. The old rule flips 39 arrays, not "the -c and -d arrays"; every "-c" array has the right sign. 36 records are affected, not 35, and rcm1 (YNL022C) IS in the dev LMDB (stored self ratio +1.08, n_rep 2); YNR004W (+0.73) is the 30th one-array-flipped record. Posted on PR #460.
3. The `[hs1990]`/`[hs1991]` prefix (1891 of 2633 deletion titles) never changes the title rule's outcome.
4. Table S1 (publisher mmc1.xlsx, sha256 `885b96ce...`) has 1484 unique `orf name` values; the main note's "1483 unique gene deletions", "Missing gene: YCR087C-A", "Known Limitations" 1 and 2 and the MED13/SSN2 duplicate are stale.
5. The paper's "four measurements" are two arrays (one per culture, dye-swapped) times two spots per gene; GEO holds at most two arrays per mutant (687 of 700 responsive with 2, 13 with 1; 462 of 784 non-responsive with 2, 322 with 1). Both loaders' "2 biological x 2 dye-swap" comment and `N_EXPECTED_MAX_REPLICATES_DELETION = 4` are wrong; the PR #460 branch keeps the comment (lines 61-71). #482.
6. GEO's `strain` characteristic says BY4742 on all 18 arrays of the ten MATa deletions (nup133 mixed across channels) and on the GSE42240 wt-matA arrays, against the paper's BY4741; the loaders correctly take strain from Table S1. #483.
7. The served reference `n_replicates` (400 for BY4742, 28 for BY4741) counts the WT arrays of the two series of that mating type; the paper compares a mutant to one pool of 200, 200, 20 or 8 arrays (line 388), never 400. The old WT-refpool extraction applied the same reversed title rule, but the CV it computed is unused. #484.
8. The PR #460 docstrings cite "702 of 705 arrays"; `_validate_channel_assignment` on the real data reports 2538 of 2560 (PR #460 body), and the report measures 2458 of 2479 ORF-matched arrays. Posted on PR #460.
9. The responsive/non-responsive flag is only in `preprocess/data.csv`, not in the served record.
10. The main note's original "Technical Design" ("Sample -a: typically deletion in ch2, reference in ch1") was right about the channels; the code comments contradicted it.
11. Of the 6169 `ORF` keys, 29 do not match the systematic-name pattern and are served as gene keys; the report did not check them further.

### What the public showcase page can state safely

- 1,484 single-gene deletion mutants (700 responsive, 784 non-responsive), BY4742 MATalpha background except 10 MATa strains in BY4741 (paper line 225; Table S1 `mating type`; GEO series GSE42527/GSE42526 titles).
- Growth: synthetic complete medium with 2% glucose at 30 C, harvested at OD600 0.6 (paper lines 245-247); expression profiled on the Holstege lab A-UMCU-Y16k-1.3 spotted 70-mer array (GPL11232, 6,169 ORFs, each spotted twice).
- Design: two independent cultures per mutant, each hybridized on one two-color array against a common wildtype reference RNA in dye-swap, giving at most two arrays per mutant in GEO (paper line 307; GEO overall design); normalized by print-tip LOESS with gene-specific dye-bias correction (paper lines 368-380).
- torchcell stores, per mutant and per ORF, log2(deletion / reference pool) averaged over the mutant's arrays, with SE, variance and the number of arrays, plus the linear signals (fixed loader; the old loader's fields are swapped and its SE is not within-array).
- The deleted gene's own transcript reads lower than the reference on 99.1% of arrays under GEO's channel labels (2458 of 2479 ORF-matched arrays, median log2 -2.60).
- The served build (old loader) has the right sign on 1,448 of 1,484 records; 36 records are sign-flipped or cancelled and are corrected by the fixed loader.
- Data sources: GEO GSE42527, GSE42526 (deletions), GSE42215, GSE42217, GSE42240, GSE42241 (wildtypes), platform GPL11232, and Table S1 mmc1.xlsx (sha256 `885b96ce9f8d0629474bfdb28ca9db38085602e73fb55cb446b177f332a9848f`).
- Do not state: that GEO's VALUE column is log2(mutant/WT); that each mutant has four arrays; any strain read from GEO's `strain` field; "1483 mutants" or "YCR087C-A missing"; "6127 genes" without saying it is the graph-mapped subset of 6169.

## Sameith 2015

Report: [sameith2015.md](assets/verification/2026.09.29/sameith2015.md). Totals: CONFIRMED 3, REFUTED 1, PARTLY 3, UNVERIFIABLE 0.

### Claim-by-claim

- S1. "The four SI pairs marked MATa (HAC1+SNT1, SNT1+SPT2, CUP2+HAA1, SIP4+YER184C) are BY4741 and all other double mutants are BY4742; the old loader stored all 72 double-mutant records as BY4742". CONFIRMED. Paper line 115; SI `comments` reads `MATa` on exactly four passed rows; GEO carries `strain: BY4741` on exactly those 8 arrays; dev DM LMDB `{'BY4742': 72}` (`parse_gsm.py`, `lmdb_check.py`).
- S2. GEO declares both VALUE orientations, which is why #72 recomputed from `Signal Norm_Cy5` / `Signal Norm_Cy3`. PARTLY: six header strings occur (the loader's 132/127 counts two of them and omits 28 arrays); numerically 217 arrays are log2(Cy5/Cy3) and 70 log2(Cy3/Cy5), and the header contradicts the numbers on those 70 (`orientation.py`, purity 1.000 on every array). Recomputing from the signals is correct on all 287.
- S3. Paper design: replicates, dye swaps, common reference, fold-change and epistasis model. PARTLY: the design is sourced (paper lines 33, 119, 129; GEO `!Series_overall_design`), but the loader's "Sameith et al. 2015, Cell Reports" quotations are Kemmeren 2014's (mirror lines 32, 130), and its "2 biological x 2 dye-swap = 4" arrays never occur: at most 2 arrays per mutant.
- S4. "69 of the 72 double-mutant pairs are present in GEO and 3 are absent". REFUTED: GSM1044629/30 (`yer184c-del+sip4-del-matA-3-a/b`), GSM1044636/37 (`ygr067c-del+mig2-del-1-a/b`) and GSM1044642/43 (`yml081w-del+rsf2-del-1-c/d`) exist; the dev DM LMDB has all 72.
- S5. The SM loader stores a double-mutant array as a deletion of the first gene, and `"wt" in title.lower()` misclassifies SWT1. PARTLY: all 143 double-mutant arrays are folded into 45 of 82 SM records, 8 of them onto the second gene of the title (`sm_contamination.py`); no title in GSE42536 contains `wt` or `swt`.
- S6. The own-gene log2 ratio is negative on the dev LMDBs. CONFIRMED: SM 0.975 over 81, median -2.058; DM 0.979 over 143, median -1.985 (`lmdb_check.py`); YPL021W and YPR015C never go down in any record that carries them.
- S7. The paper's strain background for the singles and doubles. CONFIRMED for the background (BY4742 MATalpha singles, BY4742 doubles with four BY4741 exceptions; paper line 115); the loader's "mata" docstrings and its NatMX/SGA second-deletion marker are unsupported.

### New findings

1. Every served record carries PubMed ID 26687005, an eLife 2015 paper; Sameith 2015 is PMID 26700642 (PubMed esummary; GEO `!Series_pubmed_id`). The DOI fields are correct. The PR #460 branch still has 26687005. #478.
2. All 72 double-mutant pairs are in GEO and the dev DM LMDB; the note's "69 found / 3 missing" is stale.
3. 143 double-mutant arrays are folded into 45 of 82 SM records (`n_replicates` up to 14 for YDL020C/RPN4), 8 arrays keyed to the second gene. #479.
4. The loader's replicate quotations are Kemmeren 2014 Cell's, and `verification_report.json` says "Cell Reports, Expression Profiling"; the chain is Sameith -> ref [46] -> Kemmeren 2014. #481, and posted on PR #460.
5. "Four measurements per mutant" is 2 arrays x 2 probes; GSE42536 holds 2 arrays per double mutant (1 for cin5+yap6), 2 for 62 singles and 1 for 20 (the SI's `2 replicates` rows). #482.
6. On 70 arrays the `#VALUE` and `#INV_VALUE` headers contradict the numbers; 28 arrays carry four other header strings the loader comment omits. #481, #483.
7. The cin5+yap6 DM record (key 6) is one array: `n_replicates = 1` and SE/variance NaN for all 6169 genes, which the L2 `se_nonnegative` check skips. #481.
8. NHP6A (YPR052C) has no probe on GPL11232, so the own-gene check has n = 81 (SM) and 143 (DM).
9. Six of the 82 single mutants are Holstege-lab remakes (SI `mutant source`), yet every single is typed `SgaKanMxDeletionPerturbation`. #480.
10. The KanMX-first / NatMX-second assignment on double mutants has no source and follows regex parse order (37 vs 35 records). #480.
11. GSE42536 has no WT-vs-WT arrays; `wt_samples` is always empty and the reference of every record is the refpool channel of its own arrays. #481.
12. The reference RNA is BY4742 wild type on all 287 arrays, including the four BY4741 double mutants, so after the PR #460 strain fix a record's `genome_reference.strain` (BY4741) and the reference RNA's strain (BY4742) differ; a schema decision. Posted on PR #460.
13. The mirror's `paper.md` lacks the Fig. 1 and Fig. 2 legends; the Methods are complete.
14. Suffixes a/c/g have the mutant in Cy3 and b/d/e/i in Cy5.

### What the public showcase page can state safely

- Sameith et al. 2015, BMC Biology 13:112, DOI 10.1186/s12915-015-0222-5, PMID 26700642 (PubMed esummary; GEO `!Series_pubmed_id`). Data: GEO GSE42536 on platform GPL11232 (two-channel 70-mer oligonucleotide arrays), also ArrayExpress E-MTAB-1385 (paper line 147).
- 82 single-deletion and 72 double-deletion GSTF mutants passed the authors' quality control (paper line 115: "In total, 154 deletion mutants passed our quality control criteria"; SI sheets), and arrays for all 154 are in GSE42536 (287 arrays: 144 single-mutant, 143 double-mutant).
- Background: S288c-isogenic BY4742 (MATalpha) for all single mutants and 68 double mutants; BY4741 (MATa) for four double mutants, HAC1+SNT1, SNT1+SPT2, CUP2+HAA1, SIP4+YER184C (paper line 115; SI `MATa` comments; GEO `strain: BY4741` on those 8 arrays).
- Single mutants come from the Saccharomyces Genome Deletion library (Euroscarf / Open Biosystems), six of them remade in the authors' lab (paper line 115; SI `mutant source`). Double mutants were made by haploid transformation, random spore analysis or tetrad dissection (paper line 115).
- Growth: SC medium with 2% glucose, 30 C, harvested in early mid-log phase (paper line 119; GEO protocols file).
- Design: common-reference two-channel arrays with WT RNA in one channel; each mutant hybridized from two independent cultures on two arrays with the dyes swapped between them; two probes per gene on the array; WT-like single mutants profiled once (GEO `!Series_overall_design`; paper lines 33 and 119; SI `2 replicates` rows).
- Processing: print-tip LOESS normalization without background subtraction, gene-specific dye-bias correction (Margaritis et al. 2009), limma 2.12.0 with Benjamini-Hochberg FDR, each mutant versus a collection of same-day WT cultures (GEO `!Sample_data_processing`; paper line 119).
- Expression-based epistasis: ε_txpn,XYi = |M_xΔyΔi - (M_xΔi + M_yΔi)|, counted over genes with |ε| > log2(1.5) among genes significant (P <= 0.01) in at least one of the three mutants (paper line 129).
- What torchcell stores: per record, log2(mutant/refpool) recomputed from the normalized channel signals, averaged over that mutant's arrays; the deleted gene's own probe is negative on 97.5% (SM) and 97.9% (DM) of records with median -2.06 / -1.99.
- Not to state until fixed: the four BY4741 records (stored as BY4742), the SM records for genes that also appear in double mutants (45 of 82 mix single and double arrays), the NatMX/KanMX marker assignment, the PubMed ID, and any replicate count phrased as "2 biological x 2 dye-swap arrays".

## Messner 2023

Report: [messner2023.md](assets/verification/2026.09.29/messner2023.md). Totals: CONFIRMED 13, REFUTED 1, PARTLY 5, UNVERIFIABLE 1.

### Claim-by-claim

- C1. The matrix and metadata come from Mendeley 10.17632/w8jtmnszd9.1 as `yeast5k_noimpute_wide.csv` and `yeast5k_metadata.csv`, mirrored and hash-pinned. CONFIRMED: the Mendeley files API lists sha256 `69a9df05...aa1df9` (167,754,298 B) and `48864282...878b` (377,047 B), identical to the mirror (`messner_rawfiles.py`).
- C2. The processed matrix is Mendeley-only; PRIDE/MassIVE holds a 74.8 GB raw DIA-NN report; the Cell SI defers to Mendeley; the y5k app has no scriptable download. PARTLY: MassIVE MSV000090136 holds `.wiff` raw files and `search/20210208_diannoutput.tsv` (74,825,469,404 B), no protein matrix; PRIDE answers "The project accession is not in the database: PXD036062"; the SI files were not inspected (the mirror has no `si/`), the claim rests on the Key Resources Table; the y5k app shows a "Download" tab whose content was not retrievable.
- C3. 1,850 proteins x 5,476 samples. CONFIRMED: `matrix shape (1850, 5477) first col Protein.Group`.
- C4. The metadata carries `Filename`, `sampletype`, `ORF`, plate nr. CONFIRMED, incomplete listing: `Injection nr` and `Well nr (counted row-wise)` are also columns.
- C5. KOs are single-replicate, so `n_replicates = 1` and no per-KO SE. CONFIRMED: "Strains were not measured in replicates." (paper line 579); dev LMDB n_rep 1 and SE None on all 4,699.
- C6. The reference is the 388-replicate HIS3-complemented WT over 57 batches, mean + SE + n per protein. CONFIRMED: 388 HIS3 rows over all 57 plates; reference n 304-388 per protein; SE is `std(ddof=1)/sqrt(n)` (paper lines 507, 543, 575).
- C7. 145 ORFs have more than one strain (141 x 2, 4 x 3); 146 after case normalization. CONFIRMED (metadata and dev LMDB both `{2: 142, 3: 4}` after uppercasing).
- C8. UniProt accessions map to ORFs through SGD GFF `protein_id=UniProtKB:`, 1,850 of 1,850. CONFIRMED, including `P00410 -> Q0250`.
- C9. The stored value is a linear batch-corrected MaxLFQ quantity; "Values ~340-430 confirm linear". PARTLY: linear is confirmed by reproducing the paper's median CVs (WT 11.0% vs 11.3%, QC 7.9% vs 8.1%, KO 15.5% vs 16.2%); "~340-430" is wrong (range 0.0313 to 393,221, median 235.1), and the batch correction is on precursors before MaxLFQ (paper lines 569-571).
- C10. The no-imputation matrix is used, NaN dropped per strain. CONFIRMED: per-record 1,441 to 1,849 proteins, sum 8,466,210.
- C11. Background is the S288C BY4741 MATa deletion collection with restored prototrophy, KO as `KanMxDeletionPerturbation`, prototrophy marker not modeled. PARTLY: the paper says "(S288c) haploid (MATa) deletion collection5 with restored prototrophy94" (line 507); BY4741 and kanMX come from the cited collections (refs 5, 94), not from a Messner sentence about the KOs.
- C12. SM liquid, 6.7 g/L YNB without amino acids + 2% glucose, 30 C, deferring to Mulleder 2016. CONFIRMED (paper lines 77, 513).
- C13. L0 to L4 verification passes with the recorded numbers. CONFIRMED (`preprocess/verification_report.json`).
- C14. Casing (`YML009c`, `YAL043C-a`) is normalized, not dropped. CONFIRMED: zero KO ORFs fail the systematic-name pattern after uppercasing.
- C15. Paper totals 4,699 KOs, 1,850 proteins, 8,693,150 quantities. PARTLY: 4,699 x 1,850 = 8,693,150 exactly, so the paper's number is the imputed grid; the store holds 8,466,210 measured values (97.4%).
- C16. PMID 37080200, doi 10.1016/j.cell.2023.03.026, Cell 186:2018. CONFIRMED (PubMed esummary); open-access copy PMC7615649.
- C17. The paper table lists the label as "vector (1830)". PARTLY: 1,830 is the first record's count; per-record counts run 1,441 to 1,849 and the union is 1,850.
- C18. The served graph holds "4699 KO x 1667 proteins" versus "1,850-key union". UNVERIFIABLE (no Neo4j access); the dev LMDB union is 1,850, and the two memories disagree without citing a query.
- C19. `_gene_from_filename` "only bites on a malformed filename". REFUTED: 156 records carry a numeric gene token from real filenames (e.g. `10_9_hpr57_ko_YBL007C_2824_0.49`, so SLA1 is served as "2824"), and 1,182 a lowercase ORF.
- C20. The verifier's provenance method string. CONFIRMED (each clause under C5, C6, C8, C9, C10).

### New findings

1. 156 records' `perturbed_gene_name` is an integer string from the filename; 133 of the 154 affected ORFs have SGD standard names. #485.
2. 1,182 records store a lowercase ORF as `perturbed_gene_name` and 30 the exact ORF, three spellings for "no standard name". #485.
3. The paper's "8,693,150 protein quantities" is the imputed grid; any page quoting 8.69 M must say so.
4. The harvest timeline is stated in full (47-49 h on SM agar, 19.75 h in 200 µl SM liquid, 1/10 dilution, 8 h at 30 C, 1,000 rpm) with no OD or phase word; the loader stores `duration_hours = None`, although the 8 h post-dilution duration is sourceable. #486.
5. PXD036062 is not in PRIDE; MassIVE MSV000090136 holds 6,263 `.wiff` files, the DIA-NN report, the spectral library, the UniProt FASTA, `Description_Methods_Filenaming.pdf`, `Layout_KO_library.ods` and `params.xml`; the layout file is a second source for the KO-to-well assignment.
6. The y5k app has a "Download" tab (`#shiny-tab-download`), content not inspected; "no scriptable download" must not be read as "no download". #486.
7. The depositors call the matrix "relative intensity values"; the Mendeley record has 15 files, of which only the two used are mirrored.
8. "Values ~340-430 confirm linear" is wrong (range 0.03 to 393,221, median 235); the CV reproduction is the evidence for linear scale. #486.
9. The L4 verifier message "0.945 of scmd_ohya2005's 4549 deletion genes are in Ohya" swaps its subjects (4,549 is Messner's ORF count). #486.
10. Beside the live build sit `processed.stale-messner/`, `preprocess.stale-messner/` (uid 7474) and `deprecated-stale-2026-09-14/`; the live `build_manifest.json` records `torchcell_dirty: true` at commit 875bb433.
11. The open-access full text is PMC7615649.

### What the public showcase page can state safely

- Source paper: Messner et al. 2023, Cell 186:2018-2034.e21, PMID 37080200, doi 10.1016/j.cell.2023.03.026 (PubMed esummary); open-access author manuscript PMC7615649.
- Design: proteomes of 4,699 single-gene deletion strains of the S288c haploid MATa deletion collection with restored prototrophy (paper line 507, refs 5 and 94), grown in synthetic minimal medium (6.7 g/L YNB without amino acids, 2% glucose, no amino acid or nucleobase supplement) at 30 C (paper lines 77, 513), measured by microflow LC on a TripleTOF 6600 with DIA/SWATH and processed with DIA-NN 1.7.12 (paper lines 537, 569).
- Replication: "Strains were not measured in replicates" (paper line 579); the wild-type control (his3D::kanMX complemented by HIS3) was measured 388 times across 57 plates (paper lines 507, 543, 575), and 389 pooled-digest QC injections were run.
- Stored quantity: linear-scale MaxLFQ relative protein intensity from plate-median-corrected precursors, without imputation (Mendeley readme; paper line 571); per-KO `n_replicates = 1`, no per-KO SE; the reference carries the WT mean, SE of the mean (sample SD / sqrt n) and n (304-388) per protein, restricted to the proteins that strain measured.
- Size: 4,699 records over 4,549 deletion ORFs (146 ORFs with 2-3 independent strains, kept separately as the paper does for its descriptive analysis), 1,850 proteins in the union, 8,466,210 measured values (the paper's 8,693,150 is the imputed grid).
- Provenance: `yeast5k_noimpute_wide.csv` sha256 `69a9df05b6db011f595a4e0b3ce25c1cc247f22cbdd066c79e6da9a706aa1df9` and `yeast5k_metadata.csv` sha256 `48864282c82d516ae929dc87aff7fae9e05e9b922e316c001f3d29dce0ff878b`, fetched once from Mendeley Data 10.17632/w8jtmnszd9.1 (CC BY 4.0, published 2023-04-17) on 2026-07-12 and pinned in the mirror; raw data at MassIVE MSV000090136 / ProteomeXchange PXD036062 (raw `.wiff` plus a 74.8 GB DIA-NN report, no protein matrix).
- Do not state until fixed: gene names from `perturbed_gene_name`; "1830 proteins per strain"; any served protein count.

## Mulleder 2016

Report: [mulleder2016.md](assets/verification/2026.09.29/mulleder2016.md). Totals: CONFIRMED 4, REFUTED 1, PARTLY 3, UNVERIFIABLE 0.

### Claim-by-claim

- M1. `Table_S3_Complete_Dataset.xls` comes from Mendeley Data, is the only machine-readable location, and matches the loader pin. PARTLY: a fresh Mendeley download matches the pin (sha256 `a7fcb4bc...`), but "only machine-readable location" is refuted: the Cell SI `mmc3.xls` (Europe PMC `supplementaryFiles`, PMC5055083) is byte-identical, and MetaboLights MTBLS434 holds the raw-level uM data and 6,475 mzML files.
- M2. Each strain was measured once ("4637 of 4678") and there is no per-strain SE. PARTLY: no SE column exists in any sheet (confirmed); among the 4,678 released strains 4,487 have one `data_raw` row and 191 have two to four (167 / 20 / 4); 4,637 is out of the 4,831 raw ORFs (`mulleder_xls_inspect.py`).
- M3. "The paper says 18 amino acids but the data has 19 columns". PARTLY (counted as partly in the report's totals): refuted for "the paper says 18", since no sentence in the OCR, the PDF text, Document S1 or the PMC page gives 18, and Table S5 lists 19; the 19 data columns are confirmed.
- M4. Prototrophic derivatives of the BY4741 collection, values relative to a population robust mean used as a WT proxy. CONFIRMED: Key Resources Table (pHLUM, Addgene 40276), Figure S5B legend, line 387 (MCD estimator), Figure S1F (MCD center equals the WT profile); every record's reference equals the `robust_summary_statistics` mean.
- M5. SM medium, liquid or agar, temperature, phase; ammonium sulfate amount and pH not stated. CONFIRMED (STAR Methods "Yeast", paper line 333; no amount or pH in the paper, Document S1 or the MetaboLights protocols).
- M6. `metabolite_level` is intracellular concentration in mM. CONFIRMED: workbook `Overview`: "Intracellular concentration in [mM]. Batch-normalised data, adjusted for dilution and corrected for extraction volume, cell number and volume."; the mM uses population constants (paper line 391).
- M7. The dev LMDB count matches the paper. CONFIRMED: 4,678 records x 19 amino acids = 88,882 values; paper line 395 "4678 yeast deletion strains".
- M8. `download()` verifies the sha256 of an already-present file. REFUTED: lines 134-136 return before the hash at line 142 (`test_download_trusts_a_present_file_without_hashing` already records it).

### New findings

1. The Cell SI copy of Table S3 (`mmc3.xls`) is byte-identical to the Mendeley file, so the loader can depend on the publisher SI, as memory `no-mendeley-data-source` asks, with no record change. #487.
2. MetaboLights MTBLS434 is the paper's unnamed MetaboLights deposit (6,475 samples, a 19-metabolite MAF equal to `data_raw` to 3 decimals, 6,475 `.mzML`); torchcell references it nowhere. It holds raw-level data only. #487.
3. 191 of 4,678 released strains have 2 to 4 raw measurements, and their released mM fits the mean of the normalized rows (median log-SD 0.0067 vs 0.084 for the best single row; hypothesis supported by `mulleder_multi_average_test.py`, not a published statement). `n_replicates = 1` is understated for those 191. #488.
4. 377 released strains have their only raw row in batch 12 (the repeat screen of "the 479 deletion strains that showed lowest metabolite concentrations") and 251 in batch "03-2", which the paper text does not describe; `data_raw` holds the retained injections only.
5. `phenotype_reference.n_replicates = {aa: 1}` misdescribes the MCD robust mean, a statistic over 4,678 strains. #489.
6. The complemented WT rows (his3D::kanMX4, YOR202W, batch 04; met17D::kanMX4, YLR303W, batch 02) exist in `data_raw` but not in the mM sheet, so a measured WT profile is available (bears on the note's review flag 1).
7. The note's citation "Cell 165:1282" is wrong; the article is Cell 167(2):553-565.e12 (PMC5055083).
8. The paper text prints the Mendeley DOI truncated ("10.17632/bnzdhd6c"); only the Key Resources Table has `10.17632/bnzdhd6ck8.1`.
9. No raw-mirror provenance record exists for `Table_S3_Complete_Dataset.xls` (`$DATA_ROOT/torchcell-raw/` has no mulleder entry; mirror `si_data_sources: []`). #487.
10. `download()` never hashes a present file (M8). #490.

### What the public showcase page can state safely

- Mulleder et al. 2016, Cell 167(2):553-565.e12, doi 10.1016/j.cell.2016.09.007, PMID 27693354, PMC5055083 (PMC page).
- 4,678 single-gene deletion strains, 19 amino acids each (the 20 proteinogenic minus cysteine, "omitted ... due to its property of being quickly oxidized upon cell lysis"), absolute intracellular concentration in mM (workbook `Overview` sheet; paper "Quality Control and Completeness").
- Prototrophic MATa derivatives of the BY4741 deletion collection, prototrophy restored episomally by pHLUM (HIS3 LEU2 URA3 MET17; Addgene 40276), Mulleder et al. 2012 (Key Resources Table; Figure S5B legend).
- Grown in synthetic minimal medium (6.7 g/L YNB without amino acids, Sigma Y0626, 2% glucose; spotted on 2% agar, then liquid subculture), 30 C, harvested in exponential phase (STAR Methods "Yeast"; Figure 1A legend). Ammonium sulfate amount and pH are not printed.
- Amino acids quantified by HILIC LC-SRM/MS (Agilent 1290/6460, ACQUITY BEH amide), external bracketing calibration, 237 QC injections, batch normalization plus PQN dilution correction (STAR Methods).
- The per-strain values are one measurement for 4,487 strains and 2-4 measurements for 191; no per-strain error is released; the paper's reference profile is the MCD robust mean of all strains, shown to equal the complemented his3 WT profile (Figure S1F).
- Data: Table S3 on the Cell SI and on Mendeley Data 10.17632/bnzdhd6ck8.1 (byte-identical, sha256 `a7fcb4bc...`); raw-level uM values and mzML on MetaboLights MTBLS434.

## Ohya 2005

Report: [ohya2005.md](assets/verification/2026.09.29/ohya2005.md). Totals: CONFIRMED 4, REFUTED 0, PARTLY 4, UNVERIFIABLE 0.

### Claim-by-claim

- O1. The paper says YPD and logarithmic phase and no temperature; torchcell records YPD / liquid / 25 C, with 25 C from the Ohya-lab standard (Suzuki 2018, Ohnuki 2018/2022). PARTLY: YPD, liquid and "no temperature in the paper" are confirmed ("Each strain was grown in yeast extract/ peptone/dextrose medium, and logarithmic-phase cells were fixed."); Suzuki 2018's 25 C sentences concern its 19-mutant re-validation set, not the 4718-mutant dataset; the CalMorph User Manual section 6.1 ("3. Culture the cells at 25°C on a rotator at 25-30 r/min.") is the same-lab source nobody cited.
- O2. 4718 mutant strains, 122 wild-type replicates, all 4718 kept after name resolution. CONFIRMED: `mt4718data.tsv` (4718, 502) and `wt122data.tsv` (122, 502) match the pins; dev LMDB 4718 records, exactly 17 remapped (`ohya2005_check.py`).
- O3. 501 parameters measured, 254 kept by a Shapiro-Wilk filter; torchcell keeps 281 base + 220 CV. CONFIRMED (paper lines 25, 38; Supporting Text; CalMorph manual 5.4.1); the file has zero `TCV` columns.
- O4. Missing values: the loader drops whole rows. CONFIRMED: 0 NaN and 0 inf cells in both files; the drop rule is a latent guard.
- O5. The 122 WT profiles are averaged into one reference. CONFIRMED for torchcell (reference equals the column mean over all 281 base keys, 0 mismatches); PARTLY for the paper, which imaged 126 WT samples and standardized against a Box-Cox-transformed distribution (Supporting Text).
- O6. Strain background and collection. PARTLY: the paper says "A set of haploid S. cerevisiae MATa deletion strains (4,786 strains) was obtained from the European Saccharomyces cerevisiae Archive for Functional Analysis (EUROSCARF)." and never writes BY4741; kanMX is confirmed by the SI; the loader comment's `lys2D0` is BY4742's marker (SGD: BY4741 is `MATa his3Δ1 leu2Δ0 met15Δ0 ura3Δ0`).
- O7. Parameter ids against SI Table 1 and the CalMorph definitions, carried unchanged. CONFIRMED for `C11-1_A`, `C12-1_A`, `A101_A`, `D14-3_C`, `CCV11-1_A`, `DCV14-3_C`.
- O8. Issue #271: `2005a` is a duplicate DOI record with no paper and a 0-byte `paper.md`. PARTLY: same DOI and no paper artifacts or `library_id` are confirmed, but `2005a` is a deliberate `"kind": "dataset_raw_mirror"` record and has no `paper.md` at all.

### New findings

1. The distributed matrices are a Suzuki 2018 re-analysis of the 2005 images with CalMorph 1.2, not the 2005 numbers. SCMD2 `datasheet.php`: "The data sheets here have been published by Suzuki et al. (2018, BMC Genomics) by reanalysing the images first published in Ohya et al. (2005, PNAS) after a quality control. Cell images of the mutants are available at SSBD:ssbd-repos-000349." The 2026.07.15 note's "Ohya 2005's OWN published data" is wrong, and the manifest's `reused_by_doi` is the wrong relation. Hypothesis (untested, from the report): the earlier reversal read Suzuki 2018's reference [21] without the SCMD datasheet page. #491.
2. SSBD ssbd-repos-000349 (DOI 10.24631/ssbd.repos.2024.05.349, CC BY 4.0) publishes `sha256sum.txt` lines equal to both loader pins, and also carries the `nmrt`/`dmnt` files, per-cell tables and raw images; SCMD2 announced the end of its image services on May 31, 2025 in favor of SSBD. #492.
3. Three wild-type counts are unreconciled: 126 (paper Fig. 1B, Table 9), 122 (distributed average file), 109 (Suzuki 2018). #494.
4. The wild-type `NAME` prefixes (`04his3`, `his3new`, `his3cnt`, `his3old`) carry batch labels the mean reference discards. Hypothesis (untested, from the report): they would allow a per-batch reference or a batch-effect check.
5. `mt4718nmrt.tsv` and `mt4718dmnt.tsv` (cells per ratio parameter and per specimen) are not mirrored, so the n behind each ratio and stage mean is unrecoverable from the mirror. #493.
6. Loader line 299 says `lys2D0` for BY4741 (should be `met15Δ0`); line 64 names `test_Ohya2005.py` (the file is `test_ohya2005.py`). #494.
7. The docstring attributes "grown at 25 C" to Suzuki 2018's dataset; the CalMorph User Manual section 6.1 is the citation that should stand. #494.
8. The `TCV` prefix in `_CV_PREFIXES` and the docstring does not exist (CCV 60 + ACV 33 + DCV 127 = 220). #494.
9. The paper's SCMD URL (`http://scmd.gi.k.u-tokyo.ac.jp`) timed out; the working host is `www.yeast.ib.k.u-tokyo.ac.jp`, HTTP only.
10. The Supporting Text records strain-identity caveats for rad18 (an α cell), ctf8 (a mixture of a and α), scp160 (2N DNA content) and rho4 (phenotype not linked to the kanamycin cassette); their served records carry no qc flag. #496.

The `2005a` kind-awareness fix for the literature sync (O8) is #495.

### What the public showcase page can state safely

- Source: Ohya et al. 2005, PNAS 102:19015-19020, "High-dimensional and large-scale phenotyping of yeast mutants" (PMID 16365294, DOI 10.1073/pnas.0509436102). The screened set: "A set of haploid S. cerevisiae MATa deletion strains (4,786 strains) was obtained from ... (EUROSCARF)"; "Deletion strains of 4,718 ORFs were used." (paper Methods).
- Medium and growth state: "Each strain was grown in yeast extract/peptone/dextrose medium, and logarithmic-phase cells were fixed." (paper Methods). The paper states no temperature; torchcell records 25 C from the Ohya-lab CalMorph protocol ("Culture the cells at 25°C", CalMorph User Manual section 6.1) and the same-lab Ohnuki 2018/2022 Methods. Say "25 C (lab protocol, not stated in the 2005 paper)".
- Phenotype: 501 CalMorph parameters per strain, "information about cell shape (visualized by cell wall staining), the actin cytoskeleton, and nuclear morphology of cells at a specific stage of the cell cycle" (paper); served as 281 base parameters + 220 coefficient-of-variation parameters; ids are the CalMorph ids unchanged (SI Table 1; CalMorph manual), units pixel / pixel^2 / unitless ratios.
- Values: per-strain population averages, one replicate per deletion strain ("we have only one set of data for each deletion strain", paper); wild-type reference = per-parameter mean over the 122 distributed his3 wild-type replicate averages (torchcell aggregation; the paper imaged 126 and used a Box-Cox-transformed distribution instead).
- Data files: `mt4718data.tsv` (4,718 x 501) and `wt122data.tsv` (122 x 501), sha256 `c4ba1e84...0fc0` and `ab2c31b5...27c3`, retrievable from the SCMD/CalMorph portal and from SSBD ssbd-repos-000349 (DOI 10.24631/ssbd.repos.2024.05.349, CC BY 4.0) where the published `sha256sum.txt` lists the same digests.
- Provenance caveat to state: the distributed matrices are the CalMorph 1.2 re-analysis "after a quality control" published with Suzuki et al. 2018 (BMC Genomics 19:149), computed on the 2005 images; they are not the numbers the 2005 paper's statistics were run on.
- Counts: 4,718 records, 0 dropped, 17 ORF names remapped to current SGD identifiers, 0 missing values in the source file.
