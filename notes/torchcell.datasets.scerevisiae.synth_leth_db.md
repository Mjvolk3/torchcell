---
id: qdfgck00436h71efgxwc5jf
title: Syn_leth_db_yeast
desc: ''
updated: 1721782510231
created: 1721782460950
---

## 2024.07.23 - We Will Not Be Able to Manually Populate Meta Data for all Synthetic Lethality and Synthetic Rescue Experiments

```python
len(set([i['experiment']['pubmed_id'] for i in lethality_dataset]))
1741
len(set([i['experiment']['pubmed_id'] for i in rescue_dataset]))
1918
```

## 2026.10.01 - PMID read as text, blank PMID refused, main under DATA_ROOT (issue #528)

- Before: `pd.read_csv` typed `r.pubmed_id` by content, so one blank cell turned the column float and every record stored `"111.0"` with a `.../111.0/` URL and the blank one `"nan"`; `main` built both datasets at the default relative roots under the working directory.
- Now: `_read_synlethdb_csv` reads `r.pubmed_id` as text and raises `BlankPubmedIdError` ("<path>: <n> row(s) with a blank r.pubmed_id (first at row <i>)"); `main` builds under `$DATA_ROOT/data/torchcell/synth_lethality_yeast_synth_leth_db` and `.../synth_rescue_yeast_synth_leth_db`, the directories the knowledge-graph configs read.
- Record-neutral: 0 blank PMIDs in `Yeast_SL.csv` (14,000 rows) and `Yeast_SR.csv` (6,948 rows); the text read stores the same string as before on every row (0 mismatches; SL already read as text because one row cites `24125552;19918932`). The built dev stores hold `"18676811"`-shaped PMIDs on all 14,000 and 6,948 records.
- Left open (record-changing on the pinned files, measured with the genome at `$DATA_ROOT/data/sgd/genome/data.db`): the name mapping lets another gene's alias overwrite a standard name (868 SL and 115 SR rows would map differently under standard-name precedence, for example SDC1 stored as YJR090C (GRR1) instead of YDR469W); 7 SL and 2 SR records repeat an ORF pair, all of them artifacts of that overwrite (0 under standard-name precedence); 3 SL and 6 SR rows name one gene twice (same Entrez id both sides); 0 swapped-order pairs. The raw CSV carries Entrez ids (`n1.identifier`, `n2.identifier`) that could resolve names without the alias table.
- Tests: `test_pmids_are_stored_verbatim_as_text`, `test_a_blank_pmid_refuses_the_build_by_name`, `test_main_builds_both_datasets_under_data_root`.

## 2026.10.01 - Review corrections: primed names and whitespace PMIDs

- A whitespace-only PMID (`" "`) counts as blank and raises `BlankPubmedIdError` like an empty cell (it used to be stored as-is). The pinned files have 0 such cells.
- A second wrong-ORF source, also left open: `get_systematic_name` strips the prime from `IMP2'`, so it resolves to YMR035W (IMP2) while its Entrez id 854652 is YIL154C (IMP21): 4 SL records (keys 241, 6067, 8568, 10553) and 4 SR records (keys 356, 357, 3705, 5131), verified in the built dev stores. With the alias overwrite, 872 SL and 119 SR records carry the wrong ORF. The mapping fix is record-changing and goes to its own dataset issue.

## 2026.10.02 - Genes resolved by Entrez id, self pairs dropped (issue #597)

- Was wrong: the name map let a later gene's `Alias` overwrite an earlier gene's standard name (last write wins), and `get_systematic_name` stripped the prime from `IMP2'`, so 872 of 14,000 SL and 119 of 6,948 SR records carried the wrong ORF; 7 SL and 2 SR records repeated an ORF pair; 3 SL and 6 SR records named one gene on both sides.
- Now: each side resolves by its Entrez id (`n1.identifier`, `n2.identifier`) through the RefSeq `ncbi_genomic.gff` (`GeneID` -> `locus_tag`, gene and pseudogene features), and the gene name must resolve to that same current gene through `SCerevisiaeGenome.resolve_gene_name` (standard name before alias; `IMP2'` keeps its prime and is an alias of YIL154C). A row is dropped, never re-resolved by name, under one of three rules recorded in `preprocess/dropped_records.json` (`DropLog`, with every dropped source row): `entrez_id_not_in_ncbi_gff`, `gene_name_disagrees_with_entrez_id`, `same_gene_on_both_sides`. A repeated unordered ORF pair among kept rows raises `DuplicateOrfPairError`. LMDB keys are now contiguous over kept rows, so they are no longer source-row indices.
- The loader is now pinned: `process()` verifies `Yeast_SL.csv` (`091e04db...56ca`) and `Yeast_SR.csv` (`d84fba78...18f4`), the dev-tree raw bytes the served build consumed, and `load_entrez_to_orf` verifies the GFF before reading it. `synth_leth_db` left `UNPINNED_LOADERS` in `test_raw_pins.py`.
- Files the resolution reads: `$DATA_ROOT/data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/ncbi_genomic.gff`, sha256 `8200def54936659721e3f7b4484d725667ed4e23492edefbbca667b8430209c4` (6,470 Entrez ids, header `annotation-source SGD R64-4-1`), and the genome's `data.db` for the name cross-check, sha256 `feadb0f5d7662606c5a6e1cfba873cdcd852900da9ebcb920bd4b6ef38530ded`, content digest `9fae73b7...c1f4`, 28 chrmt gene features. `data.db.bak` and `data_alt.db` have 0 chrmt gene features (different content digests); the build reads only `data.db`.
- Measured (`experiments/036-dataset-fixes-before-kg-build/scripts/synth_leth_db_entrez_resolution.py`, scratch build of both datasets against the dev stores; output `results/synth_leth_db_entrez_resolution.json` and `results/synth_leth_db_entrez_resolution_changed_rows.csv`):
  - SL: 14,000 source rows, 13,996 records. Old-store rows disagreeing with the Entrez ORF pair: 872. Of these, 870 kept rows now carry a different ORF pair and 2 were dropped (row 5232 by `entrez_id_not_in_ncbi_gff`, row 5719 by `same_gene_on_both_sides`); no kept row outside the 872 changed. Dropped: row 5232 (`TAF1,853191,YPR108W-A,1466522`; Entrez 1466522 is in neither the pinned GFF nor the current R64-5-1 NCBI GFF, while SGD R64-4-1 has YPR108W-A) and self pairs 3152 (PUS1), 5719 (TAF1), 8663 (SBA1). Duplicate unordered ORF pairs: 7 old, 0 new.
  - SR: 6,948 source rows, 6,942 records. Old-store disagreements: 119; kept rows changed: 119, all of them disagreements. Dropped: self pairs 370, 1182, 3138, 3154, 6616, 6672. Duplicates: 2 old, 0 new.
  - Name rule: 0 rows in either file (every name agrees with its Entrez id).
  - The ten issue names plus `IMP2'`, ORFs stored old -> new: STM1 YPR163C -> YLR150W, SDC1 YJR090C -> YDR469W, MFT1 YMR177W -> YML062C, NSP1 YLR178C -> YJL041W (SL only), RPL37A YPL143W -> YLR185W, CCS1 YOL081W -> YMR038C, YPK1 YNL307C -> YKL126W, TAF1 YHL047C -> YGR274C, SSL2 YLR452C -> YIL143C, HAP1 YPL101W -> YLR256W, IMP2' YMR035W -> YIL154C.
- Tests: `tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py` rewritten around a stub genome carrying the real `resolve_gene_name` and a fixture GFF (the STM1 standard-vs-alias and `IMP2'` prime cases are mirrored); `test_real_files_resolve_by_entrez_with_the_issue_counts` (`--data`) pins the drop rows, 0 duplicates and the eleven names on the real files.
- Open:
  - `ncbi_genomic.gff` is not in the genomes tier (`torchcell-genomes/sgd_S288C_R64-4-1_20230830` excludes it) and NCBI now serves R64-5-1 under GCF_000146045.2, so its retrieval cannot be reproduced; the sha256 pin is the only record. Depositing it as its own tier set with `provenance_complete=False` needs a go-ahead.
  - Row 5232 is dropped because its Entrez id is absent from the pinned GFF, although its name `YPR108W-A` is a current SGD ORF. Resolving it by name would be a fallback; reversing the drop is a decision for the user.
  - The self-pair drop assigns no meaning to a gene paired with itself; if SynLethDB documents one, the rule can be revisited.
