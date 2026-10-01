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
