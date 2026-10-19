---
id: 20w7cgtnv1kc5gi7oxejoeg
title: Data_shapes_043
desc: ''
updated: 1792450021907
created: 1792450021907
---

## 2026.10.19 - What every dataset is as tensors

Two passes over the raw LMDB (`raw_043/raw/lmdb`, 173,617 records, keys `data_<i>`):

1. A head-only pass (`buffers=True`, the first 300 bytes of each value carry
   `dataset_name`) that writes `raw_043/raw/raw_dataset_name_index.json`, dataset ->
   sorted raw indices. 563 s on the login node (page faults over the 163 GB mmap; the
   bytes themselves are not copied). The processed store's `dataset_name_index.json` is
   NOT this index: processed order is the aggregation order, so processed index 82,729
   (listed under Caudal) is raw `data_82729`, a Boocock record. The raw store is
   contiguous by dataset in query order: Kemmeren 0 to 1,481, Sameith Sm 1,482 to 1,563,
   Sameith Dm 1,564 to 1,635, Hu 1,636 to 1,897, Jackson 1,898 to 2,681, Nadal-Ribelles
   2,682 to 8,799, Messner 8,800 to 13,423, Zelezniak 13,424 to 13,520, Ohya 13,521 to
   18,198, Caudal 18,199 to 19,141, Muenzner proteome 19,142 to 19,936, Muenzner turnover
   19,937 to 19,991, Boocock 19,992 to 99,507, N'Guessan 99,508 to 102,458, Vaishnav
   102,459 to 173,616 (`raw_index_first_last` in the json).
2. The tally pass. `--mode sample --per-dataset 300` (login node, 300 evenly spaced
   records per dataset, 10 min including pass 1) and `--mode full` (slurm job 5902,
   [[experiments.043-expression-proteome-morphology.scripts.gh_data_shapes_043]], 861 s
   for the 173,617 records, 14 min 29 s wall). `sacct` MaxRSS 157 GB against a 16 GB
   request: the mmap'd LMDB pages count as resident and slurm did not kill the job; the
   process itself holds one decoded record at a time. The committed
   `results/data_shapes_043.json` is the full pass (`"mode": "full"`,
   `"slurm_job_id": "5902"`). The sample and the full pass agree on every min and median
   of the single-deletion sets; the full pass widens the isolate and segregant ranges
   (Caudal perturbations per record 195 minimum, not 2,663; Boocock blocks 46 to 1,149,
   not 85 to 1,021).

The shape table is in [[experiments.043-expression-proteome-morphology]] (2026.10.19,
"Data tensors and the dataset module"); the diagram in
[[experiments.043-expression-proteome-morphology.mermaid.data-flow]]. Findings the table
does not show (all from the full pass, n = the record count of the dataset):

- **Label keys outside the 6,607 index.** Kemmeren and both Sameith sets carry 6,169 keys
  of which 6,127 are on the index (42 outside on every record). Hu 2007 carries 5,921 to
  6,429 keys, 5,390 to 5,878 on the index (the rest are platform spot names such as
  `IntQ0185A`). Nadal-Ribelles keys are standard names: 1,441 of 6,796 distinct keys match
  the index by raw name, 6,325 after the standard-name map, 4,034 to 6,016 per record
  (median 5,665). Messner 2023 keys are UniProt accessions: 0 of 1,850 by raw name, all
  1,850 after the UniProt map. Jackson: 6,383 of 6,518 distinct keys on the index. Caudal:
  6,084 of 6,483 distinct, 5,878 to 6,062 per record. Ohya's 281 keys are CalMorph
  features, not genes (0 on the index by construction).
- **Isolate perturbation names.** `sequence_variant` names 5,973 distinct genes on the
  index and 4 outside (`YER109C`, `YFL056C`, `YFL057C`, ...); `natural_loss_of_function`
  names 4,137 on the index and 48 outside, and the outside ones are STANDARD names
  (`AAD15`, `AAD16`, `APQ13`, `ENA1`, `SSQ1`), which the dataset module resolves through
  the standard-name map at read time; `natural_gene_absence` 1,111 on the index, 7 outside
  (`YDR134C`, `YFL056C`, `YIL167W`, retired or merged ORFs); `natural_gene_presence` names
  1,614 (Caudal) and 1,643 (Muenzner) distinct pangenome ORFs, none on the index. Per
  record: variants 53 to 5,840 (median 5,069), LOF 12 to 940 (median 362), absences 43 to
  453 (median 126), presences 34 to 352 (median 78). One Muenzner proteome record has a
  plain `Genotype` with 0 perturbations (the isolate clause's `SIZE = 0` branch).
- **Segregant blocks and posteriors.** Boocock: 46 to 1,149 blocks per record (median
  720), 2,190 to 42,052 markers per record; crosses B (YJM145 x YPS163) 44,784, A (BY x
  RM) 28,137, 3004 (CBS2888 x YJM981) 6,595. Posterior is on a 0.05 grid from 0.5 to 1.0;
  34.8% of the 55,845,097 blocks but 72.0% of the 248,106,894 block-markers sit at
  posterior >= 0.9, so the 0.9 drop removes many short blocks and few markers. N'Guessan:
  57 to 2,225 blocks per record (median 483), 41,594 markers on every record (one
  reference panel), 38.1% of blocks and 94.7% of markers at >= 0.9. Both panels cover the
  16 nuclear chromosomes and nothing else.
- **Promoters.** All 71,158 Vaishnav records: 110 nt promoter, variable 80-mer at 18..97,
  scaffold `pTpA`, reporter YFP, 71,158 distinct `promoter_variant_id`s; expression level
  0.32 to 17.0 (104 records carry None).
- **Environments** (sha256 of the environment JSON): Jackson 11 (ten media plus one YPD
  record set with an environmental perturbation), Nadal-Ribelles 2 (YPD with and without
  the NaCl perturbation), Boocock 2 (the sheath-fluid medium and an SM medium), every
  other dataset 1. The store census counted Jackson at 10 because it keyed by media,
  temperature and perturbations only.
- **Single-cell depth.** Boocock 1 to 51 cells per record (median 1), 500 to 98,882 total
  UMIs (median 2,629); N'Guessan 1 to 53 cells (median 3), 1,035 to 333,210 UMIs (median
  18,221); Jackson 1 to 266 cells (median 23), 594 to 749,705 UMIs (median 43,851);
  Nadal-Ribelles 10 to 1,894 cells (median 94).
- **Reference labels.** Every record's `experiment_reference.phenotype_reference` carries
  the label: Kemmeren 6,169 keys, Messner the same 1,441 to 1,849 keys as its record,
  Zelezniak 726, Muenzner proteome 1,576, turnover 1,524, Boocock 4,696 to 5,556,
  N'Guessan 6,222, Jackson 4,555 to 5,861, Nadal-Ribelles 6,778 to 6,787, Caudal 6,483,
  Ohya 281, Vaishnav a scalar on 71,054 of 71,158 records.
