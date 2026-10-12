---
id: o0kvr8dzjkpyj6alqjqg4uv
title: Baba2006
desc: ''
updated: 1791695645213
created: 1791695645213
---

## 2026.10.11 - Loader, candidate verdict and L0-L4 table

Baba T, Ara T, Hasegawa M, Takai Y, Okumura Y, Baba M, Datsenko KA, Tomita M, Wanner BL, Mori H (2006) Mol Syst Biol 2:2006.0008, doi:10.1038/msb4100050, PMC1681482, PMID 16738554. Loader: `torchcell/datasets/ecoli/baba2006.py` (`GeneEssentialityBaba2006Dataset`, store `gene_essentiality_baba2006`). Release inventory and raw-mirror deposit: [[experiments.036-dataset-fixes-before-kg-build.scripts.baba2006_release_inventory]]. Duplication by content: [[experiments.036-dataset-fixes-before-kg-build.scripts.essentiality_call_agreement_baba2006_gerdes2003]]. Verdict: `database/candidates/babaConstructionEscherichiaColi2006.json`.

### Candidate gate

| gate | outcome | basis |
|---|---|---|
| G1 | pass | primary: a construction screen; the per-ORF tables are the artifact |
| G2 | pass | BW25113 -> `ecoli_K12_BW25113_ASM75055v1` resolves and reads |
| G3 | pass | Tables 2, 3, 6 deposited (`.xls`, counted by the inventory script: 4,288 / 4,378 / 303 rows) |
| G4 | pass | no other module holds the DOI; by content independent of the Goodall 2018 and Price 2018 stores |
| G5 | gap, #924 | `GeneEssentialityPhenotype` exists but carries no evidence type |

Outcome `admissible_with_gaps`. The literature mirror holds the paper, but its `si/si1.pdf` is a second copy of the article, not the supplement, so the three tables and PMC's plain text went to the raw mirror.

### The call and what it means

The paper's scoring rule, "Mutants were scored as correct if two or more colonies had the expected structure based on PCR tests for both junction fragments", applied to Table 3's `5. verified` column. Measured on the deposited bytes:

| quantity | Table 3 under the rule | paper |
|---|---|---|
| targeted ORFs | 4,288 (= Table 2's JW ids) | 4,288 |
| obtained (2 or more verified) | 3,985 | 3,985 |
| not obtained (0 or 1 verified) | 303 (= Table 6 exactly) | 303 |
| not obtained with 1 verified | secM (1 of 8) | "The exceptional case, secM" |

`is_essential=True` is therefore a construction outcome on the selection plates, a "candidate" in the paper's word. It is not a growth measurement, and the paper says why it differs from genetic footprinting: "Because genetic footprinting measures cell populations, a mutation causing slow growth can lead to under-representation of the mutant and hence false classification of many genes as essential. In contrast, we sought deletion mutants as survivors without regard to growth rate." The PCR counts behind each call have no field; they are kept per row in `preprocess/calls.csv`.

### Records (dev store, built at `ee131b44` with `build_dataset_lmdb --retire-existing --verify`)

| step | rows |
|---|---|
| Table 3 rows | 4,378 |
| `not_targeted` (`N.A.`: IS genes, ldr toxins, genes BW25113 lacks, re-annotated ORFs) | 90 |
| `jw_id_not_on_one_bw25113_locus` (125 not carried by the annotation, 47 sharing a locus with another JW id) | 172 (166 obtained, 6 not) |
| records | 4,116 (297 essential, 3,819 not) |

The JW id resolves as a BW25113 `gene_synonym`; every record carries `identifier_mapping` (`jw_synonym`). Resolved fraction 0.971 (4,163 of 4,288; threshold 0.95). `collection` is "Keio collection" only on an obtained deletion, since an unobtained one names no strain. Cassette "kanamycin cassette".

### Environment and reference

- Environment: the selection plates, LB agar at Baba's LB amounts (1% tryptone, 0.5% yeast extract, 0.5% NaCl, which are the `LB_LENNOX` amounts) with kanamycin at 30 ug/ml, aerobic, 37 C; agar amount unstated (`concentration=None`), incubation time a typed gap.
- Reference: the unperturbed BW25113 parent, viable, on the same LB agar WITHOUT the kanamycin. The parent carries no kanamycin marker, so "viable on the selection plates" would be false; the paper grows it "with or without antibiotics". Goodall 2018 and Price 2018 put the parent on their selection plates; that convention was not copied and not changed there.

### L0-L4 verification (`run_verification`, dev store, 2026-10-11)

| level | check | result |
|---|---|---|
| L0 | experiment and reference validate | pass, 4,116 records |
| L1 | record count | pass, 4,116 of 4,116 |
| L1 | one record per BW25113 locus | pass, 4,116 loci |
| L2 | each record's JW id finds its Table 3 row; call, collection and locus match | pass |
| L3 | scoring rule reproduces 4,288 / 3,985 / 303 and Table 6 | pass |
| L3 | reference is the viable parent without kanamycin | pass, 4,116 of 4,116 |
| L1-L3 | shared rules (gap census, canonical names, uncertainty, compound identity, media membership) | pass; kanamycin carries the resolver's typed gap |
| L4 | every deleted locus is a BW25113 GenBank gene row | pass, 4,116 of 4,116 |
| L3 | provenance audit of every sourced value | pass, 14 of 14 quotes verbatim in `paper/PMC1681482.1.txt` |

### Not stored

- Table 3's OD600 growth of the obtained mutants (LB 22 h, MOPS 24 h and 48 h; 3,984 of 3,985 with all three): a separate measurement with no released wild-type value; owner decision in #927.
- Table 3's re-served genetic / PEC / Tn5 / score columns: other studies' calls. The genetic column is used to corroborate the Gerdes 2003 Table S1 copy ([[torchcell.datasets.ecoli.gerdes2003]]).

### Files

- Loader `torchcell/datasets/ecoli/baba2006.py`; tests `tests/torchcell/datasets/ecoli/test_baba2006.py`
- Adapter `torchcell/adapters/baba2006_adapter.py`, conf `torchcell/adapters/conf/gene_essentiality_baba2006_adapter.yaml`

## 2026.10.12 - Every released row stored under its released identifier (genome-gate reversal)

Owner rule of 2026-10-12: the data is the data, and the genome never gates a record. The loader no longer calls `reconcile_locus_tags`, drops no row for an identifier the pinned annotation lacks, and writes no resolver output onto a record; the join of a released identifier to a locus is a downstream act over the gene set, and the verifier's L4 genome row is a finding (`passed=True`, with counts), never a gate. The sweep that found the gates is [[database.genome-gating-sweep.2026.10.12]].

### What changed

| before (dev store built 2026-10-11) | after |
|---|---|
| 4,116 records on `BacterialDeletionPerturbation`, each a BW25113 locus tag derived from the JW id (`identifier_mapping` `jw_synonym`) | 4,378 records on `BacterialReleasedStrainPerturbation`: `systematic_gene_name` = the Keio JW id verbatim, `perturbed_gene_name` = the released gene name (the JW id where Table 3 prints `none`), `strain_label` = the JW id, `identifier_kind` "Keio JW id", `released_kind` deletion |
| `jw_id_not_on_one_bw25113_locus`: 172 rows dropped (125 the annotation does not carry, 47 sharing a locus with another JW id) | all 172 stored; the L4 finding reports how many JW ids the annotation places and how many share a locus |
| `not_targeted`: 90 N.A. rows dropped | all 90 stored with `is_essential=None`, a typed `ProvenanceGap(field="is_essential", not_reported_by_primary)` and `released_call="N.A."` |
| `identifier_reconciliation.json` written | not written; the ECK accession and b-number of each row stay released columns of `calls.csv` |

Count arithmetic: 4,378 Table 3 rows = 3,985 obtained (`is_essential=False`) + 303 not obtained (`True`, Table 6 exactly) + 90 not targeted (`None`). Nothing dropped; `dropped_records.json` carries `rules: []`.

Typed gaps per record class: the 90 N.A. rows gap `is_essential`; every other gap (the selection plates' incubation time) is unchanged.

The schema change behind the None call (additive): [[torchcell.datamodels.schema]] `GeneEssentialityPhenotype.is_essential: bool | None` and `released_call`, record `database/schema_changes/2026-10-12-essentiality-released-call-gap-and-released-strain-construction.json`.

### L0-L4 after the rebuild

(L0-L4 table filled in after the rebuild)
