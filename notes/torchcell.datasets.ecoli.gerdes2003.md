---
id: 99ds5joqu8a8n5lmp6shae2
title: Gerdes2003
desc: ''
updated: 1791695653140
created: 1791695653140
---

## 2026.10.11 - Loader, candidate verdict and L0-L4 table

Gerdes SY, Scholle MD, Campbell JW, et al., Osterman AL (2003) J Bacteriol 185:5673-5684, doi:10.1128/JB.185.19.5673-5684.2003, PMC193955, PMID 13129938. Loader: `torchcell/datasets/ecoli/gerdes2003.py` (`GeneEssentialityGerdes2003Dataset`, store `gene_essentiality_gerdes2003`). Release inventory and raw-mirror deposit: [[experiments.036-dataset-fixes-before-kg-build.scripts.gerdes2003_release_inventory]]. Duplication by content: [[experiments.036-dataset-fixes-before-kg-build.scripts.essentiality_call_agreement_baba2006_gerdes2003]]. Verdict: `database/candidates/gerdesExperimentalDeterminationSystem2003.json`.

### Candidate gate

| gate | outcome | basis |
|---|---|---|
| G1 | pass | primary: genetic footprinting; Table S1 is the per-gene artifact |
| G2 | pass | MG1655 -> `ecoli_K12_MG1655_ASM584v2` resolves and reads |
| G3 | pass | 6 SI files in the raw mirror; Table S1 4,291 rows counted by the inventory script |
| G4 | pass | no other module holds the DOI; by content independent of the Goodall 2018 and Price 2018 stores |
| G5 | gap, #924 | `GeneEssentialityPhenotype` exists but carries no evidence type |

Outcome `admissible_with_gaps`. The gate's DOI scan could not find this key on its own: the row's key is in no module and no library manifest, and the scan crashes on the off-schema `zhangCombiningMechanisticMachine2020` raw manifest (#923), so the gate ran with `--citation-key`.

### Mirror route

The paper is in neither Zotero library (read-only search, 2026-10-10), and its body is walled for scripts: no PMC open-access copy, `journals.asm.org` 403, Europe PMC 403. The authors' Supplementary Online Material is open at `https://www.genome.wisc.edu/Gerdes2003/`; its Table S1, Table S1 page, two procedure PDFs (with `pdftotext -layout` renderings) and PubMed's XML record are deposited in `$DATA_ROOT/torchcell-raw/gerdesExperimentalDeterminationSystem2003/` by `direct_url` retrievals. The manual deposit of the article body is #925 (`by-hand`).

That the Wisconsin copy is the published Table S1 is measured, not assumed: Baba 2006's Supplementary Table 3 re-serves "Supplementary Table S1 (Gerdes et al., 2003)" per ORF, and equals this table's call on 4,113 of the 4,133 b-numbers both carry. The other 20: 15 that Baba marks absent ("-") and 5 insertion-sequence b-numbers this table lists on two rows.

### The call and what it means

E is `is_essential=True`; N is False. `N*` (ftsK, rpsA, kdsA, accD, usg, nusA) is not defined on the page, which does say "accD is called nonessential ... regardless of convincing arguments to the contrary" while writing accD's call `N*`; so `N*` is stored as N, and #925 asks the article to confirm it. X ("the gene was not covered") and ? ("ambiguous") have no boolean and are dropped. `is_essential=True` is insertion depletion: no disruptive insertion recovered after 23 population doublings of a pooled fermentor culture, which can include slow growers. The assertion error (`exp(-rL)`, the chance of missing an ORF), the insertion positions (amino-acid coordinates) and the X / ? calls have no field and are kept in `preprocess/calls.csv`.

Counts, not resolved: Table S1 holds 619 E, 3,114 N, 6 N*, 331 X, 221 ? (4,291 rows); the abstract states 620 essential and 3,126 dispensable, the procedure overview 3,746 unambiguous assertions. The table holds 3,739 E + N + N*, 7 short, with the same 4,291 total. The table is stored as released.

### Records (dev store, built at `ee131b44` with `build_dataset_lmdb --retire-existing --verify`)

| rule (applied in order) | rows |
|---|---|
| Table S1 rows | 4,291 |
| `no_blattner_id` | 64 |
| `blattner_id_on_two_rows` (9 insertion-sequence b-numbers with their own calls) | 18 |
| `blattner_id_not_on_one_mg1655_locus` (40 retired, 2 ambiguous, 54 fragments of a locus another row holds) | 96 |
| `not_covered_call` (X) | 306 |
| `ambiguous_call` (?) | 189 |
| records | 3,618 (611 E, 3,001 N, 6 N*) |

Resolved fraction 0.990 (4,167 of 4,209 distinct single Blattner ids; threshold 0.98). 29 released b-numbers are synonyms of another current MG1655 locus and are stored under it with a `locus_tag_synonym` mapping (e.g. ylbE, released b0519, stored b4572).

### Environment and reference

- Environment: the fermentor, an "LB-based rich" medium of tryptone 10 g/L, yeast extract 5 g/L, 50 mM NaCl, Neidhardt's salts and micronutrients and nine vitamins (25 components, each quoted), kanamycin 10 ug/ml, 37 C, aerobic (dissolved oxygen 30-50% of saturation), `duration_generations=23`, `duration_hours=12`. Vitamins are stated in mg/L, which is ug/mL exactly. The pH was held at 6.95 by titration; `Environment` has no pH field and a pH edit would be a perturbation every record and the reference share, so it is in the medium's provenance only. Unresolved compound labels (niacinamide, pantothenic acid, p-aminobenzoic acid, anhydrous manganese chloride, kanamycin) carry the resolver's typed gap.
- Reference: the unperturbed MG1655 parent, viable, in the same medium without the kanamycin (the paper's recovery medium, "an LB-based rich media (see below) without kanamycin").

### L0-L4 verification (`run_verification`, dev store, 2026-10-11)

| level | check | result |
|---|---|---|
| L0 | experiment and reference validate | pass, 3,618 records |
| L1 | record count | pass, 3,618 of 3,618 |
| L1 | one record per MG1655 locus | pass, 3,618 loci |
| L2 | each record's released Blattner id finds one Table S1 row; call and locus match | pass |
| L3 | Table S1 call histogram is the pinned one (paper's counts reported beside it) | pass |
| L3 | reference is the viable parent without kanamycin | pass, 3,618 of 3,618 |
| L1-L3 | shared rules (gap census, canonical names, uncertainty, compound identity, media membership) | pass; 5 compounds carry typed gaps |
| L4 | every disrupted locus is an MG1655 GenBank gene row | pass, 3,618 of 3,618 |
| L3 | provenance audit of every sourced value | pass, 23 of 23 quotes verbatim in the four raw-mirror texts |

### Files

- Loader `torchcell/datasets/ecoli/gerdes2003.py`; tests `tests/torchcell/datasets/ecoli/test_gerdes2003.py`
- Adapter `torchcell/adapters/gerdes2003_adapter.py`, conf `torchcell/adapters/conf/gene_essentiality_gerdes2003_adapter.yaml`

## 2026.10.12 - Every released row stored under its released identifier (genome-gate reversal)

Owner rule of 2026-10-12: the data is the data, and the genome never gates a record. The loader no longer calls `reconcile_locus_tags`, drops no row for an identifier the pinned annotation lacks, and writes no resolver output onto a record; the join of a released identifier to a locus is a downstream act over the gene set, and the verifier's L4 genome row is a finding (`passed=True`, with counts), never a gate. The sweep that found the gates is [[database.genome-gating-sweep.2026.10.12]].

### What changed

| before (dev store built 2026-10-11) | after |
|---|---|
| 3,618 records on `TransposonInsertionPerturbation` under the locus `reconcile_locus_tags` resolved the Blattner ID to (29 remapped with `identifier_mapping` `locus_tag_synonym`) | 4,291 records: 4,208 rows whose Blattner ID is a well-formed b-number no other row carries, on `TransposonInsertionPerturbation` under that b-number verbatim with `identifier_mapping=None`; 83 rows on `BacterialReleasedTransposonPerturbation` (PR #1077's leaf) under their ERGO id, Table S1's unique row identifier |
| `no_blattner_id` 64, `blattner_id_on_two_rows` 18, `blattner_id_not_on_one_mg1655_locus` 96 dropped | the 64 and the 18 (9 b-numbers the table lists twice, insertion-sequence copies with their own calls) and `b0018.1` are the 83 ERGO-keyed records; the 96 are stored under their released b-number, and the L4 finding reports how many of all released b-numbers the annotation carries |
| `not_covered_call` (X) 306 and `ambiguous_call` (?) 189 dropped | stored with `is_essential=None`, a typed `ProvenanceGap(field="is_essential", not_reported_by_primary)` carrying the page's own definition, and `released_call` `X` or `?`; E, N and N* rows carry their token in `released_call` too |
| `identifier_reconciliation.json` written | not written; the Blattner ID of every row stays a column of `calls.csv` beside `identifier_kind` and `systematic_gene_name` |

Count arithmetic: 4,291 rows = 4,208 b-number records + 83 ERGO records (64 + 1 + 18); calls as released, 619 E, 3,114 N, 6 N*, 331 X, 221 ?. Nothing dropped. The released-transposon leaf is #1077's `BacterialReleasedTransposonPerturbation`, on main.

### L0-L4 after the rebuild

Rebuilt from this branch under slurm (job 4455, `build_dataset_lmdb --retire-existing --verify`), 4,291 records; `run_verification`: PASS.

| level | check | result |
|---|---|---|
| L0 | experiment and reference validate | pass, 4,291 records |
| L1 | record count; one record per released identifier (4,208 b-numbers + 83 ERGO ids) | pass |
| L2 | each record's identifier finds its Table S1 row: call, gap, and the letter, insertions and assertion error on the evidence | pass |
| L3 | Table S1 call histogram pinned (619 E, 3,114 N, 6 N*, 331 X, 221 ?); reference is the viable parent without kanamycin; genetic-footprinting evidence on every record (#924), 9 E calls gapping their assertion error | pass |
| L1-L3 | shared rules (gap census 17,725 gaps over 4,291 records, canonical spellings, uncertainty, compound identity, media membership) | pass |
| L4 (finding) | `gene_containment_mg1655_locus_tags` | 4,084 of 4,208 released b-numbers are MG1655 GenBank gene rows; 124 are not; 83 records are keyed by ERGO id; reported, not gated |
| L3 | provenance audit of every sourced value | pass, 27 of 27 quotes verbatim |
