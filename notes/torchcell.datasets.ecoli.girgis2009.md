---
id: expv47psml94olx5nmp77so
title: Girgis2009
desc: ''
updated: 1791425274811
created: 1791425274811
---

## 2026.10.07 - Loader added: the 17-antibiotic transposon selection

Row 38 of the fifty bacterial datasets. Girgis, Hottes and Tavazoie 2009, PLoS ONE
4(5):e5629, doi:10.1371/journal.pone.0005629, PMC2680486, PMID 19462005, citation key
`girgisGeneticArchitectureIntrinsic2009`.

- loader: `torchcell/datasets/ecoli/girgis2009.py`, `EnvChemgenGirgis2009Dataset`
- adapter: `torchcell/adapters/girgis2009_adapter.py`, conf
  `torchcell/adapters/conf/ecoli_env_chemgen_girgis2009_adapter.yaml`
- tests: `tests/torchcell/datasets/ecoli/test_girgis2009.py`,
  `tests/torchcell/adapters/test_girgis2009_adapter.py`
- dev store: `$DATA_ROOT/data/torchcell/ecoli_env_chemgen_girgis2009`
- raw mirror: `$DATA_ROOT/torchcell-raw/girgisGeneticArchitectureIntrinsic2009/data/`

### What the paper measured

A library of about 5 x 10^5 single-insertion transposon mutants of MG1655 was propagated
for several days in each of 17 antibiotics, at concentrations that impaired but did not
abolish growth of the parent. The surviving population was read by microarray genetic
footprinting: the transposon-adjacent genomic DNA was amplified, labeled and hybridized
against labeled genomic DNA on spotted arrays, so each array spot reports how abundant the
mutants of that gene became. The released statistic is a signed per-drug score.

The loader stores **Dataset S5, "Combined Z-scores for all loci"** (`si24.xls`). It also
reads Dataset S1 (`si20.xls`, the significant subset), Dataset S3 (`si22.xls`) and Dataset
S4 (`si23.xls`) as build-time checks. Dataset S2 (the normalized ratios) is not consumed
and is recorded in `NOT_MIRRORED`.

### Sourcing table

Every value is a module-level `SourcedValue` anchored to `paper.md`
(sha256 `974add90e9f1737adb55ef2e4131c1eaf1d1a0931d03e34962a1d99e60afac0d`) or to the OCR of
Text S1, `si/si1.md` (sha256
`36a4fec61377744520f578041252c16b1f9aa9829e57366fbff1385717082dfe`), both in the literature
mirror. All 24 plus the 17 Table 1 dose rows pass `audit_sourced_value` in the L0-L4 run.

| key | value | source | verbatim quote (abridged where long) |
|---|---|---|---|
| `reference_strain` | MG1655 | paper.md, Methods | "All experiments were performed using $E$ . coli MG1655 [79]." |
| `library_construction` | a MG1655 DlacZ strain | paper.md, Methods | "Transposon insertion mutants were generated in a MG1655 DlacZ strain as described in a previous study [25]." |
| `library_parent` | MG1655 delta-lacZ | si1.md, Direct Fitness Competitions | "the library's parental strain (MG1655 (delta)lacZ), were pelleted and washed with M9 media." |
| `medium` | M9 salts + 0.4% glucose + 0.1% casamino acids + 1 mM MgSO4 + 0.1 mM CaCl2 + 1.5 uM thiamine | paper.md, Methods | "All experiments were conducted in M9 salts [80] supplemented with 0.4% glucose, 0.1% casamino acids, 1 mM MgSO4, 0.1 mM CaCl2, and 1.5 uM thiamine." (OCR carries the LaTeX math) |
| `temperature_c` | 37.0 | paper.md, Methods | "Unless otherwise noted, cultures were shaken at 37 C ." |
| `aerobicity` | aerobic | paper.md, Results | "Using an aerobic environment ... likely increased the number and types of beneficial mutations identified." |
| `library_size` | 500000 | paper.md, Figure 1 legend | "An aliquot of a library containing ~5 x 10^5 mutants each with a single transposon insertion [25] ..." |
| `subinhibitory_doses` | impaired but not abolished growth | paper.md, Results | "we used antibiotic concentrations that impaired but did not completely inhibit the growth of the wildtype strain (Table 1)." |
| `serial_transfer` | 2% transferred daily | paper.md, Figure 1 legend | "Each day, an aliquot was frozen, and 2% of the culture was transferred to fresh media to continue the selection." |
| `assay` | microarray genetic footprinting | paper.md, Methods | "Genetic footprinting and subsequent hybridization to DNA spotted arrays were performed as described in Girgis et al. [25]." |
| `gene_level` | insertions in or near a gene | paper.md, Figure S4 legend | "Yellow (blue) indicates that transposon insertions in or near a gene were beneficial (deleterious)." |
| `replicate_selections` | 2 (the floor) | paper.md, Methods | "Samples from at least two independent replicate selections were hybridized for each antibiotic." |
| `z_score_definition` | z = (log2(r) - mean) / sd of the reference hybridizations | paper.md, Methods | "Two Z-scores were calculated for each ratio, r, where z = (x - mu)/sigma, x = log2(r), and mu and sigma are the mean and standard deviation, respectively, of the log2 ratios for the gene from reference hybridizations." |
| `combination_rule` | the z-score closest to zero when all agree in sign, else 0 | paper.md, Methods | "When all of the Z-scores had the same sign, we assigned the gene the Z-score in the set that was closest to zero ... When a gene had Z-scores of different signs, the gene was assigned a score of 0, indicating no consistent fitness effect." |
| `sign_convention` | positive = the disruption is beneficial | paper.md, Methods | "disruption of a locus was classified as beneficial in aminoglycosides if the gene had positive Z-scores in all four drugs" |
| `technical_replicates_averaged` | PIP, FOX, TET, TRM | si1.md, Array pre-processing | "For pipercillin, cefoxitin, tetracycline, and trimethoprim, two separate hybridizations were done for one of the samples. ... Corresponding values were averaged and treated as a single repetition during subsequent analysis." |
| `min_two_repetitions` | 2 | si1.md, Z-Score Calculation | "For a gene to be assigned a non-zero z-score, data from at least two repetitions was needed." |
| `significance_thresholds` | {2: 2.15, 3: 1.5} | si1.md, False Positive Rate | "a cutoff of 2.15 (positive or negative) was chosen as the threshold for antibiotics for which we had two hybridizations. For those antibiotics with three hybridizations ... a cutoff of 1.5" |
| `false_positive_budget` | 2 | paper.md, Methods | "The significance threshold was set so that two false positives are expected per antibiotic." |
| `global_sd_component` | 0.15 | si1.md, Z-Score Calculation | "The value of 0.15 was chosen heuristically based on simulations of the number of false positives expected at significance thresholds ..." |
| `ratio_floor` | 0.05 | si1.md, Z-Score Calculation | "Ratios smaller than 0.05 were set to 0.05." |
| `array_normalization` | 5000 | si1.md, Array pre-processing | "arrays were normalized so that the sum of the ratios for the set of genes present on all arrays (3334 genes) was 5000 (arbitrarily chosen)." |
| `released_datasets` | Dataset S5 | paper.md, Methods | "Supplementary information contains normalized ratios (Dataset S2) ... combined Z-scores (Dataset S5), and the combined z-scores considered significant (Dataset S1)." |
| per-drug dose and day | Table 1 | paper.md, Table 1 | each drug's own `<tr>...</tr>` row verbatim, e.g. `<tr><td>Fusidic acid</td><td>FUS</td><td>180</td><td>4</td><td>2</td><td></td><td>Protein synthesis, 50S</td><td>Bacteriostatic</td></tr>` |

Raw-data pins (all four `pmc_cloud` retrievals re-run on 2026-10-07 and byte-identical):

| file | released as | sha256 | bytes |
|---|---|---|---|
| `si20.xls` | Dataset S1 | `b6c4daed3db9cbdd6e9683e5ace995467cac79cacc8ca3c7a8f181675e7edf38` | 902144 |
| `si22.xls` | Dataset S3 | `7f190bec346242ce0d883ad22803d0eca7f09e350fee994f281f51f66915ee21` | 3575808 |
| `si23.xls` | Dataset S4 | `6e76fc1c450cf9f63f1a5ecb06c9e7caeddcbdbe099458704a964ac4cd8c7d8b` | 3580416 |
| `si24.xls` | Dataset S5 | `823c6aaffe26a2419a0fa5bbf0b0e72dc6cd14a390995a28696011a5e1ab9da5` | 1237504 |

### Record type, with the measured sign distribution

`BacterialEnvironmentResponseExperiment` carrying `EnvironmentResponsePhenotype` with
`measurement_type=z_score`, `assay_type=other`, one record per (gene, antibiotic).

**Measured over the 63,766 stored records: 9,600 positive (15.1%), 11,523 negative (18.1%),
42,643 exactly zero (66.9%).** That is why the record is not a `FitnessPhenotype`: that
class is a strictly positive ko/wt ratio which clamps non-positive values and whose verifier
requires a 1.0 reference, so it would erase 18% of the measurements outright and flatten
two thirds of them.

The zeros are measured statements, not missing data. The combination rule assigns 0 when the
drug's z-scores disagree in sign, which is the paper's own way of saying "no consistent
fitness effect". A cell with fewer than two usable repetitions is released as `ND` instead
and is dropped.

`assay_type` is `other` because `AssayType` has no member for microarray genetic
footprinting. It is pooled competitive growth, but not
`pooled_competitive_growth_barcode`: there is no molecular barcode, the insertion's own
genomic position is what hybridizes. A `genetic_footprinting_microarray` member would be
more precise; `schema.py` is not edited here.

`n_samples` is the number of independent replicate selections hybridized, with
`sample_unit=biological_replicate`: 3 for ampicillin, lomefloxacin, sulfamonomethoxine and
doxycycline, 2 for the other 13. No uncertainty exists per record; both
`environment_response_uncertainty` and `environment_response_se` carry
`not_reported_by_primary` gaps. The `Stdev` column of Datasets S3 and S4 is NOT an error
bar: it is the standard deviation of the reference hybridizations, which is the z-score's
own denominator, with a heuristic global component of 0.15 added to it. The combined score
is also a minimum over the drug's z-scores, a deliberately conservative order statistic with
no released spread.

### Table 1 disagrees with the release for two drugs

Table 1 prints `# Samples` **3 for streptomycin and 2 for sulfamonomethoxine**, and the
released data says the opposite. Three independent readings of the release agree against the
printed table:

1. Datasets S2, S3 and S4 carry **two** `STR_R*` hybridization columns and **three**
   `SLF_R*` columns.
2. Table 1's own day footnote marks SLF (with DOX and LOM) as "Two samples from day 2 and
   one from day 3", which is three.
3. Text S1 sets the significance cut at 2.15 for a drug hybridized twice and 1.5 for one
   hybridized three times. In Dataset S1, STR's smallest significant `|z|` is 2.1505 with
   nothing below 2.15; SLF has 8 significant loci below 2.15, the smallest 1.5246.

The printed cells look transposed between the STR and SLF rows. `n_samples` is therefore
derived from the release, and `process()` refuses a build whose Dataset S1 values disagree
with the cut the hybridization count implies. Both disagreements are written to
`preprocess/replicate_structure.json`.

Per-drug repetition counts, measured and cross-checked (`min_abs` is the smallest
significant `|z|` in Dataset S1):

| drug | S3/S4 columns | repetitions | cut | Table 1 | n significant | min_abs | n below 2.15 |
|---|---|---|---|---|---|---|---|
| amk | 2 | 2 | 2.15 | 2 | 139 | 2.1504 | 0 |
| gen | 2 | 2 | 2.15 | 2 | 117 | 2.1525 | 0 |
| str | 2 | 2 | 2.15 | **3** | 122 | 2.1505 | 0 |
| tob | 2 | 2 | 2.15 | 2 | 433 | 2.1577 | 0 |
| lom | 3 | 3 | 1.5 | 3 | 69 | 1.5039 | 47 |
| nal | 2 | 2 | 2.15 | 2 | 65 | 2.1540 | 0 |
| slf | 3 | 3 | 1.5 | **2** | 10 | 1.5246 | 8 |
| dox | 3 | 3 | 1.5 | 3 | 56 | 1.5168 | 30 |
| tet | 3 (R2+R3 averaged) | 2 | 2.15 | 2 | 30 | 2.1686 | 0 |
| blm | 2 | 2 | 2.15 | 2 | 33 | 2.1625 | 0 |
| fox | 3 (R2+R3 averaged) | 2 | 2.15 | 2 | 42 | 2.1591 | 0 |
| ery | 2 | 2 | 2.15 | 2 | 1 | 3.3520 | 0 |
| nit | 2 | 2 | 2.15 | 2 | 14 | 2.1793 | 0 |
| amp | 3 | 3 | 1.5 | 3 | 33 | 1.5401 | 21 |
| pip | 3 (R2+R3 averaged) | 2 | 2.15 | 2 | 69 | 2.1926 | 0 |
| trm | 3 (R2+R3 averaged) | 2 | 2.15 | 2 | 17 | 2.2231 | 0 |
| fus | 2 | 2 | 2.15 | 2 | 3 | 2.6848 | 0 |

### The combination rule reproduces the release

`process()` re-derives every Dataset S5 cell from Datasets S3 and S4 by the paper's own
rule: pool the z-scores of the drug's hybridizations against both reference sets, average
repetitions 2 and 3 for the four drugs Text S1 names, drop an `LQ` array, then take the
score closest to zero when all signs agree, else 0.

**Measured: 66,240 of the 66,287 numeric cells reproduce EXACTLY, 47 are cells where the
rule gives `|z| < 0.001` and the sheet prints 0 (the largest such value is 0.000976), and 0
disagree.** `MAX_ROUNDED_TO_ZERO = 1e-3` bounds the second class; a cell outside it raises
`CombinationRuleError`. The per-drug counts are in `preprocess/combination_rule.json`.

This is a strong L3: the published Methods, the published Text S1 and three released tables
all have to be self-consistent for the build to complete.

### Identifier route, with counts

`UNIQID` holds 2009-vintage MG1655 b-numbers; every one of the 3,976 matches `b` plus four
digits. Against the pinned GCA_000005845.2 annotation (`reconcile_locus_tags`,
`ecoli_k12_mg1655_bnumber`):

| resolver layer | status | count | kept? |
|---|---|---|---|
| locus tag | current | 3,758 | yes |
| locus tag | non_gene_feature (a pseudogene locus) | 63 | yes |
| gene synonym | renamed (a synonym of another CURRENT locus) | 39 | no |
| gene synonym | non_gene_feature (a synonym of a pseudogene) | 71 | no |
| not found | retired | 45 | no |

3,821 of the 3,976 **are locus tags of the pinned annotation in their own right**, so they
are stored exactly as released and **no record carries a `DerivedIdentifierMapping`**. The
reconciliation also reports 13 remapped and 101 kept-as-given-on-collision, which is why
`reconcile_locus_tags`' own output is not what decides retention here: both groups are
b-numbers the annotation has merged away, and whether `reconcile_locus_tags` proposes the
merge target or keeps the released tag depends only on whether another released b-number
collides on the same target. The rule the loader applies is the stable one: keep the row iff
the released b-number IS a locus of the assembly.

The 155 dropped b-numbers are the issue #753 case. Storing the merged locus would need a
`DerivedIdentifierMapping`, and `DerivedIdentifierRoute` has no member for a retired tag of
the pinned strain's OWN namespace (`eck_crosswalk` crosses strains, `jw_synonym` is a Keio
id, `gene_symbol` is a name). Storing the released tag would place a record on a locus the
assembly does not have and fail L4 containment. So the rows are dropped and each b-number is
ledgered with the annotation's own resolution note in
`preprocess/identifier_reconciliation.json`. Same handling and same reason as `rapp2026`'s
`b_number_remapped_by_the_annotation` rule.

### Retention ledger (arithmetic)

```
released loci                                3,976
x antibiotics                                   17
= source cells                               67,592

- b_number_is_not_a_locus_tag_of_the_pinned_annotation
    155 loci x 17 drugs                       2,635   (2,521 numeric + 114 ND)
- no_combined_score_released
    ND cells among the 3,821 kept loci        1,191
= stored records                             63,766
```

Check: 2,635 + 1,191 = 3,826; 67,592 - 3,826 = 63,766. The ledger refuses to write unless
the rules account for every dropped cell. The full ND census over all 3,976 loci is 1,305
cells; 114 of those sit on dropped loci and are counted under the identifier rule, which is
why the two rules do not double count.

### Medium

One medium for all 17 conditions, derived from the `M9` MEDIA_LIBRARY key, so every
condition's base medium has a library entry and **no condition is dropped for a medium
reason**. `media.py` is not edited.

The four M9 salts carry `concentration=None`: the paper names them only as "M9 salts [80]",
a reference to Ausubel's Current Protocols, and prints no amounts. The `M9` library object's
amounts are Borchert 2024's and Kang 2026's, so asserting them for this paper would
fabricate numbers. The five supplements carry the paper's own doses (0.4% w/v glucose, 0.1%
w/v casamino acids, 1 mM MgSO4, 0.1 mM CaCl2, 1.5 uM thiamine). Casamino acids is
`intrinsically_undefined`, which is what makes `is_synthetic` False.

Each drug is a `SmallMoleculePerturbation` at its Table 1 dose in `ug/mL`. `mg/L` is not
needed: it is numerically identical to the `ug/mL` the enum already carries, and every dose
is printed in ug/ml anyway.

**Nine of the 17 antibiotics are not in the curated compound table** (amikacin, ampicillin,
cefoxitin, doxycycline hyclate, fusidic acid, gentamycin, nitrofurantoin, piperacillin,
streptomycin), so `resolved_compound` keeps the paper's label and attaches a typed
`deferred_pending_source_review` gap on `inchikey`. The L3 compound-identity rule passes on
the gap (33,487 of 63,766 environment-edit compound references are gapped, 9 distinct
compounds); `compound_identity_table.json` is not edited here, and adding those nine is a
separate curation decision.

### Genotype

One gene-level `TransposonInsertionPerturbation` per record. Four fields are None and all
four are typed in `PERTURBATION_FIELD_GAPS` (written to
`preprocess/perturbation_field_gaps.json`), because the leaf has no `provenance_gaps` slot:

- `barcode`: genetic footprinting counts transposon-adjacent DNA by hybridization, so the
  library has no molecular barcode at all.
- `insertion_position`, `insertion_strand`: a record is one array spot standing for every
  mutant whose transposon-adjacent DNA hybridizes to that gene, insertions NEAR it included,
  so no single site exists for it.
- `transposon`: `deferred_pending_source_review`, `resolve_with` Girgis et al. 2007
  (doi:10.1371/journal.pgen.0030154), which built the library and names the transposon and
  which is NOT in the mirror. This paper names none.

The "in or near a gene" caveat is real and worth keeping in view: the array measures a spot,
not an insertion, so a strong neighbour can contribute to a gene's score.

The library parent, MG1655 `delta-lacZ`, is a `BacterialStrainBackground` on the MG1655
assembly with `alleles=[]`. That empty list is a statement: both sources write the genotype
as a delta and neither says whether the lacZ lesion is a full deletion, an internal one or a
cassette replacement, and `AlleleEdit` has no member for an unspecified edit, so typing one
would assert a mechanism the paper never gave. The verbatim genotype string is kept in
`genotype_statement`. Same precedent as `cui2018`.

### Exposure duration is a typed gap

The selection is dated in DAYS of 2% serial transfer (Table 1's Day column) and no hours or
generations per transfer are stated; three drugs are dated "2/3*", two samples from day 2 and
one from day 3, which has no single value at all. `Environment.duration_hours` carries a
`not_reported_by_primary` gap and the day stays verbatim in the phenotype's `screen_id`
(`"amk day 4"`, `"dox day 2/3"`), which is what the environment-response verifier folds into
its condition signature. The 17 conditions have 17 distinct signatures.

### Build numbers

```
python -m torchcell.database.build_dataset_lmdb --dataset EnvChemgenGirgis2009Dataset
BUILT EnvChemgenGirgis2009Dataset: 63766 records
  at $DATA_ROOT/data/torchcell/ecoli_env_chemgen_girgis2009 in 44s;
  gene_set size 3821; references 17
```

`python -m torchcell.provenance.build_manifest` reports
`ecoli_env_chemgen_girgis2009 -> status fresh, drift []`.

### Verification

`python -m torchcell.datasets.ecoli.girgis2009 verify` runs the environment-response
L0-L4 gate (`verify_environment_response_dataset_streaming`) over the dev store with the
MG1655 resolver and the MG1655 GenBank gene universe, then audits every `SourcedValue` and
every Table 1 dose row. Result: **PASS, every row green.** Highlights:

- L0 structural: 63,766 records validated.
- L1 count 63,766 = expected; pair_uniqueness 63,766 unique (study, strain, condition).
- L1 canonical_gene_names: 3,821 systematic names, each current in the genome; the 63
  the resolver cannot place by common name are the pseudogene loci, which resolve to
  themselves.
- L2 value_fidelity on 63,766 values; L2 uncertainty_sanity: 0 labeled uncertainties, 63,766
  records reporting `n_samples >= 2` with no uncertainty, which is the honest state.
- L3 measurement_type_consistent (single `z_score`), reference_zero (reference response 0 on
  all 63,766), environment_perturbed (all 63,766 carry an edit), compound_identity (30,279
  identified, 33,487 gapped over 9 distinct compounds), media_membership (63,766 on a
  medium deriving from a library key).
- L3 provenance_audit: every sourced value and every dose row backed by its verbatim quote.
- L4 gene_containment 1.000 of 3,821; every systematic name a gene of the current genome.

### Open items

- `AssayType` has no member for microarray genetic footprinting. `assay_type=other` plus
  `units` is the honest typing today; a `genetic_footprinting_microarray` member would be
  more precise and is a schema change, so it is recorded rather than made.
- `DerivedIdentifierRoute` has no member for a retired tag of the pinned strain's own
  namespace (issue #753). 155 of 3,976 b-numbers are affected; they are dropped and
  ledgered rather than mislabeled as `gene_symbol` or `eck_crosswalk`.
- Nine antibiotics are name-only in `compound_identity_table.json`. Adding them is a
  curation decision, not part of this change.
- `transposon` is unsourced because this paper defers the library to Girgis et al. 2007,
  which is not in the mirror. Mirroring that paper would close the gap.
