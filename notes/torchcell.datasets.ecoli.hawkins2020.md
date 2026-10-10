---
id: tl671m4ci740ygw7f7f7roy
title: Hawkins2020
desc: ''
updated: 1791615480746
created: 1791615480746
---

## 2026.10.10 - Mismatch-CRISPRi loaded as 24,149 measured relative fitnesses; the predicted dose refused

Row 51 of the ranked bacterial candidate table (`experiments/database/scripts/build_bacteria_candidate_datasets_table.py`), E. coli, doi:10.1016/j.cels.2020.09.009, mirrored as `hawkinsMismatchCRISPRiRevealsCovarying2020`. `MismatchCrispriFitnessHawkins2020Dataset`, store `mismatch_crispri_fitness_hawkins2020`.

### What the release separates, and what this loader stores

Table S3's E. coli sheet (`si/si4.xlsx`, sha256 `a9aa39f5...412a`, publisher `mmc4.xlsx`) carries the measurement and the model side by side, measured on the pinned bytes by `experiments/036-dataset-fixes-before-kg-build/scripts/hawkins2020_release_inventory.py`:

| column | what it is | stored |
|---|---|---|
| `relative fitness (mean)` | mean of 4 biological replicates of relative fitness after about 10 doublings | yes, as `EnvironmentResponsePhenotype.environment_response` |
| `relative fitness (stddev)` | SD of those replicate values | yes, `sample_sd` with `n_samples=4` |
| `relative fitness (predicted)` | the linear model's predicted sgRNA ACTIVITY, exactly 1.0 on all 3,110 parent spacers and -0.151 to 1.400 on the 33,180 singly mismatched ones | NO (schema finding) |
| `family_retained` | the paper's own curve-pipeline series filter | not applied; kept per row in `preprocess/guide_retention.csv` |

The refused column is the row's written schema need, "a graded knockdown whose level is a prediction". It is a model output at species-averaged R-squared 0.56 with an 11-fold cross-validation mean squared error of 0.10 plus or minus 0.08, and a preliminary version of that same model chose the library's mismatches, so the dose is both imputed and entangled with the design. No field on `BacterialCrisprInterferencePerturbation` or `CrisprConstruct` carries a perturbation-level covariate with its own error, so storing it anywhere available would present a prediction as a measurement.

What the record carries instead is the part of the dose that IS measured: the guide's mismatch design, on the perturbation's `description` (the parent spacer, the 0-based mismatch position from the 5' end, and the base substitution), with the 20-nt mismatched spacer on `crispr.guide_sequence`. The Mutalik 2020 loader sets the same field the same way.

### Counts

| quantity | value |
|---|---|
| released (sgRNA, gene) rows | 36,290 |
| parent spacers / singly mismatched | 3,110 / 33,180 (Hamming 0 or 1 in every row) |
| released genes (MG1655 b-numbers) | 318 |
| dropped: `sgrna_has_no_released_fitness` | 12,104 (blank mean; the authors' own 100-counts-at-t0 floor) |
| dropped: `target_gene_has_no_bw25113_locus` | 37 (`sokE` / `b4700`, retired in ASM584v2, no ECK synonym) |
| STORED records | 24,149 over 317 genes and 2,948 parent spacers |
| of those, fully complementary / singly mismatched | 2,378 / 21,771 |
| stored values below 0 (active depletion) | 778 |
| stored records with no released SD | 1,870 |
| stored records in a series the paper's curves exclude | 4,863 |

### Identifiers: a BW25113 host named by MG1655 b-number

Every library strain was built in wild-type BW25113 (strain CAG78830, BW25113 Tn7att::PlLac-O1-dcas9(Gent)), while Table S3 names targets by b-number. Each b-number is carried to its BW25113 locus through the one-to-one ECK synonym join and typed on the leaf as `DerivedIdentifierMapping(route="eck_crosswalk")`. The crosswalk is checked rather than trusted: the workbook's own `10-15 relfit (eco)` sheet names the same rows by `BW25113_` tag and all 317 crossed tags agree with the authors' tag (0 disagreements).

### The 10-to-15-doubling sheet is refused

`10-15 relfit (eco)` is deposited and not loaded, on three measurements: its `relative fitness (stddev)` column is byte-identical to the 10-doubling sheet's in all 22,313 rows where both are present; its 1,000 non-targeting controls spread 2.3-fold wider (SD 0.1935 against 0.0825, the latter reproducing the paper's stated noise floor as 0.08254); and the two windows correlate at Pearson r 0.598 over the 18,780 sgRNAs both measure. The Methods describe one 10-doubling design for E. coli and no 15-doubling sampling, so no replicate count can be sourced for the late window.

### Duplication verdict: independent

Overlap with the two served E. coli CRISPRi screens is on fully complementary spacers only: 955 shared with Wang 2018's essentiality screen (Pearson r 0.662) and 364 with each Cui 2018 screen (r 0.640 and 0.669), out of 24,149 stored records. The workbook's own comparison columns agree (957 rows against Wang at r 0.663, 290 against Rousset at r 0.663, and 0 of either on a mismatched guide). 21,771 stored records are mismatched guides that exist in no other release, the host is BW25113 rather than MG1655, and the paper states the design difference the correlation reflects. Loaded whole.

### Verifier: a third reference baseline

The readout is a RATIO against a control measured in the same run ("Strains with a relative fitness of 1 grow as well as the wild-type does"), so its reference is 1.0, not 0. The environment-response verifier's `reference_zero` rule assumed a relative readout is 0 at its control, and its absolute relief explicitly refuses a relative type, so neither branch fit. Added, additively and gated the same way: `RATIO_MEASUREMENT_TYPES` (only `relative_growth_rate`) plus `RATIO_REFERENCE_VALUE` in the schema, and a `reference_unit_scaled=True` branch in both the eager and the streaming verifier that requires every reference to be exactly 1.0 and every record's `measurement_type` to be in that set. Schema impact: both new symbols are new module-level bindings, impacted datasets none.

### L0 to L4 on the built dev store (24,149 records)

| level | row | verdict |
|---|---|---|
| L0 | structural | PASS, 24,149 records validated |
| L1 | count | PASS, observed 24,149 = expected |
| L1 | pair_uniqueness | PASS, 24,149 unique (study, strain, condition) |
| L1 | provenance_gaps | PASS, 102,206 documented gaps over 24,149/24,149 records |
| L1 | canonical_gene_names | PASS, 317 names, one spelling each, each current |
| L1 | stored_loci_are_bw25113_loci_derived_from_the_released_b_number | PASS, 317 loci, 24,149 of 24,149 perturbations carry the ECK mapping |
| L2 | value_fidelity | PASS, 24,149 values |
| L2 | se_nonnegative | PASS, 22,279 values |
| L2 | interval_orientation | PASS, 0 of 0 |
| L2 | uncertainty_sanity | PASS, 22,279 labeled uncertainties, 1,870 records with n >= 2 and none |
| L2 | guide_spacers_are_twenty_nt_acgt | PASS, 24,149 distinct spacers, 0 malformed |
| L2 | perturbation_descriptions_state_the_mismatch_design | PASS, 0 without a readable design |
| L3 | measurement_type_consistent | PASS, single `relative_growth_rate` |
| L3 | reference_zero | PASS, ratio rule: reference == 1 for all 24,149 |
| L3 | environment_perturbed | PASS, every record carries the IPTG edit |
| L3 | compound_identity / media_compound_identity / media_membership | PASS |
| L3 | one_screen_and_the_negative_tail_is_intact | PASS, 778 negative values intact |
| L4 | gene_containment_sgd | PASS, 1.000 of 317 |
| L4 | current_genome_genes | PASS, 317 of 317 |

### Environment

LB with 100 ug/mL ampicillin at 37 C, 1 mM IPTG inducing the chromosomal dcas9, over about 10 doublings of exponential growth maintained by back dilution. The LB formulation is unstated anywhere in the paper, so the three ingredients carry no amounts; the ampicillin is a medium component (selection, present in every record and in the reference) and the IPTG is the one `Environment.perturbation`. The E. coli Methods state their culture as the B. subtilis one with two changes, so the 10 doublings are sourced through that deferral and the xylose it names is what IPTG replaces.

### Owner decisions taken by recommendation

1. The predicted sgRNA activity is refused and filed as a schema finding rather than stored in `units`, `screen_id` or a note. Every value stays in the ledger CSV, so a later field can be backfilled from the same bytes.
2. `family_retained=False` records are STORED (4,863 of 24,149). The filter bears on whether a series' predicted activity scale is trustworthy, which is the quantity already refused, not on whether the fitness was measured; the flag is per row in the ledger so the curve population is reconstructible.
3. The screen's released noise floor (SD 0.0825 over the 1,000 non-targeting controls) is recorded as a sourced value and NOT stored as the reference's uncertainty: its samples are control strains, which no `SampleUnit` member names.
4. No PubMed id is recorded; the mirror's manifest and the pinned OCR carry only the DOI.

Regenerating script: `experiments/036-dataset-fixes-before-kg-build/scripts/hawkins2020_release_inventory.py`, results `experiments/036-dataset-fixes-before-kg-build/results/hawkins2020_release_inventory.json`.
