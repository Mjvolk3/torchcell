---
id: ldhf4a9izm4dc0r8ft3o3a9
title: Mean_experiment_deduplicate
desc: ''
updated: 1727813635189
created: 1727813635189
---

## 2026.09.30 - Uncertainty Pooled by Statistic (Issue #532)

The deduplicator runs in the experiment dataset builds (`Neo4jCellDataset` and the 0xx `*_build` trees), not in the served Neo4j graph. A duplicate group (same experiment type and sorted perturbed genes) is merged into one record.

### Before

Every per-key uncertainty dict in the vector families went through one RMS pool, `sqrt(mean(x_i^2))`, over the records that carried it:

- `expression_log2_ratio_se` (microarray), `metabolite_level_se`, `protein_abundance_se`: RMS of the SEs.
- `expression_log2_ratio_variance` (microarray): RMS of the variances, so 0.5 and 0.1 gave 0.3606 instead of 0.3.
- A key whose value was averaged from records of which only some carried the statistic got the carriers-only pool. For microarray this produced an SE dict covering fewer keys than the log2 dict, which the phenotype validator rejects (`expression_log2_ratio_se must have the same keys as expression_log2_ratio`).

### After

Each field is routed by the statistic it is:

| field | phenotype | rule |
|---|---|---|
| `fitness_std` (scalar) | fitness | SD: RMS pool `sqrt(mean(sd_i^2))`, unchanged |
| `expression_log2_ratio_se` | microarray | SE of the mean of means `sqrt(sum(se_i^2)) / m` |
| `metabolite_level_se` | metabolite | SE of the mean of means |
| `protein_abundance_se` | protein abundance | SE of the mean of means |
| `expression_log2_ratio_variance` | microarray | arithmetic mean `mean(var_i)` |
| `calmorph_coefficient_of_variation` | CalMorph | arithmetic mean, unchanged |

The SE rule is the one the RMS-pooled SD implies: a pooled SD over `m` records of `n` replicates each gives `SE = pooled_sd / sqrt(m n)`, and with `se_i = sd_i / sqrt(n)` that is `sqrt(sum(se_i^2)) / m`. It keeps the schema identity `variance = SE^2 * n_replicates` when the variances are averaged and `n_replicates` summed (for equal `n`).

Coverage: a key keeps its uncertainty only when every record averaged into its value carries one; the uncertainty of a mean with an unreported component is unknown. The metabolite and protein validators allow an SE on a subset of keys, so such a key is left out of the SE dict (for example a Messner 2023 knockout, `protein_abundance_se=None`, merged with a Zelezniak 2018 kinase knockout of the same ORF that carries an SE: merged SE None, previously the Zelezniak SE alone). With the committed loaders this drop is reachable only for protein abundance: no two metabolite loaders share a metabolite key (Lopez `isobutanol`, Cachera `betaxanthin`, Mulleder amino-acid names, Zelezniak TSV metabolite ids, DaSilveira lipid ids) and each emits one record per gene, so the metabolite branch is the helper's contract, not a live case. The Messner and Zelezniak pair also merges different `measurement_type` values (`swath_ms_maxlfq_batch_corrected_quantity` and `swath_ms_label_free_log_signal_sva`) under the first record's label, a pre-existing hazard this fix does not change. The microarray validator requires full coverage, so microarray records that disagree on carrying `expression_log2_ratio_se` or `expression_log2_ratio_variance` are refused with `ValueError("Cannot merge microarray duplicates: <field> is present on k of n records, ...")`. No loader produces that mix today: Kemmeren and Sameith experiment phenotypes always carry both, their references carry neither.

Evidence: `tests/torchcell/data/test_mean_experiment_deduplicate.py`, `test_dict_helpers_mean_pool_and_sum_over_the_key_union`, `test_microarray_variance_is_averaged_and_se_is_the_se_of_the_mean`, `test_microarray_merge_refuses_records_that_disagree_on_carrying_an_uncertainty`, `test_metabolite_se_is_dropped_where_a_contributing_record_has_none`, and the updated microarray, metabolite and protein closed forms.

### Follow-ups Left Open

- The scalar fitness SD still pools over carriers only: stds 0.3 and None give 0.3 (`test_fitness_merge_skips_missing_stds_and_keeps_the_first_records_context`), the carriers-only semantic this fix rejects for the vector families.
- The fitness merge builds `FitnessPhenotype(fitness, fitness_std)` only, so `fitness_se` and the `fitness_uncertainty` fields are dropped on merge.
- Protein merges can combine different `measurement_type` values under the first record's label (Messner 2023 with Zelezniak 2018).
