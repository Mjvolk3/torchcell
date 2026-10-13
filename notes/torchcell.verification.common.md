---
id: ecrh5qmk9sefr007o482avp
title: Common
desc: ''
updated: 1790872920010
created: 1790872920010
---

## 2026.10.01 - Empty gene set passes containment; American spelling

Previously, with `sgd_genes` given and no gene perturbations, `gene_containment_sgd` defined the overlap as 0.0 and failed ("0.000 of 0 measured genes ...") while `current_genome_genes` passed the same empty set. The overlap of an empty measured set is now 1.0 (vacuous containment), the result passes, and its message reads "no measured genes (the measured gene set is empty); containment holds vacuously". The `uncertainty_sanity` messages now spell "labeled". No code parses these messages; historical dataset notes that quote old reports were left verbatim. The same 0.0 convention still exists in `runners._l4_rnaseq_gene_containment` and `segregant_growth`, outside this fix. Issue #541; tests `test_gene_containment_passes_a_dataset_with_no_genes_vacuously`, `test_uncertainty_*` in [[tests.torchcell.verification.test_common]].

## 2026.10.01 - Review fix: measured_genes_present

The vacuous pass above removed the only failing result for a dataset whose records carry no gene perturbation (review of PR #588: three wild-type records at three temperatures, or one wild-type record, passed with nothing failing). When `sgd_genes` is given and the measured set is empty, a FAILING L4 `measured_genes_present` result now precedes the two vacuous passes ("no measured genes: none of the N records carries a gene perturbation outside the background genes, ..."), and `current_genome_genes` also names the empty set. The row is emitted only for an empty set, so reports of datasets that measure genes keep their rows. Per the reviewer, none of the 43 stored reports carrying `gene_containment_sgd` has `n_measured == 0`, so no stored `passed` flag would flip. Tests: `test_gene_containment_with_no_genes_fails_on_measured_genes_present`, `test_background_only_genes_count_as_no_measured_genes`, `test_a_dataset_with_no_gene_perturbations_fails_for_no_measured_genes`.

## 2026.10.10 - Helpers for the single-pass family verifiers (#889)

- `declared_member_validator(class_name, union=...)`: validates a stored mapping as the one union member a registry declares, after checking its type tag. Validating a Costanzo 2016 record against the 33-way `ExperimentType` union costs 12 ms (85 records per second, 68 hours for 20,705,612 records); against the declared member, 0.4 ms.
- `key_digest`: 16-byte blake2b of a signature's `repr`, for uniqueness sets over 20M records.
- `l0_validated_row`: the L0 row of a single-pass verifier; an empty store fails.

## 2026.10.11 - The typed medium gap was reverted; media_membership is unchanged

A section here recorded L3 `media_membership` counting an environment whose `media` was None beside a `ProvenanceGap(field='media')`. `Environment.media` stays required (owner decision, [[torchcell.datamodels.schema]]), so no environment can reach the rule in that shape and the rule is back to its `origin/main` form: a record with no joinable medium fails. The three stores that motivated the change are provenance records now, not datasets ([[torchcell.datasets.scerevisiae.sgd]], [[torchcell.datasets.scerevisiae.synth_leth_db]]).

## 2026.10.12 - media_membership reads a gapped Media (#901, #902, #903)

`Environment.media` stays a required `Media`, and a medium the source names none of carries `name` `None` beside a `ProvenanceGap(field='name')` ([[torchcell.datamodels.schema]]). `_media_verdict` reads that: an unnamed medium with a `name` gap is `gapped:<reason>` and passes, counted in the message ("N on a medium the source does not name, as a typed gap (reason xN)") and in `details` (`n_gapped_records`, `gapped_reasons`); an unnamed medium with no `name` gap, or no medium at all, stays free text and fails. A medium that is named but gaps only its state (SGD's "YPD with 2% glucose" on the YPD base) is judged by its name and `base_medium` like any other. Pinned by `test_media_membership_passes_a_typed_medium_gap_and_counts_it`.

## 2026.10.12 - The gene rules report, the rest gate

`canonical_gene_names` (L1), `gene_containment_sgd` and `current_genome_genes` (L4) compare stored names with OUR genome reference, so they are `report`-class rules ([[torchcell.verification.report]], owner 2026-10-12: "the genome is separate.. it doesn't gate anything"). Their computation and `passed` outcomes are unchanged; a store whose only non-holding rows are these PASSES and lists them under the verdict. `measured_genes_present` reads no annotation (it asks whether any record carries a gene perturbation outside the background genes) and stays a gate, as do `provenance_gaps`, `uncertainty_sanity`, `compound_identity`, `media_compound_identity` and `media_membership`. Only docstrings changed in this module; the class is assigned by the registry in `report.py`.

## 2026.10.13 - declared_first_union_validator: the union's L0 verdict at the declared member's cost

`TypeAdapter(ExperimentType)` is a 46-member union with no discriminator (each member's `experiment_type` is a plain `str` with a default), so pydantic tries every member on every stored record. Measured on bounded slices built into temporary roots (`RbTnseqBorchert2024Dataset`, 6 samples, 28,392 records; `CrispriChemgenChoe2025Dataset`, 1,000 guides, 11,937 records): the union validates 73 and 79 records per second; `BacterialEnvironmentResponseExperiment.model_validate` alone validates 3,390 and 3,604. Under cProfile the union call was 673 s of the 695 s streaming verifier on the Borchert slice (97%), and the verifier was 5 h 24 min of the 5 h 32 min Borchert job (slurm 4011_103: the build itself, `process()` plus `post_process`, took 455 s).

`declared_first_union_validator()` validates the one member whose tag default equals the record's `experiment_type`; a success is returned (a member's success is the union's), and anything else (an invalid record, an unknown or shared tag, a non-mapping) goes to the union, so its verdict and its error text are the union's own. The one theoretical difference is a record the declared member accepts while an earlier member's validator raises a non-validation error out of the union; such a record fails L0 (or stops the verifier) today. It is the drop-in for the union in the eager and streaming environment-response verifiers ([[torchcell.verification.environment_response]]); `declared_member_validator` stays the stricter one-class gate the fitness, interaction, essentiality and synthetic-lethality verifiers use.

Equivalence: each slice built and verified before and after the change (before on `main` 6982d2437, after on the branch) gives byte-identical LMDB records and interned entries (28,392 and 11,937 records) and a byte-identical `verification_report.json` (Borchert's after normalizing `dataset_name`, which `verify_build` takes from the root's basename). Verifier wall time on the slices: Borchert 426.9 s to 16.4 s, Choe 176.8 s to 8.9 s. Full stores built from the mirror copies into temporary roots on the branch: Borchert 1,466,920 records, build 422 s and verification 979 s (was about 19,465 s in slurm 4011_103); Choe 470,502 records, build 262 s and verification 254 s (was about 4,650 s in slurm 4011_50). Both PASS, and the L0 row reads `N records validated` as before. Pinned by `test_declared_first_validator_returns_the_union_verdict_and_text` and `test_declared_first_validator_leaves_both_reports_byte_identical` in `tests/torchcell/verification/test_environment_response_verification.py`.

## 2026.10.13 - declared_first_union_validator in four more verifiers

The union L0 validation is swapped for `declared_first_union_validator()` in `verify_fitness_dataset` (Tong 2020, Rachwalski 2024), `verify_natural_isolate_response_streaming` (Peter 2018), `verify_segregant_expression_streaming` (Boocock 2025) and `verify_promoter_activity_dataset_streaming` (Mohiuddin 2022), the four verifiers behind the slowest 4011 verifications the environment-response change did not already cover. Each store's first 5,000 records were copied read-only from the dev store into a temporary root and verified on `main` and on the branch: the reports are byte-identical (modulo the slice root's own name) for Tong, Peter, Boocock and Mohiuddin, and for the environment-response stores the earlier change speeds up (Mutalik 2020, Wetmore 2015, Wang 2018, Shiver 2016). Pinned by `test_fitness_l0_keeps_the_union_report_with_the_declared_member_first` and `test_promoter_l0_keeps_the_union_report_with_the_declared_member_first`.

Left alone, measured: the Costanzo 2016 and Kuzmin 2018/2020 builds spend their time in the build (4011_1: 16,545 s of 16,556 s; 4011_2: 15,875 of 15,887; Kuzmin runs carry no verifier), Hoepfner 2014 and Piotrowski 2017 validate once per screen column inside `process()` (Hoepfner: 602 columns in a 1,021 s build), and Hale 2024's verifier already validates its declared class with memoized sub-objects; its 5,654 s is the per-record gap-carrier walk over each record's segregant background (`_walk_carriers`, about 1,400 recursive calls per record on a 5,000-record cProfile slice) plus 32 s of fixed table reads.
