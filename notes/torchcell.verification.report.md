---
id: igrwtno0fc1cqhpefguxhqw
title: Report
desc: ''
updated: 1783563821953
created: 1783563821953
---

## 2026.07.08 - Report models: the shared vocabulary that makes "verified" auditable data

This module holds the pure pydantic data models the whole framework speaks in -- `Level` (the L0-L4 enum), `LevelResult` (one check's outcome), `VerificationReport` (provenance + all results for a dataset), plus `Provenance` and `sha256_file`. It exists so a verification outcome is a serializable, re-checkable artifact (a `verification_report.json` beside each dataset), not an ephemeral console pass/fail. `VerificationReport.passed` is deliberately strict: an EMPTY report is NOT passed, so "we never ran a check" can never masquerade as success.

- `StatDerivation` + `DerivationMethod` live here too: they record how a NON-sourced statistic was fixed (back_solve / conservative_low / median), encoding the CLAUDE.md "Adding Datasets" range-resolution policy as validated data with its diagnostics attached -- the counterpart to [[torchcell.verification.sourced]]'s quote-backed constants.
- The `StatDerivation` validator enforces that each method carries the inputs that justify it (range methods need bounds and value-in-range; back_solve needs the matched statistic) -- a derivation cannot be recorded without its evidence.
- `Level` is an ordered `IntEnum` because the tiers are a gate; the check functions that PRODUCE `LevelResult`s live in [[torchcell.verification.levels]], keeping data models and logic separate.
- Consumed by the runners ([[torchcell.verification.runners]]) which serialize reports to JSON siblings of `experiment_reference_index.json`.

## 2026.10.12 - Genome-reference rules report under the verdict, never gate it

Owner, 2026-10-12: "the data is the data!!!! it is what they reported ... we don't drop raw data unless they themselves pull it! ... the genome is separate.. it doesn't gate anything... we do that at modeling time if it is an issue."

Every `LevelResult` now carries a typed `finding_class` (`FindingClass.gate` or `FindingClass.report`), a pydantic `computed_field` read from the rule's NAME by `finding_class_of`. Classifying by name, from one registry, is what makes the rule apply to every place a rule is rendered: a family verifier, a family runner's added L4 row, the shared rules of [[torchcell.verification.common]], and the loaders' own `verify_build` gates (about 40 modules, several held by other branches today), none of which had to be edited. A `model_copy(update={"name": ...})` skips validation, so a stored field could disagree with a renamed result; the computed field cannot.

- `gate`: the data's fidelity to its release. L0 structure, L1 counts and uniqueness, L2 value fidelity to the released bytes and uncertainty sanity, L3 vocabulary, reference, compound and media membership, L4 agreement with another release.
- `report`: the data compared with OUR genome reference. Each keeps its computation and its `passed` outcome; the report lists it under the verdict with its statistic.

`VerificationReport.passed` reads only the gate rules (and a report with no gate rule is not passed); `gate_results`, `findings` and `open_findings` split the rows; `summary()` prints the gate rows, then a `genome-reference findings (reported, never gated):` header with `[ok]` / `[rp]` rows. A report written before this change (no `finding_class` key) loads and is classified by name; a report whose stored class contradicts the registry is refused ("regenerate the report").

The registry (`GENOME_REFERENCE_RULES` + `GENOME_REFERENCE_RULE_PREFIXES`):

| Rule | Where | Why it is the genome reference |
|---|---|---|
| `canonical_gene_names` | shared rules | stored names against the resolver's current names |
| `current_genome_genes` | shared rules | stored systematic names against the current annotation |
| `gene_containment_*` (prefix) | shared rules, runners, loaders | measured ids inside an annotation's gene set, or another store's deletion set keyed by resolved names |
| `perturbed_gene_containment_*`, `protein_and_perturbed_locus_containment_*` (prefixes) | bioproduction runners, Ishii 2007 | ids inside the pinned assembly |
| `assembly_pin*` (prefix) | bacterial loaders | the record's pinned assembly is the genome reference itself |
| `marker_reference`, `variant_reference` | segregant verifiers | released reference allele against the R64 base |
| `mat_fixation`, `mat_orientation`, `cis_concordance` | segregant verifiers | genotype at the marker nearest the annotation's MAT, or at each gene's annotated midpoint |
| `parents_pinned` | segregant verifiers | parent assemblies in the genomes tier |
| `induced_gene_containment`, `s288c_gene_containment`, `pangenome_orf_containment`, `feature_containment`, `deleted_gene_containment_sgd` | family verifiers, Hu 2007 | ids against S288C, the pangenome, or R64 loci |
| `promoter_gene_is_a_locus`, `promoter_genes_in_the_host_gene_universe` | promoter verifier, Mohiuddin 2022 | promoter genes resolving to themselves |
| `stored_tags_are_loci`, `stored_tags_are_loci_of_the_pinned_assembly`, `stored_targets_are_loci_of_the_pinned_assembly`, `stored_protein_keys_are_loci_of_the_pinned_assembly`, `stored_loci_are_bw25113_loci_derived_from_the_released_b_number` | bacterial loaders | stored tags against the pinned assembly |

Rules beside these that read no annotation stay gates: `measured_genes_present` (does any record carry a gene perturbation at all), `fitness_companion_containment` (keys of two arms of one release), `caudal_genotype_agreement`, `guide_targets` (served guide ORF against the release's SD3), `strain_background` and `reference_genome_is_the_named_denominator_strain` (the strain the authors report), `orf_uniqueness` and the `one_record_per_*_locus` rows, `released_annotations`, `no_viable_deletion_in_smf_costanzo2016`.

`torchcell/database/build_dataset_lmdb.py --verify` decides on `report.passed`, so a genome finding never retires a build manifest; the PASS line counts the findings. Tests: `tests/torchcell/verification/test_finding_class.py` (registry pin, both classes, verdict, summary, JSON round trip, old-report load, contradiction refused, rename reclassified, the shared rules' classes) and `test_verify_keeps_the_manifest_when_only_a_genome_rule_fails` in `tests/torchcell/database/test_build_dataset_lmdb.py`.
