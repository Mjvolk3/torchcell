---
id: iwieqze1ae069druc5ovm1m
title: Runners
desc: ''
updated: 1783563836060
created: 1783563836060
---

## 2026.07.08 - Runners: the importable harness that runs every family's L0-L4 gate and writes reports

This module is the executable top of the framework: it knows WHICH built LMDBs exist, pins each one's `Provenance` and expected-count oracle, loads its records, dispatches to the matching per-family verifier, adds the cross-source L4 checks, and writes a `verification_report.json` beside each dataset. It was moved out of `scripts/` into `src` (`d457b5d8`) so the runners can be imported, tested, and reused -- it REPLACES the old `scripts/verify_expression_datasets.py` + `scripts/verify_morphology_datasets.py`, which could only be run, never composed.

- The per-family dataset registries (expression WS5, morphology WS6, visual-score WS7, metabolite WS8, protein/metabolite WS9, rnaseq WS10) are the single place each source's provenance + count oracle is declared -- provenance travels with the run, not the loader.
- Owns the L4 cross-source assertions that no single verifier can make: expression datasets share one platform gene universe; deletion screens are gene-contained in Ohya's morphology set; RNA-seq genes are contained in the S288C SGD reference. Per-family verifiers only expose the gene-set key; the runner joins them.
- Reads the Ohya LMDB from the (possibly read-only) KG-build tree but WRITES reports to the writable `data/torchcell/...` tree -- decoupling verification output from the build user's ownership.
- `main`/`run_all` returns a shell exit code, so the whole abstract's data can gate CI. Uses the checks in [[torchcell.verification.levels]] and the models in [[torchcell.verification.report]].

## 2026.09.14 - S288C gene universe and the Bloom member index from the genomes tier

`SGD_GENE_FASTAS` holds filenames now and `_sgd_gene_set` resolves them from the tier
set `sgd_S288C_R64-4-1_20230830` (7,146 names, unchanged); the Bloom entry's
`assembly_index` is resolved from `peter2018_1011_assemblies`. `_genome` still passes
`genome_root=data/sgd/genome` as the cache root with `overwrite=False`. See
[[torchcell.sequence.genome.registry]].

## 2026.09.23 - Issue 92 closed by rebuilds: all 11 datasets pass L0 again

Issue #92 reported L0-structural failing on 11/11 pre-2026-07-14 datasets: `Media.is_synthetic`
became a required field with no default in `1cf60cdc` (2026-07-14), so every LMDB built before
that date stored a `media` dict the current `ExperimentType` union no longer accepts, and
`l0_structural` rejected 100% of records with
`loc: ('Experiment', 'environment', 'media', 'is_synthetic') type: missing`.

All 11 dev-tree stores have since been rebuilt (LMDB `data.mdb` mtime 2026-09-14), and the
verifiers now pass every level on every one of them. Re-run today with
`PYTHONPATH=. python scripts/verify_datasets.py`, which narrows the family registries in this
module to exactly the 11 slugs, calls `run_expression`, `run_morphology`,
`run_morphology_ohnuki`, `run_ohnuki_morphology`, `run_visual_score` and `run_metabolite`, and
tabulates the `verification_report.json` each one wrote:

| dataset | records | L0 | L1 | L2 | L3 | L4 | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| microarray_kemmeren2014 | 1484 | PASS | PASS | PASS | PASS | PASS | PASS |
| sm_microarray_sameith2015 | 82 | PASS | PASS | PASS | PASS | PASS | PASS |
| dm_microarray_sameith2015 | 72 | PASS | PASS | PASS | PASS | - | PASS |
| scmd_ohya2005 | 4718 | PASS | PASS | PASS | PASS | PASS | PASS |
| scmd_ohnuki2018 | 1112 | PASS | PASS | PASS | PASS | PASS | PASS |
| scmd_ohnuki2022 | 1979 | PASS | PASS | PASS | PASS | PASS | PASS |
| carotenoid_ozaydin2013 | 4474 | PASS | PASS | PASS | PASS | PASS | PASS |
| betaxanthin_cachera2023 | 4719 | PASS | PASS | PASS | PASS | PASS | PASS |
| amino_acid_mulleder2016 | 4678 | PASS | PASS | PASS | PASS | PASS | PASS |
| metabolite_zelezniak2018 | 95 | PASS | PASS | PASS | PASS | PASS | PASS |
| metabolite_dasilveira2014 | 127 | PASS | PASS | PASS | PASS | PASS | PASS |

`dm_microarray_sameith2015` shows `-` at L4 because it is the reference gene universe the
expression L4 compares the other two microarray datasets against, so no L4 result is recorded
on it; its L0-L3 all pass. Every other dataset carries an L4 gene-containment or gene-universe
check. Reports written 2026.09.23 19:07-19:09 under
`$DATA_ROOT/data/torchcell/<slug>/preprocess/verification_report.json` (`DATA_ROOT` =
`/scratch/projects/torchcell-scratch`), which is the writable dev tree, never `database/`:

- `microarray_kemmeren2014`, `sm_microarray_sameith2015`, `dm_microarray_sameith2015`
- `scmd_ohya2005`, `scmd_ohnuki2018`, `scmd_ohnuki2022`
- `carotenoid_ozaydin2013`
- `betaxanthin_cachera2023`, `amino_acid_mulleder2016`, `metabolite_zelezniak2018`,
  `metabolite_dasilveira2014`

**No back-compat shim was added, and that is deliberate.** Issue #92's option 2 was a
`model_validator(mode="before")` on `Media` that defaults or infers `is_synthetic` when the
stored dict omits it. Nothing needs it: `is_synthetic` is still required with no default and
`Media` carries no before-validator, so L0 passing on the stored records is itself the proof
that every rebuilt store writes the field. Adding the shim now would be a silent fallback that
lets a genuinely pre-2026-07-14 store keep loading while reporting green, which is exactly the
staleness the L0 gate exists to surface. `python -m torchcell.provenance.build_manifest`
independently reads all 11 as `fresh`; the datasets it still reads `[STALE]` on the media
closure (the four `dmf/dmi_costanzo2016` slices) are the Costanzo rebuild, tracked separately.

## 2026.10.07 - run_product_titer and run_bacterial_protein_abundance: the bioproduction families run from run_all

`run_all` ran eleven families and neither bioproduction family was one of them, so the
Carruthers 2025 isoprenol levels lived in `tests/torchcell/datasets/pputida/test_carruthers2025.py`
and ran only when someone ran that file. This closes item 4 of
[[torchcell.datasets.pputida.carruthers2025]]'s open items. Two runners were added,
`run_all` calls thirteen families now, and the Carruthers battery moved out of the test
file into the loader module.

### Why these two runners DISPATCH where the yeast runners CONFIGURE

A yeast runner holds its datasets' `Provenance` and count oracle in a spec dict and calls
one shared family verifier. That shape does not fit either bioproduction family, because
every one of their datasets already carries its own entry point in the loader module that
owns its readers:

| dataset | entry point | what only it can do |
|---|---|---|
| `isoprenol_titer_carruthers2025` | `carruthers2025.verify_build(..., family="titer")` | re-reads Supplementary Data 1, a DIFFERENT released file from the one the loader consumed |
| `isoprenol_titer_desiqueira2025` | `desiqueira2025.verify_build(..., family="titer")` | compares Table S2's maximum against the titer the Results prose states |
| `isoprenyl_acetate_titer_kang2026` | `kang2026.verify_build` | re-derives the fed-batch sum from Table S9 and the Table 1 titers from the Results quotes |
| `proteome_carruthers2025` | `carruthers2025.verify_build(..., family="proteome")` | re-reads the released Top3 sheet and reproduces the stored profile per protein |
| `proteome_desiqueira2025` | `desiqueira2025.verify_build(..., family="proteome")` | the (strain, environment) uniqueness its per-condition design needs |
| `proteome_lim2025` | `lim2025.run_proteome_verification` | re-audits every `SourcedValue` quote against pinned library bytes |
| `proteome_caglar2017` | `caglar2017.run_verification("proteome")` | the VST log-base back-solve against DESeq2's own size factors |

None of that can be written once for the family: the readers, the pinned sheets and the
quote audits are per release. So each runner calls the dataset's own entry point, adds the
one rule that IS the family's own, writes the merged report with `_write_report` and prints
the summary. `run_protein` was left to the yeast proteomes: a host-aware branch inside it
would have had to re-state L4 containment that all four bacterial entry points already
assert against their own pin, and `verify_protein_dataset` is already host-agnostic (the
bacterial class shares `ProteinAbundancePhenotype`), so the four bacterial datasets go
through the shared gate from inside their own modules instead of from a second branch.

### The rule each runner adds

`host_perturbed_gene_set` is the perturbed identifiers MINUS the heterologous ones. A
`HeterologousPathwayPerturbation` carries `source_organism`, and when that organism is not
the record's own species the identifier is a gene of another genome (`MvaSEf`, `ATF1`),
which is exactly what the class exists to say, so checking it against the host locus
universe would fail every production record for the wrong reason. An extra copy of a
NATIVE gene names its real locus tag and stays in.

- `run_product_titer` -> `perturbed_gene_containment_assembly` over that set.
- `run_bacterial_protein_abundance` -> `protein_and_perturbed_locus_containment_assembly`
  over its union with the quantified protein keys. Both halves are identifiers the records
  claim are loci of their own pin, so the union is the whole L4 question and is never empty
  (Caglar's arms are environmental, so it perturbs no gene and quantifies 4,196 proteins).

`min_containment` is 1.0 in both, not a floor with headroom: every identifier these records
carry was written by a loader that resolved it against the pinned assembly, so one that is
not a locus of that assembly is a build error.

### The new shared titer verifier

`torchcell/verification/product_titer.py` is the family verifier the three titer loaders
were each hand-rolling (de Siqueira's `titer_levels` said "there is no shared titer verifier
yet"; Kang's `verify_build` assembled the same levels inline). Ten rules, every one of them
the same rule for every dataset of the family, parameterized by what the dataset's own note
pins:

1. L0 `structural`, 2. L1 `count`, 3. L2 `value_fidelity` (titers finite, >= 0),
4. L2 `uncertainty_nonnegative` (over the RELEASED uncertainties only),
5. L2 `se_is_the_uncertainty_over_sqrt_n`, 6. L3 `titer_unit_is_the_pinned_unit`,
7. L3 `uncertainty_is_typed_or_gapped`, 8. L3 `replicate_design_is_sourced_or_gapped`,
9. L3 `heterologous_pathway_gene_counts`, 10. L3 `product_is_the_declared_one`.

Rule 5 is the generalization of Carruthers' `titer_se == SD/sqrt(n)` identity, and it is one
rule rather than two because it is one question: a record that releases an uncertainty is
checked value by value at the tolerance its note states (1e-9 for Carruthers, the float
noise of deriving it from two stored numbers), and a record that releases none is checked
for having derived nothing from nothing, since a `titer_se` with no uncertainty behind it is
a fabricated precision. Carruthers and de Siqueira pin unit decisions their notes record
(mg/L stored verbatim as the numerically identical ug/mL; Table S2's released mM), and
Carruthers' 465-versus-472 discrepancy is asserted as its documented reconciliation rather
than accepted from either side: `strain_count_reconciles_with_the_papers_472` checks that
the 465 `(construct, cycle)` strains plus the seven that carry six replicates are the
paper's 472.

Kang 2026 and de Siqueira 2025 still hand-roll their own L0-L3; collapsing them onto the
shared verifier changes their published report content without changing a record, so it is
its own change and not this one.

### Verbatim output, product titer

```
isoprenol_titer_carruthers2025: PASS
  [ok] L0 structural: 465 records validated
  [ok] L1 count: observed 465, expected 465
  [ok] L1 strain_count_reconciles_with_the_papers_472: 465 strains + 7 with six replicates = 472 (the paper's 472)
  [ok] L2 value_fidelity: 465 values checked
  [ok] L2 uncertainty_nonnegative: 465 values checked
  [ok] L2 se_is_the_uncertainty_over_sqrt_n: 465 pairs agree within 1e-09 (titer_se == titer_uncertainty / sqrt(n_samples)); 0 records release no uncertainty and store no titer_se
  [ok] L3 titer_unit_is_the_pinned_unit: stored units ['ug/mL']; Supplementary Data 1 and the Source Data release mg/L and ConcentrationUnit has no mg/L member; 1 mg/L == 1 ug/mL exactly, so the released number is stored verbatim under the numerically identical unit
  [ok] L3 uncertainty_is_typed_or_gapped: an uncertainty number and its type are both stored, or both named in provenance_gaps; a number without its type is unreadable
  [ok] L3 replicate_design_is_sourced_or_gapped: a replicate count and what one replicate IS are both stored, or both named in provenance_gaps
  [ok] L3 heterologous_pathway_gene_counts: per-record heterologous pathway gene counts [5]; the dataset declares [5]
  [ok] L3 product_is_the_declared_one: stored products ['isoprenol']; the dataset titers ['isoprenol']
  [ok] L4 single_guide_titer_vs_supplementary_data_1: 120 overlapping entities agree within 0.005
  [ok] L4 perturbed_gene_containment_assembly: 1.000 of 121 measured genes are loci of pputida_KT2440_ASM756v2 (>= 1.0)

isoprenol_titer_desiqueira2025: PASS
  [ok] L0 structural: 2 records validated
  [ok] L1 count: observed 2, expected 2
  [ok] L1 strain_condition_uniqueness: SUPPLEMENTARY: 2 distinct (strain, environment) pairs over 2 records
  [ok] L2 value_fidelity: 2 values checked
  [ok] L3 titer_unit_is_the_released_unit: stored titer units [<ConcentrationUnit.millimolar: 'mM'>]; Table S2 releases mM
  [ok] L3 stored_statistic_is_a_maximum: the stored value is the maximum over the released sampling times, an upward-biased order statistic; no uncertainty is released for it
  [ok] L3 mixed_feed_cross_source_disagreement_is_declared: Results: 'In this production regime, the baseline PT strain failed to grow and therefore did not produce any detectable isoprenol.'; Table S2: 0.05 mM. The released is stored and the disagreement is reported
  [ok] L3 assembly_pin: SUPPLEMENTARY: assembly pins ["('pputida_KT2440_ASM756v2', 'GCA_000007565.2')"]
  [ok] L3 cross_source_titer_agrees_with_the_results_text: |Table S2 - Results text| = 0.0043 mM, within Table S2's own 0.01 mM rounding
  [ok] L4 gene_containment_kt2440: 0 measured and 1 perturbed host genes; 0 outside the KT2440 locus universe
  [ok] L4 perturbed_gene_containment_assembly: 1.000 of 1 measured genes are loci of pputida_KT2440_ASM756v2 (>= 1.0)

IsoprenylAcetateTiterKang2026Dataset: PASS
  [ok] L0 structural: 19 records validated
  [ok] L1 count: observed 19, expected 19
  [ok] L2 value_fidelity: 19 values checked
  [ok] L2 cross_method: 19 pairs agree within 0.0
  [ok] L3 titer_unit_is_the_sources_mg_per_l_as_ug_per_ml: 1 mg/L == 1 ug/mL exactly, so the released number is stored verbatim
  [ok] L3 every_uncertainty_is_a_typed_gap_not_a_guess: the replicate design is sourced (n=3 biological replicates, sample SD) but no SD number is released anywhere in the mirror
  [ok] L3 every_genotype_carries_pIY670_and_one_aat: PIPA + pIY670 supplies isoprenol and exactly one AAT esterifies it
  [ok] L3 the_reference_is_the_pinned_PIPA_background: every record is an edit of PIPA on GCA_000007565.2
  [ok] L4 fed_batch_titer_vs_table_s9_phases: 7 overlapping entities agree within 1e-09
  [ok] L4 table1_titer_vs_results_prose: 6 overlapping entities agree within 1e-09
  [ok] L4 perturbed_gene_containment_assembly: 1.000 of 7 measured genes are loci of pputida_KT2440_ASM756v2 (>= 1.0)
```

### Verbatim output, bacterial protein abundance

Caglar 2017's 51 `provenance_audit` rows are elided as `[ok] L3 provenance_audit x51`; every
other line is verbatim.

```
proteome_carruthers2025: PASS
  [ok] L0 structural: 19 records validated
  [ok] L1 count: observed 19, expected 19
  [ok] L1 orf_uniqueness: 20 ORFs, 8 with multiple strains (expected)
  [ok] L1 every_record_carries_the_same_protein_keys: 19 records share one set of 1424 keys
  [ok] L2 value_fidelity: 27056 values checked
  [ok] L2 se_nonnegative: 27056 values checked
  [ok] L3 reference_finite: reference abundance finite + key-matched for all 27056 values
  [ok] L3 measurement_type_consistent: single measurement_type: 'dia_nn_top3_peptide_signal_mean'
  [ok] L3 every_sample_is_a_biological_triplicate: stored replicate counts [3]; Supplementary Fig. 13: 'All strains were cultured in triplicate (n = 3)'
  [ok] L4 stored_target_profile_vs_released_sheet: 1424 overlapping entities agree within 1e-06
  [ok] L4 protein_and_perturbed_locus_containment_assembly: 1.000 of 1425 measured genes are loci of pputida_KT2440_ASM756v2 (>= 1.0)

proteome_desiqueira2025: PASS
  [ok] L0 structural: 5 records validated
  [ok] L1 count: observed 5, expected 5
  [ok] L1 orf_uniqueness: 1 ORFs, 1 with multiple strains (expected)
  [ok] L1 strain_condition_uniqueness: SUPPLEMENTARY: 5 distinct (strain, environment) pairs over 5 records
  [ok] L2 value_fidelity: 7655 values checked
  [ok] L2 se_nonnegative: 7655 values checked
  [ok] L3 reference_finite: reference abundance finite + key-matched for all 7655 values
  [ok] L3 measurement_type_consistent: single measurement_type: 'dia_nn_top3_peptide_signal_replicate_mean'
  [ok] L3 assembly_pin: SUPPLEMENTARY: assembly pins ["('pputida_KT2440_ASM756v2', 'GCA_000007565.2')"]
  [ok] L4 gene_containment_kt2440: 1531 measured and 1 perturbed host genes; 0 outside the KT2440 locus universe
  [ok] L4 protein_and_perturbed_locus_containment_assembly: 1.000 of 1532 measured genes are loci of pputida_KT2440_ASM756v2 (>= 1.0)

ProteomeLim2025Dataset: PASS
  [ok] L0 structural: 1 records validated
  [ok] L1 count: observed 1, expected 1
  [ok] L1 orf_uniqueness: 7 unique knocked-out ORFs, one record each
  [ok] L2 value_fidelity: 2361 values checked
  [ok] L2 se_nonnegative: 2361 values checked
  [ok] L3 reference_finite: reference abundance finite + key-matched for all 2361 values
  [ok] L3 measurement_type_consistent: single measurement_type: 'dia_nn_top3_log2_mean'
  [ok] L4 gene_containment_kt2440_deleted_loci: 7 of 7 deleted_loci are pputida_KT2440_ASM756v2 gene rows
  [ok] L4 gene_containment_kt2440_quantified_loci: 2361 of 2361 quantified_loci are pputida_KT2440_ASM756v2 gene rows
  [ok] L4 protein_and_perturbed_locus_containment_assembly: 1.000 of 2361 measured genes are loci of pputida_KT2440_ASM756v2 (>= 1.0)

proteome_caglar2017: PASS
  [ok] L0 structural: 105 records validated
  [ok] L1 count: observed 105, expected 105
  [ok] L1 orf_uniqueness: 0 unique knocked-out ORFs, one record each
  [ok] L1 sample_uniqueness: 105 distinct protein_abundance profiles over 105 records
  [ok] L2 value_fidelity: 440580 values checked
  [ok] L2 se_nonnegative: 0 values checked
  [ok] L2 abundance_nonnegative: 440580 values checked
  [ok] L3 reference_finite: reference abundance finite + key-matched for all 440580 values
  [ok] L3 measurement_type_consistent: single measurement_type: 'lcmsms_spectral_count_deseq2_size_factor_normalized'
  [ok] L3 assembly_pin: assembly pins [('ecoli_B_REL606_ASM1798v1', 'GCA_000017985.1')]
  [ok] L3 log_base_back_solve: table_s3: base 2.0; counts within 1.05e-10 of integers, size factors within 1.62e-14 of DESeq2's
  [ok] L3 provenance_audit x51: value backed by verbatim quote in paper.md / si1.md
  [ok] L4 gene_containment_rel606: 4196 of 4196 measured loci are REL606 GenBank gene rows
  [ok] L4 protein_and_perturbed_locus_containment_assembly: 1.000 of 4196 measured genes are loci of ecoli_B_REL606_ASM1798v1 (>= 1.0)
```

### What could not be verified, and the findings

- **`ConcentrationUnit` still has no `mg_per_l`**, so the titer unit rule pins the stored
  `ug/mL` and its detail carries the equality that makes it honest. Adding the member is a
  served-closure rebuild decision and open item 1 of the Carruthers note, untouched here.
- **Caglar 2017's `n_replicates` is 1 everywhere**, so a replicate-count pin like
  Carruthers' triplicate rule does not generalize into the shared protein verifier; it stays
  a per-release rule in the loader module.
- **de Siqueira's unit message prints the enum repr**
  (`[<ConcentrationUnit.millimolar: 'mM'>]`) where the other datasets print `'mM'`. Its own
  `titer_levels` builds that string; not changed here.
- **A missing store is a loud `lmdb` error, not a skip.** Every one of the seven stores is
  built under the dev `DATA_ROOT`, and no skip branch was added: that is the convention the
  other eleven runners follow, and `run_all` is a data-gated human entry point, never CI.

## 2026.10.07 - run_environment_response is host-aware, and the L4 containment row names the universe it was given

`run_environment_response` built `_sgd_gene_set` and the S288C resolver ONCE at the top and
handed both to every registered dataset, unlike `run_fitness` and `run_rnaseq`, which
already select per host. A bacterial dataset could therefore not be registered at all: its
locus tags would have been resolved against S288C and would have failed containment and
genome membership for the wrong reason. That gap is why at least five landed bacterial
loaders (Tong 2020, Wang 2015, Menasalvas 2025, Borchert 2024, and Cui 2018 on PR #745)
carry their own `verify_build()` instead of a registry entry.

### What changed

- `_host_for_dataset(name, assembly_sets, reference, data_root, cache)` returns a `_Host`
  of `(universe, resolve_gene_name, label)` for the host a dataset's own records name,
  built once per assembly set and cached. `run_fitness` now uses it too, so the selection
  exists once rather than twice; its `len(assembly_sets) > 1` refusal moved in with it.
- `_gene_universe_for_assembly_sets(assembly_sets, data_root)` is the universe build split
  out of `_dataset_gene_universe`, so the STREAMING path can reach it without records.
- `_first_genome_reference(abs_root)` reads the FIRST record of a streamed store. A
  streamed dataset's verifier consumes the records once and needs the universe and
  resolver before that pass starts, so the host cannot be read off every record. A second
  host in the same store is then caught by the per-record `current_genome_genes` rule,
  which fails every record whose systematic names are not loci of the universe it was
  given. The streaming branch still reads `spec["expected_count"]` BEFORE opening the
  store, so a registry spec missing its oracle still fails without touching the LMDB.

### The mislabelled claim this surfaced, and the fix

`SharedRecordRules._gene_containment_results` hardcoded the message "N of M measured genes
are **S288C reference** genes". Four landed bacterial loaders already pass their strain's
locus universe into that rule (`price2018`, `tong2020`, `menasalvas2025`, `borchert2024`,
each `sgd_genes=set(genome.genbank.loci)`), so four published reports asserted that an
`ECB_`, `b`-number or `PP_` tag is an S288C reference gene, which nothing checked.

The rule now takes `gene_universe_label` and names what it was handed. The runners pass
`S288C reference` for a yeast host and the pinned sets for a bacterial one. The DEFAULT is
the host-free word `reference`, which is what removes the false claim from those four
loaders without touching them: a caller that does not name its universe now makes no claim
about one, and its row reads "N of M measured genes are reference genes". Naming each
strain's own universe there is a one-line addition per loader and is left for a change that
owns those four releases' report content.

The result NAME is still `gene_containment_sgd` for every host. Renaming it changes a key
every report and about eight test assertions read, so it is its own change; it is recorded
here as the remaining half of the finding.

### Verbatim output, environment response

Every one of the thirteen registered datasets is YEAST (measured read-only: each store's
first record has `species="Saccharomyces cerevisiae"` and no `assembly_set`), so the
host-aware selection picks the same universe, the same resolver and the same
`S288C reference` label the old code hardcoded. That is the honest result: what changed is
that a bacterial dataset can now be registered here and verified against its own genome,
not that any record moved.

`run_environment_response` was then run over the real dev stores. It takes hours (about
8M records over the thirteen datasets, five of them streamed at 0.3M to 3.1M records), so
what is recorded here is what had completed, verbatim; the rest was still running.

```
yeastphenome: FAIL
  [ok] L0 structural: 296777 records validated
  [ok] L1 count: observed 296777, expected 296777
  [ok] L1 pair_uniqueness: 296777 unique (study, strain, condition) records, one each
  [ok] L1 canonical_gene_names: 5011 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 296777 values checked
  [ok] L3 measurement_type_consistent: single measurement_type: 'z_score'
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 296777 records
  [XX] L3 compound_identity: environment edits: 67 compounds are name-only (no identifier, no gap) over 296777 references: hydrogen peroxide (perturbation.compound) x13340, paraquat (perturbation.compound) x13276, ...
  [XX] L3 media_membership: 4 free-text media over 296777 records join nothing: YPD (base_medium=None) x202782, SC (base_medium=None) x79887, SD (base_medium=None) x13680, CSM (base_medium=None) x428
  [ok] L4 gene_containment_sgd: 1.000 of 5011 measured genes are S288C reference genes (>= 0.9)
  [ok] L4 current_genome_genes: every one of the 5011 measured systematic names is a gene of the current genome

env_chemgen_vanacloig2022: PASS
  [ok] L4 gene_containment_sgd: 1.000 of 3587 measured genes are S288C reference genes (>= 0.9)
  [ok] L4 current_genome_genes: every one of the 3587 measured systematic names is a gene of the current genome

env_chemgen_mota2024: PASS
  [ok] L4 gene_containment_sgd: 1.000 of 600 measured genes are S288C reference genes (>= 0.9)
  [ok] L4 current_genome_genes: every one of the 600 measured systematic names is a gene of the current genome
```

(The three reports' other rows all pass and are in the written
`preprocess/verification_report.json` of each store; only yeastphenome's two failures and
every dataset's L4 rows are reproduced above.)

**yeastphenome FAILs, on two rules this change does not touch.** The failures are
`compound_identity` (67 curated compounds carry a name and neither an identifier nor a
typed gap) and `media_membership` (four free-text media, `YPD` / `SC` / `SD` / `CSM`, that
join nothing in `MEDIA_LIBRARY`) -- both the honest consequence of consuming a secondary
curation layer, and both rules landed with the shared record rules long before this. Its
two L4 gene rows, the ones a host selection decides, pass. I did not re-run yeastphenome
under `origin/main` to measure the before state (another 18-minute materialization per
pass), so "it failed these two before" is an inference from the diff -- neither rule's code
nor its inputs are in it -- and not a measurement.
