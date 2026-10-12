---
id: xfl1rdpnyjsb8ahz6g1kxxm
title: Kuzmin2018
desc: ''
updated: 1725988042050
created: 1705123822425
---
## Things We Know About Dmf Kuzmin

- It exists! This dataset is not just trigenic mutants, it has over 400,000 double mutants. But are they really? And that's where the trouble begins. Technically these are triple mutants but since the ho deletion in considered inert... or something like that, the triple mutants with one ho deletion are considered double mutants.
- One reference temperature as no temperature column data

## Unsure of Origin of Alleles

These alleles don't have much of an explanation, all other alleles are deletions signified by `Δ`.

```python
df[~df['Query allele name_1'].str.contains("Δ")]['Query allele name_1'].unique()
array(['ALG14-ph', 'rfc5-1', 'SPP381-ph', 'cdc7-4', 'uso1-1', 'cop1-1',
       'DAD1-ph', 'sec7-1', 'GPI19-PH', 'smc1-259', 'cdc4-1', 'sec53-6',
       'ret2-1', 'erg26-1', 'spc105-15', 'sec27-1', 'cdc43-2', 'swc4-4',
       'okp1-5', 'BIG1-ph', 'tid3-1', 'pan1-4', 'cdc35-1', 'sds22-5',
       'cof1-5', 'spc3-4', 'PWP1-ph', 'stt4-4', 'kap95-E126K', 'smc6-9',
       'gab1-2', 'ero1-1', 'tap42-11', 'arp9-1', 'RNA14-ph', 'lcb1-5',
       'smc5-6', 'sgt1-3', 'ufe1-1', 'ala1-1', 'kre5-ts2', 'sec16-2',
       'cdc60-ts', 'spn1-K192N'], dtype=object)
# Length is 44
```

```python
df[~df['Query allele name_2'].str.contains("Δ")]['Query allele name_2'].unique()
array(['PRE7-ph', 'sec31-1', 'brl1-3231', 'sam35-2', 'cdc23-1',
       'GPI16-PH', 'phs1-1', 'ask1-3', 'sec59-ts', 'SPC24-9', 'SPC98-PH'],
      dtype=object)
```

## All Double Mutants Contain the ho deletion

This is where we must face the ideal vs reality and choose. In the mood of a German idealist. While it betrays reality a bit to represent the triple deletions as double deletions, and the quadruple deletions and triple deletions as they are so labeled, it goes with the understanding and representation presented by the authors. We trust that the authors know the important points to emphasize, basically a deferral to experts justification for our idealism.

```python
(df['Query allele name'].str.contains("hoΔ")).sum()
410399
len(df)
410399
```

Annoyingly the `hoΔ` can be on the left hand or right hand side of the `+`.

## Processing Kuzmin Double Mutants in Trigenic Rows

I didn't know if the double mutants reported in the trigenic rows were new mutant data or from the double mutants in the previous rows. I thought that it would be redundant to add the double mutant information form the query so it could be possible that the double mutant data was new. After processing all rows we get a bunch of redundant data.

I used this to investigate the duplicates. It is not efficient to run this during creation of the dataset, so I just use it to find that the double mutant data is duplicated in the trigenic rows.

```python
>>> df["md5"] = df.apply(
       lambda row: hashlib.md5(
              json.dumps(
              self.create_experiment(row, self.reference_phenotype_std)[
                     0
              ].model_dump()
              ).encode("utf-8")
       ).hexdigest(),
       axis=1,
)
>>> print(df['md5'].value_counts()>1).sum()
0 # ??? digenic and trigenic 
0 # digenic only

>>> (df['md5'].value_counts()==1).sum()
410399 # digenic and trigenic
410399 # digenic only
```

We could then get rid of this code block.

```python
elif row["Combined mutant type"] == "trigenic":
       # Query 1
       if "KanMX_deletion" in row["query_perturbation_type_1"]:
              genotype.append(
              DeletionGenotype(
                     perturbation=SgaKanMxDeletionPerturbation(
                     systematic_gene_name=row["Query systematic name_1"],
                     perturbed_gene_name=row["Query allele name_1"],
                     strain_id=row["Query strain ID"],
                     )
              )
              )
       elif "allele" in row["query_perturbation_type_1"]:
              genotype.append(
              BaseGenotype(
                     perturbation=SgdAllelePerturbation(
                     systematic_gene_name=row["Query systematic name_1"],
                     perturbed_gene_name=row["Query allele name_1"],
                     strain_id=row["Query strain ID"],
                     )
              )
              )
       # Query 2
       if "KanMX_deletion" in row["query_perturbation_type_2"]:
              genotype.append(
              DeletionGenotype(
                     perturbation=SgaKanMxDeletionPerturbation(
                     systematic_gene_name=row["Query systematic name_2"],
                     perturbed_gene_name=row["Query allele name_2"],
                     strain_id=row["Query strain ID"],
                     )
              )
              )
       elif "allele" in row["query_perturbation_type_2"]:
              genotype.append(
              BaseGenotype(
                     perturbation=SgdAllelePerturbation(
                     systematic_gene_name=row["Query systematic name_2"],
                     perturbed_gene_name=row["Query allele name_2"],
                     strain_id=row["Query strain ID"],
                     )
              )
       )
```

## SmfKuzmin2018 Docker Import Issues with None and Special Characters

`SmfKuzmin2018` has stds don't have values so I was putting `None` which is somewhere converted to nan and the import fails. Another issue I fixed was that the `'` and the `Δ` symbols cannot be imported so I replaced them with `'` with `_prime` and `Δ` with `_delta`.  It is unclear what to do about the missing stds if we cannot import `None.` I thought this should just be mapped to null.

When I replace the `None` with some arbitrary double it works.

## Always Close the lmdb before pickling

```python
print(dataset.env)
> Environment() # some lmdb environment
# ⛔️ Cannot at this point be pickled
dataset.close_lmdb
print(dataset.env)
> None
# ✅ Can be pickled
```

First thing while debugging was that we could not pickle due to an object called `Environment`, naturally I assumed this was the `Environment` object related to data that I wrote. Instead it was the `lmdb` `Environment` object. This object cannot be pickled and it is created whenever data from the `lmdb` is accessed. Prior to pickling it must be made None. The `lmdb` can then be initialized after the dataset object is deserialized on whichever process. A sneaky one for sure 😈.

## Running Docker Interactive Will Merge Stdout and Return Value

This first script results in the print statements to be merged with the return value making bash_script_path_cleaned more like a log. And it prevents us from getting the proper string to call the bash script.

```bash
bash_script_path_cleaned=$(docker exec -it tc-neo4j python -m torchcell.knowledge_graphs.create_scerevisiae_kg_small)
```

The string is wrong Error ⛔️, we can see it is going line by line and trying to run bash on it. The `'INFO'` and `'This'` are output from biocypher.

```bash
----------------NOW BUILDING GRAPH---------------------
chmod: cannot access 'INFO': No such file or directory
chmod: cannot access 'This': No such file or directory
chmod: cannot access 'is': No such file or directory
chmod: cannot access 'BioCypher': No such file or directory
chmod: cannot access 'v0.5.37.'$'\r': No such file or directory
/bin/bash: line 2: biocypher-log/biocypher-20240221-063654.log: Permission denied
/bin/bash: line 2: INFO: command not found
/bin/bash: line 3: frozen: No such file or directory
/bin/bash: line 3: this: command not found
/bin/bash: line 4: ?,: No such file or directory
/bin/bash: line 5: ?,: No such file or directory
/bin/bash: line 6: INFO: command not found
/bin/bash: line 7: INFO:biocypher:Loading: command not found
/bin/bash: line 8: INFO: command not found
/bin/bash: line 9: INFO:biocypher:Instantiat
```

This is the **correct** command.

```bash
bash_script_path_cleaned=$(docker exec tc-neo4j python -m torchcell.knowledge_graphs.create_scerevisiae_kg_small)
```

## 2024.03.06 - Why the Dataset methods Come after Super

Not sure where to put this note but I noticed that child classes of `ExperimentDataset` must always put their attributes and methods before `super()` because this is what runs process. If there are any attributes defined in the dataset that are needed in process they won't be in the scope of process.

## 2024.09.10 - Kuzmin2018 Alternative Download Source

Can download dataset from here too. [kuzmin2018 dataset](https://boonelab.ccbr.utoronto.ca/supplement/kuzmin2018/supplement.html). Might be more stable than science source.

## 2026.09.20 - Ingest the double-mutant query strain fitness

The published trigenic score subtracts the double-mutant query strain's own fitness, and every trigenic row reports it in `Query single/double mutant fitness`, yet the loader never emitted it: `preprocess_raw` filtered the frame to digenic rows, so the trigenic branch of `create_experiment` was unreachable (and would have paired the concatenated query name with the array gene). Without that term a reconstruction of the published score falls from 99.98 percent of rows to r = 0.499, so the loader now keeps it.

- One record per distinct double-mutant query strain, 172 of them, with the query pair as the genotype and the query strain id on both perturbations (main `4c4a4f950`).
- Digenic records keep their positions and serialize byte-identically (410,399 before), so the served graph takes the new records as a superset admission without moving any content-addressed id [[torchcell.knowledge_graphs.incremental-admission]].
- No SD: Data File S4 carries it, but S4 is not among this loader's raw files.
- Consumed at read time by [[torchcell.data.label_policy]], which joins a triple to its query strain on the tm token.

## 2026.10.01 - Issue #533 Refusals for Unknown Arrays, Blank SDs and Repeated Crosses

- Previously `SmfKuzmin2018Dataset.create_experiment` never bound `genotype` for an array strain that was neither `tsa` nor `dma` and failed with `UnboundLocalError` (the `"allele"` array branch was dead); the other four loaders stopped at their genotype-size assertion. Now `_array_perturbation_type` classifies the array column in every `preprocess_raw` and refuses any other strain id with a named `ValueError`, so the dead branch is gone.
- A blank Dmf or Tmf SD used to be stored as `fitness_std` NaN while the uncertainty fields were None. `_reported_sd` stores it as None. The SI lists the column only as "12. Combined mutant fitness standard deviation" (`kuzminSystematicAnalysisComplex2018/si/si1.md` line 427) and defines no meaning for a blank, so None records "no value reported".
- Dmf and Dmi stored a repeated digenic (query strain, array strain) cross twice. `_refuse_repeated_crosses` now refuses it by name.
- Record-neutral: on the pinned `aao1729_data_s1.tsv` each input occurs 0 times (0 other array strains, 0 blank SDs, 0 repeated crosses among 501,510 rows). Tests: `test_unknown_array_strain_type_refuses_the_build`, `test_dmf_allele_pairs_and_blank_sd`, `test_repeated_digenic_cross_is_refused_by_dmf_and_dmi` in `tests/torchcell/datasets/scerevisiae/test_kuzmin2018.py`.
- Observation from the PR #594 review (no code change; identical on `main` and this branch): rebuilding `DmfKuzmin2018Dataset` today gives a reference SD 1 ulp away from the stored one. The store holds 0.061791207824580474; pandas 2.3.3 without bottleneck gives 0.06179120782458047, and a sequential sum reproduces the stored value. The difference touches the reference of all 410,571 records. Hypothesis (untested): the summation order of `Series.mean` in the build environment differs from the current one.
- Review follow-up tests: `test_tmf_blank_sd_is_stored_as_none` pins the Tmf blank-SD mapping; the repeated-cross test now repeats the cross with a different fitness (0.73 against 0.72), so the refusal is keyed on the cross, not the whole row.

## 2026.10.10 - L0-L4 verification of the dev stores (#889)

Fitness stores run the existing fitness battery unchanged through its single-pass twin `verify_fitness_dataset_streaming` (L0 validates each record as `FitnessExperiment`; every rule of the eager verifier applies, none skipped). `pair_uniqueness` keys on the record's perturbation SET plus strain ids and `screen_id`, so the double and triple arms needed no host-aware variant (the #889 hypothesis held: 0 duplicates on every Kuzmin arm). Interaction stores run the new gene-interaction battery (`torchcell/verification/gene_interaction.py`): every rule applies; `released_values` compares the stored (score, p-value) multiset to the sha256-pinned release, and `fitness_companion_containment` checks every (strain, environment) key against the companion fitness store.

Release counts equal the stores: 410,399 digenic rows; 91,111 trigenic rows; DMF adds the 172 trigenic double-mutant query strains that state a fitness (410,571); SMF is the 1,181 array alleles plus the 358 of 361 digenic query alleles that state a query fitness (1,539).

Findings, each left failing and filed: #894 (`dmf_kuzmin2018`, 4 released SDs of exactly 0 stored as a sample SD), #895 (`tmf_kuzmin2018`, 5 of them), #896 (`tmi_kuzmin2018`: Additional Data S1 carries 91,110 negative and 1 zero trigenic tau and no positive one, so `signed_unclamped` fails although `released_values` passes). `smf_kuzmin2018` and `dmi_kuzmin2018` pass every rule.

Tables generated by `experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_report_tables.py` from the reports the sweep wrote (`kg4_pre_build_verification_sweep.py --dataset <Class>`, read-only on `processed/`); count oracles derived by `no_verifier_store_release_counts.py` (results `results/no_verifier_store_release_counts.json`).

`smf_kuzmin2018`: **PASS**, 14 rules, 0 failing (report written 2026-10-10T20:52:18)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 1539 records validated |
| L1 | `count` | PASS | observed 1539, expected 1539 |
| L1 | `pair_uniqueness` | PASS | 1539 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 1539 values checked |
| L2 | `se_nonnegative` | PASS | 0 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 1539 records |
| L1 | `provenance_gaps` | PASS | 1539 documented provenance gaps over 1539/1539 records; 0 deferred field(s): []; 60021 undeclared None values over 15 carrier fields (top: Compound.inchi x20007, Compound.chebi_id x13851, Compound.smiles x4617, Compou... |
| L1 | `canonical_gene_names` | PASS | 1399 systematic names, one canonical spelling each, each current in the genome; 1170 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 10773 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 1539 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 1399 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 1399 measured systematic names is a gene of the current genome |

`dmf_kuzmin2018`: **FAIL**, 14 rules, 1 failing (report written 2026-10-10T20:56:31)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 410571 records validated |
| L1 | `count` | PASS | observed 410571, expected 410571 |
| L1 | `pair_uniqueness` | PASS | 410571 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 410571 values checked |
| L2 | `se_nonnegative` | PASS | 410399 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 410571 records |
| L1 | `provenance_gaps` | PASS | 410571 documented provenance gaps over 410571/410571 records; 0 deferred field(s): []; 13549875 undeclared None values over 15 carrier fields (top: Compound.inchi x5337423, Compound.chebi_id x3695139, Compound.smiles ... |
| L1 | `canonical_gene_names` | PASS | 1400 systematic names, one canonical spelling each, each current in the genome; 1171 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | **FAIL** | 4/410399 labeled uncertainties are a sample dispersion of exactly 0 (a pseudocount artifact, not perfect precision) |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 2873997 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 410571 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 1400 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 1400 measured systematic names is a gene of the current genome |

`tmf_kuzmin2018`: **FAIL**, 14 rules, 1 failing (report written 2026-10-10T20:57:46)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 91111 records validated |
| L1 | `count` | PASS | observed 91111, expected 91111 |
| L1 | `pair_uniqueness` | PASS | 91111 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 91111 values checked |
| L2 | `se_nonnegative` | PASS | 91111 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 91111 records |
| L1 | `provenance_gaps` | PASS | 91111 documented provenance gaps over 91111/91111 records; 0 deferred field(s): []; 3006663 undeclared None values over 9 carrier fields (top: Compound.inchi x1184443, Compound.chebi_id x819999, Compound.smiles x27333... |
| L1 | `canonical_gene_names` | PASS | 1400 systematic names, one canonical spelling each, each current in the genome; 1171 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | **FAIL** | 5/91111 labeled uncertainties are a sample dispersion of exactly 0 (a pseudocount artifact, not perfect precision) |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 637777 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 91111 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 1400 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 1400 measured systematic names is a gene of the current genome |

`dmi_kuzmin2018`: **PASS**, 24 rules, 0 failing (report written 2026-10-10T21:02:53)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 410399 records validated as GeneInteractionExperiment |
| L0 | `reference_structural` | PASS | 410399 records validated as GeneInteractionExperimentReference |
| L1 | `count` | PASS | observed 410399, expected 410399 |
| L1 | `pair_uniqueness` | PASS | 410399 unique (strain, environment) records, one each |
| L1 | `interaction_order` | PASS | all 410399 records perturb 2 distinct genes at graph level 'edge' |
| L2 | `score_finite` | PASS | 410399 interaction scores are finite numbers |
| L2 | `p_value_in_unit_interval` | PASS | 410399 stated p-values lie in [0, 1] (0 records state none) |
| L2 | `released_values` | PASS | the 410399 stored (score, p-value) pairs are the released multiset (the 'digenic' rows of Additional Data S1) |
| L3 | `reference_zero` | PASS | every one of the 410399 reference scores is 0 with no p-value, at the experiment's graph level |
| L3 | `reference_environment` | PASS | every one of the 410399 references is measured in its experiment's own environment |
| L3 | `signed_unclamped` | PASS | 213015 aggravating and 197374 alleviating scores, 10 at exactly zero |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L1 | `provenance_gaps` | PASS | 410399 documented provenance gaps over 410399/410399 records; 0 deferred field(s): []; 15184763 undeclared None values over 13 carrier fields (top: Compound.inchi x5335187, Compound.chebi_id x3693591, Compound.smiles ... |
| L1 | `canonical_gene_names` | PASS | 1400 systematic names, one canonical spelling each, each current in the genome; 1171 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 2872793 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 410399 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 1400 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 1400 measured systematic names is a gene of the current genome |
| L4 | `fitness_companion_containment` | PASS | every one of the 410399 (strain, environment) keys is a record of dmf_kuzmin2018 |

`tmi_kuzmin2018`: **FAIL**, 24 rules, 1 failing (report written 2026-10-10T21:04:21)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 91111 records validated as GeneInteractionExperiment |
| L0 | `reference_structural` | PASS | 91111 records validated as GeneInteractionExperimentReference |
| L1 | `count` | PASS | observed 91111, expected 91111 |
| L1 | `pair_uniqueness` | PASS | 91111 unique (strain, environment) records, one each |
| L1 | `interaction_order` | PASS | all 91111 records perturb 3 distinct genes at graph level 'hyperedge' |
| L2 | `score_finite` | PASS | 91111 interaction scores are finite numbers |
| L2 | `p_value_in_unit_interval` | PASS | 91111 stated p-values lie in [0, 1] (0 records state none) |
| L2 | `released_values` | PASS | the 91111 stored (score, p-value) pairs are the released multiset (the 'trigenic' rows of Additional Data S1) |
| L3 | `reference_zero` | PASS | every one of the 91111 reference scores is 0 with no p-value, at the experiment's graph level |
| L3 | `reference_environment` | PASS | every one of the 91111 references is measured in its experiment's own environment |
| L3 | `signed_unclamped` | **FAIL** | 91110 aggravating and 0 alleviating scores, 1 at exactly zero |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L1 | `provenance_gaps` | PASS | 91111 documented provenance gaps over 91111/91111 records; 0 deferred field(s): []; 3371107 undeclared None values over 13 carrier fields (top: Compound.inchi x1184443, Compound.chebi_id x819999, Compound.smiles x2733... |
| L1 | `canonical_gene_names` | PASS | 1400 systematic names, one canonical spelling each, each current in the genome; 1171 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 637777 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 91111 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 1400 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 1400 measured systematic names is a gene of the current genome |
| L4 | `fitness_companion_containment` | PASS | every one of the 91111 (strain, environment) keys is a record of tmf_kuzmin2018 |

## 2026.10.12 - Released SDs of exactly 0 are stored as released (#894, #895)

Owner decision of 2026-10-12 on #894, #895, #897 and #930: "if no uncertainty is provided we can't do anything about it ... but if uncertainty exists we cannot be missing it!". The uncertainty a source releases is stored exactly as released, found wherever the source states it, and gapped only when the source states none. This reverses the earlier fix direction ("store a typed gap on the uncertainty for these rows").

Measured by [[experiments.036-dataset-fixes-before-kg-build.scripts.keep_records_887_sd_zero_inventory]] (results `results/keep_records_887_sd_zero_inventory.json`, slurm job 4201).

**Case: the release gives exactly 0 and states nothing else.** Additional Data S1 (`aao1729_data_s1.tsv`, sha256 `c82b60f98590071fcb07b74b672958a5863b58e03b78631e6235cb664ee82afb`) releases the SD as `0.0000` on 4 digenic and 5 trigenic rows of 501,510, and the P-value on all 9 as `0.000e+00`. The SI defines the column by its name only, "12. Combined mutant fitness standard deviation" (`si/si1.md` line 427; also "12) Combined mutant fitness standard deviation", line 398; sha256 `2ec80d05d823976e12add17699ad759bcd983768d2f53e7eb6c0185b963b8291`). No SI or main-text line gives a meaning for 0, a per-row replicate count or a second uncertainty column (every line matching "standard deviation", "zero", "not computed", "not calculated", "undefined", "single colony" or "pseudocount" is listed in the results JSON; none concerns this column). The deferred method, Baryshnikova 2010, defines it as the local replicate-colony SD "(based on $4 -$ 8 colonies for the corresponding double mutant)" and notes that such "variance estimates derived from local double mutant colony reproducibility can occasionally dramatically underestimate the true variance" (`baryshnikovaQuantitativeAnalysisFitness2010/si/si1.md` lines 262 and 268), a caveat on the column, not a per-row value.

**Stored as released already; no store changes.** `dmf_kuzmin2018` (410,571 records) holds the 4 and `tmf_kuzmin2018` (91,111) the 5, each with `fitness_uncertainty` = `fitness_std` = 0.0, `sample_sd`, `n_samples` 4, `colony`, `fitness_se` 0.0 and no gap. Only the docstring of `_combined_mutant_uncertainty` changed, to record the decision; no store is rebuilt. The typed note the decision asks for ("the source's uncertainty for this row is 0 as released") has no carrier on `FitnessPhenotype`, so that part waits for the owner question of PR #1054. Until then L2 `uncertainty_sanity` keeps failing on these rows; its message calls a 0 "a pseudocount artifact", a mechanism no source states.

Tests: `tests/torchcell/datasets/scerevisiae/test_kuzmin_released_zero_sd.py` ([[tests.torchcell.datasets.scerevisiae.test_kuzmin_released_zero_sd]]).

L0-L4, read-only (`kg4_pre_build_verification_sweep.py`, job 4201):

`dmf_kuzmin2018`: **FAIL**, 14 rules, 1 failing (report written 2026-10-12T07:19:52)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 410571 records validated |
| L1 | `count` | PASS | observed 410571, expected 410571 |
| L1 | `pair_uniqueness` | PASS | 410571 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 410571 values checked |
| L2 | `se_nonnegative` | PASS | 410399 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 410571 records |
| L1 | `provenance_gaps` | PASS | 410571 documented provenance gaps over 410571/410571 records; 0 deferred field(s): []; 13549875 undeclared None values over 15 carrier fields (top: Compound.inchi x5337423, Compound.chebi_id x3695139, Compound.smiles ... |
| L1 | `canonical_gene_names` | PASS | 1400 systematic names, one canonical spelling each, each current in the genome; 1171 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | **FAIL** | 4/410399 labeled uncertainties are a sample dispersion of exactly 0 (a pseudocount artifact, not perfect precision) |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 2873997 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 410571 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 1400 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 1400 measured systematic names is a gene of the current genome |

`tmf_kuzmin2018`: **FAIL**, 14 rules, 1 failing (report written 2026-10-12T07:21:26)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 91111 records validated |
| L1 | `count` | PASS | observed 91111, expected 91111 |
| L1 | `pair_uniqueness` | PASS | 91111 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 91111 values checked |
| L2 | `se_nonnegative` | PASS | 91111 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 91111 records |
| L1 | `provenance_gaps` | PASS | 91111 documented provenance gaps over 91111/91111 records; 0 deferred field(s): []; 3006663 undeclared None values over 9 carrier fields (top: Compound.inchi x1184443, Compound.chebi_id x819999, Compound.smiles x27333... |
| L1 | `canonical_gene_names` | PASS | 1400 systematic names, one canonical spelling each, each current in the genome; 1171 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | **FAIL** | 5/91111 labeled uncertainties are a sample dispersion of exactly 0 (a pseudocount artifact, not perfect precision) |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 637777 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 91111 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 1400 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 1400 measured systematic names is a gene of the current genome |
