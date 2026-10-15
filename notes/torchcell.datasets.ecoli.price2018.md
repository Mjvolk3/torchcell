---
id: gd97r906vlym2qqh9x2od8v
title: Price2018
desc: ''
updated: 1791384778647
created: 1791384778647
---

## 2026.10.07 - The E. coli BW25113 arm of the Price 2018 compendium

Module: `torchcell/datasets/ecoli/price2018.py` (`RbTnseqPrice2018EcoliDataset`, root
`data/torchcell/rbtnseq_price2018_ecoli`). Tests:
`tests/torchcell/datasets/ecoli/test_price2018.py`. Plan:
[[plan.bacteria-ontology-genome]] section 4. Subsumed row and RB-TnSeq deferral target:
[[torchcell.datasets.ecoli.wetmore2015]]. Shared API: [[torchcell.datasets.bacteria_common]].
Media: [[torchcell.datamodels.media]] (2026.10.07, bacterial media).

Price et al. 2018 (Nature, doi:10.1038/s41586-018-0124-0) is the Fitness Browser
compendium of 32 bacteria. Only its E. coli BW25113 arm is loaded here (`orgId` `Keio`,
library KEIO_ML9): the 162 successful samples of Supplementary Table 5, valued from the
compendium release's `fit_logratios_good.tab` (statistics version 1.0.3). Its KT2440 arm is
the Borchert 2024 loader's. It is rank 21 of the fifty, promoted because it is the
superset of rank 2 (Wetmore 2015).

### What was built (measured 2026-10-07)

| quantity | value |
|---|---|
| Table S5 Keio samples | 162 (Wetmore 2015 92, Price 2018 70) |
| kept samples | 147 (Wetmore 2015 84, Price 2018 63) |
| kept samples by group | carbon source 58, nitrogen source 32, stress 55, plain LB 2 |
| kept samples by medium | `M9 minimal media_noCarbon` 56, `M9 minimal media_noNitrogen` 32, `LB` 57, `MOPS minimal media_noCarbon` 2 |
| release genes / mapped to BW25113 | 3,789 / 3,768 |
| records (`len(dataset)`) | **553,896** = 147 x 3,768 |
| references | 147 (one per sample) |
| gene set | 3,768 |
| build time, LMDB size | 282 s, 2.2 GB |
| `build_manifest` | fresh (`is_stale=False`, no drift) |

The sample, gene and record counts are printed by
`python -m torchcell.datasets.ecoli.price2018 report` from the sha256-pinned files and
asserted by the data-gated tests; the build line is the `build_dataset_lmdb` output.

### Sample inventory by source and the drops

The source paper of each sample is `wetmore2015.subsumption_record().carried` (92 names).
The other 70 cite Price 2018, which re-analyzes "385 successful experiments from Wetmore
et al.9 and 36 successful experiments from Melnyk et al.12" and calls the rest "described
here for the first time". **Inference, not a per-sample statement:** Table S20 gives
KEIO_ML9 one reference (PMID 25968644, Wetmore 2015) and lists no other E. coli library,
so a Keio sample not carried from Wetmore is attributed to Price 2018. Table S5's Keio
samples come from sets set1, set2 and set6 only (Mutalik 2020's phage sets are not among
them).

| rule | scope | items | records not written |
|---|---|---|---|
| `authors_withdrew_sucrose_and_mannitol_2021` | sample | set1IT007, set1IT008 (sucrose), set1IT043, set1IT044 (D-mannitol) | 15,156 |
| `medium_not_in_media_library` | sample | set1IT067 to set1IT070 (`MOPS Rich Defined media_noCarbon`) | 15,156 |
| `motility_soft_agar_assay` | sample | set6IT068 to set6IT070 (outer cut), set6IT072 to set6IT075 (inner cut) | 26,523 |
| `no_one_to_one_eck_pair` | gene | 21 b-numbers (below) | 3,087 |

613,818 source values = 553,896 kept + 56,835 (dropped samples x 3,789 genes) + 3,087
(unmapped genes x 147 kept samples). The withdrawn set equals
`subsumption_record().disregarded`; the build refuses a difference. All eight dropped
Wetmore samples are the four withdrawn and the four MOPS Rich Defined ones; all seven
motility samples are Price 2018's.

The withdrawal, verbatim from the compendium page (`data/bigfit/index.html` in the Wetmore
raw mirror): "Please disregard any of the data from this publication regarding sucrose or
D-mannitol." The compendium release still carries all four samples.

### Strain, library and genes

- Strain: BW25113. Table S14 row: `Escherichia coli BW25113 | Coli genetic stock center |
  Derivative of strain K12 | Coli genetic stock center 7636 | PMID 10829079`; Wetmore 2015
  agrees (`wetmore2015.STRAIN`). Records pin `assembly_reference("BW25113")`
  (`ecoli_K12_BW25113_ASM75055v1`, `GCA_000750555.1`).
- Library: Table S20 row `Escherichia coli BW25113 | KEIO_ML9 | Tn5 | 152018 | 12 |
  electroporation | NA | NA | NA | LB | 37 | Kanamycin; 50 | PMID 25968644`. The leaf is
  `TransposonInsertionPerturbation(transposon="Tn5", library_pool="KEIO_ML9")`, gene-level
  (barcode and insertion site `None`).
- Identifiers: the release names MG1655 b-numbers (all 3,789 `sysName` match `b\d{4}`) for
  this BW25113 library. Each is placed through its one-to-one ECK pair
  (`eck_crosswalk`), and the ECK ids are then run through `reconcile_locus_tags` on
  BW25113, which must return the crosswalk's tag for every one (it does). The threshold is
  stated as `MIN_ECK_ROUTE_FRACTION = 0.99`; the route places 3,768 of 3,789 (0.9945).
  Every record's leaf `description` says the tag is DERIVED that way; the leaf has no typed
  slot for a derived mapping (see Gaps).

| ECK route histogram | value |
|---|---|
| `reconcile_locus_tags` statuses (3,768 ECK ids) | renamed 3,625, non_gene_feature 143, current 0, retired 0, ambiguous 0 |
| layers | gene synonym 3,768 |
| numeric disagreement among mapped pairs | b0018 -> BW25113_4412 (ECK0018) |
| unmapped: b-number not in the MG1655 annotation | 8: b0500, b4104, b4339, b4590, b4600, b4659, b4694, b4695 |
| unmapped: ECK id absent from BW25113 | 7: b1172, b2650, b2863, b4416, b4524, b3683, b4486 |
| unmapped: ECK id not one-to-one | 6: b1370, b4294, b4499, b4576, b4587, b4693 |
| `perturbed_gene_name` | BW25113 gene symbol for 3,625; the locus tag for the 143 pseudogene loci, whose symbol does not resolve as a current gene |

### Phenotype, uncertainty and the t statistic

- `EnvironmentResponsePhenotype(measurement_type=log2_ratio,
  assay_type=pooled_competitive_growth_barcode)` in `BacterialEnvironmentResponseExperiment`.
  Price 2018: "the fitness value of each strain (an individual transposon mutant) is the
  normalized log2(strain barcode abundance at end of experiment/strain barcode abundance at
  start of experiment). The fitness value of each gene is the weighted average of the
  fitness of its strains." The release page: "Gene fitness is a log<SUB>2</SUB> ratio."
- `screen_id` = `Keio:<sample>`. Table S5: "name – an internal identifier for this
  experiment; these are unique only within each organism". Replicates are separate samples
  ("Samples with the same value are replicates"), so each is its own record set.
- Reference: per sample, the typical gene of the same sample, response 0. Price 2018: "Our
  fitness calculations for stress experiments do not correct for the fitness values in the
  unstressed condition."
- **Standard error: the maximum of the two released estimates.** Price 2018 (Methods,
  "Computation of fitness values"): "which is the gene’s fitness divided by the standard
  error9 . The standard error is the maximum of two estimates. The first estimate is based
  on the consistency of the fitness for the strains in that gene. The second estimate is
  based on the number of reads for the gene." The two are `fit_standard_error_obs.tab`
  ("estimated standard error (based on variation across strains)", the release's `se`) and
  `fit_standard_error_naive.tab` ("naive standard error (based on total counts)", the
  release's `sdNaive`). Measured on all 613,818 values of the 162 samples:
  `t = fit / sqrt(0.1^2 + max(se_obs, se_naive)^2)` with max deviation 6.4e-13, while
  `fit / se_obs` alone misses by up to 144 and with sigma 0.1 still misses by up to 29.7.
  The naive estimate is the larger one for 219,448 values (35.75%), so the `se`
  column alone would overstate precision there. The build re-checks the identity
  (tolerance 1e-9) and refuses a release that breaks it. sigma 0.1 is Wetmore 2015's ("we
  use 0.1"). This refines decision (3) of the hand-off, which named the release's
  `se` column; the paper's own definition is the one the released t reproduces.
- The t statistic is not stored: `EnvironmentResponsePhenotype` has no field for it.
- `n_samples` and `sample_unit`: typed gaps (`deferred_pending_source_review`). The usable
  strain count per gene is released only in the R image (`fit.image`, "n -- number of
  usable strains for each gene"); Table S20 note 3 gives its median, 12 ("The median number
  of usable strains for calculating gene fitness per gene in each bacterium"). Neither
  back-solve nor a range rule applies: it is a per-gene count, not one number for the
  dataset, and the stored SE needs no n.

### Environment

Table S5's medium through the media library, its `Temperature` column ("Temperature – the
temperature that the mutant library was grown at"), aerobic (all 162), and Condition_1:

| Table S5 | record |
|---|---|
| `LB` | `LB_LENNOX` (Price's Table S18 'LB' is 5 g/L NaCl) |
| `M9 minimal media_noCarbon` | `M9_NOCARBON_PRICE2018` + `EnvironmentPhysicalPerturbation(carbon_source, dose, agent)` |
| `M9 minimal media_noNitrogen` | `M9_NONITROGEN_PRICE2018` (4 g/L glucose fixed) + `EnvironmentPhysicalPerturbation(nitrogen_source, dose, agent)` |
| `MOPS minimal media_noCarbon` | `MOPS_MINIMAL` + the carbon-source perturbation |
| stress on `LB` | `SmallMoleculePerturbation(compound, dose, solvent)`, the vehicle from Table S4's `Solvent` (2026.10.08 section below) |
| `lb` group | `LB_LENNOX`, no perturbation |

Units: `mM` -> `mM`; `mg/ml` -> `g/L` with the printed value (1 mg/mL is 1 g/L); `g/L`;
`vol%` -> `percent_v/v`. Temperatures are 37 C (tubes), 28 C and 25 C (LB stress and
plain LB, 48-well plates). Shaking and vessel are not stored: the bacterial experiment
class takes a plain `Environment`, not `CultureEnvironment`.

**53 Condition_1 labels of kept samples are not in the pinned compound-identity table**
(listed in `preprocess/dropped_records.json`, `unidentified_compounds`), so their
`Compound` carries the resolver's typed `inchikey` gap and keeps the label as its name.
Resolved: D-glucose, D-serine, glycerol, the amino-acid nitrogen sources, adenosine,
ammonium chloride, cisplatin, DMSO, 1-ethyl-3-methylimidazolium chloride, L-lysine,
sodium chloride, sodium acetate, vanillin, benzoic acid, methylglyoxal, sodium fluoride,
syringaldehyde. The yeast chemogenomic loaders DROP unidentifiable compounds, after their
labels went through the curator; these labels never have, so they are kept with the
typed gap rather than dropping the 99 of the 147 kept samples that name one. Growing the table with
these labels (a `compound_identity_inputs/price2018.txt` list run through the curator)
is a prerequisite for KG admission; `casamino acids` should be entered as an undefined
mixture.

### Raw data

| file | mirror | sha256 (first 16) | role |
|---|---|---|---|
| `data/bigfit/html/Keio/fit_logratios_good.tab` | Wetmore 2015 raw mirror (referenced) | `b37f038702ef0b79` | the stored values |
| `data/bigfit/html/Keio/index.html` | Wetmore 2015 raw mirror (referenced) | `f05323ffce7701df` | release-page quotes |
| `data/bigfit/index.html` | Wetmore 2015 raw mirror (referenced) | `c4819e452ab54dfd` | the 2021 withdrawal |
| `data/bigfit/html/Keio/fit_quality.tab` | Wetmore 2015 raw mirror (referenced) | `7d4b15733d8e2586` | read by `subsumption_record` |
| `data/bigfit/html/Keio/fit_standard_error_obs.tab` | this key | `1ab5a4f016f2e6ab` | SE estimate 1 |
| `data/bigfit/html/Keio/fit_standard_error_naive.tab` | this key | `b1260e6833dd17ac` | SE estimate 2 |
| `data/bigfit/html/Keio/fit_t.tab` | this key | `444f4cac23cd1028` | the SE identity check |
| `si/si3.xlsx` | literature mirror | `e5dbf3d5c97cfc12` | Table S5 metadata |

The four compendium files already in the Wetmore raw mirror are byte-identical to a fresh
retrieval of the same URLs on 2026-10-07 (same sha256), so they are consumed there through
that mirror's manifest, never copied twice. The three new files were retrieved with
`torchcell.literature.retrieve.direct_url` (`python -m torchcell.datasets.ecoli.price2018
retrieve --dest DIR`, then `deposit --source-dir DIR`) into
`$DATA_ROOT/torchcell-raw/priceMutantPhenotypesThousands2018/`; a second deposit changed
no file. `expsUsed` was checked and not kept: its metadata columns equal Table S5's for
all 162 samples. The R image is not mirrored.

### Verifier result (verbatim)

`python -m torchcell.datasets.ecoli.price2018 verify` on the final dev build (streaming
environment-response verifier, BW25113 gene universe and resolver selected by the records'
assembly pin, one SUPPLEMENTARY row). Report:
`data/torchcell/rbtnseq_price2018_ecoli/preprocess/verification_report.json`.

```
RbTnseqPrice2018EcoliDataset: FAIL
  [ok] L0 structural: 553896 records validated
  [ok] L1 count: observed 553896, expected 553896
  [ok] L1 pair_uniqueness: 553896 unique (study, strain, condition) records, one each
  [ok] L1 provenance_gaps: 1688064 documented provenance gaps over 553896/553896 records; 4 deferred field(s): ['inchikey', 'n_samples', 'sample_unit', 'solvent']; 9371016 undeclared None values over 9 carrier fields (top: Compound.inchi x3745392, Compound.chebi_id x1375320, Compound.pubchem_cid x802584, Compound.smiles x802584, Environment.duration_generations x553896)
  [XX] L1 canonical_gene_names: 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 143 systematic names are not the genome's current name; 0 common names resolve to another gene
  [ok] L1 stored_tags_are_loci_of_the_pinned_assembly: SUPPLEMENTARY: 3768 stored tags, statuses {'current': 3625, 'non_gene_feature': 143}; 0 do not resolve to themselves
  [ok] L2 value_fidelity: 553896 values checked
  [ok] L2 se_nonnegative: 553896 values checked
  [ok] L2 uncertainty_sanity: 553896 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty
  [ok] L3 measurement_type_consistent: single measurement_type: 'log2_ratio'
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 553896 records
  [ok] L3 environment_perturbed: all 553896 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media='LB, Lennox (10 g/L tryptone, 5 g/L yeast extract, 5 g/L NaCl), liquid')
  [ok] L3 compound_identity: environment edits: 173328 compound references carry a structure identifier; 373032 declare a typed gap (53 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 2317320 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_membership: 553896 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (4 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 3768 measured genes are S288C reference genes (>= 1.0)
  [ok] L4 current_genome_genes: every one of the 3768 measured systematic names is a gene of the current genome
```

Every row passes except the shared `canonical_gene_names`, whose only finding is the 143
pseudogene loci (status `non_gene_feature`, never `current`); the supplementary row shows
all 3,768 stored tags resolve to themselves (3,625 current, 143 non_gene_feature). Zero
common names resolve to another gene. The verifier takes about 5.7 ms per record (55 min
for the build), dominated by validating each record against the 27-member
`ExperimentType` union.

### Tests

`tests/torchcell/datasets/ecoli/test_price2018.py`: 42 hermetic tests and 39 data-gated
ones (81 with `--data`). The hermetic half builds the loader end to end under `tmp_path`
on the shared synthetic K-12 assemblies (`tests/torchcell/sequence/genome/_bacterial_fixtures.py`):
the ECK route meets every case there (a numeric disagreement, a pseudogene, an id that is
not one-to-one, one BW25113 lacks, an unknown b-number), the build writes 8 records from
a 7-gene x 4-sample release, and `verify`, `report` and the command line run on it.
Hermetic diff coverage of the module is 98%. The data-gated half re-derives the build's
numbers and audits every quote against the pinned mirrors.

### Gaps and open questions

- **Compound identities** (above): 53 labels need the curator before KG admission.
- **Derived mapping has no typed slot.** The leaf's `description` carries it. The typed
  form would be a field on `TransposonInsertionPerturbation`, e.g.
  `source_identifier: str | None` plus `identifier_route: Literal["direct",
  "eck_crosswalk"] | None` (a schema change; the Mutalik 2020 loader names the same need).
- **The t statistic has no slot** on `EnvironmentResponsePhenotype`.
- **MOPS Rich Defined medium** (4 Wetmore samples) and **LB Lennox soft agar** (7 motility
  samples) need `MEDIA_LIBRARY` entries before those samples can load.
- **Verifier, `canonical_gene_names`:** the shared row requires status `current`, which a
  pseudogene locus never has; 143 of the 3,768 stored tags are pseudogene loci. The
  supplementary row `stored_tags_are_loci_of_the_pinned_assembly` accepts a gene or a
  pseudogene that resolves to itself, as the Tong 2020 loader does.
- **Runner registration:** `torchcell.verification.runners.run_environment_response`
  resolves every dataset against S288C, so this dataset is verified by its own
  `verify()` (bacterial gene universe and resolver) until the runner selects by the
  record's assembly pin.
- **Adapter map:** this bullet is stale. `RbTnseqPrice2018EcoliAdapter` (conf
  `rbtnseq_price2018_ecoli_adapter.yaml`) landed before this note was written, and
  `GeneEssentialityPrice2018EcoliAdapter` (conf
  `gene_essentiality_price2018_ecoli_adapter.yaml`) landed 2026.10.08; both are in
  `dataset_adapter_map` and in `kg_bacteria.yaml`.

## 2026.10.07 - Rebased onto the REL606 schema: the staleness gate does not see a widened namespace Literal

Rebased past the REL606 genome tier, Wang 2015, Goodall 2018 and the Caglar 2017 loaders.
Main's REL606 commit widened two module-level aliases in `schema.py`:
`BacterialGeneNamespace` gained `ecoli_b_rel606_locus_tag` and
`BACTERIAL_LOCUS_TAG_PATTERNS` gained its `^ECB_[rt]?\d{5}$` pattern.

**Measured:** the build manifest of the existing dev LMDB still reads `is_stale=False`
with an empty drift list, and 2,000 stored records re-validate against the rebased
schema. So no rebuild was needed.

**Why the gate is silent, and the question it raises.** The manifest's closure holds 40
symbols, and the two bacterial ones are the CLASSES `TransposonInsertionPerturbation` and
`BacterialStrainBackground`; the module-level `Literal` aliases their fields are annotated
with are not closure members. A widened namespace vocabulary therefore moves no
fingerprint. For this dataset that is harmless, since its records store
`ecoli_k12_bw25113_locus_tag` and that member's pattern is untouched. The general case is
worth the owner's attention: a namespace REMOVED from the Literal, or a changed pattern
for a namespace a served dataset uses, would be equally invisible to the staleness gate
while changing what those records are allowed to say.

One CI-only type error was fixed in the same round: `lmdb`'s `txn.get` is `bytes | None`
where CI resolves its stubs and `Any` locally (no stubs in the env), so the dev-build
test now binds the payload to an annotated local and asserts it is not None before
`pickle.loads` (narrowed, never cast).

## 2026.10.08 - Table S4's `Solvent` fills the stress vehicle (SI audit rank 18)

The loader's `SOLVENT_GAP` is gone. `TableS4_Stress` has a column literally named
`Solvent`, so every stress record now carries
`SmallMoleculePerturbation.solvent = Solvent(name=<Table S4 value>, percent=None,
compound=resolved_compound(name))`, matched to Table S5's `Condition_1` by
case-insensitive exact compound name. The gap's recorded reason ("no vehicle is stated
per stress compound in the paper, Table S4 or Table S5") was simply false, which
[[plan.bacteria-si-phenotype-audit-ecoli]] found while auditing.

**Measured on the pinned workbook** (`si/si3.xlsx`, sha256 `e5dbf3d5c97cfc12...`;
reproduced by `python -m torchcell.datasets.ecoli.price2018 report` and asserted by
`test_the_stress_vehicles_of_every_kept_sample`):

| quantity | value |
|---|---|
| Table S4 data rows | 55, one per stress compound, every `Solvent` cell filled |
| distinct `Solvent` values | water, Dimethyl Sulfoxide, Ethanol |
| kept Keio stress samples | 55 of the 147, all matched to a Table S4 compound |
| distinct `Condition_1` labels among them | 35 |
| solvent by kept SAMPLE | water 45, Dimethyl Sulfoxide 7, Ethanol 3 |
| solvent by distinct COMPOUND | water 29, Dimethyl Sulfoxide 5, Ethanol 1 |
| records that gained a vehicle | 207,240 = 55 x 3,768 |

**The caveat is on the record, not only here.** Table S4 is the WILD-TYPE IC50 prescreen
("For each compound, we grew the wildtype bacterium across a 1,000-fold range of
inhibitor concentrations in a rich medium"), so its `Solvent` is the solvent of THAT
assay's stock. The paper never says the mutant fitness assays drew on those stocks
(`paper.md` contains none of "stock solution", "dissolved", "solvent", "DMSO" or
"dimethyl"), and the sheet's own caveat is about concentration only ("The concentrations
reported here are not necessarily the concentrations used for the mutant fitness
assays"). Every stress perturbation therefore carries `STRESS_DESCRIPTION`, which says
the vehicle is DERIVED from the prescreen stock and that the release does not tie it to
the fitness assays, and `Solvent.percent` stays `None` because the final vehicle fraction
is never released. There is no field for the percent, and no gap can name it
(`Solvent` is not a gap carrier), so that absence lives in the same description.

`water` is not in the pinned compound-identity table, so its `Compound` carries the
resolver's typed `inchikey` gap, the same treatment the 53 unresolved stress labels get.
Ethanol and dimethyl sulfoxide resolve with an InChIKey. The verifier's deferred-field
list therefore drops `solvent` and keeps `inchikey`, `n_samples` and `sample_unit`.

**Rebuild, not a migration.** The dev store was retired to
`/scratch/projects/torchcell-deprecated/2026-10-07_235615__processed` (and
`...235620__preprocess`) and rebuilt: **553,896 records in 276 s**, gene set 3,768,
147 references, `build_manifest` fresh. The counts are unchanged; what changed is what
each stress record says.

## 2026.10.08 - Supplementary Table 1 as a second dataset: likely-essential genes

`GeneEssentialityPrice2018EcoliDataset`, root `data/torchcell/gene_essentiality_price2018_ecoli`,
adapter `torchcell/adapters/price2018_ecoli_essentiality_adapter.py` with conf
`gene_essentiality_price2018_ecoli_adapter.yaml`. SI audit rank 9.

### What was built (measured 2026.10.08)

| quantity | value |
|---|---|
| Table S1 `orgId == "Keio"` rows | **324** |
| records (`len(dataset)`) | **320** |
| dropped (`no_one_to_one_eck_pair`) | 4: `b0062` araA, `b0063` araB, `b3903` rhaA, `b3904` rhaB |
| intersection with the 3,789 fitness genes | **0** |
| `geneClass` | Arole 200, Bspecific 98, Dhypo 19, Cvague 7 |
| ECK route | 320 of 324 (0.98765), all at the gene-synonym layer, status `renamed` 320 |
| references | 1 (the unperturbed parent on the selection plates) |
| build time | 5 s |

`python -m torchcell.datasets.ecoli.price2018 essentiality-report` prints every number
above from the sha256-pinned files; the data-gated
`test_table_s1s_keio_rows_and_their_disjointness_from_the_fitness_genes` asserts them.

### The label is not plain essentiality, and the records say so

The paper's own label, verbatim (Methods, "Identifying essential or nearly essential
genes"): "Genes that lack insertions or that have very low coverage in the start samples
are likely to be essential or important for growth (nearly essential) in rich medium, as
except for S. elongatus, pools of mutants were produced and recovered in medium that
contained yeast extract." The rule, verbatim: "Protein-coding genes were considered
essential or important for growth (nearly essential) if we did not estimate fitness
values for the gene and both the normalized insertion density and the normalized read
density were under 0.2."

The false-discovery rate, verbatim from Supplementary Note 1 (`si/si1.md`, sha256
`1c68b123ebb7516b...`): "So, we expect that the true rate of false positives in our list
of $E .$ coli proteins that are essential, or nearly so, for growth in rich media is
somewhere between $6 \%$ and $16 \%$ ." The naive rate against the PEC and Keio-collection
list is higher, also verbatim: "Our list of essential genes also includes 67
non-essential genes, which corresponds to a false discovery rate (FDR) of $21 \%$ ." The
note then argues 15 of those 67 are nearly essential in the library-isolation condition,
which is how 21% becomes 16%.

The condition is the library-isolation condition, not any fitness assay, verbatim:
"However, some of these "false positives" are likely to be essential, or nearly so, in
the condition that we used to isolate our mutant library, namely growth on LB plates."
and "...nearly-essential for growth on LB plates at $3 7 ^ { \circ } \mathsf { C }$".

`GeneEssentialityPhenotype` is one boolean (`is_essential`) with no slot for "nearly", no
slot for an FDR and no slot for the coverage the call is computed from. So the caveat is
carried three ways, and the third is the one that travels with a single record:

1. **`preprocess/essentiality_label.json`** holds eleven `SourcedValue`s verbatim, each
   with its source uri and sha256: the label, the rule, the barcode-free analysis, the
   condition, its temperature, both FDR statements, the PEC benchmark, the per-organism
   range and the fitness set's "non-essential" restriction.
2. **`preprocess/dropped_records.json`** (`EssentialityInventory`) lists the quantities
   that have no field at all, each keyed to the quote that states it.
3. **Every record's `TransposonInsertionPerturbation.description`**
   (`ESSENTIAL_DESCRIPTION`) says the call is a no-insertion call, not a measurement,
   names the label "essential or important for growth (nearly essential)", gives the
   condition as LB plates at 37 C, and states the 6% to 16% FDR. A verifier row,
   `label_caveat_on_every_record`, fails if any record drops it.

### Why 320 and not 324: BW25113 deletes araBAD and rhaBAD

The four unplaced rows are araA, araB, rhaA and rhaB, whose ECK ids
(`ECK0063`, `ECK0064`, `ECK3896`, `ECK3897`) no BW25113 locus carries. The reason is not
asserted from the literature: `BW25113_BACKGROUND_LESIONS` holds `(araBAD)567` and
`(rhaBAD)568` verbatim from the `source` feature of the sha256-pinned
`ecoli_K12_BW25113_ASM75055v1` GenBank file, and `schema.py` already records that
"araB, araA, rhaB, rhaA are absent, which is what being deleted looks like". So a library
built in BW25113 can have no insertion in genes it does not have, and the release's
no-insertion rule calls them essential: a false positive by construction of MG1655
numbering on a BW25113 library.

Three checks, all at build time:

1. `dropped_genes_explained_by_background` ties each dropped gene to the lesion that
   accounts for it (measured: `araA` and `araB` to `(araBAD)567`, `rhaA` and `rhaB` to
   `(rhaBAD)568`), and a drop the genotype cannot account for RAISES, because that would
   mean the mapping lost a gene rather than that BW25113 lacks one.
2. Table S1 corroborates the absence itself: those four are the **only** Keio rows whose
   `locus_tag` (its one BW25113 identifier) is empty, and the build refuses a release
   where the set the ECK route cannot place differs from the set Table S1 leaves blank.
3. The route floor (`MIN_ESSENTIAL_ECK_ROUTE_FRACTION = 0.98`, below the fitness tables'
   0.99) still stops a wrong annotation, which would place far fewer than 320 of 324.

### Disjointness is an invariant, checked twice

A gene is called likely-essential only where the fitness analysis produced no value for
it ("if we did not estimate fitness values for the gene"), and that analysis covers "the
123,255 different non-essential protein-coding genes for which we collected gene fitness
data". **Measured: the 324 essential b-numbers and the 3,789 `fit_logratios_good.tab`
b-numbers intersect in 0**, on `sysName` and on `locusId` alike. `process()` refuses a
release that breaks it (the gene would otherwise be stored under two contradicting
phenotypes), and the verifier re-reads both files and proves it again as
`disjoint_from_the_fitness_genes`.

### Identifiers: a second route corroborates the first

The stored namespace is `ecoli_k12_bw25113_locus_tag` (`^BW25113_\d{4}$`), so Table S1's
RefSeq `BW25113_RS*` tags cannot be stored as they are; the b-numbers go through the same
one-to-one ECK pair the fitness dataset uses. Table S1's own `locus_tag` is then a second,
independent route: **318 of the 320 RefSeq tags resolve to exactly the locus the ECK route
chose, 2 (`b1457`, `b4047`) are tags this annotation release does not carry, and 0 name a
different locus.** Reported by `refseq_route_agreement`, not enforced, because the RefSeq
layer is a second release of the same assembly.

35 of the 320 genes carry a Table S1 `name` that is not the genome's current symbol
(`imp` -> `lptD`, `yaeT` -> `bamA`, `GroEL` -> `groL`, ...). The stored
`perturbed_gene_name` is the genome's, so a stored common name always resolves to the
stored locus tag; the sheet's older name stays in `preprocess/essential_genes.csv`.

### Environment: the selection plates, derived from LB Lennox

`selection_medium()` is a medium DERIVED from `LB_LENNOX` (`base_medium` `LB`, so it joins
there): solid, with agar at `concentration=None` and kanamycin at 50 ug/mL, at 37 C. The
sources are Table S20's E. coli row (`Media` `LB`, `Temperature for selecting mutants` 37,
`Antibiotic; concentration (in ug/mL)` `Kanamycin; 50`), its note 5 ("Media used for for
both the conjugation ... and for the selection of transposon mutants (with supplemented
kanamycin)"), and Supplementary Note 1 for the plates being solid. `LB_AGAR`'s 2% agar is
Menasalvas 2025's and Schmidt 2016's bench value and is NOT asserted here, the same call
`goodall2018.selection_medium()` makes. `duration_hours` and `duration_generations` are
typed gaps (`not_reported_by_primary`); `aerobicity` keeps the field default, since plates
incubated in air are never stated and the field cannot be None.

The reference is the unperturbed BW25113 parent on the same plates, `is_essential=False`:
the library was built in that parent and isolated there, so its viability is a fact of the
experiment, not an inference.

### Verifier result (verbatim)

`python -m torchcell.datasets.ecoli.price2018 essentiality-verify`. Report:
`data/torchcell/gene_essentiality_price2018_ecoli/preprocess/verification_report.json`.

```
GeneEssentialityPrice2018EcoliDataset: PASS
  [ok] L0 structural: 320 records validated
  [ok] L1 count: observed 320, expected 320
  [ok] L1 provenance_gaps: 1280 documented provenance gaps over 320/320 records; 1 deferred field(s): ['inchikey']; 5760 undeclared None values over 6 carrier fields (top: Compound.inchi x1600, Compound.smiles x1280, Compound.chebi_id x960, Compound.pubchem_cid x960, Compound.inchikey x640)
  [ok] L1 canonical_gene_names: 320 systematic names, one canonical spelling each, each current in the genome
  [ok] L1 stored_tags_are_loci_of_the_pinned_assembly: SUPPLEMENTARY: 320 stored tags, statuses {'current': 320}; 0 do not resolve to themselves
  [ok] L2 calls_match_table_s1: 320 records, one likely-essential locus each, from 324 released Keio rows
  [ok] L2 uncertainty_sanity: 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty
  [ok] L3 disjoint_from_the_fitness_genes: 324 likely-essential genes and 3789 genes with fitness values share none
  [ok] L3 label_caveat_on_every_record: all 320 records carry the 'nearly essential' label and the 6% to 16% FDR on their perturbation description
  [ok] L3 compound_identity: environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 640 compound references carry a structure identifier; 320 declare a typed gap (1 distinct compounds, unencodable)
  [ok] L3 media_membership: 0 records on a shared MEDIA_LIBRARY medium, 320 on a medium deriving from one (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 320 measured genes are BW25113 genes (>= 1.0)
  [ok] L4 current_genome_genes: every one of the 320 measured systematic names is a gene of the current genome
```

Every row passes, unlike the fitness dataset's `canonical_gene_names`: no essential gene
is a pseudogene locus, so all 320 stored tags are status `current`. The one unencodable
medium component is **kanamycin**, which the pinned identity table does not resolve;
`agar` resolves without an InChIKey, so it is identified with an `inchikey` gap.

### Not loaded, and why

- **Table S1's coverage columns** (`GC`, `nReads`, `normreads`, `nPosCentral`, `dens`):
  the sequencing evidence the call is computed FROM plus a sequence property, not a
  phenotype. Kept in `preprocess/essential_genes.csv`.
- **Tables S2 and S3's wild-type carbon and nitrogen growth calls** (192 rows, SI audit
  rank 10): NOT loaded. I read both legends at the source rather than taking the audit's
  word, and they say MORE against loading than the audit reported. Verbatim, Table S2
  rows 1, 2 and 4 of `TableS2_Carbon`:

  > True | Postitive for growth on the indicated carbon substrate with the wild-type bacterium or a successful genome-wide mutant fitness assay was done.
  > False | No growth on the indicated carbon substrate with the wild-type bacterium.
  > Notes: | A call of FALSE does not necessary mean that the bacterium does not grow on a given substrate. Our threshold for TRUE growth is somewhat conservative. In addition, alternative conditions (for example, lower concentrations or different base media) could support growth.

  `TableS3_Nitrogen` repeats it with "at the indicated concentration" added. So TRUE is a
  disjunction of a growth observation and a data-availability fact, and FALSE is hedged by
  the sheet itself. **Agreed with the audit: not loadable as they stand.**

  Two further legend lines, not in the audit, make the same point sharper. A FALSE is
  sometimes a CURATION OVERRIDE, not a reading: "L-tyrosine disodium salt often
  precipitated out of solution in our growth media which caused high OD600 readings. We
  set this compound to FALSE for all organisms except Pseudomonas stutzeri RCH2, for which
  we successfully collected fitness data with this compound as a carbon source." And the
  sheet names a mechanism for false negatives it did not correct: "In some experiments the
  ethanol evaporated in the stock compound plate, which may result in false negative growth
  calls." A loaded record could not say which of the three kinds of statement it is.
  What would make them loadable is not a schema change but a split of the TRUE column into
  "grew" and "assayed", which only the authors can give.
- **The other 31 organisms' essential genes** (13,545 rows): out of scope for an E. coli
  loader; each needs a host genome tier and an assembly pin.
- **Per-strain fitness** (SI audit rank 1, upper bound 24.6M records): not mirrored, and
  the count is an upper bound rather than a count. Untouched here.

### Gaps and open questions

- **Kanamycin needs the curator** before KG admission, alongside the fitness dataset's 53
  stress labels: one `compound_identity_inputs/price2018.txt` list covers both.
- **No field for "nearly essential"**, for the 6% to 16% FDR, or for the insertion and read
  densities. The honest form today is the three-way caveat above; a typed form would be a
  confidence or call-rule field on `GeneEssentialityPhenotype`, which is a schema change
  and a full-rebuild trigger.
- **The PEC / Baba 2006 benchmark is not in the mirror** (its key holds no Supplementary
  Table 3), so the 78% agreement the note quotes cannot be recomputed here.
- **`Environment.duration_generations` on the FITNESS records is still an undeclared
  None.** The essentiality records declare both duration gaps; the fitness records do not,
  and the audit flagged that. Out of scope here, so it stays open.

## 2026.10.10 - Wetmore 2015 is now loaded from its own release; this store is its cross-check

Under the owner policy of 2026-10-10 (a served compendium is a processing reference and a sanity check, not the source of record), Wetmore 2015's own 1.0.0 release is loaded as `RbTnseqWetmore2015Dataset` ([[torchcell.datasets.ecoli.wetmore2015]], 2026.10.10 section). Nothing in this loader changed: it still attributes its 92 Wetmore samples to that paper through `wetmore2015.subsumption_record()`, still drops the four withdrawn samples, and its store is read read-only by the Wetmore verifier's L4 row (`compendium_cross_check_price2018`). The two datasets key a shared sample identically (`Keio:<name>`), so the 1.0.0 and 1.0.3 analyses of one sample join on it. The schema change that came with that loader (`environment_response_t_statistic` on `EnvironmentResponsePhenotype`) moves this store's closure, so it reads stale until the KG 4.0 rebuild; its records would serialize that field as None.

## 2026.10.11 - Author-withdrawn data is refused by both loaders: measured, this loader already did

Owner decision, 2026-10-11: author-withdrawn data is bad data and must not be stored. Measured against the built dev store (`$DATA_ROOT/data/torchcell/rbtnseq_price2018_ecoli/preprocess/samples.json`, built on main): this loader already REFUSES all four, with the drop rule `authors_withdrew_sucrose_and_mannitol_2021`, and the ledger (`dropped_records.json`) already counts them as 15,156 refused released values (4 samples x 3,789 release genes). 147 samples are kept, 553,896 records stored. **The record delta of the decision for this dataset is 0** and the store's bytes do not change, so no rebuild is called for on its account. (The store is not in the served tree either: `$DATA_ROOT/database/data/torchcell/` holds 19 yeast datasets and no `rbtnseq_price2018_ecoli`. It is a dev store awaiting KG 4.0, and the t-statistic schema change makes it stale for that rebuild regardless.)

What did change here is that the refusal is now typed once instead of twice. `DROP_DISREGARDED` and the sucrose / D-mannitol condition set come from `wetmore2015.AUTHOR_WITHDRAWAL`, the frozen `AuthorWithdrawal` carrying the rule name, the `Condition_1` labels, and the authors' instruction and its cause as two `SourcedValue`s pinned to `data/bigfit/index.html` (sha256 `c4819e45...`). `DropRuleCount` gains `source`, so the withdrawal's verbatim quote and sha256 ride into `dropped_records.json` beside the count rather than living only in the rule's prose. The rule-name string is unchanged, so `samples.json` is byte-identical.

## 2026.10.11 - 52 of the 53 unidentified condition labels resolve (#726)

#726 opened on this dataset's 53 Condition_1 labels that the compound-identity table did not
resolve, across 99 of the kept samples. The curator pass
([[torchcell.datamodels.compound_identity]], 2026.10.11) curated all of them plus the solvent
`water`. Measured from the store's own ledger (`preprocess/dropped_records.json`,
`unidentified_compounds`): **53 labels over 99 samples before** (the store array 3733 built
at main `52d376d6d`), **1 label over 4 samples after** (slurm array 3809, task 35): casamino
acids, an acid hydrolysate of casein, whose row is UNDEFINED_MIXTURE and whose gap is now
terminal. Bacitracin and Benzalkonium Chloride resolve to homologue-mixture rows (a PubChem
CID, no single-molecule InChIKey), so the ledger counts them identified while the records
keep a terminal `inchikey` gap. Kept records are unchanged at 553,896.

Price 2018's own Tables S2-S4 state a CAS number for these labels (not for `water`), and
every one was looked up: 54 of the 63 curated labels with a CAS agree with the CID the curator adopted; the nine
others are explained in the compound-identity note, including three where the CAS names a
different form than the label (the anhydrous salt for `a-Ketoglutaric acid disodium salt
hydrate` and `Adenine hydrochloride hydrate`, lactic acid itself for `Sodium D,L-Lactate`).
The label's form is kept there.

Record compound NAMES that change, each a fold onto an existing row or a shared spelling:
`2-Furfuraldehyde` -> `furfural`, `5-Hydroxymethylfurfural` -> `5-(hydroxymethyl)furfural`,
`Cobalt chloride hexahydrate` -> `cobalt(II) chloride hexahydrate`, `copper (II) chloride
dihydrate` -> `copper(II) chloride dihydrate`, `D-Cycloserine` -> `cycloserine`,
`D-Fructose` -> `fructose`, `D-Galactose` -> `galactose`, `D-Mannose` -> `mannose`,
`D-Sorbitol` -> `sorbitol`, `D-Xylose` -> `xylose`, `N-Acetyl-D-Glucosamine` ->
`N-acetylglucosamine` (the CID its Table S2 CAS gives), `Sodium succinate dibasic
hexahydrate` -> `disodium succinate hexahydrate`; and case only for `Chloramphenicol`,
`Cytidine`, `D-Alanine`, `D-Ribose`, `Doxycycline hyclate`, `Glycolic Acid` and `Sodium
pyruvate`. The module docstring and `stress_solvent`'s docstring no longer say water and
the stress labels carry a gap, and the test now pins the resolved identities
(`Cephalothin sodium salt` keeps its name and gains CID 23675321; `water` gains CID 962).

`GeneEssentialityPrice2018EcoliDataset` was rebuilt too (task 14): its two kanamycin
references now carry the key.

### L0-L4 after the rebuild

#### `RbTnseqPrice2018EcoliDataset`

553,896 records; store built 2026-10-11 07:35 UTC; 1 report(s); verdict **PASS**.

Report `verification_report.json` (written 2026-10-11 09:21 UTC):

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 553896 records validated |
| L1 | `count` | pass | observed 553896, expected 553896 |
| L1 | `pair_uniqueness` | pass | 553896 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 553896 values checked |
| L2 | `se_nonnegative` | pass | 553896 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'log2_ratio' |
| L3 | `reference_zero` | pass | numeric rule: reference response == 0 for all 553896 records |
| L3 | `environment_perturbed` | pass | all 553896 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media='LB, Lennox (10 g/L tryptone, 5 g/L yeast extract, 5 g/L NaCl), liquid') |
| L1 | `provenance_gaps` | pass | 1134168 documented provenance gaps over 553896/553896 records; 2 deferred field(s): ['n_samples', 'sample_unit']; 13316112 undeclared None values over 17 carrier fields (top: Compound.inchi x3952632, Compound.chebi_id x1386624, Environment.dilution_rate_per_hour x553896, Environment.duration_generations x553896, Environment.duration_hours x553896) |
| L1 | `canonical_gene_names` | pass | 3768 systematic names, one canonical spelling each, each current in the genome; 143 common names the resolver cannot place; 143 are pseudogene loci the genome resolves to themselves |
| L2 | `uncertainty_sanity` | pass | 553896 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 738528 compound references carry a structure identifier; 15072 declare a typed gap (1 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 2317320 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 553896 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (4 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 3768 measured genes are reference genes (>= 1.0) |
| L4 | `current_genome_genes` | pass | every one of the 3768 measured systematic names is a gene of the current genome |
| L1 | `stored_tags_are_loci_of_the_pinned_assembly` | pass | SUPPLEMENTARY: 3768 stored tags, statuses {'current': 3625, 'non_gene_feature': 143}; 0 do not resolve to themselves |

#### `GeneEssentialityPrice2018EcoliDataset`

320 records; store built 2026-10-11 07:02 UTC; 1 report(s); verdict **PASS**.

Report `verification_report.json` (written 2026-10-11 08:03 UTC):

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 320 records validated |
| L1 | `count` | pass | observed 320, expected 320 |
| L2 | `calls_match_table_s1` | pass | 320 records, one likely-essential locus each, from 324 released Keio rows |
| L3 | `disjoint_from_the_fitness_genes` | pass | 324 likely-essential genes and 3789 genes with fitness values share none |
| L3 | `label_caveat_on_every_record` | pass | all 320 records carry the 'nearly essential' label and the 6% to 16% FDR on their perturbation description |
| L1 | `provenance_gaps` | pass | 960 documented provenance gaps over 320/320 records; 0 deferred field(s): []; 5440 undeclared None values over 7 carrier fields (top: Compound.inchi x1600, Compound.chebi_id x960, Compound.smiles x960, Compound.inchikey x640, Compound.pubchem_cid x640) |
| L1 | `canonical_gene_names` | pass | 320 systematic names, one canonical spelling each, each current in the genome |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 960 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 320 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 320 measured genes are BW25113 genes (>= 1.0) |
| L4 | `current_genome_genes` | pass | every one of the 320 measured systematic names is a gene of the current genome |
| L1 | `stored_tags_are_loci_of_the_pinned_assembly` | pass | SUPPLEMENTARY: 320 stored tags, statuses {'current': 320}; 0 do not resolve to themselves |

### After the rebase onto main

The stores above were built at the branch's pre-rebase commit `cb310e339`. Rebasing onto
main, where the Parsons 2006 and De Chiara 2022 loaders had landed, changed two of this
dataset's labels ([[torchcell.datamodels.compound_identity]], 2026.10.11):

- `Doxycycline hyclate` now resolves to the Parsons 2006 row, PubChem CID 54705095
  (`DWBSXBGBGHKWOT-WBYAVNBMSA-N`, the record PubChem titles "Doxycycline hyclate"), not the
  CID 54686183 (`UHHHTIKWXBRCLT-VDBOFHIQSA-N`, titled Doryx) the pass first adopted and the
  RB-TnSeq store above carries. Table S4's CAS 24390-14-5 looked up by name returns CID 54686183, so the CAS
  cross-check is now 53 of 63 in agreement; this tenth disagreement is a second PubChem record
  of the same salt, kept for the reason the other four of that kind are.
- `Potassium acetate` is renamed, case only, to `potassium acetate`, because De Chiara 2022
  spells the same compound in lowercase and a key spelled two ways takes the lowercase
  canonical spelling. Its InChIKey and CID do not change.

The RB-TnSeq store carries both labels and takes both changes at its next build; every dev
store is stale on main anyway.

## 2026.10.12 - #1057 on main: released rows stored, rebuilt and verified

Owner rule of 2026-10-12: a released row is stored under the identifier its release gives, and the genome never gates storage. Rebuilt from main `db632f2b0` with `--retire-existing --verify` (slurm array 4506; the stores with no per-dataset verifier entry point verified by job 4582).

### `rbtnseq_price2018_ecoli`: 553,896 -> 572,139 records

Row 16 MOPS rich medium (+4 samples), row 23 t statistic on every record, the 21 genes with no one-to-one ECK pair stored as released MG1655 b-numbers.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 572139 records validated |
| L1 | `count` | pass | observed 572139, expected 572139 |
| L1 | `pair_uniqueness` | pass | 572139 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 572139 values checked |
| L2 | `se_nonnegative` | pass | 572139 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'log2_ratio' |
| L3 | `reference_zero` | pass | numeric rule: reference response == 0 for all 572139 records |
| L3 | `environment_perturbed` | pass | all 572139 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media='LB, Lennox (10 g/L tryptone, 5 g/L ... |
| L1 | `provenance_gaps` | pass | 1170801 documented provenance gaps over 572139/572139 records; 2 deferred field(s): ['n_samples', 'sample_unit']; 14648274 undeclared None values over 17 carrier fields (top: Co... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 21 systematic names are not the genome's current name; 2 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 572139 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 757800 compound references carry a structure identifier; 15156 declare a typed gap (1 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 3012255 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 572139 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (5 distinct media) |
| L4 | `gene_containment_sgd` | finding (genome reference; reports, does not gate) | 0.994 of 3789 measured genes are reference genes (>= 1.0) |
| L4 | `current_genome_genes` | finding (genome reference; reports, does not gate) | 21 systematic names are absent from the current genome over 3171 records: b0500 x151, b1172 x151, b1370 x151, b2650 x151, b2863 x151, b3683 x151, b4104 x151, b4294 x151, b4339 x... |
| L1 | `stored_tags_are_loci_of_the_pinned_assembly` | pass | SUPPLEMENTARY: 3768 stored tags, statuses {'current': 3625, 'non_gene_feature': 143}; 0 do not resolve to themselves |

### `gene_essentiality_price2018_ecoli`: 320 -> 324 records

The ECK-unplaced essential genes stored as released.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 324 records validated |
| L1 | `count` | pass | observed 324, expected 324 |
| L2 | `calls_match_table_s1` | pass | 324 records, one likely-essential locus each, from 324 released Keio rows |
| L2 | `evidence_matches_table_s1` | pass | 324 records carry rb_tnseq_coverage_and_fitness evidence with their Table S1 coverage, every density under 0.2 |
| L3 | `disjoint_from_the_fitness_genes` | pass | 324 likely-essential genes and 3789 genes with fitness values share none |
| L3 | `label_caveat_on_every_record` | pass | all 324 records carry the 'nearly essential' label and the 6% to 16% FDR on their perturbation description |
| L1 | `provenance_gaps` | pass | 972 documented provenance gaps over 324/324 records; 0 deferred field(s): []; 9072 undeclared None values over 18 carrier fields (top: Compound.inchi x1620, Compound.chebi_id x9... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 4 systematic names are not the genome's current name; 0 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 972 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 324 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | finding (genome reference; reports, does not gate) | 0.988 of 324 measured genes are BW25113 genes (>= 1.0) |
| L4 | `current_genome_genes` | finding (genome reference; reports, does not gate) | 4 systematic names are absent from the current genome over 4 records: b0062 x1, b0063 x1, b3903 x1, b3904 x1 |
| L1 | `stored_tags_are_loci_of_the_pinned_assembly` | pass | SUPPLEMENTARY: 320 stored tags, statuses {'current': 320}; 0 do not resolve to themselves |

## 2026.10.12 - #1045 part 2: the rest of the released rows, under their released identifiers

Every row #1057 still dropped that is not an author withdrawal is stored, and no identifier is translated: the stored key is the one the release prints, and a crosswalk to an annotation is a modeling-time join outside the record. Rebuilt from the part-2 branch with `--retire-existing --verify` (slurm array 4583; the stores with no per-dataset verifier entry point verified by jobs 4645 and 4644).

### `rbtnseq_price2018_ecoli`: 572,139 -> 598,662 records

Every gene under its released MG1655 b-number (no ECK liftover to BW25113; the crosswalk is a derived report) and the 7 soft-agar motility samples kept (151 -> 158 samples).

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 598662 records validated |
| L1 | `count` | pass | observed 598662, expected 598662 |
| L1 | `pair_uniqueness` | pass | 598662 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 598662 values checked |
| L2 | `se_nonnegative` | pass | 598662 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'log2_ratio' |
| L3 | `reference_zero` | pass | numeric rule: reference response == 0 for all 598662 records |
| L3 | `environment_perturbed` | pass | all 598662 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media='LB, Lennox (10 g/L tryptone, 5 g/L ... |
| L1 | `provenance_gaps` | pass | 1250370 documented provenance gaps over 598662/598662 records; 2 deferred field(s): ['n_samples', 'sample_unit']; 15337872 undeclared None values over 18 carrier fields (top: Co... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 8 systematic names are not the genome's current name; 1 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 598662 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 784323 compound references carry a structure identifier; 15156 declare a typed gap (1 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 3038778 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 598662 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (5 distinct media) |
| L4 | `gene_containment_sgd` | finding (genome reference; reports, does not gate) | 0.998 of 3789 measured genes are reference genes (>= 1.0) |
| L4 | `current_genome_genes` | finding (genome reference; reports, does not gate) | 8 systematic names are absent from the current genome over 1264 records: b0500 x158, b4104 x158, b4339 x158, b4590 x158, b4600 x158, b4659 x158, b4694 x158, b4695 x158 |

### `gene_essentiality_price2018_ecoli`: 324 -> 324 records

Every gene under its released sysName and Table S1 name.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 324 records validated |
| L1 | `count` | pass | observed 324, expected 324 |
| L2 | `calls_match_table_s1` | pass | 324 records, one likely-essential locus each, from 324 released Keio rows |
| L2 | `evidence_matches_table_s1` | pass | 324 records carry rb_tnseq_coverage_and_fitness evidence with their Table S1 coverage, every density under 0.2 |
| L3 | `disjoint_from_the_fitness_genes` | pass | 324 likely-essential genes and 3789 genes with fitness values share none |
| L3 | `label_caveat_on_every_record` | pass | all 324 records carry the 'nearly essential' label and the 6% to 16% FDR on their perturbation description |
| L1 | `provenance_gaps` | pass | 972 documented provenance gaps over 324/324 records; 0 deferred field(s): []; 9072 undeclared None values over 18 carrier fields (top: Compound.inchi x1620, Compound.chebi_id x9... |
| L1 | `canonical_gene_names` | pass | 324 systematic names, one canonical spelling each, each current in the genome; 1 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 972 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 324 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 324 measured genes are MG1655 (the namespace of the released b-numbers) genes (>= 1.0) |
| L4 | `current_genome_genes` | pass | every one of the 324 measured systematic names is a gene of the current genome |

## 2026.10.13 - Compendium-derived: hidden assumptions and missed representation

This store is derived from Price 2018 Fitness Browser compendium (`priceMutantPhenotypesThousands2018`). Data collected from a compendium, a database aggregate or a follow-up study may embed that source's preprocessing (normalization, filtering, re-keying, merged conditions) and may lack ontological representation the primary release carries (per-sample metadata, replicate structure, the released statistic). Its records are a derived copy, and the primary release is still to be collected from source ([[torchcell.knowledge_graphs.dataset-admission-loop]], 2026.10.13).

What it may hide here: 92 of the 162 Keio samples are Wetmore 2015's experiments re-analyzed at statistics version 1.0.3 and kept only when they passed the compendium's quality filter (`fit_logratios_good`); the other 70 are attributed to Price 2018 by inference from Table S20, not by a per-sample statement. Wetmore 2015 is served from its own release (statistics 1.0.0), so those samples exist twice under two processings; the own release is the record of the measurement.

## 2026.10.15 - Where the BW25113 lesion constant lives now

`BW25113_BACKGROUND_LESIONS`, which this loader reads to tie each Table S1 row without a
BW25113 `locus_tag` to the lesion that accounts for it, moved from
`torchcell/datamodels/schema.py` to the genomes tier,
`torchcell/sequence/genome/registry.py`. The loader imports it from there and its value is
byte-identical, so the inventory, the finding and every stored record are unchanged. Read
the section above's "`schema.py` already records that ..." as the registry's comment on
the `ECOLI_K12_BW25113` set. Reasoning: [[torchcell.verification.perturbation_families]].
