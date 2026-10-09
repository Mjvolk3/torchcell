---
id: uf1b34jqgjauvtj9u6ar98g
title: Schastnaya2021
desc: ''
updated: 1791423877162
created: 1791423877162
---

## 2026.10.07 - Loader, sourcing and the schema gap the phosphomutant arm hits

`torchcell/datasets/ecoli/schastnaya2021.py` serves `MetabolomeSchastnaya2021Dataset`,
row 30 of the fifty bacterial datasets. Schastnaya et al. 2021, *Nat Commun* 12:5650,
doi:10.1038/s41467-021-25988-4, citation key `schastnayaExtensiveRegulationEnzyme2021`.

### What the stored quantity is

A **relative metabolite level**: the log2 fold change of one annotated deprotonated
ion's abundance in the mutant over its abundance in the wild type, pooled over the
strain's replicates. Verbatim, from `paper.md` (sha256
`f299eee75a1f022e08ba7f1bc444922db357f220be94f19466a36881214546c3`), Methods, "FIA
TOF-MS measurement and untargeted metabolomics data processing":

> For every ion, the abundances from all replicates of a given mutant were pooled and
> compared to the pooled abundances of the wild-type sample. The $\log _ { 2 }$ fold
> change of an ion abundance in the mutant compared to the wild-type was determined,
> and a two-sided 2-sample t-test with unequal variance was performed. The obtained
> $\boldsymbol { p }$ -values were corrected for multiple testing using the
> Benjamini-Hoch[berg procedure.]

The fifty-dataset table classes this row "Metabolome / flux". Checked against the
release: it is metabolome only. The flux arm of the paper is four strains in one
condition (Fig. 4, a source-data figure, not a released matrix) and the activity arm is
four purified enzymes (Supplementary Data 7), so neither is a dataset. It is also not a
phosphosite occupancy: Supplementary Data 1 carries a literature occupancy for nine of
the 52 sites, taken from Soares 2013 and Sultan 2021, not measured here.

### Sourcing table

Every value the loader needs, with the artifact it is pinned to. All of these are
`SOURCED_VALUES` entries and all are audited by `verify_build` (the L3
`provenance_audit` rows).

| Value | Stored as | Verbatim quote (abridged where marked) | Source + sha256 |
|---|---|---|---|
| What SD3 is | the consumed table | "File name: Supplementary data 3 Description: Changes in metabolites for phosphomutant and knockout strains" | `si/si3.md` `32bb2072...ab5e` |
| The statistic | `measurement_type` | see the block quote above | `paper.md` `f299eee7...46c3` |
| Ion annotation | the key's meaning | "Deprotonated ions were annotated based on mass using 0.001 Da tolerance using a genome-wide reconstruction model of $E$ . coli metabolism41." | `paper.md` |
| Ionization mode | note on the keys | "Mass spectra were recorded in negative ionization mode within a mass/charge ratio range of $5 0 { - } 1 0 0 0 \mathrm { m / z }$ ..." | `paper.md` |
| Call threshold | recorded, never applied | "Ions with a $\log _ { 2 }$ fold change $> \pm 0 . 3 8 $ and a corrected $\boldsymbol { p }$ - value $< 0 . 0 5$ were considered si[gnificantly changing]" | `paper.md` |
| Wild type | `BacterialStrainBackground` | "E. coli MG1655 ΔmutS with kanamycin resistance (referred to as wild-type) harboring the temperature-sensitive $\lambda$ -Red recombineering plasmid $\mathsf { p S I M } 5 ^ { 6 7 }$ with chloramphenicol resistance" | `paper.md` |
| Deletion collection | `collection` | "E. coli gene deletion mutants were retrieved from the KEIO collection38." | `paper.md` |
| Deletion background | note + `strains.csv` | "The Keio collection is comprised of 3985 deletions in duplicate (7970 total) of E. coli K-12 strain BW25113 (Datsenko and Wanner, 2000)" | Baba 2006 `paper.md` `ca71475b...6da9`. Deferral target of "KEIO collection38" |
| Cassette | `cassette` | "Open-reading frame coding regions were replaced with a kanamycin cassette flanked by FLP recognition target sites" | Baba 2006 `paper.md` |
| Media | the five `Media` | "E. coli cells were grown in M9 minimal medium supplemented with either $5 \mathrm { g / L }$ glucose, $5 \mathrm { g / L }$ fructose, $6 . 8 \mathrm { g / L }$ sodium acetate, $6 . 1 \mathrm { g / L }$ sodium pyruvate or ${ 5 . 1 \mathrm { g } } / { \mathrm` (the quote is cut at the OCR's glycerol markup) | `paper.md` |
| Culture, 37 C, aerobic | `Environment` | "E. coli strains were grown in $1 \mathrm { m L }$ M9 minimal medium supplemented with the same carbon sources used for growth analysis. Cultures were grown in deep 96-well plates at $3 7 ^ { \circ } \mathrm { C }$ and $2 5 0 \mathrm { r p m }$ until the $\mathrm { O D } _ { 6 0 0 }$ of 0.4-1.5 was reached (mid-exponential phase)." | `paper.md` |
| Extraction split | the detected-ion check | "Metabolic extracts of AcnB, Adk, AhpC, GpmM, KdsD, ManX, MetK, Pck, Pgi, Pta, and TalB mutants were prepared using the hot extraction procedure, and for other mutants cold extraction was used." | `paper.md` |
| Replicates | `n_replicates = 3` | "$n = 1 0 ,$ , 3, 4, 5 biological replicates with two technical replicates were measured for the wild-type, knockout, S115E, S115A mutants, respectively." | `paper.md`, Fig. 3b |
| Raw deposit | recorded, not mirrored | "The metabolomics data generated in this study have been deposited in the MassIVE database under accession code MSV000087795." | `paper.md` |

**Replicate count, and why 3.** The per-record count is not a released column and it
varies: Fig. 3b states 10, 3, 4 and 5 biological replicates for the wild type, the
`sucB` knockout and two phosphomutants on fructose, each with two technical replicates.
There is no companion statistic to back-solve from (the release gives the fold change
and a Benjamini-Hochberg **adjusted** p-value, and the standard deviation is not
released, so the two unknowns cannot be separated), so CLAUDE.md's rule (2) applies and
the **conservative lower end** is stored. Three is also the floor the Reporting Summary
states: its Sample size cell reads "Sample sizes were not statistically predetermined
and were chosen based on the standards in the field for producing results with
sufficient accuracy. All experiments were performed at least in triplicates, and the
sample sizes were increased whenever possible", and its Replication cell "Biological
replicates were performed independently at least three times." That form (`si/si12.pdf`,
sha256 `0102c096...c13c2`) is a scan with no text layer, so those cells survive only as
the MinerU table image
`si/images/si12/c9ce0df8de37533f33875de9d8b4f61debd4b1ac0bb6755506bab5db4f7495d1.jpg`
(sha256 `530ff753...1092`). The provenance audit reads text, so the Fig. 3b sentence is
the audited quote and the Reporting Summary is carried in the value's `note`.

### Raw mirror

`$DATA_ROOT/torchcell-raw/schastnayaExtensiveRegulationEnzyme2021/data/`, both files
retrieved by `RetrievalMethod.pmc_cloud` through
`torchcell.literature.retrieve.pmc_cloud_object` from the PMC Article Datasets bucket
(`PMC8463566.1`), re-runnable by `python -m torchcell.datasets.ecoli.schastnaya2021
deposit --retrieve`:

| File | What | sha256 | bytes |
|---|---|---|---|
| `si6.xlsx` | Supplementary Data 3, the values | `ad9fb64f29b4d7cac7c9575217f7fa8fadb401d4910fb8d330685d42b43d49d4` | 1,624,605 |
| `si4.xlsx` | Supplementary Data 1, the phosphomutant strains | `8acdfa9fcc3a09f73145e9188f11779b75da8c30cea36ebb0a120d4921a083ff` | 13,644 |

Not mirrored: MassIVE MSV000087795 (raw spectra, not this matrix), PRIDE PXD027243
(intact-protein MS of purified enzymes, a different assay) and Supplementary Data 2, 4,
5, 6, 7, 8, none of which the loader reads.

### Retention ledger

Supplementary Data 3's LOG2(FC) block is 464 ion rows by 200 (strain, carbon source)
columns, and the adjusted-p-value block repeats the same 200 columns in the same order
(the loader asserts that). The arithmetic:

```
200 released columns
- 171 phosphomutant columns   (phosphosite_substitution_has_no_bacterial_leaf)
=  29 stored records          (16 KEIO deletion strains over four carbon sources)
```

The 171 are 87 distinct phosphomutant strains, one ledger item each, naming the strain's
enzyme b-number and whether the substitution abolishes or mimics phosphorylation
(`preprocess/dropped_records.json`). Supplementary Data 1 lists 89 phosphomutants; the
two with no metabolome column are `Icd-S113A` and `Icd-S113E`, whose mutation of a
catalytic residue the paper reports as condition-specific lethality.

**Why they are dropped, which is the finding.** The perturbation of a phosphomutant is a
genomic codon substitution made by MAGE ("Genomic point mutations of $E _ { * }$ coli
phosphosites were constructed using MAGE34."). The schema has no bacterial
sequence-level perturbation leaf:

- `AllelePerturbation` is the amino-acid-substitution leaf, but it inherits
  `GenePerturbation.validate_sys_gene_name`, which matches only R64 ORF names.
  `AllelePerturbation(systematic_gene_name="b2029", perturbed_gene_name="gnd")` raises
  `ValidationError: Invalid systematic gene name format`.
- The five leaves that DO take a bacterial locus tag are `bacterial_deletion`
  (`state="absent"`), `transposon_insertion`, `bacterial_crispr_interference`,
  `promoter_replacement` and `heterologous_pathway`. Each asserts a genotype these
  strains do not have.
- `BacterialBackgroundAllele` does carry `AlleleEdit.sequence_variant` on a locus tag,
  but it is a property of a strain BACKGROUND, not a perturbation. Putting the
  phosphosite substitution there would leave the record's `Genotype` empty, which makes
  the mutant indistinguishable from the wild type.

So the honest shape is a drop with a named rule, not a leaf that misstates 171 records.
What would unblock them is one new leaf, a bacterial sibling of `AllelePerturbation`
with the `_validate_bacterial_locus_tag` validator and a `gene_namespace`, plus a field
for the substitution itself (residue, position, from and to). That is a `schema.py`
change and therefore a separate, reviewed PR.

### Two further gaps, recorded rather than forced

1. **The adjusted p-value has no field.** Every stored value has one, and
   `MetabolitePhenotype` has no p-value or q-value field (only
   `GeneInteractionPhenotype` carries a p-value anywhere in the schema). It is NOT a
   standard error, so it does not go in `metabolite_level_se`; that field is a typed
   `ProvenanceGap`. The 464 x 29 matrix is written to
   `preprocess/adjusted_p_values.csv` so the information is not lost.
2. **The deletion strains and the control have different backgrounds.** The fold change
   compares a Keio BW25113 deletion against an MG1655 ΔmutS wild type, and
   `BacterialDeletionPerturbation` has no field for a perturbed strain's own background
   (the background lives on the reference, which describes the control). The record pins
   MG1655, which is both the control strain and the namespace Supplementary Data 1's own
   b-numbers are written in, and `preprocess/strains.csv` carries
   `deletion_background=BW25113` beside `reference_background=MG1655 dmutS kan` on every
   row.

### Identifiers

Gene identity: the release names a deletion only by gene symbol (`gnd knockout`). All
16 symbols go through `reconcile_locus_tags` on the pinned MG1655 GenBank annotation and
all 16 resolve (status `renamed`, layer `gene symbol`, 0 retired, 0 ambiguous, 0 outside
the namespace), giving `aceA b4015, acnB b0118, ahpC b0605, fbaB b2097, gnd b2029,
gpmA b0755, gpmM b3612, kdsD b3197, manX b1817, pck b3403, pgi b4025, pta b2297,
pykA b1854, pykF b1676, sucB b0727, talB b0008`. `aceA -> b4015` agrees with
Supplementary Data 1's own "Systematic protein name" for AceA, which is the independent
check that the symbol route lands in the namespace the paper used. A symbol that did not
resolve would be a build error, not a drop.

**Metabolite identity (Rapp 2026's rule).** The stored key is the released `Formula`,
the neutral formula of the deprotonated ion, which is unique over the 464 rows (asserted
at build time). `target_metabolite_ids` maps a key to its KEGG id only where the row
names exactly ONE compound; a row whose mass cannot separate several KEGG compounds is
ABSENT from the map rather than assigned one of its candidates, and the full candidate
set of every such row is in `preprocess/metabolite_identity.json`. Counts:

| Candidates per ion | Ions |
|---|---|
| 1 (mapped) | 350 |
| 2 | 65 |
| 3 | 25 |
| 4 | 11 |
| 5 | 2 |
| 6 | 3 |
| 7 | 1 |
| 8 | 4 |
| 9 | 1 |
| 11 | 1 |
| 18 | 1 |

350 + 114 = 464; coverage 0.7543. The widest group is `C6H13O9P`, 18 KEGG compounds of
equal mass spanning the hexose 6-phosphates (glucose, fructose, mannose, galactose,
allose, tagatose) and the inositol monophosphates. A typed
gap cannot sit beside a partially populated map (`ProvenanceGapMixin` requires a gapped
field to be `None`, issue #753), so the uncovered keys live in the ledger, this note and
the PR, not in a gap on the field.

### Detection is a batch property, and it matches the paper's extraction split

37 percent of the matrix is `nd`. Those cells are not stored: the release states no
value for them, so a record carries only its detected ions. Measured on the release,
among the 29 kept columns the detected-ion SET depends only on (extraction protocol,
carbon source) and never on the strain, which is exactly what the paper's hot/cold
extraction list predicts. The build refuses any other split.

| Extraction | Carbon source | Strains | Detected ions |
|---|---|---|---|
| hot | GLUCOSE | 9 | 332 |
| hot | ACETATE | 7 | 309 |
| cold | GLUCOSE | 7 | 284 |
| cold | FRUCTOSE | 4 | 284 |
| cold | ACETATE | 1 | 284 |
| cold | GLYCEROL | 1 | 284 |

9 x 332 + 7 x 309 + 13 x 284 = 2,988 + 2,163 + 3,692 = **8,843 stored values**, which is
the count the L2 row reports.

### Build

```
python -m torchcell.datasets.ecoli.schastnaya2021 deposit
python -m torchcell.database.build_dataset_lmdb --dataset MetabolomeSchastnaya2021Dataset
python -m torchcell.datasets.ecoli.schastnaya2021 verify
```

| | |
|---|---|
| Records | 29 |
| Stored values | 8,843 |
| Gene set | 16 MG1655 b-numbers |
| References | 1 `AssemblyReferenceGenome` (`GCA_000005845.2`, background `MG1655 dmutS kan`), 6 distinct `(medium, key set)` reference phenotypes |
| Media | 4 of the 5 built (no deletion strain was profiled on pyruvate) |
| Wall time | 7.4 s (dev tree, after deprecating the previous `processed/` and `preprocess/`) |
| Store size | 580 KB |
| `python -m torchcell.provenance.build_manifest` | `fresh` |

### Verification

The metabolite family verifier keys L1 uniqueness on the GENOTYPE alone, and this
dataset is one record per (genotype, carbon source), so `verify_build` composes the
rows in the loader instead: `l0_structural`, `l1_count`, a (deletion set, medium)
uniqueness row, `l2_value_fidelity`, a reference-zero row that also asserts the
reference's keys are the record's own keys, a measurement-type row, the L4 MG1655
containment and the provenance audit of all 20 sourced values.

```
metabolome_schastnaya2021: PASS
  [ok] L0 structural: 29 records validated
  [ok] L1 count: observed 29, expected 29
  [ok] L1 genotype_environment_uniqueness: 29 unique (deletion set, medium) pairs, one record each
  [ok] L2 value_fidelity: 8843 values checked
  [ok] L3 reference_zero: reference level == 0 on the record's own keys for all 8843 values
  [ok] L3 measurement_type_consistent: single measurement_type: 'fia_tof_ms_ion_log2_fold_change_vs_wild_type'
  [ok] L3 provenance_audit x 20
  [ok] L4 gene_containment_mg1655_b_numbers: 16 of 16 deleted loci are MG1655 GenBank gene rows
```

### Adapter

`torchcell/adapters/schastnaya2021_adapter.py` serves `MetabolomeSchastnaya2021Adapter`
over `torchcell/adapters/conf/metabolome_schastnaya2021_adapter.yaml`, the same node and
edge set as the Fuhrer 2017 metabolome: one `bacterial perturbation` per record, a
`metabolite phenotype`, and no environment-perturbation pair, because the carbon source
is part of the medium. The dataset is case 11 of `BACTERIAL` in
`tests/torchcell/adapters/_bacterial_adapter_cases.py` and entry 11 of the E. coli block
of `kg_bacteria.yaml`, which the gate asserts positionally.
