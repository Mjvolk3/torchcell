---
id: pk93bs6uiog4jmgkgw0ljhk
title: Rapp2026
desc: ''
updated: 1791407580253
created: 1791407580253
---

## 2026.10.07 - Rapp 2026 CRISPRi metabolome loader

Row 23 of the fifty bacterial datasets ([[plan.bacteria-ontology-genome]]).
`MetabolomeRapp2026Dataset` in `torchcell/datasets/ecoli/rapp2026.py`, the second
bacterial metabolome after [[torchcell.datasets.ecoli.fuhrer2017]] and the first
CRISPRi one.

### What the paper measured

Rapp et al. sorted a pooled CRISPRi library into an arrayed one covering all 1,515
genes of the iML1515 model, grew every strain on M9 glucose at 37 C with 200 nM
anhydrotetracycline inducing a genome-integrated dCas9, and profiled 3,026 metabolite
extracts by flow-injection mass spectrometry. Each m/z feature annotated to an iML1515
metabolite is released as a linear fold change relative to the per-batch median.

### Record type

`BacterialMetaboliteExperiment` / `BacterialMetaboliteExperimentReference` carrying a
`MetabolitePhenotype`, the same pair Fuhrer 2017 uses. The readout is a quantitative
per-strain metabolite profile keyed by metabolite id, which is exactly what
`MetabolitePhenotype` holds, and the bacterial pair pins the assembly the genotype is
written against. The perturbation is `BacterialCrisprInterferencePerturbation` with
`gene_namespace="ecoli_k12_mg1655_bnumber"` and a `CrisprConstruct` carrying
`effector="dCas9"`, the 20-nt spacer from Table S1 and `n_guides=1`.

### Sourcing table

Every quote is a verbatim substring of the pinned `paper.md`
(sha256 `3c63b665c7e69d8e433f3b5a48a579a01956be919bc22669cbc789bf74d643e5`, MinerU OCR of
the publisher PDF in the literature mirror). The OCR carries the source's LaTeX math
markup, so a quote keeps it verbatim and the plain reading lives in
`SourcedValue.value`. All 24 entries of `SOURCED_VALUES` pass `audit_sourced_value`.

| key | value | where in the paper | quote (abridged) |
|---|---|---|---|
| `library_genes` | 1515 | Results, first paragraph | "arrayed CRISPRi library that targets all 1,515 genes in the iML1515 genome-scale model" |
| `effector` | dCas9 | Results, first paragraph | "an anhydrotetracycline (aTc)-inducible dCas9 on the genome and an sgRNA on a plasmid" |
| `host_strain` | YYdCas9 | Strains and culture | "E. coli YYdCas9 strain19 was the wild-type strain used in this study" |
| `host_genotype` | BW25993 intC:tetR-dcas9-aadA lacY:ypet-cat | KEY RESOURCES TABLE | "YYdCas9: BW25993 intC:tetR-dcas9-aadA lacY:ypet-cat" |
| `gene_namespace_host` | MG1655 | E. coli metabolic pathways and reactants | "Pathways of E. coli K-12 substr. MG1655 were extracted from the EcoCyc database" |
| `guide_choice` | 1 guide per gene | Construction of the sorted CRISPRi library | "the sgRNA closest to the start codon was chosen for each gene" |
| `guide_plasmid` | pgRNA-bacteria (Addgene #44251) | KEY RESOURCES TABLE | "pgRNA-bacteria ... Addgene plasmid #44251" |
| `control_sgrna` | empty sgRNA | Construction of the sorted CRISPRi library | "Position A1 of each 96-deep-well plate contains the control strain, which expresses an empty sgRNA" |
| `control_replicates` | 15 | Results | "15 replicates of a control strain carrying a non-targeting sgRNA" |
| `n_biological_replicates` | 2 | Results | "Each CRISPRi strain was sampled twice from two independent plates" |
| `od_filter` | 17 strains | Results | "17 did not reach the required OD ... and were not further analyzed" |
| `n_extracts` | 3026 | Results | "screen the 3,026 metabolite extracts (1,498 CRISPRi strains and 15 control strains, each in biological duplicates)" |
| `temperature_c` | 37.0 | Metabolome sampling | "incubated for 6.5 h at 37 C, 220 rpm" |
| `duration_hours` | 6.5 | Metabolome sampling | same quote |
| `aerobic_shaking` | aerobic | Metabolome sampling | same quote |
| `media_recipe` | M9 + 5 g/L glucose | Media | "M9 minimal medium with glucose as the sole carbon source (5 g/L). M9 medium contained (per liter): 7.52 g Na2HPO4 2H2O, 5 g KH2PO4, 1.5 g (NH4)2SO4, 0.5 g NaCl ..." |
| `glucose_preculture` | 1.75 g/L | Metabolome sampling | "Overnight cultures were prepared in M9 with 1.75 g/L glucose" |
| `selection_and_inducer` | 100 ug/mL Amp, 200 nM aTc | Media | "LB, LB agar, and M9 media contained 100 ug/mL ampicillin (Amp). Anhydrotetracycline (aTc) was added to a final concentration of 200 nM" |
| `isobaric_metabolites` | 802 | Processing of data from FI-MS | "all isobaric metabolites were merged resulting in 802 metabolites with unique monoisotopic masses (Table S9)" |
| `adducts` | 6 named | Processing of data from FI-MS | "annotated in their single protonated form ([M+H]+) or their deprotonated form ([M-H]-) ... [M+2H]2+, [M+3H]3+ and [M-2H]2-, [M-3H]3-" |
| `normalization` | linear fold change vs the per-batch median | Processing of data from FI-MS | "The intensity of the annotated m/z features was used to calculate fold changes relative to the median on a per batch basis (Table S4)" |
| `imputation` | baseline imputed per batch | Processing of data from FI-MS | "Samples with missing m/z features were baseline imputed if at least one sample of the batch had an m/z feature with a peak height and prominence of at least 5000" |
| `replicate_noise` | 1.7 log2 units | Results | "99% of the variability between biological replicates being smaller than a log2-fold change of 1.7" |
| `data_availability` | 3 MassIVE accessions | Data and code availability | "FI-MS data are accessible with the accession number MSV000098712 ..." |

### The statistic, and why it is the paper's own

The column the loader consumes is Table S4's per-sample value, a LINEAR fold change
relative to the median of that sample's batch. Three measurements settle what it is:

1. **It is a ratio, not a log.** Within each of the 6 batches, every populated feature's
   median over that batch's sample columns is exactly 1.0
   (`preprocess/batch_normalization.json`). A log scale would give 0.
2. **The stored level is the paper's own per-strain statistic.** The arithmetic mean of a
   strain's two plate values reproduces Table S5's `Mean_FC` for all 1,385 accumulating
   strain-metabolite pairs with a maximum absolute difference of **0.0**
   (`preprocess/mean_fc_crosscheck.json`). Table S5's `R1_FC` / `R2_FC` are also
   bit-identical to Table S4's two per-sample values, so Table S5 is a selection of
   Table S4 rather than a separate quantification.
3. **The uncertainty type.** No uncertainty column is released. The two plate values ARE
   released, so `metabolite_level_se` is the standard error of the stored mean over
   those two biological replicates (`|r1 - r2| / 2`), in the same units as the value.
   `measurement_type` names the statistic in full:
   `fi_ms_iml1515_feature_fold_change_vs_batch_median_mean_of_2_plates`.
   R1 / R2 are two independent cultivation plates, not two injections of one extract:
   the two FI-MS injections per extract are the two ionization polarities.
   The only spread the paper states is dataset-level (`replicate_noise`) and is stored
   on no record.

### Metabolite identity

The identity layer is Table S9 (`si10.xlsx`), the 802 isobaric groups of unique
monoisotopic mass, each with a BiGG id, a KEGG id, the monoisotopic mass and the neutral
formula. The join is exact and checked three ways at build time: every Table S4
abbreviation is a Table S9 abbreviation (802 of 802, no leftovers either way), Table S4's
own `Kegg` string equals Table S9's for every one of the 1,880 feature rows, and every
`[M+H]+` feature's m/z is its monoisotopic mass plus 1.00728 Da to within 2e-5.

- Metabolite keys are Table S4's `Abbr` **verbatim**: the group abbreviation plus the
  adduct (`frdp[M-H]-`), so two adducts of one metabolite stay two measured features.
- `target_metabolite_ids` maps a key to its single BiGG id wherever the group names
  exactly ONE metabolite: **1,077 of the 1,321 stored features** (594 of the 723 covered
  iML1515 groups).
- The remaining **244 features (129 groups)** are merged isobaric sets: FI-MS cannot
  separate equal masses, so the group is a set of 2 to 11 candidates and no single id
  covers it. Those keys are ABSENT from `target_metabolite_ids` rather than assigned one
  of their candidates. `preprocess/metabolite_identity.json` lists every such key with
  its full BiGG candidate tuple, and `preprocess/metabolites.csv` carries the whole
  table (key, abbreviation, adduct, m/z, monoisotopic mass, formula, KEGG, BiGG ids,
  isobaric-set size, target id, names).
- **Why the record carries no `ProvenanceGap` on that field.** `ProvenanceGapMixin`
  requires a gapped field to be `None` ("cannot both store a value and declare it
  missing"), so a partial map and a typed gap are mutually exclusive on
  `MetabolitePhenotype`. Keeping the 1,077 sourced identities was the choice; the 244
  uncovered keys are recorded in the build ledger instead. The key itself still carries
  the released candidate set verbatim, so a reader of the record can see that
  `cellb-lcts-malt-melib-sucr-tre[M-H]-` names six metabolites.
- Isobaric-set-size histogram over the stored features:
  `{1: 1077, 2: 168, 3: 47, 4: 8, 5: 8, 6: 4, 7: 3, 8: 2, 10: 2, 11: 2}`.
- **One arity anomaly, found by the build refusing it.** Table S9's `Metabolite` column
  separates isobaric members with a semicolon and a space, but row `didp` reads
  `DIDP; 2'-deoxyinosine-5'-diphosphate(3-)`, which is one metabolite under one BiGG and
  one KEGG id. The `Abbreviation`, `BIGG` and `KEGG` fields agree on the arity for all
  802 rows, so the loader counts the group by its id list and keeps `names` verbatim
  without counting it.
- **Not released as [M+2H]2+.** The Methods name `[M+2H]2+` among the multiply charged
  adducts, but Table S4 carries no such row: measured adduct counts are 802 `[M+H]+`,
  802 `[M-H]-`, and 92 each of `[M+3H]3+`, `[M-2H]2-` and `[M-3H]3-`, totaling the
  1,880 feature rows. Recorded, not resolved.

### Strain and assembly pin

The host is `YYdCas9`, written `BW25993 intC:tetR-dcas9-aadA lacY:ypet-cat` and deferred
to Lawson 2017, which is NOT in the literature mirror. BW25993 has no deposited assembly
set, and every identifier the paper releases is an iML1515 b-number, so:

- `REFERENCE_STRAIN = "MG1655"`, `gene_namespace="ecoli_k12_mg1655_bnumber"`, and the
  records pin `GCA_000005845.2` (`ecoli_K12_MG1655_ASM584v2`), the sha256-pinned bytes
  those b-numbers mean.
- `AssemblyReferenceGenome.background` is a `BacterialStrainBackground` named `YYdCas9`
  with `parents=["BW25993"]`, the key-resources genotype string as `genotype_statement`,
  `alleles=[]` (the source states only a genotype string) and a typed `ProvenanceGap` on
  `construction` (`not_reported_by_primary`): no construction step, and none of
  BW25993's own lesions, is quotable from this paper.

### Identifier reconciliation

`reconcile_locus_tags` on the MG1655 GenBank annotation, over the 1,497 distinct
b-numbers Table S3 releases (`preprocess/identifier_reconciliation.json`):

| status | count |
|---|---|
| current | 1,495 |
| renamed | 0 |
| non_gene_feature (pseudogene locus) | 2 |
| retired | 0 |
| ambiguous | 0 |

Resolved fraction **1.000** (threshold `MIN_RESOLVED_FRACTION = 0.99`). Resolver layers:
1,496 by locus tag, 1 by gene synonym. One b-number is remapped (`b4104` to `b4583`), and
0 stored names fall outside the namespace. Two released gene SYMBOLS do not resolve to
their own b-number's locus (`flc` and `rhmA`, both retired symbols on this annotation);
the record stores the b-number's locus and `canonical_symbol` either way, and the
disagreements are ledgered.

### Retention ledger

| step | count |
|---|---|
| Table S4 strain tokens | 1,513 |
| control strains (`ctrl1`, `ctrl3` .. `ctrl16`; `ctrl2` is absent) | 15 |
| source records (1,513 - 15) | 1,498 |
| dropped: `no_target_gene_assigned` | 1 |
| dropped: `b_number_remapped_by_the_annotation` | 1 |
| **kept records** | **1,496** |

- `no_target_gene_assigned` -- `argR` (plate 4, well D12) carries the controls' `b0000`
  placeholder in Table S3 and has no Table S1 sgRNA, so neither the knocked-down locus
  nor the guide spacer is released. It is one of the paper's 1,498 analyzed strains: the
  paper's 17 OD failures leave 1,498, while Table S1 has 18 genes with no Table S4
  sample, and `argR` is exactly that one-strain difference. Recorded, not resolved.
- `b_number_remapped_by_the_annotation` -- `phnE` carries `b4104`, which the MG1655
  annotation does not hold as a locus tag but as a `/gene_synonym` of the pseudogene
  locus `b4583` (`phnE1`, the IS-interrupted K-12 allele). See the finding below.

### Finding for the owner: a missing `DerivedIdentifierRoute` member

`DerivedIdentifierRoute` is `Literal["eck_crosswalk", "jw_synonym", "gene_symbol"]`, and
`DerivedIdentifierMapping` validates that the released identifier has the form its route
reads: `eck_crosswalk` requires another strain's locus tag, `jw_synonym` a Keio `JW` id,
and `gene_symbol` refuses a locus tag. `b4104` is a locus tag of the pinned strain's OWN
namespace that the annotation retired into a `/gene_synonym` of `b4583`, so no route
describes the remap and the mapping cannot be recorded on the record. Storing `b4583`
anyway would be exactly the silent derived mapping section 4 of the plan forbids, and
storing `b4104` verbatim would fail L4 containment (it is not a gene row). A fourth
member -- `locus_tag_synonym`: the source released a retired tag of the pinned strain's
own namespace that the annotation carries as a `/gene_synonym` of exactly one current
locus -- would recover this record and every future one like it. `schema.py` was left
untouched for this row, so the deliverable is this finding plus the one dropped record.

### Media

`MEDIA_LIBRARY` has no entry for Rapp's M9 ([[torchcell.datamodels.media]] lists it as a
deliberate deferral), and `media.py` is a value surface, so the medium is built in the
loader as `SCREEN_MEDIA` with `base_medium="M9"` (which resolves in the library, so the
record still joins the M9 family). Every compound it needs already resolves through
`resolved_compound`, so no value surface was touched: 15 typed components, every one with
a concentration, so `Media.is_fully_characterized` is True. The separately sterilized
additions carry the stock-times-volume arithmetic in `MediaComponent.note` (CaCl2 0.1 mM
from 1 mL of 0.1 M; MgSO4 1 mM from 1 mL of 1 M; FeCl3 60 uM from 0.6 mL of 0.1 M;
thiamine-HCl 2.8 uM from 2 mL of 1.4 mM; each trace salt at 1/100 of its stock), and a
hydrate in the recipe is typed as the anhydrous salt with that said in the note.
Ampicillin is a `selection_agent` component at 100 ug/mL; the 200 nM anhydrotetracycline
inducer is a `SmallMoleculePerturbation` on the environment, since it is what realizes
every knockdown. **For the PR:** `M9_GLUCOSE_RAPP2026` belongs in `MEDIA_LIBRARY`; the
loader carries the recipe inline until it is added there.

### Raw mirror

`$DATA_ROOT/torchcell-raw/rappMetabolomeColiCRISPRi2026/` holds the five workbooks the
loader consumes, each with a `RetrievalRecord` naming
`torchcell.literature.retrieve.elsevier_mmc` (the Elsevier CDN serves these directly;
ScienceDirect answers scripted clients with 403), so the retrieval re-runs.

| mirror path | SI table | bytes | sha256 (first 12) | why it is consumed |
|---|---|---|---|---|
| `data/si2.xlsx` | Table S1 (`mmc2`) | 107,114 | `1057e6db44c5` | the sgRNA spacer of every library gene |
| `data/si4.xlsx` | Table S3 (`mmc4`) | 199,690 | `4a7fd186daba` | sample id to strain, b-number, OD, plate, well |
| `data/si5.xlsx` | Table S4 (`mmc5`) | 43,827,170 | `fdc5ac2c759a` | the values |
| `data/si6.xlsx` | Table S5 (`mmc6`) | 318,926 | `f3a03ddce7b4` | the `Mean_FC` cross-check only |
| `data/si10.xlsx` | Table S9 (`mmc10`) | 70,584 | `46aed36e634f` | the metabolite-identity layer |

`NOT_MIRRORED` records what is deliberately left out: the three MassIVE deposits (raw
spectra, not these matrices), the four SI PDFs (already in the literature mirror) and the
eight SI workbooks the loader does not read.

### Build

```
PYTHONPATH=$PWD python -m torchcell.database.build_dataset_lmdb \
  --dataset MetabolomeRapp2026Dataset
```

| quantity | value |
|---|---|
| records | 1,496 |
| gene-set size | 1,496 |
| references | 1 |
| metabolite keys per record | 1,321 |
| stored values (level, SE, n) | 1,976,216 each |
| wall time | 63 s |
| store size | 112 MB |
| build manifest | `fresh` under `python -m torchcell.provenance.build_manifest` |

1,321 of the 1,880 annotated feature rows carry values; the other 559 are empty in every
one of the 3,026 columns, and no row is partial (per-batch baseline imputation makes a
detected feature present for every sample, and `populated_features` refuses a partial
row rather than storing a key whose absence differs per strain).

### Verification

`python -m torchcell.datasets.ecoli.rapp2026 verify` and
`runners.run_metabolite` both PASS:

- L0 structural: 1,496 records validated
- L1 count: 1,496 observed, 1,496 expected
- L1 genotype_uniqueness: 1,496 unique strains, one record each
- L2 value_fidelity / se_nonnegative: 1,976,216 values each
- L3 reference_finite: finite and key-subset for all 1,976,216 values
  (`reference_centered=False`: the reference is the measured control profile on the
  per-batch-median scale, not a 0-centered score)
- L3 measurement_type_consistent: one `measurement_type`
- L3 provenance_audit: 24 of 24 sourced values backed by a verbatim quote
- L4 gene_containment: 1,496 of 1,496 knocked-down loci are MG1655 GenBank gene rows

### Tests

`tests/torchcell/datasets/ecoli/test_rapp2026.py`: 40 hermetic + 28 data-gated, 68
passing with `--data`. The hermetic build runs the real `EcoliK12MG1655Genome` over a
synthetic MG1655 assembly derived from `_bacterial_fixtures.MG1655_LOCI` with `b0099`
added as a `gene_synonym` of `b0005`, so both drop rules are exercised on a six-strain
screen. Hermetic line+branch coverage of the loader is 94%.
