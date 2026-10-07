---
id: mcnio9d1whn99onxi6kj0d4
title: Wang2018
desc: ''
updated: 1791406852957
created: 1791406852957
---

Row 25 of the fifty bacterial datasets ([[plan.bacteria-ontology-genome]]): Wang et al.
2018, "Pooled CRISPR interference screening enables genome-scale functional genomics
study in bacteria with superior performance", Nat Commun 9:2475,
doi:10.1038/s41467-018-04899-x, citation key `wangPooledCRISPRInterference2018`.

Loader: `torchcell/datasets/ecoli/wang2018.py`, class
`CrispriGuideFitnessWang2018Dataset`. Tests:
`tests/torchcell/datasets/ecoli/test_wang2018.py`.

## 2026.10.07 - Loader, sourcing and the first build

### What one record is

One record is one `(sgRNA, screen)` row of Supplementary Data 6 to 10: the per-guide
`sgRNA fitness` of one pooled CRISPRi screen. The paper ran five screens of one
genome-scale library (55,671 gene-targeting guides plus 400 non-targeting controls),
so the released grain is guide-by-screen, not gene-by-screen.

### Record type, and why it is not a `FitnessPhenotype`

`BacterialEnvironmentResponseExperiment` with an `EnvironmentResponsePhenotype`,
`measurement_type=log2_ratio`, `assay_type=pooled_competitive_growth_barcode`.

The released statistic is signed: Methods equation (2) is
`Log2((Read count)_selective / (Read count)_control)` and equation (3) subtracts the
median of the non-targeting guides. The essentiality screen spans -9.82 to +3.92.
`FitnessPhenotype` is a strictly positive ko/wt growth ratio that clamps non-positive
values to 0 and whose verifier requires a 1.0 reference, so it cannot hold this number.
The reference's `environment_response` is exactly 0.0, which is what equation (3) makes
it.

The genotype is one `BacterialCrisprInterferencePerturbation` per gene the guide
represses, with `gene_namespace="ecoli_k12_mg1655_bnumber"`,
`CrisprConstruct(effector="dCas9", guide_sequence=<the 20-mer>, n_guides=1,
library_pool=None)`. The spacer is load-bearing: the verifier's L1 strain key joins
`crispr.guide_sequence`, so without it the up-to-15 guides of one gene would collapse
into one strain and 14 of them would read as duplicates.

`assay_type=pooled_competitive_growth_barcode` is exact rather than approximate here:
the molecular barcode the abundance is read from IS the guide's own N20 spacer, which the
pipeline extracts out of the `GCACN20GTTT` 28-mer.

### Sourcing table

Every quote is a verbatim substring of the mirrored `paper.md`,
sha256 `9980415d606835ab1ae3a1ad92a64784c822aad1ca01b514662d2fb21d1a3a55` (MinerU OCR of
the publisher PDF in `$DATA_ROOT/torchcell-library/wangPooledCRISPRInterference2018/`).
The OCR renders math as TeX, so some quotes carry its markup; they are stored as they
appear in those bytes.

| field | value | section | verbatim quote |
|---|---|---|---|
| reference strain | MG1655 (`GCA_000005845.2`) | Results, genome-wide screens | "We transformed the sgRNA library by electroporation into E. coli strain MCm (a K12 MG1655 derivative with an integrated chloramphenicol-resistance cassette) carrying pdCas9-J23111" |
| annotation | NC_000913.3 | Methods, sgRNA library design | "The E. coli K12 MG1655 genome sequence and relevant protein- or RNA-coding gene annotation of NC_000913.3 was used for the sgRNA library (20-mer) design" |
| effector | dCas9 | Methods, strain and plasmid construction | "The dCas9 expression plasmid was constructed by replacing the promoter and resistance marker region of Addgene plasmid ... with a constitutive promoter (wild-type promoter for Cas9 from Streptococcus pyogenes)" |
| `n_samples` | 2 | Methods, screening experiments | "The library was independently transformed twice into either MCm/pdCas9-J23111 or MCm/pKanaNC, providing two biological replicates for each" |
| `sample_unit` | `biological_replicate` | Methods, screening experiments | "All experiments were carried out with two biological replicates" |
| replicate combination | geometric mean of read counts | Methods, NGS data processing | "Subsequently, the read counts for each sgRNA in two biological replicates were averaged as the geometric mean" |
| statistic | log2 ratio, control-median normalized | Methods, NGS data processing | "by dividing the number of reads for this sgRNA in the corresponding selective condition by the number of reads in the relevant control condition and subsequently took the Log2 value (equation 2). Then, the median of fitness for all negative control sgRNAs was determined and was used to normalize the fitness data for all sgRNAs" |
| Z scale | fitness / sigma(NC fitness) | Methods, NGS data processing | "we first fit the fitness for all negative control sgRNAs to a normal distribution, giving rise to the standard deviation ... The Z score for each sgRNA was then calculated by dividing the sgRNA fitness by the sigma value (equation 4)" |
| quality flag | control reads >= 20 | Methods, NGS data processing | "We annotated the quality of the sgRNA fitness by checking the read counts for each sgRNA in the control condition. Those sgRNAs with <20 reads were eliminated from the following analysis to calculate the gene fitness." |
| library size | 55,671 + 400 | Results, genome-wide library | "we designed a genome-scale CRISPRi sgRNA library consisting of 55,671 sgRNAs (Supplementary Data 3 and 4), as well as 400 negative control sgRNAs" |
| cluster rule | one guide per cluster, all members | Results, genome-wide library | "for genes with multiple copies in the genome, we used the BLASTN program to categorize genes with highly similar sequences into clusters (Supplementary Data 2) and designed sgRNAs to target all members of a cluster. Hence, genes in one cluster are regarded as functionally identical" |
| temperature | 37 C | Methods, DNA manipulations and reagents | "MOPS medium was prepared according to standard laboratory techniques53 (10 g/L glucose). All cultures were carried out at 37 C" |
| MOPS carbon source | 10 g/L glucose | Methods, DNA manipulations and reagents | same quote; reference 53 is "Neidhardt, F. C., Bloch, P. L. & Smith, D. F. Culture medium for enterobacteria. J. Bacteriol. 119, 736-747 (1974)" |
| antibiotics | kanamycin 50, ampicillin 100 mg/L | Methods, DNA manipulations and reagents | "Antibiotic concentrations for kanamycin and ampicillin were 50 and 100 mg/L, respectively." |
| casamino acid | 0.5 g/L, tryptophan-free | Results, metabolic network | "we performed another screening with MOPS medium supplemented with 0.5 g/L casamino acid, which is composed of all amino acids except for tryptophan" |
| chemical doses | furfural 0.4, isobutanol 4 g/L | Results, chemical tolerance | "We performed screening in MOPS medium with 0.4 g/L furfural or 4 g/L isobutanol to profile the chemical-tolerance profile in E. coli at the genome level" |
| outgrowth | 9 h, ~15 doublings | Methods, screening experiments | "incubated with 100 mL LB broth (with kanamycin and ampicillin) in a 500-mL flask with shaking at 37 C until OD600 ~1.0 was reached (~9 h), allowing for around fifteen doublings" |
| screen exposure | ~5 doublings | Methods, screening experiments | "we cultivated these cultures to OD600 of ~1.0, thus allowing the cells to reproduce for around five doubling times for each experiment" |
| initial library | the mixed dCas9 replicates | Methods, screening experiments | "The cultures representing the two replicates of MCm/pdCas9-J23111 with the sgRNA library were also mixed together, serving as the initial library for the following phenotypes to be tested (Supplementary Fig. 7)" |
| barcode | the guide's N20 spacer | Methods, NGS data processing | "Customized python scripts were then used to extract the 20-mer variable sequences from the raw NGS data via searching for the 'GCACN20GTTT' 28_mer in the sequencing reads (and the reverse complementary sequence)" |
| single pool | 1 | Methods, sgRNA library design | "we divided the genome-scale sgRNA library into ten sublibraries according to the functions of the corresponding gene products (Supplementary Data 14). Customized barcode sequences were accordingly incorporated within the region flanking the N20 variable part of the library, enabling PCR amplification of these libraries separately from pooled DNA oligomers based on customized primers." |
| raw reads | PRJNA450392 | Methods, Data availability | "NGS raw data of CRISPR screening results for the tiling library and genome-scale library can be accessed from the NCBI Short Read Archive with BioProject ID PRJNA450392" |

The five screens come from Table 2, quoted cell by cell:

| screen_id | Table 2 Phenotype | Selective condition | Control condition | Supplementary Data | `nc_sigma` |
|---|---|---|---|---|---|
| `essentiality` | Essentiality | `dCas9, LB` | `Empty plasmid, LB` | 6 | 0.8490671688617 |
| `auxotrophy` | Auxotrophy | `MOPS` | `LB` | 7 | 1.0761377330569 |
| `trp_biosynthesis` | L-Trp biosynthesis | `0.5 g/L casamino acid, MOPS` | `LB` | 8 | 0.7305571957057 |
| `furfural_tolerance` | Furfural tolerance | `0.4 g/L furfural, MOPS` | `initial, see Supplementary Fig. 7` | 9 | 1.8472428072314 |
| `isobutanol_tolerance` | Isobutanol tolerance | `4 g/L isobutanol, MOPS` | `initial, see Supplementary Fig. 7` | 10 | 1.6023976401506 |

### The sigma back-solve, which is the one value the paper does not print

Equation (4) divides the fitness by one sigma per screen and the release prints both
columns, so the ratio recovers sigma exactly. **Measured on the pinned bytes:
`fitness / Z` is constant within each screen to 2.1e-12** (the spread over 54,116 rows
of the essentiality screen), which is floating-point noise. The five values are the
table above; `read_screen` recomputes each one at build time and refuses a screen whose
ratio is not flat, so a release that stopped being equation (4) would stop the build
rather than be averaged.

Two consequences. The Z score carries no information the stored fitness does not, so it
is not stored twice. And sigma is the spread of the NULL distribution (the non-targeting
guides), NOT this measurement's uncertainty, so
`environment_response_uncertainty`, `environment_response_uncertainty_type` and
`environment_response_se` are typed `ProvenanceGap`s. The two replicates are combined as
a geometric mean of read counts BEFORE the ratio is taken, so no per-guide dispersion
survives into the release at all; this is a measured absence, not an unexamined one.

### What is in the medium, and the one `MEDIA_LIBRARY` addition

- `LB` (existing) for the essentiality screen (with kanamycin 50 ug/mL and ampicillin
  100 ug/mL) and for every control culture.
- `MOPS_MINIMAL` (existing, Neidhardt 1974, carbon-free) for auxotrophy, furfural and
  isobutanol, with the stated 10 g/L glucose as
  `EnvironmentPhysicalPerturbation(factor=carbon_source, agent=D-glucose,
  magnitude=10 g/L)`. This is the Tong 2020 and Price 2018 pattern for a carbon-free
  base, and it keeps all four MOPS conditions on one shared base object.
- `MOPS_CASAMINO_WANG2018` (**new**) for the L-Trp screen: `MOPS_MINIMAL` plus 0.5 g/L
  casamino acids as an `intrinsically_undefined` complex ingredient. Casamino acids is
  an acid hydrolysate of casein with no structure to resolve, so it belongs in the
  medium beside tryptone and yeast extract rather than as a
  `SmallMoleculePerturbation`, which requires a typed `Compound`. Registered in
  `MEDIA_LIBRARY`, `CARBON_FREE_MEDIA` and `BACTERIAL_MEDIA_USES`.

**The mg/L to ug/mL restatement, said out loud.** The source prints the antibiotic doses
in mg/L. `ConcentrationUnit.mg_per_l` is a deliberately deferred enum member this branch
does not add, and 1 mg/L is exactly 1 ug/mL, so the doses are stored as 50 and 100
`ug/mL`. That is a unit identity, not a conversion, and it is recorded on the
`SourcedValue`s `KANAMYCIN_MG_PER_L` and `AMPICILLIN_MG_PER_L`.

`duration_hours` is 9.0 for the LB outgrowth (the essentiality screen and the initial
library) and a typed gap for the four re-seeded screens, which the paper doses in
doublings only; `duration_generations` is 15.0 and 5.0.

### Identifier reconciliation

The release names its own b-numbers: every library id is `<symbol><bNNNN>_<position>`
(`gspKb3332_817`) and Supplementary Data 2 spells each cluster member the same way, so
the b-number is a substring of a released identifier rather than a crosswalk result. No
ECK crosswalk is used and `identifier_mapping` is `None` on every perturbation.

`reconcile_locus_tags` over the 4,317 distinct cluster-member b-numbers, against
`ecoli_K12_MG1655_ASM584v2`:

| status | count |
|---|---|
| `current` | 4,310 |
| `renamed` | 0 |
| `non_gene_feature` (pseudogene locus, resolves to itself) | 3 |
| `retired` | 4 |
| `ambiguous` | 0 |

Resolver layer: 4,313 resolved as a locus tag directly, 4 not found. `remapped` 0,
`kept_on_collision` 0, `case_insensitive` 0, `outside_namespace` 0. Resolved fraction
4,313 / 4,317 = **0.9991**, above the loader's `MIN_RESOLVED_FRACTION` of 0.999.

The 4 retired tags are `b4590` (ybfK), `b4629` (ptwF), `b4635` (pauD) and `b4700`
(sokE). Two of them (`b4590`, `b4700`) carry designed guides and are the retired-target
drop below; the other two have no guide.

The 3 pseudogene loci are `b4614` (sokA), `b4643` (pawZ) and `b4645` (psaA). They are
NOT a drop rule: the L1 canonical-name rule passes a non-gene feature that resolves to
itself, and a repressed pseudogene is a real measurement. None of their clusters carries
a designed guide, so no pseudogene appears among the 4,218 measured genes; that is the
library's off-target and GC filters, not this loader's doing.

**253 of the source's gene symbols disagree with the assembly's primary symbol for the
same b-number** (the source's `acrS` is the annotation's `envR`, its `acuI` is `yhdH`).
`perturbed_gene_name` is the annotation's symbol where the annotation names one, so one
gene carries one spelling across datasets; the source's own spelling for every cluster
member is kept in `preprocess/guide_library.csv`.

### Retention ledger

Per screen, with the arithmetic the build asserts
(`rows - non_targeting - bad + both - retired = records`):

| screen | rows | non-targeting | Bad | both | retired | records | perturbations |
|---|---|---|---|---|---|---|---|
| `essentiality` | 54,116 | 398 | 1,468 | 6 | 11 | 52,245 | 52,844 |
| `auxotrophy` | 48,308 | 386 | 1,706 | 2 | 11 | 46,207 | 46,495 |
| `trp_biosynthesis` | 48,308 | 386 | 1,706 | 2 | 11 | 46,207 | 46,495 |
| `furfural_tolerance` | 48,308 | 386 | 0 | 0 | 11 | 47,911 | 48,230 |
| `isobutanol_tolerance` | 48,308 | 386 | 0 | 0 | 11 | 47,911 | 48,230 |
| **total** | **247,348** | **1,942** | **4,880** | **10** | **55** | **240,481** | **242,294** |

247,348 - 1,942 - 4,880 + 10 - 55 = 240,481.

The three rules:

1. **non-targeting**, 1,942 rows. The `sgRNA` cell is `NC_<n>` and the `gene` cell is
   the source's `0` sentinel. A guide with no genomic target cannot be a
   `BacterialCrisprInterferencePerturbation`, whose `systematic_gene_name` must be a
   locus tag, and the schema has no non-targeting-guide leaf. These rows are not
   discarded information: their median IS the zero of every stored fitness and their
   fitted sigma IS each screen's Z scale, both of which the build recomputes and records.
2. **`Quality == "Bad"`**, 4,880 rows, 10 of them also controls. The flag is the paper's
   own and it excludes these rows from every downstream statistic. No phenotype field can
   carry the flag, so storing them unmarked would present a log2 ratio whose denominator
   is below the authors' stated floor as equal in quality to the other 240,481. The two
   tolerance screens report no Bad row because their control is the initial library, to
   which the same filter had already been applied.
3. **retired target**, 55 rows (11 per screen: the singleton clusters `ybfKb4590` and
   `sokEb4700`). `GCA_000005845.2` carries neither tag, so the L1 canonical-name rule
   fails a retired stored name and the L4 gene universe does not contain it. For a
   multi-member cluster only the retired member's perturbation would be removed; here
   both clusters are singletons, so whole records go.

### Library shape, asserted at build time

56,071 guides in Supplementary Data 3 (55,671 gene-targeting + 400 non-targeting), every
spacer a 20-mer; 4,205 clusters in Supplementary Data 2 (4,090 protein-coding, 115
ncRNA-coding) over 4,317 member genes; 39 clusters hold more than one member and the
largest holds 10 (the `insH1` IS5 transposase copies, ahead of the 8-member `rrfH` 5S
rRNA cluster). 4,147 of the 4,205 clusters carry at least
one guide. 25 spacers appear twice in the library and none more than twice; measured,
every one of the 25 sits on two DIFFERENT clusters, over six distinct cluster pairs
(`esrE`/`ubiJ`, `hcaR`/`iroK`, `hokB`/`mokB`, `hokC`/`mokC`, `rzoQ`/`rzpQ`,
`sgrS`/`sgrT`: overlapping or nested genes). No pair falls inside one cluster, so no two
records collide on (gene, spacer).

### Build numbers

```
python -m torchcell.database.build_dataset_lmdb --dataset CrispriGuideFitnessWang2018Dataset
BUILT CrispriGuideFitnessWang2018Dataset: 240481 records
  at $DATA_ROOT/data/torchcell/crispri_guide_fitness_wang2018 in 72s
  gene_set size 4218; references 5
```

`python -m torchcell.provenance.build_manifest` reads
`crispri_guide_fitness_wang2018` as **`fresh`** with no drift.

Raw mirror: `$DATA_ROOT/torchcell-raw/wangPooledCRISPRInterference2018/si/si_data/`,
seven files, each with a `pmc_cloud` `RetrievalRecord` naming
`torchcell.literature.retrieve.pmc_cloud_object` and its key in the PMC Article Datasets
bucket under `PMC6018678.1/`. The recorded retrieval was re-run on 2026-10-07 and
reproduced all seven pins byte for byte.

| Supplementary Data | file | sha256 | what the loader reads |
|---|---|---|---|
| 2 | `41467_2018_4899_MOESM5_ESM.xlsx` | `8bb51873557bed15067f947053f84a9428883bc3acf771a6edf67a0a8740cdeb` | the gene clusters |
| 3 | `41467_2018_4899_MOESM6_ESM.xlsx` | `78a05c25da94df158065c18a316d0d29869587e4c839ad3365c911503900ef62` | guide id -> 20-mer spacer |
| 6 | `41467_2018_4899_MOESM9_ESM.xlsx` | `6500e9105fc6c7482794a558a6c69325196067dc89e295b8eda056ff7e284d48` | essentiality fitness |
| 7 | `41467_2018_4899_MOESM10_ESM.xlsx` | `3de9fd0aa6e14f543ae35703278558c10085ad63ee2e5f8be214dfc22b996a6e` | auxotrophy fitness |
| 8 | `41467_2018_4899_MOESM11_ESM.xlsx` | `4a322c70ac984e31cf7f931cec6a3f5af9bfcb3166d73e47f904306e1a4c2ae1` | L-Trp fitness |
| 9 | `41467_2018_4899_MOESM12_ESM.xlsx` | `b40c08b07d74b07f992929f866c6aa2f7cbff4e4059148597bf7d36e989df2a9` | furfural fitness |
| 10 | `41467_2018_4899_MOESM13_ESM.xlsx` | `29658ef0c45af0b2ac43c33d0858741938a2939ec34c150f77d01bfebc10ba74` | isobutanol fitness |

Not deposited, each for a stated reason: Supplementary Data 4 (guides per gene) is the
per-cluster count of Data 3, verified identical for all 4,205 clusters; Data 1, 11 to 14
feed no record; the Supplementary Information PDFs live in the torchcell-library mirror,
which is where Table 2 and the Methods are quoted from; BioProject PRJNA450392 holds raw
reads no loader consumes.

### What could not be sourced

- **The PubMed id.** It is in none of the mirrored bytes and not in the bibliography
  store's entry for this key, and the NCBI id converter answered HTTP 429 when asked.
  `Publication` therefore carries the DOI and `doi_url` only, with `pubmed_id=None`.
- **A per-guide uncertainty.** There is none to find, for the reason above (the
  replicates are combined before the ratio). Typed as three `ProvenanceGap`s rather than
  left silent.
- **Wall-clock duration of the four re-seeded screens.** The paper doses them in
  doublings; `duration_hours` is a typed gap and `duration_generations` carries 5.0.
- **The host MCm's chloramphenicol cassette as a locus.** The paper names the insertion
  site (`smf`) but gives the cassette no locus tag, and the library was designed against
  plain NC_000913.3. `AssemblyReferenceGenome.background` is `None` and the cassette is
  recorded on the `REFERENCE_BACKGROUND` sourced value, not asserted as a perturbation:
  it is identical in all 240,481 records.

### Finding: the gene level needs a significance field the schema does not have

**Supplementary Data 5 is not loaded**, and this is a schema gap rather than a choice.
It is the paper's GENE-level call: per cluster per screen, the median fitness of a
position-selected guide subset with a Mann-Whitney `Z score`, an `FDRvalue` and an
`FPRvalue` (4,142 + 4,004 x 4 = 20,158 rows). Those three statistics are what the table
exists for: the quasi-gene FPR interpolation behind them cannot be recomputed from the
stored guide rows, so they are genuinely additional information and not an aggregate.

`EnvironmentResponsePhenotype` has no significance field. Grepping the whole schema,
`GeneInteractionPhenotype.gene_interaction_p_value` is the only p-value anywhere in it.
Loading the gene rows today would keep a median and silently drop the hit call, which is
why this branch loads the guide level only. The deliverable for the gene level is a
field (a q-value or an FDR with its own typed `significance_type`, mirroring the
uncertainty ontology) plus the leaf and adapter work that rides it; adding one is a
served-class change and so a full-rebuild trigger, which is outside this branch.

### Verification

`verify_build` streams the LMDB twice and materializes nothing: once through
`verify_environment_response_dataset_streaming` with the MG1655 genome as the
canonical-name resolver and `set(genome.genbank.loci)` (4,651 tags, pseudogenes and RNA
included) as the L4 universe, and once through `SupplementaryRows`, which adds three
rows: every screen at its measured record count, every knockdown carrying its 20-mer
spacer under the dCas9 effector, and one assembly pin
(`ecoli_K12_MG1655_ASM584v2`, `GCA_000005845.2`, no background).

Run against the built store on 2026.10.07: **`crispri_guide_fitness_wang2018: PASS`,
19 of 19 rows, 0 failures**, 1,823 s wall (L0 revalidates all 240,481 records through
`TypeAdapter(ExperimentType)`, which is the whole cost; peak RSS 2.3 GB). The rows worth
quoting:

```
PASS L0 structural :: 240481 records validated
PASS L1 count :: observed 240481, expected 240481
PASS L1 pair_uniqueness :: 240481 unique (study, strain, condition) records, one each
PASS L1 canonical_gene_names :: 4218 systematic names, one canonical spelling each, each current in the genome
PASS L2 value_fidelity :: 240481 values checked
PASS L3 measurement_type_consistent :: single measurement_type: 'log2_ratio'
PASS L3 reference_zero :: numeric rule: reference response == 0 for all 240481 records
PASS L3 environment_perturbed :: all 240481 experiments carry an environmental edit
PASS L3 media_membership :: 240481 records on a shared MEDIA_LIBRARY medium, 0 deriving (3 distinct media)
PASS L4 gene_containment_sgd :: 1.000 of 4218 measured genes are reference genes
PASS L4 current_genome_genes :: every one of the 4218 measured systematic names is a gene of the current genome
PASS L1 every_screen_at_its_measured_record_count :: 5 screens, records {auxotrophy: 46207, essentiality: 52245, furfural_tolerance: 47911, isobutanol_tolerance: 47911, trp_biosynthesis: 46207}
PASS L1 every_knockdown_carries_its_20mer_spacer :: 242294 knockdowns, 0 without a 20-mer spacer; effectors {dCas9: 242294}
PASS L3 assembly_pin_is_mg1655_genbank_with_no_asserted_background :: 1 distinct assembly pin
```

Two rows are informational and read as expected rather than as a problem.
`L2 uncertainty_sanity` reports "0 labeled uncertainties, none a zero dispersion;
240481 records report n_samples >= 2 with no uncertainty", which is the typed absence
above: the release publishes no per-guide dispersion. And `L1 provenance_gaps` counts
1,109,991 documented gaps over all 240,481 records, which is the three uncertainty gaps
per record plus the environment's own duration and solvent gaps and the compound
layer's `inchikey` deferral. `L4 gene_containment_sgd` keeps the shared rule's name but
ran against the MG1655 locus universe this loader passes it (4,651 tags), not S288C.

### The adapter, which the step-9 set test makes part of the loader

`tests/torchcell/adapters/test_bacterial_adapters.py` (landed in `a80e404f`) asserts
that `dataset_adapter_map` holds exactly the registered bacterial dataset classes and
that `kg_bacteria.yaml` names exactly those, so registering this loader without an
adapter fails two tests rather than deferring step 9.
`CrispriGuideFitnessWang2018Adapter` is therefore here: the Menasalvas shape exactly
(`Shape(RESPONSE, crispr=True, bacterial=True)`), with a conf enable-list byte-identical
to the Menasalvas one below its header comment. The one behavioral difference is that
every Wang record states 37 C, so the temperature methods emit for every record rather
than for none. Three pins move with it: the two set tests from 20 to 21 adapters, and
`test_build_time_projection`'s `dataset_adapter_map` size from 71 to 72 with the new
class in its uncalibrated list. No knowledge-graph build was run.

### Hand-checked row

`gspKb3332_817` is the first data row of Supplementary Data 3 and of 6 to 10. Read off
the xlsx with openpyxl and asserted against the store by a data-gated test: spacer
`CTTTTCACCTGAGCAACCAG`, stored as `b3332` / `gspK`, with fitness -0.678534122869
(essentiality), -0.944735829468 (auxotrophy), -3.04918078714 (L-Trp), 2.61336100679
(furfural) and 5.15061288135 (isobutanol), each verbatim.

### Tests

`tests/torchcell/datasets/ecoli/test_wang2018.py`. The hermetic block builds a synthetic
release in `tmp_path` (6 clusters over 7 members, 9 guides, 2 screens) against the real
`EcoliK12MG1655Genome` over the synthetic MG1655 assembly of
`tests/torchcell/sequence/genome/_bacterial_fixtures.py`, with the network refused. It
exercises all three drop rules, the multi-member cluster, the sigma back-solve, both
environment families, the accounting arithmetic and the three verifier rows. The
data-gated block (`@pytest.mark.data`) re-audits every quote against the pinned OCR and
pins the real numbers above; it skips cleanly where the mirror or the store is absent.
