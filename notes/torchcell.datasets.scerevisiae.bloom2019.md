---
id: vqch607r90adh8qjvb4bg1a
title: Bloom2019
desc: ''
updated: 1789203204495
created: 1789203204495
---

## 2026.09.12 - Bloom 2019 segregant panels, the 50th dataset

Loader: `torchcell/datasets/scerevisiae/bloom2019.py` (`Bloom2019Dataset`). Source: Bloom JS, Boocock J, Treusch S, Sadhu MJ, Day L, Oates-Barker H, Kruglyak L. eLife 2019;8:e49212, doi:10.7554/eLife.49212, citation key `bloomRareVariantsContribute2019`. Plan: [[plan.bloom2019-segregant-dataset.2026.09.11]]. Design the genotype rests on: [[torchcell.datamodels.eqtl-data-model]].

### What it is

16 biparental crosses among 1011-collection strains (Peter 2018), 13,950 haploid segregants, each genotyped at 38,829 to 95,397 markers per cross and grown on 38 agar-plate conditions in duplicate, read out as end-point colony size. One record per (segregant, condition): 530,100 records. The first ontology extension pushed through the incremental admission gate: a segregant carries no gene edit, so its genotype is a `SegregantGenotype`, a sibling of `Genotype` (never a subclass) holding a haplotype mosaic `{(chromosome, start, end, parent, posterior)}` against two sha256-pinned parent assemblies.

### Representation

- Genotype: `SegregantGenotype{cross, segregant_id, parent_1, parent_2, blocks, call_method, marker_matrix_sha256}`. Each `HaplotypeBlock` is a run of consecutive markers called to the same parent on one chromosome; `start`/`end` are the first and last marker positions of the run (S288C R64 coordinates, see the L4 check below) and the crossover position inside the gap between blocks is deliberately unassigned. `posterior=1.0` everywhere: the release is the R/qtl argmax hard call (`analysis/mapping.R`, `g=pull.argmaxgeno(cross)`). Each `SegregantParent` carries the README label verbatim (`BYa`, `RMx`, `273614xa`), the Peter 2018 strain id, the assembly member inside the pinned `1011Assemblies.tar.gz` (sha256 `53540d09...`), and the parent's engineered background verbatim from the xls (`RM MatAlpha AMN1-BY ho::HphMX flo8::NatMX (YLK1950)`). BY is the S288C reference. The xls writes strain 273614 as `SACE_MAA`; the assembly member index carries it as `MAA` (the `SACE_` prefix that dropped an isolate's variants once, issue `#73`), so the mapping is explicit and a parent absent from the index raises.
- Environment: temperature on `Environment.temperature` (15, 30, 37 C); pH 3 and 8 as `EnvironmentPhysicalPerturbation(factor=ph)`; carbon sources, glycerol and both ethanol conditions as media with typed components (`torchcell/datamodels/media.py`: `YP`, `YP_FRUCTOSE` ... `YP_XYLOSE`, `YP_GLYCEROL`, `YP_ETHANOL`, `YPD_ETHANOL`, `YNB_GLUCOSE_SOLID`), so `media_to_bounds` resolves the sugar to an exchange reaction; the 20 stress compounds as `SmallMoleculePerturbation`s on the shipped `YPD`. All plates `state="solid"`; `duration_hours=48`.
- Phenotype: `EnvironmentResponsePhenotype`, `assay_type=colony_size_array`, `n_samples=2` (`sample_unit=technical_replicate`, the duplicate plates), `environment_response_se=None` with a `ProvenanceGap` (not released). Two new `MeasurementType` members: `control_regression_residual` (36 conditions) and `colony_size` (the two control plates, absolute mean radius in image pixels). Reference is 0 for every condition, exact by construction for a residual; for the two absolute conditions 0 is the physical zero (no colony) with a gap, since the parents' radii are not released.
- Interning: the loader overrides `_intern_record` to intern the genotype (one mosaic per segregant, not one per condition); the base class was deliberately left alone.
- Gene set: the S288C genes whose span overlaps at least one haplotype block of any cross, computed from the marker tables, not by walking records (`extract_systematic_gene_names` raises).

### The 38 conditions

Column strings are the `phenotypes.tsv.gz` header verbatim. `30*` is the documented representative temperature (see the review flag). `SMP` = `SmallMoleculePerturbation`; `ph(n)` = pH perturbation. The control column is the same-cross, same-batch plate the residual was regressed on (`process_images.R`).

| column | medium | temp | perturbation | type | control |
|---|---|---|---|---|---|
| `6-azauracil;50ug/mL;3` | YPD | 30* | SMP 6-azauracil 50 ug/mL | residual | `YPD;;3` |
| `Cadmium_Chloride;75uM;2` | YPD | 30* | SMP cadmium chloride 75 uM | residual | `YPD;;2` |
| `Caffeine;15mM;2` | YPD | 30* | SMP caffeine 15 mM | residual | `YPD;;2` |
| `Cobalt_Chloride;2mM;2` | YPD | 30* | SMP cobalt chloride 2 mM | residual | `YPD;;2` |
| `Congo_Red;75ug/mL;3` | YPD | 30* | SMP Congo red 75 ug/mL | residual | `YPD;;3` |
| `Copper_Sulfate;;1` | YPD | 30* | SMP copper sulfate 6 mM (xls only) | residual | `YPD;;1` |
| `Diamide;1.5mM;3` | YPD | 30* | SMP diamide 1.5 mM | residual | `YPD;;3` |
| `EGTA;5mM;3` | YPD | 30* | SMP EGTA 5 mM | residual | `YPD;;3` |
| `EtOH;;1` | YP + 2% ethanol | 30* | none | residual | `YPD;;1` |
| `EtOH_Glucose;;1` | YPD + 2% ethanol | 30* | none | residual | `YPD;;1` |
| `Fluconazole;100uM;2` | YPD | 30* | SMP fluconazole 100 uM | residual | `YPD;;2` |
| `Formamide;2%;2` | YPD | 30* | SMP formamide 2% v/v | residual | `YPD;;2` |
| `Fructose;;1` | YP + 2% fructose | 30* | none | residual | `YPD;;1` |
| `Galactose;;1` | YP + 2% galactose | 30* | none | residual | `YPD;;1` |
| `Glycerol;;1` | YP + 3% glycerol | 30* | none | residual | `YPD;;1` |
| `Lactate;;1` | YP + 2% lactate | 30* | ph(6) | residual | `YPD;;1` |
| `Lithium_Chloride;100mM;2` | YPD | 30* | SMP lithium chloride 100 mM | residual | `YPD;;2` |
| `Magnesium_Chloride;;1` | YPD | 30* | SMP magnesium chloride 100 mM (xls only) | residual | `YPD;;1` |
| `Maltose;;1` | YP + 2% maltose | 30* | none | residual | `YPD;;1` |
| `Manganese_Sulfate;;1` | YPD | 30* | SMP manganese sulfate 10 mM (xls only) | residual | `YPD;;1` |
| `Mannose;;1` | YP + 2% mannose | 30* | none | residual | `YPD;;1` |
| `Methotrexate;0.4mM;2` | YPD | 30* | SMP methotrexate 0.4 mM | residual | `YPD;;2` |
| `Neomycin;5mg/mL;2` | YPD | 30* | SMP neomycin 5 mg/mL | residual | `YPD;;2` |
| `Paraquat;0.75mM;3` | YPD | 30* | SMP paraquat 0.75 mM | residual | `YPD;;3` |
| `Raffinose;;1` | YP + 2% raffinose | 30* | none | residual | `YPD;;1` |
| `SDS;0.075%;3` | YPD | 30* | SMP sodium dodecyl sulfate 0.075% w/v | residual | `YPD;;3` |
| `Sorbitol;1.5M;2` | YPD | 30* | SMP sorbitol 1.5 M (header only; absent from the xls) | residual | `YPD;;2` |
| `Sucrose;;1` | YP + 2% sucrose | 30* | none | residual | `YPD;;1` |
| `Trehalose;;1` | YP + 2% trehalose | 30* | none | residual | `YPD;;1` |
| `Tunicamycin;3uM;3` | YPD | 30* | SMP tunicamycin 3 uM | residual | `YPD;;3` |
| `Xylose;;1` | YP + 2% xylose | 30* | none | residual | `YPD;;1` |
| `YNB;;1` | YNB + 2% glucose (solid) | 30* | none | colony_size | itself |
| `YNB;ph3;1` | YNB + 2% glucose (solid) | 30* | ph(3) | residual | `YNB;;1` |
| `YNB;ph8;1` | YNB + 2% glucose (solid) | 30* | ph(8) | residual | `YNB;;1` |
| `YPD;;1` | YPD | 30* | none | colony_size | itself |
| `YPD;15;1` | YPD | 15 | none | residual | `YPD;;1` |
| `YPD;37;1` | YPD | 37 | none | residual | `YPD;;1` |
| `Zeocin;25ug/mL;3` | YPD | 30* | SMP zeocin 25 ug/mL | residual | `YPD;;3` |

Not served: `YPD;;2` and `YPD;;3`, the batch-2 and batch-3 control plates. They are the regressors for the batch-2 and batch-3 conditions and the source drops them from every analysis (`mapping.R`: "remove the results from the two additional YPD replicates"). Their environment is the `YPD;;1` environment (`CONTROL_ENVIRONMENT_COLUMN`). 4NQO was removed upstream ("4NQO phenotyping failed for technical reasons, remove it") and is absent from the release.

### Four corrections to the working brief, all measured on the release

1. The trailing digit of a trait name is the plate batch (`rr_fx.R`: `paste(Condition, Concentration, PermutationGroup, sep=';')`), not a replicate or round index. `Caffeine;15mM;2` was residualized against `YPD;;2`.
2. 36 of the 38 served values are `residuals(lm(s.radius.mean ~ ctrl.s.radius.mean))` against the same-cross, same-batch, same-layout control plate (`process_images.R`); only `YPD;;1` and `YNB;;1` are raw mean radii (`YPD;;1` mean 57.67, min 24.04, max 75.65; `YNB;;1` mean 40.91; the other 36 column means lie in [-1.19, 0.16]).
3. `mapping.R` averages the duplicate plates and then mean-imputes missing cells per trait within a cross (`x[is.na(x)]=mean(x, na.rm=T)`; the source comment: "95% of traits and crosses have 0 missing the data, the rest typically have only one or two missing data points, four specific trait cross combinations have more than 10 missing phenotypes"). The release has zero NaN and no missingness mask, so imputed cells cannot be identified. Recorded in the `units` string as a dataset-level limitation, not as a per-record gap.
4. Marker columns are grouped by chromosome in karyotype order but positions are not monotone within a chromosome (10 to 85 backsteps per chromosome on cross A); the loader sorts by (chromosome, position) and asserts the sort is a pure column permutation before run-length encoding.

A fifth, found by the loader's own consistency check: the xls `Parent 1`/`Parent 2` columns do not always follow the README's order (cross 375: xls M22/BY, README BYa/M22). The README defines the 1/2 allele coding, so its order is authoritative; the xls is matched as an unordered pair.

### Provenance chain

Raw mirror `$DATA_ROOT/torchcell-raw/bloomRareVariantsContribute2019/` with `manifest.json` (`torchcell.literature.manifest.Manifest`, 25 files), the first raw-mirror key to carry one:

- `data/`: the 18 members of the authors' Dropbox share (`phenotypes.tsv.gz` sha256 `3942dbbc...`, 16 `genotype_<cross>.tsv.gz`, `cross_genotypes_README`), each retrieved by the new `torchcell.literature.retrieve.zip_member` retriever with the container's sha256 (`78ded0db...`, 841,495,961 bytes) pinned, since a Dropbox share re-zips on every request; and the eLife Figure 1 source data 1 xls (`990e7516...`) by `direct_url`.
- `code/`: the four R files quoted above from `joshsbloom/yeast-16-parents` at commit `c913c9ae`, by `direct_url` from `raw.githubusercontent.com`.
- `paper/`: the eLife JATS XML (`0cfa345e...`) and PDF (`3f96730f...`) by `direct_url` from the eLife CDN. The Methods quotes (`in duplicate`, `incubated for 48 hr`) anchor to the XML.
- Not deposited: the zip itself, the 16 `joint_biallelic_coding_chr*.tsv.gz`, the parent VCF, the rest of the GitHub clone. The parent assemblies stay under `peterGenomeEvolution10112018/data/` and are never extracted.

The paper is bib-only in Zotero and no library key was created; if it is later added to the group library the XML anchors stay valid as they are.

### Provenance flags for review

- Incubation temperature. The Methods state the 48 h incubation but not its temperature; 30 C is the documented representative on every plate except the 15 C and 37 C series, following the Nadal-Ribelles precedent. It is a constant across all crosses and conditions, so it cannot move any residual.
- Reference for the two absolute conditions. The parents' colony radii are not released; 0 is the scale's physical zero with a `ProvenanceGap` on the reference phenotype.
- YNB assay medium. The xls states `YNB | 2% glucose` for the YNB, pH 3 and pH 8 rows; the nitrogen source and the YNB trace-metal and salt rows are not stated and remain the shipped `YNB`'s open gaps.
- The `a`/`x` suffixes on parent labels (`BYa`, `RMx`) are stored verbatim and not interpreted.
- SO term for a haplotype block: not assigned; `HaplotypeBlock` carries no SO field until the id is verified against an SO release.
