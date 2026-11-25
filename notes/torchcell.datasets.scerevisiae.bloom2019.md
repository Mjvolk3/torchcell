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

### Build result (530,100 records)

Dev build at `$DATA_ROOT/data/torchcell/bloom2019` (final build at worktree commit `241627e6`, clean tree, 23.2 min sequential, `io_workers=0`; an earlier build at `d67bdbe2` was set aside after the quote fix below): 13,950 segregants x 38 conditions = **530,100** records; 4 references (one per control environment x measurement type); gene set 6,557 S288C genes spanned by the mosaics; `processed/lmdb` 1.9 GB, `processed/interned` 142 MB (13,950 interned mosaics plus the 38 environments and 4 references). `preprocess/block_counts.json` holds the per-segregant block counts.

Blocks per segregant, measured on the built store (n = 13,950): min 45, median 106, mean 153.8, max 1,847. Per cross:

| cross | n | min | median | mean | max |
|---|---|---|---|---|---|
| 375 | 876 | 64 | 107 | 136.8 | 1,095 |
| A | 951 | 55 | 83 | 85.6 | 278 |
| 376 | 709 | 52 | 114 | 199.8 | 1,847 |
| B | 953 | 47 | 90 | 97.2 | 631 |
| 377 | 841 | 63 | 107 | 217.4 | 1,525 |
| 393 | 865 | 52 | 100 | 145.5 | 1,604 |
| 381 | 944 | 45 | 94 | 117.2 | 1,231 |
| 3008 | 795 | 54 | 114 | 171.2 | 1,690 |
| 2999 | 936 | 72 | 124.5 | 141.0 | 986 |
| 3000 | 896 | 59 | 115 | 179.6 | 1,599 |
| 3001 | 648 | 69 | 143 | 242.5 | 1,719 |
| 3049 | 884 | 56 | 114 | 142.5 | 1,224 |
| 3003 | 899 | 58 | 101 | 149.1 | 1,304 |
| 3004 | 867 | 47 | 117 | 189.7 | 1,429 |
| 3043 | 943 | 52 | 100 | 112.4 | 1,070 |
| 3028 | 943 | 60 | 122 | 184.7 | 1,360 |

The plan's hypothesis that every cross would look like cross A (about 86 blocks, tracking crossovers plus chromosomes) is falsified: cross A is the lightest, the medians of the other 15 run from 90 to 143, and the means are pulled up by long tails of segregants with hundreds to 1,847 blocks. Hypothesis (untested): the tail is single-marker or few-marker runs from genotyping noise in the argmax hard calls, which a posterior threshold would collapse; the release carries no posteriors, so the blocks record the calls as released.

### Admission (`kg_manifest admit`, measured)

`python -m torchcell.knowledge_graphs.kg_manifest --manifest /scratch/projects/torchcell/database/kg_manifest.json admit --dataset Bloom2019Dataset --data-root /scratch/projects/torchcell-scratch` at worktree commit `241627e6` (re-run after the rebuild; identical at `d67bdbe2`): **ADMISSIBLE**. Closure 34 symbols, 10 novel (`AssayType`, `EnvironmentPhysicalPerturbation`, `EnvironmentResponsePhenotype`, `HaplotypeBlock`, `MeasurementType`, `PhysicalFactor`, `SegregantGenotype`, `SegregantGrowthExperiment`, `SegregantGrowthExperimentReference`, `SegregantParent`), 24 shared with served datasets; served datasets with schema drift 0; graph schema 0 changed, 2 added; adapter methods added 5, drift touching served datasets none. The served store was built at `513cbfa1`.

### Verification (`run_segregant_growth`, measured on the final build)

`preprocess/verification_report.json`, 22.4 min streaming: **L0-L4 PASS**, 17 levels. L0 530,100 records validate; L1 count 530,100, 530,100 unique (segregant, condition) pairs, 13,950 phenotype ids <-> 13,950 genotype rows, 530,100 documented gaps (the unreleased SE on every record, 0 deferred fields); L2 530,100 values finite and equal to the released tsv cells, 13,950 mosaics re-expand to their released marker rows, block invariants hold; L3 36 residual + 2 absolute conditions, reference 0 everywhere, 38/38 documented conditions with exactly `YNB;;1` and `YPD;;1` unedited, 53 sourced values audited against the raw mirror; L4 15 parents resolve in the assembly index, xls pairs and per-cross counts match, ref allele equals the S288C base at 0.9924 of 32,000 sampled markers (R64 coordinates, settled), 6,557 spanned genes all in SGD.

The first run failed one level: three xls-anchored media quotes had skipped the row's volume columns and the YP base quote paraphrased two rows; `Media.provenance` was not audited, which let the paraphrase through. Quotes are now verbatim row strings, the audit walks medium-level provenance, and the store was rebuilt so the stored media carry the corrected quotes.

## 2026.09.12 - Compound identity: the 18 name-only stress compounds resolved, store rebuilt

Serve-50 review finding: 18 of the 20 dosed stress compounds were name-only (only
fluconazole and sorbitol carried an InChIKey), which is 251,100 of the 530,100 records,
47.4%. Under the serve-50 rule that a compound with no structure identifier cannot be
encoded, those records would have been dropped rather than served.

### No loader change was needed, and that is a measured result

`build_conditions()` already routes every stress compound through
`torchcell.datamodels.compound_identity.resolved_compound` (the `_smp` helper), so the
18 name-only compounds were a gap in the CURATED TABLE, not in this loader. All 20 labels
are now in `torchcell/datamodels/compound_identity_inputs/bloom2019.txt` and resolve:

| status | n | compounds |
|---|---|---|
| `RESOLVED` (InChIKey + CID, most also ChEBI) | 19 | 6-azauracil, cadmium chloride, caffeine, cobalt chloride, Congo red, copper sulfate, diamide, EGTA, fluconazole, formamide, lithium chloride, magnesium chloride, manganese sulfate, methotrexate, neomycin, paraquat, sodium dodecyl sulfate, sorbitol, zeocin |
| `RESOLVED_MIXTURE` (ChEBI, no single-molecule InChIKey) | 1 | tunicamycin (`CHEBI:29699`, at least ten homologues) |

Tunicamycin keeps its typed `ProvenanceGap(field="inchikey")`, which is the honest encoding
for a substance with no single structure, and the L3 identity rule accepts ChEBI in that
case. `torchcell/datasets/scerevisiae/bloom2019.py` is UNCHANGED: the 38 xls-anchored
`dose_quote` strings stay verbatim (the L3 `sourced_values` audit re-reads them out of the
raw mirror), and the media constants the loader imports all still resolve after the library
retype (`SD_MINIMAL` is now `SD`, which this loader never imported; it imports `YPD`,
`YPD_ETHANOL`, `YNB_GLUCOSE_SOLID` and the eleven `YP_*` carbon-source variants, and all
14 media it builds are `MEDIA_LIBRARY` objects).

Two tests were added to `tests/torchcell/datasets/scerevisiae/test_bloom2019.py`:
`test_every_stress_compound_resolves_to_a_structure_identifier` (no compound is left with
neither an identifier nor a typed gap; tunicamycin is the only one without an InChIKey) and
`test_every_condition_medium_is_a_shared_library_object`.

### Rebuild

The stored `Compound` objects change when the table fills in, so the dev store was rebuilt
rather than patched. The previous build moved aside to
`data/torchcell/bloom2019/deprecated-compound-identity-2026-09-12/` (nothing deleted);
rebuild wrote 530,100 records in about 24 min, and `preprocess/block_counts.json` is
written as before. Verified on the new store: 20 distinct stress compounds, 20 with an identifier,
tunicamycin the only one without an InChIKey.

### Verification (`verify_segregant_growth_streaming`, measured on the rebuilt store)

Run from a scratch script with the same parameters `run_segregant_growth` passes
(raw mirror, assembly index, genome, SGD gene set, per-cross gene set); report written to
`preprocess/verification_report.json`. **L0-L4 PASS, 21 levels.** The two lines the review
was about:

```
  [ok] L3 compound_identity: environment edits: 279000 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 920700 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_membership: 530100 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (14 distinct media)
```

Before the rebuild the same store served 251,100 records whose only chemical identity was a
name. The rest of the report is unchanged from the 2026.09.12 run above: L0 530,100 records
validate, L1 count 530,100 with 530,100 unique (segregant, condition) pairs and a 13,950
<-> 13,950 id bijection, L2 values equal to the released tsv cells and 13,950 mosaics
re-expanding to their marker rows, L3 36 residual + 2 absolute conditions with reference 0
everywhere and 53 sourced values audited against the raw mirror, L4 15 parents pinned,
per-cross counts matching the xls, 0.9924 marker-reference agreement and 6,557 spanned genes
all in SGD.

The gap census moved from 530,100 to 544,050 documented gaps: the 530,100 unreleased
`environment_response_se` values plus 13,950 tunicamycin `inchikey` gaps (one per segregant
on the single tunicamycin condition). `media_membership` also now reports all 14 media as
shared library objects rather than derived ones, because the retyped library owns them
directly.

### Open flags

- **Records to drop: none.** The review's fallback of dropping 251,100 records is moot now
  that all 20 compounds resolve.
- **No `dropped_records.json`** is written for this dataset: nothing is dropped, so there is
  no rule to record.
- The `RESOLVED_MIXTURE` route means tunicamycin's `environment perturbation` node joins on
  ChEBI, not on an InChIKey. Any query keyed strictly on `inchikey` will miss that one
  condition (13,950 records).

## 2026.09.14 - Parent assemblies resolved from the genomes tier

Amends the 2026.09.12 statement that the parent assemblies "stay under
`peterGenomeEvolution10112018/data/`": they are still never extracted, but the
loader-facing copy is now the genomes tier set `peter2018_1011_assemblies`
([[torchcell.sequence.genome.registry]]), resolved and sha256-verified at `process()`.
`PETER_DIR_REL` is gone; before reading the member index, `process()` checks that the
tier's manifest pins `1011Assemblies.tar.gz` at `ASSEMBLY_TAR_SHA256`, so the stored
`assembly_sha256` literals and the dereferenceable bytes cannot drift apart. Stored records
are byte-identical (they hold `<tar>::<member>` and the tarball sha256, no path); the dev
store still reads 530,100 records, and the member index read through the tier equals the
library copy (1,010 entries). The library key's `data/` is retained untouched.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no; refused deposit leaving a directory: n/a (records source digest).

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against raw-mirror manifest, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.07 - Parent assemblies as ArtifactRefs

Phase 3 of the artifact tier ([[plan.artifact-tier.2026.10.07]]; schema side in [[torchcell.datamodels.schema]]). `SegregantParent.assembly_member` + `assembly_sha256` (the `1011Assemblies.tar.gz::<member>` string beside a literal sha256) became one `assembly_ref: ArtifactRef`. `ParentAssemblyRefs.from_tier()` reads the member index and both file refs from the genomes-tier manifests; `build_parent(label, genotype, assemblies)` gives a Peter parent the tarball narrowed to its member (`tc://genomes/peter2018_1011_assemblies/1011Assemblies.tar.gz#GENOMES_ASSEMBLED/AAA_6.re.fa` for RMx) and BY the SGD reference sequence file (`tc://genomes/sgd_S288C_R64-4-1_20230830/S288C_reference_sequence_R64-4-1_20230830.fsa`), replacing the free-text sentinel. `ASSEMBLY_TAR_SHA256` and the build-time literal-vs-tier comparison are gone: the manifest is the only pin, so they cannot drift. The verifier's `parents_pinned` result no longer reports that literal.

Measured by `experiments/036-dataset-fixes-before-kg-build/scripts/artifact_ref_loader_slice.py` (output `experiments/036-dataset-fixes-before-kg-build/results/artifact_ref_loader_slice.json`): the parents of crosses A (BYa, RMx) and 375 (BYa, M22) built through `ParentAssemblyRefs.from_tier` + `build_parent` carry tarball sha256 `53540d09...b4da` (3,999,623,201 bytes) or SGD sha256 `dbf065ff...f86b` (12,361,395 bytes), parse back from their tc:// strings, and resolve against the local genomes tier. Not rebuilt here (530,100 records); the dev rebuild of `bloom2019` runs under slurm. `SENTINEL_ASSEMBLY_SETS` in the genomes registry still maps the old BY sentinel, which the served store holds until the rebuild.

## 2026.11.24 - One record per released replicate: the YPD rounds 2 and 3 restored

Part of #1176 (per-replicate policy, #1178). Before: 530,100 records (13,950 segregants x 38 columns); `NOT_SERVED = {"YPD;;2", "YPD;;3"}` dropped two released columns. Now: 558,000 records (13,950 x 40 columns) over the same 38 conditions.

### What a replicate is here (sources, sha256 from the raw-mirror manifest)

- The YPD control plate was run in each of the three phenotyping rounds. `code/process_images.R` (`f8e90016...`) reads `RR_Round1`, `RR_Round2`, `RR_Round3` and sets `Key1$PermutationGroup=as.factor(1)` (L133), `as.factor(2)` (L148), `as.factor(3)` (L161); `code/rr_fx.R` (`4be671ea...`) L102 builds the trait name as `paste(key_sub$Condition, key_sub$Concentration, key_sub$PermutationGroup, sep=';')`. So the trailing digit of a trait name is the round. The 2026.09.12 section above calls it "not a replicate or round index"; that wording was wrong about the round.
- The authors call rounds 2 and 3 YPD replicates: `code/mapping.R` (`4785c533...`) L139 "# remove the results from the two additional YPD replicates (very similar to rep 1)", and `code/mapping_fx.R` (`899075024...`) L574 "# by default remove indices 37 and 38 corresponding to the two extra YPD replicate experiments". Both are reporting exclusions (QTL counts, variance-component plots); the values are released and `mapping.R` still maps them, so this is not a withdrawal and nothing is dropped.
- `process_images.R` L302-305 copies `s.radius.mean` unregressed for every `YPD;;`/`YNB;;` plate, so `YPD;;2`/`YPD;;3` are the same absolute colony radius (`MeasurementType.colony_size`) as `YPD;;1`.
- Released: `data/phenotypes.tsv.gz` (`3942dbbc...`), 13,950 x 40, 0 NaN in every column. Pearson r with `YPD;;1`: 0.916 (`YPD;;2`), 0.922 (`YPD;;3`); round means 57.67, 60.59, 60.30 pixels.

### Records

- The three YPD columns are three records of the one YPD condition (YPD, 30 C, 48 h, same reference) with `replicate_id` = the round digit the release prints (`"1"`, `"2"`, `"3"`; `YPD_REPLICATE_IDS`). Every other condition is measured in one round only and keeps `replicate_id=None`. `build_served_columns()` returns the 40 columns (the 38 conditions, then `YPD;;2`, `YPD;;3`); `build_conditions()` still returns the 38 conditions.
- Within a round each value is still the authors' mean over the duplicate liquid plates (eLife XML `0cfa345e...`, Methods: "Segregants were arrayed to 384-well liquid plates in duplicate with different plate positions across the duplicates."; `mapping.R` L91 `mPheno = sapply(subPheno[-1], function(x) sapply(x, mean, na.rm=T))`), so `n_samples=2`, `sample_unit=technical_replicate` and the `not_reported_by_primary` SE gap are unchanged. The per-plate input (`NORMpheno.RData`) is linked from the repository README at commit `c913c9ae` (`https://drive.google.com/file/d/0ByJ-NQ1MGtWIUkUwaDhpbTZSTWM`); on 2026-11-25 that link, its `/view` form, `drive.google.com/uc?export=download&id=...` and `drive.usercontent.google.com/download?id=...` all answered HTTP 404, and the companion folder link (`1nIV-DgJQ0qPnyqj1UWbN1FkpfBlnJH1k`) answered 404 too. The Dropbox share (37 members) holds no per-plate file. Nothing was substituted.
- Open owner question: the duplicate plates are two liquid wells grown separately for 48 h ("arrayed to 384-well liquid plates in duplicate ... grown in YPD for approximately 48 hr"), which under policy 1 reads as two cultures (biological), while the loader and the 2026.11.24 plan call them technical. The label only matters for `sample_unit` while the per-plate file is unobtainable.
- Also measured: 24.04 px is a floor value shared by the three rounds (6, 5, 5 cells), and 4 segregants carry identical values in all three rounds; neither is an imputation artifact the loader can detect.

### Build, verification, mean reproduction

Scratch build `$DATA_ROOT/tmp/scratch/per-replicate-bloom2019/bloom2019` (27.8 min, `io_workers=0`): 558,000 records; `replicate_id` None 516,150, `"1"` 13,950, `"2"` 13,950, `"3"` 13,950; 38 distinct environments, the same set as the old store.

Mean reproduction against the old dev store `$DATA_ROOT/data/torchcell/bloom2019` (530,100 records), grouping by (segregant, environment): 530,100 groups in both, same keys; 516,150 groups of one record and 13,950 YPD groups of three. The non-YPD record equals the old value exactly (max |diff| 0.0), and the YPD round-1 record equals the old YPD value exactly (max |diff| 0.0). The three-round mean is NOT the old value (mean |diff| 1.963 px, max 8.385 px, 13,949 of 13,950 groups differ): the old store held round 1 only, not a mean, so nothing old is reproduced by averaging; the difference is the round-to-round spread the old store hid.

Verifier on the scratch build (`preprocess/verification_report.json`, 43.5 min streaming on a shared node): **L0-L4 PASS**, 22 results. 558,000 records validate and are unique on (segregant, condition, replicate); every column holds 13,950 records; 13,950 ids <-> 13,950 genotype rows; 558,000 values equal the released tsv cells; 13,950 mosaics round trip; 36 residual + 4 absolute columns; reference 0 everywhere; 40/40 columns over 38 conditions; 55 sourced values audited against the raw mirror; marker reference 0.9924 of 32,000; 6,557 spanned genes all in SGD. New L1 rule `column_counts` (every one of the 40 columns holds one record per stored segregant); `pair_uniqueness` keys on (segregant, condition, replicate id); `measurement_partition` is 36 residual + 4 absolute columns; `conditions_documented` sees 40 columns over 38 conditions; the L3 audit carries the two YPD-replicate quotes (55 audits).

Tests: `tests/torchcell/datasets/scerevisiae/test_bloom2019.py` (served columns; real-data test pinning 13,950 x 40, 0 NaN, r 0.916 / 0.922, both quotes), `test_bloom2019_synthetic.py` (120 records, the three YPD records of one segregant), `tests/torchcell/verification/test_segregant_growth.py`, `test_runners.py` oracle 558,000.
