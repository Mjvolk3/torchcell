---
id: pzyze52l6m2bu31dbyvo4w6
title: 'Bloom 2019 segregant dataset'
desc: ''
updated: 1789189223634
created: 1789189223634
---

## Context

Bloom 2019 (eLife 8:e49212, 16 biparental crosses among 1011-collection strains, 13,950 segregants, 38 growth conditions) becomes the 50th dataset. It is the first ONTOLOGY extension pushed through the incremental admission gate rather than a new phenotype: a segregant carries no gene edit, so `Genotype.perturbations` (regex-validated `systematic_gene_name`, `schema.py:74-103`, `921-975`) cannot hold it, which is exactly the blocker [[torchcell.knowledge_graphs.dataset-admission-loop]] ("The 50th dataset") names.

[[torchcell.datamodels.eqtl-data-model]] (2026.08.26, 2026.08.31) settled the genotype design as a haplotype mosaic `{(chr, start, end, parent, p)}` against sha256-pinned parent assemblies, not `SequenceVariantPerturbation`, and left the attachment point open; this plan closes it with a sibling class (Key Design Decision 1). Deliverable: loader + additive schema + media constants + raw mirror with the first `torchcell-raw` `manifest.json` + L0-L4 verifier + adapter + `kg_manifest admit` ADMISSIBLE + notes. PR only, no merge, no KG increment `sbatch`.

Evidence in hand (2026-09-11 session; reuse, do not re-derive). Scratchpad `/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/c9adb0ba-07ff-4030-910b-ed8ce6d42537/scratchpad/`:

- `bloom2019_dropbox_data.zip`, 841,495,961 bytes, sha256 `78ded0dbef051878f322b26c9fa6bc8f1768a5fdaf7ef83362260d8dde7c0a5d`, from `https://www.dropbox.com/sh/jqm7a11zz9laytd/AABaE0EfQxLH6ounPhJ7yYWya?dl=1`. Members used: `phenotypes.tsv.gz` (sha256 `3942dbbc9280536f90cf4fc41ce39649cd2c638be623706a152b3a5406765575`; 13,950 rows x 40 trait columns; MEASURED zero NaN, 558,000 cells), 16 x `genotype_<cross>.tsv.gz` (`genotype_A` sha256 `21c4bed2445cff9e9d473dae348e59cc1f75b381075370b8d330b7eb299fbba4`, 951 x 49,707, values {1,2}), `cross_genotypes_README`. Extracted copies in `scratchpad/bloom2019_data/`.
- `elife-49212-fig1-data1-v2.xls`, sha256 `990e75168a77522b9b684b8d0151e45c24c75ccbe7c360d18c500008cfdad8eb`, from `https://cdn.elifesciences.org/articles/49212/elife-49212-fig1-data1-v2.xls`; sheets `Crosses and Strains` (16 x 7) and `Phenotypes` (compounds, media block, vendors). Reads with `pd.read_excel(..., engine="xlrd")` (xlrd 2.0.2, pandas 2.3.3; Scout A section 8).
- `joshsbloom/yeast-16-parents` at commit `c913c9ae7fd237329f639de02e7ec511b048730f`, clone in `scratchpad/yeast-16-parents/`.
- Parent assemblies already pinned at `$DATA_ROOT/torchcell-library/peterGenomeEvolution10112018/data/1011Assemblies.tar.gz` + `.member_index.tsv` (3.8 GB, never extracted; Scout C 18).
- Served manifest: `/scratch/projects/torchcell/database/kg_manifest.json` (36 served closures; `Genotype`/`Environment`/`Media`/`Phenotype` in all 36, `MeasurementType`/`AssayType`/`EnvironmentResponsePhenotype` in 0; Scout C 1-2).

Related issues: `#73` (`SACE_`-prefixed FASTA headers silently dropped variants; `273614` maps to `SACE_MAA`), `#92` (stored media lacking `Media.is_synthetic` fails L0), `#143` (SM media stub), `#319` (address genes by stable id, do not disturb `Genotype.sort_perturbations`).

Four corrections to the request brief, all MEASURED on the scratchpad copies: (1) the trailing digit of a trait name is the batch `PermutationGroup` (`phenotyping/code/rr_fx.R:102`, `paste(Condition, Concentration, PermutationGroup, sep=';')`), not a phenotyping round; (2) 36 of 38 served values are `residuals(lm(s.radius.mean ~ ctrl.s.radius.mean))` against the same-cross, same-batch control plate (`process_images.R:281-311`), only `YPD;;1` and `YNB;;1` are raw mean radii (tsv: `YPD;;1` mean 57.67, min 24.04, max 75.65; `YNB;;1` mean 40.91; every other column centered near 0, 36 column means in [-1.19, 0.16]); (3) `analysis/mapping.R` averages duplicates then MEAN-IMPUTES missing cells per trait within a cross (`x[is.na(x)]=mean(x, na.rm=T)`), and the release has no mask; (4) marker columns are grouped by chromosome in karyotype order but positions are NOT monotone within a chromosome (10 to 85 backsteps per chromosome on cross A).

## Relevant Files

| path | action | purpose | stance |
|---|---|---|---|
| `torchcell/datamodels/schema.py` | MODIFY | add `HaplotypeBlock`, `SegregantParent`, `SegregantGenotype`, `SegregantGrowthExperiment(+Reference)`, two `MeasurementType` members; union/map entries only (L2925-2990) | in-flux, disciplined |
| `torchcell/datamodels/media.py` | MODIFY | `YP` base + 12 YP-derived media + a solid `YNB` assay copy; register in `MEDIA_LIBRARY` (L421) | undocumented (note missing) |
| `torchcell/data/experiment_dataset.py` | REFERENCE | `_maybe_intern` L334, `_intern_record` L353 (genotype stays inline by design), `extract_systematic_gene_names` L467, `compute_gene_set` L475, `post_process` L167; NOT edited | stable |
| `torchcell/datasets/scerevisiae/bloom2019.py` | NEW | `Bloom2019Dataset` loader: per-cross int8 read, RLE mosaic, 38 environments, interning override, gene-set override | n/a |
| `torchcell/datasets/scerevisiae/__init__.py` | MODIFY | import + `bloom_datasets` list + `__all__` (do not copy the `CaudalPanTranscriptome2024Dataset` omission, Scout A 1) | undocumented |
| `torchcell/verification/segregant_growth.py` | NEW | streaming L0-L4 verifier for mosaic genotypes | n/a |
| `torchcell/verification/runners.py` | MODIFY | `SEGREGANT_GROWTH_DATASETS` registry + `run_segregant_growth`, hooked into `run_all` L1357 | stable |
| `torchcell/adapters/cell_adapter.py` | MODIFY | new `_segregant_genotype_node` + `_environment_response_phenotype_node` (+ reference method) + method-table entries; no served method touched | undocumented (note 2024.07.23) |
| `torchcell/adapters/bloom2019_adapter.py` | NEW | `Bloom2019Adapter`, 24-line clone of `nadal_ribelles2025_adapter.py` | n/a |
| `torchcell/adapters/conf/bloom2019_adapter.yaml` | NEW | enable list; no `perturbation (chunked)` | n/a |
| `torchcell/adapters/__init__.py` | MODIFY | re-export (L41-42 style) + `__all__` (L115) | undocumented |
| `torchcell/knowledge_graphs/dataset_adapter_map.py` | MODIFY | `Bloom2019Dataset: Bloom2019Adapter` (L132 style) | undocumented |
| `biocypher/config/torchcell_schema_config.yaml` | MODIFY | new `segregant genotype` (is_a genotype), `environment response phenotype` node classes; add `segregant genotype` to `genotype member of` source (L252) | in-flux |
| `torchcell/literature/retrieve.py` | MODIFY | new `zip_member(url, member)` retriever + `RETRIEVERS` entry (L82-86) | stable |
| `torchcell/literature/manifest.py` | REFERENCE | `ArtifactRecord` L123, `Manifest` L136, `RetrievalRecord` L87, `RetrievalMethod` L62 | stable |
| `torchcell/literature/provenance.py` | REFERENCE | `run_retriever` L55, `verify_artifact` L66 | stable |
| `experiments/database/scripts/build_supported_datasets_table.py` | MODIFY | one `CuratedRow` (L52-60 fields) under `"Environmental / chemogenomic"` | undocumented |
| `notes/torchcell.datasets.scerevisiae.bloom2019.md` | NEW | loader note: 38-condition table, provenance chain, four corrections, review flags | n/a |
| `notes/torchcell.knowledge_graphs.dataset-admission-loop.md` | MODIFY | dated section: first ontology extension through the gate | live |
| `tests/torchcell/datasets/scerevisiae/test_bloom2019.py` | NEW | RLE round trip on cross A, environment table, `_intern_record`, gene set | n/a |
| `tests/torchcell/datamodels/test_schema_invariants.py` | REFERENCE | map parity L57-68 polices the new experiment pair | stable |
| `tests/torchcell/datamodels/test_segregant_genotype.py` | NEW | sibling validates, base `Experiment` rejects, `TypeAdapter(ExperimentType)` resolves, round trip | n/a |
| `tests/torchcell/metabolism/test_media.py` | MODIFY | `YP_GALACTOSE` resolves galactose exchange; `YP` alone leaves the carbon source unresolved | stable |
| `tests/torchcell/adapters/test_bloom2019_adapter.py` | NEW | node id = sha256 of `genotype.model_dump()`, edge targets experiment hash (model: `test_nadal_ribelles2025_adapter.py` L54/L78) | n/a |
| `tests/torchcell/verification/test_segregant_growth.py` | NEW | synthetic 2-segregant, 3-condition fixture through every level | n/a |
| `torchcell/datasets/scerevisiae/hillenmeyer2008.py` | REFERENCE | condition-name rulebook (temperature on `Environment`, `pH<x>` as ph factor) | stable |
| `torchcell/datasets/scerevisiae/nadal_ribelles2025.py` | REFERENCE | raw-mirror symlink + sha256 `download()` L207-228; injected genome L174; `root` default L174 | stable |
| `torchcell/datasets/scerevisiae/caudal2024.py` | REFERENCE | assembly sha256 on a derived genotype L516-534; the 133-absences lesson | stable |

## Key Design Decisions

1. **`SegregantGenotype` is a sibling of `Genotype`, not a subclass, and `Genotype` is untouched (D1).** MEASURED: `SegregantGrowthExperiment(Experiment)` declaring `genotype: SegregantGenotype  # type: ignore[assignment]` validates and dumps every mosaic field, and base `Experiment` rejects a `SegregantGenotype` dict, so `TypeAdapter(ExperimentType)` resolves unambiguously. A `Genotype` subclass with `perturbations=[]` was rejected because every consumer reading only `perturbations` would silently see a wild-type S288C (the Caudal 133-absences failure); the sibling makes gene-keyed consumers fail loudly. Fingerprint-neutral as long as the new names appear only in the module-level union assignments and their own bodies (Gotcha 1).
2. **Class shape.** `HaplotypeBlock(ModelStrict)` {chromosome, start, end, parent (1|2), posterior in [0,1], n_markers}; `SegregantParent(ModelStrict)` {name verbatim (`BYa`, `RMx`, ...), peter_strain_id (None for BY), assembly_member (from `1011Assemblies.tar.gz.member_index.tsv`; S288C reference for BY), assembly_sha256 (the tar's pinned sha256), engineered_background verbatim from the xls `Parent 1`/`Parent 2` cells}; `SegregantGenotype(ModelStrict)` {cross, segregant_id, parent_1, parent_2, blocks, call_method, marker_matrix_sha256}. The pair `SegregantGrowthExperiment`/`SegregantGrowthExperimentReference` (`experiment_type="segregant_growth"`) joins `ExperimentType`, `ExperimentReferenceType`, `EXPERIMENT_TYPE_MAP`, `EXPERIMENT_REFERENCE_TYPE_MAP` only. No `_ENV_FACTORY` entry: it covers environment-perturbation leaves only (`test_ontology_all_trees.py:214-242`).
3. **Block coordinates are the first and last marker positions of the run, S288C coordinates as the marker names carry them; inter-block gaps are unassigned.** The crossover lies somewhere inside the gap and the release does not resolve it (MEASURED cross A: gap median 866 bp, p90 3.7 kb, max 29.8 kb, n=1,366). Filling gaps would encode an inference as an observation. MEASURED cross A blocks per segregant: min 55, median 83, mean 85.6, max 278, so |G| ~ 10^2 as the eQTL note predicted.
4. **`posterior=1.0` with a sourced `call_method`.** `analysis/mapping.R` does `g=pull.argmaxgeno(cross)` (R/qtl argmax of the HMM), removes fixed loci, writes the matrix. Record `call_method="R/qtl argmax.geno (hard call), release genotype_<cross>.tsv.gz"` with that quote anchored to the deposited `code/mapping.R`.
5. **`ReferenceGenome` stays `S288C`, haploid; the two parents live on the genotype.** S288C is the coordinate system of the markers and of the gene set; the parents are per-cross ingestion dependencies (eQTL note), pinned by the tar sha256 + member path, never extracted.
6. **Temperature on `Environment.temperature`; pH as `EnvironmentPhysicalPerturbation(factor=ph)` (D2).** `PhysicalFactor` docstring (`schema.py:1057-1062`) makes temperature deliberately absent from the factor enum; the brief's "15/37 C as perturbations" is wrong. YPD at 15/30/37 C are three `Environment`s.
7. **Carbon sources, glycerol, and both ethanol conditions are MEDIA, not `carbon_source` perturbations (D2).** `media_to_bounds` (`torchcell/metabolism/media.py:647`) resolves a typed `MediaComponent` compound to an exchange reaction; a `carbon_source` perturbation has no exchange reaction, so FBA would never see galactose. This contradicts hillenmeyer's `YP glycerol` rule; migrating hillenmeyer is out of scope. New constants: `YP` (yeast extract + peptone, no sugar, `state="solid"`, `is_synthetic=False`), `YP_{FRUCTOSE,GALACTOSE,MALTOSE,MANNOSE,RAFFINOSE,SUCROSE,TREHALOSE,XYLOSE,LACTATE}` at 2%, `YP_GLYCEROL` 3%, `YP_ETHANOL` 2%, `YPD_ETHANOL` (2% glucose + 2% ethanol); compounds via `resolved_compound` (`compound_identity.py:196`); concentrations quoted from the xls `Phenotypes` sheet. Lactate pH 6 = `YP_LACTATE` + `EnvironmentPhysicalPerturbation(factor=PhysicalFactor.ph, magnitude=Concentration(value=6.0, unit=ConcentrationUnit.ph))`, the exact form `hillenmeyer2008.py:273-279` uses. Stress compounds are `SmallMoleculePerturbation`s on the shipped `YPD` (`media.py:240`, solid, `is_synthetic=False`, so `#92` does not bite). `media.py::YNB` (L378) is liquid and vitamin-only, so the assay medium is a new solid copy (U2).
8. **`EnvironmentResponsePhenotype`, `assay_type=colony_size_array`, two new `MeasurementType` members (D3).** `control_regression_residual` (36 conditions) and `colony_size` (2 conditions, absolute mean radius in image pixels). `FitnessPhenotype` is a strictly positive ratio and the released values are signed (min down to -44). `differential_fitness` is bound by its docstring to Costanzo 2021 subtraction. Adding members moves only `MeasurementType`, the one enum with no served closure (Gotcha 2). One dataset, not a Hillenmeyer-style split: the two absolute columns are the regressors of the other 36.
9. **Reference is 0 for every condition, with gaps where it is not observed (D3).** Residual reference 0.0 is exact by construction. For `YPD;;1`/`YNB;;1` the parents' radii are not released; the reference is the scale's physical zero (no colony) with `ProvenanceGap(field="environment_response", reason=not_reported_by_primary)` and a review flag (U3). Never assume 1. The reference environment of a residual condition is its matched control environment; the `units` string names the matched control column (`YPD;;2` for `Caffeine;15mM;2`, since the batch digit selects the regressor).
10. **Citation key `bloomRareVariantsContribute2019`**, the Better BibTeX pattern the mirror uses (`peterGenomeEvolution10112018` skips "across"). MEASURED: the local `generate_citation_key` yields `bloomRareVariantsContributeDisproportionatelyQuantitative2019`, so it does not reproduce BBT; do not use it. `_bib/library.bib` has no Bloom 2019 entry, so `capture_by_doi` is unavailable and no library key is created. Paper text is retrieved by `direct_url` into `torchcell-raw/<key>/paper/`, NOT `torchcell-library/` (a library key with no Zotero item would be an orphan for the nightly lit-sync). Nothing goes to Zotero.
11. **38 served columns, not 40.** `YPD;;2` and `YPD;;3` are "two additional YPD replicates" dropped from every analysis (`analysis/mapping.R:139-141`, `mapping_fx.R:574`), but they remain the regressors for batch-2 and batch-3 conditions; they are read, used for the reference description, and not served as conditions. 4NQO was removed upstream (`mapping.R:87-96`) and is absent from the release.
12. **Gene set = S288C genes whose span overlaps at least one block of any cross**, computed once per dataset from the injected `SCerevisiaeGenome`, by overriding `compute_gene_set`; `extract_systematic_gene_names` is overridden to raise `NotImplementedError`. This sidesteps the empty-gene-set `ValueError` that would block `admit` (Gotcha 6). Meaning: "genes the panel's mosaics span", not "genes perturbed". The query-side callers (`neo4j_query_raw.py:454`, `base_cell.py:306`) are out of scope; mosaic genotypes cannot yet be consumed by the `Neo4jCellDataset` gene-set path.
13. **Node id is the sha256 of `genotype.model_dump()`, hashed once per record, never per block.** `_genotype_to_experiment_edge` (`cell_adapter.py:1563-1578`) reads no gene-keyed attribute and hashes the same way, so it is reused; `_genotype_node` (L512) emits `systematic_gene_names` and is NOT reused. The yaml class is `segregant genotype` with `is_a: genotype`: it IS a genotype, and existing `:Genotype` queries keyed on `systematic_gene_names` evaluate to null on it and filter it out rather than crash. No `haplotype block` nodes: 13,950 x 86 = 1.2M per-segregant-unique nodes with no sharing; blocks travel in `serialized_data`.
14. **Interning via a loader-level `_intern_record` override (D7).** `_intern_record` (L353) interns environment/reference/publication only, by design. Loaders call `self._intern_record` from their own `process()`, so overriding in the Bloom class needs no base edit and the "gate is blind to `experiment_dataset.py`" concern does not arise; `resolve_interned` is recursive, so reads need nothing new. MEASURED sizing: 85.6 blocks x ~70 bytes JSON = ~6 KB per segregant (> `INTERN_MIN_BYTES=512`, L91); 13,950 interned copies = ~84 MB versus ~3.2 GB inline at 38 copies each.
15. **Raw mirror holds exactly what the loader and verifier read (D6)**, each file an `ArtifactRecord` with an executable retriever. `data/`: the 18 zip members (`phenotypes.tsv.gz`, 16 `genotype_<cross>.tsv.gz`, `cross_genotypes_README`) by a new `zip_member` retriever (member name + `container_sha256` in `params`) plus the xls by `direct_url`, read in place with `engine="xlrd"` and never converted. `code/`: the four quoted R files by `direct_url` from `raw.githubusercontent.com` at the pinned commit. `paper/`: the eLife XML + PDF by `direct_url`. Not deposited: the zip itself, `joint_biallelic_*`, the VCF, the whole clone.

## Approach

Work in `$WT=/home/michaelvolk/Documents/projects/torchcell.worktrees/plan/bloom2019-segregant-dataset`; every command is prefixed `PYTHONPATH=$WT` and uses `~/miniconda3/envs/torchcell/bin/python`. Commit after each numbered step; keep every Edit importable.

1. **Schema additions.** Add the three genotype classes, the experiment pair, and the two `MeasurementType` members near their families in `schema.py`; append the pair to the four unions/maps (L2925-2990). The one snippet that matters, because mypy will otherwise reject narrowing the inherited annotation:

   ```python
   class SegregantGrowthExperiment(Experiment, ModelStrict):
       experiment_type: Literal["segregant_growth"] = "segregant_growth"
       genotype: SegregantGenotype  # type: ignore[assignment]
       environment: Environment
       phenotype: EnvironmentResponsePhenotype
   ```

   Run `PYTHONPATH=$WT ~/miniconda3/envs/torchcell/bin/python scripts/schema_impact_check.py --base HEAD`; expected: `[added]` for the five classes and `[stale]` for `MeasurementType` only, no `Genotype`/`Environment` entry. Commit with `PYTHONPATH="$WT" TORCHCELL_SCHEMA_ACK=1 git commit`; the first attempt fails on the ontology-figure hook by design (Gotcha 10), so re-run it with the three SVGs staged.
2. **Media constants** in `media.py` per Decision 7, using the existing `_sv`/`_c` helpers (L57/L74) with `source_uri` pointing at the raw-key xls and `citation_key="bloomRareVariantsContribute2019"`; add each to `MEDIA_LIBRARY`.
3. **Raw-mirror deposit.** A function `deposit_raw_mirror()` in the loader module (invoked once by hand, idempotent by sha256) writes `$DATA_ROOT/torchcell-raw/bloomRareVariantsContribute2019/{data,code,paper}/` and `manifest.json` (`Manifest`, `doi="10.7554/eLife.49212"`, `si_data_sources` = Dropbox share, eLife CDN, GitHub repo).
   - Add `zip_member(url: str, member: str) -> bytes` to `retrieve.py` and register it as `"torchcell.literature.retrieve.zip_member"` in `RETRIEVERS` (L82-86); it downloads the container, asserts `container_sha256`, and returns the member bytes. Each zip member's `RetrievalRecord`: `method=direct_url`, `source_url` = the `?dl=1` URL, that retriever, `params={"url", "member", "container_sha256"}`.
   - R files: `params` carry the commit hash.
   - Paper: Hypothesis (untested): the CDN serves `https://cdn.elifesciences.org/articles/49212/elife-49212-v2.xml` and `elife-49212-v2.pdf`; verify the URLs resolve, then record their sha256 on first retrieval.
   - Hand the extracted scratchpad copies to the deposit only after `verify_artifact` matches the recorded sha256.
4. **Loader `Bloom2019Dataset`** (`@register_dataset`, `root="data/torchcell/bloom2019"` exactly, injected `SCerevisiaeGenome(overwrite=False)`). `raw_file_names` = the 19 data files; `download()` symlinks the mirror into `raw/` and verifies every sha256 against `manifest.json` (nadal L207-228 pattern, now manifest-driven). `process()` streams one cross at a time.
   - Read `pd.read_csv(..., sep="\t", compression="gzip", dtype="int8", index_col=0)`, parse marker names as `chr_pos_ref_alt_index` (Gotcha 7), sort columns by (chromosome, position) and assert the sort is a pure column permutation, then run-length encode per segregant per chromosome. The invariant to code and test: re-expanding the blocks at the cross's sorted marker positions reproduces the int8 row, and `n_markers` sums to the cross's marker count.
   - Environment table: a module-level `CONDITIONS: dict[str, ConditionSpec]` (pydantic) keyed by the 38 column strings, each carrying media constant, temperature, perturbation list, measurement type, control column, and the xls quote for the concentration; the loader raises on any column not in the table or in `{"id", "YPD;;2", "YPD;;3"}`.
   - Phenotype: `EnvironmentResponsePhenotype(environment_response=<cell>, n_samples=2, sample_unit=<Methods "in duplicate" quote>, environment_response_se=None + gap)`; `units` names the matched control column and states the mean-imputation limitation. Reference per Decision 9.
   - Override `_intern_record` to also `_maybe_intern(rec["experiment"], "genotype", hint=segregant_id)` before delegating to `super()`. Override `compute_gene_set` per Decision 12. Docstring carries the four corrections.
5. **Dev build**: `PYTHONPATH=$WT ~/miniconda3/envs/torchcell/bin/python -c "..."` constructing the dataset with `process_workers=4, io_workers=0` (gene-set fork deadlock, memory `perturbation-ontology-refactor-landed`), in the background with a log. Expected `len(dataset) == 530100` (13,950 x 38) and `preprocess/build_manifest.json` present. Rebuilds per Gotcha 13.
6. **Verifier** `segregant_growth.py` + `SEGREGANT_GROWTH_DATASETS` + `run_segregant_growth` in `runners.py`, hooked into `run_all`; NOT in `ENVIRONMENT_RESPONSE_DATASETS` (D4: `_l3_environment_perturbed` flags the 13,950 YPD 30 C records, `_genotype_signature` reads `perturbations`). Run: `PYTHONPATH=$WT ~/miniconda3/envs/torchcell/bin/python -m torchcell.verification.runners`. Levels:
   - L0: `TypeAdapter(ExperimentType)`.
   - L1: count 530,100; uniqueness on (segregant_id, condition); phenotype ids <-> genotype files bijective (MEASURED cross A 951/951; ids are bare strings, `375_G1_01` vs `A01_01`, cross membership from the genotype file, Gotcha 15); gap census.
   - L2: every stored value equals the parsed tsv cell; every mosaic re-expands to its marker row (100%); block invariants (sorted, non-overlapping, parents alternate, `n_markers` sums).
   - L3: partition exactly {36 residual, 2 absolute}; reference 0 everywhere; every environment one of the 38 and exactly two carry no environmental edit; xls-sourced values re-read with `engine="xlrd"` (Gotcha 14); text-anchored `SourcedValue`s audited with `audit_sourced_value(sv, raw_root)` (root-agnostic, `sourced.py:78-82`).
   - L4: the 15 Peter IDs present in the member index (missing = reported, never substituted); xls parent pairs == README pairs; per-cross counts == xls `Number of Segregants Analyzed` (MEASURED from id prefixes: 375:876, 376:709, 377:841, 381:944, 393:865, 2999:936, 3000:896, 3001:648, 3003:899, 3004:867, 3008:795, 3028:943, 3043:943, 3049:884, A:951, B:953; sum 13,950; cross A drops plate `A11` before export and 951 already reflects it); marker ref allele == S288C base on a sample (expect >99%, settles the coordinate system empirically, U7); gene set subset of the genome.
7. **Adapter + admission.** `_segregant_genotype_node` projecting `cross`, `parent_1`, `parent_2`, `segregant_id`, `n_blocks`, `serialized_data`; `_environment_response_phenotype_node` (+ reference) mirroring the fitness pair at `cell_adapter.py:763`; method-table entries (L91/L201); `bloom2019_adapter.py` + conf enabling experiment, segregant genotype, environment, media, temperature, environment perturbation, the phenotype pair, and edges (never `perturbation (chunked)`); yaml node classes + `genotype member of` source list; the four registration surfaces. Then `PYTHONPATH=$WT ~/miniconda3/envs/torchcell/bin/python -m torchcell.knowledge_graphs.kg_manifest --manifest /scratch/projects/torchcell/database/kg_manifest.json admit --dataset Bloom2019Dataset --data-root /scratch/projects/torchcell-scratch`; expect ADMISSIBLE, `novel_symbols` = the five classes, `shared_symbols` drift 0, `graph_schema_added` = the two node classes.
8. **Tests** per the Relevant Files table; the loader test builds cross A only from the scratchpad copies via a fixture root.
9. **Notes.** `dendron-cli note write --fname "torchcell.datasets.scerevisiae.bloom2019"` then the dated section with the condition table below, the provenance chain (Dropbox share -> zip_member -> sha256; xls; GitHub commit; eLife CDN), the four corrections, and review flags U1-U4. Dated section in the admission-loop note: first ontology extension through the gate, the fingerprint measurement, and the `MeasurementType` caveat. `CuratedRow(section="Environmental / chemogenomic", name="bloom2019", genotypes="13,950 haploid segregants, 16 crosses, haplotype mosaic", env="38 conditions", phenotype="colony size residual / absolute", data_subpath="data/torchcell/bloom2019")`; regenerate the table if `build_supported_datasets_table.py` finishes in bounded time, else the row alone and say so in the PR (U9).
10. **PR**, no merge. The PR states: the base `_intern_record` was deliberately left alone; if the user later adds Bloom 2019 to the group library the XML anchors stay valid; hillenmeyer's `YP glycerol` migration is a follow-up.

Out of scope (D8): the KG increment `sbatch`; Albert 2018 and other BY x RM neighbors; the intergenic Regulatory DNA class; `haplotype block` nodes and locus queries; a structured `Genotype` for the parents' engineered background (verbatim string now); hillenmeyer media migration; the query-side gene-set path; Zotero; migrating other raw keys to `manifest.json`; `plot_supported_datasets_signal.py`.

### The 38 conditions

Column strings are the `phenotypes.tsv.gz` header verbatim (MEASURED). Base for compound conditions is YPD by the Methods sentence, not per row in the xls. `30*` = documented default with `ProvenanceGap(field="temperature", reason=deferred_pending_source_review)` unless the held XML states it (U1). `SMP` = `SmallMoleculePerturbation`; `ph(n)` = `EnvironmentPhysicalPerturbation(factor=PhysicalFactor.ph, magnitude=Concentration(value=n, unit=ConcentrationUnit.ph))`. Concentrations marked `xls` are absent from the column header and come only from the `Phenotypes` sheet.

| column | medium | temp | perturbation | type | control |
|---|---|---|---|---|---|
| `6-azauracil;50ug/mL;3` | YPD | 30* | SMP 6-azauracil 50 ug/mL | residual | `YPD;;3` |
| `Cadmium_Chloride;75uM;2` | YPD | 30* | SMP CdCl2 75 uM | residual | `YPD;;2` |
| `Caffeine;15mM;2` | YPD | 30* | SMP caffeine 15 mM | residual | `YPD;;2` |
| `Cobalt_Chloride;2mM;2` | YPD | 30* | SMP CoCl2 2 mM | residual | `YPD;;2` |
| `Congo_Red;75ug/mL;3` | YPD | 30* | SMP Congo red 75 ug/mL | residual | `YPD;;3` |
| `Copper_Sulfate;;1` | YPD | 30* | SMP CuSO4 6 mM (xls) | residual | `YPD;;1` |
| `Diamide;1.5mM;3` | YPD | 30* | SMP diamide 1.5 mM | residual | `YPD;;3` |
| `EGTA;5mM;3` | YPD | 30* | SMP EGTA 5 mM | residual | `YPD;;3` |
| `EtOH;;1` | `YP_ETHANOL` | 30* | none | residual | `YPD;;1` |
| `EtOH_Glucose;;1` | `YPD_ETHANOL` | 30* | none | residual | `YPD;;1` |
| `Fluconazole;100uM;2` | YPD | 30* | SMP fluconazole 100 uM | residual | `YPD;;2` |
| `Formamide;2%;2` | YPD | 30* | SMP formamide 2% | residual | `YPD;;2` |
| `Fructose;;1` | `YP_FRUCTOSE` | 30* | none | residual | `YPD;;1` |
| `Galactose;;1` | `YP_GALACTOSE` | 30* | none | residual | `YPD;;1` |
| `Glycerol;;1` | `YP_GLYCEROL` | 30* | none | residual | `YPD;;1` |
| `Lactate;;1` | `YP_LACTATE` | 30* | ph(6) | residual | `YPD;;1` |
| `Lithium_Chloride;100mM;2` | YPD | 30* | SMP LiCl 100 mM | residual | `YPD;;2` |
| `Magnesium_Chloride;;1` | YPD | 30* | SMP MgCl2 100 mM (xls) | residual | `YPD;;1` |
| `Maltose;;1` | `YP_MALTOSE` | 30* | none | residual | `YPD;;1` |
| `Manganese_Sulfate;;1` | YPD | 30* | SMP MnSO4 10 mM (xls) | residual | `YPD;;1` |
| `Mannose;;1` | `YP_MANNOSE` | 30* | none | residual | `YPD;;1` |
| `Methotrexate;0.4mM;2` | YPD | 30* | SMP methotrexate 0.4 mM | residual | `YPD;;2` |
| `Neomycin;5mg/mL;2` | YPD | 30* | SMP neomycin 5 mg/mL | residual | `YPD;;2` |
| `Paraquat;0.75mM;3` | YPD | 30* | SMP paraquat 0.75 mM | residual | `YPD;;3` |
| `Raffinose;;1` | `YP_RAFFINOSE` | 30* | none | residual | `YPD;;1` |
| `SDS;0.075%;3` | YPD | 30* | SMP SDS 0.075% | residual | `YPD;;3` |
| `Sorbitol;1.5M;2` | YPD | 30* | SMP sorbitol 1.5 M (header only; absent from xls) | residual | `YPD;;2` |
| `Sucrose;;1` | `YP_SUCROSE` | 30* | none | residual | `YPD;;1` |
| `Trehalose;;1` | `YP_TREHALOSE` | 30* | none | residual | `YPD;;1` |
| `Tunicamycin;3uM;3` | YPD | 30* | SMP tunicamycin 3 uM | residual | `YPD;;3` |
| `Xylose;;1` | `YP_XYLOSE` | 30* | none | residual | `YPD;;1` |
| `YNB;;1` | YNB solid | 30* | none | colony_size | itself (absolute) |
| `YNB;ph3;1` | YNB solid | 30* | ph(3) | residual | `YNB;;1` |
| `YNB;ph8;1` | YNB solid | 30* | ph(8) | residual | `YNB;;1` |
| `YPD;;1` | YPD | 30* | none | colony_size | itself (absolute) |
| `YPD;15;1` | YPD | 15 | none | residual | `YPD;;1` |
| `YPD;37;1` | YPD | 37 | none | residual | `YPD;;1` |
| `Zeocin;25ug/mL;3` | YPD | 30* | SMP zeocin 25 ug/mL | residual | `YPD;;3` |

Not served: `YPD;;2`, `YPD;;3` (regressors for batch 2 and 3). All plates `state="solid"`; the 48 h liquid YPD outgrowth is a pre-culture and is not modeled. `Environment.duration_hours=48` from "Plates were incubated for 48 hr", anchored to the XML.

## Gotchas

1. **The new class name must never appear inside an existing `ClassDef` in `schema.py`** (`schema_deps.py:232-235`, `287-292`): even a string forward-ref changes annotation text and moves `Genotype`, present in all 36 served closures. Union assignments are `ast.Assign`, invisible to the surface walk (`schema_deps.py:302-308`).
2. **`SampleUnit`/`UncertaintyType` are in 14 served closures**; adding a member is an instant BLOCK. Only `MeasurementType` (0 closures) may grow; re-run `admit` before landing in case a chemogenomic dataset is admitted first.
3. **`_perturbation_node` (`cell_adapter.py:530-557`) and `_perturbation_to_genotype_edges` (L1581-1605) read `systematic_gene_name` unconditionally**; editing them is drift on every served conf (`kg_manifest.py:560-603`, `660-667`). Add new methods; `_StripMethodTables` (`kg_manifest.py:307-318`) makes a new table entry non-drift.
4. **The undeclared-class guard covers only `... phenotype (chunked)` names (`kg_manifest.py:690-701`)**; a `segregant genotype (chunked)` method without a yaml class passes admission and BioCypher writes nothing. Declare both classes by hand and eyeball the emitted CSV headers.
5. **`admit` locates the LMDB by the class `root` default joined to `--data-root` (`kg_manifest.py:448-450`, `545`)** and requires a fresh `preprocess/build_manifest.json` (L544-557, blocker L684-689). Default must be exactly `"data/torchcell/bloom2019"`; the manifest path is `/scratch/projects/torchcell/database/kg_manifest.json` (`BUILD_ROOT`), not `torchcell-scratch`.
6. **An empty gene set kills the manifest**: `compute_gene_set_sequential` (`experiment_dataset.py:484`) -> `extract_systematic_gene_names` (L467) walks `genotype["perturbations"]`; setter L574-581 raises; `post_process` (L167) runs it before `write_build_manifest`. Decision 12 sidesteps it.
7. **RLE on file order produces phantom blocks** (correction 4: positions are not monotone within a chromosome); sort by (chromosome, position) first and assert the permutation. Marker names split on `_` into exactly five fields, but MEASURED on the `genotype_A` header the alt field carries commas at multiallelic sites (`chrI_693_T_A,G_1`) and ref/alt are multi-base at indels (`chrI_770_TA_T_5`), so the L4 ref-allele check compares the whole ref string to the S288C substring of that length at the position, not one base.
8. **pandas defaults hold the matrix as int64** (~5.6 GB for all 16 crosses vs ~700 MB int8); read per cross with `dtype="int8"`, RLE immediately, drop the frame.
9. **The schema-impact hook crashes from a worktree without `PYTHONPATH`** (`schema_deps.py:326-331` resolves the installed package, `schema_impact.py:339-346` returns the worktree, `relative_to()` fails); `run-schema-impact.sh:11` does not set it, `run-ontology-figure.sh:30-31` does.
10. **First `schema.py` commit fails by design** (ontology-figure hook exits 1 after `git add`ing three SVGs into `$ASSET_IMAGES_DIR/schema-ontology`, which `setup-worktree.sh:63` pointed at the worktree). Re-run; never `--no-verify`.
11. **`SCerevisiaeGenome(overwrite=True)` is the default and destroys `data.db`** (`s288c.py:471`). Always `overwrite=False`.
12. **`process_workers=0` has silently lost LMDB writes** (memory `abstract-biocypher-adapters`); build with `process_workers>0`, `io_workers=0`, and check `len(dataset)` afterward.
13. **The base class skips `process()` whenever `processed/` exists** (`experiment_dataset.py:262-275`); a stale build is reused silently. Move `processed/` + `preprocess/` aside with `/deprecate` before rebuilding.
14. **`SourcedValue` quotes anchored to the binary xls cannot be substring-audited**; anchor concentrations to the xls with a sheet/cell locator in the quote and audit structurally (Approach 6, L3). Text anchors (XML, R files, README) audit with `audit_sourced_value(sv, raw_root)`.
15. **Phenotype ids and genotype ids share a bare-string format only** (`375_G1_01` vs `A01_01`); cross membership comes from which `genotype_<cross>` file holds the id, never from parsing the id.
16. **`Copper_Sulfate;;1`, `Magnesium_Chloride;;1`, `Manganese_Sulfate;;1` carry no dose in the header**; the 6 mM / 100 mM / 10 mM values exist only in the xls media block. `Sorbitol;1.5M;2` is the reverse: header only, absent from the xls.
17. **The interned env is capped at `map_size=int(1e10)` (`experiment_dataset.py:329-331`)**; ~84 MB of mosaics fits, but do not also intern per-record phenotype dicts.
18. **Five identical worktrees at `513cbfa1` exist under `.claude/worktrees/`** and are not this branch; work only in `$WT`.

## Verification

- `PYTHONPATH=$WT ~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/datamodels -xq` (invariants: map parity L57-68, round trip L127, `_ENV_FACTORY` leaf equality).
- `PYTHONPATH=$WT ~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/datamodels/test_segregant_genotype.py tests/torchcell/datasets/scerevisiae/test_bloom2019.py tests/torchcell/metabolism/test_media.py tests/torchcell/adapters/test_bloom2019_adapter.py tests/torchcell/verification/test_segregant_growth.py tests/torchcell/knowledge_graphs/test_adapter_schema_consistency.py tests/torchcell/literature -xq`.
- `PYTHONPATH=$WT ~/miniconda3/envs/torchcell/bin/python scripts/schema_impact_check.py --base HEAD` before the schema commit: added symbols + `MeasurementType` stale, nothing else.
- Pre-commit (strict mypy + ruff on `torchcell/` and `tests/`, markdownlint on `notes/`) passes on every commit.
- `PYTHONPATH=$WT ~/miniconda3/envs/torchcell/bin/python -m torchcell.verification.runners` writes `$DATA_ROOT/data/torchcell/bloom2019/preprocess/verification_report.json` with every level passed and the per-cross block-count distribution reported (U8).
- Admit command from Approach 7 prints ADMISSIBLE.
- Manual smoke checks: `len(dataset) == 530100`; one record's `genotype.blocks` re-expanded at the cross's sorted marker positions equals its `genotype_<cross>.tsv.gz` row; `media_to_bounds(YP_GALACTOSE)` resolves the galactose exchange reaction and reports yeast extract/peptone as unresolved `intrinsically_undefined`; `resolve_interned` on a raw LMDB record returns a full `SegregantGenotype`; the interned env holds 13,950 genotype entries.

## Open Questions

Documented defaults with review flags; none blocks implementation.

- U1. Incubation temperature for the 36 non-temperature conditions: search the held XML first; else `Temperature(30)` + `ProvenanceGap(deferred_pending_source_review)` + note flag.
- U2. YNB assay medium recipe (carbon and nitrogen source): search the XML Methods; if absent, the `Media` carries YNB's components only with `open_gaps` naming the unstated carbon source, and FBA reports it unresolved rather than adding a presumed glucose.
- U3. Reference for the two absolute conditions: physical zero + gap + flag; the user may prefer another convention.
- U4. Mean-imputed cells are unidentifiable; dataset-level limitation in docstring, note, and `units`, not a per-record `ProvenanceGap`, since the affected cells are unknown.
- U5. SO term for `HaplotypeBlock`: candidate `SO:0001024` (haplotype); verify id and name against the SO release and record the version, or omit the field.
- U6. Parent-name suffixes `a`/`x` (`BYa`, `RMx`): stored verbatim; interpreted as mating type only if the paper states it.
- U7. Marker coordinates assumed S288C until the L4 ref-allele check runs.
- U8. Hypothesis (untested): block counts on the other 15 crosses are ~86 like cross A, since they track crossovers plus chromosomes, not marker density; the verifier reports the distribution.
- U9. Whether `build_supported_datasets_table.py` completes in bounded time across 49 datasets; state the time in the PR or ship the row alone.
