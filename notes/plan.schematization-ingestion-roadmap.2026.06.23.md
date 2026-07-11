---
id: xpywhp9bf8glrka52mejdgp
title: Schematization + Ingestion Roadmap (2026.06.23)
desc: ''
updated: 1783751757578
created: 1782265720149
---

# Schematization + Ingestion Roadmap

Each `## WS<n>` below is a self-contained workstream that becomes ONE GitHub issue
linking back to this note (`(issue: WS<n>)` placeholders swap to `#N` once cut).
Revised 2026.06.24 after user review: two-phase ordering (datasets+schema first,
graph second), three new datasets (Ozaydin beta-carotene, Cachera CRi-SPA
betaxanthin, Zelezniak metabolite/protein), pan-transcriptome promoted into scope,
`n_samples` confirmed canonical, Radiant clarified as the OpenStack DB host.
Amended 2026.07.05: added **WS9b** (Mülleder2016 amino-acid metabolome), a
genome-scale `MetabolitePhenotype` dataset that reuses the WS4 schema — sibling
to Zelezniak (WS9). Inserted as WS9b (not renumbered) to keep the Phase B
"WS11-14" group stable.

## Progress (updated 2026.07.07)

Legend: ✅ done · 🔨 in progress · ⬜ not started. Issues are cut as each WS is
started ("Schematization Roadmap — WS<n>" titles, distinct from the closed
CI-finish WS series).

| WS      | Title                                      | Status | Issue / PRs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|---------|--------------------------------------------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WS1     | Schema hardening + freeze                  | 🔨     | #21 + #24 **merged** (ontology + DRY validators + invariant tests); **remaining:** no-structure-without-data field audit                                                                                                                                                                                                                                                                                                                                                                                                                   |
| WS2     | n_samples/fitness_se (Kuzmin)              | 🔨→✅   | #22 **merged** (Costanzo SE fix); #23 (Kuzmin n=4 back-solve) merging on CI                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| WS3     | Provenance + L0–L4 framework               | 🔨→✅   | #25; framework built (`torchcell/verification/`); `StatDerivation` **merged (#26)**; expression verifier + report artifacts landed via WS5; **remaining:** wire the artifact emission into each dataset's own `post_process`                                                                                                                                                                                                                                                                                                               |
| WS4     | New phenotypes (metabolite/protein/visual) | 🔨→✅   | `VisualScorePhenotype` (Ozaydin) + `MetabolitePhenotype` (Cachera, keyed by metabolite id + measurement_type + optional Yeast9 id) **added**. Protein-abundance phenotype **added** (`ProteinAbundancePhenotype`, Zelezniak WS9) — all three phenotype families landed                                                                                                                                                                                                                                                                     |
| WS5     | Verify Sameith2015 + Kemmeren2014          | 🔨→✅   | Rebuilt DM(72)+SM(82)+Kemmeren(1450) with SE+n_replicates; L0–L4 all PASS; SM Sameith built for the first time. See [[torchcell.verification.expression]]. **✅ landed (main)**                                                                                                                                                                                                                                                                                                                                                             |
| WS6     | Verify Ohya2005 CalMorph                   | 🔨→✅   | Ohya (4718) already L0-clean vs current schema (no rebuild); L0–L4 morphology verifier PASSES. See [[torchcell.verification.morphology]]. **✅ landed (main)**                                                                                                                                                                                                                                                                                                                                                                              |
| WS7     | Ozaydin2013 beta-carotene                  | 🔨→✅   | `CarotenoidOzaydin2013Dataset` built (4474 records, VisualScorePhenotype); SI+PDF+OCR in library mirror w/ manifest; L0–L4 PASS. See [[torchcell.datasets.scerevisiae.ozaydin2013]]. **✅ landed (main)**                                                                                                                                                                                                                                                                                                                                   |
| WS8     | Cachera2023 betaxanthin                    | 🔨→✅   | `BetaxanthinCachera2023Dataset` built (4735 records, MetabolitePhenotype from CRI-SPA GitHub data); PDF+OCR+CSV in library mirror w/ manifest; L0–L4 PASS. See [[torchcell.datasets.scerevisiae.cachera2023]]. **✅ landed (main)**                                                                                                                                                                                                                                                                                                         |
| WS9     | Zelezniak2018 metabolite + protein         | ✅      | **protein ✅ landed** (PR #35); **metabolite ✅ landed** (`MetaboliteZelezniak2018Dataset`, PR #41 `7632333d`; 95 strains × 50 metabolites; first populated `target_metabolite_ids` → Yeast9 `s_NNNN`; L0–L4 PASS). WS9 complete                                                                                                                                                                                                                                                                                                             |
| WS9b    | Mülleder2016 amino-acid metabolome         | ✅      | **landed** (`AminoAcidMulleder2016Dataset`, PR #34; 4678 strains × 19 AAs; L0–L4 PASS)                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| WS10    | Pan-transcriptome (Caudal 2024)            | ✅      | **landed** (`CaudalPanTranscriptome2024Dataset`, PR #42 `a7408e7f`; 943 natural isolates = Caudal ∩ Peter2018 genomes). Natural isolate = perturbations off S288C: new `SequenceVariantPerturbation` (~5047/iso) + `CopyNumberVariantPerturbation` (accessory-present/core-absent) + `RNASeqExpressionPhenotype` (absolute TPM+count). Caudal+Peter OCR'd+mirrored; L0–L4 PASS. See [[torchcell.datasets.scerevisiae.caudal2024]]. **Remaining:** off-graph sequence STORE + apply-to-sequence dereferencer (perturbations carry pointers) |
| WS11–14 | Phase B (graph/adapters/build/deploy)      | ⬜      | — (cut when Phase A done)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

## Context

Rebuild the Neo4j knowledge graph (KG) from schematized, L0-L4-verified datasets and
deploy it to UIUC NCSA so it substantiates the Volk & Zhao abstract (Cell Graph
Transformer predicts trigenic interaction r=0.454, expression r=0.543, morphology
r=0.619, and recommends gene deletions for beta-carotene/betaxanthin production).
Those claims only stand if the data is in the KG and ingested correctly; several of
these datasets are not in the deployed DB today. Because this backs a conference
abstract, correctness and provenance outrank breadth.

**Build order (user directive): datasets + schema FIRST, graph SECOND.** Phase A
lands all pydantic schema work and dataset classes (records persisted to LMDB,
verified L0-L4) with NO BioCypher. Phase B then annotates to Biolink, builds
adapters, rebuilds the KG, and deploys. This keeps the sensitive schematization
work decoupled from graph infrastructure churn.

In-scope datasets: Sameith2015 + Kemmeren2014 (microarray expression), Ohya2005
CalMorph (morphology), Ozaydin2013 (beta-carotene visual screen), Cachera2023
CRi-SPA (betaxanthin), Zelezniak2018 (metabolite + protein abundance),
Mülleder2016 (genome-scale amino-acid metabolome across the deletion collection),
and a pan-transcriptome dataset (NGS expression + per-strain genomes). Success: that KG
running on the NCSA OpenStack host returning correct counts to smoke-test Cypher,
every exposed statistic backed by a stored pydantic field, every ingest gated by
L0-L4.

## Naming + Provenance Convention

- **Module**: short `lastnameYYYY.py` (existing pattern: `kuzmin2018.py`,
  `ohya2005.py`). New: `ozaydin2013.py`, `cachera2023.py`, `zelezniak2018.py`,
  `mulleder2016.py`.
- **Class**: `<Prefix><Author><Year>Dataset` (existing `SmfCostanzo2016Dataset`,
  `ScmdOhya2005Dataset`). When one paper yields multiple datasets, differentiate by
  phenotype/class (e.g. Zelezniak -> a metabolite dataset AND a protein dataset).
- **Provenance citation_key** (Zotero/library identity, recorded in the manifest):
  `ozaydinCarotenoidbasedPhenotypicScreen2013`,
  `cacheraCRISPAHighthroughputMethod2023`, `zelezniakMachineLearningPredicts2018`,
  `mullederFunctionalMetabolomicsDescribes2016`.

## Relevant Files

| Path                                                                                                                | Action    | Purpose                                               | Stance       |
|---------------------------------------------------------------------------------------------------------------------|-----------|-------------------------------------------------------|--------------|
| `torchcell/datamodels/schema.py`                                                                                    | MODIFY    | Pydantic Experiment/Reference/Phenotype models        | in-flux      |
| `torchcell/datamodels/pydant.py`                                                                                    | MODIFY    | Base model + shared validator                         | in-flux      |
| `torchcell/datamodels/calmorph_labels.py`                                                                           | REFERENCE | CalMorph label vocabulary used by validators          | undocumented |
| `torchcell/data/experiment_dataset.py`                                                                              | REFERENCE | Base dataset class (LMDB, post_process)               | provisional  |
| `torchcell/datasets/scerevisiae/{sameith2015,kemmeren2014,ohya2005}.py`                                             | REFERENCE | Existing datasets (verify)                            | stable       |
| `torchcell/datasets/scerevisiae/{kuzmin2018,kuzmin2020}.py`                                                         | MODIFY    | Add n_samples/fitness_se (Costanzo done)              | stable       |
| `torchcell/datasets/scerevisiae/{ozaydin2013,cachera2023,zelezniak2018,mulleder2016}.py`                            | NEW       | New dataset classes                                   | n/a          |
| `torchcell/metabolism/yeast_GEM.py`                                                                                 | REFERENCE | Yeast9 v9.0.2 SBML; metabolite node IDs for compat    | stable       |
| `torchcell/datasets/dataset_registry.py`                                                                            | MODIFY    | Registers datasets by class name                      | provisional  |
| `torchcell/adapters/cell_adapter.py`                                                                                | REFERENCE | Base adapter (Phase B)                                | provisional  |
| `torchcell/adapters/ohya2005_adapter.py`                                                                            | REFERENCE | CalMorph adapter (exists)                             | provisional  |
| `torchcell/adapters/{sameith2015,kemmeren2014,ozaydin2013,cachera2023,zelezniak2018,mulleder2016,pantx}_adapter.py` | NEW       | Phase-B adapters                                      | n/a          |
| `biocypher/config/torchcell_schema_config.yaml`                                                                     | MODIFY    | BioCypher node/edge schema -> Neo4j fields (Phase B)  | provisional  |
| `torchcell/knowledge_graphs/create_kg.py`                                                                           | REFERENCE | KG build entry point (+ conf/)                        | provisional  |
| `torchcell/database/dataset_adapter_map.py`                                                                         | MODIFY    | Datasets -> adapters for the build                    | provisional  |
| `database/scripts/{export_database.sh,import_database_openstack.sh}`                                                | REFERENCE | Export dump; import on OpenStack host                 | provisional  |
| `torchcell/literature/{manifest,calmorph,extract,scanned}.py`                                                       | REFERENCE | Extraction + provenance subsystem                     | provisional  |
| NEW genome store module + `tests/torchcell/{adapters,datamodels}/`                                                  | NEW       | Genome FASTA/GFF store; first adapter/invariant tests | n/a          |

## Cross-cutting Design Decisions

1. **Two-phase build (user directive).** Phase A = schema + datasets + record-level
   L0-L4 verification, NO BioCypher. Phase B = Biolink annotation + adapters + KG
   rebuild + deploy. Datasets must be correct as pydantic/LMDB records before any
   graph work; this is the sensitive part and it is isolated.

2. **`n_samples` is the canonical replicate-count field across ALL phenotypes.**
   Confirmed: `n_samples`, not `n_replicates`. SE is the ML-facing statistic
   (precision of the mean, used for training weights + dedup); it is stored when the
   source provides it, else derived `SE = SD/sqrt(n_samples)`. Rationale +
   flowchart: [[user.Mjvolk3.torchcell.tasks.weekly.2026.03.fitness-interaction-n_samples]]
   and [[user.Mjvolk3.torchcell.tasks.weekly.2026.05.phenotype-derivable-statistics]].
   This is the field-set freeze that gates the expression + new-dataset schema.

3. **No new structure without data -- but Zelezniak/Ozaydin/Cachera DO justify new
   phenotypes.** Metabolite + protein-abundance + a visual/ordinal-score phenotype
   are net-new (no existing class holds them) and are warranted because named
   datasets supply the measurements. We add exactly these and no speculative
   granularity (no environment small-molecule/drug encoding, no unused subclasses).

4. **Yeast9 compatibility for metabolites.** The metabolite phenotype keys
   measurements by IDs that align to `YeastGEM` metabolite node IDs
   (`torchcell/metabolism/yeast_GEM.py`, SBML v9.0.2), so measured metabolites link
   to the constraint-based metabolic network in the KG.

5. **Expression across technologies (microarray vs NGS).** Sameith/Kemmeren are
   microarray; the pan-transcriptome is NGS (RNA-seq). Keep platform-specific
   storage but provide a normalization function mapping both to a common
   `log2(sample/reference)` representation, so the model trains on one convention.
   For this round, NGS expression may still be stored on the Neo4j graph (genomes
   are NOT -- see decision 6).

6. **Genomes are pointers, never payloads.** Neo4j stores `GenomeAssembly` /
   `SequenceRecord` / `GenomeFeature` pointer nodes (`assembly_id`, `species`,
   `strain`, `fasta_uri`, `gff_uri`, `sha256`, `version`); FASTA/GFF live external,
   stored next to the reference S. cerevisiae genome. TorchCell builds annotated
   genomes + gene embeddings in-memory at dataload, referencing strains by name.

7. **BioCypher-exposure constraint (Phase B).** Only pydantic `model_fields` are
   Neo4j-queryable; `@property` is invisible. Every exposed statistic must be a
   stored field declared in `torchcell_schema_config.yaml` -> yields a
   no-property-exposure test and a schema<->YAML field-parity check.

8. **Deps + rigor.** Database-side deps (biocypher, cobra) live in the separate db
   env, NOT `env/requirements.txt` (ML env); new code is mypy-strict + unit-tested
   per CLAUDE.md.

## Phasing

- **Phase A (schema + datasets, no graph):** WS1-WS10 (incl. WS9b).
- **Phase B (graph: annotate, adapt, build, deploy):** WS11-WS14.

---

## WS1. Schema hardening + freeze (land on main)

**Status:** 🔨 in progress. SE/uncertainty ontology landed (#21, merged). DRY of the
6 `validate_label_fields` + invariant test suite in PR #24 (open). Remaining: the
no-structure-without-data field audit.

**Goal/Scope.** Land pure-pydantic structural invariants + the validator dedup on
main, and execute decision (2): make `n_samples` the canonical replicate field
across phenotypes, keep SE as the stored/derived ML statistic. No new structure
here (new phenotypes are WS4).

**Key files.** `torchcell/datamodels/schema.py`, `pydant.py`,
`tests/torchcell/datamodels/test_schema_invariants.py` (NEW).

**Dependencies.** None (root).

**Invariants.** Liskov substitutability across the `*Type` unions; serialization
round-trip `obj == Model(**obj.model_dump())` for every Experiment/Reference/
Phenotype; `*Type`-union and `*_TYPE_MAP` registry completeness; DRY the six
duplicated `validate_label_fields` (schema.py approx lines 342, 371, 410, 449, 491,
556) into one shared Phenotype-base validator; no-structure-without-data field
audit (every field maps to a populating dataset).

### Checks that must pass

- All invariant tests green (Liskov, round-trip, registry completeness, dedup).
- `n_samples` present + consistent on every phenotype that has replicate counts; SE
  storage/derivation rule documented.
- mypy-strict + ruff clean.

## WS2. Finish n_samples/fitness_se refactor (Kuzmin)

**Status:** ✅ done (open PRs, ready to merge). Costanzo SE corrected + regex cleanup
(#22); Kuzmin 2018/2020 mapped to the ontology with `n_samples=4` resolved by
p-value back-solve + a data-gated tripwire test (#23). All ruff + mypy-strict green.

**Goal/Scope.** Costanzo2016 already carries `n_samples` + `fitness_se`; Kuzmin2018
and Kuzmin2020 still use `fitness_std` only. Bring both Kuzmin fitness datasets to
the same `n_samples`/`fitness_se` convention so all fitness data is consistent.

**Key files.** `torchcell/datasets/scerevisiae/{kuzmin2018,kuzmin2020}.py`,
`torchcell/datamodels/schema.py` (FitnessPhenotype, REFERENCE).

**Dependencies.** WS1 (frozen field semantics).

### Checks that must pass

- Kuzmin2018/2020 emit `n_samples` + `fitness_se` (derived where only SD exists).
- Record counts unchanged vs current build; L0 instantiation passes.
- Cross-dataset consistency: fitness datasets share an identical phenotype field set.

## WS3. Provenance + L0-L4 verification framework

**Status:** 🔨 in progress — issue #25. The framework already exists
(`torchcell/verification/{report,levels,sourced}.py` + tests: `Provenance`,
`Level`, `LevelResult`, `VerificationReport`, and l0–l4 checks). `StatDerivation`
(the record deferred from WS2) added in PR #26. **Remaining:** wire each dataset's
`process()`/post-process to emit a `VerificationReport` + `StatDerivation` artifact
(sibling of `experiment_reference_index.json`) — done alongside WS5/WS6.

**Goal/Scope.** Build the reusable verification framework once: a
`VerificationReport` carrying provenance (sha256 via `literature/manifest.py`,
source URI/Zotero key, method, page) and L0-L4 results. Every Phase-A dataset runs
through it at the record level (no graph needed).

**Key files.** `torchcell/literature/manifest.py` (REFERENCE), NEW verification
module under `torchcell/`, `schema.py` (L0 target).

**Dependencies.** WS1.

### Checks that must pass

- Instantiates a schema model (L0); runs completeness oracle (L1); cross-method +
  type/range/NaN (L2); unit/convention assertions (L3); cross-source overlap (L4).
- Provenance fields populated; unit-tested; mypy-strict + ruff clean.

## WS4. New phenotype schema: metabolite, protein, visual-score

**Goal/Scope.** Add the three net-new phenotype classes the new datasets require
(decision 3): `MetabolitePhenotype` (Dict[metabolite_id -> float] abundance, with
`n_samples`/SE; metabolite_id aligned to YeastGEM nodes), `ProteinAbundancePhenotype`
(Dict[gene/protein_id -> float]), and a generic ordinal `VisualScorePhenotype`
(integer color/intensity score from visual inspection -- for Ozaydin, reusable by
Cachera). Plus an `ExpressionPhenotype` family decision (microarray vs NGS) carried
into WS10. Add only these; no speculative fields.

**Key files.** `torchcell/datamodels/schema.py`, `pydant.py`,
`torchcell/metabolism/yeast_GEM.py` (REFERENCE for metabolite IDs),
`tests/torchcell/datamodels/test_schema_invariants.py` (extend).

**Dependencies.** WS1.

### Checks that must pass

- New classes pass all WS1 invariants (Liskov, round-trip, registry membership).
- Metabolite IDs validate against the YeastGEM metabolite ID set (compat check).
- VisualScorePhenotype encodes the integer scale + its semantics (ordinal, range).
- Each new field traces to a named dataset (Zelezniak / Ozaydin / Cachera).

## WS5. Verify Sameith2015 + Kemmeren2014 (records)

**Status:** 🔨→✅ done (PR pending). Rebuilt all three LMDBs against the current schema
so records carry `expression_log2_ratio_se` + `n_replicates` (the cached Dec-2025
LMDBs predated the SE code); SM Sameith was **never built before** and now exists (82
records). L0-L4 all PASS on DM(72)/SM(82)/Kemmeren(1450); `verification_report.json`
written next to each `experiment_reference_index.json`. Verifier +
harness + synthetic tests landed (`torchcell/verification/expression.py`,
`scripts/verify_expression_datasets.py`). **Gotcha found:** Sameith rebuild REQUIRES an
injected `SCerevisiaeGenome` (its GEO titles use common names) or it silently
collapses 72→2 records. Full writeup: [[torchcell.verification.expression]].

**Goal/Scope.** Both datasets EXIST (`kemmeren2014.py` ~91KB, validated notes);
verify their LMDB records against the frozen expression schema at L0-L4. No adapter
yet (Phase B). Microarray expression; preserve dye-swap sign + refpool CV-scaled std

- multi-pass gene-name resolution already implemented.

**Key files.** `torchcell/datasets/scerevisiae/{sameith2015,kemmeren2014}.py`
(REFERENCE), verification framework (WS3).

**Dependencies.** WS1, WS3, WS4 (expression family decision).

### Checks that must pass

- L0-L4 pass for both; `n_samples` present per the freeze.
- L3 semantic: expression orientation = log2(sample/reference) correct (dye-swap).
- No genes silently dropped vs the source (completeness oracle).

## WS6. Verify Ohya2005 CalMorph (records)

**Goal/Scope.** TRUE verification (adapter exists, Phase B). Cross-check the
`calmorph_labels` vocabulary used by `schema.py` validators against the born-digital
extraction (`torchcell/literature/calmorph.py`, 501/501), and that the SCMD matrix
(external TSV) + paper schema are both correct.

**Key files.** `calmorph_labels.py`, `torchcell/literature/calmorph.py` (REFERENCE),
`torchcell/datasets/scerevisiae/ohya2005.py`.

**Dependencies.** WS1, WS3.

### Checks that must pass

- L0-L4 pass; `calmorph_labels` parity 501/501 vs extraction.
- Record counts: 4718 mutants + 122 WT accounted for.

## WS7. New dataset: Ozaydin2013 beta-carotene

**Goal/Scope.** Build `ozaydin2013.py`. Data lives in the paper SI as an Excel
spreadsheet of integer visual color-scale scores (visual inspection, integer
scaling) -- map to `VisualScorePhenotype` (WS4). Interpreting the integer scale
("what the numbers mean") needs care; document the encoding. citation_key
`ozaydinCarotenoidbasedPhenotypicScreen2013`.

**Key files.** `torchcell/datasets/scerevisiae/ozaydin2013.py` (NEW), the SI Excel
(via `literature/si_data.py`), `schema.py` VisualScorePhenotype (WS4).

**Dependencies.** WS1, WS3, WS4.

### Checks that must pass

- SI Excel parsed; integer color-scale semantics documented + encoded.
- L0-L4 pass; records carry gene/strain + visual score (+ replicate count if present).

## WS8. New dataset: Cachera2023 CRi-SPA betaxanthin

**Goal/Scope.** Build `cachera2023.py`. Betaxanthin data is inside a PDF table in
the paper -- extract via the literature subsystem (born-digital text layer first,
else VLM). Map to `VisualScorePhenotype` or a continuous fluorescence measurement
depending on the extracted shape (decide from the data, not ahead of it).
citation_key `cacheraCRISPAHighthroughputMethod2023`.

**Key files.** `torchcell/datasets/scerevisiae/cachera2023.py` (NEW),
`torchcell/literature/{extract,scanned}.py`, `schema.py` (WS4).

**Dependencies.** WS1, WS3, WS4.

### Checks that must pass

- PDF table extracted + verified (completeness oracle vs the paper's row count).
- L0-L4 pass; phenotype mapping justified by the extracted field shape.

## WS9. New dataset: Zelezniak2018 metabolite + protein

**Goal/Scope.** Build `zelezniak2018.py` -> two datasets (metabolite, protein),
differentiated by phenotype class per the naming convention. Covers a range of
metabolite + protein-abundance measurements; metabolite IDs aligned to Yeast9 for
constraint-based-model compatibility (decision 4).

**Key files.** `torchcell/datasets/scerevisiae/zelezniak2018.py` (NEW),
`schema.py` Metabolite/ProteinAbundance phenotypes (WS4),
`torchcell/metabolism/yeast_GEM.py` (REFERENCE).

**Dependencies.** WS1, WS3, WS4.

### Checks that must pass

- L0-L4 pass for both metabolite + protein datasets.
- Metabolite IDs map to YeastGEM nodes (report unmatched IDs).
- `n_samples`/SE present per the freeze.

## WS9b. New dataset: Mülleder2016 amino-acid metabolome

**Status:** ⬜ not started (added 2026.07.05). Paper is in the Zotero `database`
collection: citation_key `mullederFunctionalMetabolomicsDescribes2016`, DOI
`10.1016/j.cell.2016.09.007` (Cell, 2016). Aleksej Zelezniak is a co-author,
tying it to WS9.

**Goal/Scope.** Build `mulleder2016.py` -> a metabolite dataset over the
*systematic gene-deletion collection*, storing the absolute amino-acid
metabolome (LC-MS/MS quantification) per deletion strain. Maps to the existing
`MetabolitePhenotype` (WS4) -- **no new phenotype class** -- with metabolite_id
aligned to YeastGEM amino-acid nodes (decision 4). Per the abstract: "over
one-third of coding genes ... contribute to biosynthetic metabolism," and
metabolomic signatures functionally annotate more than half of uncharacterized
genes -- i.e. a genome-scale genotype × amino-acid-concentration matrix.

**Relationship to WS9 (complementary, not redundant).** Zelezniak2018 (WS9)
measures a ~50-metabolite metabolome across only 97 kinase KOs (narrow
genotype, broad readout); Mülleder2016 covers the whole deletion collection but
a narrower amino-acid panel (broad genotype, narrow readout). Together they
bracket the metabolite-phenotype space. Sharing an author + phenotype class,
they should be built with a common metabolite-ID/YeastGEM alignment helper.

**Why it slots in cleanly (design validation).** WS4's `MetabolitePhenotype`
was built for Cachera betaxanthin and designed to be reusable (keyed by
metabolite id + measurement_type + optional Yeast9 id). Mülleder is its **third
consumer** (Cachera, Zelezniak, Mülleder), confirming decision 3 ("no new
structure without data"): this dataset needs a new *loader*, not new *schema*.

**Key files.** `torchcell/datasets/scerevisiae/mulleder2016.py` (NEW), the Cell
SI data table (via `literature/si_data.py`), `schema.py` `MetabolitePhenotype`
(WS4, REFERENCE), `torchcell/metabolism/yeast_GEM.py` (REFERENCE for amino-acid
metabolite IDs).

**Dependencies.** WS1, WS3, WS4 (all landed / near-landed) -- same dependency
set as WS9, no new blockers.

**Values to SOURCE, never guess (CLAUDE.md "Adding Datasets" rule).** The
abstract gives only qualitative scope; before building, source from the Cell
SI / Methods and record with verbatim quote + sha256 + section: the exact
deletion-strain count (genotypes), the amino-acid panel (which metabolites +
their Yeast9 IDs), the replicate structure (`n_samples`) and whether the
reported uncertainty is sample SD vs SE, and the concentration units. Do not
carry a remembered strain/metabolite count into the loader unsourced.

### Checks that must pass

- SI table parsed; per-strain amino-acid concentrations + units documented.
- L0-L4 pass; metabolite IDs map to YeastGEM amino-acid nodes (report unmatched).
- `n_samples`/SE present per the freeze, sourced from the SI (not guess-derived).
- Completeness oracle: strain × metabolite record count matches the SI table.

## WS10. Pan-transcriptome: genomes + NGS expression

**Goal/Scope.** Bring in the pan-transcriptome this round: (a) an external genome
store holding per-strain FASTA + GFF next to the reference S. cerevisiae genome,
with in-memory annotated-genome + gene-embedding construction at dataload
(reference-by-name); (b) an NGS expression dataset; (c) a normalization function
reconciling microarray (Sameith/Kemmeren) vs NGS expression into a common
`log2(sample/reference)`. Genome files external; expression may sit on the graph
this round (decisions 5, 6).

**Key files.** NEW genome-store module under `torchcell/`, NEW
`torchcell/datasets/scerevisiae/<pantx>YYYY.py`, `schema.py` ExpressionPhenotype
family (WS4), reference-genome loader.

**Dependencies.** WS1, WS3, WS4. Source dataset TBD -- see Open Questions.

### Checks that must pass

- Per-strain FASTA + GFF stored with sha256; reference + >=1 wild strain load
  in-memory and produce gene embeddings.
- NGS expression dataset L0-L4 pass.
- Normalization function: microarray and NGS samples map to the same log2 convention
  (round-trip / sign check on a shared gene set).

## WS11. BioCypher <-> Biolink annotation (Phase B)

**Goal/Scope.** Map ONLY the in-scope models (incl. the new metabolite/protein/
visual-score phenotypes + genome pointer nodes) to Biolink categories/predicates:
experiments -> `information_content_entity`, phenotypes -> `phenotypic feature`,
membership -> `part_of`. Resolve the 2026.01.21 entity-mapping flip-flop with one
written principle. Metabolite measurements link to YeastGEM metabolite nodes.

**Key files.** `biocypher/config/torchcell_schema_config.yaml`, `schema.py` (REF).

**Dependencies.** All of Phase A (WS1-WS10) -- the schema must be settled.

### Checks that must pass

- schema<->YAML field-parity passes; no-property-exposure test passes.
- `is_a` hierarchy is an acyclic DAG.
- BioCypher loads the config without error; mapping principle documented.

## WS12. Build + register adapters (Phase B)

**Goal/Scope.** Author + register adapters for every in-scope dataset and stand up
`tests/torchcell/adapters/`. Sameith2015 + Kemmeren2014 are NET-NEW (largest);
Ohya2005 exists (verify); Ozaydin/Cachera/Zelezniak/pan-transcriptome are new.

**Key files.** `torchcell/adapters/{sameith2015,kemmeren2014,ozaydin2013,cachera2023,
zelezniak2018,pantx}_adapter.py` (NEW), adapters conf + KG conf yamls,
`torchcell/database/dataset_adapter_map.py` (MODIFY), `tests/torchcell/adapters/`.

**Dependencies.** WS11 (Biolink/YAML), and each dataset's Phase-A WS.

### Checks that must pass

- Each adapter emits nodes + edges; registered in `dataset_adapter_map.py`.
- Per-dataset KG sub-build succeeds; node/edge counts match the dataset records.
- Adapter suite is mypy-strict + unit-tested (first `tests/torchcell/adapters/`).

## WS13. Full Neo4j KG rebuild (Phase B)

**Goal/Scope.** Run the full BioCypher build via `create_kg.py` over all adapters;
produce `experiment_reference_index` + `gene_set` + `neo4j-admin-import-call.sh`.
Pin BioCypher (currently `Mjvolk3/biocypher@main` -> pin a commit); watch Neo4j 4.4
vs 5.x bulk-import format.

**Key files.** `torchcell/knowledge_graphs/create_kg.py` + conf,
`torchcell/database/dataset_adapter_map.py`, db-env biocypher pin.

**Dependencies.** WS12.

### Checks that must pass

- Build emits the import call; import succeeds; counts match per dataset.
- No dataset silently skipped (audit `experiment_reference_index`/`gene_set` paths).
- `biocypher-out` set `chmod a-w`; BioCypher pinned to a commit.

## WS14. Deploy to NCSA OpenStack (Radiant)

**Goal/Scope.** Deploy the rebuilt KG. Clarified: the mounted data drive lives on
**Delta**; the **OpenStack cloud (Radiant) hosts the database**. So:
export the `.dump`, stage via the Delta mount, import into the OpenStack-hosted
Neo4j (`import_database_openstack.sh`), smoke-test. Fill the missing loader/auth/
health-check around that script.

**Key files.** `database/scripts/export_database.sh`,
`database/scripts/import_database_openstack.sh`, NEW health-check.

**Dependencies.** WS13.

### Checks that must pass

- `.dump` exported + md5-verified after transfer.
- Neo4j up on the OpenStack host (auth + service confirmed).
- Smoke-test Cypher returns expected node/edge counts per dataset.

## Gotchas

1. **LMDB map_size OOM.** Slurm `--mem` unit is MB not GB -- an off-by-1000 OOMs
   the build before any traceback.
2. **Wrong-dataset-path bug.** Verify data paths before blaming infra; see
   `notes/experiments.010-kuzmin-tmi.false-torchmetrics-bug-bc-wrong-dataset-path.md`.
3. **rsync + symlinks.** `readlink -f` symlinked dataset dirs or rsync copies the
   link, not the data.
4. **Silent dataset skip.** `experiment_reference_index`/`gene_set` skip datasets on
   wrong paths -- audit counts, do not trust a clean exit (WS13).
5. **Yeast9 auto-downloads** to `data/torchcell/yeast-GEM-9.0.2/`; metabolite-ID
   alignment (WS4/WS9) must target that exact model version.
6. **Expression convention.** Microarray dye-swap vs NGS orientation differ;
   normalize to log2(sample/reference) explicitly or signs silently invert (WS10).
7. **Visual-score interpretation.** Ozaydin/Cachera integer scales are subjective
   visual inspection -- document the encoding; do not assume linear/quantitative.

## 2026.07.07 - Status reconciliation + emergent workstreams

Reconciled after landing the WS9/WS9b datasets + a new sequence-design track. The
2026.07.04 Progress table above is updated in place; this section records what moved and
what the 06.23 plan did NOT anticipate.

### Phase A -- essentially complete at the data layer

WS1-WS8 + WS9b are ✅ landed on `main` (schema hardening, n_samples/SE, L0-L4 framework,
all three new phenotypes, expression + morphology verification, Ozaydin, Cachera,
Mülleder). WS9 is half done: **Zelezniak protein landed**; the Zelezniak **metabolite**
sheet (~46 LC-SRM metabolites) is not yet ingested. WS10 (pan-transcriptome) is still
blocked on OQ #1 (no source paper chosen). So the ONLY Phase-A remnants are: Zelezniak
metabolite + the pan-transcriptome.

### Emergent workstreams NOT in the 06.23 plan

- **Sequence / strain-design track (new).** `GeneAddition` perturbation for heterologous
  cassettes (branch `fix/ws8-ozaydin-cachera-followups`, **✅ landed** `745acc6a`;
  Ozaydin+Cachera canonical LMDBs rebuilt under the new schema, L0–L4 PASS) +
  `ProteinAbundancePhenotype` + the SBOL-aligned **plasmid sequence store** (`ws10`
  branch, LANDED) + the "total genomic content in the cell" model. This is a distinct
  axis from the roadmap's WS10 pan-transcriptome genome store. See
  `[[torchcell.datamodels.gene-addition-perturbation-design]]`,
  `[[torchcell.sequence.plasmid-and-genomic-content-design]]`.
- **North-star / manuscript framing (new).** `[[paper.north-star]]` (virtual-cell
  consensus + our differentiation) + `[[paper.north-star.dataset-triage]]` (~75-dataset
  candidate backlog, incl. SPELL).

### Bookkeeping: branch-numbering divergence

The GIT BRANCH numbers diverged from these roadmap WS numbers: branch `ws8` = the
Ozaydin/Cachera GeneAddition followups, `ws9` = Mülleder/Zelezniak builds, `ws10` =
plasmid store -- whereas roadmap WS8=Cachera, WS9=Zelezniak, WS10=pan-transcriptome.
Reconcile before "WS10" is ambiguous in conversation.

### WS10 pan-transcriptome (Caudal 2024) -- off-graph seq/genome infra

Source resolved: **Caudal et al. 2024** (`caudalPantranscriptomeRevealsLarge2024`) -- a
pan-transcriptome over natural isolates giving per-strain **genomes (FASTA/GFF) + NGS
expression**. This workstream is architecturally DIFFERENT from the phenotype datasets:
once we ingest full per-strain sequence + genomes, they are TOO LARGE and structurally
unlike phenotype records to live in Neo4j. Plan (extends decision 6 "genomes are pointers,
never payloads"):

- **Sequences/genomes stored OFF-GRAPH** (FASTA/GFF next to the reference genome, sha256-
  pinned), reusing the WS10 external-sequence-store direction now prototyped by the
  SBOL-aligned plasmid store (`[[torchcell.sequence.plasmid-and-genomic-content-design]]`)
  -- a strain genome is another set of contigs in the same owned sequence model.
- **Neo4j holds only REFERENCE/POINTER nodes** (`GenomeAssembly`/`SequenceRecord`/
  `GenomeFeature` with `assembly_id`, `strain`, `fasta_uri`, `gff_uri`, `sha256`,
  `version`) that the loader dereferences to READ sequence INTO other in-memory data
  structures (annotated genome + gene embeddings) at dataload -- the graph is an index,
  not the payload store.
- The NGS expression phenotype still maps to the expression family + the microarray/NGS
  log2 normalization (decision 5).

### Proposed two tracks (decide next priority)

- **Track A -- finish the abstract:** ~~land `ws8` (GeneAddition)~~ **✅ done** ->
  Zelezniak metabolite + WS10 pan-transcriptome (in progress, see 2026.07.07b note) ->
  **Phase B (WS11-14): KG rebuild + NCSA deploy**. The original deliverable; in reach.
- **Track B -- virtual-cell substrate:** dataset backlog from
  `[[paper.north-star.dataset-triage]]` (SIMB metabolic-engineering priority) + the
  sequence/strain-design track (apply-to-sequence, GeneAddition landing, plasmid design).

## 2026.07.07b - ws8 GeneAddition landed; Zelezniak-metabolite + WS10 kickoff

**Chose Track A** (finish the abstract). Sequence of remaining Phase-A data work:

### ✅ Landed: ws8 GeneAddition + Ozaydin/Cachera migration (`745acc6a`, PR #39)

`GeneAdditionPerturbation` (heterologous cassette / extra native copy; relaxes the
systematic-name validator for non-yeast symbols) wired into both Ozaydin (carotenogenic
`YB/I/BTS1`) and Cachera (`Btx-cassette` at XII-5) as the constant engineered background;
PubMed IDs added; `qc_flags -> comment_annotations` rename. **Both canonical LMDBs rebuilt
under the new schema** (Ozaydin 4474, Cachera 4735) and L0–L4 PASS — the `_deleted_genes`
verifier guard excludes `gene_addition` so the cassette never enters the deleted-ORF key
(`L1 orf_uniqueness` still one record per unique deletion). Design:
`[[torchcell.datamodels.gene-addition-perturbation-design]]`.

### ⬜ In progress (this session, overnight): two remaining Phase-A datasets

Running as two coordinated worktrees, coordinating on any schema change (MetabolitePhenotype
already exists, so the metabolite build is expected to be loader-only):

1. **Zelezniak metabolite (finishes WS9).** Add the ~46 LC-SRM targeted-metabolite sheet as
   a sibling of the landed `ProteomeZelezniak2018Dataset`, mapping metabolite names to
   Yeast9 / Yeast-GEM `s_NNNN` species ids via `torchcell/metabolism/yeast_GEM.py` (reuse the
   Mülleder amino-acid → `s_NNNN` mapping pattern). Reuses `MetabolitePhenotype` (WS4);
   provenance (n_replicates, uncertainty type, units) sourced from the paper/SI.
2. **WS10 pan-transcriptome (Caudal 2024).** First OCR the mirrored paper through the MinerU
   pipeline (already added to the library), then determine whether per-strain **genomes**
   (FASTA/GFF) **and transcriptomes** (expression matrices) are both obtainable. If so, build
   the off-graph seq/genome infra (extend the owned sequence store; Neo4j pointer nodes; NGS
   expression → the expression family) per the WS10 note above.

### Bookkeeping still open

Branch-vs-WS numbering divergence (branch `ws8/ws9/ws10` ≠ roadmap `WS8/9/10`) — reconcile
once the pan-transcriptome infra name is settled.

## 2026.07.07c - WS9 + WS10 landed; Phase A data layer complete

Overnight push landed the two remaining Phase-A datasets:

- **WS9 Zelezniak metabolite** (PR #41 `7632333d`) — `MetaboliteZelezniak2018Dataset`, 95
  kinase-KO strains × 50 metabolites; first populated `target_metabolite_ids` (Yeast9
  `s_NNNN` via `build_metabolite_s_id_map`, KEGG-cytosol + BiGG fallback). WS9 complete.
- **WS10 Caudal pan-transcriptome** (PR #42 `a7408e7f`) — `CaudalPanTranscriptome2024Dataset`,
  943 natural isolates (Caudal 969 ∩ Peter-2018 genomes). **Natural isolate modeled as a
  perturbation set off S288C** (decision w/ user): new `SequenceVariantPerturbation`
  (~5047/isolate, off-graph pointer) + `CopyNumberVariantPerturbation` (accessory-present /
  core-absent; PAV = CNV extreme) + `RNASeqExpressionPhenotype` (absolute TPM+count).
  Caudal + Peter 2018 OCR'd (MinerU) + all raw data mirrored/sha256-pinned. L0–L4 PASS.
  See `[[torchcell.datasets.scerevisiae.caudal2024]]`.

**Phase A (schema + datasets, WS1–WS10 incl. WS9b) is now essentially complete at the data
layer.** Remaining threads: the WS10 off-graph sequence STORE + apply-to-sequence
dereferencer (perturbations currently carry pointers; the store/loader is unbuilt), and the
enhancement backlog (Zelezniak/Mülleder pHLUM-as-GeneAddition, Ozaydin 2μ plasmid). Next
major phase: **Phase B (WS11–14) — BioCypher/Biolink annotation, adapters, KG rebuild, NCSA
deploy** — the abstract's finish line.

## Verification

L0-L4 is the universal gate. Phase A verifies at the pydantic/LMDB record level;
Phase B adds graph-level count checks.

- L0 structural -- record instantiates its `schema.py` model.
- L1 completeness -- count / known-keys / contiguity vs an oracle.
- L2 value fidelity -- cross-method agreement, type/range/NaN.
- L3 semantic -- units/conventions (fitness = ko/wt; expression = log2(sample/ref)).
- L4 cross-source consistency -- overlapping entities agree across datasets.

```bash
~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/... -xvs
# mypy-strict + ruff on changed files (/mypy, /ruff skills)
# KG build smoke: create_kg.py sub-build, inspect counts + neo4j-admin-import-call.sh
```

## Open Questions

Most prior ambiguities are now resolved (n_samples canonical; Radiant = OpenStack DB
host with Delta as the data mount; beta-carotene = Ozaydin Excel SI; betaxanthin =
Cachera PDF table). Remaining:

1. **Pan-transcriptome source (WS10) — RESOLVED 2026.07.07.** = **Caudal et al. 2024**
   pan-transcriptome (citation_key `caudalPantranscriptomeRevealsLarge2024`): per-strain
   genomes (FASTA/GFF) + NGS expression across natural isolates. Build out the infra next
   (see the 2026.07.07 off-graph note).
2. **Beta-carotene/betaxanthin extracted artifacts (WS7/WS8).** Source format is
   known (Ozaydin Excel SI; Cachera PDF table); user has extracted-data outputs to
   share. Confirm the exact columns (gene/strain IDs, score/value, replicate counts)
   so the phenotype mapping is fixed -- VisualScorePhenotype vs a continuous value.
3. **VisualScorePhenotype range/semantics (WS4).** Confirm the integer scale once
   extracted (e.g. ordinal -5..+5 color intensity) so the encoding is faithful.
