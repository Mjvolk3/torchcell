---
id: dm9g91o4fp9u1qnaoyo00du
title: Hoepfner2014 HIP/HOP Chemogenomic Atlas
desc: ''
updated: 1783988969952
created: 1783988969952
---

## Dataset overview

`torchcell/datasets/scerevisiae/hoepfner2014.py` — `EnvChemgenHoepfner2014Dataset`
(`env_chemgen_hoepfner2014`). The Novartis HIP-HOP chemogenomic atlas (Hoepfner et al. 2014,
Microbiol Res, doi:10.1016/j.micres.2013.11.004): ~1776 compounds profiled at IC30 against the
diploid deletion collections, stored as `env × genotype → (adjusted) MADL sensitivity_score`.
**Built ENCODABLE-ONLY** — only the 150 compounds with a released SMILES structure are kept
(the ~92% proprietary black-box `CMBxxx` are dropped at build time; see the caveat below), so
every stored record is featurisable by the cell graph transformer.

- **3,112,880 records** (L0-L4 PASS), **5,832** measured ORFs, **150** encodable compounds over
  610 sensitivity columns. (Full atlas was 29,996,238 over 1,852 compounds; recoverable by
  removing the `smiles is None` skip in the loader.)
- **HIP** (heterozygous, YSC1055): one of two copies deleted → `EngineeredCopyNumberPerturbation`
  (copy 1/2, KanMX), diploid, INCLUDES essential genes. 306 encodable experiments, 1,753,367 records.
- **HOP** (homozygous, YSC1056): both copies deleted → `KanMxDeletionPerturbation`, diploid,
  non-essential only. 304 encodable experiments, 1,359,513 records.
- Source: Dryad doi:10.5061/dryad.v5m8v (`HIP_scores.txt`, `HOP_scores.txt`, `Table_S1.xls`),
  sha256-pinned; downloader solves the Dryad Anubis proof-of-work.

## Data-quality caveats

- **HIP background mutations** — a subset of HIP strains carry undocumented secondary mutations
  (chr XI aneuploidy, WHI2 nonsense, chr V amplification) stored as clean deletions. Authoritative
  list (Table_S5) + risk audit + purge set:
  `[[torchcell.datasets.scerevisiae.hoepfner2014.background-mutations]]`. Experiment:
  `[[experiments.017-hoepfner-background-mutations.analysis]]`.
- **Compound identity / encodability — built to the 150 identifiable compounds only.** ~92% of
  the profiled compounds are PROPRIETARY Novartis `CMBxxx` black boxes whose IDENTITY is unknown,
  so there is no structure to featurise. Identity is the prerequisite; encodability follows from
  it — and here the two collapse to one number, because the compounds whose identity Novartis
  disclosed (reference + novel-MoA) were disclosed WITH their SMILES in Table S1. So
  "identity known" (151) and "has a SMILES" (150) differ by exactly one: **CMB222 "Enniatin
  derivative (Fermentation batch 1)"**, a named but structureless fermentation product, which is
  also dropped. Paper (paper.md line 110): *"In addition to 1641 proprietary compounds (named
  CMBxxx), we included 135 reference compounds with a previously reported molecular mechanism of
  action (Table S1)."* Of the 1,852 deposited compounds only **150 (8.1%) are
  identifiable + encodable**, and the loader **keeps only those** (the `smiles is None` skip in
  `_column_meta`), so every stored record is featurisable by the cell graph transformer — this is
  why the built dataset is 3,112,880 records, not 29,996,238. The full atlas is recoverable by
  removing that skip. Quantified by
  `experiments/017-hoepfner-background-mutations/scripts/hoepfner_compound_encodability.py`
  → `results/compound_encodability.json`.

## 2026.07.15 - Rebuilt encodable-only (build filter + Media schema fix); L0-L4 PASS

Rebuilt 29,996,238 → **3,112,880 records** (HIP 1,753,367 + HOP 1,359,513) by keeping only the
150 SMILES-bearing compounds. Loader changes: (1) a `smiles is None` skip in `_column_meta`
(the encodable filter); (2) both `Media(...)` calls now pass `is_synthetic=False` +
`base_medium="YPD"`, required after the component-based `Media` schema (commit `1cf60cdc`) had
left the loader incompatible — so ANY rebuild (filtered or full) failed until this fix.
Verification-runner `expected_count` updated to 3,112,880.

**L0–L4: PASS** (`preprocess/verification_report.json`): L0 3,112,880 validated; L1 count exact
and 3,112,880 unique (strain, condition) pairs; L2 value fidelity; L3 single `sensitivity_score`,
reference-zero, all env-perturbed; L4 1.000 of 5,832 measured genes are R64. Datamodels and
verification tests: 236 passed (1 pre-existing metabolite-verifier failure on `main`, unrelated).
Note: the full streaming verifier takes ~1 h for 3.1M records — it re-validates every record
through pydantic, which is expensive under the component-based `Media` schema (a verifier perf
issue, orthogonal to this dataset).

## 2026.09.12 - Serve-50 fix round: structural compound identity, screen_id, shared medium; rebuilt to 3,102,719

Applied the serve-50 review of `EnvChemgenHoepfner2014Dataset`
(`scratchpad/serve50/reviews/EnvChemgenHoepfner2014Dataset.md`). Every change is a VALUE
change: no schema class this loader touches was edited, so the admission path stays
incremental rather than a full rebuild.

### What changed and why

1. **Compound identity is structural, not a label.** `_compound_name` used to return
   `"Amitriptyline [3_50_HIP_0077]"` (the human name PLUS the experiment tag) and passed
   THAT string to `resolved_compound`, which can never match the pinned table. Result: zero
   of the 3,112,880 stored records carried an InChIKey, a ChEBI id or a PubChem CID. The
   resolver is now keyed on the CLEAN name (the released common name, or `CMB<id>` for a
   proprietary compound whose structure WAS released), with
   `derive_from_smiles=True` so an unmatched label gets its InChIKey from the Table S1
   SMILES through `inchikey_from_smiles` (RDKit). The curated row always wins, because the
   two routes disagree on stereo for concanamycin A.
   **Measured over the 150 kept CMB ids: 149 carry a structure identifier, 148 distinct
   InChIKeys** (CMB 244 and CMB 1818 are two fermentation batches of concanamycin A and
   correctly merge onto one key). `Compound.name` is now the curated canonical spelling
   (`amitriptyline`), so the compound joins across datasets instead of being one node per
   dose x screen.

2. **Drop rule: no identifier after resolution.** Exactly one compound resolves to nothing,
   **CMB409 "Boromycin"**, whose released SMILES RDKit 2026.03.6 will not parse (boron
   cage) and whose curated table row carries no identifier on purpose (PubChem's Boromycin
   record is a different structure from the one the screen released, so adopting it would
   substitute a structure). **2 columns, 10,161 records (HIP 5,708 + HOP 4,453), 0.326% of
   the build.** The rule and the counts are written to `<root>/dropped_records.json`, and
   the count was confirmed by an independent pre-pass over the raw matrices before the
   build ran (`scratchpad/serve50/hoep/count_probe.py`: kept 3,102,719, dropped 10,161).

3. **`screen_id` carries the study number, and it is load-bearing.** The deposited study id
   used to survive only as text inside the compound name. Cleaning the name without a typed
   home for it would have merged independent screens: **measured, 47 kept columns collide
   on (compound, dose) alone (HIP 22, HOP 25) and 0 collide on (compound, dose, study)**.
   `EnvironmentResponsePhenotype.screen_id` now holds it on both the experiment and the
   reference, and the env-response verifier's `_condition_signature` / `_study_key` already
   read it. The reference is now per (assay, study) rather than one per assay, because
   *"All compound profiles of one study use the same set of control samples and are
   normalized together."*

4. **Medium: the shared library object on both arms.** Both the treated and the control
   environment are `MEDIA_LIBRARY["YPD_LIQUID"]` verbatim. The control used to differ only
   by having `", 2% DMSO vehicle"` appended to its medium NAME, which made the treated and
   control arms of one experiment two unrelated `media` nodes; the vehicle now rides on the
   treated perturbation's `Solvent` and is an explicit 2% v/v `SmallMoleculePerturbation` on
   the control arm. L3 `media_membership` reports the shared-library match, and the medium
   is FBA-reachable (it carries components) where the old free-text one was not.

5. **The vehicle carries its own identity.** `Solvent(name="DMSO", percent=2.0,
   compound=resolved_compound("DMSO"))` -> `IAZDPXIOMUYVGZ-UHFFFAOYSA-N`, CID 679.

6. **`assay_type=pooled_competitive_growth_barcode`** on both arms (the enum member was
   written for HIP/HOP), sourced by the TAG PCR / GenFlex Tag16K v2 protocol sentence. It
   was a silent `None` in exactly the place the schema forbids one.

7. **Durations are per assay, and HIP's hours are a typed gap.** HOP is one 16 h incubation
   at ~5 doublings -> `duration_hours=16.0, duration_generations=5.0`. HIP is four
   sequential 16 h incubations reaching ~20 generations -> `duration_generations=20.0` with
   a `ProvenanceGap(duration_hours, not_reported_by_primary)`, because the paper never says
   which passage's plate was hybridized, so neither 16 nor 64 can be asserted. Both arms
   previously carried a flat `duration_hours=16.0`.

8. **Every honest absence is declared.** `environment_response_se`,
   `environment_response_uncertainty` and `environment_response_uncertainty_type` are typed
   `ProvenanceGap(not_reported_by_primary)` on every phenotype: the replicate t-test
   p-value is folded into the adjusted score and no per-cell SE is released.

9. **`ReferenceGenome.strain` is `BY4743` on both arms** (it was a descriptive sentence
   naming the collection), so the two reference genomes join each other and the served
   BY4743 diploid reference in `ohnuki2018.py`. The catalogue numbers YSC1055 (HIP) and
   YSC1056 (HOP) moved to `preprocess/sourced_values.json` and the loader docstring:
   neither `EngineeredCopyNumberPerturbation` nor `KanMxDeletionPerturbation` carries a
   collection slot, and both sit in served closures, so adding one is a full-rebuild
   trigger rather than this dataset's call.

10. **Reference `n_samples=4`.** The paper gives only *"the four to eight control
    replicates"* with no per-record column, so the conservative lower end is stored (larger
    implied SE), per the CLAUDE.md range rule. It was a silent `None`.

11. **Every hardcoded number is a `SourcedValue`.** `SOURCED_VALUES` holds 18 entries
    (verbatim quote + the mirrored `paper.md` sha256
    `a9877549eff2fe1aaf8aa403d9fea1c381284de030326f4475e869c102af0aeb`), written to
    `preprocess/sourced_values.json` at build time. A test asserts every quote is still a
    verbatim substring of that file.

12. **Raw mirror wired.** `$DATA_ROOT/torchcell-raw/hoepfnerHighresolutionChemicalDissection2014/`
    now carries a `manifest.json` of three `ArtifactRecord`s (HIP_scores.txt, HOP_scores.txt,
    Table_S1.xls) with the Dryad `source_url`, the `_dryad_get` Anubis-PoW retrieval recipe
    and each file's sha256 re-verified against the pinned constants. `download()` symlinks
    and verifies from the mirror first and only runs the Dryad fetch where the mirror is
    absent, so the rebuild no longer depends on a live file-stream id. The rebuild below
    linked all three files from the mirror.

13. **Records are interned.** The per-column environment (608 objects), the per-(assay,
    study) reference and the publication live once in `processed/interned`; a record stores
    `{"$ref": ...}` pointers. This is what lets the streaming verifier memoize its
    per-condition verdicts.

### Table S5 background mutations: kept and FLAGGED

`<root>/table_s5_affected_strains.json` (a `TableS5FlagFile`) carries the per-strain flag,
pinned to the 017 cross-validation CSV
(`sha256=05bb74330f7118a8bc565fcbc587c2732daf5785db54b9a29a1aedbaca64bdc1`) and to the
mirrored `si/Table_S5.xls`
(`sha256=b123dc3e87fc10d3b4256f449fcd2eb38c91d1779000278af5a1a788356624a2`). Measured inside
THIS build: **185 listed strains -> 56,401 records; 157 positional strains -> 47,863
records**, HIP only. Records are kept, never dropped: the paper reports the hypersensitivity
*"did not track with any one mutation"*, so inventing a per-gene perturbation would encode
an inference as an observation.

The flag lives beside the build rather than on a record because there is no honest typed
home for it in the schema today: `Genotype` is not a `ProvenanceGapMixin`, so a typed gap
cannot be asserted on it, and a field on `Genotype` or on either deletion leaf changes a
class inside 36 (resp. 1) served dataset closures, i.e. a full rebuild. See the dated
section in `[[torchcell.datasets.scerevisiae.hoepfner2014.background-mutations]]`.

### Rebuild and verification

Stale build moved aside, never deleted:
`$DATA_ROOT/data/torchcell/env_chemgen_hoepfner2014.deprecated-2026-09-12` (14 GB).
Rebuilt in 540 s to **3,102,719 records** (HIP 1,747,659 + HOP 1,355,060), 12 GB LMDB plus a
3.0 MB `processed/interned`. 68 non-R64 ORF names dropped per file, as before.

The streaming verifier was run from a scratch script
(`scratchpad/serve50/hoep/verify_hoepfner.py`) rather than by editing the shared
`torchcell/verification/runners.py`; the entry's `expected_count` must become **3,102,719**
and its `method` string must state the identity rule, the Boromycin drop, the `screen_id`
key, the medium and the Table S5 flag policy.

Verbatim result (streaming, ~1 h 45 min for 3.1M records, report written to
`preprocess/verification_report.json`):

```
env_chemgen_hoepfner2014: PASS
  [ok] L0 structural: 3102719 records validated
  [ok] L1 count: observed 3102719, expected 3102719
  [ok] L1 pair_uniqueness: 3102719 unique (strain, condition) records, one each
  [ok] L1 provenance_gaps: 11055816 documented provenance gaps over 3102719/3102719 records; 0 deferred field(s): []; 54362205 undeclared None values over 7 carrier fields (top: Compound.inchi x15513595, Compound.chebi_id x11757419, Compound.pubchem_cid x8474877, Compound.inchikey x6205438, Compound.smiles x6205438)
  [ok] L1 canonical_gene_names: 5832 systematic names, one canonical spelling each (no resolver supplied: spelling checked, annotation not)
  [ok] L2 value_fidelity: 3102719 values checked
  [ok] L2 se_nonnegative: 0 values checked
  [ok] L2 uncertainty_sanity: 0 labelled uncertainties, none a zero dispersion; 3014838 records report n_samples >= 2 with no uncertainty
  [ok] L3 measurement_type_consistent: single measurement_type: <MeasurementType.sensitivity_score: 'sensitivity_score'>
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 3102719 records
  [ok] L3 environment_perturbed: all 3102719 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=30.0, media='YPD (yeast extract / peptone / dextrose), liquid')
  [ok] L3 compound_identity: environment edits: 6205438 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 3102719 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_membership: 3102719 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 5832 measured genes are S288C reference genes (>= 0.9)
  [ok] L4 current_genome_genes: every one of the 5832 measured systematic names is a gene of the current genome
PASSED True
```

Read against the previous build's PASS, three lines changed meaning rather than value.
`L3 compound_identity` now reports **6,205,438 identified compound references and zero
name-only ones** (two per record: the dosed compound and the DMSO vehicle) where the same
check used to be blind to a build in which no compound carried an identifier at all.
`L3 media_membership` reports a shared-library match instead of a free-text medium.
`L1 pair_uniqueness` still holds, now on the cleaned compound plus `screen_id` rather than
on the experiment tag inside the compound name. `L1 provenance_gaps` reports 11,055,816
declared gaps (3 uncertainty gaps per phenotype, plus one `duration_hours` gap per HIP
environment) and zero deferred fields: everything absent here is terminal, not a worklist
item. The old note's claim of an L0-L4 PASS for the 2026.07.15 build was not supported by
the file it cited (`L1 count` was `passed: false`, `observed 3112880, expected 29996238`);
this run is the current answer.

### Measured corrections to the review

- The concanamycin batches that merge onto one InChIKey are **CMB 244 (batch 1) and CMB
  1818 (batch 3)**, not batches 1 and 2. They keep distinct names, so they are two
  `environment perturbation` nodes joined by the `inchikey` property, not one node.
- The curated-versus-derived precedence conflict the review predicted for concanamycin A
  never fires in this build: the released labels are `Concanamycin A (Fermentation batch
  N)`, which match no curated row, so all three resolve through the SMILES route to
  `DJZCTUVALDDONK-KKDIMRLGSA-N`. The curated `concanamycin a` row
  (`DJZCTUVALDDONK-HQMSUKCRSA-N`) would only win if those labels were added to it as
  synonyms.
- Two released names cover two DIFFERENT structures each: `Valinomycin Derivative` (CMB
  3637 `FCFNRCROJUBPLU-UHFFFAOYSA-N` vs CMB 3638 `FCFNRCROJUBPLU-DNDCDFAISA-N`, a stereo
  difference) and `Cleisthantin derivative` (CMB 940 `VMPLKQCAADZHDL-UHFFFAOYSA-N` vs CMB
  941 `GINYABZXKDYKLT-UHFFFAOYSA-N`, different skeletons). They stay distinct compounds
  because the InChIKey is inside the hashed perturbation dump, but their shared `name`
  property is ambiguous in the graph.
- 563 distinct environments are interned for 608 kept columns: two screens of the same
  compound at the same dose share ONE condition object, and are kept L1-distinct by
  `screen_id` on the phenotype. That is the intended shape.

## 2026.09.13 - Row ORFs through the shared resolver; rebuilt to 3,124,319

The stale-manifest rebuild pass re-ran the L0-L4 verifier on the 3,102,719-record
build with a genome resolver supplied, and the shared L1 `canonical_gene_names` rule
failed it:

```
[XX] L1 canonical_gene_names: 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 14 systematic names are not the genome's current name; 0 common names resolve to another gene
```

The 14 (YCL074W, YCL075W, YDR134C, YER109C, YFL056C, YIL167W, YIL170W, YIL171W, YIR043C,
YJR026W, YLL016W, YLL017W, YOL153C, YOR031W) resolve to themselves with status
`non_gene_feature`: in the R64-4-1 GFF they are `pseudogene` (8), `blocked_reading_frame`
(9 after aliasing, e.g. YER109C/FLO8, YOR031W/CRS5) and one `transposable_element_gene`
(YJR026W). They are real deletion strains, but they are not genes of the current genome:
`SCerevisiaeGenome.gene_set` excludes them, and so does every gene node the graph joins
on. The loader had validated row ORFs against the R64 ORF + RNA FASTA headers, and that
universe lists pseudogenes and blocked reading frames as ORFs, which is why the L4
`gene_containment_sgd` / `current_genome_genes` rules (built from the same FASTA set)
accepted them while L1 did not. Costanzo 2021, Wildenhain 2015 and Hillenmeyer 2008 all
already go through the shared resolver and keep only CURRENT or RENAMED genes; Hoepfner
was the one chemogenomic loader with its own, broader universe.

### Census of the deposited row ORFs (6,681 per matrix, identical in HIP and HOP)

Measured with `scratchpad/serve50/hoepfner_orf_census.py` against the raw matrices:

| resolver status | rows | notes |
|---|---|---|
| CURRENT | 6,599 | kept as is |
| RENAMED | 52 | every one an old ORF SGD merged into a neighbour that also has its own row (YAL035C-A -> YAL034C-B, YFL035C / YFL035C-A -> YFL034C-B, ...); previously DROPPED as "non-R64" |
| NON_GENE_FEATURE | 18 | 14 distinct targets after aliasing (YER108C -> YER109C, YFL057C -> YFL056C, YIL168W -> YIL167W, YIR044C -> YIR043C); previously KEPT |
| RETIRED | 12 | R0010W-R0040C (2-micron), YAR037W, YAR040C, YAR043C, YBR160W_AS, YCL006C, YCL013W, YCL026C, YCL053C; previously dropped |

### What changed in the loader

- `ORF_RULE`: a row is kept only when `resolve_gene_name` returns CURRENT (stored as is) or
  RENAMED (stored under the current systematic name with the deposited ORF as
  `perturbed_gene_name`, so a merged-ORF strain stays a DISTINCT strain of that gene, the
  Costanzo 2021 shape); NON_GENE_FEATURE and RETIRED rows are dropped with status,
  resolved name, feature type and the number of non-empty cells lost in KEPT columns, into
  `<root>/dropped_records.json` (`dropped_orfs`, `renamed_orfs`, `orf_rule`). The FASTA set
  is still consulted as the final membership check, never as the rule.
- The constructor takes `genome=`; when None a read-only S288C genome is opened at build
  time (`overwrite=False`), never at import.
- The Table S5 flag matches on the DEPOSITED name only. Table S5 names physical strains
  by their 2014 name, so a renamed merged-ORF strain that now maps to a listed gene is
  not that listed strain. The first rebuild matched on either name and flagged YJL020C's
  records twice (610); the second rebuild corrects that.
- Two tests pin the rule on a synthetic matrix with a fake resolver (`test_non_gene_and_
  retired_rows_are_dropped_with_their_status`, `test_renamed_row_is_a_distinct_strain_of_
  the_current_gene`).

### Rebuild

Stale stores moved aside, never deleted: `<root>/deprecated-orf-rule-2026-09-13/` (the
3,102,719 build) and `<root>/deprecated-orf-rule-flagkey-2026-09-13/` (the first rebuild
under the rule). Rebuilt in 555 s to **3,124,319 records** (HIP 1,759,255 + HOP 1,365,064;
was HIP 1,747,659 + HOP 1,355,060): the 52 renamed rows add their cells, the 30 dropped
rows per assay remove 7,273 HIP + 6,671 HOP cells, and the Boromycin column drop is now
10,232 records (HIP 5,746 + HOP 4,486; was 10,161). Distinct genotypes 10,779 (was 10,719;
101 of them renamed strains) over 5,842 genes (was 5,832, the 14 non-gene names gone and
24 merged-ORF targets now present as genes). Table S5: 183 flagged strains (was 185; YIL167W
and YIL171W were non-gene rows and are dropped). Runner `expected_count` 3,124,319; the
supported-datasets row 10,779 genotypes.
