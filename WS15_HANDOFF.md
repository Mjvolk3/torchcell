# WS15 Handoff — Environmental/Chemogenomic Ingestion + Ontology Hardening

**For a fresh session.** Read this first, then `WS15_PLAN.md`, `WS15_STATUS.md`, and
`notes/paper.database.ontological-enforcement.md`. Memory:
`environmental-chemogenomic-ingestion-plan` + `paper-r5-chemogenomic-and-inference-thesis`.

## 2026.07.10 — Tier-1 items 1–3 DONE (landing); item 4 IN PROGRESS next

Implemented + committed on `ws15-env-chemogenomic` (5 commits `0b5d542f`→`6576027f`):

1. **Schema refactor** (`0b5d542f`): `SmallMoleculePerturbation` carries a typed
   `Compound` (InChIKey-keyed, ChEBI roles) + `EnvironmentPhysicalPerturbation` (neutral
   scalar factor); `EnvironmentStressType`/`stress_type`/`stress_category` DELETED (M1);
   typed `ConcentrationUnit`/`DoseBasis`/`TemperatureUnit` enums (G2); `copy_number > 0`
   validators on all 3 dosage leaves (M2 — absence = `NaturalGeneAbsence` only).
2. **Invariant tests** (`90971766`): `tests/torchcell/datamodels/test_ontology_all_trees.py`
   — S3–S7 across env/phenotype/experiment/reference trees + M1–M4 + G1–G3. Deferred
   gene-tree M1 (suppressor/ts) and phenotype measurement-typing are `strict=True` xfails
   that Tier 2 flips green. All datamodels tests pass; changed source files mypy-clean.
3. **Loaders + rebuilds** (`09e7e74a`, `db862707`): 5 env loaders migrated; all 4 built
   env datasets REBUILT on the new schema and **L0–L4 PASS**: Vanacloig 164,115 · Mota
   1,273 · Wildenhain 428,573 · Auesukaree 525. Auesukaree heat is now a raised
   `Environment.temperature` (no perturbation, M2); the L3 `environment_perturbed` check
   accepts a non-baseline temperature as a valid edit.

**NEXT — item 4 (this session, after landing 1–3):** streaming `post_process` +
Hoepfner/FitDb/Lee. Blocker is concrete: `ExperimentReferenceIndex.index` is a dense
`list[bool]` of length N stored PER unique reference — Wildenhain (428k) already took
~15 min in that loop; Hoepfner (~30M, 70×) would be tens of GB + hours and can't build
as-is. Plan: compact/sparse reference-index redesign FIRST (touches every
`ExperimentReferenceIndex` consumer), then Dryad downloads + multi-hour builds; FitDb/Lee
loaders don't exist yet. **Tier 2** (Sga collapse, gene-side suppressor/ts, CopyNumber
axis ABC, phenotype measurement-typing backfill) still starts after the env studies merge.

---

**(Original gate note, superseded by the above:)**

## Where we came from (the arc)

1. Goal: back the Nature-Biotech paper's **Result R5 = "Drug exposure and representation
   robustness"** (chemogenomic drug×YKO) — the Mac PDF `temp_editing.pdf` is ahead of repo
   `results.tex`; R5=drug-exposure, R6=metabolism/strain-design.
2. Ran a chemogenomic-YKO deep-research pass → added rows 76-79 to
   `notes/paper.north-star.dataset-triage.md` (landed on main, `188bd313`) and tagged all 40
   Zotero `database`-collection items `torchcell.paper.results{1-6}`.
3. Selected env×genotype datasets, OCR'd 16 papers via MinerU into
   `$DATA_ROOT/torchcell-library/<citation_key>/paper.md`.
4. Built datasets on a NEW `EnvironmentPerturbation` schema (this branch).
5. User flagged that `stress_type` encodes a *consequence*, not an edit → spawned 4 ontology
   critics → wrote the enforcement spec. **We are now at the ontology-refactor gate.**

## Current state — branch `ws15-env-chemogenomic` (off main `188bd313`)

Worktree: `~/Documents/projects/torchcell.worktrees/ws15-env-chemogenomic`. Interpreter
`~/miniconda3/envs/torchcell/bin/python`, `PYTHONPATH=<worktree>`. Nothing on main; land via
merge queue after review.

Commits (newest last): `c7882949` EnvironmentPerturbation schema + Vanacloig · `d30ff71b`
Mota · `08ca8b16` Wildenhain · `291b2a2d` EngineeredCopyNumberPerturbation + ReferenceGenome.ploidy
· `2923575c` Hoepfner LOADER (build deferred) · `888a5c85` Auesukaree · `03f846df` Nadal
deferral flag. (Ontology note + this handoff currently UNCOMMITTED in the worktree.)

**Built + L0–L4 verified (4):** Vanacloig2022 (164,115 rec, GEO GSE186866, log2_ratio, n=3),
Mota2024 (1,273, categorical, 3 acids), Wildenhain2015 (428,573, z_score, PubChem AID 1159580),
Auesukaree2009 (525, categorical, 6 stresses, first PhysicalStress). LMDBs at
`$DATA_ROOT/data/torchcell/env_chemgen_<name>/`.

**Deferred:** Hoepfner/FitDb/Lee (each ~30M rec/~100GB, need streaming `post_process`; Hoepfner
loader committed, raw on Dryad `10.5061/dryad.v5m8v`, partial LMDB already deleted). Nadal
(pseudobulk; data only in R/Seurat S4 objects — needs R installed; deferral flag committed).

## APPROVED PLAN (do next)

### Tier 1 — this branch, before merge
1. **Schema refactor** (`torchcell/datamodels/schema.py`): env perturbations →
   `SmallMoleculePerturbation` (chemical species) + `EnvironmentPhysicalPerturbation`
   (scalar factor: pH/osmolarity/carbon; temperature stays on `Environment`); **DROP**
   `EnvironmentStressType`, `PhysicalStressPerturbation.stress_type`,
   `SmallMoleculePerturbation.stress_category`. Add typed `Compound` block
   `{name, inchikey(canonical, validated), inchi, smiles(aux), pubchem_cid:int, chebi_id, roles:[ChEBI]}`
   (reuse for solvent + physical agent); mode-of-action → ChEBI role. Typed UO units on
   `Concentration`/`Temperature`. **`copy_number > 0` validator** on both CNV leaves (absence is
   the presence/absence leaf only — copy_number=0 is meaningless). Backfill `measurement_type`(enum)
   +`units` on env phenotypes as needed.
2. **New invariants** = implement every rule in `notes/paper.database.ontological-enforcement.md`
   §"Detailed test specifications" that is Tier-1-scoped: port S3/S4/S5(unique only)/S6/S7 to
   env/phenotype/experiment trees; add M1 banned-vocab lint (ALLOWLIST empty for env),
   M2 canonical-form, M3, M4, G1(SO allowlist + InChIKey), G2(unit enums), G3.
3. **Rebuild** the 4 env datasets on the refactored schema; re-verify L0–L4.
4. **Streaming `post_process`** (memory-efficient reference-index build — current one holds ~60GB
   RAM at 30M records) → build Hoepfner/FitDb/Lee, and Nadal if R is provisioned.
5. Commit incrementally; land via `/enqueue-merge`.

### Tier 2 — IMMEDIATELY AFTER MERGE (separate branch; touches landed Costanzo/Kuzmin/Kemmeren)
`Sga*` collapse (assay method → experiment/provenance metadata; fixes S5 inherited-tag defect);
gene-side `suppressor`/`ts` (allowlist as identity-handle vs demote to phenotype — the ts/suppressor
names are handles to published reagent collections for future sequence population); `CopyNumber`
axis ABC (S8); phenotype-typing backfill on Fitness/GeneInteraction/Microarray (rebuilds those
datasets); conversion-registry still deferred (let the model learn readouts; revisit only for
output-size reduction).

## Key files

- `torchcell/datamodels/schema.py` — the ontology (perturbations, environment, phenotypes,
  experiments). ~2232 lines.
- `torchcell/datamodels/conversion.py` (+ `*_to_fitness_conversion.py`) — EXISTING typed
  conversion registry (synth-lethal→fitness=0); leave as-is (Tier-2+).
- `tests/torchcell/datamodels/test_ontology_invariants.py` (gene-only today; extend to all trees),
  `test_schema_invariants.py`, `test_uncertainty_ontology.py`.
- `torchcell/datasets/scerevisiae/{vanacloig_...,mota2024,wildenhain2015,auesukaree2009,hoepfner2014}.py`
  + the deferred nadal loader (not built).
- `torchcell/verification/{environment_response.py,runners.py,levels.py}` — L0–L4.
- `biocypher/config/torchcell_schema_config.yaml` — external Biolink grounding (`is_a:`), Phase B only.
- `notes/paper.database.ontological-enforcement.md` — the 16-rule spec (LaTeX + mermaid + detailed
  test specs). SOURCE FOR PAPER FIGURES.
- `notes/paper.north-star.dataset-triage.md` (on main) — dataset backlog; rows 76-79 added.
- `$DATA_ROOT/torchcell-library/<ck>/paper.md` — OCR'd provenance for all 16 env papers.

## Gotchas
- ONE agent per worktree (shared-git clobber); instruct dataset agents to **build SYNCHRONOUSLY,
  never detach** (a detached Hoepfner build ran away to 116GB + a poll-looping agent had to be
  TaskStop'd).
- Genome-scale chemogenomic builds need the streaming `post_process` FIRST.
- Source-or-stop: never guess provenance; sha256-pin raw; recompute-from-raw-with-recipe is the
  adopted policy when processed SI is un-scriptable.
