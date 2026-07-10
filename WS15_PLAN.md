# WS15 — Environmental / Chemogenomic Ingestion (autonomous, 2026-07-09)

**Goal:** ingest 8 datasets as `deletion genotype × EnvironmentPerturbation → phenotype`
records on a NEW reusable `EnvironmentPerturbation` schema, each **L0–L4 verified**.
Work sequentially. **Stop-on-fail with a written note; NEVER fabricate provenance.**

## Ground rules
- Work ONLY in this worktree (`ws15-env-chemogenomic`). Never touch primary `main`.
  No parallel git ops. Commit per verified dataset.
- Interpreter: `~/miniconda3/envs/torchcell/bin/python`. `DATA_ROOT` shared with main.
- OCR'd papers: `$DATA_ROOT/torchcell-library/<citation_key>/paper.md` — READ these to
  SOURCE every statistic (n_samples, units, readout convention, background genotype)
  per CLAUDE.md "Adding Datasets". If a value isn't sourceable → STOP + note, don't guess.
- pydantic-first; NO fallback/try-except padding; mypy-strict + ruff on changed files;
  the ontology-integrity tests must still pass; reuse existing dataset/verifier patterns;
  add the MINIMUM new structure.

## New schema (design in Phase 1 on Vanacloig; FREEZE after fit-check)
```
EnvironmentPerturbation:
  SmallMolecule{ compound_name, compound_id (PubChem CID/ChEBI | None),
                 smiles (|None), concentration{value,unit,basis e.g. IC30},
                 solvent{name,pct} }
  | Stress{ stress_type ∈ {temperature,osmotic,oxidative,pH,carbon_source,alcohol},
            magnitude{value,unit} }
  shared: base_medium, aerobicity ∈ {aerobic,anaerobic}, duration, temperature
Readout: FitnessPhenotype (or existing phenotype) +
  measurement_type ∈ {log2_ratio, z_score, sensitivity_score, categorical, growth_rate}
  + n_samples + SE + units   (ALL sourced)
Constant background: drug-sensitized backgrounds (Vanacloig 3ΔAlpha = pdr1Δ pdr3Δ snq2Δ)
  modeled as a CONSTANT GeneDeletion background (reuse Ozaydin/Cachera constant-bg pattern).
```

## Datasets (in order) — citation_key · data source · readout · scale
1. **vanacloig-pedrosComparativeChemicalGenomic2022** — GEO **GSE186866** — log2(inh/ctrl)
   barcode, **n=3 biological triplicate** — 4309 del (3ΔAlpha bg) × 34 inhibitors (incl
   isobutanol, ethanol), anaerobic, IC30. **SCHEMA ANCHOR.**
2. **hoepfnerHighresolutionChemicalDissection2014** — Dryad **10.5061/dryad.v5m8v**
   (`HIP-scores.txt`,`HOP-scores.txt`, systematic-ORF rows) — HIP/HOP sensitivity score —
   het(CNV1)+hom(absence) × ~1776 cpds at IC30.
3. **auesukareeGenomewideIdentificationGenes2009** — J Appl Genet SI/tables — categorical
   sensitive/tolerant — 4828 del × 6 stresses (ethanol, methanol, 1-propanol, heat, NaCl, H2O2).
4. **motaSharedMoreSpecific2024** — Microb Cell Fact SI (PMC10903034) — tolerance score —
   del × {acetic, butyric, octanoic} acid.
5. **wildenhainPredictionSynergismChemicalGenetic2015** — PubChem BioAssay **AID 1159580** /
   ChemGRID — Z-score, duplicate — 195 sentinel del × 4915 cpds. (Skip the cryptagen
   chemical-chemical synergy layer — out of scope.)
6. **hillenmeyerChemicalGenomicPortrait2008** (FitDb) — `chemogenomics.stanford.edu`
   supplement — fitness-defect score — het+hom × 1144 conditions.
7. **leeMappingCellularResponse2014** — Science SI + Nislow/Giaever portal — fitness score —
   HIP+HOP × 3356 cpds.
8. **nadal-ribellesSinglecellResolvedGenotypephenotype2025** — GEO (find accession in
   paper.md) — **PSEUDOBULK**: aggregate single cells → per `(genotype × environment)` mean
   expression vector, reuse `RNASeqExpressionPhenotype`. Genome-wide KO × environmental
   (incl osmotic). **Sequenced LAST.** If pseudobulk aggregation is non-trivial → STOP + flag
   for supervised work (do NOT invent a single-cell schema).

## Per-dataset gates
1. **Fit-check** — does it map to `(deletion × EnvironmentPerturbation → phenotype)`? If not, STOP + note.
2. **Source provenance** from paper.md / SI. Unsourceable → STOP, don't guess.
3. **Build loader** — mirror an existing dataset class + the LMDB `experiment_dataset` base.
4. **L0–L4 verify** (`torchcell/verification`); write `verification_report.json`.
5. **Commit** `feat(datasets): add <name> env×geno chemogenomic dataset (L0-L4)`.
6. **Append** pass/fail + notes to `WS15_STATUS.md`.

## Phase 1 (first unit of work)
Design `EnvironmentPerturbation` in `schema.py` on **Vanacloig**; **fit-check** it by reading
paper.md for #2–#7 and confirming their environment shapes map (compound+conc; het/hom CNV;
physical stresses; z-score/categorical readouts) — adjust schema BEFORE building more. Build
Vanacloig loader, run L0–L4, commit, append STATUS. Then STOP and report; the orchestrator
launches #2 next.
