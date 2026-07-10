# WS15 Status — Phase 1 (EnvironmentPerturbation schema + Vanacloig2022)

Date: 2026-07-09 · Branch: `ws15-env-chemogenomic` (worktree only)

## What was done

1. Designed + froze the `EnvironmentPerturbation` ontology in `torchcell/datamodels/schema.py`,
   fit-checked it against papers #2–#7, built the Vanacloig2022 dataset from GEO GSE186866,
   ran L0–L4 verification, added a family verifier + runner, kept the ontology-integrity
   tests green, and ran mypy-strict + ruff on every changed file.
2. New source: `torchcell/datasets/scerevisiae/vanacloig2022.py`,
   `torchcell/verification/environment_response.py`; edits to `schema.py`,
   `datamodels/__init__.py`, `datasets/scerevisiae/__init__.py`,
   `verification/runners.py`, `tests/torchcell/datamodels/test_ontology_invariants.py`.

## Final EnvironmentPerturbation schema shape (FROZEN)

Environment axis (parallel to the gene-perturbation ontology):

```
Concentration{ value:float|None, unit:str|None, basis:str|None }   # value|basis required; value⇒unit
Solvent{ name:str, percent:float|None }
EnvironmentStressType ∈ {temperature, osmotic, oxidative, pH, carbon_source, alcohol}

EnvironmentPerturbation(base){ perturbation_type, description }
  ├─ SmallMoleculePerturbation{ compound_name, compound_id|None, smiles|None,
  │                              concentration:Concentration, solvent:Solvent|None,
  │                              stress_category|None }        # type="small_molecule"
  └─ PhysicalStressPerturbation{ stress_type:EnvironmentStressType,
                                 magnitude:Concentration|None, agent|None }  # type="physical_stress"
EnvironmentPerturbationType = SmallMoleculePerturbation | PhysicalStressPerturbation
```

`Environment` was EXTENDED (backward-compatible; all new fields default so every existing
dataset is unchanged):

```
Environment{ media, temperature,
             perturbations: list[EnvironmentPerturbationType] = [],   # NEW
             aerobicity: str = "aerobic"  (aerobic|anaerobic|microaerobic),  # NEW
             duration_hours: float|None = None }                      # NEW
```

Readout phenotype (signed score; distinct from `FitnessPhenotype`, which clamps ≤0 → 0):

```
MeasurementType ∈ {log2_ratio, z_score, sensitivity_score, categorical, growth_rate}
EnvironmentResponsePhenotype{ measurement_type:MeasurementType,
  environment_response:float|None,          # None only for categorical
  category:str|None,                        # required iff categorical
  environment_response_se|None,             # DERIVED, ML-facing SE
  environment_response_uncertainty|None + _uncertainty_type:UncertaintyType|None,
  n_samples:int|None, sample_unit:SampleUnit|None, units:str|None }
+ EnvironmentResponseExperiment / EnvironmentResponseExperimentReference (type="environment_response")
```

Also added `MarkerDeletionPerturbation(DeletionPerturbation)` (a new concrete leaf on the
EXISTING gene-perturbation ontology) with a `marker` field, to honestly represent
auxotrophic-marker gene replacements (Vanacloig `pdr3::KlURA3`, `snq2::KlLEU2`) rather than
mislabelling them KanMX/NatMX. Registered in `GenePerturbationType`; ontology-integrity
FACTORY updated; all 202 datamodels tests pass.

Constant drug-sensitized background (3ΔAlpha) is modeled with the Ozaydin/Cachera
constant-background pattern: every record's `Genotype` = the library `KanMxDeletion` +
`NatMxDeletion(PDR1/YGL013C)` + `MarkerDeletion(PDR3/YBL005W, KlURA3)` +
`MarkerDeletion(SNQ2/YDR011W, KlLEU2)` (systematic names verified against SGD R64).

## Fit-check verdicts (#2–#7) — schema covers all; two GENOTYPE-side gaps flagged

Environment + readout side of every dataset maps cleanly onto the schema. All facts below
are sourced from each paper's mirrored `paper.md`.

| # | dataset | environment → schema | readout → measurement_type | n | verdict |
|---|---------|----------------------|-----------------------------|---|---------|
| 2 | hoepfner2014 | ~1776 compounds @ IC30 (µM) → SmallMolecule(basis IC30) | MADL sensitivity / z-score → `sensitivity_score` (+`z_score`) | n=2 | FITS (genotype gap) |
| 3 | auesukaree2009 | 10% EtOH,16% MeOH,7% 1-PrOH,1 M NaCl,5 mM H2O2 → SmallMolecule(+stress_category); 37 °C heat → PhysicalStress(temperature,37 Celsius) | sensitive/tolerant call → `categorical` (`category`) | n=3 | FITS |
| 4 | mota2024 | 75 mM acetic / 14 mM butyric / 0.3 mM octanoic acid, YPD pH 4.5 → SmallMolecule(value+unit, stress_category='acid') | 0/+/++ susceptibility → `categorical` | single call | FITS |
| 5 | wildenhain2015 | 4915 compounds @ 20 µM → SmallMolecule(value=20,unit=uM) | normalized-OD Z-score → `z_score` | n=2 | FITS |
| 6 | hillenmeyer2008 | 1144 conditions (small molecules + env stresses); dose in SOM → SmallMolecule/PhysicalStress | growth-defect sensitivity score → `sensitivity_score` | not in main text | FITS (genotype gap) |
| 7 | lee2014 | 3250 compounds (paper says 3250, NOT plan's 3356) @ ~IC20 → SmallMolecule(basis) | fitness-defect (FD) score → `sensitivity_score` | not in main text | FITS (genotype gap) |

No schema ADJUSTMENTS were needed after fit-check — the two-leaf `EnvironmentPerturbation`
union + `EnvironmentResponsePhenotype` (all 5 measurement types exercised: log2_ratio,
sensitivity_score, z_score, categorical, growth_rate) covered every environment shape and
readout convention encountered.

### Flagged GENOTYPE-side gap (NOT an EnvironmentPerturbation issue) — needs a decision before #2/#6/#7

Hoepfner/Hillenmeyer/Lee are HIP/HOP: HIP = a DIPLOID with ONE of two copies deleted
(reduced dosage, essential genes), HOP = homozygous deletion. The current gene-perturbation
ontology has no ENGINEERED heterozygous-deletion / dosage leaf (`CopyNumberVariantPerturbation`
is `provenance="natural"`). Building #2/#6/#7 will need either a new engineered
heterozygous-deletion leaf or an engineered-CNV variant. This is orthogonal to the
EnvironmentPerturbation schema (which is frozen and complete) and does not block it.

## Vanacloig build + verification

- Raw source: GEO **GSE186866** `GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz`
  (scriptable FTP; sha256 `e29eb027…4a85a`, pinned + verified on download). This is the
  ONLY scriptable source — the OUP processed Table S1 / Dataset2 log2 matrix is 403
  (Silverchair auth) and R/edgeR is unavailable, so the paper's exact edgeR logFC could
  not be obtained OR reproduced.
- Readout recomputed deterministically from the raw counts (documented in the loader
  docstring, `units`, and provenance): per-sample CPM, then per gene
  `log2((CPM_compound_rep+1)/(CPM_pooled_control_mean+1))` for each of 3 biological
  replicates; stored response = mean of the 3, uncertainty = sample SD (SE = SD/√3),
  n_samples=3, sample_unit=biological_replicate. This is the paper's DEFINED quantity
  ("log2 of the normalized read counts for inhibitor/control ratio") reproduced from the
  canonical raw artifact — it is NOT the published edgeR logFC and is labelled as such.
- Scale: 3647 ORF barcodes × 45 compound columns = **164,115 records**. 4 barcode rows
  dropped: 2 all-NaN QC rows (DUR1, ADE5) + 2 that ARE background genes (PDR3/YBL005W,
  SNQ2/YDR011W — a gene already deleted in the 3ΔAlpha background cannot be an independent
  screened deletion). 16 pooled `ControlN` columns; anaerobic, 30 °C, 48 h.
- **L0–L4: PASS** (L0 164,115 validated; L1 count=164,115 + exact (ORF,compound) pair
  uniqueness; L2 signed-response finiteness + SE≥0; L3 single measurement_type +
  reference_zero + environment_perturbed; L4 0.988 of 3647 screened genes are S288C
  reference genes ≥ 0.90). Report at
  `$DATA_ROOT/data/torchcell/env_chemgen_vanacloig2022/preprocess/verification_report.json`
  (sibling of `experiment_reference_index.json`).
- Tests: 278 passed (datamodels + verification), incl. the 8 ontology-integrity invariants.
  Also fixed one PRE-EXISTING stale verification fixture (`test_verification.py`
  `_fitness_record` used `perturbation_type:"deletion"`, an abstract base since the
  933ee5ef ontology refactor → now `sga_kanmx_deletion`). mypy-strict + ruff: clean on all
  changed files.

## Blockers / decisions needed before #2

1. **HIP/HOP genotype dosage** (above) — decide engineered-het-deletion representation
   before hoepfner(#2)/hillenmeyer(#6)/lee(#7).
2. **Un-scriptable processed supplements are the norm** for OUP/Science SIs (Vanacloig
   Table S1, Lee/Hillenmeyer SOM). Downstream datasets will similarly need either a
   manual-deposit-once mirror or (like Vanacloig) recomputation from a raw deposit —
   confirm this recompute-from-raw policy is acceptable, since recomputed readouts are not
   byte-identical to published processed values.
3. **Per-compound concentrations / DMSO pairing** for Vanacloig live in the un-scriptable
   Table S1; `concentration` currently carries only `basis="IC30"` and the control is the
   pooled mean of all `ControlN` columns. If exact IC30 molar values / paired controls are
   wanted, Table S1 must be manually deposited into the library mirror.
4. **Compound-count reconciliations** for later builds: lee = 3250 (not 3356);
   wildenhain = 4915 @ 20 µM; note in each loader.
