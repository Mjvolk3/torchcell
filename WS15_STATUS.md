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

## 2026.07.10 — Dataset #4: Mota2024 acetic/butyric/octanoic acid chemogenomic (L0–L4 PASS)

Built `torchcell/datasets/scerevisiae/mota2024.py` (`EnvChemgenMota2024Dataset`,
`env_chemgen_mota2024`) on the FROZEN EnvironmentPerturbation schema — no schema change
needed; the categorical readout mapped cleanly (`measurement_type=categorical`, `category`).

**Records: 1273** — acetic 373, butyric 416, octanoic 484. One categorical record per
`(single KanMX deletion in BY4741) × SmallMoleculePerturbation(acid) →
EnvironmentResponsePhenotype(category)`. 3 unique references (one per acid).

**Source (all scriptable + sha256-pinned):** BMC open-access supplementary spreadsheets,
`static-content.springer.com/esm/art%3A10.1186%2Fs12934-024-02309-0/MediaObjects/12934_2024_2309_MOESM{1,2,3}_ESM.xlsx`
(Additional files 1/2/3 = Tables S1/S2/S3 = acetic/butyric/octanoic). sha256:
MOESM1 `b23ad281…205e42`, MOESM2 `a7a1aaee…a1e81f`, MOESM3 `27f15086…f5d455`. Downloaded +
verified in the loader `download()`. Parsed susceptible-row counts match the paper's
headline totals EXACTLY: acetic 377 (46 ++, 331 +), butyric 422 (51 ++, 371 +), octanoic
490 (53 ++, 437 +).

**Sourced provenance (verbatim quotes, `paper.md`):**

- **Background (confirmed haploid, NOT HIP/HOP):** *"The haploid parental strain S.
  cerevisiae BY4741 (MATa, his3∆1, leu2∆0, met15∆0, ura3∆0) and the collection of derived
  single deletion mutants, obtained from Euroscarf … were used for the chemogenomic
  analysis."* (Methods, "Strains and growth conditions", line 197). BY4741 auxotrophies are
  the standard collection background → captured by `ReferenceGenome(strain="BY4741")`, NOT
  modeled as perturbations (matches Costanzo/Kuzmin convention); genotype = one
  `KanMxDeletionPerturbation`. No constant-background genes (unlike Vanacloig 3ΔAlpha).
- **Conditions / concentrations + pH:** *"Equivalent mild growth inhibitory concentrations
  … by the supplementation of YPD solid medium at pH 4.5 with 75 mM (4.58 g/L) of acetic
  acid, or 14 mM (1.23 g/L) of butyric acid, or 0.3 mM (0.04 g/L) of octanoic acid … These
  acid concentrations were used for the planned genome-wide analysis."* (Results, line 39);
  30 °C, YPD solid, pH 4.5 (Methods "Genome-wide search…", line 227). Modeled as
  `SmallMoleculePerturbation(compound_name, Concentration(value, unit="mM"),
  stress_category="acid")`, `Media(name="YPD, pH 4.5", state="solid")`, 30 °C, aerobic,
  48 h. (Media schema has no pH field → pH encoded in the media name.)
- **Readout (categorical):** *"the susceptibility phenotype of each single deletion mutant
  was scored, after 48 h as (+) if the mutant strain showed, compared with the parental
  strain, a slight to moderate growth inhibition, and (++) if no growth was observed"*
  (Table S1/S2/S3 captions + Methods line 229); *"'0' corresponds to an absence of a
  detectable susceptibility phenotype"* (Fig. S2, line 271). Category strings:
  `+`→`minor_to_moderate_growth_inhibition`, `++`→`total_growth_inhibition`, reference
  (parental)→`no_detectable_susceptibility`.
- **n_samples — NOT SOURCEABLE → `None` (flagged, not guessed):** the genome-wide spot
  screen Methods give NO replicate count (*"Photographs were taken after 24 h of incubation
  for control plates (YPD medium) or 36–48 h in the presence of the acids"*, line 227). The
  *"at least three independent experiments"* statements apply ONLY to the CFU-viability
  (line 209) and intracellular-pH (line 221) PHYSIOLOGICAL assays, NOT the disruptome
  screen. So `n_samples=None`, `sample_unit=None`, no uncertainty/SE (categorical readout
  carries no SE anyway). Documented in loader + this note.

**Source quirks handled deterministically (no guessing):**

- **RNR4 (YGR180C) listed twice** in every table (a source duplicate the paper's totals
  count) — acetic/butyric both `+`, octanoic conflicting `+`/`++`. Deduped per (resolved
  ORF, acid) keeping the MORE SEVERE score → RNR4 = one record/acid (octanoic → `++`).
- **6 gene tokens unresolvable to an SGD R64 systematic ORF** (genome alias table) →
  DROPPED, never guessed (13 records total): REF1, RLM2, SBR2 (all 3 acids), ILM2
  (butyric), VPS236 (butyric+octanoic), SIW15 (octanoic). Logged per-acid in `process()`.
  Likely SI typos (VPS236→VPS36? SIW15→SIW14? ILM2→ILM1? RLM2→RLM1? REF1/SBR2 unclear) —
  **flagged follow-up**; resolving them would recover ≤13 records. This is why final counts
  are acetic 373 / butyric 416 / octanoic 484 (raw 377/422/490 − 1 RNR4 dup − unresolvable).

**L0–L4: PASS** (report `…/env_chemgen_mota2024/preprocess/verification_report.json`, sibling
of `experiment_reference_index.json`): L0 1273 validated; L1 count=1273 + 1273 unique
(ORF, compound) pairs; L2 value/SE fidelity (0 numeric values — categorical); L3 single
measurement_type=categorical + reference_zero (n=0, categorical refs) + all 1273 env-perturbed;
L4 0.993 of 601 measured genes are S288C reference genes (≥0.90). Runner entry added to
`ENVIRONMENT_RESPONSE_DATASETS` (`background_genes=frozenset()`, `expected_count=1273`).
Tests: 278 passed (datamodels + verification, incl. 8 ontology-integrity invariants).
mypy-strict + ruff clean on all changed files (added `openpyxl` to pyproject untyped-module
mypy override).

**No blockers for Mota** — self-contained, fully scriptable/sourced. The only flagged item
is the 6 unresolvable gene tokens (≤13 records), left for a supervised typo-resolution pass.

## 2026.07.10 — Dataset #5: Wildenhain2015 chemical-genetic matrix (CGM) (L0–L4 PASS)

Built `torchcell/datasets/scerevisiae/wildenhain2015.py`
(`EnvChemgenWildenhain2015Dataset`, `env_chemgen_wildenhain2015`) on the FROZEN
EnvironmentPerturbation schema — **no schema change needed**; the numeric z-score readout
mapped cleanly (`measurement_type=z_score`). Scope: ONLY the chemical-genetic matrix
(strain × compound → z-score); the 128×128 cryptagen chemical-chemical synergy layer is out
of scope as instructed.

**Records: 428,573** — one cell per `(haploid single KanMX deletion in BY4741) ×
SmallMoleculePerturbation(compound @ 20 µM in DMSO) → EnvironmentResponsePhenotype(z_score)`.
Covers **242 distinct systematic ORFs** (the released data spans more strains than the
195-strain "sentinel panel" headline — extra strains screened across the 4 libraries; all
242 are valid SGD R64 genes, L4 containment = 1.000). Reference = parent BY4741 in the same
compound environment, control z-score = 0.

**Data source (structured, scriptable, sha256-pinned, byte-stable):** PubChem BioAssay
**AID 1159580** (the paper's ACCESSION NUMBER). chemgrid.org/cgm is an interactive PHP portal
with NO bulk-downloadable processed matrix, so the PubChem datapoint export is the canonical
scriptable full-data artifact. Pulled from the byte-stable NCBI FTP range archive
`ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/1159001_1160000.zip`, member
`1159001_1160000/1159580.csv.gz` (fixed 2022-12-16 mtime), **sha256
`c461c679b63ac56045cef0f03ed9bcbb8e7f9c12146f1fc7cc8ac0c113188d64`** (verified on extract).
Confirmed **bit-identical** to the PUG-REST `/assay/aid/1159580/CSV` export (492,126
datapoints). Each row carries `PUBCHEM_CID` + `PUBCHEM_EXT_DATASOURCE_SMILES` →
**compound→CID mapping 99.9%** (428,206 of 428,573 records carry a `CID:` CURIE; 367 lack a
CID and fall back to the PubChem SID). SMILES stored per record.

**Sourced provenance (verbatim quotes):**

- **195 strains, 4,915 compounds, 20 µM, z-score averaged over the duplicate screens
  (n_samples=2):** *"we screened a total of 4,915 unique compounds derived from four
  different chemical libraries against a panel of 195 non-essential deletion strains"* and
  *"We carried out over 600 growth-based screens in duplicate at a compound concentration of
  20 µM … Z scores were calculated and averaged for the replicate screens."* (`paper.md`
  RESULTS "Generation of a Chemical-Genetic Matrix", lines 68 + 70).
- **Background (confirmed haploid, isogenic BY4741, Euroscarf — NOT HIP/HOP):** *"S.
  cerevisiae deletion strains were obtained from the Euroscarf deletion set and are isogenic
  to BY4741"* (`paper.md` EXPERIMENTAL PROCEDURES, line 217). A sentinel = an ordinary
  single-gene deletion strain ⇒ one `KanMxDeletionPerturbation` in the `ReferenceGenome`
  (BY4741) background; the standard KanMX Euroscarf MATa collection (Mota precedent). No
  constant-background genes.
- **Medium / temperature / duration / solvent / duplicate (PubChem AID 1159580 protocol):**
  *"All strains were grown and screened in synthetic complete (SC) medium with 2% glucose."*;
  *"final concentration of 20 M. Screens were conducted in technical duplicate … DMSO solvent
  only controls … incubated at 30 C without shaking for approximately 18 h … reading OD600"*;
  BioAssay column defs *"Raw OD read 1 = first replicate"*, *"Raw OD read 2 = second
  replicate"*, *"Z_score = … per screen"*. Modeled: `Media("synthetic complete (SC), 2%
  glucose", state="liquid")`, 30 °C, aerobic, 18 h, `Solvent("DMSO")`,
  `Concentration(value=20, unit="uM")`. **`n_samples=2` (the two duplicate replicate screens
  read1/read2), `sample_unit=technical_replicate`.**

**n_samples nuance (documented, more-rigorous-than-source):** read1/read2 ARE the two
duplicate replicate screens underlying each released z (PubChem "first/second replicate" +
paper "600 screens in duplicate … averaged for the replicate screens"). A given
(strain, compound) can additionally recur across the four libraries: **46,195 cells (11%)
had 2–3 released screen rows** with distinct z-scores. These are collapsed to one matrix cell
by **averaging** the per-screen z-scores — exactly the paper's stated matrix construction
("Z scores … averaged for the replicate screens") and the Vanacloig recompute-from-canonical
precedent — with `n_samples = 2 × (number of screen rows)` to count all contributing
duplicate-screen reads (2 for the 89% single-screen cells, 4/6 for the repeats). No per-record
SE is released for the z (so `environment_response_se=None`; L2 se_nonnegative trivially
passes). 7,296 `NA`/`NULL` non-strain control rows dropped; only systematic-ORF strains enter
the matrix. Compound identity = CID when present else SID; `compound_name` carries this
identifier because the structured artifact provides **no human compound name** (chemgrid
regid→CID join not available in the pinned export).

**L0–L4: PASS** (report at
`…/env_chemgen_wildenhain2015/preprocess/verification_report.json`, sibling of
`experiment_reference_index.json`): L0 428,573 validated; L1 count=428,573 + 428,573 unique
(ORF, compound) pairs; L2 value-fidelity (428,573 finite signed z's) + se_nonnegative (0
values); L3 single measurement_type=z_score + reference_zero (all references z=0) + all
428,573 env-perturbed; L4 1.000 of 242 measured genes are S288C reference genes (≥0.90).
Runner entry added to `ENVIRONMENT_RESPONSE_DATASETS` (`background_genes=frozenset()`,
`expected_count=428573`). Tests: 215 passed (datamodels + ontology-integrity invariants).
mypy-strict + ruff clean on all changed files.

**No blockers for Wildenhain** — fully scriptable + sha256-pinned. Flagged (non-blocking):
(1) released z-score is a point value with no released SE; (2) compound human names not in the
structured artifact (CID/SID/SMILES only); (3) the multi-library z-averaging is a documented
deterministic reconstruction, not a byte-verbatim released cell value.

## 2026.07.10 - Schema extension: EngineeredCopyNumberPerturbation + ReferenceGenome.ploidy (HIP/HOP support)

Schema-only step (no dataset build) to unblock the HIP/HOP chemogenomic datasets
(Hoepfner / FitDb / Lee), which use **heterozygous** deletions in a **diploid**.

### New leaf: `EngineeredCopyNumberPerturbation` (`torchcell/datamodels/schema.py`)

The ENGINEERED counterpart of the natural `CopyNumberVariantPerturbation` — an engineered
copy-number/dosage change of a **present** native gene.

- **Parent = `GenePerturbation` directly** (NOT a new intermediate). Justification: the
  dosage axis currently has no ABC — the natural CNV leaf also hangs directly off
  `GenePerturbation` (only AXIS-1 presence/absence and AXIS-3 sequence have ABCs). Mirroring
  that pattern is the minimal, ontology-consistent choice: it reparents nothing (natural CNV
  leaf + Caudal untouched), keeps the DAG single-rooted + acyclic, and preserves discriminator
  uniqueness. Introducing a shared `CopyNumberPerturbation` intermediate would not make any
  invariant cleaner and would add structure against the "minimal new structure" rule.
- **Fields:** `copy_number: float` (engineered target copies), `reference_copy_number: float`
  (copies in the reference = ploidy for an autosomal gene), `marker: str | None = None`
  (optional selection cassette, e.g. `"KanMX"`), `state="present"`,
  `mechanism_so_id="SO:0001019"`, `mechanism_so_name="copy_number_variation"`,
  `provenance="engineered"`, `perturbation_type: Literal["engineered_copy_number"]`.
- Uses the **real S288C systematic-name validator** (inherited from `GenePerturbation`, NOT
  relaxed — these are reference genes, not pangenome ORFs). Does NOT carry
  `strain_id`/`pangenome_orf_id` (those are natural-isolate fields).
- **Semantics:** HIP heterozygous deletion = `EngineeredCopyNumberPerturbation(copy_number=1,
  reference_copy_number=2, marker="KanMX")`. HOP homozygous deletion stays the existing
  absence leaf (`KanMxDeletionPerturbation`, `state="absent"`).
- Wired into `GenePerturbationType` (the discriminated union IS the perturbation registry;
  there is no separate perturbation `*_TYPE_MAP` — the experiment TYPE_MAPs are unrelated).

### Ploidy: `ReferenceGenome.ploidy` (`torchcell/datamodels/schema.py`)

Added `ploidy: Literal["haploid","diploid"] = "haploid"` to `ReferenceGenome` (the genome-wide
baseline belongs on the genome, not a perturbation). Default `"haploid"` keeps every existing
dataset valid/backward-compatible. Per-locus `copy_number` is the deviation from this baseline
(diploid autosomal gene 2 → 1 for a HIP heterozygous deletion).

### Tests (`tests/torchcell/datamodels/test_ontology_invariants.py`)

The generic parametrized invariants auto-cover the new leaf once it is added to `FACTORY`
(discriminator uniqueness, DAGness/single-root, Liskov, union↔leaf, round-trip serialize/JSON,
SO well-formedness, provenance ∈ {engineered,natural}, identity, factory↔leaves lockstep).
Added focused unit tests: engineered-CNV defaults + discriminator, real-validator rejection of a
pangenome id, optional marker, ploidy default/values/rejection, and HIP-style (diploid +
copy 1/2) + HOP-style (diploid + KanMX deletion) round-trips.

**Results:** `tests/torchcell/datamodels/ + tests/torchcell/verification/` → **292 passed**
(includes all 8 ontology-integrity invariants). mypy-strict + ruff clean on both changed files.
Existing built datasets (Vanacloig/Mota/Wildenhain and all other `ReferenceGenome(species=…,
strain=…)` callsites) still instantiate unchanged via the `ploidy="haploid"` default.
