---
id: 5un38ft4fztmugwycjepevs
title: Uncertainty Statistics Ontology
desc: ''
updated: 1783043936218
created: 1783043936218
---

## 2026.07.02 - Design decision (awaiting owner sign-off)

Core-schema/ontology decision for how phenotype uncertainty is stored. Prompted
by a real bug on `main`: `costanzo2016.py` computes `fitness_se = fitness_std /
sqrt(n_samples)` uniformly, which is statistically wrong. Analysis lives in
[[torchcell.datasets.scerevisiae.costanzo2016.noise-computation]].

### The problem: one field conflates two statistical objects

Costanzo publishes one "std" per fitness value, but it means different things:

- **SMF (single-mutant):** a **bootstrap SD of the mean** -> it is ALREADY an SE
  (spread across 17 query / 350 array screens). Dividing again by `sqrt(n)`
  corrupts a valid SE; and `n` should be screens (17/350), not colonies (68/1400).
- **DMF (double-mutant):** a **raw sample SD** over 4 colonies -> `SD/sqrt(4)` IS
  the correct SE, but only with `n = 4`.

So `fitness_std` is a category error (stores a bootstrap-SE and a raw-SD under one
name) and any uniform `std/sqrt(n)` derivation is wrong for one of them. This is
why published p-values did not reproduce (an L2/L3 verification failure).

### Verdict on naming

Calling the current derived number `fitness_se` is technically wrong (it is not a
valid SE). The fix is NOT a rename -- it is to separate three concerns currently
mashed into two fields:

1. What the source REPORTED (value + its statistical KIND) -- provenance.
2. The replicate DESIGN (`n` + what `n` counts) -- 17 screens != 68 colonies.
3. The ML-facing DERIVED SE (computed correctly per kind) -- downstream utility.

### Proposed ontology (generalizes to every phenotype, not just fitness)

```python
class UncertaintyKind(str, Enum):
    sample_sd = "sample_sd"            # raw SD of observations -> SE = sd/sqrt(n)
    standard_error = "standard_error"  # already SE of the mean -> use as-is
    bootstrap_se = "bootstrap_se"      # bootstrap SD of the mean ~ SE -> use as-is
    variance = "variance"              # SE = sqrt(var/n)
    ci95 = "ci95"                      # half-width -> SE = hw / 1.96
    unknown = "unknown"                # -> derived SE is None (never fabricate)

class ReplicateUnit(str, Enum):
    colony = "colony"
    screen = "screen"
    biological = "biological"
    technical = "technical"
    unknown = "unknown"

# on FitnessPhenotype (pattern repeats for expression/metabolite/morphology):
fitness: float
fitness_reported: float | None            # uncertainty number verbatim from source
fitness_reported_kind: UncertaintyKind
n_replicates: int | None
n_replicates_unit: ReplicateUnit
fitness_se: float | None                  # DERIVED, stored (Neo4j-queryable)
```

Single documented derivation rule (`fitness_se` becomes an HONEST SE):

| `fitness_reported_kind`        | `fitness_se` =                       |
|--------------------------------|--------------------------------------|
| `standard_error`/`bootstrap_se`| `fitness_reported` (no division)     |
| `sample_sd`                    | `fitness_reported / sqrt(n)`         |
| `variance`                     | `sqrt(fitness_reported / n)`         |
| `ci95`                         | `fitness_reported / 1.96`            |
| `unknown`                      | `None`                               |

`label_statistic_name` stays `"fitness_se"` -- the model still trains on SE /
inverse-variance weighting, but the value is now correct and backed by the honest
reported value + declared kind.

### Costanzo mapping (once implemented)

| Measurement | `fitness_reported_kind` | `n_replicates` | `n_replicates_unit` |
|-------------|-------------------------|----------------|---------------------|
| Query SMF   | `bootstrap_se`          | 17             | `screen`            |
| Array SMF   | `bootstrap_se`          | 350            | `screen`            |
| DMF         | `sample_sd`             | 4              | `colony`            |

### Why this altitude

- Shared ontology, not a Costanzo patch: every phenotype with uncertainty
  (expression, metabolite, morphology) answers "what kind, over what unit".
- Plugs into the merged L0-L4 framework (`torchcell/verification/`): an L3 check
  asserts `fitness_se` matches its declared kind and that `n_replicates_unit` is
  set -- catching exactly the class of error that broke the p-values.
- All fields are stored `model_fields` -> Neo4j-queryable (roadmap decision 7):
  audit "which values have `kind = unknown`?" across the KG.

## 2026.07.02 - Decisions after owner discussion

Refinements agreed in discussion (pending final "go"):

- **Field set: FULL.** Semantics must be explicit/queryable/enforceable, since the
  bug arose from implicit semantics.
- **Count name: `n_samples` (general) + `sample_unit` enum**, NOT `n_replicates`.
  Rationale: the count is a statistical sample size; "replicate" over-claims a
  designed repeat, but Costanzo SMF averages over control *screens* that are a
  reference distribution, not designed replicates. `sample_unit` carries the
  specificity: `biological_replicate | technical_replicate | screen | colony |
  pooled | ...` -- "replicate" is one unit family, not the default. **Rename main's
  expression `n_replicates` (dict) -> `n_samples` (dict)** for one concept schema-wide.
- **NO `unknown` kind (strict labelling).** Invariant: `reported is not None`
  IFF `kind` is set. "Not reported" is a legitimate `None`; "reported but kind
  unsure" is forbidden -- do not ingest it. So `UncertaintyKind` drops `unknown`.
- **Scope: current phenotypes only.** `FitnessPhenotype` +
  `MicroarrayExpressionPhenotype` get the full treatment now.
  `CalMorphPhenotype` (coefficient_of_variation -- a different, already-normalized
  representation) and `GeneInteractionPhenotype` (p-value -- a test result, not a
  precision-of-mean statistic) are NOT restructured; CV may join as a kind later.
  Essentiality/lethality/rescue untouched. Yeast9/small-molecule = later (the
  metabolite phenotype is not on main yet).
- **Acceptance test:** reproduce Costanzo's published gene-interaction p-values.
  The interaction p-value is downstream of the component fitness SEs
  (Var(eps) propagated from f_i, f_j, f_ij SEs); once each SE is derived correctly
  per kind, the p-values should replicate. This L2/L4 check is the definition of done.

Revised `UncertaintyKind` (no `unknown`): `sample_sd | standard_error |
bootstrap_se | variance | ci95`.

### Files touched when implemented

- `torchcell/datamodels/schema.py` (enums + FitnessPhenotype fields + derivation)
- `torchcell/datasets/scerevisiae/costanzo2016.py` (map SMF/DMF -> kind/unit; stop
  the naive `std/sqrt(68|1400)`)
- `torchcell/datasets/scerevisiae/{kuzmin2018,kuzmin2020}.py` (once SI n sourced)
- `torchcell/verification/levels.py` (add an L3 SE-consistency check)
- `biocypher/config/torchcell_schema_config.yaml` (expose new fields, Phase B)
