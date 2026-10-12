---
id: grwratc4jmft2xpchtxiyyt
title: Thermo_curation
desc: ''
updated: 1789167211225
created: 1789167211225
---

## 2026.09.11 - The typed curation record, and the stage that prices from it

### Why the module exists

`torchcell/metabolism/thermo_pipeline.py` prices what an automatic route can reach. What is
left is not a coding problem: a species with no structure in any namespace needs a person to
find one, a protein-bound redox carrier has a midpoint potential rather than a formation
energy, and a GEM formula that is impossible needs a person to say what the right one is.
Each is a curation act, and a curation act with no provenance is a guess with a number
attached. This module is the typed record of one, the loader that validates a directory of
them, and the stage that turns the admitted ones into prices. Nothing in it names an
organism, a metabolite or a file.

### Admission

`Admission` is `measured | tabulated | computed | provisional | none`, and only the first
two enter the table a model reads (`ADMITTED`). The rest are loaded, validated and reported
under `UnpricedReason.RECORDED_NOT_ADMITTED`, and never priced. Keeping them is what stops
the next person repeating the search; not admitting them is what stops a computed value
becoming indistinguishable from a measured one once it is a float in a parquet.

### The record

`CuratedThermoRecord` is strict at the top level and tolerant inside `value`. The top level
is the contract between the curator and the pipeline, so an unexpected key there raises;
`value` is where a curator writes whatever the chemistry needed, so it allows extras and its
shape follows the record's own `kind` through `VALUE_MODEL_BY_KIND`. `ProvenanceSource`
requires all seven provenance keys and permits them to be empty, which is the difference
between a stated gap and an unnoticed one.

### The routes

| kind | what happens | route |
|---|---|---|
| `structure`, `polymer_convention` | the curated SMILES goes through the same structure route as any other | `curated_structure` |
| `pka_ladder` | the ladder is handed to the compound bridge with the structure | `curated_pka` |
| `formation_energy` | Legendre-transformed to each compartment by `legendre_transform_kj` | `curated_formation` |
| `formula_correction` | `apply_formula_corrections` rewrites the identity before any composition check; prices only when a structure is also supplied | `curated_formula` |
| `carrier_potential`, `reaction_energy` | `CuratedReactionPricer` prices the REACTION; the species stay unpriced under `PRICED_AT_REACTION_LEVEL` | `carrier_potential` |

A record may also carry the curator's own transformed value per metabolite
(`CuratedPrice`), which is what a polymer residue with attachment points needs, and its
stated conditions are checked against the pipeline's before it is used.

### Two environments, one handshake

Component contribution lives in `equilibrator-ext` and cobra does not, so the GEM driver and
the compound bridge cannot share a process. `StructurePricer` is the seam:
`pricing_requests` derives every (structure, ladder, conditions) call an admitted record
implies, `EquilibratorStructurePricer` runs them under the other environment, and
`PricedStructureTable` replays the artifact. A pairing the artifact does not hold raises,
and the driver refuses an artifact whose `requests_digest` is not the one its own request
list hashes to.

### What the resolver refuses to do

- price a structure whose heavy-atom composition disagrees with the GEM's, corrected,
  formula, which would be pricing a different molecule;
- price a structure carrying an attachment point, unless the curator measured that the
  decomposer refuses it, in which case the reason is the decomposition failure and not
  `PLACEHOLDER_STRUCTURE`, which would claim nothing can ever price it;
- create a compound for a molecule with ionizable centers and no ladder;
- price a carrier reaction whose direction or partner potential is not determined.

Every refusal carries an `UnpricedReason` and a detail naming the record.

### Files

- `torchcell/metabolism/thermo_curation.py`
- `tests/torchcell/metabolism/test_thermo_curation.py`, 40 tests
- `experiments/026-metabolism-flux/scripts/thermo_curation_price.py`
- Findings from the first run: [[experiments.026-metabolism-flux.thermo-curation-closure]]

## 2026.09.11 - Round two: the carrier pair, the conserved moiety, and a filter that can unprice

### The carrier is a pair of pseudo formation energies

`CuratedResolver._price_carrier` fixes the oxidized form at zero and the reduced form at
`-nFE'0`. Only the difference is physical, and it cancels the arbitrary gauge in any
reaction that conserves the carrier, so the ordinary stoichiometric sum prices every redox
step the carrier takes part in and the partner couple prices itself through its own
metabolites. No `partner_couple_e_prime_mv` and no electron chemical potential.
`e_prime_slope_mv_per_ph` moves the potential to the compartment's pH when the record
measured a slope, and nothing moves it when the record did not.

The gauge is why `carrier_conservation_violations` exists in the pipeline: a reaction that
MAKES or destroys the carrier is refused under `UnpricedReason.CARRIER_NOT_CONSERVED`
rather than priced with a constant in it.

### `PriceFilter`: the one thing a stage cannot do

A resolver is offered only the metabolites still unpriced, so it can add a value and never
withdraw one. Withdrawing one is a different act with a different justification: the value
exists and is wrong, because the structure behind it is a different molecule.
`run_thermo_pipeline(price_filters=...)` runs filters after every stage and before the
reactions are assembled, and raises if a filter changes which metabolites the result
covers. `thermo_pipeline_yeast9.CompositionGuard` is the first one.

### Estimability is a first-class outcome

`SIGMA_INF_TOLERANCE` and `UnpricedReason.GROUP_IN_NULL_SPACE`. A structure can decompose
completely and still be unpriceable, because one of its groups is in the null space of the
component-contribution training set and its coefficient has infinite variance. That is a
different fact from a missing group definition, and it now has its own reason, its own
ladder class, and the group named in the detail. Two species reached it in the first run:
nitroethane and 4-nitrophenolate through `-NO2`, and chitobiosyldiphosphodolichol through
the fully protonated diphosphate diester.

### Guards the compound cache needed

- a group vector that EXISTS is not a decomposition: the test is a nonzero group COUNT,
  because both the empty list and an all-zero vector occur on cache rows;
- a cache row whose heavy-atom composition disagrees with the GEM's formula is UNTRUSTED
  and is not used as a structure, which is what keeps the iron-free heme b row out of the
  conserved-moiety route;
- a curated SMILES reaches the decomposer verbatim, never through an RDKit round trip,
  which would rewrite a pentavalent nitro group back into the charge-separated spelling
  eQuilibrator's SMARTS rejects.

### Conserved moiety

`MoietyReactionTable` replays what the other environment computed;
`moiety_requests` derives the request list from a first pass, so the definition of
"blocked" is the pipeline's own. The compute half lives in
`experiments/026-metabolism-flux/scripts/thermo_curation_price.py price_moieties` and is
described in [[experiments.026-metabolism-flux.thermo-curation-closure]].
