---
id: 51tv9h6cjyevpj0g8ietfjz
title: Thermo_pipeline
desc: ''
updated: 1789101215559
created: 1789101215559
---

## 2026.09.11 - An organism-agnostic thermodynamic pipeline

### Why the module exists

Every thermodynamic number the flux layer consumes was produced by a chain of yeast-GEM
scripts under `experiments/026-metabolism-flux/scripts/`. That chain works, and it is not
portable: it hardcodes yeast-GEM's file layout, its compartment letters, its shipped
`model_metDeltaG.csv` and a lipid-name grammar only yeast-GEM writes. A new organism, which
is the point of the enzyme-constrained layer, would have to reimplement all of it.

This module is that chain with the organism removed. It takes

- a GEM as any cobra model, whose metabolites carry whatever identifiers they carry
  (InChI, SMILES, KEGG, MetaNetX, ChEBI, SEED) plus a formula and a charge,
- a compartment table of pH, ionic strength and pMg,
- an ORDERED list of resolver stages,

and returns one `ThermoPipelineResult`: a formation energy per metabolite with its route, a
`delta_r G'^\circ` per reaction with a propagated sigma, and for every reaction it cannot
price a reason from a closed enum plus the metabolites that blocked it.

### The four rules that make the output readable

1. **A reaction is priced only when EVERY participant carries a formation energy.** One
   unpriced participant makes the sum a number that looks like a free energy and is not
   one.
2. **Every value carries its route, and routes are never averaged.** A reaction assembled
   from two routes is tagged `mixed`, because its uncertainty is not either route's.
3. **Every failure carries an `UnpricedReason`, not a blank.** "No structure exists" and "a
   structure exists and the decomposer refused it" are the difference between work that is
   possible and work that is not, and a coverage percentage hides it.
4. **No stage may silently substitute a value.** A composition surrogate is a route of its
   own, admitted only under an explicit spread bound, and carries that spread as its sigma.

### What is organism-specific, and where it lives

Nothing in the module names a species, a compartment letter, a metabolite or a file. Three
things enter as data or as a plugged-in stage:

- compartment conditions, as `CompartmentConditions`; a compartment the table does not name
  is priced at the stated defaults AND listed in
  `ThermoPipelineResult.compartments_at_default`, so a missing organelle is a reported
  assumption rather than a silent one;
- a GEM's own shipped table, through `TableResolver`, which is handed a plain
  `{met_id: value}` mapping and the unit it is written in, and which supports a
  per-entity exemption list for a table that MIXES units (yeast-GEM needs that: 33 of its
  3,210 reaction rows are kJ/mol inside a kcal/mol column);
- a structure writer for species whose structure is implied by their NAME, through the
  `StructureWriter` protocol. yeast-GEM's lipid grammar is one implementation and lives in
  the experiment folder.

Resolvers that need eQuilibrator import it inside their methods, so the module imports in
the plain torchcell environment where `equilibrator_api` is absent.

### Precedence is the stage order and nothing else

`resolve_metabolites` runs the stages in order, each on what is still unpriced, and keeps
the first hit. A later stage never overwrites an earlier one, and a metabolite priced by
stage 1 is not even offered to stage 2. That is what makes "database first, prediction
second" a property of the configuration rather than of the code. A stage that answers for a
metabolite it was not handed raises.

One exception, and it was found by a test: a structural fact about the MODEL outranks a
stage's report about itself. A biomass pool that a structure writer declined is not "no
structure available", it is not a molecule, and a ceiling counted the first way would put
it on the work queue forever. `PSEUDO_METABOLITE` and `PLACEHOLDER_FORMULA` therefore win
over whatever a stage said.

### A new organism needs no code when it has a table

`python -m torchcell.metabolism.thermo_pipeline` is the generic entry point. It takes the
GEM, a JSON of `CompartmentConditions`, one or more per-metabolite tables in PRECEDENCE
order (`--metabolite-table PATH[:VALUE_COL[:SIGMA_COL[:ROUTE[:UNITS]]]]`, CSV or parquet),
an optional reaction-level last-resort table with its units and its per-row unit
exemptions, an optional flux-carrying reaction list and an optional pseudo-metabolite list,
and it writes the full result. `FileTableResolver` is the stage it builds from each table.

Verified against yeast-GEM 9.0.2 with one table and no driver code: 2,528 of 2,806
metabolites and 3,732 of 4,131 reactions, which is the staged run minus the 6 reactions the
shipped reaction column fills, and it lists the ten compartments it priced at the default
conditions because the conditions file named only four.

A stage that must RUN something -- a structure writer, a live eQuilibrator call, a pKa
predictor -- still needs code, and that is what the `MetaboliteResolver` protocol and a
driver like `thermo_pipeline_yeast9.py` are for.

### Sigma

`assemble_reactions` propagates per-metabolite formation sigmas in quadrature weighted by
the stoichiometric coefficients. That treats them as INDEPENDENT and is an underestimate
whenever they are not: component contribution's covariance is strongly correlated across
compounds sharing groups, and the correlated version is the norm of `S_j^T Sigma^{1/2}`. A
caller holding the square-root factor should propagate it and overwrite; the yeast driver
does, through `parameters._reaction_sigma`.

### Test

`tests/torchcell/metabolism/test_thermo_pipeline.py`, 15 tests. Fourteen run on a synthetic
five-metabolite GEM built to carry one of each outcome: a metabolite with a structure, one
with only an accession, one with a placeholder formula, a biomass pool, and one a writer can
compose from its name. The fifteenth runs against yeast-GEM 9.0.2 when it is on disk and
checks only what must hold for any model.

### Files

- `torchcell/metabolism/thermo_pipeline.py`
- `tests/torchcell/metabolism/test_thermo_pipeline.py`
- `experiments/026-metabolism-flux/scripts/thermo_pipeline_yeast9.py` (the yeast driver)

## 2026.09.11 - Two stages added: a literature table, and what its values do not change

### `IonFormationResolver`

A bare ion is the one class of species component contribution cannot reach even in
principle: no groups to decompose, and not in the training reactions. Its formation energy
is measured and evaluated in the thermochemical literature instead, so it enters as a pinned
table and gets a route of its own, `ThermoRoute.LITERATURE_TABLE`, so no reader mistakes it
for a prediction. Keyed by (element symbol, charge); a metabolite matches when its formula
is a single element symbol. Nothing in the stage names a metabolite or a model.

Two properties the caller has to know, both documented on the class:

- **The values cancel where the reaction is balanced.** In a reaction elementally balanced
  in the ion's element the coefficients sum to zero and the absolute value drops out, as
  does the ionic-strength term whenever the compartments share an ionic strength. Measured
  by `thermo_ion_gauge_check.py`: of 22 yeast-GEM reactions touching a tabled ion, 6 move
  under a 1,000 kJ/mol gauge shift and all 6 are elementally unbalanced sinks.
- **A GEM may declare the wrong charge.** yeast-GEM declares potassium and sodium neutral,
  the metal rather than the aqueous ion. Where the model's charge matches no tabulated
  species and the element has exactly one tabulated ion, that ion is used and
  `charge_source: table_single_ion` is recorded per metabolite; where the element has
  several, the metabolite is left unpriced, because then the annotation is the only thing
  that could pick one and it is the thing in doubt.

The Debye-Huckel transform is restated in `debye_hueckel_kj` rather than imported, because
this module must import without eQuilibrator. It is not trusted: the table builder asserts
the restatement equals `equilibrator_cache.thermodynamic_constants._legendre_transform` over
a grid of charges, ionic strengths and temperatures on every rebuild, and records the
deviation, currently 8.9e-16 kJ/mol.

### `FileTableResolver`

The generic stage over a per-metabolite table on disk, CSV or parquet, with a stated unit.
It is what makes `python -m torchcell.metabolism.thermo_pipeline` usable on a new organism
with no code at all.

### A stage that is also a pipeline input has to read a FIXED upstream artifact

`thermo_accession_sweep.py` is both a script and a stage, and the first version took its
candidate list from the pipeline's own last result. That is a rebuild-order trap: every
metabolite the sweep priced would be absent from the next run's candidate list, and the run
after that would unprice it. The candidate list now comes from the eQuilibrator
recomputation parquet, which is strictly upstream and never written by the sweep, so the
sweep is idempotent and running it twice changes nothing.

The audit table was tried first and is the wrong input for a second reason worth recording:
it lists hydrogen sulfide as recovered with an empty final blocker while the recomputation
carries no value for it, so taking its word would have skipped the one metabolite the sweep
exists to catch.

## 2026.09.11 - Four reaction rules: cancellation, non-chemical, reaction pricers, balance

### A reaction can be priced without pricing every participant

Rule 1 said a reaction is priced only when EVERY participant carries a formation energy.
That is right for a sum and wrong for a cancellation: a species on both sides with
coefficients summing to zero contributes nothing to `sum_i S_ij dfG'_i`, so whether the
table has a number for it cannot change the reaction's chemical energy.
`cancelling_participants` groups the participants by `chemical_identity`, which is
(normalized name, formula, charge), and drops any group whose coefficients sum to zero.

Two properties are deliberate. Only groups holding an UNPRICED member cancel, so no
reaction that already had a value gets a different one: the two compartment copies sit at
their own pH, their transformed energies differ by the pH term, and dropping a fully priced
pair would move that term silently into the transport term. And the compartment work itself
is left to `torchcell.metabolism.transport`, which builds it from the stoichiometry, the
compartment of each metabolite and its CHARGE, and needs no price, so a species with no
formation energy still contributes its transport work and nothing is double counted.

Measured on yeast-GEM 9.0.2: 22 reactions gained a value, 0 of the 3,779 already priced
changed, 0 of 2,561 metabolite values changed. Nineteen of the 22 cancel entirely and carry
the new route `cancelled_participants` with a chemical energy of exactly zero.

### An exchange is not chemistry, and it was in the denominator

`UnpricedReason.NON_CHEMICAL` and `ThermoPipelineResult.reaction_counts_chemical`. An
exchange, demand, sink or biomass or pool pseudoreaction is not elementally balanced and
has no standard reaction energy to find; 284 of yeast-GEM's 4,131 reactions are one, and
leaving them in the denominator made every coverage fraction a different number from the
one it claimed to be. The rule is three config fields, because no single test finds them
all: cobra's `boundary` catches the 274 exchanges, a name token catches the carbohydrate
and cofactor pseudoreactions, and an SBO term catches the biomass pseudoreaction. A
non-chemical reaction that DOES carry a value keeps it and carries the flag, so no number
changed; what changed is which denominator it is reported against.

### A reaction energy that is not a sum of formation energies

`ReactionPricer` is a stage that prices a reaction directly, offered every reaction the
metabolite table cannot price after cancellation and ahead of the reaction-level fill
table. It exists for protein-bound redox carriers, whose measured quantity is a midpoint
potential of a COUPLE: the oxidized and reduced forms have no separate formation energy and
never will, so `dG'0 = -nF dE'0` prices the reaction and both species stay unpriced under
`PRICED_AT_REACTION_LEVEL`. The implementation is
`torchcell.metabolism.thermo_curation.CuratedReactionPricer`; nothing organism-specific
enters this module.

### The balance check the coverage number needed

`reaction_balance` computes the elemental and charge imbalance of every chemical reaction
from the participants' own formulas, skipping any reaction where a placeholder or empty
formula makes the sum undefined, so "we know this is wrong" is never confused with "we
cannot tell". Diagnostic only: no price depends on it. 205 of yeast-GEM 9.0.2's chemical
reactions do not balance.

### `legendre_transform_kj`

The single-microspecies Legendre transform, eQuilibrator's form, generalizing the
ionic-strength-only version `IonFormationResolver` used. It uses eQuilibrator's ROUNDED gas
constant 0.00831 rather than the CODATA value: the difference is 0.29 kJ/mol on a
twelve-proton species at pH 7.8, and a transform that disagrees with the one the cache
applied to its own compounds would put a systematic offset between curated rows and every
other row. Verified against `equilibrator_cache.thermodynamic_constants._legendre_transform`
over 1,008 grid points by `thermo_curation_price.py`, worst deviation 1.7e-13 kJ/mol, which
that script re-checks on every rebuild and refuses to write its output if it exceeds 1e-9.

### A curated reason outranks a later diagnosis

`MetabolitePrice.reason_is_final`. The precedence rule was "the last stage to speak wins",
which is right for two heuristics and wrong when the earlier one is a measurement: a
curator who ran the group decomposer on heme a and recorded that it refused the structure
is stating evidence, and a namespace heuristic that later calls the same species "no
structure available" moves the coverage ladder by reclassifying work that is not retrieval
as retrieval. A stage may now mark a reason final, and only the curated stage does.
