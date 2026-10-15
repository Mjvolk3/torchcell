---
id: o9rcaoow70g1lg4y1vl32qv
title: Media_name_composition_scan
desc: ''
updated: 1792095207363
created: 1792095207363
---

## 2026.10.15 - Lane X: the composition a medium states only in its name

Lane W, measuring the environment-response identity sites
([[torchcell.verification.environment-response-identity]]), handed on two findings about
environment fidelity. This lane verified both, fixed them, and scanned every other dataset
for the same shape. Every number below regenerates from
`experiments/database/scripts/media_name_composition_scan.py` into
`experiments/database/results/media_name_composition_scan.json`; every store was read
READ-ONLY from the dev tree `/scratch/projects/torchcell-scratch/data/torchcell/`.

### Finding 1 verified: what YeastPhenome states, and what the loader stored

The curated column header is `<zygosity> | <phenotype> | <condition> | [<media> |]
<author>`, and the media field is where the release states the medium. The v1.0 pin
(`83e2917bf869`) releases **28 distinct media tokens** over the 44 retained growth screens,
27 of which reach stored records (`SC - Ura + DMSO [0.5%]`'s two columns are dropped by the
condition filter, which cannot represent `expression of human CYP1A2`). The loader stored
`media_field.split("+")[0]` as `Media.name` and nothing else, so the defect had two halves.

**The `+` half of the token reached no field at all.** 474,837 of 524,682 records carried a
name that is not the released token:

| stored `Media.name` | records | released tokens that collapse onto it |
|---|---|---|
| `YPD` | 239,081 | `YPD` (203,553), `YPD + G418 [200 ug/ml] + DMSO [1%]` (9,816), `YPD + MeOH [1%]` (9,079), `YPD + DMSO [2%]` (8,348), `YPD + EtOH` (4,228), `YPD + G418 [200 ug/ml]` (4,057) |
| `SC` | **212,402** | `SC` (160,040), `SC + DMSO` (38,695), `SC + DMSO [1%]` (4,835), `SC + DMSO [0.5%]` (4,767), `SC + MeOH` (3,977), `SC + DMSO [0.2%]` (88) |
| `SD` | 23,354 | `SD + his + leu + lys + ura` (9,450), `SD + His + Leu + Lys + Ura, -N` (4,837), `SD + His + Leu + Lys + Ura (2)` (4,837), `SD + His + Leu + Met + Ura` (4,181), `SD` (49) |

Lane W's 212,402 is exactly the `SC` row: **six** released tokens, one stored name. The
vehicles are not decoration. Nine of those eleven DMSO/MeOH/EtOH columns dose a compound in
the condition field (amphotericin B, clozapine, aflatoxin B1, quinine, two guanidinium
chlorides), so the vehicle is that compound's own; the other two are the vehicle CONTROLS
(`SC + DMSO [1%]` and `SC + MeOH`, both condition `standard`), which were stored as
indistinguishable from plain `SC`.

**What the retained half states sat inside a name `media_identity` drops.** The 27 stored
tokens reached **8** distinct media identities and **117** served environment identities, and
**3** of those environment nodes were reached from more than one released medium, over
**53,951 records**:

| served environment node | records | the media measured on it |
|---|---|---|
| `8ca16afd90cd` | 30,628 | `SC`, `SC - Ade`, `SC - Arg`, `SC - Lys`, `SC - Met`, `SC - Trp` (Acton 2017, all condition `standard`) |
| `f60ce2fdfe0f` | 13,649 | `SC + DMSO [1%]`, `SC + MeOH`, `SC, HEPES [25 mM]` (two vehicle controls and a buffered medium) |
| `d2a4a90f9a76` | 9,674 | `SD + His + Leu + Lys + Ura, -N` and `SD + His + Leu + Lys + Ura (2)` (Galardini 2019: a nitrogen-starvation arm and its own control) |

**And the L1 rule inherited it.** Re-keying every record the loader builds under
`environment_response._pair_key` reproduces lane W's number exactly, 519,845 unique triples
and **4,837 duplicated**, and names them: the whole residual allowance is the nitrogen-free
arm counted as a repeat of its control. An instruction to "aggregate the declared duplicates"
would have destroyed 4,837 measurements. That is lane W's aerobicity failure mode one level
down, and it is why this was worth doing before the build rather than after.

Quotes, all verbatim media fields of the pinned matrices (sha256 per screen in `SCREENS`):

- `SD + His + Leu + Lys + Ura, -N` and `SD + His + Leu + Lys + Ura (2)`, both with condition
  `standard`, `galardini_beltrao_2019_valuez.txt`
  (sha256 `d293951be28e1ddd84b2581d288e78ea3b0febb1b03c9261d4b0b99ee0d042ee`).
- `SC (-C)`, whose column's condition is `glucose [20%]`: the base is carbon-free and the
  carbon is dosed back as the condition, same screen.
- `SC, Glycerol [2%]`, `SC, HEPES [25 mM]`, same screen.
- `SC - Ade`, `SC - Arg`, `SC - Lys`, `SC - Met`, `SC - Trp`, each 4,195 records, all with
  condition `standard`, `acton_giaever_2017_valuez.txt`.

### What the record carries now

`Media.name` is the token VERBATIM, and `_parse_media_token` reads what it states into typed
fields. An unknown token form RAISES; there is no fallback, because the alternative is a
stored medium whose statement sits unread inside its name.

| released token | typed fields it now carries |
|---|---|
| `SC - Ade` / `- Arg` / `- Lys` / `- Met` / `- Trp` / `SC - Ura` | `dropouts=[adenine]` / `[L-arginine]` / `[L-lysine]` / `[L-methionine]` / `[L-tryptophan]` / `[uracil]`, each through the shared resolver, so the dropout joins the compound entity another dataset lists as a component |
| `SC, Glycerol [2%]` | `components=[glycerol, carbon_source, 2 percent]` |
| `SC, HEPES [25 mM]` | `components=[HEPES, buffer, 25 mM]` (HEPES has no row in the compound table, so its `Compound` carries the honest typed gap on `inchikey`) |
| `SC (-C)`, `SD + His + Leu + Lys + Ura, -N` | `dropout_roles=[carbon_source]`, `[nitrogen_source]` |
| `SC + DMSO [x%]`, `SC + MeOH`, `YPD + EtOH` | `components=[dimethyl sulfoxide / methanol / ethanol, other, at the stated fraction]` |
| `YPD + G418 [200 ug/ml]` | `components=[G418 (geneticin), selection_agent, 200 ug/mL]` |
| `SD + his + leu + lys + ura` | four components: `L-histidine`, `L-leucine`, `L-lysine` (`amino_acid`) and `uracil` (`nucleobase`), no dose, because the curation states none |
| every token | one `Media.provenance` `SourcedValue` quoting the token against the sha256 of the screen file whose header states it |

Three placement calls, each with its reason.

1. **A vehicle is a medium component (`other`), not `SmallMoleculePerturbation.solvent`.** The
   curation states it in the medium field, beside the base and not in the condition; it is the
   only encoding that holds for the two vehicle CONTROLS, whose condition is `standard` and
   which therefore have no perturbation to carry a `Solvent`; and one fact belongs in one
   field, not in two depending on whether a drug happens to be dosed beside it. The
   consequence: YeastPhenome's DMSO vehicle does not join Hoepfner's `solvent` DMSO.
2. **A selection agent is a component** by the schema's own ruling in the `Media` section
   ("constant to the medium, not the studied edit").
3. **Auxotroph supplements are components, not `CultureEnvironment.auxotroph_supplements`.**
   That slot is the documented home, and this dataset's experiment class states the base
   `Environment`, which has no such slot; narrowing the class is a different lane. The token
   states them as part of the medium, `base_medium` still carries the join to `SD`, and the
   alternative is dropping a stated fact, which outranks the placement preference.

### What stays a typed gap, and why

`YPG [3%]` states a **dose for an ingredient the curation abbreviates to a letter**. The
library distinguishes `YPG` from `YPGal`, but YeastPhenome never writes either out, so the
3 % is stored against an unexpanded `composition_deferred` carbon-source line
(`Compound(name="YPG carbon source, abbreviated by the curation")`) rather than against a
guessed molecule, and `Media.open_gaps` reports it. That is the `SM_DEFERRED` /
`M9_DEFERRED_BANERJEE2025` treatment: an undefined preparation is a bare `Compound` with a
`composition_deferred` definition, not a gap, "because undefined is the truth about the
bottle".

Two things deliberately unchanged, both recorded rather than papered over:

- **The base recipe stays unexpanded.** No library object is substituted for a token, so
  `SC` keeps `components=[]` with `base_medium="SC"`. That is this loader's standing treatment
  of a secondary curation layer that names a medium without printing it, and changing it would
  import a 30-component recipe with two other papers' quotes into a record whose source states
  only the label. The cost is that "every record on SC" still joins nothing from YeastPhenome,
  which is a cross-loader policy question, not this finding.
- **`SD + His + Leu + Lys + Ura (2)` and `SD + his + leu + lys + ura` share one media
  identity.** The `(2)` is the curation's own index for a second medium it spells identically,
  and it states no composition; the letter case is not identity either. They are two columns of
  two different screens with different conditions, so no record pair collapses.

### Measured before and after

| quantity | before | after |
|---|---|---|
| records stored | 524,682 | 524,682 |
| released media tokens | 28 | 28 |
| distinct stored `Media.name` | 13 | 27 |
| distinct media identities | 8 | 28 |
| served environment identities | 117 | 125 |
| environment nodes reached from more than one released medium | 3 | **0** |
| records on such a node | 53,951 | **0** |
| L1 duplicated records (`_pair_key`) | 4,837 | **0** |

`ENVIRONMENT_RESPONSE_DATASETS["yeastphenome"]["expected_duplicated_pairs"]` is therefore
**0**, from 4,837, from the 14,508 of job 4606. Nothing is dropped and nothing is merged.

### Finding 2: `growth_phase` is identity-bearing, and today it splits nothing

`Environment.growth_phase` was in no identity projection, served or verifying.
`CultureGrowthPhase`'s own definition settles the call: the harvest MOMENT is a property of
the sample and rides on the phenotype, while `Environment.growth_phase` is a culture HELD in a
phase AS the condition (Schmidt 2016's starved arms), so two records differing only in it are
two conditions and must be two environment nodes. It joins
`ENVIRONMENT_OPTIONAL_IDENTITY_FIELDS` rather than the required tuple, which is what its own
construction already says: `exclude_if=lambda v: v is None` leaves it out of the dump when
unstated, so projecting it only when set keeps every existing environment id byte-identical.

Measured effect on duplicate counts, over all 181 readable dev stores:

| store | records | phases sampled | environments | environments that MERGE without the phase |
|---|---|---|---|---|
| `growth_rate_s23_schmidt2016` | 26 | 24 none, 2 stationary | 22 | 0 |
| `proteome_schmidt2016` | 21 | 19 none, 2 stationary | 21 | 0 |
| `proteome_srm_set1_schmidt2016` | 18 | 16 none, 2 stationary | 18 | 0 |
| `proteome_srm_set2_schmidt2016` | 21 | 19 none, 2 stationary | 21 | 0 |

**4 stores state a phase and 0 environments merge without it**, because both stationary arms
also differ in `duration_hours` (24 h and 72 h) from each other and from every exponential
arm. So **no store's duplicate count changes today**: the projection is correct for the first
release that holds a phase at ONE duration, which is the case the field exists for, and the
measured null is a property of these stores rather than of the field. Reading it costs nothing
either: it is one scalar, and the fix is three lines across the two projections.

### Is it one dataset's bug? The scan, and what it found

Lane Q established that a defect of this family is worth scanning for rather than assuming.
The detector flags a medium whose NAME states something its typed fields do not carry, and it
is deliberately conservative: a statement counts as CARRIED when a typed field on the medium
says it, when the compound the name mentions is a component or a dropout under any role, when
the marker is part of a REAGENT's name ("yeast nitrogen base w/o amino acids"), when the
composition is LISTED and no component fills the role the name says is absent (how every
`M9 ... noCarbon` object states it), or when the ENVIRONMENT states it beside the medium as a
physical factor (where a carbon-source series puts it). A statement reachable only through the
`base_medium` string is reported with a `(base_medium)` suffix, because `media_identity` does
not follow that string.

**The shared library: 3 of 109 objects.** `M9_AA_SCHMIDT2016` (the word `Glycerol` is inside
the source's own arm label and the carbon source is an environment factor, so its store is not
flagged), `M9_MODIFIED_FANG2025` (pH, below) and `M9_DEFERRED_BANERJEE2025`, whose name states
"no carbon source" over a deferred whole-medium mixture: the one other place `dropout_roles`
has an obvious use. Its store is not flagged because the environment names the carbon sources,
so it is listed, not fixed.

**The dev stores: 7 of 181 readable, with 26 unreadable.**

| store | records | what the name states that no typed field carries | disposition |
|---|---|---|---|
| `yeastphenome` | 524,682 | 9 media: five dropouts, a role dropout, a carbon source, a buffer, three doses | **fixed here** |
| `phage_rbtnseq_mutalik2020` | 287,815 | `LB with SM buffer (Teknova) plus 10 mM CaCl2 and MgSO4`: both compounds ARE typed components, with `concentration=None`, so only the `10 mM` is in the name | **listed**: the token reads "10 mM CaCl2 and MgSO4", which is 10 mM each or 10 mM of the pair, and resolving it needs the source rather than a reading of the name |
| `crispri_guide_ffa_enrichment_fang2025` | 15,711 | `(pH 7.2)` | **listed**: `Media` has no pH slot and the loader sets no `ph` factor. A buffered formulation pH is not the same claim as a pH-titration arm, so this is a design question (a `Media.ph` field) and not a one-line fix |
| `growth_yield_baba2006` | 11,735 | `MOPS medium`, whose buffer is a component of the base `MOPS_MINIMAL` and so reachable only through the `base_medium` string | **listed**: carried, but not by anything `media_identity` reads |
| `_verify_ozaydin_geneadd`, `_verify_ozaydin_rename` | 4,474 each | `SC-URA` with no typed dropout | **stale**: built 2026-07-05; the current loader builds `OZAYDIN_SC_URA_AGAR` off the library `SC_URA`, which carries `dropouts=[uracil]`, and no live code emits that name |
| `gene_essentiality_sgd` | 1,329 | `YPD with 2% glucose`: the annotation's own carbon source and dose were in the name only, with an empty component list | **fixed here**: a typed `carbon_source` component at 2 percent, carrying the annotation's own quote. The loader already typed the bare `2% glucose` condition as a `carbon_source` factor, so the two SGD condition rows are now consistent |

The 26 unreadable stores are the known cross-lane state and not this lane's: 24 raise
`AttributeError: Can't get attribute 'ReleasedGeneKind'` or `'ReleasedStrainKind'` (the P3
enum rename; lane W named 3 of them, and the whole set is 24), and 2 showcase stores raise
`UnpicklingError: invalid load key, '['`. The sweep rebuilds them, and the scan names each one
rather than skipping it.

A scan limitation worth recording, because it changes a count: a store that INTERNS its
constant sub-objects stores `{"$ref": <hash>, "name": <hint>}` in place of the environment, so
a reader that does not splice the `processed/interned` table back sees a two-key dict and
reports the store as carrying no medium. The first pass of this scan did exactly that and
missed `phage_rbtnseq_mutalik2020`, `growth_yield_baba2006` and `crispri_guide_ffa_enrichment_fang2025`
entirely, and reported `growth_phase` as unstated on all four Schmidt stores. The committed
script resolves refs.

Coverage, stated honestly: each store is read on a per-store key stride (`--cap`, default
3,000 records spread over the whole key range), so a medium used by fewer than one record in
`entries / 3000` can be missed. Stores at or under the cap are read in full, which is 174 of
the 181. The shared-library pass is exhaustive over the 109 objects every loader draws from,
which is the completeness argument for the loaders that use them.

### Tests

Each verified to FAIL on `origin/environment-identity` with the assertion named.

| test | failing assertion before the fix |
|---|---|
| `test_condition_signature_separates_a_nitrogen_free_medium_from_its_control` | `assert _condition_signature(experiment([])) != _condition_signature(experiment(["nitrogen_source"]))`, both sides `((), None, 'SD + His + Leu + Lys + Ura', ...)` |
| `test_condition_signature_separates_a_held_growth_phase_from_its_control` | `assert len(signatures) == 3`, left side `1` |
| `test_condition_signature_reads_every_served_identity_field` (lane W's tripwire, extended) | `assert set(edits) == set(fields)`, extra item `'growth_phase'` |
| `test_a_role_level_dropout_is_identity_and_an_empty_one_changes_no_id` | the two `identity_sha256` values are equal, both `6df8b535d7a7...` |
| `test_a_dropped_role_is_not_the_same_statement_as_a_dropped_compound` | `KeyError: 'dropout_roles'` |
| `test_the_order_of_dropped_roles_is_not_identity` | `KeyError: 'dropout_roles'` |
| `test_a_filled_role_cannot_also_be_declared_omitted` | `ValidationError: dropout_roles Extra inputs are not permitted` where `cannot both state a component` was expected |
| `test_the_growth_phase_a_culture_is_held_in_is_part_of_the_environment_id` | `assert len(ids) == 3`, left side `1` |
| `test_a_stated_medium_projects_no_gap_key_and_keeps_its_id` | `assert ('provenance_gaps',) == ('dropout_roles', 'provenance_gaps')` |

The nine loader-side tests in `test_yeastphenome_synthetic.py` (the verbatim name, the named
dropout, the carbon source and the buffer, the role dropout and the identity it separates, the
dosed base, the raising parser, the medium statement, the vehicle and selection agent, the
supplements) cannot fail on an assertion before the fix: they call a parser that does not
exist there, so the file does not collect. The pre/post evidence for that half is the
before-and-after table above, regenerated by the committed script.

### Record

`database/schema_changes/2026-10-15-media-dropout-roles-and-growth-phase-identity.json`,
`additive_field` on `Media`, `rebuild_scope` tool-computed as `full_kg_rebuild` over 51 served
datasets. The rebuild scope is what any `Media` fingerprint change implies, and the KG 4.0
full rebuild is already the plan.

The field carries `exclude_if=lambda v: not v`, which is load-bearing and was found by a
failing gate rather than by design: without it every medium's dump gains
`"dropout_roles": []`, which changes the content-addressed experiment id of every record in
every store for a field that says nothing, and **ten adapter tests that pin the yeast node
lists and the content-addressed experiment, genotype and publication ids fail**
(`test_cell_adapter`, `test_ohya2005_adapter`, `test_bacterial_graph_classes`). With it, a
medium that states no role-level omission serializes to exactly the bytes it did before the
field existed and all 1,002 adapter tests pass. That is the `Environment.growth_phase`
precedent, and it is why an additive field on a class in every served closure is worth
checking against the adapter suite rather than the datamodel suite alone.
