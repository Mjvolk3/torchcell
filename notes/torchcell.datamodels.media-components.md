---
id: w48yalavrhbsxz0kp0ufdfq
title: Media Components
desc: ''
updated: 1784089462236
created: 1784089462236
---

## 2026.07.14 - Component-based Media (design + SGA media/temperature correction)

Design + provenance rationale for the component-based `Media` model
(`torchcell/datamodels/schema.py`) and the reusable media library
(`torchcell/datamodels/media.py`). Branch `feat/media-component-schema`.

### What changed

`Media` was `name` + `state` only. It is now provenance-first and compositional:

- `MediaComponent` = a reused `Compound` (name-only when undefined; ChEBI / InChIKey
  / SMILES / PubChem filled **as sourced, never guessed**) at a `Concentration`, with
  a `role` (`MediaComponentRole`), a `definition` (`ComponentDefinition`), a **list**
  of `SourcedValue` `provenance`, a `defers_to` list, and a `note`.
- `Media` adds a **required** `is_synthetic` (no legacy `None`), `base_medium`,
  `components`, `dropouts`, recipe-level `provenance`, and computed
  `is_fully_characterized` / `open_gaps`.

Two orthogonal axes make gaps first-class and queryable:

- **Identity** (`ComponentDefinition`): `defined` (single sourced chemical) /
  `composition_deferred` (a defined sub-mix not yet expanded — commercial YNB, the
  SC amino-acid supplement — fillable via `defers_to`) / `intrinsically_undefined`
  (batch-variable digest: peptone, yeast extract, corn steep liquor).
- **Amount**: `concentration is None` == not yet sourced.

`is_synthetic` (made-from-defined-chemicals) is **orthogonal** to characterization:
a natural medium (`is_synthetic=False`, e.g. corn steep liquor) that is mass-spec'd
becomes a set of `defined` components, each with a measurement `SourcedValue`, while
its raw digest line stays `intrinsically_undefined`.

### Provenance: a LIST of SourcedValue + defers_to (deferral-chain traceability)

`SourcedValue` requires `quote` + `sha256` + `citation_key`, so it only holds papers
we have actually READ. A component keeps a **list** because (a) a value is often
corroborated by several papers and (b) the paper we read may differ from the one that
ORIGINATES the recipe. `defers_to` holds citation-keys of referenced papers not yet
mirrored+quoted; following one = mirror it, add a `SourcedValue`, flip the component
to `defined`. Motivating case: the SGA 5% FBA rule is Suthers'; "YNB = these 9
vitamins" is the Difco spec — **different facts, different sources on one medium**.

### Selection agents are COMPONENTS, not perturbations (documented stopgap)

Canavanine / thialysine / G418 / clonNAT are stored as `selection_agent`
`MediaComponent`s of the SGA medium, NOT as `EnvironmentPerturbation`s, because they
are **constant** to the medium, not the studied edit (a perturbation names the edit
vs the base medium; a studied drug at IC30 is a `SmallMoleculePerturbation`).

**This is a stopgap.** With a full genotype×medium mechanistic layer, "the strain
carries `kanMX` and the medium contains G418 ⇒ G418 does not kill it" would be a
DERIVED logical rule from (genome content) × (medium components), not a hardcoded
membership. We lack that layer today, so selection outcome is implicit in how the
medium+genotype are declared. Revisit when strain-design/DNA-level modeling lands.

### Wet-lab record vs model adapter (Cobra / AMICI)

`Media` records the **wet-lab truth** — real ingredients at real concentrations. FBA
conventions (Suthers' 5%-of-glucose supplement rule, ±1000 exchange bounds, Yeast9
`r_####` ids) are **model** artifacts and belong in a future **cobra adapter**
(`Media → exchange bounds`) / **AMICI adapter** (`Media → species initial amounts`),
NOT here. `Compound.chebi_id` is the join key an adapter uses to map an ingredient to
a model metabolite (`s_####`); `Concentration` is what it converts to a bound. So the
class is adapter-ready without importing cobra.

### Sourcing (all mirrored, sha256-pinned)

- **SGA media + SC amino-acid supplement recipe**: Tong & Boone 2006
  (`yantongSyntheticGeneticArray2006`, Methods Mol Biol 313:171-192, Materials 2.1,
  recipe #5 supplement + #16 SD/MSG). Full recipe #16 per L: 1.7 g YNB w/o AA &
  ammonium sulfate, 1 g MSG, 2 g DO(−His/Arg/Lys) supplement, 20 g agar, 20 g glucose,
  canavanine 50 mg/L, thialysine 50 mg/L, G418 200 mg/L, clonNAT 100 mg/L.
- **MSG rationale + media names**: Kuzmin 2016 CSH Protocols
  (`kuzminSyntheticGeneticArray2016`, `pdb.prot088807`, the ref-67 deferral target).
- **Screen temperature 26 °C** (whole SGA screen): Kuzmin 2018 SI
  (`kuzminSystematicAnalysisComplex2018`, si1.md).
- **FBA 5% supplement convention** (adapter-layer, NOT in the Media record): Suthers
  2020 (`suthersGenomescaleMetabolicReconstruction2020`). **DOI FIX**: the iBioFoundry
  `media_setup.py` docstring cites `10.1016/j.ymben.2020.03.010` — that is an unrelated
  DeBerardinis paper. The correct DOI is **`10.1016/j.mec.2020.e00148`** (*I. orientalis*
  SD108, Metab Eng Communications, open-access). Verified 5% quote in that paper.

### The correction this lands (the original task)

- **Media mislabel**: SGA fitness was scored on the SD/MSG **selection** medium, not
  YEPD (YEPD is only the intermediate mating/diploid step). `costanzo2016` +
  `baryshnikova2010` → `SGA_DM_SELECTION` (−His/Arg/Lys); `kuzmin2018` + `kuzmin2020`
  → `SGA_TM_SELECTION` (adds −Ura for the KlURA3 third marker).
- **Temperature**: `costanzo2016` was already per-record correct (raw `Arraytype/Temp`
  → 26/30). **Kuzmin was hardcoded 30 °C but the whole τ-SGA screen ran at 26 °C** →
  `kuzmin2018` + `kuzmin2020` fixed to `Temperature(value=26)`. `baryshnikova2010`
  kept at 30 °C as a documented representative (SMF control screens averaged 26+30 per
  the Costanzo SOM) — flagged for a decision.

### Migration + open follow-ups

- `is_synthetic` (required) added to all ~50 `Media(...)` sites: SC/SD/SM/Syn*/SED →
  `True`; YPD/YEPD/YP* → `False` (the medium **name** is the classification, not a
  guess). Non-SGA datasets keep their existing medium names for now (not yet routed to
  library constants) — per-dataset media enrichment is a follow-up.
- **ChEBI / InChIKey / SMILES / PubChem** cross-refs are empty in `media.py`; a sourced
  resolver pass (ChEBI/PubChem API, never guessed) fills them. `open_gaps` lists what
  is unfilled.
- **Rebuild**: media + Kuzmin temperature are embedded per-record, so the Costanzo
  (dmf/dmi) + Kuzmin (dmf/dmi/tmf/tmi) 43-45 GB LMDBs must be REBUILT + re-verified
  L0-L4 + re-landed. Heavy; user-triggered, not done in this PR.
- **Cobra/AMICI adapter** module (separate) to consume these media.
- The `Media` schema relocated below `Compound`/`Concentration` in `schema.py` (its
  component form depends on them); only consumer is `Environment`.
