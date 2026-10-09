---
id: 8qc2ra55yg8h9ve0gah7v61
title: Compound_identity
desc: ''
updated: 1784601915913
created: 1784601915914
---

## 2026.07.20 - UI-2 compound-identity resolver

Shared, offline, pure resolver that fills `Compound` structure IDs from a committed,
sha256-pinned name->structure table, and gap-marks the unresolved residue. Part of the
env/chemogenomic audit pipeline (UI-2; plan [[plan.env-compound-identity-resolver.2026.07.20]]).

- **`resolve_compound_identity(name=, pubchem_cid=, known_proprietary=)`** -> typed
  `CompoundIdentityResolution` (status RESOLVED / UNRESOLVED_PUBLIC / PROPRIETARY). Reads
  ONLY `compound_identity_table.json` (sha256-self-checked at import); NEVER hits the
  network at build/CI/test time. Mirrors the single-shared-resolver shape of
  `SCerevisiaeGenome.resolve_gene_name`.
- **`resolved_compound(...)`** loader helper builds a `Compound` fill-or-gap ADDITIVELY:
  fills only structure fields that are None (never clobbers a caller's smiles/pubchem_cid,
  e.g. hoepfner2014), and attaches `ProvenanceGap(field="inchikey", reason=...)` only when
  inchikey is truly None -- `deferred_pending_source_review` for unresolved public names,
  `not_reported_by_primary` for known-proprietary (hoepfner CMB codes, smith2016 vendor IDs).
- **Table** = 33 pinned records (32 resolved to a valid InChIKey; `tunicamycin` a mixture ->
  UNRESOLVED_PUBLIC gap, honest not guessed). Built ONCE offline by
  `scripts/build_compound_identity_table.py` (PubChem PUG REST, `RetrievalMethod.pubchem_api`);
  the committed JSON is canonical thereafter.
- Wired into 13 loaders + the yeastphenome plant-defensin peptide retype (-> BiologicPerturbation).
- NOT done here (follow-ups): the strict structure-or-gap `Compound` validator (UI-2b -- ~65
  name-only sites incl. media.py); `Compound` gaps vs `Media.open_gaps` reconciliation; the
  full DB rebuild the loaders' new output implies (UI-3 / KG build).

## 2026.09.12 - serve-50 regrow: 33 rows to 5,498, and the canonical-name policy

The serve-50 reviews of the 14 unserved datasets all landed on the same shared blocker:
the pinned table had 33 rows, so almost every chemogenomic record carried a compound with
no structure identifier and would have been dropped. This regrows the registry from the
union of what those reviews ask for, from committed input lists, with a rerunnable
curator.

### What changed

- **`torchcell/datamodels/compound_identity_curate.py`** (new) is the only thing that
  touches the network: `--names FILE --cids FILE --out TABLE` (both list flags
  repeatable), PubChem PUG REST at <= 5 req/s, CID property and synonym lookups batched
  100 per POST, `retrieval_method: pubchem_api`, `retrieved_at: 2026-09-12`. It replaces
  `scripts/build_compound_identity_table.py`, whose 28-name seed list is now a subset of
  `compound_identity_inputs/curated_core.txt`. **That old builder is stale and would
  shrink the table if rerun; it should be retired by whoever owns `scripts/`.**
- **`torchcell/datamodels/compound_identity_inputs/*.txt`** (new, 11 files) carry one
  label per line with a header comment naming the review the list came from, plus typed
  directives (`query=`, `cid=`, `canonical`, `chebi=`, `mixture=`, `unresolved=`,
  `proprietary=`, `undefined=`). The curation decisions live in these files, not in code.
- **`compound_identity_table.json`** goes from 33 to **5,498 rows** (schema_version 2,
  new `synonyms` and `unresolved_reason` fields), 3.1 MB, and re-pinned at
  `da91da75e84743e4b782600ead0fe3bc220c4bdc02ee8f6d89734de4934c037c`.
  Import plus load measured at 0.23 s.
- **`compound_identity.py`** gains synonym indexing, three statuses, the RDKit route, and
  the canonical-name policy below.

### Counts

Rows before: **33** (32 with an InChIKey). Rows after: **5,498**.

| status | rows |
|---|---|
| RESOLVED (InChIKey) | 5,441 |
| RESOLVED_MIXTURE (ChEBI/CID, no single key) | 2 |
| UNRESOLVED_PUBLIC (recoverable) | 28 |
| PROPRIETARY (vendor code, terminal) | 22 |
| UNDEFINED_MIXTURE (no structure exists) | 5 |

5,442 rows carry a PubChem CID, 2,270 carry a ChEBI CURIE, and 5,292 synonym keys point
onto them, so `_BY_NAME` holds 10,785 lookup keys for 5,498 compounds.

Per source, counted over the input labels (a label is counted for every review that asks
for it, so the total 5,691 exceeds the row count where sources overlap):

| source | labels | resolved | mixture | unresolved | proprietary | undefined |
|---|---|---|---|---|---|---|
| wildenhain2015_cids | 5,173 | 5,173 | 0 | 0 | 0 | 0 |
| hillenmeyer2008 | 345 | 308 | 1 | 24 | 12 | 0 |
| media_library | 56 | 50 | 1 | 0 | 0 | 5 |
| vanacloig2022 | 45 | 42 | 0 | 3 | 0 | 0 |
| curated_core | 26 | 26 | 0 | 0 | 0 | 0 |
| smith2016 | 20 | 10 | 0 | 0 | 10 | 0 |
| bloom2019 | 20 | 19 | 1 | 0 | 0 | 0 |
| hoepfner2014 | 2 | 1 | 0 | 1 | 0 | 0 |
| nadal_ribelles2025 | 2 | 2 | 0 | 0 | 0 | 0 |
| costanzo2021 | 1 | 0 | 1 | 0 | 0 | 0 |
| mota2024 | 1 | 1 | 0 | 0 | 0 | 0 |

Hillenmeyer's 345 labels are the union of the HET and HOM small-molecule `cond1`/`cond2`
labels; 309 went to PubChem and 308 came back with an InChIKey, against the review's
projection of 302 growable names. Fourteen of them are bare `CID <n>` labels resolved by
CID. The 5,173 Wildenhain CIDs all resolved, so the 427,177 records the review measured as
"CID + SMILES, no InChIKey" now key on a structure.

### Canonical-name policy

`resolved_compound` returns the table's canonical `name` in `Compound.name`, not the label
the loader passed: a row's name is PubChem's `Title` lowercased unless an input line
curates a spelling (`media.py` owns `D-glucose`; the pre-serve-50 table owns
`actinomycin D`), and every other spelling is a `synonyms` entry that resolves onto that
row. Loaders therefore pass their SOURCE LABEL as a lookup key and get one identity back,
which is what collapses Hillenmeyer's `NaCl` and the served Nadal-Ribelles loader's
`sodium chloride` onto a single `environment perturbation` node instead of two nodes
joinable only on the `inchikey` property.

`schema.py` is NOT changed: `Compound` has no `source_label` slot and sits in all 36
served closures, so adding one would be a full-rebuild trigger. The label is preserved in
the table's `synonyms` instead. `compound_identity.py` is outside the class-contract
surface (`schema_deps.default_surface_modules` is `schema.py` plus `pydant.py`), so none
of this is a schema fingerprint change; a rebuilt loader's OUTPUT does change, which is
adapter-level drift the manifest gate reports.

A lookup key claimed by two compounds is awarded by precedence, curated > PubChem name
index > CID only, and the losing rows are renamed `<name> (CID <n>)`; a tie leaves the key
unassigned, because a label that names two compounds equally well names neither. One row
needed this: Wildenhain's `cisplatin` CID collided with the CID PubChem's name index
returns for `cisplatin`, so the name-route row keeps the bare name and the other became
`cisplatin (CID 5460033)`. `_load_table` raises on any residual duplicate key rather than
letting the last row loaded win.

### Two structure routes, and the curated one wins

`inchikey_from_smiles(smiles) -> str | None` is a pure RDKit derivation for datasets that
release structures but no name a table can key on. Measured on Hoepfner's `Table_S1.xls`
"All Structures" sheet: **151 of 152 SMILES parse, yielding 150 distinct InChIKeys**; the
one failure is CMB 409, Boromycin, whose boron cage RDKit will not read. That reproduces
the review's measurement independently.

`resolved_compound(..., derive_from_smiles=True)` uses the route ONLY when the table has
no row, because the two routes disagree: concanamycin A resolves to
`DJZCTUVALDDONK-HQMSUKCRSA-N` from the curated PubChem row and to
`DJZCTUVALDDONK-UHFFFAOYSA-N` from its SMILES, same skeleton, different stereo block. A
curated row always outranks the derivation, and the derived case is reported as
`RESOLVED_FROM_SMILES`.

### What stays unresolved, and why each one is honest

Twenty-two PROPRIETARY rows: the ten Smith 2016 vendor catalog codes (`0KPI-0099`,
`1181-0519`, `4130-1276`, `6630449`, `7312221`, `9121982`, `9125678`, `9150499`,
`CBF-666774`, `ST016598`), the ten Hillenmeyer `chemical diversity labs <code>` codes, and
`RIL1 (Biomol)` / `telomerase inhibitor ix (Biomol)`. Additional file 8 and the Hillenmeyer
SOM release no structure, SMILES or CAS for any of them, so the drop is terminal and now
auditable from the table's `unresolved_reason` rather than from a code comment.

Five UNDEFINED_MIXTURE rows: `yeast extract`, `peptone`, the two `yeast nitrogen base`
preparations, and the `SC amino-acid supplement powder (DO -His/Arg/Lys)`. These are
autolysates, digests and formulated blends; no structure exists to find, which is a
different claim from "not looked up yet", and the split keeps them off the worklist.

Twenty-eight UNRESOLVED_PUBLIC rows, each with a recorded reason: activity-class labels
(`DNA protein kinase inhibitor`, `phosphatase inhibitor`, `tyrphostin`,
`cantharidin analog`, `parkinson-inducing peptide`); genus names that do not pick a
congener (`amphotericin`, `bisphenol`, `latrunculin`, `motuporamine`, `helenine`);
abbreviations the SOM never expands (`DMAEC`, `BCS`); `ptp2`, which SOM Table S3 shows is
a condition and not a compound; `FeCl4`, which does not exist as written; three
`DMSO <n>%` labels and four `<compound>, <n>ul total up/dn` labels that fuse a dose or a
hybridization volume into the compound name; `MBO`, where the Vanacloig paper's
Abbreviations block and its Introduction name different compounds (2-methyl-3-butyn-2-ol
vs 2-methyl-3-buten-2-ol) and the contradiction needs adjudication against Table S1;
`QUADRIS1` and `QUADRIS2`, a commercial suspension the paper says is not a defined
species; `Boromycin`, whose released SMILES will not parse (PubChem carries CID 76962270,
but adopting it would substitute a different structure for the one the screen released);
and three labels measured against PubChem on 2026-09-12 and found absent,
`AG 1387 (Biomol)`, `rutilantin` (the label does not pick A vs B), and
`dimethylellipticinium` (elliptinium, CID 42723, is the MONOmethyl congener).

Two RESOLVED_MIXTURE rows, which DO satisfy the identity rule through ChEBI or a CID:
`tunicamycin` carries `CHEBI:29699`, whose definition reads "A mixture of antiviral
nucleoside antibiotics ... at least 10 homologues", and PubChem returns no CID for the
name, so `inchikey` stays null rather than borrowing one homologue's key; `agar` keeps
CID 71571511 and `CHEBI:2509` but not PubChem's InChIKey, which belongs to the agarobiose
repeat unit and not to the algal polysaccharide weighed into a plate.

### ChEBI rule

A ChEBI CURIE is adopted from PubChem's synonym list only when that list carries exactly
ONE distinct ChEBI id. Acetic acid's synonyms carry both `CHEBI:15366` and `CHEBI:47622`,
and choosing between them is a curation decision, so such rows stay null unless an input
line states the id with `chebi=`. The six hand-verified ids of the original table are
carried forward that way.

### Known limitation, and it is a real one

PubChem's name index is the authority for a name lookup, and it can map a bench label onto
a neighboring species: `triphenyltin` returns CID 6460, Triphenylstannane (the hydride),
where a yeast screen almost certainly used the chloride or the hydroxide. Every row records
the exact endpoint queried, so a later pass can audit and correct these, but no
case-by-case chemistry judgment was applied beyond what PubChem answers.

### Not done here

`media.py` still builds most of its components with bare `Compound(name=...)` rather than
`resolved_compound(...)`, so the shared media objects do not yet carry the identifiers this
table now holds for 50 of their 55 names. Wiring that (and the loader-side changes each
review lists) is the orchestrator's follow-up.

## 2026.10.09 - #726: the bacterial gap, measured; the curator run blocked by PubChem

#726 asks for a curator pass over the bacterial condition labels and lists what was known
when it was filed (Price 2018's 53, Tong 2020's 17, Goodall's chloramphenicol, isoprenol,
PRECISE-1K's 22 plus kanamycin, then D-alanine, L-malate and alpha-pinene in the
comments). That is a running tally, not a measurement, so the gap was measured from the
built stores first.

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/bacterial_compound_identity_gaps.py`
Results: `experiments/036-dataset-fixes-before-kg-build/results/bacterial_compound_identity_gaps.json`
and `.../bacterial_compound_identity_gaps.csv`

### What the gap actually is

Every dataset in `torchcell/knowledge_graphs/conf/kg_bacteria.yaml` (57) was scanned
against the pinned table
(`_TABLE_SHA256 = 5a476defa3b109b373c7957001cb186c37668485e35afcf66ce058e739a85921`).

| quantity | measured |
|---|---|
| datasets scanned | 57 |
| compound references read | 77,536 |
| distinct labels that DO resolve | 127 |
| distinct labels with a typed InChIKey gap | **276** |
| of those, name-only (no identifier at all) | 275 |
| of those, that would RENAME a served node if a row were added | **0** |

The scan reads each store's `processed/interned` env, which holds every deduplicated
environment and reference, so one cheap pass covers every medium component, dosed small
molecule, solvent, physical-factor agent and media dropout however many records carry
them. Records are read only for the stores whose PHENOTYPE can hold a compound of its own
(`ProductTiterPhenotype.product` is the only one) or that interned nothing, which is what
keeps the pass off the multi-gigabyte stores. Two consecutive runs agree exactly on 276
and 77,536; the numbers are a snapshot of the dev tree, which is shared across worktrees,
so a concurrent rebuild moves them (an earlier run on this branch read 303 and 78,100).

**The 0 renames is the load-bearing number.** The hazard the issue's own comment records
is that a compound row is not additive: adding IPTG once changed what
`resolved_compound("IPTG").name` returns and so renamed the IPTG node of already-served
Foo 2014 records. Measured here, none of the 276 gap labels resolves to a row today, so
none of them can be renamed by a row that did not exist; what a row changes is only
whether the node carries an InChIKey. A `canonical` directive per label keeps it that way,
which is the precedent `bacterial_media.txt` set ("media.py owns the bench spelling").

### The 276 split three ways, and only one of them wants a PubChem row

This triage is by label text and is a grouping aid for the pass, not a measurement; each
assignment stays a curation decision.

| proposed family | labels | what it wants |
|---|---|---|
| `single_substance_candidate` | **198** | a `canonical` row, with `query=` where PubChem's name index needs another spelling |
| `environment_placeholder` | **61** | NO row ever: these are `Compound` objects naming a whole growth environment the release did not carry (every one is `growth environment of putidaPRECISE321 condition '...'`, from Lim 2022) |
| `undefined_or_proprietary_preparation` | **17** | `undefined=` or `proprietary=`: medium bases (`LB base, PRECISE-1K formulation`, `Davis Minimal medium salts (Lenski 1991 formulation)`, `M9 salts (Difco)`), trace-element stocks (`sauer trace element mixture`, `trace metal solution (Teknova T1001)`), an amino-acid mix (`20AA_mix`), and vendor products (`Durasyn 164`, `SM buffer (Teknova)`) |

The 198 concentrate in the two largest condition panels: Borchert 2024 (63 labels), Price
2018 (56), Lamoureux PRECISE-1K (23 plus 17 on its public-K12 arm) and Tong 2020 (18).
`kanamycin` alone appears in 9 stores and `yeast extract` and `tryptone` in 19 and 17,
which is why a single row moves many datasets at once.

### The pinene row Niu 2019 needs (#799)

Re-measured against the pinned table: **no spelling of pinene has a row.** `pinene`,
`alpha-pinene`, `alpha-Pinene`, `(-)-alpha-pinene` and `(+)-alpha-pinene` all return
`UNRESOLVED_PUBLIC` with a null InChIKey, exactly as #726's comment recorded. The
recommended input line stays the one that comment proposes,

```
alpha-pinene | canonical
pinene
```

with `alpha-pinene` canonical because this repository's notes and the bioproduction
literature spell it that way, and bare `pinene` on its own line so Niu's own label reaches
the same node IF PubChem's name index answers both with one CID. Whether it does is NOT
measured (the lookup needs the network), so that collapse is a hypothesis until the run.
The paper never states the enantiomer, so the unspecified parent rather than (+) or (-) is
the record to take.

### Why the run itself did not happen today: a measured PubChem IP block

The pass requires re-querying PubChem for every row. Measured on 2026.10.09 from
GilaHyper: PubChem answers **HTTP 429 with an HTML body** to every request, including
trivial ones (`water`, `glucose`, `ethanol`), from any User-Agent and with no
`Retry-After` and no `X-Throttling-Control` header. The block is IP-based: a
browser-like User-Agent, no User-Agent, and IPv6 all behave the same. The curator's own
abort names the shape of it: `PubChem stayed busy after 0 busy and 21 rate-limited
retries for .../name/tolfenamic%20acid/...`, i.e. 21 consecutive attempts spaced a minute
apart, every one a 429, with the dynamic-throttle budget untouched. So the table was
NOT regenerated and `_TABLE_SHA256` is unchanged; a partially built table is not committed,
because a missing row and an unqueried row are different claims.

Two things were fixed so the run is possible when the block lifts (details in
[[torchcell.datamodels.compound_identity_curate]]): the client now treats a 429 as the
throttle it is, on its own patient schedule, instead of raising `JSONDecodeError` on an
HTML body, and `--min-interval` lets a long pass run slower than the published 5
requests/second ceiling. The exact command to resume, which reuses a response cache so an
interrupted pass costs time rather than progress, is in the module docstring.

### A separate finding: five bacterial dev stores cannot be read by current code

The scan found five stores whose interned pickles name schema classes that no longer
exist, so no current reader can load them:

| dataset | missing class |
|---|---|
| `ProteinTurnoverGupta2024Dataset` | `Censoring` |
| `IsoprenolTiterDeSiqueira2025Dataset` | `BacterialVariantType` |
| `CrispriArrayYunus2026Dataset` | `FoldChangeScale` |
| `CrispriKnockdownYunus2026Dataset` | `FoldChangeScale` |
| `CrispriDifferentialProteomeYunus2026Dataset` | `FoldChangeScale` |

All five must be rebuilt before KG 4.0. **Four of the five are invisible to the staleness
gate**: `python -m torchcell.database.build_dataset_lmdb --list-stale --include-private`
names `CrispriDifferentialProteomeYunus2026Dataset` but not the other four, because the
gate fingerprints the schema CLOSURE and a class that has been deleted outright is not in
any closure to compare. A store that cannot be unpickled is the strongest possible
staleness signal and the gate does not see it.
