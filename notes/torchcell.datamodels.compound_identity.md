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

## 2026.10.11 - #726 and #860: the curator pass over the bacterial condition labels

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860.py`
(the curator on ONE list, merged additively), with its log
`results/compound_curator_726_860_log.txt`, per-label evidence
`results/compound_curator_726_860.csv` / `.json`, the lookup-key diff
`results/compound_identity_table_diff.json`, and the store measurement
`scripts/compound_curator_726_860_store_reach.py` ->
`results/compound_curator_726_860_store_reach_{before,after}.json`.
Input list: `compound_identity_inputs/bacterial_conditions.txt` (sha256 `190b5b14...`).

### PubChem answered from GilaHyper

Three manual probes (`water`, `glucose`, `chloramphenicol`) returned HTTP 200 on
2026-10-10, about 5.2 s each (the resolver stall, not a throttle), and the pass completed
without the curator raising. 434 PubChem responses are cached at
`$DATA_ROOT/tmp/pubchem_curator_cache_726_860_2026-10-10.json` (sha256 `1119efcc...`), so
the run reproduces offline.

### Scope: 178 labels

Every single-substance label with a typed InChIKey gap in the built stores of the six
datasets the two issues name (Price 2018 RB-TnSeq, Tong 2020, Goodall 2018, PRECISE-1K
and its public-K12 arm, Borchert 2024), measured by the 2026.10.09 gap inventory; then
every other gap label whose stores all lie in the set those rows already force a rebuild
of (Girgis 2009, Choe 2025, Schmidt 2016 and others, so no extra store is touched); Banerjee
2025's `D-alanine` and `L-malate` and Niu 2019's `pinene` from the issue comments; the
three RCH2 base salts; and `4-Hydroxyvalerate`, a carbon source of the 20 dropped Borchert
samples the inventory could not see. The 17 media-base and trace-stock preparations
(`LB base, PRECISE-1K formulation`, `sauer trace element mixture`, `20AA_mix` and the rest)
are NOT here: their loaders build a bare `Compound` without the resolver, so a row would
change nothing; the 61 Lim 2022 environment placeholders never get a row.

### Outcome

| status | labels | meaning |
|---|---|---|
| RESOLVED | 166 | InChIKey + CID |
| RESOLVED_MIXTURE | 4 | CID (and ChEBI where unique), no single-molecule key: Bacitracin, Benzalkonium Chloride, gentamycin, trimethoprim-sulfamethoxazole |
| UNDEFINED_MIXTURE | 3 | no structure exists: casamino acids, Tween 20, Povidone-Iodine |
| UNRESOLVED_PUBLIC | 5 | a recorded reason: DPD, PQ, ET2DA, polymyxin B, pinene |

Table: 5,533 rows to 5,679 (146 rows added, 26 labels folded into the existing row that
already holds their CID), sha256 `5a476def...` to `d45d3ded...`. The lookup-key diff
against origin/main's table: 178 keys added, 0 keys renamed, 0 lost, 0 removed, 0
identifier changes, so no existing row moved.

**Folds (one compound, one node).** A label whose PubChem CID an existing row holds is
added to that row as a synonym rather than given a second row; the InChIKeys must agree
or the merge stops. Only the source label is added, never the PubChem Title.

| label(s) | existing row | CID |
|---|---|---|
| `2,2'-Dipyridyl` | `2,2'-bipyridine` | 1474 |
| `3-methyl-3-butenol` | `isoprenol` | 12988 |
| `doxycycline` | the row Wildenhain's CID list named by PubChem's IUPAC Title (`4-(dimethylamino)-1,5,10,11,12a-pentahydroxy-...`) | 54671203 |
| `4-Aminobutyric acid` | `gamma-aminobutyric acid` | 119 |
| `5-Hydroxymethylfurfural` | `5-(hydroxymethyl)furfural` | 237332 |
| `e-Amino-N-Caproic Acid` | `6-aminohexanoic acid` | 564 |
| `Cobalt chloride hexahydrate` | `cobalt(II) chloride hexahydrate` | 24643 |
| `copper (II) chloride dihydrate` | `copper(II) chloride dihydrate` | 61482 |
| `D-Cycloserine` | `cycloserine` | 6234 |
| `D-Fructose` | `fructose` | 2723872 |
| `D-Galactose` | `galactose` | 6036 |
| `D-Mannose` | `mannose` | 18950 |
| `D-Sorbitol` | `sorbitol` | 5780 |
| `D-Xylose` | `xylose` | 135191 |
| `2-Furfuraldehyde` | `furfural` | 7362 |
| `arginine`, `leucine`, `methionine`, `phenylalanine`, `threonine`, `tyrosine` | the `L-` rows | 6322, 6106, 6137, 6140, 6288, 6057 |
| `glutamate` | `L-glutamic acid` | 33032 |
| `L-Aspartate` | `L-aspartic acid` | 5960 |
| `Nicotinic Acid` | `niacin` | 938 |
| `thiamin` | `thiamine` | 1130 |
| `Na-citrate` | `sodium citrate` | 6224 |

The six unstereo amino-acid names (PRECISE-1K's `Supplement` cells) reach the `L-` rows
because PubChem's name index answers each bare name with the L-enantiomer record; the
Lamoureux loader deliberately leaves those labels unstereo, so this is PubChem's reading,
not the paper's, and is listed here for review.

**Lookups that are not the label as written** (each with its reason in the input list):
`Propandiamine` -> `1,3-diaminopropane` (Schmidt 2022's conditions table, the source study
of set29, lists "1,3 diaminopropane (N)"); `a-Ketoglutaric acid disodium salt hydrate`,
`alpha-ketoglutarate` (`α-`) and `beta-hydroxyisobutanoate` (`3-hydroxyisobutyrate`), which
the name index does not read as written; `Sodium D,L-Lactate` -> `sodium DL-lactate`;
`Na-citrate` -> `sodium citrate`; `N-Acetyl-D-Glucosamine` by the CAS Price 2018 states
(7512-17-6, CID 1738118), the record the Tong 2020 and PRECISE-1K spellings resolve to;
and `4-Aminobutyric acid` by `cid=119`, because the name index answers it with CID 223130,
whose structure `CC(CCC(=O)O)N` is 4-aminopentanoic acid (it answers
"4-aminopentanoic acid" with the same CID).

**The Price 2018 CAS cross-check.** Tables S2-S4 of `si3.xlsx` state a CAS number for 63
of the labels. Each was looked up: 54 give the same CID the curator adopted. Of the other
nine, two CAS numbers are absent from PubChem (Benzalkonium Chloride, D-Galacturonic Acid
monohydrate); four give a second PubChem record of the same substance and form (copper
chloride dihydrate, maltose monohydrate, mannose, nickel chloride hexahydrate), where the
adopted record is kept because it is the one another dataset's label already resolves to
or, for maltose and nickel, the one whose Title matches the label; and three name a
different FORM than the label, an internal inconsistency of the release that is recorded,
not resolved: `a-Ketoglutaric acid disodium salt hydrate` (CAS 305-72-6 is the anhydrous
salt), `Adenine hydrochloride hydrate` (2922-28-3, anhydrous), and `Sodium D,L-Lactate`
(50-21-5, lactic acid itself). The label's form is kept in all three.

**Pinene (#799).** Measured, not hypothesized: the name index answers `alpha-pinene` with
CID 6654, titled (+-)-alpha-Pinene, and bare `pinene` with CID 15837102, titled (-)-Pinene,
whose connectivity `CC1=C2CC(C2(C)C)CC1` is neither alpha- nor beta-pinene. So the two do
NOT collapse: `alpha-pinene` has its own RESOLVED row and `pinene`, the label Niu 2019
passes, is an UNRESOLVED_PUBLIC row recording why. Niu 2019's records are unchanged.

### Renames, and the one served dataset the rows change

None of the 178 labels resolved before, so a rename can only take a record's compound from
its release spelling to the row's canonical spelling. The store measurement over EVERY
mapped dev store (`store_reach_before.json`, yeast included) finds 48 such
(spelling, new name) pairs in 9 stores; 15 are case only (`Chloramphenicol` ->
`chloramphenicol`), the rest are folds or collapses (`D-Fructose` -> `fructose`, `Ribose`
-> `D-ribose`, `N-acetyl Glucosamine` -> `N-acetylglucosamine`). Every other store the
rows reach only gains identifiers under an unchanged name.

Of the 51 datasets the served KG v2.0 holds, exactly one is touched:
**EnvChemgenHoepfner2014Dataset**, whose records carry `D-Sorbitol` and `Potassium
Chloride` with InChIKeys derived from Hoepfner's own SMILES. A curated row outranks that
derivation, so after a rebuild `D-Sorbitol` becomes `sorbitol` (same InChIKey, CID 5780
filled) and `Potassium Chloride` becomes `potassium chloride` with PubChem's
`WCUXLLCKKVVCTQ-UHFFFAOYSA-M` in place of the `-N` key Hoepfner's `[K+].Cl` SMILES gives.
`scripts/schema_impact_check.py --base origin/main` reports "No schema contract changes"
and so names no dataset, because the compound table is not in the schema closure; that is
the #726 hazard, and it is why the store measurement exists. KG 4.0 is a full rebuild, so
this is in scope; it is not an increment.

## 2026.10.11 - The #726 pass rebased onto main: doxycycline hyclate joins the Parsons 2006 row, potassium acetate goes lowercase (#1008)

Main gained the Parsons 2006, Messner 2021 and Jackson 2020 rows (5,533 rows to 5,578,
sha256 `2771b19b...`) while the pass above was on its branch. Row by row the two sides are
disjoint, with one exception: both added a row named `doxycycline hyclate`, main's at
PubChem CID 54705095 (the record PubChem titles "Doxycycline hyclate", keyed by Parsons
Table S1's CAS 24390-14-5) and the pass's at CID 54686183 (titled Doryx, what the name index
answers). The table refuses a key claimed twice. Two input-list edits resolve it, each by a
rule the list already states, and the pass was then rerun against main's table
(`compound_curator_726_860.py --write` on a copy of the 2026-10-10 response cache, which
gained three PubChem responses: `$DATA_ROOT/tmp/pubchem_curator_cache_726_860_2026-10-11.json`,
437 responses, sha256 `6baf585f...`; input list sha256 `a9262d85...`):

- **`doxycycline hyclate | cid=54705095`.** Where two PubChem records are one substance, the
  record another label of it already resolves to is taken (the N-Acetyl-D-Glucosamine
  precedent), so the label folds into main's row and the salt stays one node. Girgis 2009
  already resolved through that row on main. Price 2018's Table S4 CAS, looked up by name,
  returns CID 54686183, so the CAS cross-check reads 53 of 63 in agreement; the tenth is a
  second record of the same substance and form, kept for the same reason as the other four
  of that kind.
- **`potassium acetate | canonical`**, lowercase. Main's De Chiara 2022 media spell it so
  (`SPORULATION_KAC` and the loader's YPA pre-culture), and a key spelled two ways takes the
  lowercase spelling, as `isopropanol` did. Price 2018 and Borchert 2024 records spelled
  `Potassium acetate` rename, case only. This also closes #1008: the KAc sporulation medium's
  potassium acetate now carries PubChem CID 517044 and `SCVFZCLFOSHCOH-UHFFFAOYSA-M` instead
  of a typed gap, so `SPORULATION_KAC`'s `media_identity` digest moves (re-pinned in
  `test_media_bacterial.py`) and `test_media_library.py` lists only cellobiose as gapped.
  De Chiara 2022 is not in the served KG v2.0.

Measured on the rerun: 178 labels, the same statuses (166 RESOLVED, 4 RESOLVED_MIXTURE, 3
UNDEFINED_MIXTURE, 5 UNRESOLVED_PUBLIC); 145 rows added and 27 labels folded; table 5,578 to
5,723 rows, sha256 `2771b19b...` to `77fbc85b...`, re-pinned in `_TABLE_SHA256`. The rerun's
table equals, byte for byte, a row-level three-way merge of the two sides without the
pass's doxycycline hyclate row, before the potassium acetate edit. Lookup-key diff against
main's table (`compound_identity_table_diff.py`): 177 keys added, 0 renamed, 0 lost, 0
removed, 0 identifier changes, so no existing row moved. The two edited labels reach, in
the branch's store scan (`store_reach_after.json`), only the Price 2018, Girgis 2009 and
Borchert 2024 stores, none of them served, so Hoepfner 2014 (`D-Sorbitol`, `Potassium
Chloride`) is still the only served dataset the pass touches.

Two loaders that landed on main after the store scan also reach the pass's rows, found by
their tests rather than by a scan (their stores were not rescanned): Peter 2018's `potassium
chloride` (the 2 M KCl condition) gains CID 4873 and `WCUXLLCKKVVCTQ-UHFFFAOYSA-M` in place
of a `deferred_pending_source_review` gap, and its bare `ribose` becomes `D-ribose` (CID
10975657), the reading PubChem gives the bare name and Tong 2020's `Ribose` already takes;
and De Chiara 2022's potassium acetate, above. Neither dataset is in the served KG v2.0.

The dev stores this branch rebuilt carry the pre-rebase rows for these two labels: the
Price 2018 RB-TnSeq and Girgis 2009 stores hold doxycycline hyclate at CID 54686183, and the
Price 2018 and Borchert 2024 stores hold `Potassium acetate`. Their L0-L4 tables describe
those stores as built at `cb310e339`; every dev store is stale again on main after later
schema changes, and the KG 4.0 full rebuild takes the merged table.
