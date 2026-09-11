# heme a curation log

Date: 2026-09-11. Row `heme_a`, yeast-GEM 9.0.2 metabolites `s_0811` (m), `s_3714` (c),
`s_4332` (e). Declared formula `C49H55FeN4O6`, declared charge `-3`. Exclusion reason
`no_structure_available`. Blocked reaction that needs the structure: `r_0530`, heme O
monooxygenase.

Machine-readable record: `experiments/026-metabolism-flux/results/curation/heme_a.json`.
Artifacts, hashes and the verification script:
`$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/heme_a/`.

## Result in one line

An Fe-coordinated, RDKit-verified structure was found and recorded. The GEM's charge of
`-3` is not reachable by any protonation state of heme a and the GEM's declared values
are internally inconsistent with its own SMILES; the structure does not decompose under
eQuilibrator's component contribution, so `r_0530` remains unpriced.

## What was searched

### Accepted

| what | where | outcome |
|---|---|---|
| CHEBI:61715 `ferroheme a(2-)` | ChEBI backend API | HIT. C49H54FeN4O6, charge -2, Fe bonded into the macrocycle. The parent of the recorded structure. |
| PubChem CID 146026556 `Ferroheme a(2-)` | PUG REST | HIT. C49H54FeN4O6-2, InChIKey `ZGGYGTCPXNDTRV-YRJWGCQVSA-J`. Independent agreement with ChEBI. |
| ExPASy ENZYME EC 1.17.99.9 | flat file | HIT. `Fe(II)-heme o + 2 A + H2O = Fe(II)-heme a + 2 AH2.` Fixes the iron oxidation state and the oxygen source. |
| BiGG universal metabolite `hemeA` | BiGG API | HIT, as evidence of the defect. `"name": "Heme A C49H55FeN4O6", "charges": [-6, -3, -2]`. |
| yeast-GEM 9.0.2 `model/yeast-GEM.yml` | local mirror | HIT. The GEM's own record, including a SMILES that contradicts its own declared formula and charge. |
| CHEBI:24479 `heme a` | ChEBI backend API | HIT, as evidence of the gap. `"default_structure":null`, formula C49H56FeN4O6, charge 0. This is the accession the GEM cites, and it carries no structure. |

### Rejected, and why

| what | why rejected |
|---|---|
| `chebi/saveStructure.do?...chebiId=24479` | Returns an 829 kB HTML page, not an SDF. The legacy endpoint no longer serves structures. Kept as `chebi_saveStructure_24479_FAILED_html.html`. |
| ChEBI SOAP `getCompleteEntity` | HTTP 500. The working route is `https://www.ebi.ac.uk/chebi/backend/api/public/compound/<id>/`. |
| HMDB0006901 (a lead from the prior pass) | `hmdb.ca` serves a Cloudflare interstitial to curl, so nothing is retrievable there. Not needed: the lead's Fe-containing C49H56FeN4O6 is the neutral species, which ChEBI:24479 and PubChem CID 46173807 both confirm. |
| MetaCyc CPD-13734 | The BioCyc compound page loads its Reactions panel by script; the fetched HTML holds only `Loading...`. Superseded by ExPASy. |
| ChemSpider 21865245 `ferroheme a` (a lead from the prior pass) | Not pursued. Scriptable retrieval needs an RSC API key, and two independent curated sources already agree on formula, charge and InChIKey. Recorded so the gap is visible. |
| Every PubChem and BiGG SMILES for heme a | Each writes the iron as a disconnected `.[Fe+2]`. Composition is right, coordination is not, and a disconnected metal is what the task rules out. |
| Formation energy searches (5 queries: calorimetric, Alberty/BasicBiochemData2, group contribution, eQuilibrator, metalloporphyrin) | NEGATIVE. No measured or tabulated standard formation energy for heme a exists in any of them. |
| Measured redox potentials of heme a | Real measurements exist, Em,7 near 380 mV with the companion heme oxidized and near 230 mV with it reduced, but they describe heme a bound inside cytochrome c oxidase and electrostatically coupled to heme a3. That is a protein-environment property of a redox step; `r_0530` is a formylation, not an iron redox change, so the number cannot price it. |

The full query list with per-query outcomes is in the `searches` array of the JSON.

## The structure

Recorded species is the ferroheme a monoanion, derived from CHEBI:61715 by protonating the
propionate on the formyl-bearing pyrrole ring. Which propionate carries the proton is
arbitrary and is documented as such.

Verified in the torchcell env with RDKit 2026.03.6:

```
neutral_heme_a                        C49H56FeN4O6    charge  0   ZGGYGTCPXNDTRV-YRJWGCQVSA-L
monoanion_heme_a (GEM formula match)  C49H55FeN4O6-   charge -1   ZGGYGTCPXNDTRV-YRJWGCQVSA-K
dianion_ferroheme_a(2-) CHEBI:61715   C49H54FeN4O6-2  charge -2   ZGGYGTCPXNDTRV-YRJWGCQVSA-J
```

All three are one connected fragment with four Fe-N bonds. The dianion's key reproduces
PubChem CID 146026556 exactly, which is what validates the InChI generation procedure
(standard InChI always disconnects the metal, so the four Fe-N bonds are broken and Fe is
set to +2 before the InChI is emitted).

Reproduce with
`$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/heme_a/verify_heme_a_structure.py`:

```bash
~/miniconda3/envs/torchcell/bin/python verify_heme_a_structure.py rdkit
~/miniconda3/envs/equilibrator-ext/bin/python verify_heme_a_structure.py equilibrator
```

Write the SMILES with plain bonds, the way ChEBI does. RDKit's canonical output for the
same molecule uses dative-bond syntax (`<-`, `->`), and OpenBabel, which is what
eQuilibrator reads with, rejects it outright: `SMILES string contains a character '<'
which is invalid`.

## Why the charge could not be matched

The task asked for `C49H55FeN4O6` and `-3`. The formula is matched exactly; the charge is
`-1`.

The Fe(II)-porphyrinate core is charge-neutral, Fe(2+) against a macrocycle dianion, so
every microspecies in this family satisfies `charge = -(56 - nH)`. That gives 0, -1 and -2
for 56, 55 and 54 hydrogens and nothing else. There is no microspecies with 55 hydrogens
and charge -3, and none was manufactured by writing a disconnected iron or a formal-charge
trick.

Three independent lines say the GEM is wrong rather than unusual.

1. The GEM's own SMILES for `s_0811` ends in `.[Fe]`, a disconnected and formally
   zero-valent iron beside a porphyrin dianion with both propionic acids. RDKit reads that
   string as `C49H56FeN4O6`, charge -2, which agrees with neither the declared formula nor
   the declared charge. The sibling heme o uses `.[Fe+2]` instead, so the two hemes are not
   even written in one convention.
2. Heme o (`s_0812`, C49H56FeN4O5, -2) is exactly ferroheme o(2-), the correctly
   deprotonated ferrous dianion. The model's convention is the dianion, and heme a is the
   odd one out.
3. EC 1.17.99.9 writes the reaction as `Fe(II)-heme o + 2 A + H2O = Fe(II)-heme a + 2 AH2`,
   so substrate and product share an iron oxidation state and heme a must be the ferrous
   dianion too.

Proposed correction, which needs a human decision and must not be applied silently:
`s_0811`, `s_3714` and `s_4332` become `C49H54FeN4O6`, charge -2, matching CHEBI:61715.

Applying it breaks `r_0530` as currently written, `heme o + NADH + O2 -> H2O + heme a +
NAD`, which balances only because heme a carries the non-physical pair; with the corrected
formula the equation is short one hydrogen and one negative charge. That is a real defect
in `r_0530`, not a reason to keep the bad formula. `r_0530` is annotated `eccodes:
1.18.1.6`, a ferredoxin-NADP+ reductase, while EC 1.17.99.9 says the oxygen comes from
water and the step is an oxidation consuming two oxidizing equivalents rather than an O2-
and NADH-consuming monooxygenation; the reaction as written also lumps two successive
hydroxylations and a dehydration into one step. Fix the formula and the reaction in the
same change.

## Decomposition

`GroupDecomposer().Decompose(Molecule.FromSmiles(smiles), ignore_protonations=False,
raise_exception=True)`, the same call `equilibrator_assets/generate_compound.py:119` makes,
run under `~/miniconda3/envs/equilibrator-ext/bin/python`.

FAILS, for all three microspecies and also for the Fe-free BiGG cache structure the
pipeline currently reaches. Five atoms are left unassigned every time, atomic numbers
`[26, 7, 6, 7, 6]`: the iron, two of the four macrocycle nitrogens and two ring carbons.
`ignore_protonations=True` fails identically. Component contribution has no group for the
Fe-porphyrin core, so the protonation state is not the obstacle and the missing iron was
not the obstacle either.

The `LocalCompoundCache.get_compounds` route that `thermo_structure_attempt.py` uses could
not be exercised: in that environment it returns `status='failed'` with `method='read-only'`
because cxcalc is not installed. That is an environment limitation, not a verdict on the
structure, and the direct decomposer call above is the decisive test.

## What this buys, and what it does not

It closes `no_structure_available` for the three metabolites and moves them to
`group_decomposition_failed`, a more honest reason but not a price. No structure and no pKa
ladder will price heme a through the default route.

## What a human still needs to do

1. Decide on the formula and charge correction and on `r_0530`'s stoichiometry, and take it
   upstream to SysBioChalmers/yeast-GEM if confirmed. BiGG's `Heme A C49H55FeN4O6` with
   charges `[-6, -3, -2]` is the shared root, so the fix belongs there too.
2. Check Alberty's BasicBiochemData2 directly rather than through a web search before
   calling the absence of a tabulated formation energy final.
3. Choose a surrogate deliberately. Hypothesis (untested): heme o and heme a are the same
   molecule differing by a single CH3 to CHO at C-8, so `r_0530`'s transformed reaction
   energy may be estimable from a decomposable methyl-to-formyl analog pair, letting the
   undecomposable macrocycle cancel the way a conserved cofactor group does. That would
   price the reaction without pricing either heme. It has not been tested and must be
   validated against something measured before admission.
4. If a formation energy for heme a itself is ever wanted, expect only a computed value,
   which under the project's admission rule is recorded as `computed` and stays out of the
   default route.

## Artifacts

All under `$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/heme_a/`, hashes in
`SHA256SUMS`.

| file | sha256 |
|---|---|
| `chebi_61715.json` | `dff387d569e7371e120d90cfe9bd0b797dcf549ad6dd9ed46ea747458ef580bb` |
| `chebi_24479.json` | `df70390ced431fbfbdca401c23f68848f06685bc2dceefb1e418483db8f71898` |
| `chebi_24480_hemeo.json` | `033118abbd928733c15d2484a9f7d55dc924b2927bf85130f4feaabe86708aef` |
| `pubchem_hemeA_props.json` | `1ef679b5d176996f438afa8e4b64dd2e2039a9ea84b6cf01acf913930833d5f1` |
| `pubchem_hemeA_search.json` | `3aa6a973803d9aa87bd682741cb5935afdb17c83e1cc363f0685b869df9bd698` |
| `pubchem_cid146026556_record.json` | `d5593b0fb27d5c5da35f6e2a2def63bd439188faf62b40135b4f10d82fd22e02` |
| `bigg_hemeA.json` | `a93d5f6fb7a5d8481140b7a02dff918fa871c841f6ce404cfd0c55d2571fb348` |
| `expasy_EC_1.17.99.9.txt` | `bb667140ca7883df7da0d457f2713b505e9f5daaca468fa35ac8473faf40fb0c` |
| `yeastgem_902_heme_a_records.txt` | `21a58cc1a75b717ae2267c9f325ef4af864c79fbe2dde8c5545cdbbd50a96bb0` |
| `verify_heme_a_structure.py` | `30aacf5613ec5d8f7a02ae24a470d7d22ca541c0635c9d314a951933865c7565` |
| `verify_heme_a_structure.out` | `550de54926504ccaf94b7152beb56765b1f36c3113884ce5cfb5e6fc9c6dd760` |
| `chebi_saveStructure_24479_FAILED_html.html` | `1b6dac6fe30c57320e4361ac5dc36bb5e88cac1bd5a0acaf9a8f2709da623d57` |
| `chebi_ws_24479.xml` | `353c0504c258b849476a7c7b51b2b6c933e34da02742653adc635c25cba1cc45` |
| `hmdb_HMDB0006901.xml` | `2626ddae87a92be5735265f808519881024471b41cf95fc7977c949d94e340de` |

Source model file:
`$DATA_ROOT/data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.yml`, sha256
`ecfe5f816471f2f62921c9448cdd5bf9803263aaf9aa91da02d1969077786aad`.
