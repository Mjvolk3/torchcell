# siroheme (s_1430) curation search log

Date: 2026-09-11. Curator: coding agent. Time box: 3 hours.

Artifact mirror, every file sha256 recorded in `siroheme.json`:
`$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/siroheme/`

## The row

| field | value |
|---|---|
| metabolite | `s_1430` siroheme [c] |
| GEM formula | C42H36FeN4O16 |
| GEM charge | -8 |
| blocked reaction | `r_0999` sirohydrochlorin ferrochelatase |
| prior exclusion reason | `no_structure_available` (accessions resolve to the iron-free macrocycle) |
| result | curated structure found and verified; no energy exists |

## Reaction and microspecies check, done first

Read straight from the model with cobra:

```
r_0999  s_0924 + s_1431 --> 2.0 s_0794 + s_1430
  -1 s_0924 iron(2+)          Fe               charge +2
  -1 s_1431 sirohydrochlorin  C42H38N4O16      charge -8
  +2 s_0794 H+                H                charge +1
  +1 s_1430 siroheme          C42H36FeN4O16    charge -8
check_mass_balance() -> {}
```

Both sides are C42H38FeN4O16 at net charge -6, so `r_0999` balances elementally and by
charge, and cobra agrees. The GEM's sirohydrochlorin formula is an exact match to
ChEBI:58351 sirohydrochlorin(8-).

Which microspecies gives exactly C42H36FeN4O16 at -8. Start from the neutral free acid,
siroheme C42H44FeN4O16 at charge 0, which is Fe(II) inside the dianionic
tetrahydroporphyrin macrocycle plus eight COOH side chains (four acetate, four
propionate). Remove all eight carboxylic protons: C42H44FeN4O16 minus 8 H gives
C42H36FeN4O16, and 0 minus 8 gives -8. Every unit of the net -8 is a carboxylate. The
iron center contributes zero because Fe(II) is compensated by the two deprotonated
pyrrole nitrogens; the chelatase reaction is exactly that swap, two pyrrole protons out
and Fe(II) in, which is why the reaction releases 2 H+. ChEBI writes the same bookkeeping
as `[Fe-2]` with two `[N+]` pyrrolic nitrogens, summing to the same zero core and the same
overall -8. The physical oxidation state is Fe(II).

## Sources kept

**ChEBI:60052 siroheme(8-)** is the record that carries the row. Exact match on both
fields the GEM cares about, and it names the pH:

> A cyclic tetrapyrrole anion obtained by deprotonation of the carboxy groups of siroheme;
> major species at pH 7.3.

Formula C42H36FeN4O16, net charge -8, average mass 908.607, InChIKey
DLKSSIHHLYNIKN-QIISWYHFSA-D, and a SMILES with iron coordinated inside the macrocycle.
Retrieved from the ChEBI backend API as JSON; the human-readable entity page was mirrored
alongside it.

**MetaNetX MNXM82173** (which redirects to MNXM1104857) independently carries
molecularFormula C42H36FeN4O16, the same InChIKey, and an Fe-coordinated SMILES. RDKit
canonicalizes the ChEBI and MetaNetX SMILES to the identical string. This matters for
diagnosis: `metanetx.chemical` is the accession the pipeline resolved through, and it is
not the thing that is wrong.

**PDB chemical component SRM**, the prior lead, checks out as described:
`_chem_comp.name SIROHEME`, `_chem_comp.formula "C42 H44 Fe N4 O16"`, formula weight
916.661, model coordinates from PDB 1AOP. Kept as corroboration for connectivity and the
7S,8S,12S,13S stereochemistry, and as the neutral free-acid reference point that the -8
species is derived from. Not used as the charge authority: `pdbx_formal_charge` is `?`,
and RDKit reads the ideal SDF at net -2 (neutral Fe plus two pyrrolide nitrogens), which
contradicts the entry's own neutral formula.

**KEGG C00748** carries only the neutral free acid, FORMULA C42H44FeN4O16, and
cross-links ChEBI: 28599 and PDB-CCD: SRM. Consistent, but it does not hold the
microspecies the pipeline needs.

**ChEBI:28599** (neutral siroheme, C42H44FeN4O16, charge 0) and **ChEBI:58351**
(sirohydrochlorin(8-), C42H38N4O16, charge -8) were pulled to anchor the derivation and
the reaction balance.

## Rejected, and why

**PubChem CID 70678579 "Siroheme hexaanion"**, the other prior lead, is rejected as the
structure source. Its formula string is C42H36FeN4O16-10 with `"Charge": -10`, and its
SMILES ends in `.[Fe]`, a disconnected neutral iron atom. RDKit reads two fragments at
charge -10. The charge does not match the GEM's -8, the iron is outside the molecular
graph, and the -10 convention implies Fe(0), which is not the chemistry here. Its
InChIKey BGDZOENYZCAFCO-QIISWYHFSA-D differs from ChEBI's precisely because its InChI
lacks the `/q;+2` iron-charge layer. The title "hexaanion" matches neither convention.

**ChEBI `saveStructure.do`**, both the molfile and the `sdf=true` form, returns the HTML
entity page rather than a structure file. Both responses are kept under `rejected/` so
the dead endpoint is on record. The structure came from the backend API JSON instead.

**Rhea 24363** could not be retrieved. WebFetch returns HTTP 403 and the `.rxn` endpoint
returns a Cloudflare interstitial; the JSON query endpoint returned count 0. Rhea
therefore carries none of the claims in this record. The pH-7.3 microspecies statement
rests on ChEBI's own definition field.

## Structure verification, run in the torchcell env

RDKit on the ChEBI SMILES:

```
formula   C42H36FeN4O16-8     (CalcMolFormula)
charge    -8                  (GetFormalCharge)
fragments 1                   (GetMolFrags)
carboxylate [CX3](=O)[O-] matches  8
carboxylic acid [CX3](=O)[OX2H1] matches  0
InChIKey  DLKSSIHHLYNIKN-QIISWYHFSA-D   (equals ChEBI's)
```

Formula and charge equal the GEM's exactly, the iron is inside a single connected graph,
and exactly eight carboxylates are deprotonated with none left protonated, which is the
microspecies assignment stated above.

## Decomposition outcome: it fails, and for two independent reasons

Called the same way `experiments/026-metabolism-flux/scripts/thermo_structure_attempt.py`
calls it, in the `equilibrator-ext` env against a private copy of
`~/.cache/equilibrator/compounds.sqlite`.

First, the group decomposer itself, called directly on the curated structure:

```
n group definitions: 163
Fe-bearing group SMARTS: []
ignore_protonations=False: FAILED GroupDecompositionError: Unable to decompose ... into groups
ignore_protonations=True:  FAILED GroupDecompositionError: Unable to decompose ... into groups
```

Not one of the 163 installed group definitions contains iron. Component contribution has
no group for Fe, so no Fe-containing macrocycle can be decomposed, and better retrieval
cannot change that.

Second, and separately, the `LocalCompoundCache` route does not actually test the curated
structure. `local.get_compounds([smiles], mol_format="smiles", bypass_chemaxon=True)`
resolves by InChIKey to the pre-existing cache compound 67540, and that entry is
internally inconsistent:

```
compound 67540  inchi_key DLKSSIHHLYNIKN-QIISWYHFSA-D   (correct siroheme key)
                inchi     ...C42H46N4O16.Fe.../q;+2/p-10 (correct siroheme InChI)
                mass      908.611                        (correct siroheme mass)
                smiles    iron-free macrocycle, no Fe atom
                atom_bag  {'C': 42, 'O': 16, 'H': 38, 'N': 4, 'e-': 454}   no Fe key
                group_vector  []   (empty list, not None)
                standard_dg_formation -> (None, None, None)
```

Right identifiers, wrong molecule. The formula the cache derives from that atom bag is
C42H38N4O16, which is why the pipeline scores `composition_mismatch` against the GEM's
C42H36FeN4O16 and records `no_structure_available`. The exclusion reason on this row is
accurate in its effect but the cause sits in eQuilibrator's cache, not in the GEM's
accessions. `cxcalc` is not installed in that env, so the cache is read-only and cannot
build a fresh compound from the curated SMILES even if asked.

One thing for whoever integrates these records: `thermo_structure_attempt.py` tests
`created.group_vector is not None`, which is true for an empty list. Here the composition
check catches the compound first, so nothing downstream was harmed, but the test should be
a length test.

## Energy: searched, and there is none

Seven independent lines, all negative.

- General search for a siroheme standard Gibbs energy of formation: nothing but generic
  thermodynamics tutorials.
- NIST TECRDB, which covers roughly 400 enzyme-catalyzed reactions from roughly 1000
  papers: no ferrochelatase and no siroheme entry surfaced. No TECRDB file ships with the
  installed equilibrator packages, so there was nothing to grep locally.
- Alberty-type tables: the tabulations cover 53 to 64 reactants at 298.15 K and pH 7;
  heme, protoheme and siroheme are not among them.
- BRENDA EC 4.99.1.4, the SirB crystal structure (Dalton Trans 2019) and the Met8p
  bifunctional-enzyme literature: mechanism, active-site residues His10, Glu43, His76,
  and kinetics, but no equilibrium constant and no reaction enthalpy.
- Iron porphyrin metalation thermodynamics: the closest hit, Chang and Fajer, J. Am.
  Chem. Soc. 1980, 102, 848, is about pi cation radicals of iron(II) isobacteriochlorin
  model compounds, not a metalation equilibrium. Nothing transferable.

So the record is `kind: structure`, `admission: tabulated`, `sigma_kj_per_mol: null`. No
number is claimed.

## What a human still has to decide

The GEM already prices the other three participants of `r_0999`: iron(2+) at -82.18
kJ/mol, sirohydrochlorin at -796.96 kJ/mol via the composition surrogate, and H+ at 0.0,
all read from `thermo_pipeline_yeast9.json`. One number for siroheme closes the reaction.

Since component contribution can never supply it, the choice is a surrogate rule, and
that is a modeling decision rather than a retrieval one. Hypothesis, untested: ferrous
insertion into a tetrapyrrole with release of the two pyrrole protons has a small standard
reaction energy, which would make dgf(siroheme(8-)) approximately dgf(sirohydrochlorin(8-))
plus dgf(Fe2+) minus twice dgf(H+), with the chelation term set to zero and a wide sigma.
I did not test this and it must be labeled a surrogate wherever it is used. The
alternatives are a quantum-chemistry formation energy for the Fe(II) isobacteriochlorin
core, which would be admission `computed` and stays out of the default route, or a
spectrophotometric metalation equilibrium on a model compound, which these searches
suggest has not been published.

Worth deciding separately: whether to report the cache defect upstream. Compound 67540's
stored SMILES and atom bag disagree with its own stored InChI and mass, and every
Fe-containing macrocycle in that cache is likely to carry the same defect.
