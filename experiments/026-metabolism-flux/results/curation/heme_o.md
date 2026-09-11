# heme_o search log

Row: yeast-GEM 9.0.2 metabolite `s_0812`, heme o, compartment `m`, formula `C49H56FeN4O5`,
charge `-2`. Blocked reactions `r_0531` (Heme O synthase, YPL172C) and `r_0530`
(heme O monooxygenase, YER141W or YDR376W and YPL252C). Exclusion reason on entry:
`no_structure_available`, with the pipeline detail
`thermo_structure_attempt outcome=composition_mismatch`.

Result: `kind = structure`, `admission = tabulated` (a curated ChEBI record, admitted as a
curated structure; the enum has no `curated` member). The structure alone does not price
the metabolite, because the component-contribution group decomposer refuses it.

Artifacts and hashes:
`/scratch/projects/torchcell-scratch/data/torchcell/thermo_equilibrator/curated/heme_o/`,
with `SHA256SUMS.txt` in that directory.

## What the GEM itself already carried

`yeast-GEM.yml` (sha256 `ecfe5f81...`, lines 8518 to 8522) gives `s_0812` a `smiles` field
of its own:

```
CC1=C(C2=CC3=NC(=CC4=C(C(=C([N-]4)C=C5C(=C(C(=N5)C=C1[N-]2)C=C)C)C(CCC=C(C)CCC=C(C)CCC=C(C)C)O)C)C(=C3CCC(=O)O)C)CCC(=O)O.[Fe+2]
```

That string is a porphyrin dianion and a separate `[Fe+2]` ion. RDKit reads it as two
fragments, formula `C49H58FeN4O5`, net charge 0. It is neither connected nor the right
charge, so it cannot satisfy the row. It is byte-identical to PubChem CID 15719509, which
is where it came from.

The elemental check that fixes the microspecies was run against the same file. For
`r_0531`, `C15H25O7P2(3-) + C34H30FeN4O4(2-) + H2O -> HO7P2(3-) + C49H56FeN4O5(2-)`
balances at `C49H57FeN4O12P2` and net charge -5 on both sides. For `r_0530`,
`C49H56FeN4O5(2-) + C21H27N7O14P2(2-) + O2 -> H2O + C49H55FeN4O6(3-) + C21H26N7O14P2(-)`
balances at `C70H83FeN11O21P2` and net charge -4 on both sides. The doubly deprotonated
propionate form is therefore the one the GEM's own stoichiometry requires, not a choice.

## Searched and found

| Source | What it gave | Verdict |
|---|---|---|
| ChEBI:60530 `ferroheme o(2-)` | `C49H56FeN4O5`, charge -2, Fe bonded to all four macrocycle nitrogens, standard InChI and InChIKey `FISPASSVCDRERW-ARQJTVBPSA-J` | ACCEPTED, this is the answer |
| ChEBI:36301 `ferroheme o` | the neutral conjugate acid, `C49H58FeN4O5`, same coordinated skeleton | accepted as the form to hand to a pKa predictor |
| RHEA:28070, EC 2.5.1.141 | `heme b + (2E,6E)-farnesyl diphosphate + H2O = Fe(II)-heme o + diphosphate`, participants `CHEBI:60344;CHEBI:175763;CHEBI:15377;CHEBI:60530;CHEBI:33019` | identity proof: the reference reaction for the enzyme behind `r_0531` uses CHEBI:60530 |
| PDB CCD `HEO` | `C49 H58 Fe N4 O5`, Fe bonded into the ring, InChI connectivity layer identical to ChEBI's | independent structural confirmation |
| PubChem CID 15719509 | the disconnected form the GEM copied | documents the defect |
| ChEBI:60344 `ferroheme b(2-)` | `C34H30FeN4O4`, charge -2 | matches the GEM's `s_0712`, so both microspecies conventions agree |

ChEBI defines the -2 species as "The cyclic tetrapyrrole anion that is ferroheme o
protonated to pH 7.3." That is within 0.1 pH unit of the pipeline's pH 7.2 and is not a
source of ambiguity for a pair of carboxylates.

Stereochemistry: (1S) at the carbinol carbon, (4E,8E) in the farnesyl-derived chain. The
E assignment is the one consistent with the (2E,6E)-farnesyl diphosphate named in
RHEA:28070. The PDB entry records the 4-ene as Z, which is why its InChIKey
(`FRKORVCRVCLRBA`) differs from ChEBI's while the connectivity layer is identical.

## Searched and rejected

**Metabolomics Workbench RegNo 163805.** The prior pass offered this as a structure
source. It is wrong. Its formula is `C49H60FeN4O5`, two hydrogens heavier than heme O,
and its systematic name reads `22,23-dihydroporphyrin`, which is a chlorin, a reduced
macrocycle, not a porphyrin. Its iron is a disconnected `iron(2+)` and its SMILES field
contains a literal U+0004 control character. Not usable.

**ChEBI:24480 `heme o`,** the accession the GEM annotates. The API returns
`default_structure: null`. The accession carries no structure at all, which is the
literal grounding for the `no_structure_available` exclusion reason.

**HMDB0001162.** Could not be retrieved. `curl` on the `.xml` endpoint returns a
Cloudflare interstitial and `WebFetch` on the HTML page returns HTTP 403. The lead was
not needed: ChEBI, PubChem, the PDB and Rhea agree on the skeleton independently. Adding
this cross reference would take a manual browser session.

## RDKit verification

Run under `~/miniconda3/envs/torchcell/bin/python`
(`rdkit_verify.py`, sha256 `79ad343a...`; log `rdkit_verify.log`, sha256 `ead11519...`).

```
formula            = C49H56FeN4O5-2
formal charge      = -2
n fragments        = 1
Fe explicit degree = 4
```

Formula and charge equal the GEM row exactly, iron is inside the molecular graph, and the
molecule is a single connected fragment. RDKit regenerates the InChIKey as
`FISPASSVCDRERW-SCCPKUNWSA-J`. The connectivity block and the charge block agree with
ChEBI; only the stereo block differs, because RDKit drops the ten macrocycle ring
double-bond descriptors ChEBI records. Stereochemistry does not enter the group
decomposition, so this is cosmetic, but it is recorded so a later hash comparison is not
read as a mismatch.

## Decomposition check

The decomposer was called the way
`equilibrator_assets.generate_compound._populate_compound_information` calls it,
`GroupDecomposer().Decompose(mol, ignore_protonations=False, raise_exception=True)`,
under `~/miniconda3/envs/equilibrator-ext/bin/python`
(`decomposition_probe.py`, sha256 `85f33e81...`).

```
ferroheme_o_2minus_CHEBI_60530   FAILED  GroupDecompositionError: Unable to decompose ... into groups
ferroheme_o_neutral_CHEBI_36301  FAILED  GroupDecompositionError: Unable to decompose ... into groups
```

The full `LocalCompoundCache(pkas_provided_externally=True).get_compounds(...,
bypass_chemaxon=True, specified_pkas=...)` route that `lipid_structure_writer.py` uses
returns status `failed` with no compound created.

Scope of the failure was measured rather than assumed
(`decomposition_probe_context.py`, sha256 `2a5a0748...`):

```
acetate_control                             DECOMPOSED
farnesol_control                            DECOMPOSED
protoporphyrin_IX_free_base_CHEBI_15430     FAILED
ferroheme_b_2minus_CHEBI_60344              FAILED
ferroheme_o_2minus_CHEBI_60530              FAILED
heme_o_iron_free_macrocycle_2minus          FAILED
```

The barrier is the porphyrin macrocycle, not the coordinated iron. The iron-free
macrocycle fails as well, and so does ferroheme b.

## What is left for a human

1. The pipeline reason for `s_0812` should move from `no_structure_available` to
   `group_decomposition_failed`. That reclassification is the measured contribution of
   this record beyond the SMILES.
2. Pricing heme o alone unblocks neither reaction. `r_0531` also needs `s_0712`
   ferroheme b, currently `no_pka_prediction` with final blocker
   `structure_absent_from_cache`, and `r_0530` also needs `s_0811` heme a. Porphyrins
   need a decision as a class.
3. The defensible route is a conserved-moiety pseudo-group. The Fe-porphyrin macrocycle
   is conserved across `r_0530` and `r_0531`, so adding it to the group incidence matrix
   as a black-box group makes both reaction energies well defined without asserting a
   formation energy for the macrocycle. This is a pipeline code change and needs the
   user's sign-off, since it changes what the default route may do.
4. A literature formation energy is the alternative and searching did not find one. No
   calorimetric or equilibrium value for any Fe-porphyrin surfaced; no heme reactant
   appears in the Alberty and Goldberg transformed-formation tables, which cover 53
   reactants at 298.15 K, pH 7 and ionic strength 0, 0.1 and 0.25 M; no measured Keq
   exists for EC 2.5.1.141, so a `reaction_energy` record for `r_0531` is unavailable
   too. Any surrogate would be computed, not measured, and stays out of the default
   route under the stated admission rule.
5. No pKa values are supplied. The only number available for the two propionates would
   be a generic carboxylate estimate, which is a provisional single-source value.
