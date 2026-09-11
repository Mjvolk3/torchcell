# mannan (s_1107 [c], s_1108 [er]) -- curation search log

Curated 2026-09-11. GEM: yeast-GEM 9.0.2, formula `C6H10O5`, charge 0.
Artifacts and their sha256 live in
`/scratch/projects/torchcell-scratch/data/torchcell/thermo_equilibrator/curated/mannan/`
and are cited from `mannan.json`.

## Result in one line

`kind = polymer_convention`, `admission = tabulated`. The species is ONE
alpha-(1->6)-linked anhydromannose residue, SMILES
`*OC[C@H]1O[C@H](*)[C@@H](O)[C@@H](O)[C@@H]1O`, RDKit-verified as `C6H10O5`, charge 0.

## Why the two existing routes missed it

Both refusals are recorded in the pipeline's own artifacts, and both are the guard
working rather than a bug.

1. **Structure route.** `thermo_structure_attempt.json` says
   `no_structure_in_any_namespace` for both metabolite rows. Cause now pinned down: every
   accession yeast-GEM cites for mannan is a structureless class term. ChEBI:28808 returns
   `definition: null`, `chemical_data: null`, `default_structure: null`; KEGG C00464 has no
   `FORMULA` field and no `ATOM` block; and in the local eQuilibrator cache the
   corresponding compounds (ids 1798 and 28338) have empty `smiles`/`inchi` and a NULL
   `group_vector`.
2. **Repeat-unit route.** `thermo_polymer_repeat_unit.json` says
   `no_hydrolysis_reaction_identifies_a_monomer`. Its precondition PASSED
   (`stoichiometry_is_per_repeat_unit = true`, `unbalanced_reactions = []`). What failed is
   the monomer lookup: the route reads the monomer off a model reaction shaped
   `species + n H2O -> n monomer`, and yeast-GEM 9.0.2 carries no mannanase or mannosidase
   acting on the mannan pool. This record supplies that monomer.

## Searches, in order

| # | Query or endpoint | Engine | Outcome |
|---|---|---|---|
| 1 | Rhea 49408 mannan H2O(C6H10O5)n | WebSearch | No direct hit; polymer notation confirmed in general use |
| 2 | rhea-db.org REST, RHEA:49408, tsv | curl | RHEA:49408 is a **(1->2)-beta**-D-mannan phosphorylase (CHEBI:59573). The lead's linkage is wrong for S. cerevisiae |
| 3 | ChEBI 28808 mannan polysaccharide definition | WebSearch | Surfaced CHEBI:59573 and general literature; pointed at the API |
| 4 | ChEBI legacy SOAP `getCompleteEntity` CHEBI:28808 | curl | **FAILED**, HTTP 500 EBI error page. Switched endpoints |
| 5 | ChEBI backend API CHEBI:28808 | curl | Class term, no definition, no formula, no structure. The decisive negative |
| 6 | `ontology_relations` of CHEBI:28808 | local read | Children: 27912 (1->6)-alpha, 59573 (1->2)-beta, 27857 (1->4)-beta, 61300 branched yeast repeat |
| 7 | ChEBI backend API CHEBI:27912 | curl | **HIT.** `(C6H10O5)n`, charge 0, SMILES with two attachment points. Adopted |
| 8 | rest.kegg.jp C00464 | curl | No FORMULA, no ATOM block |
| 9 | rest.kegg.jp C01654 | curl | **HIT.** "1,6-Mannan", `FORMULA (C6H10O5)n`, two `Z *` attachment atoms, `BRACKET ... n`. Independent corroboration of 27912 |
| 10 | rest.expasy.org enzyme 2.4.1.109 | curl | **FAILED**, host not resolvable from this machine |
| 11 | rest.kegg.jp ec:2.4.1.109 | curl | r_0362 is protein O-mannosyltransferase; one mannosyl per turnover |
| 12 | PMC10958358 | WebFetch, then grep on the stored HTML | "Yeast mannan has an alpha-1,6-linked backbone and alpha-1,2- and alpha-1,3-linked mannose branches" |
| 13 | Tewari Goldberg mannobiose hydrolysis | WebSearch | **NEGATIVE for mannobiose.** They measured cellobiose, gentiobiose, isomaltose, maltose only |
| 14 | eQuilibrator polysaccharide residue group contribution | WebSearch | Background only; settled against the local cache instead |
| 15 | cache lookup: mannan, starch, glycogen accessions | sqlite3, read-only | **No residue-level estimate exists.** Mannan and starch have no structure; glycogen's is a whole C24H42O21 tetrasaccharide |
| 16 | cache lookup: `inchi_key LIKE 'DLRVVLDZNNYCBX%'` | sqlite3, read-only | **HIT.** alpha-1,6-mannobiose present with a group vector (ids 691514, 47650, 460418) |

## The convention, and why alpha-(1->6)

Two attachment points, exactly as ChEBI:27912 and KEGG C01654 write it: one at the
anomeric carbon C1 (the acetal oxygen belongs to the next residue) and one on the O6
oxygen (which belongs to this residue). A chain of n residues is `(C6H10O5)n`; hydrolyzed
at both ends it is `H2O(C6H10O5)n`, the form ChEBI gives CHEBI:59573. yeast-GEM has no
chain-length variable and no reducing-end species, so no terminal water is added and the
GEM species is exactly one residue.

Alpha-(1->6) is chosen because it is the backbone of S. cerevisiae cell-wall mannan
(PMC10958358) and the only bare-mannan linkage ChEBI writes a repeat unit for. Two
readings of the GEM agree that the species is one residue.

- **Mass balance.** r_0362 is `C26H47O9P -> C20H37O4P + C6H10O5`, which balances element
  for element on C, H, O and P. The pipeline's own check lists r_0362 among s_1108's
  balanced reactions.
- **Structure.** ChEBI:61300, the branched yeast pentamannoside repeat, has formula
  `(C30H50O25)n = 5 x C6H10O5`, so the per-mannose formula is `C6H10O5` whether the
  alpha-1,2 and alpha-1,3 branches are counted or not.

**The caveat, stated rather than hidden.** r_0362 carries EC 2.4.1.109, protein
O-mannosyltransferase, whose real product is 3-O-(alpha-D-mannosyl)-L-seryl or
-threonyl-[protein]: an O-mannosyl residue on a protein hydroxyl, not an alpha-1,6 chain
extension. yeast-GEM lumps that into the same pool the carbohydrate pseudo-reaction r_4048
draws cell-wall mannan from. Both readings give the same `C6H10O5` residue and both attach
the anomeric carbon through an ether oxygen to an alcohol, so at this surrogate's
resolution the choice does not move the number.

## Can eQuilibrator price it directly? No, and the negative is measured

`equilibrator_assets.group_decompose.GroupDecomposer` REJECTS the attachment-point SMILES
with `GroupDecompositionError: Unable to decompose ... into groups`. In the same call it
SUCCEEDS on alpha-D-mannopyranose (5 nonzero groups) and on closed-ring
alpha-1,6-mannobiose (7 nonzero groups), so the failure is the dummy atoms, not the sugar.
There is likewise no group-contribution estimate anywhere in the cache for a polysaccharide
residue as such: mannan and starch have no structure at all, and glycogen's only structure
is a whole tetrasaccharide.

## The two candidate values

Component contribution over a copy of the shipped cache, I = 0.25 M, pMg 3.0,
T = 303.15 K. D-mannose reprices to -401.6376 at pH 7.2, identical to the pipeline's
s_0571, so the conditions reproduce the pipeline exactly.

| Route | s_1107 [c], pH 7.2 | s_1108 [er], pH 7.1 | sigma |
|---|---|---|---|
| Drop-in for the existing route: `G(D-mannose) - G(H2O) + 6.0504` | -242.640 | -249.601 | 3.648 |
| **Preferred**, dimer difference: `[G(alpha-1,6-mannobiose) - G(H2O)] / 2` | **-248.704** | **-254.505** | **1.439** |

The drop-in needs only a curated monomer map (`c -> s_0571`, `er -> s_3933`,
`n_residues = 1`) and no code change; it uses the shared correction calibrated on six
disaccharides, none of them a mannoside, and the route's single cytosolic water for both
compartments. The dimer difference is the same algebra with a linkage- and sugar-matched
calibration substrate, and it is what I recommend. The two differ by 6.1 and 4.9 kJ/mol,
1.7 and 1.3 sigma of the looser one.

Reported sigma 1.439 is the component-contribution propagation (1.212) in quadrature with
the measured spread of the three (1->6) dimer-difference residues (0.775).

## One anomaly, undiagnosed

In the HYDROLYSIS view, component contribution puts the alpha-1,6-mannosidic bond at
**+0.027 kJ/mol**, while its own alpha-1,6-glucosidic (isomaltose) value is -8.427 and its
beta-1,6 (gentiobiose) value is -7.242. The latter two sit close to the measured -7.06 and
-7.15 kJ/mol implied by Tewari and Goldberg 1989 (K = 17.25 and 17.9 at 298.15 K), so
mannobiose is a 7 to 8 kJ/mol outlier there.

In the RESIDUE view the outlier largely disappears: -248.704 (mannobiose) against -247.166
(isomaltose) and -247.759 (gentiobiose), because the residue is half a dimer and the
monomer never enters. That is the argument for the dimer-difference form. **I have not
diagnosed the cause of the hydrolysis-view outlier** and it should not be explained away
without measurement.

## What a human still needs to do

1. Pick between the two values above. My recommendation is the dimer difference.
2. Decide whether the two compartment copies keep separate values. They differ by 5.8
   kJ/mol purely through pH 7.2 versus 7.1; I followed the pipeline's existing convention
   of pricing compartment copies separately.
3. Look at two GEM defects found in passing.
   - yeast-GEM's own `smiles` field for s_1107/s_1108 reads as **C24H42O21**, a
     four-mannose oligosaccharide, contradicting the entry's own `C6H10O5` by a factor of
     four. Nothing in the pipeline reads that field today.
   - **Live, and not this row: glycogen.** s_0773/s_0774 carry the identical defect
     (formula `C6H10O5`, smiles a C24H42O21 tetrasaccharide) and they ARE priced, by route
     `cache_accession`, at -1127.25 and -1370.87 kJ/mol. Those are tetrasaccharide
     energies sitting on a one-residue formula; a single glucose residue prices near -248
     by the same dimer difference. The accession route appears to apply no composition
     check. Same failure mode as mannan, already in the served numbers.
4. Optional and probably not worth it: the alpha-1,2 and alpha-1,3 branch residues are not
   distinguished. All three linkages give the same `C6H10O5` residue, and the 0.775 kJ/mol
   spread of the (1->6) family bounds what a per-linkage treatment could buy.

## Files written

- `/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/results/curation/mannan.json`
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/results/curation/mannan.md`
- `/scratch/projects/torchcell-scratch/data/torchcell/thermo_equilibrator/curated/mannan/`
