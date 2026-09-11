# pectin (s_1309) search log

Curated 2026-09-11. Metabolite `s_1309` pectin [e], yeast-GEM 9.0.2 (`yeast-GEM.xml`
sha256 `9fd2c572cace73c2ea835205617313554d1bf89f4ef077f49defbdfa219a4ad7`). Shipped
formula `C6H7O6`, shipped charge `0`. Blocked reactions `r_0365` (endopolygalacturonase,
EC 3.2.1.15, YJR153W) and `r_1999` (pectin exchange, closes by the exchange rule and is
not evidence here).

Record: `experiments/026-metabolism-flux/results/curation/pectin.json`.
Artifacts, each hashed: `$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/pectin/`.

## Result in one paragraph

`r_0365` balances elementally as shipped and fails only on charge, by exactly one unit.
The residue the GEM means is tabulated in ChEBI as `CHEBI:140523`
`[(1->4)-alpha-D-galacturonosyl](n)`, formula `(C6H7O6)n.H2O`, charge `-1`, so the GEM's
elemental formula is right and only `fbc:charge` is wrong. Setting it to `-1` balances
`r_0365` exactly and satisfies the precondition the repeat-unit surrogate route refused
on. No measured or tabulated formation energy exists for pectin in any source searched.

## Step 1, does r_0365 balance as shipped

Read with cobra from the GEM. `H2O + pectin --> D-galacturonate`, that is
`H2O` (`H2O`, 0) + `s_1309` (`C6H7O6`, 0) gives `s_0560` (`C6H9O7`, -1).

- Elements: `C6H7O6 + H2O = C6H9O7`. Exact. `check_mass_balance()` reports no element key.
- Charge: substrates 0, product -1. `check_mass_balance()` returns `{'charge': -1.0}`.

So this row is a charge correction plus a polymer convention, and not a formula
correction in the elemental sense.

## Step 2, why pectin did not take the repeat-unit surrogate route

`thermo_polymer_repeat_unit.py` refuses any species whose non-boundary reactions do not
mass balance, because a residue energy computed against per-chain stoichiometry would be
wrong by the chain length. The existing `thermo_polymer_repeat_unit.json` therefore
records `s_1309` with outcome `stoichiometry_not_per_repeat_unit` and detail
`mass balance fails on r_0365; a residue energy would be wrong by the chain length`.

The refusal was correct. The diagnosis was not: the stoichiometry IS per repeat unit and
the imbalance is a charge annotation, not a chain length. The script's monomer finder
would have succeeded on its own, since `r_0365` has exactly the shape it looks for
(`species + n H2O -> n monomer`, n = 1) and `s_0560` is priced.

## Step 3, searches

| # | query | engine | outcome |
|---|---|---|---|
| 1 | ChEBI polygalacturonate polymer entry galacturonate residue formula charge | WebSearch | CHEBI:62969, CHEBI:60190, CHEBI:58658. None is the GEM residue |
| 2 | Rhea endopolygalacturonase EC 3.2.1.15 reaction ChEBI polygalacturonate | WebSearch | ExPASy and BRENDA descriptions only |
| 3 | `ec:3.2.1.15` | Rhea TSV API | NEGATIVE, header row only. Rhea curates no EC 3.2.1.15 reaction |
| 4 | `polygalacturonate` | Rhea TSV API | RHEA:13573, polymer participant CHEBI:140523. The decisive lead |
| 5 | `pectin`, `galacturonan` | Rhea TSV API | NEGATIVE for both |
| 6 | `polygalacturonate` | ChEBI es_search | CHEBI:25860 (class, no formula), CHEBI:62969 |
| 7 | `galacturonosyl` | ChEBI es_search | DECISIVE. CHEBI:140523, 15446, 140522, 141004, 68837 |
| 8 | NIST TECRDB polygalacturonase equilibrium constant galacturonide hydrolysis | WebSearch | NEGATIVE, no apparent equilibrium constant |
| 9 | Gibbs energy of formation pectin polygalacturonic acid kJ/mol | WebSearch | NEGATIVE |
| 10 | enthalpy of combustion polygalacturonic acid pectin calorimetry | WebSearch | NEGATIVE |
| 11 | PMC5153499 full text | WebFetch + PMC efetch | REJECTED, activation and denaturation only |
| 12 | PubMed 20668400 abstract | NCBI efetch | REJECTED, activation and binding parameters only |
| 13 | nine accessions in the eQuilibrator compound cache | equilibrator_api | See below |

## Step 4, what the namespaces say

| entity | formula | charge | verdict |
|---|---|---|---|
| CHEBI:140523 `[(1->4)-alpha-D-galacturonosyl](n)` | `(C6H7O6)n.H2O` | -1 | ADOPTED. Matches the GEM formula with the charge corrected |
| CHEBI:15446 same name, neutral | `(C6H8O6)n.H2O` | 0 | The protonated residue. Rules out charge 0 on `C6H7O6` |
| CHEBI:140522 methyl ester residue | `(C7H10O6)n.H2O` | 0 | Fixes degree of methyl esterification at 0 |
| CHEBI:62969 polygalacturonic acid | `(C6H8O6)n.C12H18O13` | 0 | Confirms the `(C6H8O6)n` repeat. Depicted only as a trimer |
| CHEBI:60190 homogalacturonan | `(C17H22O15)n` | 0 | REJECTED, esterified and acetylated dimer repeat |
| CHEBI:17309 pectin, the GEM's own annotation | none | none | No structure, no formula. An underdetermined class |
| CHEBI:68837 pectate | none | none | Names the right thing, carries nothing |
| KEGG C00714, the GEM's own annotation | `(C26H36O24)n` | n/a | REJECTED, four residues at 50 percent methyl esterification |
| MetaNetX MNXM107600, the GEM's own annotation | deprecated | n/a | A TRAP. Redirects to MNXM726528 = chebi:15446 rendered as the free monomer `C6H10O7` charge 0, which would unbalance `r_0365` by one water |

eQuilibrator cache: `CHEBI:17309`, `CHEBI:68837`, `MNXM107600` and `kegg:C00714` all
resolve to cache compound `82801`, which has no formula, no InChIKey and no energy. That
is the mechanical reason the pipeline reported `no_structure_in_any_namespace` on this
species. `CHEBI:140523` is absent from the cache entirely.

## Step 5, the convention

One alpha-1,4-linked D-galactopyranuronate residue, degree of methyl esterification 0, no
acetylation, C6 carboxyl deprotonated.

```
repeat unit SMILES  *O[C@@H]1[C@H](O)[C@@H](O)[C@@H](*)O[C@@H]1C(=O)[O-]
repeat unit formula C6H7O6      repeat unit charge  -1
```

The two attachment points are, in written order, the O4 oxygen (bonded to the preceding
residue's anomeric C1) and the anomeric C1 itself (bonded to the following residue's O4).
Each residue owns exactly one bridging oxygen, its own O4, so the bridging oxygen is
counted once per residue. That accounting is what makes the residue `C6H7O6` and not
`C6H7O7`. The SMILES was obtained by substituting attachment points into CHEBI:140523's
own SMILES without touching any stereo descriptor.

No end groups are counted. ChEBI writes a chain as `(C6H7O6)n.H2O`, n residues plus one
water; the GEM writes one residue with no terminal water, which is the per-repeat-unit
convention the surrogate route requires.

## Step 6, verification

`verification_rdkit_cobra.txt`, sha256
`d76e1ca9e3a1fce7831f83f8a094c57d7d11d7be6fb9b0997d35d95be4285525`, RDKit 2026.03.6:

- The proposed SMILES has non-dummy composition `{C: 6, H: 7, O: 6}`, formal charge -1,
  two attachment points.
- Capping O4 with H and C1 with OH reproduces CHEBI:58658 canonically, so residue plus
  one water is exactly the monomer, which is exactly what `r_0365` says.
- `r_0365` as shipped: `{'charge': -1.0}`. With `s_1309` charge set to -1: `{}`.
- Explicit sums with the proposed repeat unit: elements `{}`, charge `0.0`.

`verification_group_decomposition.txt`, sha256
`f8998147e5882fe0b117c4bac02e848a908d2165b20a71ed2b98616370cbae07`: the repeat unit does
NOT decompose (`GroupDecompositionError`), as expected for a structure carrying attachment
points, while the monomer decomposes into 8 groups. The residue's energy must come from
the surrogate route off the priced monomer, never from feeding this SMILES to the
structure route.

`verification_equilibrator_crosscheck.txt`, sha256
`b5169006229eaf86188de08e86096813f2bbbc14b9acd963a9eddfb9bc33e11c`, at the compartment e
conditions in the pipeline config (pH 5.0, I 0.25 M, pMg 3.0, T 303.15 K). Both sides are
component-contribution values, so this tests the CONVENTION and not the accuracy of any
energy:

- surrogate rule, `G'(monomer) - G'(H2O) + 6.0505` = **-635.62 kJ/mol**
- independent, from the cached `C18H23O19` trianion read as 3 residues plus one water,
  `(G'(trimer) - G'(H2O)) / 3` = **-639.95 kJ/mol**
- difference **+4.33 kJ/mol**, against the surrogate's per-residue sigma of 3.41

Reproducing `s_0560` at -820.1443 and `s_0805` at -178.4697, matching the pipeline
exactly, is what shows the conditions were set correctly.

## Step 7, energies, and the negative that matters

No measured and no tabulated formation energy exists for pectin, polygalacturonic acid or
the galacturonate residue in anything searched. Rhea curates no EC 3.2.1.15 reaction, the
NIST TECRDB search returned no apparent equilibrium constant for a
(1->4)-alpha-D-galacturonide hydrolysis, and no combustion or formation table covers the
polymer.

The two leads carried into this row are rejected, and must not be recorded as equilibrium
energies:

- **PMC5153499**: activation energy of catalysis 15.1 kJ/mol, plus thermal denaturation
  parameters (Ea(d) 125.5 kJ/mol, deltaH(d) 124.4 to 124.7 kJ/mol, deltaS(d) 203.9 to
  238.2 J/mol/K, deltaG(d) about 50 kJ/mol).
- **PubMed 20668400** (Maisuria et al., J Microbiol Biotechnol 2010;20(7):1077-85):
  "Thermodynamic parameters for polygalacturonic acid hydrolysis by purified PG were,
  deltaH* = 7.98 kJ/mol, deltaG* = 68.86 kJ/mol, deltaS*= -194.48 J/mol/K, deltaG(E-S) =
  -1.04 kJ/mol and deltaG(E-T) = -8.96 kJ/mol." Starred quantities are activation
  parameters; the last two are enzyme-substrate and enzyme-transition-state binding
  energies. The rest of the paper is irreversible thermal inactivation.

That negative is the reason the energy has to come from the surrogate route rather than
from a table.

## Step 8, what a human still needs to decide

1. Whether the `fbc:charge` fix on `s_1309` lands in a local copy of yeast-GEM or as a
   pipeline-side override, and whether to report it upstream. The shipped model has a
   charge-unbalanced `r_0365` and the fix is one attribute.
2. Whether to re-annotate `s_1309` from ChEBI:17309 to ChEBI:140523, and to drop the
   deprecated MNXM107600 rather than follow it to MNXM726528.
3. Whether degree of methyl esterification 0 is the intended reading. It is forced by the
   elemental balance of `r_0365` as written, but real pectin is partially esterified, so
   the model is describing pectate and the species name is a misnomer. That is a modeling
   judgment, not a curation one.
4. Whether the attachment-point SMILES is stored at all. It is proven here not to
   decompose and must never reach the structure route; its only use is documentation.
5. A visual check of the alpha anomeric configuration. It was inherited unchanged from
   CHEBI:140523's SMILES and the capped form matches CHEBI:58658, but no independent
   depiction was inspected.
