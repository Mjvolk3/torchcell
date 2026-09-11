# gpi_intermediate (s_0332) search log

Curated 2026-09-11. Machine record: `gpi_intermediate.json` in this directory.
Fetched artifacts, with sha256 recorded in the JSON:
`/scratch/projects/torchcell-scratch/data/torchcell/thermo_equilibrator/curated/gpi_intermediate/`.

Python used: `~/miniconda3/envs/torchcell/bin/python`. Model read:
`/scratch/projects/torchcell-scratch/data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.xml`,
sha256 `9fd2c572cace73c2ea835205617313554d1bf89f4ef077f49defbdfa219a4ad7`.

## Result in one line

s_0332's shipped formula C43H44N9O is a solver artifact. The formula the reaction requires is
**C55H100N3O39P3R2, charge 0**. Admission is **provisional**, because the other participants of
r_0523 are not definite species. No structure exists for this metabolite in any namespace.

## 1. What the model says

`r_0523` "GPI-anchor assembly, step 9", GPR `YDR302W or YLL031C`, EC 2.-.-.-, KEGG pathway sce00563:

```
s_0338 + s_1352 <=> s_0332 + s_0620

-1  s_0338  [er]  formula=None        charge=0   chebi:53059  mnx:MNXM1179
-1  s_1352  [er]  formula=C5H10NO4P   charge=0   chebi:16038  phosphatidylethanolamine backbone
+1  s_0332  [er]  formula=C43H44N9O   charge=0   chebi:53060  mnx:MNXM38542
+1  s_0620  [er]  formula=C3H4O       charge=0   chebi:18035  diglyceride backbone

cobra check_mass_balance: {'C': 41.0, 'H': 38.0, 'N': 8.0, 'O': -2.0, 'P': -1.0}
```

Dumping the whole `Glycosylphosphatidylinositol (gpi)-anchor biosynthesis` subsystem showed the
decisive fact: **every intermediate in the s_033x series has no formula at all except s_0332**, which
carries the impossible one. s_0329 (`C17H28NO17PR2`) is the last species in the branch with a usable
formula, and s_0332 is a dead end, r_0523 being its only reaction.

Probe scripts and their combined output are stored as artifacts: `gem_probe.py`, `gpi_pathway.py`,
`derive.py`, `build_smiles.py`, `chebi52597.py`, `gem_r0523_probe.txt`.

## 2. Where C43H44N9O came from

Tag bisection over the yeast-GEM releases (`v8.0.0`, `v8.0.1`, `v8.0.2`, `v8.1.0`, `v8.1.3`, `v8.2.0`,
`v8.3.0`, `v8.6.0`, `v9.0.0`, `main`) put the change between v8.0.2 and v8.1.0.

In v8.0.0's SBML the field reads `fbc:chemicalFormula="(GlcN)1 (Ino(acyl)-P)1 (Man)3 (EtN)2 (P)2"`,
which is the KEGG G13044 composition string, not a chemical formula. Comparing v8.0.2 to v8.1.0 gives
17 commits; the one that touches the field is `2eb8e907`, "fix-met.prop: metFormulas with (", in
PR #111 "Fix/met formulas". Its message:

> fixed all remaining formulas that had "(" and ")", either with COBRA's "computeMetFormulae" or by
> manual inspection. Some rxns remain mass-unbalanced after this.

The script it added, `ComplementaryScripts/modelCuration/makeFormulasCompliant.m`, does this:

```matlab
%Try to use COBRA's "computeMetFormulae":
problems = contains(model.metFormulas,'(');
model_test = model;
model_test.metFormulas(problems) = '';
[~,metFormulae,~,~,~,~,~,~] = computeMetFormulae(model_test);
```

`computeMetFormulae` infers a formula from mass balance. For s_0332 that system is underdetermined,
since its only reaction has an unformulated substrate and two backbone pseudo-species, and the solver
returned C43H44N9O. Reading the commit's yml patch, all 17 other hunks are hand truncations of
polymer notation such as `C26H47O9P(C5H8)n -> C26H47O9P`; s_0332 is the only entry whose formula was
invented. The value is unchanged in v9.1.1 dated 2026-08-28, so the defect is live upstream.

GitHub code search for `C43H44N9O` returns exactly one hit outside yeast-GEM, in EPFL-LCSB/SubNetX,
a downstream copy. GitHub issue searches for `s_0332`, `GPI`, and `computeMetFormulae` in the
yeast-GEM tracker all return zero. The only relevant open issue is #194, an umbrella report on mass
and charge balance from 2019 that names none of this.

A blame of `model/yeast-GEM.yml` was attempted and is useless: the release bot rewrites lines 16 to
36269 in one commit.

## 3. What the compound actually is

KEGG REST resolved the identity cleanly.

- `gl:G13044` (the ChEBI:53060 cross-reference): `COMPOSITION (GlcN)1 (Ino(acyl)-P)1 (Man)3 (EtN)2 (P)2`.
  Its NODE and EDGE block puts one ethanolamine phosphate on Man-I position 2 and one on Man-III
  position 6, with Man-III alpha1->2 to Man-II. **Two** EtN-P, not three.
- `gl:G00149` (the ChEBI:53059 cross-reference, the substrate s_0338): the same with **one** EtN-P.
- `rn:R08107`: `Phosphatidylethanolamine + G00149 <=> 1,2-Diacyl-sn-glycerol + G13044`. All four
  participants map one to one onto r_0523, so r_0523 is R08107 and its product is G13044.
- `rn:R12702`: the next step, `PE + G13044 <=> DAG + G13152`, makes the three-EtN-P complete
  precursor. yeast-GEM has no reaction for it.

The ChEBI record is internally inconsistent and yeast-GEM inherited the wrong half. CHEBI:53060's
definition describes two EtN-P and agrees with G13044; CHEBI:53060's **name**, which yeast-GEM copied
verbatim into s_0332's metName, lists three aminoethylphosphoryl groups and writes the Man-III
linkage as (1->6). That name describes G13152. The reaction stoichiometry, the definition and the
KEGG reaction all agree on two, so the derivation follows the stoichiometry.

## 4. Structure search, negative

- CHEBI:53055, 53056, 53057, 53058, 53059 and 53060 all return `"default_structure": null` and
  `"chemical_data": null` from the ChEBI backend API. No structure, formula, charge or mass.
- MetaNetX MNXM38542 (s_0332), MNXM1179 (s_0338) and MNXM7903 (s_4002) all have empty formula,
  charge, mass, InChI and SMILES. C43H44N9O did not come from MetaNetX.
- A ChEBI free-text search for the family returned HTTP 404, no compound matches.
- WebSearch for a LIPID MAPS or MetaCyc entry with defined acyl chains surfaced only review
  literature, which confirms the biology (Gpi13p transfers phosphoethanolamine to the third mannose;
  the inositol acyl in yeast is usually palmitate) but no compound record.
- The legacy ChEBI SOAP service `getCompleteEntity` now returns HTTP 500; the working route is
  `https://www.ebi.ac.uk/chebi/backend/api/public/compound/<accession>/`.

The one structure in the family is **CHEBI:52597**, the 2-acyl child of CHEBI:53060, and it is an
R-group structure for the three-EtN-P species, so it is not a defined-acyl representative either.
It is also drawn wrong, which turned out to be useful (section 6).

**Conclusion: no defined-acyl-chain representative exists.** `smiles` and `inchikey` are null in the
record.

## 5. The derivation

R means an undefined fatty acyl chain in the ChEBI sense, with the ester carbonyl carbon written
explicitly and the chain radical left as `[1*]` or `[2*]`. RDKit parsing the CHEBI:17049 SMILES gives
C17H28NO17P plus two dummy atoms at charge 0, exactly yeast-GEM's `s_0329 = C17H28NO17PR2`, which
fixes both the anchor and the convention.

Walking forward, taking each transferred group from the donor and acceptor pair that is definite:

| step | reaction | transferred group | species | formula |
|---|---|---|---|---|
| anchor | | | s_0329 | C17H28NO17PR2 |
| 3 | r_0137 | palmitoyl-CoA minus CoA = C16H30O | s_0330 | C33H58NO18PR2 |
| 5 | r_0519 | Dol-P-Man minus Dol-P = C6H10O5 | s_0334 | C39H68NO23PR2 |
| 6 | r_0520 | s_1352 minus s_0620 = C2H6NO3P | s_0337 | C41H74N2O26P2R2 |
| 7 | r_0521 | C6H10O5 | s_0339 | C47H84N2O31P2R2 |
| 8 | r_0522 | C6H10O5 | s_0338 | C53H94N2O36P2R2 |
| 9 | r_0523 | C2H6NO3P | **s_0332** | **C55H100N3O39P3R2** |

The SLIME pair is the load-bearing trick: `s_1352 - s_0620 = C5H10NO4P - C3H4O = C2H6NO3P`, which is
exactly the phosphoethanolamine group a transferase moves onto a hydroxyl. Neither backbone formula
is a real molecule, but their difference is right, and the same difference held before the SLIME
refactor, when v8.0.2 wrote `s_1352 = C7H12NO8PR2` and `s_0620 = C5H6O5R2`.

Closing r_0523:

```
LHS  C53H94N2O36P2R2 + C5H10NO4P = C58 H104 N3 O40 P3 R2   charge 0
RHS  C55H100N3O39P3R2 + C3H4O    = C58 H104 N3 O40 P3 R2   charge 0
```

## 6. Two independent checks

**RDKit on a constructed scaffold.** A SMILES for the G13044 topology was built on the CHEBI:17049
scaffold, with palmitoyl on inositol C2, GlcN alpha1->6 inositol, Man-I alpha1->4 GlcN bearing EtN-P
on C2, Man-II alpha1->6 Man-I, and Man-III alpha1->2 Man-II bearing EtN-P on C6. RDKit returns
**C55H100N3O39P3R2, charge 0**. The same scaffold one EtN-P short returns **C53H94N2O36P2R2**, the
derived s_0338. The constructed SMILES is a verification device and is not a curated structure; it
lives in `value.derivation` in the JSON, not in `value.smiles`.

**Agreement with a tabulated ChEBI formula.** CHEBI:52597 tabulates `C42H75N4O39P4R3, charge 0`.
Redoing the walk in an all-R convention gives C40H69N3O39P3R3 for the two-EtN-P species and
C42H75N4O42P4R3 for the three-EtN-P species. C, H, N, P and R match exactly; oxygen differs by
exactly 3. The reason is visible in the deposited SMILES: its three ethanolamine phosphates are
written `P(=O)(O)CCN`, a P-C phosphonate, instead of `P(=O)(O)OCCN`. RDKit confirms both readings,
returning C42H75N4O39P4R3 as deposited and C42H75N4O42P4R3 with the three bridging oxygens restored.
Swapping the R-acyl on inositol for the explicit palmitoyl that r_0137 forces turns C40H69N3O39P3R3
into C55H100N3O39P3R2.

## 7. Acyl convention in yeast-GEM

Mixed, and not a chosen chain length. The two phosphatidyl chains stay generic R groups: s_3878
C11H16O13PR2, s_3879 C19H29NO18PR2, s_0329 C17H28NO17PR2, s_4002 C32H60N2O31P2R2. The inositol
2-O-acyl is the exception, because r_0137 acylates with palmitoyl-CoA explicitly, so from s_0330
onward a mass-balanced species carries an explicit C16 palmitoyl. In an all-R convention the same
species is C40H69N3O39P3R3.

## 8. Why provisional

The admission rule allows "tabulated" for a mass-balance formula correction only when the other
participants are definite species. They are not: s_0338 has no formula anywhere, and s_1352 and
s_0620 are SLIME pseudo-species. The derivation is anchored five steps upstream and closed three
independent ways, but no tabulated source states C55H100N3O39P3R2. Admission is **provisional**.

Correcting the formula also does not unblock r_0523 for FBA. s_0332 is a dead-end metabolite, listed
as one in yeast-GEM's own `qc_findings.md`. The correction removes r_0523 from the thermodynamic
exclusion list and nothing more, and group-contribution decomposition stays impossible because the
species has two undefined acyl chains.

## 9. Other upstream defects found, none filed

1. Five GPI intermediates with no formula: s_0330, s_0334, s_0337, s_0338, s_0339. Fixing s_0332
   alone will not balance the branch. The derived values are in the table in section 5.
2. `r_0520` consumes the whole phosphatidylethanolamine backbone and releases no diglyceride
   backbone, so it transfers C5H10NO4P where the chemistry transfers C2H6NO3P.
3. `r_0522` releases no dolichyl phosphate.
4. `s_4002`, named "G00149", duplicates s_0338. Both are KEGG G00149 and CHEBI:53059, confirmed via
   MNXM7903 and MNXM1179. s_4002 carries a third formula, C32H60N2O31P2R2, matching neither, and
   sits with s_4003 on an orphan reaction r_4322 disconnected from the r_0518 to r_0523 chain.
5. s_0332's metName is the CHEBI:53060 name and describes the three-EtN-P species G13152, not the
   G13044 the reaction produces.
6. ChEBI side: CHEBI:52597's three P-C phosphonate bonds, and CHEBI:53060's name conflicting with
   its own definition.

An upstream yeast-GEM issue covering 1 to 5 together is warranted, and a ChEBI report for 6.
Per instruction, nothing was filed.
