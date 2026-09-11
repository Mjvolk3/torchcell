# cytochrome_c curation log

Date: 2026-09-11. Curator: coding agent, 026-metabolism-flux thermodynamic exclusions.
Record: `experiments/026-metabolism-flux/results/curation/cytochrome_c.json`.
Artifacts: `$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/cytochrome_c/`.

## What was being looked for

A measured midpoint potential for Saccharomyces cerevisiae iso-1-cytochrome c (CYC1,
YJR048W), so the GEM's ferri/ferro cytochrome c pair can be priced through a carrier
potential. In yeast-GEM 9.0.2 the pair is s_0709 ferricytochrome c [m]
(C42H44FeN8O8S2R4, charge 0) and s_0710 ferrocytochrome c [m] (C42H45FeN8O8S2R4,
charge 0), both excluded with reason `placeholder_formula` because of the R group.
Together they block seven reactions: r_0001, r_0002, r_0004 (lactate:ferricytochrome c
oxidoreductases), r_0437 (cytochrome c peroxidase), r_0438 (cytochrome c oxidase),
r_0439 (ubiquinol:ferricytochrome c reductase) and r_4382
(thiosulfate:ferricytochrome c oxidoreductase).

## Result in one line

Admitted as **measured**: E prime zero = 272 mV vs SHE at pH 7.4, 0.1 M sodium
phosphate, about 22 degrees C, by cyclic voltammetry, reported to +/- 2 mV, on the
standard WT* (K72A/C102S) mimic of yeast iso-1-cytochrome c (Deng 2018, Table 2).

## Searches run

Web searches (WebSearch, 2026-09-11):

1. `yeast iso-1-cytochrome c reduction potential 290 mV Rafferty Mauk 1990 Biochemistry
   position 82`. Confirmed Rafferty 1990 reports shifts (10 to 43 mV) and not an
   absolute wild-type value.
2. `Battistuzzi Borsari Sola redox thermodynamics yeast iso-1-cytochrome c 0.290 V 25 C
   pH 7 reduction potential`. Surfaced Bortolotti 2006 (immobilized protein, +370 mV at
   pH 7). No free solution value at pH 7 in the Modena series.

WebFetch:

3. PMC1222387, Lett and Guillemette (2002) Biochem J 362:281-287. Reports only the
   +117 mV shift of the Y48K variant relative to wild type. No absolute value.

PMC full text searches (NCBI eutils esearch, db=pmc):

4. `"iso-1-cytochrome c" AND "290 mV"`: 7 hits. PMC6177310 (Deng 2018) carries Table 2
   with wild type and WT* values plus full conditions, and became the admitted source.
5. `"iso-1-cytochrome" AND "reduction potential of 290"`: 0 hits.

PubMed searches (db=pubmed):

6. `Cutler RL[Author] AND cytochrome c`: 3 hits. Cutler 1987 (Protein Eng 1:95-99) gives
   272 mV vs NHE for the C107T background; Cutler 1989 (Biochemistry 28:3188-97) gives
   Arg-38 shifts.
7. `Battistuzzi G[Author] AND iso-1-cytochrome c`: 12 hits, all alkaline transition
   thermodynamics or immobilized protein.
8. `Battistuzzi G[Author] AND cytochrome c AND redox thermodynamics`: 23 hits. The 1997
   Biochemistry survey covers beef heart and bacterial c2, not yeast.
9. Title search for Rafferty 1990: PMID 2174257.

Other:

10. cobra read of `$DATA_ROOT/data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.xml`
    listing every metabolite and reaction whose name contains "cytochrome". This is how
    the met ids, the seven blocked reactions and the r_4152 finding below were fixed.
11. Rhea TSV export for RHEA:11436.

## Evidence admitted

From `pmc6177310_deng2018_biochemistry.xml`
(sha256 `9edfa93cacc1be188ef396046da1b88b9098c31cf4770d99e3bfd063987d6d68`),
Deng Y, Zhong F, Alden SL, Hoke KR, Pletneva EV (2018) Biochemistry 57(40),
"The K79G Mutation Reshapes the Heme Crevice and Alters Redox Properties of
Cytochrome c":

- Table 2, "Reduction Potentials of Yeast Iso-1 Ferric Cyt c Variants", E m in mV vs
  SHE. WT 290 at pH 6.0 in 0.1 M sodium phosphate (footnoted as taken from their
  reference 15). WT* 286 at pH 6.0 in phosphate, 287 in chloride, 288 in acetate, and
  272 at pH 7.4 in phosphate, with 274 +/- 4 by spectroelectrochemistry at pH 7.4.
- Table 2 footnote a: "All the reduction potentials (+/- 2 mV vs. SHE) were measured
  using cyclic voltammetry unless otherwise indicated."
- Methods: saturated calomel reference, scan rates 0.02 and 0.1 V/s, "measured
  electrode potentials E SCE were converted to the Standard Hydrogen Electrode E SHE,
  based on E SHE = E SCE + 243 mV at 22 degrees C". This fixes T at about 295 K.
- Results: "The reduction potential of WT* (K72A/C102S) measured at pH 6 (286 mV) is not
  significantly different from that (290 mV) of WT under the same conditions."
- Results: the measured pH dependence, 14 mV down from pH 6.0 to pH 7.4 for WT*.

## Why pH 7.4 and not pH 6.0

The authentic wild-type entry in the same table is 290 mV, but at pH 6.0 and taken from
Rosell, Ferrer and Mauk (1998) J Am Chem Soc 120:11234-11245 rather than measured in
Deng 2018. The thermodynamics pipeline works near pH 7, and the pH dependence is
measured in the same paper. Choosing the pH 7.4 entry keeps every condition traceable
to a measurement inside the archived artifact. The mimic penalty (4 mV at pH 6.0) is
smaller than the pH shift (14 mV), so WT* is the cheaper approximation of the two.
A code agent that prefers the authentic protein should use 290 mV and carry pH 6.0.

## Corroboration

- Spectroelectrochemistry on WT* at pH 7.4 in 0.1 M sodium phosphate: 274 +/- 4 mV
  (Deng 2018 Table 2, footnote d).
- Cutler RL, Pielak GJ, Mauk AG, Smith M (1987) Protein Eng 1(2):95-99, independent
  laboratory, C107T background: "a mid-point reduction potential of 272 mV versus NHE".
  pH, temperature and ionic strength are not in the abstract and the full text is
  paywalled, so this is recorded at abstract level.
- Bortolotti 2006 (J Am Chem Soc 128:5444-51): +370 mV at pH 7 for the protein
  immobilized on gold, stated to be about 100 mV above the solution value, implying
  about +270 mV in solution. This is also the warning that immobilized measurements are
  not on the same scale as solution ones.

Four independent handles land between 270 and 276 mV, which is why this record is
admitted rather than provisional.

## Uncertainty

sigma_kj_per_mol = 0.19 kJ/mol = n F dE with n = 1 and dE = 2 mV, the reported
measurement uncertainty. That does not cover the pH transfer. Moving from pH 7.4 to
pH 7.0 is worth roughly 4 mV (0.4 kJ/mol) on the measured slope, and the WT versus WT*
difference is worth about 4 mV more, so a defensible total for a pH 7.0 application is
about 1 kJ/mol per electron.

## No single partner couple

Unlike cytochrome b5, cytochrome c has no NADH partner anywhere in yeast-GEM, so
`partner_couple_e_prime_mv` is null. Each blocked reaction needs its own partner:
r_0001 and r_0002 against D-lactate/pyruvate, r_0004 against L-lactate/pyruvate, r_0437
against H2O2/H2O, r_0438 against O2/H2O, r_0439 against ubiquinol/ubiquinone, r_4382
against thiosulfate/tetrathionate. Only the cytochrome c half is supplied here.

## r_4152 is not covered

The task brief named r_4152 as a cytochrome c reaction to fix. It is not fixable by a
midpoint potential. r_4152 "Cytochrome c apocytochrome-c-lyase" is
`s_3748 <=> s_0712 + s_3749`, that is holocytochrome c synthase forming thioether bonds
between heme vinyls and apoprotein cysteines. s_3748 Cytochrome c [m] (C40H40FeN6O6S2,
charge -2) and s_3749 Apocytochrome c [m] (C6H10N2O2S2, charge 0) are excluded for
`placeholder_structure`, and the chemistry is bond formation, not electron transfer. The
same holds for r_4192 (cytochrome c lysine N-methyltransferase, s_3816 and s_3817),
which is a methyl transfer. Both need a different remedy.

## Rhea representation

Rhea models cytochrome c exactly as it models cytochrome b5. RHEA:11436 (cytochrome c
oxidase, EC 7.1.1.9) is
`4 Fe(II)-[cytochrome c] + O2 + 8 H(+)(in) = 4 Fe(III)-[cytochrome c] + 2 H2O + 4 H(+)(out)`,
with the two carrier states resolving to CHEBI:29033 iron(2+) and CHEBI:29034 iron(3+)
attached to the named polypeptide. One carrier class, two oxidation states, a midpoint
potential, no formation energy.

## What a human still needs to do

1. Decide pH 7.4 versus pH 6.0, and whether to interpolate to the compartment pH used by
   the flux model. Record the choice.
2. Decide whether the WT* mimic is acceptable or whether the authentic trimethylated
   yeast protein is required.
3. If the pH profile matters, obtain Rosell, Ferrer and Mauk (1998) J Am Chem Soc
   120:11234-11245, which measured the wild-type potential across pH, and deposit it in
   the mirror. It is paywalled and is quoted here only secondhand through Deng 2018.
4. Supply partner potentials for the six distinct partner couples listed above, or price
   those halves from eQuilibrator where the small molecules decompose.
