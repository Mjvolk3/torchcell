# nitrophenyl_phosphate search log

Curated 2026-09-11. Metabolites s_3859 4-nitrophenyl phosphate [c] (C6H4NO6P, charge -2)
and s_3860 4-nitrophenol [c] (C6H4NO3, charge -1). Blocked reaction r_4220,
4-nitrophenyl phosphate phosphohydrolase. Exclusion reason recorded by the pipeline:
`group_decomposition_failed`.

Everything fetched is under
`$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/nitrophenyl_phosphate/`
with `SHA256SUMS.txt`. The record is
`experiments/026-metabolism-flux/results/curation/nitrophenyl_phosphate.json`.

## 1. What the pipeline already knew

`experiments/026-metabolism-flux/results/unresolved_metabolites.csv` gives, for both
metabolites, `diagnosis=cache_structure_without_group_decomposition` and
`final_blocker=structure_known_but_no_group_decomposition`. The GEM's accessions are
kegg C03360 / chebi CHEBI:61146 / metanetx MNXM3867 / bigg HC01104 for s_3859 and
kegg C00870 / chebi CHEBI:57917 / metanetx MNXM526 / bigg 4nph for s_3860. The KEGG ids
guessed in the curation brief, C08297 and C06214, are not the ones the GEM cites.

## 2. Reading the group definitions before running anything

`~/.cache/equilibrator/group_definitions.csv` line 46:

```
"-NO2",0,0,0,"[C,c,N,n]N(=O)=O","1|2|3","Nitro",
```

The SMARTS is the pentavalent spelling. That is the reverse of the brief's hypothesis,
which expected the charge-separated form to be the one eQuilibrator wants.

## 3. Decomposition probe

Ran `equilibrator_assets.group_decompose.GroupDecomposer.Decompose` on 14 structures
under `~/miniconda3/envs/equilibrator-ext/bin/python`, each first checked in RDKit for
formula, charge and InChIKey. Output archived as `decomposition_probe.json`; a second
pass with `raise_exception=False` recorded the unassigned atoms in
`decomposition_unassigned_atoms.txt`.

| structure | SMILES | decomposes |
|---|---|---|
| s_3860 phenolate, pentavalent nitro | `[O-]c1ccc(cc1)N(=O)=O` | yes |
| s_3860 phenolate, charge separated | `[O-]c1ccc(cc1)[N+](=O)[O-]` | no |
| s_3860 neutral phenol, pentavalent | `Oc1ccc(cc1)N(=O)=O` | yes |
| s_3860 neutral phenol, charge separated | `Oc1ccc(cc1)[N+](=O)[O-]` | no |
| s_3859 dianion, pentavalent | `O=N(=O)c1ccc(OP(=O)([O-])[O-])cc1` | no |
| s_3859 monoanion, pentavalent | `O=N(=O)c1ccc(OP(=O)([O-])O)cc1` | no |
| s_3859 neutral, pentavalent | `O=N(=O)c1ccc(OP(=O)(O)O)cc1` | no |
| s_3859, all three charge-separated variants | | no |
| control nitrobenzene, pentavalent | `O=N(=O)c1ccccc1` | yes |
| control nitrobenzene, charge separated | `[O-][N+](=O)c1ccccc1` | no |
| control phenol and phenolate | `Oc1ccccc1`, `[O-]c1ccccc1` | yes |
| control phenyl phosphate, no nitro | `c1ccccc1OP(=O)([O-])[O-]` | no |
| control methyl phosphate | `COP(=O)([O-])[O-]` | yes |
| control sn-glycerol 3-phosphate | `OCC(O)COP(=O)([O-])[O-]` | yes |

Two independent blockers, separated by the controls. The nitrobenzene pair isolates the
nitro spelling. The phenyl phosphate control, which has no nitro group at all, isolates
the aryl phosphate ester: the unassigned atoms are the phosphorus and its three oxygens,
while the same `-OPO3 [H0 Z-2 Mg0]` group is found without complaint on the two aliphatic
phosphate esters.

## 4. Why the compound cache does not fix itself

Queried eQuilibrator's cache through every accession the GEM cites (`cc_check.py`). Both
molecules are present, s_3860 as compound 500 and s_3859 as compound 3341, both stored
with the charge-separated nitro SMILES, both with an empty group vector, both returning
`None` from `standard_dg_formation`. Handing `LocalCompoundCache.get_compounds` the
corrected SMILES returns the same stale row, because it resolves by InChIKey. Verified
separately that RDKit's `MolToSmiles` rewrites `N(=O)=O` back to `[N+](=O)[O-]`, so a
pipeline stage that canonicalizes through RDKit undoes the fix silently.

Injecting the corrected group vector in memory (`inject_gv.py`) makes component
contribution answer for s_3860: standard dgf -99.71 kJ/mol, and dgf' +109.39 kJ/mol at
pH 7.2, I = 0.25 M, T = 303.15 K, pMg 3.0. That number is a computed estimate, quoted
only to show the decomposition is productive.

Also checked: neither compound is in the 631-compound reactant-contribution training set,
so no measured reaction in eQuilibrator's training data touches either species.

## 5. An option tried and rejected

`hand_vector2.py` assembles a group vector for the aryl phosphate by hand, as nitrobenzene
with one `ring =c- [H1]` promoted to `ring =c< [H0]` plus one `-OPO3 [H0 Z-2]`. The
control in the same script, sn-glycerol 3-phosphate hydrolysis with both sides genuinely
decomposable, returns -13.0 +/- 2.1 kJ/mol, the right size for a phosphomonoester, so the
machinery works. The hand-built vector gives r_4220 a standard dg' of -62.2 +/- 2.8
kJ/mol, far more negative than a phosphomonoester hydrolysis should be, with no fitted
support for an `-OPO3` group on an aromatic carbon and nothing to corroborate it. Computed,
not admitted, recorded so it is not repeated.

## 6. Literature

Searched for Goldberg, Lang, Lo, Ross and Tewari, J. Chem. Thermodyn. 41 (2009) 654-665,
doi:10.1016/j.jct.2008.12.017.

- ScienceDirect: HTTP 403.
- Semantic Scholar graph API: `isOpenAccess false`, `openAccessPdf` status CLOSED.
- NIST publication landing page: served, archived, sha256 `5aa3291e...`. Its abstract
  says "Calorimetrically determined enthalpies of reaction ΔrH(cal) were measured for
  reactions (1) to (5) and the apparent equilibrium constant K was measured for reaction
  (2)." Reaction (3) is ours; reaction (2) is sn-glycerol 3-phosphate. So no equilibrium
  constant was measured for the 4-nitrophenyl phosphate hydrolysis.

The numbers were then recovered from a different artifact. A search for TECRDB entries
surfaced `https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=904030`, which is the
NIST-deposited author manuscript of Decker, Goldberg, Lang and Michener on the
1-naphthyl acetate, 4-nitrophenyl acetate and 4-nitrophenyl alpha-L-arabinofuranoside
hydrolyses, sha256 `92942a81...`. Its working-notes pages, left in after the References,
quote the authors' own [09GOL/LAN] verbatim:

```
For the hydrolysis of 4-nitrophenyl phosphate [09GOL/LAN],
4-nitrophenyl phosphate2-(aq) + H2O(l) = 4-nitrophenol(aq) + HPO2-4 (aq) ΔrH° = -(13.7 ± 0.5) kJ·mol-1     (A)
Ionization data:
4-nitophenyl phosphate- = H+ + 4-nitophenyl phosphate2- pK = 5.42 and ΔrH° = -9.2 kJ·mol-1                 (B)
H2PO4- = H+ + HPO2-4                                    pK = 7.198 and ΔrH° = 3.6 kJ·mol-1                 (C)
```

Plus, from the same file, the 4-nitrophenol ionization pK 7.15 with drH0 19.4 kJ/mol
(line 1095) and drfH0 -190.53 kJ/mol for 4-nitrophenol(aq) (line 803). Standard state
throughout is the hypothetical ideal solution of unit molality. The temperature is not
stated in the quoted lines.

This is a secondary transcription. It is enough to state the enthalpy with a quote and a
hash, and not enough to state the conditions, so the primary still has to be read.

Two older primaries were located and both are behind 403s: Axelrod, "The Free Energy of
Hydrolysis of p-Nitrophenyl Phosphate", Science 114 (1951) 525-526,
doi:10.1126/science.114.2968.525, the only Gibbs energy lead found for this reaction; and
Sturtevant, "The Heat of Hydrolysis of p-Nitrophenylphosphate", J. Am. Chem. Soc. 77
(1955) 255-258, an independent calorimetric enthalpy.

## 7. TECRDB

`randr.nist.gov/enzyme/Default.aspx` and `/enzyme/Search.aspx` both redirect to
`/Error.html`. The landing page `randr.nist.gov/Default.aspx` serves, and its Downloads
page offers only the BioEqCalc Mathematica package. The NIST PDR record for SRD 74,
`ECBCC1C130302ED9E04306570681B10754`, doi:10.18434/t40w22, carries no downloadable data
component. Two candidate copies of eQuilibrator's `TECRDB.csv` on GitLab and GitHub both
404. No TECRDB entry for EC 3.1.3.1 or 3.1.3.2 with 4-nitrophenyl phosphate could be
queried; the search interface is gone, not merely empty.

## 8. NIST WebBook, 4-nitrophenol

CAS 100-02-7. Solid drfH0 -212.4 +/- 1.0 kJ/mol (Finch, Gardner et al. 1983) and
-207.1 +/- 1.1 (Sabbah and Gouali 1994); gas -117.7 +/- 2.0 and -114.7 +/- 1.2;
sublimation 92.39 +/- 0.43. No Gibbs energy of formation, no aqueous phase. Archived,
sha256 `61f09c69...`. Nothing here becomes an aqueous drfG without a solvation model,
which would be computed.

## 9. Outcome

Kind `structure`, admission `tabulated`. s_3860 gets a working curated representation and
a rule that transfers to any nitro species, including the ethylnitronate row. s_3859 does
not, and r_4220 therefore stays blocked. The measured value found is a reaction ENTHALPY,
-(13.7 +/- 0.5) kJ/mol, which is admissible as an enthalpy and is not a Gibbs energy.
