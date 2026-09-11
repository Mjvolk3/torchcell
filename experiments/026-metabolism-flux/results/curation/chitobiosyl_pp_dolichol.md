# chitobiosyl_pp_dolichol search log

Curation of one thermodynamic exclusion in yeast-GEM 9.0.2: metabolite `s_3767`
N,N'-Chitobiosyldiphosphodolichol [er], formula `C36H64N2O17P2`, charge 0, excluded with
reason `no_structure_available`, blocking reaction `r_4159` (ALG13/ALG14, EC 2.4.1.141).

Date: 2026-09-11. All fetched artifacts and their sha256 sums are in
`$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/chitobiosyl_pp_dolichol/`
(`MANIFEST.sha256`). The machine record is `chitobiosyl_pp_dolichol.json` in this folder.

## Answer in one paragraph

yeast-GEM writes the ChEBI and KEGG generic formula `(C5H8)n.<core>` with **n = 0**, keeping
only the core. The core is not dolichol free: it already carries a C20 dolichyl, four
isoprene units with the alpha unit saturated, and ChEBI's `n` counts units beyond those
four. ChEBI's *depicted* structure, however, is the **n = 1** member (C25 dolichyl), and
that is what MetaNetX and the equilibrator compound cache hold, so all eleven priced
dolichol siblings were priced against a molecule with exactly five more carbons than the GEM
formula attached to them. `s_3767` was excluded only because it carries no accession at all,
not because its chemistry is hard.

## What was queried, in order

1. **cobra scan of yeast-GEM 9.0.2** for metabolite names matching
   `dolich|chitobio|glucosaminyldiphospho|oligosacchar`, then every reaction touching them.
   Result: 12 species. Eleven carry a `metanetx.chemical` accession; `s_3767` carries only
   `sbo: SBO:0000247`. `r_4159` returns an empty `check_mass_balance()`, so it balances in
   mass and charge as written. `s_3767` [er] appears in `r_4159` only, and `s_1184` [c],
   which is the same molecule at charge -2, appears in `r_0228` only.
2. **`thermo_structure_attempt.json`**, record for `s_3767`. Verbatim:
   `"accession_used": null, ... "outcome": "no_structure_in_any_namespace"`. That is the
   whole exclusion.
3. **`metabolite_dgf_prime_with_external.parquet`** for the family. Eleven siblings are
   priced, all by route `cache_accession`, detail `metanetx.chemical`. `s_3767` has no row.
4. **`~/.cache/equilibrator/compounds.sqlite`**, joined on the nine MetaNetX accessions the
   GEM cites. All nine resolve, all carry a group vector, and every resolved structure has
   exactly **five more carbons** than the GEM formula. Compound 49402
   (`QBRZRKKUHJLMCY-HRUJLXCPSA-N`, atom bag C41 O17 P2 H70 N2) is reached from
   `MNXM63078`, `CHEBI:12427`, `C04537` and `chito2pdol` alike.
5. **`https://www.rhea-db.org/rhea/23380.rdf`** with a default curl user agent. HTTP 403.
   The plain `.rdf` path is refused.
6. **`https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity?chebiId=CHEBI:12427`**.
   HTTP 500. The legacy ChEBI SOAP test endpoint is dead and returns the same 10135 byte
   error page for every identifier, so a naive fetch looks like a hit. Worth knowing before
   another agent records that page as an artifact.
7. **Rhea REST with a browser user agent**,
   `?query=RHEA:23380&columns=rhea-id,equation,chebi-id,ec&format=tsv`. HTTP 200. Verbatim:
   `an N-acetyl-alpha-D-glucosaminyl-diphospho-di-trans,poly-cis-dolichol + UDP-N-acetyl-alpha-D-glucosamine = an N,N'-diacetylchitobiosyl-diphospho-di-trans,poly-cis-dolichol + UDP + H(+)`,
   participants `CHEBI:58427;CHEBI:57705;CHEBI:57269;CHEBI:58223;CHEBI:15378`. Rhea's
   participants are the (2-) microspecies, and the lipid is named as a family.
8. **ChEBI backend API**, `https://www.ebi.ac.uk/chebi/backend/api/public/compound/12427/`.
   HTTP 200 and the decisive hit. Verbatim:
   `"chemical_data":{"formula":"(C5H8)n.C36H64N2O17P2","charge":0,"mass":"926.972"}`.
   That core is character for character the yeast-GEM `s_3767` formula and charge. The same
   record's `default_structure` is `C41H72N2O17P2`, that is the n = 1 member, with InChIKey
   `QBRZRKKUHJLMCY-HRUJLXCPSA-N`, matching the cache compound in step 4.
9. **Same API for 57269, 16091, 16214, 15809, 57525, 18278, 58427, 18396, 28067.** HTTP 200
   for all nine. Every one is written `(C5H8)n.<core>` and every core matches the yeast-GEM
   formula of the corresponding species. This is what turns a coincidence into a convention:

   | GEM species | GEM formula, charge | ChEBI | ChEBI formula |
   |---|---|---|---|
   | s_0642 dolichol | C20H36O, 0 | 16091 | (C5H8)n.C20H36O, 0 |
   | s_0645, s_0646 dolichyl phosphate | C20H37O4P, 0 | 16214 | (C5H8)n.C20H37O4P, 0 |
   | s_3888 dolichyl diphosphate | C20H35O7P2, -3 | 57497 | (C5H8)n.C20H35O7P2, -3 |
   | s_0644 dolichyl D-mannosyl phosphate | C26H47O9P, 0 | 15809 | (C5H8)n.C26H47O9P, 0 |
   | s_3911 dolichyl beta-D-glucosyl phosphate | C26H46O9P, -1 | 57525 | (C5H8)n.C26H46O9P, -1 |
   | s_3765 GlcNAc-PP-Dol | C28H51NO12P2, 0 | 18278 | (C5H8)n.C28H51NO12P2, 0 |
   | **s_3767 chitobiosyl-PP-Dol** | **C36H64N2O17P2, 0** | **12427** | **(C5H8)n.C36H64N2O17P2, 0** |
   | s_1184 same, cytosolic | C36H62N2O17P2, -2 | 57269 | (C5H8)n.C36H62N2O17P2, -2 |
   | s_0414 Man2-GlcNAc2-PP-Dol | C48H84N2O27P2, 0 | 28067 | (C5H8)n.C48H84N2O27P2, 0 |

10. **KEGG REST**, `get/C04537` and `get/R05970`. HTTP 200. Verbatim from C04537:
    `NAME  N,N'-Diacetylchitobiosyldiphosphodolichol; Chitobiosyldiphosphodolichol` and
    `FORMULA  C36H64N2O17P2(C5H8)n`. The GEM's species *name* for `s_3767` is KEGG's second
    synonym and its formula is KEGG's core, so the identity was known at model build time
    and only the accession was dropped. R05970 is the `kegg.reaction` annotation on `r_4159`
    and its product G00002 is C04537, which is CHEBI:12427, which is `s_1184`.
11. **WebSearch**, `ChEBI 12427 N,N'-diacetylchitobiosyldiphosphodolichol (C5H8)n formula`.
    Confirmed the CHEBI:12427 and CHEBI:57269 conjugate acid and base pair. No new primary
    fact beyond the API records.
12. **WebSearch**, `yeast Saccharomyces cerevisiae dolichol chain length isoprene units`.
    Yeast dolichols run about 14 to 17 isoprene units in vegetative cells. Context only, not
    a sourced value here. It does show that both the GEM's n = 0 (four units) and ChEBI's
    depicted n = 1 (five units) are conventions rather than the yeast molecule.
13. **FVA on `r_4159`**, `fraction_of_optimum=0.0`. Minimum 0.0, maximum 0.0.

## Structure produced

The n = 0 member, derived from ChEBI:12427's own depicted structure by deleting its single
cis isoprene unit (the one adjacent to the alpha carbon, InChI bond `28-19-`), leaving the
alpha saturated unit, two trans units and the terminal isopropylidene. Nothing was invented:
the sugar stereocenters and the two trans bonds are ChEBI's, and the alpha carbon is left
undefined exactly as ChEBI leaves it (`/t24?` here, `/t29?` there).

```
CC(=O)N[C@H]1[C@@H](OP(=O)(O)OP(=O)(O)OCCC(C)CC/C=C(\C)CC/C=C(\C)CCC=C(C)C)O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2NC(C)=O)[C@@H]1O
```

RDKit gives `C36H64N2O17P2`, charge 0, InChIKey `WPXHWCJFZGZSHJ-HUKUIXITSA-N`. Both match
the GEM exactly. The microspecies is the neutral diphosphate diacid, which is what charge 0
demands and not what dominates at pH 7.1. Writing both P-OH as `[O-]` gives
`C36H62N2O17P2` at charge -2, InChIKey `WPXHWCJFZGZSHJ-HUKUIXITSA-L`, which is character
for character what the GEM declares for `s_1184`. The two GEM species are one molecule.

## Decomposition check

`~/miniconda3/envs/equilibrator-ext/bin/python decomposition_check_final.py`. **It
decomposes**: compound 694327, formula `C36H64N2O17P2`, group vector present, sigma 11.59
kJ/mol, `dgf'0` 1700.17 kJ/mol at pH 7.1.

Two things had to be worked around and both are recorded. ChemAxon cxcalc is not installed
on this host, so the plain `LocalCompoundCache` is read only and creates no compound at all
(`status: failed, method: read-only`); the check ran through
`LocalCompoundCache(pkas_provided_externally=True)` with the pKa ladder
`[13.43, 12.78, 5.85, 5.25, 3.19, 1.74]` copied from cache compound 49402, which carries the
identical ionizable set. This is the same route `external_pka_apply_selection.py` uses.

## Why the structure is not the recommended default route

Measured, both created fresh by the same route with the same pKas: the n = 0 core prices at
1737.31 kJ/mol at pH 7.2 and the n = 1 member at 2190.14, so one isoprene unit is worth
**452.83 kJ/mol** at pH 7.2 and 448.19 at pH 7.1. The other dolichol-linked participant of
`r_4159`, `s_3765`, is priced at n = 1. Mixing conventions inside the reaction gives

| `s_3767` priced as | `dgf'0` at pH 7.1 | resulting `dGr'0` for r_4159 |
|---|---|---|
| fresh n = 0 curated structure | 1700.17 | **+1475.83 kJ/mol** |
| n = 1 accession, as its twin s_1184 was | 210.63 | **-13.72 kJ/mol** |

using the shipped values for UDP-GlcNAc (-2076.847), s_3765 (303.895), UDP (-1997.294) and
H+ (0). The second is the expected magnitude for a UDP-sugar glycosyltransferase. The
1489 kJ/mol between them is convention artifact.

A further caution, observed and not explained: a freshly created compound and a shipped
cache compound for the *same* molecule are not interchangeable either. Created fresh from
the neutral parent, the n = 1 member prices at 2190.14 kJ/mol against the shipped compound's
251.24 at pH 7.2, and their group vectors differ at exactly two indices, 44 and 46, which
tracks the reference microspecies (shipped anchors the (2-) state at 70 protons, fresh
anchors the neutral at 72). I have not traced that 1938.9 kJ/mol offset to a specific group
energy and claim no mechanism for it.

## What a human still needs to decide or do

1. **Pick the remedy.** Preferred: attach `metanetx.chemical:MNXM63078`, `CHEBI:12427` and
   `kegg.compound:C04537` to `s_3767` and let the accession route price it as it priced
   `s_1184` (210.627 kJ/mol at the ER pH 7.1, sigma 14.055). That is the only option keeping
   `s_3767` on the same footing as `s_3765` inside `r_4159`. Alternative: enforce the
   composition check and move the whole family to n = 0 in one pass, using the SMILES above
   for `s_3767` and its (2-) form for `s_1184`. Do not mix the two.
2. **Decide what to do about the family-wide +5 carbon offset.** It is not specific to this
   species. Eleven siblings are already priced against structures one isoprene unit larger
   than the formula they carry, because the accession sweep applies no composition check
   while the later `thermo_structure_attempt.py` does. If `s_3767` were simply given the
   accession, the newer stage would reject it as `composition_mismatch` even though eleven
   siblings passed the older one.
3. **Fix the model defect, separately.** `s_3767` [er] and `s_1184` [c] are the same
   molecule with no transport or equivalence between them, so the ER branch of N-glycan
   precursor assembly (`r_4170` then `r_4159`) dead-ends and FVA puts `r_4159` at exactly
   zero. Pricing `s_3767` adds one reaction to the thermodynamic coverage count and zero
   flux-carrying reactions. Merging the two species is a yeast-GEM curation change and was
   out of scope here.
4. **Label any energy that follows as computed.** `admission` is `tabulated` for the
   structure and the convention, which come verbatim from ChEBI and KEGG. Every formation
   energy in play is component contribution, not measured.

## Files

Artifacts, one folder, sha256 in `MANIFEST.sha256`:

```
/scratch/projects/torchcell-scratch/data/torchcell/thermo_equilibrator/curated/chitobiosyl_pp_dolichol/
```
