# cytochrome_b5 curation log

Date: 2026-09-11. Curator: coding agent, 026-metabolism-flux thermodynamic exclusions.
Record: `experiments/026-metabolism-flux/results/curation/cytochrome_b5.json`.
Artifacts: `$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/cytochrome_b5/`.

## What was being looked for

A measured midpoint potential for Saccharomyces cerevisiae cytochrome b5 (Cyb5p, gene
CYB5, YNL111C, UniProt P40312), so that r_4197 (NADH-cytochrome b5 reductase,
`s_1205 + 2 s_3826 <=> s_0799 + s_1200 + 2 s_3827`) can be priced through a carrier
potential instead of through formation energies. The two GEM species s_3826
Ferricytochrome b5 [m] (formula Fe, charge 3) and s_3827 Ferrocytochrome b5 [m]
(formula Fe, charge 2) are excluded from the eQuilibrator route with reason
`no_structure_available`, and no structure exists because the species is a heme iron
inside a polypeptide.

## Result in one line

Nothing measured for yeast. Admitted as **provisional** on the bovine microsomal
(type A) protein: E prime zero = +5.5 +/- 1.8 mV vs SHE, pH 7.0, 50 mM sodium
phosphate (ionic strength 0.11 M), mediated redox potentiometry, temperature not
stated (Parthasarathy 2011, Table 2).

## Searches run

Web searches (WebSearch, 2026-09-11):

1. `Saccharomyces cerevisiae cytochrome b5 CYB5 midpoint redox potential measured`.
   No yeast value. Surfaced the human CYB5A electrochemistry paper.
2. `Reid Taniguchi Gray Mauk oxidation-reduction equilibrium cytochrome b5 JACS 1982
   reduction potential 5.1 mV`. Citation confirmed (J Am Chem Soc 104:7516-7519), full
   text paywalled.
3. `"yeast cytochrome b5" reduction potential mV spectroelectrochemistry purified
   recombinant Saccharomyces`. Hits are rat b5 expressed in yeast, and a yeast
   b5-like protein assayed spectroscopically. No potential.
4. `Cyb5p Saccharomyces cerevisiae "cytochrome b5" purification heme redox potential
   midpoint YNL111C characterization`. Genetic and functional papers only.
5. `"cytochrome b5" midpoint potential comparison table species "Saccharomyces
   cerevisiae" mV heme edge Rivera`. No species table containing yeast.
6. `"Saccharomyces cerevisiae" cytochrome b5 "redox potential" OR "midpoint potential"
   measured mV recombinant heme domain 2020`. Negative.
7. `Iyanagi 1977 Biochemistry redox properties microsomal NADH-cytochrome b5 reductase
   and cytochrome b5 potential +13 mV pH 7`. Citation confirmed, number not visible.

PMC full text searches (NCBI eutils esearch, db=pmc):

8. `"cytochrome b5" AND "Saccharomyces cerevisiae" AND "midpoint potential"`: 0 hits.
9. `"yeast cytochrome b5" AND "reduction potential"`: 0 hits.
10. `"Cyb5" AND "redox potential" AND "Saccharomyces"`: 3 hits, none relevant.
11. `"cytochrome b5" AND "trypsin-solubilized" AND "reduction potential"`: 6 hits. This
    is the search that found PMC4484759, the admitted source.
12. `"cytochrome b5" AND "5.1 mV"`: 1 hit, an orthopedics paper. The Reid 1982 number is
    not quoted verbatim anywhere in the PMC open access subset.
13. `"NAD+/NADH" AND "-320 mV"`: 139 hits, used to source the partner couple.

PubMed searches (db=pubmed):

14. `Reid LS[Author] AND cytochrome b5`: 3 hits, none the 1982 JACS paper. JACS 1982 is
    not indexed in PubMed, which is why that abstract could not be archived.
15. Title search for the Aono 2010 paper: PMID 21129218.
16. Title search for the Funk 1990 paper: PMID 2117468.

Other:

17. UniProt REST, `gene:CYB5 AND organism_id:559292`, fields `cc_cofactor` and
    `cc_function`. P40312 CYB5_YEAST returns a function comment and no cofactor or
    redox annotation.
18. Rhea TSV exports for RHEA:46680 and RHEA:62776.
19. WebFetch of the ACS landing page for Reid 1982 (doi 10.1021/ja00390a021): HTTP 403.
20. WebFetch of PMC3014896 (Aono 2010 full text) for the buffer and temperature.
21. Attempted retrieval of Thauer, Jungermann and Decker (1977) Bacteriol Rev
    41:100-180 from PMC413997. The publisher blocks XML full text and the scanned PDF
    is behind the PMC proof-of-work gate, so it could not be archived with a sha256.

## Evidence admitted

From `pmc4484759_parthasarathy2011_biochemistry.xml`
(sha256 `87ee6904b4c5597799b60c8717ccc3320ce0881fe29e76eafa7a33e473262b25`),
Parthasarathy S, Altuve A, Terzyan S, Zhang X, Kuczera K, Rivera M, Benson DR (2011)
Biochemistry 50(26):5544-54:

- Table 2: bCYB5A E prime zero = +5.5 +/- 1.8 mV vs SHE.
- Methods: "All experiments were conducted in 50 mM sodium phosphate buffer, pH 7.0
  (ionic strength = 0.11 M)". Calomel reference taken as +241 mV vs SHE. Titration
  temperature is not stated.
- Three independent determinations on the same protein at pH 7 and 24 to 25 degrees C
  and ionic strength about 0.1 M: +13 mV by redox potentiometry (Iyanagi 1977), +5 mV
  by OTTLE spectroelectrochemistry (Reid 1982), and -1.9 mV (Walker 1988).
- Class statement: "Type A cytochromes b 5 have been found to exhibit similar redox
  potentials (0 +/- 10 mV; all reported potentials vs. SHE)". Yeast Cyb5p is type A
  (microsomal, ER anchored). Type B proteins (outer mitochondrial membrane) sit near
  -55 mV and are the wrong analog.

Supporting, from `pubmed_21129218_aono2010_abstract.txt`: human microsomal b5 gives
-3.2 mV by spectroscopic electrochemical titration (pH 7.0, 25 degrees C, 50 mM sodium
phosphate, per the full text).

Method caveat, from `pubmed_9484218_rivera1998_abstract.txt`: potentials measured on
surface-modified electrodes run 40 to 100 mV positive of potentiometric values, so only
potentiometric or spectroelectrochemical numbers belong on the same scale.

Recombinant caveat, from `pubmed_2117468_funk1990_abstract.txt`: a recombinant bovine b5
already sits about 17 mV below the authentic protein at pH 7 and 25 degrees C.

## Partner couple

NAD+/NADH E prime zero = -320 mV vs SHE at pH 7, taken from
`pmc13340500_castaneda-losada2025_angew_review.xml`
(sha256 `c68a7b416f4a1ed0c598deb6c5041945384661d5a9cc61206cc7a613fd50e4ce`), which
attributes it to Bar-Even, Flamholz, Noor and Milo (2012) Biochim Biophys Acta
1817:1646-1659. Admission for that half is tabulated, not measured. The Thauer 1977
table carries the same value but could not be archived, as noted above.

With the two recorded potentials, dE prime zero = 5.5 - (-320) = 325.5 mV and, with
n = 2 for r_4197, drG prime zero = -n F dE = -62.8 kJ/mol. That number is computed from
the record, not measured, and it carries the provisional admission.

## Rhea representation

Rhea models this pair as a protein-bound carrier class rather than a small molecule.
RHEA:46680 (EC 1.6.2.2) is exactly the r_4197 chemistry:
`2 Fe(III)-[cytochrome b5] + NADH = 2 Fe(II)-[cytochrome b5] + NAD(+) + H(+)`.
RHEA:62776 (the sterol C4 methyl oxidase step, the GEM's r_0241) uses the same two
species. Both resolve to CHEBI:29033 iron(2+) and CHEBI:29034 iron(3+) attached to the
named polypeptide. One carrier class, two oxidation states, an electron count and a
midpoint potential, with no formation energy for either state.

## Same carrier class elsewhere in the GEM

The ER membrane copies s_4209 Ferrocytochrome b5 [erm] and s_4210 Ferricytochrome b5
[erm] carry no formula and are flagged `pseudo_metabolite`. They block r_4196
(NADH:ferricytochrome-b5 oxidoreductase) and appear in r_0241 (C-4 sterol methyl
oxidase). The same potential prices them. Note the GEM's own inconsistency: the
mitochondrial pair has formula Fe with charges 3 and 2 while the ER pair has no formula,
and the yeast enzyme is in fact ER localized.

## What a human still needs to do

1. Decide whether a bovine or human type A potential may stand in for yeast Cyb5p at
   all, or whether r_4197 and r_4196 stay unpriced.
2. If a real yeast number is wanted, it has to be measured: express the soluble heme
   domain of P40312 and run a mediated potentiometric titration at pH 7.0.
3. Fix the temperature. Parthasarathy 2011 does not state it for their own titrations.
4. Deposit Reid 1982 (J Am Chem Soc 104:7516-7519) and Thauer 1977 (Bacteriol Rev
   41:100-180) into the mirror so those primary values carry their own sha256 instead of
   being quoted secondhand.
