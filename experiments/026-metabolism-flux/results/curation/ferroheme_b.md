# ferroheme_b (s_0712) curation log

Date: 2026.09.11. Worktree: `feat/kinetics-equilibrator-datasets`.
Artifacts: `$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/ferroheme_b/`.
Record: `experiments/026-metabolism-flux/results/curation/ferroheme_b.json`.

Row: `s_0712`, ferroheme b [m], GEM formula C34H30FeN4O4, charge -2.
Blocked reactions r_0436 (ferrochelatase), r_0531 (heme O synthase), r_4152 (cytochrome c heme lyase).

## Outcome in one line

`kind = pka_ladder`, `admission = provisional`: two propionate rungs at 4.75 and 4.15,
statistically split from the single measured macroscopic value 4.45 (range 4.4 to 4.5) of
Das and Medhi 1998, ferrous species, 27 C, ionic strength 0.2 M, in aqueous surfactant
micelles. Not aqueous monomeric heme, so it stays out of the default route. Supplying it
will not unblock the three reactions, for the reason in the next section.

## What was actually run, and what it showed

### Group decomposition (the real blocker)

`equilibrator_assets.group_decompose.GroupDecomposer().Decompose(...)` under
`~/miniconda3/envs/equilibrator-ext`, on five structures. Artifact
`group_decomposition_check.json`, sha256 `50fce71e...`.

| structure | decomposes |
|---|---|
| Fe-coordinated neutral | no, GroupDecompositionError |
| Fe-coordinated dianion | no, GroupDecompositionError |
| iron-stripped protoporphyrin IX neutral | no, GroupDecompositionError |
| iron-stripped protoporphyrin IX dianion | no, GroupDecompositionError |
| 3-(2-methyl-1H-pyrrol-3-yl)propanoic acid control | yes, 7 nonzero groups |

The iron-free macrocycle fails as well, so the porphyrin ring, not the iron and not the
pKa ladder, is what component contribution cannot handle. The cache agrees: its own heme
entry carries a `group_vector` of length 0. **The row's stated premise that the structure
decomposes is wrong.** A pKa ladder alone cannot price s_0712.

### Why the pKa cascade behaved the way it did

Artifact `predictor_check_ferroheme_b.json`, sha256 `0f6d4be0...`; the raw Uni-pKa
enumerator output is `unipka_enumerator_ensemble.tsv`, sha256 `4b8963ac...`.

**Uni-pKa, simple template: fails at enumeration, and the cause is exact.** The
enumerator writes an empty ensemble row for the Fe-coordinated SMILES at every charge
from -4 to +2, so no macrostate pair reaches the model. `FILTER_PATTERNS` in
`Uni-pKa/enumerator/main.py` lists the SMARTS `[*-2]` as an unreasonable structure, and
`[Fe-2]` matches it once, so `sanitize_checker` returns False for every heme microstate.
Verified by direct call: `sanitize_checker(fe_neutral)` is False and the `[*-2]` match
count is 1. The same run enumerates the iron-stripped macrocycle over charges -3 to +1
and the propionate fragment over -2 to 0. Inference was never reached, so no GPU was
needed and none was used.

**MolGpKa: succeeds, on the right input form.** On the Fe-coordinated NEUTRAL form it
returns exactly the two propionates, 4.371 and 4.598, and no basic site. On the
Fe-coordinated DIANION it returns nothing. The apply pipeline handed it the dianion, so
the empty MolGpKa ladder in `external_pka_apply_ladders_molgpka.json` is an
input-protonation-form artifact, not a tool limit for this molecule.

**pKasolver-lite: succeeds, and is what the cascade actually used.** [3.394, 3.301] on
both Fe forms. dimorphite-dl rejects every pyrrole-N protonated Fe microstate as a
poorly formed SMILES, so it walks only the two propionates, which is the correct site
set. The values sit about 1.1 units below the measured micellar ferrous value and near
the ferric one.

So the cascade did not return nothing. `external_pka_apply_combined.json` records
`members_used: ["pkasolver"]` with pKas [3.394, 3.301], and
`external_pka_applied.json` then records `outcome: compound_creation_failed`,
`detail: method=empty status=failed`. The exclusion reason `no_pka_prediction` in
`thermo_exclusions.json` predates that run.

### What the eQuilibrator compound cache holds

Artifact `equilibrator_cache_heme_extract.json`, sha256 `d9cc78ea...`, read from
`/home/michaelvolk/.cache/equilibrator/compounds.sqlite` (sha256 `12a28e5a...`).

Two entries, compound 236 (`pheme`, CHEBI:17627, CHEBI:60344, MNXM249, KEGG via 495078)
and compound 495078 (`C00032`, HMDB03178). Both carry:

- InChIKey `KABFMIBPWCXCRK-RGGAHWMASA-J` and `-L`, and an InChI whose formula layer is
  `C34H34N4O4.Fe` with `/q;+2/p-4`, so the InChI does contain iron.
- A stored SMILES with **no iron**, and `atom_bag = {C: 34, O: 4, N: 4, H: 32, e-: 298}`,
  also with no iron. The electron count 298 balances only for the iron-free dianion.
- `dissociation_constants = [5.06, 4.50, 4.08, 3.62]`, ChemAxon-predicted, so **computed**.
- Five microspecies from charge -2 (32 H, major) to +2 (36 H), that is two carboxylate
  rungs plus two pyrrole-N protonations.
- `group_vector` of length 0, that is no decomposition.

Two consequences. The four cached pKas are for the metal-free porphyrin: the two pyrrole
rungs cannot exist once Fe(II) occupies the core, so the list is computed and for the
wrong species. And the missing iron in the atom bag is exactly why every accession route
in `unresolved_metabolites.json` returned `composition_mismatch` for s_0712.

### Structure identity

`chebi_60344_ols4.json` (sha256 `3e28fd6e...`) confirms CHEBI:60344 is ferroheme b(2-),
C34H30FeN4O4, charge -2, "major species at pH 7.3", InChIKey
`KABFMIBPWCXCRK-RGGAHWMASA-J`, matching the GEM row exactly.

The prior pass's lead pointed at PubChem CID 53356659. That record
(`pubchem_cid_53356659.json`, sha256 `ce6f2137...`) is `C34H32FeN4O4-2`, but its -2 sits
on the two pyrrolide nitrogens with **both propionic acids protonated** and the iron
drawn as a disconnected `[Fe]`. It is a different microspecies from the GEM's, and the
lead should not be reused as the row's structure.

## Literature: what exists and what does not

**Carries the value.** Das D K and Medhi O K, J Inorg Biochem 70(2):83-90 (1998), doi
10.1016/s0162-0134(98)10002-8, PubMed 9666570 (`pubmed_9666570.xml`, sha256
`7264a646...`):

> The dependence of mid-point potential on pH indicates that the heme propionate
> undergoes a redox-linked changed in pKa from 3.2-3.5 in the ferric form to 4.4-4.5 in
> the ferrous form (27 degrees C, mu = 0.2 M).

Two caveats that belong with the number. The medium is aqueous surfactant micelles (SDS,
Triton X-100, CTAB), stated in the same abstract. And the ferrous value is inferred from
the pH dependence of the square-wave midpoint potential, about -59 mV per pH unit, not
titrated directly; the directly titrated Soret value in that paper, about 3.5, is the
ferric one.

**Explains why no aqueous value exists.** Asher C, de Villiers K A, Egan T J, Inorg Chem
48(16):7994-8003 (2009), doi 10.1021/ic900647y, PubMed 19572726 (`pubmed_19572726.xml`,
sha256 `98393ba0...`): Fe(III)PPIX in water is a concentration-, pH-, salt-, temperature-
and solvent-dependent mixture of monomer, pi-pi dimer, mu-oxo dimer and stacked
aggregates. That is what drives every quantitative study into micelles, vesicles or
co-solvent.

**Negatives, each searched for and not found.**

- No measured propionate pKa for ferrous heme in aqueous monomeric solution.
- No measured Keq or dGr for ferrochelatase (EC 4.99.1.1); no TECRDB entry surfaced.
  Only kinetics, kchel about 0.97 per second and Ks about 16.3 uM at pH 8.1.
- No measured equilibrium for cytochrome c heme lyase (r_4152). Its product is
  protein-bound heme, so a small-molecule value is not expected to exist.
- No published standard Gibbs energy of formation for heme or protoheme.

**One claim checked and refuted.** A search summary asserted pKas of 8.7 plus or minus
0.1 and 10.9 plus or minus 0.2 for a heme propionate and a coordinated water, attributed
to Silver et al., JBIC 27:297-313 (2022). The full text was fetched from Europe PMC
(PMC8960585, `pmc8960585.xml`, sha256 `2a0f2378...`) and grepped: it contains neither
value and reports no propionate pKa at all. Discarded.

**One lead left open.** Savitski A P, Vorobyova E V, Berezin I V, Ugarova N N, J Colloid
Interface Sci 84:175-181 (1981), doi 10.1016/0021-9797(81)90272-1
(`crossref_jcis1981.json`, sha256 `fdf60409...`), titrates the propionic acid residues of
both protoporphyrin IX and heme. ScienceDirect returned HTTP 403 and PubMed does not
index it, so only Crossref metadata was obtained and no number from it entered the
record. It is micellar too, so at best it splits the single value into two site values
without changing the admission class.

**Also checked, not usable.** Kim et al., JACS 133:8544 (2011) measures PCET at a heme
propionate but in acetonitrile. The open-access review "Role of Propionate Side Chain in
Heme-Containing Metalloenzymes" (PMC12679310, sha256 `5d93ca17...`) was fetched and
grepped: every pKa in it is for a propionate inside a protein, for example about 5.9
oxidized and about 7.0 reduced in Pseudomonas aeruginosa cytochrome c551.

## Searches run

WebSearch, all on 2026.09.11:

1. propionate pKa ferrous protoporphyrin IX aqueous solution measured heme
2. "heme propionate" pKa measured JACS iron protoporphyrin IX model compound proton-coupled electron transfer
3. hemin aqueous solution acid dissociation constant propionic acid groups pKa aggregation dimerization monomeric
4. ferrochelatase reaction equilibrium constant thermodynamics protoporphyrin IX Fe2+ pH dependence proton stoichiometry
5. protoporphyrin IX propionic acid pKa 3.68 4.96 spectrophotometric titration measured
6. Fe(II) protoporphyrin IX ferrous heme pKa propionate "aqueous" titration Biochemistry deoxyheme dithionite
7. "8.7" "10.9" pKa heme propionic acid coordinated water Fe(II) protoporphyrin IX aqueous solution high pH Silver Wilson
8. eQuilibrator heme protoheme formation energy component contribution group decomposition iron metal complexes not supported
9. "ferric heme b in aqueous micellar and vesicular systems" Quarterly Reviews of Biophysics state-of-the-art challenges monomeric speciation
10. free heme propionate carboxylate pKa water potentiometric titration "protoporphyrin IX" iron measured non-micellar aqueous DMSO cosolvent
11. cytochrome c heme lyase thioether bond formation equilibrium reversible pH dependence thermodynamics free energy
12. ferrochelatase EC 4.99.1.1 equilibrium constant Keq NIST thermodynamics of enzyme-catalyzed reactions TECRDB protoheme
13. Gibbs energy of formation heme protoheme thermodynamic estimate group contribution literature kJ/mol standard

API searches:

14. Europe PMC REST, query `"heme propionate" AND "pKa"`, 204 hits, all protein-bound.
15. PubMed esearch for the Savitski 1981 title, 0 hits (not indexed).

Fetches that failed: ScienceDirect 0021979781902721 returned HTTP 403; the ChEBI SOAP
`getCompleteEntity` endpoint returned HTTP 500 for both CHEBI:17627 and CHEBI:60344 (the
failed responses are kept as `failed_chebi_soap_*.html`), and OLS4 was used instead.

## What a human still needs to do

1. Decide the route. A pKa ladder cannot price this metabolite; the porphyrin ring does
   not decompose. Either accept r_0436, r_0531 and r_4152 as permanently unpriced, or
   source a dGf for heme b from outside component contribution and carry it with its own
   admission class.
2. Read Savitski et al. 1981 through institutional access. It is the last primary source
   that might give two separate propionate site values.
3. Fix the cache entry's missing iron, or make the accession path say so. Today it
   reports `composition_mismatch` with no hint that the cache, not the GEM, is wrong.
4. Two pipeline defects, independent of this row. MolGpKa is fed the anionic target
   SMILES rather than the neutral parent, which empties its ladder on any already
   deprotonated input. The Uni-pKa enumerator's `[*-2]` filter rejects every
   metal-complex microstate, so Uni-pKa is unavailable for that whole class.
5. If the two-rung ladder is ever used, keep it labeled as a statistical split of one
   measured macroscopic value. It is not two measured site pKas.
