---
id: iqmpwr70icm8vo7psa0tqu5
title: Lipid Structure Writer
desc: ''
updated: 1789064540587
created: 1789064540587
---

## 2026.09.10 - Writing the tier-b lipid structures, and what they bought

### The question

[[experiments.026-metabolism-flux.external-pka-route]] left one open item as the only thing
that could change the reach number: 409 of the 665 unpriced yeast-GEM 9.0.2 metabolites, 206
distinct names, have a definite elemental formula and no structure in any namespace, and
they are lipids whose acyl composition is written into the name. That note's estimate was an
upper bound of 425 of 665 if a writer worked perfectly, with the caveat that the
group-decomposition rate on freshly written lipids was unknown and 2 of the 5 pilot
structures had failed decomposition.

The writer now exists: `experiments/026-metabolism-flux/scripts/lipid_structure_writer.py`,
five subcommands run under three environments, plus
`experiments/026-metabolism-flux/scripts/plot_lipid_structures.py` for the figure.

### What was built

| subcommand | environment | output |
|---|---|---|
| `parse` | torchcell | `results/lipid_names_parsed.json` |
| `build` | equilibrator-ext | `results/lipid_structures.json`, `results/lipid_structures.csv` |
| `pkas` | pka-tools | `results/lipid_pkas.json` |
| `price` | equilibrator-ext | `results/lipid_priced.json`, `results/lipid_priced.csv` |
| `coverage` | torchcell | `results/lipid_coverage.json` |

`parse` decomposes a name into a headgroup template and a list of acyl chains carrying
position, carbon count and double-bond count, as pydantic models. `build` assembles the
SMILES in RDKit and runs the acceptance test. `pkas` walks the MolGpKa ladder, importing
`_macro_from_sites` and `_apply_proton_change` from
`experiments/026-metabolism-flux/scripts/external_pka_benchmark.py` so the aggregation is
the same function the 300-compound benchmark scored rather than a second implementation.
`price` runs two routes and reports both. `coverage` reuses `shipped_series` from
`audit_reaction_thermo_coverage.py` and `TIGHT_SPREAD_KJ` from
`improve_parameter_coverage.py`; the `priceable` predicate is a closure inside
`audit_reaction_thermo_coverage.main` and cannot be imported, so it is restated in three
lines with a comment saying where it comes from.

**A correction to the earlier note's run instructions.** The environment that carries
`equilibrator-assets` master with `pkas_provided_externally` is
`~/miniconda3/envs/equilibrator-ext` (`0.6.0.post29+git.c79e371e`), not
`/scratch/projects/torchcell-scratch/envs/equilibrator`, which is on the 0.6.0 PyPI release
and has no such keyword.

### Design decisions, none of which is read off the name

1. **Double-bond position.** yeast-GEM writes `16:1` and never says where the double bond
   is. Every double bond is placed at Delta-9 counting from the carbonyl carbon, and at
   every third carbon after that, in the cis configuration. `build` asserts the convention
   rather than asserting it in prose: the free 16:1 and 18:1 acids it writes hash to
   `SECPZKHBENQXJG-FPLPWBNLSA-N` and `ZQPPMHVWECSIRJ-KTKRTIGZSA-N`, the InChIKeys of
   palmitoleic and oleic acid, and the run aborts if they do not. Group contribution reads a
   group vector that does not carry the double-bond position, so the choice does not move
   the energy; it is recorded because the structures are published and the convention is not
   recoverable from the name.
2. **Stereocenters are left unspecified**, including in the coenzyme A moiety. The glycerol
   sn-2 carbon, the inositol ring and the (R)-3-hydroxyacyl carbon all carry stereochemistry
   in the real molecule and none of it is in the yeast-GEM name for the glycerolipids.
   Double-bond geometry IS written, since the acyl-CoA names state cis or trans and the
   Delta-9 convention is cis. This decision has a measured cost, in the cache-lookup section
   below.
3. **Protonation.** Every structure is written neutral, so the reference microspecies
   eQuilibrator derives from the SMILES is the neutral one. yeast-GEM declares a formula at
   the model's own charge state, so the acceptance test compares against the neutral formula,
   `H_neutral = H_gem - charge_gem` with every other element unchanged. For all 168 names
   that pass, `charge_gem` is 0 and the adjustment is the identity; the code carries it
   anyway because the tier-b set contains charged species that do not parse.
4. **Coenzyme A comes from the cache, not from a keyboard.** `build` looks up InChIKey
   prefix `RGJOEKWQDUBAIZ`, neutralizes the stored -4 microspecies with RDKit's `Uncharger`,
   asserts the result is `C21H36N7O16P3S`, and thioesterifies it with a reaction SMARTS. The
   48 heavy atoms of the nucleotide are therefore not ours to get wrong.
5. **The SLIME backbone pseudo-metabolites are not molecules and no structure is
   attempted.** yeast-GEM's lipid pseudo-reactions carry species named `<class> backbone`
   whose formula is a difference formula: `triglyceride backbone` is `C3H2`, `ceramide
   backbone` is `HNO2`, `long-chain base backbone` is `H4N`. They are reported as unparsed
   with that reason rather than forced into a structure.

### Parser coverage

From `results/lipid_names_parsed.json`: **168 of the 206 distinct tier-b names parse, which
is 347 of the 409 metabolite rows.** The 38 that do not split into exactly two reasons.

| reason | n names |
|---|---|
| `slime_backbone_pseudo_metabolite` | 17 |
| `no_acyl_composition_in_name` | 21 |

Every unparsed name, by class:

| class | names |
|---|---|
| other (12) | `Apocytochrome c`, `Cytochrome c`, `Dextrin`, `Ferricytochrome b5`, `Ferrocytochrome b5`, `chitin`, `chitosan`, `heme a`, `heme o`, `mannan`, `pectin`, `tyrosyl acetate` |
| sphingolipid (6) | `ceramide backbone`, `inositol phosphomannosylinositol phosphoceramide backbone`, `inositol-P-ceramide backbone`, `long-chain base backbone`, `long-chain base phosphate backbone`, `mannosylinositol phosphorylceramide backbone` |
| glycan (4) | `(1->3)-beta-D-glucan`, `(1->6)-beta-D-glucan`, `N,N'-Chitobiosyldiphosphodolichol`, `Starch` |
| protein_linked (4) | `[cytochrome c]-L-lysine`, `[cytochrome c]-N6-methyl-L-lysine`, `[protein]-L-lysine`, `[protein]-N(6)-acetyl-L-lysine` |
| phosphatidylinositol_PI (3) | `1-phosphatidyl-1D-myo-inositol backbone`, `sn-2-acyl-1-lysophosphatidylinositol backbone`, the GPI anchor `6-[6-(2-aminoethylphosphoryl)-alpha-mannosyl-(1->6)-...]-O-acyl-1-phosphatidyl-1D-myo-inositol` |
| triacylglycerol_TAG (2) | `diglyceride backbone`, `triglyceride backbone` |
| one each (7) | `cardiolipin backbone`, `ergosterol ester backbone`, `phosphatidate backbone`, `phosphatidyl-L-serine backbone`, `phosphatidylcholine backbone`, `phosphatidylethanolamine backbone`, `phosphatidylglycerol backbone` |

**The sphingolipids are all backbones.** The reach analysis counted 12 sphingolipid
metabolites in tier b, and every one of them is a SLIME pseudo-metabolite: yeast-GEM 9.0.2
carries no ceramide, inositol phosphoceramide or mannosylated inositol phosphoceramide with
an acyl composition in its tier-b name, so there was nothing for a ceramide template to
build. The template set covers glycerophospholipids (PA, PE, PMME, PDME, PC, PS, PG, PGP,
PI and its 3-, 4-, 3,5- and 4,5-phosphates, DGPP, CDP-DAG), triacyl-, diacyl- and
monoacylglycerols, cardiolipin and monolysocardiolipin, lysophosphatidylinositol,
acylglycerone phosphate, free fatty acyl chains and acyl-CoAs. PS and DAG have templates and
no tier-b name uses them; they exist because the same parser is run over all 2,806 yeast-GEM
metabolites for the figure's comparison population, where it recognizes 287 distinct lipid
names.

### The acceptance test

**168 of 168 written structures reproduce yeast-GEM's own formula exactly**
(`results/lipid_structures.json`, `n_formula_pass: 168`, `n_formula_fail: 0`, covering 347
metabolite rows). There are no failures to list. By template: CL 60, TAG 18, ACYLCOA 9,
PI35P2 8, DGPP 7, PI45P2 7, PI4P 7, PDME 7, PMME 7, PI3P 6, FFA 6, PE 6, PGP 5, CDPDAG 5,
MAG1 3, PC 2, PG 2, PI 1, ACYLDHAP 1, LPI2 1.

A zero-failure acceptance test is weak evidence on its own, so two independent checks were
run. The Delta-9 witnesses above are one. The other is the cache: 120 of the 168 written
InChIKeys resolve a compound eQuilibrator already holds, which is an external structure
nobody here wrote. Our phosphatidylcholine (1-16:0, 2-18:1) hashes to
`WTJKGGKOPKCXLL-MRCUWXFGSA-N` and POPC in the cache to `WTJKGGKOPKCXLL-VQVTYTSYSA-N`, the
same constitution written with and without the sn-2 center.

### Pricing, and the two routes

`price` runs two routes per structure at pH 7.2, I = 0.25 M, T = 303.15 K, pMg 3.0.

**Route 1, cache lookup.** The audit could never try this: it had no structure to key on. A
written structure gives one. The search runs twice, first on the skeleton and
stereochemistry blocks (the fix `pathway_thermo.py` needed, since the cache stores the major
microspecies and the last block records the protonation state), then on the skeleton alone,
because design decision 2 leaves stereocenters unspecified while the cache's entries often
carry them. Both passes require every element but hydrogen to match the written formula, so
a hit cannot be a different molecule. Result: **120 of 168, split 19 on stereo plus
skeleton and 101 on skeleton alone.**

**Route 2, external pKa.** MolGpKa ladders for all 168 structures in 74 s
(`results/lipid_pkas.json`), then `LocalCompoundCache(pkas_provided_externally=True)` on
assets master. 148 compounds were created, 20 were not because their ladder is empty, and
**148 of 148 created compounds passed group decomposition.**

That last number answers the earlier note's open question. Group decomposition was flagged
as an unmeasured ceiling after 2 of 5 pilot structures failed it; on written lipids the
failure rate is 0 of 148. The two pilot failures, ferroheme b and methionyl acetate, are an
iron porphyrin and an ester of an amino acid, not lipids, so they were never evidence about
this class.

**All 168 names are priced**, 120 by cache lookup and 48 by the external route, covering 347
metabolite rows (`results/lipid_priced.json`).

**The 20 empty ladders are 18 triglycerides and 2 phosphatidylcholines**, and they are the
shape the interface cannot express: a triacylglycerol has no ionizable group and a
phosphatidylcholine zwitterion has none between pH 0 and 14, so the correct pKa list is
empty, and `local_compound_cache.py:128` guards on truthiness and raises `MissingPKasError`
instead of creating a single-microspecies compound. All 20 were recovered by the cache
lookup, so the interface limitation costs nothing here, but it is still a real defect and it
will bite any compound that is genuinely absent and genuinely non-ionizable.

Of the 48 that only the external route reached, 26 are cardiolipins, 6 DGPP, 6 PDME, 6 PMME,
3 CDP-DAG and 1 acyl-DHAP. None of their ladders hit the rung budget.

### A created compound is not on the same energy zero as a cached one

This is the result that changes how the numbers should be used, and it was not anticipated.

For 90 names both routes produced a number. Their `dGf'0` values differ by 1930 kJ/mol on
average, up to 3023. That is not a matching error: the InChIKey skeleton and every heavy
atom agree, so the two are the same chemical. Restricting to the 7 pairs whose reference
microspecies carries the same number of protons isolates the drawing convention from the
protonation state, and there the gap is almost a constant:

| name | formula | cache minus external (kJ/mol) |
|---|---|---|
| phosphatidylethanolamine (1-16:0, 2-16:1) | C37H72NO8P | -776.11 |
| phosphatidylethanolamine (1-16:0, 2-18:1) | C39H76NO8P | -776.11 |
| phosphatidylethanolamine (1-16:1, 2-18:1) | C39H74NO8P | -776.11 |
| phosphatidylethanolamine (1-18:0, 2-16:1) | C39H76NO8P | -776.11 |
| phosphatidylethanolamine (1-18:0, 2-18:1) | C41H80NO8P | -776.10 |
| phosphatidyl-N-methylethanolamine (1-16:0, 2-16:1) | C40H78NO8P | -779.22 |
| phosphatidyl-N,N-dimethylethanolamine (1-16:0, 2-16:1) | C41H80NO8P | -782.84 |

Taking phosphatidylethanolamine (1-16:0, 2-16:1) apart: the cache compound is id 57341,
SMILES `...COP([O-])(=O)OCC[NH3+]...`, atom bag `{C: 37, H: 72, N: 1, O: 8, P: 1}`,
untransformed -1087.72 kJ/mol, Legendre transform +3066.82, `dGf'0` 1979.09. The created
compound is id 694430, SMILES `...COP(=O)(O)OCCN`, the same atom bag, untransformed
-311.47, transform +3066.68, `dGf'0` 2755.20. **The transforms agree to 0.14 kJ/mol and the
entire gap is in the untransformed term**, which is the group-contribution estimate for
whichever microspecies the SMILES draws. Plausible mechanism, unverified: eQuilibrator's
component contribution was trained on ChemAxon's major microspecies, so the group vector for
a phosphate anion beside an ammonium and the group vector for a phosphate acid beside an
amine are both in range but are not required to agree, and drawing the un-ionized protomer
puts the compound outside the convention.

Two consequences.

- **Where a cache compound exists it is the value reported**, because the recomputed table
  the flux layer reads is built from cache compounds. That is what `route: cache_lookup`
  means in `results/lipid_priced.csv`, and it is 120 of 168 names, 289 of 347 metabolite
  rows.
- **The 48 external-route values carry an offset of unknown size.** It is 777 kJ/mol on the
  aminophospholipids where it can be measured, and it has not been measured on a
  cardiolipin, a DGPP or a CDP-DAG, which is what those 48 are. They are reported and
  counted, and the coverage arithmetic below gives the cache-route-only number beside the
  full one so a claim can pick the conservative one.

This also retires a reading of the earlier pilot. That note put the written
phosphatidylethanolamine (1-16:1, 2-16:1) at 1967.93 kJ/mol beside the cache-priced
phosphatidylethanolamine (1-18:1, 2-16:1) at 2413.32 and called them the same family. They
are not on one zero, and the difference between two such numbers is not a chemistry
difference.

### Coverage

From `results/lipid_coverage.json`, over yeast-GEM 9.0.2's 4,131 reactions and 2,806
metabolites.

**347 of the 665 unpriced metabolites are now priced**, 52.2 percent, of which 289 come
through the cache lookup and 58 through the external route. By class: PI 131, CL 60, TAG 51,
PE 50, PA 23, acyl-CoA 9, fatty acyl chain 8, PC 8, PG 7.

| route | reactions with every participant priced | fraction |
|---|---|---|
| eQuilibrator recomputation, before | 2741 | 66.35 % |
| plus written lipids, cache route only | 3257 | 78.84 % |
| plus written lipids, both routes | 3564 | 86.27 % |
| yeast-GEM shipped metabolite column | 3505 | 84.85 % |
| union of shipped and recomputed, before | 3716 | 89.95 % |
| plus written lipids | 3716 | 89.95 % |
| recommended set (union plus tight surrogates), before | 3730 | 90.29 % |
| plus written lipids | 3732 | 90.34 % |
| recommended, restricted to the 2,845 flux-carrying reactions, before | 2685 | 94.38 % |
| plus written lipids | 2685 | 94.38 % |

**The gain is 823 reactions on the eQuilibrator route and 0 on the union.** Read those two
rows together, because the second is the one that governs what to claim. yeast-GEM's shipped
metabolite column already carries values for these lipids, matched by metabolite name out of
yETFL and ModelSEED, so at the union level the writer adds nothing: 3716 before and 3716
after. The recommended set gains 2 reactions, and the flux-carrying restriction that the
parameter-coverage report quotes at 94.38 percent does not move at all.

What the writer buys is **method consistency, not coverage**. Before it, one self-consistent
eQuilibrator recomputation covered 2,741 reactions, 764 fewer than the shipped column's
3,505, and the shipped column is three different methods behind one column (yETFL by name,
ModelSEED by KEGG id, dG-predictor by row position; see
`audit_reaction_thermo_coverage.py`'s `SHIPPED_PROVENANCE`). After it, the eQuilibrator
route covers 3,564, which is 59 more than the shipped column, and every one of those values
comes from the same group-contribution method with an uncertainty attached. That is the
result worth quoting, and the 90.0 percent union is unchanged.

### Figure

![**Figure 1. What the lipid structure writer reached, and what it bought.** **a**, The 206
distinct tier-b names by yeast-GEM chemical class, each bar partitioned into the five
mutually exclusive stages of the pipeline. No name is lost to a formula mismatch and none to
a written structure that failed to price, so those two segments are empty everywhere; the
whole loss is the 38 names the parser refuses, which are the 17 SLIME backbone
pseudo-metabolites and 21 names with no acyl composition (glycans, hemes, cytochromes,
protein-linked lysines, a GPI anchor). **b**, Reactions of 4,131 with every participant
priced, before the written lipids, with the 120 names that resolve an existing cache
compound, and with all 168. The eQuilibrator recomputation moves from 66.35 to 86.27
percent and passes the shipped metabolite column's 84.85 percent (dashed). The union with
the shipped table does not move at all, because that table already prices these lipids. **c**,
Uncertainty on the formation energy, one value per distinct name: the 113 lipid names the
recomputed table already prices, the 120 written names priced by cache lookup, and the 48
priced through MolGpKa pKas and a created compound. Medians 13.19, 14.79 and 18.92 kJ/mol.
Drawn by `experiments/026-metabolism-flux/scripts/plot_lipid_structures.py` from
`results/lipid_names_parsed.json`, `results/lipid_structures.json`,
`results/lipid_priced.json` and
`results/lipid_coverage.json`.](./assets/images/026-metabolism-flux/lipid_structure_writer.svg)

### The extended compound cache, and where it lives

Compounds created here are written to
`/scratch/projects/torchcell-scratch/tmp/lipid_ccache.sqlite`, a copy of the shipped cache
made before the run. **The shipped cache at `~/.cache/equilibrator/compounds.sqlite` is never
opened for writing**, and `price` refuses to run if `--cache` resolves to it, because a
compound created here would otherwise be indistinguishable from a MetaNetX one on the next
read. The copy holds the 148 created compounds and is reusable: point `price --cache` at it,
or open it with `LocalCompoundCache.load_cache`. It sits under `DATA_ROOT/tmp` rather than in
the agent job directory so that it survives the session, next to the pilot's
`pka_local_ccache.sqlite`.

### What remains

- **The 777 kJ/mol convention offset is measured on aminophospholipids only.** Closing it
  means writing each structure as its major microspecies at pH 7.2 rather than as the
  neutral parent, which the MolGpKa ladder already contains enough information to do. Until
  that is done the 48 external-route values should not be mixed into a table with cache
  values, and `lipid_coverage.json` reports the cache-route-only counts for that reason.
- **The compartment-pH mismatch.** The new values are all at pH 7.2, while
  `metabolite_dgf_prime.parquet` transforms each metabolite at its own compartment pH: 5.0
  extracellular and in the cell envelope, 6.2 in the vacuole, 6.6 in the Golgi, 7.0 in the
  peroxisome, 7.1 in the endoplasmic reticulum, 7.8 in the mitochondrion, 7.2 in the
  cytoplasm, nucleus and lipid particle. Only the reaction COUNTS in
  the table above are comparable; before any of these numbers enters the flux layer they
  need re-transforming per compartment, which is a call to `Compound.transform` and not a
  re-derivation.
- **Magnesium is still unsuppliable** through `specified_pkas`, so the 48 created compounds
  carry an empty magnesium dissociation list. Measured cost elsewhere: 0.32 kJ/mol for UDP,
  1.10 kJ/mol for ADP.
- **38 names remain unwritable and 318 of the 665 metabolites remain unpriced.** The 17
  backbones are pseudo-metabolites and no structure should exist for them; the 21 others are
  polymers, metalloproteins and protein-linked residues, which is tier c's ceiling arriving
  under a different label.
- **The empty-pKa interface defect** (`local_compound_cache.py:128` guards on truthiness) is
  worth reporting upstream. It cost nothing here only because every affected compound
  happened to be in the cache already.

## 2026.09.11 - Re-pricing the 48 external-route lipids, and why the writer cannot reach the other 62

### The convention was wrong, and it was worth 1,585 kJ/mol

The 48 lipid records whose route is `external_pka` were priced by submitting the WRITTEN
NEUTRAL structure to the compound bridge. eQuilibrator derives a compound's reference
microspecies from the SMILES it is handed, so that anchors the formation energy to a
different microstate than the cache's own entries, which are ChemAxon's major microspecies
at pH 7. `external_pka_reprice.json` measured the size of that on 7 targets priced both
ways: mean residual against the cache -777.5 kJ/mol on the neutral parent against +0.128
kJ/mol on the pH 7 microspecies.

`experiments/026-metabolism-flux/scripts/external_pka_apply_selection.py` re-prices them
under the pinned convention with the pKa method chosen in
[[experiments.026-metabolism-flux.pka-method-selection]]. Result
(`results/external_pka_applied.json`): **48 of 48 lipid targets priced, covering 58
yeast-GEM metabolite rows**, with a median absolute change of **1,585.2 kJ/mol** (range
779.2 to 1,985.3). The old values were wrong by that much, not by a rounding error.

The selected method is the cascade Uni-pKa simple template -> MolGpKa -> pKasolver-lite,
and on these 53 targets Uni-pKa answered for 52 and pKasolver-lite for the remaining one;
MolGpKa was never reached. The cascade earns its place on the 2,400-compound benchmark
rather than here, where it degenerates to its first member.

The split of labor inside the re-pricing is deliberate and is recorded per compound. The
pH 7 major microspecies comes from pKasolver's `States.ph7_mol`, which dimorphite-dl
enumerates: that is a protonation-state enumerator, not a pKa regressor. The pKa VALUES
come from the selected method. Using one tool for the convention and another for the values
is the same split `reprice-microspecies` measured the convention shift with.

By chemical class, the 48 are 24 monolysocardiolipins, 6 diacylglycerol 3-diphosphates, 6
phosphatidyl-N,N-dimethylethanolamines, 6 phosphatidyl-N-methylethanolamines, 3
CDP-diacylglycerols, 2 cardiolipins and 1 acylglycerone phosphate.

The 120 cache-lookup lipids are NOT re-priced and must not be: a cache compound is already
on the cache's own convention, and it carries ChemAxon pKas and magnesium dissociation
constants that no external route can reproduce.

### The tier-a targets: 2 of 5 close, and the other 3 say why

Same run, same convention. `phosphatidylethanolamine (1-16:1, 2-16:1)` prices at 1,971.11
kJ/mol over 6 metabolite rows and `N-(4-oxoglutarate)-L-cysteinylglycine` at -539.61 over
2. The other three do not, and each carries its reason rather than a blank:

| target | rows | outcome | why |
|---|---|---|---|
| ferroheme b | 1 | `compound_creation_failed` | the iron macrocycle has no group decomposition; a pKa list cannot supply one |
| methionyl acetate | 2 | `compound_creation_failed` | same, on the mixed anhydride |
| superoxide | 2 | `compound_creation_failed` | the group decomposer refuses it. Under MolGpKa alone this target had failed one step earlier, with no pKa inside (0, 14) at all; the cascade's first member supplies one and the compound still cannot be created, which moves the blocker from the pKa stage to the decomposition and does not close it |

### Why the writer cannot compose the remaining 62 tier-b rows

`results/lipid_names_parsed.json`: 206 distinct tier-b names over 409 metabolite rows, 168
names parsed covering 347 rows, **38 names unparsed covering 62 rows**. Two reasons, and
they are different kinds of thing.

| class | reason | names | rows |
|---|---|---|---|
| other | `no_acyl_composition_in_name` | 12 | 16 |
| sphingolipid | `slime_backbone_pseudo_metabolite` | 6 | 12 |
| glycan | `no_acyl_composition_in_name` | 4 | 6 |
| triacylglycerol TAG | `slime_backbone_pseudo_metabolite` | 2 | 5 |
| protein-linked | `no_acyl_composition_in_name` | 4 | 4 |
| phosphatidylinositol PI | `slime_backbone_pseudo_metabolite` | 2 | 3 |
| phosphatidylethanolamine PE | `slime_backbone_pseudo_metabolite` | 1 | 3 |
| cardiolipin CL | `slime_backbone_pseudo_metabolite` | 1 | 2 |
| sterol ester | `slime_backbone_pseudo_metabolite` | 1 | 2 |
| phosphatidate PA | `slime_backbone_pseudo_metabolite` | 1 | 2 |
| generic pseudo-metabolite | `slime_backbone_pseudo_metabolite` | 1 | 2 |
| phosphatidylcholine PC | `slime_backbone_pseudo_metabolite` | 1 | 2 |
| phosphatidylglycerol PG | `slime_backbone_pseudo_metabolite` | 1 | 2 |
| phosphatidylinositol PI | `no_acyl_composition_in_name` | 1 | 1 |

**`slime_backbone_pseudo_metabolite`, 17 names over 37 rows, is closed and always will
be.** A SLIME backbone is not a molecule: its formula is a DIFFERENCE formula, the part of
a lipid class that is not acyl chain. `triglyceride backbone` is `C3H2`, `ceramide
backbone` is `HNO2`. No structure writer, pKa predictor or decomposer applies, because
there is nothing to write a structure of. In the staged pipeline these carry
`UnpricedReason.PSEUDO_METABOLITE`, and they are the single largest remaining block: 215 of
the 393 unpriced reactions are blocked by a pseudo-metabolite and nothing else.

**`no_acyl_composition_in_name`, 21 names over 25 rows, needs a source rather than a
parser.** These are names that give a class and no chains: glycans, protein-linked species,
and a handful of unclassified lipids. A structure exists in principle and the name does not
determine it, so composing one would be inventing a species. They carry
`UnpricedReason.NO_STRUCTURE_AVAILABLE` and the remedy is a structure from a namespace or a
curator, not more parsing.

### Files

- `experiments/026-metabolism-flux/scripts/external_pka_apply_selection.py`
- `experiments/026-metabolism-flux/results/external_pka_apply_targets.json`
- `experiments/026-metabolism-flux/results/external_pka_apply_ladders_molgpka.json`
- `experiments/026-metabolism-flux/results/external_pka_apply_ladders_pkasolver.json`
- `experiments/026-metabolism-flux/results/external_pka_apply_microspecies.json`
- `experiments/026-metabolism-flux/results/external_pka_apply_combined.json`
- `experiments/026-metabolism-flux/results/external_pka_applied.json`
