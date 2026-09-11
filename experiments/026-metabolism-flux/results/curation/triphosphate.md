# triphosphate (s_4098 [c], s_4099 [e]) curation log

Date: 2026-09-11. Curated against yeast-GEM 9.0.2, formula `O10P3`, charge -5.
Blocked reaction of interest: `r_4403`, triphosphate phosphohydrolase,
H2O + triphosphate -> diphosphate + phosphate. `r_4369` (proton symport, same species
both sides) and `r_4528` (exchange) close by separate rules and are out of scope.

Result in one line: a **measured pKa ladder** was found and admitted; **no formation
energy and no reaction energy** was found anywhere, so `r_4403` stays blocked.

Mirror, every artifact with its sha256:
`$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/triphosphate/`

## What the pipeline itself already says

Run first, before any searching, so the search had a target.

| probe | result |
|---|---|
| eQuilibrator cache, compound 314 (`kegg:C00536`) | present, with a structure and six microspecies |
| `standard_dg_formation` | `(None, None, None)` |
| `group_vector` | empty list, length 0 |
| present in component contribution `train_S` (631 x 3999) | **no**, 0 training reactions |
| phosphate (id 12) for contrast | `group_vector` also empty, 659 training reactions, dfG = -1095.21 kJ/mol |
| diphosphate (id 14) for contrast | `group_vector` also empty, 345 training reactions, dfG = -1966.80 kJ/mol |
| `standard_dg_prime` of the r_4403 reaction | returns the infinite uncertainty sentinel, 1e5 kJ/mol |
| yeast-GEM `model_metDeltaG.csv` rows 2579, 2580 | `10000000`, the release's no-value sentinel |
| yeast-GEM `model_rxnDeltaG.csv` row 3760 | `r_4403,10000000` |

The first correction this produced: the exclusion label `group_decomposition_failed` is
true but not the operative reason. Phosphate and diphosphate carry an empty group vector
too and are priced anyway, from the reactant contribution. Triphosphate cannot use that
route because it appears in none of the 3999 training reactions. The binding statement is
"no group decomposition **and** absent from the training set", and no pKa predictor or
structure writer can move the second half.

Artifacts: `equilibrator_cache_probe.json`, sha256
`4e99aba6fb705102177a39b5a32e1189b8610c11b2051bb5ec64bb70c1032988`.

## Decomposition on alternative protonation states

Asked, per the brief, whether some other protonation state decomposes. It does not.

| SMILES tried | decomposes | with `ignore_protonations=True` |
|---|---|---|
| P3O10(5-) | no | no |
| HP3O10(4-), the cache's own SMILES | no | no |
| H2P3O10(3-), two terminal protons | no | no |
| H2P3O10(3-), bridge protonated | no | no |
| H3P3O10(2-) | no | no |
| H5P3O10 neutral acid | no | no |
| P2O7(4-) control | no | no |
| PO4(3-) control | no | no |
| ATP control | **yes**, 13 groups | yes |

Every failure is the same `GroupDecompositionError: Unable to decompose ... into groups`.
The controls settle what it means: ATP decomposes and its groups include
`-OPO3 [H0 Z-2 Mg0]` and `-OPO2-OPO2- [H0 Z-2 Mg0]`, so the group system recognizes
phosphate groups only when they hang off a carbon skeleton. A bare inorganic polyphosphate
has no such anchor in any protonation state.

Artifact: `group_decomposition_probe.json`, sha256
`0f9466b77ea1d4ef6fbeba476d5976ad9ad2bb9d8bab1f9b66e52c9ef1e39f6c`.

## Search for a tabulated formation energy

**NBS tables (Wagman et al. 1982).** NIST has digitized them and publishes the CSV with a
checksum. Downloaded from the NIST Public Data Repository record `mds2-2124`; the bytes
match the published `.sha256`. Searched the whole file.

The aqueous phosphorus oxyanion block runs, verbatim, rows 1286 to 1288:

```
PO4<sup>3-</sup>,,,,ao,94.9714,-,-1277.4,-1018.7,-,-220.3*,-,
P2O7<sup>4-</sup>,,,,ao,173.9434,-,-2271.1,-1919.0,-,-117.,-,
P4O6,,,,cr,219.8916,-,-1640.1,-,-,-,-,
```

Diphosphate is there with a Gibbs energy. Triphosphate is not: the block jumps straight
from the diphosphate anion to P4O6. Every one of the eleven rows in the file whose formula
contains `P3` is a solid (`Zn5(P3O10)2`, `Li5P3O10`, `Na5P3O10` in two crystal forms and as
the hexahydrate) or the single solution entry `Na5P3O10, in 5200 H2O`, which carries an
enthalpy of -4468.9 kJ/mol and no Gibbs energy and no entropy. **The NBS tables evaluate no
aqueous triphosphate ion in any protonation state.**

**CODATA Key Values (Cox, Wagman and Medvedev 1989).** Already mirrored in this repo by
`fetch_ion_formation_energies.py`. A case insensitive search of the extracted text for
`P3O10`, `triphosph` and `tripoly` returns nothing. The only aqueous phosphorus species
CODATA evaluates are `HPO4-2` and `H2PO4-`. Negative.

So the ion table route the brief pointed at cannot be used. Two things block it, and both
would need fixing: there is no tabulated value to put in the table, and
`IonFormationResolver` in `torchcell/metabolism/thermo_pipeline.py` keys on a formula that
is a single element symbol of at most two characters, so a polyatomic anion would not match
it even if a value existed.

## Search for a measured reaction energy

**NIST TECRDB.** The `srd.nist.gov` reprint of Part 3 (Hydrolases) returned HTTP 503 twice,
so the database was queried directly through eQuilibrator's Zenodo export
(`doi:10.5281/zenodo.3978440/TECRDB.csv`, 4544 rows). Exactly one row mentions triphosphate,
line 3245, verbatim:

```
http://xpdb.nist.gov/enzyme_thermodynamics/enzyme_data1.pl?col=1.&T1=53MEY/SHA_1069,53MEY/SHA,calorimetry,B,3.6.1.2,trimetaphosphatase,kegg:C02466 + kegg:C00001 = kegg:C00536,trimetaphosphate(aq) + H2O(l) = triphosphate(aq),,,306.15,,6.95,
```

Three reasons it does not help. The `K` and `K_prime` fields are both empty, so no
equilibrium constant is carried. The reaction is trimetaphosphate hydrolysis, not `r_4403`,
and trimetaphosphate has no formation energy either, so it anchors nothing. And a
calorimetric enthalpy is not a Gibbs energy. There is no TECRDB row for EC 3.6.1.25
(triphosphate + H2O = diphosphate + phosphate) at all. The 226 row formation energy table
and the 13 row redox table have zero triphosphate rows. This is also the direct explanation
of why compound 314 is missing from `train_S`.

The primary paper behind `53MEY/SHA` is Meyerhof, Shatas and Kaplan, *Biochim. Biophys.
Acta* **12**, 121 (1953), "Heat of hydrolysis of trimetaphosphate". The PubMed record was
fetched and mirrored to close the trail; it carries no abstract and the full text was not
retrieved. A web search summary attributed a free energy to that paper, but the primary text
was never seen here, so no number from it is recorded. It is the wrong reaction in any case.

Searches for a measured hydrolysis free energy under other framings (Van Wazer and the
condensed phosphate literature, EC 3.6.1.25 apparent equilibrium constants, geochemical
databases, Bard, Parsons and Jordan, Alberty's species tables) returned kinetics,
activation energies and generic ATP bioenergetics, never a measured thermodynamic value for
this species. Every query is listed in the JSON record.

## The one thing that was found: a measured pKa ladder

Maki, H., Tsujito, M., Sakurai, M., Yamada, T., Nariai, H. and Mizuhata, M. (2013),
"Stabilities of the Divalent Metal Ion Complexes of a Short-Chain Polyphosphate Anion and
Its Imino Derivative", *Journal of Solution Chemistry* **42**, 2104,
doi 10.1007/s10953-013-0099-2, PMC3843374. Open access, mirrored as Europe PMC full text
XML, sha256 `129ce1cf08f7775a1f45a72205275145595f26e9d23ebf1095d217d2f5f64269`.

Table 1, verbatim after tag and MathML stripping: "Logarithmic stepwise protonation
constants of P3O(10-n)(NH)n5- ( n = 0, 2) anions determined by potentiometric titrations at
I = 0.1 mol.L-1 (NaNO3)", with "Numbers in parentheses indicate standard deviations derived
from the nonlinear least-squares approximation". Methods, verbatim: "40 cm3 solutions of
0.002 mol.L-1 Na5P3O10 or Na5P3O8(NH)2 + 0.10 mol.L-1 NaNO3 were titrated stepwise by a
solution of 0.01 mol.L-1 HNO3 + 0.10 mol.L-1 NaNO3", carried out "at 5.0, 15.0, 25.0, 30.0,
or 35.0 (+/-0.5) degrees C under a N2 atmosphere".

The P3O10(5-) columns, which are the first two of five (the other three belong to the imino
derivative and are not triphosphate):

| T | log10 K1, gives pKa of HP3O10(4-) | log10 K2, gives pKa of H2P3O10(3-) |
|---|---|---|
| 5 C | 7.33 (0.01) | 5.12 (0.02) |
| 15 C | 7.55 (0.01) | 5.22 (0.01) |
| 25 C | 7.69 (0.01) | 5.34 (0.01) |
| **30 C** | **7.77 (0.02)** | **5.40 (0.02)** |
| 35 C | 7.89 (0.02) | 5.49 (0.01) |

30 C is 303.15 K, the pipeline's own temperature, so no temperature extrapolation is needed.

These are conditional constants at I = 0.1 mol/L, and eQuilibrator's transform applies its
own Debye-Huckel term, so feeding them in raw would double count the ionic strength. They
are extrapolated to I = 0 here with eQuilibrator's own coefficients:

pKa_obs(I) = pKa(I=0) + dh(I) * (2 z_acid - 2) / (RT ln10)

with dh(0.1, 303.15 K) = 0.62773 kJ/mol. That gives pKa(I=0) = **8.852** for HP3O10(4-) and
**6.266** for H2P3O10(3-). This is arithmetic on the measured numbers, not a second
measurement, and it is labeled as derived in the JSON. As a sanity check only, the commonly
quoted textbook pair for triphosphoric acid at zero ionic strength is close to 8.8 and 6.

Three lower rungs were not measured in this source. Their populations at pH 6.8 to 7.2 are
below 1e-3, so they change nothing at the pipeline's conditions. Irani et al., *J. Inorg.
Nucl. Chem.* 1966, "Metal complexing by phosphorus compounds IX", is the obvious source for
them; it is paywalled and was not retrieved.

## What the measured ladder shows about the cached one

Computed with `equilibrator_cache.thermodynamic_constants._legendre_transform` called
directly, at pH 7.0, I = 0.25 mol/L, T = 303.15 K, pMg 3. Artifact
`transform_comparison.json`, sha256
`00f360fc9d06d77fbc7241c5b4692e3dc0c632283ef6f7425574d9cef57c9620`.

The cache stores six microspecies, from charge -6 to charge -1. That is one rung too far in
both directions: P3O10(5-) is fully deprotonated, so a charge -6 species would mean stripping
a proton from an oxide, and the neutral acid H5P3O10 at the other end is missing. The
implied pKas are 0.890, 2.450, 3.240, 7.120 and a fifth at 7.730 that belongs to the species
that does not exist.

| microspecies | population under the cache ladder | population under the measured ladder |
|---|---|---|
| charge -6 (does not exist) | 0.9027 | not present |
| charge -5, P3O10(5-) | 0.0928 | 0.2726 |
| charge -4, HP3O10(4-) | 0.0045 | 0.7180 |
| charge -3, H2P3O10(3-) | 0.0000 | 0.0095 |

Relative transform, cache ladder 14.552 kJ/mol, measured ladder 27.314 kJ/mol,
**difference 12.762 kJ/mol**. That is what the ladder error would inject into `r_4403` the
day an anchor exists, so the ladder is worth fixing before that, not after. It prices
nothing on its own.

## The computed value, recorded and not admitted

Sanden, S. A., Butch, C. J., Bartlett, S., Virgo, N., Sekine, Y. and McGlynn, S. E. (2024),
*iScience* **27**, 111088, PMC11530844. Mirrored as Europe PMC full text XML, sha256
`2e227d7c03476936c24097f5c66852f84efe011bb9c38ed9ca6ad2563454f49c`. Verbatim:

> Our calculations also include inorganic triphosphate, which is expected to yield
> comparable values to ATP, but with delta G'pH7 = -56.3 kJ mol-1 the highly charged
> HP3O10-4 species is likely subject to errors that are not accounted for by implicit and
> explicit solvation by SMD.

Method: DFT, PBE0/6-311G(2d,2p) in Gaussian 16 Rev. C.01, SMD continuum with four explicit
waters. Conditions: 1 M reactants and products, 55.55 mol water, pH 7, 300 K; no ionic
strength and no magnesium stated. Computed, so recorded and not admitted.

The same paragraph supplies its own calibration, which is the strongest reason to keep it
out rather than the authors' hedge alone:

> Similarly, P2O7-4 with delta G'pH7 = -36.7 kJ mol-1 exceeds the expected -13.9 kJ mol-1
> based on the values given by Alberty et al.

That is a 22.8 kJ/mol overshoot on the neighboring tetraanion by the same protocol. Any use
of -56.3 should carry an uncertainty at least that large and a route that is not the default
one.

A correction to this article exists (*iScience* 2025 Jun 5;28(6):112790). It changes a
reference label in Figure 1 and nothing else; the number stands.

## What remains for a human

1. **To price it, find one absolute anchor.** Route A, the NBS solid: NBS gives
   Na5P3O10 (cr, form I) delta_f G = -4093.9 kJ/mol and, for the same salt in 5200 H2O, an
   enthalpy but no entropy. A measured solubility product with a Pitzer treatment would
   close the cycle to P3O10(5-) through delta_f G(Na+, aq). That is a literature search plus
   a derivation, and the activity treatment at high ionic strength is where it can go wrong.
   Route B, an equilibrium measurement: none appears to exist, plausibly because the
   reaction is effectively irreversible, so this means isotope exchange or a coupled
   equilibrium rather than a literature find. Route C, accept the computed value with an
   explicit non-default route label and an uncertainty of at least 23 kJ/mol.
2. **Decide whether the pipeline may override a cached pKa ladder.** If yes, the charge -6
   microspecies should be dropped for every fully deprotonated inorganic anion, not just
   this one; the same predictor artifact will recur.
3. **Optionally complete the ladder** from Irani et al. 1966. It changes nothing at pH 7.

## Files

- `experiments/026-metabolism-flux/results/curation/triphosphate.json`
- `experiments/026-metabolism-flux/results/curation/triphosphate.md`
- `$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/triphosphate/` (10 artifacts,
  each with its sha256 in the JSON record)
