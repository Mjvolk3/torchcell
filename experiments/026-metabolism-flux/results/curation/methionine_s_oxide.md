# methionine_s_oxide search log

Curated 2026-09-11. Species s_3837 L-Methionine S-oxide [c], s_4176 L-Methionine S-oxide [e],
s_3807 L-methionine (S)-S-oxide [c]. GEM formula C5H11NO3S, charge 0. Blocked reactions
r_4186 and r_4202 (methionine sulfoxide reductases with thioredoxin). r_4478 (diffusion) and
r_4550 (exchange) close by separate code rules and were not investigated.

Result: `kind = not_found`, `admission = none`. Both the structure hypothesis and the
literature route are documented negatives.

Fetched artifacts, sha256 in `SHA256SUMS.txt`:
`/scratch/projects/torchcell-scratch/data/torchcell/thermo_equilibrator/curated/methionine_s_oxide/`

## Part 1, the structure hypothesis, disproved

The hypothesis was that the decomposition fails only because of how the sulfoxide is
written. It does not.

Eleven representations were passed to `equilibrator_assets.group_decompose.GroupDecomposer`
directly, with `raise_exception=True`, so the failure is read from the decomposer rather
than inferred from a null group vector:

| form | SMILES | decomposes |
|---|---|---|
| neutral amino acid, S=O | `CS(=O)CC[C@H](N)C(=O)O` | no |
| neutral amino acid, charge separated | `C[S+]([O-])CC[C@H](N)C(=O)O` | no |
| zwitterion, S=O | `CS(=O)CC[C@H]([NH3+])C(=O)[O-]` | no |
| zwitterion, charge separated | `C[S+]([O-])CC[C@H]([NH3+])C(=O)[O-]` | no |
| cache form of compound 1993 | `CS(=O)CC[C@H]([NH3+])C([O-])=O` | no |
| (S)-sulfoxide, cache 691225 | `C[S@](=O)CC[C@H]([NH3+])C([O-])=O` | no |
| (R)-sulfoxide, cache 1992 | `C[S@@](=O)CC[C@H]([NH3+])C([O-])=O` | no |
| (R)-sulfoxide, neutral | `C[S@@](=O)CC[C@H](N)C(=O)O` | no |
| (R)-sulfoxide, charge separated | `C[S@@+]([O-])CC[C@H](N)C(=O)O` | no |
| (S)-sulfoxide, neutral | `C[S@](=O)CC[C@H](N)C(=O)O` | no |
| no stereochemistry | `CS(=O)CCC(N)C(=O)O` | no |
| CONTROL dimethyl sulfoxide | `CS(C)=O` | no |
| CONTROL L-methionine | `CSCC[C@H]([NH3+])C([O-])=O` | yes |
| CONTROL dimethyl sulfide | `CSC` | yes |
| CONTROL methionine sulfone | `CS(=O)(=O)CC[C@H]([NH3+])C([O-])=O` | yes |

Every failure is the same failure. Exactly one heavy atom is left unassigned and it is
always the sulfoxide oxygen: unassigned index 2, atomic number 8, for the methionine
sulfoxides, and index 3 for dimethyl sulfoxide. The assigned groups in every case are
`['-S<', '-N', '-COO', '-C<', '-C-' x2, '-C']`, so the sulfur IS matched, by the sulfonium
group whose SMARTS is `[S;H0;X3]`. That SMARTS matches a three-coordinate sulfur in either
representation, which is why S=O and `[S+]-[O-]` behave identically. RDKit and OpenBabel
also canonicalize the two to one molecule: all eleven forms collapse to InChIKey
`QEFRNWWLZKMPFJ-*` and to three cache compounds already present.

The root cause is in `~/.cache/equilibrator/group_definitions.csv`, 140 rows. It defines
sulfate `-SO3`, sulfonyl `-SO2-`, sulfonium `-S<`, thioether `-S-`, thiol `-S`, disulfide
and thioester. It defines no sulfoxide group, so a sulfoxide oxygen matches nothing.

Reusable rule for every sulfoxide row in this campaign: no SMILES choice will fix it. The
sulfone decomposes and the thioether decomposes; only the sulfoxide is missing from the
group table. Do not spend another curation pass writing sulfoxide structures.

A trap worth fixing in the integrating pass: the cache compounds for these species carry a
`group_vector` that is not `None` but is the empty list. A test of the form
`compound.group_vector is not None` reads that as a successful decomposition.
`thermo_structure_attempt.py` uses exactly that test.

Reactant contribution is empty too. Compound ids 1993, 1992 and 691225 are absent from the
631-compound training set in `cc_params.npz` (`train_S_index`), so neither half of component
contribution can price the species. `cc.standard_dg_prime` on MetSO + H2 = Met + H2O returns
the infinite-uncertainty flag, 0.0 +/- 1.0e5 kJ/mol.

Identity was fixed against ChEBI, because the GEM is internally inconsistent. s_3837 and
s_4176 carry `bigg.metabolite=metsox_S__L`, a name asserting the (S) diastereomer, but their
ChEBI, KEGG and MetaNetX annotations all resolve to the diastereomer-unspecified species,
CHEBI:17016, InChIKey `QEFRNWWLZKMPFJ-YGVKFDHGSA-N`. s_3807 is the genuine (S)-sulfoxide,
CHEBI:49031, InChIKey `QEFRNWWLZKMPFJ-MFXDVPHUSA-N`, defined by ChEBI as "The (S)-oxido
diastereomer of L-methionine S-oxide." RDKit confirms C5H11NO3S and formal charge 0 for
every candidate.

## Part 2, literature

Seventeen searches are recorded in the JSON with query, engine, date and outcome. The
substantive ones:

**No measured E°' for the Met-S-oxide/Met couple.** The literature states plainly that the
reduction potential of methionine has never been measured and that small compounds such as
dimethyl sulfide are used in its place. No potentiometric or redox-buffer titration of this
couple was found.

**No TECRDB entry.** `tecrdb.nist.gov` does not resolve and the original `xpdb.nist.gov`
host is dead. The machine-readable successor, openTECR, was downloaded in full as ODS
(3.5 MB, sha256 `102b53fc...`) from the link published at `https://w3id.org/opentecr/tecrdb`.
Searching its `content.xml`: `methionine sulfoxide` 0 hits, `methionine sulphoxide` 0,
`methionine-S-oxide` 0, EC `1.8.4.11` 0, `1.8.4.12` 0, `1.8.4.13` 0, `1.8.4.5` 0. The only
EC 1.8.4 entry in the whole database is 1.8.4.3, glutathione-CoA-glutathione
transhydrogenase. All 227 hits for the substring `sulfoxide` are dimethyl sulfoxide used as
a cosolvent, for example "c(MgCl2)=0.0009 M; solute=dimethyl sulfoxide (50% v/v)".

**No NIST thermochemistry.** The Chemistry WebBook returns "Name Not Found" for
`methionine sulfoxide` and "Registry Number Not Found" for CAS 3226-65-3, 454-41-1 and
7101-76-1. The L-methionine control returns "Search Results", so the negative is real and
not a broken query. Combustion and formation enthalpies exist for methionine but not for
the sulfoxide.

**The computed paper stays out.** JCIM 10.1021/acs.jcim.4c01617 (January 2025) computes the
Met plus H2O2 oxidation free energy profile by QM/MM with classical MD. Computed, in a
protein context, and not the aqueous formation energy of free methionine sulfoxide.

**The thioredoxin half of the couple IS measured**, and is citable if the sulfoxide half is
ever obtained: Aslund, Berndt and Holmgren, J Biol Chem 1997, PMID 9388218, E°' about
-270 mV at pH 7 by direct protein-protein redox equilibria, with the thioredoxin family
quoted at -270 to -300 mV and a later E. coli value of -284.2 +/- 1.0 mV.

## Part 3, the one analog, and why it must not be borrowed from eQuilibrator

The nearest tabulated value is for dimethyl sulfoxide, not methionine sulfoxide. Kim, Weiss
and Levine, Biochim Biophys Acta 2014;1840, doi 10.1016/j.bbagen.2013.04.038, PMC3766491,
introduction: "Indeed, the standard redox potential for the two electron reduction of
dimethyl sulfoxide is +160 mV [17] while that for cystine is +220 mV [18]." Reference [17]
in that article is Wood PM, FEBS Lett 1981;124:11-14, doi 10.1016/0014-5793(81)80042-7. The
primary paper is paywalled: HTTP 403 at ScienceDirect and Wiley, no abstract in the PubMed
record, and the abstract elided by the publisher on Semantic Scholar.

eQuilibrator already carries that number, and carries it wrong. Its single DMSO training
entry, column 3997 of `cc_params.npz`, is the half reaction "+1 Dimethyl sulfide, -1 Dimethyl
sulfoxide" with b = -30.8752 kJ/mol. That is exactly -2 F (0.160 V) to five figures, so Wood
1981 is provably the source. The entry omits the water product, so the fitted formation
energy of DMSO is high by the transformed formation energy of water.

Measured in this pass at pH 7.0, I = 0.25 M, 298.15 K, p_Mg 3.0:

| reaction | eQuilibrator dG'0, kJ/mol | implied E°' | measured E°' |
|---|---|---|---|
| GSSG + NADH = 2 GSH + NAD+ | -15.56 +/- 0.84 | -239.4 mV | about -240 mV |
| fumarate + NADH = succinate + NAD+ | -61.22 +/- 2.07 | -2.8 mV | +31 mV |
| DMSO + NADH = DMS + NAD+ + H2O | -251.05 +/- 3.02 | **+981 mV** | **+160 mV** |

The controls behave and the DMSO row does not. The 158 kJ/mol gap matches
dGf'(H2O) = -156.6 kJ/mol under the same conditions, which is the missing term. An E°' of
+981 mV would make DMSO a stronger oxidant than oxygen, which is not chemistry.

So: do not borrow eQuilibrator's DMSO formation energy as a sulfoxide surrogate, here or in
any other sulfoxide row. This looks like an upstream defect worth reporting.

## What remains for a human

1. Read Wood 1981 FEBS Lett 124:11-14 in full through library access. Four pages, an
   evaluation rather than a new measurement, and it may state the underlying aqueous
   formation energies explicitly. Those would be tabulated values capable of anchoring a
   sulfoxide-to-sulfide increment. Highest-value follow-up and not reachable by an agent.
2. Decide whether a Wood-derived increment may enter as `provisional`. It is not admissible
   under the current rule, since +160 mV is dimethyl sulfoxide, and this record leaves the
   row open rather than laundering an analog into a value. If it is admitted later, it must
   be built from Wood's own numbers, not from eQuilibrator's DMSO entry.
3. Consider measuring it. MsrA runs in both directions (Lim, You, Kim and Levine, PNAS
   2011, doi 10.1073/pnas.1101275108), so an equilibrium constant against a calibrated
   thiol-disulfide redox buffer is experimentally reachable and would be the first measured
   value for this couple.
4. Note that r_4186 blocks on the (S) diastereomer and r_4202 on the generic species, so
   they are not duplicates, though their standard free energies will be nearly equal since
   the diastereomers differ only in configuration at sulfur.
