# porphyrin (s_4022 [m], s_4023 [c]) -- curation search log

Date: 2026-09-11. Machine: GilaHyper. Worktree:
`/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets`.

**Verdict:** `kind = not_found`, `admission = none`, `value = null`. No measured or
tabulated aqueous standard transformed Gibbs formation energy exists for porphine, and
eQuilibrator's group decomposer refuses every valid structure of it. r_4337 closes by
cancellation regardless, since s_4022 and s_4023 are the same molecule on opposite sides.

## What the species is

yeast-GEM 9.0.2 calls both s_4022 and s_4023 "porphyrin" with formula C20H14N4 and charge
0. That is porphine, the unsubstituted free-base macrocycle, not the porphyrin class. Its
identifiers agree across three namespaces:

| source | formula | charge | InChIKey | note |
|---|---|---|---|---|
| PubChem CID 66868 | C20H14N4 | 0 | JZRYQZJSTWVBBD-UHFFFAOYSA-N | title "Porphyrin", NH tautomer 21H,22H, no stereo layer |
| ChEBI:8337 | C20H14N4 | 0 | RKCAIXNGYQCCAL-CEVVSZFKSA-N | synonyms "porphine", "21H,23H-porphine"; CAS 101-60-0; KEGG C05113 |
| eQuilibrator cache compound 2399 | C20H14N4 | 0 | RKCAIXNGYQCCAL-CEVVSZFKSA-N | reached from chebi, kegg and metanetx accessions alike |

RDKit 2026.3.6 confirms C20H14N4 and formal charge 0 for the PubChem structure, the RDKit
aromatic canonicalization of it, the cache's own stored SMILES, and the ChEBI standard
InChI round trip.

## Step 1: the group decomposer

Run under `~/miniconda3/envs/equilibrator-ext/bin/python` with equilibrator-api 0.6.0,
equilibrator-cache 0.6.2, equilibrator-assets 0.6.0.post29+git.c79e371e,
component-contribution 0.6.0, openbabel-wheel 3.1.1.23. Cache:
`~/.cache/equilibrator/compounds.sqlite`,
sha256 `12a28e5a639097d39c7fb03bb740b7ddb204a3da8ee6d4bbac413e8b22e5cb28`.

Direct `GroupDecomposer.Decompose(..., raise_exception=True)`:

| structure | source | decomposes | result |
|---|---|---|---|
| `C1=CC2=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C1N2` | PubChem CID 66868, kekulized | no | `GroupDecompositionError: Unable to decompose <openbabel...OBMol ...> into groups` |
| `C1=Cc2cc3ccc(cc4ccc(cc5nc(cc1n2)C=C5)[nH]4)[nH]3` | RDKit aromatic form of the above | no | same error |
| `N1C2=CC=C1C=C1C=CC(C=C3C=CC(C=C4NC(C=C4)=C2)=N3)=N1` | the cache's own SMILES for compound 2399 | no | same error |
| `C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3` | ChEBI standard InChI round trip, 21H,23H tautomer | no | same error |
| `C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)n5)C=C4)n3` | ChEBI `default_structure` SMILES verbatim | **yes** | **wrong molecule, see below** |

The error is raised at `equilibrator_assets/group_decompose.py` line 607. Kekulized versus
aromatic makes no difference, and neither does the 21H,22H versus 21H,23H NH tautomer.

In lenient mode (`raise_exception=False`) the decomposition covers the molecule with 8
nonzero group types and leaves exactly 2 atoms unassigned. Both unassigned atoms are
aromatic carbons with three heavy neighbors and no hydrogen (OpenBabel indices 3 and 23 in
the PubChem spelling, 5 and 8 in the ChEBI tautomer). Hypothesis (untested): these are ring
fusion or meso carbons of the macrocycle for which the component-contribution group table
has no fitted group. I did not verify which ring positions they are.

**The trap.** The ChEBI `default_structure` SMILES string does decompose, with 4 nonzero
groups, and must not be used. RDKit reads it as **C20H12N4** (InChIKey
AJYZYPAMLMSWHG-UHFFFAOYSA-N), two hydrogens short of the GEM formula, because ChEBI writes
both pyrrole nitrogens as bare lowercase `n` rather than `[nH]`. The ChEBI standard InChI
for the same entry is correct at C20H14N4; only the SMILES serialization is short. The
heavy-atom composition guard in `thermo_structure_attempt.py` would not catch this, since
it drops hydrogens by design. Anyone revisiting this row should feed the InChI, not the
ChEBI SMILES.

## Step 2: the eQuilibrator compound cache

`chebi:CHEBI:8337`, `kegg:C05113` and `metanetx.chemical:MNXM2714` all resolve to cache
compound **2399**, formula C20H14N4, atom bag `{N: 4, C: 20, H: 14, e-: 162}`. Its group
vector is **present but empty**, zero nonzero entries, and
`ComponentContribution.standard_dg_formation` returns `None`.

This corrects a reading of the existing pipeline output:
`thermo_accession_sweep.json` records `has_group_vector: true` for this compound, which
looks like a successful decomposition. It is not one. The exclusion reason
`group_decomposition_failed` recorded in `thermo_exclusions.json` is correct, and the
fresh-creation route in `thermo_structure_attempt.py` cannot rescue the species, because
the decomposer refuses the very structure the cache already holds.

The public eQuilibrator search page could not be reached: both `curl` and `WebFetch`
failed TLS chain validation against `equilibrator.weizmann.ac.il` (curl exit 60, "unable
to get local issuer certificate"). No artifact stored for that attempt. The local compound
cache is the same eQuilibrator compound cache, so the query above is the authoritative
answer.

## Step 3: literature search for a tabulated formation energy

| query | engine | outcome |
|---|---|---|
| porphine "Gibbs energy of formation" aqueous kJ/mol thermodynamic | WebSearch | miss; generic group-contribution methodology and substituted metalloporphyrin complexation enthalpies |
| "21H,23H-porphine" NIST WebBook enthalpy of formation | WebSearch | pointer hit; located NIST species C101600 for the unsubstituted compound |
| porphine standard Gibbs free energy of formation aqueous solution thermodynamic properties CHNOSZ | WebSearch | miss; no CHNOSZ or other aqueous-species tabulation |
| equilibrator porphyrin C05113 formation energy group contribution missing | WebSearch | miss; only the general eQuilibrator statement that some structures fall outside the method |
| "porphine" OR "porphin" aqueous solubility standard state thermodynamic Gibbs energy hydration free energy | WebSearch | miss; only deliberately water-solubilized substituted porphyrins |
| Longo Finarelli Schmalzbach Adler 1970 "enthalpy of formation of porphin" J Phys Chem 74 3296 | WebSearch | citation confirmed; no aqueous companion value |

**The prior pass was too pessimistic on one point and right on the conclusion.** NIST does
carry the unsubstituted compound, species C101600, 21H,23H-porphine, CAS 101-60-0,
C20H14N4, not merely octaethylporphyrin and other substituted analogs:

- `dfH(solid) = -1106.9 +/- 3.6 kJ/mol` and `dcH(solid) = -8764.1 +/- 1.8 kJ/mol`, method
  Ccb, from Longo, F.R.; Finarelli, J.D.; Schmalzbach, E.; Alder, A.D., "The enthalpy of
  formation of porphin", J. Phys. Chem., 1970, 74, 3296-3297.
- `dsubH = 87 +/- 3 kJ/mol` over 424 to 507 K, method Fluoresc, from Stefanov, A.; Stibor,
  A.; Dominguez-Clarimon, A.; Arndt, M., "Sublimation enthalpy of dye molecules measured
  using fluorescence", J. Chem. Phys., 2004, 121, 6935.

**None of it is admitted**, for three independent reasons: it is an enthalpy rather than a
Gibbs energy; the condensed-phase value is the crystalline solid rather than an aqueous
species; and no entropy and no aqueous solvation free energy is tabulated, so there is no
route from these numbers to a `dGf'(aq)` without computation, and computed-only values
stay out of the default route. The species index page for C101600 lists exactly three data
classes, condensed phase thermochemistry, phase change, and gas phase ion energetics, and
no solution-phase or Gibbs-energy class. That enumeration is the direct evidence for the
negative.

Consistency check I ran on the NIST pair, so the two rows are known to agree rather than
assumed to: `dfH(solid) = 20*dfH(CO2,g) + 7*dfH(H2O,l) - dcH = -9871.0 + 8764.1 = -1106.9`
kJ/mol, combustion to CO2, liquid water and N2. I did not adjudicate whether the 1970
measurement is itself correct, and nothing in this record depends on it.

## Why the row does not block anything

r_4337 is named "Iron-sulfur clusters transporter ATM1, mitochondrial" in yeast-GEM 9.0.2
and carries 7 participants. Read from the model YAML,
`/scratch/projects/torchcell-scratch/data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.yml`,
its stoichiometry includes `s_4022: -1` and `s_4023: 1`. The same molecule appears on both
sides with equal and opposite coefficients, so its formation energy cancels and the
reaction prices from ATP hydrolysis alone under the code agent's cancellation rule.

One caveat for that rule, flagged rather than resolved: `thermo_pipeline.py` applies
per-compartment pH, ionic strength and pMg through
`CompartmentConditions.for_compartment`, so the cancellation is exact for the untransformed
formation energy. Any residual Legendre-transform difference between m and c is a transport
term rather than a formation term. I did not measure its size for this species.

## Artifacts

All stored under
`/scratch/projects/torchcell-scratch/data/torchcell/thermo_equilibrator/curated/porphyrin/`.

| file | sha256 | retrieved (UTC) |
|---|---|---|
| `pubchem_cid66868_properties.json` | `488b78c1f52d48f073a60242628a4a6a6afa87e3c89b87e36bf7574a8ce7e761` | 2026-09-11T21:52:59Z |
| `pubchem_cid66868_record.json` | `e9311dd4f79dbedfb05b1bbfaa7bfe670d23d1431ac23722c187e2c0cda105a5` | 2026-09-11T21:52:53Z |
| `chebi_8337_compound.json` | `4fcae199ee8ae679f8ae3d9727d5c23c95e3dced037bb6b20735fb7624f44f95` | 2026-09-11T21:53:58Z |
| `ols4_chebi_8337.json` | `c6d3c33ba8c12d9b5dc89c80d45b316098e92d59f75ae561848d7a6b00bdffe8` | 2026-09-11T21:54:03Z |
| `nist_webbook_C101600_index.html` | `06a8df3267d382200d2a5c3ed7d502037632fa09c2383846a84bafc5fec46892` | 2026-09-11T21:58:09Z |
| `nist_webbook_C101600_thermo.html` | `517b6a0c4ec7d310899442de4f50f9cbcb36bbd38bfbcd23a166673a7625af93` | 2026-09-11T21:58:04Z |
| `nist_webbook_C101600_condensed.html` | `48481d04cae8747519fab07cf081debc45bd9c7544e3b91730e4c16cb8962055` | 2026-09-11T21:58:30Z |
| `nist_webbook_C101600_phasechange.html` | `61c9474f63720116a95809923b689e68b5406b3998ead145128497f00b704863` | 2026-09-11T21:59:34Z |
| `porphyrin_probe.py` | `cd6becbd4ea626d8e610f0bb9ee408722b04b508ab12a6e4d7f4c9ed38cbfe60` | probe source |
| `porphyrin_probe.json` | `685f6f88d95e61049c6a3124478c44e1ea17e3ccf5c17808cc45b1f732b4f1a3` | 2026-09-11T22:00:59Z |
| `porphyrin_probe2.py` | `3b4cd102bfc4515e2ff3f1e285218765e4725a3485638b692ede7c5c410e1832` | probe source |
| `porphyrin_probe2.json` | `5b29894ecb2141f1db4a74592a2f84adbc0e707f099e9b6ba8a47b1e93f59fb0` | 2026-09-11T22:00:59Z |
| `porphyrin_probe3.py` | `f564035944a6819f0b542c43154490e2219717ed6bd2bfa6b89195d79072b8e6` | probe source |
| `porphyrin_probe3.json` | `d256ccb1bc02d967f6c24b8b40cac25573bf68816566a81ddfd7918673e0d99a` | 2026-09-11T22:00:59Z |
| `porphyrin_probe4.py` | `20ee7acd8a6919f6382acfb3a1623aa78003831dfe08b38f43040a4b8a2c6feb` | probe source |
| `porphyrin_probe4.json` | `2c6078685b18ed051f6b59adef3c9b1c52c7c257dbbf2c46e42fba90ed2e2714` | 2026-09-11T22:00:59Z |

Retrieval commands are recorded per artifact in `porphyrin.json`. The probe scripts are
read-only against the compound cache and touch nothing under `torchcell/` or
`experiments/*/scripts/`; `porphyrin_probe.py` and `porphyrin_probe2.py` take the cache
path as a second argument, `porphyrin_probe3.py` and `porphyrin_probe4.py` need only the
group decomposer.
