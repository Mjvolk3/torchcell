# ethylnitronate (s_3934) search log

Curated 2026-09-11. Metabolite s_3934 "Ethylnitronate" [c], GEM formula C2H5NO2, charge 0.
Blocked reaction r_4273 (EC 1.13.12.16, gene YJR149W). Recorded exclusion reason:
`group_decomposition_failed`.

Artifacts, each with its sha256, live in
`$DATA_ROOT/data/torchcell/thermo_equilibrator/curated/ethylnitronate/`. Full absolute
paths are at the end of this file.

## 1. Which species balances r_4273

Ran cobrapy `check_mass_balance` on r_4273 in yeast-GEM 9.0.2.

```
r_4273: s_0717 + s_1275 + s_3934 <=> s_0359 + s_0714 + 2.0 s_0794 + s_0803 + s_3935
check_mass_balance() -> {}
```

Balanced as written, with s_3934 = C2H5NO2 at charge 0. By hand: LHS C19 H26 N5 O13 P1,
charge -2; RHS the same. That is **nitroethane**, not the ethylnitronate anion.

Substituting the anion (C2H4NO2, charge -1) leaves the equation one hydrogen and one
negative charge short. It re-balances only if the 2 H+ on the product side become 1 H+,
which is exactly what RHEA:28767 writes for this EC number:

```
RHEA:28767  ethylnitronate + O2 = chemical entity + acetaldehyde + nitrite + H(+)
```

So the GEM's formula and charge are self-consistent and only the name is wrong. The two
GEM accessions also disagree with each other:

| accession | database entry | formula | charge |
|---|---|---|---|
| `chebi=CHEBI:16268` | nitroethane | C2H5NO2 | 0 |
| `kegg.compound=C18091` | Ethylnitronate | C2H4NO2 | -1 |

KEGG itself keeps them apart: C01837 Nitroethane (EC 1.7.3.1, nitroalkane oxidase) versus
C18091 Ethylnitronate (EC 1.13.12.16, nitronate monooxygenase). The ExplorEnz comment for
EC 1.13.12.16 says the enzyme acts on nitronates and that the Neurospora crassa enzyme
"can also utilize neutral nitroalkanes, but with lower activity", so a yeast NMO homolog
acting on nitroethane is defensible. Renaming s_3934 or re-specifying it as the anion is a
GEM curation decision for a human.

## 2. Group decomposition, tried on six spellings plus controls

Called `equilibrator_assets.group_decompose.GroupDecomposer` directly, so the cache lookup
does not hide the result. A structure decomposes when it leaves zero unassigned atoms.

| SMILES | what it is | decomposes | groups matched |
|---|---|---|---|
| `CCN(=O)=O` | nitroethane, pentavalent | yes | -NO2, -C-[H2], -C[H3], Origin |
| `CC[N+](=O)[O-]` | nitroethane, charge separated (ChEBI:16268) | no | atom 3 unassigned |
| `CC=N(O)=O` | aci form, **what the eQuilibrator cache stores** | no | atom 4 unassigned |
| `C/C=[N+]([O-])[O-]` | ethylnitronate anion | yes | -N<, 2x [N]-O[Z-1], =C-[H1], -C[H3], Origin |
| `CC=N(=O)[O-]` | anion, pentavalent spelling | no | atom 3 unassigned |
| `CC=[N+]([O-])O` | aci-nitroethane acid | yes | -N<, [N]-O[H1], [N]-O[Z-1], =C-[H1], -C[H3], Origin |
| `CN(=O)=O` | nitromethane control | yes | -NO2, -C[H3], Origin |
| `C[N+](=O)[O-]` | nitromethane control | no | atom 2 unassigned |
| `c1ccccc1N(=O)=O` | nitrobenzene control | yes | -NO2, ring carbons, Origin |
| `[O-][N+](=O)c1ccccc1` | nitrobenzene control | no | atom 2 unassigned |
| `CC=O` | acetaldehyde positive control | yes | -C=O[H1], -C[H3], Origin |

The cause is a SMARTS mismatch, not chemistry. `~/.cache/equilibrator/group_definitions.csv`
line 46 defines the nitro group as

```
"-NO2",0,0,0,"[C,c,N,n]N(=O)=O","1|2|3","Nitro",
```

and OpenBabel's matcher does not treat `[N+](=O)[O-]` as the same thing. Hence the
representation rule: hand the decomposer the pentavalent `N(=O)=O` spelling.

RDKit confirms the identity of each SMILES. `CCN(=O)=O` and `CC[N+](=O)[O-]` both give
C2H5NO2, charge 0, InChIKey `MCSAJNNLRCFZED-UHFFFAOYSA-N`; `C/C=[N+]([O-])[O-]` gives
C2H4NO2-, charge -1, `YERBBVNYIKLXDM-UHFFFAOYSA-N`; `CC=[N+]([O-])O` gives C2H5NO2,
charge 0, `CPZLOQHKKRZRSD-UHFFFAOYSA-N`.

## 3. Why the cache holds an empty group vector

`kegg:C18091`, `chebi:CHEBI:16268` and `metanetx.chemical:MNXM2143` all resolve to the same
eQuilibrator compound, id 1907, whose synonym list merges nitroethane, aci-nitroethane and
ethylnitronate(1-). It stores SMILES `CC=N(O)=O`, which does not decompose, so its
`group_vector` is a list of length 0 and `standard_dg_formation` returns None.

That empty list is a trap. Code that tests `group_vector is not None` as a proxy for
decomposability, which `thermo_structure_attempt.py` does, passes on this compound and only
discovers the failure one step later as a missing formation energy. A test for a non-empty
vector would report it honestly.

## 4. The fix decomposes but still cannot be priced

Patching compound 1907's group vector in memory with the decomposition of `CCN(=O)=O` and
asking component contribution, at pH 7.2, I = 0.25 M, T = 303.15 K, pMg 3.0:

```
dfG0        = -31.17 kJ/mol
sigma_fin   =   3.97 kJ/mol
dfG'0       = 181.71 kJ/mol
r_4273 dG'0 = -394.78 kJ/mol,  sigma_fin 6.59 kJ/mol   (proton term 0)
```

**Do not use those numbers.** The -NO2 group is in the component-contribution null space.
Two independent checks:

- `||L_inf e_g|| = 1.0` for `-NO2 [H0 Z0 Mg0]`, and also for `[N]-O [H1 Z0]` and
  `[N]-O [H0 Z-1]`, the groups the two anionic spellings route through. The same quantity
  is 0.0 for `-C-`, `-C` and `Origin`. Zeroing the -NO2 entry of the compound vector drops
  the projection to 1.5e-15, so that group carries all of it. 85 of 163 groups are
  unconstrained, so this is a general gap rather than a nitro-specific quirk.
- Column 67 of `train_G`, which is `-NO2`, is used by 0 of the 631 training compounds.
  For scale, `Origin` is used by 581, `-C-` by 332, `-C` by 196.

`RMSE_inf` is 100000 kJ/mol, so the estimate is undefined rather than uncertain.
`build_equilibrator_thermo.py` already handles this: its `estimable` flag is
`||S^T sigma_inf|| < 1e-10`, which is 1.0 here, so applying the structure fix leaves r_4273
with `estimable = False`. The fix changes the diagnosis from `group_decomposition_failed` to
"group present but unparameterized", and buys no price.

## 5. pKa, measured and tabulated

Not the blocker, recorded because it is real. The IUPAC compilation served by PubChem gives
four aqueous values for the nitroethane / ethylnitronate couple, all by glass-electrode
titration:

| pKa | T (C) | source | remarks |
|---|---|---|---|
| 8.6 | 25 | Wheland and Farr, JACS 65, 1433 (1943) | |
| 8.49 | 18 | Turnbull and Maron, JACS 65, 212 (1943) | C = 0.003 to 0.007, mixed constant |
| 8.46 | 25 | Turnbull and Maron | pH equilibrium requires ca 5 min |
| 8.44 | 30 | Turnbull and Maron | pH equilibrium requires ca 5 min |

PubChem's "Dissociation Constants" field separately states "pKa = 8.6", cited to Serjeant
and Dempsey, IUPAC Chemical Data Series No. 23, 1979, p. 33. No ionic strength is stated
anywhere. At the pipeline's 303.15 K the tabulated value is 8.44.

The eQuilibrator cache already carries a ladder for this compound, `ddg_over_rt` =
20.838395091596116 for the charge -1 microspecies, which is a predicted pKa of 9.05.
Substituting the measured value moves dfG'0 at pH 7.2 from 181.71 to 181.55 kJ/mol, a
0.16 kJ/mol change against a group sigma of 3.97 kJ/mol. Recorded for a later external-pKa
pass rather than as its own row.

## 6. Documented negatives

- **No aqueous formation data anywhere.** NIST Chemistry WebBook C79243 lists only
  liquid-phase dfH (-144.0 +/- 0.4 kJ/mol, Lebedeva and Ryadenko 1973; -143.6 +/- 1.1,
  Cass et al. 1958; -140.1 +/- 1.3, Holcomb and Dorsey 1949) and combustion enthalpies. No
  dfG at all, no aqueous phase, and the Henry's law table is empty. Turning a liquid-phase
  enthalpy into an aqueous Gibbs energy needs a gas-phase entropy and a solvation model,
  which would be a computed value; that route was not taken.
- **No TECRDB entry.** No apparent equilibrium constant for a nitronate or nitroalkane
  reaction surfaced. The same statement from inside the tool: nitroethane is not among
  eQuilibrator's 631 reactant-contribution training compounds, which is why nitrite, whose
  group vector is also empty, still gets a value while this one does not.
- **Nothing in Alberty.** No nitroethane or nitronate in the tabulated reactant sets.
- **No published discussion** of nitro-group coverage in component contribution was found;
  the L_inf computation above answers it directly instead.

## 7. Verdict

kind `structure`, admission `tabulated`. The curated structure is `CCN(=O)=O`
(InChIKey MCSAJNNLRCFZED-UHFFFAOYSA-N, C2H5NO2, charge 0) with the reusable rule: write
nitro groups pentavalent for the eQuilibrator decomposer. `sigma_kj_per_mol` is null,
because no admissible energy came out of this. The reaction stays unpriced, and the honest
label for the exclusion is "unparameterized group", not "decomposition failed".

Remaining for a human: decide whether to rename s_3934 or re-specify it as the anion with
1 H+; decide whether r_4273 simply stays unpriced or whether a nitro-group contribution is
worth fitting from external data, which is a research task given that no aqueous nitroalkane
data was found; harden the empty-group-vector test; and apply the same representation rule
to s_3859 (4-nitrophenyl phosphate) and s_3860 (4-nitrophenol), expecting the same
unparameterized-group verdict.

## Artifacts

All under
`/scratch/projects/torchcell-scratch/data/torchcell/thermo_equilibrator/curated/ethylnitronate/`

```
206462700e12386347e3082fda8a3500747a7994437934b96088133fc30c867a  chebi_16268.json
a9bc46e4f0ee248e03661be33104b646b0b821c4e014b4386c05446494f795d0  chebi_55327.json
18b139e77c18139333269a9716a92b3c2dc2877afa48b7574f383cc3cd085e36  chebi_77894.json
0bdeb725efb623ed91f19ff02a360b18847a25886413b76622668eb87691485b  en_decomp3.json
c3f47eccfb2e21e794057198cf49fd484bcbdd1d152dbb3870408e8d8ada5280  en_decomp3.py
522232ed0bfac1c1400629fa291cfa8121b7bf9f4200f10f95a380623f31aba9  en_decomp.py
5f0d77c0b5728c1b15d6a5cfbf8a5a80f5bb3a1c4446c4495de272d14b2eb6cd  en_final_inf.py
c025dbbcba70609f01c0c2fe41b616b39a22fdacee5945f30ccdf363ce077417  en_price2.py
31fb4ff0d92f04073288f59f3775842fd6e79b47f7e106b5cf2446bc6c569cf0  explorenz_ec_1_13_12_16.html
5658bee14b0cbd0c86d9b5f734800699f112e0ca982db3520f542668a4678c6f  kegg_C01837.txt
c49a099446830a27bda483d0698966f70b6c166a738d3f84b0f92dfe1ac41224  kegg_C18091.txt
d312708423595c043d755602bab3160aa3b0cff504ad71ff338d26e777313e70  kegg_R00025.txt
b7abe65c7b561c29bc6fdd5860d1ceec94f1048248416f5fee8fa0f30a9ce927  nist_webbook_nitroethane.html
cfef9ed8336759085edb99b010c59943a4b9f8591f194df36bae67404e33882d  nist_webbook_nitroethane_thermo.html
bf6e2675fb1edf6fa326a93525f5279497c19e14888b982248422f4b1b83f22f  pubchem_cid6587_nitroethane.json
a936e749e9e25c6b2ba249fa064ca02372b1a84d8501e27f89d2d83c8a22b7ae  pubchem_iupac_pka_6587.json
ee79850caf5333bf909ce819e39c3b46b31627d6e0b648a5ddd83953fcab98f6  rhea_28767.tsv
bc93cfbeda720f83891067e2771767c899ea5f3fecef45f64f867aba126aad5d  rhea_ec_1_13_12_16.tsv
```

The four `en_*.py` files are the probe scripts that produced sections 2, 3 and 4, copied in
so the numbers are re-runnable; they take the compound cache path as an argument and run
under `~/miniconda3/envs/equilibrator-ext/bin/python`. `en_decomp3.json` is the raw
decomposition table.
