---
id: d82nnqotzmvvwb98oged2t4
title: Media
desc: ''
updated: 1789259608089
created: 1789259608089
---

## 2026.09.12 - The library as the join layer

Source: `torchcell/datamodels/media.py`
Tests: `tests/torchcell/datamodels/test_media_library.py`,
`tests/torchcell/metabolism/test_media.py`,
`tests/torchcell/datamodels/test_ontology_coherence.py`

The shared media library stopped being a list of recipes and became the layer the
cross-dataset environment join actually keys on. Three changes did that, and each one
is now locked by a test rather than by convention.

### 1. Every single substance goes through the shared identity table

`_defined(...)` builds a component through
`torchcell.datamodels.compound_identity.resolved_compound`, so the compound carries
the table's canonical name plus an InChIKey / ChEBI / PubChem CID. Measured over the
51 library media and their 673 component slots: **55 distinct substances identified**,
**1 gapped** (`cellobiose`, a Vanacloig dropout the pinned table has not resolved
yet, carrying a typed `ProvenanceGap` on `inchikey`), and **9 distinct undefined
preparations**.

The 9 are NOT gaps and are deliberately not run through the resolver:
`yeast extract` and `peptone` are `intrinsically_undefined`; the two commercial
`yeast nitrogen base` lines, the SGA `SC amino-acid supplement powder`, Lian's
`CSM-URA`, Smith's `potassium phosphate buffer`, `Tween 40` and the `SynH3-` base are
`composition_deferred`. Undefined is the truth about what was in the bottle, so a
`ProvenanceGap` there would read as "not found yet" and would never close. The two
identity checks (`ontology_checks.media_library_compound_issues` and the verifier's
L3 `media_compound_identity`) both key on `ComponentDefinition`, so these are skipped
by construction rather than by an exception list.

`agar` and `tunicamycin` are the third case, `RESOLVED_MIXTURE`: a ChEBI and a CID
exist, a single-molecule InChIKey does not, so they satisfy the identity rule as they
are.

### 2. Every `base_medium` names a `MEDIA_LIBRARY` key

`_check_library()` runs at import and raises on any base that resolves to nothing. Two
labels used to dangle:

- **`SD`** was declared by `SD_MINIMAL` and named nothing. The constant is now `SD`
  and it is registered under that key, so the label is its own object.
- **`SD_MSG`** was declared by both SGA selection media and named nothing. It is now a
  first-class base: YNB without amino acids or ammonium sulfate at 1.7 g/L plus MSG at
  1 g/L, and **no carbon source**, because a base must be a component subset of
  everything deriving from it and the carbon source is stated per recipe (glucose for
  the standard screens, galactose for Costanzo 2021's alternative-carbon condition).
  `YP`, `YPB`, `YNB_YE_B` and `YNB` are carbon-free for the same reason;
  `CARBON_FREE_MEDIA` names each one with its reason and the metabolism test reads it.

The subset rule is what makes `YPD_LIQUID` a sibling of `YPD` rather than a lookalike:
`YPD` holds the three ingredients every member shares, and the agar row lives on
`YPD_AGAR`, whose source (Mota 2024) is the paper that states it.

### 3. `SC` was corrected, not forked

The shipped `SC` carried the 9 YNB vitamins, all 20 amino acids, uracil and adenine,
and **no carbon source at all**, so `media_to_bounds(SC)` opened no carbon exchange
and every dataset growing on SC reached FBA describing a medium nothing can grow in.
Wildenhain's review asked for an `SC_GLUCOSE` sibling; the honest fix is to correct
`SC`, because served loaders already import it and the next full rebuild picks the
correction up. `SC` now carries D-glucose at 20 g/L with two independent quotes
(Mormino 2022's 20 g/L and Wildenhain 2015's 2%, the same amount). Mormino's other two
figures, CSM 0.77 g/L and YNB w/o AA 6.9 g/L, sit in `SC.provenance` rather than as
components: this library EXPANDS the CSM powder into the amino acids and nucleobases
and the YNB into its vitamins, so listing the powders as well would double-count them.

### The library

51 media. `anchor` is the citation key(s) whose mirrored `paper.md` (or, for Bloom,
the sha256-pinned source-data xls) the medium's quotes are pinned to; every quote was
verified to be a literal substring of the pinned bytes.

| medium | base | state | anchor |
|---|---|---|---|
| `SC` | `SC` | liquid | morminoIdentificationAceticAcid2022, wildenhainPredictionSynergismChemicalGenetic2015 |
| `SC_MINUS_4_AMINOBENZOIC_ACID` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_MINUS_ADENINE` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_MINUS_FOLIC_ACID` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_MINUS_L_ARGININE` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_MINUS_L_ISOLEUCINE` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_MINUS_L_LYSINE` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_MINUS_L_THREONINE` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_MINUS_L_TRYPTOPHAN` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_MINUS_L_TYROSINE` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_MINUS_MYO_INOSITOL` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_MINUS_NIACIN` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_MINUS_THIAMINE_HYDROCHLORIDE` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_PARTIAL_BIOTIN` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_PARTIAL_CALCIUM_PANTOTHENATE` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_PARTIAL_PYRIDOXINE_HYDROCHLORIDE` | `SC` | liquid | hillenmeyerChemicalGenomicPortrait2008 (+ the SC component quotes) |
| `SC_URA` | `SC` | liquid | morminoIdentificationAceticAcid2022, wildenhainPredictionSynergismChemicalGenetic2015 |
| `SD` | `SD` | liquid | none yet (Difco formulation, cited in the module docstring only) |
| `SD_MSG` | `SD_MSG` | liquid | yantongSyntheticGeneticArray2006, kuzminSyntheticGeneticArray2016 |
| `SED_URA` | `SED_URA` | liquid | lianMultifunctionalGenomewideCRISPR2019 |
| `SED_URA_G418` | `SED_URA` | liquid | lianMultifunctionalGenomewideCRISPR2019 |
| `SGA_DM_SELECTION` | `SD_MSG` | solid | yantongSyntheticGeneticArray2006, kuzminSyntheticGeneticArray2016 |
| `SGA_DM_SELECTION_GALACTOSE` | `SD_MSG` | solid | costanzoEnvironmentalRobustnessGlobal2021 (+ the Tong base) |
| `SGA_TM_SELECTION` | `SD_MSG` | solid | kuzminSystematicAnalysisComplex2018 (+ the Tong base) |
| `SYNBASE` | `SYNH3_MINUS` | liquid | vanacloig-pedrosComparativeChemicalGenomic2022 |
| `SYNH3_MINUS` | `SYNH3_MINUS` | liquid | vanacloig-pedrosComparativeChemicalGenomic2022 |
| `YNB` | `YNB` | liquid | none yet (Difco formulation, cited in the module docstring only) |
| `YNB_GLUCOSE_SOLID` | `YNB` | solid | bloomRareVariantsContribute2019 |
| `YNB_YE_B` | `YNB_YE_B` | solid | smithExpressionFunctionalProfiling2006 |
| `YP` | `YP` | solid | bloomRareVariantsContribute2019 (+ the Tong YEPD base) |
| `YPAD` | `YPD` | solid | yantongSyntheticGeneticArray2006 |
| `YPB` | `YPB` | solid | smithExpressionFunctionalProfiling2006 |
| `YPBA` | `YNB_YE_B` | solid | smithExpressionFunctionalProfiling2006 |
| `YPBM` | `YNB_YE_B` | solid | smithExpressionFunctionalProfiling2006 |
| `YPBO` | `YPB` | solid | smithExpressionFunctionalProfiling2006 |
| `YPD` | `YPD` | solid | yantongSyntheticGeneticArray2006 |
| `YPD_AGAR` | `YPD` | solid | motaSharedMoreSpecific2024 (+ the Tong YEPD base) |
| `YPD_ETHANOL` | `YPD` | solid | bloomRareVariantsContribute2019 (+ the Tong YEPD base) |
| `YPD_LIQUID` | `YPD` | liquid | hoepfnerHighresolutionChemicalDissection2014 (+ the Tong YEPD base) |
| `YP_ETHANOL` | `YP` | solid | bloomRareVariantsContribute2019 |
| `YP_FRUCTOSE` | `YP` | solid | bloomRareVariantsContribute2019 |
| `YP_GALACTOSE` | `YP` | solid | bloomRareVariantsContribute2019 |
| `YP_GLYCEROL` | `YP` | solid | bloomRareVariantsContribute2019 |
| `YP_GLYCEROL_LIQUID` | `YP` | liquid | hillenmeyerChemicalGenomicPortrait2008 |
| `YP_LACTATE` | `YP` | solid | bloomRareVariantsContribute2019 |
| `YP_MALTOSE` | `YP` | solid | bloomRareVariantsContribute2019 |
| `YP_MANNOSE` | `YP` | solid | bloomRareVariantsContribute2019 |
| `YP_RAFFINOSE` | `YP` | solid | bloomRareVariantsContribute2019 |
| `YP_SUCROSE` | `YP` | solid | bloomRareVariantsContribute2019 |
| `YP_TREHALOSE` | `YP` | solid | bloomRareVariantsContribute2019 |
| `YP_XYLOSE` | `YP` | solid | bloomRareVariantsContribute2019 |

The review claimed every library entry is solid. Measured: it was not. `SD_MINIMAL`,
`YNB`, `SC` and `SC_URA` were already `state="liquid"` before this round; only the
YPD, YP and SGA families were solid. `state` is a property of each object, and the
base relation is about composition, so a liquid medium deriving from a solid base
(`YPD_LIQUID` off `YPD`) is not a contradiction.

### Reuse rules, so no dataset agent invents a sibling

- **Mota 2024's "YPD, pH 4.5" is not a new medium.** It is `YPD_LIQUID` (or
  `YPD_AGAR` for the spot-assay and CFU plates) plus
  `EnvironmentPhysicalPerturbation(factor=ph, magnitude=Concentration(value=4.5, unit=ph))`.
  pH rides as a typed physical perturbation, following the Bloom 2019 precedent.
- **Mormino 2022's "SC, pH 3.5" is not a new medium.** It is `SC` plus the same pH
  perturbation. Mormino's ATc is an inducer added at the start of cultivation; pick
  one convention for it across mormino2022 and smith2016 before either lands.
- **Smith 2016's "SCM-Ura" is `SC_URA`.** Smith releases no SCM-Ura recipe, so the
  shipped gaps are the honest state; do not create a Smith-specific sibling.
- **Wildenhain 2015's "SC with 2% glucose" is `SC`.** That is what the correction
  above is for.
- **Hillenmeyer's `minimal media` is `SD` and `synthetic complete` is `SC`**, both
  `is_synthetic=True`. The loader currently emits them as name-only media with
  `is_synthetic=False`, which is a factual error on 66,984 records.
- **Hillenmeyer's `YP glycerol` is `YP_GLYCEROL_LIQUID`, never Bloom's
  `YP_GLYCEROL`.** Bloom's 3% is Bloom's bench value on Bloom's plates, quoted from a
  sha256-pinned xls; asserting it for a Hillenmeyer liquid pool would fabricate a
  number. The Hillenmeyer object leaves the concentration `None` with
  `defers_to=["pierceGenomewideAnalysisBarcoded2006"]`, which the SOM defers the whole
  growth protocol to and which is not mirrored. The two join at
  `base_medium == "YP"` and at the glycerol compound.
- **Hillenmeyer's 15 named-nutrient dropouts are derived media off `SC`**, keyed by
  the HOM matrix's own condition label in `HILLENMEYER_DROPOUT_MEDIA`. The 16th label,
  `vitamin drop-out control media`, is not a dropout: it maps to plain `SC`. Three
  vitamin conditions the source calls "partial" keep the nutrient with its
  concentration cleared and a note, because a partial reduction is not a removal and
  the reduced level is never stated. The SOM states no base recipe for the series; SC
  is the defined medium a named-nutrient dropout is taken against, and that inference
  is recorded on every one of the objects rather than left implicit.
- **Costanzo 2021's galactose condition is a carbon SWAP, not an additive.** The paper
  calls it "an alternative carbon source", so it is `SGA_DM_SELECTION_GALACTOSE` (the
  glucose row replaced by galactose at 2% w/v) with no environment perturbation. The
  additive form asserted glucose AND galactose in the same flask, and lost the
  compound in the graph besides, since the adapter writes no `compound_name` for an
  `EnvironmentPhysicalPerturbation`.
- **Lian 2019's SED-URA is its own base, not a derivative of `SD_MSG`.** Chemically it
  is the same nitrogen base (0.17% w/v YNB is 1.7 g/L; 0.1% MSG is 1 g/L), but the
  paper writes "yeast nitrogen base" without the ammonium-sulfate-free qualifier, and
  inventing that qualifier to force the join would be a guess in the join key itself.
  Sourcing it later is what would let `SED_URA` derive from `SD_MSG`.

### Rebuild consequence, stated plainly

Media node ids are the sha256 of the model dump, so resolving the library's compounds
changes the node id of every medium in it, including the SGA selection media that five
served datasets carry. This is an instance-level change with no schema-class change:
`Media`, `MediaComponent` and `Compound` contracts are untouched, so `kg_manifest`'s
admission gate does not see it. It is therefore a change that must land WITH a full
rebuild, not slipped in ahead of one, and the same holds for migrating the six loaders
that still spell liquid YPD as a bare `Media(name="YPD", state="liquid")` onto
`YPD_LIQUID`.

### Open gaps carried forward

- `SD` and `YNB` carry no `SourcedValue` at all: their Difco formulation is cited in
  the module docstring but never quoted against a mirrored artifact.
- The 9 YNB vitamins and the 20 SC amino acids are identified but have no
  concentration, so `SC.open_gaps` lists them all.
- `SC` names no nitrogen source: the ammonium sits inside the (unlisted) YNB line, so
  FBA gets its nitrogen from `torchcell/metabolism/media.py`'s `SM_FBA` expansion
  rather than from the ontology object.
- `cellobiose` is the one library substance the pinned compound table has not
  resolved; it is honestly gapped and is a one-row builder addition away from closing.

## 2026.10.07 - Bacterial media

Source: `torchcell/datamodels/media.py` (the BACTERIAL MEDIA section, `BACTERIAL_MEDIA_USES`)
Tests: `tests/torchcell/datamodels/test_media_bacterial.py`
Plan: [[plan.bacteria-ontology-genome]] section 3d and Step 5 (`feat/bacterial-media-library`).

Twenty-five `MEDIA_LIBRARY` keys added (79 total, from 54), each quoting a mirrored file
of one of the fifty bacterial rows with that file's sha256 from the key's `manifest.json`.
No schema change: `Media`, `MediaComponent` and `ConcentrationUnit` express all of it.
Every quote was re-read against the mirror (`pytest --data`, 190 passed).

### Conventions the entries follow

- **"M9" is not one recipe.** Across the fifty it names at least six formulations:
  ammonium chloride or ammonium sulfate as the nitrogen salt, anhydrous or hydrated
  phosphate, with or without trace metals, MOPS or a vendor salts powder. So `M9` is the
  four salts and each paper's formulation is its own key, suffixed with the paper.
- **A replaced base component is a dropout** (the SynBase convention). The
  ammonium-sulfate formulations drop the base's ammonium chloride; a medium that weighs
  the phosphate as a hydrate drops the anhydrous salt. `media_derivation_issues` stays
  empty.
- **A hydrate is its own compound.** 7.52 g of Na2HPO4·2H2O is not 7.52 g of Na2HPO4, and
  PubChem gives each hydrate its own InChIKey.
- **A varied carbon or nitrogen source is not a component.** Such media are listed in
  `CARBON_FREE_MEDIA` with the reason, and the loader carries the variable as
  `EnvironmentPhysicalPerturbation(factor=carbon_source | nitrogen_source)`.
- **xlsx quotes are row renderings.** Wetmore 2015 Data Set S1 and Price 2018
  Supplementary Table 18 are binary, so a quote is a block of consecutive rows of one
  sheet, cells joined by ` | `, rows by ` / `; the test re-reads the sheet.

### Entries and sources

Rank is the row's rank among the fifty. Base is `base_medium`.

| key | base | carbon | source (citation key, file) | rows |
|---|---|---|---|---|
| `LB` | LB | none (complex) | Menasalvas 2025, Schmidt 2016 (`paper.md`); Goodall 2018, Babu 2014 SI 13 corroborate | 6, 11, 20, 22, 23, 24, 28, 32, 40, 42 |
| `LB_AGAR` | LB | none (complex) | Menasalvas 2025, Schmidt 2016 | 20, 32 |
| `LB_LENNOX` | LB | none (complex) | Schastnaya 2021; Wetmore 2015 Data Set S1; Price 2018 Table S18 | 2, 21, 30, 46 |
| `YT_2X` | YT_2X | none (complex) | Wang 2015 | 15 |
| `M9` | M9 | none (base) | Kang 2026; Borchert 2024 corroborates | 18, 8; base of 14 more |
| `M9_GLUCOSE` | M9 | 2 g/L glucose | Choe 2019 | 45 |
| `M9_NOCARBON_WETMORE2015` | M9 | varied | Wetmore 2015 Data Set S1 | 2 |
| `M9_NONITROGEN_WETMORE2015` | M9 | 4 g/L glucose (N varied) | Wetmore 2015 Data Set S1 | 2 |
| `M9_NOCARBON_PRICE2018` | M9 | varied | Price 2018 Table S18 | 21 |
| `M9_NONITROGEN_PRICE2018` | M9 | 4 g/L glucose (N varied) | Price 2018 Table S18 | 21 |
| `M9_GLUCOSE_CASEIN_FUHRER2017` | M9 | 4 g/L glucose | Fuhrer 2017 | 1 |
| `M9_SCHMIDT2016` | M9 | varied | Schmidt 2016 | 32 |
| `M9_NREL_DESIQUEIRA2025` | M9 | varied | de Siqueira 2025 | 14 |
| `M9_NREL_LIM2025` | M9 | 4 g/L glucose | Lim 2025 | 16 |
| `M9_NREL_KANG2026` | M9 | varied | Kang 2026 | 18 |
| `M9_NREL_HIGH_N_KANG2026` | M9 | varied | Kang 2026 | 18 |
| `M9_MOPS_KANG2026` | M9 | varied | Kang 2026 | 18 |
| `M9_NREL_CARRUTHERS2025` | M9 | 20 g/L glucose | Carruthers 2025 | 6 |
| `M9_NREL_MOPS_MENASALVAS2025` | M9 | 2% glucose | Menasalvas 2025 | 20 |
| `M9_DIFCO` | M9_DIFCO | none (base) | Foo 2014 | 12 |
| `M9_DIFCO_GLUCOSE_FOO2014` | M9_DIFCO | 0.4% glucose | Foo 2014 | 12 |
| `MM9_FOO2014` | M9_DIFCO | 1% glucose | Foo 2014 | 12 |
| `DAVIS_MINIMAL` | DAVIS_MINIMAL | varied | Caglar 2017 (recipe deferred to Lenski 1991) | 9 |
| `DM500` | DAVIS_MINIMAL | 500 mg/L glucose | Caglar 2017 | 9 |
| `MOPS_MINIMAL` | MOPS_MINIMAL | none (base) | Price 2018 Table S18; Tong 2020 cites Neidhardt 1974 | 4, 21 |

The anchoring quotes, verbatim (OCR markup kept, as in `media.py`):

- `LB`: `menasalvasBiosensordrivenStrainEngineering2025/paper.md` (sha256 `d1453694...`):
  `The LB Miller (Luria-Bertani) medium [tryptone $( 1 0 \mathrm { g / }$ liter), yeast extract $( 5 ~ \mathrm { g / l i t e r } )$ , and NaCl (10 g/liter)]`;
  `schmidtQuantitativeConditiondependentEscherichia2016/paper.md` (`67bedae8...`):
  `Five grams of yeast extract (BD), $_ { 1 0 \mathrm { ~ g ~ } }$ Tryptone (BD) and $1 0 \ \mathrm { g \ N a C l }$ were dissolved in one liter of water`.
- `LB_AGAR`: Menasalvas 2025:
  `LB medium was supplemented with $2 \%$ $\scriptstyle \left( \mathbf { w } / \mathbf { v } \right)$ solid agar`;
  Schmidt 2016: `LB plates were produced by adding $2 0 \mathrm { g }$ agar (BD)`.
- `LB_LENNOX`: `schastnayaExtensiveRegulationEnzyme2021/paper.md` (`f299eee7...`):
  `LB-Lennox medium $\mathrm { { \Delta } _ { 1 0 } g / L }$ tryptone, $5 \mathrm { g / L }$ yeast extract, $5 \mathrm { g / L }$ NaCl)`;
  `wetmoreRapidQuantificationMutant2015/si/si1.xlsx` (`428a06ca...`), sheet `Media`:
  `LB / defined | False / desc | Luria-Bertani broth / ... / Sodium Chloride | 5 | g/L`;
  `priceMutantPhenotypesThousands2018/si/si3.xlsx` (`e5dbf3d5...`), sheet
  `TableS18_Medias`: `Media | LB / Description | Luria-Bertani broth / ... / Sodium Chloride | 5 | g/L`.
- `YT_2X`: `wangDynamicInterplayMultidrug2015/paper.md` (`9cd90f00...`):
  `in 2YT medium (Bacto-tryptone $1 6 { \mathrm { g } } ,$ Bacto-yeast extract $1 0 { \mathrm { g } } { \mathrm { . } }$ and sodium chloride $5 \mathrm { g }$ per liter`.
- `M9`: `kangMultilayeredMetabolicRemodeling2026/paper.md` (`894aee24...`):
  `M9 salts $( 6 . 7 8 \ g / \mathrm { L }$`, then 3 g/L KH2PO4, 1 g/L NH4Cl and 0.5 g/L
  NaCl in the same parenthesis; `borchertMachineLearningAnalysis2024/paper.md`
  (`9de6b077...`): `$( 6 . 7 8 ~ \mathrm { g / L ~ N a _ { 2 } H P O _ { 4 } , 3 ~ \mathrm { g / L ~ K H _ { 2 } P O _ { 4 } , 0 . 5 ~ \mathrm { g / L ~ N a C l } , } }$ 1 g/L $\mathsf { N H } _ { 4 } \mathsf { C l }$`.
- `M9_GLUCOSE`: `choeAdaptiveLaboratoryEvolution2019/paper.md` (`11a04220...`):
  `Cells were grown in M9 glucose medium (47.75 $\mathrm { m M }$ of ${ \mathrm { N a } } _ { 2 } { \mathrm { H P O } } _ { 4 } ,$`
  through `and $2 { \bf g } 1 ^ { - 1 }$ of glucose)`.
- `M9_NOCARBON_WETMORE2015` / `M9_NONITROGEN_WETMORE2015`: Data Set S1, sheet `Media`,
  blocks `M9 minimal media_noCarbon` (`Sodium phosphate dibasic heptahydrate | 13 | g/L`)
  and `M9 minimal media_noNitrogen` (`D-Glucose | 4 | g/L`); sheet `Expts_Keio` rows
  `... | M9 minimal media_noCarbon | ... | D-Glucose | 20 | mM | ...` and
  `... | M9 minimal media_noNitrogen | ... | L-Arginine | 10 | mM | ...` show the varied
  source.
- `M9_NOCARBON_PRICE2018` / `M9_NONITROGEN_PRICE2018`: Table S18 blocks
  `Media | M9 minimal media_noCarbon` and `Media | M9 minimal media_noNitrogen`
  (`Sodium phosphate dibasic heptahydrate | 12.8 | g/L`). Table S5 counts 60 and 32 Keio
  experiments on them.
- `M9_GLUCOSE_CASEIN_FUHRER2017`: `fuhrerGenomewideLandscapeGene2017/paper.md`
  (`ec87736b...`): `were grown on glucose minimal medium supplemented with casein hydrolysate containing (per liter):`
  through `$1 ~ \mathrm { m g }$ thiamine HCl.`
- `M9_SCHMIDT2016`: Schmidt 2016: `$2 0 0 ~ \mathrm { m l }$ f $5 \times$ base salt solution (211 mM`
  and the trace-element, CaCl2, MgSO4, thiamine and FeCl3 stock clauses; each amount is
  the stated stock diluted by the stated volume into one liter.
- `M9_NREL_DESIQUEIRA2025`: `desiqueiraAlternateRoutesAcetate2025/paper.md`
  (`a3ea14ad...`): `minimal salt (M9) medium composed of $1 \times 1 \mathsf { M } 9$ salts $( 2 ~ { \mathfrak { g } } / { \mathsf { L } } ~ ( { \mathsf { N H } } _ { 4 } ) _ { 2 } { \mathsf { S O } } _ { 4 } ,$`.
- `M9_NREL_LIM2025`: `limEvolutionguidedToleranceEngineering2025/paper.md`
  (`26b88d81...`): `The M9 medium contained $2 \ g / \mathrm { L }$` and
  `As a carbon source, $4 \ g / \mathrm { L }$ glucose was added to the minimal medium unless otherwise stated.`
- Kang 2026 (three keys): `M9 medium was prepared with the following components:`,
  `the concentration of $\mathrm { ( N H } _ { 4 } \mathrm { ) } _ { 2 } S 0 _ { 4 }$ was increased to $4 0 ~ \mathrm { m M }$ and is referred to as modified M9.`
  and `M9-MOPS was prepared with the following components:`.
- `M9_NREL_CARRUTHERS2025`: `carruthersAutomationMachineLearning2025/paper.md`
  (`ca9a8a25...`): `Briefly, the medium composition included $2 0 \mathrm { g / L }$ glucose,`.
- `M9_NREL_MOPS_MENASALVAS2025`: Menasalvas 2025:
  `At the 1X working concentration, M9 medium contains 47.9 mM` and
  `This formulation of M9 used for $P .$ putida is sometimes referred to as "NREL`.
- Foo 2014 (three keys): `fooImprovingMicrobialBiogasoline2014/paper.md`
  (`b24baad4...`): `Growth assays were performed in M9 minimal medium, which consisted of $1 \times$ M9 salt (Difco)`
  and `Isopentenol production strains were grown in a modified M9 3-morpholinopropane-1-sulfonic acid (MOPS) minimal medium (MM9)`.
- `DAVIS_MINIMAL` / `DM500`: `caglarColiMolecularPhenotype2017/paper.md`
  (`0878d5e7...`): `Davis Minimal medium supplemented with $2 \mu \mathrm { g } / 1$ thiamine $( \mathrm { D M } ) ^ { 3 6 }$ and limiting glucose at $5 0 0 \mathrm { m g / l }$ (DM500)`;
  ref 36: `36. Lenski, R. E., Rose, M. R., Simpson, S. C. & Tadler, S. C. Long-Term Experimental Evolution in Escherichia coli.`
- `MOPS_MINIMAL`: Table S18 block `Media | MOPS minimal media_noCarbon` (MOPS 40 mM through
  `Zinc sulfate heptahydrate | 1e-08 | M`); `tongGeneDispensabilityEscherichia2020/paper.md`
  (`daea2b92...`): `Using a chemically defined minimal medium (morpholinepropanesulfonic acid [MOPS]) and changing only the carbon source (34)`
  and `34. Neidhardt FC, Bloch PL, Smith DF. 1974. Culture medium for enterobacteria.`

### Adjudications and readings

- **MOPS sulfate row.** Price 2018's Table S18 lists
  `Aluminum potassium sulfate dodecahydrate | 0.276 | mM`; the same lab's Wetmore 2015
  Data Set S1 lists `Potassium Sulfate | 0.276 | mM` in its MOPS formulation, and the two
  other mirrored MOPS recipes (Schmidt 2022, Thompson 2020) name K2SO4. Recorded as
  potassium sulfate, both rows quoted on the component. Neidhardt 1974 would settle it
  and is not mirrored.
- **Wetmore 2015 and Price 2018 "LB" is Lennox.** Their prose says "LB"; their own media
  tables give 5 g/L NaCl. Their loaders take `LB_LENNOX`.
- **Wetmore 13 g/L vs Price 12.8 g/L** of the heptahydrate under the same medium name:
  two objects, each paper's own.
- **Foo 2014 spellings.** "manganese chloride" is the anhydrous MnCl2 row (PubChem's name
  index answers that spelling with a four-water record, so it was not looked up by
  name); "cupric sulfate" is the existing copper sulfate row; "ammonium molybdate"
  resolves through PubChem's name index to the heptamolybdate record (CID 485454), the
  compound Kang 2026 writes as (NH4)6Mo7O24.
- **OCR readings recorded in notes, never silently:** Carruthers' NaCl `0.58/L` is 0.5 g/L
  (PDF text layer agrees); Kang's KH2PO4 `3 \gimel A` is 3 g/L; Choe's `g 1 ^ { - 1 }` is
  g/L; Schastnaya's `\Delta _ { 1 0 }` is `(10`; Goodall's `9` is `g`. Menasalvas'
  70 mM ammonium sulfate was checked against the PDF text layer because it is far above
  the other NREL papers' 10 to 15 mM.

### Compound table additions

`torchcell/datamodels/compound_identity_inputs/bacterial_media.txt` lists the 34 labels;
the curator was run on that list alone (PubChem PUG REST, 2026-10-07) and its 32 rows
were merged into `compound_identity_table.json` without touching an existing row (the
diff is additions only, and the merge asserted that no new name, synonym or CID was
already claimed). `_TABLE_SHA256` re-pinned to `85d6c25a...`. Labels that already had a
row were reused: sodium chloride, ammonium sulfate, cobalt chloride, copper sulfate,
manganese sulfate, MnCl2 (`manganese (ii) chloride`), thiamine hydrochloride, D-glucose,
agar, yeast extract.

### Still unsourced (named, not guessed)

- **Nichols 2011** (rank 10): not mirrored, no PDF. No medium entry.
- **PRECISE-1K / Lamoureux 2023** (rank 5): names "M9 minimal media with glucose" for the
  control condition and prints no recipe.
- **Yunus 2026** (rank 13): "M9 medium with 2% glucose", no recipe.
- **Mutalik 2020** (rank 3): "LB" deferred to ref 96 (Bertani 2004), not mirrored.
- **Tong 2020** (rank 4): the carbon-source concentrations live in an external web app
  (Carbon Phenotype Explorer), not in the mirror.
- **Tian 2019, Wang 2022** (ranks 17, 19, both blocked): EZ Rich (Teknova, vendor
  formulation); Wang 2022's M9 is deferred to its ref 29.
- **Lim 2022, Borchert 2024** (ranks 7, 8, aggregations): no recipe for the aggregated
  data; Borchert's own growth-assay M9 (salts + Mg + Ca + 18 uM FeSO4) is stated but not
  added, since its fitness data come from other studies.
- **Vendor and deferred lines** inside entries: Teknova T1001 trace metals (and Carruthers'
  volume basis, Menasalvas' "1X"), Difco M9 salts, the Lim 2020 / Linger 2014 2000x
  trace element solution, the Davis Minimal salts (Lenski 1991), and Schmidt 2016's
  thiamine amount, which the pinned OCR lost (the PDF text layer reads 1.4 mM stock, 2.8
  uM final; not recorded until the OCR is redone).
- **Stated in the mirror but not added** (outside the first tranche or needing many new
  compounds): the MOPS Rich Defined medium (Wetmore and Price, 4 Keio experiments each;
  Wetmore's micronutrient units are corrupt in Data Set S1), Shiver 2016's LB Lennox agar
  (90 mM NaCl) and M9 variants (rank 27), Rapp 2026's M9 (rank 23), the modified MOPS of
  Schmidt 2022 and Thompson 2020 (ranks 24, 28), Fuhrer 2017's perturbation medium
  without casein hydrolysate, and Gupta 2024's and Rachwalski 2024's Teknova MOPS kits
  (vendor formulations).

### Metabolism test scope

`tests/torchcell/metabolism/test_media.py::test_every_library_medium_resolves_or_says_why_not`
now runs over the yeast keys only: its toy model carries the yeast recipes' species and
the resolver's dissociation table covers the yeast salts, so the M9 phosphates, ammonium
chloride, MOPS, tricine, borate, molybdate and cobalt have no exchange there. Mapping a
bacterial medium onto a bacterial GEM is the bacterial FBA work. The carbon-source test
still covers every key.

### Admission consequence

`media.py`, `compound_identity.py` and `compound_identity_table.json` are all in
`VALUE_SURFACE_RELPATHS`, so the next `kg_manifest admit` reports value drift on all
three and blocks until `--ack-value-drift` is given. The honest reason, now checkable:
25 `MEDIA_LIBRARY` keys added and 32 compound-table rows added; no pre-existing key's
`media_identity` digest changed (`test_media_bacterial.py` pins all 54, computed on main
at `bb31eabbe` and on this branch, identical) and no pre-existing table row changed (the
table diff is additions only). A served medium or compound node id therefore does not
move, which is what makes the acknowledgment honest rather than a bypass.
