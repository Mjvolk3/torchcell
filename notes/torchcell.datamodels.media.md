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
