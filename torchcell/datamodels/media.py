# torchcell/datamodels/media
# [[torchcell.datamodels.media]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/media
"""Reusable, provenance-first definitions of the common growth media.

Each constant is a fully-typed :class:`~torchcell.datamodels.schema.Media` with
component-level composition and ``SourcedValue`` provenance, so a dataset loader
imports one canonical object instead of re-declaring ``Media(name=..., state=...)``.

The library is the JOIN LAYER
----------------------------
Two properties make that join real, and both are enforced by tests rather than by
convention:

- **Every single-substance component and every dropout goes through**
  :func:`torchcell.datamodels.compound_identity.resolved_compound`, so it carries an
  InChIKey / ChEBI / PubChem CID and joins to a chemogenomic dataset's compound for
  the same substance. An undefined preparation (yeast extract, peptone, commercial
  YNB, an SC supplement powder, a polysorbate) is NOT run through the resolver: it is
  a bare ``Compound`` with ``definition=intrinsically_undefined`` or
  ``composition_deferred``, because "undefined" is the truth about the bottle, not a
  gap to be filled. Agar and tunicamycin are the third case, ``RESOLVED_MIXTURE``:
  ChEBI and a CID exist, a single-molecule InChIKey does not.
- **Every ``base_medium`` names a key of** :data:`MEDIA_LIBRARY`, checked at import by
  :func:`_check_library`. A base that resolves to no object carries no components and
  no provenance, so "aggregate every record on an SD/MSG base" would join nothing. A
  base is also a component SUBSET of everything deriving from it, which is what lets
  ``SC-Ura`` be a typed edit of ``SC`` instead of a lookalike recipe.

Sourcing:
- SGA selection media + the SC amino-acid supplement powder recipe come from
  **Tong & Boone 2006** (``yantongSyntheticGeneticArray2006``, Methods Mol Biol
  313:171-192, Materials sec. 2.1) -- the canonical Boone-lab SGA protocol -- with
  the modern names/MSG rationale corroborated by **Kuzmin 2016** CSH Protocols
  (``kuzminSyntheticGeneticArray2016``, the ref-67 deferral target) and the
  screen temperature from the **Kuzmin 2018** SI (``kuzminSystematicAnalysisComplex2018``).
- The YNB vitamin set + SC amino-acid/nucleobase inventory follow the standard
  Difco/Sigma YNB and SC formulations (ported from the iBioFoundry Yeast9 FBA
  media inventory); the FBA supplement-uptake convention (5% of glucose) is a
  MODEL convention from **Suthers 2020** (``suthersGenomescaleMetabolicReconstruction2020``,
  Metab Eng Commun, doi 10.1016/j.mec.2020.e00148) and lives in a future
  cobra/AMICI adapter, NOT in these wet-lab records.
- Every medium added for the serve-50 review round quotes ONE sentence of a mirrored
  ``paper.md`` (sha256 from that key's ``manifest.json``). Those files are MinerU OCR
  output, so a quote carries the source's LaTeX math markup verbatim rather than a
  cleaned-up rendering of it; the plain reading goes in ``SourcedValue.value`` and the
  quote stays a literal substring of the pinned bytes.
- The bacterial media (Step 5 of ``[[plan.bacteria-ontology-genome]]``) quote the
  mirrored Methods or SI of the fifty bacterial rows; the comment above them states the
  conventions (paper-suffixed keys for per-paper formulations, a replaced base salt as a
  dropout, a hydrate as its own compound, row-rendered quotes for xlsx sources), and
  ``BACTERIAL_MEDIA_USES`` records which rows use each entry.

Follow-ups (documented gaps, fillable later):
- Concentrations are still absent for the YNB vitamins and the SC amino acids: the
  identity is sourced, the wet-lab amount is not (``Media.open_gaps`` reports them).
- ``SC``'s nitrogen source is inside the (unlisted) YNB line; the ontology object
  names no ammonium salt, so FBA gets its nitrogen from
  ``torchcell/metabolism/media.py``'s ``SM_FBA`` expansion rather than from here.

Design note: ``[[torchcell.datamodels.media-components]]``.
Round note: ``[[torchcell.datamodels.media]]``.
"""

from __future__ import annotations

from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    Media,
    MediaComponent,
    MediaComponentRole,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

# --------------------------------------------------------------------------- #
# Provenance anchors. ``paper.txt`` sha256 for the two SGA protocol papers (the
# pre-existing pins); ``paper.md`` sha256 from each key's manifest.json for the
# serve-50 round.
# --------------------------------------------------------------------------- #
_TONG2006 = "yantongSyntheticGeneticArray2006"
_TONG2006_SHA = "dda5fc727c5e532e02884cd1d30ad0774bfb773ed55e8f6b7074ee2158ab9aca"
_KUZMIN2016 = "kuzminSyntheticGeneticArray2016"
_KUZMIN2016_SHA = "02360306e6d0eb6324a9af962b8970cad019b3e06d5073688925614a657848ca"
_HOEPFNER2014 = "hoepfnerHighresolutionChemicalDissection2014"
_HOEPFNER2014_SHA = "a9877549eff2fe1aaf8aa403d9fea1c381284de030326f4475e869c102af0aeb"
_HILLENMEYER2008 = "hillenmeyerChemicalGenomicPortrait2008"
_HILLENMEYER2008_SHA = (
    "cf4759f00083de78dd953b12dd66d4360a2f645321305f768403f26b451c1df0"
)
_VANACLOIG2022 = "vanacloig-pedrosComparativeChemicalGenomic2022"
_VANACLOIG2022_SHA = "0b5d938b54b8424fa08203a4357bc8f7c7dfae3fbe1a6d07d422848b92f37ba3"
_WILDENHAIN2015 = "wildenhainPredictionSynergismChemicalGenetic2015"
_WILDENHAIN2015_SHA = "f46409eb8f23412c9c1015d0f8f5bb581bfddfe2796d319d407585e23c757ac2"
_SMITH2006 = "smithExpressionFunctionalProfiling2006"
_SMITH2006_SHA = "eb5ab21b842365e2138528bbce936bd68134dd97ee99eb9a58502c25ca2948c6"
_LIAN2019 = "lianMultifunctionalGenomewideCRISPR2019"
_LIAN2019_SHA = "63fe2b7101fc48feb297f9e34b83d108b74f03f28bbc280e08c7219bc975086c"
_MORMINO2022 = "morminoIdentificationAceticAcid2022"
_MORMINO2022_SHA = "f5d38e486148527bfba9dc9e40a9eb06ba051766aeca3e8ff67663551bf043c3"
_MOTA2024 = "motaSharedMoreSpecific2024"
_MOTA2024_SHA = "a19769f757fd912139551f39736dd2b67581cb03f83a7a9b9385e28516b1f1b6"
_COSTANZO2021 = "costanzoEnvironmentalRobustnessGlobal2021"
_COSTANZO2021_SHA = "ba22973ed0c53c00c37bcfb9f659d3b0373c451a3f7633158afae274035559fb"

#: Deferral targets that are NOT mirrored. Naming one in ``defers_to`` is the typed
#: way to say "the fuller definition exists, in a paper we do not hold".
_PIERCE2006 = "pierceGenomewideAnalysisBarcoded2006"
_ZHANG2019 = "zhangMultiomicFermentationUsing2019"


def _sv(
    value: object,
    quote: str,
    *,
    ck: str = _TONG2006,
    sha: str = _TONG2006_SHA,
    uri: str = "paper.txt",
    note: str | None = None,
) -> SourcedValue:
    """A SourcedValue pinned to a mirrored artifact (quote + sha256)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(source_uri=uri, citation_key=ck, sha256=sha),
        quote=quote,
        note=note,
    )


def _c(value: float, unit: ConcentrationUnit) -> Concentration:
    return Concentration(value=value, unit=unit)


_PCT = ConcentrationUnit.percent_w_v
_GL = ConcentrationUnit.g_per_l
_UGML = ConcentrationUnit.ug_per_ml

#: Recorded once and referenced by every component whose source writes a bare "%".
_PERCENT_BASIS_NOTE = (
    "the source writes '%' with no w/v or v/v basis; recorded as w/v, which is the "
    "basis for every other percentage in this library"
)


def _defined(
    name: str,
    role: MediaComponentRole,
    *,
    concentration: Concentration | None = None,
    provenance: list[SourcedValue] | None = None,
    note: str | None = None,
    defers_to: list[str] | None = None,
) -> MediaComponent:
    """A single-substance component, resolved through the shared identity table.

    ``resolved_compound`` returns the table's CANONICAL name, so two recipes spelling
    one substance differently produce one compound, and it attaches a typed
    ``ProvenanceGap`` on ``inchikey`` when no structure is available rather than
    leaving a silent ``None``.
    """
    return MediaComponent(
        compound=resolved_compound(name),
        role=role,
        concentration=concentration,
        provenance=provenance or [],
        note=note,
        defers_to=defers_to or [],
    )


def _mixture(
    name: str,
    role: MediaComponentRole,
    definition: ComponentDefinition,
    *,
    concentration: Concentration | None = None,
    provenance: list[SourcedValue] | None = None,
    note: str | None = None,
    defers_to: list[str] | None = None,
) -> MediaComponent:
    """A preparation that is not one substance: bare ``Compound``, no identity gap.

    Peptone, yeast extract, commercial YNB, an SC supplement powder and a polysorbate
    have no structure to find, so demanding one (or recording its absence as a gap)
    would misstate what is in the flask. The ``definition`` field is the honest
    encoding, and it is what the identity checks key on.
    """
    return MediaComponent(
        compound=Compound(name=name),
        role=role,
        concentration=concentration,
        definition=definition,
        provenance=provenance or [],
        note=note,
        defers_to=defers_to or [],
    )


_UNDEFINED = ComponentDefinition.intrinsically_undefined
_DEFERRED = ComponentDefinition.composition_deferred


def dropout(
    base: Media,
    *compound_names: str,
    name: str,
    partial: tuple[str, ...] = (),
    provenance: list[SourcedValue] | None = None,
    partial_note: str | None = None,
) -> Media:
    """``base`` with the named compounds moved from ``components`` into ``dropouts``.

    A nutrient dropout is a typed EDIT of a defined medium, not a perturbation of an
    unrelated one: encoding it this way is what makes a tryptophan dropout join to
    ``SC``, to ``SC_URA`` and to the SGA selection media at the same base.

    ``compound_names`` are looked up through ``resolved_compound`` so a source's
    spelling ("tryptophan", "PABA") selects the base's canonical component
    ("L-tryptophan", "4-aminobenzoic acid"). A name that matches nothing in ``base``
    raises: a silent no-op dropout would claim an edit that never happened.

    ``partial`` names compounds the source REDUCED rather than removed. They stay in
    ``components`` with the concentration cleared, because the reduced level is a
    number the source does not give; dropping them instead would overstate the edit.
    ``partial_note`` replaces the default note on those components when the source
    does say something about the level (e.g. a released percentage whose basis it
    never defines).
    """
    removed = [resolved_compound(n) for n in compound_names]
    reduced = [resolved_compound(n) for n in partial]
    present = {c.compound.name for c in base.components}
    for compound in [*removed, *reduced]:
        if compound.name not in present:
            raise ValueError(
                f"{compound.name!r} is not a component of {base.name!r}; a dropout "
                "must name something the base medium actually contains"
            )
    removed_names = {c.name for c in removed}
    reduced_names = {c.name for c in reduced}
    components = [
        (
            component.model_copy(
                update={
                    "concentration": Concentration(
                        value=None, basis=DoseBasis.reduced_from_standard
                    ),
                    "note": partial_note
                    or "partial drop-out; the source does not state the reduced "
                    "level, so the amount is a typed reduction rather than a removal",
                }
            )
            if component.compound.name in reduced_names
            else component
        )
        for component in base.components
        if component.compound.name not in removed_names
    ]
    return Media(
        name=name,
        state=base.state,
        is_synthetic=base.is_synthetic,
        base_medium=base.base_medium,
        components=components,
        dropouts=[*base.dropouts, *removed],
        provenance=provenance or list(base.provenance),
    )


def restated(base: Media, *statements: SourcedValue) -> Media:
    """``base`` with a dataset paper's own medium statements appended to its provenance.

    The loader-side half of a deferral: a paper that names a library medium ("YPD",
    "synthetic complete") but prints no recipe, or prints the same recipe, keeps the
    library object's composition and adds its own verbatim sentence(s). ``name`` and
    ``provenance`` are not part of ``media_identity``, so the result joins the library
    object exactly; only the record's statement of who said so grows.
    """
    if not statements:
        raise ValueError("restated() needs at least one SourcedValue from the paper")
    return base.model_copy(update={"provenance": [*base.provenance, *statements]})


# --------------------------------------------------------------------------- #
# SD/MSG -- the SGA base. Tong & Boone 2006 recipe #16 (per L): 1.7 g YNB w/o amino
# acids or ammonium sulfate, 1 g MSG, 2 g amino-acids supplement powder
# (DO -His/Arg/Lys), 20 g bacto agar, 50 mL 40% glucose (= 20 g/L), canavanine
# 50 mg/L, thialysine 50 mg/L, G418 200 mg/L, clonNAT 100 mg/L. Whole SGA screen
# incubated at 26 C (Kuzmin 2018 SI).
#
# The BASE object carries only the nitrogen half (YNB w/o AA+AS, MSG): a base must be
# a component subset of every medium deriving from it, and the carbon source is stated
# per recipe (glucose for the standard screens, galactose for Costanzo 2021's
# alternative-carbon condition).
# --------------------------------------------------------------------------- #
_SGA_SUPPLEMENT_QUOTE = (
    "Amino-acids supplement powder mixture for synthetic media (complete): "
    "Contains 3 g adenine (Sigma), 2 g uracil (ICN), 2 g inositol, 0.2 g "
    "para-aminobenzoic acid (Acros Organics), 2 g alanine, 2 g arginine, 2 g "
    "asparagine, 2 g aspartic acid, 2 g cysteine, 2 g glutamic acid, 2 g "
    "glutamine, 2 g glycine, 2 g histidine, 2 g isoleucine, 10 g leucine, 2 g "
    "lysine, 2 g methionine, 2 g phenylalanine, 2 g proline, 2 g serine, 2 g "
    "threonine, 2 g tryptophan, 2 g tyrosine, 2 g valine (Fisher). Drop-out (DO) "
    "powder mixture is a combination of the aforementioned ingredients minus the "
    "appropriate supplement. 2 g of the DO powder mixture is used per liter of medium"
)
_SGA_BASE_QUOTE = (
    "(SD/MSG) - His/Arg/Lys + canavanine/thialysine/G418/clonNAT: Add 1.7 g yeast "
    "nitrogen base without amino acids or ammonium sulfate, 1 g MSG, 2 g "
    "amino-acids supplement powder mixture (DO - His/Arg/Lys), ... add 50 mL 40% "
    "glucose ... add 0.5 mL canavanine (50 mg/L), 0.5 mL thialysine (50 mg/L), and "
    "1 mL G418 (200 mg/L)"
)

_YNB_NO_AA_NO_AS = _mixture(
    "yeast nitrogen base (w/o amino acids and ammonium sulfate)",
    MediaComponentRole.other,
    _DEFERRED,
    concentration=_c(1.7, _GL),
    provenance=[
        _sv(
            "1.7 g/L",
            "Add 1.7 g yeast nitrogen base without amino acids or "
            "ammonium sulfate (BD Difco)",
        )
    ],
    note="defined vitamin/salt mix (Difco); expand to per-component from the "
    "Difco YNB spec",
    defers_to=[_KUZMIN2016],
)
_MSG = _defined(
    "monosodium L-glutamate",
    MediaComponentRole.nitrogen_source,
    concentration=_c(1.0, _GL),
    provenance=[
        _sv("1 g/L", "1 g MSG (L-glutamic acid sodium salt hydrate; Sigma)"),
        _sv(
            "MSG replaces (NH4)2SO4",
            "MSG instead of ammonium sulfate is used as a nitrogen source in "
            "this medium, because the latter interferes with the activity of "
            "the antibiotic",
            ck=_KUZMIN2016,
            sha=_KUZMIN2016_SHA,
        ),
    ],
    note="N source; ammonium sulfate would antagonize G418 selection",
)

SD_MSG = Media(
    name="SD/MSG base (YNB w/o amino acids and ammonium sulfate + MSG; "
    "carbon source added per recipe)",
    state="liquid",
    is_synthetic=True,
    base_medium="SD_MSG",
    components=[_YNB_NO_AA_NO_AS, _MSG],
    provenance=[_sv("SD/MSG nitrogen base", _SGA_BASE_QUOTE)],
)
"""The SGA nitrogen base every Costanzo/Kuzmin record sits on.

It is a base, not a bench medium: like ``YP``, it names no carbon source, so a recipe
deriving from it states its own (glucose for the standard screens, galactose for the
Costanzo 2021 alternative-carbon condition).
"""

_SGA_GLUCOSE = _defined(
    "D-glucose",
    MediaComponentRole.carbon_source,
    concentration=_c(2.0, _PCT),
    provenance=[_sv("20 g/L", "add 50 mL 40% glucose")],
)
_SGA_AGAR = _defined(
    "agar",
    MediaComponentRole.gelling_agent,
    concentration=_c(2.0, _PCT),
    provenance=[_sv("20 g/L", "Add 20 g bacto agar")],
)
_G418 = _defined(
    "G418 (geneticin)",
    MediaComponentRole.selection_agent,
    concentration=_c(200.0, _UGML),
    provenance=[_sv("200 mg/L", "1 mL G418 (200 mg/L)")],
    note="selects the kanMX marker",
)


def _sga_components(carbon: MediaComponent) -> list[MediaComponent]:
    """Shared SD/MSG selection-medium components (Tong & Boone 2006 recipe #16)."""
    return [
        *SD_MSG.components,
        carbon,
        _mixture(
            "SC amino-acid supplement powder (DO -His/Arg/Lys)",
            MediaComponentRole.amino_acid,
            _DEFERRED,
            concentration=_c(2.0, _GL),
            provenance=[_sv("2 g DO powder/L", _SGA_SUPPLEMENT_QUOTE)],
            note="complete SC supplement MINUS the His/Arg/Lys dropout; full per-"
            "ingredient grams are in the provenance quote (3 g adenine, 2 g "
            "uracil, 2 g inositol, 0.2 g PABA, all 20 AAs @2 g except leucine "
            "@10 g, per 55.2 g mix, used at 2 g mix/L)",
        ),
        _SGA_AGAR,
        _defined(
            "L-canavanine",
            MediaComponentRole.selection_agent,
            concentration=_c(50.0, _UGML),
            provenance=[_sv("50 mg/L", "add 0.5 mL canavanine (50 mg/L)")],
            note="toxic L-arginine analog; selects can1-delta haploids",
        ),
        _defined(
            "thialysine (S-(2-aminoethyl)-L-cysteine)",
            MediaComponentRole.selection_agent,
            concentration=_c(50.0, _UGML),
            provenance=[_sv("50 mg/L", "0.5 mL thialysine (50 mg/L)")],
            note="toxic L-lysine analog; selects lyp1-delta haploids",
        ),
        _G418,
        _defined(
            "nourseothricin (clonNAT)",
            MediaComponentRole.selection_agent,
            concentration=_c(100.0, _UGML),
            provenance=[_sv("100 mg/L", "1 mL clonNAT (100 mg/L)")],
            note="selects the natMX marker",
        ),
    ]


_HIS = resolved_compound("L-histidine")
_ARG = resolved_compound("L-arginine")
_LYS = resolved_compound("L-lysine")
_URA = resolved_compound("uracil")

SGA_DM_SELECTION = Media(
    name="SGA double-mutant selection (SD-MSG, -His/Arg/Lys, +canavanine/thialysine/G418/clonNAT)",
    state="solid",
    is_synthetic=True,
    base_medium="SD_MSG",
    components=_sga_components(_SGA_GLUCOSE),
    dropouts=[_HIS, _ARG, _LYS],
    provenance=[_sv("SD/MSG -His/Arg/Lys selection medium", _SGA_BASE_QUOTE)],
)
"""Costanzo 2016 / Baryshnikova 2010 digenic SGA fitness-scoring medium."""

SGA_TM_SELECTION = Media(
    name="SGA triple-mutant selection (SD-MSG, -His/Arg/Lys/Ura, +canavanine/thialysine/G418/clonNAT)",
    state="solid",
    is_synthetic=True,
    base_medium="SD_MSG",
    components=_sga_components(_SGA_GLUCOSE),
    dropouts=[_HIS, _ARG, _LYS, _URA],
    provenance=[
        _sv(
            "SDMSG - His/Arg/Lys/Ura selection medium",
            "pinning the double/triple mutant haploid mix onto SD_MSG - "
            "His/Arg/Lys/Ura + canavanine/thialysine/G418/clonNAT to select "
            "for final triple mutants",
            ck="kuzminSystematicAnalysisComplex2018",
            sha="2ec80d05d823976e12add17699ad759bcd983768d2f53e7eb6c0185b963b8291",
        )
    ],
)
"""Kuzmin 2018 / 2020 trigenic SGA fitness-scoring medium (adds the Ura dropout for
the KlURA3-marked third mutation)."""

SGA_DM_SELECTION_GALACTOSE = Media(
    name="SGA double-mutant selection, 2% galactose "
    "(SD-MSG, -His/Arg/Lys, +canavanine/thialysine/G418/clonNAT)",
    state="solid",
    is_synthetic=True,
    base_medium="SD_MSG",
    components=_sga_components(
        _defined(
            "galactose",
            MediaComponentRole.carbon_source,
            concentration=_c(2.0, _PCT),
            provenance=[
                _sv(
                    "2% w/v galactose replacing the 2% glucose",
                    "We examined 14 diverse conditions, including an alternative "
                    "carbon source, osmotic stress, genotoxic stress, and 11 "
                    "bioactive compounds that target distinct yeast biological "
                    "processes",
                    ck=_COSTANZO2021,
                    sha=_COSTANZO2021_SHA,
                    uri="paper.md",
                    note="'an alternative carbon source' is a REPLACEMENT statement, "
                    "which is why galactose is a derived medium here rather than an "
                    "additive SmallMoleculePerturbation on a medium that still "
                    "contains glucose; the percentage is the SI condition sheet's "
                    "0.02 read as percent",
                )
            ],
        )
    ),
    dropouts=[_HIS, _ARG, _LYS],
    provenance=[_sv("SD/MSG -His/Arg/Lys selection medium", _SGA_BASE_QUOTE)],
)
"""Costanzo 2021's alternative-carbon condition: the SGA scoring medium with galactose
in place of glucose."""

# --------------------------------------------------------------------------- #
# YEPD / YPD family -- complex (NOT chemically defined). Tong & Boone 2006 #9.
#
# ``YPD`` is the ROOT: the three ingredients every member shares. ``YPD_LIQUID`` and
# ``YPD_AGAR`` differ from it only in state and in the agar row, and each names the
# paper that states its own difference, so a Bloom plate and a Hoepfner 24-well plate
# join at ``base_medium == "YPD"`` without either inheriting the other's bench quote.
# --------------------------------------------------------------------------- #
_YEPD_QUOTE = (
    "YEPD: Add 120 mg adenine (Sigma), 10 g yeast extract, 20 g "
    "peptone, 20 g bacto agar ... add 50 mL of 40% glucose solution"
)

_YPD_CORE = [
    _mixture(
        "yeast extract",
        MediaComponentRole.complex_ingredient,
        _UNDEFINED,
        concentration=_c(1.0, _PCT),
        provenance=[_sv("10 g/L", "10 g yeast extract")],
    ),
    _mixture(
        "peptone",
        MediaComponentRole.complex_ingredient,
        _UNDEFINED,
        concentration=_c(2.0, _PCT),
        provenance=[_sv("20 g/L", "20 g peptone")],
    ),
    _defined(
        "D-glucose",
        MediaComponentRole.carbon_source,
        concentration=_c(2.0, _PCT),
        provenance=[
            _sv("20 g/L (50 mL 40% glucose)", "add 50 mL of 40% glucose solution")
        ],
    ),
]

YPD = Media(
    name="YPD (yeast extract / peptone / dextrose)",
    state="solid",
    is_synthetic=False,
    base_medium="YPD",
    components=_YPD_CORE,
    provenance=[_sv("YEPD recipe", _YEPD_QUOTE)],
)
"""The YPD ROOT: the three ingredients every YPD member shares, and the join anchor
(``base_medium == "YPD"``). Loaders do not use it directly: a plate is ``YPD_AGAR``
(states its agar row) and a culture is ``YPD_LIQUID``; peptone and yeast extract are
intrinsically undefined."""

YPD_LIQUID = Media(
    name="YPD (yeast extract / peptone / dextrose), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="YPD",
    components=_YPD_CORE,
    provenance=[
        _sv("YEPD recipe", _YEPD_QUOTE),
        _sv(
            "1% yeast extract, 2% BactoPeptone, 2% glucose",
            "11-point serial dilutions (3.1 dilution factor) were prepared in 96 well "
            "plates with log phase growth yeast cultures (HIP pool) in YPD $2 \\%$ "
            "glucose, $2 \\%$ BactoPeptone, $1 \\%$ yeast extract)",
            ck=_HOEPFNER2014,
            sha=_HOEPFNER2014_SHA,
            uri="paper.md",
            note="the same three ingredients and the same percentages as the Tong and "
            "Boone YEPD recipe, stated independently by a liquid-culture screen",
        ),
        _sv(
            "liquid",
            "The HIP assay was performed in 24 well plates (Greiner 662102), with "
            "$1 6 0 0 \\mu \\mathrm { l } /$ well YPD.",
            ck=_HOEPFNER2014,
            sha=_HOEPFNER2014_SHA,
            uri="paper.md",
        ),
    ],
)
"""Liquid YPD: the object every pooled-culture screen on YPD should carry.

Hoepfner 2014 and Hillenmeyer 2008 carry it directly; since issue #622 the Ohya 2005,
Ohnuki 2018, Nadal-Ribelles 2025 and Yoshida 2012 loaders carry it through
:func:`restated` with their own paper's sentence appended. da Silveira 2014 prints a
different "YPD" recipe and carries a loader-local object instead.
"""

YPD_AGAR = Media(
    name="YPD (solid, 2% agar)",
    state="solid",
    is_synthetic=False,
    base_medium="YPD",
    components=[
        *_YPD_CORE,
        _defined(
            "agar",
            MediaComponentRole.gelling_agent,
            concentration=_c(2.0, _PCT),
            provenance=[
                _sv(
                    "20 g/L",
                    "Solid media were prepared by addition of $2 0 \\ \\mathrm { g / L }$ "
                    "agar (NZYTech, Lisbon, Portugal).",
                    ck=_MOTA2024,
                    sha=_MOTA2024_SHA,
                    uri="paper.md",
                )
            ],
        ),
    ],
    provenance=[
        _sv("YEPD recipe", _YEPD_QUOTE),
        _sv(
            "20 g/L glucose, 10 g/L yeast extract, 20 g/L peptone",
            "in liquid YPD medium containing, $2 0 ~ \\mathrm { g / L }$ glucose "
            "(Merck, Darmstadt, Germany), $1 0 ~ \\mathrm { g / L }$ yeast extract and "
            "$2 0 ~ \\mathrm { g / L }$ peptone, both from BD Biosciences (Franklin "
            "Lakes, NJ, USA)",
            ck=_MOTA2024,
            sha=_MOTA2024_SHA,
            uri="paper.md",
            note="the pH 4.5 this sentence goes on to state is NOT a property of the "
            "medium; it rides as EnvironmentPhysicalPerturbation(factor=ph)",
        ),
    ],
)
"""Solid YPD with the agar row stated: Mota 2024's spot-assay and CFU plates."""

YPAD = Media(
    name="YPAD (YPD + adenine)",
    state="solid",
    is_synthetic=False,
    base_medium="YPD",
    components=[
        *_YPD_CORE,
        _defined(
            "adenine",
            MediaComponentRole.nucleobase,
            concentration=_c(120.0, _UGML),  # 120 mg/L == 120 ug/mL
            provenance=[_sv("120 mg/L", "Add 120 mg adenine (Sigma) ... to ... 1 L")],
            note="adenine to suppress ade2 revertant pigment",
        ),
    ],
    provenance=YPD.provenance,
)
"""YPD supplemented with adenine (common for ade2 backgrounds)."""

# --------------------------------------------------------------------------- #
# Defined synthetic family (SD minimal / YNB / SC), inventory from the iBioFoundry
# Yeast9 FBA media_setup + standard Difco/Sigma formulations. Concentrations are
# left None where a single canonical wet-lab value is not yet sourced (open gap).
# --------------------------------------------------------------------------- #
_YNB_VITAMINS = [
    "biotin",
    "calcium pantothenate",
    "folic acid",
    "myo-inositol",
    "niacin",
    "4-aminobenzoic acid",
    "pyridoxine hydrochloride",
    "riboflavin",
    "thiamine hydrochloride",
]
_SC_AMINO_ACIDS = [
    "L-alanine",
    "L-arginine",
    "L-asparagine",
    "L-aspartic acid",
    "L-cysteine",
    "L-glutamic acid",
    "L-glutamine",
    "glycine",
    "L-histidine",
    "L-isoleucine",
    "L-leucine",
    "L-lysine",
    "L-methionine",
    "L-phenylalanine",
    "L-proline",
    "L-serine",
    "L-threonine",
    "L-tryptophan",
    "L-tyrosine",
    "L-valine",
]


def _named(names: list[str], role: MediaComponentRole) -> list[MediaComponent]:
    """Identified components whose wet-lab concentration is still an open gap."""
    return [
        _defined(
            n,
            role,
            note="identity resolved through the shared compound table; the wet-lab "
            "concentration is still pending a sourced enrichment pass",
        )
        for n in names
    ]


SD = Media(
    name="SD minimal (YNB + ammonium sulfate + glucose)",
    state="liquid",
    is_synthetic=True,
    base_medium="SD",
    components=[
        _defined(
            "D-glucose", MediaComponentRole.carbon_source, concentration=_c(2.0, _PCT)
        ),
        _defined(
            "ammonium sulfate",
            MediaComponentRole.nitrogen_source,
            concentration=_c(5.0, _GL),
        ),
        _mixture(
            "yeast nitrogen base (w/o amino acids)",
            MediaComponentRole.other,
            _DEFERRED,
            concentration=_c(1.7, _GL),
            note="Difco YNB: the 9 vitamins + trace metals + salts",
        ),
    ],
)
"""Synthetic minimal (defined) medium: glucose + ammonium sulfate + YNB."""

YNB = Media(
    name="YNB (yeast nitrogen base, defined vitamins + trace metals)",
    state="liquid",
    is_synthetic=True,
    base_medium="YNB",
    components=_named(_YNB_VITAMINS, MediaComponentRole.vitamin),
)
"""Defined YNB vitamin set (9 vitamins; Difco/Sigma). Trace-metal + salt rows +
concentrations pending the sourced enrichment pass."""

_SC_QUOTE = (
    "yeast cells were cultivated in synthetic complete medium (SC) "
    "$( 0 . 7 7 \\textrm { g L } ^ { - 1 }$ complete supplement mix drop out (CSM), "
    "$6 . 9 \\ \\mathrm { g \\ L ^ { - 1 } }$ yeast nitrogen base without amino acids "
    "$( \\mathrm { Y N B ~ w / o ~ }$ AA), $2 0 \\ \\mathrm { g \\ L ^ { - 1 } }$ "
    "glucose, $\\mathrm { p H } ~ 5 . 5$ , 4.5 or 3.5)"
)

SC = Media(
    name="SC (synthetic complete: YNB + 20 amino acids + uracil + adenine + glucose)",
    state="liquid",
    is_synthetic=True,
    base_medium="SC",
    components=[
        *YNB.components,
        *_named(_SC_AMINO_ACIDS, MediaComponentRole.amino_acid),
        _defined(
            "uracil", MediaComponentRole.nucleobase, note="SC nucleobase supplement"
        ),
        _defined(
            "adenine", MediaComponentRole.nucleobase, note="SC nucleobase supplement"
        ),
        _defined(
            "D-glucose",
            MediaComponentRole.carbon_source,
            concentration=_c(20.0, _GL),
            provenance=[
                _sv(
                    "20 g/L",
                    _SC_QUOTE,
                    ck=_MORMINO2022,
                    sha=_MORMINO2022_SHA,
                    uri="paper.md",
                ),
                _sv(
                    "2% glucose",
                    "All fungal species were grown and screened in synthetic complete "
                    "(SC) medium with $2 \\%$ glucose.",
                    ck=_WILDENHAIN2015,
                    sha=_WILDENHAIN2015_SHA,
                    uri="paper.md",
                    note="20 g/L and 2% w/v are the same amount; two independent "
                    "papers state the SC carbon source that the shipped constant had "
                    "been missing entirely",
                ),
            ],
            note="the carbon source SC had been missing: before this the medium "
            "resolved no carbon exchange at all under media_to_bounds",
        ),
    ],
    provenance=[
        _sv(
            "CSM 0.77 g/L + YNB w/o AA 6.9 g/L + glucose 20 g/L",
            _SC_QUOTE,
            ck=_MORMINO2022,
            sha=_MORMINO2022_SHA,
            uri="paper.md",
            note="the two gram figures are recorded at the medium level rather than as "
            "components because this library EXPANDS the CSM powder into the 20 amino "
            "acids plus uracil and adenine, and the YNB into its 9 vitamins; listing "
            "the powders as well would double-count them",
        )
    ],
)
"""Synthetic complete (defined): YNB + all 20 amino acids + uracil + adenine + glucose."""

SC_URA = dropout(
    SC,
    "uracil",
    name="SC-Ura (synthetic complete minus uracil; URA3 plasmid selection)",
)
"""SC with uracil dropped (selects a URA3-bearing plasmid).

Also the medium for Smith 2016's "SCM-Ura": Smith releases no SCM-Ura recipe, so the
shipped gaps are the honest state and no dataset-specific sibling is created.
"""

# --------------------------------------------------------------------------- #
# Hillenmeyer 2008 nutrient-dropout media. Table S1 classifies a named-nutrient
# dropout as an environmental change (a media swap), not a small molecule, so each is
# a DERIVED MEDIUM off SC rather than an EnvironmentPhysicalPerturbation on YPD --
# which is what lets a tryptophan dropout join SC, SC-Ura and the SGA media.
#
# The SOM states no base recipe; the base is SC because a named-nutrient dropout is
# only defined against a synthetic complete medium, and that inference is recorded on
# every object. The SOM also never states the reduced level of the three "partial"
# conditions, but the RELEASE does: the HOM score-matrix header carries it in the
# dose fields (``biotin partial drop-out:25:%``, ``calcium pantothenate partial
# drop-out:25:%``, ``pyridoxine HCl partial drop-out:12.5:%``; #505 E4). What the
# percentage is a percentage OF is defined nowhere, so the level rides in the
# component note and the medium's provenance verbatim, and the amount stays a typed
# ``reduced_from_standard`` (no ConcentrationUnit means "percent of the recipe level").
# --------------------------------------------------------------------------- #
_HILLENMEYER_DROPOUT_QUOTE = (
    "we restricted our analysis to small molecule experiments, excluding conditions "
    "of environmental change, such as amino acid dropout"
)
#: sha256 of the raw-mirror HOM score matrix whose header states the partial levels
#: (``$DATA_ROOT/torchcell-raw/hillenmeyerChemicalGenomicPortrait2008/data/``).
_HILLENMEYER2008_HOM_MATRIX_SHA = (
    "0b7d5e4dad0e5b4336b1dac97fb32ef14d7d0bf9f5d1b0d5ddb382c393362b71"
)
#: condition label -> the released partial level, verbatim from the matrix header's
#: ``conc1:unit1`` fields (every array of a label carries the same level).
HILLENMEYER_PARTIAL_DROPOUT_LEVELS: dict[str, tuple[str, str]] = {
    "biotin partial drop-out": ("25", "%"),
    "calcium pantothenate partial drop-out": ("25", "%"),
    "pyridoxine HCl partial drop-out": ("12.5", "%"),
}


def _hm_dropout(label: str, compound: str, *, partial: bool = False) -> Media:
    provenance = [
        _sv(
            f"{label} is an environmental change (a derived medium), not a compound",
            _HILLENMEYER_DROPOUT_QUOTE,
            ck=_HILLENMEYER2008,
            sha=_HILLENMEYER2008_SHA,
            uri="paper.md",
            note="the SOM states no recipe for the dropout series; SC is the defined "
            "medium a named-nutrient dropout is taken against, and the whole growth "
            "protocol is deferred to Pierce 2006, which is not mirrored",
        )
    ]
    if partial:
        value, unit = HILLENMEYER_PARTIAL_DROPOUT_LEVELS[label]
        token = f"{label}:{value}:{unit}"
        provenance.append(
            _sv(
                f"{value} {unit}",
                token,
                ck=_HILLENMEYER2008,
                sha=_HILLENMEYER2008_HOM_MATRIX_SHA,
                uri="data/hom.z_result_nm.pub",
                note="the released partial level, read from the HOM score-matrix "
                "header (raw mirror torchcell-raw/hillenmeyerChemicalGenomicPortrait2008"
                "/data/hom.z_result_nm.pub). The SOM does not define the basis of the "
                "percentage. Hypothesis (untested): percent of the standard SC level",
            )
        )
        return dropout(
            SC,
            name=f"SC, partial {compound} drop-out (Hillenmeyer 2008 '{label}')",
            partial=(compound,),
            provenance=provenance,
            partial_note=f"partial drop-out; the release states the level as "
            f"'{value} {unit}' (HOM matrix header '{token}') and never defines what it "
            "is a percentage of, so the amount is a typed reduction with the level "
            "recorded here and in the medium's provenance",
        )
    return dropout(
        SC,
        compound,
        name=f"SC - {compound} (Hillenmeyer 2008 '{label}')",
        provenance=provenance,
    )


#: (condition label in the HOM score matrix, the SC component it names, partial?).
#: "partial" is the source's own word for three vitamin conditions: the nutrient is
#: reduced, not removed. The SOM never states the reduced level; the release's matrix
#: header does (``HILLENMEYER_PARTIAL_DROPOUT_LEVELS``), without defining its basis.
_HILLENMEYER_DROPOUTS: tuple[tuple[str, str, bool], ...] = (
    ("adenine dropout", "adenine", False),
    ("arginine dropout", "L-arginine", False),
    ("isoleucine dropout", "L-isoleucine", False),
    ("lysine dropout", "L-lysine", False),
    ("threonine dropout", "L-threonine", False),
    ("tryptophan dropout", "L-tryptophan", False),
    ("tyrosine dropout", "L-tyrosine", False),
    ("PABA drop-out", "4-aminobenzoic acid", False),
    ("folic acid drop-out", "folic acid", False),
    ("inositol drop-out", "myo-inositol", False),
    ("niacin drop-out", "niacin", False),
    ("thiamine HCl drop-out", "thiamine hydrochloride", False),
    ("biotin partial drop-out", "biotin", True),
    ("calcium pantothenate partial drop-out", "calcium pantothenate", True),
    ("pyridoxine HCl partial drop-out", "pyridoxine hydrochloride", True),
)


def _hm_key(compound: str, partial: bool) -> str:
    """MEDIA_LIBRARY key for one Hillenmeyer dropout medium."""
    stem = resolved_compound(compound).name.upper().replace("-", "_").replace(" ", "_")
    return f"SC_{'PARTIAL' if partial else 'MINUS'}_{stem}"


#: Hillenmeyer 2008 HOM condition label -> the medium it denotes. The control label is
#: not a dropout at all: it is plain SC, which is why it maps to the shared constant.
HILLENMEYER_DROPOUT_MEDIA: dict[str, Media] = {
    label: _hm_dropout(label, compound, partial=partial)
    for label, compound, partial in _HILLENMEYER_DROPOUTS
} | {"vitamin drop-out control media": SC}


# --------------------------------------------------------------------------- #
# Vanacloig-Pedros 2022: a defined hydrolysate-mimicking base whose composition the
# primary defers to Zhang 2019 (not mirrored), plus the modifications this paper made.
# The pH 5.0 the same sentence states is NOT a medium field: it rides as
# EnvironmentPhysicalPerturbation(factor=ph).
# --------------------------------------------------------------------------- #
_SYNBASE_QUOTE = (
    "yeast strains were grown in a modified version of synthetic "
    "${ \\mathrm { S y n H } } 3 ^ { - }$ medium (‘SynBase’ medium) described in "
    "(Zhang et al. 2019). SynBase medium used in this study was prepared identically "
    "as ${ \\mathrm { S y n H 3 ^ { - } } }$ except for the following changes: "
    "acetamide, sodium acetate, and cellobiose were not included, and ammonium "
    "sulfate was replaced with ${ \\mathrm { ~ 1 ~ g / L ~ } }$ monosodium glutamate "
    "(MSG, Fisher Scientific) and adjusted to $\\mathrm { p H } ~ 5 . 0$ with HCl."
)

SYNH3_MINUS = Media(
    name="SynH3- (defined synthetic hydrolysate base)",
    state="liquid",
    is_synthetic=True,
    base_medium="SYNH3_MINUS",
    components=[
        _mixture(
            "SynH3- defined hydrolysate base",
            MediaComponentRole.other,
            _DEFERRED,
            provenance=[
                _sv(
                    "composition deferred to Zhang 2019",
                    _SYNBASE_QUOTE,
                    ck=_VANACLOIG2022,
                    sha=_VANACLOIG2022_SHA,
                    uri="paper.md",
                )
            ],
            note="the sugars, salts and amino acids of SynH3- are specified in Zhang "
            "et al. 2019, Front Microbiol 10:2596, which is not mirrored; the carbon "
            "source therefore sits inside this deferred line",
            defers_to=[_ZHANG2019],
        )
    ],
    provenance=[
        _sv(
            "SynH3- base",
            _SYNBASE_QUOTE,
            ck=_VANACLOIG2022,
            sha=_VANACLOIG2022_SHA,
            uri="paper.md",
        )
    ],
)
"""The Vanacloig-Pedros 2022 base, composition deferred to its unmirrored originator."""

SYNBASE = Media(
    name="SynBase (SynH3- minus acetamide/sodium acetate/cellobiose, MSG for "
    "ammonium sulfate)",
    state="liquid",
    is_synthetic=True,
    base_medium="SYNH3_MINUS",
    components=[
        *SYNH3_MINUS.components,
        _defined(
            "monosodium L-glutamate",
            MediaComponentRole.nitrogen_source,
            concentration=_c(1.0, _GL),
            provenance=[
                _sv(
                    "1 g/L, replacing ammonium sulfate",
                    _SYNBASE_QUOTE,
                    ck=_VANACLOIG2022,
                    sha=_VANACLOIG2022_SHA,
                    uri="paper.md",
                )
            ],
            note="same compound and same amount as the SGA SD/MSG nitrogen source, "
            "and for the same reason: ammonium sulfate blocks antibiotic selection",
        ),
    ],
    dropouts=[
        resolved_compound("acetamide"),
        resolved_compound("sodium acetate"),
        resolved_compound("cellobiose"),
        # "ammonium sulfate was replaced with 1 g/L monosodium glutamate": the SynH3-
        # nitrogen source is omitted, which the MSG component alone does not say (#501).
        resolved_compound("ammonium sulfate"),
    ],
    provenance=[
        _sv(
            "SynH3- with three omissions and MSG for ammonium sulfate",
            _SYNBASE_QUOTE,
            ck=_VANACLOIG2022,
            sha=_VANACLOIG2022_SHA,
            uri="paper.md",
            note="the pH 5.0 stated in the same sentence is a typed environment "
            "perturbation, not a Media field",
        )
    ],
)
"""Vanacloig-Pedros 2022's chemical-genomics medium."""

# --------------------------------------------------------------------------- #
# Smith 2006 fatty-acid plates. One Methods sentence gives all three recipes; the two
# buffered YNB plates share a base, which is what makes myristate and acetate
# comparable to each other and separable from the peptone-based oleate plate.
# --------------------------------------------------------------------------- #
_SMITH_MEDIA_QUOTE = (
    "Omnitrays contained $4 0 \\mathrm { m l }$ of YPBA agar $( 0 . 6 7 \\%$ yeast "
    "nitrogen base, $0 . 1 \\%$ yeast extract, $0 . 5 \\%$ potassium phosphate "
    "buffer, pH 6.0, $2 \\%$ agar, $2 \\%$ acetate), $2 0 \\mathrm { m l }$ of YPBO "
    "agar $( 0 . 3 \\%$ yeast extract, $0 . 5 \\%$ potassium phosphate buffer, "
    "$\\mathrm { p H } 6 . 0$ $0 . { \\bar { 5 } } \\%$ peptone, $0 . 2 \\%$ Tween "
    "40, $2 \\%$ agar, $0 . 1 \\%$ oleic acid) or $2 0 \\mathrm { m l }$ of YPBM agar "
    "$( 0 . 6 7 \\%$ yeast nitrogen base, $0 . 1 \\%$ yeast extract, $0 . 5 \\%$ "
    "potassium phosphate buffer, $\\mathrm { p H } 6 . 0$ $2 \\%$ agar, $0 . 5 \\%$ "
    "Tween 40, $0 . 1 2 5 \\%$ myristic acid)."
)


def _smith_sv(value: object, *, note: str | None = None) -> SourcedValue:
    return _sv(
        value,
        _SMITH_MEDIA_QUOTE,
        ck=_SMITH2006,
        sha=_SMITH2006_SHA,
        uri="paper.md",
        note=note,
    )


def _k_phosphate(percent: float) -> MediaComponent:
    return _mixture(
        "potassium phosphate buffer",
        MediaComponentRole.buffer,
        _DEFERRED,
        concentration=_c(percent, _PCT),
        provenance=[_smith_sv(f"{percent:g}% at pH 6.0")],
        note="a KH2PO4 / K2HPO4 pair; the source gives the buffer's pH (6.0) but not "
        "the ratio of the two salts, so the composition stays deferred. "
        + _PERCENT_BASIS_NOTE,
    )


def _tween40(percent: float) -> MediaComponent:
    return _mixture(
        "Tween 40",
        MediaComponentRole.other,
        _DEFERRED,
        concentration=_c(percent, _PCT),
        provenance=[_smith_sv(f"{percent:g}%")],
        note="polysorbate 40 is a polydisperse ethoxylated sorbitan ester, so no "
        "single InChIKey names it; it emulsifies the fatty acid. "
        + _PERCENT_BASIS_NOTE,
    )


def _smith_agar() -> MediaComponent:
    return _defined(
        "agar",
        MediaComponentRole.gelling_agent,
        concentration=_c(2.0, _PCT),
        provenance=[_smith_sv("2%")],
    )


YPB = Media(
    name="YPB (yeast extract / peptone / potassium phosphate buffer, pH 6.0)",
    state="solid",
    is_synthetic=False,
    base_medium="YPB",
    components=[
        _mixture(
            "yeast extract",
            MediaComponentRole.complex_ingredient,
            _UNDEFINED,
            concentration=_c(0.3, _PCT),
            provenance=[_smith_sv("0.3%")],
        ),
        _mixture(
            "peptone",
            MediaComponentRole.complex_ingredient,
            _UNDEFINED,
            concentration=_c(0.5, _PCT),
            provenance=[_smith_sv("0.5%")],
        ),
        _k_phosphate(0.5),
    ],
    provenance=[_smith_sv("YPBO base")],
)
"""The buffered peptone base of Smith 2006's oleate plate; no carbon source of its own."""

YPBO = Media(
    name="YPBO (YPB + 0.2% Tween 40 + 0.1% oleic acid, solid)",
    state="solid",
    is_synthetic=False,
    base_medium="YPB",
    components=[
        *YPB.components,
        _tween40(0.2),
        _smith_agar(),
        _defined(
            "oleic acid",
            MediaComponentRole.carbon_source,
            concentration=_c(0.1, _PCT),
            provenance=[_smith_sv("0.1%")],
            note=_PERCENT_BASIS_NOTE,
        ),
    ],
    provenance=[_smith_sv("YPBO agar")],
)
"""Smith 2006's oleate clear-zone plate."""

YNB_YE_B = Media(
    name="YNB-YE-B (YNB w/o amino acids + yeast extract + potassium phosphate "
    "buffer, pH 6.0)",
    state="solid",
    is_synthetic=False,
    base_medium="YNB_YE_B",
    components=[
        _mixture(
            "yeast nitrogen base (w/o amino acids)",
            MediaComponentRole.other,
            _DEFERRED,
            concentration=_c(0.67, _PCT),
            provenance=[_smith_sv("0.67%")],
        ),
        _mixture(
            "yeast extract",
            MediaComponentRole.complex_ingredient,
            _UNDEFINED,
            concentration=_c(0.1, _PCT),
            provenance=[_smith_sv("0.1%")],
        ),
        _k_phosphate(0.5),
    ],
    provenance=[_smith_sv("YPBM / YPBA base")],
)
"""The buffered YNB base shared by Smith 2006's myristate and acetate plates."""

YPBM = Media(
    name="YPBM (YNB-YE-B + 0.5% Tween 40 + 0.125% myristic acid, solid)",
    state="solid",
    is_synthetic=False,
    base_medium="YNB_YE_B",
    components=[
        *YNB_YE_B.components,
        _smith_agar(),
        _tween40(0.5),
        _defined(
            "myristic acid",
            MediaComponentRole.carbon_source,
            concentration=_c(0.125, _PCT),
            provenance=[_smith_sv("0.125%")],
            note=_PERCENT_BASIS_NOTE,
        ),
    ],
    provenance=[_smith_sv("YPBM agar")],
)
"""Smith 2006's myristate clear-zone plate."""

YPBA = Media(
    name="YPBA (YNB-YE-B + 2% acetate, solid)",
    state="solid",
    is_synthetic=False,
    base_medium="YNB_YE_B",
    components=[
        *YNB_YE_B.components,
        _smith_agar(),
        _defined(
            "acetate",
            MediaComponentRole.carbon_source,
            concentration=_c(2.0, _PCT),
            provenance=[_smith_sv("2%")],
            note="the shared table's canonical name for the bench 'acetate' is acetic "
            "acid, the conjugate pair's neutral form. " + _PERCENT_BASIS_NOTE,
        ),
    ],
    provenance=[_smith_sv("YPBA agar")],
)
"""Smith 2006's acetate growth plate."""

# --------------------------------------------------------------------------- #
# Lian 2019 SED-URA. One Methods sentence gives the recipe and the G418 supplement.
# --------------------------------------------------------------------------- #
_LIAN_MEDIA_QUOTE = (
    "Yeast strains were cultivated in complex medium consisting of $2 \\%$ peptone, "
    "$1 \\%$ yeast extract, and $2 \\%$ glucose (YPD) or synthetic complete medium "
    "consisting of $0 . 1 7 \\%$ yeast nitrogen base, $0 . 1 \\%$ mono-sodium "
    "glutamate, $0 . 0 7 7 \\%$ CSM-URA, and $2 \\%$ glucose (SED-URA) at "
    "$3 0 ^ { \\circ } \\mathrm { C } ,$ . $2 5 0 \\mathrm { r p m }$ . When "
    "necessary, $2 0 0 \\mu \\mathrm { g } \\mathrm { m L } ^ { - 1 }$ G418 (KSE "
    "Scientific, Durham, NC, USA) was supplemented."
)


def _lian_sv(value: object, *, note: str | None = None) -> SourcedValue:
    return _sv(
        value,
        _LIAN_MEDIA_QUOTE,
        ck=_LIAN2019,
        sha=_LIAN2019_SHA,
        uri="paper.md",
        note=note,
    )


SED_URA = Media(
    name="SED-URA (YNB + monosodium glutamate + CSM-URA + 2% glucose)",
    state="liquid",
    is_synthetic=True,
    base_medium="SED_URA",
    components=[
        _mixture(
            "yeast nitrogen base (w/o amino acids)",
            MediaComponentRole.other,
            _DEFERRED,
            concentration=_c(0.17, _PCT),
            provenance=[_lian_sv("0.17%")],
            note="0.17% w/v is 1.7 g/L, the same amount as the SGA SD/MSG line; the "
            "paper does not state whether the YNB is also ammonium-sulfate free, so "
            "this medium is its own base rather than a derivative of SD_MSG",
        ),
        _defined(
            "monosodium L-glutamate",
            MediaComponentRole.nitrogen_source,
            concentration=_c(0.1, _PCT),
            provenance=[_lian_sv("0.1% mono-sodium glutamate")],
        ),
        _mixture(
            "CSM-URA (complete supplement mixture minus uracil)",
            MediaComponentRole.amino_acid,
            _DEFERRED,
            concentration=_c(0.077, _PCT),
            provenance=[_lian_sv("0.077%")],
            note="commercial drop-out powder; expand from the vendor's CSM spec",
        ),
        _defined(
            "D-glucose",
            MediaComponentRole.carbon_source,
            concentration=_c(2.0, _PCT),
            provenance=[_lian_sv("2%")],
        ),
    ],
    dropouts=[_URA],
    provenance=[_lian_sv("SED-URA recipe")],
)
"""Lian 2019's URA3-selective medium for the MAGIC CRISPR-AID library."""

SED_URA_G418 = Media(
    name="SED-URA + 200 ug/mL G418",
    state="liquid",
    is_synthetic=True,
    base_medium="SED_URA",
    components=[
        *SED_URA.components,
        _defined(
            "G418 (geneticin)",
            MediaComponentRole.selection_agent,
            concentration=_c(200.0, _UGML),
            provenance=[_lian_sv("200 ug/mL G418")],
            note="same compound spelling as the SGA selection media, so the two "
            "G418-selected families share one selection-agent identity",
        ),
    ],
    dropouts=[_URA],
    provenance=[_lian_sv("SED-URA with G418 selection")],
)
"""SED-URA with the kanMX selection agent added."""

# --------------------------------------------------------------------------- #
# Bloom 2019 segregant-panel media (eLife 8:e49212). The assay plates are solid
# agar; the compound conditions sit on plain YPD (above), the carbon-source
# conditions replace glucose in YP with 2% of another sugar, and the YNB plates
# carry 2% glucose. Every concentration is quoted from the paper's Figure 1
# source data 1 (the ``Phenotypes`` sheet of ``elife-49212-fig1-data1-v2.xls``),
# deposited in the raw mirror under ``bloomRareVariantsContribute2019/data/``;
# the xls is binary, so the quote is the row's cell strings and the audit is
# structural (re-read the sheet), not a substring search.
# --------------------------------------------------------------------------- #
_BLOOM2019 = "bloomRareVariantsContribute2019"
_BLOOM2019_XLS = "data/elife-49212-fig1-data1-v2.xls"
_BLOOM2019_XLS_SHA = "990e75168a77522b9b684b8d0151e45c24c75ccbe7c360d18c500008cfdad8eb"


def _bloom_sv(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A SourcedValue anchored to the Bloom 2019 Figure 1 source data 1 xls."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=_BLOOM2019_XLS,
            citation_key=_BLOOM2019,
            sha256=_BLOOM2019_XLS_SHA,
        ),
        quote=quote,
        note=note,
    )


_YP_COMPONENTS = [c for c in YPD.components if c.compound.name != "D-glucose"]

YP = Media(
    name="YP (yeast extract / peptone, no added carbon source)",
    state="solid",
    is_synthetic=False,
    base_medium="YP",
    components=_YP_COMPONENTS,
    provenance=[
        _bloom_sv(
            "YP base",
            "Add 2% following (instead of Glucose)",
            note="the Phenotypes sheet's 'Carbon Sources' block; every carbon-source "
            "row in it reads Media = YP with the named sugar replacing glucose",
        )
    ],
)
"""YP base with no carbon source; a real medium only once a sugar is added."""


def _yp_plus(
    sugar: str, *, percent: float, quote: str, name: str | None = None
) -> Media:
    """YP + one carbon source at ``percent`` % (w/v) on the Bloom 2019 assay plates.

    The bench name is the compound name; the metabolism resolver's synonym table maps
    it to the model's form (``galactose`` -> ``D-galactose``, ``lactate`` ->
    ``(S)-lactate``), so a medium never has to know a model's naming.
    """
    compound = resolved_compound(sugar)
    return Media(
        name=name or f"YP + {percent:g}% {sugar}",
        state="solid",
        is_synthetic=False,
        base_medium="YP",
        components=[
            *_YP_COMPONENTS,
            MediaComponent(
                compound=compound,
                role=MediaComponentRole.carbon_source,
                concentration=_c(percent, _PCT),
                provenance=[_bloom_sv(f"{percent:g}%", quote)],
            ),
        ],
        provenance=YP.provenance,
    )


YP_FRUCTOSE = _yp_plus("fructose", percent=2.0, quote="Fructose | 20 | % | H2O | YP")
YP_GALACTOSE = _yp_plus("galactose", percent=2.0, quote="Galactose | 20 | % | H2O | YP")
YP_LACTATE = _yp_plus(
    "lactate", percent=2.0, quote="Lactate | 20 | % | H2O, pH = 6 | YP"
)
YP_MALTOSE = _yp_plus("maltose", percent=2.0, quote="Maltose | 20 | % | H2O | YP")
YP_MANNOSE = _yp_plus("mannose", percent=2.0, quote="Mannose | 20 | % | H2O | YP")
YP_RAFFINOSE = _yp_plus("raffinose", percent=2.0, quote="Raffinose | 20 | % | H2O | YP")
YP_SUCROSE = _yp_plus("sucrose", percent=2.0, quote="Sucrose | 20 | % | H2O | YP")
YP_TREHALOSE = _yp_plus("trehalose", percent=2.0, quote="Trehalose | 20 | % | H2O | YP")
YP_XYLOSE = _yp_plus("xylose", percent=2.0, quote="Xylose | 20 | % | H2O | YP")
YP_GLYCEROL = _yp_plus(
    "glycerol",
    percent=3.0,
    quote="Glycerol 3% | 40 | % | H2O | 0.03 | 0.03 | 3 | 500 | 37500 | 37500 | 3500 | 262500 | 262.5 | 2888 | YP",
)
YP_ETHANOL = _yp_plus(
    "ethanol",
    percent=2.0,
    quote="Ethanol NO glucose | 100 | % | H2O | 0.02 | 0.08 | 8 | 0.25",
    name="YP + 2% ethanol (no glucose)",
)

YP_GLYCEROL_LIQUID = Media(
    name="YP + glycerol, liquid (Hillenmeyer 2008 'YP glycerol'; percentage not stated)",
    state="liquid",
    is_synthetic=False,
    base_medium="YP",
    components=[
        *_YP_COMPONENTS,
        _defined(
            "glycerol",
            MediaComponentRole.carbon_source,
            provenance=[
                _sv(
                    "glycerol replaces glucose; percentage not stated",
                    "media change YP glycerol, minimal media, sorbitol, "
                    "synthetic complete",
                    ck=_HILLENMEYER2008,
                    sha=_HILLENMEYER2008_SHA,
                    uri="paper.md",
                    note="SOM Table S1 classifies 'YP glycerol' as a media change, "
                    "not a small molecule, which is why it is a derived medium and "
                    "not a carbon_source perturbation on YPD",
                )
            ],
            note="concentration is an OPEN GAP: the Hillenmeyer SOM never states a "
            "glycerol percentage and defers the growth protocol to Pierce 2006, which "
            "is not mirrored. Bloom 2019's 3% is that paper's bench value on that "
            "paper's plates and must not be copied here",
            defers_to=[_PIERCE2006],
        ),
    ],
    provenance=[
        _sv(
            "the whole growth protocol is deferred to Pierce 2006",
            "The protocol for pooled, competitive growth of the deletion strains, "
            "genomic DNA purification and PCR, and tag hybridization follows Ref. (2).",
            ck=_HILLENMEYER2008,
            sha=_HILLENMEYER2008_SHA,
            uri="paper.md",
        )
    ],
)
"""Hillenmeyer 2008's liquid YP + glycerol pool. Joins Bloom 2019's solid ``YP_GLYCEROL``
at ``base_medium == "YP"`` and at the glycerol compound, without borrowing its 3%."""

YPD_ETHANOL = Media(
    name="YPD + 2% ethanol (2% glucose + 2% ethanol)",
    state="solid",
    is_synthetic=False,
    base_medium="YPD",
    components=[
        *YPD.components,
        MediaComponent(
            compound=resolved_compound("ethanol"),
            role=MediaComponentRole.carbon_source,
            concentration=_c(2.0, _PCT),
            provenance=[
                _bloom_sv(
                    "2%",
                    "Ethanol with Glucose | 100 | % | H2O | 0.02 | 0.08 | 8 | 0.25 | 500 | "
                    "40000 | 10000 | 3500 | 70000 | 70 | 3080 | 4 | YP | 2% glucose",
                )
            ],
        ),
    ],
    provenance=YPD.provenance,
)
"""YPD with 2% ethanol added on top of the 2% glucose (Bloom 2019 ``EtOH_Glucose``)."""

YNB_GLUCOSE_SOLID = Media(
    name="YNB + 2% glucose (solid; Bloom 2019 minimal-medium plates)",
    state="solid",
    is_synthetic=True,
    base_medium="YNB",
    components=[
        *YNB.components,
        _defined(
            "D-glucose",
            MediaComponentRole.carbon_source,
            concentration=_c(2.0, _PCT),
            provenance=[
                _bloom_sv(
                    "2%",
                    "YNB | 20 | % | H2O | 0.02 | 2 | % | 500 | 3500 | YNB | 2% glucose",
                )
            ],
        ),
    ],
    provenance=[
        _bloom_sv(
            "YNB + 2% glucose",
            "YNB | 20 | % | H2O | 0.02 | 2 | % | 500 | 3500 | YNB | 2% glucose",
            note="the pH 3 and pH 8 rows also read 'YNB | 2% glucose'; the nitrogen "
            "source and the YNB trace-metal and salt rows are not stated in the "
            "sheet and remain the shipped YNB's open gaps",
        )
    ],
)
"""Solid YNB + 2% glucose as pinned for the Bloom 2019 YNB / pH 3 / pH 8 plates."""

# --------------------------------------------------------------------------- #
# SM (synthetic minimal) -- the Ralser-lab prototrophic-collection medium.
#
# "Synthetic minimal" is not one recipe across papers, which is why these are three
# objects rather than one ``Media(name="SM")`` stub. Two of the three are the SAME
# formulation, stated independently: Mulleder 2016 writes it out, and Messner 2023
# both defers to Mulleder ("grown as previously published15", ref 15 = Mulleder 2016
# Cell) AND restates the identical 6.7 g/L YNB + 2% glucose + 2% agar line. So they
# share the object, and the agar plate is the derived medium of the liquid culture.
#
# Zelezniak 2018 is the third case and it is NOT the same object: that paper states no
# recipe at all. It names the medium ("minimal medium", "synthetic minimal (SM)") and
# sources the strains to the same prototrophic collection, deferring to Mulleder 2012
# (Nat Biotechnol 30:1176-1178), which is not mirrored. Copying Mulleder 2016's grams
# onto it would be a guess dressed as provenance, exactly what ``YP_GLYCEROL_LIQUID``
# refuses to do with Bloom's 3% glycerol, so the composition stays deferred.
#
# Documented gaps, all three: no pH is stated by any of the three papers; the YNB
# trace-metal, salt and vitamin rows sit inside the commercial YNB line; and the
# ammonium sulfate amount is recorded as an identity without a number (see below).
# --------------------------------------------------------------------------- #
_MULLEDER2016 = "mullederFunctionalMetabolomicsDescribes2016"
_MULLEDER2016_SHA = "20412bec5b930d1fa43d326d8f9baca267130ddef5fdaa804f19e1209f925e6e"
_MESSNER2023 = "messnerProteomicLandscapeGenomewide2023"
_MESSNER2023_SHA = "edd0fe288641b9c4775f8972f2a13bcf09dfbf3bcd98d3e444e548d5cf4b3c15"
_ZELEZNIAK2018 = "zelezniakMachineLearningPredicts2018"
_ZELEZNIAK2018_SHA = "072bfb2d5b601d578dd5370ed25bfe2709bba0273843c87b5580e248573ce196"

#: Zelezniak 2018's deferral target, and NOT mirrored: the prototrophic-collection
#: paper its "Strains and Culture" section sources the strains and the medium to.
_MULLEDER2012 = "mullederPrototrophicDeletionMutant2012"

_MULLEDER_SM_QUOTE = (
    "The strains were transferred to synthetic minimal (SM) agar medium "
    "$\\left( 6 . 7 ~ \\mathfrak { g } / \\right.$ yeast nitrogen base without amino "
    "acids (Y0626, SIGMA), $2 \\%$ glucose, $2 \\%$ agar)"
)
_MULLEDER_LIQUID_QUOTE = (
    "These spots were used for the inoculation of cultures in liquid SM"
)
#: The nitrogen-starvation medium of the SAME Methods section. It is quoted here because
#: it is what establishes, from the paper's own words, that the SM's YNB carries the
#: ammonium sulfate: the nitrogen-FREE medium is a DIFFERENT Sigma product, named "without
#: amino acids and ammonium sulfate", at 1.7 g/L against SM's 6.7 g/L.
_MULLEDER_SDN_QUOTE = (
    "Starvation medium, SD $( - N )$ , was prepared with $1 . 7 ~ { \\mathfrak { g } } / "
    "{ \\mathfrak { l } }$ yeast nitrogen base without amino acids and ammonium sulfate "
    "(Y1251 SIGMA) and $2 \\%$ glucose."
)
_MESSNER_SM_QUOTE = (
    "The thawed stock cultures were spotted with the pinning robot onto SM agar medium "
    "$( 6 . 7 \\ : \\mathfrak { g } / |$ yeast nitrogen base without amino acids, "
    "$2 \\%$ glucose, $2 \\%$ agar)"
)
_MESSNER_LIQUID_QUOTE = (
    "Subsequently, these cells were used for inoculation in "
    "$2 0 0 \\mu \\ S \\mathsf { M }$ liquid medium in 96-well plates"
)
_MESSNER_NO_SUPPLEMENT_QUOTE = (
    "We grew a prototrophic derivative of the yeast gene deletion collection in a "
    "synthetic minimal (SM) medium without amino acid and nucleobase supplementation"
)
_MESSNER_DEFERRAL_QUOTE = (
    "The yeast strains were grown as previously published15 with slight modifications."
)
_ZELEZNIAK_STRAIN_QUOTE = (
    "Yeast strains used in this study were obtained from our published prototrophic "
    "gene deletion collection (Mulleder et al., 2012 € )."
)
_ZELEZNIAK_CULTURE_QUOTE = (
    "97 of the strains grew in triplicates $\\scriptstyle ( \\mathsf { n } = 3 )$ in "
    "minimal medium without a substantial growth defect (Figure S1), were pre-cultured "
    "overnight in $1 0 ~ \\mathsf { m l }$ minimal medium, at "
    "$_ { 3 0 ^ { \\circ } \\mathrm { C } }$ ,"
)


def _mulleder_sv(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    return _sv(
        value, quote, ck=_MULLEDER2016, sha=_MULLEDER2016_SHA, uri="paper.md", note=note
    )


def _messner_sv(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    return _sv(
        value, quote, ck=_MESSNER2023, sha=_MESSNER2023_SHA, uri="paper.md", note=note
    )


_SM_YNB = _mixture(
    "yeast nitrogen base (w/o amino acids)",
    MediaComponentRole.other,
    _DEFERRED,
    concentration=_c(6.7, _GL),
    provenance=[
        _mulleder_sv("6.7 g/L", _MULLEDER_SM_QUOTE),
        _messner_sv(
            "6.7 g/L",
            _MESSNER_SM_QUOTE,
            note="the same product and the same amount, stated independently seven "
            "years later; Messner's Key Resources Table prints the catalog number as "
            "Cat#Y0262 against Mulleder's Y0626, a two-digit transposition of the same "
            "Sigma YNB-without-amino-acids product, recorded as read and not corrected",
        ),
    ],
    note="6.7 g/L is the full-strength commercial YNB-without-amino-acids line, four "
    "times the 1.7 g/L amino-acid- AND ammonium-sulfate-free product the same Methods "
    "section uses for its nitrogen-starvation medium; the vitamins, trace metals and "
    "salts sit inside it (expand from the Sigma/Difco YNB spec)",
    defers_to=[_MULLEDER2012],
)
_SM_AMMONIUM_SULFATE = _defined(
    "ammonium sulfate",
    MediaComponentRole.nitrogen_source,
    provenance=[
        _mulleder_sv(
            "present inside the 6.7 g/L YNB line; amount not printed",
            _MULLEDER_SDN_QUOTE,
            note="the nitrogen source is recorded as an IDENTITY with no concentration, "
            "on purpose. Neither paper prints an ammonium sulfate amount, and its mass "
            "is already counted in the 6.7 g/L YNB line, so giving it a number here "
            "would double-count the medium. That it is present at all is the paper's "
            "own statement, not an assumption: this sentence defines the "
            "nitrogen-starvation medium as a DIFFERENT product, 'without amino acids "
            "and ammonium sulfate' at 1.7 g/L, against SM's 6.7 g/L 'without amino "
            "acids'. The 5.0 g/L difference between the two lines is the standard "
            "ammonium sulfate content of the full-strength product, which is "
            "corroboration and not a sourced number, so it is not recorded as one",
        )
    ],
    note="the only nitrogen source in SM; a prototrophic collection grows on it with "
    "no amino acid or nucleobase supplement",
)
_SM_GLUCOSE = _defined(
    "D-glucose",
    MediaComponentRole.carbon_source,
    concentration=_c(2.0, _PCT),
    provenance=[
        _mulleder_sv("2% (20 g/L)", _MULLEDER_SM_QUOTE, note=_PERCENT_BASIS_NOTE),
        _messner_sv("2% (20 g/L)", _MESSNER_SM_QUOTE),
    ],
)

SM = Media(
    name="SM (synthetic minimal: 6.7 g/L YNB without amino acids + 2% glucose), liquid",
    state="liquid",
    is_synthetic=True,
    base_medium="SM",
    components=[_SM_YNB, _SM_AMMONIUM_SULFATE, _SM_GLUCOSE],
    provenance=[
        _mulleder_sv(
            "6.7 g/L YNB w/o amino acids + 2% glucose, no agar",
            _MULLEDER_LIQUID_QUOTE,
            note="the recipe sentence gives 'SM agar medium (... 2% agar)', so the "
            "liquid SM this sentence inoculates is that recipe without the agar; both "
            "papers name the two media 'SM agar' and 'SM liquid' off one formulation",
        ),
        _messner_sv("SM liquid medium", _MESSNER_LIQUID_QUOTE),
        _messner_sv(
            "no amino acid or nucleobase supplement",
            _MESSNER_NO_SUPPLEMENT_QUOTE,
            note="why ``dropouts`` is empty rather than listing the 20 amino acids: SM "
            "is a minimal medium, not an edit of a supplemented one, so there is no "
            "base medium the supplements were removed FROM",
        ),
        _messner_sv(
            "Messner's growth protocol defers to Mulleder 2016",
            _MESSNER_DEFERRAL_QUOTE,
            note="reference 15 of Messner 2023 is Mulleder et al. 2016 Cell 167:553, "
            "the mirrored key this object's other quotes come from, so the deferral "
            "chain closes inside the mirror instead of leaving the mirror",
        ),
    ],
)
"""Liquid SM: the medium the Mulleder 2016 and Messner 2023 cultures were grown in.

Both papers state the same formulation independently, and Messner's protocol also
defers to Mulleder, so one object carries both. ``open_gaps`` reports the commercial
YNB line (composition deferred) and the ammonium sulfate amount; no paper states a pH.
"""

SM_AGAR = Media(
    name="SM agar (synthetic minimal: 6.7 g/L YNB without amino acids + 2% glucose "
    "+ 2% agar)",
    state="solid",
    is_synthetic=True,
    base_medium="SM",
    components=[
        *SM.components,
        _defined(
            "agar",
            MediaComponentRole.gelling_agent,
            concentration=_c(2.0, _PCT),
            provenance=[
                _mulleder_sv("2%", _MULLEDER_SM_QUOTE, note=_PERCENT_BASIS_NOTE),
                _messner_sv("2%", _MESSNER_SM_QUOTE),
            ],
        ),
    ],
    provenance=[
        _mulleder_sv("SM agar recipe", _MULLEDER_SM_QUOTE),
        _messner_sv("SM agar recipe", _MESSNER_SM_QUOTE),
    ],
)
"""Solid SM: the spotting/transfer plates of both screens, liquid ``SM`` plus 2% agar.

Mulleder's loader records the screen environment as ``state="solid"``, which is this
object. The paper's amino-acid measurements are taken from the LIQUID subculture the
spots inoculate ("These spots were used for the inoculation of cultures in liquid SM"),
so whether that loader should carry ``SM`` instead of ``SM_AGAR`` is a separate question
about the record, not about the recipe, and it is not decided here.
"""

SM_DEFERRED = Media(
    name="SM (synthetic minimal, composition deferred to Mulleder 2012), liquid",
    state="liquid",
    is_synthetic=True,
    base_medium="SM_DEFERRED",
    components=[
        _mixture(
            "synthetic minimal (SM) medium, prototrophic-collection formulation",
            MediaComponentRole.other,
            _DEFERRED,
            provenance=[
                _sv(
                    "composition deferred to Mulleder 2012",
                    _ZELEZNIAK_STRAIN_QUOTE,
                    ck=_ZELEZNIAK2018,
                    sha=_ZELEZNIAK2018_SHA,
                    uri="paper.md",
                )
            ],
            note="Zelezniak 2018 states NO recipe: its Strains and Culture section "
            "names 'minimal medium' and 'synthetic minimal (SM)' and sources both the "
            "strains and the cultivation to Mulleder et al. 2012, Nat Biotechnol "
            "30:1176-1178, which is not mirrored. The medium is plausibly the same "
            "6.7 g/L YNB + 2% glucose formulation the same lab writes out in Mulleder "
            "2016 and Messner 2023 restates, but that identification is nowhere stated, "
            "so the grams are NOT copied in and the carbon source stays inside this "
            "deferred line",
            defers_to=[_MULLEDER2012],
        )
    ],
    provenance=[
        _sv(
            "minimal medium, no composition given",
            _ZELEZNIAK_CULTURE_QUOTE,
            ck=_ZELEZNIAK2018,
            sha=_ZELEZNIAK2018_SHA,
            uri="paper.md",
            note="the pre-culture and the 30 mL main culture are both 'minimal medium'; "
            "the 30 C is a Temperature on the Environment, not a Media field",
        )
    ],
)
"""Zelezniak 2018's SM, kept a DISTINCT object because its recipe is unstated.

Separating it is the whole point of the component treatment: with empty components, this
medium and the Mulleder/Messner SM were indistinguishable strings. They are still not
joined on composition here, and that is the honest state until Mulleder 2012 is mirrored.
"""

# =========================================================================== #
# BACTERIAL MEDIA ([[plan.bacteria-ontology-genome]] section 3d, Step 5).
#
# Every entry quotes the mirrored Methods or SI of a paper among the fifty bacterial rows,
# with the sha256 of the quoted file from that key's ``manifest.json``. Three rules carry
# over from the yeast half and two are new:
#
# - A recipe that one paper states is that paper's object. "M9" names at least six
#   different formulations across the fifty (ammonium chloride or ammonium sulfate as the
#   nitrogen salt, anhydrous or hydrated phosphate, with or without trace metals), so the
#   per-paper variants carry the paper in the key and derive from the ``M9`` salts base.
# - A replaced base component is a dropout. The ammonium-sulfate formulations drop the
#   base's ammonium chloride; a medium that weighs the phosphate as a hydrate drops the
#   anhydrous salt. That is the SynBase convention (ammonium sulfate replaced by MSG).
# - A hydrate is its own compound: the weighed reagent is part of the recipe, and PubChem
#   gives each hydrate its own InChIKey.
# - When a paper varies the carbon (or nitrogen) source across conditions, the medium
#   leaves it out and is listed in ``CARBON_FREE_MEDIA``; the loader carries the variable
#   as ``EnvironmentPhysicalPerturbation(factor=carbon_source | nitrogen_source)``.
# - The xlsx sources (Wetmore 2015 Data Set S1, Price 2018 Supplementary Table 18) are
#   binary, so their quotes are row renderings: a row's non-empty cells joined by " | ",
#   consecutive rows of one sheet joined by " / ". The audit re-reads the sheet.
# =========================================================================== #


def _cite(
    source: Provenance, value: object, quote: str, *, note: str | None = None
) -> SourcedValue:
    """A ``SourcedValue`` quoting the pinned artifact ``source`` names."""
    return SourcedValue(value=value, provenance=source, quote=quote, note=note)


_MENASALVAS2025 = Provenance(
    citation_key="menasalvasBiosensordrivenStrainEngineering2025",
    sha256="d14536948af5ba67d52362ae71fb804a2fac03a1f7ab152817a01df3aa92d080",
    source_uri="paper.md",
)
_SCHMIDT2016 = Provenance(
    citation_key="schmidtQuantitativeConditiondependentEscherichia2016",
    sha256="67bedae8f934421086c23b7fa9582b0950a07206bffe7c4ffdd11e4d003a8710",
    source_uri="paper.md",
)
_KANG2026 = Provenance(
    citation_key="kangMultilayeredMetabolicRemodeling2026",
    sha256="894aee24472194d22c33ecc0a31994660b19bde41e4aadee2cb0080f0c016c63",
    source_uri="paper.md",
)
_BORCHERT2024 = Provenance(
    citation_key="borchertMachineLearningAnalysis2024",
    sha256="9de6b0772124fb77795764ce2f44187ec95737de7dcefb7335941543e75ce2f4",
    source_uri="paper.md",
)
_CHOE2019 = Provenance(
    citation_key="choeAdaptiveLaboratoryEvolution2019",
    sha256="11a042209df0f0518197d594340ed4ab573d90ddc2574a37139451ae697f7713",
    source_uri="paper.md",
)
_SCHASTNAYA2021 = Provenance(
    citation_key="schastnayaExtensiveRegulationEnzyme2021",
    sha256="f299eee75a1f022e08ba7f1bc444922db357f220be94f19466a36881214546c3",
    source_uri="paper.md",
)
_FUHRER2017 = Provenance(
    citation_key="fuhrerGenomewideLandscapeGene2017",
    sha256="ec87736bda4d30fa4c5eafa49a5d033bfff4f1ee3d641bbeb4c7f9e5d55f055b",
    source_uri="paper.md",
)
_CAGLAR2017 = Provenance(
    citation_key="caglarColiMolecularPhenotype2017",
    sha256="0878d5e7d49bcea4570aa2db225318a8469f4755645f1ddf73563effe7b3109b",
    source_uri="paper.md",
)
_FOO2014 = Provenance(
    citation_key="fooImprovingMicrobialBiogasoline2014",
    sha256="b24baad46bdf488cf93a7e59c51eceb2af9ba4ad565459130897a6201cd62407",
    source_uri="paper.md",
)
_BANERJEE2025 = Provenance(
    citation_key="banerjeeAddressingGenomeScale2025",
    sha256="74d7040a0a7ec721f18e5fc49d855a5c9c6eaeee9ff7bf166ec675824e27c489",
    source_uri="paper.md",
)
_CARRUTHERS2025 = Provenance(
    citation_key="carruthersAutomationMachineLearning2025",
    sha256="ca9a8a2593d2ae3ab3bacfb767e797ece2bdaa0228a4f798f1c38e1af73ef88d",
    source_uri="paper.md",
)
_DESIQUEIRA2025 = Provenance(
    citation_key="desiqueiraAlternateRoutesAcetate2025",
    sha256="a3ea14adcbe77144f02fb92ae6b344e4dec637a4211c702aa29fa121eda2de9c",
    source_uri="paper.md",
)
_LIM2025 = Provenance(
    citation_key="limEvolutionguidedToleranceEngineering2025",
    sha256="26b88d819d429f49cad4ebe85047cfd7354d532d84d584b0f00b222355772642",
    source_uri="paper.md",
)
_GOODALL2018 = Provenance(
    citation_key="goodallEssentialGenomeEscherichia2018",
    sha256="facfe1fac8ab6dab5cd0ecb88876b33616cc00de7c1fe68b60590f7a1883b429",
    source_uri="paper.md",
)
_BABU2014_SI13 = Provenance(
    citation_key="babuQuantitativeGenomeWideGenetic2014",
    sha256="c6cb4faf231286ae313c78f0ba5b2972f4c5259381dfec43ff7cc2019941e2c9",
    source_uri="si/si13.md",
)
_WANG2015 = Provenance(
    citation_key="wangDynamicInterplayMultidrug2015",
    sha256="9cd90f00599871689fa8e62849b4220fa0801a502160b575bca4d04067fc3cd2",
    source_uri="paper.md",
)
_TONG2020 = Provenance(
    citation_key="tongGeneDispensabilityEscherichia2020",
    sha256="daea2b924f553b75c3bfc626b2dbd10147e4a23c4272dd4b03b703242ac1623a",
    source_uri="paper.md",
)
_WANG2018 = Provenance(
    citation_key="wangPooledCRISPRInterference2018",
    sha256="9980415d606835ab1ae3a1ad92a64784c822aad1ca01b514662d2fb21d1a3a55",
    source_uri="paper.md",
)
#: Wetmore 2015 Data Set S1 ("The defined medium formulations used for each experiment
#: are contained in Data Set S1"), sheets ``Media`` and ``Expts_Keio``.
_WETMORE2015_DS1 = Provenance(
    citation_key="wetmoreRapidQuantificationMutant2015",
    sha256="428a06cae37867c8d21541f64d80da082e67743e1b1def9238a14e7726bc7150",
    source_uri="si/si1.xlsx",
)
#: Price 2018 Supplementary Tables (``si3.xlsx``), sheet ``TableS18_Medias``.
_PRICE2018_S3 = Provenance(
    citation_key="priceMutantPhenotypesThousands2018",
    sha256="e5dbf3d5c97cfc12f49d7fd83f84bc95c16cbe963309a561fff20f442788b879",
    source_uri="si/si3.xlsx",
)

#: Deferral targets that are NOT mirrored, keyed the way the library keys a paper.
_NEIDHARDT1974 = "neidhardtCultureMediumEnterobacteria1974"
_LENSKI1991 = "lenskiLongtermExperimentalEvolution1991"
_LIM2020 = "limGenerationIonicLiquid2020"
_LINGER2014 = "lingerLigninValorizationIntegrated2014"

_M = ConcentrationUnit.molar
_MM = ConcentrationUnit.millimolar
_UM = ConcentrationUnit.micromolar
_NM = ConcentrationUnit.nanomolar
_VV = ConcentrationUnit.percent_v_v

_SALT = MediaComponentRole.bulk_salt
_NSRC = MediaComponentRole.nitrogen_source
_CSRC = MediaComponentRole.carbon_source
_TRACE = MediaComponentRole.trace_element
_BUFFER = MediaComponentRole.buffer
_VITAMIN = MediaComponentRole.vitamin
_COMPLEX = MediaComponentRole.complex_ingredient


def _stated(
    name: str,
    role: MediaComponentRole,
    value: float,
    unit: ConcentrationUnit,
    *provenance: SourcedValue,
    note: str | None = None,
    defers_to: list[str] | None = None,
) -> MediaComponent:
    """A single-substance component at the amount its source(s) state."""
    return _defined(
        name,
        role,
        concentration=_c(value, unit),
        provenance=list(provenance),
        note=note,
        defers_to=defers_to,
    )


#: The anhydrous phosphate of the ``M9`` salts, dropped by a medium that weighs it as a
#: hydrate, and the ammonium chloride an ammonium-sulfate formulation does not contain.
_M9_ANHYDROUS_PHOSPHATE = resolved_compound("disodium hydrogen phosphate")
_M9_AMMONIUM_CHLORIDE = resolved_compound("ammonium chloride")
_HYDRATE_DROPOUT_NOTE = (
    "the phosphate is weighed as a hydrate, a different reagent with its own InChIKey, so "
    "the base's anhydrous disodium hydrogen phosphate is recorded as replaced (a dropout)"
)
_AMMONIUM_SULFATE_DROPOUT_NOTE = (
    "the nitrogen salt is ammonium sulfate, so the base's ammonium chloride is recorded "
    "as replaced (a dropout), the SynBase convention"
)

# Verbatim quotes, cut from the pinned files (paper.md OCR keeps its LaTeX markup; the
# xlsx quotes are row renderings, see the section comment above).
_Q_MENASALVAS_LB = (
    "The LB Miller (Luria-Bertani) medium [tryptone $( 1 0 \\mathrm { g / }$ "
    "liter), yeast extract $( 5 ~ \\mathrm { g / l i t e r } )$ , and NaCl (10 "
    "g/liter)] was purchased from Becton Dickinson (BD Difco, product no. "
    "244620)."
)
_Q_MENASALVAS_AGAR = (
    "When cells were cultured on petri dishes, LB medium was supplemented with $2 "
    "\\%$ $\\scriptstyle \\left( \\mathbf { w } / \\mathbf { v } \\right)$ solid agar "
    "(Becton Dickinson, Bacto Agar)"
)
_Q_MENASALVAS_M9 = (
    "At the 1X working concentration, M9 medium contains 47.9 mM ${ \\mathrm { N a "
    "} } _ { 2 } { \\mathrm { H P O } } _ { 4 } ,$ 22 mM ${ \\mathrm { K H } } _ { "
    "2 } { \\mathrm { P O } } _ { 4 }$ 8.56 mM NaCl, 2 mM $\\mathrm { M g S O _ { 4 "
    "} , }$ $1 0 0 ~ \\mu \\mathrm { M } \\mathrm { \\ C a C l } _ { 2 }$ with 1X "
    "trace metal solution (catalog no. T1001, Teknova Inc., Hollister, CA), $2 "
    "\\%$ glucose, $7 0 ~ \\mathrm { m M }$ $\\mathrm { ( N H _ { 4 } ) _ { 2 } S O "
    "_ { 4 } } ,$ and $3 0 \\mathrm { m M }$ Mops (Sigma-Aldrich, catalog no. "
    "M1254) adjusted to a $\\mathrm { p H }$ of 7.0."
)
_Q_MENASALVAS_NREL = (
    "This formulation of M9 used for $P .$ putida is sometimes referred to as "
    '"NREL ${ \\bf M 9 ^ { \\mathrm { * } } }$ or "Modified ${ \\bf M 9 ^ { \\mathrm '
    "{ * } } }$ (71, 72)."
)
_Q_SCHMIDT16_LB = (
    "Lysogeny broth (LB) medium was prepared as follows. Five grams of yeast "
    "extract (BD), $_ { 1 0 \\mathrm { ~ g ~ } }$ Tryptone (BD) and $1 0 \\ \\mathrm "
    "{ g \\ N a C l }$ were dissolved in one liter of water and the mixture "
    "sterilized by autoclaving."
)
_Q_SCHMIDT16_AGAR = (
    "LB plates were produced by adding $2 0 \\mathrm { g }$ agar (BD) to the LB "
    "medium mixture before autoclaving."
)
_Q_SCHMIDT16_SALTS = (
    "$2 0 0 ~ \\mathrm { m l }$ f $5 \\times$ base salt solution (211 mM ${ \\mathrm "
    "{ N a } } _ { 2 } { \\mathrm { H P O } } _ { 4 } ,$ $1 1 0 \\mathrm { m M }$ "
    "${ \\mathrm { K H } } _ { 2 } { \\mathrm { P O } } _ { 4 } ,$ 42.8 mM NaCl, $5 "
    "6 . 7 \\mathrm { m M }$ $\\mathrm { ( N H _ { 4 } ) _ { 2 } S O _ { 4 } , }$ "
    "in $\\mathrm { H } _ { 2 } \\mathrm { O } ,$ autoclaved)"
)
_Q_SCHMIDT16_TRACE = (
    "$1 0 ~ \\mathrm { m l }$ of te elemen $\\mathrm { 0 . 6 3 \\ m M }$ $\\mathrm { "
    "Z n S O _ { 4 } }$ . $0 . 7 \\ \\mathrm { m M } \\ \\mathrm { C u C l } _ { 2 }$ "
    ", $0 . 7 1 \\mathrm { \\ m M } \\mathrm { M n } \\mathrm { S O } _ { 4 }$ $0 . 7 "
    "6 \\mathrm { m M C o C l } _ { 2 } ,$ in $\\mathrm { H } _ { 2 } \\mathrm { O } "
    ",$ autoclaved)"
)
_Q_SCHMIDT16_CACL2 = (
    "$1 \\mathrm { m l } 0 . 1 \\mathrm { M C a C l } _ { 2 }$ solution"
)
_Q_SCHMIDT16_MGSO4 = "$1 \\mathrm { m l } 1$ M $\\mathrm { M g S O _ { 4 } }$ solution"
_Q_SCHMIDT16_THIAMINE = (
    "$2 \\mathrm { m l }$ of $5 0 0 \\times$ thiamine solution $\\mathrm { . 4 . m M }$"
)
_Q_SCHMIDT16_FECL3 = (
    "$0 . 6 \\ \\mathrm { m l } \\ 0 . 1 \\ \\mathrm { M } \\ \\mathrm { F e C l } _ { 3 "
    "}$ solution"
)
_Q_SCHMIDT16_CARBON = (
    "M9 minimal medium was complemented with carbon source by mixing appropriate "
    "amounts of carbon source free M9 minimal medium and carbon source stock "
    "solutions."
)
_Q_SCHMIDT16_GLUCOSE = (
    "The following carbon sources and concentrations were used: acetate (sodium "
    "acetate, $3 . 5 \\mathrm { g } / \\mathrm { L }$ ), fumarate (disodium "
    "fumarate, $2 . 8 \\mathrm { g } / \\mathrm { L }$ , galactose $( 2 . 3 \\mathrm "
    "{ g } / \\mathrm { L } )$ , glucose $( 5 \\mathrm { g } / \\mathrm { L } )$ , "
    "glucosamine $( 2 . 1 \\mathrm { g } / \\mathrm { L } )$ , glycerol $( 2 . 2 "
    "\\mathrm { g } / \\mathrm { L } )$ , pyruvate (sodium pyruvate, $3 . 3 \\mathrm "
    "{ g / L }$ ), sucnate (disodium succinate hexahydrate, $5 . 7 \\mathrm { g / "
    "L }$ , fructose $( 5 \\mathrm { g } / \\mathrm { L } )$ , mannose $( 5 \\mathrm "
    "{ g } / \\mathrm { L } ) \\mathrm { g }$ and xylose $( 5 \\mathrm { g } / "
    "\\mathrm { L } )$ ."
)
_Q_KANG_M9 = (
    "M9 medium was prepared with the following components: $2 \\ g / \\mathrm { L "
    "}$ $\\mathrm { ( N H _ { 4 } ) _ { 2 } S O _ { 4 } } ,$ $6 . 8 ~ \\ g / "
    "\\mathrm { L }$ ${ \\mathrm { N a } } _ { 2 } { \\mathrm { H P O } } _ { 4 }$ , "
    "$3 \\ \\gimel A$ ${ \\mathrm { K H } } _ { 2 } { \\mathrm { P O } } _ { 4 }$ , "
    "$0 . 5 ~ \\ g / \\mathrm { L }$ NaCl, $1 \\ \\mathrm { m L } / \\mathrm { L }$ "
    "trace element solution (Teknova, Hollister, CA), $0 . 1 ~ \\mathrm { m M }$ "
    "$\\mathrm { C a C l } _ { 2 }$ , and $2 \\mathrm { \\ m M \\ M g { S O _ { 4 } } "
    "}$ ."
)
_Q_KANG_MODIFIED = (
    "For experiments requiring modified nitrogen levels, the concentration of "
    "$\\mathrm { ( N H } _ { 4 } \\mathrm { ) } _ { 2 } S 0 _ { 4 }$ was increased "
    "to $4 0 ~ \\mathrm { m M }$ and is referred to as modified M9."
)
_Q_KANG_MOPS = (
    "M9-MOPS was prepared with the following components: M9 salts $( 6 . 7 8 \\ g "
    "/ \\mathrm { L }$ ${ \\mathrm { N a } } _ { 2 } { \\mathrm { H P O } } _ { 4 }$ "
    ", $_ { 3 } \\ g / \\mathrm { L }$ $\\mathrm { K H _ { 2 } P O _ { 4 } }$ , $^ "
    "\\textrm { \\scriptsize 1 g / L }$ $\\mathrm { N H } _ { 4 } \\mathrm { C l } ,$ "
    "and $0 . 5 ~ \\mathrm { \\ g / L }$ NaCl), $7 5 ~ \\mathrm { \\ m M }$ "
    "3-morpholinopropane-1-sulfonic acid (MOPS), $1 \\ \\mathrm { m g } / \\mathrm { "
    "L }$ thiamine, $1 0 \\ \\mathrm { n M \\ F e S O _ { 4 } }$ , micronutrients $( "
    "3 ^ { * } 1 0 ^ { - 8 } \\mathrm { ~ M ~ }$ $\\mathrm { ( N H _ { 4 } ) _ { 6 "
    "} M o _ { 7 } O _ { 2 4 } }$ , $4 ^ { * } 1 0 ^ { - 6 }$ M boric acid, $3 ^ "
    "{ * } 1 0 ^ { - 7 }$ M $\\mathrm { C o C l } _ { 2 }$ , $1 . 5 ^ { * } 1 0 ^ "
    "{ - 7 }$ M $\\mathrm { C u S O } _ { 4 }$ , $8 ^ { * } 1 0 ^ { - 7 } \\mathrm "
    "{ \\ : M \\ : M n C l _ { 2 } } ,$ , and $1 ^ { * } 1 0 ^ { - 7 } \\mathrm { M "
    "} \\mathrm { Z n } \\mathrm { S O } _ { 4 } )$ , $2 \\mathrm { m M } \\mathrm { "
    "M g } S 0 _ { 4 }$ , and 0.1 $\\mathbf { m } \\mathbf { M } \\mathbf { C a C l "
    "} _ { 2 }$ ."
)
_Q_KANG_SALTS = (
    "M9 salts $( 6 . 7 8 \\ g / \\mathrm { L }$ ${ \\mathrm { N a } } _ { 2 } { "
    "\\mathrm { H P O } } _ { 4 }$ , $_ { 3 } \\ g / \\mathrm { L }$ $\\mathrm { K H "
    "_ { 2 } P O _ { 4 } }$ , $^ \\textrm { \\scriptsize 1 g / L }$ $\\mathrm { N H "
    "} _ { 4 } \\mathrm { C l } ,$ and $0 . 5 ~ \\mathrm { \\ g / L }$ NaCl)"
)
_Q_KANG_SUGAR = (
    "Unless otherwise noted, cultures contained $2 0 ~ \\ g / \\mathrm { L }$ total "
    "sugar (either glucose alone or a 2:1 glucose:xylose mixture)"
)
_Q_BORCHERT24_M9 = (
    "$1 \\times 1 \\mathsf { M } 9$ medium $( 6 . 7 8 ~ \\mathrm { g / L ~ N a _ { 2 "
    "} H P O _ { 4 } , 3 ~ \\mathrm { g / L ~ K H _ { 2 } P O _ { 4 } , 0 . 5 ~ "
    "\\mathrm { g / L ~ N a C l } , } }$ 1 g/L $\\mathsf { N H } _ { 4 } \\mathsf { "
    "C l }$"
)
_Q_CHOE19_M9 = (
    "Cells were grown in M9 glucose medium (47.75 $\\mathrm { m M }$ of ${ \\mathrm "
    "{ N a } } _ { 2 } { \\mathrm { H P O } } _ { 4 } ,$ , $2 2 . 0 4 \\mathrm { m "
    "M }$ of ${ \\mathrm { K H } } _ { 2 } { \\mathrm { P O } } _ { 4 }$ , $8 . 5 6 "
    "\\mathrm { m M }$ of NaCl, $1 8 . 7 0 \\mathrm { m M }$ of $\\mathrm { N H _ { "
    "4 } C l , }$ 2 mM of $\\mathrm { M g S O _ { 4 } }$ , $0 . 1 \\mathrm { m M }$ "
    "of $\\mathrm { C a C l } _ { 2 }$ , and $2 { \\bf g } 1 ^ { - 1 }$ of glucose)"
)
_Q_SCHASTNAYA_LENNOX = (
    "in LB-Lennox medium $\\mathrm { { \\Delta } _ { 1 0 } g / L }$ tryptone, $5 "
    "\\mathrm { g / L }$ yeast extract, $5 \\mathrm { g / L }$ NaCl)"
)
_Q_FUHRER_M9 = (
    "were grown on glucose minimal medium supplemented with casein hydrolysate "
    "containing (per liter): $_ \\textrm { 4 g }$ glucose, $_ { 2 \\mathrm { ~ g ~ "
    "} }$ N-Z Case Plus, $7 . 5 2 \\ \\mathrm { g }$ $\\mathrm { N a } _ { 2 } "
    "\\mathrm { H P O } _ { 4 } { \\cdot } 2 \\mathrm { H } _ { 2 } \\mathrm { O }$ , "
    "$3 \\textrm { g K H } _ { 2 } \\mathrm { P O } _ { 4 }$ , $0 . 5 \\mathrm { ~ g "
    "~ N a C l }$ , $2 . 5 \\ \\mathrm { g }$ $\\mathrm { \\Omega _ { 5 } } \\left( "
    "\\mathrm { N H _ { 4 } } \\right) _ { 2 } \\mathrm { S O _ { 4 } }$ , $1 4 . 7 "
    "~ \\mathrm { m g }$ $\\mathrm { C a C l } _ { 2 } { \\cdot } 2 \\mathrm { H } _ "
    "{ 2 } 0$ , $2 4 6 . 5 ~ \\mathrm { m g }$ $\\mathrm { M g S O } _ { 4 } { "
    "\\cdot } 7 \\mathrm { H } _ { 2 } \\mathrm { O }$ , $1 6 . 2 ~ \\mathrm { m g }$ "
    "$\\mathrm { F e C l } _ { 3 } { \\cdot } 6 \\mathrm { H } _ { 2 } 0$ , $1 8 0 ~ "
    "\\mu \\ g$ $\\mathrm { Z n S O } _ { 4 } { \\cdot } 7 \\mathrm { H } _ { 2 } "
    "\\mathrm { O }$ , $1 2 0 ~ \\mu \\ g$ $\\mathrm { C u C l } _ { 2 } { \\cdot } 2 "
    "\\mathrm { H } _ { 2 } 0$ , $1 2 0 ~ \\mu \\ g$ $\\mathrm { M n S O } _ { 4 } { "
    "\\cdot } \\mathrm { H } _ { 2 } \\mathrm { O }$ , $1 8 0 ~ \\mu \\ g$ $\\mathrm { "
    "C o C l } _ { 2 } { \\cdot } 6 \\mathrm { H } _ { 2 } 0$ , $1 ~ \\mathrm { m g "
    "}$ thiamine HCl."
)
_Q_CAGLAR_DM = (
    "Davis Minimal medium supplemented with $2 \\mu \\mathrm { g } / 1$ thiamine $( "
    "\\mathrm { D M } ) ^ { 3 6 }$ and limiting glucose at $5 0 0 \\mathrm { m g / "
    "l }$ (DM500)"
)
_Q_CAGLAR_MG = (
    "concentrations were varied by changing the amount of $\\mathrm { M g S O _ { "
    "4 } }$ added to DM media from the concentration of $0 . 8 3 \\mathrm { m M }$ "
    "that is normally present."
)
_Q_CAGLAR_NA = (
    "The base recipe for DM already contains ${ \\sim } 5 \\mathrm { m M N a ^ { + "
    "} }$ due to the inclusion of sodium citrate"
)
_Q_CAGLAR_CARBON = (
    "For tests of different carbon sources, the Davis Minimal (DM) medium used "
    "was supplemented with $0 . 5 { \\mathrm { g } } / { \\mathrm { L } }$ of the "
    "specified compound (glycerol, lactate, or gluconate) instead of glucose."
)
_Q_CAGLAR_REF36 = (
    "36. Lenski, R. E., Rose, M. R., Simpson, S. C. & Tadler, S. C. Long-Term "
    "Experimental Evolution in Escherichia coli. I. Adaptation and Divergence "
    "During 2,000 Generations. Am. Nat. 138, 1315–1341 (1991)."
)
_Q_FOO_M9 = (
    "Growth assays were performed in M9 minimal medium, which consisted of $1 "
    "\\times$ M9 salt (Difco), 2 mM $\\mathrm { M g S O _ { 4 } }$ , $1 0 0 \\mu "
    "\\mathrm { M C a C l } _ { 2 }$ , 0.5 mg liter-1 thiamine, and $0 . 4 \\%$ "
    "glucose."
)
_Q_FOO_MM9 = (
    "Isopentenol production strains were grown in a modified M9 "
    "3-morpholinopropane-1-sulfonic acid (MOPS) minimal medium (MM9), which "
    "consisted of $1 \\times$ M9 salt (Difco), $7 5 \\mathrm { m M }$ MOPS $\\mathrm "
    "{ \\Phi _ { \\mathrm { p H } } } 7 . 4 0$ , $2 \\mathrm { m M }$ $\\mathrm { M g "
    "S O _ { 4 } , }$ $1 0 ~ \\mu \\mathrm { M }$ $\\mathrm { C a C l } _ { 2 }$ , "
    "$1 0 ~ \\mu \\mathrm { M }$ $\\mathrm { F e S O _ { 4 } } ,$ , $1 \\times$ "
    "micronutrient, and $1 \\%$ glucose. The $1 \\times$ micronutrient was composed "
    "of $4 \\mu \\mathrm { M }$ boric acid, $0 . 8 \\mu \\mathrm { M }$ manganese "
    "chloride, $0 . 3 ~ \\mu \\mathrm { M }$ cobalt chloride, $0 . 1 5 ~ \\mu "
    "\\mathrm { M }$ cupric sulfate, $0 . 1 \\mu \\mathrm { M }$ zinc sulfate, and "
    "$0 . 0 3 \\mu \\mathrm { M }$ ammonium molybdate."
)
_Q_CARRUTHERS_M9 = (
    "Briefly, the medium composition included $2 0 \\mathrm { g / L }$ glucose, "
    "$\\mathbf { 0 . 5 8 } / \\mathbf { L }$ NaCl, ${ 6 . 8 } \\mathrm { g } / "
    "\\mathrm { L }$ ${ \\sf N a } _ { 2 } { \\sf H P O } _ { 4 }$ , $_ { 3 \\mathrm "
    "{ g } / \\mathrm { L } }$ ${ \\mathrm { K H } } _ { 2 } { \\mathsf { P O } } _ "
    "{ 4 }$ $1 0 0 \\mu \\mathrm { M }$ $\\mathbf { C a C l } _ { 2 }$ , 2 mM $\\bf { "
    "M g S O _ { 4 } }$ , $1 0 \\mathrm { m M }$ $( \\mathsf { N H } _ { 4 } ) _ { "
    "2 } \\mathsf { S O } _ { 4 }$ , and ${ 5 0 0 } \\mu \\mathrm { L }$ of a trace "
    "metal solution (Teknova Cat no. T1001; Teknova, Hollister, CA)."
)
_Q_CARRUTHERS_NREL = (
    "M9-NREL medium was selected owing to its prevalence as a baseline $P .$ . "
    "putida production medium."
)
_Q_DESIQUEIRA_M9 = (
    "P. putida tolerization and the subsequent phenotypic characterization "
    "experiments were performed using minimal salt (M9) medium composed of $1 "
    "\\times 1 \\mathsf { M } 9$ salts $( 2 ~ { \\mathfrak { g } } / { \\mathsf { L } "
    "} ~ ( { \\mathsf { N H } } _ { 4 } ) _ { 2 } { \\mathsf { S O } } _ { 4 } ,$ "
    "$6 . 8 \\ \\mathsf { g } / \\mathsf { L N a } _ { 2 } \\mathsf { H P O } _ { 4 } "
    ",$ 3 g/L $\\mathsf { K H } _ { 2 } \\mathsf { P O } _ { 4 } , \\mathsf { 0 } . "
    "5 \\mathsf { \\mathsf { g } } / \\mathsf { L } \\mathsf { N }$ aCl), 2 mM MgSO4, "
    "$0 . 1 \\mathrm { ~ m M ~ C a C l } _ { 2 } ,$ and trace metal solution $5 0 "
    "0 ~ \\mu \\iota$ per 1L medium; Product No. 1001, Tekova Inc, Hollister, CA)."
)
_Q_DESIQUEIRA_CARBON = (
    "For all experiments, except when noted, acetate $5 0 ~ \\mathsf { m M }$ ${ "
    "\\it \\simeq } 0 . 3 \\%$ wt/vol) or glucose $1 \\%$ (wt/vol) was used as the "
    "single sole carbon source in M9 medium."
)
_Q_LIM25_M9 = (
    "The M9 medium contained $2 \\ g / \\mathrm { L }$ $( \\mathrm { N H } _ { 4 } ) "
    "_ { 2 } S 0 _ { 4 }$ , $6 . 8 ~ \\ g / \\mathrm { L }$ ${ \\mathrm { N a } } _ "
    "{ 2 } { \\mathrm { H P O } } _ { 4 }$ , $3 \\ \\mathrm { g / L }$ ${ \\mathrm { "
    "K H } } _ { 2 } { \\mathrm { P O } } _ { 4 }$ , $0 . 5 ~ { \\ g / \\mathrm { L "
    "} }$ NaCl, $2 \\ \\mathrm { m M }$ $\\mathrm { M g } { \\bf S } 0 _ { 4 } ,$ 0.1 "
    "mM $\\mathrm { C a C l } _ { 2 }$ , $5 0 0 ~ \\mu \\mathrm { L } / \\mathrm { L "
    "}$ $2 0 0 0 \\times$ trace element solution (Lim et al., 2020; Linger et al., "
    "2014)."
)
_Q_LIM25_GLUCOSE = (
    "As a carbon source, $4 \\ g / \\mathrm { L }$ glucose was added to the minimal "
    "medium unless otherwise stated."
)
_Q_GOODALL_LB = (
    "Luria broth (LB) $. 1 0 ~ 9$ tryptone, 5 g yeast extract, $1 0 \\ 9 \\ { "
    "\\mathsf { N a C l } } )$"
)
_Q_BABU_LB = (
    "E. coli cells were grown in LB ( $1 0 ~ \\mathrm { g / L }$ Bacto-tryptone, "
    "$5 \\ \\mathrm { g / L }$ Yeast extract, and $1 0 ~ \\mathrm { g / L } \\mathrm "
    "{ N a C l ) }$"
)
_Q_WANG15_2YT = (
    "in 2YT medium (Bacto-tryptone $1 6 { \\mathrm { g } } ,$ Bacto-yeast extract "
    "$1 0 { \\mathrm { g } } { \\mathrm { . } }$ and sodium chloride $5 \\mathrm { g "
    "}$ per liter, adjusted to $\\mathrm { p H } 7 . 0 $ )"
)
_Q_TONG_MOPS = (
    "Using a chemically defined minimal medium (morpholinepropanesulfonic acid "
    "[MOPS]) and changing only the carbon source (34)"
)
_Q_TONG_TEKNOVA = "MOPS minimal media (Teknova) was used for all work in minimal media."
_Q_TONG_REF34 = (
    "34. Neidhardt FC, Bloch PL, Smith DF. 1974. Culture medium for "
    "enterobacteria. J Bacteriol 119:736 –747."
)
_Q_WETMORE_LB = (
    "LB / defined | False / desc | Luria-Bertani broth / minimal | False / "
    "Controlled vocabulary | Concentration | Units / Tryptone | 10 | g/L / Yeast "
    "Extract | 5 | g/L / Sodium Chloride | 5 | g/L"
)
_Q_WETMORE_M9 = (
    "M9 minimal media_noCarbon / defined | True / desc | E. coli defined media "
    "with no carbon source / minimal | True / Controlled vocabulary | "
    "Concentration | Units / Sodium phosphate dibasic heptahydrate | 13 | g/L / "
    "Potassium phosphate monobasic | 3 | g/L / Sodium Chloride | 0.5 | g/L / "
    "Ammonium chloride | 1 | g/L / Magnesium sulfate | 2 | mM / CaCl2  | 0.1 | mM"
)
_Q_WETMORE_MOPS_K2SO4 = (
    "MOPS Rich Defined media_noCarbon / defined | True / desc | E. coli MOPS "
    "defined rich media with no carbon source / minimal | False / Controlled "
    "vocabulary | Concentration | Units / MOPS | 40 | mM / Tricine | 4 | mM / "
    "Iron Sulfate Stock | 0.01 | mM / Ammonium Chloride | 9.5 | mM / Potassium "
    "Sulfate | 0.276 | mM"
)
_Q_WETMORE_GLUCOSE_EXPT = (
    "Keio_ML9_set1 | Keio_ML9 | D-Glucose carbon source | IT003 | BarSeq98 | M9 "
    "minimal media_noCarbon | tube | carbon source | 37 | Liquid | Aerobic | 200 "
    "rpm | D-Glucose | 20 | mM | 0.02 | 1.85 | 6.529820947 | 3 | set1IT003 | "
    "D-Glucose (C)"
)
_Q_WETMORE_M9_NON = (
    "M9 minimal media_noNitrogen / defined | True / desc | E. coli defined media "
    "with no nitrogen source; glucose carbon source / minimal | True / Controlled "
    "vocabulary | Concentration | Units / D-Glucose | 4 | g/L / Sodium phosphate "
    "dibasic heptahydrate | 13 | g/L / Potassium phosphate monobasic | 3 | g/L / "
    "Sodium Chloride | 0.5 | g/L / Magnesium sulfate | 2 | mM / CaCl2  | 0.1 | mM"
)
_Q_PRICE_M9_NON = (
    "Media | M9 minimal media_noNitrogen / Description | E. coli defined media no "
    "nitrogen / Minimal | =TRUE() / Controlled vocabulary | Concentration | Units "
    "/ D-Glucose | 4 | g/L / Magnesium sulfate | 2 | mM / Calcium chloride | 0.1 "
    "| mM / Sodium phosphate dibasic heptahydrate | 12.8 | g/L / Potassium "
    "phosphate monobasic | 3 | g/L / Sodium Chloride | 0.5 | g/L"
)
_Q_WETMORE_NITROGEN_EXPT = (
    "Keio_ML9_set1 | Keio_ML9 | L-Arginine nitrogen source | IT071 | BarSeq98 | "
    "M9 minimal media_noNitrogen | tube | nitrogen source | 37 | Liquid | Aerobic "
    "| 200 rpm | L-Arginine | 10 | mM | 0.02 | 0.67 | 5.06608919 | 71 | set1IT071 "
    "| L-Arginine (N)"
)
_Q_PRICE_LB = (
    "Media | LB / Description | Luria-Bertani broth / Minimal | =FALSE() / "
    "Controlled vocabulary | Concentration | Units / Tryptone | 10 | g/L / Yeast "
    "Extract | 5 | g/L / Sodium Chloride | 5 | g/L"
)
_Q_PRICE_M9 = (
    "Media | M9 minimal media_noCarbon / Description | E. coli defined media no "
    "carbon / Minimal | =TRUE() / Controlled vocabulary | Concentration | Units / "
    "Magnesium sulfate | 2 | mM / Calcium chloride | 0.1 | mM / Sodium phosphate "
    "dibasic heptahydrate | 12.8 | g/L / Potassium phosphate monobasic | 3 | g/L "
    "/ Sodium Chloride | 0.5 | g/L / Ammonium chloride | 1 | g/L"
)
_Q_PRICE_MOPS = (
    "Media | MOPS minimal media_noCarbon / Description | MOPS minimal media with "
    "no carbon source / Minimal | =TRUE() / Controlled vocabulary | Concentration "
    "| Units / 3-(N-morpholino)propanesulfonic acid | 40 | mM / Tricine | 4 | mM "
    "/ K2HPO4 | 1.32 | mM / Iron (II) sulfate heptahydrate | 0.01 | mM / Ammonium "
    "chloride | 9.5 | mM / Aluminum potassium sulfate dodecahydrate | 0.276 | mM "
    "/ Calcium chloride | 0.0005 | mM / Magnesium chloride hexahydrate | 0.525 | "
    "mM / Sodium Chloride | 50 | mM / Ammonium heptamolybdate tetrahydrate | "
    "3e-09 | M / Boric Acid | 4e-07 | M / Cobalt chloride hexahydrate | 3e-08 | M "
    "/ Copper (II) sulfate pentahydrate | 1e-08 | M / Manganese (II) chloride "
    "tetrahydrate | 8e-08 | M / Zinc sulfate heptahydrate | 1e-08 | M"
)

# --------------------------------------------------------------------------- #
# LB family -- complex (NOT chemically defined): tryptone and yeast extract are
# intrinsically undefined digests. ``LB`` is the Miller formulation (10 g/L NaCl), which
# Menasalvas 2025 names and states and Schmidt 2016 states per liter; ``LB_LENNOX`` is the
# 5 g/L NaCl formulation. Both derive from ``LB`` because they share every ingredient by
# name and role and differ only in the NaCl amount.
# --------------------------------------------------------------------------- #
_LB_TRYPTONE = _mixture(
    "tryptone",
    _COMPLEX,
    _UNDEFINED,
    concentration=_c(10.0, _GL),
    provenance=[
        _cite(_MENASALVAS2025, "10 g/L", _Q_MENASALVAS_LB),
        _cite(_SCHMIDT2016, "10 g per liter", _Q_SCHMIDT16_LB),
    ],
)
_LB_YEAST_EXTRACT = _mixture(
    "yeast extract",
    _COMPLEX,
    _UNDEFINED,
    concentration=_c(5.0, _GL),
    provenance=[
        _cite(_MENASALVAS2025, "5 g/L", _Q_MENASALVAS_LB),
        _cite(_SCHMIDT2016, "five grams per liter", _Q_SCHMIDT16_LB),
    ],
)
_LB_MILLER_NACL = _stated(
    "sodium chloride",
    _SALT,
    10.0,
    _GL,
    _cite(_MENASALVAS2025, "10 g/L", _Q_MENASALVAS_LB),
    _cite(_SCHMIDT2016, "10 g per liter", _Q_SCHMIDT16_LB),
)

LB = Media(
    name="LB, Miller (10 g/L tryptone, 5 g/L yeast extract, 10 g/L NaCl), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="LB",
    components=[_LB_TRYPTONE, _LB_YEAST_EXTRACT, _LB_MILLER_NACL],
    provenance=[
        _cite(
            _MENASALVAS2025,
            "LB Miller",
            _Q_MENASALVAS_LB,
            note="the source names the formulation",
        ),
        _cite(
            _SCHMIDT2016,
            "lysogeny broth at the Miller amounts, per liter",
            _Q_SCHMIDT16_LB,
            note="Schmidt 2016's LB growth condition is this medium",
        ),
        _cite(
            _GOODALL2018,
            "10 g tryptone, 5 g yeast extract, 10 g NaCl",
            _Q_GOODALL_LB,
            note="the same three amounts with no per-volume basis printed, so this "
            "corroborates the Miller ratio rather than the per-liter reading; the OCR "
            "renders 'g' as '9'",
        ),
        _cite(
            _BABU2014_SI13,
            "10 g/L tryptone, 5 g/L yeast extract, 10 g/L NaCl",
            _Q_BABU_LB,
        ),
    ],
)
"""LB Miller, liquid: the formulation most of the fifty name or state.

Goodall 2018's TraDIS cultures (LB1, LB2), Schmidt 2016's LB condition and Menasalvas
2025's revival cultures state these amounts; several more rows name "LB Miller" without
amounts (``BACTERIAL_MEDIA_USES``). Wetmore 2015 and Price 2018 do NOT use this object:
their "LB" carries 5 g/L NaCl in their own media tables, which is ``LB_LENNOX``.
"""

LB_AGAR = Media(
    name="LB, Miller, solid (2% agar)",
    state="solid",
    is_synthetic=False,
    base_medium="LB",
    components=[
        *LB.components,
        _defined(
            "agar",
            MediaComponentRole.gelling_agent,
            concentration=_c(2.0, _PCT),
            provenance=[
                _cite(_MENASALVAS2025, "2% (w/v)", _Q_MENASALVAS_AGAR),
                _cite(
                    _SCHMIDT2016,
                    "20 g per liter of LB, which is 2% w/v",
                    _Q_SCHMIDT16_AGAR,
                    note="the agar is added to the one-liter LB mixture the preceding "
                    "sentence prepares",
                ),
            ],
        ),
    ],
    provenance=[
        _cite(_MENASALVAS2025, "LB Miller agar plates", _Q_MENASALVAS_AGAR),
        _cite(_SCHMIDT2016, "LB plates", _Q_SCHMIDT16_AGAR),
    ],
)
"""LB Miller plates: liquid ``LB`` plus 2% agar, stated independently by two papers."""

LB_LENNOX = Media(
    name="LB, Lennox (10 g/L tryptone, 5 g/L yeast extract, 5 g/L NaCl), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="LB",
    components=[
        _mixture(
            "tryptone",
            _COMPLEX,
            _UNDEFINED,
            concentration=_c(10.0, _GL),
            provenance=[
                _cite(_SCHASTNAYA2021, "10 g/L", _Q_SCHASTNAYA_LENNOX),
                _cite(_WETMORE2015_DS1, "10 g/L", _Q_WETMORE_LB),
                _cite(_PRICE2018_S3, "10 g/L", _Q_PRICE_LB),
            ],
        ),
        _mixture(
            "yeast extract",
            _COMPLEX,
            _UNDEFINED,
            concentration=_c(5.0, _GL),
            provenance=[
                _cite(_SCHASTNAYA2021, "5 g/L", _Q_SCHASTNAYA_LENNOX),
                _cite(_WETMORE2015_DS1, "5 g/L", _Q_WETMORE_LB),
                _cite(_PRICE2018_S3, "5 g/L", _Q_PRICE_LB),
            ],
        ),
        _stated(
            "sodium chloride",
            _SALT,
            5.0,
            _GL,
            _cite(_SCHASTNAYA2021, "5 g/L", _Q_SCHASTNAYA_LENNOX),
            _cite(_WETMORE2015_DS1, "5 g/L", _Q_WETMORE_LB),
            _cite(_PRICE2018_S3, "5 g/L", _Q_PRICE_LB),
        ),
    ],
    provenance=[
        _cite(
            _SCHASTNAYA2021,
            "LB-Lennox",
            _Q_SCHASTNAYA_LENNOX,
            note="the source names the formulation; the OCR renders the opening "
            "'(10' as a Delta subscript",
        ),
        _cite(
            _WETMORE2015_DS1,
            "the release's 'LB' is the Lennox formulation",
            _Q_WETMORE_LB,
            note="Data Set S1 names the medium 'LB' / 'Luria-Bertani broth' and gives "
            "5 g/L sodium chloride, the Lennox amount",
        ),
        _cite(
            _PRICE2018_S3,
            "the release's 'LB' is the Lennox formulation",
            _Q_PRICE_LB,
            note="Supplementary Table 18 repeats Wetmore 2015's LB row",
        ),
    ],
)
"""LB Lennox, liquid: the RB-TnSeq rows' "LB" (Wetmore 2015, Price 2018) and Schastnaya 2021.

The two RB-TnSeq papers call it "LB" in prose; their own media tables state 5 g/L NaCl,
so their loaders take this object, not ``LB``.
"""

YT_2X = Media(
    name="2YT (16 g/L tryptone, 10 g/L yeast extract, 5 g/L NaCl), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="YT_2X",
    components=[
        _mixture(
            "tryptone",
            _COMPLEX,
            _UNDEFINED,
            concentration=_c(16.0, _GL),
            provenance=[_cite(_WANG2015, "16 g per liter", _Q_WANG15_2YT)],
        ),
        _mixture(
            "yeast extract",
            _COMPLEX,
            _UNDEFINED,
            concentration=_c(10.0, _GL),
            provenance=[_cite(_WANG2015, "10 g per liter", _Q_WANG15_2YT)],
        ),
        _stated(
            "sodium chloride",
            _SALT,
            5.0,
            _GL,
            _cite(_WANG2015, "5 g per liter", _Q_WANG15_2YT),
        ),
    ],
    provenance=[
        _cite(
            _WANG2015,
            "2YT recipe",
            _Q_WANG15_2YT,
            note="the pH 7.0 the same sentence states is not a Media field; it rides as "
            "EnvironmentPhysicalPerturbation(factor=ph)",
        )
    ],
)
"""Wang 2015's 2YT, the medium of its isoprenol-tolerance growth assays."""

# --------------------------------------------------------------------------- #
# M9 family -- chemically defined. ``M9`` is the four salts, as Kang 2026 names them
# ("M9 salts") and Borchert 2024 states them at the same amounts; it names no carbon
# source, and every M9 formulation below derives from it.
# --------------------------------------------------------------------------- #
M9 = Media(
    name="M9 salts (6.78 g/L Na2HPO4, 3 g/L KH2PO4, 1 g/L NH4Cl, 0.5 g/L NaCl)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        _stated(
            "disodium hydrogen phosphate",
            _SALT,
            6.78,
            _GL,
            _cite(_KANG2026, "6.78 g/L", _Q_KANG_SALTS),
            _cite(_BORCHERT2024, "6.78 g/L", _Q_BORCHERT24_M9),
        ),
        _stated(
            "potassium dihydrogen phosphate",
            _SALT,
            3.0,
            _GL,
            _cite(_KANG2026, "3 g/L", _Q_KANG_SALTS),
            _cite(_BORCHERT2024, "3 g/L", _Q_BORCHERT24_M9),
        ),
        _stated(
            "ammonium chloride",
            _NSRC,
            1.0,
            _GL,
            _cite(_KANG2026, "1 g/L", _Q_KANG_SALTS),
            _cite(_BORCHERT2024, "1 g/L", _Q_BORCHERT24_M9),
        ),
        _stated(
            "sodium chloride",
            _SALT,
            0.5,
            _GL,
            _cite(_KANG2026, "0.5 g/L", _Q_KANG_SALTS),
            _cite(_BORCHERT2024, "0.5 g/L", _Q_BORCHERT24_M9),
        ),
    ],
    provenance=[
        _cite(
            _KANG2026, "M9 salts", _Q_KANG_SALTS, note="the source names the four salts"
        ),
        _cite(
            _BORCHERT2024,
            "the same four salts at the same four amounts",
            _Q_BORCHERT24_M9,
            note="Borchert 2024's 1x M9 also carries 2 mM MgSO4, 100 uM CaCl2 and 18 uM "
            "FeSO4; those belong to that paper's growth medium, not to the salts",
        ),
    ],
)
"""The M9 salts: the base every M9 formulation in the library derives from."""

M9_GLUCOSE = Media(
    name="M9 glucose (M9 salts + 2 mM MgSO4 + 0.1 mM CaCl2 + 2 g/L glucose; Choe 2019)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        _stated(
            "disodium hydrogen phosphate",
            _SALT,
            47.75,
            _MM,
            _cite(_CHOE2019, "47.75 mM", _Q_CHOE19_M9),
        ),
        _stated(
            "potassium dihydrogen phosphate",
            _SALT,
            22.04,
            _MM,
            _cite(_CHOE2019, "22.04 mM", _Q_CHOE19_M9),
        ),
        _stated(
            "sodium chloride",
            _SALT,
            8.56,
            _MM,
            _cite(_CHOE2019, "8.56 mM", _Q_CHOE19_M9),
        ),
        _stated(
            "ammonium chloride",
            _NSRC,
            18.70,
            _MM,
            _cite(_CHOE2019, "18.70 mM", _Q_CHOE19_M9),
        ),
        _stated(
            "magnesium sulfate", _SALT, 2.0, _MM, _cite(_CHOE2019, "2 mM", _Q_CHOE19_M9)
        ),
        _stated(
            "calcium chloride",
            _SALT,
            0.1,
            _MM,
            _cite(_CHOE2019, "0.1 mM", _Q_CHOE19_M9),
        ),
        _stated(
            "D-glucose",
            _CSRC,
            2.0,
            _GL,
            _cite(
                _CHOE2019,
                "2 g/L",
                _Q_CHOE19_M9,
                note="the OCR renders 'g l^-1' as 'g 1^-1'",
            ),
        ),
    ],
    provenance=[
        _cite(
            _CHOE2019,
            "M9 glucose medium",
            _Q_CHOE19_M9,
            note="the four salt amounts are the M9 salts' 6.78 / 3 / 0.5 / 1 g/L in molar "
            "units (arithmetic by molar mass, not a sourced conversion); they are "
            "recorded in the units the paper prints, so this medium and ``M9`` do not "
            "share a concentration identity",
        )
    ],
)
"""Choe 2019's M9 glucose: the complete classic recipe, salts plus Mg, Ca and glucose."""


def _rbtnseq_m9(
    source: Provenance, block: str, heptahydrate_g_per_l: float, *, nitrogen: bool
) -> list[MediaComponent]:
    """The RB-TnSeq E. coli M9 rows of one release's media table (Wetmore or Price).

    ``nitrogen=False`` is the release's ``M9 minimal media_noNitrogen``: ammonium chloride
    absent and 4 g/L D-glucose fixed, because the nitrogen source is the variable.
    """
    components = [
        _stated(
            "disodium hydrogen phosphate heptahydrate",
            _SALT,
            heptahydrate_g_per_l,
            _GL,
            _cite(source, f"{heptahydrate_g_per_l:g} g/L", block),
        ),
        _stated(
            "potassium dihydrogen phosphate",
            _SALT,
            3.0,
            _GL,
            _cite(source, "3 g/L", block),
        ),
        _stated("sodium chloride", _SALT, 0.5, _GL, _cite(source, "0.5 g/L", block)),
        _stated("magnesium sulfate", _SALT, 2.0, _MM, _cite(source, "2 mM", block)),
        _stated("calcium chloride", _SALT, 0.1, _MM, _cite(source, "0.1 mM", block)),
    ]
    if nitrogen:
        return [
            *components,
            _stated(
                "ammonium chloride", _NSRC, 1.0, _GL, _cite(source, "1 g/L", block)
            ),
        ]
    return [
        *components,
        _stated("D-glucose", _CSRC, 4.0, _GL, _cite(source, "4 g/L", block)),
    ]


M9_NOCARBON_WETMORE2015 = Media(
    name="M9 minimal medium, no carbon source (Wetmore 2015 'M9 minimal media_noCarbon')",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=_rbtnseq_m9(_WETMORE2015_DS1, _Q_WETMORE_M9, 13.0, nitrogen=True),
    dropouts=[_M9_ANHYDROUS_PHOSPHATE],
    provenance=[
        _cite(
            _WETMORE2015_DS1,
            "M9 minimal media_noCarbon",
            _Q_WETMORE_M9,
            note=_HYDRATE_DROPOUT_NOTE,
        ),
        _cite(
            _WETMORE2015_DS1,
            "the carbon source is the variable",
            _Q_WETMORE_GLUCOSE_EXPT,
            note="each carbon-source experiment names this medium and states its carbon "
            "source and dose in Condition_1 (here D-Glucose, 20 mM); the loader carries "
            "it as EnvironmentPhysicalPerturbation(factor=carbon_source)",
        ),
    ],
)
"""Wetmore 2015's E. coli RB-TnSeq carbon-source base (64 Keio experiments)."""

M9_NONITROGEN_WETMORE2015 = Media(
    name="M9 minimal medium, no nitrogen source, 4 g/L glucose (Wetmore 2015 "
    "'M9 minimal media_noNitrogen')",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=_rbtnseq_m9(_WETMORE2015_DS1, _Q_WETMORE_M9_NON, 13.0, nitrogen=False),
    dropouts=[_M9_ANHYDROUS_PHOSPHATE, _M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(
            _WETMORE2015_DS1,
            "M9 minimal media_noNitrogen",
            _Q_WETMORE_M9_NON,
            note=_HYDRATE_DROPOUT_NOTE,
        ),
        _cite(
            _WETMORE2015_DS1,
            "the nitrogen source is the variable",
            _Q_WETMORE_NITROGEN_EXPT,
            note="each nitrogen-source experiment names this medium and states its "
            "nitrogen source in Condition_1 (here L-Arginine, 10 mM); the loader carries "
            "it as EnvironmentPhysicalPerturbation(factor=nitrogen_source), and the "
            "ammonium chloride dropout is the medium's own 'no nitrogen' statement",
        ),
    ],
)
"""Wetmore 2015's E. coli RB-TnSeq nitrogen-source base (26 Keio experiments)."""

M9_NOCARBON_PRICE2018 = Media(
    name="M9 minimal medium, no carbon source (Price 2018 'M9 minimal media_noCarbon')",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=_rbtnseq_m9(_PRICE2018_S3, _Q_PRICE_M9, 12.8, nitrogen=True),
    dropouts=[_M9_ANHYDROUS_PHOSPHATE],
    provenance=[
        _cite(
            _PRICE2018_S3,
            "M9 minimal media_noCarbon",
            _Q_PRICE_M9,
            note="the same medium name as Wetmore 2015's Data Set S1 with 12.8 g/L of the "
            "heptahydrate where Wetmore prints 13, so the two papers' objects stay "
            "distinct. " + _HYDRATE_DROPOUT_NOTE,
        )
    ],
)
"""Price 2018's E. coli carbon-source base (60 Keio experiments in Table S5)."""

M9_NONITROGEN_PRICE2018 = Media(
    name="M9 minimal medium, no nitrogen source, 4 g/L glucose (Price 2018 "
    "'M9 minimal media_noNitrogen')",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=_rbtnseq_m9(_PRICE2018_S3, _Q_PRICE_M9_NON, 12.8, nitrogen=False),
    dropouts=[_M9_ANHYDROUS_PHOSPHATE, _M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(
            _PRICE2018_S3,
            "M9 minimal media_noNitrogen",
            _Q_PRICE_M9_NON,
            note="the nitrogen source is the variable (EnvironmentPhysicalPerturbation"
            "(factor=nitrogen_source)). " + _HYDRATE_DROPOUT_NOTE,
        )
    ],
)
"""Price 2018's E. coli nitrogen-source base (32 Keio experiments in Table S5)."""


def _fuhrer(
    name: str,
    role: MediaComponentRole,
    value: float,
    unit: ConcentrationUnit,
    read: str,
) -> MediaComponent:
    return _stated(name, role, value, unit, _cite(_FUHRER2017, read, _Q_FUHRER_M9))


M9_GLUCOSE_CASEIN_FUHRER2017 = Media(
    name="glucose M9 minimal medium with casein hydrolysate (Fuhrer 2017; ammonium "
    "sulfate, trace elements, thiamine)",
    state="liquid",
    is_synthetic=False,
    base_medium="M9",
    components=[
        _fuhrer("D-glucose", _CSRC, 4.0, _GL, "4 g per liter"),
        _mixture(
            "casein hydrolysate (N-Z Case Plus)",
            _COMPLEX,
            _UNDEFINED,
            concentration=_c(2.0, _GL),
            provenance=[_cite(_FUHRER2017, "2 g per liter", _Q_FUHRER_M9)],
        ),
        _fuhrer(
            "disodium hydrogen phosphate dihydrate",
            _SALT,
            7.52,
            _GL,
            "7.52 g per liter",
        ),
        _fuhrer("potassium dihydrogen phosphate", _SALT, 3.0, _GL, "3 g per liter"),
        _fuhrer("sodium chloride", _SALT, 0.5, _GL, "0.5 g per liter"),
        _fuhrer("ammonium sulfate", _NSRC, 2.5, _GL, "2.5 g per liter"),
        _fuhrer("calcium chloride dihydrate", _SALT, 14.7, _UGML, "14.7 mg per liter"),
        _fuhrer(
            "magnesium sulfate heptahydrate", _SALT, 246.5, _UGML, "246.5 mg per liter"
        ),
        _fuhrer(
            "iron(III) chloride hexahydrate", _TRACE, 16.2, _UGML, "16.2 mg per liter"
        ),
        _fuhrer("zinc sulfate heptahydrate", _TRACE, 0.18, _UGML, "180 ug per liter"),
        _fuhrer(
            "copper(II) chloride dihydrate", _TRACE, 0.12, _UGML, "120 ug per liter"
        ),
        _fuhrer(
            "manganese(II) sulfate monohydrate", _TRACE, 0.12, _UGML, "120 ug per liter"
        ),
        _fuhrer(
            "cobalt(II) chloride hexahydrate", _TRACE, 0.18, _UGML, "180 ug per liter"
        ),
        _fuhrer("thiamine hydrochloride", _VITAMIN, 1.0, _UGML, "1 mg per liter"),
    ],
    dropouts=[_M9_ANHYDROUS_PHOSPHATE, _M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(
            _FUHRER2017,
            "glucose minimal medium supplemented with casein hydrolysate",
            _Q_FUHRER_M9,
            note="is_synthetic is False because N-Z Case Plus is an undefined digest. "
            + _HYDRATE_DROPOUT_NOTE
            + "; "
            + _AMMONIUM_SULFATE_DROPOUT_NOTE,
        )
    ],
)
"""Fuhrer 2017's metabolome medium: every Keio mutant was grown on it (row 1)."""

M9_SCHMIDT2016 = Media(
    name="M9 minimal medium, no carbon source (Schmidt 2016; ammonium sulfate, trace "
    "elements, thiamine, FeCl3)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        _stated(
            "disodium hydrogen phosphate",
            _SALT,
            42.2,
            _MM,
            _cite(
                _SCHMIDT2016,
                "211 mM in the 5x stock, 200 mL per liter: 42.2 mM",
                _Q_SCHMIDT16_SALTS,
            ),
        ),
        _stated(
            "potassium dihydrogen phosphate",
            _SALT,
            22.0,
            _MM,
            _cite(
                _SCHMIDT2016,
                "110 mM in the 5x stock, 200 mL per liter: 22 mM",
                _Q_SCHMIDT16_SALTS,
            ),
        ),
        _stated(
            "sodium chloride",
            _SALT,
            8.56,
            _MM,
            _cite(
                _SCHMIDT2016,
                "42.8 mM in the 5x stock, 200 mL per liter: 8.56 mM",
                _Q_SCHMIDT16_SALTS,
            ),
        ),
        _stated(
            "ammonium sulfate",
            _NSRC,
            11.34,
            _MM,
            _cite(
                _SCHMIDT2016,
                "56.7 mM in the 5x stock, 200 mL per liter: 11.34 mM",
                _Q_SCHMIDT16_SALTS,
            ),
        ),
        _stated(
            "zinc sulfate",
            _TRACE,
            6.3,
            _UM,
            _cite(
                _SCHMIDT2016,
                "0.63 mM in the trace stock, 10 mL per liter: 6.3 uM",
                _Q_SCHMIDT16_TRACE,
            ),
        ),
        _stated(
            "copper(II) chloride",
            _TRACE,
            7.0,
            _UM,
            _cite(
                _SCHMIDT2016,
                "0.7 mM in the trace stock, 10 mL per liter: 7 uM",
                _Q_SCHMIDT16_TRACE,
            ),
        ),
        _stated(
            "manganese sulfate",
            _TRACE,
            7.1,
            _UM,
            _cite(
                _SCHMIDT2016,
                "0.71 mM in the trace stock, 10 mL per liter: 7.1 uM",
                _Q_SCHMIDT16_TRACE,
            ),
        ),
        _stated(
            "cobalt chloride",
            _TRACE,
            7.6,
            _UM,
            _cite(
                _SCHMIDT2016,
                "0.76 mM in the trace stock, 10 mL per liter: 7.6 uM",
                _Q_SCHMIDT16_TRACE,
            ),
        ),
        _stated(
            "calcium chloride",
            _SALT,
            0.1,
            _MM,
            _cite(_SCHMIDT2016, "1 mL of 0.1 M per liter: 0.1 mM", _Q_SCHMIDT16_CACL2),
        ),
        _stated(
            "magnesium sulfate",
            _SALT,
            1.0,
            _MM,
            _cite(_SCHMIDT2016, "1 mL of 1 M per liter: 1 mM", _Q_SCHMIDT16_MGSO4),
        ),
        _defined(
            "thiamine",
            _VITAMIN,
            provenance=[
                _cite(
                    _SCHMIDT2016,
                    "2 mL of a 500x stock per liter; stock concentration lost in the OCR",
                    _Q_SCHMIDT16_THIAMINE,
                )
            ],
            note="the amount is an OPEN GAP on purpose: the pinned paper.md lost the "
            "stock concentration (it reads '.4.m M'). The PDF text layer of the same key "
            "reads '(1.4 mM, in H2O, filter sterilized)', which at 2 mL per liter is "
            "2.8 uM, but no pinned quote carries that number, so it is not recorded "
            "until the OCR is redone",
        ),
        _stated(
            "iron(III) chloride",
            _TRACE,
            60.0,
            _UM,
            _cite(_SCHMIDT2016, "0.6 mL of 0.1 M per liter: 60 uM", _Q_SCHMIDT16_FECL3),
        ),
    ],
    dropouts=[_M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(
            _SCHMIDT2016,
            "carbon-source-free M9 minimal medium",
            _Q_SCHMIDT16_CARBON,
            note="every amount is the stated stock diluted by the stated volume into one "
            "liter (200 mL of the 5x salts, 10 mL of trace elements, 1 mL of each "
            "salt stock, 0.6 mL of FeCl3); the OCR garbles 'trace elements' to 'te "
            "elemen' and 'of' to 'f'. " + _AMMONIUM_SULFATE_DROPOUT_NOTE,
        ),
        _cite(
            _SCHMIDT2016,
            "the carbon source is the variable",
            _Q_SCHMIDT16_GLUCOSE,
            note="eleven carbon sources at stated concentrations (glucose 5 g/L); the "
            "loader carries each as EnvironmentPhysicalPerturbation(factor=carbon_source)",
        ),
    ],
)
"""Schmidt 2016's M9 base for the proteome map's minimal-medium conditions (row 32)."""


# --------------------------------------------------------------------------- #
# The P. putida "M9" of the JBEI / NREL isoprenol campaigns: ammonium sulfate as the
# nitrogen salt and a commercial trace-metal solution. Menasalvas 2025 names it ("NREL M9"
# or "Modified M9"); each paper states a different nitrogen level, trace-metal volume or
# extra buffer, so each paper has its own object.
# --------------------------------------------------------------------------- #
_TEKNOVA_T1001 = "trace metal solution (Teknova T1001)"


def _nrel_salts(
    source: Provenance,
    quote: str,
    *,
    ammonium_sulfate: Concentration,
    ammonium_sulfate_read: str,
    phosphate_read: str = "6.8 g/L",
    dihydrogen_read: str = "3 g/L",
    dihydrogen_note: str | None = None,
) -> list[MediaComponent]:
    """The five salts every NREL-M9 paper states at 6.8 / 3 / 0.5 g/L, 2 mM, 0.1 mM."""
    return [
        _defined(
            "ammonium sulfate",
            _NSRC,
            concentration=ammonium_sulfate,
            provenance=[_cite(source, ammonium_sulfate_read, quote)],
        ),
        _stated(
            "disodium hydrogen phosphate",
            _SALT,
            6.8,
            _GL,
            _cite(source, phosphate_read, quote),
        ),
        _stated(
            "potassium dihydrogen phosphate",
            _SALT,
            3.0,
            _GL,
            _cite(source, dihydrogen_read, quote),
            note=dihydrogen_note,
        ),
        _stated("sodium chloride", _SALT, 0.5, _GL, _cite(source, "0.5 g/L", quote)),
        _stated("magnesium sulfate", _SALT, 2.0, _MM, _cite(source, "2 mM", quote)),
        _stated("calcium chloride", _SALT, 0.1, _MM, _cite(source, "0.1 mM", quote)),
    ]


M9_NREL_DESIQUEIRA2025 = Media(
    name="M9 (NREL type: ammonium sulfate + Teknova trace metals), no carbon source "
    "(de Siqueira 2025)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        *_nrel_salts(
            _DESIQUEIRA2025,
            _Q_DESIQUEIRA_M9,
            ammonium_sulfate=_c(2.0, _GL),
            ammonium_sulfate_read="2 g/L",
        ),
        _mixture(
            _TEKNOVA_T1001,
            _TRACE,
            _DEFERRED,
            concentration=_c(0.05, _VV),
            provenance=[
                _cite(_DESIQUEIRA2025, "500 uL per 1 L: 0.05% v/v", _Q_DESIQUEIRA_M9)
            ],
            note="a vendor solution; the source writes 'Product No. 1001, Tekova Inc', "
            "the Teknova T1001 the other NREL-M9 papers name",
        ),
    ],
    dropouts=[_M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(_DESIQUEIRA2025, "minimal salt (M9) medium", _Q_DESIQUEIRA_M9),
        _cite(
            _DESIQUEIRA2025,
            "the carbon source is the variable",
            _Q_DESIQUEIRA_CARBON,
            note="acetate (50 mM) or glucose (1% w/v) as the sole carbon source; the "
            "loader carries it as EnvironmentPhysicalPerturbation(factor=carbon_source). "
            + _AMMONIUM_SULFATE_DROPOUT_NOTE,
        ),
    ],
)
"""de Siqueira 2025's P. putida tolerization medium (row 14)."""

M9_NREL_LIM2025 = Media(
    name="M9 (NREL type: ammonium sulfate + 2000x trace elements) + 4 g/L glucose "
    "(Lim 2025)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        *_nrel_salts(
            _LIM2025,
            _Q_LIM25_M9,
            ammonium_sulfate=_c(2.0, _GL),
            ammonium_sulfate_read="2 g/L",
        ),
        _mixture(
            "2000x trace element solution (Lim 2020; Linger 2014)",
            _TRACE,
            _DEFERRED,
            concentration=_c(0.05, _VV),
            provenance=[_cite(_LIM2025, "500 uL/L: 0.05% v/v", _Q_LIM25_M9)],
            note="the composition is deferred to the two papers the source cites for it, "
            "neither of which is mirrored",
            defers_to=[_LIM2020, _LINGER2014],
        ),
        _stated(
            "D-glucose",
            _CSRC,
            4.0,
            _GL,
            _cite(
                _LIM2025,
                "4 g/L unless otherwise stated",
                _Q_LIM25_GLUCOSE,
                note="the default carbon source; a condition stating another carbon "
                "source is a different medium",
            ),
        ),
    ],
    dropouts=[_M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(
            _LIM2025,
            "modified M9 minimal medium",
            _Q_LIM25_M9,
            note=_AMMONIUM_SULFATE_DROPOUT_NOTE,
        )
    ],
)
"""Lim 2025's P. putida isoprenol-TALE medium (row 16)."""

# --------------------------------------------------------------------------- #
# Lim 2022 (row 7) states the SAME five NREL salts as Lim 2025 and, uniquely among the
# NREL-M9 papers, states its 2000x trace solution's full composition rather than
# deferring it to Lim 2020 / Linger 2014. So its trace component is a sub-mix whose
# recipe is QUOTED on the component, not a deferral; it is kept as one component rather
# than expanded into ten, because six of the ten labels have no compound-identity row
# (measured 2026-10-07: manganese chloride tetrahydrate, cobalt chloride hexahydrate,
# copper sulfate dihydrate, sodium molybdate dihydrate, potassium iodide and disodium
# EDTA are UNRESOLVED_PUBLIC, while zinc sulfate heptahydrate, calcium chloride
# dihydrate, iron(II) sulfate heptahydrate and boric acid resolve). A partial expansion
# would assert a partial recipe, so the component carries the whole stated one.
# Two keys, because the compendium uses both: the medium as stated (4 g/L glucose) and
# the aromatic project's version, whose carbon source replaces the glucose.
# --------------------------------------------------------------------------- #
_LIM2022 = Provenance(
    citation_key="limMachinelearningPseudomonasPutida2022",
    sha256="64e1df3103b221051fb52e9d62afb39582367e20996bedc46f4d6f23e73f4750",
    source_uri="si/si1.docx",
    method="paragraph text of word/document.xml (docx_paragraphs)",
    page="Supplementary Method 1. Transcriptome sequencing (RNA-seq)",
)
_Q_LIM22_M9 = (
    "Briefly, cells were cultured in either LB medium (10 g/L tryptone, 5 g/L yeast "
    "extract, 10 g/L NaCl) or the modified minimal M9 medium. The minimal medium "
    "contains 4 g/L glucose, 2 g/L (NH4)2SO4, 6.8 g/L Na2HPO4, 3 g/L KH2PO4, 0.5 g/L "
    "NaCl, 2 mM MgSO4, 0.1 mM CaCl2, 500 μL/L 2000× trace element solution)."
)
_Q_LIM22_TRACE = (
    "The composition of the trace element solution is 4.5 g/L ZnSO4·7H2O, 0.7 g/L "
    "MnCl2·4H2O, 0.3 g/L CoCl2·6H2O, 0.2 g/L CuSO4·2H2O, 0.4 g/L "
    "Na2MoO4·2H2O, 4.5 g/L CaCl2·2H2O, 3.0 g/L FeSO4·7H2O, 1.0 g/L "
    "H3BO3, 0.1 g/L KI, 15 g/L disodium ethylenediaminetetraacetate."
)
_Q_LIM22_AROMATIC = (
    "Cells were grown in the M9 medium with 2.5 g/L of either coumarate, ferulate, a "
    "mixture of coumarate and ferulate, or glucose."
)

_LIM2022_TRACE_SOLUTION = _mixture(
    "2000x trace element solution (Lim 2022)",
    _TRACE,
    _DEFERRED,
    concentration=_c(0.05, _VV),
    provenance=[
        _cite(_LIM2022, "500 uL/L: 0.05% v/v", _Q_LIM22_M9),
        _cite(
            _LIM2022,
            {
                "ZnSO4.7H2O": "4.5 g/L",
                "MnCl2.4H2O": "0.7 g/L",
                "CoCl2.6H2O": "0.3 g/L",
                "CuSO4.2H2O": "0.2 g/L",
                "Na2MoO4.2H2O": "0.4 g/L",
                "CaCl2.2H2O": "4.5 g/L",
                "FeSO4.7H2O": "3.0 g/L",
                "H3BO3": "1.0 g/L",
                "KI": "0.1 g/L",
                "disodium ethylenediaminetetraacetate": "15 g/L",
            },
            _Q_LIM22_TRACE,
            note="the 2000x stock in g/L; at 500 uL/L each is diluted 2000-fold",
        ),
    ],
    note="the composition IS stated (the second quote) and is kept here rather than "
    "expanded into ten components, because six of the ten labels have no "
    "compound-identity row; a partial expansion would assert a partial recipe. This is "
    "``composition_deferred`` for that reason, not for an unmirrored source",
)


def _lim2022_salts() -> list[MediaComponent]:
    """The five salts and the nitrogen salt Lim 2022's own sentence states."""
    return [
        *_nrel_salts(
            _LIM2022,
            _Q_LIM22_M9,
            ammonium_sulfate=_c(2.0, _GL),
            ammonium_sulfate_read="2 g/L",
        ),
        _LIM2022_TRACE_SOLUTION,
    ]


M9_NREL_LIM2022 = Media(
    name="M9 (NREL type: ammonium sulfate + 2000x trace elements, composition stated) "
    "+ 4 g/L glucose (Lim 2022)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        *_lim2022_salts(),
        _stated("D-glucose", _CSRC, 4.0, _GL, _cite(_LIM2022, "4 g/L", _Q_LIM22_M9)),
    ],
    dropouts=[_M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(
            _LIM2022,
            "the modified minimal M9 medium",
            _Q_LIM22_M9,
            note=_AMMONIUM_SULFATE_DROPOUT_NOTE,
        )
    ],
)
"""Lim 2022's in-house minimal medium as stated, 4 g/L glucose (row 7)."""

M9_NREL_NOCARBON_LIM2022 = Media(
    name="M9 (NREL type: ammonium sulfate + 2000x trace elements, composition stated), "
    "no carbon source (Lim 2022 aromatic project)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=_lim2022_salts(),
    dropouts=[_M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(
            _LIM2022,
            "the modified minimal M9 medium",
            _Q_LIM22_M9,
            note=_AMMONIUM_SULFATE_DROPOUT_NOTE,
        ),
        _cite(
            _LIM2022,
            "the carbon source is the variable",
            _Q_LIM22_AROMATIC,
            note="the aromatic project replaces the base's 4 g/L glucose with 2.5 g/L "
            "of p-coumarate, ferulate, both, or glucose, so the carbon source is an "
            "EnvironmentPhysicalPerturbation(factor=carbon_source) and is left out here",
        ),
    ],
)
"""Lim 2022's minimal medium without its carbon source, the aromatic base (row 7)."""

_KANG_KH2PO4_NOTE = "the OCR renders the '3 g/L' of KH2PO4 as '3 \\gimel A'"

_KANG_NREL_SALTS = _nrel_salts(
    _KANG2026,
    _Q_KANG_M9,
    ammonium_sulfate=_c(2.0, _GL),
    ammonium_sulfate_read="2 g/L",
    dihydrogen_note=_KANG_KH2PO4_NOTE,
)
_KANG_TRACE = _mixture(
    "trace element solution (Teknova)",
    _TRACE,
    _DEFERRED,
    concentration=_c(0.1, _VV),
    provenance=[_cite(_KANG2026, "1 mL/L: 0.1% v/v", _Q_KANG_M9)],
    note="a vendor solution; the source names Teknova but no catalog number",
)

M9_NREL_KANG2026 = Media(
    name="M9 (NREL type: 2 g/L ammonium sulfate + Teknova trace elements), no carbon "
    "source (Kang 2026)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[*_KANG_NREL_SALTS, _KANG_TRACE],
    dropouts=[_M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(_KANG2026, "M9 medium", _Q_KANG_M9),
        _cite(
            _KANG2026,
            "the carbon source is the variable",
            _Q_KANG_SUGAR,
            note="20 g/L total sugar, glucose alone or 2:1 glucose:xylose, so the sugar "
            "is carried as EnvironmentPhysicalPerturbation(factor=carbon_source). "
            + _AMMONIUM_SULFATE_DROPOUT_NOTE,
        ),
    ],
)
"""Kang 2026's P. putida M9 (row 18)."""

M9_NREL_HIGH_N_KANG2026 = Media(
    name="modified M9 (NREL type, 40 mM ammonium sulfate), no carbon source (Kang 2026)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        _stated(
            "ammonium sulfate",
            _NSRC,
            40.0,
            _MM,
            _cite(_KANG2026, "increased to 40 mM", _Q_KANG_MODIFIED),
        ),
        *_KANG_NREL_SALTS[1:],
        _KANG_TRACE,
    ],
    dropouts=[_M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(_KANG2026, "modified M9", _Q_KANG_MODIFIED),
        _cite(
            _KANG2026,
            "the carbon source is the variable",
            _Q_KANG_SUGAR,
            note=_AMMONIUM_SULFATE_DROPOUT_NOTE,
        ),
    ],
)
"""Kang 2026's "modified M9": the M9 above with the ammonium sulfate raised to 40 mM."""


def _kang_mops(
    name: str,
    role: MediaComponentRole,
    value: float,
    unit: ConcentrationUnit,
    read: str,
) -> MediaComponent:
    return _stated(name, role, value, unit, _cite(_KANG2026, read, _Q_KANG_MOPS))


M9_MOPS_KANG2026 = Media(
    name="M9-MOPS (M9 salts + 75 mM MOPS + thiamine + FeSO4 + micronutrients), no "
    "carbon source (Kang 2026)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        _kang_mops("disodium hydrogen phosphate", _SALT, 6.78, _GL, "6.78 g/L"),
        _kang_mops("potassium dihydrogen phosphate", _SALT, 3.0, _GL, "3 g/L"),
        _kang_mops("ammonium chloride", _NSRC, 1.0, _GL, "1 g/L"),
        _kang_mops("sodium chloride", _SALT, 0.5, _GL, "0.5 g/L"),
        _kang_mops("3-(N-morpholino)propanesulfonic acid", _BUFFER, 75.0, _MM, "75 mM"),
        _kang_mops("thiamine", _VITAMIN, 1.0, _UGML, "1 mg/L"),
        _kang_mops("iron(II) sulfate", _TRACE, 10.0, _NM, "10 nM"),
        _kang_mops("ammonium heptamolybdate", _TRACE, 3e-08, _M, "3*10^-8 M"),
        _kang_mops("boric acid", _TRACE, 4e-06, _M, "4*10^-6 M"),
        _kang_mops("cobalt chloride", _TRACE, 3e-07, _M, "3*10^-7 M"),
        _kang_mops("copper sulfate", _TRACE, 1.5e-07, _M, "1.5*10^-7 M"),
        _kang_mops("MnCl2", _TRACE, 8e-07, _M, "8*10^-7 M"),
        _kang_mops("zinc sulfate", _TRACE, 1e-07, _M, "1*10^-7 M"),
        _kang_mops("magnesium sulfate", _SALT, 2.0, _MM, "2 mM"),
        _kang_mops("calcium chloride", _SALT, 0.1, _MM, "0.1 mM"),
    ],
    provenance=[
        _cite(
            _KANG2026,
            "M9-MOPS",
            _Q_KANG_MOPS,
            note="the optional 1 or 5 g/L yeast extract the next sentence allows is a "
            "per-condition addition, not part of this medium",
        ),
        _cite(_KANG2026, "the carbon source is the variable", _Q_KANG_SUGAR),
    ],
)
"""Kang 2026's M9-MOPS: the classic M9 salts with the Neidhardt micronutrients (row 18)."""

M9_NREL_CARRUTHERS2025 = Media(
    name="M9-NREL + 20 g/L glucose (10 mM ammonium sulfate + Teknova T1001; "
    "Carruthers 2025)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        _stated(
            "D-glucose",
            _CSRC,
            20.0,
            _GL,
            _cite(_CARRUTHERS2025, "20 g/L", _Q_CARRUTHERS_M9),
        ),
        _stated(
            "sodium chloride",
            _SALT,
            0.5,
            _GL,
            _cite(_CARRUTHERS2025, "0.5 g/L", _Q_CARRUTHERS_M9),
            note="the OCR renders '0.5 g/L' as '0.58/L'; the PDF text layer reads 0.5 g/L",
        ),
        _stated(
            "disodium hydrogen phosphate",
            _SALT,
            6.8,
            _GL,
            _cite(_CARRUTHERS2025, "6.8 g/L", _Q_CARRUTHERS_M9),
        ),
        _stated(
            "potassium dihydrogen phosphate",
            _SALT,
            3.0,
            _GL,
            _cite(_CARRUTHERS2025, "3 g/L", _Q_CARRUTHERS_M9),
        ),
        _stated(
            "calcium chloride",
            _SALT,
            100.0,
            _UM,
            _cite(_CARRUTHERS2025, "100 uM", _Q_CARRUTHERS_M9),
        ),
        _stated(
            "magnesium sulfate",
            _SALT,
            2.0,
            _MM,
            _cite(_CARRUTHERS2025, "2 mM", _Q_CARRUTHERS_M9),
        ),
        _stated(
            "ammonium sulfate",
            _NSRC,
            10.0,
            _MM,
            _cite(_CARRUTHERS2025, "10 mM", _Q_CARRUTHERS_M9),
        ),
        _mixture(
            _TEKNOVA_T1001,
            _TRACE,
            _DEFERRED,
            provenance=[
                _cite(
                    _CARRUTHERS2025, "500 uL; volume basis not stated", _Q_CARRUTHERS_M9
                )
            ],
            note="the amount is an OPEN GAP: the source states 500 uL of the solution "
            "without the medium volume it goes into, so no concentration is recorded",
        ),
    ],
    dropouts=[_M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(_CARRUTHERS2025, "M9-NREL medium", _Q_CARRUTHERS_M9),
        _cite(
            _CARRUTHERS2025,
            "M9-NREL",
            _Q_CARRUTHERS_NREL,
            note="the plasmid-maintenance antibiotics and the 2 g/L L-arabinose inducer "
            "are per-strain, per-condition additions, not part of the medium. "
            + _AMMONIUM_SULFATE_DROPOUT_NOTE,
        ),
    ],
)
"""Carruthers 2025's P. putida isoprenol production medium (row 6)."""

M9_NREL_MOPS_MENASALVAS2025 = Media(
    name="NREL M9 + 30 mM MOPS + 2% glucose (70 mM ammonium sulfate + 1X Teknova "
    "T1001; Menasalvas 2025)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        _stated(
            "disodium hydrogen phosphate",
            _SALT,
            47.9,
            _MM,
            _cite(_MENASALVAS2025, "47.9 mM", _Q_MENASALVAS_M9),
        ),
        _stated(
            "potassium dihydrogen phosphate",
            _SALT,
            22.0,
            _MM,
            _cite(_MENASALVAS2025, "22 mM", _Q_MENASALVAS_M9),
        ),
        _stated(
            "sodium chloride",
            _SALT,
            8.56,
            _MM,
            _cite(_MENASALVAS2025, "8.56 mM", _Q_MENASALVAS_M9),
        ),
        _stated(
            "magnesium sulfate",
            _SALT,
            2.0,
            _MM,
            _cite(_MENASALVAS2025, "2 mM", _Q_MENASALVAS_M9),
        ),
        _stated(
            "calcium chloride",
            _SALT,
            100.0,
            _UM,
            _cite(_MENASALVAS2025, "100 uM", _Q_MENASALVAS_M9),
        ),
        _mixture(
            _TEKNOVA_T1001,
            _TRACE,
            _DEFERRED,
            provenance=[
                _cite(_MENASALVAS2025, "1X working strength", _Q_MENASALVAS_M9)
            ],
            note="the amount is an OPEN GAP: '1X' is a working strength of a vendor "
            "stock whose fold the source does not state, and the library has no 'X' unit",
        ),
        _stated(
            "D-glucose",
            _CSRC,
            2.0,
            _PCT,
            _cite(_MENASALVAS2025, "2%", _Q_MENASALVAS_M9),
        ),
        _stated(
            "ammonium sulfate",
            _NSRC,
            70.0,
            _MM,
            _cite(
                _MENASALVAS2025,
                "70 mM",
                _Q_MENASALVAS_M9,
                note="read twice, from the OCR and the PDF text layer, because 70 mM is "
                "far above the 10 to 15 mM the other NREL-M9 papers state",
            ),
        ),
        _stated(
            "3-(N-morpholino)propanesulfonic acid",
            _BUFFER,
            30.0,
            _MM,
            _cite(_MENASALVAS2025, "30 mM", _Q_MENASALVAS_M9),
        ),
    ],
    dropouts=[_M9_AMMONIUM_CHLORIDE],
    provenance=[
        _cite(
            _MENASALVAS2025,
            "M9 medium at the 1X working concentration",
            _Q_MENASALVAS_M9,
            note="the pH 7.0 the sentence states is not a Media field; it rides as "
            "EnvironmentPhysicalPerturbation(factor=ph). "
            + _AMMONIUM_SULFATE_DROPOUT_NOTE,
        ),
        _cite(_MENASALVAS2025, "NREL M9", _Q_MENASALVAS_NREL),
    ],
)
"""Menasalvas 2025's P. putida isoprenol medium, NREL M9 with MOPS (row 20)."""

# --------------------------------------------------------------------------- #
# Banerjee 2025 (row 39, P. putida KT2440): the one bacterial medium in this library
# whose SALTS are a composition DEFERRED, not a stated recipe. The Methods name the
# formulation and hand its amounts to an earlier paper -- "Engineered strains were grown
# in a modified M9 minimal medium as previously described21" -- and reference 21 (Eng
# et al. 2023, Cell Rep. 42, 113087) is not mirrored. The three carbon sources it
# DOES state are handed to the figure legends, so they are not components either: the loader carries each
# condition's carbon regime as an EnvironmentPhysicalPerturbation(factor=carbon_source),
# which is the de Siqueira 2025 convention. What is left is one honest object: M9-based,
# salts unstated, carbon-free.
# --------------------------------------------------------------------------- #
_Q_BANERJEE_M9 = (
    "Engineered strains were grown in a modified M9 minimal medium as previously "
    "described21, and $\\boldsymbol { p }$ - CA (Sigma-Aldrich, Product No. C9008), "
    "L-malic acid sodium salt (Sigma-Aldrich, Product No. M1125) and D-alanine "
    "(Sigma-Aldrich, Product No. A7377) were used at the concentrations indicated in "
    "the figure legends."
)
_Q_BANERJEE_PROTEOMICS_MEDIA = (
    "The D1b_gf strains designed in this study were grown in triplicates in M9 "
    "$6 0 \\ : \\mathrm { m M } \\ : p – \\mathrm { C } A$ , or M9 50 mM "
    "$\\boldsymbol { p }$ -CA supplemented with $7 0 \\mathrm { m M }$ D-alanine and "
    "$7 0 \\mathrm { m M L }$ -malate when indicated, using $1 0 \\mathrm { m L }$ "
    "culture tubes."
)
_ENG_BANERJEE2023 = (
    "Eng et al. 2023, Cell Rep. 42, 113087 (the paper's reference 21, which its own "
    "Supplementary Table 1 cites as 'Eng andBanerjee,2023'); not mirrored"
)

M9_DEFERRED_BANERJEE2025 = Media(
    name="modified M9 minimal medium, composition deferred to Eng et al. 2023, "
    "no carbon source (Banerjee 2025)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9_DEFERRED_BANERJEE2025",
    components=[
        _mixture(
            "modified M9 minimal medium (amounts deferred to Eng et al. 2023)",
            MediaComponentRole.other,
            _DEFERRED,
            provenance=[
                _cite(
                    _BANERJEE2025,
                    "composition deferred to reference 21 (Eng et al. 2023)",
                    _Q_BANERJEE_M9,
                )
            ],
            note="the source states that the medium is a MODIFIED M9 and gives no "
            "amounts for any of its salts, so no gram or millimolar figure is copied "
            "in from another paper's M9. It is therefore its OWN base rather than a "
            "derivative of the stated M9 salts (the SM_DEFERRED treatment): a "
            "derivative must restate or drop every base component, and this medium can "
            "do neither without asserting amounts the source withheld. The PRODUCTION "
            "runs add 30 mM MOPS at pH 7 and 1.5 percent w/v L-arabinose on top of the "
            "same base; no medium object carries them because this paper releases no "
            "titer and no growth rate, so nothing is served from a production run",
            defers_to=[_ENG_BANERJEE2023],
        )
    ],
    provenance=[
        _cite(_BANERJEE2025, "modified M9 minimal medium", _Q_BANERJEE_M9),
        _cite(
            _BANERJEE2025,
            "the carbon source is the variable",
            _Q_BANERJEE_PROTEOMICS_MEDIA,
            note="60 mM p-CA for the promoter-variant proteomes and 50 mM p-CA + 70 mM "
            "D-alanine + 70 mM L-malate for the cross-feeding proteomes; the loader "
            "carries each as EnvironmentPhysicalPerturbation(factor=carbon_source)",
        ),
    ],
)
"""Banerjee 2025's carbon-free P. putida M9, salts deferred to Eng 2023 (row 39)."""


# --------------------------------------------------------------------------- #
# Fang 2025 (row 43, E. coli MG1655(DE3)): the "modified M9" of the CRISPRi-FACS screen
# and of every flask fermentation. Stated in full, which is why it is a recipe and not a
# deferral, and distinctive on three counts: 2 g/L yeast extract makes it SEMI-defined
# rather than minimal, 30 g/L glycerol is the screen's carbon source, and 0.1% (v/v)
# Triton X-100 is a surfactant no other medium in this library carries. The glucose
# variant the same sentence offers ("or 30 g/L glucose, specifically used in Fig. 4") is
# NOT a second object: no screen record uses it, and the one figure that does is a
# titer panel this loader does not read.
#
# The phosphate is weighed as the DODECAHYDRATE, a different reagent with its own
# InChIKey, so this medium does not derive from ``M9`` by dropout; it states its own
# salts against the ``M9`` label. The trace solution is a 1000-fold dilution of a stock
# whose five salts the Methods DO print, but the diluted amounts are a derivation rather
# than a stated value, so it is carried as one mixture at the stated 1 mL/L with the
# stock recipe in its note -- the Menasalvas convention for a trace mixture, without
# that row's open gap, because here the composition is published.
# --------------------------------------------------------------------------- #
_FANG2025 = Provenance(
    citation_key="fangGenomescaleCRISPRiScreen2025",
    sha256="93d4303ce7aad2cec712ad09fd41319e08f4b735dc8dfc69659d55e590f2ee3d",
    source_uri="paper.md",
)
_Q_FANG_M9 = (
    "Modified M9 medium58 $\\mathrm { ( pH } 7 . 2 \\mathrm { ) }$ for tube and flask "
    "fermentation was prepared as follows: $1 7 . 1 \\mathrm { g } \\mathrm { L } ^ { -1 "
    "}$ $\\mathrm { N a _ { 2 } H P O _ { 4 } } { \\cdot } 1 2 \\mathrm { H } _ { 2 } O$ "
    ", $3 { \\bf g } { \\bf L } ^ { -1 } { \\bf \\ K H } _ { 2 } { \\bf P O } _ { 4 } ,$ "
    "$0 . 5 \\mathrm { g L } ^ { -1 }$ NaCl, $2 { \\bf g } { \\bf L } ^ { -1 } \\ { \\sf "
    "N H } _ { 4 } { \\bf C } { \\bf l }$ , $2 { \\tt g } { \\tt L } ^ { -1 }$ yeast "
    "extract, $3 0 { \\bf g } { \\bf L } ^ { -1 }$ glycerol (or $3 0 { \\bf g } { \\bf L "
    "} ^ { -1 }$ glucose, specifically used in Fig. 4), $0 . 2 5 { \\bf g } { \\bf L } ^ "
    "{ -1 }$ $\\mathrm { M g S O _ { 4 } } { \\cdot } 7 \\mathrm { H } _ { 2 } \\mathrm "
    "{ O }$ , $\\mathrm { 1 1 . 1 m g L ^ { -1 } \\ C a C l _ { 2 } } ,$ $1 0 \\mathrm { "
    "m g L ^ { -1 } }$ thiamine,"
)
_Q_FANG_TRACE = (
    "$0 . 1 \\%$ (v/v) Triton-X100, and $1 \\mathsf { m L L } ^ { -1 }$ metal trace "
    "stock solution."
)
_FANG_TRACE_STOCK = (
    "a 1 mL/L dilution of the stock the same Methods paragraph states: 27 g/L "
    "FeCl3.6H2O, 2 g/L ZnCl2, 2 g/L Na2MoO4.2H2O, 1.9 g/L CuSO4.5H2O and 0.5 g/L "
    "H3BO3. The five salts are published, so this is not a composition gap; the "
    "DILUTED amounts would be a derivation, so the component carries the stated 1 mL/L "
    "and the stock recipe stays verbatim here"
)

M9_MODIFIED_FANG2025 = Media(
    name="modified M9 + 2 g/L yeast extract + 30 g/L glycerol + 0.1% Triton X-100 "
    "(pH 7.2; Fang 2025)",
    state="liquid",
    is_synthetic=False,
    base_medium="M9",
    components=[
        _stated(
            "disodium hydrogen phosphate dodecahydrate",
            _SALT,
            17.1,
            _GL,
            _cite(_FANG2025, "17.1 g/L", _Q_FANG_M9),
        ),
        _stated(
            "potassium dihydrogen phosphate",
            _SALT,
            3.0,
            _GL,
            _cite(_FANG2025, "3 g/L", _Q_FANG_M9),
        ),
        _stated(
            "sodium chloride", _SALT, 0.5, _GL, _cite(_FANG2025, "0.5 g/L", _Q_FANG_M9)
        ),
        _stated(
            "ammonium chloride", _NSRC, 2.0, _GL, _cite(_FANG2025, "2 g/L", _Q_FANG_M9)
        ),
        _mixture(
            "yeast extract",
            MediaComponentRole.complex_ingredient,
            _UNDEFINED,
            concentration=_c(2.0, _GL),
            provenance=[_cite(_FANG2025, "2 g/L", _Q_FANG_M9)],
            note="2 g/L yeast extract is what makes this medium semi-defined rather "
            "than minimal, so is_synthetic is False",
        ),
        _stated("glycerol", _CSRC, 30.0, _GL, _cite(_FANG2025, "30 g/L", _Q_FANG_M9)),
        _stated(
            "magnesium sulfate heptahydrate",
            _SALT,
            0.25,
            _GL,
            _cite(_FANG2025, "0.25 g/L", _Q_FANG_M9),
        ),
        _stated(
            "calcium chloride",
            _SALT,
            0.0111,
            _GL,
            _cite(
                _FANG2025,
                "11.1 mg/L",
                _Q_FANG_M9,
                note="stated in mg/L; the library has no mg/L unit, so the same amount "
                "is carried in g/L",
            ),
        ),
        _stated(
            "thiamine",
            _VITAMIN,
            0.01,
            _GL,
            _cite(
                _FANG2025,
                "10 mg/L",
                _Q_FANG_M9,
                note="stated in mg/L; the library has no mg/L unit, so the same amount "
                "is carried in g/L",
            ),
        ),
        _mixture(
            "Triton X-100",
            MediaComponentRole.other,
            _UNDEFINED,
            concentration=_c(0.1, ConcentrationUnit.percent_v_v),
            provenance=[_cite(_FANG2025, "0.1% (v/v)", _Q_FANG_TRACE)],
            note="a polydisperse octylphenol ethoxylate, so it is a preparation rather "
            "than one substance; no other medium in this library carries a surfactant",
        ),
        _mixture(
            "metal trace stock solution",
            _TRACE,
            _DEFERRED,
            concentration=_c(0.1, ConcentrationUnit.percent_v_v),
            provenance=[_cite(_FANG2025, "1 mL/L", _Q_FANG_TRACE)],
            note=_FANG_TRACE_STOCK,
        ),
    ],
    dropouts=[_M9_ANHYDROUS_PHOSPHATE],
    provenance=[
        _cite(
            _FANG2025,
            "modified M9 medium (pH 7.2)",
            _Q_FANG_M9,
            note=_HYDRATE_DROPOUT_NOTE,
        )
    ],
)
"""Fang 2025's modified M9: the CRISPRi-FACS screen medium and every flask run (row 43)."""


# --------------------------------------------------------------------------- #
# Foo 2014 (E. coli DH1): "1x M9 salt (Difco)" is a commercial salts powder whose
# composition the paper does not print, so the two Foo media derive from their own
# ``M9_DIFCO`` base rather than from the stated ``M9`` salts.
# --------------------------------------------------------------------------- #
M9_DIFCO = Media(
    name="M9 salts, Difco (1x; composition not stated)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9_DIFCO",
    components=[
        _mixture(
            "M9 salts (Difco)",
            _SALT,
            _DEFERRED,
            provenance=[_cite(_FOO2014, "1x M9 salt (Difco)", _Q_FOO_M9)],
            note="a vendor salts powder at its 1x working strength; the source prints no "
            "per-salt composition, and the vendor sheet is not a mirrored source",
        )
    ],
    provenance=[_cite(_FOO2014, "1x M9 salt (Difco)", _Q_FOO_M9)],
)
"""The Difco M9 salts powder, composition deferred to the vendor: the Foo 2014 base."""

M9_DIFCO_GLUCOSE_FOO2014 = Media(
    name="M9 minimal medium (Difco M9 salts + MgSO4 + CaCl2 + thiamine + 0.4% glucose; "
    "Foo 2014)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9_DIFCO",
    components=[
        *M9_DIFCO.components,
        _stated(
            "magnesium sulfate", _SALT, 2.0, _MM, _cite(_FOO2014, "2 mM", _Q_FOO_M9)
        ),
        _stated(
            "calcium chloride", _SALT, 100.0, _UM, _cite(_FOO2014, "100 uM", _Q_FOO_M9)
        ),
        _stated(
            "thiamine", _VITAMIN, 0.5, _UGML, _cite(_FOO2014, "0.5 mg/L", _Q_FOO_M9)
        ),
        _stated("D-glucose", _CSRC, 0.4, _PCT, _cite(_FOO2014, "0.4%", _Q_FOO_M9)),
    ],
    provenance=[_cite(_FOO2014, "M9 minimal medium for the growth assays", _Q_FOO_M9)],
)
"""Foo 2014's growth-assay medium (row 12)."""


def _foo_mm9(
    name: str,
    role: MediaComponentRole,
    value: float,
    unit: ConcentrationUnit,
    read: str,
) -> MediaComponent:
    return _stated(name, role, value, unit, _cite(_FOO2014, read, _Q_FOO_MM9))


MM9_FOO2014 = Media(
    name="MM9 (Difco M9 salts + 75 mM MOPS + micronutrients + 1% glucose; Foo 2014)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9_DIFCO",
    components=[
        *M9_DIFCO.components,
        _foo_mm9("3-(N-morpholino)propanesulfonic acid", _BUFFER, 75.0, _MM, "75 mM"),
        _foo_mm9("magnesium sulfate", _SALT, 2.0, _MM, "2 mM"),
        _foo_mm9("calcium chloride", _SALT, 10.0, _UM, "10 uM"),
        _foo_mm9("iron(II) sulfate", _TRACE, 10.0, _UM, "10 uM"),
        _foo_mm9("boric acid", _TRACE, 4.0, _UM, "4 uM"),
        _stated(
            "MnCl2",
            _TRACE,
            0.8,
            _UM,
            _cite(_FOO2014, "0.8 uM", _Q_FOO_MM9),
            note="the source writes 'manganese chloride'; with no hydrate named it is the "
            "anhydrous MnCl2 row (PubChem's name index answers that spelling with a "
            "four-water record)",
        ),
        _foo_mm9("cobalt chloride", _TRACE, 0.3, _UM, "0.3 uM"),
        _stated(
            "copper sulfate",
            _TRACE,
            0.15,
            _UM,
            _cite(_FOO2014, "0.15 uM", _Q_FOO_MM9),
            note="the source writes 'cupric sulfate', the copper(II) sulfate row",
        ),
        _foo_mm9("zinc sulfate", _TRACE, 0.1, _UM, "0.1 uM"),
        _stated(
            "ammonium molybdate",
            _TRACE,
            0.03,
            _UM,
            _cite(_FOO2014, "0.03 uM", _Q_FOO_MM9),
            note="PubChem's name index resolves 'ammonium molybdate' to the "
            "heptamolybdate record, the same compound Kang 2026 writes as (NH4)6Mo7O24",
        ),
        _foo_mm9("D-glucose", _CSRC, 1.0, _PCT, "1%"),
    ],
    provenance=[
        _cite(
            _FOO2014,
            "MM9",
            _Q_FOO_MM9,
            note="the MOPS pH 7.40 is not a Media field; it rides as "
            "EnvironmentPhysicalPerturbation(factor=ph)",
        )
    ],
)
"""Foo 2014's isopentenol production medium (row 12)."""


# --------------------------------------------------------------------------- #
# Davis Minimal (Caglar 2017, E. coli B REL606). The paper names the medium and defers its
# recipe to Lenski 1991 (ref 36, not mirrored); it states only the parts it varies or
# supplements: thiamine, the magnesium sulfate it titrates, and the sodium citrate behind
# the baseline sodium. Everything else stays inside one deferred line.
# --------------------------------------------------------------------------- #
DAVIS_MINIMAL = Media(
    name="Davis Minimal (DM) medium with thiamine, no carbon source (recipe deferred to "
    "Lenski 1991)",
    state="liquid",
    is_synthetic=True,
    base_medium="DAVIS_MINIMAL",
    components=[
        _mixture(
            "Davis Minimal medium salts (Lenski 1991 formulation)",
            _SALT,
            _DEFERRED,
            provenance=[
                _cite(_CAGLAR2017, "recipe deferred to ref 36", _Q_CAGLAR_DM),
                _cite(_CAGLAR2017, "ref 36 is Lenski 1991", _Q_CAGLAR_REF36),
            ],
            note="every DM ingredient Caglar 2017 does not itself state (the phosphate "
            "and ammonium salts of the Lenski formulation); not mirrored, so not "
            "expanded",
            defers_to=[_LENSKI1991],
        ),
        _stated(
            "magnesium sulfate",
            _SALT,
            0.83,
            _MM,
            _cite(_CAGLAR2017, "0.83 mM, normally present", _Q_CAGLAR_MG),
            note="the paper's Mg2+ conditions change this amount; a changed amount is an "
            "environment edit on this medium",
        ),
        _defined(
            "sodium citrate",
            MediaComponentRole.other,
            provenance=[
                _cite(
                    _CAGLAR2017,
                    "present; amount not stated",
                    _Q_CAGLAR_NA,
                    note="the ~5 mM is the medium's total sodium, not a citrate amount, "
                    "so the citrate is recorded as an identity without a number",
                )
            ],
        ),
        _stated(
            "thiamine",
            _VITAMIN,
            0.002,
            _UGML,
            _cite(
                _CAGLAR2017,
                "2 ug/L",
                _Q_CAGLAR_DM,
                note="2 ug/L is 0.002 ug/mL; the OCR renders 'ug/l' as 'ug/1'",
            ),
        ),
    ],
    provenance=[
        _cite(
            _CAGLAR2017,
            "Davis Minimal medium supplemented with thiamine (DM)",
            _Q_CAGLAR_DM,
        ),
        _cite(
            _CAGLAR2017,
            "the carbon source is the variable",
            _Q_CAGLAR_CARBON,
            note="glycerol, lactate or gluconate at 0.5 g/L instead of glucose; the "
            "loader carries each as EnvironmentPhysicalPerturbation(factor=carbon_source)",
        ),
    ],
)
"""Caglar 2017's DM base, composition deferred to the unmirrored Lenski 1991 (row 9)."""

DM500 = Media(
    name="DM500 (Davis Minimal + 500 mg/L glucose; Caglar 2017)",
    state="liquid",
    is_synthetic=True,
    base_medium="DAVIS_MINIMAL",
    components=[
        *DAVIS_MINIMAL.components,
        _stated(
            "D-glucose", _CSRC, 0.5, _GL, _cite(_CAGLAR2017, "500 mg/L", _Q_CAGLAR_DM)
        ),
    ],
    provenance=[_cite(_CAGLAR2017, "DM500", _Q_CAGLAR_DM)],
)
"""Caglar 2017's reference medium: the glucose condition and the base of the Na+ and
Mg2+ arms (row 9)."""


# --------------------------------------------------------------------------- #
# MOPS minimal (Neidhardt 1974). The originating paper is not mirrored; Tong 2020 cites it
# for the medium (ref 34) and buys it from Teknova, and Price 2018's Supplementary Table 18
# tabulates it in full. One row of that table conflicts with the same lab's earlier table:
# Price lists 0.276 mM "Aluminum potassium sulfate dodecahydrate" where Wetmore 2015's Data
# Set S1 lists 0.276 mM "Potassium Sulfate" in its MOPS formulation. ADJUDICATED to
# potassium sulfate (rule: the same amount under the same lab's earlier, independent
# tabulation, and the two other mirrored MOPS recipes, Schmidt 2022 and Thompson 2020, also
# name K2SO4; an alum in a MOPS base is a single-source outlier). Both rows are quoted.
# --------------------------------------------------------------------------- #
def _mops(
    name: str,
    role: MediaComponentRole,
    value: float,
    unit: ConcentrationUnit,
    read: str,
    *,
    note: str | None = None,
) -> MediaComponent:
    return _stated(
        name,
        role,
        value,
        unit,
        _cite(_PRICE2018_S3, read, _Q_PRICE_MOPS),
        note=note,
        defers_to=[_NEIDHARDT1974],
    )


MOPS_MINIMAL = Media(
    name="MOPS minimal medium, no carbon source (Neidhardt 1974 formulation as tabulated "
    "by Price 2018)",
    state="liquid",
    is_synthetic=True,
    base_medium="MOPS_MINIMAL",
    components=[
        _mops("3-(N-morpholino)propanesulfonic acid", _BUFFER, 40.0, _MM, "40 mM"),
        _mops("tricine", _BUFFER, 4.0, _MM, "4 mM"),
        _mops("dipotassium hydrogen phosphate", _SALT, 1.32, _MM, "1.32 mM"),
        _mops("iron(II) sulfate heptahydrate", _TRACE, 0.01, _MM, "0.01 mM"),
        _mops("ammonium chloride", _NSRC, 9.5, _MM, "9.5 mM"),
        _stated(
            "potassium sulfate",
            _SALT,
            0.276,
            _MM,
            _cite(
                _PRICE2018_S3,
                "0.276 mM (the row names 'Aluminum potassium sulfate dodecahydrate')",
                _Q_PRICE_MOPS,
            ),
            _cite(
                _WETMORE2015_DS1,
                "0.276 mM potassium sulfate",
                _Q_WETMORE_MOPS_K2SO4,
                note="the same lab's MOPS formulation names potassium sulfate at the same "
                "amount; the adjudication rule is in the comment above MOPS_MINIMAL",
            ),
            note="ADJUDICATED: Price 2018's row names an alum, Wetmore 2015's names "
            "potassium sulfate at the same 0.276 mM; recorded as potassium sulfate",
            defers_to=[_NEIDHARDT1974],
        ),
        _mops("calcium chloride", _SALT, 0.0005, _MM, "0.0005 mM"),
        _mops("magnesium chloride hexahydrate", _SALT, 0.525, _MM, "0.525 mM"),
        _mops("sodium chloride", _SALT, 50.0, _MM, "50 mM"),
        _mops("ammonium heptamolybdate tetrahydrate", _TRACE, 3e-09, _M, "3e-09 M"),
        _mops("boric acid", _TRACE, 4e-07, _M, "4e-07 M"),
        _mops("cobalt(II) chloride hexahydrate", _TRACE, 3e-08, _M, "3e-08 M"),
        _mops("copper(II) sulfate pentahydrate", _TRACE, 1e-08, _M, "1e-08 M"),
        _mops("manganese(II) chloride tetrahydrate", _TRACE, 8e-08, _M, "8e-08 M"),
        _mops("zinc sulfate heptahydrate", _TRACE, 1e-08, _M, "1e-08 M"),
    ],
    provenance=[
        _cite(_PRICE2018_S3, "MOPS minimal media_noCarbon", _Q_PRICE_MOPS),
        _cite(
            _TONG2020,
            "MOPS minimal medium, carbon source varied",
            _Q_TONG_MOPS,
            note="Tong 2020 defines its minimal medium by citation and varies only the "
            "carbon source, carried as EnvironmentPhysicalPerturbation(factor="
            "carbon_source)",
        ),
        _cite(_TONG2020, "ref 34 is Neidhardt 1974", _Q_TONG_REF34),
        _cite(_TONG2020, "bought from Teknova", _Q_TONG_TEKNOVA),
    ],
)
"""Neidhardt's MOPS minimal medium without a carbon source: Tong 2020's minimal medium by
its own citation, and Price 2018's 'MOPS minimal media_noCarbon' (rows 4 and 21)."""

_Q_WANG18_MOPS = (
    "MOPS medium was prepared according to standard laboratory techniques53 "
    "$\\scriptstyle ( 1 0 \\mathrm { g } / \\mathrm { L }$ glucose). All cultures were "
    "carried out at $3 7 ^ { \\circ } \\mathrm { C }$"
)
_Q_WANG18_CASAMINO = (
    "we performed another screening with MOPS medium supplemented with "
    "$0 . 5 \\mathrm { g } / \\mathrm { L }$ casamino acid, which is composed of all "
    "amino acids except for tryptophan"
)

MOPS_CASAMINO_WANG2018 = Media(
    name="MOPS minimal medium with 0.5 g/L casamino acids, no carbon source "
    "(Wang 2018; Neidhardt 1974 formulation as tabulated by Price 2018)",
    state="liquid",
    is_synthetic=False,
    base_medium="MOPS_MINIMAL",
    components=[
        *MOPS_MINIMAL.components,
        _mixture(
            "casamino acids",
            _COMPLEX,
            _UNDEFINED,
            concentration=_c(0.5, _GL),
            provenance=[_cite(_WANG2018, "0.5 g/L", _Q_WANG18_CASAMINO)],
            note="an acid hydrolysate of casein, so there is no structure to resolve; "
            "the source states its one compositional fact, that tryptophan is absent, "
            "which is why this medium is the L-Trp-biosynthesis selective condition",
        ),
    ],
    provenance=[
        _cite(
            _WANG2018,
            "MOPS medium supplemented with 0.5 g/L casamino acid",
            _Q_WANG18_CASAMINO,
            note="is_synthetic is False because casamino acids is an undefined "
            "hydrolysate; the MOPS base is Neidhardt 1974, which Wang 2018 cites as "
            f"its reference 53 ('{_Q_WANG18_MOPS}'), and the stated 10 g/L glucose is "
            "carried by the loader as an EnvironmentPhysicalPerturbation(factor="
            "carbon_source), so this entry stays carbon-free like its base",
        )
    ],
)
"""Wang 2018's L-Trp-biosynthesis selective medium: MOPS minimal plus 0.5 g/L casamino
acids, the amino-acid mixture that carries every amino acid except tryptophan (row 25)."""


#: Which of the fifty bacterial rows (rank in [[plan.bacteria-ontology-genome]]'s table,
#: first author, year) use each bacterial entry, and how the paper states it. "names" means
#: the paper names the formulation without printing amounts; every other use is a stated
#: recipe the entry quotes. These keys are also the set of bacterial media.
BACTERIAL_MEDIA_USES: dict[str, tuple[str, ...]] = {
    "LB": (
        "6 Carruthers 2025: precultures; its transformation plates state the Miller "
        "amounts",
        "11 Goodall 2018: the LB1/LB2 TraDIS cultures state the Miller amounts",
        "20 Menasalvas 2025: revival plates, names and states LB Miller",
        "22 Choe 2025, 23 Rapp 2026, 24 Schmidt 2022, 28 Thompson 2020, 40 Rachwalski "
        "2024: name LB Miller",
        "32 Schmidt 2016: the LB condition of the proteome map",
        "42 Babu 2014: SI states the Miller amounts",
        "25 Wang 2018: names LB broth; the essentiality screen's selective and control "
        "cultures, the auxotrophy and L-Trp control cultures, and the initial library "
        "the two chemical-tolerance screens are scored against",
    ),
    "LB_AGAR": (
        "20 Menasalvas 2025: LB Miller + 2% agar",
        "32 Schmidt 2016: LB + 20 g/L agar plates",
    ),
    "LB_LENNOX": (
        "2 Wetmore 2015: Data Set S1 'LB' (Time0 recovery and LB conditions)",
        "21 Price 2018: Table S18 'LB' (64 Keio experiments)",
        "30 Schastnaya 2021: states LB-Lennox",
        "46 Hawkins 2020: names LB Lennox",
    ),
    "YT_2X": ("15 Wang 2015: the isoprenol-tolerance growth medium",),
    "M9": (
        "18 Kang 2026: names and states the M9 salts (inside M9-MOPS)",
        "8 Borchert 2024: states the salts (its growth medium adds Mg, Ca, FeSO4)",
        "base of every M9 entry",
    ),
    "M9_GLUCOSE": ("45 Choe 2019: the ALE and growth medium",),
    "M9_NOCARBON_WETMORE2015": ("2 Wetmore 2015: 64 Keio carbon-source experiments",),
    "M9_NONITROGEN_WETMORE2015": (
        "2 Wetmore 2015: 26 Keio nitrogen-source experiments",
    ),
    "M9_NOCARBON_PRICE2018": ("21 Price 2018: 60 Keio carbon-source experiments",),
    "M9_NONITROGEN_PRICE2018": ("21 Price 2018: 32 Keio nitrogen-source experiments",),
    "M9_GLUCOSE_CASEIN_FUHRER2017": ("1 Fuhrer 2017: the metabolome screen",),
    "M9_SCHMIDT2016": ("32 Schmidt 2016: the minimal-medium proteome conditions",),
    "M9_NREL_DESIQUEIRA2025": ("14 de Siqueira 2025: tolerization and phenotyping",),
    "M9_NREL_LIM2025": ("16 Lim 2025: growth and TALE cultures",),
    "M9_NREL_LIM2022": (
        "7 Lim 2022: the in-house RNA-seq cultures of putidaPRECISE321, as the SI "
        "states them (4 g/L glucose)",
    ),
    "M9_NREL_NOCARBON_LIM2022": (
        "7 Lim 2022: the aromatic project, whose 2.5 g/L carbon source replaces the "
        "base's glucose",
    ),
    "M9_NREL_KANG2026": ("18 Kang 2026: 'M9'",),
    "M9_NREL_HIGH_N_KANG2026": ("18 Kang 2026: 'modified M9'",),
    "M9_MOPS_KANG2026": ("18 Kang 2026: 'M9-MOPS'",),
    "M9_DEFERRED_BANERJEE2025": (
        "39 Banerjee 2025: the shotgun-proteomics cultures; names a modified M9 and defers its amounts to reference 21",
    ),
    "M9_NREL_CARRUTHERS2025": ("6 Carruthers 2025: isoprenol production",),
    "M9_NREL_MOPS_MENASALVAS2025": ("20 Menasalvas 2025: isoprenol production",),
    "M9_DIFCO": ("12 Foo 2014: base of both Foo media",),
    "M9_DIFCO_GLUCOSE_FOO2014": ("12 Foo 2014: growth and tolerance assays",),
    "MM9_FOO2014": ("12 Foo 2014: isopentenol production and microarray cultures",),
    "DAVIS_MINIMAL": ("9 Caglar 2017: the carbon-source arm",),
    "DM500": ("9 Caglar 2017: the glucose reference and the Na+ / Mg2+ arms",),
    "MOPS_MINIMAL": (
        "4 Tong 2020: names Teknova MOPS minimal, cites Neidhardt 1974",
        "21 Price 2018: Table S18 (2 Keio experiments)",
        "25 Wang 2018: the auxotrophy, furfural and isobutanol selective conditions; "
        "cites Neidhardt 1974 and states 10 g/L glucose, carried as a carbon-source "
        "factor",
    ),
    "MOPS_CASAMINO_WANG2018": (
        "25 Wang 2018: the L-Trp-biosynthesis selective condition (0.5 g/L casamino "
        "acid in MOPS)",
    ),
    "M9_MODIFIED_FANG2025": (
        "43 Fang 2025: both rounds of the CRISPRi-FACS screen (500 mL flasks holding "
        "100 mL), and every tube and flask fermentation; the glucose variant the same "
        "sentence offers is used by one titer figure and by no screen record",
    ),
}
"""Use map for the bacterial entries; the keys are exactly the bacterial media."""


# Registry of the canonical media (name -> object), for discovery/migration.
MEDIA_LIBRARY: dict[str, Media] = {
    "SD_MSG": SD_MSG,
    "SGA_DM_SELECTION": SGA_DM_SELECTION,
    "SGA_TM_SELECTION": SGA_TM_SELECTION,
    "SGA_DM_SELECTION_GALACTOSE": SGA_DM_SELECTION_GALACTOSE,
    "YPD": YPD,
    "YPD_LIQUID": YPD_LIQUID,
    "YPD_AGAR": YPD_AGAR,
    "YPAD": YPAD,
    "SD": SD,
    "YNB": YNB,
    "SC": SC,
    "SC_URA": SC_URA,
    "SYNH3_MINUS": SYNH3_MINUS,
    "SYNBASE": SYNBASE,
    "YPB": YPB,
    "YPBO": YPBO,
    "YNB_YE_B": YNB_YE_B,
    "YPBM": YPBM,
    "YPBA": YPBA,
    "SED_URA": SED_URA,
    "SED_URA_G418": SED_URA_G418,
    "YP": YP,
    "YP_FRUCTOSE": YP_FRUCTOSE,
    "YP_GALACTOSE": YP_GALACTOSE,
    "YP_LACTATE": YP_LACTATE,
    "YP_MALTOSE": YP_MALTOSE,
    "YP_MANNOSE": YP_MANNOSE,
    "YP_RAFFINOSE": YP_RAFFINOSE,
    "YP_SUCROSE": YP_SUCROSE,
    "YP_TREHALOSE": YP_TREHALOSE,
    "YP_XYLOSE": YP_XYLOSE,
    "YP_GLYCEROL": YP_GLYCEROL,
    "YP_GLYCEROL_LIQUID": YP_GLYCEROL_LIQUID,
    "YP_ETHANOL": YP_ETHANOL,
    "YPD_ETHANOL": YPD_ETHANOL,
    "YNB_GLUCOSE_SOLID": YNB_GLUCOSE_SOLID,
    "SM": SM,
    "SM_AGAR": SM_AGAR,
    "SM_DEFERRED": SM_DEFERRED,
    # bacterial media ([[plan.bacteria-ontology-genome]] Step 5; uses in BACTERIAL_MEDIA_USES)
    "LB": LB,
    "LB_AGAR": LB_AGAR,
    "LB_LENNOX": LB_LENNOX,
    "YT_2X": YT_2X,
    "M9": M9,
    "M9_GLUCOSE": M9_GLUCOSE,
    "M9_NOCARBON_WETMORE2015": M9_NOCARBON_WETMORE2015,
    "M9_NONITROGEN_WETMORE2015": M9_NONITROGEN_WETMORE2015,
    "M9_NOCARBON_PRICE2018": M9_NOCARBON_PRICE2018,
    "M9_NONITROGEN_PRICE2018": M9_NONITROGEN_PRICE2018,
    "M9_GLUCOSE_CASEIN_FUHRER2017": M9_GLUCOSE_CASEIN_FUHRER2017,
    "M9_SCHMIDT2016": M9_SCHMIDT2016,
    "M9_NREL_DESIQUEIRA2025": M9_NREL_DESIQUEIRA2025,
    "M9_NREL_LIM2025": M9_NREL_LIM2025,
    "M9_NREL_LIM2022": M9_NREL_LIM2022,
    "M9_NREL_NOCARBON_LIM2022": M9_NREL_NOCARBON_LIM2022,
    "M9_NREL_KANG2026": M9_NREL_KANG2026,
    "M9_NREL_HIGH_N_KANG2026": M9_NREL_HIGH_N_KANG2026,
    "M9_MOPS_KANG2026": M9_MOPS_KANG2026,
    "M9_DEFERRED_BANERJEE2025": M9_DEFERRED_BANERJEE2025,
    "M9_NREL_CARRUTHERS2025": M9_NREL_CARRUTHERS2025,
    "M9_NREL_MOPS_MENASALVAS2025": M9_NREL_MOPS_MENASALVAS2025,
    "M9_DIFCO": M9_DIFCO,
    "M9_DIFCO_GLUCOSE_FOO2014": M9_DIFCO_GLUCOSE_FOO2014,
    "MM9_FOO2014": MM9_FOO2014,
    "DAVIS_MINIMAL": DAVIS_MINIMAL,
    "DM500": DM500,
    "MOPS_MINIMAL": MOPS_MINIMAL,
    "MOPS_CASAMINO_WANG2018": MOPS_CASAMINO_WANG2018,
    "M9_MODIFIED_FANG2025": M9_MODIFIED_FANG2025,
} | {
    _hm_key(compound, partial): HILLENMEYER_DROPOUT_MEDIA[label]
    for label, compound, partial in _HILLENMEYER_DROPOUTS
}

#: Media that name no carbon source, each for a stated reason. A base exists to be
#: derived from (``YP``, ``SD_MSG``, ``YPB``, ``YNB_YE_B``, ``YNB``), and ``SYNH3_MINUS``
#: / ``SYNBASE`` keep their sugars inside a composition deferred to Zhang 2019.
CARBON_FREE_MEDIA: dict[str, str] = {
    "YP": "base; the carbon source is the thing a YP derivative adds",
    "SD_MSG": "base; the SGA recipes state glucose, the Costanzo 2021 condition "
    "states galactose",
    "YNB": "base; the vitamin set only",
    "YPB": "base of YPBO; the oleic acid is the derivative's",
    "YNB_YE_B": "base of YPBM and YPBA; the fatty acid or acetate is the derivative's",
    "SYNH3_MINUS": "the hydrolysate sugars sit inside a composition deferred to "
    "Zhang 2019, which is not mirrored",
    "SYNBASE": "same deferral as its SynH3- base",
    "SM_DEFERRED": "Zelezniak 2018 states no recipe, so the carbon source sits inside "
    "a composition deferred to Mulleder 2012, which is not mirrored",
    "LB": "complex; the carbon is in the undefined tryptone and yeast extract, and no "
    "carbon source is added",
    "LB_AGAR": "complex; the carbon is in the undefined tryptone and yeast extract",
    "LB_LENNOX": "complex; the carbon is in the undefined tryptone and yeast extract",
    "YT_2X": "complex; the carbon is in the undefined tryptone and yeast extract",
    "M9": "base; the M9 salts carry no carbon source by definition",
    "M9_NOCARBON_WETMORE2015": "the carbon source is the variable of the RB-TnSeq "
    "carbon-source experiments",
    "M9_NOCARBON_PRICE2018": "the carbon source is the variable of the RB-TnSeq "
    "carbon-source experiments",
    "M9_SCHMIDT2016": "the carbon source is the variable across the eleven carbon-source "
    "conditions",
    "M9_NREL_DESIQUEIRA2025": "acetate or glucose, varied per condition",
    "M9_DEFERRED_BANERJEE2025": "p-CA alone, or p-CA with D-alanine and L-malate, "
    "varied per condition; the Methods hand the amounts to the figure legends",
    "M9_NREL_NOCARBON_LIM2022": "the aromatic project varies the carbon source "
    "(p-coumarate, ferulate, both, or glucose) at 2.5 g/L",
    "M9_NREL_KANG2026": "glucose alone or a glucose:xylose mixture, varied per condition",
    "M9_NREL_HIGH_N_KANG2026": "same sugar regimes as the Kang 2026 M9",
    "M9_MOPS_KANG2026": "same sugar regimes as the Kang 2026 M9",
    "M9_DIFCO": "base; the Difco salts powder",
    "DAVIS_MINIMAL": "base; the carbon source is the variable of Caglar 2017's carbon arm",
    "MOPS_MINIMAL": "base; the carbon source is the variable (Tong 2020) and Price 2018 "
    "tabulates it without one",
    "MOPS_CASAMINO_WANG2018": "same as its MOPS base; Wang 2018's stated 10 g/L glucose "
    "is carried as a carbon-source factor so all four of its MOPS conditions share one "
    "base object",
}


def _check_library() -> None:
    """Every ``base_medium`` in the library names a ``MEDIA_LIBRARY`` key.

    Run at import, because a base that resolves to nothing is invisible: the medium
    still validates, still serializes, and simply joins to no other record.
    """
    unresolved = {
        key: media.base_medium
        for key, media in MEDIA_LIBRARY.items()
        if media.base_medium is None or media.base_medium not in MEDIA_LIBRARY
    }
    if unresolved:
        raise RuntimeError(
            "torchcell/datamodels/media.py: base_medium must name a MEDIA_LIBRARY "
            f"key, but these do not: {unresolved}"
        )


_check_library()
