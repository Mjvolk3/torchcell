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
                    "note": "partial drop-out; the source does not state the reduced "
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

Hoepfner 2014, Hillenmeyer 2008 and the served Nadal-Ribelles / Ohya / Ohnuki /
da Silveira loaders all describe liquid YPD and today emit a bare
``Media(name="YPD", state="liquid", is_synthetic=False)`` with no components, which
reaches no FBA bound. Migrating them is a full-rebuild event, so it is scheduled with
one, not slipped in per dataset.
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
# The SOM states neither the base recipe nor the reduced level of the three "partial"
# conditions; the base is SC because a named-nutrient dropout is only defined against
# a synthetic complete medium, and that inference is recorded on every object.
# --------------------------------------------------------------------------- #
_HILLENMEYER_DROPOUT_QUOTE = (
    "we restricted our analysis to small molecule experiments, excluding conditions "
    "of environmental change, such as amino acid dropout"
)


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
        return dropout(
            SC,
            name=f"SC, partial {compound} drop-out (Hillenmeyer 2008 '{label}')",
            partial=(compound,),
            provenance=provenance,
        )
    return dropout(
        SC,
        compound,
        name=f"SC - {compound} (Hillenmeyer 2008 '{label}')",
        provenance=provenance,
    )


#: (condition label in the HOM score matrix, the SC component it names, partial?).
#: "partial" is the source's own word for three vitamin conditions: the nutrient is
#: reduced, not removed, and the reduced level is never stated.
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
