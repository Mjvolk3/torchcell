# torchcell/datamodels/media
# [[torchcell.datamodels.media]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/media
"""Reusable, provenance-first definitions of the common growth media.

Each constant is a fully-typed :class:`~torchcell.datamodels.schema.Media` with
component-level composition and ``SourcedValue`` provenance, so a dataset loader
imports one canonical object instead of re-declaring ``Media(name=..., state=...)``.

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

Follow-ups (documented gaps, fillable later):
- ``Compound`` ChEBI / InChIKey / SMILES cross-refs are left empty here and get
  populated by a sourced ChEBI/PubChem resolver pass (never guessed).
- The SC amino-acid supplement is one ``defined`` component carrying the full
  per-ingredient gram breakdown in its provenance quote; a cobra adapter can
  expand it to per-metabolite bounds.

Design note: ``[[torchcell.datamodels.media-components]]``.
"""

from __future__ import annotations

from torchcell.datamodels.schema import (
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    Media,
    MediaComponent,
    MediaComponentRole,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

# --------------------------------------------------------------------------- #
# Provenance anchors (paper.txt sha256, verified in the library mirror).
# --------------------------------------------------------------------------- #
_TONG2006 = "yantongSyntheticGeneticArray2006"
_TONG2006_SHA = "dda5fc727c5e532e02884cd1d30ad0774bfb773ed55e8f6b7074ee2158ab9aca"
_KUZMIN2016 = "kuzminSyntheticGeneticArray2016"
_KUZMIN2016_SHA = "02360306e6d0eb6324a9af962b8970cad019b3e06d5073688925614a657848ca"


def _sv(
    value: object,
    quote: str,
    *,
    ck: str = _TONG2006,
    sha: str = _TONG2006_SHA,
    note: str | None = None,
) -> SourcedValue:
    """A SourcedValue pinned to a mirrored paper.txt (quote + sha256)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(source_uri="paper.txt", citation_key=ck, sha256=sha),
        quote=quote,
        note=note,
    )


def _c(value: float, unit: ConcentrationUnit) -> Concentration:
    return Concentration(value=value, unit=unit)


_PCT = ConcentrationUnit.percent_w_v
_GL = ConcentrationUnit.g_per_l
_UGML = ConcentrationUnit.ug_per_ml

# --------------------------------------------------------------------------- #
# SGA final selection medium (SD/MSG) -- Tong & Boone 2006 recipe #16 (per L):
#   1.7 g YNB w/o amino acids or ammonium sulfate, 1 g MSG, 2 g amino-acids
#   supplement powder (DO -His/Arg/Lys), 20 g bacto agar, 50 mL 40% glucose
#   (= 20 g/L), canavanine 50 mg/L, thialysine 50 mg/L, G418 200 mg/L,
#   clonNAT 100 mg/L. Whole SGA screen incubated at 26 C (Kuzmin 2018 SI).
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


def _sga_components() -> list[MediaComponent]:
    """Shared SD/MSG selection-medium components (Tong & Boone 2006 recipe #16)."""
    return [
        MediaComponent(
            compound=Compound(name="D-glucose"),
            role=MediaComponentRole.carbon_source,
            concentration=_c(2.0, _PCT),
            provenance=[_sv("20 g/L", "add 50 mL 40% glucose")],
        ),
        MediaComponent(
            compound=Compound(name="monosodium L-glutamate"),
            role=MediaComponentRole.nitrogen_source,
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
        ),
        MediaComponent(
            compound=Compound(
                name="yeast nitrogen base (w/o amino acids and ammonium sulfate)"
            ),
            role=MediaComponentRole.other,
            concentration=_c(1.7, _GL),
            definition=ComponentDefinition.composition_deferred,
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
        ),
        MediaComponent(
            compound=Compound(name="SC amino-acid supplement powder (DO -His/Arg/Lys)"),
            role=MediaComponentRole.amino_acid,
            concentration=_c(2.0, _GL),
            provenance=[_sv("2 g DO powder/L", _SGA_SUPPLEMENT_QUOTE)],
            note="complete SC supplement MINUS the His/Arg/Lys dropout; full per-"
            "ingredient grams are in the provenance quote (3 g adenine, 2 g "
            "uracil, 2 g inositol, 0.2 g PABA, all 20 AAs @2 g except leucine "
            "@10 g, per 55.2 g mix, used at 2 g mix/L)",
        ),
        MediaComponent(
            compound=Compound(name="agar"),
            role=MediaComponentRole.gelling_agent,
            concentration=_c(2.0, _PCT),
            provenance=[_sv("20 g/L", "Add 20 g bacto agar")],
        ),
        MediaComponent(
            compound=Compound(name="L-canavanine"),
            role=MediaComponentRole.selection_agent,
            concentration=_c(50.0, _UGML),
            provenance=[_sv("50 mg/L", "add 0.5 mL canavanine (50 mg/L)")],
            note="toxic L-arginine analog; selects can1-delta haploids",
        ),
        MediaComponent(
            compound=Compound(name="thialysine (S-(2-aminoethyl)-L-cysteine)"),
            role=MediaComponentRole.selection_agent,
            concentration=_c(50.0, _UGML),
            provenance=[_sv("50 mg/L", "0.5 mL thialysine (50 mg/L)")],
            note="toxic L-lysine analog; selects lyp1-delta haploids",
        ),
        MediaComponent(
            compound=Compound(name="G418 (geneticin)"),
            role=MediaComponentRole.selection_agent,
            concentration=_c(200.0, _UGML),
            provenance=[_sv("200 mg/L", "1 mL G418 (200 mg/L)")],
            note="selects the kanMX marker",
        ),
        MediaComponent(
            compound=Compound(name="nourseothricin (clonNAT)"),
            role=MediaComponentRole.selection_agent,
            concentration=_c(100.0, _UGML),
            provenance=[_sv("100 mg/L", "1 mL clonNAT (100 mg/L)")],
            note="selects the natMX marker",
        ),
    ]


_HIS = Compound(name="L-histidine")
_ARG = Compound(name="L-arginine")
_LYS = Compound(name="L-lysine")
_URA = Compound(name="uracil")

SGA_DM_SELECTION = Media(
    name="SGA double-mutant selection (SD-MSG, -His/Arg/Lys, +canavanine/thialysine/G418/clonNAT)",
    state="solid",
    is_synthetic=True,
    base_medium="SD_MSG",
    components=_sga_components(),
    dropouts=[_HIS, _ARG, _LYS],
    provenance=[_sv("SD/MSG -His/Arg/Lys selection medium", _SGA_BASE_QUOTE)],
)
"""Costanzo 2016 / Baryshnikova 2010 digenic SGA fitness-scoring medium."""

SGA_TM_SELECTION = Media(
    name="SGA triple-mutant selection (SD-MSG, -His/Arg/Lys/Ura, +canavanine/thialysine/G418/clonNAT)",
    state="solid",
    is_synthetic=True,
    base_medium="SD_MSG",
    components=_sga_components(),
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

# --------------------------------------------------------------------------- #
# YEPD / YPD family -- complex (NOT chemically defined). Tong & Boone 2006 #9.
# --------------------------------------------------------------------------- #
YPD = Media(
    name="YPD (yeast extract / peptone / dextrose)",
    state="solid",
    is_synthetic=False,
    base_medium="YPD",
    components=[
        MediaComponent(
            compound=Compound(name="yeast extract"),
            role=MediaComponentRole.complex_ingredient,
            concentration=_c(1.0, _PCT),
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[_sv("10 g/L", "10 g yeast extract")],
        ),
        MediaComponent(
            compound=Compound(name="peptone"),
            role=MediaComponentRole.complex_ingredient,
            concentration=_c(2.0, _PCT),
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[_sv("20 g/L", "20 g peptone")],
        ),
        MediaComponent(
            compound=Compound(name="D-glucose"),
            role=MediaComponentRole.carbon_source,
            concentration=_c(2.0, _PCT),
            provenance=[
                _sv("20 g/L (50 mL 40% glucose)", "add 50 mL of 40% glucose solution")
            ],
        ),
    ],
    provenance=[
        _sv(
            "YEPD recipe",
            "YEPD: Add 120 mg adenine (Sigma), 10 g yeast extract, 20 g "
            "peptone, 20 g bacto agar ... add 50 mL of 40% glucose solution",
        )
    ],
)
"""Rich complex medium; peptone + yeast extract are intrinsically undefined."""

YPAD = Media(
    name="YPAD (YPD + adenine)",
    state="solid",
    is_synthetic=False,
    base_medium="YPD",
    components=[
        *YPD.components,
        MediaComponent(
            compound=Compound(name="adenine"),
            role=MediaComponentRole.nucleobase,
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
    """Name-only defined components (ChEBI/concentration filled by later passes)."""
    return [
        MediaComponent(
            compound=Compound(name=n),
            role=role,
            note="identity defined; ChEBI/InChIKey/SMILES + wet-lab concentration "
            "pending a sourced enrichment pass",
        )
        for n in names
    ]


SD_MINIMAL = Media(
    name="SD minimal (YNB + ammonium sulfate + glucose)",
    state="liquid",
    is_synthetic=True,
    base_medium="SD",
    components=[
        MediaComponent(
            compound=Compound(name="D-glucose"),
            role=MediaComponentRole.carbon_source,
            concentration=_c(2.0, _PCT),
        ),
        MediaComponent(
            compound=Compound(name="ammonium sulfate"),
            role=MediaComponentRole.nitrogen_source,
            concentration=_c(5.0, _GL),
        ),
        MediaComponent(
            compound=Compound(name="yeast nitrogen base (w/o amino acids)"),
            role=MediaComponentRole.other,
            concentration=_c(1.7, _GL),
            definition=ComponentDefinition.composition_deferred,
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

SC = Media(
    name="SC (synthetic complete: YNB + 20 amino acids + uracil + adenine)",
    state="liquid",
    is_synthetic=True,
    base_medium="SC",
    components=[
        *YNB.components,
        *_named(_SC_AMINO_ACIDS, MediaComponentRole.amino_acid),
        MediaComponent(
            compound=Compound(name="uracil"),
            role=MediaComponentRole.nucleobase,
            note="SC nucleobase supplement",
        ),
        MediaComponent(
            compound=Compound(name="adenine"),
            role=MediaComponentRole.nucleobase,
            note="SC nucleobase supplement",
        ),
    ],
)
"""Synthetic complete (defined): YNB + all 20 amino acids + uracil + adenine."""

SC_URA = Media(
    name="SC-Ura (synthetic complete minus uracil; URA3 plasmid selection)",
    state="liquid",
    is_synthetic=True,
    base_medium="SC",
    components=[c for c in SC.components if c.compound.name != "uracil"],
    dropouts=[_URA],
)
"""SC with uracil dropped (selects a URA3-bearing plasmid)."""

# Registry of the canonical media (name -> object), for discovery/migration.
MEDIA_LIBRARY: dict[str, Media] = {
    "SGA_DM_SELECTION": SGA_DM_SELECTION,
    "SGA_TM_SELECTION": SGA_TM_SELECTION,
    "YPD": YPD,
    "YPAD": YPAD,
    "SD_MINIMAL": SD_MINIMAL,
    "YNB": YNB,
    "SC": SC,
    "SC_URA": SC_URA,
}
