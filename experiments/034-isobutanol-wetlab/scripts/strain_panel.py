# experiments/034-isobutanol-wetlab/scripts/strain_panel.py
# [[experiments.034-isobutanol-wetlab.scripts.strain_panel]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-isobutanol-wetlab/scripts/strain_panel
"""The eleven strains on hand, mapped onto the paper that built them.

Emits the two tables the 034 document prints and a CSV of the same records.
Every field that describes what a strain IS carries a verbatim quote from the
mirrored paper, so the table is checkable line by line against
``$DATA_ROOT/torchcell-library/zhangBiosensorBranchedchainAmino2022/``.

Run from the repository root::

    ~/miniconda3/envs/torchcell/bin/python \
        experiments/034-isobutanol-wetlab/scripts/strain_panel.py
"""

from __future__ import annotations

import csv
import os
import os.path as osp
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXPERIMENT = "034-isobutanol-wetlab"
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, EXPERIMENT, "results")

# The document these tables are printed in. Kept here rather than passed in, so
# the emitted .tex lands beside the sections that \input it and nothing has to
# be told twice where the document lives.
REPO = osp.abspath(osp.join(EXPERIMENT_ROOT, os.pardir))
TABLES_DIR = osp.join(REPO, "notes-tex", EXPERIMENT, "tables")

CITATION_KEY = "zhangBiosensorBranchedchainAmino2022"
PAPER_SHA256 = "551ba08ec3bf584ef49b72186050d4bb3230df59edfa664a8a6af3460a2a8bec"
# si/si3.pdf is the Supplementary Information, which carries Supplementary
# Table 1, the only place a full genotype is printed for any of these strains.
SI3_SHA256 = "74de9b6cff654c5d9b6988a0a0c08672eb7defad4f002ecce6b9e6b9fa48824f"

Configuration = Literal["isobutanol", "isopentanol", "none", "unknown"]
ColorGroup = Literal["green", "blue", "orange", "gray"]


class Titer(BaseModel):
    """A reported product titer, with the carbon source it was measured on.

    Titers in this paper are NOT comparable across rows. The isobutanol numbers
    span 15% galactose, 15% glucose and 2% glucose in the dark, and the
    isopentanol numbers are a different product entirely, so ``condition`` has
    to travel with the value.
    """

    product: Literal["isobutanol", "isopentanol"]
    mg_per_l: float
    sd: float | None = None
    condition: str
    figure: str
    derived: bool = Field(
        default=False,
        description=(
            "True when the value is back-computed from a fold-change the paper "
            "reports rather than printed as a number. A derived value is ours, "
            "not theirs, and the document flags it."
        ),
    )


class Plasmid(BaseModel):
    """An episomal plasmid a strain carries, from Supplementary Table 2.

    Every plasmid in this panel is the same backbone family, ``AmpR, CEN,
    URA3``, built off the empty parent pYZ125. Two consequences the genotype
    string alone does not make obvious: the strain must be held on uracil
    dropout or it segregates the plasmid, and URA3 is spent, so the strain
    cannot also take a URA3-marked library.
    """

    name: str
    backbone: str = "AmpR, CEN, URA3"
    cargo: str
    # AmpR means the plasmid itself is propagated in E. coli DH5-alpha in the
    # source lab. Requesting a plasmid is therefore a separate line item from
    # requesting a yeast strain.
    ecoli_host: str = "DH5-alpha"


class Strain(BaseModel):
    """One tube in the freezer, and what the paper says it is.

    All eleven are *S. cerevisiae*. Every one appears in Supplementary
    Table 1, titled "Yeast strains used in this study". The only E. coli in
    the paper is DH5-alpha as a cloning host, plus ``Ec_ilvC``, which is an
    E. coli gene expressed in yeast rather than an E. coli strain.
    """

    collection_id: str
    tube_label: str
    paper_strain: str
    color_group: ColorGroup
    configuration: Configuration
    role: str
    parent: str
    plasmid: Plasmid | None = None
    production_carbon: str = Field(
        description=(
            "Carbon source the 48 h fermentation step uses for this strain. "
            "Not a free choice: YZy452 and its unplasmided self make no "
            "isobutanol on glucose because their Ll_ilvD sits behind PGAL10."
        )
    )
    pick_and_use: bool = Field(
        default=True,
        description=(
            "True when the strain runs on the paper's ordinary 24-well "
            "protocol: single colony, shaker, plate reader, HPLC, no sorting "
            "and no special illumination."
        ),
    )
    round_one: str = Field(
        default="",
        description="Call for the first sequencing round, empty when not in it.",
    )

    @property
    def medium(self) -> str:
        """Uracil dropout exactly when a CEN URA3 plasmid has to be held."""
        return "SC-ura" if self.plasmid else "SC"

    genotype: str = Field(
        description=(
            "Verbatim from Supplementary Table 1, 'Yeast strains used in this "
            "study'. Greek and superscripts are written ASCII here and become "
            "symbols only on the way into LaTeX."
        )
    )
    titer: Titer | None = None
    quote: str
    quote_locator: str


# ---------------------------------------------------------------------------
# Records. Quote locators are section names in the mirrored paper.md, so they
# survive a re-OCR that renumbers lines.
# ---------------------------------------------------------------------------

STRAINS: list[Strain] = [
    # The green group is three cells of a 2x2 that Fig. 3c reports as Strains
    # A through D: {YZy91, YZy363} crossed with {ILV6 wild type, ILV6V110E}.
    # The missing cell is YZy312, Strain B.
    Strain(
        collection_id="C0043",
        production_carbon="15% glucose",
        tube_label="Yzy311",
        paper_strain="YZy311",
        color_group="green",
        configuration="isobutanol",
        role=(
            "Strain A of Fig. 3c: the ILV6 screening host carrying wild-type "
            "ILV6 on a CEN plasmid, with no isobutanol pathway overexpressed."
        ),
        parent="YZy91",
        plasmid=Plasmid(name="pYZ127", cargo="PTDH3-ILV6-TADH1"),
        genotype="YZy91, CEN URA3 plasmid (PTDH3-ILV6-TADH1)",
        quote="YZy311 YZy91, pYZ127 YZy91, CEN URA3 plasmid (PTDH3-ILV6-TADH1)",
        quote_locator="Supplementary Table 1",
    ),
    Strain(
        collection_id="C0044",
        production_carbon="15% glucose",
        round_one="yes -- ILV6 wild type, low arm of the 1.6-fold pair",
        tube_label="YZy313",
        paper_strain="YZy313",
        color_group="green",
        configuration="isobutanol",
        role=(
            "Strain C of Fig. 3c: wild-type ILV6 on a CEN plasmid in YZy363, "
            "which carries the mitochondrial isobutanol pathway "
            "delta-integrated. The denominator of the 1.6-fold improvement."
        ),
        parent="YZy363",
        plasmid=Plasmid(name="pYZ127", cargo="PTDH3-ILV6-TADH1"),
        genotype="YZy363, CEN URA3 plasmid (PTDH3-ILV6-TADH1)",
        quote="YZy313 YZy363, pYZ127 YZy363, CEN URA3 plasmid (PTDH3-ILV6-TADH1)",
        quote_locator="Supplementary Table 1",
    ),
    Strain(
        collection_id="C0045",
        production_carbon="15% glucose",
        round_one="yes -- ILV6V110E, high arm of the 1.6-fold pair",
        tube_label="YZy314",
        paper_strain="YZy314",
        color_group="green",
        configuration="isobutanol",
        role=(
            "Strain D of Fig. 3c: the ILV6V110E hit in the same "
            "pathway-overexpressing host, 1.6-fold over C0044. One point "
            "mutation separates the two."
        ),
        parent="YZy363",
        plasmid=Plasmid(name="pYZ228", cargo="PTDH3-ILV6V110E-TADH1"),
        genotype="YZy363, CEN URA3 plasmid (PTDH3-ILV6V110E-TADH1)",
        quote=(
            "YZy314 YZy363, pYZ228 YZy363, CEN URA3 plasmid (PTDH3-ILV6V110E-TADH1)"
        ),
        quote_locator="Supplementary Table 1",
    ),
    Strain(
        collection_id="C0046",
        production_carbon="15% glucose",
        tube_label="YZy148",
        paper_strain="YZy148",
        color_group="blue",
        configuration="isopentanol",
        role=(
            "Screening host for the LEU4 mutagenesis libraries. Carries the "
            "isopentanol configuration with the LEU4DS547 allele removed, so a "
            "plasmid-borne LEU4 is the only source of the enzyme."
        ),
        parent="CEN.PK2-1C",
        genotype=(
            "CEN.PK2-1C, his3::HIS3-PLEU1-yEGFP-TADH1 bat1D::hphMX "
            "leu4D::lox71-kanMX-lox66 leu9D::lox71-natMX-lox66 leu2::LEU2"
        ),
        quote=(
            "These libraries were separately used to transform a "
            "DBAT1/DLEU4/DLEU9 strain containing a modified isopentanol "
            "configuration of the biosensor lacking the LEU4DS547 (YZy148, "
            "Supplementary Tables 1 and 2)."
        ),
        quote_locator="Results, LEU4 mutant screen",
    ),
    Strain(
        collection_id="C0047",
        production_carbon="15% glucose",
        tube_label="YZy148+LEU4WT",
        paper_strain="YZy148 + LEU4 (wild type)",
        color_group="blue",
        configuration="isopentanol",
        role=(
            "Wild-type LEU4 control for the isopentanol screen: YZy148 "
            "retransformed with a plasmid carrying unmutagenized LEU4."
        ),
        parent="YZy148",
        plasmid=Plasmid(name="pYZ149", cargo="PTDH3-LEU4-TADH1"),
        genotype="YZy148, CEN URA3 plasmid (PTDH3-LEU4-TADH1)",
        titer=Titer(
            product="isopentanol",
            mg_per_l=201.0,
            condition="48 h fermentation",
            figure="Fig. 4, back-computed from the 4.8-fold ratio",
            derived=True,
        ),
        quote=(
            "we cured each strain from its LEU4-containing plasmid and "
            "retransformed the parent strain (YZy148) with"
        ),
        quote_locator="Results, LEU4 mutant screen",
    ),
    Strain(
        collection_id="C0048",
        production_carbon="15% glucose",
        tube_label="YZy148+LEU4deltaS547",
        paper_strain="YZy148 + LEU4DS547",
        color_group="blue",
        configuration="isopentanol",
        role=(
            "The published leucine-insensitive allele, a single Ser547 "
            "deletion, carried back into the screening host. This is the "
            "benchmark the evolved LEU4 variants were scored against."
        ),
        parent="YZy148",
        plasmid=Plasmid(name="pYZ154", cargo="PTDH3-LEU4DS547-TADH1"),
        genotype="YZy148, CEN URA3 plasmid (PTDH3-LEU4DS547-TADH1)",
        titer=Titer(
            product="isopentanol",
            mg_per_l=842.0,
            sd=23.0,
            condition="48 h fermentation",
            figure="Fig. 4",
        ),
        quote="a Leu4p variant with a Ser547 deletion (Leu4DS547)",
        quote_locator="Results, biosensor configurations",
    ),
    Strain(
        collection_id="C0049",
        production_carbon="15% galactose",
        round_one="yes -- ladder floor, dehydratase off on glucose",
        tube_label="YZy452",
        paper_strain="YZy452",
        color_group="orange",
        configuration="isobutanol",
        role=(
            "Best cytosolic-pathway isolate from three rounds of FACS on "
            "delta-site integrants, and the host the Ll_ilvD plasmid library "
            "was then screened in."
        ),
        parent="YZy449",
        genotype=(
            "YZy449, d-integration-PTDH3-Ec_ilvCP2D1-A1-TADH1-"
            "[PTEF1-Ec_ilvCP2D1-A1-TACT1]. YZy449 is ilv3D tma29D with "
            "PGAL10-Ll_ilvD and PTEF1-Bs_alsS."
        ),
        titer=Titer(
            product="isobutanol",
            mg_per_l=310.0,
            sd=15.0,
            condition="15% galactose",
            figure="Supplementary Fig. 12",
        ),
        quote=(
            "the highest producing strain (YZy452) achieving 310 +/- 15 mg/L "
            "isobutanol (Supplementary Fig. 12)"
        ),
        quote_locator="Results, cytosolic pathway screen",
    ),
    Strain(
        collection_id="C0050",
        production_carbon="15% glucose",
        round_one="yes -- wild-type allele anchor",
        tube_label="YZy454",
        paper_strain="YZy454",
        color_group="orange",
        configuration="isobutanol",
        role=(
            "Wild-type Ll_ilvD control: YZy452 carrying a CEN plasmid with the "
            "unmutated dehydratase. The denominator of the 3.1-fold "
            "improvement."
        ),
        parent="YZy452",
        plasmid=Plasmid(name="pYZ126", cargo="PTDH3-Ll_ilvD-TADH1"),
        genotype="YZy452, CEN URA3 plasmid (PTDH3-Ll_ilvD-TADH1)",
        titer=Titer(
            product="isobutanol",
            mg_per_l=73.0,
            condition="15% glucose",
            figure="Fig. 5c, back-computed from the 3.1-fold ratio",
            derived=True,
        ),
        quote=(
            "which is 2- to 3.5-times higher than a strain (YZy454) containing "
            "a plasmid with the wild-type Ll_ilvD"
        ),
        quote_locator="Results, Ll_ilvD variant screen",
    ),
    Strain(
        collection_id="C0051",
        production_carbon="15% glucose",
        round_one="yes -- Ll_ilvDI433V, 3.1-fold over C0050",
        tube_label="YZy469",
        paper_strain="YZy469",
        color_group="orange",
        configuration="isobutanol",
        role=(
            "The Ll_ilvD hit, I433V, on a CEN plasmid. Same host and same "
            "carbon source as YZy454, so the pair is a clean one-allele "
            "contrast."
        ),
        parent="YZy452",
        plasmid=Plasmid(name="pYZ353", cargo="PTDH3-Ll_ilvDI433V-TADH1"),
        genotype="YZy452, CEN URA3 plasmid (PTDH3-Ll_ilvDI433V-TADH1)",
        titer=Titer(
            product="isobutanol",
            mg_per_l=227.0,
            sd=13.0,
            condition="15% glucose",
            figure="Fig. 5c",
        ),
        quote=(
            "The strain retransformed with the most active variant "
            "Ll_ilvDI433V (YZy469) produces 227 +/- 13 mg/L of isobutanol from "
            "15% glucose, which is 3.1-fold higher than the strain harboring "
            "the wild-type Ll_ilvD (YZy454, Fig. 5c)."
        ),
        quote_locator="Results, Ll_ilvD variant screen",
    ),
    Strain(
        collection_id="C0052",
        production_carbon="15% glucose",
        tube_label="YZy91",
        paper_strain="YZy91",
        color_group="gray",
        configuration="isobutanol",
        role=(
            "Screening host for the ILV6 mutagenesis libraries. BAT1 and BAT2 "
            "are deleted so valine and alpha-KIV cannot interconvert, and ILV6 "
            "is deleted so the only Ilv6p present is the plasmid-borne one "
            "under test. The tube itself carries no ILV6; the screen's best "
            "hit, this host plus ILV6V110E, reached 378 +/- 10 mg/L."
        ),
        parent="CEN.PK2-1C",
        genotype=(
            "CEN.PK2-1C, his3::HIS3-PLEU1-yEGFP-PEST-TADH1-PTPI1-"
            "LEU41-410-TPGK1, bat1D::hphMX bat2D::lox71-kanMX-lox66 "
            "ilv6D::lox71-natMX-lox66"
        ),
        # 378 +/- 10 mg/L is YZy91 carrying the ILV6V110E plasmid, which is a
        # different strain from the tube. The host alone has no reported titer.
        titer=None,
        quote=(
            "the parent strain of these libraries (YZy91) contains BAT1 and "
            "BAT2 deletions, which eliminates interconversion between valine "
            "and alpha-KIV, as well as an ILV6 deletion to rule out endogenous "
            "valine inhibition of Ilv2p (Supplementary Table 1, Fig. 3a)"
        ),
        quote_locator="Results, Ilv6p mutant screen",
    ),
    Strain(
        collection_id="C0053",
        production_carbon="2% glucose, dark",
        pick_and_use=False,
        tube_label="YZy502",
        paper_strain="YZy502",
        color_group="gray",
        configuration="isobutanol",
        role=(
            "Optogenetic host: triple pdcD with light-inducible PDC1 under "
            "OptoEXP and the dark-inducible cytosolic pathway under "
            "OptoINVRT7, delta-integrated. One plasmid and two FACS rounds "
            "short of the paper's best isobutanol strain."
        ),
        parent="YZy487",
        genotype=(
            "YZy487, d-integration-PC120-PDC1-TACT1-PTDH3-"
            "Ec_ilvCP2D1-A1-TCYC1-PTEF1-Ll_ilvDI433V-TTPS1-PGAL1-S-"
            "Bs_alsS-TACT1"
        ),
        # Fig. 6b plots YZy502, but the main text gives no number for it. The
        # only ratio stated, 20-fold, is YZy505 over YZy480, a different strain,
        # so nothing here can be back-computed.
        titer=None,
        quote=(
            "The starting strain for this experiment (YZy502) has a triple PDC "
            "deletion background with a light-inducible PDC1, a dark-inducible "
            "cytosolic isobutanol pathway, and our biosensor in its isobutanol "
            "configuration (Supplementary Table 1)."
        ),
        quote_locator="Results, optogenetic strains",
    ),
]


class NotHeld(BaseModel):
    """A strain the paper reports that is one construction step past a tube we hold."""

    paper_strain: str
    built_from: str
    step: str
    titer: Titer | None = None


NOT_HELD: list[NotHeld] = [
    # The 2x2 of Fig. 3c is missing exactly one cell, and it is the cell that
    # carries the improvement: YZy91 with the hit allele.
    NotHeld(
        paper_strain="YZy312 (Strain B)",
        built_from="C0052 YZy91",
        step=(
            "transform with pYZ228, the CEN URA3 plasmid carrying "
            "PTDH3-ILV6V110E-TADH1; this completes the Fig. 3c 2x2 that C0043, "
            "C0044 and C0045 are three quarters of"
        ),
        titer=None,
    ),
    NotHeld(
        paper_strain="YZy505",
        built_from="C0053 YZy502",
        step=(
            "transform with the 2 micron plasmid pYZ350 (two copies of "
            "Ec_ilvCP2D1-A1, one each of Ll_IlvDI433V, ARO10, Ll_adhARE1), "
            "then two rounds of FACS"
        ),
        titer=Titer(
            product="isobutanol",
            mg_per_l=681.0,
            sd=29.0,
            condition="2% glucose, dark",
            figure="Fig. 6b",
        ),
    ),
    NotHeld(
        paper_strain="YZy470",
        built_from="C0051 YZy469",
        step="move Ll_ilvDI433V from the CEN plasmid to a 2 micron plasmid",
        titer=Titer(
            product="isobutanol",
            mg_per_l=443.0,
            sd=6.0,
            condition="15% glucose",
            figure="Fig. 5c",
        ),
    ),
    NotHeld(
        paper_strain="YZy148 + LEU4 mutant #6",
        built_from="C0046 YZy148",
        step=(
            "transform with the evolved LEU4 plasmid "
            "(E86D K191N K374R A445T S481R N515I A568V S601A)"
        ),
        titer=Titer(
            product="isopentanol",
            mg_per_l=963.0,
            sd=32.0,
            condition="48 h fermentation",
            figure="Fig. 4",
        ),
    ),
]


def _fmt_titer(t: Titer | None) -> str:
    if t is None:
        return "--"
    value = f"{t.mg_per_l:.0f}"
    if t.sd is not None:
        value += f" $\\pm$ {t.sd:.0f}"
    if t.derived:
        value += "$^{\\dagger}$"
    return value


def _tex_escape(text: str) -> str:
    out = (
        text.replace("%", "\\%")
        .replace("&", "\\&")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )
    out = out.replace("+/-", "$\\pm$")
    # Greek and delta forms are written ASCII in the records so the source file
    # stays diffable; they become real symbols only on the way into LaTeX.
    out = out.replace(
        "DBAT1/DLEU4/DLEU9",
        "\\gene{bat1}$\\Delta$ \\gene{leu4}$\\Delta$ \\gene{leu9}$\\Delta$",
    )
    out = out.replace("LEU4DS547", "\\gene{LEU4}$\\Delta$S547")
    out = out.replace("Leu4DS547", "Leu4p$\\Delta$S547")
    out = out.replace("alpha-KIV", "$\\alpha$-KIV")
    out = out.replace("2 micron", "2$\\mu$")
    out = out.replace("delta-site", "$\\delta$-site")
    out = out.replace("delta-integrated", "$\\delta$-integrated")
    for gene in (
        "BAT1",
        "BAT2",
        "ILV6",
        "ILV3",
        "PDC1",
        "PDC5",
        "PDC6",
        "LEU4",
        "LEU9",
        "TMA29",
        "URA3",
        "GAL80",
    ):
        # Supplementary Table 1 writes a deletion lowercase ("bat1D::hphMX"),
        # the running text sometimes uppercase. Both become the same italic
        # lowercase gene name plus a delta, which is SGD style.
        out = out.replace(f"{gene}D", f"\\gene{{{gene.lower()}}}$\\Delta$")
        out = out.replace(f"{gene.lower()}D", f"\\gene{{{gene.lower()}}}$\\Delta$")
    out = out.replace("pdcD", "\\gene{pdc}$\\Delta$")
    # "d-integration" is the SI's ASCII for delta-site integration.
    out = out.replace("d-integration", "$\\delta$-integration")
    # Gene names in roman prose become italic; done last so the delta forms
    # above, which already carry their own \gene{}, are not double-wrapped.
    for gene in (
        "Ll\\_ilvD",
        "Ec\\_ilvC",
        "ILV6",
        "ILV2",
        "LEU4",
        "PDC1",
        "ARO10",
        "BAT1",
        "BAT2",
    ):
        out = out.replace(f" {gene} ", f" \\gene{{{gene}}} ")
        out = out.replace(f" {gene},", f" \\gene{{{gene}}},")
        out = out.replace(f" {gene}.", f" \\gene{{{gene}}}.")
    return out


def write_panel_table(path: str) -> None:
    lines = [
        f"%% SOURCE: experiments/{EXPERIMENT}/scripts/strain_panel.py",
        f"%% Mirror: $DATA_ROOT/torchcell-library/{CITATION_KEY}/paper.pdf",
        f"%% sha256 {PAPER_SHA256}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{\\textbf{The eleven strains on hand, and what each one is in "
        "the paper that built them.} Color groups are as the tubes are labeled. "
        "Titers are not comparable down the column: the carbon source differs by "
        "row and two rows are a different product. A dagger marks a value we "
        "back-computed from a reported fold change rather than one the paper "
        "prints.}",
        "\\label{tab:panel}",
        "\\footnotesize",
        "\\begin{tabular}{@{}llll >{\\raggedright\\arraybackslash}p{52mm} l@{}}",
        "\\toprule",
        "ID & Tube & Group & Config. & What it is & Titer (mg/L) \\\\",
        "\\midrule",
    ]
    for s in STRAINS:
        config = "--" if s.configuration in ("unknown", "none") else s.configuration
        lines.append(
            f"{s.collection_id} & {_tex_escape(s.tube_label)} & {s.color_group} & "
            f"{config} & {_tex_escape(s.role)} & {_fmt_titer(s.titer)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_genotype_table(path: str) -> None:
    """Supplementary Table 1's rows for the eleven tubes, verbatim."""
    lines = [
        f"%% SOURCE: experiments/{EXPERIMENT}/scripts/strain_panel.py",
        f"%% Mirror: $DATA_ROOT/torchcell-library/{CITATION_KEY}/si/si3.pdf",
        f"%% sha256 {SI3_SHA256}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{\\textbf{Genotypes, as Supplementary Table 1 prints them, "
        "and which strains carry a plasmid.} All eleven are \\org{S.\\ "
        "cerevisiae}; none is an \\org{E.\\ coli} strain. Parent names the "
        "strain each was built from, so a chain can be walked back to "
        "CEN.PK2-1C (MATa \\gene{ura3}-52 \\gene{trp1}-289 \\gene{leu2}-3,112 "
        "\\gene{his3}-1 MAL2-8c SUC2). Seven of the eleven carry an episomal "
        "plasmid and four do not. Every plasmid is the same backbone, AmpR "
        "CEN URA3, off the empty parent pYZ125, so each of those seven must be "
        "held on uracil dropout or it segregates the plasmid and reverts to "
        "its parent, and each has URA3 spent. The plasmids themselves are "
        "propagated in \\org{E.\\ coli} DH5$\\alpha$, which makes a plasmid a "
        "separate request from a strain.}",
        "\\label{tab:genotypes}",
        "\\footnotesize",
        "\\begin{tabular}{@{}llll >{\\raggedright\\arraybackslash}p{72mm}@{}}",
        "\\toprule",
        "ID & Strain & Parent & Plasmid & Genotype \\\\",
        "\\midrule",
    ]
    for s in STRAINS:
        plasmid = s.plasmid.name if s.plasmid else "none"
        lines.append(
            f"{s.collection_id} & {_tex_escape(s.tube_label)} & "
            f"{_tex_escape(s.parent)} & {plasmid} & "
            f"{_tex_escape(s.genotype)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_gap_table(path: str) -> None:
    lines = [
        f"%% SOURCE: experiments/{EXPERIMENT}/scripts/strain_panel.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{\\textbf{Three of the paper's headline strains are one "
        "construction step past a tube already in the freezer.} Each row is a "
        "single transformation, and in two cases a FACS enrichment, away from a "
        "strain we hold.}",
        "\\label{tab:gap}",
        "\\footnotesize",
        "\\begin{tabular}{@{}ll >{\\raggedright\\arraybackslash}p{62mm} l@{}}",
        "\\toprule",
        "Strain & From & Step & Titer (mg/L) \\\\",
        "\\midrule",
    ]
    for n in NOT_HELD:
        lines.append(
            f"{_tex_escape(n.paper_strain)} & {n.built_from} & "
            f"{_tex_escape(n.step)} & {_fmt_titer(n.titer)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_round_one_table(path: str) -> None:
    """Round one, and how every tube is grown if it is picked up at all.

    Ordered so the five in the round come first, then the rest in ID order,
    because the table answers two questions and the first one is "what do I
    inoculate on Monday".
    """
    lines = [
        f"%% SOURCE: experiments/{EXPERIMENT}/scripts/strain_panel.py",
        f"%% Growth conditions from the Methods of {CITATION_KEY}:",
        "%% single colony -> 1 mL SC or SC-ura + 2% glucose overnight -> 10 uL into",
        "%% 1 mL of the same in a 24-well plate -> 20 h -> spin, resuspend in 1 mL",
        "%% of the same + 15% sugar -> seal -> 48 h, 30 C, 200 rpm -> OD600, HPLC.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{\\textbf{Round one, and how each tube is grown.} Medium is "
        "uracil dropout exactly when a CEN \\gene{URA3} plasmid has to be held, "
        "and the fermentation carbon source is not a free choice for C0049. "
        "Every strain marked yes runs on the paper's ordinary 24-well protocol: "
        "single colony, shaker, plate reader, HPLC, with no sorting and no "
        "special illumination. Leucine must be present in all cases, because "
        "\\gene{leu2}$\\Delta$ makes these strains leucine auxotrophs; standard "
        "SC supplies it.}",
        "\\label{tab:roundone}",
        "\\footnotesize",
        "\\begin{tabular}{@{}lllll >{\\raggedright\\arraybackslash}p{50mm}@{}}",
        "\\toprule",
        "ID & Strain & Medium & Fermentation & Pick and use & Round one \\\\",
        "\\midrule",
    ]
    ordered = sorted(STRAINS, key=lambda s: (s.round_one == "", s.collection_id))
    in_round = sum(1 for s in STRAINS if s.round_one)
    for i, s in enumerate(ordered):
        if i == in_round:
            lines.append("\\midrule")
        lines.append(
            f"{s.collection_id} & {_tex_escape(s.tube_label)} & {s.medium} & "
            f"{_tex_escape(s.production_carbon)} & "
            f"{'yes' if s.pick_and_use else '\\textbf{no}'} & "
            f"{_tex_escape(s.round_one) if s.round_one else '--'} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_csv(path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "collection_id",
                "tube_label",
                "paper_strain",
                "color_group",
                "configuration",
                "parent",
                "medium",
                "production_carbon",
                "pick_and_use",
                "round_one",
                "plasmid",
                "plasmid_cargo",
                "genotype",
                "role",
                "product",
                "mg_per_l",
                "sd",
                "condition",
                "figure",
                "derived",
                "quote",
                "quote_locator",
            ]
        )
        for s in STRAINS:
            t = s.titer
            w.writerow(
                [
                    s.collection_id,
                    s.tube_label,
                    s.paper_strain,
                    s.color_group,
                    s.configuration,
                    s.parent,
                    s.medium,
                    s.production_carbon,
                    s.pick_and_use,
                    s.round_one,
                    s.plasmid.name if s.plasmid else "",
                    s.plasmid.cargo if s.plasmid else "",
                    s.genotype,
                    s.role,
                    t.product if t else "",
                    t.mg_per_l if t else "",
                    t.sd if t and t.sd is not None else "",
                    t.condition if t else "",
                    t.figure if t else "",
                    t.derived if t else "",
                    s.quote,
                    s.quote_locator,
                ]
            )


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    write_panel_table(osp.join(TABLES_DIR, "t1-strain-panel.tex"))
    write_gap_table(osp.join(TABLES_DIR, "t2-one-step-away.tex"))
    write_genotype_table(osp.join(TABLES_DIR, "t3-genotypes.tex"))
    write_round_one_table(osp.join(TABLES_DIR, "t4-round-one.tex"))
    write_csv(osp.join(RESULTS_DIR, "strain_panel.csv"))

    print(f"wrote {len(STRAINS)} strains, {len(NOT_HELD)} one step away")
    print(f"  tables -> {TABLES_DIR}")
    print(f"  csv    -> {osp.join(RESULTS_DIR, 'strain_panel.csv')}")


if __name__ == "__main__":
    main()
