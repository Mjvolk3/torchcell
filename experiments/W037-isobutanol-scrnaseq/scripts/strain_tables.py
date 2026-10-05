#!/usr/bin/env python
# experiments/W037-isobutanol-scrnaseq/scripts/strain_tables.py
# [[experiments.W037-isobutanol-scrnaseq.strain-selection]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W037-isobutanol-scrnaseq/scripts/strain_tables.py
"""Strain records for the twelve JC strains on hand, and the tables they emit.

Every value here is read from the primary paper or its Supplementary Information
(Zhang et al. 2022, Nat Commun 13:270, doi 10.1038/s41467-021-27852-x) and
carries the locator it was read from plus a verbatim quote. Two kinds of titer
are distinguished and the distinction is load bearing:

* ``stated`` -- the number appears in the paper as a number.
* ``derived`` -- the paper gives only a fold change against another strain, so
  the absolute was obtained by dividing. The absolute lives only in a figure bar
  and the Source Data file, which is not held locally. A derived value is
  approximate and is printed with a leading tilde.

The source PDFs are the canonical Zotero copies, pinned by sha256 in
``SOURCE_ARTIFACTS``. ``--verify`` recomputes those hashes. The OCR mirror
(``$DATA_ROOT/torchcell-library``) does not exist on this machine, so the Zotero
PDF is the artifact of record here.

Usage:
    python experiments/W037-isobutanol-scrnaseq/scripts/strain_tables.py
    python experiments/W037-isobutanol-scrnaseq/scripts/strain_tables.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import os.path as osp
from enum import Enum

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #

ZOTERO_ROOT = "/Users/michaelvolk/Zotero/storage"

SOURCE_ARTIFACTS: dict[str, dict[str, str]] = {
    "paper": {
        "path": osp.join(
            ZOTERO_ROOT,
            "RW8U8ERB",
            "Zhang et al. - 2022 - Biosensor for branched-chain amino acid "
            "metabolism in yeast and applications in isobutanol and isope.pdf",
        ),
        "sha256": "551ba08ec3bf584ef49b72186050d4bb3230df59edfa664a8a6af3460a2a8bec",
        "role": "primary article",
    },
    "si": {
        "path": osp.join(ZOTERO_ROOT, "669BLB7N", "41467_2021_27852_MOESM1_ESM.pdf"),
        "sha256": "74de9b6cff654c5d9b6988a0a0c08672eb7defad4f002ecce6b9e6b9fa48824f",
        "role": "Supplementary Information (Tables 1-9, Notes 1-5)",
    },
}

CITATION_KEY = "zhangBiosensorBranchedchainAmino2022"
DOI = "10.1038/s41467-021-27852-x"

HOST = "CEN.PK2-1C"
HOST_GENOTYPE = r"MATa ura3-52 trp1-289 leu2-3,112 his3-1 MAL2-8c SUC2"
HOST_QUOTE = (
    "CEN.PK2-1C Wild-type Saccharomyces cerevisiae MATa ura3-52 trp1-289 "
    "leu2-3,112 his3-1 MAL2-8c SUC2"
)


class Product(str, Enum):
    ISOBUTANOL = "isobutanol"
    ISOPENTANOL = "isopentanol"


class Compartment(str, Enum):
    NONE = "none"
    MITOCHONDRIAL = "mitochondrial"
    CYTOSOLIC = "cytosolic"


class Evidence(str, Enum):
    STATED = "stated"
    DERIVED = "derived"
    NOT_REPORTED = "not reported"


class Provenance(BaseModel):
    """Where one value was read from, and the words it was read from."""

    artifact: str = Field(description="key into SOURCE_ARTIFACTS")
    locator: str = Field(description="table row, figure panel, or text location")
    quote: str = Field(default="", description="verbatim source text")


class Titer(BaseModel):
    value_mg_l: float | None = None
    sd_mg_l: float | None = None
    carbon: str = ""
    evidence: Evidence = Evidence.NOT_REPORTED
    derivation: str = ""
    #: Set when the paper gives a fold change but no absolute for either strain,
    #: so there is nothing to divide into. Rendering the fold keeps the only
    #: quantitative statement the paper makes about this strain in the table.
    fold: float | None = None
    fold_vs: str = ""
    provenance: Provenance | None = None

    def render(self) -> str:
        """LaTeX cell. A derived value is marked so it cannot be mistaken."""
        if self.value_mg_l is None:
            if self.fold is not None:
                return rf"{self.fold:g}$\times$ {self.fold_vs}"
            return "not reported"
        if self.evidence is Evidence.DERIVED:
            return rf"$\sim${self.value_mg_l:.0f}\,$^{{\dagger}}$"
        if self.sd_mg_l is not None:
            return rf"\textbf{{{self.value_mg_l:.0f} $\pm$ {self.sd_mg_l:.0f}}}"
        return f"{self.value_mg_l:.0f}"


def tex_escape_underscore(text: str) -> str:
    """Escape an underscore that is not already escaped.

    Gene names here are bacterial (``Ll_ilvD``, ``Bs_alsS``), so a bare
    underscore reaches LaTeX as a subscript operator in text mode and aborts the
    build. Escaping at the render boundary means a record can hold the gene name
    as it is written in the paper.
    """
    out, prev = [], ""
    for ch in text:
        out.append(r"\_" if ch == "_" and prev != "\\" else ch)
        prev = ch
    return "".join(out)


class Plasmid(BaseModel):
    name: str = ""
    backbone: str = ""
    cargo: str = ""

    @property
    def is_present(self) -> bool:
        return bool(self.name)

    def render(self) -> str:
        return tex_escape_underscore(self.cargo) if self.cargo else "none"


class Strain(BaseModel):
    jc_id: str
    paper_name: str
    parent: str = ""
    deletions: list[str] = Field(default_factory=list)
    leu2_restored: bool = False
    biosensor: str = Field(default="", description="cassette at its integration locus")
    pathway_cassettes: list[str] = Field(default_factory=list)
    compartment: Compartment = Compartment.NONE
    delta_integrated: bool = False
    plasmid: Plasmid = Field(default_factory=Plasmid)
    markers: list[str] = Field(default_factory=list)
    product: Product = Product.ISOBUTANOL
    titer: Titer = Field(default_factory=Titer)
    selected: bool = False
    axis: str = ""
    role: str = ""
    defer_reason: str = ""
    genotype: str = ""
    provenance: Provenance

    @property
    def n_added_cassettes(self) -> int:
        """Cassettes added, excluding selection markers.

        The reporter counts, because it is transcribed and will appear in a
        single-cell library like any other transgene.
        """
        n = len(self.pathway_cassettes) + len(self.biosensor.split("+"))
        return n + (1 if self.plasmid.is_present else 0)

    @property
    def media(self) -> str:
        return "SC-ura" if self.plasmid.is_present else "SC + uracil"


# --------------------------------------------------------------------------- #
# Records. Locators name Supplementary Table 1 unless stated otherwise.
# --------------------------------------------------------------------------- #

BIOSENSOR_ISOBUTANOL = "yEGFP-PEST + LEU4(1-410)"
BIOSENSOR_ISOPENTANOL_BARE = "yEGFP"
BIOSENSOR_ISOPENTANOL_FULL = "yEGFP + LEU4dS547"

MITO_PATHWAY = [
    "ILV2",
    "ILV3",
    "ILV5",
    "mito-ARO10",
    "mito-Ll_adhA(RE1)",
]
CYTO_PATHWAY = [
    "AFT1",
    "Bs_alsS",
    "Ec_ilvC(P2D1-A1)",
    "Ec_ilvC(P2D1-A1) x2 (delta)",
    "P_GAL10-Ll_ilvD",
]
OPTO_CASSETTES = [
    "EL222",
    "GAL80-ODCmut x2",
    "GAL4",
    "light-PDC1",
    "Ec_ilvC(P2D1-A1)",
    "Ll_ilvD(I433V)",
    "Bs_alsS",
]

STRAINS: list[Strain] = [
    Strain(
        jc_id="JC0052",
        paper_name="YZy91",
        parent=HOST,
        deletions=["bat1", "bat2", "ilv6"],
        biosensor=BIOSENSOR_ISOBUTANOL,
        markers=["hphMX", "kanMX", "natMX", "HIS3"],
        product=Product.ISOBUTANOL,
        titer=Titer(
            carbon="15\\% glucose",
            evidence=Evidence.NOT_REPORTED,
            provenance=Provenance(
                artifact="paper",
                locator="Fig. 3b, control bars",
                quote=(
                    "Titers obtained from YZy91 with (red) or without (blue) an "
                    "empty vector"
                ),
            ),
        ),
        selected=True,
        axis="1",
        role="ilv6 deletion state",
        genotype=(
            r"\textit{his3}::HIS3-P\psub{LEU1}-yEGFP-PEST-T\psub{ADH1}-"
            r"P\psub{TPI1}-LEU4\gsup{1-410}-T\psub{PGK1}, "
            r"\textit{bat1}$\Delta$::hphMX, \textit{bat2}$\Delta$::lox71-kanMX-lox66, "
            r"\textit{ilv6}$\Delta$::lox71-natMX-lox66"
        ),
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 1, YZy91",
            quote=(
                "bat1D, bat2D, ilv6D, isobutanol-configured biosensor (cassette "
                "from pYZ16)"
            ),
        ),
    ),
    Strain(
        jc_id="JC0043",
        paper_name="YZy311",
        parent="YZy91",
        deletions=["bat1", "bat2", "ilv6"],
        biosensor=BIOSENSOR_ISOBUTANOL,
        plasmid=Plasmid(name="pYZ127", backbone="CEN/URA3", cargo="ILV6 WT"),
        markers=["hphMX", "kanMX", "natMX", "HIS3", "URA3"],
        product=Product.ISOBUTANOL,
        titer=Titer(
            value_mg_l=118.0,
            carbon="15\\% glucose",
            evidence=Evidence.DERIVED,
            derivation="378 mg/L stated as 3.2-fold over wild-type ILV6",
            provenance=Provenance(
                artifact="paper",
                locator="main text, ILV6 variants; Fig. 3b green bar",
                quote=(
                    "produces the most isobutanol (378 mg/L +/- 10 mg/L), which "
                    "is a 3.2-fold increase over the wild-type ILV6"
                ),
            ),
        ),
        selected=True,
        axis="1",
        role="ILV6 add-back",
        genotype=r"YZy91, CEN URA3 plasmid (P\psub{TDH3}-ILV6-T\psub{ADH1})",
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 1, YZy311",
            quote="YZy311 YZy91, pYZ127 YZy91, CEN URA3 plasmid (PTDH3-ILV6-TADH1)",
        ),
    ),
    Strain(
        jc_id="JC0044",
        paper_name="YZy313",
        parent="YZy363",
        deletions=["bat1"],
        biosensor=BIOSENSOR_ISOBUTANOL,
        pathway_cassettes=MITO_PATHWAY,
        compartment=Compartment.MITOCHONDRIAL,
        delta_integrated=True,
        plasmid=Plasmid(name="pYZ127", backbone="CEN/URA3", cargo="ILV6 WT"),
        markers=["hphMX", "HIS3", "ShBle", "URA3"],
        product=Product.ISOBUTANOL,
        titer=Titer(
            carbon="15\\% glucose",
            evidence=Evidence.NOT_REPORTED,
            provenance=Provenance(
                artifact="paper",
                locator="Fig. 3c, Strain C",
                quote=(
                    "a strain overexpressing the mitochondrial isobutanol pathway "
                    "(YZy363) transformed with CEN plasmids containing ILV6WT "
                    "(Strain C)"
                ),
            ),
        ),
        selected=True,
        axis="2",
        role="mitochondrial pathway, ILV6 WT",
        genotype=(
            r"YZy363 ($\delta$-integrated P\psub{TDH3}-ILV2, P\psub{PGK1}-ILV3, "
            r"P\psub{TEF1}-CoxIVMLS-Ll\_adhA\gsup{RE1}, "
            r"P\psub{TDH3}-CoxIVMLS-ARO10, P\psub{TEF1}-ILV5), "
            r"CEN URA3 plasmid (P\psub{TDH3}-ILV6-T\psub{ADH1})"
        ),
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 1, YZy313 and YZy363",
            quote="YZy313 YZy363, pYZ127 YZy363, CEN URA3 plasmid (PTDH3-ILV6-TADH1)",
        ),
    ),
    Strain(
        jc_id="JC0045",
        paper_name="YZy314",
        parent="YZy363",
        deletions=["bat1"],
        biosensor=BIOSENSOR_ISOBUTANOL,
        pathway_cassettes=MITO_PATHWAY,
        compartment=Compartment.MITOCHONDRIAL,
        delta_integrated=True,
        plasmid=Plasmid(name="pYZ228", backbone="CEN/URA3", cargo="ILV6 V110E"),
        markers=["hphMX", "HIS3", "ShBle", "URA3"],
        product=Product.ISOBUTANOL,
        titer=Titer(
            carbon="15\\% glucose",
            evidence=Evidence.DERIVED,
            derivation="1.6-fold over JC0044; no absolute given for either",
            fold=1.6,
            fold_vs="JC0044",
            provenance=Provenance(
                artifact="paper",
                locator="main text, ILV6 V110E in pathway strain; Fig. 3c Strain D",
                quote=(
                    "the production is 1.6 times higher than the equivalent strain "
                    "overexpressing the wild-type ILV6"
                ),
            ),
        ),
        selected=True,
        axis="2",
        role="mitochondrial pathway, ILV6 V110E",
        genotype=(
            r"YZy363, CEN URA3 plasmid "
            r"(P\psub{TDH3}-ILV6\gsup{V110E}-T\psub{ADH1})"
        ),
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 1, YZy314",
            quote=(
                "YZy314 YZy363, pYZ228 YZy363, CEN URA3 plasmid "
                "(PTDH3-ILV6V110E-TADH1)"
            ),
        ),
    ),
    Strain(
        jc_id="JC0050",
        paper_name="YZy454",
        parent="YZy452",
        deletions=["ilv3", "tma29"],
        biosensor=BIOSENSOR_ISOBUTANOL,
        pathway_cassettes=CYTO_PATHWAY,
        compartment=Compartment.CYTOSOLIC,
        delta_integrated=True,
        plasmid=Plasmid(name="pYZ126", backbone="CEN/URA3", cargo="Ll_ilvD WT"),
        markers=["hphMX", "kanMX", "HIS3", "ShBle", "URA3"],
        product=Product.ISOBUTANOL,
        titer=Titer(
            value_mg_l=73.0,
            carbon="15\\% glucose",
            evidence=Evidence.DERIVED,
            derivation="227 mg/L stated as 3.1-fold over wild-type Ll_ilvD",
            provenance=Provenance(
                artifact="paper",
                locator="main text, Ll_ilvD variants; Fig. 5c",
                quote=(
                    "produces 227 +/- 13 mg/L of isobutanol from 15% glucose, "
                    "which is 3.1-fold higher than the strain harboring the "
                    "wild-type Ll_ilvD (YZy454)"
                ),
            ),
        ),
        selected=True,
        axis="3",
        role="cytosolic pathway, Ll_ilvD WT",
        genotype=(
            r"YZy452, CEN URA3 plasmid (P\psub{TDH3}-Ll\_ilvD-T\psub{ADH1})"
        ),
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 1, YZy454",
            quote=(
                "YZy454 YZy452, pYZ126 YZy452, CEN URA3 plasmid "
                "(PTDH3-Ll_ilvD-TADH1)"
            ),
        ),
    ),
    Strain(
        jc_id="JC0051",
        paper_name="YZy469",
        parent="YZy452",
        deletions=["ilv3", "tma29"],
        biosensor=BIOSENSOR_ISOBUTANOL,
        pathway_cassettes=CYTO_PATHWAY,
        compartment=Compartment.CYTOSOLIC,
        delta_integrated=True,
        plasmid=Plasmid(name="pYZ353", backbone="CEN/URA3", cargo="Ll_ilvD I433V"),
        markers=["hphMX", "kanMX", "HIS3", "ShBle", "URA3"],
        product=Product.ISOBUTANOL,
        titer=Titer(
            value_mg_l=227.0,
            sd_mg_l=13.0,
            carbon="15\\% glucose",
            evidence=Evidence.STATED,
            provenance=Provenance(
                artifact="paper",
                locator="main text, Ll_ilvD variants; Fig. 5c",
                quote=(
                    "The strain retransformed with the most active variant "
                    "Ll_ilvDI433V (YZy469) produces 227 +/- 13 mg/L of isobutanol "
                    "from 15% glucose"
                ),
            ),
        ),
        selected=True,
        axis="3",
        role="cytosolic pathway, Ll_ilvD I433V",
        genotype=(
            r"YZy452, CEN URA3 plasmid "
            r"(P\psub{TDH3}-Ll\_ilvD\gsup{I433V}-T\psub{ADH1})"
        ),
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 1, YZy469",
            quote=(
                "YZy469 YZy452, pYZ353 YZy452, CEN URA3 plasmid "
                "(PTDH3-Ll_ilvDI433V-TADH1)"
            ),
        ),
    ),
    # ------------------------------------------------------------------ #
    # On hand, not selected for the first run.
    # ------------------------------------------------------------------ #
    Strain(
        jc_id="JC0049",
        paper_name="YZy452",
        parent="YZy449",
        deletions=["ilv3", "tma29"],
        biosensor=BIOSENSOR_ISOBUTANOL,
        pathway_cassettes=CYTO_PATHWAY,
        compartment=Compartment.CYTOSOLIC,
        delta_integrated=True,
        markers=["hphMX", "kanMX", "HIS3", "ShBle"],
        product=Product.ISOBUTANOL,
        titer=Titer(
            value_mg_l=310.0,
            sd_mg_l=15.0,
            carbon="15\\% galactose",
            evidence=Evidence.STATED,
            provenance=Provenance(
                artifact="paper",
                locator="main text, delta-integrated Ec_ilvC",
                quote=(
                    "the highest producing strain (YZy452) achieving 310 +/- 15 "
                    "mg/L isobutanol"
                ),
            ),
        ),
        axis="3",
        role="plasmid-free parent of JC0050 and JC0051",
        defer_reason=(
            r"galactose only: its chromosomal \textit{Ll\_ilvD} sits behind "
            r"P\psub{GAL10}, so carbon source would confound every comparison "
            r"in a shared run"
        ),
        genotype=(
            r"\textit{ilv3}$\Delta$, \textit{tma29}$\Delta$, isobutanol biosensor, "
            r"\textit{ura3}::P\psub{TDH3}-AFT1-[P\psub{TEF1}-Bs\_alsS-"
            r"P\psub{TDH3}-Ec\_ilvC\gsup{P2D1-A1}]-P\psub{GAL10}-Ll\_ilvD, "
            r"$\delta$::Ec\_ilvC\gsup{P2D1-A1} $\times$ 2"
        ),
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 1, YZy452",
            quote=(
                "YZy452 YZy449, (cassette from pYZ206) YZy449, delta-integration-"
                "PTDH3-Ec_ilvCP2D1-A1-TADH1-[PTEF1- Ec_ilvCP2D1-A1-TACT1]"
            ),
        ),
    ),
    Strain(
        jc_id="JC0053",
        paper_name="YZy502",
        parent="YZy487",
        deletions=["pdc1", "pdc5", "pdc6", "gal80"],
        biosensor=BIOSENSOR_ISOBUTANOL,
        pathway_cassettes=OPTO_CASSETTES,
        compartment=Compartment.CYTOSOLIC,
        delta_integrated=True,
        markers=["natMX", "HIS3", "ShBle"],
        product=Product.ISOBUTANOL,
        titer=Titer(
            carbon="2\\% glucose, dark",
            evidence=Evidence.NOT_REPORTED,
            derivation=(
                "its derivative YZy505, carrying 2mu pYZ350, is reported at "
                "681 +/- 29 mg/L"
            ),
            provenance=Provenance(
                artifact="paper",
                locator="Methods, applying the biosensor in optogenetic strains",
                quote=(
                    "The colony with the highest GFP fluorescence intensity, "
                    "corresponding to the highest isobutanol titer, was chosen as "
                    "the host strain (YZy502)"
                ),
            ),
        ),
        axis="",
        role="plasmid-free parent of the paper's best isobutanol strain",
        defer_reason=(
            "triple \\textit{pdc}$\\Delta$ with \\gene{PDC1} restored only under a "
            "blue-light promoter, so growth and production are separate light "
            "phases and a light-shift response would sit on top of the production "
            "response"
        ),
        genotype=(
            r"\textit{pdc1}$\Delta$ \textit{pdc5}$\Delta$ \textit{pdc6}$\Delta$, "
            r"\textit{gal80}::biosensor, \textit{his3}::OptoINVRT7 "
            r"(EL222, P\psub{C120}-GAL80-ODCmut $\times$ 2, GAL4), "
            r"$\delta$::P\psub{C120}-PDC1, Ec\_ilvC\gsup{P2D1-A1}, "
            r"Ll\_ilvD\gsup{I433V}, Bs\_alsS"
        ),
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 1, YZy502",
            quote=(
                "YZy502 YZy487, OptoEXP PDC1 and OptoINVRT7 cytosolic isobutanol "
                "pathway"
            ),
        ),
    ),
    Strain(
        jc_id="JC0046",
        paper_name="YZy148",
        parent=HOST,
        deletions=["bat1", "leu4", "leu9"],
        leu2_restored=True,
        biosensor=BIOSENSOR_ISOPENTANOL_BARE,
        product=Product.ISOPENTANOL,
        titer=Titer(
            carbon="15\\% glucose",
            evidence=Evidence.NOT_REPORTED,
            provenance=Provenance(
                artifact="paper",
                locator="Fig. 4c, basal strain",
                quote="Titers obtained from the basal strain YZy148",
            ),
        ),
        role="isopentanol host for the LEU4 library",
        defer_reason="makes isopentanol, not isobutanol",
        genotype=(
            r"\textit{his3}::HIS3-P\psub{LEU1}-yEGFP-T\psub{ADH1}, "
            r"\textit{bat1}$\Delta$, \textit{leu4}$\Delta$, \textit{leu9}$\Delta$, "
            r"\textit{leu2}::LEU2"
        ),
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 1, YZy148",
            quote=(
                "bat1D, leu4D, leu9D, LEU2 restored, modified isopentanol-"
                "configured biosensor (cassette from pYZ24)"
            ),
        ),
    ),
    Strain(
        jc_id="JC0047",
        paper_name="YZy148 + LEU4 WT",
        parent="YZy148",
        deletions=["bat1", "leu4", "leu9"],
        leu2_restored=True,
        biosensor=BIOSENSOR_ISOPENTANOL_BARE,
        plasmid=Plasmid(name="pYZ149", backbone="CEN/URA3", cargo="LEU4 WT"),
        product=Product.ISOPENTANOL,
        titer=Titer(
            value_mg_l=201.0,
            carbon="15\\% glucose",
            evidence=Evidence.DERIVED,
            derivation="963 mg/L stated as 4.8 times the wild-type LEU4 titer",
            provenance=Provenance(
                artifact="paper",
                locator="main text, LEU4 variants; Fig. 4c",
                quote=(
                    "achieves the highest isopentanol titer (963 mg/L +/- 32 "
                    "mg/L), which is 4.8-times higher than the titer achieved by "
                    "overexpressing the wild-type LEU4"
                ),
            ),
        ),
        role="isopentanol reference arm",
        defer_reason="makes isopentanol, not isobutanol",
        genotype=r"YZy148, CEN URA3 plasmid (P\psub{TDH3}-LEU4-T\psub{ADH1})",
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 2, pYZ149",
            quote="pYZ149 AmpR, CEN, URA3, PTDH3-LEU4-TADH1",
        ),
    ),
    Strain(
        jc_id="JC0048",
        paper_name="YZy148 + LEU4dS547",
        parent="YZy148",
        deletions=["bat1", "leu4", "leu9"],
        leu2_restored=True,
        biosensor=BIOSENSOR_ISOPENTANOL_BARE,
        plasmid=Plasmid(name="pYZ154", backbone="CEN/URA3", cargo="LEU4dS547"),
        product=Product.ISOPENTANOL,
        titer=Titer(
            value_mg_l=842.0,
            sd_mg_l=23.0,
            carbon="15\\% glucose",
            evidence=Evidence.STATED,
            provenance=Provenance(
                artifact="paper",
                locator="main text, LEU4 variants; Fig. 4c",
                quote=(
                    "significantly higher than the titer obtained with the "
                    "LEU4dS547 mutant previously described (842 mg/L +/- 23 mg/L; "
                    "P-value = 0.0021)"
                ),
            ),
        ),
        role="highest reported titer among the twelve",
        defer_reason="makes isopentanol, not isobutanol",
        genotype=(
            r"YZy148, CEN URA3 plasmid "
            r"(P\psub{TDH3}-LEU4$\Delta$S547-T\psub{ADH1})"
        ),
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 2, pYZ154",
            quote="pYZ154 AmpR, CEN, URA3, PTDH3-LEU4dS547-TADH1",
        ),
    ),
    Strain(
        jc_id="JC0054",
        paper_name="SHy134",
        parent=HOST,
        deletions=["bat1", "leu4", "leu9"],
        leu2_restored=True,
        biosensor=BIOSENSOR_ISOPENTANOL_FULL,
        product=Product.ISOPENTANOL,
        titer=Titer(
            carbon="15\\% glucose",
            evidence=Evidence.NOT_REPORTED,
            provenance=Provenance(
                artifact="paper",
                locator="Fig. 1d, via SHy158 (SHy134 with empty vector)",
                quote="Strains used (from left to right): SHy187, SHy188, SHy192",
            ),
        ),
        role="isopentanol biosensor with LEU4dS547 integrated",
        defer_reason="makes isopentanol, not isobutanol",
        genotype=(
            r"\textit{his3}::HIS3-P\psub{LEU1}-yEGFP-T\psub{ADH1}-"
            r"P\psub{TPI1}-LEU4$\Delta$S547-T\psub{PGK1}, "
            r"\textit{bat1}$\Delta$, \textit{leu4}$\Delta$, \textit{leu9}$\Delta$, "
            r"\textit{leu2}::LEU2"
        ),
        provenance=Provenance(
            artifact="si",
            locator="Supplementary Table 1, SHy134",
            quote=(
                "bat1D, leu4D, leu9D, LEU2 restored, isopentanol-configured "
                "biosensor (cassette from pYZ25)"
            ),
        ),
    ),
]

JC_ORDER = [f"JC{n:04d}" for n in range(43, 55)]
SRC_LINE = (
    "%% SOURCE: experiments/W037-isobutanol-scrnaseq/scripts/strain_tables.py"
)
HEADER = "%% GENERATED FILE -- do not hand-edit.\n" + SRC_LINE + "\n"


def by_id() -> dict[str, Strain]:
    return {s.jc_id: s for s in STRAINS}


def _dels(s: Strain) -> str:
    if not s.deletions:
        return "--"
    return ", ".join(rf"\textit{{{d}}}$\Delta$" for d in s.deletions)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def table_inventory() -> str:
    """Every strain on hand, in JC order, with what it makes and how much."""
    rows = []
    idx = by_id()
    for jc in JC_ORDER:
        s = idx[jc]
        mark = r"\checkmark" if s.selected else ""
        rows.append(
            " & ".join(
                [
                    s.jc_id,
                    s.paper_name,
                    s.product.value,
                    s.titer.render(),
                    s.titer.carbon or "--",
                    mark,
                ]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    return f"""{HEADER}\\begin{{table}}[H]\\centering
\\footnotesize
\\caption[Strains on hand]{{The twelve strains on hand, all of them
\\org{{Saccharomyces cerevisiae}} {HOST} derivatives. \\emph{{Titer}} is the
48\\,h high-cell-density value for the product named beside it; a bold value is
printed in the paper as a number, and a value marked $\\dagger$ was obtained by
dividing a stated fold change, so it is approximate and carries no uncertainty.
Carbon source differs across the set and is not comparable between rows: the
three figures these numbers come from use 15\\% glucose, 15\\% galactose, and
2\\% glucose respectively. The last column marks the six selected for the first
run.}}\\label{{tab:inventory}}
\\begin{{tabular}}{{llllll}}
\\toprule
id & strain & product & titer (mg/L) & carbon & first run \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_selected() -> str:
    """The six, grouped by axis, with the genetics that distinguishes them."""
    idx = by_id()
    rows = []
    for axis in ("1", "2", "3"):
        members = [s for s in STRAINS if s.selected and s.axis == axis]
        for i, s in enumerate(members):
            lead = rf"\multirow{{2}}{{*}}{{{axis}}}" if i == 0 else ""
            rows.append(
                " & ".join(
                    [
                        lead,
                        s.jc_id,
                        s.paper_name,
                        _dels(s),
                        s.compartment.value if s.compartment.value != "none" else "--",
                        s.plasmid.render(),
                        s.titer.render(),
                    ]
                )
                + r" \\"
            )
        if axis != "3":
            rows.append(r"\addlinespace")
    body = "\n".join(rows)
    del idx
    return f"""{HEADER}\\begin{{table}}[H]\\centering
\\footnotesize
\\caption[The six selected]{{The six selected, as three axes of one variable
each. Within an axis the two strains share a background and differ by the
plasmid they carry, so a difference measured between them is attributable to
that plasmid. \\emph{{pathway}} names the compartment the heterologous
isobutanol route was built in. Every strain here ferments 15\\% glucose, so all
six run under one protocol. Titer conventions follow
Table~\\ref{{tab:inventory}}.}}\\label{{tab:selected}}
\\begin{{tabular}}{{lllllll}}
\\toprule
axis & id & strain & deletions & pathway & plasmid & titer (mg/L) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_genotypes() -> str:
    """Full genotype of each selected strain, for the bench."""
    rows = []
    for s in [x for x in STRAINS if x.selected]:
        rows.append(
            rf"\textbf{{{s.jc_id}}} & {s.paper_name} \\" + "\n"
            rf" & {{\scriptsize {s.genotype}}} \\" + "\n"
            rf" & {{\scriptsize markers: {', '.join(s.markers)}"
            rf" \quad medium: {s.media}}} \\"
        )
    body = "\n\\addlinespace\n".join(rows)
    return f"""{HEADER}\\begin{{table}}[H]\\centering
\\footnotesize
\\caption[Genotypes]{{Genotypes of the six, as Supplementary Table 1 of the
source paper gives them, with the selection markers and the growth medium each
one needs. \\emph{{medium}} follows from the plasmid: a strain carrying a
URA3 plasmid is held on SC-ura, and JC0052, which carries none, needs uracil
instead. Section~\\ref{{sec:before}} treats that mismatch, which is the one
thing in this list that has to be fixed before the strains are
grown.}}\\label{{tab:genotypes}}
\\begin{{tabular}}{{@{{}}l p{{0.86\\textwidth}}@{{}}}}
\\toprule
id & strain and genotype \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_deferred() -> str:
    """What is on hand but held back, and why."""
    rows = []
    for s in [x for x in STRAINS if not x.selected]:
        rows.append(
            " & ".join(
                [
                    s.jc_id,
                    s.paper_name,
                    s.product.value,
                    s.titer.render(),
                    f"{{\\scriptsize {s.defer_reason}}}",
                ]
            )
            + r" \\"
        )
    body = "\n\\addlinespace\n".join(rows)
    return f"""{HEADER}\\begin{{table}}[H]\\centering
\\footnotesize
\\caption[Held back]{{The six held back from the first run, with the reason in
each case. Four are held back because they make isopentanol rather than
isobutanol. The other two make isobutanol but cannot share a run with the
selected six: one produces only on galactose, and one needs a light schedule.
Titer conventions follow Table~\\ref{{tab:inventory}}.}}\\label{{tab:deferred}}
\\begin{{tabular}}{{llllp{{0.42\\textwidth}}}}
\\toprule
id & strain & product & titer (mg/L) & reason \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


TABLES = {
    "t1-inventory.tex": table_inventory,
    "t2-selected.tex": table_selected,
    "t3-genotypes.tex": table_genotypes,
    "t4-deferred.tex": table_deferred,
}

DOC_DIR = "notes-tex/w037-isobutanol-scrnaseq"


def verify() -> int:
    """Recompute the sha256 of each source PDF and compare to the pin."""
    bad = 0
    for key, rec in SOURCE_ARTIFACTS.items():
        path = rec["path"]
        if not osp.exists(path):
            print(f"MISSING  {key}: {path}")
            bad += 1
            continue
        with open(path, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        ok = got == rec["sha256"]
        print(f"{'OK      ' if ok else 'MISMATCH'} {key}  {got}")
        bad += 0 if ok else 1
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="recompute source PDF hashes and exit",
    )
    ap.add_argument("--doc-dir", default=DOC_DIR)
    args = ap.parse_args()

    if args.verify:
        raise SystemExit(verify())

    out_dir = osp.join(args.doc_dir, "tables")
    for name, fn in TABLES.items():
        path = osp.join(out_dir, name)
        with open(path, "w") as fh:
            fh.write(fn())
        print(f"wrote {path}")

    n_sel = sum(1 for s in STRAINS if s.selected)
    print(f"{len(STRAINS)} strains, {n_sel} selected, host {HOST}")


if __name__ == "__main__":
    main()
