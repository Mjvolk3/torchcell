# experiments/019-simb-multimodal/scripts/build_figure_option_boards.py
# [[experiments.019-simb-multimodal.scripts.build_figure_option_boards]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/build_figure_option_boards
"""Generate draw.io OPTION BOARDS for manuscript Figures 3 and 6.

WHAT AN OPTION BOARD IS. Not a finished figure. Each output is one `.drawio` file whose
first page is a PANEL BANK holding every candidate panel that exists today at true print
size, and whose remaining pages are alternative COMPOSITES built from that bank. Choosing
between compositions, or cutting one down, is then a matter of opening the file and
deleting pages rather than re-importing images. The boards are deliberately over-supplied:
a panel that turns out to be unnecessary is cheap to delete and expensive to rebuild.

WHY THIS IS A SCRIPT AND NOT A HAND-DRAWN FILE. Every panel in the bank traces to a
committed plot script, and the placement obeys the repo's Nature widths
(`torchcell.utils.PANEL_WIDTHS_MM`). Hand-placing them would lose both properties on the
first edit. Re-running this regenerates the bank from whatever the plot scripts currently
write, so a re-plotted panel does not silently leave a stale copy inside the figure.

UNITS. draw.io's native geometry unit is 1/100 inch, which is also the unit
`torchcell.utils.savefig_true_size_svg` rewrites its SVG roots into. So an SVG whose root
reads `width="704"` is 7.04 inches = 179 mm, and placing it in a cell of width 704 puts it
on the canvas at exactly that size. Panel geometry here is therefore taken from each SVG's
own root rather than assumed, and the page frame is drawn at Nature's 180 mm.

IMAGES ARE EMBEDDED, not linked. draw.io does not resolve relative image paths portably,
so each panel goes in as a base64 `data:image/svg+xml` URI. That makes the files a few
megabytes; the alternative is a figure that opens with broken images on another machine.

TEXT BOXES ARE KEPT. Several panels in the Figure 6 compositions are labeled with
alternatives that have not been run. Those boxes are intentionally left in place as
options, not deleted, so that a composition still reads if a planned result does not
arrive.

Run from repo root:
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/build_figure_option_boards.py
"""

from __future__ import annotations

import base64
import os
import os.path as osp
import re
from xml.sax.saxutils import escape

from dotenv import load_dotenv

from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, PLOT_PALETTE_FILL

# draw.io units per inch, and the conversion the panels were authored in.
UNITS_PER_INCH = 100.0
MM_PER_INCH = 25.4


def mm(value: float) -> float:
    return value * UNITS_PER_INCH / MM_PER_INCH


# Nature print frame. Full width 180 mm, single column 88 mm, max height 170 mm.
PAGE_W = mm(180.0)
PAGE_H = mm(170.0)
MARGIN = mm(4.0)
GAP = mm(3.0)

# Panel bank. (key, image directory slug, svg basename, one-line description).
# Ordered so the bank reads as an inventory rather than as a composition.
FIG3_PANELS = [
    ("ceiling", "019-simb-multimodal", "retrospective_achieved_vs_ceiling",
     "Achieved vs measured ceiling, all strands"),
    ("peak", "019-simb-multimodal", "retrospective_peak_position",
     "Where the peak sits inside the run"),
    ("residual", "019-simb-multimodal", "residual_covariance_diagnostic",
     "Residual gene-gene structure, effective rank 33"),
    ("proteome", "019-simb-multimodal",
     "proteome_expression_per_gene_corr_hist_2026-07-22-17-15-26",
     "mRNA vs protein per-gene correlation, median 0.08"),
    ("proteome_map", "019-simb-multimodal",
     "proteome_expression_linear_map_r2_2026-07-22-17-15-26",
     "Linear map between modalities, held-out R2"),
    ("knn", "019-simb-multimodal", "knn_embedding_probe",
     "kNN embedding probe on the perturbed side"),
    ("kem_spread", "012-sameith-kemmeren-expression", "single_mutant_kemmeren",
     "Kemmeren single-mutant expression spread"),
    ("sam_spread", "012-sameith-kemmeren-expression", "single_mutant_sameith",
     "Sameith single-mutant expression spread"),
    ("cross_corr", "012-sameith-kemmeren-expression",
     "gene_expression_correlation_dist_pearson",
     "Kemmeren vs Sameith per-gene agreement"),
]

FIG6_PANELS = [
    ("bx_ceiling", "019-simb-multimodal", "pigment_noise_ceiling",
     "Pigment reliability: betaxanthin 0.914, carotene 0.544"),
    ("aa_coupling", "023-metabolome-betaxanthin-joint",
     "betaxanthin_amino_acid_predictivity",
     "Amino-acid profile predicts betaxanthin, tyrosine does not"),
    ("strand_ceiling", "019-simb-multimodal", "retrospective_achieved_vs_ceiling",
     "Achieved vs ceiling, all strands"),
    ("m_spread", "020-cachera-betaxanthin",
     "merzbacher_fig1_scatter_spread_2026-08-02-23-43-08",
     "Per-gene spread; the two methods disagree"),
    ("m_accuracy", "020-cachera-betaxanthin",
     "merzbacher_fig2_accuracy_artifact_2026-08-02-23-43-08",
     "Why the published accuracy comparison is empty"),
    ("m_precision", "020-cachera-betaxanthin",
     "merzbacher_fig3_precision_at_k_2026-08-02-23-43-08",
     "Precision at k for high producers"),
    ("m_byclass", "020-cachera-betaxanthin",
     "merzbacher_fig4_score_by_class_2026-08-02-23-43-08",
     "Score by true class, percentile rank"),
    ("m_roc", "020-cachera-betaxanthin",
     "merzbacher_fig5_roc_high_producers_2026-08-02-23-43-08",
     "ROC for high-producer detection"),
    ("m_cells", "020-cachera-betaxanthin",
     "merzbacher_fig6_cell_spread_2026-08-02-23-43-08",
     "Grid-cell spread exceeds the method gap"),
    ("m_labels", "020-cachera-betaxanthin",
     "merzbacher_fig7_label_provenance_2026-08-02-23-43-08",
     "Label headroom: 19 percent bin disagreement"),
    ("m_coverage", "020-cachera-betaxanthin",
     "merzbacher_fig8_screen_coverage_2026-08-02-23-43-08",
     "yeast-GEM reaches 19 percent of the screen"),
    ("m_capability", "020-cachera-betaxanthin",
     "merzbacher_fig9_metabolic_vs_nonmetabolic_2026-08-02-23-43-08",
     "CGT finds high producers a flux model cannot represent"),
]


def read_panel(images_dir: str, slug: str, name: str) -> tuple[str, float, float]:
    """Return (data URI, width, height) in draw.io units, from the SVG's own root."""
    path = osp.join(images_dir, slug, f"{name}.svg")
    with open(path, "rb") as fh:
        raw = fh.read()
    head = raw[:4000].decode("utf-8", "replace")
    tag = re.search(r"<svg\b[^>]*>", head)
    if tag is None:
        raise ValueError(f"{path}: no <svg> root tag")
    width = re.search(r'width="([\d.]+)', tag.group(0))
    height = re.search(r'height="([\d.]+)', tag.group(0))
    if width is None or height is None:
        raise ValueError(f"{path}: <svg> root carries no width/height")
    uri = "data:image/svg+xml," + base64.b64encode(raw).decode("ascii")
    return uri, float(width.group(1)), float(height.group(1))


class Page:
    """One draw.io diagram page, accumulating cells."""

    def __init__(self, name: str) -> None:
        """Start an empty page under the name draw.io shows in its page tabs."""
        self.name = name
        self.cells: list[str] = []
        self.n = 0

    def _id(self) -> str:
        self.n += 1
        return f"c{self.n}"

    def image(self, uri: str, x: float, y: float, w: float, h: float) -> None:
        """Place an embedded SVG at an exact draw.io geometry (1 unit = 1/100 inch)."""
        self.cells.append(
            f'<mxCell id="{self._id()}" value="" '
            f'style="shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;'
            f'verticalAlign=top;image={uri};" vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'as="geometry"/></mxCell>'
        )

    def label(self, text: str, x: float, y: float, w: float, h: float, bold: bool = False) -> None:
        """A plain text cell, used for panel letters and bank descriptions."""
        weight = "fontStyle=1;" if bold else ""
        self.cells.append(
            f'<mxCell id="{self._id()}" value="{escape(text)}" '
            f'style="text;html=1;align=left;verticalAlign=top;{weight}fontSize=9;'
            f'fontFamily=Arial;whiteSpace=wrap;" vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'as="geometry"/></mxCell>'
        )

    def box(self, text: str, x: float, y: float, w: float, h: float, slot: int = 0,
            dashed: bool = False) -> None:
        """A schematic box in the repo palette; dashed marks a panel that does not exist yet."""
        style = (
            f"rounded=1;whiteSpace=wrap;html=1;fontSize=9;fontFamily=Arial;"
            f"fillColor={PLOT_PALETTE_FILL[slot % 18]};strokeColor={PLOT_PALETTE[slot % 18]};"
            f"strokeWidth=1.5;align=center;verticalAlign=middle;"
            + ("dashed=1;" if dashed else "")
        )
        self.cells.append(
            f'<mxCell id="{self._id()}" value="{escape(text)}" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'as="geometry"/></mxCell>'
        )

    def frame(self, x: float, y: float, w: float, h: float, text: str) -> None:
        """A dashed print-size frame, so a composition's true footprint is visible."""
        self.cells.append(
            f'<mxCell id="{self._id()}" value="{escape(text)}" '
            f'style="rounded=0;whiteSpace=wrap;html=1;dashed=1;dashPattern=8 8;'
            f'fillColor=none;strokeColor=#999999;verticalAlign=top;align=left;'
            f'fontSize=8;fontFamily=Arial;fontColor=#999999;spacingLeft=4;spacingTop=2;" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'as="geometry"/></mxCell>'
        )

    def to_xml(self) -> str:
        """Serialize the page as one <diagram> element of an mxfile."""
        body = "".join(self.cells)
        return (
            f'<diagram name="{escape(self.name)}" id="{escape(self.name).replace(" ", "-")}">'
            f'<mxGraphModel dx="1200" dy="900" grid="1" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{PAGE_W:.0f}" pageHeight="{PAGE_H:.0f}" math="0" shadow="0">'
            f"<root><mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>{body}</root>"
            f"</mxGraphModel></diagram>"
        )


def panel_bank(name: str, panels: list, images_dir: str) -> Page:
    """Every candidate panel at true size, in a single column with its description."""
    page = Page(name)
    page.label(
        "PANEL BANK. Every panel that exists today, at true print size. Copy into a "
        "composition page; do not resize.",
        MARGIN, MARGIN, PAGE_W - 2 * MARGIN, mm(8),
        bold=True,
    )
    y = MARGIN + mm(10)
    for key, slug, svg, description in panels:
        uri, w, h = read_panel(images_dir, slug, svg)
        page.label(f"{key}  --  {description}", MARGIN, y, PAGE_W - 2 * MARGIN, mm(4))
        y += mm(5)
        page.image(uri, MARGIN, y, w, h)
        y += h + GAP * 2
    return page


def fig3_options(images_dir: str) -> list[Page]:
    """Three compositions for Figure 3, differing in what leads."""
    load = lambda slug, svg: read_panel(images_dir, slug, svg)  # noqa: E731
    pages = []

    # --- Option A: lead with the honest ceiling-relative picture -----------------
    page = Page("Fig3 opt A ceiling-first")
    page.frame(MARGIN, MARGIN, mm(180), mm(170), "Nature full width 180 mm x 170 mm")
    x0, y0 = MARGIN, MARGIN + mm(5)
    page.box(
        "a  Shared cell embedding\n\nwildtype graph -> encoder -> perturbation operator\n"
        "-> per-phenotype heads\n\n(draw.io schematic, to be drawn)",
        x0, y0, mm(PANEL_WIDTHS_MM["half_plus"]), mm(52), slot=0, dashed=True,
    )
    uri, w, h = load("019-simb-multimodal", "retrospective_achieved_vs_ceiling")
    page.image(uri, x0 + mm(PANEL_WIDTHS_MM["half_plus"]) + GAP, y0, w, h)
    page.label("b", x0 + mm(PANEL_WIDTHS_MM["half_plus"]) + GAP, y0 - mm(4), mm(6), mm(4), bold=True)
    y1 = y0 + mm(58)
    uri, w, h = load("019-simb-multimodal", "retrospective_peak_position")
    page.image(uri, x0, y1, w, h)
    page.label("c", x0, y1 - mm(4), mm(6), mm(4), bold=True)
    uri, w, h = load("019-simb-multimodal", "residual_covariance_diagnostic")
    page.image(uri, x0 + mm(PANEL_WIDTHS_MM["half_plus"]) + GAP, y1, w * 0.5, h * 0.5)
    page.label(
        "d  (residual covariance, shown at half scale; recrop to a single panel before use)",
        x0 + mm(PANEL_WIDTHS_MM["half_plus"]) + GAP, y1 - mm(4), mm(88), mm(4), bold=True,
    )
    y2 = y1 + mm(58)
    page.box(
        "e  Morphology at full scale\n\n4,718 Ohya deletions, not the 1,440 expression overlap\n"
        "ceiling 0.611 over 278 features\n\n(NOT YET RUN)",
        x0, y2, mm(PANEL_WIDTHS_MM["half_plus"]), mm(42), slot=1, dashed=True,
    )
    page.box(
        "f  Does a second label help\n\nmorphology alone vs morphology + fitness\n"
        "4,220 shared strains\n\n(NOT YET RUN)",
        x0 + mm(PANEL_WIDTHS_MM["half_plus"]) + GAP, y2,
        mm(PANEL_WIDTHS_MM["half_plus"]), mm(42), slot=2, dashed=True,
    )
    pages.append(page)

    # --- Option B: lead with the data, then the model ---------------------------
    page = Page("Fig3 opt B data-first")
    page.frame(MARGIN, MARGIN, mm(180), mm(170), "Nature full width 180 mm x 170 mm")
    x0, y0 = MARGIN, MARGIN + mm(5)
    page.box(
        "a  What the compendium is\n\n1,484 single deletions x 6,169 reporters\n"
        "9.2 M scalars, but each gene deleted ONCE\n\n(schematic, to be drawn)",
        x0, y0, mm(PANEL_WIDTHS_MM["half_plus"]), mm(46), slot=0, dashed=True,
    )
    uri, w, h = load("012-sameith-kemmeren-expression", "gene_expression_correlation_dist_pearson")
    page.image(uri, x0 + mm(PANEL_WIDTHS_MM["half_plus"]) + GAP, y0, w * 0.6, h * 0.6)
    page.label("b  Kemmeren vs Sameith agreement (0.6 scale)",
               x0 + mm(PANEL_WIDTHS_MM["half_plus"]) + GAP, y0 - mm(4), mm(88), mm(4), bold=True)
    y1 = y0 + mm(52)
    uri, w, h = load("019-simb-multimodal", "retrospective_achieved_vs_ceiling")
    page.image(uri, x0, y1, w, h)
    page.label("c", x0, y1 - mm(4), mm(6), mm(4), bold=True)
    uri, w, h = load("019-simb-multimodal", "proteome_expression_per_gene_corr_hist_2026-07-22-17-15-26")
    page.image(uri, x0 + mm(PANEL_WIDTHS_MM["half_plus"]) + GAP, y1, w * 0.6, h * 0.6)
    page.label("d  mRNA vs protein, median r = 0.08 (0.6 scale)",
               x0 + mm(PANEL_WIDTHS_MM["half_plus"]) + GAP, y1 - mm(4), mm(88), mm(4), bold=True)
    y2 = y1 + mm(58)
    page.box(
        "e  Fitness panel\n\nthe strand that works; carries the figure\n\n(from 010, to be placed)",
        x0, y2, mm(PANEL_WIDTHS_MM["half_plus"]), mm(40), slot=3, dashed=True,
    )
    page.box(
        "f  Morphology scalar warm-up\n\nA113 actin_n_ratio, ceiling 0.873, robust CV 2.37\n"
        "or C124_C medium bud ratio\n\n(NOT YET RUN)",
        x0 + mm(PANEL_WIDTHS_MM["half_plus"]) + GAP, y2,
        mm(PANEL_WIDTHS_MM["half_plus"]), mm(40), slot=4, dashed=True,
    )
    pages.append(page)

    # --- Option C: the narrow version ------------------------------------------
    page = Page("Fig3 opt C narrow")
    page.frame(MARGIN, MARGIN, mm(180), mm(100), "Nature full width 180 mm x 100 mm")
    x0, y0 = MARGIN, MARGIN + mm(5)
    third = mm(PANEL_WIDTHS_MM["third"])
    page.box(
        "a  Shared embedding,\nthree phenotype heads\n\n(schematic)",
        x0, y0, third, mm(52), slot=0, dashed=True,
    )
    uri, w, h = load("019-simb-multimodal", "retrospective_achieved_vs_ceiling")
    page.image(uri, x0 + third + GAP, y0, w * 0.65, h * 0.65)
    page.label("b  achieved vs ceiling (0.65 scale)", x0 + third + GAP, y0 - mm(4),
               mm(60), mm(4), bold=True)
    uri, w, h = load("019-simb-multimodal", "retrospective_peak_position")
    page.image(uri, x0 + 2 * (third + GAP), y0, w * 0.65, h * 0.65)
    # Width capped so the label ends inside the 180 mm frame; at mm(60) it ran 5.6 mm past.
    page.label("c  peak position (0.65 scale)", x0 + 2 * (third + GAP), y0 - mm(4),
               mm(54), mm(4), bold=True)
    page.label(
        "Narrow option: three panels only. Drops every diagnostic, keeps the two claims "
        "that survive scrutiny (ceiling-relative standing, and that the budget rather "
        "than the model is binding on expression).",
        x0, y0 + mm(58), mm(170), mm(12),
    )
    pages.append(page)
    return pages


def fig6_options(images_dir: str) -> list[Page]:
    """Compositions for Figure 6, including the requested a-to-e ordering."""
    load = lambda slug, svg: read_panel(images_dir, slug, svg)  # noqa: E731
    pages = []

    # --- Option A: the requested order -----------------------------------------
    page = Page("Fig6 opt A requested order")
    page.frame(MARGIN, MARGIN, mm(180), mm(170), "Nature full width 180 mm x 170 mm")
    x0, y0 = MARGIN, MARGIN + mm(5)
    half = mm(PANEL_WIDTHS_MM["half_plus"])
    page.box(
        "a  Setup\n\nCRI-SPA transfers the four-gene betalain cassette into every YKO strain;\n"
        "CGT predicts production from genotype on a fixed cassette background.\n\n"
        "OPTIONS IF NEEDED: beta-carotene (Ozaydin) as a second product; isobutanol\n"
        "(Lopez) as a third; the iBioFoundry design loop as the outer cycle.",
        x0, y0, mm(174), mm(34), slot=0, dashed=True,
    )
    y1 = y0 + mm(38)
    uri, w, h = load("019-simb-multimodal", "pigment_noise_ceiling")
    page.image(uri, x0, y1, w * 0.48, h * 0.48)
    page.label("b  betaxanthin performance vs its 0.914 ceiling (0.48 scale)",
               x0, y1 - mm(4), mm(88), mm(4), bold=True)
    uri, w, h = load("023-metabolome-betaxanthin-joint", "betaxanthin_amino_acid_predictivity")
    page.image(uri, x0 + half + GAP, y1, w * 0.48, h * 0.48)
    page.label("c  amino-acid coupling (0.48 scale)", x0 + half + GAP, y1 - mm(4),
               mm(88), mm(4), bold=True)
    y2 = y1 + mm(42)
    page.box(
        "d  Combination\n\nbetaxanthin head + metabolome head, paired within grid cell.\n"
        "MEASURED TODAY: -0.0265 +/- 0.0159 over five comparable cells.\n\n"
        "OPTIONS IF THE REPLICATION TURNS POSITIVE: tyrosine-only arm as the negative\n"
        "control; weight ladder on the auxiliary head; depth as the conditioning variable.",
        x0, y2, half, mm(38), slot=1, dashed=True,
    )
    uri, w, h = load("020-cachera-betaxanthin", "merzbacher_fig3_precision_at_k_2026-08-02-23-43-08")
    page.image(uri, x0 + half + GAP, y2, w * 0.48, h * 0.48)
    page.label("e  FCL comparison, precision at k (0.48 scale)", x0 + half + GAP,
               y2 - mm(4), mm(88), mm(4), bold=True)
    y3 = y2 + mm(42)
    uri, w, h = load("020-cachera-betaxanthin", "merzbacher_fig9_metabolic_vs_nonmetabolic_2026-08-02-23-43-08")
    page.image(uri, x0, y3, w * 0.42, h * 0.42)
    page.label("f  the capability gap: genes FCL cannot represent (0.42 scale)",
               x0, y3 - mm(4), mm(174), mm(4), bold=True)
    pages.append(page)

    # --- Option B: coverage-led, which is the strongest argument ----------------
    page = Page("Fig6 opt B coverage-led")
    page.frame(MARGIN, MARGIN, mm(180), mm(170), "Nature full width 180 mm x 170 mm")
    x0, y0 = MARGIN, MARGIN + mm(5)
    page.box(
        "a  Setup: genome-wide production screen on a fixed cassette background\n\n"
        "OPTIONS: second product; the design loop; the metabolic-model boundary drawn "
        "explicitly so panel b lands.",
        x0, y0, mm(174), mm(24), slot=0, dashed=True,
    )
    y1 = y0 + mm(28)
    uri, w, h = load("020-cachera-betaxanthin", "merzbacher_fig8_screen_coverage_2026-08-02-23-43-08")
    page.image(uri, x0, y1, w * 0.48, h * 0.48)
    page.label("b  yeast-GEM reaches 19 percent of the screen (0.48 scale)",
               x0, y1 - mm(4), mm(88), mm(4), bold=True)
    uri, w, h = load("020-cachera-betaxanthin", "merzbacher_fig9_metabolic_vs_nonmetabolic_2026-08-02-23-43-08")
    page.image(uri, x0 + half + GAP, y1, w * 0.42, h * 0.42)
    page.label("c  CGT delivers inside that gap (0.42 scale)", x0 + half + GAP,
               y1 - mm(4), mm(88), mm(4), bold=True)
    y2 = y1 + mm(48)
    uri, w, h = load("020-cachera-betaxanthin", "merzbacher_fig3_precision_at_k_2026-08-02-23-43-08")
    page.image(uri, x0, y2, w * 0.48, h * 0.48)
    page.label("d  both methods are enriched at the top (0.48 scale)", x0, y2 - mm(4),
               mm(88), mm(4), bold=True)
    uri, w, h = load("020-cachera-betaxanthin", "merzbacher_fig2_accuracy_artifact_2026-08-02-23-43-08")
    page.image(uri, x0 + half + GAP, y2, w * 0.48, h * 0.48)
    page.label("e  and the published accuracy metric is empty (0.48 scale)",
               x0 + half + GAP, y2 - mm(4), mm(88), mm(4), bold=True)
    y3 = y2 + mm(46)
    uri, w, h = load("023-metabolome-betaxanthin-joint", "betaxanthin_amino_acid_predictivity")
    page.image(uri, x0, y3, w * 0.48, h * 0.48)
    page.label("f  the metabolome coupling, as data rather than as a training result "
               "(0.48 scale)", x0, y3 - mm(4), mm(174), mm(4), bold=True)
    pages.append(page)

    # --- Option C: the narrow version ------------------------------------------
    page = Page("Fig6 opt C narrow")
    page.frame(MARGIN, MARGIN, mm(180), mm(110), "Nature full width 180 mm x 110 mm")
    x0, y0 = MARGIN, MARGIN + mm(5)
    page.box(
        "a  Setup\n\nCRI-SPA cassette into every YKO strain; CGT predicts production "
        "from genotype.\n\nOPTIONS KEPT: second product, design loop.",
        x0, y0, mm(174), mm(22), slot=0, dashed=True,
    )
    y1 = y0 + mm(26)
    third = mm(PANEL_WIDTHS_MM["third"])
    uri, w, h = load("020-cachera-betaxanthin", "merzbacher_fig8_screen_coverage_2026-08-02-23-43-08")
    page.image(uri, x0, y1, w * 0.32, h * 0.32)
    page.label("b  coverage gap (0.32)", x0, y1 - mm(4), mm(58), mm(4), bold=True)
    uri, w, h = load("020-cachera-betaxanthin", "merzbacher_fig9_metabolic_vs_nonmetabolic_2026-08-02-23-43-08")
    page.image(uri, x0 + third + GAP, y1, w * 0.28, h * 0.28)
    page.label("c  capability gap (0.28)", x0 + third + GAP, y1 - mm(4), mm(58), mm(4), bold=True)
    uri, w, h = load("020-cachera-betaxanthin", "merzbacher_fig3_precision_at_k_2026-08-02-23-43-08")
    page.image(uri, x0 + 2 * (third + GAP), y1, w * 0.32, h * 0.32)
    page.label("d  precision at k (0.32)", x0 + 2 * (third + GAP), y1 - mm(4), mm(58), mm(4), bold=True)
    page.label(
        "Narrow option: the coverage argument only. Drops the accuracy panel, the ROC, "
        "the cell-spread qualifier and the amino-acid coupling. Every one of those is in "
        "the panel bank if a reviewer asks for it.",
        x0, y1 + mm(48), mm(170), mm(14),
    )
    pages.append(page)
    return pages


def write_board(path: str, pages: list) -> None:
    xml = (
        '<mxfile host="app.diagrams.net" type="device">'
        + "".join(p.to_xml() for p in pages)
        + "</mxfile>"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)


def main() -> None:
    load_dotenv()
    images_dir = os.environ["ASSET_IMAGES_DIR"]
    drawio_dir = osp.join(osp.dirname(images_dir), "drawio")
    os.makedirs(drawio_dir, exist_ok=True)

    fig3 = osp.join(drawio_dir, "Fig3-options.drawio")
    write_board(fig3, [panel_bank("Fig3 panel bank", FIG3_PANELS, images_dir), *fig3_options(images_dir)])
    fig6 = osp.join(drawio_dir, "Fig6-options.drawio")
    write_board(fig6, [panel_bank("Fig6 panel bank", FIG6_PANELS, images_dir), *fig6_options(images_dir)])

    for path in (fig3, fig6):
        print(f"{path}  ({osp.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
