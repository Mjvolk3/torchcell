# experiments/019-simb-multimodal/scripts/fig3_mockup_drawio.py
# [[experiments.019-simb-multimodal.scripts.fig3_mockup_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/fig3_mockup_drawio
"""Compose the three Figure 3 gate mockups as draw.io figures at true Nature print size.

Reads the true-size panel SVGs written by fig3_mockup_panels.py and the same committed
result files, and writes four figures into notes/assets/drawio/:

  fig3-mockup-A-conditioning.drawio.svg    measure one modality, predict another
  fig3-mockup-B-genotype-only.drawio.svg   what the deletion alone predicts
  fig3-mockup-SI-1.drawio.svg              how the splits are made, training curves
  fig3-mockup-SI-2.drawio.svg              representations, triangle, graph key, model
                                           size, the perturbation operator

WHY A SCRIPT. Every count and ridge value printed in a native schematic panel (the strain
counts and overlaps of panel a, the edge values of the triangle) is read from a results
JSON at build time, so a schematic cannot drift from the file it summarizes. Plot panels
go in as embedded true-size SVGs, byte-identical to the files under
$ASSET_IMAGES_DIR/019-simb-multimodal/, so
notes/assets/publish/scripts/drawio_true_size_import.py --identify reports them CURRENT.

STATUS MARKS. Each panel letter is followed by the section-status glyph of
notes-tex/common/tcdoc.sty (green check, amber square, red x; same colors), and each
figure carries a one-line key. They are mockup-only annotations: they live in their own
draw.io layer, "mockup status marks", and STATUS_MARKS = False (or FIG3_STATUS_MARKS=0 in
the environment) leaves them out entirely.

UNITS. 100 draw.io units per inch. A full-width figure is designed to 702 units or less
(the export adds a 1-unit border per side). Font sizes are the ladder of CLAUDE.md:
8.3 units (5.98 pt) for figure text, 11.1 bold for panel letters.

MATH. Formulas are LaTeX between $$ delimiters with draw.io math typesetting on
(mxGraphModel math="1"), the convention of manuscript Fig 1
(notes/assets/drawio/Fig1-torchcell-overview.drawio.svg). MathJax sets math at 1.17 times
the cell font size (measured on a headless export: the cap height of $$\mathrm{H}$$
against Arial H at cell sizes 6, 7, 8.3 and 9.7), so a formula cell at 8.3 prints its
main size at 7.0 pt, the top of Nature's band, and its subscripts near 5 pt.

The script writes each figure as plain .drawio XML into a temporary directory, then calls
draw.io headless for the .drawio.svg (diagram embedded) and for a PNG preview under
$ASSET_IMAGES_DIR/019-simb-multimodal/.

    python experiments/019-simb-multimodal/scripts/fig3_mockup_drawio.py
"""

from __future__ import annotations

import base64
import json
import os
import os.path as osp
import re
import subprocess
import tempfile
from typing import Any
from xml.sax.saxutils import escape

from dotenv import load_dotenv

from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, PLOT_PALETTE_FILL
from torchcell.utils.paths import experiment_results_dir

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
DRAWIO = os.environ.get("DRAWIO", "/Applications/draw.io.app/Contents/MacOS/draw.io")
STATUS_MARKS = os.environ.get("FIG3_STATUS_MARKS", "1") == "1"
RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-simb-multimodal")
DRAWIO_DIR = osp.join(osp.dirname(ASSET_IMAGES_DIR), "drawio")

UNITS_PER_MM = 100.0 / 25.4
FULL = 702.0
THIRD = PANEL_WIDTHS_MM["third"] * UNITS_PER_MM
WIDE = PANEL_WIDTHS_MM["wide"] * UNITS_PER_MM
HALF = PANEL_WIDTHS_MM["half"] * UNITS_PER_MM
# Column grid shared by every row of every mockup.
COL3 = [0.0, (FULL - THIRD) / 2, FULL - THIRD]  # three thirds
COL_AFTER_WIDE = FULL - THIRD  # a third beside a wide panel
COL_HALF2 = FULL - HALF  # second of two halves
ROW_GAP = 10.0
FONT = 8.3
LETTER = 11.1

STROKE = dict(zip(["orange", "red", "purple", "yellow", "blue", "gray"], PLOT_PALETTE))
FILL = dict(zip(["orange", "red", "purple", "yellow", "blue", "gray"], PLOT_PALETTE_FILL))
# The formula-box fill of manuscript Fig 1f (Fig1-torchcell-overview.drawio.svg), reused
# so the operator schematic reads as the same notation.
FIG1_FORMULA_FILL = "#F5EEDD"
# Section-status notation of notes-tex/common/tcdoc.sty: \ding{51} stgrn, \ding{110}
# styel, \ding{55} stred.
STATUS = {
    "ready": ("✓", "#43A047"),
    "partial": ("■", "#E0A020"),
    "todo": ("✗", "#E53935"),
}
MARKS_LAYER = "marks"

# The style draw.io itself writes for an embedded SVG image cell; kept identical to
# notes/assets/publish/scripts/drawio_true_size_import.py so --identify matches.
IMAGE_STYLE = (
    "shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;"
    "verticalAlign=top;aspect=fixed;imageAspect=0;image=data:image/svg+xml,"
)


def _load(name: str) -> dict[str, Any]:
    with open(osp.join(RESULTS, name)) as fh:
        data: dict[str, Any] = json.load(fh)
    return data


class Figure:
    """One draw.io page, accumulating cells at explicit geometry."""

    def __init__(self, name: str, math: bool = False) -> None:
        """Start an empty figure; ``name`` is the output basename, ``math`` turns on
        draw.io's MathJax typesetting of $$...$$ labels."""
        self.name = name
        self.math = math
        self.cells: list[str] = []
        self.height = 0.0

    def _add(
        self,
        value: str,
        style: str,
        x: float,
        y: float,
        w: float,
        h: float,
        parent: str = "1",
    ) -> None:
        self.cells.append(
            f'<mxCell id="c{len(self.cells) + 2}" value="{escape(value, {chr(34): "&quot;"})}" '
            f'style="{style}" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'as="geometry" /></mxCell>'
        )
        self.height = max(self.height, y + h)

    def image(self, svg_name: str, x: float, y: float) -> float:
        """Embed a true-size SVG at its own declared size; returns its height."""
        path = osp.join(IMG_DIR, f"{svg_name}.svg")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        tag = re.search(r"<svg[^>]*>", text)
        if tag is None:
            raise ValueError(f"{path}: no <svg> root")
        w = float(re.search(r'width="([\d.]+)"', tag.group(0)).group(1))  # type: ignore[union-attr]
        h = float(re.search(r'height="([\d.]+)"', tag.group(0)).group(1))  # type: ignore[union-attr]
        payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self._add("", IMAGE_STYLE + payload, x, y, w, h)
        return h

    def letter(self, letter: str, x: float, y: float, status: str) -> None:
        """A panel letter at the panel's outer top-left corner, then its status mark."""
        self._add(
            letter,
            f"text;html=1;whiteSpace=wrap;align=left;verticalAlign=top;spacing=0;"
            f"fontFamily=Arial;fontSize={LETTER};fontStyle=1;fontColor=#000000;"
            f"strokeColor=none;fillColor=none;",
            x,
            y,
            10,
            14,
        )
        if STATUS_MARKS:
            glyph, color = STATUS[status]
            self._add(
                glyph,
                f"text;html=1;whiteSpace=wrap;align=left;verticalAlign=top;spacing=0;"
                f"fontFamily=Arial;fontSize={FONT};fontColor={color};"
                f"strokeColor=none;fillColor=none;",
                x + 10,
                y + 2.5,
                10,
                11,
                parent=MARKS_LAYER,
            )

    def status_key(self) -> None:
        """The one-line key to the status marks, under the figure."""
        if not STATUS_MARKS:
            return
        parts = [
            ("ready", "ready: can be drawn now from committed results"),
            ("partial", "partial, or needs an author decision"),
            ("todo", "not done, placeholder"),
        ]
        html = "&nbsp;&nbsp;&nbsp; ".join(
            f'<font color="{STATUS[k][1]}">{STATUS[k][0]}</font> {label}'
            for k, label in parts
        )
        self._add(
            html,
            f"text;html=1;whiteSpace=wrap;align=left;verticalAlign=top;spacing=0;"
            f"fontFamily=Arial;fontSize={FONT};fontColor=#000000;strokeColor=none;"
            f"fillColor=none;",
            0.0,
            self.height + 6,
            FULL,
            12,
            parent=MARKS_LAYER,
        )

    def text(
        self,
        html: str,
        x: float,
        y: float,
        w: float,
        h: float,
        align: str = "left",
        fill: str = "none",
    ) -> None:
        """A borderless wrapping label. ``html`` may carry <b>, <br>, <sub>, <sup>."""
        valign = "top" if fill == "none" else "middle"
        self._add(
            html,
            f"text;html=1;whiteSpace=wrap;align={align};verticalAlign={valign};spacing=0;"
            f"fontFamily=Arial;fontSize={FONT};fontColor=#000000;strokeColor=none;"
            f"fillColor={fill};" + ("rounded=1;" if fill != "none" else ""),
            x,
            y,
            w,
            h,
        )

    def box(
        self,
        html: str,
        x: float,
        y: float,
        w: float,
        h: float,
        color: str,
        dashed: bool = False,
        white: bool = False,
    ) -> None:
        """A labeled box in one palette slot; ``dashed`` marks proposed or unbuilt."""
        fill = "#FFFFFF" if white else FILL[color]
        self._add(
            html,
            f"rounded=0;whiteSpace=wrap;html=1;fontFamily=Arial;fontSize={FONT};"
            f"fontColor=#000000;fillColor={fill};strokeColor={STROKE[color]};"
            f"strokeWidth=1;align=center;verticalAlign=middle;spacing=2;"
            + ("dashed=1;" if dashed else ""),
            x,
            y,
            w,
            h,
        )

    def edge(
        self,
        points: list[tuple[float, float]],
        color: str = "gray",
        arrow: bool = True,
        both: bool = False,
        dashed: bool = False,
    ) -> None:
        """A straight or elbowed line through explicit points."""
        style = (
            f"endArrow={'classic' if arrow else 'none'};html=1;rounded=0;"
            f"strokeWidth=1;strokeColor={STROKE[color]};endSize=4;startSize=4;"
            + ("startArrow=classic;" if both else "")
            + ("dashed=1;" if dashed else "")
        )
        (sx, sy), (tx, ty) = points[0], points[-1]
        mid = "".join(f'<mxPoint x="{px:.2f}" y="{py:.2f}" />' for px, py in points[1:-1])
        waypoints = f'<Array as="points">{mid}</Array>' if mid else ""
        self.cells.append(
            f'<mxCell id="c{len(self.cells) + 2}" style="{style}" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{sx:.2f}" y="{sy:.2f}" as="sourcePoint" />'
            f'<mxPoint x="{tx:.2f}" y="{ty:.2f}" as="targetPoint" />'
            f"{waypoints}</mxGeometry></mxCell>"
        )

    def xml(self) -> str:
        """The page as plain mxGraph XML, status marks on their own layer."""
        return (
            '<mxfile host="app.diagrams.net" agent="claude-code">'
            f'<diagram name="{self.name}" id="{self.name}">'
            '<mxGraphModel dx="1400" dy="1000" grid="0" page="1" '
            f'pageWidth="{FULL:.0f}" pageHeight="{self.height + 2:.0f}" '
            f'math="{int(self.math)}" shadow="0">'
            '<root><mxCell id="0" /><mxCell id="1" value="figure" parent="0" />'
            f'<mxCell id="{MARKS_LAYER}" value="mockup status marks" parent="0" />'
            + "".join(self.cells)
            + "</root></mxGraphModel></diagram></mxfile>"
        )


def data_schematic(fig: Figure, x0: float, y0: float, status: str) -> None:
    """Panel a: the three single-deletion panels, strain counts and pairwise overlaps.

    Counts: fig3_overlap_census.json (kemmeren_unique, ohya_unique),
    fig3_proteome_build_census.json (protein_abundance records, records with both
    labels), and n_shared_strains of the two morphology covariation files.
    """
    census = _load("fig3_overlap_census.json")["db_overlap_census"]
    prot_census = _load("fig3_proteome_build_census.json")
    pm = _load("proteome_morphology_covariation.json")
    em = _load("expression_morphology_covariation.json")
    n_expr = census["kemmeren_unique"]
    n_prot = prot_census["n_records_by_label"]["protein_abundance"]
    n_morph = census["ohya_unique"]
    n_ep = prot_census["n_records_with_proteome_and_expression"]
    n_pm = pm["n_shared_strains"]
    n_em = em["n_shared_strains"]

    fig.letter("a", x0, y0, status)
    fig.box("single-gene deletion (genotype)", x0 + 93, y0 + 14, 160, 20, "gray")
    boxes = [
        (x0 + 4, "orange", "Transcriptome", "Kemmeren 2014", n_expr),
        (x0 + 119, "red", "Proteome", "Messner 2023", n_prot),
        (x0 + 234, "purple", "Morphology", "Ohya 2005, CalMorph", n_morph),
    ]
    for bx, color, title, study, n in boxes:
        fig.edge([(x0 + 173, y0 + 34), (bx + 54, y0 + 62)])
        fig.box(
            f"<b>{title}</b><br>{study}<br>{n:,} deletions", bx, y0 + 62, 108, 50, color
        )
    # pairwise overlaps: adjacent pairs on one level, the outer pair below
    fig.box(f"{n_ep:,} shared", x0 + 66, y0 + 126, 100, 18, "gray", white=True)
    fig.edge([(x0 + 58, y0 + 112), (x0 + 58, y0 + 135), (x0 + 66, y0 + 135)], arrow=False)
    fig.edge([(x0 + 170, y0 + 112), (x0 + 170, y0 + 135), (x0 + 166, y0 + 135)], arrow=False)
    fig.box(f"{n_pm:,} shared", x0 + 180, y0 + 126, 100, 18, "gray", white=True)
    fig.edge([(x0 + 176, y0 + 112), (x0 + 176, y0 + 135), (x0 + 180, y0 + 135)], arrow=False)
    fig.edge([(x0 + 288, y0 + 112), (x0 + 288, y0 + 135), (x0 + 280, y0 + 135)], arrow=False)
    fig.box(f"{n_em:,} shared", x0 + 123, y0 + 156, 100, 18, "gray", white=True)
    fig.edge([(x0 + 24, y0 + 112), (x0 + 24, y0 + 165), (x0 + 123, y0 + 165)], arrow=False)
    fig.edge([(x0 + 322, y0 + 112), (x0 + 322, y0 + 165), (x0 + 223, y0 + 165)], arrow=False)


def triangle_schematic(fig: Figure, x0: float, y0: float, letter: str, status: str) -> None:
    """Three modality nodes; each side labeled with the ridge value in both directions.

    Values are the ones fig3_mockup_panels.py draws in the triangle bar panel:
    proteome_morphology_covariation.json, expression_morphology_covariation.json, and
    the proteome and expression pair from proteome_morphology_covariation.json context.
    """
    pm = _load("proteome_morphology_covariation.json")
    em = _load("expression_morphology_covariation.json")
    ctx = pm["context"]
    p_e = ctx["proteome_to_expression_ridge_per_feature"]
    e_p = ctx["expression_to_proteome_ridge_per_feature"]
    p_m = pm["proteome_to_morphology"]["all_features"]["median"]
    p_m_mov = pm["proteome_to_morphology"]["moving_features"]["median"]
    m_p = pm["morphology_to_proteome"]["all_proteins"]["median"]
    e_m = em["expression_to_morphology"]["all_features"]["median"]
    e_m_mov = em["expression_to_morphology"]["moving_features"]["median"]
    m_e = em["morphology_to_expression"]["all_genes"]["median"]
    arrow = "&#8594;"

    fig.letter(letter, x0, y0, status)
    fig.box("Expression (E)", x0 + 75, y0 + 16, 78, 20, "orange")
    fig.box("Proteome (P)", x0 + 2, y0 + 150, 78, 20, "red")
    fig.box("Morphology (M)", x0 + 148, y0 + 150, 78, 20, "purple")
    fig.edge([(x0 + 95, y0 + 38), (x0 + 46, y0 + 148)], both=True)
    fig.edge([(x0 + 133, y0 + 38), (x0 + 182, y0 + 148)], both=True)
    fig.edge([(x0 + 82, y0 + 160), (x0 + 146, y0 + 160)], both=True)
    fig.text(
        f"P {arrow} E&nbsp; {p_e:.2f}<br>E {arrow} P&nbsp; {e_p:.2f}",
        x0 + 2, y0 + 62, 58, 24,
    )
    fig.text(
        f"E {arrow} M&nbsp; {e_m:.2f}<br>moving&nbsp; {e_m_mov:.2f}<br>"
        f"M {arrow} E&nbsp; {m_e:.2f}",
        x0 + 172, y0 + 56, 56, 36,
    )
    fig.text(
        f"P {arrow} M&nbsp; {p_m:.2f}<br>moving&nbsp; {p_m_mov:.2f}<br>"
        f"M {arrow} P&nbsp; {m_p:.2f}",
        x0 + 86, y0 + 104, 56, 36, align="center",
    )
    fig.text(
        "ridge from the measured modality, held-out Pearson",
        x0, y0 + 176, THIRD, 12, align="center",
    )


def mock_placeholder(fig: Figure, x0: float, y0: float, letter: str, h: float) -> None:
    """A labeled placeholder for a panel with no data behind it."""
    fig.letter(letter, x0, y0, "todo")
    fig.box(
        "<b>MOCK</b><br>conditioned morphology<br>(expression or proteome revealed)"
        "<br><b>not yet run</b>",
        x0 + 14, y0 + 16, THIRD - 28, h - 32, "gray", dashed=True,
    )


def operator_schematic(fig: Figure, x0: float, y0: float, letter: str, status: str) -> None:
    """The perturbation operator in the notation of manuscript Fig 1f, twice.

    Left, the gate as Fig 1f writes it (formulas copied from that figure's cells). Right,
    what the 019 rounds compute, written as the manuscript Methods write it:
    EquivariantPerturbationTransform in torchcell/models/equivariant_cell_graph_transformer.py
    calls nn.MultiheadAttention with the perturbed genes as the only keys (a softmax over
    t), and null_sink defaults to False and is not set by the v19 or v20 configuration,
    so with one perturbed gene the weight is 1 for every querying gene. No data.
    """
    ratio = r"\frac{(h_iW_Q)\cdot(\mathbf{p}_tW_K)}{\sqrt{d_k}}"
    update = r"$$\Delta h_i=\sum_t \beta_{i, t}\left(\mathbf{p}_t W_V\right)$$"
    fused = r"$$h_i^{\text{p}}=h_i+\Delta h_i$$"
    pert_set = r"$$p=\left\{\left(e_t, \tau_t, m_t\right)\right\}_{t=1}^M$$"
    fig.letter(letter, x0, y0, status)
    versions = [
        (
            x0,
            24.0,
            "As drawn in Fig 1f",
            rf"$$\beta_{{i,t}}=\sigma\!\left({ratio}\right)$$",
            r"$$\beta_3\uparrow$$",
            r"$$\beta_5\approx 0$$",
            "With one perturbed gene, weights that differ between genes need the "
            "function to act on each pair, a sigmoid. Methods write a softmax over t.",
        ),
        (
            x0 + WIDE - 228,
            0.0,
            "As run in v19 and v20 (null sink off)",
            rf"$$\beta_{{i,t}}=\operatorname*{{softmax}}_{{t\in[M]}}{ratio}$$",
            r"$$\beta_3=1$$",
            r"$$\beta_5=1$$",
            "Softmax over the perturbed genes: with one perturbed gene the weight is 1 "
            "for every gene and the update is the same vector for every gene.",
        ),
    ]
    for ox, title_dx, title, gate, beta_a, beta_b, note in versions:
        fig.text(f"<b>{title}</b>", ox + title_dx, y0 + 2, 228 - title_dx, 12)
        fig.text(gate, ox, y0 + 18, 138, 36, align="center", fill=FIG1_FORMULA_FILL)
        fig.text(update, ox, y0 + 58, 138, 30, align="center", fill=FIG1_FORMULA_FILL)
        fig.text(fused, ox, y0 + 92, 138, 16, align="center", fill=FIG1_FORMULA_FILL)
        fig.text(pert_set, ox, y0 + 112, 138, 22, align="center", fill=FIG1_FORMULA_FILL)
        fig.box(r"$$\mathbf{p}_1$$", ox + 144, y0 + 62, 24, 18, "purple")
        fig.box(r"$$h_3$$", ox + 204, y0 + 34, 24, 18, "orange")
        fig.box(r"$$h_5$$", ox + 204, y0 + 90, 24, 18, "orange")
        fig.edge([(ox + 170, y0 + 67), (ox + 202, y0 + 46)])
        fig.edge([(ox + 170, y0 + 75), (ox + 202, y0 + 96)])
        fig.text(beta_a, ox + 142, y0 + 26, 54, 14, align="center")
        fig.text(beta_b, ox + 142, y0 + 104, 54, 14, align="center")
        fig.text(note, ox, y0 + 140, 228, 36)


def split_schematic(fig: Figure, x0: float, y0: float, letter: str, status: str) -> None:
    """How a split seed becomes the train, validation and test strains of a v19 run.

    Drawn from the code: CellDataModule._compute_and_save_index
    (torchcell/datamodules/cell.py) seeds Python's random with the split seed, shuffles
    the record indices under every key of phenotype_label_index and
    perturbation_count_index, cuts each 80 / 10 / 10 and intersects; train_cgt_multitask.py
    then keeps, inside each split, the records carrying every label in require_modalities.
    Counts: split_gene_overlap_audit.json (fig3_proteome_full, per split seed and side: n
    records and the both-label count) and gene_share_components for the store size.
    """
    audit = _load("split_gene_overlap_audit.json")
    full = audit["fig3_proteome_full"]
    n_store = audit["gene_share_components"]["fig3_proteome"]["genotypes"]
    seeds = sorted(full, key=int)
    first = full[seeds[0]]
    n_both = sum(first[side]["both"] for side in ("train", "val", "test"))
    fig.letter(letter, x0, y0, status)
    fig.box(
        f"<b>Store fig3_proteome</b><br>{n_store:,} records<br>one record = one deleted "
        "gene set (strain)",
        x0 + 24, y0 + 34, 130, 50, "gray",
    )
    fig.edge([(x0 + 156, y0 + 59), (x0 + 178, y0 + 59)])
    fig.box(
        "<b>split seed s</b><br>random.seed(s); within each phenotype label and each "
        "perturbation count, shuffle the records and cut 80 / 10 / 10",
        x0 + 180, y0 + 26, 150, 66, "yellow",
    )
    sides = [("train", "orange"), ("val", "red"), ("test", "purple")]
    for i, (side, color) in enumerate(sides):
        by = y0 + 14 + i * 32
        fig.edge([(x0 + 332, y0 + 59), (x0 + 354, by + 13)])
        fig.box(f"{side}: {first[side]['n']:,} records", x0 + 356, by, 112, 26, color)
        fig.edge([(x0 + 470, by + 13), (x0 + 520, by + 13)])
        fig.box(
            f"{side}: {first[side]['both']:,} strains", x0 + 522, by, 112, 26, color
        )
    fig.text(f"split seed {seeds[0]}, all records", x0 + 356, y0, 112, 12, align="center")
    fig.text(
        f"both labels kept ({n_both:,})", x0 + 522, y0, 112, 12, align="center"
    )
    rows = "<br>".join(
        f"split seed {s}: {full[s]['train']['both']:,} / {full[s]['val']['both']:,} / "
        f"{full[s]['test']['both']:,}"
        for s in seeds
    )
    fig.text(
        f"<b>Both-label train / val / test</b><br>{rows}", x0 + 24, y0 + 112, 160, 60
    )
    fig.text(
        "<b>Each split seed is an independent draw.</b><br>Round v19 uses split seeds 0 "
        "to 11; counts are on file for 0 to 3.",
        x0 + 200, y0 + 112, 150, 48,
    )
    fig.text(
        "<b>Every arm of a split seed shares its strains.</b><br>Single-head and joint "
        "arms see the same rows, steps and batches (v19 configuration).",
        x0 + 366, y0 + 112, 160, 48,
    )
    fig.text(
        "<b>One initialization seed per split seed.</b><br>A split seed fixes the "
        "partition; it is not a replicate of one partition.",
        x0 + 542, y0 + 112, 158, 48,
    )


def _panel(fig: Figure, name: str, x: float, y: float, letter: str, status: str) -> float:
    h = fig.image(f"fig3_mockup_{name}", x, y)
    fig.letter(letter, x, y, status)
    return h


def mockup_a() -> Figure:
    fig = Figure("fig3-mockup-A-conditioning")
    data_schematic(fig, 0.0, 0.0, "ready")
    h1 = _panel(fig, "ceiling_vs_model", COL_HALF2, 0.0, "b", "partial")
    y2 = h1 + ROW_GAP
    triangle_schematic(fig, COL3[0], y2, "c", "partial")
    h2 = _panel(fig, "v20_partial", COL3[1], y2, "d", "partial")
    mock_placeholder(fig, COL3[2], y2, "e", h2)
    y3 = y2 + h2 + ROW_GAP
    _panel(fig, "baselines_by_seed_third", COL3[0], y3, "f", "partial")
    _panel(fig, "graph_effect_third", COL3[1], y3, "g", "partial")
    _panel(fig, "covariation", COL3[2], y3, "h", "ready")
    fig.status_key()
    return fig


def mockup_b() -> Figure:
    fig = Figure("fig3-mockup-B-genotype-only")
    data_schematic(fig, 0.0, 0.0, "ready")
    h1 = _panel(fig, "ceiling_vs_model", COL_HALF2, 0.0, "b", "partial")
    y2 = h1 + ROW_GAP
    h2 = _panel(fig, "v19_paired", 0.0, y2, "c", "ready")
    _panel(fig, "baselines_by_seed_half", COL_HALF2, y2, "d", "partial")
    y3 = y2 + h2 + ROW_GAP
    _panel(fig, "graph_effect_wide", 0.0, y3, "e", "partial")
    _panel(fig, "embedding_reduced", COL_AFTER_WIDE, y3, "f", "ready")
    fig.status_key()
    return fig


def mockup_si_1() -> Figure:
    fig = Figure("fig3-mockup-SI-1")
    split_schematic(fig, 0.0, 0.0, "a", "partial")
    y2 = fig.height + ROW_GAP
    h2 = _panel(fig, "curves_proteome", 0.0, y2, "b", "ready")
    _panel(fig, "curves_expression", COL_HALF2, y2, "c", "ready")
    y3 = y2 + h2 + ROW_GAP
    _panel(fig, "v19_timing", 0.0, y3, "d", "ready")
    _panel(fig, "v19_per_seed", COL_HALF2, y3, "e", "ready")
    fig.status_key()
    return fig


def mockup_si_2() -> Figure:
    fig = Figure("fig3-mockup-SI-2", math=True)
    h_full = _panel(fig, "embedding_full", 0.0, 0.0, "a", "ready")
    h_tri = _panel(fig, "triangle", COL_HALF2, 0.0, "b", "partial")
    _panel(fig, "graph_retrieval", COL_HALF2, h_tri + ROW_GAP, "c", "ready")
    y2 = h_full + ROW_GAP
    _panel(fig, "size_vs_score", 0.0, y2, "d", "partial")
    operator_schematic(fig, FULL - WIDE, y2, "e", "partial")
    fig.status_key()
    return fig


def export(fig: Figure, tmp: str) -> None:
    src = osp.join(tmp, f"{fig.name}.drawio")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(fig.xml())
    out_svg = osp.join(DRAWIO_DIR, f"{fig.name}.drawio.svg")
    out_png = osp.join(IMG_DIR, f"{fig.name}.png")
    subprocess.run(
        [DRAWIO, "--no-sandbox", "-x", "-f", "svg", "--embed-diagram", "-o", out_svg, src],
        check=True,
    )
    subprocess.run(
        [DRAWIO, "--no-sandbox", "-x", "-f", "png", "-s", "4", "-o", out_png, src],
        check=True,
    )
    with open(out_svg, encoding="utf-8") as fh:
        head = fh.read(2000)
    size = re.search(r'width="([\d.]+)px" height="([\d.]+)px"', head)
    if size is None:
        raise ValueError(f"{out_svg}: no export size in the svg root")
    w, h = (float(v) / UNITS_PER_MM for v in size.groups())
    print(f"{out_svg}\n  {w:.2f} x {h:.2f} mm, preview {out_png}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for fig in (mockup_a(), mockup_b(), mockup_si_1(), mockup_si_2()):
            export(fig, tmp)


if __name__ == "__main__":
    main()
