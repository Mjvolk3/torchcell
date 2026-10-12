# experiments/019-simb-multimodal/scripts/fig3_si_morphology_mockup_drawio.py
# [[experiments.019-simb-multimodal.scripts.fig3_si_morphology_mockup_drawio]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/fig3_si_morphology_mockup_drawio
"""The morphology SI figure as a mockup: the sketch's panels filled with the v24 data.

Composes the true-size SVG panels of fig3_si_morphology_panels.py (and the v20
conditioning triangle of fig3_mockup_panels.py) at Nature full width with the gate's two
status symbols after each letter, the family key above the figure, and the shaded band
under the transductive row. The isolates box of the sketch stays as the dashed idea it is.

Writes notes/assets/drawio/fig3-si-morphology-mockup.drawio.svg and the PNG preview under
$ASSET_IMAGES_DIR/019-simb-multimodal/.

    python experiments/019-simb-multimodal/scripts/fig3_si_morphology_mockup_drawio.py
"""

from __future__ import annotations

import os.path as osp
import sys
import tempfile

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from fig3_mockup_drawio import (  # noqa: E402
    COL3,
    COL_HALF2,
    FULL,
    ROW_GAP,
    THIRD,
    Figure,
    _load,
    export,
)

BAND_PAD = 4.0


def _svg_height(name: str) -> float:
    """The declared height of a true-size panel SVG, in draw.io units."""
    import re

    from fig3_mockup_drawio import IMG_DIR

    with open(osp.join(IMG_DIR, f"{name}.svg"), encoding="utf-8") as fh:
        tag = re.search(r"<svg[^>]*>", fh.read(4000))
    assert tag is not None
    return float(re.search(r'height="([\d.]+)"', tag.group(0)).group(1))  # type: ignore[union-attr]


def _panel(fig: Figure, name: str, x: float, y: float, letter: str, status: str) -> float:
    h = fig.image(name, x, y)
    fig.letter(letter, x, y, status)
    return h


def strain_sets(fig: Figure, x: float, y: float, letter: str, status: str) -> float:
    """Panel a as a text box: the strain sets of the two stores, from morph_v24_dataset.json."""
    s = _load("morph_v24_dataset.json")["stores"]
    core, prot = s["fig3_core"], s["fig3_morph_proteome"]
    fig.letter(letter, x, y, status)
    text = (
        "<b>The strain sets</b><br>single-gene deletions with a label: morphology (Ohya 2005, "
        f"CalMorph) {core['by_label']['morphology']:,}; expression (Kemmeren 2014, Sameith 2015) "
        f"{core['by_label']['expression']:,}; proteome (Messner 2023) {prot['by_label']['proteome']:,}."
        f"<br>with two labels: morphology and expression {core['morphology_and_expression']:,}; "
        f"morphology and proteome {prot['morphology_and_proteome']:,}; all three {prot['all_three']:,}."
        f"<br>genotype-only arms train on {core['arms']['M_cls, M_pool']['train']:,} strains; the "
        f"expression-input arms on {core['arms']['CM_expr, CM_exprperm, CMP_expr, CMP_exprperm']['train']:,}; "
        f"the proteome-input arms on {prot['arms']['MP_cls, CM_prot, CM_protperm']['train']:,}."
    )
    fig.box(text, x + 14, y + 16, THIRD - 16, 150, "gray", white=True)
    return 170.0


def mockup() -> Figure:
    fig = Figure("fig3-si-morphology-mockup")
    y1 = 0.0
    h_a = strain_sets(fig, COL3[0], y1, "a", "ready")
    h_b = _panel(fig, "fig3_si_morph_target_reliability", COL3[1], y1, "b", "ready")
    h_c = _panel(fig, "fig3_si_morph_genotype_only", COL3[2], y1, "c", "ready")
    y2 = y1 + max(h_a, h_b, h_c) + ROW_GAP
    # The transductive row: every panel reads the strain's own measurement. The band is
    # drawn first, so the panels sit on it (draw.io stacks cells in insertion order).
    band_h = max(_svg_height("fig3_si_morph_measured_input"), _svg_height("fig3_si_morph_paired_difference")) + 2 * BAND_PAD
    fig.band(-BAND_PAD, y2, FULL + 2 * BAND_PAD, band_h, "transductive: the strain's own measurement as input")
    _panel(fig, "fig3_si_morph_measured_input", 0.0, y2 + BAND_PAD, "d", "partial")
    _panel(fig, "fig3_si_morph_paired_difference", COL_HALF2, y2 + BAND_PAD, "e", "partial")
    y3 = y2 + band_h + ROW_GAP
    h_f = _panel(fig, "fig3_si_morph_fit_diagnostic", COL3[0], y3, "f", "ready")
    h_g = _panel(fig, "fig3_si_morph_ladder", COL3[1], y3, "g", "ready")
    y4 = y3 + max(h_f, h_g) + ROW_GAP
    # The v20 conditioning triangle is mockup A's panel c and is not repeated here.
    fig.letter("h", COL3[0], y4, "todo")
    fig.box(
        "<b>More training strains for the measured modalities: the 1,011 natural isolates (Peter 2018).</b> "
        "Not a morphology panel (no CalMorph on these strains); Caudal 2024 expression (served, 943 isolates) and "
        "Muenzner 2024 proteome (next build, 796) as extra training data for the measured-input arms of d. "
        "<b>Idea only; likely not in time for this round.</b>",
        COL3[0] + 14,
        y4 + 14,
        FULL - 14,
        40,
        "gray",
        dashed=True,
        white=True,
    )
    fig.status_key(["sequence", "measured phenotype", "control"])
    return fig


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        export(mockup(), tmp)


if __name__ == "__main__":
    main()
