# experiments/019-simb-multimodal/scripts/fig3_si_morphology_sketch_drawio.py
# [[experiments.019-simb-multimodal.scripts.fig3_si_morphology_sketch_drawio]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/fig3_si_morphology_sketch_drawio
"""The sketch of the morphology SI figure: boxes, no data, written as draw.io.

A sketch is what the experiments are run against, so it is drawn before the panels exist
and carries the Figure 3 gate's status marks: a panel whose result files are committed is
ready, one with runs in flight is partial, one with nothing behind it is not done. The
boxes name the quantity, the statistic and the sample each panel will show; the numbers
the captions will carry come from the result files named under each box, never from the
sketch.

Reuses the draw.io writer of fig3_mockup_drawio.py (page width 180 mm, Arial 8.3 for
figure text and 11.1 for panel letters in draw.io units, the palette strokes and fills,
the status-mark layer). Writes

  notes/assets/drawio/fig3-si-morphology-sketch.drawio.svg
  $ASSET_IMAGES_DIR/019-simb-multimodal/fig3-si-morphology-sketch.png

    python experiments/019-simb-multimodal/scripts/fig3_si_morphology_sketch_drawio.py
"""

from __future__ import annotations

import sys
import os.path as osp
import tempfile

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from fig3_mockup_drawio import (  # noqa: E402
    COL3,
    FULL,
    ROW_GAP,
    THIRD,
    Figure,
    export,
)

H = 118.0  # panel height, draw.io units (about 30 mm)
PAD = 14.0


def panel(fig: Figure, x: float, y: float, letter: str, status: str, title: str, body: str, color: str, source: str, dashed: bool = False) -> None:
    """One sketch panel: letter and status, a titled box, and the source line under it."""
    fig.letter(letter, x, y, status)
    fig.box(f"<b>{title}</b><br>{body}", x + PAD, y + 16, THIRD - PAD - 2, H - 40, color, dashed=dashed)
    fig.text(f"<i>{source}</i>", x + PAD, y + H - 22, THIRD - PAD - 2, 20)


def sketch() -> Figure:
    fig = Figure("fig3-si-morphology-sketch")
    fig.text(
        "<b>Morphology from node-level measurements.</b> Can a cell-level phenotype (CalMorph) be predicted "
        "from the genotype, and does a measured gene-level modality (expression, proteome) help when revealed "
        "as input? Transductive: every validation strain carries the revealed modality.",
        0.0, 0.0, FULL, 26,
    )
    y1 = 30.0
    panel(fig, COL3[0], y1, "a", "ready", "The strain sets",
          "three panels and their overlaps: morphology 4,695, expression 1,554, proteome 4,476; "
          "morph. and expr. 1,439, morph. and prot. 4,281, all three 1,321. Counts, no statistic.",
          "blue", "results/morph_v24_dataset.json")
    panel(fig, COL3[1], y1, "b", "ready", "The target: 116 moving features",
          "per-feature replicate reliability (122 wild-type profiles against 4,718 mutants) as a histogram; "
          "the 0.5 threshold; ceiling = sqrt(reliability), mean 0.847 on the moving set, 0.611 on all 278.",
          "orange", "results/morphology_feature_ceiling.csv, calmorph_moving_features.json")
    panel(fig, COL3[2], y1, "c", "ready", "Genotype only",
          "bars: ceiling; sequence ridge and neighbor average; the CGT off the perturbed CLS (twelve split seeds, "
          "window mean, sd). Pearson per feature across held-out strains, moving features.",
          "purple", "results/morph_v24_readout.json (M_cls, M_pool)")
    y2 = y1 + H + ROW_GAP
    panel(fig, COL3[0], y2, "d", "partial", "Measured modality revealed",
          "paired by split seed: the strain's own proteome (expression) revealed against another strain's, "
          "and against genotype only on the same strains; twelve seeds; beside the ridge from the same measurement.",
          "red", "morph_v24_readout.json (CM_prot, CM_protperm, MP_cls; CMP_expr, CMP_exprperm); in flight")
    panel(fig, COL3[1], y2, "e", "todo", "Per feature",
          "held-out Pearson of the best revealed arm against each feature's reliability; which features a "
          "measured modality reaches and which stay at the floor. Dots, 116 features, one seed by rule.",
          "yellow", "a per-feature dump of the CM_prot checkpoint; not written yet", dashed=True)
    panel(fig, COL3[2], y2, "f", "ready", "The conditioning triangle",
          "the same contrast on the other two sides: proteome given expression +0.054 (11 of 11), expression "
          "given proteome +0.033 (10 of 11), both below their ridge; morphology joins as the third vertex.",
          "gray", "results/joint_checkpoint_readout.json (v20), the covariation files")
    y3 = y2 + H + ROW_GAP
    fig.letter("g", COL3[0], y3, "todo")
    fig.box(
        "<b>Natural variation: the 1,011 isolates</b><br>the same conditioning on wild strains: Caudal 2024 expression "
        "(built, 943 isolates) and Muenzner 2024 proteome (next build, 796 isolates); morphology not measured on the "
        "panel, so the target there is the proteome from expression and the reverse. <b>Planned; likely not in time "
        "for this round.</b>",
        COL3[0] + PAD, y3 + 16, FULL - PAD, 46, "gray", dashed=True,
    )
    fig.text("<i>the expansion document, next-build band; no result file</i>", COL3[0] + PAD, y3 + 64, FULL - PAD, 14)
    fig.status_key()
    return fig


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        export(sketch(), tmp)


if __name__ == "__main__":
    main()
