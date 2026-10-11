# experiments/019-simb-multimodal/scripts/fig3_si_morphology_sketch_drawio.py
# [[experiments.019-simb-multimodal.scripts.fig3_si_morphology_sketch_drawio]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/fig3_si_morphology_sketch_drawio
"""The sketch of the morphology SI figure: boxes, no data, written as draw.io.

A sketch is what the experiments are run against, so it is drawn before the panels exist
and carries the gate's two symbol sets after every panel letter (section status, the
author's approval mark, red cross throughout; figure status, filled circle on file, half
circle partial, open circle idea only). Color is the INPUT FAMILY an arm reads, as in the
metabolism and environment gate figures: taupe for the gene's own sequence (genotype only),
the fourth palette slot for the strain's measured phenotype as input, gray for a permuted
control; a panel that shows data and not an arm is white. The shaded band marks the
transductive panels, the ones that need a measurement of the strain itself. The key sits
above the figure. The boxes name the quantity, the statistic and the sample each panel
will show; the numbers the captions will carry come from the result files named under
each box, never from the sketch.

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
BAND_PAD = 4.0


def panel(
    fig: Figure,
    x: float,
    y: float,
    letter: str,
    status: str,
    title: str,
    body: str,
    color: str,
    source: str,
    dashed: bool = False,
    white: bool = False,
) -> None:
    """One sketch panel: letter and the two status marks, a titled box in its input
    family's color (white for a data panel), and the source line under it."""
    fig.letter(letter, x, y, status)
    fig.box(
        f"<b>{title}</b><br>{body}",
        x + PAD,
        y + 16,
        THIRD - PAD - 2,
        H - 40,
        color,
        dashed=dashed,
        white=white,
    )
    fig.text(f"<i>{source}</i>", x + PAD, y + H - 22, THIRD - PAD - 2, 20)


def sketch() -> Figure:
    fig = Figure("fig3-si-morphology-sketch")
    fig.text(
        "<b>Morphology from node-level measurements.</b> Can a cell-level phenotype (CalMorph) be predicted "
        "from the genotype, and does the strain's own measured gene-level modality (expression, proteome) help "
        "when given as input? Panels a and b describe the data; c is the inductive arm; d, e and f need the "
        "strain's measurement (transductive); g is more training data for the measured modalities, not a panel "
        "of its own.",
        0.0,
        0.0,
        FULL,
        38,
    )
    y1 = 42.0
    panel(
        fig, COL3[0], y1, "a", "ready", "The strain sets",
        "three single-deletion panels and their overlaps: morphology 4,695, expression 1,554, proteome 4,476; "
        "morph. and expr. 1,439, morph. and prot. 4,281, all three 1,321. Counts, no statistic.",
        "gray", "results/morph_v24_dataset.json", white=True,
    )
    panel(
        fig, COL3[1], y1, "b", "ready", "The target: 116 moving features",
        "per-feature replicate reliability (122 wild-type profiles against 4,718 mutants) as a histogram; "
        "the 0.5 threshold; ceiling = sqrt(reliability), mean 0.847 on the moving set, 0.611 on all 278.",
        "gray", "results/morphology_feature_ceiling.csv, calmorph_moving_features.json", white=True,
    )
    panel(
        fig, COL3[2], y1, "c", "ready", "Genotype only (inductive)",
        "bars: ceiling; sequence ridge and neighbor average; the CGT off the perturbed CLS (twelve split seeds, "
        "window mean, sd). Pearson per feature across held-out strains, moving features.",
        "sequence", "results/morph_v24_readout.json (M_cls, M_pool)",
    )
    # The transductive band sits behind row 2: every panel in it needs the strain's own
    # measured expression or proteome.
    y2 = y1 + H + ROW_GAP
    fig.band(
        COL3[0] - BAND_PAD,
        y2 - BAND_PAD,
        FULL + 2 * BAND_PAD,
        H + 2 * BAND_PAD,
        "transductive: the strain's own measurement as input",
    )
    panel(
        fig, COL3[0], y2, "d", "partial", "Measured modality as input",
        "paired by split seed: the strain's own proteome (expression) as input against another strain's "
        "(permuted, control) and against genotype only on the same strains; twelve seeds; beside the ridge "
        "from the same measurement.",
        "measured phenotype", "morph_v24_readout.json (CM_prot, CM_protperm, MP_cls; CMP_expr, CMP_exprperm); in flight",
    )
    panel(
        fig, COL3[1], y2, "e", "todo", "Per feature",
        "held-out Pearson of the best measured-input arm against each feature's reliability; which features a "
        "measured modality reaches and which stay at the floor. Dots, 116 features, one seed by rule.",
        "measured phenotype", "a per-feature dump of the CM_prot checkpoint; not written yet", dashed=True,
    )
    panel(
        fig, COL3[2], y2, "f", "ready", "The conditioning triangle",
        "the same contrast on the other two sides: proteome given the strain's expression +0.054 (11 of 11), "
        "expression given its proteome +0.033 (10 of 11), both below their ridge; morphology is the third vertex.",
        "measured phenotype", "results/joint_checkpoint_readout.json (v20), the covariation files",
    )
    y3 = y2 + H + ROW_GAP + BAND_PAD
    fig.letter("g", COL3[0], y3, "todo")
    fig.box(
        "<b>More training strains for the measured modalities: the 1,011 natural isolates (Peter 2018)</b><br>"
        "Not a morphology panel: no CalMorph exists for these strains. The idea is to add them as extra training "
        "data for the expression head (Caudal 2024 pan-transcriptome, built, 943 isolates) and the proteome head "
        "(Muenzner 2024, 796 isolates, next build), about 1,000 strains whose expression differs far more between "
        "strains than the deletion panels do, and then ask whether the deletion-strain heads, and through them the "
        "transductive morphology arms in d, improve. Open question first: how an isolate's genotype enters a model "
        "that takes a set of deleted genes. <b>Idea only; likely not in time for this round.</b>",
        COL3[0] + PAD,
        y3 + 16,
        FULL - PAD,
        62,
        "gray",
        dashed=True,
        white=True,
    )
    fig.text(
        "<i>the expansion document, next-build band; no result file</i>",
        COL3[0] + PAD,
        y3 + 80,
        FULL - PAD,
        14,
    )
    fig.status_key(["sequence", "measured phenotype", "control"])
    return fig


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        export(sketch(), tmp)


if __name__ == "__main__":
    main()
