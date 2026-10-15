"""Generate the datasets, pools and splits figure of the Figure 2 gate.

    PYTHONPATH=$PWD python notes/assets/drawio/030-datasets-splits.gen.py notes/assets/drawio

writes 030-datasets-splits.drawio: the three builds of the Kuzmin screens (010 triples
only, the 025 closure pool with merged labels, the 030 per-entry pool), one triple with
the seven records the published identity reads and which of them train on each arm, and
the two arms with their counts. 702 units wide (178 mm at draw.io's 100 units per inch);
fonts on the ladder only (8.3 body, 9.7 box titles, 11.1 bold row letters). One color
per record order: yellow singles, orange doubles, purple the triple; a record withheld
from training is drawn white with a dashed border of the same color. Every count is read
from a committed result file of experiments/030-solid-growth-multi/results/:
subset_definitions_030_summary.json, subset_definitions_030_q_summary.json and
transfer_010_tmi_splits_030_summary.json. Nothing is typed.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from xml.sax.saxutils import escape

from dotenv import load_dotenv

load_dotenv()
OUT_DIR = sys.argv[1]
RESULTS = osp.join(os.environ["EXPERIMENT_ROOT"], "030-solid-growth-multi", "results")

W = 702
FS = "8.3"
FS_T = "9.7"
FS_P = "11.1"

WHITE = ("#FFFFFF", "#666666")
ORANGE = ("#FFE6CC", "#D79B00")
PURPLE = ("#E1D5E7", "#9673A6")
YELLOW = ("#FFF2CC", "#D6B656")
GRAY = ("#F5F5F5", "#666666")


def n(v: int) -> str:
    return f"{v:,}"


def read(name: str) -> dict:
    with open(osp.join(RESULTS, name)) as fh:
        return json.load(fh)


class Canvas:
    def __init__(self) -> None:
        self.cells: list[str] = []
        self._id = 10
        self.height = 0.0

    def nid(self) -> str:
        self._id += 1
        return str(self._id)

    def box(self, title: str, body: str, x: float, y: float, w: float, h: float,
            color: tuple[str, str], dashed: bool = False, stroke_width: float = 0.8,
            align: str = "left") -> str:
        fill, stroke = color
        i = self.nid()
        label = f'<span style="font-size: {FS_T}px"><b>{title}</b></span>' if title else ""
        if body:
            label += ("<br>" if title else "") + body
        style = (
            f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;fontFamily=Arial;fontSize={FS};"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            f"align={align};verticalAlign=top;spacing=3;spacingLeft=4;spacingRight=4;"
        )
        if dashed:
            style += "dashed=1;dashPattern=4 3;"
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
            f'height="{h:.2f}" as="geometry"/></mxCell>'
        )
        self.height = max(self.height, y + h)
        return i

    def chip(self, label: str, x: float, y: float, w: float, h: float,
             color: tuple[str, str], held: bool) -> str:
        fill, stroke = color
        i = self.nid()
        style = (
            f"rounded=1;arcSize=20;whiteSpace=wrap;html=1;fontFamily=Arial;fontSize={FS};"
            f"strokeColor={stroke};strokeWidth=0.8;align=center;verticalAlign=middle;spacing=1;"
            + (f"fillColor=#FFFFFF;dashed=1;dashPattern=3 2;" if held else f"fillColor={fill};")
        )
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
            f'height="{h:.2f}" as="geometry"/></mxCell>'
        )
        self.height = max(self.height, y + h)
        return i

    def text(self, value: str, x: float, y: float, w: float, h: float, fs: str = FS,
             align: str = "left", bold: bool = False) -> None:
        i = self.nid()
        style = (
            f"text;html=1;align={align};verticalAlign=middle;fontFamily=Arial;fontSize={fs};"
            "strokeColor=none;fillColor=none;whiteSpace=wrap;spacing=0;"
            + ("fontStyle=1;" if bold else "")
        )
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
            f'height="{h:.2f}" as="geometry"/></mxCell>'
        )
        self.height = max(self.height, y + h)

    def arrow(self, src: str, dst: str) -> None:
        i = self.nid()
        style = (
            "endArrow=blockThin;endFill=1;endSize=4;html=1;strokeColor=#000000;strokeWidth=0.8;"
            "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
        )
        self.cells.append(
            f'<mxCell id="{i}" value="" style="{style}" edge="1" parent="1" source="{src}" '
            f'target="{dst}"><mxGeometry relative="1" as="geometry"/></mxCell>'
        )

    def write(self, path: str, name: str) -> None:
        h = int(self.height + 0.5)
        xml = (
            f'<mxfile><diagram id="{name}" name="Page-1"><mxGraphModel dx="0" dy="0" grid="1" '
            'gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
            f'pageScale="1" pageWidth="{W}" pageHeight="{h}" math="0" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
            + "".join(self.cells)
            + "</root></mxGraphModel></diagram></mxfile>"
        )
        with open(path, "w") as fh:
            fh.write(xml)
        print(f"{osp.basename(path)}: canvas {W} x {h} units = {W * 0.254:.1f} x {h * 0.254:.1f} mm")


def build() -> Canvas:
    s3 = read("subset_definitions_030_summary.json")
    q = read("subset_definitions_030_q_summary.json")
    r = read("transfer_010_tmi_splits_030_summary.json")
    rc, qc = r["pinned_counts"], q["pinned_counts"]
    per = s3["n_entry_rows_per_order"]

    c = Canvas()
    # ------------------------------------------------------------ row a: three builds
    c.text("a", 0, 0, 14, 14, fs=FS_P, bold=True)
    c.text("Three builds of the same screens, one pool of records", 16, 0, 500, 14, bold=True)
    y, bh, bw, gap = 18, 84, 222, 18
    b1 = c.box(
        "010 build: triples only",
        f"the {n(s3['n_triples'])} Kuzmin 2018 and 2020 trigenic records, one value per "
        "label (the mean over sources); no singles, no doubles. The model sees only "
        "gene identity and the triple's own score.",
        0, y, bw, bh, GRAY,
    )
    b2 = c.box(
        "025 closure pool S3, labels merged",
        f"every single ({n(s3['n_singles'])}), every double whose gene pair lies inside "
        f"some triple ({n(s3['n_closure_doubles'])}) and every triple "
        f"({n(s3['n_triples'])}): {n(s3['n_s3_025'])} records. One value per label, the "
        "mean over sources, so a converted 0 can average with a measurement.",
        bw + gap, y, bw, bh, GRAY,
    )
    b3 = c.box(
        "030 per-entry pool, nothing merged",
        f"the same pool, {n(s3['n_s3'])} records, keeps every source entry: "
        f"{n(s3['n_entry_rows'])} entry rows ({n(per['1'])} on singles, {n(per['2'])} on "
        f"doubles, {n(per['3'])} on triples), each tagged with its source dataset, and the "
        "double-mutant query strain's own fitness beside the array screen's.",
        2 * (bw + gap), y, bw, bh, GRAY,
    )
    c.arrow(b1, b2)
    c.arrow(b2, b3)

    # ------------------------------------------- row b: one triple and the seven records
    y = y + bh + 16
    c.text("b", 0, y, 14, 14, fs=FS_P, bold=True)
    c.text(
        "One triple {i, j, k} of query pair ij and array gene k, the records the published "
        "identity reads, and which of them train",
        16, y, 686, 14, bold=True,
    )
    y += 18
    c.text(
        "&#964;<sub>ijk</sub> = f<sub>ijk</sub> &#8722; f<sub>ij</sub> f<sub>k</sub> &#8722; "
        "&#949;<sub>ik</sub> &#8722; &#949;<sub>jk</sub>&#160;&#160;&#160;&#160;"
        "filled: a training record&#160;&#160;&#160;&#160;white, dashed: withheld from training",
        0, y, 702, 14,
    )
    y += 18
    labw, cw, ch, cg = 70, 72, 30, 6
    groups = [
        ("singles", YELLOW, ["f<sub>i</sub>", "f<sub>j</sub>", "f<sub>k</sub>"]),
        ("query double", ORANGE, ["f<sub>ij</sub>, &#949;<sub>ij</sub>"]),
        ("array doubles", ORANGE, ["f<sub>ik</sub>, &#949;<sub>ik</sub>", "f<sub>jk</sub>, &#949;<sub>jk</sub>"]),
        ("the triple", PURPLE, ["f<sub>ijk</sub>, &#964;<sub>ijk</sub>"]),
    ]
    # header row of group names
    x = labw
    for name, _, members in groups:
        c.text(name, x, y, len(members) * (cw + cg) - cg, 12, align="center")
        x += len(members) * (cw + cg) + 14
    y += 14
    held = {
        "arm R": {"the triple"},
        "arm Q": {"the triple", "query double"},
    }
    for arm, held_groups in held.items():
        c.text(arm, 0, y, labw - 6, ch, bold=True)
        x = labw
        for name, color, members in groups:
            for m in members:
                c.chip(m, x, y, cw, ch, color, held=name in held_groups)
                x += cw + cg
            x += 14
        y += ch + 8
    c.text(
        "On both arms the held-out triple's own fitness and score leave together, since a "
        "record carries all of a genotype's entries. On arm Q the held-out query pair's "
        "double leaves as well; the array gene, the singles and the array-side doubles stay.",
        labw, y, 702 - labw, 26,
    )
    y += 30

    # ------------------------------------------------------------------ row c: the arms
    c.text("c", 0, y, 14, 14, fs=FS_P, bold=True)
    c.text("The two arms, every count from the split artifacts", 16, y, 500, 14, bold=True)
    y += 18
    bw2, bh2 = 342, 64
    c.box(
        "Arm R: random over triples",
        f"the 010 split carried by gene-set identity: {n(rc['train'])} train, "
        f"{n(rc['val'])} validation, {n(rc['test'])} test triples; every single and every "
        "closure double trains; a held-out triple's query pair, array gene and all six "
        "lower-order records are in training.",
        0, y, bw2, bh2, GRAY,
    )
    c.box(
        "Arm Q: query-pair disjoint",
        f"{n(q['n_recurring_pairs'])} recurring query pairs, {n(q['n_heldout_pairs'])} held "
        f"out with every triple they appear in: {n(qc['train'])} train, {n(qc['val'])} "
        f"validation, {n(qc['test'])} test triples; the {n(q['n_excluded_doubles'])} doubles "
        "whose pair is held out leave training, so f<sub>ij</sub> and &#949;<sub>ij</sub> "
        "are never seen; array genes are shared across the split.",
        702 - bw2, y, bw2, bh2, GRAY,
    )
    return c


canvas = build()
canvas.write(osp.join(OUT_DIR, "030-datasets-splits.drawio"), "datasets-splits")
