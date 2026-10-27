#!/usr/bin/env python
# notes/assets/drawio/FigS-ontology-growth.split.gen.py
# [[paper.nature-biotech.si-database-figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes/assets/drawio/FigS-ontology-growth.split.gen.py
# Copied 2026-10-26 from the uncommitted worktree fig/si-database-panels (same path there)
# for manuscript SI figures; that copy is the origin of this file.
"""Split FigS-ontology-growth.drawio into its three rows, each a draw.io file of its own.

The dissertation (author, 2026-10-23) embeds the ontology-growth figure in the prose one row
at a time rather than as one full-page figure: row 1 (a, the admission loop; b, a paper
decomposed to typed records), row 2 (c, one record and its closure; d, datasets D2 and D3
arriving; e, the ontology growing), row 3 (f, the join at query time). The source of truth
stays ``FigS-ontology-growth.drawio`` (hand-maintained since 2026-10-20); this script reads
it, assigns every cell to the row that holds it, writes ``FigS-ontology-growth-r1.drawio``
(letters a, b unchanged), ``FigS-ontology-growth-r2.drawio`` (c, d, e relettered a, b, c)
and ``FigS-ontology-growth-r3.drawio`` (f, its letter dropped: one panel), and never edits the source. Re-run
after the source changes. The row logic is that of ``Fig1-torchcell-overview.split.gen.py``,
generalized to several bands.

Run from the repo root::

    python notes/assets/drawio/FigS-ontology-growth.split.gen.py [--inspect]

``--inspect`` prints the letter cells and any cell that straddles a band edge and writes
nothing.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "FigS-ontology-growth.drawio"
OUTS = {
    "r1": HERE / "FigS-ontology-growth-r1.drawio",
    "r2": HERE / "FigS-ontology-growth-r2.drawio",
    "r3": HERE / "FigS-ontology-growth-r3.drawio",
}

# The panel letters sit at canvas y 0, 220 and 440 (16 x 14 cells), each row's content below
# its letters, so the blank bands are just above y 220 and y 440.
BAND_EDGES = [218.0, 438.0]
ROWS = ["r1", "r2", "r3"]
RELETTER = {"r2": {"c": "a", "d": "b", "e": "c"}}
DROP_LETTERS = {"r3": {"f"}}  # a one-panel figure carries no letter


def read_model(path: Path) -> ET.Element:
    """The ``<mxGraphModel>`` of an uncompressed single-page ``.drawio``."""
    root = ET.parse(path).getroot()
    model = root.find(".//mxGraphModel")
    if model is None:
        sys.exit(f"{path} holds no uncompressed mxGraphModel (save it uncompressed in draw.io)")
    return model


def absolute_box(cell: ET.Element, cells: dict[str, ET.Element]) -> tuple[float, float, float, float] | None:
    """(x, y, w, h) of a vertex in page units, parents' offsets folded in."""
    geom = cell.find("mxGeometry")
    if geom is None:
        return None
    x, y = float(geom.get("x", 0)), float(geom.get("y", 0))
    w, h = float(geom.get("width", 0)), float(geom.get("height", 0))
    parent = cells.get(cell.get("parent", ""))
    while parent is not None and parent.get("id") not in ("0", "1"):
        pg = parent.find("mxGeometry")
        if pg is not None:
            x += float(pg.get("x", 0))
            y += float(pg.get("y", 0))
        parent = cells.get(parent.get("parent", ""))
    return x, y, w, h


def letter_of(cell: ET.Element) -> str | None:
    """The panel letter a cell carries, or None when it is not a letter cell."""
    if cell.get("vertex") != "1":
        return None
    plain = re.sub(r"<[^>]+>", "", html.unescape(cell.get("value") or "")).strip()
    return plain if len(plain) == 1 and plain in "abcdefgh" else None


def row_of_y(y: float) -> str:
    """The band a vertical position falls in."""
    for edge, row in zip(BAND_EDGES, ROWS):
        if y < edge:
            return row
    return ROWS[-1]


def top_ancestor(cell: ET.Element, cells: dict[str, ET.Element]) -> ET.Element:
    """The cell's outermost ancestor below the layer cell ``1``."""
    while cell.get("parent") not in ("0", "1"):
        cell = cells[cell.get("parent")]
    return cell


def assign_rows(root: ET.Element) -> tuple[dict[str, str], list[str]]:
    """Row of every cell id, and the ids of vertices that straddle a band edge.

    A group moves as one: every vertex takes the row of its outermost ancestor, read off
    the union box of the whole group, so a child never lands in another band than its
    parent. An edge follows its parent, else its source or target, else its points.
    """
    cells = {c.get("id"): c for c in root.iter("mxCell")}
    rows: dict[str, str] = {}
    straddlers: list[str] = []
    union: dict[str, list[float]] = {}
    edge_labels: list[str] = []
    for cid, cell in cells.items():
        if cell.get("vertex") != "1":
            continue
        top = top_ancestor(cell, cells)
        if top.get("edge") == "1":
            edge_labels.append(cid)
            continue
        box = absolute_box(cell, cells)
        if box is None:
            continue
        u = union.setdefault(top.get("id"), [box[1], box[1] + box[3]])
        u[0], u[1] = min(u[0], box[1]), max(u[1], box[1] + box[3])
    for cid, cell in cells.items():
        if cell.get("vertex") != "1" or cid in edge_labels:
            continue
        y0, y1 = union[top_ancestor(cell, cells).get("id")]
        rows[cid] = row_of_y((y0 + y1) / 2)
        if any(y0 < e - 4 and y1 > e + 4 for e in BAND_EDGES):
            straddlers.append(cid)
    for cid, cell in cells.items():
        if cell.get("edge") != "1":
            continue
        if cell.get("parent") in rows:
            rows[cid] = rows[cell.get("parent")]
            continue
        for end in ("source", "target"):
            ref = cell.get(end)
            if ref in rows:
                rows[cid] = rows[ref]
                break
        else:
            geom = cell.find("mxGeometry")
            ys: list[float] = []
            if geom is not None:
                for pt in geom.iter("mxPoint"):
                    if pt.get("y") is not None:
                        ys.append(float(pt.get("y")))
                parent = cells.get(cell.get("parent", ""))
                while parent is not None and parent.get("id") not in ("0", "1"):
                    pg = parent.find("mxGeometry")
                    if pg is not None:
                        ys = [v + float(pg.get("y", 0)) for v in ys]
                    parent = cells.get(parent.get("parent", ""))
            if not ys:
                raise RuntimeError(f"edge {cid} has no endpoint and no points")
            rows[cid] = row_of_y(sum(ys) / len(ys))
    for cid in edge_labels:
        rows[cid] = rows[top_ancestor(cells[cid], cells).get("id")]
    return rows, straddlers


def build_row(root: ET.Element, rows: dict[str, str], keep: str) -> ET.Element:
    """A copy of the model holding only the cells of one row, shifted to y = 0."""
    model = ET.fromstring(ET.tostring(root))
    rootcell = model.find("root")
    for cell in list(rootcell):
        cid = cell.get("id")
        if cid in ("0", "1"):
            continue
        if rows.get(cid) != keep:
            rootcell.remove(cell)
    ids = {c.get("id") for c in rootcell}
    for cell in list(rootcell):
        if cell.get("id") in ("0", "1"):
            continue
        if cell.get("edge") == "1":
            for end in ("source", "target"):
                ref = cell.get(end)
                if ref is not None and ref not in ids:
                    raise RuntimeError(f"edge {cell.get('id')} crosses a band edge via {end}={ref}")
        if cell.get("parent") not in ids:
            raise RuntimeError(f"cell {cell.get('id')} lost its parent {cell.get('parent')}")
    return model


def reletter(model: ET.Element, mapping: dict[str, str]) -> None:
    """Replace each panel letter in place, keeping the cell's HTML around it."""
    for cell in model.iter("mxCell"):
        letter = letter_of(cell)
        if letter in mapping:
            value = cell.get("value")
            new = re.sub(rf"(?<![A-Za-z]){letter}(?![A-Za-z])", mapping[letter], html.unescape(value), count=1)
            cell.set("value", new)


def write_drawio(model: ET.Element, name: str, out: Path) -> None:
    """Write an uncompressed single-page ``.drawio`` draw.io opens and exports."""
    body = ET.tostring(model, encoding="unicode")
    out.write_text(f'<mxfile host="65bd71144e">\n<diagram name="{name}" id="{name}">{body}</diagram>\n</mxfile>\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inspect", action="store_true", help="print the letters and straddlers, write nothing")
    args = parser.parse_args(argv)
    root = read_model(SOURCE)
    cells = {c.get("id"): c for c in root.iter("mxCell")}
    rows, straddlers = assign_rows(root)
    letters = [(letter_of(c), cid, absolute_box(c, cells)[1], rows[cid]) for cid, c in cells.items() if letter_of(c)]
    for letter, cid, y, row in sorted(letters, key=lambda t: t[2]):
        print(f"letter {letter} cell {cid} y={y:.1f} -> {row}")
    for cid in straddlers:
        c = cells[cid]
        print(f"straddles a band edge: {cid} box={absolute_box(c, cells)} style={(c.get('style') or '')[:80]} -> {rows[cid]}")
    counts = {r: sum(1 for v in rows.values() if v == r) for r in ROWS}
    print(f"cells per row: {counts}")
    if args.inspect:
        return 0
    for row in ROWS:
        model = build_row(root, rows, row)
        if row in RELETTER:
            reletter(model, RELETTER[row])
        if row in DROP_LETTERS:
            rootcell = model.find("root")
            for cell in list(rootcell):
                if letter_of(cell) in DROP_LETTERS[row]:
                    rootcell.remove(cell)
        write_drawio(model, f"FigS-ontology-growth-{row}", OUTS[row])
        print(f"wrote {OUTS[row]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
