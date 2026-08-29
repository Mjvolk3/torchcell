#!/usr/bin/env python
# paper/nature-biotech/scripts/drawio_font_band.py
# [[paper.nature-biotech.style-guide]]
# https://github.com/Mjvolk3/torchcell/tree/main/paper/nature-biotech/scripts/drawio_font_band.py
"""Audit and retype draw.io font sizes against Nature's figure-text band.

draw.io's canvas is 100 units per inch and its font-size fields are in those
canvas units, so a number typed on the canvas prints at ``number * 0.72`` pt. A
label typed ``8`` is 5.76 pt on the page. The conversion is recorded in CLAUDE.md
and paper/nature-biotech/figures/README.md; this is the tool that applies it.

Nature's band is 5 pt minimum and 7 pt maximum for figure text, with 8 pt bold
lowercase for panel letters only. The maximum matters as much as the minimum:
figure lettering has to sit below the caption in the page hierarchy.

TWO PLACES CARRY A SIZE, and only one of them is obvious. An mxCell's style has
``fontSize=N``, but a cell whose value is HTML can also carry an inline
``font-size: Npx``, and the inline rule WINS. Fig 7's node numbers are the worked
example: their cells say ``fontSize=12`` (8.64 pt) while their values say
``font-size: 6px`` (4.32 pt), and 4.32 pt is what prints. An audit that reads
only the style reports those labels as too big when they are in fact too small.

GROWING TEXT IS THE DANGEROUS DIRECTION. draw.io does not reflow a box when its
label changes size, so raising a label that is already tight overflows it, and
there is no error -- the text simply runs outside its shape in the exported PDF.
Shrinking is safe. That asymmetry is why ``--target`` exists: on a dense diagram
the right move is usually the 5 pt floor, the smallest compliant size and so the
smallest change, rather than the 6 pt house size.

Usage::

    # report only, no writes
    python paper/nature-biotech/scripts/drawio_font_band.py --check FILE...

    # retype: undersized -> 5 pt, oversized -> 6 pt, panel letters -> 8 pt
    python paper/nature-biotech/scripts/drawio_font_band.py --fix FILE... \\
        --grow-to 5 --shrink-to 6 --panel-letters 8

``--panel-letters`` applies only to cells whose entire label is a single
lowercase letter, which is what a panel letter is. Nothing else is promoted to
8 pt, because 8 pt is over the maximum for ordinary figure text.
"""

from __future__ import annotations

import argparse
import base64
import collections
import html
import pathlib
import re
import sys
import urllib.parse
import zlib

# 100 canvas units per inch, 72 pt per inch.
UNITS_TO_PT = 0.72
NATURE_MIN_PT = 5.0
NATURE_MAX_PT = 7.0
PANEL_LETTER_PT = 8.0

# THE LADDER. Every label lands on one of these four canvas values and nothing
# in between, so a figure carries a handful of deliberate sizes rather than a
# spread of near-identical ones that no reader can tell apart. Fig 1 had labels
# at 5.76 pt beside labels at 5.98 pt, which is a 0.2 pt difference doing no work.
#
# Canvas values are kept to TENTHS. A whole printed point needs a repeating
# decimal on the canvas, because 0.72 = 18/25 and only point values that are
# multiples of 18 come out whole, so exactness is not on offer at any sane
# precision. Tenths land within 0.03 pt of the target, which is far below what
# prints differently, and they stay readable in the draw.io font field.
#
# 5 pt rounds UP to 7.0: the exact value is 6.944, and 6.9 prints at 4.97 pt,
# still under the floor.
LADDER_UNITS = {
    5.0: 7.0,    # 5.04 pt -- the floor, for labels too cramped to grow further
    6.0: 8.3,    # 5.98 pt -- the house size, matches the matplotlib panels
    7.0: 9.7,    # 6.98 pt -- the maximum for ordinary figure text
}
PANEL_LETTER_UNITS = 11.1  # 7.99 pt -- panel letters ONLY, never body text


def _decode_diagram(body: str) -> str:
    """The XML of one <diagram>, whether stored plain or deflate+base64."""
    body = body.strip()
    if body.startswith("<mxGraphModel"):
        return body
    raw = zlib.decompress(base64.b64decode(body), -15).decode("utf-8")
    return urllib.parse.unquote(raw)


def _encode_diagram(xml: str) -> str:
    """Re-encode as draw.io writes it: quote, raw-deflate, base64."""
    quoted = urllib.parse.quote(xml, safe="~()*!.'")
    comp = zlib.compressobj(9, zlib.DEFLATED, -15)
    packed = comp.compress(quoted.encode("utf-8")) + comp.flush()
    return base64.b64encode(packed).decode("ascii")


class Drawio:
    """A .drawio or .drawio.svg file, with its diagram XML addressable."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.text = path.read_text(errors="ignore")
        self.is_svg = path.name.endswith(".drawio.svg")
        if self.is_svg:
            m = re.search(r'content="([^"]*)"', self.text)
            if not m:
                sys.exit(f"{path}: .drawio.svg with no embedded content attribute")
            self.span = m.span(1)
            self.container = html.unescape(m.group(1))
        else:
            self.container = self.text
        self.bodies = re.findall(r"<diagram[^>]*>(.*?)</diagram>", self.container, re.S)
        self.xmls = [_decode_diagram(b) for b in self.bodies]

    def write(self, new_xmls: list[str]) -> None:
        container = self.container
        for old_body, new_xml in zip(self.bodies, new_xmls):
            was_plain = old_body.strip().startswith("<mxGraphModel")
            new_body = new_xml if was_plain else _encode_diagram(new_xml)
            container = container.replace(old_body, new_body, 1)
        if self.is_svg:
            a, b = self.span
            self.text = self.text[:a] + html.escape(container, quote=True) + self.text[b:]
        else:
            self.text = container
        self.path.write_text(self.text)


def _verdict(pt: float) -> str:
    if abs(pt - PANEL_LETTER_PT) < 0.06:
        return "panel-letter"
    if pt < NATURE_MIN_PT:
        return "under"
    if pt > NATURE_MAX_PT:
        return "over"
    return "ok"


def _is_panel_letter(value: str) -> bool:
    txt = re.sub(r"<[^>]+>", "", html.unescape(value)).strip()
    return len(txt) == 1 and txt.isalpha() and txt.islower()


def audit(xml: str) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for style in re.findall(r'style="([^"]*)"', xml):
        m = re.search(r"fontSize=([0-9.]+)", style)
        if m:
            c[("style", float(m.group(1)))] += 1
    for value in re.findall(r'value="([^"]*)"', xml):
        for px in re.findall(r"font-size:\s*([0-9.]+)px", html.unescape(value)):
            c[("inline", float(px))] += 1
    return c


def fix(xml: str, grow_pt: float, shrink_pt: float, letter_pt: float) -> tuple[str, int]:
    """Retype every out-of-band size. Returns the new XML and a change count."""
    n = 0

    def size_for(cur_units: float, is_letter: bool) -> float | None:
        """The ladder rung this size belongs on, or None if it is already there.

        Out-of-band sizes move to the requested rung. IN-BAND sizes are snapped
        to the nearest rung too, which is the whole point of a ladder: a legal
        but off-ladder 5.76 pt sitting beside 5.98 pt is two sizes doing one job.
        """
        pt = cur_units * UNITS_TO_PT
        if is_letter:
            units = PANEL_LETTER_UNITS
        elif pt < NATURE_MIN_PT:
            units = LADDER_UNITS[grow_pt]
        elif pt > NATURE_MAX_PT:
            units = LADDER_UNITS[shrink_pt]
        else:
            units = LADDER_UNITS[min(LADDER_UNITS, key=lambda t: abs(t - pt))]
        return None if abs(units - cur_units) < 0.05 else units

    def do_cell(m: re.Match) -> str:
        nonlocal n
        tag = m.group(0)
        value = re.search(r'value="([^"]*)"', tag)
        letter = bool(value) and _is_panel_letter(value.group(1))

        def sub_style(sm: re.Match) -> str:
            nonlocal n
            new = size_for(float(sm.group(1)), letter)
            if new is None:
                return sm.group(0)
            n += 1
            return f"fontSize={new:g}"

        tag = re.sub(r"fontSize=([0-9.]+)", sub_style, tag)

        # The inline rule wins over the style, so it has to be retyped too.
        def sub_inline(im: re.Match) -> str:
            nonlocal n
            new = size_for(float(im.group(1)), letter)
            if new is None:
                return im.group(0)
            n += 1
            return f"font-size: {new:g}px"

        # value="" is HTML-escaped inside the tag; operate on the escaped form so
        # the surrounding attribute quoting is untouched.
        return re.sub(r"font-size:\s*([0-9.]+)px", sub_inline, tag)

    xml = re.sub(r"<mxCell\b[^>]*?>", do_cell, xml)
    grow_units = LADDER_UNITS[grow_pt]
    return _widen_grown_labels(xml, grow_units), n


# A borderless wrapping label is sized to its text, so raising its font makes it
# wrap onto more lines and spill into whatever sits above or below. Fig 1's
# "Annotate datasets for query" landed on top of the "Ontology" heading, and
# "CRISPR AID" broke across two lines with "AID" over the diagram.
#
# Widening such a label by the same ratio the text grew keeps it on one line.
# This is only safe because the cells it applies to have no stroke and no fill:
# nothing about the box is visible, so a wider box changes nothing on the page
# except where the text is allowed to run. Cells with a border or a fill are left
# alone, because widening those WOULD be visible.
_BORDERLESS = re.compile(r"strokeColor=none")
_HAS_FILL = re.compile(r"fillColor=(?!none)")


def _widen_grown_labels(xml: str, grow_units: float) -> str:
    """Widen borderless wrapping labels in proportion to the font they gained.

    ``grow_units`` is the canvas size those labels were raised TO. The size they
    came from is whatever was under the floor, in practice 6, so the width scales
    by ``grow_units / 6``.
    """
    ratio = grow_units / 6.0
    if ratio <= 1.0:
        return xml
    marker = f"font-size: {grow_units:g}px"

    def do(m: re.Match) -> str:
        cell, geom = m.group(1), m.group(2)
        style = (re.search(r'style="([^"]*)"', cell) or [None, ""])[1]
        if not (_BORDERLESS.search(style) and "whiteSpace=wrap" in style):
            return m.group(0)
        if _HAS_FILL.search(style):
            return m.group(0)
        if marker not in html.unescape(cell):
            return m.group(0)
        new_geom = re.sub(
            r'width="([0-9.]+)"',
            lambda wm: f'width="{float(wm.group(1)) * ratio:.0f}"',
            geom,
        )
        return cell + m.group(0)[len(cell):].replace(geom, new_geom, 1)

    return re.sub(r'(<mxCell\b[^>]*?>)(\s*<mxGeometry\b[^>]*?/>)', do, xml, flags=re.S)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--check", action="store_true", help="report only (default)")
    ap.add_argument("--fix", action="store_true", help="rewrite the sources in place")
    ap.add_argument("--grow-to", type=float, default=5.0, metavar="PT",
                    choices=sorted(LADDER_UNITS),
                    help="ladder rung for labels under the floor (default 5, the "
                         "smallest change and so the least likely to overflow)")
    ap.add_argument("--shrink-to", type=float, default=6.0, metavar="PT",
                    choices=sorted(LADDER_UNITS),
                    help="ladder rung for labels over the maximum (default 6)")
    ap.add_argument("--panel-letters", type=float, default=8.0, metavar="PT",
                    help="target pt for single lowercase-letter labels (default 8)")
    args = ap.parse_args()

    bad_total = 0
    for name in args.files:
        p = pathlib.Path(name)
        if not p.exists():
            print(f"{name}: MISSING")
            bad_total += 1
            continue
        d = Drawio(p)
        print(f"\n=== {p.name} ===")
        counts: collections.Counter = collections.Counter()
        for xml in d.xmls:
            counts.update(audit(xml))
        for (where, units), k in sorted(counts.items(), key=lambda kv: kv[0][1]):
            pt = units * UNITS_TO_PT
            v = _verdict(pt)
            mark = "" if v in ("ok", "panel-letter") else "   <-- out of band"
            print(f"  {where:6} {units:<6g} -> {pt:5.2f} pt  x{k:<4} {v}{mark}")
            if v in ("under", "over"):
                bad_total += k

        if args.fix:
            new_xmls, total = [], 0
            for xml in d.xmls:
                nx, n = fix(xml, args.grow_to, args.shrink_to, args.panel_letters)
                new_xmls.append(nx)
                total += n
            if total:
                d.write(new_xmls)
                print(f"  -> retyped {total} size(s); re-export with `make -C paper/nature-biotech fig`")
            else:
                print("  -> nothing to change")

    if args.check and bad_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
