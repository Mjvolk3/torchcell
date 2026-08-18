# notes/assets/publish/scripts/drawio_true_size_import.py
# [[paper.nature-biotech.figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes/assets/publish/scripts/drawio_true_size_import.py
#
# Place SVG plots into draw.io figures at TRUE size, bypassing the GUI
# importer's dpi guessing and its maxImageSize clamp. Geometry is taken
# straight from each SVG's width/height header, converted to draw.io units
# (100 units/inch) -- WYSIWYG preserved end to end.
#
# Two modes:
#
# 1. Paste-XML (default): emit an mxGraphModel fragment to stdout.
#      python .../drawio_true_size_import.py PANEL.svg [PANEL2.svg ...] | pbcopy
#      -> in draw.io: click the canvas, Cmd+V
#
# 2. In-place replace (fully scripted, the reproducible path): swap the image
#    payload of existing image cells in a .drawio / .drawio.svg figure whose
#    geometry matches each SVG's true size, keeping the geometry untouched.
#    Cells are matched in document order; count must equal the SVGs given.
#      python .../drawio_true_size_import.py PANEL1.svg PANEL2.svg \
#        --into notes/assets/drawio/FigN.drawio.svg
#    For .drawio.svg the embedded <mxfile> is updated; the outer SVG preview
#    regenerates the next time draw.io saves the file (or `make fig` re-exports).
#
# Unit handling (per SVG width/height attribute):
#   unitless -> already draw.io units (savefig_true_size_svg convention), used as-is
#   pt -> x 100/72, mm -> x 100/25.4, cm -> x 100/2.54, in -> x 100, px -> x 100/96

import argparse
import base64
import glob
import hashlib
import html
import os.path as osp
import re
import sys
import urllib.parse
import zlib
from xml.sax.saxutils import escape

DRAWIO_UNITS_PER: dict[str, float] = {
    "": 1.0,  # savefig_true_size_svg output: numbers ARE draw.io units
    "pt": 100.0 / 72.0,
    "mm": 100.0 / 25.4,
    "cm": 100.0 / 2.54,
    "in": 100.0,
    "px": 100.0 / 96.0,
}

# Exact style draw.io writes for an embedded SVG image cell (mirrored from
# Fig7-Traditional-ML-justification-of-CGT.drawio.svg).
STYLE_PREFIX = (
    "shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;"
    "verticalAlign=top;aspect=fixed;imageAspect=0;image=data:image/svg+xml,"
)

GAP_UNITS = 20.0  # horizontal spacing between multiple pasted panels


def svg_size_in_drawio_units(svg_text: str, path: str) -> tuple[float, float]:
    m = re.search(r"<svg[^>]*>", svg_text)
    if m is None:
        raise ValueError(f"{path}: no <svg> tag found")
    tag = m.group(0)
    dims: list[float] = []
    for attr in ("width", "height"):
        am = re.search(rf'{attr}="([\d.]+)\s*([a-z]*)"', tag)
        if am is None:
            raise ValueError(f"{path}: <svg> has no {attr} attribute")
        value, unit = float(am.group(1)), am.group(2)
        if unit not in DRAWIO_UNITS_PER:
            raise ValueError(f"{path}: unsupported {attr} unit '{unit}'")
        dims.append(value * DRAWIO_UNITS_PER[unit])
    return dims[0], dims[1]


def image_cell(cell_id: str, svg_path: str, x: float) -> tuple[str, float]:
    with open(svg_path, encoding="utf-8") as fh:
        svg_text = fh.read()
    width, height = svg_size_in_drawio_units(svg_text, svg_path)
    payload = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    cell = (
        f'        <mxCell id="{cell_id}" value="" style="{STYLE_PREFIX}{payload}" '
        f'vertex="1" parent="1">\n'
        f'          <mxGeometry x="{x:.4f}" y="0" width="{width:.4f}" '
        f'height="{height:.4f}" as="geometry" />\n'
        f"        </mxCell>"
    )
    print(
        f"{svg_path}: {width:.1f} x {height:.1f} units "
        f"({width * 25.4 / 100:.1f} x {height * 25.4 / 100:.1f} mm)",
        file=sys.stderr,
    )
    return cell, width


def load_drawio(path: str) -> tuple[str, str, bool]:
    """Return (full file text, mxfile XML with uncompressed diagrams, svg_wrapped)."""
    text = open(path, encoding="utf-8").read()
    svg_wrapped = path.endswith(".svg")
    if svg_wrapped:
        m = re.search(r'content="(&lt;mxfile.*?)"', text)
        if m is None:
            raise ValueError(f"{path}: no embedded mxfile content attribute")
        # html.unescape also decodes numeric entities like &#10; (newlines), which
        # drawio CLI exports use inside the content attribute
        mxfile = html.unescape(m.group(1))
    else:
        mxfile = text

    def decompress(dm: re.Match) -> str:
        body = dm.group(2).strip()
        if body.startswith("<"):
            return dm.group(0)
        raw = zlib.decompress(base64.b64decode(body), -15).decode("utf-8")
        return dm.group(1) + urllib.parse.unquote(raw) + dm.group(3)

    mxfile = re.sub(
        r"(<diagram[^>]*>)(.*?)(</diagram>)", decompress, mxfile, flags=re.S
    )
    return text, mxfile, svg_wrapped


def save_drawio(path: str, original_text: str, mxfile: str, svg_wrapped: bool) -> None:
    if svg_wrapped:
        # escape newlines/tabs numerically so XML attribute-value normalization
        # cannot mangle them (mirrors what drawio itself writes)
        escaped = escape(
            mxfile, {'"': "&quot;", "\n": "&#10;", "\t": "&#9;", "\r": "&#13;"}
        )
        new_text = re.sub(
            r'content="&lt;mxfile.*?"',
            lambda _m: f'content="{escaped}"',
            original_text,
            count=1,
        )
        open(path, "w", encoding="utf-8").write(new_text)
    else:
        open(path, "w", encoding="utf-8").write(mxfile)


def replace_in_figure(
    fig_path: str, svg_paths: list[str], match_size: str | None = None
) -> None:
    original, mxfile, svg_wrapped = load_drawio(fig_path)

    panels = []  # (payload_b64, width, height)
    for path in svg_paths:
        svg_text = open(path, encoding="utf-8").read()
        w, h = svg_size_in_drawio_units(svg_text, path)
        panels.append((base64.b64encode(svg_text.encode()).decode("ascii"), w, h))

    if match_size is not None:
        mw, mh = (float(v) for v in match_size.lower().split("x"))
        targets = [(mw, mh)]
    else:
        targets = [(pw, ph) for (_, pw, ph) in panels]

    cell_re = re.compile(
        r'(image=data:image/svg\+xml,)([A-Za-z0-9+/=]+)([";][^>]*?>\s*'
        r'<mxGeometry[^>]*?width="([\d.]+)"[^>]*?height="([\d.]+)")'
    )
    matches = [
        m
        for m in cell_re.finditer(mxfile)
        if any(
            abs(float(m.group(4)) - tw) < 1.0 and abs(float(m.group(5)) - th) < 1.0
            for (tw, th) in targets
        )
    ]
    if len(matches) != len(panels):
        raise SystemExit(
            f"{fig_path}: found {len(matches)} image cell(s) matching the target "
            f"geometries, but {len(panels)} SVG(s) were given -- refusing."
        )

    out, last = [], 0
    for m, (payload, w, h) in zip(matches, panels):
        out.append(mxfile[last : m.start(2)])
        out.append(payload)
        suffix = mxfile[m.end(2) : m.end(3)]
        # geometry is rewritten to the panel's true size (position untouched)
        suffix = re.sub(r'width="[\d.]+"', f'width="{w:g}"', suffix, count=1)
        suffix = re.sub(r'height="[\d.]+"', f'height="{h:g}"', suffix, count=1)
        out.append(suffix)
        last = m.end(3)
        print(
            f"replaced cell {m.group(4)}x{m.group(5)} -> {w:g}x{h:g} panel",
            file=sys.stderr,
        )
    out.append(mxfile[last:])
    save_drawio(fig_path, original, "".join(out), svg_wrapped)
    print(f"wrote {fig_path}", file=sys.stderr)


CELL_RE = re.compile(
    r'image=data:image/svg\+xml,([A-Za-z0-9+/=]+)[";][^>]*?>\s*<mxGeometry([^/]*)/>'
)


def identify_panels(fig_path: str, assets_dir: str) -> None:
    """For each embedded image cell: geometry + which assets file it matches by hash.

    'current' = byte-identical to the file on disk (up to date); 'STALE/unknown' cells
    print their first text labels as identification hints.
    """
    _, mxfile, _ = load_drawio(fig_path)
    by_hash = {}
    for path in glob.glob(osp.join(assets_dir, "**", "*.svg"), recursive=True):
        by_hash[hashlib.sha256(open(path, "rb").read()).hexdigest()] = path
    for i, m in enumerate(CELL_RE.finditer(mxfile)):
        raw = base64.b64decode(m.group(1))
        geo = " ".join(m.group(2).split())
        match = by_hash.get(hashlib.sha256(raw).hexdigest())
        if match:
            print(f"cell {i}: CURRENT {osp.basename(match)} | {geo}")
        else:
            texts = re.findall(r"<text[^>]*>([^<]{2,40})</text>",
                               raw.decode("utf-8", errors="ignore"))
            hints = ", ".join(dict.fromkeys(texts))[:90]
            print(f"cell {i}: STALE/unknown | {geo} | labels: {hints}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place SVGs into draw.io at true physical size."
    )
    parser.add_argument("svgs", nargs="*", help="SVG file(s) to import")
    parser.add_argument(
        "--into",
        metavar="FIG",
        help="draw.io figure (.drawio or .drawio.svg): replace the image payloads "
        "of existing cells whose geometry matches each SVG (document order)",
    )
    parser.add_argument(
        "--identify",
        metavar="FIG",
        help="list every embedded image cell in FIG: geometry, and whether its "
        "payload is byte-identical to a current file under --assets-dir "
        "(CURRENT) or stale/unknown (with text-label hints)",
    )
    parser.add_argument(
        "--assets-dir",
        default="notes/assets/images",
        help="directory scanned (recursively) for candidate SVGs in --identify "
        "(default: notes/assets/images)",
    )
    parser.add_argument(
        "--match-size",
        metavar="WxH",
        help="with --into: match cells by this OLD geometry (draw.io units, e.g. "
        "260x220) instead of the SVGs' size, and rewrite the geometry to each "
        "SVG's true size (for panels whose dimensions changed)",
    )
    args = parser.parse_args()

    if args.identify:
        identify_panels(args.identify, args.assets_dir)
        return
    if not args.svgs:
        parser.error("SVG file(s) required unless using --identify")
    if args.into:
        replace_in_figure(args.into, args.svgs, args.match_size)
        return

    cells = []
    x = 0.0
    for i, path in enumerate(args.svgs):
        cell, width = image_cell(f"tsi-{i}", path, x)
        cells.append(cell)
        x += width + GAP_UNITS

    body = "\n".join(cells)
    print(
        "<mxGraphModel><root>\n"
        '        <mxCell id="0" />\n'
        '        <mxCell id="1" parent="0" />\n'
        f"{body}\n"
        "</root></mxGraphModel>"
    )


if __name__ == "__main__":
    main()
