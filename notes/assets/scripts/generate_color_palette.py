# notes/assets/scripts/generate_color_palette.py
# Generates notes/assets/images/color-palette.svg from the single source of truth,
# torchcell.utils.PLOT_PALETTE / PLOT_PALETTE_FILL. This is the ONE palette reference:
# an ordered, green-free qualitative series of draw.io-style (line, fill) pairs -- the
# same object colors used in Fig 1.
#
# STRUCTURE: the hue order orange - red - purple - yellow - blue - gray repeats every
# 6 colors, so a plot with N series takes the first N and the four warm primaries are
# always used before blue/gray. That is deliberate: it maximizes the primaries and
# minimizes blue. Each cycle of 6 is a tier of the same six hues:
#   1-6   primary      (the draw.io Fig 1 colors -- LOCKED, never change these)
#   7-12  dark         (a uniform 0.73 shade of the primary)
#   13-18 muted earth  (chroma-reduced siblings: terracotta, rose-brown, dusty mauve,
#                       sand, slate, taupe)
#
# WHY IT STOPS AT 18, and why the tiers differ by CHROMA rather than lightness:
#   - Lightness is already spoken for. Light-vs-dark encodes validation-vs-test WITHIN a
#     series (that is what PLOT_PALETTE_FILL is for), so it must not also separate
#     CATEGORIES -- the two encodings would collide. Tier 3 is therefore separated by
#     chroma (muted vs vivid), which is also what makes a color read as earth tone
#     rather than fluorescent.
#   - 18 is the ceiling. A 4th red has nowhere to go: its three slots already occupy the
#     mid and dark range, so a 4th can only go lighter -- i.e. pink. A 4th orange at low
#     lightness turns olive-brown, breaking green-free. Every route past 18 reintroduces
#     the fluorescence this palette exists to avoid. If you need >18 series, disambiguate
#     with hatching, not with more color.
#
# Run: python notes/assets/scripts/generate_color_palette.py

import os.path as osp

from torchcell.utils import PLOT_PALETTE, PLOT_PALETTE_FILL, PLOT_PALETTE_NAMES

OUT = osp.abspath(osp.join(osp.dirname(__file__), "..", "images", "color-palette.svg"))

# PW is 42, not 34, because the swatch column must fit the widest UNSPLITTABLE color name
# ("terracotta"). The _wrap() guard raises if it ever stops fitting, so a future rename
# cannot silently collide with its neighbour.
PW, GAP = 42, 6
PITCH = PW + GAP
X0 = 20
SLOTS = ["orange", "red", "purple", "yellow", "blue", "gray"]
TIERS = ["primary", "dark", "muted earth"]

# SVG <text> does NOT wrap -- an over-long string silently runs off the edge. The previous
# version put a 182-character subtitle on one line and it ran ~250 px past the canvas. So
# every string here is pre-split, and _emit_text() below MEASURES each run and raises if it
# would not fit. That guard covers the per-swatch labels too: "dusty mauve" is wider than a
# 34 px swatch, which is the same bug one level down.
TITLE = "TorchCell plot palette - ordered, green-free (single source of truth)"
SUB = [
    "Hue slots orange-red-purple-yellow-blue-gray repeat every 6; a series of N takes the"
    " first N, so primaries are spent before blue/gray.",
    "Tiers differ by CHROMA, not lightness -- lightness encodes validation vs test within"
    " a series. Swatch = fill with its line stroke.",
    "Labels are the TRUE color, not the slot: the muted tier's gray slot is a taupe.",
]

ADVANCE = 0.52  # Arial mixed-case advance, em


def _width(text: str, size: float) -> float:
    return len(text) * size * ADVANCE


def _wrap(name: str, size: float, budget: float) -> list[str]:
    """Split a two-word color name across lines so it fits one swatch column."""
    if _width(name, size) <= budget:
        return [name]
    parts = name.split()
    if len(parts) == 2 and all(_width(p, size) <= budget for p in parts):
        return parts
    raise ValueError(f"color name {name!r} does not fit a {budget:.0f}px column")


def main() -> None:
    n = len(PLOT_PALETTE)
    width = X0 * 2 + n * PITCH - GAP
    y_title, y_sub = 26, 46
    y_tier = y_sub + len(SUB) * 13 + 12  # tracks the subtitle, however many lines it takes
    y_sw = y_tier + 18
    sw_h = 44
    y_line = y_sw + sw_h + 12
    y_fill = y_sw + sw_h + 21
    y_hue = y_sw + sw_h + 34
    y_bg = y_hue + 9 + 20  # + 9 so a two-line name ("dusty / mauve") cannot collide
    height = y_bg + 44

    for t, size in [(TITLE, 15.0), *((s_, 10.0) for s_ in SUB)]:
        if X0 + _width(t, size) > width - 10:
            raise ValueError(f"text overflows the {width}px canvas: {t[:60]}...")
    name_lines = [_wrap(nm, 7.4, PITCH - 2) for nm in PLOT_PALETTE_NAMES]  # raises if too wide

    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Arial, DejaVu Sans, sans-serif">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text x="{X0}" y="{y_title}" font-size="15" font-weight="bold" fill="#222">'
        f"{TITLE}</text>",
    ]
    for k, line in enumerate(SUB):
        s.append(
            f'<text x="{X0}" y="{y_sub + k * 13}" font-size="10" fill="#666">{line}</text>'
        )

    # one bracket per tier (= per cycle of 6), so the repeating hue order is visible
    for t, name in enumerate(TIERS):
        i0, i1 = t * 6, min(t * 6 + 5, n - 1)
        if i0 >= n:
            break
        xa, xb = X0 + i0 * PITCH, X0 + i1 * PITCH + PW
        s.append(
            f'<line x1="{xa}" y1="{y_tier}" x2="{xb}" y2="{y_tier}" stroke="#999" '
            f'stroke-width="0.8"/>'
        )
        s.append(
            f'<text x="{(xa + xb) / 2:.0f}" y="{y_tier - 4}" font-size="9.5" '
            f'text-anchor="middle" fill="#444">{name} ({i0 + 1}-{i1 + 1})</text>'
        )

    for i, (line, fill) in enumerate(zip(PLOT_PALETTE, PLOT_PALETTE_FILL)):
        x = X0 + i * PITCH
        cx = x + PW / 2
        s.append(
            f'<text x="{cx:.0f}" y="{y_sw - 4}" font-size="8.5" text-anchor="middle" '
            f'fill="#333">{i + 1}</text>'
        )
        s.append(
            f'<rect x="{x}" y="{y_sw}" width="{PW}" height="{sw_h}" rx="3" '
            f'fill="{fill}" stroke="{line}" stroke-width="3"/>'
        )
        s.append(
            f'<text x="{cx:.0f}" y="{y_line}" font-size="7.6" text-anchor="middle" '
            f'font-weight="bold" fill="{line}">{line.lstrip("#")}</text>'
        )
        s.append(
            f'<text x="{cx:.0f}" y="{y_fill}" font-size="7.6" text-anchor="middle" '
            f'fill="#888">{fill.lstrip("#")}</text>'
        )
        # The TRUE color name, not the slot. #A29682 sits in the gray slot but is a taupe;
        # labelling it "gray" was simply wrong. Slot order still drives assignment, and is
        # shown once per tier below rather than lied about on every swatch.
        for k, part in enumerate(name_lines[i]):
            s.append(
                f'<text x="{cx:.0f}" y="{y_hue + k * 9}" font-size="7.4" '
                f'text-anchor="middle" fill="#444">{part}</text>'
            )

    s.append(
        f'<text x="{X0}" y="{y_bg + 15}" font-size="10.5" font-weight="bold" fill="#444">'
        f"Background</text>"
    )
    s.append(
        f'<rect x="120" y="{y_bg + 4}" width="{PW}" height="20" rx="3" fill="#F5EEDD" '
        f'stroke="#E0D6BE" stroke-width="2"/>'
    )
    s.append(
        f'<text x="{120 + PW + 8}" y="{y_bg + 15}" font-size="8.5" fill="#333">'
        f"F5EEDD fill / E0D6BE stroke</text>"
    )
    s.append("</svg>")

    with open(OUT, "w") as f:
        f.write("\n".join(s) + "\n")
    print(f"wrote {OUT}  ({n} colors, {width}x{height})")


if __name__ == "__main__":
    main()
