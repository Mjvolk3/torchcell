# notes/assets/scripts/generate_color_palette.py
# Generates notes/assets/images/color-palette.svg from the single source of truth,
# torchcell.utils.PLOT_PALETTE / PLOT_PALETTE_FILL. This is the ONE palette reference:
# an ordered, green-free qualitative series of draw.io-style (line, fill) pairs -- the
# same object colors used in Fig 1 (light fill + bold darker line). Assign in order:
# the four warm primaries (orange/red/purple/yellow) first, then blue, gray, then the
# darker variants. Run: python notes/assets/scripts/generate_color_palette.py

import os.path as osp
from torchcell.utils import PLOT_PALETTE, PLOT_PALETTE_FILL

OUT = osp.abspath(osp.join(osp.dirname(__file__), "..", "images", "color-palette.svg"))

PW, GAP = 34, 6
PITCH = PW + GAP
X0, Y_SW, SW_H = 20, 78, 44


def main():
    n = len(PLOT_PALETTE)
    width = X0 * 2 + n * PITCH - GAP
    height = 200
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Arial, DejaVu Sans, sans-serif">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        '<text x="20" y="26" font-size="15" font-weight="bold" fill="#222">'
        "TorchCell plot palette — ordered, green-free (single source of truth)</text>",
        '<text x="20" y="42" font-size="10" fill="#666">'
        "draw.io (line, fill) pairs, matching Fig 1. Assign in order: primaries (1–4), "
        "then blue/gray (5–6), then darker variants (7–). Swatch = fill with line "
        "stroke; labels are LINE / fill.</text>",
    ]

    def bracket(label, i0, i1, y=60):
        xa = X0 + i0 * PITCH
        xb = X0 + i1 * PITCH + PW
        s.append(
            f'<line x1="{xa}" y1="{y}" x2="{xb}" y2="{y}" stroke="#999" stroke-width="0.8"/>'
        )
        s.append(
            f'<text x="{(xa + xb) / 2:.0f}" y="{y - 4}" font-size="9.5" '
            f'text-anchor="middle" fill="#444">{label}</text>'
        )

    bracket("primary", 0, 3)
    bracket("secondary", 4, 5)
    bracket("extended", 6, n - 1)

    for i, (line, fill) in enumerate(zip(PLOT_PALETTE, PLOT_PALETTE_FILL)):
        x = X0 + i * PITCH
        s.append(
            f'<text x="{x + PW / 2:.0f}" y="{Y_SW - 4}" font-size="8.5" '
            f'text-anchor="middle" fill="#333">{i + 1}</text>'
        )
        # object look: fill with a bold line stroke (as used in the figures)
        s.append(
            f'<rect x="{x}" y="{Y_SW}" width="{PW}" height="{SW_H}" rx="3" '
            f'fill="{fill}" stroke="{line}" stroke-width="3"/>'
        )
        s.append(
            f'<text x="{x + PW / 2:.0f}" y="{Y_SW + SW_H + 12}" font-size="7.6" '
            f'text-anchor="middle" font-weight="bold" fill="{line}">{line.lstrip("#")}</text>'
        )
        s.append(
            f'<text x="{x + PW / 2:.0f}" y="{Y_SW + SW_H + 21}" font-size="7.6" '
            f'text-anchor="middle" fill="#888">{fill.lstrip("#")}</text>'
        )

    by = Y_SW + SW_H + 40
    s.append(
        f'<text x="20" y="{by + 15}" font-size="10.5" font-weight="bold" fill="#444">'
        "Background</text>"
    )
    s.append(
        f'<rect x="120" y="{by}" width="{PW}" height="24" rx="3" fill="#F5EEDD" '
        'stroke="#E0D6BE" stroke-width="2"/>'
    )
    s.append(
        f'<text x="{120 + PW + 8}" y="{by + 11}" font-size="8.5" fill="#333">'
        "F5EEDD fill / E0D6BE stroke</text>"
    )
    s.append("</svg>")

    with open(OUT, "w") as f:
        f.write("\n".join(s) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
