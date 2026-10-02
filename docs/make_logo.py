# docs/make_logo.py
# [[docs.make_logo]]
# https://github.com/Mjvolk3/torchcell/tree/main/docs/make_logo.py
"""Derive the docs logo and the cell mark from the draw.io export.

``notes/assets/drawio/torchcell-logo.drawio.png`` is exported with an opaque white
canvas, which shows as a white box on the theme's grey sidebar. This script flood-fills
transparency inward from the four edges over pixels that are white (every channel at or
above ``WHITE``), so white inside the letters and the cell stays, and writes the result
to ``docs/source/_static/torchcell-logo.png``.

It then cuts the cell out of that logo, without the wordmark, onto a transparent
square: ``docs/source/_static/torchcell-mark.png``, and the same image as the website's
favicon, ``website/static/img/favicon.png``. A wordmark is unreadable at favicon size;
the cell is not. Run from the repo root::

    python docs/make_logo.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "notes" / "assets" / "drawio" / "torchcell-logo.drawio.png"
TARGET = REPO / "docs" / "source" / "_static" / "torchcell-logo.png"
MARK_TARGET = REPO / "docs" / "source" / "_static" / "torchcell-mark.png"
WEBSITE_FAVICON = REPO / "website" / "static" / "img" / "favicon.png"
WHITE = 245
MARK_PADDING = 2


def transparent_background(image: Image.Image) -> Image.Image:
    """Return a copy with the edge-connected white region made fully transparent."""
    out = image.convert("RGBA").copy()
    width, height = out.size
    pixels = out.load()
    assert pixels is not None
    seen = bytearray(width * height)
    queue: deque[tuple[int, int]] = deque()

    def white(x: int, y: int) -> bool:
        value = pixels[x, y]
        assert isinstance(value, tuple)
        r, g, b, a = value
        return a == 0 or (r >= WHITE and g >= WHITE and b >= WHITE)

    for x in range(width):
        queue.append((x, 0))
        queue.append((x, height - 1))
    for y in range(height):
        queue.append((0, y))
        queue.append((width - 1, y))
    while queue:
        x, y = queue.popleft()
        index = y * width + x
        if seen[index] or not white(x, y):
            continue
        seen[index] = 1
        pixels[x, y] = (0, 0, 0, 0)
        if x > 0:
            queue.append((x - 1, y))
        if x < width - 1:
            queue.append((x + 1, y))
        if y > 0:
            queue.append((x, y - 1))
        if y < height - 1:
            queue.append((x, y + 1))
    return out


def cell_mark(logo: Image.Image) -> Image.Image:
    """Return the cell alone, on a transparent square, from the transparent logo.

    The cell is the leftmost drawing of the logo and touches no letter, so it is the
    set of opaque pixels connected to the first opaque pixel of the middle row. Those
    pixels are cut out, cropped to their bounding box, and centered on a square canvas
    ``MARK_PADDING`` pixels larger on every side, which is the shape a favicon needs.
    """
    width, height = logo.size
    pixels = logo.load()
    assert pixels is not None

    def opaque(x: int, y: int) -> bool:
        value = pixels[x, y]
        assert isinstance(value, tuple)
        return value[3] > 0

    middle = height // 2
    seed = next((x, middle) for x in range(width) if opaque(x, middle))
    mark = Image.new("RGBA", logo.size, (0, 0, 0, 0))
    mark_pixels = mark.load()
    assert mark_pixels is not None
    seen = bytearray(width * height)
    queue: deque[tuple[int, int]] = deque([seed])
    while queue:
        x, y = queue.popleft()
        index = y * width + x
        if seen[index] or not opaque(x, y):
            continue
        seen[index] = 1
        mark_pixels[x, y] = pixels[x, y]
        if x > 0:
            queue.append((x - 1, y))
        if x < width - 1:
            queue.append((x + 1, y))
        if y > 0:
            queue.append((x, y - 1))
        if y < height - 1:
            queue.append((x, y + 1))
    box = mark.getbbox()
    assert box is not None
    cell = mark.crop(box)
    side = max(cell.size) + 2 * MARK_PADDING
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(cell, ((side - cell.size[0]) // 2, (side - cell.size[1]) // 2))
    return square


def main() -> None:
    """Write the transparent logo, the cell mark, and the website favicon."""
    logo = transparent_background(Image.open(SOURCE))
    logo.save(TARGET, optimize=True)
    cleared = logo.getchannel("A").histogram()[0]
    print(
        f"{TARGET.relative_to(REPO)}: {logo.size[0]}x{logo.size[1]}, {cleared} transparent pixels"
    )
    mark = cell_mark(logo)
    for target in (MARK_TARGET, WEBSITE_FAVICON):
        mark.save(target, optimize=True)
        print(f"{target.relative_to(REPO)}: {mark.size[0]}x{mark.size[1]}")


if __name__ == "__main__":
    main()
