# docs/make_logo.py
# [[docs.make_logo]]
# https://github.com/Mjvolk3/torchcell/tree/main/docs/make_logo.py
"""Derive the docs logo from the draw.io export with a transparent background.

``notes/assets/drawio/torchcell-logo.drawio.png`` is exported with an opaque white
canvas, which shows as a white box on the theme's grey sidebar. This script flood-fills
transparency inward from the four edges over pixels that are white (every channel at or
above ``WHITE``), so white inside the letters and the cell stays, and writes the result
to ``docs/source/_static/torchcell-logo.png``. Run from the repo root::

    python docs/make_logo.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "notes" / "assets" / "drawio" / "torchcell-logo.drawio.png"
TARGET = REPO / "docs" / "source" / "_static" / "torchcell-logo.png"
WHITE = 245


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


def main() -> None:
    """Write the transparent logo and report how many pixels were cleared."""
    logo = transparent_background(Image.open(SOURCE))
    logo.save(TARGET, optimize=True)
    cleared = logo.getchannel("A").histogram()[0]
    print(
        f"{TARGET.relative_to(REPO)}: {logo.size[0]}x{logo.size[1]}, {cleared} transparent pixels"
    )


if __name__ == "__main__":
    main()
