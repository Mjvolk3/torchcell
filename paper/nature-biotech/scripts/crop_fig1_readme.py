# paper/nature-biotech/scripts/crop_fig1_readme.py
# [[paper.nature-biotech.scripts.crop_fig1_readme]]
# https://github.com/Mjvolk3/torchcell/tree/main/paper/nature-biotech/scripts/crop_fig1_readme
"""Crop the top row (panels a, b, c) of the paper's Figure 1 for the README.

Reads the manuscript's own export, ``paper/nature-biotech/figures/
Fig1-torchcell-overview.pdf``, renders it with poppler, finds the blank
horizontal band that separates the top panel row from the rest, and writes
the top row to ``$ASSET_IMAGES_DIR/Fig1-torchcell-overview-abc.png``. The
README embeds that file. Re-run after the figure changes.

Run from the repo root::

    python paper/nature-biotech/scripts/crop_fig1_readme.py
"""

import os
import os.path as osp
import subprocess
import tempfile

import numpy as np
from dotenv import load_dotenv
from PIL import Image

PDF = "paper/nature-biotech/figures/Fig1-torchcell-overview.pdf"
OUT_NAME = "Fig1-torchcell-overview-abc.png"
DPI = 300
# The row gap sits under the top third of the page; search this band of the height.
SEARCH_FRACTION = (0.35, 0.65)
INK_THRESHOLD = 240  # gray level below which a pixel counts as ink
BLANK_ROW_FRACTION = 0.002  # a row with fewer ink pixels than this is blank
PAD_PX = 12


def render(pdf: str, dpi: int, workdir: str) -> Image.Image:
    """Rasterize the single-page PDF with poppler and return it as RGB."""
    stem = osp.join(workdir, "page")
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", "-singlefile", pdf, stem], check=True
    )
    return Image.open(stem + ".png").convert("RGB")


def top_row_bottom(gray: np.ndarray) -> int:
    """Index of the first row past the top panel row, at the middle of the gap."""
    ink = (gray < INK_THRESHOLD).mean(axis=1)
    h = gray.shape[0]
    lo, hi = (int(h * f) for f in SEARCH_FRACTION)
    blank = ink[lo:hi] < BLANK_ROW_FRACTION
    best_start, best_len, start = -1, 0, None
    for i, b in enumerate(np.append(blank, False)):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    if best_len == 0:
        raise RuntimeError(
            "no blank band between panel rows found in the search window"
        )
    return lo + best_start + best_len // 2


def trim(img: np.ndarray, pad: int) -> np.ndarray:
    """Cut the white margins down to ``pad`` pixels around the ink."""
    gray = img.mean(axis=2)
    rows = np.where((gray < INK_THRESHOLD).any(axis=1))[0]
    cols = np.where((gray < INK_THRESHOLD).any(axis=0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def main() -> None:
    """Render, crop to the top panel row, trim, and write the README image."""
    load_dotenv()
    out_dir = os.environ["ASSET_IMAGES_DIR"]
    with tempfile.TemporaryDirectory() as workdir:
        page = np.asarray(render(PDF, DPI, workdir))
    cut = top_row_bottom(page.mean(axis=2))
    top = trim(page[:cut], PAD_PX)
    out = osp.join(out_dir, OUT_NAME)
    Image.fromarray(top).save(out, optimize=True)
    print(
        f"cut at row {cut} of {page.shape[0]}; wrote {out} ({top.shape[1]}x{top.shape[0]})"
    )


if __name__ == "__main__":
    main()
