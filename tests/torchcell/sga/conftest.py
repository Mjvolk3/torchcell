# tests/torchcell/sga/conftest.py
# [[tests.torchcell.sga.conftest]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sga/conftest.py
"""Synthetic 3x4 colony plates shared by the ``torchcell.sga`` image tests.

One geometry, two illuminations. The array is 3 rows x 4 columns on a 60 px pitch with
node rows at y = 90, 150, 210 and node columns at x = 110, 170, 230, 290 (image order,
1-based (row, col) below). Every colony is an axis-aligned shape whose pixel count and
centroid are hand-computable:

* normal wells: a 7x7 square (49 px, centroid exactly on the node);
* (1, 4) "M" well: a 9x9 primary square (81 px) plus a 7x7 secondary square 18 px down
  and 18 px right of it (center distance 18*sqrt(2) = 25.46 px, more than the 0.4-pitch
  multi-colony separation of 24 px);
* (2, 2) "C" well: a U shape in a 15x19 bounding box (two 5-wide bars 15 tall joined by
  a 5-tall connector across rows 10-14; 195 px, boundary 82 px, circularity
  4*pi*195/82^2 = 0.3644), drawn one row up so its centroid sits 0.15 px below the node;
* (2, 3): empty (a missing colony);
* (3, 2): a normal square plus a 14x14 "gash" patch (rows +10..+23, cols -20..-7 from the
  node) at the tear intensity (0 on the dark-field plate, 255 on the backlit plate).

Dark-field plate: 300x400, background 0, agar 60 over rows 20-279 / cols 20-379,
colonies 200. Backlit plate: 300x400 all at agar 213 (plate on a light panel), colonies
190. The "disk" variant replaces the (3, 3) square with a digital disk of radius 15
(709 px, the number of integer points with x^2 + y^2 <= 225) so grid-guided recovery
has a colony that fills the 0.28-pitch probe core.

``plate_masks`` is the instance-label image Cellpose would return for the standard
geometry: ids run in (row, col) raster order, 1..12, with the M primary = 4 and the M
secondary = 5.
"""

from __future__ import annotations

import os.path as osp
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

ROWS_Y = (90, 150, 210)
COLS_X = (110, 170, 230, 290)
PITCH = 60
MISSING_WELL = (2, 3)
U_WELL = (2, 2)
M_WELL = (1, 4)
DISK_WELL = (3, 3)
GASH_WELL = (3, 2)
SECONDARY_OFFSET = (18, 18)
DISK_RADIUS = 15


def paint_square(arr: NDArray[Any], cy: int, cx: int, side: int, val: int) -> None:
    """Fill the ``side`` x ``side`` square centered on (cy, cx) with ``val``."""
    h = side // 2
    arr[cy - h : cy - h + side, cx - h : cx - h + side] = val


def paint_u(arr: NDArray[Any], cy: int, cx: int, val: int) -> None:
    """The 15x19 U: bars over cols 0-4 and 14-18, connector over rows 10-14."""
    r0, c0 = cy - 8, cx - 9
    arr[r0 : r0 + 15, c0 : c0 + 5] = val
    arr[r0 : r0 + 15, c0 + 14 : c0 + 19] = val
    arr[r0 + 10 : r0 + 15, c0 : c0 + 19] = val


def paint_disk(arr: NDArray[Any], cy: int, cx: int, r: int, val: int) -> None:
    """Fill the digital disk x^2 + y^2 <= r^2 centered on (cy, cx)."""
    y, x = np.ogrid[-cy : arr.shape[0] - cy, -cx : arr.shape[1] - cx]
    arr[(x * x + y * y) <= r * r] = val


def well_shapes(
    disk_at_33: bool = False,
) -> dict[tuple[int, int], list[tuple[int, int, str]]]:
    """(row, col) -> [(cy, cx, kind)], kind in {"sq7", "sq9", "u", "disk"}."""
    out: dict[tuple[int, int], list[tuple[int, int, str]]] = {}
    for ri, y in enumerate(ROWS_Y, 1):
        for ci, x in enumerate(COLS_X, 1):
            if (ri, ci) == MISSING_WELL:
                out[(ri, ci)] = []
            elif (ri, ci) == U_WELL:
                out[(ri, ci)] = [(y, x, "u")]
            elif (ri, ci) == M_WELL:
                dy, dx = SECONDARY_OFFSET
                out[(ri, ci)] = [(y, x, "sq9"), (y + dy, x + dx, "sq7")]
            elif (ri, ci) == DISK_WELL and disk_at_33:
                out[(ri, ci)] = [(y, x, "disk")]
            else:
                out[(ri, ci)] = [(y, x, "sq7")]
    return out


def paint(arr: NDArray[Any], cy: int, cx: int, kind: str, val: int) -> None:
    """Draw one shape of ``well_shapes``."""
    if kind == "u":
        paint_u(arr, cy, cx, val)
    elif kind == "disk":
        paint_disk(arr, cy, cx, DISK_RADIUS, val)
    elif kind == "sq9":
        paint_square(arr, cy, cx, 9, val)
    else:
        paint_square(arr, cy, cx, 7, val)


def make_plate(mode: str, disk_at_33: bool = False) -> NDArray[Any]:
    """uint8 grayscale plate; ``mode`` is "dark" (dark-field) or "backlit"."""
    if mode == "dark":
        agar, colony, gash = 60, 200, 0
        g = np.zeros((300, 400), np.uint8)
        g[20:280, 20:380] = agar
    else:
        agar, colony, gash = 213, 190, 255
        g = np.full((300, 400), agar, np.uint8)
    for items in well_shapes(disk_at_33).values():
        for cy, cx, kind in items:
            paint(g, cy, cx, kind, colony)
    gy, gx = ROWS_Y[GASH_WELL[0] - 1], COLS_X[GASH_WELL[1] - 1]
    g[gy + 10 : gy + 24, gx - 20 : gx - 6] = gash
    return g


def make_masks(
    disk_at_33: bool = False, skip: tuple[tuple[int, int], ...] = ()
) -> NDArray[Any]:
    """Instance labels in raster (row, col) order; wells in ``skip`` keep their id
    but are left unpainted (a Cellpose miss).
    """
    m = np.zeros((300, 400), np.int32)
    k = 0
    for key, items in well_shapes(disk_at_33).items():
        for cy, cx, kind in items:
            k += 1
            if key in skip:
                continue
            paint(m, cy, cx, kind, k)
    return m


def _save(tmp_path: Path, name: str, g: NDArray[Any]) -> str:
    path = osp.join(tmp_path, name)
    Image.fromarray(g, "L").save(path)
    return path


@pytest.fixture
def dark_plate_array() -> NDArray[Any]:
    """The dark-field plate as a uint8 array, for tests that add a perturbation."""
    return make_plate("dark")


@pytest.fixture
def backlit_plate_array() -> NDArray[Any]:
    """The backlit plate as a uint8 array, for tests that add a perturbation."""
    return make_plate("backlit")


@pytest.fixture
def save_plate(tmp_path: Path) -> Callable[[str, NDArray[Any]], str]:
    """``save_plate(name, array) -> path``: write a uint8 array as a grayscale PNG."""

    def _saver(name: str, g: NDArray[Any]) -> str:
        return _save(tmp_path, name, g)

    return _saver


@pytest.fixture
def dark_plate_path(tmp_path: Path) -> str:
    """Dark-field plate PNG (bright colonies on mid-gray agar, black surround)."""
    return _save(tmp_path, "dark.png", make_plate("dark"))


@pytest.fixture
def backlit_plate_path(tmp_path: Path) -> str:
    """Backlit plate PNG (dark colonies on a uniformly bright field)."""
    return _save(tmp_path, "backlit.png", make_plate("backlit"))


@pytest.fixture
def backlit_disk_plate_path(tmp_path: Path) -> str:
    """Backlit plate with the (3, 3) colony a radius-15 disk."""
    return _save(tmp_path, "backlit_disk.png", make_plate("backlit", True))


@pytest.fixture
def plate_masks() -> NDArray[Any]:
    """Instance labels for the standard geometry (ids 1..12; M primary 4, secondary 5)."""
    return make_masks()


@pytest.fixture
def plate_masks_missing_disk() -> NDArray[Any]:
    """Labels for the disk variant with the (3, 3) disk (id 11) left undetected."""
    return make_masks(True, skip=(DISK_WELL,))
