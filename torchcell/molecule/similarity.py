# torchcell/molecule/similarity.py
# [[torchcell.molecule.similarity]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/molecule/similarity.py
"""Pairwise similarity matrices over encoder outputs, numpy only.

:func:`tanimoto_matrix` is the generalized (min/max) Tanimoto, which on 0/1 vectors
equals the bit Tanimoto ``|a & b| / |a | b|``; :func:`cosine_matrix` is the cosine
similarity. Both return ``(n, m)`` float64 for ``a`` of shape ``(n, d)`` and ``b`` of
shape ``(m, d)``. A pair whose denominator is zero (two all-zero rows, or a zero-norm
row) is reported as ``0.0``; NaN inputs propagate as NaN.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _as_2d(x: NDArray[Any], name: str) -> NDArray[Any]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, d); got shape {x.shape}")
    return x


def _safe_div(num: NDArray[Any], den: NDArray[Any]) -> NDArray[Any]:
    out = np.zeros_like(num, dtype=np.float64)
    ok = den != 0
    out[ok] = num[ok] / den[ok]
    return out


def tanimoto_matrix(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
    """Generalized Tanimoto ``sum(min(a_i, b_j)) / sum(max(a_i, b_j))`` for every
    row pair; inputs must be non-negative (bit or count fingerprints).

    Uses the identity ``min(x, y) = (x + y - |x - y|) / 2`` so each row of ``a`` is
    compared against all of ``b`` with one ``(m, d)`` temporary.
    """
    a, b = _as_2d(a, "a"), _as_2d(b, "b")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"width mismatch: a {a.shape}, b {b.shape}")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("tanimoto_matrix needs non-negative inputs")
    sa = a.sum(1)
    sb = b.sum(1)
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
    for i in range(a.shape[0]):
        l1 = np.abs(a[i][None, :] - b).sum(1)
        inter = (sa[i] + sb - l1) / 2.0
        union = (sa[i] + sb + l1) / 2.0
        out[i] = _safe_div(inter, union)
    return out


def cosine_matrix(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
    """Cosine similarity ``a_i . b_j / (|a_i| |b_j|)`` for every row pair."""
    a, b = _as_2d(a, "a"), _as_2d(b, "b")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"width mismatch: a {a.shape}, b {b.shape}")
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return _safe_div(a @ b.T, na[:, None] * nb[None, :])
