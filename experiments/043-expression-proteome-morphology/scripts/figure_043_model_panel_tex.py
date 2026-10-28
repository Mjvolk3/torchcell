# experiments/043-expression-proteome-morphology/scripts/figure_043_model_panel_tex.py
# [[experiments.043-expression-proteome-morphology.scripts.figure_043_model_panel_tex]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/figure_043_model_panel_tex
r"""The three model panels of ``figure_043_model_panel`` typeset with real LaTeX.

Owner, 2026-10-28: "i am liking these but would prefer to have math in latex for
clarity i think. try to make a duplicate with this". The panels are the SAME drawing
functions (same equations, numbers, layout and ``CODE_LINES`` audit); this module only
runs them with ``text.usetex`` on and every string passed through ``tex_safe``, which
escapes the text outside ``$...$`` for TeX, turns runs of spaces into ``\quad``, and
writes a bold header as ``\textbf``. The math is Computer Modern, the body Helvetica
(``helvet``, sans by default), 6 pt.

GilaHyper has ``latex`` and ``dvips`` but no ``dvipng``, so Agg cannot draw TeX text
(measuring it works). A sheet holding these panels is saved as PDF, which reads the DVI
itself, and rasterized to PNG with ``pdftoppm``: ``figure_043_common.save(..., tex=True)``.
``PREAMBLE`` must be in ``text.latex.preamble`` when the sheet is saved, not only while
the panels are drawn, because TeX is run again at draw time.

    python experiments/043-expression-proteome-morphology/scripts/figure_043_model_panel_tex.py
    -> $ASSET_IMAGES_DIR/043-expression-proteome-morphology/figure_043_model_panel_tex{_<stamp>.png,.pdf}
"""

from __future__ import annotations

import os.path as osp
import re
import sys
from collections.abc import Callable
from typing import Any

import matplotlib
from matplotlib.axes import Axes

sys.path.insert(0, osp.dirname(__file__))
import figure_043_common as C  # noqa: E402
import figure_043_model_panel as M  # noqa: E402

PREAMBLE = (
    r"\usepackage{amsmath,amssymb}"
    r"\usepackage{helvet}"
    r"\renewcommand{\familydefault}{\sfdefault}"
    # Computer Modern math sets wider than the Arial mathtext the rows were laid out
    # for; tighter binary-operator and relation spacing keeps each row inside its box
    r"\medmuskip=2mu plus 1mu minus 1mu"
    r"\thickmuskip=3mu plus 2mu"
)
#: in force while the sheet is SAVED: matplotlib wraps every usetex string in the font
#: command of ``font.family`` read at draw time (``\rmfamily`` for a named family such
#: as Arial, which would undo ``\familydefault``), so the family is the generic sans.
#: Helvetica picks ``helvet`` for TeX; Arial is what non-TeX text (the panel letters)
#: resolves to, since GilaHyper has no system Helvetica.
SAVE_RC: dict[str, Any] = {
    "text.latex.preamble": PREAMBLE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
}
#: in force while the panels are DRAWN
TEX_RC: dict[str, Any] = {**SAVE_RC, "text.usetex": True}
#: text-mode characters TeX reserves, and the unicode the mathtext panels print
TEXT_ESCAPES = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "σ": r"$\sigma$"}


def _escape_text(s: str) -> str:
    s = "".join(TEXT_ESCAPES.get(ch, ch) for ch in s)
    return re.sub(r" {2,}", lambda _: r"\quad ", s)


def tex_safe(s: str, kw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    r"""A panel string made TeX-safe: text outside ``$...$`` escaped, math untouched,
    ``fontweight="bold"`` written as ``\textbf`` per line (usetex ignores the weight).
    """
    parts = s.split("$")
    assert len(parts) % 2 == 1, f"unbalanced $ in {s!r}"
    out = "$".join(p if k % 2 else _escape_text(p) for k, p in enumerate(parts))
    if kw.get("fontweight") == "bold":
        kw = {k: v for k, v in kw.items() if k != "fontweight"}
        out = "\n".join(r"\textbf{" + line + "}" for line in out.split("\n"))
    return out, kw


def _tex(fn: Callable[[Axes, C.Reads], str]) -> Callable[[Axes, C.Reads], str]:
    def draw(ax: Axes, R: C.Reads) -> str:
        with matplotlib.rc_context(TEX_RC):
            M.TEXT_HOOK = tex_safe
            try:
                return fn(ax, R)
            finally:
                M.TEXT_HOOK = None

    draw.__doc__ = fn.__doc__
    return draw


PANELS: dict[str, Callable[[Axes, C.Reads], str]] = {
    k: _tex(fn) for k, fn in M.PANELS.items()
}


def main() -> None:
    """Draw the three panels on a demo sheet (PDF and PNG) and print the PNG path."""
    R = C.Reads()
    keys = list(PANELS)
    # new_sheet applies the sheet rc (font.family Arial), so it runs before SAVE_RC
    fig = C.new_sheet(70.0 * ((len(keys) + 1) // 2))
    axes = C.rows(fig, (len(keys) + 1) // 2, 2)
    with matplotlib.rc_context(SAVE_RC):
        statuses = [PANELS[k](ax, R) for ax, k in zip(axes, keys, strict=False)]
        for ax in axes[len(keys) :]:
            ax.axis("off")
        C.letters_status(fig, axes[: len(keys)], statuses)
        C.status_key(fig)
        out = C.save(fig, "figure_043_model_panel_tex", tex=True)
    print(out["png"])


if __name__ == "__main__":
    main()
