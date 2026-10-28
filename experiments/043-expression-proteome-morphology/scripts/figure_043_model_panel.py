# experiments/043-expression-proteome-morphology/scripts/figure_043_model_panel.py
# [[experiments.043-expression-proteome-morphology.scripts.figure_043_model_panel]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/figure_043_model_panel
"""The model panel of the 043 setup-and-pretraining figure: the cell model in math.

Three panels, each half a sheet wide (88 mm):

``model``
    The forward pass and the objective of ``torchcell.models.cell_model_043`` as five
    boxed blocks read top to bottom (the 040 thesis sheet's panel d style): i. the
    composite gene token, ii. the cell and the perturbation, iii. the trunk, iv. the mark
    operator with ontology conditioning, v. the heads and the loss. Every equation was
    audited against the code (owner rule 2026-10-27, "function view"); the line map is
    in ``CODE_LINES`` below.
``symbols``
    The symbol table of every symbol drawn in ``model``, with its size.
``conditioning``
    Where the condition enters under name conditioning, ontology conditioning, the
    ontology token injected before the operator (``inject: operator``), and what route A
    and route B change in the input (``RepresentationConfig.route``).

Counts come from ``results/model_math_043_counts.json`` (gene count, mark maximum,
parameter counts), ``results/route_compare_043_v2.json`` (contexts and marks per route),
the config files under ``conf/`` (widths, heads, layers, batch, learning rate, bins,
quantiles), and ``ENCODER_DIMS`` of the model module (the four frozen encoder widths).
Nothing is typed in.
"""

from __future__ import annotations

import os.path as osp
import sys
from collections.abc import Callable
from typing import Any

import yaml
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# imported BEFORE the common module: the model's import chain resets matplotlib's
# rcParams (font.size 16, DejaVu Sans), which would undo the sheet's Arial 6 pt
from torchcell.models.cell_model_043 import ENCODER_DIMS  # noqa: E402

sys.path.insert(0, osp.dirname(__file__))
import figure_043_common as C  # noqa: E402

CONF = osp.join(C.EXPERIMENT_ROOT, C.SLUG, "conf")
SIZE_COLOR = "#555555"

#: equation -> lines of torchcell/models/cell_model_043.py it was checked against
CODE_LINES: dict[str, str] = {
    "i.1 composite x~": "CompositeToken.forward 414-427 (LN + Linear per block, gap row)",
    "i.2 pos": "PositionTerm.forward 447-467",
    "i.3 x = x~ + pos": "CellModel043.context_tokens 732-750",
    "ii.1 H0 = [CLS; x]": "CellTrunk043.forward 539-542",
    "ii.2 mark token p~": "CellModel043.mark_keys 754-767",
    "iii.1 trunk": "CellTrunk043.forward 545-547; mask build_head_mask 470-499",
    "iii.2 H_b = H_c(b), inject": "CellModel043.forward 798-811",
    "iv.1 t_b": "CellModel043.tokens 776-791; OntologyDagEncoder (experiment_ontology.py 572-640)",
    "iv.2 keys p_t": "CellModel043.mark_keys 768-774",
    "iv.3 attention": "MarkOperator043.forward 602-614",
    "iv.4 post-norm": "MarkOperator043.forward 615-618",
    "iv.5 z_S": "CellModel043.forward 814-815",
    "v.1 gene head": "CellModel043.forward 818-830; PerGeneReadout 175-209",
    "v.2 record head": "CellModel043.forward 835-843; GlobalScalarReadout 242-270",
    "v.3 loss": "ReadoutLoss043 868-889; HLGaussLoss 104-139; train_043 combine_loss 2530-2559",
    "v.4 point": "HLGaussLoss.expectation 149-152",
}


def _yaml(*parts: str) -> dict[str, Any]:
    with open(osp.join(CONF, *parts)) as fh:
        return dict(yaml.safe_load(fh))


def _numbers(R: C.Reads) -> dict[str, Any]:
    """Every number the panels print, read from the results and the configs."""
    cfg = _yaml("train_043_default.yaml")
    hl = _yaml("loss", "hlgauss.yaml")
    pin = _yaml("loss", "pinball.yaml")
    m = cfg["model"]
    counts = R.model_counts
    route = R.json("route_compare_043_v2.json")
    hb = hl["hl_gauss"]
    return {
        "N": counts["n_genes"],
        "M": counts["marks_max_joint_9"],
        "n_env": counts["n_envs"],
        "params": counts["params"]["hlgauss"],
        "dims": ENCODER_DIMS,
        "d": m["hidden_dim"],
        "h": m["num_heads"],
        "dk": m["hidden_dim"] // m["num_heads"],
        "L": m["num_layers"],
        "mask_layers": m["mask_layers"],
        "n_graphs": len(m["head_graphs"]),
        "tok": m["type_token_dim"] + m["dataset_token_dim"] + m["env_token_dim"],
        "type_dim": m["type_token_dim"],
        "data_dim": m["dataset_token_dim"],
        "env_dim": m["env_token_dim"],
        "state_dim": m["conditioning"]["state_dim"],
        "ffn": m["operator_ffn_mult"],
        "head_hidden": m["head_hidden"],
        "four": m["fourier_frequencies"],
        "B": cfg["data"]["batch_size"],
        "lr": cfg["optim"]["lr"],
        "K": hb["num_bins"],
        "low": hb["low"],
        "high": hb["high"],
        "sigma_bins": hb["sigma_bins"],
        "nq": pin["num_quantiles"],
        "ctx_A": route["contexts"]["A"],
        "ctx_B": route["contexts"]["B"],
        "caudal_A": route["datasets"]["caudal"]["A"]["marks_median"],
        "caudal_B": route["datasets"]["caudal"]["B"]["marks_median"],
    }


def _n(v: float) -> str:
    """An integer with thousands commas."""
    return f"{int(v):,}"


def _clear(ax: Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _frame(ax: Axes, x0: float, y0: float, x1: float, y1: float, fill: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor=fill,
            edgecolor="black",
            linewidth=0.5,
            zorder=1,
        )
    )


def _arrow(ax: Axes, xy0: tuple[float, float], xy1: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            xy0,
            xy1,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=4,
            linewidth=0.5,
            color="black",
            zorder=3,
        )
    )


#: rewrites every panel string and its text keywords before drawing; ``None`` draws them
#: as written (mathtext). ``figure_043_model_panel_tex`` sets it to a TeX escaper.
TEXT_HOOK: Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]] | None = None


def _t(ax: Axes, x: float, y: float, s: str, **kw: Any) -> None:
    kw.setdefault("fontsize", 6)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    if TEXT_HOOK is not None:
        s, kw = TEXT_HOOK(s, kw)
    ax.text(x, y, s, transform=ax.transAxes, zorder=4, **kw)


def _size(ax: Axes, y: float, s: str) -> Any:
    """A tensor size, right-aligned at the box edge."""
    _t(ax, 0.975, y, s, ha="right", fontsize=5, color=SIZE_COLOR)
    return ax.texts[-1]


def _lift_colliding_sizes(ax: Axes, rows: list[tuple[Any, Any, Any]]) -> None:
    """A size label that overlaps its own equation moves up to its block's header line,
    when that line has room and no other size of the block moved there. Rows are
    ``(header, equation, size)`` texts. The LaTeX panels need it (Computer Modern math
    sets wider than mathtext); a mathtext row that fits is never moved.
    """
    fig = ax.figure.get_figure(root=True)
    fig.draw_without_rendering()
    renderer = fig._get_renderer()
    lifted: set[int] = set()
    for head, eq, size in rows:
        if eq.get_window_extent(renderer).x1 + 2 < size.get_window_extent(renderer).x0:
            continue
        free = head.get_window_extent(renderer).x1 + 4 < (
            size.get_window_extent(renderer).x0
        )
        if id(head) in lifted or not free:
            print(f"  size label {size.get_text()!r} overlaps its equation", flush=True)
            continue
        size.set_y(head.get_position()[1])
        lifted.add(id(head))


# --------------------------------------------------------------------------- #
# Panel: model
# --------------------------------------------------------------------------- #
def draw_model(ax: Axes, R: C.Reads) -> str:
    """The forward pass and the loss, five boxed blocks."""
    n = _numbers(R)
    C.title(ax, "the model: token, cell, operator, heads")
    _clear(ax)
    dims = n["dims"]
    masked = ", ".join(str(i + 1) for i in n["mask_layers"])
    lr = f"{n['lr']:.0e}".replace("e-0", "e-")
    # (header, fill, rows of (equation, size))
    blocks: list[tuple[str, str, list[tuple[str, str]]]] = [
        (
            f"i. the composite gene token, {_n(sum(dims))} frozen dims per gene",
            C.AMBER_FILL,
            [
                (
                    r"$x_{c,i}={\sum}_{q}\,W_q\,\mathrm{LN}_q(e_{q,c,i})+\mathrm{pos}_{c,i}$"
                    r"   (gap $\kappa$: $\Lambda_{q\kappa}$)",
                    f"[C, N, d], d = {n['d']}",
                ),
                (
                    r"$\mathrm{pos}_{c,i}=\nu_{c,i}E_{chr(i)}+W_{\xi}\mathrm{F}(\xi_i)"
                    r"+w\,(\log_2(\eta_{c,i}+1)-1)+\Delta_{\lambda}$",
                    f"e: {' + '.join(_n(v) for v in dims)}",
                ),
            ],
        ),
        (
            "ii. the cell and the perturbation",
            C.SAND_FILL,
            [
                (r"$X_c=[\,h_0;\,x_{c,1};\dots;\,x_{c,N}\,]$", "[C, N+1, d]"),
                (
                    r"$\tilde{p}_t=x_t+\Gamma_{\tau_t}+\Pi_{o_t}$   for each mark $t$ of record $b$",
                    "[B, M, d]",
                ),
            ],
        ),
        (
            "iii. the trunk, one pass per distinct context",
            C.BRICK_FILL,
            [
                (
                    r"$H_c=F_{\theta}(X_c)$"
                    f": {n['L']} post-LN layers, {n['h']} heads; layer {masked}: "
                    r"head $a$ sees graph $k(a)$",
                    "[C, N+1, d]",
                ),
                (
                    r"$H_b=H_{c(b)}$;   inject = operator: "
                    r"$H_b\leftarrow H_b+W_s\,\mathrm{LN}(t_b)$",
                    "[B, N+1, d]",
                ),
            ],
        ),
        (
            "iv. the mark operator, ontology conditioning",
            C.PURPLE_FILL,
            [
                (
                    r"$t_b=\mathrm{DAG}_{\psi}(a_b)$   (name: $t_b=[E_{type};\,E_{s};\,E_{\varepsilon}]$)",
                    f"[B, {n['tok']}]",
                ),
                (
                    r"$p_t=W_p\,\mathrm{LN}(\tilde{p}_t)+[\,j_t\ \mathrm{on\ graph}\,]\,H_{b,j_t}$,"
                    r"   $P_b=(p_t)_{t\leq M_b}$",
                    "[B, M, d]",
                ),
                (
                    r"$H_1=\mathrm{LN}(H_b+\mathrm{MHA}(H_b;\ [k_{\emptyset};P_b],\ [0;P_b]))$",
                    "[B, N+1, d]",
                ),
                (
                    r"$\bar{H}_b=\mathrm{LN}(H_1+\mathrm{FFN}(H_1))$,   $z_S=\mathrm{mean}_t\ p_t$",
                    "[B, N+1, d], [B, d]",
                ),
            ],
        ),
        (
            "v. the heads and the loss",
            C.SAND_FILL,
            [
                (
                    r"$\hat{o}_{b,i}=\mathrm{MLP}_g([\bar{H}_{b,i};\,t_b])+W_g\,t_b$",
                    f"[B_g, N, K], K = {n['K']}",
                ),
                (
                    r"$\hat{o}_b=\mathrm{MLP}_r([\bar{H}_{b,0};\,z_S;\,t_b])+W_r\,t_b$",
                    "[B_r, K]",
                ),
                (
                    r"$L=\mathrm{mean}_s\ \mathrm{mean}_{m_{b,i}=1}\ \mathrm{CE}(\omega(z_{b,i}),"
                    r"\ \mathrm{softmax}\ \hat{o}_{b,i})$",
                    f"bins on [{n['low']:g}, {n['high']:g}], σ = {n['sigma_bins']:g} bin",
                ),
                (
                    r"$\hat{z}={\sum}_u\,\mathrm{softmax}(\hat{o})_u\,\bar{v}_u$"
                    f";   pinball arm: {n['nq']} quantiles",
                    f"B = {n['B']}, lr {lr}",
                ),
            ],
        ),
    ]
    # rows are laid out in points, so the spacing holds at any panel height
    fig = ax.figure.get_figure(root=True)
    h_pt = ax.get_window_extent().height * 72.0 / fig.dpi
    head_pt, eq_pt, pad_pt, gap_pt = 7.6, 9.0, 1.2, 3.6
    need = sum(head_pt + eq_pt * len(r) + 2 * pad_pt for _, _, r in blocks)
    need += gap_pt * (len(blocks) - 1)
    if need > h_pt:
        print(
            f"  model panel needs {need:.0f} pt, has {h_pt:.0f}; rows compressed",
            flush=True,
        )
    s = min(1.0, h_pt / need) / h_pt
    y = 1.0
    left = 0.03
    spans: list[tuple[float, float]] = []
    placed: list[tuple[Any, Any, Any]] = []
    for head, fill, rows_ in blocks:
        top = y
        bottom = top - s * (head_pt + eq_pt * len(rows_) + 2 * pad_pt)
        _frame(ax, 0.0, bottom, 1.0, top, fill)
        cy = top - s * (pad_pt + head_pt / 2)
        _t(ax, 0.012, cy, head, fontweight="bold")
        head_text = ax.texts[-1]
        cy -= s * (head_pt / 2 + eq_pt / 2)
        for eq, size in rows_:
            _t(ax, left, cy, eq)
            eq_text = ax.texts[-1]
            placed.append((head_text, eq_text, _size(ax, cy, size)))
            cy -= s * eq_pt
        spans.append((top, bottom))
        y = bottom - s * gap_pt
    _lift_colliding_sizes(ax, placed)
    for (_, b0), (t1, _) in zip(spans[:-1], spans[1:], strict=True):
        _arrow(ax, (0.5, b0), (0.5, t1))
    return "ready"


# --------------------------------------------------------------------------- #
# Panel: symbols
# --------------------------------------------------------------------------- #
def draw_symbols(ax: Axes, R: C.Reads) -> str:
    """Every symbol of the model panel with its meaning and size."""
    n = _numbers(R)
    C.title(ax, "symbols of the model panel")
    _clear(ax)
    dims = n["dims"]
    rows_: list[tuple[str, str]] = [
        (r"$b,\ B$", f"record; records per batch, {n['B']}"),
        (r"$c,\ C$", "reference context; distinct per batch"),
        (r"$i,\ N$", f"gene; genes of the index, {_n(n['N'])}"),
        (r"$q,\ e_q$", "encoder block; its frozen vector"),
        (r"$d_q$", ", ".join(_n(v) for v in dims) + f" = {_n(sum(dims))}"),
        ("", "FUDT up/down, ProtT5-XL, CaLM"),
        (r"$\kappa,\ \Lambda$", "token gap kind; its learned row"),
        (r"$\lambda,\ \Delta$", "position gap kind; its learned row"),
        (r"$\nu,\ \eta$", "chromosome copy; gene copy"),
        (r"$\xi_i,\ \mathrm{F}$", f"relative midpoint; Fourier, {n['four']} freq."),
        (r"$E_{chr(i)}$", "chromosome row of gene $i$"),
        (r"$d,\ d_k$", f"width {n['d']}; head width {n['dk']}"),
        (r"$h_0$", "learned CLS token, position 0"),
        (r"$X_c,\ H_c$", "trunk input; trunk output"),
        (r"$F_\theta$", f"trunk, {n['L']} layers, {n['h']} heads"),
        (r"$k(a)$", f"graph of head $a$, {n['n_graphs']} graphs"),
        (r"$t,\ M$", f"mark; padded marks, at most {_n(n['M'])}"),
        (r"$M_b,\ j_t$", "marks of record $b$; marked gene"),
        (r"$\Gamma_\tau,\ \Pi_o$", "change-type row; allele-parent row"),
        (r"$\tilde{p}_t,\ p_t$", f"mark token; mark key, width {n['d']}"),
        (r"$k_\emptyset$", "learned null key, value 0"),
        (r"$\mathrm{MHA}$", f"attention, {n['h']} heads (queries; keys, values)"),
        (r"$\bar{H}_b$", "perturbed cell state"),
        (r"$z_S$", "mean mark key (0 without marks)"),
        (r"$a_b$", "record's experiment-ontology annotation"),
        (r"$\mathrm{DAG}_\psi$", f"ontology DAG encoder, state {n['state_dim']}"),
        (
            r"$t_b$",
            f"condition token, {n['tok']} = {n['type_dim']}+{n['data_dim']}+{n['env_dim']}",
        ),
        (r"$E_{type,s,\varepsilon}$", "name tokens: type, dataset, env."),
        (r"$W_s$", "inject map, token to width $d$"),
        (r"$W_g,\ W_r$", f"bias maps, token to {n['K']} logits"),
        (r"$K,\ \bar{v}_u$", f"HL-Gauss bins, {n['K']}; bin centers"),
        (r"$\omega(z)$", "Gaussian histogram target of $z$"),
        (r"$z,\ m$", "standardized target; measured mask"),
        (r"$\mathrm{mean}_s$", "equal weight per dataset in the batch"),
    ]
    half = (len(rows_) + 1) // 2
    dy = 1.0 / half
    for col, chunk in enumerate((rows_[:half], rows_[half:])):
        x0 = 0.0 if col == 0 else 0.515
        for r, (sym, meaning) in enumerate(chunk):
            yy = 1.0 - (r + 0.5) * dy
            _t(ax, x0, yy, sym)
            _t(ax, x0 + 0.13, yy, meaning, fontsize=5.5)
    ax.plot(
        [0.505, 0.505],
        [0.0, 1.0],
        color="#BBBBBB",
        linewidth=0.4,
        transform=ax.transAxes,
    )
    return "ready"


# --------------------------------------------------------------------------- #
# Panel: conditioning
# --------------------------------------------------------------------------- #
STACK_Y = (0.81, 0.59, 0.37)
STACK_H = 0.11


def _mini(ax: Axes, x0: float, w: float) -> list[tuple[float, float, float, float]]:
    """Trunk, operator, heads as a vertical stack with arrows; returns the extents."""
    labels = [r"trunk $F_\theta$", "operator", "heads"]
    fills = [C.BRICK_FILL, C.PURPLE_FILL, C.SAND_FILL]
    out = []
    for yc, lab, fill in zip(STACK_Y, labels, fills, strict=True):
        _frame(ax, x0, yc - STACK_H / 2, x0 + w, yc + STACK_H / 2, fill)
        _t(ax, x0 + w / 2, yc, lab, ha="center")
        out.append((x0, yc - STACK_H / 2, x0 + w, yc + STACK_H / 2))
    for a, b in zip(out[:-1], out[1:], strict=True):
        _arrow(ax, (x0 + w / 2, a[1]), (x0 + w / 2, b[3]))
    return out


def draw_conditioning(ax: Axes, R: C.Reads) -> str:
    """Where the condition enters, and what the route changes in the input."""
    n = _numbers(R)
    route = R.json("route_compare_043_v2.json")
    C.title(ax, "where the condition and the isolate enter")
    _clear(ax)
    col_w = [0.205, 0.205, 0.205, 0.29]
    gap = 0.025
    xs = [sum(col_w[:k]) + k * gap for k in range(4)]
    bw = 0.125
    heads = ["name", "ontology", "inject_op", "route A vs B"]
    notes = [
        f"$[E_{{type}};E_s;E_\\varepsilon]$\n({n['tok']} dims) and a\nbias table per\ndataset; heads only",
        "$t_b=\\mathrm{DAG}_\\psi(a_b)$\nand bias maps;\nno dataset id\nin forward",
        "ontology, plus\n$W_s\\mathrm{LN}(t_b)$ added\nto every token\nbefore the operator",
    ]
    for k in range(3):
        x0 = xs[k]
        _t(ax, x0, 0.95, heads[k], fontweight="bold")
        boxes = _mini(ax, x0, bw)
        tx0 = x0 + bw + 0.02
        tw, th = col_w[k] - bw - 0.02, 0.09
        # heads: the token always enters there
        hy = STACK_Y[2]
        _frame(ax, tx0, hy - th / 2, tx0 + tw, hy + th / 2, C.AMBER_FILL)
        _t(ax, tx0 + tw / 2, hy, r"$t_b$", ha="center")
        _arrow(ax, (tx0, hy), (boxes[2][2], hy))
        if k == 2:
            # inject_op: the same token, mapped, onto the trunk -> operator edge
            iy = (STACK_Y[0] + STACK_Y[1]) / 2
            _frame(ax, tx0, iy - th / 2, tx0 + tw, iy + th / 2, C.AMBER_FILL)
            _t(ax, tx0 + tw / 2, iy, r"$t_b$", ha="center")
            _arrow(ax, (tx0, iy), (x0 + bw / 2 + 0.004, iy))
            _t(ax, x0 + bw / 2 + 0.012, iy + 0.035, "+", fontsize=6)
        _t(ax, x0, 0.25, notes[k], va="top", fontsize=5.5, linespacing=1.25)
    x0, w = xs[3], col_w[3]
    _t(ax, x0, 0.95, heads[3], fontweight="bold")
    rows_ = [
        (
            "A",
            f"context: the isolate,\n{_n(n['ctx_A'])} contexts, own keys",
            "marks: HO deletion,\naccessory ORFs",
        ),
        (
            "B",
            f"context: S288C ({_n(n['ctx_B'])}),\nisolate keys on variant\nand LoF genes",
            "marks: every\nnon-reference gene",
        ),
    ]
    for (r, ctx, mk), yc in zip(rows_, (0.73, 0.43), strict=True):
        _frame(ax, x0, yc - 0.135, x0 + w, yc + 0.135, "white")
        _t(ax, x0 + 0.012, yc + 0.095, r, fontweight="bold")
        _t(ax, x0 + 0.045, yc + 0.045, ctx, fontsize=5.5, linespacing=1.15)
        _t(ax, x0 + 0.045, yc - 0.075, mk, fontsize=5.5, linespacing=1.15)
    same = "identical" if route["Y_identical"] else "differ"
    _t(
        ax,
        x0,
        0.25,
        f"Caudal median marks\nper record: A {n['caudal_A']:g}, B {_n(n['caudal_B'])};\n"
        f"targets {same} (prep_v4)",
        va="top",
        fontsize=5.5,
        linespacing=1.25,
    )
    return "ready"


PANELS: dict[str, Callable[[Axes, C.Reads], str]] = {
    "model": draw_model,
    "symbols": draw_symbols,
    "conditioning": draw_conditioning,
}


def main() -> None:
    """Draw the three panels on a demo sheet and print the PNG path."""
    out = C.demo(PANELS, "figure_043_model_panel", 70)
    print(out["png"])


if __name__ == "__main__":
    main()
