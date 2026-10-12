"""Generate the sketch figures of notes-tex/030-scaling-plots (no data).

Writes, next to itself:
  030-scaling-plots-surface.drawio      S1-S4: loss against D per size, against N per subset,
                                        the 5 x 4 heatmap, the fitted forms on L - E
  030-scaling-plots-compute.drawio      C1-C4: learning curves, the FLOPs envelope, GPU hours,
                                        cost per record and pace against size
  030-scaling-plots-diagnostics.drawio  D1-D6: bootstrap, held-out, E vs floor, seed spread,
                                        the lr check, per-screen Pearson
  030-scaling-plots-pair.drawio         P1-P2: the 030 minus 025 gap, paired learning curves

Panel titles come from experiments/030-solid-growth-multi/results/scaling_plot_catalog_030.json
(scaling_plot_catalog_030.py), so a panel and its row of the catalog table carry the same
name. Every curve is drawn by hand to show the shape the plot would have; nothing is
measured. The shape helpers are those of 030-scaling-study.gen.py, imported from it.

Export with the recipe in [[drawio-headless-export-gilahyper]] (input file FIRST).
"""

import importlib.util
import json
import os.path as osp

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(osp.dirname(osp.dirname(HERE)))
CATALOG_JSON = osp.join(
    REPO, "experiments/030-solid-growth-multi/results/scaling_plot_catalog_030.json"
)

_spec = importlib.util.spec_from_file_location(
    "scaling_study_gen", osp.join(HERE, "030-scaling-study.gen.py")
)
assert _spec is not None and _spec.loader is not None
_g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g)
Canvas, panel, xticks = _g.Canvas, _g.panel, _g.xticks
OR, ORF, RD, PU, YE, BL, GR, GRF, BK, W = (
    _g.OR, _g.ORF, _g.RD, _g.PU, _g.YE, _g.BL, _g.GR, _g.GRF, _g.BK, _g.W,
)
SIZE_COLORS = [OR, RD, PU, YE, BL]  # 45x2 .. 180x8
SUBSET_COLORS = [OR, RD, PU, YE]  # 1/64 .. full
SIZES = ["45x2", "63x3", "90x4", "126x6", "180x8"]
SUBSETS = ["1/64", "1/16", "1/4", "full"]
SIZE_NOTE = "sizes 45x2 to 180x8 in orange, red, purple, yellow, blue"
SUBSET_NOTE = "subsets 1/64 to full in orange, red, purple, yellow"
TOP = 30  # the sketch strip sits above the panel letters


def titles() -> dict[str, str]:
    with open(CATALOG_JSON) as f:
        return {p["id"]: f"{p['id']}  {p['title']}" for p in json.load(f)}


def shade(t: float) -> str:
    """Orange fill between the pale fill and the line color, t in [0, 1]."""
    a, b = (0xFF, 0xE6, 0xCC), (0xD7, 0x9B, 0x00)
    return "#" + "".join(f"{int(round(x + (y - x) * t)):02X}" for x, y in zip(a, b))


def legend(c: Canvas, x, y, items, w=110, square=False):
    """Marks and labels in a column; the text is clamped to w so no label can widen
    the export past its panel."""
    for i, (color, label) in enumerate(items):
        c.dot(x + 4, y + 6 + 11 * i, color, r=2.2, square=square)
        c.text(x + 10, y + 11 * i, w - 10, 11, label)


def strip(c: Canvas, s: str) -> None:
    c.text(0, 0, 560, 12, s, italic=True, color=GR)


def grid2x2(c: Canvas, ids, draw, ph=196):
    pw = (W - 12) / 2
    X = [0, pw + 12]
    cells = []
    for k, pid in enumerate(ids):
        cells.append((pid, X[k % 2], TOP + (k // 2) * (ph + 18), pw, ph))
    for pid, px, py, pw_, ph_ in cells:
        draw(pid, px, py, pw_, ph_)
    for k, (pid, px, py, _, _) in enumerate(cells):
        c.letter(px, py - 14, "abcdefgh"[k])
    return TOP + 2 * (ph + 18)


# ------------------------------------------------------------------- surface
def fig_surface(T: dict[str, str]) -> str:
    c = Canvas("030-scaling-plots-surface")
    strip(c, "sketch, no data: the loss surface L(N, D) read both ways, and the fitted forms")

    def draw(pid, px, py, pw, ph):
        if pid == "S1":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["S1"], "D, training records (log)",
                                   "val MSE, label units (log)",
                                   note=f"one curve per size, {SIZE_NOTE}; parallel curves = one beta, a fan = beta depends on N; dashed = replicate floor",
                                   note_lines=3)
            xs = xticks(c, ax, ay, aw, ah, SUBSETS)
            for i, col in enumerate(SIZE_COLORS):
                off = 30 - 6 * i
                ys = [ay + off + 6, ay + off + 22, ay + off + 36, ay + off + 46]
                c.line(list(zip(xs, ys)), col, sw=1.2, curved=True)
                for xx, yy in zip(xs, ys):
                    c.dot(xx, yy, col, r=2.0)
            c.line([(ax + 2, ay + ah - 8), (ax + aw - 2, ay + ah - 8)], GR, sw=0.75, dashed=True)
        elif pid == "S2":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["S2"], "N, non-embedding parameters (log)",
                                   "val MSE, label units (log)",
                                   note=f"one curve per subset, {SUBSET_NOTE}; each flattens at its own plateau E + B D^-beta (dashed)",
                                   note_lines=3)
            xs = xticks(c, ax, ay, aw, ah, SIZES)
            for i, col in enumerate(SUBSET_COLORS):
                base = ay + 10 + 16 * (3 - i)
                drop = 8 + 8 * i
                ys = [base, base + drop * 0.55, base + drop * 0.8, base + drop * 0.93, base + drop]
                c.line(list(zip(xs, ys)), col, sw=1.2, curved=True)
                c.line([(xs[2], ys[-1] + 3), (ax + aw - 2, ys[-1] + 3)], col, sw=0.6, dashed=True)
                for xx, yy in zip(xs, ys):
                    c.dot(xx, yy, col, r=2.0)
        elif pid == "S3":
            c.rect(px, py, pw, ph, fill="#FFFFFF", stroke=GR, sw=0.5)
            c.text(px + 2, py + 1, pw - 4, 24, T["S3"], bold=True)
            cw, chh = 46, 22
            gx, gy = px + 64, py + 36
            for i, size in enumerate(SIZES):
                c.text(px + 6, gy + i * chh + 5, 56, 11, size, align="right")
                for j in range(4):
                    t = 1.0 - (0.18 * i + 0.22 * j) / (0.18 * 4 + 0.22 * 3)
                    c.rect(gx + j * cw, gy + i * chh, cw, chh, fill=shade(t), stroke="#FFFFFF", sw=1.0)
            for j, sub in enumerate(SUBSETS):
                c.text(gx + j * cw, gy - 14, cw, 11, sub, align="center")
            c.text(gx, gy + 5 * chh + 3, 4 * cw, 11, "D, four subsets", align="center")
            c.text(px + 6, gy + 5 * chh + 3, 56, 11, "N, five sizes", align="right")
            bx, by = gx + 4 * cw + 14, gy
            for k in range(10):
                c.rect(bx, by + k * (5 * chh / 10), 10, 5 * chh / 10, fill=shade(1 - k / 9), stroke="#FFFFFF", sw=0.5)
            c.text(bx + 14, by - 2, 86, 22, "high loss\n(matched read)")
            c.text(bx + 14, by + 5 * chh - 12, 86, 11, "low loss")
            c.text(px + 4, py + ph - 30, pw - 8, 28,
                   "the steeper gradient at a cell, down a column or along a row, is where the next unit of effort goes from there; a second heatmap carries the selected read")
        elif pid == "S4":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["S4"], "D or N (log)",
                                   "val MSE minus fitted E (log)",
                                   note="a straight line on L - E is the floored form holding (orange, fit on x1 to x3; band = 300 bootstraps); dashed purple = pure power law; ring = held-out point",
                                   note_lines=3)
            xs = xticks(c, ax, ay, aw, ah, ["x1", "x2", "x3", "x4 (held out)"])
            c.line([(xs[0] - 8, ay + 18), (xs[3] + 8, ay + ah - 36)], ORF, sw=14)
            c.line([(xs[0] - 8, ay + 22), (xs[3] + 8, ay + ah - 30)], OR, sw=1.2)
            ys = [ay + 24, ay + 42, ay + 60, ay + ah - 30]
            for xx, yy in zip(xs[:3], ys[:3]):
                c.dot(xx, yy, OR)
            c.dot(xs[3], ys[3] + 4, OR)
            c.dot(xs[3], ys[3] + 4, BK, r=5.5, fill="none", sw=0.75)
            c.line([(xs[0] - 8, ay + 28), (xs[3] + 8, ay + ah - 52)], PU, sw=1.0, dashed=True)

    h = grid2x2(c, ["S1", "S2", "S3", "S4"], draw)
    return c.write(h)


# ------------------------------------------------------------------- compute
def fig_compute(T: dict[str, str]) -> str:
    c = Canvas("030-scaling-plots-compute")
    strip(c, "sketch, no data: what the compute bookkeeping lets the grid show")

    def trajectories(ax, ay, aw, ah, x0s, x1s):
        ends = []
        for i, col in enumerate(SIZE_COLORS):
            x0, x1 = x0s[i], x1s[i]
            y0 = ay + 8
            y1 = ay + 16 + 14 * i  # larger sizes end lower: less loss for more compute
            c.line([(x0, y0), (x0 + (x1 - x0) * 0.3, y1 - 5), (x1, y1)], col, sw=1.1, curved=True)
            ends.append((x1, y1))
        return ends

    def draw(pid, px, py, pw, ph):
        if pid == "C1":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["C1"], "records seen (log)",
                                   "val MSE, label units (log)",
                                   note=f"one panel per subset in the real figure (here 1/4); {SIZE_NOTE}; dot = selected epoch; the vertical line is B",
                                   note_lines=3)
            xb = ax + aw * 0.52
            c.line([(xb, ay + 2), (xb, ay + ah - 2)], RD, sw=0.9, dashed=True)
            c.text(xb + 3, ay + 2, 40, 11, "B", color=RD)
            ends = trajectories(ax, ay, aw, ah, [ax + 8] * 5, [ax + aw - 10] * 5)
            for i, (x1, y1) in enumerate(ends):
                c.dot(x1 - 10 - 18 * i, y1 - 1 - 1.5 * i, SIZE_COLORS[i], r=2.0)
        elif pid == "C2":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["C2"], "cumulative training FLOPs (log)",
                                   "val MSE, label units (log)",
                                   note=f"trajectories by size, {SIZE_NOTE}; the dashed envelope is the compute-efficient frontier, one budget line of an IsoFLOP sweep",
                                   note_lines=3)
            ends = trajectories(ax, ay, aw, ah, [ax + 6 + 24 * i for i in range(5)], [ax + 70 + 38 * i for i in range(5)])
            c.line([(ax + 6, ay + 8)] + [(x, y + 3) for x, y in ends], BK, sw=1.0, dashed=True, curved=True)
            c.text(ax + aw - 70, ay + ah - 26, 66, 11, "envelope", align="right")
        elif pid == "C3":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["C3"], "GPU hours (log)",
                                   "val MSE, label units (log)",
                                   note="the same envelope in queue time; the horizontal offset of a size between C2 and C3 is its utilization",
                                   note_lines=2)
            ends = trajectories(ax, ay, aw, ah, [ax + 6 + 12 * i for i in range(5)], [ax + 80 + 36 * i for i in range(5)])
            c.line([(ax + 6, ay + 8)] + [(x, y + 3) for x, y in ends], BK, sw=1.0, dashed=True, curved=True)
        elif pid == "C4":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["C4"], "N, non-embedding parameters (log)",
                                   "FLOPs per record; records/s",
                                   note="bars = FLOPs per record with the min-to-max whisker over the five measured steps; purple = pace, solid if it follows the FLOPs (GPU-bound), dashed if flat (loader-bound)",
                                   note_lines=4)
            xs = xticks(c, ax, ay, aw, ah, SIZES)
            for i, xx in enumerate(xs):
                h = 8 + 13 * i
                c.rect(xx - 9, ay + ah - 4 - h, 18, h, fill=ORF, stroke=OR, sw=0.6)
                c.line([(xx, ay + ah - 4 - h - 4), (xx, ay + ah - 4 - h + 4)], OR, sw=1.0)
            c.line(list(zip(xs, [ay + 14, ay + 20, ay + 30, ay + 42, ay + 56])), PU, sw=1.2, curved=True)
            c.line(list(zip(xs, [ay + 16] * 5)), PU, sw=1.0, dashed=True)

    h = grid2x2(c, ["C1", "C2", "C3", "C4"], draw)
    return c.write(h)


# --------------------------------------------------------------- diagnostics
def fig_diagnostics(T: dict[str, str]) -> str:
    c = Canvas("030-scaling-plots-diagnostics")
    strip(c, "sketch, no data: the diagnostics that say how far each exponent can be trusted")
    pw = (W - 24) / 3
    ph = 172
    X = [0, pw + 12, 2 * (pw + 12)]
    cells = [("D1", X[0], TOP), ("D2", X[1], TOP), ("D3", X[2], TOP),
             ("D4", X[0], TOP + ph + 18), ("D5", X[1], TOP + ph + 18), ("D6", X[2], TOP + ph + 18)]
    for pid, px, py in cells:
        if pid == "D1":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["D1"], "exponent", "resamples",
                                   note="300 resamples; the 5 to 95% interval shaded", note_lines=2)
            heights = [3, 8, 18, 32, 44, 38, 24, 11, 5]
            bw = (aw - 16) / len(heights)
            for i, hgt in enumerate(heights):
                fill = ORF if 2 <= i <= 6 else "#FFFFFF"
                c.rect(ax + 8 + i * bw, ay + ah - 4 - hgt, bw, hgt, fill=fill, stroke=OR, sw=0.6)
            c.text(ax + 4, ay + 3, 100, 11, "CI90 shaded")
        elif pid == "D2":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["D2"], "predicted at the held-out point", "observed",
                                   note="one marker per fit; on the diagonal = the fit extrapolated one step", note_lines=2)
            c.line([(ax + 6, ay + ah - 6), (ax + aw - 6, ay + 6)], GR, sw=0.75, dashed=True)
            pts = [(0.25, 0.27, OR), (0.45, 0.44, OR), (0.6, 0.52, PU), (0.7, 0.84, PU), (0.8, 0.79, OR)]
            for fx, fy, col in pts:
                c.dot(ax + 6 + (aw - 12) * fx, ay + ah - 6 - (ah - 12) * fy, col, r=2.4, fill="#FFFFFF" if col == PU else None)
            legend(c, ax + 6, ay + 3, [(OR, "power"), (PU, "floored")], w=70)
        elif pid == "D3":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["D3"], "screen", "loss, label units",
                                   note="band = floor bounds; top tick = predicting the screen mean; dot = fitted E with CI90", note_lines=3)
            xs = xticks(c, ax, ay, aw, ah, ["2018", "2020", "pooled"])
            for i, xx in enumerate(xs):
                c.line([(xx - 16, ay + 16 + 3 * i), (xx + 16, ay + 16 + 3 * i)], BK, sw=1.0)
                b0, b1 = ay + ah - 28 - 5 * i, ay + ah - 14 - 2 * i
                c.rect(xx - 16, b0, 32, b1 - b0, fill=GRF, stroke=GR, sw=0.5)
                ey = b0 - 10 - 5 * i
                c.line([(xx, ey - 8), (xx, ey + 8)], OR, sw=1.0)
                c.dot(xx, ey, OR)
        elif pid == "D4":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["D4"], "(size, subset) cell", "val MSE, label units",
                                   note="three seeds per replicated cell; the arrow is the step to the next cell along D", note_lines=2)
            xs = xticks(c, ax, ay, aw, ah, ["45x2 1/64", "45x2 1/16", "180x8 1/64", "180x8 1/16"])
            ys = [ay + 20, ay + 40, ay + 32, ay + 56]
            for xx, yy in zip(xs, ys):
                for k in (-1, 0, 1):
                    c.dot(xx + 4 * k, yy + 3 * abs(k), OR, r=2.0)
            c.line([(xs[0] + 12, ys[0] + 2), (xs[1] - 12, ys[1] - 2)], GR, sw=0.9, arrow=True)
            c.line([(xs[2] + 12, ys[2] + 2), (xs[3] - 12, ys[3] - 2)], GR, sw=0.9, arrow=True)
        elif pid == "D5":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["D5"], "epoch", "val MSE, label units (log)",
                                   note="45x2 on 1/4 only: lr 1e-4 purple, 2.5e-4 orange, 1e-3 red; says nothing about 180x8 at batch 256", note_lines=3)
            xs_ = [ax + 8 + (aw - 16) * k / 4 for k in range(5)]
            c.line(list(zip(xs_, [ay + 8, ay + 20, ay + 30, ay + 38, ay + 44])), PU, sw=1.1, curved=True)
            c.line(list(zip(xs_, [ay + 8, ay + 30, ay + 46, ay + 56, ay + 62])), OR, sw=1.1, curved=True)
            c.line(list(zip(xs_, [ay + 8, ay + 34, ay + 26, ay + 40, ay + 34])), RD, sw=1.1)
        elif pid == "D6":
            ax, ay, aw, ah = panel(c, px, py, pw, ph, T["D6"], "D, training records (log)", "val Pearson",
                                   note="pooled (black) carries the 0.35 screen-offset floor (dashed); 2018 orange and 2020 purple per token do not", note_lines=3)
            xs = xticks(c, ax, ay, aw, ah, SUBSETS)
            fl = ay + ah - 30
            c.line([(ax + 2, fl), (ax + aw - 2, fl)], GR, sw=0.75, dashed=True)
            c.text(ax + 4, fl - 17, 80, 11, "0.35 floor", color=GR)
            for col, ys in ((BK, [fl - 4, fl - 14, fl - 22, fl - 28]),
                            (OR, [fl + 14, fl + 2, fl - 8, fl - 14]),
                            (PU, [fl + 22, fl + 14, fl + 6, fl + 2])):
                c.line(list(zip(xs, ys)), col, sw=1.1)
                for xx, yy in zip(xs, ys):
                    c.dot(xx, yy, col, r=2.0)
    for k, (pid, px, py) in enumerate(cells):
        c.letter(px, py - 14, "abcdef"[k])
    return c.write(TOP + 2 * (ph + 18))


# ---------------------------------------------------------------------- pair
def fig_pair(T: dict[str, str]) -> str:
    c = Canvas("030-scaling-plots-pair")
    strip(c, "sketch, no data: the 025 twin against the 030 arm at 180x8")
    pw = (W - 12) / 2
    ph = 190
    ax, ay, aw, ah = panel(c, 0, TOP, pw, ph, T["P1"], "D, training records (log)",
                           "030 minus 025 pooled val MSE",
                           note="seed mean and sd where both sides have three seeds; the full point is two unconverged ten-epoch runs, hollow",
                           note_lines=3)
    xs = xticks(c, ax, ay, aw, ah, SUBSETS)
    zero = ay + ah * 0.4
    c.line([(ax + 2, zero), (ax + aw - 2, zero)], GR, sw=0.75, dashed=True)
    c.text(ax + aw - 40, zero - 16, 38, 11, "0", color=GR, align="right")
    for i, (xx, yy) in enumerate(zip(xs, [zero + 30, zero + 24, zero + 16, zero + 10])):
        if i < 2:
            c.line([(xx, yy - 10), (xx, yy + 10)], OR, sw=1.0)
        c.dot(xx, yy, OR, fill="#FFFFFF" if i == 3 else None)
    c.text(ax + 4, ay + 4, 170, 24, "below 0: every stored measurement and the\ntoken lower the loss at that D")
    ax, ay, aw, ah = panel(c, pw + 12, TOP, pw, ph, T["P2"], "records seen (log)",
                           "val MSE, label units (log)",
                           note="one panel per subset in the real figure (here 1/16); 030 orange, 025 purple dashed; B marked; a gap that holds past B is supervision, one that closes is a faster start",
                           note_lines=3)
    xb = ax + aw * 0.6
    c.line([(xb, ay + 2), (xb, ay + ah - 2)], RD, sw=0.9, dashed=True)
    c.text(xb + 3, ay + 2, 40, 11, "B", color=RD)
    c.line([(ax + 8, ay + 10), (ax + 60, ay + 56), (ax + aw - 10, ay + ah - 26)], OR, sw=1.2, curved=True)
    c.line([(ax + 8, ay + 10), (ax + 80, ay + 46), (ax + aw - 10, ay + ah - 40)], PU, sw=1.2, curved=True, dashed=True)
    c.letter(0, TOP - 14, "a")
    c.letter(pw + 12, TOP - 14, "b")
    return c.write(TOP + ph + 6)


def main() -> None:
    T = titles()
    for path in (fig_surface(T), fig_compute(T), fig_diagnostics(T), fig_pair(T)):
        print("wrote", path)


if __name__ == "__main__":
    main()
