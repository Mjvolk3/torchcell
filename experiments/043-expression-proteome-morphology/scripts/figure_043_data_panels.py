# experiments/043-expression-proteome-morphology/scripts/figure_043_data_panels.py
# [[experiments.043-expression-proteome-morphology.scripts.figure_043_data_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/figure_043_data_panels
"""The data-side panels of the 043 setup sheet: what was read, how it was split, where
each dataset sits on the conditioning ontology, what r each dataset can attain against
itself, and (a placeholder) the target distributions.

Every number comes from a file under ``results/``:

* ``records``: ``build_043_store_census_v2.json`` (``n_records_by_dataset_index``,
  ``first_sample_by_dataset[*].label_name``) and ``global_split_v2_summary.json``
  (``counts[*]``: the split unit, the units per split, and the role of a dataset the
  trainer does not fit: probe, split B test, excluded).
* ``split``: ``global_split_v2_summary.json`` (``record_counts``, ``counts[*].units_*``
  and ``unit``, ``n_genes``, ``n_isolates``, ``n_segregant_records``,
  ``n_promoter_80mers``, ``fold_sizes``).
* ``ontology``: ``ontology_annotation_043_v2.json`` (``datasets[*].profiles[*].nodes``
  for each dataset's phenotype class, ``vocabulary.nodes`` for the class parents,
  ``datasets[*].n_profiles``, ``vocabulary_size``, ``vocabulary_depth``,
  ``n_distinct_annotations``).
* ``ceilings``: ``within_dataset_ceilings.json`` (Kemmeren test-retest, Jackson
  same-genotype construction pairs, Boocock and N'Guessan nearest-genotype deviation r,
  the Kemmeren x Hu bound). Datasets the file does not measure are written as such.
* ``targets``: a placeholder; ``data_shapes_043.json`` was run with
  ``per_dataset_sample = null`` and holds ranges, not distributions.

    python experiments/043-expression-proteome-morphology/scripts/figure_043_data_panels.py
    -> $ASSET_IMAGES_DIR/043-expression-proteome-morphology/figure_043_data_panels_<stamp>.png
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
import sys
from collections.abc import Callable
from fractions import Fraction
from math import comb
from typing import Any

import numpy as np
import yaml
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

sys.path.insert(0, osp.dirname(__file__))

import figure_043_common as C  # noqa: E402

from torchcell.utils import PLOT_PALETTE, PLOT_PALETTE_FILL  # noqa: E402

#: the store's dataset class behind each of the twelve heads of ``C.DATASETS``
HEAD_CLASS: dict[str, str] = {
    "kemmeren": "MicroarrayKemmeren2014Dataset",
    "messner": "ProteomeMessner2023Dataset",
    "zelezniak": "ProteomeZelezniak2018Dataset",
    "caudal": "CaudalPanTranscriptome2024Dataset",
    "muenzner": "ProteomeMuenzner2024Dataset",
    "turnover": "ProteinTurnoverMuenzner2024Dataset",
    "boocock": "Boocock2025Dataset",
    "nguessan": "Nguessan2025Dataset",
    "vaishnav": "PooledPromoterExpressionVaishnav2022Dataset",
    "hu": "MicroarrayScoreHu2007Dataset",
    "nadal": "NadalRibellesPerturbSeq2025Dataset",
    "doubles": "DmMicroarraySameith2015Dataset",
}
#: datasets the store holds that are not one of the twelve validation heads
EXTRA: list[tuple[str, str, str]] = [
    ("SmMicroarraySameith2015Dataset", "Sameith 2015 singles", "mRNA"),
    ("TfDeletionPseudobulkJackson2020Dataset", "Jackson 2020", "mRNA"),
    ("ScmdOhya2005Dataset", "Ohya 2005", "morphology"),
]
#: the store holds both Boocock panels (pooled and de novo), so the record count is not
#: the pooled head's alone
LABEL_OVERRIDE = {"Boocock2025Dataset": "Boocock 2025"}
MODALITIES = ["mRNA", "protein", "promoter", "morphology"]


def _catalog() -> list[tuple[str, str, str]]:
    """(class, label, modality) in the order every data panel draws: modality first,
    then the order of ``C.DATASETS``, the extra datasets after their modality's heads.
    """
    rows = [
        (HEAD_CLASS[k], LABEL_OVERRIDE.get(HEAD_CLASS[k], lab), m)
        for k, lab, m in C.DATASETS
    ]
    rows += EXTRA
    return sorted(rows, key=lambda r: MODALITIES.index(r[2]))


CATALOG = _catalog()
LABEL = {cls: lab for cls, lab, _ in CATALOG}
MODALITY = {cls: m for cls, _, m in CATALOG}
FS = 5.0  # annotation size inside the panels, Nature's floor
#: the label name a record carries, shortened for a bar annotation
SHORT_LABEL = {
    "expression_log2_ratio": "log2 ratio",
    "expression_tpm": "TPM",
    "expression_score": "error-model score",
    "expression_level": "sort-seq level",
    "umi_count": "UMI count",
    "protein_abundance": "abundance",
    "degradation_rate": "degradation rate",
    "calmorph": "CalMorph",
}


def _fill(line: str) -> str:
    """The light companion of a palette line color (the lighter member of a pair)."""
    return PLOT_PALETTE_FILL[PLOT_PALETTE.index(line)]


def _plural(unit: str, n: int) -> str:
    return unit if n == 1 else f"{unit}s"


def _vgrid(ax: Axes, step: float, label_every: int = 2) -> None:
    """``C.hgrid`` for the x axis: gridlines every ``step``, labels every other."""
    ax.xaxis.set_major_locator(MultipleLocator(step * label_every))
    ax.xaxis.set_minor_locator(MultipleLocator(step))
    ax.grid(axis="x", which="both", color="#DDDDDD", linewidth=0.4, zorder=0)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_axisbelow(True)


def _rows_axis(ax: Axes, n: int) -> None:
    ax.set_ylim(n - 0.4, -0.6)
    ax.tick_params(axis="y", length=0, pad=1.5)


def _modality_legend(ax: Axes, present: list[str]) -> None:
    handles = [
        Patch(facecolor=C.MODALITY_COLOR[m], edgecolor="black", linewidth=0.5, label=m)
        for m in MODALITIES
        if m in present
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncols=len(handles),
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.7,
        borderaxespad=0.1,
        fontsize=FS,
    )


def _keep_inside(ax: Axes, texts: list[Any], inside_x: float) -> None:
    """Move an annotation that would run past the box's right edge to ``inside_x``
    (data units), where it sits over its own bar.
    """
    fig = ax.figure.get_figure(root=True)
    fig.draw_without_rendering()
    renderer = fig._get_renderer()
    right = ax.get_window_extent().x1
    for t in texts:
        if t.get_window_extent(renderer).x1 > right - 2:
            t.set_x(inside_x)


# ----------------------------------------------------------------------------- records


def _role(counts: dict[str, Any]) -> str:
    """The role of a dataset in the split: '' when the trainer fits it."""
    if counts["train"] > 0:
        return "" if counts["val"] + counts["test"] > 0 else "train only"
    if counts["probe"] > 0:
        return "probe only"
    if counts["split_b_test"] > 0:
        return "split B test"
    if counts["excluded"] > 0:
        return "excluded"
    raise ValueError(f"no records in any split: {counts}")


def panel_records(ax: Axes, R: C.Reads) -> str:
    n_rec = R.census["n_records_by_dataset_index"]
    first = R.census["first_sample_by_dataset"]
    counts = R.split["counts"]
    assert set(n_rec) == set(LABEL), set(n_rec) ^ set(LABEL)
    classes = [cls for cls, _, _ in CATALOG]
    y = np.arange(len(classes))
    vals = np.array([n_rec[c] for c in classes], dtype=float)
    left = 10.0 ** np.floor(np.log10(vals.min()))
    faces = []
    for c in classes:
        line = C.MODALITY_COLOR[MODALITY[c]]
        faces.append(line if counts[c]["train"] > 0 else _fill(line))
    ax.barh(
        y,
        vals - left,
        left=left,
        height=0.72,
        color=faces,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlim(left, 10.0 ** (np.ceil(np.log10(vals.max())) + 1))
    texts = []
    for yi, c, v in zip(y, classes, vals, strict=True):
        desc = SHORT_LABEL[first[c]["label_name"]]
        role = _role(counts[c])
        if role:
            desc += f", {role}"
        texts.append(
            ax.text(
                v * 1.25,
                yi,
                f"{int(v):,}  {desc}",
                va="center",
                ha="left",
                fontsize=FS,
                zorder=4,
            )
        )
    ax.set_yticks(y, [LABEL[c] for c in classes], fontsize=FS)
    _rows_axis(ax, len(classes))
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", which="major", color="#DDDDDD", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("records read (pale: not trained on)", fontsize=FS, labelpad=1)
    _modality_legend(ax, [MODALITY[c] for c in classes])
    C.box(ax)
    C.title(ax, "records read, by dataset and modality", legend_rows=1)
    _keep_inside(ax, texts, left * 1.3)
    return "ready"


# ------------------------------------------------------------------------------- split

AXIS_OF_UNIT = {
    "gene": "gene",
    "double deletion": "gene",
    "single deletion": "gene",
    "isolate": "isolate",
    "segregant record": "segregant",
    "distinct 80-mer": "promoter",
}
SPLIT_PARTS = [  # key in record_counts, legend label, color
    ("train", "train", C.AMBER),
    ("val", "validation", C.BRICK),
    ("test", "test", C.PURPLE),
    ("split_b_test", "test", C.PURPLE),
    ("probe", "probe", C.SAND),
    ("excluded", "excluded", C.GRAY),
]


def _axis_header(S: dict[str, Any], axis: str) -> str:
    total = {
        "gene": (S["n_genes"], "genes"),
        "isolate": (S["n_isolates"], "isolates"),
        "segregant": (S["n_segregant_records"], "segregant records"),
        "promoter": (S["n_promoter_80mers"], "80-mers"),
    }[axis]
    folds = list(S["fold_sizes"][axis].values())
    return f"{total[0]:,} {total[1]}, {len(folds)} folds of {min(folds):,} to {max(folds):,}"


def panel_split(ax: Axes, R: C.Reads) -> str:
    S = R.split
    rc, counts = S["record_counts"], S["counts"]
    assert set(rc) == set(LABEL), set(rc) ^ set(LABEL)
    order: list[tuple[str, str | None]] = []  # (axis, class or None for a header)
    for axis in ("gene", "isolate", "segregant", "promoter"):
        members = [
            cls for cls, _, _ in CATALOG if AXIS_OF_UNIT[counts[cls]["unit"]] == axis
        ]
        order.append((axis, None))
        order += [(axis, cls) for cls in members]
    labels = []
    for yi, (axis, cls) in enumerate(order):
        if cls is None:
            ax.text(
                0.005,
                yi + 0.1,
                _axis_header(S, axis),
                fontsize=FS,
                va="center",
                ha="left",
                color="black",
                style="italic",
                zorder=4,
            )
            labels.append("")
            continue
        r = rc[cls]
        total = sum(r[k] for k, _, _ in SPLIT_PARTS)
        x = 0.0
        for k, _, color in SPLIT_PARTS:
            if r[k] == 0:
                continue
            w = r[k] / total
            ax.barh(
                yi,
                w,
                left=x,
                height=0.72,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            x += w
        k = counts[cls]
        units = [k[f"units_{p}"] for p in ("train", "val", "test")]
        if sum(units):
            nz = [u for u in units if u]
            txt = f"{' / '.join(f'{u:,}' for u in nz)} {_plural(k['unit'], max(nz))}"
        else:
            role = _role(k)
            n = r["probe"] + r["excluded"] + r["split_b_test"]
            txt = f"{n:,} {_plural(k['unit'], n)}, {role}"
        on_dark = r["excluded"] == total
        ax.text(
            0.015,
            yi,
            txt,
            fontsize=FS,
            va="center",
            ha="left",
            zorder=5,
            color="white" if on_dark else "black",
        )
        labels.append(LABEL[cls])
    ax.set_yticks(range(len(order)), labels, fontsize=FS)
    _rows_axis(ax, len(order))
    ax.set_xlim(0, 1)
    _vgrid(ax, 0.1)
    ax.set_xlabel("fraction of the dataset's records", fontsize=FS, labelpad=1)
    seen: dict[str, str] = {}
    for _, lab, color in SPLIT_PARTS:
        seen.setdefault(lab, color)
    handles = [
        Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=lab)
        for lab, c in seen.items()
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncols=len(handles),
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.6,
        borderaxespad=0.1,
        fontsize=FS,
    )
    C.box(ax)
    C.title(ax, "what is held out: genes, isolates, 80-mers", legend_rows=1)
    return "ready"


# ---------------------------------------------------------------------------- ontology


def _phenotype_class(
    profiles: list[dict[str, Any]], parents: dict[str, list[str]]
) -> str:
    """A dataset's most specific phenotype class: a ``*Phenotype`` class node of its
    profiles that is no other profile class's parent.
    """
    found = {
        n
        for p in profiles
        for n in p["nodes"]
        if n.endswith("Phenotype")
        and n != "Phenotype"
        and ":" not in n
        and "." not in n
    }
    leaves = [n for n in found if not any(n in parents[m] for m in found if m != n)]
    assert len(leaves) == 1, leaves
    return leaves[0]


def panel_ontology(ax: Axes, R: C.Reads) -> str:
    onto = R.ontology
    parents = {n["name"]: n["parents"] for n in onto["vocabulary"]["nodes"]}
    leaf_of = {
        cls: _phenotype_class(v["profiles"], parents)
        for cls, v in onto["datasets"].items()
    }
    assert set(leaf_of) == set(LABEL), set(leaf_of) ^ set(LABEL)

    # the chain from Phenotype down to each leaf, through phenotype-class parents only
    def chain(c: str) -> list[str]:
        up = [p for p in parents[c] if p.endswith("Phenotype")]
        return [c] if up == ["Phenotype"] else chain(up[0]) + [c]

    # the tree, children in the order their first dataset appears in the catalog
    children: dict[str, list[str]] = {}
    for cls, _, _ in CATALOG:
        path = ["Phenotype"] + chain(leaf_of[cls])
        for up, node in zip(path, path[1:]):
            if node not in children.setdefault(up, []):
                children[up].append(node)
    rows: list[
        tuple[int, str, list[str]]
    ] = []  # (depth, class, datasets on that class)

    def walk(node: str, depth: int) -> None:
        for ch in children.get(node, []):
            rows.append((depth, ch, [c for c, _, _ in CATALOG if leaf_of[c] == ch]))
            walk(ch, depth + 1)

    walk("Phenotype", 0)
    n_lines = 1 + sum(max(1, len(m)) for _, _, m in rows) + 1
    line = 1.0 / (n_lines + 0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    y = 1.0 - line * 0.9
    # dataset names end flush against the modality square, so a long class name and a
    # long dataset name on the same row can share the width
    x_ds, x_sq, x_prof, indent = 0.86, 0.885, 0.985, 0.032

    def put(x: float, yy: float, s: str, **kw: Any) -> None:
        ax.text(x, yy, s, fontsize=FS, va="center", transform=ax.transAxes, **kw)

    put(0.02, y, "Phenotype", fontweight="bold")
    put(x_prof, y, "dataset, profiles", fontweight="bold", ha="right")
    y_parent = {-1: y}
    for depth, node, members in rows:
        y -= line
        x = 0.02 + indent * (depth + 1)
        ax.plot(
            [x - indent + 0.01, x - indent + 0.01, x - 0.005],
            [y_parent[depth - 1] - line * 0.4, y, y],
            color=C.GRAY,
            linewidth=0.4,
            transform=ax.transAxes,
            clip_on=False,
        )
        y_parent[depth] = y
        put(x, y, node.removesuffix("Phenotype"))
        for i, c in enumerate(members):
            yy = y - i * line
            ax.scatter(
                [x_sq],
                [yy],
                s=6,
                marker="s",
                color=C.MODALITY_COLOR[MODALITY[c]],
                edgecolors="black",
                linewidths=0.4,
                transform=ax.transAxes,
                zorder=4,
            )
            put(x_ds, yy, LABEL[c], ha="right")
            put(x_prof, yy, f"{onto['datasets'][c]['n_profiles']:,}", ha="right")
        y -= line * (max(1, len(members)) - 1)
    put(
        0.02,
        line * 0.55,
        f"{onto['vocabulary_size']:,} terms, depth {onto['vocabulary_depth']}, "
        f"{onto['n_distinct_annotations']} distinct annotations",
        color=C.GRAY,
    )
    _modality_legend(ax, MODALITIES)
    C.box(ax)
    C.title(ax, "each dataset's phenotype class, ontology", legend_rows=1)
    return "ready"


# ---------------------------------------------------------------------------- ceilings


def _ceiling_rows(W: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per dataset class: the within-dataset agreement the file measures, as median,
    quartiles, its null median (or None), what is compared (``tag``), its n, and whether it is a
    replicate agreement (False: a bound from another dataset).
    """
    km = W["kemmeren2014"]
    jk = W["jackson2020"]["pooled"]["condition_sum"]["10"]["effect_r"]
    cmp_ = W["segregants"]["comparisons"]
    bh = W["hu2007"]["bound_kemmeren_x_hu"]["per_genotype_pearson"]
    bo, ng = cmp_["ap_pooled_x_a_de_novo"], cmp_["ap_pooled_x_nguessan"]
    return {
        "MicroarrayKemmeren2014Dataset": dict(
            s=km["per_strain_pearson"],
            null=km["per_strain_null_different_mutants"]["median"],
            tag="test-retest",
            n=f"{km['per_strain_pearson']['n']:,} strains",
            replicate=True,
        ),
        "TfDeletionPseudobulkJackson2020Dataset": dict(
            s=jk["same_genotype"],
            null=jk["different_genotype_null"]["median"],
            tag="same-TF pairs",
            n=f"{jk['same_genotype']['n']:,} pairs",
            replicate=True,
        ),
        "Boocock2025Dataset": dict(
            s=bo["deviation_r_nearest"],
            null=bo["deviation_r_random"]["median"],
            tag="pooled x de novo",
            n=f"{bo['deviation_r_nearest']['n']:,} segregants",
            replicate=True,
        ),
        "Nguessan2025Dataset": dict(
            s=ng["deviation_r_nearest"],
            null=ng["deviation_r_random"]["median"],
            tag="x Boocock pooled",
            n=f"{ng['deviation_r_nearest']['n']:,} segregants",
            replicate=True,
        ),
        "MicroarrayScoreHu2007Dataset": dict(
            s=bh,
            null=None,
            tag="bound, x Kemmeren",
            n=f"{bh['n']:,} deletions",
            replicate=False,
        ),
    }


def panel_ceilings(ax: Axes, R: C.Reads) -> str:
    rows = _ceiling_rows(R.ceilings)
    classes = [cls for cls, _, _ in CATALOG]
    placed: list[tuple[int, dict[str, Any], float]] = []
    for yi, c in enumerate(classes):
        if c not in rows:
            ax.text(
                0.01,
                yi,
                "not measured",
                fontsize=FS,
                va="center",
                ha="left",
                color=C.GRAY,
                zorder=4,
            )
            continue
        r = rows[c]
        s = r["s"]
        ax.barh(
            yi,
            s["median"],
            height=0.72,
            color=C.MODALITY_COLOR[MODALITY[c]],
            edgecolor="black",
            linewidth=0.5,
            hatch=None if r["replicate"] else "//////",
            zorder=3,
        )
        ax.plot([s["q1"], s["q3"]], [yi, yi], color="black", linewidth=0.6, zorder=4)
        if r["null"] is not None:
            ax.plot(
                [r["null"]],
                [yi],
                marker="|",
                markersize=5,
                markeredgewidth=0.8,
                color="black",
                zorder=5,
            )
        placed.append((yi, r, max(s["q3"], r["null"] or 0) + 0.02))
    ax.set_yticks(range(len(classes)), [LABEL[c] for c in classes], fontsize=FS)
    _rows_axis(ax, len(classes))
    ax.set_xlim(0, 1)
    _vgrid(ax, 0.1)
    ax.set_xlabel(
        "Pearson r (bar median, line quartiles, tick null median)",
        fontsize=FS,
        labelpad=1,
    )
    C.box(ax)
    C.title(ax, "what r each dataset attains against itself")
    # the median always sits right of the whisker; what is compared follows it when it
    # fits, else goes inside the bar left of the null tick and the whisker, else is left
    # to the caption
    fig = ax.figure.get_figure(root=True)
    fig.draw_without_rendering()
    renderer = fig._get_renderer()
    right = ax.get_window_extent().x1
    for yi, r, x in placed:
        s = r["s"]
        t = ax.text(
            x,
            yi,
            f"{s['median']:.2f}, {r['tag']}, {r['n']}",
            fontsize=FS,
            va="center",
            ha="left",
            zorder=6,
        )
        for long_ in (f"{s['median']:.2f}, {r['tag']}", f"{s['median']:.2f}"):
            if t.get_window_extent(renderer).x1 <= right - 2:
                break
            t.set_text(long_)
        if t.get_text() == f"{s['median']:.2f}":
            free = min(s["q1"], r["null"] if r["null"] is not None else 1.0)
            inner = ax.text(
                0.015, yi, r["tag"], fontsize=FS, va="center", ha="left", zorder=6
            )
            limit = ax.transData.transform((free, yi))[0] - 2
            if inner.get_window_extent(renderer).x1 > limit:
                inner.remove()
    return "partial"


# ------------------------------------------------------------------------- epoch share

SCRIPTS = osp.dirname(osp.abspath(__file__))
CONF = osp.join(osp.dirname(SCRIPTS), "conf")
#: the recipe-v2 cells are submitted through this script; its defaults name the arm,
#: the sampler and the weighting (notes 2026-10-25, "Recipe v2 and ontology injection")
RECIPE_V2_SLURM = osp.join(SCRIPTS, "gh_train_043_sweep_v2.slurm")
RECIPE_V2_PREP = "prep_v4"
#: trainer short name -> store class, for the datasets an arm can train on
#: the fraction of the epoch-share box the longest bar spans
BAR_SPAN = 0.42
TRAINER_CLASS = {**HEAD_CLASS, "jackson": "TfDeletionPseudobulkJackson2020Dataset"}


def _slurm_default(name: str) -> str:
    """The ``NAME="${NAME:-default}"`` default of the recipe-v2 submission script."""
    with open(RECIPE_V2_SLURM) as fh:
        found = re.findall(
            rf'^{name}="\$\{{{name}:-([^}}]+)\}}"$', fh.read(), flags=re.MULTILINE
        )
    assert len(found) == 1, (name, found)
    return found[0]


def equal_weight_shares(counts: list[int], batch: int) -> list[float]:
    """Exact expected loss weight of each dataset in one uniformly drawn batch (the
    records sampler: ``batch`` records without replacement from all training records)
    under equal weighting (``combine_loss``: the mean of the per-dataset means of the
    datasets present, so a present dataset weighs 1 / datasets present).

    The same expectation as ``training_design_043_tables.expected_weights``' ``w_equal``,
    computed by a dynamic program over (records drawn, datasets present) of the other
    datasets instead of enumerating every composition of the batch.
    """
    total = comb(sum(counts), batch)
    out = []
    for s, c_s in enumerate(counts):
        others = [c for i, c in enumerate(counts) if i != s]
        # ways[t][k]: ways the other datasets fill t slots with k of them present
        ways = [[0] * (len(others) + 1) for _ in range(batch + 1)]
        ways[0][0] = 1
        for c in others:
            new = [[0] * (len(others) + 1) for _ in range(batch + 1)]
            for t in range(batch + 1):
                for k in range(len(others) + 1):
                    if ways[t][k]:
                        for m in range(min(c, batch - t) + 1):
                            new[t + m][k + (m > 0)] += ways[t][k] * comb(c, m)
            ways = new
        w = sum(
            Fraction(comb(c_s, m) * ways[batch - m][k], k + 1)
            for m in range(1, min(c_s, batch) + 1)
            for k in range(len(others) + 1)
        )
        out.append(float(w / total))
    return out


def _recipe_v2() -> dict[str, Any]:
    """The recipe-v2 training set per dataset and the batch it is drawn in."""
    arm = _slurm_default("ARM")
    sampler = _slurm_default("SAMPLER")
    weighting = _slurm_default("WEIGHTING")
    with open(osp.join(CONF, "arm", f"{arm}.yaml")) as fh:
        arm_cfg = yaml.safe_load(fh)
    with open(osp.join(CONF, "train_043_default.yaml")) as fh:
        data_cfg = yaml.safe_load(fh)["data"]
    batch, route = int(data_cfg["batch_size"]), data_cfg["route"]
    with open(osp.join(CONF, "weighting", f"{weighting}.yaml")) as fh:
        w_cfg = yaml.safe_load(fh)
    assert (sampler, w_cfg["kind"], w_cfg["by"]) == ("records", "equal", "dataset")
    meta_path = osp.join(
        os.environ["DATA_ROOT"],
        "data/torchcell/experiments",
        C.SLUG,
        "train_043_prep",
        f"{RECIPE_V2_PREP}_route{route}",
        "meta.json",
    )
    with open(meta_path) as fh:
        meta = json.load(fh)
    assert meta["prep_version"] == RECIPE_V2_PREP, meta["prep_version"]
    shorts = [
        s for s in arm_cfg["datasets"] if s not in arm_cfg["exclude_from_training"]
    ]
    counts = [int(meta["counts"][s]["train"]) for s in shorts]
    return {
        "arm": arm,
        "batch": batch,
        "shorts": shorts,
        "counts": counts,
        "weights": equal_weight_shares(counts, batch),
    }


def panel_epoch_share(ax: Axes, R: C.Reads) -> str:
    """Per training dataset of recipe v2: the records one epoch draws, their mean count
    in a batch, and the expected loss weight under equal weighting.
    """
    E = _recipe_v2()
    order = sorted(
        range(len(E["shorts"])),
        key=lambda i: [c for c, _, _ in CATALOG].index(TRAINER_CLASS[E["shorts"][i]]),
    )
    shorts = [E["shorts"][i] for i in order]
    counts = np.array([E["counts"][i] for i in order], dtype=float)
    weights = [E["weights"][i] for i in order]
    n, batch = int(counts.sum()), E["batch"]
    steps = -(-n // batch)
    classes = [TRAINER_CLASS[s] for s in shorts]
    y = np.arange(len(shorts))
    left = 10.0 ** np.floor(np.log10(counts.min()))
    # the bars take the left of the box, the two number columns the right
    top = 10.0 ** (
        np.log10(left) + (np.log10(counts.max()) - np.log10(left)) / BAR_SPAN
    )
    ax.barh(
        y,
        counts - left,
        left=left,
        height=0.72,
        color=[C.MODALITY_COLOR[MODALITY[c]] for c in classes],
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlim(left, top)
    decades = [
        10.0**e for e in range(int(np.log10(left)), int(np.log10(counts.max())) + 2)
    ]
    for d in decades:
        ax.axvline(d, color="#DDDDDD", linewidth=0.4, zorder=0)
    ax.set_xticks(decades[::2])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    to_axes = ax.get_yaxis_transform()
    col_batch, col_w = 0.80, 0.995
    for yi, v, w in zip(y, counts, weights, strict=True):
        ax.text(v * 1.2, yi, f"{int(v):,}", va="center", ha="left", fontsize=FS)
        ax.text(
            col_batch,
            yi,
            f"{batch * v / n:.2f}",
            va="center",
            ha="right",
            fontsize=FS,
            transform=to_axes,
        )
        ax.text(
            col_w,
            yi,
            f"{w:.3f}",
            va="center",
            ha="right",
            fontsize=FS,
            transform=to_axes,
        )
    for x, head in ((col_batch, f"of {batch}"), (col_w, "loss w")):
        ax.text(
            x,
            -1.1,
            head,
            va="center",
            ha="right",
            fontsize=FS,
            fontweight="bold",
            transform=to_axes,
        )
    labels = [
        C.DATASET_LABEL[s] if s in C.DATASET_LABEL else LABEL[TRAINER_CLASS[s]]
        for s in shorts
    ]
    ax.set_yticks(y, labels, fontsize=FS)
    ax.set_ylim(len(shorts) - 0.4, -1.7)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel(
        f"training records = records per epoch (records sampler, each once)\n"
        f"{RECIPE_V2_PREP} meta.json, arm {E['arm']}, equal weighting, "
        f"{len(shorts)} datasets",
        fontsize=FS,
        labelpad=1,
    )
    C.box(ax)
    C.fit_title(
        ax,
        [
            f"an epoch draws {n:,} records in {steps:,} steps of {batch}",
            f"an epoch: {n:,} records, {steps:,} steps of {batch}",
            f"epoch: {n:,} records, {steps:,} steps",
        ],
    )
    return "ready"


# ----------------------------------------------------------------------------- targets


def panel_targets(ax: Axes, R: C.Reads) -> str:
    assert R.shapes["per_dataset_sample"] is None
    C.box(ax)
    return C.placeholder(
        ax,
        "the label distribution of every dataset as the\n"
        "trainer sees it: z-scored targets per head and\n"
        "the HL-Gauss bin edges over them.\n"
        "data_shapes_043.json holds value ranges only\n"
        "(per_dataset_sample = null); this needs a\n"
        "sampled pass over the store.",
        "targets per dataset",
    )


PANELS: dict[str, Callable[[Axes, C.Reads], str]] = {
    "records": panel_records,
    "split": panel_split,
    "ontology": panel_ontology,
    "ceilings": panel_ceilings,
    "epoch_share": panel_epoch_share,
    "targets": panel_targets,
}


def main() -> None:
    out = C.demo(PANELS, "figure_043_data_panels")
    print(out["png"])


if __name__ == "__main__":
    main()
