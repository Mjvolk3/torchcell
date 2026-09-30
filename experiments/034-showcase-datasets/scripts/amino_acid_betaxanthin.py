# experiments/034-showcase-datasets/scripts/amino_acid_betaxanthin.py
# [[experiments.034-showcase-datasets.scripts.amino_acid_betaxanthin]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/amino_acid_betaxanthin
"""Generate every table, figure and record dump on the amino-acid + betaxanthin showcase page.

Reads the three dev-tree LMDBs read-only (``$DATA_ROOT/data/torchcell/amino_acid_mulleder2016``,
``amino_acid_cooper2010`` and ``betaxanthin_cachera2023``, the stores
``AminoAcidMulleder2016Dataset``, ``AminoAcidCooper2010Dataset`` and
``BetaxanthinCachera2023Dataset`` build), resolves interned ``$ref`` pointers the way
``ExperimentDataset.get_single_item`` does, and compares each store with what the served
release returned: the release's ``content_sha256`` from
``experiments/034-showcase-datasets/results/amino_acid_betaxanthin_query.json`` against the
same digest over the dev store, and, where they differ, a field-by-field diff against the
query job's cached raw LMDB (``$DATA_ROOT/data/torchcell/showcase_amino_acid_betaxanthin``,
written by ``query_amino_acid_betaxanthin.py`` under slurm; this script never opens Neo4j).
Writes:

- MyST fragments to ``docs/source/showcase/_generated/amino-acid-betaxanthin/``
  (``record_mulleder.md``, ``record_cooper.md``, ``record_cachera.md``,
  ``summary_tables.md``, ``served_vs_dev.md``, ``figures.md``, ``provenance.md``), each
  opening with an HTML comment naming this script and the run timestamp;
- figures as true-size SVG (``torchcell.utils.savefig_true_size_svg``) plus PNG to
  ``$ASSET_IMAGES_DIR/034-showcase-datasets/``, with the SVGs copied into the
  ``_generated`` directory so Sphinx can embed them;
- ``experiments/034-showcase-datasets/results/amino_acid_betaxanthin_summary.json`` with
  every number the fragments print.

Run from the repo root, after the query job::

    python experiments/034-showcase-datasets/scripts/amino_acid_betaxanthin.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import os.path as osp
import pickle
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.data.experiment_dataset import resolve_interned  # noqa: E402
from torchcell.datamodels.schema import (  # noqa: E402
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

SCRIPT = "experiments/034-showcase-datasets/scripts/amino_acid_betaxanthin.py"
FRAGMENT_DIR = Path("docs/source/showcase/_generated/amino-acid-betaxanthin")
RESULTS_DIR = Path("experiments/034-showcase-datasets/results")
RESULTS_JSON = RESULTS_DIR / "amino_acid_betaxanthin_summary.json"
QUERY_JSON = RESULTS_DIR / "amino_acid_betaxanthin_query.json"
QUERY_CACHE = "data/torchcell/showcase_amino_acid_betaxanthin/raw/lmdb"
IMAGE_SUBDIR = "034-showcase-datasets"

MULLEDER = "AminoAcidMulleder2016Dataset"
COOPER = "AminoAcidCooper2010Dataset"
CACHERA = "BetaxanthinCachera2023Dataset"
DATASETS = {
    MULLEDER: "data/torchcell/amino_acid_mulleder2016",
    COOPER: "data/torchcell/amino_acid_cooper2010",
    CACHERA: "data/torchcell/betaxanthin_cachera2023",
}
SHORT = {MULLEDER: "Mulleder", COOPER: "Cooper", CACHERA: "Cachera"}

# Cooper keys that name one amino acid, the keys a Mulleder amino acid can be compared
# with; the composite peaks (e.g. ``glutamine+valine``) are left out.
SHARED_SINGLE_KEYS = (
    "alanine",
    "arginine",
    "aspartate",
    "glutamate",
    "glycine",
    "serine",
    "threonine",
)
# Mulleder amino acids shown in the record dump; the rest of the 19 are cut.
MULLEDER_DUMP_KEYS = ("arginine", "glutamine", "tyrosine")
COOPER_DUMP_KEYS = ("arginine", "glutamine+valine", "asparagine+tyrosine")

COMMON_FIELDS = (
    "experiment.experiment_type",
    "experiment.dataset_name",
    "experiment.genotype.perturbations[{d}].systematic_gene_name",
    "experiment.genotype.perturbations[{d}].perturbed_gene_name",
    "experiment.genotype.perturbations[{d}].perturbation_type",
    "experiment.environment.media.name",
    "experiment.environment.media.state",
    "experiment.environment.media.base_medium",
    "experiment.phenotype.graph_level",
    "experiment.phenotype.label_name",
    "experiment.phenotype.measurement_type",
    "reference.experiment_reference_type",
    "reference.genome_reference.strain",
    "publication.doi",
)


# --------------------------------------------------------------------------- reading
def read_records(processed_dir: str) -> list[dict[str, Any]]:
    """Every record of a built ExperimentDataset, interned pointers resolved.

    Mirrors ``ExperimentDataset.get_single_item``: pickled records from ``lmdb``, the
    sibling ``interned`` env (when present) loaded once, ``resolve_interned`` splicing
    each ``$ref`` back. A store with no ``interned`` env is a legacy inline store.
    """
    interned: dict[str, Any] = {}
    interned_dir = osp.join(processed_dir, "interned")
    if osp.isdir(interned_dir):
        env = lmdb.open(interned_dir, readonly=True, lock=False)
        with env.begin() as txn:
            for key, value in txn.cursor():
                interned[key.decode()] = pickle.loads(value)
        env.close()
    env = lmdb.open(osp.join(processed_dir, "lmdb"), readonly=True, lock=False)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        n = txn.stat()["entries"]
        for idx in range(n):
            raw = txn.get(f"{idx}".encode())
            assert raw is not None, f"missing key {idx} in {processed_dir}"
            records.append(resolve_interned(pickle.loads(raw), interned))
    env.close()
    return records


def read_query_cache(lmdb_dir: str) -> list[dict[str, Any]]:
    """Every entry of the query job's raw LMDB (one JSON dict per key)."""
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    out: list[dict[str, Any]] = []
    with env.begin() as txn:
        for _, value in txn.cursor():
            out.append(json.loads(value.decode()))
    env.close()
    return out


def dump_record(record: dict[str, Any], keep: tuple[str, ...]) -> str:
    """Debugger-style tree of a stored record, cut to the dotted paths in ``keep``.

    The experiment and reference are re-validated through the pydantic classes the
    dataset stored them as (``EXPERIMENT_TYPE_MAP`` / ``EXPERIMENT_REFERENCE_TYPE_MAP``)
    and dumped with ``model_dump()``; the publication dict is used as stored. At every
    level, the number of fields cut is printed as a comment.
    """
    experiment = EXPERIMENT_TYPE_MAP[record["experiment"]["experiment_type"]](
        **record["experiment"]
    )
    reference = EXPERIMENT_REFERENCE_TYPE_MAP[
        record["reference"]["experiment_reference_type"]
    ](**record["reference"])
    full = {
        "experiment": experiment.model_dump(),
        "reference": reference.model_dump(),
        "publication": record["publication"],
    }
    kept: dict[str, Any] = {}
    for path in keep:
        node = kept
        for part in path.split("."):
            node = node.setdefault(part, {})
        node["__leaf__"] = _pick(full, path)
    return "\n".join(_render(kept, full, 0))


def _pick(obj: Any, path: str) -> Any:
    node = obj
    for part in path.split("."):
        name, _, index = part.partition("[")
        node = node[name]
        if index:
            node = node[int(index.rstrip("]"))]
    return node


def _resolve(obj: Any, key: str) -> Any:
    name, _, index = key.partition("[")
    node = obj[name]
    return node[int(index.rstrip("]"))] if index else node


def _render(kept: dict[str, Any], full: Any, depth: int) -> list[str]:
    pad = "    " * depth
    lines: list[str] = []
    for key, child in kept.items():
        if "__leaf__" in child:
            value = child["__leaf__"]
            text = value.value if hasattr(value, "value") else value
            lines.append(f"{pad}{key} = {text!r}")
        else:
            node = _resolve(full, key)
            lines.append(f"{pad}{key}:")
            lines.extend(_render(child, node, depth + 1))
    if isinstance(full, dict):
        shown = {k.partition("[")[0] for k in kept}
        cut = len([k for k in full if k not in shown])
        if cut:
            lines.append(f"{pad}# ... {cut} other field(s) cut")
    return lines


# --------------------------------------------------------------------------- helpers
def deletion(experiment: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Index and dict of the one deletion perturbation of an experiment."""
    hits = [
        (i, p)
        for i, p in enumerate(experiment["genotype"]["perturbations"])
        if p["perturbation_type"].endswith("deletion")
    ]
    assert len(hits) == 1, f"expected one deletion, got {len(hits)}"
    return hits[0]


def experiment_id(experiment: dict[str, Any]) -> str:
    """KG node id of a stored experiment dict (``scripts/package_dataset_lmdb.py``)."""
    return hashlib.sha256(json.dumps(experiment).encode("utf-8")).hexdigest()


def content_sha256(ids: list[str]) -> str:
    """sha256 of the sorted, newline-joined ids (``torchcell.datasets.artifact``)."""
    digest = hashlib.sha256()
    for item in sorted(ids):
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def leaf_paths(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of a nested dict/list as ``{dotted.path: value}``."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(leaf_paths(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if not obj:
            out[prefix] = []
        for i, v in enumerate(obj):
            out.update(leaf_paths(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


DIFF_DEPTH = 3


def _generic(path: str) -> str:
    """A leaf path cut to ``DIFF_DEPTH`` segments, list indices dropped."""
    parts = []
    for part in path.split(".")[:DIFF_DEPTH]:
        parts.append(part.partition("[")[0])
    return ".".join(parts)


def _same(a: Any, b: Any) -> bool:
    """Equality that treats two float NaNs as the same value."""
    if (
        isinstance(a, float)
        and isinstance(b, float)
        and math.isnan(a)
        and math.isnan(b)
    ):
        return True
    return bool(a == b)


def quantiles(values: list[float]) -> dict[str, float]:
    """n, min, q1, median, mean, q3, max of a list."""
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "q1": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "q3": float(np.percentile(a, 75)),
        "max": float(a.max()),
    }


def sha256_of(path: str) -> str:
    """sha256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- served vs dev
def served_vs_dev(
    name: str,
    dev_records: list[dict[str, Any]],
    served_entries: list[dict[str, Any]],
    served_hash: str,
    served_n: int,
    gene_set: set[str],
) -> dict[str, Any]:
    """Compare one dev store with the served release and the entries the query returned."""
    dev_hash = content_sha256([experiment_id(r["experiment"]) for r in dev_records])
    dev_genes = [
        deletion(r["experiment"])[1]["systematic_gene_name"] for r in dev_records
    ]
    outside = sorted({g for g in dev_genes if g not in gene_set})
    out: dict[str, Any] = {
        "dev_records": len(dev_records),
        "dev_content_sha256": dev_hash,
        "served_experiments": served_n,
        "served_content_sha256": served_hash,
        "identical": dev_hash == served_hash,
        "returned_entries": len(served_entries),
        "dev_records_deleting_a_gene_outside_gene_set": sum(
            1 for g in dev_genes if g not in gene_set
        ),
        "dev_genes_outside_gene_set": outside,
    }
    if out["identical"]:
        return out
    # Match returned entries to dev records by deleted gene; count the leaf paths that
    # differ over the pairs that match one to one.
    dev_by_gene: dict[str, list[dict[str, Any]]] = {}
    for r in dev_records:
        g = deletion(r["experiment"])[1]["systematic_gene_name"]
        dev_by_gene.setdefault(g, []).append(r["experiment"])
    served_by_gene: dict[str, list[dict[str, Any]]] = {}
    for e in served_entries:
        g = deletion(e["experiment"])[1]["systematic_gene_name"]
        served_by_gene.setdefault(g, []).append(e["experiment"])
    diff_paths: Counter[str] = Counter()
    pairs = 0
    pairs_differing = 0
    examples: dict[str, tuple[str, Any, Any]] = {}
    for g, served in served_by_gene.items():
        dev = dev_by_gene.get(g, [])
        if len(served) != 1 or len(dev) != 1:
            continue
        pairs += 1
        a = leaf_paths(served[0])
        b = leaf_paths(dev[0])
        leaves = sorted(
            p
            for p in set(a) | set(b)
            if not _same(a.get(p, "<absent>"), b.get(p, "<absent>"))
        )
        paths = {_generic(p) for p in leaves}
        if paths:
            pairs_differing += 1
        for p in paths:
            diff_paths[p] += 1
        for p in leaves:
            examples.setdefault(
                _generic(p), (p, a.get(p, "<absent>"), b.get(p, "<absent>"))
            )
    out.update(
        {
            "matched_one_to_one": pairs,
            "matched_pairs_differing": pairs_differing,
            "genes_returned_more_than_once": sorted(
                g for g, v in served_by_gene.items() if len(v) > 1
            ),
            "differing_paths": dict(diff_paths.most_common()),
            "differing_path_examples": {
                p: {"leaf": leaf, "served": repr(sv)[:120], "dev": repr(dv)[:120]}
                for p, (leaf, sv, dv) in examples.items()
            },
        }
    )
    return out


# Verbatim sentences of the Cachera paper the Caveats cite, located in the mirror OCR.
CACHERA_QUOTES = (
    "we picked cells from each of the four colonies obtained for each strain on the "
    "final CRI-SPA screen plate (YPD-G418)",
    "filter taking the geometric mean of Value and Saturation of image pixels in the "
    "HSV color space",
    "which was our observation on YPD media",
)


def cachera_sources(data_root: str) -> dict[str, Any]:
    """Line numbers of ``CACHERA_QUOTES`` in the mirror OCR, and its temperature mentions.

    Every quote must occur exactly once. The temperature search counts lines holding a
    degree sign, a LaTeX circ command or the word ``temperature``, the forms a growth
    temperature takes in this OCR.
    """
    path = osp.join(
        data_root,
        "torchcell-library",
        "cacheraCRISPAHighthroughputMethod2023",
        "paper.md",
    )
    lines = Path(path).read_text().split("\n")
    found = {}
    for quote in CACHERA_QUOTES:
        hits = [i + 1 for i, line in enumerate(lines) if quote in line]
        assert len(hits) == 1, f"{quote!r} found on lines {hits}"
        found[quote] = hits[0]
    temperature_lines = [
        i + 1
        for i, line in enumerate(lines)
        if "\u00b0" in line or "\\circ" in line or "temperature" in line.lower()
    ]
    return {
        "paper_md": "$DATA_ROOT/torchcell-library/cacheraCRISPAHighthroughputMethod2023/paper.md",
        "sha256": sha256_of(path),
        "quotes": found,
        "temperature_lines": temperature_lines,
    }


def cachera_fragment(stamp: str, c: dict[str, Any]) -> str:
    """The cachera_sources.md fragment: each quote with its line, and the search."""
    lines = [
        header(stamp).rstrip("\n"),
        "",
        f"Mirror OCR `{c['paper_md']}`, sha256 `{c['sha256']}`:",
        "",
    ]
    for quote, line in c["quotes"].items():
        lines.append(f'- line {line}: "{quote}"')
    tl = c["temperature_lines"]
    lines += [
        "",
        f'Lines holding a degree sign, `\\circ` or the word "temperature": {len(tl)}'
        + (f" (lines {', '.join(str(n) for n in tl)})" if tl else "")
        + ".",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- figures
def _box(ax: Any) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)


def _save(fig: Any, out_dir: str, stem: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    svg = osp.join(out_dir, f"{stem}.svg")
    png = osp.join(out_dir, f"{stem}.png")
    savefig_true_size_svg(fig, svg)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"svg": svg, "png": png}


def figure_mulleder_levels(
    levels: dict[str, list[float]], reference: dict[str, float], out_dir: str
) -> dict[str, str]:
    """Per-amino-acid log10 mM box plots, ordered by median, reference mean marked."""
    apply_paper_style()
    order = sorted(levels, key=lambda aa: float(np.median(levels[aa])))
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(55)),
        constrained_layout=True,
    )
    data = [np.log10(np.asarray(levels[aa])) for aa in order]
    ax.boxplot(
        data,
        widths=0.6,
        patch_artist=True,
        boxprops={"facecolor": PLOT_PALETTE[0], "edgecolor": "black", "linewidth": 0.4},
        medianprops={"color": "black", "linewidth": 0.6},
        whiskerprops={"linewidth": 0.4},
        capprops={"linewidth": 0.4},
        flierprops={
            "marker": "o",
            "markersize": 0.8,
            "markerfacecolor": "black",
            "markeredgewidth": 0,
        },
    )
    ax.scatter(
        np.arange(1, len(order) + 1),
        [math.log10(reference[aa]) for aa in order],
        marker="D",
        s=6,
        color=PLOT_PALETTE[1],
        edgecolor="black",
        linewidth=0.3,
        zorder=3,
        label="reference (robust mean over all strains)",
    )
    ax.set_xticks(np.arange(1, len(order) + 1), order, rotation=45, ha="right")
    ax.set_ylabel("log10 intracellular concentration (mM)")
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(which="minor", length=0)
    ax.grid(True, which="both", axis="y", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6, frameon=False)
    _box(ax)
    return _save(fig, out_dir, "mulleder_amino_acid_levels")


def figure_cachera_histogram(
    scores: list[float], n_colonies: list[int], out_dir: str
) -> dict[str, str]:
    """Betaxanthin CRI-SPA score histogram (top) and colonies per strain (bottom)."""
    apply_paper_style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(70)),
        constrained_layout=True,
    )
    lo = math.floor(min(scores) * 4) / 4
    hi = math.ceil(max(scores) * 4) / 4
    axes[0].hist(
        scores,
        bins=np.arange(lo, hi + 0.125, 0.125),
        color=PLOT_PALETTE[3],
        edgecolor="black",
        linewidth=0.3,
    )
    axes[0].axvline(0.0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_xlabel("betaxanthin CRI-SPA score (reference 0)")
    axes[0].set_ylabel("strains")
    axes[0].xaxis.set_major_locator(MultipleLocator(1.0))
    axes[0].xaxis.set_minor_locator(MultipleLocator(0.5))
    counts = Counter(n_colonies)
    xs = sorted(counts)
    axes[1].bar(
        xs,
        [counts[x] for x in xs],
        width=0.8,
        color=PLOT_PALETTE[3],
        edgecolor="black",
        linewidth=0.3,
    )
    axes[1].set_xlabel("colonies per strain (n_replicates)")
    axes[1].set_ylabel("strains")
    for ax in axes:
        ax.tick_params(which="minor", length=0)
        ax.grid(True, which="both", axis="x", linewidth=0.3, color="0.85")
        ax.set_axisbelow(True)
        _box(ax)
    return _save(fig, out_dir, "cachera_betaxanthin_score")


def figure_agreement(
    rows: list[dict[str, Any]], replicate: list[dict[str, Any]], out_dir: str
) -> dict[str, str]:
    """Spearman rho per shared amino acid: Mulleder vs Cooper, and Cooper vs itself."""
    apply_paper_style()
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(45)),
        constrained_layout=True,
    )
    x = np.arange(len(rows))
    width = 0.38
    series = (
        (rows, -width / 2, PLOT_PALETTE[4], "Mulleder vs Cooper, same gene"),
        (
            replicate,
            width / 2,
            PLOT_PALETTE_FILL[4],
            "Cooper vs Cooper, duplicate strains",
        ),
    )
    for data, offset, color, label in series:
        assert [r["key"] for r in data] == [r["key"] for r in rows]
        ax.bar(
            x + offset,
            [r["spearman"] for r in data],
            width,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(x, [r["key"] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_ylim(-0.2, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(True, which="both", axis="y", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=6, frameon=False)
    _box(ax)
    return _save(fig, out_dir, "mulleder_cooper_agreement")


# --------------------------------------------------------------------------- fragments
def header(stamp: str) -> str:
    return f"<!-- generated by {SCRIPT} at {stamp}; do not edit -->\n\n"


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def record_fragment(
    stamp: str, record: dict[str, Any], keep: tuple[str, ...], index: int, name: str
) -> str:
    """A record dump fragment: the Python that produced it, then the dump."""
    source_dir = DATASETS[name]
    code = (
        f"records = read_records(osp.join(DATA_ROOT, {source_dir!r}, 'processed'))\n"
        f"print(dump_record(records[{index}], keep=KEEP))  # KEEP: the {len(keep)} "
        "paths printed below\n"
    )
    return (
        header(stamp)
        + f"Record {index} of `{source_dir}/processed/lmdb`, produced by\n\n"
        + "```python\n"
        + inspect.getsource(read_records)
        + "\n"
        + inspect.getsource(dump_record)
        + "\n"
        + code
        + "```\n\n"
        + "```text\n"
        + dump_record(record, keep)
        + "\n```\n"
    )


def summary_fragment(stamp: str, s: dict[str, Any]) -> str:
    """The summary_tables.md fragment from the summary dict."""
    lines = [header(stamp).rstrip("\n"), ""]
    lines += [
        "**Counts per dataset** (dev stores)",
        "",
        "| dataset | records | deleted genes | metabolite keys | `measurement_type` | medium (`name`, `state`) | temperature | parent strain | `n_replicates` | SE stored |",
        "|---|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for name in DATASETS:
        d = s["datasets"][name]
        lines.append(
            f"| `{name}` | {d['records']:,} | {d['deleted_genes']:,} | {d['n_keys']} | "
            f"`{d['measurement_type']}` | {d['media_name']} ({d['media_state']}) | "
            f"{d['temperature']} | {d['strain']} | {d['n_replicates_range']} | "
            f"{d['se_stored']} |"
        )
    ov = s["overlap"]
    lines += [
        "",
        f"Deleted genes shared by the dev stores: Mulleder and Cooper "
        f"{ov['mulleder_and_cooper']:,}, Mulleder and Cachera {ov['mulleder_and_cachera']:,}, "
        f"Cooper and Cachera {ov['cooper_and_cachera']:,}, all three {ov['all_three']:,}, "
        f"union {ov['union']:,}.",
        "",
        "**Mulleder 2016, intracellular concentration (mM) per amino acid**, "
        "`phenotype.metabolite_level` over every record, with "
        "`reference.phenotype_reference.metabolite_level` (the same on every record)",
        "",
        "| amino acid | n | min | median | mean | max | reference |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for aa, q in s["mulleder_levels"].items():
        lines.append(
            f"| {aa} | {q['n']:,} | {_fmt(q['min'])} | {_fmt(q['median'])} | "
            f"{_fmt(q['mean'])} | {_fmt(q['max'])} | {_fmt(q['reference'])} |"
        )
    lines += [
        "",
        "**Cooper 2010, peak-area ratio to the plate mean per peak**, over the records "
        "that carry the key (reference 1.0 on every key)",
        "",
        "| peak key | records | exact zeros | median | mean | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, q in s["cooper_levels"].items():
        lines.append(
            f"| `{key}` | {q['n']:,} | {q['zeros']:,} | {_fmt(q['median'])} | "
            f"{_fmt(q['mean'])} | {_fmt(q['max'])} |"
        )
    led = s["cooper_ledger"]
    lines += [
        "",
        f"Cooper build ledger (`preprocess/dropped_records.json`): {led['n_source_rows']:,} "
        f"Table 4 rows, {led['n_kept']:,} kept, {led['n_dropped_orfs']:,} dropped by the "
        f"name resolver, {led['n_renamed']:,} kept under a renamed systematic name, "
        f"{led['n_duplicate_rows']:,} later rows of a duplicated identifier written to the "
        f"ledger and not served, {led['n_normalized']:,} identifiers normalized.",
        "",
        "**Cachera 2023, betaxanthin CRI-SPA score**",
        "",
        "| quantity | n | min | q1 | median | mean | q3 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, q in s["cachera"].items():
        lines.append(
            f"| {label} | {q['n']:,} | {_fmt(q['min'])} | {_fmt(q['q1'])} | "
            f"{_fmt(q['median'])} | {_fmt(q['mean'])} | {_fmt(q['q3'])} | "
            f"{_fmt(q['max'])} |"
        )
    lines += [
        "",
        f"{s['cachera_se_nan']:,} Cachera records store a NaN standard error (one colony, "
        "so no spread).",
        "",
        "**Mulleder and Cooper on the same deletion**: Spearman correlation between the "
        "Mulleder concentration and the Cooper ratio over the genes both stores hold, for "
        "each Cooper key that names one amino acid",
        "",
        "| amino acid | genes | Spearman rho |",
        "|---|---:|---:|",
    ]
    for r in s["agreement"]:
        lines.append(f"| {r['key']} | {r['n']:,} | {r['spearman']:.3f} |")
    lines += [
        "",
        "**Cooper against itself**: the same statistic between the kept row and the "
        "later row of each duplicated Table 4 identifier (two strains of one gene in the "
        "collection; the later rows are in the build ledger, not served)",
        "",
        "| amino acid | pairs | Spearman rho |",
        "|---|---:|---:|",
    ]
    for r in s["cooper_duplicate_agreement"]:
        lines.append(f"| {r['key']} | {r['n']:,} | {r['spearman']:.3f} |")
    dels = s["cachera_deletions_of_cassette_genes"]
    lines += [
        "",
        "Btx-cassette genes on every Cachera record: "
        + ", ".join(f"`{g}`" for g in s["cachera_cassette_genes"])
        + f". Cachera records whose deletion is of a cassette gene's native copy: "
        f"{len(dels):,}"
        + (" (" + ", ".join(f"`{g}` ({n})" for g, n in dels) + ")" if dels else "")
        + ".",
        "",
    ]
    return "\n".join(lines)


def served_fragment(stamp: str, s: dict[str, Any], release: str) -> str:
    """The served_vs_dev.md fragment."""
    lines = [
        header(stamp).rstrip("\n"),
        "",
        f"| dataset | dev records | served experiments (`{release}`) | query returned | dev content sha256 | served content sha256 | same |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for name in DATASETS:
        c = s["served_vs_dev"][name]
        lines.append(
            f"| `{name}` | {c['dev_records']:,} | {c['served_experiments']:,} | "
            f"{c['returned_entries']:,} | `{c['dev_content_sha256'][:16]}...` | "
            f"`{c['served_content_sha256'][:16]}...` | "
            f"{'yes' if c['identical'] else 'no'} |"
        )
    lines.append("")
    for name in DATASETS:
        c = s["served_vs_dev"][name]
        outside = c["dev_records_deleting_a_gene_outside_gene_set"]
        lines.append(
            f"- `{name}`: {outside:,} dev records delete a gene outside "
            f"`SCerevisiaeGenome.gene_set`"
            + (
                f" ({', '.join(f'`{g}`' for g in c['dev_genes_outside_gene_set'])})"
                if outside
                else ""
            )
            + "."
        )
        if c["identical"]:
            continue
        lines.append(
            f"  Of {c['matched_one_to_one']:,} returned entries matched one to one with a "
            f"dev record by deleted gene, {c['matched_pairs_differing']:,} differ. Fields "
            "that differ (paths cut to three levels; records), with the first differing "
            "leaf and its served and dev values:"
        )
        if c["genes_returned_more_than_once"]:
            lines.append(
                "  Genes the query returned more than once: "
                + ", ".join(f"`{g}`" for g in c["genes_returned_more_than_once"])
                + "."
            )
        lines.append("")
        lines.append("  | field | records | first differing leaf | served | dev |")
        lines.append("  |---|---:|---|---|---|")
        for path, n in c["differing_paths"].items():
            ex = c["differing_path_examples"][path]
            served = ex["served"].replace("|", "\\|")
            dev = ex["dev"].replace("|", "\\|")
            lines.append(
                f"  | `{path}` | {n:,} | `{ex['leaf']}` | `{served}` | `{dev}` |"
            )
        lines.append("")
    lines.append("")
    return "\n".join(lines)


def figures_fragment(stamp: str, s: dict[str, Any]) -> str:
    """The figures.md fragment: the three SVGs with captions."""
    cach = s["cachera"]["betaxanthin score"]
    col = s["cachera"]["colonies per strain (n_replicates)"]
    base = "_generated/amino-acid-betaxanthin"
    return (
        header(stamp)
        + f"```{{figure}} {base}/mulleder_amino_acid_levels.svg\n"
        + ":name: fig-mulleder-levels\n"
        + ":width: 100%\n\n"
        + "Intracellular amino-acid concentration in `AminoAcidMulleder2016Dataset`, "
        f"log10 mM, one box per amino acid over {s['datasets'][MULLEDER]['records']:,} "
        "deletion strains (box: quartiles; whiskers: 1.5 times the interquartile range; "
        "points beyond them drawn individually), ordered by median. Diamonds: the record "
        "reference, the robust (Minimum Covariance Determinant) mean over all strains.\n"
        + "```\n\n"
        + f"```{{figure}} {base}/cachera_betaxanthin_score.svg\n"
        + ":name: fig-cachera-score\n"
        + ":width: 88mm\n\n"
        + "Top: betaxanthin CRI-SPA score in `BetaxanthinCachera2023Dataset`, 0.125-wide "
        f"bins (n = {cach['n']:,}, median {_fmt(cach['median'])}); the dashed line is the "
        "reference level 0. Bottom: colonies behind each strain's score "
        f"(`n_replicates`, median {_fmt(col['median'])}).\n"
        + "```\n\n"
        + f"```{{figure}} {base}/mulleder_cooper_agreement.svg\n"
        + ":name: fig-mulleder-cooper\n"
        + ":width: 88mm\n\n"
        + "Spearman correlation per amino acid. Dark: the Mulleder 2016 concentration "
        "against the Cooper 2010 peak ratio over the deletions both stores hold "
        f"({min(r['n'] for r in s['agreement']):,} to "
        f"{max(r['n'] for r in s['agreement']):,} genes). Light: Cooper against itself, "
        "the kept row against the later row of each duplicated Table 4 identifier "
        f"({min(r['n'] for r in s['cooper_duplicate_agreement'])} to "
        f"{max(r['n'] for r in s['cooper_duplicate_agreement'])} pairs). Per-key values "
        "are in the tables above.\n" + "```\n"
    )


def provenance_fragment(stamp: str, prov: dict[str, Any]) -> str:
    """The provenance.md fragment."""
    cols = [prov[name] for name in DATASETS]
    rows = [
        ("loader", [f"`{c['loader']}`" for c in cols]),
        ("dev store", [f"`$DATA_ROOT/{c['store']}`" for c in cols]),
        ("raw input", [c["raw"] for c in cols]),
        ("source of record", [c["source"] for c in cols]),
        ("paper mirror", [c["paper"] for c in cols]),
        ("publication", [c["publication"] for c in cols]),
    ]
    lines = [
        header(stamp).rstrip("\n"),
        "",
        "| | " + " | ".join(f"`{n}`" for n in DATASETS) + " |",
        "|---|---|---|---|",
    ]
    for label, cells in rows:
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def provenance(
    data_root: str, records: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Where each store's inputs came from, as the loaders and mirrors record them."""
    lib = osp.join(data_root, "torchcell-library")
    raw_mirror = osp.join(data_root, "torchcell-raw")
    mul_raw = osp.join(
        data_root, DATASETS[MULLEDER], "raw", "Table_S3_Complete_Dataset.xls"
    )
    coo_raw = osp.join(
        raw_mirror,
        "cooperHighthroughputProfilingAmino2010",
        "data",
        "SupplementalTable4.txt",
    )
    cac_raw = osp.join(data_root, DATASETS[CACHERA], "raw", "GA1_2_4_6.csv")
    cac_mirror = osp.join(
        lib, "cacheraCRISPAHighthroughputMethod2023", "si", "GA1_2_4_6.csv"
    )
    coo_manifest = json.loads(
        Path(
            raw_mirror, "cooperHighthroughputProfilingAmino2010", "manifest.json"
        ).read_text()
    )
    coo_methods = sorted(
        {
            str(f["retrieval"]["method"])
            for f in coo_manifest["files"]
            if f.get("retrieval") is not None
        }
    )
    keys = {
        MULLEDER: "mullederFunctionalMetabolomicsDescribes2016",
        COOPER: "cooperHighthroughputProfilingAmino2010",
        CACHERA: "cacheraCRISPAHighthroughputMethod2023",
    }
    mul_sha = sha256_of(mul_raw)
    cac_sha = sha256_of(cac_raw)
    out: dict[str, Any] = {}
    for name, key in keys.items():
        pub = records[name][0]["publication"]
        out[name] = {
            "store": osp.join(DATASETS[name], "processed", "lmdb"),
            "citation_key": key,
            "paper_md_sha256": sha256_of(osp.join(lib, key, "paper.md")),
            "publication": f"DOI `{pub['doi']}`, PubMed `{pub['pubmed_id']}`",
        }
        out[name]["paper"] = (
            f"`$DATA_ROOT/torchcell-library/{key}/paper.md`, sha256 "
            f"`{out[name]['paper_md_sha256'][:16]}...`"
        )
    out[MULLEDER].update(
        {
            "loader": "torchcell/datasets/scerevisiae/mulleder2016.py::AminoAcidMulleder2016Dataset",
            "raw_sha256": mul_sha,
            "raw": f"`$DATA_ROOT/{DATASETS[MULLEDER]}/raw/Table_S3_Complete_Dataset.xls`, "
            f"sha256 `{mul_sha}`",
            "source": "Table S3 of the paper (Cell SI `mmc3.xls`, byte-identical to the "
            "loader's pinned Mendeley Data copy, 10.17632/bnzdhd6ck8.1); no raw-mirror "
            "record under `$DATA_ROOT/torchcell-raw/` yet (issue #487)",
        }
    )
    out[COOPER].update(
        {
            "loader": "torchcell/datasets/scerevisiae/cooper2010.py::AminoAcidCooper2010Dataset",
            "raw_sha256": sha256_of(coo_raw),
            "raw": "`$DATA_ROOT/torchcell-raw/cooperHighthroughputProfilingAmino2010/data/"
            f"SupplementalTable4.txt`, sha256 `{sha256_of(coo_raw)}` (the dev store's "
            "`raw/` links to it)",
            "source": "Genome Research Supplemental Table 4; raw-mirror `manifest.json` "
            f"retrieval method {', '.join(f'`{m}`' for m in coo_methods)} (the supplement "
            "sits behind an institutional login)",
        }
    )
    out[CACHERA].update(
        {
            "loader": "torchcell/datasets/scerevisiae/cachera2023.py::BetaxanthinCachera2023Dataset",
            "raw_sha256": cac_sha,
            "mirror_sha256": sha256_of(cac_mirror),
            "raw": f"`$DATA_ROOT/{DATASETS[CACHERA]}/raw/GA1_2_4_6.csv`, sha256 `{cac_sha}`",
            "source": "the authors' CRI-SPA repository file named by the paper's Data "
            "Availability; mirrored at `$DATA_ROOT/torchcell-library/"
            "cacheraCRISPAHighthroughputMethod2023/si/GA1_2_4_6.csv`, sha256 "
            + (
                "identical to the raw file"
                if sha256_of(cac_mirror) == cac_sha
                else f"`{sha256_of(cac_mirror)}` (DIFFERS from the raw file)"
            ),
        }
    )
    return out


# --------------------------------------------------------------------------- main
def _dev_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and constant fields of one dev store."""
    exps = [r["experiment"] for r in records]
    keys = {k for e in exps for k in e["phenotype"]["metabolite_level"]}
    n_reps = [n for e in exps for n in e["phenotype"]["n_replicates"].values()]
    env = exps[0]["environment"]
    temps = {
        None
        if e["environment"]["temperature"] is None
        else e["environment"]["temperature"]["value"]
        for e in exps
    }
    assert len(temps) == 1, f"temperature varies: {temps}"
    temp = temps.pop()
    media = {
        (e["environment"]["media"]["name"], e["environment"]["media"]["state"])
        for e in exps
    }
    assert len(media) == 1, f"medium varies: {media}"
    strains = {r["reference"]["genome_reference"]["strain"] for r in records}
    assert len(strains) == 1
    mtypes = {e["phenotype"]["measurement_type"] for e in exps}
    assert len(mtypes) == 1
    se_stored = sum(
        1 for e in exps if e["phenotype"]["metabolite_level_se"] is not None
    )
    return {
        "records": len(records),
        "deleted_genes": len({deletion(e)[1]["systematic_gene_name"] for e in exps}),
        "n_keys": len(keys),
        "measurement_type": mtypes.pop(),
        "media_name": env["media"]["name"],
        "media_state": env["media"]["state"],
        "temperature": "not recorded (typed gap)" if temp is None else f"{temp:g} °C",
        "strain": strains.pop(),
        "n_replicates_range": f"{min(n_reps)} to {max(n_reps)}"
        if min(n_reps) != max(n_reps)
        else f"{min(n_reps)} on every key",
        "se_stored": f"{se_stored:,} records" if se_stored else "none",
    }


def main() -> None:
    """Read the stores and the query result, compute, plot, write fragments and JSON."""
    load_dotenv()
    data_root = os.getenv("DATA_ROOT")
    asset_dir = os.getenv("ASSET_IMAGES_DIR")
    assert data_root is not None, "DATA_ROOT must be set in .env"
    assert asset_dir is not None, "ASSET_IMAGES_DIR must be set in .env"
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    image_dir = osp.join(asset_dir, IMAGE_SUBDIR)
    FRAGMENT_DIR.mkdir(parents=True, exist_ok=True)

    records = {
        name: read_records(osp.join(data_root, d, "processed"))
        for name, d in DATASETS.items()
    }
    for name, recs in records.items():
        print(f"{name}: {len(recs)} records")

    # ---- per-dataset counts
    summary: dict[str, Any] = {
        "script": SCRIPT,
        "generated_utc": stamp,
        "sources": {
            name: osp.join(data_root, d, "processed", "lmdb")
            for name, d in DATASETS.items()
        },
        "datasets": {name: _dev_summary(recs) for name, recs in records.items()},
    }

    genes = {
        name: {deletion(r["experiment"])[1]["systematic_gene_name"] for r in recs}
        for name, recs in records.items()
    }
    mul, coo, cac = genes[MULLEDER], genes[COOPER], genes[CACHERA]
    summary["overlap"] = {
        "mulleder_and_cooper": len(mul & coo),
        "mulleder_and_cachera": len(mul & cac),
        "cooper_and_cachera": len(coo & cac),
        "all_three": len(mul & coo & cac),
        "union": len(mul | coo | cac),
    }

    # ---- Mulleder levels
    mul_levels: dict[str, list[float]] = {}
    for r in records[MULLEDER]:
        for aa, v in r["experiment"]["phenotype"]["metabolite_level"].items():
            mul_levels.setdefault(aa, []).append(float(v))
    mul_ref = {
        aa: float(v)
        for aa, v in records[MULLEDER][0]["reference"]["phenotype_reference"][
            "metabolite_level"
        ].items()
    }
    assert all(
        r["reference"]["phenotype_reference"]["metabolite_level"]
        == records[MULLEDER][0]["reference"]["phenotype_reference"]["metabolite_level"]
        for r in records[MULLEDER]
    ), "Mulleder reference differs across records"
    summary["mulleder_levels"] = {
        aa: {**quantiles(v), "reference": mul_ref[aa]}
        for aa, v in sorted(mul_levels.items(), key=lambda kv: -float(np.median(kv[1])))
    }

    # ---- Cooper levels
    coo_levels: dict[str, list[float]] = {}
    for r in records[COOPER]:
        for key, v in r["experiment"]["phenotype"]["metabolite_level"].items():
            coo_levels.setdefault(key, []).append(float(v))
    summary["cooper_levels"] = {
        key: {**quantiles(v), "zeros": sum(1 for x in v if x == 0.0)}
        for key, v in sorted(coo_levels.items(), key=lambda kv: -len(kv[1]))
    }
    ledger = json.loads(
        Path(
            data_root, DATASETS[COOPER], "preprocess", "dropped_records.json"
        ).read_text()
    )
    summary["cooper_ledger"] = {
        k: ledger[k]
        for k in (
            "n_source_rows",
            "n_kept",
            "n_dropped_orfs",
            "n_renamed",
            "n_normalized",
            "n_duplicate_rows",
            "n_excluded_non_deletion",
            "n_essentiality_flagged",
        )
    }

    # ---- Cachera
    scores = [
        float(r["experiment"]["phenotype"]["metabolite_level"]["betaxanthin"])
        for r in records[CACHERA]
    ]
    colonies = [
        int(r["experiment"]["phenotype"]["n_replicates"]["betaxanthin"])
        for r in records[CACHERA]
    ]
    ses = [
        float(r["experiment"]["phenotype"]["metabolite_level_se"]["betaxanthin"])
        for r in records[CACHERA]
    ]
    summary["cachera"] = {
        "betaxanthin score": quantiles(scores),
        "colonies per strain (n_replicates)": quantiles([float(c) for c in colonies]),
        "standard error (non-NaN)": quantiles([x for x in ses if not math.isnan(x)]),
    }
    summary["cachera_se_nan"] = sum(1 for x in ses if math.isnan(x))

    # ---- Mulleder vs Cooper on the same deletion
    mul_by_gene = {
        deletion(r["experiment"])[1]["systematic_gene_name"]: r["experiment"][
            "phenotype"
        ]["metabolite_level"]
        for r in records[MULLEDER]
    }
    coo_by_gene: dict[str, dict[str, float]] = {}
    for r in records[COOPER]:
        g = deletion(r["experiment"])[1]["systematic_gene_name"]
        # A RENAMED merge puts two Cooper strains on one gene; the first in store order
        # is kept here and the skip is counted in ``cooper_genes_with_two_strains``.
        coo_by_gene.setdefault(g, r["experiment"]["phenotype"]["metabolite_level"])
    agreement = []
    for key in SHARED_SINGLE_KEYS:
        pairs = [
            (float(mul_by_gene[g][key]), float(lv[key]))
            for g, lv in coo_by_gene.items()
            if g in mul_by_gene and key in lv
        ]
        frame = pd.DataFrame(pairs, columns=["mulleder", "cooper"])
        rho = float(frame["mulleder"].corr(frame["cooper"], method="spearman"))
        agreement.append({"key": key, "n": len(pairs), "spearman": rho})
    summary["agreement"] = agreement

    # Cooper's own replicate strains: the ledger's later rows of a duplicated identifier
    # against the kept row they duplicate (``preprocess/data.csv`` by ``row_index``).
    kept_rows = pd.read_csv(
        Path(data_root, DATASETS[COOPER], "preprocess", "data.csv")
    ).set_index("row_index")
    dup_rows = ledger["duplicate_strain_rows"]
    replicate = []
    for key in SHARED_SINGLE_KEYS:
        pairs = [
            (float(kept_rows.loc[d["kept_row_index"], key]), float(d["levels"][key]))
            for d in dup_rows
            if key in d["levels"]
            and not pd.isna(kept_rows.loc[d["kept_row_index"], key])
        ]
        frame = pd.DataFrame(pairs, columns=["kept", "duplicate"])
        replicate.append(
            {
                "key": key,
                "n": len(pairs),
                "spearman": float(
                    frame["kept"].corr(frame["duplicate"], method="spearman")
                ),
            }
        )
    summary["cooper_duplicate_agreement"] = replicate

    # Cachera deletions of a gene the Btx cassette also carries: GenotypeAggregator keys
    # on the set of systematic names, so such a record's key is the cassette's own set.
    cassette = {
        p["systematic_gene_name"]
        for p in records[CACHERA][0]["experiment"]["genotype"]["perturbations"]
        if p["perturbation_type"] == "gene_addition"
    }
    summary["cachera_cassette_genes"] = sorted(cassette)
    summary["cachera_deletions_of_cassette_genes"] = sorted(
        (
            deletion(r["experiment"])[1]["systematic_gene_name"],
            deletion(r["experiment"])[1]["perturbed_gene_name"],
        )
        for r in records[CACHERA]
        if deletion(r["experiment"])[1]["systematic_gene_name"] in cassette
    )
    summary["cooper_genes_with_two_strains"] = len(records[COOPER]) - len(coo_by_gene)

    # ---- served vs dev
    query = json.loads(QUERY_JSON.read_text())
    release = query["served_release"]
    cache = read_query_cache(osp.join(data_root, QUERY_CACHE))
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    gene_set = set(genome.gene_set)
    assert len(gene_set) == query["gene_set_size"], (
        "gene_set differs from the query job"
    )
    summary["served_vs_dev"] = {
        name: served_vs_dev(
            name,
            records[name],
            [e for e in cache if e["experiment"]["dataset_name"] == name],
            release["datasets"][name]["content_sha256"],
            release["datasets"][name]["n_experiments"],
            gene_set,
        )
        for name in DATASETS
    }
    summary["query_slurm_job_id"] = query["slurm_job_id"]
    summary["served_release"] = release["release"]

    # ---- figures
    figs = {
        "mulleder_levels": figure_mulleder_levels(mul_levels, mul_ref, image_dir),
        "cachera_score": figure_cachera_histogram(scores, colonies, image_dir),
        "agreement": figure_agreement(
            agreement, summary["cooper_duplicate_agreement"], image_dir
        ),
    }
    for paths in figs.values():
        shutil.copy2(paths["svg"], FRAGMENT_DIR / osp.basename(paths["svg"]))
    summary["figures"] = figs

    # ---- record dumps: the same deletion in the three stores
    shared = sorted(mul & coo & cac)
    gene = shared[0]
    index = {
        name: next(
            i
            for i, r in enumerate(recs)
            if deletion(r["experiment"])[1]["systematic_gene_name"] == gene
        )
        for name, recs in records.items()
    }
    summary["record_gene"] = gene
    summary["record_indices"] = index
    for name, fname, extra in (
        (
            MULLEDER,
            "record_mulleder.md",
            [
                "experiment.environment.temperature.value",
                *[
                    f"experiment.phenotype.metabolite_level.{k}"
                    for k in MULLEDER_DUMP_KEYS
                ],
                *[f"experiment.phenotype.n_replicates.{k}" for k in MULLEDER_DUMP_KEYS],
                "experiment.phenotype.metabolite_level_se",
                *[
                    f"reference.phenotype_reference.metabolite_level.{k}"
                    for k in MULLEDER_DUMP_KEYS
                ],
            ],
        ),
        (
            COOPER,
            "record_cooper.md",
            [
                "experiment.environment.temperature",
                "experiment.environment.provenance_gaps[0].field",
                "experiment.environment.provenance_gaps[0].reason",
                "experiment.environment.duration_hours",
                *[
                    f"experiment.phenotype.metabolite_level.{k}"
                    for k in COOPER_DUMP_KEYS
                ],
                *[f"experiment.phenotype.n_replicates.{k}" for k in COOPER_DUMP_KEYS],
                "experiment.phenotype.metabolite_level_se",
                *[
                    f"reference.phenotype_reference.metabolite_level.{k}"
                    for k in COOPER_DUMP_KEYS
                ],
            ],
        ),
        (
            CACHERA,
            "record_cachera.md",
            [
                "experiment.environment.temperature.value",
                "experiment.phenotype.metabolite_level.betaxanthin",
                "experiment.phenotype.metabolite_level_se.betaxanthin",
                "experiment.phenotype.n_replicates.betaxanthin",
                "reference.phenotype_reference.metabolite_level.betaxanthin",
            ],
        ),
    ):
        rec = records[name][index[name]]
        d, _ = deletion(rec["experiment"])
        keep = [p.format(d=d) for p in COMMON_FIELDS]
        if name == CACHERA:
            # Every perturbation in index order, the cassette genes with their origin.
            genotype: list[str] = []
            for i, p in enumerate(rec["experiment"]["genotype"]["perturbations"]):
                prefix = f"experiment.genotype.perturbations[{i}]"
                genotype += [
                    f"{prefix}.systematic_gene_name",
                    f"{prefix}.perturbed_gene_name",
                    f"{prefix}.perturbation_type",
                ]
                if i != d:
                    genotype += [
                        f"{prefix}.source_organism",
                        f"{prefix}.integration_locus",
                    ]
                    if p["variant"] is not None:
                        genotype.append(f"{prefix}.variant")
            keep = [k for k in keep if ".genotype." not in k]
            keep[2:2] = genotype
        keep += extra
        # Group paths by their parent so the tree prints each parent once, in order.
        order = sorted(keep, key=lambda p: _path_order(p, keep))
        (FRAGMENT_DIR / fname).write_text(
            record_fragment(stamp, rec, tuple(order), index[name], name)
        )

    (FRAGMENT_DIR / "summary_tables.md").write_text(summary_fragment(stamp, summary))
    (FRAGMENT_DIR / "served_vs_dev.md").write_text(
        served_fragment(stamp, summary, release["release"])
    )
    (FRAGMENT_DIR / "figures.md").write_text(figures_fragment(stamp, summary))
    summary["cachera_sources"] = cachera_sources(data_root)
    (FRAGMENT_DIR / "cachera_sources.md").write_text(
        cachera_fragment(stamp, summary["cachera_sources"])
    )
    summary["provenance"] = provenance(data_root, records)
    (FRAGMENT_DIR / "provenance.md").write_text(
        provenance_fragment(stamp, summary["provenance"])
    )
    RESULTS_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("datasets", "overlap", "agreement", "record_gene")
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                n: {k: v for k, v in c.items() if k != "differing_path_examples"}
                for n, c in summary["served_vs_dev"].items()
            },
            indent=2,
        )
    )
    print("finished")


def _path_order(path: str, keep: list[str]) -> tuple[int, ...]:
    """Sort key placing each path after the first path sharing each of its prefixes."""
    parts = path.split(".")
    key: list[int] = []
    for depth in range(1, len(parts) + 1):
        prefix = ".".join(parts[:depth])
        key.append(
            next(
                i
                for i, p in enumerate(keep)
                if p == prefix or p.startswith(prefix + ".")
            )
        )
    return tuple(key)


if __name__ == "__main__":
    main()
