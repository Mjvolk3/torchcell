# experiments/032-amino-acid-module/scripts/isobutanol_amino_acid_correlation.py
# [[experiments.032-amino-acid-module.scripts.isobutanol_amino_acid_correlation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/032-amino-acid-module/scripts/isobutanol_amino_acid_correlation

r"""Does a deletion's isobutanol biosensor level correlate with its amino-acid pool?

The same question the betaxanthin read asks, on the third endpoint: Montano Lopez 2024
measured a branched-chain-amino-acid biosensor's fold change in every strain of the
knockout collection, and Mulleder 2016 measured nineteen free amino-acid concentrations
in the same collection, so every gene both screens hold gives one paired observation.

The biosensor reports branched-chain amino acids by construction (yeast-enhanced GFP
under a Leu3p-responsive promoter), so a correlation with leucine, isoleucine and valine
is close to mechanically expected; what the read decides is whether anything beyond
those three carries signal, and whether the pool as a whole predicts the endpoint better
than the deleted gene's annotation does.

Three reads, on the strains both screens measured:

* per amino acid: the Spearman of the isobutanol fold change against the amino-acid
  value, with a strain bootstrap for the interval and the overlap of the top 5 percent
  of strains on the two axes against the count expected by chance;
* the pool as a whole: ridge from the standardized log amino-acid profile to the log10
  fold change, out of fold over five strain folds, beside the same ridge from the
  deleted gene's propagated GO multi-hot and from GO with the pool appended, so the
  pool's contribution beyond the gene's identity is read directly;
* the ceiling each read is bounded by: the geometric mean of the two endpoints'
  reliabilities, the isobutanol screen's own duplicate-strain agreement (its within-screen
  repeat, not the selected re-screen panel, which is chosen on effect size) and
  Mulleder's per-amino-acid retest.

The fold change is a ratio to the plate's wild type and is right-skewed, so it is read as
log10 throughout, which is also how the screen's own reliability was computed.

Writes ``results/isobutanol_amino_acid_correlation.json`` and a one-panel figure to
``ASSET_IMAGES_DIR/032-amino-acid-module/``.

Run from the repo root::

    PYTHONPATH=$PWD python \
        experiments/032-amino-acid-module/scripts/isobutanol_amino_acid_correlation.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
import sys
from typing import Any

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel  # noqa: E402

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "032-amino-acid-module", "scripts"))
from betaxanthin_amino_acid_correlation import (  # noqa: E402
    CACHERA,
    GO_STORE,
    IMAGES,
    MULLEDER,
    MULLEDER_KEYS,
    MULLEDER_RETEST,
    N_FOLDS,
    RESULTS,
    FeatureStat,
    PoolFit,
    feature_stat,
    manifest,
    pool_fit,
    short,
)

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

LOPEZ = osp.join(DATA_ROOT, "data/torchcell/isobutanol_screen_lopez2024")
RELIABILITY_027 = osp.join(
    EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results", "target_reliability.json"
)
IB_RELIABILITY = osp.join(
    EXPERIMENT_ROOT,
    "027-betaxanthin-metabolic",
    "results",
    "isobutanol_retest_reliability.json",
)


class Report(BaseModel):
    """Manifests, shared counts, per-amino-acid statistics and the pool fits."""

    lopez_manifest: dict[str, Any]
    mulleder_manifest: dict[str, Any]
    n_lopez: int
    n_shared: int
    isobutanol_reliability: float
    isobutanol_reliability_source: str
    #: Strains the screen shares with Cachera's betaxanthin score.
    n_shared_betaxanthin: int
    features: list[FeatureStat]
    #: The betaxanthin score against the biosensor, the third pair of the endpoint matrix.
    betaxanthin: FeatureStat
    pool: list[PoolFit]
    figure: str
    note: str


def isobutanol_by_gene() -> dict[str, float]:
    """Map each gene to the strain's isobutanol biosensor fold change, from the LMDB.

    One record per strain and one deleted gene per record (asserted), so the map is the
    screen as released; a gene measured twice would raise rather than be averaged.
    """
    env = lmdb.open(
        osp.join(LOPEZ, "processed", "lmdb"), readonly=True, lock=False, subdir=True
    )
    out: dict[str, float] = {}
    with env.begin() as txn:
        for _, raw in txn.cursor():
            rec = pickle.loads(raw)
            exp = rec["experiment"]
            perts = exp["genotype"]["perturbations"]
            assert len(perts) == 1, perts
            gene = str(perts[0]["systematic_gene_name"])
            level = exp["phenotype"]["metabolite_level"]["isobutanol"]
            assert gene not in out, f"{gene} twice in the released screen"
            out[gene] = float(level)
    env.close()
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    ib = isobutanol_by_gene()
    mul = pd.read_csv(osp.join(MULLEDER, "preprocess", "data.csv")).set_index("orf")
    assert not mul.index.duplicated().any()
    with open(IB_RELIABILITY) as fh:
        ib_rel_all = json.load(fh)
    # The WITHIN-screen duplicate agreement, not the re-screen panel: the panel is
    # selected on |log fold change|, so its agreement is not the screen's reliability.
    ib_rel = float(ib_rel_all["within_first_screen"]["pairs"]["spearman_log10"])
    with open(MULLEDER_RETEST) as fh:
        mul_rel = {
            row["amino_acid"]: float(row["spearman_retest_log10"])
            for row in json.load(fh)["per_amino_acid"]
        }
    shared = sorted(set(ib) & set(mul.index))
    print(
        f"Lopez {len(ib)} strains; shared with Mulleder {len(shared)}; "
        f"isobutanol reliability {ib_rel:.3f} (within-screen duplicates)"
    )
    y = np.log10(np.array([ib[g] for g in shared], dtype=float))
    assert np.isfinite(y).all(), "a non-finite log10 fold change"
    aa = mul.loc[shared, MULLEDER_KEYS].to_numpy(float)

    features = [
        feature_stat("mulleder2016", key, y, aa[:, j], mul_rel[key], ib_rel, rng)
        for j, key in enumerate(MULLEDER_KEYS)
    ]
    for s in features:
        ceiling = "" if s.ceiling is None else f" ceiling {s.ceiling:.2f}"
        print(
            f"  {s.key:22s} n {s.n:5d} rho {s.spearman:+.3f} "
            f"[{s.ci_low:+.3f}, {s.ci_high:+.3f}] top5 "
            f"{s.top5_overlap}/{s.top5_expected:.1f} (x{s.top5_enrichment:.1f}){ceiling}"
        )

    go_store = np.load(GO_STORE)
    go_rows = {str(g): i for i, g in enumerate(go_store["genes"])}
    go_all = np.asarray(go_store["v_deletion"], dtype=np.float32)
    go = np.zeros((len(shared), go_all.shape[1]), dtype=np.float32)
    for i, s in enumerate(shared):
        if s in go_rows:
            go[i] = go_all[go_rows[s]]
    print(
        f"  {int((go.sum(axis=1) > 0).sum())} of {len(shared)} strains carry a GO row"
    )

    folds = np.array_split(rng.permutation(y.size), N_FOLDS)
    pool = [
        pool_fit("lopez2024", "pool", y, aa, None, folds, rng),
        pool_fit("lopez2024", "go", y, None, go, folds, rng),
        pool_fit("lopez2024", "go+pool", y, aa, go, folds, rng),
    ]
    for p in pool:
        print(
            f"  {p.inputs:8s} n {p.n} features {p.n_features} "
            f"spearman_oof {p.spearman_oof:+.4f} pearson_oof {p.pearson_oof:+.4f} "
            f"permuted {np.mean(p.permuted_spearman_oof):+.4f}"
        )

    # THE THIRD PAIR OF THE ENDPOINT MATRIX (2026-09-27): the betaxanthin score against
    # the biosensor, the one pair the cross-endpoint reads had never covered, and the
    # number that says how much a joint model of the two could share.
    bx = pd.read_csv(osp.join(CACHERA, "preprocess", "data.csv")).set_index("orf")
    shared_bx = sorted(set(ib) & set(bx.index))
    y_bx = np.log10(np.array([ib[g] for g in shared_bx], dtype=float))
    bx_vals = bx.loc[shared_bx, "level"].to_numpy(float)
    # Cachera's own reliability, the stored variance-components value the sibling read uses.
    with open(RELIABILITY_027) as fh:
        bx_rel = float(
            next(t for t in json.load(fh)["targets"] if t["target"] == "Betaxanthin")[
                "reliability_stored_value"
            ]
        )
    betaxanthin = feature_stat(
        "cachera2023", "betaxanthin_score", y_bx, bx_vals, bx_rel, ib_rel, rng
    )
    print(
        f"  betaxanthin_score     n {betaxanthin.n:5d} rho {betaxanthin.spearman:+.3f} "
        f"[{betaxanthin.ci_low:+.3f}, {betaxanthin.ci_high:+.3f}] "
        f"ceiling {betaxanthin.ceiling:.2f}"
    )

    os.makedirs(IMAGES, exist_ok=True)
    # The repo's figure conventions: Arial 6 pt, real text in the SVG (svg.fonttype
    # "none", so the built PDF embeds a font rather than outlining the labels), a
    # standard panel width, and the true-size writer so the SVG imports at true mm.
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    fig = plt.figure(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(52.0)))
    ax = fig.add_axes([0.07, 0.34, 0.90, 0.60])
    order = np.argsort([f.spearman for f in features])
    xs = np.arange(len(order))
    ax.bar(
        xs,
        [features[i].spearman for i in order],
        yerr=[
            [features[i].spearman - features[i].ci_low for i in order],
            [features[i].ci_high - features[i].spearman for i in order],
        ],
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.5,
        error_kw={"linewidth": 0.5, "capsize": 1.5},
    )
    for x, i in zip(xs, order, strict=True):
        c = features[i].ceiling
        if c is not None:
            ax.plot([x - 0.4, x + 0.4], [c, c], color="black", linewidth=0.6)
            ax.plot([x - 0.4, x + 0.4], [-c, -c], color="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([short(features[i].key) for i in order], rotation=90, fontsize=6)
    ax.set_ylabel("Spearman, isobutanol vs amino acid", fontsize=6)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.tick_params(labelsize=6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    svg = osp.join(IMAGES, "isobutanol_amino_acid_correlation.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(svg.replace(".svg", ".png"), dpi=300)
    plt.close(fig)
    figure = svg

    report = Report(
        lopez_manifest=manifest(LOPEZ),
        mulleder_manifest=manifest(MULLEDER),
        n_lopez=len(ib),
        n_shared=len(shared),
        isobutanol_reliability=ib_rel,
        isobutanol_reliability_source="within_first_screen.pairs.spearman_log10",
        n_shared_betaxanthin=len(shared_bx),
        betaxanthin=betaxanthin,
        features=features,
        pool=pool,
        figure=figure,
        note=(
            "The Montano Lopez biosensor fold change (log10) against Mulleder's nineteen "
            "amino-acid concentrations on the strains both screens measured; the ceiling "
            "is the geometric mean of the screen's within-screen duplicate agreement and "
            "the amino acid's retest."
        ),
    )
    out = osp.join(RESULTS, "isobutanol_amino_acid_correlation.json")
    with open(out, "w") as fh:
        fh.write(report.model_dump_json(indent=1))
    print(out)


if __name__ == "__main__":
    main()
