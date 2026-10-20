# experiments/040-inhibitor-synergy-wetlab/scripts/compound_embedding.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.compound_embedding]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/compound_embedding
r"""Every compound the mixture trainer can read, placed in two dimensions by its FCFP4
fingerprint, so a panel can show that the screens and this study share one compound table.

The compound table is one array: every InChIKey the four screens serve in the 040 store
plus this study's six inhibitors, by 2,048 FCFP4 counts computed from the same SMILES the
trainer featurizes a compound token from (``mixture_data.compound_smiles``). Until the
store move the rows came from the 031 ``fcfp4_count.npz`` (5,472 InChIKeys), which the
recipe reproduces exactly. The screens pick rows out of the table and so does this study,
so "one compound table" is a statement about which rows each source holds, and the figure
should show that rather than assert it.

The embedding is UMAP on the Tanimoto (Jaccard) distance between the fingerprints taken as
bit sets, which is the similarity a chemist reads between two FCFP4 vectors and the one
``nearest_vanacloig`` already uses to pick a neighbor. The counts are binarized first,
because Tanimoto is defined on the bit set; the count is what the model reads, the bit set
is what similarity is defined on, and the two differ only for a repeated substructure.

Two tables and a preview, under ``results/`` and ``ASSET_IMAGES_DIR``:

``compound_embedding.csv``          one row per InChIKey: the two coordinates, which of
                                    the four screens hold it, whether this study dosed it,
                                    the number of screens holding it, and a name when one
                                    is on file
``compound_embedding_summary.csv``  per source, the compounds it holds and the median
                                    pairwise Tanimoto inside it, plus the median Tanimoto
                                    from each of this study's six inhibitors to that source
``compound_embedding.{png,svg}``    the preview the note embeds

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \\
        experiments/040-inhibitor-synergy-wetlab/scripts/compound_embedding.py
"""

from __future__ import annotations

import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numpy.typing import NDArray

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from mixture_data import (  # noqa: E402
    SPECS,
    STORE,
    _condition_frame,
    compound_smiles,
    fcfp4_count,
    load_cells,
    wetlab_inhibitors,
)

load_dotenv()
EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
IMAGES = osp.join(os.environ["ASSET_IMAGES_DIR"], "040-inhibitor-synergy-wetlab")

#: the screens in the order the sheet lists them
SOURCE_ORDER = ("vanacloig", "hoepfner_hop", "hillenmeyer_het", "wildenhain")
#: how a source is named on a figure
SOURCE_LABEL = {
    "vanacloig": "Vanacloig 2022",
    "hoepfner_hop": "Hoepfner 2014 HOP",
    "hillenmeyer_het": "Hillenmeyer 2008 HET",
    "wildenhain": "Wildenhain 2015",
}
#: UMAP's settings, fixed so the figure is the same on every run
UMAP_KW = dict(n_neighbors=15, min_dist=0.10, metric="jaccard", random_state=0)


def source_inchikeys() -> tuple[dict[str, set[str]], dict[str, str]]:
    """The InChIKeys each screen holds, from the cells the trainer reads, and the SMILES
    the trainer fingerprints each from (``mixture_data.compound_smiles``).
    """
    cells = load_cells(STORE, frozenset(SPECS[n].dataset for n in SOURCE_ORDER))
    out: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    served: dict[str, str | None] = {}
    for name in SOURCE_ORDER:
        frame = _condition_frame(SPECS[name], cells)
        out[name] = set(frame["inchikeys"])
        for key, cname, smiles in zip(
            frame["inchikeys"], frame["compound_names"], frame["smiles"], strict=True
        ):
            names.setdefault(key, cname)
            served.setdefault(key, smiles or None)
    ours = wetlab_inhibitors()
    for cname, key in ours.items():
        names.setdefault(key, cname)
    smiles_of = compound_smiles(sorted(names), served, names)
    return out, smiles_of


def tanimoto_rows(
    bits: NDArray[np.bool_], against: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Tanimoto of every row of ``bits`` against every row of ``against``."""
    a = bits.astype(np.float32)
    b = against.astype(np.float32)
    inter = a @ b.T
    size_a = a.sum(axis=1)[:, None]
    size_b = b.sum(axis=1)[None, :]
    union = size_a + size_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0).astype(
        np.float64
    )


def median_within(bits: NDArray[np.bool_]) -> float:
    """Median Tanimoto over the distinct pairs of ``bits``; NaN under two rows."""
    if len(bits) < 2:
        return float("nan")
    sim = tanimoto_rows(bits, bits)
    iu = np.triu_indices(len(bits), k=1)
    return float(np.median(sim[iu]))


def main() -> None:
    """Embed the compound table, write the two tables and the preview."""
    # the compound table exactly as the trainer builds it: every compound the four
    # screens serve plus the six wet-lab inhibitors, FCFP4 counts from SMILES
    held, smiles_of = source_inchikeys()
    ours = wetlab_inhibitors()
    keys: NDArray[np.str_] = np.array(sorted(smiles_of), dtype=np.str_)
    counts: NDArray[np.float32] = np.stack([fcfp4_count(smiles_of[k]) for k in keys])
    assert len(keys) == len(counts), "the fingerprint table is ragged"
    index = {k: i for i, k in enumerate(keys)}
    bits = counts > 0

    from umap import UMAP

    xy = UMAP(n_components=2, **UMAP_KW).fit_transform(bits)

    frame = pd.DataFrame({"inchikey": keys, "x": xy[:, 0], "y": xy[:, 1]})
    for name in SOURCE_ORDER:
        frame[name] = frame.inchikey.isin(held[name])
    frame["n_screens"] = frame[list(SOURCE_ORDER)].sum(axis=1)
    name_of = {k: n for n, k in ours.items()}
    frame["this_study"] = frame.inchikey.isin(set(ours.values()))
    frame["compound"] = frame.inchikey.map(name_of)
    frame.to_csv(osp.join(RESULTS, "compound_embedding.csv"), index=False)

    rows = []
    our_bits = bits[[index[k] for k in ours.values()]]
    for name in SOURCE_ORDER:
        rows_in = [index[k] for k in held[name] if k in index]
        sim = tanimoto_rows(our_bits, bits[rows_in])
        rows.append(
            {
                "source": name,
                "label": SOURCE_LABEL[name],
                "n_compounds": len(held[name]),
                "n_with_fingerprint": len(rows_in),
                "median_tanimoto_within": median_within(bits[rows_in]),
                "median_tanimoto_to_this_study": float(np.median(sim)),
                "max_tanimoto_to_this_study": float(sim.max()),
            }
        )
    rows.append(
        {
            "source": "this_study",
            "label": "this study",
            "n_compounds": len(ours),
            "n_with_fingerprint": len(ours),
            "median_tanimoto_within": median_within(our_bits),
            "median_tanimoto_to_this_study": 1.0,
            "max_tanimoto_to_this_study": 1.0,
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(osp.join(RESULTS, "compound_embedding_summary.csv"), index=False)
    print(summary.to_string(index=False))

    preview(frame)


def preview(frame: pd.DataFrame) -> None:
    """A standalone look at the embedding, for the note."""
    import matplotlib.pyplot as plt

    from torchcell.utils import (
        PLOT_PALETTE,
        PLOT_PALETTE_FILL,
        apply_paper_style,
        mm_to_in,
    )

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(mm_to_in(88), mm_to_in(80)))
    ax.scatter(
        frame.x,
        frame.y,
        s=2.0,
        color="#CCCCCC",
        linewidth=0,
        label=f"the compound table ({len(frame):,})",
    )
    # largest source first, so the small screens are not buried under Wildenhain
    order = sorted(SOURCE_ORDER, key=lambda n: -int(frame[n].sum()))
    # the figure panel's colors: Wildenhain, the bulk, pale; the small screens dark
    color_of = {
        "vanacloig": PLOT_PALETTE[7],
        "hoepfner_hop": PLOT_PALETTE[8],
        "hillenmeyer_het": PLOT_PALETTE[11],
        "wildenhain": PLOT_PALETTE_FILL[17],
    }
    for name in order:
        sub = frame[frame[name]]
        ax.scatter(
            sub.x,
            sub.y,
            s=2.0 if name == "wildenhain" else 6.0,
            color=color_of[name],
            linewidth=0,
            label=f"{SOURCE_LABEL[name]} ({len(sub):,})",
        )
    ours = frame[frame.this_study]
    ax.scatter(
        ours.x,
        ours.y,
        s=12.0,
        facecolor="none",
        edgecolor="black",
        linewidth=0.7,
        label="this study's six",
        zorder=5,
    )
    for n, r in enumerate(ours.sort_values("y", ascending=False).itertuples()):
        ax.annotate(
            r.compound,
            (r.x, r.y),
            xytext=(8, 10 - 9 * (n % 3)),
            textcoords="offset points",
            fontsize=5,
            arrowprops=dict(arrowstyle="-", linewidth=0.3, color="black"),
        )
    ax.set_xlabel("UMAP 1, Tanimoto on FCFP4 bits")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", frameon=False, fontsize=5, markerscale=2)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(osp.join(IMAGES, f"compound_embedding.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
