# experiments/028-gene-essentiality/scripts/fba_background_probe.py
# [[experiments.028-gene-essentiality.scripts.fba_background_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/fba_background_probe

r"""Does the deletion collection's strain background explain the essential genes FBA passes?

``consensus_missed.py`` finds that the essential genes every learned predictor ranks low
are enriched for an FBA growth ratio of 1.0 and for transport, and the document states a
hypothesis: some are essential only in the background the collection was made in. The
yeast deletion collection is BY4741, which carries ``his3``, ``leu2``, ``met15`` and
``ura3`` deletions, so the strain lives on imported histidine, leucine, methionine and
uracil. Every FBA column in this experiment deletes one gene from the wild-type model.

This probe deletes the four background genes first, takes that strain's growth as the
reference, and recomputes the single-gene knockout growth ratio for the 1,121 labeled
genes on YPD and on SC. It reports, per medium, the AUROC of one minus the ratio with and
without the background on all labeled genes and on the released 195, the number of genes
whose ratio moves by more than 0.01, and those genes by name with their label.

    PYTHONPATH=$PWD python experiments/028-gene-essentiality/scripts/fba_background_probe.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

import numpy as np
from cobra.flux_analysis import single_gene_deletion
from cobra.manipulation import knock_out_model_genes
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from fba_essentiality_baseline import (  # noqa: E402
    binary_auroc,
    medium_model,
    read_labels,
    read_our_genes,
)
from hard_genes import common_names  # noqa: E402

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
#: BY4741: his3-delta-1, leu2-delta-0, met15-delta-0, ura3-delta-0.
BACKGROUND = {
    "YOR202W": "HIS3",
    "YCL018W": "LEU2",
    "YLR303W": "MET15",
    "YEL021W": "URA3",
}
MEDIA = ("YPD", "SC")
#: MEASURED: the experiment's YPD opens no uracil uptake, so a ura3 deletion alone takes
#: the model's YPD growth from 0.5352 to zero, while BY4741 grows on YPD. The probe opens
#: the uracil exchange on YPD at the bound the medium gives its amino acids, for the
#: wild-type arm and the background arm alike, so the two differ only by the background.
URACIL_EXCHANGE = "r_2090"
URACIL_BOUND_FROM = "r_1893"
MOVED = 0.01
RATIO_DECIMALS = 6
PROCESSES = 4


class MovedGene(BaseModel):
    gene: str
    name: str
    essential: int
    ratio_wild_type: float
    ratio_background: float


class MediumReport(BaseModel):
    medium: str
    #: As the experiment defines the medium, before the probe opens uracil uptake.
    growth_as_defined: float
    growth_as_defined_ura3: float
    uracil_uptake_opened: bool
    growth_wild_type: float
    growth_background: float
    background_genes_in_model: list[str]
    auroc_labeled_wild_type: float
    auroc_labeled_background: float
    auroc_released_wild_type: float
    auroc_released_background: float
    n_essential_ratio_one_wild_type: int
    n_essential_ratio_one_background: int
    n_moved: int
    moved: list[MovedGene]


class Report(BaseModel):
    background: dict[str, str]
    n_labeled: int
    n_essential: int
    media: list[MediumReport]
    note: str


def ratios(model, genes: list[str]) -> tuple[float, dict[str, float]]:
    """Reference growth and gene -> knockout growth over it; a gene outside the model is 1."""
    reference = float(model.slim_optimize())
    assert reference > 1e-6, reference
    inside = [g for g in genes if g in model.genes]
    table = single_gene_deletion(model, gene_list=inside, processes=PROCESSES)
    out = {g: 1.0 for g in genes}
    for ids, growth in zip(table["ids"], table["growth"], strict=True):
        (gene,) = tuple(ids)
        value = 0.0 if growth is None or np.isnan(growth) else float(growth)
        # Solver noise near a ratio of 1.0 reorders ties between runs (the labeled AUROC
        # moved by 0.017 between two identical runs), so ratios are rounded first.
        out[gene] = round(max(value, 0.0) / reference, RATIO_DECIMALS)
    return reference, out


def main() -> None:
    labels, _ = read_labels()
    released = sorted(read_our_genes(labels))
    genes = sorted(labels)
    names = common_names(genes)
    y_all = np.array([labels[g] for g in genes])
    y_rel = np.array([labels[g] for g in released])
    reports = []
    for medium in MEDIA:
        model, _, _ = medium_model(medium)
        uracil = model.reactions.get_by_id(URACIL_EXCHANGE)
        growth_defined = float(model.slim_optimize())
        with model:
            knock_out_model_genes(model, ["YEL021W"])
            growth_defined_ura3 = max(float(model.slim_optimize()), 0.0)
        opened = uracil.lower_bound == 0
        if opened:
            uracil.lower_bound = model.reactions.get_by_id(
                URACIL_BOUND_FROM
            ).lower_bound
            print(f"{medium}: opened uracil uptake at {uracil.lower_bound}")
        growth_wt, wt = ratios(model, genes)
        present = [g for g in BACKGROUND if g in model.genes]
        knock_out_model_genes(model, present)
        growth_bg, bg = ratios(model, genes)

        def auroc(table: dict[str, float], subset: list[str], y: np.ndarray) -> float:
            return binary_auroc(-np.array([table[g] for g in subset]), y)

        moved = [
            MovedGene(
                gene=g,
                name=names[g],
                essential=labels[g],
                ratio_wild_type=wt[g],
                ratio_background=bg[g],
            )
            for g in genes
            if abs(wt[g] - bg[g]) > MOVED and g not in BACKGROUND
        ]
        moved.sort(key=lambda m: m.ratio_background - m.ratio_wild_type)
        report = MediumReport(
            medium=medium,
            growth_as_defined=growth_defined,
            growth_as_defined_ura3=growth_defined_ura3,
            uracil_uptake_opened=opened,
            growth_wild_type=growth_wt,
            growth_background=growth_bg,
            background_genes_in_model=present,
            auroc_labeled_wild_type=auroc(wt, genes, y_all),
            auroc_labeled_background=auroc(bg, genes, y_all),
            auroc_released_wild_type=auroc(wt, released, y_rel),
            auroc_released_background=auroc(bg, released, y_rel),
            n_essential_ratio_one_wild_type=sum(
                labels[g] == 1 and wt[g] >= 0.999 for g in genes
            ),
            n_essential_ratio_one_background=sum(
                labels[g] == 1 and bg[g] >= 0.999 for g in genes
            ),
            n_moved=len(moved),
            moved=moved,
        )
        reports.append(report)
        print(
            f"{medium}: growth {growth_wt:.4f} -> {growth_bg:.4f}; AUROC labeled "
            f"{report.auroc_labeled_wild_type:.4f} -> {report.auroc_labeled_background:.4f}; "
            f"released {report.auroc_released_wild_type:.4f} -> "
            f"{report.auroc_released_background:.4f}; essential at ratio 1.0 "
            f"{report.n_essential_ratio_one_wild_type} -> "
            f"{report.n_essential_ratio_one_background}; moved {report.n_moved}"
        )
        for m in moved:
            print(
                f"   {m.gene:9s} {m.name:8s} essential {m.essential} "
                f"{m.ratio_wild_type:.3f} -> {m.ratio_background:.3f}"
            )
    out = osp.join(RESULTS, "fba_background_probe.json")
    with open(out, "w") as fh:
        json.dump(
            Report(
                background=BACKGROUND,
                n_labeled=len(genes),
                n_essential=int(y_all.sum()),
                media=reports,
                note=(
                    "Ratio is knockout growth over the reference strain's growth in the "
                    "same medium; the background reference carries the four BY4741 "
                    "deletions. A gene outside the model has ratio 1."
                ),
            ).model_dump(),
            fh,
            indent=2,
        )
    print(out)


if __name__ == "__main__":
    main()
