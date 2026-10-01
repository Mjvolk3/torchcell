# experiments/003-fit-int/scripts/soft_label_underflow_audit.py
# [[experiments.003-fit-int.scripts.soft_label_underflow_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/003-fit-int/scripts/soft_label_underflow_audit.py
"""Audit the soft-label underflow fix of #529 on the real 003-fit-int label_df.

The soft-label configs of 003-fit-int (32 bins, equal_frequency, label_type soft, sigma
scale 3, with the minmax normalizer the scripts apply) used to build each row as
exp(-0.5 (d / sigma)^2) normalized only when its sum was positive, so a row whose weights
all underflowed in float32 stayed all zeros. The transform now takes a softmax of the
log-weights. This script counts, per label, the labeled rows that were all-zero under
the old rule, the bins they fall in, the row sums and smallest peak probability under
the new rule, and the duplicate edges (zero-width bins, now refused for soft labels).

Then it runs the two losses those configs train with on an all-zero target row:
``CombinedCELoss`` (loss_type "ce") and the entropy regularizer inside
``MseCategoricalEntropyRegLoss`` (loss_type "mse_entropy_reg").

Run from the repo root: python experiments/003-fit-int/scripts/soft_label_underflow_audit.py
"""

import os
import os.path as osp
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from torchcell.losses.multi_dim_nan_tolerant import (
    CategoricalEntropyRegLoss,
    CombinedCELoss,
    MultiDimNaNTolerantCELoss,
)
from torchcell.transforms.regression_to_classification import (
    EqualFrequencyStrategy,
    LabelBinningTransform,
    LabelNormalizationTransform,
)

NUM_BINS = 32
SIGMA_SCALE = 3.0


def old_soft_rows(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """The pre-#529 rule: exp weights, divided by the sum only when the sum is > 0."""
    centers = (edges[1:] + edges[:-1]) / 2
    sigma = torch.min(edges[1:] - edges[:-1]) * SIGMA_SCALE
    clamped = torch.clamp(values, min=edges[0], max=edges[-1])
    weights = torch.exp(-0.5 * ((clamped[:, None] - centers[None, :]) / sigma) ** 2)
    sums = weights.sum(dim=1, keepdim=True)
    return torch.where(sums > 0, weights / torch.where(sums > 0, sums, 1.0), weights)


def audit_labels(label_df: pd.DataFrame) -> None:
    """Print the underflow counts per label, without and with the minmax normalizer."""
    labels = [c for c in label_df.columns if c != "index"]
    # stands in for the Neo4jCellDataset: the transforms read only label_df
    dataset: Any = SimpleNamespace(label_df=label_df)
    bin_config = {
        "num_bins": NUM_BINS,
        "strategy": "equal_frequency",
        "label_type": "soft",
    }
    for use_norm in (False, True):
        norm = (
            LabelNormalizationTransform(
                dataset, {c: {"strategy": "minmax"} for c in labels}
            )
            if use_norm
            else None
        )
        binning = LabelBinningTransform(
            dataset, {c: dict(bin_config) for c in labels}, norm
        )
        for label in labels:
            edges = torch.tensor(
                binning.label_metadata[label]["bin_edges"], dtype=torch.float
            )
            raw = label_df[label].replace([np.inf, -np.inf], np.nan).to_numpy()
            values = torch.tensor(raw, dtype=torch.float)
            if norm is not None:
                values = norm.normalize(values, label)
            labeled = ~torch.isnan(values)
            old = old_soft_rows(values[labeled], edges)
            new = EqualFrequencyStrategy().compute_soft_labels(
                values[labeled], edges, "equal_frequency", SIGMA_SCALE
            )
            zero = old.sum(dim=1) == 0
            n_zero = int(zero.sum())
            n_labeled = int(labeled.sum())
            widths = edges[1:] - edges[:-1]
            print(
                f"minmax={use_norm} {label}: labeled {n_labeled}, all-zero before "
                f"{n_zero} ({100 * n_zero / n_labeled:.2f}%), duplicate edges "
                f"{int((widths == 0).sum())}, min width {widths.min().item():.4g}"
            )
            if n_zero:
                bins = torch.bucketize(values[labeled][zero], edges[1:-1], right=True)
                found, counts = torch.unique(bins, return_counts=True)
                print(
                    f"  bins {found.tolist()} counts {counts.tolist()}; widths of "
                    f"bins 0 and {NUM_BINS - 1}: {widths[0].item():.4g}, "
                    f"{widths[-1].item():.4g}; new row sums in "
                    f"[{new[zero].sum(1).min().item():.7f}, "
                    f"{new[zero].sum(1).max().item():.7f}], smallest peak "
                    f"{new[zero].max(1).values.min().item():.4f}"
                )


def audit_losses() -> None:
    """Run both 003 losses on a batch whose second row has an all-zero task-1 target."""
    torch.manual_seed(0)
    logits = torch.randn(2, 8, requires_grad=True)
    dist = torch.tensor([0.1, 0.2, 0.3, 0.4])
    targets = torch.stack([torch.cat([dist, dist]), torch.cat([dist, torch.zeros(4)])])
    dim_losses, mask = MultiDimNaNTolerantCELoss(4, 2)(logits, targets)
    log_probs = torch.log_softmax(logits.view(2, 2, 4), dim=-1)
    row_ce = -(targets.view(2, 2, 4) * log_probs).sum(dim=-1)
    total, _ = CombinedCELoss(4, 2)(logits, targets)
    total.backward()
    assert logits.grad is not None
    print(f"CE: zero row counted valid: {bool(mask[1, 1])}; its CE {row_ce[1, 1]:.1f}")
    print(
        f"CE: task-1 mean {dim_losses[1].item():.7f} = row-0 CE / 2 = "
        f"{row_ce[0, 1].item() / 2:.7f} (the zero row dilutes the mean)"
    )
    print(f"CE: gradient on the zero row's logits {logits.grad[1, 4:].tolist()}")
    eps_row = torch.zeros(1, 4) + 1e-10
    print(
        "entropy reg: a zero row enters the KL distances as "
        f"{(eps_row / eps_row.sum(1, keepdim=True)).tolist()[0]} (uniform)"
    )
    reg = CategoricalEntropyRegLoss(lambda_d=1.0, lambda_t=1.0, num_classes=4)
    feats = torch.randn(3, 5)
    cat = torch.stack(
        [
            torch.cat([dist, dist]),
            torch.cat([dist, torch.zeros(4)]),
            torch.cat([dist.flip(0), dist.flip(0)]),
        ]
    )
    reg_total, _, _ = reg(feats, cat, torch.tensor([True, True, True]))
    print(
        "entropy reg: total finite "
        f"{bool(torch.isfinite(reg_total))}; the zero row adds no tightness term "
        "(read from the class_prob > 0 guard, not measured)"
    )


def main() -> None:
    """Load the 001-small-build label_df and run both audits."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    path = osp.join(
        data_root,
        "data/torchcell/experiments/003-fit-int/001-small-build/processed/"
        "label_df.parquet",
    )
    audit_labels(pd.read_parquet(path))
    audit_losses()


if __name__ == "__main__":
    main()
