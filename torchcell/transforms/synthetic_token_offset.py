# torchcell/transforms/synthetic_token_offset
# [[torchcell.transforms.synthetic_token_offset]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/transforms/synthetic_token_offset
# Test file: tests/torchcell/transforms/test_synthetic_token_offset.py

"""A fictitious second source dataset whose values sit a known offset above the real ones.

The smoke test of the per-entry dataset token (plan.030-per-entry-dataset-token,
decision 10). Every entry row of the configured labels is cloned once: the clone keeps
the genotype and the label, its value is the original plus ``delta`` (in whatever scale
the values are in when this transform runs, so after the normalizer it is a normalized
offset), and its source token is ``token_index``, a vocabulary slot that no real dataset
holds. A readout that reads the token predicts the clone ``delta`` above the original;
one that cannot must split the difference. The control run (``control=True``) writes
the clones under their ORIGINAL token, which is exactly the case the token exists to
resolve: two targets for one genotype under one source, and a loss floor of
``delta**2 / 4`` per row for any predictor.

The clone's ``phenotype_values_original`` is the denormalized clone value when a
normalizer is given, so original-scale metrics see the offset in label units.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.transforms import BaseTransform

from torchcell.transforms.coo_regression_to_classification import (
    COOLabelNormalizationTransform,
)


class SyntheticTokenOffset(BaseTransform):  # type: ignore[misc]  # BaseTransform is Any (torch_geometric untyped)
    """Clone the entry rows of ``labels`` under a synthetic token, ``delta`` higher."""

    def __init__(
        self,
        delta: float,
        token_index: int,
        labels: list[str],
        normalizer: COOLabelNormalizationTransform | None = None,
        control: bool = False,
    ) -> None:
        """Store the offset, the synthetic token and the labels to clone.

        Args:
            delta: Added to every cloned value, in the scale the values carry when the
                transform runs (normalized units after ``COOLabelNormalizationTransform``).
            token_index: The clone's ``phenotype_dataset_indices`` value; must be a slot
                the model's ``dataset_token.vocab_size`` covers and no real dataset uses.
            labels: Phenotype label names whose rows are cloned.
            normalizer: When given, the clone's original-scale value is
                ``normalizer.denormalize(value + delta)``; otherwise the offset is added
                to the original-scale value as well.
            control: Keep the clone under its original token (the no-token control).
        """
        super().__init__()
        if delta == 0:
            raise ValueError("delta must be non-zero")
        if token_index < 0:
            raise ValueError("token_index must be a vocabulary slot")
        if not labels:
            raise ValueError("labels must name at least one phenotype")
        self.delta = float(delta)
        self.token_index = int(token_index)
        self.labels = list(labels)
        self.normalizer = normalizer
        self.control = bool(control)

    def forward(self, data: HeteroData | Batch) -> HeteroData | Batch:
        """Append the offset clones to every phenotype tensor of ``data['gene']``."""
        gene = data["gene"]
        if not hasattr(gene, "phenotype_dataset_indices"):
            raise ValueError(
                "SyntheticTokenOffset needs phenotype_dataset_indices; build the dataset "
                "with Perturbation(dataset_vocabulary=...)"
            )
        types = gene.phenotype_types
        if isinstance(types[0], list):
            types = types[0]
        values = gene.phenotype_values
        type_idx = gene.phenotype_type_indices
        label_slots = torch.tensor(
            [types.index(label) for label in self.labels if label in types],
            dtype=type_idx.dtype,
        )
        sel = torch.isin(type_idx, label_slots) & ~torch.isnan(values)
        if not bool(sel.any()):
            return data

        clone_values = values[sel] + self.delta
        tokens = gene.phenotype_dataset_indices
        clone_tokens = (
            tokens[sel]
            if self.control
            else torch.full_like(tokens[sel], self.token_index)
        )
        gene.phenotype_values = torch.cat([values, clone_values])
        gene.phenotype_type_indices = torch.cat([type_idx, type_idx[sel]])
        gene.phenotype_sample_indices = torch.cat(
            [gene.phenotype_sample_indices, gene.phenotype_sample_indices[sel]]
        )
        gene.phenotype_dataset_indices = torch.cat([tokens, clone_tokens])
        # Which rows are clones, so the smoke check can tell them from the originals in
        # the control, where the token no longer can.
        gene.phenotype_synthetic = torch.cat(
            [
                torch.zeros(values.numel(), dtype=torch.bool),
                torch.ones(int(sel.sum()), dtype=torch.bool),
            ]
        )
        if hasattr(gene, "phenotype_values_original"):
            original = gene.phenotype_values_original
            if self.normalizer is None:
                clone_original = original[sel] + self.delta
            else:
                clone_original = clone_values.clone()
                for label in self.labels:
                    if label not in types:
                        continue
                    mask = type_idx[sel] == types.index(label)
                    clone_original[mask] = self.normalizer.denormalize(
                        clone_values[mask], label
                    )
            gene.phenotype_values_original = torch.cat([original, clone_original])
        return data

    def __repr__(self) -> str:
        """Describe the offset, token and mode."""
        return (
            f"{self.__class__.__name__}(delta={self.delta}, token_index="
            f"{self.token_index}, labels={self.labels}, control={self.control})"
        )
