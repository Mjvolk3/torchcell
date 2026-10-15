# torchcell/transforms/synthetic_token_offset
# [[torchcell.transforms.synthetic_token_offset]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/transforms/synthetic_token_offset
# Test file: tests/torchcell/transforms/test_synthetic_token_offset.py

"""A fictitious twin of a source dataset whose values sit a known offset above it.

The smoke test of the per-entry dataset token (plan.030-per-entry-dataset-token,
decision 10). Every entry row of the configured labels whose source token is a key of
``token_map`` is cloned once: the clone keeps the genotype and the label, its value is
the original plus ``delta`` (in whatever scale the values are in when this transform
runs, so after the normalizer it is a normalized offset), and its source token is the
key's twin, a vocabulary slot no real dataset holds. A readout that reads the token
predicts the clone ``delta`` above the original; one that cannot must split the
difference.

One twin PER SOURCE, not one synthetic token for everything. A single synthetic token
over every source learns one global shift, so on rows of a source whose mean sits far
from the global one (the Kuzmin 2018 triples at -0.87 normalized against a training mean
of 0) the difference from the own token is ``delta`` plus the source's residual gap, not
``delta``, until the encoder has absorbed every source mean (GilaHyper jobs 2863 and
2866 measured 0.79 to 0.93 for a 0.3 offset). A twin's rows are the source's rows
shifted, so its optimal bias is the source's plus ``delta`` at any training stage.

The control (``control=True``) writes the clones under their ORIGINAL token: two targets
for one genotype under one source, which no readout can separate.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.transforms import BaseTransform

from torchcell.transforms.coo_regression_to_classification import (
    COOLabelNormalizationTransform,
)


class SyntheticTokenOffset(BaseTransform):  # type: ignore[misc]  # BaseTransform is Any (torch_geometric untyped)
    """Clone the entry rows of ``labels`` under each source's twin token, ``delta`` higher."""

    def __init__(
        self,
        delta: float,
        token_map: Mapping[int, int],
        labels: list[str],
        normalizer: COOLabelNormalizationTransform | None = None,
        control: bool = False,
    ) -> None:
        """Store the offset, the source-to-twin map and the labels to clone.

        Args:
            delta: Added to every cloned value, in the scale the values carry when the
                transform runs (normalized units after ``COOLabelNormalizationTransform``).
            token_map: ``{source token index: twin token index}``; rows of other tokens
                are not cloned. Twins must be slots the model's
                ``dataset_token.vocab_size`` covers and no real dataset uses.
            labels: Phenotype label names whose rows are cloned.
            normalizer: When given, the clone's original-scale value is
                ``normalizer.denormalize(value + delta)``; otherwise the offset is added
                to the original-scale value as well.
            control: Keep the clone under its original token (the no-token control).
        """
        super().__init__()
        if delta == 0:
            raise ValueError("delta must be non-zero")
        if not token_map:
            raise ValueError("token_map must name at least one source")
        if any(k < 0 or v < 0 for k, v in token_map.items()):
            raise ValueError("token indices must be vocabulary slots")
        if set(token_map) & set(token_map.values()):
            raise ValueError("a twin slot cannot also be a source")
        if not labels:
            raise ValueError("labels must name at least one phenotype")
        self.delta = float(delta)
        self.token_map = {int(k): int(v) for k, v in token_map.items()}
        self.labels = list(labels)
        self.normalizer = normalizer
        self.control = bool(control)
        size = max(max(self.token_map), max(self.token_map.values())) + 1
        lookup = torch.full((size,), -1, dtype=torch.long)
        for src, twin in self.token_map.items():
            lookup[src] = twin
        self._lookup = lookup

    def twin_of(self, tokens: torch.Tensor) -> torch.Tensor:
        """Twin index per token, ``-1`` where the token has no twin."""
        out = torch.full_like(tokens, -1)
        inside = tokens < self._lookup.numel()
        out[inside] = self._lookup.to(tokens.device)[tokens[inside]]
        return out

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
        tokens = gene.phenotype_dataset_indices
        label_slots = torch.tensor(
            [types.index(label) for label in self.labels if label in types],
            dtype=type_idx.dtype,
        )
        twins = self.twin_of(tokens)
        sel = torch.isin(type_idx, label_slots) & ~torch.isnan(values) & (twins >= 0)
        if not bool(sel.any()):
            gene.phenotype_synthetic = torch.zeros(values.numel(), dtype=torch.bool)
            return data

        clone_values = values[sel] + self.delta
        clone_tokens = tokens[sel] if self.control else twins[sel]
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
        """Describe the offset, the twin map and the mode."""
        return (
            f"{self.__class__.__name__}(delta={self.delta}, token_map="
            f"{self.token_map}, labels={self.labels}, control={self.control})"
        )
