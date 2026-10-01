# torchcell/transforms/coo_regression_to_classification
# [[torchcell.transforms.coo_regression_to_classification]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/transforms/coo_regression_to_classification
# Test file: tests/torchcell/transforms/test_coo_regression_to_classification.py

"""Transforms converting COO-format regression labels to classification targets."""

from abc import ABC
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.transforms import BaseTransform, Compose

if TYPE_CHECKING:
    from torchcell.data.neo4j_cell import Neo4jCellDataset

NORMALIZATION_STRATEGIES = ("minmax", "robust", "standard")
LABEL_TYPES = ("categorical", "ordinal", "soft")


class COOLabelNormalizationTransform(BaseTransform):  # type: ignore[misc]  # BaseTransform is Any (torch_geometric untyped)
    """Transform for normalizing labels in COO format with different strategies."""

    def __init__(
        self,
        dataset: "Neo4jCellDataset",
        label_configs: dict[str, dict[str, Any]],
        eps: float = 1e-8,
        fit_indices: Iterable[int] | None = None,
    ):
        """Compute per-label normalization statistics from the dataset.

        Args:
            dataset: Neo4jCellDataset instance
            label_configs: Dictionary mapping label names to their configurations
                Example:
                {
                    'gene_interaction': {
                        'strategy': 'standard',  # or 'minmax' or 'robust'
                    }
                }
            eps: Small constant to avoid division by zero
            fit_indices: Record indices the statistics are computed over. ``None`` uses
                every record, which is right when the dataset IS the arm's data and was
                the only behavior before.

                It stops mattering only on a single-population build. On 025 the same
                ``gene_interaction`` column holds 13,142,648 digenic and 376,732 trigenic
                values with different spreads: sd 0.0444 over the whole column against
                0.0633 over the triples alone. A trigenic arm fitted on the whole column
                divides its targets by the wrong constant, inflating every normalized
                label by 1.43x and with it the effective learning rate, so its loss curve
                is not comparable to a run whose statistics came from the triples. Nothing
                errors; the run simply is not the experiment it is labeled as.

                Indices name RECORDS, and ``label_df`` carries them in its ``index``
                COLUMN rather than positionally (row 2 is record 10), so they are resolved
                through that column.
        """
        super().__init__()
        self.label_configs = label_configs
        self.eps = eps
        self.stats = {}

        # Calculate statistics for each label
        df = dataset.label_df.replace([np.inf, -np.inf], np.nan)
        if fit_indices is not None:
            wanted = set(fit_indices)
            df = df[df["index"].isin(wanted)]
            missing = len(wanted) - len(df)
            if missing:
                raise ValueError(
                    f"fit_indices names {missing} record indices absent from label_df"
                )
        for label, config in label_configs.items():
            if config["strategy"] not in NORMALIZATION_STRATEGIES:
                raise ValueError(
                    f"Unknown normalization strategy {config['strategy']!r} for label "
                    f"{label!r}; valid strategies: {', '.join(NORMALIZATION_STRATEGIES)}"
                )
            if label not in df.columns:
                raise ValueError(f"Label {label} not found in dataset")

            values = cast(npt.NDArray[Any], df[label].dropna().values)
            stats = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "strategy": config["strategy"],
            }
            self.stats[label] = stats

    def normalize(self, values: torch.Tensor, label: str) -> torch.Tensor:
        """Normalize values based on specified strategy."""
        if torch.all(torch.isnan(values)):
            return values

        stats = self.stats[label]
        strategy = stats["strategy"]

        if strategy == "standard":
            return cast(
                torch.Tensor, (values - stats["mean"]) / (stats["std"] + self.eps)
            )
        elif strategy == "minmax":
            return cast(
                torch.Tensor,
                (values - stats["min"]) / (stats["max"] - stats["min"] + self.eps),
            )
        elif strategy == "robust":
            iqr = stats["q75"] - stats["q25"]
            return cast(torch.Tensor, (values - stats["q25"]) / (iqr + self.eps))
        else:
            raise ValueError(f"Unknown normalization strategy: {strategy}")

    def denormalize(self, values: torch.Tensor, label: str) -> torch.Tensor:
        """Denormalize values based on specified strategy."""
        if torch.all(torch.isnan(values)):
            return values

        stats = self.stats[label]
        strategy = stats["strategy"]

        if strategy == "standard":
            return cast(torch.Tensor, values * stats["std"] + stats["mean"])
        elif strategy == "minmax":
            return cast(
                torch.Tensor, values * (stats["max"] - stats["min"]) + stats["min"]
            )
        elif strategy == "robust":
            iqr = stats["q75"] - stats["q25"]
            return cast(torch.Tensor, values * iqr + stats["q25"])
        else:
            raise ValueError(f"Unknown normalization strategy: {strategy}")

    def forward(self, data: HeteroData | Batch) -> HeteroData | Batch:
        """Transform the data by normalizing specified labels in COO format."""
        # Check if we have phenotype data in COO format
        if not hasattr(data["gene"], "phenotype_values"):
            return data

        # Get phenotype types - handle batch case where it might be a list of lists
        phenotype_types = data["gene"].phenotype_types
        if (
            isinstance(phenotype_types, list)
            and phenotype_types
            and isinstance(phenotype_types[0], list)
        ):
            # In batch mode, all items should have the same phenotype types
            phenotype_types = phenotype_types[0]

        # Clone the phenotype values to ensure we can modify them
        new_phenotype_values = data["gene"].phenotype_values.clone()

        # Handle scalar tensors by ensuring at least 1D
        if new_phenotype_values.dim() == 0:
            new_phenotype_values = new_phenotype_values.unsqueeze(0)
            is_scalar = True
        else:
            is_scalar = False

        # Store original values if not already stored
        if not hasattr(data["gene"], "phenotype_values_original"):
            data["gene"].phenotype_values_original = data[
                "gene"
            ].phenotype_values.clone()

        # Process each configured label
        for label, config in self.label_configs.items():
            if label not in phenotype_types:
                continue

            # Find indices where this phenotype appears
            label_idx = phenotype_types.index(label)
            mask = data["gene"].phenotype_type_indices == label_idx

            if mask.sum() == 0:
                continue

            # Get values for this phenotype
            values = new_phenotype_values[mask]

            # Normalize and update values
            normalized_values = self.normalize(values, label)
            new_phenotype_values[mask] = normalized_values

        # Update the data with normalized values
        # Convert back to scalar if it was originally scalar
        if is_scalar:
            new_phenotype_values = new_phenotype_values.squeeze(0)
        data["gene"].phenotype_values = new_phenotype_values

        return data

    def inverse(self, data: HeteroData | Batch) -> HeteroData | Batch:
        """Inverse transform to recover original scale."""
        # Check if we have phenotype data in COO format
        if not hasattr(data["gene"], "phenotype_values"):
            return data

        # Get phenotype types - handle batch case where it might be a list of lists
        phenotype_types = data["gene"].phenotype_types
        if (
            isinstance(phenotype_types, list)
            and phenotype_types
            and isinstance(phenotype_types[0], list)
        ):
            # In batch mode, all items should have the same phenotype types
            phenotype_types = phenotype_types[0]

        # Clone the phenotype values to ensure we can modify them
        new_phenotype_values = data["gene"].phenotype_values.clone()

        # Handle scalar tensors by ensuring at least 1D
        if new_phenotype_values.dim() == 0:
            new_phenotype_values = new_phenotype_values.unsqueeze(0)
            is_scalar = True
        else:
            is_scalar = False

        # Process each configured label
        for label in self.label_configs:
            if label not in phenotype_types:
                continue

            # Find indices where this phenotype appears
            label_idx = phenotype_types.index(label)
            mask = data["gene"].phenotype_type_indices == label_idx

            if mask.sum() == 0:
                continue

            # Get values for this phenotype
            values = new_phenotype_values[mask]

            # Denormalize and update values
            denormalized_values = self.denormalize(values, label)
            new_phenotype_values[mask] = denormalized_values

        # Update the data with denormalized values
        # Convert back to scalar if it was originally scalar
        if is_scalar:
            new_phenotype_values = new_phenotype_values.squeeze(0)
        data["gene"].phenotype_values = new_phenotype_values

        return data


### Binning strategies adapted for COO format


class BaseBinningStrategy(ABC):
    """Base class for strategies that bin continuous values into classes."""

    if TYPE_CHECKING:
        # Declared for typing only: every concrete strategy implements
        # ``compute_bins`` with its own signature. The ``*args``/``**kwargs``
        # form is a compatible supertype so subclass overrides do not conflict.
        def compute_bins(
            self, *args: Any, **kwargs: Any
        ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
            """Compute bin edges and metadata for the binning strategy.

            Returns:
                A tuple of the computed bin edges and a metadata dict.
                Implemented by each concrete strategy subclass.
            """
            ...

    def clamp_values(
        self, values: torch.Tensor, bin_edges: torch.Tensor
    ) -> torch.Tensor:
        """Clamp values to be within bin edge range."""
        # Move bin_edges to the same device as values
        bin_edges = bin_edges.to(values.device)
        return torch.clamp(values, min=bin_edges[0], max=bin_edges[-1])

    def compute_ordinal_labels(
        self, values: torch.Tensor, bin_edges: torch.Tensor
    ) -> torch.Tensor:
        """Compute ordinal labels with clamping for out-of-bounds values.

        Threshold ``k`` is 1 when the clamped value is ``>=`` interior edge ``k + 1``,
        the same left-closed convention as the one-hot path (bin ``i`` is
        ``[edge_i, edge_{i+1})``, as ``torch.bucketize(..., right=True) - 1`` and
        ``np.digitize``): the number of ones equals the one-hot bin index, so a value
        exactly on an interior edge belongs to the bin that edge opens under both.
        """
        # Move bin_edges to the same device as values
        bin_edges = bin_edges.to(values.device)
        ordinal_labels = torch.zeros(
            (len(values), len(bin_edges) - 2), device=values.device
        )
        clamped_values = self.clamp_values(values, bin_edges)

        for i, val in enumerate(values):
            if torch.isnan(val):
                ordinal_labels[i] = torch.nan
            else:
                ordinal_labels[i] = (clamped_values[i] >= bin_edges[1:-1]).float()
        return ordinal_labels

    def compute_soft_labels(
        self,
        values: torch.Tensor,
        bin_edges: torch.Tensor,
        strategy: str = "equal_width",
        sigma_scale: float = 3,
    ) -> torch.Tensor:
        """Compute soft labels with clamping for out-of-bounds values.

        Each row is a Gaussian on the bin centers normalized to sum to one, computed as
        a softmax of the log-weights ``-0.5 * (distance / sigma) ** 2``. The softmax
        subtracts the row maximum before exponentiating, so a narrow sigma whose raw
        weights would all underflow to 0 still yields a distribution.
        """
        # Move bin_edges to the same device as values
        bin_edges = bin_edges.to(values.device)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        min_bin_width = torch.min(bin_edges[1:] - bin_edges[:-1])
        sigma = min_bin_width * sigma_scale

        soft_labels = torch.zeros((len(values), len(bin_centers)), device=values.device)
        clamped_values = self.clamp_values(values, bin_edges)

        for i, val in enumerate(values):
            if torch.isnan(val):
                soft_labels[i] = torch.nan
            else:
                # Use clamped value for gaussian computation
                distances = torch.abs(clamped_values[i] - bin_centers)
                soft_labels[i] = torch.softmax(-0.5 * (distances / sigma) ** 2, dim=0)

        return soft_labels

    def compute_onehot_labels(
        self, values: torch.Tensor, bin_edges: torch.Tensor
    ) -> torch.Tensor:
        """Compute one-hot labels with clamping for out-of-bounds values."""
        # Move bin_edges to the same device as values
        bin_edges = bin_edges.to(values.device)
        num_bins = len(bin_edges) - 1
        onehot = torch.zeros((len(values), num_bins), device=values.device)
        clamped_values = self.clamp_values(values, bin_edges)

        for i, val in enumerate(values):
            if torch.isnan(val):
                onehot[i] = torch.nan
            else:
                # Use clamped value for bin assignment
                bin_idx = (
                    torch.searchsorted(bin_edges, clamped_values[i], right=True) - 1
                )
                # Clamp to handle any numerical precision edge cases
                bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)
                onehot[i, bin_idx] = 1.0

        return onehot


class EqualWidthStrategy(BaseBinningStrategy):
    """Binning strategy with bins of equal width across the value range."""

    def compute_bins(
        self, values: npt.NDArray[Any], num_bins: int
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        """Compute equal-width bins."""
        non_nan = values[~np.isnan(values)]
        bin_edges = np.linspace(non_nan.min(), non_nan.max(), num_bins + 1)
        metadata = {
            "min": non_nan.min(),
            "max": non_nan.max(),
            "mean": non_nan.mean(),
            "std": non_nan.std(),
            "bin_edges": bin_edges,
            "bin_widths": np.diff(bin_edges),
            "strategy": "equal_width",
        }
        return bin_edges, metadata


class EqualFrequencyStrategy(BaseBinningStrategy):
    """Binning strategy with bins holding roughly equal sample counts."""

    def compute_bins(
        self, values: npt.NDArray[Any], num_bins: int
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        """Compute equal-frequency (quantile) bins."""
        non_nan = values[~np.isnan(values)]
        bin_edges = np.percentile(non_nan, np.linspace(0, 100, num_bins + 1))
        metadata = {
            "min": non_nan.min(),
            "max": non_nan.max(),
            "mean": non_nan.mean(),
            "std": non_nan.std(),
            "bin_edges": bin_edges,
            "bin_counts": np.histogram(non_nan, bin_edges)[0],
            "strategy": "equal_frequency",
        }
        return bin_edges, metadata


class AutoBinStrategy(BaseBinningStrategy):
    """Binning strategy that picks the bin count from the data's spread."""

    def compute_bins(
        self, values: npt.NDArray[Any], num_bins: int | None = None
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        """Compute bins based on data std."""
        non_nan = values[~np.isnan(values)]
        std = np.std(non_nan)
        range_width = np.max(non_nan) - np.min(non_nan)
        num_bins = int(range_width / std) if num_bins is None else num_bins
        bin_edges, metadata = EqualWidthStrategy().compute_bins(values, num_bins)
        metadata["strategy"] = "auto"
        return bin_edges, metadata


class COOLabelBinningTransform(BaseTransform):  # type: ignore[misc]  # BaseTransform is Any (torch_geometric untyped)
    """Transform binning COO-format continuous labels into class indices."""

    def __init__(
        self,
        dataset: "Neo4jCellDataset",
        label_configs: dict[str, dict[str, Any]],
        normalizer: COOLabelNormalizationTransform | None = None,
    ):
        """Initialize binning strategies and per-label bin parameters.

        Args:
            dataset: Neo4jCellDataset instance
            label_configs: Dict of configurations
            normalizer: Optional normalization transform applied before binning
        """
        super().__init__()
        self.label_configs = label_configs
        self.normalizer = normalizer
        self.strategies = {
            "equal_width": EqualWidthStrategy(),
            "equal_frequency": EqualFrequencyStrategy(),
            "auto": AutoBinStrategy(),
        }

        for label, config in label_configs.items():
            label_type = config.get("label_type", "categorical").lower()
            if label_type not in LABEL_TYPES:
                raise ValueError(
                    f"Unknown label_type {label_type!r} for label {label!r}; "
                    f"valid label types: {', '.join(LABEL_TYPES)}"
                )

        # Initialize binning parameters for each label
        self.label_metadata = {}
        df = dataset.label_df.replace([np.inf, -np.inf], np.nan)

        # Create normalized df for binning
        if self.normalizer is not None:
            normalized_df = df.copy()
            for label in label_configs:
                if label in df.columns:
                    # Create a mock data object to use normalizer
                    mock_data = HeteroData()
                    mock_data["gene"].phenotype_values = torch.tensor(
                        df[label].values, dtype=torch.float
                    )
                    mock_data["gene"].phenotype_type_indices = torch.zeros(
                        len(df[label]), dtype=torch.long
                    )
                    mock_data["gene"].phenotype_types = [label]
                    normalized_data = self.normalizer(mock_data)
                    normalized_df[label] = normalized_data["gene"][
                        "phenotype_values"
                    ].numpy()
            df = normalized_df

        # Now compute bin edges on the normalized data
        for label, config in label_configs.items():
            if label not in df.columns:
                raise ValueError(f"Label {label} not found in dataset")

            strategy = self.strategies[config["strategy"]]
            bin_edges, metadata = strategy.compute_bins(
                df[label].values, config.get("num_bins", None)
            )
            self.label_metadata[label] = metadata

        # If normalizer is provided, also store denormalized bin edges
        if self.normalizer is not None:
            for label, metadata in self.label_metadata.items():
                bin_edges_normalized = torch.tensor(
                    metadata["bin_edges"], dtype=torch.float
                )
                # Create a mock data object
                mock_data = HeteroData()
                mock_data["gene"].phenotype_values = bin_edges_normalized
                mock_data["gene"].phenotype_type_indices = torch.zeros(
                    len(bin_edges_normalized), dtype=torch.long
                )
                mock_data["gene"].phenotype_types = [label]
                # Apply inverse transform
                temp_data = self.normalizer.inverse(mock_data)
                self.label_metadata[label]["bin_edges_denormalized"] = temp_data[
                    "gene"
                ]["phenotype_values"].numpy()

    def label_width(self, label: str) -> int:
        """Number of COO entries one value of ``label`` expands into.

        Categorical and soft labels emit one entry per bin (``num_bins``); ordinal
        labels emit one per interior edge (``num_bins - 1`` thresholds).
        """
        num_bins = len(self.label_metadata[label]["bin_edges"]) - 1
        label_type = self.label_configs[label].get("label_type", "categorical").lower()
        return num_bins - 1 if label_type == "ordinal" else num_bins

    def forward(self, data: HeteroData | Batch) -> HeteroData | Batch:
        """Transform the data by binning specified labels in COO format.

        The type list is rewritten in its own order: a configured label becomes
        ``<label>_bin_0 .. <label>_bin_{w-1}`` (``w`` from ``label_width``) and any other
        label passes through under its own name. Each COO entry is replaced in place by
        its ``w`` binned entries (configured label) or kept as is (other label), with
        type indices taken from that rewritten list.
        """
        # Check if we have phenotype data in COO format
        if not hasattr(data["gene"], "phenotype_values"):
            return data

        # Get phenotype types - handle batch case where it might be a list of lists
        phenotype_types = data["gene"].phenotype_types
        if (
            isinstance(phenotype_types, list)
            and phenotype_types
            and isinstance(phenotype_types[0], list)
        ):
            # In batch mode, all items should have the same phenotype types
            phenotype_types = phenotype_types[0]

        if not any(label in self.label_configs for label in phenotype_types):
            return data

        # Handle scalar tensors by ensuring at least 1D
        phenotype_values = data["gene"].phenotype_values
        if phenotype_values.dim() == 0:
            phenotype_values = phenotype_values.unsqueeze(0)
        type_indices = data["gene"].phenotype_type_indices
        sample_indices = data["gene"].phenotype_sample_indices

        # Rewritten type list and the explicit old type index -> new type indices map
        new_phenotype_types: list[str] = []
        old_to_new: list[list[int]] = []
        for ptype in phenotype_types:
            start = len(new_phenotype_types)
            if ptype in self.label_configs:
                width = self.label_width(ptype)
                new_phenotype_types.extend(f"{ptype}_bin_{j}" for j in range(width))
            else:
                new_phenotype_types.append(ptype)
            old_to_new.append(list(range(start, len(new_phenotype_types))))

        # Binned rows, keyed by the position of the COO entry they replace
        binned_rows: dict[int, torch.Tensor] = {}
        for label, config in self.label_configs.items():
            if label not in phenotype_types:
                continue

            label_idx = phenotype_types.index(label)
            mask = type_indices == label_idx
            if mask.sum() == 0:
                continue

            values = phenotype_values[mask]
            bin_edges = torch.tensor(self.label_metadata[label]["bin_edges"])
            strategy = self.strategies[config["strategy"]]

            # Store continuous values if requested
            if config.get("store_continuous", True):
                if not hasattr(data["gene"], f"{label}_continuous"):
                    data["gene"][f"{label}_continuous"] = values

            label_type = config.get("label_type", "categorical").lower()
            if label_type == "categorical":
                binned_values = strategy.compute_onehot_labels(values, bin_edges)
            elif label_type == "soft":
                sigma = config.get("sigma", 3)
                # NOTE: passes the strategy object where the signature wants a
                # str; the param is unused in the body, so this is a no-op at
                # runtime. Preserving behavior, so silence the type only.
                binned_values = strategy.compute_soft_labels(
                    values,
                    bin_edges,
                    strategy,  # type: ignore[arg-type]  # unused str param; object passed, runtime no-op
                    sigma,
                )
            else:  # "ordinal"; label_type was validated at construction
                binned_values = strategy.compute_ordinal_labels(values, bin_edges)

            positions = torch.nonzero(mask).flatten().tolist()
            for row, position in zip(binned_values, positions, strict=True):
                binned_rows[position] = row

        all_values: list[torch.Tensor] = []
        all_type_indices: list[int] = []
        all_sample_indices: list[int] = []
        for position in range(len(phenotype_values)):
            new_types = old_to_new[int(type_indices[position])]
            sample = int(sample_indices[position])
            if position in binned_rows:
                row = binned_rows[position]
                all_values.extend(row.unbind())
            else:
                all_values.append(phenotype_values[position])
            all_type_indices.extend(new_types)
            all_sample_indices.extend([sample] * len(new_types))

        data["gene"].phenotype_values = torch.stack(all_values)
        data["gene"].phenotype_type_indices = torch.tensor(
            all_type_indices, dtype=torch.long
        )
        data["gene"].phenotype_sample_indices = torch.tensor(
            all_sample_indices, dtype=torch.long
        )
        data["gene"].phenotype_types = new_phenotype_types

        return data

    def inverse(self, data: HeteroData | Batch, seed: int = 42) -> HeteroData | Batch:
        """Inverse transform to recover continuous values from binned COO format.

        Mirrors ``forward``: the ``<label>_bin_<j>`` types of each configured label
        collapse back to ``<label>`` at the position of its first bin, and every other
        type passes through with its entries unchanged and its type index remapped.
        Output entries follow the rewritten type list; a binned label's samples are
        decoded in sorted sample order, one seeded draw each where the decode samples.
        """
        torch.manual_seed(seed)
        # Check if we have phenotype data in COO format
        if not hasattr(data["gene"], "phenotype_values"):
            return data

        # For inverse transform, we need to reconstruct continuous values from bins
        # This is complex for COO format since we need to aggregate bin information

        # First, identify which phenotypes are binned
        phenotype_types = data["gene"].phenotype_types
        if (
            isinstance(phenotype_types, list)
            and phenotype_types
            and isinstance(phenotype_types[0], list)
        ):
            # In batch mode, all items should have the same phenotype types
            phenotype_types = phenotype_types[0]

        # Handle scalar tensors by ensuring at least 1D
        phenotype_values = data["gene"].phenotype_values
        if phenotype_values.dim() == 0:
            phenotype_values = phenotype_values.unsqueeze(0)

        # Group bin types by original phenotype; every other type passes through.
        # ``order`` lists the rewritten types: a configured label, or a pass-through
        # type with its old type index.
        phenotype_groups: dict[str, list[int]] = {}
        order: list[tuple[str, int | None]] = []
        for i, ptype in enumerate(phenotype_types):
            owner = None
            for label in self.label_configs:
                if ptype.startswith(f"{label}_bin_"):
                    owner = label
                    break
            if owner is None:
                order.append((ptype, i))
            else:
                if owner not in phenotype_groups:
                    phenotype_groups[owner] = []
                    order.append((owner, None))
                phenotype_groups[owner].append(i)

        if not phenotype_groups:
            return data

        # Reconstruct continuous values
        new_values: list[float] = []
        new_type_indices: list[int] = []
        new_sample_indices: list[int] = []
        new_phenotype_types: list[str] = []

        for label, old_idx in order:
            if old_idx is not None:
                mask = data["gene"].phenotype_type_indices == old_idx
                new_values.extend(phenotype_values[mask].tolist())
                new_type_indices.extend([len(new_phenotype_types)] * int(mask.sum()))
                new_sample_indices.extend(
                    data["gene"].phenotype_sample_indices[mask].tolist()
                )
                new_phenotype_types.append(label)
                continue

            config = self.label_configs[label]
            label_type = config.get("label_type", "categorical").lower()
            bin_edges = torch.tensor(
                self.label_metadata[label]["bin_edges"],
                device=data["gene"].phenotype_values.device,
                dtype=torch.float32,
            )

            # Get unique sample indices for this phenotype
            bin_indices = phenotype_groups[label]
            sample_indices_set = set()
            for bin_idx in bin_indices:
                mask = data["gene"].phenotype_type_indices == bin_idx
                sample_indices_set.update(
                    data["gene"].phenotype_sample_indices[mask].tolist()
                )

            # Process each sample
            for sample_idx in sorted(sample_indices_set):
                # Collect bin values for this sample
                bin_values = []
                for bin_idx in bin_indices:
                    mask = (data["gene"].phenotype_type_indices == bin_idx) & (
                        data["gene"].phenotype_sample_indices == sample_idx
                    )
                    if mask.sum() > 0:
                        bin_values.append(phenotype_values[mask][0])
                    else:
                        bin_values.append(torch.tensor(0.0))

                if not bin_values:
                    continue

                # Use a distinct name for the stacked tensor so the variable type
                # stays a list above and a Tensor here (no runtime change).
                bin_values_tensor = torch.stack(bin_values)

                # Check for NaN
                continuous_value: float
                if torch.isnan(bin_values_tensor).any():
                    continuous_value = float("nan")
                else:
                    # Reconstruct continuous value based on label type
                    if label_type == "ordinal":
                        # Count number of 1s to determine bin
                        crossings = torch.sum(bin_values_tensor > 0.5)
                        # cast for typing only: a summed bool count is an int.
                        bin_pos = cast(int, crossings.item())
                        low, high = bin_edges[bin_pos], bin_edges[bin_pos + 1]
                        rand_val = (
                            torch.rand(1, device=bin_edges.device) * (high - low) + low
                        )
                        continuous_value = rand_val.item()

                    elif label_type == "soft":
                        # Soft labels are already probabilities: average the bin
                        # centers weighted by them (no softmax) over a window of
                        # 2 bins either side of the argmax.
                        window_size = 2
                        probs = bin_values_tensor
                        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
                        # cast for typing only: a 0-d index tensor behaves like an
                        # int for the slicing/arithmetic below (no runtime change).
                        max_prob_bin = cast(int, torch.argmax(probs))

                        start_idx = max(0, max_prob_bin - window_size)
                        end_idx = min(len(bin_centers), max_prob_bin + window_size + 1)

                        if (end_idx - start_idx) < (2 * window_size + 1):
                            continuous_value = bin_centers[max_prob_bin].item()
                        else:
                            window_probs = probs[start_idx:end_idx]
                            window_centers = bin_centers[start_idx:end_idx]
                            window_probs = window_probs / window_probs.sum()
                            continuous_value = (
                                (window_probs * window_centers).sum().item()
                            )

                    elif label_type == "categorical":
                        # Use argmax to find bin
                        probs = torch.softmax(bin_values_tensor, dim=-1)
                        # cast for typing only: a 0-d argmax tensor indexes like an int.
                        argmax_bin = cast(int, torch.argmax(probs))
                        low, high = bin_edges[argmax_bin], bin_edges[argmax_bin + 1]
                        rand_val = (
                            torch.rand(1, device=bin_edges.device) * (high - low) + low
                        )
                        continuous_value = rand_val.item()

                new_values.append(continuous_value)
                new_type_indices.append(len(new_phenotype_types))
                new_sample_indices.append(sample_idx)

            new_phenotype_types.append(label)

        # Update data with reconstructed continuous values
        data["gene"].phenotype_values = torch.tensor(new_values, dtype=torch.float)
        data["gene"].phenotype_type_indices = torch.tensor(
            new_type_indices, dtype=torch.long
        )
        data["gene"].phenotype_sample_indices = torch.tensor(
            new_sample_indices, dtype=torch.long
        )
        data["gene"].phenotype_types = new_phenotype_types

        return data

    def get_bin_info(self, label: str) -> dict[str, Any]:
        """Get binning information for a label."""
        if label not in self.label_metadata:
            raise ValueError(f"No binning metadata found for label {label}")
        return self.label_metadata[label]


class COOInverseCompose(BaseTransform):  # type: ignore[misc]  # BaseTransform is Any (torch_geometric untyped)
    """A transform that applies the inverse of a sequence of transforms in reverse order."""

    def __init__(self, transforms: Compose | list[BaseTransform]):
        """Store the transforms and verify each implements an inverse method."""
        super().__init__()
        if isinstance(transforms, Compose):
            self.transforms = transforms.transforms
        elif isinstance(transforms, list):
            self.transforms = transforms
        else:
            raise ValueError(
                "transforms must be either a Compose object or a list of transforms"
            )

        # Verify all transforms have inverse method
        for t in self.transforms:
            if not hasattr(t, "inverse"):
                raise ValueError(
                    f"Transform {t.__class__.__name__} does not implement inverse method"
                )

    def forward(self, data: HeteroData | Batch) -> HeteroData | Batch:
        """Apply inverse transforms in reverse order."""
        # Apply transforms in reverse order
        for t in reversed(self.transforms):
            data = t.inverse(data)
        return data

    def __repr__(self) -> str:
        """Return a string listing the composed transforms."""
        args = [f"\n  {t}" for t in self.transforms]
        return f"{self.__class__.__name__}({''.join(args)}\n)"
