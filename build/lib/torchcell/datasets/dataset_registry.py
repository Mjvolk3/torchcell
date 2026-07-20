# torchcell/datasets/dataset_registry
# [[torchcell.datasets.dataset_registry]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/dataset_registry
# Test file: tests/torchcell/datasets/test_dataset_registry.py
"""Global registry mapping dataset class names to classes for lookup by name."""

dataset_registry: dict[str, type] = {}


def register_dataset[T: type](cls: T) -> T:
    """Register a dataset class by its name and return it unchanged (decorator)."""
    dataset_registry[cls.__name__] = cls
    return cls
