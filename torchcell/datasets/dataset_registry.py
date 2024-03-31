# torchcell/datasets/dataset_registry
# [[torchcell.datasets.dataset_registry]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/dataset_registry
# Test file: tests/torchcell/datasets/test_dataset_registry.py

dataset_registry = {}

def register_dataset(cls):
    dataset_registry[cls.__name__] = cls
    return cls
