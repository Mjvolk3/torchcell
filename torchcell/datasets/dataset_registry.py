dataset_registry = {}

def register_dataset(cls):
    dataset_registry[cls.__name__] = cls
    return cls
