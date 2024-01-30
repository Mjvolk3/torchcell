from .dataset import Dataset
from torchcell.data import (
    ExperimentReferenceIndex,
    serialize_for_hashing,
    compute_sha256_hash,
)
from tqdm import tqdm


def compute_experiment_reference_index(
    dataset: Dataset,
) -> list[ExperimentReferenceIndex]:
    # Hashes for each reference
    print("Computing experiment_reference_index hashes...")
    reference_hashes = [
        compute_sha256_hash(serialize_for_hashing(data["reference"]))
        for data in tqdm(dataset)
    ]

    # Identify unique hashes
    unique_hashes = set(reference_hashes)

    # Initialize ExperimentReferenceIndex list
    reference_indices = []

    print("Finding unique references...")
    for unique_hash in tqdm(unique_hashes):
        # Create a boolean list where True indicates the presence of the unique reference
        index_list = [ref_hash == unique_hash for ref_hash in reference_hashes]

        # Find the corresponding reference object for the unique hash
        ref_index = reference_hashes.index(unique_hash)
        unique_ref = dataset[ref_index]["reference"]

        # Create ExperimentReferenceIndex object
        exp_ref_index = ExperimentReferenceIndex(reference=unique_ref, index=index_list)
        reference_indices.append(exp_ref_index)

    return reference_indices
