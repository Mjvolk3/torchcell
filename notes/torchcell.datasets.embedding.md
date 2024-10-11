---
id: co0dnmoyafs9tlmofsn3e5y
title: Embedding
desc: ''
updated: 1712255705988
created: 1692275134695
---

## Disabling Genome Due to Sqlite and Parsed Genome

We commented out genome. I am not sure if this is necessary because returning `None` in `parse_genome` method in [[Parse Genome and Return None For Dunder Add|dendron://torchcell/torchcell.datasets.fungal_up_down_transformer#parse-genome-and-return-none-for-dunder-add]] solved the issue. It would be worth seeing if we could pass `genome` as it could be useful for data joins. Obviously the only reason we could comment it out is because it is not yet being used for joins.

## Difficult to Add Datasets with Sum operator

`a + b` and `sum([a, b])` seem to work slightly different in python. `sum([a, b])` might use `__radd__` and has and tries to add 0 first. I think to use `sum` with will need a dummy data.

Also another issue is that when adding datasets if one of the datasets is not of the same size. It's data does not get combined.

## Require a DNA Window to be Included

Works when we are just looking at a set string of sequence, but if we take CDS we are cutting out introns, so the we need multiple positions to define where the sequence comes from.

```python
def __add__(self, other):
    if isinstance(other, int) and other == 0:
        return self
    # Ensure the other object is of the same type
    if not isinstance(other, BaseEmbeddingDataset):
        raise ValueError("Can only add datasets of the same type.")

    combined_data_list = []

    # Create a dictionary from the current dataset for efficient lookup
    current_data_dict = {data_item.id: data_item for data_item in self}

    # Lists to store duplicate keys
    duplicate_dna_windows_keys = []
    duplicate_embeddings_keys = []

    # Combine the data from the other dataset
    for data_item in other:
        if data_item.id in current_data_dict:
            # Check for duplicate keys in dna_windows
            for key in data_item.dna_windows:
                if key in current_data_dict[data_item.id].dna_windows:
                    duplicate_dna_windows_keys.append(key)
                else:
                    # Merge the dna_windows dictionaries
                    current_data_dict[data_item.id].dna_windows[
                        key
                    ] = data_item.dna_windows[key]
                print()
            # Check for duplicate keys in embeddings
            for key in data_item.embeddings:
                if key in current_data_dict[data_item.id].embeddings:
                    duplicate_embeddings_keys.append(key)
                else:
                    # Merge the embeddings dictionaries
                    current_data_dict[data_item.id].embeddings[
                        key
                    ] = data_item.embeddings[key]
                print()
        else:
            combined_data_list.append(data_item)

    # If there are duplicates, raise an error
    if duplicate_dna_windows_keys:
        raise ValueError(
            "Duplicate keys found in dna_windows:"
            f"{', '.join(duplicate_dna_windows_keys)}"
        )
    if duplicate_embeddings_keys:
        raise ValueError(
            "Duplicate keys found in embeddings:"
            f"{', '.join(duplicate_embeddings_keys)}"
        )

    # Add the modified data items from the current dataset to the combined list
    combined_data_list.extend(current_data_dict.values())

    # Use collate to convert the combined list into the format InMemoryDataset expects
    data, slices = self.collate(combined_data_list)

    # Create a new dataset instance with the combined data
    combined_dataset = self.__class__(root=self.root, genome=None, model_name=None)
    combined_dataset.data, combined_dataset.slices = data, slices

    return combined_dataset
```

## 2024.04.04 - MODEL_TO_WINDOW changed to some configuration

Now that we are using DNA, protein, and other arbitrary node features it will be best to rename this.
