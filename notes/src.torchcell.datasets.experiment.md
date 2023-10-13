---
id: vgx68e142piq5ecmw0r8r7g
title: Experiment
desc: ''
updated: 1697155716882
created: 1697150864641
---
## Dunder Adding of Experiments is a Bad Idea

```python
# TODO unify Costanzo2016
class ExperimentDataset(Dataset):
    """
    An intermediate layer for experimental datasets.
    This class is designed to handle the combination logic
    for datasets that share common attributes such as genotype, phenotype, and environment.
    """

    def __init__(self, *args, **kwargs):
        # Call the base class's __init__
        super(ExperimentDataset, self).__init__(*args, **kwargs)

      def get_genotype_key(self, data_item):
          return data_item["genotype"]

      def get_environment_key(self, data_item):
          return data_item["phenotype"]["environment"]

      def get_phenotype_key(self, data_item):
          return data_item["phenotype"]["observation"]

      def contains_key(self, key, data_list):
          return key in [self.get_genotype_key(data_item) for data_item in data_list]

      def contains_environment_key(self, genotype_key, environment_key, data_list):
          return environment_key in [
              self.get_environment_key(data_item)
              for data_item in data_list
              if self.get_genotype_key(data_item) == genotype_key
          ]

      def contains_phenotype_key(
          self, genotype_key, environment_key, phenotype_key, data_list
      ):
          return phenotype_key in [
              self.get_phenotype_key(data_item)
              for data_item in data_list
              if self.get_genotype_key(data_item) == genotype_key
              and self.get_environment_key(data_item) == environment_key
          ]

      def __add__(self, other):
          """
          Combines two experimental datasets by checking if genotype, environment,
          and phenotype keys are contained in the other dataset.
          """

          # Ensure the other object is of the same type
          if not isinstance(other, ExperimentDataset):
              raise ValueError("Can only add datasets of the same type.")

          combined_data_list = []

          # Loop over the data items in the current dataset
          for data_item_self in self:
              genotype_key_self = self.get_genotype_key(data_item_self)
              environment_key_self = self.get_environment_key(data_item_self)
              phenotype_key_self = self.get_phenotype_key(data_item_self)

              # Check if genotype key from self exists in the other dataset
              if not self.contains_key(genotype_key_self, other):
                  combined_data_list.append(data_item_self)

              else:
                  # If genotype key exists, check for environment key
                  if not self.contains_environment_key(
                      genotype_key_self, environment_key_self, other
                  ):
                      combined_data_list.append(data_item_self)

                  else:
                      # If environment key exists, check for phenotype key
                      if not self.contains_phenotype_key(
                          genotype_key_self,
                          environment_key_self,
                          phenotype_key_self,
                          other,
                      ):
                          combined_data_list.append(data_item_self)
                      else:
                          raise ValueError(
                              f"Data item with genotype key: {genotype_key_self}, "
                              f"environment key: {environment_key_self}, and "
                              f"phenotype key: {phenotype_key_self} already exists in the other dataset."
                          )

          # Append all items from the other dataset to the combined list
          combined_data_list.extend([data_item for data_item in other])

          # Create a new dataset instance with the combined data
          combined_dataset = self.__class__() # ⛔️ we need another lmdb, because the merging could potentially take long.
          combined_dataset.data_list = combined_data_list

          return combined_dataset
```

## Use Dataset Logic but Use Process for Merger Operations

Some of the unique features of this class include:

- Merge operations - These will need to be further systematized.
- `experiment_indices` - To track all indices of the different data. In the future we can imagine that if `genotype` and `environment` are compatible based on ontology specification, we could merge phenotype.
