# sequence_plot.py
# #BUG This code does not work yet, but has scratch for some plotting ideas.
from abc import ABC, abstractmethod
from collections import Counter

import matplotlib.pyplot as plt
from attrs import define, field
from sklearn import base

from torchcell.sgd.sequence import AbcGenome, SCerevisiaeGenome


class GenomePlot(ABC):
    genome: AbcGenome
    pass

    @abstractmethod
    def plot() -> None:
        raise NotImplementedError("Subclasses must implement plot() method.")

    def save(self, file_path_noext: str) -> None:
        # add some automatic saving of pdf and png based on notes/assets/images...
        pass


@define
class PlotFeatureTypeCounts(GenomePlot):
    genome: AbcGenome

    @property
    def feature_type_counts(self):
        feature_types = [
            feat.featuretype for feat in list(self.genome.db.all_features())
        ]
        return dict(Counter(feature_types))

    def plot(self) -> None:
        feature_type_counts = self.feature_type_counts
        types = list(feature_type_counts.keys())
        counts = list(feature_type_counts.values())

        plt.figure(figsize=(10, 5))
        plt.barh(types, counts)
        plt.xlabel("Count")
        plt.ylabel("Feature type")
        plt.title("Feature Type Counts")
        plt.show()


# @property
# def mrna_attribute_table(self) -> pd.DataFrame:
#     data = []
#     for gene_feature in self.db.features_of_type("mRNA"):
#         gene_data = {}
#         for attr_name in gene_feature.attributes.keys():
#             # We only add attributes with length 1 or less
#             if len(gene_feature.attributes[attr_name]) <= 1:
#                 # If the attribute is a list with one value, we unpack it
#                 gene_data[attr_name] = (
#                     gene_feature.attributes[attr_name][0]
#                     if len(gene_feature.attributes[attr_name]) == 1
#                     else None
#                 )
#         data.append(gene_data)
#     return pd.DataFrame(data)


def main():
    genome = SCerevisiaeGenome()
    genome_plot = PlotFeatureTypeCounts(genome)
    genome_plot.plot()
    print()


if __name__ == "__main__":
    main()
