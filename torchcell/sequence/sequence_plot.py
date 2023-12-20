import os
import os.path as osp
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
from attrs import define, field
from sklearn import base

from torchcell.sequence import Genome
from torchcell.sgd.sequence import SCerevisiaeGenome
import torchcell

style_file_path = osp.join(osp.dirname(torchcell.__file__), 'torchcell.mplstyle')
plt.style.use(style_file_path)

class GenomePlot(ABC):
    genome: Genome

    @abstractmethod
    def plot() -> None:
        raise NotImplementedError("Subclasses must implement plot() method.")

    def save(self, as_pdf: bool = False, as_pickle: bool = False) -> None:
        current_dir = osp.dirname(osp.abspath(__file__))
        # Get current date and time and format it as year-month-day-hour-minute-second.
        timestamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        file_path = osp.join(
            ".notes/assets/images",
            osp.relpath(current_dir),
            f"{self.__class__.__name__}-{timestamp}",  # Append timestamp to file name.
        )
        dir_name = osp.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f"{file_path}.png", format="png", bbox_inches="tight")
        print(f"Saved figure: {file_path}.png")
        if as_pdf:
            plt.savefig(f"{file_path}.pdf", format="pdf", bbox_inches="tight")
            print(f"Saved figure: {file_path}.pdf")
        if as_pickle:
            with open(f"{file_path}.pkl", "wb") as f:
                pickle.dump(plt.gcf(), f)  # Save the figure using pickle
            print(f"Saved figure data: {file_path}.pkl")


@define
class PlotFeatureTypeCounts(GenomePlot):
    genome: Genome
    plt: plt = field(init=False)

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

        # Set the figure size based on the number of feature types.
        plt.figure(figsize=(10, len(types) * 0.4))

        bars = plt.barh(types, counts)
        plt.xlabel("Count")
        plt.ylabel("Feature type")
        plt.title("Feature Type Counts")

        # Add counts as text on the bars.
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f" {width}",
                va="center",
                ha="left",  # Align the labels to the left
            )

        # Adjust xlim to provide more space for the labels
        plt.xlim(0, max(counts) * 1.1)  # Increase xlim by 10%

        self.plt = plt

    def show(self):
        plt.show()

    def close(self):
        plt.close()


def main():
    genome = SCerevisiaeGenome()
    genome_plot = PlotFeatureTypeCounts(genome)
    genome_plot.plot()
    genome_plot.save(pdf=True, mpl=True)


if __name__ == "__main__":
    main()
