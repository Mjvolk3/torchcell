# torchcell/datamodels/fitness_composite_conversion
# [[torchcell.datamodels.fitness_composite_conversion]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/fitness_composite_conversion
# Test file: tests/torchcell/datamodels/test_fitness_composite_conversion.py
"""Converter that maps gene essentiality and synthetic lethality data into fitness."""

from typing import TYPE_CHECKING

from torchcell.datamodels.conversion import ConversionMap, Converter
from torchcell.datamodels.gene_essentiality_to_fitness_conversion import (
    GeneEssentialityToFitnessConverter,
)
from torchcell.datamodels.schema import ExperimentReferenceType, ExperimentType
from torchcell.datamodels.synthetic_lethality_to_fitness_conversion import (
    SyntheticLethalityToFitnessConverter,
)

if TYPE_CHECKING:
    from torchcell.data.neo4j_query_raw import Neo4jQueryRaw


class CompositeFitnessConverter(Converter):
    """Converter combining gene essentiality and synthetic lethality to fitness."""

    def __init__(self, root: str, query: "Neo4jQueryRaw"):
        """Set up the gene essentiality and synthetic lethality sub-converters."""
        super().__init__(root, query)
        self.gene_essentiality_converter = GeneEssentialityToFitnessConverter(
            root, query
        )
        self.synthetic_lethality_converter = SyntheticLethalityToFitnessConverter(
            root, query
        )

    @property
    def conversion_map(self) -> ConversionMap:
        """Return the merged conversion map of both sub-converters."""
        entries = (
            self.gene_essentiality_converter.conversion_map.entries
            + self.synthetic_lethality_converter.conversion_map.entries
        )
        return ConversionMap(entries=entries)

    def convert(
        self, data: dict[str, ExperimentType | ExperimentReferenceType]
    ) -> dict[str, ExperimentType | ExperimentReferenceType | None]:
        """Convert data via gene essentiality, falling back to synthetic lethality."""
        # Try conversion with gene essentiality converter
        converted_data = self.gene_essentiality_converter.convert(data)

        # If conversion didn't happen (i.e., data remained unchanged), try synthetic lethality converter
        if converted_data == data:
            converted_data = self.synthetic_lethality_converter.convert(data)

        return converted_data


if __name__ == "__main__":
    pass
