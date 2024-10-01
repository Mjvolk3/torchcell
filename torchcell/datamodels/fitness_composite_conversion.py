# torchcell/datamodels/fitness_composite_conversion
# [[torchcell.datamodels.fitness_composite_conversion]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/fitness_composite_conversion
# Test file: tests/torchcell/datamodels/test_fitness_composite_conversion.py


from torchcell.datamodels.gene_essentiality_to_fitness_conversion import (
    GeneEssentialityToFitnessConverter,
)
from torchcell.datamodels.synthetic_lethality_to_fitness_conversion import (
    SyntheticLethalityToFitnessConverter,
)
from torchcell.datamodels.conversion import Converter, ConversionMap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchcell.data.neo4j_query_raw import Neo4jQueryRaw


class CompositeFitnessConverter(Converter):
    def __init__(self, root: str, query: "Neo4jQueryRaw"):
        super().__init__(root, query)
        self.gene_essentiality_converter = GeneEssentialityToFitnessConverter(
            root, query
        )
        self.synthetic_lethality_converter = SyntheticLethalityToFitnessConverter(
            root, query
        )

    @property
    def conversion_map(self) -> ConversionMap:
        entries = (
            self.gene_essentiality_converter.conversion_map.entries
            + self.synthetic_lethality_converter.conversion_map.entries
        )
        return ConversionMap(entries=entries)

    def convert(self, data: dict):
        # Try conversion with gene essentiality converter
        converted_data = self.gene_essentiality_converter.convert(data)

        # If conversion didn't happen (i.e., data remained unchanged), try synthetic lethality converter
        if converted_data == data:
            converted_data = self.synthetic_lethality_converter.convert(data)

        return converted_data


if __name__ == "__main__":
    pass
