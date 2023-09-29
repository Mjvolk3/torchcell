---
id: ga2dbwqtac7m5imc0lw8kll
title: Fungal_up_down_transformer
desc: ''
updated: 1695956613474
created: 1695537019509
---

## Model Variants Support

Since the complexity increased with encoding, and the `species` model looks better than the `agnostic` model, I am only supporting the `species` model for now. Also only using `allow_undersize` as it is the most general case.

## Parse Genome and Return None For Dunder Add

When we add two `BaseEmbeddingDataset` it goes though the `__init__` again yet genome was deleted so there is no state. It is not needed for combining datasets. In fact this forced me to comment it out for the time being [[Disabling Genome Due to Sqlite and Parsed Genome|dendron://torchcell/src.torchcell.datasets.nucleotide_embedding#disabling-genome-due-to-sqlite-and-parsed-genome]].

```python
class ParsedGenome(ModelStrictArbitrary):
    gene_set: GeneSet

    @validator("gene_set")
    def validate_gene_set(cls, value):
        if not isinstance(value, GeneSet):
            raise ValueError(f"gene_set must be a GeneSet, got {type(value).__name__}")
        return value


class FungalUpDownTransformerDataset(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {
        "species_downstream": ("window_three_prime", 300, True, True),
        "species_upstream": ("window_five_prime", 1003, True, True),
    }

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        transformer_model_name: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        super().__init__(root, transformer_model_name, transform, pre_transform)
        self.genome = self.parse_genome(genome)
        del genome

        self.transformer_model_name = transformer_model_name

        if self.transformer_model_name:
            if not os.path.exists(self.processed_paths[0]):
                self.transformer = self.initialize_transformer()
                self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @staticmethod
    def parse_genome(genome) -> ParsedGenome:
        # BUG we have to do this black magic because when you merge datasets with +
        # the genome is None
        if genome is None:
            return None
        else:
            data = {}
            data["gene_set"] = genome.gene_set
            return ParsedGenome(**data)
...
```
