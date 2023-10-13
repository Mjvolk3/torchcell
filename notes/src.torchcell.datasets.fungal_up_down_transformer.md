---
id: ga2dbwqtac7m5imc0lw8kll
title: Fungal_up_down_transformer
desc: ''
updated: 1696010759598
created: 1695537019509
---

## Model Variants Support

Since the complexity increased with encoding, and the `species` model looks better than the `agnostic` model, I am only supporting the `species` model for now. Also only using `allow_undersize` as it is the most general case.

## Parse Genome and Return None For Dunder Add

When we add two `BaseEmbeddingDataset` it goes though the `__init__` again yet genome was deleted so there is no state. It is not needed for combining datasets. In fact this forced me to comment it out for the time being [[Disabling Genome Due to Sqlite and Parsed Genome|dendron://torchcell/src.torchcell.datasets.embedding#disabling-genome-due-to-sqlite-and-parsed-genome]].

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

### Parse Genome and Return None For Dunder Add - Need Genome in Process

We need the `Genome` object to be able to run

```python
for gene_id in tqdm(self.genome.gene_set):
    sequence = self.genome[gene_id]
```

yet if we have this object, we cannot run adds

This solution seems to work. The logic is as follows, set genome, and if process isn't run yet, run it, then convert the genome to `ParsedGenome`, and delete the genome to remove any potential reference to `sqlite` database. If we look at [[Parse Genome and Return None For Dunder Add|dendron://torchcell/src.torchcell.datasets.fungal_up_down_transformer#parse-genome-and-return-none-for-dunder-add]] there is some ambiguity as to what the true issue is. Unfortunately, it is difficult to test all of these at once.

```python
def __init__(
    self,
    root: str,
    genome: SCerevisiaeGenome,
    transformer_model_name: str | None = None,
    transform: Callable | None = None,
    pre_transform: Callable | None = None,
):
    self.genome = genome
    super().__init__(root, transformer_model_name, transform, pre_transform)
    # convert genome to parsed genome after process, so have potential issue
    # with sqlite database
    # TODO try without parsed_genome on ddp to see if issue was
    # BaseEmbeddingDataset previously taking genome as a parameter
    self.genome = self.parse_genome(genome)
    del genome
```

I have successfully tested with this implementation that we can still add datasets.

```python
nt_dataset = NucleotideTransformerDataset(
    root="data/scerevisiae/nucleotide_transformer_embed",
    genome=genome,
    transformer_model_name="nt_window_5979",
)

fud3_dataset = FungalUpDownTransformerDataset(
    root="data/scerevisiae/fungal_up_down_embed",
    genome=genome,
    transformer_model_name="species_downstream",
)

seq_embeddings = nt_dataset + fud3_dataset
```
