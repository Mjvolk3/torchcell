# src/torchcell/sgd/sequence/sequence.py
from abc import ABC, abstractmethod

import gffutils
import matplotlib.pyplot as plt
import pandas as pd
from attrs import define, field
from Bio import SeqIO
from Bio.Seq import Seq
from gffutils import Feature, FeatureDB
from gffutils.biopython_integration import to_seqfeature
from matplotlib import pyplot as plt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from torchcell.data_models import BaseModelStrict
from torchcell.models.constants import DNA_LLM_MAX_TOKEN_SIZE
from torchcell.sequence import AbcGenome, DnaSelectionResult, get_chr_from_description
from torchcell.sgd.sequence.constants import CHROMOSOMES


@define
class SCerevisiaeGenome(AbcGenome):
    _fasta_path: str = "data/sgd/genome/S288C_reference_genome_R64-3-1_20210421/S288C_reference_sequence_R64-3-1_20210421.fsa"
    _gff_path: str = "data/sgd/genome/S288C_reference_genome_R64-3-1_20210421/saccharomyces_cerevisiae_R64-3-1_20210421.gff"
    db = field(init=False, repr=False)
    fasta_sequences = field(init=False, default=None, repr=False)
    chr_to_nc = field(init=False, default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        # Create the database
        self.db = gffutils.create_db(
            self._gff_path,
            dbfn="data.db",
            force=True,
            keep_order=True,
            merge_strategy="merge",
            sort_attribute_values=True,
        )
        # Read the fasta file
        self.fasta_sequences = SeqIO.to_dict(SeqIO.parse(self._fasta_path, "fasta"))
        # Create mapping from chromosome number to sequence identifier
        self.chr_to_nc = {
            get_chr_from_description(self.fasta_sequences[key].description): key
            for key in self.fasta_sequences.keys()
        }
        print()

    def get_seq(
        self, chr: int | str, start: int, end: int, strand: str
    ) -> DnaSelectionResult:
        chr_num = chr
        if isinstance(chr, int):
            chr = self.chr_to_nc[chr]
        if strand == "+":
            seq = self.fasta_sequences[chr].seq[start:end]
        elif strand == "-":
            seq = self.fasta_sequences[chr].seq[start:end].reverse_complement()
        return DnaSelectionResult(
            seq=str(seq),
            chromosome=chr_num,
            start=start,
            end=end,
            strand=strand,
        )

    def select_feature_window(
        self, feature: str, window_size: int
    ) -> DnaSelectionResult:
        feature = self[feature]
        if not feature:
            raise ValueError(f"feature {feature} not found.")

        start = self[feature].start
        end = self[feature].stop
        # start_window =
        # end_window =
        strand = self[feature].strand
        chr = CHROMOSOMES.index(feature.seqid)
        dna_selection_result = self.get_seq(chr, start, end, strand)

        # Make DnaWindowResult which inherits from DnaSelectionResult and has two extra attributes (start_window, end_window, and is_maxed:bool where the window is adjusted to be the maximum size possible without going out of bounds, it will include sequence from the side of str where there won't be an index error. when this is false it will just selec the largest window possible without going out of bounds, this option will always have equal length sequence on both sides of the feature)

        # return DNAWindowResult(
        pass

    @property
    def gene_attribute_table(self) -> pd.DataFrame:
        data = []
        for gene_feature in self.db.features_of_type("gene"):
            gene_data = {}
            for attr_name in gene_feature.attributes.keys():
                # We only add attributes with length 1 or less
                if len(gene_feature.attributes[attr_name]) <= 1:
                    # If the attribute is a list with one value, we unpack it
                    gene_data[attr_name] = (
                        gene_feature.attributes[attr_name][0]
                        if len(gene_feature.attributes[attr_name]) == 1
                        else None
                    )
            data.append(gene_data)
        return pd.DataFrame(data)

    @property
    def feature_types(self) -> list[str]:
        return list(self.db.featuretypes())

    def __getitem__(self, item: str):
        # For now we only support the systematic names
        try:
            return self.db[item]
        except KeyError:
            return None


def main() -> None:
    genome = SCerevisiaeGenome()
    # print(genome.get_sequence(1, 0, 10))  # Replace with valid parameters
    print(
        genome["YFL039C"]
    )  # Replace with valid gene... we only support systematic names
    genome.get_seq(chr=0, start=10, end=120, strand="+")
    print(genome.gene_attribute_table)
    print(genome.feature_types)
    print(genome.select_feature_window("YFL039C", 100))
    print()


if __name__ == "__main__":
    main()
