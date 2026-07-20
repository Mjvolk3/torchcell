"""Genome sequence access from NCBI GFF/FASTA files for S. cerevisiae."""

from abc import ABC, abstractmethod

import gffutils
import pandas as pd
from attrs import define, field
from Bio import SeqIO


@define
class BaseGenome(ABC):
    """Abstract interface for genome sequence and annotation access."""

    @abstractmethod
    def get_sequence(self, chr: int, start: int, end: int) -> str:
        """Return the sequence on the given chromosome between start and end."""
        raise NotImplementedError

    # @abstractmethod
    # def get_chr_len(self, chr: int) -> int:
    #     # returns the length of the queried chromosome
    #     raise NotImplementedError

    @abstractmethod
    def get_gene_sequence(self, gene: str) -> str:
        """Return the nucleotide sequence for the named gene."""
        raise NotImplementedError

    # @abstractmethod
    # def get_gene_position(self, gene: str) -> int:

    #     raise NotImplementedError
    @property
    @abstractmethod
    def translation_table(self) -> pd.DataFrame:
        """Return a table mapping locus tags to gene names."""
        raise NotImplementedError


def mismatch_positions(seq1: str, seq2: str) -> list[int]:
    """Return indices where two equal-length sequences differ."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length")
    mismatches = [i for i, (n1, n2) in enumerate(zip(seq1, seq2)) if n1 != n2]
    return mismatches


def get_chr_from_description(description: str) -> int:
    """Parse a FASTA description into a chromosome number (0 for mitochondrion)."""
    desc_split = [part.rstrip(",") for part in description.split()]
    if "mitochondrion" in desc_split:
        return 0
    else:
        roman_num = desc_split[desc_split.index("chromosome") + 1]
        return roman_to_int(roman_num)


def roman_to_int(s: str) -> int:
    """Convert a Roman numeral string to its integer value."""
    roman_to_int_mapping = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }
    result = 0
    for i in range(len(s)):
        if i > 0 and roman_to_int_mapping[s[i]] > roman_to_int_mapping[s[i - 1]]:
            result += roman_to_int_mapping[s[i]] - 2 * roman_to_int_mapping[s[i - 1]]
        else:
            result += roman_to_int_mapping[s[i]]
    return result


@define
class SCerevisiaeGenome(BaseGenome):
    """S. cerevisiae genome backed by NCBI GFF annotations and a FASTA file."""

    _gff_path: str = (
        "data/ncbi/s_cerevisiae/ncbi_dataset/data/GCF_000146045.2/genomic.gff"
    )
    _fasta_path: str = "data/ncbi/s_cerevisiae/ncbi_dataset/data/GCF_000146045.2/GCF_000146045.2_R64_genomic.fna"
    db = field(init=False, repr=False)
    fasta_sequences = field(init=False, default=None, repr=False)
    chr_to_nc = field(init=False, default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        """Build the gffutils database and load FASTA sequences and chr mapping."""
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
        self.fasta_sequences = SeqIO.to_dict(  # type: ignore[no-untyped-call]  # Bio.SeqIO untyped
            SeqIO.parse(self._fasta_path, "fasta")  # type: ignore[no-untyped-call]  # Bio.SeqIO untyped
        )
        # Create mapping from chromosome number to sequence identifier
        self.chr_to_nc = {
            get_chr_from_description(self.fasta_sequences[key].description): key
            for key in self.fasta_sequences.keys()
        }

    def get_sequence(self, chr: int, start: int, end: int) -> str:
        """Return the sequence slice on the given chromosome between start and end."""
        # Get the identifier corresponding to the given chromosome number
        chr_id = self.chr_to_nc[chr]
        # Get the sequence for the chromosome with this identifier
        return str(self.fasta_sequences[chr_id].seq[start:end])

    def get_gene_sequence(self, gene_name: str) -> str:
        """Return the genomic sequence for a gene matched by Name or locus_tag."""
        # Iterate through all genes in the gffutils database
        for gene_feature in self.db.features_of_type("gene"):
            # If the gene's Name attribute matches the provided gene_name
            if (
                "Name" in gene_feature.attributes
                and gene_feature["Name"][0] == gene_name
            ) or (
                "locus_tag" in gene_feature.attributes
                and gene_feature["locus_tag"][0] == gene_name
            ):
                # Extract the sequence for this gene from the fasta sequences
                sequence = self.fasta_sequences[gene_feature.seqid].seq[
                    gene_feature.start - 1 : gene_feature.end
                ]

                return str(sequence)

        # If no matching gene is found, raise an error
        raise ValueError(f"No gene found with name {gene_name}")

    @property
    def translation_table(self) -> pd.DataFrame:
        """Return a DataFrame mapping each gene's locus_tag to its name."""
        data = []
        for gene_feature in self.db.features_of_type("gene"):
            if (
                "locus_tag" in gene_feature.attributes
                and "Name" in gene_feature.attributes
            ):
                gene_id = gene_feature["locus_tag"][0]  # Get the locus_tag
                # Some of the locus_tag names do not match SGD Systematic Names.
                # This is especially true for mitochondrial genes. e.g. gene_id == "tE(UUC)Q"
                gene_name = gene_feature["Name"][0]  # Get the gene name
                data.append((gene_id, gene_name))
        return pd.DataFrame(data, columns=["locus_tag", "name"])


def main() -> None:
    """Run a small demonstration of genome sequence lookups."""
    genome = SCerevisiaeGenome()
    print(genome.get_sequence(1, 0, 10))  # Replace with valid parameters
    print(genome.get_gene_sequence("YFL039C"))  # Replace with valid gene
    print(genome.get_gene_sequence("ACT1"))  # Replace with valid gene
    assert genome.get_gene_sequence("YFL039C") == genome.get_gene_sequence("ACT1"), (
        "genes should have same sequence"
    )
    print(genome.translation_table)
    print()


if __name__ == "__main__":
    main()
