import os
import os.path as osp
from typing import Set

import gffutils
import pandas as pd
from attrs import define, field
from Bio import Seq, SeqIO
from Bio.SeqRecord import SeqRecord
from gffutils.feature import Feature
from goatools.obo_parser import GODag
from sortedcontainers import SortedDict, SortedSet
from torch_geometric.data import download_url

from torchcell.sequence import (
    DnaSelectionResult,
    DnaWindowResult,
    Gene,
    Genome,
    calculate_window_bounds,
    calculate_window_bounds_symmetric,
    get_chr_from_description,
    mismatch_positions,
    roman_to_int,
)

# We put MT at 0, because it is circular, and this preserves arabic to roman
CHROMOSOMES = [
    "chrmt",
    "chrI",
    "chrII",
    "chrIII",
    "chrIV",
    "chrV",
    "chrVI",
    "chrVII",
    "chrVIII",
    "chrIX",
    "chrX",
    "chrXI",
    "chrXII",
    "chrXIII",
    "chrXIV",
    "chrXV",
    "chrXVI",
]


# IDEA we might be able to move some of the window functions into the ABC... this would be much nicer for the introductino of new genomes.
@define
class SCerevisiaeGene(Gene):
    feature: Feature = field(repr=False)
    fasta_sequences: dict[str, SeqRecord] = field(repr=False)
    chr_to_nc: dict[str, str] = field(repr=False)
    chromosome_lengths: dict[str, int] = field(repr=False)
    # below are set in __attrs_post_init__
    id: str = field(default=None)
    chromosome: int = field(default=None)
    start: int = field(default=None)
    end: int = field(default=None)
    strand: str = field(default=None)
    seq: str = field(default=None, repr=True)

    def __attrs_post_init__(self) -> None:
        self.id = self.feature.id
        # chromosome
        seqid = self.feature.seqid
        maybe_roman_numeral = seqid.split("chr")[-1]
        if maybe_roman_numeral == "mt":
            self.chromosome = 0
        else:
            self.chromosome = roman_to_int(maybe_roman_numeral)
        # others
        self.start = self.feature.start
        self.end = self.feature.end
        self.strand = self.feature.strand

        # sequence
        chr = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            self.seq = str(self.fasta_sequences[chr].seq[self.start : self.end])
        elif self.strand == "-":
            self.seq = str(
                self.fasta_sequences[chr]
                .seq[self.start : self.end]
                .reverse_complement()
            )

        # TODO consider adding these to ABC...
        # Some might be too specific to S. cerevisiae, but so they coudl be optional
        self.alias = self.feature.attributes["Alias"]
        self.name = self.feature.attributes["Name"]
        self.ontology_term = self.feature.attributes["Ontology_term"]
        self.note = self.feature.attributes["Note"]
        self.display = self.feature.attributes["display"]
        self.dbxref = self.feature.attributes["dbxref"]
        self.orf_classifcation = self.feature.attributes["orf_classification"]
        # we give a new attr for GO since most known ontology
        if self.ontology_term is not None:
            self.go = [term for term in self.ontology_term if term.startswith("GO:")]
        else:
            self.go = None

    def window(self, window_size: int, is_max_size: bool = True) -> DnaWindowResult:
        if is_max_size:
            start_window, end_window = calculate_window_bounds(
                start=self.start,
                end=self.end,
                strand=self.strand,
                window_size=window_size,
                chromosome_length=self.chromosome_lengths[self.chromosome],
            )

        else:
            start_window, end_window = calculate_window_bounds_symmetric(
                start=self.start,
                end=self.end,
                window_size=window_size,
                chromosome_length=self.chromosome_lengths[self.chromosome],
            )
        chr_id = self.chr_to_nc[self.chromosome]
        seq = str(self.fasta_sequences[chr_id].seq[start_window:end_window])
        return DnaWindowResult(
            id=self.id,
            chromosome=self.chromosome,
            strand=self.strand,
            start=self.start,
            end=self.end,
            seq=seq,
            start_window=start_window,
            end_window=end_window,
        )

    def window_5utr(
        self, window_size: int, allow_undersize: bool = False
    ) -> DnaWindowResult:
        chr_id = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            start_window = self.start - window_size
            end_window = self.start
            if start_window < 0 and allow_undersize:
                start_window = 0
                end_window = self.start
            elif start_window < 0 and not allow_undersize:
                outside = abs(start_window)
                raise ValueError(
                    f"5utr size ({window_size}) too large ('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(self.fasta_sequences[chr_id].seq[start_window:end_window])
        elif self.strand == "-":
            start_window = self.end
            end_window = self.end + window_size
            if (
                end_window > self.chromosome_lengths[self.chromosome]
                and allow_undersize
            ):
                end_window = self.chromosome_lengths[self.chromosome]
            elif (
                end_window > self.chromosome_lengths[self.chromosome]
                and not allow_undersize
            ):
                outside = abs(end_window - self.chromosome_lengths[self.chromosome])
                raise ValueError(
                    f"5utr size ({window_size}) too large ('{self.strand} strand {outside}bp outside.)"
                )

            seq = str(
                self.fasta_sequences[chr_id]
                .seq[start_window:end_window]
                .reverse_complement()
            )
        return DnaWindowResult(
            id=self.id,
            seq=seq,
            chromosome=self.chromosome,
            start=self.start,
            end=self.end,
            strand=self.strand,
            start_window=start_window,
            end_window=end_window,
        )

    def window_3utr(
        self, window_size: int, allow_undersize: bool = False
    ) -> DnaWindowResult:
        chr_id = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            start_window = self.end
            end_window = self.end + window_size
            if (
                end_window > self.chromosome_lengths[self.chromosome]
                and allow_undersize
            ):
                end_window = self.chromosome_lengths[self.chromosome]
            elif (
                end_window > self.chromosome_lengths[self.chromosome]
                and not allow_undersize
            ):
                outside = abs(end_window - self.chromosome_lengths[self.chromosome])
                raise ValueError(
                    f"3utr size ({window_size}) too large ('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(self.fasta_sequences[chr_id].seq[start_window:end_window])
        elif self.strand == "-":
            start_window = self.start - window_size
            end_window = self.start
            if start_window < 0 and allow_undersize:
                start_window = 0
            elif start_window < 0 and not allow_undersize:
                outside = abs(start_window)
                raise ValueError(
                    f"3utr size ({window_size}) too large ('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(
                self.fasta_sequences[chr_id]
                .seq[start_window:end_window]
                .reverse_complement()
            )

        return DnaWindowResult(
            id=self.id,
            seq=seq,
            chromosome=self.chromosome,
            start=self.start,
            end=self.end,
            strand=self.strand,
            start_window=start_window,
            end_window=end_window,
        )

    def __repr__(self):
        return f"DnaSelectionResult(id={self.id}, chromosome={self.chromosome}, strand={self.strand}, start={self.start}, end={self.end},  seq={self.seq})"


@define
class SCerevisiaeGenome(Genome):
    data_root: str = field(init=True, repr=False, default="data/sgd/genome")
    db: dict[str, SeqRecord] = field(init=False, repr=False)
    fasta_sequences = field(init=False, default=None, repr=False)
    chr_to_nc: dict[str, str] = field(init=False, default=None, repr=False)
    nc_to_chr: dict[str, str] = field(init=False, default=None, repr=False)
    chr_to_len: dict[str, int] = field(init=False, default=None, repr=False)
    _gene_set: SortedSet = field(init=False, default=None, repr=False)
    _fasta_path: str = field(init=False, default=None, repr=False)
    _gff_path: str = field(init=False, default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        self._fasta_path: str = osp.join(
            self.data_root,
            "S288C_reference_genome_R64-3-1_20210421/S288C_reference_sequence_R64-3-1_20210421.fsa",
        )
        self._gff_path: str = osp.join(
            self.data_root,
            "S288C_reference_genome_R64-3-1_20210421/saccharomyces_cerevisiae_R64-3-1_20210421.gff",
        )
        # Create the database
        self.db = gffutils.create_db(
            self._gff_path,
            dbfn=osp.join(self.data_root, "data.db"),
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
        self.nc_to_chr = {v: k for k, v in self.chr_to_nc.items()}
        self.chr_to_len = {
            self.nc_to_chr[chr]: len(self.fasta_sequences[chr].seq)
            for chr in self.fasta_sequences.keys()
        }

        # TODO Not sure if this is now to tightly coupled to GO
        # We do want to remove inaccurate info as early as possible
        # Initialize the GO ontology DAG (Directed Acyclic Graph)
        data_dir = "data/go"
        obo_path = "data/go/go.obo"
        if not osp.exists(obo_path):
            os.makedirs(data_dir, exist_ok=True)
            download_url("http://current.geneontology.org/ontology/go.obo", data_dir)
        self.go_dag = GODag(obo_path)
        # Call the method to remove deprecated GO terms
        # self.remove_deprecated_go_terms()

    def remove_deprecated_go_terms(self):
        # Iterate over each feature in the database
        for feature in self.db.features_of_type("gene"):
            # Check if the feature has the "Ontology_term" attribute
            if "Ontology_term" in feature.attributes:
                # Filter out deprecated GO terms
                valid_go_terms = [
                    term
                    for term in feature.attributes["Ontology_term"]
                    if term.startswith("GO:")
                    and (term not in self.go_dag or not self.go_dag[term].is_obsolete)
                ]

                # Update the "Ontology_term" attribute for the feature
                if valid_go_terms:
                    feature.attributes["Ontology_term"] = valid_go_terms
                else:
                    del feature.attributes["Ontology_term"]

                # Update the feature in the database
                # Assuming `feature` is already the correct type
                self.db.update([feature])

    @property
    def go(self) -> SortedSet[str]:
        go_set = SortedSet()  # Initialize an empty set to hold all GO terms
        for gene_feature in self.db.features_of_type("gene"):
            if "Ontology_term" in gene_feature.attributes:
                go_terms_for_gene = [
                    term
                    for term in gene_feature.attributes["Ontology_term"]
                    if term.startswith("GO:")
                ]
                go_set.update(go_terms_for_gene)

        return go_set

    def go_subset(self, gene_set: SortedSet[str]) -> SortedSet[str]:
        go_subset = SortedSet()
        for gene_feature in self.db.features_of_type("gene"):
            if gene_feature.id in gene_set:
                if "Ontology_term" in gene_feature.attributes:
                    go_terms_for_gene = [
                        term
                        for term in gene_feature.attributes["Ontology_term"]
                        if term.startswith("GO:")
                    ]
                    go_subset.update(go_terms_for_gene)
        return go_subset

    @property
    def go_genes(self) -> SortedDict[str, SortedSet[str]]:
        go_genes_dict = SortedDict()
        for gene_feature in self.db.features_of_type("gene"):
            if "Ontology_term" in gene_feature.attributes:
                go_terms_for_gene = [
                    term
                    for term in gene_feature.attributes["Ontology_term"]
                    if term.startswith("GO:")
                ]
                gene_id = gene_feature.id
                for go_term in go_terms_for_gene:
                    if go_term not in go_genes_dict:
                        go_genes_dict[go_term] = SortedSet()
                    go_genes_dict[go_term].add(gene_id)
        return go_genes_dict

    def go_subset_genes(
        self, gene_set: SortedSet[str]
    ) -> SortedDict[str, SortedSet[str]]:
        go_subset_genes_dict = SortedDict()
        for gene_feature in self.db.features_of_type("gene"):
            if gene_feature.id in gene_set:
                if "Ontology_term" in gene_feature.attributes:
                    go_terms_for_gene = [
                        term
                        for term in gene_feature.attributes["Ontology_term"]
                        if term.startswith("GO:")
                    ]
                    gene_id = gene_feature.id
                    for go_term in go_terms_for_gene:
                        if go_term not in go_subset_genes_dict:
                            go_subset_genes_dict[go_term] = SortedSet()
                        go_subset_genes_dict[go_term].add(gene_id)
        return go_subset_genes_dict

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
            id=self.id,
            chromosome=chr_num,
            strand=strand,
            start=start,
            end=end,
            seq=str(seq),
        )

    def select_feature_window(
        self, feature: str, window_size: int, is_max_size: bool = True
    ) -> DnaWindowResult:
        feature = self[feature]
        if not feature:
            raise ValueError(f"feature {feature} not found.")

        start = feature.start
        end = feature.stop
        strand = feature.strand
        chr = CHROMOSOMES.index(feature.seqid)
        if is_max_size:
            start_window, end_window = calculate_window_bounds(
                start, end, window_size, self.chr_to_len[chr]
            )

        else:
            start_window, end_window = calculate_window_bounds_symmetric(
                start, end, window_size, self.chr_to_len[chr]
            )
        seq = self.get_seq(chr, start_window, end_window, strand).seq
        return DnaWindowResult(
            id=self.id,
            chromosome=chr,
            start=start,
            end=end,
            strand=strand,
            seq=seq,
            size=window_size,
            start_window=start_window,
            end_window=end_window,
        )

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

    def compute_gene_set(self) -> SortedSet[str]:
        genes = [feat.id for feat in list(self.db.features_of_type("gene"))]
        # not yet sure how we will deal with duplicates. There shouldn't be any considering the systematic naming scheme.
        # TODO add test for duplicates.
        assert len(genes) == len(
            set(genes)
        ), "Duplicate genes found... havne't decided how to deal with this yet."
        return SortedSet(genes)

    def __getitem__(self, item: str) -> SCerevisiaeGene | None:
        # For now we only support the systematic names
        try:
            gene = SCerevisiaeGene(
                feature=self.db[item],
                fasta_sequences=self.fasta_sequences,
                chr_to_nc=self.chr_to_nc,
                chromosome_lengths=self.chr_to_len,
            )
            return gene
        except KeyError:
            print(
                f"Gene {item} not found in genome, only systematic names (ID) are supported."
            )
            return None


def main() -> None:
    import os
    import random

    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
    # print(genome.get_sequence(1, 0, 10))  # Replace with valid parameters
    print(
        genome["YFL039C"]
    )  # Replace with valid gene... we only support systematic names

    # Iterate through all gene features and check if they have an Ontology_term attribute

    genome["YFL039C"].window(1000)
    genome["YFL039C"].window(1000, is_max_size=False)
    genome["YFL039C"].window_3utr(1000)
    genome["YFL039C"].window_3utr(1000, allow_undersize=True)
    genome["YFL039C"].window_3utr(1000, allow_undersize=False)
    genome["YFL039C"].window_5utr(1000, allow_undersize=True)
    genome["YFL039C"].window_5utr(1000, allow_undersize=False)
    print(genome.go[:10])


if __name__ == "__main__":
    main()
    pass
