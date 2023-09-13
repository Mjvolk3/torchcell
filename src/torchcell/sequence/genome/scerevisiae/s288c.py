import glob
import gzip
import os
import os.path as osp
import shutil
import tarfile
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
        self.alias = self.feature.attributes.get("Alias", None)
        self.name = self.feature.attributes.get("Name", None)
        self.ontology_term = self.feature.attributes.get("Ontology_term", None)
        self.note = self.feature.attributes.get("Note", None)
        self.display = self.feature.attributes.get("display", None)
        self.dbxref = self.feature.attributes.get("dbxref", None)
        self.orf_classification = self.feature.attributes.get(
            "orf_classification", None
        )

        # Handle GO terms
        if self.ontology_term is not None:
            self.go = SortedSet(
                [term for term in self.ontology_term if term.startswith("GO:")]
            )
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
        self._fasta_path: str = os.path.join(
            self.data_root,
            "S288C_reference_genome_R64-3-1_20210421/S288C_reference_sequence_R64-3-1_20210421.fsa",
        )
        self._gff_path: str = os.path.join(
            self.data_root,
            "S288C_reference_genome_R64-3-1_20210421/saccharomyces_cerevisiae_R64-3-1_20210421.gff",
        )

        # Check if the necessary files exist, if not download them
        if not os.path.exists(self._fasta_path) or not os.path.exists(self._gff_path):
            self.download_and_extract_genome_files()
        db_path = osp.join(self.data_root, "data.db")

        if os.path.exists(db_path):
            self.db = gffutils.FeatureDB(db_path)
        else:
            self.db = gffutils.create_db(
                self._gff_path,
                dbfn=db_path,
                force=True,
                keep_order=True,
                merge_strategy="merge",
                sort_attribute_values=True,
            )
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
        self.remove_deprecated_go_terms()

    def download_and_extract_genome_files(self):
        """
        Download and extract genome files if they do not exist.
        """
        url = "http://sgd-archive.yeastgenome.org/sequence/S288C_reference/genome_releases/S288C_reference_genome_R64-3-1_20210421.tgz"
        save_dir = self.data_root
        download_url(url, save_dir)
        downloaded_file_path = os.path.join(save_dir, url.split("/")[-1])
        self.untar_tgz_file(downloaded_file_path, save_dir)
        self.gunzip_all_files_in_dir(save_dir)

    def untar_tgz_file(self, path_to_input_tgz: str, path_to_output_dir: str):
        """
        Extract a .tgz file
        """
        with tarfile.open(path_to_input_tgz, "r:gz") as tar_ref:
            tar_ref.extractall(path_to_output_dir)
        print(f"Extracted .tgz file to {path_to_output_dir}")
        os.remove(path_to_input_tgz)  # remove the original .tgz file after extraction

    def gunzip_all_files_in_dir(self, directory: str):
        """
        Unzip all .gz files in a directory.
        """
        gz_files = glob.glob(f"{directory}/**/*.gz", recursive=True)
        for gz_file in gz_files:
            with gzip.open(gz_file, "rb") as f_in:
                with open(
                    gz_file[:-3], "wb"
                ) as f_out:  # remove '.gz' from output file name
                    shutil.copyfileobj(f_in, f_out)
            print(f"Unzipped {gz_file}")
            os.remove(gz_file)  # remove the original .gz file

    def remove_deprecated_go_terms(self):
        # Create a list to hold updated features
        updated_features = []

        # Iterate over each feature in the database
        for feature in self.db.features_of_type("gene"):
            # Check if the feature has the "Ontology_term" attribute
            if "Ontology_term" in feature.attributes:
                # Filter out deprecated GO terms
                valid_go_terms = []
                for term in feature.attributes["Ontology_term"]:
                    if term.startswith("GO:"):
                        if term not in self.go_dag:
                            print(f"Removing GO term not found in go_dag: {term}")
                            continue
                        if self.go_dag[term].is_obsolete:
                            print(f"Removing obsolete GO term: {term}")
                            continue
                        valid_go_terms.append(term)

                # Update the "Ontology_term" attribute for the feature
                if valid_go_terms:
                    feature.attributes["Ontology_term"] = valid_go_terms
                else:
                    del feature.attributes["Ontology_term"]

                # Add the updated feature to the list
                updated_features.append(feature)

        # Update all features in the database at once
        self.db.update(updated_features, merge_strategy="replace")

        # Commit the changes to the database
        self.db.conn.commit()

    @property
    def go(self) -> SortedSet[str]:
        all_go = SortedSet()

        # Iterate through all genes in self.gene_set
        for gene_id in self.gene_set:
            gene = self[gene_id]  # Retrieve the gene object

            # Use the go attribute of the gene object if it exists and is not None
            if gene and hasattr(gene, "go") and gene.go is not None:
                all_go.update(gene.go)

        return all_go

    def go_subset(self, gene_set: SortedSet[str]) -> SortedSet[str]:
        go_subset = SortedSet()

        # Iterate through the provided subset of genes
        for gene_id in gene_set:
            gene = self[gene_id]  # Retrieve the gene object

            # Use the go attribute of the gene object if it exists and is not None
            if gene and hasattr(gene, "go") and gene.go is not None:
                go_subset.update(gene.go)

        return go_subset

    @property
    def go_genes(self) -> SortedDict[str, SortedSet[str]]:
        go_genes_dict = SortedDict()

        # Iterate through all genes in self.gene_set
        for gene_id in self.gene_set:
            gene = self[gene_id]  # Retrieve the gene object

            # Use the go attribute of the gene object if it exists and is not None
            if gene and hasattr(gene, "go") and gene.go is not None:
                for go_term in gene.go:
                    if go_term not in go_genes_dict:
                        go_genes_dict[go_term] = SortedSet()
                    go_genes_dict[go_term].add(gene_id)

        return go_genes_dict

    def go_subset_genes(
        self, gene_set: SortedSet[str]
    ) -> SortedDict[str, SortedSet[str]]:
        go_subset_genes_dict = SortedDict()

        # Iterate through the provided subset of genes
        for gene_id in gene_set:
            gene = self[gene_id]  # Retrieve the gene object

            # Use the go attribute of the gene object if it exists and is not None
            if gene and hasattr(gene, "go") and gene.go is not None:
                for go_term in gene.go:
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
        assert len(genes) == len(
            set(genes)
        ), "Duplicate genes found... chekc handled by gff."
        return SortedSet(genes)

    def drop_chrmt(self) -> None:
        mitochondrial_features = [
            f for f in self.db.all_features() if f.seqid == "chrmt"
        ]

        # Remove these features from the gene set cache if it existsc
        if self._gene_set is not None:
            for feature in mitochondrial_features:
                self._gene_set.discard(feature.id)

        # Remove these features from the database
        for feature in mitochondrial_features:
            self.db.delete(feature.id, feature_type=feature.featuretype)

        # Commit the changes to the database
        self.db.conn.commit()

    def drop_empty_go(self) -> None:
        # Initialize a list to hold genes to be removed
        genes_to_remove = []

        # Iterate through all genes in the current gene_set
        for gene_id in self.gene_set:
            gene = self[gene_id]
            if gene is not None:
                # Check if the GO terms are empty
                # None case for never annotated, 0 for no GO terms
                if gene.go is None or len(gene.go) == 0:
                    genes_to_remove.append(gene_id)

        # Remove these genes from the gene set cache
        for gene_id in genes_to_remove:
            self._gene_set.discard(gene_id)

        # Remove these genes from the database
        for gene_id in genes_to_remove:
            self.db.delete(gene_id, feature_type="gene")

        # Commit the changes to the database
        self.db.conn.commit()

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
    genome.go
    # print(genome.get_sequence(1, 0, 10))  # Replace with valid parameters # 4903
    print(
        genome["YFL039C"]
    )  # Replace with valid gene... we only support systematic names

    # Iterate through all gene features and check if they have an Ontology_term attribute
    print(len(genome.gene_set))
    genome.drop_chrmt()
    print(len(genome.gene_set))
    genome.drop_empty_go()
    print(len(genome.gene_set))
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
