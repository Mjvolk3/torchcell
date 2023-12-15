# torchcell/sequence/genome/scerevisiae/s288c.py
# [[torchcell.sequence.genome.scerevisiae.s288c]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/scerevisiae/s288c.py
# Test file: torchcell/sequence/genome/scerevisiae/test_s288c.py

import glob
import gzip
import logging
import os
import os.path as osp
import shutil
import tarfile
from itertools import product
from typing import Set

import gffutils
import pandas as pd
from attrs import define, field
from Bio import Seq, SeqIO
from Bio.SeqRecord import SeqRecord
from gffutils import FeatureDB
from gffutils.feature import Feature
from goatools.obo_parser import GODag
from sortedcontainers import SortedDict, SortedSet
from torch_geometric.data import download_url

from torchcell.sequence import (
    DnaSelectionResult,
    DnaWindowResult,
    Gene,
    GeneSet,
    Genome,
    calculate_window_bounds,
    calculate_window_bounds_symmetric,
    compute_codon_frequency,
    get_chr_from_description,
    mismatch_positions,
    roman_to_int,
)

log = logging.getLogger(__name__)

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


nucleotides = ["A", "T", "G", "C"]
all_codons = ["".join(codon) for codon in product(nucleotides, repeat=3)]


@define
class SCerevisiaeGene(Gene):
    id: str = field(repr=False)
    db: str = field(repr=False)
    fasta_dna: dict[str, SeqRecord] = field(repr=False)
    fasta_protein: dict[str, SeqRecord] = field(repr=False)
    fasta_cds: dict[str, SeqRecord] = field(repr=False)
    chr_to_nc: dict[str, str] = field(repr=False)
    chromosome_lengths: dict[str, int] = field(repr=False)
    # below are set in __attrs_post_init__
    chromosome: int = field(default=None)
    start: int = field(default=None)
    end: int = field(default=None)
    seq: str = field(default=None, repr=True)
    feature: Feature = field(default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        # process the feature region and produce a feature
        feature_region = self.db.region(
            region=(
                self.db[self.id].chrom,
                self.db[self.id].start,
                self.db[self.id].end,
            ),
            completely_within=True,
        )
        features = [feature for feature in feature_region]
        contains_five_prime_UTR_intron = False
        no_middle_intron = True
        not_chrmt = self.db[self.id].chrom != "chrmt"
        for some_feature in features:
            if some_feature.featuretype == "five_prime_UTR_intron":
                contains_five_prime_UTR_intron = True
                five_prime_UTR_intron_feature = some_feature

        # 4 genes with single bp CDS 5prime
        no_five_prime_one_bp_cds = True
        for some_feature in features:
            if some_feature.featuretype == "CDS":
                cds_difference = some_feature.start - some_feature.end
                if (
                    some_feature.strand == "+"
                    and cds_difference == 0
                    and self.db[self.id].start == some_feature.start
                ):
                    no_five_prime_one_bp_cds = False

                elif (
                    some_feature.strand == "-"
                    and cds_difference == 0
                    and self.db[self.id].end == some_feature.end
                ):
                    no_five_prime_one_bp_cds = False

            self.db[self.id].start
        # No guarantee introns are same as five_prime_UTR_intron
        if contains_five_prime_UTR_intron:
            if (
                five_prime_UTR_intron_feature.start > self.db[self.id].start
                and five_prime_UTR_intron_feature.end < self.db[self.id].end
            ):
                no_middle_intron = False

        # TODO logic is bit complicated, might want to abstract away.
        if (
            contains_five_prime_UTR_intron
            and no_middle_intron
            and not_chrmt
            and no_five_prime_one_bp_cds
        ):
            cds_features = [
                feature for feature in features if feature.featuretype == "CDS"
            ]
            if len(cds_features) == 1:
                feature = cds_features[0]
            # sometimes we have more than one CDS, we need to select the one we have most confidence in with "Verified" ORF
            elif len(cds_features) > 1:
                verified_orfs = [
                    feature
                    for feature in cds_features
                    if feature.attributes["orf_classification"][0] == "Verified"
                ]
                if len(verified_orfs) == 1:
                    feature = verified_orfs[0]
                if len(verified_orfs) > 1:
                    feature = Feature()
                    feature.chrom = self.db[self.id].chrom
                    feature.strand = self.db[self.id].strand
                    feature.start = min([feature.start for feature in verified_orfs])
                    feature.end = max([feature.end for feature in verified_orfs])
            assert isinstance(feature, Feature), "feature is not a gffutils Feature"
            # log.warning(f"{self.id} - Using CDS Sequence")
        else:
            feature = self.db[self.id]
        gene_feature = self.db[self.id]

        #
        self.id = self.id
        # chromosome
        seqid = gene_feature.seqid
        maybe_roman_numeral = seqid.split("chr")[-1]
        if maybe_roman_numeral == "mt":
            self.chromosome = 0
        else:
            self.chromosome = roman_to_int(maybe_roman_numeral)
        # others
        self.start = feature.start
        self.end = feature.end
        self.strand = feature.strand

        # dna sequence
        chr = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            self.seq = str(self.fasta_dna[chr].seq[self.start - 1 : self.end])
        elif self.strand == "-":
            self.seq = str(
                self.fasta_dna[chr].seq[self.start - 1 : self.end].reverse_complement()
            )

        # protein sequence
        self.protein = self.fasta_protein.get(self.id)
        # cds sequence
        self.cds = self.fasta_cds.get(self.id)

        # TODO consider adding these to ABC...
        # Some might be too specific to S. cerevisiae, but so they could be optional
        # Must use the gene since it has all of the annotations, not the OverflowError
        self.alias = gene_feature.attributes.get("Alias", None)
        self.name = gene_feature.attributes.get("Name", None)
        self.ontology_term = gene_feature.attributes.get("Ontology_term", None)
        self.note = gene_feature.attributes.get("Note", None)
        self.display = gene_feature.attributes.get("display", None)
        self.dbxref = gene_feature.attributes.get("dbxref", None)
        self.orf_classification = gene_feature.attributes.get(
            "orf_classification", None
        )

        # Handle GO terms
        if self.ontology_term is not None:
            self.go = SortedSet(
                [term for term in self.ontology_term if term.startswith("GO:")]
            )
        else:
            self.go = None

    # use
    @property
    def codon_frequency(self) -> SortedDict[str, float]:
        codon_frequency = compute_codon_frequency(self.cds.seq)
        return codon_frequency

    def window(self, window_size: int, is_max_size: bool = True) -> DnaWindowResult:
        if is_max_size:
            start_window, end_window = calculate_window_bounds(
                start=self.start - 1,
                end=self.end,
                strand=self.strand,
                window_size=window_size,
                chromosome_length=self.chromosome_lengths[self.chromosome],
            )

        else:
            start_window, end_window = calculate_window_bounds_symmetric(
                start=self.start - 1,
                end=self.end,
                window_size=window_size,
                chromosome_length=self.chromosome_lengths[self.chromosome],
            )
        chr_id = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            seq = str(self.fasta_dna[chr_id].seq[start_window:end_window])
        elif self.strand == "-":
            seq = str(
                self.fasta_dna[chr_id].seq[start_window:end_window].reverse_complement()
            )
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

    def window_five_prime(
        self,
        window_size: int,
        include_start_codon: bool = False,
        allow_undersize: bool = False,
    ) -> DnaWindowResult:
        # offset for gff file 1
        start = self.start - 1
        chr_id = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            if include_start_codon:
                start = start + 3
            else:
                start = self.start
            start_window = start - window_size
            end_window = start
            if start_window < 0 and allow_undersize:
                start_window = 0
                end_window = start
            elif start_window < 0 and not allow_undersize:
                outside = abs(start_window)
                raise ValueError(
                    f"5utr size ({window_size}) too large ('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(self.fasta_dna[chr_id].seq[start_window:end_window])
        elif self.strand == "-":
            if include_start_codon:
                end = self.end - 3
            else:
                end = self.end
            start_window = end
            end_window = end + window_size
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
                self.fasta_dna[chr_id].seq[start_window:end_window].reverse_complement()
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

    def window_three_prime(
        self,
        window_size: int,
        include_stop_codon: bool = False,
        allow_undersize: bool = False,
    ) -> DnaWindowResult:
        # offset for gff file 1
        start = self.start - 1
        chr_id = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            if include_stop_codon:
                end = self.end - 3
            else:
                end = self.end
            start_window = end
            end_window = end + window_size
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
                    f"3utr size ({window_size}) too large"
                    f"('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(self.fasta_dna[chr_id].seq[start_window:end_window])
        elif self.strand == "-":
            if include_stop_codon:
                start = start + 3
            else:
                start = start
            start_window = start - window_size
            end_window = start
            if start_window < 0 and allow_undersize:
                start_window = 0
            elif start_window < 0 and not allow_undersize:
                outside = abs(start_window)
                raise ValueError(
                    f"3utr size ({window_size}) too large"
                    f"('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(
                self.fasta_dna[chr_id].seq[start_window:end_window].reverse_complement()
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
    overwrite: bool = field(init=True, repr=True, default=True)
    db: dict[str, SeqRecord] = field(init=False, repr=False)
    fasta_dna = field(init=False, default=None, repr=False)
    chr_to_nc: dict[str, str] = field(init=False, default=None, repr=False)
    nc_to_chr: dict[str, str] = field(init=False, default=None, repr=False)
    chr_to_len: dict[str, int] = field(init=False, default=None, repr=False)
    _gene_set: GeneSet = field(init=False, default=None, repr=False)
    _dna_fasta_path: str = field(init=False, default=None, repr=False)
    _protein_fasta_path: str = field(init=False, default=None, repr=False)
    _cds_fasta_path: str = field(init=False, default=None, repr=False)
    _gff_path: str = field(init=False, default=None, repr=False)
    _go: SortedSet[str] = field(init=False, default=None, repr=False)
    _go_genes: SortedDict[str, SortedSet[str]] = field(
        init=False, default=None, repr=False
    )

    def __attrs_post_init__(self) -> None:
        genome_reference = "S288C_reference_genome"
        self.genome_version = "R64-4-1_20230830"
        self.sgd_base_url = "http://sgd-archive.yeastgenome.org"
        self.sequence_s288c = "sequence/S288C_reference"
        self.genome_version_full = genome_reference + "_" + self.genome_version

        self._dna_fasta_path: str = osp.join(
            self.data_root,
            self.genome_version_full,
            "S288C_reference_sequence_" + self.genome_version + ".fsa",
        )
        self._gff_path: str = osp.join(
            self.data_root,
            self.genome_version_full,
            "saccharomyces_cerevisiae_" + self.genome_version + ".gff",
        )
        self._protein_fasta_path = osp.join(
            self.data_root,
            self.genome_version_full,
            "orf_trans_all_" + self.genome_version + ".fasta",
        )
        self._cds_fasta_path = osp.join(
            self.data_root,
            self.genome_version_full,
            "orf_coding_all_" + self.genome_version + ".fasta",
        )
        # Download genome data
        if not osp.exists(self._dna_fasta_path) or not osp.exists(self._gff_path):
            self.download_and_extract_genome_files()

        db_path = osp.join(self.data_root, "data.db")

        # CHECK if this works with ddp
        if osp.exists(db_path) and not self.overwrite:
            self.db = gffutils.FeatureDB(db_path)
        elif self.overwrite:
            # TODO remove sort_attribute_values since this can be time consuming.
            self.db = gffutils.create_db(
                self._gff_path,
                dbfn=db_path,
                force=True,
                keep_order=True,
                merge_strategy="merge",
                sort_attribute_values=True,
            )

        self.fasta_dna = SeqIO.to_dict(SeqIO.parse(self._dna_fasta_path, "fasta"))
        self.fasta_protein = SeqIO.to_dict(
            SeqIO.parse(self._protein_fasta_path, "fasta")
        )
        self.fasta_cds = SeqIO.to_dict(SeqIO.parse(self._cds_fasta_path, "fasta"))
        # Create mapping from chromosome number to sequence identifier
        self.chr_to_nc = {
            get_chr_from_description(self.fasta_dna[key].description): key
            for key in self.fasta_dna.keys()
        }
        self.nc_to_chr = {v: k for k, v in self.chr_to_nc.items()}
        self.chr_to_len = {
            self.nc_to_chr[chr]: len(self.fasta_dna[chr].seq)
            for chr in self.fasta_dna.keys()
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
        # BUG this line doesn't work with ddp, I think the issue is merge=replace
        # self.remove_deprecated_go_terms()

    def download_and_extract_genome_files(self):
        """
        Download and extract genome files if they do not exist.
        """
        zipped_version = f"{self.genome_version_full}.tgz"
        url = osp.join(
            self.sgd_base_url, self.sequence_s288c, "genome_releases", zipped_version
        )

        save_dir = self.data_root
        download_url(url, save_dir)
        downloaded_file_path = osp.join(save_dir, url.split("/")[-1])
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
        invalid_go_terms = {"not_in_go_dag": [], "obsolete": []}
        for feature in self.db.features_of_type("gene"):
            # Check if the feature has the "Ontology_term" attribute
            if "Ontology_term" in feature.attributes:
                # Filter out deprecated GO terms
                valid_onto_terms = []
                valid_go_terms = []
                for term in feature.attributes["Ontology_term"]:
                    if term.startswith("GO:"):
                        if term not in self.go_dag:
                            invalid_go_terms["not_in_go_dag"].append(term)
                        elif self.go_dag[term].is_obsolete:
                            invalid_go_terms["obsolete"].append(term)
                        else:
                            valid_go_terms.append(term)
                    else:
                        valid_onto_terms.append(term)
                # Update the "Ontology_term" attribute for the feature
                if valid_go_terms:
                    feature.attributes["Ontology_term"] = (
                        valid_go_terms + valid_onto_terms
                    )
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
        if self._go is None:
            all_go = SortedSet()

            # Iterate through all genes in self.gene_set
            for gene_id in self.gene_set:
                gene = self[gene_id]  # Retrieve the gene object

                # Use the go attribute of the gene object if it exists and is not None
                if gene and hasattr(gene, "go") and gene.go is not None:
                    all_go.update(gene.go)
            self._go = all_go
            return self._go
        else:
            return self._go

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
        # CHECK could this contain obselete terms? We don't check if the terms are in self.go...
        if self._go_genes is None:
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
            self._go_genes = go_genes_dict
            return go_genes_dict
        else:
            return self._go_genes

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
            seq = self.fasta_dna[chr].seq[start:end]
        elif strand == "-":
            seq = self.fasta_dna[chr].seq[start:end].reverse_complement()
        return DnaSelectionResult(
            id=self.id,
            chromosome=chr_num,
            strand=strand,
            start=start,
            end=end,
            seq=str(seq),
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
        return GeneSet(genes)

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
        # ising region instead, since it give more options on dealing with gene processing in gene class
        try:
            gene = SCerevisiaeGene(
                id=item,
                db=self.db,
                fasta_dna=self.fasta_dna,
                fasta_protein=self.fasta_protein,
                fasta_cds=self.fasta_cds,
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

    import matplotlib.pyplot as plt
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    print()
    # orf_classes = []
    # lengths = []
    # for gene in genome.gene_set:
    #     orf_classes.append(genome[gene].orf_classification[0])
    #     lengths.append(len(genome[gene].protein.seq))
    # print(pd.Series(orf_classes).value_counts())
    # genome.go
    genome.drop_chrmt()
    # print(len(genome.gene_set))
    genome.drop_empty_go()

    # # genes_not_divisible_by_3 = [
    # #     gene for gene in genome.gene_set if len(genome[gene]) % 3 != 0
    # # ]
    # # print(len(genes_not_divisible_by_3))
    # # print(genes_not_divisible_by_3)
    # # genes_no_start = [
    # #     gene for gene in genes_not_divisible_by_3 if genome[gene].seq[:3] != "ATG"
    # # ]
    # # print(len(genes_no_start))
    # # print(genes_no_start)
    # # print()

    # not_divisible_by_3 = []
    # for gene in genome.gene_set:
    #     if len(str(genome["YIL111W"].cds.seq)) % 3 != 0:
    #         not_divisible_by_3.append(gene)
    # print(len(not_divisible_by_3))
    # print(compute_codon_frequency(str(genome["YIL111W"].cds.seq)))
    print()


if __name__ == "__main__":
    main()
