# torchcell/sequence/genome/scerevisiae/s288c_gb.py
# [[torchcell.sequence.genome.scerevisiae.s288c_gb]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/scerevisiae/s288c_gb.py
# Test file: torchcell/sequence/genome/scerevisiae/test_s288c_gb.py

import glob
import gzip
import logging
import os
import os.path as osp
import shutil
import subprocess
import tarfile
import zipfile
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

from torchcell.sequence import (  # Genome,
    DnaSelectionResult,
    DnaWindowResult,
    Gene,
    GeneSet,
    calculate_window_bounds,
    calculate_window_bounds_symmetric,
    get_chr_from_description,
    mismatch_positions,
    roman_to_int,
)


@define
class SCerevisiaeGenome:
    data_root: str = field(init=True, repr=False, default="data/sgd/genome")
    # Additional field to store genomes
    genomes: SortedDict = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.genomes = SortedDict()

        # Define the exact location of your .gbff file
        gbff_file_path = os.path.join(
            self.data_root, "ncbi_dataset/data/GCF_000146045.2/genomic.gbff"
        )

        # If the .gbff file doesn't exist, download and extract it
        if not os.path.exists(gbff_file_path):
            print("The .gbff file doesn't exist. Downloading now.")
            self.download_and_extract_genome_files()

        # Read the .gbff file
        self.read_single_gbff_file()

    def read_single_gbff_file(self) -> None:
        # Reusing the gbff_file_path
        gbff_file_path = os.path.join(
            self.data_root, "ncbi_dataset/data/GCF_000146045.2/genomic.gbff"
        )

        # Check if the .gbff file exists
        if not os.path.exists(gbff_file_path):
            print(f"The file {gbff_file_path} does not exist.")
            return

        # Parse the .gbff file and store the result
        for record in SeqIO.parse(gbff_file_path, "genbank"):
            self.genomes[record.id] = record
        print()

    def download_and_extract_genome_files(self) -> None:
        # Full path to the download location
        download_path = os.path.abspath("GCF_000146045.2.zip")

        print(f"Downloading to {download_path}")

        # Download the dataset
        try:
            subprocess.run(
                [
                    "datasets",
                    "download",
                    "genome",
                    "accession",
                    "GCF_000146045.2",
                    "--include",
                    "gbff",
                    "--filename",
                    download_path,
                ]
            )
        except Exception as e:
            print(f"Failed to download: {e}")
            return

        # Check if the zip file is present
        if not os.path.exists(download_path):
            print("The zip file was not downloaded correctly.")
            return

        print("Zip file exists, attempting to unzip.")

        # Unzip the downloaded file
        try:
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(".")
            print("Unzip operation completed.")
        except Exception as e:
            print(f"Failed to unzip: {e}")
            return

        # Print the directory contents to debug
        for root, dirs, files in os.walk("./GCF_000146045.2"):
            print(root)
            print("Directories:", dirs)
            print("Files:", files)
            print("---")

        # Define the destination directory
        destination_directory = os.path.abspath("data/scerevisiae/genome/s288c/")

        # Create destination directory if it does not exist
        os.makedirs(destination_directory, exist_ok=True)

        print(f"Moving to {destination_directory}")

        # Move the 'ncbi_dataset' to the destination directory
        try:
            shutil.move("./ncbi_dataset", destination_directory)
            print("Move operation completed.")
        except Exception as e:
            print(f"An error occurred while moving the file: {e}")
            return

        # Remove the ZIP file and temporary directory
        try:
            os.remove(download_path)
            print("Cleanup operation completed.")
        except Exception as e:
            print(f"An error occurred while cleaning up: {e}")


def main():
    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/scerevisiae/genome/s288c")
    )
    pass


if __name__ == "__main__":
    main()
