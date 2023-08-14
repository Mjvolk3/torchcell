# genome.py
import glob
import gzip
import os
import shutil
import tarfile
from typing import List

import requests


def download_file(url: str, save_dir: str) -> str:
    """
    Download a file from the specified URL to the save directory.

    Parameters:
        url (str): URL of the file to be downloaded.
        save_dir (str): Directory where the file will be saved.

    Returns:
        str: Path of the downloaded file.
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # The path to the file once it's downloaded
    save_path = os.path.join(save_dir, url.split("/")[-1])

    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check that the request was successful
    response.raise_for_status()

    # Write the contents of the response to a file
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Downloaded file to {save_path}")

    return save_path


def untar_tgz_file(path_to_input_tgz: str, path_to_output_dir: str) -> None:
    """
    Extract a .tgz file

    Parameters:
        path_to_input_tgz (str): The path to the .tgz file.
        path_to_output_dir (str): The path to the output directory.

    Returns:
        None
    """
    with tarfile.open(path_to_input_tgz, "r:gz") as tar_ref:
        tar_ref.extractall(path_to_output_dir)
    print(f"Extracted .tgz file to {path_to_output_dir}")
    os.remove(path_to_input_tgz)  # remove the original .tgz file after extraction


def gunzip_all_files_in_dir(directory: str) -> None:
    """
    Unzip all .gz files in a directory.

    Parameters:
        directory (str): The path to the directory containing .gz files.

    Returns:
        None
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


def main(url: str, save_dir: str) -> None:
    """
    Download, extract and unzip files from a given url.

    Parameters:
        url (str): The URL of the .tgz file to download.
        save_dir (str): The directory to save and extract files.

    Returns:
        None
    """
    downloaded_file_path = download_file(url, save_dir)
    untar_tgz_file(downloaded_file_path, save_dir)
    gunzip_all_files_in_dir(save_dir)


if __name__ == "__main__":
    url = "http://sgd-archive.yeastgenome.org/sequence/S288C_reference/genome_releases/S288C_reference_genome_R64-3-1_20210421.tgz"
    save_dir = "data/sgd/genome"
    main(url, save_dir)
