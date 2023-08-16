import requests
import os
import zipfile

url = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_000146045.2/download"

params = {
    "include_annotation_type": "GENOME_FASTA,GENOME_GFF,RNA_FASTA,CDS_FASTA,PROT_FASTA,SEQUENCE_REPORT",
    "filename": "GCF_000146045.2.zip",
}

headers = {"Accept": "application/zip"}

response = requests.get(url, headers=headers, params=params, stream=True)

# Specify the directory where you want to save the file
directory = "data/ncbi/s_cerevisiae/"

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Now the file will be saved to the specified directory
filename = directory + params["filename"]

with open(filename, "wb") as d:
    for chunk in response.iter_content(chunk_size=128):
        d.write(chunk)

# Unzip the file in the same location
with zipfile.ZipFile(filename, "r") as zip_ref:
    zip_ref.extractall(directory)

# Delete the .zip file
os.remove(filename)
