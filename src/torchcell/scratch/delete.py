import json

import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from torchcell.data_priorsequence import SCerevisiaeGenome
from torchcell.datasets.scerevisiae import Baryshnikovna2010Dataset
from torchcell.models import DeepSet
from torchcell.sequence import roman_to_int

# Genome and dataset
genome = SCerevisiaeGenome()
dataset = Baryshnikovna2010Dataset()

# find overlap in datasetset and genome
dataset_ids = [id[0].split("_")[0] for id in dataset.genotype["ids"]]
gene_intersection = list(set(genome.gene_list).intersection(set(dataset_ids)))
print(f"len(gene_intersection): {len(gene_intersection)}")
# Get sequence data
data = {}
for i, gene in tqdm(enumerate(gene_intersection)):
    gene_feature = genome[gene]
    chr = roman_to_int((gene_feature.chrom).split("chr")[-1])
    start = gene_feature.start
    stop = gene_feature.stop
    strand = gene_feature.strand
    sequence = genome.get_seq(chr, start, stop, strand).seq
    if len(sequence) > 5994:
        print(f"skipping gene: {gene}, length: {len(sequence)}")
        continue
    # build data
    gene_data = {
        "index": i,
        "gene": gene,
        "sequence": sequence,
        "fitness": float(
            [
                dataset[i].phenotype["fitness"]
                for i in range(len(dataset))
                if dataset[i].genotype["ids"][0].split("_")[0] == gene
            ][0]
        ),
    }
    data[gene] = gene_data  # Use the gene as the key for each gene_data dictionary

# Dump the data dictionary to a JSON file
with open("data/bary_data.json", "w") as f:
    json.dump(data, f)


def main():
    print()


if __name__ == "__main__":
    main()
