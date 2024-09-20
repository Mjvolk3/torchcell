# experiments/S288C_genes_multiple_CDS.py
# [[experiments.S288C_genes_multiple_CDS]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/S288C_genes_multiple_CDS.py
# Test file: experiments/test_S288C_genes_multiple_CDS.py


from torchcell.sequence.genome.scerevisiae.S288C import SCerevisiaeGenome
import numpy as np
import pandas as pd
from collections import defaultdict
from torchcell.sequence.genome.scerevisiae.S288C import SCerevisiaeGenome
import numpy as np


def main():
    genome = SCerevisiaeGenome()
    genome.drop_chrmt()
    genome.drop_empty_go()
    stop_codons = ["TAA", "TAG", "TGA"]
    start_codon = "ATG"

    # Dictionary to hold the count of CDS features per gene
    cds_count_per_gene = defaultdict(int)
    # List to keep track of genes with more than one CDS
    genes_with_multiple_cds = []

    for gene in genome.gene_set:
        features = [
            feature
            for feature in genome.db.region(
                region=(
                    genome.db[gene].chrom,
                    genome.db[gene].start,
                    genome.db[gene].stop,
                )
            )
        ]
        for feature in features:
            if feature.featuretype == "CDS":
                cds_count_per_gene[gene] += 1

        if cds_count_per_gene[gene] > 1:
            genes_with_multiple_cds.append(gene)

    print(f"Number of genes with multiple CDS: {len(genes_with_multiple_cds)}")
    print(f"Genes with multiple CDS: {genes_with_multiple_cds}")


if __name__ == "__main__":
    main()
