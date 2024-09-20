# experiments/S288C_stop_start_codons.py
# [[experiments.S288C_stop_start_codons]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/S288C_stop_start_codons.py
# Test file: experiments/test_S288C_stop_start_codons.py

from torchcell.sequence.genome.scerevisiae.S288C import SCerevisiaeGenome
import numpy as np
import pandas as pd
from collections import defaultdict
from torchcell.sequence.genome.scerevisiae.S288C import SCerevisiaeGenome
import numpy as np


def main():
    genome = SCerevisiaeGenome()
    # genome.drop_chrmt()
    # genome.drop_empty_go()
    stop_codons = ["TAA", "TAG", "TGA"]
    start_codon = "ATG"
    #
    has_start_codon = []
    has_stop_codon = []
    #
    genes_no_start_codon = []
    genes_no_stop_codon = []
    #
    genes_no_start_codon_has_5utr_intron = []
    genes_no_start_codon_has_cds = []
    for gene in genome.gene_set:
        if genome[gene].seq[:3] == start_codon:
            has_start_codon.append(True)
        else:
            has_start_codon.append(False)
            genes_no_start_codon.append(gene)
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
            if "CDS" in [feature.featuretype for feature in features]:
                genes_no_start_codon_has_cds.append(gene)
            if "five_prime_UTR_intron" in [feature.featuretype for feature in features]:
                genes_no_start_codon_has_5utr_intron.append(gene)
        if genome[gene].seq[-3:] in stop_codons:
            has_stop_codon.append(True)
        else:
            has_stop_codon.append(False)
            genes_no_stop_codon.append(gene)

    print()

    assert np.sum(has_start_codon) == len(
        has_start_codon
    ), "Not all genes have a start codon"

    assert np.sum(has_stop_codon) == len(
        has_stop_codon
    ), "Not all genes have a stop codon"


if __name__ == "__main__":
    main()
