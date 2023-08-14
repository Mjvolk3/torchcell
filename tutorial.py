from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.models import FungalUtrTransformer, NucleotideTransformer

# Getting Genome Data
genome = SCerevisiaeGenome()
print(len(genome))
print(genome["YFL039C"])
print(genome["YFL039C"].seq)
genome["YFL039C"].window(1000)
genome["YFL039C"].window(1000, is_max_size=False)
genome["YFL039C"].window_3utr(300)
genome["YFL039C"].window_3utr(300, allow_undersize=True)
genome["YFL039C"].window_3utr(300, allow_undersize=False)
genome["YFL039C"].window_5utr(1000, allow_undersize=True)
genome["YFL039C"].window_5utr(1000, allow_undersize=False)

# LLMs
# 4 options - ['downstream_300', 'species_downstream_300', 'species_upstream_1000', 'upstream_1000']
fungal_utr_transformer = FungalUtrTransformer("downstream_300")
# nucleotide_transformer = NucleotideTransformer() #takes a while...
sequence = [genome["YFL039C"].window_3utr(300, allow_undersize=True).seq]
mean_embedding = fungal_utr_transformer.embed(sequence, mean_embedding=True)
print(mean_embedding.shape)
print(mean_embedding)