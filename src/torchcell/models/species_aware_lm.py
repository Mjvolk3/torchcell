# src/torchcell/models/species_aware_lm.py
# [[src.torchcell.models.species_aware_lm]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/models/species_aware_lm.py
# Test file: src/torchcell/models/test_species_aware_lm.py


import collections
import itertools
import math
import os
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
)

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


def main():
    genome = SCerevisiaeGenome()

    tokenizer = AutoTokenizer.from_pretrained(
        "gagneurlab/SpeciesLM", revision="downstream_species_lm"
    )
    model = AutoModelForMaskedLM.from_pretrained(
        "gagneurlab/SpeciesLM", revision="downstream_species_lm"
    )

    def kmers_stride1(seq, k=6):
        # splits a sequence into overlapping k-mers
        return [seq[i : i + k] for i in range(0, len(seq) - k + 1)]

    def tok_func_species(x, species_proxy, seq_col):
        res = tokenizer(species_proxy + " " + " ".join(kmers_stride1(x[seq_col])))
        return res

    #
    seq_col = (
        "three_prime_seq"  # name of the column in the df that stores the sequences
    )

    proxy_species = "candida_glabrata"  # species token to use
    target_layer = (8,)  # what hidden layers to use for embedding

    #
    tok_func = lambda x: tok_func_species(x, proxy_species, seq_col)

    # TODO
    tokenizer(proxy_species + " " + " ".join(kmers_stride1(genome["YDR210W"].seq[3:])))
    tokenizer(proxy_species + " " + " ".join(kmers_stride1(genome["YDR210W"].seq[3:])))

    print()
    # TODO

    # TODO overwrite
    ds = Dataset.from_pandas(dataset[[seq_col]])

    tok_ds = ds.map(tok_func, batched=False, num_proc=2)

    rem_tok_ds = tok_ds.remove_columns(seq_col)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    data_loader = torch.utils.data.DataLoader(
        rem_tok_ds, batch_size=1, collate_fn=data_collator, shuffle=False
    )


if __name__ == "__main__":
    main()
