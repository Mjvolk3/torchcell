# torchcell/models/species_aware_lm.py
# [[torchcell.models.species_aware_lm]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/species_aware_lm.py
# Test file: torchcell/models/test_species_aware_lm.py


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


def embed_sequence(sequence: str) -> np.ndarray:
    def kmers_stride1(seq, k=6):
        return [seq[i : i + k] for i in range(0, len(seq) - k + 1)]

    # Tokenizing the sequence
    proxy_species = "candida_glabrata"
    tokenized_data = tokenizer(
        proxy_species + " " + " ".join(kmers_stride1(sequence)),
        return_tensors="pt",  # Return PyTorch tensors
    )

    # Running model and getting embeddings
    with torch.no_grad():
        outputs = model(**tokenized_data, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    target_layer = 8  # Use the layer that suits your needs

    if isinstance(target_layer, int):
        embedding = hidden_states[target_layer]
    else:
        # If target_layer is a tuple or list, average over the specified range of layers
        embedding = torch.mean(
            torch.stack(hidden_states[target_layer[0] : target_layer[1] + 1]), dim=0
        )

    # Averaging over tokens to get a single vector and converting it to a NumPy array
    averaged_embedding = embedding.mean(dim=1).cpu().numpy()

    return averaged_embedding


# Initialize tokenizer and model outside the function
tokenizer = AutoTokenizer.from_pretrained(
    "gagneurlab/SpeciesLM", revision="downstream_species_lm"
)
model = AutoModelForMaskedLM.from_pretrained(
    "gagneurlab/SpeciesLM", revision="downstream_species_lm"
)
model.eval()  # Set the model to evaluation mode


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

    # I want a function that would all me to run embed(genome[gene].window_three_prime(300, include_stop_codon=True, allow_undersize=True).seq) and it would return the embedded vector

    genes_three_prime = []
    for gene in genome.gene_set:
        genes_three_prime.append(
            genome[gene]
            .window_three_prime(300, include_stop_codon=True, allow_undersize=True)
            .seq
        )
    dataset = pd.DataFrame({seq_col: genes_three_prime})

    ds = Dataset.from_pandas(dataset[[seq_col]])

    tok_ds = ds.map(tok_func, batched=False, num_proc=2)

    rem_tok_ds = tok_ds.remove_columns(seq_col)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    data_loader = torch.utils.data.DataLoader(
        rem_tok_ds, batch_size=1, collate_fn=data_collator, shuffle=False
    )
    # Running model

    # CHECK needed for running model?
    def count_special_tokens(tokens, tokenizer, where="left"):
        count = 0
        if where == "right":
            tokens = tokens[::-1]
        for pos in range(len(tokens)):
            tok = tokens[pos]
            if tok in tokenizer.all_special_ids:
                count += 1
            else:
                break
        return count

    def embed_on_batch(
        tokenized_data,
        dataset,
        seq_idx,
        special_token_offset,
        target_layer=target_layer,
    ):
        label = dataset.iloc[seq_idx][seq_col]
        label_len = len(label)
        if label_len < 6:
            print("This should not occur")
            return torch.zeros(label_len, label_len, 768)
        else:
            res = tokenized_data["input_ids"].clone()
            # can change later
            device = "cpu"
            res = res.to(device)
            with torch.no_grad():
                embedding = model(res, output_hidden_states=True)["hidden_states"]
        if isinstance(target_layer, int):
            embedding = embedding[target_layer]
        elif len(target_layer) == 1:
            embedding = torch.stack(embedding[target_layer[0] :], axis=0)
            embedding = torch.mean(embedding, axis=0)
        else:
            embedding = torch.stack(
                embedding[target_layer[0] : target_layer[1]], axis=0
            )
            embedding = torch.mean(embedding, axis=0)
        embedding = embedding.detach().cpu().numpy()
        return embedding

    averaged_embeddings = []

    for no_of_index, tokenized_data in tqdm.tqdm(enumerate(data_loader)):
        label = dataset.iloc[no_of_index][seq_col]
        label_len = len(label)

        left_special_tokens = count_special_tokens(
            tokenized_data["input_ids"].numpy()[0], tokenizer, where="left"
        )
        # TODO why are these computed? Are they needed for upstream?
        # right_special_tokens = count_special_tokens(
        #     tokenized_data["input_ids"].numpy()[0], tokenizer, where="right"
        # )
        assert label_len > 11, "Cannot embed sequences < 11"

        hidden_states = embed_on_batch(
            tokenized_data,
            dataset,
            no_of_index,
            special_token_offset=left_special_tokens,
        )
        avg = hidden_states.mean(
            axis=(0, 1)
        )  # breakpoint here, hidden state size is (1, 298, 768)

        averaged_embeddings.append(avg)
        if no_of_index > 10:
            break
    print()


if __name__ == "__main__":
    # main()
    # main()
    # single embed
    # Get genome sequence. Replace the next line with the actual way you are getting the sequence.
    genome = SCerevisiaeGenome()
    genome.drop_chrmt()
    genome.drop_empty_go()
    count = 0
    lens = []
    for gene in genome.gene_set:
        try:
            genome[gene].window_five_prime(
                1003, include_start_codon=True, allow_undersize=False
            )
        except ValueError:
            count += 1
            lens.append(
                len(
                    genome[gene]
                    .window_five_prime(
                        1003, include_start_codon=True, allow_undersize=True
                    )
                    .seq
                )
            )

    genome_sequence = (
        genome["YDR210W"]
        .window_three_prime(300, include_stop_codon=True, allow_undersize=True)
        .seq
    )

    # Get the embedded vector by calling the function
    embedded_vector = embed_sequence(genome_sequence)
    print()
