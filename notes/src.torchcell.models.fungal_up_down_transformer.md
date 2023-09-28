---
id: 8qkvwbscuf4ix0rkmuks886
title: Fungal_up_down_transformer
desc: ''
updated: 1695836780604
created: 1694984769843
---
## Tokenizing Data Procedure Taken from ModelUsage.py

[ModelUsage.py GitHub](https://github.com/gagneurlab/SpeciesLM/blob/main/ModelUsage.ipynb)

```python
import math
import itertools
import collections
from collections.abc import Mapping
import numpy as np
import pandas as pd
import tqdm
import os
import torch
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from transformers import Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

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
seq_col = "three_prime_seq"  # name of the column in the df that stores the sequences

proxy_species = "candida_glabrata"  # species token to use
target_layer = (8,)  # what hidden layers to use for embedding

#
tok_func = lambda x: tok_func_species(x, proxy_species, seq_col)

ds = Dataset.from_pandas(dataset[[seq_col]])

tok_ds = ds.map(tok_func, batched=False, num_proc=2)

rem_tok_ds = tok_ds.remove_columns(seq_col)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
data_loader = torch.utils.data.DataLoader(
    rem_tok_ds, batch_size=1, collate_fn=data_collator, shuffle=False
)
```

## ModelUsage.py Padding for Upstream Models

Copy pasted

```md
Sequence length constraints

The 3' model expects an input sequence which is 300bp long (stop codon + 297). It will handle shorter sequences (although < 11 cannot be masked) and in theory can even predict up to 512 - but this is out-of-distribution and likely performs very poorly as the positional encodings are not adapted for this.

The 5' model expects an input which is 1003bp long (1000 + start codon). Longer sequences will not work, shorter sequences must be padded (e.g. with a fixed sequence), otherwise the start codon gets the wrong positional encoding which confuses the model.
```

## Upstream Model Padding GitHub Issue

In ModelUsage.py you have the following comment.

> Basic Information
>
> Sequence length constraints
>
> The 3' model expects an input sequence which is 300bp long (stop codon + 297). It will handle shorter sequences (although < 11 cannot be masked) and in theory can even predict up to 512 - but this is out-of-distribution and likely performs very poorly as the positional encodings are not adapted for this.
>
> The 5' model expects an input which is 1003bp long (1000 + start codon). Longer sequences will not work, shorter sequences must be padded (e.g. with a fixed sequence), otherwise the start codon gets the wrong positional encoding which confuses the model.

I am looking to use the model on some 5’ sequences that have < 1003 bp. When you say “"shorter sequences must be padded with (e.g. with a fixed sequence)” what do you mean exactly? Choose any of “A”, “T”, “C”, or “G”? Does this affect the model at all? How did you all train for sequences like this? I would like to follow whatever you did in your model development.

I’ve also considered padding with “N” which would give the token [UNK].

Thanks for your help.

## How input_ids_len Changes with Different Sequences

I believe that we get `input_ids_len==1001` when the `sequnece_length==1003`, because the first 3 bp is tokenized to one token (i.e. `ATG` is cast into one token). This would mean start codon goes to one token, so in terms of total tokens we get `1003 - 2 == 1001`.

```python
>>>model = FungalUpDownTransformer(model_name="upstream_species_lm", target_layer=(8,))
>>>sequence = "A" * (1000) + "ATG"
>>>model.embed([sequence], mean_embedding=True)
# breakpoint at input_ids_len = tokenized_data["input_ids"].shape[-1]
>>> input_ids_len = 1001
```

```python
>>>model = FungalUpDownTransformer(model_name="upstream_species_lm", target_layer=(8,))
>>>sequence = "A" * (1000 - 3) + "ATG"
>>>model.embed([sequence], mean_embedding=True)
# breakpoint at input_ids_len = tokenized_data["input_ids"].shape[-1]
>>> input_ids_len = 998
```
