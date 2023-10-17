---
id: vtnlvfypudrp5102nrrmgk3
title: protT5
desc: ''
updated: 1697576855434
created: 1697574717834
---
## Embed all Proteins Can Like 0 Non Expressed Protein

Instead of computing multiple different embedding datasets where the there are 0's for `'dubious'` or `'uncharacterized'` proteins and embedding vectors for `'verified'` proteins. We compute all embeddings, and leave the down selection to pipeline design. This allows us to only compute one set of embeddings.

## Protein_Data vs Dna_Windows

We have to use `dna_windows` for now. This is a bit of a hack.

```python
>>>prot_T5_dataset[0]
Data(
  id='Q0010',
  protein_data={ prot_t5_xl_uniref50='MYYIMFLYNMLLIIILIFYSIVGVPIIIFNNNYYWDPDIFLFIIYYFIKFIIIFNLYLYYMINYIVYTPSGSPPGRGTYILLYNMLYSYNMFIDYVMKFITCVTYMYLMFWLLSPTPSPYYVSEVPVS*' },
  embeddings={ prot_t5_xl_uniref50=[1, 1024] }
)
fud_downstream[0]
Data(
  id='Q0010',
  dna_windows={ species_downstream=id='Q0010' chromosome=0 strand='+' start=3952 end=4338 seq='TAAGAAACCGGGACTTATATATTTATAAATATAAATCTAACTTAATTAATAATTTAAATAATATACTTTATATTTTATAAATAAAAATAATTATAACCTTTTTTATAATTATATATAATAATAATATATATTATCAAATAATTATTATTTCTTTTTTTTCTTTAATTAATTAATTAATTAATATTTTATAAAAATATATTTCTCCTTACGGGGTTCCGGCTCCCGTAGCCGGGGCCCGAAACTAAATAAAATATATTATTAATAATATTATATAATATAATAATAATATAATAATTTTAT' start_window=4335 end_window=4635 },
  embeddings={ species_downstream=[1, 768] }
)
```

Two datasets we may want to combine.

```bash
Traceback (most recent call last):
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/data/storage.py", line 79, in __getattr__
    return self[key]
           ~~~~^^^^^
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/data/storage.py", line 104, in __getitem__
    return self._mapping[key]
           ~~~~~~~~~~~~~^^^^^
KeyError: 'dna_windows'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/michaelvolk/Documents/projects/torchcell/src/torchcell/datasets/embedding.py", line 119, in __add__
    for key in data_item.dna_windows:
               ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/data/data.py", line 441, in __getattr__
    return getattr(self._store, key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/data/storage.py", line 81, in __getattr__
    raise AttributeError(
AttributeError: 'GlobalStorage' object has no attribute 'dna_windows'
```

We get this error because I was bit myopic 👀 at first thinking there weren't any models that would allow us to encode the entire genome. Since this has changed we should be more general in the way we include the meta data of the embeddings. Instead of `dna_windows` it should probably be something like `embedding_meta` and this could be a series of different objects. Still unclear to me the best thing to do here, and it is not absolutely necessary right now.
