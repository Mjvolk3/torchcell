---
id: a6fpajtfy0v7071yr1a2032
title: '103642'
desc: ''
updated: 1728315430170
created: 1728315404550
---
One of the builds with `conversion`, `deduplication`, and `aggregation`.

```bash
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torch
cell/torchcell/data/neo4j_cell.py
data/go/go.obo: fmt(1.2) rel(2024-01-17) 45,869 Terms
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
Download finished.
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
Download finished.
================
raw root_dir: /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/003-fit-int/test_dataset
================
INFO:torchcell.data.neo4j_query_raw:Processing data...
0it [00:00, ?it/s]INFO:torchcell.data.neo4j_query_raw:Connecting to Neo4j and executing query...
INFO:torchcell.data.neo4j_query_raw:Running query...
INFO:torchcell.data.neo4j_query_raw:Query executed, about to process results...
2074501it [09:38, 4206.88it/s]INFO:torchcell.data.neo4j_query_raw:All records processed.
2074718it [09:44, 3552.31it/s]
INFO:torchcell.data.neo4j_query_raw:Total records processed: 2074718
INFO:torchcell.data.neo4j_query_raw:Computing experiment reference index...
INFO:torchcell.data.neo4j_query_raw:Computing experiment reference index sequentially
INFO:torchcell.data.neo4j_query_raw:Computing gene set...
2074718it [00:30, 67793.85it/s] 

Processing...
Converting and writing to LMDB: 2074718it [04:10, 8266.84it/s]
INFO:torchcell.datamodels.conversion:Conversion complete. LMDB database written to /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/003-fit-int/test_dataset/conversion/lmdb
INFO:torchcell.datamodels.conversion:Number of instances converted: 1140
INFO:torchcell.datamodels.conversion:Total number of instances processed: 2074718
Deduplicating and writing to LMDB:  14%|██████▉                                          | 276439/1944355 [00:09<01:02, 26480.58it/s]/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/mean_experiment_deduplicate.py:234: RuntimeWarning: divide by zero encountered in scalar divide
  t_stat = mean_x / sem
Deduplicating and writing to LMDB: 100%|████████████████████████████████████████████████| 1944355/1944355 [00:57<00:00, 33589.23it/s]
INFO:torchcell.data.deduplicate:Deduplication complete. LMDB database written to /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/003-fit-int/test_dataset/deduplication/lmdb
INFO:torchcell.data.deduplicate:Number of instances deduplicated: 130363
INFO:torchcell.data.deduplicate:Total number of instances after deduplication: 1944355
Aggregating data: 1944355it [04:18, 7524.65it/s] 
Aggregation complete. LMDB database written to /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/003-fit-int/test_dataset/aggregation/lmdb
Total number of aggregated groups: 1124583
Total number of experiments after aggregation: 1944355
Done!
Computing phenotype label index...
Computing dataset name index...
Computing perturbation count index...
1124583
INFO:torchcell.datamodules.cell:Generating indices for train, val, and test sets...
  6%|█████▏                                                                              | 34413/562292 [3:28:47<55:39:30,  2.63it/s]
```

Cleaned for presentation

```bash
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torch
cell/torchcell/data/neo4j_cell.py
================
raw root_dir: /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/003-fit-int/test_dataset
================
INFO:torchcell.data.neo4j_query_raw:Processing data...
0it [00:00, ?it/s]INFO:torchcell.data.neo4j_query_raw:Connecting to Neo4j and executing query...
INFO:torchcell.data.neo4j_query_raw:Running query...
INFO:torchcell.data.neo4j_query_raw:Query executed, about to process results...
2074501it [09:38, 4206.88it/s]INFO:torchcell.data.neo4j_query_raw:All records processed.
2074718it [09:44, 3552.31it/s]
INFO:torchcell.data.neo4j_query_raw:Total records processed: 2074718
INFO:torchcell.data.neo4j_query_raw:Computing experiment reference index...
INFO:torchcell.data.neo4j_query_raw:Computing experiment reference index sequentially
INFO:torchcell.data.neo4j_query_raw:Computing gene set...
2074718it [00:30, 67793.85it/s] 
Processing...
Converting and writing to LMDB: 2074718it [04:10, 8266.84it/s]
INFO:torchcell.datamodels.conversion:Conversion complete. LMDB database written to /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/003-fit-int/test_dataset/conversion/lmdb
INFO:torchcell.datamodels.conversion:Number of instances converted: 1140
INFO:torchcell.datamodels.conversion:Total number of instances processed: 2074718
Deduplicating and writing to LMDB: 100%|████████████████████████████████████████████████| 1944355/1944355 [00:57<00:00, 33589.23it/s]
INFO:torchcell.data.deduplicate:Deduplication complete. LMDB database written to /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/003-fit-int/test_dataset/deduplication/lmdb
INFO:torchcell.data.deduplicate:Number of instances deduplicated: 130363
INFO:torchcell.data.deduplicate:Total number of instances after deduplication: 1944355
Aggregating data: 1944355it [04:18, 7524.65it/s] 
Aggregation complete. LMDB database written to /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/003-fit-int/test_dataset/aggregation/lmdb
Total number of aggregated groups: 1124583
Done!
Computing phenotype label index...
Computing dataset name index...
Computing perturbation count index...
1124583
INFO:torchcell.datamodules.cell:Generating indices for train, val, and test sets...
  6%|█████▏                                                                              | 34413/562292 [3:28:47<55:39:30,  2.63it/s]
```
