---
id: 6zvfbx191tqmjv2yeuxxfns
title: Issue
desc: ''
updated: 1745814201194
created: 1745814194506
---
## Query Isn't Returning All Data

Only Returns `57,883` recordss.

```cypher
// TmiKuzmin2018Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmiKuzmin2018Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
WHERE phen.graph_level = 'hyperedge'
 AND m.name = 'YEPD'
 AND t.value = 30
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.perturbation_type = 'deletion'
 AND p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
// LIMIT 100
RETURN e, ref
```

```bash
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/query.py    23:04
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
/Users/michaelvolk/Documents/projects/torchcell/torchcell/sequence/data.py:43: PydanticDeprecatedSince20: Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  @root_validator(pre=True)
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/goatools/__init__.py:2: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  from pkg_resources import get_distribution, DistributionNotFound
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
data/go/go.obo: fmt(1.2) rel(2024-11-03) 43,983 Terms
length of gene_set: 6579
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
BertForMaskedLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
Download finished.
/Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/fungal_up_down_transformer.py:54: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.data, self.slices = torch.load(self.processed_paths[0])
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
Download finished.
Creating dataset with metabolism network...
INFO:cobra.core.model:The current solver interface glpk doesn't support setting the optimality tolerance.
Processing...
================
raw root_dir: /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build
================
INFO:torchcell.data.neo4j_query_raw:Processing data...
0it [00:00, ?it/s]INFO:torchcell.data.neo4j_query_raw:Connecting to Neo4j and executing query...
INFO:torchcell.data.neo4j_query_raw:Running query...
INFO:torchcell.data.neo4j_query_raw:Query executed, about to process results...
57564it [00:17, 4375.81it/s]INFO:torchcell.data.neo4j_query_raw:All records processed.
57883it [00:17, 3320.90it/s]
INFO:torchcell.data.neo4j_query_raw:Total records processed: 57883
INFO:torchcell.data.neo4j_query_raw:Computing experiment reference index...
```

## Unrestrained Query Returns all Data

Returns all `91,111` records.

```cypher
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmiKuzmin2018Dataset'
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
RETURN e, ref
```

```bash
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/query.py    23:08
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
/Users/michaelvolk/Documents/projects/torchcell/torchcell/sequence/data.py:43: PydanticDeprecatedSince20: Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  @root_validator(pre=True)
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/goatools/__init__.py:2: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  from pkg_resources import get_distribution, DistributionNotFound
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
data/go/go.obo: fmt(1.2) rel(2024-11-03) 43,983 Terms
length of gene_set: 6579
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
BertForMaskedLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
Download finished.
/Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/fungal_up_down_transformer.py:54: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.data, self.slices = torch.load(self.processed_paths[0])
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
Download finished.
Creating dataset with metabolism network...
INFO:cobra.core.model:The current solver interface glpk doesn't support setting the optimality tolerance.
Processing...
================
raw root_dir: /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build
================
INFO:torchcell.data.neo4j_query_raw:Processing data...
0it [00:00, ?it/s]INFO:torchcell.data.neo4j_query_raw:Connecting to Neo4j and executing query...
INFO:torchcell.data.neo4j_query_raw:Running query...
INFO:torchcell.data.neo4j_query_raw:Query executed, about to process results...
90892it [00:21, 4277.40it/s]INFO:torchcell.data.neo4j_query_raw:All records processed.
91111it [00:21, 4279.05it/s]
INFO:torchcell.data.neo4j_query_raw:Total records processed: 91111
INFO:torchcell.data.neo4j_query_raw:Computing experiment reference index...
```

## Constrained to only Gene Set Gives Redundancy

This is returning 273,333 which is more data than exists in that dataset...

```cypher
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmiKuzmin2018Dataset'
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
WHERE ANY(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
    WHERE p.systematic_gene_name IN $gene_set)
RETURN e, ref
```

```bash
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/query.py    23:10
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
/Users/michaelvolk/Documents/projects/torchcell/torchcell/sequence/data.py:43: PydanticDeprecatedSince20: Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  @root_validator(pre=True)
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/goatools/__init__.py:2: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  from pkg_resources import get_distribution, DistributionNotFound
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
data/go/go.obo: fmt(1.2) rel(2024-11-03) 43,983 Terms
length of gene_set: 6579
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
BertForMaskedLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
Download finished.
/Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/fungal_up_down_transformer.py:54: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.data, self.slices = torch.load(self.processed_paths[0])
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
Download finished.
Creating dataset with metabolism network...
INFO:cobra.core.model:The current solver interface glpk doesn't support setting the optimality tolerance.
Processing...
================
raw root_dir: /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build
================
INFO:torchcell.data.neo4j_query_raw:Processing data...
0it [00:00, ?it/s]INFO:torchcell.data.neo4j_query_raw:Connecting to Neo4j and executing query...
INFO:torchcell.data.neo4j_query_raw:Running query...
INFO:torchcell.data.neo4j_query_raw:Query executed, about to process results...
273321it [01:06, 4084.19it/s]INFO:torchcell.data.neo4j_query_raw:All records processed.
273333it [01:06, 4139.69it/s]
INFO:torchcell.data.neo4j_query_raw:Total records processed: 273333
INFO:torchcell.data.neo4j_query_raw:Computing experiment reference index...
```

```python
if __name__ == "__main__":
    # Fitness
    print("Fitness")
    dataset = SmfKuzmin2018Dataset()
    print(dataset[0])
    print(len(dataset))
    dataset = DmfKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
    dataset = TmfKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
    print()
    print("Interactions")
    # Interactions
    dataset = DmiKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
    dataset = TmiKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
```

```bash
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/kuzmin2018.py       23:13
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
/Users/michaelvolk/Documents/projects/torchcell/torchcell/sequence/data.py:43: PydanticDeprecatedSince20: Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  @root_validator(pre=True)
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/goatools/__init__.py:2: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  from pkg_resources import get_distribution, DistributionNotFound
Fitness
{'experiment': {'experiment_type': 'fitness', 'dataset_name': 'SmfKuzmin2018Dataset', 'genotype': {'perturbations': [{'systematic_gene_name': 'YAL048C', 'perturbed_gene_name': 'gem1_delta', 'description': 'Deletion via KanMX or NatMX gene replacement', 'perturbation_type': 'deletion', 'deletion_description': 'Deletion via KanMX gene replacement.', 'deletion_type': 'KanMX', 'kan_mx_description': 'KanMX Deletion Perturbation information specific to SGA experiments.', 'strain_id': 'YAL048C_dma5203', 'kanmx_deletion_type': 'SGA'}]}, 'environment': {'media': {'name': 'YEPD', 'state': 'solid'}, 'temperature': {'value': 30.0, 'unit': 'Celsius'}}, 'phenotype': {'graph_level': 'global', 'label_name': 'fitness', 'label_statistic_name': 'fitness_std', 'fitness': 0.8595, 'fitness_std': None}}, 'reference': {'experiment_reference_type': 'fitness', 'dataset_name': 'SmfKuzmin2018Dataset', 'genome_reference': {'species': 'Saccharomyces cerevisiae', 'strain': 'S288C'}, 'environment_reference': {'media': {'name': 'YEPD', 'state': 'solid'}, 'temperature': {'value': 30.0, 'unit': 'Celsius'}}, 'phenotype_reference': {'graph_level': 'global', 'label_name': 'fitness', 'label_statistic_name': 'fitness_std', 'fitness': 1.0, 'fitness_std': 0.06314361508245099}}, 'publication': {'pubmed_id': '29674565', 'pubmed_url': 'https://pubmed.ncbi.nlm.nih.gov/29674565/', 'doi': '10.1126/science.aao1729', 'doi_url': 'https://www.science.org/doi/10.1126/science.aao1729'}}
1539
410399
91111

Interactions
410399
91111 # FLAG
```

## More Constrained Similar to Original

Properly returns 91,111

```cypher
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmiKuzmin2018Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
WHERE phen.graph_level = 'hyperedge'
 AND m.name = 'YEPD'
 AND t.value = 30
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
 AND EXISTS {
   MATCH (g)<-[:PerturbationMemberOf]-(some_p)
   WHERE some_p.systematic_gene_name IN $gene_set
 }
WITH DISTINCT e, ref
ORDER BY e.id
RETURN e, ref
```

```bash
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/query.py    23:15
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
/Users/michaelvolk/Documents/projects/torchcell/torchcell/sequence/data.py:43: PydanticDeprecatedSince20: Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  @root_validator(pre=True)
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/goatools/__init__.py:2: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  from pkg_resources import get_distribution, DistributionNotFound
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
data/go/go.obo: fmt(1.2) rel(2024-11-03) 43,983 Terms
length of gene_set: 6579
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
BertForMaskedLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
Download finished.
/Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/fungal_up_down_transformer.py:54: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.data, self.slices = torch.load(self.processed_paths[0])
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
Download finished.
Creating dataset with metabolism network...
INFO:cobra.core.model:The current solver interface glpk doesn't support setting the optimality tolerance.
Processing...
================
raw root_dir: /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build
================
INFO:torchcell.data.neo4j_query_raw:Processing data...
0it [00:00, ?it/s]INFO:torchcell.data.neo4j_query_raw:Connecting to Neo4j and executing query...
INFO:torchcell.data.neo4j_query_raw:Running query...
INFO:torchcell.data.neo4j_query_raw:Query executed, about to process results...
90745it [00:28, 4406.76it/s]INFO:torchcell.data.neo4j_query_raw:All records processed.
91111it [00:28, 3224.82it/s]
INFO:torchcell.data.neo4j_query_raw:Total records processed: 91111
INFO:torchcell.data.neo4j_query_raw:Computing experiment reference index...
```

## Summary Of Issue Related To Gene Perturbation Type

Summary from Claude

Original query used:

```cypher
AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.perturbation_type = 'deletion'
AND p.systematic_gene_name IN $gene_set)
```

The working query uses:

```cypher
AND EXISTS {
  MATCH (g)<-[:PerturbationMemberOf]-(some_p)
  WHERE some_p.systematic_gene_name IN $gene_set
}
```

Here are the critical differences:

1. **ALL vs EXISTS**:
   - The `ALL` predicate requires that **every** perturbation in a genotype must satisfy both conditions: be of type 'deletion' AND have a systematic_gene_name in your gene_set
   - The `EXISTS` pattern only requires that **at least one** perturbation has its systematic_gene_name in your gene_set

2. **Removal of the perturbation_type restriction**:
   - The original query required `p.perturbation_type = 'deletion'`
   - Looking at your data sample, we can see some perturbations have `perturbation_type: "allele"`, which would be excluded by the original query

3. **Query pattern vs. collection predicate**:
   - The new query uses a more efficient pattern that avoids potential Cartesian product issues that can occur with list comprehensions

The most important issue was that the original query used `ALL`, which is too restrictive. An experiment would only be included if **every single** perturbation in that experiment was both:

1. Of type 'deletion' (excluding experiments with allele perturbations)
2. Had a gene name in your gene_set (excluding experiments with any gene not in your set)

The new query with `EXISTS` simply checks that there is at least one perturbation with a gene in your gene_set, which matches your actual requirement.

This explains why:

- Original query: 57,883 records (too restrictive)
- Simple query with no filters: 91,111 records (the correct total)
- New query with `EXISTS`: 91,111 records (correctly returning all records)
