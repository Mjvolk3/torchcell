---
id: 8xq4nxh3mki8vcznor43jxb
title: '145537'
desc: ''
updated: 1710791739206
created: 1710791739206
---

```python
dataset
Neo4jCellDataset(289044)
dataset.env
dataset[0]
>>> HeteroData(
  gene={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    cell_graph_idx_pert=[2],
    x=[6577, 1536],
    x_pert=[2, 1536],
    graph_level='global',
    label='dmf',
    label_error='dmf_std',
    label_value=0.9696,
    label_value_std=0.0445,
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 138602],
    num_edges=138602,
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 9496],
    num_edges=9496,
  }
)
dataset.env
>>> <Environment object at 0x2eb3e93b0>
```

If we access with get that sets the env.

***

```python
len(set([pert.systematic_gene_name for pert in data['experiment'].genotype.perturbations]).intersection(set(cell_graph['gene'].node_ids))) !=  len([pert.systematic_gene_name for pert in data['experiment'].genotype.perturbations])
```

From breakpoint inspection.

'YIR043C' or 'YBR289W' not in `genome.gene_set`

***

```md
I put this condition break where it says breakpoint to find the data in question. len(set([pert.systematic_gene_name for pert in data['experiment'].genotype.perturbations]).intersection(set(cell_graph['gene'].node_ids))) !=  len([pert.systematic_gene_name for pert in data['experiment'].genotype.perturbations]) def process_graph(cell_graph: HeteroData, data: dict[str, Any]) -> HeteroData:
    processed_graph = HeteroData() #breakpoint here

    # Nodes to remove based on the perturbations
    nodes_to_remove = {
        pert.systematic_gene_name for pert in data["experiment"].genotype.perturbations
    }

    # Assuming all nodes are of type 'gene', and copying node information to processed_graph
    processed_graph["gene"].node_ids = [
        nid for nid in cell_graph["gene"].node_ids if nid not in nodes_to_remove
    ]
    processed_graph["gene"].num_nodes = len(processed_graph["gene"].node_ids)
    # Additional information regarding perturbations
    processed_graph["gene"].ids_pert = list(nodes_to_remove)
    processed_graph["gene"].cell_graph_idx_pert = torch.tensor(
        [cell_graph["gene"].node_ids.index(nid) for nid in nodes_to_remove],
        dtype=torch.long,
    )

    # Populate x and x_pert attributes
    node_mapping = {nid: i for i, nid in enumerate(cell_graph["gene"].node_ids)}
    x = cell_graph["gene"].x
    processed_graph["gene"].x = x[
        torch.tensor([node_mapping[nid] for nid in processed_graph["gene"].node_ids])
    ]
    processed_graph["gene"].x_pert = x[processed_graph["gene"].cell_graph_idx_pert]

    # Add fitness phenotype data
    phenotype = data["experiment"].phenotype
    processed_graph["gene"].graph_level = phenotype.graph_level
    processed_graph["gene"].label = phenotype.label
    processed_graph["gene"].label_error = phenotype.label_error
    # TODO we actually want to do this renaming in the datamodel
    # We do it here to replicate behavior for downstream
    # Will break with anything other than fitness obviously
    processed_graph["gene"].label_value = phenotype.fitness
    processed_graph["gene"].label_value_std = phenotype.fitness_std

    # Mapping of node IDs to their new indices after filtering
    new_index_map = {nid: i for i, nid in enumerate(processed_graph["gene"].node_ids)}

    # Processing edges
    for edge_type in cell_graph.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = cell_graph[src_type, _, dst_type].edge_index.numpy()
        filtered_edges = []

        for src, dst in edge_index.T:
            src_id = cell_graph[src_type].node_ids[src]
            dst_id = cell_graph[dst_type].node_ids[dst]

            if src_id not in nodes_to_remove and dst_id not in nodes_to_remove:
                new_src = new_index_map[src_id]
                new_dst = new_index_map[dst_id]
                filtered_edges.append([new_src, new_dst])

        if filtered_edges:
            new_edge_index = torch.tensor(filtered_edges, dtype=torch.long).t()
            processed_graph[src_type, _, dst_type].edge_index = new_edge_index
            processed_graph[src_type, _, dst_type].num_edges = new_edge_index.shape[1]
        else:
            processed_graph[src_type, _, dst_type].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            processed_graph[src_type, _, dst_type].num_edges = 0

    return processed_graph data['experiment'].model_dump()
{'genotype': {'perturbations': [...]}, 'environment': {'media': {...}, 'temperature': {...}}, 'phenotype': {'graph_level': 'global', 'label': 'smf', 'label_error': 'smf_std', 'fitness': 0.2595, 'fitness_std': 0.0033}}
special variables
function variables
'genotype':
{'perturbations': [{...}, {...}]}
special variables
function variables
'perturbations':
[{'systematic_gene_name': 'YBR289W', 'perturbed_gene_name': 'snf5', 'description': 'Deletion via KanMX or NatMX gene replacement', 'perturbation_type': 'deletion', 'deletion_description': 'Deletion via NatMX gene replacement.', 'deletion_type': 'NatMX', 'nat_mx_description': 'NatMX Deletion Perturbation information specific to SGA experiments.', 'strain_id': 'YBR289W_sn1762', 'natmx_deletion_type': 'SGA'}, {'systematic_gene_name': 'YIR043C', 'perturbed_gene_name': 'yir043c', 'description': 'Deletion via KanMX or NatMX gene replacement', 'perturbation_type': 'deletion', 'deletion_description': 'Deletion via KanMX gene replacement.', 'deletion_type': 'KanMX', 'kan_mx_description': 'KanMX Deletion Perturbation information specific to SGA experiments.', 'strain_id': 'YIR043C_dma2441', 'kanmx_deletion_type': 'SGA'}]
special variables
function variables
0:
{'systematic_gene_name': 'YBR289W', 'perturbed_gene_name': 'snf5', 'description': 'Deletion via KanMX or NatMX gene replacement', 'perturbation_type': 'deletion', 'deletion_description': 'Deletion via NatMX gene replacement.', 'deletion_type': 'NatMX', 'nat_mx_description': 'NatMX Deletion Perturbation information specific to SGA experiments.', 'strain_id': 'YBR289W_sn1762', 'natmx_deletion_type': 'SGA'}
1:
{'systematic_gene_name': 'YIR043C', 'perturbed_gene_name': 'yir043c', 'description': 'Deletion via KanMX or NatMX gene replacement', 'perturbation_type': 'deletion', 'deletion_description': 'Deletion via KanMX gene replacement.', 'deletion_type': 'KanMX', 'kan_mx_description': 'KanMX Deletion Perturbation information specific to SGA experiments.', 'strain_id': 'YIR043C_dma2441', 'kanmx_deletion_type': 'SGA'}
len():
2
len():
1
'environment':
{'media': {'name': 'YEPD', 'state': 'solid'}, 'temperature': {'value': 30.0, 'unit': 'Celsius'}}
special variables
function variables
'media':
{'name': 'YEPD', 'state': 'solid'}
'temperature':
{'value': 30.0, 'unit': 'Celsius'}
len():
2
'phenotype':
{'graph_level': 'global', 'label': 'smf', 'label_error': 'smf_std', 'fitness': 0.2595, 'fitness_std': 0.0033}
special variables
function variables
'graph_level':
'global'
'label':
'smf'
'label_error':
'smf_std'
'fitness':
0.2595
'fitness_std':
0.0033
len():
5
len():
3 How is this data getting past the query?    query = """
        MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
        WITH e, g, COLLECT(p) AS perturbations
        WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
        WITH DISTINCT e, perturbations
        MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
        MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
        WHERE phen.graph_level = 'global'
        AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
        AND phen.fitness_std < 0.005
        MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
        MATCH (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'})
        MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
        WITH e, ref, perturbations
        WHERE ALL(p IN perturbations WHERE p.systematic_gene_name IN $gene_set)
        RETURN e, ref
    """
```

***

refined conditional breakpoint...

```python
len([pert.systematic_gene_name for pert in data['experiment'].genotype.perturbations if pert.systematic_gene_name not in cell_graph['gene'].node_ids]) > 0
```

***

problematic genes on my current query.

```python
raw_db.gene_set - raw_db.gene_set.intersection(genome.gene_set)
GeneSet(size=8, items=['YFL057C', 'YIL167W', 'YIR043C']...)
>>> set(raw_db.gene_set - raw_db.gene_set.intersection(genome.gene_set))
{'YFL057C', 'YIL167W', 'YOR031W', 'YOL153C', 'YKL200C', 'YKL158W', 'YIR043C', 'YLL017W'}
```

***

Works - but we need the other restraints

```cypher
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE 
    ALL(p IN perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
RETURN e, ref
```

***

Failed - There are `systematic_gene_name` in not in `gene_set`

```cypher
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf') AND phen.fitness_std < 0.005
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'})
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
RETURN e, ref
```

***

Failed - Returned no e, ref pairs

```cypher
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set)
WITH e, COLLECT(g) AS genotypes
WHERE ALL(g IN genotypes WHERE ALL(p IN g.perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set))
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf') AND phen.fitness_std < 0.005
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'})
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
RETURN e, ref
```

***

Failed - There are `systematic_gene_name` in not in `gene_set`

```bash
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell
/bin/python /Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo
4j_cell.py
data/go/go.obo: fmt(1.2) rel(2024-01-17) 45,869 Terms
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
Download finished.
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
Download finished.
2813it [00:09, 309.67it/s]
INFO:torchcell.data.neo4j_query_raw:Computing gene set...
2813it [00:00, 99533.29it/s]
Processing...
INFO:__main__:Processing raw data into LMDB
100%|██████████████████████████████████| 2813/2813 [00:00<00:00, 11148.73it/s]
Done!
  4%|█▌                                     | 44/1125 [00:36<14:50,  1.21it/s]
Traceback (most recent call last):
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py", line 495, in <module>
    main()
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py", line 488, in main
    for i in tqdm(data_module.train_dataloader()):
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/tqdm/std.py", line 1182, in __iter__
    for obj in iterable:
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 630, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1325, in _next_data
    return self._process_data(data)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1371, in _process_data
    data.reraise()
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/_utils.py", line 694, in reraise
    raise exception
ValueError: Caught ValueError in DataLoader worker process 4.
Original Traceback (most recent call last):
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    data = fetcher.fetch(index)
           ^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 49, in fetch
    data = self.dataset.__getitems__(possibly_batched_index)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/utils/data/dataset.py", line 364, in __getitems__
    return [self.dataset[self.indices[idx]] for idx in indices]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/utils/data/dataset.py", line 364, in <listcomp>
    return [self.dataset[self.indices[idx]] for idx in indices]
            ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/data/dataset.py", line 272, in __getitem__
    data = self.get(self.indices()[idx])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py", line 364, in get
    subsetted_graph = process_graph(self.cell_graph, data)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py", line 158, in process_graph
    [cell_graph["gene"].node_ids.index(nid) for nid in nodes_to_remove],
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py", line 158, in <listcomp>
    [cell_graph["gene"].node_ids.index(nid) for nid in nodes_to_remove],
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: 'YLL017W' is not in list
```

```cypher
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set)
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference),
    (e)<-[:PhenotypeMemberOf]-(phen:Phenotype),
    (e)<-[:EnvironmentMemberOf]-(env:Environment),
    (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'}),
    (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
WHERE phen.graph_level = 'global' 
    AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
    AND phen.fitness_std < 0.005
RETURN e, ref
```

***

I put this at where I have ##breakpoint len(set([pert.systematic_gene_name for pert in data['experiment'].genotype.perturbations]).intersection(set(cell_graph['gene'].node_ids))) !=  len([pert.systematic_gene_name for pert in data['experiment'].genotype.perturbations]) def process_graph(cell_graph: HeteroData, data: dict[str, Any]) -> HeteroData:
    processed_graph = HeteroData()  # breakpoint here

    # Nodes to remove based on the perturbations
    nodes_to_remove = {
        pert.systematic_gene_name for pert in data["experiment"].genotype.perturbations
    }

    # Assuming all nodes are of type 'gene', and copying node information to processed_graph
    processed_graph["gene"].node_ids = [
        nid for nid in cell_graph["gene"].node_ids if nid not in nodes_to_remove
    ]
    processed_graph["gene"].num_nodes = len(processed_graph["gene"].node_ids)
    # Additional information regarding perturbations
    processed_graph["gene"].ids_pert = list(nodes_to_remove)
    processed_graph["gene"].cell_graph_idx_pert = torch.tensor(
        [cell_graph["gene"].node_ids.index(nid) for nid in nodes_to_remove],
        dtype=torch.long,
    )

    # Populate x and x_pert attributes
    node_mapping = {nid: i for i, nid in enumerate(cell_graph["gene"].node_ids)}
    x = cell_graph["gene"].x
    processed_graph["gene"].x = x[
        torch.tensor([node_mapping[nid] for nid in processed_graph["gene"].node_ids])
    ]
    processed_graph["gene"].x_pert = x[processed_graph["gene"].cell_graph_idx_pert]

    # Add fitness phenotype data
    phenotype = data["experiment"].phenotype
    processed_graph["gene"].graph_level = phenotype.graph_level
    processed_graph["gene"].label = phenotype.label
    processed_graph["gene"].label_error = phenotype.label_error
    # TODO we actually want to do this renaming in the datamodel
    # We do it here to replicate behavior for downstream
    # Will break with anything other than fitness obviously
    processed_graph["gene"].label_value = phenotype.fitness
    processed_graph["gene"].label_value_std = phenotype.fitness_std

    # Mapping of node IDs to their new indices after filtering
    new_index_map = {nid: i for i, nid in enumerate(processed_graph["gene"].node_ids)}

    # Processing edges
    for edge_type in cell_graph.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = cell_graph[src_type, _, dst_type].edge_index.numpy()
        filtered_edges = []

        for src, dst in edge_index.T:
            src_id = cell_graph[src_type].node_ids[src]
            dst_id = cell_graph[dst_type].node_ids[dst]

            if src_id not in nodes_to_remove and dst_id not in nodes_to_remove:
                new_src = new_index_map[src_id]
                new_dst = new_index_map[dst_id]
                filtered_edges.append([new_src, new_dst])

        if filtered_edges:
            new_edge_index = torch.tensor(filtered_edges, dtype=torch.long).t()
            processed_graph[src_type, _, dst_type].edge_index = new_edge_index
            processed_graph[src_type, _, dst_type].num_edges = new_edge_index.shape[1]
        else:
            processed_graph[src_type, _, dst_type].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            processed_graph[src_type, _, dst_type].num_edges = 0

    return processed_graph It hits with this data... data
{'experiment': FitnessExperiment(genotype=Genotype(perturbations=[SgaKanMxDeletionPerturbation(syste...std', fitness=0.9735, fitness_std=0.0039)), 'reference': FitnessExperimentReference(reference_genome=ReferenceGenome(species='saccharomyces Ce...ss=1.0, fitness_std=0.040367310730687875))}
special variables
function variables
'experiment':
FitnessExperiment(genotype=Genotype(perturbations=[SgaKanMxDeletionPerturbation(systematic_gene_name='YGR023W', perturbed_gene_name='mtl1', description='Deletion via KanMX or NatMX gene replacement', perturbation_type='deletion', deletion_description='Deletion via KanMX gene replacement.', deletion_type='KanMX', kan_mx_description='KanMX Deletion Perturbation information specific to SGA experiments.', strain_id='YGR023W_dma1811', kanmx_deletion_type='SGA'), SgaNatMxDeletionPerturbation(systematic_gene_name='YIL167W', perturbed_gene_name='sdl1', description='Deletion via KanMX or NatMX gene replacement', perturbation_type='deletion', deletion_description='Deletion via NatMX gene replacement.', deletion_type='NatMX', nat_mx_description='NatMX Deletion Perturbation information specific to SGA experiments.', strain_id='YIL167W_sn4320', natmx_deletion_type='SGA')]), environment=BaseEnvironment(media=Media(name='YEPD', state='solid'), temperature=Temperature(value=30.0, unit='Celsius')), phenotype=FitnessPhenotype(...
'reference':
FitnessExperimentReference(reference_genome=ReferenceGenome(species='saccharomyces Cerevisiae', strain='s288c'), reference_environment=BaseEnvironment(media=Media(name='YEPD', state='solid'), temperature=Temperature(value=30.0, unit='Celsius')), reference_phenotype=FitnessPhenotype(graph_level='global', label='smf', label_error='smf_std', fitness=1.0, fitness_std=0.040367310730687875))
len():
2
data['experiment']
FitnessExperiment(genotype=Genotype(perturbations=[SgaKanMxDeletionPerturbation(systematic_gene_name='YGR023W', perturbed_gene_name='mtl1', description='Deletion via KanMX or NatMX gene replacement', perturbation_type='deletion', deletion_description='Deletion via KanMX gene replacement.', deletion_type='KanMX', kan_mx_description='KanMX Deletion Perturbation information specific to SGA experiments.', strain_id='YGR023W_dma1811', kanmx_deletion_type='SGA'), SgaNatMxDeletionPerturbation(systematic_gene_name='YIL167W', perturbed_gene_name='sdl1', description='Deletion via KanMX or NatMX gene replacement', perturbation_type='deletion', deletion_description='Deletion via NatMX gene replacement.', deletion_type='NatMX', nat_mx_description='NatMX Deletion Perturbation information specific to SGA experiments.', strain_id='YIL167W_sn4320', natmx_deletion_type='SGA')]), environment=BaseEnvironment(media=Media(name='YEPD', state='solid'), temperature=Temperature(value=30.0, unit='Celsius')), phenotype=FitnessPhenotype(graph_level='global', label='smf', label_error='smf_std', fitness=0.9735, fitness_std=0.0039))
data['experiment'].genotype.perturbations
[SgaKanMxDeletionPerturbation(systematic_gene_name='YGR023W', perturbed_gene_name='mtl...R023W_dma1811', kanmx_deletion_type='SGA'), SgaNatMxDeletionPerturbation(systematic_gene_name='YIL167W', perturbed_gene_name='sdl...IL167W_sn4320', natmx_deletion_type='SGA')]
[pert.systematic_gene_name for pert in data['experiment'].genotype.perturbations]
['YGR023W', 'YIL167W'] raw_db
Neo4jQueryRaw(uri=bolt://localhost:7687, root_dir=/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/neo4j, query=
        MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
        WITH e, g, COLLECT(p) AS perturbations
        WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set)
        WITH DISTINCT e
        MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference),
            (e)<-[:PhenotypeMemberOf]-(phen:Phenotype),
            (e)<-[:EnvironmentMemberOf]-(env:Environment),
            (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'}),
            (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
        WHERE phen.graph_level = 'global'
        AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
        AND phen.fitness_std < 0.005
        RETURN e, ref
    )
raw_db.gene_set
GeneSet(size=2086, items=['YAL004W', 'YAL008W', 'YAL013W']...)
"YGR023W" in raw_db.gene_set
True
"YGR023W" in gene_set
True
"YGR023W" in genome.gene_set
True
"YIL167W" in raw_db.gene_set
True
"YIL167W" in gene_set
False
"YIL167W" in genome.gene_set
False Somehow this experiment has perturbation "YIL167W" yet it is getting through in the query...

***

Fails - Returns records but some of the `systematic_gene_name` in perturbations are not in `gene_set`

```cypher
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set)
WITH e, g
WHERE ALL(p IN g.perturbations WHERE p.systematic_gene_name IN $gene_set)
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference),
    (e)<-[:PhenotypeMemberOf]-(phen:Phenotype),
    (e)<-[:EnvironmentMemberOf]-(env:Environment),
    (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'}),
    (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
WHERE phen.graph_level = 'global'
AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
AND phen.fitness_std < 0.005
RETURN e, ref
```

```bash
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michae
lvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py
data/go/go.obo: fmt(1.2) rel(2024-01-17) 45,869 Terms
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
Download finished.
Downloading downstream_species_lm model to /Users/michaelvolk/Documents/projects/torchcell/torchcell/models/pretrained_LLM/fungal_up_down_transformer/gagneurlab/SpeciesLM/downstream_species_lm...
Download finished.
0it [00:05, ?it/s]
INFO:torchcell.data.neo4j_query_raw:Computing gene set...
0it [00:00, ?it/s]
Traceback (most recent call last):
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py", line 497, in <module>
    main()
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py", line 474, in main
    dataset = Neo4jCellDataset(
              ^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py", line 242, in __init__
    self.raw_db = self.load_raw(uri, username, password, root, query, self.genome)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py", line 298, in load_raw
    raw_db = Neo4jQueryRaw(
             ^^^^^^^^^^^^^^
  File "<attrs generated init torchcell.data.neo4j_query_raw.Neo4jQueryRaw>", line 19, in __init__
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_query_raw.py", line 152, in __attrs_post_init__
    self.process()
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_query_raw.py", line 213, in process
    self.gene_set = self.compute_gene_set()
    ^^^^^^^^^^^^^
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_query_raw.py", line 371, in gene_set
    raise ValueError("Cannot set an empty or None value for gene_set")
ValueError: Cannot set an empty or None value for gene_set
```

***

Fails - Query selected a perturbation with a `systematic_gene_name` not in `gene_set`

```cypher
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set)
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference),
    (e)<-[:PhenotypeMemberOf]-(phen:Phenotype),
    (e)<-[:EnvironmentMemberOf]-(env:Environment),
    (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'}),
    (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
WHERE phen.graph_level = 'global' 
    AND (phen.label IN ['smf', 'dmf', 'tmf'])
    AND phen.fitness_std < 0.005
RETURN e, ref
```

`ValueError: 'YOR031W' is not in list`

***

Fails - This query returns no results.

```cypher
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set)
WITH e
WHERE ALL(g IN e.genotypes WHERE ALL(p IN g.perturbations WHERE p.systematic_gene_name IN $gene_set))
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference),
      (e)<-[:PhenotypeMemberOf]-(phen:Phenotype),
      (e)<-[:EnvironmentMemberOf]-(env:Environment),
      (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'}),
      (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
WHERE phen.graph_level = 'global'
  AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
  AND phen.fitness_std < 0.005
RETURN e, ref
```

***

Fails - This query returns no results.

```cypher
MATCH (e:Experiment)-[:GenotypeMemberOf]->(g:Genotype)-[:PerturbationMemberOf]->(p:Perturbation)
WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set
WITH e, g, COLLECT(p) AS perturbations
WITH e, COLLECT(g) AS genotypes, perturbations
WHERE ALL(g IN genotypes WHERE SIZE(perturbations) > 0)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference),
      (e)<-[:PhenotypeMemberOf]-(phen:Phenotype),
      (e)<-[:EnvironmentMemberOf]-(env:Environment),
      (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'}),
      (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
WHERE phen.graph_level = 'global' 
      AND (phen.label IN ['smf', 'dmf', 'tmf'])
      AND phen.fitness_std < 0.005
RETURN DISTINCT e, ref
```

***

Fails - Returns no records

```cypher
MATCH (e:Experiment)-[:GenotypeMemberOf]->(g:Genotype)-[:PerturbationMemberOf]->(p:Perturbation)
WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set
WITH e, COUNT(p) AS matchingPerturbations
MATCH (e)-[:GenotypeMemberOf]->(g)-[:PerturbationMemberOf]->(p2)
WITH e, matchingPerturbations, COUNT(p2) AS totalPerturbations
WHERE matchingPerturbations = totalPerturbations
WITH DISTINCT e
MATCH (e)-[:ExperimentReferenceOf]->(ref:ExperimentReference),
      (e)-[:PhenotypeMemberOf]->(phen:Phenotype),
      (e)-[:EnvironmentMemberOf]->(env:Environment),
      (env)-[:MediaMemberOf]->(m:Media {name: 'YEPD'}),
      (env)-[:TemperatureMemberOf]->(t:Temperature {value: 30})
WHERE phen.graph_level = 'global' 
      AND phen.label IN ['smf', 'dmf', 'tmf']
      AND phen.fitness_std < 0.005
RETURN e, ref

```

***

Fails - Returns no records

```cypher
MATCH (e:Experiment)-[:GenotypeMemberOf]->(g:Genotype)-[:PerturbationMemberOf]->(p:Perturbation)
WHERE p.perturbation_type = 'deletion'
WITH e, g, COLLECT(p.systematic_gene_name) AS perturbationNames
WHERE ALL(name IN perturbationNames WHERE name IN $gene_set)
WITH DISTINCT e
MATCH (e)-[:ExperimentReferenceOf]->(ref:ExperimentReference),
      (e)-[:PhenotypeMemberOf]->(phen:Phenotype),
      (e)-[:EnvironmentMemberOf]->(env:Environment),
      (env)-[:MediaMemberOf]->(m:Media {name: 'YEPD'}),
      (env)-[:TemperatureMemberOf]->(t:Temperature {value: 30})
WHERE phen.graph_level = 'global' 
      AND phen.label IN ['smf', 'dmf', 'tmf']
      AND phen.fitness_std < 0.005
RETURN e, ref
```

***

Fails - Returns no records

```cypher
MATCH (e:Experiment)-[:GenotypeMemberOf]->(g:Genotype)-[:PerturbationMemberOf]->(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set)
WITH e, COLLECT(g) AS genotypes
MATCH (e)-[:ExperimentReferenceOf]->(ref:ExperimentReference),
      (e)-[:PhenotypeMemberOf]->(phen:Phenotype),
      (e)-[:EnvironmentMemberOf]->(env:Environment),
      (env)-[:MediaMemberOf]->(m:Media {name: 'YEPD'}),
      (env)-[:TemperatureMemberOf]->(t:Temperature {value: 30})
WHERE phen.graph_level = 'global' 
      AND (phen.label IN ['smf', 'dmf', 'tmf'])
      AND phen.fitness_std < 0.005
RETURN DISTINCT e, ref

```
