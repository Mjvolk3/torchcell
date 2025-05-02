---
id: fibg7evna9eeka9ijez1msk
title: 221621-Add-Phenotype
desc: ''
updated: 1746157163657
created: 1746155785563
---

This is what data object looks like. We want a more general way of representing the phenotype info.

```python
dataset[0]
HeteroData(
  gene={
    node_ids=[6604],
    num_nodes=6604,
    ids_pert=[3],
    cell_graph_idx_pert=[3],
    x=[6607, 0],
    x_pert=[3, 0],
    gene_interaction=[1],
    gene_interaction_p_value=[1],
    pert_mask=[6607],
    gene_interaction_original=[1],
    mask=[6607],
  },
  (gene, string9_1_coexpression, gene)={
    edge_index=[2, 320173],
    num_edges=320173,
    pert_mask=[320295],
    adj_mask=[6607, 6607],
  },
  (gene, string9_1_cooccurence, gene)={
    edge_index=[2, 9263],
    num_edges=9263,
    pert_mask=[9266],
    adj_mask=[6607, 6607],
  },
  (gene, string9_1_database, gene)={
    edge_index=[2, 40083],
    num_edges=40083,
    pert_mask=[40093],
    adj_mask=[6607, 6607],
  },
  (gene, string9_1_experimental, gene)={
    edge_index=[2, 225570],
    num_edges=225570,
    pert_mask=[226057],
    adj_mask=[6607, 6607],
  },
  (gene, string9_1_fusion, gene)={
    edge_index=[2, 7962],
    num_edges=7962,
    pert_mask=[7965],
    adj_mask=[6607, 6607],
  },
  (gene, string9_1_neighborhood, gene)={
    edge_index=[2, 52205],
    num_edges=52205,
    pert_mask=[52208],
    adj_mask=[6607, 6607],
  }
)
```

```python
def _add_phenotype_data(
    self,
    integrated_subgraph: HeteroData,
    phenotype_info: list[PhenotypeType],
    data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
) -> None:
    phenotype_fields = []
    for phenotype in phenotype_info:
        phenotype_fields.extend(
            [
                phenotype.model_fields["label_name"].default,
                phenotype.model_fields["label_statistic_name"].default,
            ]
        )
    for field in phenotype_fields:
        field_values = []
        for item in data:
            value = getattr(item["experiment"].phenotype, field, None)
            if value is not None:
                field_values.append(value)
        integrated_subgraph["gene"][field] = torch.tensor(
            field_values if field_values else [float("nan")],
            dtype=torch.float,
            device=self.device,
        )
```

From the schema this is what a phenotype looks like.

```python
class Phenotype(ModelStrict):
    graph_level: str = Field(
        description="most natural level of graph at which phenotype is observed"
    )
    label_name: str = Field(description="name of label")
    label_statistic_name: Optional[str] = Field(
        default=None,
        description="name of error or confidence statistic related to label",
    )

    @model_validator(mode="after")
    def validate_fields(self):
        valid_graph_levels = {
            "edge",
            "node",
            "hyperedge",
            "subgraph",
            "global",
            "metabolism",
            "gene ontology",
        }
        if self.graph_level not in valid_graph_levels:
            raise ValueError(
                f"graph_level must be one of: {', '.join(valid_graph_levels)}"
            )
        return self

    def __getitem__(self, key):
        return getattr(self, key)

```

With a breakpoint at the beginning of `_add_phenotype_data`.

```python
phenotype_info
[<class 'torchcell.datamodels.schema.GeneInteractionPhenotype'>]
integrated_subgraph
HeteroData(
  gene={
    node_ids=[6604],
    num_nodes=6604,
    ids_pert=[3],
    cell_graph_idx_pert=[3],
    x=[6604, 0],
    x_pert=[3, 0],
  },
  (gene, string9_1_coexpression, gene)={
    edge_index=[2, 319985],
    num_edges=319985,
    pert_mask=[320295],
  },
  (gene, string9_1_cooccurence, gene)={
    edge_index=[2, 9263],
    num_edges=9263,
    pert_mask=[9266],
  },
  (gene, string9_1_database, gene)={
    edge_index=[2, 40083],
    num_edges=40083,
    pert_mask=[40093],
  },
  (gene, string9_1_experimental, gene)={
    edge_index=[2, 225864],
    num_edges=225864,
    pert_mask=[226057],
  },
  (gene, string9_1_fusion, gene)={
    edge_index=[2, 7962],
    num_edges=7962,
    pert_mask=[7965],
  },
  (gene, string9_1_neighborhood, gene)={
    edge_index=[2, 52032],
    num_edges=52032,
    pert_mask=[52208],
  }
)
data
[{'experiment': GeneInteractionExperiment(experiment_type='gene interaction', dataset_name='TmiKuzmin....028108, gene_interaction_p_value=0.1702)), 'experiment_reference': GeneInteractionExperimentReference(experiment_reference_type='gene interaction', data...ction=0.0, gene_interaction_p_value=None))}]
```

We are adapting to a new setting where some of the labels are vector... Also we many more labels. For efficient packing we have to fill missing data with nan right now. Repeatedly filling vectors size > 100 with nan will be waste. Instead I would like to save in a coo format. `labels` can be concatenated scalars to vectors to form one long vector. We can do same with `label_statistic`. I was thinking for different names we can have mapping to ints the have a vector of ints same size as phenotype. This will make selective prediction more easy. I was also thinking we could do a sort of within batch ptr to point to where each phenotype is but that seems more clumsy. If we have phenotypes `labels`, `label_statistics` with a same sized vector for indexing into them and then a list of phenotype str for identifying which int matches to which sample. Or we can make it a dict but I don't know if dicts can be packaged across batches like `list[str]`.


within data instance ptr idea.. I think it's a bad idea.

```python
batch
HeteroDataBatch(
  gene={
    node_ids=[2],
    num_nodes=13208,
    ids_pert=[2],
    cell_graph_idx_pert=[6],
    x=[13214, 0],
    x_batch=[13214],
    x_ptr=[3],
    x_pert=[6, 0],
    x_pert_batch=[6],
    x_pert_ptr=[3],
    gene_interaction=[2],
    gene_interaction_p_value=[2],
    pert_mask=[13214],
    gene_interaction_original=[2],
    mask=[13214],
    batch=[13208],
    ptr=[3],
  },
  (gene, string9_1_coexpression, gene)={
    edge_index=[2, 640508],
    num_edges=[2],
    pert_mask=[640590],
    adj_mask=[13214, 6607],
  },
  (gene, string9_1_cooccurence, gene)={
    edge_index=[2, 18526],
    num_edges=[2],
    pert_mask=[18532],
    adj_mask=[13214, 6607],
  },
  (gene, string9_1_database, gene)={
    edge_index=[2, 80113],
    num_edges=[2],
    pert_mask=[80186],
    adj_mask=[13214, 6607],
  },
  (gene, string9_1_experimental, gene)={
    edge_index=[2, 451368],
    num_edges=[2],
    pert_mask=[452114],
    adj_mask=[13214, 6607],
  },
  (gene, string9_1_fusion, gene)={
    edge_index=[2, 15924],
    num_edges=[2],
    pert_mask=[15930],
    adj_mask=[13214, 6607],
  },
  (gene, string9_1_neighborhood, gene)={
    edge_index=[2, 104408],
    num_edges=[2],
    pert_mask=[104416],
    adj_mask=[13214, 6607],
  }
)
batch["gene"].ptr
tensor([    0,  6604, 13208])
```

Evidence showing we know that `list[str]` can be batched nicely. Unsure of dict[str,int].

```python
batch
HeteroDataBatch(
  gene={
    node_ids=[2],
    num_nodes=13208,
    ids_pert=[2],
    cell_graph_idx_pert=[6],
    x=[13214, 0],
    x_batch=[13214],
    x_ptr=[3],
    x_pert=[6, 0],
    x_pert_batch=[6],
    x_pert_ptr=[3],
    gene_interaction=[2],
    gene_interaction_p_value=[2],
    pert_mask=[13214],
    gene_interaction_original=[2],
    mask=[13214],
    batch=[13208],
    ptr=[3],
  },
  (gene, string9_1_coexpression, gene)={
    edge_index=[2, 640508],
    num_edges=[2],
    pert_mask=[640590],
    adj_mask=[13214, 6607],
  },
  (gene, string9_1_cooccurence, gene)={
    edge_index=[2, 18526],
    num_edges=[2],
    pert_mask=[18532],
    adj_mask=[13214, 6607],
  },
  (gene, string9_1_database, gene)={
    edge_index=[2, 80113],
    num_edges=[2],
    pert_mask=[80186],
    adj_mask=[13214, 6607],
  },
  (gene, string9_1_experimental, gene)={
    edge_index=[2, 451368],
    num_edges=[2],
    pert_mask=[452114],
    adj_mask=[13214, 6607],
  },
  (gene, string9_1_fusion, gene)={
    edge_index=[2, 15924],
    num_edges=[2],
    pert_mask=[15930],
    adj_mask=[13214, 6607],
  },
  (gene, string9_1_neighborhood, gene)={
    edge_index=[2, 104408],
    num_edges=[2],
    pert_mask=[104416],
    adj_mask=[13214, 6607],
  }
)
batch["gene"][node_ids]
Traceback (most recent call last):
  File "<string>", line 1, in <module>
NameError: name 'node_ids' is not defined
batch["gene"].node_ids
[['Q0010', 'Q0017', 'Q0032', 'Q0045', 'Q0050', 'Q0055', 'Q0060', 'Q0065', 'Q0070', 'Q0075', 'Q0080', 'Q0085', 'Q0092', 'Q0105', 'Q0110', 'Q0115', 'Q0120', 'Q0130', 'Q0140', ...], ['Q0010', 'Q0017', 'Q0032', 'Q0045', 'Q0050', 'Q0055', 'Q0060', 'Q0065', 'Q0070', 'Q0075', 'Q0080', 'Q0085', 'Q0092', 'Q0105', 'Q0110', 'Q0115', 'Q0120', 'Q0130', 'Q0140', ...]]
```