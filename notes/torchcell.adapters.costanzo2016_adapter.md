---
id: cg7e5soujlboc77wykbj1n3
title: Costanzo2016_adapter
desc: ''
updated: 1705540264975
created: 1705537951301
---
## Using Static Methods like in get_perturbation

This has the luxury of avoiding iterating over the entire dataset, but it doesn't have the ability to check for duplicates. When you iterate over the entire dataset with a regular method it is easier to create a set of data within the scope that that can be used for removing duplicates. It is so cumbersome with a `@staticmtehod` pushing the duplicate tracking a function above making things difficult to read. 

I think think the `@staticmethod` is justified when the data has some guarantee that there are no duplicates. This is why it can be safely used on perturbation. With the hourglass strategy we known that `strain_id` is always unique. This comes from knowledge about the preprocessing of the data and probably should be used with some care. We have been using the hourglass strategy with some discretion, for instance it is not used on media. There isn't any additional information that would allow us to used this design pattern. Besides we would end up with weird things like the two different nodes at a specific temperature but with some additional differentiating attribute. We don't want this property for graph querying.

 ![](./assets/drawio/ontology_pydantic_hourglass_data_model.drawio.png)

```python
@staticmethod
def _get_perturbation(
    genotype: BaseGenotype,
) -> Generator[BioCypherNode, None, None]:
    if genotype.perturbation:
        i = 1
        perturbation_id = hashlib.md5(
            json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
        ).hexdigest()

        yield BioCypherNode(
            node_id=perturbation_id,
            preferred_id=f"perturbation_{i}",
            node_label="perturbation",
            properties={
                "systematic_gene_name": [
                    genotype.perturbation.systematic_gene_name
                ],
                "perturbed_gene_name": [genotype.perturbation.perturbed_gene_name],
                "description": genotype.perturbation.description,
                "perturbation_type": genotype.perturbation.perturbation_type,
                "strain_id": genotype.perturbation.strain_id,
                "serialized_data": json.dumps(genotype.perturbation.model_dump()),
            },
        )
```

## Useful Functions for Debugging Adapter and Printing Ontology

`bc.show_ontology_structure` can be used to print the ontology before processing any of the node data.

```python
if __name__ == "__main__":
  from biocypher import BioCypher

  #Simple Testing
  dataset = SmfCostanzo2016Dataset()
  adapter = SmfCostanzo2016Adapter(dataset=dataset)
  [i for i in adapter.get_nodes()]
  [i for i in adapter.get_edges()]
  
  ## Advanced Testing
  bc = BioCypher()
  dataset = SmfCostanzo2016Dataset()
  adapter = SmfCostanzo2016Adapter(dataset=dataset)
  print(bc.show_ontology_structure())
  bc.write_nodes(adapter.get_nodes())
  bc.write_edges(adapter.get_edges())

  # # Write admin import statement and schema information (for biochatter)
  bc.write_import_call()
  bc.write_schema_info(as_node=True)

  # # Print summary
  bc.summary()
  print()
```