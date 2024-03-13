---
id: 3l2s60astw2v7iiq2pwc398
title: Datasets
desc: ''
updated: 1690934720266
created: 1690835565997
---
## Datasets Outline Schematic

- Datasets is probably the most crucial component of the library to get right. We want to allow users to be able to quickly construct `torchcell.datasets` for subsequent downstream machine learning tasks.
- If this is done correct we should be able to use datasets for both inference (supervised) and generative learning tasks.

Rough Outline of Datasets

![](./assets/drawio/Pipeline.drawio.png)

- Under this model, datasets will need to be able to construct mutant geneotypes, for the **sequence view** and the **gene node view**.
  - **Sequence view** allows for further modelling with DNA language models. Possibly even protein models, but this can also
  - **Gene node view** allows for inclusion of knowledge contained about any given gene within the *S. cereviae* gene. A minimal definition of a node will allow for the inclusion of heterologous genes.
- Datasets should be able to take in the genome strain name such as `"S288c"` or `"BY4741"`, or a list of genes, which would be a longer specification.

## Mapping out a Future for Higher Order Gene Manipulation

"While gffutils and Biopython provide powerful utilities to parse and analyze genome data, they don't have explicit functionality for simulating gene modifications or genome editing.
However, you can still use their functionalities to perform modifications manually. Here is a high-level workflow you could consider:

1. Parse the genome sequence and its annotation (the GFF file) using Biopython and gffutils respectively.
2. Identify the locus where you want to insert the gene. You can do this using the gffutils database which gives you access to the features of the genome and their coordinates.
3. To insert a gene, you can slice the genome sequence at the desired position, insert the new gene sequence, and then concatenate the rest of the genome sequence. You can perform these operations using standard Python string manipulation since Biopython treats sequences as strings.
4. Once the genome sequence has been modified, you will also need to update the annotation to reflect the insertion of the new gene. You can create a new feature for the inserted gene with its coordinates and add it to the gffutils database. All downstream features (those located after the insertion point) will need to have their coordinates updated to account for the length of the inserted sequence.
5. If you're frequently making small insertions, you may want to keep a separate record of the modifications (start point, end point, and inserted sequence) and apply these modifications as needed, rather than creating a new modified genome sequence each time. This would be more space and computationally efficient.
You'll need to write custom code for steps 3-5, as these tasks involve specific genome editing operations that are beyond the scope of general-purpose libraries like gffutils and Biopython. Additionally, keep in mind that this is a simplified view of genome editing and doesn't take into account many complexities such as potential impacts on existing genes, regulatory elements, and other features.
If you're looking for more sophisticated genome editing simulations, you may want to explore specialized software or libraries tailored for this purpose." - #ChatGPT

## Joining Datasets with Union and Intersection

```python
from torchcell.datasets import SCerevisiae, experiments
scerevisiae_experiments = experiments[SCerevisiae] # dict containing metadata
scerevisiae_experiments.to_list() # list of experiment names
dataset = SCerevisiae(experiments = scerevisiae_experiments.to_list()) # Loading dataset with all experiments
```

| dataset_name            | Size | Prediction Level                | Description |
|:------------------------|:-----|---------------------------------|-------------|
| Costanzo                | 2    | Global                          |             |
| Kuzmin                  | 2    | Global                          |             |
| Baryshnikovna           | 2    | Global                          |             |
| Morphology              | 2    | Global                          |             |
| GFP                     | 2    | Node (Gene)                     |             |
| Expression              | 2    | Node (Gene)                     |             |
| Cell paper              | 2    | Node (Metabolite)               |             |
| Compartmentalization    | 2    | Node (Gene) Subset... Organelle |             |
| Magic                   | 2    | Global + Env                    |             |
| Chemical Tolerance Cell | 2    | Global + Env                    |             |

- A symmetric indicator matrix for Union and Intersection can be constructured for each to build a dataset for subsequent machine learning tasks.
