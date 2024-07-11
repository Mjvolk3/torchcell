---
id: nm1xfoeq1t8o5t5ronnky70
title: '02'
desc: ''
updated: 1720233760628
created: 1718668991023
---
Goal journal: Nature Machine Intelligence

Plan from [[paper-outline-01|dendron://torchcell/paper.outline.01]] was to exploit 4 lines we have high confidence in and explore 3 lines that have high probability of failure. I'll outline 7 sections that are the goal for submission of the paper. We can call them Type A and B respectively.

Also adapting from [[paper-outline sections|dendron://torchcell/paper.outline#sections]].

1. P: Data problem in systems biology and metabolic engineering S: Torchcell, Neo4j, Biocypher
2. P: DCell visible neural network (Fitness) S: One Hot set net (Fitness)  
3. P: DCell (interactions) S:  DiffPool (interactions) - Compare ontology to learned clustering. Interpretability.
4. P: Multimodal models with expression data and morphology (multimodal learning) S: morphologic state prediction benefit from fitness and expression data?
5. P: Generative strain design (generative modeling) S: Solve the combinatorics problem when constructing multiplex mutants.

## TorchCell Software Helps Standardize Systems and Biological Data Types for Machine Learning Tasks (Type A - Software Dev)

- 1. Key Idea: systems biology and metabolic engineering objects can be naturally represented as graphs. In other words graphs are the most general object necessary for representing the objects we care about. We also can leverage sequence models. View cell states as aggregations over different representations.
- 2. Key Idea: View experiments as perturbations to objects. This is important for how we load and store data.
- 3. Example of *S. cerevisiae* with graphs, node data, sequence embeddings.
- 4. Types of tasks that we wish to facilitate from this data.
- 5. (Transition)

## TorchCell Neo4j Graph Database Standardizes Format and Modularizes Addition of Experiments (Type A - Software Dev)

- 1. Many experiments pertrub same objects.
- 2. Data schema linked to biolink via pydantic classes. This isn't perfect but it is transparent etc. We eventually want a way to continue to update `create_experiment` that gets tested as the schema updates. This is likely could be delegated to an llm to test loop. This would scale.
- 3. Querying data, deduplication, to lmdb where.
- 4. Queried dataset then used as perturbation to some base graph. Examples.
- 5. (Transition)

## Fitness prediction without Deep Learning (Type A - Done Before)

- 1. Transitive learning setting to see if we can predict global phenotypes from sets of nodes - exploration of representation. Would be nice to have some theory here which I believe should be feasible.
- 2. `elastic-net`, `random-forest`, `svr` over different dataset sizes - This works for predicting fitness.
- 3. Interpretable in that we should be able to look at feature importance say in RF for one hot, then look at GO enrichment and compare this to `DCell`... Not sure how far we should pursue this line.
- 4. Comparison to Mechanistic-Aware ML.
- 5. Can we use expression as a label. Using their data show best over previous best feature. Then use expression as label, not feature. Allows for flexibility and motivates inclusion.
- 6. Theory on simplest cases? Sets of random numbers?
- 7. (Transition) Show when fails, should fail in essential gene case. Is this possible with interactions?

## Interaction prediction without Deep Learning (Type A - Done Before)

- 1. Motivation via non deep methods, `elastic-net`, `random-forest`, `svr`.
- 2. MLP over LLM features
- 3. We know PPI helps, Dango, Add interactions
- 4. Comparison to DCell

## Multi-Task Supervised Learning on Interactions and Fitness with DCell Comparison

- 1. In the interactions fitness case does fitness data improved interaction prediction?
- 2. Best model from previous plus training with fitness data.
- 3. DCell trained on both - Does it improve?
- 4. 
- 5. (Transition) Show if models fail on some essential genes

## Multi-Task Supervised ML with all data.

- 1. labels: expression, fitness, interactions, morphology
- 2.
- 3.
- 4.
- 5.

## Discussion

- 1. We want to expand to datasets that have environment conditions. As shown by Science paper controlling environment conditions can larger effect than genetic engineering.
- 2. Want to include metabolic engineering datasets so we can test hypothesis related to whether systems biology data can help improve predicitons related to metabolic engineering design.
- 3. Want to include strains with greater deviations from `s288c` where we have different genomes sequence file and annotations.
- 4. Want to include similarly related species so we can attempt transfer learning approaches to non-model organisms.
- 5. Dynamic simulation for supplementing real world data. The stoichiometric matrix is powerful, we should use it. After connection to `cobrapy`, we would want connections and standardization with dynamical models. These type of things will need more eyes / lots of help.
- 6. This is the natural thing to do given the state of the wet lab experimental technology and state of computing technology. The vision combines physics/chemistry based understanding in simulation and more correlative approaches based on large high throughput experimentation. It is uncertain if either software or database is the critical component, and there are opportunities to move more data into database where database becomes the primary export from the efforts on the paper. The software should provide a nice way to quickly get going with your selected dataset. Leave much up to developers to decided what are reasonable assumptions while making the combination of data relatively easy.
