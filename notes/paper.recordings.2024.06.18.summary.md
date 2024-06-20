---
id: taiwlsry3rx2r5komm6mtbv
title: Summary
desc: ''
updated: 1718912946972
created: 1718912735006
---

Torch Cell Project Overview

## Speed Comparison

- Random forest speed up seems to be real, but GPU-enabled version performs significantly worse (around 4 times faster according to logs)
- Need to test performance at a larger scale (1 million data points)

## Database Design

- Goal: Create a general knowledge graph approach
- Limitations: Neo4j property likely has a memory limit (1-10 MB)
- Solution: Store genome sequence and annotation files in a key-value store, mapped to the knowledge graph
- Knowledge graph captures experiments, genomes, and genotypes
- Future improvements: Automate transition between sequence and annotation using machine learning or bioinformatics

## Integration of Historical Data

- Incorporate data from publications and computational biologists
- Data includes correlative descriptions of biology (e.g., covariance matrices, directed interactions)
- Focus on modeling the central dogma around one gene
- Interaction networks: protein-protein, regulatory, and signaling networks
- Machine learning models can learn the importance of different networks

## Torch Cell Software Segregation

1. Torch Cell Database
   - Neo4j database with a unifying schema language
   - Links to the bio-link ontology
   - Slower data addition process
2. Torch Cell Genome Database
   - Key-value store for genome name, FASTA file, and annotation files (GAF)
   - Enforce minimal standards on genomes
   - Focus on S288C genome initially
   - Implement genome similarity criterion for joining genomes

## Project Timeline

- Three iterations within the next year
- Version 1 deliverables by the end of August (2.5 months)
  - Paper on work motivation, genomes, networks, Torch Cell database, software, and future vision
  - Demonstrate improved prediction using interaction data, fitness data, and protein interaction data - I think this needs to be in the small N regime.
  - Compare performance with published machine learning methods
- Parallel routes:
   1. Torch Cell software, documentation, testing, and engineering
   2. Torch Cell database builds, monitoring, and data addition
   3. Single-task learning for fitness and interactions using traditional machine learning models and graph neural networks
   4. GPU deep learning models. We can devote something 
