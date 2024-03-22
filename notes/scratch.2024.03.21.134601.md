---
id: j03jeikh0u7b8lfrt6uqeei
title: '134601'
desc: ''
updated: 1711046763359
created: 1711046763359
---
```mermaid
graph TD;
    torchcell.data --> torchcell.data.neo4j_cell;
    torchcell.data.neo4j_cell --> torchcell.datasets.embedding.BaseEmbeddingDataset;
    torchcell.datasets.embedding.BaseEmbeddingDataset --> torchcell.datasets.dcell;
    torchcell.datasets.dcell --> torchcell.datasets.scerevisiae.costanzo2016;
    torchcell.datasets.scerevisiae.costanzo2016 --> torchcell.data;
```