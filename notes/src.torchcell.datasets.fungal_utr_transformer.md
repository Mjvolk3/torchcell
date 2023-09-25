---
id: 8ln9jow8vq3wxj41jtqvwt9
title: Fungal_utr_transformer
desc: ''
updated: 1692115257874
created: 1692115133147
---
This file will construct datasets for sequence embeddings since they can take a long time to compute. Then we should also be able to aggregate the embedddings into one object. This aggegation should only really happen over the superset or reference genome. Since we don't want to compute the embeddings during a training loop