---
id: vz6vfvt1dglgnzsauts03vz
title: Self_attention_sag
desc: ''
updated: 1713024691164
created: 1713023392937
---
This network is supposed to take a set of nodes without any `edge_index` and to use an attention layer on the set of nodes, pass these through a softmax, binarize the weights and use these as the `edge_index` in for the start of the SAG pool. We call this `SelfAttentionSAG`.

Like the SelfAttentionDeepSet I want the attention weights to be returned as [num_heads, num_nodes, num_nodes] for the first attention layer that is used on the set of nodes. 

In this model I want their to be multiple sag pool layers So i can pool down to size 1 for each graph. I have a supervised regression label that can be size 1 or two and that is why I need to do this. The idea is to take a set of nodes from a graph with no edges, compute attention over nodes, take attention and pass through softmax, binarize to get edge index for sag pool, pool down to a single representation of the graph, then use a prediction head on the supervised label.