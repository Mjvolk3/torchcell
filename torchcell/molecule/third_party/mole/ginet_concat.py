# torchcell/molecule/third_party/mole/ginet_concat.py
# [[torchcell.molecule.third_party.mole.ginet_concat]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/molecule/third_party/mole/ginet_concat.py
"""MolE ``gin_concat`` backbone, vendored verbatim in structure from
https://github.com/rolayoalarcon/MolE (``models/ginet_concat.py``, MIT License,
Copyright (c) 2024 Roberto Olayo Alarcon; see ``LICENSE`` beside this file).

Only cosmetic edits: the ``print`` calls in ``__init__`` and ``load_my_state_dict``
are removed, ``load_my_state_dict`` is dropped in favor of a strict key check in the
encoder, and type hints are added. Layer names and the forward pass are unchanged so
the Zenodo checkpoint (record 10803099) loads by name.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import add_self_loops

num_atom_type = 119  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5  # including aromatic and self-loop edge
num_bond_direction = 3


class GINEConv(MessagePassing):  # type: ignore[misc]
    """GIN edge-aware convolution as used by MolE."""

    def __init__(self, emb_dim: int) -> None:
        """Sum aggregation (PyG default), edge embeddings for bond type and direction."""
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Message passing with self loops typed as bond type 4."""
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # type: ignore[no-any-return]

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Neighbor embedding plus edge embedding."""
        return x_j + edge_attr

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Apply the GIN MLP to the aggregated messages."""
        return self.mlp(aggr_out)  # type: ignore[no-any-return]


class GINet(nn.Module):
    """Graph Isomorphism Network whose graph representation is the concatenation of
    every layer's pooled node embeddings (``num_layer * emb_dim`` wide).

    ``forward`` returns ``(h_global_embedding, out)``: the static representation MolE
    calls ``r`` (what downstream ML models consume) and the projection ``z`` that
    fed the Barlow-Twins loss at pre-training. Encoders keep only the first.
    """

    def __init__(
        self,
        num_layer: int = 5,
        emb_dim: int = 300,
        feat_dim: int = 256,
        drop_ratio: float = 0,
        pool: str = "mean",
    ) -> None:
        """Defaults are upstream's; the released checkpoint uses 5 x 200, 8000, add."""
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.concat_dim = num_layer * emb_dim

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for _layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for _layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        elif pool == "add":
            self.pool = global_add_pool
        else:
            raise ValueError(f"unknown pool {pool!r}; expected mean, max or add")

        self.feat_lin = nn.Linear(self.concat_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim),  # Is not reduced to half size!
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    def forward(self, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """``(static representation r, projection z)`` for a PyG batch."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h_init = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        # Perform the convolutions
        h_dict: dict[str, torch.Tensor] = {}

        for layer in range(self.num_layer):
            if layer == self.num_layer - 1:
                tmp_h = self.gnns[layer](
                    h_dict[f"h_{layer - 1}"], edge_index, edge_attr
                )
                tmp_h = self.batch_norms[layer](tmp_h)
                h_dict[f"h_{layer}"] = F.dropout(
                    tmp_h, self.drop_ratio, training=self.training
                )
            elif layer == 0:
                tmp_h = self.gnns[layer](h_init, edge_index, edge_attr)
                tmp_h = self.batch_norms[layer](tmp_h)
                h_dict[f"h_{layer}"] = F.dropout(
                    F.relu(tmp_h), self.drop_ratio, training=self.training
                )
            else:
                tmp_h = self.gnns[layer](
                    h_dict[f"h_{layer - 1}"], edge_index, edge_attr
                )
                tmp_h = self.batch_norms[layer](tmp_h)
                h_dict[f"h_{layer}"] = F.dropout(
                    F.relu(tmp_h), self.drop_ratio, training=self.training
                )

        # Graph representation
        h_list_pooled = [
            self.pool(h_dict[f"h_{layer}"], data.batch)
            for layer in range(self.num_layer)
        ]
        h_global_embedding = torch.cat(h_list_pooled, dim=1)

        assert h_global_embedding.shape[1] == self.concat_dim

        # Projection
        h_expansion = self.feat_lin(h_global_embedding)
        out = self.out_lin(h_expansion)

        return h_global_embedding, out
