from typing import Dict, List, Optional, Tuple, Union, Literal, Set
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn.aggr.attention import AttentionalAggregation

from torchcell.nn.self_attention_block import SelfAttentionBlock
from torchcell.nn.masked_attention_block import NodeSelfAttention


class _HeteroNSA_Block(nn.Module):
    """
    Internal helper block for HeteroNSA.
    If layer_type=='M', applies masked attention (MAB) using per–edge-type NodeSelfAttention
    (with an optional aggregator if aggregation=='attention').
    If layer_type=='S', applies self-attention (SAB) using per–node-type SelfAttentionBlock.
    """
    def __init__(
        self,
        layer_type: Literal["M", "S"],
        hidden_dim: int,
        node_types: Set[str],
        edge_types: Set[Tuple[str, str, str]],
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        aggregation: Literal["sum", "mean", "attention"] = "sum",
    ):
        super().__init__()
        if layer_type not in ["M", "S"]:
            raise ValueError(f"Invalid layer_type '{layer_type}'. Must be 'M' or 'S'.")
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.node_types = node_types
        self.edge_types = edge_types
        self.aggregation = aggregation

        if layer_type == "M":
            self.masked_blocks = nn.ModuleDict()
            for (src, rel, dst) in edge_types:
                key = f"{src}__{rel}__{dst}"
                self.masked_blocks[key] = NodeSelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=activation,
                )
            if aggregation == "attention":
                self.node_aggregators = nn.ModuleDict()
                for nt in node_types:
                    connected = [(s, r, d) for (s, r, d) in edge_types if s == nt or d == nt]
                    if len(connected) > 1:
                        gate_nn = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim // 2),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim // 2, 1),
                        )
                        transform_nn = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        )
                        self.node_aggregators[nt] = AttentionalAggregation(
                            gate_nn=gate_nn, nn=transform_nn
                        )
        else:  # layer_type == "S"
            self.self_blocks = nn.ModuleDict()
            for nt in node_types:
                self.self_blocks[nt] = SelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=activation,
                )

    def _process_with_mask(
        self,
        block: nn.Module,
        embeddings: Tensor,
        mask: Tensor,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        original_dim = embeddings.dim()
        if mask.dim() == 2 and embeddings.dim() == 2:
            mask = mask.unsqueeze(0)
            embeddings = embeddings.unsqueeze(0)
        elif mask.dim() == 2 and embeddings.dim() == 3:
            mask = mask.unsqueeze(0).expand(embeddings.size(0), -1, -1)
        elif mask.dim() == 3 and embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        if isinstance(block, NodeSelfAttention):
            if edge_attr is not None and edge_index is not None:
                out = block(embeddings, mask, edge_attr, edge_index)
            else:
                out = block(embeddings, mask)
        else:
            out = block(embeddings)
        if original_dim == 2 and out.dim() == 3:
            out = out.squeeze(0)
        return out

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        data: HeteroData,
        batch_idx: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        if self.layer_type == "M":
            relation_outputs = {nt: [] for nt in self.node_types}
            for (src, rel, dst) in self.edge_types:
                key = f"{src}__{rel}__{dst}"
                if key not in self.masked_blocks:
                    continue
                if (src, rel, dst) not in data.edge_types:
                    continue
                block = self.masked_blocks[key]
                src_emb = x_dict.get(src, None)
                dst_emb = x_dict.get(dst, None)
                if src_emb is None or dst_emb is None:
                    continue
                edge_store = data[src, rel, dst]
                if hasattr(edge_store, "adj_mask") and edge_store.adj_mask is not None:
                    adj_mask = edge_store.adj_mask
                    edge_attr = getattr(edge_store, "edge_attr", None)
                    edge_index = getattr(edge_store, "edge_index", None)
                    if rel == "rmr" and hasattr(edge_store, "stoichiometry"):
                        edge_attr = edge_store.stoichiometry
                        edge_index = edge_store.edge_index
                    if src == dst:
                        out_src = self._process_with_mask(block, src_emb, adj_mask, edge_attr, edge_index)
                        relation_outputs[src].append(out_src)
                    else:
                        out_src = self._process_with_mask(block, src_emb, adj_mask, edge_attr, edge_index)
                        relation_outputs[src].append(out_src)
                        adj_mask_t = adj_mask.transpose(-2, -1)
                        out_dst = self._process_with_mask(block, dst_emb, adj_mask_t, edge_attr, edge_index)
                        relation_outputs[dst].append(out_dst)
                elif hasattr(edge_store, "inc_mask") and edge_store.inc_mask is not None:
                    inc_mask = edge_store.inc_mask
                    edge_attr = None
                    edge_index = None
                    if rel == "rmr" and hasattr(edge_store, "stoichiometry"):
                        edge_attr = edge_store.stoichiometry
                        edge_index = getattr(edge_store, "hyperedge_index", None) or edge_store.edge_index
                    out_src = self._process_with_mask(block, src_emb, inc_mask, edge_attr, edge_index)
                    relation_outputs[src].append(out_src)
                    if src != dst:
                        inc_mask_t = inc_mask.transpose(-2, -1)
                        out_dst = self._process_with_mask(block, dst_emb, inc_mask_t, edge_attr, edge_index)
                        relation_outputs[dst].append(out_dst)
                elif hasattr(edge_store, "edge_index"):
                    edge_index = edge_store.edge_index
                    edge_attr = getattr(edge_store, "edge_attr", None)
                    if src == dst:
                        adj = torch.eye(src_emb.size(0), device=src_emb.device).bool()
                        out_src = self._process_with_mask(block, src_emb, adj, edge_attr, edge_index)
                        relation_outputs[src].append(out_src)
                    else:
                        src_adj = torch.eye(src_emb.size(0), device=src_emb.device).bool()
                        dst_adj = torch.eye(dst_emb.size(0), device=dst_emb.device).bool()
                        out_src = self._process_with_mask(block, src_emb, src_adj, edge_attr, edge_index)
                        out_dst = self._process_with_mask(block, dst_emb, dst_adj, edge_attr, edge_index)
                        relation_outputs[src].append(out_src)
                        relation_outputs[dst].append(out_dst)
            new_x = {}
            for nt, outs in relation_outputs.items():
                if not outs:
                    new_x[nt] = x_dict.get(nt, None)
                    continue
                if len(outs) == 1:
                    new_x[nt] = outs[0]
                else:
                    if self.aggregation == "sum":
                        new_x[nt] = sum(outs)
                    elif self.aggregation == "mean":
                        new_x[nt] = sum(outs) / len(outs)
                    elif self.aggregation == "attention" and hasattr(self, "node_aggregators"):
                        if nt in self.node_aggregators:
                            flat_embs = torch.cat(outs, dim=0)
                            node_indices = torch.arange(len(outs), device=flat_embs.device).repeat_interleave(outs[0].size(0))
                            new_x[nt] = self.node_aggregators[nt](flat_embs, index=node_indices)
                        else:
                            new_x[nt] = sum(outs) / len(outs)
                    else:
                        new_x[nt] = sum(outs) / len(outs)
            return new_x
        else:
            new_x = {}
            for nt, emb in x_dict.items():
                if nt not in self.self_blocks:
                    new_x[nt] = emb
                    continue
                block = self.self_blocks[nt]
                if emb.dim() == 2:
                    emb = emb.unsqueeze(0)
                    out = block(emb).squeeze(0)
                else:
                    out = block(emb)
                new_x[nt] = out
            return new_x


class HeteroNSA(nn.Module):
    """
    Heterogeneous Node-Set Attention (HeteroNSA) module.
    Instead of reusing a single set of modules for every block, this version builds a
    stack of independent blocks—one per element in the provided pattern.
    """
    def __init__(
        self,
        hidden_dim: int,
        node_types: Set[str],
        edge_types: Set[Tuple[str, str, str]],
        pattern: List[str],
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        aggregation: Literal["sum", "mean", "attention"] = "sum",
    ):
        super().__init__()
        if not pattern:
            raise ValueError("Pattern list cannot be empty")
        for block_type in pattern:
            if block_type not in ["M", "S"]:
                raise ValueError(f"Invalid block type '{block_type}'. Must be 'M' or 'S'.")
        if aggregation not in ["sum", "mean", "attention"]:
            raise ValueError(f"Invalid aggregation '{aggregation}'. Must be 'sum', 'mean', or 'attention'.")
        self.hidden_dim = hidden_dim
        self.node_types = node_types
        self.edge_types = edge_types
        self.pattern = pattern
        self.aggregation = aggregation

        self.blocks = nn.ModuleList([
            _HeteroNSA_Block(
                layer_type=layer_type,
                hidden_dim=hidden_dim,
                node_types=node_types,
                edge_types=edge_types,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
                aggregation=aggregation,
            )
            for layer_type in pattern
        ])
        # Expose the first block's _process_with_mask for monkey patching.
        self._process_with_mask = self.blocks[0]._process_with_mask

    def forward(
        self,
        node_embeddings: Dict[str, Tensor],
        data: HeteroData,
        batch_idx: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        x_dict = {k: v.clone() for k, v in node_embeddings.items()}
        for block in self.blocks:
            x_dict = block(x_dict, data, batch_idx)
        return x_dict


class HeteroNSAEncoder(nn.Module):
    """
    Full encoder using HeteroNSA with input projections and multiple layers.
    The constructor accepts a 'pattern' (a list of 'M'/'S' strings) and a 'num_layers'
    count; a separate HeteroNSA is built for each layer.
    """
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int,
        node_types: Set[str],
        edge_types: Set[Tuple[str, str, str]],
        pattern: List[str],
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        aggregation: Literal["sum", "mean", "attention"] = "sum",
    ) -> None:
        super().__init__()
        if aggregation not in ["sum", "mean", "attention"]:
            raise ValueError(f"Invalid aggregation '{aggregation}'. Must be 'sum', 'mean', or 'attention'.")
        self.hidden_dim = hidden_dim
        self.node_types = node_types
        self.edge_types = edge_types

        self.input_projections = nn.ModuleDict(
            {nt: nn.Linear(dim, hidden_dim) for nt, dim in input_dims.items()}
        )

        # Build a separate HeteroNSA layer for each repetition.
        self.nsa_layers = nn.ModuleList([
            HeteroNSA(
                hidden_dim=hidden_dim,
                node_types=node_types,
                edge_types=edge_types,
                pattern=pattern,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
                aggregation=aggregation,
            )
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleDict(
            {nt: nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
             for nt in node_types}
        )
        self.graph_projections = nn.ModuleDict(
            {nt: nn.Linear(hidden_dim, hidden_dim) for nt in node_types}
        )
        self.final_projection = nn.Linear(hidden_dim * len(node_types), hidden_dim)

    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Tensor]:
        x_dict = {}
        batch_idx = {}
        for nt in self.node_types:
            if nt in data.node_types and hasattr(data[nt], "x") and data[nt].x is not None:
                proj = self.input_projections[nt](data[nt].x.clone())
                x_dict[nt] = proj
                if hasattr(data[nt], "batch") and data[nt].batch is not None:
                    batch_idx[nt] = data[nt].batch
        final_embeddings = {k: v.clone() for k, v in x_dict.items()}
        for i, nsa_layer in enumerate(self.nsa_layers):
            new_x = nsa_layer(x_dict, data, batch_idx)
            for nt in new_x:
                if nt in x_dict:
                    residual = x_dict[nt]
                    if residual.dim() != new_x[nt].dim():
                        if residual.dim() == 2 and new_x[nt].dim() == 3:
                            residual = residual.unsqueeze(0)
                        elif residual.dim() == 3 and new_x[nt].dim() == 2:
                            residual = residual.squeeze(0)
                    new_x[nt] = self.layer_norms[nt][i](new_x[nt] + residual)
            x_dict = new_x
            final_embeddings.update(x_dict)
        graph_embeddings = {}
        for nt, emb in final_embeddings.items():
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)
            proj = self.graph_projections[nt](emb)
            graph_embeddings[nt] = proj.mean(dim=1)
        sorted_nts = sorted(self.node_types)
        cat_list = []
        device = next(self.parameters()).device
        for nt in sorted_nts:
            if nt in graph_embeddings:
                cat_list.append(graph_embeddings[nt])
            else:
                cat_list.append(torch.zeros(1, self.hidden_dim, device=device))
        concatenated = torch.cat(cat_list, dim=-1)
        final_graph_embedding = self.final_projection(concatenated)
        return final_embeddings, final_graph_embedding
