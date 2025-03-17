import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
from torch_geometric.datasets import StochasticBlockModelDataset
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple, Union, Set
from torch_scatter import scatter


def load_sample_data_batch():
    import os
    import os.path as osp
    from dotenv import load_dotenv
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.datamodules import CellDataModule
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )

    # from torchcell.datasets.fungal_up_down_transformer import (
    #     FungalUpDownTransformerDataset,
    # )
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.data import MeanExperimentDeduplicator
    from torchcell.data import GenotypeAggregator
    from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.data import Neo4jCellDataset
    from torchcell.data.neo4j_cell import SubgraphRepresentation
    from tqdm import tqdm
    from torchcell.metabolism.yeast_GEM import YeastGEM

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    print(f"DATA_ROOT: {DATA_ROOT}")

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
    )
    # IDEA we are trying to use all gene reprs
    # genome.drop_chrmt()
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    # selected_node_embeddings = ["codon_frequency"]
    selected_node_embeddings = ["empty"]
    node_embeddings = {}
    # if "fudt_downstream" in selected_node_embeddings:
    #     node_embeddings["fudt_downstream"] = FungalUpDownTransformerDataset(
    #         root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
    #         genome=genome,
    #         model_name="species_downstream",
    #     )

    # if "fudt_upstream" in selected_node_embeddings:
    #     node_embeddings["fudt_upstream"] = FungalUpDownTransformerDataset(
    #         root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
    #         genome=genome,
    #         model_name="species_upstream",
    #     )
    if "codon_frequency" in selected_node_embeddings:
        node_embeddings["codon_frequency"] = CodonFrequencyDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
            genome=genome,
        )

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    reaction_map = gem.reaction_map

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        incidence_graphs={"metabolism": reaction_map},
        node_embeddings=node_embeddings,
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    seed = 42
    # Base Module
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=seed,
        num_workers=2,
        pin_memory=False,
    )
    cell_data_module.setup()

    # 1e4 Module
    size = 5e4
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    max_num_nodes = len(dataset.gene_set)
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break
    input_channels = dataset.cell_graph["gene"].x.size()[-1]
    return dataset, batch, input_channels, max_num_nodes


# Create a container class to properly register modules
class BlockContainer(nn.Module):
    def __init__(self, block_type, content):
        super().__init__()
        self.block_type = block_type

        # Handle different module structures
        if isinstance(content, dict):
            # For SAB case with dict of node_type -> module
            self.content_dict = nn.ModuleDict(content)
            self.type = "dict"
        elif isinstance(content, tuple) and all(
            isinstance(m, nn.Module) for m in content
        ):
            # For M_GPR and M_MRM with tuple of modules
            self.content_list = nn.ModuleList(content)
            self.type = "tuple"
        else:
            # For M_GENE with single module
            self.content = content
            self.type = "single"

    def get_content(self):
        if self.type == "dict":
            return self.content_dict
        elif self.type == "tuple":
            return tuple(self.content_list)
        else:
            return self.content


class AttentionBlock(nn.Module):
    """Base Attention Block with common components for both MAB and SAB"""

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Define the common components
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # MLP after attention
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x, adj_matrix=None):
        raise NotImplementedError("Implemented in subclasses")


class MAB(AttentionBlock):
    """Masked Attention Block - Uses graph structure to mask attention"""

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__(hidden_dim, num_heads)

    def forward(self, x, adj_matrix):
        device = x.device

        # First normalization layer
        normed_x = self.norm1(x)

        # Reshape for multi-head attention
        batch_size, num_nodes, _ = normed_x.shape
        q = k = v = normed_x.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_nodes, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Choose implementation based on device
        if device.type == "cuda":
            # Use FlexAttention with GPU
            def node_mask_mod(b, h, q_idx, kv_idx):
                return adj_matrix[b, q_idx, kv_idx]

            block_mask = create_block_mask(
                node_mask_mod,
                B=batch_size,
                H=None,
                Q_LEN=num_nodes,
                KV_LEN=num_nodes,
                device=device,
            )

            attn_output = flex_attention(q, k, v, block_mask=block_mask)
        else:
            # Manual implementation for CPU
            scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)

            # Expand adjacency matrix to match batch and heads
            mask = adj_matrix.unsqueeze(1).expand(batch_size, self.num_heads, -1, -1)

            # Apply mask: -inf where there's no edge
            masked_scores = torch.where(
                mask > 0, scores, torch.tensor(-float("inf"), device=device)
            )

            # Apply softmax and compute weighted values
            attn_weights = torch.nn.functional.softmax(masked_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, num_nodes, self.hidden_dim)

        # First residual connection
        x = x + attn_output

        # Second normalization and MLP
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)

        # Second residual connection
        output = x + mlp_output

        return output


class SAB(AttentionBlock):
    """Self Attention Block - Uses standard self-attention with no masking"""

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__(hidden_dim, num_heads)

    def forward(self, x, adj_matrix=None):
        device = x.device

        # First normalization layer
        normed_x = self.norm1(x)

        # Reshape for multi-head attention
        batch_size, num_nodes, _ = normed_x.shape
        q = k = v = normed_x.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_nodes, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Choose implementation based on device
        if device.type == "cuda":
            # Just use regular attention with no masking
            attn_output = flex_attention(q, k, v)
        else:
            # Manual implementation for CPU
            scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)

            # Apply softmax and compute weighted values (no masking)
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, num_nodes, self.hidden_dim)

        # First residual connection
        x = x + attn_output

        # Second normalization and MLP
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)

        # Second residual connection
        output = x + mlp_output

        return output


class StoichiometricMAB(AttentionBlock):
    """Masked Attention Block with stoichiometry support for metabolite-reaction interactions"""

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__(hidden_dim, num_heads)
        # Add a stoichiometry gate
        self.stoich_gate = nn.Sequential(nn.Linear(1, hidden_dim), nn.Sigmoid())

    def forward(self, x, adj_matrix, stoichiometry=None):
        device = x.device

        # First normalization layer
        normed_x = self.norm1(x)

        # Reshape for multi-head attention
        batch_size, num_nodes, _ = normed_x.shape
        q = k = v = normed_x.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_nodes, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Process using manual attention to incorporate stoichiometry
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)

        # Expand adjacency matrix to match batch and heads
        mask = adj_matrix.unsqueeze(1).expand(batch_size, self.num_heads, -1, -1)

        # Apply mask: -inf where there's no edge
        masked_scores = torch.where(
            mask > 0, scores, torch.tensor(-float("inf"), device=device)
        )

        # Apply stoichiometry if provided - weight the attention scores
        if stoichiometry is not None:
            # Process stoichiometry values - shape: [batch_size, num_nodes, num_nodes]
            stoich_gates = self.stoich_gate(stoichiometry.unsqueeze(-1))
            stoich_gates = stoich_gates.unsqueeze(1)  # add heads dimension

            # Apply stoichiometry weights to masked scores
            masked_scores = masked_scores * stoich_gates

        # Apply softmax and compute weighted values
        attn_weights = torch.nn.functional.softmax(masked_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, num_nodes, self.hidden_dim)

        # First residual connection
        x = x + attn_output

        # Second normalization and MLP
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)

        # Second residual connection
        output = x + mlp_output

        return output


def group(xs: List[torch.Tensor], aggr: Optional[str]) -> Optional[torch.Tensor]:
    """Aggregation function similar to the one in HeteroConv"""
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return torch.cat(xs, dim=-1)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class CellGraphHeteroNSA(nn.Module):
    """
    A specialized heterogeneous Node-Set Attention for cell graphs with gene, metabolite, and reaction nodes.

    This module is designed to handle:
    - Gene multigraph with physical and regulatory interactions
    - Metabolite-reaction-metabolite hypergraph with stoichiometry
    - Gene-reaction bipartite connections

    Args:
        hidden_dim: Hidden dimension for all node types
        pattern: List of strings specifying the sequence of attention blocks
                Each element should be one of:
                - 'S': Self-attention for all node types
                - 'M_GENE': Masked attention for gene multigraph
                - 'M_GPR': Masked attention for gene-reaction bipartite connections
                - 'M_MRM': Masked attention for metabolite-reaction-metabolite hypergraph
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        node_counts: Dict[str, int],
        hidden_dim: int,
        pattern: List[str],
        num_heads: int = 8,
        aggr: str = "sum",
    ):
        super().__init__()

        # Node counts for embedding layers
        self.node_counts = node_counts
        self.hidden_dim = hidden_dim
        self.pattern = pattern
        self.num_heads = num_heads
        self.aggr = aggr

        # Create embeddings for each node type
        self.embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(count, hidden_dim)
                for node_type, count in node_counts.items()
            }
        )

        # Create attention blocks according to the pattern
        self.blocks = nn.ModuleList()

        # Parse and create blocks from the pattern
        for i, block_type in enumerate(pattern):
            if block_type == "S":
                # Self attention blocks for each node type
                self.blocks.append(
                    BlockContainer(
                        "SAB",
                        {
                            node_type: SAB(hidden_dim, num_heads)
                            for node_type in node_counts.keys()
                        },
                    )
                )
                print(f"  Added layer {i}: Self Attention Blocks for all node types")

            elif block_type == "M_GENE":
                # Masked attention for gene multigraph
                self.blocks.append(BlockContainer("M_GENE", MAB(hidden_dim, num_heads)))
                print(f"  Added layer {i}: Masked Attention Block for gene multigraph")

            elif block_type == "M_GPR":
                # Masked attention for gene-reaction bipartite
                self.blocks.append(
                    BlockContainer(
                        "M_GPR",
                        (
                            MAB(hidden_dim, num_heads),  # gene -> reaction
                            MAB(hidden_dim, num_heads),  # reaction -> gene
                        ),
                    )
                )
                print(
                    f"  Added layer {i}: Masked Attention for gene-reaction bipartite"
                )

            elif block_type == "M_MRM":
                # Masked attention for metabolite-reaction-metabolite hypergraph
                self.blocks.append(
                    BlockContainer(
                        "M_MRM",
                        (
                            StoichiometricMAB(
                                hidden_dim, num_heads
                            ),  # metabolite -> reaction
                            StoichiometricMAB(
                                hidden_dim, num_heads
                            ),  # reaction -> metabolite
                        ),
                    )
                )
                print(
                    f"  Added layer {i}: Stoichiometric Masked Attention for metabolite-reaction hypergraph"
                )

    def forward(self, data: HeteroData, batch_size: int = 1):
        """
        Forward pass through the heterogeneous NSA model.

        Args:
            data: A HeteroData object containing node features and edge connections
            batch_size: Number of graphs in the batch

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping node types to their updated representations
        """
        device = next(self.parameters()).device

        # Initialize node embeddings for each type
        node_embeddings = {}
        node_masks = {}
        node_ranges = {}  # Track the index ranges for each node type in each batch

        # Process each node type
        for node_type, embedding_layer in self.embeddings.items():
            if node_type not in data:
                continue

            # Get number of nodes for this type
            num_nodes = data[node_type].num_nodes

            # Create indices for embeddings
            if hasattr(data[node_type], "batch"):
                # Handle batched data
                batch = data[node_type].batch
                ptr = data[node_type].ptr

                # Track max nodes per batch for this node type
                max_nodes = (ptr[1:] - ptr[:-1]).max().item()

                # Initialize the embeddings tensor
                embeddings = torch.zeros(
                    batch_size, max_nodes, self.hidden_dim, device=device
                )

                # Track node indices for each batch element
                node_ranges[node_type] = []

                # For each batch element
                for b in range(batch_size):
                    batch_mask = batch == b
                    nodes_in_batch = batch_mask.nonzero().squeeze()
                    num_nodes_in_batch = nodes_in_batch.size(0)

                    # Get embeddings for these nodes
                    indices = torch.arange(num_nodes_in_batch, device=device)
                    node_embeddings_batch = embedding_layer(indices)

                    # Store in the right place in the tensor
                    embeddings[b, :num_nodes_in_batch] = node_embeddings_batch

                    # Track this batch's node range
                    start_idx = ptr[b].item()
                    end_idx = ptr[b + 1].item()
                    node_ranges[node_type].append((start_idx, end_idx))

                # Create a mask for valid nodes
                mask = torch.zeros(
                    batch_size, max_nodes, dtype=torch.bool, device=device
                )
                for b in range(batch_size):
                    mask[b, : (ptr[b + 1] - ptr[b])] = True

                node_embeddings[node_type] = embeddings
                node_masks[node_type] = mask

            else:
                # Single graph case
                indices = torch.arange(num_nodes, device=device)
                embeddings = embedding_layer(indices).unsqueeze(
                    0
                )  # Add batch dimension
                node_embeddings[node_type] = embeddings
                node_masks[node_type] = torch.ones(
                    1, num_nodes, dtype=torch.bool, device=device
                )
                node_ranges[node_type] = [(0, num_nodes)]

        # Process edge types and create adjacency matrices
        edge_adjacencies = {}

        # Gene multigraph edges
        gene_edge_types = [
            ("gene", "physical_interaction", "gene"),
            ("gene", "regulatory_interaction", "gene"),
        ]

        # Create combined adjacency matrix for gene multigraph
        gene_size = node_embeddings["gene"].size(1)
        gene_multigraph_adj = torch.zeros(
            batch_size, gene_size, gene_size, device=device
        )

        for edge_type in gene_edge_types:
            if edge_type in data.edge_types:
                edge_index = data[edge_type].edge_index

                if hasattr(data["gene"], "batch"):
                    # Handle batched data
                    gene_batch = data["gene"].batch
                    edge_batch = gene_batch[edge_index[0]]

                    for b in range(batch_size):
                        # Get edges in this batch
                        batch_mask = edge_batch == b
                        batch_edges = edge_index[:, batch_mask]

                        # Adjust indices to be batch-local
                        start_idx, _ = node_ranges["gene"][b]
                        batch_edges = batch_edges - start_idx

                        # Add to adjacency matrix
                        gene_multigraph_adj[b, batch_edges[0], batch_edges[1]] = 1.0
                else:
                    # Single graph case
                    gene_multigraph_adj[0, edge_index[0], edge_index[1]] = 1.0

        # Store the gene multigraph adjacency
        edge_adjacencies["gene_multigraph"] = gene_multigraph_adj

        # Process gene-reaction bipartite connections
        if ("gene", "gpr", "reaction") in data.edge_types:
            hyperedge_index = data["gene", "gpr", "reaction"].hyperedge_index

            # Create adjacency matrices
            gene_size = node_embeddings["gene"].size(1)
            reaction_size = node_embeddings["reaction"].size(1)

            # Gene -> Reaction adjacency
            gene_to_reaction_adj = torch.zeros(
                batch_size, reaction_size, gene_size, device=device
            )

            # Reaction -> Gene adjacency
            reaction_to_gene_adj = torch.zeros(
                batch_size, gene_size, reaction_size, device=device
            )

            if hasattr(data["gene"], "batch"):
                # Handle batched data
                gene_batch = data["gene"].batch
                reaction_batch = data["reaction"].batch

                # Assuming the hyperedge indices match the batching
                for b in range(batch_size):
                    gene_start, gene_end = node_ranges["gene"][b]
                    reaction_start, reaction_end = node_ranges["reaction"][b]

                    # Find edges for this batch
                    gene_mask = (hyperedge_index[0] >= gene_start) & (
                        hyperedge_index[0] < gene_end
                    )
                    reaction_mask = (hyperedge_index[1] >= reaction_start) & (
                        hyperedge_index[1] < reaction_end
                    )
                    edge_mask = gene_mask & reaction_mask

                    batch_edges = hyperedge_index[:, edge_mask]

                    # Adjust indices to be batch-local
                    batch_edges[0] = batch_edges[0] - gene_start
                    batch_edges[1] = batch_edges[1] - reaction_start

                    # Fill adjacency matrices
                    gene_to_reaction_adj[b, batch_edges[1], batch_edges[0]] = 1.0
                    reaction_to_gene_adj[b, batch_edges[0], batch_edges[1]] = 1.0
            else:
                # Single graph case
                gene_to_reaction_adj[0, hyperedge_index[1], hyperedge_index[0]] = 1.0
                reaction_to_gene_adj[0, hyperedge_index[0], hyperedge_index[1]] = 1.0

            # Store the bipartite adjacencies
            edge_adjacencies["gene_to_reaction"] = gene_to_reaction_adj
            edge_adjacencies["reaction_to_gene"] = reaction_to_gene_adj

        # Process metabolite-reaction-metabolite hypergraph
        if ("metabolite", "reaction", "metabolite") in data.edge_types:
            hyperedge_index = data[
                "metabolite", "reaction", "metabolite"
            ].hyperedge_index
            stoichiometry = (
                data["metabolite", "reaction", "metabolite"].stoichiometry
                if hasattr(
                    data["metabolite", "reaction", "metabolite"], "stoichiometry"
                )
                else None
            )

            # Create adjacency matrices
            metabolite_size = node_embeddings["metabolite"].size(1)
            reaction_size = node_embeddings["reaction"].size(1)

            # Metabolite -> Reaction adjacency
            metabolite_to_reaction_adj = torch.zeros(
                batch_size, reaction_size, metabolite_size, device=device
            )

            # Reaction -> Metabolite adjacency
            reaction_to_metabolite_adj = torch.zeros(
                batch_size, metabolite_size, reaction_size, device=device
            )

            # Stoichiometry values matrix
            stoich_matrix = None
            if stoichiometry is not None:
                stoich_matrix = torch.zeros(
                    batch_size, metabolite_size, reaction_size, device=device
                )

            if hasattr(data["metabolite"], "batch"):
                # Handle batched data
                metabolite_batch = data["metabolite"].batch
                reaction_batch = data["reaction"].batch

                # Assuming the hyperedge indices match the batching
                for b in range(batch_size):
                    metab_start, metab_end = node_ranges["metabolite"][b]
                    reaction_start, reaction_end = node_ranges["reaction"][b]

                    # Find edges for this batch
                    metab_mask = (hyperedge_index[0] >= metab_start) & (
                        hyperedge_index[0] < metab_end
                    )
                    reaction_mask = (hyperedge_index[1] >= reaction_start) & (
                        hyperedge_index[1] < reaction_end
                    )
                    edge_mask = metab_mask & reaction_mask

                    batch_edges = hyperedge_index[:, edge_mask]

                    # Get stoichiometry values for this batch if available
                    batch_stoich = None
                    if stoichiometry is not None:
                        batch_stoich = stoichiometry[edge_mask]

                    # Adjust indices to be batch-local
                    batch_edges[0] = batch_edges[0] - metab_start
                    batch_edges[1] = batch_edges[1] - reaction_start

                    # Fill adjacency matrices
                    metabolite_to_reaction_adj[b, batch_edges[1], batch_edges[0]] = 1.0
                    reaction_to_metabolite_adj[b, batch_edges[0], batch_edges[1]] = 1.0

                    # Fill stoichiometry matrix if available
                    if batch_stoich is not None:
                        stoich_matrix[b, batch_edges[0], batch_edges[1]] = batch_stoich
            else:
                # Single graph case
                metabolite_to_reaction_adj[
                    0, hyperedge_index[1], hyperedge_index[0]
                ] = 1.0
                reaction_to_metabolite_adj[
                    0, hyperedge_index[0], hyperedge_index[1]
                ] = 1.0

                # Fill stoichiometry matrix if available
                if stoichiometry is not None:
                    stoich_matrix[0, hyperedge_index[0], hyperedge_index[1]] = (
                        stoichiometry
                    )

            # Store the hypergraph adjacencies and stoichiometry
            edge_adjacencies["metabolite_to_reaction"] = metabolite_to_reaction_adj
            edge_adjacencies["reaction_to_metabolite"] = reaction_to_metabolite_adj
            if stoich_matrix is not None:
                edge_adjacencies["metabolite_reaction_stoich"] = stoich_matrix

        # In the forward method of CellGraphHeteroNSA
        for block_idx, block in enumerate(self.blocks):
            print(f"Processing block {block_idx}: {block.block_type}")
            block_type = block.block_type
            content = block.get_content()
            
            if block_type == "SAB":
                # Self-attention for each node type
                for node_type, sab in content.items():
                    if node_type in node_embeddings:
                        # Apply self-attention
                        node_embeddings[node_type] = sab(node_embeddings[node_type])

            elif block_type == "M_GENE":
                # Masked attention for gene multigraph
                mab = content
                if "gene" in node_embeddings and "gene_multigraph" in edge_adjacencies:
                    node_embeddings["gene"] = mab(
                        node_embeddings["gene"], edge_adjacencies["gene_multigraph"]
                    )

            elif block_type == "M_GPR":
                # Masked attention for gene-reaction bipartite
                gene_to_reaction_mab, reaction_to_gene_mab = content

                if (
                    "gene" in node_embeddings
                    and "reaction" in node_embeddings
                    and "gene_to_reaction" in edge_adjacencies
                    and "reaction_to_gene" in edge_adjacencies
                ):
                    # Update reaction nodes based on gene nodes
                    node_embeddings["reaction"] = gene_to_reaction_mab(
                        node_embeddings["reaction"],
                        edge_adjacencies["gene_to_reaction"],
                    )

                    # Update gene nodes based on reaction nodes
                    node_embeddings["gene"] = reaction_to_gene_mab(
                        node_embeddings["gene"], edge_adjacencies["reaction_to_gene"]
                    )

            elif block_type == "M_MRM":
                # Masked attention for metabolite-reaction-metabolite hypergraph
                metab_to_reaction_mab, reaction_to_metab_mab = content

                if (
                    "metabolite" in node_embeddings
                    and "reaction" in node_embeddings
                    and "metabolite_to_reaction" in edge_adjacencies
                    and "reaction_to_metabolite" in edge_adjacencies
                ):
                    # Get stoichiometry if available
                    stoich_matrix = edge_adjacencies.get(
                        "metabolite_reaction_stoich", None
                    )

                    # Update reaction nodes based on metabolite nodes
                    node_embeddings["reaction"] = metab_to_reaction_mab(
                        node_embeddings["reaction"],
                        edge_adjacencies["metabolite_to_reaction"],
                        (
                            stoich_matrix.transpose(1, 2)
                            if stoich_matrix is not None
                            else None
                        ),
                    )

                    # Update metabolite nodes based on reaction nodes
                    node_embeddings["metabolite"] = reaction_to_metab_mab(
                        node_embeddings["metabolite"],
                        edge_adjacencies["reaction_to_metabolite"],
                        stoich_matrix if stoich_matrix is not None else None,
                    )
            else:
                raise ValueError(f"Unknown block type: {block_type}")

        # Return the processed node embeddings
        return node_embeddings


class CellGraphNSAModel(nn.Module):
    """
    Full model for cell graph prediction using Node-Set Attention.

    Args:
        node_counts: Dictionary with counts of each node type
        hidden_dim: Hidden dimension for embeddings and attention layers
        pattern: Pattern of attention blocks to apply
        num_heads: Number of attention heads
        output_dim: Dimension of the output (prediction)
    """

    def __init__(
        self,
        node_counts: Dict[str, int],
        hidden_dim: int,
        pattern: List[str],
        num_heads: int = 8,
        output_dim: int = 1,
    ):
        super().__init__()

        # Node-Set Attention encoder
        self.nsa_encoder = CellGraphHeteroNSA(
            node_counts=node_counts,
            hidden_dim=hidden_dim,
            pattern=pattern,
            num_heads=num_heads,
        )

        # Readout layers for gene predictions
        self.gene_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: HeteroData):
        # Get batch size
        batch_size = 1
        if hasattr(data["gene"], "ptr"):
            batch_size = len(data["gene"].ptr) - 1

        # Process through NSA encoder
        node_embeddings = self.nsa_encoder(data, batch_size)

        # Get perturbation-specific predictions
        if hasattr(data["gene"], "ids_pert") and hasattr(
            data["gene"], "cell_graph_idx_pert"
        ):
            # Get perturbed gene IDs and their batch indices
            pert_ids = data["gene"].ids_pert
            batch_indices = data["gene"].cell_graph_idx_pert

            # List to store predictions for each batch
            batch_preds = []

            # For each batch element
            for b in range(batch_size):
                # Get perturbation IDs for this batch
                batch_mask = batch_indices == b
                if batch_mask.sum() == 0:
                    # No perturbations in this batch
                    continue

                batch_pert_ids = pert_ids[batch_mask]

                # Get the embeddings for these perturbed genes
                # Convert global IDs to batch-local indices
                gene_start = data["gene"].ptr[b]
                batch_local_ids = batch_pert_ids - gene_start

                # Get the embeddings
                pert_embeddings = node_embeddings["gene"][b, batch_local_ids]

                # Generate predictions
                batch_pred = self.gene_readout(pert_embeddings)
                batch_preds.append(batch_pred)

            # Combine predictions
            if batch_preds:
                predictions = torch.cat(batch_preds, dim=0)
            else:
                # Fallback if no perturbations
                predictions = torch.zeros(
                    0,
                    self.gene_readout[-1].out_features,
                    device=next(self.parameters()).device,
                )
        else:
            # Default to using all gene embeddings
            gene_embeddings = node_embeddings["gene"]
            predictions = self.gene_readout(gene_embeddings)

        return predictions


def main():
    # Import necessary libraries for flex_attention if running on CUDA
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask

        flex_attention_available = True
    except ImportError:
        print("FlexAttention not available, will use manual CPU implementation")
        flex_attention_available = False

    import torch.optim as optim
    from tqdm import tqdm
    import time

    print("Loading sample data batch...")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch()
    print(f"Loaded data: input channels={input_channels}, max nodes={max_num_nodes}")

    # Define model parameters
    node_counts = {
        "gene": max_num_nodes,
        "metabolite": dataset.cell_graph["metabolite"].num_nodes,
        "reaction": dataset.cell_graph["reaction"].num_nodes,
    }
    hidden_dim = 128
    pattern = [
        "S",  # Self-attention for all node types
        "M_GENE",  # Masked attention on gene multigraph
        "M_GPR",  # Gene-reaction bipartite
        "S",  # Self-attention again
    ]
    num_heads = 8
    output_dim = 1  # For fitness prediction

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    print("Initializing model...")
    model = CellGraphNSAModel(
        node_counts=node_counts,
        hidden_dim=hidden_dim,
        pattern=pattern,
        num_heads=num_heads,
        output_dim=output_dim,
    ).to(device)

    # Move batch to device
    batch = batch.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define loss function
    loss_fn = torch.nn.MSELoss()

    # Training loop
    print("\nStarting training loop...")
    model.train()

    # Forward pass
    start_time = time.time()
    print("Forward pass...")
    predictions = model(batch)
    forward_time = time.time() - start_time
    print(f"Forward pass complete in {forward_time:.2f} seconds")

    # Get target (using fitness from batch)
    if hasattr(batch["gene"], "fitness"):
        target = batch["gene"].fitness.float()
        print(f"Target shape: {target.shape}")
        print(f"Predictions shape: {predictions.shape}")

        # If shapes don't match, adjust
        if predictions.shape != target.shape:
            if len(target.shape) == 1 and len(predictions.shape) == 2:
                if predictions.shape[0] == target.shape[0]:
                    # Squeeze last dimension of predictions
                    predictions = predictions.squeeze(-1)
                    print(f"Adjusted predictions shape: {predictions.shape}")

        # Compute loss
        print("Computing loss...")
        loss = loss_fn(predictions, target)
        print(f"Loss: {loss.item():.6f}")

        # Backward pass
        print("Backward pass...")
        start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        backward_time = time.time() - start_time
        print(f"Backward pass complete in {backward_time:.2f} seconds")

        # Optimizer step
        optimizer.step()

        # Check gradients
        total_grad = 0.0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad += grad_norm
                param_count += 1
                print(f"Grad norm for {name}: {grad_norm:.6f}")

        if param_count > 0:
            avg_grad = total_grad / param_count
            print(f"Average gradient norm: {avg_grad:.6f}")

        print("\nBackpropagation test completed successfully!")
    else:
        print("No 'fitness' attribute found in batch['gene']. Can't compute loss.")


if __name__ == "__main__":
    main()
