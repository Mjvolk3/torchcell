# torchcell/data/graph_processor
# [[torchcell.data.graph_processor]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/graph_processor
# Test file: tests/torchcell/data/test_graph_processor.py


from abc import ABC, abstractmethod
from typing import Any, Dict
from torchcell.datamodels import PhenotypeType, ExperimentReferenceType, ExperimentType
import torch
from torch_geometric.utils._subgraph import bipartite_subgraph, subgraph
from torch_geometric.utils import k_hop_subgraph
from torch_scatter import scatter
from torchcell.data.hetero_data import HeteroData
from torchcell.datamodels import ExperimentReferenceType, ExperimentType, PhenotypeType
from torchcell.profiling.timing import time_method


class GraphProcessor(ABC):
    @abstractmethod
    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: (
            dict[str, ExperimentType | ExperimentReferenceType]
            | list[dict[str, ExperimentType | ExperimentReferenceType]]
        ),
    ) -> HeteroData:
        pass


class SubgraphRepresentation(GraphProcessor):
    def __init__(self) -> None:
        super().__init__()
        # Always use CPU for pin_memory compatibility
        self.device: torch.device = torch.device("cpu")
        self.masks: Dict[str, Dict[str, torch.Tensor]] = {}

    @time_method
    def _initialize_masks(self, cell_graph: HeteroData) -> None:
        # Initialize masks dictionary
        self.masks = {
            "gene": {
                "kept": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
                "perturbed": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
            }
        }

        # Only create masks for reaction and metabolite if they exist in the cell graph
        if "reaction" in cell_graph.node_types and hasattr(
            cell_graph["reaction"], "num_nodes"
        ):
            self.masks["reaction"] = {
                "kept": torch.zeros(
                    cell_graph["reaction"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "removed": torch.zeros(
                    cell_graph["reaction"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
            }

        if "metabolite" in cell_graph.node_types and hasattr(
            cell_graph["metabolite"], "num_nodes"
        ):
            self.masks["metabolite"] = {
                "kept": torch.ones(
                    cell_graph["metabolite"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "removed": torch.zeros(
                    cell_graph["metabolite"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
            }

    @time_method
    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[Any],
        data: list[Dict[str, Any]],
    ) -> HeteroData:
        # Always use CPU for pin_memory compatibility
        # The model will handle moving tensors to GPU after DataLoader
        self.device = torch.device("cpu")
        self._initialize_masks(cell_graph)
        integrated_subgraph = HeteroData()

        # Process gene-level information
        gene_info = self._process_gene_info(cell_graph, data)
        self._add_gene_data(integrated_subgraph, cell_graph, gene_info)
        self._process_gene_interactions(integrated_subgraph, cell_graph, gene_info)

        # Process reaction-level (GPR) information only if reaction nodes exist
        if "reaction" in cell_graph.node_types and hasattr(
            cell_graph["reaction"], "num_nodes"
        ):
            reaction_info = self._process_reaction_info(
                cell_graph, gene_info, integrated_subgraph
            )
            self._add_reaction_data(integrated_subgraph, reaction_info, cell_graph)

            # Process metabolic network only if metabolite nodes also exist
            if "metabolite" in cell_graph.node_types and hasattr(
                cell_graph["metabolite"], "num_nodes"
            ):
                self._process_metabolic_network(
                    integrated_subgraph, cell_graph, reaction_info
                )
        else:
            reaction_info = None

        # Add phenotype data to the gene nodes
        self._add_phenotype_data(integrated_subgraph, phenotype_info, data)

        # Attach the masks to the output graph
        integrated_subgraph["gene"].pert_mask = self.masks["gene"]["perturbed"]

        # Only add reaction and metabolite masks if they exist
        if "reaction" in self.masks:
            integrated_subgraph["reaction"].pert_mask = self.masks["reaction"][
                "removed"
            ]

        if "metabolite" in self.masks:
            integrated_subgraph["metabolite"].pert_mask = self.masks["metabolite"][
                "removed"
            ]

        return integrated_subgraph

    @time_method
    def _process_gene_info(
        self, cell_graph: HeteroData, data: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        perturbed_names = {
            p.systematic_gene_name
            for item in data
            for p in item["experiment"].genotype.perturbations
        }
        node_ids = cell_graph["gene"].node_ids
        keep_idx = [i for i, name in enumerate(node_ids) if name not in perturbed_names]
        remove_idx = [i for i, name in enumerate(node_ids) if name in perturbed_names]

        self.masks["gene"]["kept"][keep_idx] = True
        self.masks["gene"]["perturbed"][remove_idx] = True

        return {
            "perturbed_names": perturbed_names,
            "keep_subset": torch.tensor(keep_idx, dtype=torch.long, device=self.device),
            "remove_subset": torch.tensor(
                remove_idx, dtype=torch.long, device=self.device
            ),
            "keep_node_ids": [node_ids[i] for i in keep_idx],
        }

    def _add_gene_data(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
    ) -> None:
        integrated_subgraph["gene"].node_ids = gene_info["keep_node_ids"]
        integrated_subgraph["gene"].num_nodes = len(gene_info["keep_node_ids"])
        integrated_subgraph["gene"].ids_pert = list(gene_info["perturbed_names"])
        integrated_subgraph["gene"].perturbation_indices = gene_info["remove_subset"]

        x_full = cell_graph["gene"].x
        integrated_subgraph["gene"].x = x_full[gene_info["keep_subset"]]
        integrated_subgraph["gene"].x_pert = x_full[gene_info["remove_subset"]]

    @time_method
    def _process_gene_interactions(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
    ) -> None:
        # Process all gene-to-gene edge types
        for et in cell_graph.edge_types:
            if et[0] == "gene" and et[2] == "gene":
                orig_edge_index = cell_graph[et].edge_index
                edge_index, _, edge_mask = subgraph(
                    subset=gene_info["keep_subset"],
                    edge_index=orig_edge_index,
                    relabel_nodes=True,
                    num_nodes=cell_graph["gene"].num_nodes,
                    return_edge_mask=True,
                )
                integrated_subgraph[et].edge_index = edge_index
                integrated_subgraph[et].num_edges = edge_index.size(1)
                integrated_subgraph[et].pert_mask = ~edge_mask

    @time_method
    def _process_reaction_info(
        self,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
        integrated_subgraph: HeteroData,
    ) -> Dict[str, Any]:
        max_reaction_idx = cell_graph["reaction"].num_nodes

        # Create Growth subsystem indicator first (before any conditional returns)
        w_growth = torch.zeros(max_reaction_idx, dtype=torch.float, device=self.device)

        # Check if subsystem attribute exists
        if hasattr(cell_graph["reaction"], "subsystem"):
            # Check data type and structure
            subsystems = cell_graph["reaction"].subsystem

            # Populate the w_growth tensor
            for i in range(max_reaction_idx):
                if isinstance(subsystems, list):
                    if i < len(subsystems) and subsystems[i] == "Growth":
                        w_growth[i] = 1.0
                elif torch.is_tensor(subsystems):
                    if subsystems[i] == "Growth":
                        w_growth[i] = 1.0
                elif hasattr(subsystems, "__getitem__"):
                    if subsystems[i] == "Growth":
                        w_growth[i] = 1.0

        # If no GPR relationship exists, assume all reactions are valid.
        if ("gene", "gpr", "reaction") not in cell_graph.edge_types:
            valid_reactions = torch.arange(max_reaction_idx, device=self.device)
            self.masks["reaction"]["kept"].fill_(True)
            self.masks["reaction"]["removed"].fill_(False)

            gene_map = torch.full(
                (cell_graph["gene"].num_nodes,),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            gene_map[gene_info["keep_subset"]] = torch.arange(
                len(gene_info["keep_subset"]), device=self.device
            )
            reaction_map = torch.arange(max_reaction_idx, device=self.device)

            # Include w_growth in the return dictionary
            return {
                "valid_reactions": valid_reactions,
                "gene_map": gene_map.tolist(),
                "reaction_map": reaction_map.tolist(),
                "w_growth": w_growth,
            }

        # Process gene–reaction (GPR) edges
        gpr_edge_index = cell_graph["gene", "gpr", "reaction"].hyperedge_index.to(
            self.device
        )
        gene_mask = torch.zeros(
            cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
        )
        gene_mask[gene_info["keep_subset"]] = True

        gene_indices = gpr_edge_index[0]
        reaction_indices = gpr_edge_index[1]
        max_gene_idx = cell_graph["gene"].num_nodes

        valid_mask = (gene_indices < max_gene_idx) & (
            reaction_indices < max_reaction_idx
        )
        gene_indices = gene_indices[valid_mask]
        reaction_indices = reaction_indices[valid_mask]

        has_genes = torch.zeros(max_reaction_idx, dtype=torch.bool, device=self.device)
        has_genes[reaction_indices.unique()] = True

        reaction_gene_sum = scatter(
            gene_mask[gene_indices].float(),
            reaction_indices,
            dim=0,
            dim_size=max_reaction_idx,
            reduce="sum",
        )
        total_gene_count = scatter(
            torch.ones_like(gene_indices, dtype=torch.float),
            reaction_indices,
            dim=0,
            dim_size=max_reaction_idx,
            reduce="sum",
        )
        valid_with_genes_mask = (reaction_gene_sum == total_gene_count) & has_genes
        valid_without_genes_mask = ~has_genes
        valid_mask_combined = valid_with_genes_mask | valid_without_genes_mask
        valid_reactions = torch.nonzero(valid_mask_combined).squeeze(-1)

        self.masks["reaction"]["kept"] = valid_mask_combined
        self.masks["reaction"]["removed"] = ~valid_mask_combined

        edge_mask = torch.isin(
            reaction_indices, torch.where(valid_with_genes_mask)[0]
        ) & torch.isin(gene_indices, gene_info["keep_subset"])
        new_gpr_edge_index = gpr_edge_index[:, edge_mask].clone()

        gene_map = torch.full(
            (cell_graph["gene"].num_nodes,), -1, dtype=torch.long, device=self.device
        )
        gene_map[gene_info["keep_subset"]] = torch.arange(
            len(gene_info["keep_subset"]), device=self.device
        )
        reaction_map = torch.full(
            (max_reaction_idx,), -1, dtype=torch.long, device=self.device
        )
        reaction_map[valid_reactions] = torch.arange(
            len(valid_reactions), device=self.device
        )

        new_gpr_edge_index[0] = gene_map[new_gpr_edge_index[0]]
        new_gpr_edge_index[1] = reaction_map[new_gpr_edge_index[1]]

        integrated_subgraph["gene", "gpr", "reaction"].hyperedge_index = (
            new_gpr_edge_index
        )
        integrated_subgraph["gene", "gpr", "reaction"].num_edges = (
            new_gpr_edge_index.size(1)
        )
        integrated_subgraph["gene", "gpr", "reaction"].pert_mask = ~edge_mask

        # Include w_growth in the return
        return {
            "valid_reactions": valid_reactions,
            "gene_map": gene_map.tolist(),
            "reaction_map": reaction_map.tolist(),
            "w_growth": w_growth,
        }

    def _add_reaction_data(
        self,
        integrated_subgraph: HeteroData,
        reaction_info: Dict[str, Any],
        cell_graph: HeteroData,
    ) -> None:
        if not reaction_info:
            return
        valid_reactions = reaction_info["valid_reactions"]
        integrated_subgraph["reaction"].num_nodes = len(valid_reactions)
        integrated_subgraph["reaction"].node_ids = valid_reactions.tolist()

        # Subset w_growth to valid reactions if it exists in cell_graph
        if hasattr(cell_graph["reaction"], "w_growth"):
            w_growth = cell_graph["reaction"].w_growth
            integrated_subgraph["reaction"].w_growth = w_growth[valid_reactions]

    def _process_metabolic_network(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        reaction_info: Dict[str, Any],
    ) -> None:
        # Only process if bipartite representation exists
        if not reaction_info or "reaction" not in cell_graph.node_types:
            return
        if ("reaction", "rmr", "metabolite") in cell_graph.edge_types:
            self._process_metabolism_bipartite(
                integrated_subgraph, cell_graph, reaction_info
            )

    @time_method
    def _process_metabolism_bipartite(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        reaction_info: Dict[str, Any],
    ) -> None:
        valid_reactions = reaction_info["valid_reactions"]
        rmr_edges = cell_graph["reaction", "rmr", "metabolite"]
        hyperedge_index = rmr_edges.hyperedge_index.to(self.device)
        stoichiometry = rmr_edges.stoichiometry.to(self.device)

        # Fast path optimization: Since we keep all metabolites (100% of production cases),
        # avoid expensive bipartite_subgraph() gather/scatter operations.
        # Instead, use direct boolean masking and reaction index remapping.
        num_reactions = cell_graph["reaction"].num_nodes

        # Create reaction mapping for O(1) lookup
        reaction_map = torch.full(
            (num_reactions,), -1, dtype=torch.long, device=self.device
        )
        reaction_map[valid_reactions] = torch.arange(
            len(valid_reactions), dtype=torch.long, device=self.device
        )

        # Filter edges: keep only edges from valid reactions
        reaction_indices = hyperedge_index[0]
        metabolite_indices = hyperedge_index[1]
        edge_mask = reaction_map[reaction_indices] != -1

        # Apply filter and remap reaction indices (metabolite indices unchanged)
        final_edge_index = torch.stack([
            reaction_map[reaction_indices[edge_mask]],
            metabolite_indices[edge_mask]
        ], dim=0)
        final_edge_attr = stoichiometry[edge_mask]

        edge_type = ("reaction", "rmr", "metabolite")
        integrated_subgraph[edge_type].hyperedge_index = final_edge_index
        integrated_subgraph[edge_type].stoichiometry = final_edge_attr
        integrated_subgraph[edge_type].num_edges = final_edge_index.size(1)

        integrated_subgraph["metabolite"].node_ids = cell_graph["metabolite"].node_ids
        integrated_subgraph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes
        self.masks["metabolite"]["kept"].fill_(True)
        self.masks["metabolite"]["removed"].fill_(False)

    # TODO Add option for including non-coo version.
    # def _add_phenotype_data(
    #     self,
    #     integrated_subgraph: HeteroData,
    #     phenotype_info: list[PhenotypeType],
    #     data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    # ) -> None:
    #     phenotype_fields = []
    #     for phenotype in phenotype_info:
    #         phenotype_fields.extend(
    #             [
    #                 phenotype.model_fields["label_name"].default,
    #                 phenotype.model_fields["label_statistic_name"].default,
    #             ]
    #         )
    #     for field in phenotype_fields:
    #         field_values = []
    #         for item in data:
    #             value = getattr(item["experiment"].phenotype, field, None)
    #             if value is not None:
    #                 field_values.append(value)
    #         integrated_subgraph["gene"][field] = torch.tensor(
    #             field_values if field_values else [float("nan")],
    #             dtype=torch.float,
    #             device=self.device,
    #         )

    def _add_phenotype_data(
        self,
        integrated_subgraph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> None:
        """
        Add phenotype data to the graph in COO format.
        Optimized version that ensures all tensors are on the same device.
        """
        # Always use CPU for pin_memory compatibility
        if self.device is None:
            self.device = torch.device("cpu")

        # Initialize storage for phenotype values and metadata
        all_values = []
        all_type_indices = []
        all_sample_indices = []  # Which sample each value belongs to
        phenotype_types = []

        # Initialize storage for statistic values
        all_stat_values = []
        all_stat_type_indices = []
        all_stat_sample_indices = []  # Which sample each stat value belongs to
        stat_types = []

        # Extract phenotype type information
        for phenotype_class in phenotype_info:
            label_name = phenotype_class.model_fields["label_name"].default
            stat_name = phenotype_class.model_fields["label_statistic_name"].default

            phenotype_types.append(label_name)
            if stat_name:
                stat_types.append(stat_name)

        # Process each experimental data point
        for item_idx, item in enumerate(data):
            # Get phenotype object from experiment
            phenotype = item["experiment"].phenotype

            # Process each phenotype type
            for type_idx, field_name in enumerate(phenotype_types):
                value = getattr(phenotype, field_name, None)
                if value is not None:
                    # Convert single values to lists for consistent handling
                    values = [value] if not isinstance(value, (list, tuple)) else value

                    # Add all values with their type indices and sample indices
                    all_values.extend(values)
                    all_type_indices.extend([type_idx] * len(values))
                    all_sample_indices.extend([item_idx] * len(values))

            # Process statistics in the same way
            for stat_type_idx, stat_field_name in enumerate(stat_types):
                stat_value = getattr(phenotype, stat_field_name, None)
                if stat_value is not None:
                    # Convert single values to lists for consistent handling
                    stat_values = (
                        [stat_value]
                        if not isinstance(stat_value, (list, tuple))
                        else stat_value
                    )

                    # Add all statistic values with their type indices and sample indices
                    all_stat_values.extend(stat_values)
                    all_stat_type_indices.extend([stat_type_idx] * len(stat_values))
                    all_stat_sample_indices.extend([item_idx] * len(stat_values))

        # Store phenotype data in the graph - create all tensors directly on the target device
        if all_values:
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                all_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                all_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                all_sample_indices, dtype=torch.long, device=self.device
            )
        else:
            # Handle empty case with placeholder values - create directly on target device
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                [float("nan")], dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )

        # Store phenotype type names (non-tensor data)
        integrated_subgraph["gene"]["phenotype_types"] = phenotype_types

        # Store statistic data in the graph - create directly on target device
        if all_stat_values:
            integrated_subgraph["gene"]["phenotype_stat_values"] = torch.tensor(
                all_stat_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_type_indices"] = torch.tensor(
                all_stat_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_sample_indices"] = torch.tensor(
                all_stat_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_types"] = stat_types


class IncidenceSubgraphRepresentation(GraphProcessor):
    """
    Graph processor using precomputed node-to-edge incidence structures.

    Algorithm:
    - Precomputes which edges touch each node (incidence mapping) once at initialization
    - For each perturbation, directly looks up edges to remove via the incidence map
    - Achieves O(k×d) complexity for edge filtering where k=removed nodes, d=degree
    - Still performs node relabeling and returns standard subgraph format

    Complexity:
    - Precomputation: O(E) once per dataset
    - Per sample: O(k×d + E') where k<<N (small perturbations) and E'≈E
    - vs standard subgraph: O(N + E) per edge type per sample

    Memory:
    - Stores list of edge positions for each node (~O(E) total)
    - Minimal overhead compared to graph structure itself

    Output:
    - Identical format to SubgraphRepresentation (filtered, relabeled subgraphs)
    - Full backward compatibility with existing models
    """

    def __init__(self) -> None:
        super().__init__()
        # Always use CPU for pin_memory compatibility
        self.device: torch.device = torch.device("cpu")
        self.masks: Dict[str, Dict[str, torch.Tensor]] = {}
        self._edge_incidence_cache: Dict[Any, list[torch.Tensor]] | None = None

    @property
    def edge_incidence_cache(self) -> Dict[Any, list[torch.Tensor]]:
        """
        Lazily build and return edge incidence cache.

        Maps each node to tensor of edge positions where that node appears.
        Built once on first access, then cached for all subsequent calls.
        """
        if self._edge_incidence_cache is None:
            raise RuntimeError(
                "Incidence cache not built. Call _build_incidence_cache() first."
            )
        return self._edge_incidence_cache

    def build_cache(self, cell_graph: HeteroData) -> dict[str, Any]:
        """
        Explicitly build incidence cache and return timing information.

        This should be called once during dataset initialization or before
        benchmarking to separate one-time cache build cost from per-sample
        processing time.

        Returns:
            dict with cache build timing: {
                'total_time_ms': float,
                'num_edge_types': int,
                'total_edges': int
            }
        """
        import time

        if self._edge_incidence_cache is not None:
            return {'total_time_ms': 0.0, 'num_edge_types': 0, 'total_edges': 0}

        start = time.time()
        self._build_incidence_cache(cell_graph)
        elapsed = (time.time() - start) * 1000

        # Cache is guaranteed to be built now
        assert self._edge_incidence_cache is not None

        num_edge_types = len(self._edge_incidence_cache)
        total_edges = sum(
            len(edge_list)
            for node_to_edges in self._edge_incidence_cache.values()
            for edge_list in node_to_edges
        ) // 2  # Divide by 2 since each edge counted twice (src and dst)

        return {
            'total_time_ms': elapsed,
            'num_edge_types': num_edge_types,
            'total_edges': total_edges
        }

    def _build_incidence_cache(self, cell_graph: HeteroData) -> None:
        """
        Build node-to-edge incidence mappings for gene-gene edge types.

        For each gene node, stores tensor of edge positions (indices) where
        that gene appears as either source or destination.

        Complexity: O(E) per edge type, one-time cost
        Memory: O(E) total across all edge types
        """
        if self._edge_incidence_cache is not None:
            return  # Already built

        num_genes = cell_graph["gene"].num_nodes
        cache = {}

        for edge_type in cell_graph.edge_types:
            if edge_type[0] == "gene" and edge_type[2] == "gene":
                edge_index = cell_graph[edge_type].edge_index
                num_edges = edge_index.size(1)

                # Initialize empty lists for each gene
                node_to_edges = [[] for _ in range(num_genes)]

                # Build incidence mapping: node -> [edge_positions]
                for edge_pos in range(num_edges):
                    src = edge_index[0, edge_pos].item()
                    dst = edge_index[1, edge_pos].item()
                    node_to_edges[src].append(edge_pos)
                    if src != dst:  # Avoid duplicate for self-loops
                        node_to_edges[dst].append(edge_pos)

                # Convert lists to tensors for faster operations
                cache[edge_type] = [
                    torch.tensor(edges, dtype=torch.long, device=self.device)
                    if edges else torch.tensor([], dtype=torch.long, device=self.device)
                    for edges in node_to_edges
                ]

        self._edge_incidence_cache = cache

    @time_method
    def _initialize_masks(self, cell_graph: HeteroData) -> None:
        # Initialize masks dictionary
        self.masks = {
            "gene": {
                "kept": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
                "perturbed": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
            }
        }

        # Only create masks for reaction and metabolite if they exist in the cell graph
        if "reaction" in cell_graph.node_types and hasattr(
            cell_graph["reaction"], "num_nodes"
        ):
            self.masks["reaction"] = {
                "kept": torch.zeros(
                    cell_graph["reaction"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "removed": torch.zeros(
                    cell_graph["reaction"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
            }

        if "metabolite" in cell_graph.node_types and hasattr(
            cell_graph["metabolite"], "num_nodes"
        ):
            self.masks["metabolite"] = {
                "kept": torch.ones(
                    cell_graph["metabolite"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "removed": torch.zeros(
                    cell_graph["metabolite"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
            }

    @time_method
    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[Any],
        data: list[Dict[str, Any]],
    ) -> HeteroData:
        # Always use CPU for pin_memory compatibility
        # The model will handle moving tensors to GPU after DataLoader
        self.device = torch.device("cpu")
        self._initialize_masks(cell_graph)
        integrated_subgraph = HeteroData()

        # Process gene-level information
        gene_info = self._process_gene_info(cell_graph, data)
        self._add_gene_data(integrated_subgraph, cell_graph, gene_info)
        self._process_gene_interactions(integrated_subgraph, cell_graph, gene_info)

        # Process reaction-level (GPR) information only if reaction nodes exist
        if "reaction" in cell_graph.node_types and hasattr(
            cell_graph["reaction"], "num_nodes"
        ):
            reaction_info = self._process_reaction_info(
                cell_graph, gene_info, integrated_subgraph
            )
            self._add_reaction_data(integrated_subgraph, reaction_info, cell_graph)

            # Process metabolic network only if metabolite nodes also exist
            if "metabolite" in cell_graph.node_types and hasattr(
                cell_graph["metabolite"], "num_nodes"
            ):
                self._process_metabolic_network(
                    integrated_subgraph, cell_graph, reaction_info
                )
        else:
            reaction_info = None

        # Add phenotype data to the gene nodes
        self._add_phenotype_data(integrated_subgraph, phenotype_info, data)

        # Attach the masks to the output graph
        integrated_subgraph["gene"].pert_mask = self.masks["gene"]["perturbed"]

        # Only add reaction and metabolite masks if they exist
        if "reaction" in self.masks:
            integrated_subgraph["reaction"].pert_mask = self.masks["reaction"][
                "removed"
            ]

        if "metabolite" in self.masks:
            integrated_subgraph["metabolite"].pert_mask = self.masks["metabolite"][
                "removed"
            ]

        return integrated_subgraph

    @time_method
    def _process_gene_info(
        self, cell_graph: HeteroData, data: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        perturbed_names = {
            p.systematic_gene_name
            for item in data
            for p in item["experiment"].genotype.perturbations
        }
        node_ids = cell_graph["gene"].node_ids
        keep_idx = [i for i, name in enumerate(node_ids) if name not in perturbed_names]
        remove_idx = [i for i, name in enumerate(node_ids) if name in perturbed_names]

        self.masks["gene"]["kept"][keep_idx] = True
        self.masks["gene"]["perturbed"][remove_idx] = True

        return {
            "perturbed_names": perturbed_names,
            "keep_subset": torch.tensor(keep_idx, dtype=torch.long, device=self.device),
            "remove_subset": torch.tensor(
                remove_idx, dtype=torch.long, device=self.device
            ),
            "keep_node_ids": [node_ids[i] for i in keep_idx],
        }

    def _add_gene_data(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
    ) -> None:
        integrated_subgraph["gene"].node_ids = gene_info["keep_node_ids"]
        integrated_subgraph["gene"].num_nodes = len(gene_info["keep_node_ids"])
        integrated_subgraph["gene"].ids_pert = list(gene_info["perturbed_names"])
        integrated_subgraph["gene"].perturbation_indices = gene_info["remove_subset"]

        x_full = cell_graph["gene"].x
        integrated_subgraph["gene"].x = x_full[gene_info["keep_subset"]]
        integrated_subgraph["gene"].x_pert = x_full[gene_info["remove_subset"]]

    @time_method
    def _process_gene_interactions(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
    ) -> None:
        """
        Process gene-gene edge types using incidence-based filtering.

        Algorithm:
        1. Build incidence cache on first call (lazy initialization)
        2. Compute node relabeling mapping once for all edge types
        3. For each edge type:
           - Use incidence map to directly find edges touching removed nodes
           - Create edge mask in O(k×d) instead of O(E) scan
           - Filter and relabel edges

        Complexity: O(k×d + E') per edge type where:
        - k = number of removed nodes
        - d = average degree of removed nodes
        - E' = number of kept edges (for filtering/relabeling)
        """
        # Build cache on first use
        if self._edge_incidence_cache is None:
            self._build_incidence_cache(cell_graph)

        # Pre-compute gene mapping ONCE (not per edge type)
        num_genes = cell_graph["gene"].num_nodes
        gene_map = torch.full((num_genes,), -1, dtype=torch.long, device=self.device)
        gene_map[gene_info["keep_subset"]] = torch.arange(
            len(gene_info["keep_subset"]), device=self.device
        )

        # Use remove_subset directly (already a tensor)
        removed_nodes = gene_info["remove_subset"]

        # Process all gene-gene edge types
        for et in cell_graph.edge_types:
            if et[0] == "gene" and et[2] == "gene":
                edge_index = cell_graph[et].edge_index
                num_edges = edge_index.size(1)

                # Use incidence cache for O(k×d) edge lookup - pure tensor ops
                node_to_edges = self.edge_incidence_cache[et]

                # Gather edges to remove using tensor concatenation
                edges_to_remove_list = [node_to_edges[node_idx.item()] for node_idx in removed_nodes]

                # Create edge mask
                edge_mask = torch.ones(num_edges, dtype=torch.bool, device=self.device)
                if edges_to_remove_list and any(len(t) > 0 for t in edges_to_remove_list):
                    # Concatenate all edge tensors and remove duplicates
                    edges_to_remove = torch.cat([t for t in edges_to_remove_list if len(t) > 0])
                    edges_to_remove = edges_to_remove.unique()
                    edge_mask[edges_to_remove] = False

                # Filter and relabel using precomputed mapping
                kept_edges = edge_index[:, edge_mask]
                new_edge_index = torch.stack([
                    gene_map[kept_edges[0]],
                    gene_map[kept_edges[1]]
                ])

                # Store results (same format as SubgraphRepresentation)
                integrated_subgraph[et].edge_index = new_edge_index
                integrated_subgraph[et].num_edges = new_edge_index.size(1)
                integrated_subgraph[et].pert_mask = ~edge_mask

    @time_method
    def _process_reaction_info(
        self,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
        integrated_subgraph: HeteroData,
    ) -> Dict[str, Any]:
        max_reaction_idx = cell_graph["reaction"].num_nodes

        # Create Growth subsystem indicator first (before any conditional returns)
        w_growth = torch.zeros(max_reaction_idx, dtype=torch.float, device=self.device)

        # Check if subsystem attribute exists
        if hasattr(cell_graph["reaction"], "subsystem"):
            # Check data type and structure
            subsystems = cell_graph["reaction"].subsystem

            # Populate the w_growth tensor
            for i in range(max_reaction_idx):
                if isinstance(subsystems, list):
                    if i < len(subsystems) and subsystems[i] == "Growth":
                        w_growth[i] = 1.0
                elif torch.is_tensor(subsystems):
                    if subsystems[i] == "Growth":
                        w_growth[i] = 1.0
                elif hasattr(subsystems, "__getitem__"):
                    if subsystems[i] == "Growth":
                        w_growth[i] = 1.0

        # If no GPR relationship exists, assume all reactions are valid.
        if ("gene", "gpr", "reaction") not in cell_graph.edge_types:
            valid_reactions = torch.arange(max_reaction_idx, device=self.device)
            self.masks["reaction"]["kept"].fill_(True)
            self.masks["reaction"]["removed"].fill_(False)

            gene_map = torch.full(
                (cell_graph["gene"].num_nodes,),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            gene_map[gene_info["keep_subset"]] = torch.arange(
                len(gene_info["keep_subset"]), device=self.device
            )
            reaction_map = torch.arange(max_reaction_idx, device=self.device)

            # Include w_growth in the return dictionary
            return {
                "valid_reactions": valid_reactions,
                "gene_map": gene_map.tolist(),
                "reaction_map": reaction_map.tolist(),
                "w_growth": w_growth,
            }

        # Process gene–reaction (GPR) edges
        gpr_edge_index = cell_graph["gene", "gpr", "reaction"].hyperedge_index.to(
            self.device
        )
        gene_mask = torch.zeros(
            cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
        )
        gene_mask[gene_info["keep_subset"]] = True

        gene_indices = gpr_edge_index[0]
        reaction_indices = gpr_edge_index[1]
        max_gene_idx = cell_graph["gene"].num_nodes

        valid_mask = (gene_indices < max_gene_idx) & (
            reaction_indices < max_reaction_idx
        )
        gene_indices = gene_indices[valid_mask]
        reaction_indices = reaction_indices[valid_mask]

        has_genes = torch.zeros(max_reaction_idx, dtype=torch.bool, device=self.device)
        has_genes[reaction_indices.unique()] = True

        reaction_gene_sum = scatter(
            gene_mask[gene_indices].float(),
            reaction_indices,
            dim=0,
            dim_size=max_reaction_idx,
            reduce="sum",
        )
        total_gene_count = scatter(
            torch.ones_like(gene_indices, dtype=torch.float),
            reaction_indices,
            dim=0,
            dim_size=max_reaction_idx,
            reduce="sum",
        )
        valid_with_genes_mask = (reaction_gene_sum == total_gene_count) & has_genes
        valid_without_genes_mask = ~has_genes
        valid_mask_combined = valid_with_genes_mask | valid_without_genes_mask
        valid_reactions = torch.nonzero(valid_mask_combined).squeeze(-1)

        self.masks["reaction"]["kept"] = valid_mask_combined
        self.masks["reaction"]["removed"] = ~valid_mask_combined

        edge_mask = torch.isin(
            reaction_indices, torch.where(valid_with_genes_mask)[0]
        ) & torch.isin(gene_indices, gene_info["keep_subset"])
        new_gpr_edge_index = gpr_edge_index[:, edge_mask].clone()

        gene_map = torch.full(
            (cell_graph["gene"].num_nodes,), -1, dtype=torch.long, device=self.device
        )
        gene_map[gene_info["keep_subset"]] = torch.arange(
            len(gene_info["keep_subset"]), device=self.device
        )
        reaction_map = torch.full(
            (max_reaction_idx,), -1, dtype=torch.long, device=self.device
        )
        reaction_map[valid_reactions] = torch.arange(
            len(valid_reactions), device=self.device
        )

        new_gpr_edge_index[0] = gene_map[new_gpr_edge_index[0]]
        new_gpr_edge_index[1] = reaction_map[new_gpr_edge_index[1]]

        integrated_subgraph["gene", "gpr", "reaction"].hyperedge_index = (
            new_gpr_edge_index
        )
        integrated_subgraph["gene", "gpr", "reaction"].num_edges = (
            new_gpr_edge_index.size(1)
        )
        integrated_subgraph["gene", "gpr", "reaction"].pert_mask = ~edge_mask

        # Include w_growth in the return
        return {
            "valid_reactions": valid_reactions,
            "gene_map": gene_map.tolist(),
            "reaction_map": reaction_map.tolist(),
            "w_growth": w_growth,
        }

    def _add_reaction_data(
        self,
        integrated_subgraph: HeteroData,
        reaction_info: Dict[str, Any],
        cell_graph: HeteroData,
    ) -> None:
        if not reaction_info:
            return
        valid_reactions = reaction_info["valid_reactions"]
        integrated_subgraph["reaction"].num_nodes = len(valid_reactions)
        integrated_subgraph["reaction"].node_ids = valid_reactions.tolist()

        # Subset w_growth to valid reactions if it exists in cell_graph
        if hasattr(cell_graph["reaction"], "w_growth"):
            w_growth = cell_graph["reaction"].w_growth
            integrated_subgraph["reaction"].w_growth = w_growth[valid_reactions]

    def _process_metabolic_network(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        reaction_info: Dict[str, Any],
    ) -> None:
        # Only process if bipartite representation exists
        if not reaction_info or "reaction" not in cell_graph.node_types:
            return
        if ("reaction", "rmr", "metabolite") in cell_graph.edge_types:
            self._process_metabolism_bipartite(
                integrated_subgraph, cell_graph, reaction_info
            )

    @time_method
    def _process_metabolism_bipartite(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        reaction_info: Dict[str, Any],
    ) -> None:
        valid_reactions = reaction_info["valid_reactions"]
        rmr_edges = cell_graph["reaction", "rmr", "metabolite"]
        hyperedge_index = rmr_edges.hyperedge_index.to(self.device)
        stoichiometry = rmr_edges.stoichiometry.to(self.device)

        # Fast path optimization: Since we keep all metabolites (100% of production cases),
        # avoid expensive bipartite_subgraph() gather/scatter operations.
        # Instead, use direct boolean masking and reaction index remapping.
        num_reactions = cell_graph["reaction"].num_nodes

        # Create reaction mapping for O(1) lookup
        reaction_map = torch.full(
            (num_reactions,), -1, dtype=torch.long, device=self.device
        )
        reaction_map[valid_reactions] = torch.arange(
            len(valid_reactions), dtype=torch.long, device=self.device
        )

        # Filter edges: keep only edges from valid reactions
        reaction_indices = hyperedge_index[0]
        metabolite_indices = hyperedge_index[1]
        edge_mask = reaction_map[reaction_indices] != -1

        # Apply filter and remap reaction indices (metabolite indices unchanged)
        final_edge_index = torch.stack([
            reaction_map[reaction_indices[edge_mask]],
            metabolite_indices[edge_mask]
        ], dim=0)
        final_edge_attr = stoichiometry[edge_mask]

        edge_type = ("reaction", "rmr", "metabolite")
        integrated_subgraph[edge_type].hyperedge_index = final_edge_index
        integrated_subgraph[edge_type].stoichiometry = final_edge_attr
        integrated_subgraph[edge_type].num_edges = final_edge_index.size(1)

        integrated_subgraph["metabolite"].node_ids = cell_graph["metabolite"].node_ids
        integrated_subgraph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes
        self.masks["metabolite"]["kept"].fill_(True)
        self.masks["metabolite"]["removed"].fill_(False)

    def _add_phenotype_data(
        self,
        integrated_subgraph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> None:
        """
        Add phenotype data to the graph in COO format.
        Optimized version that ensures all tensors are on the same device.
        """
        # Always use CPU for pin_memory compatibility
        if self.device is None:
            self.device = torch.device("cpu")

        # Initialize storage for phenotype values and metadata
        all_values = []
        all_type_indices = []
        all_sample_indices = []  # Which sample each value belongs to
        phenotype_types = []

        # Initialize storage for statistic values
        all_stat_values = []
        all_stat_type_indices = []
        all_stat_sample_indices = []  # Which sample each stat value belongs to
        stat_types = []

        # Extract phenotype type information
        for phenotype_class in phenotype_info:
            label_name = phenotype_class.model_fields["label_name"].default
            stat_name = phenotype_class.model_fields["label_statistic_name"].default

            phenotype_types.append(label_name)
            if stat_name:
                stat_types.append(stat_name)

        # Process each experimental data point
        for item_idx, item in enumerate(data):
            # Get phenotype object from experiment
            phenotype = item["experiment"].phenotype

            # Process each phenotype type
            for type_idx, field_name in enumerate(phenotype_types):
                value = getattr(phenotype, field_name, None)
                if value is not None:
                    # Convert single values to lists for consistent handling
                    values = [value] if not isinstance(value, (list, tuple)) else value

                    # Add all values with their type indices and sample indices
                    all_values.extend(values)
                    all_type_indices.extend([type_idx] * len(values))
                    all_sample_indices.extend([item_idx] * len(values))

            # Process statistics in the same way
            for stat_type_idx, stat_field_name in enumerate(stat_types):
                stat_value = getattr(phenotype, stat_field_name, None)
                if stat_value is not None:
                    # Convert single values to lists for consistent handling
                    stat_values = (
                        [stat_value]
                        if not isinstance(stat_value, (list, tuple))
                        else stat_value
                    )

                    # Add all statistic values with their type indices and sample indices
                    all_stat_values.extend(stat_values)
                    all_stat_type_indices.extend([stat_type_idx] * len(stat_values))
                    all_stat_sample_indices.extend([item_idx] * len(stat_values))

        # Store phenotype data in the graph - create all tensors directly on the target device
        if all_values:
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                all_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                all_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                all_sample_indices, dtype=torch.long, device=self.device
            )
        else:
            # Handle empty case with placeholder values - create directly on target device
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                [float("nan")], dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )

        # Store phenotype type names (non-tensor data)
        integrated_subgraph["gene"]["phenotype_types"] = phenotype_types

        # Store statistic data in the graph - create directly on target device
        if all_stat_values:
            integrated_subgraph["gene"]["phenotype_stat_values"] = torch.tensor(
                all_stat_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_type_indices"] = torch.tensor(
                all_stat_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_sample_indices"] = torch.tensor(
                all_stat_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_types"] = stat_types


class LazySubgraphRepresentation(GraphProcessor):
    """
    Graph processor that returns full graph with masks instead of filtered subgraphs.

    Zero-copy approach for edges: References original edge_index tensors,
    only computes boolean masks. Uses incidence cache for O(k×d) edge mask
    computation instead of O(E) scanning.

    Output structure:
        Node data (same as SubgraphRepresentation):
            - node_ids: IDs of kept nodes (filtered)
            - num_nodes: count of kept nodes
            - ids_pert: IDs of perturbed nodes
            - perturbation_indices: indices of perturbed nodes in original graph
            - x: features of kept nodes (sliced from cell_graph)
            - x_pert: features of perturbed nodes (sliced from cell_graph)
            - pert_mask: boolean mask over ALL nodes, True for perturbed
            - mask: boolean mask over ALL nodes, True for kept (inverse of pert_mask)

        Edge data (ZERO-COPY - different from SubgraphRepresentation):
            - edge_index: FULL edge_index (reference to cell_graph, not filtered!)
            - num_edges: FULL edge count (not filtered count)
            - mask: boolean mask, True for edges to keep (both endpoints non-perturbed)

        Reaction/Metabolite data:
            - Same as SubgraphRepresentation

    Key difference from SubgraphRepresentation:
        - Edges are NOT filtered or relabeled
        - Edge indices refer to original node numbering
        - Model must apply edge masks during message passing

    Complexity:
        - Cache build: O(E) one-time per dataset
        - Per sample: O(k×d) where k=perturbed genes, d=average degree
        - vs SubgraphRepresentation: O(N+E) per sample

    Memory per sample:
        - Edge tensors: 0 bytes (references only)
        - Edge masks: ~2.7MB (9 edge types × 2.4M edges × 1 bit)
        - vs SubgraphRepresentation: ~170MB (tensor copies)
    """

    def __init__(self) -> None:
        super().__init__()
        # Always use CPU for pin_memory compatibility
        self.device: torch.device = torch.device("cpu")
        self.masks: Dict[str, Dict[str, torch.Tensor]] = {}
        self._edge_incidence_cache: Dict[Any, list[torch.Tensor]] | None = None

    @property
    def edge_incidence_cache(self) -> Dict[Any, list[torch.Tensor]]:
        """
        Lazily build and return edge incidence cache.

        Maps each node to tensor of edge positions where that node appears.
        Built once on first access, then cached for all subsequent calls.
        """
        if self._edge_incidence_cache is None:
            raise RuntimeError(
                "Incidence cache not built. Call _build_incidence_cache() first."
            )
        return self._edge_incidence_cache

    def build_cache(self, cell_graph: HeteroData) -> dict[str, Any]:
        """
        Explicitly build incidence cache and return timing information.

        This should be called once during dataset initialization or before
        benchmarking to separate one-time cache build cost from per-sample
        processing time.

        Returns:
            dict with cache build timing: {
                'total_time_ms': float,
                'num_edge_types': int,
                'total_edges': int
            }
        """
        import time

        if self._edge_incidence_cache is not None:
            return {'total_time_ms': 0.0, 'num_edge_types': 0, 'total_edges': 0}

        start = time.time()
        self._build_incidence_cache(cell_graph)
        elapsed = (time.time() - start) * 1000

        # Cache is guaranteed to be built now
        assert self._edge_incidence_cache is not None

        num_edge_types = len(self._edge_incidence_cache)
        total_edges = sum(
            len(edge_list)
            for node_to_edges in self._edge_incidence_cache.values()
            for edge_list in node_to_edges
        ) // 2  # Divide by 2 since each edge counted twice (src and dst)

        return {
            'total_time_ms': elapsed,
            'num_edge_types': num_edge_types,
            'total_edges': total_edges
        }

    def _build_incidence_cache(self, cell_graph: HeteroData) -> None:
        """
        Build node-to-edge incidence mappings for gene-gene edge types.

        For each gene node, stores tensor of edge positions (indices) where
        that gene appears as either source or destination.

        Complexity: O(E) per edge type, one-time cost
        Memory: O(E) total across all edge types
        """
        if self._edge_incidence_cache is not None:
            return  # Already built

        num_genes = cell_graph["gene"].num_nodes
        cache = {}

        for edge_type in cell_graph.edge_types:
            if edge_type[0] == "gene" and edge_type[2] == "gene":
                edge_index = cell_graph[edge_type].edge_index
                num_edges = edge_index.size(1)

                # Initialize empty lists for each gene
                node_to_edges = [[] for _ in range(num_genes)]

                # Build incidence mapping: node -> [edge_positions]
                for edge_pos in range(num_edges):
                    src = edge_index[0, edge_pos].item()
                    dst = edge_index[1, edge_pos].item()
                    node_to_edges[src].append(edge_pos)
                    if src != dst:  # Avoid duplicate for self-loops
                        node_to_edges[dst].append(edge_pos)

                # Convert lists to tensors for faster operations
                cache[edge_type] = [
                    torch.tensor(edges, dtype=torch.long, device=self.device)
                    if edges else torch.tensor([], dtype=torch.long, device=self.device)
                    for edges in node_to_edges
                ]

        self._edge_incidence_cache = cache

    @time_method
    def _initialize_masks(self, cell_graph: HeteroData) -> None:
        # Same as SubgraphRepresentation
        self.masks = {
            "gene": {
                "kept": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
                "perturbed": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
            }
        }

        if "reaction" in cell_graph.node_types and hasattr(
            cell_graph["reaction"], "num_nodes"
        ):
            self.masks["reaction"] = {
                "kept": torch.zeros(
                    cell_graph["reaction"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "removed": torch.zeros(
                    cell_graph["reaction"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
            }

        if "metabolite" in cell_graph.node_types and hasattr(
            cell_graph["metabolite"], "num_nodes"
        ):
            self.masks["metabolite"] = {
                "kept": torch.ones(
                    cell_graph["metabolite"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "removed": torch.zeros(
                    cell_graph["metabolite"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
            }

    @time_method
    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[Any],
        data: list[Dict[str, Any]],
    ) -> HeteroData:
        # Always use CPU for pin_memory compatibility
        self.device = torch.device("cpu")
        self._initialize_masks(cell_graph)
        integrated_subgraph = HeteroData()

        # Process gene-level information (same as SubgraphRepresentation)
        gene_info = self._process_gene_info(cell_graph, data)
        self._add_gene_data(integrated_subgraph, cell_graph, gene_info)

        # Process gene interactions (DIFFERENT - zero-copy with masks)
        self._process_gene_interactions(integrated_subgraph, cell_graph, gene_info)

        # Process reaction-level (GPR) information (same as SubgraphRepresentation)
        if "reaction" in cell_graph.node_types and hasattr(
            cell_graph["reaction"], "num_nodes"
        ):
            reaction_info = self._process_reaction_info(
                cell_graph, gene_info, integrated_subgraph
            )
            self._add_reaction_data(integrated_subgraph, reaction_info, cell_graph)

            if "metabolite" in cell_graph.node_types and hasattr(
                cell_graph["metabolite"], "num_nodes"
            ):
                self._process_metabolic_network(
                    integrated_subgraph, cell_graph, reaction_info
                )
        else:
            reaction_info = None

        # Add phenotype data (same as SubgraphRepresentation)
        self._add_phenotype_data(integrated_subgraph, phenotype_info, data)

        # Attach masks
        integrated_subgraph["gene"].pert_mask = self.masks["gene"]["perturbed"]
        integrated_subgraph["gene"].mask = ~self.masks["gene"]["perturbed"]

        if "reaction" in self.masks:
            integrated_subgraph["reaction"].pert_mask = self.masks["reaction"]["removed"]
            integrated_subgraph["reaction"].mask = self.masks["reaction"]["kept"]  # Phase 4.1: Add reaction.mask

        if "metabolite" in self.masks:
            integrated_subgraph["metabolite"].pert_mask = self.masks["metabolite"]["removed"]
            integrated_subgraph["metabolite"].mask = self.masks["metabolite"]["kept"]  # Phase 4.1: Add metabolite.mask

        return integrated_subgraph

    @time_method
    def _process_gene_info(
        self, cell_graph: HeteroData, data: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Same as SubgraphRepresentation
        perturbed_names = {
            p.systematic_gene_name
            for item in data
            for p in item["experiment"].genotype.perturbations
        }
        node_ids = cell_graph["gene"].node_ids
        keep_idx = [i for i, name in enumerate(node_ids) if name not in perturbed_names]
        remove_idx = [i for i, name in enumerate(node_ids) if name in perturbed_names]

        self.masks["gene"]["kept"][keep_idx] = True
        self.masks["gene"]["perturbed"][remove_idx] = True

        return {
            "perturbed_names": perturbed_names,
            "keep_subset": torch.tensor(keep_idx, dtype=torch.long, device=self.device),
            "remove_subset": torch.tensor(
                remove_idx, dtype=torch.long, device=self.device
            ),
            "keep_node_ids": [node_ids[i] for i in keep_idx],
        }

    def _add_gene_data(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
    ) -> None:
        # ZERO-COPY: Reference full graph, don't filter anything
        # This ensures x indices align with edge_index (which uses original node IDs)
        integrated_subgraph["gene"].node_ids = cell_graph["gene"].node_ids  # All nodes
        integrated_subgraph["gene"].num_nodes = cell_graph["gene"].num_nodes  # Full count
        integrated_subgraph["gene"].ids_pert = list(gene_info["perturbed_names"])
        integrated_subgraph["gene"].perturbation_indices = gene_info["remove_subset"]

        # ZERO-COPY: Reference full x tensor (no slicing/copying)
        integrated_subgraph["gene"].x = cell_graph["gene"].x  # [num_nodes, feat_dim]
        # No x_pert - use pert_mask to access perturbed features: x[pert_mask]

    @time_method
    def _process_gene_interactions(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
    ) -> None:
        """
        Process gene-gene edge types using zero-copy approach with masks.

        DIFFERENT from SubgraphRepresentation:
        - Returns FULL edge_index (reference to cell_graph)
        - Does NOT filter edges
        - Does NOT relabel nodes
        - Computes boolean mask using incidence cache (O(k×d))

        Model must apply edge masks during message passing.
        """
        # Build cache on first use
        if self._edge_incidence_cache is None:
            self._build_incidence_cache(cell_graph)

        # Get perturbed nodes
        perturbed_nodes = gene_info["remove_subset"]

        # Process all gene-gene edge types
        for et in cell_graph.edge_types:
            if et[0] == "gene" and et[2] == "gene":
                # Return FULL edge_index (reference, not copy)
                integrated_subgraph[et].edge_index = cell_graph[et].edge_index
                num_edges = cell_graph[et].edge_index.size(1)
                integrated_subgraph[et].num_edges = num_edges

                # Compute edge mask using incidence cache - O(k×d)
                edge_mask = torch.ones(num_edges, dtype=torch.bool, device=self.device)

                node_to_edges = self.edge_incidence_cache[et]
                for node_idx in perturbed_nodes:
                    # Set False for all edges touching this perturbed node
                    edge_mask[node_to_edges[node_idx.item()]] = False

                # Store mask (True = keep edge)
                integrated_subgraph[et].mask = edge_mask

    @time_method
    def _process_reaction_info(
        self,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
        integrated_subgraph: HeteroData,
    ) -> Dict[str, Any]:
        """
        Process reaction information using lazy approach.

        Returns full reaction set with masks instead of filtered reactions.
        Computes reaction validity based on whether all required genes are present.
        """
        max_reaction_idx = cell_graph["reaction"].num_nodes

        w_growth = torch.zeros(max_reaction_idx, dtype=torch.float, device=self.device)

        if hasattr(cell_graph["reaction"], "subsystem"):
            subsystems = cell_graph["reaction"].subsystem

            for i in range(max_reaction_idx):
                if isinstance(subsystems, list):
                    if i < len(subsystems) and subsystems[i] == "Growth":
                        w_growth[i] = 1.0
                elif torch.is_tensor(subsystems):
                    if subsystems[i] == "Growth":
                        w_growth[i] = 1.0
                elif hasattr(subsystems, "__getitem__"):
                    if subsystems[i] == "Growth":
                        w_growth[i] = 1.0

        # If no GPR edges, all reactions are valid
        if ("gene", "gpr", "reaction") not in cell_graph.edge_types:
            # Keep all reactions
            self.masks["reaction"]["kept"].fill_(True)
            self.masks["reaction"]["removed"].fill_(False)

            # Identity mappings (no relabeling in lazy approach)
            gene_map = torch.arange(cell_graph["gene"].num_nodes, device=self.device)
            reaction_map = torch.arange(max_reaction_idx, device=self.device)
            valid_reactions = torch.arange(max_reaction_idx, device=self.device)

            return {
                "valid_reactions": valid_reactions,
                "gene_map": gene_map.tolist(),
                "reaction_map": reaction_map.tolist(),
                "w_growth": w_growth,
            }

        # Get FULL GPR edge_index (zero-copy reference)
        gpr_edge_index = cell_graph["gene", "gpr", "reaction"].hyperedge_index.to(
            self.device
        )

        # Create gene mask (True = gene is kept/not deleted)
        gene_mask = torch.zeros(
            cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
        )
        gene_mask[gene_info["keep_subset"]] = True

        gene_indices = gpr_edge_index[0]
        reaction_indices = gpr_edge_index[1]
        max_gene_idx = cell_graph["gene"].num_nodes

        # Validate indices
        valid_mask = (gene_indices < max_gene_idx) & (
            reaction_indices < max_reaction_idx
        )
        gene_indices = gene_indices[valid_mask]
        reaction_indices = reaction_indices[valid_mask]

        # Track which reactions have gene associations
        has_genes = torch.zeros(max_reaction_idx, dtype=torch.bool, device=self.device)
        has_genes[reaction_indices.unique()] = True

        # For each reaction, count how many of its genes are kept
        reaction_gene_sum = scatter(
            gene_mask[gene_indices].float(),
            reaction_indices,
            dim=0,
            dim_size=max_reaction_idx,
            reduce="sum",
        )
        # Count total genes per reaction
        total_gene_count = scatter(
            torch.ones_like(gene_indices, dtype=torch.float),
            reaction_indices,
            dim=0,
            dim_size=max_reaction_idx,
            reduce="sum",
        )

        # Reaction is valid if ALL its genes are kept (not deleted)
        # OR if it has no gene associations
        valid_with_genes_mask = (reaction_gene_sum == total_gene_count) & has_genes
        valid_without_genes_mask = ~has_genes
        valid_mask_combined = valid_with_genes_mask | valid_without_genes_mask

        # Store reaction validity masks
        self.masks["reaction"]["kept"] = valid_mask_combined
        self.masks["reaction"]["removed"] = ~valid_mask_combined

        # Compute GPR edge mask (edge is valid if gene is NOT deleted)
        edge_mask = torch.ones(gpr_edge_index.size(1), dtype=torch.bool, device=self.device)
        edge_mask[valid_mask] = gene_mask[gene_indices]

        # Return FULL GPR hyperedge_index (zero-copy reference)
        integrated_subgraph["gene", "gpr", "reaction"].hyperedge_index = gpr_edge_index
        integrated_subgraph["gene", "gpr", "reaction"].num_edges = gpr_edge_index.size(1)
        integrated_subgraph["gene", "gpr", "reaction"].mask = edge_mask

        # Return identity mappings (no node relabeling in lazy approach)
        gene_map = torch.arange(cell_graph["gene"].num_nodes, device=self.device)
        reaction_map = torch.arange(max_reaction_idx, device=self.device)
        valid_reactions = torch.arange(max_reaction_idx, device=self.device)  # Keep ALL reactions

        return {
            "valid_reactions": valid_reactions,
            "gene_map": gene_map.tolist(),
            "reaction_map": reaction_map.tolist(),
            "w_growth": w_growth,
        }

    def _add_reaction_data(
        self,
        integrated_subgraph: HeteroData,
        reaction_info: Dict[str, Any],
        cell_graph: HeteroData,
    ) -> None:
        """
        Add reaction node data using lazy approach.

        Returns ALL reactions (no filtering), with w_growth attribute.
        """
        if not reaction_info:
            return

        valid_reactions = reaction_info["valid_reactions"]
        integrated_subgraph["reaction"].num_nodes = len(valid_reactions)
        integrated_subgraph["reaction"].node_ids = valid_reactions.tolist()

        # Return full w_growth (reference, no filtering since valid_reactions contains all indices)
        if hasattr(cell_graph["reaction"], "w_growth"):
            w_growth = reaction_info["w_growth"]
            integrated_subgraph["reaction"].w_growth = w_growth

    def _process_metabolic_network(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        reaction_info: Dict[str, Any],
    ) -> None:
        # Same as SubgraphRepresentation
        if not reaction_info or "reaction" not in cell_graph.node_types:
            return
        if ("reaction", "rmr", "metabolite") in cell_graph.edge_types:
            self._process_metabolism_bipartite(
                integrated_subgraph, cell_graph, reaction_info
            )

    @time_method
    def _process_metabolism_bipartite(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        reaction_info: Dict[str, Any],
    ) -> None:
        """
        Process RMR (Reaction-Metabolite-Reaction) edges using lazy approach.

        Returns full hyperedge_index and stoichiometry (zero-copy references)
        with edge masks based on reaction validity from Phase 4.1.

        Phase 4.2: Zero-copy for metabolism bipartite edges
        """
        # Get FULL RMR hyperedge_index and stoichiometry (zero-copy references)
        rmr_edges = cell_graph["reaction", "rmr", "metabolite"]
        hyperedge_index = rmr_edges.hyperedge_index.to(self.device)
        stoichiometry = rmr_edges.stoichiometry.to(self.device)

        # Compute edge mask: edge is valid if source reaction is valid
        reaction_indices = hyperedge_index[0]
        edge_mask = self.masks["reaction"]["kept"][reaction_indices]

        # Return FULL graph with mask (zero-copy)
        edge_type = ("reaction", "rmr", "metabolite")
        integrated_subgraph[edge_type].hyperedge_index = hyperedge_index  # Reference
        integrated_subgraph[edge_type].stoichiometry = stoichiometry      # Reference
        integrated_subgraph[edge_type].mask = edge_mask                   # Boolean mask
        integrated_subgraph[edge_type].num_edges = hyperedge_index.size(1)  # Full count

        # Metabolite nodes (keep all)
        integrated_subgraph["metabolite"].node_ids = cell_graph["metabolite"].node_ids
        integrated_subgraph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes
        self.masks["metabolite"]["kept"].fill_(True)
        self.masks["metabolite"]["removed"].fill_(False)

    def _add_phenotype_data(
        self,
        integrated_subgraph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> None:
        # Same as SubgraphRepresentation - copy entire method
        if self.device is None:
            self.device = torch.device("cpu")

        all_values = []
        all_type_indices = []
        all_sample_indices = []
        phenotype_types = []

        all_stat_values = []
        all_stat_type_indices = []
        all_stat_sample_indices = []
        stat_types = []

        for phenotype_class in phenotype_info:
            label_name = phenotype_class.model_fields["label_name"].default
            stat_name = phenotype_class.model_fields["label_statistic_name"].default

            phenotype_types.append(label_name)
            if stat_name:
                stat_types.append(stat_name)

        for item_idx, item in enumerate(data):
            phenotype = item["experiment"].phenotype

            for type_idx, field_name in enumerate(phenotype_types):
                value = getattr(phenotype, field_name, None)
                if value is not None:
                    values = [value] if not isinstance(value, (list, tuple)) else value

                    all_values.extend(values)
                    all_type_indices.extend([type_idx] * len(values))
                    all_sample_indices.extend([item_idx] * len(values))

            for stat_type_idx, stat_field_name in enumerate(stat_types):
                stat_value = getattr(phenotype, stat_field_name, None)
                if stat_value is not None:
                    stat_values = (
                        [stat_value]
                        if not isinstance(stat_value, (list, tuple))
                        else stat_value
                    )

                    all_stat_values.extend(stat_values)
                    all_stat_type_indices.extend([stat_type_idx] * len(stat_values))
                    all_stat_sample_indices.extend([item_idx] * len(stat_values))

        if all_values:
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                all_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                all_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                all_sample_indices, dtype=torch.long, device=self.device
            )
        else:
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                [float("nan")], dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )

        integrated_subgraph["gene"]["phenotype_types"] = phenotype_types

        if all_stat_values:
            integrated_subgraph["gene"]["phenotype_stat_values"] = torch.tensor(
                all_stat_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_type_indices"] = torch.tensor(
                all_stat_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_sample_indices"] = torch.tensor(
                all_stat_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_types"] = stat_types


class Unperturbed(GraphProcessor):
    """
    Processes graph data by preserving the original graph structure and storing perturbation
    and phenotype data alongside it for later processing.

    This processor:
    1. Keeps the original graph structure intact
    2. Stores perturbation information separately
    3. Records phenotype data without modifying the graph
    4. Can be used as a base for applying perturbations later in the pipeline

    Attributes remain unchanged:
        - Node features: X ∈ ℝ^(N×d) stays as X ∈ ℝ^(N×d)
        - Edge structure: E ∈ ℤ^(2×|E|) stays as E ∈ ℤ^(2×|E|)
    """

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> HeteroData:
        if not data:
            raise ValueError("Data list is empty")

        processed_graph = HeteroData()

        # Copy graph structure and features
        processed_graph["gene"].x = cell_graph["gene"].x
        processed_graph["gene"].node_ids = cell_graph["gene"].node_ids
        processed_graph["gene"].num_nodes = cell_graph["gene"].num_nodes

        # Store perturbation information
        perturbed_genes = set()
        for item in data:
            if "experiment" not in item or "experiment_reference" not in item:
                raise ValueError(
                    "Each item in data must contain both 'experiment' and "
                    "'experiment_reference' keys"
                )
            perturbed_genes.update(
                pert.systematic_gene_name
                for pert in item["experiment"].genotype.perturbations
            )

        processed_graph["gene"].perturbed_genes = list(perturbed_genes)
        processed_graph["gene"].perturbation_indices = torch.tensor(
            [cell_graph["gene"].node_ids.index(nid) for nid in perturbed_genes],
            dtype=torch.long,
        )

        # Add phenotype fields
        phenotype_fields = []
        for phenotype in phenotype_info:
            phenotype_fields.append(phenotype.model_fields["label_name"].default)
            phenotype_fields.append(
                phenotype.model_fields["label_statistic_name"].default
            )

        # Add experiment data
        for field in phenotype_fields:
            field_values = []
            for item in data:
                value = getattr(item["experiment"].phenotype, field, None)
                if value is not None:
                    field_values.append(value)
            if field_values:
                processed_graph["gene"][field] = torch.tensor(field_values)
            else:
                processed_graph["gene"][field] = torch.tensor([float("nan")])

        # Copy edge information
        for edge_type in cell_graph.edge_types:
            if edge_type[1] in ["physical_interaction", "regulatory_interaction"]:
                processed_graph[edge_type].edge_index = cell_graph[edge_type].edge_index
                processed_graph[edge_type].num_edges = cell_graph[edge_type].num_edges

        # Handle metabolite data
        if "metabolite" in cell_graph.node_types:
            processed_graph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes
            processed_graph["metabolite"].node_ids = cell_graph["metabolite"].node_ids

            edge_type = ("metabolite", "reactions", "metabolite")
            if any(e_type == edge_type for e_type in cell_graph.edge_types):
                # Copy hypergraph structure
                processed_graph[edge_type].hyperedge_index = cell_graph[
                    edge_type
                ].hyperedge_index
                processed_graph[edge_type].stoichiometry = cell_graph[
                    edge_type
                ].stoichiometry
                processed_graph[edge_type].num_edges = cell_graph[edge_type].num_edges

                # Only create reaction_to_genes_indices mapping
                node_id_to_idx = {
                    nid: idx for idx, nid in enumerate(cell_graph["gene"].node_ids)
                }
                reaction_to_genes_indices = {}

                for reaction_idx, genes in cell_graph[
                    edge_type
                ].reaction_to_genes.items():
                    gene_indices = []
                    for gene in genes:
                        gene_idx = node_id_to_idx.get(gene, -1)
                        gene_indices.append(gene_idx)
                    reaction_to_genes_indices[reaction_idx] = gene_indices

                processed_graph[edge_type].reaction_to_genes_indices = (
                    reaction_to_genes_indices
                )

        return processed_graph


class Perturbation(GraphProcessor):
    """
    Processes graph data by storing only perturbation-specific information without duplicating
    the base graph structure, using COO format for phenotypes.
    """

    def __init__(self) -> None:
        super().__init__()
        # Always use CPU for pin_memory compatibility
        self.device = torch.device("cpu")

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> HeteroData:
        if not data:
            raise ValueError("Data list is empty")

        # Always use CPU for pin_memory compatibility
        # The model will handle moving tensors to GPU after DataLoader
        self.device = torch.device("cpu")

        # Create a minimal HeteroData object to store perturbation data
        processed_graph = HeteroData()

        # Add required node count to avoid PyG warnings
        processed_graph["gene"].num_nodes = cell_graph["gene"].num_nodes

        # Process perturbed genes
        perturbed_names = {
            p.systematic_gene_name
            for item in data
            for p in item["experiment"].genotype.perturbations
        }
        node_ids = cell_graph["gene"].node_ids
        perturbed_indices = [
            i for i, name in enumerate(node_ids) if name in perturbed_names
        ]

        # Create perturbation mask: True for perturbed genes
        pert_mask = torch.zeros(
            cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
        )
        pert_mask[perturbed_indices] = True

        # Create regular mask: EXPLICITLY as logical NOT of pert_mask
        mask = ~pert_mask

        # Store perturbation information
        processed_graph["gene"].perturbed_genes = list(perturbed_names)
        processed_graph["gene"].perturbation_indices = torch.tensor(
            perturbed_indices, dtype=torch.long, device=self.device
        )

        # Store masks
        processed_graph["gene"].pert_mask = pert_mask
        processed_graph["gene"].mask = mask

        # Verify mask sums are correct
        assert pert_mask.sum() == len(
            perturbed_indices
        ), "pert_mask sum doesn't match perturbed indices"
        assert mask.sum() == cell_graph["gene"].num_nodes - len(
            perturbed_indices
        ), "mask sum isn't complement of pert_mask"

        # Add phenotype data in COO format
        self._add_phenotype_data(processed_graph, phenotype_info, data)

        return processed_graph

    def _add_phenotype_data(
        self,
        integrated_subgraph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> None:
        # Rest of the phenotype data processing remains the same...
        # Initializing storage, processing phenotypes, etc.

        # Initialize storage for phenotype values and metadata
        all_values = []
        all_type_indices = []
        all_sample_indices = []
        phenotype_types = []

        # Initialize storage for statistic values
        all_stat_values = []
        all_stat_type_indices = []
        all_stat_sample_indices = []
        stat_types = []

        # Extract phenotype type information
        for phenotype_class in phenotype_info:
            label_name = phenotype_class.model_fields["label_name"].default
            stat_name = phenotype_class.model_fields["label_statistic_name"].default

            phenotype_types.append(label_name)
            if stat_name:
                stat_types.append(stat_name)

        # Process each experimental data point
        for item_idx, item in enumerate(data):
            # Get phenotype object from experiment
            phenotype = item["experiment"].phenotype

            # Process each phenotype type
            for type_idx, field_name in enumerate(phenotype_types):
                value = getattr(phenotype, field_name, None)
                if value is not None:
                    # Convert single values to lists for consistent handling
                    values = [value] if not isinstance(value, (list, tuple)) else value

                    # Add all values with their type indices and sample indices
                    all_values.extend(values)
                    all_type_indices.extend([type_idx] * len(values))
                    all_sample_indices.extend([item_idx] * len(values))

            # Process statistics in the same way
            for stat_type_idx, stat_field_name in enumerate(stat_types):
                stat_value = getattr(phenotype, stat_field_name, None)
                if stat_value is not None:
                    # Convert single values to lists for consistent handling
                    stat_values = (
                        [stat_value]
                        if not isinstance(stat_value, (list, tuple))
                        else stat_value
                    )

                    # Add all statistic values with their type indices and sample indices
                    all_stat_values.extend(stat_values)
                    all_stat_type_indices.extend([stat_type_idx] * len(stat_values))
                    all_stat_sample_indices.extend([item_idx] * len(stat_values))

        # Store phenotype data in the graph
        if all_values:
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                all_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                all_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                all_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_types"] = phenotype_types
        else:
            # Handle empty case with placeholder values
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                [float("nan")], dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_types"] = phenotype_types

        # Store statistic data in the graph
        if all_stat_values:
            integrated_subgraph["gene"]["phenotype_stat_values"] = torch.tensor(
                all_stat_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_type_indices"] = torch.tensor(
                all_stat_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_sample_indices"] = torch.tensor(
                all_stat_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_types"] = stat_types


class DCellGraphProcessor(GraphProcessor):
    """
    Graph processor for DCell model that applies perturbations directly to the gene ontology graph.
    This processor updates the gene ontology node states based on perturbations rather than
    deleting nodes from the graph before conversion.
    """

    def __init__(self) -> None:
        super().__init__()
        # Always use CPU for pin_memory compatibility
        self.device = torch.device("cpu")

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> HeteroData:
        if not data:
            raise ValueError("Data list is empty")

        # Always use CPU for pin_memory compatibility
        # The model will handle moving tensors to GPU after DataLoader
        self.device = torch.device("cpu")

        # Create a new HeteroData object for processing
        processed_graph = HeteroData()

        # Add gene node information
        processed_graph["gene"].num_nodes = cell_graph["gene"].num_nodes
        processed_graph["gene"].node_ids = cell_graph["gene"].node_ids

        if hasattr(cell_graph["gene"], "x"):
            processed_graph["gene"].x = cell_graph["gene"].x

        # Process perturbed genes
        perturbed_names = {
            p.systematic_gene_name
            for item in data
            for p in item["experiment"].genotype.perturbations
        }
        node_ids = cell_graph["gene"].node_ids
        perturbed_indices = [
            i for i, name in enumerate(node_ids) if name in perturbed_names
        ]

        # Create perturbation mask: True for perturbed genes
        pert_mask = torch.zeros(
            cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
        )
        pert_mask[perturbed_indices] = True

        # Store perturbation information
        processed_graph["gene"].perturbed_genes = list(perturbed_names)
        processed_graph["gene"].perturbation_indices = torch.tensor(
            perturbed_indices, dtype=torch.long, device=self.device
        )
        processed_graph["gene"].pert_mask = pert_mask

        # Group perturbation indices by experiment for batch processing
        batch_indices = []
        batch_mapping = {}  # Maps experiment index to list of perturbation indices

        for i, item in enumerate(data):
            perturbed_genes = [
                p.systematic_gene_name
                for p in item["experiment"].genotype.perturbations
            ]
            gene_indices = [
                node_ids.index(gene) for gene in perturbed_genes if gene in node_ids
            ]

            batch_indices.extend([i] * len(gene_indices))
            batch_mapping[i] = gene_indices

        # Add batch indices for perturbations - needed for processing by experiment
        if batch_indices:
            processed_graph["gene"].perturbation_indices_batch = torch.tensor(
                batch_indices, dtype=torch.long, device=self.device
            )

        # Process gene ontology if it exists in cell_graph
        if "gene_ontology" in cell_graph.node_types:
            # Copy ALL gene ontology information from cell_graph
            processed_graph["gene_ontology"].num_nodes = cell_graph[
                "gene_ontology"
            ].num_nodes
            processed_graph["gene_ontology"].node_ids = cell_graph[
                "gene_ontology"
            ].node_ids

            # Copy the go_gene_strata_state and apply perturbations
            if hasattr(cell_graph["gene_ontology"], "go_gene_strata_state"):
                self._apply_go_gene_perturbations(
                    processed_graph, cell_graph, perturbed_indices
                )

        # Add phenotype data in COO format
        self._add_phenotype_data(processed_graph, phenotype_info, data)

        return processed_graph

    def _apply_go_gene_perturbations(
        self,
        processed_graph: HeteroData,
        cell_graph: HeteroData,
        perturbed_indices: list,
    ) -> None:
        """
        Copy go_gene_strata_state from cell_graph and flip perturbation bits.
        """
        # Copy the base state tensor from cell_graph and keep on CPU
        # This ensures compatibility with pin_memory in DataLoader
        base_state = (
            cell_graph["gene_ontology"].go_gene_strata_state.clone().cpu()
        )

        # Convert perturbed indices to set for O(1) lookup
        perturbed_set = set(perturbed_indices)

        # Flip state bits for perturbed genes
        # Column 1 contains gene_idx, column 3 contains state
        gene_indices = base_state[:, 1].long()
        mask = torch.tensor(
            [idx.item() in perturbed_set for idx in gene_indices],
            dtype=torch.bool,
            device=self.device,
        )
        base_state[mask, 3] = 0.0

        # Store the modified state tensor
        processed_graph["gene_ontology"].go_gene_strata_state = base_state

    def _add_phenotype_data(
        self,
        integrated_subgraph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> None:
        """
        Add phenotype data to the graph in COO format.
        Optimized version that ensures all tensors are on the same device.
        """
        # Always use CPU for pin_memory compatibility
        if self.device is None:
            self.device = torch.device("cpu")

        # Initialize storage for phenotype values and metadata
        all_values = []
        all_type_indices = []
        all_sample_indices = []  # Which sample each value belongs to
        phenotype_types = []

        # Initialize storage for statistic values
        all_stat_values = []
        all_stat_type_indices = []
        all_stat_sample_indices = []  # Which sample each stat value belongs to
        stat_types = []

        # Extract phenotype type information
        for phenotype_class in phenotype_info:
            label_name = phenotype_class.model_fields["label_name"].default
            stat_name = phenotype_class.model_fields["label_statistic_name"].default

            phenotype_types.append(label_name)
            if stat_name:
                stat_types.append(stat_name)

        # Process each experimental data point
        for item_idx, item in enumerate(data):
            # Get phenotype object from experiment
            phenotype = item["experiment"].phenotype

            # Process each phenotype type
            for type_idx, field_name in enumerate(phenotype_types):
                value = getattr(phenotype, field_name, None)
                if value is not None:
                    # Convert single values to lists for consistent handling
                    values = [value] if not isinstance(value, (list, tuple)) else value

                    # Add all values with their type indices and sample indices
                    all_values.extend(values)
                    all_type_indices.extend([type_idx] * len(values))
                    all_sample_indices.extend([item_idx] * len(values))

            # Process statistics in the same way
            for stat_type_idx, stat_field_name in enumerate(stat_types):
                stat_value = getattr(phenotype, stat_field_name, None)
                if stat_value is not None:
                    # Convert single values to lists for consistent handling
                    stat_values = (
                        [stat_value]
                        if not isinstance(stat_value, (list, tuple))
                        else stat_value
                    )

                    # Add all statistic values with their type indices and sample indices
                    all_stat_values.extend(stat_values)
                    all_stat_type_indices.extend([stat_type_idx] * len(stat_values))
                    all_stat_sample_indices.extend([item_idx] * len(stat_values))

        # Store phenotype data in the graph - create all tensors directly on the target device
        if all_values:
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                all_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                all_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                all_sample_indices, dtype=torch.long, device=self.device
            )
        else:
            # Handle empty case with placeholder values - create directly on target device
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                [float("nan")], dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )

        # Store phenotype type names (non-tensor data)
        integrated_subgraph["gene"]["phenotype_types"] = phenotype_types

        # Store statistic data in the graph - create directly on target device
        if all_stat_values:
            integrated_subgraph["gene"]["phenotype_stat_values"] = torch.tensor(
                all_stat_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_type_indices"] = torch.tensor(
                all_stat_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_sample_indices"] = torch.tensor(
                all_stat_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_types"] = stat_types

class NeighborSubgraphRepresentation(GraphProcessor):
    """
    GraphProcessor that creates k-hop induced subgraphs around perturbed genes.

    Unlike SubgraphRepresentation (which filters out perturbed genes) or
    LazySubgraphRepresentation (which keeps full graph with masks), this processor
    creates small induced subgraphs containing only k-hop neighborhoods around
    perturbed genes.

    Benefits:
    - Smaller graphs for faster message passing
    - Reduced data loading overhead
    - Preserves original node indices for easy mapping back to full graph

    Args:
        num_hops: Number of hops to include in neighborhood (default: 2)
    """

    def __init__(self, num_hops: int = 2):
        self.num_hops = num_hops
        self.device = torch.device("cpu")
        self.masks = {}

    def _initialize_masks(self, cell_graph: HeteroData):
        """Initialize boolean masks for all nodes and edges."""
        self.masks = {}
        num_genes = cell_graph["gene"].num_nodes
        self.masks["gene"] = {
            "perturbed": torch.zeros(num_genes, dtype=torch.bool, device=self.device),
            "kept": torch.zeros(num_genes, dtype=torch.bool, device=self.device),
        }

    def _process_gene_info(self, cell_graph: HeteroData, data):
        """Identify perturbed genes from experiment data."""
        perturbed_names = set()
        for item in data:
            for p in item["experiment"].genotype.perturbations:
                perturbed_names.add(p.systematic_gene_name)
        node_ids = cell_graph["gene"].node_ids
        perturbed_indices = []
        for i, name in enumerate(node_ids):
            if name in perturbed_names:
                perturbed_indices.append(i)
        perturbed_indices = torch.tensor(perturbed_indices, dtype=torch.long)
        self.masks["gene"]["perturbed"][perturbed_indices] = True
        return {
            "perturbed_names": perturbed_names,
            "perturbed_indices": perturbed_indices,
            "perturbed_node_ids": [node_ids[i] for i in perturbed_indices],
        }

    def _build_khop_subgraph(self, cell_graph: HeteroData, gene_info: dict) -> dict:
        """Build k-hop induced subgraph around perturbed genes."""
        perturbed_indices = gene_info["perturbed_indices"]
        num_genes = cell_graph["gene"].num_nodes
        all_neighbors = set(perturbed_indices.tolist())
        edge_data = {}
        for et in cell_graph.edge_types:
            if et[0] == "gene" and et[2] == "gene":
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_idx=perturbed_indices,
                    num_hops=self.num_hops,
                    edge_index=cell_graph[et].edge_index,
                    relabel_nodes=False,
                    num_nodes=num_genes,
                )
                all_neighbors.update(subset.tolist())
                edge_data[et] = {"edge_index": edge_index, "edge_mask": edge_mask, "subset": subset}
        subset_nodes = torch.tensor(sorted(all_neighbors), dtype=torch.long)
        subset_node_ids = [cell_graph["gene"].node_ids[i] for i in subset_nodes]

        # Vectorized perturbed mask creation
        perturbed_mask = self.masks["gene"]["perturbed"][subset_nodes]

        # Vectorized edge filtering
        filtered_edge_data = {}
        for et, data_dict in edge_data.items():
            edge_index = cell_graph[et].edge_index
            # Use torch.isin for vectorized membership testing
            src_mask = torch.isin(edge_index[0], subset_nodes)
            dst_mask = torch.isin(edge_index[1], subset_nodes)
            mask = src_mask & dst_mask
            filtered_edge_index = edge_index[:, mask]
            filtered_edge_data[et] = {"edge_index": filtered_edge_index, "num_edges": filtered_edge_index.size(1)}
        return {"subset_nodes": subset_nodes, "subset_node_ids": subset_node_ids, "edge_data": filtered_edge_data, "perturbed_mask": perturbed_mask}

    def _process_metabolism(self, integrated_subgraph: HeteroData, cell_graph: HeteroData, subgraph_info: dict):
        """Add metabolism edges for genes in the k-hop subgraph."""
        if "reaction" not in cell_graph.node_types:
            return
        subset_nodes = subgraph_info["subset_nodes"]

        if ("gene", "gpr", "reaction") in cell_graph.edge_types:
            gpr_et = ("gene", "gpr", "reaction")
            gpr_hyperedge_index = cell_graph[gpr_et].hyperedge_index

            # Vectorized GPR edge filtering
            gpr_mask = torch.isin(gpr_hyperedge_index[0], subset_nodes)
            included_reaction_indices = gpr_hyperedge_index[1, gpr_mask].unique()

            integrated_subgraph[gpr_et].hyperedge_index = gpr_hyperedge_index
            integrated_subgraph[gpr_et].num_edges = gpr_hyperedge_index.size(1)
            integrated_subgraph[gpr_et].mask = gpr_mask

            if included_reaction_indices.numel() > 0:
                reaction_indices = included_reaction_indices.sort()[0]
                integrated_subgraph["reaction"].node_ids = [cell_graph["reaction"].node_ids[i] for i in reaction_indices]
                integrated_subgraph["reaction"].num_nodes = len(reaction_indices)
                integrated_subgraph["reaction"].pert_mask = torch.zeros(len(reaction_indices), dtype=torch.bool)
                integrated_subgraph["reaction"].mask = torch.ones(len(reaction_indices), dtype=torch.bool)

                if ("reaction", "rmr", "metabolite") in cell_graph.edge_types:
                    rmr_et = ("reaction", "rmr", "metabolite")
                    rmr_hyperedge_index = cell_graph[rmr_et].hyperedge_index
                    rmr_stoichiometry = cell_graph[rmr_et].stoichiometry

                    # Vectorized RMR edge filtering
                    rmr_mask = torch.isin(rmr_hyperedge_index[0], included_reaction_indices)

                    integrated_subgraph[rmr_et].hyperedge_index = rmr_hyperedge_index
                    integrated_subgraph[rmr_et].stoichiometry = rmr_stoichiometry
                    integrated_subgraph[rmr_et].num_edges = rmr_hyperedge_index.size(1)
                    integrated_subgraph[rmr_et].mask = rmr_mask
                    integrated_subgraph["metabolite"].node_ids = cell_graph["metabolite"].node_ids
                    integrated_subgraph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes
                    integrated_subgraph["metabolite"].pert_mask = torch.zeros(cell_graph["metabolite"].num_nodes, dtype=torch.bool)
                    integrated_subgraph["metabolite"].mask = torch.ones(cell_graph["metabolite"].num_nodes, dtype=torch.bool)

    def _add_gene_data(self, integrated_subgraph: HeteroData, cell_graph: HeteroData, gene_info: dict, subgraph_info: dict):
        """Add gene node data to the subgraph."""
        subset_nodes = subgraph_info["subset_nodes"]
        integrated_subgraph["gene"].node_ids = subgraph_info["subset_node_ids"]
        integrated_subgraph["gene"].num_nodes = len(subset_nodes)
        integrated_subgraph["gene"].ids_pert = list(gene_info["perturbed_names"])
        integrated_subgraph["gene"].perturbation_indices = gene_info["perturbed_indices"]
        integrated_subgraph["gene"].x = cell_graph["gene"].x[subset_nodes]
        integrated_subgraph["gene"].pert_mask = subgraph_info["perturbed_mask"]
        integrated_subgraph["gene"].x_pert = integrated_subgraph["gene"].x.clone()
        integrated_subgraph["gene"].x_pert[subgraph_info["perturbed_mask"]] = 0.0

    def _add_phenotype_data(self, integrated_subgraph: HeteroData, phenotype_info: list, data: list):
        """Add phenotype data in COO format."""
        all_values, all_type_indices, all_sample_indices, phenotype_types = [], [], [], []
        all_stat_values, all_stat_type_indices, all_stat_sample_indices, stat_types = [], [], [], []
        for phenotype_class in phenotype_info:
            label_name = phenotype_class.model_fields["label_name"].default
            stat_name = phenotype_class.model_fields["label_statistic_name"].default
            phenotype_types.append(label_name)
            if stat_name:
                stat_types.append(stat_name)
        for item_idx, item in enumerate(data):
            phenotype = item["experiment"].phenotype
            for type_idx, field_name in enumerate(phenotype_types):
                value = getattr(phenotype, field_name, None)
                if value is not None:
                    values = [value] if not isinstance(value, (list, tuple)) else value
                    all_values.extend(values)
                    all_type_indices.extend([type_idx] * len(values))
                    all_sample_indices.extend([item_idx] * len(values))
            for stat_type_idx, stat_field_name in enumerate(stat_types):
                stat_value = getattr(phenotype, stat_field_name, None)
                if stat_value is not None:
                    stat_values = [stat_value] if not isinstance(stat_value, (list, tuple)) else stat_value
                    all_stat_values.extend(stat_values)
                    all_stat_type_indices.extend([stat_type_idx] * len(stat_values))
                    all_stat_sample_indices.extend([item_idx] * len(stat_values))
        if all_values:
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(all_values, dtype=torch.float, device=self.device)
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(all_type_indices, dtype=torch.long, device=self.device)
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(all_sample_indices, dtype=torch.long, device=self.device)
            integrated_subgraph["gene"]["phenotype_types"] = phenotype_types
        if all_stat_values:
            integrated_subgraph["gene"]["phenotype_stat_values"] = torch.tensor(all_stat_values, dtype=torch.float, device=self.device)
            integrated_subgraph["gene"]["phenotype_stat_type_indices"] = torch.tensor(all_stat_type_indices, dtype=torch.long, device=self.device)
            integrated_subgraph["gene"]["phenotype_stat_sample_indices"] = torch.tensor(all_stat_sample_indices, dtype=torch.long, device=self.device)
            integrated_subgraph["gene"]["phenotype_stat_types"] = stat_types

    def process(self, cell_graph: HeteroData, phenotype_info: list[PhenotypeType], data: list[Dict[str, ExperimentType | ExperimentReferenceType]]) -> HeteroData:
        """Main processing method. Creates k-hop induced subgraph around perturbed genes."""
        self._initialize_masks(cell_graph)
        integrated_subgraph = HeteroData()
        gene_info = self._process_gene_info(cell_graph, data)
        subgraph_info = self._build_khop_subgraph(cell_graph, gene_info)
        self._add_gene_data(integrated_subgraph, cell_graph, gene_info, subgraph_info)
        for et, edge_data in subgraph_info["edge_data"].items():
            integrated_subgraph[et].edge_index = edge_data["edge_index"]
            integrated_subgraph[et].num_edges = edge_data["num_edges"]
        self._process_metabolism(integrated_subgraph, cell_graph, subgraph_info)
        self._add_phenotype_data(integrated_subgraph, phenotype_info, data)
        return integrated_subgraph
