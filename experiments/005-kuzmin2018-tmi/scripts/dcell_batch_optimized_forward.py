"""
DCell batch processing optimization experiment.

This script contains a test implementation of the optimized DCell forward method
using stratum-based batch processing to improve parallel execution on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def optimized_stratum_forward(self, cell_graph, batch):
    """
    Vectorized forward pass for the DCell model using stratum-based parallelization.
    Processes GO terms in parallel by stratum, processing each stratum sequentially.
    This optimized version groups subsystems by input/output dimensions for batch processing.

    Args:
        cell_graph: The cell graph containing gene ontology structure
        batch: HeteroDataBatch containing perturbation information and mutant states

    Returns:
        Tuple of (root_output, outputs_dictionary)
    """
    # Initialize subsystems from cell_graph if not done yet
    if not self.initialized:
        self._initialize_from_cell_graph(cell_graph)

    # Double-check that we're initialized properly
    if len(self.subsystems) == 0:
        raise ValueError(
            "Model has no subsystems after initialization. "
            "This indicates a problem with the GO hierarchy structure or filtering."
        )

    # Get the device from the model parameters for consistency
    device = next(self.parameters()).device
    
    # Ensure data is on the correct device
    cell_graph = cell_graph.to(device)
    batch = batch.to(device)
    num_graphs = batch.num_graphs

    # Verify mutant state exists and has the right format
    if not hasattr(batch["gene_ontology"], "mutant_state"):
        raise ValueError("Batch must contain gene_ontology.mutant_state for DCell model")

    mutant_state = batch["gene_ontology"].mutant_state
    
    # Verify mutant state has right column count [term_idx, gene_idx, stratum, gene_state]
    if mutant_state.shape[1] != 4:
        raise ValueError(f"Mutant state must have 4 columns, got {mutant_state.shape[1]}")
    
    # Verify strata groups are available
    if self.stratum_to_systems is None:
        raise ValueError("Stratum to systems mapping is not initialized")
        
    print(f"Processing with {len(self.stratum_to_systems)} strata")
        
    # Map from term IDs to indices in the cell_graph
    term_id_to_idx = {
        term_id: idx for idx, term_id in enumerate(cell_graph["gene_ontology"].node_ids)
    }
    
    # Dictionary to store all subsystem outputs
    subsystem_outputs = {}
    
    # Extract information from mutant state
    term_indices = mutant_state[:, 0].long()
    gene_indices = mutant_state[:, 1].long()
    strata_indices = mutant_state[:, 2].long()
    gene_states = mutant_state[:, 3]
    
    # Extract batch information if available
    if hasattr(batch["gene_ontology"], "mutant_state_batch"):
        batch_indices = batch["gene_ontology"].mutant_state_batch
    else:
        batch_indices = torch.zeros(mutant_state.size(0), dtype=torch.long, device=device)
    
    # Pre-process gene states for all terms
    term_gene_states = {}
    term_indices_set = torch.unique(term_indices).tolist()
    
    # Efficiently process gene states for each term
    for term_idx in term_indices_set:
        term_idx = term_idx if isinstance(term_idx, int) else term_idx.item()
        
        # Get genes for this term
        genes = cell_graph["gene_ontology"].term_to_gene_dict.get(term_idx, [])
        num_genes = max(1, len(genes))
        
        # Create gene state tensor based on embedding mode
        if self.learnable_embedding_dim is not None:
            # Initialize tensor with embeddings - [batch_size, num_genes, embedding_dim]
            term_gene_states[term_idx] = torch.zeros(
                (num_graphs, num_genes, self.learnable_embedding_dim),
                dtype=torch.float, device=device
            )
            
            # Set default embeddings
            for gene_local_idx, gene_idx in enumerate(genes):
                if gene_idx < self.gene_embeddings.weight.size(0):
                    term_gene_states[term_idx][:, gene_local_idx] = self.gene_embeddings.weight[gene_idx]
        else:
            # Binary encoding - all genes present (1.0) by default
            term_gene_states[term_idx] = torch.ones(
                (num_graphs, num_genes), dtype=torch.float, device=device
            )
        
        # Apply perturbations from mutant state
        term_mask = term_indices == term_idx
        term_data = mutant_state[term_mask]
        
        if term_data.size(0) > 0:
            # Get batch indices for this term
            term_batch_indices = batch_indices[term_mask] if batch_indices is not None else torch.zeros(
                term_data.size(0), dtype=torch.long, device=device
            )
            
            # Apply perturbations
            for i in range(term_data.size(0)):
                batch_idx = term_batch_indices[i].item()
                gene_idx = term_data[i, 1].long().item()
                state_value = term_data[i, 3].item()
                
                # Find gene in the term's gene list
                if gene_idx < len(genes):
                    gene_local_idx = genes.index(gene_idx) if gene_idx in genes else -1
                    if gene_local_idx >= 0 and state_value != 1.0:
                        # Zero out gene or embedding for perturbed genes
                        if self.learnable_embedding_dim is not None:
                            term_gene_states[term_idx][batch_idx, gene_local_idx] = 0.0
                        else:
                            term_gene_states[term_idx][batch_idx, gene_local_idx] = 0.0
    
    # Process strata in order (from lowest to highest)
    for stratum in self.sorted_strata:
        # Get all systems at this stratum
        stratum_systems = self.stratum_to_systems[stratum]
        
        # Prepare for batch processing of subsystems in this stratum
        # Group subsystems by input and output sizes for efficient batch processing
        term_ids = []
        subsystem_models = []
        gene_states_list = []
        child_outputs_list = []
        input_sizes = []
        output_sizes = []
        combined_inputs = []
        
        # First, collect data for all subsystems in this stratum
        for term_id, subsystem_model in stratum_systems:
            # Get the term index
            if term_id == "GO:ROOT":
                term_idx = -1
            else:
                term_idx = term_id_to_idx.get(term_id, -1)
                if term_idx == -1:
                    continue
            
            # Get gene states for this term
            if term_idx in term_gene_states:
                gene_states = term_gene_states[term_idx]
            else:
                # Create default state tensor for terms not in mutant_state
                genes = cell_graph["gene_ontology"].term_to_gene_dict.get(term_idx, [])
                
                if self.learnable_embedding_dim is not None:
                    gene_states = torch.zeros(
                        (num_graphs, max(1, len(genes)), self.learnable_embedding_dim),
                        dtype=torch.float, device=device
                    )
                    
                    for gene_local_idx, gene_idx in enumerate(genes):
                        if gene_idx < self.gene_num:
                            gene_states[:, gene_local_idx] = self.gene_embeddings.weight[gene_idx]
                else:
                    gene_states = torch.ones(
                        (num_graphs, max(1, len(genes))),
                        dtype=torch.float, device=device
                    )
            
            # Reshape embeddings if needed
            if self.learnable_embedding_dim is not None and len(gene_states.shape) == 3:
                batch_size = gene_states.size(0)
                gene_states = gene_states.reshape(batch_size, -1)
            
            # Get outputs from child nodes
            child_outputs = []
            for child in self.go_graph.successors(term_id):
                if child in subsystem_outputs:
                    child_outputs.append(subsystem_outputs[child])
            
            # Combine inputs
            if child_outputs:
                child_tensor = torch.cat(child_outputs, dim=1)
                
                if term_id == "GO:ROOT":
                    # Special handling for root node
                    combined_input = child_tensor
                    padding = torch.zeros((combined_input.size(0), 1), device=device)
                    combined_input = torch.cat([combined_input, padding], dim=1)
                else:
                    combined_input = torch.cat([gene_states, child_tensor], dim=1)
            else:
                combined_input = gene_states
            
            # Validate input size
            expected_size = subsystem_model.layers[0].weight.size(1)
            actual_size = combined_input.size(1)
            
            if actual_size != expected_size:
                # Provide helpful diagnostics for size mismatch
                if term_id == "GO:ROOT":
                    print(f"\nDiagnostic information for GO:ROOT node:")
                    print(f"  Expected input size: {expected_size}")
                    print(f"  Actual input size: {actual_size}")
                    
                    child_output_sum = 0
                    for child in self.go_graph.successors(term_id):
                        if child in subsystem_outputs:
                            child_size = subsystem_outputs[child].size(1)
                            child_output_sum += child_size
                            print(f"  Child '{child}' output size: {child_size}")
                    
                    print(f"  Sum of child output sizes: {child_output_sum}")
                
                # Fail with clear error message
                raise ValueError(
                    f"Size mismatch for subsystem '{term_id}' in stratum {stratum}: "
                    f"expected {expected_size}, got {actual_size}."
                )
            
            # Save all data for batch processing
            term_ids.append(term_id)
            subsystem_models.append(subsystem_model)
            combined_inputs.append(combined_input)
            input_sizes.append(expected_size)
            output_sizes.append(subsystem_model.output_size)
            
        # Now group by input size and output size for parallel processing
        # This way we can batch process subsystems with the same input/output dimensions
        groups = {}
        for i, (term_id, model, inp_size, out_size, combined_input) in enumerate(
            zip(term_ids, subsystem_models, input_sizes, output_sizes, combined_inputs)
        ):
            key = (inp_size, out_size)
            if key not in groups:
                groups[key] = []
            groups[key].append((i, term_id, model, combined_input))
        
        # Process each group in batches
        for (inp_size, out_size), group in groups.items():
            indices = [g[0] for g in group]
            batch_term_ids = [g[1] for g in group]
            batch_models = [g[2] for g in group]
            batch_inputs = [g[3] for g in group]
            
            # Skip batch processing for single subsystems
            if len(batch_inputs) == 1:
                term_id = batch_term_ids[0]
                model = batch_models[0]
                combined_input = batch_inputs[0]
                output = model(combined_input)
                subsystem_outputs[term_id] = output
                continue
            
            # Process in batches when we have multiple subsystems with the same dimensions
            # This will use GPU parallelism to speed up computation
            try:
                # Stack inputs (these all have the same shape)
                batch_size = batch_inputs[0].size(0)
                num_subsystems = len(batch_inputs)
                
                # Stack inputs into a single tensor
                stacked_inputs = torch.cat(batch_inputs, dim=0)
                
                # For simple models (1 layer), we can optimize with a batched matrix multiply
                if all(model.num_layers == 1 for model in batch_models):
                    # Stack all model weights and biases for the first layer
                    weights = torch.stack([model.layers[0].weight for model in batch_models])
                    biases = torch.stack([model.layers[0].bias for model in batch_models])
                    
                    # Create a big batch multiplication
                    # Reshape inputs to (num_subsystems, batch_size, input_size)
                    reshaped_inputs = stacked_inputs.view(num_subsystems, batch_size, inp_size)
                    
                    # Perform batch matrix multiplication
                    # weights: (num_subsystems, out_size, inp_size)
                    # reshaped_inputs: (num_subsystems, batch_size, inp_size)
                    # Result: (num_subsystems, batch_size, out_size)
                    outputs = torch.bmm(reshaped_inputs, weights.transpose(1, 2))
                    
                    # Add biases: biases is (num_subsystems, out_size)
                    # Need to reshape to (num_subsystems, 1, out_size) for broadcasting
                    outputs = outputs + biases.unsqueeze(1)
                    
                    # Apply activation and normalization
                    normalized_outputs = []
                    for i, model in enumerate(batch_models):
                        # Extract this model's output
                        output = outputs[i]  # Shape: (batch_size, out_size)
                        
                        # Apply activation
                        output = model.activation(output)
                        
                        # Apply normalization if needed
                        if model.norms[0] is not None and model.norm_type != "none":
                            # Skip BatchNorm for single samples
                            if model.norm_type == "batch" and output.size(0) == 1:
                                pass
                            else:
                                output = model.norms[0](output)
                        
                        normalized_outputs.append(output)
                    
                    # Store the results
                    for i, term_id in enumerate(batch_term_ids):
                        subsystem_outputs[term_id] = normalized_outputs[i]
                
                # For multi-layer models, process each subsystem individually
                else:
                    for i, (term_id, model, combined_input) in enumerate(zip(batch_term_ids, batch_models, batch_inputs)):
                        output = model(combined_input)
                        subsystem_outputs[term_id] = output
                        
            except Exception as e:
                # Fallback to sequential processing if batched processing fails
                print(f"Batch processing failed with error: {e}. Falling back to sequential processing.")
                for i, (term_id, model, combined_input) in enumerate(zip(batch_term_ids, batch_models, batch_inputs)):
                    output = model(combined_input)
                    subsystem_outputs[term_id] = output
    
    # Get the root output
    if "GO:ROOT" in subsystem_outputs:
        root_output = subsystem_outputs["GO:ROOT"]
    else:
        raise ValueError("Root node 'GO:ROOT' not found in outputs")
    
    return root_output, {"subsystem_outputs": subsystem_outputs}