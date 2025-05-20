#!/usr/bin/env python3
# Test script to verify DCell device handling

import sys
import os
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Import required modules
from torchcell.models.dcell import DCellModel
from torchcell.scratch.load_batch_005 import load_sample_data_batch

def test_dcell_device():
    """
    Test that verifies DCell can properly run on GPU with all tensors on the correct device.
    """
    print("\n=" * 40)
    print("DCELL DEVICE TEST")
    print("=" * 40)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use smaller batch size for testing
    batch_size = 2
    num_workers = 0  # Minimize potential data loading issues

    # Load test data
    print("\nLoading test data...")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=batch_size, 
        num_workers=num_workers, 
        config="dcell", 
        is_dense=False
    )

    # Move data to device
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Print dataset information
    print(f"Dataset: {len(dataset)} samples")
    print(f"Batch: {batch.num_graphs} graphs")
    print(f"Cell graph device: {cell_graph['gene'].x.device}")
    print(f"Batch device: {batch['gene'].phenotype_values.device}")

    # Initialize model with default parameters
    print("\nCreating DCellModel...")
    model_params = {
        "gene_num": max_num_nodes,
        "subsystem_output_min": 20,
        "subsystem_output_max_mult": 0.3,
        "output_size": 1,
        "norm_type": "batch", 
        "norm_before_act": False,
        "subsystem_num_layers": 1, 
        "activation": torch.nn.Tanh(),
        "init_range": 0.001,
    }

    # Initialize model (device handling now happens inside model)
    model = DCellModel(**model_params)
    print(f"Model will use device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Run forward pass to initialize the model
    print("\nRunning initial forward pass to initialize model...")
    with torch.no_grad():
        try:
            predictions, outputs = model(cell_graph, batch)
            
            # Check prediction device
            print(f"Predictions device: {predictions.device}")
            print(f"Root output device: {outputs['subsystem_outputs']['GO:ROOT'].device}")
            
            # Check diversity
            diversity = predictions.std().item()
            print(f"Initial predictions diversity: {diversity:.6f}")
            
            if diversity < 1e-6:
                print("WARNING: Predictions lack diversity!")
            else:
                print("✓ Predictions are diverse")
                
            # Success message
            print("\n✓ Forward pass succeeded with all tensors on correct device")
            return True
            
        except RuntimeError as e:
            print(f"ERROR: Forward pass failed with error: {e}")
            return False

if __name__ == "__main__":
    success = test_dcell_device()
    if success:
        print("\nTest passed! 🎉")
        sys.exit(0)
    else:
        print("\nTest failed! 😞")
        sys.exit(1)