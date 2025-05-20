#!/usr/bin/env python3
# Test script for the simplified DCell implementation

import sys
import os
import torch
import time
from torch_geometric.data import HeteroData
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from torchcell.scratch.load_batch_005 import load_sample_data_batch
from torchcell.models.dcell import DCellModel
from torchcell.losses.dcell import DCellLoss
from torchcell.timestamp import timestamp

# Load environment variables for saving figures
load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "assets/images")
os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)


def verify_tensors_on_device(name, obj, device, depth=0):
    """Verify recursively that all tensors in an object are on the specified device."""
    indent = "  " * depth
    if isinstance(obj, torch.Tensor):
        tensor_device = obj.device
        is_correct = tensor_device == device
        print(f"{indent}{name}: {'✓' if is_correct else '❌'} on {tensor_device}")
        return is_correct
    elif isinstance(obj, dict):
        all_correct = True
        for key, value in obj.items():
            if not verify_tensors_on_device(f"{name}[{key}]", value, device, depth + 1):
                all_correct = False
        return all_correct
    elif isinstance(obj, (list, tuple)):
        all_correct = True
        for i, item in enumerate(obj):
            if not verify_tensors_on_device(f"{name}[{i}]", item, device, depth + 1):
                all_correct = False
        return all_correct
    elif hasattr(obj, "to") and hasattr(obj, "__dict__"):
        # For objects with a 'to' method (like HeteroData)
        all_correct = True
        for attr_name, attr_value in vars(obj).items():
            if torch.is_tensor(attr_value):
                is_correct = attr_value.device == device
                print(f"{indent}{name}.{attr_name}: {'✓' if is_correct else '❌'} on {attr_value.device}")
                if not is_correct:
                    all_correct = False
        return all_correct
    return True


def main():
    # Configure parameters
    batch_size = 8
    norm_type = "batch"
    subsystem_num_layers = 1
    activation = torch.nn.Tanh()

    # Test on both CPU and available GPU
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        print(f"\n{'='*80}")
        print(f"TESTING SIMPLIFIED DCELL ON {device.type.upper()}")
        print(f"{'='*80}")

        # Load test data
        print("\nLoading test data...")
        dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
            batch_size=batch_size, num_workers=0, config="dcell", is_dense=False
        )

        # Move data to device
        cell_graph = dataset.cell_graph.to(device)
        batch = batch.to(device)

        print(f"Dataset: {len(dataset)} samples")
        print(f"Batch: {batch.num_graphs} graphs")
        print(f"Max Number of Nodes: {max_num_nodes}")

        # Initialize model
        print("\nInitializing DCellModel...")
        model_params = {
            "gene_num": max_num_nodes,
            "subsystem_output_min": 20,
            "subsystem_output_max_mult": 0.3,
            "output_size": 1,
            "norm_type": norm_type,
            "subsystem_num_layers": subsystem_num_layers,
            "activation": activation,
        }

        start_time = time.time()
        model = DCellModel(**model_params).to(device)
        init_time = time.time() - start_time
        print(f"Model initialization time: {init_time:.3f}s")

        # Create the target tensor
        target = batch["gene"].phenotype_values.view_as(
            torch.zeros(batch.num_graphs, 1, device=device)
        )

        # Create loss function
        criterion = DCellLoss(alpha=0.3, use_auxiliary_losses=True)

        # Run the model multiple times to ensure stable behavior
        print("\nRunning forward pass with model...")
        for i in range(3):
            # Forward pass
            forward_start = time.time()
            predictions, outputs = model(cell_graph, batch)
            forward_time = time.time() - forward_start
            
            # Compute loss 
            loss, loss_components = criterion(predictions, outputs, target)
            
            # Get prediction metrics
            pred_diversity = predictions.std().item()
            
            # Print metrics
            print(f"Run {i+1}: Forward time: {forward_time:.3f}s, Loss: {loss.item():.6f}, Diversity: {pred_diversity:.6f}")
            
            # Verify device consistency
            print("\nVerifying tensor devices:")
            outputs_on_device = verify_tensors_on_device("outputs", outputs, device)
            predictions_on_device = verify_tensors_on_device("predictions", predictions, device)
            
            if outputs_on_device and predictions_on_device:
                print(f"\n✅ All tensors verified to be on {device}")
            else:
                print(f"\n❌ Some tensors are not on {device}")

        # Test training
        print("\nTesting a few training steps...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Track loss history
        history = {"loss": [], "primary_loss": [], "auxiliary_loss": []}
        
        # Training loop
        for epoch in range(5):
            # Forward pass
            predictions, outputs = model(cell_graph, batch)
            
            # Compute loss
            loss, loss_components = criterion(predictions, outputs, target)
            
            # Store loss components
            history["loss"].append(loss.item())
            history["primary_loss"].append(loss_components["primary_loss"].item())
            history["auxiliary_loss"].append(loss_components["auxiliary_loss"].item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
        
        # Create a simple plot
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(history["loss"])+1))
        plt.plot(epochs, history["loss"], 'b-', label='Total Loss')
        plt.plot(epochs, history["primary_loss"], 'r-', label='Primary Loss')
        plt.plot(epochs, history["auxiliary_loss"], 'g-', label='Auxiliary Loss')
        
        plt.title(f"DCell Loss Components on {device.type.upper()}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        save_path = os.path.join(ASSET_IMAGES_DIR, f"dcell_simple_test_{device.type}_{timestamp()}.png")
        plt.savefig(save_path)
        print(f"\nLoss plot saved to {save_path}")
        plt.close()
    
    print("\nAll tests complete!")


if __name__ == "__main__":
    main()