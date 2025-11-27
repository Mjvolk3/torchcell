import torch
from torch.optim import AdamW
from torchcell.scheduler.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
import yaml
import sys

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

# Load YAML configuration
yaml_path = "/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/conf/equivariant_cell_graph_transformer_delta_008.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract lr_scheduler configuration
lr_scheduler_config = config.get('regression_task', {}).get('lr_scheduler')

# Extract optimizer configuration
optimizer_config = config.get('regression_task', {}).get('optimizer', {})
optimizer_lr = float(optimizer_config.get('lr', 1e-4))

# Create dummy model for optimizer
model = torch.nn.Linear(10, 1)
optimizer = AdamW(model.parameters(), lr=optimizer_lr)

# Get max_epochs from trainer configuration
max_epochs = config.get('trainer', {}).get('max_epochs', 300)
epochs = max_epochs

# Check if lr_scheduler configuration exists
if lr_scheduler_config is None:
    print("lr_scheduler is not configured in the YAML file.")
    sys.exit(0)

# Get scheduler type
scheduler_type = lr_scheduler_config.get('type', 'CosineAnnealingWarmupRestarts')

if scheduler_type == 'Constant':
    print(f"Using constant learning rate: {optimizer_lr:.1e}")
    scheduler = None
    # Set default values for variables used in plotting
    first_cycle_steps = 30
    num_cycles = 10
elif scheduler_type == 'CosineAnnealingWarmupRestarts':
    # Extract scheduler parameters from YAML
    first_cycle_steps = int(lr_scheduler_config.get('first_cycle_steps', 30))
    cycle_mult = float(lr_scheduler_config.get('cycle_mult', 1.0))
    max_lr = float(lr_scheduler_config.get('max_lr', 1e-3))
    min_lr = float(lr_scheduler_config.get('min_lr', 1e-8))
    warmup_steps = int(lr_scheduler_config.get('warmup_steps', 5))
    gamma = float(lr_scheduler_config.get('gamma', 0.6))
    
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=cycle_mult,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        gamma=gamma,
    )
    num_cycles = epochs // first_cycle_steps
else:
    print(f"Scheduler type '{scheduler_type}' not supported yet.")
    sys.exit(0)

# Track learning rates
lrs = []

# Don't force initial LR - let the scheduler handle it
# The CosineAnnealingWarmupRestarts initializes to min_lr
lrs.append(optimizer.param_groups[0]["lr"])

dummy_param = torch.tensor([1.0], requires_grad=True)

for epoch in range(epochs - 1):
    optimizer.zero_grad()
    loss = dummy_param.sum()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])

# Main plot
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(lrs, linewidth=2.5, color="blue")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Learning Rate", fontsize=12)
if scheduler_type == "Constant":
    ax.set_title(
        f"Constant Learning Rate from YAML - LR: {optimizer_lr:.1e}",
        fontsize=14,
    )
else:
    ax.set_title(
        f"CosineAnnealingWarmupRestarts from YAML - Start: {optimizer_lr:.1e}, Max: {max_lr:.1e}, Min: {min_lr:.1e}, Gamma: {gamma}",
        fontsize=14,
    )
ax.set_yscale("log")
ax.set_ylim([5e-9, 2e-3])
ax.grid(True, alpha=0.3, which="both")

# Mark minima points based on cycle length from config (only for non-constant schedulers)
if scheduler_type != "Constant":
    for i in range(first_cycle_steps, epochs + 1, first_cycle_steps):
        ax.axvline(x=i, color="red", linestyle="--", alpha=0.5, linewidth=1)

# Annotate key values
ax.annotate(
    f"Start: {lrs[0]:.1e}",
    xy=(0, lrs[0]),
    xytext=(10, lrs[0] * 2),
    ha="left",
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="green", alpha=0.7),
)

# Annotate peaks and minima only for non-constant schedulers
if scheduler_type != "Constant":
    num_cycles = epochs // first_cycle_steps
    for cycle in range(num_cycles):
        peak_epoch = cycle * first_cycle_steps  # Peak at start of each cycle
        if peak_epoch < len(lrs):
            peak_lr = lrs[peak_epoch]
            if cycle in [0, num_cycles//2, num_cycles-1]:  # Annotate first, middle, and last peaks
                ax.annotate(
                    f"Peak {cycle+1}: {peak_lr:.1e}",
                    xy=(peak_epoch, peak_lr),
                    xytext=(peak_epoch + 5, peak_lr * 2),
                    ha="left",
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="red", alpha=0.5),
                )

    # Annotate a minimum
    min_epoch = first_cycle_steps - 1  # Just before first cycle ends
    if min_epoch < len(lrs):
        ax.annotate(
            f"Min: {lrs[min_epoch]:.1e}",
            xy=(min_epoch, lrs[min_epoch]),
            xytext=(min_epoch, lrs[min_epoch] / 5),
            ha="center",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.5),
        )

plt.tight_layout()

# Save the main plot
title = "cosine_annealing_warmup_restarts_lr_schedule"
save_path = osp.join(ASSET_IMAGES_DIR, f"{title}_{timestamp()}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved main plot to: {save_path}")
plt.close()

# Detailed analysis plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Full schedule
axes[0, 0].plot(lrs, linewidth=2.5, color="blue")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Learning Rate")
axes[0, 0].set_title(f"Full Schedule ({epochs} epochs)")
axes[0, 0].set_yscale("log")
axes[0, 0].grid(True, alpha=0.3, which="both")

# Plot 2: First 60 epochs (2 cycles)
axes[0, 1].plot(lrs[:60], linewidth=2.5, color="blue")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Learning Rate")
axes[0, 1].set_title("First 2 Cycles (Detail)")
axes[0, 1].set_yscale("log")
axes[0, 1].grid(True, alpha=0.3, which="both")
axes[0, 1].axvline(x=first_cycle_steps, color="red", linestyle="--", alpha=0.5)

# Plot 3: Peak values across cycles (or constant value)
if scheduler_type == "Constant":
    # For constant LR, show the same value across epochs
    constant_values = [optimizer_lr] * 10  # Show 10 points for visualization
    axes[1, 0].scatter(range(len(constant_values)), constant_values, color="blue", s=100)
    axes[1, 0].plot(range(len(constant_values)), constant_values, "b-", linewidth=2)
    axes[1, 0].set_xlabel("Epoch / 30")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title(f"Constant Learning Rate: {optimizer_lr:.1e}")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3, which="both")
    axes[1, 0].set_ylim([optimizer_lr * 0.5, optimizer_lr * 2])
else:
    peak_values = []
    for cycle in range(num_cycles):
        peak_epoch = cycle * first_cycle_steps
        if peak_epoch < len(lrs):
            peak_values.append(lrs[peak_epoch])

    axes[1, 0].scatter(range(len(peak_values)), peak_values, color="red", s=100)
    axes[1, 0].plot(range(len(peak_values)), peak_values, "r--", alpha=0.5)
    axes[1, 0].set_xlabel("Cycle Number")
    axes[1, 0].set_ylabel("Peak Learning Rate")
    axes[1, 0].set_title(f"Peak Decay with Gamma={gamma}")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3, which="both")
    axes[1, 0].set_ylim([5e-6, 2e-3])

    # Add max_lr line
    axes[1, 0].axhline(
        y=max_lr, color="green", linestyle=":", alpha=0.7, label=f"Max LR: {max_lr:.1e}"
    )
    axes[1, 0].legend()

# Plot 4: Min values verification (or constant for constant scheduler)
if scheduler_type == "Constant":
    # For constant LR, show the same value
    constant_values = [optimizer_lr] * 10  # Show 10 points for visualization
    axes[1, 1].scatter(range(len(constant_values)), constant_values, color="blue", s=100)
    axes[1, 1].plot(range(len(constant_values)), constant_values, "b-", linewidth=2)
    axes[1, 1].set_xlabel("Epoch / 30")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Constant Learning Rate (No Minima)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3, which="both")
    axes[1, 1].set_ylim([optimizer_lr * 0.5, optimizer_lr * 2])
else:
    min_values = []
    for i in range(first_cycle_steps-1, len(lrs), first_cycle_steps):  # Minima occur just before each cycle ends
        if i < len(lrs):
            min_values.append(lrs[i])

    axes[1, 1].scatter(range(len(min_values)), min_values, color="blue", s=100)
    axes[1, 1].set_xlabel("Cycle Number")
    axes[1, 1].set_ylabel("Minimum Learning Rate")
    axes[1, 1].set_title(f"Minimum Values (Target: {min_lr:.1e})")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3, which="both")
    axes[1, 1].axhline(
        y=min_lr, color="green", linestyle=":", alpha=0.7, label=f"Target: {min_lr:.1e}"
    )
    axes[1, 1].legend()

plt.tight_layout()

# Save the detailed analysis plot
title = "cosine_annealing_warmup_restarts_detailed_analysis"
save_path = osp.join(ASSET_IMAGES_DIR, f"{title}_{timestamp()}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved detailed analysis plot to: {save_path}")
plt.close()

print(f"Starting value: {lrs[0]:.2e}")

if scheduler_type == "Constant":
    print(f"Constant LR: {optimizer_lr:.2e}")
else:
    print(f"First peak: {peak_values[0]:.2e}")
    print(f"Last peak: {peak_values[-1]:.2e}")
    print(f"Min values: {min_values[0]:.2e} to {min_values[-1]:.2e}")

print(f"\nConfiguration loaded from:")
print(f"  {yaml_path}")
print(f"\nScheduler parameters:")
print(f"  Type: {scheduler_type}")

if scheduler_type == "Constant":
    print(f"  Learning rate: {optimizer_lr:.1e}")
else:
    print(f"  First cycle steps: {first_cycle_steps}")
    print(f"  Cycle multiplier: {cycle_mult}")
    print(f"  Max LR: {max_lr:.1e}")
    print(f"  Min LR: {min_lr:.1e}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Gamma: {gamma}")
