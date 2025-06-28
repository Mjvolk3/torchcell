import torch
from torch.optim import AdamW
from torchcell.scheduler.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.timestamp import timestamp

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

# Create dummy model for optimizer
model = torch.nn.Linear(10, 1)

# Calculate gamma for desired decay
# Want peaks to go from 1e-3 to 1e-5 over 10 cycles
# 1e-5 = 1e-3 * gamma^9
gamma_calculated = (1e-5 / 1e-3) ** (1 / 9)

# Main configuration
optimizer = AdamW(model.parameters(), lr=1e-3)

scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=30,  # Minima every 30 epochs
    cycle_mult=1.0,  # Keep cycles the same length
    max_lr=1e-3,  # Starting peak
    min_lr=1e-8,  # Minimum value
    warmup_steps=0,  # No warmup to start at max
    gamma=gamma_calculated,  # ~0.599 for desired decay
)

# Track learning rates
lrs = []
epochs = 300

# Force initial LR to be max_lr
optimizer.param_groups[0]["lr"] = 1e-3
lrs.append(optimizer.param_groups[0]["lr"])

dummy_param = torch.tensor([1.0], requires_grad=True)

for epoch in range(epochs - 1):
    optimizer.zero_grad()
    loss = dummy_param.sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])

# Main plot
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(lrs, linewidth=2.5, color="blue")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Learning Rate", fontsize=12)
ax.set_title(
    f"CosineAnnealingWarmupRestarts - Start: 1e-3, Peaks: 1e-3 → 1e-5, Mins: 1e-8",
    fontsize=14,
)
ax.set_yscale("log")
ax.set_ylim([5e-9, 2e-3])
ax.grid(True, alpha=0.3, which="both")

# Mark minima points
for i in range(30, epochs + 1, 30):
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

# Annotate peaks
for cycle in range(10):
    peak_epoch = cycle * 30  # Peak at start of each cycle
    if peak_epoch < len(lrs):
        peak_lr = lrs[peak_epoch]
        if cycle in [0, 4, 9]:  # Annotate first, middle, and last peaks
            ax.annotate(
                f"Peak {cycle+1}: {peak_lr:.1e}",
                xy=(peak_epoch, peak_lr),
                xytext=(peak_epoch + 5, peak_lr * 2),
                ha="left",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.5),
            )

# Annotate a minimum
min_epoch = 29  # Just before first cycle ends
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
axes[0, 0].set_title("Full Schedule (300 epochs)")
axes[0, 0].set_yscale("log")
axes[0, 0].grid(True, alpha=0.3, which="both")

# Plot 2: First 60 epochs (2 cycles)
axes[0, 1].plot(lrs[:60], linewidth=2.5, color="blue")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Learning Rate")
axes[0, 1].set_title("First 2 Cycles (Detail)")
axes[0, 1].set_yscale("log")
axes[0, 1].grid(True, alpha=0.3, which="both")
axes[0, 1].axvline(x=30, color="red", linestyle="--", alpha=0.5)

# Plot 3: Peak values across cycles
peak_values = []
for cycle in range(10):
    peak_epoch = cycle * 30
    if peak_epoch < len(lrs):
        peak_values.append(lrs[peak_epoch])

axes[1, 0].scatter(range(len(peak_values)), peak_values, color="red", s=100)
axes[1, 0].plot(range(len(peak_values)), peak_values, "r--", alpha=0.5)
axes[1, 0].set_xlabel("Cycle Number")
axes[1, 0].set_ylabel("Peak Learning Rate")
axes[1, 0].set_title("Peak Decay: 1e-3 → 1e-5")
axes[1, 0].set_yscale("log")
axes[1, 0].grid(True, alpha=0.3, which="both")
axes[1, 0].set_ylim([5e-6, 2e-3])

# Add target lines
axes[1, 0].axhline(
    y=1e-3, color="green", linestyle=":", alpha=0.7, label="Target start"
)
axes[1, 0].axhline(y=1e-5, color="orange", linestyle=":", alpha=0.7, label="Target end")
axes[1, 0].legend()

# Plot 4: Min values verification
min_values = []
for i in range(29, len(lrs), 30):  # Minima occur at epochs 29, 59, 89, etc.
    if i < len(lrs):
        min_values.append(lrs[i])

axes[1, 1].scatter(range(len(min_values)), min_values, color="blue", s=100)
axes[1, 1].set_xlabel("Cycle Number")
axes[1, 1].set_ylabel("Minimum Learning Rate")
axes[1, 1].set_title("Minimum Values (should be ~1e-8)")
axes[1, 1].set_yscale("log")
axes[1, 1].grid(True, alpha=0.3, which="both")
axes[1, 1].axhline(
    y=1e-8, color="green", linestyle=":", alpha=0.7, label="Target: 1e-8"
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
print(f"First peak: {peak_values[0]:.2e}")
print(f"Last peak: {peak_values[-1]:.2e}")
print(f"Min values: {min_values[0]:.2e} to {min_values[-1]:.2e}")
print(f"Gamma: {gamma_calculated:.6f}")
