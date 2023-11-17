# src/torchcell/losses/dcell.py
# [[src.torchcell.losses.dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/losses/dcell.py
# Test file: src/torchcell/losses/test_dcell.py

import torch
import torch.nn as nn


def dcell_loss(outputs, targets, weights, alpha=0.3, lambda_reg=0.01):
    criterion = nn.MSELoss()
    root_loss = criterion(outputs["root"], targets)

    non_root_loss = sum(criterion(outputs[t], targets) for t in outputs if t != "root")

    reg_loss = lambda_reg * torch.norm(weights, 2)

    total_loss = root_loss + alpha * non_root_loss + reg_loss
    return total_loss / len(targets)
