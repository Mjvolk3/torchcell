# src/torchcell/losses/dcell.py
# [[src.torchcell.losses.dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/losses/dcell.py
# Test file: src/torchcell/losses/test_dcell.py

import torch
import torch.nn as nn


class DCellLoss(nn.Module):
    def __init__(self, alpha=0.3, lambda_reg=0.01):
        super().__init__()
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.criterion = nn.MSELoss()

    def forward(self, outputs, target, weights):
        # Assuming 'GO:ROOT' is the root subsystem
        root_output = outputs["GO:ROOT"]
        root_loss = self.criterion(root_output.squeeze(-1), target)

        # Loss for non-root subsystems
        non_root_loss = sum(
            self.criterion(outputs[t].squeeze(-1), target)
            for t in outputs
            if t != "GO:ROOT"
        )

        # Regularization loss for weights
        all_weights = torch.cat([w.view(-1) for w in weights])
        reg_loss = self.lambda_reg * torch.norm(all_weights, 2)

        # Total loss
        total_loss = root_loss + self.alpha * non_root_loss + reg_loss
        return total_loss
