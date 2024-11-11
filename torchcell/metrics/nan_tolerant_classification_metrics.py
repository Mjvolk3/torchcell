import torch
from torch import Tensor
from torchmetrics.metric import Metric
from typing import Any, Tuple
from torch.nn import functional as F


def _handle_nan_mask(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Create mask for non-NaN values and return valid predictions and targets."""
    valid_mask = ~torch.isnan(preds) & ~torch.isnan(target)
    valid_preds = preds[valid_mask]
    valid_target = target[valid_mask]
    return valid_mask, valid_preds, valid_target


def _nan_tolerant_binary_update(
    preds: Tensor, target: Tensor, threshold: Tensor, num_outputs: int = 1
) -> tuple[Tensor, Tensor, int]:
    """Update state with predictions and targets for binary metrics, handling NaN values."""
    valid_mask, valid_preds, valid_target = _handle_nan_mask(preds, target)

    if valid_mask.any():
        # Apply sigmoid to valid predictions
        valid_preds = torch.sigmoid(valid_preds)

        # Binarize targets using threshold
        valid_target = (valid_target > threshold).float()

        if num_outputs > 1:
            true_positives = torch.sum(
                (valid_preds > 0.5) & (valid_target > 0.5), dim=0
            )
            true_negatives = torch.sum(
                (valid_preds <= 0.5) & (valid_target <= 0.5), dim=0
            )
            false_positives = torch.sum(
                (valid_preds > 0.5) & (valid_target <= 0.5), dim=0
            )
            false_negatives = torch.sum(
                (valid_preds <= 0.5) & (valid_target > 0.5), dim=0
            )
            num_observations = torch.tensor(valid_mask.size(0), device=preds.device)
        else:
            true_positives = torch.sum((valid_preds > 0.5) & (valid_target > 0.5))
            true_negatives = torch.sum((valid_preds <= 0.5) & (valid_target <= 0.5))
            false_positives = torch.sum((valid_preds > 0.5) & (valid_target <= 0.5))
            false_negatives = torch.sum((valid_preds <= 0.5) & (valid_target > 0.5))
            num_observations = torch.tensor(
                valid_mask.sum().item(), device=preds.device
            )
    else:
        true_positives = torch.zeros(num_outputs, device=preds.device)
        true_negatives = torch.zeros(num_outputs, device=preds.device)
        false_positives = torch.zeros(num_outputs, device=preds.device)
        false_negatives = torch.zeros(num_outputs, device=preds.device)
        num_observations = torch.tensor(0, device=preds.device)

    return (
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
    ), num_observations


class NaNTolerantAccuracy(Metric):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, threshold=torch.tensor([0.9305, -0.0018]), **kwargs: Any):
        super().__init__(**kwargs)
        self.register_buffer("threshold", threshold)
        self.add_state("correct", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        # Get index from input shape
        index = 0 if preds.ndim == 1 else 1
        threshold_idx = 0 if index == 0 else preds.shape[1]-1

        # Handle NaN values
        valid_mask = ~torch.isnan(target)
        if valid_mask.any():
            valid_preds = preds[valid_mask]
            valid_target = target[valid_mask]

            # Apply sigmoid to predictions
            valid_preds = torch.sigmoid(valid_preds)
            
            # Get correct threshold
            threshold = self.threshold[threshold_idx].to(valid_target.device)
            
            # Binarize target using threshold
            valid_target = (valid_target > threshold).float()
            
            # Compare predictions
            correct_preds = (valid_preds > 0.5) == valid_target
            self.correct += torch.sum(correct_preds)
            self.total += valid_mask.sum()

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(float("nan"), device=self.threshold.device)
        return self.correct.float() / self.total


class NaNTolerantF1Score(Metric):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, threshold=torch.tensor([0.9305, -0.0018]), **kwargs: Any):
        super().__init__(**kwargs)
        self.register_buffer("threshold", threshold)
        self.add_state("true_positives", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        # Get index from input shape
        index = 0 if preds.ndim == 1 else 1
        threshold_idx = 0 if index == 0 else preds.shape[1]-1

        # Handle NaN values
        valid_mask = ~torch.isnan(target)
        if valid_mask.any():
            valid_preds = preds[valid_mask]
            valid_target = target[valid_mask]

            # Apply sigmoid to predictions
            valid_preds = torch.sigmoid(valid_preds)
            
            # Get correct threshold
            threshold = self.threshold[threshold_idx].to(valid_target.device)
            
            # Binarize target using threshold
            valid_target = (valid_target > threshold).float()
            
            # Binary predictions
            binary_preds = (valid_preds > 0.5).float()
            
            self.true_positives += torch.sum((binary_preds == 1) & (valid_target == 1))
            self.false_positives += torch.sum((binary_preds == 1) & (valid_target == 0))
            self.false_negatives += torch.sum((binary_preds == 0) & (valid_target == 1))

    def compute(self) -> Tensor:
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-10)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1


class NaNTolerantAUROC(Metric):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, threshold=torch.tensor([0.9305, -0.0018]), **kwargs: Any):
        super().__init__(**kwargs)
        self.register_buffer("threshold", threshold)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        # Get index from input shape
        index = 0 if preds.ndim == 1 else 1
        threshold_idx = 0 if index == 0 else preds.shape[1]-1

        # Handle NaN values
        valid_mask = ~torch.isnan(target)
        if valid_mask.any():
            valid_preds = preds[valid_mask]
            valid_target = target[valid_mask]

            # Apply sigmoid to predictions
            valid_preds = torch.sigmoid(valid_preds)
            
            # Get correct threshold
            threshold = self.threshold[threshold_idx].to(valid_target.device)
            
            # Binarize target using threshold
            valid_target = (valid_target > threshold).float()
            
            self.preds.append(valid_preds)
            self.target.append(valid_target)

    def compute(self) -> Tensor:
        if not self.preds:
            return torch.tensor(float("nan"), device=self.threshold.device)
        
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)
        
        # Sort predictions and corresponding targets
        sorted_indices = torch.argsort(preds, descending=True)
        sorted_target = target[sorted_indices]
        
        # Calculate TPR and FPR at each threshold
        tpr = torch.cumsum(sorted_target, 0) / (torch.sum(sorted_target) + 1e-10)
        fpr = torch.cumsum(1 - sorted_target, 0) / (torch.sum(1 - sorted_target) + 1e-10)
        
        return torch.trapz(tpr, fpr)