import torch
from torch import Tensor

from torchmetrics.classification import Accuracy, F1Score, AUROC


from typing import Tuple


class NaNTolerantMetricBase:
    def _prepare_inputs(self, preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert logits to probabilities and handle NaN values."""
        valid_mask = ~torch.isnan(target)
        if valid_mask.any():
            preds = preds[valid_mask]
            target = target[valid_mask]
            binary_preds = torch.sigmoid(preds)
            return binary_preds, target
        return preds, target


class NaNTolerantF1Score(F1Score, NaNTolerantMetricBase):
    def __init__(self, task="binary", **kwargs):
        super().__init__(task=task, **kwargs)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        preds, target = self._prepare_inputs(preds, target)
        return super().forward(preds, target)

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._prepare_inputs(preds, target)
        super().update(preds, target)


class NaNTolerantAccuracy(Accuracy, NaNTolerantMetricBase):
    def __init__(self, task="binary", **kwargs):
        super().__init__(task=task, **kwargs)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        preds, target = self._prepare_inputs(preds, target)
        return super().forward(preds, target)

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._prepare_inputs(preds, target)
        super().update(preds, target)


class NaNTolerantAUROC(AUROC, NaNTolerantMetricBase):
    def __init__(self, task="binary", **kwargs):
        super().__init__(task=task, **kwargs)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        preds, target = self._prepare_inputs(preds, target)
        return super().forward(preds, target)

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._prepare_inputs(preds, target)
        super().update(preds, target)
