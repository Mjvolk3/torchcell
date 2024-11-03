import torch
from torch import Tensor
from torchmetrics import (
    Metric,
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)


class NaNTolerantMetric(Metric):
    def __init__(
        self, base_metric_class, base_metric_kwargs=None, default_value=torch.nan
    ):
        super().__init__()
        self.base_metric = base_metric_class(**(base_metric_kwargs or {}))
        self.default_value = default_value

    def update(self, preds: Tensor, target: Tensor) -> None:
        # Create mask for non-NaN values
        mask = ~torch.isnan(preds) & ~torch.isnan(target)

        # If there are any non-NaN values, update the metric
        if torch.any(mask):
            self.base_metric.update(preds[mask], target[mask])

    def compute(self) -> Tensor:
        try:
            # Compute the metric if there were valid samples
            result = self.base_metric.compute()
        except ValueError:
            # If no valid samples, return default_value
            result = self.default_value
        return result


class NaNTolerantMSE(NaNTolerantMetric):
    is_differentiable = True
    higher_is_better = False

    def __init__(self, squared: bool = True):
        super().__init__(MeanSquaredError, {"squared": squared})


class NaNTolerantMAE(NaNTolerantMetric):
    is_differentiable = True
    higher_is_better = False

    def __init__(self):
        super().__init__(MeanAbsoluteError)


class NaNTolerantRMSE(NaNTolerantMetric):
    is_differentiable = True
    higher_is_better = False

    def __init__(self):
        super().__init__(MeanSquaredError, {"squared": False})


class NaNTolerantPearsonCorrCoef(NaNTolerantMetric):
    is_differentiable = True
    higher_is_better = True

    def __init__(self):
        super().__init__(PearsonCorrCoef)


class NaNTolerantSpearmanCorrCoef(NaNTolerantMetric):
    is_differentiable = False
    higher_is_better = True

    def __init__(self):
        super().__init__(SpearmanCorrCoef)
