import torch
from torch import Tensor
from torchmetrics.metric import Metric
from typing import Tuple, Literal, Optional, Any


class NaNTolerantMetricBase(Metric):
    def __init__(self, **kwargs):
        # Configure for DDP compatibility
        kwargs["compute_on_cpu"] = False
        kwargs["sync_on_compute"] = False
        kwargs["dist_sync_on_step"] = True
        super().__init__(**kwargs)
        self.register_buffer("_device_buffer", torch.zeros(1))

    def _track_device(self, tensor: Tensor) -> None:
        if tensor.device != self._device_buffer.device:
            self._device_buffer = self._device_buffer.to(tensor.device)

    def _create_tensor_on_device(self, value, *shape):
        return torch.full(shape, value, device=self._device_buffer.device)

    def _prepare_inputs(self, preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """Handle NaN values and convert predictions to appropriate format."""
        self._track_device(preds)
        device = self._device_buffer.device

        if target.numel() == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        # Handle NaN values
        if target.dim() > 1:
            valid_mask = ~torch.isnan(target).any(dim=-1)
        else:
            valid_mask = ~torch.isnan(target)

        if not valid_mask.any():
            return torch.empty(0, device=device), torch.empty(0, device=device)

        # Get valid samples
        valid_preds = preds[valid_mask]
        valid_target = target[valid_mask]

        # Handle binary classification
        if valid_target.dim() > 1 and valid_target.size(-1) == 2:
            target_indices = valid_target[:, 1].long()  # Use second column for target
        else:
            target_indices = valid_target.long()

        return valid_preds, target_indices


class NaNTolerantAccuracy(NaNTolerantMetricBase):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, task: Literal["binary", "multiclass"] = "binary", **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._prepare_inputs(preds, target)
        if preds.numel() > 0:
            if self.task == "binary":
                preds = torch.softmax(preds, dim=-1)
                pred_classes = torch.argmax(preds, dim=1)
            else:
                pred_classes = torch.argmax(preds, dim=1)

            self.correct += (pred_classes == target).sum()
            self.total += target.numel()

    def compute(self) -> Tensor:
        if self.total == 0:
            return self._create_tensor_on_device(float("nan"))
        return self.correct.float() / self.total


class NaNTolerantF1Score(NaNTolerantMetricBase):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, task: Literal["binary", "multiclass"] = "binary", **kwargs):
        super().__init__(**kwargs)
        self.task = task
        if task == "binary":
            num_classes = 2
        else:
            num_classes = kwargs.get("num_classes", 2)
            if num_classes is None:
                raise ValueError("num_classes must be provided for multiclass task")

        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._prepare_inputs(preds, target)
        if preds.numel() > 0:
            if self.task == "binary":
                preds = torch.softmax(preds, dim=-1)
                pred_classes = torch.argmax(preds, dim=1)
            else:
                pred_classes = torch.argmax(preds, dim=1)

            target = target.long()
            for i in range(len(self.tp)):
                self.tp[i] += ((pred_classes == i) & (target == i)).sum()
                self.fp[i] += ((pred_classes == i) & (target != i)).sum()
                self.fn[i] += ((pred_classes != i) & (target == i)).sum()

    def compute(self) -> Tensor:
        if self.tp.sum() == 0:
            return self._create_tensor_on_device(float("nan"))

        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        if self.task == "binary":
            return f1[1]  # Return F1 for positive class only
        return f1.mean()  # Return macro F1 for multiclass


class NaNTolerantAUROC(NaNTolerantMetricBase):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, task: Literal["binary"] = "binary", **kwargs):
        if task != "binary":
            raise ValueError("AUROC currently only supports binary classification")
        super().__init__(**kwargs)

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._prepare_inputs(preds, target)
        if preds.numel() > 0:
            preds = torch.softmax(preds, dim=-1)[
                :, 1
            ]  # Get probability of positive class
            self.preds.append(preds)
            self.targets.append(target)

    def compute(self) -> Tensor:
        if len(self.preds) == 0 or len(self.targets) == 0:
            return self._create_tensor_on_device(float("nan"))

        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        # Need at least one sample of each class for AUROC
        if not (targets.bool().any() and (~targets.bool()).any()):
            return self._create_tensor_on_device(float("nan"))

        # Sort predictions and corresponding targets
        sorted_indices = torch.argsort(preds, descending=True)
        sorted_targets = targets[sorted_indices].float()

        # Compute TPR and FPR
        tps = torch.cumsum(sorted_targets, 0)
        fps = torch.cumsum(~sorted_targets.bool().float(), 0)

        tpr = tps / (tps[-1] + 1e-10)
        fpr = fps / (fps[-1] + 1e-10)

        # Compute AUC using trapezoidal rule
        width = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1]) / 2
        auc = torch.sum(width * height)

        return auc
