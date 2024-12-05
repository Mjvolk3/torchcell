import torch
from torch import Tensor
from torchmetrics.metric import Metric
from typing import Any
from torch.nn import functional as F
from typing import Tuple


def _handle_nan_mask(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Create mask for non-NaN values and return valid predictions and targets."""
    valid_mask = ~torch.isnan(preds) & ~torch.isnan(target)
    valid_preds = preds[valid_mask]
    valid_target = target[valid_mask]
    return valid_mask, valid_preds, valid_target


def _nan_tolerant_mse_update(
    preds: Tensor, target: Tensor, num_outputs: int = 1
) -> tuple[Tensor, int]:
    """Update state with predictions and targets, handling NaN values."""
    valid_mask, valid_preds, valid_target = _handle_nan_mask(preds, target)

    if valid_mask.any():
        if num_outputs > 1:
            sum_squared_error = torch.sum((valid_preds - valid_target) ** 2, dim=0)
            num_observations = torch.tensor(valid_mask.size(0), device=preds.device)
        else:
            sum_squared_error = torch.sum((valid_preds - valid_target) ** 2)
            num_observations = torch.tensor(
                valid_mask.sum().item(), device=preds.device
            )
    else:
        sum_squared_error = torch.zeros(num_outputs, device=preds.device)
        num_observations = torch.tensor(0, device=preds.device)

    return sum_squared_error, num_observations


def _nan_tolerant_mae_update(
    preds: Tensor, target: Tensor, num_outputs: int = 1
) -> tuple[Tensor, int]:
    """Update state with predictions and targets, handling NaN values."""
    valid_mask, valid_preds, valid_target = _handle_nan_mask(preds, target)

    if valid_mask.any():
        if num_outputs > 1:
            sum_abs_error = torch.sum(torch.abs(valid_preds - valid_target), dim=0)
            num_observations = torch.tensor(valid_mask.size(0), device=preds.device)
        else:
            sum_abs_error = torch.sum(torch.abs(valid_preds - valid_target))
            num_observations = torch.tensor(
                valid_mask.sum().item(), device=preds.device
            )
    else:
        sum_abs_error = torch.zeros(num_outputs, device=preds.device)
        num_observations = torch.tensor(0, device=preds.device)

    return sum_abs_error, num_observations


def _nan_tolerant_error_compute(sum_error: Tensor, num_observations: Tensor) -> Tensor:
    """Compute mean error over state."""
    if num_observations == 0:
        return torch.full_like(sum_error, float("nan"))
    return sum_error / num_observations


# class NaNTolerantMSE(Metric):
#     is_differentiable = True
#     higher_is_better = False
#     full_state_update = False

#     def __init__(self, squared: bool = True, num_outputs: int = 1, **kwargs: Any):
#         super().__init__(**kwargs)
#         self.squared = squared
#         self.num_outputs = num_outputs

#         self.add_state(
#             "sum_squared_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum"
#         )
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, preds: Tensor, target: Tensor) -> None:
#         sum_squared_error, num_obs = _nan_tolerant_mse_update(
#             preds, target, self.num_outputs
#         )
#         self.sum_squared_error += sum_squared_error
#         self.total += num_obs

#     def compute(self) -> Tensor:
#         mean_squared_error = _nan_tolerant_error_compute(
#             self.sum_squared_error, self.total
#         )
#         return mean_squared_error if self.squared else torch.sqrt(mean_squared_error)


class NaNTolerantRMSE(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_outputs: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs

        self.add_state(
            "sum_squared_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        sum_squared_error, num_obs = _nan_tolerant_mse_update(
            preds, target, self.num_outputs
        )
        self.sum_squared_error += sum_squared_error
        self.total += num_obs

    def compute(self) -> Tensor:
        mean_squared_error = _nan_tolerant_error_compute(
            self.sum_squared_error, self.total
        )
        return torch.sqrt(mean_squared_error)


class NaNTolerantMAE(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_outputs: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs

        self.add_state(
            "sum_abs_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        sum_abs_error, num_obs = _nan_tolerant_mae_update(
            preds, target, self.num_outputs
        )
        self.sum_abs_error += sum_abs_error
        self.total += num_obs

    def compute(self) -> Tensor:
        return _nan_tolerant_error_compute(self.sum_abs_error, self.total)


def _final_aggregation(
    means_x: Tensor,
    means_y: Tensor,
    vars_x: Tensor,
    vars_y: Tensor,
    corrs_xy: Tensor,
    nbs: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Aggregate statistics from multiple devices, handling NaN values."""
    if len(means_x) == 1:
        return means_x[0], means_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0]

    mx1, my1, vx1, vy1, cxy1, n1 = (
        means_x[0],
        means_y[0],
        vars_x[0],
        vars_y[0],
        corrs_xy[0],
        nbs[0],
    )
    for i in range(1, len(means_x)):
        mx2, my2, vx2, vy2, cxy2, n2 = (
            means_x[i],
            means_y[i],
            vars_x[i],
            vars_y[i],
            corrs_xy[i],
            nbs[i],
        )
        nb = n1 + n2
        mean_x = (n1 * mx1 + n2 * mx2) / nb
        mean_y = (n1 * my1 + n2 * my2) / nb

        # var_x
        element_x1 = (n1 + 1) * mean_x - n1 * mx1
        vx1 += (element_x1 - mx1) * (element_x1 - mean_x) - (element_x1 - mean_x) ** 2
        element_x2 = (n2 + 1) * mean_x - n2 * mx2
        vx2 += (element_x2 - mx2) * (element_x2 - mean_x) - (element_x2 - mean_x) ** 2
        var_x = vx1 + vx2

        # var_y
        element_y1 = (n1 + 1) * mean_y - n1 * my1
        vy1 += (element_y1 - my1) * (element_y1 - mean_y) - (element_y1 - mean_y) ** 2
        element_y2 = (n2 + 1) * mean_y - n2 * my2
        vy2 += (element_y2 - my2) * (element_y2 - mean_y) - (element_y2 - mean_y) ** 2
        var_y = vy1 + vy2

        # corr
        cxy1 += (element_x1 - mx1) * (element_y1 - mean_y) - (element_x1 - mean_x) * (
            element_y1 - mean_y
        )
        cxy2 += (element_x2 - mx2) * (element_y2 - mean_y) - (element_x2 - mean_x) * (
            element_y2 - mean_y
        )
        corr_xy = cxy1 + cxy2

        mx1, my1, vx1, vy1, cxy1, n1 = mean_x, mean_y, var_x, var_y, corr_xy, nb
    return mean_x, mean_y, var_x, var_y, corr_xy, nb


def _nan_tolerant_pearson_update(
    preds: Tensor,
    target: Tensor,
    mean_x: Tensor,
    mean_y: Tensor,
    var_x: Tensor,
    var_y: Tensor,
    corr_xy: Tensor,
    n_total: Tensor,
    num_outputs: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Update Pearson correlation statistics, handling NaN values."""
    # Handle NaN values
    valid_mask = ~torch.isnan(preds) & ~torch.isnan(target)
    if not torch.any(valid_mask):
        return mean_x, mean_y, var_x, var_y, corr_xy, n_total

    # Ensure tensors are on the same device
    device = preds.device
    valid_preds = preds[valid_mask]
    valid_target = target[valid_mask]

    # Move all state tensors to the input device if needed
    mean_x = mean_x.to(device)
    mean_y = mean_y.to(device)
    var_x = var_x.to(device)
    var_y = var_y.to(device)
    corr_xy = corr_xy.to(device)
    n_total = n_total.to(device)

    # First compute sum on original device, then convert
    n = torch.tensor(valid_mask.sum().item(), dtype=n_total.dtype, device=device)
    n_total = n_total + n

    # Update means
    delta_x = valid_preds - mean_x
    mean_x = mean_x + delta_x.sum() / n_total

    delta_y = valid_target - mean_y
    mean_y = mean_y + delta_y.sum() / n_total

    # Update variances and correlation
    var_x = var_x + (delta_x * (valid_preds - mean_x)).sum()
    var_y = var_y + (delta_y * (valid_target - mean_y)).sum()
    corr_xy = corr_xy + (delta_x * delta_y).sum()

    return mean_x, mean_y, var_x, var_y, corr_xy, n_total


class NaNTolerantPearsonCorrCoef(Metric):
    is_differentiable = True
    higher_is_better = None  # both -1 and 1 are optimal
    full_state_update = True
    plot_lower_bound = -1.0
    plot_upper_bound = 1.0

    def __init__(self, num_outputs: int = 1, **kwargs: Any):
        super().__init__(**kwargs)

        self.num_outputs = num_outputs

        # States with no reduction function - we'll handle reduction in compute()
        self.add_state(
            "mean_x", default=torch.zeros(self.num_outputs), dist_reduce_fx=None
        )
        self.add_state(
            "mean_y", default=torch.zeros(self.num_outputs), dist_reduce_fx=None
        )
        self.add_state(
            "var_x", default=torch.zeros(self.num_outputs), dist_reduce_fx=None
        )
        self.add_state(
            "var_y", default=torch.zeros(self.num_outputs), dist_reduce_fx=None
        )
        self.add_state(
            "corr_xy", default=torch.zeros(self.num_outputs), dist_reduce_fx=None
        )
        self.add_state(
            "n_total", default=torch.zeros(self.num_outputs), dist_reduce_fx=None
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total = (
            _nan_tolerant_pearson_update(
                preds,
                target,
                self.mean_x,
                self.mean_y,
                self.var_x,
                self.var_y,
                self.corr_xy,
                self.n_total,
                self.num_outputs,
            )
        )

    def compute(self) -> Tensor:
        """Compute Pearson correlation coefficient over state."""
        # Handle multi-device aggregation if needed
        if (self.num_outputs == 1 and self.mean_x.numel() > 1) or (
            self.num_outputs > 1 and self.mean_x.ndim > 1
        ):
            _, _, var_x, var_y, corr_xy, n_total = _final_aggregation(
                self.mean_x,
                self.mean_y,
                self.var_x,
                self.var_y,
                self.corr_xy,
                self.n_total,
            )
        else:
            var_x = self.var_x
            var_y = self.var_y
            corr_xy = self.corr_xy
            n_total = self.n_total

        # Handle zero-division and no-sample cases
        if n_total == 0:
            return torch.full_like(var_x, float("nan"))

        denom = torch.sqrt(var_x * var_y)
        pearson = torch.where(
            denom > 0, corr_xy / denom, torch.tensor(float("nan"), device=denom.device)
        )

        return pearson


class NaNTolerantSpearmanCorrCoef(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    plot_lower_bound = -1.0
    plot_upper_bound = 1.0

    def __init__(self, num_outputs: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        from torchmetrics.utilities import rank_zero_warn

        rank_zero_warn(
            "Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer."
            " For large datasets, this may lead to large memory footprint."
        )

        if not isinstance(num_outputs, int) and num_outputs < 1:
            raise ValueError(
                f"Expected argument `num_outputs` to be an int larger than 0, but got {num_outputs}"
            )
        self.num_outputs = num_outputs

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        # Add counter to track number of valid samples
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        # Handle NaN values
        valid_mask = ~torch.isnan(preds) & ~torch.isnan(target)
        if torch.any(valid_mask):
            self.preds.append(preds[valid_mask].float())
            self.target.append(target[valid_mask].float())
            self.num_samples += valid_mask.sum()

    def compute(self) -> Tensor:
        """Compute Spearman correlation coefficient over state."""
        from torchmetrics.utilities.data import dim_zero_cat

        # Check if we have any valid samples
        if self.num_samples == 0:
            # Create nan tensor on same device as num_samples
            return torch.tensor(float("nan"), device=self.num_samples.device)

        try:
            # Concatenate all collected values
            preds = dim_zero_cat(self.preds)
            target = dim_zero_cat(self.target)
        except (RuntimeError, IndexError):
            # Handle case where concatenation fails
            return torch.tensor(float("nan"), device=self.num_samples.device)

        # Convert to ranks
        preds_rank = torch.argsort(torch.argsort(preds)).float()
        target_rank = torch.argsort(torch.argsort(target)).float()

        # Compute correlation on ranks
        pearson = NaNTolerantPearsonCorrCoef(num_outputs=self.num_outputs)
        pearson = pearson.to(preds.device)  # Move to same device as data
        return pearson(preds_rank, target_rank)


#######


class NaNTolerantMetricBase(Metric):
    def __init__(self, **kwargs):
        # Configure for DDP compatibility
        kwargs["compute_on_cpu"] = False  # Keep computation on GPU
        kwargs["sync_on_compute"] = False  # Let Lightning handle sync
        kwargs["dist_sync_on_step"] = True  # Sync after each step
        super().__init__(**kwargs)
        # Register device tracking buffer
        self.register_buffer("_device_buffer", torch.zeros(1))

    def _track_device(self, tensor: Tensor) -> None:
        """Track the device of input tensors"""
        if tensor.device != self._device_buffer.device:
            self._device_buffer = self._device_buffer.to(tensor.device)

    def _create_tensor_on_device(self, value, *shape):
        """Create a new tensor on the tracked device"""
        return torch.full(shape, value, device=self._device_buffer.device)


class NaNTolerantMSE(NaNTolerantMetricBase):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, squared: bool = True, num_outputs: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        self.squared = squared
        self.num_outputs = num_outputs

        self.add_state(
            "sum_squared_error",
            default=torch.zeros(num_outputs),
            dist_reduce_fx="sum",
            persistent=True,  # Save in state dict
        )
        self.add_state(
            "total",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
            persistent=True,  # Save in state dict
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        self._track_device(preds)  # Track current device
        sum_squared_error, num_obs = _nan_tolerant_mse_update(
            preds, target, self.num_outputs
        )
        self.sum_squared_error += sum_squared_error
        self.total += num_obs

    def compute(self) -> Tensor:
        # Handle empty case
        if self.total == 0:
            return self._create_tensor_on_device(float("nan"), self.num_outputs)

        mean_squared_error = _nan_tolerant_error_compute(
            self.sum_squared_error, self.total
        )
        return mean_squared_error if self.squared else torch.sqrt(mean_squared_error)


class NaNTolerantR2Score(NaNTolerantMetricBase):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_outputs: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs

        # For computing R², we need both squared error and total variance
        self.add_state(
            "sum_squared_error",
            default=torch.zeros(num_outputs),
            dist_reduce_fx="sum",
            persistent=True,
        )
        self.add_state(
            "sum_squared_deviation",
            default=torch.zeros(num_outputs),
            dist_reduce_fx="sum",
            persistent=True,
        )
        self.add_state(
            "total", default=torch.tensor(0), dist_reduce_fx="sum", persistent=True
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self._track_device(preds)

        # Handle NaN values
        valid_mask = ~torch.isnan(preds) & ~torch.isnan(target)
        if not torch.any(valid_mask):
            return

        # Get valid samples
        valid_preds = preds[valid_mask]
        valid_target = target[valid_mask]

        # Compute target mean for this batch
        target_mean = torch.mean(valid_target)

        # Update squared error and total variance
        squared_error = (valid_preds - valid_target) ** 2
        squared_deviation = (valid_target - target_mean) ** 2

        if self.num_outputs > 1:
            self.sum_squared_error += torch.sum(squared_error, dim=0)
            self.sum_squared_deviation += torch.sum(squared_deviation, dim=0)
        else:
            self.sum_squared_error += torch.sum(squared_error)
            self.sum_squared_deviation += torch.sum(squared_deviation)

        self.total += valid_mask.sum()

    def compute(self) -> Tensor:
        """Compute R² score."""
        # Handle empty case
        if self.total == 0:
            return self._create_tensor_on_device(float("nan"), self.num_outputs)

        # Compute R² score: 1 - (sum squared error / sum squared deviation)
        # Handle the case where deviation is 0 (constant target)
        r2 = torch.where(
            self.sum_squared_deviation != 0,
            1 - self.sum_squared_error / self.sum_squared_deviation,
            torch.tensor(float("nan"), device=self.sum_squared_error.device),
        )

        return r2
