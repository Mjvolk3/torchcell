# torchcell/losses/point_dist_graph_reg
# [[torchcell.losses.point_dist_graph_reg]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/point_dist_graph_reg
# Test file: tests/torchcell/losses/test_point_dist_graph_reg.py


from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.distributed as dist

from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.multi_dim_nan_tolerant import WeightedMSELoss, WeightedDistLoss


# Import Wasserstein loss components from mle_wasserstein
try:
    from torchcell.losses.mle_wasserstein import (
        WeightedWassersteinLoss,
        BufferedWeightedWassersteinLoss,
    )
    WASSERSTEIN_AVAILABLE = True
except ImportError:
    WASSERSTEIN_AVAILABLE = False


class PointDistGraphReg(nn.Module):
    """
    Modular composite loss for Cell Graph Transformer combining:
    1. Point estimator (MSE or LogCosh) - required
    2. Distribution loss (Dist or Wasserstein) - optional
    3. Graph regularization (from model) - optional

    Designed for single-phenotype predictions with graph-regularized attention.
    """

    def __init__(
        self,
        # Point estimator configuration
        point_estimator: Optional[Dict] = None,

        # Distribution loss configuration
        distribution_loss: Optional[Dict] = None,

        # Graph regularization configuration
        graph_regularization: Optional[Dict] = None,

        # Buffer configuration
        buffer: Optional[Dict] = None,

        # DDP configuration
        ddp: Optional[Dict] = None,

        # Other
        weights: Optional[torch.Tensor] = None,

        # Backward compatibility - deprecated, use nested dicts
        point_loss_type: Optional[str] = None,
        lambda_point: Optional[float] = None,
        dist_loss_type: Optional[str] = None,
        lambda_dist: Optional[float] = None,
        lambda_graph_reg: Optional[float] = None,
        dist_bandwidth: Optional[float] = None,
        wasserstein_blur: Optional[float] = None,
        wasserstein_p: Optional[float] = None,
        wasserstein_scaling: Optional[float] = None,
        use_buffer: Optional[bool] = None,
        buffer_size: Optional[int] = None,
        min_samples_for_dist: Optional[int] = None,
        use_ddp_gather: Optional[bool] = None,
        gather_interval: Optional[int] = None,
    ):
        super().__init__()

        # Parse configuration from nested dicts (new structure) or flat params (backward compat)
        # Point estimator config
        if point_estimator is not None:
            _point_type = point_estimator.get("type", "logcosh")
            _lambda_point = point_estimator.get("lambda", 1.0)
        else:
            _point_type = point_loss_type or "logcosh"
            _lambda_point = lambda_point if lambda_point is not None else 1.0

        # Distribution loss config
        if distribution_loss is not None:
            _dist_type = distribution_loss.get("type", "dist")
            _lambda_dist = distribution_loss.get("lambda", 0.1)
            _dist_bandwidth = distribution_loss.get("dist_bandwidth", 0.5)
            _wasserstein_blur = distribution_loss.get("wasserstein_blur", 0.05)
            _wasserstein_p = distribution_loss.get("wasserstein_p", 2)
            _wasserstein_scaling = distribution_loss.get("wasserstein_scaling", 0.9)
            _min_samples_dist = distribution_loss.get("min_samples_for_dist", 64)
            _min_samples_wasserstein = distribution_loss.get("min_samples_for_wasserstein", 224)
        else:
            _dist_type = dist_loss_type or "dist"
            _lambda_dist = lambda_dist if lambda_dist is not None else 0.1
            _dist_bandwidth = dist_bandwidth if dist_bandwidth is not None else 0.5
            _wasserstein_blur = wasserstein_blur if wasserstein_blur is not None else 0.05
            _wasserstein_p = wasserstein_p if wasserstein_p is not None else 2
            _wasserstein_scaling = wasserstein_scaling if wasserstein_scaling is not None else 0.9
            _min_samples_dist = min_samples_for_dist if min_samples_for_dist is not None else 64
            _min_samples_wasserstein = 224

        # Graph regularization config
        if graph_regularization is not None:
            _lambda_graph_reg = graph_regularization.get("lambda", 1.0)
        else:
            _lambda_graph_reg = lambda_graph_reg if lambda_graph_reg is not None else 1.0

        # Buffer config
        if buffer is not None:
            _use_buffer = buffer.get("use_buffer", True)
            _buffer_size = buffer.get("buffer_size", 256)
        else:
            _use_buffer = use_buffer if use_buffer is not None else True
            _buffer_size = buffer_size if buffer_size is not None else 256

        # DDP config
        if ddp is not None:
            _use_ddp_gather = ddp.get("use_ddp_gather", True)
            _gather_interval = ddp.get("gather_interval", 1)
        else:
            _use_ddp_gather = use_ddp_gather if use_ddp_gather is not None else True
            _gather_interval = gather_interval if gather_interval is not None else 1

        # Store configuration
        self.lambda_point = _lambda_point
        self.lambda_dist = _lambda_dist
        self.lambda_graph_reg = _lambda_graph_reg
        self.point_loss_type = _point_type
        self.dist_loss_type = _dist_type
        self.use_buffer = _use_buffer
        self.use_ddp_gather = _use_ddp_gather
        self.gather_interval = _gather_interval

        # Initialize point estimator loss
        if _point_type == "logcosh":
            self.point_loss = LogCoshLoss(reduction="mean")
        elif _point_type == "mse":
            self.point_loss = WeightedMSELoss(weights=weights)
        else:
            raise ValueError(f"Unknown point_loss_type: {_point_type}")

        # Initialize distribution loss (optional)
        self.dist_loss = None
        if _dist_type is not None and _lambda_dist > 0:
            if _dist_type == "dist":
                if _use_buffer:
                    # Import buffered version
                    from torchcell.losses.mle_dist_supcr import BufferedWeightedDistLoss
                    self.dist_loss = BufferedWeightedDistLoss(
                        buffer_size=_buffer_size,
                        bandwidth=_dist_bandwidth,
                        weights=weights,
                        min_samples=_min_samples_dist,
                    )
                else:
                    self.dist_loss = WeightedDistLoss(
                        bandwidth=_dist_bandwidth,
                        weights=weights,
                    )
            elif _dist_type == "wasserstein":
                if not WASSERSTEIN_AVAILABLE:
                    raise ImportError(
                        "Wasserstein loss requires geomloss. "
                        "Install with: pip install geomloss"
                    )
                # Use wasserstein-specific min_samples
                _min_samples = _min_samples_wasserstein
                if _use_buffer:
                    self.dist_loss = BufferedWeightedWassersteinLoss(
                        buffer_size=_buffer_size,
                        blur=_wasserstein_blur,
                        p=_wasserstein_p,
                        scaling=_wasserstein_scaling,
                        weights=weights,
                        min_samples=_min_samples,
                    )
                else:
                    self.dist_loss = WeightedWassersteinLoss(
                        blur=_wasserstein_blur,
                        p=_wasserstein_p,
                        scaling=_wasserstein_scaling,
                        weights=weights,
                    )
            else:
                raise ValueError(
                    f"Unknown dist_loss_type: {_dist_type}. "
                    "Must be 'dist', 'wasserstein', or None"
                )

        # Tracking
        self.register_buffer("forward_count", torch.zeros(1, dtype=torch.long))

        # Pre-allocate zero tensors to avoid dynamic tensor creation in forward()
        # This prevents torch.compile graph breaks when loss components are disabled
        self.register_buffer("_zero", torch.tensor(0.0))

    def gather_across_gpus(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather tensors from all GPUs for DDP synchronization.

        Ensures all ranks see same data for buffer updates, preventing
        distribution mismatch across GPUs.
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return tensor

        world_size = dist.get_world_size()
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        representations: Dict[str, Any],
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass computing composite loss.

        Args:
            predictions: Model predictions [batch_size, 1]
            targets: Target values [batch_size, 1]
            representations: Dict from model containing "graph_reg_loss"
            epoch: Current training epoch (optional, for adaptive weighting)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Use pre-allocated zero buffer to avoid dynamic tensor creation
        # This prevents torch.compile graph breaks
        total_loss = self._zero.clone()
        loss_dict = {}

        # 1. Point estimator loss (required)
        if self.point_loss_type == "logcosh":
            point_val = self.point_loss(predictions.squeeze(), targets.squeeze())
        else:  # mse
            point_val, _ = self.point_loss(predictions, targets)

        weighted_point = self.lambda_point * point_val
        total_loss = total_loss + weighted_point

        loss_dict.update({
            "point_loss": point_val.item(),
            "weighted_point": weighted_point.item(),
        })

        # 2. Distribution loss (optional)
        if self.dist_loss is not None and self.lambda_dist > 0:
            # DDP Synchronization: Gather samples from all GPUs
            # CRITICAL: When using buffers, all ranks must see same data to prevent
            # distribution mismatch (following pattern from mle_wasserstein.py:245-253)
            all_predictions = None
            all_targets = None
            if self.use_ddp_gather and self.forward_count % self.gather_interval == 0:
                all_predictions = self.gather_across_gpus(predictions)
                all_targets = self.gather_across_gpus(targets)

            # Compute distribution loss
            if self.use_buffer:
                # Buffered version: Pass gathered data for consistent buffer updates
                # The buffered loss will update its buffer with all_* if provided,
                # ensuring all GPUs maintain identical buffer contents
                dist_val, dist_dims = self.dist_loss(
                    predictions,  # Local batch for gradient computation
                    targets,
                    all_predictions,  # Gathered for buffer update
                    all_targets,
                    buffer_weight=1.0,  # Can be made adaptive if needed
                )
            else:
                # Non-buffered version: Uses only local batch
                dist_val, dist_dims = self.dist_loss(predictions, targets)

            weighted_dist = self.lambda_dist * dist_val
            total_loss = total_loss + weighted_dist

            loss_dict.update({
                "dist_loss": dist_val.item(),
                "weighted_dist": weighted_dist.item(),
            })
        else:
            # Use pre-allocated zero to prevent torch.compile graph breaks
            dist_val = self._zero
            loss_dict.update({
                "dist_loss": 0.0,
                "weighted_dist": 0.0,
            })

        # 3. Graph regularization loss (from model)
        if "graph_reg_loss" in representations and self.lambda_graph_reg > 0:
            graph_reg_val = representations["graph_reg_loss"]
            weighted_graph_reg = self.lambda_graph_reg * graph_reg_val

            total_loss = total_loss + weighted_graph_reg

            loss_dict.update({
                "graph_reg_loss": graph_reg_val.item(),
                "weighted_graph_reg": weighted_graph_reg.item(),
            })
        else:
            # Use pre-allocated zero to prevent torch.compile graph breaks
            graph_reg_val = self._zero
            loss_dict.update({
                "graph_reg_loss": 0.0,
                "weighted_graph_reg": 0.0,
            })

        # Compute normalized contributions
        if total_loss > 0:
            loss_dict["norm_weighted_point"] = (
                loss_dict["weighted_point"] / total_loss.item()
            )
            loss_dict["norm_weighted_dist"] = (
                loss_dict["weighted_dist"] / total_loss.item()
            )
            loss_dict["norm_weighted_graph_reg"] = (
                loss_dict["weighted_graph_reg"] / total_loss.item()
            )

        # Add totals
        loss_dict["total_loss"] = total_loss.item()
        loss_dict["total_weighted"] = total_loss.item()

        # Compute unweighted normalized (for analysis)
        total_unweighted = point_val + dist_val + graph_reg_val
        if total_unweighted > 0:
            loss_dict["norm_unweighted_point"] = (
                point_val.item() / total_unweighted.item()
            )
            loss_dict["norm_unweighted_dist"] = (
                dist_val.item() / total_unweighted.item()
            )
            loss_dict["norm_unweighted_graph_reg"] = (
                graph_reg_val.item() / total_unweighted.item()
            )

        # Update forward counter
        self.forward_count += 1

        return total_loss, loss_dict
