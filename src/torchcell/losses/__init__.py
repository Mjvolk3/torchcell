from .dcell import DCellLoss
from .weighted_mse import WeightedMSELoss

standard_losses = {"weighted_mse": WeightedMSELoss}

model_losses = {"dcell": DCellLoss}

__all__ = ["standard_losses", "model_losses"]
