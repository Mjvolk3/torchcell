from .dcell_regression import DCellRegressionTask
from .dcell_regression_slim import DCellRegressionSlimTask
from .graph_convolution_regression import GraphConvRegressionTask
from .regression import RegressionTask
from .simple_linear_regression import SimpleLinearRegressionTask

__all__ = [
    "RegressionTask",
    "SimpleLinearRegressionTask",
    "GraphConvRegressionTask",
    "DCellRegressionTask",
]
