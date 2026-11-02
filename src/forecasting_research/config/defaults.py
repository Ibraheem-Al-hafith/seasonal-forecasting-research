"""Default constants. Every value here is recorded in dataset metadata when it is used."""

GENERATOR_VERSION = "1.0"

DEFAULT_NOISE_SIGMA = {"low": 0.02, "medium": 0.05, "high": 0.10}
DEFAULT_METHODS = ["classical_decomposition", "holt_winters", "sarima"]
METRIC_NAMES = ["MAE", "MSE", "RMSE", "MAPE"]
MIN_TRAIN_CYCLES = 2
