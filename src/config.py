from pathlib import Path


class Config:
    """Configuration settings for the Model Drift Experiment."""

    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"

    # Model settings
    CNN_MODEL_NAME = "simple_cnn"
    BERT_MODEL_NAME = "bert-base-uncased"

    # Drift detection settings
    DRIFT_THRESHOLD = 0.05  # P-value threshold for statistical tests

    # Device settings
    # Force CPU to avoid CUDA capability mismatches on older GPUs (like GeForce 940MX)
    DEVICE = "cpu"

    # API settings
    API_TITLE = "Model Drift Experiment API"
    API_VERSION = "0.1.0"
    API_DESCRIPTION = (
        "API for demonstrating model drift detection in CNN and BERT models."
    )

    @classmethod
    def ensure_dirs(cls):
        """Ensure necessary directories exist."""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
