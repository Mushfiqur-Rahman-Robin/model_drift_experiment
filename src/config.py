# src/config.py
import contextvars
import logging
from pathlib import Path

# Context variable to hold the request ID for the current request
request_id_ctx = contextvars.ContextVar("request_id", default="no-request-id")


class Config:
    """Configuration settings for the Model Drift Experiment."""

    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"

    # Model settings
    CNN_MODEL_NAME = "simple_cnn"
    BERT_MODEL_NAME = "bert-base-uncased"

    # Drift detection settings
    DRIFT_THRESHOLD = 0.05

    # Device settings
    DEVICE = "cpu"

    # API settings
    API_TITLE = "Model Drift Experiment API"
    API_VERSION = "0.1.0"
    API_DESCRIPTION = (
        "API for demonstrating model drift detection in CNN and BERT models."
    )

    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FILE = LOGS_DIR / "app.log"
    ERROR_LOG_FILE = LOGS_DIR / "error.log"
    LOG_FORMAT = (
        "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
    )

    @classmethod
    def ensure_dirs(cls):
        """Ensure necessary directories exist."""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured MODEL_DIR exists: {cls.MODEL_DIR}")
        logging.info(f"Ensured DATA_DIR exists: {cls.DATA_DIR}")
        logging.info(f"Ensured LOGS_DIR exists: {cls.LOGS_DIR}")


class CustomFormatter(logging.Formatter):
    """Custom formatter that includes request_id in all log records."""

    def format(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = request_id_ctx.get()
        return super().format(record)


def setup_logging():
    """Configure logging for the application with multiple handlers."""
    # Ensure logs directory exists FIRST
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    formatter = CustomFormatter(Config.LOG_FORMAT)

    root_logger = logging.getLogger()
    root_logger.setLevel(Config.LOG_LEVEL)

    # Clear existing handlers
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # 1. Stream Handler (console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)

    # 2. File Handler for general logs
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # 3. File Handler for error logs
    error_file_handler = logging.FileHandler(Config.ERROR_LOG_FILE)
    error_file_handler.setFormatter(formatter)
    error_file_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_file_handler)

    logging.info("Logging configured with custom formatter and multiple handlers.")


# Only set up a basic logger at import time (no file handlers yet)
# File handlers will be set up during startup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

config = Config()
