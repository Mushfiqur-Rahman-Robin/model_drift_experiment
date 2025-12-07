from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Make predictions on the given data."""
        pass

    @abstractmethod
    def get_embeddings(self, data: Any) -> np.ndarray:
        """Extract embeddings/features from the data for drift detection."""
        pass

    def clear(self) -> None:
        """Clear the model from memory."""
        self.model = None
