import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .base import BaseModel


class SimpleCNN(nn.Module):
    """A simple CNN for image classification (e.g., MNIST-like)."""

    def __init__(self):
        # UP008: Use super() instead of super(__class__, self)
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        features = F.relu(self.fc1(x))
        x = self.fc2(features)
        return x, features


class CNNModel(BaseModel):
    """Wrapper for SimpleCNN."""

    def __init__(self, model_name: str = "simple_cnn"):
        super().__init__(model_name)
        from ..config import config

        self.device = torch.device(config.DEVICE)

    def load(self) -> None:
        """Initialize the CNN model."""
        self.model = SimpleCNN().to(self.device)
        self.model.eval()

    def predict(self, data: np.ndarray) -> list[int]:
        """
        Predict class labels for images.
        Data expected to be numpy array of shape (N, 28, 28) or (N, 1, 28, 28).
        """
        if self.model is None:
            self.load()

        tensor_data = self._preprocess(data)
        with torch.no_grad():
            outputs, _ = self.model(tensor_data)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().tolist()

    def get_embeddings(self, data: np.ndarray) -> np.ndarray:
        """Extract features from the penultimate layer."""
        if self.model is None:
            self.load()

        tensor_data = self._preprocess(data)
        with torch.no_grad():
            _, features = self.model(tensor_data)
        return features.cpu().numpy()

    def _preprocess(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy data to torch tensor."""
        if data.ndim == 3:
            data = data[:, np.newaxis, :, :]

        # Normalize to 0-1 if not already
        if data.max() > 1.0:
            data = data / 255.0

        tensor = torch.FloatTensor(data).to(self.device)
        return tensor
