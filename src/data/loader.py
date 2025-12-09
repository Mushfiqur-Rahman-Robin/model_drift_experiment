import logging

import numpy as np


class DataLoader:
    logger = logging.getLogger(__name__)
    """
    Simulates data loading for the experiment.
    We use synthetic data to avoid downloading large datasets.
    """

    @staticmethod
    def get_mnist_data(n_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic MNIST-like data (28x28 grayscale images).
        Returns: (images, labels)
        """
        DataLoader.logger.debug(f"Generating {n_samples} synthetic MNIST-like images.")
        # Random noise images for base distribution
        images = np.random.rand(n_samples, 28, 28).astype(np.float32)
        # Random labels 0-9
        labels = np.random.randint(0, 10, size=n_samples)
        return images, labels

    @staticmethod
    def get_text_data(n_samples: int = 20) -> list[str]:
        """
        Generate synthetic text data.
        """
        DataLoader.logger.debug(f"Generating {n_samples} synthetic text sentences.")
        base_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "I love machine learning and artificial intelligence.",
            "Model drift is a serious issue in production systems.",
            "Python is a great programming language for data science.",
            "FastAPI makes building APIs easy and fast.",
        ]

        data = []
        for _ in range(n_samples):
            data.append(np.random.choice(base_sentences))
        return data
