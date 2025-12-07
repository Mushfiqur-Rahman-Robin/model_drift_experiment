from typing import Any

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from .base import BaseModel


class BERTModel(BaseModel):
    """Wrapper for BERT model."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__(model_name)
        self.tokenizer = None
        from ..config import config

        self.device = torch.device(config.DEVICE)

    def load(self) -> None:
        """Load BERT model and tokenizer."""
        from ..config import config

        cache_dir = config.MODEL_DIR / "bert"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir
        )
        self.model = BertModel.from_pretrained(self.model_name, cache_dir=cache_dir).to(
            self.device
        )
        self.model.eval()

    def predict(self, data: list[str]) -> Any:
        """
        Predict (dummy implementation for base BERT as it's not fine-tuned).
        Returns embeddings as 'prediction' proxy for this experiment,
        or we could add a classification head.
        For drift detection, we care about embeddings.
        """
        # Since this is a base model, we don't have classes.
        # We will return the pooled output as a "representation".
        return self.get_embeddings(data)

    def get_embeddings(self, data: list[str]) -> np.ndarray:
        """Get BERT embeddings (CLS token)."""
        if self.model is None:
            self.load()

        inputs = self.tokenizer(
            data, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()
