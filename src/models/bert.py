import logging
from typing import Any

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from .base import BaseModel

logger = logging.getLogger(__name__)


class BERTModel(BaseModel):
    """Wrapper for BERT model."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__(model_name)
        self.tokenizer = None
        from ..config import config

        self.device = torch.device(config.DEVICE)

    def load(self) -> None:
        """Load BERT model and tokenizer."""
        if self.model is not None:
            logger.debug(f"BERT model '{self.model_name}' already loaded.")
            return

        from ..config import config

        cache_dir = config.MODEL_DIR / "bert"
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Loading BERT model '{self.model_name}' from cache_dir: {cache_dir} to device: {self.device}"
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir
        )
        self.model = BertModel.from_pretrained(self.model_name, cache_dir=cache_dir).to(
            self.device
        )
        self.model.eval()
        logger.info(f"BERT model '{self.model_name}' loaded successfully.")

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
            logger.debug(f"BERT model '{self.model_name}' loaded for get_embeddings.")

        inputs = self.tokenizer(
            data, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()

    def clear(self) -> None:
        """Clear the model from memory."""
        if self.model is not None:
            logger.info(f"Clearing BERT model '{self.model_name}' from memory.")
            self.model = None
            self.tokenizer = None
        else:
            logger.debug(f"BERT model '{self.model_name}' is already cleared.")
