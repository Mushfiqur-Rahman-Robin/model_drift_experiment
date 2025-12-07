import pytest
import numpy as np
from src.models.cnn import CNNModel
from src.models.bert import BERTModel

@pytest.mark.unit
class TestModels:
    def test_cnn_initialization(self):
        model = CNNModel()
        assert model.model_name == "simple_cnn"
        assert model.model is None

    def test_cnn_prediction(self):
        model = CNNModel()
        # Mock data: 2 samples, 28x28
        data = np.random.rand(2, 28, 28).astype(np.float32)
        preds = model.predict(data)
        assert len(preds) == 2
        assert all(isinstance(p, int) for p in preds)
        assert all(0 <= p <= 9 for p in preds)

    def test_cnn_embeddings(self):
        model = CNNModel()
        data = np.random.rand(2, 28, 28).astype(np.float32)
        embeddings = model.get_embeddings(data)
        assert embeddings.shape == (2, 128) # 128 is the output of fc1

    def test_bert_initialization(self):
        model = BERTModel()
        assert model.model_name == "bert-base-uncased"
        assert model.model is None

    def test_bert_embeddings(self):
        model = BERTModel()
        data = ["Hello world", "Testing BERT"]
        embeddings = model.get_embeddings(data)
        # BERT base hidden size is 768
        assert embeddings.shape == (2, 768)

        # Call again to test cached model branch
        embeddings2 = model.get_embeddings(data)
        assert np.array_equal(embeddings, embeddings2)

    def test_cnn_preprocess_normalization(self):
        model = CNNModel()
        # Data > 1.0
        data = np.random.randint(0, 255, (2, 28, 28)).astype(np.float32)
        tensor = model._preprocess(data)
        assert tensor.max() <= 1.0

        # Data already normalized
        data_norm = np.random.rand(2, 28, 28).astype(np.float32)
        tensor_norm = model._preprocess(data_norm)
        assert tensor_norm.max() <= 1.0

    def test_cnn_preprocess_dimensions(self):
        model = CNNModel()
        # 3D input (N, H, W) -> (N, 1, H, W)
        data = np.random.rand(2, 28, 28).astype(np.float32)
        tensor = model._preprocess(data)
        assert tensor.shape == (2, 1, 28, 28)

        # 4D input (N, C, H, W)
        data_4d = np.random.rand(2, 1, 28, 28).astype(np.float32)
        tensor_4d = model._preprocess(data_4d)
        assert tensor_4d.shape == (2, 1, 28, 28)

from src.models.base import BaseModel
class ConcreteModel(BaseModel):
    def load(self): super().load()
    def predict(self, data): super().predict(data)
    def get_embeddings(self, data): super().get_embeddings(data)

def test_base_model():
    model = ConcreteModel("test")
    assert model.model_name == "test"
    model.model = "something"
    model.clear()
    assert model.model is None
    # Cover abstract methods
    model.load()
    model.predict(None)
    model.get_embeddings(None)

def test_cnn_cached_prediction():
    model = CNNModel()
    data = np.random.rand(2, 28, 28).astype(np.float32)
    # First call loads model
    model.predict(data)
    # Second call uses cached model
    model.predict(data)
    assert model.model is not None
