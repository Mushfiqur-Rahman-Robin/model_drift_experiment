import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import unittest.mock

client = TestClient(app)

@pytest.mark.integration
class TestAPI:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_predict_cnn(self):
        response = client.post("/api/v1/predict/cnn", json={"simulate": True})
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "cnn"
        assert isinstance(data["prediction"], list)

    def test_predict_bert(self):
        response = client.post("/api/v1/predict/bert", json={"simulate": True})
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "bert"
        assert "shape" in data["prediction"]

    def test_drift_cnn(self):
        # First call might take longer due to model loading
        response = client.post("/api/v1/detect-drift/cnn", json={"severity": 0.5})
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "cnn"
        assert "analysis" in data
        assert "drift_detected" in data["analysis"]

    def test_cleanup(self):
        # Mock torch.cuda.is_available to return True to cover that branch
        # Also mock config.DEVICE to be "cuda"
        with unittest.mock.patch("torch.cuda.is_available", return_value=True):
            with unittest.mock.patch("torch.cuda.empty_cache") as mock_empty:
                with unittest.mock.patch("src.config.config.DEVICE", "cuda"):
                    response = client.post("/api/v1/cleanup")
                    assert response.status_code == 200
                    assert response.json()["status"] == "cleaned"
                    mock_empty.assert_called_once()

    def test_predict_404(self):
        response = client.post("/api/v1/predict/invalid_model", json={"simulate": True})
        assert response.status_code == 404

    def test_predict_501(self):
        response = client.post("/api/v1/predict/cnn", json={"simulate": False})
        assert response.status_code == 501

    def test_drift_404(self):
        response = client.post("/api/v1/detect-drift/invalid_model", json={"severity": 0.5})
        assert response.status_code == 404

    def test_drift_cnn_cached(self):
        # Call once to load
        client.post("/api/v1/detect-drift/cnn", json={"severity": 0.5})
        # Call again to hit cached branch
        response = client.post("/api/v1/detect-drift/cnn", json={"severity": 0.5})
        assert response.status_code == 200

    def test_drift_bert(self):
        # Ensure clean state to hit initialization branch
        client.post("/api/v1/cleanup")

        # First call might take longer due to model loading
        response = client.post("/api/v1/detect-drift/bert", json={"severity": 0.5})
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "bert"
        assert "analysis" in data
