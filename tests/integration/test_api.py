# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import unittest.mock
from src.config import request_id_ctx
import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_client():
    """
    Fixture to provide a test client for the FastAPI application.
    """
    dummy_request_id = "test-fixture-request-id"
    token = request_id_ctx.set(dummy_request_id)

    logger.info(f"Setting test request ID: {dummy_request_id}")

    with TestClient(app) as client:
        yield client

    request_id_ctx.reset(token)
    logger.info("Resetting test request ID context.")


@pytest.mark.integration
class TestAPI:
    def test_root(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "X-Request-ID" in response.headers

    def test_predict_cnn(self, test_client):
        response = test_client.post("/api/v1/predict/cnn", json={"simulate": True})
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "cnn"
        assert isinstance(data["prediction"], list)
        assert "X-Request-ID" in response.headers

    def test_predict_bert(self, test_client):
        response = test_client.post("/api/v1/predict/bert", json={"simulate": True})
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "bert"
        assert "shape" in data["prediction"]
        assert "X-Request-ID" in response.headers

    def test_drift_cnn(self, test_client):
        response = test_client.post("/api/v1/detect-drift/cnn", json={"severity": 0.5})
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "cnn"
        assert "analysis" in data
        assert "drift_detected" in data["analysis"]
        assert "X-Request-ID" in response.headers

    def test_cleanup(self, test_client):
        with unittest.mock.patch("torch.cuda.is_available", return_value=True):
            with unittest.mock.patch("torch.cuda.empty_cache") as mock_empty:
                with unittest.mock.patch("src.config.config.DEVICE", "cuda"):
                    response = test_client.post("/api/v1/cleanup")
                    assert response.status_code == 200
                    assert response.json()["status"] == "cleaned"
                    mock_empty.assert_called_once()
                    assert "X-Request-ID" in response.headers

    def test_predict_404(self, test_client):
        response = test_client.post("/api/v1/predict/invalid_model", json={"simulate": True})
        assert response.status_code == 404
        assert "X-Request-ID" in response.headers

    def test_predict_501(self, test_client):
        response = test_client.post("/api/v1/predict/cnn", json={"simulate": False})
        assert response.status_code == 501
        assert "X-Request-ID" in response.headers
        assert response.json()["detail"] == "Custom data input not fully implemented for demo"

    def test_drift_404(self, test_client):
        response = test_client.post("/api/v1/detect-drift/invalid_model", json={"severity": 0.5})
        assert response.status_code == 404
        assert "X-Request-ID" in response.headers

    def test_drift_cnn_cached(self, test_client):
        # Call once to load
        test_client.post("/api/v1/detect-drift/cnn", json={"severity": 0.5})
        # Call again to hit cached branch
        response = test_client.post("/api/v1/detect-drift/cnn", json={"severity": 0.5})
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers

    def test_drift_bert(self, test_client):
        # Ensure clean state to hit initialization branch
        test_client.post("/api/v1/cleanup")

        # First call might take longer due to model loading
        response = test_client.post("/api/v1/detect-drift/bert", json={"severity": 0.5})
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "bert"
        assert "analysis" in data
        assert "X-Request-ID" in response.headers
