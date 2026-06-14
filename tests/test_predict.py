"""
test_predict.py — Unit tests for the /predict endpoint.

Mocks the ML model and asserts that /predict returns the expected
JSON structure with prediction, confidence, and model_version keys.
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Mock the model before importing app so no real model file is needed
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    """Create a test client with a mocked model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    # Patch the module-level model state in app
    with patch("server.app._model", mock_model), \
         patch("server.app._model_version", 1), \
         patch("server.app._run_ingestion", return_value=None), \
         patch("server.app._run_retrain_watcher", return_value=None):

        from fastapi.testclient import TestClient
        from server.app import app
        yield TestClient(app)


def test_predict_returns_correct_structure(client):
    """Response must contain prediction, confidence, and model_version keys."""
    response = client.post(
        "/predict",
        json={"features": [1.5, 2.0]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction"    in data, "Response must contain 'prediction' key"
    assert "confidence"    in data, "Response must contain 'confidence' key"
    assert "model_version" in data, "Response must contain 'model_version' key"


def test_predict_returns_valid_types(client):
    """Prediction must be int, confidence must be float between 0 and 1."""
    response = client.post(
        "/predict",
        json={"features": [1.5, 2.0]}
    )
    data = response.json()
    assert isinstance(data["prediction"], int),   "prediction must be an integer"
    assert isinstance(data["confidence"], float), "confidence must be a float"
    assert 0.0 <= data["confidence"] <= 1.0,      "confidence must be between 0 and 1"


def test_predict_correct_values(client):
    """Prediction should return valid prediction and model_version."""
    response = client.post(
        "/predict",
        json={"features": [1.5, 2.0]}
    )
    data = response.json()
    assert data["prediction"] in [0, 1], "Expected binary prediction (0 or 1)"
    assert data["model_version"] > 0,    "Expected positive model version"


def test_health_endpoint(client):
    """/health must return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_missing_features(client):
    """Missing features field should return 422 validation error."""
    response = client.post("/predict", json={})
    assert response.status_code == 422