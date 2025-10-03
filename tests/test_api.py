from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.routes import get_summarizer_service
from model.summarization import SummarizerService

# --- Test Setup ---
client = TestClient(app)

MOCK_SUMMARY_TEXT = "The service successfully processed the request."
EXPECTED_SUMMARY_RESPONSE = {"summary": MOCK_SUMMARY_TEXT}

VALID_TEXT_FOR_TEST = "a" * 250
VALID_REQUEST_BODY = {
    "text": VALID_TEXT_FOR_TEST,
    "min_length": 50,
    "max_length": 150,
    "num_beams": 4,
    "repetition_penalty": 2.5,
}

# Define the path for the Dependency Injection function
SERVICE_DI_PATH = "app.routes.get_summarizer_service"

# --- Fixtures and Overrides ---


@pytest.fixture
def mock_ready_service() -> MagicMock:
    """Returns a mock service configured to report model as LOADED (ready)."""
    mock_service = MagicMock(spec=SummarizerService)
    mock_service.get_status.return_value = {
        "model_name": "facebook/bart-large-cnn",
        "status": "Loaded",
        "is_ready": True,
    }
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"summary_text": MOCK_SUMMARY_TEXT}]
    mock_service.get_pipeline.return_value = mock_pipeline
    return mock_service


@pytest.fixture
def mock_unready_service() -> MagicMock:
    """Returns a mock service configured to report model as LOADING (unready)."""
    unready_mock_service = MagicMock(spec=SummarizerService)
    unready_mock_service.get_status.return_value = {
        "model_name": "facebook/bart-large-cnn",
        "status": "Loading",
        "is_ready": False,
    }
    unready_mock_service.get_pipeline.return_value = None
    return unready_mock_service


@contextmanager
def override_dependency(mock_service: MagicMock) -> Generator[MagicMock, None, None]:
    """Temporarily overrides the service dependency for reliable testing."""
    # Set the dependency to return the mock service
    app.dependency_overrides[get_summarizer_service] = lambda: mock_service
    try:
        yield mock_service
    finally:
        # Crucial: Always clear the override after the test
        app.dependency_overrides.pop(get_summarizer_service, None)


class TestAPIEndpoints:

    def test_01_health_check_liveness(self):
        """Test the basic liveness check (/health)."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

    def test_02_status_check_readiness(self, mock_ready_service):
        """Test the detailed readiness check (/api/v1/status) using the READY mock."""
        # Use the override pattern to inject the mock service
        with override_dependency(mock_ready_service):
            response = client.get("/api/v1/status")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["is_ready"] is True

    def test_03_summarization_success(self, mock_ready_service):
        """Test a successful summarization request using the mocked pipeline."""
        with override_dependency(mock_ready_service):
            response = client.post("/api/v1/summarize", json=VALID_REQUEST_BODY)

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == EXPECTED_SUMMARY_RESPONSE

    def test_04_summarization_validation_failure(self):
        """Test the Pydantic validation for minimum text length (422 error)."""
        invalid_body = VALID_REQUEST_BODY.copy()
        invalid_body["text"] = "Too short text."

        response = client.post("/api/v1/summarize", json=invalid_body)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_05_summarization_model_not_ready(self, mock_unready_service):
        """Test the 503 error when the model is not yet loaded."""

        with override_dependency(mock_unready_service):
            response = client.post("/api/v1/summarize", json=VALID_REQUEST_BODY)

            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            assert "Model is not yet ready" in response.json()["detail"]
