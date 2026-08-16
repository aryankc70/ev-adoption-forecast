"""Tests for the FastAPI model serving endpoints.

These tests require the real trained model artifacts to exist at
models/production/ (run `python -m ev_forecast.pipelines.train_final_model`
first if they're missing) -- this is an integration-style test of the API
layer, not a fully isolated unit test, since mocking out the model loading
would defeat the purpose of testing that predictions actually work end to end.
"""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from ev_forecast.api.main import MODEL_DIR, app

pytestmark = pytest.mark.skipif(
    not MODEL_DIR.exists(),
    reason="Model artifacts not found -- run `python -m ev_forecast.pipelines.train_final_model` first",
)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


class TestPredictEndpoint:
    def test_known_country_returns_prediction(self, client: TestClient) -> None:
        response = client.post(
            "/predict", json={"region_country": "Germany", "mode": "Cars", "year": 2023}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["region_country"] == "Germany"
        assert body["mode"] == "Cars"
        assert body["year"] == 2023
        assert isinstance(body["predicted_share"], float)
        assert 0.0 <= body["predicted_share"] <= 100.0

    def test_unknown_country_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/predict", json={"region_country": "Atlantis", "mode": "Cars", "year": 2023}
        )
        assert response.status_code == 404
        assert "Atlantis" in response.json()["detail"]

    def test_year_below_minimum_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/predict", json={"region_country": "Germany", "mode": "Cars", "year": 2020}
        )
        assert response.status_code == 422  # Pydantic validation: year must be >= 2022

    def test_multi_year_forecast_is_internally_consistent(self, client: TestClient) -> None:
        """
        A later year's prediction should come from the same recursive walk as an
        earlier year's -- not identical values, but both should be valid, finite
        numbers in range, confirming the multi-step walk doesn't blow up.
        """
        response_2023 = client.post(
            "/predict", json={"region_country": "Norway", "mode": "Cars", "year": 2023}
        )
        response_2025 = client.post(
            "/predict", json={"region_country": "Norway", "mode": "Cars", "year": 2025}
        )
        assert response_2023.status_code == 200
        assert response_2025.status_code == 200

        share_2023 = response_2023.json()["predicted_share"]
        share_2025 = response_2025.json()["predicted_share"]
        assert 0.0 <= share_2023 <= 100.0
        assert 0.0 <= share_2025 <= 100.0

    def test_missing_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict", json={"region_country": "Germany", "mode": "Cars"})
        assert response.status_code == 422
