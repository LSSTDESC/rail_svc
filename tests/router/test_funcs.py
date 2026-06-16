"""Tests for router/funcs.py — the funcs API endpoints.

Uses FastAPI TestClient with mocked local_async.funcs to test HTTP behavior
without requiring real DB or file operations.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rail_svc import models
from rail_svc.router.funcs import funcs_router


@pytest.fixture
def app():
    """Create a FastAPI app with just the funcs router."""
    app = FastAPI()
    app.include_router(funcs_router, prefix="/api/v1")
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestEstimatePdf:
    """Test /funcs/estimate-pdf endpoint."""

    def test_success(self, client):
        """Test successful PDF estimation."""
        mock_result = {"z": [0.1, 0.2, 0.3], "pdf": [0.5, 0.3, 0.2]}

        with patch("rail_svc.router.funcs.local_async.funcs.estimate_pdf", new_callable=AsyncMock) as mock:
            mock.return_value = mock_result

            response = client.post(
                "/api/v1/funcs/estimate-pdf",
                json={"estimator_id": 1, "dataset_id": 2, "row": 5},
            )

            assert response.status_code == 200
            assert response.json() == mock_result

    def test_internal_error(self, client):
        """Test that exceptions return 500."""
        with patch("rail_svc.router.funcs.local_async.funcs.estimate_pdf", new_callable=AsyncMock) as mock:
            mock.side_effect = RuntimeError("Model file not found")

            response = client.post(
                "/api/v1/funcs/estimate-pdf",
                json={"estimator_id": 1, "dataset_id": 2, "row": 5},
            )

            assert response.status_code == 500

    def test_validation_error_missing_field(self, client):
        """Test that missing required fields return 422."""
        response = client.post(
            "/api/v1/funcs/estimate-pdf",
            json={"estimator_id": 1},
        )
        assert response.status_code == 422


class TestEstimateEnsemble:
    """Test /funcs/estimate-ensemble endpoint."""

    def test_success(self, client):
        """Test successful ensemble estimation."""
        with patch(
            "rail_svc.router.funcs.local_async.funcs.estimate_ensemble", new_callable=AsyncMock
        ) as mock:
            mock.return_value = Path("/output/estimates.hdf5")

            response = client.post(
                "/api/v1/funcs/estimate-ensemble",
                json={"estimator_id": 1, "dataset_id": 2, "output_file_path": "/output/est.hdf5"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "output_file" in data
            assert "message" in data


class TestGetEstimatorsForDataset:
    """Test /funcs/get-estimators-for-dataset endpoint."""

    def test_success(self, client):
        """Test successful estimator retrieval."""
        mock_estimators = [
            models.Estimator(id_=1, name="bpz", config={}, model_id=1),
            models.Estimator(id_=2, name="flexzboost", config={"n_trees": 100}, model_id=2),
        ]

        with patch(
            "rail_svc.router.funcs.local_async.funcs.get_estimators_for_dataest", new_callable=AsyncMock
        ) as mock:
            mock.return_value = mock_estimators

            response = client.get("/api/v1/funcs/get-estimators-for-dataset/5")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["name"] == "bpz"

    def test_error(self, client):
        """Test error handling."""
        with patch(
            "rail_svc.router.funcs.local_async.funcs.get_estimators_for_dataest", new_callable=AsyncMock
        ) as mock:
            mock.side_effect = ValueError("Dataset not found")

            response = client.get("/api/v1/funcs/get-estimators-for-dataset/999")
            assert response.status_code == 500


class TestGetDatasetAndEstimates:
    """Test /funcs/get-dataset-and-estimates endpoint."""

    def test_success(self, client):
        """Test successful dataset+estimates retrieval."""
        mock_dataset = models.Dataset(
            id_=1,
            name="test_ds",
            path="/data/test.hdf5",
            n_objects=100,
            is_collection=False,
            catalog_tag_id=1,
        )
        mock_estimates = {
            "bpz": models.Estimates(
                id_=1, name="est1", path="/est/1.hdf5", n_objects=100, dataset_id=1, estimator_id=1
            )
        }

        with patch(
            "rail_svc.router.funcs.local_async.funcs.get_dataset_and_estimates", new_callable=AsyncMock
        ) as mock:
            mock.return_value = (mock_dataset, mock_estimates)

            response = client.get("/api/v1/funcs/get-dataset-and-estimates/1")

            assert response.status_code == 200
            data = response.json()
            assert data["dataset"]["name"] == "test_ds"


class TestCreateMatchedDataset:
    """Test /funcs/create-matched-dataset endpoint."""

    def test_success(self, client):
        """Test successful matched dataset creation."""
        mock_dataset = models.Dataset(
            id_=10,
            name="matched",
            path="/data/matched.hdf5",
            n_objects=1000,
            is_collection=True,
            catalog_tag_id=1,
        )
        mock_assocs = [
            models.DatasetAssoc(id_=1, name="matched_comp1", matched_dataset_id=10, component_dataset_id=2)
        ]

        with patch(
            "rail_svc.router.funcs.local_async.funcs.create_matched_dataset", new_callable=AsyncMock
        ) as mock:
            mock.return_value = (mock_dataset, mock_assocs)

            response = client.post(
                "/api/v1/funcs/create-matched-dataset",
                json={
                    "matched_dataset_name": "matched",
                    "catalog_tag_name": "lsst",
                    "component_dataset_names": ["comp1"],
                    "path": "/data/matched.hdf5",
                    "n_objects": 1000,
                },
            )

            assert response.status_code == 200


class TestEstimatePdfForSlice:
    """Test /funcs/estimate-pdf-for-slice endpoint."""

    def test_success(self, client):
        """Test successful PDF for slice."""
        mock_result = {"z": [0.1], "pdf": [1.0]}

        with patch(
            "rail_svc.router.funcs.local_async.funcs.estimate_pdf_for_slice", new_callable=AsyncMock
        ) as mock:
            mock.return_value = mock_result

            response = client.post(
                "/api/v1/funcs/estimate-pdf-for-slice",
                json={
                    "estimator_id": 1,
                    "dataset_id": 2,
                    "the_slice": "0:10",
                    "recompute_if_exists": False,
                },
            )

            assert response.status_code == 200


class TestEstimateDataset:
    """Test /funcs/estimate-dataset endpoint."""

    def test_success(self, client):
        """Test successful dataset estimation."""
        mock_result = models.Estimates(
            id_=1, name="est", path="/est.hdf5", n_objects=100, dataset_id=1, estimator_id=1
        )

        with patch(
            "rail_svc.router.funcs.local_async.funcs.estimate_dataset", new_callable=AsyncMock
        ) as mock:
            mock.return_value = mock_result

            response = client.post(
                "/api/v1/funcs/estimate-dataset",
                json={"estimator_id": 1, "dataset_id": 2, "raise_if_exists": False},
            )

            assert response.status_code == 200

    def test_error(self, client):
        """Test error propagation."""
        with patch(
            "rail_svc.router.funcs.local_async.funcs.estimate_dataset", new_callable=AsyncMock
        ) as mock:
            mock.side_effect = ValueError("Estimates already exist")

            response = client.post(
                "/api/v1/funcs/estimate-dataset",
                json={"estimator_id": 1, "dataset_id": 2, "raise_if_exists": True},
            )

            assert response.status_code == 500
