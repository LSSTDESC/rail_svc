"""Tests for custom router endpoints (dataset/estimates/model load, read_slice, download).

Uses FastAPI TestClient with mocked local_async operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rail_svc import models
from rail_svc.router.app import create_fastapi_app


@pytest.fixture
def client():
    """Create a test client with the full app (mocking DB init)."""
    with patch("rail_svc.router.app.init_db"):
        app = create_fastapi_app(api_prefix="/api/v1")
    return TestClient(app)


class TestDatasetLoad:
    """Test /dataset/load endpoint."""

    def test_success(self, client):
        mock_result = models.Dataset(
            id_=1, name="ds", path="/d.hdf5", n_objects=100, is_collection=False, catalog_tag_id=1
        )

        with patch("rail_svc.router.rail_svc.local_async.dataset.load", new_callable=AsyncMock) as mock:
            mock.return_value = mock_result

            response = client.post(
                "/api/v1/dataset/load",
                json={
                    "path": "/data/cat.hdf5",
                    "load_type": "in_place",
                    "data": {"catalog_tag_name": "lsst"},
                },
                params={"validate": True},
            )

            assert response.status_code == 201
            assert response.json()["name"] == "ds"


class TestDatasetReadSlice:
    """Test /dataset/read_slice endpoint."""

    def test_success(self, client):

        with (
            patch("rail_svc.router.rail_svc.local_async.dataset.read_slice", new_callable=AsyncMock) as mock_read,
            patch("rail_svc.router.rail_svc.tables_io.convert", return_value='{"mag_g": [22.5]}'),
        ):
            mock_read.return_value = {"mag_g": [22.5]}

            response = client.get("/api/v1/dataset/read_slice/1", params={"read_slice": "0:5"})

            assert response.status_code == 200
            assert "data" in response.json()

    def test_no_slice(self, client):
        with (
            patch("rail_svc.router.rail_svc.local_async.dataset.read_slice", new_callable=AsyncMock) as mock_read,
            patch("rail_svc.router.rail_svc.tables_io.convert", return_value='{"flux": [1.0]}'),
        ):
            mock_read.return_value = {"flux": [1.0]}

            response = client.get("/api/v1/dataset/read_slice/1")

            assert response.status_code == 200


class TestDatasetDownload:
    """Test /dataset/download endpoint."""

    def test_success(self, client, tmp_path):
        data_file = tmp_path / "data.hdf5"
        data_file.write_bytes(b"file content")

        mock_result = MagicMock()
        mock_result.path = str(data_file)

        with (
            patch("rail_svc.router.rail_svc.local_async.dataset.get_row", new_callable=AsyncMock) as mock_get,
            patch("rail_svc.router.rail_svc.global_config.storage.archive", str(tmp_path)),
        ):
            mock_get.return_value = mock_result

            response = client.get("/api/v1/dataset/download/1")

            assert response.status_code == 200


class TestEstimatesLoad:
    """Test /estimates/load endpoint."""

    def test_success(self, client):
        mock_result = models.Estimates(
            id_=1, name="est", path="/e.hdf5", n_objects=500, dataset_id=1, estimator_id=1
        )

        with patch("rail_svc.router.rail_svc.local_async.estimates.load", new_callable=AsyncMock) as mock:
            mock.return_value = mock_result

            response = client.post(
                "/api/v1/estimates/load",
                json={
                    "path": "/data/est.hdf5",
                    "load_type": "in_place",
                    "data": {"dataset_name": "ds", "estimator_name": "bpz"},
                },
                params={"validate": True},
            )

            assert response.status_code == 201
            assert response.json()["name"] == "est"


class TestEstimatesReadSlice:
    """Test /estimates/read_slice endpoint."""

    def test_success(self, client):
        mock_ensemble = MagicMock()
        mock_ensemble.to_json.return_value = {"class": "qp.Ensemble", "data": []}

        with patch("rail_svc.router.rail_svc.local_async.estimates.read_slice", new_callable=AsyncMock) as mock:
            mock.return_value = mock_ensemble

            response = client.get("/api/v1/estimates/read_slice/1", params={"read_slice": "0:5"})

            assert response.status_code == 200


class TestEstimatesDownload:
    """Test /estimates/download endpoint."""

    def test_success(self, client, tmp_path):
        data_file = tmp_path / "est.hdf5"
        data_file.write_bytes(b"estimates content")

        mock_result = MagicMock()
        mock_result.path = str(data_file)

        with (
            patch("rail_svc.router.rail_svc.local_async.estimates.get_row", new_callable=AsyncMock) as mock_get,
            patch("rail_svc.router.rail_svc.global_config.storage.archive", str(tmp_path)),
        ):
            mock_get.return_value = mock_result

            response = client.get("/api/v1/estimates/download/1")

            assert response.status_code == 200


class TestModelLoad:
    """Test /model/load endpoint."""

    def test_success(self, client):
        mock_result = models.Model(id_=1, name="rf", path="/m.pkl", algo_id=1, catalog_tag_id=1)

        with patch("rail_svc.router.rail_svc.local_async.model.load", new_callable=AsyncMock) as mock:
            mock.return_value = mock_result

            response = client.post(
                "/api/v1/model/load",
                json={
                    "path": "/models/rf.pkl",
                    "load_type": "in_place",
                    "data": {"algo_name": "RF", "catalog_tag_name": "lsst"},
                },
                params={"validate": True},
            )

            assert response.status_code == 201
            assert response.json()["name"] == "rf"


class TestModelDownload:
    """Test /model/download endpoint."""

    def test_success(self, client, tmp_path):
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"model content")

        mock_result = MagicMock()
        mock_result.path = str(model_file)

        with (
            patch("rail_svc.router.rail_svc.local_async.model.get_row", new_callable=AsyncMock) as mock_get,
            patch("rail_svc.router.rail_svc.global_config.storage.archive", str(tmp_path)),
        ):
            mock_get.return_value = mock_result

            response = client.get("/api/v1/model/download/1")

            assert response.status_code == 200
