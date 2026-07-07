"""Tests for client/base.py extended operations (load, read_slice, download).

Uses httpx.MockTransport to simulate HTTP responses.
"""

import json
from unittest.mock import patch

import httpx
import pytest

from rail_svc import models
from rail_svc.client.base import RemoteDatasetOperations, RemoteEstimatesOperations, RemoteModelOperations
from macon.common import LoadType


def make_dataset_ops(handler) -> tuple[httpx.AsyncClient, RemoteDatasetOperations]:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    ops = RemoteDatasetOperations(
        client=client,
        endpoint="http://test/api/v1/dataset",
        response_model=models.Dataset,
        create_model=models.DatasetCreate,
    )
    return client, ops


def make_estimates_ops(handler) -> tuple[httpx.AsyncClient, RemoteEstimatesOperations]:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    ops = RemoteEstimatesOperations(
        client=client,
        endpoint="http://test/api/v1/estimates",
        response_model=models.Estimates,
        create_model=models.EstimatesCreate,
    )
    return client, ops


def make_model_ops(handler) -> tuple[httpx.AsyncClient, RemoteModelOperations]:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    ops = RemoteModelOperations(
        client=client,
        endpoint="http://test/api/v1/model",
        response_model=models.Model,
        create_model=models.ModelCreate,
    )
    return client, ops


class TestDatasetLoad:
    """Test RemoteDatasetOperations.load."""

    @pytest.mark.asyncio
    async def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/load" in str(request.url)
            return httpx.Response(
                201,
                json={
                    "id_": 1,
                    "name": "ds",
                    "path": "/d.hdf5",
                    "n_objects": 100,
                    "is_collection": False,
                    "catalog_tag_id": 1,
                },
            )

        client, ops = make_dataset_ops(handler)
        async with client:
            result = await ops.load(
                path="/data/file.hdf5", load_type=LoadType.in_place, catalog_tag_name="lsst"
            )
            assert isinstance(result, models.Dataset)
            assert result.name == "ds"


class TestDatasetReadSlice:
    """Test RemoteDatasetOperations.read_slice."""

    @pytest.mark.asyncio
    async def test_success(self):
        json_data = json.dumps({"mag_g": [22.5, 23.1]})

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/read_slice/1" in str(request.url)
            return httpx.Response(200, json={"data": json_data})

        client, ops = make_dataset_ops(handler)
        async with client:
            result = await ops.read_slice(row_id=1, the_slice=slice(0, 2))
            assert "mag_g" in result


class TestDatasetDownload:
    """Test RemoteDatasetOperations.download."""

    @pytest.mark.asyncio
    async def test_success(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/download/1" in str(request.url)
            return httpx.Response(
                200,
                content=b"file data",
                headers={"content-disposition": 'attachment; filename="data.hdf5"'},
            )

        client, ops = make_dataset_ops(handler)
        with patch("macon.client.base.global_config.storage.download_area", str(tmp_path)):
            async with client:
                result = await ops.download(row_id=1)
                assert result.exists()
                assert result.read_bytes() == b"file data"


class TestEstimatesLoad:
    """Test RemoteEstimatesOperations.load."""

    @pytest.mark.asyncio
    async def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/load" in str(request.url)
            return httpx.Response(
                201,
                json={
                    "id_": 1,
                    "name": "est",
                    "path": "/e.hdf5",
                    "n_objects": 500,
                    "dataset_id": 1,
                    "estimator_id": 1,
                },
            )

        client, ops = make_estimates_ops(handler)
        async with client:
            result = await ops.load(
                path="/data/est.hdf5", load_type=LoadType.copy, dataset_name="ds", estimator_name="bpz"
            )
            assert isinstance(result, models.Estimates)
            assert result.name == "est"


class TestEstimatesReadSlice:
    """Test RemoteEstimatesOperations.read_slice."""

    @pytest.mark.asyncio
    async def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/read_slice/1" in str(request.url)
            return httpx.Response(200, json={"class": "qp.Ensemble", "data": []})

        client, ops = make_estimates_ops(handler)
        async with client:
            with patch("qp.from_json", return_value={"z_pdf": [0.1, 0.5]}):
                result = await ops.read_slice(row_id=1, the_slice=slice(0, 5))
                assert result is not None


class TestEstimatesDownload:
    """Test RemoteEstimatesOperations.download."""

    @pytest.mark.asyncio
    async def test_success(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/download/1" in str(request.url)
            return httpx.Response(
                200,
                content=b"estimates data",
                headers={"content-disposition": 'attachment; filename="est.hdf5"'},
            )

        client, ops = make_estimates_ops(handler)
        with patch("macon.client.base.global_config.storage.download_area", str(tmp_path)):
            async with client:
                result = await ops.download(row_id=1)
                assert result.exists()
                assert result.read_bytes() == b"estimates data"


class TestModelLoad:
    """Test RemoteModelOperations.load."""

    @pytest.mark.asyncio
    async def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/load" in str(request.url)
            return httpx.Response(
                201,
                json={
                    "id_": 1,
                    "name": "rf",
                    "path": "/m.pkl",
                    "algo_id": 1,
                    "catalog_tag_id": 1,
                },
            )

        client, ops = make_model_ops(handler)
        async with client:
            result = await ops.load(
                path="/models/rf.pkl", load_type=LoadType.in_place, algo_name="RF", catalog_tag_name="lsst"
            )
            assert isinstance(result, models.Model)
            assert result.name == "rf"


class TestModelDownload:
    """Test RemoteModelOperations.download."""

    @pytest.mark.asyncio
    async def test_success(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/download/1" in str(request.url)
            return httpx.Response(
                200,
                content=b"model data",
                headers={"content-disposition": 'attachment; filename="model.pkl"'},
            )

        client, ops = make_model_ops(handler)
        with patch("macon.client.base.global_config.storage.download_area", str(tmp_path)):
            async with client:
                result = await ops.download(row_id=1)
                assert result.exists()
                assert result.read_bytes() == b"model data"
