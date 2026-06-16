"""Tests for client/funcs.py — RemoteFuncsOperations with mocked httpx transport.

Uses httpx.MockTransport to simulate server responses without a real server.
"""

from pathlib import Path
from typing import Any

import httpx
import pytest

from rail_svc.client.funcs import RemoteFuncsOperations
from rail_svc.models import (
    EstimateEnsembleResponse,
    GetDataAndEstimatesDataResponse,
    GetDatasetAndEstimatesResponse,
    LoadCatalogYamlResponse,
    RemoteAPIError,
)


def make_client(handler) -> tuple[httpx.AsyncClient, RemoteFuncsOperations]:
    """Create an async client + RemoteFuncsOperations with a mock transport."""
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    ops = RemoteFuncsOperations(client=client, endpoint="http://test/api/v1/funcs")
    return client, ops


class TestEstimatePdf:
    """Test RemoteFuncsOperations.estimate_pdf."""

    @pytest.mark.asyncio
    async def test_success(self):
        expected = {"z": [0.1, 0.5], "pdf": [0.3, 0.7]}

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/estimate-pdf" in str(request.url)
            return httpx.Response(200, json=expected)

        client, ops = make_client(handler)
        async with client:
            result = await ops.estimate_pdf(estimator_id=1, dataset_id=2, row=5)
            assert result == expected

    @pytest.mark.asyncio
    async def test_server_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "Internal error"})

        client, ops = make_client(handler)
        async with client:
            with pytest.raises(RemoteAPIError, match="500"):
                await ops.estimate_pdf(estimator_id=1, dataset_id=2, row=5)


class TestEstimateEnsemble:
    """Test RemoteFuncsOperations.estimate_ensemble."""

    @pytest.mark.asyncio
    async def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/estimate-ensemble" in str(request.url)
            return httpx.Response(
                200, json={"output_file": "/output/est.hdf5", "message": "Done"}
            )

        client, ops = make_client(handler)
        async with client:
            result = await ops.estimate_ensemble(
                estimator_id=1, dataset_id=2, output_file_path="/output/est.hdf5"
            )
            assert isinstance(result, EstimateEnsembleResponse)
            assert result.output_file == "/output/est.hdf5"


class TestGetEstimatorsForDataset:
    """Test RemoteFuncsOperations.get_estimators_for_dataset."""

    @pytest.mark.asyncio
    async def test_success(self):
        expected = [
            {"id_": 1, "name": "bpz", "config": {}, "model_id": 1},
            {"id_": 2, "name": "flexz", "config": {"n": 5}, "model_id": 2},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/get-estimators-for-dataset/5" in str(request.url)
            return httpx.Response(200, json=expected)

        client, ops = make_client(handler)
        async with client:
            result = await ops.get_estimators_for_dataset(dataset_id=5)
            assert len(result) == 2
            assert result[0]["name"] == "bpz"


class TestLoadCatalogYaml:
    """Test RemoteFuncsOperations.load_catalog_yaml."""

    @pytest.mark.asyncio
    async def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/load-catalog-yaml" in str(request.url)
            return httpx.Response(
                200,
                json={"bands": [], "catalog_tags": [], "catalog_band_assocs": []},
            )

        client, ops = make_client(handler)
        async with client:
            result = await ops.load_catalog_yaml(catalog_yaml=Path("/path/cat.yaml"))
            assert isinstance(result, LoadCatalogYamlResponse)
            assert result.bands == []

    @pytest.mark.asyncio
    async def test_with_filter_dir(self):
        def handler(request: httpx.Request) -> httpx.Response:
            import json

            body = json.loads(request.content)
            assert body["filter_dir"] == "/path/filters"
            return httpx.Response(
                200,
                json={"bands": [], "catalog_tags": [], "catalog_band_assocs": []},
            )

        client, ops = make_client(handler)
        async with client:
            await ops.load_catalog_yaml(
                catalog_yaml=Path("/path/cat.yaml"),
                filter_dir=Path("/path/filters"),
            )


class TestGetDatasetAndEstimates:
    """Test RemoteFuncsOperations.get_dataset_and_estimates."""

    @pytest.mark.asyncio
    async def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/get-dataset-and-estimates/1" in str(request.url)
            return httpx.Response(
                200,
                json={
                    "dataset": {
                        "id_": 1,
                        "name": "ds",
                        "path": "/d.hdf5",
                        "n_objects": 100,
                        "is_collection": False,
                        "catalog_tag_id": 1,
                    },
                    "estimates": {},
                },
            )

        client, ops = make_client(handler)
        async with client:
            result = await ops.get_dataset_and_estimates(dataset_id=1)
            assert isinstance(result, GetDatasetAndEstimatesResponse)
            assert result.dataset.name == "ds"


class TestGetDataAndEstimatesData:
    """Test RemoteFuncsOperations.get_data_and_estimates_data."""

    @pytest.mark.asyncio
    async def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/get-data-and-estimates-data/1/5" in str(request.url)
            return httpx.Response(
                200,
                json={"data": {"mag_g": [22.5]}, "estimates_dict": {}},
            )

        client, ops = make_client(handler)
        async with client:
            result = await ops.get_data_and_estimates_data(dataset_id=1, row=5)
            assert isinstance(result, GetDataAndEstimatesDataResponse)


class TestCreateMatchedDataset:
    """Test RemoteFuncsOperations.create_matched_dataset."""

    @pytest.mark.asyncio
    async def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/create-matched-dataset" in str(request.url)
            return httpx.Response(200, json={"id_": 10, "name": "matched"})

        client, ops = make_client(handler)
        async with client:
            result = await ops.create_matched_dataset(
                matched_dataset_name="matched",
                catalog_tag_name="lsst",
                component_dataset_names=["c1", "c2"],
                path="/m.hdf5",
                n_objects=1000,
            )
            assert result["name"] == "matched"


class TestEstimatePdfForSlice:
    """Test RemoteFuncsOperations.estimate_pdf_for_slice."""

    @pytest.mark.asyncio
    async def test_success(self):
        expected = {"z": [0.1], "pdf": [1.0]}

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/estimate-pdf-for-slice" in str(request.url)
            return httpx.Response(200, json=expected)

        client, ops = make_client(handler)
        async with client:
            result = await ops.estimate_pdf_for_slice(
                estimator_id=1, dataset_id=2, the_slice="0:10"
            )
            assert result == expected


class TestEstimateDataset:
    """Test RemoteFuncsOperations.estimate_dataset."""

    @pytest.mark.asyncio
    async def test_success(self):
        expected = {"id_": 1, "name": "est", "path": "/e.hdf5"}

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/estimate-dataset" in str(request.url)
            return httpx.Response(200, json=expected)

        client, ops = make_client(handler)
        async with client:
            result = await ops.estimate_dataset(estimator_id=1, dataset_id=2)
            assert result == expected


class TestHandleResponse:
    """Test error handling in _handle_response."""

    @pytest.mark.asyncio
    async def test_error_with_details(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"detail": {"error": "Validation failed", "details": "field x required"}},
            )

        client, ops = make_client(handler)
        async with client:
            with pytest.raises(RemoteAPIError, match="Validation failed.*field x required"):
                await ops.estimate_pdf(estimator_id=1, dataset_id=2, row=0)

    @pytest.mark.asyncio
    async def test_error_plain_text(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="Bad Gateway")

        client, ops = make_client(handler)
        async with client:
            with pytest.raises(RemoteAPIError, match="502"):
                await ops.estimate_pdf(estimator_id=1, dataset_id=2, row=0)
