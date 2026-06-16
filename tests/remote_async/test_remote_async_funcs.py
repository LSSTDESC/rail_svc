"""Tests for remote_async/funcs.py — AsyncRemoteFuncs with mocked httpx transport.

Tests the full context manager flow and method delegation.
"""

import httpx
import pytest

from rail_svc.models import EstimateEnsembleResponse
from rail_svc.remote_async.funcs import AsyncRemoteFuncs


def make_transport(handler):
    """Create a mock transport from a handler function."""
    return httpx.MockTransport(handler)


@pytest.fixture
def success_handler():
    """A handler that returns 200 with a simple JSON response."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    return handler


class TestContextManager:
    """Test AsyncRemoteFuncs as async context manager."""

    @pytest.mark.asyncio
    async def test_enter_and_exit(self, success_handler):
        """Test context manager lifecycle."""
        funcs = AsyncRemoteFuncs(base_url="http://test:8000")

        # Patch the RemoteAPI to use mock transport
        from unittest.mock import patch, AsyncMock

        with patch("rail_svc.remote_async.funcs.RemoteAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_client = httpx.AsyncClient(transport=make_transport(success_handler))
            mock_api.client = mock_client
            MockAPI.return_value = mock_api

            async with funcs:
                assert funcs._client is not None

            # After exit, should be cleaned up
            assert funcs._client is None


class TestEstimatePdf:
    """Test AsyncRemoteFuncs.estimate_pdf end-to-end."""

    @pytest.mark.asyncio
    async def test_success(self):
        """Test estimate_pdf through the full stack."""
        expected = {"z": [0.1], "pdf": [0.9]}

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/funcs/estimate-pdf" in str(request.url)
            return httpx.Response(200, json=expected)

        from unittest.mock import patch, AsyncMock

        funcs = AsyncRemoteFuncs(base_url="http://test:8000")

        with patch("rail_svc.remote_async.funcs.RemoteAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.client = httpx.AsyncClient(transport=make_transport(handler))
            MockAPI.return_value = mock_api

            async with funcs:
                result = await funcs.estimate_pdf(estimator_id=1, dataset_id=2, row=0)
                assert result == expected


class TestEstimateEnsemble:
    """Test AsyncRemoteFuncs.estimate_ensemble."""

    @pytest.mark.asyncio
    async def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"output_file": "/out.hdf5", "message": "Done"}
            )

        from unittest.mock import patch, AsyncMock

        funcs = AsyncRemoteFuncs(base_url="http://test:8000")

        with patch("rail_svc.remote_async.funcs.RemoteAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.client = httpx.AsyncClient(transport=make_transport(handler))
            MockAPI.return_value = mock_api

            async with funcs:
                result = await funcs.estimate_ensemble(
                    estimator_id=1, dataset_id=2, output_file_path="/out.hdf5"
                )
                assert isinstance(result, EstimateEnsembleResponse)


class TestGetEstimatorsForDataset:
    """Test AsyncRemoteFuncs.get_estimators_for_dataset."""

    @pytest.mark.asyncio
    async def test_success(self):
        expected = [{"id_": 1, "name": "bpz", "config": {}, "model_id": 1}]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=expected)

        from unittest.mock import patch, AsyncMock

        funcs = AsyncRemoteFuncs(base_url="http://test:8000")

        with patch("rail_svc.remote_async.funcs.RemoteAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.client = httpx.AsyncClient(transport=make_transport(handler))
            MockAPI.return_value = mock_api

            async with funcs:
                result = await funcs.get_estimators_for_dataset(dataset_id=5)
                assert result == expected


class TestCreateMatchedDataset:
    """Test AsyncRemoteFuncs.create_matched_dataset."""

    @pytest.mark.asyncio
    async def test_success(self):
        expected = {"id_": 10, "name": "matched"}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=expected)

        from unittest.mock import patch, AsyncMock

        funcs = AsyncRemoteFuncs(base_url="http://test:8000")

        with patch("rail_svc.remote_async.funcs.RemoteAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.client = httpx.AsyncClient(transport=make_transport(handler))
            MockAPI.return_value = mock_api

            async with funcs:
                result = await funcs.create_matched_dataset(
                    matched_dataset_name="m",
                    catalog_tag_name="lsst",
                    component_dataset_names=["c1"],
                    path="/m.hdf5",
                    n_objects=100,
                )
                assert result == expected
