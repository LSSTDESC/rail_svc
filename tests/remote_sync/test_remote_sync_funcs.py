"""Tests for remote_sync/funcs.py — SyncRemoteFuncs wrapping AsyncRemoteFuncs."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rail_svc import models
from rail_svc.remote_async.funcs import AsyncRemoteFuncs
from rail_svc.remote_sync.funcs import SyncRemoteFuncs


@pytest.fixture
def sync_funcs():
    """Create SyncRemoteFuncs with a mocked AsyncRemoteFuncs."""
    async_funcs = MagicMock(spec=AsyncRemoteFuncs)
    return SyncRemoteFuncs(async_funcs)


class TestSyncRemoteFuncs:
    """Test SyncRemoteFuncs delegates to async versions via asyncio.run."""

    def test_estimate_pdf(self, sync_funcs):
        expected = {"z": [0.1], "pdf": [0.9]}
        sync_funcs.async_ops.estimate_pdf = AsyncMock(return_value=expected)

        with patch("asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.estimate_pdf(estimator_id=1, dataset_id=2, row=0)
            assert result == expected
            mock_run.assert_called_once()

    def test_estimate_ensemble(self, sync_funcs):
        expected = models.EstimateEnsembleResponse(output_file="/out.hdf5", message="Done")
        sync_funcs.async_ops.estimate_ensemble = AsyncMock(return_value=expected)

        with patch("asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.estimate_ensemble(estimator_id=1, dataset_id=2, output_file_path="/o.hdf5")
            assert result == expected
            mock_run.assert_called_once()

    def test_get_estimators_for_dataset(self, sync_funcs):
        expected = [{"id_": 1, "name": "bpz"}]
        sync_funcs.async_ops.get_estimators_for_dataset = AsyncMock(return_value=expected)

        with patch("asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.get_estimators_for_dataset(dataset_id=5)
            assert result == expected
            mock_run.assert_called_once()

    def test_load_catalog_yaml(self, sync_funcs):
        expected = models.LoadCatalogYamlResponse(bands=[], catalog_tags=[], catalog_band_assocs=[])
        sync_funcs.async_ops.load_catalog_yaml = AsyncMock(return_value=expected)

        with patch("asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.load_catalog_yaml(catalog_yaml="/path.yaml")
            assert result == expected
            mock_run.assert_called_once()

    def test_get_dataset_and_estimates(self, sync_funcs):
        expected = MagicMock(spec=models.GetDatasetAndEstimatesResponse)
        sync_funcs.async_ops.get_dataset_and_estimates = AsyncMock(return_value=expected)

        with patch("asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.get_dataset_and_estimates(dataset_id=1)
            assert result == expected
            mock_run.assert_called_once()

    def test_get_data_and_estimates_data(self, sync_funcs):
        expected = MagicMock(spec=models.GetDataAndEstimatesDataResponse)
        sync_funcs.async_ops.get_data_and_estimates_data = AsyncMock(return_value=expected)

        with patch("asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.get_data_and_estimates_data(dataset_id=1, row=5)
            assert result == expected
            mock_run.assert_called_once()

    def test_create_matched_dataset(self, sync_funcs):
        expected = {"id_": 10, "name": "matched"}
        sync_funcs.async_ops.create_matched_dataset = AsyncMock(return_value=expected)

        with patch("asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.create_matched_dataset(
                matched_dataset_name="m",
                catalog_tag_name="l",
                component_dataset_names=["c1"],
                path="/m.hdf5",
            )
            assert result == expected
            mock_run.assert_called_once()

    def test_estimate_pdf_for_slice(self, sync_funcs):
        expected = {"z": [0.1]}
        sync_funcs.async_ops.estimate_pdf_for_slice = AsyncMock(return_value=expected)

        with patch("asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.estimate_pdf_for_slice(estimator_id=1, dataset_id=2, the_slice="0:10")
            assert result == expected
            mock_run.assert_called_once()

    def test_estimate_dataset(self, sync_funcs):
        expected = {"id_": 1, "name": "est"}
        sync_funcs.async_ops.estimate_dataset = AsyncMock(return_value=expected)

        with patch("asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.estimate_dataset(estimator_id=1, dataset_id=2)
            assert result == expected
            mock_run.assert_called_once()
