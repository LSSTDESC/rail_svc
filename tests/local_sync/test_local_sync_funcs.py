"""Tests for rail_svc.local_sync.funcs — sync wrappers around async funcs."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rail_svc import models
from rail_svc.local_sync import funcs as sync_funcs


class TestSyncWrappers:
    """Verify sync wrappers correctly delegate to async funcs via asyncio.run."""

    def test_get_dataset_and_estimates(self):
        """Test sync get_dataset_and_estimates delegates correctly."""
        expected = (
            MagicMock(spec=models.Dataset),
            [MagicMock(spec=models.Estimates)],
        )

        with patch(
            "rail_svc.local_async.funcs.get_dataset_and_estimates",
            new_callable=lambda: lambda *a, **k: AsyncMock(return_value=expected),
        ) as mock_async:
            mock_async.return_value = expected
            with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
                result = sync_funcs.get_dataset_and_estimates(1)
                assert result == expected
                mock_run.assert_called_once()

    def test_create_matched_dataset(self):
        """Test sync create_matched_dataset delegates correctly."""
        expected = (
            MagicMock(spec=models.Dataset),
            [MagicMock(spec=models.DatasetAssoc)],
        )

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.create_matched_dataset(
                matched_dataset_name="test",
                catalog_tag_name="tag",
                component_dataset_names=["c1"],
                path=None,
                n_objects=100,
            )
            assert result == expected
            mock_run.assert_called_once()

    def test_estimate_pdf(self):
        """Test sync estimate_pdf delegates correctly."""
        expected = MagicMock()

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.estimate_pdf(1, 2, 3)
            assert result == expected
            mock_run.assert_called_once()

    def test_estimate_ensemble(self):
        """Test sync estimate_ensemble delegates correctly."""
        expected = Path("/output.hdf5")

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.estimate_ensemble(1, 2, "/output.hdf5")
            assert result == expected
            mock_run.assert_called_once()

    def test_get_estimators_for_dataest(self):
        """Test sync get_estimators_for_dataest delegates correctly."""
        expected = [MagicMock(), MagicMock()]

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.get_estimators_for_dataest(1)
            assert result == expected
            mock_run.assert_called_once()

    def test_build_pdf_estimation_wrapper(self):
        """Test sync build_pdf_estimation_wrapper delegates correctly."""
        expected = MagicMock()

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.build_pdf_estimation_wrapper(1)
            assert result == expected
            mock_run.assert_called_once()

    def test_build_ensemble_estimation_wrapper(self):
        """Test sync build_ensemble_estimation_wrapper delegates correctly."""
        expected = MagicMock()

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.build_ensemble_estimation_wrapper(1)
            assert result == expected
            mock_run.assert_called_once()

    def test_load_catalog_yaml(self):
        """Test sync load_catalog_yaml delegates correctly."""
        expected = ([], [], [])

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.load_catalog_yaml("catalog.yaml")
            assert result == expected
            mock_run.assert_called_once()

    def test_get_catalog_row(self):
        """Test sync get_catalog_row delegates correctly."""
        expected = {"mag_g": [1.0]}

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.get_catalog_row(1, 0)
            assert result == expected
            mock_run.assert_called_once()

    def test_get_estimates_row(self):
        """Test sync get_estimates_row delegates correctly."""
        expected = {"z_mean": [0.5]}

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.get_estimates_row(1, 0)
            assert result == expected
            mock_run.assert_called_once()

    def test_get_data_and_estimates_data(self):
        """Test sync get_data_and_estimates_data delegates correctly."""
        expected = ({}, {})

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.get_data_and_estimates_data(1, 0)
            assert result == expected
            mock_run.assert_called_once()

    def test_build_cat_estimator_pdf_wrappers_for_dataset(self):
        """Test sync build_cat_estimator_pdf_wrappers_for_dataset delegates correctly."""
        expected = [MagicMock()]

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.build_cat_estimator_pdf_wrappers_for_dataset(1)
            assert result == expected
            mock_run.assert_called_once()

    def test_build_cat_estimator_ensemble_wrappers_for_dataset(self):
        """Test sync build_cat_estimator_ensemble_wrappers_for_dataset delegates correctly."""
        expected = [MagicMock()]

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.build_cat_estimator_ensemble_wrappers_for_dataset(1)
            assert result == expected
            mock_run.assert_called_once()

    def test_estimate_pdf_for_slice(self):
        """Test sync estimate_pdf_for_slice delegates correctly."""
        expected = MagicMock()

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.estimate_pdf_for_slice(1, 2, slice(0, 10))
            assert result == expected
            mock_run.assert_called_once()

    def test_estimate_dataset(self):
        """Test sync estimate_dataset delegates correctly."""
        expected = MagicMock(spec=models.Estimates)

        with patch("rail_svc.local_sync.funcs.asyncio.run", return_value=expected) as mock_run:
            result = sync_funcs.estimate_dataset(1, 2)
            assert result == expected
            mock_run.assert_called_once()
