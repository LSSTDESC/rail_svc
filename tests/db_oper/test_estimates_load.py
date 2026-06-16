"""Tests for Estimates load and read_slice operations."""

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import qp
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from rail_svc import db, models
from rail_svc.common import LoadType
from rail_svc.db_oper.estimates import estimates


@pytest.fixture
def mock_get_session(engine):
    """Patch get_session to return fresh sessions from test engine."""
    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def _get_session():
        async with factory() as sess:
            yield sess

    return _get_session


class TestEstimatesReadSlice:
    """Test EstimatesOperations.read_slice against real DB."""

    @pytest.mark.asyncio
    async def test_read_slice(self, session, sample_estimates):
        """Test reading a slice from estimates."""
        mock_ensemble = MagicMock(spec=qp.Ensemble)

        with patch(
            "rail_svc.db_oper.estimates.read_estimates_slice", return_value=mock_ensemble
        ) as mock_read:
            result = await estimates.read_slice(session, sample_estimates.id_, slice(0, 5))

            mock_read.assert_called_once_with(sample_estimates.path, slice(0, 5))
            assert result == mock_ensemble

    @pytest.mark.asyncio
    async def test_read_slice_none(self, session, sample_estimates):
        """Test reading all data when the_slice is None."""
        mock_ensemble = MagicMock(spec=qp.Ensemble)

        with patch(
            "rail_svc.db_oper.estimates.read_estimates_slice", return_value=mock_ensemble
        ) as mock_read:
            result = await estimates.read_slice(session, sample_estimates.id_, None)

            mock_read.assert_called_once_with(sample_estimates.path, None)
            assert result == mock_ensemble

    @pytest.mark.asyncio
    async def test_read_slice_single_row(self, session, sample_estimates):
        """Test reading a single row index."""
        mock_data = {"z_pdf": np.array([0.1, 0.5, 0.3, 0.1])}

        with patch(
            "rail_svc.db_oper.estimates.read_estimates_slice", return_value=mock_data
        ) as mock_read:
            result = await estimates.read_slice(session, sample_estimates.id_, 42)

            mock_read.assert_called_once_with(sample_estimates.path, 42)

    @pytest.mark.asyncio
    async def test_read_slice_invalid_id(self, session):
        """Test read_slice with non-existent estimates ID."""
        with pytest.raises(Exception):
            await estimates.read_slice(session, 99999, None)


class TestEstimatesLoad:
    """Test EstimatesOperations.load."""

    @pytest.mark.asyncio
    async def test_load_without_validation(self, mock_get_session, sample_dataset, sample_estimator, tmp_path):
        """Test loading estimates with validate_file=False."""
        source_file = tmp_path / "estimates.hdf5"
        source_file.write_bytes(b"fake qp data")

        with patch("rail_svc.db_oper.base.get_session", mock_get_session):
            result = await estimates.load(
                name="test_estimates",
                orig_path=str(source_file),
                load_type=LoadType.in_place,
                validate_file=False,
                dataset_name=sample_dataset.name,
                estimator_name=sample_estimator.name,
                n_objects=5000,
            )

            assert isinstance(result, models.Estimates)
            assert result.name == "test_estimates"
            assert result.n_objects == 5000
            assert result.dataset_id == sample_dataset.id_
            assert result.estimator_id == sample_estimator.id_

    @pytest.mark.asyncio
    async def test_load_with_copy(self, mock_get_session, sample_dataset, sample_estimator, tmp_path):
        """Test loading estimates with copy creates file in archive."""
        source_file = tmp_path / "est_output.hdf5"
        source_file.write_bytes(b"qp ensemble data")

        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        (archive_dir / "estimates").mkdir()

        with (
            patch("rail_svc.db_oper.base.get_session", mock_get_session),
            patch("rail_svc.common.global_config.storage.archive", str(archive_dir)),
        ):
            result = await estimates.load(
                name="copied_est",
                orig_path=str(source_file),
                load_type=LoadType.copy,
                validate_file=False,
                dataset_name=sample_dataset.name,
                estimator_name=sample_estimator.name,
                n_objects=2000,
            )

            assert result.name == "copied_est"
            copied_path = archive_dir / "estimates" / "copied_est_est_output.hdf5"
            assert copied_path.exists()

    @pytest.mark.asyncio
    async def test_load_resolves_by_name(self, mock_get_session, sample_dataset, sample_estimator, tmp_path):
        """Test loading estimates resolving foreign keys by name."""
        source_file = tmp_path / "by_name.hdf5"
        source_file.write_bytes(b"data")

        with patch("rail_svc.db_oper.base.get_session", mock_get_session):
            result = await estimates.load(
                name="by_name_est",
                orig_path=str(source_file),
                load_type=LoadType.in_place,
                validate_file=False,
                dataset_name=sample_dataset.name,
                estimator_name=sample_estimator.name,
                n_objects=100,
            )

            assert result.dataset_id == sample_dataset.id_
            assert result.estimator_id == sample_estimator.id_
