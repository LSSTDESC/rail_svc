"""Tests for Dataset load and read_slice operations."""

from contextlib import asynccontextmanager
from unittest.mock import patch

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from rail_svc import db, models
from rail_svc.common import LoadType
from rail_svc.db_oper.dataset import dataset


@pytest.fixture
def session_factory(engine):
    """Create a session factory that yields fresh sessions (no pre-existing transaction)."""
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def mock_get_session(session_factory):
    """Patch get_session to return fresh sessions from test engine."""

    @asynccontextmanager
    async def _get_session():
        async with session_factory() as sess:
            yield sess

    return _get_session


class TestDatasetReadSlice:
    """Test DatasetOperations.read_slice against real DB."""

    @pytest.mark.asyncio
    async def test_read_single_dataset(self, session, sample_dataset):
        """Test reading a slice from a non-collection dataset."""
        expected_data = {"mag_g": np.array([22.5, 23.1]), "mag_r": np.array([21.8, 22.3])}

        with patch(
            "rail_svc.db_oper.dataset.read_single_catalog_slice", return_value=expected_data
        ) as mock_read:
            result = await dataset.read_slice(session, sample_dataset.id_, slice(0, 2))

            mock_read.assert_called_once_with(sample_dataset.path, slice(0, 2))
            assert "mag_g" in result
            assert "mag_r" in result
            np.testing.assert_array_equal(result["mag_g"], expected_data["mag_g"])

    @pytest.mark.asyncio
    async def test_read_slice_none_reads_all(self, session, sample_dataset):
        """Test that the_slice=None reads all data."""
        expected_data = {"flux": np.array([1.0, 2.0, 3.0])}

        with patch(
            "rail_svc.db_oper.dataset.read_single_catalog_slice", return_value=expected_data
        ) as mock_read:
            result = await dataset.read_slice(session, sample_dataset.id_, None)

            mock_read.assert_called_once_with(sample_dataset.path, None)
            assert result == expected_data

    @pytest.mark.asyncio
    async def test_read_collection_dataset(self, session_factory, sample_catalog_tag):
        """Test reading from a collection dataset resolves component paths."""
        async with session_factory() as sess:
            async with sess.begin():
                # Create a collection dataset
                collection = db.Dataset(
                    name="collection_ds",
                    n_objects=100,
                    path="/data/matched.hdf5",
                    is_collection=True,
                    catalog_tag_id=sample_catalog_tag.id_,
                )
                sess.add(collection)

                # Create component datasets
                comp1 = db.Dataset(
                    name="comp1",
                    n_objects=50,
                    path="/data/comp1.hdf5",
                    is_collection=False,
                    catalog_tag_id=sample_catalog_tag.id_,
                )
                comp2 = db.Dataset(
                    name="comp2",
                    n_objects=50,
                    path="/data/comp2.hdf5",
                    is_collection=False,
                    catalog_tag_id=sample_catalog_tag.id_,
                )
                sess.add_all([comp1, comp2])
                await sess.flush()

                # Create associations
                from rail_svc.db import DatasetAssoc

                assoc1 = DatasetAssoc(
                    name="coll_comp1",
                    matched_dataset_id=collection.id_,
                    component_dataset_id=comp1.id_,
                )
                assoc2 = DatasetAssoc(
                    name="coll_comp2",
                    matched_dataset_id=collection.id_,
                    component_dataset_id=comp2.id_,
                )
                sess.add_all([assoc1, assoc2])
                await sess.flush()

        async with session_factory() as sess:
            expected_data = {"merged_col": np.array([1.0, 2.0])}

            with patch(
                "rail_svc.db_oper.dataset.read_multi_catalog_slice", return_value=expected_data
            ) as mock_read:
                result = await dataset.read_slice(sess, collection.id_, slice(0, 2))

                mock_read.assert_called_once()
                call_args = mock_read.call_args[0]
                assert call_args[0] == collection.path
                component_paths = call_args[1]
                assert "comp1" in component_paths
                assert "comp2" in component_paths
                assert result == expected_data

    @pytest.mark.asyncio
    async def test_read_slice_invalid_id(self, session):
        """Test read_slice with non-existent dataset ID raises error."""
        with pytest.raises(Exception):
            await dataset.read_slice(session, 99999, None)


class TestDatasetLoad:
    """Test DatasetOperations.load."""

    @pytest.mark.asyncio
    async def test_load_without_validation(self, mock_get_session, sample_catalog_tag, tmp_path):
        """Test loading a dataset with validate_file=False."""
        source_file = tmp_path / "catalog.hdf5"
        source_file.write_bytes(b"fake hdf5 data")

        with patch("rail_svc.db_oper.base.get_session", mock_get_session):
            result = await dataset.load(
                name="test_catalog",
                orig_path=str(source_file),
                load_type=LoadType.in_place,
                validate_file=False,
                catalog_tag_name=sample_catalog_tag.name,
                n_objects=1000,
                is_collection=False,
            )

            assert isinstance(result, models.Dataset)
            assert result.name == "test_catalog"
            assert result.n_objects == 1000
            assert "catalog.hdf5" in result.path

    @pytest.mark.asyncio
    async def test_load_with_copy(self, mock_get_session, sample_catalog_tag, tmp_path):
        """Test loading with copy creates file in archive."""
        source_file = tmp_path / "source.hdf5"
        source_file.write_bytes(b"dataset content")

        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        (archive_dir / "datasets").mkdir()

        with (
            patch("rail_svc.db_oper.base.get_session", mock_get_session),
            patch("rail_svc.common.global_config.storage.archive", str(archive_dir)),
        ):
            result = await dataset.load(
                name="copied_ds",
                orig_path=str(source_file),
                load_type=LoadType.copy,
                validate_file=False,
                catalog_tag_name=sample_catalog_tag.name,
                n_objects=500,
                is_collection=False,
            )

            assert result.name == "copied_ds"
            copied_path = archive_dir / "datasets" / "copied_ds_source.hdf5"
            assert copied_path.exists()
            assert copied_path.read_bytes() == b"dataset content"
