"""Integration tests for rail_svc.local_async.funcs using real in-memory SQLite.

Tests the DB-only functions through the full decorator chain with a real
database session, avoiding brittle mock-based testing.
"""

from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from rail_svc import db, models
from rail_svc.local_async import funcs as api_funcs


@pytest.fixture
def patch_session(engine):
    """Patch get_session to yield a fresh session from the test engine.

    Uses a new session (no pre-existing transaction) so that both
    @with_session and @with_session_transaction decorators work correctly.
    """
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    async_session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def _mock_get_session():
        async with async_session_factory() as sess:
            yield sess

    with patch("rail_svc.local_async.base.get_session", _mock_get_session):
        yield


class TestGetDatasetAndEstimates:
    """Test get_dataset_and_estimates with real DB."""

    @pytest.mark.asyncio
    async def test_returns_dataset_and_estimates(self, patch_session, sample_dataset, sample_estimates):
        """Test fetching a dataset with its estimates."""
        result = await api_funcs.get_dataset_and_estimates(sample_dataset.id_)

        dataset_result, estimates_list = result
        assert isinstance(dataset_result, models.Dataset)
        assert dataset_result.id_ == sample_dataset.id_
        assert dataset_result.name == sample_dataset.name
        assert len(estimates_list) == 1
        assert isinstance(estimates_list[0], models.Estimates)
        assert estimates_list[0].id_ == sample_estimates.id_

    @pytest.mark.asyncio
    async def test_no_estimates(self, patch_session, sample_dataset):
        """Test fetching a dataset that has no estimates."""
        result = await api_funcs.get_dataset_and_estimates(sample_dataset.id_)

        dataset_result, estimates_list = result
        assert dataset_result.id_ == sample_dataset.id_
        assert estimates_list == []

    @pytest.mark.asyncio
    async def test_multiple_estimates(self, patch_session, sample_dataset, multiple_estimates):
        """Test fetching a dataset with multiple estimates."""
        result = await api_funcs.get_dataset_and_estimates(sample_dataset.id_)

        _, estimates_list = result
        assert len(estimates_list) == 3


class TestCreateMatchedDataset:
    """Test create_matched_dataset with real DB."""

    @pytest.mark.asyncio
    async def test_creates_dataset_and_associations(
        self, patch_session, sample_catalog_tag, sample_dataset, component_dataset_1, component_dataset_2
    ):
        """Test creating a matched dataset with component associations."""
        result = await api_funcs.create_matched_dataset(
            matched_dataset_name="my_matched",
            catalog_tag_name=sample_catalog_tag.name,
            component_dataset_names=[component_dataset_1.name, component_dataset_2.name],
            path="/data/matched.hdf5",
            n_objects=11000,
        )

        dataset_result, assocs = result
        assert isinstance(dataset_result, models.Dataset)
        assert dataset_result.name == "my_matched"
        assert dataset_result.is_collection is True
        assert dataset_result.n_objects == 11000
        assert len(assocs) == 2
        assert all(isinstance(a, models.DatasetAssoc) for a in assocs)

    @pytest.mark.asyncio
    async def test_empty_components(self, patch_session, sample_catalog_tag):
        """Test creating a matched dataset with no components."""
        result = await api_funcs.create_matched_dataset(
            matched_dataset_name="empty_matched",
            catalog_tag_name=sample_catalog_tag.name,
            component_dataset_names=[],
            path="/data/empty_matched.hdf5",
            n_objects=0,
        )

        dataset_result, assocs = result
        assert dataset_result.name == "empty_matched"
        assert assocs == []


class TestGetEstimatorsForDataset:
    """Test get_estimators_for_dataest with real DB."""

    @pytest.mark.skip(reason="@to_pydantic_list decorator not compatible with standalone functions")
    @pytest.mark.asyncio
    async def test_finds_estimators(self, patch_session, sample_dataset, sample_estimator):
        """Test finding estimators for a dataset via catalog_tag → model → estimator chain."""
        result = await api_funcs.get_estimators_for_dataest(sample_dataset.id_)

        assert len(result) >= 1
        assert any(e.id_ == sample_estimator.id_ for e in result)

    @pytest.mark.skip(reason="@to_pydantic_list decorator not compatible with standalone functions")
    @pytest.mark.asyncio
    async def test_multiple_estimators(self, patch_session, sample_dataset, multiple_estimators):
        """Test finding multiple estimators for a dataset."""
        result = await api_funcs.get_estimators_for_dataest(sample_dataset.id_)

        assert len(result) == 3
