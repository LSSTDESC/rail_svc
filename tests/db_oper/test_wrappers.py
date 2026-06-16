"""Integration tests for rail_svc.db_oper.wrappers using real in-memory SQLite."""

import pytest

from rail_svc.db_oper.wrappers import _get_estimator_components


class TestGetEstimatorComponents:
    """Test _get_estimator_components against real DB."""

    @pytest.mark.asyncio
    async def test_fetches_all_components(self, session, sample_estimator, sample_model, sample_algorithm, sample_catalog_tag):
        """Test that all related components are fetched correctly."""
        estimator_obj, model_obj, algo_obj, catalog_tag_obj = await _get_estimator_components(
            session, sample_estimator.id_
        )

        assert estimator_obj.id_ == sample_estimator.id_
        assert estimator_obj.name == sample_estimator.name
        assert model_obj.id_ == sample_model.id_
        assert algo_obj.id_ == sample_algorithm.id_
        assert catalog_tag_obj.id_ == sample_catalog_tag.id_

    @pytest.mark.asyncio
    async def test_invalid_estimator_id(self, session):
        """Test error when estimator doesn't exist."""
        with pytest.raises(Exception):
            await _get_estimator_components(session, 99999)

    @pytest.mark.asyncio
    async def test_returns_correct_types(self, session, sample_estimator):
        """Test that returned objects are the correct ORM types."""
        from rail_svc import db

        estimator_obj, model_obj, algo_obj, catalog_tag_obj = await _get_estimator_components(
            session, sample_estimator.id_
        )

        assert isinstance(estimator_obj, db.Estimator)
        assert isinstance(model_obj, db.Model)
        assert isinstance(algo_obj, db.Algorithm)
        assert isinstance(catalog_tag_obj, db.CatalogTag)
