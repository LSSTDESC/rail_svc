"""Shared parametrized tests for all database entity models.

Tests the common Base class interface (tablename, pydantic integration, CRUD)
across all entities without duplicating per-entity test files.
"""

import pytest
from sqlalchemy import select

from rail_svc import db, models


ENTITY_CONFIGS = [
    {
        "db_class": db.Algorithm,
        "pydantic_class": models.Algorithm,
        "create_class": models.AlgorithmCreate,
        "tablename": "algorithm",
        "fixture": "sample_algorithm",
    },
    {
        "db_class": db.Band,
        "pydantic_class": models.Band,
        "create_class": models.BandCreate,
        "tablename": "band",
        "fixture": "sample_band",
    },
    {
        "db_class": db.CatalogTag,
        "pydantic_class": models.CatalogTag,
        "create_class": models.CatalogTagCreate,
        "tablename": "catalog_tag",
        "fixture": "sample_catalog_tag",
    },
    {
        "db_class": db.CatalogBandAssoc,
        "pydantic_class": models.CatalogBandAssoc,
        "create_class": models.CatalogBandAssocCreate,
        "tablename": "catalog_band_assoc",
        "fixture": "sample_catalog_band_assoc",
    },
    {
        "db_class": db.Dataset,
        "pydantic_class": models.Dataset,
        "create_class": models.DatasetCreate,
        "tablename": "dataset",
        "fixture": "sample_dataset",
    },
    {
        "db_class": db.DatasetAssoc,
        "pydantic_class": models.DatasetAssoc,
        "create_class": models.DatasetAssocCreate,
        "tablename": "dataset_assoc",
        "fixture": "sample_dataset_assoc",
    },
    {
        "db_class": db.Estimates,
        "pydantic_class": models.Estimates,
        "create_class": models.EstimatesCreate,
        "tablename": "estimates",
        "fixture": "sample_estimates",
    },
    {
        "db_class": db.Estimator,
        "pydantic_class": models.Estimator,
        "create_class": models.EstimatorCreate,
        "tablename": "estimator",
        "fixture": "sample_estimator",
    },
    {
        "db_class": db.Model,
        "pydantic_class": models.Model,
        "create_class": models.ModelCreate,
        "tablename": "model",
        "fixture": "sample_model",
    },
]


@pytest.fixture(params=ENTITY_CONFIGS, ids=[c["tablename"] for c in ENTITY_CONFIGS])
def entity_config(request):
    """Parametrized fixture providing entity configuration."""
    return request.param


@pytest.fixture
def entity_instance(request, entity_config):
    """Get the sample instance for the current entity."""
    return request.getfixturevalue(entity_config["fixture"])


class TestEntityTableMetadata:
    """Test that all entities have correct table metadata."""

    def test_tablename(self, entity_config):
        assert entity_config["db_class"].__tablename__ == entity_config["tablename"]

    def test_class_string(self, entity_config):
        assert entity_config["db_class"].class_string() == entity_config["tablename"]

    def test_pydantic_create_class(self, entity_config):
        assert entity_config["db_class"].pydantic_create_class() == entity_config["create_class"]

    def test_pydantic_model_class(self, entity_config):
        assert entity_config["db_class"].pydantic_model_class() == entity_config["pydantic_class"]


class TestEntityPydanticConversion:
    """Test Pydantic conversion for all entities."""

    def test_to_pydantic(self, entity_config, entity_instance):
        pydantic_obj = entity_config["db_class"].to_pydantic(entity_instance)
        assert isinstance(pydantic_obj, entity_config["pydantic_class"])
        assert pydantic_obj.id_ == entity_instance.id_

    def test_to_pydantic_dict(self, entity_config, entity_instance):
        data = entity_config["db_class"].to_pydantic_dict(entity_instance)
        assert isinstance(data, dict)
        assert data["id_"] == entity_instance.id_

    def test_to_pydantic_list(self, entity_config, entity_instance):
        result = entity_config["db_class"].to_pydantic_list([entity_instance])
        assert len(result) == 1
        assert isinstance(result[0], entity_config["pydantic_class"])

    def test_to_pydantic_dict_list(self, entity_config, entity_instance):
        result = entity_config["db_class"].to_pydantic_dict_list([entity_instance])
        assert len(result) == 1
        assert isinstance(result[0], dict)


class TestEntityCRUD:
    """Test basic CRUD operations for all entities."""

    @pytest.mark.asyncio
    async def test_query_by_id(self, session, entity_config, entity_instance):
        result = await session.get(entity_config["db_class"], entity_instance.id_)
        assert result is not None
        assert result.id_ == entity_instance.id_

    @pytest.mark.asyncio
    async def test_query_by_name(self, session, entity_config, entity_instance):
        db_class = entity_config["db_class"]
        result = await session.execute(
            select(db_class).where(db_class.name == entity_instance.name)
        )
        row = result.scalar_one()
        assert row.id_ == entity_instance.id_

    @pytest.mark.asyncio
    async def test_delete(self, session, entity_config, entity_instance):
        entity_id = entity_instance.id_
        await session.delete(entity_instance)
        await session.commit()

        result = await session.get(entity_config["db_class"], entity_id)
        assert result is None


class TestEntityEdgeCases:
    """Test edge cases common to all entities."""

    @pytest.mark.asyncio
    async def test_query_nonexistent(self, session, entity_config):
        result = await session.get(entity_config["db_class"], 99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, session, entity_config, entity_instance):
        db_class = entity_config["db_class"]
        results = []
        for _ in range(5):
            result = await session.execute(
                select(db_class).where(db_class.id_ == entity_instance.id_)
            )
            results.append(result.scalar_one())
        assert len(results) == 5
        assert all(r.id_ == entity_instance.id_ for r in results)

    @pytest.mark.asyncio
    async def test_refresh_after_commit(self, session, entity_config, entity_instance):
        await session.commit()
        await session.refresh(entity_instance)
        assert entity_instance.id_ is not None

    def test_has_type_annotations(self, entity_config):
        db_class = entity_config["db_class"]
        assert hasattr(db_class, "id_")
        assert hasattr(db_class, "name")
