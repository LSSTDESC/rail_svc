"""Unit tests for CatalogTag table operations."""

import pytest
from macon.db_oper.base import TableContext

from rail_svc.db import CatalogTag
from rail_svc.db_oper.catalog_tag import CatalogTagOperations, catalog_tag
from rail_svc.models import CatalogTag as CatalogTagModel

# ============================================================================
# CatalogTagOperations class tests
# ============================================================================


async def test_catalog_tag_operations_inherits_crud_methods(session, sample_catalog_tag):
    """Test that CatalogTagOperations inherits CRUD methods from base."""
    context = TableContext.from_db_class(CatalogTag)
    ops = CatalogTagOperations(context)

    # Should have get_row method from base class
    result = await ops.get_row(session, sample_catalog_tag.id_)
    assert result.id_ == sample_catalog_tag.id_


@pytest.mark.asyncio
async def test_catalog_tag_operations_inherits_filter_methods(session, multiple_catalog_tags):
    """Test that CatalogTagOperations inherits filter methods from base."""
    from macon.models.filtering import Filter, FilterOp

    context = TableContext.from_db_class(CatalogTag)
    ops = CatalogTagOperations(context)

    # Should have filter_rows method from base class
    filters = [Filter(field="name", op=FilterOp.EQ, value="roman")]
    results = await ops.filter_rows(session, filters=filters)

    assert len(results) == 1
    assert results[0].name == "roman"


@pytest.mark.asyncio
async def test_catalog_tag_operations_inherits_pydantic_conversion(session, sample_catalog_tag):
    """Test that CatalogTagOperations inherits Pydantic conversion methods."""
    context = TableContext.from_db_class(CatalogTag)
    ops = CatalogTagOperations(context)

    # Should have to_pydantic method from base class
    pydantic_obj = ops.to_pydantic(sample_catalog_tag)

    assert isinstance(pydantic_obj, CatalogTagModel)
    assert pydantic_obj.name == sample_catalog_tag.name


# ============================================================================
# Module singleton functionality tests
# ============================================================================


@pytest.mark.asyncio
async def test_module_singleton_get_row(session, sample_catalog_tag):
    """Test that module singleton can perform get_row operation."""
    result = await catalog_tag.get_row(session, sample_catalog_tag.id_)

    assert result.id_ == sample_catalog_tag.id_
    assert result.name == sample_catalog_tag.name


@pytest.mark.asyncio
async def test_module_singleton_create_row(session):
    """Test that module singleton can create rows."""
    result = await catalog_tag.create_row(session, name="singleton_catalog", validate=False)
    await session.commit()

    assert result.name == "singleton_catalog"


@pytest.mark.asyncio
async def test_module_singleton_filter_rows(session, multiple_catalog_tags):
    """Test that module singleton can filter rows."""
    from macon.models.filtering import Filter, FilterOp

    filters = [Filter(field="name", op=FilterOp.IN, value=["roman", "rubin"])]
    results = await catalog_tag.filter_rows(session, filters=filters)

    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"roman", "rubin"}


@pytest.mark.asyncio
async def test_module_singleton_to_pydantic(session, sample_catalog_tag):
    """Test that module singleton can convert to Pydantic."""
    pydantic_obj = catalog_tag.to_pydantic(sample_catalog_tag)

    assert isinstance(pydantic_obj, CatalogTagModel)
    assert pydantic_obj.name == sample_catalog_tag.name


@pytest.mark.asyncio
async def test_module_singleton_count_rows(session, multiple_catalog_tags):
    """Test that module singleton can count rows."""
    count = await catalog_tag.count_rows(session)

    assert count == 3
