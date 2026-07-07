"""Unit tests for Algorithm table operations."""

import pytest
from macon.db_oper.base import TableContext

from rail_svc.db import Algorithm
from rail_svc.db_oper.algorithm import AlgorithmOperations, algorithm
from rail_svc.models import Algorithm as AlgorithmModel

# ============================================================================
# AlgorithmOperations class tests
# ============================================================================


async def test_algorithm_operations_inherits_crud_methods(session, sample_algorithm):
    """Test that AlgorithmOperations inherits CRUD methods from base."""
    context = TableContext.from_db_class(Algorithm)
    ops = AlgorithmOperations(context)

    # Should have get_row method from base class
    result = await ops.get_row(session, sample_algorithm.id_)
    assert result.id_ == sample_algorithm.id_


@pytest.mark.asyncio
async def test_algorithm_operations_inherits_filter_methods(session, multiple_algorithms):
    """Test that AlgorithmOperations inherits filter methods from base."""
    from macon.models.filtering import Filter, FilterOp

    context = TableContext.from_db_class(Algorithm)
    ops = AlgorithmOperations(context)

    # Should have filter_rows method from base class
    filters = [Filter(field="name", op=FilterOp.EQ, value="knn")]
    results = await ops.filter_rows(session, filters=filters)

    assert len(results) == 1
    assert results[0].name == "knn"


@pytest.mark.asyncio
async def test_algorithm_operations_inherits_pydantic_conversion(session, sample_algorithm):
    """Test that AlgorithmOperations inherits Pydantic conversion methods."""
    context = TableContext.from_db_class(Algorithm)
    ops = AlgorithmOperations(context)

    # Should have to_pydantic method from base class
    pydantic_obj = ops.to_pydantic(sample_algorithm)

    assert isinstance(pydantic_obj, AlgorithmModel)
    assert pydantic_obj.name == sample_algorithm.name


# ============================================================================
# Module singleton functionality tests
# ============================================================================


@pytest.mark.asyncio
async def test_module_singleton_get_row(session, sample_algorithm):
    """Test that module singleton can perform get_row operation."""
    result = await algorithm.get_row(session, sample_algorithm.id_)

    assert result.id_ == sample_algorithm.id_
    assert result.name == sample_algorithm.name


@pytest.mark.asyncio
async def test_module_singleton_create_row(session):
    """Test that module singleton can create rows."""
    result = await algorithm.create_row(
        session, name="singleton_test", class_name="singleton.class", validate=False
    )
    await session.commit()

    assert result.name == "singleton_test"
    assert result.class_name == "singleton.class"


@pytest.mark.asyncio
async def test_module_singleton_filter_rows(session, multiple_algorithms):
    """Test that module singleton can filter rows."""
    from macon.models.filtering import Filter, FilterOp

    filters = [Filter(field="name", op=FilterOp.IN, value=["knn", "xgboost"])]
    results = await algorithm.filter_rows(session, filters=filters)

    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"knn", "xgboost"}


@pytest.mark.asyncio
async def test_module_singleton_to_pydantic(session, sample_algorithm):
    """Test that module singleton can convert to Pydantic."""
    pydantic_obj = algorithm.to_pydantic(sample_algorithm)

    assert isinstance(pydantic_obj, AlgorithmModel)
    assert pydantic_obj.name == sample_algorithm.name
    assert pydantic_obj.class_name == sample_algorithm.class_name


@pytest.mark.asyncio
async def test_module_singleton_count_rows(session, multiple_algorithms):
    """Test that module singleton can count rows."""
    count = await algorithm.count_rows(session)

    assert count == 3
