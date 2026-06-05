"""Unit tests for Estimator table operations."""

import pytest

from rail_svc.db import Estimator
from rail_svc.db_oper.base import TableContext
from rail_svc.db_oper.estimator import EstimatorOperations, estimator
from rail_svc.models import Estimator as EstimatorModel


# ============================================================================
# EstimatorOperations class tests
# ============================================================================


def test_estimator_operations_can_be_instantiated():
    """Test that EstimatorOperations can be instantiated."""
    context = TableContext.from_db_class(Estimator)
    ops = EstimatorOperations(context)

    assert isinstance(ops, EstimatorOperations)


# ============================================================================
# Module-level singleton tests
# ============================================================================


def test_module_singleton_exists():
    """Test that module-level estimator singleton exists."""
    assert estimator is not None


def test_module_singleton_is_estimator_operations():
    """Test that module singleton is an EstimatorOperations instance."""
    assert isinstance(estimator, EstimatorOperations)


def test_module_singleton_is_singleton():
    """Test that module exports the same instance."""
    from rail_svc.db_oper.estimator import estimator as estimator2

    # Should be the exact same object
    assert estimator is estimator2


# ============================================================================
# Integration with base class tests
# ============================================================================


@pytest.mark.asyncio
async def test_estimator_operations_inherits_crud_methods(session, sample_estimator):
    """Test that EstimatorOperations inherits CRUD methods from base."""
    context = TableContext.from_db_class(Estimator)
    ops = EstimatorOperations(context)

    # Should have get_row method from base class
    result = await ops.get_row(session, sample_estimator.id_)
    assert result.id_ == sample_estimator.id_


@pytest.mark.asyncio
async def test_estimator_operations_inherits_filter_methods(session, multiple_estimators):
    """Test that EstimatorOperations inherits filter methods from base."""
    from rail_svc.models.filtering import Filter, FilterOp

    context = TableContext.from_db_class(Estimator)
    ops = EstimatorOperations(context)

    # Should have filter_rows method from base class
    filters = [Filter(field="name", op=FilterOp.EQ, value="estimator_v1")]
    results = await ops.filter_rows(session, filters=filters)

    assert len(results) == 1
    assert results[0].name == "estimator_v1"


@pytest.mark.asyncio
async def test_estimator_operations_inherits_pydantic_conversion(session, sample_estimator):
    """Test that EstimatorOperations inherits Pydantic conversion methods."""
    context = TableContext.from_db_class(Estimator)
    ops = EstimatorOperations(context)

    # Should have to_pydantic method from base class
    pydantic_obj = ops.to_pydantic(sample_estimator)

    assert isinstance(pydantic_obj, EstimatorModel)
    assert pydantic_obj.name == sample_estimator.name


# ============================================================================
# Module singleton functionality tests
# ============================================================================


@pytest.mark.asyncio
async def test_module_singleton_get_row(session, sample_estimator):
    """Test that module singleton can perform get_row operation."""
    result = await estimator.get_row(session, sample_estimator.id_)

    assert result.id_ == sample_estimator.id_
    assert result.name == sample_estimator.name


@pytest.mark.asyncio
async def test_module_singleton_filter_rows(session, multiple_estimators):
    """Test that module singleton can filter rows."""
    from rail_svc.models.filtering import Filter, FilterOp

    filters = [Filter(field="name", op=FilterOp.IN, value=["estimator_v1", "estimator_v2"])]
    results = await estimator.filter_rows(session, filters=filters)

    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"estimator_v1", "estimator_v2"}


@pytest.mark.asyncio
async def test_module_singleton_to_pydantic(session, sample_estimator):
    """Test that module singleton can convert to Pydantic."""
    pydantic_obj = estimator.to_pydantic(sample_estimator)

    assert isinstance(pydantic_obj, EstimatorModel)
    assert pydantic_obj.name == sample_estimator.name


@pytest.mark.asyncio
async def test_module_singleton_count_rows(session, multiple_estimators):
    """Test that module singleton can count rows."""
    count = await estimator.count_rows(session)

    assert count == 3


# ============================================================================
# get_create_kwargs tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_create_kwargs_with_model_id(session, sample_model):
    """Test get_create_kwargs with model_id provided."""
    context = TableContext.from_db_class(Estimator)
    ops = EstimatorOperations(context)

    kwargs = await ops.get_create_kwargs(
        session, model_id=sample_model.id_, name="test_estimator", config={"param": "value"}
    )

    assert kwargs["model_id"] == sample_model.id_
    assert kwargs["name"] == "test_estimator"
    assert kwargs["config"] == {"param": "value"}


@pytest.mark.asyncio
async def test_get_create_kwargs_with_model_name(session, sample_model):
    """Test get_create_kwargs with model_name provided."""
    context = TableContext.from_db_class(Estimator)
    ops = EstimatorOperations(context)

    kwargs = await ops.get_create_kwargs(
        session, model_name=sample_model.name, name="test_estimator_2", config={"learning_rate": 0.01}
    )

    assert kwargs["model_id"] == sample_model.id_
    assert kwargs["name"] == "test_estimator_2"
    assert kwargs["config"] == {"learning_rate": 0.01}


@pytest.mark.asyncio
async def test_get_create_kwargs_nonexistent_model_name(session):
    """Test get_create_kwargs raises for nonexistent model name."""
    context = TableContext.from_db_class(Estimator)
    ops = EstimatorOperations(context)

    with pytest.raises((ValueError, KeyError)):
        await ops.get_create_kwargs(session, model_name="nonexistent", name="test")


@pytest.mark.asyncio
async def test_get_create_kwargs_with_extra_fields(session, sample_model):
    """Test get_create_kwargs passes through extra kwargs."""
    context = TableContext.from_db_class(Estimator)
    ops = EstimatorOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        model_id=sample_model.id_,
        name="test",
        description="Test estimator",
        version="1.0.0",
        extra_field="extra",
    )

    assert kwargs["model_id"] == sample_model.id_
    assert kwargs["name"] == "test"
    assert kwargs["description"] == "Test estimator"
    assert kwargs["version"] == "1.0.0"
    assert kwargs["extra_field"] == "extra"


# ============================================================================
# create_row integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_row_with_model_id(session, sample_model):
    """Test creating estimator using model_id."""
    result = await estimator.create_row(
        session,
        name="test_estimator_create",
        model_id=sample_model.id_,
        config={"n_estimators": 100},
        validate=False,
    )
    await session.commit()

    assert result.name == "test_estimator_create"
    assert result.model_id == sample_model.id_
    assert result.config == {"n_estimators": 100}


@pytest.mark.asyncio
async def test_create_row_with_model_name(session, sample_model):
    """Test creating estimator using model_name."""
    result = await estimator.create_row(
        session,
        name="test_estimator_by_name",
        model_name=sample_model.name,
        config={"max_depth": 10},
        validate=False,
    )
    await session.commit()

    assert result.model_id == sample_model.id_
    assert result.config == {"max_depth": 10}


@pytest.mark.asyncio
async def test_create_row_with_all_fields(session, sample_model):
    """Test creating estimator with all common fields."""
    result = await estimator.create_row(
        session,
        name="full_estimator",
        model_id=sample_model.id_,
        config={"n_estimators": 100, "max_depth": 10, "learning_rate": 0.1},
        validate=False,
    )
    await session.commit()

    assert result.name == "full_estimator"
    assert result.model_id == sample_model.id_
    assert result.config["n_estimators"] == 100
    assert result.config["max_depth"] == 10
    assert result.config["learning_rate"] == 0.1
