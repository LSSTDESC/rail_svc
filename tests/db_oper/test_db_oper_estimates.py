"""Unit tests for Estimates table operations."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rail_svc.db import Estimates
from macon.db_oper.base import FileValidatedOperations, TableContext
from rail_svc.db_oper.estimates import EstimatesOperations, estimates
from rail_svc.models import Estimates as EstimatesModel

# ============================================================================
# EstimatesOperations class tests
# ============================================================================


def test_estimates_operations_can_be_instantiated():
    """Test that EstimatesOperations can be instantiated."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    assert isinstance(ops, EstimatesOperations)


def test_estimates_operations_inherits_file_validated_operations():
    """Test that EstimatesOperations inherits from FileValidatedOperations."""
    assert issubclass(EstimatesOperations, FileValidatedOperations)


# ============================================================================
# Module-level singleton tests
# ============================================================================


def test_module_singleton_exists():
    """Test that module-level estimates singleton exists."""
    assert estimates is not None


def test_module_singleton_is_estimates_operations():
    """Test that module singleton is an EstimatesOperations instance."""
    assert isinstance(estimates, EstimatesOperations)


def test_module_singleton_is_singleton():
    """Test that module exports the same instance."""
    from rail_svc.db_oper.estimates import estimates as estimates2

    # Should be the exact same object
    assert estimates is estimates2


# ============================================================================
# Integration with base class tests
# ============================================================================


@pytest.mark.asyncio
async def test_estimates_operations_inherits_crud_methods(session, sample_estimates):
    """Test that EstimatesOperations inherits CRUD methods from base."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    # Should have get_row method from base class
    result = await ops.get_row(session, sample_estimates.id_)
    assert result.id_ == sample_estimates.id_


@pytest.mark.asyncio
async def test_estimates_operations_inherits_filter_methods(session, multiple_estimates):
    """Test that EstimatesOperations inherits filter methods from base."""
    from macon.models.filtering import Filter, FilterOp

    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    # Should have filter_rows method from base class
    filters = [Filter(field="name", op=FilterOp.EQ, value="estimates_v1")]
    results = await ops.filter_rows(session, filters=filters)

    assert len(results) == 1
    assert results[0].name == "estimates_v1"


@pytest.mark.asyncio
async def test_estimates_operations_inherits_pydantic_conversion(session, sample_estimates):
    """Test that EstimatesOperations inherits Pydantic conversion methods."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    # Should have to_pydantic method from base class
    pydantic_obj = ops.to_pydantic(sample_estimates)

    assert isinstance(pydantic_obj, EstimatesModel)
    assert pydantic_obj.name == sample_estimates.name


# ============================================================================
# Module singleton functionality tests
# ============================================================================


@pytest.mark.asyncio
async def test_module_singleton_get_row(session, sample_estimates):
    """Test that module singleton can perform get_row operation."""
    result = await estimates.get_row(session, sample_estimates.id_)

    assert result.id_ == sample_estimates.id_
    assert result.name == sample_estimates.name


@pytest.mark.asyncio
async def test_module_singleton_filter_rows(session, multiple_estimates):
    """Test that module singleton can filter rows."""
    from macon.models.filtering import Filter, FilterOp

    filters = [Filter(field="n_objects", op=FilterOp.EQ, value=5000)]
    results = await estimates.filter_rows(session, filters=filters)

    assert len(results) >= 1
    assert all(r.n_objects == 5000 for r in results)


@pytest.mark.asyncio
async def test_module_singleton_to_pydantic(session, sample_estimates):
    """Test that module singleton can convert to Pydantic."""
    pydantic_obj = estimates.to_pydantic(sample_estimates)

    assert isinstance(pydantic_obj, EstimatesModel)
    assert pydantic_obj.name == sample_estimates.name
    assert pydantic_obj.n_objects == sample_estimates.n_objects


@pytest.mark.asyncio
async def test_module_singleton_count_rows(session, multiple_estimates):
    """Test that module singleton can count rows."""
    count = await estimates.count_rows(session)

    assert count == 3


# ============================================================================
# get_create_kwargs tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_create_kwargs_with_ids_no_validation(session, sample_dataset, sample_estimator):
    """Test get_create_kwargs with IDs and no file validation."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        path="estimates/test.h5",
        dataset_id=sample_dataset.id_,
        estimator_id=sample_estimator.id_,
        n_objects=1000,
        validate_file=False,
        name="test_estimates",
    )

    assert kwargs["path"] == "estimates/test.h5"
    assert kwargs["dataset_id"] == sample_dataset.id_
    assert kwargs["estimator_id"] == sample_estimator.id_
    assert kwargs["n_objects"] == 1000
    assert kwargs["name"] == "test_estimates"


@pytest.mark.asyncio
async def test_get_create_kwargs_with_names_no_validation(session, sample_dataset, sample_estimator):
    """Test get_create_kwargs with names and no file validation."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        path="estimates/test2.h5",
        dataset_name=sample_dataset.name,
        estimator_name=sample_estimator.name,
        n_objects=2000,
        validate_file=False,
    )

    assert kwargs["dataset_id"] == sample_dataset.id_
    assert kwargs["estimator_id"] == sample_estimator.id_
    assert kwargs["n_objects"] == 2000


@pytest.mark.asyncio
async def test_get_create_kwargs_mixed_ids_and_names(session, sample_dataset, sample_estimator):
    """Test get_create_kwargs with mixed IDs and names."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        path="estimates/test3.h5",
        dataset_id=sample_dataset.id_,
        estimator_name=sample_estimator.name,
        n_objects=3000,
        validate_file=False,
    )

    assert kwargs["dataset_id"] == sample_dataset.id_
    assert kwargs["estimator_id"] == sample_estimator.id_


@pytest.mark.asyncio
async def test_get_create_kwargs_no_path_with_n_objects(session, sample_dataset, sample_estimator):
    """Test get_create_kwargs without path but with n_objects."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        path=None,
        dataset_id=sample_dataset.id_,
        estimator_id=sample_estimator.id_,
        n_objects=500,
        validate_file=False,
    )

    assert kwargs["path"] is None
    assert kwargs["n_objects"] == 500


@pytest.mark.asyncio
async def test_get_create_kwargs_no_path_no_n_objects_raises(session, sample_dataset, sample_estimator):
    """Test get_create_kwargs raises when neither path nor n_objects provided."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    with pytest.raises(ValueError, match="Either 'path' or 'n_objects' must be provided"):
        await ops.get_create_kwargs(
            session,
            path=None,
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
            validate_file=True,
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_validation_disabled_no_n_objects_raises(
    session, sample_dataset, sample_estimator
):
    """Test get_create_kwargs raises when validation disabled but no n_objects."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    with pytest.raises(ValueError, match="n_objects' must be provided"):
        await ops.get_create_kwargs(
            session,
            path="estimates/test.h5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
            validate_file=False,
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_nonexistent_dataset_name(session, sample_estimator):
    """Test get_create_kwargs raises for nonexistent dataset name."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    with pytest.raises((ValueError, KeyError)):
        await ops.get_create_kwargs(
            session,
            path="estimates/test.h5",
            dataset_name="nonexistent",
            estimator_id=sample_estimator.id_,
            n_objects=100,
            validate_file=False,
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_nonexistent_estimator_name(session, sample_dataset):
    """Test get_create_kwargs raises for nonexistent estimator name."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    with pytest.raises((ValueError, KeyError)):
        await ops.get_create_kwargs(
            session,
            path="estimates/test.h5",
            dataset_id=sample_dataset.id_,
            estimator_name="nonexistent",
            n_objects=100,
            validate_file=False,
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_filters_n_objects_from_extra_kwargs(
    session, sample_dataset, sample_estimator
):
    """Test that n_objects in extra_kwargs is filtered out."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        path="estimates/test.h5",
        dataset_id=sample_dataset.id_,
        estimator_id=sample_estimator.id_,
        n_objects=1000,
        validate_file=False,
        extra_field="extra_value",
    )

    # Should use n_objects from parameters
    assert kwargs["n_objects"] == 1000
    assert kwargs["extra_field"] == "extra_value"


# ============================================================================
# get_file_length tests
# ============================================================================


def test_get_file_length():
    """Test get_file_length calls qp.data_length correctly."""
    context = TableContext.from_db_class(Estimates)
    ops = EstimatesOperations(context)

    test_path = Path("/estimates/test.h5")

    with patch("rail_svc.db_oper.estimates.qp.data_length") as mock_data_length:
        mock_data_length.return_value = 5000

        result = ops.get_file_length(test_path)

        assert result == 5000
        mock_data_length.assert_called_once_with(str(test_path))


# ============================================================================
# create_row integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_row_without_file_validation(session, sample_dataset, sample_estimator):
    """Test creating estimates without file validation."""
    result = await estimates.create_row(
        session,
        name="test_estimates",
        path="estimates/test.h5",
        dataset_id=sample_dataset.id_,
        estimator_id=sample_estimator.id_,
        n_objects=1500,
        validate_file=False,
        validate=False,
    )
    await session.commit()

    assert result.name == "test_estimates"
    assert result.path == "estimates/test.h5"
    assert result.n_objects == 1500
    assert result.dataset_id == sample_dataset.id_
    assert result.estimator_id == sample_estimator.id_


@pytest.mark.asyncio
async def test_create_row_with_names(session, sample_dataset, sample_estimator):
    """Test creating estimates using dataset and estimator names."""
    result = await estimates.create_row(
        session,
        name="test_estimates_2",
        path="estimates/test2.h5",
        dataset_name=sample_dataset.name,
        estimator_name=sample_estimator.name,
        n_objects=2500,
        validate_file=False,
        validate=False,
    )
    await session.commit()

    assert result.dataset_id == sample_dataset.id_
    assert result.estimator_id == sample_estimator.id_
