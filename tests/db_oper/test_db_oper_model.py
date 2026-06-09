"""Unit tests for Model table operations."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from rail_svc.db import Model
from rail_svc.db_oper.base import TableContext
from rail_svc.db_oper.model import ModelOperations, model
from rail_svc.models import Model as ModelModel

# ============================================================================
# ModelOperations class tests
# ============================================================================


def test_model_operations_can_be_instantiated():
    """Test that ModelOperations can be instantiated."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    assert isinstance(ops, ModelOperations)


# ============================================================================
# Module-level singleton tests
# ============================================================================


def test_module_singleton_exists():
    """Test that module-level model singleton exists."""
    assert model is not None


def test_module_singleton_is_model_operations():
    """Test that module singleton is a ModelOperations instance."""
    assert isinstance(model, ModelOperations)


def test_module_singleton_is_singleton():
    """Test that module exports the same instance."""
    from rail_svc.db_oper.model import model as model2

    # Should be the exact same object
    assert model is model2


# ============================================================================
# Integration with base class tests
# ============================================================================


@pytest.mark.asyncio
async def test_model_operations_inherits_crud_methods(session, sample_model):
    """Test that ModelOperations inherits CRUD methods from base."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    # Should have get_row method from base class
    result = await ops.get_row(session, sample_model.id_)
    assert result.id_ == sample_model.id_


@pytest.mark.asyncio
async def test_model_operations_inherits_filter_methods(session, multiple_models):
    """Test that ModelOperations inherits filter methods from base."""
    from rail_svc.models.filtering import Filter, FilterOp

    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    # Should have filter_rows method from base class
    filters = [Filter(field="name", op=FilterOp.EQ, value="model_v1")]
    results = await ops.filter_rows(session, filters=filters)

    assert len(results) == 1
    assert results[0].name == "model_v1"


@pytest.mark.asyncio
async def test_model_operations_inherits_pydantic_conversion(session, sample_model):
    """Test that ModelOperations inherits Pydantic conversion methods."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    # Should have to_pydantic method from base class
    pydantic_obj = ops.to_pydantic(sample_model)

    assert isinstance(pydantic_obj, ModelModel)
    assert pydantic_obj.name == sample_model.name


# ============================================================================
# Module singleton functionality tests
# ============================================================================


@pytest.mark.asyncio
async def test_module_singleton_get_row(session, sample_model):
    """Test that module singleton can perform get_row operation."""
    result = await model.get_row(session, sample_model.id_)

    assert result.id_ == sample_model.id_
    assert result.name == sample_model.name


@pytest.mark.asyncio
async def test_module_singleton_filter_rows(session, multiple_models):
    """Test that module singleton can filter rows."""
    from rail_svc.models.filtering import Filter, FilterOp

    filters = [Filter(field="name", op=FilterOp.IN, value=["model_v1", "model_v2"])]
    results = await model.filter_rows(session, filters=filters)

    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"model_v1", "model_v2"}


@pytest.mark.asyncio
async def test_module_singleton_to_pydantic(session, sample_model):
    """Test that module singleton can convert to Pydantic."""
    pydantic_obj = model.to_pydantic(sample_model)

    assert isinstance(pydantic_obj, ModelModel)
    assert pydantic_obj.name == sample_model.name


@pytest.mark.asyncio
async def test_module_singleton_count_rows(session, multiple_models):
    """Test that module singleton can count rows."""
    count = await model.count_rows(session)

    assert count == 3


# ============================================================================
# get_create_kwargs tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_create_kwargs_with_ids_no_validation(session, sample_algorithm, sample_catalog_tag):
    """Test get_create_kwargs with IDs and no file validation."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        name="test_model",
        path="models/test.pkl",
        algo_id=sample_algorithm.id_,
        catalog_tag_id=sample_catalog_tag.id_,
        validate_file=False,
    )

    assert kwargs["name"] == "test_model"
    assert kwargs["path"] == "models/test.pkl"
    assert kwargs["algo_id"] == sample_algorithm.id_
    assert kwargs["catalog_tag_id"] == sample_catalog_tag.id_


@pytest.mark.asyncio
async def test_get_create_kwargs_with_names_no_validation(session, sample_algorithm, sample_catalog_tag):
    """Test get_create_kwargs with names and no file validation."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        name="test_model_2",
        path="models/test2.pkl",
        algo_name=sample_algorithm.name,
        catalog_tag_name=sample_catalog_tag.name,
        validate_file=False,
    )

    assert kwargs["algo_id"] == sample_algorithm.id_
    assert kwargs["catalog_tag_id"] == sample_catalog_tag.id_


@pytest.mark.asyncio
async def test_get_create_kwargs_mixed_ids_and_names(session, sample_algorithm, sample_catalog_tag):
    """Test get_create_kwargs with mixed IDs and names."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        name="test_model_3",
        path="models/test3.pkl",
        algo_id=sample_algorithm.id_,
        catalog_tag_name=sample_catalog_tag.name,
        validate_file=False,
    )

    assert kwargs["algo_id"] == sample_algorithm.id_
    assert kwargs["catalog_tag_id"] == sample_catalog_tag.id_


@pytest.mark.asyncio
async def test_get_create_kwargs_nonexistent_algo_name(session, sample_catalog_tag):
    """Test get_create_kwargs raises for nonexistent algorithm name."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    with pytest.raises((ValueError, KeyError)):
        await ops.get_create_kwargs(
            session,
            name="test",
            path="models/test.pkl",
            algo_name="nonexistent",
            catalog_tag_id=sample_catalog_tag.id_,
            validate_file=False,
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_nonexistent_catalog_tag_name(session, sample_algorithm):
    """Test get_create_kwargs raises for nonexistent catalog tag name."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    with pytest.raises((ValueError, KeyError)):
        await ops.get_create_kwargs(
            session,
            name="test",
            path="models/test.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_name="nonexistent",
            validate_file=False,
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_with_extra_fields(session, sample_algorithm, sample_catalog_tag):
    """Test get_create_kwargs passes through extra kwargs."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        name="test",
        path="models/test.pkl",
        algo_id=sample_algorithm.id_,
        catalog_tag_id=sample_catalog_tag.id_,
        validate_file=False,
        extra_field="extra_value",
    )

    assert kwargs["extra_field"] == "extra_value"


# ============================================================================
# _convert_informer_to_estimator tests
# ============================================================================


def test_convert_informer_to_estimator_pattern():
    """Test converting Informer to Estimator using pattern."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    result = ops._convert_informer_to_estimator("RandomForestInformer")
    assert result == "RandomForestEstimator"

    result = ops._convert_informer_to_estimator("CustomInformer")
    assert result == "CustomEstimator"


def test_convert_informer_to_estimator_mapping():
    """Test converting using explicit mapping."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    result = ops._convert_informer_to_estimator("dummy")
    assert result == "dummy"


def test_convert_informer_to_estimator_invalid():
    """Test converting invalid class name raises ValueError."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    with pytest.raises(ValueError, match="Cannot convert Informer class name"):
        ops._convert_informer_to_estimator("InvalidClassName")


# ============================================================================
# validate_model tests
# ============================================================================


@pytest.mark.asyncio
async def test_validate_model_file_not_found(session, sample_algorithm, sample_catalog_tag):
    """Test validate_model raises FileNotFoundError for missing file."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    nonexistent_path = Path("/tmp/nonexistent_model_12345.pkl")

    with pytest.raises(FileNotFoundError, match="not found"):
        await ops.validate_model(nonexistent_path, sample_algorithm, sample_catalog_tag)


@pytest.mark.asyncio
async def test_validate_model_read_error(session, sample_algorithm, sample_catalog_tag, tmp_path):
    """Test validate_model handles read errors."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    test_file = tmp_path / "test.pkl"
    test_file.write_text("invalid model data")

    with patch("rail_svc.db_oper.model.RailModel.read", side_effect=Exception("Read error")):
        with pytest.raises(ValueError, match="Could not read model"):
            await ops.validate_model(test_file, sample_algorithm, sample_catalog_tag)


@pytest.mark.asyncio
async def test_validate_model_catalog_tag_mismatch(session, sample_algorithm, sample_catalog_tag, tmp_path):
    """Test validate_model catches catalog tag mismatch."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    test_file = tmp_path / "test.pkl"
    test_file.write_text("test")

    mock_model = Mock()
    mock_model.catalog_tag = "different_tag"
    mock_model.creation_class_name = None

    with patch("rail_svc.db_oper.model.RailModel.read", return_value=mock_model):
        with pytest.raises(ValueError, match="CatalogTag mismatch"):
            await ops.validate_model(test_file, sample_algorithm, sample_catalog_tag)


@pytest.mark.asyncio
async def test_validate_model_algorithm_mismatch(session, sample_algorithm, sample_catalog_tag, tmp_path):
    """Test validate_model catches algorithm mismatch."""
    context = TableContext.from_db_class(Model)
    ops = ModelOperations(context)

    test_file = tmp_path / "test.pkl"
    test_file.write_text("test")

    mock_model = Mock()
    mock_model.catalog_tag = sample_catalog_tag.name
    mock_model.creation_class_name = "DifferentInformer"

    with patch("rail_svc.db_oper.model.RailModel.read", return_value=mock_model):
        with pytest.raises(ValueError, match="Algorithm mismatch"):
            await ops.validate_model(test_file, sample_algorithm, sample_catalog_tag)


# ============================================================================
# create_row integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_row_without_file_validation(session, sample_algorithm, sample_catalog_tag):
    """Test creating model without file validation."""
    result = await model.create_row(
        session,
        name="test_model_create",
        path="models/test.pkl",
        algo_id=sample_algorithm.id_,
        catalog_tag_id=sample_catalog_tag.id_,
        validate_file=False,
        validate=False,
    )
    await session.commit()

    assert result.name == "test_model_create"
    assert result.path == "models/test.pkl"
    assert result.algo_id == sample_algorithm.id_
    assert result.catalog_tag_id == sample_catalog_tag.id_


@pytest.mark.asyncio
async def test_create_row_with_names(session, sample_algorithm, sample_catalog_tag):
    """Test creating model using algorithm and catalog tag names."""
    result = await model.create_row(
        session,
        name="test_model_by_names",
        path="models/test2.pkl",
        algo_name=sample_algorithm.name,
        catalog_tag_name=sample_catalog_tag.name,
        validate_file=False,
        validate=False,
    )
    await session.commit()

    assert result.algo_id == sample_algorithm.id_
    assert result.catalog_tag_id == sample_catalog_tag.id_
