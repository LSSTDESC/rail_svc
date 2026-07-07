"""Unit tests for Dataset table operations."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rail_svc.db import Dataset
from macon.db_oper.base import FileValidatedOperations, TableContext
from rail_svc.db_oper.dataset import DatasetOperations, dataset
from rail_svc.models import Dataset as DatasetModel

# ============================================================================
# DatasetOperations class tests
# ============================================================================


def test_dataset_operations_can_be_instantiated():
    """Test that DatasetOperations can be instantiated."""
    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    assert isinstance(ops, DatasetOperations)


def test_dataset_operations_inherits_file_validated_operations():
    """Test that DatasetOperations inherits from FileValidatedOperations."""
    assert issubclass(DatasetOperations, FileValidatedOperations)


# ============================================================================
# Module-level singleton tests
# ============================================================================


def test_module_singleton_exists():
    """Test that module-level dataset singleton exists."""
    assert dataset is not None


def test_module_singleton_is_dataset_operations():
    """Test that module singleton is a DatasetOperations instance."""
    assert isinstance(dataset, DatasetOperations)


def test_module_singleton_is_singleton():
    """Test that module exports the same instance."""
    from rail_svc.db_oper.dataset import dataset as dataset2

    # Should be the exact same object
    assert dataset is dataset2


# ============================================================================
# Integration with base class tests
# ============================================================================


@pytest.mark.asyncio
async def test_dataset_operations_inherits_crud_methods(session, sample_dataset):
    """Test that DatasetOperations inherits CRUD methods from base."""
    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    # Should have get_row method from base class
    result = await ops.get_row(session, sample_dataset.id_)
    assert result.id_ == sample_dataset.id_


@pytest.mark.asyncio
async def test_dataset_operations_inherits_filter_methods(session, multiple_datasets):
    """Test that DatasetOperations inherits filter methods from base."""
    from macon.models.filtering import Filter, FilterOp

    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    # Should have filter_rows method from base class
    filters = [Filter(field="name", op=FilterOp.EQ, value="photometric_data")]
    results = await ops.filter_rows(session, filters=filters)

    assert len(results) == 1
    assert results[0].name == "photometric_data"


@pytest.mark.asyncio
async def test_dataset_operations_inherits_pydantic_conversion(session, sample_dataset):
    """Test that DatasetOperations inherits Pydantic conversion methods."""
    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    # Should have to_pydantic method from base class
    pydantic_obj = ops.to_pydantic(sample_dataset)

    assert isinstance(pydantic_obj, DatasetModel)
    assert pydantic_obj.name == sample_dataset.name


# ============================================================================
# Module singleton functionality tests
# ============================================================================


@pytest.mark.asyncio
async def test_module_singleton_get_row(session, sample_dataset):
    """Test that module singleton can perform get_row operation."""
    result = await dataset.get_row(session, sample_dataset.id_)

    assert result.id_ == sample_dataset.id_
    assert result.name == sample_dataset.name


@pytest.mark.asyncio
async def test_module_singleton_filter_rows(session, multiple_datasets):
    """Test that module singleton can filter rows."""
    from macon.models.filtering import Filter, FilterOp

    filters = [Filter(field="is_collection", op=FilterOp.EQ, value=False)]
    results = await dataset.filter_rows(session, filters=filters)

    assert len(results) >= 2
    assert all(r.is_collection is False for r in results)


@pytest.mark.asyncio
async def test_module_singleton_to_pydantic(session, sample_dataset):
    """Test that module singleton can convert to Pydantic."""
    pydantic_obj = dataset.to_pydantic(sample_dataset)

    assert isinstance(pydantic_obj, DatasetModel)
    assert pydantic_obj.name == sample_dataset.name
    assert pydantic_obj.n_objects == sample_dataset.n_objects


@pytest.mark.asyncio
async def test_module_singleton_count_rows(session, multiple_datasets):
    """Test that module singleton can count rows."""
    count = await dataset.count_rows(session)

    assert count == 3


# ============================================================================
# get_create_kwargs tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_create_kwargs_with_catalog_tag_id_no_validation(session, sample_catalog_tag):
    """Test get_create_kwargs with catalog_tag_id and no file validation."""
    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        path="data/test.h5",
        catalog_tag_id=sample_catalog_tag.id_,
        n_objects=1000,
        validate_file=False,
        name="test_dataset",
    )

    assert kwargs["path"] == "data/test.h5"
    assert kwargs["catalog_tag_id"] == sample_catalog_tag.id_
    assert kwargs["n_objects"] == 1000
    assert kwargs["name"] == "test_dataset"


@pytest.mark.asyncio
async def test_get_create_kwargs_with_catalog_tag_name_no_validation(session, sample_catalog_tag):
    """Test get_create_kwargs with catalog_tag_name and no file validation."""
    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        path="data/test.h5",
        catalog_tag_name=sample_catalog_tag.name,
        n_objects=2000,
        validate_file=False,
        name="test_dataset_2",
    )

    assert kwargs["catalog_tag_id"] == sample_catalog_tag.id_
    assert kwargs["n_objects"] == 2000


@pytest.mark.asyncio
async def test_get_create_kwargs_no_path_with_n_objects(session, sample_catalog_tag):
    """Test get_create_kwargs without path but with n_objects."""
    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    kwargs = await ops.get_create_kwargs(
        session, path=None, catalog_tag_id=sample_catalog_tag.id_, n_objects=500, validate_file=False
    )

    assert kwargs["path"] is None
    assert kwargs["n_objects"] == 500


@pytest.mark.asyncio
async def test_get_create_kwargs_no_path_no_n_objects_raises(session, sample_catalog_tag):
    """Test get_create_kwargs raises when neither path nor n_objects provided."""
    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    with pytest.raises(ValueError, match="Either 'path' or 'n_objects' must be provided"):
        await ops.get_create_kwargs(
            session, path=None, catalog_tag_id=sample_catalog_tag.id_, validate_file=True
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_validation_disabled_no_n_objects_raises(session, sample_catalog_tag):
    """Test get_create_kwargs raises when validation disabled but no n_objects."""
    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    with pytest.raises(ValueError, match="n_objects' must be provided"):
        await ops.get_create_kwargs(
            session, path="data/test.h5", catalog_tag_id=sample_catalog_tag.id_, validate_file=False
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_nonexistent_catalog_tag_name(session):
    """Test get_create_kwargs raises for nonexistent catalog tag name."""
    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    with pytest.raises(ValueError, match="CatalogTag with name 'nonexistent' not found"):
        await ops.get_create_kwargs(
            session, path="data/test.h5", catalog_tag_name="nonexistent", n_objects=100, validate_file=False
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_filters_n_objects_from_extra_kwargs(session, sample_catalog_tag):
    """Test that n_objects in extra_kwargs is filtered out."""
    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        path="data/test.h5",
        catalog_tag_id=sample_catalog_tag.id_,
        n_objects=1000,
        validate_file=False,
        extra_field="extra_value",
    )

    # Should use n_objects from parameters, not extra_kwargs
    assert kwargs["n_objects"] == 1000
    assert kwargs["extra_field"] == "extra_value"


# ============================================================================
# get_file_length tests
# ============================================================================


def test_get_file_length():
    """Test get_file_length calls tables_io correctly."""
    context = TableContext.from_db_class(Dataset)
    ops = DatasetOperations(context)

    test_path = Path("/data/test.h5")

    with patch("rail_svc.db_oper.dataset.tables_io.hdf5.get_input_data_length") as mock_get_length:
        mock_get_length.return_value = 5000

        result = ops.get_file_length(test_path)

        assert result == 5000
        mock_get_length.assert_called_once_with(str(test_path))


# ============================================================================
# create_row integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_row_without_file_validation(session, sample_catalog_tag):
    """Test creating dataset without file validation."""
    result = await dataset.create_row(
        session,
        name="test_dataset",
        path="data/test.h5",
        catalog_tag_id=sample_catalog_tag.id_,
        n_objects=1500,
        is_collection=False,
        validate_file=False,
        validate=False,
    )
    await session.commit()

    assert result.name == "test_dataset"
    assert result.path == "data/test.h5"
    assert result.n_objects == 1500
    assert result.catalog_tag_id == sample_catalog_tag.id_


@pytest.mark.asyncio
async def test_create_row_with_catalog_tag_name(session, sample_catalog_tag):
    """Test creating dataset using catalog tag name."""
    result = await dataset.create_row(
        session,
        name="test_dataset_2",
        path="data/test2.h5",
        catalog_tag_name=sample_catalog_tag.name,
        n_objects=2500,
        is_collection=False,
        validate_file=False,
        validate=False,
    )
    await session.commit()

    assert result.catalog_tag_id == sample_catalog_tag.id_
